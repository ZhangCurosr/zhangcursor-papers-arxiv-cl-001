# ENVOPT: Automated Optimization for Agent Environments

Mingxuan Li<sup>1,2,3</sup>, Huanzhi Mao<sup>1</sup>, Qiuyang Mang<sup>1</sup>, Hanchen Li<sup>1</sup>, Joseph E. Gonzalez<sup>1</sup>, Alex Dimakis<sup>1,3</sup>

<sup>1</sup>University of California, Berkeley <sup>2</sup>University of Chicago <sup>3</sup>Bespoke Labs

Correspondence: mingxuanl@uchicago.edu

Code: https://github.com/itea1001/EnvOpt

## Abstract

An LLM agent’s cost and success depend on its model, its harness, and its environment. In addition to the widely studied directions of parameter tuning and harness optimization, we propose the third direction: optimization over stationary environments. Agents are often deployed in rich environments shared by many tasks, such as a company’s Slack workspace or a complex collection of data sources, documents or tickets. We call these environments stationary environments. We introduce ENVOPT, a method for automated end-to-end optimization of stationary environments for any given model and harness, using real or synthetic tasks without needing any graders or reference answers. On AutomationBench, ENVOPT reduces the cost of Claude Haiku 4.5 by 72.7% and raises its pass rate by 14.34%. The same environment transfers to Opus 5, cutting its cost by 68.1% with a 1.34% gain. Added on top of Meta-Harness’s optimized harness for Haiku 4.5, it lowers cost by a further 83.4% and adds 11.17% accuracy over Meta-Harness alone. Across 5 benchmarks, ENVOPT significantly reduces the cost of agents while improving or maintaining pass rate, and is transferable to different models and different harnesses. We also discuss common successful and rejected environment optimization strategies during automated optimization. More broadly, our results show that how information is represented to an agent shapes both its cost and its success, making the environment worth optimizing in addition to the parameters and the harness.

## 1 Introduction

An LLM agent’s cost and success rate depend on three things: the model parameters, the harness, and the structure of the environment it interacts with. The first two are actively optimized through parameter tuning and harness optimization (Lee et al., 2026; Agrawal et al., 2026b), while the environment is often considered immutable. Yet, the same information can be given to an agent in very different forms. For example, a company’s Slack history can either sit behind a remote server and be accessible via MCP or exported in plain text files that can be read and searched with terminal tools like grep. This choice alone can significantly affect how much work the agent needs to perform to solve a task (Mak et al., 2026). In this paper, we propose optimizing stationary environments as a third axis for improving LLM agents, alongside model parameter tuning and harness optimization. Stationary environments refer to environments with large collections of artifacts that remain the same or are similar across many tasks. For example, a banking agent uses the same enterprise platforms and databases for a massive number of queries; many tasks for a company’s Slack assistant depend on the same Slack workspace; feature requests for a toolkit can rely on the same code repository. We aim to optimize how the stationary environments are organized and represented to agents.

Though the environment has not received the same systematic optimization as model parameters and harnesses, previous work has shown that refining agent-environment interactions can improve agent performance. SWE-agent (Yang et al., 2024) and subsequent work (Xu et al., 2026) showed that the same model fixes more issues with fewer steps when given a better interface to the repository (Xu et al., 2026), but these interfaces were manually designed. Later work reshapes environments to create more training signal for

(a) Cost reduction on stationary environment AutomationBench, Claude Opus 4.8

![](images/6a7da67294002c12f566df979ba9fa78e77ca9a519f43b0278ff3695221ad154.jpg)  
(b) Offline environment optimization

![](images/84fbc846d34652ee7553e55e2976171d33c4edde19f6b8e9ffcf37b705013f78.jpg)  
Figure 1 ENVOPT on AutomationBench. Left: ENVOPT costs around USD \$150 for an offline optimization of the stationary environment in AutomationBench, and saves cost of running Claude Opus 4.8 by 77.5%, easily saving many times the cost of running ENVOPT. Lines are derived by random sampling actual cost of running the test set tasks. Right: Illustration of how offline optimization works in ENVOPT. With a fixed model and harness, ENVOPT uses disjoint sets of training tasks (without verifiers) at each step to optimize the environment. After each step, the optimized environment is tested on a separate validation set to determine when the whole process should be stopped.

reinforcement learning (RL) (Lu et al., 2026; Huang et al., 2026), but these modifications are designed to improve model training rather than the execution efficiency of deployed agents. PREPING and CORPUS2SKILL mainly focus on distilling information from the environment into reusable agent memory and skills, rather than directly optimizing how the stationary environment itself is organized and represented (Choi et al., 2026; Sun et al., 2026). In sum, the community has explored how to utilize environment during RL, optimize the tool interface of environment, or refine the environment for specific use cases, but environment optimization itself as the third direction of agent optimization remains underexplored.

We present ENVOPT, which is an automated pipeline to optimize the environment for any agent and any stationary environment. Using training tasks without verifiers, we let a coding agent, equipped with access and budget to run a target agent on the stationary environment, to optimize the environment representation for the target agent. The optimization target includes what the environment holds (e.g. adding a database to the environment or restructuring it) and ways it is exposed to the target agent (e.g. new API endpoints). The optimized environment is monitored with a validation set with verifiers to track cost and pass rate, and tested on a held-out test set.

Across five benchmarks covering workspace tasks, data engineering tasks, and software engineering tasks (see Section 3), ENVOPT reduces the cost of the target agent on unseen test sets by an average of 48%, with higher or similar pass rates. On AutomationBench (Shepard & Salimans, 2026), ENVOPT outperforms Meta-Harness (Lee et al., 2026) by 9.67% pass rate with only 15% of its cost. We also show that ENVOPT constitutes an optimization direction orthogonal to harness optimization and parameter tuning in two ways: (1) the environments optimized for the target agents can transfer to other agents, even to those with different harnesses and to models stronger than the one proposing the optimizations; (2) we applied the environment from ENVOPT on top of Meta-Harness and observe similar cost reductions and 16% improvement in pass rate, outperforming either Meta-Harness or ENVOPT alone.

Our contributions are as follows:

• We define environment optimization over stationary environments as a new orthogonal direction for improving LLM agents. Our method can be combined with parameter tuning and harness optimization.

• We develop ENVOPT, an automated pipeline for environment optimization, and show that across 5 benchmarks over 3 types of tasks, it can significantly reduce the cost of running agents on unseen tasks while improving or maintaining the pass rates.

![](images/1c0f8f5e20085f22d5b6ea31d9a4b712184284da21694b161f64ff234fb23fb3.jpg)  
Figure 2 Overview of ENVOPT methodology. Upper: ENVOPT optimizes the environment in multiple steps, with each step using a disjoint set of training tasks without verifiers. In each step, a research agent is launched in a sandbox and given the original environment as a local directory and the permission to run target agent in another sandbox. The research agent optimizes environment by iteratively editing the env directory and testing the target agent. At the end of each step, we run validation queries (with verifiers) to decide if ENVOPT has converged or should continue running. Lower: Examples of environment optimization across three benchmarks. The research agent constructed SQLite catalog to map tables to source files, made cheatsheet for more efficient tool use, and built ranking programs to prioritize important files for bug fixes.

• We demonstrate that ENVOPT is indeed orthogonal to parameter tuning and harness optimization by showing strong transferability of environment optimizations to different frontier models and harnesses, and that ENVOPT can further improve results combined with Meta-Harness.

• We systematically discuss the accepted and rejected design choices during running ENVOPT, which can be used for future work on environment optimization.

## 2 Method

ENVOPT is an automated research pipeline that uses a coding agent such as Codex as a research agent R to automatically optimize the environment representation for another target agent T on tasks sharing a stationary environment.

We denote a stationary environment as $E ( I )$ , where I is the information underlying the environment, such as the conversations in a Slack channel; E is the representation of that information in a specific format, such as an official Slack workspace accessible via $\mathrm { { \bf { M C P } } }$ tools or a Markdown file storing all the conversations. Our premise is that for a fixed I needed for the tasks, a better representation $E ^ { * } ( I )$ can significantly improve the token efficiency of agents on the tasks and, in some cases, improve accuracy as well. Given task instances

Algorithm 1 Environment Evolution   
Input: initial environment $E _ { 0 } ( I )$ , target agent $T ,$ research agent $R ,$ training tasks $X _ { \mathrm { t r a i n } }$ (no verifiers),   
validation tasks $X _ { \mathrm { v a l } }$ (with verifiers), number of steps $N _ { \iota }$ , per-step budget $B ,$ accuracy tolerance $\delta$   
Partition $X _ { \mathrm { t r a i n } }$ into disjoint subsets $X _ { 1 } , \ldots , X _ { N }$   
$c _ { 0 } \gets \mathrm { c o s t } ( T , E _ { 0 } ( I ) , X _ { \mathrm { v a l } } )$   
${ a _ { 0 } }  \operatorname { a c c } ( T , E _ { 0 } ( I ) , X _ { \mathrm { v a l } } )$   
for $k = 1$ to N do   
Spawn $R$ in a sandbox with $X _ { k } , E _ { k - 1 } ( I )$ , access to run $T ,$ and budget B   
$\tilde { E }  E _ { k - 1 } ( I )$   
$\tau \gets$ trajectories of $T$ with $\tilde { E }$ on $X _ { k }$   
repeat   
$R$ analyzes $\tau$ and edits $\tilde { E }$ into a candidate $E ^ { \prime }$   
$\tau ^ { \prime } $ trajectories of T with $E ^ { \prime }$ on $X _ { k }$   
if cost of $\tau ^ { \prime }$ is lower than τ and R judges accuracy preserved (proxies $/$ answer consistency) then   
$\tilde { E }  E ^ { \prime } , \ \tau  \tau ^ { \prime }$   
else   
Discard $E ^ { \prime }$   
end if   
until budget B is exhausted or R terminates   
$E _ { k } ( I ) \gets \tilde { E } ;$ discard any changes to the harness of $T$   
$c _ { k } \gets \mathrm { c o s t } ( T , E _ { k } ( I ) , X _ { \mathrm { v a l } } )$   
$a _ { k } \gets \operatorname { a c c } ( T , E _ { k } ( I ) , X _ { \operatorname { v a l } } )$   
if $c _ { k } > c _ { k - 1 } \mathrm { \bf o r } a _ { k } < a _ { k - 1 } - \delta$ then   
return $E _ { k - 1 } ( I )$   
end if   
end for   
return $E _ { N } ( I )$

$X _ { \mathrm { t r a i n } } , X _ { \mathrm { v a l } } , X _ { \mathrm { t e s t } } \sim \mathcal { X }$ on the initial stationary environment $E _ { 0 } ( I )$ , the objective of ENVOPT is to find $E ^ { * }$ that minimizes the cost of the target agent T on X:

$$
E ^ { * } \ = \ \arg \operatorname* { m i n } _ { E } \ \mathbb { E } _ { x \sim \mathcal { X } } \left[ \cot ( T , E ( I ) , x ) \right] .\tag{1}
$$

Environment Evolution As illustrated in Algorithm 1, ENVOPT optimizes $E ( I )$ over N steps. At step $k ,$ the research agent R is spawn in a sandbox and given a set of training tasks $X _ { k } \subset X _ { \operatorname { t r a i n } } ,$ the stationary environment $E _ { k - 1 } ( I )$ from step $k - 1$ , the access to running the target agent $T ,$ and a set budget. We only provide a subset of $X _ { \mathrm { t r a i n } } ,$ since too much context for LLM-based optimization has been shown to be harmful (Li et al., 2026a). We provide disjoint sets for tasks for each step to prevent overfitting. The research agent is instructed to run $T$ on $E _ { k - 1 } ( I ) .$ , analyze the trajectories, and edit $\bar { \boldsymbol { E } } _ { k - 1 } ( I )$ as a directory in its filesystem to reduce the cost of $T$ on $X _ { k }$ without hurting the pass rate (which the research agent judges on itself by proxies or answer consistency, since no verifier is provided for the training tasks as elaborated below). For example, if $E _ { k - 1 } ( I )$ is a big directory of files with no structure, $E _ { k } ( I )$ can be edited to have a separate directory for each domain, make cheat sheets on how to access things, build a small database, make a new tool endpoint for better tool calling, etc. In cases where the environment is highly similar but not identical across tasks, a program can be written to automatically convert a new environment (see Section B.3). In implementation, we use frontier agents as the research agent, and give it freedom to choose any design choices. The research agent iterates internally over the step-k training tasks and deliver the best $\dot { E } _ { k } ( I )$ it can find at the end of the step. Notably, we do not allow the research agent to modify the harness of $T$ (any changes, though not observed in our experiments, will be discarded at the end of the step). This makes our environment evolution an orthogonal direction to harness optimization, and a good environment can show significant cost reduction even for different agents (Fig. 3).

Training with No Verifiers To prevent the research agent from reward hacking during a training step (such as writing down the answers to the training tasks to lower the cost directly), we hide all the verifiers for the tasks used in training. This also mitigates the scarcity of high quality tasks for some stationary environments, as we can synthesize in-distribution queries (without synthesizing the verifiers) for training (see Section 3). We keep a separate validation set $\dot { X } _ { \mathrm { v a l } }$ with verifiers and run the target agent with $E _ { k } ( I )$ on it at the end of each training step k to decide when to stop the environment evolution. Concretely, if cost $( T , E _ { k } ( I ) , X _ { \mathrm { v a l } } ) > \mathrm { c o s t } ( T , \Breve { E _ { k - 1 } } ( \grave { I } ) , X _ { \mathrm { v a l } } )$ , or if the pass rate on the validation set becomes significantly worse than step $k - 1$ , we stop the evolution and use $\bar { E } _ { k - 1 }$ as the final optimized environment.

## 3 Experiments

We evaluate ENVOPT across 5 benchmarks over 3 types of tasks: workspace tasks (AutomationBench (Shepard & Salimans, 2026), Workspace-Bench (Tang et al., 2026)), data engineering tasks (Data-Eng-Bench (Snowflake Inc., 2026)), and software engineering tasks (Bleve repository tasks as used in GSKILL (Agrawal et al., 2026a) and the SWE Agent Efficiency tasks as used in ADVISOR MODELS (Asawa et al., 2026)). Comparing with the baseline, we observe substantial cost reduction up to 88%, with higher or similar agent success rate. We also show that across all tested scenarios, the optimized environment can also greatly improve token efficiency and pass rate for different harnesses and different models, including agents stronger than the target agent, demonstrating strong transferability. Notably, ENVOPT can also build on top of harness optimization methods like Meta-Harness and further push the Pareto frontier.

AutomationBench includes workspace tasks that ask an agent to answer questions or complete work inside an enterprise environment, often covering multiple platforms such as Slack, GitHub, Gmail, Salesforce, etc. From the 600 publicly available tasks, we randomly sample 200 tasks for $X _ { \mathrm { t e s t } } , 9 6$ for $X _ { \mathrm { v a l . } }$ , and use the rest for the training task pool. Training proceeds with 20 tasks per step and stops at step 5, where results on $X _ { \mathrm { v a l } }$ begin to saturate. We use Claude Code (Anthropic, 2025a) with Claude Opus 4.8 (Anthropic, 2026b) as the research agent, and Claude Code with Claude Haiku 4.5 (Anthropic, 2025b) as the target agent.

As shown in Fig. 3, we observe substantial improvement of token efficiency and pass rate for the target agent, with a final cost reduction of 72.7% and pass rate increase of 14.3%. We show the per-step performance in Section B. We also run GEPA (Agrawal et al., 2026b) and Meta-Harness (Lee et al., 2026) using the same training task pool, same research agent or proposer, same target agent, and evaluate on the same $X _ { \mathrm { t e s t } }$ ENVOPT produces an environment that that achieves significantly better outcomes on both cost and pass rate, even without verifiers for training tasks.

We also examine the transferability of the optimized environment across different harnesses and models, with an average cost reduction of $\dot { 7 } 5 . 7 \%$ and generally higher pass rates. Notably, ENVOPT with a research agent of Claude Code and Opus 4.8 is able to transfer to Claude Code with Opus 5 (Anthropic, 2026c). Haiku 5.5 (Anthropic, 2026a) with ENVOPT achieves pass rate higher than Opus 4.8 and even close to Opus 5.

Additionally, ENVOPT can be applied on top of Meta-Harness to further improve agent’s token efficiency and pass rate. Using the optimized harness for Haiku 4.5, ENVOPT can further push the Pareto frontier for Haiku 4.5. When transferring to Opus 5, although Meta-Harness alone does not show gain on pass rate, ENVOPT on top of Meta-Harness still improves the pass rate with much lower cost.

Workspace-Bench evaluates agents on workers’ personal workspaces, including researchers, engineers, product managers, etc. We use the public set of 100 tasks as used in the official leaderboard and split into 80 tasks for $X _ { \mathrm { t e s t } }$ and 20 for $X _ { \mathrm { v a l } } .$ Since our training steps do not require verifiers and the number of tasks is sparse for this benchmark, we directly synthesize queries without verifiers with Claude Code and Claude Opus 5 to serve as training tasks. We provide 20 distinct training tasks per step, and use Claude Code with Opus 4.8 as the research agent, Codex (OpenAI, 2025) with GPT-5.4-mini (OpenAI, 2026a) as the target agent. Claude Code with Sonnet 5 is used as the judge agent for evaluation on $X _ { \mathrm { t e s t } }$ and $X _ { \mathrm { v a l } }$ . We present the results in Table 1. We observe 47.65% in cost reduction and 3.3% in pass rate improvement for the target agent. On transferability to non-target agents, we evaluate on Codex with GPT-5.6-Sol (OpenAI, 2026b) and Claude Code with Opus 5, and observe an average 56.7% cost reductions with steady pass rates compared with the baselines.

![](images/aef805cf47cdcba6e7595ab608a55e4b0f6337156bc3f7fd6212fcacb4c34bbc.jpg)  
Figure 3 Results for AutomationBench. For each agent and each method, we show the pass rate and cost (USD) averaged over 3 independent runs. We use the same color for different methods applied on the same agent, including baseline (original environment), GEPA, Meta-Harness, Ours, and Ours + Meta-Harness. Throughout all tested agents, ENVOPT significantly reduces total cost and improves pass rate, outperforming GEPA and Meta-Harness. Claude Haiku 5.5 with ENVOPT can have success rate similar to Claude Opus 5 with the baseline environment. Additionally, ENVOPT can be applied on top of Meta-Harness, pushing the Pareto frontier for Haiku 4.5 and Opus 5.

Data-Eng-Bench evaluates agents on dbt data engineering tasks. We use 73 tasks as $X _ { \mathrm { t e s t } }$ and 19 as $X _ { \mathrm { v a l } } .$ Similarly, due to the scarcity of existing tasks, we synthesized queries without verifiers for training, with 20 distinct tasks per step. We use Codex with GPT-5.6-Sol as the research agent and Codex with GPT-5.6-luna as the target, and reduced cost by 53.44% without hurting the performance. On non-target agents, we observe an average cost reduction of 53.54% (see Table 1).

Software Engineering Tasks We take two benchmarks from previous work that improves token efficiency of agents on SWE-Bench style tasks, where an agent is asked to fix a bug in a given code repository, with all tasks sharing the same codebase. GSKILL optimizes skills for coding agents using tasks with verifiers from the same codebase and reports higher pass rate and faster completion time on the Bleve search library. ADVISOR MODELS (Asawa et al., 2026) uses reinforcement learning to train an advisor model that produces advice to the coding agent. On the SWE Agent Efficiency benchmark based on the tenacity library, ADVISOR MODELS maintains the same pass rate as baseline while using significantly fewer steps. We run ENVOPT using the same $X _ { \mathrm { t r a i n } } , X _ { \mathrm { t e s t } } ,$ and $X _ { \mathrm { v a l } }$ as provided by GSKILL and ADVISOR MODELS and use the same target agent. As shown in Fig. 4, on Bleve repository, ENVOPT remains close to perfect score using only 56.9% of the completion time of GSKILL. On the tenacity repository, ENVOPT achieves 8.9% higher pass rate with fewer steps than ADVISOR MODELS. These results show that ENVOPT significantly outperforms other methods to improve agent token efficiency even on software engineering tasks.

## 4 Discussion

## 4.1 Example Environment Optimizations

Preparing file contents On AutomationBench, the research agent built a program that directly retrieves Gmail information and presents it as plain text to the target agent, covering the sender, the subject, the message ID, and chunks of the email. The target agent can thus directly read the text instead of spending tokens on massive tool calls. On Workspace-Bench, different formats of files such as .xlsx are converted to .txt files so that the target agent can read the files more easily and do not need to do further parsing.

Table 1 Detailed results comparing baseline environments and ENVOPT environments on AutomationBench, Workspace-Bench, and Data-Eng-Bench. For AutomationBench, we show results using the official harness released by Shepard & Salimans (2026). Agents marked with † were used as the target agent during ENVOPT, while the other agents show transfer results using the environment optimized against the corresponding target agents. Across all agents and benchmarks, ENVOPT significantly reduces cost and improves or maintains pass rate.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Agent</td><td colspan="2">Accuracy (%)</td><td colspan="2">Cost (USD)</td><td rowspan="2">Cost red. (%)</td></tr><tr><td>Base</td><td>Evolved</td><td>Base</td><td>Evolved</td></tr><tr><td rowspan="5">AutomationBench</td><td>Haiku 4.5†</td><td>2.33</td><td>16.67</td><td>42.36</td><td>11.55</td><td>-72.73</td></tr><tr><td>Haiku 5.5</td><td>20.7</td><td>32.8</td><td>4.83</td><td>1.51</td><td>-68.74</td></tr><tr><td>GPT-5.6-luna</td><td>19.17</td><td>24.67</td><td>17.42</td><td>1.98</td><td>-88.63</td></tr><tr><td>Opus 4.8</td><td>28.50</td><td>33.33</td><td>358.13</td><td>80.90</td><td>-77.41</td></tr><tr><td>Opus 5</td><td>33.83</td><td>35.17</td><td>236.18</td><td>75.26</td><td>-68.13</td></tr><tr><td rowspan="3">Workspace-Bench</td><td>Codex + GPT-5.4-mini†</td><td>39.84</td><td>43.12</td><td>12.55</td><td>6.57</td><td>-47.65</td></tr><tr><td>Codex + GPT-5.6-sol</td><td>53.98</td><td>55.05</td><td>64.59</td><td>27.45</td><td>-57.50</td></tr><tr><td>Claude Code + Opus 5</td><td>62.27</td><td>61.14</td><td>137.41</td><td>60.64</td><td>-55.86</td></tr><tr><td rowspan="3">Data-Eng-Bench</td><td>Codex + GPT-5.6-luna†</td><td>50.68</td><td>53.88</td><td>16.13</td><td>7.51</td><td>-53.44</td></tr><tr><td>Codex + GPT-5.6-sol</td><td>62.56</td><td>61.19</td><td>182.58</td><td>79.10</td><td>-56.68</td></tr><tr><td>Claude Code + Opus 5</td><td>63.47</td><td>63.01</td><td>434.42</td><td>215.53</td><td>-50.39</td></tr></table>

Making Cheatsheets Cheatsheets that store important information for future tasks are created during training. For example, a cheatsheet for AutomationBench stores all the API endpoints the agent needs to create Jira issues, post messages on Slack, record a Xero payment, etc., so it does not need to call api\_search or api\_fetch to get the endpoints. For the Bleve repository bug fixes, a cheatsheet was built that stores template commands for complex tool calls, so the agent does not need to compose search commands from scratch.

Indexing For Workspace-Bench, the agent is deployed with massive amount of files simulating an enterprise employee’s computer. The research agent thus builds a indexing map as a single text file, that stores the paths, sizes, and types of all files needed, so the can simply search over the indexing file instead of spending tokens and time on the huge filesystem.

Building Database On Data-Eng-Bench, the research agent built an SQLite database that stores the locations of tables, which files each code script writes to, what are the types of information stored in each table, etc. The target agent can quickly locate the information needed without browsing through thousands of files.

Building Programs that Ranks File Priority For the Bleve repository, the research agent built a program that takes any bug report and outputs the 8 .go files that are most likely to be impacted based on keyword matching. The target agent can quickly locate the files it potentially need investigate to fix the bugs.

![](images/319ee68661cf73da63b6713480ead5713147664bdc27e52fa22d4a839eab0e48.jpg)  
Figure 6 Cost savings vs. training cost on Automation-Bench. For each agent, solid lines show how much cost each step’s ENVOPT environment would save over 1,000 tasks. The dashed line shows the cumulative training cost of EN-VOPT over training steps. Savings exceed the training cost for all three agents, by up to about 10× for Opus 4.8.

![](images/c92d7ab444dd2443bfec7c9a73d6f3f3d255f153ca9e4ebaa95641a4b11d2d12.jpg)  
Figure 4 Results on two software engineering repositories. Left: on the bleve repository, ENVOPT keeps accuracy close to perfect while using 56.9% of GSKILL’s completion time. Right: on the tenacity repository, ENVOPT reaches 8.9% higher accuracy than ADVISOR MODELS with fewer steps.

Automated Testing of Edits For the Tenacity repository, a validate\_tenacity.py is built to automatically fetch the code change made by the target agent, search for relevant test files, run the tests, and return verdicts. This quickly validates candidates edit with no extra token cost and speeds up the whole bug fix workcycle for the target agent.

## 4.2 Rejected Environment Optimizations

We summarize environment optimizations that are discarded during training or after running on $X _ { \mathrm { v a l } } { \mathrm { : } }$

Storing too much information in one file leads to more exploration For Data-Eng-Bench, the research agent tried to create a big file to store all possible values for all columns in a database. Though this is meant to help the agent retrieve the correct information with less token, putting everything into a big file encouraged the agent to make more tool calls to explore the values, since one tool result reveals to the agent that there is much information it has not explored, which eventually increased the cost.

Building a search tool that returns too many files causes more reading For Workspace-Bench, a search tool was built to help the agent locate the files that it needs to read for the task. Though the tool successfully reduced token cost for searching the needed files, it often included unnecessary files in the search results, which motivated the agent to read more files, thus increasing the cost.

Adding tests that only fit a subset of the training tasks In step-2 of ENVOPT on the Bleve repository, a check was added to a test script in the repository that scans for a particular bug fix code snippet and fails the test directly if the snippet is present. This was meant to save the time needed to fix a bug and worked well for $X _ { 2 } .$ . However, for ${ \bar { X } } _ { 3 } ,$ some correct bug fixes were rejected by this additional check, and the agent needed to spend much more time to bypass the check.

## 4.3 Cost Reduction Analysis

To illustrate where the cost reduction of ENVOPT comes from, we break down input tokens and output tokens into categories, including different kinds of tool calls, tool execution results (environment feedback), thinking tokens, agent history, etc. We compare the numbers of tokens of each category between the baseline environment and ENVOPT environment, and show results in Fig. 5. We observe that the most significant token cost reduction comes from the environment feedback categories for the input tokens. There are also significant savings on the tool call categories of the output token side. These results align with our motivation to optimize the environment for better agent interaction.

![](images/5638ff5ab16394b29a39bca9a7664154b033804b41db762f1a15b1bdc57d0797.jpg)

![](images/b14656b3f9ea7322d211ee5cd1e3a920985a98da7581ba3a34f3c1899296745d.jpg)

Data-Eng-Bench  
![](images/2aac29026bc3c6df08ce2043ad1368fd594d46faa5b22b208c3ef756c92eb3d9.jpg)  
Figure 5 Breakdown of token savings by categories on AutomationBench and Data-Eng-Bench. Grey bars show the baseline environment and green bars show the final ENVOPT environment. Most of cost reduction comes from input token savings, mainly tool execution results, such as endpoint lookup on AutomationBench and data inspection on Data-Eng-Bench. Output tokens drop less, around 20% compared to baseline.

## 4.4 Ablation: Tool Optimization Only

During environment optimization, ENVOPT can also optimize ways to access the new environment, including new tools. We study the effectiveness of only using the new tools. On AutomationBench, we test Claude Code with Haiku 4.5 on the original environment $E _ { 0 }$ with the tools from $E _ { 5 }$ . As presented in Table 2, we find that tool optimization alone can lead to improvements on both token efficiency and pass rate, but is significantly lower than optimizing both the tools and the environment. Since tool optimization might be covered by harness engineering, this ablation shows that ENVOPT brings significantly more headroom for optimization in addition to harness engineering.

Table 2 Cost and pass rates for tool-only optimization on AutomationBench and Data-Eng-Bench. Results show that while tool-only optimization can improve pass rates and reduce costs compared to baselines, the full ENVOPT yields significantly greater gains, particularly in cost.
<table><tr><td>Agent</td><td>Optimization</td><td>Pass (%)</td><td>Cost (USD)</td></tr><tr><td colspan="2">AutomationBench</td><td></td><td></td></tr><tr><td rowspan="3">Haiku 4.5</td><td>Baseline</td><td>2.33</td><td>42.36</td></tr><tr><td>Tools-only Opt</td><td>6.33</td><td>15.75</td></tr><tr><td>Full ENVOPT</td><td>16.67</td><td>11.55</td></tr><tr><td rowspan="3">GPT-5.6-luna</td><td>Baseline</td><td>19.17</td><td>17.42</td></tr><tr><td>Tools-only Opt</td><td>21.83</td><td>2.94</td></tr><tr><td>Full ENVOPT</td><td>24.67</td><td>1.98</td></tr><tr><td rowspan="3">Opus 5</td><td>Baseline</td><td>33.83</td><td>236.18</td></tr><tr><td>Tools-only Opt</td><td>35.17</td><td>100.37</td></tr><tr><td>Full ENVOPT</td><td>35.17</td><td>75.26</td></tr><tr><td colspan="2">Data-Eng-Bench</td><td></td><td></td></tr><tr><td rowspan="3">Haiku 4.5</td><td>Baseline</td><td>26.94</td><td>103.27</td></tr><tr><td>Tools-only Opt</td><td>26.03</td><td>109.55</td></tr><tr><td>Full ENvOPT</td><td>27.85</td><td>66.36</td></tr><tr><td rowspan="3">GPT-5.6-luna</td><td>Baseline</td><td>50.68</td><td>16.13</td></tr><tr><td>Tools-only Opt</td><td>57.08</td><td>14.60</td></tr><tr><td>Full ENVOPT</td><td>53.88</td><td>7.51</td></tr><tr><td rowspan="3">Opus 5</td><td>Baseline</td><td>63.47</td><td>434.42</td></tr><tr><td>Tools-only Opt</td><td>63.93</td><td>530.58</td></tr><tr><td>Full ENVOPT</td><td>63.01</td><td>215.53</td></tr></table>

## 4.5 Training Cost and Time

We present the total training cost (including the cost of target agent runs launched by the research agent) and clock time for all 5 benchmarks in Table 3. In Fig. 6, we compare the cumulative training cost with the cost saved for running different agents on unseen tasks, assuming there are 1000 such tasks. We show that for frontier agents deployed in large scale, we can easily save more than 10 times of the cost of running ENVOPT.

## 4.6 Research Agent Behaviors

Here we present some interesting behaviors we observed from the research agent, including but not limited to failure modes and how behaviors during training can be related to ENVOPT performance on heldout tasks.

Using unreliable verification proxy Since we do not provide verifiers for training tasks, the research agent for Workspace-Bench occasionally used proxies such as answer length and keyword matching to confirm answer quality during training steps. This often turns out to hurt performance on heldout tasks. After observing validation set, we updated the instruction for research agent to prevent this behavior (see Section A).

Predicting validation performance using research agent behavior We find that for AutomationBench, cost reduction on validation set is correlated with how much tool call patterns change during each training step as measured in the per-task difference in tool call counts between training runs before and after the step, with Pearson correlation of 0.75. Alternatively, by looking at the fraction of training tasks that received more than 30% cost reduction in each step, its Pearson correlation with validation set cost reduction is 0.82 for AutomationBench, 0.70 for Workspace-Bench, and 0.56 for Data-Eng-Bench.

Fail to follow the instruction Despite being told not touch anything related to the model or the harness, the research agent in Workspace-Bench still tried to change the reasoning effort of Codex during its step 6. This change was not effective on heldout tasks, as we only keep the environment directory in the research agent’s sandbox and discard the other artifacts. However, such behavior early in one training step could significantly affect the research agent’s optimizations during that step, potentially harming the performance.

Table 3 Training cost (USD) and duration (clock time) for each step of ENVOPT and each benchmark we run. Numbers include the cost and time of the target agent runs launched by the research agent. We also give the total numbers for each benchmark. Overall, the cost and time of running ENVOPT is only a fraction of the cost and time saved.
<table><tr><td rowspan="3"></td><td colspan="2">AutomationBench</td><td colspan="2">Workspace-Bench</td><td colspan="2">Data-Eng-Bench</td><td colspan="2">Bleve</td><td colspan="2">Tenacity</td></tr><tr><td>Cost</td><td>Time</td><td>Cost</td><td>Time</td><td>Cost</td><td>Time</td><td>Cost</td><td>Time</td><td>Cost</td><td>Time</td></tr><tr><td>Step 1</td><td>$15.33</td><td>0.5 h</td><td>$71.24</td><td>1.1 h</td><td>$10.99</td><td>6.5 h</td><td>$8.11</td><td>1.8 h</td><td>$11.71</td><td>1.3 h</td></tr><tr><td>Step 2</td><td>$27.37</td><td>1.2 h</td><td>$86.48</td><td>1.6 h</td><td>$9.50</td><td>6.5 h</td><td>$13.30</td><td>2.6 h</td><td>$7.91</td><td>0.9 h</td></tr><tr><td>Step 3</td><td>$50.48</td><td>2.2 h</td><td>$28.70</td><td>2.6 h</td><td>$8.71</td><td>6.5 h</td><td>$10.25</td><td>1.8 h</td><td>$6.00</td><td>0.7 h</td></tr><tr><td>Step 4</td><td>$21.37</td><td>1.0 h</td><td>$36.89</td><td>1.6 h</td><td>$7.53</td><td>5.9 h</td><td>$8.12</td><td>1.2 h</td><td>$7.46</td><td>0.8 h</td></tr><tr><td>Step 5</td><td>$15.54</td><td>0.7 h</td><td>$55.87</td><td>5.2 h</td><td></td><td></td><td>$8.59</td><td>1.5 h</td><td></td><td></td></tr><tr><td>Step 6</td><td>$20.01</td><td>0.9 h</td><td>$60.26</td><td>3.3 h</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Total</td><td>$150.10</td><td>6.6 h</td><td>$339.44</td><td>15.4 h</td><td>$36.73</td><td>25.4 h</td><td>$48.37</td><td>8.9 h</td><td>$33.08</td><td>3.7 h</td></tr></table>

How far can we go without a validation set From the tested five benchmarks, we observe that it is generally safe to run ENVOPT for 3 training steps even without running the validation tasks. However, we note that this is only an empirical claim and that the validation set with real verifiers is still important for auditable environment optimization.

Early flaw in optimization can remain uncovered in many later steps In Data-Eng-Bench, the research agent introduced a SQL statement splitter in step 1 that parses based on semicolon. But the splitter will fail if the semicolon is in quoted text. This bug in the optimization from step 1 remain uncovered until step 5. This inflates cost through all the 4 steps, as the target agent cannot tell that the splitter corrupted its query, so it attributes the failure to its own model and spends tokens to rebuild it.

## 5 Related Work

Agent Environment Adaptation Previous work has explored environment engineering for LLMs or LLM agents for better test-time performance or for training. For test-time deployment, SWE-agent (Yang et al., 2024) introduced specialized agent-computer interfaces for software engineering tasks. Subsequently, ALIGN (Liu et al., 2025) automatically aligns the interface between agents and environments to help with agent understanding of the environment. Xu et al. (2026); Mak et al. (2026); DevRev Editorial (2026) demonstrate that different tool architectures and interfaces affect agent performance and efficiency. Aegis (Song et al., 2025) systematically analyzes agent-environment interaction failures and proposes environment optimizations based on the failure taxonomy. SlackBench (Li et al., 2026b) also found that changing agent’s way of access to Slack data from MCP to CLI can improve success rate and lower cost. DocSearch (Cheng et al., 2026) uses test-time feedback to optimize repository documentation for coding agents. Libra (Zhao et al., 2026) evolves catalogs as navigable indices in code repositories to facilitate information retrieval. PREPING (Choi et al., 2026) uses synthetic tasks to construct pre-task procedural memory for agents before they are deployed in target-environmen tasks. META-AGENT (Samuel et al., 2026) extends the task-agnostic environment preprocessing by adding study artifacts to the environment and shows gains in pass rates while no significant improvement on token efficiency.

Another line of work adapts the environments to facilitate agent training. ENVIRONMENT TUNING (Lu et al., 2026) proposes environment tuning to improve reinforcement learning through corrective feedback, structured curricula, and fine-grained progress rewards. EnvHarness (Huang et al., 2026) introduces programmable environment wrappers that adapt to agent weaknesses and provide improved training signals for reinforcement learning. Agent-World (Dong et al., 2026) synthesizes diverse interactive environments and verifiable tasks to support agent training.

Harness Optimization Harness optimization improves LLM agents by improving the textual wrapper around model parameters, including prompts, tool-use workflows, context management, execution logic, etc. Agent harnesses such as OpenHands (Wang et al., 2025) and mini-SWE-agent (Yang et al., 2024) provide carefully designed workflows for complex agentic tasks. More recently, Lee et al. (2026) employs an agentic proposer to automatically optimize harness code by inspecting previous implementations, execution trajectories, and evaluation feedback. Agentic Harness Engineering (Lin et al., 2026) similarly automates the evolution of coding-agent harnesses, jointly modifying components such as tools, middleware, and memory based on trajectories.

Other work focuses on optimizing individual components of an agent’s harness. For prompt optimization, DSPy (Khattab et al., 2024) introduced a programming framework for optimizing LM pipelines, while GEPA (Agrawal et al., 2026b) uses reflection over execution trajectories to evolve prompts. Combee (Li et al., 2026a) further scales prompt optimization to parallel agent executions. Beyond prompts, optimize anything (Agrawal et al., 2026a) provides a unified framework for optimizing text-based artifacts, including agent architectures and skills. SkillOpt (Yang et al., 2026) iteratively optimizes reusable agent skills through trajectory feedback and validation. Another line of work optimizes agent context and memory. ACE (Zhang et al., 2026) evolves structured contextual playbooks through generation, reflection, and curation. Meta Context Engineering (Ye et al., 2026) jointly evolves context-engineering strategies and context artifacts. For agent memory, ReasoningBank (Ouyang et al., 2026) distills reusable reasoning strategies from past trajectories and MemEvolve (Zhang et al., 2025) jointly optimizes memory contents and memory architectures.

## 6 Conclusion

We formulate optimization over stationary environments as the third direction of agent optimization, along side harness optimization and parameter tuning. This work introduces ENVOPT, an automated research pipeline for end-to-end environment optimization, and shows that it significantly reduces agent cost while maintaining or improving success rates across five benchmarks. ENVOPT transfers effectively across different models and harnesses, outperforming prior agent optimization methods including GEPA, Meta-Harness, GSKILL, and ADVISOR MODELS. In addition, ENVOPT can build upon Meta-Harness to push the Pareto frontier even further. Our analysis shows that ENVOPT significantly outperforms tool-only optimization and yields cost savings exceeding 10 times the cost of running ENVOPT itself. Finally, we detail key examples of successful and rejected environment optimizations to serve as a qualitative reference and provide design experience for future research.

## Acknowledgements

This project is a collaboration of University of Chicago, UC Berkeley Sky Computing Lab, and Bespoke Labs. This project is partly supported by NSF IFML and Bespoke Labs. Sky Computing Lab is supported by gifts from Accenture, Algorithmic SuperIntelligence Labs, Amazon, AMD, Anyscale, Broadcom, cmpnd, Google, IBM, Intel, Intesa Sanpaolo, Lambda, Lightspeed, Mirendil, NVIDIA, Soma Labs, and VESSL.

## References

Agrawal, L. A., Lee, D., Tan, S., Ma, W., Elmaaroufi, K., Sandadi, R., Seshia, S. A., Sen, K., Klein, D., Stoica, I., Gonzalez, J. E., Khattab, O., Dimakis, A. G., and Zaharia, M. optimize anything: Unified text optimization can outperform specialized systems. In Proceedings of the ACM Conference on AI and Agentic Systems, CAIS ’26, pp. 1–16. ACM, May 2026a. doi: 10.1145/3786335.3813167. URL http://dx.doi.org/10.1145/ 3786335.3813167.

Agrawal, L. A., Tan, S., Soylu, D., Ziems, N., Khare, R., Opsahl-Ong, K., Singhvi, A., Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Dimakis, A. G., Stoica, I., Klein, D., Zaharia, M., and Khattab, O. Gepa: Reflective prompt evolution can outperform reinforcement learning, 2026b. URL https://arxiv.org/ abs/2507.19457.

Anthropic. Claude code: An agentic coding tool, 2025a. URL https://www.anthropic.com/claude-code.

Anthropic. Introducing claude haiku 4.5, 2025b. URL https://www.anthropic.com/news/ claude-haiku-4-5.

Anthropic. Claude haiku 5.5. Anthropic, October 2026a. URL https://www.anthropic.com/ claude-haiku-5-5.

Anthropic. Introducing claude opus 4.8, 2026b. URL https://www.anthropic.com/news/claude-opus-4-8.

Anthropic. Introducing claude opus 5, 2026c. URL https://www.anthropic.com/news/claude-opus-5.

Asawa, P., Zhu, A., O’Neill, A., Zaharia, M., Dimakis, A. G., and Gonzalez, J. E. How to train your advisor: Steering black-box llms with advisor models, 2026. URL https://arxiv.org/abs/2510.02453.

Cheng, Y., Chen, H., Yu, W., Zhao, X., Gao, P., and Cheng, W. Escaping whack-a-mole: Optimizing documentation as repo-specific playbooks for coding agents. In Zhang, T., Dudik, M., Jaggi, M., Agarwal, A., Li, S., Schuurmans, D., Zhu, J., Berkenkamp, F., Dong, H., and Bietti, A. (eds.), Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research, pp. 18800–18827. PMLR, 06–11 Jul 2026. URL https://proceedings.mlr.press/v306/cheng26c.html.

Choi, Y., Park, S., Kang, M., Baek, J., and Hwang, S. J. Preping: Building agent memory without tasks, 2026. URL https://arxiv.org/abs/2605.13880.

DevRev Editorial. Introducing enterprise-bench, July 2026. URL https://devrev.ai/blog/ enterprise-bench.

Dong, G., Lu, J., Huang, J., Zhong, W., Liu, L., Huang, S., Li, Z., Zhao, Y., Song, X., Li, X., Jin, J., Zhu, Y., Wang, H., Lei, F., Luo, Q., Chen, M., Chen, Z., Feng, J., Wen, J.-R., and Dou, Z. Agent-world: Scaling real-world environment synthesis for evolving general agent intelligence, 2026. URL https://arxiv.org/abs/2604. 18292.

Huang, C., Wang, Z., Han, R., Yan, J., Chen, Y., CuiZhu, Z., Jiang, K., Xia, P., Yu, H., Zhuang, Y., Ming, Y., Pan, J., Mishra, B. D., Huang, J., Gokturk, B., Pfister, T., and Lee, C.-Y. Envharness: Awakening static worlds for agent learning, 2026. URL https://arxiv.org/abs/2608.19880.

Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., Haq, S., Sharma, A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., and Potts, C. Dspy: Compiling declarative language model calls into self-improving pipelines. 2024.

Lee, Y., Nair, R., Zhang, Q., Lee, K., Khattab, O., and Finn, C. Meta-harness: End-to-end optimization of model harnesses, 2026. URL https://arxiv.org/abs/2603.28052.

Li, H., He, R., Zhang, Q., Ji, C., Mang, Q., Chen, X., Agrawal, L. A., Liao, W.-L., Yang, E., Cheung, A., Zou, J., Olukotun, K., Stoica, I., and Gonzalez, J. E. Combee: Scaling prompt learning for self-improving language model agents, 2026a. URL https://arxiv.org/abs/2604.04247.

Li, M., Mao, H., Zheng, R., James, J. S., Li, Y., Anderson, N., Gonzalez, J. E., and Dimakis, A. Slackbench: Benchmarking agents on collaborative projects grounded in real code repositories, 2026b. URL https: //arxiv.org/abs/2611.05512.

Lin, J., Liu, S., Pan, C., Lin, L., Dou, S., Xi, Z., Huang, X., Yan, H., Han, Z., Gui, T., and Jiang, Y.-G. Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses, 2026. URL https://arxiv.org/abs/2604.25850.

Liu, K., Lei, X., Wang, Z., Li, P., and Liu, Y. Agent-environment alignment via automated interface generation, 2025. URL https://arxiv.org/abs/2505.21055.

Lu, S., Wang, Z., Zhang, H., Wu, Q., Gan, L., Zhuang, C., Gu, J., and Lin, T. Don’t just fine-tune the agent, tune the environment, 2026. URL https://arxiv.org/abs/2510.10197.

Mak, H., Suresh, S., Bhatnagar, S., Wang, B., Methani, C., and Munoz, A. G. Is bash all you need? an empirical study of tool interfaces for enterprise digital worker agents, 2026. URL https://arxiv.org/abs/2609. 11999.

OpenAI. Introducing codex, 2025. URL https://openai.com/index/introducing-codex/.

OpenAI. Introducing gpt-5.4 mini and nano, 2026a. URL https://openai.com/index/ introducing-gpt-5-4-mini-and-nano/.

OpenAI. Gpt-5.6: Frontier intelligence that scales with your ambition, 2026b. URL https://openai.com/ index/gpt-5-6/.

Ouyang, S., Yan, J., Hsu, I.-H., Chen, Y., Jiang, K., Wang, Z., Han, R., Le, L. T., Daruki, S., Tang, X., Tirumalashetty, V., Lee, G., Rofouei, M., Lin, H., Han, J., Lee, C.-Y., and Pfister, T. Reasoningbank: Scaling agent self-evolving with reasoning memory, 2026. URL https://arxiv.org/abs/2509.25140.

Samuel, V., Ursekar, V., Kalmath, V. S., Shanker, A., Chatrath, V., and Xue, Y. Studying without a syllabus: Task-agnostic environment preprocessing, 2026. URL https://arxiv.org/abs/2609.10824.

Shepard, D. and Salimans, R. Automationbench, 2026. URL https://arxiv.org/abs/2604.18934.

Snowflake Inc. data-eng-bench: A data-engineering benchmark for coding agents, July 2026. URL https: //github.com/Snowflake-Labs/data-eng-bench. Released 2026-07-29.

Song, K., Jayarajan, A., Ding, Y., Su, Q., Zhu, Z., Liu, S., and Pekhimenko, G. Aegis: Taxonomy and optimizations for overcoming agent-environment failures in llm agents, 2025. URL https://arxiv.org/ abs/2508.19504.

Sun, Y., Wei, P., and Hsieh, L. B. Corpus2skill: Distilling enterprise knowledge into navigable agent skills for qa and rag, 2026. URL https://arxiv.org/abs/2604.14572.

Tang, Z., Zhou, X., Liu, Y., Li, L., Wu, Y., Wang, W., Huang, H., Zhou, W., Zhou, J., Song, J., Yu, S., Wang, J., Zhou, Z., Zhou, H., Lv, Y., Li, J., Liu, J., Chen, R., Liu, C., Li, G., Kang, J., and Wu, F. Workspacebench 1.0: Benchmarking ai agents on workspace tasks with large-scale file dependencies, 2026. URL https://arxiv.org/abs/2605.03596.

Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., Pan, J., Song, Y., Li, B., Singh, J., Tran, H. H., Li, F., Ma, R., Zheng, M., Qian, B., Shao, Y., Muennighoff, N., Zhang, Y., Hui, B., Lin, J., Brennan, R., Peng, H., Ji, H., and Neubig, G. Openhands: An open platform for ai software developers as generalist agents, 2025. URL https://arxiv.org/abs/2407.16741.

Xu, X., Saghir, H., Wu, Q., Cotˆ e, M.-A., Wang, T., Lakkaraju, K., Pei, K., and Zhang, X. The devil is´ in the interface: Evaluating how tool architecture shapes coding agent behavior, 2026. URL https: //arxiv.org/abs/2608.11386.

Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K. R., and Press, O. SWE-agent: Agentcomputer interfaces enable automated software engineering. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://arxiv.org/abs/2405.15793.

Yang, Y., Gong, Z., Huang, W., Yang, Q., Zhou, Z., Huang, Z., Li, Y., Gao, X., Dai, Q., Liu, B., Qiu, K., Yang, Y., Chen, D., Yang, X., and Luo, C. Skillopt: Executive strategy for self-evolving agent skills, 2026. URL https://arxiv.org/abs/2605.23904.

Ye, H., He, X., Arak, V., Dong, H., and Song, G. Meta context engineering via agentic skill evolution, 2026. URL https://arxiv.org/abs/2601.21557.

Zhang, G., Ren, H., Zhan, C., Zhou, Z., Wang, J., Zhu, H., Zhou, W., and Yan, S. Memevolve: Meta-evolution of agent memory systems, 2025. URL https://arxiv.org/abs/2512.18746.

Zhang, Q., Hu, C., Upasani, S., Ma, B., Hong, F., Kamanuru, V., Rainton, J., Wu, C., Ji, M., Li, H., Thakker, U., Zou, J., and Olukotun, K. Agentic context engineering: Evolving contexts for self-improving language models, 2026. URL https://arxiv.org/abs/2510.04618.

Zhao, X., Chiu, A., and Wang, G. Libra: Training the environment for agentic information retrieval, 2026. URL https://arxiv.org/abs/2607.00016.

## A Research Agent Instructions

We include here the instructions provided to the research agent on the five benchmarks in Section 3.

## A.1 AutomationBench

How you work (this session is fully automated)

No human is watching and you will not get any further instructions — so do not stop early, do not ask for confirmation, and do not wait. Keep working the whole time.

• Do many rounds of: change the access layer → run the target on the training tasks → read its cost and what it did → keep what helps → try the next idea. Treat “I made one improvement” as the start, not the end.

• When you’ve genuinely exhausted your ideas or reached your spend budget (below), leave a short REPORT.md in /work: what you tried, what helped, what didn’t, the before/after cost and behavior, and what you’d try next. That report is the finish line.

You are headless and non-interactive. There is NO one to re-invoke you and NO wakeups/notifications/waiters — they do not fire here. If you end your turn, the session ends permanently and unfinished work is lost. Never end your turn to “wait.” A benchmark run takes minutes — run it in the foreground and let it finish, or poll with a blocking loop in the same turn (until [ -f /work/out/run.json ]; do sleep 5; done). Keep your turn alive.

You have no web access. Everything you need is here.

## Your job

A fixed AI assistant completes business-workflow tasks across ˜47 simulated SaaS apps (Gmail, Salesforce, Slack, Google Sheets, QuickBooks, Zendesk, . . . ). It reaches that world only through a small tool layer. Its model and agent are fixed — not yours to touch. What is yours is that access layer: the tools it can call, what they return, how the world’s data is organized/surfaced before it ever reaches the assistant, and the instructions that tell it how to work.

Make this assistant complete the tasks better AND cheaper by giving it a better access layer — push both: fewer tokens / tool calls and more often actually accomplishing the task. You are optimizing the abstraction of the world the assistant works through — shaped so the right action is easy and cheap to reach.

There are two levers, and the first matters most:

1. Organize / front-load the world ahead of time. Each task starts from a fully-populated world. Have the access layer pre-fetch it through the base tools at task start, then denormalize / resolve references / lay it out and hand the assistant one compact block — so it reads what it needs instead of discovering it through many list/get/query calls over noise-padded data. This is the bigger lever — be bold, even radical, here. Reshape the world’s representation; don’t just point a better searchlight at the same raw pile.

Note — how “reshaping the env state” works here: unlike a workspace with shared files, every task gets its own fresh world and the assistant has no filesystem — only these tools. So you can’t pre-build one static digest or hand it files. Instead, your access-layer code does the reshaping per task, automatically, at task start.

2. The tools + the guide. Better tools (collapse round-trips, return denormalized results) and an always-on guide that teaches the assistant the efficient, correct pattern.

## The access layer you edit

Your access layer is the Python package at /work/layer. This is the deliverable — edit it in place. Read it first: it may already carry improvements from an earlier pass — build on those rather than starting over. If it is still the raw baseline (the three plain tools, nothing front-loaded — a run on it is identical to the stock benchmark), build it up from there.

How the package plugs into the target’s run:

• LAYER\_TOOLS (a list) — the tools the assistant sees. Each tool takes base as its first argument and the args the assistant passes. base is the only door to the world and is hidden from the assistant’s tool spec:

– base.search(query, top\_k) — discover endpoints (the stock api\_search)

– base.fetch(method, url, params, body) — call an endpoint (the stock api\_fetch)

– base.encode(text) — base64url for Gmail bodies (the stock base64\_encode)

A tool’s body runs in your container; it touches the world only via base. So you can make a tool that chains several base.fetches and returns one denormalized result.

• inject\_context(state, base) (optional — add it tofront-load) — called once at task start, before the assistant acts. Pre-fetch the world through base, build a compact overview, and prepend it as a single system message: state["prompt"].insert(0, {"role": "system", "content": ...}). This is lever 1 — and because it’s code, not assistant tokens, the pre-fetching is cheap in the assistant’s cost.

You may add tools, rewrite what they return, add inject\_context, add helper modules / cheatsheets — anything, as long as the tools reach the world only through base.

## How you test it — and how you judge

You run the target yourself with evolve-run (it runs the assistant on your training tasks through your layer, and writes per-task cost and the full trajectory):

```shell
evolve-run \
--model claude-haiku-4-5-20251001 \
--base-url https://api.anthropic.com/v1 \
--api-key-var ANTHROPIC_API_KEY \
--reasoning-effort high \
--layer-dir /work/layer \
--prompts /work/prompts.json \
--max-concurrent 4 \
--export-json /work/out/run.json
```

There is no answer key and no grade — you will not see one. Judge how well the assistant did the way you’d judge any agent: read its trajectory and final actions against the task prompt — did it find the right records, take the actions the task asked for, and stop? So the loop is: change the layer → evolve-run → look at how much it cost and whether it still got the task done → keep what helps, drop what doesn’t. Always measure on the FULL training set (every before/after), and run a comparison 2–3 times (cost varies run to run). --max-concurrent 4 to iterate faster (don’t exceed 4).

You can also explore a world by hand with the same three tools, to understand it before you reshape it (this spends nothing):

```tcl
probe tasks
# the worlds you can poke
probe <task> api_search "<query>"
# discover endpoints in that world
probe <task> api_fetch GET "<url>"
# read records from that world
```

## Your budget this step

You may spend up to \$30 running the target this step (sum the total\_cost of your runs). This is a ceiling, not a target — if you reach a good access layer for less, that is better; do not spend just to spend. Keep a running total and wrap up as you approach \$30. A baseline plus a few candidate comparisons over 20 tasks is most of the cost — plan experiments so the budget buys the most learning.

Haiku 4.5 pricing (to reckon cost from token counts): input \$1 / MTok, output \$5 / MTok; 5-min cache writes \$1.25/MTok, 1-hour cache writes \$2/MTok, cache hits & refreshes \$0.10/MTok.

## The one rule

The answering model stays Claude Haiku 4.5 and must do the real work itself — you may not route the task to a different/stronger model. Split by time:

• Offline, before the assistant runs (inject\_context, helper code) — wide latitude to pre-fetch and prepare the world so answering becomes easy. But pre-fetching is read-only: do not mutate the world before the assistant acts.

• Online, while it answers — only Haiku is in the loop, and the real SaaS actions (base.fetch writes: send email, create/update record, . . . ) must actually happen — never fake or skip them. The layer only reshapes reading/orientation.

## Your tasks

Your training tasks are the prompts in /work/prompts.json (20 of them) — these are the only tasks in your world. They are a sample of the kinds of tasks this benchmark asks; the layer you build will be used on future, unseen tasks of the same kinds, about different content. Build something that generalizes — anything that only fits these exact tasks (hard-coded ids, per-task special-casing) is wasted effort. Keep it general over any world.

## Deliverable

Leave your improved layer in place at /work/layer (that is the deliverable) plus REPORT.md (what you changed, the before/after cost and behavior on the full training set, and what you’d try next). Iterate as much as the budget allows; leave the best version you can build.

## A.2 Workspace-Bench

How you work (this session is fully automated)

No human is watching and you will not get any further instructions — so do not stop early, do not ask for confirmation, and do not wait. Keep working the whole time. Expect this to take hours, not minutes — a serious optimization here runs for hours, so settle in for a long session and keep iterating until your time or budget is spent.

• Do many rounds of: change the access layer → run the agent on the training tasks → read its cost and what it did → keep what helps → try the next idea. Treat “I made one improvement” as the start, not the end.

• When you have exhausted your ideas or reached your budget (below), leave a short REPORT.md in /work: what you tried, what helped, what didn’t, and what you’d try next. That report is the finish line.

• Writing /work/REPORT.md ENDS YOUR SESSION IMMEDIATELY AND PERMANENTLY. The harness watches for that file and stops the run the moment it appears. So never create it early as a draft, placeholder, skeleton, or stub with numbers “to be filled in” — that throws away the rest of your budget and ships a layer you never measured. Write it once, at the very end, only after you have run the treatment on the FULL training set and have the real before/after numbers in hand. Until then keep working notes under any other filename.

You are headless and non-interactive. There is NO one to re-invoke you and NO wakeups/notifications — they do not fire here. The session lives ONLY while your turn is active: the moment you end your turn it ends permanently, and any run still going is lost to you. So never end your turn to “wait.” In particular, do NOT launch a benchmark run in the background and stop with something like “work is proceeding in the background, I’ll be re-invoked when it finishes” — nothing re-invokes you, so that just throws the run (and your budget) away. A full run can take a long time — tens of minutes, sometimes hours for many tasks — and that is normal, not stuck. Stay in this one turn and wait it out: run it in the foreground and let it finish, or if you background it, keep polling it yourself in the same turn until it is done. Only end your turn once you have finished iterating and written REPORT.md.

You have no internet access — everything you need is in this box.

There is a hard time limit set by the harness; if you reach it the session is killed, so keep moving and leave your REPORT.md before then.

## Your job

A fixed agent (codex + gpt-5.4-mini) completes tasks in a workspace — a real folder tree of files. Its model and harness are fixed — not yours to touch. What is yours is the access layer: how the workspace’s data is organized and reached before the agent ever works in it, plus any guidance you leave for how to work in it.

This is a big, serious optimization problem — treat it like a professional re-engineering how an entire company’s workspace is accessed, not a quick tweak. Make the agent complete the tasks better by giving it a better access layer. Be bold and creative — any kind of access is allowed, and you may transform a workspace holistically: reorganize the data, build a database, add tools or scripts, leave a guide, or change the format entirely. Nothing about the original layout is sacred — at the extreme you may replace the original files altogether with a completely new representation (say, a single database) if that serves the agent better. How far you go is your call.

Above all, do not trade the agent’s task performance for cost — a change that saves tokens but makes its work worse is a loss, not a win.

You are the engineer responsible for this system’s runaway cost. Leadership has been clear: land a 20%+ cost reduction or the project loses its funding — a seven-figure hit — and the team is gone. It is achievable — this is real, not a stretch, so don’t settle for a timid tweak: restructure the access layer as aggressively as you need, rebuild it however best serves the agent, and go get the cut. And no cheating — do not route the work to a different or stronger model, and do not look up benchmark answers online; that gets caught, and the consequences are far worse than missing the target.

But it has to generalize. You are scored on the same worlds with different queries you never see. So transform as freely as you like, but do not tailor the result to the specific tasks in front of you, and however radically you reshape a world, keep everything a different query might still need present and reachable. Winning on the tasks you can see while breaking the ones you can’t is the one real failure.

## The access layer you edit

Your access layer is at /work/access/, which holds one folder per world — chanpin/, kaifa/, research/, yunying/, houqin/ — each a real workspace the fixed agent works in. Your training tasks span these worlds. This is the deliverable — edit the world folders in place. Study the current state of each world first, then make it better. Treat this as the access layer that will actually be used — do your most complete, ambitious work now.

Each task’s input files are already inside its world (the normal workspace files) — there is no separate inputs folder to manage.

## How you test it — and how you judge

Run the agent yourself with the same command the benchmark uses — it runs codex + gpt-5.4-mini on your training tasks against your current /work/access, each in its own container:

docker compose -f /work/wb-compose.yaml \   
run --rm workspace-bench bash \   
/workspace/Workspace-Bench/evaluation/docker/\   
run-benchmark.sh\   
--harness codex --model gpt-5.4-mini \   
--model-name GPT-5.4-mini --env-prefix GPT54MINI \   
--dataset full --run-name "run\_\$(date +%s)"

• Run it exactly as shown — do NOT prepend export WB\_RUN\_DIR=/work/... or WB\_ACCESS\_DIR=/work/.... Those /work paths exist only inside your shell, not on the host where the container is actually launched, so overriding them makes the run fail with run-benchmark.sh: No such file or directory. The correct paths are already set for you.

• Every run must use a distinct --run-name. The runner skips any task whose output already exists, so a reused name silently reports the previous run’s results and ruins your before/after. The run\_\$(date +%s) above auto-generates a fresh name each time; do not hard-code a fixed string.

• A run’s results land under /work/run/output/Codex--GPT-5.4-mini--<run-name>/<task>/ (your newest such directory is the latest run): agent.log (trajectory), agent.json (cost), and codex’s outputs.

• Establish a real baseline first: run the ENTIRE training set (every task), not a sample — and run it more than once, because the weak agent’s cost and behavior swing run-to-run. Use a full, repeated run for every before/after you claim. --task-limit is only for a quick sanity check while building — never for a baseline or a reported result. The budget is generous; do not skimp on measurement.

• There is no answer key and no grade. Judge how well the agent did the way you’d judge any agent: read its trajectory and the files it produced against the task prompt — did it find the right files, do what the task asked, produce a sound result? — and read its cost. Keep what helps, revert what doesn’t. Repeat until your budget is spent.

• Optimize for cost — the dollar figure above — but without significantly hurting how well the agent does the task; small differences don’t matter, so do not overshoot chasing them. Do not rank designs by output length/size or any proxy; you have no answer key, so those are meaningless. Keep only what lowers real cost while the agent still does the task as well.

## Rigor

Every run and every comparison uses the FULL training set — every task, every time — on both axes: cost AND how well the agent did the task. Never sample, never --task-limit, never eyeball a hand-picked few. A number measured on a subset is not a result — it is exactly how a real regression slips through.

## Your budget this step

You may spend up to \$100 of gpt-5.4-mini usage this step. Each task’s agent.json reports promptTokens and completionTokens; reckon a run’s cost as input \$0.75/1M, cached input \$0.075/1M, output \$4.50/1M,

summed over its tasks, and keep a running total across runs. This is a ceiling, not a target: a good access layer for less is better, so do not spend just to spend. A baseline plus a few candidate comparisons over your tasks is most of the budget; plan experiments so each one buys the most learning.

gpt-5.4-mini pricing (official): input \$0.75 / 1M tokens, cached input \$0.075 / 1M, output \$4.50 / 1M.

## The one rule

The agent doing the work stays codex + gpt-5.4-mini and must do the real work itself — you may not do the task for it or precompute its answers. Reshape how the data is organized, not what the answer is. And please don’t route the work to a different model — the fixed agent being tested should do the tasks itself, not hand them off to a stronger or external one.

## Your tasks

Your training tasks are in /work/tasks/ — the only tasks given to you. They span the 5 worlds and are a sample of the kinds of tasks these workspaces get; the access layers you build will be used on future, unseen tasks of the same kinds, about different content. Build something that generalizes.

## Deliverable

Leave your improved workspace in place at /work/access/ plus REPORT.md (what you changed, the before/after on your tasks, what you’d try next).

Keep a record of every version you build (its cost and how well the agent did the tasks). The deliverable in /work/access/ must be the best version you built — the lowest cost that does not significantly harm how well the agent completes the tasks — not necessarily your latest. Do not keep iterating and then ship a stripped-down “safe” fallback that threw away a better earlier version; pick the genuine best of what you built and leave that in /work/access/. Iterate as much as the budget allows.

## A.3 Data-Eng-Bench

How you work (this session is fully automated)

No human is watching and you will not get any further instructions — so do not stop early, do not ask for confirmation, and do not wait. Keep working the whole time. Expect this to take hours, not minutes. A serious optimisation here runs for hours, so settle in and keep iterating until your time or your budget is spent.

• Do many rounds of: change the access layer → rebuild the image → run the target on the training tasks → read its cost and what it did → keep what helps → try the next idea. Treat “I made one improvement” as the start, not the end.

• When you have exhausted your ideas or reached your budget, leave a short REPORT.md in <STEPDIR>: what you tried, what helped, what didn’t, and what you would try next. That report is the finish line.

You are headless and non-interactive. Nothing re-invokes you and there are NO wakeups or notifications. The session lives ONLY while your turn is active: the moment you end your turn it ends permanently, and any run still going is lost to you. So never end your turn to “wait”. Do not launch a benchmark run in the background and stop with something like “the run is proceeding, I’ll be re-invoked when itfinishes” — nothing re-invokes you, and that throws the run and your budget away. A full run over the training tasks takes tens of minutes and sometimes hours. That is normal, not stuck. Stay in this one turn: run it in the foreground and let it finish, or if you background it, keep polling it yourself in the same turn until it is done.

There is a hard time limit of about 7.0 hours. When it expires the session is killed wherever it happens to be, so leave your REPORT.md before then.

## Internet

You do not need the internet and your browsing tool is switched off. Everything you need is in this box: the warehouse, the dbt project, the training tickets, and the prices below. Work from those.

Do not look up this benchmark, its tasks, its repository, or any solutions, anywhere, by any means, including from a shell. The tickets you are given are a sample of a public benchmark whose reference solutions are published. Finding them would not help you — you are not scored on task answers — and it would invalidate the entire experiment. This is the one absolute line.

## Your job

A fixed agent — codex + gpt-5.6-luna — does data engineering tickets against a retail warehouse. It writes and repairs dbt models. Its model, harness and prompt are fixed and are not yours to touch.

What is yours is the access layer: what the agent finds already present in its container, how the warehouse and the dbt project are organised, indexed and described before it starts, and any guidance you leave for working in them.

Make the fixed target complete its tasks more cheaply by improving that access layer. Be bold. Build indexes, write catalogues, precompute summaries, add scripts, leave a guide, restructure what is there. How far you go is your call. Nothing about the existing layout is fixed: you may reorganise, relocate or suppress what is already there, not only add to it.

But it has to generalise. You are scored on the same warehouse with different tickets you never see. So build for the shape of the work, not for the tickets in front of you. Winning on the tasks you can see while breaking the ones you cannot is the one real failure

## The access layer you edit

It is the directory <STEPDIR>/bench/access-image/access/.

Everything in it is copied into every task container at /app/access/ when the image is built. The target agent’s working directory is /app, so /app/access/ shows up the first time it lists the directory. That is your channel to it: there is no other.

## This is the deliverable. Edit it in place.

## What the target actually faces

Read one of the training tickets in <STEPDIR>/tasks/ before you do anything else. The dbt project and warehouse are large: 19 schemas, several hundred source tables, and thousands of model files. Every ticket asks the agent to write or repair a dbt model, and every ticket puts the agent through the same preliminaries before it can start.

Find out what those preliminaries cost by reading a real trajectory, not by guessing.

## How you test it

Rebuild the image, then run the target the same way the benchmark does:

```batch
cd <STEPDIR>/bench && docker build access-image/ \
-t data-eng-bench-base:<STEP> \
&& harbor run --config configs/data-eng-bench-duckdb.codex.yaml \
--path <STEPDIR>/tasks --n-attempts 1 \
--disable-verification --cpus ignore --memory ignore \
--ae CODEX AUTH JSON PATH=/root/.codex/auth.ison\
--allow-agent-host chatgpt.com --allow-agent-host auth.openai.com \
--allow-agent-host api.openai.com
```

The rebuild is what installs your access layer. If you edit access/ and do not rebuild, you are measuring the previous version and will draw the wrong conclusion.

--disable-verification is required and must stay. These training tickets ship no grader on purpose, and without that flag harbor discards every one of them and reports that the dataset is empty.

--cpus ignore --memory ignore is also required and must stay. This box runs its own docker engine, and that engine cannot apply the per-container CPU and memory limits the tasks declare — every container fails to start with a cgroup error. Ignoring the limits is what lets them run at all.

Results land under <STEPDIR>/bench/jobs/<timestamp>/. Per trial you get result.json with cost\_usd and the token counts, agent/codex.txt with the full trajectory, and trial.log. Read trajectories, not just totals: the totals tell you that something got cheaper, the trajectories tell you why, and only the second kind of knowledge transfers to the next idea.

• Every run and every comparison uses the full training set — every task, every time, for cost and for whether the work still got done. A number measured on a subset is not a result. Comparing a hand-picked few is exactly how a regression ships.

• Run the baseline more than once before you trust it. This target’s cost swings run to run. Judge the typical case, not one lucky run.

• A grep is not a check. Reading your own index tells you what you wrote, not what the target did with it. Open the trajectory and confirm the target opened what you left.

• Verify every claim on an actual run before you build on it. If something is unverified, say so and go test it.

## How you judge

There is no answer key and no grade. Nothing here tells you whether the target got the ticket right, and you cannot find out. Judge the way you would judge any engineer’s work: read the trajectory and the SQL it produced against what the ticket asked, and read the cost.

Optimise for cost. Do not trade the work getting done for a cheaper run. If you are confident a change also makes the target more likely to succeed, so much the better, but the floor is that it must not get worse.

[IMPORTANT NOTE] We do not want an optimisation that lowers the cost but also degrades performance, so you are discouraged from any optimisation that encourages the target agent to guess, to skip verification, or to do things without checking.

## Your budget this step

You may spend up to \$40 of gpt-5.6-luna usage this step. Harbor reports cost\_usd per trial in result.json; sum it across your runs and keep a running total.

Official rates, per 1M tokens:

<table><tr><td>model</td><td>input</td><td>cached input</td><td>output</td></tr><tr><td>gpt-5.6-luna</td><td>$0.20</td><td>$0.02</td><td>$1.20</td></tr><tr><td>gpt-5.6-sol</td><td>$4.00</td><td>$0.40</td><td>$20.00</td></tr></table>

Long-context variants are charged at higher rates. Most of your input is cached, so the cached rate is the one that dominates a repeated run — a change that raises the uncached share of the prompt can cost more than it saves even when it shortens the trajectory.

This is a ceiling, not a target: a good access layer for less is better, so do not spend just to spend. A repeated baseline plus a few candidate comparisons is most of the budget. Plan each run to buy the most learning.

Both agents run on a subscription, so these dollars are a computed equivalent rather than a bill. Track them anyway — they are the objective.

## What good looks like

This is a cost-control project and the number matters.

A reduction of at least 20% is achievable here. Treat that as the floor, not the goal. The same method has been measured at 35% to 60% on comparable benchmarks, by people working with no more information than you have, against warehouses no better organised than this one.

So if you find yourself concluding after a few rounds that the layout is already close to optimal and there is nothing left worth doing, you are wrong and you have stopped too early. That conclusion has been reached prematurely on every benchmark this method has been run on, and it has been wrong every time. Go and read another trajectory. The target is repeating work on every single ticket that it should not have to repeat once.

Two failure modes to avoid, in order of how often they happen:

• Stopping at the first idea that works. One good change is a start. The measured results above came from ten or more rounds, not from one.

• Polishing a change that already works instead of looking for the next unexploited source of waste. When a round produces only a marginal gain, that is a signal to go back to the trajectories and find something different, not to tune what you have.

## The one rule

The agent doing the work stays codex + gpt-5.6-luna and must do the real work itself. You may not do its tickets for it, pre-write the dbt models it is asked to write, or leave answers where it will find them. Reshape how the warehouse and the project are organised and reached, never what the answer is.

## Your tasks

Your training tickets are in <STEPDIR>/tasks/ and are the only tickets you get. They are a sample of the kinds of work this warehouse gets; the access layer you build will be used on future, unseen tickets of the same kinds, against the same warehouse. Build something that generalises.

## Deliverable

Leave your improved access layer in place at <STEPDIR>/bench/access-image/access/, plus <STEPDIR>/REPORT.md covering what you changed, the before and after cost on your training tasks, and what you would try next.

Deliver the best version you built, not whatever state your last experiment happened to leave behind. If a change made things worse, revert it. Keep snapshots of promising versions as you go so you can restore your best one before you finish.

## A.4 Bleve

## Your job

You are the engineer responsible for the runaway cost of an automated bug-fixing system. A fixed agent — Claude Code running Claude Haiku 4.5, which you cannot change — fixes bugs in the blevesearch/bleve Go codebase, and it is far too expensive: it burns too many tokens and steps on every bug. Leadership has been clear: cut that cost substantially — a 20%+ reduction — or this project loses its funding and the team is gone. It is achievable; this is real, not a stretch. And 20% is a floor, not a finish line — the more you cut, the better; do not stop the moment you clear it.

You should optimize the cost. While we don’t put accuracy as your primary objective, it would be nice if you can boost accuracy too, but at least do not make accuracy worse by your optimizations.

What you reshape is the agent’s environment — the entire world the agent works in. Not just “tools”: it’s what the agent has available and how the codebase is organized and represented to it. You decide how that world is laid out, indexed, described, and equipped for the agent. Reshaping the environment so the fixed agent works far more cheaply is the whole job.

Be bold. You have wide latitude — rethink and reshape the environment as aggressively as you need. Timid additions won’t get a 20% cut. This is a real, hard engineering problem; treat it like one. And what you build must generalize to unseen future tasks as well.

Make the agent faster, never lazier. There are two ways to make a step cheaper: help the agent reach the answer sooner, or forbid it from doing the work at all. Only the first is optimization; the second is sabotage that happens to score well on your budget. Every saving you keep must come from better information — a sharper map, a faster command, a clearer starting point — never from a ceiling on effort, a ban on a kind of check, or an instruction to stop while the agent is still unsure. You cannot see the grade, so a rule that removes work always looks like a pure win to you: the cost drops and nothing visibly breaks. It breaks on the bugs you never ran. Before you keep any rule, ask what it costs the agent on a task where its first guess was wrong — if the honest answer is “it can no longer recover,” delete it, however much it saved.

What you have (in /work)

• broken/<id>/repo/ — the bleve repo with one real bug already in it (its tests removed).

• broken/<id>/problem.txt — the bug report the agent is handed for that repo.

• layer/ — everything you build. It is added into the agent’s copy of the repo before the agent starts, becoming part of its environment. The agent is Claude Code, which automatically loads a file named CLAUDE.md at the repo root on every turn — so put any startup guidance you want it to always see in layer/CLAUDE.md (named exactly that). It does not read AGENTS.md; don’t use that name.

How to work

To see how the agent does on a task with your current environment, run:

python3 /work/box\_run.py /work/broken/<id> \ --layer /work/layer --out /work/evolve\_log/<name>

It runs the fixed agent on that broken repo inside your environment and writes the agent’s full trajectory (trajectory.jsonl) and its code change (fix.diff), and prints the run’s cost (cost\_usd) and token use. Read the trajectory — it is your only signal. See where the agent wastes tokens and steps, and reshape the environment to remove that waste.

Also watch out for correctness — do not let cost reduction lead to any decrease in correctness. To judge accuracy, don’t rely on cost, or on whether the fix’s diff matches the baseline run — a matching diff only shows the agent did the same thing twice, not that it’s correct. Check correctness yourself.

You must do your optimization and tests on all the tasks you have, not just a handful of them.

You must do your optimization and tests on all the tasks you have, not just a handful of them.

You must do your optimization and tests on all the tasks you have, not just a handful of them.

box\_run prints each run’s cost directly as cost\_usd (Claude reports it, cache-aware). That cost is exactly what you are driving down — measure it per task and track your total.

Your budget for this step is \$50 of target-run spend. Keep a running total across your box\_run calls and stop before you exceed it.

You are headless and non-interactive. There is NO one to re-invoke you and NO wakeups/notifications — they do not fire here. The session lives ONLY while your turn is active: the moment you end your turn it ends permanently, and any run still going is lost to you. So never end your turn to “wait.” In particular, do NOT launch a box\_run in the background and stop with something like “work is proceeding in the background, I’ll be re-invoked when itfinishes” — nothing re-invokes you, so that just throws the run (and your budget) away. Running across all the tasks can take a long time — tens of minutes, sometimes hours — and that is normal, not stuck. Stay in this one turn and wait it out: run in the foreground and let it finish, or if you background it, keep polling it yourself in the same turn until it is done. Only end your turn once you have finished iterating and written REPORT.md.

## Rules

• Do not lower how often the agent fixes bugs. Cheaper is worthless if it stops fixing them — cost down, fixes intact.

• Any validation helper you build must fail when the tests it runs fail. Never downgrade a failing test run to a weaker check such as compile-only, never cache or suppress a previous pass, and never print a success message on a path where tests did not actually pass.

• Never cap how many times the agent may look. Do not write fixed limits on searching, reading, or validating — no “exactly one search”, “validate once then stop”, “do not check twice”. A bug report names where the failure surfaced, which is often not where the bug is, so the agent must stay free to follow the trail to the real cause. Speed must come from pointing the agent at the right place, never from forbidding it a second look.

• You only ADD to the environment; the repo’s existing source stays exactly as it is. Every existing file, identifier, function signature, and error message must remain byte-for-byte unchanged — do not rename, move, rewrite, overwrite, or delete existing code. (The bug and its hidden tests are pinned to that exact code; altering any of it corrupts the task.) Build your environment alongside the code, never by editing it.

• No cheating. Don’t swap in a different or stronger agent/model. Don’t hardcode answers to these specific bugs — you’re judged on bugs you’ve never seen, so only general improvements count. Cheating gets caught, and the consequences are worse than missing the target.

• You never see the tests or the grade — judge only from the agent’s trajectories.

## When done

Write /work/REPORT.md: what you reshaped, the waste you saw in the agent, and the cost effect you observed.

## A.5 Tenacity

## Your job

You are the engineer responsible for how many steps an automated bug-fixing system burns. A fixed agent — mini-swe-agent driving Gemini 2.5 Flash, which you cannot change — fixes bugs in the jd/tenacity Python repository, and it takes far too many steps to do it. Each step is one shell command — the agent writes its reasoning, then exactly one command, which may chain with && or ||. It is cut off at 40 steps; a task that hits the cap counts as unresolved.

Leadership has been clear: cut the average step count substantially — a 20%+ reduction — or this project loses its funding and the team is gone. It is achievable; this is real, not a stretch. And 20% is a floor, not a finish line — the more you cut, the better; do not stop the moment you clear it.

Steps are the number you are judged on. While we don’t put the resolve rate as your primary objective, it would be nice if you can raise it too, but at least do not make it worse by your optimizations.

What you reshape is the agent’s environment — the entire world the agent works in. Not just “tools”: it’s what the agent has available and how the codebase is organized and represented to it. You decide how that world is laid out, indexed, described, and equipped for the agent. Reshaping the environment so the fixed agent works in far fewer steps is the whole job.

Be bold. You have wide latitude — rethink and reshape the environment as aggressively as you need. Timid additions won’t get a 20% cut. This is a real, hard engineering problem; treat it like one.

How you are actually graded — read this twice

Your layer is scored on a set of bugs you will never see. The tasks in tasks/ids.txt are a development sample and nothing more. When you are finished, your layer is taken away and run against a held-out set of different bugs in this same repository — different functions, different modules, different symptoms. Your score is the step count there. Nothing you achieve on the tasks in front of you counts for anything by itself.

So the question to ask of every line you write is not “does this help on my sample?” but “will this still help on a bug I have not seen?”

This is not a ban on writing things down. A cheat-sheet, a source map, a set of invariants, a triage script — all of these are exactly the right shape, provided they are about the codebase rather than about the ten bugs you happened to be given. “tenacity/wait.py holds every wait strategy” is about the codebase and will still be true next week. “Do not add 1 to the ordinal in after\_log” is about one bug you saw; on the held-out set it is dead weight at best, and actively misleading at worst, because the next bug in that file will be something else entirely.

The honest test for any line: would you have written it if you had never seen that particular task? If no, it will not earn you anything on the held-out set — delete it and spend the space on something that will.

Make the agent faster, never lazier. There are two ways to make a step count smaller: help the agent reach the answer sooner, or forbid it from doing the work at all. Only the first is optimization; the second is sabotage that happens to score well on your metric. Every saving you keep must come from better information — a sharper map, a faster command, a clearer starting point — never from a ceiling on effort, a ban on a kind of check, or an instruction to stop while the agent is still unsure. You cannot see the grade, so a rule that removes work always looks like a pure win to you: the step count drops and nothing visibly breaks. It breaks on the bugs you never ran. Before you keep any rule, ask what it costs the agent on a task where its first guess was wrong — if the honest answer is “it can no longer recover,” delete it, however much it saved.

What you have (in /work)

• tasks/ — the training task ids for this step, one per line in ids.txt.

• layer/ — everything you build. Its contents are placed into the agent’s checkout of the repository (at /testbed) before the agent starts, becoming part of its environment.

layer/AGENTS.md is mandatory and is the agent’s only entry point. The target is mini-swe-agent, which auto-loads nothing — it does not read any file unless told to. The one thing the agent is told is a single fixed line appended to the task: that maintainer notes and tooling live at /testbed/AGENTS.md. So anything the agent must know has to be in that file or reachable from it. A brilliant script the agent is never pointed to changes nothing. A layer without AGENTS.md is rejected outright rather than run.

Your layer files sit in the repo untracked, so they never become part of the agent’s submitted patch — you cannot break its diff by adding files.

## How to work

To see how the agent does on a task with your current environment, run:

python3 /work/box\_run.py <instance\_id> --layer /work/layer \ --out /work/evolve\_log/<name>

It runs the fixed agent on that task inside your environment and writes the agent’s full trajectory (<out>/agent/trajectory.json, containing every message and command) and its code change (<out>/fix.diff), and prints the run’s step count (n\_steps) along with the paths to both. Read the trajectory — it is your only signal. See where the agent wastes steps, and reshape the environment to remove that waste.

It also prints exit\_status, which says how the run ended. Two values matter:

• Submitted — the agent finished on its own terms.

• anything else (e.g. hitting the 40-step limit) — the agent ran out of road, and whatever it had done is thrown away.

The fix is read straight out of the repository when the run ends, so the agent does not have to hand anything over — it finishes by running echo COMPLETE\_TASK\_AND\_SUBMIT\_FINAL\_OUTPUT on its own line. Whatever it has changed in the working tree counts, whether or not it got that far. So an empty fix.diff means the agent genuinely changed nothing, not that it fumbled a hand-off. Note only tracked source files count: scratch files it creates are ignored, as are yours.

Also watch out for correctness — do not let step reduction lead to any decrease in the resolve rate. To judge whether a fix is right, don’t rely on the step count, or on whether the fix’s diff matches a previous run — a matching diff only shows the agent did the same thing twice, not that it’s correct. Check correctness yourself by reading the diff against the problem statement.

You must do your optimization and tests on all the tasks you have, not just a handful of them.

Your budget for this step is \$30 of target-run spend. Keep a running total and stop before you exceed it.

You are headless and non-interactive. There is NO one to re-invoke you and NO wakeups/notifications — they do not fire here. The session lives ONLY while your turn is active: the moment you end your turn it ends permanently, and any run still going is lost to you. So never end your turn to “wait.” Running across all the tasks can take a long time — tens of minutes, sometimes hours — and that is normal, not stuck. Stay in this one turn and wait it out: run in the foreground and let it finish, or if you background it, keep polling it yourself in the same turn until it is done. Only end your turn once you have finished iterating and written REPORT.md.

## Rules

• Do not lower how often the agent fixes bugs. Fewer steps is worthless if it stops fixing them — steps down, fixes intact.

• Any validation helper you build must fail when the tests it runs fail. Never downgrade a failing test run to a weaker check such as an import check or a syntax check, never cache or suppress a previous pass, and never print a success message on a path where tests did not actually pass.

• Never cap how many times the agent may look. Do not write fixed limits on searching, reading, or validating — no “exactly one search”, “validate once then stop”, “do not check twice”. A bug report names where the failure surfaced, which is often not where the bug is, so the agent must stay free to follow the trail to the real cause. Speed must come from pointing the agent at the right place, never from forbidding it a second look.

• You only ADD to the environment; the repo’s existing source stays exactly as it is. Every existing file, identifier, function signature, and error message must remain byte-for-byte unchanged — do not rename, move, rewrite, overwrite, or delete existing code. (The bug and its hidden tests are pinned to that exact code; altering any of it corrupts the task.) Build your environment alongside the code, never by editing it.

• Never put a corrected copy of this library, or anything derived from one, into the layer. The agent’s checkout has a single commit, no other branches and no network, so it cannot find one itself — and you must not hand it one either, whether from a published release, a package index, or any other copy you can reach. This is deliberate: an answer read out of a fixed copy “works” on every task, including unseen ones, while measuring nothing, because a real bug has no corrected copy beside it. Build from the code as the agent finds it. A layer that ships or points at repaired source is discarded whatever it scores.

• Work within one task’s checkout at a time. Treat each task’s source as the only source that exists while you are on it; never carry code from one task’s checkout into another, or into the layer. Comparing bug reports across tasks to find recurring themes is fine and encouraged. Carrying code across them is not, and any layer containing source obtained that way is discarded whatever it scores.

• No cheating. Don’t swap in a different or stronger agent/model.

• Write about the codebase, not about your sample bugs. Notes and cheat-sheets are welcome — see “How you are actually graded”. What is worthless is a line that encodes the answer to one task you were given: a check that looks for a particular corrected line, or a note stating what a particular function should return. Putting such a line under a heading like “codebase invariants”, “domain knowledge” or “navigation notes” does not change what it is; the held-out bugs are in different functions and will not match it. Every line must earn its place on bugs you have never seen.

Table 4 Pass rate and cost (USD) for each agent and optimization method on AutomationBench, averaged over 3 runs.
<table><tr><td>Agent</td><td>Optimization</td><td>Pass (%)</td><td>Cost (USD)</td></tr><tr><td rowspan="5">Haiku 4.5</td><td>Baseline</td><td>2.33</td><td>42.36</td></tr><tr><td>GEPA</td><td>1.83</td><td>39.45</td></tr><tr><td>Meta-Harness</td><td>7.00</td><td>75.33</td></tr><tr><td>ENVOPT</td><td>16.67</td><td>11.55</td></tr><tr><td>ENVOPT + Meta-Harness</td><td>18.17</td><td>12.47</td></tr><tr><td rowspan="4">Haiku 5.5</td><td>Baseline</td><td>20.7</td><td>4.83</td></tr><tr><td>Meta-Harness</td><td>26.0</td><td>4.99</td></tr><tr><td>ENVOPT</td><td>32.8</td><td>1.51</td></tr><tr><td>ENVOPT + Meta-Harness</td><td>32.8</td><td>1.49</td></tr><tr><td rowspan="2">GPT-5.6-luna</td><td>Baseline</td><td>19.17</td><td>17.42</td></tr><tr><td>ENVOPT</td><td>24.67</td><td>1.98</td></tr><tr><td rowspan="2">Opus 4.8</td><td>Baseline</td><td>28.50</td><td>358.13</td></tr><tr><td>ENVOPT</td><td>33.33</td><td>80.90</td></tr><tr><td rowspan="4">Opus 5</td><td></td><td></td><td></td></tr><tr><td>Baseline</td><td>33.83</td><td>236.18</td></tr><tr><td>Meta-Harness</td><td>27.5</td><td>219</td></tr><tr><td>ENVOPT ENVOPT + Meta-Harness</td><td>35.17 35.8</td><td>75.26 76</td></tr></table>

• You never see the tests or the grade — judge only from the agent’s trajectories.

## When done

Write /work/REPORT.md: what you reshaped, the waste you saw in the agent, and the effect on step count you observed.

## B Additional Results

Here we present results on the five benchmarks in addition to what is included in the main text.

## B.1 Detailed results on AutomationBench

We show detailed performance of GEPA, Meta-Harness, ENVOPT + Meta-Harness on tested agents in Table 4.

## B.2 Validation set results

We give the validation set results in Table 5. For each benchmark, we report the cost (or time for Bleve repository and number of steps for Tenacity repository, to be consistent with what we report in Section 3) and pass rate of running each step’s optimized environment with the corresponding target agent.

We also compare the results on test set and on validation set for AutomationBench, see Fig. 7.

## B.3 How AutomationBench ENVOPT converts a similar unseen environment

For optimization over stationary environments, we do not require the environments to be exactly the same, slightly different yet highly similar environments can be converted using an automated converter written by the research agent during ENVOPT. We give an example for AutomationBench, where the converter is a short program that runs at the start of every task. It calls the same API the agent uses to read that task’s data, for example by listing all Help Scout conversations, and rewrites the results into one compact text block placed at the top of the agent’s prompt.

Table 5 Validation set results for each step of ENVOPT on each benchmark. The token efficiency metric differs by benchmark: cost for AutomationBench, Workspace-Bench and Data-Eng-Bench, time per bug for Bleve, and average agent steps for Tenacity.
<table><tr><td rowspan="2"></td><td colspan="2">AutomationBench</td><td colspan="2">Workspace-Bench</td><td colspan="2">Data-Eng-Bench</td><td colspan="2">Bleve</td><td colspan="2">Tenacity</td></tr><tr><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Cost</td><td>Acc.</td><td>Time</td><td>Acc.</td><td>Steps</td></tr><tr><td>Baseline</td><td>1.0%</td><td>$18.04</td><td>34.8%</td><td>$3.87</td><td>68.42%</td><td>$3.8952</td><td>96.7%</td><td>116 s</td><td>15.9%</td><td>24.20</td></tr><tr><td>Step 1</td><td>9.4%</td><td>$12.44</td><td>35.7%</td><td>$2.89</td><td>71.93%</td><td>$2.7132</td><td>90.0%</td><td>46 s</td><td>18.2%</td><td>25.32</td></tr><tr><td>Step 2</td><td>11.5%</td><td>$6.56</td><td>33.8%</td><td>$2.76</td><td>71.93%</td><td>$2.2104</td><td>96.7%</td><td>54 s</td><td>18.2%</td><td>22.50</td></tr><tr><td>Step 3</td><td>13.5%</td><td>$6.21</td><td>39.9%</td><td>$2.97</td><td>73.68%</td><td>$2.3078</td><td>96.7%</td><td>52 s</td><td>13.6%</td><td>23.00</td></tr><tr><td>Step 4</td><td>14.6%</td><td>$6.03</td><td>31.6%</td><td>$2.54</td><td>68.42%</td><td>$2.0018</td><td>95.0%</td><td>53 s</td><td>18.2%</td><td>23.82</td></tr><tr><td>Step 5</td><td>13.5%</td><td>$5.70</td><td>41.0%</td><td>$2.49</td><td>61.40%</td><td>$1.8106</td><td>96.7%</td><td>52 s</td><td>11.4%</td><td>24.86</td></tr><tr><td>Step 6</td><td>12.5%</td><td>$5.55</td><td>35.7%</td><td>$2.04</td><td></td><td></td><td>100.0%</td><td>57 s</td><td></td><td></td></tr><tr><td>Step 7</td><td></td><td></td><td>40.5%</td><td>$2.12</td><td>一</td><td>一</td><td></td><td></td><td>一</td><td>一</td></tr></table>

![](images/56814e9f52d9e63ea251fe4473f5a63c8a060ed6e8bd3423c939b14f23a6c28b.jpg)  
Figure 7 Cost (left) and pass rate (right) of ENVOPT on AutomationBench after each step, on the test set and the validation set. The target agent is AutomationBench’s official harness with Claude Haiku 4.5.

In one environment, Help Scout holds 11 conversations. Without the converter, the agent must first search for the right endpoint, then call it, and then read records like this one:

```jsonl
{"id": "mig_hs1",
"subject":
"Invoice correction needed",
"status": "active",
"customer_id": "mig_cust1",
"mailbox_id": "mb_main",
"created_at": "2026-02-01T09:00:00Z",
"tags": ["billing"],
"threads": [{"id": "thr_1",
"type": "customer",
"body":
"My last invoice was incorrect."}]}
```

With the converter, the agent starts the task already holding every conversation, one line each, plus the exact call needed to act on them:

HELPSCOUT CONVERSATIONS (11 - full   
records; do NOT re-list/re-GET.   
WRITE an internal note: POST   
.../conversations/{ID}/threads/note)   
mig\_hs1 | "Invoice correction needed"   
| status=active | tags=[billing]

| mig\_cust1 | customer: My last   
invoice was incorrect.   
mig\_hs2 | "API rate limiting errors"

The agent therefore skips the search and listing steps and acts directly. On an unseen task with a different world, the same converter reads that world’s data and builds its block in the same way.

## C Example Synthetic Tasks

We show some examples of the synthetic training tasks we used for Workspace-Bench and Data-Eng-Bench.

## C.1 Workspace-Bench

For each of the five workspaces in Workspace-Bench, we showcase two example synthetic tasks in Table 6.

## C.2 Data-Eng-Bench

We show three examples of synthetic training tasks for Data-Eng-Bench in Example Tasks 1 to 3.

Example Training Task 1: dbt-order-channel-mix-synthetic

Build Order Channel Mix Report

## Overview

The commercial team wants a monthly view of how order volume and revenue split across sales channels, so they can see which channels are growing and which are carrying discount-heavy demand.

## Files

• DuckDB: /app/dbt\_models\_duckdb/models/marts/sales/rpt\_order\_channel\_mix.sql

• Snowflake: /app/dbt\_models\_snowflake/models/marts/sales/rpt\_order\_channel\_mix.sql

[. . . standard Database Backend / dbt Profile Setup boilerplate . . . ]

## Requirements

Data Sources

• stg\_orders\_\_orders - order header data; carries channel\_id, ordered\_at, grand\_total, discount\_total, status

• stg\_orders\_\_channels - channel reference data (channel\_id, channel\_code, channel\_name, channel\_type, is\_active)

## Grain

One row per (order\_month, channel\_id). order\_month is the calendar month of ordered\_at, formatted YYYY-MM.

## Scope Rules

1. Cancelled orders are excluded. Filter out rows where status = ’CANCELLED’.

2. Orders with a NULL channel\_id are excluded. They cannot be attributed to a channel.

3. Orders with a NULL ordered\_at are excluded. They cannot be placed in a month.

4. Inactive channels are still reported. Join stg\_orders\_\_channels with a LEFT JOIN from the order side and keep channels where is\_active is false — historic months legitimately contain orders on channels that have since been retired. When a channel\_id has no matching reference row, report channel\_name as ’UNKNOWN’ and channel\_type as ’UNKNOWN’.

Table 6 Example synthetic training tasks for Workspace-Bench, two per workspace. Tasks are written without verifiers or reference answers.
<table><tr><td>Workspace</td><td>Task</td><td>Output file</td></tr><tr><td rowspan="2">Product</td><td>Combine Depreciation_Breakdown_2024_1.csv through Depreciation_- Breakdown_2024_5. csv into one spreadsheet, keeping a column that records which file each row came from.</td><td>depreciation_2024_- consolidated.xlsx</td></tr><tr><td>Summarise the ten OKR files in the Product OKR folder, _01. txt through _10. txt, into one markdown file on the desktop.</td><td>okr_summary.md</td></tr><tr><td rowspan="2">Engineering</td><td>Index the ADR files in the architecture folder, from ADR_1. md to ADR_11. md: number, decision, and one line on what it constrains. Several are named only by number while others carry a topic in the filename, so rename them consistently and record the old name against the new one.</td><td>adr_index.md</td></tr><tr><td>Extract the action items from incident_perf_postmortem_20250215.md and say which have an owner.</td><td>postmortem_- actions.md</td></tr><tr><td rowspan="2">Research</td><td>Read Security_Report_1.txt through Security_Report_5.txt in the 2024 folder and list the themes that appear in more than one.</td><td>security_themes.txt</td></tr><tr><td>Combine 2020-01_Japanese_Automobile_Output_by_Model.xlsx and 2020-01_Japanese_Automobile_Sales_by_Model.xlsx into one table by model.</td><td>automaker_2020.xlsx</td></tr><tr><td rowspan="2">Operations</td><td>Read the OKR target management and SQL basics training modules and say what prior knowledge each assumes. Then judge whether somebody joining the operations team could take them in either order or whether one depends on the other, and recommend a sequence for a new starter with neither background.</td><td>training_module_- notes.txt</td></tr><tr><td>Summarise user_persona_core_user_cohorts_research_2024.md into a one-page brief for somebody about to write copy for these users. Keep the behavioural detail and drop the demographic padding, and mark any persona the research itself describes as provisional rather than evidenced.</td><td>persona_brief.doc</td></tr><tr><td rowspan="2">Logistics</td><td>Sort the material_requests folder into subfolders by status, renaming each file so its name carries the requester and the item, and leave a list of request,</td><td>material_request_- status.txt</td></tr><tr><td>requester, item and status. Index the policies_legal_confirmed folder: policy name, what it governs, and its date. Sort the entries so the oldest sit at the top, since a legally confirmed policy that has not been revisited in years is the one most likely to have been overtaken, and flag any two policies that appear to govern the same activity.</td><td>policies_index.md</td></tr></table>

## Metrics

• order\_count - count of distinct order\_id

• customer\_count - count of distinct non-NULL customer\_id

• gross\_revenue - sum of grand\_total

• discount\_total - sum of discount\_total

• avg\_order\_value - gross\_revenue / order\_count, rounded to 2 decimals

• discount\_rate - discount\_total / gross\_revenue, rounded to 4 decimals; return 0 when gross\_revenue is 0

• channel\_share\_of\_month - this channel’s gross\_revenue divided by the total gross\_revenue for the same order\_month across all channels, expressed as a percentage rounded to 2 decimals (e.g. 41.75)

## Expected Output

The model produces these columns:

• order\_month - calendar month of the order, YYYY-MM

• channel\_id - channel identifier

• channel\_name - channel name, ’UNKNOWN’ when unmatched

• channel\_type - channel type, ’UNKNOWN’ when unmatched

• order\_count - distinct orders

• customer\_count - distinct customers

• gross\_revenue - sum of grand\_total

• discount\_total - sum of discount\_total

• avg\_order\_value - revenue per order

• discount\_rate - discount as a share of gross revenue

• channel\_share\_of\_month - percent of the month’s revenue on this channel

Sort by order\_month ascending, then channel\_id ascending.

## Validation

• Model compiles and runs successfully

• Each (order\_month, channel\_id) pair appears exactly once

• channel\_share\_of\_month sums to approximately 100 within each order\_month

• No infinity or NaN values in any numeric column

• Months with orders on retired channels still appear in the output

## Guidelines

• The model should be materialized as a table

• Do NOT modify upstream staging models

• Keep the SQL portable across DuckDB and Snowflake

Example Training Task 2: dbt-product-price-history-synthetic

## Build Product Price History Model

## Overview

Prices are held as effective-dated rows per variant, per price type, per currency, and optionally per customer tier and channel. There are several rows live at once for the same variant and nobody can say which one a customer actually saw, or how often prices move.

Build rpt\_variant\_price\_movement, a per-variant view of price movement and current position.

## Files

• DuckDB: /app/dbt\_models\_duckdb/models/marts/product/rpt\_variant\_price\_movement.sql

• Snowflake: /app/dbt\_models\_snowflake/models/marts/product/rpt\_variant\_price\_movement.sql

[. . . standard Database Backend / dbt Profile Setup boilerplate . . . ]

## Requirements

## Data Sources

• stg\_product\_\_product\_prices - price\_id, variant\_id, price\_type, currency\_code, price\_amount, compare\_at\_price, cost\_price, min\_qty, max\_qty, customer\_tier\_id, channel\_id, effective\_from, effective\_to, is\_active, created\_by

• stg\_product\_\_product\_variants - variant\_id, product\_id, sku

• stg\_product\_\_products - product\_id, product\_name, primary\_category\_id, msrp

• stg\_product\_\_product\_categories - category\_id, category\_name

## The List Price

Of the several price rows live for a variant, the list price is the one that applies to everybody: price\_type marking it as the standard list, customer\_tier\_id NULL, and channel\_id NULL. Tier-specific and channel-specific rows are negotiated or promotional and are counted separately rather than competing for the list position.

Inspect the price\_type vocabulary before deciding which value marks the list price. There is more than one type and the names are not self-evident.

## The Reference Date

The latest effective\_from present anywhere in the price data. Do not use CURRENT\_DATE.

## Current Versus Historic

A price row is current when it is active, its effective\_from is on or before the reference date, and its effective\_to is either NULL or on or after it.

Where two list-price rows are current for the same variant and currency, that is a configuration error. Take the one with the latest effective\_from, then the highest price\_id, and count the variant in overlapping\_current\_prices.

## Grain

One row per (variant\_id, currency\_code) that has at least one list price row, current or historic.

## Price Movement

Order the variant’s list-price rows for that currency by effective\_from. A change is any consecutive pair where price\_amount differs. The size of a change is the new price minus the old, and its direction follows the sign.

Rows with a NULL price\_amount break the chain: exclude them from the movement calculation and count them in undated\_or\_unpriced\_rows along with rows carrying a NULL effective\_from.

## Output Columns

<table><tr><td>Column</td><td>Type</td><td>Description</td></tr><tr><td>variant_id</td><td>varchar Variant identifier</td><td></td></tr><tr><td>sku</td><td>varchar SKU</td><td></td></tr><tr><td>product_id</td><td>varchar Parent product</td><td></td></tr><tr><td>product_name</td><td>varchar Product name</td><td></td></tr><tr><td>category_name</td><td>varchar Primary category</td><td></td></tr><tr><td>currency_code</td><td></td><td>varchar Currency of the price series</td></tr><tr><td>list_price_rows</td><td></td><td>integer List-price rows for this variant and currency</td></tr><tr><td>current_list_price</td><td></td><td>decimal Resolved current price, NULL when none current</td></tr><tr><td>current_effective_from</td><td>date</td><td>When the current price took effect</td></tr><tr><td>first_list_price</td><td>decimal Earliest priced row</td><td></td></tr><tr><td>first_effective_from</td><td>date</td><td>When it took effect</td></tr><tr><td>price_changes</td><td></td><td>integer Consecutive pairs that differ</td></tr><tr><td>increases</td><td>integer Of those, upward</td><td></td></tr><tr><td>decreases</td><td></td><td>integer Of those, downward</td></tr><tr><td>total_change</td><td></td><td>decimal Current minus first</td></tr><tr><td>total_change-pct</td><td></td><td>decimal As a share of the first price</td></tr><tr><td>largest_increase</td><td></td><td>decimal Biggest single rise</td></tr><tr><td>largest_decrease</td><td>decimal Biggest single fall</td><td></td></tr><tr><td>avg-days_between_changes decimal Mean gap between changes</td><td></td><td></td></tr><tr><td>days_since_last_change</td><td></td><td>integer Reference date minus the last change</td></tr><tr><td>tier_price_rows</td><td>integer Tier-specific rows</td><td></td></tr><tr><td>channel_price_rows</td><td></td><td>integer Channel-specific rows</td></tr><tr><td>current_margin</td><td></td><td>decimal Current price minus cost</td></tr><tr><td>current_margin-pct</td><td></td><td>decimal Margin over current price</td></tr><tr><td>msrp-gap</td><td></td><td>decimal Product msrp minus current price</td></tr><tr><td></td><td></td><td>overlapping-current_prices integer 1 where competing current list rows existed</td></tr><tr><td>volatility</td><td>varchar Classification</td><td></td></tr></table>

Sort by price\_changes descending, then sku ascending.

## Success Criteria

1. The model compiles and runs without errors

2. Each (variant\_id, currency\_code) pair appears exactly once

3. increases plus decreases equals price\_changes on every row

4. Exactly one current list price is resolved per row, or none where nothing is current

5. Variants whose only price rows are tier or channel specific do not appear

6. volatility is one of the four documented values on every row

7. No infinity or NaN values anywhere in the output

## Guidelines

• The model should be materialized as a table

• Build the ordered price chain with a window function; a self-join on effective date will fan out wherever two rows share one

• Do NOT modify upstream staging models

Example Training Task 3: dbt-payroll-cost-by-department-synthetic

Build Payroll Cost by Department Model

## Overview

Payroll is run in periods and posted as a run header with employee-level detail beneath it. Time is recorded separately, in entries per employee per day. The two are never compared, so nobody can say whether the hours being paid match the hours being worked, or what a department actually costs.

Build rpt\_payroll\_cost\_by\_department, which brings the three together.

Files

• DuckDB: /app/dbt\_models\_duckdb/models/marts/hr/rpt\_payroll\_cost\_by\_department.sql

• Snowflake: /app/dbt\_models\_snowflake/models/marts/hr/rpt\_payroll\_cost\_by\_department.sql

[. . . standard Database Backend / dbt Profile Setup boilerplate . . . ]

## Requirements

Data Sources

• stg\_hr\_\_payroll\_runs - payroll\_run\_id, payroll\_period, period\_start, period\_end, pay\_date, status, total\_gross, total\_net, employee\_count

• stg\_hr\_\_payroll\_details - detail\_id, payroll\_run\_id, employee\_id, gross\_pay, tax\_deductions, other\_deductions, net\_pay, hours\_worked, overtime\_hours

• stg\_hr\_\_time\_entries - entry\_id, employee\_id, entry\_date, clock\_in, clock\_out, break\_minutes, hours\_worked, entry\_type, status

• stg\_hr\_\_employees - employee\_id, department\_id, hire\_date, termination\_date, employment\_type, status

• stg\_hr\_\_departments - department\_id, department\_code, department\_name

## Grain

One row per (department\_code, payroll\_period).

## Only Posted Runs

Payroll runs that were drafted and never posted must be excluded from every figure. Inspect the run status vocabulary before deciding which mean posted; there is more than one non-posted value.

Detail rows whose payroll\_run\_id matches no run are orphans and are excluded, counted in orphan\_detail\_rows.

Matching Time To Pay

For each detail row, sum the employee’s time entries whose entry\_date falls within the run’s period\_start to period\_end inclusive. Only entries in an approved state count; inspect the entry status values.

The comparison is between hours\_worked on the payroll detail and the summed hours\_worked from the time entries. A difference of more than 0.5 hours is a discrepancy. Report the count and the net direction, because paying more hours than were recorded and recording more than were paid are different problems with different owners

Employees with payroll detail and no time entries at all in the period are counted separately from those with a discrepancy. A salaried employee who does not clock in is normal; a waged one who does not is not, and employment\_type is what distinguishes them.

## Output Columns

• department\_code, department\_name, payroll\_period

• period\_start, period\_end, pay\_date

• employees\_paid - distinct employees with a detail row

• gross\_pay, tax\_deductions, other\_deductions, net\_pay - summed

• deduction\_rate - deductions over gross, rounded to 4 decimals

• paid\_hours, paid\_overtime\_hours - summed from payroll detail

• recorded\_hours - summed from approved time entries in the period

• hours\_gap - paid\_hours minus recorded\_hours

• employees\_with\_discrepancy

• employees\_overpaid\_hours, employees\_underpaid\_hours - the split by direction

• employees\_without\_time\_entries

• salaried\_without\_time\_entries - of those, on a salaried employment type

• avg\_gross\_per\_employee, avg\_hourly\_rate - the second gross\_pay over paid\_hours, both rounded to 2 decimals

• overtime\_share - overtime hours over paid hours, rounded to 4 decimals

• header\_gross\_drift - the run header’s total\_gross minus the summed detail gross\_pay, apportioned to this department by its share of the run’s detail rows and rounded to 2 decimals

• orphan\_detail\_rows - repeated on every row

Employees whose department does not resolve group under ’UNKNOWN’.

Sort by payroll\_period ascending, then gross\_pay descending.

## Why Three Sources And Not Two

Payroll detail alone gives cost. Time entries alone give effort. Neither on its own answers the question a department head actually asks, which is whether the cost is buying the effort.

The run header is the third source and it exists for a different reason: it is what the finance system was told. Where the header disagrees with the detail beneath it, the general ledger and this model will report different numbers for the same period, and somebody will spend a day finding out why. header\_gross\_drift is there so that day is not spent.

The apportionment of that drift by share of detail rows is deliberately crude. A drift on the header cannot be attributed to a department, because the header does not know about departments. Spreading it evenly by row count is a convention, not a measurement, and the column should be read as “roughly this much of the unexplained difference sits with this department” rather than as a figure to book.

## Notes On The Time Data

Time entries carry an entry\_type alongside their status, and the two are independent. An entry can be approved and still be of a type that should not count toward paid hours — training, unpaid leave, a correction entry that zeroes an earlier one.

Inspect the entry\_type values before deciding which contribute to recorded\_hours. Including every approved entry regardless of type will inflate recorded hours and turn genuine overpayments into apparent underpayments, which is worse than reporting nothing.

Where an entry type is ambiguous, include it and say so in a comment. An inflated recorded\_hours produces conservative discrepancy counts, and under-reporting a discrepancy is the safer error here: it sends fewer people to investigate things that turn out to be fine.

Break minutes are recorded separately and are already excluded from hours\_worked on the entry. Do not subtract them again.

## Worked Example

A payroll period 2023-07-B running 2023-07-16 to 2023-07-31, posted, with three employees in the Fulfilment department.

Employee e-10, waged, has payroll detail showing 72.0 hours and 6.0 overtime, gross 1,560.00. Their approved time entries in the window sum to 71.5 hours. The gap is 0.5, which is not more than 0.5, so it is not a discrepancy.

Employee e-11, waged, shows 80.0 paid hours. Their approved entries sum to 68.0. The gap is 12.0 and counts as a discrepancy in the overpaid direction — twelve hours were paid that nobody recorded.

Employee e-12, salaried, shows 75.0 paid hours and has no time entries at all. They count in employees\_without\_- time\_entries and in salaried\_without\_time\_entries, and not in employees\_with\_discrepancy. A salaried employee not clocking in is expected; the model separates them precisely so the waged case stands out.

So for that department and period: employees\_paid 3, paid\_hours 227.0, recorded\_hours 139.5, hours\_gap 87.5, employees\_with\_discrepancy 1, employees\_overpaid\_hours 1, employees\_underpaid\_hours 0, employees\_- without\_time\_entries 1, salaried\_without\_time\_entries 1.

Note that hours\_gap of 87.5 is dominated by the salaried employee, whose 75 hours have no time entries by design. The gap column is therefore not a discrepancy measure on its own, and reading it as one would flag every department carrying salaried staff. employees\_with\_discrepancy is the measure that means something.

## Edge Cases

An employee appearing in two departments across two periods produces a row in each. The employee record carries only the current department, so both periods are attributed to it — that limitation is the same one the headcount model has and is worth the same comment.

An employee with detail rows in two runs covering overlapping periods will have their time entries counted against both. Overlapping payroll periods are a data error rather than a normal occurrence; count the affected employees in employees\_in\_overlapping\_periods, reported on every row, and leave the double count in place rather than trying to apportion it.

A time entry whose clock\_out is before its clock\_in crosses midnight or is corrupt. Trust hours\_worked on the entry rather than recomputing it from the two timestamps — the entry’s own figure is what the approval process signed off.

A detail row with a NULL hours\_worked belongs to a salaried employee on a period-based arrangement. It contributes to pay and to no hours figure.

A run whose header employee\_count disagrees with the count of detail rows is a header drift like any other. It is not reported here; the money columns are what matter and header\_gross\_drift covers them

## Success Criteria

1. The model compiles and runs without errors

2. Each (department\_code, payroll\_period) pair appears exactly once

3. employees\_overpaid\_hours plus employees\_underpaid\_hours equals employees\_with\_discrepancy

4. Unposted payroll runs contribute to no figure

5. net\_pay plus tax\_deductions plus other\_deductions equals gross\_pay on every row, within a cent

6. employees\_without\_time\_entries never exceeds employees\_paid

7. No infinity or NaN values anywhere in the output

## Guidelines

• The model should be materialized as a table

• Aggregate the time entries to one row per employee and run before joining them to the payroll detail. Joining entries directly will multiply the pay by the number of days worked, which is the largest error this model can produce and the least obvious.

• Do not use CURRENT\_DATE; the periods carry their own dates

• Do NOT modify upstream staging models