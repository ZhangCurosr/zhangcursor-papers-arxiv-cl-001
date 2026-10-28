# EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning

Sihan Yang<sup>1,∗</sup>, Weichen Bai<sup>2,∗</sup>, Ruichuan An<sup>1,∗,†</sup>, Yiming Cheng<sup>3</sup> , Zhiyuan Feng<sup>3</sup>, Tianyi Jiang<sup>1</sup>, Renye Yan<sup>1</sup>, Ning Ding<sup>4</sup> , Yifan Zhang<sup>5,†</sup>, Renrui Zhang<sup>6</sup>, Yehui Tang<sup>4,‡</sup>, Wentao Zhang<sup>1,‡</sup>

<sup>1</sup>PKU <sup>2</sup>XJTU <sup>3</sup>THU <sup>4</sup>Samsung Research <sup>5</sup>CASIA <sup>6</sup>CUHK

<sup>∗</sup>Equal contribution., <sup>†</sup>Project Leader., <sup>‡</sup>Corresponding authors.

## Abstract

As executable environments for language agents scale, generating and verifying their rollouts becomes increasingly expensive. Under finite rollout budgets, this raises an important question: can data management at the environment level make agentic reinforcement learning (RL) more eficient by organizing, evaluating, and scheduling task instances together with the environments that determine their tools, states, transitions, and verification? To address this question, we introduce EnvSched (Environment-aware Scheduling), an online scheduler that learns from rollout history. EnvSched assigns scheduling value through two complementary modeling choices: exploitation prioritizes tasks likely to provide informative reward diferences, while coverage spreads each batch across underobserved environments and interaction patterns. EnvSched organizes environment–task pairs as nodes in a dual-relation graph, with within-environment edges capturing task semantics and shared tool, state, and verifier structure, and cross-environment edges capturing abstract execution and reward topology. By propagating historical rollout statistics along both edge types, it predicts within-task reward variance. These predictions guide a scheduler that allocates each batch to tasks with high predicted reward variance and underobserved environments and tasks. Environment quotas and a similarity penalty further limit concentration and redundancy, yielding informative and diverse batches without additional pilot rollouts. In RL on Qwen3-8B, EnvSched achieves an average score of 48.81 across BFCL-v3, τ-Bench, and ACEBench-Agent after 40 training steps, exceeding the highest score attained by random task selection after 120 steps. It also improves downstream performance across model sizes at the same training budget. Statistical analyses and ablation studies highlight the complementary roles of within-task reward diferentiation and cross-environment coverage in these improvements.

Correspondence: hankyang428@gmail.com

Date: October 2026

Code: https://github.com/Hhankyangg/envsched

![](images/38a0b0f31909587f8aebbd7c05c412eddcc67ddccabe803d8d50c020b030f1ae.jpg)  
Figure 1 The left panel summarizes the growth of environments and RL resources in recent agent systems. The right panel shows EnvSched’s training-eficiency gain on Qwen3-8B under matched rollout budgets. Scores are averaged across BFCL-v3, τ-Bench, and ACEBench-Agent.

## 1 Introduction

Large language models (LLMs) are increasingly trained to act in stateful tool environments, where completing a user request requires multi-turn interaction, coordinated tool use, and consistent updates to the underlying state [1–3]. For example, an agent rescheduling a reservation must retrieve the relevant records, check availability and domain rules, and update dependent entries in a valid order. Executable environments support this interaction through database-backed states, tool interfaces, transition logic, and task-specific verifiers. Recent systems have made executable environments a central scaling axis for Agentic RL, expanding training pools from tens to thousands of environments [4–7]. This expansion comes with substantial training costs: industrial reports describe RL over thousands of environments and post-training runs consuming hundreds of thousands of GPU-hours [8–11]. Figure 1 (left) summarizes the growing scale of environments and reported RL resources.

As the number of executable environments grows, the cost of generating and verifying RL rollouts grows with them. Researchers [12, 13] and industrial technical reports [10, 11] increasingly emphasize that training eficiency must be improved through better organization of the training data itself, including environment design, trajectory generation, rollout scaling, and curriculum learning. Datacentric systems such as LLM-based subpopulation analysis and unified data workflow automation further illustrate the value of structuring and auditing data before model training [14, 15]. Existing work has begun to study online task selection during RL, estimating training value from task dificulty, prompt representations, or historical rewards [16–18]. However, these methods model the prompt or task while leaving the executable environment outside the selection model. This misses a key source of variation in Agentic RL: the environment determines the tools, initial state, transition rules, and verifier, and thus shapes how a task unfolds and what learning signal it produces. This motivates our formulation of environment-level data management.

We formulate this challenge as environment-level data management: organizing, evaluating, and scheduling task instances together with the executable environments in which they are carried out. Each environment specifies the available tools, initial and intermediate states, transition rules, and task verifier. To address this challenge, we draw on the update mechanism of group-relative RL methods such as GRPO [19]. In these methods, reward diferences among rollouts of the same task determine their relative advantages; tasks with nearly identical outcomes therefore provide little learning signal. The scheduler should prioritize tasks whose current reward distributions are likely to yield informative diferences. At the same time, repeatedly training on a small set of environments restricts the tools, states, transitions, and verification patterns seen by the policy, even when the selected tasks are individually informative. Efective scheduling must therefore preserve diversity across environments and interaction patterns. Building on these two requirements, we introduce EnvSched (Environment-aware Scheduling), an online scheduler learning from rollout history. It prioritizes task instances with high predicted reward informativeness, while it also spreads each batch across underobserved environments and structurally distinct interactions.

EnvSched implements exploitation and coverage as two coordinated parts of the same scheduling procedure. To support both decisions, EnvSched represents environment–task pairs as nodes in a dual-relation graph. Within-environment edges connect tasks through shared semantics, tools, database fields, and verifier constraints. Cross-environment edges connect tasks through abstract execution and reward topology, enabling information transfer across similar environments. For example, reassigning a consultant and moving a patient between hospital beds can share a dependency structure of entity lookup, eligibility checking, relation replacement, and dependent-state validation.

At each training step, EnvSched aggregates historical rollout statistics separately along the two edge types and combines the resulting estimates to predict each task’s reward variance under the evolving policy. It then allocates half of each batch to underobserved environments and tasks for coverage, and half to tasks with high predicted reward variance for exploitation. Environment quotas and a similarity penalty limit concentration and redundancy within the shared batch, while newly collected rollout outcomes update the node histories for subsequent rounds. This rollout–update– propagation–selection loop balances policy-dependent task prioritization with explicit environment coverage without additional pilot rollouts.

We evaluate EnvSched by conducting reinforcement learning on the EnvScaler training pool with matched per-step rollout budgets. The right side of Figure 1 shows that EnvSched reaches an average score of 48.81 on Qwen3-8B across BFCL-v3, τ-Bench, and ACEBench-Agent [20–22] after 40 steps, exceeding the highest score attained by random task selection during 120 steps. It also improves downstream performance at the same training budget, with gains across model sizes. Analysis of selected-task rollouts shows that mean within-task reward variance rises from 0.03208 to 0.04290, and environments covered per batch also rises from 36.53 to 45.96 on Qwen3-4B. Ablation studies confirm that reward diferentiation and environment coverage are complementary: either component alone is insuficient for efective training. More broadly, EnvSched demonstrates the distinctive value to eficient Agentic RL training, paving the way for scaling increasingly capable agent models with better use of training budgets.

## Our contributions are threefold:

• We are the first to formulate environment-level data management for Agentic RL, jointly modeling task specifications and executable environment structure.

• We propose EnvSched, an Agentic RL training data scheduler which combines a dual-relation graph, historical reward-variance prediction, and coverage-aware batch construction to schedule training tasks without additional pilot rollouts.

• We demonstrate consistent gains in rollout eficiency across models of varying sizes and improved performance on a range of downstream tool-use benchmarks. Our analyses show that reward diferentiation and environment coverage are complementary sources of training value, establishing environment-level data management as a promising data-centric direction for scaling eficient Agentic RL.

![](images/cd36139ac40e17fa1bc84077249c3eb9409197853c7699fac7b7abf3908ff560.jpg)  
Figure 2 EnvSched training loop. Circles with the same color denote task–environment pairs from the same environment.

## 2 Method

## 2.1 Problem Setup and Training Loop

EnvSched treats an environment–task pair as the basic unit of training and scheduling. An environment is an executable setting that defines the tool interfaces, transition rules, state representation, and verifier; a task instance specifies the request and its initial state within that environment. Each pair therefore identifies one concrete rollout problem, and the model interacts with it through text responses and tool calls to produce a trajectory and its task-specific evaluation reward. At each training step, the scheduler chooses a batch of distinct pairs for rollout and model update.

As illustrated in Figure 2, EnvSched builds a dual-relation graph from task specifications and executable environment structure before training, then initializes the schedule with a coverageoriented batch. Rollouts on the selected pairs produce rewards for model updates and statistics for the corresponding node histories. Before the next training step, historical statistics propagate along within-environment and cross-environment edges to estimate task informativeness. The scheduler combines coverage of underobserved environments and tasks with exploitation of informative tasks, using environment quotas and a similarity penalty to limit concentration and redundancy. This rollout–update–propagation–selection loop repeats throughout training.

## 2.2 dual-relation graph

We define $\mathcal { G } = ( \mathcal { V } , \mathcal { E } ^ { \mathrm { i n } } , \mathcal { E } ^ { \mathrm { c r o s s } } )$ over the environment–task pairs. Each node stores the pair’s historical reward statistics and selection count. The two edge types are constructed from similarity: withinenvironment similarities compare task semantics and concrete tool, state, and verifier features among pairs in the same environment; cross-environment similarities compare abstract execution and reward structure among pairs in diferent environments. Similarity determines which observed nodes contribute evidence and how strongly they contribute to the target node’s estimate.

Before training, a language model reads the task instruction, initial-state summary, tool descriptions, and verifier to extract a structured operation description. It records required operations, prerequisites, execution dependencies, and relationships between operations and reward checks. For withinenvironment similarity, six TF–IDF views encode instruction words and character fragments, verifiercheck text, verifier code structure, initial context, and the complete operation description. For cross-environment similarity, tool names, field names, entity identifiers, and target values are removed, and operations and conditions are mapped to categories such as lookup, eligibility checking, relation replacement, and dependent-state validation. The LLM prompt is provided in Appendix E. Let $h _ { i } ^ { ( a ) }$ encode concrete view a and $g _ { i }$ encode the abstract description. The similarities are

$$
S _ { i j } ^ { \mathrm { f u l l } } = \sum _ { a = 1 } ^ { 6 } \alpha _ { a } \cos ( h _ { i } ^ { ( a ) } , h _ { j } ^ { ( a ) } ) , \qquad S _ { i j } ^ { \mathrm { a b s } } = \cos ( g _ { i } , g _ { j } ) .\tag{1}
$$

The full similarity retains domain-specific information, while the abstract similarity supports transfer between tasks with diferent business terminology. For example, consultant reassignment and hospitalbed transfer can involve similar lookup, eligibility, and relation-update operations. At each training step, each target node selects its most similar observed neighbors separately within and across environments, using $S ^ { \mathrm { f u l l } }$ and $S ^ { \mathrm { a b s } }$ , respectively. We denote these sets by $\mathcal { N } _ { t } ^ { \mathrm { i n } } ( i )$ and $\mathcal { N } _ { t } ^ { \mathrm { c r o s s } } ( i )$ , with $| \mathcal { N } _ { t } ^ { \mathrm { i n } } ( i ) | \leq K _ { \mathrm { w i t h i n } }$ and $| \mathcal { N } _ { t } ^ { \mathrm { c r o s s } } ( i ) | \le K _ { \mathrm { c r o s s } }$ . Similarities remain fixed during training; node histories and the available observed neighbors evolve. Feature weights and neighbor configurations are specified in Appendix D. Figure $\mathrm { 3 ( a ) }$ summarizes this construction: colored regions group task–environment pairs by environment, while the two edge types transfer historical reward information through concrete and abstract structural similarities.

Figure $\mathrm { 3 ( b ) }$ summarizes how observed nodes inform estimates for the pool before the next rollout step: the two relation-specific estimates form a graph prior, which is fused with the target’s own history and then calibrated into the informativeness score. Each observation records the mean reward $m ,$ the within-task reward variance $v ,$ and the mean-corrected dispersion d. Let K denote the number of rollouts per task and $r _ { i , s , 1 } , \ldots , r _ { i , s , K }$ the rewards of the complete rollout group for task i at step $s ;$ we compute the population variance over all K rewards:

$$
m _ { i , s } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } r _ { i , s , k } , \qquad v _ { i , s } = \mathrm { V a r } _ { k = 1 , \ldots , K } ( r _ { i , s , k } ) = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } ( r _ { i , s , k } - m _ { i , s } ) ^ { 2 } .\tag{2}
$$

This statistic directly measures the reward diferentiation available within the group used for the policy update. The mean reward and mean-corrected dispersion provide complementary information about the same group-level reward distribution; their definitions appear in Appendix B.

## 2.3 Predicting Task Informativeness

For each statistic $y \in \{ v , m , d \}$ , we discount older observations and center them against their step-level mean before shifting to the current global reference $b _ { t } ^ { y }$ . Let $\bar { y } _ { i , t }$ denote the historical estimate and $W _ { i , t }$ its efective observation weight. Along each relation $r \in \{ \mathrm { i n } , \mathrm { c r o s s } \}$ , we aggregate observed neighbors with weights $a _ { i j , t } ^ { r }$ determined by similarity and the amount of recent history:

$$
G _ { i , t } ^ { r , y } = \frac { \sum _ { j \in \mathcal { N } _ { t } ^ { r } ( i ) } a _ { i j , t } ^ { r } \bar { y } _ { j , t } + \lambda _ { g } b _ { t } ^ { y } } { \sum _ { j \in \mathcal { N } _ { t } ^ { r } ( i ) } a _ { i j , t } ^ { r } + \lambda _ { g } } .\tag{3}
$$

Shrinkage toward $b _ { t } ^ { y }$ stabilizes estimates with little neighbor evidence. We weight the two relation types equally and average their estimates into a graph prior, then combine it with the target’s history:

$$
P _ { i , t } ^ { y } = { \textstyle \frac { 1 } { 2 } } G _ { i , t } ^ { \mathrm { i n } , y } + { \textstyle \frac { 1 } { 2 } } G _ { i , t } ^ { \mathrm { c r o s s } , y } , \qquad \widehat { y } _ { i , t } = \frac { W _ { i , t } \bar { y } _ { i , t } + \lambda _ { s } P _ { i , t } ^ { y } } { W _ { i , t } + \lambda _ { s } } .\tag{4}
$$

![](images/5621efa7063bc0565e9ddca8d556fc943fb2243dab269aa2c6b43624aacce418.jpg)  
Figure 3 Dual-relation graph and informativeness prediction. (a) Within-environment and cross-environment edges propagate historical statistics from observed task–environment pairs to the target. (b) Relation-specific estimates are fused with the target’s history to produce the final score.

An unobserved node relies on the graph prior; its own observations gain weight as its history grows. We combine the predicted variance with a variance estimate reconstructed from the predicted mean and dispersion, then compare estimates from long and short history windows. A recent decline lowers the priority. The resulting percentile score $q _ { i , t }$ ranks tasks for scheduling. Appendix B gives the complete calibration, fusion, and scoring definitions.

## 2.4 Coverage-Aware Task Scheduling

The scheduler assigns a fraction ρ of each batch to coverage and the remaining fraction $1 - \rho$ to exploitation. Coverage first restricts the candidate set using a lexicographic priority over current-batch environment coverage, cumulative environment exposure, and task observation count. Within the resulting candidate set, it prioritizes predicted informativeness. Exploitation prioritizes predicted informativeness while gradually discounting repeated exposure. For cumulative selection count $c _ { i , t }$ the exposure factor is $\omega _ { i , t } = ( 1 + c _ { i , t } / c _ { 0 } ) ^ { - 1 / 2 }$ , and the exploitation score is $q _ { i , t } \omega _ { i , t }$

Both phases build a shared batch greedily. Let $B _ { \mathrm { c o v } } = \lfloor \rho B \rfloor$ and $B _ { \mathrm { e x p } } = B - B _ { \mathrm { c o v } }$ be the phase budgets. Given the eligible candidates A and the current selected set B, the next node is

$$
i ^ { * } = \arg \operatorname* { m a x } _ { i \in \mathcal { A } } \left[ a _ { i , t } ^ { \mathrm { { p h a s e } } } - \lambda _ { \sin } \operatorname* { m a x } _ { j \in \mathcal { B } } { S _ { i j } ^ { \mathrm { { f u l l } } } } \right] ,\tag{5}
$$

where the phase score is $q _ { i , t }$ for coverage and $q _ { i , t } \omega _ { i , t }$ for exploitation. For coverage, A is first restricted by the lexicographic coverage priority described above; Eq. 5 then resolves the remaining choices using informativeness and similarity. The similarity penalty is zero for an empty batch. It uses the full task representation across selected pairs to reduce redundant batch content. Environment quotas and consecutive-selection limits further control concentration.

Candidate eligibility is part of the scheduler and includes two additional filters. A progress filter temporarily excludes tasks with non-positive estimated progress from exploitation while retaining coverage opportunities for reevaluation. An identifier filter uses fixed-action replay to confirm suspicious reward-producing actions and removes the afected tasks from both phases. These filters

Table 1 Main results. “Overall” refers to the weighted score. Green and red subscripts denote variations relative to random scheduling.
<table><tr><td rowspan="2">Model</td><td rowspan="2"></td><td rowspan="2">Steps</td><td colspan="5">BFCL-v3 Multi-Turn</td><td colspan="3">τ-Bench</td><td colspan="3">ACEBench-Agent</td></tr><tr><td>Base</td><td>Miss- Miss- Long- Func Param Context</td><td></td><td></td><td>Overall</td><td>Retail Airline</td><td></td><td>Overall</td><td>Multi- Multi- Step Turn</td><td></td><td>Overall</td></tr><tr><td></td><td>Qwen3-1.7B</td><td></td><td></td><td>13.50 6.00</td><td>12.50</td><td>7.00</td><td>9.75</td><td>8.99</td><td>16.00</td><td>11.11</td><td>35.0027.78</td><td></td><td>30.67</td></tr><tr><td></td><td>+ Random</td><td>200</td><td></td><td>13.17 10.50</td><td>10.50</td><td>7.67</td><td>10.46</td><td>15.36</td><td>16.00</td><td>15.56</td><td>35.0024.44</td><td></td><td>28.67</td></tr><tr><td></td><td>+ DPS</td><td>200</td><td></td><td>12.50 10.83</td><td>9.33</td><td>7.00</td><td> $\pmb { 9 . 9 2 . 0 . 5 4 }$ </td><td>16.81</td><td>17.33</td><td> $1 6 . 9 7 _ { + 1 . 4 1 }$ </td><td>33.33 26.67</td><td></td><td> $2 9 . 3 3 _ { + 0 . 6 6 }$ </td></tr><tr><td></td><td>+ LEEPS</td><td>200</td><td></td><td>13.83 11.67</td><td>11.00</td><td>8.50</td><td> $1 1 . 2 5 _ { + 0 . 7 9 }$ </td><td>16.52</td><td>17.33</td><td> $1 6 . 7 7 _ { + 1 . 2 1 }$ </td><td>35.0027.78</td><td></td><td> $3 0 . 6 7 _ { + 2 . 0 0 }$ </td></tr><tr><td></td><td>十 ENVSCHED</td><td>40</td><td></td><td>13.50 13.00</td><td>11.50</td><td>9.00</td><td> $1 1 . 7 5 _ { + 1 . 2 9 }$ </td><td>17.10</td><td>16.67</td><td> $1 6 . 9 7 _ { + 1 . 4 1 }$ </td><td>40.0024.44</td><td></td><td> $3 0 . 6 7 _ { + 2 . 0 0 }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>200</td><td></td><td>14.00 14.50</td><td>12.17</td><td>9.00</td><td> $1 2 . 4 2 _ { + 1 . 9 6 }$ </td><td>18.26</td><td>16.00</td><td> $1 7 . 5 8 _ { + 2 . 0 2 }$ </td><td>41.67</td><td>24.44</td><td> $\mathbf { 3 1 . 3 3 _ { + 2 . 6 6 } }$ </td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td>32.00 20.00</td><td>24.00</td><td>25.50</td><td>25.38</td><td>40.87</td><td>26.00</td><td>36.36</td><td>56.67 51.11</td><td></td><td>53.33</td></tr><tr><td></td><td>+ Random</td><td>200</td><td></td><td>36.67 26.17</td><td>25.50</td><td>27.00</td><td>28.83</td><td>40.87</td><td>30.67</td><td>37.78</td><td>70.00</td><td>46.67</td><td>56.00</td></tr><tr><td></td><td>+ DPS</td><td>200</td><td></td><td>38.33 27.00</td><td>26.67</td><td>29.50</td><td> $3 0 . 3 8 _ { + 1 . 5 5 }$ </td><td>39.42</td><td>30.67</td><td> $3 6 . 7 7 _ { - 1 . 0 1 }$ </td><td>70.0047.78</td><td></td><td> ${ \pmb 5 6 . 6 7 _ { + 0 . 6 7 } }$ </td></tr><tr><td></td><td>+ LEEPS</td><td>200</td><td></td><td>38.00 27.00</td><td>26.33</td><td>28.67</td><td> $\mathbf { 3 0 . 0 0 _ { + 1 . 1 7 } }$ </td><td>42.32</td><td>30.67</td><td> $\mathbf { 3 8 . 7 9 _ { + 1 . 0 1 } }$ </td><td>70.0047.78</td><td></td><td> ${ \pmb 5 6 . 6 7 _ { + 0 . 6 7 } }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>40</td><td></td><td>36.00 27.67</td><td>26.50</td><td>30.50</td><td> $\mathbf { 3 0 . 1 7 _ { + 1 . 3 4 } }$ </td><td>45.80</td><td>30.00</td><td> $4 1 . 0 1 _ { + 3 . 2 3 }$ </td><td>70.0050.00</td><td></td><td> $5 8 . 0 0 _ { + 2 . 0 0 }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>200</td><td></td><td>40.67 28.17</td><td>27.50</td><td>31.17</td><td> $3 1 . 8 8 _ { + 3 . 0 5 }$ </td><td>45.51</td><td>30.67</td><td> $4 1 . 0 1 + 3 . 2 3$ </td><td>70.0050.00</td><td></td><td> $5 8 . 0 0 _ { + 2 . 0 0 }$ </td></tr><tr><td></td><td>Qwen3-8B</td><td>1</td><td></td><td>32.00 33.50</td><td>22.00</td><td>28.00</td><td>28.88</td><td>46.38</td><td>30.00</td><td>41.41</td><td>61.67 56.67</td><td></td><td>58.67</td></tr><tr><td></td><td>+ Random</td><td>200</td><td></td><td>37.50 34.00</td><td>27.50</td><td>36.50</td><td>33.88</td><td>56.23</td><td>26.67</td><td>47.27</td><td>61.67 51.11</td><td></td><td>55.33</td></tr><tr><td></td><td>+ DPS</td><td>200</td><td></td><td>40.67 34.17</td><td>29.83</td><td>37.33</td><td> $3 5 . 5 0 _ { + 1 . 6 2 }$ </td><td>57.68</td><td>26.00</td><td> $4 8 . 0 8 _ { + 0 . 8 1 }$ </td><td>65.0053.33</td><td></td><td> $5 8 . 0 0 _ { + 2 . 6 7 }$ </td></tr><tr><td></td><td>+ LEEPS</td><td>200</td><td></td><td>41.17 35.33</td><td>29.33</td><td>37.00</td><td> $3 5 . 7 1 _ { + 1 . 8 3 }$ </td><td>56.23</td><td>28.00</td><td> $4 7 . 6 8 _ { + 0 . 4 1 }$ </td><td>60.0051.11</td><td></td><td> ${ \pmb 5 4 . 6 7 _ { - 0 . 6 6 } }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>40</td><td></td><td>45.17 35.83</td><td>31.50</td><td>38.33</td><td> $3 7 . 7 1 _ { + 3 . 8 3 }$ </td><td>58.26</td><td>33.33</td><td> $5 0 . 7 1 _ { + 3 . 4 4 }$ </td><td></td><td>70.0050.00</td><td> $5 8 . 0 0 _ { + 2 . 6 7 }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>200</td><td></td><td>46.50 35.00</td><td>31.00</td><td>39.00</td><td> $3 7 . 8 8 _ { + 4 . 0 0 }$ </td><td>58.26</td><td>32.00</td><td> $5 0 . 3 0 _ { + 3 . 0 3 }$ </td><td></td><td>70.0051.11</td><td> $5 8 . 6 7 _ { + 3 . 3 4 }$ </td></tr><tr><td></td><td>Qwen3-30B-A3B</td><td></td><td></td><td>39.50 27.00</td><td>33.00</td><td>32.00</td><td>32.88</td><td>61.45</td><td>32.00</td><td>52.53</td><td></td><td>80.0056.67</td><td>66.00</td></tr><tr><td></td><td>+ Random</td><td>200</td><td></td><td>40.67 29.17</td><td>30.00</td><td>35.67</td><td>33.88</td><td>57.39</td><td>38.67</td><td>51.72</td><td>81.67 58.89</td><td></td><td>68.00</td></tr><tr><td></td><td>ENVSCHED</td><td>40</td><td></td><td>43.00 32.17</td><td>28.17</td><td>36.67</td><td> $3 5 . 0 0 _ { + 1 . 1 3 }$ </td><td>60.00</td><td>38.67</td><td> $5 3 . 5 4 _ { + 1 . 8 2 }$ </td><td></td><td>86.67 58.89</td><td> $7 0 . 0 0 _ { + 2 . 0 0 }$ </td></tr><tr><td></td><td>+ ENVSCHED</td><td>200</td><td></td><td>47.00 35.67</td><td>30.00</td><td>38.50</td><td> $3 7 . 7 9 _ { + 3 . 9 2 }$ </td><td>60.00</td><td>42.67</td><td> $5 4 . 7 5 _ { + 3 . 0 3 }$ </td><td></td><td>86.67 60.00</td><td> $7 0 . 6 7 _ { + 2 . 6 7 }$ </td></tr></table>

change candidate eligibility rather than the ranking score. At initialization, the scheduler selects environment representatives and fills the remaining slots using task novelty. The exact eligibility rules and initialization procedure appear in Appendix C.

## 3 Experiments

## 3.1 Experimental Setup

Training Environments and Evaluation. We train Qwen3-Thinking [23] models with GRPO [19] on the EnvScaler [5] executable-environment pool, which contains 2,550 task instances across 51 executable environments. Each task instance is evaluated through its environment-specific tools, initial state, and terminal verifier. We report performance on BFCL-v3 Multi-Turn, τ-Bench, and ACEBench-Agent [20–22], covering function calling, multi-turn interaction, and long-horizon tool use.

Implementation Details. For training with EnvSched, we apply a unified GRPO configuration across Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-30B-A3B. Each training batch contains $B = 6 4$ environment–task pairs with rollout K = 8 per pair (512 trajectories). The scheduler uses a fixed coverage fraction $\rho = 1 / 2 ,$ , yielding 32 coverage pairs and 32 exploitation pairs. We train for 200 single-epoch steps using Adam $( \mathrm { l r } = 1 0 ^ { - 6 }$ , cosine schedule, 10 warmup steps) with a policy-ratio clip of 0.2 and batch-whitened, [−0.2, 0.2]-clipped advantages. Rollouts sample at temperature 0.99 $( \mathrm { t o p } { - } p = 0 . 9 9 , \mathrm { t o p } { - } k = 1 0 0 )$ with limits of 40 turns, 4,096 tokens per turn, and 32,000 tokens per sequence. The policy loss covers only model-generated tokens, including reasoning; all other tokens are masked. All scheduler and optimization hyperparameters are fixed across model sizes. The default graph neighborhood uses $K _ { \mathrm { w i t h i n } } = 6$ within-environment neighbors and $K _ { \mathrm { c r o s s } } = 1 2$ crossenvironment neighbors. Graph representations, scheduling constants, and remaining hyperparameters are listed in Appendix D. For testing on the benchmarks, we use GPT-5.6-terra [24] as the user model for τ-Bench, and GPT-4o-2024-08-06 [25] for ACEBench-Agent. We repeat all experiments with three random seeds and report seed-aggregated results. The Qwen3-30B-A3B experiment uses the same task pool, rollout budget, scheduler and optimization hyperparameters, benchmark protocol, and three-seed evaluation. We compare the untrained model, random scheduling at 200 steps, and EnvSched at 40 and 200 steps.

![](images/d844e5dfb93c54afc6ff0cd471b668ebb74d07a481b42c7c81b035e45237014e.jpg)  
Figure 4 Selected-task reward variance and environment coverage over training steps, with log-transformed vertical-axis values.

Table 2 Training statistics for random scheduling and EnvSched over 200 GRPO training steps. BFCL-v3 Multi-Turn reports the overall score.
<table><tr><td></td><td>Mean reward variance</td><td>Median reward variance</td><td>Median reward</td><td>Environments covered per batch</td><td>Tools used per batch</td><td>BFCL-v3 Multi-Turn</td></tr><tr><td>Random</td><td>0.03208</td><td>0.01468</td><td>0.595</td><td>36.53</td><td>413.41</td><td>28.83</td></tr><tr><td>ENVSCHED</td><td>0.04290</td><td>0.02108</td><td>0.592</td><td>45.96</td><td>478.39</td><td>31.88</td></tr><tr><td>Difference</td><td>+34%</td><td>+44%</td><td>-0.5%</td><td>+26%</td><td>+16%</td><td>+3.05</td></tr></table>

Baselines. The main comparison includes base model without training, random scheduling, and EnvSched, along with recent related training data scheduling methods DPS [17] and LEEPS [18]. Random samples task–environment pairs uniformly from the training pool. DPS (Dynamics-Predictive Sampling) models prompt-solving dynamics from historical rollout outcomes and selects prompts predicted to provide informative learning signals before rollout generation. LEEPS combines representation-space neighbors, historical rollout outcomes, and explore–exploit portfolios to select prompts likely to yield nonzero reward variance.

## 3.2 Main Results

We implement all baselines against the same EnvScaler pool and enforce the same batch size, rollout count, training horizon, and model-update procedure as EnvSched. The results on the benchmarks are presented in Table 1. In addition to the results after 200 training steps, we also include the performance of EnvSched after just 40 steps. Next, we analyze the results from three aspects:

Consistent gains across model sizes at 200 steps. At the 200-step budget, EnvSched improves over random scheduling for all three benchmark overall scores at every model size. Qwen3-4B illustrates the pattern clearly: BFCL-v3 rises from 28.83 to 31.88 (+3.05), τ-Bench from 37.78 to 41.01 (+3.23), and ACEBench-Agent from 56.00 to 58.00 (+2.00). The gains also hold for the other dense models: at 1.7B, the improvements are +1.96, +2.02, and +2.66, while at 8B they are +4.00, +3.03, and +3.34 on BFCL-v3, τ-Bench, and ACEBench-Agent, respectively. The BFCL gain at 4B is distributed across all four categories, including Base (36.67 to 40.67), Miss-Function (26.17 to 28.17), Miss-Parameter (25.50 to 27.50), and Long-Context (27.00 to 31.17). The gains extend to Qwen3-30B-A3B under the same configuration: BFCL-v3, τ-Bench, and ACEBench-Agent reach 37.79, 54.75, and 70.67, improving over random scheduling by 3.92, 3.03, and 2.67 points, respectively, without model-specific scheduler tuning.

High efficiency. EnvSched is already efective after 40 steps within the 200-step training budget. At 4B, it reaches 30.17 on BFCL-v3, 41.01 on τ-Bench, and 58.00 on ACEBench-Agent, matching its 200-step ACEBench-Agent score. At 8B, the 40-step τ-Bench score reaches 50.71, above the 200-step score of 50.30; at 1.7B, ACEBench-Agent reaches 30.67 versus 31.33 after 200 steps. On Qwen3-30B-A3B, the 40-step scores of 35.00, 53.54, and 70.00 exceed the 200-step random baseline on all three benchmarks. These results show that EnvSched can extract useful training signal quickly and, on some benchmarks, match or exceed its longer-run performance.

Prompt-only schedulers do not transfer reliably. DPS and LEEPS are designed around the informativeness of individual prompts, without modeling the environment coupled to each task. Here, choosing a prompt also chooses the tools, state, and constraints exposed to the policy, so prompt-level selection alone cannot reliably identify useful training data. Both methods underperform EnvSched overall, and some of their dimensions are no better than random: DPS is below random on 4B τ -Bench (36.77 vs. 37.78), and LEEPS on 8B ACEBench-Agent (54.67 vs. 55.33). The comparison supports combining reward diferentiation with environment-aware coverage.

## 3.3 Analysis

## 3.3.1 Reward Variance and Environment Coverage

On Qwen3-4B, EnvSched improves both reward diferentiation and coverage under the same 64-task batch budget (Table 2). Compared with random scheduling, it increases mean within-task reward variance from 0.03208 to 0.04290 (+34%) and median variance by 44%, while median reward remains nearly unchanged. Environment coverage rises from 36.53 to 45.96 per batch (+26%) and tool coverage from 413.41 to 478.39 (+16%), alongside a BFCL-v3 improvement from 28.83 to 31.88. The log-transformed curves in Figure 4 show higher variance and coverage for most of the 200 training steps. Over training, variance declines, but the relative magnitude relationships are preserved, while coverage stays stable, preserving reward diferentiation and environment diversity.

## 3.3.2 Computational Cost

EnvSched reuses historical training outcomes and requires zero additional probe trajectories. With cached static similarities and fixed-capacity neighbor heaps, an incremental scheduling step costs O(N∆<sub>t</sub> log $\hat { K } + N \hat { K } + N B )$ , where N is the task-pool size, $\Delta _ { t }$ counts newly eligible neighbor candidates, $\hat { K } = K _ { \mathrm { w i t h i n } } + K _ { \mathrm { c r o s s } }$ is the neighbor budget, and B is the batch size. For fixed $\Delta _ { t } , \hat { K }$ and B, and without boundary-tie fallback scans, this cost is linear in N. In the measured 200-step run, online selection takes 6,985.94 seconds in total, or 34.93 seconds per step. This is a small fraction of the training step: a Qwen3-4B step takes approximately 25 minutes on H100 hardware, so scheduling accounts for about 2.3% of the step time. The incremental backend uses $O ( F + N { \hat { K } } + N )$ persistent space, where F is the number of sparse feature entries. At the 2,550-task scale, sparse features occupy 85.87 MiB, neighbor and progress state occupies approximately 1.24 MiB, and the separately profiled two-matrix representation occupies 42.59 MiB on CPU. These bounded time and memory costs make online environment-aware scheduling practical within the training loop.

Table 3 Component ablations and neighborhood-size sensitivity. Panel (a) varies one component of the full scheduler; panel (b) varies one neighbor count at a time in complete 200-step training runs. Benchmark columns report weighted scores. Green and red subscripts denote changes relative to the full EnvSched or the default neighborhood size.
<table><tr><td>Configuration</td><td>BFCL-v3 Multi-Turn</td><td>τ-Bench</td><td>Agent</td><td>ACEBench Mean reward variance</td><td>Environments per batch</td><td>Tools used per batch</td></tr><tr><td>Random</td><td>28.83</td><td>37.78</td><td>56.00</td><td>0.0321</td><td>36.53</td><td>413.41</td></tr><tr><td>(a) Component contributions</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full ENVSCHED</td><td>31.88</td><td>41.01</td><td>58.00</td><td>0.0429</td><td>45.96</td><td>478.39</td></tr><tr><td>Within-environment edges only</td><td> $3 0 . 4 2 _ { - 1 . 4 6 }$ </td><td> $4 0 . 4 0 _ { - 0 . 6 1 }$ </td><td>58.00</td><td> $0 . 0 3 9 1 _ { - 0 . 0 0 3 8 }$ </td><td> $4 6 . 7 0 _ { + 0 . 7 4 }$ </td><td> $4 9 8 . 5 5 _ { + 2 0 . 1 6 }$ </td></tr><tr><td>Cross-environment edges only</td><td> $3 0 . 6 3 _ { - 1 . 2 5 }$ </td><td>41.01</td><td> $5 7 . 3 3 _ { - 0 . 6 7 }$ </td><td> $0 . 0 3 8 3 _ { - 0 . 0 0 4 6 }$ </td><td> $4 6 . 1 8 _ { + 0 . 2 2 }$ </td><td> $4 6 6 . 5 0 _ { - 1 1 . 8 9 }$ </td></tr><tr><td>Coverage only  $( \rho = 1 )$ </td><td> $2 9 . 7 1 _ { - 2 . 1 7 }$ </td><td> $3 9 . 3 9 _ { - 1 . 6 2 }$ </td><td> $5 6 . 6 7 _ { - 1 . 3 3 }$ </td><td> $0 . 0 3 5 7 _ { - 0 . 0 0 7 2 }$ </td><td> $5 1 . 0 0 _ { + 5 . 0 4 }$ </td><td> $5 0 2 . 1 0 _ { + 2 3 . 7 1 }$ </td></tr><tr><td>Exploitation only  $( \rho = 0 )$ </td><td> $2 9 . 1 7 _ { - 2 . 7 1 }$ </td><td> $3 8 . 5 9 _ { - 2 . 4 2 }$ </td><td> $5 6 . 0 0 _ { - 2 . 0 0 }$ </td><td> $0 . 0 4 4 2 _ { + 0 . 0 0 1 3 }$ </td><td> $3 3 . 9 5 _ { - 1 2 . 0 1 }$ </td><td> $4 2 0 . 6 3 _ { - 5 7 . 7 6 }$ </td></tr><tr><td>(b) Neighborhood size:</td><td> $( K _ { \mathrm { w i t h i n } } , K _ { \mathrm { c r o s s } } )$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Default (6, 12)</td><td>31.88</td><td>41.01</td><td>58.00</td><td>0.0429</td><td>45.96</td><td>478.39</td></tr><tr><td>Half within (3, 12)</td><td> $3 1 . 2 5 _ { - 0 . 6 3 }$ </td><td> $4 1 . 6 2 _ { + 0 . 6 1 }$ </td><td> $5 6 . 6 7 _ { - 1 . 3 3 }$ </td><td> $0 . 0 4 0 9 _ { - 0 . 0 0 2 0 }$ </td><td> $4 7 . 1 8 _ { + 1 . 2 2 }$ </td><td> $4 8 8 . 6 5 _ { + 1 0 . 2 6 }$ </td></tr><tr><td>Double within (12, 12)</td><td> $3 1 . 7 1 _ { - 0 . 1 7 }$ </td><td> $4 0 . 8 1 _ { - 0 . 2 0 }$   $4 0 . 2 0 _ { - 0 . 8 1 }$ </td><td> $5 9 . 3 3 _ { + 1 . 3 3 }$ </td><td> $0 . 0 4 3 4 _ { + 0 . 0 0 0 5 }$   $0 . 0 4 2 5 _ { - 0 . 0 0 0 4 }$ </td><td> $4 5 . 6 6 _ { - 0 . 3 0 }$ </td><td> $4 7 5 . 7 1 _ { - 2 . 6 8 }$ </td></tr><tr><td>Half cross (6, 6) Double cross (6, 24)</td><td> $3 0 . 9 2 _ { - 0 . 9 6 }$   $3 2 . 8 3 _ { + 0 . 9 5 }$ </td><td>41.01</td><td> $5 7 . 3 3 _ { - 0 . 6 7 }$   $5 8 . 6 7 _ { + 0 . 6 7 }$ </td><td> $0 . 0 4 4 1 _ { + 0 . 0 0 1 2 }$ </td><td> $4 5 . 7 0 _ { - 0 . 2 6 }$   $4 6 . 9 0 _ { + 0 . 9 4 }$ </td><td> $4 8 0 . 0 1 _ { + 1 . 6 2 }$   $4 9 3 . 8 3 _ { + 1 5 . 4 4 }$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 3.4 Ablation Studies

## 3.4.1 Component Contributions

We isolate the contributions of the two graph relations and the coverage–exploitation allocation. The relation ablations retain only within-environment edges or only cross-environment edges. The allocation ablations assign all 64 slots to coverage or all 64 slots to exploitation. Table 3 reports performance and statistics. The results show that reward diferentiation and environment coverage must be optimized together. Coverage-only scheduling reaches the broadest batches (51.00 environments) but lowers mean reward variance to 0.0357 and degrades all three downstream scores, while exploitation-only scheduling produces the highest variance (0.0442) yet covers only 33.95 environments and performs substantially worse. Combining the two objectives recovers both informative reward diferences and broad exposure. Full EnvSched has the highest $\mathrm { B F C L - v 3 }$ score among the component configurations and ties the within-environment-only variant on ACEBench-Agent. The cross-environment-only variant matches the full model on τ-Bench but performs worse on BFCL-v3 and ACEBench-Agent. Both edge ablations also reduce mean reward variance relative to the full graph. Modeling both relations therefore provides efective transfer of rollout evidence across tasks and environments; qualitative examples appear in Appendix F.

## 3.4.2 Initialization Robustness

We examine sensitivity to the first batch with 100 independent seven-step training runs. Each run starts from a randomly sampled cold-start batch of 64 tasks and then follows the same 32/32 coverage–exploitation scheduler and training procedure. Table 4 reports the mean and confidence interval across runs and compares the bottom and top quartiles ranked by the variance of the initial batch. The experiment measures how much random initialization afects the EnvSched training.

Table 4 Initialization sensitivity over 100 seven-step runs. Panel (a) reports means with 95% confidence intervals and 5th–95th percentile ranges; panel (b) compares the lowest and highest initial-variance quartiles. Hit rate measures exploitation tasks in the highest-variance quartile.
<table><tr><td colspan="5">(a) Selection quality across random initializations</td></tr><tr><td colspan="2">Metric</td><td>Step 1</td><td>Step 7</td><td>Step 7 range</td></tr><tr><td colspan="2">Mean batch reward variance</td><td>0.0596 [0.0584, 0.0609]</td><td>0.0813 [0.0803, 0.0823]</td><td>0.0726-0.0894</td></tr><tr><td colspan="2">Cumulative mean reward variance</td><td>0.0534 [0.0524, 0.0543]</td><td>0.0679 [0.0673, 0.0684]</td><td>0.0639-0.0723</td></tr><tr><td colspan="2">Exploitation hit rate (%)</td><td>57.2 [55.5, 58.9]</td><td>92.1 [91.1, 93.0]</td><td>84.4-100.0</td></tr><tr><td colspan="5">(b) Persistence of the initial-batch effect</td></tr><tr><td rowspan="2">Step</td><td colspan="2">Mean batch reward variance</td><td colspan="2">Exploitation hit rate (%)</td></tr><tr><td>Low initial group</td><td>High initial group</td><td>Low initial group</td><td>High initial group</td></tr><tr><td></td><td>0.0406</td><td></td><td colspan="2">N/A (initial batch)</td></tr><tr><td>0 1</td><td>0.0574</td><td>0.0540 0.0630</td><td>55.9</td><td>60.4</td></tr><tr><td>4</td><td>0.0746</td><td></td><td></td><td>83.4</td></tr><tr><td>7</td><td>0.0810</td><td>0.0768 0.0833</td><td>84.6 92.8</td><td>92.5</td></tr></table>

Across the 100 runs, the scheduler quickly moves toward high-signal tasks. Mean batch reward variance rises from 0.0596 at step 1 to 0.0813 at step 7, while the exploitation hit rate increases from 57.2% to 92.1%; even the 5th percentile step-7 branch reaches 84.4%. The efect of the initial batch also decays rapidly. The high-initial-variance group starts 32.8% above the low group, but the gap falls to 9.8% after one step and 2.8% by step 7. At the same point, exploitation hit rates are 92.8% and 92.5% for the low and high groups, respectively. These results show that random cold-start variation afects the first few updates but has little influence on the later task-selection quality.

## 3.4.3 Neighborhood-Size Sensitivity

We evaluate neighborhood-size sensitivity in complete 200-step training runs by independently halving or doubling either neighbor count around the default setting, $K _ { \mathrm { w i t h i n } } = 6$ and $K _ { \mathrm { c r o s s } } = 1 2 .$ As shown in Panel (b) of Table 3, performance remains stable across these settings: the largest absolute deviations from the default are only 0.96, 0.81, and 1.33 percentage points on BFCL-v3, τ-Bench, and ACEBench-Agent, respectively. All tested settings outperform random scheduling on all three benchmarks. This stability indicates that EnvSched’s gains are robust to the choice of neighborhood size within the tested range, with the default providing a reliable operating point without extensive hyperparameter tuning.

## 4 Conclusion

We presented EnvSched, an environment-aware scheduler for reinforcement learning with executable agent environments. By treating environment–task pairs as the unit of data management, EnvSched selects tasks with informative reward diferentiation while maintaining broad environment coverage. Across Qwen3-1.7B, Qwen3-4B, Qwen3-8B, and Qwen3-30B-A3B, it consistently improves performance over random scheduling on BFCL-v3, τ-Bench, and ACEBench-Agent under matched rollout budgets, with gains arising from the complementary efects of reward variance and environment diversity. More broadly, this work first establishes environment-level data management as a useful data-centric perspective for eficient agentic RL, paving the way for scaling increasingly capable agent models with better use of training rollout budgets.

## References

[1] Fei Bai, Huatong Song, Shuang Sun, Daixuan Cheng, Yike Yang, Chuan Hao, Renyuan Li, Feng Chang, Yuan Wei, Ran Tao, Bryan Dai, Jian Yang, Wayne Xin Zhao, and Ji-Rong Wen. Clawgym: A scalable framework for building efective claw agents, 2026. URL https://arxiv.org/abs/2604.26904.

[2] Jiachun Li, Zhuoran Jin, Tianyi Men, Yupu Hao, Kejian Zhu, Lingshuai Wang, Dongqi Huang, Longxiang Wang, Shengjia Hua, Lu Wang, Jinshan Gao, Hongbang Yuan, Ruilin Xu, Kang Liu, and Jun Zhao. Agentic environment engineering for large language models: A survey of environment modeling, synthesis, evaluation, and application, 2026. URL https://arxiv.org/abs/2606.12191.

[3] Bowen Ye, Rang Li, Qibin Yang, Yuanxin Liu, Linli Yao, Hanglong Lv, Zhihui Xie, Chenxin An, Lei Li, Lingpeng Kong, Qi Liu, Zhifang Sui, and Tong Yang. Claw-eval: Towards trustworthy evaluation of autonomous agents, 2026. URL https://arxiv.org/abs/2604.06132.

[4] Minrui Xu, Zilin Wang, Mengyi Deng, Zhiwei Li, Zhicheng Yang, Xiao Zhu, Yinhong Liu, Boyu Zhu, Baiyu Huang, Chao Chen, et al. Envfactory: Scaling tool-use agents via executable environments synthesis and robust rl. arXiv preprint arXiv:2605.18703, 2026.

[5] Xiaoshuai Song, Haofei Chang, Guanting Dong, Yutao Zhu, Ji-Rong Wen, and Zhicheng Dou. EnvScaler: Scaling tool-interactive environments for LLM agent via programmatic synthesis. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens, editors, Findings of the Association for Computational Linguistics: ACL 2026, pages 8326–8357, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-395-1. doi: 10.18653/v1/2026.findings-acl. 407. URL https://aclanthology.org/2026.findings-acl.407/.

[6] Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. Agent world model: Infinity synthetic environments for agentic reinforcement learning. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id= OvLnVkV14P.

[7] Guanting Dong, Junting Lu, Junjie Huang, Wanjun Zhong, Longxiang Liu, Shijue Huang, Zhenyu Li, Yang Zhao, Xiaoshuai Song, Xiaoxi Li, Jiajie Jin, Yutao Zhu, Hanbin Wang, Fangyu Lei, Qinyu Luo, Mingyang Chen, Zehui Chen, Jiazhan Feng, Ji-Rong Wen, and Zhicheng Dou. Agent-world: Scaling real-world environment synthesis for evolving general agent intelligence, 2026. URL https: //arxiv.org/abs/2604.18292.

[8] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[9] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute eficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

[10] Aohan Zeng, Xin Lv, Zhenyu Hou, Zhengxiao Du, Qinkai Zheng, Bin Chen, Da Yin, Chendi Ge, Chenghua Huang, Chengxing Xie, et al. Glm-5: from vibe coding to agentic engineering. arXiv preprint arXiv:2602.15763, 2026.

[11] Xiaomi MiMo Team. Mimo-v2.6-pro-rl. https://huggingface.co/XiaomiMiMo/MiMo-V2. 6-Pro-RL, 2026.

[12] Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547, 2025.

[13] Yuanjie Lyu, Chengyu Wang, Lei Shen, Jun Huang, and Tong Xu. Mock worlds, real skills: Building small agentic language models with synthetic tasks, simulated environments, and rubric-based rewards.

In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12529–12545, 2026.

[14] Yulin Luo, Ruichuan An, Bocheng Zou, Yiming Tang, Jiaming Liu, and Shanghang Zhang. Llm as dataset analyst: Subpopulation structure discovery with large language model. In European Conference on Computer Vision, pages 235–252. Springer, 2024.

[15] Hao Liang, Xiaochen Ma, Zhou Liu, Zhen Hao Wong, Zhengyang Zhao, Zimo Meng, Runming He, Chengyu Shen, Qifeng Cai, Zhaoyang Han, et al. Dataflow: An llm-driven framework for unified data preparation and workflow automation in the era of data-centric ai. arXiv preprint arXiv:2512.16676, 2025.

[16] Yifan Sun, Jingyan Shen, Yibin Wang, Tianyu Chen, Zhendong Wang, Mingyuan Zhou, and Huan Zhang. Improving data eficiency for llm reinforcement fine-tuning through dificulty-targeted online data selection and rollout replay. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 173449–173476. Curran Associates, Inc., 2025. doi: 10.52202 085713-5770. URL https://proceedings.neurips.cc/paper\_files/paper/2025/file/ fd54731cc635e2ed6033e8a0e193eddf-Paper-Conference.pdf.

[17] Yixiu Mao, Yun Qu, Qi Wang, Heming Zou, and Xiangyang Ji. Dynamics-predictive sampling for active rl finetuning of large reasoning models. In International Conference on Learning Representations, volume 2026, pages 25242–25270, 2026.

[18] Shuang Liang, Haoyang Zhou, Yifan Gong, Guowei Wang, and Xiting Wang. Leeps: Latent-guided explore-exploit prompt sampling for eficient rlvr in large language models, 2026. URL https:// arxiv.org/abs/2607.28077.

[19] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

[20] Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaf, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 48371–48392. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

[21] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ-bench: A benchmark for Tool-Agent-User interaction in real-world domains. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 9965–10017, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 1b126cc38b8638e07bef37e7b2bb72bf-Paper-Conference.pdf.

[22] Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Yuefeng Huang, Xiangcheng Liu, Wang Xinzhi, and Wu Liu. ACEBench: A comprehensive evaluation of LLM tool usage. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng, editors, Findings of the Association for Computational Linguistics: EMNLP 2025, pages 12970–12998, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1 2025.findings-emnlp.697. URL https://aclanthology.org/2025.findings-emnlp.697/.

[23] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[24] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[25] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

[26] Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, et al. Wow: Towards a world omniscient world model through embodied interaction. arXiv preprint arXiv:2509.22642, 2025.

[27] Yulin Luo, Chun-Kai Fan, Menghang Dong, Jiayu Shi, Xiangju Mi, Mengdi Zhao, Bo-Wen Zhang, Cheng Chi, Jiaming Liu, Gaole Dai, et al. Robobench: A comprehensive evaluation benchmark for multimodal large language models as embodied brain. In European Conference on Computer Vision, pages 501–520. Springer, 2026.

[28] Ruichuan An, Sihan Yang, Ziyu Guo, Wei Dai, Zijun Shen, Haodong Li, Renrui Zhang, Xinyu Wei, Guopeng Li, Wenshan Wu, et al. Genius: Generative fluid intelligence evaluation suite. arXiv preprint arXiv:2602.11144, 2026.

[29] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, juncai liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, Main Conference, pages 113222–113244. Curran Associates, Inc., 2025. doi: 10.52202/085713-3775. URL https://proceedings.neurips.cc/paper\_files/paper/2025/ file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf.

[30] Woojeong Kim, Ziyi Yang, Jing Nathan Yan, and Jialu Liu. Spend your rollouts where it counts: Rollout allocation for group-based rl post-training, 2026. URL https://arxiv.org/abs/2605.26606.

[31] Zhizhao Liu, Zhiliang Tian, Xi Wang, Zhihua Wen, Yihang Xiong, Zhiquan Lai, and Dongsheng Li. Eficient rlvr scheduling via graph-structured online dificulty estimation, 2026. URL https: //arxiv.org/abs/2608.17941.

[32] NextStep Team, Chunrui Han, Guopeng Li, Jingwei Wu, Quan Sun, Yan Cai, Yuang Peng, Zheng Ge, Deyu Zhou, Haomiao Tang, et al. Nextstep-1: Toward autoregressive image generation with continuous tokens at scale. arXiv preprint arXiv:2508.10711, 2025.

[33] Ruichuan An, Kai Zeng, Ming Lu, Sihan Yang, Renrui Zhang, Huitong Ji, Hao Liang, and Wentao Zhang. Concept-as-tree: A controllable synthetic data framework makes stronger personalized vlms. In European Conference on Computer Vision, pages 474–491. Springer, 2026.

## A Related Work

## A.1 Executable Environments for Agent Training

Tool-use benchmarks provide executable settings for evaluating agents across function calling and multi-turn interaction. BFCL, τ-Bench, and ACEBench assess complementary aspects of tool invocation, user interaction, and task completion [20–22]. Recent environment-synthesis systems extend such interaction to diverse training domains. EnvFactory introduces 85 executable environments with 953 task instances for tool-using agents [4]. EnvScaler [5] programmatically constructs environment skeletons, initial states, task scenarios, and terminal-state validation functions, producing 191 environments and approximately 7,000 scenarios. Agent World Model [6] scales to 1,000 code-driven, database-backed environments with MCP tool interfaces and code-augmented reward verification. Agent-World [7] combines environment–task discovery with multi-environment RL and a self-evolving arena that diagnoses capability gaps and synthesizes targeted tasks. Related work also studies embodied interaction and broad evaluation suites, including world-model interaction, embodied-agent benchmarking, and generative fluid-intelligence evaluation [26–28]. These systems expand the supply of executable environments and verifiable tasks for agent training. EnvSched addresses their use during training by organizing environment–task relationships and selecting informative, diverse tasks from the available pool.

## A.2 Task Informativeness Estimation and Online Selection

Online task selection adapts the training distribution to the evolving policy by estimating task informativeness from reward variation, historical feedback, and task relationships. Recent technical reports on Agentic RL also treat asynchronous rollout, multi-task mixing, and environment scaling as central data-scheduling problems [10, 11]. In group-relative RL, rollout reward diferences identify informative tasks [19]; DAPO [29] filters uniformly successful or unsuccessful groups and resamples prompts, while Pilot-Commit [30] uses reward variation in a pilot sample to allocate later rollouts. For binary rewards, this informativeness is captured by variance $p ( 1 - p )$ . DOTS [16] selects moderately dificult prompts using reference rollouts and an attention-based dificulty predictor, and Dynamics-Predictive Sampling (DPS) [17] models prompt-solving dynamics from historical rollout outcomes to prioritize informative prompts before rollout generation. LEEPS [18] combines representation-space neighbors, historical outcomes, and explore–exploit portfolios to identify prompts likely to yield nonzero reward variance without extra rollouts, while graph-structured online dificulty estimation [31] propagates feedback over semantic and reasoning similarities through coupled latent dificulty states and a state-level Beta–Binomial model. Stable post-training recipes and controllable synthetic-data frameworks provide complementary evidence that data composition and training procedure jointly shape downstream capability [32, 33]. Building on these foundations, EnvSched incorporates executable environment structure through a dual-relation graph: within-environment edges capture shared tools, database fields, and verifier constraints, while cross-environment edges capture abstract execution and reward topology. These predictions guide task selection with explicit environment coverage, modeling both individual task informativeness and batch-level interaction patterns.

## B Complete Historical Estimation

This appendix gives the centering, clipping, and relation-specific neighbor weights used in Section 2.3. For $y \in \{ v , m , d \}$ , let $b _ { s } ^ { y }$ be the mean of y over the tasks observed in training step s. The current global reference and the centered observation are

$$
b _ { t } ^ { y } = \frac { \sum _ { s < t } w _ { t , s } ^ { ( h ) } b _ { s } ^ { y } } { \sum _ { s < t } w _ { t , s } ^ { ( h ) } } , \qquad \widetilde { y } _ { i , s } = \mathrm { c l i p } _ { I _ { y } } ( y _ { i , s } - b _ { s } ^ { y } + b _ { t } ^ { y } ) ,\tag{6}
$$

where $I _ { v } = [ 0 , 0 . 2 5 ]$ and $I _ { m } = I _ { d } = [ 0 , 1 ]$ . If no step-level reference is available, the initialization reference is computed from the first observed batch. The task-history quantities in Eq. 11 use these centered observations.

For the two edge types, $S ^ { \mathrm { i n } } = S ^ { \mathrm { f u l l } }$ and $S ^ { \mathrm { c r o s s } } = S ^ { \mathrm { a b s } }$ . For relation type $q ,$ define the distance between tasks i and j as $\begin{array} { r } { \delta _ { i j } ^ { q } = \mathrm { c l i p } _ { [ 0 , 1 ] } ( 1 - S _ { i j } ^ { q } ) } \end{array}$ . The adaptive bandwidth and neighbor weight are

$$
\sigma _ { i , t } ^ { q , ( h ) } = \operatorname* { m a x } \Bigl \{ \operatorname* { m e d i a n } _ { j \in { \cal N } _ { t } ^ { q } ( i ) } \delta _ { i j } ^ { q } , 0 . 0 5 \Bigr \} , \qquad a _ { i j , t } ^ { q , ( h ) } = \exp \left[ - \frac { 1 } { 2 } \left( \frac { \delta _ { i j } ^ { q } } { \sigma _ { i , t } ^ { q , ( h ) } } \right) ^ { 2 } \right] \sqrt { \mathrm { c l i p } _ { [ 0 . 1 , 3 ] } ( W _ { j , t } ^ { ( h ) } ) } .\tag{7}
$$

When a relation has no available observed neighbor, its estimate is set to the current global reference $b _ { t } ^ { y }$ . Percentile ranks are computed over the full task pool of N environment–task pairs and use average ranks for ties:

$$
\mathrm { p r } ( z _ { i } ) = \frac { \mathrm { r a n k } _ { \mathrm { a v g } } ( z _ { i } ) - 0 . 5 } { N } .\tag{8}
$$

The empirical quantile function $Q _ { \widehat { v } }$ is computed from the predicted variance values of all N tasks and uses linear interpolation. The complete score in Eqs. 12–13 is recomputed independently for the long and short half-lives, including centering, neighbor aggregation, percentile ranks, and quantile mapping.

## B.1 Observation Statistics and Priority Construction

For the training implementation, each selected pair produces a complete group of K rewards $r _ { i , s , 1 } , \ldots , r _ { i , s , K }$ . The overall reward mean, population variance, and mean-corrected dispersion are

$$
\begin{array} { l l } { \displaystyle { m _ { i , s } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } r _ { i , s , k } , \qquad } } & { \displaystyle { v _ { i , s } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \left( r _ { i , s , k } - m _ { i , s } \right) ^ { 2 } , } } \\ { \displaystyle { d _ { i , s } = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \frac { v _ { i , s } } { \operatorname* { m a x } \{ m _ { i , s } ( 1 - m _ { i , s } ) , 0 . 0 1 \} } \right) . } } \end{array}\tag{9}
$$

These dispersion and clipping definitions use the training pool’s reward scale [0, 1]. Historical observations use the half-life weight

$$
w _ { t , s } ^ { ( h ) } = 2 ^ { - ( t - s ) / h } .\tag{10}
$$

For the centered observations in Eq. 6, task histories are summarized as

$$
W _ { i , t } = \sum _ { s \in \mathcal { H } _ { t } ( i ) } w _ { t , s } ^ { ( h ) } , \qquad \bar { y } _ { i , t } = \frac { \sum _ { s \in \mathcal { H } _ { t } ( i ) } w _ { t , s } ^ { ( h ) } \widetilde { y } _ { i , s } } { W _ { i , t } } ,\tag{11}
$$

Here $\mathcal { H } _ { t } ( i )$ contains earlier observation steps. Within a prediction pass using half-life h, $W _ { i , t }$ and $\bar { y } _ { i , t }$ denote the corresponding $W _ { i , t } ^ { ( h ) }$ and $\bar { y } _ { i , t } ^ { ( h ) }$ . When no history is available, $W _ { i , t } = 0$ and $\bar { y } _ { i , t } = b _ { t } ^ { y }$ Predicted mean and dispersion reconstruct a second variance estimate, which is combined with the direct variance prediction:

$$
f _ { i , t } = \widehat { d } _ { i , t } \widehat { m } _ { i , t } ( 1 - \widehat { m } _ { i , t } ) , \qquad u _ { i , t } ^ { ( h ) } = Q _ { \widehat { v } } [ 0 . 6 5 \mathrm { p r } ( \widehat { v } _ { i , t } ) + 0 . 3 5 \mathrm { p r } ( f _ { i , t } ) ] .\tag{12}
$$

All predicted quantities and the quantile function in this equation are computed independently for each half-life h. The final priority incorporates recent decline:

$$
u _ { i , t } = u _ { i , t } ^ { ( h _ { \mathrm { l o n g } } ) } - \eta _ { \mathrm { d e c l i n e } } \operatorname* { m a x } \Bigl \{ u _ { i , t } ^ { ( h _ { \mathrm { l o n g } } ) } - u _ { i , t } ^ { ( h _ { \mathrm { s h o r t } } ) } , 0 \Bigr \} , \qquad q _ { i , t } = \mathrm { p r } ( u _ { i , t } ) .\tag{13}
$$

## C Scheduling Rules and Cold Start

The coverage phase first restricts the eligible candidates lexicographically by current-step coverage count, cumulative environment selections, and task observation count; it then applies Eq. 5 using the informativeness score and batch similarity penalty. The hard constraints are parameterized by a per-round environment cap $C _ { 1 }$ , a rolling three-step environment cap $C _ { 3 }$ , and a consecutive-task cap $C _ { \mathrm { t a s k } }$ , set to $C _ { 1 } = 4 , C _ { 3 } = 1 1$ , and $C _ { \mathrm { t a s k } } = 2$ in our experiments.

The progress filter is part of candidate eligibility. It uses adjacent observations of the same task as training transitions. A ridge predictor estimates the later mean reward from the earlier mean reward, squared mean reward, variance, training step, observation count, and environment category. Each transition is predicted using only transitions from earlier completed training steps. The task’s residual history is combined with environment and relation-neighbor references. A task with at least three scorable transitions and a non-positive estimated residual is removed from the exploitation candidates. It remains eligible for coverage and can return to exploitation after new observations change its estimate.

The identifier filter is also part of candidate eligibility. It examines high-reward trajectories for identifiers that cannot be obtained from the permitted context or for replacement objects that duplicate existing records. It first replays the saved fixed action sequence and checks that the replayed reward agrees with the stored reward within 10<sup>−4</sup>. It then removes the suspicious action and dependent edits and replays the sequence again. A task is filtered only when the removal lowers the reward. Replay uses the saved actions and does not sample a new model trajectory.

At cold start, EnvSched selects one representative task from each environment using the highest mean within-environment similarity. It fills the remaining slots by combining global novelty and within-environment novelty, with no more than two tasks from each environment in the first batch. This initialization uses static structure only.

## D Static Representation and Runtime Configuration

The task pool contains 2,550 task instances from 51 environments. The default batch contains $B = 6 4$ environment–task pairs, with eight rollouts per pair. We use a fixed coverage fraction $\rho = 0 . 5$ giving 32 coverage slots and 32 exploitation slots. The within-environment and cross-environment neighbor sizes are 6 and 12, respectively; the two relation estimates use fixed equal fusion weights of 0.5 each, and the graph estimate and task-history prior use shrinkage strengths $\lambda _ { g } = \lambda _ { s } = 2$ . The long and short half-lives are $h _ { \mathrm { l o n g } } = 8$ and $h _ { \mathrm { s h o r t } } = 2$ training steps, the recent-decline coeficient is $\eta _ { \mathrm { d e c l i n e } } = 0 . 5$ , the exposure-count scale is $c _ { 0 } = 1 6$ , and the batch similarity penalty is $\lambda _ { \mathrm { s i m } } = 0 . 0 8$ Environment caps are $C _ { 1 } = 4 , C _ { 3 } = 1 1$ , and $C _ { \mathrm { t a s k } } = 2$ . These hyperparameters are fixed across model sizes.

The six view weights are $( 0 . 2 8 , 0 . 0 8 , 0 . 1 6 , 0 . 1 2 , 0 . 1 6 , 0 . 2 0 )$ . The six static views use word unigrams and bigrams, character 3–5-grams, verifier-check text, verifier code structure, initial task context, and the complete operation description. Each view has its own TF–IDF encoder with minimum document frequency 1, a 35,000-feature cap, and sublinear term frequency. The structured operation description is extracted ofline and normalized for the cross-environment representation. Numeric values are normalized into abstract categories so that cross-environment similarity reflects operation and condition types instead of exact identifiers.

## E Cross-Environment Graph Construction Prompt

This appendix provides the complete prompt used to construct cross-environment edges in the dual-relation graph (Section 2.2). Within-environment edges use TF–IDF over concrete features like tool names and verifier code. Cross-environment edges require an abstract representation that removes domain-specific identifiers.

## E.1 What This Prompt Extracts

Given a task specification (instruction, initial state, verifier checks, available tools), the prompt outputs a structured description of what operations the task requires and how they depend on each other. After removing domain-specific identifiers, this structure becomes the cross-environment similarity $S ^ { \mathrm { a b s } }$ in Eq. 1.

The prompt outputs five components:

1. Operations: The atomic actions required to complete the task. Each operation is labeled with a type that describes its efect:

• lookup: retrieve data from the environment (e.g., “find user record by username”)

• update: modify existing records (e.g., “change user status field to active”)

• create: add new entities to the environment (e.g., “create new reservation”)

• delete: remove records

• validate: verify constraints without changing state

• authorize: check permissions

Each operation lists which other operations must complete before it can start. For example, “update\_user\_status depends on lookup\_user because the user ID from lookup is needed as input to update.”

2. Gates: Logical conditions that must be satisfied before operations can proceed. If a gate cannot be satisfied, all operations depending on it are blocked. Three examples illustrate the concept:

• Identifier resolution gate: The task says “update status for user Alice”, but the update operation requires a numeric user ID, not a name. A gate captures this requirement: the user ID must first be obtained by looking up “Alice” in the user table.

• State precondition gate: The task says “assign patient to bed 301”, but the assignment operation requires that bed 301 is currently unoccupied. A gate captures this requirement: the bed’s occupancy status must be checked before assignment can proceed.

• Authorization gate: The task says “delete the billing record”, but the delete operation requires admin privileges. A gate captures this requirement: the agent must have the required permission, which may not be available in the current environment.

Each gate also records its provenance (where the required information comes from):

• task\_literal: the task instruction directly supplies the value (e.g., “update user ID 42”)

• unique\_lookup: a tool call deterministically finds it (e.g., username “alice” matches exactly one user)

• broad\_lookup: multiple records match; the agent must choose (e.g., name “John Smith” matches 3 users)

• prior\_tool\_output: an earlier operation produces it (e.g., create\_user returns the new user ID)

• agent\_generated: the agent must create the value (e.g., generate a password)

• unavailable: no available tool can obtain it

3. Dependencies: Directed edges stating which operations must finish before others can start. For example, if operation B needs the output of operation A as input, then B depends on A. This is recorded as an edge from A to B in the operation graph.

4. Reward groups: A partition of the verifier checks (the success conditions for the task). Verifier checks that are controlled by the same operation or gate belong to the same group, because they will succeed or fail together. Three examples:

• Suppose the verifier has three checks: “user status is active”, “user last\_login timestamp is updated”, and “user notification\_sent flag is true”. If all three fields are written by the same update\_user operation, the three checks belong to one group with relation type shared\_operation.

• Suppose two verifier checks both examine properties of a user record, and obtaining that user record requires satisfying an identifier-resolution gate. If the gate fails, both checks fail. They belong to one group with relation type shared\_gate.

• Suppose one verifier check examines a field that already has the correct value in the initial state, and no operation in the task will modify it. This check is already satisfied before the agent does anything. It belongs to a group with relation type initially\_satisfied.

5. Conflicts: Pairs of operations that have ordering constraints or incompatibilities beyond simple data dependencies. Three conflict types:

• order: Operation A must execute before operation B, even if B does not directly consume A’s output. For example, “verify eligibility must precede commit transaction” ensures that invalid transactions are never committed, even though the commit operation does not use the verification result as a data input.

• mutual\_exclusion: Operations A and B cannot both execute in the same task. For example, “approve request” and “reject request” are mutually exclusive: completing the task requires executing one or the other, but not both.

• destructive: Operation A removes the preconditions needed by operation B. For example, delete\_user makes any subsequent update\_user impossible because the user record no longer exists.

After extraction, we apply a deterministic abstraction procedure to the output:

1. Replace each tool name with its operation type (e.g., get\_user\_by\_name → lookup, update\_user\_status → update)

2. Remove all entity identifiers (user IDs, record keys, table names)

3. Remove all database field names (e.g., status, assignment\_id)

4. Replace literal values with type placeholders (e.g., “active” → <string>, 42 → <int>)

5. Retain the dependency structure, gate types, and group relations

This produces an abstract structure description. For example, a task “update user Alice’s status to active” with concrete extraction [lookup(get\_user\_by\_name, input=Alice), update( set\_user\_status, target=user\_id, value=active)] becomes abstract structure [lookup( type=entity\_by\_name), update(type=entity\_field)] after applying these replacements. The abstract string is TF–IDF encoded to produce $g _ { i }$ in Eq. 1, which computes cross-environment similarity $S _ { i j } ^ { \mathrm { a b s } }$ for graph edges E<sup>cross</sup>.

## E.2 System Prompt

The prompt constrains the model to use fixed vocabularies for operation types, gate types, and group relations. JSON structure is enforced by response\_format={"type":"json\_object"} in the API call.

Æ System Prompt   
You extract factual static topology from tool-agent tasks. You receive no   
rollout, trajectory, action trace, model output, reward observation, success rate,   
or variance. Do not predict any of them and do not assign difficulty,   
probability, capability, or high/low labels.   
◎ Build two connected structures   
from the supplied task, initial state, exact verifier checks, tool catalog,   
and selected real tool implementations:   
1. execution topology: the minimum correct operations, their dependencies,   
and factual prerequisite gates such as identifier provenance, state   
preconditions, ordering, authorization, exact arguments, and missing tool   
affordances;   
2. reward topology: a partition of verifier check IDs into groups that are   
controlled by the same operation or prerequisite gate. Check weights are   
not yours to estimate; downstream code computes them from the exact check   
IDs.   
ø A <sub>g</sub>ate is a <sub>p</sub>rere<sub>q</sub>uisite   
whose resolution can jointly enable or block one or more operations/checks. Do   
not create a gate for every ordinary literal. Distinguish a direct task literal,   
unique lookup, broad lookup, prior tool output, agent-generated value, and   
genuinely unavailable value. Mark reachability unsupported only when the supplied   
tools/implementation establish that no path exists; otherwise use unknown.   
Describe topology, not likely model behavior.   
õ The initial\_state\_summary   
is evidence available to this static analyzer, not information shown to the   
agent. Never mark an identifier direct merely because it appears in that   
summary. From the agent's execution perspective, an identifier is   
task\_literal only when the task supplies the execution-ready value; otherwise   
classify the actual tool path as unique\_lookup, broad\_lookup,   
prior\_tool\_output, inferred, or unavailable. Use the initial state only to   
verify that records, matches, and preconditions exist and to determine whether   
a supplied lookup can resolve them.

## Æ System Prompt

Use these enums exactly:   
operation.family: ['aggregate', 'authorize', 'create', 'delete', 'link', 'lookup',   
'other', 'report', 'schedule', 'transition', 'update', 'validate']   
gate.type: ['actor\_authorization', 'argument\_exactness', 'capacity\_or\_eligibility',   
'cross\_entity\_join', 'generated\_identifier', 'identifier\_resolution', 'ordering',   
'other', 'state\_precondition', 'temporal\_conversion', 'termination\_horizon',   
'tool\_affordance']   
gate.provenance: ['agent\_generated', 'broad\_lookup', 'inferred', 'none',   
'prior\_tool\_output', 'task\_literal', 'unavailable', 'unique\_lookup', 'unknown',   
'wall\_clock']   
gate.reachability: ['ambiguous', 'direct', 'indirect', 'unknown', 'unsupported']   
gate.observability: ['explicit\_result', 'silent', 'state\_query', 'unknown']   
gate.recoverability: ['irreversible', 'retryable', 'unknown']   
reward\_group.relation: ['fixed\_unreachable', 'independent', 'initially\_satisfied',   
'mixed', 'shared\_gate', 'shared\_operation']   
conflict.kind: ['destructive', 'mutual\_exclusion', 'order', 'preservation']

## I Return JSON with ke<sub>y</sub> "tasks".

Every task item must have exactly this structure:

```jsonl
{
"task_id": "...",
"operations": [
{"id":"o0","family":"lookup","description":"...","tool_candidates":["..."],
"depends_on_operations":[],"gate_ids":["g0"],"affected_checks":["c0"]}
],
"gates": [
{"id":"g0","type":"identifier_resolution","description":"...",
"provenance":"unique_lookup","reachability":"indirect",
"observability":"explicit_result","recoverability":"retryable",
"depends_on_gates":[],"affected_operations":["o0"],"affected_checks":["c0"]}
],
"reward_groups": [
{"id":"r0","check_ids":["c0"],"shared_gate_ids":["g0"],
"producer_operations":["o0"],"relation":"shared_gate"}
],
"conflicts": [{"left_operation":"o0","right_operation":"o1","kind":"order"}],
"independent_branches": [
{"id":"b0","operation_ids":["o0"],"check_ids":["c0"]}
],
"uncertainties": ["short factual ambiguity in the supplied static evidence"]
}
```

À Rules:   
Use only supplied check IDs and tool names.   
reward\_groups must partition every verifier check exactly once. Group checks   
only when they share a concrete gate/producer; otherwise use singleton or   
independent groups.   
A check already true in the untouched initial state belongs in an   
initially\_satisfied group unless later required operations can invalidate it.   
Operations describe the shortest correct state-changing/query plan, not   
speculative retries.   
IDs must be local sequential identifiers o0..., g0..., r0..., b0....   
Keep descriptions concise and case-specific. At most 30 operations, 30 gates,   
30 reward groups, 20 conflicts, 20 branches, and 5 uncertainties per task.   
Return JSON only.

## E.3 User Message Format

The user message is structured JSON with no additional natural language. Tasks from the same environment share a shared\_environment object:

## \_ User Message Format

```jsonl
{
"shared_environment": {
"environment": "<environment class name>",
"summary": "<prose summary>",
"state_space": "<state definition>",
"constraints": "<rules>",
"tool_catalog": [
{"name": "...", "description": "...", "required": [...],
"arguments": [...], "signature": "..."}
]
},
"tasks": [
{
"task_id": "<identifier>",
"task": "<instruction>".
"initial_state_summary": {<generated by summarize_initial_state>},
"verifier_checks": [
{"check_id": "c0", "requirement": "...", "initially_true": <bool>,
"verifier_code": "..."}
],
"relevant_tool_implementations": [
{"name": "...", "signature": "...", "documentation": "...",
"implementation": "..."}
]
}
```

## F Qualitative Case Studies

This appendix presents three cases from the Qwen3-4B run, one per subsection. Each case is written as the scheduler saw it: the task, the evidence available before the decision, the prediction that evidence produced, and the eight rollouts that followed. All three are tasks that had never been selected before, so nothing was known about the task itself and every prediction came from other environment–task pairs. Neighbor tables list the representative neighbors each relation retrieved, their similarity, the number of prior rollouts, and the variance each contributes. The deployed estimates use the full neighbor sets $( K _ { \mathrm { w i t h i n } } = 6 , K _ { \mathrm { c r o s s } } = 1 2 )$

## F.1 Within-Environment Edges: env\_141-task\_27 at Step 28

Two tasks in the same environment share tools, database fields, and verifier conditions. When one of them has been rolled out, its outcome is therefore evidence about the other: a mistake in the shared edit sequence costs both of them, and the checks that depend on the same steps fail in both.

The task. InsurancePolicyManagementSystem. Acting as Carolyn Ho, amend two policies: set the active period of policy OXL-44032-F, raise its hospitalization limit to 30,000 and lower its deductible to 400, add an Outpatient item and a Dental item while removing the Dental exclusion and keeping the Elective exclusion; then transfer policy ENL-74522-K to another account, set its active period, and add a Psychiatric exclusion. Sixteen verifier checks. The task has the characteristic policy-editing shape: an authorization check on the acting account, lookups that resolve policy numbers and coverage-item identifiers, paired deletions and creations of coverage items, and field updates whose order matters because the tool rejects a duplicate coverage type.

What the scheduler knew. The task itself had never been selected. Its environment was the most heavily sampled in the pool: 56 prior groups spanning 40 distinct tasks, with mean reward variance 0.055 against a pool-wide reference of 0.026 at this step. All six within-environment neighbors had been rolled out.
<table><tr><td>Within-environment neighbor</td><td>Similarity</td><td>Observations</td><td>Last step</td><td>Variance</td><td>Weight</td></tr><tr><td>task_5</td><td>0.554</td><td>2</td><td>26</td><td>0.0146</td><td>0.125</td></tr><tr><td>task_38</td><td>0.544</td><td>1</td><td>19</td><td>0.0077</td><td>0.086</td></tr><tr><td>task_18</td><td>0.525</td><td>1</td><td>25</td><td>0.0414</td><td>0.107</td></tr><tr><td>task_19</td><td>0.508</td><td>1</td><td>11</td><td>0.0000</td><td>0.056</td></tr><tr><td>task_17</td><td>0.506</td><td>2</td><td>21</td><td>0.1863</td><td>0.118</td></tr><tr><td>task_46</td><td>0.506</td><td>1</td><td>27</td><td>0.1084</td><td>0.112</td></tr></table>

The neighbors are not merely similar-sounding requests. They call the same functions on the same tables as the target, and all of them are subject to the same ordering constraint that the coverage-item tool rejects a duplicate coverage type.
<table><tr><td>Neighbor</td><td>Tools shared with the target</td><td>Checks</td></tr><tr><td></td><td>task_17 (var. 0.186) add_coverage_item, remove_exclusion, update_coverage_limit_or_deductible,</td><td>12</td></tr><tr><td></td><td>update_policy_active_dates task_46 (var. 0.108) add_coverage_item, remove_exclusion, update_coverage_limit_or_deductible,</td><td>8</td></tr><tr><td>task_5 (var. 0.015)</td><td>update_policy_active_dates add_coverage_item, remove_exclusion, update_coverage_limit_or_deductible, update_policy_active_dates</td><td>9</td></tr></table>

The prediction. The within-environment relation averages these six neighbors to 0.05. The crossenvironment relation, which retrieves tasks from other environments by shape, averages to 0.02, and the global reference is 0.03. The two are fused with equal weight (Eq. 4) to 0.04.

The selection. Exploitation phase, batch rank 38, predicted percentile $q = 0 . 9 7 4 7$ , predicted mean reward 0.579 and predicted dispersion 0.194.

What the rollouts showed. (0.1875, 0.1875, 0.1875, 0.1875, 0, 0.750, 0.750, 0.625); mean 0.359, variance 0.078, against a batch mean realized variance of 0.036 at this step. Seven of the eight trajectories form two clear behavioral groups, with one zero-reward exception, and the diference between them is not a matter of degree but of whether the trajectory attempts the edit sequence at all. Four trajectories call only is\_authorized\_user\_for\_policy, one or two calls each, and then stop. They pass three of the sixteen checks: the two authorization checks, and c10, which asks whether the Elective exclusion has been left in place — a check that a trajectory passes by doing nothing. The other thirteen checks fail. The three high trajectories carry out the full sequence of lookups, coverage-item insertions, exclusion removals, and the policy transfer, reaching 12, 12, and 10 of 16 checks, with what remains failing on the second policy’s start date and its transfer. The single zero trajectory makes six calls, including update\_policy\_active\_dates and update\_coverage\_limit\_or\_deductible, and then ends before reaching the verifier.

## F.2 Cross-Environment Edges: env\_190-task\_8 at Step 40

Two tasks in diferent domains can ask for the same kind of work. Reconcile a patient’s allergy list, clean up a member’s mood history, and approve two expense claims have nothing in common in what they are about, but all three ask the agent to identify a record from the request, establish that it is allowed to touch it, and then apply a short list of edits to that record and nothing else. When two tasks are put together that way, they also tend to be hard or easy in the same places, which is what makes one informative about the other despite the diferent subject matter.

The task. ElectronicHealthRecordSystem. As Dr. Marta Fong, reconcile the allergy list of a patient: confirm authorization, locate the patient by demographics, update an existing Sulfa Drugs allergy, remove a Latex entry, add Adhesive Tape and Penicillin entries with exact reaction types and confirmation statuses, and leave no duplicates. Nothing can be edited until the patient has been located and the doctor’s permission has been confirmed, so a single lookup near the start decides whether the rest of the task is reachable at all.

What the scheduler knew. No own history. The environment had been sampled 48 times over 26 distinct tasks, but the four nearest same-environment tasks had observed variances of 0.0000, 0.0123, 0.0032, and 0.0000: the earlier tasks here, which asked for the same kind of working over a patient’s records, had all come out the same way. The four nearest tasks outside the environment sit in a mood tracker, an emergency-response app, an expense system, and a video-conferencing scheduler. They involve diferent records, diferent tools, and diferent verifiers, and none of them is about allergies.

<table><tr><td>Neighbor</td><td>What it asks the agent to do</td><td>Variance</td></tr><tr><td>env_177-task_21</td><td>delete one mood entry, edit a second, add a third</td><td>0.1160</td></tr><tr><td>env_164-task_11</td><td>cancel one session, reschedule another, revise its participants</td><td>0.0830</td></tr><tr><td>env_175-task_35</td><td>report, resolve, retract, and update incidents</td><td>0.0381</td></tr><tr><td>env_161-task_32</td><td>assign an approver, verify a document, advance a claim</td><td>0.0256</td></tr></table>

What these tasks share, despite the diferent subject matter, is the work they ask for: find the record the request names, establish that the agent is allowed to touch it, then change that record in a few specific ways. The target reads the acting user’s rights, locates one patient, and edits that patient’s allergy entries, changing one, removing another, and adding two. env\_177-task\_21 does the same thing to a member’s mood history: one entry deleted, one edited, one added. env\_164-task\_- 11 works on sessions in the same manner, cancelling one and rescheduling another while revising its participant list. env\_161-task\_32 advances a claim by assigning an approver, verifying a document, and moving its status forward, and env\_175-task\_35 reports, resolves, retracts, and updates incidents.

The consequence of this arrangement shows up in the target’s own rollouts. Because the four allergy edits all belong to the same patient and every tool takes that patient identifier, a rollout that cannot resolve the patient cannot make any of the four: five of the eight never do and score zero, while the three that succeed are separated only by which of the edits they finish. The neighbor tasks in the table are put together the same way, and the variance beside each is the variance those tasks actually produced.

This is why these tasks can inform one another even though none of them is about allergies. What is compared is not what the records mean or what the tools are called, but the shape of the work: which record has to be found before anything can happen, what is done to it, and how the task is arranged around it.

The prediction. The cross-environment relation averages these neighbors to 0.04. The withinenvironment relation averages to 0.02 from neighbors whose observed variances are 0.0000, 0.0123, 0.0032, and 0.0000. Those near-zero variances are not missing information: they report that the earlier ElectronicHealthRecordSystem tasks were solved the same way every time. What they cannot report is whether a diferent patient’s allergy list, with diferent entries and diferent conflicts, will be as easy, because every task they summarize is drawn from this one environment. The cross-environment neighbors cover the tasks in other environments that are put together the same way, and their variances — 0.0830, 0.0381, 0.0256, 0.1160 — say how much such a task tends to separate the policy. The two are fused with equal weight to 0.03 (Eq. 4).

The selection. Exploitation phase, batch rank 33, q = 0.9786, predicted mean reward 0.562 and predicted dispersion 0.136.

What the rollouts showed. (0.2143, 0.2143, 0.7857, 0, 0, 0, 0, 0); mean 0.152, variance 0.066, against a batch mean realized variance of 0.038 at this step. All five zero trajectories end before the verifier after hitting the 40-turn limit, and each one dies in the same place: alternating get\_patient\_by\_- demographics and verify\_user\_authorization across 40, 40, 40, 40, and 33 calls without ever obtaining a usable patient identifier or authorization result, so no downstream edit is attempted. The single 0.7857 trajectory clears the identity and authorization steps, calls get\_all\_patients and list\_patient\_allergies, and completes the allergy edits for 11 of 14 checks. The two 0.2143 trajectories reach the verifier having satisfied only the three identity and authorization checks. The failure mode is the one the cross-environment neighbors pointed to: in all five, a single unresolved lookup blocks every downstream check at once.

## F.3 Both Relations Contributing: env\_156-task\_5 at Step 4

The two relations are used together rather than as alternatives. On a task where both find observed neighbors, the fused estimate is what determines where the task lands in the ranking.

The task. BloodDonationAppointmentSystem, early in training. Reschedule two donors’ appointments at their existing clinics, release the vacated slots and block the new ones, release a slot from a cancelled appointment, confirm a fourth donor’s eligibility, and create his appointment. Twelve checks. The shape is the many-lookup one typical of early cold-start batches: the agent must resolve facility names and time slots before it can change any state, so a small number of edits sits behind a long chain of lookups.

What the scheduler knew. No own history, and the environment itself had only five prior groups over four tasks, so its own tasks ofer little. Both relations nonetheless retrieve recorded neighbors.
<table><tr><td>Neighbor</td><td>Relation</td><td>Similarity</td><td>Observations</td><td>Variance</td><td>Weight</td></tr><tr><td>env_156-task_39</td><td>within</td><td>0.450</td><td>1</td><td>0.1073</td><td>0.131</td></tr><tr><td>env_156-task_10</td><td>within</td><td>0.372</td><td>2</td><td>0.1093</td><td>0.184</td></tr><tr><td>env_143-task_47</td><td>cross</td><td>0.630</td><td>1</td><td>0.2013</td><td>0.065</td></tr><tr><td>env_145-task_34</td><td>cross</td><td>0.614</td><td>1</td><td>0.1552</td><td>0.055</td></tr><tr><td>env_157-task_8</td><td>cross</td><td>0.619</td><td>1</td><td>0.1036</td><td>0.063</td></tr></table>

The two within-environment neighbors share this environment’s clinic and time-slot tools. The cross-environment neighbors come from hospital patient management, a review platform, and a licensing board, and share the many-lookup pattern in which a broad lookup by name decides whether anything downstream can proceed.

The prediction. The within-environment relation gives 0.07 and the cross-environment relation 0.10, fusing to 0.09 against a global reference of 0.06 (Eq. 4).

The selection. Exploitation phase, batch rank 40, q = 0.9861, predicted mean reward 0.505 and predicted dispersion 0.387.

What the rollouts showed. (1, 1, 0.9167, 0.9167, 0.0833, 0.0833, 0.0833, 0); mean 0.510, variance 0.202, twice the batch mean realized variance of 0.091 at this step. The two perfect trajectories execute the full schedule in 24 and 26 calls and pass all twelve checks. The two 0.9167 trajectories difer from them only on c9, the creation of the fourth donor’s appointment. The three 0.0833 trajectories carry out the rescheduling blocks and then fail the downstream checks. The zero trajectory spends its 40 turns alternating list\_locations and get\_location\_by\_name without ever resolving a facility name — the same lookup bottleneck the cross-environment neighbors point to.