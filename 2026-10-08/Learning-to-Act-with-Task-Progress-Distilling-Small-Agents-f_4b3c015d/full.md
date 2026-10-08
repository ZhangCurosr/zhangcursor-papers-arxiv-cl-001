# Learning to Act with Task Progress: Distilling Small Agents from Compact Teacher Supervision

Wenxi Gan City University of Hong Kong Shenzhen Loop Area Institute wenxigan2-c@my.cityu.edu.hk

## Abstract

Learning from large-model demonstrations offers a way to train small agents that can complete recurring tasks without calling a large model at every step. A central design choice is what to retain from teacher trajectories that contain reasoning, actions, and information about task progress. We introduce Task-Progress Distillation (TPD), an offline approach that pairs each demonstrated action with a short label describing the current task stage. The student learns these compact targets and selects actions by jointly scoring admissible stage–action pairs, which a deterministic harness executes in the environment. On ALFWorld, a 1.7B student trained with 404 demonstrations achieves 72.4% mean unseen task success with either TPD or action-only supervision, compared with 48.3% for a reasoning-trained student using constrained action selection. Explicit stages provide an additional benefit at 200 demonstrations, improving success from 48.0% to 67.7% over action-only supervision. With more demonstrations, the action-only student closes the gap, and both approaches reach 76.9% at 808 demonstrations. Shared-history analyses link part of TPD’s local advantage to better decisions when moving between subgoals, particularly from object acquisition to processing. These results show that compact supervision can train effective small task agents, while explicit task progress provides additional guidance at an intermediate demonstration budget.

## 1 Introduction

For recurring tasks, a large model can provide demonstrations from which a smaller model learns to act on its own. The aim is for the student to complete new instances without asking the teacher to choose each action. The teacher trajectories in our setting contain reasoning, actions, and feedback that records how the task progresses. With limited demonstrations, the choice of what the student should learn becomes central.

Action imitation already provides a useful foundation. LM-BUTLER shows that action modeling can train effective ALFWorld agents, while FireAct and AgentTuning learn from language-model trajectories and broader agent data Micheli and Fleuret (2021); Chen et al. (2023); Zeng et al. (2024). Other methods separate reasoning and action losses or adjust token updates to change how the student learns from each trajectory Liu et al. (2025, 2026). Online learning offers another source of improvement: the student can collect experience beyond the initial demonstrations Feng et al. (2025); Peng et al. (2026). We keep the student’s demonstrations fixed and compare which training targets best support task execution.

Our approach combines compact supervision with constrained action selection. We train the student to predict the next executable action from the task instruction and action–feedback history. At execution time, the student scores the actions allowed by the environment and selects one to execute. The action-only variant (C) uses this action as its entire decision target. Task-Progress Distillation (TPD, arm B) adds a short label for the current task stage: finding the target, processing it, or delivering it. TPD scores stage–action pairs; a deterministic execution harness sends the selected action to the environment and records its feedback.

Stage labels address a common difficulty in multi-step tasks: moving from one subgoal to the next. Suppose the goal is to heat a cup and place it on a table. Once the cup has been acquired, the next subgoal is to take it to the microwave. An agent can instead keep searching, choosing actions that are allowed but no longer serve the immediate subgoal. Labeling this decision as processing tells the student which subgoal the next action should serve.

Both compact variants learn to complete tasks from a few hundred demonstrations. On ALFWorld, a Qwen3-1.7B student reaches 72.4% mean unseen success from 404 demonstrations with either action-only supervision or TPD; both reach 76.9% with 808 demonstrations. A student trained on reasoning and actions reaches 5.0% at the 404-demonstration budget with free generation, and 48.3% when the same checkpoints use constrained action selection. Constrained selection recovers much of the reasoning-trained student’s performance, but both compact variants remain ahead under their respective training and execution configurations.

The additional benefit of task stages depends on the demonstration budget. With 200 demonstrations, TPD reaches 67.7% unseen success compared with 48.0% for action-only supervision, a gain of 19.7 percentage points. The action-only student catches up in mean unseen success at 404 and 808 demonstrations. We therefore examine both when stages help and which decisions change when the student uses them.

We investigate this question by comparing trajectories and then testing decisions under the same interaction history. In processing tasks, one recurring difference appears after the correct object has been acquired: TPD more often starts the required operation. On selected seen histories, fixing a stage that matches the task’s progress more often leads to an action that advances the task than fixing a mismatched stage. This sensitivity overlaps with some decisions where TPD makes progress and C does not, mainly in a few heating tasks and with results that depend on the scoring rule. Across three training seeds on seen tasks, shuffling stage labels across histories also lowers success relative to aligned labels. Together, the results suggest that task-progress labels help the student learn some transitions between subgoals; with more action demonstrations, C increasingly makes similar choices.

Our contributions are:

• A compact distillation pipeline that combines action-focused supervision, task-stage labels, and admissible-action scoring to train a small agent from fixed teacher demonstrations.

• A comparison of reasoning, action-only, and task-progress configurations across demonstration budgets, showing strong compact-agent performance and an additional stage benefit at 200 demonstrations.

• A behavioral analysis linking part of TPD’s local advantage to stage-sensitive task transitions, with smaller B/C decision differences as demonstrations increase.

## 2 Related Work

Learning agents from demonstrations. LM-BUTLER trains language models to predict expert actions and reports strong ALFWorld results, including a few-demonstration setting Micheli and Fleuret (2021). FireAct distills trajectories produced by language-model agents, and AgentTuning combines agent interactions with general instruction data Chen et al. (2023); Zeng et al. (2024). Our action-only student follows this line of work. It provides a practical agent in its own right and a baseline for measuring the added value of task-stage labels.

These methods also differ in what they ask the student to reproduce. Structured Agent Distillation separates reasoning and action segments, while OPC-SFT changes the updates assigned to lowprobability tokens Liu et al. (2025, 2026). TPD retains the demonstrated action and replaces the long reasoning target with a short task-stage label. Our reasoning baseline uses ordinary completion supervision; SAD and OPC-SFT introduce specialized training objectives Liu et al. (2025, 2026). SAD also describes online teacher queries during training Liu et al. (2025); our student learns from a fixed dataset without further teacher queries.

Learning through additional interaction. A fixed dataset may omit states that a student reaches after making a mistake. Several methods address this by collecting more experience during training. ETO adds student exploration after SFT, $\mathrm { A ^ { 3 } T }$ combines reasoning annotation with self-training, and HSL relabels collected trajectories Song et al. (2024); Yang et al. (2024); Li et al. (2026). HPL constructs preferences over trajectories, steps, and action groups, while SAGE-OPD provides selective teacher assistance during on-policy distillation Gao et al. (2025); Zhou et al. (2026). These method let the training data reflect the student’s own behavior.

Online agent RL uses rewards from interaction to update the policy. GiGPO groups decisions to estimate local learning signals, HiPER assigns credit at different planning levels, and STAPO selectively optimizes steps according to their dependence on the trajectory Feng et al. (2025); Peng et al. (2026); Qi et al. (2026). RLVMR uses verifiable meta-reasoning rewards, and EnvRL adds objectives based on environment dynamics Zhang et al. (2025); Wang et al. (2026a). Our study instead asks what the student can learn from a fixed set of demonstrations, without collecting additional training rollouts. We use the environment to collect teacher demonstrations and evaluate the student, while student weight updates use only the fixed dataset.

Task structure and progress. Task structure can guide learning as well as execution. HiPER separates planning from execution, and BEACON uses milestones to construct rewards and advantages Peng et al. (2026); Wang et al. (2026). TPD uses three task-progress labels as supervised targets and as context when scoring the next action. From History to State studies compact state-based context, while Automata from Agent Traces extracts finite-state structure for prediction and monitoring Xie et al. (2026); Cho et al. (2026). Our student retains the full action–feedback history. The stage identifies the current subgoal; the full history still supplies the objects, locations, and feedback needed to choose an action.

## 3 Method

## 3.1 Overview and Task Interface

The pipeline has an offline training stage and a closed-loop execution stage (Figure 1). Offline, we convert teacher demonstrations into compact stage–action targets and fine-tune the student. During execution, the student scores admissible stage–action pairs. The harness sends the selected action to the environment and appends the action and feedback to the history.

Let g be the goal and

$$
h _ { t } = ( g , o _ { 0 } , a _ { 0 } , o _ { 1 } , \ldots , a _ { t - 1 } , o _ { t } )\tag{1}
$$

the history available before action $a _ { t }$ . The environment provides the admissible action set $\boldsymbol { \mathcal { A } } ( \boldsymbol { h _ { t } } )$ . We form the model input $x _ { t }$ from the history and an instruction specifying the required output format. B and C use the same training histories and demonstrated actions; B additionally predicts a task-stage label.

## 3.2 Compact Decision Targets

The simplest target is the action selected by the teacher. The action-only student learns this target from the goal and history. TPD also predicts the current task stage. For each teacher decision, a frozen annotator assigns

$$
\begin{array} { r } { s _ { t } = f ( g , h _ { t } ) , \qquad s _ { t } \in \mathcal { S } = \{ \mathrm { 1 o c a t e , t r a n s f o r m , d e l i v e r } \} , } \end{array}\tag{2}
$$

where the stages distinguish finding the target, carrying out required processing, and completing delivery. For example, after the agent acquires a cup that still needs heating, the label is transform.

The annotator tracks whether the target is held, whether processing has succeeded, and whether placement has occurred. It updates these flags from successful environment feedback and combines them with the task type. It labels each decision using only feedback available before that action, so the label describes progress at decision time. Appendix A gives the implementation details.

A TPD training target can therefore be as short as

![](images/4f06d36a6cf26845b7bbc9cfcf10e9f774c5f50b2b1a0f6d41682b8b48c71ff5.jpg)  
Figure 1: Offline training and closed-loop execution. (a) A frozen annotator reads each teacherhistory prefix and pairs the current task stage with the demonstrated action for student SFT. (b) The student scores $\mathsf { \bar { 3 } } | \mathcal { A } ( h _ { t } )$ | stage–action candidates, and the harness executes the action from the highest-scoring pair. The next input includes the action and environment feedback; the predicted stage is logged only for analysis.

The action-only target contains the same action without the stage field. Both targets specify the next action; TPD also identifies its task stage.

We train each model with completion-only supervised learning. Given a target sequence $y _ { t } .$ , the objective minimizes the negative log probability of its supervised tokens:

$$
\mathcal { L } ( \theta ) \propto - \sum _ { ( x _ { t } , y _ { t } ) \in \mathcal { D } } \sum _ { j \in \mathcal { M } _ { t } } \log p _ { \theta } ( y _ { t , j } \mid x _ { t } , y _ { t , < j } ) ,\tag{3}
$$

where $\mathcal { M } _ { t }$ contains completion positions and excludes system and user tokens. We apply token-level cross-entropy to the supervised completion tokens. For TPD, the entire completion—stage, action, and template tokens—is trained jointly, with no separate segment weights. Longer reasoning targets devote a smaller fraction of supervised tokens to the action. Training and candidate scoring use the same prompt templates and compact serialization.

## 3.3 Joint Candidate Scoring and the Harness

At execution time, TPD chooses a stage and an action together. For each admissible action, we construct one candidate completion for each of the three stages. Let $y ( s , a )$ be the completion for stage s and action a. The student scores the pair by summing its completion-token log probabilities:

$$
q _ { \theta } ( s , a \mid h _ { t } ) = \sum _ { j = 1 } ^ { | y ( s , a ) | } \log p _ { \theta } ( y _ { j } ( s , a ) \mid x _ { t } , y _ { < j } ( s , a ) ) .\tag{4}
$$

It then selects the highest-scoring pair:

$$
( \hat { s } _ { t } , \hat { a } _ { t } ) = \operatorname * { a r g m a x } _ { s \in S , a \in A ( h _ { t } ) } q _ { \theta } ( s , a \mid h _ { t } ) .\tag{5}
$$

With N admissible actions, TPD scores 3N candidates. The action-only student scores $N _ { \ast }$ , using the same sum of log probabilities without length normalization. Candidate order and tie-breaking are fixed (Appendix A).

The harness executes only $\hat { a } _ { t }$ and appends that action and its feedback to the history. The next decision is made from this updated history, with no teacher call. Predicted stages are logged for analysis but are not carried into the next input. Within a decision, the stage affects the context used to

Table 1: Training targets and execution interfaces. N denotes the current admissible action count. A and $A _ { \mathrm { { c o n s t r } } }$ share weights at the corresponding budget and seed; other trained arms are initialized independently from the same Qwen3-1.7B checkpoint.
<table><tr><td>Arm</td><td>Supervised target</td><td>Execution interface</td></tr><tr><td> $\mathbf { A }$ </td><td>Teacher reasoning and action</td><td>Free generation followed by action parsing</td></tr><tr><td> $A _ { \mathrm { c o n s t r } }$ </td><td>Same checkpoint as A</td><td>Generate reasoning, then score N admissible actions</td></tr><tr><td>B / TPD</td><td>Prefix-aligned stage and action</td><td>Joint scoring over 3N stage-action pairs</td></tr><tr><td>C</td><td>Action only</td><td>Score N action candidates</td></tr><tr><td> $B _ { \mathrm { c o n s t } }$ </td><td>Constant phase field and action</td><td>Score N fixed-stage action candidates</td></tr><tr><td> $B _ { \mathrm { s h u f f e } }$ </td><td>Shuffled stage labels and unchanged actions</td><td>Same 3N candidate structure as B</td></tr></table>

score action tokens and contributes to the joint score. Across decisions, the model receives only the action–feedback history. Execution stops when the environment terminates the episode or after 50 steps.

## 4 Experiments

## 4.1 Experimental Setup

Environment and demonstrations. We evaluate multi-step task execution in ALFWorld Shridhar et al. (2021). The six task families cover picking and placing an object, picking two objects, examining an object under light, and cleaning, heating, or cooling an object. We train Qwen3-1.7B Yang et al. (2025) on successful demonstrations collected from DeepSeek-v4-pro with the teacher\_v2 prompt and temperature 0.7. The four datasets contain 98, 200, 404, and 808 trajectories. The 200-, 404-, and 808-trajectory datasets contain 3,712, 7,007, and 13,618 decision records, respectively. The 808-trajectory dataset retains all original 404 demonstrations and adds 404 more using the planned allocation across task categories.

Training and model selection. Each independently trained model starts from the same Qwen3- 1.7B checkpoint and undergoes three epochs of full-parameter SFT. We use a learning rate of $1 0 ^ { - 5 }$ an effective batch size of 32, and a maximum sequence length of 8,192. Within each budget, the three seeds (42, 1234, and 2026) share one dataset, and the arms use matching histories and demonstrated actions. After each epoch, we measure action agreement on a seen probe containing 697 decisions from 35 tasks. We select the checkpoint with the highest probe action agreement, breaking ties in favor of the earlier epoch. Appendix A provides the full recipe and software versions.

Comparisons. Table 1 separates the training targets from the execution interfaces. A is trained to reproduce the teacher’s reasoning and action, and freely generates up to 500 tokens per decision. If parsing fails, the raw output is sent to the environment and still consumes a step. $A _ { \mathrm { { c o n s t r } } }$ uses the same A checkpoint, generates reasoning, and selects an action by scoring admissible candidates. This comparison measures the effect of changing execution with the weights held fixed. C is trained on action-only targets; B uses TPD’s stage–action targets and joint scoring. The constant-stage and shuffled-stage controls test what the stage labels contribute.

Evaluation sets and metrics. Our main evaluation covers all 134 official valid\_unseen tasks, with a maximum of 50 actions per episode. For validation and diagnosis, we use the 116 of 140 official valid\_seen tasks that a rule-based expert completed within 50 steps. The 35-task checkpoint-selection probe is drawn from these 116 tasks. We therefore use seen results for validation and diagnosis, and the full unseen split for the main performance comparison.

We report task success, steps per episode, and recorded output tokens. Results are means and sample standard deviations over three training seeds unless otherwise stated; each seed is evaluated on the same tasks. Free-generation student evaluation uses temperature zero. We also report open-loop action agreement with a rule-based expert reference. Open-loop agreement compares a model’s next action with the expert action on a supplied history; a different action may still advance the task.

(a) Supervision and execution  
Table 2: Unseen success (%), mean ± sample standard deviation across three training seeds, each evaluated on 134 tasks. Blue and orange identify C and TPD; bold marks the highest observed mean in each column, including ties.
<table><tr><td rowspan="2">Configuration</td><td colspan="4">Number of training demonstrations</td></tr><tr><td>98</td><td>200</td><td>404</td><td>808</td></tr><tr><td>A: reasoning-action</td><td> $0 . 0 \pm 0 . 0$ </td><td> $0 . 0 \pm 0 . 0$ </td><td> $5 . 0 \pm 1 . 7$ </td><td> $1 6 . 4 \pm 1 . 5$ </td></tr><tr><td> $A _ { \mathrm { c o n s t r } }$ </td><td> $1 . 5 \pm 0 . 8$ </td><td> $3 0 . 8 \pm 6 . 5$ </td><td> $4 8 . 3 \pm 4 . 8$ </td><td> $4 8 . 8 \pm 5 . 6$ </td></tr><tr><td>C: action only</td><td> $4 2 . 5 \pm 2 . 0$ </td><td> $4 8 . 0 \pm 1 2 . 1$ </td><td> $7 2 . 4 \pm 1 . 5$ </td><td> $7 6 . 9 \pm 3 . 0$  </td></tr><tr><td> $B _ { \mathrm { c o n s t } }$ </td><td> $3 1 . 6 \pm 4 . 4$ </td><td> $5 1 . 5 \pm 1 1 . 7$ </td><td> ${ \bf 7 2 . 6 \pm 1 . 6 }$ </td><td> ${ \bf 7 7 . 1 \pm 6 . 8 }$ </td></tr><tr><td>B: stage-action (TPD)</td><td> ${ \bf 4 8 . 0 \pm 2 . 8 }$ </td><td> ${ \bf 6 7 . 7 \pm 2 . 4 }$ </td><td> $7 2 . 4 \pm 4 . 5$ </td><td> $7 6 . 9 \pm 3 . 3$ </td></tr><tr><td> $\mathbf { B } - \mathbf { C } \left( \mathbf { p p } \right)$ </td><td>+5.5</td><td> $+ 1 9 . 7$ </td><td>0.0</td><td>0.0</td></tr><tr><td> $\mathbf { B } - B _ { \mathrm { c o n s t } } \left( \mathrm { p p } \right)$ </td><td> $+ 1 6 . 4$ </td><td> $+ 1 6 . 2$ </td><td>-0.2</td><td>-0.2</td></tr><tr><td>Teacher reference</td><td colspan="4">88.1 (118/134; one teacher evaluation configuration)</td></tr></table>

![](images/1c78c6929014538b9d645b8f10c4aefe881f75e9ce929d502a8e01fef243a40d.jpg)

(b) The role of task progress  
![](images/393ee990250678239bb1c91897358d0a1fab0f1957f4b36c8dad042b23046bf3.jpg)  
Figure 2: Compact students perform well, with the largest stage benefit at 200 demonstrations. (a) Constrained selection improves the reasoning-trained student; the action-only configuration achieves higher success still. (b) TPD leads C most clearly at 200 demonstrations; C catches up in mean unseen success at 404 and 808. Error bars show sample standard deviations across three training seeds, and faint points show individual B/C runs.

Experiment scope. The planned main comparison comprises the original A/B/C matrix and the 404-demonstration evaluation of $A _ { \mathrm { { c o n s t r } } } .$ An early inspection of unseen aggregates was documented without changing the original run plan; checkpoint selection used the seen probe. The later data extension, additional $A _ { \mathrm { { c o n s t r } } }$ evaluations, stage controls, and prefix analyses are exploratory. The shuffled-stage control evaluates all three training seeds on seen tasks.

## 4.2 Compact Supervision Trains Effective Small Agents

Small agents trained from a few hundred demonstrations. We first measure how often the compact students complete unseen tasks. With 404 demonstrations, both C and TPD reach 72.4% mean unseen success; with 808, both reach 76.9% (Table 2). Both variants train on fixed teacher demonstrations and act without teacher assistance during evaluation. The teacher reference reaches 88.1% (118/134), leaving an 11.2-point gap at 808 demonstrations. The strong action-only result shows that compact action supervision already provides a useful agent before adding task-stage labels.

Table 3: Execution at 404 demonstrations on unseen tasks. Output totals are divided by 402 episodes; mean steps are averaged from the reported per-seed means. Token counts exclude the additional work of scoring candidates.
<table><tr><td>Arm</td><td>Success (%)</td><td>Steps / episode</td><td>Output tokens / episode</td></tr><tr><td>A</td><td>5.0</td><td>47.9</td><td>14,635</td></tr><tr><td> $A _ { \mathrm { c o n s t r } }$ </td><td>48.3</td><td>33.6</td><td>9,995</td></tr><tr><td>B/TPD</td><td>72.4</td><td>22.0</td><td>299</td></tr><tr><td>C</td><td>72.4</td><td>22.2</td><td>213</td></tr></table>

Constrained execution and the remaining performance gap. We first hold the reasoning-trained student’s weights fixed and change how it selects actions. At 404 demonstrations, A reaches 5.0% success and frequently produces outputs that fail parsing. Using the same checkpoints with $A _ { \mathrm { { c o n s t r } } }$ raises success to 48.3%, a gain of 43.3 points. At 98 demonstrations, about 99% of A’s steps lack a parseable action, and all episodes reach the 50-step limit without success. Constrained selection raises success to only 1.5% at this budget. Changing the execution interface alone is therefore insufficient for this checkpoint configuration.

At the same 404-demonstration budget, C and TPD each exceed $A _ { \mathrm { { c o n s t r } } }$ by 24.1 percentage points. At 808 demonstrations, the same separation holds: A reaches 16.4%, $A _ { \mathrm { { c o n s t r } } }$ reaches 48.8%, and both compact variants reach 76.9%. More demonstrations improve A from 5.0% to 16.4%, although its reported parse-failure rate remains 9.5–11.3% across seeds at 808 demonstrations. It also records about 12,000 output tokens per episode, more than the other configurations.

Output volume during execution. At 404 demonstrations, TPD averages 22.0 steps and 299 output tokens per episode, compared with 33.6 steps and 9,995 tokens for $A _ { \mathrm { { c o n s t r } } }$ (Table 3). C averages 22.2 steps and 213 tokens. The ratio of $A _ { \mathrm { { c o n s t r } } }$ to TPD output tokens is 33.4:1. These counts describe trajectory text; they exclude the computation required to score B’s 3N or C’s N candidates at each decision.

Reported results and training resources. Figure 4 plots reported success against demonstration count for selected offline-supervised agents. LM-BUTLER reaches 95% with a 345M model and 3,553 expert demonstrations Micheli and Fleuret (2021). ETO’s SFT stage reaches 67.2% with a 7B model and 3,321 demonstrations containing expert actions and post-hoc reasoning annotations Song et al. (2024). Our compact students reach 72.4% with a 1.7B model and 404 LLM-teacher demonstrations. These points show the model and demonstration sizes associated with the reported results; training and evaluation protocols differ across papers. Appendix C lists the resource details and includes methods whose actual demonstration counts remain unresolved. Appendix C.1 adds the HSL paper’s SFT results at 800 and 1,600 demonstrations Li et al. (2026).

## 4.3 Stage Benefits across Demonstration Budgets

The action-only student already performs well with 404 demonstrations. We next compare B and C across all four budgets to measure the added benefit of task stages (Figure 2(b)). The largest observed gain is at 200 demonstrations: B reaches 67.7% mean unseen success and C reaches 48.0%, a difference of 19.7 percentage points. All three seeds favor B, by 29.9, 19.4, and 9.7 points for seeds 42, 1234, and 2026.

At 98 demonstrations, the gain is smaller at 5.5 points; at 404 and 808, the mean gap is zero. Among the tested budgets, stages provide the largest additional benefit at 200 demonstrations; C matches B’s mean unseen success with more action demonstrations.

The paired outcomes make the difference concrete (Table 4). At 200 demonstrations, B alone succeeds in 89 of the 402 seed–task evaluations, whereas C alone succeeds in ten. At 808 demonstrations, each alone succeeds in 25 cases. Equal mean success at 808 demonstrations therefore does not mean that the two policies succeed on exactly the same evaluations.

On the seen diagnostic set, B leads C by 6.0, 13.8, 7.2, and 2.0 percentage points at 98, 200, 404, and 808 demonstrations, respectively. The catch-up result refers specifically to unseen means. We next test whether the stage benefit depends on labels that match the current task progress.

Table 4: Paired B/C unseen outcomes over 134 tasks evaluated with three training seeds (402 seed– task records per budget).
<table><tr><td>Demonstrations</td><td>Both succeed</td><td>B only</td><td>C only</td><td>Both fail</td></tr><tr><td>200</td><td>183</td><td>89</td><td>10</td><td>120</td></tr><tr><td>808</td><td>284</td><td>25</td><td>25</td><td>68</td></tr></table>

Table 5: Stage-alignment controls at 200 demonstrations: success (%) on the 116 seen diagnostic tasks. Results are shown for three training seeds and their mean; bold marks the highest value in each column.
<table><tr><td>Arm</td><td>Seed 42</td><td>Seed 1234</td><td>Seed 2026</td><td>Mean</td></tr><tr><td>B/ TPD</td><td>55.2</td><td>53.4</td><td>56.9</td><td>55.2</td></tr><tr><td>C</td><td>35.3</td><td>37.9</td><td>50.9</td><td>41.4</td></tr><tr><td> $B _ { \mathrm { c o n s t } }$ </td><td>31.9</td><td>49.1</td><td>50.0</td><td>43.7</td></tr><tr><td> $B _ { \mathrm { s h u f f e } }$ </td><td>31.0</td><td>42.2</td><td>39.7</td><td>37.6</td></tr></table>

## 4.4 The Value of Aligned Stage Labels

Constant-stage supervision. B adds a stage field to C’s action target and changes candidate scoring to include that field. We first replace every stage value with the constant word phase, giving $B _ { \mathrm { c o n s t } } .$ At 200 demonstrations, this model reaches 51.5% unseen success, compared with 67.7% for B and 48.0% for C. The constant field does not reproduce B’s advantage at this budget. At 98 demonstrations, $B _ { \mathrm { c o n s t } }$ reaches 31.6%, below B and C; at 404 and 808, the three means are close.

B and $B _ { \mathrm { c o n s t } }$ have identical target lengths for every 200-demonstration record: each stage value occupies one token, and the mean completion length is 13.69 tokens. The constant control also removes label variation, changes the stage instruction, and reduces the candidate set from 3N to N. The shuffled-stage control keeps all three labels and B’s candidate structure, but changes the labels assigned to individual histories.

Stage alignment during training. $B _ { \mathrm { s h u f f e } }$ permutes stage labels among decision records within each task type. Histories, demonstrated actions, stage counts, and the 3N candidate structure remain unchanged. All three training seeds use the same shuffled dataset, constructed with permutation seed 20261006. Because labels can be permuted onto records with the same original value, 62.55% of shuffled labels still match the originals.

Across the 116 seen tasks, B averages 55.2% success and $B _ { \mathrm { s h u f f i e } }$ averages 37.6% (Table 5). The difference is 17.5 percentage points when computed from unrounded success counts. B leads in all three seeds; C and $B _ { \mathrm { c o n s t } }$ reach 41.4% and 43.7%. Keeping the label vocabulary and candidate structure is therefore insufficient to retain B’s seen-set performance when the labels are shuffled. The result supports using labels that match the current task progress. The comparison also includes the greater difficulty of predicting shuffled labels from the history.

Appendix B reports the additional open-loop diagnostics and replay-annotation coverage.

## 4.5 Task Progress and the Next Decision

At 200 demonstrations, B outperforms $B _ { \mathrm { c o n s t } }$ on unseen tasks and $B _ { \mathrm { s h u f f i e } }$ on seen tasks. We now examine what the agent does differently. We first locate behavioral differences in complete trajectories, then hold the history fixed to compare the next action.

Starting processing after object acquisition. At 200 demonstrations, B’s unseen gain over C is largest for heating (+36.2 percentage points), followed by simple picking (+22.2) and cleaning (+20.4). The gains for picking two objects, looking under light, and cooling are 13.7, 13.0, and 7.9 points. We use environment feedback to identify when an agent acquires the target, reaches a facility, and performs the required processing.

Table 6: Successful processing after correct acquisition, restricted to pairs in which both agents acquire the target (200 demonstrations, unseen). Task and training seed are paired; the acquisitiontime environment states are not identical.
<table><tr><td>Task</td><td>Paired records</td><td>B processes successfully</td><td>C processes successfully</td></tr><tr><td>Heat</td><td>52</td><td>49/52 (94.2%)</td><td>28/52 (53.8%)</td></tr><tr><td>Clean</td><td>60</td><td>59/60 (98.3%)</td><td>51/60 (85.0%)</td></tr><tr><td>Cool</td><td>49</td><td>44/49 (89.8%)</td><td>45/49 (91.8%)</td></tr></table>

The heating trajectories show where the policies begin to differ after acquisition. After acquiring the correct object, B reaches the microwave in 56/58 cases, compared with 32/56 for C. It successfully heats the object in 55/58 cases, compared with 30/56 for C. Neither arm had reached the microwave before acquisition in these episodes. In these episodes, a substantial difference appears between acquiring the object and initiating the required processing.

We then restrict the comparison to seed–task pairs in which both agents acquired the correct object (Table 6). B completes heating in 49/52 records and C in 28/52. Cleaning shows a smaller gap, and cooling shows little difference. All heating and cleaning attempts in these cohorts succeed when issued, so the main difference at this point is whether the agent initiates processing. The two agents may still have different histories and remaining steps when they acquire the object. We address this difference by comparing actions on shared histories below.

The policies also differ before processing begins. C also more often picks up the wrong type of object across several task families. In failed simple-picking episodes, the target never appears in the observation history in 11/24 C cases and 3/8 B cases. Loops of at least five identical consecutive actions occur at similar overall rates (22.6% for B and 21.9% for C), but in 90/130 B failures and 88/209 C failures. The gap therefore includes both failure to find or acquire the right object and failure to move on to processing it.

Comparing actions on shared histories. We evaluate B and C on 72 frozen seen prefixes from 50 tasks. Each prefix ends just before an action, giving both models the same interaction history and admissible actions; only their output-format instructions differ. We test the 200-, 404-, and 808- demonstration checkpoints for all three seeds. The pool contains 24 pilot prefixes and 48 additional prefixes sampled without consulting the new scores.

We use two tests to examine how B’s stage output affects action selection. First, we keep the stage in the candidate text but sum only the action-value token scores, excluding stage and JSON-structure scores. We call this action-value-only scoring. Second, we fix each possible stage in turn and compare the highest-scoring actions under those stage conditions. The first test changes which tokens contribute to the score; the second changes the stage on which the action is conditioned.

Removing the non-action score terms changes B’s selected action in 86/648 evaluations (13.3%), across 72 prefixes, three budgets, and three seeds. Fixing different stage values changes the top action in 130/648 (20.1%). A blinded review of 36 actions changed by score masking judges 11 changes helpful, six harmful, and 19 undetermined. Both the stage condition and the score terms can change a choice; the blinded review shows why a changed choice need not be a better one. On the same prefix set, B/C action agreement rises from about 83% at 200 demonstrations to about 90% at 404 and 808 across the three seeds.

Local progress under fixed stage conditions. We next test whether matching the stage to the task’s progress improves the selected action. For 38 prefixes from 30 tasks, the reference stage is reliable and the rules identify at least one admissible action that makes progress. Using only the current prefix, the rules classify candidate actions as PROGRESS, COUNTERPRODUCTIVE, or UNDETERMINED. For each B checkpoint, we select an action with the matching stage fixed, then repeat with each of the two mismatched stages. All conditions use action-value-only scoring.

At 200 demonstrations, fixing the matching stage yields a progress action in 83.3% of evaluations, compared with 65.4% averaged over the two mismatched stages. Computed from unrounded counts, the gap is 18.0 percentage points, with positive differences for all three seeds (14.5, 18.4, and 21.1 points). Giving each task equal weight yields a 16.3-point gap. The clearest differences occur in heating and some visible-target acquisition decisions, while cleaning and cooling subsets are already near saturation. These rules cover identifiable progress actions, but do not evaluate how well the agent searches for targets it has not yet observed.

Table 7: Local progress and stage sensitivity at 200 demonstrations: 38 seen prefixes evaluated with three training seeds. Overlap fractions use the 15 B-only progress cases as their denominator. Coverage and scoring checks show how many tasks support the overlap and whether it persists under both scoring rules.
<table><tr><td>Criterion or coverage</td><td>Result</td></tr><tr><td>Both original policies choose progress</td><td>81/114</td></tr><tr><td>Only original B chooses progress</td><td>15/114</td></tr><tr><td>Only original C chooses progress</td><td>1/114</td></tr><tr><td>Neither original policy chooses progress</td><td>17/114</td></tr><tr><td>B-only cases with useful matching stage and sensitivity to a mismatch</td><td>11/15</td></tr><tr><td>Also require original B to select reference stage and matching action</td><td>11/15</td></tr><tr><td>Also require full-joint and global action-value-only actions to agree</td><td>2/15</td></tr><tr><td>Unique coverage of the 11 strictly aligned cases</td><td>5 prefixes; 3 tasks</td></tr><tr><td>Heating/transform cases within those 11</td><td>9/11</td></tr></table>

![](images/05be28d9701cb9bc6751f5a4f849807cef66788638a2193401d406bc50fbdda9.jpg)  
(a) Choosing the microwave after acquiring the cup.

![](images/eb2f6a6d220db22ca8cdebedc208c17c88d9aa0af295c043a2b1fcd33f662504.jpg)  
(b) Local progress under fixed stage conditions.  
Figure 3: Moving from object acquisition to processing. (a) On the same history after acquiring a cup, B chooses the microwave at all nine checkpoints. C chooses it at one of three 200-demonstration checkpoints and all six larger-budget checkpoints. (b) Across 38 rule-evaluable seen prefixes from 30 tasks, fixing the matching stage more often yields a progress action than fixing a mismatched stage. Curves show local progress rates averaged over three seeds at each budget. Table 7 relates these tests to the original B/C choices.

Figure 3 illustrates the decision change with a shared history from the cup task. On the same history immediately after acquisition, all three B checkpoints at 200 demonstrations choose the microwave, while two of the three C checkpoints continue searching. All six C checkpoints at 404 and 808 demonstrations choose the microwave. Across the 38 evaluable prefixes, the matchingstage advantage narrows as demonstrations increase: progress under mismatched stages rises, while progress under the matching stage falls. The example shows C learning to make a choice that B already makes at 200 demonstrations; the aggregate test shows a smaller effect of fixing the stage at larger budgets.

Overlap with B’s local advantage. The fixed-stage tests show which action B selects when we supply a stage. We now compare B’s normal joint selection with C’s action-only selection on the same 38 prefixes. At 200 demonstrations, both choose a progress action in 81 of the 114 prefix–seed evaluations, only B in 15, only C in one, and neither in 17. B’s local progress rate is 84.2%, compared with 71.9% for C; the three seeds contribute five, six, and four B-only cases.

![](images/7f9630b93fa1df0309c6f78d186e37046707d133b5b6fe05975eec04f8c5c2a9.jpg)  
Figure 4: Reported success and demonstration counts for offline-supervised agents. External points show LM-BUTLER, ETO’s SFT stage, and the 800- and 1,600-demonstration SFT baselines reported in HSL Micheli and Fleuret (2021); Song et al. (2024); Li et al. (2026). Our B/C results are three-seed means; their markers overlap at 404 and 808 demonstrations. Each work uses its own student model, demonstration source, and evaluation protocol. Appendix C.1 lists the plotted values and their sources.

In 11 of those 15 B-only cases, the matching-stage action makes progress and at least one mismatchedstage action does not. In all eleven cases, B’s normal joint selection already chooses the matching stage and the same progress action. Part of B’s local advantage therefore overlaps with decisions that are sensitive to the stage condition. The eleven cases come from five prefixes and three tasks; nine cases involve heating at the transform stage. Only two of the eleven cases preserve B’s selected action when switching from full-joint to global action-value-only scoring (Table 7). The overlap is therefore sensitive to the scoring rule.

Useful stage sensitivity occurs in 73% of B-only cases, 21% of cases where both policies select progress, and 6% of cases where neither does. In 16 of the 17 cases where neither policy selects progress, B still selects a non-progress action when the matching stage is fixed. Fixing the stage alone therefore does not resolve these local failures. Across 200, 404, and 808 demonstrations, B-only cases decrease from 15 to seven to two, and stage-sensitive cases from 29 to 16 to nine.

The results suggest that task-stage supervision helps with some transitions between subgoals and that the stage condition can guide the next action at those points. With more demonstrations, C agrees more often with B on the sampled histories, while their mean unseen success rates converge. These local tests identify decisions consistent with this explanation; determining their contribution to episode success would require closed-loop interventions.

## 5 Limitations

The experiments cover one environment and one student model size. ALFWorld supplies admissible actions, and the student receives the full interaction history. The annotator supplies task knowledge through hand-defined progress rules. Using the method in another environment would require suitable progress rules and an interface for constructing admissible actions.

Several factors change together in the comparisons. Reasoning and compact configurations differ in target length, the fraction of action tokens, and the reasoning context used by $A _ { \mathrm { { c o n s t r } } }$ during action selection. The constant-stage control changes candidate count, and shuffling changes label predictability. Keeping training at three epochs means that larger datasets increase both demonstration coverage and training computation. Preserving task-category proportions does not control the examples or difficulty within each category. The current results do not separate these contributions.

The three seeds measure training variation on one fixed dataset at each budget. The seen set is filtered by expert success and includes the checkpoint-selection probe, whereas unseen evaluation uses the full split. The shuffle and shared-history analyses are exploratory studies on this seen distribution. The overlap between stage sensitivity and B-only progress covers three tasks and changes with the scoring rule. Measuring its contribution to episode success would require a closed-loop intervention.

Output-token counts measure trajectory text and exclude candidate-scoring computation. End-to-end latency and total training cost remain to be measured.

## 6 Conclusion

We train small agents to execute multi-step tasks from fixed teacher demonstrations using compact decision targets and admissible-action scoring. On ALFWorld, both the action-only student and TPD reach 72.4% mean unseen success with 404 demonstrations and 76.9% with 808 using a 1.7B model. TPD adds its largest observed benefit at 200 demonstrations, raising success from 48.0% to 67.7%; the action-only student matches its mean unseen success at the two larger budgets. Shared-history tests link some of this local advantage to decisions that advance the task, such as moving to the microwave after acquiring an object that needs heating. These results establish compact action supervision as a useful baseline and show when explicit task progress adds value in this setting.

## A Implementation and Reproducibility Details

Stage annotation. The annotator is defined in version 2 of configs/fsm.yaml and implemented in src/annotate\_states.py. It uses the task type and three tracked facts: whether the target is held, whether processing is complete, and whether placement has occurred. After labeling the current decision, it updates these facts from recognized success messages in the environment feedback. Heating, cleaning, and cooling require a transformation. The rules use a fixed object vocabulary and target-inference logic to identify the target. Unsupported action verbs trigger an assertion.

Training recipe. We fine-tune all parameters using bfloat16, scaled dot-product attention, and gradient checkpointing, without LoRA. We use fused PyTorch AdamW with a learning rate of 10<sup>−5</sup>, cosine decay, 10 warmup steps, and zero weight decay. The microbatch size is eight, with four gradient-accumulation steps and effective batch size 32. Sequences are limited to 8,192 tokens. We train for three epochs and evaluate the seen probe after each epoch. The epoch with the highest action agreement is selected; ties favor the earlier epoch. The probe has six tasks from each of five task families and five picking-two tasks, totaling 697 decisions from 35 tasks.

Loss and serialization. The loss covers the completion, with system and user tokens masked out. For TPD, this includes the stage and action values, JSON syntax, the empty thinking block, and the chat-template end-of-message token. There are no separate loss weights for stage and action tokens. Training logs record aggregate completion loss, without separate stage and action loss curves. Dataset construction checks assistant targets against the prompt templates. Candidate scoring uses the same compact serialization. Prefix-reconstruction checks verify that the scoring inputs match the intended histories.

Candidate order. The outer loop follows the environment’s admissible-action order, and the inner loop follows the fixed stage order. The first candidate wins a tie. Scores are sums of completion-token log probabilities, with no length normalization.

Evaluation subset. The rule-based expert completes 116 of the 140 official seen tasks within 50 steps; all 116 form the seen diagnostic set. It contains 35 simple-picking, 26 cleaning, 24 cooling, 14 heating, 12 looking-under-light, and five picking-two tasks. The 35 checkpoint-probe tasks are a subset of this set. Unseen evaluation uses all 134 official tasks, with no expert-success filtering.

Software and model versions. The recorded environment uses ALFWorld 0.4.2, TextWorld 1.7.0, PyTorch 2.11.0, Transformers 5.17, and TRL 1.13.0. Student inference uses a local Hugging Face Transformers service. The common initialization is the post-trained Qwen/Qwen3-1.7B snapshot at revision:

Teacher collection uses deepseek-v4-pro, the teacher\_v2 prompt, temperature 0.7, and full interaction history.

## B Shuffled-Stage Diagnostic Protocol

Free-generation and constrained-execution legality. At seed 42, the shuffled model produces legal outputs on 79.5% of 1,870 open-loop records when evaluated by free generation. An output is counted as legal if it can be parsed, its stage is allowed, and its action belongs to the admissible set. The shuffled rates at seeds 1234 and 2026 are 82.5% and 84.0%; B’s rates at seeds 42, 1234, and 2026 are 84.1%, 83.2%, and 84.5%.

Closed-loop execution selects from candidates built with allowed stages and admissible actions. Those selected actions are legal by construction. The free-generation diagnostic measures how often the model produces such an output without candidate selection.

Replay annotation coverage. At seed 42, the annotator can label 76/116 B trajectories and 51/116 shuffled-model trajectories. Both can be labeled on 47 tasks, only B on 29, only the shuffled model on four, and neither on 36.

In these runs, the annotator fails when the trajectory never successfully operates on the target object. The annotatable sets therefore depend on the policies’ behavior. The common set for a paired replay comparison contains 47 tasks, whereas the success rates in Table 5 use all 116.

Interpretation of open-loop diagnostics. At seed 42, the shuffled model and B have similar open-loop action agreement (49.1% and 48.4%), despite their different closed-loop success rates. The open-loop test supplies the history and compares one action with a reference. In closed-loop execution, each policy must also handle the histories created by its earlier choices.

The shuffled model’s 71.7% agreement with reference stages is measured during evaluation. The 62.55% agreement between shuffled and original labels describes the training data. Because these percentages use different data and denominators, their difference does not measure recovery of stage knowledge.

## C Comparison with Published Agent-Training Results

Table 8 lists the training resources and evaluation protocols of the methods discussed in the main text. It includes methods whose actual demonstration counts remain unresolved and are therefore absent from Figure 4.

Table 8: Training resources and reported ALFWorld results under each work’s evaluation protocol. Our values are three-seed means; standard deviations appear in Table 2. Trajectory counts and decision-record counts are listed separately.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Student</td><td colspan="3">Training resources</td><td rowspan="2">Evaluation</td><td rowspan="2">Success (%)</td></tr><tr><td>Demos</td><td>Demo steps</td><td>Interaction protocol</td></tr><tr><td colspan="8">Offline SFT / imitation: external reported results</td></tr><tr><td>LM-BUTLER Micheli Fleuret (2021)</td><td>and GPT2-medium (345M)</td><td>42</td><td>NR</td><td>No</td><td>Original paper</td><td>40.0</td></tr><tr><td>LM-BUTLER Micheli Fleuret (2021)</td><td>and GPT2-medium (345M)</td><td>3,553</td><td>NR</td><td>No</td><td>Original paper</td><td>95.0</td></tr><tr><td>ETO: SFT stage Song et al. Llama-2-7B-Chat (2024)</td><td></td><td>3,321</td><td>NR</td><td>No</td><td>Original paper</td><td>67.2</td></tr><tr><td>HSL: SFT baseline Li et al. Llama3.2-1B-Instruct (2026)</td><td></td><td>&gt;3,200</td><td>NR</td><td>No</td><td>Original paper</td><td>78.36</td></tr><tr><td>SFT Liu et al. (2026)</td><td>Qwen2.5-1.5B</td><td>UV</td><td>UV</td><td>No</td><td>EMBod-Bench</td><td>70.90</td></tr><tr><td>OPC-SFT Liu et al. (2026)</td><td>Qwen2.5-1.5B</td><td>UV</td><td>UV</td><td>No</td><td>EMBod-Bench</td><td>72.39</td></tr><tr><td>SFT Liu et al. (2026)</td><td>Qwen2.5-1.5B</td><td>UV</td><td>UV</td><td>No</td><td>AgentBoard</td><td>73.88</td></tr><tr><td>OPC-SFT Liu et al. (2026)</td><td>Qwen2.5-1.5B</td><td>UV</td><td>UV</td><td>No</td><td>AgentBoard</td><td>76.12</td></tr><tr><td colspan="7">Additional environment-interaction training: external references</td></tr><tr><td>ETO Song et al. (2024)</td><td>Llama-2-7B-Chat</td><td>3,321</td><td>NR</td><td>Yes</td><td>Original paper</td><td>72.4</td></tr><tr><td>SFT + HSL Li et al. (2026)</td><td>Llama3.2-1B-Instruct</td><td>&gt;3,200</td><td>NR</td><td>Yes</td><td>Original paper</td><td>97.76</td></tr><tr><td colspan="7">This work: offline SFT with constrained candidate selection</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>200</td><td>3,712</td><td>No</td><td>Our harness</td><td>48.0</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>200</td><td>3,712</td><td>No</td><td>Our harness</td><td>67.7</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>404</td><td>7,007</td><td>No</td><td>Our harness</td><td>72.4</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>404</td><td>7,007</td><td>No</td><td>Our harness</td><td>72.4</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>808</td><td>13,618</td><td>No</td><td>Our harness</td><td>76.9</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>808</td><td>13,618</td><td>No</td><td>Our harness</td><td>76.9</td></tr></table>

Notes. Demos counts demonstration trajectories; Demo steps counts individual decision records. Student interaction records whether training uses environment rollouts beyond the fixed demonstrations; teacher collection and evaluation are excluded. The demonstration counts for interaction-based methods describe their initial supervised data and do not measure total training experience. The SFT + HSL method Li et al. (2026) also uses a 70B relabeler; its SFT-only baseline does not include this additional stage.

External methods differ in teacher source, model, checkpoint selection, prompts, and execution. Our results cover all 134 valid\_unseen tasks with a 50-step limit. External values retain their published precision. Table 9 provides the full four-budget B/C comparison. The remaining internal arms are reported in the main-text tables.

Abbreviations. NR: not reported in the cited source. UV: actual training-data usage has not been verified. For OPC-SFT Liu et al. (2026), the available data-volume accounts do not establish the amount used for each reported run.

Source locations. LM-BUTLER: Table 1 Micheli and Fleuret (2021); ETO: Tables 1 and 7 Song et al. (2024); OPC-SFT: Tables 1 and 9, before RL Liu et al. (2026); HSL: Table 1 and Figure 3 Li et al. (2026).

## C.1 Reported Performance across Demonstration Budgets

A method’s full-data result does not show how well it learns from fewer demonstrations. Table 9 compares the smaller-budget SFT results reported in HSL Li et al. (2026) with LM-BUTLER Micheli and Fleuret (2021), ETO’s SFT stage Song et al. (2024), and our four demonstration budgets. Table 8 provides the corresponding training-resource and evaluation details.

Table 9: Reported ALFWorld unseen success across demonstration budgets. External entries use each paper’s evaluation protocol. Our entries are means over three training seeds; their standard deviations appear in Table 2.
<table><tr><td>Method</td><td>Student</td><td>Demos</td><td>Success (%)</td><td>Source</td></tr><tr><td colspan="5">Published offline-supervised results</td></tr><tr><td>LM-BUTLER, partial Micheli and Fleuret (2021)</td><td>GPT2-medium (345M)</td><td>42</td><td>40.0</td><td>Table 1</td></tr><tr><td>LM-BUTLER, full Micheli and Fleuret (2021)</td><td>GPT2-medium (345M)</td><td>3,553</td><td>95.0</td><td>Table 1</td></tr><tr><td>HSL: SFT baseline Li et al. (2026)</td><td>Llama3.2-1B-Instruct</td><td>800</td><td>55.2</td><td>Figure 3</td></tr><tr><td>HSL: SFT baseline Li et al. (2026)</td><td>Llama3.2-1B-Instruct</td><td>1,600</td><td>64.2</td><td>Figure 3</td></tr><tr><td>HSL: SFT baseline Li et al. (2026)</td><td>Llama3.2-1B-Instruct</td><td>&gt; 3,200</td><td>78.36</td><td>Table 1</td></tr><tr><td>ETO: SFT stage Song et al. (2024)</td><td>Llama-2-7B-Chat</td><td>3,321</td><td>67.2</td><td>Tables 1, 7</td></tr><tr><td colspan="5">This work: offline SFT with constrained action selection</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>98</td><td>42.5</td><td>Table 2</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>98</td><td>48.0</td><td>Table 2</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>200</td><td>48.0</td><td>Table 2</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>200</td><td>67.7</td><td>Table 2</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>404</td><td>72.4</td><td>Table 2</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>404</td><td>72.4</td><td>Table 2</td></tr><tr><td>Action-only (C)</td><td>Qwen3-1.7B</td><td>808</td><td>76.9</td><td>Table 2</td></tr><tr><td>TPD (B)</td><td>Qwen3-1.7B</td><td>808</td><td>76.9</td><td>Table 2</td></tr></table>

Notes. Demos counts training demonstration trajectories. All entries report offline-supervised performance before any additional student exploration or online optimization. The HSL rows refer to its SFT baseline, not the full HSL method Li et al. (2026). HSL’s smaller-budget values come from Figure 3 and its full-budget score from Table 1; the full demonstration count is reported only as > 3,200. Methods with unresolved actual demonstration counts are omitted.

Reference results at smaller budgets. With 808 demonstrations, both of our compact students reach 76.9% mean unseen success; the HSL paper reports 55.2% for its SFT baseline with 800 demonstrations Li et al. (2026). TPD also reaches 67.7% with 200 demonstrations, compared with 64.2% for that baseline at 1,600 demonstrations. The HSL SFT baseline Li et al. (2026) reaches 78.36% at its full budget of more than 3,200 demonstrations. These results provide reference points for learning from a few hundred demonstrations. Because the studies differ in student models, demonstration sources, and execution protocols, these are comparisons of reported systems rather than isolated training methods.

## References

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and others. Qwen3 Technical Report. arXiv:2505.09388, 2025. https://arxiv.org/abs/2505.09388

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. ALFWorld: Aligning Text and Embodied Environmentsfor Interactive Learning. In International Conference on Learning Representations (ICLR), 2021. https://github.com alfworld/alfworld

Vincent Micheli and François Fleuret. Language Models are Few-Shot Butlers. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9312–9318. Association for Computational Linguistics, 2021. https://aclanthology.org/ 2021.emnlp-main.734/

Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. FireAct: Toward Language Agent Fine-tuning. arXiv:2310.05915, 2023. https://arxiv.org/ abs/2310.05915

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, and Jie Tang. AgentTuning: Enabling Generalized Agent Abilitiesfor LLMs. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 3053–3077. Association for Computational Linguistics, 2024. https://aclanthology.org/2024.findings-acl.181/

Jun Liu, Zhenglun Kong, Peiyan Dong, Changdi Yang, Tianqi Li, Hao Tang, Geng Yuan, Wei Niu, Wenbin Zhang, Pu Zhao, Xue Lin, Dong Huang, and Yanzhi Wang. Structured Agent Distillation for Large Language Model Agents. arXiv:2505.13820, 2025. https://arxiv.org/abs/2505. 13820

Tian-Shuo Liu, Chengxing Jia, Haoyu Liu, Pengyuan Wang, Shiyuan Zhang, Jie Fu, and Yang Yu. Clipping Low-Probability Tokens in SFT Yields a Generalizable Initialization for RL. In Proceedings of the 43rd International Conference on Machine Learning (ICML), volume 306 of Proceedings of Machine Learning Research, pages 76871–76896. PMLR, 2026. https: //proceedings.mlr.press/v306/liu26bm.html

Yifan Song, Da Yin, Xiang Yue, Jie Huang, Sujian Li, and Bill Yuchen Lin. Trial and Error: Exploration-Based Trajectory Optimization ofLLM Agents. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7584– 7600. Association for Computational Linguistics, 2024. https://aclanthology.org/2024. acl-long.409/

Zonghan Yang, Peng Li, Ming Yan, Ji Zhang, Fei Huang, and Yang Liu. ReAct Meets ActRe: Autonomous Annotation of Agent Trajectories for Contrastive Self-Training. In First Conference on Language Modeling (COLM), 2024. https://arxiv.org/abs/2403.14589

Heyang Gao et al. Solving the Granularity Mismatch: Hierarchical Preference Learningfor Long-Horizon LLM Agents. arXiv:2510.03253, 2025. https://arxiv.org/abs/2510.03253

Yuhang Zhou, Lizhu Zhang, Yifan Wu, Mingyi Wang, Bo Peng, Jiayi Liu, Xiangjun Fan, and Zhuokai Zhao. SAGE-OPD: Selective Agent-Guided Intervention for Multi-Turn On-Policy Distillation. arXiv:2606.19659, 2026. https://arxiv.org/abs/2606.19659

Zichao Li, Gang Wu, Zichao Wang, Ruiyi Zhang, Wanrong Zhu, Ryan A. Rossi, Vlad I. Morariu, and Jihyung Kil. Spinning Straw into Gold: Relabeling LLM Agent Trajectories in Hindsightfor Successful Demonstrations. In International Conference on Learning Representations (ICLR), 2026. https://arxiv.org/abs/2607.04235

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-Group Policy Optimization for LLM Agent Training. In Advances in Neural Information Processing Systems (NeurIPS), 2025. https://arxiv.org/abs/2505.10978

Jiangweizhi Peng, Yuanxin Liu, Ruida Zhou, Charles Fleming, Zhaoran Wang, Alfredo Garcia, and Mingyi Hong. HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents. In Proceedings ofthe 43rd International Conference on Machine Learning (ICML), 2026. https://arxiv.org/abs/2602.16165

Qiuyi Qi, Tian Liang, Mutian Bao, Jinjian Zhang, Dongnan Liu, Wei Zhou, Linjian Mo, Ming Kong, Jie Liu, Feng Zhang, and Qiang Zhu. STAPO: Selective Trajectory-Aware Policy Optimization for LLM Agent Training. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computa tional Linguistics (Volume 1: Long Papers), pages 28371–28392. Association for Computational Linguistics, 2026. https://aclanthology.org/2026.acl-long.1308/

Zixuan Wang, Yuchen Yan, Hongxing Li, Teng Pan, Dingming Li, Ruiqing Zhang, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Milestone-Guided Policy Learningfor Long-Horizon Language Agents. In Proceedings of the 43rd International Conference on Machine Learning (ICML), 2026. https://arxiv.org/abs/2605.06078

Zijing Zhang, Ziyang Chen, Mingxiao Li, Zhaopeng Tu, and Xiaolong Li. RLVMR: Reinforcement Learning with Verifiable Meta-Reasoning Rewardsfor Robust Long-Horizon Agents. arXiv:2507.22844, 2025. https://arxiv.org/abs/2507.22844

Zhitong Wang et al. EnvRL: Learn from Environment Dynamics in Agentic Reinforcement Learning. arXiv:2606.17680, 2026. https://arxiv.org/abs/2606.17680

Haoyang Xie et al. From History to State: Constant-Context Skill Learning for LLM Agents. arXiv:2605.05413, 2026. https://arxiv.org/abs/2605.05413

Seonglae Cho et al. Automata from Agent Traces: Failure and Next-Step Prediction. arXiv:2608.23670, 2026. https://arxiv.org/abs/2608.23670