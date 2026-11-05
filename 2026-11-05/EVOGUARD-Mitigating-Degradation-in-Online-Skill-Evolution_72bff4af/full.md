# EVOGUARD: Mitigating Degradation in Online Skill Evolution

Yanli Wang<sup>1,</sup> <sup>2,</sup> <sup>3,</sup> <sup>⋄</sup>, Zichen Bai<sup>1,</sup> <sup>⋄</sup>, Tianyue Jiang<sup>2</sup>, Kunming Shao<sup>4</sup>, Junhao Shi<sup>1</sup> Xueqin Liu<sup>1</sup>, Jinguo Sun<sup>1</sup>, Muye Li<sup>1</sup>, Tingting Bi<sup>3</sup>, Yanlin Wang<sup>2</sup> Shaoqiang Zheng<sup>1,</sup> <sup>†</sup>, Xuefei Diao<sup>1</sup>, Zhuwei Zeng<sup>1</sup>, Nan Duan<sup>1</sup>, Peng Cao<sup>1</sup>

<sup>1</sup>JD.Com <sup>2</sup>Sun Yat-sen University <sup>3</sup>The University of Melbourne <sup>4</sup>The Hong Kong University of Science and Technology

## Abstract

Skills let language agents reuse experience across tasks as natural-language documents. Recent work evolves skills from interaction, but existing methods have limitations when skills evolve online. Prior methods update a skill offline from a batch of trajectories collected over many tasks. Some then adopt the new skill without any check, which can degrade the skill over time. Others keep it only if it improves performance on a held-out validation set, which is too slow and costly to run after every update. We introduce EVOGUARD, a pairwise skill selector that mitigates skill degradation at low latency and cost. We first run the agent iteratively to build pools of skills and score each skill on a private test set to obtain utility labels. We pair the skills and use rejection sampling to keep only pairs with a clear utility gap. The selector is trained on the logits of two label tokens to pick the better skill. At deployment it decides in a single forward pass, which is significantly faster than evaluation on the private test set. Compared with adopting every new skill, EVOGUARD improves success by 11.0% on ALFWorld, 5.9% on ScienceWorld, and 2.7% on LiveCodeBench.

## 1 Introduction

Language agents increasingly face streams of related tasks. Solving each task from scratch wastes what earlier interactions have revealed. A skill packages such experience as a reusable naturallanguage document. It records procedures, decision rules, and recovery strategies. A frozen agent can then adapt through external text instead of weight updates (Wang et al., 2024a; Anthropic, 2025; Xu and Yan, 2026). Recent work builds and evolves skills from interaction (Xia et al., 2026; Yang et al., 2026b; Yao et al., 2026). However, existing methods face three limitations when skills evolve online.

In this setting a deployed agent updates its skill after every task.

Gap 1: Offline multi-trajectory evolution does not match deployment. Most methods update a skill offline from a batch of trajectories (Xia et al., 2026; Yang et al., 2026b). The update runs in a training phase before deployment. It analyzes trajectories from many tasks at once. A deployed agent rarely works this way. Each task arrives once and is attempted once. The next task is usually a different one. The skill must therefore be revised from a single trajectory and then reused by later tasks. SkillRise carries a skill across tasks in this way (Yao et al., 2026). However, it trains the executor jointly with skill curation and does not check individual updates.

Gap 2: Adopting every new skill causes degradation. A rewrite based on one trajectory can overfit that trajectory. A failed search may add a rule tied to one room. Even a successful episode may add redundant or misleading advice. This risk is measurable. We collect single-trajectory rewrites on ALFWorld and score every skill on a private test set of its task family (Figure 1).

Figure 1(a) shows that the success rates of these skills spread over a wide range. Skills of very different quality therefore coexist. Figure 1(b) shows the change in success rate after one rewrite, computed as the rewrite minus the skill it came from. The empty skill also counts as a parent. Among the 1,137 rewrites written by the base generator, 42.2% score below their parent. Another 7.7% tie, and 50.0% improve. A rewrite is thus nearly neutral on average. However, 7.7% of rewrites lose at least 10%, and 11.3% gain at least 10%. A stronger generator does not remove the problem. Of the 900 rewrites written by an expert generator, which is trained with reinforcement learning, 40.0% also score below their parent. Adopting every rewrite therefore lets the skill drift. This motivates an explicit acceptance step.

![](images/19197971b94d8a04c33b0f003a5376e7b01491dd41ad5a15a4a08d7b6f8d74a1.jpg)  
Figure 1: (a) Success rate of the base-written and expert-written skills on the private test set for each task family. (b) Change in success rate after a single-trajectory rewrite (child minus parent). The shaded region marks degradation.

Gap 3: Validating every update is too costly. Some methods guard against bad updates (Alzubi et al., 2026; Yang et al., 2026b). They keep a new skill only if it improves performance on a held-out validation set. This requires many extra rollouts for every update. Each check thus costs far more than the task that triggered the update. An online agent cannot afford this cost after every task.

To address these gaps, we propose EVOGUARD, a pairwise skill selector that amortizes validation. Tasks are grouped into task families, and each family keeps one skill. After each task, a fixed generator writes a new skill from the latest trajectory. EVOGUARD then decides whether this new skill replaces the current skill. This decision only requires the relative order of the two skills, not their absolute utilities. We therefore cast it as a pairwise preference problem. We pay the cost of validation once before deployment and use it to train the selector. After that, each decision only compares two skill texts.

EVOGUARD works in three stages (Figure 2). (1) Multi-source skill pool construction. We run the rewrite loop on training tasks with a base generator and an expert generator, and collect the generated skills into a pool. (2) Utility-grounded pairwise preference learning. We estimate the utility of every skill on a private test set of its family. Rejection sampling keeps pairs from the same family with a clear utility gap. We then train a pairwise selector to predict which skill in a pair has the higher utility. (3) Online skill selection. After every task, the frozen selector reads the family description and the two skills. It keeps the skill it predicts to be better. This takes one forward pass and runs no extra rollouts in the environment.

We evaluate EVOGUARD on ALFWorld (Shridhar et al., 2021) with three executors. The baselines include running without skills and adopting every new skill. They also include random selection from the skill history and three LLM-Judge variants. On skill pairs unseen during training, EVOGUARD picks the better skill 95.0% of the time. Online, it outperforms every other method in all six settings. Its average success with the Qwen3-4B and Qwen3- 8B executors is 56.5%. Adopting every new skill reaches 45.5%, and running without skills reaches 36.7%. The Qwen3-32B executor contributes no training labels. With this executor EVOGUARD still reaches 69.2% against 54.9% without skills. On ScienceWorld (Wang et al., 2022), EVOGUARD raises success from 10.2% to 15.3%. Our main contributions are as follows.

• We formulate online skill evolution as a per-update acceptance problem. After each single-trajectory rewrite, the agent must decide whether the new skill replaces the current skill. We show that this decision matters. On ALFWorld, 42% of rewrites lower performance on a private test set.

• We propose EVOGUARD, a pairwise skill selector that amortizes validation. Validation runs once offline to supervise the selector. Each online decision then needs one forward pass over two skill texts. It generates no tokens and runs no extra rollouts.

• EVOGUARD achieves the highest success among all methods on ALFWorld, Science-World, and LiveCodeBench. It also transfers to an executor unseen during selector training.

## 2 Related Work

Skill evolution. Skills package reusable experience as executable code or structured documents (Wang et al., 2024a; Anthropic, 2025; Xu and Yan, 2026). Recent work evolves skills from interaction. SkillRL integrates recursive skill evolution with reinforcement learning (Xia et al., 2026). SkillOpt treats a skill as the external state of a frozen agent and optimizes it under feedback (Yang et al., 2026b). SkillRise is the closest work to ours (Yao et al., 2026). It also carries an evolving skill across distinct tasks and trains the executor jointly with skill curation through reinforcement learning. However, our method keeps both the executor and the generator frozen. It only decides whether a singletrajectory rewrite should replace the current skill. This decision is learned from supervised pairwise labels without policy optimization. EVOGUARD is thus a separate module that can be attached to any rewrite loop.

Guarding skill updates. Several methods check an update before they keep it. EvoSkill and SkillOpt keep an update only if it improves performance on a held-out validation set (Alzubi et al., 2026; Yang et al., 2026b). SkillEvo adds a governance layer that repairs degraded skills instead of only rejecting them (Yan et al., 2026). SkillEvoReg regularizes skill evolution against overfitting and tests each candidate for regressions (Nie et al., 2026). These methods execute additional tasks for every update. However, our method pays the validation cost only once before deployment. Each online decision then needs only a single forward pass over two skill texts. Other work chooses which skill to read from a library during an episode (Li et al., 2026a). Our method instead decides whether a new skill should replace the current one. Appendix C discusses agent memory, learned skill writers, and further related work.

## 3 Problem Setting

Tasks arrive one at a time. Let $\mathcal { F }$ be the set of task families. The t-th task is $x _ { t } .$ , and its family is $f _ { t } \in \mathcal { F }$ . Each family $f$ keeps one textual skill $s _ { f }$ . Every skill starts as the empty document ∅. A frozen executor $E$ attempts $x _ { t }$ with the current skill $s _ { f _ { t } }$ . It returns a trajectory $\tau _ { t }$ and an outcome $r _ { t }$ that measures task success:

$$
( \tau _ { t } , r _ { t } ) = E ( x _ { t } , s _ { f _ { t } } ) .\tag{1}
$$

A fixed generator $G$ then writes a new skill from the current skill, the trajectory, and the outcome:

$$
\tilde { s } _ { t } = G ( s _ { f _ { t } } , \tau _ { t } , r _ { t } ) .\tag{2}
$$

A format check $v ( \tilde { s } _ { t } ) \in \{ 0 , 1 \}$ tests whether the new skill is well formed. If $v ( \tilde { s } _ { t } ) = 1$ , a selector C chooses the next skill of the family:

$$
s _ { f _ { t } } \gets C ( f _ { t } , s _ { f _ { t } } , \tilde { s } _ { t } ) \in \{ s _ { f _ { t } } , \tilde { s } _ { t } \} .\tag{3}
$$

Otherwise the current skill is kept. The skills of other families do not change. The selected skill serves later tasks of the same family. It cannot change the outcome of the task that has just finished.

Information boundary. The executor observes the current task and the environment feedback. The generator observes the current skill, the latest trajectory, and its outcome. The selector observes only the family name, a generic family description, and the two skill texts. It does not see the latest task, its trajectory, or its outcome.

Objective. Let $\mathcal { D } _ { f }$ be the task distribution of family $f .$ Let $r ( x , s )$ denote the outcome of $E ( x , s )$ We define the utility of a skill s as the expected outcome of the executor with this skill:

$$
\begin{array} { r } { u _ { f } ( s ) = \mathbb { E } _ { x \sim \mathcal { D } _ { f } } \left[ r ( x , s ) \right] . } \end{array}\tag{4}
$$

We omit E from the notation because the executor is fixed. The selector should keep the skill with the higher utility. This utility is unknown at deployment. We therefore learn the relative order of skills from measurements made before deployment. In this paper, cross-task transfer means transfer between different tasks of the same family. It does not mean transfer between families.

## 4 Method

Figure 2 shows the workflow of EVOGUARD. Before deployment, we first construct a multi-source skill pool with the rewrite loop. We then estimate the utility of every skill on a private test set. Finally, we train a pairwise selector on pairs of scored skills. At deployment the frozen selector decides after every task whether the new skill replaces the current one. We drop the task index t in this section.

## 4.1 Single-Trajectory Skill Rewriting

A skill is a SKILL.md document. It has YAML metadata and four Markdown sections. The sections are When to use, Workflow, Decision rules, and Recovery and pitfalls. After each task, the generator G rewrites the whole document. Its inputs are the current skill s, the trajectory $\tau$ of actions and observations, and the outcome r:

![](images/389e0b2e666b07feb3264dc8731f8d49fae599209f098644ab302c91b3a53597.jpg)  
Figure 2: Overview of EVOGUARD. (a) A rewrite loop on training tasks fills a skill pool from two sources, a base generator and an expert generator. (b) The utility of every skill is estimated on a private test set. Rejection sampling keeps same-family pairs with a clear utility gap to train a pairwise selector. (c) At deployment the frozen selector accepts or rejects each rewrite in one forward pass.

$$
\tilde { s } = G ( s , \tau , r ) .\tag{5}
$$

The format check $v ( \tilde { s } )$ is purely structural. It requires the family-specific name and a nonempty description. It also requires each of the four sections to appear exactly once with nonempty content. The check does not test whether the skill helps. A rewrite with $v ( \tilde { s } ) = 0$ is discarded, and the current skill stays.

## 4.2 Multi-Source Skill Pool Construction

For each family $f ,$ we split the training tasks into two parts. A small part serves as the private test set $\mathcal { H } _ { f }$ . The other part is used to build the skill pool $\mathcal { P } _ { f }$ . The pool is filled by the rewrite loop itself. The first few tasks run without a skill, and their valid rewrites initialize the pool. For every later task $x ,$ we draw a parent skill s uniformly from the pool. The executor attempts x with s and returns $( \tau , r ) = E ( x , s )$ The generator then rewrites s. The new skill joins the pool if it passes the format check:

$$
{ \mathcal { P } } _ { f }  { \mathcal { P } } _ { f } \cup \{ G ( s , \tau , r ) \} .\tag{6}
$$

We record s as the parent of the new skill.

Two skill sources. The pool should contain both good and poor skills with clear utility gaps between them. Such gaps give the selector informative training pairs. Skills from a single generator tend to cover only a narrow quality range. We therefore fill the pool from two sources. The first source is the base generator G itself. The second source is an expert generator, a stronger model that writes skills of higher quality. Section 5 gives the expert generator of each benchmark.

## 4.3 Skill Utility Estimation

We evaluate every valid skill once on each task of $\mathcal { H } _ { f }$ . We also evaluate the empty skill of each family. The empirical utility of a skill is its mean outcome on the private test set:

$$
\hat { u } _ { f } ( s ) = \frac { 1 } { | \mathcal { H } _ { f } | } \sum _ { x \in \mathcal { H } _ { f } } r ( x , s ) .\tag{7}
$$

The tasks in $\mathcal { H } _ { f }$ are never used to generate skills. They are also disjoint from the online evaluation tasks.

## 4.4 Pairwise Preference Learning

Skill-level split. We split the scored skills into training skills and validation skills. The split is stratified by family and source. The empty skills stay on the training side. A validation skill never appears in a training pair.

Rejection sampling of pairs. We draw two training skills $s _ { i }$ and $s _ { j }$ of the same family. We keep the pair only if their utilities are clearly apart:

$$
| \hat { u } _ { f } ( s _ { i } ) - \hat { u } _ { f } ( s _ { j } ) | \geq \delta .\tag{8}
$$

Each utility is estimated from a finite set of tasks. Pairs with a smaller gap therefore have unreliable labels, and we reject them. No unordered pair is kept twice. We place the two skills at positions A and B in random order. The label $y \in \{ A , B \}$ is the position of the skill with the higher utility. A pair does not need to consist of a parent and its child.

Pairwise preference objective. We cast the comparison as pairwise preference learning over two label tokens. The selector is a cross-encoder that reads both skills in one prompt. The prompt names the family and describes its generic task. It then presents the two texts as Skill A and Skill B. The prompt ends with the assistant prefix The better skill is Skill. The answer is thus the next token. Both $\_ \mathrm { A }$ and $\_ \mathrm { B }$ are single tokens in the vocabulary. Let $\mathbf { h } _ { \theta } \in \mathbb { R } ^ { d }$ be the finallayer hidden state at the last prompt position. Here θ denotes the trainable parameters. Let $\mathbf { w } _ { A }$ and $\mathbf { w } _ { B }$ be the rows of the unembedding matrix for the two label tokens. The selector uses only the logits of these two tokens instead of the full next-token distribution:

$$
z _ { A } = \mathbf { w } _ { A } ^ { \top } \mathbf { h } _ { \theta } , \qquad z _ { B } = \mathbf { w } _ { B } ^ { \top } \mathbf { h } _ { \theta } .\tag{9}
$$

We renormalize over the two tokens to obtain the probability that the skill at position y is better. We then minimize its negative log-likelihood:

$$
\mathcal { L } ( \theta ) = - \log \frac { \exp z _ { y } } { \exp z _ { A } + \exp z _ { B } } .\tag{10}
$$

This probability equals $\sigma ( z _ { y } - z _ { \bar { y } } )$ , where y¯ is the other position. The loss thus depends only on the margin between the two logits and has the form of a Bradley–Terry preference loss (Bradley and Terry, 1952).

## 4.5 Online Skill Selection

At deployment every family starts from the empty skill. Consider a task of family f whose rewrite s˜ passes the format check. We place the current skill $s _ { f }$ and s˜ at positions A and B in random order, as in training. The selector is then queried once. It returns the position with the larger logit. The skill at that position becomes the family skill:

$$
a ^ { \star } = \{ { \begin{array} { l l } { A } & { { \mathrm { i f } } \ z _ { A } \geq z _ { B } , } \\ { B } & { { \mathrm { o t h e r w i s e } } , } \end{array} } \quad \quad s _ { f } \gets s _ { a ^ { \star } } .\tag{11}
$$

This rule implements the selector C in Eq. (3). The prompt is the same as in training. One decision costs one forward pass. It generates no tokens and runs no extra rollouts. Neither the selector nor the executor is updated online. Appendix A.2 gives the prompts, Appendix A.4 describes how the selector is served, and Appendix A.5 gives the full online loop.

## 5 Experimental Setup

## 5.1 Benchmarks

We evaluate on four benchmarks and use their test sets for online evaluation. ALFWorld (Shridhar et al., 2021) is a text-based household environment. An agent completes tasks such as finding, heating, or cleaning objects through text commands. Its tasks fall into six families, namely Pick, Look, Clean, Heat, Cool, and Pick2. We use its seen and unseen test splits with 140 and 134 tasks. Science-World (Wang et al., 2022) is a text-based environment for elementary science experiments. Each of its 30 task types forms a family. We draw at most 30 test tasks per type, which gives 651 test tasks. LiveCodeBench (Jain et al., 2025) collects competitive programming problems over time. We use the 175 problems added in its sixth release as the test set. Codex with GPT-6.1-sol assigns each problem to one of nine algorithmic categories (Appendix A.1). These categories serve as families. WebShop (Yao et al., 2022) simulates online shopping. An agent searches a store and buys a product that matches an instruction. Its five product categories serve as families, and we use its 500 test tasks.

## 5.2 Evaluation Metrics

On ALFWorld, we report the success rate of each family and of the whole split. On ScienceWorld, we report the success rate and the mean environment score. The score ranges from 0 to 100 and also credits partial progress. On LiveCodeBench, we report the pass rate. A problem counts as passed if the generated program passes all hidden tests. On WebShop, we report the success rate and the mean purchase score. The score rates the purchased product on its type, attributes, options, and price.

## 5.3 Baselines

No-Skill runs the executor with an empty skill on every task.

Accept-All adopts every valid rewrite.

Random samples a skill uniformly from all valid rewrites of the family so far, including the empty skill.

LLM-Judge gives the selector prompt to the untrained backbone and lets it generate A or B.

LLM-Judge (Think) does the same with thinking enabled, so the model reasons before it answers. LLM-Judge (Logit) applies our assistant prefix and our A/B logit rule to the untrained backbone.

## 5.4 Implementation Details

Training. To build pairs with clear utility gaps, we sample skills from generators of different quality. Skills from different generators follow different distributions and cover different quality levels. We use Qwen3-4B (Yang et al., 2025) as the generator. However, its skills are of limited quality. Its skills alone rarely form pairs with clear utility gaps. We therefore add a second source of skills. On ALFWorld, ScienceWorld, and WebShop, the second source is an expert generator. We train it from Qwen3-4B with GRPO on the training tasks. On LiveCodeBench, the second source is GPT-6.1- sol. The second source samples more high-quality skills. Together with the base skills, it yields pairs with clear utility gaps.

All experiments run on a server with four NVIDIA H200 GPUs. The selector backbone stays frozen, and we train LoRA adapters (Hu et al., 2022) on all attention and feed-forward projections. The adapters have rank 32 and scaling factor 64. We use AdamW with a learning rate of 10<sup>−4</sup> and a cosine schedule. We train for one epoch and use the final checkpoint. Appendix A.3 gives the remaining details.

Evaluation. The online evaluation samples at temperature 1.0. An episode has at most 30 actions on ALFWorld, ScienceWorld, and WebShop. Each setting is run three times with different sampling seeds. We report the mean over the three runs. WebShop results are reported in Appendix B.1.

## 6 Experimental Results

## 6.1 Main Performance

Table 1 reports online success on ALFWorld. We train the selector once. We then evaluate the same selector with executors built on three backbones, namely Qwen3-4B, Qwen3-8B, and Qwen3-32B.

Overall success. EVOGUARD achieves the best overall success with every executor. With the 4B executor, it reaches 55.7% on seen and 62.4% on unseen tasks. The strongest baselines reach 44.8% and 51.2%. With the 8B executor, EVOGUARD reaches 53.3% and 54.7%. The strongest baselines reach 44.3% and 48.3%. Averaged over these four settings, EVOGUARD reaches 56.5%. Accept-All and the three LLM-Judge baselines all average about 45.5%. The executor alone averages 36.7%.

An executor without training labels. No training label comes from Qwen3-32B. Its rows therefore test transfer to an unseen executor. EVO-GUARD reaches 66.7% on seen and 71.6% on unseen tasks. The strongest baselines reach 59.3% and 68.7%. The executor alone reaches 51.7% and 58.2%.

Effect of training. LLM-Judge (Logit) uses the same prompt and the same decision rule as EVO-GUARD. The comparison between the two thus isolates the effect of training. EVOGUARD improves over LLM-Judge (Logit) by 6.4% to 16.9% in overall success.

Per-family results. The gains differ across task families. They are largest on families that need multi-step procedures. With the 4B executor on seen tasks, EVOGUARD raises Heat from 10.4% without skills to 54.2%. It raises Cool from 1.3% to 37.3%. On Pick, every method already reaches at least 75%, which leaves little room for skills. On a few families, skills even hurt. On Pick2 with the 8B executor, the executor alone reaches 61.1% on seen tasks, while EVOGUARD reaches 50.0%. On Heat with the 32B executor, the executor alone reaches 50.7% on unseen tasks, while EVOGUARD reaches 33.3%. The selector can only choose among the rewrites it receives. It cannot repair a family where written skills do not help.

Acceptance behavior. Table 2 reports how each method treats the rewrites. Almost all rewrites pass the format check. Only LLM-Judge and LLM-Judge (Think) see a few invalid ones. EVOGUARD accepts only 42.3% and 43.2% of the valid rewrites. The LLM-Judge baselines accept 68.4% to 72.0%. The skill in use is also shorter under EVOGUARD. It averages 361 to 371 tokens, against 528 to 567 tokens under Accept-All and the LLM-Judge baselines. Yet EVOGUARD achieves the best overall success with these shorter skills. This reveals that EVOGUARD tends to keep content that is general and useful, rather than content that is specific to individual cases.

## 6.2 Performance on Other Benchmarks

We also report results on ScienceWorld and Live-CodeBench in Table 3.

<table><tr><td></td><td colspan="7">Seen</td><td colspan="7">Unseen</td></tr><tr><td>Method</td><td>Pick (35)</td><td>Look (13)</td><td>Clean (27)</td><td>Heat (16)</td><td>Cool (25)</td><td>Pick2 (24)</td><td>All (140)</td><td>Pick (24)</td><td>Look (18)</td><td>Clean (31)</td><td>Heat (23)</td><td>Cool (21)</td><td>Pick2 (17)</td><td>All (134)</td></tr><tr><td>Executor: Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Skill</td><td>82.9</td><td>10.3</td><td>23.5</td><td>10.4</td><td>1.3</td><td>48.6</td><td>36.0</td><td>81.9</td><td>29.6</td><td>29.0</td><td>13.0</td><td>11.1</td><td>23.5</td><td>32.3</td></tr><tr><td>Accept-All</td><td>90.5</td><td>20.5</td><td>27.2</td><td>29.2</td><td>14.7</td><td>30.6</td><td>41.0</td><td>86.1</td><td>11.1</td><td>78.5</td><td>30.4</td><td>54.0</td><td>19.6</td><td>51.2</td></tr><tr><td>Random</td><td>87.6</td><td>12.8</td><td>37.0</td><td>33.3</td><td>16.0</td><td>30.6</td><td>42.1</td><td>75.0</td><td>16.7</td><td>58.1</td><td>30.4</td><td>44.4</td><td>25.5</td><td>44.5</td></tr><tr><td>LLM-Judge</td><td>82.9</td><td>5.1</td><td>35.8</td><td>25.0</td><td>29.3</td><td>38.9</td><td>42.9</td><td>77.8</td><td>20.4</td><td>78.5</td><td>21.7</td><td>60.3</td><td>23.5</td><td>51.0</td></tr><tr><td>LLM-Judge (Think)</td><td>91.4</td><td>15.4</td><td>42.0</td><td>39.6</td><td>24.0</td><td>18.1</td><td>44.3</td><td>76.4</td><td>1.9</td><td>67.7</td><td>47.8</td><td>50.8</td><td>19.6</td><td>48.3</td></tr><tr><td>LLM-Judge (Logit)</td><td>85.7</td><td>12.8</td><td>45.7</td><td>22.9</td><td>22.7</td><td>38.9</td><td>44.8</td><td>84.7</td><td>11.1</td><td>67.7</td><td>20.3</td><td>57.1</td><td>23.5</td><td>47.8</td></tr><tr><td>EVOGUARD</td><td>87.6</td><td>33.3</td><td>46.9</td><td></td><td>54.2 37.3</td><td>51.4</td><td>55.7</td><td>84.7</td><td>24.1</td><td>76.3</td><td>68.1</td><td></td><td>65.1 35.3</td><td>62.4</td></tr><tr><td>Executor: Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Skill</td><td>85.7</td><td>23.1</td><td>34.6</td><td>8.3</td><td>8.0</td><td>61.1</td><td>43.1</td><td>86.1</td><td>22.2</td><td>32.3</td><td>5.8</td><td>12.7</td><td>51.0</td><td>35.3</td></tr><tr><td>Accept-All</td><td>89.5</td><td>20.5</td><td>37.0</td><td>2.1</td><td>30.7</td><td>41.7</td><td>44.3</td><td>86.1</td><td>22.2</td><td>68.8</td><td>2.9</td><td>39.7</td><td>35.3</td><td>45.5</td></tr><tr><td>Random</td><td>88.6</td><td>35.9</td><td>35.8</td><td>4.2</td><td>17.3</td><td>47.2</td><td>44.0</td><td>87.5</td><td>24.1</td><td>61.3</td><td>8.7</td><td>50.8</td><td>35.3</td><td>47.0</td></tr><tr><td>LLM-Judge</td><td>79.0</td><td>23.1</td><td>37.0</td><td>8.3</td><td>34.7</td><td>40.3</td><td>43.1</td><td>87.5</td><td>29.6</td><td>72.0</td><td>2.9</td><td>23.8</td><td>35.3</td><td>45.0</td></tr><tr><td>LLM-Judge (Think)</td><td>81.9</td><td>20.5</td><td>42.0</td><td>4.2</td><td>10.7</td><td>56.9</td><td>42.6</td><td>79.2</td><td>33.3</td><td>73.1</td><td>1.4</td><td>50.8</td><td>25.5</td><td>47.0</td></tr><tr><td>LLM-Judge (Logit)</td><td>85.7</td><td>20.5</td><td>28.4</td><td>2.1</td><td>32.0</td><td>37.5</td><td>41.2</td><td>84.7</td><td>33.3</td><td>77.4</td><td>1.4</td><td>39.7</td><td>33.3</td><td>48.3</td></tr><tr><td>EVOGUARD</td><td>88.6</td><td>30.8</td><td>51.9</td><td></td><td>16.7 44.0</td><td>50.0</td><td>53.3</td><td>86.1</td><td>31.5</td><td>75.3</td><td>20.3</td><td>52.4</td><td>47.1</td><td>54.7</td></tr><tr><td>Executor: Qwen3-32B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>No-Skill</td><td>89.5</td><td>66.7</td><td>33.3</td><td>29.2</td><td>22.7</td><td>54.2</td><td>51.7</td><td>88.9</td><td>50.0</td><td>49.5</td><td>50.7</td><td>57.1</td><td>51.0</td><td>58.2</td></tr><tr><td>Accept-All</td><td>90.5</td><td>71.8</td><td>38.3</td><td>10.4</td><td>42.7</td><td>47.2</td><td>53.6</td><td>86.1</td><td>57.4</td><td>80.6</td><td>15.9</td><td>76.2</td><td>37.3</td><td>61.2</td></tr><tr><td>Random</td><td>88.6</td><td>76.9</td><td>56.8</td><td>29.2</td><td>30.7</td><td>59.7</td><td>59.3</td><td>90.3</td><td>63.0</td><td>68.8</td><td>21.7</td><td>85.7</td><td>51.0</td><td>64.2</td></tr><tr><td>LLM-Judge</td><td>87.6</td><td>84.6</td><td>33.3</td><td>14.6</td><td>40.0</td><td>43.1</td><td>52.4</td><td>86.1</td><td>79.6</td><td>88.2</td><td>15.9</td><td>81.0</td><td>52.9</td><td>68.7</td></tr><tr><td>LLM-Judge (Think)</td><td>89.5</td><td>74.4</td><td>44.4</td><td>14.6</td><td>45.3</td><td>62.5</td><td>58.3</td><td>86.1</td><td>66.7</td><td>88.2</td><td>7.2</td><td>66.7</td><td>39.2</td><td>61.4</td></tr><tr><td>LLM-Judge (Logit)</td><td>88.6</td><td>74.4</td><td>35.8</td><td>8.3</td><td>21.3</td><td>52.8</td><td>49.8</td><td>88.9</td><td>70.4</td><td>81.7</td><td>7.2</td><td>77.8</td><td>49.0</td><td>63.9</td></tr><tr><td>EVOGUARD</td><td>96.2</td><td>76.9</td><td>51.9</td><td>33.3</td><td>52.0</td><td>72.2</td><td>66.7</td><td>91.7</td><td>79.6</td><td>87.1</td><td>33.3</td><td>77.8</td><td>51.0</td><td>71.6</td></tr></table>

Table 1: Online success rate (%, ↑) on ALFWorld by task family for three executors. Each value is the mean of three runs. Bold marks the best value in each column for each executor.

<table><tr><td></td><td colspan="3">Qwen3-4B</td><td colspan="3">Qwen3-8B</td></tr><tr><td>Method</td><td>Valid</td><td>Accepted Tokens</td><td></td><td>Valid</td><td>Accepted</td><td>Tokens</td></tr><tr><td>Accept-All</td><td>100.0</td><td>100.0</td><td>553</td><td>100.0</td><td>100.0</td><td>567</td></tr><tr><td>Random</td><td>100.0</td><td>12.8</td><td>303</td><td>100.0</td><td>13.0</td><td>307</td></tr><tr><td>LLM-Judge</td><td>99.8</td><td>68.9</td><td>541</td><td>99.9</td><td>70.9</td><td>536</td></tr><tr><td>LLM-Judge (Think)</td><td>98.4</td><td>68.4</td><td>565</td><td>99.1</td><td>68.8</td><td>548</td></tr><tr><td>LLM-Judge (Logit)</td><td>100.0</td><td>69.0</td><td>543</td><td>100.0</td><td>72.0</td><td>528</td></tr><tr><td>EVOGUARD</td><td>100.0</td><td>42.3</td><td>371</td><td>100.0</td><td>43.2</td><td>361</td></tr></table>

Table 2: Acceptance statistics of the runs in Table 1. Valid is the share of rewrites that pass the format check (%). Accepted is the share of valid rewrites that are adopted (%). Tokens is the mean length of the skill in use.

ScienceWorld. EVOGUARD reaches 15.3% success and a score of 27.3. Without skills, the executor reaches 10.2% and 20.6. The other methods that use skills reach between 9.4% and 10.8% success. None of them is clearly better than using no skill. Accept-All even falls below it.

LiveCodeBench. EVOGUARD achieves the best pass rate of 28.4%. This is 2.3% higher than running without skills. In contrast, Accept-All and Random both drop slightly to 25.7%. The LLM-Judge baselines reach between 25.3% and 27.0%, which is close to running without skills.

Analysis. The two benchmarks show the same pattern. The other methods that use skills stay close to the executor without skills, and Accept-All falls below it on both benchmarks. These methods cannot evolve the skill in a stable way. Good rewrites are mixed with harmful ones, and the gains of the former are cancelled by the latter. EVOGUARD filters out the harmful rewrites before they replace the current skill. The skill can therefore improve over the task stream instead of drifting. These results show that EVOGUARD mitigates skill degradation during evolution, and that this benefit is not limited to ALFWorld.

<table><tr><td rowspan="2">Method</td><td colspan="2">ScienceWorld</td><td>LiveCodeBench</td></tr><tr><td>Success</td><td>Score</td><td>Pass Rate</td></tr><tr><td>No-Skill</td><td>10.2</td><td>20.6</td><td>26.1</td></tr><tr><td>Accept-All</td><td>9.4</td><td>21.2</td><td>25.7</td></tr><tr><td>Random</td><td>10.0</td><td>20.7</td><td>25.7</td></tr><tr><td>LLM-Judge</td><td>10.0</td><td>21.6</td><td>26.7</td></tr><tr><td>LLM-Judge (Think)</td><td>10.8</td><td>21.3</td><td>25.3</td></tr><tr><td>LLM-Judge (Logit)</td><td>9.6</td><td>22.0</td><td>27.0</td></tr><tr><td>EVOGUARD</td><td>15.3</td><td>27.3</td><td>28.4</td></tr></table>

Table 3: Online results on ScienceWorld and Live-CodeBench (%). ScienceWorld reports the success rate and the mean score. LiveCodeBench reports the pass rate. Bold marks the best value in each column.

## 6.3 Training Performance

Figure 3 tracks the selector during training. On validation pairs, the untrained Qwen3-4B reaches

![](images/7f673aa7284922a4143917605cce54fada3714b5a626a65f9e4e573374690687.jpg)

Figure 3: Selector accuracy during training on the training batches (Train), on validation pairs (Validation), and on parent-child validation pairs (Parent-child).  
![](images/4471ff08a02c1231c6f5725d6f87b988372fb418cd9a16cce207a484f83c069e.jpg)  
Figure 4: Share of validation pairs for which the selector keeps the same skill after the two skills swap positions. The dashed line marks a random choice.

48.2%. This is no better than chance, because it answers A for 72% of the prompts. Accuracy rises to 89.6% after 100 steps. It reaches 95.0% at the final step. Parent-child pairs match the comparison of the online loop. On these pairs, accuracy rises from 58.1% to 86.3%.

## 6.4 Position Stability

Pairwise LLM judges are known to suffer from position bias (Wang et al., 2024b; Zheng et al., 2023). We test this bias on the validation pairs. For each pair, we swap the positions of the two skills and check whether the selector keeps the same skill. Figure 4 reports this agreement rate. The untrained LLM-Judge (Logit) agrees on only 55.6% of the pairs. This is close to the 50% of a random choice. The untrained backbone thus cannot judge reliably which of the two skills is better. Our training places the two skills in random order and optimizes the two label logits. This calibrates the logits and makes the decision nearly invariant to the order of the inputs. EVOGUARD agrees on 98.3% of the pairs.

## 6.5 Ablation Study

Table 4 removes one component of the training data at a time. Both variants reduce the average success by more than 4%. w/o expert-written pool removes the expert-written skills from the pool. The pool then lacks high-quality skills, because the base generator writes skills of limited quality. These skills are also generated by the same backbone as the selector, so their distribution is close to that of the selector backbone itself. The resulting pairs are difficult to distinguish and provide only a weak training signal. w/o rejection sampling draws pairs at random and discards only ties. As a result, 44% of the training pairs differ by less than 10% in utility, and their labels are dominated by estimation noise. Rejection sampling retains only pairs with a clear utility gap, which allows the selector to learn the features that separate good skills from poor ones. Both components are thus needed to obtain informative training pairs.

<table><tr><td></td><td colspan="2">Qwen3-4B</td><td colspan="2">Qwen3-8B</td><td></td></tr><tr><td>Variant</td><td>Seen</td><td>Unseen</td><td>Seen</td><td>Unseen</td><td>Avg.</td></tr><tr><td>EVOGUARD</td><td>55.7</td><td>62.4</td><td>53.3</td><td>54.7</td><td>56.5</td></tr><tr><td>w/o expert-written pool</td><td>47.1</td><td>58.2</td><td>48.3</td><td>54.5</td><td>52.0</td></tr><tr><td>w/o rejection sampling</td><td>49.5</td><td>57.0</td><td>48.3</td><td>54.7</td><td>52.4</td></tr></table>

Table 4: Ablation study of EVOGUARD, reported as overall online success (%).

## 7 Conclusion

Online skill evolution lets a deployed agent improve its skill after every task. However, each rewrite is based on a single trajectory and can degrade the skill. Adopting every rewrite lets the skill drift, while validating every update is too costly. We propose EVOGUARD, a pairwise skill selector for skill selection. We first build a dataset of skills with utility labels through iterative rollouts and skill rewriting. Each label is the empirical utility of the skill on a private test set. Rejection sampling then keeps high-confidence pairs with clear utility gaps as preference supervision. We cast the comparison between the current skill and the new one as pairwise preference learning. The selector takes only the logits of the two label tokens and renormalizes over them. It aligns this conditional distribution with the preference labels. At deployment, each decision needs a single forward pass and generates no tokens. Its cost and latency are therefore low. Experiments on two agent benchmarks and one code benchmark confirm the effectiveness of EVO-GUARD. On ALFWorld, it reaches 56.5% average success against 45.5% for adopting every rewrite. On ScienceWorld, it raises success from 10.2% to 15.3%. On LiveCodeBench, it achieves the best pass rate of 28.4%.

## Limitations

Our experiments cover text-based agent environments and one code generation benchmark. Extending EVOGUARD to multimodal or embodied settings is left for future work. All executors come from one model family. Other model families may rank skills somewhat differently. Building and scoring the skill pool requires offline computation before deployment. This cost is paid once and is shared by all later decisions. We train one selector per benchmark. A single selector that transfers across domains would further reduce the setup cost. The selector compares the current skill with one new skill at a time. Richer update rules, such as keeping several versions or reverting to an earlier one, may bring further gains. Finally, the selector can only choose among the rewrites it receives. Its benefit is therefore smaller in settings where skills offer little help.

## Ethical Considerations

Our experiments use public benchmarks in simulated environments and model-generated trajectories. They involve no human subjects and no private user data. The selector is supervised by task success, which does not guarantee that a skill is safe outside these environments. A reusable skill may also carry incorrect instructions to later tasks. Real deployments should therefore add safety checks suited to their domain and keep a record of skill versions. We use all benchmarks and models in accordance with their licenses. Generative AI assisted with literature search and with drafting and editing this manuscript.

## References

Salaheddin Alzubi, Noah Provenzano, Jaydon Bingham, Christian Alexander Calvo, Weiyuan Chen, and Tu Vu. 2026. EvoSkill: Automated skill discovery for multi-agent systems. In Third Conference on Language Modeling.

Anthropic. 2025. Equipping agents for the real world with Agent Skills. Anthropic Engineering Blog.

Ralph Allan Bradley and Milton E. Terry. 1952. Rank analysis of incomplete block designs: I. The method of paired comparisons. Biometrika, 39(3/4):324– 345.

Tianle Cai, Xuezhi Wang, Tengyu Ma, Xinyun Chen, and Denny Zhou. 2024. Large language models as tool makers. In International Conference on Learning Representations.

Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, Yong Mao, Ke Li, and Xing Sun. 2025. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191.

Chishui Chen, Jiaye Lin, Te Sun, Yi Yang, Junxi Wang, Cong Qin, Yangen Hu, Lu Pan, and Ke Zeng. 2026. Skill or skip? learning selective skill invocation in agentic tasks via dual-granularity preference learning. arXiv preprint arXiv:2606.00510.

Chenhao Dang, Siyuan Xiong, Conghui He, and Weijia Li. 2026. SKILLER: Language-level reinforcement learning for reusable skill extraction in small language models. arXiv preprint arXiv:2608.10538.

Aniket Didolkar, Nicolas Ballas, Sanjeev Arora, and Anirudh Goyal. 2025. Metacognitive reuse: Turning recurring LLM reasoning into concise behaviors. arXiv preprint arXiv:2509.13237.

Gen Dong, Yanjie Gao, Liqun Li, Tianyin Xu, Yu Hua, and Fan Yang. 2026. Agent skills can be harmful: An empirical study of skill-induced failures in LLM agents. arXiv preprint arXiv:2608.11888.

Yuyao Ge, Yiwei Wang, Yuchen He, Baolong Bi, Lingrui Mei, Jiayu Yao, Lizhe Chen, and Shenghua Liu. 2026. SkillForge: Co-evolving skills and agents via dynamic skill lifecycles. In Advances in Neural Information Processing Systems.

Shuang Guo, Xiaobo Sui, Fei Jiang, Ruixin Li, Manxiang Li, Rong Li, and Ziyang Ou. 2026. Towards strategy-level RSI for skill-augmented agents: Learning when to reuse skills from execution feedback. arXiv preprint arXiv:2609.26863.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Siyuan Huang, Pengyu Cheng, Haotian Liu, Tao Chen, Yihao Liu, Jingwei Ni, Shijie Zhou, Ziyi Yang, Gangwei Jiang, Mengyu Zhou, Yu Cheng, Xiaoxi Jiang, and Guanjun Jiang. 2026. Skill self-play: Pushing the frontier of LLM capability with co-evolving skills. arXiv preprint arXiv:2607.22529.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. 2025. Live-CodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations.

Guanyu Jiang, Zhaochen Su, Xiaoye Qu, and Yi R. Fung. 2026a. XSkill: Continual learning from experience and skills in multimodal agents. In Proceedings of the 43rd International Conference on Machine Learning.

Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi, Zifeng Wang, Luxi He, Yichen Wu, Ming Zhong, Peiyang Song, Qizheng Zhang, Heng Wang, Xueqiang Xu, Hanwen Xu, Pengrui Han, Dylan Zhang, Jiashuo Sun, Chaoqi Yang, Kun Qian, Tian Wang, Changran Hu, and 15 others. 2025. Adaptation of agentic AI: A survey of post-training, memory, and skills. arXiv preprint arXiv:2512.16301.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. 2026b. SoK: Agentic skills — beyond tool use in LLM agents. arXiv preprint arXiv:2602.20867.

Qingyao Li, Wenxiang Jiao, Shuai Shao, Kangning Zhang, Yuan Lu, Yi Guo, Weiwen Liu, Weinan Zhang, and Yong Yu. 2026a. SkillGate: Training inpolicy skill selection in long-horizon agents. arXiv preprint arXiv:2608.18852.

Xiangyi Li, Yimin Liu, Wenbo Chen, Bingran You, Zonglin Di, Yifeng He, Shenghan Zheng, Kyoung Whan Choe, Jiankai Sun, Shuyi Wang, Chujun Tao, Binxu Li, Xuandong Zhao, Hejia Geng, Xiaojun Wu, Junwei Zhou, Xiaokun Chen, Hanwen Xing, Yubo Li, and 59 others. 2026b. SkillsBench: Benchmarking how well agent skills work across diverse tasks. arXiv preprint arXiv:2602.12670.

Zhiwei Li and Yong Hu. 2026. SkillHone: A harness for continual agent skill evolution through persistent decision history. arXiv preprint arXiv:2606.08671.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. 2026. SKILL0: In-context agentic reinforcement learning for skill internalization. arXiv preprint arXiv:2604.02268.

Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng, Dexin Wang, Erchao Zhao, Xiaoxi Jiang, and Guanjun Jiang. 2026. Trace2Skill: Distill trajectory-local lessons into transferable agent skills. arXiv preprint arXiv:2603.25158.

Guanyu Nie, Fangzhou Zhu, Shixiong Kai, Xiongwei Han, Tao Zhong, and Mingxuan Yuan. 2026. SkillEvoReg: Regularizing agent skill evolution against overfitting. arXiv preprint arXiv:2609.30861.

Siru Ouyang, Jun Yan, Yanfei Chen, Rujun Han, Zifeng Wang, Bhavana Dalvi Mishra, Rui Meng, Chun-Liang Li, Yizhu Jiao, Kaiwen Zha, Maohao Shen, Vishy Tirumalashetty, George Lee, Jiawei Han, Tomas Pfister, and Chen-Yu Lee. 2026a. SkillOS: Learning skill curation for self-evolving agents. arXiv preprint arXiv:2605.06614.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2026b. ReasoningBank: Scaling agent self-evolving with reasoning

memory. In International Conference on Learning Representations.

Libin Qiu, Zhirong Gao, Junfu Chen, Yuhang Ye, Liangyu Li, Weizhi Huang, Xiaobo Xue, Wenkai Qiu, and Shuo Tang. 2026. AutoRefine: Compiling trajectories into validated typed agent artifacts. arXiv preprint arXiv:2601.22758.

Junhao Shen, Zhanqiu Zhang, Yiwen Guo, and Hong Cheng. 2026. Progressive agent skill generation via reinforcement learning. arXiv preprint arXiv:2608.01678.

Yaorui Shi, Yuxin Chen, Zhengxi Lu, Yuchun Miao, Shugui Liu, Qi GU, Xunliang Cai, Xiang Wang, and An Zhang. 2026. Skill1: Unified evolution of skillaugmented agents via reinforcement learning. arXiv preprint arXiv:2605.06130.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. 2021. ALFWorld: Aligning text and embodied environments for interactive learning. In International Conference on Learning Representations.

Shuzheng Si, Haozhe Zhao, Yu Lei, Qingyi Wang, Dingwei Chen, Zhitong Wang, Zhenhailong Wang, Kangyang Luo, Zheng Wang, Gang Chen, Fanchao Qi, Minjia Zhang, and Maosong Sun. 2026. From context to skills: Can language models learn from context skillfully? arXiv preprint arXiv:2604.27660.

Charlie Snell, Dan Klein, and Ruiqi Zhong. 2022. Learning by distilling context. arXiv preprint arXiv:2209.15189.

Chenxi Wang, Zhuoyun Yu, Xin Xie, Wuguannan Yao, Runnan Fang, Shuofei Qiao, Kexin Cao, Guozhou Zheng, Xiang Qi, Peng Zhang, and Shumin Deng. 2026a. SkillX: Automatically constructing skill knowledge bases for agents. arXiv preprint arXiv:2604.04804.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2024a. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research.

Hanyu Wang, Yifan Lan, Bochuan Cao, Lu Lin, and Jinghui Chen. 2026b. SkillGrad: Optimizing agent skills like gradient descent. arXiv preprint arXiv:2605.27760.

Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, and Lin Lee Cheong. 2026c. Reinforcement learning for self-improving agent with

skill library. In Proceedings ofthe 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1529–1550.

Peiyi Wang, Lei Li, Liang Chen, Zefan Cai, Dawei Zhu, Binghuai Lin, Yunbo Cao, Lingpeng Kong, Qi Liu, Tianyu Liu, and Zhifang Sui. 2024b. Large language models are not fair evaluators. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9440–9450.

Ruoyao Wang, Peter Jansen, Marc-Alexandre Côté, and Prithviraj Ammanabrolu. 2022. ScienceWorld: Is your agent smarter than a 5th grader? In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 11279–11298.

Zifei Wang, Wei Wen, Qiang Ji, Keyu Chen, Ruizhi Qiao, and Xing Sun. 2026d. Skill is not document: Query-conditioned compatibility for LLM agent skill routing. arXiv preprint arXiv:2606.03565.

Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. 2025. Agent workflow memory. In Proceedings ofthe 42nd International Conference on Machine Learning.

Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, and Derek Zhiyuan Cheng. 2025. Evo-Memory: Benchmarking LLM agent test-time learning with self-evolving memory. arXiv preprint arXiv:2511.20857.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. 2026. SkillRL: Evolving agents via recursive skill-augmented reinforcement learning. In Advances in Neural Information Processing Systems.

Renjun Xu and Yang Yan. 2026. Agent skills for large language models: Architecture, acquisition, security, and the path forward. arXiv preprint arXiv:2602.12430.

Qianxi Yan, Chunrong Chen, Jiuzhou Zhao, Min Zhang, Yongzhou Xu, and Xiaochuan Xu. 2026. SkillEvo: Self-renewing evolution gradients from multi-turn interaction feedback. arXiv preprint arXiv:2608.13120.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Ling Yang, Zhaochen Yu, Tianjun Zhang, Shiyi Cao, Minkai Xu, Wentao Zhang, Joseph E. Gonzalez,

and Bin Cui. 2024. Buffer of thoughts: Thoughtaugmented reasoning with large language models. In Advances in Neural Information Processing Systems, volume 37.

Min Yang, Jinghua Piao, Xu Xia, Xiaochong Lan, Jiaju Chen, Yongshun Gong, and Yong Li. 2026a. Skill-Master: Toward autonomous skill mastery in LLM agents. arXiv preprint arXiv:2605.08693.

Yifan Yang, Ziyang Gong, Weiquan Huang, Qihao Yang, Ziwei Zhou, Zisu Huang, Yan Li, Xuemei Gao, Qi Dai, Bei Liu, Kai Qiu, Yuqing Yang, Dongdong Chen, Xue Yang, and Chong Luo. 2026b. SkillOpt: Executive strategy for self-evolving agent skills. arXiv preprint arXiv:2605.23904.

Yutao Yang, Junsong Li, Qianjun Pan, Bihao Zhan, Yuxuan Cai, Lin Du, Jie Zhou, Kai Chen, Qin Chen, Xin Li, Bo Zhang, and Liang He. 2026c. AutoSkill: Experience-driven lifelong learning via skill selfevolution. arXiv preprint arXiv:2603.01145.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. WebShop: Towards scalable realworld web interaction with grounded language agents. In Advances in Neural Information Processing Systems, volume 35.

Zhiyuan Yao, Yuxin Chen, Zhengxi Lu, Zishan Xu, Yueqing Sun, Yifu Guo, Yuquan Lu, Zhengzhou Cai, Kangning Zhang, Zhuowen Han, Zi-Han Wang, Ziang Ye, Qi Gu, Xunliang Cai, Weiwen Liu, and Yongliang Shen. 2026. SkillRise: Agentic reinforcement learning for cross-task skill evolution. arXiv preprint arXiv:2607.26784.

Tianzhu Ye, Li Dong, Xun Wu, Shaohan Huang, and Furu Wei. 2026. On-policy context distillation for language models. arXiv preprint arXiv:2602.12275.

Simon Yu, Gang Li, Weiyan Shi, and Peng Qi. 2026. PolySkill: Learning generalizable skills through polymorphic abstraction for continual learning. In International Conference on Learning Representations.

Hanrong Zhang, Shicheng Fan, Henry Peng Zou, Yankai Chen, Zhenting Wang, Jiayu Zhou, Chengze Li, Wei-Chieh Huang, Yifei Yao, Kening Zheng, Xue Liu, Xiaoxiao Li, and Philip S. Yu. 2026a. CoEvoSkills: Self-evolving agent skills via co-evolutionary verification. In Third Conference on Language Modeling.

Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James Zou, and Kunle Olukotun. 2026b. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations.

Zhiwei Zhang, Yudi Lin, Nikki Lijing Kuang, Linlin Wu, Xiaomin Li, Songtao Liu, and Fenglong Ma. 2026c. Co-evolving skill generation and policy optimization. arXiv preprint arXiv:2606.08755.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. 2024. ExpeL: LLM agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632–19642.

Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, and Yu Su. 2025. SkillWeaver: Web agents can self-improve by discovering and honing skills. arXiv preprint arXiv:2504.07079.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36.

Yifan Zhou, Zhentao Zhang, Ziming Cheng, Shuo Zhang, Qizhen Lan, Zhangquan Chen, Zhi Yang, Qianyu Xu, Ronghao Chen, Huacan Wang, and Sen Hu. 2026a. SkillGenBench: Benchmarking skill generation pipelines for LLM agents. arXiv preprint arXiv:2605.18693.

Yingli Zhou, Wang Shu, Yaodong Su, Wenchuan Du, Yixiang Fang, and Xuemin Lin. 2026b. A comprehensive survey on agent skills: Taxonomy, techniques, and applications. arXiv preprint arXiv:2605.07358.

Jiapeng Zhu, Jianxiang Yu, Yibo Zhao, Chengcheng Han, Qi Gu, Xunliang Cai, Xiang Li, and Weining Qian. 2026. Skill0.5: Joint skill internalization and utilization for out-of-distribution generalization in agentic reinforcement learning. arXiv preprint arXiv:2605.28424.

## A Additional Implementation Details

This appendix gives the problem categories of Live-CodeBench, the prompts, the training of the selector, its use at inference, and the full online loop.

## A.1 LiveCodeBench Problem Categories

LiveCodeBench provides no grouping of its problems into task families. We therefore group the problems by the kind of algorithm they need, so that problems of one category can share a skill. We use the 879 problems released up to version 5 that have sample tests as the training set and the 175 problems added in version 6 as the test set (Jain et al., 2025).

Labeling procedure. We label all 1,054 problems with GPT-6.1-sol through the Codex command-line interface. Every call runs in an empty directory with a read-only sandbox. The prompt starts with the instruction to answer from the given text only, to run no commands, and to reply with one JSON value. The model sees the title, the statement, and the starter code of a problem. The hidden tests are never shown to it. The training and the test problems go through the same procedure, which has two steps.

In the first step, the model tags the problems in batches of 20. The prompt is followed by the problems:

Below are {N} competitive programming problems. For   
each one, name the algorithmic topics a solver   
needs for the intended solution: 1 to 3 short   
lower-case tags (for example "dynamic   
programming", "greedy", "graph traversal",   
"simulation"), the most specific that still   
describe a family of problems, and which one   
of them is the primary topic.   
Reply with a JSON array with one object per   
problem, in the same order: {"id": "<problem   
id>", "tags": ["<tag>", ...], "primary": "<one   
of the tags>"}

An answer is kept only if the primary topic is one of the listed tags. This step yields 206 distinct primary tags.

In the second step, one call merges these tags into categories. The prompt is followed by every tag with its number of problems:

{N} competitive programming problems (LeetCode and   
AtCoder) were each given one primary topic   
tag. Below is every tag with its number of   
problems. Merge the tags into 8 to 12   
categories so that problems of one category   
are solved with a shared set of techniques and   
a note written for the category could help on   
each of its problems. Every tag goes to   
exactly one category; no category should hold   
fewer than about 4% of the problems.   
Reply with a JSON object: {"categories": [{"name":   
"<lower-case-kebab-case id>", "title": "<short   
title>", "description": "<one sentence in the   
imperative saying what such problems ask for,   
e.g. Find an optimal value by filling a table   
over subproblems.>"}, ...], "tags": {"<tag>":   
"<category name>", ...}}

We accept the answer only if it has 8 to 12 categories and assigns every tag to one of them. Each problem then belongs to the category of its primary tag. The one-sentence description of a category serves as its family description in the selector prompt.

Category distribution. The procedure yields nine categories. Figure 5 shows their sizes in the training and the test set. The categories are reasonably balanced. The largest one, Greedy and ordering, holds 158 problems (15.0%), and the smallest one, Graphs and connectivity, holds 72 (6.8%). No category falls below the 4% floor requested in the prompt. The four largest categories are Greedy and ordering, Dynamic programming, Range counting and search, and Mathematics and geometry. Together they cover 56.8% of the problems. The test set follows a similar distribution. Every category has at least 11 test problems, so each skill is updated several times during the online evaluation.

![](images/3e6f27d9d5373a1a2c671ed0ccc55cf428bedb32d273bcccaf0a880c1bc411af.jpg)  
Figure 5: Number of LiveCodeBench problems per category in the training and the test set.

## A.2 Prompts

Figure 6 shows the two prompts of EVOGUARD on ALFWorld. The selector prompt names the task family and gives its generic description. It then presents the two skills within <skill\_a> and <skill\_b> delimiters and asks which one should be kept. An empty skill is shown as (empty skill). EVOGUARD and LLM-Judge (Logit) append the assistant prefix The better skill is Skill, so that the answer is the next token. LLM-Judge and LLM-Judge (Think) use no prefix and generate their answer. The rewrite prompt gives the generator the current skill and the latest attempt. It asks for the complete updated skill in the format of §4.1.

Prompts on the other benchmarks. The other benchmarks reuse the two prompts of Figure 6. Only the parts listed below change.

ScienceWorld. The domain wording becomes “ScienceWorld science-experiment tasks of type”, and the skill is named scienceworld-{family}. The command list of the rewrite prompt is:

look around; look at {object}; look in {object}; go   
{location}; open {object}; close {object};   
pick up {object}; put down {object}; move   
{object} to {object}; pour {object} in   
{object}; dunk {object} in {object}; mix   
{object}; activate {object}; deactivate   
{object}; use {object} on {object}; connect   
{object} to {object}; disconnect {object};   
read {object}; focus on {object}; eat   
{object}; flush {object}; inventory; task;   
wait; wait1

The executor prompt, which is the same for every method, also carries the following advice on the action format:

Write the action exactly as one of the action   
templates, with every OBJ replaced by one of   
the listed objects. Only the listed objects   
can be used from here: to reach a room that is   
not listed, go to a listed room on the way,   
opening its door first if it is closed. If the   
last observation says that no known action   
matches, that action cannot work here, so do   
not repeat it. If the observation asks for a   
number to choose among actions, answer with   
that number alone.

WebShop. The domain wording becomes “WebShop online-shopping tasks of the product category”, and the skill is named webshop-{family}. The command list of the rewrite prompt is:

search[{query}]; click[{product id}]; click[{option   
value}]; click[next >]; click[< prev];   
click[back to search]; click[description];   
click[features]; click[reviews]; click[buy now]

Two sentences follow the command list:

The skill serves every task of this category, and   
each task asks for a different product: do not   
write this task’s product, attributes, option   
values or price into the skill; refer to them   
as what the instruction asks for.   
How the pages work: a search opens a results page,   
where the way forward is to click a product   
id; "buy now" exists only on a product page. A   
product page lists its options (such as size,   
color or flavor) as clickable values: the   
options the instruction asks for must be   
clicked before "buy now". The purchase ends   
the task and is scored on product type,   
attributes, selected options and price.

LiveCodeBench. There is no command list. The selector prompt opens with:

An agent solves competitive programming problems in   
Python of type "{category title}": {category   
description}   
It reads one reusable skill before every problem of   
this type. Keep the better of the two skills   
below, the one with which the agent’s programs   
will pass the hidden tests more often on   
future problems of this type.

The rewrite prompt uses the same domain wording, and the skill is named livecode-{category}. In place of a trajectory, it shows the problem statement, the program of the agent, and its results on the sample

(a) Selector prompt   
An agent solves ALFWorld household tasks of type "{family}": {family description}   
It reads one reusable skill at the start of every task of this type. Keep the better of the two skills   
below, the one with which the agent will succeed more often on future tasks of this type.   
Skill A:   
<skill\_a>   
{skill A, or "(empty skill)"}   
</skill\_a>   
Skill B:   
<skill\_b>   
{skill B, or "(empty skill)"}   
</skill\_b>   
Which skill should be kept? Answer with A or B.

Assistant prefix of EVOGUARD:

(b) Rewrite prompt   
You maintain a reusable skill for an agent that solves ALFWorld household tasks of type "{family}": {family   
description}   
The agent reads the skill at the start of every task of this type.   
Current skill:   
{current skill, or "(none yet)"}   
The agent just attempted one task with the current skill:   
{task, outcome, and the action and observation of every step}   
Write the updated skill for future tasks of this type, using what this attempt shows.   
Available commands: look; inventory; go to {receptacle}; open {receptacle}; close {receptacle}; take   
{object} from {receptacle}; move {object} to {receptacle}; examine {target}; use {lamp}; heat {object}   
with {microwave}; clean {object} with {sink}; cool {object} with {fridge}; slice {object} with {knife}.   
Output the complete skill as a standard SKILL.md document inside <skill> and </skill>, at most 1500 tokens,   
and nothing else:   
<skill>   
name: alfworld-{family}   
description: <what the skill does and when to use it>   
# ALFWorld {family}   
## When to use   
...   
## Workflow   
...   
## Decision rules   
...   
## Recovery and pitfalls   
</skill>  
Figure 6: The two prompts of EVOGUARD on ALFWorld. Braces mark inserted content. (a) The selector reads the family description and the two skills. It compares the next-token logits of A and B after the assistant prefix. (b) The generator rewrites the skill after every task. In the prompt the four section headings are separated by blank lines.

tests. The instruction to the generator becomes:

Write the updated skill for future problems of this   
type, using what this attempt shows. The next   
problems are different problems of the same   
type, so the skill must not depend on this   
problem’s statement.

## A.3 Selector Training

The selector backbone is Qwen3-4B. We train LoRA adapters with rank 32, scaling factor 64, and no dropout on all attention and feed-forward projections. We optimize with AdamW at a learning rate of $1 0 ^ { - 4 }$ and without weight decay. The learning rate warms up linearly over the first 3% of training and then follows a cosine decay. Gradients are clipped at norm 1.0. On ALFWorld, the private test set $\mathcal { H } _ { f }$ of each family holds 100 tasks. The rejection threshold in Eq. (8) is $\delta = 0 . 1$ , so the two skills of a pair differ by at least 10% in success rate. Each batch holds 36 pairs, six from every family, and training makes one pass over the 24,000 training pairs. We use the final checkpoint. The validation skills serve for reporting only. They are not used to select a checkpoint or to tune hyperparameters. The backbone runs in BF16 with FlashAttention-2. The adapter weights, the labeltoken logits, the loss, and the optimizer states are kept in FP32.

## A.4 Inference

At deployment, we merge the trained adapter into the backbone and serve the merged model. For each query, the server returns the log-probabilities of the two label tokens at the next position. Both values share the same normalization term. Comparing them is therefore equivalent to comparing the logits z<sub>A</sub> and z in Eq. (11).

## A.5 Online Loop

Algorithm 1 gives the online selection loop of §4.5. Every family starts from the empty skill. After each task, the generator rewrites the skill of the family, and the selector decides whether the rewrite replaces it.

Algorithm 1 Online skill selection with EVO-  
GUARD   
Require: frozen executor E, generator G, trained   
selector C, tasks $x _ { 1 } , \ldots , x _ { T }$   
1: $s _ { f } \gets \emptyset$ for every family f   
2: for $t = 1 , \dots , T$ do   
3: f ← family of $x _ { t } ; ~ ( \tau , r ) \gets E ( x _ { t } , s _ { f } )$   
4: $\tilde { s }  G ( s _ { f } , \tau , r )$   
5: if s˜ passes the format check then   
6: place $s _ { f } , \tilde { s }$ at A/B in random order   
7: $s _ { f } \gets$ the skill with the larger A/B logit   
under $C ( f , \cdot , \cdot )$   
8: end if   
9: end for

## B Further Experimental Results

## B.1 Results on WebShop

Table 5 reports the online results on WebShop. The selector is trained with Qwen3-4B as the executor, and the Qwen3-8B columns change only the online executor. EVOGUARD achieves the highest success rate and the highest score with both executors. With Qwen3-4B, it reaches 16.2% success and a score of 55.9, against 14.7% and 53.7 without skills. With Qwen3-8B, it reaches 18.5% and 57.1, against 16.1% and 54.9. The gains are smaller than on ALFWorld. A likely reason is that the Qwen3-4B generator lacks the capability to write high-quality skills for this environment, which limits what the selector can choose from.

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-4B</td><td colspan="2">Qwen3-8B</td></tr><tr><td>Success</td><td>Score</td><td>Success</td><td>Score</td></tr><tr><td>No-Skill</td><td>14.7</td><td>53.7</td><td>16.1</td><td>54.9</td></tr><tr><td>Accept-All</td><td>15.8</td><td>55.6</td><td>18.1</td><td>55.5</td></tr><tr><td>Random</td><td>16.0</td><td>55.0</td><td>16.9</td><td>54.3</td></tr><tr><td>LLM-Judge</td><td>15.6</td><td>55.3</td><td>18.1</td><td>55.3</td></tr><tr><td>LLM-Judge (Think)</td><td>15.3</td><td>55.1</td><td>16.9</td><td>54.8</td></tr><tr><td>LLM-Judge (Logit)</td><td>15.5</td><td>54.8</td><td>17.5</td><td>55.1</td></tr><tr><td>EVOGUARD</td><td>16.2</td><td>55.9</td><td>18.5</td><td>57.1</td></tr></table>

Table 5: Online results on WebShop with two executors. Success is the success rate (%) and Score is the mean purchase score. Bold marks the best value in each column.

## B.2 Combination with Other Methods

EVOGUARD only decides which of two skill documents to keep. It can therefore be attached to other skill evolution methods without further training. We attach it to SkillRise (Yao et al., 2026), which trains one policy both to solve tasks and to curate the skill document. SkillRise keeps every curated document. With EVOGUARD, the selector instead chooses between the old document and the curated one. The selector and its prompt are unchanged.

Table 6 reports the results on ALFWorld. EVO-GUARD improves our reproduction of SkillRise on both splits. Success rises from 86.0% to 86.4% on seen tasks and from 77.9% to 80.1% on unseen tasks. The selector thus also benefits a method whose executor it was not trained with.

<table><tr><td>Method</td><td>Seen (140)</td><td>Unseen (134)</td></tr><tr><td>SkillRise (our reproduction)</td><td>86.0</td><td>77.9</td></tr><tr><td>+ EVOGUARD</td><td>86.4</td><td>80.1</td></tr></table>

Table 6: EVOGUARD attached to our reproduction of SkillRise on ALFWorld, reported as overall success rate (%).

## B.3 Efficiency

Table 7 compares the cost of one selection. EVO-GUARD reads two logits from one forward pass and generates no tokens. It returns a decision in 24 ms. LLM-Judge generates about 16 tokens and needs 71 ms. LLM-Judge (Think) generates 1,144 tokens and needs 4,184 ms. EVOGUARD is thus about 3 times faster than LLM-Judge and 174 times faster than LLM-Judge (Think).

<table><tr><td>Method</td><td>Tokens Latency (ms)</td></tr><tr><td>LLM-Judge 15.7</td><td>71</td></tr><tr><td>LLM-Judge (Think) 1,144</td><td>4,184</td></tr><tr><td>LLM-Judge (Logit) 0</td><td>26</td></tr><tr><td>EVOGUARD</td><td>0 24</td></tr></table>

Table 7: Cost of one selection during the online runs: generated tokens and latency.

## C Additional Related Work

Skill learning and optimization. Beyond SkillRL and SkillRise, further methods combine policy learning with the construction or revision of a skill library (Huang et al., 2026; Shi et al., 2026; Ge et al., 2026; Zhang et al., 2026c; Wang et al., 2026c). Other methods keep the executor fixed and revise the skill text through prompted optimization (Li and Hu, 2026; Ni et al., 2026; Zhang et al., 2026a; Wang et al., 2026b). These methods focus on how a skill is produced or improved. However, our method does not change how skills are written. It only decides whether a rewrite replaces the current skill.

Skill construction and evaluation. LATM separates tool making from tool use (Cai et al., 2024). A line of work constructs and reuses skills in different settings (Zheng et al., 2025; Yu et al., 2026; Yang et al., 2026c; Jiang et al., 2026a; Wang et al., 2026a; Qiu et al., 2026; Si et al., 2026). Several surveys organize the area (Zhou et al., 2026b; Jiang et al., 2026b, 2025). SkillsBench and SkillGen-Bench benchmark the effect of skills and of skill generation pipelines (Li et al., 2026b; Zhou et al., 2026a). An empirical study documents failures induced by skills (Dong et al., 2026). These works construct or benchmark skills and show that a plausible skill can still hurt execution. However, our method uses the measured utility of skills as supervision for a selector that guards every online update.

Learned writers and skill selection. Skill-α, SkillOS, and SKILLER learn or prompt the writer of the skill with feedback from a frozen executor (Shen et al., 2026; Ouyang et al., 2026a; Dang et al., 2026). SkillMaster uses a counterfactual probe to decide which edits to retain during training (Yang et al., 2026a). Other work learns when to invoke a skill or to which skill a query is routed (Guo et al., 2026; Chen et al., 2026; Wang et al., 2026d). These methods shape how a skill is written or choose among the skills of a library. However, our method decides whether one rewrite replaces its predecessor.

Internalizing skills. SKILL0 withdraws contextual skills during reinforcement learning to internalize their behavior (Lu et al., 2026). Skill0.5 uses privileged distillation (Zhu et al., 2026). Context distillation trains a model to reproduce its prompted behavior without the prompt (Snell et al., 2022; Ye et al., 2026). These methods move guidance into model parameters. However, our method keeps the skill as text and leaves the executor unchanged.

Agent memory and context adaptation. Reflexion uses verbal feedback to improve subsequent attempts (Shinn et al., 2023). ExpeL extracts insights from past trajectories (Zhao et al., 2024). Agent Workflow Memory stores reusable workflows (Wang et al., 2025). ReasoningBank organizes experience as reasoning memory (Ouyang et al., 2026b), and Buffer of Thoughts keeps reusable thought templates (Yang et al., 2024). ACE treats context as an evolving resource (Zhang et al., 2026b). Training-Free GRPO and Metacognitive Reuse improve contextual guidance without updating model parameters (Cai et al., 2025; Didolkar et al., 2025). Evo-Memory evaluates evolving memory in sequential task streams (Wei et al., 2025). These methods retrieve, accumulate, or consolidate general memory. However, our method keeps one skill document per task family and only decides between the current skill and one new skill.