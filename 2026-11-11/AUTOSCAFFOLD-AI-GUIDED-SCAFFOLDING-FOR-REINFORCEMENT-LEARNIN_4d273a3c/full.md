# AUTOSCAFFOLD: AI-GUIDED SCAFFOLDING FOR REINFORCEMENT LEARNING OF LANGUAGE MODELS

PREPRINT

Zijian Zhang<sup>1</sup> Rizhen Hu<sup>1</sup> Jiangweizhi Peng<sup>1</sup> Shiyang Li<sup>1</sup> Sujay Bhatt<sup>2</sup> Sihan Zeng<sup>2</sup> Mingyi Hong<sup>1</sup> <sup>1</sup>University of Minnesota, Twin Cities <sup>2</sup>JPMorgan AI Research

![](images/2c417f6753fafc82dcfc34d0dfadc100fb136ae415ef1f1b0999e11db6fbcec6.jpg)  
Figure 1: A teacher LLM helps a student LLM learn in AutoScaffold. During RL training, the frozen teacher reads the student’s training experience and writes scaffolds into its training prompts. Scaffolds are removed at evaluation.

## ABSTRACT

Reinforcement learning with verifiable rewards (RLVR) produces rich training experience: the model’s reasoning, its actions, and the feedback it receives. Both successful and failed attempts can reveal useful strategies, recurring mistakes, and inefficient behavior, yet outcome rewards reduce each attempt to a single number and leave most of this information unused. Training-time scaffolds, such as strategies or hints added to the prompt during training, offer a natural way to turn this information into guidance for the model. In practice, however, their design and adaptation are largely left to humans and tailored to specific domains, rather than derived from the model’s own training experience.

We introduce AutoScaffold, a unified AI-for-AI framework that turns a student’s training experience into adaptive scaffolds, with minimal domain-specific information and no human intervention during training. The goal is to help the student learn more effectively with scaffolds during RL training, so that it performs better on its own when the scaffolds are removed at evaluation. A frozen teacher periodically examines this experience, diagnoses bottlenecks, and designs scaffolds in two forms: (i) strategies shared across a class of tasks and (ii) hints for individual hard tasks. The teacher adjusts the content and amount of support as the student learns, and the resulting training experience informs the next round of diagnosis and revision. We apply the same framework to ALFWorld, searchbased QA, mathematical reasoning, and GPU kernel generation. When evaluated without scaffolds, students trained with AutoScaffold outperform those trained with the corresponding RL baselines in all four domains. Gains include 7.2 points in ALFWorld success and 4.0 points in average QA exact match. On GPU kernel generation, AutoScaffold improves both correctness and execution efficiency, increasing the fraction of tasks with a correct kernel that runs faster than the PyTorch reference by up to 5.0 points. On math, our student surpasses QuestA’s reported accuracy using approximately one-third as many student training steps, starting from the same partial solutions as QuestA but without its fixed curriculum. Analyses show that the teacher identifies a different failure in each task class from the student’s trajectories and shifts its scaffolds from step-by-step workflows to targeted fixes as the student improves, and that its revised scaffolds help the student learn more than keeping the previous scaffold or using no scaffold. These results suggest that a teacher can turn training experience into temporary guidance that improves the student’s independent capabilities.

![](images/d56835e9dd11fdb6a026105a0440f5a62a55d80dfaf528416c1223b3c6179b01.jpg)  
(c) Evaluation results without scaffold

(b) Evaluation curves during training  
![](images/9f5325c25675340bb0a15c680af4cd0993de1429286c4502f18b40b117a05c51.jpg)

![](images/b00649e4652cd8d4d016928b3245eb8d94b7d99649ed586c660361ec3cc4c3f9.jpg)  
(d) How the teacher adjusts solution hints (Math)  
RL baseline prior scaffolding methods AutoScaffold

![](images/5a8af74001599fea6bd3e46a4093d24d3de5486b2a86512b22007911005e190d.jpg)

![](images/2aaee26eca7d35794b04ce457d01349340885194976ea6695bc3bfeb3e7c9b09.jpg)  
Figure 2: AutoScaffold: a teacher LLM designs and adapts scaffolds during RL training. (a) Unlike handdesigned scaffolding, AutoScaffold is automatic, adaptive, and unified. (b) Evaluation curves during training (success rate on ALFWorld, exact match on search-based QA), compared with GiGPO. (c) Gains over the RL baseline of each experiment, evaluated without scaffolds: success rate on ALFWorld, average exact match on search-based QA, average accuracy on math, and KernelBench correctness pass@3 averaged over Levels 1 to 3. Gray dots are prior scaffolding methods. All methods use the same student size as AutoScaffold, except Skill0, SkillRL and SkillC in ALFWorld using 7B models. (d) The teacher sets each problem’s solution-prefix ratio as instance-level scaffold: it first reveals more of the reference solutions and then withdraws support, whereas QuestA switches from 50% to 25% at a fixed step.

## 1 Introduction

Reinforcement learning with verifiable rewards (RLVR) has become a standard approach to post-training language models for reasoning and agentic tasks (Shao et al., 2024; Yu et al., 2025; Feng et al., 2025). Every training step produces rich experience: the model’s reasoning, its actions, and the feedback it receives. Yet RL learns from this experience only through outcome rewards, which reduce each rollout to a single number. Much of what the rollouts reveal is lost: a failed attempt may contain useful reasoning or actions, a successful one may rely on an inefficient strategy, and on tasks that are too hard for the model, nearly every attempt fails and the reward provides no signal at all (Appendix A). Making better use of training experience is therefore an opportunity to improve post-training beyond what rewards alone can teach.

One promising way to turn such experience into guidance is training-time scaffolding, which refers to the procedure of adding auxiliary text (a scaffold), such as a strategy, hint, or partial solution, to the prompt of the model being trained (the student) (Zhang et al., 2026; Li et al., 2026a; Lu et al., 2026). A scaffold conveys what the reward alone cannot: on tasks the student cannot yet solve, it helps the student succeed so that these tasks produce learning signal, and more generally it shows the student better ways to reason and act. Scaffolds serve only to help the student learn during training and are removed at evaluation, where the student must solve tasks on its own.

Existing scaffolding methods, however, fall short in three respects. First, most are not automatic and do not draw on the student’s training experience: their scaffolds are prepared before training, as hand-designed hint tiers, partial solutions, or predefined skills (Zhang et al., 2026; Li et al., 2026a; Lu et al., 2026). A few methods derive support from the student’s training experience, distilling trajectories into skills or generating guidance from failed attempts (Xia et al., 2026; He et al., 2026), but they do so through fixed pipelines. Second, they are notfully adaptive. As the student learns, its abilities change: the failures a scaffold was designed for may disappear, and new failures appear that it does not address, so a scaffold that helped early in training may no longer help the student learn. Yet support either stays fixed or changes through predefined procedures: SAGE tunes hint strength with fixed rollout-based rules (Liao et al., 2026), Skill0 withdraws skills through a predefined staged curriculum (Lu et al., 2026), and SkillRL revises skills from a prescribed sample of failures without measuring their effect on the current student (Xia et al., 2026). Third, they are not unified: each is built around one task structure, such as hints or solution prefixes for mathematical reasoning or skill libraries for agentic tasks. A mathematical solution prefix, for example, does not provide the action sequencing or object-interaction strategies needed for an agentic task, so each new domain requires a new design. Deciding what support a domain needs, and how it should change over training, is therefore left to human researchers, who design scaffolds from their understanding of the domain and the model’s failures, often refining them across repeated training runs.

These limitations motivate the central question of our work:

How can an LLM teacher turn a student’s training experience into scaffolds, designing and adapting them throughout RL post-training, across domains, without human intervention?

The difficulty is that the teacher’s ability to solve a task does not by itself tell it how to help a weaker, evolving student learn. Making scaffolding automatic, adaptive, and unified (Figure 2a) poses three challenges.

• Diagnosing the student’s bottlenecks. Aggregate rewards reveal whether the student succeeds, but not why it fails, while a few trajectories may not represent a whole task class; the teacher must decide which evidence to inspect and which failure patterns warrant intervention.

• Choosing the right scaffold. The teacher must turn each diagnosis into a scaffold that suits the domain and targets either a weakness shared by a task class or a single hard task. The scaffold must also be of the right strength, because if it is too weak, every attempt may fail, and if it is too strong, every attempt may succeed. In particular, under binary outcome rewards, both cases yield identical rewards within a rollout group and no reward-based policy-gradient signal. A scaffold that is too strong can also let the student succeed without learning to solve the task on its own.

• Validating and adapting scaffolds. A scaffold tailored to a few observed failures may not help the student solve other tasks in the same class, and a scaffold that helps at the current point can become unnecessary as the student improves. The teacher must assess proposed changes and decide when to revise or withdraw scaffolds, with the goal of improving the student’s performance without scaffolds.

Together, these challenges make obtaining effective scaffolds non-trivial, even with access to a strong teacher. They motivate a carefully designed framework that connects diagnosis, scaffold construction, and validation throughout training, so that the scaffold remains matched to the student’s evolving needs.

In this paper, we introduce AUTOSCAFFOLD, a unified AI-for-AI framework in which a frozen teacher LLM observes an ongoing RL training run and automatically turns the student’s training experience into scaffolds injected into its training prompts (Figure 1). The approach draws on the educational principle of scaffolding: temporary, learner-contingent support that enables a learner to accomplish tasks beyond their current independent ability, and that is withdrawn as competence develops (Wood et al., 1976; Vygotsky, 1978). AutoScaffold automatically identifies the student’s bottlenecks and proposes new scaffolds. Every K training steps the teacher investigates the student’s experience, using query tools to choose among training statistics, individual trajectories, groups of attempts on the same task, and its own past decisions (Figure 3). To choose the right scaffold, it diagnoses why the student fails and writes two levels of scaffold: class-level strategies shared across a task class, and instance-level hints for tasks the student cannot yet solve, which play complementary roles in our analysis (Appendix A). Further, AutoScaffold also adaptively changes the content and strength of the scaffold based on the student’s current capability. The teacher revises the scaffolds every K steps, and before deployment each class-level edit is validated on held-out tasks against no scaffold and the current scaffold, and the teacher uses these results to decide which edits to keep. It also revises how often class-level scaffolds are injected, and training with the updated scaffolds produces new experience for the next update. Finally, AutoScaffold unifies the design of the scaffold mechanism across a wide spectrum of application domains, requiring minimal domain-specific input such as the task description and the task classes at the beginning of the training.

We evaluate AutoScaffold on four domains with distinct task structures and RL algorithms: ALFWorld (embodied household tasks, GiGPO), search-based question answering (GiGPO), mathematical reasoning (GRPO and DAPO), and KernelBench (GPU kernel generation, rubric-based GRPO). Under scaffold-free evaluation, AutoScaffold improves over the corresponding RL baselines in every domain (Figure 2b,c), by 7.2 points in success rate on ALF-

World, 4.0 points in average exact match on search-based QA, up to 2.0 points in average accuracy on mathematical reasoning, and 3.0 points in correctness on KernelBench, and is competitive with or surpasses methods that rely on human-designed scaffolding. On KernelBench, it also raises the fraction of tasks with a correct kernel that runs faster than the PyTorch reference by up to 5.0 points (Table 7). On mathematical reasoning, it surpasses QuestA with approximately one-third as many student training steps. Ablations show the value of interactive queries and scaffold validation, and analyses of scaffold evolution show that, given only the general principles of scaffolding, the teacher decides on its own what scaffold to give, to which tasks, and how much, and revises these decisions as the student learns (Figure 2d).

Our main contributions are as follows:

• Automatic, adaptive scaffolding across domains. We introduce AUTOSCAFFOLD, a unified AI-for-AI framework that turns a student’s training experience into evolving scaffolds through a frozen teacher LLM. The same procedure operates across domains with minimal domain-specific information and no human intervention during training.

• A design that makes the teacher’s scaffolds effective. We design a framework that consists of a number of key components, such as adaptive scaffolding based on interactive queries and decision memory, controlled injection, and held-out validation of class-level revisions. Ablation studies show that each of these components is critical for the entire system; for example, removing scaffold validation reduces ALFWorld success by 14.1 points.

• Principles for effective scaffolding. Under binary rewards, we characterize conditions under which classand instance-level scaffolds strengthen group-relative learning signals, and show how overly strong scaffolds can eliminate these signals (Appendix A). The analysis motivates combining strategies for shared weaknesses in a task class with hints for persistently unsolved tasks, and adapting support as the student learns.

• Stronger performance without scaffolds. Across embodied tasks, search-based QA, mathematical reasoning, and GPU kernel generation, AUTOSCAFFOLD improves over the corresponding RL baselines under scaffold-free evaluation and is competitive with or surpasses scaffolding methods that rely heavily on human design, without hand-written scaffolds or hand-designed schedules.

## 2 Related Work

AI for automating model development. LLMs increasingly improve AI systems themselves, optimizing harnesses (Lee et al., 2026), writing GPU kernels (Ouyang et al., 2025; Zhang et al., 2025b), and conducting parts of research (Lu et al., 2024; Yamada et al., 2025; Karpathy, 2026). For model training, LLMs have designed rewards (Ma et al., 2024), generated environments (Chen et al., 2026a), and tuned hyperparameters (Liu et al., 2025a), and Post-TrainBench (Rank et al., 2026) evaluates end-to-end automated post-training. Closest to our setting, LLMZero (Fang et al., 2026) diagnoses training signals at each checkpoint and searches over hyperparameter transitions for GRPO, and EvoTrainer (Chen et al., 2026b) co-evolves the policy with a training-side harness by diagnosing rollouts and backtesting interventions. These methods operate on the training configuration or harness; AutoScaffold instead intervenes within a run at the level of task classes and instances, turning the student’s rollouts into guidance attached to its training prompts and removed at evaluation.

Scaffolding in RL for LLMs. To address tasks on which nearly every rollout fails, known as the learning cliff (Zhang et al., 2026), prior work supplies task-specific hints, such as tiered hints in Scaf-GRPO (Zhang et al., 2026), partial solutions in QuestA (Li et al., 2026a) and POPE (Qu et al., 2026), stepwise hints in StepHint (Zhang et al., 2025a), and self-generated hints in SAGE (Liao et al., 2026), or reusable skills for agentic tasks (Xia et al., 2026; Lu et al., 2026; Lin et al., 2026a; He et al., 2026; Shen et al., 2026). In most methods, the form of guidance is designed by hand for one domain, and any adaptation follows predefined rules. The closest methods are SAGE, which adjusts hint strength with rollout-based rules; SkillRL, whose teacher revises skills from a fixed sample of failures without measuring their effect; and SLIM (Shen et al., 2026), which retains, retires, or adds skills by leave-one-skill-out validation but keeps some skills at inference. In contrast, AutoScaffold lets a frozen teacher choose which evidence to inspect, applies one procedure across domains, including math and GPU kernel generation, and validates class-level proposals on the current student before deployment. Table 1 positions AutoScaffold relative to both lines of work.

## 3 AutoScaffold

As mentioned in the introduction, an ideal scaffolding framework should be automatic, adaptive, and unified. Each requirement places demands on AutoScaffold. Being automatic means that no human designs or reviews the scaffolds, so the teacher must find out on its own why the student fails, write the scaffolds itself, and check its own proposals before using them. Being adaptive means that the scaffold must follow the current student, so it is revised throughout training, its effect is tracked across updates, and support is withdrawn once the student no longer needs it. Being unified means that the same procedure must apply to any domain, so it relies only on minimal domain-specific information, allows different forms of textual scaffold across domains, leaves the RL algorithm unchanged, and guides the teacher with principles from our analysis rather than domain-specific rules. Figure 3 gives an overview of AutoScaffold. We describe how the student is trained with scaffolds (Section 3.1) and how a frozen teacher LLM designs them every K training steps (Section 3.2), noting which requirement each design serves.

Table 1: Positioning of AutoScaffold. Stage: whether the method changes how a model behaves at inference time or how it is trained. What it changes: the object that is designed or edited. Granularity: whether guidance is shared by a class of tasks, specific to one task, or both. Automatic: an AI system decides the changes, without hand-designed content or schedules (◦: content is generated by a model, but its use follows hand-designed rules). Adapts to student progress: whether guidance or training settings are revised in response to the student’s current performance and observed failures (◦: changes follow predefined rules or schedules). No extra runs: changes are decided and tested within a single training run, without training separate runs or branches (◦: branches are trained from checkpoints). Scaffold-free eval.: the learner is evaluated without the guidance it received during training (◦: some guidance is kept). Domains: the task types on which the method is evaluated (A = agentic, S = search QA, M = math, K = GPU kernels). –: not applicable.
<table><tr><td>Method</td><td>Stage</td><td>What it changes</td><td>Granularity</td><td>Automatic</td><td>Adapts to student progress</td><td>No extra runs</td><td>Scaffold-free eval.</td><td>Domains</td></tr><tr><td>AI for automating model development</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AHE (Lin et al., 2026b)</td><td>inference</td><td>agent harness</td><td></td><td></td><td></td><td></td><td></td><td>coding</td></tr><tr><td>autoresearch (Karpathy, 2026)</td><td>training</td><td>training code</td><td></td><td></td><td>X</td><td></td><td></td><td>pretraining</td></tr><tr><td>Eureka (Ma et al., 2024)</td><td>training</td><td>reward function</td><td></td><td></td><td></td><td></td><td></td><td>robotics</td></tr><tr><td>Trainee-to-Trainer (Chen et al., 2026a)</td><td>training</td><td>environment</td><td></td><td></td><td></td><td></td><td></td><td>multi-agent</td></tr><tr><td>LLMZero (Fang et al., 2026)</td><td>training</td><td>hyperparameters</td><td></td><td></td><td></td><td></td><td></td><td>sci. QA, chem., music</td></tr><tr><td>EvoTrainer (Chen et al., 2026b)</td><td>training</td><td>recipe, diagnostics</td><td></td><td></td><td></td><td></td><td></td><td>M, code, SWE</td></tr><tr><td>Scaffolding in RL for LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scaf-GRPO (Zhang et al., 2026)</td><td>training</td><td>hints</td><td>instance</td><td></td><td></td><td></td><td></td><td>M</td></tr><tr><td>QuestA (Li et al., 2026a)</td><td>training</td><td>partial solutions</td><td>instance</td><td></td><td></td><td></td><td></td><td>M</td></tr><tr><td>SAGE (Liao et al., 2026)</td><td>training</td><td>hints</td><td>instance</td><td></td><td>0</td><td></td><td></td><td>M</td></tr><tr><td>SkillRL (Xia et al., 2026)</td><td>training</td><td>skills</td><td>class</td><td></td><td></td><td></td><td></td><td>A, S</td></tr><tr><td>Skill0 (Lu et al., 2026)</td><td>training</td><td>skills</td><td>class</td><td></td><td></td><td></td><td></td><td>A, S</td></tr><tr><td>SLIM (Shen et al., 2026)</td><td>training</td><td>skills</td><td>class</td><td></td><td></td><td></td><td>X</td><td>A, S</td></tr><tr><td>AUTOSCAFFOLD (ours)</td><td>training</td><td>text scaffolds: skills, hints, partial solutions</td><td>class + instance</td><td></td><td></td><td></td><td></td><td>A, S, M, K</td></tr></table>

## 3.1 Scaffold-Augmented RL Training

This subsection explains why scaffolds help group-relative learning and why they must adapt to the student, defines scaffolds at two levels, and describes how they are injected into training prompts.

Why scaffolds help group-relative RL and why they must adapt. Group-relative methods such as GRPO (Shao et al., 2024) sample G rollouts for a task x and standardize each rollout’s reward by the mean and standard deviation of its group. Under binary rewards, a group that contains both successes and failures yields advantages of roughly unit scale, whereas a group in which all rollouts fail, or all succeed, yields zero advantage and no gradient. Whether the model learns from a given task therefore depends on whether its rollout group contains mixed outcomes (Liao et al., 2026). If each rollout succeeds independently with probability $p ,$ then a mixed-outcome group occurs with probability

$$
u ( p ) = 1 - ( 1 - p ) ^ { G } - p ^ { G } ,\tag{1}
$$

which increases as p approaches $\textstyle { \frac { 1 } { 2 } }$ from either side and is approximately Gp when $p$ is small (Proposition 1). Task with $G p \ll 1$ thus rarely contribute to training.<sup>1</sup> On a task the student rarely solves, a scaffold makes some rollouts succeed, so the group contains both outcomes and the task can generate training signal. Which scaffold achieves this, however, depends on the current student (Liao et al., 2026). As training proceeds, the student’s abilities change, so the failures a scaffold was written for may disappear, and new failures appear that it does not address. Therefore, a scaffold that helped earlier may no longer help the student learn. Our ablation study further illustrates the limitations of fixed guidance: using Skill0’s published skill bank as a fixed scaffold yields an ALFWorld success rate 15.3 points below AutoScaffold and 8.1 points below the RL baseline (Section 4.3).

![](images/fde49a81cfb019552a134989756c0089398b60f8eb0fd240a418c19cac8c3bd1.jpg)  
Figure 3: Overview of AUTOSCAFFOLD. (a) A class-level scaffold, injected with probability q[C], and an instancelevel scaffold are added to the student’s training prompt. On a hard task this turns all-failure groups into mixed groups that produce training signal; the RL algorithm is unchanged, and all scaffolds are removed at evaluation. (b) At the end of each block of K steps, a frozen teacher gathers evidence from the training log, proposes edits, validates classlevel scaffolds on held-out tasks, and deploys the updated scaffold for the next block. Bottom: an actual update on ALFWorld.

Two levels of scaffold: class-level strategies and instance-level hints. A student may fail for two reasons: it may lack a strategy that a whole class of tasks requires, or a particular task may remain too hard for it even with such a strategy. Following the principle that support should match the capability that the learner currently lacks (Wood et al., 1976), AutoScaffold provides scaffolds at two corresponding levels. To define the first level, we group training tasks into classes using the categories that each training dataset already provides, such as the six task types of ALFWorld or the training sources of search-based QA, in the same way that SkillRL and Skill0 organize skills by task category (Xia et al., 2026; Lu et al., 2026); when a dataset has no such categories, all its tasks form a single class. We write C(x) for the class of task x, and Section 4.1 lists the classes of each domain.

• Class-level scaffolds S<sub>class</sub>[C] capture strategies shared by the tasks in class C, such as workflows (e.g., the order of steps that a type of household task requires), mistakes to avoid (e.g., moving an object again after it has been placed), and checks before finishing (e.g., confirming that the final answer is the quantity the problem asks for). Like the category-specific skills of SkillRL and Skill0, they transfer across tasks of the same class. By raising the success rate of a class, they increase the learning signal that the class provides, as long as they do not push its success rate far above one half (Proposition 3).

• Instance-level scaffolds S<sub>inst</sub>[x] give support specific to a single task x, such as a hint about its key step, an intermediate result, or part of its reference solution. One example is the solution prefix used by QuestA (Li et al., 2026a), which reveals the first r<sub>x</sub> fraction of the reference solution; our math experiments use this form. Such scaffolds mainly help tasks that the student cannot solve even with a class-level strategy: they let these tasks produce successful rollouts and thus contribute to training at all (Proposition 4).

The two levels are complementary: class-level scaffolds move tasks that the student can already partially solve toward a success rate near one half, where they give the most learning signal, while instance-level scaffolds bring tasks with zero success into training (Appendix A).

Algorithm 1 Scaffold-augmented RL training   
Require: student π<sub>θ</sub>; training tasks D with classes $C ( \cdot ) ;$ group size G; initial scaffold $S ^ { 0 } ;$ ; update interval K; training steps N   
1: Hold out $V _ { C }$ from each class C and set $\mathcal { D }  \mathcal { D } \setminus \bigcup _ { C } \overline { { V _ { C } } }$ ▷ held-out tasks for validation (Section 3.2)   
2: $S = ( S _ { \mathrm { c l a s s } } , q , S _ { \mathrm { i n s t } } )  S ^ { 0 }$ ▷ class-level scaffolds, injection probabilities, instance-level scaffolds   
3: $M \gets \emptyset$ ▷ teacher’s memory of its past diagnoses and decisions   
4: $\mathcal { L }  \emptyset$ ▷ training log since the last scaffold update   
5: for $t = 1 , \ldots , N$ do   
6: $\theta _ { \mathrm { o l d } }  \theta ;$ sample a batch of tasks $B \subset D$   
7: for each task $\boldsymbol { \hat { x } } \in \boldsymbol { B }$ do ▷ one rollout group per task   
8: $z _ { x } \sim$ Bernoulli $\left( q [ C ( x ) ] \right)$ $\triangleright z _ { x } = 1 :$ inject S<sub>class</sub>[C(x)]   
9: x˜ ← augment $( x , \ S _ { \mathrm { c l a s s } } ^ { ( z _ { x } ) } [ C ( x ) ] , \ S _ { \mathrm { i n s t } } [ x ] )$ ▷ Eq. (2)   
10: Sample $\{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \tilde { x } )$ and score each with the task reward $R ( x , \tau _ { i } )$   
11: Append $\left( \boldsymbol { x } , \boldsymbol { C } ( \boldsymbol { x } ) , z _ { \boldsymbol { x } } , \{ \tau _ { i } , R ( \boldsymbol { x } , \tau _ { i } ) \} _ { i = 1 } ^ { G } \right)$ to L   
12: end for   
13: Update θ by maximizing ${ \mathcal { I } } ( \theta )$ on the rollout groups of B ▷ Eq. (3)   
14: if t mod $\dot { K } = 0$ and $t \bar { < } N$ then ▷ end of block $k = \mathsf { \hat { t } } / K$   
15: (S, M) ← UPDATESCAFFOLD(k, π<sub>θ</sub>, S, L, M, {V<sub>C</sub>}) ▷ Algorithm 2   
16: $\mathcal { L }  \emptyset$   
17: end if   
18: end for   
19: return π<sub>θ</sub> ▷ evaluated without any scaffold

Injecting scaffolds into the student’s training prompts. To apply to any domain and any group-relative method, every scaffold is plain text placed in the prompt, and the RL algorithm is left unchanged. A class-level scaffold is not shown to every rollout group of its class. Instead, each class $\bar { C }$ has an injection probability $q [ C ]$ , the fraction of its rollout groups that receive $\bar { S } _ { \mathrm { c l a s s } } [ C ]$ , so support for a class can be reduced gradually by lowering q[C]. The complete scaffold is thus $S = ( S _ { \mathrm { c l a s s } } , q , S _ { \mathrm { i n s t } } ) ^ { - }$ . For each rollout group of task $x ,$ we draw $z _ { x } \sim$ Bernoull $\mathrm { i } ( q [ C \bar { ( x ) } ] )$ ) and build the prompt:

$$
\tilde { x } = \mathrm { a u g m e n t } ( x , S _ { \mathrm { c l a s s } } ^ { ( z _ { x } ) } [ C ( x ) ] , S _ { \mathrm { i n s t } } [ x ] ) ,\tag{2}
$$

where $S _ { \mathrm { c l a s s } } ^ { ( 1 ) } [ C ] = S _ { \mathrm { c l a s s } } [ C ] , S _ { \mathrm { c l a s s } } ^ { ( 0 ) } [ C ]$ is empty, and $S _ { \mathrm { i n s t } } [ x ]$ is empty for tasks without an instance-level scaffold. All G rollouts in the group share the prompt x˜, so their rewards are compared under the same scaffold. Taking GRPO as an example, the student is trained to maximize

$$
\mathcal { I } ( \theta ) = \mathbb { E } \bigg [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \tau _ { i } | } \sum _ { t = 1 } ^ { | \tau _ { i } | } \operatorname* { m i n } \Bigl ( \rho _ { i , t } A _ { i } , \ \mathrm { c l i p } \bigl ( \rho _ { i , t } , 1 - \varepsilon _ { \mathrm { c l i p } } , 1 + \varepsilon _ { \mathrm { c l i p } } \bigr ) A _ { i } \Bigr ) \bigg ] ,\tag{3}
$$

where the expectation is over scaffolded tasks x˜ built by Eq. (2) and rollouts $\{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \tilde { x } )$ , and

$$
\rho _ { i , t } = \frac { \pi _ { \theta } ( \tau _ { i , t } \mid \tilde { x } , \tau _ { i , < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( \tau _ { i , t } \mid \tilde { x } , \tau _ { i , < t } ) } , \qquad A _ { i } = \frac { R ( x , \tau _ { i } ) - \mathrm { m e a n } \big ( \{ R ( x , \tau _ { j } ) \} _ { j = 1 } ^ { G } \big ) } { \mathrm { s t d } \big ( \{ R ( x , \tau _ { j } ) \} _ { j = 1 } ^ { G } \big ) + \epsilon }\tag{4}
$$

are the importance ratio and the group-standardized advantage, $\varepsilon _ { \mathrm { c l i p } }$ is the clipping range, and $\epsilon \geq 0$ is a small constant for numerical stability; the KL penalty is omitted for brevity. Compared with standard GRPO, only the prompt changes: rollouts are sampled and their likelihoods computed conditioned on the scaffolded prompt x˜, while each rollout is scored by the original task reward $R ( x , \tau _ { i } )$ . The scaffold is part of the prompt, so its tokens receive no gradient. Other group-relative methods used in our experiments, such as DAPO and GiGPO, are modified in the same way. Because all scaffolds are removed at evaluation, the student must also practice without help during training, so that it learns to solve tasks on its own rather than depend on the scaffold; these unscaffolded rollouts also let the teacher see when a scaffold is no longer needed. We therefore cap $q [ C ] \leq q _ { \operatorname* { m a x } } < 1$ , so each group has probability at least $1 - q _ { \mathrm { m a x } }$ of being trained without the class-level scaffold, following privileged information dropout (Kamienny et al., 2020); in our ablation, injecting the class-level scaffold into every task $( q _ { \mathrm { m a x } } = 1 )$ lowers success below the RL baseline (Section 4.3). Each rollout is logged with its task, class, reward, and whether the class-level scaffold was injected, which lets the teacher compare the student’s success with and without it. Algorithm 1 summarizes the training procedure.

## 3.2 Automatic Scaffold Design and Adaptation

AutoScaffold updates the scaffold every K RL training steps. We call each such period a block: the scaffold stays fixed within a block, and at the end of block k a frozen teacher LLM updates it for block $k + 1 \ ( \mathrm { F i g u r e } \ 3 ( \mathbf { b } ) )$ . Training starts with no class-level scaffold, so every class-level scaffold the student sees is written by the teacher. The teacher makes all scaffold decisions; its instructions contain no domain-specific rules and differ across domains only in a short description of the task domain and the list of classes. The teacher may see reference answers of training tasks but never sees evaluation tasks. Each update takes four steps: gathering evidence, proposing edits, validating class-level scaffolds, and deploying the update. Algorithm 2 summarizes one update.

Algorithm 2 UPDATESCAFFOLD: one teacher update at the end of block k   
Require: block k; student $\pi _ { \boldsymbol { \theta } _ { k } } ;$ ; scaffold S; block log $\mathcal { L } ;$ memory M; held-out tasks $\{ V _ { C } \} ;$ ; teacher T; query budget B; cap q<sub>max</sub>   
1: $\mathbf { \bar { \Sigma } } { E }  \{ ( v _ { C } ^ { \mathrm { n o n e } } , v _ { C } ^ { \mathrm { c u r } } ) \} _ { C } \colon$ success rates of $\pi _ { \boldsymbol { \theta } _ { k } }$ on each $V _ { C }$ without and with $S _ { \mathrm { c l a s s } } [ C ] ; \ : b  0$ ▷ gathering evidence   
2: while T requests more evidence and $b < B$ do ▷ b: number of queries so far   
$3 \colon \quad E  \dot { E ^ { \cup } } $ {response to one query chosen by $T \colon Q _ { \mathrm { s t a t } } , Q _ { \mathrm { t r a j } } , \mathrm { o r } Q _ { \mathrm { g r o u p } }$ on L, or a read of M}; $b \gets b + 1$   
4: end while   
5: $( D , \Delta S ) \gets T ( E , S )$ , with $\Delta S = ( \Delta S _ { \mathrm { c l a s s } } , \Delta q , \Delta S _ { \mathrm { i n s t } } )$ and $q [ C ] \in [ 0 , q _ { \operatorname* { m a x } } ]$ ▷ proposing edits; D: diagnosis   
6: for each class $C$ for which $\Delta S _ { \mathrm { c l a s s } }$ proposes new text $S _ { C } ^ { \prime }$ do ▷ validating class-level scaffolds   
7: $v _ { C } ^ { \mathrm { n e w } } $ success rate of $\pi _ { \boldsymbol { \theta } _ { k } }$ on $\hat { V } _ { C }$ with $S _ { C } ^ { \prime }$   
8: $\check { T }$ revises $\Delta S _ { \mathrm { c l a s s } } [ C ]$ based on $v _ { C } ^ { \mathrm { n o n e } } , v _ { C } ^ { \mathrm { c u r } } ,$ and $v _ { C } ^ { \mathrm { n e w } }$   
9: end for   
10: Apply $\Delta S$ to S ▷ deploying the update   
11: ${ \hat { M } } ^ { \cdot \cdot } \gets M \cup \{ ( k , D , \Delta S ) \}$ ▷ record the diagnosis and decisions   
12: return $S ,$ M

Gathering evidence. Since no human diagnoses the student, the teacher must find out on its own why the student fails, and which evidence answers this is not known in advance. A low success rate in a class may come from one shared mistake or from several unrelated ones; telling them apart requires reading trajectories, comparing failed attempts with successful ones on the same task, and checking whether a pattern seen in one trajectory recurs across the class. Unlike prior methods, which give the teacher a fixed sample of failures, AutoScaffold lets the teacher decide what to read. The teacher accesses the training log of the current block through three query tools, each returning a view of the log at a different level of detail:

• Class statistics. $Q _ { \mathrm { s t a t } }$ returns each class’s success rates with and without its class-level scaffold, and the numbers of tasks whose rollouts all failed or all succeeded.

• Individual trajectories. $Q _ { \mathrm { t r a j } }$ returns trajectories filtered by class and outcome.

• Rollout groups. $Q _ { \mathrm { g r o u p } }$ returns all G attempts on one task, showing how successful and failed attempts on the same task differ.

The teacher gathers evidence in a multi-turn loop. At each turn, it either calls one tool, whose response is appended to its context, or stops and proposes edits. Starting from initial evidence, such as each class’s success rate on its held-out tasks $V _ { C }$ (described below) with and without the current scaffold, it chooses which trajectories or groups to read next based on what it has already seen; each response is kept within its context, and the teacher must stop after B calls. The log of one block, however, shows only how the student behaves now, not how it responded to the teacher’s earlier edits. The teacher therefore keeps a memory M of its past diagnoses and decisions, and can query M for its latest three decisions to judge whether those edits helped. For example, at step 40 on ALFWorld, the teacher first calls $Q _ { \mathrm { s t a t } } .$ finds that Pick2, Look, and Heat are the weakest classes, then calls $Q _ { \mathrm { t r a j } }$ to read four all-failure trajectories from each of them, and finally reads its last three decisions from memory (Figure 4, top). Replacing the query tools and memory with a fixed sample of failed trajectories, as in SkillRL, lowers ALFWorld success by 8.6 points (Section 4.3).

Proposing edits. From this evidence, the teacher diagnoses why the student fails in each class and proposes an edit $\Delta \bar { S = } ( \bar { \Delta S } _ { \mathrm { c l a s s } } , \Delta q , \Delta S _ { \mathrm { i n s t } } )$ , which covers three decisions:

• Class-level strategies. The teacher adds, revises, or removes $S _ { \mathrm { c l a s s } } [ C ]$ to address a weakness shared across class C. It writes these scaffolds as general strategies rather than answers to specific tasks. For example, after observing that the student in ALFWorld’s Pick2 class (two-object placement tasks) often places only one object, the teacher adds a strategy for finding and placing the second object (Section 4.4).

• Injection frequency. The teacher sets $q [ C ] \in [ 0 , q _ { \mathrm { m a x } } ]$ for each class, for example lowering $q [ C ]$ as the student succeeds in class C without the scaffold.

• Instance-level hints. The teacher writes or revises $S _ { \mathrm { i n s t } } [ x ]$ , or adjusts the fraction $r _ { x }$ of a verified reference solution shown to the student, giving more support to tasks the student keeps failing and less to tasks it can already solve. As a simple illustration, a hint for $x ^ { 4 } - 3 x ^ { 2 } + 1 = 0$ could suggest substituting $t = x ^ { 2 }$ , leaving the student to solve the resulting quadratic and recover x.

Instead of domain-specific rules, the teacher’s instructions state two principles from the analysis in Section 3.1, which hold in every domain. First, focus on classes with low success rates or many all-failure groups, where scaffolds add the most learning signal (Propositions 3 and 4). Second, reduce support as the student improves; for class-level scaffolds, start withdrawing once success without the scaffold exceeds 50%, a heuristic threshold motivated by our analysis (Propositions 1 and 3). The teacher may also leave the scaffold unchanged. Appendix B gives examples of its diagnoses and edits.

Validating class-level scaffolds. A class-level scaffold is inferred from a few trajectories, so it may fix the failures the teacher has read without helping other tasks in the same class. Because it is shown to the whole class for an entire block, a harmful scaffold affects many training steps. AutoScaffold therefore checks each proposed class-level scaffold on the current student before deploying it:

• Held-out tasks. Before training, we reserve a set of tasks $V _ { C }$ from the training data of each class and exclude them from training. At the start of each update, the current student is evaluated on $V _ { C }$ without a class-level scaffold and with the current one, and these success rates are part of the initial evidence.

• Three conditions. The success rates on $V _ { C }$ without a class-level scaffold and with the current scaffold come from the evaluation at the start of the update. For each proposed scaffold, we additionally run the current student $\pi _ { \boldsymbol { \theta } _ { k } }$ , the checkpoint at the end of block $k ,$ on $V _ { C }$ with the proposed scaffold, using n rollouts per task and matched seeds.

• Comparisons. The teacher receives the three success rates $( v _ { C } ^ { \mathrm { n o n e } } , v _ { C } ^ { \mathrm { c u r } }$ , and $v _ { C } ^ { \mathrm { n e w } }$ in Algorithm 2). Comparing the proposed scaffold with the current one shows whether the edit improves on existing guidance; comparing either with no scaffold shows whether the student still benefits from support in this class.

• Decision. Using these results and its diagnosis, the teacher adopts the proposed scaffold, keeps the current one, combines the $\mathbf { t w o } ,$ or removes the scaffold of the class.

A higher success rate under the scaffold is not by itself the goal: the teacher prefers scaffolds that help the student succeed while leaving meaningful work for it, and how often the student sees a scaffold is set separately by $q [ C ]$ Without validation, ALFWorld success drops from 93.9% to 79.8%, below the RL baseline (Section 4.3). Instancelevel scaffolds are not validated, since in our experiments they are prefixes of verified reference solutions rather than text written by the teacher.

Deploying the update. The accepted edits are applied to S automatically in a fixed format and form the scaffold for block k + 1. The teacher’s diagnosis and decisions are added to M, and training under the new scaffold produces the evidence for the next update. Repeating this process lets the scaffold follow the student: at representative ALFWorld updates, continuing training with the revised scaffold outperforms both keeping the previous scaffold and using none (Section 4.4).

## 4 Experiments

Our experiments test whether AutoScaffold delivers the three properties argued for in Section 1. Automatic: without hand-written scaffolds or hand-designed schedules, does it improve the student’s performance without scaffolds over RL baselines, and is it competitive with methods built on human-designed scaffolds? Unified: does the same procedure work across domains with different task structures and RL algorithms? Section 4.2 answers both. Adaptive: does revising scaffolds during training matter, and how does the teacher design and adjust them as the student learns? Sections 4.3 and 4.4 answer this.

## 4.1 Experimental Setup

Domains and tasks. We evaluate four domains with distinct task structures: ALFWorld, with six household task types (Pick, Look, Clean, Heat, Cool, Pick2) requiring planning and object interaction (Shridhar et al., 2021); searchbased QA, covering single-hop (NQ, TriviaQA, PopQA) and multi-hop questions (HotpotQA, 2Wiki, MuSiQue, Bamboogle) (Kwiatkowski et al., 2019; Joshi et al., 2017; Mallen et al., 2023; Yang et al., 2018; Ho et al., 2020; Trivedi et al., 2022; Press et al., 2023); KernelBench, with GPU kernel tasks ranging from single operators to full networks (Ouyang et al., 2025); and math reasoning, covering eight competition-level benchmarks (Mathematical Association of America, 2024; 2025; Harvard-MIT Mathematics Tournament, 2025; Brown University Mathematical

Olympiad, 2025; Mathematical Association of America, 2023; Lightman et al., 2023; Lewkowycz et al., 2022; He et al., 2024). Task classes follow each training dataset: the six task types in ALFWorld, the two training sources (NQ and HotpotQA) in search-based QA, six kernel types in KernelBench, and a single class for math (Table 2). Appendix C provides details.

Table 2: Task classes used for class-level scaffolds in each domain. Classes follow the categories provided by each training dataset; math has no such categories and forms a single class.
<table><tr><td>Domain</td><td>Source of classes</td><td>Classes</td></tr><tr><td>ALFWorld</td><td>task type</td><td>Pick, Look, Clean, Heat, Cool, Pick2</td></tr><tr><td>Search-based QA</td><td>training source</td><td>NQ, HotpotQA</td></tr><tr><td>KernelBench</td><td>kernel type</td><td>Conv, Elementwise, Loss, Matmul, Norm/Softmax, Reduce</td></tr><tr><td>Math reasoning</td><td>none</td><td>a single class</td></tr></table>

Baselines. We compare with three types of baselines: (i) training-free methods, which prompt the base model without any training; (ii) training without scaffolds, which train the student with the same task reward but no scaffold; and (iii) training with scaffolds, which add human-designed scaffolds, such as skills, hints, or partial solutions, to the student’s prompts during training.

For ALFWorld, we compare with (i) ReAct (Yao et al., 2023) and Reflexion (Shinn et al., 2023); (ii) PPO (Schulman et al., 2017), RLOO (Ahmadian et al., 2024), GRPO (Shao et al., 2024), and GiGPO (Feng et al., 2025); and (iii) SkillRL (Xia et al., 2026), Skill0 (Lu et al., 2026), and SkillC (Lin et al., 2026a), which are reported only with Qwen2.5-7B-Instruct and serve as references. For search-based QA, we compare with (ii) R1-Instruct, Search-R1 (Jin et al., 2025), ZeroSearch (Sun et al., 2026), StepSearch (Wang et al., 2025), and GiGPO; and (iii) SkillRL and Skill0, at both the same scale as our student and 7B. For KernelBench, we compare with (ii) StitchCUDA (Li et al., 2026b), the rubric-based RL method whose training setting we follow. For math reasoning, we follow two settings. With Qwen3-4B-Instruct, we compare with (ii) SFT and GRPO, and (iii) LUFFY (Yan et al., 2025), Scaf-GRPO (Zhang et al., 2026), and SAGE (Liao et al., 2026). With Nemotron-1.5B, we compare with (ii) DAPO (Yu et al., 2025) and (iii) QuestA (Li et al., 2026a). Baseline results are taken from the original papers or from the work whose training settings we follow, as noted in each table. Among them, SkillRL keeps its skills in the prompt at evaluation, whereas AutoScaffold is always evaluated without scaffolds.

Implementation. In all domains, the teacher is GPT-5.5 with default reasoning effort and uses the same instructions apart from the domain description and class list. Scaffold updates occur every $\check { K } = 1 0$ steps with $q _ { \mathrm { m a x } } = 0 . 5$ , and each update allows at most B = 15 teacher queries. The teacher writes class-level scaffolds in all domains, while instancelevel scaffolds are used only for math. Students and RL algorithms follow prior work in each domain: GiGPO’s settings for ALFWorld and QA, with Qwen2.5-1.5B-Instruct and Qwen2.5-3B-Instruct; StitchCUDA’s rubric-based RL for KernelBench, with a Qwen3-8B student cold-started as in Dr. Kernel (Liu et al., 2026) and 600 training tasks; and SAGE (Qwen3-4B-Instruct with GRPO) and QuestA (Nemotron-1.5B with DAPO) for math, using their official training data. Experiments use 4–8 H200 GPUs, and results are averaged over 3 independent runs. The teacher’s API cost is a few dollars per run (Appendix D). We evaluate AutoScaffold without scaffolds, so the results measure the student’s own capability.

## 4.2 Main Results

Across the four domains, the same framework, with teacher instructions that differ only in the domain description and class list, improves over the corresponding RL baseline under four RL algorithms (GiGPO, rubric-based GRPO, GRPO, and DAPO) and five students. In every domain where scaffolding methods exist, it also matches or surpasses them, although it starts from no class-level scaffold and uses no hand-written scaffold or hand-designed schedule.

ALFWorld. AutoScaffold reaches 93.9% success, 7.2 points above GiGPO, and is best on five of the six task classes (Table 3). The largest gains are on Look (+19.7), Cool (+16.2), and Pick2 (+15.3), the three classes where GiGPO is weakest and successful rollouts are rarest. The 1.5B student also exceeds SkillRL (89.9%) and SkillC (90.6%), which use the larger Qwen2.5-7B-Instruct, and SkillRL additionally keeps its skills at evaluation.

Search-based QA. AutoScaffold achieves 46.1% average exact match, 4.0 points above GiGPO, and is best among 3B models on six of seven benchmarks (Table 4). At the same scale, it exceeds SkillRL and Skill0 by 7.2 and 5.3 points, and it also surpasses 7B Skill0 (44.4%). Although training uses only NQ and HotpotQA, the gain over GiGPO on the five out-of-domain benchmarks (+4.1 on average) is as large as on the two in-domain ones (+3.8), so the scaffolds improve skills that transfer beyond the training data.

Mathematical reasoning. With Qwen3-4B-Instruct, AutoScaffold reaches 70.1% average accuracy, 1.4 points above GRPO, and is on par with SAGE (70.0%), the strongest scaffolding method, without SAGE’s hand-designed hint levels and scheduling rules (Table 5). Guidance does not help by default: LUFFY, which trains on off-policy reference traces, falls 5.2 points below the base model. With Nemotron-1.5B, AutoScaffold reaches 62.4%, 2.0 points above DAPO and 1.3 points above QuestA, using 350 training steps compared with QuestA’s 1,100, roughly one-third (Table 6). The 1.5B student also surpasses DeepSeek-R1-Distill-32B (58.1%).

GPU kernel generation. AutoScaffold improves all six KernelBench metrics over StitchCUDA (Table 7), raising correctness from 59% to 63% on Level 1 and from 50% to 54% on Level 2, and Level 2 Fast from 20% to 25%. Gains on the full-network tasks of Level 3 are smaller (+1 point on both metrics). Since this domain uses rubric-based rather than binary rewards, these results also show that AutoScaffold is not limited to the binary-reward setting of our analysis.

Table 3: Success rates (%) on ALFWorld with Qwen2.5-1.5B-Instruct. Overall for AutoScaffold is the unweighted mean over the six task types. Prompting and RL baseline results are from Feng et al. (2025); ours are mean ± standard deviation over 3 seeds. Gray rows are from Lin et al. (2026a) and use the larger Qwen2.5-7B-Instruct (reference only). Bold: best among 1.5B models.
<table><tr><td>Method</td><td>Pick</td><td>Look</td><td>Clean</td><td>Heat</td><td>Cool</td><td>Pick2</td><td>Overall</td></tr><tr><td>Qwen2.5-1.5B-Instruct</td><td>5.9</td><td>5.5</td><td>3.3</td><td>9.7</td><td>4.2</td><td>0.0</td><td>4.1</td></tr><tr><td colspan="8">Training-free methods</td></tr><tr><td>ReAct</td><td>17.4</td><td>20.5</td><td>15.7</td><td>6.2</td><td>7.7</td><td>2.0</td><td>12.8</td></tr><tr><td>Reflexion</td><td>35.3</td><td>22.2</td><td>21.7</td><td>13.6</td><td>19.4</td><td>3.7</td><td>21.8</td></tr><tr><td colspan="8">RL baselines without scaffold</td></tr><tr><td>PPO</td><td>64.8</td><td>40.5</td><td>57.1</td><td>60.6</td><td>46.4</td><td>47.4</td><td>54.4</td></tr><tr><td>RLOO</td><td>88.3</td><td>52.8</td><td>71.0</td><td>62.8</td><td>66.4</td><td>56.9</td><td>69.7</td></tr><tr><td>GRPO</td><td>85.3</td><td>53.7</td><td>84.5</td><td>78.2</td><td>59.7</td><td>53.5</td><td>72.8</td></tr><tr><td>GiGPO</td><td>94.4</td><td>67.5</td><td>94.8</td><td>94.4</td><td>79.8</td><td>76.4</td><td>86.7</td></tr><tr><td colspan="8">RL baselines with scaffold (Qwen2.5-7B-Instruct, reference only)</td></tr><tr><td>SkillRL</td><td>97.9</td><td>71.4</td><td>90.0</td><td>90.0</td><td>95.5</td><td>87.5</td><td>89.9</td></tr><tr><td>Skill0</td><td>100.0</td><td>84.6</td><td>91.1</td><td>84.2</td><td>88.9</td><td>55.5</td><td>85.9</td></tr><tr><td>SkillC</td><td>88.5</td><td>78.6</td><td>91.2</td><td>94.7</td><td>100.0</td><td>88.9</td><td>90.6</td></tr><tr><td>AUTOSCAFFOLD (ours)</td><td>100.0±0.0</td><td>87.2±4.4</td><td>95.1±2.1</td><td>93.8±0.0</td><td>96.0±0.0</td><td>91.7±4.2</td><td>93.9±0.3</td></tr></table>

Table 4: Exact match (%) on search-based QA with Qwen2.5-3B-Instruct, trained on NQ and HotpotQA (†: indomain; ⋆: out-of-domain). RL baseline results are from Feng et al. (2025); SkillRL and Skill0 are from their original papers (Xia et al., 2026; Lu et al., 2026); ours are mean ± standard deviation over 3 seeds. Gray rows use the larger Qwen2.5-7B-Instruct (reference only). Bold: best among 3B models.
<table><tr><td rowspan="2">Method</td><td colspan="3">Single-Hop QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ†</td><td></td><td>TriviaQA* PopQA*</td><td>HotpotQA†</td><td>2Wiki*</td><td>MuSiQue*</td><td>Bamboogle*</td></tr><tr><td colspan="8">RL baselines without scaffold</td><td></td></tr><tr><td>R1-Instruct</td><td>27.0</td><td>53.7</td><td>19.9</td><td>23.7</td><td>29.2</td><td>7.2</td><td>29.3</td><td>27.1</td></tr><tr><td>Search-R1</td><td>34.1</td><td>54.5</td><td>37.8</td><td>32.4</td><td>31.9</td><td>10.3</td><td>26.4</td><td>32.5</td></tr><tr><td>ZeroSearch</td><td>41.4</td><td>57.4</td><td>44.8</td><td>27.4</td><td>30.0</td><td>9.8</td><td>11.1</td><td>31.7</td></tr><tr><td>StepSearch</td><td></td><td></td><td></td><td>34.5</td><td>32.0</td><td>17.4</td><td>34.4</td><td></td></tr><tr><td>GiGPO</td><td>42.0</td><td>59.5</td><td>42.4</td><td>36.9</td><td>37.0</td><td>12.6</td><td>64.1</td><td>42.1</td></tr><tr><td colspan="8">RL baselines with scaffold</td><td></td></tr><tr><td>SkillRL</td><td>38.6</td><td>57.6</td><td>40.3</td><td>33.6</td><td>31.1</td><td>13.3</td><td>58.1</td><td>38.9</td></tr><tr><td>Skill0</td><td>39.8</td><td>57.5</td><td>42.3</td><td>35.1</td><td>33.7</td><td>13.3</td><td>63.7</td><td>40.8</td></tr><tr><td colspan="8">RL baselines with scaffold (Qwen2.5-7B-Instruct, reference only)</td><td></td></tr><tr><td>SkillRL</td><td>45.9</td><td>63.3</td><td>45.9</td><td>43.2</td><td>40.3</td><td>20.2</td><td>73.8</td><td>47.1</td></tr><tr><td>Skill0</td><td>42.7</td><td>61.1</td><td>45.3</td><td>40.0</td><td>38.3</td><td>16.4</td><td>66.9</td><td>44.4</td></tr><tr><td>AUTOSCAFFOLD (ours)</td><td>46.3±1.3</td><td>62.3±2.0</td><td>49.6±1.2</td><td>40.2±0.4</td><td>39.7±0.6</td><td>16.6±0.5</td><td>67.7±0.5</td><td>46.1±0.6</td></tr></table>

## 4.3 Ablation Study

We evaluate the contribution of AutoScaffold’s main design choices through ablations on ALFWorld with Qwen2.5- 1.5B-Instruct (Table 8). Each of the four main designs matters: removing the query tools and memory, removing validation, fixing the scaffold, or always injecting it drops the student below the RL baseline, so the gains come from how scaffolds are designed and used rather than from extra guidance alone.

Table 5: Accuracy (%) on math benchmarks with Qwen3-4B-Instruct. RL baseline results are from Liao et al. (2026); ours are mean ± standard deviation over 3 seeds. Avg.: mean over the six benchmarks. Bold / underline: best / second best.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>AMC23</td><td>MATH-500</td><td>Minerva</td><td>Olympiad</td><td>Avg.</td></tr><tr><td>Qwen3-4B-Instruct</td><td>52.1</td><td>43.1</td><td>92.2</td><td>93.6</td><td>46.1</td><td>67.7</td><td>65.8</td></tr><tr><td></td><td colspan="7">RL baselines without scaffold</td></tr><tr><td>SFT</td><td>14.4</td><td>22.1</td><td>55.2</td><td>78.9</td><td>38.1</td><td>40.2</td><td>41.5</td></tr><tr><td>GRPO</td><td>55.8</td><td>45.0</td><td>95.0</td><td>96.0</td><td>50.1</td><td>70.4</td><td>68.7</td></tr><tr><td></td><td colspan="7">RL baselines with scaffold</td></tr><tr><td>LUFFY</td><td>42.3</td><td>36.0</td><td>86.4</td><td>91.1</td><td>48.2</td><td>59.4</td><td>60.6</td></tr><tr><td>Scaf-GRPO</td><td>59.8</td><td>45.2</td><td>92.2</td><td>95.1</td><td>48.9</td><td>69.8</td><td>68.5</td></tr><tr><td>SAGE-LIGHT</td><td>59.2</td><td>47.1</td><td>92.2</td><td>95.1</td><td>49.5</td><td>70.5</td><td>68.9</td></tr><tr><td>SAGE</td><td>58.1</td><td>52.1</td><td>94.2</td><td>95.4</td><td>49.2</td><td>71.2</td><td>70.0</td></tr><tr><td>AUTOSCAFFOLD (ours)</td><td>59.3±0.4</td><td>51.4±0.4</td><td>93.9±0.5</td><td>95.5±0.1</td><td>49.4±0.2</td><td>71.0±0.5</td><td>70.1±0.3</td></tr></table>

Table 6: Accuracy (%) on math benchmarks with Nemotron-1.5B. DAPO and QuestA results are from Li et al. (2026a). Gray: larger models (reference only). Bold: best among small models.
<table><tr><td>Method</td><td>AIME24</td><td>AIME25</td><td>HMMT Feb 25</td><td>OlympiadBench</td><td>BRUMO25</td><td>Avg.</td></tr><tr><td colspan="7">Training-free base models</td></tr><tr><td>DeepSeek-R1-Distill-1.5B</td><td>28.7</td><td>22.3</td><td>12.0</td><td>52.4</td><td>31.8</td><td>29.4</td></tr><tr><td>Qwen3-1.7B</td><td>48.3</td><td>36.8</td><td>22.2</td><td>56.1</td><td>44.1</td><td>41.5</td></tr><tr><td>Nemotron-1.5B</td><td>61.8</td><td>49.5</td><td>31.6</td><td>64.6</td><td>58.2</td><td>53.1</td></tr><tr><td colspan="7">Larger models (reference only)</td></tr><tr><td rowspan="2">DeepSeek-R1-Distill-32B Qwen3-8B</td><td>72.6</td><td>51.8</td><td>33.0</td><td>65.0</td><td>68.0</td><td>58.1</td></tr><tr><td>76.0</td><td>67.3</td><td>44.8</td><td>68.6</td><td>68.3</td><td>65.0</td></tr><tr><td colspan="7">RL baselines without scaffold</td></tr><tr><td rowspan="2">DAPO (2K steps)</td><td>69.5</td><td>59.8</td><td>38.9</td><td>68.1</td><td>66.0</td><td>60.4</td></tr><tr><td colspan="6">RL baselines with scaffold</td></tr><tr><td rowspan="2">QuestA (1.1K steps)</td><td>69.3</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>60.0</td><td>37.9</td><td>69.7</td><td>68.3</td><td>61.1</td></tr><tr><td>AUTOSCAFFOLD (ours, 350 steps)</td><td>72.6</td><td>58.1</td><td>42.1</td><td>70.2</td><td>69.2</td><td>62.4</td></tr></table>

Table 7: Pass@3 of Correctness and Fast on the KernelBench test set. Fast : the generated kernel is correct and faster than the PyTorch reference (Ouyang et al., 2025). Bold / underline: best / second best.
<table><tr><td rowspan="3">Method</td><td colspan="2">Level 1</td><td colspan="2">Level 2</td><td colspan="2">Level 3</td></tr><tr><td>Correctness</td><td>Fast1</td><td>Correctness</td><td>Fast1</td><td>Correctness</td><td>Fast1</td></tr><tr><td>Qwen3-8B-Cold-Start</td><td>51</td><td>14</td><td>43</td><td>16</td><td>32</td><td>8</td></tr><tr><td>StitchCUDA</td><td>59</td><td>17</td><td>50</td><td>20</td><td>32</td><td>13</td></tr><tr><td>AUTOSCAFFOLD (ours)</td><td>63</td><td>19</td><td>54</td><td>25</td><td>33</td><td>14</td></tr></table>

Table 8: Ablations on ALFWorld (Qwen2.5-1.5B-Instruct), success rate (%) evaluated without scaffold. Static: Skill0’s published ALFWorld skill bank used as a fixed class-level scaffold. ∆: gap to full AUTOSCAFFOLD.
<table><tr><td rowspan="3"></td><td rowspan="3">AUTOSCAFFOLD GiGPO</td><td rowspan="3"></td><td colspan="2">Teacher decision</td><td colspan="4">Scaffold adaptation &amp; injection</td></tr><tr><td>w/o tools &amp; memory w/o validation</td><td></td><td>Static K=5 K=20</td><td></td><td></td><td> $q _ { \mathrm { m a x } } { = } 1$ </td></tr><tr><td>Success</td><td>93.9</td><td>86.7</td><td>85.3</td><td>79.8</td><td>78.6</td><td>90.1</td><td>90.4</td><td>86.0</td></tr><tr><td>Δ</td><td></td><td>-7.2</td><td>-8.6</td><td>-14.1</td><td>-15.3</td><td>-3.8</td><td>-3.5</td><td>-7.9</td></tr></table>

Query tools and memory. Replacing the query tools and memory with a fixed sample of failed trajectories and a single teacher call, as in SkillRL (Xia et al., 2026), lowers success to 85.3%. With a fixed sample of failures only, the teacher cannot compare failures with successes, look further when the sample is inconclusive, or check how the student responded to its earlier edits.

Scaffold validation. Deploying every proposal without validation causes the largest drop among teacher components, to 79.8%, 6.9 points below GiGPO. Because a class-level scaffold is shown to a whole class for an entire block, an unhelpful edit affects many updates before the teacher can revise it.

Static scaffolds. Using Skill0’s published ALFWorld skill bank (Lu et al., 2026) as a fixed class-level scaffold lowers success to 78.6%, 15.3 points below AutoScaffold and 8.1 points below GiGPO. The bank was not built for this student and cannot change as the student’s failures change; Section 4.4 shows how much the teacher’s own scaffold changes over training.

![](images/174f1919ba545b1c0289d49893cdaf6f0a5d7a7ef28ba27be5e69c87ea44de4e.jpg)  
Figure 4: One teacher update on ALFWorld (step 40). Top: the teacher queries tools and memory, then gathers evidence from class statistics, failed trajectories, and its own past decisions. It reads them and finds meaningful information for further diagnosis and edits. Bottom: it diagnoses a different failure in each class and responds with a matching edit: adding, rewriting, reducing, or omitting a scaffold.

Update interval K. Updating every 5 or 20 steps gives 90.1% and 90.4%, both above GiGPO but below the default $\bar { K = 1 0 }$ . Shorter blocks give the teacher less evidence per update, while longer blocks let the scaffold lag behind the student; K = 10 also needs half as many teacher calls as $K = 5$

Injection probability. Injecting the class-level scaffold into every rollout group $( q _ { \mathrm { m a x } } = 1 )$ lowers success to 86.0%, slightly below GiGPO, although the scaffold content is produced in the same way. The student then never practices without help, which it must do at evaluation (Section 3.1).

## 4.4 Analysis

The ablations show that each design is needed. This section looks at what the teacher actually does with them. We first follow a single update in detail, then trace how the scaffold changes over a full run, test whether revisions help the student at the time they are made, compare the scaffolds the teacher writes in different domains, and finally follow how the teacher adjusts instance-level scaffolds in math.

How the teacher decides at one update (ALFWorld, step 40). Figure 4 follows this update. The teacher checks class statistics with $Q _ { \mathrm { s t a t } } .$ inspects four all-failure trajectories per weak class (Pick2, Look, and Heat) using $Q _ { \mathrm { t r a j } }$ , and reviews its last three decisions in memory M. Its decisions depend on the needs of each class.

• Pick2. The student picks up the first object after placing it or never searches for the second. The teacher adds a strategy for finding and placing the second object.

• Look. The student finds the object or lamp but fails to bring the object to the lamp. The teacher specifies taking the object to the desk lamp, using the lamp, and then examining the object.

• Heat. Even with the existing scaffold, the student reheats or retrieves the object after placing it. The teacher rewrites the scaffold to emphasize placing the object once and stopping.

• Clean. The student already succeeds often and gains little from the scaffold, so the teacher keeps its text but lowers its injection probability.

• Cool and Pick. The teacher identifies no failure requiring a scaffold and provides no text.

Before deployment, each proposed class-level scaffold is also validated on V . Appendix B.1 gives another case, from a separate run, that records the full process of one update: how the teacher gathers evidence, what it diagnoses and proposes, and how validation decides which scaffolds are adopted.

Why these scaffolds help the student learn. These decisions show how the teacher turns observed failures into useful guidance. For Look tasks, the scaffold adds the missing step of bringing the object to the lamp. By helping some rollouts succeed, the scaffold can turn an all-failure group into a mixed group with a useful learning signal. The scaffolds describe strategies that apply across tasks in a class, while leaving the student to select objects and carry out the actions. The teacher also adjusts its guidance based on the student’s response. It clarifies the Heat scaffold when the student repeats actions after placing the object and reduces support for Clean when the scaffold provides little additional benefit.

![](images/372e2a5e2c82a0d57631fafb2a5b118bb2eda3e11a6b36dd2d1ee6fcdb694358.jpg)  
Figure 5: Scaffold changes over training on ALFWorld. Left: scaffold-free success (top) and mean class-level injection probability (bottom), with training divided into three stages. Right: what the teacher changed in each stage (+ add, ∼ revise, − remove, ↑↓ raise or lower injection probability).

How the scaffold changes over a full run. Across training, the teacher adjusts both the content of each class’s scaffold and how often it is used. The ALFWorld run in Figure 5 illustrates three stages.

• Stage 1 (steps 0–50). When success is low across most classes, the teacher writes procedures for two-object, examine-in-light, and heating tasks. The mean injection probability rises, giving the student more frequent support.

• Stage 2 (steps 50–80). As success improves, the teacher removes the broad workflows and retains a targeted reminder to heat the specified object. The mean injection probability falls close to zero.

• Stage 3 (steps 80–150). The teacher addresses remaining failures with specific reminders, such as using the lamp’s exact name in Look and leaving the first object where it was placed in Pick2. It raises the injection probability for the still-difficult Pick2 class.

The support thus evolves from broad procedures to targeted fixes, with changes guided by each class’s training evidence. For comparison, Skill0 uses a withdrawal schedule fixed before training.

Do revisions help the student when they are made? We measure the effect of scaffold revisions while holding the starting student checkpoint and number of subsequent training steps fixed. At representative ALFWorld checkpoints between steps 20 and 80, we create three branches and train each for 20 additional steps using the updated scaffold (AutoScaffold), the previous scaffold kept fixed, or no scaffold (vanilla RL). The updated scaffold yields the highest success at every tested checkpoint (Figure 6). Its advantage over the previous scaffold shows that the revisions improve learning over the following 20 steps. Its advantage over vanilla RL shows that support remains useful at these checkpoints.

What scaffolds the teacher writes in different domains. The teacher adapts scaffold content to the failures observed in each domain. In ALFWorld, its guidance describes action sequences for household tasks. In math, its class-level notes target recurring errors in otherwise sound reasoning, such as stopping after an intermediate result or boxing a quantity other than the one the problem asks for (Appendix B.2).

How the teacher adjusts solution hints in math. We also examine how the teacher adjusts instance-level hints during math training, where each hint reveals a fraction $r _ { x }$ of the problem’s reference solution (Figure 7). We initialize the hints with QuestA’s data (Li et al., 2026a), in which each problem has a 50% or 25% solution prefix, mix both sets from the start, and let the teacher adjust the prefix ratios alongside class-level guidance. During the first 100 steps, the teacher reveals more of the reference solutions, raising the mean prefix ratio from 28.7% to 33.7%. The mean ratio then declines to 26.7% by the end of training; at step 100, for example, the teacher raises the prefixes of all-fail problems that still have room and lowers them for problems the student already solves (Appendix B.2). Our instructions ask the teacher to reduce support for problems the student can already solve. The teacher determines the timing and amount of each adjustment from training evidence. QuestA follows a fixed schedule, using 50% prefixes for the first 100 steps and 25% thereafter. AutoScaffold begins withdrawing support near this scheduled switch and reaches a similar final mean ratio through adjustments to individual problems. Together with the ALFWorld results, this illustrates how the same framework adapts both class-level strategies and instance-level hints as the student learns.

![](images/11d4e8bbf6ce7c8066c2ae6d57e12d2cd72964507cff554d7b89d80823312c6c.jpg)

![](images/2bdb6aa5313d3b8d5c92d0fac45363553ea7ebfbab43849e2fa059df0ff2a70a.jpg)

Figure 6: From each branching point, we train 20 more steps with the updated scaffold (AUTOSCAFFOLD, dark blue), with the previous scaffold kept fixed (orange), or with vanilla RL (gray). AUTOSCAFFOLD gives the highest success at every branching point.  
![](images/62db161ad4ffc8fd0f96ee26b3e8197cf77efce18b9323121677c03cad2c873f.jpg)  
Figure 7: Solution-prefix ratios during math training (Nemotron-1.5B). Mean ratio across all training problems. AutoScaffold (red) adjusts each problem’s ratio using training evidence. QuestA (gray) switches from 50% to 25% at step 100.

## 5 Conclusion

We presented AUTOSCAFFOLD, an AI-for-AI framework that uses a frozen teacher to turn a student’s RL training experience into adaptive guidance. The teacher gathers evidence through query tools and memory, designs class-level strategies and instance-level hints, and validates class-level proposals before use. It adjusts both the guidance and the amount of support as the student learns. Across four domains, the same procedure improves performance over the corresponding RL baselines under scaffold-free evaluation and is competitive with or surpasses methods built on human-designed scaffolds. The gains include both correctness and execution efficiency on GPU kernel generation. Ablations and controlled comparisons support the value of evidence gathering, validation, and scaffold adaptation. Future work includes reducing the cost of teacher interventions and characterizing when learning with scaffolds improves the student’s independent capabilities.

## 6 Disclaimer

This paper was prepared for informational purposes by the Artificial Intelligence Research group of JPMorgan Chase & Co. and its affiliates (“JP Morgan”), and is not a product of the Research Department of JP Morgan. JP Morgan makes no representation and warranty whatsoever and disclaims all liability, for the completeness, accuracy or reliability of the information contained herein. This document is not intended as investment research or investment advice, or a recommendation, offer or solicitation for the purchase or sale of any security, financial instrument, financial product or service, or to be used in any way for evaluating the merits of participating in any transaction, and shall not constitute a solicitation under any jurisdiction or to any person, if such solicitation under such jurisdiction or to such person would be unlawful.

## References

Arash Ahmadian, Chris Cremer, Matthias Gallé, Marzieh Fadaee, Julia Kreutzer, Olivier Pietquin, Ahmet Üstün, and Sara Hooker. Back to basics: Revisiting reinforce style optimization for learning from human feedback in llms, 2024. URL https://arxiv.org/abs/2402.14740.

Brown University Mathematical Olympiad. BRUMO 2025. Problems available at https://huggingface.co/ datasets/MathArena/brumo\_2025, 2025. Accessed September 2026.

Chao Chen, Chengzu Li, Zhiwei Li, Yinhong Liu, and Zhijiang Guo. From trainee to trainer: Llm-designed training environment for rl with multi-agent reasoning, 2026a. URL https://arxiv.org/abs/2606.17682.

Guhong Chen, Yingcheng Shi, Yongbin Li, Binhua Li, Xander Xu, Hu Wei, Shiwen Ni, Min Yang, and Jieping Ye. Evotrainer: Co-evolving llm policies and training harnesses for autonomous agentic reinforcement learning, 2026b. URL https://arxiv.org/abs/2606.03108.

Haoyang Fang, Wei Zhu, Boran Han, Alex Zhang, Zhenyu Pan, Shuo Yang, Shuai Zhang, Jiading Gai, Peng Tang, Cuixiong Hu, Xuan Zhu, Huzefa Rangwala, George Karypis, and Bernie Wang. Llmzero: Discovering adaptive training strategies for rl post-training via llm agents, 2026. URL https://arxiv.org/abs/2606.18388.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for LLM agent training. In Advances in Neural Information Processing Systems, 2025.

Harvard-MIT Mathematics Tournament. HMMT February 2025. https://www.hmmt.org/; problems available at https://huggingface.co/datasets/MathArena/hmmt\_feb\_2025, 2025. Accessed September 2026.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Leng Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems, 2024. URL https://arxiv. org/abs/2402.14008.

Zelin He, Haotian Lin, Boran Han, Wei Zhu, Haoyang Fang, Bernie Wang, Xuan Zhu, Runze Li, and Matthew Reimherr. Reskill: Reconciling skill creation with policy optimization in agentic rl, 2026. URL https:// arxiv.org/abs/2606.01619.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps, 2020. URL https://arxiv.org/abs/2011.01060.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017. URL https://arxiv.org/abs/1705.03551.

Pierre-Alexandre Kamienny, Kai Arulkumaran, Feryal Behbahani, Wendelin Boehmer, and Shimon Whiteson. Privileged information dropout in reinforcement learning, 2020. URL https://arxiv.org/abs/2005.09220.

Andrej Karpathy. autoresearch: AI agents running research on single-GPU nanochat training automatically. https: //github.com/karpathy/autoresearch, 2026. GitHub repository.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl\_a\_00276. URL https://aclanthology.org/Q19-1026/.

Yoonho Lee, Roshen Nair, Qizheng Zhang, Kangwook Lee, Omar Khattab, and Chelsea Finn. Meta-harness: End-to end optimization of model harnesses, 2026. URL https://arxiv.org/abs/2603.28052.

Aitor Lewkowycz, Anders Andreassen, David Dohan, Ethan Dyer, Henryk Michalewski, Vinay Ramasesh, Ambrose Slone, Cem Anil, Imanol Schlag, Theo Gutman-Solo, Yuhuai Wu, Behnam Neyshabur, Guy Gur-Ari, and Vedant Misra. Solving quantitative reasoning problems with language models, 2022. URL https://arxiv.org/ abs/2206.14858.

Jiazheng Li, Hongzhou Lin, Hong Lu, Kaiyue Wen, Zaiwen Yang, Jiaxuan Gao, Yi Wu, and Jingzhao Zhang. Questa: Expanding reasoning capacity in llms via question augmentation, 2026a. URL https://arxiv.org/abs/ 2507.13266.

Shiyang Li, Zijian Zhang, Winson Chen, Yuebo Luo, Mingyi Hong, and Caiwen Ding. Stitchcuda: An automated multi-agents end-to-end gpu programing framework with rubric-based agentic reinforcement learning, 2026b. URL https://arxiv.org/abs/2603.02637.

Baohao Liao, Hanze Dong, Xinxing Xu, Christof Monz, and Jiang Bian. Self-hinting language models enhance reinforcement learning, 2026. URL https://arxiv.org/abs/2602.03143.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. URL https://arxiv.org/abs/2305. 20050.

Hongxiang Lin, Zhirui Kuai, Erpeng Xue, and Lei Wang. Skillc: Learning autonomous skill internalization in llm agents via contrastive credit assignment, 2026a. URL https://arxiv.org/abs/2605.27899.

Jiahang Lin, Shichun Liu, Chengjun Pan, Lizhi Lin, Shihan Dou, Zhiheng Xi, Xuanjing Huang, Hang Yan, Zhenhua Han, Tao Gui, and Yu-Gang Jiang. Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses, 2026b. URL https://arxiv.org/abs/2604.25850.

Siyi Liu, Chen Gao, and Yong Li. Large language model agent for hyper-parameter optimization, 2025a. URL https://arxiv.org/abs/2402.01881.

Wei Liu, Jiawei Xu, Yingru Li, Longtao Zheng, Tianjian Li, Qian Liu, and Junxian He. Dr. kernel: Reinforcement learning done right for triton kernel generations, 2026. URL https://arxiv.org/abs/2602.05885.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding R1-Zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist: Towards fully automated open-ended scientific discovery, 2024. URL https://arxiv.org/abs/2408.06292.

Zhengxi Lu, Zhiyuan Yao, Jinyang Wu, Chengcheng Han, Qi Gu, Xunliang Cai, Weiming Lu, Jun Xiao, Yueting Zhuang, and Yongliang Shen. Skill0: In-context agentic reinforcement learning for skill internalization, 2026. URL https://arxiv.org/abs/2604.02268.

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Eureka: Human-level reward design via coding large language models, 2024. URL https://arxiv.org/abs/2310.12931.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories, 2023. URL https: //arxiv.org/abs/2212.10511.

Mathematical Association of America. American Mathematics Competitions (AMC) 12, 2023. Problems available at https://huggingface.co/datasets/AI-MO/aimo-validation-amc, 2023. Accessed September 2026.

Mathematical Association of America. American Invitational Mathematics Examination (AIME) 2024. Problem available at https://huggingface.co/datasets/HuggingFaceH4/aime\_2024, 2024. Accessed September 2026.

Mathematical Association of America. American Invitational Mathematics Examination (AIME) 2025. Problems available at https://huggingface.co/datasets/math-ai/aime25, 2025. Accessed September 2026.

Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, and Azalia Mirhoseini. Ker nelbench: Can llms write efficient gpu kernels?, 2025. URL https://arxiv.org/abs/2502.10517.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models, 2023. URL https://arxiv.org/abs/2210.03350.

Yuxiao Qu, Amrith Setlur, Virginia Smith, Ruslan Salakhutdinov, and Aviral Kumar. POPE: Learning to reason on hard problems via privileged on-policy exploration, 2026. URL https://arxiv.org/abs/2601.18779.

Ben Rank, Hardik Bhatnagar, Ameya Prabhu, Shira Eisenberg, Karina Nguyen, Matthias Bethge, and Maksym Andriushchenko. Posttrainbench: Can llm agents automate llm post-training?, 2026. URL https://arxiv.org/ abs/2603.08640.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Junhao Shen, Teng Zhang, Xiaoyan Zhao, and Hong Cheng. Dynamic skill lifecycle management for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2605.10923.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning, 2023. URL https://arxiv.org/abs/2303.11366.

Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alf world: Aligning text and embodied environments for interactive learning, 2021. URL https://arxiv.org/ abs/2010.03768.

Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. Zerosearch: Incentivize the search capability of llms without searching, 2026. URL https: //arxiv.org/abs/2505.04588.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition, 2022. URL https://arxiv.org/abs/2108.00573.

Lev S Vygotsky. Mind in society: The development of higher psychological processes, 1978.

Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, and Yichao Wu. Stepsearch: Igniting llms search ability via step-wise proximal policy optimization, 2025. URL https://arxiv.org/abs/2505. 15107.

David Wood, Jerome S Bruner, and Gail Ross. The role of tutoring in problem solving. Journal of child psychology and psychiatry, 17(2):89–100, 1976.

Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, Zeyu Zheng, Cihang Xie, and Huaxiu Yao. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning, 2026. URL https://arxiv.org/abs/2602.08234.

Yutaro Yamada, Robert Tjarko Lange, Cong Lu, Shengran Hu, Chris Lu, Jakob Foerster, Jeff Clune, and David Ha. The ai scientist-v2: Workshop-level automated scientific discovery via agentic tree search, 2025. URL https: //arxiv.org/abs/2504.08066.

Jianhao Yan, Yafu Li, Zican Hu, Zhi Wang, Ganqu Cui, Xiaoye Qu, Yu Cheng, and Yue Zhang. Learning to reason under off-policy guidance, 2025. URL https://arxiv.org/abs/2504.14945.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering, 2018. URL https:// arxiv.org/abs/1809.09600.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models, 2023. URL https://arxiv.org/abs/2210.03629.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale, 2025. URL https:// arxiv.org/abs/2503.14476.

Kaiyi Zhang, Ang Lv, Jinpeng Li, Yongbo Wang, Feng Wang, Haoyuan Hu, and Rui Yan. StepHint: Multi-level stepwise hints enhance reinforcement learning to reason, 2025a. URL https://arxiv.org/abs/2507. 02841.

Xichen Zhang, Sitong Wu, Yinghao Zhu, Haoru Tan, Shaozuo Yu, Ziyi He, and Jiaya Jia. Scaf-grpo: Scaffolded group relative policy optimization for enhancing llm reasoning, 2026. URL https://arxiv.org/abs/2510. 19807.

Zijian Zhang, Rong Wang, Shiyang Li, Yuebo Luo, Mingyi Hong, and Caiwen Ding. Cudaforge: An agent framework with hardware feedback for cuda kernel optimization, 2025b. URL https://arxiv.org/abs/2511.01884.

## A Theoretical Analysis

We analyze how injecting a scaffold changes the learning signal that GRPO receives, which motivates the design of AutoScaffold. Following Liao et al. (2026), we consider binary rewards $R ( x , \tau ) \in \{ 0 , 1 \}$

Setup. For a prompt x, GRPO samples a group of G rollouts $\{ \tau _ { i } \} _ { i = : } ^ { G }$ from the current policy $\pi _ { \theta }$ and computes standardized advantages

$$
A _ { i } = \frac { R _ { i } - \bar { R } } { s + \epsilon } , \quad \bar { R } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } R _ { i } , \quad s ^ { 2 } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } ( R _ { i } - \bar { R } ) ^ { 2 } ,\tag{5}
$$

where $\epsilon \geq 0$ is a numerical stabilizer. The policy gradient is weighted by these advantages: when $A _ { i } = 0$ for all i, the group contributes no gradient regardless of the rollout content.

Degenerate groups and the learning cliff. Under binary rewards, each rollout independently succeeds with probability $p = p _ { \theta } ( x )$ . A group is degenerate if all rollouts share the same outcome $( \bar { R } \in \{ 0 , 1 \} )$ , in which case $s = 0$ and $A _ { i } = 0$ for all i. This underlies the learning cliff (Zhang et al., 2026; Li et al., 2026a): when p is very small, nearly all groups are all-failure and the task contributes almost no learning signal. The probability of obtaining a non-degenerate group therefore determines whether a prompt contributes to learning at all.

Proposition 1 (Non-degenerate group probability (Liao et al., 2026)). For a prompt with success probability $p \in [ 0 , 1 ]$ and group size $G \geq 2$

$$
u ( p ) : = \mathrm { P r } [ s > 0 ] = 1 - ( 1 - p ) ^ { G } - p ^ { G } .\tag{6}
$$

u is strictly concave on $[ 0 , 1 ]$ and symmetric about $\begin{array} { r } { p = \frac { 1 } { 2 } , } \end{array}$ , its unique maximizer, with $\begin{array} { r } { u ( \frac { 1 } { 2 } ) = 1 - 2 ^ { 1 - G } } \end{array}$ . Hence u is 2 strictly increasing on $[ 0 , \textstyle { \frac { 1 } { 2 } } ]$ and strictly decreasing on $[ \textstyle { \frac { 1 } { 2 } } , 1 ]$ , and $u ( p ) = G p + O ( p ^ { 2 } )$ as $p  0 .$

Proof. A group is degenerate exactly when all G rollouts fail or all succeed, which occurs with probability $( 1 - p ) ^ { G } +$ $p ^ { G }$ . Moreover, $u ^ { \prime \prime } ( p \bar { ) } = - G ( G - \bar { 1 } ) \big [ p ^ { G - 2 } + ( 1 - p ) ^ { G - 2 } \big ] < 0$ for all $p \in [ 0 , 1 ]$ , so u is strictly concave. Together with the symmetry $\boldsymbol { u } ( p ) = \boldsymbol { u } ( 1 - p )$ , this implies that u has a unique maximum at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ □

When $p \ll 1$ , which is the common regime for hard tasks, $u ( p ) \approx G p { \mathrm { : } }$ the group is non-degenerate only when at least one rollout succeeds, an event that becomes rare as $p  0$ . Conversely, when $p \approx 1 , u ( p ) \approx G ( 1 - p )$ : almost all groups are all-success, again yielding no gradient.

Proposition 2 (Expected within-group reward variance (Liao et al., 2026)). Define the within-group reward variance $\begin{array} { r } { V = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } ( R _ { i } - \bar { R } ) ^ { 2 } } \end{array}$ . Under binary rewards with success probability p,

$$
\operatorname { \mathbb { E } } [ V \mid x ] = { \frac { G - 1 } { G } } p ( 1 - p ) ,\tag{7}
$$

which is strictly concave in p and uniquely maximized at $\begin{array} { r } { p = \frac { 1 } { 2 } . } \end{array}$

Proof. For binary rewards, $V = \bar { R } ( 1 - \bar { R } )$ . Since $G \bar { R } \sim$ Binomia $( G , p )$ , we have $\mathbb { E } [ \bar { R } ] = p$ and $\mathbb { E } [ \bar { R } ^ { 2 } ] = p ^ { 2 } + p ( 1 -$ $p ) / G$ . Hence $\begin{array} { r } { \mathbb { E } [ V \mid x ] = p - p ^ { 2 } - p ( 1 - p ) / G = \frac { G - 1 } { G } p ( 1 - p ) } \end{array}$ □

Propositions 1 and 2 show that $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ maximizes both the probability of a non-degenerate group and its expected reward variance. The two quantities play different roles under GRPO. Because advantages are standardized, any nondegenerate group yields advantages of roughly unit scale $\begin{array} { r } { ( \frac { 1 } { G } \sum _ { i } A _ { i } ^ { 2 } = s ^ { 2 } / ( s + \epsilon ) ^ { 2 } \approx \mathsf { \breve { 1 } } ) } \end{array}$ , so the primary quantity is $u ( p )$ , i.e., whether a prompt contributes at all. The reward variance measures signal strength when advantages are not normalized by $s ,$ and we use it below as a tractable proxy for class-level analysis. Both quantities are symmetric and concave, vanish at $p \in \{ 0 , 1 \}$ , and peak at $\begin{array} { r } { p = \frac { 1 } { 2 } } \end{array}$ , supporting the same qualitative conclusions.

Implications for scaffold design. Let $p _ { \theta } ( x )$ denote the student’s success probability without scaffold and $p _ { \theta } ( x , S )$ the probability under scaffold S. A scaffold that raises $p _ { \theta } ( x , S )$ from near zero toward $\frac { 1 } { 2 }$ increases both the nondegenerate group probability and the expected reward variance, enabling gradient updates on tasks that would oth erwise contribute almost nothing. A scaffold that is too strong instead overshoots: if $p _ { \theta } ( x , S ) \approx 1$ , groups become degenerate again (all-success), and the learning signal collapses despite a high success rate. An effective scaffold should therefore make success reachable while leaving meaningful work for the student, for example by providing strategies or partial hints rather than complete solutions.

Remark 1 (Scaffold should adapt to the student). The per-step signal is largest when $\begin{array} { r } { p _ { \theta } ( x , S ) \approx \frac { 1 } { 2 } . } \end{array}$ . Since $p _ { \theta } ( x , S )$ depends on $\theta ,$ a scaffold that brings a task close to this point at one stage of training generally does not keep doing so as the student changes: as the student improves, the same scaffold may push $p _ { \theta } ( x , S )$ toward 1, and as its failures shift, the scaffold may no longer address them. Hence no fixed scaffold keeps every task in the high-signal regime throughout training (cf. Liao et al., 2026, Proposition 3.3). This statement concerns the per-step signal and does not by itself characterize the long-term learning trajectory.

Class-level scaffold and aggregate learning signal. The preceding analysis concerns a single prompt with success probability $p .$ Per-task success rates can only be estimated from $\check { G }$ rollouts and are therefore noisy, so we analyze scaffolds at the class level, in terms of the class success rates without scaffold $\left( s _ { C } \right)$ and with scaffold $( \sigma _ { C } )$

Consider a task x drawn uniformly from class $C .$ . Its success probability $p _ { x }$ varies across tasks, reflecting the heterogeneous difficulty of the class. We write $p _ { x } = s _ { C } + \delta _ { x }$ , where $\delta _ { x }$ is a zero-mean perturbation with $\mathrm { V a } \bar { \mathbf { r } } ( \delta _ { x } ) = v _ { C }$ The class-average within-group reward variance is

$$
\bar { V } _ { C } : = \frac { G - 1 } { G } \mathbb { E } _ { x } \big [ p _ { x } ( 1 - p _ { x } ) \big ] = \frac { G - 1 } { G } \big [ s _ { C } ( 1 - s _ { C } ) - v _ { C } \big ] ,\tag{8}
$$

where the cross term vanishes because $\mathbb { E } [ \delta _ { x } ] = 0$ . The term $s _ { C } ( 1 - s _ { C } )$ corresponds to a homogeneous class with uniform difficulty $s _ { C }$ , and the subtracted $v _ { C }$ is the reduction due to task heterogeneity (consistent with Jensen’s inequality, since $p ( 1 - p )$ is concave).

Assumption 1 (Uniform shift). A class-level scaffold shifts every task equally: $p _ { x } ^ { S } = p _ { x } + ( \sigma _ { C } - s _ { C } ) \in [ 0 , 1 ]$ for $x \in C ,$ , preserving variance $v _ { C } .$

Assumption 1 is an idealization. In practice, a shared strategy may help some tasks more than others and may leave the hardest tasks at zero success; we treat such tasks separately below.

Proposition 3 (Class-level scaffold benefit). For fixed v<sub>C</sub> and $s _ { C } \in [ 0 , \frac { 1 } { 2 } )$ ), $\bar { V } _ { C }$ increases with $s _ { C } .$

$$
\frac { \partial \bar { V } _ { C } } { \partial s _ { C } } = \frac { G - 1 } { G } ( 1 - 2 s _ { C } ) > 0 .\tag{9}
$$

Under Assumption $^ { l , }$ if a class-level scaffold raises the class success rate from $s _ { C }$ to $\sigma _ { C } ,$ , the change in class-average reward variance is

$$
\bar { V } _ { C } ^ { S } - \bar { V } _ { C } = \frac { G - 1 } { G } ( \sigma _ { C } - s _ { C } ) ( 1 - \sigma _ { C } - s _ { C } ) ,\tag{10}
$$

which is positive ifand only $i f s _ { C } < \sigma _ { C } < 1 - s _ { C }$

Proof. The derivative follows from Eq. (8). Under Assumption 1, scaffolding gives mean $\sigma _ { C }$ and unchanged variance $v _ { C }$ , so $\begin{array} { r } { \bar { V } _ { C } ^ { S } = \frac { G - 1 } { G } \big [ \sigma _ { C } ( 1 - \sigma _ { C } ) - v _ { C } \big ] } \end{array}$ . Subtracting gives

$$
\bar { V } _ { C } ^ { S } - \bar { V } _ { C } = \frac { G - 1 } { G } \big [ \sigma _ { C } ( 1 - \sigma _ { C } ) - s _ { C } ( 1 - s _ { C } ) \big ] = \frac { G - 1 } { G } ( \sigma _ { C } - s _ { C } ) ( 1 - \sigma _ { C } - s _ { C } ) .
$$

The first factor is positive if and only if $\sigma _ { C } > s _ { C }$ , and the second if and only if $\sigma _ { C } < 1 - s _ { C }$

The improvement depends only on the two class-level rates $s _ { C }$ and $\sigma _ { C }$ . Writing $\Delta _ { C } = \sigma _ { C } - s _ { C }$ for the scaffold effect, the improvement $\begin{array} { r } { \frac { G - 1 } { G } \dot { \Delta _ { C } } ( 1 - 2 s _ { C } - \Delta _ { C } ) } \end{array}$ is larger for classes with lower $s _ { C }$ at a fixed effect, and it is negative for every $\Delta _ { C } > 0$ once $s _ { C } \geq { \frac { 1 } { 2 } } ;$ a scaffold that raises the success rate of a class already above one half reduces the class-level signal. It is proportional to both the scaffold effect $\left( \sigma _ { C } - s _ { C } \right)$ and the factor $\left( 1 - \sigma _ { C } - s _ { C } \right)$ , and for a given $s _ { C }$ it is largest at $\begin{array} { r } { \sigma _ { C } = \frac { 1 } { 2 } } \end{array}$ . If the scaffold is so strong that $\sigma _ { C } \geq 1 - s _ { C }$ , the class-level signal no longer increases and can even decrease, although the success rate is higher. This formalizes why class-level scaffolds should raise success rates toward, but not far beyond, $\frac { 1 } { 2 }$

Instance-level scaffold and zero-success tasks. Class-level scaffolds mainly benefit tasks that the student can already partially solve $( p _ { x } > 0 )$ , lifting their success rates toward the high-signal regime. For tasks where the student’s success probability is zero, a shared strategy is often insufficient, and targeted support such as a partial solution or an intermediate result is needed.

Let $\mathcal { Z } _ { C } = \{ x \in C : p _ { \theta } ( x ) = 0 \}$ denote the set of zero-success tasks in class $C ,$ with $\alpha _ { C } = | \mathcal { Z } _ { C } | / | C |$ . These tasks produce only all-failure groups, contributing no reward-relative gradient or within-group reward variance:

$$
\bar { V } _ { C } = \frac { G - 1 } { G } \left( 1 - \alpha _ { C } \right) \mathbb { E } _ { x \not \in \mathcal { Z } _ { C } } \left[ p _ { x } ( 1 - p _ { x } ) \right] .\tag{11}
$$

Proposition 4 (Instance-level scaffold for zero-success tasks). Suppose an instance-level scaffold $S _ { \mathrm { i n s t } } [ x ]$ is applied to each zero-success task x ${ \mathrm { : } } \in { \mathcal { Z } } _ { C } ,$ , raising its success probability from 0 to $p _ { x } ^ { S } \in ( 0 , 1 )$ , while the remaining tasks are unchanged. The class-average variance increases by

$$
\bar { V } _ { C } ^ { S } - \bar { V } _ { C } = \frac { G - 1 } { G } \alpha _ { C } \mathbb { E } _ { x \in \mathcal { Z } _ { C } } \big [ p _ { x } ^ { S } ( 1 - p _ { x } ^ { S } ) \big ] > 0 .\tag{12}
$$

Proof. With instance-level scaffolds, the class-average reward variance becomes $\begin{array} { r } { \bar { V } _ { C } ^ { S } = \frac { G - 1 } { G } \bigl [ ( 1 - \alpha _ { C } ) \mathbb { E } _ { x \not \in \mathcal { Z } _ { C } } [ p _ { x } ( 1 - } \end{array}$ $p _ { x } ) ] + \alpha _ { C } \mathbb { E } _ { x \in { \mathcal Z } _ { C } } [ p _ { x } ^ { S } ( 1 - p _ { x } ^ { S } ) ] ]$ . Subtracting Eq. (11) yields Eq. (12), which is positive because $p _ { x } ^ { S } ( 1 - \dot { p } _ { x } ^ { S } ) > 0$ for $p _ { x } ^ { S } \in ( 0 , 1 )$ □

The improvement is proportional to $\alpha _ { C } \colon$ the more zero-success tasks there are, the greater the benefit, which is particularly relevant early in training. The requirement $p _ { x } ^ { S } < 1$ matters: a hint that reveals the complete solution makes every rollout succeed, so the group is again degenerate and contributes nothing. Propositions 3 and 4 thus formalize the complementary roles of the two scaffold levels: class-level scaffolds shift the difficulty of tasks the student can already attempt toward the high-signal regime, while instance-level scaffolds expand the set of tasks that contribute to training at all. This motivates using instance-level scaffolds mainly for tasks that remain at zero success under class-level scaffold, with hints that are partial rather than complete

Bounded rewards. The analysis extends to rewards in $[ 0 , 1 ]$ , such as the rubric scores used for KernelBench. A group yields zero advantage whenever all its rewards are equal, so for a reward taking values v with probabilities $\pi _ { v } ,$ the probability of a non-degenerate group is $1 - \textstyle \sum _ { v } \pi _ { v } ^ { G }$ , which is small whenever one value dominates, as the zero reward does on hard tasks. The expected within-group reward variance is $\begin{array} { r } { \mathbb { E } [ V \mid x ] = \frac { G - 1 } { G } \operatorname { V a r } ( R \mid x ) } \end{array}$ , and for a reward in [0, 1] with mean $\mu ,$ Var $\cdot ( R \mid x ) \leq \mu ( 1 - \mu )$ , with equality for binary rewards. Both quantities therefore vanish when nearly every rollout receives the same reward, and a scaffold helps in the same way: by making rewards within a group differ.

Scope of the analysis. The analysis above characterizes the learning signal available under the scaffolded prompt, i.e., the signal for updating $\pi _ { \theta } ( \cdot \mid x , S )$ . It does not model how improvements under scaffolds transfer to the unscaffolded policy $\pi _ { \theta } ( \cdot \mid x )$ , which is what we evaluate. In particular, with injection probability $q ,$ the expected class-level signal $( 1 - q ) \bar { V } _ { C } ( s _ { C } ) + q \bar { V } _ { C } ( \sigma _ { C } )$ is linear in q and would favor $q = 1$ whenever the scaffold helps. In practice, an always-on scaffold creates a mismatch between training and evaluation and encourages reliance on privileged information (cf. the $q _ { \mathrm { m a x } } { = } 1$ ablation in Section 4.3), which motivates capping q at $q _ { \mathrm { m a x } } < 1$ . Characterizing transfer remains future work.

## B Case Study: How the Teacher Gathers Evidence and Decides

## B.1 ALFWorld

This section gives the full record of one teacher update on ALFWorld, at step 30 of a separate training run with the same setting, taken from the teacher’s decision log. The tool returns are rebuilt from the logged training rollouts with the same tool code, and their lengths match the logged lengths exactly.

Task classes. ALFWorld tasks are played by issuing text commands such as $\Im \mathrm { O }$ to microwave 1, take mug 1 from countertop 1, or heat mug 1 with microwave 1. The classes involved in this update are Heat (heat an object, which is possible only in the microwave, then put it somewhere), Look (examine an object under a lamp, which requires turning the lamp on with use), Pick2 (put two objects of the same kind in one place), Clean (clean an object at the sink, then put it somewhere), and Cool (cool an object in the fridge, then put it somewhere). Each training game is attempted 8 times, and a game in which all 8 attempts fail gives no learning signal.

Scaffold before the update. Heat, Look, and Clean each have a class-level scaffold, with injection probabilities 0.45, 0.4, and 0.4; the other classes have none.

Initial evidence. The update starts from two evaluations on the held-out tasks V . Without a scaffold, the student solves 27.3% of the tasks: 6.2% of Heat, 19.6% of Look, 21.6% of Clean, 26.5% of Cool, 24.2% of Pick2, and 47.5% of Pick. On 20 games per class, evaluated with and without the current scaffold, success with the scaffold is 55% versus 0% on Heat, 40% versus 10% on Clean, and 45% versus 35% on Look.

Gathering evidence. The teacher made six tool calls, in this order:

1. $Q \mathrm { { s t a t } }$ (training statistics of the current block). Heat attempts succeed 3% of the time without a scaffold and 27% with it, and seven of eight Heat games without a scaffold fail on every attempt. In five of eight Look games that received the scaffold, every attempt still fails. In 30 of 39 Pick2 games, every attempt fails.

2. Memory M (records of past updates). In the previous update, attempts with a scaffold already did better than those without on Look (23% versus 14%) and Heat (18% versus 4%).

3–6. $Q _ { \mathrm { t r a j } }$ , four failed attempts from each of Pick2, Cool, Heat, and Look (the first 24 steps of each):

• Pick2: the student puts the first object down and picks it up again, and once tries to move both objects with one command, put both spraybottles into garbagecan 1, which the environment rejects.

• Cool: the student keeps putting the pot back on the stove and never goes to the fridge.

• Heat: 38 actions go to or use the coffee machine or the stove; none involve the microwave.

• Look: the student repeats examine cellphone 1 with desklamp, which the environment rejects, and never issues use, so the lamp is never turned on.

Diagnosis. The teacher’s diagnosis, which it wrote before proposing edits, makes five points:

• Heat is the weakest class without a scaffold (6.2% on V<sub>C</sub>; seven of eight training games without a scaffold fail on every attempt), while the current scaffold gives a large gain (55% versus 0%). The teacher therefore keeps showing Heat a scaffold often, but replaces its text with one written specifically for heating.

• Clean still benefits from its scaffold (34% versus 14% in training, 40% versus 10% on V ), so the teacher keeps it.

• Pick2 has the most games in which every attempt fails (30 of 39), and its failed attempts place only one object or invent a command that moves both objects at once, so the teacher proposes a step-by-step procedure for placing two objects.

• Look attempts repeat the invalid command examine <object> with desklamp and never use the lamp, so the teacher proposes a scaffold about turning the lamp on.

• Cool is improving without a scaffold, so the teacher leaves it without one for now.

Proposed edits. The teacher proposed new class-level text for each of the three failures it found, rewriting the current Heat and Look scaffolds and adding one for Pick2:

• Heat (q = 0.5): “For heat tasks, do not use the coffeemachine or stoveburner. Find and take the exact target object, go to microwave 1, then run ‘heat <object> with microwave 1’. After it is heated, carry that same object to the requested receptacle and put it in/on there once.”

• Look (q = 0.4): “For look-in-light tasks, first find and take the exact target object. Then go to a lamp such as desklamp 1 or floorlamp 1 and use the lamp with ‘use <lamp>’. Do not keep trying ‘examine <object> with desklamp’; the importantfinal step is to be carrying the object at the lamp and use the lamp.”

• Pick2 (q = 0.45): “For two-object tasks, place two separate objects with the requested name in the same receptacle. Do not use ‘both’ or a plural command. Take one target object, go to the receptacle, put it in/on there; thenfind a different second target object, take it, and put it in/on that same receptacle.”

It kept the Clean scaffold and wrote nothing for Cool.

Validation and decision. Each proposed scaffold was validated on 30 games per class from $V _ { C }$ (Table 9). The Heat scaffold improves on the current text (40.0% versus 30.0%), and the Look scaffold matches the current one, so both are adopted. The Pick2 scaffold measures below the no-scaffold condition (10.0% versus 13.3%), so the teacher drops it and leaves Pick2 without a scaffold. The final update replaces the Heat and Look scaffolds with the new text, keeps the Clean scaffold, and sets q to 0.5 for Heat, 0.4 for Look, 0.45 for Clean, and 0 for the other classes.

Table 9: Validation of the scaffolds proposed in this update: success rate (%) on 30 held-out games per class. Pick2 has no current scaffold.
<table><tr><td>Class</td><td>No scaffold</td><td>Current scaffold</td><td>Proposed scaffold</td><td>Outcome</td></tr><tr><td>Heat</td><td>6.7</td><td>30.0</td><td>40.0</td><td>adopted</td></tr><tr><td>Look</td><td>26.7</td><td>30.0</td><td>30.0</td><td>adopted</td></tr><tr><td>Pick2</td><td>13.3</td><td></td><td>10.0</td><td>dropped</td></tr></table>

Outcome. In the next block, attempts with the new scaffolds succeed far more often than attempts without them on Look (67% versus 4%) and Heat (51% versus 16%). At the next update, success without a scaffold on $V _ { C }$ has risen from 27.3% to 36.7% overall, from 6.2% to 16.7% on Heat, and from 19.6% to 34.7% on Look. Clean, whose scaffold was not changed, also rises, from 21.6% to 47.4%, so part of these gains comes from RL training itself. The Look and Heat scaffolds are removed three and four updates later, once they no longer help the student.

## B.2 Mathematical Reasoning

This section gives the full record of one teacher update in math, at step 100 of the training run shown in Figure 7, where the mean prefix ratio stops rising and begins to fall. The record is taken from the teacher’s decision log and the run’s per-update statistics. The log stores which tools the teacher called and in what order, its written diagnosis, and its proposed edits; it does not store the tool arguments or the returned contents, so we report the statistics the teache had access to at this update rather than the exact tool returns.

Setting. The student is Nemotron-1.5B (OpenMath-Nemotron-1.5B), trained on the 12,506 problems released with QuestA. The instance-level scaffold follows QuestA (Li et al., 2026a): the first $r _ { x }$ percent of a reference solution is prepended to problem x, with $r _ { x } \in [ 0 , 5 0 ]$ . Each problem starts at the ratio its QuestA difficulty filter selected it for: 1,853 problems at $r _ { x } = 5 0$ and 10,653 at $r _ { x } = 2 5$ , so no problem starts without a hint. Each update covers one block of 10 training steps, in which $2 { , } 5 6 0$ problems are each attempted 8 times. A problem whose 8 attempts all fail, or all succeed, gives no learning signal. The teacher can change $r _ { x }$ for groups of problems selected by their outcome in the current block (all-fail, all-pass, or mixed), by their current $r _ { x }$ , and by their success rate. Math has a single task class (Table 2), so its class-level scaffold consists of short notes that are added to a training prompt with injection probability q, with at most 8 notes in force at a time. A problem that is solved with $r _ { x } = 0$ is marked as graduated.

Scaffold before the update. Across the pool, the mean prefix ratio is 33.7. About 3,500 problems are at $r _ { x } = 2 5$ and about 4,200 at $r _ { x } = 5 0$ , and the rest are at intermediate ratios; 130 problems have been reduced to $r _ { x } = 0 ,$ , and 46 have graduated. Eight class-level notes are in force with $q = 0 . 2$ . They tell the student to check that the boxed answer is the quantity the problem asks for; to box the option letter in multiple-choice problems; not to stop after finding an intermediate result; to use a contingency table for two-attribute counts; to respect the order of indexed variables in extremal problems; to use direction vectors when counting lines and planes; to read “sufficient” and “necessary” as implications; and to follow inequality chains by index residue.

Initial evidence. In the current block, 575 of the 2,560 problems are all-fail, 1,829 are mixed, and 156 are all-pass, so 28.6% of the problems (the all-fail and all-pass ones) give no learning signal. Among problems that received the notes, 21.6% are all-fail, compared with 22.7% among problems that did not. The teacher also has the student’s hint-free success on the held-out tasks $V _ { C }$ (200 problems), which it uses to check that the student is not becoming dependent on hints.

Gathering evidence. The teacher made five tool calls, in this order:

$Q _ { \mathrm { s t a t } }$ (training statistics of the current block): the number of all-fail, mixed, and all-pass problems at each prefix ratio, and the outcomes of earlier ratio changes on problems seen again since (revisits).

2–4. $Q \mathrm { { s t a t } }$ three more times, to list individual problems (the selection arguments are not recorded).

5. $Q _ { \mathrm { t r a j } } .$ , to read failed attempts on all-fail problems.

Diagnosis. The teacher’s diagnosis, written before it proposed edits (in its notes, r is the prefix ratio $r _ { x } ,$ “probes” are the hint-free evaluations on $V _ { C }$ , and “text probability” is q):

“Cycle 10 still has a large zero-gradient bottleneck: 575/2560 groups are all-fail. Most are already highdose, but the 119 all-fail problems at $r \leq 2 5$ still have useful headroom; revisit evidence says raising prior all\_fail@25 to r45/r50 escaped about 55–60%, so those should be boosted. For all-fail problems at intermediate/high r with remaining headroom, raising toward 50 is stilljustified; prior all\_fail@45 → r50 escaped about 39%. Problems already mixed have gradient, so follow the standing slow-wean rule for succ>50%; all-pass groups can be annealed faster. Hint-free probes are not clearly deteriorating overall, so keep text probability unchanged, but add a couple ofconcise general skills aimed at observed max-dosefailures where ratio cannot help: two-way table/conditional-count discipline and geometric halving processes.”

The diagnosis makes four points:

• All-fail problems with room to raise $r _ { x }$ . Of the 575 all-fail problems, 119 have $r _ { x } \le 2 5$ . When the teacher earlier raised all-fail problems from $r _ { x } = 2 5$ to 45 or 50, about 55–60% of them were no longer all-fail when seen again, so the teacher raises these problems again.

• All-fail problems at a high ratio. Most all-fail problems are already at or near $r _ { x } = 5 0 .$ . For those below 50, raising $r _ { x }$ still helped in the past (about 39% of problems raised from 45 to 50 escaped all-fail), so the teacher raises them toward 50. For those already at 50, a larger prefix is not possible.

• Mixed and all-pass problems. Mixed problems already give a learning signal, so the teacher only reduces $r _ { x }$ slowly, for problems solved in more than half of their attempts. All-pass problems are reduced faster.

• Notes. Hint-free success on $V _ { C }$ is not clearly falling, so the teacher keeps q unchanged. For problems that fail at the maximum ratio, where a longer prefix cannot help, it proposes two new notes based on failures it read: one on two-way tables and conditional counts, and one on processes that halve a quantity at each stage.

Decision. The teacher made adjustments in opposite directions for different groups of problems (Table 10). It proposed two notes:

• “For problems with two attributes or categories, make a 2x2 table. Distinguish marginal countsfrom conditional counts such as ‘among the clayey loam’; subtract only within the same row/column, then add the cells the question asks for.”

• “When a process halves the remaining amount at each stage, successive equal-fraction stages usually take half as long as the previous one. Use a geometric sum for total time/amount, not a constant time per stage.”

It left q at 0.2. All-fail problems at $r _ { x } = 5 0$ were left unchanged.

Table 10: Prefix-ratio changes in this update. Ratios are capped at 50.
<table><tr><td>Outcome in the current block</td><td>Selection</td><td>Change in  $r _ { x }$ </td></tr><tr><td>All-fail</td><td> $0 \leq r _ { x } \leq 2 5$ </td><td>+20</td></tr><tr><td>All-fail</td><td> $2 6 \leq r _ { x } \leq 4 9$ </td><td>+10</td></tr><tr><td>Mixed</td><td> $r _ { x } \ge 1$  , success rate  $> 5 0 \%$ </td><td>-5</td></tr><tr><td>All-pass</td><td> $r _ { x } \ge 1$ </td><td>-10</td></tr></table>

Outcome. The four ratio changes were applied. The two notes were not: eight notes were already in force, the maximum, and the framework rejects note additions that would exceed it, so the notes in force stayed the same. Over the next three updates, the number of problems the teacher had reduced to $r _ { x } = 0$ grew from 130 to 177, 244, and 303, the number of graduated problems rose from 46 to 90, all-pass problems in each block rose from 156 to 241, and the mean prefix ratio fell from 33.7 to 33.3. At the next update the teacher reported, from revisits, that raising all-fail problems from $r _ { x } = 2 5$ to 50 had let 86 of 117 escape all-fail, and that cuts of about 10 points on all-pass problems were mostly safe, while larger cuts at high ratios sometimes turned a problem back to all-fail. Hint-free pass@1 on $V _ { C }$ rose from 28.8% at step 80 to 34.1% at step 120 and 38.3% at step 160.

As in ALFWorld, the update follows the student’s current ability: where more of the same support can still help, the teacher gives more of it; where the student already succeeds, it gives less; and where the support cannot be increased further, it proposes a different kind of support.

## C Evaluation Domains

We evaluate AUTOSCAFFOLD on four domains that differ in task structure, interaction mode, and reward.

ALFWorld. ALFWorld (Shridhar et al., 2021) is a text-based household environment in which the agent completes long-horizon tasks through multi-step planning and object interaction. It defines six task types, each used as a task class for class-level scaffolds: Pick & Place (Pick), Examine in Light (Look), Clean & Place (Clean), Heat & Place (Heat), Cool & Place (Cool), and Pick Two & Place (Pick2). We evaluate on the standard 140-game test split; an episode succeeds if the task is completed within the environment’s step limit, and we report the success rate.

Search-based QA. The model answers open-domain questions by interacting with a search tool, deciding when and what to query and how to combine the retrieved passages into a final answer. We train on NQ and HotpotQA and evaluate on three single-hop datasets, NQ (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al., 2023), and four multi-hop datasets, HotpotQA (Yang et al., 2018), 2Wiki (Ho et al., 2020), MuSiQue (Trivedi et al., 2022), and Bamboogle (Press et al., 2023). NQ and HotpotQA are in-domain; the others are out-of-domain. Exact match (EM) is used as the reward and the evaluation metric.

KernelBench. KernelBench (Ouyang et al., 2025) asks the model to write GPU kernels that reproduce a reference PyTorch implementation and run faster than it. Level 1 contains 100 relatively simple tasks involving basic operators (e.g., matrix multiplication); Level 2 contains 100 medium-difficulty tasks composed of multi-step operator combinations; Level 3 contains 50 challenging tasks involving full network architectures (e.g., AlexNet). We train on a dataset of 600 kernel generation tasks. Correctness is verified against the reference output, and we report correctness and Fast<sub>1</sub>, the fraction of tasks with a correct kernel that is faster than the PyTorch reference. Speed is measured on H200.

Math reasoning. The student performs multi-step reasoning and outputs a final answer, which is checked against the ground truth. We use eight competition-level benchmarks: the recent contests AIME 24 (Mathematical As sociation of America, 2024), AIME 25 (Mathematical Association of America, 2025), HMMT Feb 25 (Harvard MIT Mathematics Tournament, 2025), and BRUMO 25 (Brown University Mathematical Olympiad, 2025), together with AMC 23 (Mathematical Association of America, 2023), MATH-500 (Lightman et al., 2023), Minerva Math (Lewkowycz et al., 2022), and OlympiadBench (He et al., 2024). Following the corresponding baselines, the Qwen3-4B-Instruct setting (following SAGE) reports AIME 24/25, AMC 23, MATH-500, Minerva Math, and OlympiadBench (Table 5), and the Nemotron-1.5B setting (following QuestA) reports AIME 24/25, HMMT Feb 25, OlympiadBench, and BRUMO 25 (Table 6). We follow the evaluation settings of each baseline’s paper.

## D Teacher Cost

Cost per scaffold update. Each scaffold update consists of (i) investigation, in which the teacher queries $Q _ { \mathrm { s t a t } } , Q _ { \mathrm { t r a j } }$ and $Q _ { \mathrm { g r o u p } }$ and reads its memory; (ii) proposal; and (iii) held-out validation of class-level edits, which runs student rollouts under three conditions. Teacher calls run on an external API; validation rollouts run on the training GPUs. Table 11 reports the cost per domain. Validation dominates the time per update, while the API cost of the teacher is a few dollars per run.

Table 11: Teacher cost per domain. Per-update values are means over all scaffold updates in a run.
<table><tr><td></td><td>ALFWorld</td><td>Search QA</td><td>Math</td><td>KernelBench</td></tr><tr><td>Teacher wall-clock per update (min)</td><td>4</td><td>3</td><td>5</td><td>4</td></tr><tr><td>Validation wall-clock per update (min)</td><td>11</td><td>13</td><td>21</td><td>16</td></tr><tr><td>Total teacher API cost per run (USD)</td><td>2.35</td><td>3.13</td><td>5.57</td><td>2.76</td></tr></table>