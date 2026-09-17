# STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution

Yajie Yu   
School of Computer Science   
University of Birmingham   
Birmingham   
United Kingdom   
yxy616@student.bham.ac.uk   
Mark Lee   
School of Computer Science   
University of Birmingham   
Birmingham   
United Kingdom   
m.g.lee@bham.ac.uk   
Yue Feng   
School of Computer Science   
University of Birmingham   
Birmingham   
United Kingdom   
y.feng.6@bham.ac.uk

## Abstract

Large language models (LLMs) often suffer from capability stagnation in self-improvement training because fixed difficulty levels fail to adapt to their evolving proficiency. To address this issue, we propose STRETCH (Self-Taught Reasoning Evolution via Targeted CHallenge), a unified framework inspired by cognitive scaffolding theory. STRETCH introduces a dynamic Stretch Zone mechanism that continuously aligns question difficulty with the model’s solving capability. Within a single parameter space, the model alternates between a Scaffolder that generates adaptive, boundarypushing challenges and a Learner that that optimizes its solving trajectories through reinforcement learning. This dual-loop co-evolution effectively stabilizes training, mitigates reward hacking and promote progressive reasoning growth. Experiments on both negotiation and operation research benchmarks demonstrate that STRETCH consistently outperforms strong prompting and domain-specific baselines. Further scaffolder configuration analysis shows that dynamic difficulty alignment is critical for sustained capability improvement and synchronized reasoning evolution.<sup>1</sup>

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in complex reasoning domains, ranging from mathematical problemsolving to strategic social interactions (Sicilia et al., 2024; Yu and Feng, 2025; Patil and Jadon, 2025; Zeng et al., 2025; Wu et al., 2025a; Liu et al., 2026b). To push the boundaries of these models beyond imitation learning from humanannotated data, recent advancements have increasingly adopted post-training paradigms based on self-play exploration mechanism (Zhao et al., 2025;

![](images/d1c3e5c859e966a22a51cbee5e4f1ab90f3b869a7189da0ba02dc58f3091acd4.jpg)

(a)  
![](images/401fe682c4d0b8a301065550bf03145883b0fb7c56bdeba31773056504c24297.jpg)  
(b)  
Figure 1: Motivation of the STRETCH Framework. (a) Cognitive Zones: Comfort, stretch (optimal learning), and difficulty. (b) Paradigm Comparison: Top: static general questions leading to learner capability stagnation. Bottom: adaptive questions achieving dynamic difficulty matching via a learnable Scaffolder for progressive improving of Learner.

Guo et al., 2025; Zhao et al., 2026; Liu et al., 2026a). Despite these promising strides, current self-improvement frameworks face a critical bottleneck: they largely lack a mechanism for progressive capability improvement. Traditional reinforcement learning (RL) approaches typically benchmark models on static datasets or predefined difficulty levels (Schulman et al., 2017; Rafailov et al., 2024; Lee et al., 2025; Wei et al., 2026). Under such static setups, models easily get “stuck”: as the LLM masters the fixed data distribution, it quickly falls into a capability plateau, leading to premature convergence. Conversely, if initial tasks are excessively complex, the agent fails to find valid reasoning trajectories, resulting in catastrophic reward collapse.

To systematically overcome this stagnation, we draw inspiration from cognitive scaffolding theory (Wilson and Devereux, 2014; van Nooijen et al., 2024; Newen and Fabry, 2025), which partitions human learning into three cognitive territories (as illustrated in Figure 1a). For optimal learning, an agent should avoid the Comfort Zone (where repeatedly reinforcing basic knowledge leads to stagnation) and the Difficulty Zone (where tasks are inaccessible due to limited ability). Instead, learning must be anchored in the Stretch Zone—a dynamic cognitive boundary just beyond the Comfort Zone where achievements and challenges fluidly coexist. As shown in the paradigm comparison in Figure 1b (Top), traditional training paradigms fail because they rely on static general questions; as the model grows stronger, the Stretch Zone shifts upward, but the environment remains fixed, dropping the model back into the Comfort Zone. Breaking through early plateaus necessitates a dynamic paradigm that performs continuous difficulty matching (Figure 1b, Bottom), ensuring the generation of adaptive questions that scale in tandem with the model’s growing proficiency.

To achieve this cognitive-inspired alignment, we propose STRETCH, a unified self-taught framework for progressive LLM evolution. Unlike prior curriculum learning or self-play methods that rely on frozen environment generators or multiple disjoint models (Yang et al., 2026; Karlekar et al., 2026), STRETCH coordinates a unified large language model to seamlessly alternate between two complementary cognitive roles within a singular parameter space. Acting as a learnable Scaffolder, the model utilizes historical outcome feedback to dynamically generate adaptive multi-task constraints (e.g., strategic negotiation bottom lines or complex mathematical resource limits) that reside strictly within the model’s current Stretch Zone. Subsequently, switching to the Learner role, the model interacts with external environments to navigate these self-generated challenges. Because both roles share the same underlying weights, optimizing the Learner’s trajectories via a hybrid pipeline of Group Relative Policy Optimization (GRPO)

(Shao et al., 2024; Zhang et al., 2025) enhances the Scaffolder’s strategic capacity to propose realistic, perfectly aligned challenges. This joint parametric co-evolution creates a continuous selfimprovement loop.

We empirically evaluate the cross-domain generalization capability of STRETCH on two highly divergent constrained reasoning benchmarks: multiturn bargaining (the Craigslist dataset) and advanced mathematical operation research (the Mano Complex and Complex OR datasets). Experimental results demonstrate that STRETCH consistently breaks through the early capability plateaus that shackle traditional static RL, significantly outperforming state-of-the-art domain-specific baselines across both social game play and rigid mathematical optimization.

Our main contributions are as follows:

• Cognitive-Inspired Diffculty Alignment: We formalize the post-training of LLMs through cognitive scaffolding theory. By conceptualizing the Stretch Zone, we establish a concrete framework that utilizes continuous difficulty matching to overcome capability stagnation of model.

• Unified Parametric Co-evolution: We propose STRETCH, characterized by the simultaneous evolution of both the task-generating Scaffolder and the task-solving Learner within a single parameter space, eliminating the need for disjoint models.

• Cross-Domain Breakthrough: We design a robust self-taught loop powered by feedbackdriven task generation, achieving significant performance gains on both strategic negotiation and operation research tasks, proving the efficacy of our evolution framework.

## 2 Related Works

## 2.1 Co-Evolutionary Frameworks

Self-play has emerged as a promising paradigm for LLM self-improvement, where models generate and solve their own problems. Zhao et al. (2025) proposed Absolute Zero Reasoner (AZR), a selfplay reasoning framework that operates without any external human-annotated or distillation data. In parallel, Shao et al. (2024) introduced DeepSeek-Math, which leverages self-generated data for mathematical reasoning and introduces Group Relative

![](images/82cdf345d0702161bbde86ae80f1f237e73d8bf7c9f0da5818e4c7b51682c15e.jpg)  
Figure 2: The dual-loop asynchronous updating mechanism of STRETCH. Difficulty Alignment (Left Loop):π is updated to generate questions within the Learner’s Stretch Zone by driving a frozen Learner to sample solutions and evaluating its empirical accuracy. Exploration (Right Loop): $\pi _ { L e a r n e r }$ is updated to solve adaptive questions generated by a frozen Scaffolder.

Policy Optimization (GRPO) to enhance reasoning abilities while optimizing memory usage.

Multiple studies have explored the joint evolution of task generators and solvers (Chen et al., 2026a; Xiang et al., 2026; Luo et al., 2026).Guo et al. (2025) proposed GenEnv, a difficulty-aligned co-evolution framework between an LLM agent and a scalable environment simulator. Huang et al. (2025) introduced R-Zero, which separately optimizes a Challenger and a Solver initialized from a common base model. Yang et al. (2026) introduced TTCS, a test-time curriculum synthesis framework where a question synthesizer and a reasoning solver co-evolve from the same pretrained model. Sygkounas et al. (2026) proposed COvolve, which models the interaction between environment and policy designers as a two-player zero-sum game, ensuring adversarial co-evolution. Huang et al. (2026) presented G-Zero, a verifier-free co-evolutionary framework that drives continuous self-evolution through hint-induced response.

## 2.2 Cognitive Scaffolding in LLMs

Drawing from cognitive science, the concept of scaffolding has been increasingly applied to LLM reasoning. Cognitive scaffolding theory originates from the concept of the Zone of Proximal Development (ZPD) (McLeod, 2024). Wilson and Devereux (2014) provided a foundational articulation of scaffolding as “high challenge, high support”, arguing that effective scaffolding enables learners to achieve far beyond what they could accomplish individually. Newen and Fabry (2025) developed a pattern theory of scaffolding, characterizing how environmental resources contribute to the realization of mental abilities and offering a framework for understanding the functional role of scaffolding in cognitive systems. van Nooijen et al. (2024) showed how scaffolding regulates the flow of information within the learner’s working memory, thereby reducing cognitive load.

Recent work has operationalized this theory for LLMs (Chen et al., 2026b; Cui and Sachan, 2025; Wallis, 2026). Inspired by cognitive scaffolding, Kim et al. (2025) introduced SMART (Small Reasons, Large Hints), where LLMs provide targeted, selective guidance to augment small language model reasoning. Another line of research explores how prompt-level inductive biases serve as cognitive artifacts. Li et al. (2025) proposed SEELE, a supervision-aided RLVR framework that dynamically adjusts problem difficulty by appending hints of adaptive length, keeping rollout accuracy near the theoretically optimal 50% regime.

## 3 Methodology

To overcome the capability stagnation prevalent in static reinforcement learning, we introduce STRETCH, a progressive self-evolution framework. As illustrated in Figure 2, STRETCH employs a singular parameter space that seamlessly alternates between two cognitive roles: the Scaffolder $( \pi _ { S c a f f o l d e r } )$ and the Learner $( \pi _ { L e a r n e r } )$ Our dual-loop asynchronous updating mechanism structurally enforces continuous difficulty alignment and progressive exploration.

At a high level, STRETCH optimizes task validity, learnability, and replay-stabilized downstream

behavior jointly:

$$
\operatorname* { m a x } _ { \theta } \ \mathbb { E } _ { q \sim \pi _ { \theta } } [ \mathbb { I } _ { \mathrm { v a l i d } } ( q ) R _ { \mathrm { d i f f } } ( q ) ] + \lambda \mathcal { R } _ { \mathrm { r e p l a y } } ( \theta ) ,\tag{1}
$$

where $\mathbb { I } _ { \mathrm { v a l i d } } ( q )$ is supplied by the domain verifier, $R _ { \mathrm { d i f f } }$ favors tasks at the Learner’s capability boundary, and $\mathcal { R } _ { \mathrm { r e p l a y } } = - \mathcal { L } _ { \mathrm { S F T } }$ rewards likelihood on verified successful trajectories. Equation 1 is implemented by the alternating GRPO updates and epoch-level replay described below, rather than optimized as a single differentiable loss.

## 3.1 Role Switching within a Singular Space

Unlike traditional co-evolution frameworks that require maintaining and syncing two separate large language models (Guo et al., 2025; Yang et al., 2026; Zhou et al., 2026), STRETCH unifies task generation and task solving within a single base model $\pi _ { \theta }$ . The dual roles are activated purely via role-specific system prompts.

To enable stable mutual learning, we implement an alternating phase-update mechanism. During the Difficulty Alignment (Left Loop), we optimize the active Scaffolder $\pi _ { S c a f f o l d e r }$ while keeping a historical snapshot of the Learner frozen. Conversely, during the Exploration (Right Loop), we optimize the active Learner $\pi _ { L e a r n e r }$ against constraints generated by the frozen Scaffolder. This decoupling loop ensures that each updating iteration receives consistent and reliable reward signals.

## 3.2 Scaffolder Optimization

The primary objective of the left loop is to train the Scaffolder to dynamically discover the Learner’s Stretch Zone. It must learn to propose questions that are neither trivially easy (falling into the Comfort Zone) nor impossibly hard (falling into the Difficulty Zone.

Group Question Generation. Given a base task context $c ,$ the active Scaffolder $\pi _ { S c a f f o l d e r }$ samples a group of n candidate adaptive questions:

$$
Q = \{ q _ { 1 } , q _ { 2 } , \dots , q _ { m } \} \sim \pi _ { S c a f f o l d e r } ( \cdot \mid c )\tag{2}
$$

Frozen Learner Rollout. To empirically measure the difficulty of these generated questions, we utilize the frozen Learner $\pi _ { L e a r n e r } ^ { \mathrm { ( F r o z e n ) } }$ . For each candidate question $q _ { i }$ , the frozen Learner independently samples n diverse reasoning trajectories (solutions):

$$
S _ { i } = \{ s _ { i 1 } , s _ { i 2 } , . . . , s _ { i m } \} \sim \pi _ { L e a r n e r } ^ { ( \mathrm { F r o z e n } ) } ( \cdot \mid q _ { i } )\tag{3}
$$

Difficulty Alignment Reward. An external LLMas-a-Judge evaluates the correctness of each solution $s _ { i j }$ . The empirical success rate (accuracy) for a given question $q _ { i }$ is computed as $\begin{array} { r l } { \hat { p } _ { i } } & { { } = } \end{array}$ $\textstyle { \frac { 1 } { m } } \sum _ { j = 1 } ^ { m }$ I(correct). To explicitly force the Scaffolder to target the Stretch Zone, we formulate a difficulty-alignment reward that peaks at 50% success rate:

$$
\begin{array} { r } { R _ { d i f f } ( q _ { i } ) = ( \hat { p } _ { i } ( 1 - \hat { p } _ { i } ) ) ^ { \alpha } } \end{array}\tag{4}
$$

where α curves the reward sharpness. Since a Bernoulli outcome with success probability $p$ has variance $p ( 1 - p )$ , this reward is maximized at $p \ = \ 0 . 5 \mathrm { : }$ the regime with the most informative success/failure feedback for relative-policy updates. Validity is not inferred from the target rate alone: infeasible OR constraints receive zero verifier reward, while negotiation trajectories are replay-regularized using successful, rule-consistent dialogues. This reward penalizes the Scaffolder if the frozen Learner effortlessly solves all samples $( \hat { p } _ { i } \to 1 )$ or fails entirely $( \hat { p } _ { i } \to 0 )$

GRPO Update for Scaffolder. We normalize the rewards within the group of m questions to compute the advantages $A _ { j }$ , and then update Scaffolder’s parameters $\theta$ using the GRPO objective to maximize the probability of generating difficultyaligned questions:

$$
\begin{array} { c } { { \displaystyle { \mathcal { L } } _ { S c a f f o l d e r } ( \theta ) = { \mathbb { E } } \Bigg [ \sum _ { j = 1 } ^ { m } \operatorname* { m i n } \Big ( r _ { j } ( \theta ) A _ { j } , } } \\ { { \displaystyle \mathrm { c l i p } \big ( r _ { j } ( \theta ) , 1 - \epsilon , 1 + \epsilon \big ) A _ { j } \Big ) \Bigg ] } } \end{array}\tag{5}
$$

where $\begin{array} { r } { r _ { j } ( { \boldsymbol { \theta } } ) = \frac { \pi _ { S c a f f o l d e r } ( q _ { j } | { c } ) } { \pi _ { o l d } ( q _ { j } | { c } ) } } \end{array}$ is the probability ratio for the Scaffolder’s responses.

## 3.3 Learner Optimization

After the Scaffolder’s parameters are updated to target the model’s cognitive boundaries, we freeze it and shift to the right loop. The objective here is to advance the Learner’s reasoning capabilities to conquer newly generated constraints.

Adaptive Constraint Specification. The newly frozen Scaffolder $\pi _ { S c a f f o l d e r } ^ { ( \mathrm { F r o z e n } ) }$ is prompted to generate a specific, difficulty-aligned question q<sub>adapt</sub> based on the training context.

Group Solution Sampling. The active Learner $\pi _ { L e a r n e r }$ interacts with $q _ { a d a p t }$ to sample a group of m candidate reasoning trajectories:

$$
S = \{ s _ { 1 } , s _ { 2 } , . . . , s _ { m } \} \sim \pi _ { L e a r n e r } ( \cdot \mid q _ { a d a p t } )\tag{6}
$$

Accuracy Evaluation and GRPO Update. The LLM-as-a-Judge evaluates each trajectory $s _ { j }$ for logical correctness and constraint adherence, assigning an accuracy-based reward $R _ { a c c } ( s _ { j } )$ . For OR tasks, $R _ { a c c }$ is a binary reward strictly based on code execution and optimality. For Negotiation tasks, $R _ { a c c }$ is a joint score assigned by the LLMas-a-Judge evaluating deal success and strategic consistency. We compute the relative advantages $A _ { j }$ within this group of m solutions. The Learner’s parameters $\theta$ are then updated via the GRPO objective to master the current Stretch Zone:

$$
\begin{array} { r l l } & { \displaystyle \mathcal { L } _ { \mathrm { L e a r n e r } } ( \theta ) = \mathbb { E } \Bigg [ \sum _ { j = 1 } ^ { m } \operatorname* { m i n } \Big ( \rho _ { j } ( \theta ) A _ { j } , } \\ & { \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad } \\ & { \displaystyle \quad \quad \quad \quad \quad \mathrm { c l i p } \big ( \rho _ { j } ( \theta ) , 1 - \epsilon , 1 + \epsilon \big ) A _ { j } \Big ) \Bigg ] } \end{array}\tag{7}
$$

where $\begin{array} { r } { \rho _ { j } ( \theta ) = \frac { \pi _ { L e a r n e r } \left( s _ { j } \left| q _ { a d a p t } \right. \right) } { \pi _ { o l d } \left( s _ { j } \left| q _ { a d a p t } \right. \right) } } \end{array}$ is the probability ratio for the Learner’s trajectories.

## 3.4 The Progressive Spiral with Golden Experience Replay

To prevent either cognitive role from overoptimizing or collapsing, the dual objectives of STRETCH are integrated into a micro-interleaved optimization loop. At each training step, the framework performs exactly one Scaffolder update followed immediately by one Learner update, culminating in an epoch-level Golden Experience Replay to solidify behavioral patterns.

Concretely, the execution flow is structured as: Micro-Interleaved GRPO Updates for Scaffolder and Learner (Step-Level).

• Scaffolder Turn: The active Scaffolder $\pi _ { S c a f f o l d e r }$ samples questions, where the shared parameters $\theta$ are updated to push the difficulty boundary.

• Learner Turn: The active Learner π<sub>Learner</sub> samples solutions against the updated constraints, and θ is updated again via GRPO to master the new difficulty.

This step-by-step alternation ensures that the task difficulty (Scaffolder) and the solving capacity (Learner) evolve in exact lockstep, preventing catastrophic gradient divergence.

Golden Experience Replay (Epoch-Level). After N interleaved steps, we harvest the “golden interactions” from the successful trials. These consist of both the high-quality adaptive questions generated by the Scaffolder and the optimal reasoning traces produced by the Learner. We then format them as standard prompt-completion pairs $( x , y ) -$ where x represents either the base context or the generated constraint and $y$ represents the desired output. These pairs are merged into a unified online replay buffer $\mathcal { D } _ { \mathrm { g o l d } }$ . The unified model $\pi _ { \theta }$ is then optimized via Supervised Fine-Tuning (SFT) cross-entropy loss over this mixed distribution:

$$
\mathcal { L } _ { \mathrm { S F T } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \mathrm { g o l d } } } \left[ \log \pi _ { \theta } ( y \mid x ) \right]\tag{8}
$$

This combined architecture acts as a powerful self-stabilizing engine. While the step-level interleaved GRPO drives relentless exploration and boundary-pushing, the epoch-level Golden Experience Replay serves as a behavioral regularizer. It anchors the shared parameters in well-formed, highly logical trajectories, effectively mitigating the format collapse commonly observed in longhorizon reinforcement learning. Ultimately, consolidating the Learner’s optimal trajectories inherently enhances the Scaffolder’s precision in calibrating difficulty-aligned constraints, ensuring a continuous, self-sustaining upward spiral of capability.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Evaluation Metrics. To rigorously evaluate the generalizability and robustness of STRETCH, we select two constraint-solving domains: dynamic adversarial interactions and mathematical planning.

• Social Negotiation (Soft Constraints): We evaluate on the CraigslistBargain dataset (He et al., 2018), a classic multi-turn buyer-seller negotiation scenario. In this domain, constraints (e.g., counterpart’s bottom line and patience) are implicit and constantly shifting. Following standard protocols (Ahmad et al., 2023; Liu et al., 2025), we evaluate model performance using three metrics: Average Turn (AT) (lower is better for efficiency), Success Rate (SR) (rate of reaching a valid agreement), and Price Gap (PG) (the fraction of the final selling price relative to the initial proposed price, indicating negotiation profitability).

• Operation Research (Hard Constraints): We utilize two hardcore mathematical programming benchmarks: Mano Complex (Huang et al., 2024b) and Complex OR (Xiao et al., 2024). These tasks require the agent to parse real-world resource constraints into executable optimization code. The evaluation relies on objective programmatic metrics: Code Execution Rate (ER) (syntactic and runtime correctness) and Solving Accuracy (SA) (whether the generated code yields the optimal objective value while strictly satisfying all mathematical constraints) (Jiang et al., 2025).

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">Mano Complex</td><td colspan="2">Complex OR</td></tr><tr><td>ER (%)</td><td>SA (%)</td><td>ER (%)</td><td>SA (%)</td></tr><tr><td rowspan="8">Qwen2.5-7B-Instruct</td><td>Vanilla</td><td>36.4</td><td>31.6</td><td>16.7</td><td>10.5</td></tr><tr><td>Vanilla + CoT</td><td>42.3</td><td>38.1</td><td>25.6</td><td>19.1</td></tr><tr><td>ORLM</td><td>57.1</td><td>50.6</td><td>51.8</td><td>41.3</td></tr><tr><td>LLMOPT</td><td>62.7</td><td>54.1</td><td>56.0</td><td>53.7</td></tr><tr><td>StepORLM</td><td>65.8</td><td>62.4</td><td>57.3</td><td>52.6</td></tr><tr><td>Absolute Zero</td><td>63.4</td><td>60.1</td><td>56.8</td><td>54.1</td></tr><tr><td>R-Zero</td><td>64.8</td><td>61.7</td><td>58.2</td><td>53.2</td></tr><tr><td>STRETCH (Ours)</td><td>67.9</td><td>63.6</td><td>58.7</td><td>55.0</td></tr><tr><td rowspan="8">Llama-3.1-8B-Instruct</td><td>Vanilla</td><td>31.9</td><td>26.5</td><td>15.3</td><td>11.4</td></tr><tr><td>Vanilla + CoT</td><td>38.4</td><td>34.7</td><td>30.6</td><td>25.3</td></tr><tr><td>ORLM</td><td>52.5</td><td>47.6</td><td>46.1</td><td>42.3</td></tr><tr><td>LLMOPT</td><td>56.9</td><td>50.8</td><td>52.2</td><td>47.4</td></tr><tr><td>StepORLM</td><td>58.3</td><td>49.5</td><td>53.6</td><td>45.8</td></tr><tr><td>Absolute Zero</td><td>58.9</td><td>53.7</td><td>53.3</td><td>48.4</td></tr><tr><td>R-Zero</td><td>60.5</td><td>52.3</td><td>55.3</td><td>46.8</td></tr><tr><td>STRETCH (Ours)</td><td>61.2</td><td>54.8</td><td>56.2</td><td>49.5</td></tr></table>

Table 1: Main experimental results on Operation Research (OR) tasks. We report Execution Rate (ER) and Solving Accuracy (SA) on both the Mano Complex and Complex OR, where the higher values of all reported metrics indicate superior performance.

Baselines. We benchmark our framework against both general-purpose Large Language Models and domain-specific methodologies. For the Negotiation task, we compare against (1) Prompt-based methods: Vanilla LLM, GDPZero (Yu et al., 2023), and Pro-CoT (Deng et al., 2023); and (2) Trainingbased methods: DPDP (He et al., 2024) and DMNA (Liu et al., 2025). For the Operation Research (OR) task, we evaluate against general reasoning baselines (Vanilla, CoT) (Wei et al., 2022) and specialized OR-LLM frameworks: ORLM (Huang et al., 2024a), LLMOPT(Jiang et al., 2025), and StepORLM (Zhou et al., 2026). We further include controlled implementations of two closely related self-play approaches, Absolute Zero (AZR) (Zhao et al., 2025) and R-Zero (Huang et al., 2025), using the same task contexts and evaluation metrics as STRETCH. In contrast to STRETCH’s shared Scaffolder–Learner parameterization, epoch-level Golden Experience Replay, and explicit target difficulty, these methods do not instantiate this threepart design in our setting.

Implementation Details. Our unified Scaffolder-Learner framework is instantiated on two opensourced base models: Qwen2.5-7B-Instruct (Yang et al., 2025) and Llama-3.1-8B-Instruct (Llama Team, 2024). During the micro-interleaved difficulty alignment and exploration phase, the Scaffolder generates n = 8 adaptive questions, and the Learner samples m = 8 trajectories per question. For the Negotiation tasks, the external LLMas-a-Judge (GPT-4o-mini) (OpenAI et al., 2024) evaluates deal outcomes to compute the rewards. For the OR tasks, an automated Python compiler along with GPT-4o-mini (OpenAI et al., 2024) acts as the environment judge, returning binary rewards for execution and optimality. All experiments are conducted using LoRA (Hu et al., 2022) fine-tuning on two NVIDIA H100 GPUs.

For more experimental configuration details and related prompts, please refer to the Appendix A and Appendix D.

## 4.2 Main Results

The main experimental results across the Operation Research (OR) and Social Negotiation benchmarks are summarized in Table 1 and Table 2, respectively. Absolute Zero and R-Zero are reported in these main tables as direct self-play baselines. Their results are obtained from controlled implementations under the same task contexts and evaluation protocols as STRETCH; they should therefore be interpreted as in-domain comparisons rather than a universal ranking over the original papers’ evaluation suites. Overall, STRETCH improves upon both general-purpose prompting methods and domainspecific training frameworks across two backbone models, demonstrating cross-domain generalization of the unified co-evolution paradigm.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">CraigslistBargain</td></tr><tr><td>AT</td><td>SR PG</td></tr><tr><td rowspan="6">Qwen 2.5-7B -Instruct</td><td>Vanilla</td><td>10.18 0.33</td><td>0.83</td></tr><tr><td>GDPZero</td><td>9.36 0.54</td><td>0.87</td></tr><tr><td>Pro-CoT</td><td>9.25 0.47</td><td>0.81</td></tr><tr><td>DPDP</td><td>8.78 0.68</td><td>0.73</td></tr><tr><td>DMNA Absolute Zero</td><td>7.94 0.71 7.88 0.67</td><td>0.67 0.75</td></tr><tr><td>R-Zero</td><td>7.56 0.72</td><td>0.69</td></tr><tr><td rowspan="6">Llama-3.1- 8B -Instruct</td><td>STRETCH</td><td>7.32 0.79</td><td>0.61</td></tr><tr><td>Vanilla</td><td>9.52 0.41</td><td>0.91</td></tr><tr><td>GDPZero</td><td>8.09 0.57</td><td>0.85</td></tr><tr><td>Pro-CoT DPDP</td><td>7.71 0.69</td><td>0.83 0.84</td></tr><tr><td>DMNA</td><td>6.83 0.65 7.36</td><td>0.72 0.76</td></tr><tr><td>Absolute Zero</td><td>7.13 0.66</td><td>0.79</td></tr><tr><td rowspan="3"></td><td>R-Zero</td><td>7.46 0.73</td><td>0.84</td></tr><tr><td></td><td></td><td></td></tr><tr><td>STRETCH</td><td>6.96</td><td>0.75 0.73</td></tr></table>

Table 2: Negotiation performance on CraigslistBargain dataset. Performance is evaluated via Average Turns (AT, ↓), Success Rate (SR, ↑), and Price Gap (PG, ↓).

Superior Performance on Hard Constraints (OR). As shown in Table 1, pure reasoning baselines (Vanilla and Vanilla+CoT) struggle significantly with complex mathematical problems. While optimization-specific language models (e.g., ORLM, LLMOPT, and StepORLM) improve performance via static domain-specific fine-tuning, STRETCH further raises the upper bound. On the highly challenging Complex OR dataset, STRETCH built upon Qwen2.5-7B-Instruct achieves an Execution Rate (ER) of 58.7% and a Solving Accuracy (SA) of 55.0%, compared with 57.3% ER and 52.6% SA for StepORLM. It also exceeds the controlled AZR and R-Zero baselines on both Mano Complex and Complex OR. For example, on Mano Complex with Qwen2.5-7B-Instruct, STRETCH reaches 67.9% ER and 63.6% SA, compared with 63.4% and 60.1% for AZR and 64.8% and 61.7% for R-Zero. On the Llama-3.1-8B-Instruct backbone, STRETCH obtains the highest SA of 54.8% on Mano Complex and 49.5% on Complex OR. These results support the value of coupling adaptive difficulty with shared-role consolidation, rather than relying on self-play alone.

Strategy Mastery in Soft Constraints (Negotiation). Negotiation represents a highly adversarial environment where an agent must balance reaching an agreement (SR) with maximizing its own profit (represented by a lower Price Gap, PG) in as few turns as possible (AT). On the Qwen2.5-7B-Instruct backbone, STRETCH obtains the highest SR (0.79), the lowest PG (0.61), and the shortest AT (7.32). It improves over R-Zero by 0.07 SR and 0.08 lower PG, and over AZR by 0.06 SR and 0.07 lower PG. On Llama-3.1-8B-Instruct, STRETCH again achieves the best SR (0.75) and PG (0.73), while DPDP attains the shortest AT (6.83). Thus, the result is a success–profit advantage rather than a claim that STRETCH is best on every metric for every backbone. The lower PG indicates that the model does not merely compromise to reach agreement; its co-evolutionary training also improves bargaining outcomes.

In summary, the consistent superiority of STRETCH on both rigid mathematical planning (Table 1) and flexible strategic game play (Table 2) empirically validates that our unified, dual-role parametric co-evolution mechanism can successfully break the capability stagnation inherent in static RL pipelines. For validating the generalization of STRETCH on simpler datasets, please refer to Appendix C.

## 4.3 Ablation Studies: The Necessity of Difficulty Alignment

To thoroughly investigate the necessity of dynamic difficulty alignment, we conduct an extensive ablation study over the co-evolution process. We compare STRETCH against following Scaffolder configurations using the Qwen2.5-7B-Instruct on the Mano Complex and Complex OR.

Untrained-Frozen Scaffolder: The Scaffolder is frozen at its initialized state without undergoing any updates, generating problems based solely on its base capabilities. SFT-Frozen Scaffolder: The Scaffolder undergoes an initial SFT phase to learn basic constraint generation patterns but remains strictly frozen during the multi-epoch selfplay. External Scaffolder (GPT-4o-mini): We replace the internal co-evolving Scaffolder with an external model. Binary Reward: The Scaffolder is dynamically updated, but $R _ { d i f f }$ is replaced with a coarse binary signal. It receives a reward of 0 if the Learner answers all samples correctly or fails entirely, and 1 for any mixed accuracy.

![](images/04030aa109362d41cd8ee4ebe82adad0151d5f0a47dcbf65ae397e1ccf44d332.jpg)  
Figure 3: Ablation study on the necessity of difficulty alignment over a 6-epoch training on Learner by using the Qwen2.5-7B-Instruct backbone. We report Solving Accuracy (SA) on the Mano Complex (left) and Complex OR (right) datasets.

![](images/d72f21921cbbaada96bcc74cc7851c12759f95ad5d035a73bb3d3c669c592615.jpg)  
Figure 4: Quantitative co-evolution of question complexity (Scaffolder) and reasoning depth (Learner). The average token length of the generated questions and the successful reasoning trajectories exhibit a synchronized upward trend.

The performance across 6 epochs (visualized in Figure 3) reveals critical insights into how task difficulty impacts solving capacity.

First, static scaffolder inevitably leads to capability stagnation. As observed in both benchmarks, the Untrained-Frozen variant serves as a lower bound. While the SFT-Frozen variant establishes a significantly higher initial baseline, its performance decisively flatlines after Epoch 2. Because the constraint difficulty remains fixed, the Learner rapidly exhausts the training value of the generated questions, collapsing into a Comfort Zone.

Second, disjoint capabilities hinder long-term difficulty alignment. Relying on the External

Scaffolder yields strong early-epoch performance, as the proprietary model initially proposes highly challenging constraints. However, its trajectory noticeably plateaus midway. Because the external model cannot structurally internalize the Learner’s specific cognitive boundaries and benefit from shared golden experience replay, it ultimately fails to calibrate its questions to the Learner’s nuanced capacity shifts.

Finally, the contrast between Binary Reward and STRETCH underscores the necessity of precise difficulty alignment. Driven by a flat binary reward, the Binary Reward Scaffolder randomly oscillates between generating trivial and impossible constraints. This erratic behavior denies the Learner consistent gradient signals, resulting in severe performance variance and unpredictable drops. In stark contrast, STRETCH sustains a stable upward spiral. By utilizing the continuous reward $R _ { d i f f } .$ , the Scaffolder is forcefully anchored to the exact edge of the Learner’s capabilities.

These ablation results empirically prove our core hypothesis: dynamic difficulty alignment within the Stretch Zone is the essential driver for continuous LLM evolution.

## 4.4 Analysis of Co-Evolution Complexity

To look into the Stretch Zone visually, we quantitatively analyze the structural evolution of both the Scaffolder’s generated tasks and the Learner’s successful solutions. We utilize the Average Token Length as a proxy for cognitive complexity. For the Scaffolder, longer constraints indicate the intricate piecewise mathematical limits. For the Learner, longer successful trajectories reflect more extensive reasoning, profound logical planning, and complex code generation.

<table><tr><td rowspan="2">Configuration</td><td rowspan="2">Backbone</td><td colspan="2">Mano Complex</td><td colspan="2">CraigslistBargain</td></tr><tr><td>ER (%)</td><td>SA (%)</td><td>SR</td><td>PG</td></tr><tr><td rowspan="2">Separate LoRAs</td><td>Qwen2.5-7B-Instruct</td><td>63.2</td><td>55.6</td><td>0.64</td><td>0.76</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>57.8</td><td>52.6</td><td>0.67</td><td>0.82</td></tr><tr><td rowspan="2">STRETCH (shared)</td><td>Qwen2.5-7B-Instruct</td><td>67.9</td><td>63.6</td><td>0.79</td><td>0.61</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>61.2</td><td>54.8</td><td>0.75</td><td>0.73</td></tr><tr><td>No Golden Experience Replay</td><td>Qwen2.5-7B-Instruct</td><td>54.3</td><td>44.7</td><td>0.47</td><td>0.76</td></tr><tr><td>Target p = 0.8 (easy)</td><td>Qwen2.5-7B-Instruct</td><td>62.4</td><td>56.7</td><td>0.73</td><td>0.71</td></tr><tr><td>Target p = 0.2 (hard)</td><td>Qwen2.5-7B-Instruct</td><td>57.8</td><td>50.5</td><td>0.66</td><td>0.83</td></tr><tr><td>STRETCH (p = 0.5)</td><td>Qwen2.5-7B-Instruct</td><td>67.9</td><td>63.6</td><td>0.79</td><td>0.61</td></tr></table>

Table 3: Analysis of parameter sharing, Golden Experience Replay, and target difficulty on Mano Complex and CraigslistBargain.

As illustrated in Figure 4, both roles exhibit a remarkable, synchronized upward trajectory across the 6-epoch evolution. In the Operation Research domain, as the Scaffolder’s question descriptions grow from an average of 35 tokens to 156 tokens, the Learner’s corresponding optimal code and reasoning traces expand from 123 tokens to over 416 tokens. This synchronized escalation provides definitive empirical evidence for parametric co-evolution. The Scaffolder is not simply generating impossible noise; rather, it systematically calibrates the difficulty. Concurrently, the Learner is not collapsing under harder constraints; instead, it matches the escalating difficulty by unlocking deeper reasoning capacities.

To provide a concrete, intuitive understanding of how this structural expansion translates into semantic difficulty, Table 4 presents actual text segments generated by the STRETCH Scaffolder across different training epochs.

As observed in the examples, the Scaffolder does not merely pad the prompt with meaningless tokens to exploit the reward function. Instead, it systematically escalates the cognitive challenge, anchoring the constraints precisely within the Learner’s Stretch Zone. For concrete qualitative examples illustrating how these token expansions translate into explicit mathematical logic and negotiation strategies, please refer to Appendix B.

## 4.5 Analysis of Shared Roles, Replay, and Target Difficulty

Table 3 examines three mechanisms that make the shared co-evolution process stable and productive. First, sharing one adapter between the Scaffolder and Learner is consistently stronger than maintaining separate role-specific LoRAs: it improves both OR execution/solving and negotiation success/profit on the two backbones.

Second, Golden Experience Replay is essential for preserving useful behavior while the task distribution becomes harder. Without replay, performance drops sharply and the configuration exhibits format collapse by Epoch 4. Replaying highquality trajectories therefore serves not only as a performance buffer, but also as a stabilizer for longhorizon self-play updates.

Finally, the boundary target p = 0.5 is stronger than both an easier target $( p = 0 . 8 )$ and a harder target (p = 0.2). The easy setting supplies problems that are too often already solved, whereas the hard setting yields fewer useful successful trajectories. The balanced target achieves 67.9 ER and 63.6 SA on Mano Complex, together with 0.79 SR and 0.61 PG in negotiation. Together, these results support maintaining tasks near the current capability boundary.

## 5 Conclusion

In this paper, we introduced STRETCH, a unified framework for progressive self-improvement in verifiable, constraint-grounded LLM tasks. By conceptualizing the Stretch Zone and unifying the Scaffolder and Learner within a single parameter space, STRETCH replaces static training with a continuous, difficulty-aligned co-evolutionary loop. The controlled comparisons and component analyses show the benefits of shared role information, replay stabilization, and a boundary-level difficulty target on our OR and negotiation settings. These findings support STRETCH as a practical approach to self-improvement where reliable task generation and verification are available.

## Limitations

Dependency on Reward Verifiability and Simulation. The efficacy of the difficulty-alignment reward and the Learner’s accuracy reward inherently relies on environmental feedback. In OR, invalid or infeasible constraints are deterministically rejected by the programmatic verifier. In negotiation, however, GPT-4o-mini serves as both a seller simulator and part of the reward/evaluation pipeline. This creates simulator dependence: improvements may partly reflect adaptation to that model’s behaviors and preferences, and the service can change over time. We therefore limit our claims to settings with reliable task generation and verification, document prompts and deterministic conversation seeds in the released code, and design the pipeline so that GPT-4o-mini can be replaced by an alternative evaluator. Independent human or cross-simulator evaluation remains necessary to establish transfer beyond this setup.

Generalization to Unconstrained Environments. STRETCH is currently validated in domains where cognitive complexity and task constraints can be formally defined and objectively evaluated. Extending this parametric co-evolution to purely open-ended tasks—where objective difficulty gradients are inherently subjective, multidimensional, or lack a clear mathematical boundary—presents a broader challenge. Formulating a computable and universally applicable “Stretch Zone” for subjective, unconstrained domains is an open question that warrants further theoretical investigation.

Scope of Claims. Accordingly, our empirical claims are restricted to the evaluated, constraintgrounded settings; they do not establish general open-ended or autonomous LLM evolution.

## References

Zishan Ahmad, Suman Saurabh, Vaishakh Menon, Asif Ekbal, Roshni Ramnani, and Anutosh Maitra. 2023. INA: An integrative approach for enhancing negotiation strategies with reward-based dialogue agent. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 2536–2549, Singapore. Association for Computational Linguistics.

Jiaqi Chen, Bang Zhang, Ruotian Ma, Peisong Wang, Xiaodan Liang, Zhaopeng Tu, Xiaolong Li, and Kwan-Yee K. Wong. 2026a. SPC: Evolving selfplay critic via adversarial games for LLM reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Xuanzhong Chen, Zile Qiao, Guoxin Chen, Liangcai Su, Zhen Zhang, Xinyu Wang, Pengjun Xie, Fei Huang, Jingren Zhou, Yong Jiang, and Ting Chen. 2026b. Expanding the capability frontier of LLM agents with ZPD-guided data synthesis. In The Fourteenth International Conference on Learning Representations.

Peng Cui and Mrinmaya Sachan. 2025. Investigating the zone of proximal development of language models for in-context learning. In Findings ofthe Association for Computational Linguistics: NAACL 2025, pages 6485–6498, Albuquerque, New Mexico. Association for Computational Linguistics.

Yang Deng, Lizi Liao, Liang Chen, Hongru Wang, Wenqiang Lei, and Tat-Seng Chua. 2023. Prompting and evaluating large language models for proactive dialogues: Clarification, target-guided, and noncollaboration. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10602–10621, Singapore. Association for Computational Linguistics.

Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, and Mengdi Wang. 2025. Genenv: Difficulty-aligned co-evolution between llm agents and environment simulators. Preprint, arXiv:2512.19682.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. 2018. Decoupling strategy and generation in negotiation dialogues. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2333–2343, Brussels, Belgium. Association for Computational Linguistics.

Tao He, Lizi Liao, Yixin Cao, Yuanxing Liu, Ming Liu, Zerui Chen, and Bing Qin. 2024. Planning like human: A dual-process framework for dialogue planning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4768–4791, Bangkok, Thailand. Association for Computational Linguistics.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations.

Chengsong Huang, Haolin Liu, Tong Zheng, Runpeng Dai, Langlin Huang, Jinyuan Li, Zongxia Li, Zhepei Wei, Yu Meng, and Jiaxin Huang. 2026. G-zero: Self-play for open-ended generation from zero data. Preprint, arXiv:2605.09959.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. 2025. R-zero: Selfevolving reasoning llm from zero data. Preprint, arXiv:2508.05004.

Chenyu Huang, Zhengyang Tang, Dongdong Ge, Shixi Hu, Ruoqing Jiang, Benyou Wang, Zizhuo Wang, and Xin Zheng. 2024a. Orlm: A customizable framework in training large models for automated optimization modeling. arXiv e-prints, pages arXiv–2405.

Xuhan Huang, Qingning Shen, Yan Hu, Anningzhe Gao, and Benyou Wang. 2024b. Mamo: a mathematical modeling benchmark with solvers. Preprint, arXiv:2405.13144.

Caigao Jiang, Xiang Shu, Hong Qian, Xingyu Lu, Jun Zhou, Aimin Zhou, and Yang Yu. 2025. Llmopt: Learning to define and solve general optimization problems from scratch. In Proceedings of the Thirteenth International Conference on Learning Representations (ICLR), Singapore, Singapore.

Sweta Karlekar, Carolina Zheng, Magnus Saebo, Nicolas Beltran-Velez, Shuyang Yu, John Bowlan, Michal Kucer, and David Blei. 2026. Duelevolve: Reward-free test-time scaling via llm selfpreferences. Preprint, arXiv:2602.21585.

Yujin Kim, Euiin Yi, Minu Kim, Se-Young Yun, and Taehyeon Kim. 2025. Guiding reasoning in small language models with llm assistance. Preprint, arXiv:2504.09923.

Kuang-Huei Lee, Ian Fischer, Yueh-Hua Wu, Dave Marwood, Shumeet Baluja, Dale Schuurmans, and Xinyun Chen. 2025. Evolving deeper llm thinking. Preprint, arXiv:2501.09891.

Ziheng Li, Zexu Sun, Jinman Zhao, Erxue Min, Yongcheng Zeng, Hui Wu, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, Xu Chen, and Zhi-Hong Deng. 2025. Staying in the sweet spot: Responsive reasoning evolution via capability-adaptive hint scaffolding. Preprint, arXiv:2509.06923.

Wei Liu, Siya Qi, Yali Du, and Yulan He. 2026a. Position: Self-play only evolves when self-synthetic pipeline ensures learnable information gain. In Fortythird International Conference on Machine Learning Position Paper Track.

Xianyang Liu, Shangding Gu, and Dawn Song. 2026b. Agenticpay: A multi-agent llm negotiation system for buyer-seller transactions. arXiv preprint arXiv:2602.06008.

Yutong Liu, Lida Shi, Rui Song, and Hao Xu. 2025. A dual-mind framework for strategic and expressive negotiation agent. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23840– 23860, Vienna, Austria. Association for Computational Linguistics.

AI@Meta Llama Team. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Jinghao Luo, Yuchen Tian, Chuxue Cao, Ziyang Luo, Hongzhan Lin, Kaixin Li, Chuyi Kong, Ruichao Yang, and Jing Ma. 2026. From storage to experience: A survey on the evolution of llm agent memory mechanisms. Preprint, arXiv:2605.06716.

Saul McLeod. 2024. Vygotsky’s zone of proximal development.

Albert Newen and Regina E Fabry. 2025. A pattern theory of scaffolding. Review of Philosophy and Psychology, 16(1):65–90.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, and 262 others. 2024. Gpt-4 technical report. Preprint, arXiv:2303.08774.

Avinash Patil and Aryan Jadon. 2025. Advancing reasoning in large language models: Promising methods and approaches. Preprint, arXiv:2502.03671.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. 2024. Direct preference optimization: Your language model is secretly a reward model. Preprint, arXiv:2305.18290.

Rindranirina Ramamonjison, Timothy Yu, Raymond Li, Haley Li, Giuseppe Carenini, Bissan Ghaddar, Shiqi He, Mahdi Mostajabdaveh, Amin Banitalebi-Dehkordi, Zirui Zhou, and Yong Zhang. 2022. Nl4opt competition: Formulating optimization problems based on their natural language descriptions. In Proceedings of the NeurIPS 2022 Competitions Track, volume 220 of Proceedings ofMachine Learning Research, pages 189–203. PMLR.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. Preprint, arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Anthony Sicilia, Hyunwoo Kim, Khyathi Chandu, Malihe Alikhani, and Jack Hessel. 2024. Deal, or no deal (or who knows)? forecasting uncertainty in conversations using large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 11700–11726, Bangkok, Thailand. Association for Computational Linguistics.

Alkis Sygkounas, Rishi Hazra, Andreas Persson, Pedro Zuidberg Dos Martires, and Amy Loutfi. 2026. Covolve: Adversarial co-evolution of large-languagemodel-generated policies and environments via twoplayer zero-sum game. Preprint, arXiv:2603.28386.

Christine van Nooijen, Björn de Koning, Wichor Bramer, Anna Isahakyan, Maryam Asoodar, Ellen Kok, Jeroen Merrienboer, and Fred Paas. 2024. A cognitive load theory approach to understanding expert scaffolding of visual problem-solving tasks: A scoping review. Educational Psychology Review, 36.

Peter Wallis. 2026. LLMs and the ZPD. Preprint, arXiv:2605.12016.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Ed H. Chi, Quoc Le, and Denny Zhou. 2022. Chain of thought prompting elicits reasoning in large language models. CoRR, abs/2201.11903.

Tianxin Wei, Noveen Sachdeva, Benjamin Coleman, Zhankui He, Yuanchen Bei, Xuying Ning, Mengting Ai, Yunzhe Li, Jingrui He, Ed H. Chi, Chi Wang, Shuo Chen, Fernando Pereira, Wang-Cheng Kang, and Derek Zhiyuan Cheng. 2026. Evo-memory: Benchmarking llm agent test-time learning with selfevolving memory. Preprint, arXiv:2511.20857.

Kate Wilson and Linda Devereux. 2014. Scaffolding theory: High challenge, high support in academic language and learning (all) contexts. Journal of Academic Language and learning, 8(3):A91–A100.

Junde Wu, Jiayuan Zhu, Yuyuan Liu, Min Xu, and Yueming Jin. 2025a. Agentic reasoning: A streamlined framework for enhancing LLM reasoning with agentic tools. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 28489– 28503, Vienna, Austria. Association for Computational Linguistics.

Shirley Wu, Michel Galley, Baolin Peng, Hao Cheng, Gavin Li, Yao Dou, Weixin Cai, James Zou, Jure Leskovec, and Jianfeng Gao. 2025b. CollabLLM: From passive responders to active collaborators. In Forty-second International Conference on Machine Learning.

Zhishang Xiang, Chengyi Yang, Zerui Chen, Zhimin Wei, Yunbo Tang, Zongpei Teng, Zexi Peng, Zongxia Li, Chengsong Huang, Yicheng He, and 1 others. 2026. A systematic survey of self-evolving agents: From model-centric to environment-driven co-evolution.

Ziyang Xiao, Dongxiang Zhang, Yangjun Wu, Lilin Xu, Yuan Jessica Wang, Xiongwei Han, Xiaojin Fu, Tao Zhong, Jia Zeng, Mingli Song, and Gang Chen. 2024. Chain-of-experts: When LLMs meet complex operations research problems. In The Twelfth International Conference on Learning Representations.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 23 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Chengyi Yang, Zhishang Xiang, Yunbo Tang, Zongpei Teng, Chengsong Huang, Fei Long, Yuhan Liu, and Jinsong Su. 2026. TTCS: Test-time curriculum synthesis for self-evolving. arXiv preprint arXiv:2601.22628.

Xiao Yu, Maximillian Chen, and Zhou Yu. 2023. Prompt-based Monte-Carlo tree search for goaloriented dialogue policy planning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7101–7125, Singapore. Association for Computational Linguistics.

Yajie Yu and Yue Feng. 2025. Policyevol-agent: Evolving policy via environment perception and self-awareness with theory of mind. Preprint, arXiv:2504.15313.

Weiqi Zeng, Bo Wang, Dongming Zhao, Zongfeng Qu, Ruifang He, Yuexian Hou, and Qinghua Hu. 2025. Dynamic personality in LLM agents: A framework for evolutionary modeling and behavioral analysis in the prisoner’s dilemma. In Findings of the Association for Computational Linguistics: ACL 2025, pages 23087–23100, Vienna, Austria. Association for Computational Linguistics.

Xiaoying Zhang, Yipeng Zhang, Hao Sun, Kaituo Feng, Chaochao Lu, Chao Yang, and Helen Meng. 2025. Critique-grpo: Advancing llm reasoning with natural language and numerical feedback. Preprint, arXiv:2506.03106.

Andrew Zhao, Yiran Wu, Yang Yue, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. 2025. Absolute zero: Reinforced self-play reasoning with zero data. Preprint, arXiv:2505.03335.

Jihao Zhao, Ding Chen, Zhaoxin Fan, Kerun Xu, Mengting Hu, Bo Tang, Feiyu Xiong, and Zhiyu Li. 2026. Inside out: Evolving user-centric core memory trees for long-term personalized dialogue systems. Preprint, arXiv:2601.05171.

Chenyu Zhou, Tianyi Xu, Jianghao Lin, and Dongdong Ge. 2026. StepORLM: A self-evolving framework with generative process supervision for operations research language models. In The Fourteenth International Conference on Learning Representations.

## A Other Experiment Details

## A.1 Simulation and Reward Formulation for Negotiation

Unlike static mathematical reasoning, negotiation is a dynamic, multi-turn adversarial game. To properly evaluate and optimize the Learner in the CraigslistBargain dataset, we deploy GPT-4o-mini as a rule-abiding simulator acting as the Seller, while our Learner takes on the role of the Buyer.

To accurately estimate the value of each intermediate response generated by the Learner during the exploration phase, we adopt a rollout-based reward estimation mechanism, conceptually similar to CollabLLM(Wu et al., 2025b). Specifically, for each candidate reply generated by the Learner, we do not rely on a naive step-level heuristic. Instead, we simulate the remainder of the conversation to its conclusion by sampling multiple future trajectories (rollouts). The reward $R _ { a c c }$ assigned to the Learner’s current reply is proportionally derived from the number of successful deals reached across these multiple simulations. If a reply consistently leads to a successful agreement within the target budget across the simulated futures, it receives a high reward; conversely, replies leading to negotiation breakdowns (walk-aways) are penalized. This Monte Carlo-style exploration ensures that the Learner optimizes for long-term strategic success rather than myopic, single-turn appeasement.

## A.2 Datasets and Training Configurations

Dataset Statistics. For the Operations Research domain, we utilize the training splits of the Mano Complex and Complex OR datasets as the seed contexts for the Scaffolder. The Scaffolder generates constraints based on roughly 1,500 base linear and non-linear programming scenarios. For the Social Negotiation domain (CraigslistBargain), we utilize approximately 3,000 dialogue scenarios encompassing various item categories (e.g., housing, vehicles, electronics) to serve as the context conditions.

Hyperparameters. All models are fine-tuned using Low-Rank Adaptation (LoRA) to ensure memory efficiency. The LoRA rank is set to 16, with an alpha of 32 and a dropout rate of 0.05. We apply LoRA to all linear layers (q\_proj, k\_proj, v\_proj, o\_proj, gate\_proj, up\_proj, down\_proj). The training spans a total of 6 self-play epochs, which we empirically found sufficient to reach the capability plateau for both domains. We use the AdamW optimizer with a peak learning rate of 2e-5, accompanied by a cosine learning rate scheduler and a 3% warmup ratio. During the Golden Experience Replay phase at the end of each epoch, the unified model is trained with a batch size of 16 for standard supervised fine-tuning (SFT).

## A.3 Detailed Description of Baselines

## Negotiation Baselines:

• Vanilla & Pro-CoT: Standard zero-shot prompting and Proactive Chain-of-Thought prompting, which injects planning steps before generating the dialogue response.

• GDPZero: A prompt-based framework utilizing Monte-Carlo Tree Search (MCTS) for goal-oriented dialogue policy planning without updating model weights.

• DPDP & DMNA: State-of-the-art trainingbased frameworks. DPDP employs a dualprocess planning mechanism for dialogue, while DMNA (Dual-Mind Negotiation Agent) fine-tunes models to balance strategic bottom lines and expressive conversational tactics.

## Operations Research (OR) Baselines:

• Vanilla & Vanilla+CoT: Standard direct generation and step-by-step reasoning prompts for translating math word problems into optimization code.

• ORLM: A customized, training-based framework specifically designed for automated optimization modeling.

• LLMOPT: A recent method that learns to define and solve general optimization problems from scratch using structured formulations.

• StepORLM: The previous state-of-the-art that utilizes a self-evolving framework guided by generative process supervision, tailored for operations research language models.

## B Qualitative Examples

In Section 4.4, we quantitatively demonstrated that the average token length of constraints generated by the Scaffolder increases significantly over the course of the 6-epoch self-play.

In the Social Negotiation domain, the Scaffolder transitions from basic numerical boundaries (Epoch 1) to multi-turn adversarial tactics involving bundled conditions and persona shifts (Epoch 6). Similarly, in the Operations Research domain, it shifts from elementary linear inequalities to complex Mixed-Integer Linear Programming (MILP) logic—such as Big-M formulations for fixed charges—that inherently requires advanced mathematical modeling capabilities from the Learner. This qualitative evidence corroborates that STRETCH acts as an intelligent curriculum designer, fostering authentic strategy emergence.

## C Generalization to Simpler OR Datasets

To verify that our STRETCH framework does not suffer from catastrophic forgetting on fundamental problems while escalating cognitive difficulty, we

further evaluate it on two relatively simpler Operations Research benchmarks: NL4Opt and Mano Easy.

## Dataset Descriptions:

• NL4Opt: (Ramamonjison et al., 2022) A widely used benchmark focused on extracting and formulating linear programming (LP) problems from natural language descriptions. It primarily consists of textbook-level, explicit LP problems with straightforward continuous variables and linear constraints, serving as a fundamental test for optimization modeling.

• Mano Easy: (Huang et al., 2024b) A subset of the MAMO benchmark containing elementary mathematical modeling scenarios. Unlike the Mano Complex dataset used in our main experiments, the Easy subset focuses on basic linear relationships without requiring complex conditional logic or Mixed-Integer Linear Programming (MILP) formulations.

## Results and Analysis:

As shown in Table 5, STRETCH achieves highly comparable, and in most cases slightly superior, performance on these elementary datasets relative to strong domain-specific baselines (e.g., StepORLM and LLMOPT).

Crucially, these results address a common vulnerability in self-improvement and curriculum learning paradigms: catastrophic forgetting. While the Scaffolder dynamically pushes the Learner into the upper bounds of its Stretch Zone (escalating to complex, adversarial, and MILP formulations in later epochs as discussed in Section 4.4), the Learner completely retains its foundational reasoning skills.

This retention is largely attributed to our epochlevel Golden Experience Replay mechanism. By consistently consolidating high-quality trajectories into the shared parameter space, STRETCH ensures that the model maintains near-ceiling performance on fundamental tasks while simultaneously expanding its upper capability limits on hardcore constraints.

## D Related Prompts

In this section, we detail the exact end-to-end prompt templates used to instantiate the Scaffolder and the Learner within our STRETCH framework across the two evaluated domains. The dual roles are activated within the singular parameter space purely via these instructions.

## D.1 Prompts for Social Negotiation

For the CraigslistBargain dataset, the Scaffolder acts as a strategic director imposing specific bargaining targets, while the Learner acts as the actual buyer executing the dialogue.

## Scaffolder Prompt (Constraint Generation):

You are an expert negotiation strategist. Your goal is to generate a specific, challenging, but achievable negotiation constraint for a buyer. It must not be trivially easy (e.g., just asking for a 1% discount) nor impossibly hard (e.g., asking for a 90% discount). Based on the following item description and base price: {Base\_Item\_Description\_and\_Price}, generate a specific target and a behavioral constraint for the buyer. Output your constraint in a single concise paragraph.

## Learner Prompt (Task Solving):

You are a buyer negotiating the price of an item with a seller. You must strictly adhere to your internal constraints and goals while maintaining a realistic conversational tone. The item you are negotiating for is: {Base\_Item\_Description\_and\_Price}. Your specific constraint is: {Generated\_Constraint\_from\_Scaffolder}. Here is the dialogue history so far: {Dialogue\_History}. Generate your next dialogue response to the seller to maximize your strategic advantage and achieve your constraint.

## D.2 Prompts for Operations Research

For the mathematical modeling tasks (Mano Complex and Complex OR), the Scaffolder acts as an environment designer adding realistic resource limits, while the Learner acts as the mathematical solver.

## Scaffolder Prompt (Constraint Generation):

You are an expert Operations Research (OR) professor. Your task is to increase the complexity of a basic linear programming problem by adding a realistic, mathematically sound constraint. This could include adding a piecewise cost function, a budget limit, or a conditional integer constraint (e.g., Big-M formulation). The added constraint must be logically consistent with the base problem. Given the following base math problem: {Base\_Math\_Problem}, generate ONE new constraint paragraph to be appended to the problem description. Do not solve the problem.

## Learner Prompt (Task Solving):

You are an expert Operations Research solver. Your task is to read a complex mathematical word problem, formulate it accurately, and write executable Python code using the Gurobi or PuLP library to solve it. The base problem is: {Base\_Math\_Problem}. Please also consider the following additional constraints: {Generated\_Constraint\_from\_Scaffolder}. Provide a step-by-step mathematical reasoning (Chain-of-Thought) followed by the complete, executable Python code block.

<table><tr><td>Epoch</td><td>Social Negotiation (Dynamic Soft Constraints)</td><td>Operations Research (Static Hard Constraints)</td></tr><tr><td>Epoch 1 (Comfort)</td><td>Generated Target: “Your target price is $150. Do not accept anything above $180. Be polite.&quot; Complexity: Single numerical constraint with a basic persona. Easily solved by the base Learner.</td><td>Generated Constraint: “Add a production capacity limit: The total number of units produced cannot ex- ceed 500.&quot; Complexity: Basic linear inequality  $( \sum x _ { i } \ \leq \ 5 0 0 ) .$  Trivially mapped to executable code.</td></tr><tr><td>Epoch 3 (Stretch)</td><td>Generated Target: “Your target price is $130. You must persuade the seller to include free delivery. If they refuse, express disappointment and threaten to walk away.&quot; Complexity: Multi-objective constraint requiring con- ditional dialogue acts and emotional shifts.</td><td>Generated Constraint: “Add a piecewise cost con- straint: The first 200 units cost $5 each, and any addi- tional units cost $7 each due to overtime labor.&quot; Complexity: Non-smooth piecewise objective requir- ing auxiliary variables and sequential logic parsing.</td></tr><tr><td>Epoch 6 Boundary</td><td>Generated Target: “Your absolute limit is $110. Em- ploy a &#x27;Good Cop/Bad Cop&#x27; persona. First, offer $100 and heavily critique the item&#x27;s condition. If rejected, reluctantly offer $115 but strictly demand the original receipt and an extended 30-day warranty.&quot; Complexity: Highly adversarial multi-turn strategic planning with strict preconditions and bundled trade- offs.</td><td>Generated Constraint: “Formulate a conditional inte- ger constraint: If warehouse A is used (binary variable  $\overset { \cdot } { y } _ { A } = 1 )$  , then at least 100 units must be stored there  $( x _ { A } \ge 1 0 0 )$  , and the total logistics cost must incorpo- rate a fixed setup fee of $1000.&quot; Complexity: Hardcore Mixed-Integer Linear Program- ming (MILP) requiring Big-M formulations.</td></tr></table>

Table 4: Qualitative evolution of the constraints generated by the STRETCH Scaffolder. As the training progresses, the Scaffolder autonomously learns to escalate the cognitive complexity, perfectly matching the Learner’s growing capacity without relying on human-curated curriculum.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">NL4Opt</td><td colspan="2">Mano Easy</td></tr><tr><td>ER (%)</td><td>SA (%)</td><td>ER (%)</td><td>SA (%)</td></tr><tr><td rowspan="6">Qwen2.5-7B-Instruct</td><td>Vanilla</td><td>60.5</td><td>55.2</td><td>65.1</td><td>60.3</td></tr><tr><td>Vanilla + CoT</td><td>75.3</td><td>70.1</td><td>78.4</td><td>72.5</td></tr><tr><td>ORLM</td><td>88.5</td><td>85.1</td><td>89.2</td><td>86.4</td></tr><tr><td>LLMOPT</td><td>90.1</td><td>87.5</td><td>91.5</td><td>88.0</td></tr><tr><td>StepORLM</td><td>92.4</td><td>90.1</td><td>93.1</td><td>91.2</td></tr><tr><td>STRETCH (Ours)</td><td>91.8</td><td>90.6</td><td>93.6</td><td>91.5</td></tr><tr><td rowspan="6">Llama-3.1-8B-Instruct</td><td>Vanilla</td><td>55.4</td><td>49.8</td><td>60.2</td><td>54.1</td></tr><tr><td>Vanilla + CoT</td><td>70.2</td><td>64.5</td><td>72.8</td><td>67.9</td></tr><tr><td>ORLM</td><td>84.3</td><td>80.6</td><td>85.4</td><td>81.2</td></tr><tr><td>LLMOPT</td><td>86.8</td><td>83.2</td><td>88.1</td><td>87.6</td></tr><tr><td>StepORLM</td><td>89.5</td><td>87.2</td><td>89.1</td><td>86.3</td></tr><tr><td>STRETCH (Ours)</td><td>90.1</td><td>86.4</td><td>90.3</td><td>88.4</td></tr></table>

Table 5: Performance comparison on simpler OR datasets (NL4Opt and Mano Easy). STRETCH maintains near-ceiling performance, demonstrating that escalating challenge complexity during self-play does not lead to catastrophic forgetting of foundational reasoning skills.

## D.3 Prompts for LLM-as-a-Judge (Reward Evaluation)

To compute the empirical accuracy $( R _ { a c c } )$ during the micro-interleaved GRPO updates, we utilize an external judge (GPT-4o-mini). Below is the end-to-end evaluation prompt for the Negotiation domain.

## Judge Prompt (Negotiation Evaluation):

You are an impartial negotiation referee. Read the following completed negotiation dialogue and the buyer’s secret constraint. Dialogue: {Completed\_Dialogue\_Trajectory}. Buyer’s Constraint: {Generated\_Constraint\_from\_Scaffolder}. Did the buyer successfully reach a deal with the seller that STRICTLY satisfies their secret constraint? Output only $" 1 "$ for Yes, or $" 0 "$ for No.