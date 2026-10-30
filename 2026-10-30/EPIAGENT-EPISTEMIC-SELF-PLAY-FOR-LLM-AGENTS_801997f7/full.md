# EPIAGENT: EPISTEMIC SELF-PLAY FOR LLM AGENTS

Xuanfa Jin<sup>1,2</sup> Xinyu Cui<sup>1,2</sup> Zhijian Ma<sup>1,2</sup> Xue Yan<sup>1,2</sup> Jian Zhao<sup>3</sup> Haifeng Zhang<sup>1,2</sup> Jun Wang<sup>∗,4</sup>

<sup>1</sup>Institute of Automation, Chinese Academy of Sciences <sup>2</sup>School of Artificial Intelligence, UCAS <sup>3</sup>Zhongguancun Academy <sup>4</sup>AI Centre, Department of Computer Science, UCL

## ABSTRACT

Epistemic game theory explains an agent’s strategic behavior via the beliefs it holds about others and how these beliefs are updated as the interactions unfold. Modern LLM-based agents, however, often lack an explicit mechanism for learning and updating such beliefs to guide decisions under incomplete information. To bridge this gap, we propose EpiAgent, a framework that equips LLM-based agents with an explicit, learnable epistemic layer, which maintains beliefs over others’ private information and combines these beliefs with the agent’s available information into an epistemic state for action selection. Concretely, the agent’s policy is decomposed into two coupled stages: Epistemic Update, which updates these beliefs from interaction histories, and Action Planning, which selects actions conditioned on the updated epistemic state. By formulating their dependency as a bi-level optimization problem, we introduce Epistemic Self-Play, an algorithm that alternately optimizes the two stages through self-play, utilizing an intrinsic prediction reward to evaluate how well inferred beliefs predict subsequent observations. Empirical results across both mixed and competitive scenarios show that EpiAgent consistently outperforms prompting-based and RL-based baselines. Further analyses reveal the benefits of dedicated epistemic training, improvements in belief inference accuracy, and the contribution of learned beliefs to action planning. These results demonstrate the value of an explicit, learnable epistemic layer for effective multi-agent interactions.

## 1 Introduction

Large Language Models (LLMs) have demonstrated impressive capabilities in reasoning and planning [Achiam et al., 2023, Guo et al., 2025], sparking a growing interest in deploying LLM-based agents with Multi-Agent Systems (MAS) to tackle complex tasks [Qian et al., 2024, Liang et al., 2024, Schmidgall et al., 2025]. Unlike single-agent tasks where LLMs have excelled [Lu et al., 2025a, Jin et al., 2025], multi-agent scenarios introduce significant non-stationarity and incomplete information. In these settings, environment dynamics and outcomes are determined not only by the ego agent’s own decisions but also by the policies of other agents, who often possess private information that is unobservable to others [Littman, 1994, Albrecht et al., 2024]. Since such private information can influence other agents’ behavior, inferring it from interaction histories can help an agent anticipate their future behaviors and adapt its own decisions accordingly.

Inspired by epistemic game theory (EGT) [Aumann, 1999, Dekel and Siniscalchi, 2015], we view an agent’s strategic behavior as being explained via the beliefs that agents hold about others, and the way these beliefs are updated through the game. From this perspective, an agent’s decision can be naturally understood in terms of two coupled components: an interim belief over the private information of others, updated from the observed interaction history, and a beliefconditioned action that is guided by this belief. The agent’s own private information, observed history and its current belief jointly constitute its epistemic state, which guides the agent’s strategic reasoning and action selection [Perea, 2012]. Many LLM-based agents, however, lack such an explicit epistemic layer in their decision-making pipeline. They usually map observations directly to actions [Nakano et al., 2021, Wang et al., 2023], or rely on prompting to elicit free-form reasoning [Wei et al., 2022, Zhao et al., 2024] that is treated as a byproduct of generation rather than as a structured epistemic process. Even when explicit opponent modeling is introduced, it is typically decoupled from the agent’s policy learning and does not co-evolve with it [Cross et al., 2024, Kim et al., 2025]. This motivates developing an explicit, learnable epistemic layer whose learning is coordinated with that of the action policy through interaction.

To this end, we propose EpiAgent, a framework that equips LLM-based agents with an explicit epistemic layer for decision-making. We first formulate multi-agent interaction scenarios with fixed private information within a partially observable stochastic game (POSG) [Hansen et al., 2004] framework, which formally specifies each agent’s private information and the interaction dynamics of multiple agents. Within this framework, an agent’s policy is unfolded into two coupled stages: Epistemic Update, which updates the agent’s belief from its observed interaction histories, and Action Planning, which selects actions based on the updated epistemic state. However, training such an agent is non-trivial due to the mutual dependence between the two stages: epistemic update learns from interaction trajectories generated by the action policy, while action planning learns to make decisions based on the evolving beliefs. We cast this dependency as a bi-level optimization problem and introduce Epistemic Self-Play, an algorithm that alternately optimizes the two stages through self-play. Given that the ground-truth labels for other agents’ private information are often inaccessible in real-world scenarios, we further introduce an intrinsic prediction reward to train epistemic updates with subsequent observations.

To validate the performance of our method, we adopt two distinct yet challenging multi-agent environments: Bargaining and Kuhn Poker, both of which exemplify the complexities of incomplete information. The Bargaining scenario represents a mixed negotiation game where agents must trade off self-interest against the risk of breakdown. Meanwhile, the Kuhn Poker scenario represents a competitive game involving stochasticity and deception. Empirical results demonstrate that EpiAgent greatly outperforms both prompting-based baselines and an RL-trained baseline without an explicit epistemic layer. Furthermore, ablation results demonstrate that dedicated training of the epistemic update stage provides additional performance benefits, while evaluations of belief inference show improved accuracy over training. Belief interventions further show that the learned beliefs contribute to effective action planning. These results suggest that the performance gains of our method are not solely attributable to action planning optimization, but also benefit from improved epistemic reasoning. Our main contributions are summarized as follows:

• We propose EpiAgent, an LLM-based agent framework that introduces an explicit epistemic layer into the agent’s decision-making pipeline by decomposing the agent’s policy into an Epistemic Update stage and an Action Planning stage.

• We introduce Epistemic Self-Play, an algorithm designed to promote the co-evolution of epistemic update and action planning through alternating self-play, equipped with an intrinsic prediction reward that trains the agent’s epistemic inference without ground-truth labels.

• We provide comprehensive empirical results showing that EpiAgent outperforms both prompting-based and RL-based baselines, and that its learned beliefs effectively improve action planning, demonstrating the value of an explicit, learnable epistemic layer for multi-agent interactions.

## 2 Related Work

LLM-based Agent for Decision-Making. Recent advancements in Large Language Models (LLMs) have significantly expanded their utility from natural language processing to autonomous decision-making tasks [Zhu et al., 2023, He et al., 2024, Zhang et al., 2025a]. Foundational techniques such as Chain-of-Thought (CoT) [Wei et al., 2022] enable LLMs to decompose complex tasks into simpler ones, while frameworks like ReAct [Yao et al., 2022] and Reflexion [Shinn et al., 2023] further enhance agents’ decision-making capabilities by interleaving reasoning with action execution. Beyond inference-time prompting, recent works have also explored Reinforcement Learning (RL) to fundamentally improve agent policies [Zhang et al., 2025b], particularly for tasks with verifiable rewards, such as mathematics [Shao et al., 2024, Zuo et al., 2025] and coding [Wei et al., 2025, Zeng et al., 2025]. However, extending these successes to multi-turn scenarios is non-trivial, as the immediate ground-truth feedback in those scenarios is often sparse or absent. To address this, SWEET-RL [Zhou et al., 2025] adopts a turn-level critic to improve credit assignments, thereby accelerating the learning process. And RAGEN [Wang et al., 2025] provides a unified framework for multi-turn, trajectory-level LLM-based agent training. While these methods can be directly applied to multi-agent scenarios, they do not explicitly model other agents’ private information. By equipping the LLM-based agent with an explicit epistemic layer, EpiAgent can infer such information from interaction histories to guide its subsequent actions.

LLM-based Multi-Agent Systems. Applying LLMs in multi-agent systems (MAS) is becoming a vibrant research area, and the applications can be broadly categorized into world simulation and problem solving [Li et al., 2024]. In world simulation, Generative Agents [Park et al., 2023] construct a sandbox environment to observe human-like behaviors emerge from LLM-based agents. Furthermore, strategy games such as Diplomacy [Guan et al., 2024, Peng et al., 2025], Werewolf [Xu et al., 2024, Jin et al., 2024], and Avalon [Shi et al., 2023, Lan et al., 2024] are often adopted to test LLMs’ strategic decision-making abilities. When it comes to problem solving, researchers mainly focus on leveraging role-playing and collaboration to tackle various tasks. CAMEL [Li et al., 2023a] and AutoGen [Wu et al., 2024] design cooperative agents for general reasoning and question answering. SWE-Agent [Yang et al., 2024] and ChatDev [Qian et al., 2024] incorporate standard operating procedures (SOPs) into agent interactions, simulating human workflows where agents with specialized roles collaborate to develop software. Conversely, Multi-Agent Debate [Liang et al., 2024, Liu et al., 2025a, Jin et al., 2026] takes a competitive way by encouraging agents to critique each other’s views to reach a consensus, thereby improving the quality of the final solution. These works mostly focus on cooperative settings or reason about other agents implicitly, without explicitly modeling their private information. In contrast, we integrate and train an epistemic update stage to explicitly infer others’ private information, enabling agents adaptive decision-making across various scenarios.

![](images/f41a22141930edfc0a7664b50468839a60c436ed6c5e3d5f69cbc0ed244140fd.jpg)  
Figure 1: The EpiAgent framework trained with Epistemic Self-Play. Agent 0 and Agent 1 share a single LLM, performing self-play in the multi-agent environment and collecting interaction histories (middle). The LLM is then optimized alternately between the epistemic update (left) and the action planning (right) stage, by reconstructing the histories into corresponding trajectories for RL training.

Belief Modeling and Theory of Mind in LLM Agents. A growing body of work has investigated whether and how LLMs can reason about others’ mental states. Many works study LLMs’ Theory of Mind (ToM) [Frith and Frith, 2005] capabilities through false-belief tasks and social reasoning benchmarks [Gandhi et al., 2023, Wu et al., 2023, Shinoda et al., 2025], or attempt to enhance such capabilities through prompt engineering [Li et al., 2023b], workflow design [Zhang et al., 2025c, Kim et al., 2025], or fine-tuning [Lu et al., 2025b, Hwang et al., 2025]. However, these methods mostly treat ToM as a standalone capability to be evaluated or improved, rather than as an active component in the agent’s decision-making pipeline. Epistemic game theory provides a perspective on this connection, explaining strategic behavior through the beliefs agents hold about others and how these beliefs are updated as the interaction unfolds [Aumann, 1999, Dekel and Siniscalchi, 2015, Perea, 2012]. Some works have begun to use such reasoning for decision-making: Hypothetical Minds [Cross et al., 2024] prompts LLMs to maintain hypotheses about other agents’ strategies, which further guide action generation, and Zhang et al. [2024] conduct an empirical study of the Mutual ToM process in human-AI collaboration. However, these methods mainly rely on the inference-time capabilities of a fixed LLM, leaving belief inference decoupled from the agent’s policy learning. In multi-agent RL, Policy Belief Learning [Tian et al., 2020] alternately trains a belief module and a belief-conditioned policy, but supervises beliefs with ground-truth private information. In contrast, EpiAgent trains epistemic update and action planning within the same LLM through self-play, learning to infer others’ private information from interaction feedback without ground-truth labels and to act on the resulting beliefs.

## 3 Preliminaries

Problem Formulation. We consider finite-horizon, turn-based interactions among agents with private information. Following the static private-information setting of Tian et al. [2020], we distinguish each agent’s fixed private information from the dynamic interaction state. We describe the environment by $\mathcal { E } \stackrel { < } { = } \langle \mathcal { N } , \mathcal { X } , \bar { \mathcal { Z } } , \mathcal { A } , \mathcal { O } , \rho , \hat { P _ { , } } \{ r _ { i } \} _ { i \in \mathcal { N } } \rangle$ , where $\mathcal { N } = \{ 1 , \ldots , n \}$ is the agent set and X is the space of dynamic states. The joint private-information, action, and observation spaces are denoted as $\mathcal { Z } = \times _ { i } \mathcal { Z } _ { i } , \mathcal { A } \overset { - } { = } \times _ { i } \mathcal { A } _ { i }$ , and $\mathcal { O } = \times _ { i } \mathcal { O } _ { i }$ , respectively.

At the start of an episode, $( x ^ { 1 } , \mathbf { z } , \mathbf { o } ^ { 1 } ) \sim \rho$ specifies the initial state, each agent’s private information, and their observations. Each $z _ { i } \in { \mathcal { Z } } _ { i }$ is available to agent i but not directly observable by other agents, and remains fixed throughout the episode. At interaction step t, the joint action $\mathbf { a } ^ { t }$ induces $( x ^ { t + 1 } , \mathbf { o } ^ { t + 1 } ) \sim P ( \cdot \mid x ^ { t } , \mathbf { z } , \mathbf { a } ^ { t } )$ , where $P$ is the transition and observation kernel, and agent i receives reward $r _ { i } ^ { t } = r _ { i } ( \boldsymbol { x } ^ { t } , \mathbf { z } , \mathbf { a } ^ { t } )$ . The dynamic state $\mathrm { i } \mathit { x } ^ { t }$ may itself be partially observed. Turn-taking is encoded in $x ^ { t }$ , with inactive agents restricted to a null action. This formulation is an instance of a partially observable stochastic game (POSG) [Hansen et al., 2004] with full state $\boldsymbol { s } ^ { t } = ( x ^ { t } , \mathbf { z } )$ , whose static component z is preserved by the transition. Agent i’s interaction history is denoted as $h _ { i } ^ { t } = ( o _ { i } ^ { 1 } , a _ { i } ^ { 1 } , \ldots , a _ { i } ^ { t - 1 } , o _ { i } ^ { t } )$ where $o _ { i } ^ { t }$ is the observation available to that agent at step t, including other agents’ actions when observable. Together, the interaction history $h _ { i } ^ { t }$ and the agent’s own private information $z _ { i }$ constitute the information available to agent i for decision-making. Thus, agent $i \ ' s$ policy can be defined as $\pi _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } )$

Beliefs and Epistemic States. In partially observable decision problems, a belief is a posterior distribution over the underlying state given an agent’s available information [Kaelbling et al., 1998]. In our setting, we focus on the agent’s belief over others’ fixed private information, which is inferred based on its interaction history. Specifically, agent i’s belief at step t is defined as $b _ { i } ^ { t } \in \Delta ( \mathcal { Z } _ { - i } )$ , where $\mathcal { Z } _ { - i } = \times _ { j \neq i } \mathcal { Z } _ { j }$ . Although $z _ { - i }$ remains fixed within an episode, its posterior can be updated as new observations arrive. Together with the agent’s own private information and interaction history, this belief defines agent i’s epistemic state at each step:

$$
e _ { i } ^ { t } = ( z _ { i } , h _ { i } ^ { t } , b _ { i } ^ { t } ) , \quad b _ { i } ^ { t } \in \Delta ( \mathcal { Z } _ { - i } ) ,\tag{1}
$$

where $z _ { i }$ and $h _ { i } ^ { t }$ summarize what agent i knows, and $b _ { i } ^ { t }$ summarizes what it infers about others. While epistemic game theory in its full generality concerns belief hierarchies of arbitrary order [Dekel and Siniscalchi, 2015], here we focus on the first-order layer, each agent’s belief over others’ private information, as the foundational case for LLM-based agents, leaving extensions to higher-order beliefs for future work.

Policy Optimization. In reinforcement learning, an agen $\mathrm { \Delta } \mathrm { t } ^ { \prime } \mathrm { s }$ goal is usually to learn a policy that maximizes its expected cumulative reward. Assuming the agent’s policy π is parameterized by $\phi ,$ the optimization objective is typically formulated as:

$$
\mathcal { I } ( \phi ) = \mathbb { E } _ { \tau \sim \pi _ { \phi } } \left[ \sum _ { t = 1 } ^ { T } r ^ { t } \right] ,\tag{2}
$$

where $\tau \sim \pi _ { \phi }$ denotes a trajectory generated under $\pi _ { \phi } ,$ , and $T$ denotes the episode length. To optimize this objective, policy gradient (PG) methods [Sutton et al., 1999] are widely adopted to update the policy parameters. The gradient of the objective is given by:

$$
\nabla _ { \phi } \mathcal { I } ( \phi ) = \mathbb { E } _ { \tau \sim \pi _ { \phi } } \left[ \sum _ { { t = 1 } } ^ { T } \nabla _ { \phi } \log \pi _ { \phi } ( a ^ { t } \mid h ^ { t } ) \cdot G ^ { t } \right] ,\tag{3}
$$

where $h ^ { t }$ denotes the information available to the policy before taking action $a ^ { t } .$ , and $G ^ { t }$ denotes the return signal at step $t ,$ instantiated as cumulative reward, return-to-go, advantage, or other variants depending on the specific algorithms.

## 4 EpiAgent: Framework and Training

In this section, we present EpiAgent, a framework that equips LLM-based agents with an explicit epistemic layer for decision-making, as shown in Figure 1. We first formalize the decomposition of an agent’s policy into two coupled stages, Epistemic Update and Action Planning, and cast their joint training as a bi-level optimization problem. We then develop optimization objectives for both stages, including an intrinsic prediction reward that enables learning epistemic updates from interaction feedback without ground-truth supervision of other agents’ private information. Finally, we introduce Epistemic Self-Play, an alternating training algorithm designed to promote the co-evolution of epistemic update and action planning through self-play.

## 4.1 Epistemic Decomposition of Agent Policy

To formalize how agents make decisions in multi-agent interactions, we begin by modeling the interaction process from a single agent’s perspective. The generative

![](images/a110fc6eb7d15d0eddcc486854cd62486f7274d48e477842cfdc025410b4e092.jpg)  
Figure 2: The generative model of multi-agent interaction from agent $i \ ' _ { \mathrm { { s } } }$ perspective. Agent i makes decisions based on its own private information $z _ { i }$ and the observed history $h _ { i } ^ { t } ,$ while the transition of $h _ { i } ^ { t }$ is driven by other agents’ actions, which depend on their private information $z _ { - i }$

model in Figure 2 describes the evolution of agent i’s interaction history. At step t, agent i selects action $a _ { i } ^ { t }$ based on its own private information $z _ { i }$ and history $h _ { i } ^ { t } .$ . The subsequent observation $\bar { o _ { i } ^ { t + 1 } }$ extends this history as $h _ { i } ^ { t + 1 } = \bar { ( } h _ { i } ^ { t } , a _ { i } ^ { t } , o _ { i } ^ { t + 1 } )$ . Other agents’ behavior can depend on their fixed private information $z _ { - i }$ and interaction histories, making the observed interaction a source of evidence for inferring $z _ { - i }$

From an epistemic perspective, an agent’s decision-making can be understood in terms of what it knows about the interaction and what it believes about others. These components are represented by its epistemic state $e _ { i } ^ { t } = ( z _ { i } , h _ { i } ^ { t } , b _ { i } ^ { t } )$ Building on this perspective, we unfold agent i’s policy $\pi _ { i }$ into two coupled stages:

• Epistemic Update: Given its interaction history $h _ { i } ^ { t }$ and own private information $z _ { i } ,$ , agent i first maintains a belief model $b _ { i }$ to update its current belief $b _ { i } ^ { t } = b _ { i } ( \cdot \mid h _ { i } ^ { t ^ { \prime } } , z _ { i } )$ . As new observations become available, this model revises the agent’s belief about the fixed private information $z _ { - i }$

• Action Planning: Given a hypothesis $\hat { z } _ { - i } ^ { t }$ about other agents’ private information, agent i then employs a planning model $\mu _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } )$ to generate an action $a _ { i } ^ { t } .$ . The planning model combines this hypothesis with the interaction history and the agent’s own private information.

Together, these two stages define a decomposition of the agent’s overall policy: at each step, the action distribution is obtained by marginalizing the planning model over its updated belief:

$$
\pi _ { i } ( a _ { i } ^ { t } \mid h _ { i } ^ { t } , z _ { i } ) = \mathbb { E } _ { \hat { z } _ { - i } ^ { t } \sim b _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } ) } \left[ \mu _ { i } ( a _ { i } ^ { t } \mid h _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } ) \right] .\tag{4}
$$

This policy can be realized by first sampling $\hat { z } _ { - i } ^ { t } \sim b _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } )$ and then generating $a _ { i } ^ { t } \sim \mu _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } )$ . In this way, inference about other agents becomes an explicit intermediate step in the decision process. This sequential generation process naturally aligns with the autoregressive structure of LLM-based agents.

The decomposition gives rise to two interdependent learning objectives: action planning learns to make decisions under the inferred beliefs, while epistemic update draws on evidence from the interactions generated by the planning model. We formalize their joint training as a bi-level optimization problem:

$$
\begin{array} { r l } { \underset { \mu _ { i } } { \operatorname* { m a x } } } & { \mathcal { T } _ { \mathrm { p l a n n i n g } } \big ( \mu _ { i } , b _ { i } ^ { \star } ( \mu _ { i } ) \big ) , } \\ { \mathrm { s . t . } } & { b _ { i } ^ { \star } ( \mu _ { i } ) \in \arg \underset { b _ { i } } { \operatorname* { m a x } } \mathcal { T } _ { \mathrm { b e l i e f } } ( b _ { i } , \mu _ { i } ) . } \end{array}\tag{5}
$$

Here, $\mathcal { T } _ { \mathrm { b e l i e f } }$ and $\mathcal { I } _ { \mathrm { p l a n n i n g } }$ are the objectives of epistemic update and action planning, respectively. For a given planning model $\mu _ { i }$ , the inner problem optimizes the belief model using the induced trajectories. The outer problem evaluates the planning model together with its corresponding adapted belief model. We develop these objectives in Section 4.2 and introduce Epistemic Self-Play in Section 4.3 to train the two coupled stages.

## 4.2 Optimization Objectives

Epistemic Update. The goal of the epistemic update stage is to infer other agents’ private information $z _ { - i }$ from the interaction history. However, a primary challenge in realistic multi-agent interaction scenarios is that the private information is usually unrevealed, making direct supervision of belief inference difficult. We address this challenge by learning from how well the inferred private information predicts subsequent observations. Following Bayesian inverse planning [Baker et al., 2009, Zhi-Xuan et al., 2020], we motivate this learning signal through the posterior:

$$
p ( z _ { - i } \mid h _ { i } ^ { t } , z _ { i } ) \propto p ( z _ { - i } \mid z _ { i } ) \cdot \prod _ { k = 1 } ^ { t } p ( o _ { i } ^ { k } \mid h _ { i } ^ { k - 1 } , a _ { i } ^ { k - 1 } , z _ { i } , z _ { - i } ) .\tag{6}
$$

Here, $p ( z _ { - i } \mid z _ { i } )$ is the prior conditioned on the agent’s own private information. We set $h _ { i } ^ { 0 } = a _ { i } ^ { 0 } = \emptyset$ as placeholders before the initial observation, so that the $k = 1$ factor denotes $p ( o _ { i } ^ { 1 } \mid z _ { i } , z _ { - i } )$ . Each likelihood factor measures how well a candidate $z _ { - i }$ explains the observed interactions. This formulation suggests a criterion for evaluating epistemic inference: although other agents’ private information is not directly observed, an inferred hypothesis can be assessed by how well it predicts their observed behavior. Interaction feedback can therefore provide a learning signal for the belief model without requiring ground-truth labels.

Based on this insight, we introduce an intrinsic prediction reward to train the epistemic update stage. Specifically, after sampling $\hat { z } _ { - i } ^ { t } \sim b _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } )$ and taking action $a _ { i } ^ { t } .$ , the agent receives a new observation $o _ { i } ^ { t + 1 }$ , which provides feedback on the sampled hypothesis. We compute the post-hoc reward as the log-likelihood of this observation under that hypothesis:

$$
\begin{array} { r } { r _ { i , \mathrm { p r e d } } ^ { t } \big ( \hat { z } _ { - i } ^ { t } \big ) = \log p \big ( o _ { i } ^ { t + 1 } \mid h _ { i } ^ { t } , a _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } \big ) . } \end{array}\tag{7}
$$

In practice, this likelihood is computed by the agent’s own LLM with a dedicated prediction prompt (Appendix D), normalized by the length of $o _ { i } ^ { t + 1 }$ . Since other agents’ responses are generated by the same LLM during self-play, it naturally serves as an approximate model of their behavior. This reward encourages the belief model to produce hypotheses under which the actual subsequent observations are more likely to occur. Accordingly, the objective for the agent i’s epistemic update stage is to maximize the expected cumulative intrinsic reward:

$$
\mathcal { I } _ { \mathrm { b e l i e f } } ( b _ { i } , \mu _ { i } ) = \mathbb { E } _ { b _ { i } , \mu _ { i } } \left[ \sum _ { { t = 1 } } ^ { T - 1 } r _ { i , \mathrm { p r e d } } ^ { t } ( \hat { z } _ { - i } ^ { t } ) \right] ,\tag{8}
$$

where $T$ denotes the episode length. By optimizing this objective, the belief model is optimized purely by the agent’s own predictive consistency, bypassing the need for ground-truth labels of other agents’ private information.

Action Planning. Given a hypothesis $\hat { z } _ { - i } ^ { t } \sim b _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } )$ sampled during epistemic update, agent i’s planning model $\mu _ { i }$ selects actions conditioned on this hypothesis, its interaction history $h _ { i } ^ { t } .$ , and its own private information $z _ { i }$ . With the latent variables that influence the interaction dynamics explicitly estimated as $\hat { z } _ { - i } ^ { t }$ , the planning model uses these estimates to guide action selection, with the primary objective of maximizing the expected cumulative environment reward. However, the beliefs used for planning are themselves learned and continue to evolve during training. Consequently, the sampled hypothesis $\hat { z } _ { - i } ^ { t }$ may not accurately reflect other agents’ private information. Concentrating too early on a narrow set of actions under such estimates may limit the exploration of alternative policies while the belief model is still adapting. We therefore encourage the agent to maintain diversity in its action selection conditioned on the inferred beliefs, allowing broader exploration of potential policies as its beliefs evolve.

To this end, we adopt a maximum-entropy reinforcement learning framework that encourages exploration over beliefconditioned policies. In addition to maximizing the cumulative reward, this framework augments the reward with the entropy of the conditional action distribution. Accordingly, the optimization objective for agent i’s action planning stage is defined as:

$$
\mathcal { T } _ { \mathrm { p l a n n i n g } } ( \mu _ { i } , b _ { i } ) = \mathbb { E } _ { b _ { i } , \mu _ { i } } \left[ \sum _ { { t = 1 } } ^ { T } \left( r _ { i } ^ { t } + \alpha \mathcal { H } ( \mu _ { i } ^ { t } ) \right) \right] ,\tag{9}
$$

where $\mu _ { i } ^ { t }$ abbreviates the action distribution $\mu _ { i } ( \cdot \mid h _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } )$ , H(·) denotes the entropy, and $\alpha \geq 0$ is a coefficient that controls the trade-off between reward and action diversity.

Algorithm 1 Epistemic Self-Play   
Require: Pretrained LLM $p _ { \phi } ,$ switch frequency L   
1: // 0 for Epistemic Update, 1 for Action Planning   
2: Set initial training stage ψ ← 0   
3: while not done do   
4: for l = 1 to L do   
5: Rollout M self-play episodes using $p _ { \phi }$   
6: for i = 1 to n do   
7: Extract agent i’s trajectories ${ \cal B } _ { i } = \{ \tau _ { i } ^ { ( m ) } \} _ { m = 1 } ^ { M }$ , where $\tau _ { i } ^ { ( m ) } = \left( o _ { i } ^ { ( m ) , t } , \hat { z } _ { - i } ^ { ( m ) , t } , a _ { i } ^ { ( m ) , t } \right) _ { t = 1 } ^ { T }$   
8: Calculate role-based advantages by Eq. (10)   
9: end for   
10: Combine all agents’ trajectories $\textstyle B = \bigcup _ { i = 1 } ^ { n } B _ { i }$   
11: Update $p _ { \phi }$ on B using policy gradient with objective Eq. (8) (ψ = 0) or Eq. (9) (ψ = 1)   
12: end for   
13: Switch training stage $\psi  1 - \psi$   
14: end while

## 4.3 Epistemic Self-Play

Training the two-stage decision process involves a hierarchical dependency: the action planning stage requires stable and accurate beliefs to optimize its policy, while the epistemic update stage requires informative interaction trajectories generated by the action policy to learn different patterns in other agents’ behavior. To coordinate learning under this dependency, we introduce Epistemic Self-Play, an alternating training algorithm designed to promote the co-evolution of epistemic update and action planning. The algorithm alternates between optimizing the two stages using interaction trajectories collected through self-play, allowing each stage to learn from feedback shaped by the evolving other stage.

Meanwhile, self-play provides experience against continually updated interaction partners, with the aim of reducing specialization to a single fixed training opponent.

To realize this framework, we seek a unified agent that can both infer other agents’ private information and use these inferences to guide its own actions. Therefore, we instantiate both the belief model $b _ { i }$ and the planning model $\mu _ { i }$ using a single LLM parameterized by ϕ. Specifically, we adopt $p _ { \phi } \big ( \hat { z } _ { - i } ^ { t } \ \big | \ h _ { i } ^ { t } , z _ { i } \big )$ to represent agent i’s belief model, and $p _ { \phi } ( a _ { i } ^ { t } \mid h _ { i } ^ { t } , z _ { i } , \hat { z } _ { - i } ^ { t } )$ to represent its planning model. This implementation enables parameter sharing across the two stages. At each training iteration, the current model $p _ { \phi }$ engages in self-play, acting as all agents to generate complete episodes. We then extract each agent’s local interaction trajectories and combine them into a training batch B. Using this batch, we update $\phi$ through policy gradient to optimize the objective of the current training stage, namely $\mathcal { T } _ { \mathrm { b e l i e f } }$ or $\mathcal { I } _ { \mathrm { p l a n n i n g } } .$ Finally, after certain iterations, training switches to the other stage. This alternating schedule approximates the bi-level problem in Eq. (5), allowing each stage to adapt to the other’s changes. The complete training procedure is outlined in Algorithm 1.

A practical concern in multi-agent training is that different agents may exhibit asymmetric expected returns due to inherent role asymmetries. Directly combining raw trajectories from different agents can introduce high variance, destabilizing the training process. Inspired by Liu et al. [2025b], we adopt a role-based advantage estimation:

$$
\begin{array} { r } { A _ { i } ^ { \psi } ( \tau _ { i } ) = R _ { i } ^ { \psi } ( \tau _ { i } ) - \mathrm { m e a n } ( \mathbf { R } _ { i } ^ { \psi } ) , } \end{array}\tag{10}
$$

where $\psi$ indicates the stage, $R _ { i } ^ { \psi } ( \tau _ { i } )$ is agent $i \ ' s$ return for that stage on trajectory $\tau _ { i } ,$ , and $\mathbf { R } _ { i } ^ { \psi }$ contains the corresponding returns across all rollout trajectories. This approach helps the unified LLM to learn a robust policy that generalizes across various roles.

## 5 Experiments

In this section, we design experiments to investigate the effectiveness of our proposed EpiAgent framework and the Epistemic Self-Play training algorithm. Specifically, we focus on answering the following research questions:

• RQ1: How does EpiAgent perform in multi-agent interaction scenarios? We investigate whether EpiAgent, which integrates explicit epistemic update with belief-conditioned action planning, can improve the agent’s overall performance in multi-agent interactions.

• RQ2: Does training the epistemic update provide additional performance benefits? We investigate whether learning epistemic updates from interaction feedback helps improve action planning, where the agent makes decisions based on its evolving beliefs.

• RQ3: Do learned beliefs provide useful information for action planning? We examine how well beliefs learned from intrinsic prediction reward reflect opponents’ private information, and whether these inferred beliefs help guide the agent’s subsequent action planning.

## 5.1 Experiment Setups

We train and evaluate LLM-based agents in modified two-player versions of Bargaining and Kuhn Poker. In both environments, each agent, regardless of its method, is privately assigned by the environment a behavioral persona that remains fixed within an episode and serves as the main private information for epistemic inference. We introduce three personas in Bargaining and four in Kuhn Poker that encourage different negotiation and betting styles. Bargaining additionally includes private target and bottomline prices, while our modified Kuhn Poker comprises five rounds with newly dealt private cards in each round. Detailed environment settings are provided in Appendix A.

In our experiments, we employ Qwen2.5-3B-Instruct as our base model for all methods and use LoRA for training. Models are trained based on REINFORCE++ [Hu et al., 2025] with role-based advantage estimation (Eq. (10)) for approximately 120 rollout-update iterations. The switch frequency L for our Epistemic Self-Play is set to 10, and training starts from the epistemic update stage. During evaluation, model parameters remain fixed. We conduct 128 episodes per agent pairing in each evaluation run with a temperature of 0.5. Further training and evaluation details can be found in Appendix C.

We adopt the following baselines in our experiments: ReAct [Yao et al., 2022] integrates reasoning into the decisionmaking process in LLMs, allowing models to generate thought traces before taking actions; Reflexion [Shinn et al., 2023] evaluates LLM-based agent’s current states and actions through self-reflection, providing feedback to guide its subsequent decisions; and Hypothetical Minds (HM) [Cross et al., 2024] enhances the LLM-based agent through a prompting-based opponent modeling module that generates and evaluates hypotheses about other agents’ policies. Also, we employ ReAct+RL as a training baseline, which applies the same RL algorithm to fine-tune the ReAct agent through self-play with the same number of iterations, optimizing its policy directly from the environment rewards. Furthermore, we include EpiAgent w/o Epistemic Training as an ablation baseline, which retains both stages of the decision process but optimizes only the action planning stage.

A. Bargaining (Agent 0)  
![](images/05e31557103377315144e81217b90f83fd69bbbea344a8ed598b3aeaaedca8e0.jpg)  
C. Kuhn Poker (Agent 0)

![](images/4885bde39a0ea93353506ac544b62bfc8626f82df749e159bc68fe26a38ef97f.jpg)  
D. Kuhn Poker (Agent 1)

![](images/a6474fa11757ad2b570d813c9ab42b839e58127dc17f280dcb0b18dfa4cc040a.jpg)

![](images/54236af6e5167422869d046c2e92b2e227ea3e19a7a3575c39dfbd39c80ee1c8.jpg)  
Figure 3: Cross-play evaluation results in Bargaining and Kuhn Poker, split by evaluated role. Columns identify the evaluated method and rows identify its opponent. Each cell reports the average reward (with format penalty) for the method. The bottom row averages over the five opponents. Full results with standard deviations are provided in Table 4.

Table 1: Action validity diagnostics for methods evaluated as Agent 0 in both environments. Valid reward is the mean reward over episodes in which neither agent commits a violation, averaged equally over opponents.
<table><tr><td rowspan="2">Method</td><td colspan="2">Bargaining</td><td colspan="2">Kuhn Poker</td></tr><tr><td>Violation (%)</td><td>Valid reward</td><td>Violation (%)</td><td>Valid reward</td></tr><tr><td>ReAct</td><td>0.00</td><td>-0.142</td><td>22.54</td><td>-0.189</td></tr><tr><td>Reflexion</td><td>0.36</td><td>-0.021</td><td>47.84</td><td>-0.111</td></tr><tr><td>HM</td><td>0.00</td><td>-0.203</td><td>55.34</td><td>-0.015</td></tr><tr><td>ReAct+RL</td><td>0.00</td><td>0.041</td><td>0.06</td><td>-0.026</td></tr><tr><td>EpiAgent</td><td>0.00</td><td>0.092</td><td>0.05</td><td>0.042</td></tr></table>

## 5.2 RQ1: How does EpiAgent perform in multi-agent interaction scenarios?

To answer this question, we conduct cross-play experiments to evaluate EpiAgent with prompting-based agents (ReAct, Reflexion, HM) and RL-trained agents (ReAct+RL) across the two environments. Figure 3 presents the results separately for each role, with full numerical results in Table 4. It can be seen that EpiAgent achieves the highest average reward over all opponents in both environments and roles, with gains distributed across most pairings, including interactions with RL-trained opponents. Since model parameters are kept fixed during evaluation, these results demonstrate that the inference and decision capabilities learned through self-play remain effective against agents with different decision frameworks, without additional training for each opponent.

![](images/83db7b17bb71c4f531fc925e765b7ff273229be4339831f88695dcb507c7c86d.jpg)

![](images/b9f02908b02522595d7cb4492d3be72fd88c345c9e08356d209c8c78bce65d38.jpg)  
Figure 4: Training evaluation of EpiAgent versus ReAct+RL baseline (which lacks the epistemic update stage) in both environments. The average reward at each checkpoint is evaluated against a fixed ReAct opponent.

Table 2: Ablation results on dedicated epistemic training in both environments. The variant w/o Epistemic Training retains both stages but trains only action planning. Each cell reports the average reward as Agent 1 against each opponent, and the Mean row is averaged over the five opponents.
<table><tr><td rowspan="2">Opponent</td><td colspan="2">Bargaining</td><td colspan="2">Kuhn Poker</td></tr><tr><td>EpiAgent</td><td>w/o Epistemic Training</td><td>EpiAgent</td><td>w/o Epistemic Training</td></tr><tr><td>ReAct</td><td>0.429</td><td>0.302</td><td>0.172</td><td>0.146</td></tr><tr><td>Reflexion</td><td>0.328</td><td>0.198</td><td>0.143</td><td>0.086</td></tr><tr><td>HM</td><td>0.260</td><td>0.139</td><td>0.034</td><td>0.063</td></tr><tr><td>ReAct+RL</td><td>0.304</td><td>0.212</td><td>0.070</td><td>0.004</td></tr><tr><td>EpiAgent</td><td>0.283</td><td>0.168</td><td>0.036</td><td>-0.146</td></tr><tr><td>Mean</td><td>0.321</td><td>0.204</td><td>0.091</td><td>0.031</td></tr></table>

In our environment settings, an invalid action would immediately terminate the episode and incur a −3 penalty for the responsible agent. Therefore, to examine whether the performance gains extend beyond improved compliance with action constraints, we further analyze each agent’s per-episode violation rate and its reward on episodes in which neither agent commits a violation, averaged equally over opponents. Table 1 first shows that prompting-based agents frequently produce invalid actions in Kuhn Poker, while their violation rates remain low in Bargaining. This is because the valid actions for agents in Kuhn Poker change in different situations, making prompting-based agents easily select invalid actions. In contrast, EpiAgent and ReAct+RL exhibit very low violation rates in both environments, indicating that RL training helps agents reliably follow the action constraints. When evaluation is restricted to violation-free episodes, EpiAgent still achieves the highest reward over opponents. Its advantage over ReAct+RL, despite their similarly low violation rates, further demonstrates the value of integrating an explicit epistemic layer with action planning.

## 5.3 RQ2: Does training the epistemic update provide additional performance benefits?

To investigate whether training the epistemic update provides additional performance benefits, we compare EpiAgent with EpiAgent w/o Epistemic Training, which retains both stages but optimizes only action planning using the same RL training procedure. The comparison results are shown in Table 2. EpiAgent achieves higher average rewards in both environments, with improvements across most pairings. Since both variants generate beliefs and condition actions on them, these gains demonstrate the benefit of training the epistemic update beyond simply including a belief stage in the decision process. The intrinsic prediction reward provides the epistemic update stage with a dedicated signal on whether its hypotheses about others’ private information account for subsequent observations, complementing the environment reward used to train action planning. Learning from this feedback therefore brings additional benefits to the decisions made under the inferred beliefs.

Beyond the final performance gains, we also examine how the explicit epistemic layer and its training affect the whole learning process. Figure 4 compares EpiAgent and ReAct+RL on successive checkpoints against a fixed ReAct opponent, obtained from independent training runs with identical configurations since intermediate checkpoints of the main runs were not retained. Both methods show broadly comparable early progress and improve over training, while EpiAgent achieves higher rewards at later checkpoints in both environments. During self-play, evolving opponents continually change the interaction evidence on which the agent bases its decisions. Training epistemic update on this evidence allows EpiAgent to refine its interpretation of others’ behavior alongside learning which actions to take, helping belief-conditioned action planning remain responsive to the interactions encountered during training. The performance advantage observed at later checkpoints is consistent with this design, complementing the ablation result and validating the value of coordinating epistemic update and action planning through Epistemic Self-Play.

Table 3: Results of test-time belief interventions, replacing epistemic update outputs (Learned) with Random and Oracle beliefs. Each cell reports the average reward against ReAct using the same trained EpiAgent model.
<table><tr><td rowspan="2">Belief Condition</td><td colspan="2">Bargaining</td><td colspan="2">Kuhn Poker</td></tr><tr><td>Agent 0</td><td>Agent 1</td><td>Agent 0</td><td>Agent 1</td></tr><tr><td>Random</td><td>0.009</td><td>0.406</td><td>0.049</td><td>0.091</td></tr><tr><td>Learned (EpiAgent)</td><td>0.160</td><td>0.429</td><td>0.181</td><td>0.172</td></tr><tr><td>Oracle</td><td>0.197</td><td>0.506</td><td>0.232</td><td>0.195</td></tr></table>

## 5.4 RQ3: Do learned beliefs provide useful information for action planning?

Given that training the epistemic update stage has shown benefits for the agent’s decision-making, a natural follow-up question is to what extent Epistemic Self-Play enhances the accuracy of the agent’s inferred beliefs. To explore this, we conduct quantitative evaluations of the agent’s beliefinference accuracy, defined as the percentage of times the agent correctly infers the opponent’s ground-truth persona, at three training phases: the initial (Step 0), intermediate (Step 60), and final (Step 120) phase against a fixed ReAct opponent. As illustrated in Figure 5, belief inference accuracy increases across the evaluated checkpoints in both environments, although ground-truth labels are not used for training. Instead, the intrinsic prediction reward evaluates how well the agent’s hypotheses explain subsequent observations. Since personas can influence opponents’ behavior, this signal helps the agent identify their private information from behavioral evidence. These results indicate that EpiAgent’s gains do not only stem from those training steps over the action planning stage, but are also rooted in a great improvement of the agent’s epistemic reasoning.

![](images/f4ffb5ced7240d5767053f76bb92dfc3475895fc9f2a3036e5f63316c779b176.jpg)  
Figure 5: Belief inference accuracy of EpiAgent at the initial (Step 0), intermediate (Step 60), and final (Step 120) stages of the Epistemic Self-Play training. Each accuracy is evaluated against a fixed ReAct opponent.

To further investigate whether these inferred beliefs help action planning, we conduct test-time belief intervention analysis. We evaluate EpiAgent under three belief conditions: Random samples a persona uniformly and keeps it fixed within each episode, Learned uses the model’s regular epistemic updates, and Oracle supplies the opponent’s assigned persona. Table 3 shows that our method (Learned) obtains higher rewards than Random in both environments and roles, demonstrating the benefit of conditioning action planning on beliefs inferred from the interaction. Since these comparisons use the same trained model, the differences show that its final performance depends on the belief condition supplied to action planning. Oracle further improves rewards in every setting, showing that the planner can exploit access to the opponent’s true persona. These results demonstrate that learned beliefs already contribute to effective action planning despite imperfect belief inference, while more accurate beliefs could bring further gains.

## 6 Conclusion

In this work, we present EpiAgent, a framework that equips LLM-based agents with an explicit, learnable epistemic layer by decomposing their decision-making process into two coupled stages, Epistemic Update and Action Planning, that jointly maintain and act on an explicit epistemic state. We formulate their learning dependency as a bi-level optimization problem and propose Epistemic Self-Play, an algorithm that alternately optimizes the two stages through self-play, leveraging an intrinsic reward to train epistemic updates from interaction feedback. Empirical evaluations on Bargaining and Kuhn Poker demonstrate that EpiAgent consistently outperforms both prompting-based and RL-based baselines.

Further analyses show that dedicated epistemic training provides additional performance benefits, belief inference improves over training, and learned beliefs contribute to effective action planning. These results demonstrate the value of learning to infer others’ private information and act on these inferences, suggesting a path toward LLM-based agents that can adaptively reason about and respond to others in complex social interactions.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, et al. Chatdev: Communicative agents for software development. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15174–15186, 2024.

Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi-agent debate. In Proceedings ofthe 2024 conference on empirical methods in natural language processing, pages 17889–17904, 2024.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227, 2025.

Yuxuan Lu, Bingsheng Yao, Hansu Gu, Jing Huang, Zheshen Jessie Wang, Yang Li, Jiri Gesi, Qi He, Toby Jia-Jun Li, and Dakuo Wang. Uxagent: An llm agent-based usability testing framework for web design. In Proceedings ofthe Extended Abstracts of the CHI Conference on Human Factors in Computing Systems, pages 1–12, 2025a.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. Searchr1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Michael L Littman. Markov games as a framework for multi-agent reinforcement learning. In Machine learning proceedings 1994, pages 157–163. Elsevier, 1994.

Stefano V Albrecht, Filippos Christianos, and Lukas Schäfer. Multi-agent reinforcement learning: Foundations and modern approaches. MIT Press, 2024.

Robert J Aumann. Interactive epistemology i: knowledge. International Journal of Game Theory, 28(3):263–300, 1999.

Eddie Dekel and Marciano Siniscalchi. Epistemic game theory. In Handbook of game theory with economic applications, volume 4, pages 619–702. Elsevier, 2015.

Andrés Perea. Epistemic game theory: reasoning and choice. Cambridge University Press, 2012.

Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. arXiv preprint arXiv:2305.16291, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chainof-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. Expel: Llm agents are experiential learners. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 19632– 19642, 2024.

Logan Cross, Violet Xiang, Agam Bhatia, Daniel LK Yamins, and Nick Haber. Hypothetical minds: Scaffolding theory of mind for multi-agent tasks with large language models. arXiv preprint arXiv:2407.07086, 2024.

Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B Tenenbaum, and Yejin Choi. Hypothesis-driven theory-of-mind reasoning for large language models. arXiv preprint arXiv:2502.11881, 2025.

Eric A. Hansen, Daniel S. Bernstein, and Shlomo Zilberstein. Dynamic programming for partially observable stochastic games. In Proceedings of the Nineteenth National Conference on Artificial Intelligence, pages 709–715. AAAI Press, 2004.

Xizhou Zhu, Yuntao Chen, Hao Tian, Chenxin Tao, Weijie Su, Chenyu Yang, Gao Huang, Bin Li, Lewei Lu, Xiaogang Wang, et al. Ghost in the minecraft: Generally capable agents for open-world environments via large language models with text-based knowledge and memory. arXiv preprint arXiv:2305.17144, 2023.

Hongliang He, Wenlin Yao, Kaixin Ma, Wenhao Yu, Yong Dai, Hongming Zhang, Zhenzhong Lan, and Dong Yu. Webvoyager: Building an end-to-end web agent with large multimodal models. arXiv preprint arXiv:2401.13919, 2024.

Xinnong Zhang, Jiayu Lin, Xinyi Mou, Shiyue Yang, Xiawei Liu, Libo Sun, Hanjia Lyu, Yihang Yang, Weihong Qi, Yue Chen, et al. Socioverse: A world model for social simulation powered by llm agents and a pool of 10 million real-world users. arXiv preprint arXiv:2504.10157, 2025a.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The eleventh international conference on learning representations, 2022.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in Neural Information Processing Systems, 36:8634–8652, 2023.

Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547, 2025b.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, Yuchen Zhang, Xinwei Long, Ermo Hua, et al. Ttrl: Test-time reinforcement learning. arXiv preprint arXiv:2504.16084, 2025.

Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, and Sida I Wang. Swe-rl: Advancing llm reasoning via reinforcement learning on open software evolution. arXiv preprint arXiv:2502.18449, 2025.

Huaye Zeng, Dongfu Jiang, Haozhe Wang, Ping Nie, Xiaotong Chen, and Wenhu Chen. Acecoder: Acing coder rl via automated test-case synthesis. arXiv preprint arXiv:2502.01718, 2025.

Yifei Zhou, Song Jiang, Yuandong Tian, Jason Weston, Sergey Levine, Sainbayar Sukhbaatar, and Xian Li. Sweet-rl: Training multi-turn llm agents on collaborative reasoning tasks. arXiv preprint arXiv:2503.15478, 2025.

Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, et al. Ragen: Understanding self-evolution in llm agents via multi-turn reinforcement learning. arXiv preprint arXiv:2504.20073, 2025.

Xinyi Li, Sai Wang, Siqi Zeng, Yu Wu, and Yi Yang. A survey on llm-based multi-agent systems: workflow, infrastructure, and challenges. Vicinagearth, 1(1):9, 2024.

Joon Sung Park, Joseph O’Brien, Carrie Jun Cai, Meredith Ringel Morris, Percy Liang, and Michael S Bernstein. Generative agents: Interactive simulacra of human behavior. In Proceedings ofthe 36th annual acm symposium on user interface software and technology, pages 1–22, 2023.

Zhenyu Guan, Xiangyu Kong, Fangwei Zhong, and Yizhou Wang. Richelieu: Self-evolving llm-based agents for ai diplomacy. Advances in Neural Information Processing Systems, 37:123471–123497, 2024.

Jianxiang Peng, Ling Shi, Xinwei Wu, Hanwen Zhang, Fujiang Liu, Haocheng Lyu, and Deyi Xiong. Diplomacyagent: Do llms balance interests and ethical principles in international events? In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 13732–13750, 2025.

Zelai Xu, Chao Yu, Fei Fang, Yu Wang, and Yi Wu. Language agents with reinforcement learning for strategic play in the werewolf game. In Proceedings of the 41st International Conference on Machine Learning, pages 55434–55464, 2024.

Xuanfa Jin, Ziyan Wang, Yali Du, Meng Fang, Haifeng Zhang, and Jun Wang. Learning to discuss strategically: A case study on one night ultimate werewolf. Advances in Neural Information Processing Systems, 37:77060–77097, 2024.

Zijing Shi, Meng Fang, Shunfeng Zheng, Shilong Deng, Ling Chen, and Yali Du. Cooperation on the fly: Exploring language agents for ad hoc teamwork in the avalon game. arXiv preprint arXiv:2312.17515, 2023.

Yihuai Lan, Zhiqiang Hu, Lei Wang, Yang Wang, Deheng Ye, Peilin Zhao, Ee-Peng Lim, Hui Xiong, and Hao Wang. Llm-based agent society investigation: Collaboration and confrontation in avalon gameplay. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 128–145, 2024.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. Camel: Communicative agents for "mind" exploration of large language model society. Advances in Neural Information Processing Systems, 36: 51991–52008, 2023a.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, et al. Autogen: Enabling next-gen llm applications via multi-agent conversations. In First Conference on Language Modeling, 2024.

John Yang, Carlos E Jimenez, Alexander Wettig, Kilian Lieret, Shunyu Yao, Karthik Narasimhan, and Ofir Press. Swe-agent: Agent-computer interfaces enable automated software engineering. Advances in Neural Information Processing Systems, 37:50528–50652, 2024.

Yexiang Liu, Jie Cao, Zekun Li, Ran He, and Tieniu Tan. Breaking mental set to improve reasoning through diverse multi-agent debate. In The Thirteenth International Conference on Learning Representations, 2025a.

Xuanfa Jin, Zhijian Ma, Yongcheng Zeng, Xinyu Cui, Haifeng Zhang, and Jun Wang. Remember and reweight: Enhancing multi-agent debate with experience memory and confidence estimation. arXiv preprint arXiv:2609.03619, 2026.

Chris Frith and Uta Frith. Theory of mind. Current biology, 15(17):R644–R645, 2005.

Kanishk Gandhi, Jan-Philipp Fränken, Tobias Gerstenberg, and Noah Goodman. Understanding social reasoning in language models with language models. Advances in Neural Information Processing Systems, 36:13518–13529, 2023.

Yufan Wu, Yinghui He, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. Hi-tom: A benchmark for evaluating higher-order theory of mind reasoning in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10691–10706, 2023.

Kazutoshi Shinoda, Nobukatsu Hojo, Kyosuke Nishida, Saki Mizuno, Keita Suzuki, Ryo Masumura, Hiroaki Sugiyama, and Kuniko Saito. Tomato: Verbalizing the mental states of role-playing llms for benchmarking theory of mind. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 1520–1528, 2025.

Huao Li, Yu Chong, Simon Stepputtis, Joseph P Campbell, Dana Hughes, Charles Lewis, and Katia Sycara. Theory of mind for multi-agent collaboration via large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 180–192, 2023b.

Zhining Zhang, Chuanyang Jin, Mung Yao Jia, and Tianmin Shu. Autotom: Automated bayesian inverse planning and model discovery for open-ended theory of mind. In ICLR 2025 Workshop on Foundation Models in the Wild, 2025c.

Yi-Long Lu, Chunhui Zhang, Jiajun Song, Lifeng Fan, and Wei Wang. Do theory of mind benchmarks need explicit human-like reasoning in language models? arXiv preprint arXiv:2504.01698, 2025b.

EunJeong Hwang, Yuwei Yin, Giuseppe Carenini, Peter West, and Vered Shwartz. Infusing theory of mind into socially intelligent llm agents. arXiv preprint arXiv:2509.22887, 2025.

Shao Zhang, Xihuai Wang, Wenhao Zhang, Yongshan Chen, Landi Gao, Dakuo Wang, Weinan Zhang, Xinbing Wang, and Ying Wen. Mutual theory of mind in human-ai collaboration: An empirical study with llm-driven ai agents in a real-time shared workspace task. arXiv preprint arXiv:2409.08811, 2024.

Zheng Tian, Shihao Zou, Ian Davies, Tim Warr, Lisheng Wu, Haitham Bou Ammar, and Jun Wang. Learning to communicate implicitly by actions. Proceedings ofthe AAAI Conference on Artificial Intelligence, 34(05):7261–7268, 2020. doi: 10.1609/aaai.v34i05.6217.

Leslie Pack Kaelbling, Michael L. Littman, and Anthony R. Cassandra. Planning and acting in partially observable stochastic domains. Artificial Intelligence, 101(1–2):99–134, 1998. doi: 10.1016/S0004-3702(98)00023-X.

Richard S Sutton, David McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. Advances in neural information processing systems, 12, 1999.

Chris L Baker, Rebecca Saxe, and Joshua B Tenenbaum. Action understanding as inverse planning. Cognition, 113(3): 329–349, 2009.

Tan Zhi-Xuan, Jordyn Mann, Tom Silver, Josh Tenenbaum, and Vikash Mansinghka. Online bayesian goal inference for boundedly rational planning agents. Advances in neural information processing systems, 33:19238–19250, 2020.

Bo Liu, Leon Guertler, Simon Yu, Zichen Liu, Penghui Qi, Daniel Balcells, Mickel Liu, Cheston Tan, Weiyan Shi, Min Lin, et al. Spiral: Self-play on zero-sum games incentivizes reasoning via multi-agent multi-turn reinforcement learning. arXiv preprint arXiv:2506.24119, 2025b.

Jian Hu, Jason Klein Liu, Haotian Xu, and Wei Shen. Reinforce++: Stabilizing critic-free policy optimization with global advantage normalization, 2025. URL https://arxiv.org/abs/2501.03262.

He He, Derek Chen, Anusha Balakrishnan, and Percy Liang. Decoupling strategy and generation in negotiation dialogues. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2333–2343, 2018.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

## A Detailed Environment Settings

To train and validate our method with other baselines, we construct and modify two distinct yet challenging multi-agent environments: Bargaining and Kuhn Poker, both of which retain the characteristic of incomplete information and non-stationarity in multi-agent interactions. Specifically, the Bargaining combines a shared incentive to reach agreement with conflicting preferences over the deal price: the seller seeks a higher price, while the buyer seeks a lower one. If no agreement is reached, both players receive zero game reward. Therefore, it requires strategic negotiation, concessions, and inference about the opponent’s preferences and constraints. Meanwhile, the Kuhn Poker is a classic competitive game involving hidden information and deception. Pure strategies in this game are usually bad choices, as they can be exploited easily. So the agent requires the capabilities to make decisions under uncertainty while reasoning about opponents’ behavioral tendencies.

## A.1 Bargaining

Environment Description. The Bargaining involves two players: a Seller and a Buyer, who negotiate over the deal price of a single item. In this game, both the seller and buyer are assigned private information consisting of a Target Price, a Bottomline Price, and the item’s description (can be different for each player, as the seller usually knows more than the buyer), with an Initial Price public to all. All scenario settings in our experiments are extracted from the Craigslist Bargain dataset [He et al., 2018]. The objectives for all players are:

• The Seller aims to MAXIMIZE the final deal price, with its Target Price above its Bottomline Price. Thus, the seller is incentivized to start negotiations above its target price and concede slowly, strictly rejecting any deal below its bottomline price.

• The Buyer aims to MINIMIZE the final deal price, with its Target Price below its Bottomline Price. So the buyer is supposed to start negotiations below its target price and concede slowly upward, strictly rejecting any deal above its bottomline price.

The bargaining process proceeds in a maximum of 10 turns. If both players fail to reach a deal in the end, then this episode will terminate immediately.

Action Space. The bargaining process proceeds in turns. Since there is a public initial price, we set the buyer (Agent 0) as the first player in our environment. For the players’ action space, in addition to the natural language messages used for negotiation and persuasion, we also employ a set of symbolic actions to formalize the state transitions. These actions are listed as follows:

• [OFFER:\$M]: Make an initial price \$M offer for the item.

• [COUNTER:\$M]: Make a counteroffer \$M in response to the opponent’s proposal.

• [QUESTION]: Ask questions to gather information or clarify details.

• [REJECT]: Reject the current offer without making a counteroffer.

• [ACCEPT]: Accept the current offer and close the deal.

• [QUIT]: End the bargaining without reaching an agreement.

Behavioral Personas. While LLM-based agents demonstrate strong capabilities in role-playing, they tend to exhibit behavioral homogenization, often converging towards "average" strategies, behavior patterns, or generic preferences present in their pre-training data. This lack of diversity presents two critical issues: (1) It limits the richness of the generated interaction data; (2) It trivializes the challenge of belief inference, as there is little variation in opponents’ latent states to distinguish.

To mitigate this and introduce realistic variance into the environment, we employ three distinct yet representative behavioral personas in the Bargaining environment:

• Patient: Prioritizes obtaining a favorable price. This persona adopts a "tough" stance, making minimal and slow concessions. They strictly adhere to their target price and are willing to walk away if the offer does not meet their expectations.

• Balanced: Balances price with the prospect of reaching an agreement. This persona simulates a rational negotiator who values reciprocity, making gradual concessions in response to the opponent’s moves to reach an equitable agreement.

• Urgent: Prioritizes reaching a deal quickly over maximizing the price margin. Driven by a preference for reaching agreement quickly, this persona is more likely to make larger and faster concessions to secure an agreement within their bottomline constraints.

Reward Design. The game reward of each player in the Bargaining environment depends on the final deal price and their role. If no deal is reached, both players receive zero game reward. Given a deal at terminal step T with price $p ^ { \star }$ the Seller and Buyer’s rewards are defined as:

$$
r _ { \mathrm { S } } ^ { T } = \frac { p ^ { \star } - p _ { \mathrm { S } } ^ { \mathrm { b o t t o m } } } { p _ { \mathrm { S } } ^ { \mathrm { t a r g e t } } - p _ { \mathrm { S } } ^ { \mathrm { b o t t o m } } } , \quad r _ { \mathrm { B } } ^ { T } = \frac { p _ { \mathrm { B } } ^ { \mathrm { b o t t o m } } - p ^ { \star } } { p _ { \mathrm { B } } ^ { \mathrm { b o t t o m } } - p _ { \mathrm { B } } ^ { \mathrm { t a r g e t } } } ,\tag{11}
$$

where $p _ { \mathrm { S } } ^ { \mathrm { b o t t o m } }$ and $p _ { \mathrm { S } } ^ { \mathrm { t a r g e t } }$ denote the seller’s bottomline and target price, respectively, with analogous definitions for the buyer. For each player, the reward equals zero at its bottomline and one at its target, and lies in [0, 1] when the deal price falls between these thresholds. Rewards are not clipped: outcomes more favorable than the target yield rewards above one, while outcomes worse than the bottomline yield negative rewards.

Besides the game reward, we also introduce a format penalty (−3) in the textual environment to ensure parsable outputs from the LLM-based agents. And once an invalid response is taken, the environment will terminate immediately.

## A.2 Kuhn Poker

Environment Description. The Kuhn Poker is a simplified form of poker, serving as a simple model of a zero-sum, two-player, imperfect information game. In this game, the deck includes only three playing cards, usually a Jack, Queen, and King. Both players are dealt a card, and their available actions depend on the current betting state. If both players check, or a bet is called, the player with the higher card wins at showdown. If a player folds, the opponent wins the pot. The opponent’s card is revealed only at showdown.

One round of the Kuhn Poker has a maximum depth of 3 actions. The dynamics allow for complex strategic maneuvers, such as bluffing (betting with a weak card) or slow-play (checking with a strong card to induce a bet). To better help the agent infer the opponent’s potential strategy style, we extend the gameplay to 5 rounds with newly dealt cards in each round, which means there are a maximum of 15 turns in an interaction episode. And we set player 1 (Agent 1) as the first player in this environment.

Action Space. Different from the Bargaining game, where the player’s action contains natural language messages, the action space in the Kuhn Poker game is discrete and only contains four possible actions:

• [check]: Pass without betting (only if no bet is on the table).

• [bet]: Add 1 chip to the pot (only if no bet is on the table).

• [call]: Match an opponent’s bet by adding 1 chip to the pot.

• [fold]: Surrender and let your opponent win the pot.

Behavioral Personas. Similarly, to increase the diversity of the interaction trajectories and align more closely with the strategic biases of human players, we also employ four representative behavioral personas in the Kuhn Poker environment. Each player is privately assigned a persona through its prompt, and this assignment remains fixed across all five rounds:

• Balanced (GTO): Approximates a Game Theoretic Optimal strategy. This persona encourages mixed strategies, such as probabilistic bluffing and calling, to maintain unpredictable behavior across hands.

• Aggressive (Over-Bluffer): Adopts a hyper-offensive style. This persona bets frequently with weak hands (J) to pressure opponents, relying on intimidation rather than card strength.

Table 4: Cross-play evaluation results in Bargaining and Kuhn Poker. Each cell reports the average reward (with format penalty) and standard deviation for a specific Agent 0 (row) v.s. Agent 1 (column) pairing: the upper number is Agent 0’s average reward, and the lower one is Agent 1’s. Red values mark the best Agent 0 per column; the blue values mark the best Agent 1 per row.
<table><tr><td rowspan="2">Agent 0</td><td colspan="5">Agent 1</td></tr><tr><td>ReAct</td><td>Reflexion</td><td>HM</td><td>ReAct+RL</td><td>EpiAgent</td></tr><tr><td colspan="6">Bargaining</td></tr><tr><td>ReAct</td><td> $- 0 . 0 5 2 \pm 0 . 0 2 1$ </td><td> $- 0 . 1 3 1 \pm 0 . 0 8 6$ </td><td> $- 0 . 2 3 6 \pm 0 . 2 1 9$ </td><td> $- 0 . 1 5 5 \pm 0 . 0 3 8$ </td><td> $- 0 . 1 3 6 \pm 0 . 0 4 3$ </td></tr><tr><td>Reflexion</td><td> $0 . 3 2 6 \pm 0 . 0 0 9$   $0 . 0 1 7 \pm 0 . 0 1 5$ </td><td> $0 . 2 9 5 \pm 0 . 0 4 7$   $- 0 . 0 0 5 \pm 0 . 0 4 5$ </td><td> $0 . 3 2 9 \pm 0 . 1 2 9$   $- 0 . 1 3 6 \pm 0 . 1 6 1$ </td><td> $0 . 3 9 0 \pm 0 . 0 4 5$   $- 0 . 0 1 8 \pm 0 . 1 3 5$ </td><td> $0 . 4 2 9 \pm 0 . 0 3 1$   $- 0 . 0 2 0 \pm 0 . 0 5 6$ </td></tr><tr><td>HM</td><td> $0 . 1 8 7 \pm 0 . 0 2 8$   $- 0 . 0 5 5 \pm 0 . 1 1 9$ </td><td> $0 . 0 2 7 \pm 0 . 0 5 5$   $- 0 . 2 2 9 \pm 0 . 0 9 4$ </td><td> $0 . 1 6 4 \pm 0 . 1 3 3$   $- 0 . 3 3 2 \pm 0 . 0 4 7$ </td><td> $0 . 2 1 8 \pm 0 . 0 6 4$   $- 0 . 2 1 4 \pm 0 . 1 0 1$ </td><td> $0 . 3 2 8 \pm 0 . 0 2 2$   $- 0 . 1 8 2 \pm 0 . 2 2 0$ </td></tr><tr><td>ReAct+RL</td><td> $0 . 0 9 6 \pm 0 . 0 8 2$   $0 . 1 0 4 \pm 0 . 0 4 4$ </td><td> $0 . 0 7 0 \pm 0 . 0 8 7$   $0 . 0 5 0 \pm 0 . 0 5 0$ </td><td> $0 . 2 0 6 \pm 0 . 0 5 3$   $- 0 . 0 1 4 \pm 0 . 1 3 1$ </td><td> $0 . 1 3 3 \pm 0 . 1 2 0$   $0 . 0 2 3 \pm 0 . 0 0 8$ </td><td> $0 . 2 6 0 \pm 0 . 1 4 7$   $0 . 0 4 1 \pm 0 . 0 3 3$ </td></tr><tr><td>EpiAgent</td><td> $0 . 1 7 5 \pm 0 . 0 1 5$   $0 . 1 6 0 \pm 0 . 0 2 0$ </td><td> $0 . 1 4 7 \pm 0 . 0 4 3$   $0 . 1 0 8 \pm 0 . 0 5 5$ </td><td> $0 . 1 7 7 \pm 0 . 0 7 4$   $0 . 0 1 4 \pm 0 . 0 2 0$ </td><td> $0 . 2 3 8 \pm 0 . 0 1 5$   $0 . 0 9 1 \pm 0 . 0 6 9$ </td><td> $0 . 3 0 4 \pm 0 . 0 2 5$   $0 . 0 8 5 \pm 0 . 0 3 1$ </td></tr><tr><td colspan="6"> $0 . 2 2 7 \pm 0 . 0 1 7$   $0 . 2 3 9 \pm 0 . 0 4 7$   $0 . 2 9 8 \pm 0 . 0 2 6$ </td></tr><tr><td>ReAct</td><td> $- 0 . 7 8 6 \pm 0 . 0 9 4$   $- 0 . 1 5 1 \pm 0 . 1 3 8$ </td><td> $- 0 . 7 3 4 \pm 0 . 0 3 5$   $- 1 . 0 2 9 \pm 0 . 0 4 3$ </td><td> $- 0 . 7 9 7 \pm 0 . 0 8 1$   $- 0 . 7 6 0 \pm 0 . 1 8 3$ </td><td> $- 0 . 7 9 2 \pm 0 . 0 3 0$   $0 . 1 9 0 \pm 0 . 0 3 7$ </td><td> $- 0 . 8 9 8 \pm 0 . 0 7 5$   $0 . 1 7 2 \pm 0 . 0 9 0$ </td></tr><tr><td>Reflexion</td><td> $- 1 . 4 2 5 \pm 0 . 0 7 2$   $- 0 . 2 6 3 \pm 0 . 0 7 3$ </td><td> $- 1 . 1 7 0 \pm 0 . 0 7 6$   $- 0 . 8 0 2 \pm 0 . 1 9 0$ </td><td> $- 1 . 4 2 4 \pm 0 . 1 5 2$   $- 0 . 6 3 8 \pm 0 . 1 3 9$ </td><td> $- 1 . 6 9 5 \pm 0 . 0 4 1$   $0 . 1 2 5 \pm 0 . 0 2 3$ </td><td> $- 1 . 7 1 4 \pm 0 . 0 9 9$   $0 . 1 4 3 \pm 0 . 0 8 6$ </td></tr><tr><td>HM</td><td> $- 1 . 7 1 4 \pm 0 . 1 5 1$   $- 0 . 2 4 2 \pm 0 . 0 8 2$ </td><td> $- 1 . 3 4 9 \pm 0 . 1 9 5$   $- 1 . 2 6 6 \pm 0 . 2 1 9$ </td><td> $- 1 . 5 8 7 \pm 0 . 0 9 0$   $- 0 . 7 1 1 \pm 0 . 0 6 8$ </td><td> $- 1 . 7 0 8 \pm 0 . 0 4 7$   $- 0 . 0 1 0 \pm 0 . 0 3 0$ </td><td> $- 1 . 9 4 0 \pm 0 . 1 1 4$   $0 . 0 3 4 \pm 0 . 0 3 3$ </td></tr><tr><td>ReAct+RL</td><td> $0 . 1 0 9 \pm 0 . 0 3 5$   $- 0 . 4 6 1 \pm 0 . 0 7 5$ </td><td> $- 0 . 0 2 6 \pm 0 . 0 5 8$   $- 1 . 8 9 1 \pm 0 . 0 7 7$ </td><td> $- 0 . 0 6 0 \pm 0 . 0 6 1$   $- 1 . 1 4 3 \pm 0 . 0 9 1$ </td><td> $0 . 0 0 0 \pm 0 . 0 1 4$   $- 0 . 0 0 8 \pm 0 . 0 0 0$ </td><td> $- 0 . 0 7 8 \pm 0 . 0 4 8$   $0 . 0 7 0 \pm 0 . 0 4 1$ </td></tr><tr><td>EpiAgent</td><td> $0 . 1 8 1 \pm 0 . 0 7 6$   $- 0 . 4 4 5 \pm 0 . 0 5 1$ </td><td> $0 . 0 1 9 \pm 0 . 0 6 6$   $- 2 . 0 0 3 \pm 0 . 1 0 7$ </td><td> $- 0 . 0 5 2 \pm 0 . 0 5 5$   $- 1 . 2 6 0 \pm 0 . 0 6 3$ </td><td> $0 . 0 9 6 \pm 0 . 1 4 5$   $- 0 . 0 9 6 \pm 0 . 1 4 5$ </td><td> $- 0 . 0 4 4 \pm 0 . 1 1 5$   $0 . 0 3 6 \pm 0 . 1 0 7$ </td></tr></table>

• Conservative (Folder): Exhibits extreme risk aversion. This persona folds frequently when facing aggression and only commits chips to the pot when holding strong cards (K), making them easy to bluff but dangerous to call.  
• Calling Station (Over-Caller): A passive but stubborn style. This persona rarely folds and calls excessively, operating under the belief that opponents are constantly bluffing. They maximize showdowns regardless of expected value.

Reward Design. The game reward of each player in the Kuhn Poker environment only depends on their final chips after 5 rounds. The player with more chips wins and receives 1 reward, while its opponent receives −1 reward. If both players have the same chips (i.e., they tie), then both of them receive 0 reward.

Also, we add a penalty of −3 for invalid responses or actions, including parsing failures and actions that are illegal in the current betting state. The penalty applies to the responsible player, and the environment will terminate the episode immediately if it receives such a response.

## B Additional Experiment Results

## B.1 Full Cross-Play Rewards

Table 4 provides the numerical rewards underlying the main heatmap, including the reported standard deviations. The table preserves the original pairing layout: rows identify Agent 0 and columns identify Agent 1, with both agents’ rewards listed in each cell.

## B.2 Ablation Results of Switch Frequency

We conducted an additional ablation study on Kuhn Poker with different switch frequencies $L \in \{ 5 , 1 0 , 1 5 \}$ . After training, we select the checkpoint with the best combined performance of both agents during self-play training. This criterion relies only on training-time self-play statistics and does not involve any evaluation opponent. The selected checkpoints are then evaluated against a fixed ReAct opponent, and the average rewards are reported in Table 5.

Table 5: Performance of EpiAgent under different training switch frequencies L.
<table><tr><td>Switch Frequency L</td><td>5</td><td>10</td><td>15</td></tr><tr><td>Average Reward</td><td>0.041</td><td>0.181</td><td>0.039</td></tr></table>

The results show that both $L = 5$ and $L = 1 5$ yield lower performance than $L = 1 0$ , which is consistent with our intuition: switching too frequently (small L) may leave insufficient time to improve each stage under its current objective, while switching too infrequently (large L) may delay each stage’s response to changes in the other. Thi suggests that the switch frequency should be chosen to balance these two extremes, and $L = 1 0$ serves as an effective choice in our setting.

## C Detailed Experiment Settings

## C.1 Training Details

We implement the training pipeline based on RAGEN [Wang et al., 2025] and VeRL [Sheng et al., 2025]. As the original RAGEN framework only supports training on a single agent, we modified it to be suitable for our multi-agent rollout and training. Due to resource constraints, we select REINFORCE++ [Hu et al., 2025] as the RL training algorithm to save resources and for efficient training. We employ Qwen2.5-3B-Instruct as our base model for all experiments and use LoRA for the model training.

During the multi-agent rollout, we use the same LLM to perform self-play to collect interaction trajectories. As our experimental environments are multi-turn, multi-agent interaction scenarios, each agent acts as the "assistant" in its own message list, while the environment feedback and other agents’ actions act as the "user". For the rollout hyperparameters, we set temperature=1.0, top\_p=1.0, top\_k=-1, and use vLLM for inference acceleration. Considering the randomness of LLM’s generation, at each training step, we initialize 32 environment groups with different random seeds, each containing 16 environments with identical initialization, and let the LLM agents perform rollouts across all $3 2 \times 1 6 = 5 1 2$ environments in parallel. This yields 512 trajectories per agent. Following the rollout filtering technique in RAGEN, we adopt the within-group standard deviation of returns as the filtering criterion and set filter\_ratio=0.25, retaining 128 trajectories per agent after filtering. Finally, since we mix trajectories from all agents during training, each training step uses $1 2 8 \times 2 = 2 5 6$ trajectories under our two-agent settings.

During training, we set the LoRA rank to 32 and alpha to 16, with LoRA adapters applied to all linear layers of the base model. Since the optimal learning rate for LoRA training usually differs from that of full fine-tuning, we adopt the Adam optimizer with a learning rate of 7e-6. Following common practice in LLM training, we also apply a KL-divergence constraint between the current policy and the reference model, using the K2 estimator with a coefficient of 0.001. Based on the ablation results in Appendix B.2, we set the switch frequency between the two stages of Epistemic Self-Play to 10, which achieves the highest average reward among the tested settings. All other training-related hyperparameters follow the default settings of RAGEN and VeRL.

## C.2 Computational Consumption

All experiments are conducted on a node equipped with 8 NVIDIA A100 GPUs. We use bfloat16, Flash-Attention, and gradient checkpointing to run our experiments. For each environment, a 120-step training run takes approximately 40 hours, with most of the time spent on multi-agent rollouts.

## D Prompt Templates

In this section, we present all prompt templates used in our experiments. Specifically, the epistemic update and action planning prompts are used for EpiAgent’s decision-making process, while the opponent prediction prompt is leveraged to construct the belief-conditioned prediction trajectories, which are further used to compute the agent’s intrinsic prediction reward based on the LLM’s output log-probabilities.

## D.1 Bargaining

## Environment Prompt Environment Prompt

```markdown
You are participating in a bargaining negotiation between a seller and a buyer over
an item. You will be assigned either the seller or buyer role.
## Your Capabilities
You can perform multiple negotiation-related tasks:
- **Belief Inference**: Analyze opponent behavior to infer their strategy and type
- **Action Planning**: Determine optimal negotiation actions and messages
- **Opponent Prediction**: Predict opponent’s most likely next moves and responses
## Game Framework
### Roles and Objectives
- **Seller**: Maximize sale price within constraints
B Mi i i h i i hi i
- Each player has private information (target price, bottomline price, type, etc.)
### Price Parameters
- **Target Price**: Your ideal outcome that balances ambition with realism - the
PRIMARY objective to aim for
- Sellers: Start ABOVE your target price, concede slowly downward
- Buyers: Start BELOW your target price, concede slowly upward
- **Bottomline Price**: Your walk-away point - crossing this means no deal is better
than a bad deal
- Seller: The minimum price you will accept (never go below this)
- Buyer: The maximum price you will pay (never go above this)
### Available Actions
- **OFFER: $M**: Make an initial price $M offer for the item
- **COUNTER: $M**: Make a counteroffer $M in response to the opponent’s proposal
- **QUESTION**: Ask questions to gather information or clarify details
- **REJECT**: Reject the current offer without making a counteroffer
- **ACCEPT**: Accept the current offer and close the deal
- **QUIT**: End the negotiation without reaching an agreement
### Player Types
Participants exhibit three main types:
- **Patient**: Prioritize maximizing own benefit, make minimal and slow concessions,
willing to walk away if the price is not close to the target, rarely compromise
- **Balanced**: Seek fair outcomes with moderate patience, make gradual and
reciprocal concessions, willing to negotiate but won’t accept unfair deals
- **Urgent**: More likely to make or accept larger and faster concessions to reach
agreement, prefer securing a deal over maximizing price
### Core Principles
- **Maximize Your Outcome**: ALWAYS aim for the best possible price towards your
target
- **Strategic Communication**: Justify offers with concrete reasons (quality,
features, market value); use persuasive framing to influence opponent
- **Strategic Concessions**: Make small, justified concessions strategically, not
eagerly
- **Effective Anchoring**: Initial offers shape negotiation range; start with room
for strategic concessions toward target
- **Bottomline is Sacred**: NEVER cross your bottomline under any circumstances
```

\*\*Valid Actions\*\*: Only use OFFER, COUNTER, QUESTION, REJECT, ACCEPT, or QUIT   
\*\*Remember\*\*: You are a strategic agent with preferences, not a random actor or a   
rigid rule-follower. Make decisions that serve your goals while adapting to the   
strategic context. And always output in English.

## Epistemic Update Prompt

```markdown
State:
Now it is Turn {TURN_NUM} in the negotiation. You are the **{ROLE}** during the
bargaining negotiation.
## Product Information
Name: {PRODUCT_NAME}
Description: {PRODUCT_DESC}
Initial Price: ${INITIAL_PRICE}
## Your Negotiation Type
{PLAYER_TYPE_DESC}
## Negotiation Process
Recent Messages:
{RECENT_MESSAGES}
Offer History:
{OFFER_HISTORY}
## Task: Infer Your Opponent’s Type
Based on:
The negotiation history and opponent’s behavioral patterns
Your observations of their concession style, time sensitivity, and risk tolerance
Your own type’s perspective on their actions
Analyze your opponent ({OPPONENT_ROLE})’s behavior to estimate THEIR type, target
price, and bottomline.
## Output Format
Always output: ‘{RESPONSE_FORMAT}‘ with no extra text.
The inferred belief about your opponent must be enclosed within the
<answer></answer> tag with the structure format:
‘<answer>**ESTIMATED_OPPONENT_TYPE**: [Patient / Balanced / Urgent] (Confidence:
[High / Medium / Low])
**ESTIMATED_OPPONENT_TARGET**: $X (Range: $Y - $Z)
**ESTIMATED_OPPONENT_BOTTOMLINE**: $A (Range: $B - $C)
**STRATEGIC_ASSESSMENT**: [1-2 sentences summarizing their overall
strategy]</answer>‘
You have {TURNS_LEFT} turns left. Strictly follow the output format. Max response
length: {MAX_TOKENS} tokens.
```

## Action Planning Prompt

Now it is Turn {TURN\_NUM} in the negotiation. You are the \*\*{ROLE}\*\* during the bargaining negotiation.

Name: {PRODUCT\_NAME}   
Description: {PRODUCT\_DESC}   
Initial Price: \${INITIAL\_PRICE}   
## Your Private Information   
Your Target: \${TARGET\_PRICE}   
Your Bottomline: \${BOTTOMLINE\_PRICE}   
## Your Negotiation Type   
{PLAYER\_TYPE\_DESC}   
## Negotiation Process   
- Recent Messages:   
{RECENT\_MESSAGES}   
Offer History:   
{OFFER\_HISTORY}   
## Your Latest Belief About Opponent   
{BELIEF}   
## Task: Plan Your Next Action   
Based on:   
The negotiation history and current offer on the table   
Your latest belief about the opponent’s type and price range   
Your own type’s values and strategic priorities   
Remaining turns and negotiation phase   
Determine your next negotiation move strategically.   
\*\*Remember\*\*: Align with your type’s values while remaining strategically adaptive.   
## Output Format   
Always output: ‘{RESPONSE\_FORMAT}‘ with no extra text.   
All responses enclosed within the <answer></answer> tag must follow this format with   
NO exceptions: ‘<answer>[ACTION] your negotiation message</answer>‘   
The ACTION MUST be enclosed in [square brackets], and the message MUST be a complete   
sentence and align with the ACTION.   
### Output Example   
You are a buyer with target price \$50 and bottomline price \$80. The seller offers   
\$100 for an item.   
Example responses:   
‘<answer>[COUNTER: \$60] I think \$100 is too high. How about \$60?</answer>‘   
‘<answer>[QUESTION] What’s the condition of the laptop? Any warranty   
included?</answer>‘   
‘<answer>[REJECT] That price is way beyond my budget.</answer>‘   
You have {TURNS\_LEFT} turns left. Strictly follow the output format. Max response   
length: {MAX\_TOKENS} tokens.

## Opponent Prediction Prompt

State:   
Now it is Turn {TURN\_NUM} in the negotiation. You are the \*\*{ROLE}\*\* during the   
bargaining negotiation.   
## Product Information   
- Name: {PRODUCT\_NAME}

```markdown
- Description: {PRODUCT_DESC}
- Initial Price: ${INITIAL_PRICE}
## Your Negotiation Type
{PLAYER_TYPE_DESC}
## Negotiation Process
- Recent Messages:
{RECENT_MESSAGES}
- Offer History:
{OFFER_HISTORY}
## Your Latest Belief About Opponent
{BELIEF}
## Your Proposed Action
{ACTION}
## Task: Predict Your Opponent’s Response
Based on:
Your belief about the opponent’s type, target, and bottomline
The negotiation history and their behavioral pattern
- Your proposed action above
Predict your opponent’s most likely response to your proposed action.
**Remember**: Think from the opponent’s perspective, following their expected
negotiation pattern. Output ONLY the opponent’s predicted response, NO extra
content.
## Output Format
Always output: ‘{RESPONSE_FORMAT}‘ with no extra text.
Your predicted response enclosed within the <answer></answer> tag must follow
this format with NO exceptions: ‘<answer>OPPONENT_ROLE: their predicted
message</answer>‘
### Output Example
You are a buyer who believes the opponent is a **Patient** seller (estimated target:
$80, bottomline: $60). You just proposed: ’Given the condition, I think $40 is
fair.’
Example prediction:
‘<answer>seller: That’s too low for an almost brand-new item. I can come down to
$75, but that’s as low as I’m willing to go.</answer>‘
You have {TURNS_LEFT} turns left. Strictly follow the output format. Max response
length: {MAX_TOKENS} tokens.
```

## D.2 Kuhn Poker

Environment Prompt   
You are playing Kuhn Poker, a simplified two-player poker game. Make valid actions   
to win.   
## Your Capabilities   
You can perform multiple game-related tasks:

```markdown
- **Belief Inference**: Analyze opponent behavior to infer their types and strategy
styles
- **Action Planning**: Determine optimal actions based on your beliefs about the
opponent
- **Opponent Prediction**: Predict opponent’s most likely next moves based on the
interaction histories and your beliefs
## Game Framework
### Game Rules
Kuhn Poker uses a 3-card deck with J, Q, K (J lowest, K highest)
Each player antes 1 chip and receives 1 card each round (note that the cards are
dealt without replacement, so you cannot have the same card as your opponent)
Game continues for 5 rounds
The player with the **most chips after all rounds** wins
### Action Rules
’[check]’: Pass without betting (only if no bet is on the table)
’[bet]’: Add 1 chip to the pot (only if no bet is on the table)
’[call]’: Match an opponent’s bet by adding 1 chip to the pot
’[fold]’: Surrender your hand and let your opponent win the pot
### Player Types
Players in Kuhn Poker exhibit four main strategy styles:
- **Balanced (GTO)**: Play near-optimal mixed strategies to remain unexploitable,
use theoretically sound bluffing and calling frequencies, maintain unpredictable
patterns across hands
- **Aggressive (Over-Bluffer)**: Bet and bluff frequently with weak hands to
pressure opponents into folding, highly offensive and willing to take risks, prefer
winning through intimidation
- **Conservative (Folder)**: Extremely risk-averse and cautious, fold frequently
when facing bets, rarely bluff with weak hands, only commit chips with strong
confidence
- **Calling Station (Over-Caller)**: Call excessively and rarely fold, reluctant to
give up on any hand, believe opponents are often bluffing, prefer seeing showdowns
over folding
```

## Epistemic Update Prompt

State:   
{GAME\_OBSERVATION}   
## Task: Infer Your Opponent’s Type   
Based on:   
The interaction history and opponent’s behavioral patterns   
Your observations of their strategy style   
Your own type’s perspective on their actions   
Analyze your opponent’s behavior to estimate THEIR type and strategy.   
## Output Format   
Always output: ‘{RESPONSE\_FORMAT}‘ with no extra text.   
The inferred belief about your opponent must be enclosed within the   
<answer></answer> tag with the structure format:   
‘<answer>\*\*ESTIMATED\_OPPONENT\_TYPE\*\*: [Balanced / Aggressive / Conservative /   
Calling Station] (Confidence: [High / Medium / Low])

\*\*STRATEGIC\_ASSESSMENT\*\*: [1-2 sentences summarizing their overall strategy and   
tendencies]</answer>‘   
Strictly follow the output format. Max response length: {MAX\_TOKENS} tokens.

## Action Planning Prompt

State:   
{GAME\_OBSERVATION}   
## Your Latest Belief About Opponent   
{BELIEF}   
## Task: Plan Your Next Action   
Based on:   
The interaction history and current observation   
Your latest belief about the opponent’s type and strategy style   
Your own type’s values and strategic priorities   
Determine your next move strategically.   
\*\*Remember\*\*: Align with your type’s values while remaining strategically adaptive.   
## Output Format   
Always output: ‘{RESPONSE\_FORMAT}‘ with no extra text.   
All responses enclosed within the <answer></answer> tag must follow this format with   
NO exceptions: ‘<answer>[ACTION]</answer>‘   
The ACTION MUST be enclosed in [square brackets], and you MUST choose exactly one   
action from the current available actions above.   
Strictly follow the output format. Max response length: {MAX\_TOKENS} tokens.

## Opponent Prediction Prompt

State:   
{GAME\_OBSERVATION}   
## Your Chosen Action   
{ACTION}   
## Your Latest Belief About Opponent   
{BELIEF}   
## Task: Predict Your Opponent’s Move   
Based on:   
Your belief about the opponent’s type and strategy style   
The interaction history and their behavioral pattern   
Your chosen action above   
Predict your opponent’s most likely move to your chosen action.   
\*\*Remember\*\*: Think from the opponent’s perspective, following their expected   
behavioral pattern. Output ONLY the opponent’s predicted move, NO extra content.   
## Output Format   
Always output: ‘{RESPONSE\_FORMAT}‘ with no extra text.   
Your predicted move enclosed within the <answer></answer> tag must follow this   
format with NO exceptions: ‘<answer>[PREDICTED\_OPPONENT\_ACTION]</answer>‘   
Strictly follow the output format. Max response length: {MAX\_TOKENS} tokens.