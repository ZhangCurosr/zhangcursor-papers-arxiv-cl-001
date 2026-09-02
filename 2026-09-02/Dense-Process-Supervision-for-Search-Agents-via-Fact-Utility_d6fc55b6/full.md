# Dense Process Supervision for Search Agents via Fact Utility Estimation

Rongzhi Zhu<sup>1,∗</sup> Xiangyu Liu<sup>1,∗</sup> Yi Liu<sup>1</sup> Shuo Zhang<sup>2</sup> Ruirui Zhang<sup>2</sup>

Rui Wu<sup>2</sup> Tao Jiang<sup>2</sup> Zequn Sun<sup>1,†</sup> Wenhao Xu<sup>2</sup> Wei Hu<sup>1,3</sup>

<sup>1</sup> State Key Laboratory for Novel Software Technology, Nanjing University, China <sup>2</sup> Ant Group, China

<sup>3</sup> National Institute of Healthcare Data Science, Nanjing University, China {rzzhu, xyl, yiliu07}.nju@gmail.com, {sunzq, whu}@nju.edu.cn {suzhang.zs, zhangruirui.zrr, guli.wr, tara.jt, hao.xuwh}@antgroup.com

## Abstract

Reinforcement learning (RL) for search agents typically relies on outcome rewards. However, it often fails to achieve effective credit assignment, due to the unclear value of intermediate steps. It is hard to separate their contributions from the final result. In this paper, we propose a dense process supervision method based on fact utility estimation, which models the reasoning process as the accumulation of discrete evidence facts. We first extract structured facts from raw observations and organize them into an explicit fact store. To support credit assign ment, we then cluster semantically equivalent facts and infer the posterior utility of each fact cluster using Bayesian estimation over group rollouts. Finally, we convert the estimated fact utilities into dense step-level rewards to guide RL training. Experiments on seven single-hop and multi-hop QA benchmarks show that our method consistently outperforms existing baselines. Ablation studies validate clear relative improvements on multi-hop QA compared to outcome reward-only training.

## 1 Introduction

Large language models (LLMs) have shown strong capabilities in reasoning and language understanding. However, applying LLMs to complex, realworld scenarios remains a challenge (Mialon et al., 2023; Phan et al., 2025). To bridge this gap, LLM agents have been introduced, which can call external tools (Yao et al., 2022; Schick et al., 2023), such as programming interpreters (Gao et al., 2023), mathematical solvers (Das et al., 2024), and retrieval engines (Zong et al., 2024; Zhang et al., 2025), to solve problems that pure generation cannot handle. Notably, recent proprietary systems like Deep Research (Google, 2024; OpenAI, 2025; Team et al., 2025) have achieved remarkable success in autonomous information gathering. These systems are able to quickly retrieve and analyze a large number of web resources, and generate indepth reports for specific domains within a short time. Such tasks require strong agentic and tool-use abilities like planning and information extraction.

Reinforcement learning (RL) has been proven to be an effective training strategy to enhance tool-use abilities (Jin et al., 2025; Feng et al., 2025). Most existing RL methods for reasoning and searchbased agents rely on outcome-based supervision, where the agent is rewarded solely according to the correctness of the final answer (Chen et al., 2025; Sun et al., 2025). However, this signal is not only sparse and delayed, but also lacks fine granularity. Crucially, the contribution of a specific tool call is often implicit and context-dependent. A correct final answer does not imply that every retrieved evidence was necessary, nor does a failure invalidate all prior reasoning steps. As a result, it is hard to determine which intermediate actions are truly helpful, limiting the training effectiveness.

To address this challenge, we propose a search agent training method, FactAgent, for dense process supervision via fact utility estimation. Our key idea is to represent intermediate evidence as structured facts, estimate their utility from group rollouts, and use these utilities to provide dense steplevel rewards. FactAgent incorporates a fact store to explicitly maintain these structured facts. This design departs from standard ReAct agents (Yao et al., 2022), which rely on raw text history. In addition to the search and answer steps, we integrate an assert step to extract structured facts from retrieved observations and write them into the fact store. We treat these facts as discrete semantic units that influence the trajectory’s success. To enable credit assignment, we propose a Bayesian utility estimation method. We cluster semantically equivalent facts and estimate the posterior utility of each fact cluster with Bayesian estimation from grouplevel rollouts, capturing their statistical association with successful outcomes. This enables more precise credit assignment to intermediate steps that uncover high-value evidence, guiding the agent toward more effective exploration.

Our contributions are summarized as follows:

• FactAgent framework. We introduce FactAgent, a fact-centric search agent that extracts retrieved evidence into structured facts and organizes them in an explicit fact store. It replaces raw interaction history with a compact representation, effectively handling long contexts as interactions grow.

• Bayesian-grounded dense process supervision. Motivated by the view that the utility of an intermediate fact is implicit and only reflected in final outcomes, we estimate the posterior utility of fact clusters from group-level rollout statistics. This yields dense process rewards directly from sparse outcome signals, without requiring external reward models.

• Strong performance. FactAgent consistently outperforms baselines on seven QA benchmarks with Qwen2.5-3B-Instruct and -7B-Instruct backbones. With the 7B backbone, it achieves an average EM of 51.2, surpassing the strongest baseline by +3.2 EM. Ablations and further analyses show that our dense process supervision can particularly improve evidence precision and retrieval quality.

## 2 Related Work

## 2.1 RAG and Search-Based LLM Agents

Although LLMs are capable, their internal knowledge can be incomplete or outdated, which may cause factual errors (Mousavi et al., 2024). Retrieval-augmented generation (RAG) (Lewis et al., 2020) mitigates this issue by grounding generation on an external corpus via retrieval. Early RAG often assumes a largely static knowledge base. Many works improve retrieval quality by introducing graph structures and entity-level matching (Edge et al., 2024; Guo et al., 2024; Zhu et al., 2025). More recently, retrieval is increasingly treated as an on-demand tool that LLMs can call when needed (Zhao et al., 2025b). In this tooluse setting, the LLM is equipped with multiple tools (e.g., web search and browsing) and acts as a search agent (Zong et al., 2024; Li et al., 2025) that iteratively collects and synthesizes evidence, as explored in recent multi-agent frameworks (Zhang et al., 2025; Hong et al., 2023) and deep-research style systems (OpenAI, 2025; Google, 2024; Hu et al., 2025). Several benchmarks evaluate multistep retrieval and reasoning (Mialon et al., 2023; Phan et al., 2025; Wei et al., 2025). These benchmarks also show that planning and optimizing multi-step search remains challenging. Beyond retrieval quality, prior work has also examined whether retrieved evidence appropriately and sufficiently supports the generated answer through fine-grained citation evaluation (Zhao et al., 2024).

## 2.2 RL for Search Agents

Recent advancements leverage RL to evolve LLMs into autonomous search agents capable of interacting with dynamic environments (Zheng et al., 2025; Wu et al., 2025a; Tao et al., 2025; Sun et al., 2025). Many methods jointly optimize reasoning and search. Search-R1 (Jin et al., 2025) and Re-Search (Chen et al., 2025) model search actions as part of the reasoning trajectory and apply group relative policy optimization (GRPO) with sparse outcome rewards. RL is also used to support longhorizon information seeking by improving memory management. MemSearcher (Yuan et al., 2025) and MEM1 (Zhou et al., 2025) maintain a compact memory state that is updated over time, while Memory-R1 (Yan et al., 2025) treats memory operations (e.g., ADD, DELETE) as learnable actions. ReSum (Wu et al., 2025b) extends search beyond the context window by summarizing intermediate states. Recent works have explored process-level supervision to tackle the credit assignment challenge. AutoRefine (Shi et al., 2025) uses retrievalspecific rewards, and GIGPO (Feng et al., 2025) computes step-level advantages by clustering the same intermediate states. In contrast, our method performs clustering over extracted factual units rather than raw agent states, deriving dense intrinsic supervision directly from rollout statistics.

## 3 Preliminaries

Problem formulation. We model the multi-turn reasoning of an LLM agent as a Markov Decision Process (MDP) ⟨S, A, P, R⟩. Given an original question $Q ,$ at step t the agent observes $o _ { t }$ (e.g., retrieved text) and maintains an explicit knowledge state $\mathcal { F } _ { t }$ . We define the state $s _ { t } = ( Q , o _ { t } , \mathcal { F } _ { t } )$ . Unlike other agents that primarily rely on unstructured history as implicit memory, we use $\mathcal { F } _ { t }$ as a compact and interpretable representation of accumulated evidence updated throughout the trajectory.

![](images/d280b031b1f8128a3bdb8dadecd68a556e3ab070bbdf54dd06b0f36c1f5752ef.jpg)  
Figure 1: Framework overview. (a) Group rollouts with Assert actions populate the fact store. (b) Facts are clustered for Bayesian utility estimation. (c) Utilities are converted into dense rewards for fine-grained credit assignment.

Fact store representation. Each entry in $\mathcal { F } _ { t }$ consists of structured facts derived from observations, paired with a short description to summarize the source and reliability. We write e $\in \mathcal { F } _ { t }$ for a semantic fact unit, and denote the newly asserted facts at step t by $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ . As detailed in Section 4, we treat these units not merely as text strings, but as discrete semantic variables that serve as explicit units of evidence extracted from the observation.

Actions and transition. The action space A contains three actions: (i) search, which generates a query and receives a new observation; (ii) assert, which distills the unstructured observation into structured triples and updates the explicit fact store; and (iii) answer, which terminates the episode and outputs the final response. The state transition $\mathcal { P }$ is defined mainly based on the symbolic update of the fact store: after an assert action, the fact store updates as $\mathcal { F } _ { t + 1 } = \mathcal { F } _ { t } \cup \mathcal { K } _ { t }$ , representing the acquisition of new semantic evidence units.

Training. The objective is to maximize expected outcome rewards: $\operatorname* { m a x } _ { \pi } \mathbb { E } _ { \tau \sim \pi } \left[ R _ { \mathrm { o u t c o m e } } ( \tau ) \right]$ . In practice, $R _ { \mathrm { o u t c o m e } }$ is sparse and only observed at the end of trajectories, making direct optimization difficult. As a result, recent search agents adopt GRPO to optimize a surrogate objective defined over groups of sampled trajectories. Specifically, given a group of rollouts $\{ \tau _ { i } \} _ { i = 1 } ^ { G }$ generated under the current policy, GRPO updates the policy by maximizing a relative advantage objective:

$$
\mathcal { L } _ { \mathrm { G R P O } } ( \pi ) = \mathbb { E } _ { q \sim \mathcal { D } , \{ \tau _ { i } \} \sim \pi } \left[ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \tau _ { i } | } \sum _ { t } \frac { \pi \left( a _ { t } ^ { i } | s _ { t } ^ { i } \right) } { \pi _ { \mathrm { o l d } } \left( a _ { t } ^ { i } | s _ { t } ^ { i } \right) } A _ { i } \right] ,\tag{1}
$$

where $\begin{array} { r } { A _ { i } = \frac { R ( \tau _ { i } ) - \mu } { \sigma } } \end{array}$ is the advantage normalized by the group mean $\mu$ and standard deviation σ.

While outcome supervision provides a global signal, it obscures individual contributions, making it difficult to identify necessary steps. To address this limitation, we introduce dense process supervision by constructing informative step-level signals for credit assignment and advantage estimation under GRPO, as described in Section 4.

## 4 Methodology

## 4.1 FactAgent Framework

To support dense supervision and credit assignment, we propose FactAgent. Standard ReAct agents rely on the full interaction history, which leads to rapid context growth and limits reasoning depth. Instead of using such raw text history, we maintain a persistent fact store $\mathcal { F }$ as the working memory. During interaction, the agent sees the original question $Q ,$ the current fact store ${ \mathcal { F } } _ { : }$ and only the most recent observation $o _ { t } .$ . This keeps the prompt length approximately constant across steps while allowing the agent to reason over structured evidence. At each step t, the agent observes the current state and selects one action from the following three types:

1. Search (information seeking). When external information is needed, the agent issues a query. The environment returns an observation $o _ { t }$ . Crucially, we overwrite the previous observation in the prompt with $O t$ . This ensures the input context size remains constant regardless of the trajectory length.

2. Assert (information distillation). Given $o _ { t }$ the agent extracts a small set of structured triples and adds them into the fact store as $\boldsymbol { \mathcal { { K } } } _ { t }$ Each $\textstyle { \boldsymbol { \mathcal { K } } } _ { t }$ also includes a short textual description to verbalize the triples.

3. Answer (termination). Once the information is enough to answer the question, the agent produces the final answer based on $Q$ and ${ \mathcal { F } } _ { : }$ without relying on long raw histories.

From the lens of partial information, $\mathcal { F } _ { t }$ provides an explicit and compact representation of the evidence accumulated so far. We formally prove in Appendix A that the fact store serves as a sufficient statistic of the interaction history, ensuring the Markov property of the fact-centric state transitions and the preservation of policy optimality. This structure not only allows the agent to perform multi-step reasoning efficiently under strict context length constraints, but also enables intermediate supervision on fact store updates.

## 4.2 Dense Process Rewards via Fact Utility Estimation

We propose a strategy for deriving dense processlevel rewards from sparse outcome signals under the GRPO setting. To tackle the credit assignment challenge in complex reasoning, we reframe the problem as a Bayesian estimation task. Specifically, we aggregate semantically related reasoning facts to enable stable advantage estimation. The resulting cluster-level fact utilities are then transformed into dense process rewards via potentialbased shaping, ensuring policy invariance while providing fine-grained supervision.

Reasoning trajectories and semantic facts. Given a query $q ,$ a policy $\pi _ { \theta }$ generates a reasoning trajectory $\tau = ( a _ { 1 } , s _ { 1 } ) , ( a _ { 2 } , s _ { 2 } ) , \ldots , ( a _ { T } , s _ { T } )$ which terminates with a binary outcome reward $R ( \tau ) \in \{ 0 , 1 \}$ indicating the correctness of the final answer. We assume that each trajectory induces a set of discrete semantic facts extracted from the agent’s explicit belief state (fact store): $\mathcal { F } ( \tau ) ~ = ~ \{ f _ { 1 } , f _ { 2 } , . . . , f _ { L } \}$ , where each $f _ { j }$ represents a fact. These facts serve as a structured summary of the agent’s intermediate knowledge.

The optimization objective is to maximize the expected terminal reward $\mathbb { E } _ { \tau \sim \pi _ { \theta } } [ R ( \tau ) ]$ . However, this terminal-only signal provides no direct supervision for intermediate decisions, leading to severe credit assignment challenges. Ideally, one would like to quantify the contribution of each fact $f$ via

$$
v ( f ) \approx \operatorname* { P r } ( R = 1 | f \in { \mathcal { F } } ( \tau ) ) ,\tag{2}
$$

but such estimation is infeasible in practice due to the extreme sparsity and high dimensionality of the space of natural language facts. In typical GRPO group rollouts, most individual facts appear only once or not at all, making fact-level empirical estimates statistically unreliable.

Semantic abstraction of reasoning facts. To address the sparsity issue, we aggregate semantically equivalent facts into clusters. Let ${ \mathcal { C } } =$ $\{ C _ { 1 } , \dots , C _ { M } \}$ denote a partition of the discovered fact space, where each cluster represents a distinct semantic concept (e.g., grouping “A was born in $\mathbf { B } ^ { \ast }$ and $^ { 6 6 } \mathrm { A }$ originates from $\mathbf { B } ^ { \prime \prime } )$ . We employ a two-stage strategy to determine if two facts are semantically equivalent $( f _ { i } \sim f _ { j } )$ : first via coarse-grained embedding similarity, followed by fine-grained logical rule verification which enforces constraints such as negation consistency, exact numeric matching, and relation-level similarity, to prevent spurious merges (detailed in Appendix C). Formally, each cluster $C _ { k }$ is defined as an equivalence class:

$$
C _ { k } = \{ f \in \mathscr { F } ( \tau ) | f \sim f _ { \mathrm { r e p } } \} ,\tag{3}
$$

where $f _ { \mathrm { r e p } } \in C _ { k }$ is a representative fact. Instead of estimating utility for a rare isolated fact $f ,$ we estimate it for the semantic cluster $C _ { k }$ containing $f .$ This shares statistical strength across facts.

Bayesian estimation of cluster-level fact utility. We aim to estimate the utility of each cluster $C _ { k }$ denoted as a latent success probability $\theta _ { k }$ . Intuitively, $\theta _ { k }$ represents the likelihood that a trajectory leads to a correct answer, given that its extracted facts overlap with the semantic concept $C _ { k }$ :

$$
\theta _ { k } = \operatorname* { P r } ( R = 1 | { \mathcal { F } } ( \tau ) \cap C _ { k } \neq \emptyset ) .\tag{4}
$$

However, in GRPO, the group size G is typically small $( \mathbf { e . g . } , G \in [ 4 , 1 6 ] )$ . In such small-sample regimes, simply calculating the empirical win rate yields high-variance estimates, where a single coincidental success could skew the probability to extreme values. To enable robust estimation under this data sparsity, we adopt a Bayesian formulation by placing a symmetric Beta prior on each cluster

$$
\theta _ { k } \sim \mathrm { B e t a } ( \epsilon , \epsilon ) ,\tag{5}
$$

where $\epsilon > 0$ controls prior smoothing.

Given a GRPO group of trajectories, let $N ( C _ { k } )$ denote the number of trajectories where cluster $C _ { k }$ is present, and $S ( C _ { k } )$ the number of those trajectories with $R = 1$ . Under a Bernoulli likelihood for R conditioned on the presence of $C _ { k }$ , the posterior distribution of $\theta _ { k }$ is

$$
\theta _ { k } \mid S ( C _ { k } ) , N ( C _ { k } ) \sim \mathrm { B e t a } ( S ( C _ { k } ) + \epsilon , N ( C _ { k } ) - S ( C _ { k } ) + \epsilon ) .\tag{6}
$$

We use the posterior mean as a stable estimate:

$$
\begin{array} { r } { P ( C _ { k } ) = \mathbb { E } [ \theta _ { k } \vert S ( C _ { k } ) , N ( C _ { k } ) ] = \frac { S ( C _ { k } ) + \epsilon } { N ( C _ { k } ) + 2 \epsilon } . } \end{array}\tag{7}
$$

Relative utility and GRPO advantage approximation. In GRPO, optimization depends on relative performance within a sampled group rather than absolute success probabilities. Let

$$
\bar { P } _ { \mathrm { g r o u p } } = \frac { 1 } { G } \sum _ { i = 1 } ^ { G } R ( \tau _ { i } )\tag{8}
$$

denote the empirical success rate of the current group. We define the relative utility of cluster $C _ { k } \mathrm { : }$

$$
V ( C _ { k } ) = P ( C _ { k } ) - \bar { P } _ { \mathrm { g r o u p } } .\tag{9}
$$

This quantity represents the posterior gain over the group baseline, serving as a cluster-level proxy for the advantage function. A positive value indicates that the presence of $C _ { k }$ is associated with above-average success, while a negative value suggests correlation with failure. Centering by the group mean also stabilizes optimization and aligns naturally with GRPO’s relative policy updates.

Process reward via potential-based shaping. To transform static cluster utilities into dense, stepwise feedback, we adopt Potential-Based Reward Shaping (PBRS), which preserves policy optimality. Let $\mathcal { U } _ { t } = \{ C _ { k } \in \mathcal { C } | C _ { k } \cap \mathcal { F } ( s _ { t } ) \neq \emptyset \}$ denote the set of unique semantic clusters covered by the facts in the fact store at step t. The potential function over partial states is defined as

$$
\Phi ( s _ { t } ) = \operatorname { t a n h } \Big ( \sum _ { C \in \mathcal { U } _ { t } } V ( C ) \Big ) .\tag{10}
$$

Using only unique clusters prevents reward hacking through redundant or paraphrased fact assertions. The tanh nonlinearity bounds the potential, improving training stability.

The raw shape reward at step t is then

$$
r _ { t } ^ { \mathrm { s h a p e } } = \Delta \Phi _ { t } = \Phi ( s _ { t } ) - \Phi ( s _ { t - 1 } ) .\tag{11}
$$

This reward is non-zero only when new semantic information is acquired, providing immediate feedback for informative steps while staying consistent with the original sparse-reward objective.

Credit assignment between search and assert actions. Although potential gains materialize at assert steps, the correctness of an assertion is causally dependent on prior search actions that retrieved relevant evidence. To address this temporal lag, we redistribute the reward. Let t be an assert step and $t ^ { \prime } < t$ be the specific SEARCH step that yielded the asserted fact. Let $\alpha \in [ 0 , 1 ]$ denote the attribution coefficient. The process rewards are assigned as

$$
r _ { t } ^ { \mathrm { p r o c } } = \left( 1 - \alpha \right) r _ { t } ^ { \mathrm { s h a p e } } , r _ { t ^ { \prime } } ^ { \mathrm { p r o c } } = \alpha r _ { t } ^ { \mathrm { s h a p e } } .\tag{12}
$$

This mechanism explicitly encourages the policy to generate high-quality search queries that lead to valuable knowledge acquisition.

Auxiliary process penalties. Finally, we add auxiliary step-level penalties to discourage invalid or degenerate behaviors, including: (1) format violations, (2) repetitive searches, and (3) premature answers with an empty fact store. In the last case, we additionally enforce a hard constraint by setting the terminal reward $R ( \tau ) = 0$ , ensuring that correct answers must be grounded in explicit evidence.

## 4.3 RL Training

We employ GRPO as our training backbone. Unlike standard proximal policy methods that require a separate value network, GRPO estimates the baseline using the group mean, significantly reducing memory overhead. Unlike ReAct agents that optimize over a single growing prompt, our agent operates on step-specific state contexts constructed from the current fact store. Consequently, we perform optimization at the action level: each decision $a _ { i , j }$ is conditioned on a compact state $s _ { i , j }$ and is updated via policy gradients weighted by its specific hybrid advantage $A _ { i , j } ^ { \mathrm { t o t a l } }$

Advantage composition. We first compute the outcome advantage for each trajectory. Let $R _ { \mathrm { o u t c o m e } } ( \tau _ { i } )$ be the reward derived from the final result of trajectory $\tau _ { i } .$ We normalize this reward against the group statistics to obtain the outcome advantage $A _ { i } ^ { \mathrm { o u t c o m e } }$

$$
\begin{array} { r } { A _ { i } ^ { \mathrm { o u t c o m e } } = \frac { R _ { \mathrm { o u t c o m e } } ( \tau _ { i } ) - \mathbf { M e a n } ( \{ R _ { \mathrm { o u t c o m e } } ( \tau _ { k } ) \} _ { k = 1 } ^ { G } ) } { \mathrm { S t d } ( \{ R _ { \mathrm { o u t c o m e } } ( \tau _ { k } ) \} _ { k = 1 } ^ { G } ) + \eta } . } \end{array}\tag{13}
$$

This global signal is then broadcast to all actions within the trajectory. The total advantage for the j-th action in the i-th trajectory, denoted as $A _ { i , j } ^ { \mathrm { t o t a l } }$ is computed by augmenting this global signal with the process advantage from the previous section. Specifically, we directly utilize the redistributed process reward $r _ { i , j } ^ { \mathrm { p r o c } }$ as the process advantage term:

$$
A _ { i , j } ^ { \mathrm { t o t a l } } = A _ { i } ^ { \mathrm { o u t c o m e } } + \Omega \cdot r _ { i , j } ^ { \mathrm { p r o c } } ,\tag{14}
$$

where $\Omega$ is a coefficient balancing the impact of local process supervision against the global objective. We use $\Omega = 0 . 5$ by default and report a sensitivity study in Appendix F. Since $r _ { i , j } ^ { \mathrm { p r o c } }$ represents a difference in value potentials $\left( \Delta \Phi \right) ^ { * }$ , it intrinsically serves as a proxy for the step-wise value improvement, justifying its use as a local advantage signal.

Optimization objective. Since our trajectories consist of multiple discrete steps under iteratively updating fact store contexts, we define the optimization objective by aggregating gradients over all individual actions across the group. Let $n _ { i }$ be the number of steps in trajectory $\tau _ { i } ,$ and $s _ { i , j }$ denote the state at step j (comprising the user query and the current fact store). The policy $\pi _ { \theta }$ is optimized by maximizing the following objective:

$$
\begin{array} { r l r } {  { \mathcal { I } ( \theta ) = \mathbb { E } _ { q \sim \mathcal { D } , \{ \tau _ { i } \} \sim \pi _ { \theta _ { \mathrm { o l d } } } } \Bigg [ \frac { 1 } { \sum _ { i } n _ { i } } \sum _ { i = 1 } ^ { G } \sum _ { j = 1 } ^ { n _ { i } } ( \mathcal { L } _ { i , j } ^ { \mathrm { c l i p } } ( \theta )  } } \\ & { } & {  - \beta \mathbb { D } _ { K L } ( \pi _ { \theta } ( \cdot \vert s _ { i , j } ) \parallel \pi _ { \mathrm { r e f } } ( \cdot \vert s _ { i , j } ) ) ) \Bigg ] , } \end{array}\tag{15}
$$

where $\beta$ controls the KL divergence penalty to prevent the policy from deviating excessively from the reference model $\pi _ { \mathrm { r e f } }$ . The step-wise clipped surrogate loss, denoted by $\mathcal { L } _ { i , j } ^ { \mathrm { c l i p } } ( \theta )$ , is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { i , j } ^ { \mathrm { c l i p } } ( \theta ) = \operatorname* { m i n } \left( \rho _ { i , j } A _ { i , j } ^ { \mathrm { t o t a l } } , \mathrm { c l i p } ( \rho _ { i , j } , 1 - \epsilon _ { \mathrm { c l i p } } , 1 + \epsilon _ { \mathrm { c l i p } } ) A _ { i , j } ^ { \mathrm { t o t a l } } \right) , } \end{array}\tag{16}
$$

where $\begin{array} { r } { \rho _ { i , j } = \frac { \pi _ { \theta } \left( a _ { i , j } \vert s _ { i , j } \right) } { \pi _ { \theta _ { \mathrm { o l d } } } \left( a _ { i , j } \vert s _ { i , j } \right) } } \end{array}$ denotes the importance sampling ratio. This formulation ensures that every reasoning step contributes to the policy update, weighted by its specific hybrid advantage $A _ { i , j } ^ { \mathrm { t o t a l } }$

## 5 Experiment

## 5.1 Experiment Setups

Datasets and metrics. Following Search-R1 (Jin et al., 2025), we evaluate our method on seven QA benchmarks, covering both single-hop and multihop settings. The single-hop QA datasets include Natural Questions (NQ) (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), and PopQA (Mallen et al., 2022). The multi-hop QA datasets include HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (2Wiki) (Ho et al., 2020), MuSiQue (Trivedi et al., 2022), and Bamboogle (Press et al., 2023). More details can refer to Appendix B.2. We report the weighted average EM over all datasets, computed as the total number of exact matches divided by the total number of questions.

Baselines. To isolate the effect of RL optimization, we evaluate our method before RL training and compare our method against two groups of baselines: (1) retrieval-based methods without RL training, such as Naive RAG (Lewis et al., 2020); and (2) retrieval-based methods with RL training, including Search-R1 (Jin et al., 2025), ReSearch (Chen et al., 2025), AutoRefine (Shi et al., 2025), GiGPO (Feng et al., 2025), and $\mathrm { Z e ^ { - } }$ rosearch (Sun et al., 2025). Notably, Zerosearch and ReSearch rely on Google Search, while all other methods (including ours) retrieve from a local Wiki-18 index following Search-R1.

Implementation. We train and evaluate our method on Qwen2.5-7B-Instruct and Qwen2.5-3B-Instruct. We adopt the E5 embedding model (Wang et al., 2022) as the dense retriever. We perform RL training based on verl (Sheng et al., 2025). Following Search-R1, we construct the training set by merging the training splits of NQ and HotpotQA. We set the smoothing parameter $\epsilon = 0 . 5$ , the return coefficient $\alpha = 0 . 2$ , and the aggregation weight $\Omega = 0 . 5$ . The agent is allowed to run at most 8 rounds per question (including the answer action). The outcome reward is defined as the answer-level F1 score. Additional details are in Appendix B.

## 5.2 Main Results

Table 1 reports the EM results on seven QA benchmarks. Overall, our method with RL achieves the best weighted average performance among all baselines on both Qwen2.5-3B-Instruct and Qwen2.5- 7B-Instruct. With the 7B backbone, our method surpasses the strongest baseline by +3.2 EM on average. Notably, the performance gap is more significant on multi-hop datasets (e.g., HotpotQA, 2Wiki) than on single-hop tasks. This confirms that our dense process supervision effectively guides the LLM through complex reasoning chains, where sparse outcome rewards often fail. We observe similar improvements on the 3B backbone, indicating that our method is effective across different model sizes and does not require strong backbone.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="3">Single-Hop QA</td><td colspan="4">Multi-Hop QA</td><td rowspan="2">Avg.</td></tr><tr><td>NQ</td><td>TriviaQA</td><td>PopQA</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td></tr><tr><td rowspan="8">3B</td><td>RAG</td><td>34.8</td><td>54.4</td><td>38.7</td><td>25.5</td><td>22.6</td><td>4.7</td><td>8.0</td><td>34.4</td></tr><tr><td>Search-R1</td><td>34.1</td><td>54.5</td><td>37.8</td><td>32.4</td><td>31.9</td><td>10.3</td><td>26.4</td><td>37.7</td></tr><tr><td>Zerosearch</td><td>41.4</td><td>57.4</td><td>44.8</td><td>27.4</td><td>30.0</td><td>9.8</td><td>11.1</td><td>39.5</td></tr><tr><td>GiGPO</td><td>42.0</td><td>59.5</td><td>42.4</td><td>36.9</td><td>37.0</td><td>12.6</td><td>64.1</td><td>42.7</td></tr><tr><td>ReSearch</td><td>20.4</td><td>33.5</td><td>17.3</td><td>35.6</td><td>39.3</td><td>17.3</td><td>37.6</td><td>29.1</td></tr><tr><td>AutoRefine</td><td>43.6</td><td>59.7</td><td>44.7</td><td>40.4</td><td>38.0</td><td>16.9</td><td>33.6</td><td>44.3</td></tr><tr><td>FactAgent (w/o RL)</td><td>22.1</td><td>41.1</td><td>20.1</td><td>26.1</td><td>23.3</td><td>4.3</td><td>18.4</td><td>25.7</td></tr><tr><td>FactAgent (w/ RL)</td><td>43.9</td><td>60.3</td><td>42.3</td><td>42.6</td><td>43.1</td><td>16.9</td><td>32.8</td><td>45.4</td></tr><tr><td rowspan="7">7B</td><td>RAG</td><td>34.9</td><td>58.5</td><td>39.2</td><td>29.9</td><td>23.5</td><td>5.8</td><td>20.8</td><td>36.4</td></tr><tr><td>Search-R1</td><td>39.3</td><td>61.0</td><td>39.7</td><td>37.0</td><td>41.4</td><td>14.6</td><td>36.8</td><td>43.2</td></tr><tr><td>Zerosearch</td><td>43.6</td><td>65.2</td><td>48.8</td><td>34.6</td><td>35.2</td><td>18.4</td><td>27.8</td><td>45.2</td></tr><tr><td>GiGPO</td><td>46.4</td><td>64.7</td><td>46.1</td><td>41.6</td><td>43.6</td><td>18.9</td><td>68.9</td><td>47.7</td></tr><tr><td>ReSearch</td><td>40.9</td><td>63.7</td><td>44.6</td><td>43.5</td><td>47.6</td><td>22.3</td><td>42.4</td><td>48.0</td></tr><tr><td>FactAgent (w/o RL)</td><td>33.0</td><td>55.1</td><td>36.7</td><td>29.2</td><td>23.8</td><td>9.1</td><td>20.8</td><td>34.9</td></tr><tr><td>FactAgent (w/ RL)</td><td>46.3</td><td>69.4</td><td>46.7</td><td>44.6</td><td>51.5</td><td>19.4</td><td>44.0</td><td>51.2</td></tr></table>

Table 1: Performance comparison on single-hop and multi-hop QA datasets with Qwen2.5-Instruct backbones.

It is also worth noting that two strong baselines, Zerosearch and ReSearch, rely on Google Search to access real-time web information, whereas FactAgent retrieves exclusively from a local Wiki-18 index. Despite operating under this more restricted retrieval setting, our approach still achieves superior overall performance, suggesting that it can more effectively identify and utilize relevant information from limited retrieval results.

We also evaluate our method before RL training to assess the impact of RL optimization. As shown in Table 1, the model without RL performs poorly and even underperforms a simple one-shot RAG baseline, particularly on the 3B backbone. This is expected, as the LLM has not yet adapted to the specialized “Search–Assert–Answer” workflow. After RL training, performance improves substantially and becomes state-of-the-art. This highlights that RL is essential for grounding the agent’s behavior in the proposed fact-centric reasoning process.

Since Qwen2.5-3B-Instruct and Qwen2.5-7B-Instruct exhibit similar trends, we conduct analysis on Qwen2.5-7B-Instruct for clarity and brevity.

<table><tr><td>Variant</td><td>Single-Hop</td><td>Multi-Hop</td><td>Avg.</td></tr><tr><td>FactAgent (w/ RL)</td><td>55.5</td><td>45.8</td><td>51.2</td></tr><tr><td>w/ EM-only clustering</td><td>52.4</td><td>44.4</td><td>48.9</td></tr><tr><td>w/o reward redistribution</td><td>50.6</td><td>38.7</td><td>45.4</td></tr><tr><td>w/o dense process reward</td><td>47.5</td><td>31.4</td><td>40.5</td></tr></table>

Table 2: Ablation study on Qwen2.5-7B-Instruct.

## 5.3 Analysis

Ablation study. We conduct an ablation study to examine the effect of several key design choices, including back-propagation, dense process reward, and EM-only clustering. As shown in Table 2, removing these components consistently hurts performance, especially on multi-hop datasets.

We first consider the effect of fact clustering. When restricting clustering to exact-match facts, performance decreases across all settings, with the average EM dropping from 51.2 to 48.9, indicating the importance of our clustering design across both single-hop and multi-hop tasks. Due to the inherent diversity in model-generated facts, semantically equivalent facts may differ in surface form, making exact-match clustering insufficient for grouping them and leading to incomplete fact aggregation and weaker utility estimation.

Next, we remove reward redistribution and observe a clear performance decrease on. The overall average drops from 51.2 to 45.4, with a larger reduction on multi-hop datasets than on single-hop datasets. This indicates that propagating fact-level utility signals back to earlier retrieval steps is important for learning effective search strategies, especially when multiple evidence must be combined.

![](images/db6cf80cfd0da53ed9428037068d11afb8ab1874ee6a92f67602ddb4f29312a1.jpg)  
(a) Input token composition across turns.

![](images/20de15879d089add563a4405c83ac50fc1bd45cd2a9a4c877cc980f696e80aed.jpg)  
(b) Fact coverage over training.

![](images/13f10664523ed16ebb9f1e2485b5a83c6f3840ebc87850402c44118280114350.jpg)  
(c) Cold-start RL training dynamics.  
Figure 2: Analysis of dense process supervision. (a) Input token composition comparison between ReAct and FactAgent. (b) Fact coverage measured by the recall of reference triples improves over training. (c) Cold-start training curves show faster and more stable learning with dense process supervision.

Finally, we remove dense process-level rewards and train the agent using only the final outcome reward under GRPO. This results in the largest performance degradation among all ablations. The results demonstrate that dense process supervision is a central component of our method. It provides essential credit assignment signals that cannot be recovered from outcome-only rewards, especially under settings with restricted input and retrieval budgets. Without dense process rewards, the agent fails to learn effective intermediate behaviors, particularly for challenging multi-hop queries.

Efficiency analysis. To analyze input efficiency, we evaluate the trained model under the standard ReAct framework and record the average input length at different turns. We compare this with FactAgent under the same maximum number of actions. As shown in Figure 2a, ReAct accumulates the full interaction history in the prompt, leading to an approximately linear growth in input length. At the 8th turn, the input length under ReAct is nearly four times larger than that of FactAgent. In contrast, our method maintains a compact input by storing only extracted evidence in the fact store.

This efficiency offers two key advantages. First, it significantly reduces the computational cost of inference, as the LLM processes far fewer tokens at later steps. Second, it allows the agent to conduct extensive multi-turn searches without quickly reaching the context window limit of the backbone. This demonstrates that our fact store design is highly scalable for complex reasoning tasks that require multiple rounds of information gathering.

Fact coverage over training. To verify whether our method indeed extracts useful facts, we conduct an analysis on the HotpotQA distractor setting. This variant provides supporting contexts relevant to the answer. We use GPT-OSS-120B to extract reference triples from the provided contexts. We then track the recall of reference facts within the agent’s fact store throughout the training process. As shown in Figure 2b, the two methods exhibit distinct learning dynamics. Our method demonstrates a steady improvement in fact coverage, increasing the recall by nearly 40% relative to the initial stage. This indicates that the dense process rewards successfully reinforce the specific action of “asserting” relevant information. In contrast, the outcome-only GRPO baseline fails to show a clear upward trend. This reveals the ambiguity of sparse outcome rewards. Since the reward depends solely on the final answer, the agent gets the same signal whether it actively stores facts or skips the step. Hence, the agent fails to establish a causal link between the assert action and the final success.

## 5.4 Effect of SFT Cold Start

Supervised fine-tuning (SFT) cold start is commonly used before RL to bootstrap basic capabilities and enforce output formats, especially when the action space involves structured tool calls. In our setting, however, we find that SFT cold start does not improve the final performance ceiling. Although it changes early training dynamics, both cold-started and direct RL reach comparable final EM (Figure 2c). Meanwhile, direct RL from the base model can occasionally collapse at certain training steps, after which the LLM outputs garbled and non-executable JSON actions. We never observe this failure when training from the coldstarted policy. Overall, these results suggest that cold start mainly serves as a stabilizer that anchors the policy to valid Search–Assert–Answer formats, rather than improving final task performance. Detailed setup and results are provided in Appendix D.

## 6 Conclusion and Future Work

In this paper, we address the credit assignment challenge for search-based agents by proposing a dense process supervision method. Instead of treating history as unstructured text, our method organizes retrieved evidence into an explicit fact store, clusters semantically equivalent facts via Bayesian estimation, and converts these utilities into steplevel rewards. Experiments show that FactAgent consistently outperforms baselines, with significant gains on multi-hop reasoning tasks. Future work can extend dense process supervision to richer fact representations and continuous utility signals.

## Ethical Considerations

The datasets and LLMs used in this work are public with permissive licenses.

## Limitations

Despite these advancements, our method has several limitations. First, recovering interaction contexts to compute step-level updates increases GPU memory consumption compared to outcome-only baselines. Second, the accuracy of our utility estimation relies on the quality of the clustering algorithm and the sufficiency of GRPO rollouts. Finally, our evaluation is currently confined to static QA datasets; extending our FactAgent to dynamic, open-ended agentic tasks, e.g., GAIA (Mialon et al., 2023), remains a direction for future work.

## Acknowledgments

This work was supported by National Natural Science Foundation of China (Nos. 62676187 and 62406136), Natural Science Foundation of Jiangsu Province (No. BK20241246), and Ant Group.

## References

Mingyang Chen, Linzhuang Sun, Tianpeng Li, Haoze Sun, Yijie Zhou, Chenzheng Zhu, Haofen Wang, Jeff Z Pan, Wen Zhang, Huajun Chen, and 1 others. 2025. Learning to reason with search for LLMs via reinforcement learning. CoRR, abs/2503.19470.

Debrup Das, Debopriyo Banerjee, Somak Aditya, and Ashish Kulkarni. 2024. MATHSENSEI: a toolaugmented large language model for mathematical reasoning. CoRR, abs/2402.17231.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, and

Jonathan Larson. 2024. From local to global: A graph RAG approach to query-focused summarization. CoRR, abs/2404.16130.

Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. 2025. Group-in-group policy optimization for LLM agent training. CoRR, abs/2505.10978.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. PAL: Program-aided language models. In International Conference on Machine Learning, pages 10764–10799.

Google. 2024. Gemini deep research — your personal research assistant. Accessed: 2025-03.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. 2024. LightRAG: Simple and fast retrievalaugmented generation. CoRR, abs/2410.05779.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. CoRR, abs/2011.01060.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, and 1 others. 2023. MetaGPT: Meta programming for a multi-agent collaborative framework. In International Conference on Learning Representations.

Chen Hu, Haikuo Du, Heng Wang, Lin Lin, Mingrui Chen, Peng Liu, Ruihang Miao, Tianchi Yue, Wang You, Wei Ji, and 1 others. 2025. Step-DeepResearch technical report. CoRR, abs/2512.20491.

Bowen Jin, Hansi Zeng, Zhenrui Yue, Jinsung Yoon, Sercan Arik, Dong Wang, Hamed Zamani, and Jiawei Han. 2025. Search-R1: Training LLMs to reason and leverage search engines with reinforcement learning. CoRR, abs/2503.09516.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. CoRR, abs/1705.03551.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. 2024. Vineppo: Refining credit assignment in rl training of llms. arXiv preprint arXiv:2410.01679.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, and 1 others. 2019. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented

generation for knowledge-intensive NLP tasks. Advances in neural information processing systems, 33:9459–9474.

Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou. 2025. Search-o1: Agentic search-enhanced large reasoning models. CoRR, abs/2501.05366.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. 2022. When not to trust language models: Investigating effectiveness and limitations of parametric and nonparametric memories. CoRR, abs/2212.10511.

Grégoire Mialon, Clémentine Fourrier, Thomas Wolf, Yann LeCun, and Thomas Scialom. 2023. GAIA: a benchmark for general ai assistants. In International Conference on Learning Representations.

Seyed Mahed Mousavi, Simone Alghisi, and Giuseppe Riccardi. 2024. Is your LLM outdated? benchmarking LLMs & alignment algorithms for time-sensitive knowledge. CoRR, abs/2404.08700.

OpenAI. 2025. Introducing deep research. Accessed: 2025-02.

Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Chen Bo Calvin Zhang, Mohamed Shaaban, John Ling, Sean Shi, and 1 others. 2025. Humanity’s last exam. CoRR, abs/2501.14249.

Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2023. Measuring and narrowing the compositionality gap in language models. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 5687–5711.

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. ToolFormer: Language models can teach themselves to use tools. Advances in Neural Information Processing Systems, 36:68539–68551.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. HybridFlow: A flexible and efficient RLHF framework. In European Conference on Computer Systems, pages 1279–1297.

Yaorui Shi, Sihang Li, Chang Wu, Zhiyuan Liu, Junfeng Fang, Hengxing Cai, An Zhang, and Xiang Wang. 2025. Search and refine during think: Autonomous retrieval-augmented reasoning of LLMs. CoRR, abs/2505.11277.

Hao Sun, Zile Qiao, Jiayan Guo, Xuanbo Fan, Yingyan Hou, Yong Jiang, Pengjun Xie, Yan Zhang, Fei Huang, and Jingren Zhou. 2025. Zerosearch: Incentivize the search capability of LLMs without searching. CoRR, abs/2505.04588.

Zhengwei Tao, Jialong Wu, Wenbiao Yin, Junkai Zhang, Baixuan Li, Haiyang Shen, Kuan Li, Liwen Zhang, Xinyu Wang, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025. WebShaper: Agentically data synthesizing via information-seeking formalization. CoRR, abs/2507.15061.

Tongyi DeepResearch Team, Baixuan Li, Bo Zhang, Dingchu Zhang, Fei Huang, Guangyu Li, Guoxin Chen, Huifeng Yin, Jialong Wu, Jingren Zhou, and 1 others. 2025. Tongyi deepresearch technical report. CoRR, abs/2510.24701.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly-supervised contrastive pre-training. CoRR, abs/2212.03533.

Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. 2025. BrowseComp: A simple yet challenging benchmark for browsing agents. CoRR, abs/2504.12516.

Jialong Wu, Baixuan Li, Runnan Fang, Wenbiao Yin, Liwen Zhang, Zhengwei Tao, Dingchu Zhang, Zekun Xi, Yong Jiang, Pengjun Xie, Fei Huang, and Jingren Zhou. 2025a. WebDancer: Towards autonomous information seeking agency. CoRR, abs/2505.22648.

Xixi Wu, Kuan Li, Yida Zhao, Liwen Zhang, Litu Ou, Huifeng Yin, Zhongwang Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Minhao Cheng, Shuai Wang, Hong Cheng, and Jingren Zhou. 2025b. Re-Sum: Unlocking long-horizon search intelligence via context summarization. CoRR, abs/2509.13313.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Hinrich Schütze, Volker Tresp, and Yunpu Ma. 2025. Memory-R1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. CoRR, abs/2508.19828.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing, pages 2369–2380.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. 2022. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations.

Qianhao Yuan, Jie Lou, Zichao Li, Jiawei Chen, Yaojie Lu, Hongyu Lin, Le Sun, Debing Zhang, and Xianpei Han. 2025. MemSearcher: Training LLMs to reason, search and manage memory via end-to-end reinforcement learning. CoRR, abs/2511.02805.

Wentao Zhang, Ce Cui, Yilei Zhao, Yang Liu, and Bo An. 2025. AgentOrchestra: A hierarchical multiagent framework for general-purpose task solving. CoRR, abs/2506.12508.

Suifeng Zhao, Tong Zhou, Zhuoran Jin, Hongbang Yuan, Yubo Chen, Kang Liu, and Sujian Li. 2024. Awecita: Generating answer with appropriate and well-grained citations using llms. Data Intelligence, 6(4):1134– 1157.

Yida Zhao, Kuan Li, Xixi Wu, Liwen Zhang, Dingchu Zhang, Baixuan Li, Maojia Song, Zhuo Chen, Chenxi Wang, Xinyu Wang, and 1 others. 2025a. Repurposing synthetic data for fine-grained search agent supervision. arXiv preprint arXiv:2510.24694.

Zhejun Zhao, Yuehu Dong, Alley Liu, Lixue Zheng, Pingsheng Liu, Dongdong Shen, Long Xia, Jiashu Zhao, and Dawei Yin. 2025b. TURA: Toolaugmented unified retrieval agent for AI search. CoRR, abs/2508.04604.

Yuxiang Zheng, Dayuan Fu, Xiangkun Hu, Xiaojie Cai, Lyumanshan Ye, Pengrui Lu, and Pengfei Liu. 2025. DeepResearcher: Scaling deep research via reinforcement learning in real-world environments. CoRR, abs/2504.03160.

Zijian Zhou, Ao Qu, Zhaoxuan Wu, Sunghwan Kim, Alok Prakash, Daniela Rus, Jinhua Zhao, Bryan Kian Hsiang Low, and Paul Pu Liang. 2025. MEM1: Learning to synergize memory and reasoning for efficient long-horizon agents. CoRR, abs/2506.15841.

Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, and Wei Hu. 2025. Mitigating lost-in-retrieval problems in retrieval augmented multi-hop question answering. CoRR, abs/2502.14245.

Chang Zong, Yuchen Yan, Weiming Lu, Jian Shao, Yongfeng Huang, Heng Chang, and Yueting Zhuang. 2024. Triad: A framework leveraging a multi-role LLM-based agent to solve knowledge base question answering. In Conference on Empirical Methods in Natural Language Processing, pages 1698–1710.

## A Theoretical Analysis

In this section, we provide a theoretical justification for replacing the raw interaction history with the structured fact store. We distinguish between the environmentfeedback (search results) and the agent state used for decision-making. We show that under the assumption of semantic sufficiency, the fact store constitutes a valid state abstraction of the full history, preserving the Markov property and the optimality of policies.

## A.1 Problem Definitions

Let the search task be modeled as a sequential decision process over $T$ steps.

• Action $( a _ { t } ) \colon$ The command issued by the agent after observing $o _ { t - 1 }$ , either a Search action, an Assert action, or an Answer action that terminates the episode. We set $o _ { 0 } = \emptyset$

• Environment feedback $\mathbf { \Pi } ( o _ { t } ) \colon$ : The raw textual output returned by the search engine/environment in response to the most recent Search action $( \mathrm { e . g . }$ , search snippets). In the context of search agents, this is often termed an “observation”, but structurally it serves as input to the state transition. $\operatorname { I f } a _ { t }$ is not Search, no new feedback is generated and we set $o _ { t } = o _ { t - 1 }$

• History $( h _ { t } ) \colon$ The complete sequence of interactions up to time t:

$$
h _ { t } = ( Q , o _ { 0 } , a _ { 1 } , o _ { 1 } , a _ { 2 } , o _ { 2 } , \ldots , a _ { t } , o _ { t } ) .\tag{17}
$$

The baseline is the history-based MDP. Standard LLM agents typically treat the full history $h _ { t }$ as the state $s _ { t } ^ { \mathrm { h i s t } }$ . Since $h _ { t }$ contains all historical information available to the agent, the process defined over $h _ { t }$ satisfies the Markov property: $\mathrm { P r } ( h _ { t + 1 } | h _ { 0 : t } , a ) = \mathrm { P r } ( h _ { t + 1 } | h _ { t } , a )$

## A.2 Proposed: The FactAgent MDP

We propose a state representation that discards raw history in favor of a structured fact store. Let the proposed state be:

$$
s _ { t } ^ { \mathrm { f a c t } } = ( Q , o _ { t } , { \mathcal F } _ { t } ) ,\tag{18}
$$

where $\mathcal { F } _ { t }$ is the set of accumulated facts. The state transition is governed by a deterministic update function $\tau$

$$
\mathcal { F } _ { t + 1 } = \mathcal { T } ( \mathcal { F } _ { t } , o _ { t } , a _ { t + 1 } ) ,\tag{19}
$$

where $a _ { t + 1 }$ is chosen after observing $o _ { t } .$ . In particular, ${ \mathcal { T } } ( { \mathcal { F } } , o , a ) = { \mathcal { F } }$ if a is not Assert.

## A.3 Equivalence Analysis

To prove that the FactAgent is not an approximation but an equivalent formulation, we introduce the Semantic Sufficiency Assumption.

Assumption 1 (semantic sufficiency). The fact store $\mathcal { F } _ { t }$ is a sufficient statistic of the history $h _ { t }$ for predictingfuture environmentfeedback andfinal rewards. Formally, let Φ be the mapping such that $s _ { t } ^ { f a c t } = \Phi ( h _ { t } )$ . For any two histories $h _ { t } , h _ { t } ^ { \prime }$ such that $\Phi ( h _ { t } ) = \Phi ( h _ { t } ^ { \prime } )$ , we assume:

$$
\begin{array} { r l } & { \operatorname* { P r } ( o _ { t + 1 } \mid h _ { t } , a _ { t + 1 } ) = \operatorname* { P r } ( o _ { t + 1 } \mid \Phi ( h _ { t } ) , a _ { t + 1 } ) , } \\ & { \quad \operatorname* { P r } ( R \mid h _ { t } , a _ { t + 1 } ) = \operatorname* { P r } ( R \mid \Phi ( h _ { t } ) , a _ { t + 1 } ) . } \end{array}\tag{20}
$$

For non-Search actions, the feedback is deterministic $( { \bf e . g . } , o _ { t + 1 } = o _ { t } )$ , so the first condition holds trivially.

Theorem 1 (Markov property of fact state). Under Assumption 1, the stochastic process defined over the fact-centric states $s _ { t } ^ { \mathrm { f a c t } }$ is a MDP.

Proof. We examine the transition probability for the fact-centric state. Given $s _ { t } ^ { \mathrm { f a c t } } = ( Q , o _ { t } , \mathcal { F } _ { t } )$ and action $a _ { t + 1 }$ , the next feedback $o _ { t + 1 }$ is generated by the environment, and the fact store is updated deterministically via $\tau$

$$
\begin{array} { r l } & { \mathrm { P r } ( s _ { t + 1 } ^ { \mathrm { f a c t } } \mid s _ { 0 : t } ^ { \mathrm { f a c t } } , a _ { 1 : t + 1 } ) = \mathrm { P r } ( \mathcal { F } _ { t + 1 } , o _ { t + 1 } \mid s _ { 0 : t } ^ { \mathrm { f a c t } } , a _ { 1 : t + 1 } ) } \\ & { = \mathrm { P r } ( \mathcal { F } _ { t + 1 } \mid o _ { t + 1 } , s _ { 0 : t } ^ { \mathrm { f a c t } } , a _ { 1 : t + 1 } ) \cdot \mathrm { P r } ( o _ { t + 1 } \mid s _ { 0 : t } ^ { \mathrm { f a c t } } , a _ { 1 : t + 1 } ) } \\ & { \overset { ( i ) } { = } \Im [ \mathcal { F } _ { t + 1 } = \mathcal { T } ( \mathcal { F } _ { t } , o _ { t } , a _ { t + 1 } ) ] \cdot \mathrm { P r } ( o _ { t + 1 } \mid s _ { 0 : t } ^ { \mathrm { f a c t } } , a _ { 1 : t + 1 } ) } \\ & { \overset { ( i i ) } { = } \Im [ \mathcal { F } _ { t + 1 } = \mathcal { T } ( \mathcal { F } _ { t } , o _ { t } , a _ { t + 1 } ) ] \cdot \mathrm { P r } ( o _ { t + 1 } \mid s _ { t } ^ { \mathrm { f a c t } } , a _ { t + 1 } ) } \\ & { = \mathrm { P r } ( s _ { t + 1 } ^ { \mathrm { f a c t } } \mid s _ { t } ^ { \mathrm { f a c t } } , a _ { t + 1 } ) . } \end{array}\tag{21}
$$

Step (i) follows from the determinism of $\tau$ Step (ii) follows from Assumption 1, which implies that $o _ { t + 1 }$ depends on the past only through $s _ { t } ^ { \mathrm { f a c t } } = \Phi ( h _ { t } )$ and the current action. □

Theorem 2 (optimal policy equivalence). Under Assumption 1, for any history $h ,$ the optimal value is preserved under the mapping Φ: $V _ { \mathrm { h i s t } } ^ { * } ( h ) ~ =$ $V _ { \mathrm { f a c t } } ^ { * } ( \Phi ( h ) )$ .

Proof. Consider the Bellman optimality equation for the history-based agent:

$$
V _ { \mathrm { h i s t } } ^ { * } ( h _ { t } ) = \operatorname* { m a x } _ { a } \mathbb { E } [ r ( h _ { t } , a ) + \gamma V _ { \mathrm { h i s t } } ^ { * } ( h _ { t + 1 } ) \vert h _ { t } , a ] .\tag{22}
$$

Under Assumption 1, for any two histories $h _ { t } , h _ { t } ^ { \prime }$ such that $\Phi ( h _ { t } ) = \Phi ( h _ { t } ^ { \prime } )$ , both the environment feedback distribution and the expected reward are identical:

$$
\begin{array} { r l } & { \operatorname* { P r } ( o _ { t + 1 } \mid h _ { t } , a ) = \operatorname* { P r } ( o _ { t + 1 } \mid \Phi ( h _ { t } ) , a ) , } \\ & { r ( h _ { t } , a ) : = \operatorname { \mathbb { E } } [ R \mid h _ { t } , a ] = \operatorname { \mathbb { E } } [ R \mid \Phi ( h _ { t } ) , a ] = : \bar { r } ( \Phi ( h _ { t } ) , a ) . } \end{array}\tag{23}
$$

Therefore, the induced transition and reward model over the compressed state $s _ { t } ^ { \mathrm { f a c t } } = \Phi ( h _ { t } )$ is well-defined, and the optimal control problem can be equivalently written on this state space. In particular, restricting attention to policies that depend only on $\Phi ( h )$ is without loss of optimality, yielding $V _ { \mathrm { h i s t } } ^ { * } ( h ) = V _ { \mathrm { f a c t } } ^ { * } ( \Phi ( h ) )$ . □

This analysis demonstrates that our method is theoretically grounded. By treating the search results $O t$ as environment feedback driving the transitions of a fact-based state $s _ { t } ^ { \mathrm { f a c t } } = ( Q , o _ { t } , \mathcal { F } _ { t } )$ , we construct a valid MDP that allows for dense process supervision without violating the fundamental assumptions of reinforcement learning.

## B Experiment Setup and Implementation

## B.1 RL Training Setup

All training runs are conducted on 8× NVIDIA A100 GPUs, while evaluation is performed on 4× NVIDIA A100 GPUs. We implement RL training based on the verl framework. Unless otherwise specified, we use the same hyperparameter configuration across all datasets and model scales. The RL hyperparameters are summarized in Table 3.

## B.2 Dataset Details

We evaluate our search agent on seven QA benchmarks covering both single- and multi-hop settings. The detailed statistics of the evaluation splits in our experiments are summarized in Table 4.

## C Clustering Algorithm

In this section, we describe our clustering design. The key step is to decide whether two triples have the same meaning.

## C.1 Two-Stage Fact Aggregation Strategy

We use a simple two-stage test. First, we compute the cosine similarity between the embeddings of two triples, and only pairs above a similarity threshold are passed to the next stage. Second, we apply conservative rule filtering, where all of the following conditions must be satisfied. We require negation consistency: if one relation is negated but the other is not, we never merge them. We also enforce numeric consistency: if either triple contains numbers, then both must contain numbers and the numeric parts must match exactly, since embeddings can be unreliable for fine-grained numeric differences. Additionally, we require the relation phrases themselves to be sufficiently similar, measured by either surface-form matching or by relation-embedding similarity above a threshold.

<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>0.00001</td><td>Training batch size</td><td>128</td><td>Global steps</td><td>500</td></tr><tr><td>Number of training epochs</td><td>1</td><td>Number of rollouts</td><td>6</td><td>Rollout temperature</td><td>0.7</td></tr><tr><td>KL loss coefficient</td><td>0.001</td><td>Clip ratio</td><td>0.2</td><td>Clustering similarity threshold</td><td>0.95</td></tr><tr><td>Parse error penalty</td><td>-0.1</td><td>Duplicate search penalty</td><td>-0.01</td><td></td><td></td></tr></table>

Table 3: Hyperparameters setting of our RL training.
<table><tr><td>Dataset</td><td>NQ</td><td>TriviaQA</td><td>PopQA</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td></tr><tr><td>Evaluation set size</td><td>3,610</td><td>11,313</td><td>14,267</td><td>7,405</td><td>12,576</td><td>2,417</td><td>125</td></tr></table>

Table 4: The statistics of seven QA datasets.

To handle possible subject-object inversion (e.g., “A founded B” vs. “B was founded by A”), we test both configurations: the original triple pair and the pair with one triple’s subject and object swapped. If either configuration passes all the above filtering rules, we treat the two triples as semantically equivalent and merge them.

## C.2 Calibration of Similarity Thresholds and Filtering Rules

To ensure the robustness of our clustering algorithm, we conducted a rigorous calibration study. We utilized GPT-5.2 to generate positive and negative examples across 19 distinct scenarios, covering a wide range of linguistic variations including paraphrasing, negation, numeric alteration, and structural inversion. Our analysis of the embedding similarity distribution (using the E5-base model) revealed the following patterns:

1. High similarity for semantic equivalence: When two triples were semantically identical (e.g., simple paraphrases or formatting differences), their cosine similarity consistently exceeded 0.95.

2. False positives in specific error cases: Despite the high discrimination power for general cases, we identified three specific categories where the embedding model assigned high similarity scores (> 0.95) to semantically distinct triples:

• Affirmation vs. Negation: The model often failed to distinguish between a statement and its negation (e.g., “A is B” vs. “A is not B”).

• Numeric differences: Slight variations in numbers (e.g., dates or quantities) resulted in indistinguishably high similarity scores, despite representing factual contradictions.

• Relation ambiguity: When the subject and object were identical but the relation changed slightly (e.g., “investor in” vs. “CEO of”), embeddings alone were insufficient to separate the meanings reliably.

Based on these findings, we established a “highrecall, strict-filtering” strategy. We set a lenient similarity threshold $( \sigma _ { s i m } = 0 . 9 5 )$ to capture diverse paraphrases, while implementing deterministic rules to filter out the false positives identified above (Negation Consistency, Numeric Consistency, and Relation-Specific checks).

Furthermore, to address the issue of Subject-Object Inversion, we introduced a bidirectional test. Since rigid subject/object matching rules might incorrectly exclude these semantically equivalent but structurally inverted pairs, we explicitly test both the original configuration and the swapped configuration. If either configuration passes the similarity threshold and the consistency rules, the triples are merged.

## D Complete Results for Ablation Study

In this section, we provide detailed results for the ablation study based on Qwen2.5-7B-Instruct. As shown in Table 5, we report per-dataset EM scores across seven QA benchmarks.

## E Cold-Start Analysis

We implement cold start with a lightweight SFT stage to teach the Search–Assert–Answer workflow. Specifically, we construct 600 trajectories via rejection sampling by rolling out the base model and filtering samples that are answer-correct, contain valid assert actions, and strictly follow the JSON schema. We implement SFT using the verl framework, and summarize the SFT hyperparameters in Table 6. Complete results are reported in Table 7.

<table><tr><td>Variant</td><td>NQ</td><td>TriviaQA</td><td>PopQA</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td><td>Avg.</td></tr><tr><td>w/ EM-only clustering</td><td>43.5</td><td>65.8</td><td>44.0</td><td>45.2</td><td>48.8</td><td>19.1</td><td>42.4</td><td>48.9</td></tr><tr><td>w/o reward redistribution</td><td>42.1</td><td>63.9</td><td>42.1</td><td>38.3</td><td>43.7</td><td>14.0</td><td>38.4</td><td>45.4</td></tr><tr><td>w/o process reward</td><td>37.1</td><td>58.8</td><td>41.2</td><td>37.7</td><td>31.6</td><td>10.6</td><td>36.8</td><td>40.5</td></tr></table>

Table 5: Detailed ablation results of Qwen2.5-7B-Instruct on seven QA benchmarks (EM).

<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Learning rate</td><td>0.00002</td><td>Global batch size</td><td>64</td></tr><tr><td>Warmup ratio</td><td>0.05</td><td>Total epochs</td><td>2</td></tr></table>

Table 6: SFT cold-start training hyperparameters.

From Table 7, we observe that the SFT cold-start checkpoint does not consistently improve the initial performance of the agent. In fact, after SFT, the model exhibits a lower average EM compared to the naive initialization. Moreover, after subsequent RL training, the SFT-initialized agent still underperforms the counterpart trained from a naive initialization with the same RL procedure. These results suggest that, in our setting, the primary benefit of SFT cold-start lies in improving training stability and reducing early-stage degeneration, rather than increasing the final performance upper bound.

## F Hyperparameter Sensitivity Analysis

This section studies the sensitivity of our method to the aggregation weight Ω, with results summarized in Table 8. We observe that the final performance is sensitive to the choice of Ω. When Ω is relatively small, incorporating process-level rewards already yields substantial improvements over outcome-only supervision; for example, setting Ω = 0.3 improves the average EM by 7.3 points compared to pure outcome-based training. However, as Ω becomes too large, the dominance of process rewards can negatively affect overall performance, leading to performance degradation. Based on this analysis, we use Ω = 0.5 as the default setting in all experiments.

## G Training Efficiency and Computational Cost

To improve transparency regarding efficiency and computational overhead, we report additional latency and training statistics. During inference, evaluated under concurrency 128 on 4×A100 GPUs,

FactAgent requires an average latency of 21.1 seconds per question and performs 3.8 reasoning steps on average. During RL training, we observe that the average number of interaction steps naturally decreases as the policy improves, dropping from approximately 4.3 steps at initialization to 3.2 steps near convergence. We further study the cost and performance tradeoff under different rollout group sizes in Fig. 9.

Increasing rollout count from 4 to 6 substantially improves performance, while increasing further to 8 only yields marginal gains. To isolate FactAgent’s additional overhead, we further measure fact collection and clustering cost. The additional clustering stage requires only 2.75 seconds on average per training step. Under rollout\_n=6, average training step time remains close to standard GRPO (231.1s vs. 225.2s). These results suggest that FactAgent introduces only minor computational overhead while providing substantial performance gains.

## H Utility Distribution Analysis

To better understand the structure of utility estimation, we rerun NQ, HotpotQA, and TriviaQA using multiple sampled trajectories and analyze fact-cluster distributions.

For each question, asserted facts are clustered and normalized into probability masses. Across all datasets, utility distributions remain highly concentrated rather than uniformly sparse. Table 10 reports cumulative cluster mass statistics.

The first three clusters already explain more than 93% of total utility mass across all datasets, suggesting that successful reasoning trajectories rely primarily on a small number of highly informative evidence facts. We further analyze question difficulty by grouping questions according to empirical success rates.

Hard questions require slightly more evidence aggregation, but utility distributions remain strongly concentrated. These findings empirically support our design assumption that successful search trajectories depend on a small number of informative evidence facts rather than exponentially growing intermediate states.

<table><tr><td>Method</td><td>NQ</td><td>TriviaQA</td><td>PopQA</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td><td>Avg.</td></tr><tr><td>SFT (cold-start)</td><td>32.9</td><td>50.9</td><td>35.1</td><td>26.0</td><td>20.6</td><td>8.6</td><td>24.8</td><td>32.3</td></tr><tr><td>SFT + RL</td><td>44.8</td><td>66.0</td><td>46.1</td><td>46.2</td><td>49.0</td><td>19.6</td><td>40.0</td><td>49.5</td></tr></table>

Table 7: Cold-start results on seven QA benchmarks (EM) for Qwen2.5-7B-Instruct.
<table><tr><td>Ω</td><td>NQ</td><td>TriviaQA</td><td>PopQA</td><td>HotpotQA</td><td>2Wiki</td><td>MuSiQue</td><td>Bamboogle</td><td>Avg.</td></tr><tr><td>0</td><td>37.1</td><td>58.8</td><td>41.2</td><td>37.7</td><td>31.6</td><td>10.6</td><td>36.8</td><td>40.5</td></tr><tr><td>0.3</td><td>44.5</td><td>66.6</td><td>42.8</td><td>42.2</td><td>47.5</td><td>16.8</td><td>40.0</td><td>47.8</td></tr><tr><td>0.5</td><td>46.3</td><td>69.4</td><td>46.7</td><td>44.6</td><td>51.5</td><td>19.4</td><td>44.0</td><td>51.2</td></tr><tr><td>0.7</td><td>42.6</td><td>62.5</td><td>42.2</td><td>40.9</td><td>42.7</td><td>12.5</td><td>36.8</td><td>45.2</td></tr></table>

Table 8: Sensitivity analysis of the aggregation weight Ω.

<table><tr><td>Rollouts (n)</td><td>Time / Step (s)</td><td>Total Time (h)</td><td>EM</td></tr><tr><td>4</td><td>145.5</td><td>20.2</td><td>48.5</td></tr><tr><td>6 (Default)</td><td>231.1</td><td>32.2</td><td>51.2</td></tr><tr><td>8</td><td>281.3</td><td>39.1</td><td>51.4</td></tr></table>

Table 9: Cost–quality tradeoff under different rollout group sizes.
<table><tr><td>Dataset</td><td>Top-1</td><td>Top-2</td><td>Top-3</td></tr><tr><td>NQ</td><td>0.450</td><td>0.879</td><td>0.973</td></tr><tr><td>HotpotQA</td><td>0.414</td><td>0.822</td><td>0.936</td></tr><tr><td>TriviaQA</td><td>0.438</td><td>0.858</td><td>0.968</td></tr></table>

Table 10: Cumulative probability mass captured by topranked fact clusters.

## I Comparison with Related Process Supervision Methods

We compare FactAgent against representative process-supervision approaches.

In Table 12, compared with E-GRPO (Zhao et al., 2025a), which provides trajectory-level entity supervision, FactAgent performs step-level utility estimation without requiring additional annotation. Compared with VinePPO (Kazemnejad et al., 2024), FactAgent avoids expensive perstep branching and Monte Carlo rollout expansion, while achieving stronger performance across benchmarks.

## J System Prompts

Table 13 presents the full system prompt used for the search agent. We strictly enforce the output format to ensure the executability of the generated actions.

<table><tr><td>Dataset</td><td>Easy</td><td>Medium</td><td>Hard</td></tr><tr><td>NQ</td><td>2.32</td><td>2.43</td><td>2.54</td></tr><tr><td>HotpotQA</td><td>2.43</td><td>2.51</td><td>2.70</td></tr><tr><td>TriviaQA</td><td>2.45</td><td>2.55</td><td>2.66</td></tr></table>

Table 11: Average number of unique fact clusters across difficulty groups.
<table><tr><td>Method</td><td>E-GRPO</td><td>VinePPO</td><td>FactAgent</td></tr><tr><td>NQ</td><td>42.6</td><td>43.8</td><td>46.3</td></tr><tr><td>TriviaQA</td><td>62.8</td><td>62.1</td><td>69.4</td></tr><tr><td>PopQA</td><td>44.6</td><td>43.4</td><td>46.7</td></tr><tr><td>HotpotQA</td><td>35.1</td><td>34.4</td><td>44.6</td></tr><tr><td>2Wiki</td><td>44.2</td><td>47.1</td><td>51.5</td></tr><tr><td>MuSiQue</td><td>18.9</td><td>15.9</td><td>19.4</td></tr><tr><td>Bamboogle</td><td>40.8</td><td>40.0</td><td>44.0</td></tr><tr><td>Avg. EM</td><td>45.8</td><td>45.9</td><td>51.2</td></tr></table>

Table 12: Comparison with representative process supervision approaches.

## K AI Assistance Statement

We used ChatGPT by OpenAI<sup>1</sup> and Gemini<sup>2</sup> for linguistic polishing and stylistic editing of the draft to enhance clarity and readability.

![](images/ebd4715c86b901ecd600b2d0bd1eafb119f08dfb73cc95a9592751e6ea15e537.jpg)  
Table 13: The system prompt used for agents.