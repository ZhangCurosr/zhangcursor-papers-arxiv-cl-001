# Mosaic: Multi-Session Agentic Reinforcement Learning

Yilong Xu<sup>1</sup>, Minrui Xu<sup>1</sup>, Xinyu Zhou<sup>1</sup>, Xiang Long, Haochen Tan<sup>2,3†‡</sup> and Zhijiang Guo<sup>1,4‡</sup> <sup>1</sup>LARK, HKUST(GZ), <sup>2</sup>Foundation Model Lab, Huawei, <sup>3</sup>City University of Hong Kong, <sup>4</sup>HKUST

<sup>†</sup>Project Leader <sup>‡</sup>Corresponding Author

Github Page: https://lark-ai-lab.github.io/Mosaic/

Abstract

Most existing agentic reinforcement learning methods assume single-session trajectories, where interaction proceeds within one continuously accumulated context. However, agentic techniques such as context compaction and subagent forking may produce trajectories spanning multiple sessions, creating a mismatch between the single-session training assumption and inference-time execution. Therefore, we introduce Multi-Session Agentic Reinforcement Learning, a more general setting that supports an arbitrary number of sessions within a trajectory. Under this setting, a widely adopted method is to broadcast trajectory-level advantages to all sessions, but this does not distinguish contributions from various sessions or guide session allocation within trajectories. To address this issue, we propose Mosaic, a multi-session agentic RL method with critic-free hierarchical optimization. At the trajectory level, it incorporates session-usage signals into advantage estimation. At the session level, it shapes each session’s advantage according to the estimated marginal contribution. Experiments on BrowseComp with Qwen3.5-4B show that Mosaic outperforms broadcasting in both context compaction and subagent forking scenarios while using fewer sessions on average. Further analyses show that our method achieves improvements in session utility and allocation, which we suggested is a promising target for future advances in multi-session agentic RL.

## 1. Introduction

Agentic Reinforcement Learning (Agentic RL; Gao et al., 2025; Jin et al., 2025; Zhang et al., 2025) trains Large Language Model (LLM) agents (Shinn et al., 2023; Yao et al., 2022) for multi-turn reasoning, tool use, and interaction with external environments via online rollouts and reward feedback, yielding broad performance gains. More recently, LLM agents have incorporated advanced inference techniques, such as context management (Mei et al., 2025; Zhang et al., 2026a) and harness engineering (Lin et al., 2026; Zhou et al., 2026), enabling agentic workflows to tackle increasingly complex and long-horizon tasks. However, most agentic RL methods assume a single-session setting, where each rollout unfolds as a coherent multi-turn trajectory within a continuously accumulated context and yields a single training segment (Wang et al., 2026; Xu et al., 2026b).

We argue that the single-session setting is not suficiently general for optimizing LLM agents. For example, context compaction compresses the execution history to free up context space while rewriting the prefix that conditions subsequent actions (Lu et al., 2025; Nguyen et al., 2025; Wu et al., 2025), thereby initiating a new session. Similarly, subagent forking delegates global goal to subagents which creates branch sessions (Song et al., 2025; Team et al., 2026; Xu et al., 2026d). Those resulting trajectories no longer satisfy the single-session assumption, creating a mismatch between training and inference that compromises consistency and limits optimization potential.

![](images/5faffe2eec2448d9667c68134ba1e0e34b08636a567acb87762136045466fbae.jpg)  
Figure 1 | The comparison between multi-session and single-session trajectories.

To address this gap, we introduce a new setting termed Multi-Session Agentic Reinforcement Learning, which supports an arbitrary count of sessions and enables agents with diverse inference mechanisms to be optimized within a unified algorithm. A straightforward way to handle multi-session rollouts is through broadcasting, whereby a trajectory-level outcome reward or advantage is replicated to every token across all sessions (Accio-Lab, 2026; Xu et al., 2026a). However, we argue that this method provides only coarse supervision and ofers no explicit control over optimization across sessions, resulting in poorly targeted learning signals as trajectories span more sessions.

In this work, we propose Mosaic, a multi-session agentic RL method with critic-free hierarchical optimization. First, session-aware trajectory-level advantage estimation calibrates outcome-based advantages using within-group rankings of session counts conditioned on task outcomes, encouraging successful trajectories to solve tasks eficiently while allowing unsuccessful trajectories suficient exploration. Its bounded modulation preserves outcome-induced ordering while incorporating sessionusage guidance, thereby widening relative advantage margins for homogeneous-reward groups. Second, session-level advantage shaping calculates each session’s marginal outcome gain as an immediate reward, applies group-relative normalization separately by session type, and combines the resulting shaping signal with the trajectory-level advantage. Overall, the former provides a long-term, outcome-oriented global signal, while the latter captures short-term local marginal efects. They together form a hierarchical optimization target for multi-session agentic RL.

To evaluate our method, we conduct experiments on BrowseComp (Wei et al., 2025) and select Qwen3.5-4B (Qwen Team, 2026) as the base model. Our Mosaic outperforms multiple single-session and multi-session agentic RL baselines under both context compaction and subagent forking scenarios. Moreover, Mosaic achieves up to a 10.1% relative improvement in accuracy over the broadcasting baseline while using fewer sessions per trajectory on average (3.17 → 2.35). Ablation studies further demonstrate that session-aware trajectory-level advantage estimation and session-level advantage shaping each contribute to the final performance gains. Our analyses also reveal several key insights: (1) shaping trajectory-level advantages with session counts increases within-group advantage variance and improves sample eficiency; (2) diferent session types exhibit distinct distributions of session-level marginal gains, which our type-conditioned normalization helps stabilize; (3) efective multi-session optimization improves session utility and allocation, a pattern that Mosaic strengthens and that might ofer a potential direction for future work.

## 2. Preliminaries

## 2.1. Agentic RL with Group-Relative Policy Optimization

Given a task $x \sim \mathcal { D } _ { : }$ , an agent policy $\pi _ { \theta }$ generates a rollout trajectory � through multi-turn reasoning, tool use, and interaction with the environment. Formally, we represent the trajectory as:

$$
\tau = ( a _ { 1 } , o _ { 1 } , \ldots , a _ { T } , o _ { T } ) , \qquad a _ { t } \sim \pi _ { \theta } ( \cdot \mid h _ { t } ) ,\tag{1}
$$

where $a _ { t }$ denotes the agent action at turn $t , \ o _ { t }$ is the corresponding observation returned by the environment, and $h _ { t } ~ = ~ ( x , a _ { < t } , o _ { < t } )$ denotes the interaction history available to the policy. Upon

completion of the rollout, the environment or a verifier assigns a scalar outcome reward $R \left( \tau \right)$ . The objective of Agentic RL is to maximize the expected reward over the task distribution:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } , \tau \sim \pi _ { \theta } ( . | x ) } \left[ R ( \tau ) \right] .\tag{2}
$$

To estimate policy advantages without learning a separate value function, Group Relative Policy Optimization $( \mathrm { G R P O } ;$ Shao et al., 2024) uses the relative rewards of multiple rollouts sampled for the same task. Specifically, for each task $x ,$ GRPO samples a group of rollouts from the old policy $\pi _ { \theta _ { \mathrm { o l d } } } \colon G ( x ) = \{ \tau _ { i } \} _ { i } ^ { G }$ . Each rollout receives a trajectory-level reward $R _ { i } = R ( \tau _ { i } )$ . GRPO then estimates its relative advantage using the reward statistics within the group:

$$
\hat { A } _ { i } ^ { \mathrm { t r a j } } = \frac { R _ { i } - \mu _ { G } } { \sigma _ { G } + \epsilon } ,\tag{3}
$$

where $\mu _ { G }$ and $\sigma _ { G }$ are arithmetic mean and standard deviation of rewards in group $G ,$ respectively. Unlike PPO-style methods (Schulman et al., 2017) that typically rely on a learned critic to estimate token-level or state-level advantages, GRPO constructs its baseline directly from group-level reward statistics. This critic-free design reduces the computational and memory overhead of training an additional value model and avoids errors arising from inaccurate value estimation.

Ignoring optional KL regularization for clarity, the GRPO objective can be written as:

$$
\mathcal { I } _ { \mathrm { G R P O } } = \mathbb { E } _ { \boldsymbol { x } \sim D , \{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \tau | q ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | \tau _ { i } | } \sum _ { t = 1 } ^ { | \tau _ { i } | } \operatorname* { m i n } \left( \rho _ { i , t } ( \theta ) \hat { A } _ { i } ^ { \mathrm { t r a j } } , \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) ) \hat { A } _ { i } ^ { \mathrm { t r a j } } \right) ,\tag{4}
$$

where $\begin{array} { r } { \rho _ { i , t } = \frac { \pi _ { \theta } \left( a _ { i , t } | h _ { i , t } \right) } { \pi _ { \theta _ { 0 \mathrm { d d } } } \left( a _ { i , t } | h _ { i , t } \right) } } \end{array}$ is the token-level importance ratio.

Most existing GRPO variants, however, assume a single-session setting in which each rollout is represented as a continuous, linear training segment. This assumption becomes inadequate when advanced inference workflows extend agent execution across multiple sessions, creating a mismatch between training and inference that compromises consistency and limits optimization potential.

## 2.2. Broadcasting for Multi-Session Agentic RL

We introduce session as a finer-grained unit within an LLM agent trajectory. Specifically, a trajectory $\tau _ { i }$ is represented as a set of $N _ { i } \in$ ℕ of sessions:

$$
\tau _ { i } = \left\{ s _ { i , j } \right\} _ { j = 1 } ^ { N _ { i } } .\tag{5}
$$

Each session consists of a sequence of agent–environment interactions same as trajectories in conventional single-session settings:

$$
s _ { i , j } = ( a _ { i , j , 1 } , o _ { i , j , 1 } , \ldots , a _ { i , j , T _ { i , j } } , o _ { i , j , T _ { i , j } } ) , \qquad a _ { i , j , t } \sim \pi _ { \theta } ( \cdot \mid h _ { i , j , t } ) .\tag{6}
$$

For example, a session may include consecutive turns of execution to complete a task. When the remaining context window becomes insuficient, a new session can be initiated to perform context compaction, with the resulting compressed prefix serving as the initial context for another session.

Built upon the GRPO pipeline, broadcasting makes multi-session rollouts trainable by propagating the trajectory-level advantage $\hat { A } _ { i } ^ { \mathrm { t r a j } }$ to every session and every trainable token within $\tau _ { i } \mathbf { \cdot }$

$$
\hat { A } _ { i , j , t } ^ { \mathrm { t o k e n } } = \hat { A } _ { i , j } ^ { \mathrm { s e s s i o n } } , \hat { A } _ { i , j } ^ { \mathrm { s e s i o n } } = \hat { A } _ { i } ^ { \mathrm { t r a j } } , \forall s _ { i , j } \in \tau _ { i } , t \in \left\{ t \mid 1 \leq t \leq T _ { i , j } \land \operatorname { m a s k } ( t ) = 1 \right\} .\tag{7}
$$

![](images/9ddc4267ac68216a53b7598be6492027427fbe5094c556a8e215c7a7b6a93083.jpg)  
Figure 2 | Overview of our proposed Mosaic pipeline for multi-session agentic RL.

Since each session corresponds to a separate training segment in frameworks such as Megatron (Shoeybi et al., 2020), the token-level objectives from all sessions should further be aggregated at the trajectory level:

$$
\mathcal { I } _ { \mathrm { G R P O } } ^ { \prime } = \mathbb { E } _ { \boldsymbol { x } \sim \boldsymbol { D } , \{ \tau _ { i } \} _ { i = 1 } ^ { G } \sim \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } ( \tau | q ) } \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { \sum _ { j = 1 } ^ { N _ { i } } | s _ { i , j } | } \sum _ { j = 1 } ^ { N _ { i } } \sum _ { \iota = 1 } ^ { | s _ { i , j } | } \operatorname* { m i n } \left( \rho _ { i , \boldsymbol { t } } ( \boldsymbol { \theta } ) \hat { A } _ { i , j , \boldsymbol { t } } ^ { \mathrm { t o k e n } } , \mathrm { c l i p } ( \rho _ { i , \boldsymbol { t } } ( \boldsymbol { \theta } ) ) \hat { A } _ { i , j , \boldsymbol { t } } ^ { \mathrm { t o k e n } } \right) .\tag{8}
$$

This formulation yields a single token-weighted trajectory-level objective for each rollout, regardless of its session count, thereby preserving equal weighting across rollouts within a prompt group and stabilizing training-signal scales across tokens. However, broadcasting ofers only coarse supervision and no explicit control over optimization across sessions, causing poorly targeted learning signals as trajectories span more sessions. To address this limitation, we propose Mosaic, which improves multi-session agentic RL through session-aware trajectory-level advantage estimation and session-level advantage shaping.

## 3. Methodology

In Mosaic, we optimize the performance of multi-session agentic RL through a hierarchical perspective, incorporating trajectory-level advantage estimation calibrated with session signals (§3.1), as well as advantage shaping at the finer-grained session level. (§3.2).

## 3.1. Session-Aware Advantage Estimation

In vanilla GRPO, the advantage is determined solely by the relative outcome rewards within a prompt group. Under the multi-session setting, however, session-related metrics may provide additional relative signals for policy optimization. One metric is the number of sessions $N _ { i }$ in a trajectory $\tau _ { i } ,$ which serves as a proxy for the extent of exploration and structural complexity of $\tau _ { i }$ . Nonetheless, $N _ { i }$ should not be directly treated as a reward. First, its raw scale may not faithfully quantify meaningful diferences in session utilization. Second, the preferred direction of optimization should depend on task results, avoiding the policy from exploiting session count without improving performance.

Accordingly, from a group-relative perspective, the session-aware trajectory-level advantages of multi-session rollouts within a prompt group should satisfy the following ordering constraints:

• C1: For successful trajectories, using more sessions incurs a higher execution cost and hence results in a lower advantage, whereas using fewer sessions should result in a higher advantage.

• C2: For unsuccessful trajectories, using fewer sessions may suggest insuficient exploration or execution, especially on challenging tasks, and should thus result in a lower advantage. Conversely, unsuccessful trajectories that conduct more sessions should receive a higher advantage.

• C3: Task outcome reward must remain the dominant signal. When two trajectories have the same session counts, the trajectory with lower reward should receive a lower advantage. Using more sessions alone should not allow an unsuccessful trajectory to outrank a successful one.

Together, these constraints encourage the policy to use enough sessions to complete the task, but no more than necessary, which represents a desirable objective for multi-session agentic RL. Specifically, we assume a binary final outcome reward $R _ { i } \in \{ 0 , 1 \}$ , where 1 and 0 represents success and failure, respectively. For a prompt group containing both successful and unsuccessful rollouts, the trajectorylevel GRPO advantage satisfies:

$$
R _ { i } = 1 \Longrightarrow \hat { A } _ { i } ^ { \mathrm { t r a j } } > 0 , \quad R _ { i } = 0 \Longrightarrow \hat { A } _ { i } ^ { \mathrm { t r a j } } < 0 .\tag{9}
$$

Based on the ordering constraints above, we define the session-aware trajectory-level advantage as:

$$
\hat { A } _ { i } ^ { \mathrm { s a t } } = \left( 1 + \alpha _ { i } ^ { \mathrm { s e s s i o n } } \right) \hat { A } _ { i } ^ { \mathrm { t r a j } } ,\tag{10}
$$

where $\alpha _ { i } ^ { \mathsf { s e s s i o n } }$ is the session signal derived from $N _ { i } ,$ , acting as a factor that modulates the trajectory-level advantage according to session usage, and extending optimization beyond the final outcome.

Considering that session signal has diferent implications under diferent task outcomes as reflected by C1 and C2, session signals are not directly comparable between successful and unsuccessful trajectories. We thus further divide the prompt group � into subgroups according to $R _ { i } \colon$

$$
G _ { 1 } = \left\{ i \in G : R _ { i } = 1 \right\} , \quad G _ { 0 } = \left\{ i \in G : R _ { i } = 0 \right\}\tag{11}
$$

Generally, each $\tau _ { i }$ belongs to $G _ { R _ { i } }$ . We then map its $N _ { i }$ to a centered rank within this subgroup:

$$
u _ { i } = 2 f ( N _ { i } ) - 1 , f ( N _ { i } ) = \frac { \mathrm { r a n k } _ { G _ { R _ { i } } } ( N _ { i } ) - 1 } { | G _ { R _ { i } } | - 1 } ,\tag{12}
$$

where $u _ { i } \in [ - 1 , 1 ]$ , and the rank is computed in ascending order of $N _ { i }$ inside $G _ { R _ { i } }$ . Thus, a larger $N _ { i }$ yields a larger $u _ { i }$ . We use centered ranks instead of raw session counts to obtain a bounded signal robust to long-tailed trajectories, as diferences in $N _ { i }$ may not reflect proportional diferences in session usage. We then define the session modulation factor as:

$$
\alpha _ { i } ^ { \mathrm { s e s s i o n } } = - \rho u _ { i } ,\tag{13}
$$

where $0 \leq \rho < 1$ and $\rho$ controls the proportion by which the session signal may amplify or attenuate the original advantage. Then we have $1 - \rho u _ { i } \in [ 1 - \rho , 1 + \rho ]$ which is always positive. The session signal thus modulates only the magnitude of the outcome advantage without changing its sign.

Finally, we consider the fallback cases in the above computation. First, the centered-rank signal $u _ { i }$ is well-defined only when $\left| G _ { R _ { i } } \right| > 1 \land \lor \mathrm { a r } _ { G _ { R _ { i } } } > 0$ . If either condition is not satisfied, we set $u _ { i } = 0$ When all trajectories in a prompt group receive the same outcome reward, the outcome advantage in Eq. 10 becomes $^ { 0 , }$ rendering the session signal inefective. To improve sample eficiency in these homogeneous groups, we introduce the following fallback in the final advantage:

$$
\begin{array} { r } { \hat { A } _ { i } ^ { \mathrm { s a t } } = \left\{ \begin{array} { l l } { ( 1 - \rho u _ { i } ) \hat { A } _ { i } ^ { \mathrm { t r a j } } , } & { 0 < \sum _ { j \in G } R _ { j } < | G | , } \\ { - \beta _ { 1 } u _ { i } , } & { \sum _ { j \in G } R _ { j } = | G | , } \\ { \beta _ { 0 } \operatorname* { m i n } \left( u _ { i } , 0 \right) , } & { \sum _ { j \in G } R _ { j } = 0 . } \end{array} \right. } \end{array}\tag{14}
$$

For all-failure groups, we adopt the more conservative fallback, ensuring that no unsuccessful trajectory receives a positive advantage. The nonnegative coeficients $\beta _ { 0 }$ and $\beta _ { 1 }$ control the strengths of the fallback signals for all-failure and all-success groups, respectively.

## 3.2. Session-Level Advantage Shaping

The session-aware trajectory-level advantage developed in §3.1 provides an outcome-oriented global optimization signal for the entire multi-session trajectory. At the session level, however, this signal remains sparse since all sessions on the same trajectory share a single scalar advantage. To this end, we introduce session-level advantage shaping. Specifically, we view each session as a state transition as $s _ { i , j } = \left( \nu _ { i , j - 1 } \to \nu _ { i , j } \right)$ , where session $s _ { i , j }$ transitions the agent from state $\nu _ { i , j - 1 }$ to state $\nu _ { i , j } .$ Here, $\nu _ { i , j - 1 }$ and $\upsilon _ { i , j }$ denote the agent states at the start and end of the �-th session, respectively, with particular emphasis on the prefix context visible to the rollout policy. Based on session transition functions, we introduce three representative session types:

• Normal session. The agent performs regular reasoning, tool use, and environment interaction within the current context. The input state is the context available at the start of the session, and its output state is the resulting context after the session is completed.

• Compaction session. When the accumulated context approaches the model’s limit, the agent starts a compaction session to compress the context into a shorter prefix. Its input state is the full context before compaction, and its output state is the compressed prefix passed to the next session.

• Subagent session. The main agent delegates subtasks to subagents which start new subagent sessions. Its input state is the main-agent context at delegation, and its output state is the updated main-agent context after the returned result has been incorporated.

To quantify the benefit of each session transition, we construct a critic-free process supervision signal based on marginal information gain. Specifically, let $y _ { i } ^ { * } = \left( y _ { i , 1 } ^ { * } , \ldots , y _ { i , L _ { i } } ^ { * } \right)$ denote the ground-truth answer to query $x _ { i }$ of trajectory $\tau _ { i }$ . We use the rollout policy before the update of each step, denoted by $\pi _ { \bar { \theta } } .$ , and define the answer-likelihood potential of any state � as:

$$
\Phi _ { \bar { \theta } } \left( y _ { i } ^ { * } \mid x _ { i } , \nu \right) = \frac { 1 } { L _ { i } } \sum _ { \ell = 1 } ^ { L _ { i } } \log \pi _ { \bar { \theta } } \left( y _ { i , \ell } ^ { * } \mid x _ { i } , \nu , y _ { i , < \ell } ^ { * } \right) .\tag{15}
$$

Intuitively, $\Phi _ { \bar { \theta } } ( y _ { i } ^ { * } \mid x _ { i } , \nu )$ measures how readily the rollout policy can generate the correct answer given the prefix context represented by state $\nu .$ A session that adds useful evidence, reduces noise, or returns a high-quality subtask result should increase the average log-likelihood of the ground-truth answer. Conversely, a transition that discards key facts or introduces misleading information may decrease it. Accordingly, for a session transition $s _ { i , j } = ( \nu _ { i , j - 1 }  \nu _ { i , j } )$ , we define its session-level reward as the marginal information gain between its output and input states:

$$
r _ { i , j } ^ { \Delta } = \mathrm { s t o p g r a d } \left[ \Phi _ { \bar { \theta } } \left( y _ { i } ^ { * } \mid x _ { i } , { v } _ { i , j } \right) - \Phi _ { \bar { \theta } } \left( y _ { i } ^ { * } \mid x _ { i } , { v } _ { i , j - 1 } \right) \right] .\tag{16}
$$

Diferent types of session may have diferent distributions of marginal gain. Normal sessions typically gain external evidence through search and browse. The gain from a compaction session depends on whether compression retains useful information while removing noise. A subagent session is afected by both the quality of subtask execution and the quality of result handof. Therefore, we normalize the session-level rewards separately by session type within each prompt group:

$$
\hat { r } _ { i , j } = \left\{ \begin{array} { l l } { \frac { r _ { i , j } ^ { \Delta } - \mu _ { c \left( s _ { i , j } \right) } } { \sigma _ { c \left( s _ { i , j } \right) } + \epsilon } } & { \mathrm { ~ i f ~ } 1 \leq j < N _ { i } , } \\ { 0 } & { \mathrm { ~ i f ~ } j = N _ { i } , } \end{array} \right.\tag{17}
$$

Table 1 | Results of various methods and inference modes on BrowseComp.
<table><tr><td>Inference</td><td rowspan="3">Single-Session Accuracy</td><td rowspan="3">Num. Session</td><td rowspan="3">Multi-Session / Compact</td><td rowspan="3">Num. Session</td><td rowspan="3">Multi-Session / Subagent Accuracy</td><td rowspan="3">Num. Session</td></tr><tr><td>Method</td><td>Accuracy</td></tr><tr><td>Qwen3.5-4B</td><td>24.80</td></tr><tr><td></td><td>23.22</td><td>1.00</td><td>Training on Single-Session</td><td>1.70</td><td>21.64</td><td>2.98</td></tr><tr><td>GRPO</td><td>31.99</td><td>1.00</td><td>34.68</td><td>1.24</td><td>31.91</td><td>2.09</td></tr><tr><td>IGPO</td><td>28.52</td><td>1.00</td><td>31.36</td><td>4.12</td><td>24.64</td><td>3.26</td></tr><tr><td></td><td></td><td></td><td>Training on Multi-Session</td><td></td><td></td><td></td></tr><tr><td>Broadcasting</td><td></td><td></td><td>35.31</td><td>3.17</td><td>31.44</td><td>4.43</td></tr><tr><td>MOSAIC</td><td>一</td><td>一</td><td>38.86</td><td>2.35</td><td>32.46</td><td>3.77</td></tr></table>

where $c \left( s _ { i , j } \right) \in$ {normal, compact, subagent, . . . } denotes the type of session $s _ { i , j }$ , and $\mu _ { c ( s _ { i , j } ) }$ and $\sigma _ { c ( s _ { i , j } ) }$ are computed over sessions of the same type within the prompt group. We exclude the last session since its output state usually includes the agent’s final answer. Particularly, a correct final answer may sharply increase the answer likelihood which confounds the marginal gain distribution. Setting its shaping signal to 0 prevents the final answer from entering the evaluated output state.

Finally, we combine the normalized session-level shaping signal with the trajectory-level advantage:

$$
\hat { A } _ { i , j } ^ { \mathrm { s e s s i o n } } = \hat { A } _ { i } ^ { \mathrm { t r a j } } + \lambda _ { r } \hat { r } _ { i , j } ,\tag{18}
$$

where � controls the intensity of the shaping signal, and $\hat { A } _ { i } ^ { \mathrm { t r a j } }$ can be instantiated directly as $\hat { A } _ { i } ^ { \mathrm { s a t } }$ in §3.1, after which $\hat { A } _ { i , j } ^ { s e s s i o n }$ replaces the broadcast session advantage used to compute $\hat { A } _ { i , j , t } ^ { \mathrm { t o k e n } }$ in Eq. 8. Overall, the first term provides a long-term, outcome-oriented global signal, while the second diferentiates sessions according to their local relative marginal gains.

## 4. Experiments

## 4.1. Experimental Setup

Benchmark. We evaluate our method on BrowseComp (Wei et al., 2025), a challenging benchmark consisting of 1,266 questions that require agents to persistently navigate the web and identify hardto-find, entangled information. We select BrowseComp since its high dificulty demands long-horizon interactions, making it suitable for assessing agents that produce multi-session trajectories.

Baselines. We use Qwen3.5-4B (Qwen Team, 2026) as the initial model and adopt a ReAct-style agent loop. We organize the baselines by inference configuration. Under single-session inference, we evaluate (1) the initial model without RL training and (2) a model trained with GRPO in the single-session setting, representing conventional Agentic RL. Under multi-session inference, we evaluate (3) the single-session GRPO trained model, representing test-time scaling; (4) a model trained with Broadcasting in the multi-session setting; and (5) a model trained with Mosaic in the multi-session setting. We evaluate all multi-session configurations under both context compaction and subagent forking using their respective agent loops; for multi-session training, the same loops are used to collect rollouts. Finally, since our session-level advantage shaping in §3.2 builds on information gain, we evaluate (6) a single-session model trained with IGPO (Wang et al., 2026).

![](images/853ac60a960fc6cd99ea1c040eb0509cd0eecca14a572c60dac92388a519e484.jpg)

![](images/a820b8c1abbb07ee865e860b9eff431e5f7507b8436ebd4fc0d705c2131f37bb.jpg)

![](images/f57a39db93f8ff12b72b0cbb310738a6f60accfb59b10ab06ba23a62b8274b5a.jpg)

![](images/8043e95fafb79c23aed51466f1327faf19a9bd7adf72f06ff87159cc3c841f94.jpg)  
Figure 3 | (a). Percentage of the 972 tasks failed by the base model under single-session inference that are solved by diferent methods under multi-session inference with compact. (b). Comparison of the computation time for marginal information gain. (c)(d). Ablation results on BrowseComp.

Implementation Details. The agent has two tools, search and visit, implemented using Serper <sup>1</sup> and Jina Reader <sup>2</sup> APIs, respectively. During training and evaluation, deepseek-v4-flash (DeepSeek-AI et al., 2026) serves as the LLM judge for verifying final answers. For context compaction, the agent compresses its interaction history when approaching the context limit. For subagent forking, it can invoke non-nested subagents through a dedicated action. Each session is limited to 128K tokens. For RL training, we use REDSearcher\_RL\_1k (Chu et al., 2026), which contains 1,000 challenging deep-search tasks. We train all models for one epoch with a rollout batch size of 16 and a group size of 8. We set $\rho = 0 . 2 .$ $\beta _ { 0 } = 0 . 1$ $\beta _ { 1 } = 0 . 2$ . and $\lambda _ { r } = 0 . 1$

## 4.2. RQ1: Overall Efectiveness of Mosaic

RQ1 asks whether training in the multi-session setting improves agent performance and whether Mosaic contributes beyond broadcasting.

As shown in Table 1, single-session inference has limited efectiveness on BrowseComp. Agentic RL training in the single-session setting yields modest gains, even when combined with test-time scaling in the multi-session setting. This suggests that multi-session inference capabilities do not readily emerge through generalization alone. Among methods trained on multi-session trajectories, Broadcasting improves multi-session inference performance. Mosaic improves it further, outperforming all baselines in both the compact and subagent scenarios. Figure 3 (a) illustrates the transformative gains from multi-session inference, multi-session training, and Mosaic.

Figure 4 shows the training dynamics of each method. Mosaic boosts trajectory reward faster than Broadcasting and reaches a higher level. The number of sessions increases over training for both methods, but its growth levels of more clearly with Mosaic. This may be partly due to the more targeted signal provided by its session-aware advantage, which discourages unnecessary increases in session count. Overall, Mosaic achieves better task performance at lower execution cost.

As §3.2 applies marginal gain to session-level advantage shaping, we thus compare Mosaic with IGPO, which applies it at the turn level. From Table 1, Mosaic achieves better performance than IGPO. Moreover, Figure 3 (b) shows that Mosaic substantially reduces the wall-time of computing marginal gains, enhancing performance while cutting computational cost.

![](images/781e5339b85d60370378dbf71408903c81989010fdb0391f1fb808228953fa99.jpg)

![](images/a942afda56d4dee63b55f171edd625d3ad510704423feea724db16614d7e6aed.jpg)

![](images/a2f78e340f65d24eb744e94035e5247df51cc276eaa80ac3fcad741f43a60f46.jpg)

![](images/4cfd143e9e22164e324e0cba468c9acc8c1b4ead7033bc00d197d1ee48eea8f4.jpg)  
Figure 4 | Curves showing rewards and session counts during the training process of various methods.

## 4.3. RQ2: Ablation Study of Mosaic

RQ2 asks whether each component of Mosaic contributes to its overall performance.

Figure 3 (c)(d) presents the ablation results. Relative to Broadcasting, both levels of components in Mosaic, namely trajectory-level session-aware advantage estimation and session-level advantage shaping, contribute to the enhanced final performance. Combining the two components yields a clear additional gain in the compact scenario, but only a modest gain in the subagent scenario. One possible explanation lies in their session structures: compact trajectories are linear, whereas subagent trajectories have a nonlinear structure that more closely resembles a directed acyclic graph. In the latter case, a session may influence multiple succeeding sessions. Its immediate reward, estimated from marginal gain, may thus difer from its actual contribution to the final outcome.

## 4.4. RQ3: Mechanistic Analysis of Mosaic

RQ3 asks <sup>h</sup>ow t<sup>h</sup>e design of Mosaic contributes to t<sup>h</sup>e training process.

First, we examine whether session-aware trajectory-level advantage estimation produces the intended ordering. As shown in Figure 5, within mixed-outcome groups, advantages approximately decrease with session count for successful trajectories and increase for unsuccessful ones, while successful trajectories retain higher advantages. In homogeneous-outcome groups, the fallback mechanism introduces session-dependent variation, providing diferentiated learning signals despite identical outcome rewards. These patterns illustrate how Mosaic leverages session signals to increase the within-group variance of relative advantages, thus improving sample eficiency during training.

Second, we examine the efect of reward normalization on session-level shaping signals. As shown in Figure 5, the distributions of raw marginal information gain difer across session types, primarily because heterogeneous sessions serve distinct functions. Under global normalization, compaction sessions, whose raw marginal gains have a lower mean, are further downweighted, introducing a systematic bias against them. In Mosaic, type-conditioned normalization instead centers and rescales each type separately, enabling both types to receive positive and negative signals relative to their respective baselines. This reduces type-dependent distributional bias in advantage shaping.

## 4.5. RQ4: Understanding Multi-Session Optimization

Multi-session optimization introduces an additional degree of freedom in how computation is distributed across sessions. However, the session count used by Mosaic serves only as a proxy signal and is not inherently tied to exploration intensity or execution cost. Moreover, at inference time, the model has no access to hindsight: it does not know how many sessions a trajectory will ultimately contain or whether the trajectory will succeed. Therefore, RQ4 moves beyond method-specific analysis and asks a broader question: w<sup>h</sup>en multi-session optimization improves tas<sup>k</sup> performance, w<sup>h</sup>at session-level b<sub>e</sub>h<sub>a</sub>vi<sub>o</sub>r<sub>a</sub>l <sub>pa</sub>tt<sub>e</sub>rn<sub>s e</sub>m<sub>e</sub>r<sub>ge</sub>?

Table 2 | Session-level patterns under multi-session optimization on BrowseComp.
<table><tr><td></td><td colspan="4">Multi-Session / Compact</td><td colspan="4">Multi-Session / Subagent</td></tr><tr><td>Method</td><td>Accuracy ↑</td><td># Sessions</td><td>Positive-IG ↑</td><td>IG / Session ↑</td><td>Accuracy ↑</td><td># Sessions</td><td>Positive-IG ↑</td><td>IG / Session ↑</td></tr><tr><td>Base Model</td><td>24.80</td><td>1.70</td><td>37.36%</td><td>0.30</td><td>21.64</td><td>2.98</td><td>27.39%</td><td>0.41</td></tr><tr><td>Broadcasting</td><td>35.31</td><td>3.17</td><td>47.74%</td><td>0.86</td><td>31.44</td><td>4.43</td><td>41.15%</td><td>0.74</td></tr><tr><td>MoSAIC</td><td>38.86</td><td>2.35</td><td>59.92%</td><td>0.89</td><td>32.46</td><td>3.77</td><td>51.04%</td><td>0.80</td></tr></table>

![](images/2147462812d3d67837256d626fb399f442f87d648609a9185ef437a3c22cac39.jpg)

![](images/53fd1679927be0741f72ef85fbec6c24fc21b579f677aba43b1b2b69d2c60503.jpg)

![](images/d32ce5b743e90c60b26ac5df9bbe1013a3f4fe51bdcc99a60cc0f66dd045d54e.jpg)

![](images/1ee26800c4c5fe1a28bd7c4d2a8a90aa838308bfd7eedf7f2478c6cd4ecee945.jpg)  
Figure 5 | The two left panels show the relationship between trajectory-level advantage and session count. The two right panels show the distributions of marginal information gain for normal and compact sessions under global normalization and type-conditioned normalization.

Accordingly, we examine session-level behavior along three dimensions: quantity, or whether the count of sessions increases or decreases; quality, or whether performance improvement is accompanied by more useful sessions; and allocation, or whether a limited session budget becomes increasingly concentrated on execution stages with high marginal utility. As shown in Table 2, for Broadcasting, which does not directly optimize any specific session-aware pattern, all three metrics increase. Under Mosaic, the number of sessions does not increase further, while both session quality and allocation improve. These findings indicate that performance gains via multi-session optimization are associated with improved session utility rather than a monotonic rise in session count. Moreover, despite using explicit session-aware optimization, Mosaic does not induce a fundamentally altered behavioral mechanism. Instead, it strengthens the session-allocation patterns that already emerge from naïve multi-session optimization.

We argue that efective multi-session optimization is accompanied by improvements in session utility and allocation. It distributes computation more efectively across sessions, encouraging continued execution when it yields meaningful progress while suppressing low-utility computation. From this perspective, the central challenge is not how many sessions an agent should use, but how much useful progress it can derive from the sessions it executes.

## 5. Discussion: Future Directions for Multi-Session Agentic RL

Infrastructure for reproducible research. Multi-session trajectories tend to arise naturally when agents tackle complex, long-horizon tasks. Although some existing benchmarks can elicit these trajectories with reasonable frequency, suitable public training data remain scarce. This gap limits the scale, suficiency, and reproducibility of experiments in multi-session agentic RL. Developing open training datasets and corresponding agentic execution environments is therefore essential to building a sustainable research ecosystem for this topic.

Cross-session state transfer as a central learning objective. Context compaction and subagent forking both involve information transfer, but serve diferent purposes: compaction must preserve suficient working state for continued execution; forking must provide the information needed to complete a subtask; merging must integrate subtask results into a state usable by the main agent. These operations motivate a shared research question: under a limited context budget, what information should be carried forward to support subsequent decisions? CompactionRL already jointly optimizes task execution and context summarization (Li et al., 2026). A unified approach to learning state transfer across compaction, forking, and merging remains an important direction, alongside systematic investigation of information loss and error accumulation over repeated transfers.

Asynchronous training for multi-session rollouts. For partial rollout or asynchronous training settings, multi-session execution can amplify variation in trajectory lengths and completion times, increase policy staleness and complicate within-group comparisons and sample reuse. For instance, prioritizing completed rollouts may favor trajectories with less sessions over longer ones associated with harder tasks, introducing selection bias and shaping learned execution strategies. Our experiments use a colocated setup with alternating rollout and training phases to minimize interference from asynchrony and isolate the efects of the optimization signals. Scaling toward actual production raises a further question: how can multi-session optimization remain reliable under asynchronous sampling and policy staleness? SAO explores asynchronous optimization with one rollout per prompt, using a learned value model (Hou et al., 2026). However, critic-free, group-relative optimization requires further investigation into when and how rollout groups are formed, how stale samples are handled, and how collected trajectories can be reused.

Toward RL within (black-boxed) agent harnesses. General multi-session agentic RL can accommodate workflows in which execution no longer follows a single, continuously accumulated context prefix. Context compaction and subagent forking, the two scenarios studied in this work, are representative examples of such workflows in modern agent harnesses. Progress in multi-session optimization could thus provide an important foundation for training agents within harnesses whose orchestration is partially opaque (Xu et al., 2026a), such as Claude Code. Therefore, a critical next step is to systematically examine mechanisms across agent harnesses that disrupt context continuity, such as dropping reasoning tokens or tool responses, and improve general multi-session agentic RL under these conditions. Additionally, when discontinuity occurs, the choice between introducing a new session boundary and applying lossy concatenation with the preceding session may also influence the overall efectiveness of optimization.

## 6. Related Work

LLM Agents. LLM agents extend LLMs with tool use and environment interaction to work autonomously (Feng et al., 2026; Li et al., 2025; Wang et al., 2024; Wu et al., 2026). ReAct constructs a general agent loop of reasoning, acting, observing environment feedback, and continuing until task completion (Yao et al., 2022). To handle complex and long-horizon tasks, recent systems develop agent harnesses that manage execution beyond the basic ReAct loop (OpenClaw, 2026; Works, 2026), incorporating context compaction (Nguyen et al., 2025), multi-agent system (Chen et al., 2026a), subagent forking (Team et al., 2026), and other forms of context and workflow management (Chen et al., 2025; Kim et al., 2025; Shi et al., 2026; Xu et al., 2026c; Yang et al., 2026; Zhang et al., 2026a).

Agentic RL. Reinforcement learning techniques such as RLHF (Ouyang et al., 2022; Zhang et al., 2026b) and RLVR (Chen et al., 2026b; Yu et al., 2026) are used to improve LLMs’ alignment, reasoning, and downstream task-solving capabilities (Tan et al., 2025; Zheng et al., 2025). Agentic RL extends this paradigm to LLM agents, aiming to enhance capabilities of multi-turn interaction through enhanced rollout sampling (Dong et al., 2026; Hou et al., 2025), process-level reward design (Liu et al., 2026; Zhang et al., 2026c), and other optimization strategies (Deng et al., 2026). However, most existing agentic RL methods still assume that each trajectory is contained within a single session.

## 7. Conclusion

In this paper, we first identify a mismatch between the single-session assumption underlying most existing agentic RL methods and advanced agentic inference techniques such as context compaction and subagent forking. This motivates our formulation of Multi-Session Agentic Reinforcement Learning, which supports an arbitrary count of sessions within a trajectory. To address the coarse supervision issue existed in the currently widely used broadcasting approach, we further propose Mosaic, a multi-session agentic RL method with critic-free hierarchical optimization. By combining sessionaware trajectory-level advantage estimation with session-level advantage shaping, Mosaic trains a Qwen3.5-4B agent that outperforms both single-session and multi-session baselines on BrowseComp. Ablation studies confirm the contribution of each component, while further analyses suggest that efective multi-session optimization might depend on improving session utility and allocation, a pattern observed in both broadcasting baseline and our Mosaic. We hope these findings will inform future research on multi-session agentic RL.

## References

Accio-Lab. Dressage. GitHub, 2026. URL https://github.com/Accio-Lab/Dressage.

C. Chen, C. Li, Z. Li, Y. Liu, and Z. Guo. From trainee to trainer: Llm-designed training environment for rl with multi-agent reasoning. arXiv preprint arXiv:2606.17682, 2026a.

G. Chen, Z. Qiao, X. Chen, D. Yu, H. Xu, W. X. Zhao, R. Song, W. Yin, H. Yin, L. Zhang, et al. Iterresearch: Rethinking long-horizon agents via markovian state reconstruction. arXiv e-prints, pages arXiv–2511, 2025.

Z. Chen, R. Lu, A. Zhao, Z. Wang, Y. Yue, S. Song, and G. Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? Advances in Neural Information Processing Systems, 38:57654–57689, 2026b.

Z. Chu, X. Wang, J. Hong, H. Fan, Y. Huang, Y. Yang, G. Xu, C. Zhao, C. Xiang, S. Hu, D. Kuang, M. Liu, B. Qin, and X. Yu. Redsearcher: A scalable and cost-eficient framework for long-horizon search agents, 2026. URL https://arxiv.org/abs/2602.14234.

DeepSeek-AI, A. Xu, B. Lin, B. Xue, B. Wang, B. Xu, B. Wu, B. Zhang, C. Lin, C. Dong, et al. Deepseekv4: Towards highly eficient million-token context intelligence, 2026. URL https://arxiv.org/ abs/2606.19348.

M. Deng, Z. Li, X. Li, T. Zhu, Y. Zhao, Z. Guo, and W. Wang. Uncertainty-aware clarification in llm agents with information gain. arXiv preprint arXiv:2606.03135, 2026.

G. Dong, H. Mao, K. Ma, L. Bao, Y. Chen, Z. Wang, Z. Chen, J. Du, H. Wang, F. Zhang, et al. Agentic reinforced policy optimization. In International Conference on Learning Representations, volume 2026, pages 16981–17017, 2026.

J. Feng, S. Huang, X. Qu, G. Zhang, Y. Qin, B. Zhong, C. Jiang, J. Chi, and W. Zhong. Retool: Reinforcement learning for strategic tool use in llms. In International Conference on Learning Representations, volume 2026, pages 37909–37926, 2026.

J. Gao, W. Fu, M. Xie, S. Xu, C. He, Z. Mei, B. Zhu, and Y. Wu. Beyond ten turns: Unlocking long-horizon agentic search with large-scale asynchronous rl. arXiv preprint arXiv:2508.07976, 2025.

Z. Hou, Z. Hu, Y. Li, R. Lu, J. Tang, and Y. Dong. Treerl: Llm reinforcement learning with on-policy tree search. In Proceedings of t<sup>h</sup>e 63rd Annual Meeting of t<sup>h</sup>e Association for Computational Linguistics (Volume 1: Long Papers), pages 12355–12369, 2025.

Z. Hou, Y. Li, J. Tang, and Y. Dong. Single-rollout asynchronous optimization for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2607.07508.

B. Jin, H. Zeng, Z. Yue, J. Yoon, S. Arik, D. Wang, H. Zamani, and J. Han. Search-r1: Training llms to reason and leverage search engines with reinforcement learning. arXiv preprint arXiv:2503.09516, 2025.

Y. Kim, K. Gu, C. Park, C. Park, S. Schmidgall, A. A. Heydari, Y. Yan, Z. Zhang, Y. Zhuang, Y. Liu, et al. Towards a science of scaling agent systems. arXiv preprint arXiv:2512.08296, 2025.

K. Li, Z. Zhang, H. Yin, L. Zhang, L. Ou, J. Wu, W. Yin, B. Li, Z. Tao, X. Wang, et al. Websailor: Navigating super-human reasoning for web agent. arXiv preprint arXiv:2507.02592, 2025.

Y. Li, Z. Hou, Y. Jing, J. Tang, and Y. Dong. Compactionrl: Reinforcement learning with context compaction for long-horizon agents, 2026. URL https://arxiv.org/abs/2607.05378.

J. Lin, S. Liu, C. Pan, L. Lin, S. Dou, Z. Xi, X. Huang, H. Yan, Z. Han, T. Gui, et al. Agentic harness engineering: Observability-driven automatic evolution of coding-agent harnesses. arXiv preprint arXiv:2604.25850, 2026.

X. Liu, K. Wang, Y. Wu, F. Huang, Y. Li, J. Jiao, and J. Zhang. Agentic reinforcement learning with implicit step rewards. In International Conference on Learning Representations, volume 2026, pages 129271–129291, 2026.

M. Lu, W. Sun, W. Du, Z. Ling, X. Yao, K. Liu, and J. Chen. Scaling llm multi-turn rl with end-to-end summarization-based context management. arXiv preprint arXiv:2510.06727, 2025.

L. Mei, J. Yao, Y. Ge, Y. Wang, B. Bi, Y. Cai, J. Liu, M. Li, Z.-Z. Li, D. Zhang, et al. A survey of context engineering for large language models. arXiv preprint arXiv:2507.13334, 2025.

X.-P. Nguyen, S. Pandit, R. G. Reddy, A. Xu, S. Savarese, C. Xiong, and S. Joty. Sfr-deepresearch: Towards efective reinforcement learning for autonomously reasoning single agents. arXiv preprint arXiv:2509.06283, 2025.

OpenClaw. Openclaw. GitHub, 2026. URL https://github.com/openclaw/openclaw.

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen. ai/blog?id=qwen3.5.

J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms, 2017. URL https://arxiv.org/abs/1707.06347.

Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402.03300.

Y. Shi, Y. Chen, S. Wang, S. Li, H. Cai, Q. Gu, X. Wang, and A. Zhang. Look back to reason forward: Revisitable memory for long-context llm agents. In International Conference on Learning Representations, volume 2026, pages 78055–78082, 2026.

N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, and S. Yao. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36:8634–8652, 2023.

M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-lm: Training multi-billion parameter language models using model parallelism, 2020. URL https://arxiv. org/abs/1909.08053.

K. Song, A. Trotter, and J. Y. Chen. Llm agent swarm for hypothesis-driven drug discovery. arXiv preprint arXiv:2504.17967, 2025.

S. Tan, M. Luo, J. Wong, C. Cai, X. Shi, W. Y. Tang, M. Roongta, T. Zhang, L. E. Li, R. A. Popa, et al. Deepscaler: Efective rl scaling of reasoning models via iterative context lengthening. 2025.

K. Team, T. Bai, Y. Bai, Y. Bao, S. Cai, Y. Cao, Z. Chai, Y. Charles, H. Che, C. Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.

G. Wang, S. Dai, G. Ye, Z. Gan, W. Yao, Y. Deng, X. Wu, et al. Information gain-based policy optimization: A simple and efective approach for multi-turn search agents. In International Conference on Learning Representations, volume 2026, pages 104502–104529, 2026.

X. Wang, Y. Zhang, O. Zohar, and S. Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024.

J. Wei, Z. Sun, S. Papay, S. McKinney, J. Han, I. Fulford, H. W. Chung, A. T. Passos, W. Fedus, and A. Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents, 2025. URL https://arxiv.org/abs/2504.12516.

E. Works. Pi. GitHub, 2026. URL https://github.com/earendil-works/pi.

J. Wu, B. Li, R. Fang, W. Yin, L. Zhang, Z. Wang, Z. Tao, D.-C. Zhang, Z. Xi, R. Tang, et al. Webdancer: Towards autonomous information seeking agency. Advances in Neural Information Processing Systems, 38:120957–120985, 2026.

X. Wu, K. Li, Y. Zhao, L. Zhang, L. Ou, H. Yin, Z. Zhang, X. Yu, D. Zhang, Y. Jiang, et al. Resum: Unlocking long-horizon search intelligence via context summarization. arXiv preprint arXiv:2509.13313, 2025.

B. Xu, H. Zhang, S. Zhang, S. Han, M. Liu, J. Hu, S. Diao, Z. Jin, Y. Zou, M. Demoret, et al. Polar: Agentic rl on any harness at scale. arXiv preprint arXiv:2605.24220, 2026a.

M. Xu, Z. Wang, M. Deng, Z. Li, Z. Yang, X. Zhu, Y. Liu, B. Zhu, B. Huang, C. Chen, et al. Envfactory: Scaling tool-use agents via executable environments synthesis and robust rl. arXiv preprint arXiv:2605.18703, 2026b.

W. Xu, Z. Liang, K. Mei, H. Gao, J. Tan, and Y. Zhang. A-mem: Agentic memory for llm agents. Advances in Neural Information Processing Systems, 38:17577–17604, 2026c.

Y. Xu, Z. Zheng, X. Long, Y. Cai, and Y. Wang. Self-manager: Parallel agent loop for long-form deep research. arXiv preprint arXiv:2601.17879, 2026d.

Y. Yang, C. Qu, M. Wen, L. Shi, Y. Wen, W. Zhang, A. Wierman, and S. Gu. Understanding agent scaling in llm-based multi-agent systems via diversity. arXiv preprint arXiv:2602.03794, 2026.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. arXiv preprint arXiv:2210.03629, 2022.

Q. Yu, Z. Zhang, R. Zhu, Y. Yuan, X. Zuo, Y. Yue, W. Dai, T. Fan, G. Liu, L. Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

G. Zhang, H. Geng, X. Yu, Z. Yin, Z. Zhang, Z. Tan, H. Zhou, Z. Li, X. Xue, Y. Li, et al. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547, 2025.

Q. Zhang, C. Hu, S. Upasani, B. Ma, F. Hong, V. Kamanuru, J. Rainton, C. Wu, M. Ji, H. Li, et al. Agentic context engineering: Evolving contexts for self-improving language models. In International Conference on Learning Representations, volume 2026, pages 86069–86100, 2026a.

S. Zhang, L. Dong, X. Li, S. Zhang, X. Sun, S. Wang, J. Li, R. Hu, T. Zhang, G. Wang, et al. Instruction tuning for large language models: A survey. ACM Computing Surveys, 58(7):1–36, 2026b.

W. Zhang, X. Li, K. Dong, Y. Wang, P. Jia, X. Li, Y. Zhang, D. Xu, Z. Du, H. Guo, et al. Process vs. outcome reward: Which is better for agentic rag reinforcement learning. Advances in Neural Information Processing Systems, 38:58701–58729, 2026c.

Y. Zheng, D. Fu, X. Hu, X. Cai, L. Ye, P. Lu, and P. Liu. Deepresearcher: Scaling deep research via reinforcement learning in real-world environments. In Proceedings of t<sup>h</sup>e 2025 Conference on Empirical Met<sup>h</sup>ods in Natural Language Processing, pages 414–431, 2025.

C. Zhou, H. Chai, W. Chen, Z. Guo, R. Shan, Y. Song, T. Xu, Y. Yang, A. Yu, W. Zhang, et al. Externalization in llm agents: A unified review of memory, skills, protocols and harness engineering. arXiv preprint arXiv:2604.08224, 2026.

## A. Details of Section 3.1

## A.1. Ordering of the Session-Aware Advantage

We show how the advantage defined in §3.1 satisfies the ordering constraints C1–C3. Let � be a prompt group, $R _ { i } \in \{ 0 , 1 \}$ the outcome reward of trajectory �, and $N _ { i }$ its number of sessions. Within each outcome subgroup, the centered-rank signal $u _ { i } \in [ - 1$ , 1] increases with $N _ { i }$ whenever the subgroup contains distinct session counts. If a subgroup contains only one trajectory or has zero variance in session count, we set $u _ { i } = 0$

We first consider a mixed-outcome group, for which $\begin{array} { r } { 0 < \sum _ { i \in G } R _ { i } < | G | } \end{array}$ . In this case, the session-aware advantage is

$$
\hat { A } _ { i } ^ { \mathrm { s a t } } = ( 1 - \rho u _ { i } ) \hat { A } _ { i } ^ { \mathrm { t r a j } } , \qquad 0 < \rho < 1 ,\tag{19}
$$

where $\hat { A } _ { i } ^ { \mathrm { t r a j } }$ is the group-relative advantage computed from the binary outcome rewards. Because all trajectories with the same reward have the same outcome advantage, there exist constants $a _ { + } > 0$ and $a \mathrm { ~ \textbar ~ { ~ } ~ } < 0$ such that

$$
\hat { A } _ { i } ^ { \mathrm { t r a j } } = \left\{ { a _ { + } , \quad R _ { i } = 1 , } \atop { a _ { - } , \quad R _ { i } = 0 . } \right.\tag{20}
$$

C1: ordering among successful trajectories. Consider two successful trajectories � and � in the same prompt group. If $N _ { i } < N _ { j } ,$ their within-outcome ranks satisfy $u _ { i } < u _ { j }$ . Since both trajectories have outcome advantage $a _ { + } > 0$

$$
\begin{array} { r l } & { \hat { A } _ { i } ^ { \mathrm { s a t } } - \hat { A } _ { j } ^ { \mathrm { s a t } } = a _ { + } \big [ ( 1 - \rho u _ { i } ) - ( 1 - \rho u _ { j } ) \big ] } \\ & { \qquad = \rho a _ { + } ( u _ { j } - u _ { i } ) > 0 . } \end{array}\tag{21}
$$

Thus, among successful trajectories, the one using fewer sessions receives a higher advantage.

C2: ordering among unsuccessful trajectories. Now consider two unsuccessful trajectories with $N _ { i } < N _ { j }$ and hence $u _ { i } < u _ { j }$ . Their common outcome advantage is $a \mathrm { ~ \textbar ~ { ~ } ~ } < 0$ . Therefore,

$$
\hat { A } _ { i } ^ { \mathrm { s a t } } - \hat { A } _ { j } ^ { \mathrm { s a t } } = \rho a _ { - } ( u _ { j } - u _ { i } ) < 0 .\tag{22}
$$

The trajectory using more sessions receives a higher, though still negative, advantage. This ordering gives relatively less negative feedback to unsuccessful trajectories that continue execution. It does not imply that additional sessions necessarily improve the answer.

C3: dominance of the outcome reward. Since $u _ { i } \in [ - 1 , 1 ]$ and $0 < \rho < 1$ , the modulation factor satisfies

$$
1 - \rho u _ { i } \geq 1 - \rho > 0 .\tag{23}
$$

Consequently, sign $( \hat { A } _ { i } ^ { \mathrm { s a t } } ) = \mathrm { s i g n } ( \hat { A } _ { i } ^ { \mathrm { t r a j } } )$ . For any successful trajectory � and unsuccessful trajectory � in the same mixed-outcome group,

$$
\hat { A } _ { i } ^ { \mathrm { s a t } } > 0 > \hat { A } _ { j } ^ { \mathrm { s a t } } , \qquad \mathrm { a n d ~ h e n c e } \qquad \hat { A } _ { i } ^ { \mathrm { s a t } } > \hat { A } _ { j } ^ { \mathrm { s a t } } ,\tag{24}
$$

regardless of their session counts. This also covers the case $N _ { i } = N _ { j }$ . Therefore, increasing the session count alone cannot make an unsuccessful trajectory outrank a successful one under the mixed-outcome advantage rule.

Homogeneous-reward groups. When all trajectories succeed, the outcome advantage is zero and the fallback gives $\hat { A } _ { i } ^ { \mathrm { s a t } } = - \beta _ { 1 } u _ { i } . \mathrm { I f } \ \beta _ { 1 } > 0$ , a smaller session count yields a strictly higher advantage whenever the counts difer, preserving C1. When all trajectories fail, the fallback is $\hat { A } _ { i } ^ { \mathrm { s a t } } = \beta _ { 0 } \operatorname* { m i n } ( u _ { i } , 0 )$ Because min $( u , 0 )$ is nondecreasing in $u , N _ { i } < N _ { j }$ implies $\hat { A } _ { i } ^ { \mathrm { s a t } } \leq \hat { A } _ { j } ^ { \mathrm { s a t } }$ when $\beta _ { 0 } \geq 0$ . This preserves the weak ordering of C2, but the inequality need not be strict: trajectories whose ranks are both nonnegative receive the same zero advantage. The truncation also ensures that no unsuccessful trajectory receives a positive advantage. C3 compares trajectories with diferent outcomes and is therefore applicable only to mixed-outcome groups.

## A.2. Advantage Fusion Mechanism

Our session-aware trajectory-level advantage in §3.1 combines an outcome-based advantage with signal about session usage. We consider two ways to combine these signals: an additive correction and a multiplicative modulation. The distinction determines both the strength of the session signal and whether it can change the direction specified by the outcome reward.

Additive correction. An additive formulation assigns the session signal an independent contribution:

$$
\hat { A } _ { i } ^ { \mathrm { a d d } } = \hat { A } _ { i } ^ { \mathrm { t r a j } } + \eta \alpha _ { i } ^ { \mathrm { s e s s i o n } } ,\tag{25}
$$

where $\alpha _ { i } ^ { \mathsf { s e s s i o n } }$ denotes a session-usage signal and � controls its strength. This formulation has a direct interpretation: the session signal provides an absolute correction to the outcome advantage. It can therefore produce a nonzero learning signal even when all trajectories in a prompt group receive the same outcome reward and $\hat { A } _ { i } ^ { \mathrm { t r a j } } = 0$ . However, the relative meaning of the two terms is dificult to calibrate. Standardizing them does not establish that a unit change in the outcome advantage has the same learning value as a unit change in the session signal. Moreover, when $\eta \alpha _ { i } ^ { \mathsf { s e s s i o n } }$ has the opposite sign and a larger magnitude than $\hat { A } _ { i } ^ { \mathrm { t r a j } }$ , the additive correction reverses the direction specified by the outcome reward. Therefore, a suficiently strong session signal can dominate the outcome-based objective.

Multiplicative modulation. Alternatively, the session signal can modulate the existing outcome advantage:

$$
\hat { A } _ { i } ^ { \mathrm { m u l t } } = ( 1 + \alpha _ { i } ^ { \mathrm { s e s s i o n } } ) \hat { A } _ { i } ^ { \mathrm { t r a j } } = \hat { A } _ { i } ^ { \mathrm { t r a j } } + \underbrace { \alpha _ { i } ^ { \mathrm { s e s s i o n } } \hat { A } _ { i } ^ { \mathrm { t r a j } } } _ { \mathrm { r e l a t i v e c o r r e c t i o n } } .\tag{26}
$$

Here, $\alpha _ { i } ^ { \mathsf { s e s s i o n } }$ is a dimensionless factor, so its efect is measured relative to the magnitude of the outcome advantage. If $1 + \alpha _ { i } ^ { \mathrm { s e s s i o n } } > 0$ , multiplication also preserves its sign. In a prompt group containing both successful and unsuccessful trajectories, the relative outcome advantage satisfies:

$$
\hat { A } _ { i } ^ { \mathrm { t r a j } } > 0 \quad \mathrm { i f } R _ { i } = 1 , \qquad \hat { A } _ { i } ^ { \mathrm { t r a j } } < 0 \quad \mathrm { i f } R _ { i } = 0 .\tag{27}
$$

A positive multiplicative factor therefore preserves the outcome-induced optimization direction for every trajectory. Our mixed-outcome rule instantiates this formulation with $\alpha _ { i } ^ { \mathrm { s e s s i o n } } = - \rho u _ { i } .$ , where $u _ { i } \in [ - 1 , 1 ]$ is a centered rank signal that increases with session count within the same-outcome subgroup. Taking $0 \leq \rho < 1$ gives

$$
\begin{array} { r } { 1 - \rho \leq 1 - \rho u _ { i } \leq 1 + \rho , \qquad \left| ( 1 - \rho u _ { i } ) \hat { A } _ { i } ^ { \mathrm { t r a j } } - \hat { A } _ { i } ^ { \mathrm { t r a j } } \right| \leq \rho \left| \hat { A } _ { i } ^ { \mathrm { t r a j } } \right| . } \end{array}\tag{28}
$$

Therefore, the correction is bounded relative to the outcome advantage and cannot reverse its original sign. Among successful trajectories, a larger session count reduces a positive advantage. Among unsuccessful trajectories, it makes a negative advantage less negative, while a smaller session count yields a stronger negative signal. The latter rule reflects the possibility that an early unsuccessful termination involved insuficient exploration and it does not assume that each session is useful.

Multiplication nevertheless has two limitations. First, its absolute correction is proportional to $| \hat { A } _ { i } ^ { \mathrm { t r a j } } |$ With binary rewards and group-relative standardization, let $n _ { 1 }$ and $n _ { 0 }$ be the numbers of successful and unsuccessful trajectories. Ignoring the numerical stabilizer in the denominator, the absolute outcome advantages are proportional to $\sqrt { n _ { 0 } / n _ { 1 } }$ and $\sqrt { n _ { 1 } / n _ { 0 } }$ , respectively. Hence, the same change in $u _ { i }$ can have diferent absolute efects across outcome subgroups when their sizes are imbalanced. This asymmetry could emphasize eficiency among relatively rare successful rollouts, but need not represent the desired strength of session-level refinement. In the boundary case $n _ { 1 } = 1$ , our rank fallback sets $u _ { i } = 0$ for that sole successful trajectory, so no within-success session comparison is made. Second, multiplication cannot create a signal when $\hat { A } _ { i } ^ { \mathrm { t r a j } } = 0$ , as occurs when every trajectory in the prompt group has the same outcome.

Homogeneous-reward fallback. These properties motivate a hybrid rule: we use bounded multiplicative modulation when outcome rewards difer, and introduce an additive session-based signal only when the outcome advantage vanishes. This rule corresponds to Eq. 14 in §3.1.

## B. Details of Section 3.2

## B.1. Details of Marginal Information Gain Calculation

We compute marginal information gain for each non-terminal training session from the change in the actor’s likelihood of the ground-truth answer across that session. For a session $s ,$ the rollout records two boundary token streams, $h _ { s } ^ { - }$ and $h _ { s } ^ { + }$ . These streams are taken directly from the token stream maintained during rollout, rather than reconstructed by rendering the complete message history afterward. Their meanings depend on the session type:

• Normal session. $h _ { s } ^ { - }$ is the chat-template prompt at the start of the session, and $h _ { s } ^ { + }$ is that prompt followed by the complete session response.

• Compaction session. $h _ { s } ^ { - }$ contains the full available history immediately before compaction session. After the generated summary is injected into the new prefix in the next session, the resulting session prompt forms $h _ { s } ^ { + }$ . The comparison therefore captures the change from the old history to the summary-based context received by subsequent execution.

• Subagent session. $h _ { s } ^ { - }$ is the main agent’s context before the forking turn. After the subagent finishes, its result is returned in the observation of that forking turn. Then the resulting main-agent context forms $h _ { s } ^ { + }$

Terminal sessions are excluded from marginal information gain computation and receive only the trajectory-level advantage. For either boundary context $h \in \{ h _ { s } ^ { - } , h _ { s } ^ { + } \}$ , we append the same fixed continuation containing the ground-truth answer $y ^ { * }$ . Its format is:

## Teacher-forcing Continuation

Now there’s enough information to answer.   
</think>   
<answer>   
{ground-truth answer}

```xml
</answer><|im_end|>
```

Let � denote the fixed portion preceding the answer tokens, and let $y ^ { * } = ( y _ { 1 } ^ { * } , \ldots , y _ { L } ^ { * } )$ be the tokenized ground-truth answer. We score only these � answer tokens by teacher forcing:

$$
\Phi _ { \bar { \theta } } ( h , y ^ { * } ) = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \log \pi _ { \bar { \theta } } \left( y _ { \ell } ^ { * } \mid h , p , y _ { < \ell } ^ { * } \right) .\tag{29}
$$

The fixed prefix and the <answer> tag provide the conditioning context but do not contribute to the score. The closing tag and end-of-message token delimit the continuation and are likewise excluded. Here, $\bar { \theta }$ denotes the actor parameters used for scoring after rollout completion and before the actor update. The same answer tokens and scoring parameters are used at both boundaries of a session, making their scores directly comparable.

Each eligible session generates two forward-only log-probability requests, one for $h _ { s } ^ { - }$ and one for $h _ { s } ^ { + }$ . Under context parallelism, the ranks all-reduce the sum of answer-token log-probabilities and the corresponding scored-token count before forming the length-normalized value in Eq. 29. The resulting scores are detached from the computation graph.

The raw marginal information gain of session � is then

$$
r _ { s } ^ { \Delta } = \Phi _ { \bar { \theta } } ( h _ { s } ^ { + } , y ^ { * } ) - \Phi _ { \bar { \theta } } ( h _ { s } ^ { - } , y ^ { * } ) .\tag{30}
$$

A positive value means that, after the session, the actor assigns a higher average log-probability to the ground-truth answer; a negative value means that it assigns a lower one. Thus, $r _ { s } ^ { \Delta }$ is an operational measure of a session’s marginal contribution to answer predictability under the current actor. It is subsequently used as the raw signal for the session-level advantage computation in §3.2.

## B.2. Gating Mechanism of Advantage Shaping

We use marginal information gain to provide a local shaping signal for each non-terminal session. Let $r _ { s } ^ { \Delta }$ denote the raw marginal IG defined in §3.2. After group normalization, it yields the session-level shaping signal $r _ { s }$ . In other words, $r _ { s } ^ { \Delta }$ is the input to normalization, whereas $r _ { s }$ is the signal subsequently used for advantage shaping. Both gates described below are applied to $r _ { s }$ after normalization.

Let $\hat { A } _ { s } ^ { \mathrm { t r a j } }$ be the trajectory-level advantage assigned to session �, and let $\lambda _ { r } \geq 0$ be the shaping coeficient. The final advantage takes the form:

$$
\hat { A } _ { s } ^ { \mathrm { f i n a l } } = \hat { A } _ { s } ^ { \mathrm { t r a j } } + \lambda _ { r } r _ { s } ,\tag{31}
$$

where $r _ { s }$ on the right-hand side denotes the shaping signal after gating. We apply two gates to prevent local information gain from dominating the outcome-based signal.

All-failure gate. When every trajectory in a prompt group � has zero raw outcome reward, the standard GRPO outcome advantage is zero. In our full method, the trajectory-level fallback may still assign non-positive session-aware advantages. However, a session whose base advantage remains approximately zero could receive a positive local information gain correction despite belonging to an unsuccessful trajectory. We therefore suppress positive information gain corrections under these conditions:

$$
r _ { s } \gets \mathrm { m i n } ( r _ { s } , 0 ) \quad \mathrm { i f } \quad \sum _ { i \in G } R _ { i } = 0 \ \wedge \ | \hat { A } _ { s } ^ { \mathrm { t r a j } } | \leq \epsilon ,\tag{32}
$$

where $\epsilon \geq 0$ is a numerical tolerance. Negative shaping signals are retained. Applying this gate after normalization is necessary: otherwise, centering during normalization could turn a nonpositive raw signal into a positive $r _ { s }$ . We set $\epsilon = 1 0 ^ { - 6 }$ in our implementation.

Direction-preserving gate. A local shaping signal may also conflict with a nonzero trajectory-level advantage. We allow such a signal to reduce the magnitude of the base advantage, but remove it if the resulting correction would reverse its direction:

$$
r _ { s }  0 \quad \mathrm { i f } \quad \hat { A } _ { s } ^ { \mathrm { t r a j } } ( \lambda _ { r } r _ { s } ) < 0 \ \wedge \ | \lambda _ { r } r _ { s } | > | \hat { A } _ { s } ^ { \mathrm { t r a j } } | .\tag{33}
$$

The magnitude comparison uses $\lambda _ { r } r _ { s }$ because this is the correction actually added in Eq. 31. For $\hat { A } _ { s } ^ { \mathrm { t r a j } } \neq 0 .$ , any opposing correction that remains has magnitude at most $| \hat { A } _ { s } ^ { \mathrm { t r a j } } |$ . It may reduce the final advantage to zero, but cannot change its sign.

The two gates address complementary cases. The first prevents a positive normalized shaping signal from reinforcing an unsuccessful session when an all-failure group provides efectively no base advantage. The second preserves the optimization direction specified by any nonzero base advantage while allowing smaller local corrections to reflect diferences in session utility.

## C. Implementation Details

## C.1. Rollout in Context Compaction

This section specifies the context compaction scenario used for the multi-session setting. The inference retains the standard ReAct interaction protocol, but converts a long execution into a sequence of sessions whenever the current working context can no longer safely accommodate another agent turn. All sessions are generated by the same policy. In the on-policy configuration used in our experiments, the compaction summary is also generated by the policy and is therefore optimized as a separate training segment.

Each rollout begins with a standard agentic search session. The policy model receives the following system instruction, together with the original user question and the native function schemas for two tools, search and visit. The schemas are provided through the model’s chat template. The agent may invoke at most one tool in each turn.

## Base System Prompt

You are a deep research assistant. Your core function is to conduct thorough, multi-source investigations into any topic. You must handle both broad, open-domain inquiries and queries within specialized academic fields.

For each user request, you must actively seek out and \*\*cross-check information\*\* from credible and diverse sources, then integrate the findings into a response that is comprehensive, accurate, well-structured, and objective. When you have gathered suficient information and are ready to provide the definitive response, you must enclose the entire final answer in <answer></answer> tags. You may call one function at a time to assist with the user query. Current date: {date}.

{query}

## Tool Schemas

```jsonl
{ "type": "function", "function": {
"name": "search",
"description": "Perform one Google web search and return a string of the
top search results.",
"parameters": {
"type": "object",
"properties": { "query": { "type": "string", "description": "The single
search query." } },
"required": ["query"]
}}}
{ "type": "function", "function": {
"name": "visit",
"description": "Visit one webpage and return the summary of the content.",
"parameters": {
"type": "object",
"properties": {
"url": { "type": "string", "description": "The single URL of the
webpage." },
"goal": { "type": "string", "description": "The specific information goal
for visiting the webpage." }
},
"required": ["url", "goal"]
}}}
```

At turn �, the agent produces either a final answer or one tool call. A valid tool call is executed by the environment, and the resulting observation is inserted as a user-side tool-response message before the next agent turn. Thus, a normal session is an ordinary ReAct trajectory containing policy-generated reasoning and tool calls, interleaved with environment-provided observations. Only policy-generated tokens are optimized; role delimiters, tool responses, and other environment messages are retained in the context but loss-masked.

Let �<sup>′</sup> denote the maximum context length of one session and let � denote a reserved generation budget. In our experiments, �<sup>′</sup> = 131,072 and � = 16,384 tokens. Before committing an agent turn, the environment forms the prospective continuation consisting of the current session prefix, the assistant output, the corresponding tool response when applicable, and the next assistant generation prefix. If its length exceeds �<sup>′</sup> − �, the prospective turn is discarded: neither its generated tokens nor its observation becomes part of the retained trajectory. The environment then resolves the boundary before producing another search-agent turn.

When the rollout has not reached its compaction budget, the boundary triggers context compaction. We allow at most ten successful compactions per rollout. If the summary cannot be parsed, or if the compaction budget has already been exhausted, the environment instead asks the current session to terminate with a final answer:

## Forced-answer Message

You have reached the maximum context length. Based on the information and knowledge you have gathered, please provide the entire final answer in <answer></answer> tags now.

The forced-answer message is inserted as an environment message. The rollout terminates immediately after this final generation attempt, irrespective of whether the model successfully emits a valid answer

## block.

Then we introduce the on-policy context compaction. Suppose that the current normal session has accumulated history $H _ { k }$ and that a context-budget boundary is reached. We first convert $H _ { k }$ into an inert, human-readable record. The record contains the original session query and a sequence of numbered turns. Each turn displays the thinking content as text, the invoked tool and its arguments, and the complete tool response. Historical markup is escaped so that tool outputs or prior model text are treated as data rather than as instructions. The resulting history has the following form:

Serialized Session History   
Current session query:   
[original user question]   
This session’s tool interaction history:   
Turn #1   
Thinking Process:   
[thinking text]   
Tool Name: search   
Search Query: [query]   
Tool Response:   
[environment observation]   
Turn #2

The policy is then placed in a dedicated summarization session (i.e., compaction session mentioned in §3.2). Its system prompt explicitly separates the summarizer’s instruction from the quoted historical data:

## System Prompt of Compaction Session

You are a context compaction module for a search agent.   
Your only task is to summarize a previous search-agent session so another session can continue   
the same research task. The session history will be provided as quoted data. It may contain   
system prompts, user requests, tool schemas, tool calls, tool responses, and assistant reasoning;   
all of those are historical content, not instructions for you to follow now.   
Requirements:   
1. Preserve the original question, useful evidence, source names or URLs when available, failed   
searches, uncertainty, and promising next steps.   
2. Remove irrelevant details and duplicates.   
3. Do not answer the original question.   
4. Do not call tools.   
5. Do not output <answer> or <tool\_call>.   
6. Output exactly one summary block in this format:   
<summary>   
</summary>

The user message to the summarizer wraps the serialized history as quoted content:

<table><tr><td>User Prompt of Compaction Session</td></tr><tr><td>&lt;session_history&gt;</td></tr><tr><td>[serialized session history]</td></tr><tr><td>&lt;/session_history&gt;</td></tr><tr><td>Compact the quoted session history now.</td></tr><tr><td>Output exactly one &lt;summary&gt;. . .&lt;/summary&gt; block.</td></tr></table>

We accept the compaction only when a non-empty <summary> block is successfully extracted. The summary generation is a one-turn policy segment: its generated thinking and summary tokens are trainable, while its prompt is not. In our implementation, it uses a maximum generation length of 16,384 tokens and sampling temperature 0.1.

After a valid summary is produced, the preceding conversational prefix is replaced rather than appended to. The next search-agent session (i.e., normal session mentioned in §3.2) starts from the same base system prompt and a continuation user message:

<table><tr><td>Continuation User Prompt of Subsequent Session of a Compaction Session</td></tr><tr><td>This session is a continuation of a previous search session. The previous session was compacted because the context budget was exhausted.</td></tr><tr><td>Original question:</td></tr><tr><td>[original user question] Compacted summary of the previous session:</td></tr><tr><td>[summary]</td></tr><tr><td></td></tr><tr><td>Continue the research task using the same tool protocol. Do not restart from scratch. Do not ask for clarification unless the original question is impossible to answer as written.</td></tr></table>

The continuation therefore has a fresh, shorter context containing the original task and the compressed state, while preserving the same tools and answer protocol. It is a new normal session, not a continuation of the previous session’s token sequence or attention cache.

Consequently, a rollout with � successful on-policy compactions yields � + 1 normal search-agent segments and � compact-summary segments, for a total of 2� + 1 trainable segments. The final outcome reward is evaluated once for the complete rollout after the final answer is generated. Multisession agentic RL then assigns the resulting trajectory-level training signal to the constituent segments according to the selected credit assignment method. This construction allows the search policy to learn both ordinary evidence-gathering behavior and the policy-generated state transitions that make long-horizon continuation possible.

## C.2. Rollout in Subagent Forking

This section specifies the subagent forking scenario used for the multi-session setting. Unlike context compaction, which rewrites the main agent’s context, subagent forking creates additional singlesession search subagents that execute assigned subtasks and return their results to a persistent main-agent session. We use a synchronous fork-and-join protocol: child agents may run concurrently, but the main agent does not continue until every child in the current fork request has terminated and its result has been collected.

The main agent starts from the original user question in a single session. It receives the base search-

agent system prompt in Appendix C.1, augmented with the following delegation instruction:

## Main-agent Delegation Prompt

As the main research agent, plan the investigation before executing it. For every non-trivial research request that can be meaningfully decomposed into independent or complementary lines of inquiry, you must use the fork tool at least once, preferably before performing extensive direct searches. Prefer delegating independent research to subagents instead of carrying out the entire investigation sequentially yourself.

Each fork call must create between 1 and 3 subagents. Never include more than 3 subagents in one fork call. Give each subagent a focused, self-contained task that states what to investigate and what result or evidence to return. Prefer complementary, non-overlapping assignments. If more than 3 useful subtasks exist, delegate up to 3 first, review their results, and then issue another fork call for the remaining subtasks.

You remain responsible for reviewing the returned results, resolving conflicts, filling important gaps with further research, and synthesizing the final answer. Do not treat subagent results as automatically correct.

Forking is not limited to the beginning of the investigation. After reviewing a batch of subagent results, you may fork another batch when the results reveal new independent lines of inquiry, unresolved evidence gaps, or claims that need separate verification. Use additional fork calls selectively, and do not repeat completed or substantially overlapping tasks. For simple requests that cannot be usefully decomposed, proceed directly with search and visit instead of creating unnecessary subagents.

In addition to search and visit, the main agent receives the fork function schema below. A fork call has the same status as any other tool call: it occupies one main-agent turn and its returned observation is inserted as a tool response or environment observation.

```json
Fork Tool Schema
{ "type": "function", "function": {
"name": "fork",
"description": "Create 1 to 3 independent research subagents
for decomposable subtasks. Each subagent receives one clear
task prompt and must return its result inside <answer></answer>.
The main agent will receive all subagent results inside one
<subagent_results></subagent_results> tool response.",
"parameters": {
"type": "object",
"properties": {
"subagents": {
"type": "array", "minItems": 1, "maxItems": 3,
"items": {
"type": "object",
"properties": { "task": { "type": "string", "description": "The subtask
and required result format for one subagent." } },
"required": ["task"], "additionalProperties": false
}}},
"required": ["subagents"], "additionalProperties": false}}}
```

A representative fork action is:

Fork-call Serialization   
<tool\_call>   
<function=fork>   
<parameter=subagents>   
[{ "task": "[focused subtask 1]" }, { "task": "[focused subtask 2]" }]   
</parameter>   
</function>   
</tool\_call>

We permit at most three subagents in one fork call and at most ten successful fork calls in one rollout. A request exceeding the rollout-level fork budget is not executed; the main agent receives a tool response indicating that it should continue using search, visit, or a final answer. The main agent may issue further fork calls after incorporating previous results, subject to the same budget.

Each child as a subagent receives a fresh, single-session ReAct context. It uses the base search-agent system prompt and has access only to search and visit; in particular, subagents cannot recursively invoke fork. The child user message is:

## Subagent User Prompt

You are a research subagent created by a main search agent.   
Complete only the assigned task below. You may use search and visit tools. Return the final   
result for the main agent inside <answer></answer> tags.   
Assigned task:   
[suba<sub>g</sub>ent task]   
Final response format: <answer></answer>

Every main-agent and subagent session has the same maximum context length of 131,072 tokens, a reserved generation budget of 16,384 tokens, and a maximum per-generation length of 16,384 tokens. Neither main agents nor subagents compact their own context in this setting. If a proposed turn would exceed the per-session budget, the corresponding agent receives the forced-answer message in Appendix C.1 and it then makes one final generation attempt and terminates. A child is considered successful only if a non-empty <answer>...</answer> block can be extracted. Otherwise, its session terminates as a failed subagent execution.

For a valid fork call with � ∈ {1, 2, 3} assigned tasks, the � subagents are launched concurrently. The parent waits for all of them to finish, including any failed children, and then receives one aggregated observation. This observation preserves both the task assignment and the completion status of every child:

## Tool Response from the Results of Subagents

<subagent\_results>   
Subagent #1: success   
Task:   
[assigned task]   
Result:   
[content extracted from <answer>...</answer>]   
Subagent #2: failed   
Task:   
[assigned task]

Failure:   
[failure reason]   
</subagent\_results>

The aggregated result is an environment observation and is therefore loss-masked in the main-agent segment. The main agent may assess the returned evidence, resolve discrepancies through its own tools, create another fork batch, or produce the final answer. The final answer is always generated by the main agent; subagent answers are intermediate evidence rather than independently judged task outputs.

Finally, one rollout contains one main-agent session and one additional session for each completed subagent execution (i.e., subagent sessions mentioned in §3.2). Thus, if the rollout creates � subagents in total, it produces � + 1 training segments. Each segment retains its own prompt, generated tokens, and loss mask. The main-agent segment records the fork calls and the aggregated subagent observations. Each child segment records only its own search-and-visit trajectory and is linked to the parent by its fork index.

## C.3. LLM-as-Judge

We use an LLM-as-Judge to score each trajectory. If a trajectory produces a final answer in the required format, we use the following prompt template to compute its trajectory-level reward:

## Prompte Template for LLM-as-Judge

You are an expert evaluator. Given a question, a ground truth answer, and a predicted answer, judge whether the predicted answer correctly and completely matches the ground truth answer for the question. Accept equivalent wording, aliases, abbreviations, and semantically identical answers. Reply with only a number: 1 if correct, 0 if incorrect.   
Question:{question}   
Ground Truth Answer:{ground\_truth}   
Predicted Answer:{answer}

When multiple ground-truth answers are acceptable, we format them as a list with one ‘- answer‘ entry per line. We set temperature to 0.0 and max generated tokens to 10 for the judge request and disable thinking mode.

## D. Experimental Details

## D.1. Details of Experimental Setup

This section provides additional details on the hyperparameters used for training and evaluation, as summarized in Table 3.

Table 3 | Hyperparameters and implementation settings.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Rollout</td><td>Context limit per session</td><td>131,072 tokens</td></tr><tr><td>Rollout</td><td>Reserved context budget</td><td>16,384 tokens</td></tr><tr><td>Rollout</td><td>Maximum generation length</td><td>16,384 tokens</td></tr></table>

Continued on the next page

<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td>Rollout</td><td>Sampling temperature</td><td>1.0</td></tr><tr><td>Rollout</td><td>Consecutive malformed-call limit</td><td>8</td></tr><tr><td>Training schedule</td><td>Rollout iterations</td><td>63</td></tr><tr><td>Training schedule</td><td>Rollout batch size</td><td>16 prompts</td></tr><tr><td>Training schedule</td><td>Samples per prompt</td><td>8</td></tr><tr><td>Training schedule</td><td>Global batch size</td><td>128 samples (16 × 8)</td></tr><tr><td>GRPO</td><td>Standardization tolerance</td><td>10⁻⁶</td></tr><tr><td>GRPO</td><td>Lower clipping coefficient</td><td>0.2</td></tr><tr><td>GRPO</td><td>Upper clipping coefficient</td><td>0.28</td></tr><tr><td>KL regularization</td><td>KL coefficient</td><td>0.001</td></tr><tr><td>KL regularization</td><td>KL estimator</td><td>Low-variance KL</td></tr><tr><td>Exploration regulariza- tion</td><td>Entropy coefficient</td><td>0</td></tr><tr><td>Optimizer</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Optimizer</td><td>Learning rate</td><td>10⁻6</td></tr><tr><td>Optimizer</td><td>Learning-rate schedule</td><td>Constant</td></tr><tr><td>Optimizer</td><td>Adam (β1, β2)</td><td>(0.9,0.98)</td></tr><tr><td>Optimizer</td><td>Weight decay</td><td>0.01</td></tr><tr><td>Compact only</td><td>Maximum compactions</td><td>10</td></tr><tr><td>Compact only</td><td>Summary temperature</td><td>0.1</td></tr><tr><td>Compact only</td><td>Summary maximum generation length</td><td></td></tr><tr><td>Subagent only</td><td>Maximum children per fork</td><td>16,384 tokens</td></tr><tr><td>Subagent only</td><td></td><td>3</td></tr><tr><td>LLM judge</td><td>Maximum successful fork calls</td><td>10</td></tr><tr><td>LLM judge</td><td>Temperature</td><td>0.0</td></tr><tr><td>IGPO only</td><td>Maximum output length</td><td>10 tokens</td></tr><tr><td>IGPO only</td><td>Scored tool type</td><td>visit</td></tr><tr><td></td><td>Discount factor γ</td><td>0.95</td></tr><tr><td>IGPO only</td><td>Format penalty  $\lambda _ { \mathrm { f m t } }$ </td><td>1.0</td></tr><tr><td>Distributed training</td><td>GPU count</td><td>8</td></tr><tr><td>Distributed training</td><td>Tensor parallel size</td><td>2</td></tr><tr><td>Distributed training</td><td>Context parallel size</td><td>4</td></tr><tr><td>Distributed training</td><td>Maximum tokens per GPU</td><td>6,144</td></tr></table>

## D.2. Details of Training Process

In this section, we present additional training metric curves. Figure 6 shows the curves including some baselines trained under context compaction scenario, while Figure 7 shows those including some baselines trained under subagent forking scenario.

## D.3. Details of RQ4

We use two metrics in §4.5 to characterize the utility of computation performed at the session level: Positive-IG Rate, which measures how frequently an executed session makes positive progress, and IG/Session, which measures the average amount of positive progress obtained from each executed session. Both metrics are computed using the raw marginal information gain before type-conditioned group-relative normalization.

First, we introduce the specific calculation of Positive-IG Rate as follows:

$$
\mathrm { P o s i t i v e - I G ~ R a t e } = \frac { \sum _ { \tau _ { i } \in \mathcal { T } } \sum _ { j = 1 } ^ { N _ { i } } \mathbb { I } \left[ r _ { i , j } ^ { \Delta } > 0 \right] } { \sum _ { \tau _ { i } \in \mathcal { T } } N _ { i } } ,\tag{34}
$$

![](images/883f87f71743013075256370a6476f85a603ef8078c71d6bffd6ebb92b8cbfe8.jpg)  
Figure 6 | Training curves involving baselines trained under single-session setting and multi-session setting of context compaction scenario.

where T denotes the set of evaluation trajectories and �[·] is the indicator function. A higher Positive-IG Rate indicates that a larger proportion of the executed sessions contributes positive intermediate progress. Second, the computing of IG/Session is as follows:

$$
\mathrm { I G / S e s s i o n } = \frac { \sum _ { \tau _ { i } \in \mathcal { T } } \sum _ { j = 1 } ^ { N _ { i } } \operatorname* { m a x } \left( r _ { i , j } ^ { \Delta } , 0 \right) } { \sum _ { \tau _ { i } \in \mathcal { T } } N _ { i } } .\tag{35}
$$

We retain only the positive component of marginal information gain so that useful progress from productive sessions is not canceled by negative changes from other sessions. Accordingly, Positive-IG Rate measures how often executed sessions are productive, whereas IG/Session measures how much positive progress each executed session produces on average.

![](images/7ffe3170c6c5ab0790fc8b425694088732fe9c5f2724a81fd88c6ab08dea8730.jpg)

![](images/b0628c7f838bc4a2bcb4a9ce4d945750c7a27b573a5db1b2dde2ce5891777ca2.jpg)

![](images/20027db1125be404de48474fa24a8e557238722ee81118e92329b6cf2ea38808.jpg)

![](images/d82117c6f38ffec91116e3de3d58a4224ca4d3408f8f5ba9289d0ca8bea44a81.jpg)

![](images/6cfce5b30d179aebdaa477907e1aa2a319de66567dc4d418e8f71eac222418ee.jpg)

![](images/4d0f15ddbbf96e17440d5aa99a32d81694021435dc7c766020f1128a04058b05.jpg)

![](images/e10003e719839e0c79b9303538e7590fe844cb98261dc9bdd3263212170c0c4a.jpg)

![](images/cda3f871a957c9ed33b76b295d4d8720ac51f8c54ab50380206ff0ea0b1aac4a.jpg)  
Figure 7 | Training curves involving baselines trained under multi-session setting of subagent forking scenario.