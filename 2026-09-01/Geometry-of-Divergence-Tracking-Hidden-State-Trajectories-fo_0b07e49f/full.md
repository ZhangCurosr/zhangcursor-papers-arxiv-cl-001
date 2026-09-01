# Geometry of Divergence: Tracking Hidden-State Trajectories for Adaptive Multi-Turn Reasoning

Jie Liang1, Zhengxin Yu1\*, Hamid Nasiri2, Peter Garraghan

1School of Computing and Communications, Lancaster University, UK 2School of Computing and Engineering, University of Huddersfield, UK {j.liang14, z.yu8, p.garraghan} @lancaster.ac.uk, h.nasiri@hud.ac.uk

## Abstract

LLM agents need to sustain goal-consistent reasoning across long multi-turn interactions under strict resource constraints. However, as the multi-turn context accumulates, it can destabilize the underlying LLM's internal representation of taskrelevant information from earlier turns, blurring the boundary between constructive reasoning and representation drift. We formulate multi-turn reasoning as a hidden-state trajectory of the underlying LLM that is characterized via two complementary signals: temporal curvature that captures the directional consistency of turn-to-turn updates, and variance slope which measures the expansion or contraction of the exploration space. Across four tasks and three underlying LLMs, we observed that these geometric signals distinguish between correct and incorrect episodes prior to completion. We further decompose each episode into three-action chains formed from four actions (Read, Write, Respond, Transfer) and show that separability is action-dependent, with different signals distinguishing various chain patterns. Our experiments demonstrate that trajectory geometry can identify critical turns in the reasoning process, increasing task success rates on τ-Bench from 24.1% to 39.6% while reducing token cost by 11.2%.

## 1 Introduction

Recent advances in Large Language Models (LLMs) have accelerated a paradigm shift toward autonomous LLM agents that commonly operate through multi-turn interactions. Rather than fully specifying their task objectives at the outset, users refine their requirements over multiple turns by adding details and constraints based on the agent's intermediate feedback (Laban et al. 2025). Multi-turn interaction can therefore be viewed as an incremental reasoning process that reduces epistemic uncertainty about the task. In resourceconstrained settings (e.g., limited model size and finite context windows), accumulating interaction history can lead to context saturation and reduce attention efficiency (Liu et al. 2024). Maintaining long-horizon, goal-consistent reasoning under such resource constraints has emerged as a fundamental challenge in LLM agent research.

Existing approaches primarily rely on explicit experiencebased heuristic methods of context engineering such as planning, reflection loop, and memory architecture to constrain the agent's inference trajectory and improve reasoning over long multi-turn interactions (Mei et al. 2025). However, these methods do not resolve the mismatch between agent goals and the utilized LLM's objective. Agents are designed to complete tasks through multi-turn interactions, whereas their underlying LLMs are optimized to generate the best possible response at each turn. Existing Large Reasoning Models (LRMs) also use a fixed level of reasoning effort during dialogue, without adapting it to the needs of each turn (Zhao et al. 2026). Unnecessary reasoning can introduce semantic noise and irrelevant counterfactuals into the autoregressive history, corrupting the agent's internal representation of accumulated constraints.

![](images/8f352c342945914b1e2eda1b3da7665d20b39a0036f232f030c829e7ae2f77a0.jpg)  
Figure 1: Multi-turn reasoning as an observable hidden-state trajectory.

Diagnosing and mitigating this internal corruption requires tracking how the underlying LLM's hidden states change across turns. Capturing the hidden-state trajectory is more difficult in multi-turn interactions compared to singleturn or basic Chain-of-Thought (CoT) reasoning (Wang et al. 2025). Each hidden state update reflects both current task reasoning and the accumulated influence of earlier turns. This makes it difficult to distinguish useful hidden state updates from representation drift, and to determine when intervention is needed. Therefore, identifying the precise inflection points at which hidden states begin to deviate from the core objective is key to engineering reliable agents.

To address this challenge, we formulate multi-turn reasoning as an observable hidden-state trajectory of the underlying LLM and, as illustrated in Fig.1, track where correct and incorrect episodes diverge in order to decide when to intervene, triggering adaptive reasoning precisely at the turns where failure is imminent. Our contributions are threefold:

• We characterize the trajectory with two complementary geometric signals, temporal curvature and variance slope. Across four tasks spanning open-ended reasoning (Math, Code) and tool-grounded interaction (Airline, Retail), and three underlying LLMs (Qwen3-14B, Qwen3-32B and Llama-3.1-8B), failure trajectories exhibit directional reversals and premature convergence in their hidden states.

• We probe the source of this divergence by decomposing episodes into three-action chains, and find that separability is action-dependent, with information-output chains carrying the strongest discriminative structure while information-acquisition chains carry little.

• We further demonstrate that these geometric signals can drive online control. By triggering adaptive reasoning at the turns where failure is imminent, our geometryconditioned policy raises task success rates on τ-Bench from 24.1% to 39.6% while reducing token cost by 11.2%.

## 2 Related Work

## 2.1 Multi-Turn Degradation in LLM Agents

LLMs that perform well in single-turn settings degrade when task specifications are revealed incrementally. Laban et al. (Laban et al. 2025) report an average performance drop of 39% across six generation tasks, attributing it to premature answer attempts and over-reliance on earlier, potentially erroneous turns. Realistic agentic settings amplify this problem. On τ-Bench (Yao et al. 2024) and its dual-control successor $\tau ^ { 2 } .$ -Bench (Barres et al. 2025), even frontier function-calling agents solve fewer than half of the tasks and behave inconsistently across trials. To sustain long-horizon coherence, prior work engineers the explicit context, e.g., through verbal selfreflection (Shinn et al. 2023), OS-inspired memory management (Packer et al. 2023), and the context-engineering toolbox surveyed by Mei et al. (Mei et al. 2025). These approaches operate on the textual history itself. In contrast, we monitor the latent trajectory induced by that history and intervene only when its geometry indicates drift.

## 2.2 Hidden States Monitoring

LLM internal states contain information predictive of output quality. Early work demonstrated that models can assess the validity of their own claims (Kadavath et al. 2022). Truthfulness is also linearly decodable from hidden states, with simple probes distinguishing true from false statements (Azaria and Mitchell 2023; Marks and Tegmark 2023). At inference time, shifting activations along truth-correlated directions can further improve factuality (Li et al. 2023).

Building on these findings, internal representations have been widely used for hallucination detection through the covariance spectrum of response embeddings (Chen et al. 2024a), probes over generation-time hidden states (Duan, Yang, and Tam 2024), and attention-map features (Chuang et al. 2024). Semantic entropy offers a complementary sampling-based uncertainty signal (Farquhar et al. 2024) that can be approximated efficiently from the hidden states of a single generation (Kossen et al. 2024). Orgad et al. (Orgad et al. 2025) show that truthfulness information is concentrated in specific tokens and predicts the type of upcoming error. However, these signals do not transfer universally across tasks, which is consistent with the task-dependent behavior we observe for the variance slope.

Among these methods, the ICR Probe is most closely related to our work. It tracks the cross-layer dynamics of hidden-state updates for hallucination detection (Zhang et al. 2025b), but analyzes individual responses in isolation. We characterize hidden-state trajectories across turns using temporal curvature and variance slope, and use their geometry as an online control signal rather than a post-hoc detector.

## 2.3 Adaptive Reasoning Control

Extended chain-of-thought is not uniformly beneficial, as reasoning models overthink simple problems (Chen et al. 2024b). In agentic environments, excessive deliberation can also displace environment interaction and lower task success (Cuadron et al. 2025). Recent methods address this problem by selecting between thinking and no-thinking modes using reinforcement learning (Zhang et al. 2025a; Fang, Ma, and Wang 2026) or Pareto-optimal chain-of-thought triggering (Lou et al. 2025). These methods are derived from the input question in largely single-turn settings. In contrast, our framework conditions the trigger on the agent's internal state accumulated across turns.

## 3 Hidden State Trajectory Formulation

## 3.1 Multi-Turn Reasoning in Agent Tasks

We begin by formalizing multi-turn reasoning in realistic agent settings. A typical agent multi-turn interaction comprises three components: (1) a system prompt $c _ { \mathrm { s p } }$ , which encodes the task specification and agent behavioral constraints; (2) a sequence of user inputs $\mathcal { Q } = \{ q _ { 0 } , q _ { 1 } , . . . , q _ { T - 1 } \}$ , representing $T$ alternating turns of interaction; and (3) a sequence of model outputs $\bar { Y ^ { } } = \{ y _ { 1 } , y _ { 2 } , \dots , y _ { T } \}$ generated by a language model parameterized by $\theta ,$ where each $y _ { t + 1 }$ is the output produced at turn t conditioned on the system prompt and all preceding turns. The generation is formalized as the conditional distribution

$$
P _ { \theta } ( Y \mid c _ { \mathrm { s p } } , \mathcal { Q } ) = \prod _ { t = 0 } ^ { T - 1 } P _ { \theta } ( y _ { t + 1 } \mid c _ { \mathrm { s p } } , y _ { \le t } , q _ { \le t } ) .\tag{1}
$$

In agent settings, the agent can also interact with the external environment $\cdot \theta ^ { \prime }$ throughout the interaction. At each turn $t ,$ alongside its user-facing output, the agent may issue a tool invocation $y _ { t } ^ { \prime } .$ yielding a response $a _ { t } = \theta ^ { \prime } ( y _ { t } ^ { \prime } )$ that is appended to the context and conditions all subsequent turns. We denote the set of such external interactions by $\mathcal { A } = \{ a _ { t } = \theta ^ { \prime } ( y _ { t } ^ { \prime } ) \}$ The full generation process thus becomes

$$
P _ { \theta } ( Y \mid c _ { \mathrm { s p } } , { \cal A } , { \cal Q } ) = \prod _ { t = 0 } ^ { T - 1 } P _ { \theta } ( y _ { t + 1 } \mid c _ { \mathrm { s p } } , a _ { \le t } , y _ { \le t } , q _ { \le t } ) .\tag{2}
$$

## 3.2 Hidden States in Multi-Turn Reasoning

For each turn t, let $h _ { t } \in \mathbb { R } ^ { d }$ denote the language model's hidden states at the final token position of the user input $q _ { t } ,$ taken from a fixed intermediate probe layer (layer 22 for Qwen3-14B, layer 42 for Qwen3-32B, and layer 16 for Llama-3.1-8B). Under causal attention with residual connections, this representation encodes the full contextual semantics up to turn t; its projection onto the vocabulary determines the next-token distribution. We collect these representations into the trajectory

$$
{ \mathcal { H } } = \{ h _ { 0 } , h _ { 1 } , . . . , h _ { T - 1 } \} .\tag{3}
$$

The inter-turn dynamics are modeled as a residual process over turns

$$
h _ { t } = h _ { t - 1 } + \mathbf { v } _ { t } , \quad t = 1 , \ldots , T - 1 ,\tag{4}
$$

where the displacement $\mathbf { v } _ { t }$ represents the change in the representation of the hidden state after turn t relative to the accumulated context of all preceding turns.

## 3.3 Temporal Curvature

To measure the directional consistency of semantic drift across consecutive turns, we define the temporal curvature $\kappa _ { t }$ at turn t from the current hidden state $h _ { t }$ and its two predecessors $h _ { t - 1 } , h _ { t - 2 }$ , as the cosine of the angle between the two consecutive displacement vectors

$$
\kappa _ { t } = { \frac { \mathbf { v } _ { t } \cdot \mathbf { v } _ { t - 1 } } { \| \mathbf { v } _ { t } \| _ { 2 } \| \mathbf { v } _ { t - 1 } \| _ { 2 } } } , \quad t = 2 , \ldots , T - 1 ,\tag{5}
$$

where $\mathbf { v } _ { t } = h _ { t } - h _ { t - 1 }$ and $\mathbf { v } _ { t - 1 } = h _ { t - 1 } - h _ { t - 2 } ,$ defined whenever $\mathbf { v } _ { t } \neq \mathbf { 0 }$ and $\mathbf { v } _ { t - 1 } \neq \mathbf { 0 }$ . The measure takes values in [—1, 1]. Values near 1 indicate that context accumulates in a consistent direction, values near 0 indicate orthogonal, non-reversing updates, whereas values near —1 indicate an abrupt reversal of the internal trajectory.

## 3.4 Variance Slope

For a prefix $\{ h _ { 0 } , \ldots , h _ { \lambda } \}$ containing $\lambda + 1$ hidden states, we define its dispersion $\overset { \cdot } { V } ( \lambda )$ as the trace of the sample covariance, using an unbiased estimator

$$
\begin{array} { l } { { \displaystyle V ( \lambda ) = \mathrm { t r } \big ( \mathrm { C o v } ( h _ { 0 } , \ldots , h _ { \lambda } ) \big ) } } \\ { { \displaystyle \quad = \frac { 1 } { \lambda } \sum _ { i = 0 } ^ { \lambda } \big \| h _ { i } - \overline { { h } } ^ { ( \lambda ) } \big \| _ { 2 } ^ { 2 } , \quad \lambda \ge 1 , } } \end{array}\tag{6}
$$

$$
\overline { { { h } } } ^ { ( \lambda ) } = \frac { 1 } { \lambda + 1 } \sum _ { i = 0 } ^ { \lambda } h _ { i } ,\tag{7}
$$

where $\operatorname { t r } ( \operatorname { C o v } ( \cdot ) )$ is the sum of per-coordinate variances, and the normalization by λ (rather than $\lambda + 1 )$ reflects the unbiased estimator over the $\lambda + 1$ prefix points. Varying λ from 1 to t gives a dispersion sequence $\bar { \{ V ( 1 ) , \ldots , \bar { V ( t ) } \} }$ and we define the variance slope $\beta _ { t }$ as its ordinary leastsquares slope against prefix length

$$
\beta _ { t } = \frac { \sum _ { \lambda = 1 } ^ { t } \left( \lambda - \bar { \lambda } \right) \left( V ( \lambda ) - \bar { V } \right) } { \sum _ { \lambda = 1 } ^ { t } \left( \lambda - \bar { \lambda } \right) ^ { 2 } } , \quad t = 2 , \ldots , T - 1 ,\tag{8}
$$

where λ and $\bar { V }$ are the means of the prefix lengths and dispersion values. A positive $\beta _ { t }$ indicates that representations progressively spread out as turns accumulate, whereas a negative value indicates a contracting dispersion.

## 4 Empirical Analysis

## 4.1 Experimental Setup

Benchmarks Our experiments comprise four tasks across two multi-turn interaction scenarios. The Math and Code tasks are drawn from the sharded-prompt framework of Lost in Conversation (Laban et al. 2025), which demonstrates how an incorrect early assumption can derail a multi-turn conversation, inducing context contamination and progressively degrading reasoning efficiency. The Airline and Retail tasks are taken from τ-Bench (Yao et al. 2024), which simulates dynamic dialogue between a user and a LLM agent equipped with domain-specific API tools and policy guidelines. Together, these benchmarks cover both open-ended reasoning (Math, Code) and tool-grounded, policy-constrained interaction (Airline, Retail).

Models We employ three LLMs of different sizes as the core components of both the assistant agent and the user simulator: Qwen3-14B, Qwen3-32B (Qwen Team 2025), both LRMs with a toggleable thinking mode from long-CoT training, and Llama-3.1-8B (Grattafiori et al. 2024).

Metrics Both benchmarks provide ground-truth reference answers, from which we derive a binary correctness label for each individual task. Aggregating tasks yields the average reward, which we report at the domain level. To characterize the overhead of multi-turn reasoning, we record the true token cost of each task through the vLLM interface, accounting for context, tool calls, and thinking overhead. Significance is assessed with the two-sided Mann-Whitney U test (p-values).

Implementation Details We conduct inference for LLMs using vLLM on 8 NVIDIA L40S GPUs and 4 H200 GPUs. All LLMs' decoding hyper-parameters and prompt formatting strictly follow their respective official configurations. Details on benchmark selection for each analysis and the full experimental setup are provided in the Appendix.

## 4.2 Temporal Curvature Analysis

We compare temporal curvature κ across four task types and three underlying LLMs. In Fig.2(a), the horizontal axis denotes each task-LLM pairing, and the vertical axis is the per-episode mean of κ. We make three main observations.

κ medians are negative for most task-model pairings. κ characterizes the directional consistency of semantic drift between consecutive turns. A value of $\kappa < 0$ indicates that a hidden-state update tends to reverse the previous turn's residual direction, which we interpret as consistent with a multi-turn reasoning process that produces an update that points in the opposite direction to the preceding update. The Code correct group has a positive median (median = +0.127, mean $= + 0 . 0 9 6 )$ , suggesting that on structured, verifiable tasks hidden-state updates lean toward same-direction accumulation rather than reversal.

![](images/13fb3e8fc21d25fcbd635f822e7b48e0fd186b50f2339084e2fe8a64032c1198.jpg)  
(a) Temporal curvature κ: correct trajectories bend less.

![](images/3d33e3e3a962c11f01565067a3a5a0d105270e0485681f6162bf16cc2ed03e6f.jpg)  
(b) Variance slope $\beta _ { t }$ , z-scored within task.  
Figure 2: Trajectory geometry separates correct from incorrect episodes across task and model settings.

Failed episodes exhibit systematically more negative $\kappa _ { \bullet }$ Across all six pairings, the mean of the failure group is consistently lower than that of the corresponding correct group, with differences (Correct—Incorrect) of: Math +0.047, Code +0.148, Retail (14B) +0.030, Airline (14B) +0.049, Retail (32B) +0.035, Airline (32B) +0.088. This indicates that failure trajectories undergo directional reversals in hidden-state updates more frequently, whereas correct trajectories have κ closer to zero, and in some cases positive (e.g., the Code correct group), reflecting a tendency toward orthogonal or same-direction information accumulation.

## 4.3 Variance Slope Analysis

In Fig.2(b), the vertical axis is the variance slope $\beta _ { t }$ standardized to zero mean and unit variance within each task. Recall that a positive $\beta _ { t }$ indicates that the total dispersion of the hidden-state trajectory increases as the episode progresses, whereas a negative value indicates that it contracts.

Correct episodes exhibit higher variance slopes than incorrect episodes. Across all six pairings, correct episodes have higher variance slopes than incorrect episodes. Correct trajectories maintain a broader exploration space across turns, whereas failed trajectories contract onto a narrower, unproductive region. The gap is largest on the shardedframework tasks Math and Code and smallest on Airline. Separability is significant for all pairings except the two Airline cases (14B p=0.30, 32B p=0.057), tracking the increasing openness of the action space from structured tasks through Retail to Airline.

## 4.4 Turn Alignment

One methodological point deserves emphasis: all hidden states $h _ { t }$ entering the curvature and variance-slope computations are turn-aligned. This alignment matters because trajectory length is a confound we observed directly, as incorrect episodes are systematically longer than successful ones due to retries, stalls, and back-and-forth entanglement, and a longer trajectory is mechanically more curved and accumulates more variance regardless of correctness. Indexing by turn removes this confound. Specifically, for the Math (as shown in Fig.3), Code, and τ-Bench datasets we index turns from 0 and take one $h _ { t }$ per turn, the hidden state at the final token of the user input $q _ { t }$ before the agent acts, then compute κ and $\beta _ { t }$ on the resulting sequence $\{ h _ { 0 } , h _ { 1 } , . . . . \}$ of consecutive states. Because temporal curvature is a threepoint statistic over $h _ { t - 2 } , h _ { t - 1 } , h _ { t }$ , values are undefined for the first two turns and available only from turn 2 onward, and we adopt the same alignment for the variance slope so the two metrics are defined on identical turn indices.

![](images/08290c2f39da09c44b95f90023fcd912ed330c16270a7f15332acd39599849e7.jpg)  
Figure 3: Turn-resolved $\kappa _ { t }$ and $\beta _ { t }$ on Math tasks.

## 5 Probing Trajectory Divergence through Action Chains

## 5.1 Action Chain Construction

Based on the types of actions through which agents interact with the environment in τ-Bench, we classify action chains into four categories:

• Write refers to write-type tool calls that alter the database state $( e . g .$ , update reservation, cancel pending order).

• Read refers to read-only tool calls that query the environment without altering its state (e.g., get reservation details, get user details).

• Respond refers to replies to the user simulator.

• Transfer refers calls to transfer\_to\_human, which end the episode by escalating to a human agent.

![](images/72d1a0d067b9bd74ba5cc5693c3858819045dd1e20649eb62ffb3e6412e9d63a.jpg)  
(a) Effect sizes per action chain (Retail).

![](images/74c3b7a7b8e108c8774bb16a6d9829baf2f4b765aef1ec946d5e6ef16bfbdb7c.jpg)  
(b) Effect sizes per action chain (Airline).

![](images/6b88c055be20150b3bc7f4640ddb6239698409075d0d890696225b6c9aabf392.jpg)  
(c) PCA trajectory of the Respond ×3 chain.  
Figure 4: Discriminability of hidden-state trajectory geometry across action chains.

We construct action chains by taking sliding windows of three consecutive turns over the recorded assistant actions. Table 1 reports the proportion of three-action chains containing at least one such action in both the correct and incorrect episodes within each domain.

Table 1: Share of three-action chains containing each action.
<table><tr><td rowspan="2">Action</td><td colspan="3">Retail</td><td colspan="3">Airline</td></tr><tr><td>Correct</td><td>Incorrect</td><td>Δ</td><td>Correct</td><td>Incorrect</td><td>Δ</td></tr><tr><td>Write</td><td>49.3%</td><td>43.4%</td><td>-5.9</td><td>15.3%</td><td>38.9%</td><td>+23.7</td></tr><tr><td>Read</td><td>85.8%</td><td>82.5%</td><td>-3.3</td><td>86.1%</td><td>64.4%</td><td>-21.7</td></tr><tr><td>Transfer</td><td>4.2%</td><td>4.4%</td><td>+0.2</td><td>29.2%</td><td>7.5%</td><td>-21.7</td></tr><tr><td>Respond</td><td>57.4%</td><td>50.0%</td><td>-7.4</td><td>58.3%</td><td>61.7%</td><td>+3.4</td></tr></table>

## 5.2 Separability of Hidden-state Trajectories in Action Chains

Hidden-state trajectory separability varies with the distribution of agent actions across tasks.

As shown in Table1 and Fig.4a and Fig.4b, the discriminative capabilities of the two probes exhibit systematic differences across the two task categories. In the Retail task, differences in domain action distributions between correct and incorrect episodes are relatively small. The discriminative signal primarily comes from the variance slope. Temporal curvature provides comparatively limited separation. Conversely, in the Airline task, differences in domain action distributions between correct and incorrect episodes are larger. Variance slope shows stronger discriminative power than in Retail, while temporal curvature also demonstrates marked discriminative capability. This comparison shows that temporal curvature and variance slope are not redundant probes, but rather provide complementary signals whose relative discriminative power varies across tasks.

Table 2: Top-20 three-action chains, with correct/incorrect window shares (%) and per-domain frequency rank.
<table><tr><td colspan="2"></td><td colspan="2">Retail</td><td colspan="2">Airline</td></tr><tr><td>IDs</td><td>Action Chains</td><td>Cor/Inc</td><td>Rank</td><td>Cor/Inc</td><td>Rank</td></tr><tr><td>1</td><td>Read→Read→Read</td><td>19.6/22.2</td><td>#1</td><td>17.1/14.5</td><td>#2</td></tr><tr><td>2</td><td>Resp→Resp→Resp</td><td>3.6/8.2</td><td>#5</td><td>9.1/23.6</td><td>#1</td></tr><tr><td>3</td><td>Read→Read→Write</td><td>12.5/10.8</td><td>#2</td><td>3.2/8.0</td><td>#3</td></tr><tr><td>4</td><td>Resp→Read→Read</td><td>11.2/9.6</td><td>#3</td><td>9.3/5.1</td><td>#4</td></tr><tr><td>5</td><td>Read→Write→Resp</td><td>10.7/5.4</td><td>#4</td><td>3.4/4.5</td><td>#6</td></tr><tr><td>6</td><td>Read→Read→Resp</td><td>4.4/4.5</td><td>#6</td><td>5.5/4.7</td><td>#5</td></tr><tr><td>7</td><td>Read→Resp→Resp</td><td>3.7/2.6</td><td>#11</td><td>5.7/3.5</td><td>#7</td></tr><tr><td>8</td><td>Write→Resp→Resp</td><td>6.0/3.3</td><td>#7</td><td>0.7/2.8</td><td>#11</td></tr><tr><td>9</td><td>Read→Write→Read</td><td>2.4/4.0</td><td>#8</td><td>1.1/3.0</td><td>#9</td></tr><tr><td>10</td><td>Read→Write→Write</td><td>2.7/3.8</td><td>#9</td><td>1.1/2.7</td><td>#12</td></tr><tr><td>11</td><td>Read→Resp→Read</td><td>2.1/3.2</td><td>#12</td><td>3.6/2.4</td><td>#10</td></tr><tr><td>12</td><td>Resp→Read→Write</td><td>4.8/2.3</td><td>#10</td><td>1.1/1.9</td><td>#13</td></tr><tr><td>13</td><td>Read→Read→Transfer</td><td>0.9/1.4</td><td>#18</td><td>12.6/1.7</td><td>#8</td></tr><tr><td>14</td><td>Resp→Read→Resp</td><td>2.4/2.1</td><td>#13</td><td>2.2/1.5</td><td>#15</td></tr><tr><td>15</td><td>Write→Read→Write</td><td>1.2/2.0</td><td>#14</td><td>0.8/1.3</td><td>#18</td></tr><tr><td>16</td><td>Write→Write→Write</td><td>0.7/1.8</td><td>#16</td><td>0.3/1.4</td><td>#19</td></tr><tr><td>17</td><td>Write→Write→Resp</td><td>1.6/1.5</td><td>#15</td><td>0.5/1.2</td><td>#21</td></tr><tr><td>18</td><td>Resp→Resp→Read</td><td>1.2/1.5</td><td>#17</td><td>2.8/0.8</td><td>#20</td></tr><tr><td>19</td><td>Read→Resp→Write</td><td>0.8/1.0</td><td>#20</td><td>0.5/1.7</td><td>#16</td></tr><tr><td>20</td><td>Read→Resp→Transfer</td><td>0.3/0.5</td><td>#26</td><td>5.9/0.7</td><td>#17</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Action chains dominated by information acquisition exhibit limited discriminative power (e.g., IDs 1 and 4), whereas those dominated by information output and action switching demonstrate significant discriminative power (e.g., IDs 2 and 10).

Classifying action chains by function: chains with IDs 1 and 4 (where the last two steps consist of consecutive Read actions, or where the chain as a whole is dominated by Read actions) represent behavior patterns dominated by information acquisition. These two types of chains account for a high proportion in both tasks, but their discriminative performance regarding correct versus incorrect is not particularly outstanding. By contrast, action chains dominated by information output and action switching, specifically IDs $2 , 6 ,$ 10 and 17 in the Retail task, and IDs 2, 5, 10, 12, 19 and 20 in the Airline task that demonstrates significantly discriminative performance in both tasks.

![](images/e2720c27aa071f328c1cc983eff95515f6475a6beb89e146f7ca8c37573e5b8c.jpg)  
(a) Retail domain.

![](images/33942c8f7b3d13d8a45a07bff545aef541911bde9baf397348558cd8156f4a47.jpg)  
(b) Airline domain.  
Figure 5: Trigger-policy comparison on Qwen3-14B across the Retail and Airline domains.

Of all information-output chains, Respond→Respond→ Respond (ID 2) shows both the highest error concentration and the most pronounced discriminative structure in geometric terms.

In the geometric space shown in Fig.4a, ID 2 (Respond→Respond→Respond) consistently corresponds to the most significant segmentation location. To further characterize its internal dynamics, we visualized the turn-wise trajectory of this chain in the two-dimensional PCA space corresponding to the hidden states (Fig.4c), and performed a numerical balancing of correct and incorrect samples. The results show that the scatter plot of correct episodes gradually shifts from the Write/Transfer region to the edge of the Respond cluster (bottom right corner) as the turns progress. In contrast, the scatter of incorrect episode remains in the bottom-right region of the Respond cluster throughout, showing no significant spatial expansion. This difference suggests that, in correct episodes, the latent state representation corresponding to the Respond action continuously expands outwards, whereas incorrect episodes mean remains comparatively localized in this PCA projection.

## 6 Geometry-Triggered Adaptive Reasoning

Sections 4 and 5 establish that hidden-state geometry separates correct from incorrect episodes prior to termination, with the effect most pronounced in chains dominated by information output. We now examine whether these signals can act as online control signals rather than post-hoc diagnostics, using the trajectory geometry to determine when the underlying LLM should engage its thinking mode.

We formalize every intervention as a binary decision $\delta _ { t } \in$ {0, 1} issued at each turn $t ,$ determining whether the agent enters thinking mode and generates a long-CoT trace before acting. Methods differ only in what information produces $\delta _ { t } \colon$

• Fixed Policies set $\delta _ { t }$ to a constant (Never-thinking, Always-thinking), and Warm-up, which thinking only on the first two turns.

• Action-conditioned Policies condition on a predicted next action $\hat { a } _ { t }$ obtained from a learned classifier (F1: Read 0.826, Respond 0.711, Write 0.700, Transfer 0.531).

• Geometry-conditioned Policies threshold the trajectory signals directly, firing when $\kappa _ { t } < \tau _ { \kappa } \mathrm { o r } \beta _ { t } > \tau _ { \beta }$

• Learned policy trains a classifier on the concatenation of the recent three actions one-hots with hidden-state geometries $\left( \kappa _ { t } , \beta _ { t } \right)$ , fitted on logged episodes to predict turns at which failure is imminent.

Because $\kappa _ { t }$ is a three-point statistic, geometry-conditioned triggers are undefined before turn 2. To keep the comparison controlled, every method except Never-thinking adopts this same Warm-up. Each configuration is run over 5 trials per task on the τ-Bench Airline and Retail domains with Qwen3- 14B and Qwen3-32B.

Deliberation gains depend on timing. As shown in Fig.5, Always-thinking outperforms Never-thinking consistently, but the variation in gains does not support a simple “more thinking is better" interpretation. On Retail with Qwen3- 14B, Always-thinking increases reward from 0.286 to 0.307, while Warm-up, which thinks on just the first two turns, reaches 0.361 at a lower token cost. On Qwen3-32B, Warmup achieves 0.405 compared with 0.384 for Always-thinking, while using 12.2k fewer tokens. However, Warm-up does not perform consistently across settings. On Airline with Qwen3- 32B, it reaches only 0.280 against Always-thinking's 0.364. These results indicate that task performance depends on the timing of deliberation, rather than its total amount alone.

Table 3: Trigger policies on τ-Bench. The upper block reports task reward, the lower block mean token cost with the number of max-turn terminations in parentheses. Bold marks the best value in a row, underline the runner-up.
<table><tr><td rowspan="2">Domain</td><td rowspan="2">Model</td><td colspan="3">Fixed policies</td><td colspan="2">Geometry-conditioned (training-free)</td><td>Learned</td></tr><tr><td>Never-think</td><td>Warm-up</td><td>Always-think</td><td>Variance slope  $\beta _ { t } > \tau _ { \beta }$ </td><td>Temporal curvature  $\kappa _ { t } < \tau _ { \kappa }$ </td><td>Learned-HST</td></tr><tr><td colspan="8">Average task reward</td></tr><tr><td>Retail</td><td>Qwen3-14B</td><td>0.286</td><td>0.361</td><td>0.307</td><td>0.393</td><td> $\underline { { 0 . 3 9 7 } } ( \tau _ { \kappa } = - 0 . 2 0 )$ </td><td>0.398</td></tr><tr><td>Airline</td><td>Qwen3-14B</td><td>0.196</td><td>0.348</td><td>0.356</td><td>0.392</td><td>0.384 (−0.15)</td><td>0.420</td></tr><tr><td>Retail</td><td>Qwen3-32B</td><td>0.347</td><td>0.405</td><td>0.384</td><td>0.412</td><td>0.442 (−0.15)</td><td>0.405</td></tr><tr><td>Airline</td><td>Qwen3-32B</td><td>0.136</td><td>0.280</td><td>0.364</td><td>0.328</td><td>0.352 (−0.10)</td><td>0.364</td></tr><tr><td colspan="8">Mean token cost (thousands), with max-turn terminations</td></tr><tr><td>Retail</td><td>Qwen3-14B</td><td>112.6 (75)</td><td>111.1 (69)</td><td>113.1 (100)</td><td>104.3 (55)</td><td>101.0 (34)</td><td>104.4 (51)</td></tr><tr><td>Airline</td><td>Qwen3-14B</td><td>82.5 (28)</td><td>87.1 (27)</td><td>83.5 (26)</td><td>83.3 (19)</td><td>78.9 (16)</td><td>85.0 (28)</td></tr><tr><td>Retail</td><td>Qwen3-32B</td><td>125.7 (117)</td><td>107.1 (79)</td><td>119.3 (108)</td><td>105.1 (70)</td><td>104.6 (68)</td><td>108.5 (78)</td></tr><tr><td>Airline</td><td>Qwen3-32B</td><td>98.2 (33)</td><td>92.6 (35)</td><td>100.6 (33)</td><td>86.3 (25)</td><td>83.0 (25)</td><td>87.5 (29)</td></tr></table>

The effectiveness of action-conditioned triggers varies across tasks. Triggering on the appropriate action recovers much of the performance gain, but the most effective action does not transfer across domains. On Retail with Qwen3-14B, Read (0.393) and Write (0.381) perform well, while Respond collapses to 0.290. On Airline with the same model, the ordering inverts, with Respond (0.372) best and Read (0.352) and Write (0.340) both below Always-thinking. Therefore, the effectiveness of these triggers depends on the task-specific distribution of errors, which needs to be identified in advance.

Geometric thresholds are unimodal, transferable, and require no additional training. Sweeping $\tau _ { \kappa }$ is unimodal with an optimum locatable by a one-dimensional scan: on Retail with Qwen3-14B, thresholds of —0.15, -0.20, -0.25 give 0.388, 0.397, 0.363. That optimum shifts predictably with the correct—incorrect κ gaps of Section 4 (Retail 14B +0.030, Airline 14B +0.049, Retail 32B +0.035, Airline 32B +0.088), so it can be guided by the diagnostic statistics rather than searched per deployment. Crucially, the threshold matches or beats the trained classifier: $\kappa < - 0 . 2 0$ ties Learned-HST on Retail 14B (0.397 vs 0.398) at lower token cost (101.0k vs 104.4k), and $\kappa < - 0 . 1 5$ reaches 0.442 on Retail 32B, above the Learned-HST configuration there (0.405) and 27% above Never-thinking. A trained trigger is fitted to one model's failure distribution, which moves with scale. Temporal curvature is distribution-free and adapts by recalibrating a single scalar.

Geometry-conditioned triggers combine token efficiency with competitive reward. Since token cost is influenced by both deliberation and failure length, firing at the wrong turns spends more while earning less. Across all four domainmodel pairings, the geometry-conditioned triggers instead occupy the favorable corner of this trade-off, attaining the highest task reward among the lowest token costs, with maxturn terminations (default 50 turns) falling below every fixed policy (e.g., from 75 to 34 on Retail with Qwen3-14B under $\kappa < - 0 . 2 0 )$ . Averaging over the four settings, the geometryconditioned trigger raises task reward from 0.241 under Never-thinking to 0.396, a gain from 24.1% to 39.6% in task success, while lowering mean token cost from 104.8k to 93.0k, an 11.2% reduction.

Applicability & Future Direction. We highlight two directions for future research. First, trajectory geometry provides a new lens on multi-turn adversarial attacks, which is an active area in which jailbreaks are spread across turns to divert an agent away from its objective (Russinovich, Salem and Eldan 2025). Our signals could potentially detect such manipulation from the inside before it surfaces in the outputs. Second, as hidden states are increasingly used to detect and correct representation drift (Dongre et al. 2026), characterizing trajectory geometry contributes an interpretable, training-free account of when drift begins, which is a step toward continually learning agentic systems that adapt reliably over long horizons.

## 7 Conclusion

In this work, we formulated multi-turn reasoning as an observable hidden-state trajectory of the underlying LLM and characterized it with two complementary geometric signals, temporal curvature κ and variance slope β. Across four tasks and three underlying LLMs, these signals separate correct from incorrect episodes before termination, with failure trajectories exhibiting directional reversals and premature convergence in their hidden states. Decomposing episodes into three-action chains further showed that this separability is action-dependent, concentrated in information-output chains. Finally, we showed that trajectory geometry can act as an online control signal, triggering adaptive reasoning at critical turns, raises task success on τ-Bench from 24.1% to 39.6% while reducing token cost by 11.2%.

## References

Azaria, A.; and Mitchell, T. 2023. The internal state of an LLM knows when it's lying. In Findings of the Association for Computational Linguistics: EMNLP 2023, 967–976.

Barres, V.; Dong, H.; Ray, S.; Si, X.; and Narasimhan, K. 2025. τ2-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.

Chen, C.; Liu, K.; Chen, Z.; Gu, Y.; Wu, Y.; Tao, M.; Fu, Z.; and Ye, J. 2024a. INSIDE: LLMs’ internal states retain the power of hallucination detection. arXiv preprint arXiv:2402.03744.

Chen, X.; Xu, J.; Liang, T.; He, Z.; Pang, J.; Yu, D.; Song, L.; Liu, Q.; Zhou, M.; Zhang, Z.; et al. 2024b. Do not think that much for 2+ 3=? on the overthinking of o1-like llms. arXiv preprint arXiv:2412.21187.

Chuang, Y.-S.; Qiu, L.; Hsieh, C.-Y.; Krishna, R.; Kim, Y.; and Glass, J. 2024. Lookback lens: Detecting and mitigating contextual hallucinations in large language models using only attention maps. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 1419– 1436.

Cuadron, A.; Li, D.; Ma, W.; Wang, X.; Wang, Y.; Zhuang, S.; Liu, S.; Schroeder, L. G.; Xia, T.; Mao, H.; et al. 2025. The danger of overthinking: Examining the reasoning-action dilemma in agentic tasks. arXiv preprint arXiv:2502.08235.

Dongre, V.; Hsieh, J.; Lai, V. D.; Yoon, S.; Bui, T.; and Hakkani-Tür, D. 2026. When Attention Closes: How LLMs Lose the Thread in Multi-Turn Interaction. arXiv preprint arXiv:2605.12922.

Duan, H.; Yang, Y.; and Tam, K. Y. 2024. Do llms know about hallucination? an empirical investigation of llm's hidden states. arXiv preprint arXiv:2402.09733.

Fang, G.; Ma, X.; and Wang, X. 2026. Thinkless: Llm learns when to think. Advances in neural information processing systems, 38: 151268–151295.

Farquhar, S.; Kossen, J.; Kuhn, L.; and Gal, Y. 2024. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017): 625–630.

Grattafiori, A.; et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Kadavath, S.; Conerly, T.; Askell, A.; Henighan, T.; Drain, D.; Perez, E.; Schiefer, N.; Hatfield-Dodds, Z.; DasSarma, N.; Tran-Johnson, E.; et al. 2022. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221.

Kossen, J.; Han, J.; Razzak, M.; Schut, L.; Malik, S.; and Gal, Y. 2024. Semantic entropy probes: Robust and cheap hallucination detection in llms. arXiv preprint arXiv:2406.15927.

Laban, P.; Hayashi, H.; Zhou, Y.; and Neville, J. 2025. Llms get lost in multi-turn conversation. arXiv preprint arXiv:2505.06120.

Li, K.; Patel, O.; Viégas, F.; Pfister, H.; and Wattenberg, M. 2023. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36: 41451–41530.

Liu, N. F.; Lin, K.; Hewitt, J.; Paranjape, A.; Bevilacqua, M.; Petroni, F.; and Liang, P. 2024. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12: 157–173.

Lou, C.; Sun, Z.; Liang, X.; Qu, M.; Shen, W.; Wang, W.; Li, Y.; Yang, Q.; and Wu, S. 2025. Adacot: Pareto-optimal adaptive chain-of-thought triggering via reinforcement learning. arXiv preprint arXiv:2505.11896.

Marks, S.; and Tegmark, M. 2023. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. arXiv preprint arXiv:2310.06824.

Mei, L.; Yao, J.; Ge, Y.; Wang, Y.; Bi, B.; Cai, Y.; Liu, J.; Li, M.; Li, Z.-Z.; Zhang, D.; et al. 2025. A survey of context engineering for large language models. arXiv preprint arXiv:2507.13334.

Orgad, H.; Toker, M.; Gekhman, Z.; Reichart, R.; Szpektor, I.; Kotek, H.; and Belinkov, Y. 2025. Llms know more than they show: On the intrinsic representation of llm hallucinations. In International Conference on Learning Representations, volume 2025, 66880–66913.

Packer, C.; Fang, V.; Patil, S.; Lin, K.; Wooders, S.; and Gonzalez, J. 2023. MemGPT: towards LLMs as operating systems.

Qwen Team. 2025. Qwen3 Technical Report. arXiv:2505.09388.

Russinovich, M.; Salem, A.; and Eldan, R. 2025. Great, now write an article about that: the crescendo multi-turn LLM jailbreak attack. In Proceedings of the 34th USENIX Conference on Security Symposium, 2421–2440.

Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems, 36: 8634–8652.

Wang, Y.; Zhang, P.; Yang, B.; Wong, D.; and Wang, R. 2025. Latent space chain-of-embedding enables output-free 1lm self-evaluation. In International Conference on Learning Representations, volume 2025, 70938–70970.

Yao, S.; Shinn, N.; Razavi, P.; and Narasimhan, K. 2024. τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.

Zhang, J.; Lin, N.; Hou, L.; Feng, L.; and Li, J. 2025a. Adaptthink: Reasoning models can learn when to think. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 3716–3730.

Zhang, Z.; Hu, X.; Zhang, H.; Zhang, J.; and Wan, X. 2025b. ICR probe: Tracking hidden state dynamics for reliable hallucination detection in LLMs. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 17986–18002.

Zhao, W.; Sui, X.; Guo, J.; Hu, Y.; Deng, Y.; Zhao, Y.; Zhi, X.; Huang, Y.; He, H.; Che, W.; et al. 2026. Trade-offs in large reasoning models: An empirical analysis of deliberative and adaptive reasoning over foundational capabilities. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 34976–34984.