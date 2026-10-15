# Model What Matters: Decision-Guided Selective World Modeling for Language Agents

Maolin Ran<sup>1</sup>, Minda Chen<sup>2</sup>, Di Huang<sup>2</sup>, Wei Xiong<sup>2</sup>, Fei Huang<sup>2,∗</sup>, Weiwen Liu<sup>1,∗</sup> <sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>LongShine AI Research {maolinr03, wwliu}@sjtu.edu.cn, huangfei@longshine.com

## Abstract

Reinforcement learning provides a direct mech anism for optimizing task objectives in large language model-driven agents, but task rewards are typically sparse trajectory-level signals that only evaluate task success, without directly su pervising action-conditioned environment transitions, leaving the model with limited explicit understanding of action consequences and environment dynamics. Recent work introduces auxiliary supervision through policy and world modeling co-training, yet typically treats complete environmental observations uni formly as world modeling targets, overlooking the varying decision relevance of observation content. We propose DGS-WM (Decision-Guided Selective World Modeling), a decisionguided policy-world modeling co-training approach for language agents. DGS-WM employs a Decision-Guided Observation Selection component that filters observation units with decision value as world modeling targets based on their support for subsequent policy responses, and a Confidence-Adaptive Weight ing component that adaptively adjusts supervision strength according to the model’s prediction confidence on selected targets. Together, these components form a decision-aware auxiliary world modeling loss, jointly optimized with the RL loss on the same policy model. Across multiple benchmarks covering singlehop/multi-hop question answering and interactive web tasks, DGS-WM achieves consis tent improvements under different model scales and RL algorithms, demonstrating more stable and effective performance compared to full observation world modeling methods.

## 1 Introduction

Large language model (LLM)-driven agents can complete tasks such as retrieval, question answering, and web operations through multi-turn interactions with environments. Reinforcement learning (RL) provides a direct training mechanism for optimizing task objectives for such agents, enabling models to progressively improve action selection and task execution capabilities based on environmental observations and task rewards. In recent years, PPO, GRPO, and related methods have been widely applied to LLM reasoning optimization and agent training, achieving significant progress (Schulman et al., 2017; Ouyang et al., 2022; DeepSeek-AI, 2025).

Multi-turn tasks such as retrieval, question answering, and web operations typically exhibit longhorizon dependencies: current actions change environment states, and state changes determine subsequent available environmental observations and feasible actions (Feng et al., 2025; Jin et al., 2025). To make effective decisions continuously in such tasks, policies need to form usable representations of action consequences and environment dynamics. RL primarily optimizes policy behavior through task rewards; trajectory-level or outcome-level rewards typically only evaluate task success and do not directly supervise action-conditioned environment transitions (Jin et al., 2025; Feng et al., 2025). Therefore, models can learn which behaviors are more likely to obtain high rewards, but may not explicitly learn what environmental observations actions will produce and how these environmental observations affect subsequent interactions; the action-environmental observation relationships already recorded in rollouts are also not fully converted into independent training signals.

World modeling methods provide explicit environment dynamics supervision for policies by learning subsequent states or textualized environmental observations conditioned on actions, thereby supplementing training signals that rely solely on task rewards. Recent research further combines this learning objective with on-policy RL: ECHO optimizes the policy objective while using environmental observations produced by actions in rollouts for auxiliary prediction; PaW further filters transitions to be included in world modeling training on this basis. For transitions included in joint training, both typically use their complete next observation as the world modeling target (Figure 1 left), enabling the policy model to learn environment dynamics during interactions beyond task rewards (Shrivastava et al., 2026; Lu et al., 2026). This raises the question we focus on in this paper: within a single environmental observation, should the complete observation be treated by default as a unified world modeling target?

Our core insight is that environmental observations do not inherently possess equal supervision value. An environmental observation may simultaneously contain key information that can influence future decisions or content unrelated to the current task; if the complete environmental observation is uniformly used as a supervision target, the model must bear the same auxiliary prediction pressure for this content. Therefore, we argue that only observation content valuable for future decisions should be retained as world modeling supervision targets.

Based on this insight, we propose DGS-WM (Decision-Guided Selective World Modeling), a decision-guided policy-world modeling co-training method for text agents (Figure 1 right). DGS-WM first filters content with decision value from environmental observations through Decision-Guided Observation Selection and constructs them as world modeling targets; subsequently through Confidence-Adaptive Weighting, it adjusts auxiliary supervision strength based on the model’s current prediction confidence for retained targets. Based on selected prediction targets, DGS-WM constructs world modeling objectives to jointly train the policy model together with RL objectives.

DGS-WM first filters content with decision value from environmental observations through Decision-Guided Observation Selection and constructs them as world modeling targets; subsequently through Confidence-Adaptive Weighting, it adjusts auxiliary supervision strength based on the model’s current prediction confidence for retained targets. Based on selected prediction targets, DGS-WM constructs world modeling objectives to jointly train the policy model together with RL objectives.

The main contributions of this paper are as follows:

1. We propose DGS-WM, an RL and world modeling co-training algorithm for text agents that implements decision-oriented observation selection and adaptive supervision strength control through Decision-Guided Observation Selection and Confidence-Adaptive Weighting.

![](images/36d1972d9737d44d74b23f2c687e92666aac3b3eebbd1bef9443b23bb9625236.jpg)  
Figure 1: Comparison of typical policy-world modeling co-training versus DGS-WM on a two-hop question answering task. Given the same environment observations, typical co-training methods supervise all observation content equally as WM targets (left), while DGS-WM selectively supervises only decision-relevant content based on their support for future policy responses (right). Green highlights indicate selected content; gray text indicates filtered noise.

2. We propose a core insight: the world modeling value of environmental observations is not uniform, and complete observations should not be treated by default as unified supervision targets. Based on this, we refine supervision granularity to within observations, retaining only content valuable for future decisions as world modeling targets, and adjusting auxiliary supervision strength through confidenceadaptive weighting.

3. We conduct systematic evaluation on multiple public benchmarks covering single-hop/multihop question answering and web interaction. Results show that DGS-WM achieves consistent improvements across different tasks, model scales, and RL algorithms; in contrast, full observation world modeling methods (ECHO) show unstable performance in some settings, sometimes underperforming pure RL baselines.

## 2 Related Work

## 2.1 Reinforcement Learning for LLM Agents

Reinforcement learning has become an important paradigm for optimizing LLM policies. Existing methods first revolve around policy updates and advantage estimation: PPO (Schulman et al., 2017) constrains on-policy updates with clipped surrogate objectives, REINFORCE-style methods (Williams, 1992) and RLOO (Ahmadian et al., 2024) reduce reliance on explicit value models through sequencelevel returns and leave-one-out baselines, while GRPO (Shao et al., 2024) performs group-based optimization using relative returns from multiple samples under the same input. Subsequently, extensive work has improved the training efficiency and stability of LLM RL from aspects such as clipped ob jectives, sampling strategies, loss aggregation, and importance weights, such as DAPO (Meng et al., 2024) and GMPO (Zhao et al., 2025). For longsequence and multi-turn agents, research further focuses on how to perform finer-grained credit assignment using sparse outcome rewards, including GiGPO (Feng et al., 2025) based on episode/state grouping, SPO (Li et al., 2024) based on segmentlevel advantage, and Tree-GRPO (Li et al., 2025) that constructs intermediate process signals using structured rollouts. Compared to outcome-level rewards, process supervision provides finer-grained training signals for intermediate steps, but typically involves additional annotation, estimation, or data construction costs (Lightman et al., 2023). The method proposed in this paper provides auxiliary supervision for RL training through world model ing, requiring no additional annotations and can be combined with different on-policy RL algorithms (such as GRPO).

## 2.2 World Modeling for LLM Agents

World modeling methods for language agents mainly revolve around learning action-conditioned environment transitions, i.e., predicting subsequent states or textualized environmental observations based on interaction history and actions. Early work typically places the learned world model outside the policy optimization process, serving as an environment simulator or planning module for generating imagined rollouts, performing search, or conducting lookahead planning (Ha and Schmidhuber, 2018; Hao et al., 2023; Liu et al., 2026). Building on this, the WM-to-RL staged paradigm first uses interaction trajectories to learn environment dynamics, future state prediction, or other environment knowledge, then uses the obtained world model checkpoint or representations for subsequent policy optimization; Early Experience (Zhang et al., 2025) and RWML (Yu et al., 2026) represent the reward-free world model warm-up and the approach of first performing world model learning then task-success RL, respectively. More recent work places world model learning and onpolicy policy optimization in the same training process, forming a policy-world modeling co-training paradigm (Shrivastava et al., 2026; Lu et al., 2026). Under this paradigm, environmental feedback included in WM training typically directly uses the complete next observation as the world modeling target, without further distinguishing content within the feedback, making all content jointly bear auxiliary supervision. DGS-WM continues this cotraining paradigm but consciously selects world modeling targets, retaining only feedback content valuable for future decisions for world modeling supervision.

## 3 Methodology

We propose DGS-WM (Decision-Guided Selective World Modeling), a policy-world modeling cotraining method for language agents that introduces selective environmental observation prediction supervision during RL training. As illustrated in Figure 2, Decision-Guided Observation Selection uses subsequent policy responses to select world modeling targets from observations, and Confidence-Adaptive Weighting adjusts supervision strength based on the model’s prediction confidence for selected targets. The resulting world modeling loss is jointly optimized with the RL loss to update the same policy model.

## 3.1 Problem Formulation and Auxiliary World Modeling

In text interaction tasks, a language model policy $\pi _ { \theta }$ generates responses and executes actions $a _ { t }$ based on interaction history $h _ { t }$ (containing task instructions and prior interactions). After receiving environmental observation $o _ { t + 1 }$ , the policy generates the next complete response:

$$
u _ { t + 1 } \sim \pi _ { \bar { \theta } } ( \cdot \mid h _ { t } , a _ { t } , o _ { t + 1 } ) ,\tag{1}
$$

where $\bar { \theta }$ denotes the policy parameters at trajectory sampling time. RL optimizes the policy based on trajectory rewards, with loss denoted ${ \mathcal { L } } _ { \mathrm { R I } }$

![](images/eca43ceea4b3fcebd265f49b28f0f2ded082da6b5c826afcb782c44b1f3cf541.jpg)  
Figure 2: Overview of DGS-WM. Complete trajectories provide RL signal; observations and subsequent responses are used for target selection; history, action, and selected targets are used for world modeling prediction, producing target-level confidence and weights; weighted world modeling loss jointly updates policy with RL loss.

The same trajectories also provide supervision for auxiliary world modeling (WM): the model predicts subsequent environmental observations given history and actions, with observation prediction distribution denoted $p _ { \theta }$ . Over the batch observation sample index set $B _ { \mathrm { o b s } }$ , the full-observation world modeling loss is:

$$
\mathcal { L } _ { \mathrm { W M } } ^ { \mathrm { f u l l } } = - \frac { \sum _ { t \in \mathcal { B } _ { \mathrm { o b s } } } \log p _ { \theta } ( o _ { t + 1 } \mid h _ { t } , a _ { t } ) } { \sum _ { t \in \mathcal { B } _ { \mathrm { o b s } } } \left| o _ { t + 1 } \right| } ,\tag{2}
$$

where $\left| o _ { t + 1 } \right|$ denotes the token count of observation $o _ { t + 1 }$ , and the denominator normalizes by the total token count across all observations.

With λ<sub>WM</sub> controlling auxiliary supervision strength, the joint training objective is:

$$
\mathcal { L } _ { \mathrm { b a s e } } = \mathcal { L } _ { \mathrm { R L } } + \lambda _ { \mathrm { W M } } \mathcal { L } _ { \mathrm { W M } } ^ { \mathrm { f u l l } } .\tag{3}
$$

## 3.2 Decision-Guided Observation Selection

Complete environmental observations typically mix decision-relevant information with taskirrelevant content. Full-observation world modeling (Eq. 2) treats all content uniformly as world modeling targets, forcing the model to exert equal prediction effort on both noise and useful information. This not only wastes model capacity but may also interfere with the policy’s learning of critical information. Our core thesis is: only environmental observations that hold value for future decisions merit serving as world modeling targets. To this end, Decision-Guided Observation Selection uses subsequent policy responses to select supervisory content from observations.

We first partition environmental observation $o _ { t + 1 }$ into a set of candidate units. The partitioning follows a “minimal semantically complete unit” principle: for retrieval document feedback, we split at sentence boundaries to obtain single-sentence candidates; for structured webpage observations, we identify semantically complete UI blocks (e.g., product cards, core purchase information, interaction feedback) as candidate units. Implementation details are provided in Appendix B. Formally, we split the observation into $m _ { t }$ candidate observation units arranged in original order:

$$
\mathcal { C } _ { t } = \mathrm { S p l i t } ( o _ { t + 1 } ) = ( c _ { t , 1 } , \ldots , c _ { t , m _ { t } } ) .\tag{4}
$$

For each candidate unit $c _ { t , j }$ , we retain $h _ { t }$ and $a _ { t }$ replace the current observation with that unit, and use the sampling policy $\pi _ { \bar { \theta } }$ to compute the average token log-likelihood of the true next response under teacher forcing:

$$
s _ { t , j } = \frac { 1 } { \vert u _ { t + 1 } \vert } \log \pi _ { \bar { \theta } } \left( u _ { t + 1 } \mid h _ { t } , a _ { t } , c _ { t , j } \right) .\tag{5}
$$

We define $s _ { t , j }$ as the response support score. Intuitively, $s _ { t , j }$ measures the policy’s average token log-likelihood for the true subsequent response $u _ { t + 1 }$ given candidate unit $c _ { t , j } .$ A higher score indicates lower policy decision uncertainty conditioned on that observation unit, meaning the unit contains more information supporting subsequent decisionmaking. All candidate units are scored against the same true subsequent response.

We perform relative filtering within each observation rather than setting a global absolute threshold. This is because score scales may vary significantly across interactions: some environmental observations have high overall information content, yielding high scores for all candidate units; others contain more noise, resulting in universally lower scores. Relative filtering adapts to the scoring scale of different interactions. Let $\mu _ { t }$ and $\sigma _ { t }$ denote the mean and standard deviation of response support scores across all candidate units in the current observation. The retained set is defined as:

$$
\mathcal { C } _ { t } ^ { + } = \left\{ c _ { t , j } : s _ { t , j } > \mu _ { t } + \kappa \sigma _ { t } \right\} .\tag{6}
$$

Parameter $\kappa$ controls filtering strictness. The threshold adjusts with the score distribution of the current observation, making selection criteria adaptive to scoring scales across different interactions. If no candidate unit exceeds the threshold, we skip world modeling for that interaction. Otherwise, we concatenate retained units in original order to form world modeling target $z _ { t + 1 } .$

$$
z _ { t + 1 } = \mathrm { C o n c a t } \left( \mathcal { C } _ { t } ^ { + } \right) .\tag{7}
$$

For the resulting target, the model predicts its content given interaction history and current action, with average negative log-likelihood:

$$
\ell _ { t } = - \frac { \log p _ { \theta } ( z _ { t + 1 } \mid h _ { t } , a _ { t } ) } { | z _ { t + 1 } | } .\tag{8}
$$

Complete observations remain in the original RL interaction context; filtering only determines auxiliary world modeling target content. However, selecting target content alone is insufficient: for targets the model has already mastered, continued supervision at equal strength wastes training resources; for targets not yet mastered, sufficient learning signal must be maintained. Therefore, we further introduce the Confidence-Adaptive Weighting mechanism.

## 3.3 Confidence-Adaptive Weighting

Target selection determines what feedback content to learn. To avoid continued equal-strength auxiliary supervision on already-mastered targets, Confidence-Adaptive Weighting adjusts supervision strength for each target based on the model’s current prediction confidence.

Using the target average negative log-likelihood $\ell _ { t }$ defined in the previous section, we compute prediction confidence:

$$
q _ { t } = \exp \left( - \ell _ { t } \right) .\tag{9}
$$

q<sub>t</sub> equals the geometric mean of target content token conditional probabilities, reflecting the model’s mastery of predicting the entire target. We accordingly compute the adaptive weight:

$$
w _ { t } = \mathrm { s t o p g r a d } \left[ ( 1 - q _ { t } ) ^ { \gamma } \right] .\tag{10}
$$

where $\gamma$ controls confidence weighting decay strength. stopgrad indicates the weight does not participate in gradient backpropagation; gradients flow through the target prediction loss to update the model.

Each concatenated world modeling target corresponds to one $q _ { t }$ and one $w _ { t } ;$ all content tokens of that target share this weight. High-confidence targets receive smaller supervision weights, while lower-confidence targets retain stronger learning signal, thereby adaptively adjusting auxiliary supervision pressure according to the model’s mastery of the target.

## 3.4 Joint Optimization

Decision-Guided Observation Selection (Section 3.2) determines “what to predict,” and Confidence-Adaptive Weighting (Section 3.3) decides “at what strength to learn.” The two components jointly constitute our Decision-Aware World Modeling Loss. Let $B _ { \mathrm { W M } }$ denote the index set of interactions in the update minibatch that have valid world modeling targets. The loss is:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { W M } } ^ { \mathrm { D G S } } = \frac { \sum _ { t \in \mathcal { B } _ { \mathrm { W M } } } w _ { t } \left| \boldsymbol { z } _ { t + 1 } \right| \ell _ { t } } { \sum _ { t \in \mathcal { B } _ { \mathrm { W M } } } \left| \boldsymbol { z } _ { t + 1 } \right| } } \\ & { \qquad = - \frac { \sum _ { t \in \mathcal { B } _ { \mathrm { W M } } } w _ { t } \log p _ { \theta } \left( \boldsymbol { z } _ { t + 1 } \mid \boldsymbol { h } _ { t } , \boldsymbol { a } _ { t } \right) } { \sum _ { t \in \mathcal { B } _ { \mathrm { W M } } } \left| \boldsymbol { z } _ { t + 1 } \right| } . } \end{array}\tag{11}
$$

Compared to $\mathcal { L } _ { \mathrm { W M } } ^ { \mathrm { f u l l } }$ , this objective replaces complete observations $o _ { t + 1 }$ with selected $z _ { t + 1 }$ and introduces target-level weights $w _ { t }$ . The loss is normalized by the total content token count of selected targets; when $B _ { \mathrm { W M } }$ is empty, the auxiliary loss is set to zero. Combining with the base RL loss yields the complete training objective:

$$
\mathcal { L } _ { \mathrm { D G S - W M } } = \mathcal { L } _ { \mathrm { R L } } + \lambda _ { \mathrm { W M } } \mathcal { L } _ { \mathrm { W M } } ^ { \mathrm { D G S } } .\tag{12}
$$

Global coefficient $\lambda _ { \mathrm { W M } }$ and target-level weights w<sub>t</sub> respectively control the overall strength of auxiliary supervision and individual target contributions.

Algorithm 1 summarizes one sampling and policy update iteration.

Algorithm 1 DGS-WM Training Iteration   
Require: Policy parameters θ; selection threshold κ; confi   
dence decay γ; auxiliary coefficient $\lambda _ { \mathrm { W M } }$   
Ensure: Updated policy parameters θ   
1: $\bar { \theta }  \theta ;$ sample trajectories $\mathcal { D } \sim \pi _ { \bar { \theta } } ;$ initialize $w  \emptyset$   
2: for each $( \hat { h , } a , o , \bar { u } ) \in \mathcal { D }$ do   
3: ${ \mathcal { C } } \gets \mathrm { S p l i t } ( o )$   
4: if $c \neq { \bar { \varnothing } }$ then   
5: for each $c \in { \mathcal { C } }$ do   
6: $s ( c ) \gets \log \pi _ { \bar { \theta } } ( u \mid h , a , c ) / | u |$   
7: end for   
8: $\mu _ { t } , \sigma _ { t } \gets \mathrm { M e a n S t d } ( \{ s ( c ) : c \in \mathcal { C } \} )$   
9: ${ \mathcal { C } } ^ { + }  \{ c \in { \mathcal { C } } : s ( c ) > \mu _ { t } + \kappa \sigma _ { t } \}$   
10: i $\ R C ^ { + } \neq \emptyset$ then   
11: $\mathcal { W }  \mathcal { W } \cup \{ ( h , a , \mathrm { C o n c a t } ( \mathcal { C } ^ { + } ) ) \}$   
12: end if   
13: end if   
14: end for   
15: for each minibatch $( \mathcal { D } _ { b } , \mathcal { W } _ { b } )$ from $( \mathcal { D } , \mathcal { W } )$ do   
16: if $\mathcal { W } _ { b } = \emptyset$ then $\mathcal { L } _ { \mathrm { W M } }  0$   
17: else   
18: for each $( h , a , z ) \in \mathcal { W } _ { b }$ do   
19: $\ell _ { z } \gets - \log p \dot { \varrho } \big ( z \mid h , a \big ) / | z |$   
20: $q _ { z } \gets \exp ( - \ell _ { z } ) ; w _ { z } \gets \mathrm { \backslash } \mathrm { \backslash } \mathrm { \cup p g r a d } ( ( 1 - q _ { z } ) ^ { \gamma } )$   
21: end for   
22: $\begin{array} { r } { \mathcal { L } _ { \mathrm { W M } }   \frac { \sum _ { ( h , a , z ) \in \mathcal { W } _ { b } } w _ { z } \log p _ { \theta } ( z | h , a ) } { \Gamma } } \end{array}$   
$\overline { { \sum _ { \left( h , a , z \right) \in \mathcal { W } _ { b } } \left| z \right| } }$   
23: end if   
24: $\mathcal { I } \gets \mathcal { L } _ { \mathrm { R L } } ( \mathcal { D } _ { b } ) + \lambda _ { \mathrm { W M } } \mathcal { L } _ { \mathrm { W M } }$   
25: $\theta \gets \mathrm { U p d a t e } ( \theta , \nabla _ { \theta } \mathcal { I } )$   
26: end for

Candidate scoring and target selection use the sampling policy, while confidence and weights are computed by the current model during updates. At inference time, the model directly generates policy responses based on complete interaction history, requiring no observation filtering or confidence computation, maintaining the same interaction mode as standard agents.

## 4 Experiments

## 4.1 Experimental Setup

Datasets. We evaluate on three task categories: (i) Multi-hop QA: HotpotQA (Yang et al., 2018), 2WikiMultiHopQA (Ho et al., 2020), MuSiQue (Trivedi et al., 2022), Bamboogle (Press et al., 2023); (ii) Single-hop QA: Natural Questions (NQ) (Kwiatkowski et al., 2019), TriviaQA (Joshi et al., 2017), PopQA (Mallen et al., 2023); (iii) Interactive Web Shopping: WebShop (Yao et al., 2022).

Baselines. We compare three method categories: (i) Prompting methods: Direct Inference and Re-Act (Yao et al., 2023); (ii) RL-only methods: Primary experiments use GRPO (Shao et al., 2024); cross-algorithm experiments additionally use PPO (Schulman et al., 2017), RLOO (Ahmadian et al., 2024), and GMPO (Zhao et al., 2025); (iii) RLaugmented world modeling methods: ECHO (Shrivastava et al., 2026) and our proposed DGS-WM, which augment the RL algorithms above with world modeling. ECHO uses complete environmental observations as world modeling targets, serving as a full-observation baseline against DGS-WM’s decision-value-guided observation selection. Cross-model and cross-algorithm results compare each RL algorithm alone with its ECHOand DGS-WM-augmented variants.

Metrics. Single-hop and multi-hop QA use Exact Match (EM, %). Unless otherwise specified, single-hop results are macro-averaged over NQ, TriviaQA, and PopQA; multi-hop results are macro-averaged over HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle. In Table 1, Trivia, 2Wiki, MusiQ, and Bamb abbreviate TriviaQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle, respectively. WebShop reports Success rate (%) and Score (0–100 scale).

Implementation Details. Primary table experiments use Qwen3-1.7B and Qwen3-8B (Team, 2025); component ablation uses Qwen3-1.7B. Policy models are fine-tuned via LoRA (Hu et al., 2022) with rank 32. For QA tasks, we use the Wiki18 corpus with E5-base-v2 (Wang et al., 2022) embeddings and FAISS (Johnson et al., 2017) for retrieval. The selection standard deviation coefficient κ and confidence decay exponent γ are set to 0.5 and 1.0 respectively; following ECHO (Shrivastava et al., 2026), we set the world modeling auxiliary loss coefficient $\lambda _ { \mathrm { W M } } ~ = ~ 0 . 0 2$ . All reported results use greedy decoding for evaluation. Additional details are in Appendix A.

## 4.2 Main Results

Table 1 summarizes results across two model scales on single-hop QA, multi-hop QA, and WebShop. DGS-WM consistently outperforms the ECHO cotraining baseline across all task groups. On Qwen3- 1.7B, DGS-WM achieves 45.77% and 22.06% average EM on single-hop and multi-hop QA respectively, improving over ECHO by 1.45% and 4.35%. On Qwen3-8B, DGS-WM reaches 50.14% and 40.48%, improving over ECHO by 1.39% and 2.33%. On WebShop, DGS-WM achieves 43.20% and 47.20% success rates on the two models, improving over ECHO by 2.80% and 2.20%.

Notably, full-observation world modeling (ECHO) shows unstable performance, sometimes underperforming the pure RL baseline GRPO. For example, on Qwen3-1.7B multi-hop QA, ECHO achieves 17.71% average EM, lower than GRPO’s 18.88% (1.17% lower); single-hop average EM also drops from 44.64% to 44.32%. This suggests that uniformly supervising all environmental observation content as world modeling targets with equal strength may introduce decisionirrelevant noise, interfering with policy optimization. In contrast, DGS-WM consistently improves over GRPO across all settings: on Qwen3-1.7B, single-hop and multi-hop average EM improve by 1.13% and 3.18%; on Qwen3-8B, by 4.21% and 4.69%. This comparison supports our core thesis: through decision-guided observation selection and confidence-adaptive weighting, DGS-WM provides more stable and effective auxiliary world modeling supervision.

Table 1: Main results. QA models train on NQ and HotpotQA; † and ⋆ denote in-domain and out-of-domain QA evaluation. QA reports EM (%); WebShop reports Success (%) and Score. Indented rows augment GRPO.
<table><tr><td></td><td></td><td colspan="4">Single-hop QA</td><td colspan="5">Multi-hop QA</td><td colspan="2">WebShop</td></tr><tr><td>Model</td><td>Method</td><td>NQ†</td><td>Trivia*</td><td>PopQA*</td><td>Avg</td><td>HotpotQA†</td><td>2Wiki*</td><td>MusiQ*</td><td>Bamb*</td><td>Avg</td><td>Succ.</td><td>Score</td></tr><tr><td rowspan="5">Qwen3- 1.7B</td><td>Direct</td><td>16.18</td><td>33.99</td><td>22.12</td><td>24.10</td><td>11.03</td><td>3.75</td><td>2.03</td><td>3.20</td><td>5.00</td><td>16.60</td><td>45.51</td></tr><tr><td>ReAct</td><td>18.20</td><td>36.52</td><td>22.43</td><td>25.72</td><td>12.98</td><td>11.07</td><td>4.01</td><td>20.00</td><td>12.02</td><td>17.20</td><td>54.70</td></tr><tr><td>GRPO</td><td>36.29</td><td>55.41</td><td>42.22</td><td>44.64</td><td>26.46</td><td>24.20</td><td>5.67</td><td>19.20</td><td>18.88</td><td>37.80</td><td>64.21</td></tr><tr><td>w/ECHO</td><td>37.09</td><td>56.71</td><td>39.16</td><td>44.32</td><td>27.08</td><td>21.06</td><td>7.49</td><td>15.20</td><td>17.71</td><td>40.40</td><td>67.93</td></tr><tr><td>w/DGS-WM</td><td>37.53</td><td>57.46</td><td>42.31</td><td>45.77</td><td>30.30</td><td>25.45</td><td>7.70</td><td>24.80</td><td>22.06</td><td>43.20</td><td>70.39</td></tr><tr><td rowspan="5">Qwen3- 8B</td><td>Direct</td><td>35.57</td><td>58.03</td><td>36.74</td><td>43.45</td><td>31.72</td><td>26.66</td><td>12.54</td><td>32.80</td><td>25.93</td><td>20.20</td><td>52.12</td></tr><tr><td>ReAct</td><td>32.74</td><td>59.79</td><td>34.49</td><td>42.34</td><td>35.07</td><td>39.62</td><td>13.12</td><td>39.20</td><td>31.75</td><td>23.60</td><td>59.67</td></tr><tr><td>GRPO</td><td>37.17</td><td>61.23</td><td>39.39</td><td>45.93</td><td>38.87</td><td>42.53</td><td>14.56</td><td>47.20</td><td>35.79</td><td>43.80</td><td>69.87</td></tr><tr><td>w/ECHO</td><td>40.25</td><td>63.47</td><td>42.52</td><td>48.75</td><td>43.63</td><td>42.64</td><td>20.73</td><td>45.60</td><td>38.15</td><td>45.00</td><td>69.32</td></tr><tr><td>w/DGS-WM</td><td>43.35</td><td>64.39</td><td>42.68</td><td>50.14</td><td>44.85</td><td>43.84</td><td>22.01</td><td>51.20</td><td>40.48</td><td>47.20</td><td>71.75</td></tr></table>

## 4.3 Generalization Across Models and Algorithms

Table 2 validates DGS-WM’s generalization advantages over the ECHO-augmented variants. We test three additional models (Qwen2.5-1.5B, Llama-3.2-3B, Qwen2.5-7B) under GRPO, and three additional RL algorithms (PPO, RLOO, GMPO) on Qwen2.5-1.5B. Across all 6 configurations, DGS-WM consistently outperforms both the corresponding RL-only baseline and ECHO-augmented variant. This demonstrates that decision-guided selective world modeling provides stable auxiliary supervision across different model scales and RL algorithms, outperforming full-observation world modeling.

In contrast, the ECHO-augmented variants show inconsistent performance: for GRPO with Llama-3.2-3B or Qwen2.5-7B, they improve single-hop average EM by 1.95% and 0.79% over the corresponding RL-only runs; for GRPO with Qwen2.5-

Table 2: Generalization across models and RL algorithms. EM (%) for single-hop and multi-hop QA averaged over all benchmarks in each category. Each row’s “RL-only” result uses the listed algorithm alone; “w/ ECHO” and “w/ DGS-WM” are its corresponding augmented variants.
<table><tr><td>Alg. Model</td></tr><tr><td>Single-hop QA</td></tr><tr><td>GRPO Qwen2.5-1.5B 45.34 44.64 46.28 GRPO Llama-3.2-3B 47.87 49.82 50.75</td></tr><tr><td>GRPO Qwen2.5-7B 49.76 50.55 51.99 PPO Qwen2.5-1.5B 44.15 43.68 46.06 RLOO Qwen2.5-1.5B 44.15 44.97 45.85</td></tr><tr><td>GMPO Qwen2.5-1.5B 42.37 44.54 45.54</td></tr><tr><td>Multi-hop QA GRPO 17.22 16.70 17.74</td></tr><tr><td>Qwen2.5-1.5B GRPO Llama-3.2-3B 29.42 30.48 33.48</td></tr><tr><td>GRPO Qwen2.5-7B 32.43 34.96 36.54</td></tr><tr><td>PPO Qwen2.5-1.5B 17.21 16.27 17.92</td></tr><tr><td>RLOO Qwen2.5-1.5B 16.47 16.87 17.64</td></tr><tr><td>GMPO Qwen2.5-1.5B 15.90 15.61 17.40</td></tr><tr><td></td></tr></table>

1.5B, the single-hop average EM drops by 0.70%, and multi-hop by 0.52%. With Qwen2.5-1.5B and the three other RL algorithms, the ECHOaugmented variants are also unstable: under PPO, single-hop and multi-hop drop by 0.47% and 0.94%; under GMPO, single-hop improves by 2.17% but multi-hop drops by 0.29%.

## 4.4 Ablation Study

Table 3 validates the necessity of both DGS-WM components. On Qwen3-1.7B, we compare GRPO alone, its ECHO- and DGS-WM-augmented variants, and two DGS-WM ablations. Removing Decision-Guided Observation Selection (DGOS in Table 3) reduces DGS-WM to confidence-adaptive weighting on full environmental observations, essentially an enhanced version of ECHO; removing Confidence-Adaptive Weighting (CAW in Table 3) retains observation selection but fixes all selected target supervision weights to 1 $( \gamma = 0 )$

![](images/a52b32dd6c244d599a1ff065333d19dfff9b410dd978161bed6b4d360925b04a.jpg)

Full DGS-WM achieves 45.77% and 22.06% average EM on single-hop and multi-hop QA, outperforming all variants. Removing DGOS drops single-hop and multi-hop average EM to 44.45% and 20.31%, decreasing by 1.32% and 1.75% relative to the full method; this variant uses full observations with confidence weighting, and its performance still exceeds ECHO’s direct use of full observations, indicating that confidence-adaptive weighting can partially mitigate noise interference from full observations. Removing CAW drops singlehop and multi-hop average EM to 44.82% and 21.66%, decreasing by 0.95% and 0.40%, showing that adaptively adjusting supervision strength on top of decision-guided observation selection further improves auxiliary training effectiveness.

Table 3: Ablation results on Qwen3-1.7B (EM, % by QA category). The first three rows compare GRPO with its ECHO- and DGS-WM-augmented variants; the final two rows ablate the indicated DGS-WM components.
<table><tr><td>Method</td><td>Single-hop QA</td><td>Multi-hop QA</td></tr><tr><td>GRPO</td><td>44.64</td><td>18.88</td></tr><tr><td>GRPO w/ ECHO</td><td>44.32</td><td>17.71</td></tr><tr><td>GRPO w/ DGS-WM</td><td>45.77</td><td>22.06</td></tr><tr><td>w/o DGOS</td><td>44.45</td><td>20.31</td></tr><tr><td>w/o CAW</td><td>44.82</td><td>21.66</td></tr></table>

Figure 3: Parameter sensitivity on Qwen3-1.7B. Combined QA EM (%) is the equally weighted average of single-hop and multi-hop averages. Left: varying κ with $\gamma = 1$ fixed. Right: varying γ with $\kappa = 0 . 5$ fixed. The gray dashed line marks the ECHO baseline (31.02%).

## 4.5 Parameter Sensitivity Analysis

We examine the sensitivity of the selection threshold coefficient κ and the confidence decay exponent γ on Qwen3-1.7B. As shown in Figure 3, the combined QA average first increases and then decreases with $\kappa ,$ peaking at $\kappa = 0 . 5 ;$ confidence weighting achieves the best performance at $\gamma = 1$

## 5 Conclusion

We address the problem of non-uniform supervision value in auxiliary world modeling for language agents. Our core thesis is that not all environmental observation content holds equal value for future decisions—only decision-relevant feedback merits serving as world modeling prediction targets. We propose DGS-WM (Decision-Guided Selective World Modeling), which uses Decision-Guided Observation Selection to select supervisory content from observations based on subsequent policy responses, and Confidence-Adaptive Weighting to adaptively adjust supervision strength according to prediction confidence. Across benchmarks spanning single-hop QA, multi-hop QA, and interactive web shopping, DGS-WM achieves consistent improvements over different RL algorithms and model scales, requiring no additional teacher models or human annotations and introducing no inference overhead. Experimental results show that compared to full-observation world modeling methods that may introduce noise interference, selective world modeling provides more stable and effective auxiliary supervision. These results demonstrate that environmental observations in on-policy RL trajectories, after decision-guided selection, provide a simple and practical source of world modeling supervision for language agent training.

## 6 Limitations

Despite its gains across benchmarks, DGS-WM still has two limitations.

First, observation units depend on environment structure: QA uses retrieval-feedback sentences, while WebShop uses page structure. How to identify meaningful units in complex or unstructured feedback remains open; adaptive segmentation based on semantic boundaries, information entropy, or attention is a promising direction.

Second, the response support score estimates an observation’s decision value from the policy’s average token log-likelihood of the true subsequent response. It assumes that greater consistency with the true response implies greater decision support, but this is correlation, not causation. Counterfactual reasoning or causal interventions could help identify feedback that truly affects decisions.

## References

Arash Ahmadian et al. 2024. Direct language model alignment from online ai feedback. In arXiv preprint.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Lang Feng, Zhenghai Liu, Tingcong Cong, and Bo An. 2025. Group-in-group policy optimization for llm agent training. NeurIPS.

David Ha and Jürgen Schmidhuber. 2018. World models. arXiv preprint arXiv:1803.10122.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language model is planning with world model. EMNLP.

Xanh Ho et al. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In COLING.

Edward J Hu, Shen Yelong, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In ICLR.

Hansi Jin, Zhenrui Zeng, Yinsung Yue, Dong Yoon, Arik, Hamed Wang, Jiawei Zumani, and Jiawei Han. 2025. Search-r1: Training search agents with reinforcement learning. ICML.

Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2017. Billion-scale similarity search with gpus. IEEE Transactions on Big Data.

Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In ACL.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. 2019. Natural questions: A benchmark for question answering research. In TACL.

Guanting Li, Xiaoxiao Zhao, Zijun Wang, Linyao Pang, Yuechen Chen, Chong Quan, Jiaxin Shu, Hanyu Li, and Xingzhang Li. 2024. Self-play with execution feedback: Improving instruction-following capabilities of large language models. arXiv preprint arXiv:2406.13542.

Pandeng Li, Xiangchao Zhao, Yan Liu, and Tianyi He. 2025. Tree-search-based policy optimization under stochastic execution. arXiv preprint arXiv:2502.01523.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2023. Let’s verify step by step. arXiv preprint.

Youwei Liu, Jian Wang, Hanlin Liu, Beichen Guo, and Wenjie Li. 2026. Agent learning from adaptive lookahead world models. arXiv preprint arXiv:2601.08955.

Ning Lu, Baijiong Lin, Shengcai Liu, Jiahao Wu, Haoze Lv, Yanbin Wei, Lingting Zhu, Shengju Qian, Xin Wang, Ying-Cong Chen, Qi Wang, and Ke Tang. 2026. Policy and world modeling co-training for language agents. arXiv preprint arXiv:2606.02388.

Alex Mallen et al. 2023. Popqa: A dataset for popularity-based question answering. In EMNLP.

Runsheng Meng, Zigao Zhao, Shuang Liu, Zhaohan Fei, Yiqi Wang, Handong Li, Xinyuan Chen, Mengnan Wang, Yong Wang, and Jiayu Chen. 2024. Direct alignment of language models via quality-aware selfrefinement. arXiv preprint arXiv:2405.21040.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. NeurIPS.

Ofir Press et al. 2023. Measuring and narrowing the compositionality gap in language models. In EMNLP.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. In arXiv preprint.

Zhihong Shao et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. In arXiv preprint.

Vaishnavi Shrivastava, Piero Kauffmann, Ahmed Awadallah, and Dimitris Papailiopoulos. 2026. ECHO: Terminal agents learn world models for free. arXiv preprint arXiv:2605.24517.

Qwen Team. 2025. Qwen technical report. arXiv preprint.

Harsh Trivedi et al. 2022. Musique: Multihop questions via single-hop question composition. In TACL.

Liang Wang et al. 2022. Text embeddings by weakly-supervised contrastive pre-training. In arXiv preprint.

Ronald J Williams. 1992. Simple statistical gradientfollowing algorithms for connectionist reinforcement learning. Machine Learning, 8:229–256.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In EMNLP.

Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. 2022. Webshop: Towards scalable realworld web interaction with grounded language agents. In NeurIPS.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. React: Synergizing reasoning and acting in language models. In ICLR.

Xiao Yu, Baolin Peng, Ruize Shen, Pengcheng He, Suman Gao, Jianyong Kim, Jianfeng Gao, and Zhou Yu. 2026. Reinforcement world modeling for llmbased agents. arXiv preprint arXiv:2602.05842.

Kai Zhang, Xiangchao Chen, Bo Liu, Tianci Xue, Yufei Liao, Zhihan Liu, Xiyao Wang, Yuting Ning, Zhaorun Chen, Jian Xie, Yuxuan Sun, Boyu Yang, Jianwei Zhou, Tianyi Zhang, Xia Li, Ashish Shah, et al. 2025. Agent learning via early experience. arXiv preprint arXiv:2510.08558.

Peiyi Zhao, Xin Zhang, Yangyi Fang, Jiayin Chen, Tianwei Wang, Wei Wu, Hongyin Yuan, Keqing Chen, and Bowen Yu. 2025. Masked thought: Simply masking partial reasoning steps can improve mathematical reasoning learning of language models. arXiv preprint arXiv:2412.03729.

## A Implementation Details

Our experiments cover two types of tasks: Open-Domain Question Answering and Interactive Web Shopping.

## A.1 Datasets

## A.1.1 Open-Domain QA

Open-domain QA requires agents to answer questions by interacting with an external retrieval system without pre-given documents. We train on a mixed NQ + HotpotQA dataset and evaluate on multiple benchmarks.

Training Data. We mix the following datasets for training:

• Natural Questions (NQ) (Kwiatkowski et al., 2019): 79,168 training samples and 8,757 validation samples from real Google search queries

• HotpotQA (Yang et al., 2018): 90,447 training samples and 7,405 validation samples requiring cross-document reasoning

From the mixed dataset (169,615 training samples in total), we perform stratified sampling to obtain 27,648 samples for RL training and 1,024 samples for validation. The sampling strategy uses two-level stratification: (1) at the dataset level, NQ and HotpotQA are sampled with weights 1:4; (2) at the category level, within each dataset, samples are drawn proportionally to question categories (single-hop questions for NQ, type and level for HotpotQA). This ensures that the sampled dataset maintains the same distribution of hop counts and question types as the source dataset.

Test Datasets.

• Natural Questions (NQ): 3,610 questions evaluating single-hop fact retrieval

• TriviaQA (Joshi et al., 2017): 11,313 questions on trivia knowledge

• PopQA (Mallen et al., 2023): 14,267 questions focusing on popular culture entities

• HotpotQA: 7,405 questions requiring multi-hop reasoning

• 2WikiMultiHopQA (Ho et al., 2020): 12,576 questions requiring comparison of two entities

• MuSiQue (Trivedi et al., 2022): 2,417 questions designed to avoid shortcut reasoning

• Bamboogle (Press et al., 2023): 125 manually constructed difficult questions

Task Setup. Each question allows the agent at most 5 turns to complete the task. In each turn, the agent can execute search[query] (retrieve relevant documents) or finish[answer] (submit the final answer). The retrieval system is based on the Wiki18 corpus (a 2018 snapshot of approximately 5 million Wikipedia articles), using the E5- base-v2 encoder (Wang et al., 2022) and FAISS index (Johnson et al., 2017) for dense retrieval, returning the top-3 documents per search.

## A.1.2 Interactive Web Shopping (WebShop)

WebShop (Yao et al., 2022) simulates a realistic e-commerce environment with 1.18M products and 12,087 crowd-sourced shopping instructions. The agent needs to browse product pages and complete purchases according to user requirements (e.g., “buy a waterproof blue backpack under \$50”).

Training Data. We use all 10,587 instructions in the training split; 1,000 instructions are held out for validation and 500 for testing.

Task Setup. Each shopping task allows at most 15 steps to complete. The agent can execute search[query] (search for products), click[element] (click page elements), or buy (purchase item). Success criteria include: (1) purchasing the correct item; (2) item attributes (color, size, etc.) matching requirements; (3) price within budget.

## A.2 Training Details

## A.2.1 Model and LoRA Configuration

Base Models. Our main experiments use Qwen3- 1.7B and Qwen3-8B (Team, 2025) as base language models. Cross-model generalization experiments further test Qwen2.5-1.5B, Llama-3.2-3B, and Qwen2.5-7B.

LoRA Fine-tuning. To achieve parameterefficient fine-tuning, we apply LoRA (Hu et al., 2022) to all linear projection matrices in attention layers:

• LoRA rank: $r = 3 2$

• LoRA alpha: $\alpha = 3 2$

• Target modules: Q, K, V, O projections and all linear layers (all-linear)

## A.2.2 Reinforcement Learning Hyperparameters

We use GRPO (Shao et al., 2024) as our primary RL algorithm. Key hyperparameters are as follows:

## Batch Size and Sampling.

• Train batch size: 144

• Rollout group size: 8 (8 independent trajectories per group)

• Number of groups per rollout: 18 (144 trajectories total)

• PPO mini-batch size: 72

• PPO epochs: 1

## Temperature and Sampling Strategy.

• Rollout temperature: $T _ { \mathrm { r o l l o u t } } = 1 . 0$

• Validation temperature: $T _ { \mathrm { v a l } } = 0 . 0$ (greedy decoding)

## Optimizer and Learning Rate.

• Optimizer: AdamW $( \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9 , \epsilon =$ $1 0 ^ { - 8 } )$

• Learning rate: $5 \times 1 0 ^ { - 6 }$

• LR schedule: constant (QA); cosine decay (Web-Shop, 25% warmup, minimum LR ratio 0.1)

• Weight decay: 0.01

• Gradient clipping: maximum norm 1.0

## A.2.3 DGS-WM Hyperparameters

The core hyperparameters of our method are:

## Decision-Guided Observation Selection.

• Selection standard deviation coefficient: $\kappa = 0 . 5$

• Threshold formula: $\tau _ { t } = \mu _ { t } + \kappa \cdot \sigma _ { t }$ , so a candidate unit’s response support score must exceed the mean by κ standard deviations; $\mu _ { t }$ and $\sigma _ { t }$ are the mean and standard deviation of response support scores within the current observation

## Confidence-Adaptive Weighting.

• Confidence decay exponent: $\gamma = 1 . 0$ (linear decay)

• Weight formula: $w _ { t , j } = c _ { t } ^ { \gamma }$ , where $c _ { t }$ is the response confidence at turn t

World Modeling Loss Coefficient.

$\lambda _ { \mathrm { W M } } = 0 . 0 2$ (consistent with ECHO)

## A.2.4 Reward Function

We adopt sparse binary reward signals for all tasks:

## QA Task Rewards.

• Success (exact answer match): $R = 1$

• Failure (answer mismatch or exceeding turn limit): R = 0

• Invalid action penalty: −0.01 per invalid action (applied to corresponding action tokens in peraction mode)

## WebShop Task Rewards.

• Success (correct item purchased with all attributes fully matched): R = 1

• Failure (no purchase, wrong item, or attributes not fully matched): R = 0

• Invalid action penalty: −0.01 per invalid action (applied to corresponding action tokens in peraction mode)

## B Observation Unit Partitioning

## B.1 QA Tasks: Sentence Splitting

For retrieval document feedback, we split at sentence boundaries using forward sentence splitting with the following rules:

Split pattern: We use the regex pattern $( 2 < = [ \mathrm { ~ \bf ~ . ~ } \mathrm { ~ \bf ~ 2 ~ } ] ) \setminus \mathrm { s } + ( \mathrm { ~ \bf ~ 2 ~ } = [ \mathrm { A } - \mathrm { Z } 0 - 9 ^ { \mathrm { ~ \bf ~ n ~ } \prime } \setminus ( \mathrm { ~ \setminus ~ } [ \mathrm { ~ \bf ~ 2 ~ } ) )$

which splits after sentence-ending punctuation followed by whitespace and a capital letter or digit.

Abbreviation protection: Common abbreviations (Dr., U.S., Mr., Mrs., Ph.D., etc.) are protected from splits by checking whether the period is preceded by these patterns.

Single-letter protection: Single capital letters followed by periods (e.g., "A. Smith") are preserved as part of the same sentence to avoid breaking names or list markers.

Each resulting sentence becomes one candidate observation unit $c _ { t , j }$ with metadata tracking its paragraph ID, rank within the paragraph, sentence index, and character positions in the original observation.

## B.2 WebShop: Semantic UI Block Identification

For webpage observations, we identify semantic blocks based on page type inference:

Page type inference: We classify each page as one of: search\_results, product\_main, search\_home, or done based on page structure markers (search results list, product details section, search bar, confirmation message).

Semantic block construction by page type:

• search\_results page: Each product card becomes one block, containing ASIN, description, and price as a unit.

• product\_main page: Core purchase information (selected options + title + price + rating + "Buy Now" button) forms one block; product description paragraphs are split into sentences; customer review blocks remain intact.

• search\_home page: Search bar and category buttons each form separate blocks.

• Feedback blocks: System messages (e.g., "Added to cart") remain as complete units.

Fallback handling: For unrecognized page structures, interactive buttons remain as whole-line units, while plain text content is split into sentences using the QA sentence splitting rules.

## B.3 Granularity Design Rationale

The "minimal semantically complete unit" principle avoids two failure modes:

Over-coarse splitting (e.g., entire paragraphs or full product pages): Dilutes response support scores by mixing decision-relevant content with irrelevant context, reducing selection precision.

Over-fine splitting (e.g., individual words or tokens): Breaks semantic integrity, making isolated fragments uninformative even when the complete unit would support decisions.

Our task-aware unit definitions balance these concerns: sentences preserve propositional content in documents, while UI blocks maintain functional coherence in interactive environments.

## C Case Studies

This section presents examples of DGS-WM’s observation selection compiled from real training logs. Bold text, additionally highlighted in dark green within the observations, marks content selected as WM prediction targets.

## C.1 Single-Hop Question Answering (Qwen3-1.7B)

Question: What became the standard German term for the art song?

## Environment observation: 3 retrieved documents, 18 candidate sentences

[Document 10565234; rank 1] Title: Art song

Text: wordless vocalise written by a classical composer is sometimes considered an art song and sometimes not. Other factors help define art songs: Art songs have been composed in many languages, and are known by several names. The German tradition of art song composition is perhaps the most prominent one; it is known as "Lieder". In France, the term "mélodie" distinguishes art songs from other French vocal pieces referred to as chansons. The Spanish canción and the Italian canzone refer to songs generally and not specifically to art songs. The composer’s musical language and interpretation of the text often dictate the

[Document 12808472; rank 2] Title: Song

Text: traditions. German-speaking communities use the term art song ("Kunstlied") to distinguish so-called "serious" compositions from folk song ("Volkslied"). The lyrics are often written by a poet or lyricist and the music separately by a composer. Art songs may be more formally complicated than popular or folk songs, though many early Lieder by the likes of Franz Schubert are in simple strophic form. The accompaniment of European art songs is considered as an important part of the composition. Some art songs are so revered that they take on characteristics of national identification. Art songs emerge from the tradition of singing romantic

[Document 12168286; rank 3] Title: Art song

Text: Art song An art song is a vocal music composition, usually written for one voice with piano accompancement, and usually in the classical art music tradition. By extension, the term "art song" is used to refer to the collective genre of such songs (e.g., the "art song repertoire"). An art song is most often a musical setting of an independent poem or text, "intended for the concert repertory" "as part of a recital or other relatively formal social occasion". While many pieces of vocal music are easily recognized as art songs, others are more difficult to categorize. For example, a

Selection: 18 candidate sentences → 2 selected sentences (11.1%)

Subsequent policy response: <answer>Lieder</answer>

## C.2 Multi-Hop Question Answering (Qwen3-8B)

Question: The director of the romantic comedy "Big Stone Gap" is based in what New York city?

Environment observation: 3 retrieved documents, 27 candidate sentences

[Document 11856191; rank 1] Title: Big Stone Gap (film)

Text: Big Stone Gap is a 2014 American drama romantic comedy film written and directed by Adriana Trigiani and produced by Donna Gigliotti for Altar Identity Studios, a subsidiary of Media Society. Based on Trigiani’s 2000 best-selling novel of the same name, the story is set in the actual Virginia town of Big Stone Gap circa 1970s. The film had its world premiere at the Virginia Film Festival on November 6, 2014.

[Document 21947720; rank 2] Title: Adriana Trigiani

Text: Adriana Trigiani is an Italian American best-selling author of eighteen books, television writer, film director and entrepreneur based in Greenwich Village, New York City. She is also a published playwright. Trigiani has published a novel a year since 2000. Her tenth novel, "The Shoemaker’s Wife," her first historical novel, was published March 27, 2012, by Harper Collins. It debuted at number one on the "New York Times" Bestseller list and remained on the list for five weeks.

[Document 8647293; rank 3] Title: Big Stone Gap, Virginia

Text: Big Stone Gap is a town in Wise County, Virginia, United States. The population was 5,643 at the 2010 census. The town is home to a campus of Mountain Empire Community College. It is also home to author and film producer Adriana Trigiani who wrote and filmed the movie "Big Stone Gap" based on her novel of the same name.

Selection: 27 candidate sentences → 2 selected sentences (7.4%)

Subsequent policy response: <answer>Greenwich Village</answer>

Interpretation. This question requires reasoning across documents: the first document identifies Adriana Trigiani as the director, and the second gives her location. DGS-WM selects only these two key sentences.

## C.3 Product Search (Qwen3-8B, WebShop)

Task: i need a long lastingfragrance gift setfor women, and price lower than 80.00 dollars Current step: Select a suitable product from the search results to click.

## Search results

10 products, each with a title, price, and brief description; approximately 50 candidate UI blocks. [Product 1]   
Floral Summer Eau de Toilette 100ml   
\$45.99   
Light floral scent, perfect for daily wear   
[Product 2]   
Luxury Fragrance Gift Set for Women   
\$64.99   
Long-lasting Eau de Parfum + Body Lotion + Travel Spray   
[Product 3]   
Classic Rose Perfume 50ml   
\$89.00   
Premium rose essence, elegant bottle design   
[Product 4]   
Travel Mini Perfume Collection   
\$29.99   
5 x 10ml mini bottles, various scents   
[Product 5]   
Premium Women’s Fragrance Duo   
\$72.50   
Lasting scent, gift-ready packaging   
[Product 6]   
Fresh Ocean Breeze Spray 200ml   
\$38.00   
Refreshing aquatic scent   
[4 other products omitted...]

Interpretation. The agent selects a product that satisfies all task constraints: a long-lasting fragrance gift set for women priced below \$80.

## C.4 Product-Detail Confirmation (Qwen3-8B, WebShop)

Task: i need a long lastingfragrance gift setfor women, and price lower than 80.00 dollars Current step: After clicking the product, check its details against the task requirements.

<table><tr><td colspan="2">Product details: 64 candidate UI blocks</td></tr><tr><td>Product Details: [Core Info]</td><td>[Product Specifications] Fragrance Family: Floral Oriental</td></tr><tr><td>Name: Luxury Fragrance Gift Set for Women</td><td>Top Notes: Bergamot, Mandarin</td></tr><tr><td>Price: $64.99 Set Includes: Eau de Parfum 50ml + Body</td><td>Heart Notes: Jasmine, Rose Base Notes: Vanilla, Musk</td></tr><tr><td>Lotion 75ml + Travel Spray 10ml</td><td>Bottle Material: Glass with gold-tone cap Made in France</td></tr><tr><td>[Shipping] Free shipping on orders over $50</td><td>[Key Features]</td></tr><tr><td>Estimated delivery: 3-5 business days</td><td>• Long-lasting formula (8-10 hours)</td></tr><tr><td>30-day return policy</td><td>• Gift-ready elegant packaging</td></tr><tr><td>[Customer Reviews - Top Rated]</td><td>• Suitable for all occasions</td></tr><tr><td>★★★★★ &quot;Lasts all day! Great for gifting.&#x27; 1</td><td></td></tr><tr><td></td><td>[Recommendations]</td></tr><tr><td>★★★★★ &quot;Perfect gift set, my mom loved it.&quot;</td><td>You may also like:</td></tr><tr><td>★★★☆☆&quot;Nice scent but bottle design could be better.&quot;</td><td>- Similar Perfume Set ($79.99) - Luxury Body Care Collection ($54.99)</td></tr><tr><td>[Brand Story]</td><td>- Travel Size Sampler ($29.99)</td></tr><tr><td>Founded in 1985, our brand has been crafting luxury fragrances for over 35 years. Each fragrance is carefully formulated using premium ingredients sourced from around the world. Our</td><td>[Size Options] • 50ml Set - $64.99 (Selected) • 100ml Set - $94.99</td></tr></table>

Selection: 64 UI blocks → 8 selected (12.5%)

Subsequent policy response: <buy>

Interpretation. The agent focuses on information needed to verify the task requirements (set contents, longevity, and price), ignoring secondary content such as the brand story, ingredient details, and recommended products.

Summary. These cases illustrate how DGS-WM selectively focuses on decision-relevant observations across tasks, substantially reducing the scope of world-modeling supervision (selection rates of 7.4–12.5%) while preserving correct policy responses.