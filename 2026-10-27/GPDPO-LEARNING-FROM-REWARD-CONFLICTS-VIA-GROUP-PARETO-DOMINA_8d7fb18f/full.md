# GPDPO: LEARNING FROM REWARD CONFLICTS VIA GROUP PARETO-DOMINANCE POLICY OPTIMIZATION

Zixuan Xu<sup>1∗</sup> Haoxuan Chen<sup>1∗</sup> Jiangxin Sun<sup>2</sup> Jian-Fang Hu<sup>1†</sup>

<sup>1</sup>ISEE Lab, Sun Yat-sen University

<sup>2</sup>Computer Science and Technology, University of Trento

{chenhx253, xuzx57}@mail2.sysu.edu.cn, hujf5@mail.sysu.edu.cn

## ABSTRACT

Multi-reward reinforcement learning offers a promising framework for aligning large language models (LLMs) with diverse objectives, but conflicting reward signals can induce ambiguous update directions and hinder stable optimization. Existing approaches address such conflicts through reward reweighting, gradient coordination, or rollout filtering, yet they primarily operate on perresponse or group-level optimization signals and underexploit relational information among sampled responses. As a result, conflict-filtered rollouts may still contain useful supervision that is discarded during training. We propose Group Pareto-Dominance Policy Optimization (GPDPO), a multi-reward policy optimization method that recovers pairwise supervision from conflict-filtered rollouts. GPDPO identifies Pareto-dominance relations among sampled responses and converts them into auxiliary preference signals, which are jointly optimized with the original policy objective. This enables more effective reuse of existing rollouts without additional sampling or reward evaluation. Experiments on API-Bank across multiple model scales and reward configurations show that GPDPO consistently outperforms competitive multi-reward optimization baselines. Further analysis demonstrates that GPDPO reduces both group-relative and pairwise reward conflicts, supporting the effectiveness of exploiting relational supervision for multi-reward alignment. Code will be released.

## 1 INTRODUCTION

Reinforcement learning provides an effective framework for improving large language models (LLMs) by optimizing task-specific rewards (Shao et al., 2024; Liu et al., 2026b). Group-based policy optimization methods such as GRPO (Shao et al., 2024) construct policy updates from relative rewards among sampled responses. Extending this paradigm to multiple reward signals enables LLMs to jointly optimize diverse alignment objectives, such as task correctness, output format, and response length (Liu et al., 2026b;a). However, multi-reward optimization introduces a distinct challenge: different reward dimensions may provide conflicting signals for the same response.

Existing methods address such conflicts from several perspectives, including reward reweighting (Yuan et al., 2026), gradient coordination (Li et al., 2025a), reward-wise advantage normalization (Liu et al., 2026b), and conflict-aware rollout filtering (Liu et al., 2026a). These approaches improve the compatibility of multiple reward signals by modifying how rewards, advantages, gradients, or samples contribute to policy optimization. In particular, conflict-aware filtering removes responses whose reward-wise advantages disagree in sign, preventing ambiguous signals from directly entering the policy update. While effective for conflict removal, this response-level filtering is coarse: once a response is deemed conflicting, all supervision associated with it is discarded. This raises an important question: if a response is excluded from the policy update due to conflictfiltering, does that necessarily mean it contains no useful information for learning?

Our observation shows that it does not. A response with conflicting group-relative advantages may still form a reliable ordering relative to another. In particular, one response may consistently dominate another across all reward dimensions, yielding an unambiguous Pareto-consistent preference even when the conflicting response itself is unsuitable for a policy update. As illustrated in Figure 1, this reveals a distinction between individual update and pairwise comparison: a response can be unreliable as a direct optimization signal while remaining informative as relational supervision.

![](images/f28880a34955ea8b866a338b836558d81ae77a13c954cf39c51db81205995462.jpg)  
Figure 1: Different strategies for utilizing conflicting responses in multi-reward policy optimization. GDPO retains conflicting responses in policy updates, whereas GD<sup>2</sup>PO masks them to avoid ambiguous optimization signals. GPDPO preserves conflict-aware masking while recovering pairwise supervision from Pareto-dominance relations, i.e., one response is no worse than another on all reward dimensions and strictly better on at least one.

Motivated by this, we introduce GPDPO, a multi-reward policy optimization framework that preserves conflict-aware masking for direct updates while recovering pairwise supervision from Paretodominance relations involving filtered rollouts. Unlike prior approaches that primarily modify perresponse optimization signals, such as rewards, advantages, or gradients (Yuan et al., 2026; Li et al., 2025a; Liu et al., 2026b;a), GPDPO explicitly exploits the relational structure among sampled responses. Specifically, GPDPO identifies Pareto-dominance relations among rollouts and transforms these relations involving filtered responses into a pairwise preference loss, which encourages the policy to favor responses with consistent improvements across reward dimensions. By leveraging such supervision from filtered rollouts, GPDPO extracts additional training signals from existing samples without extra sampling or reward evaluation. This enables the policy to suppress ambiguous update directions while retaining informative pairwise preferences that would otherwise be discarded. Extensive experiments on API-Bank (Li et al., 2023), across two Qwen3 model scales (Yang et al., 2025) and two reward configurations, show that GPDPO consistently outperforms competitive baselines in multi-reward optimization.

Our contributions are summarized as follows: 1) We show that responses with conflicting grouprelative advantages can still yield reliable pairwise preferences, revealing supervision overlooked by response-level conflict filtering. 2) We propose GPDPO, which preserves conflict-aware masking for direct updates while recovering pairwise supervision from filtered responses through Paretodominance relations. 3) Extensive experiments on API-Bank across varied model scales and reward configurations show that GPDPO consistently outperforms competitive multi-reward baselines.

## 2 RELATED WORK

Reinforcement learning with verifiable rewards. RLVR has shown strong potential for improving LLM reasoning (Shao et al., 2024; Guo et al., 2025). Prior work advances this paradigm through group-relative advantage estimation (Shao et al., 2024), sampling and objective refinement (Yu et al., 2026; Liu et al., 2025b), and sequence-level policy optimization (Zheng et al., 2025). These methods improve the efficiency and stability of reasoning optimization. However, applying GRPO-style algorithms to multiple rewards through scalar aggregation may obscure reward-specific distinctions, limiting their effectiveness in multi-reward optimization (Liu et al., 2026b).

Multi-reward policy optimization. Prior work coordinates multiple rewards through fixed or adaptive weighting (Zhou et al., 2024; Lu et al., 2026), gradient balancing (Li et al., 2025a), and rewardwise or blockwise advantage estimation (Liu et al., 2026b; Pavlenko et al., 2026). GD<sup>2</sup>PO further filters rollouts with conflicting reward-wise advantages (Liu et al., 2026a). These methods mitigate multi-reward conflicts by adjusting how reward signals contribute to policy updates. However, response-level conflict handling may overlook consistent inter-response relations and discard useful supervision. GPDPO instead complements group-relative conflict handling with pairwise comparisons, recovering reward-consistent supervision from filtered rollouts.

Pareto dominance relations. Pareto dominance is widely used in multi-objective search, including NSGA-II (Emmerich & Deutz, 2018; Deb et al., 2002). In LLM alignment, related methods use Pareto or reward-consistency criteria to assign trajectory advantages (Li et al., 2026), construct preference pairs (Li et al., 2025b; Xu et al., 2026), or downweight conflicting pairs (Xie et al., 2026). These methods use Pareto dominance to guide multi-objective learning. Insipred by these, GPDPO uses Pareto dominance to identify consistent comparison structures among rollouts and draws on these structures to provide supplementary supervision for multi-reward optimization.

## 3 METHOD

In this section, we first introduce multi-reward optimization and the associated conflict-filtering mechanism (Section 3.1). We then revisit the sign-based filtering criterion, identify two potential limitations, and empirically examine their effects (Section 3.2). Building on these findings, we propose GPDPO, which exploits Pareto-dominance relations between responses to recover finegrained pairwise supervision for improved conflict mitigation (Section 3.3).

## 3.1 PRELIMINARIES

Multi-reward group-relative optimization. Given a prompt x, the sampling policy $\pi _ { \theta _ { \mathrm { o l d } } }$ generates a group of $G$ responses, denoted by $\mathcal { G } ( x ) = \{ y _ { 1 } , . . . , y _ { G } \}$ . Each response $y _ { n }$ receives an M-dimensional reward vector $\mathbf { r } _ { n } = ( r _ { n } ^ { 1 } , \cdot \cdot \cdot , \dot { r } _ { n } ^ { M } )$ , where $r _ { n } ^ { i } = r _ { i } ( x , y _ { n } )$ and higher values are preferred on every reward dimension. A straightforward way to extend GRPO (Shao et al., 2024) to the multi-reward setting is to aggregate the rewards before applying group-relative normalization:

$$
R _ { n } = \sum _ { i = 1 } ^ { M } w _ { i } r _ { n } ^ { i } , \qquad { \overline { { R } } } = { \frac { 1 } { G } } \sum _ { n = 1 } ^ { G } R _ { n } , \qquad A _ { n } ^ { \mathrm { G R P O } } = { \frac { R _ { n } - { \overline { { R } } } } { s _ { R } } } .\tag{1}
$$

Here, $w _ { i }$ is the weight of reward dimension $i ,$ and $s _ { R } > 0$ is the within-group normalization scale. The resulting scalar advantage is used in the clipped policy objective (Schulman et al., 2017) to update the probability of the corresponding response. GDPO (Liu et al., 2026b) observes that aggregating rewards too early can lose information specific to individual reward dimensions. It therefore normalizes each reward separately, aggregates the resulting advantages, and applies batch-level scale normalization. $\mathrm { G D ^ { 2 } P O }$ (Liu et al., 2026a) further addresses conflicts among these advantages: when a response has both positive and negative advantages across reward dimensions, direct aggregation can cause the learning signals to cancel each other out. $\mathrm { G D ^ { 2 } P O }$ therefore filters responses with conflicting advantage signs and adjusts the update through reweighting. Specifically, the reward-wise advantages and filtering gate are defined as

$$
A _ { n } ^ { i } = \frac { r _ { n } ^ { i } - \mu _ { i } } { s _ { i } } , \qquad \mu _ { i } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } r _ { j } ^ { i } , \qquad g ( y _ { n } ) = { \bf 1 } \bigl \{ \mathrm { s i g n } ( A _ { n } ^ { i } ) \mathrm { s i g n } ( A _ { n } ^ { k } ) \geq 0 , \forall i , k \bigr \} .\tag{2}
$$

Here, $s _ { i }$ denotes the normalization scale for reward dimension i. The binary gate $g ( y _ { n } )$ equals 1 for retained responses and 0 for filtered responses. Specifically, the gate selects responses whose reward-wise advantages share a consistent sign across dimensions, while removing the contribution of responses with conflicting directions.

Pareto dominance. Pareto dominance compares two vectors in a multi-objective space. a Paretodominates b if a is at least as good as b in all dimensions and strictly better in at least one dimension (Emmerich & Deutz, 2018):

$$
\mathbf { a } \succ \mathbf { p } \ \mathbf { b } \iff \ \left( \forall i : a _ { i } \geq b _ { i } \right) \ \wedge \ \left( \exists i : a _ { i } > b _ { i } \right) .\tag{3}
$$

Pareto dominance does not allow gains on one objective to compensate for losses on another. Thus, when two vectors are better on different objectives, neither dominates the other.

![](images/0a18b414d36cdc051548df67f6e2c3342dc1871da85986bc79eebafc1d92a4a2.jpg)  
(a) Conflict-label flip rate: different group size

![](images/6ec2a7af3221265b8bfca3032fef5fb3aa849e1259dd4e0a811b819c584ac7de.jpg)  
(b) Conflict-label flip rate: same group size

![](images/c3f441d5706dca3a6f5e0a79cc1a6efff4f7f1e65b47c99a28f39677d7bf213b.jpg)  
(c) GT matches among filtered responses

![](images/4bc4f39b4e093159e1f8fc7cf7e00abf8a628473b363b788352a7beff615dbc3.jpg)  
(d) Consistent comparisons among filtered response  
Figure 2: Conflict filtering is sensitive to the reference group and can discard useful learning signals. (a,b) Conflict-label flip rates for fixed responses under different group sizes and different peers at the same group size, respectively. (c) Percentage of filtered responses that match the ground-truth. (d) Consistent comparison relations between filtered responses and their peers. “Both” denotes being consistently better than some peers and consistently worse than others.

## 3.2 RETHINKING CONFLICT FILTERING IN MULTI-REWARD RL

Here, we revisit conflict filtering in multi-reward RL from the perspective of group-relative advantages. Our analysis reveals two limitations: filtering decisions are sensitive to the reference group, and responses removed due to group-wise conflicts may still contain useful pairwise supervision. Through empirical studies, we show that reward conflicts at the group level do not necessarily imply the absence of consistent response-level comparisons.

Prior work (Liu et al., 2026a) filters responses whose advantages disagree in sign across reward dimensions, which can be rewritten as follows:

$$
\operatorname { s i g n } ( A _ { n } ^ { i } ) = \operatorname { s i g n } ( r _ { n } ^ { i } - \mu _ { i } ) = \operatorname { s i g n } \left[ \sum _ { j = 1 } ^ { G } ( r _ { n } ^ { i } - r _ { j } ^ { i } ) \right] .\tag{4}
$$

This formulation shows that the sign of a response’s advantage depends on the sum of its reward differences from the other responses in the group. Conflict filtering based on these group-level signs has two limitations. First, conflict filtering is sensitive to the reference group: the same response can receive different filtering decisions as its peers change. Moreover, group-wise conflict filtering can discard useful learning signals. Coarse-grained conflict signals obtained by summing withingroup reward differences cannot capture fine-grained comparisons between responses, leaving useful pairwise training signals unrecognized by the filter. To further analyze these limitations, we conduct the empirical experiments shown in Figure 2. Detailed experiments are provided in the appendix.

Conflict filtering is sensitive to the rollout group. We hold the target responses fixed, vary the reference group size, and examine whether the filtering decisions change. As shown in Figure 2(a), expanding the original group from $G = 4$ to $G = 6 4$ changes the conflict filtering decisions for 16.19% and 27.09% of the fixed targets under 2R and 3R, respectively. Furthermore, we keep the group size fixed and replace the peer responses. Figure 2(b) shows decision inconsistency rates of 9.15% and 15.92% at $\dot { G } = 4$ , which remain at 5.43% and 6.09% at $G = 6 4$ . These results show that conflict filtering signals are highly sensitive to the reference group.

Group-wise conflict filtering can discard useful learning signals. We further analyze responses filtered due to reward conflicts. As shown in Figure 2(c), 23.22% and 27.33% of these responses under 2R and 3R, respectively, exactly match the ground-truth (GT) ordered tool-call structure. This shows that filtered responses can still contain valuable training signals. We then investigate the finegrained information in these responses. As shown in Figure 2(d), 88.42% and 86.67% of filtered responses under 2R and 3R, respectively, have a consistent comparison relation with at least one other response in the group, yet these relations are discarded by group-wise conflict filtering. These findings show that group-wise conflict signals lose substantial fine-grained information, causing the filter to incorrectly discard many useful training samples and signals.

These observations lead to our motivation. First, conflict filtering signals should be decoupled from the group mean to improve training stability. Moreover, conflict detection should go beyond grouplevel relations to exploit comparisons between individual rollouts, which depend only on the properties of the two rollouts themselves. Finally, even when a response has conflicting advantages relative to the group mean, pairwise comparisons with individual rollouts can still reveal consistent preference relations and provide useful supervision.

![](images/b8a2a4b94c90a40276d881186c8421aa2c5b26fc0c4d5ddfdedcb3c1f7db9a75.jpg)  
Figure 3: Overview of GPDPO. Based on conflict filtering, we identify Pareto dominance relations within filtered responses and leverage them to construct preference loss. The resulting preference loss is combined with the standard multi-reward policy-gradient objective for policy optimization.

## 3.3 GPDPO: GROUP PARETO-DOMINANCE POLICY OPTIMIZATION

In this section, we introduce GPDPO, a multi-reward policy optimization method that recovers pairwise supervision from conflict-filtered rollouts through Pareto dominance. Rather than relying solely on group-level conflict handling, GPDPO exploits consistent preference relations between individual responses and incorporates the resulting pairwise supervision into the original policy objective. As illustrated in Figure 3, GPDPO consists of three components: (1) Pareto-dominance identification, (2) preference-loss construction, and (3) joint policy optimization.

Pareto-dominance identification. Based on the conflict filtering mechanism of prior work (Liu et al., 2026a), we obtain the filtered set ${ \mathcal { F } } ( x ) ~ = ~ \{ y _ { n } ~ \in ~ { \mathcal { G } } ( x ) ~ : ~ c ( y _ { n } ) ~ = ~ 1 \}$ and retained set $\mathcal { R } ( x ) = \mathcal { G } ( x ) \setminus \mathcal { F } ( x )$ from the sampled group $\mathcal G ( x )$ . Here, $c ( y _ { n } ) = 1 - g ( \bar { y } _ { n } ) = 1$ indicates that the response is filtered due to conflict. We define a consistent preference relation based on Pareto dominance (Emmerich & Deutz, 2018). Following Section 3.1, for two responses $y _ { a }$ and $y _ { b }$ to the same prompt $x ,$ we say that $y _ { a }$ Pareto-dominates $y _ { b } .$ , if and only if $\mathbf { r } _ { a } \succ \mathbf { P } \ \mathbf { r } _ { b }$ . For notational simplicity, we use $y _ { a } \succ \mathrm { P } \ y _ { b }$ to denote this relation between responses $y _ { a }$ and $y _ { b }$ .

Pareto dominance depends only on the relative reward vectors of the two responses, independent of the group mean or other rollouts. This pairwise criterion therefore exposes reward-consistent, conflict-free comparison structures that can remain hidden under group-level aggregation.

We collect all Pareto dominance relations within the sampled group into the following set:

$$
\begin{array} { r } { \mathcal { D } ( x ) = \big \{ ( y _ { w } , y _ { l } ) \in \mathcal { G } ( x ) ^ { 2 } : y _ { w } \succ _ { \mathrm { P } } y _ { l } \big \} . } \end{array}\tag{5}
$$

Here, $y _ { w }$ represents the dominating response and y<sub>l</sub> represents the dominated response.

Preference-loss construction. To convert the dominance set into optimization signals, we construct a preference-pair set by selecting dominance edges in ${ \mathcal { D } } ( x )$ involving at least one response in the filtered set $\mathcal { F } ( x )$

$$
\mathcal { E } ( x ) = \big \{ ( y _ { w } , y _ { l } ) \in \mathcal { D } ( x ) : y _ { w } \in \mathcal { F } ( x ) \ \vee \ y _ { l } \in \mathcal { F } ( x ) \big \} .\tag{6}
$$

This selection focuses the auxiliary objective on supervision recovered from filtered rollouts, while retaining the original policy optimization pathway for non-filtered responses.

We then define the mean token-level log-probability ratio to measure the policy update of a response relative to the sampling policy:

$$
u _ { \theta } ( y \mid x ) = { \frac { 1 } { | y | } } \sum _ { t = 1 } ^ { | y | } \log { \frac { \pi _ { \theta } ( y _ { t } \mid x , y _ { < t } ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } \mid x , y _ { < t } ) } } .\tag{7}
$$

Here, |y| denotes the number of tokens in the response, and $\pi _ { \theta _ { \mathrm { o l d } } }$ is the sampling policy used to generate the rollout group. The score $u _ { \theta } ( y \mid x )$ captures the relative change in the response likelihood under the updated policy.

To ensure that the auxiliary objective updates only on filtered responses, we stop gradients through retained responses when they participate in selected preference pairs:

$$
\widetilde { u } _ { \theta } ( y \mid x ) = \left\{ \begin{array} { l l } { u _ { \theta } ( y \mid x ) , } & { y \in \mathcal { F } ( x ) , } \\ { \mathrm { s g } [ u _ { \theta } ( y \mid x ) ] , } & { y \notin \mathcal { F } ( x ) , } \end{array} \right.\tag{8}
$$

where sg denotes the stop-gradient operator. Retained responses serve as fixed comparison anchors, whereas filtered responses receive the preference supervision. When both endpoints of a preference pair are filtered, both responses contribute gradients to the auxiliary update.

For each selected dominance relation, we optimize the following pairwise preference objective:

$$
\ell _ { \mathrm { p a i r } } ( y _ { w } , y _ { l } \mid x ) = - \log \sigma ( \beta \left[ \widetilde { u } _ { \boldsymbol { \theta } } ( y _ { w } \mid x ) - \widetilde { u } _ { \boldsymbol { \theta } } ( y _ { l } \mid x ) \right] ) ,\tag{9}
$$

where $y _ { w }$ and $y _ { l }$ denote the dominating and dominated responses, respectively. This objective encourages the policy to increase the relative likelihood of responses consistent with Pareto dominance, thereby incorporating reward-consistent supervision into policy optimization. Here, $\sigma ( \cdot )$ is the sigmoid function and $\beta > 0$ controls the preference margin.

Joint policy optimization. For each training batch $B ,$ we assign equal weight to all selected preference pairs and define the auxiliary objective as:

$$
{ \mathcal { L } } _ { \mathrm { p a i r } } = { \frac { \sum _ { x \in { \mathcal { B } } } \sum _ { ( y _ { w } , y _ { l } ) \in { \mathcal { E } } ( x ) } \ell _ { \mathrm { p a i r } } { \big ( } y _ { w } , y _ { l } \mid x { \big ) } } { \sum _ { x \in { \mathcal { B } } } | { \mathcal { E } } ( x ) | } } .\tag{10}
$$

We finally formulate the GPDPO objective by combining the original training loss with the auxiliary pairwise objective, weighted by λ:

$$
{ \mathcal { L } } _ { \mathrm { G P D P O } } = { \mathcal { L } } _ { \mathrm { s t a n d a r d } } + \lambda { \mathcal { L } } _ { \mathrm { p a i r } } ,\tag{11}
$$

where $\mathcal { L } _ { \mathrm { s t a n d a r d } }$ denotes the standard multi-reward policy-gradient objective used in GDPO (Liu et al., 2026b) and $\mathrm { G D ^ { 2 } P O }$ (Liu et al., 2026a), and λ controls the contribution of the pairwise loss.

Our GPDPO recovers conflict-related supervision at a finer granularity by operating on pairwise relations between rollouts. Responses filtered by group-level conflict criterion may still form rewardconsistent comparisons with specific peers. GPDPO identifies and exploits these fine-grained relations, reusing supervision already present in the rollout group without additional sampling or reward evaluation. The complete algorithm is provided in the appendix.

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Task and rewards. We study multi-reward tool-call policy optimization on the 3,920 training examples from RLLA-4K (Qian et al., 2026). The upstream ToolRL training mixture draws on ToolACE (Liu et al., 2025a), Hammer (Masked) (Lin et al., 2024), and xLAM (Zhang et al., 2025). In the two-reward setting (2R), the reward consists of correctness and length; the three-reward setting (3R) additionally includes format. The correctness reward measures agreement between generated and annotated tool calls in tool names, argument names, and argument values, and is scaled to [−3, 3]. The length reward is defined as min $\mathsf { \Omega } _ { \mathrm { l } } ( n _ { \mathrm { t h i n k } } / 5 1 2 , 1 )$ , where $n _ { \mathrm { t h i n k } }$ denotes the number of words inside the <think> block; responses without a <think> block receive zero. The format reward is binary and indicates whether the response follows the prescribed output-tag structure. Unless otherwise stated, all reward dimensions are weighted equally.

Model training and evaluation. We evaluate Qwen3-1.7B and Qwen3-4B (Yang et al., 2025), comparing our GPDPO against GRPO (Shao et al., 2024), GDPO (Liu et al., 2026b), and $\mathrm { G D ^ { 2 } P O }$ (Liu et al., 2026a). GRPO aggregates reward dimensions before normalization, whereas GDPO normalizes each reward dimension separately before aggregation; $\mathrm { G D ^ { 2 } P O }$ further introduces conflict filtering and query-level reweighting. All methods are trained for 200 rollout steps with 512 prompts per step and $G = 4$ responses per prompt. We use one epoch per step, mini-batches of 128 prompts, and a learning rate of $\bar { 1 0 } ^ { - 6 }$ . Responses are sampled with temperature 1.0, without a KL penalty or entropy bonus. The maximum prompt and response lengths are 2,048 and 1,024 tokens, respectively. Our implementation is built on verl (Sheng et al., 2025) and vLLM (Kwon et al., 2023). Unless otherwise specified, GPDPO uses uniform pair weighting with $\lambda = 0 . 1$ and $\beta = 0 . 1$ , and refreshes the sampling-policy reference after each rollout step.

We evaluate on API-Bank (Li et al., 2023), which contains 597 questions across Levels 1/2/3, with 399/67/131 questions, respectively. Each row in the main results table corresponds to a single evaluation of the selected step-200 checkpoint using greedy decoding and a maximum generation length of 4,096 tokens. We report accuracy for each difficulty level, overall accuracy across all questions (CORRECT ACC.), format accuracy in the 3R setting, and mean length reward. Accuracy metrics are reported as percentages, while length reward remains on its original [0, 1] scale. The OVER-ALL score is defined as CORRECT ACC./100 + LENGTH REW. for 2R, and additionally includes FORMAT ACC./100 for 3R. We treat CORRECT ACC. as the primary measure of task performance.

Table 1: Results on API-Bank under the two-reward (2R) setting. Accuracy metrics are reported as percentages. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Level 1 Acc.</td><td>Level 2 Acc.</td><td>Level 3 Acc.</td><td>Correct Acc.</td><td>Length Rew.</td><td>Overall</td></tr><tr><td>Qwen3-1.7B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO(Shao et al., 2024)</td><td>66.17</td><td>53.73</td><td>46.56</td><td>60.47</td><td>0.9643</td><td>1.569</td></tr><tr><td>GDPO (Liu et al., 2026b)</td><td>66.67</td><td>61.19</td><td>43.51</td><td>60.97</td><td>0.9875</td><td>1.597</td></tr><tr><td>GD²PO (Liu et al., 2026a)</td><td>69.17</td><td>61.19</td><td>42.75</td><td>62.48</td><td>0.9793</td><td>1.604</td></tr><tr><td>GPDPO</td><td>70.93</td><td>61.19</td><td>48.85</td><td>64.99</td><td>0.9812</td><td>1.631</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO (Shao et al., 2024)</td><td>71.93</td><td>59.70</td><td>45.80</td><td>64.82</td><td>0.9898</td><td>1.638</td></tr><tr><td>GDPO (Liu et al., 2026b)</td><td>72.93</td><td>56.72</td><td>47.33</td><td>65.49</td><td>0.9978</td><td>1.653</td></tr><tr><td>GD²PO (Liu et al., 2026a)</td><td>71.68</td><td>59.70</td><td>45.80</td><td>64.66</td><td>0.9987</td><td>1.645</td></tr><tr><td>GPDPO</td><td>72.93</td><td>64.18</td><td>45.80</td><td>66.00</td><td>0.9995</td><td>1.659</td></tr></table>

Table 2: Results on API-Bank under the three-reward (3R) setting. Accuracy metrics are reported as percentages. The best and second-best results are highlighted in bold and underlined, respectively.
<table><tr><td>Method</td><td>Level 1 Acc.</td><td>Level 2 Acc.</td><td>Level 3 Acc.</td><td>Correct Acc.</td><td>Format Acc.</td><td>Length Rew.</td><td>Overall</td></tr><tr><td>Qwen3-1.7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO(Shao et al., 2024)</td><td>65.91</td><td>62.69</td><td>39.69</td><td>59.80</td><td>98.99</td><td>0.9519</td><td>2.540</td></tr><tr><td>GDPO (Liu et al., 2026b)</td><td>68.92</td><td>53.73</td><td>41.22</td><td>61.14</td><td>100.00</td><td>0.9608</td><td>2.572</td></tr><tr><td>GD²PO (Liu et al., 2026a)</td><td>68.92</td><td>59.70</td><td>44.27</td><td>62.48</td><td>100.00</td><td>0.9756</td><td>2.600</td></tr><tr><td>GPDPO</td><td>70.43</td><td>61.19</td><td>42.75</td><td>63.32</td><td>99.66</td><td>0.9760</td><td>2.606</td></tr><tr><td>Qwen3-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GRPO(Shao et al., 2024)</td><td>70.93</td><td>58.21</td><td>46.56</td><td>64.15</td><td>98.99</td><td>0.9874</td><td>2.619</td></tr><tr><td>GDPO (Liu et al., 2026b)</td><td>68.42</td><td>56.72</td><td>46.56</td><td>62.31</td><td>99.83</td><td>0.9976</td><td>2.619</td></tr><tr><td>GD²PO (Liu et al., 2026a)</td><td>71.18</td><td>56.72</td><td>42.75</td><td>63.32</td><td>100.00</td><td>0.9967</td><td>2.630</td></tr><tr><td>GPDPO</td><td>71.43</td><td>64.18</td><td>47.33</td><td>65.33</td><td>100.00</td><td>0.9839</td><td>2.637</td></tr></table>

## 4.2 MAIN RESULTS

As shown in Tables 1 and 2, GPDPO attains the highest Correct Acc. and Overall score across all four combinations of model scale and reward setting. Under 2R, GPDPO improves Correct Acc. over $\mathrm { G D ^ { 2 } P O }$ from 62.48% to 64.99% on Qwen3-1.7B and from 64.66% to 66.00% on Qwen3- 4B, corresponding to gains of 2.51 and 1.34 percentage points, respectively. The improvement is particularly pronounced on the more challenging Level 3 subset for Qwen3-1.7B, where accuracy increases from 42.75% to 48.85%, a gain of 6.10 points. These results demonstrate that GPDPO effectively extracts useful supervision from reward conflicts, leading to improvements in the primary task objective rather than gains driven solely by auxiliary rewards or the aggregate Overall score.

The advantage persists under the more demanding 3R setting. GPDPO achieves 63.32% Correct Acc. on Qwen3-1.7B and 65.33% on Qwen3-4B, outperforming $\mathrm { G D ^ { 2 } P O }$ by 0.84 and 2.01 percentage points, respectively. Importantly, format accuracy is already near saturation across methods, yet GPDPO still improves Correct Acc. and attains the highest Overall score at both model scales. This indicates that GPDPO does more than improve auxiliary objectives such as Format Acc. and Length Rew.; by recovering pairwise supervision from filtered responses, it extracts additional learning signal that directly benefits task correctness.

![](images/351e1f3f5bc67f173142d540523c6318ec6cb20bcf0a658e6b173f8a36973e9a.jpg)

![](images/1b98a5a229d66556cddb69cc88afc23038d6ea5d21bf774dff8728ae511f8508.jpg)

![](images/5504bb87a76f3353e0325aa23f686c8e80677cef672820837be1ce1f7b1e9e0e.jpg)

![](images/9a65353dbd75d7835ba263349d8839c1086f6f0585f0543e7335c23de9200bbf.jpg)  
Figure 4: Training dynamics of Qwen3-1.7B under the three-reward (3R) setting.

![](images/6e6bc725bff7a87f4090769e6ae3bd1cb1b0f88efc13caaedddf022e168b979b.jpg)

![](images/05723bc6dfdac9c663224a6fc6b071713d93e235d6a7160e627044072dd93391.jpg)  
Figure 5: Conflict dynamics on Qwen3-1.7B under 3R setting. Left: pairwise preference conflict rate over all non-tied within-group response pairs, including retained–retained pairs. Right: $\mathrm { G D ^ { 2 } P O }$ conflict rate over valid rollouts.

## 4.3 TRAINING AND CONFLICT DYNAMICS

Beyond improving downstream performance, GPDPO exhibits the training dynamics: it reduces reward conflicts and achieves stronger reward optimization without accelerating entropy collapse.

Conflict metrics. We measure multi-reward conflict at two levels. Following $\mathrm { G D ^ { 2 } P O }$ , the grouprelative conflict rate is the fraction of valid rollouts whose reward-wise advantages contain both positive and negative entries, with zeros treated as neutral. We further define the pairwise preference conflict rate as the fraction of Pareto-incomparable pairs among all non-tied, unordered response pairs within a group. These metrics capture conflict at the aggregate rollout level and the atomic pairwise level, respectively.

Training dynamics. As shown in Figure 4, GPDPO achieves stronger late-stage reward optimization while retaining higher policy entropy. Over steps 151–200, its mean correctness reward reaches 1.6260, compared with 1.4242 for GRPO and 1.4035 for GD<sup>2</sup>PO. Its format and length rewards are also higher than $\mathrm { G D ^ { 2 } P O }$ (0.9566/0.9914 vs. 0.8902/0.9328), while maintaining higher entropy (0.0795 vs. 0.0746 for GRPO and 0.0719 for $\mathrm { G D ^ { 2 } P O ) }$ . Thus, the higher rewards do not coincide with faster policy collapse in this representative run.

Conflict dynamics. As shown in Figure 5, our method exhibits faster declines in conflict rates at both levels and achieves lower final conflict rates than GRPO, GDPO, and $\mathrm { G D ^ { 2 } P O }$ . This is consistent with GPDPO’s design: reward-consistent Pareto relations recover auxiliary supervision from filtered responses, and each group-relative deviation can be decomposed into aggregated pairwise reward differences. The simultaneous reduction in both conflict measures suggests that recovering rewardconsistent pairwise supervision is an effective strategy for alleviating multi-reward conflict.

## 4.4 ABLATION AND SENSITIVITY

Component analysis of GPDPO. To understand the contribution of each design choice in GPDPO, we conduct ablations on four variants: (1) GPDPO (Strict Pareto Dominance), which requires strict improvement on every reward dimension instead of allowing ties; (2) GPDPO (Scalarized Reward Ordering), which replaces Pareto-based comparisons with ordering based on an aggregated scalar reward; (3) GPDPO (Fixed Reference Model), which keeps the initial policy as the reference model throughout training; and (4) GPDPO (Pareto-Guided Advantage Adjustment), which incorporates recovered Pareto relations by modifying advantages instead of directly optimizing preference margins. All variants are evaluated on Qwen3-1.7B under 2R setting (Table 3).

Table 3: Ablations on Qwen3-1.7B with 2R setting. Best Correct Acc. and Length Rew. are bold.
<table><tr><td>Method</td><td>Level 1 Acc.</td><td>Level 2 Acc.</td><td>Level 3  $\operatorname { A c c } .$ </td><td>Correct Acc.</td><td>Length Rew.</td></tr><tr><td> $\mathrm { G D ^ { 2 } P O }$ </td><td>69.17</td><td>61.19</td><td>42.75</td><td>62.48</td><td>0.9793</td></tr><tr><td>GPDPO (Strict Pareto Dominance)</td><td>68.67</td><td>62.69</td><td>48.09</td><td>63.48</td><td>0.9650</td></tr><tr><td>GPDPO (Scalarized Reward Ordering)</td><td>68.42</td><td>49.25</td><td>48.09</td><td>61.81</td><td>0.9675</td></tr><tr><td>GPDPO (Fixed Reference Model)</td><td>71.18</td><td>58.21</td><td>39.69</td><td>62.81</td><td>0.9767</td></tr><tr><td>GPDPO (Pareto-Guided Advantage Adjustment)</td><td>70.68</td><td>61.19</td><td>44.27</td><td>63.82</td><td>0.9762</td></tr><tr><td>GPDPO (Full)</td><td>70.93</td><td>61.19</td><td>48.85</td><td>64.99</td><td>0.9812</td></tr></table>

Strict Pareto dominance reduces supervision coverage. GPDPO with strict Pareto dominance achieves 63.48% Correct Acc., 1.51 percentage points below full GPDPO. Strict dominance requires improvement on every reward dimension and therefore discards comparisons with ties. Standard Pareto dominance retains such pairs when one response is no worse on all rewards and strictly better on at least one, recovering additional supervision without violating reward consistency. These results suggest that allowing ties under standard Pareto dominance recovers broader yet still reward consistent supervision, leading to more effective policy optimization.

Scalarized ordering loses reward consistency. Replacing Pareto dominance with scalarized reward ordering reduces Correct Acc. from 64.99% to 61.81%, falling below $\mathrm { { G D ^ { 2 } P O ^ { \prime } s } }$ 62.48%. This degradation indicates that the benefit of recovered supervision comes from preserving consistent preference directions across reward dimensions rather than from additional pairwise signals alone.

Reference refreshing improves optimization. Replacing the refreshed reference policy with a fixed reference reduces Correct Acc. to 62.81%, compared with 64.99% for full GPDPO. The fixed-reference variant keeps the initial policy as the reference throughout training, with $\beta = 3 2$ and $\lambda = 0 . 0 0 0 3 1 2 5$ selected to match the zero-margin gradient scale of full GPDPO. This result suggests that refreshing the reference policy along with the evolving sampling policy provides a more effective optimization target under the tested configuration.

Recovered relations remain useful under alternative updates. GPDPO (Pareto-Guided Advantage Adjustment) improves Correct Acc. from GD<sup>2</sup>PO’s 62.48% to 63.82%, confirming that the recovered Pareto relations themselves contain useful optimization signals. Full GPDPO further improves performance to 64.99%, suggesting that directly optimizing pairwise preference margins provides a more effective way to exploit the recovered relational supervision.

Robustness to Auxiliary Supervision Strength. Figure 6 evaluates the sensitivity of GPDPO to the parameter λ. Across all tested values $( \lambda = 0 . 0 5 .$ 0.1, 0.15, and 0.2), GPDPO consistently outperforms $\mathrm { G D ^ { 2 } P O } ,$ , achieving Correct Acc. scores of 63.82%, 64.99%, 64.66%, and 63.15%, respectively, compared with 62.48% for GD<sup>2</sup>PO. These results indicate that GPDPO maintains consistent improvements across a range of auxiliary supervision strengths rather than relying on a single carefully tuned coefficient.

![](images/e2742b84f7bb131670135f38e9d2167b89c3d22d7dadb26c1ffa57b77c13eaa8.jpg)  
Figure 6: Sensitivity of λ on Qwen3-1.7B under 2R setting.

## 5 CONCLUSION

We presented GPDPO, a multi-reward policy optimization method that recovers reward-consistent pairwise supervision from responses discarded by conflict filtering. Our key observation is that conflicts in group-relative update signals do not necessarily eliminate reliable pairwise preferences between responses. GPDPO exploits Pareto-dominance relations within existing rollout groups to recover this supervision without additional sampling or reward evaluation. Experiments show that GPDPO consistently improves multi-reward optimization performance while reducing both grouprelative and pairwise reward conflicts. These results demonstrate the value of fine-grained relational supervision for more effective multi-objective LLM alignment.

## AI USE STATEMENT

We used generative AI tools solely to assist with manuscript writing and improve the clarity, grammar, and readability of the text. We did not use generative AI tools for any other aspect of the research. The authors reviewed and revised all AI-assisted text to ensure that it accurately reflected the methods and reported results. The authors take full responsibility for the final manuscript, in cluding its accuracy, originality, and the integrity of its claims and conclusions.

## REPRODUCIBILITY STATEMENT

To facilitate reproducibility, we describe the GPDPO formulation in Section 3 and provide detailed pseudocode and implementation notes in Appendix A.2. Section 4.1 introduces the datasets, base models, baselines, reward definitions, and evaluation metrics. Appendix A.1 documents the training and evaluation configurations, including hyperparameters, hardware, decoding settings, and checkpoint reporting conventions. Appendix C specifies the experimental setups, sampling procedures, and statistical definitions used in the conflict-filtering analysis in Figure 2. Our implementation builds on verl and vLLM, and we will release the source code soon.

## REFERENCES

K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: Nsga-ii. IEEE Transactions on Evolutionary Computation, 6(2):182–197, 2002. doi: 10.1109/42 35.996017.

Michael TM Emmerich and André H Deutz. A tutorial on multiobjective optimization: fundamentals and evolutionary methods. Natural computing, 17(3):585–609, 2018.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023.

Chengao Li, Hanyu Zhang, Yunkun Xu, Hongyan Xue, Xiang Ao, and Qing He. Gradient-adaptive policy optimization: Towards multi-objective alignment of large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11214–11232, 2025a.

Junyi Li, Xiaowei Qian, Yingyi Zhang, Wenlin Zhang, Guojing Li, Sheng Zhang, Xiao Han, Yichao Wang, and Xiangyu Zhao. Towards pareto-optimal tool-integrated agents with pareto ranking policy optimization. arXiv preprint arXiv:2606.16111, 2026.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. Api-bank: A comprehensive benchmark for tool-augmented llms. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 3102–3116, 2023.

Moxin Li, Yuantao Zhang, Wenjie Wang, Wentao Shi, Zhuo Liu, Fuli Feng, and Tat-Seng Chua. Self-improvement towards pareto optimality: Mitigating preference conflicts in multi-objective alignment. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pp. 11010– 11031, 2025b.

Qiqiang Lin, Muning Wen, Qiuying Peng, Guanyu Nie, Junwei Liao, Jun Wang, Xiaoyun Mo, Jiamu Zhou, Cheng Cheng, Yin Zhao, et al. Hammer: Robust function-calling for on-device language models via function masking. arXiv preprint arXiv:2410.04587, 2024.

Haotian Liu, Yihao Liu, Jingwei Ni, Siyuan Huang, Xinpeng Liu, Pengyu Cheng, Jiajun Song, Ruijin Ding, Junfeng Li, Zhechao Yu, et al. GD<sup>2</sup>PO: Mitigating multi-reward conflicts via groupdynamic reward-decoupled policy optimization. arXiv preprint arXiv:2606.16771, 2026a.

Shih-Yang Liu, Xin Dong, Ximing Lu, Shizhe Diao, Peter Belcak, Mingjie Liu, Min-Hung Chen, Hongxu Yin, Yu-Chiang Frank Wang, Kwang-Ting Cheng, et al. Gdpo: Group rewarddecoupled normalization policy optimization for multi-reward rl optimization. arXiv preprint arXiv:2601.05242, 2026b.

Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong WANG, et al. Toolace: Winning the points of llm function calling. In International conference on learning representations, volume 2025, pp. 41359–41381, 2025a.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025b.

Yining Lu, Zilong Wang, Shiyang Li, Xin Liu, Changlong Yu, Qingyu Yin, Zhan Shi, Zixuan Zhang, and Meng Jiang. Learning to optimize multi-objective alignment through dynamic reward weighting. Transactions ofthe Associationfor Computational Linguistics, 14:1051–1073, 2026.

Kirill Pavlenko, Alexander Golubev, Simon Karasik, and Boris Yangel. Blockwise advantage estimation for multi-objective rl with verifiable rewards. arXiv preprint arXiv:2602.10231, 2026.

Cheng Qian, Emre Can Acikgoz, Qi He, Hongru Wang, Xiusi Chen, Dilek Hakkani-Tur, Gokhan Tur, and Heng Ji. Toolrl: Reward is all tool learning needs. In Advances in Neural Information Processing Systems, volume 38, pp. 105523–105553, 2026.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Hongyan Xie, Yikun Ban, Ruiyu Fang, Di Wu, Zixuan Huang, Deqing Wang, Jianxin Li, and Shuangyong Song. Direct preference optimization with pareto dominance constraint for online multi-objective alignment. Neurocomputing, pp. 134604, 2026.

Zhihao Xu, Yongqi Tong, Xin Zhang, Jun Zhou, and Xiting Wang. Understanding conflicts in multiobjective alignment through reward consistency. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pp. 5450–5472, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. In Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244, 2026.

Yu Yuan, Yaoyou Fan, Lili Zhao, Guangting Zheng, Kai Zhang, Lu Pan, Ke Zeng, and Qi Liu. Uncovering and mitigating aggregation-induced reward hacking in multi-reward reinforcement learning. arXiv preprint arXiv:2609.00213, 2026.

Jianguo Zhang, Tian Lan, Ming Zhu, Zuxin Liu, Thai Hoang, Shirley Kokane, Weiran Yao, Juntao Tan, Zhiwei Liu, Yihao Feng, et al. xlam: A family of large action models to empower ai agent systems. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 11583–11597, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Zhanhui Zhou, Jie Liu, Jing Shao, Xiangyu Yue, Chao Yang, Wanli Ouyang, and Yu Qiao. Beyond one-preference-fits-all alignment: Multi-objective direct preference optimization. In Findings of the Associationfor Computational Linguistics: ACL 2024, pp. 10586–10613, 2024.

## A IMPLEMENTATION DETAILS

## A.1 DETAILED SETTINGS

We implement GPDPO with verl and use vLLM for rollout generation. Experiments use Qwen3- 1.7B and Qwen3-4B on RLLA-4K, whose training file contains 3,920 examples; overlength prompts are filtered during loading. Tables 4 and 5 summarize the settings. Unless stated otherwise, all reward dimensions have unit aggregation weights, and GPDPO uses uniform pair weights with λ = 0.1 and $\beta = 0 . 1$

Table 4: Training settings for GPDPO. Batch sizes are measured in prompts.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Training hardware</td><td>One node, 4× NVIDIA RTX 5880 Ada (48 GB each)</td></tr><tr><td>Base models</td><td>Qwen3-1.7B, Qwen3-4B</td></tr><tr><td>Training data</td><td>RLLA-4K, 3,920 records before prompt filtering</td></tr><tr><td>Max prompt / response length</td><td>2,048 / 1,024 tokens</td></tr><tr><td>Train batch / mini-batch size</td><td>512 / 128 prompts</td></tr><tr><td>Rollout group size G</td><td>4 (2,048 responses per rollout step)</td></tr><tr><td>Training steps / PPO epochs</td><td>200 rollout steps / 1 epoch per step</td></tr><tr><td>Optimizer / learning rate</td><td>AdamW  $/ 1 0 ^ { - \hat { 6 } }$  , constant, no warmup</td></tr><tr><td>Temperature / top-p / top-k</td><td>1.0 / 1.0 / disabled</td></tr><tr><td>PPO clip range / dual-clip factor</td><td> $\left( \epsilon _ { \mathrm { l o w } } , \epsilon _ { \mathrm { h i g h } } \right) = \left( 0 . 2 , 0 . 2 \right) / 3 . 0$ </td></tr><tr><td>Policy-loss aggregation</td><td>Token mean</td></tr><tr><td>Dynamic batching</td><td>Enabled; response groups remain intact</td></tr><tr><td>vLLM GPU memory utilization</td><td>0.60</td></tr><tr><td>KL penalty / entropy bonus</td><td>Neither used</td></tr><tr><td>Rewards</td><td>2R: correctness + length; 3R: additionally format</td></tr><tr><td>Pairwise loss parameters</td><td>λ = 0.1, β = 0.1; uniform pair weights</td></tr><tr><td>Reference policy</td><td>Sampling policy; refreshed every rollout step</td></tr><tr><td>Checkpoint interval / main results</td><td>Every 50 steps / selected step-200 checkpoint</td></tr></table>

Each rollout step generates $5 1 2 \times 4$ responses and performs four mini-batch updates. Correctness assigns partial credit for matching tool names, parameter names, and values, scaled to [−3, 3]. Length is min(round $( n _ { \mathrm { t h i n k } } / 5 1 2 , 2 ) , 1 )$ , where $n _ { \mathrm { t h i n k } }$ counts whitespace-separated words in the think block; a missing complete block receives zero. Format is binary and checks balanced, nonrepeated, correctly ordered blocks, with think always required and the other required blocks determined by GT.

Table 5: Evaluation settings for the selected step-200 checkpoints.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Evaluation hardware</td><td>One NVIDIA RTX A6000 (48 GB)</td></tr><tr><td>Dataset</td><td>Processed API-Bank, 597 questions</td></tr><tr><td>Level 1 / 2 / 3 questions</td><td>399 / 67 / 131</td></tr><tr><td>Decoding</td><td>Greedy, one response per question</td></tr><tr><td>Temperature / top-p</td><td>0.0 / 1.0</td></tr><tr><td>Max new tokens / total sequence length</td><td>4,096 / 4,096</td></tr><tr><td>Generation batch / tensor parallel size</td><td>1/1</td></tr><tr><td>Evaluation seed</td><td>42</td></tr><tr><td>Metrics</td><td>Level accuracy, Correct Acc., Length Rew.; Format Acc. for 3R</td></tr></table>

Correct Acc. is accuracy over all 597 questions. Overall is CORRECT ACC./100 + LENGTH REW. for 2R, with FORMAT ACC./100 added for 3R. Each main-table result is a single evaluation of a selected run. The 4,096-token total sequence limit includes the prompt and can reduce the available generation length.

## A.2 DETAILED ALGORITHM

Algorithm 1 follows the three components in Section 3.3. Dominance relations and preference pairs are built once per rollout batch. Scores and joint updates are evaluated per mini-batch using the sampling policy fixed at the start of the rollout step.

Algorithm 1 Group Pareto-Dominance Policy Optimization (GPDPO)   
Input: Initial policy π<sub>θ</sub>, prompt distribution ${ \mathcal { P } } ,$ reward functions $\{ r _ { i } \} _ { i = 1 } ^ { M }$   
group size $G ,$ reward weights $\{ w _ { i } \} _ { i = 1 } ^ { M }$ , loss weight $\lambda ,$ margin scale $\beta$   
Output: Trained policy $\pi \theta$   
1: for each training iteration do   
2: $\theta _ { \mathrm { o l d } }  \theta , \overline { { \mathrm { ~  ~ { ~ \cal ~ B ~ } ~ } } }  \mathrm { { S } }$ ampleBatch $( \mathcal { P } )$   
Stage 1: Pareto-dominance identification   
3: for each prompt $x \in B$ do   
4: $\mathcal { G } ( \boldsymbol { x } ) \mathrm { \tilde { \Omega } } {  } \mathrm { S a m p l e } ( \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot \mid \boldsymbol { x } ) , G )$   
5: $\ell _ { n , t } ^ { \mathrm { o l d } } \gets \log \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { n , t } \mid x , y _ { n , < t } )$ $\left( \forall n , t \right)$   
6: $\mathbf { r } _ { n } \gets ( r _ { 1 } ( x , y _ { n } ) , \dots , r _ { M } ( x , y _ { n } ) )$ (∀n)   
7: $\begin{array} { r l r l r } { \mu _ { i }  G ^ { - 1 } \sum _ { n = 1 } ^ { G } r _ { n } ^ { i } , } & { { } } & { A _ { n } ^ { i }  ( r _ { n } ^ { i } - \mu _ { i } ) / s _ { i } } & { { } } & { ( \forall n , i ) } \end{array}$   
8: $g ( y _ { n } ) \gets \mathbf { 1 } \{ \mathrm { s i g n } ( A _ { n } ^ { i } ) \mathrm { s i g n } ( A _ { n } ^ { j } ) \geq 0 , \forall i , j \} \qquad ( \forall n )$   
9: ${ \bar { \mathcal { F } } } ( x ) \gets \{ y _ { n } \in { \dot { \mathcal { G } } } ( x ) : { \bar { g ( } } y _ { n } ) = 0 \} , \qquad { \bar { \mathcal { R } } } ( x ) \gets { \mathcal { G } } ( x ) \setminus { \mathcal { F } } ( x )$   
10: $\begin{array} { c c } { { \hat { \kappa } ( x ) \gets G ^ { - 1 } \sum _ { n = 1 } ^ { G } g ( y _ { n } ) , \qquad \widetilde { A } _ { n } \gets \hat { \kappa } ( x ) g ( y _ { n } ) \sum _ { i = 1 } ^ { M } w _ { i } A _ { n } ^ { i } } } \end{array}$   
// Apply the $\mathrm { G D ^ { 2 } P O }$ conflict filter and retain its standard advantages.   
11: $\mathcal { D } ( x ) \stackrel {  } {  } \{ ( y _ { w } , y _ { l } ) \in \mathcal { G } ( x ) ^ { 2 } : \mathbf { r } _ { w } \succ _ { \mathrm { P } } \mathbf { r } _ { l } \}$   
12: end for   
Stage 2: Preference-loss construction   
13: $\mathsf { \tilde { \mathcal { E } } } ( x ) \gets \{ ( y _ { w } , y _ { l } ) \in { \mathcal { D } } ( x ) : y _ { w } \in { \mathcal { F } } ( x ) \vee y _ { l } \in { \mathcal { F } } ( x ) \}$ $( \forall x \in B )$   
// Select dominance edges involving at least one filtered response.   
14: $\begin{array} { r } { A \gets \mathrm { M a s k e d W h i t e n } ( \widetilde { A } , g ) , \qquad N _ { \mathcal { E } } \gets \sum _ { \boldsymbol { x } \in \mathcal { B } } | \mathcal { E } ( \boldsymbol { x } ) | } \end{array}$   
15: $\{ \mathcal { M } _ { k } \} \gets \mathrm { G r o u p P a r t i t i o n } ( \mathcal { B } )$ // Keep response groups intact.   
16: for each mini-batch $\mathcal { M } _ { k }$ do   
17: $\ell _ { n , t } ^ { \theta } \gets \log \pi _ { \theta } ( y _ { n , t } \mid x , y _ { n , < t } )$ $( \forall n , t$ in $\boldsymbol { \mathcal { M } } _ { k } )$   
18: $u _ { \theta } ( y _ { n } \mid x )  | y _ { n } | ^ { - 1 } \sum _ { t = 1 } ^ { | y _ { n } | } ( \ell _ { n , t } ^ { \theta } - \ell _ { n , t } ^ { \mathrm { o l d } } )$   
19: $\widetilde { u } _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \gets ( \mathrm { i } - g ( \boldsymbol { y } ) ) \widetilde { u } _ { \boldsymbol { \theta } } \widehat { ( \boldsymbol { y } \mid \boldsymbol { x } ) } + g ( \boldsymbol { y } ) \operatorname { s g } [ u _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) ]$   
// Retained endpoints serve as fixed anchors in the pairwise loss.   
20: $\ell _ { \mathrm { p a i r } } ( y _ { w } , y _ { l } \mid x ) \stackrel { . } { \longleftarrow } - \log \sigma ( \beta [ \widetilde u _ { \theta } ( y _ { w } \mid x ) - \widetilde u _ { \theta } ( y _ { l } \mid \stackrel { . } { x } ) ] )$   
Stage 3: Joint policy optimization   
21: $\mathcal { L } _ { \mathrm { p a i r } } ^ { ( k ) }  0$ // An empty pair set contributes zero.   
22: $\mathbf { i f } ^ { \prime } \bar { N } \bar { \varepsilon } > 0$ then   
23: $\begin{array} { r } { \mathcal { L } _ { \mathrm { p a i r } } ^ { ( k ) }  N _ { \mathcal { E } } ^ { - 1 } \sum _ { x \in \mathcal { M } _ { k } } \sum _ { ( y _ { w } , y _ { l } ) \in \mathcal { E } ( x ) } \ell _ { \mathrm { p a i r } } ( y _ { w } , y _ { l } \mid x ) } \end{array}$   
24: end if   
25: $\mathcal { L } _ { \mathrm { s t a n d a r d } } ^ { ( k ) }  \mathrm { C l i p p e d P o l i c y L o s s } ( \ell ^ { \theta } , \ell ^ { \mathrm { o l d } } , A ; \mathcal { M } _ { k } )$   
26: ${ \mathcal { L } } _ { \mathrm { G P D P O } } ^ { ( k ) } \gets { \mathcal { L } } _ { \mathrm { s t a n d a r d } } ^ { ( k ) } + \lambda { \mathcal { L } } _ { \mathrm { p a i r } } ^ { ( k ) }$   
27: $\theta \gets \mathrm { O p t i m i z e r S t e p } ( \theta , \nabla _ { \theta } \mathcal { L } _ { \mathrm { G P D P O } } ^ { ( k ) } )$   
28: end for   
29: end for   
30: return π<sub>θ</sub>

Pair construction and gradient flow. The implementation compares the reward-wise normalized advantages before filtering, with tolerance $1 0 ^ { - 8 }$ ; positive per-dimension normalization preserves Pareto ordering in exact arithmetic. Sequence scores average log-probability ratios over valid response tokens. Retained endpoints are detached only in the pairwise term; both endpoints receive gradients when both are filtered. Cached sampling-policy probabilities provide the reference without an additional fixed model.

Loss normalization and updates. We implement each pair loss as sof $\mathrm { \ t p l u s } ( - \beta m )$ , where $m =$ $\widetilde { u } _ { \theta } ( y _ { w } \mid x ) - \widetilde { u } _ { \theta } ( y _ { l } \mid x )$ . Every mini-batch and micro-batch uses the full rollout batch’s pair count as its denominator. GroupPartition preserves group order; MaskedWhiten whitens retained-token advantages and leaves filtered advantages zero. ClippedPolicyLoss is $\mathrm { G D ^ { 2 } P O ^ { \prime } s }$ standard objective; OptimizerStep uses AdamW. Pairs reuse the rollout responses and rewards.

Table 6: Detailed results for Figure 2. Entries report count (percentage). The denominator is 65,536 paired label comparisons per setting for (a,b), and 35,937 / 58,689 filtered responses under 2R / 3R for (c,d). The four categories in (d) partition the filtered set.
<table><tr><td>Panel</td><td>Measurement</td><td>2R: count (%)</td><td>3R: count (%)</td></tr><tr><td rowspan="2">(a)</td><td>Expansion, 4 → 64: changed</td><td>10,608 (16.19)</td><td>17,756 (27.09)</td></tr><tr><td>Expansion, 4 → 64: unchanged</td><td>54,928 (83.81)</td><td>47,780 (72.91)</td></tr><tr><td rowspan="3">(b)</td><td>Peer replacement, G = 4: disagreement</td><td>5,999 (9.15)</td><td>10,434 (15.92)</td></tr><tr><td>Peer replacement, G = 16: disagreement</td><td>6,312 (9.63)</td><td>8,119 (12.39)</td></tr><tr><td>Peer replacement, G = 64: disagreement</td><td>3,561 (5.43)</td><td>3,994 (6.09)</td></tr><tr><td>(c)</td><td>GT tool-call structure matches</td><td>8,345 (23.22)</td><td>16,039 (27.33)</td></tr><tr><td rowspan="5">(d)</td><td>Consistently better</td><td>4,900 (13.63)</td><td>8,905 (15.17)</td></tr><tr><td>Both</td><td>11,122 (30.95)</td><td>16,163 (27.54)</td></tr><tr><td>Consistently worse</td><td>15,752 (43.83)</td><td>25,797 (43.96)</td></tr><tr><td>No consistent relation</td><td>4,163 (11.58)</td><td>7,824 (13.33)</td></tr><tr><td>At least one consistent relation</td><td>31,774 (88.42)</td><td>50,865 (86.67)</td></tr></table>

## B LIMITATIONS

Our evaluation is limited to two- and three-reward settings, as suitable benchmarks with a larger number of reward dimensions are currently unavailable. Future work will explore reward conflicts in higher-dimensional multi-reward settings and investigate scalable strategies for exploiting informative signals from conflicting responses.

## C ADDITIONAL DETAILS FOR THE CONFLICT-FILTERING ANALYSIS

We supplement Figure 2 with settings and results for Qwen3-1.7B trained with $\mathrm { G D ^ { 2 } P O }$ under correctness and length rewards (2R), and with the additional format reward (3R).

## C.1 SENSITIVITY TO THE REFERENCE GROUP

Experimental setup. We use the step-50 checkpoint from each of the separate 2R and 3R training runs. Both settings use a common set of 128 RLLA-4K training prompts. Each model generates four target responses and a pool of 128 additional responses per prompt, with temperature 1.0 and at most 1,024 response tokens.

Experimental procedure. Target responses and rewards remain fixed. For panel (a), we add 60 pool responses to the original four-response group and compare the four targets’ filtering labels at $\bar { G } = 4$ and $G = 6 4 .$ For panel (b), each target is combined with two sets of $G - 1$ peers from the pool, for $G \in \{ 4 , 1 6 , 6 4 \}$ . Sampling is without replacement within each set, with overlap allowed between sets. We repeat each comparison 128 times.

Results. Each rate uses $1 2 8 \times 4 \times 1 2 8 = 6 5 { , } 5 3 6$ paired label comparisons per reward setting. Expansion changes 10,608/65,536 (16.19%) labels under 2R and 17,756/65,536 (27.09%) under 3R. At G = 64, peer replacement still produces 3,561/65,536 (5.43%) and 3,994/65,536 (6.09%) inconsistent labels, respectively (Table 6). Thus, a fixed response’s filtering decision remains sensitive to group composition.

## C.2 INFORMATION RETAINED IN FILTERED RESPONSES

Experimental setup. For panels (c,d), we analyze rollout logs from steps 1–200 of the two separate GD<sup>2</sup>PO training runs, each with G = 4. Conflict filtering identifies 35,937 responses under 2R and 58,689 under 3R.

Experimental procedure. Panel (c) measures GT matches of ordered tool-call lists, including call order, tool names, and parameter names and values. Panel (d) compares each filtered response with its three original peers using Pareto dominance, assigning it to one of the four mutually exclusive categories in Table 6. “Both” means better than some peers and worse than others. All percentages use the full filtered set as the denominator.

Table 7: Comparison-edge statistics under 2R. All edge counts involve at least one filtered response. Percentages for added edges use the Pareto-edge count as the denominator.
<table><tr><td>Measurement</td><td> $\mathrm { G D ^ { 2 } P O }$ </td><td>GPDPO (λ = 0.1)</td></tr><tr><td>All-dimension strict-improvement edges</td><td>6,374</td><td>6,338</td></tr><tr><td>Pareto-dominance edges</td><td>52,421</td><td>52,631</td></tr><tr><td>Edges added by allowing ties</td><td>46,047 (87.84%)</td><td>46,293 (87.96%)</td></tr><tr><td>Tied correctness, improved length</td><td>29,229</td><td>29,578</td></tr><tr><td>Tied length, improved correctness</td><td>16,818</td><td>16,715</td></tr></table>

Results. GT matches account for 8,345/35,937 (23.22%) and 16,039/58,689 (27.33%) of the 2R and 3R filtered sets, respectively. At least one consistent relation exists for 31,774/35,937 (88.42%) and 50,865/58,689 (86.67%), leaving 4,163 and 7,824 responses without such relations. These results show that filtered responses can retain task-relevant content and consistent comparisons, supporting the recovery of pairwise supervision.

## D EXPERIMENTS ON PARETO-DOMINANCE EDGES

We examine how allowing ties in individual reward dimensions affects the comparison edges avail able for recovering pairwise supervision from filtered responses.

Experimental setup. We analyze steps 1–200 of historical $\mathrm { G D ^ { 2 } P O }$ and GPDPO $( \lambda = 0 . 1 )$ training runs with Qwen3-1.7B on RLLA-4K, using correctness and length rewards (2R) and response groups of size $G = 4$

Experimental procedure. We label responses using the training-time conflict filter and compare two criteria: all-dimension strict improvement, which requires higher rewards in every dimension; and Pareto dominance, which requires no lower reward in any dimension and a higher reward in at least one. Both criteria use the same candidate pairs: within-group pairs involving at least one filtered response. Retained–retained pairs are excluded, and each edge is counted once in the betterto-worse direction. We count edges under each criterion and identify the tied dimension in edges admitted by Pareto dominance but excluded by strict improvement.

Results. Table 7 shows that Pareto dominance retains 52,421 edges under $\mathrm { G D ^ { 2 } P O } ,$ , compared with 6,374 under all-dimension strict improvement. The additional 46,047/52,421 (87.84%) edges include 29,229 ties in correctness and 16,818 ties in length, with improvement in the other reward. The GPDPO run shows a similar pattern, with 46,293/52,631 (87.96%) Pareto edges involving such ties.

Pareto dominance allows ties in some reward dimensions while using improvements in others to provide supervision. It retains more comparison edges than all-dimension strict improvement, thereby providing richer pairwise learning signals.