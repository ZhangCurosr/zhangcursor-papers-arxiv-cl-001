# Adaptive Entropy-Sparing for Efficient Reasoning

Cong Jiang <sup>1,2</sup> Xiaofeng Zhang <sup>1</sup> Tom Ko <sup>2</sup> Zheng Zhang <sup>1,2∗</sup>

<sup>1</sup>Harbin Institute of Technology, Shenzhen

<sup>2</sup>Shenzhen Loop Area Institute

jerryjiang.jc@gmail.com, darrenzz219@gmail.com

## Abstract

Large Reasoning Models (LRMs) produce explicit chains of thought to improve transparency, but they frequently overthink, generating long yet inefficient reasoning chains that inflate computational costs and risk introducing errors. While numerous reinforcement learning methods struggle to penalize verbosity to achieve conciseness, they typically apply the penalty uniformly across lengthy sequences, which leads to credit confounding, that is, the failure to distinguish essential reasoning steps from superfluous ones. In this paper, we study a measurable proxy for this confounding: high-entropy tokens that often coincide with surface transition points where reasoning re-enters verification. Based on this insight, we propose Adaptive Entropy-Sparing (AES), which selectively spares high-entropy tokens that occur before an efficient in-group reference, while penalizing overlength highentropy suffix tokens associated with superfluous branching. Our AES thereby navigates the suppression-exploration trade-off, reducing overthinking while preserving average reasoning accuracy. We also present an entropy-length model that motivates why local uncertainty can serve as a proxy for future reasoning cost. Across eight reasoning benchmarks, AES reduces reasoning length by up to 52.9% while increasing mean accuracy by up to 3.8% over the base models, consistently outperforming existing efficiency-oriented methods. Our code is available at https://github.com/MobiusDai/AES.

## 1 Introduction

Large reasoning models (LRMs) have recently demonstrated remarkable problem-solving capabilities, particularly in mathematical reasoning [Shao et al., 2024], code generation [Jiang et al., 2026b], and agentic tasks [Wu et al., 2025]. The reasoning ability of large language models (LLMs) can be traced back to chain-of-thought (CoT) prompting [Wei et al., 2022], which enables models to solve problems step-by-step through simple instructions without additional fine-tuning. While such promptbased reasoning represents an early form of deliberative thinking, recent advances have given rise to slow-thinking models [Min et al., 2024], such as OpenAI-o1 [Jaech et al., 2024], DeepSeek-R1 [Guo et al., 2025], and QWQ [Qwen Team, 2025], which analyze problems through extended reasoning trajectories before producing final answers. Drawing an analogy to dual-process theory in cognitive psychology, these models are often characterized as System-2 reasoners, in contrast to conventional LLMs that produce direct responses akin to System-1 processing [Li et al., 2025b]. Although LRMs substantially outperform their predecessors, this improvement comes at the cost of significantly increased token consumption. In particular, these models frequently exhibit overthinking [Sui et al., 2025], characterized by redundant verification steps and unnecessary backtracking that inflate computational costs without improving, and sometimes even degrading answer quality.

To address the overthinking problem, various methods have been proposed to reduce output length while maintaining reasoning quality [Feng et al., 2025]. Existing techniques can be broadly categorized into three research lines: efficiency-oriented post-training [Aggarwal and Welleck, 2025, Hou et al., 2025, Cheng et al., 2025], latent reasoning [Hao et al., 2024, Ruan et al., 2025, Jiang et al., 2026a], and model collaboration [Liao et al., 2025, Su et al., 2025]. Among these, efficiency-oriented post-training is the most widely adopted, as it preserves the original model architecture and does not require auxiliary models. These methods typically augment outcome-based rewards with cost-aware terms [Luo et al., 2025, Liu et al., 2025], incentivizing the model to produce more concise responses.

Despite their efficacy, these approaches often suffer from performance degradation [Arora and Zanette, 2025] compared to the original models, particularly on challenging tasks. Cost-aware penalties uniformly penalize all tokens in a trajectory, conflating steps that are essential for correct reasoning with those that merely introduce superfluous elaboration. We refer to this limitation as credit confounding. Consequently, the model is discouraged from engaging in deliberate, multistep reasoning even when such exploration is essential for correctness, leading to a suboptimal suppression-exploration trade-off.

In this work, we begin by analyzing model-generated reasoning trajectories through token-level conditional entropy. We observe that transition words and phrases $( e . g .$ ., ‘but,’ ‘wait,’ ‘let us verify’) frequently align with high-entropy spikes, indicating elevated local uncertainty. These high-entropy tokens serve as transition triggers that may either facilitate essential reasoning or induce superfluous exploration. When the model hesitates among multiple candidate directions, these tokens often initiate additional verification steps rather than concluding the reasoning process [Muennighoff et al., 2025]. To further investigate this phenomenon, we conduct controlled experiments by manually truncating reasoning trajectories at different transition points. These results reveal that later transition steps contribute minimally to final answer quality, and a substantial portion of high-entropy transitions induce redundant branching rather than productive exploration. These observations yield a key insight: overthinking is often associated with the model’s insufficient confidence to terminate reasoning after reaching a plausible conclusion, where high-entropy tokens guide the model toward unnecessary subsequent verification steps.

Building on this insight, we propose Adaptive Entropy-Sparing (AES), a reinforcement learning based post-training method that applies selective advantage reduction to overlength high-entropy suffix tokens in correct trajectories. To operationalize this idea, we sample multiple responses for each prompt and treat the shortest correct response as a reference for efficient reasoning, which allows us to estimate the degree of overthinking in longer correct responses and apply penalties selectively. Extensive experiments demonstrate that AES substantially reduces response length while improving accuracy across both mathematical reasoning and out-of-domain benchmarks.

## 2 Empirical Analysis of Overthinking

To investigate patterns associated with the overthinking phenomenon in LRMs, we analyze reasoning trajectories from an information-theoretic perspective. We examine whether overthinking is associated with identifiable patterns in the model’s token-level uncertainty during generation. To do so, we first inspect the entropy distribution across tokens within reasoning trajectories and find that specific transitional markers frequently exhibit elevated entropy. We then conduct truncation experiments with generated reasoning trajectories to determine whether extended reasoning steps contribute meaningfully to final answer quality.

## 2.1 Entropy Distribution Across Reasoning Trajectories

Settings. Let $\pi _ { \theta }$ denote a slow-thinking language model. In our implementation, we employ DeepSeek-R1-Distill-Qwen-7B [Guo et al., 2025] and evaluate on the MATH-500 [Hendrycks et al., 2021] benchmark. Given a prompt x, the generation process is formalized as $[ R , y ] \sim \pi _ { \theta } ( \cdot \mid x )$ where $R = [ r _ { 1 } , r _ { 2 } , . . . , r _ { n } ]$ represents the reasoning trajectory, y denotes the summarized answer, and $[ \cdot , \cdot ]$ indicates concatenation. Under greedy decoding, the output is deterministically obtained as $y \sim \pi _ { \boldsymbol { \theta } } ( \cdot \cdot | \left[ x , R \right] )$ . Our analysis focuses on the token-level uncertainty within R. When generating the i-th token $r _ { i } ,$ , the conditional distribution is given by $\pi ( r _ { i } ) = \pi _ { \theta } ( r _ { i } \mid x , r _ { 1 } , \ldots , r _ { i - 1 } )$ . We quantify the local uncertainty at position i using the conditional entropy:

$$
H _ { i } = - \sum _ { v \in \mathcal { V } } \pi ( v ) \log \pi ( v ) ,\tag{1}
$$

where V denotes the vocabulary. To identify high-entropy segments, which we treat as candidate highuncertainty regions, we employ a relative thresholding strategy. A token is classified as high-entropy if its entropy value falls within the top 5% of the trajectory. To ensure robustness and filter out noise, we retain a segment only if the majority of tokens within a five-token sliding window satisfy this criterion.

Observations. As illustrated in Figure 1, the vocabulary distribution within high-entropy segments is highly non-uniform, predominantly featuring what we term transition markers rather than domain-specific terminology. The most frequent high-entropy tokens serve as distinct markers of uncertainty and redirection. We observe a strong prevalence of uncertainty markers such as “maybe”, “perhaps”, and “seems”, which are consistent with candidate branching points where the model deliberates between multiple reasoning paths. Furthermore, the high frequency of self-correction markers such as “wait”, “check”, and “hold” suggests that elevated entropy correlates with moments of retrospective verification. These patterns indicate that overthinking is characterized by recursive cycles of hesitation and selfcorrection, where substantial probability mass is allocated to re-evaluating prior reasoning steps rather than advancing toward the solution.

![](images/dcbd2c879db1bcb8441c4c8d5304959f01cc962ac71f661b2dfc680f0bbad071.jpg)  
Figure 1: Top-20 most frequent tokens in highentropy reasoning segments.

## 2.2 Impact of Trajectory Truncation on Answer Quality

The preceding analysis suggests that high-entropy segments correspond to moments of model uncertainty and potential branching. A natural follow-up question arises: do the reasoning steps that follow these transition points contribute meaningfully to thefinal answer? To address this question, we design a truncation experiment.

![](images/509772163717f1a7b5b9a9a79248c46ffdb09d3773f525a9a213f0095dccd744.jpg)  
Figure 2: Impact of reasoning trajectory truncation on model performance. The plot compares the percentage of token savings against the accuracy drop relative to the baseline.

Settings. We identify several high-entropy tokens within a complete reasoning trajectory R using the criteria defined above. These tokens, which we refer to as transition points, partition the trajectory into sequential segments. We construct four experimental conditions, denoted as Trunc@−k for $k \in \{ 1 , \bar { 2 } , 3 , 4 \}$ , where the final k segments are discarded. For each condition, the model generates the final answer y based solely on the truncated prefix. We evaluate performance using two metrics: accuracy drop relative to the full-trajectory baseline and token savings measured as the percentage reduction in reasoning tokens.

Observations. Figure 2 presents the trade-off between computational cost and model performance, providing evidence consistent with some late reasoning segments being redundant. Most notably, the Trunc@−1 condition demonstrates that removing only the final segment yields a substantial 26.8% reduction in token consumption while incurring a negligible accuracy drop of merely 0.2% (from 92.8% to 92.6%). This result suggests that the model typically resolves the core problem well before concluding its generation, which renders the final reasoning phase computationally redundant in many examples. Furthermore, even under the aggressive truncation of Trunc@−4, where 67.4% of reasoning tokens are removed, the model retains the majority of its problem-solving capability, suffering only an 8.6% decline in accuracy.

## 2.3 Summary and Implications

These findings collectively suggest that LRMs exhibit significant overthinking behavior, as a substantial portion of the generated reasoning, especially in later stages, contributes minimally to answer correctness despite substantial computational cost. Crucially, our entropy analysis suggests that this redundancy is not uniformly distributed but rather concentrated around high-entropy transition points. This observation motivates our proposed approach: rather than applying uniform length penalties that indiscriminately suppress all tokens, we focus credit reduction on late high-entropy regions that are empirically associated with unnecessary reasoning extensions.

## 3 Methodology

![](images/8f44d5a853c6e2387e98626d4fa80e57a1f6cb9d09772391faafaf854f20ec33.jpg)  
Figure 3: Overview of Adaptive Entropy-Sparing (AES). AES navigates the suppression-exploration trade-off by selectively penalizing overlength high-entropy suffix tokens associated with superfluous branching, while preserving the original learning signal for concise correct trajectories.

## 3.1 Preliminaries

Reinforcement Learning with Verifiable Rewards. Reinforcement learning has emerged as an effective paradigm for enhancing the reasoning capabilities of LLMs across diverse domains. In settings where ground-truth answers are available, rule-based outcome rewards provide a simple yet effective training signal. Following prior work, we adopt a reward function defined as: $R ( x , y ) = + 1$ for correct responses, $R ( x , y ) = - 1$ for incorrect responses, where x denotes the input query and y denotes the model response.

The standard RL objective with KL regularization can be formulated as:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { x \sim \mathcal { D } } \mathbb { E } _ { y \sim \pi _ { \theta } ( \cdot | x ) } \left[ R ( x , y ) \right] - \beta _ { \mathrm { K L } } D _ { \mathrm { K L } } \left[ \pi _ { \theta } ( \cdot | x ) \right| \left| \pi _ { \mathrm { r e f } } ( \cdot | x ) \right] ,\tag{2}
$$

where $\pi _ { \mathrm { r e f } }$ denotes the reference model and $\beta _ { \mathrm { K L } }$ controls the regularization strength.

Group Relative Policy Optimization. We adopt Group Relative Policy Optimization (GRPO) [Shao et al., 2024] as our base training algorithm. For each query $x ,$ the behavior policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples

a group of G responses $\{ y _ { i } \} _ { i = 1 } ^ { G }$ . The advantage for the i-th response is computed by normalizing rewards within the group:

$$
A _ { i } = \frac { R _ { i } - \mathrm { m e a n } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } \} _ { j = 1 } ^ { G } ) } ,\tag{3}
$$

where $R _ { i } = R ( x , y _ { i } )$ . The GRPO objective is then formulated as:

$$
\mathcal { I } ( \theta ) = \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } , y _ { i } \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | \boldsymbol { x } ) } \Big [ \frac { 1 } { G } \sum _ { i = 1 } ^ { G } \frac { 1 } { | y _ { i } | } \sum _ { t = 1 } ^ { | y _ { i } | } \operatorname* { m i n } \big ( \rho _ { i , t } ( \theta ) A _ { i } , \mathrm { c l i p } ( \rho _ { i , t } ( \theta ) , 1 - \epsilon , 1 + \epsilon ) A _ { i } \big ) \Big ]\tag{4}
$$

where $\begin{array} { r } { \rho _ { i , t } ( { \boldsymbol { \theta } } ) = \frac { \pi _ { \boldsymbol { \theta } } \left( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } \right) } { \pi _ { \boldsymbol { \theta } _ { \mathrm { o l d } } } \left( y _ { i , t } | \boldsymbol { x } , y _ { i , < t } \right) } } \end{array}$ is the importance sampling ratio and ϵ is the clipping threshold.   
We omit the KL term for readability.

## 3.2 Limitations of Global Length Penalties

A common approach to mitigate overthinking is to incorporate a global length penalty into the reward function, penalizing longer responses regardless of content. The modified objective is:

$$
\operatorname* { m a x } _ { \theta } \mathbb { E } _ { \tau \sim \pi _ { \theta } } \left[ R ( \tau ) - \beta _ { \mathrm { l e n } } \cdot | \tau | \right] ,\tag{5}
$$

where |τ | denotes the total number of tokens in trajectory τ and $\beta _ { \mathrm { l e n } }$ is the length penalty coefficient.

Credit Confounding. While conceptually simple, lengthy responses often contain substantial portions of valid intermediate reasoning steps that are essential for reaching the correct answer. The global length penalty uniformly penalizes all tokens in the trajectory, which fails to distinguish between tokens that contribute to sound reasoning and those that belong to unnecessary elaboration.

## 3.3 Adaptive Entropy-Sparing

To address credit confounding, we propose Adaptive Entropy-Sparing (AES), which preserves high entropy tokens in concise or pre-reference regions while penalizing overlength high-entropy suffix tokens. AES is not a classifier of every transition token. Instead, it implements a conservative selective-penalization rule: extra penalties are applied only when a token is high-entropy, appears in a correct trajectory that is longer than an in-group efficient reference, and lies in the overlength suffix beyond that reference. This conservative rule improves the suppression-exploration trade-off by reducing overthinking while preserving useful exploration in concise correct trajectories. An overview of our method is illustrated in Figure 3.

Our method operates in three stages: (1) identifying high-entropy tokens within each trajectory, (2) classifying trajectories based on correctness and length, and (3) applying differentiated reward shaping according to trajectory type. Here, a trajectory denotes the full generated completion, including the reasoning trace and final answer.

## 3.3.1 High-Entropy Token Identification

For each sampled trajectory $\tau = ( y _ { 1 } , \dots y _ { T } )$ , we compute the token-level conditional entropy at each generation step from the rollout logits produced by the behavior policy that sampled the trajectory:

$$
H _ { t } = - \sum _ { v \in \mathcal { V } } \pi _ { \theta _ { \mathrm { o l d } } } ( v | x , y _ { < t } ) \log \pi _ { \theta _ { \mathrm { o l d } } } ( v | x , y _ { < t } ) ,\tag{6}
$$

where V denotes the vocabulary. This uses information already available during rollout collection and keeps the shaped advantages fixed during the subsequent policy update. We then identify the set of high-entropy tokens as:

$$
{ \cal { S } } ( \tau { } ) = \{ t : H _ { t } \geq \gamma \} ,\tag{7}
$$

where $\gamma$ is the entropy threshold.

Threshold Selection. The choice of $\gamma$ determines which tokens are candidates for differentiated treatment. We adopt a quantile-based thresholding strategy: for each trajectory, we set γ as the q-th quantile of the entropy values $\{ H _ { t } \} _ { t = 1 } ^ { T } , \mathrm { e . g . , } q = 0 . 9$ selects the top 10% highest-entropy tokens. This approach adapts to varying difficulty levels across problems and to entropy scale shifts across training stages; unlike a fixed absolute threshold, it remains active when the policy becomes more confident and absolute entropy values decrease.

## 3.3.2 Trajectory Categorization and Advantage Shaping

Given a set of sampled trajectories for each query, we first compute the trajectory-level advantage via GRPO normalization across the sampling group. We then classify trajectories into three categories based on correctness and length, applying differentiated advantage shaping to each:

Category A: Correct and Concise Trajectories. For trajectories that are both correct $( C ( \tau ) = 1 )$ and concise $( | \tau | = T _ { \mathrm { r e f } } )$ , we preserve the original advantage for all tokens to reinforce high-quality reasoning behavior:

$$
\tilde { A } _ { t } = A ( \tau ) , \quad \forall t \in \{ 1 , \dots , | \tau | \} ,\tag{8}
$$

Category B: Incorrect Trajectories. For incorrect trajectories $( C ( \tau ) = 0 )$ , we preserve the original advantage for all tokens to maintain learning pressure toward correctness, without adding AESspecific suffix penalties before the policy has found a valid solution:

$$
\tilde { A } _ { t } = A ( \tau ) , \quad \forall t \in \{ 1 , \dots , | \tau | \} ,\tag{9}
$$

Category C: Correct but Overthinking Trajectories. For trajectories that are correct but exhibit overthinking $( C ( \tau ) = 1$ and $| \tau | > T _ { \mathrm { r e f } } )$ , we selectively penalize high-entropy tokens while sparing the rest. Specifically, we re-compute the token-level advantage as:

$$
\tilde { A } _ { i , t } = \left\{ \begin{array} { l l } { A _ { i } - \beta _ { \mathrm { A E S } } \cdot \eta _ { i } \cdot \lambda ( t ) , } & { \mathrm { i f ~ } t \in S ( \tau _ { i } ) \mathrm { ~ a n d ~ } t > T _ { \mathrm { r e f } } , } \\ { A _ { i } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

where $\beta _ { \mathrm { A E S } }$ is the AES penalty coefficient, $\eta _ { i }$ is a group-wise min-max normalized trajectory-level excess-length factor, λ(t) is a suffix-position scaling factor, and $S ( \tau _ { i } )$ denotes the set of high-entropy tokens; full definitions are provided in Appendix C. This formulation ensures that tokens most likely to be associated with superfluous branching receive reduced advantage values, while those outside the overlength high-entropy suffix, including high-entropy positions before $T _ { \mathrm { r e f } }$ , retain their original learning signals.

The reference length $T _ { \mathrm { r e f } }$ is defined as the length of the shortest correct trajectory within each sampling group. It serves only as the boundary for identifying overlength suffixes, rather than as a token-level imitation target. If the group contains no correct trajectory, AES disables selective shaping and retains the original GRPO advantages for all trajectories in the group.

## 3.4 Entropy and Expected Reasoning Length

We relate token-level entropy to future reasoning cost through a simple branching view of generated reasoning trajectories. The central intuition is that high-entropy tokens often occur at states where multiple continuations are plausible, and such states may lead the model into additional verification or backtracking.

Effective branching factor. Let $\boldsymbol { s } _ { t } = \left( x , y _ { < t } \right)$ denote the state at generation step $t ,$ comprising the input and all previously generated tokens. The conditional entropy $\mathsf { \bar { H } } _ { t } = H ( \pi _ { \theta } ( \cdot \mid s _ { t } ) )$ captures the model’s uncertainty over the next token. We define the effective branchingfactor as $B _ { t } : = \exp ( H _ { t } )$ which represents the effective number of plausible continuations available at state $s _ { t }$

Branching model of overthinking. We model the reasoning process as a stochastic branching process. At each step t, the model either proceeds toward termination or initiates a verification/backtracking phase with probability $q _ { t }$ . We use the following local parameterization:

$$
q _ { t } = 1 - \exp ( - \alpha B _ { t } ) ,\tag{11}
$$

where $\alpha > 0$ controls how strongly the effective branching factor increases the probability of additional reasoning. Under the branching assumptions detailed in Appendix F, this yields the following local relation:

Proposition 3.1. Let $L _ { t } = \mathbb { E } [ T - t \mid s _ { t } ]$ denote the expected remaining trajectory length at step t. Under the local homogeneous branching model with $q _ { t } = 1 - \exp ( - \alpha B _ { t } ) , i f \alpha B _ { t } \le 1$ , then there exist constants $c _ { 1 } , c _ { 2 } > 0$ such that

$$
c _ { 1 } B _ { t } \leq L _ { t } \leq c _ { 2 } B _ { t } .\tag{12}
$$

Since $B _ { t } = \exp ( H _ { t } )$ , the expected residual length locally scales with the exponential of token-level entropy under this branching approximation.

Table 1: Main results on mathematical reasoning benchmarks. We report accuracy (%) and average response length (Len). In the Mean columns, bold and underlined values indicate the best and second-best aggregate values among rows shown, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">AMC</td><td colspan="2">OlympiadBench</td><td colspan="2">Mean</td></tr><tr><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td rowspan="6">R1-1.5B</td><td>Base Model</td><td>82.1</td><td>1813</td><td>83.5</td><td>5736</td><td>49.3</td><td>6820</td><td>44.8</td><td>11497</td><td>64.9</td><td>6467</td></tr><tr><td>Efficient Reason</td><td>78.1</td><td>324</td><td>81.8</td><td>2522</td><td>46.6</td><td>3345</td><td>45.2</td><td>6302</td><td>62.9</td><td>3123</td></tr><tr><td>LC-R1</td><td>81.2</td><td>745</td><td>80.2</td><td>2505</td><td>50.3</td><td>3432</td><td>43.0</td><td>5836</td><td>63.7</td><td>3130</td></tr><tr><td>Laser</td><td>87.0</td><td>1778</td><td>84.8</td><td>3321</td><td>51.4</td><td>4259</td><td>49.4</td><td>5819</td><td>68.2</td><td>3794</td></tr><tr><td>ThinkPrune</td><td>85.7</td><td>1131</td><td>83.7</td><td>2791</td><td>51.1</td><td>3527</td><td>46.3</td><td>5520</td><td>66.7</td><td>3242</td></tr><tr><td>Thinkless</td><td>85.1</td><td>986</td><td>82.3</td><td>3586</td><td>51.8</td><td>4463</td><td>47.0</td><td>7359</td><td>66.6</td><td>4099</td></tr><tr><td></td><td>AES (Ours)</td><td>87.4</td><td>1121</td><td>85.7</td><td>2643</td><td>51.6</td><td>3274</td><td>49.9</td><td>5156</td><td>68.7</td><td>3048</td></tr><tr><td rowspan="6">R1-7B</td><td>Base Model</td><td>90.1</td><td>1032</td><td>92.7</td><td>3917</td><td>55.3</td><td>5100</td><td>58.1</td><td>8800</td><td>74.1</td><td>4712</td></tr><tr><td>Efficient Reason</td><td>87.8</td><td>278</td><td>90.7</td><td>2189</td><td>50.9</td><td>3050</td><td>55.9</td><td>6063</td><td>71.3</td><td>2895</td></tr><tr><td>LC-R1</td><td>87.3</td><td>423</td><td>89.1</td><td>1552</td><td>50.3</td><td>2446</td><td>55.0</td><td>4440</td><td>70.4</td><td>2215</td></tr><tr><td>GRPO-LEAD</td><td>90.1</td><td>814</td><td>92.3</td><td>3220</td><td>55.4</td><td>3991</td><td>58.4</td><td>6767</td><td>74.1</td><td>3698</td></tr><tr><td>VeriThinker</td><td>92.5</td><td>763</td><td>92.1</td><td>2589</td><td>51.3</td><td>3589</td><td>58.2</td><td>6292</td><td>73.5</td><td>3308</td></tr><tr><td>AES (Ours)</td><td>92.9</td><td>734</td><td>91.7</td><td>2103</td><td>57.4</td><td>2897</td><td>58.9</td><td>4305</td><td>75.2</td><td>2510</td></tr></table>

Table 2: Results on competition-level mathematics and out-of-domain benchmarks. We report accuracy (%) and average response length (Len). In the Mean columns, bold and underlined values indicate the best and second-best aggregate values among rows shown, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">LSAT</td><td colspan="2">GPQA</td><td colspan="2">Mean</td></tr><tr><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td><td>Acc</td><td>Len</td></tr><tr><td rowspan="7">R1-1.5B</td><td>Base Model</td><td>27.9</td><td>16645</td><td>23.1</td><td>16623</td><td>22.8</td><td>14549</td><td>34.1</td><td>10090</td><td>27.0</td><td>14477</td></tr><tr><td>Efficient Reason</td><td>29.2</td><td>10517</td><td>22.1</td><td>9755</td><td>23.2</td><td>7694</td><td>33.0</td><td>6037</td><td>26.9</td><td>8501</td></tr><tr><td>LC-R1</td><td>24.6</td><td>8684</td><td>22.1</td><td>8526</td><td>24.4</td><td>11711</td><td>35.5</td><td>8575</td><td>26.7</td><td>9374</td></tr><tr><td>Laser</td><td>34.4</td><td>8492</td><td>22.5</td><td>8368</td><td>24.1</td><td>8059</td><td>35.5</td><td>5842</td><td>29.1</td><td>7690</td></tr><tr><td>ThinkPrune</td><td>30.4</td><td>7690</td><td>22.5</td><td>7750</td><td>25.5</td><td>6969</td><td>34.6</td><td>5785</td><td>28.3</td><td>7049</td></tr><tr><td>Thinkless</td><td>29.2</td><td>10493</td><td>23.5</td><td>10491</td><td>25.7</td><td>11214</td><td>30.9</td><td>10358</td><td>27.3</td><td>10639</td></tr><tr><td>AES (Ours)</td><td>31.5</td><td>7401</td><td>24.1</td><td>7933</td><td>25.9</td><td>7047</td><td>38.0</td><td>5194</td><td>29.9</td><td>6894</td></tr><tr><td rowspan="6">R1-7B</td><td>Base Model</td><td>53.1</td><td>13107</td><td>37.3</td><td>14485</td><td>41.1</td><td>10854</td><td>48.9</td><td>8079</td><td>45.1</td><td>11631</td></tr><tr><td>Efficient Reason</td><td>50.8</td><td>9855</td><td>35.6</td><td>11075</td><td>42.0</td><td>7836</td><td>48.3</td><td>5679</td><td>44.2</td><td>8611</td></tr><tr><td>LC-R1</td><td>52.5</td><td>6915</td><td>32.1</td><td>8193</td><td>43.3</td><td>6954</td><td>49.7</td><td>4756</td><td>44.4</td><td>6705</td></tr><tr><td>GRPO-LEAD</td><td>54.2</td><td>10437</td><td>36.9</td><td>11621</td><td>42.3</td><td>8622</td><td>50.2</td><td>6754</td><td>45.9</td><td>9359</td></tr><tr><td>VeriThinker</td><td>54.6</td><td>9927</td><td>37.9</td><td>10548</td><td>29.5</td><td>6073</td><td>46.8</td><td>4456</td><td>42.2</td><td>7751</td></tr><tr><td>AES (Ours)</td><td>54.9</td><td>7075</td><td>38.4</td><td>7209</td><td>43.9</td><td>5318</td><td>51.3</td><td>4157</td><td>47.1</td><td>5940</td></tr></table>

This proposition provides an interpretation of why high-entropy tokens can serve as useful indicators of branching-prone regions in reasoning trajectories. Consistent with this view, longer correct trajectories exhibit higher mean and 90th-percentile token entropy than the shortest correct trajectory in the same sampling group (Appendix E). AES uses this signal conservatively by applying credit reduction only to high-entropy tokens in the overlength suffix of correct trajectories, while preserving earlier high-entropy tokens that may be necessary for productive exploration.

## 4 Experiments

## 4.1 Main Results

Overall Performance. AES consistently achieves the best trade-off between accuracy and efficiency across both model scales, while using the shortest mean length in three settings and the secondshortest mean length in the remaining setting. On mathematical reasoning benchmarks (Table 1), AES attains mean accuracies of 68.7% for R1-1.5B and 75.2% for R1-7B, while achieving length reductions of 52.9% and 46.7% compared to the base models, respectively. On the more challenging competition-level and out-of-domain benchmarks (Table 2), AES improves mean accuracy by 2.9% for R1-1.5B and 2.0% for R1-7B, while reducing response length by 52.4% and 48.9%, respectively.

![](images/b6a506d9d067b00fab9681969eeb19f2b20eaa695fcae25c2f2d035917ea3fd1.jpg)  
(a) AMC

![](images/db5863a82ba806fd405c5773bd44720b919fe02a31dd4b4026f482e6247f7a7c.jpg)  
(b) MATH-500

![](images/dd195e6dd04f3548954c3b40da0e68bc5ac7570bd9abca90b8e551f0b0c2446b.jpg)  
(c) GSM8K  
Figure 4: Performance Comparison under Varying Token Budgets. We compare the pass rates of AES and the baseline model under budget forcing, with maximum output lengths ranging from 512 to 4,096 tokens.

The tables report accuracy and length as separate axes rather than a formal Pareto-frontier analysis.   
We therefore interpret AES primarily as an average accuracy-efficiency improvement.

Comparison with Uniform Penalty Methods. Methods that apply uniform length penalties, such as Efficient Reason and LC-R1, achieve large length reductions but suffer from notable accuracy degradation. For instance, on R1-1.5B, Efficient Reason reduces length by 51.7% on mathematical benchmarks but incurs a 2.0% accuracy drop (64.9% to 62.9%). Similarly, LC-R1 exhibits a 1.2% accuracy decrease despite moderate length savings. These results support the concern that uniform penalties can cause credit confounding by suppressing both essential reasoning steps and superfluous elaboration. By contrast, AES achieves comparable or greater length reductions while improving accuracy, indicating that entropy-guided selective penalization better preserves task-relevant reasoning while discouraging redundant generation.

Comparison with Adaptive Methods. Among adaptive approaches, GRPO-LEAD and VeriThinker maintain competitive accuracy but provide limited efficiency gains. GRPO-LEAD achieves competitive accuracy but reduces length by only 21.5% compared to the base model. VeriThinker shows similar patterns, trading modest length reduction for accuracy preservation. Thinkless, which adaptively selects reasoning modes, achieves only modest length reduction on challenging benchmarks (10639 vs. 14477 for R1-1.5B). AES outperforms these methods by applying credit reduction to overlength high-entropy suffix tokens, achieving superior efficiency without sacrificing accuracy.

Performance on Challenging Benchmarks. The advantages of AES become more pronounced on difficult tasks. On AIME24 and AIME25, which represent competition-level mathematics, AES provides competitive accuracy while substantially shortening responses. For R1-7B, AES improves AIME25 accuracy from 37.3% to 38.4% while reducing length by 50.2%. On out-of-domain benchmarks, AES shows positive transfer within the evaluated model family. On GPQA, AES improves accuracy by 3.9% for R1-1.5B and 2.4% for R1-7B, with length reductions exceeding 48%. On LSAT, AES achieves 3.1% accuracy improvement for R1-1.5B and 2.8% for R1-7B, while reducing response length by 51.6% and 51.0%, respectively. These results suggest that AES improves the average accuracy-efficiency trade-off by reducing empirically redundant continuation.

## 4.2 Analysis and Ablation Studies

Token Budget Constraints. To examine whether AES produces more concise reasoning paths, we evaluate performance under varying token budgets (512–4K) against the budget forcing baseline [Muennighoff et al., 2025]. As shown in Figure 4, AES has higher pass rates than budget forcing across the evaluated budget levels. On GSM8K, AES achieves 85.2% at 1K tokens compared to 80.1% for budget forcing (+5.1%), with similar improvements observed on MATH-500 (+3.6% at 1K) and AMC (+1.2% at 2K). Notably, on GSM8K, AES at a 1K budget already exceeds budget forcing at 2K. These results suggest that AES not only improves token efficiency, but also learns to construct more streamlined reasoning trajectories, achieving competitive accuracy with substantially fewer tokens.

Component Analysis. Table 3 presents an ablation study on the key components of AES. Random selection yields the lowest accuracy (88.5%), indicating that the entropy criterion is more informative than a matched random token set in this setting. This suggests that indiscriminate penalization can substantially reduce response length, but may also suppress reasoning steps that are necessary for problem solving, thereby degrading accuracy. Removing trajectory-level scaling reduces accuracy to 91.1%, suggesting that differentiated penalty weights based on the degree of overthinking across trajectories contribute meaningfully to performance. Similarly, ablating token-level scaling results in 90.0% accuracy with increased token usage, suggesting that adaptive penalties within each trajectory help the model focus on high-uncertainty suffix positions. The full AES achieves the best accuracy (91.7%) with moderate token consumption, supporting the usefulness of both hierarchical scaling mechanisms.

Table 3: We evaluate the contribution of each design choice in AES on MATH-500. “Random Selection” penalizes randomly selected tokens with the same budget as the high-entropy set. “w/o Scaling (Trajectory-Level)” applies uniform penalty weights across all overthinking trajectories. “w/o Scaling (Token-Level)” uses identical penalty weights for all high-entropy tokens within each trajectory.
<table><tr><td>Variant</td><td>Accuracy</td><td>Tokens</td></tr><tr><td>Full AES</td><td>91.7</td><td>2103</td></tr><tr><td>w/o Scaling (Trajectory-Level)</td><td>91.1</td><td>2271</td></tr><tr><td>w/o Scaling (Token-Level)</td><td>90.0</td><td>2546</td></tr><tr><td>Random Selection</td><td>88.5</td><td>1734</td></tr></table>

Table 4: Sensitivity to AES hyperparameters on R1-7B. We report mean accuracy (%) and average response length on the mathematical benchmark group and the hard/OOD benchmark group.
<table><tr><td>Setting</td><td>Hard/OOD Acc</td><td>Hard/OOD Len</td><td>Math Acc</td><td>Math Len</td></tr><tr><td> $\beta _ { \mathrm { A E S } } = 1 . 0$ </td><td>47.2</td><td>6544</td><td>75.2</td><td>3016</td></tr><tr><td> $\beta _ { \mathrm { A E S } } = 1 . 5$ </td><td>46.9</td><td>6045</td><td>75.1</td><td>2667</td></tr><tr><td> $\beta _ { \mathrm { A E S } } = 2 . 0$ </td><td>47.1</td><td>5940</td><td>75.2</td><td>2510</td></tr><tr><td> $\beta _ { \mathrm { A E S } } = 2 . 5$ </td><td>46.8</td><td>5432</td><td>74.7</td><td>2273</td></tr><tr><td> $q = 0 . 6 0$ </td><td>46.1</td><td>5537</td><td>74.3</td><td>2061</td></tr><tr><td> $q = 0 . 7 5$ </td><td>47.0</td><td>5770</td><td>74.9</td><td>2149</td></tr><tr><td> $q = 0 . 9 0$ </td><td>47.1</td><td>5940</td><td>75.2</td><td>2510</td></tr></table>

Hyperparameter Sensitivity. Table 4 evaluates the sensitivity of AES to the penalty coefficient $\beta _ { \mathrm { A E S } }$ and the entropy threshold q. Overall, AES remains robust across a range of hyperparameter settings. Increasing $\beta _ { \mathrm { A E S } }$ generally leads to shorter responses, although overly large penalty weights can introduce a modest reduction in accuracy. Similarly, decreasing q expands the set of penalized tokens and can further reduce generation length; however, the resulting accuracy trends are not strictly monotonic across benchmark groups, suggesting that the optimal degree of token suppression depends on task-specific reasoning characteristics. Based on this trade-off, we set $\beta _ { \mathrm { A E S } } = 2 . 0$ and $q = 0 . 9$ as a balanced configuration that achieves substantial length reduction while preserving strong accuracy. We keep the group size fixed at $G = 8 ,$ following standard GRPO practice. Since AES modifies token-level credit assignment after group-level rewards are computed, its mechanism is largely orthogonal to the choice of group size.

## 5 Conclusion

In this work, we introduce AES, a post-training method for mitigating overthinking in LRMs. We first conduct an empirical analysis of reasoning trajectories and observe that high-entropy tokens often function as transition points associated with unnecessary reasoning expansion. Motivated by this observation, AES selectively penalizes high-entropy suffix tokens that are likely to induce redundant elaboration, while preserving tokens that support productive reasoning. We further provide a branching interpretation that relates token-level entropy to expected residual reasoning cost under local modeling assumptions. Extensive experiments demonstrate that AES reduces response length by approximately 50% while simultaneously improving accuracy, substantially outperforming existing efficiency-oriented methods

## Acknowledgments and Disclosure of Funding

This work was supported in part by the National Natural Science Foundation of China under Grant 62372132. The authors would also like to thank Huawei Ascend Cloud Ecological Development Project for providing high-performance Ascend 910 processors. The authors declare no competing interests.

## References

Pranjal Aggarwal and Sean Welleck. L1: Controlling how long a reasoning model thinks with reinforcement learning. arXiv preprint arXiv:2503.04697, 2025.

Daman Arora and Andrea Zanette. Training language models to reason efficiently. arXiv preprint arXiv:2502.04463, 2025.

Simon A Aytes, Jinheon Baek, and Sung Ju Hwang. Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching. arXiv preprint arXiv:2503.05179, 2025.

Emil Biju, Shayan Talaei, Zhemin Huang, Mohammadreza Pourreza, Azalia Mirhoseini, and Amin Saberi. Sprint: Enabling interleaved planning and parallelized execution in reasoning models. arXiv preprint arXiv:2506.05745, 2025.

Zigeng Chen, Xinyin Ma, Gongfan Fang, Ruonan Yu, and Xinchao Wang. Verithinker: Learning to verify makes reasoning model efficient. arXiv preprint arXiv:2505.17941, 2025.

Zhengxiang Cheng, Dongping Chen, Mingyang Fu, and Tianyi Zhou. Optimizing length compression in large reasoning models. arXiv preprint arXiv:2506.14755, 2025.

Zheng Chu, Jingchang Chen, Qianglong Chen, Weijiang Yu, Tao He, Haotian Wang, Weihua Peng, Ming Liu, Bing Qin, and Ting Liu. Navigate through enigmatic labyrinth a survey of chain of thought reasoning: Advances, frontiers and future. In Proc. Association for Computational Linguistics (ACL), pages 1173–1203, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Muzhi Dai, Chenxu Yang, and Qingyi Si. S-grpo: Early exit via reinforcement learning in reasoning models. arXiv preprint arXiv:2505.07686, 2025.

Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step. arXiv preprint arXiv:2405.14838, 2024.

Gongfan Fang, Xinyin Ma, and Xinchao Wang. Thinkless: Llm learns when to think. arXiv preprint arXiv:2505.13379, 2025.

Sicheng Feng, Gongfan Fang, Xinyin Ma, and Xinchao Wang. Efficient reasoning models: A survey. arXiv preprint arXiv:2504.10903, 2025.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think before you speak: Training language models with pause tokens. arXiv preprint arXiv:2310.02226, 2023.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, et al. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. In Proc. Association for Computational Linguistics (ACL), pages 3828–3850, 2024.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

Bairu Hou, Yang Zhang, Jiabao Ji, Yujian Liu, Kaizhi Qian, Jacob Andreas, and Shiyu Chang. Thinkprune: Pruning long chain-of-thought of llms via reinforcement learning. arXiv preprint arXiv:2504.01296, 2025.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Cong Jiang, Xiaofeng Zhang, Fangzhi Zhu, XiaoWei Chen, Junxiong Zhu, and Zheng Zhang. Rethinking LLM reasoning: From explicit trajectories to latent representations. In The Fourteenth International Conference on Learning Representations, 2026a.

Juyong Jiang, Fan Wang, Jiasi Shen, Sungju Kim, and Sunghun Kim. A survey on large language models for code generation. ACM Trans. Softw. Eng. Methodol., 35(2), January 2026b. ISSN 1049-331X. doi: 10.1145/3747588. URL https://doi.org/10.1145/3747588.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. C3ot: Generating shorter chain-of-thought without compromising effectiveness. In Proc. Association for the Advancement of Artificial Intelligence (AAAI), pages 24312–24320, 2025.

Ruosen Li, Ziming Luo, Quan Zhang, Ruochen Li, Ben Zhou, Ali Payani, and Xinya Du. AALC: Large language model efficient reasoning via adaptive accuracy-length control. arXiv preprint arXiv:2506.20160, 2025a.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, et al. From system 1 to system 2: A survey of reasoning large language models. arXiv preprint arXiv:2502.17419, 2025b.

Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, and Caiming Xiong. Reward-guided speculative decoding for efficient llm reasoning. arXiv preprint arXiv:2501.19324, 2025.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In Proc. International Conference on Learning Representations (ICLR), 2024.

Wei Liu, Ruochen Zhou, Yiyun Deng, Yuzhen Huang, Junteng Liu, Yuntian Deng, Yizhe Zhang, and Junxian He. Learn to reason efficiently with adaptive length-based reward shaping. arXiv preprint arXiv:2505.15612, 2025.

Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. O1-pruner: Length-harmonizing fine-tuning for o1-like reasoning pruning. arXiv preprint arXiv:2501.12570, 2025.

MAA. American mathematics competitions. In American Mathematics Competitions, 2023.

Mathematical Association of America. American Invitational Mathematics Examination (AIME), 2025.

Weikang Meng, Yadan Luo, Liangyu Huo, Yaowei Wang, Xin Li, and Zheng Zhang. NaLaFormer: Norm-aware linear attention for Transformer models. arXiv preprint arXiv:2506.21137, 2025.

Weikang Meng, Yadan Luo, Liangyu Huo, Yingjian Li, Yaowei Wang, Zheng Zhang, and Heng Tao Shen. PolaFormer++: Polarity-aware linear attention with channel-wise spikiness. IEEE Transactions on Pattern Analysis and Machine Intelligence, 48(11):15412–15426, 2026.

Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, et al. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. arXiv preprint arXiv:2412.09413, 2024.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. s1: Simple test-time scaling. arXiv preprint arXiv:2501.19393, 2025.

Qwen Team. Qwq-32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.github.io/blog/qwq-32b/.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. In Proc. Conference on Language Modeling (COLM), 2024.

Yangjun Ruan, Neil Band, Chris J Maddison, and Tatsunori Hashimoto. Reasoning to learn from latent thoughts. arXiv preprint arXiv:2503.18866, 2025.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Tiancheng Su, Meicong Zhang, and Guoxiu He. Entropy-aware speculative decoding toward improved llm reasoning. arXiv preprint arXiv:2512.23765, 2025.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, et al. Stop overthinking: A survey on efficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

Leandro von Werra, Younes Belkada, Lewis Tunstall, Edward Beeching, Tristan Thrush, Nathan Lambert, Shengyi Huang, Kashif Rasul, and Quentin Gallouédec. Trl: Transformer reinforcement learning. https://github.com/huggingface/trl, 2020.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, Yuqiong Liu, An Yang, Andrew Zhao, Yang Yue, Shiji Song, Bowen Yu, Gao Huang, and Junyang Lin. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for LLM reasoning. In Advances in Neural Information Processing Systems, 2025.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.

Xumeng Wen, Zihan Liu, Shun Zheng, Zhijian Xu, Shengyu Ye, Zhirong Wu, Xiao Liang, Yang Wang, Junjie Li, Ziming Miao, et al. Reinforcement learning with verifiable rewards implicitly incentivizes correct reasoning in base llms. arXiv preprint arXiv:2506.14245, 2025.

Junde Wu, Jiayuan Zhu, Yuyuan Liu, Min Xu, and Yueming Jin. Agentic reasoning: A streamlined framework for enhancing llm reasoning with agentic tools. arXiv preprint arXiv:2502.04644, 2025.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. Chain of draft: Thinking faster by writing less. arXiv preprint arXiv:2502.18600, 2025.

Edward Yeo, Yuxuan Tong, Morry Niu, Graham Neubig, and Xiang Yue. Demystifying long chain-of-thought reasoning in llms. arXiv preprint arXiv:2502.03373, 2025.

Xixian Yong, Xiao Zhou, Yingying Zhang, Jinlin Li, Yefeng Zheng, and Xian Wu. Think or not? exploring thinking efficiency in large reasoning models via an information-theoretic lens. arXiv preprint arXiv:2505.18237, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Wenhao Zeng, Yaoning Wang, Chao Hu, Yuling Shi, Chengcheng Wan, Hongyu Zhang, and Xiaodong Gu. Pruning the unsurprising: Efficient LLM reasoning via first-token surprisal. arXiv preprint arXiv:2508.05988, 2025.

Jixiao Zhang and Chunsheng Zuo. Grpo-lead: A difficulty-aware reinforcement learning approach for concise mathematical reasoning in language models. arXiv preprint arXiv:2504.09696, 2025.

Liang Zhang, Yu Zhao, Longyue Wang, Tianqi Shi, Weihua Luo, Kaifu Zhang, and Jinsong Su. A state-transition framework for efficient LLM reasoning. In Proc. International Conference on Learning Representations (ICLR), 2026.

Wanjun Zhong, Ruixiang Cui, Yiduo Guo, Yaobo Liang, Shuai Lu, Yanlin Wang, Amin Saied, Weizhu Chen, and Nan Duan. Agieval: A human-centric benchmark for evaluating foundation models. arXiv preprint arXiv:2304.06364, 2023.

## A Scope and Limitations

Our evidence is empirical and tied to a fixed evaluation protocol. The main results are most direct for DeepSeek-R1-Distill-Qwen 1.5B and 7B models, and cross-family validation on larger LRMs remains an important direction for future work.

AES inherits a basic ambiguity shared by efficient-reasoning methods: additional reasoning can be either redundant verification or genuinely useful exploration. The shortest correct in-group response provides a simple within-prompt efficiency reference, but it is not guaranteed to be the only valid reference, especially for problems near the model’s capability boundary. Finally, AES is designed to address uncertainty-driven branching behavior rather than low-entropy repetitive loops; the latter were rare in our rollouts, but remain outside the method’s direct scope.

## B Related Work

Large Reasoning Models. Chain-of-Thought prompting [Wei et al., 2022, Chu et al., 2024] has enabled LLMs to solve complex problems through step-by-step reasoning. Building on this foundation, recent slow-thinking models [Jaech et al., 2024, Guo et al., 2025, Qwen Team, 2025] leverage reinforcement learning with verifiable rewards [Wen et al., 2025] to generate extended reasoning trajectories. While effective, these models often suffer from overthinking [Sui et al., 2025], producing verbose outputs with redundant verification steps that inflate computational costs.

Efficient Reasoning. To mitigate overthinking, prior work has explored several strategies to enhance reasoning efficiency [Feng et al., 2025]. A prominent line of research employs reinforcement learning to explicitly penalize verbosity by incorporating length-based penalties into the reward function [Luo et al., 2025, Aggarwal and Welleck, 2025, Hou et al., 2025, Yeo et al., 2025, Li et al., 2025a]. However, these methods apply uniform penalties across all tokens, failing to distinguish essential reasoning steps from superfluous ones. Other approaches focus on inference-time prompting to guide more concise generation [Xu et al., 2025, Aytes et al., 2025] or fine-tuning on compressed reasoning chains [Kang et al., 2025, Goyal et al., 2023, Deng et al., 2024]. Recent reasoning-efficiency work also explores early stopping and alternative execution structures: S-GRPO constructs serial early-exit groups and decaying rewards to encourage earlier correct answers [Dai et al., 2025]; Think-or-Not uses information-theoretic signals for entropy-based adaptive halting at inference time [Yong et al., 2025]; and SPRINT reorganizes reasoning trajectories into planning and parallel execution rounds to reduce sequential tokens [Biju et al., 2025]. These methods show that regulating intermediate reasoning is a central theme in efficient LRMs. Token entropy has also been used to locate decision points in reasoning, either to concentrate RLVR updates on high-entropy forking tokens [Wang et al., 2025] or to prune low-surprisal steps for efficient chain-of-thought distillation [Zeng et al., 2025], whereas AES uses it conditionally to reduce credit only for high-entropy suffix tokens of overlong correct trajectories. Beyond shortening outputs, linear attention reduces the per-token cost of generation [Meng et al., 2026, 2025] and has been used to model reasoning as state transitions [Zhang et al., 2026]. AES occupies a complementary point in this design space: it keeps the standard rollout format, adds no test-time controller, and performs token-level advantage shaping only on high-entropy suffix positions in correct overlong samples.

## C Details of Suffix-Aware Scaling

For each correct overthinking trajectory τ<sub>i</sub>, we compute its relative excess length:

$$
\Delta _ { i } = \frac { \left| \tau _ { i } \right| - T _ { \mathrm { r e f } } } { T _ { \mathrm { r e f } } } .\tag{13}
$$

Let $\mathcal { O } = \{ i : C ( \tau _ { i } ) = 1 , | \tau _ { i } | > T _ { \mathrm { r e f } } \}$ denote the set of correct overthinking trajectories in the same sampling group. We apply min-max normalization within this set:

$$
\eta _ { i } = \frac { \Delta _ { i } - \operatorname* { m i n } _ { j \in \mathcal { O } } \Delta _ { j } } { \operatorname* { m a x } _ { j \in \mathcal { O } } \Delta _ { j } - \operatorname* { m i n } _ { j \in \mathcal { O } } \Delta _ { j } + \epsilon } ,\tag{14}
$$

where ϵ is a small constant for numerical stability. If all trajectories in O have the same excess length, we set $\eta _ { i } = 1$

For token position t, we define the token-level suffix scaling factor as

$$
\lambda ( t ) = \log \left( 1 + \operatorname* { m a x } \left( 0 , \frac { t - T _ { \mathrm { r e f } } } { T _ { \mathrm { r e f } } } \right) \right) .\tag{15}
$$

The trajectory-level factor $\eta _ { i }$ differentiates correct overthinking trajectories according to their relative excess length within the sampling group, while the token-level factor λ(t) differentiates high-entropy suffix tokens according to their distance from the reference boundary. The logarithmic form makes λ(t) grow sublinearly, preventing very late tokens in extremely long rollouts from dominating the update. Together, these two factors apply mild or no credit reduction near the efficient reference boundary, while assigning stronger but controlled penalties to later high-entropy suffix tokens in trajectories with larger excess length. This focuses shaping on transition points that are more likely to correspond to redundant verification or superfluous branching, consistent with the truncation results in Figure 2, while preserving pre-reference high-entropy tokens that may support necessary exploration.

## D Experimental Settings

Datasets. For post-training, we utilize the DAPO-Math-17k dataset [Yu et al., 2025]. To comprehensively evaluate our method, we select several reasoning benchmarks spanning mathematical, logical, and scientific domains. For mathematical reasoning, we include GSM8K [Cobbe et al., 2021] for multi-step arithmetic word problems, and MATH-500 [Hendrycks et al., 2021, Lightman et al., 2024] for competition-level mathematics. We further incorporate AMC [MAA, 2023], AIME24, and AIME25 [Mathematical Association of America, 2025] from the American Mathematics Competitions, which demand creative problem-solving and non-routine strategies. OlympiadBench [He et al., 2024] provides olympiad-level problems in mathematics and physics requiring rigorous multi-step proofs. For out-of-domain evaluation, we include LSAT [Zhong et al., 2023] for analytical deduction, and GPQA [Rein et al., 2024] for graduate-level scientific reasoning in physics, chemistry, and biology.

Comparison Models. We compare AES against several baseline and published efficient-reasoning methods. The Base Model refers to the original DeepSeek-R1-Distill-Qwen-1.5B and -7B models [Guo et al., 2025] (R1-1.5B and R1-7B) without any efficiency optimization. Efficient Reason [Arora and Zanette, 2025] applies standard GRPO with a uniform length penalty term added to the reward. GRPO-LEAD [Zhang and Zuo, 2025] introduces length-regularized rewards, explicit penalties for incorrect solutions, and difficulty-aware advantage reweighting to enhance both reasoning accuracy and efficiency. LC-R1 [Cheng et al., 2025] combines length and compression rewards to encourage the model to retain only the most critical reasoning steps. ThinkPrune [Hou et al., 2025] trains the model under a limited token budget to enforce concise generation. Laser [Liu et al., 2025] introduces a difficulty-aware length penalty reward that adapts penalization strength based on problem complexity. Thinkless [Fang et al., 2025] adaptively selects between short-form and long-form reasoning depending on problem complexity. VeriThinker [Chen et al., 2025] fine-tunes the model to verify the correctness of CoT solutions, thereby avoiding unnecessary self-reflection steps. Due to the availability of official implementations, we include different subsets of baselines for each model scale.

Experimental Details. Our framework is implemented using the open-source TRL library [von Werra et al., 2020]. We train with GRPO [Shao et al., 2024] for 1,500 optimization steps using a learning rate of $1 \times 1 0 ^ { - 6 }$ , a batch size of 512, a group size of $G = 8$ , and a maximum completion length of 12,000 tokens. The GRPO clipping range is set to [0.8, 1.28]. We use separate coefficients for policy regularization and AES shaping: the GRPO KL coefficient is $\beta _ { \mathrm { K L } } \stackrel { - } { = } 0 . 0 0 1$ , while the AES penalty coefficient is $\beta _ { \mathrm { A E S } } = 2 . 0$ unless otherwise specified. The entropy quantile threshold is set to $q = 0 . 9$ , which selects the top 10% highest-entropy tokens within each trajectory. We use public benchmark evaluation sets and benchmark-specific exact-match or rule-based answer checking after extracting the final answer from each response. All baselines are re-evaluated under the same decoding and evaluation protocol. All training experiments are conducted on 8 NVIDIA H100 GPUs. For inference, we use temperature 0.6, top- $\cdot p = 0 . 9 5$ . Each benchmark problem is evaluated with 16 independent sampled responses, without majority voting or best-of-N selection. We report: (1) Accuracy: average pass@1 percentage of problems solved correctly across all samples; (2) Token Count: average number of tokens in the full generated response, including the reasoning trace and final answer.

Existing Assets and Licenses. The third-party models, datasets, benchmarks, and software libraries used in this work are credited in the paper and used under their respective licenses or terms of use, including Apache-2.0, MIT, and CC-BY licenses where applicable. We do not redistribute third-party assets; users should obtain them from their official sources and comply with the corresponding license terms.

## E Entropy of Longer versus Shortest Correct Trajectories

To check whether the entropy signal used by AES is associated with the trajectories it targets, we compare, within each activated sampling group, every longer correct trajectory with the shortest correct trajectory of the same group.

Table 5: Entropy comparison between longer correct trajectories and the shortest correct trajectory in the same activated sampling group $( \mathbf { R } 1 \mathbf { \bar { - } } 7 \mathbf { B } , G = 1 6 )$ . Each entry is the mean paired difference (longer − shortest) of the per-trajectory mean token entropy or 90th-percentile token entropy.
<table><tr><td>Dataset</td><td>Mean entropy diff.</td><td>90th-pct. entropy diff.</td></tr><tr><td>AIME25</td><td>+0.066</td><td>+0.160</td></tr><tr><td>AIME24</td><td>+0.031</td><td>+0.092</td></tr><tr><td>AMC</td><td>+0.056</td><td>+0.133</td></tr></table>

Table 5 reports the mean paired difference. On all three datasets the differences are positive for both the mean token entropy and the 90th-percentile token entropy, indicating that longer correct trajectories carry a heavier high-entropy tail than the efficient in-group reference.

## F Derivation of the Entropy-Length Relationship

We provide a derivation of Proposition 3.1 under a local branching model.

Model setup. At generation step $t ,$ let $B _ { t } = \exp ( H _ { t } )$ denote the effective branching factor. We assume that the model initiates a verification/backtracking branch with probability

$$
q _ { t } = 1 - \exp ( - \alpha B _ { t } ) ,\tag{16}
$$

where $\alpha > 0$ . This parameterization is increasing in $B _ { t }$ and saturates below one.

Residual length. Under a local homogeneous approximation, $q _ { t }$ is treated as approximately constant over the immediate continuation. The number of verification/backtracking branches then follows a geometric process with expected count

$$
\mathbb { E } [ \# \mathrm { b r a n c h e s } ] = \frac { q _ { t } } { 1 - q _ { t } } .\tag{17}
$$

If $\ell _ { 0 } > 0$ denotes the baseline cost to terminate and $c > 0$ denotes the expected token cost of each branch, the expected residual length is

$$
L _ { t } \approx \ell _ { 0 } + c { \frac { q _ { t } } { 1 - q _ { t } } } .\tag{18}
$$

Bounding the branching term. Substituting $q _ { t } = 1 - \exp ( - \alpha B _ { t } )$ gives

$$
\frac { q _ { t } } { 1 - q _ { t } } = \frac { 1 - e ^ { - \alpha B _ { t } } } { e ^ { - \alpha B _ { t } } } = e ^ { \alpha B _ { t } } - 1 .\tag{19}
$$

Let $x = \alpha B _ { t }$ . In the moderate-entropy regime $x \leq 1$ , we have

$$
x \leq e ^ { x } - 1 \leq 2 x .\tag{20}
$$

Therefore,

$$
\alpha B _ { t } \leq \frac { q _ { t } } { 1 - q _ { t } } \leq 2 \alpha B _ { t } .\tag{21}
$$

Substituting this into (18) yields

$$
\ell _ { 0 } + c \alpha B _ { t } \le L _ { t } \le \ell _ { 0 } + 2 c \alpha B _ { t } .\tag{22}
$$

Since $B _ { t } = \exp ( H _ { t } ) \geq 1$ , the additive term can be absorbed into the upper constant:

$$
L _ { t } \leq ( \ell _ { 0 } + 2 c \alpha ) B _ { t } .\tag{23}
$$

The lower bound is already linear in $B _ { t }$ . Thus, for suitable constants $c _ { 1 } , c _ { 2 } > 0 .$

$$
c _ { 1 } B _ { t } \leq L _ { t } \leq c _ { 2 } B _ { t } ,\tag{24}
$$

which gives $L _ { t } = \Theta ( B _ { t } ) = \Theta ( \exp ( H _ { t } ) )$ in the local moderate-entropy regime.

Discussion of assumptions. The homogeneous approximation assumes that the branching probability remains approximately stable over the immediate continuation after state $s _ { t } .$ This approximation is most appropriate for a local analysis around a high-entropy state. It is reasonable when:

• the model operates in a stable regime of uncertainty, so entropy does not fluctuate sharply across nearby continuation steps;

• the analysis focuses on the local effect of a high-entropy transition point.

This derivation is intended as an explanation of why high-entropy states can indicate larger future reasoning cost. More general time-varying branching processes may yield different functional forms, so the empirical truncation and ablation results remain the primary evidence for AES.