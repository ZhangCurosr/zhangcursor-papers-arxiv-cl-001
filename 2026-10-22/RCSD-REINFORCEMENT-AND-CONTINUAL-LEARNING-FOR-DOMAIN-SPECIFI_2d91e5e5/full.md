# RCSD: REINFORCEMENT AND CONTINUAL LEARNING FOR DOMAIN-SPECIFIC SPECULATIVE DECODING

Mingkang Liu (liumingkang@hisense.com), Weiming Meng, PeiPei Jiao, Jiawei Zhang, Tianyuan Liu, Junyan Li, Lichen Pang (cliff.pan@hisense.com) <sup>∗</sup>,

## ABSTRACT

Speculative decoding accelerates large language model inference by employing a compact draft model to propose candidate tokens that are verified in parallel by a target model. Conventional training relies on soft-label distillation, which forces the draft model to mimic the target’s full output distribution—including high-entropy regions that degrade acceptance rates and increase early rejection risk. We propose RCSD (Reinforcement and Continual Learning for Speculative Decoding), a training framework that aligns draft model optimization with the system-level objective of maximizing acceptance length. RCSD leverages naturally occurring accepted and rejected sequences during speculative inference to construct implicit preference pairs, with masked loss to avoid penalizing target-aligned tokens. Evaluated across multiple benchmarks, RCSD improves average acceptance length by 35% over HASS and achieves 1.35× faster end-to-end inference than HASS on practical workloads, without compromising generation quality.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities across a wide range of tasks. However, their autoregressive nature leads to high computational costs and significant latency during inference, which hinders their deployment in real-time applications. To address this bottleneck, speculative decoding has emerged as a prominent acceleration paradigm[4, 2]. This approach leverages a fast draft model to generate multiple candidate tokens specula tively, which are then verified in parallel by the larger, more accurate target model. By accepting or rejecting these candidates in bulk, speculative decoding can substantially improve the overall throughput of the system.

The prevailing training strategy for draft models relies heavily on soft-label distillation[5], where the draft model is supervised to mimic the full output probability distribution of the target model. This method implicitly minimizes the forward Kullback-Leibler (KL) divergence, compelling the draft model to cover all modes of the teacher’s distribution, including high-entropy regions with low-probability tokens. While it is theoretically true that minimizing the KL divergence to zero would result in the draft model perfectly replicating the target model and thus achieving the maximum possible acceptance length, this goal is practically unattainable. The draft model, being orders of magnitude smaller than the target, lacks the capacity to faithfully reproduce the complex distribution of its teacher. However, exact probability replication is not requisite. To maximize acceptance length, it suffices for the relative ordering of the top-k most probable tokens in the draft model to align with that of the base model. Consequently, a fundamental misalignment arises: the training objective (distribution matching) diverges from the ultimate system-level goal (maximizing the number of accepted tokens). For instance Figure 1, a draft model might accurately capture the relative ranking of the top-k tokens but fail to precisely replicate their exact probabilities, leading to a perfect acceptance rate despite a large distillation loss.

This misalignment motivates our core insight: the draft model should transcend the role of a faithful imitator to become an efficiency-oriented predictor. Specifically, we advocate for a ‘mode-seeking’ behavior rather than ‘mode-covering’, thereby concentrating predictive capacity on sequences with high acceptance probability, even if this necessitates disregarding low-probability tails of the target distribution. Ultimately, the optimization goal is redefined: rather than minimizing the divergence of logits, the system aims to maximize the expected acceptance length.

![](images/8131534450928f77ac327fae118439cac66bf4bfb5db4d93ed2712e758f36873.jpg)

![](images/5680f6ca0f29ba2f5c23530d5334594ca94f7223b8a6faf9d102ede9e361c047.jpg)

![](images/594b2c2169fea37db50e3bda3bcd4f8898c09c36f3370b889849953f4ed37485.jpg)

![](images/4fbe30d35cf99893b18483ed45333ad1913f7971909cfc197a93f83a3a70b9c3.jpg)

![](images/ec8aad6211fbd517018806c2f1ebdb69c9e5dc528df595fb00b249ed82811d74.jpg)  
Figure 1: Misalignment between KL divergence and acceptance rate. Due to limited parameters, the draft model cannot perfectly replicate the target distribution. However, low KL loss may still lead to token rejection. This discrepancy arises because speculative decoding acceptance depends on the relative ranking of top candidate tokens rather than exact distribution matching.

To bridge this gap, we propose RCSD (Reinforcement & Continual Learning for Speculative Decoding), a novel training framework that directly aligns the draft model’s optimization with the system’s performance metric. RCSD capitalizes on the intrinsic feedback loop of speculative decoding, where every verification step naturally produces pairs of accepted and rejected token sequences. We leverage these implicit preference signals to train the draft model using reinforcement learning principles. Specifically, we adapt two powerful algorithms: Direct Preference Optimization (DPO)[7] and Group-wise Relative Policy Optimization (GRPO)[8]. In our DPO formulation, a sequence sampled purely from the target model serves as the “chosen” response, while a sequence generated through periodic alternation between the target and draft models (every k steps) acts as the “rejected” response. Our GRPO approach constructs groups of sequences with varying drafting strategies (different k-step intervals) and uses the acceptance rate over these sequences as a direct and verifiable reward signal to optimize the expected acceptance length.

Our experiments on both Qwen3-1.7B and Llama-3.1-8B demonstrate the efficacy of RCSD. It achieves an average improvement of 35% in acceptance length over conventional distillation baselines(HASS) [10] and delivers a 1.35× speedup on practical workloads like ShareGPT, all without compromising the quality of the final output. To our knowledge, this work is the first to systematically integrate preference optimization techniques like DPO and GRPO into the training pipeline for speculative decoding, establishing a new paradigm for building highly efficient draft models.

## 2 Related Work

## 2.1 Speculative Decoding for Autoregressive Inference

Speculative decoding has emerged as an effective paradigm for accelerating autoregressive inference in large language models by decoupling token proposal and verification. The foundational work on speculative decoding introduces a draft–target–verification framework, where a lightweight draft model generates candidate tokens that are subsequently verified in parallel by a larger target model, enabling substantial throughput improvements when multiple tokens are accepted at once [4].

Subsequent works have focused on improving the efficiency and scalability of speculative decoding through architectural and algorithmic innovations. Medusa proposes multi-head token prediction to increase the number of speculative candidates per verification step [2]. Other approaches explore lookahead decoding and multi-token prediction strategies to further amortize verification costs [6]. More recently, EAGLE introduces hidden-state guidance to enhance draft token quality and improve acceptance rates [5]. While these methods significantly advance speculative inference at the decoding level, they generally assume a pre-trained draft model and devote limited attention to the alignment between the draft model’s training objective and the system-level acceptance efficiency.

Speculative Decoding Performance  
![](images/984a5e3ec7e52559597a63db4bcc0c674ec0249577ce491238d442f0f1390757.jpg)  
Figure 2: Output Throughput (TPS) and Mean Accept Length across model configurations. Evaluated on Llama-3.1- 8B-Instruct with SGLang 5.5.0, the results demonstrate that SFT+DPO and SFT+GRPO consistently outperform the baseline SFT approach on both metrics.

## 2.2 Draft Model Training via Knowledge Distillation

The dominant approach to training draft models relies on knowledge distillation, where the draft model is supervised to mimic the output distribution of the target model. Classic distillation methods minimize the forward Kullback–Leibler divergence between the draft and target distributions using soft labels [3]. This paradigm has been widely adopted in speculative decoding frameworks due to its simplicity and stability [4, 5].

However, minimizing forward KL divergence encourages mode-covering behavior, forcing the draft model to allocate capacity to low-probability regions of the target distribution. For speculative decoding, this objective is misaligned with the system goal of maximizing the expected acceptance length, which primarily depends on accurately proposing high probability token sequences. As discussed in prior work on divergence optimization, KL-based objectives fundamentally favor coverage over decision optimality [1]. Various heuristics, such as temperature scaling and truncated distillation, have been explored to mitigate this issue, but the core training objective remains distribution matching rather than acceptance-oriented optimization.

![](images/0181570f77e56b819a33bbc7d527ba918fdf468bbb8dcccc60e8cbed4fec2eb3.jpg)  
Figure 3: Overview of the RCSD framework. Given requests and dataset samples, a draft model $\pi _ { \theta }$ first proposes k tokens via speculative decoding, and a target model $\pi _ { T }$ performs parallel verification. The verification stage partitions outputs into accepted tokens (forming the chosen sequence $y ^ { + }$ ) and early-rejected tokens (forming the rejected sequence $y ^ { - } )$ , from which preference or reward signals are constructed. The resulting pairs $( x , y ^ { + } , y ^ { - } )$ or response groups G are then used to update $\pi _ { \theta }$ via DPO- or GRPO-based optimization. In DPO-based optimization, target-model preferences between $y ^ { + }$ and $y ^ { - }$ guide parameter updates. In GRPO-based optimization, grouped responses are optimized using an acceptance-rate reward.

## 2.3 Reinforcement Learning Methods

Reinforcement learning has been extensively applied to language model alignment, most notably through RLHF pipelines that optimize policies using human feedback and KL-regularized objectives [11]. While effective, these methods require explicit reward modeling and incur significant training complexity.

Direct Preference Optimization (DPO) offers a simpler alternative by directly optimizing policies using pairwise preference data, eliminating the need for an explicit reward model [7]. DPO and related preference-based methods have demonstrated strong performance in instruction tuning and alignment tasks. In parallel, group-wise and relative policy optimization techniques, such as GRPO-style clipping and ranking-based learning, leverage relative comparisons across multiple candidates to stabilize learning and improve sample efficiency [8].

Existing applications of preference optimization primarily focus on improving response quality or alignment with human preferences. In contrast, our work applies both pairwise and group-wise preference optimization directly to speculative decoding, where preferences and rewards are automatically derived from the target model and explicitly aligned with inference-time acceptance efficiency.

## 2.4 Reinforcement Learning in Speculative Decoding

Recent studies have explored integrating reinforcement learning (RL) into speculative inference systems. To date, these efforts have primarily focused on optimizing inference-time hyperparameters via RL, such as sampling tree depth, step size, and verification length [9]. While these methods demonstrate promising results in single-request settings, they encounter limitations in concurrent scenarios. Due to engineering constraints, requests within the same batch typically require uniform speculative decoding hyperparameters, restricting the ability to adapt parameters based on specific input characteristics. Consequently, per-instance hyperparameter tuning yields diminishing returns in batched inference, where simple adjustments based on batch size often suffice. Furthermore, hyperparameter in inference and prediction accuracy constitute two distinct determinants of inference speed within speculative decoding systems. Irrespective of the chosen inference hyperparameters, enhancing prediction accuracy invariably leads to improved inference throughput. Motivated by this observation, this work employs reinforcement learning to directly optimize the prediction accuracy of the draft model itself. Therefore, our approach is orthogonal to and independent of the aforementioned inference-time hyperparameter optimization methods. Building on the preference optimization paradigms discussed above, we treat the target model as an implicit reward mechanism that simultaneously generates reference outputs, verifies speculative tokens, and provides preference signals. Unlike prior self-feedback methods that focus on output quality or safety, we leverage this intrinsic model feedback to optimize the acceptance length in speculative decoding fig. 3. This approach eliminates the need for additional external reward models or human annotations, establishing a direct link between the training objective and the system-level performance metric.

## 3 Methodology

In this section, we present two complementary optimization strategies for training the draft model in the Eagle speculative inference framework. Both approaches leverage the target model $\mathcal { M } _ { \mathrm { t a r g e t } }$ as an implicit reward signal to align the draft policy $\pi _ { \theta }$ with the target model’s generation behavior.

We first introduce a pairwise preference-based optimization method based on Direct Preference Optimization (DPO). This approach constructs binary preference signals by contrasting target-model generations with speculative generations produced by the draft model. We then extend this formulation to a group-wise relative optimization strategy, referred to as Group-wise Relative Policy Optimization (GRPO), which exploits relative quality signals across multiple speculative generation paths.

Together, these methods enable efficient training of draft models that maximize token verification acceptance rates during speculative inference.

## 3.1 Direct Preference Optimization for Eagle Training

In the Eagle speculative inference framework, we optimize a policy draft-model $\pi _ { \theta }$ via Direct Preference Optimization to align with a target model ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ . The DPO dataset $\mathcal { D } _ { \mathrm { d p o } }$ consists of pairs $( \mathbf { y } ^ { + } , \mathbf { y } ^ { - } )$ where:

• Chosen sequences $\mathbf { y } ^ { + }$ are sampled directly from ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ using autoregressive decoding

• Rejected sequences $\mathbf { y } ^ { - }$ are generated through periodic alternation: ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ generates 1 token, followed by $\mathcal { M } _ { \mathrm { d r a f t } }$ generating blocks of size k without verification

This creates an implicit preference $\mathbf { y } ^ { + } \succ \mathbf { y } ^ { - }$ . Since the draft-model’s input structure differs from the target model’s, the sequence representation y incorporates not only tokens but also hidden states from ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$

## 3.1.1 DPO Objective

Let $\pi _ { \mathrm { r e f } }$ denote a frozen reference draft-model (initialized from $\pi _ { \boldsymbol { \theta } } \mathbf { \dot { \theta } }$ s pretrained weights). For sequence y given prompt $\mathbf { x } ,$ define the log-ratio:

$$
r _ { \theta } ( \mathbf { x } , \mathbf { y } ) = \log \pi _ { \theta } ( \mathbf { y } _ { > | \mathbf { x } | } \mid \mathbf { x } ) - \log \pi _ { \mathrm { r e f } } ( \mathbf { y } _ { > | \mathbf { x } | } \mid \mathbf { x } )\tag{1}
$$

where $\mathbf { y } _ { > | \mathbf { x } | }$ denotes tokens after the prompt. The DPO loss minimizes:

$$
\mathcal { L } _ { D P O } ( \boldsymbol { \theta } ) = - E _ { ( \mathbf { x } , \mathbf { y } ^ { + } , \mathbf { y } ^ { - } ) \sim \mathcal { D } _ { \mathrm { d p o } } } \left[ \log \sigma \left( \beta \left( r _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { y } ^ { + } ) - r _ { \boldsymbol { \theta } } ( \mathbf { x } , \mathbf { y } ^ { - } ) \right) \right) \right]\tag{2}
$$

with $\beta > 0$ controlling optimization strength and $\sigma ( \cdot )$ the sigmoid function.

## 3.1.2 Training Protocol

Algorithm 1 outlines the optimization procedure:

1. Sequence Generation:

• $\mathbf { y } ^ { + }$ is sampled from $\mathcal { M } _ { \mathrm { t a r g e t } }$ via standard autoregressive decoding

$\mathbf { y } ^ { - }$ is generated through periodic alternation: for each position $t ,$

$$
y _ { t } = { \left\{ \begin{array} { l l } { \operatorname { s a m p l e } \operatorname { f r o m } \mathcal { M } _ { \mathrm { d r a f t } } ( \cdot \mid \mathbf { y } _ { < t } ) } & { { \mathrm { i f } } \ t \mod ( k + 1 ) \neq 0 } \\ { \operatorname { s a m p l e } \operatorname { f r o m } \mathcal { M } _ { \mathrm { t a r g e t } } ( \cdot \mid \mathbf { y } _ { < t } ) } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }
$$

Algorithm 1 DPO Training for Speculative Inference   
Require: Policy model $\pi _ { \theta } .$ , frozen reference model $\pi _ { \mathrm { r e f } } ,$ preference dataset $\mathcal { D } _ { \mathrm { d p o } }$ , temperature $\beta$   
Ensure: Optimized policy parameters θ   
1: Initialize θ from pretrained weights   
2: while training not converged do   
3: Sample batch $\mathcal { B } = \{ ( \mathbf { \bar { x } } ^ { ( i ) } , \mathbf { y } ^ { + } ^ { ( i ) } , \mathbf { y } ^ { - } ^ { ( i ) } ) \} _ { i = 1 } ^ { B } \sim \mathcal { D } _ { \mathrm { d p o } }$   
4: Compute log-probabilities for all sequences:   
$\begin{array} { r } { \log p _ { { \mathbf { y } } ^ { + } } ^ { \bar { \pi } _ { \theta } }  \bar { \sum _ { t > | \mathbf { x } | } } \log \pi _ { \theta } ( y _ { t } \mid \mathbf { x } _ { < t } ) } \end{array}$ ▷ Policy on chosen   
log $\begin{array} { r } { p _ { \mathbf { y } ^ { - } } ^ { \mathbf { \check { \pi } } _ { \theta } } \gets \sum _ { t > | \mathbf { x } | } \log \pi _ { \theta } ( z _ { t } \mid \mathbf { x } _ { < t } ) } \end{array}$ ▷ Policy on rejected   
log $\begin{array} { r } { p _ { \mathbf { y } ^ { + } } ^ { \pi _ { \mathrm { r e f } } }  \sum _ { t > | \mathbf { x } | } \log \pi _ { \mathrm { r e f } } ( y _ { t } \mid \mathbf { x } _ { < t } ) } \end{array}$ ▷ Reference on chosen   
log $\begin{array} { r } { p _ { \mathbf { y } ^ { - } } ^ { \pi _ { \mathrm { r e f } } }  \sum _ { t > | \mathbf { x } | } \log \pi _ { \mathrm { r e f } } ( z _ { t } \mid \mathbf { x } _ { < t } ) } \end{array}$ ▷ Reference on rejected   
5: Calculate preference ratios:   
$\Delta ^ { \pi _ { \theta } }  \log ^ { \overline { { \alpha } } _ { \theta } } - \log p _ { \mathbf { y } ^ { - } } ^ { \pi _ { \theta } }$   
$\Delta ^ { \pi _ { \mathrm { r e f } } }  \mathrm { d e t a c h } ( \log p _ { \mathbf { y } ^ { + } } ^ { \pi _ { \mathrm { r e f } } } - \log p _ { \mathbf { y } ^ { - } } ^ { \pi _ { \mathrm { r e f } } } )$   
6: Compute loss: $\begin{array} { r } { \dot { \mathcal { L } }  \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log \sigma ( \beta ( \Delta _ { i } ^ { \pi _ { \theta } } - \Delta _ { i } ^ { \pi _ { \mathrm { r e f } } } ) ) } \end{array}$   
7: Update $\theta \gets \theta - \eta \nabla _ { \theta } \dot { \mathcal { L } }$   
8: end while

2. Log-Probability Calculation: Sequence log-probabilities are computed exclusively over post-prompt tokens $\mathbf { \rho } ( t > | \mathbf { x } | )$ with next-token prediction alignment. Additionally, tokens in rejected sequences generated by ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ are masked to reduce penalty on tokens already aligned with ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ . Given the length sensitivity of DPO, identical masking patterns (for both prompt exclusion and $\mathcal { M } _ { \mathrm { t a r g e t } ^ { - } } \mathrm { g e }$ nerated tokens) are consistently applied across chosen and rejected sequences during probability calculations.

## 3.2 Group-wise Relative Policy Optimization for Eagle Training

In the Eagle speculative inference framework, we optimize a policy draft-model $\pi _ { \theta }$ via Group-wise Relative Policy Optimization to align with a target model ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ . The GRPO dataset $\mathcal { D } _ { \mathrm { g r p o } }$ consists of tuples $( \mathbf { x } , \{ \mathbf { y } ^ { ( g ) } \} _ { g \in \mathcal { G } } )$ where:

• x is the input prompt

• $\mathcal { G }$ denotes the set of sequence groups $( { \bf e . g . } , { \mathcal G } = \cdot$ {chosen, spec\_step1, spec\_step2, spec\_step3})

• Chosen group $\mathbf { y } ^ { ( \mathrm { c h o s e n } ) }$ is sampled directly from ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ using autoregressive decoding

• Speculative groups $\mathbf { y } ^ { ( \mathrm { s p e c \_ s t e p } k ) }$ are generated through periodic alternation: ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ generates 1 token, followed by $\mathcal { M } _ { \mathrm { d r a f t } }$ generating blocks of size k without verification

Sequence quality scores $s _ { g }$ are computed by ${ \mathcal { M } } _ { \mathrm { t a r g e t } }$ for each group $g .$ Unlike binary preferences, GRPO leverages relative quality signals across multiple generation paths to optimize the draft policy.

## 3.2.1 GRPO Objective

Let $\pi _ { \mathrm { r e f } }$ denote a frozen reference draft-model (initialized from $\pi \theta ^ { \mathbf { \hat { s } } }$ pretrained weights). For sequence $\mathbf { y } ^ { ( g ) }$ given prompt x, define the probability ratio:

$$
r _ { \theta } ( \mathbf { x } , \mathbf { y } ^ { ( g ) } ) = \frac { \pi _ { \theta } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } ) } { \pi _ { \mathrm { r e f } } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } ) }\tag{3}
$$

where $\mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) }$ denotes tokens after the prompt. The normalized reward for group $g$ is:

$$
s _ { g } = \mathcal { N } ( \log \mathcal { M } _ { \mathrm { t a r g e t } } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } ) ) + \mathcal { N } ( A c c _ { > | \mathbf { x } | } ^ { ( g ) } )\tag{4}
$$

with $\mathcal { N } ( \cdot )$ denotes Z-score normalization. The GRPO loss minimizes:

$$
\mathcal { L } _ { G R P O } ( \theta ) = \frac { 1 } { \left| \mathcal { G } \right| } \sum _ { g \in \mathcal { G } } E _ { \left( \mathbf { x } , \mathbf { y } ^ { \left( g \right) } \right) \sim \mathcal { D } _ { \mathbf { y p o } } } \left[ - \operatorname* { m i n } \left( r _ { \theta } ( \mathbf { x } , \mathbf { y } ^ { \left( g \right) } ) \hat { s } _ { g } , \mathrm { c l i p } _ { 1 \pm \delta } \left( r _ { \theta } ( \mathbf { x } , \mathbf { y } ^ { \left( g \right) } ) \right) \hat { s } _ { g } \right) + \beta \cdot \mathbf { K L } _ { g } \right]\tag{5}
$$

Algorithm 2 GRPO Training for Speculative Inference   
Require: Policy $\pi _ { \theta } ,$ reference $\pi _ { \mathrm { r e f } } .$ , dataset $\mathcal { D } _ { \mathrm { g r p o } } , \beta , \delta$   
Ensure: Optimized θ   
1: Initialize θ from pretrained weights   
2: while not converged do   
3: Sample $\mathcal B = \{ ( \mathbf x ^ { ( i ) } , \{ \mathbf y ^ { ( g ) } ^ { ( i ) } \} _ { g \in \mathcal G } ) \} _ { i = 1 } ^ { B } \sim \mathcal D _ { \mathrm { g r p o } }$   
4: Compute normalized rewards $\forall g , i \colon$   
$s _ { g } ^ { ( i ) } \gets \mathcal { N } ( \log \mathcal { M } _ { \mathrm { t a r g e t } } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } ) ) + \mathcal { N } ( A c c _ { > | \mathbf { x } | } ^ { ( g ) } )$   
5: $\mathcal { L }  0$   
6: for each group $g \in { \mathcal { G } }$ do   
7: $\Delta _ { g } \gets \log \pi _ { \theta } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } ) - \log \pi _ { \mathrm { r e f } } ( \mathbf { y } _ { > | \mathbf { x } | } ^ { ( g ) } \mid \mathbf { x } )$ ▷ Log-ratio   
8: $r _ { g } \gets \exp ( \Delta _ { g } ) , \quad \dot { \tilde { r } } _ { g } \gets \mathrm { c l i p } ( r _ { g } , 1 \pm \delta )$   
9: $\mathrm { K L } _ { g }  \mathrm { K L } [ \pi _ { \theta } \| \pi _ { \mathrm { r e f } } ] _ { \mathbf { y } ^ { ( g ) } }$ ▷ Per-token KL   
10: $\mathcal { L } _ { g } \gets - E _ { i } \left[ \operatorname* { m i n } ( r _ { g } ^ { ( i ) } s _ { g } ^ { ( i ) } , \tilde { r } _ { g } ^ { ( i ) } s _ { g } ^ { ( i ) } ) \right] + \beta \cdot \mathbf { K } \mathbf { L } _ { g }$   
11: $\mathcal { L }  \mathcal { L } + \mathcal { L } _ { g } ^ {  } / | \mathcal { G } |$   
12: end for   
13: Update $\theta \gets \theta - \eta \nabla _ { \theta } \mathcal { L }$   
14: end while

## 3.2.2 Training Protocol

Algorithm 2 outlines the optimization procedure:

## 1. Group Configuration:

• Quality scores $s _ { g }$ are computed exclusively over post-prompt tokens $\mathbf { \rho } ( t > | \mathbf { x } | )$ using $\mathcal { M } _ { \mathrm { t a r g e t } } \mathrm { ^ { \circ } s }$ logprobabilities

## 2. Sequence Processing:

• Log-probabilities for $\pi _ { \theta }$ and $\pi _ { \mathrm { r e f } }$ are computed with next-token prediction alignment (hidden states $\mathbf { h } _ { t }$ predict token $y _ { t + 1 } )$

• Target-generated tokens in speculative sequences are masked during log-probability calculation to avoid over-penalizing aligned segments

• Identical masking patterns are applied consistently across all groups for fair reward comparison

## 4 Experiments

## 4.1 Setup

Models and datasets. We evaluate our methods on two model families: Qwen3-1.7B and Llama-3.1-8B, trained on 90% of ShareGPT. Evaluation uses GSM8K, HumanEval, MT-Bench, and the remaining 10% of ShareGPT to cover both reasoning and open-ended instruction-following tasks.

Training variants. We compare: (1) HASS (baseline draft model), (2) HASS + DPO , (3) HASS + GRPO . Results are reported for speculative steps 3–6, i.e., drafting 3–6 candidate tokens per verification round, to examine performance across decoding configurations.

## 4.2 Main Results

To provide a holistic comparison, we compute the average acceptance length across the four evaluation datasets at each speculative step.

Qwen3-1.7B. Table 1 shows the results for Qwen3-1.7B. RL significantly improves over the baseline across all datasets and speculative steps. Averaged over the four datasets, acceptance length at step 6 reaches 2.26, 2.97, and 3.00 for HASS, HASS+DPO, and HASS+GRPO, respectively. GRPO generally achieves the best performance, with consistent gains in GSM8K and MT-Bench.

<table><tr><td>Dataset</td><td>Model</td><td>Step 3</td><td>Step 4</td><td>Step 5</td><td>Step 6</td></tr><tr><td rowspan="3">HumanEval</td><td>HASS</td><td>2.322</td><td>2.446</td><td>2.510</td><td>2.533</td></tr><tr><td>HASS + DPO</td><td>2.616</td><td>2.780</td><td>2.922</td><td>3.014</td></tr><tr><td>HASS + GRPO</td><td>2.629</td><td>2.807</td><td>2.928</td><td>3.015</td></tr><tr><td rowspan="3">MT-Bench</td><td>HASS</td><td>1.986</td><td>2.047</td><td>2.072</td><td>2.092</td></tr><tr><td> $\mathrm { H A S S + D P O }$ </td><td>2.333</td><td>2.442</td><td>2.537</td><td>2.571</td></tr><tr><td> $_ \mathrm { H A S S + G R P O }$ </td><td>2.346</td><td>2.467</td><td>2.549</td><td>2.601</td></tr><tr><td rowspan="3">GSM8K</td><td>HASS</td><td>2.102</td><td>2.155</td><td>2.176</td><td>2.176</td></tr><tr><td> $\mathrm { H A S S + D P O }$ </td><td>2.746</td><td>3.001</td><td>3.121</td><td>3.212</td></tr><tr><td> $_ \mathrm { H A S S + G R P O }$ </td><td>2.754</td><td>3.002</td><td>3.161</td><td>3.246</td></tr><tr><td rowspan="3">ShareGPT</td><td>HASS</td><td>2.050</td><td>2.166</td><td>2.180</td><td>2.220</td></tr><tr><td> $\mathrm { H A S S + D P O }$ </td><td>2.581</td><td>2.880</td><td>3.042</td><td>3.089</td></tr><tr><td>HASS + GRPO</td><td>2.563</td><td>2.882</td><td>3.051</td><td>3.136</td></tr></table>

Table 1: Qwen3-1.7B: results at Steps 3–6. Baseline HASS, HASS+DPO, and HASS+GRPO are compared. Higher values indicate better acceptance length. HASS+GRPO generally achieves better performance.

<table><tr><td>Dataset</td><td>Model</td><td>Step 3</td><td>Step 4</td><td>Step 5</td><td>Step 6</td></tr><tr><td rowspan="3">HumanEval</td><td>HASS</td><td>2.731</td><td>2.950</td><td>3.103</td><td>3.202</td></tr><tr><td> $\mathrm { H A S S + D P O }$ </td><td>2.896</td><td>3.185</td><td>3.391</td><td>3.540</td></tr><tr><td>HASS + GRPO</td><td>2.897</td><td>3.189</td><td>3.402</td><td>3.558</td></tr><tr><td rowspan="3">MT-Bench</td><td>HASS</td><td>2.282</td><td>2.397</td><td>2.438</td><td>2.497</td></tr><tr><td> $\mathrm { H A S S + D P O }$ </td><td>2.420</td><td>2.595</td><td>2.674</td><td>2.731</td></tr><tr><td>HASS + GRPO</td><td>2.426</td><td>2.596</td><td>2.692</td><td>2.736</td></tr><tr><td rowspan="3">GSM8K</td><td>HASS</td><td>2.487</td><td>2.640</td><td>2.712</td><td>2.756</td></tr><tr><td>HASS + DPO</td><td>2.648</td><td>2.836</td><td>2.945</td><td>3.008</td></tr><tr><td> $_ \mathrm { H A S S + G R P O }$ </td><td>2.654</td><td>2.864</td><td>2.958</td><td>3.025</td></tr><tr><td rowspan="3">ShareGPT</td><td>HASS</td><td>2.384</td><td>2.524</td><td>2.639</td><td>2.672</td></tr><tr><td>HASS + DPO</td><td>2.928</td><td>3.283</td><td>3.571</td><td>3.779</td></tr><tr><td> $_ \mathrm { H A S S + G R P O }$ </td><td>2.964</td><td>3.310</td><td>3.588</td><td>3.828</td></tr></table>

Table 2: Llama-3.1-8B: results at Steps 3–6. HASS+GRPO generally achieves better performance.

Llama-3.1-8B. Table 2 shows similar trends for Llama-3.1-8B. Across most datasets and steps, GRPO achieves the highest or tied-best performance, with particularly strong improvements on ShareGPT. These results indicate that preference optimization substantially improves acceptance length, while GRPO consistently provides the best overall performance.

## 4.3 Analysis

Across both model families and all speculative steps, preference-based optimization consistently improves acceptance length over the HASS distillation baseline. GRPO tends to outperform DPO, highlighting the advantage of leveraging multiple speculative paths. Moreover, the relative gain grows as more tokens are drafted per step (e.g., on Qwen3-1.7B + GSM8K, from 31% at step 3 to 49% at step 6), suggesting that better draft quality compounds over longer speculation horizons.

## 5 Conclusion

We propose RCSD (Reinforcement and Continual Learning for Speculative Decoding), a training framework that fundamentally reorients draft model optimization from distribution matching to acceptance-length maximization—the true system-level objective of speculative decoding. By leveraging the intrinsic feedback loop inherent in speculative inference, RCSD constructs implicit preference signals from naturally occurring accepted and rejected token sequences without requiring external reward models or human annotations. Through two complementary formulations—pairwise preference optimization via DPO and group-wise relative policy optimization via GRPO—RCSD trains draft models to adopt mode-seeking behavior that prioritizes high-probability token sequences aligned with the target model’s verification decisions. Crucially, our masked loss mechanism ensures training signal purity by excluding targetgenerated tokens from gradient updates.

Extensive experiments on Qwen3-1.7B and Llama-3.1-8B across diverse benchmarks (ShareGPT, GSM8K, HumanEval, and MT-Bench) demonstrate RCSD’s effectiveness. Our method improves average acceptance length by 35–40% over conventional distillation baselines (e.g., HASS) and achieves 1.2–1.35× end-to-end inference speedup on practical workloads while preserving generation quality. Notably, GRPO consistently accelerates early-stage training convergence by leveraging relative quality signals across multiple speculative paths, highlighting the advantage of group-wise optimization over binary preferences.

## References

[1] M. Arjovsky and L. Bottou. Towards principled methods for training generative adversarial networks, 2017.

[2] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024.

[3] G. Hinton, O. Vinyals, and J. Dean. Distilling the knowledge in a neural network, 2015.

[4] Y. Leviathan, M. Kalman, and Y. Matias. Fast inference from transformers via speculative decoding, 2023.

[5] Y. Li, F. Wei, C. Zhang, and H. Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty, 2025.

[6] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, Z. Zhang, R. Y. Y. Wong, A. Zhu, L. Yang, X. Shi, C. Shi, Z. Chen, D. Arfeen, R. Abhyankar, and Z. Jia. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, Volume 3, ASPLOS ’24, page 932–949. ACM, Apr. 2024.

[7] R. Rafailov, A. Sharma, E. Mitchell, S. Ermon, C. D. Manning, and C. Finn. Direct preference optimization: Your language model is secretly a reward model, 2024.

[8] Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.

[9] J. Zhang, Z. Yu, L. Wang, N. Yang, E. J. Yu, Z. Li, Y. Song, D. Zhu, X. Zhang, F. Wei, and S. Li. Learning to draft: Adaptive speculative decoding with reinforcement learning. In The Fourteenth International Conference on Learning Representations, 2026.

[10] L. Zhang, X. Wang, Y. Huang, and R. Xu. Learning harmonized representations for speculative sampling. In International Conference on Learning Representations, 2025.

[11] D. M. Ziegler, N. Stiennon, J. Wu, T. B. Brown, A. Radford, D. Amodei, P. Christiano, and G. Irving. Fine-tuning language models from human preferences, 2020.