# Reinforcement Learning Done Right for Explicit Confidence Targets

Zefeng He<sup>1,∗</sup>, Yichen He<sup>1</sup>, Yuan Lin<sup>1,†</sup>, Yang Yang<sup>2,†</sup>, Yu Cheng<sup>3</sup>

<sup>1</sup>ByteDance Seed, <sup>2</sup>Shanghai Jiao Tong University, <sup>3</sup>NTU

<sup>∗</sup>Work done during an internship at Bytedance Seed, <sup>†</sup>Corresponding authors

## Abstract

Large language models (LLMs) can produce plausible but incorrect answers when they are uncertain, which can mislead users and undermine model reliability. Training models to abstain with reinforcement learning is a natural approach, but most existing methods use a fixed error penalty and therefore cannot adapt their behavior to diferent confidence requirements at inference time. We study explicit confidence targets, where a user specifies a target p in the prompt and the model directly chooses whether to answer or abstain, with the goal of achieving an accuracy of at least p among answered questions. However, learning to respond to diferent confidence targets with a single model has proven dificult, and prior attempts have failed to distinguish diferent targets. We show that this failure comes from two problems: the reward signal that distinguishes diferent values of p is often lost during optimization, while the original error penalty grows without bound as p approaches one. We propose RLEC (Reinforcement Learning with Explicit Confidence Targets), which preserves the learning signal needed to distinguish confidence targets, introduces a stable reward family that keeps the desired answer condition unchanged, and uses reward scheduling to balance efective learning and training stability. Together, these designs enable the first successful training of a single LLM for explicit confidence target control. Extensive experiments across diverse benchmarks and model scales show that RLEC largely preserves general capabilities and achieves competitive calibration performance. RLEC also supports fine-grained target control without an external decision system or threshold calibration after training. By making explicit confidence targets part of the trained policy, RLEC opens a new and more direct path toward behavioral calibration.

Date: October 26, 2026 Code: https://github.com/ByteDance-Seed/RLEC

## 1 Introduction

Large language models (LLMs) [1, 6, 29] have shown strong abilities in knowledge-intensive and reasoning tasks, but they can still generate plausible yet incorrect responses, commonly known as hallucinations [13, 17]. Such errors reduce the reliability of LLMs and may mislead users, posing particular risks when incorrect answers are costly. Rather than guessing when uncertain, a reliable model should know when not to answer [18–20].

Kalai et al. [19] argue that common training objectives reward guessing over acknowledging uncertainty. A natural solution is to train abstention with

![](images/a49e06d7cfb5f006bb049176992fecfcf1e504f8fb3a2a85ad15b92e75d477ef.jpg)

![](images/427855b75f09028c7ce1a1b6c0a735dbb1101f79e52451970a9dd65bdcba0056.jpg)  
Figure 1 Explicit Confidence Target control. (a) RLEC’s average accuracy across eight benchmarks increases with p specified in the prompt. (b) RLEC’s fine-grained control on AIME with p.

reinforcement learning [16, 24, 25, 35], rewarding correct answers, penalizing incorrect answers, and assigning no reward to abstentions. For example, TruthRL [35] uses such a reward with a fixed error penalty. However, a fixed penalty represents only one risk preference, while diferent applications may require diferent levels of reliability. To represent these requirements, we study an explicit confidence target: the user specifies a target p in the prompt, and the model directly chooses whether to answer or abstain. The desired behavior is

$$
\operatorname { A c c } ( p ) = \operatorname* { P r } ( { \mathrm { c o r r e c t } } \mid { \mathrm { a n s w e r } } , p ) \geq p ,\tag{1}
$$

and a single model should respond to diferent values of p without retraining or an external decision system. However, learning such behavior has proved dificult. Wu et al. [36] propose Explicit Risk Thresholding (ERT), which trains on randomly sampled values of $p$ using GRPO [10, 27], assigning rewards of 1, $- p / ( 1 - p )$ and 0 to correct answers, incorrect answers, and abstentions, respectively. However, they report that the model’s hallucination rates remain insensitive to the input target. They attribute this failure to the noisy optimization problem created by training across many thresholds. Other work [24] further argues that limited model capacity may force a policy to specialize in one risk preference. These findings leave it unclear whether the dificulty is inherent to a single model or caused by the training method.

We show that this failure does not mean that the task is beyond the capacity of a single model. Instead, it can be explained by two problems in the training setup. First, ERT, like many recent RL methods for LLMs, uses GRPO. We observe that most of its rollout groups contain only one or two distinct reward values. Groups with a single reward value yield zero advantages, while in groups with two values, standard deviation normalization removes the efect of the reward scale induced by p on the advantages. The resulting advantages depend only on the fraction of the two values, rather than their magnitudes. As a result, reward magnitudes provide little direct signal for distinguishing confidence targets. Second, the original error reward grows without bound as p approaches one. Once the reward magnitudes directly afect the advantages, samples with large p can dominate the gradient and destabilize training. Based on these findings, we introduce RLEC (Reinforcement Learning with Explicit Confidence Targets). RLEC removes division by the group standard deviation while retaining mean centering, thereby preserving the efect of the reward scale induced by p on the advantages. It further introduces a reward family controlled by a stability coeficient b, which controls reward magnitudes at large p without changing the desired decision rule. Finally, we observe that fixed values of b tend to trade of the strength of the learning signal against training stability. We therefore introduce a schedule for b that combines a strong signal for learning diferent confidence targets with stable training.

We evaluate RLEC using Qwen3-4B-Base and Qwen3-30B-A3B-Base [37] across eight benchmarks covering mathematical reasoning, knowledge and hallucination, and scientific reasoning. Our results show that a single model can successfully learn diferent behaviors for diferent explicit confidence targets, as shown in Figure 1(a). We also compare RLEC with a standard RLVR [10] baseline, showing that RLEC largely preserves the baseline’s general capabilities at both model scales. For additional context, we further compare RLEC with RLCR [8], which focuses on confidence reporting rather than direct abstention control. RLEC achieves calibration performance comparable to RLCR at 4B and obtains a lower overconfidence rate and a higher overall score at 30B. RLEC also provides several practical advantages. Methods based on verbalized confidence [8, 30] often require an external rule to convert their confidence estimates into answer or abstention decisions, while other selective prediction methods [20, 32] further rely on separate calibrators or held-out calibration sets. In contrast, RLEC receives $p$ in the prompt and directly produces the final answer or abstention without external processing. It also supports fine-grained control over answer or abstention, with smooth changes across densely spaced values of $p ,$ as illustrated in Figure 1(b).

Our main contributions are as follows:

• We identify how the optimization algorithm and reward design used in prior work weaken the confidence target signal, providing a concrete explanation for the previously reported failure.

• We propose RLEC, which preserves the confidence target signal, introduces a stable reward family that keeps the desired answer condition unchanged, and uses reward scheduling to balance efective learning and training stability. To our knowledge, RLEC thereby enables the first successful training of a single LLM for explicit confidence target control.

• Extensive experiments across diverse benchmarks and model scales show that RLEC largely preserves general capabilities and achieves competitive calibration performance. It also requires no external decision system or calibration after training and supports fine-grained target control.

## 2 Background

## 2.1 Fixed-Penalty Abstention

To mitigate hallucinations [13, 17], a reliable model should know when to answer and when to abstain. Given a question x, the model chooses an action $a \in \{ \mathrm { A N S } , \mathrm { A B S } \}$ , where ANS denotes answering and ABS denotes abstaining. We primarily evaluate on verifiable question answering tasks following prior work [8, 35]. This setup yields three outcomes: a correct answer, an incorrect answer, or an abstention. We treat an incorrect answer as a hallucination. We define accuracy among answered questions as

$$
\mathrm { A c c } = \mathrm { P r } ( \mathrm { c o r r e c t } \mid a = \mathrm { A N S } ) .\tag{2}
$$

Recent work [16, 24, 35] has used reinforcement learning with verifiable rewards (RLVR) [10, 27] to directly train LLMs to decide whether to answer or abstain. A common formulation assigns a positive reward to a correct answer, a neutral reward to abstention, and a fixed negative reward to an incorrect answer:

$$
R _ { \lambda } ( a , y ) = \left\{ { \begin{array} { l l } { 1 , } & { a = { \mathrm { A N S ~ a n d ~ } } y { \mathrm { ~ i s ~ c o r r e c t } } , } \\ { 0 , } & { a = { \mathrm { A B S } } , } \\ { - \lambda , } & { a = { \mathrm { A N S ~ a n d ~ } } y { \mathrm { ~ i s ~ i n c o r r e c t } } , } \end{array} } \right. \quad \lambda \geq 0 .\tag{3}
$$

TruthRL [35] optimizes this reward with λ = 1 using Group Relative Policy Optimization (GRPO) [27], whereas Reinforced Hesitation [24] trains separate models for diferent values of λ. Let $q ( x )$ be the true conditional probability that the model’s intended answer to x is correct. Under Equation 3, the expected reward of answering is

$$
\begin{array} { r l } { \mathbb { E } [ R _ { \lambda } \mid x , a = \mathrm { A N S } ] = q ( x ) - \lambda \big ( 1 - q ( x ) \big ) } & { } \\ { = ( 1 + \lambda ) q ( x ) - \lambda . } \end{array}\tag{4}
$$

Since abstention yields zero reward, the policy answers when

$$
q ( x ) > { \frac { \lambda } { 1 + \lambda } } ,\tag{5}
$$

and abstains when the inequality is reversed. Equivalently, λ determines the fixed confidence threshold

$$
p _ { \lambda } = \frac { \lambda } { 1 + \lambda } .\tag{6}
$$

Under the idealized assumptions that $q ( x )$ gives the true probability of correctness and that the policy exactly follows Equation 5, its selective accuracy satisfies

$$
{ \begin{array} { r l } & { \operatorname* { P r } ( { \mathrm { c o r r e c t ~ } } | \ a = { \mathrm { A N S } } ) = \mathbb { E } [ q ( x ) \ | \ a = { \mathrm { A N S } } ] } \\ & { \qquad = \mathbb { E } [ q ( x ) \ | \ q ( x ) > p _ { \lambda } ] \geq p _ { \lambda } . } \end{array} }\tag{7}
$$

This characterizes idealized behavior, not an empirical guarantee, since the learned policy may not optimize or generalize perfectly and its uncertainty may not match the true probability of correctness.

Training with a fixed penalty provides an interpretable trade-of between accuracy and coverage, but each λ represents only one operating point. Adapting to diferent error costs therefore generally requires retraining the model [16, 24] or selecting a diferent model.

## 2.2 Explicit Confidence Targets

Explicit confidence targets [19] extend fixed-penalty abstention by allowing users to specify the required accuracy in the prompt. Let $p \in [ 0 , 1 )$ denote this target. Setting

$$
\lambda ( p ) = { \frac { p } { 1 - p } }\tag{8}
$$

in Equation 3 gives

$$
R _ { p } = \left\{ { \begin{array} { l l } { 1 , } & { { \mathrm { c o r r e c t ~ a n s w e r } } , } \\ { - { \frac { p } { 1 - p } } , } & { { \mathrm { i n c o r r e c t ~ a n s w e r } } , } \\ { 0 , } & { { \mathrm { a b s t e n t i o n } } . } \end{array} } \right.\tag{9}
$$

Suppose the probability of answering correctly is $q ( x )$ . The expected reward for answering is

$$
\begin{array} { l l } { \mathbb { E } [ R _ { p } \mid x , \mathrm { A N S } ] = q ( x ) - \frac { p } { 1 - p } \big ( 1 - q ( x ) \big ) } \\ { \quad \quad = \frac { q ( x ) - p } { 1 - p } . } \end{array}\tag{10}
$$

Abstention receives zero reward. A policy that maximizes expected reward therefore answers when $q ( x ) > p ,$ abstains when $q ( x ) < p ,$ , and is indiferent when $q ( x ) = p .$ . Thus, p directly controls the decision between answering and abstaining. When $p = 0 .$ , answering is at least as good as abstaining for every question and is strictly better whenever $q ( x ) > 0$ . We use this endpoint to represent the setting in which the user requires the model to answer, with ties resolved in favor of answering.

Our goal is to train a single policy $\pi _ { \theta } ( \cdot \mid x , p )$ that follows arbitrary confidence targets without retraining. For every p at which the model answers with nonzero probability, the desired behavior is

$$
\operatorname { A c c } ( p ) : = \operatorname* { P r } ( { \mathrm { c o r r e c t } } \mid \operatorname { A N S } , p ) \geq p .\tag{11}
$$

The model should answer when doing so yields positive expected reward, rather than defaulting to abstention on every question. Importantly, p is a target specified by the user in the prompt, and the model directly chooses whether to answer or abstain. Inference therefore requires no external system to compare a reported confidence score [8] with a threshold.

Wu et al. [36] propose Explicit Risk Thresholding (ERT) to study this approach by sampling p during training, including it in the prompt, and using the reward in Equation 9. However, their hallucination rates remain largely unchanged across diferent $p ,$ showing that the model does not adapt its behavior to the target. They attribute this failure to a noisy optimization problem caused by sampling targets across the full range, making it dificult to learn a consistent strategy for diferent p. Mohamadi et al. [24] also argue that a single model cannot support all risk preferences because limited model capacity prevents it from optimizing every error penalty simultaneously. Together, these findings illustrate the dificulty of training one model to follow arbitrary confidence targets.

## 3 Method

## 3.1 Signal Preservation

We first identify a key problem in prior work: most of the direct reward signal distinguishing diferent confidence targets is lost during training. ERT uses GRPO to optimize the reward in Equation 9. For each prompt, GRPO samples G responses and estimates their advantages from relative rewards. Let $R _ { i }$ denote the reward of response i. The corresponding advantage is

$$
A _ { i } = \frac { R _ { i } - \overline { { { R } } } } { \sigma _ { R } } , \qquad \overline { { { R } } } = \frac { 1 } { G } \sum _ { j = 1 } ^ { G } R _ { j } , \qquad \sigma _ { R } = \sqrt { \frac { 1 } { G } \sum _ { j = 1 } ^ { G } ( R _ { j } - \overline { { { R } } } ) ^ { 2 } } .\tag{12}
$$

However, this standardization can remove the efect of the reward scale induced by p on the advantages. At a fixed $p ,$ consider a rollout group with two distinct reward values $r _ { \mathrm { H } } > r _ { \mathrm { L } }$ . Let $\rho \in ( 0 , 1 )$ denote the fraction of responses receiving $r _ { \mathrm { H } }$ . The group mean and standard deviation are

$$
\overline { { R } } = \rho r _ { \mathrm { H } } + ( 1 - \rho ) r _ { \mathrm { L } } , \qquad \sigma _ { R } = ( r _ { \mathrm { H } } - r _ { \mathrm { L } } ) \sqrt { \rho ( 1 - \rho ) } .\tag{13}
$$

Substituting Equation 13 into Equation 12 gives

$$
A _ { \mathrm { H } } = { \sqrt { \frac { 1 - \rho } { \rho } } } , \qquad A _ { \mathrm { L } } = - { \sqrt { \frac { \rho } { 1 - \rho } } } .\tag{14}
$$

The resulting advantages are independent of $p ,$ so the training signal carried by the reward scale of $p$ disappears. For example, in a group with two distinct reward values, a higher p imposes a larger reward penalty on an incorrect answer, but does not make its standardized advantage more negative. Also, for a group with one reward value, all advantages in Equation 12 are computed as zero in our implementation, so the group provides no reward signal. Only groups containing all three outcomes can retain the learning signal from changes in the reward scale induced by $p .$

Figure 2 shows how the three group types evolve during training. Groups with three reward values are rare and become less frequent, while groups with one value eventually dominate and most others contain two. Thus, only a small fraction of groups retains a direct signal from reward magnitudes that vary with $p ,$ explaining why the policy remains insensitive to the confidence target.

![](images/d8217402bc0ba2acf816415063238b9e84f5f23947c2b79f159add7876ce4d65.jpg)

To preserve the signal, we remove the group standard deviation while retaining mean centering:

![](images/4c81c1eb7be73445ffa9170e8a22f809cdb7aee65b49c7c13c8d54ad40773d3a.jpg)  
Figure 2 Fractions of rollout groups.

$$
{ \widetilde { A } } _ { i } = R _ { i } - { \overline { { R } } } .\tag{15}
$$

Without standard deviation normalization, a reward gap that varies with p produces advantages that vary accordingly, preserving the direct reward signal across confidence targets. This also exposes the large penalty in Equation 9, which we address next.

## 3.2 Stable Rewards

Removing the group standard deviation preserves the efect of reward magnitudes on the advantages, but also makes policy updates sensitive to those magnitudes. In Equation 9, the rewards for a correct answer and abstention remain fixed at 1 and 0, while the reward for an incorrect answer is $- p / ( 1 - p )$ , which tends to −∞ as $p$ approaches one. Groups containing both incorrect answers and other outcomes can therefore produce large advantages. This creates two related problems.

Gradient dominance. Because the policy shares parameters across confidence targets, such groups at large $p$ can dominate the overall gradient, reducing the influence of lower targets.

Training instability. The number and composition of such groups vary across batches. At large $p ,$ their centered advantages can cause sharp fluctuations in gradient norms and destabilize training.

Multiplying all rewards by a single constant scales the centered advantages equally across confidence targets, leaving their relative influence unchanged. We instead use a positive factor that decreases with $p$ to reduce the penalty at large $p$ without changing the theoretical decision to answer or abstain.

A simple way to bound the error penalty is to multiply all rewards by $1 - p \colon$

$$
( 1 - p ) R _ { p } = \left\{ { \begin{array} { l l } { 1 - p , } & { { \mathrm { c o r r e c t ~ a n s w e r } } , } \\ { - p , } & { { \mathrm { i n c o r r e c t ~ a n s w e r } } , } \\ { 0 , } & { { \mathrm { a b s t e n t i o n } } . } \end{array} } \right.\tag{16}
$$

The magnitude of the error penalty is now bounded by one. This scaling also changes the reward for a correct answer to $1 - p$ , while abstention remains at zero.

To control how strongly rewards at large p are reduced, we introduce a family of rewards:

$$
R _ { p } ^ { ( b ) } = ( 1 - p ) ^ { b } R _ { p } = \left\{ { \begin{array} { l l } { ( 1 - p ) ^ { b } , } & { { \mathrm { c o r r e c t ~ a n s w e r } } , } \\ { - p ( 1 - p ) ^ { b - 1 } , } & { { \mathrm { i n c o r r e c t ~ a n s w e r } } , } \\ { 0 , } & { { \mathrm { a b s t e n t i o n } } , } \end{array} } \right. \quad \quad b \geq 0 .\tag{17}
$$

Here, b is a stability coeficient. For a fixed $p \in ( 0 , 1 )$ , increasing b reduces the magnitude of the error penalty and the reward for a correct answer. In particular, $b = 0$ recovers Equation 9, while $b = 1$ recovers Equation 16. Since the same scaling applies to all responses in a rollout group, it also scales their centered advantages:

$$
\widetilde { A } _ { i } ^ { ( b ) } = ( 1 - p ) ^ { b } \widetilde { A } _ { i } ^ { ( 0 ) } .\tag{18}
$$

Thus, larger values of b reduce the relative influence of groups with large $p .$

Importantly, scaling all rewards at a given p by the same positive factor does not change the theoretical decision to answer or abstain. Suppose the model answers correctly with probability $q ( x )$ . Its expected reward for answering is

$$
\begin{array} { r l } & { \mathbb { E } \Big [ R _ { p } ^ { ( b ) } \mid x , \mathrm { A N S } \Big ] = q ( x ) ( 1 - p ) ^ { b } - \big ( 1 - q ( x ) \big ) p ( 1 - p ) ^ { b - 1 } } \\ & { \qquad = ( 1 - p ) ^ { b - 1 } \big ( q ( x ) - p \big ) . } \end{array}\tag{19}
$$

Since $( 1 - p ) ^ { b - 1 } > 0$ for $p < 1$ , answering has positive expected reward exactly when $q ( x ) > p .$ . Thus, every value of b preserves the same theoretical answer condition.

## 3.3 Reward Scheduling

The stability coeficient b balances the strength of the learning signal against training stability. For any fixed $p \in ( 0 , 1 )$ , the centered advantages in Equation 18 become smaller in magnitude as b increases. This reduction is strongest at large p, where the original error penalty is largest. It can improve stability, but it also weakens the signal for learning when to abstain by reducing the error penalty relative to abstention. Our ablations suggest that keeping b fixed at a relatively large value, such as $b = 1$ , can make it harder for the model to learn when to abstain and to vary its behavior across confidence targets. Details are shown in Section 4.4.

By contrast, a smaller b, such as $b = 0 .$ , preserves stronger centered advantages at large p, helping the model learn when to abstain and distinguish confidence targets more efectively early in training. However, keeping b small throughout training can lead to gradient dominance and training instability, as discussed in Section 3.2. These efects may harm final performance and general capabilities.

Guided by this trade-of, we linearly increase b from 0 to 1 over T training steps. At step $t ,$ we set

$$
b _ { t } = \frac { t } { T } , \qquad 0 \leq t \leq T .\tag{20}
$$

This schedule preserves a stronger learning signal early while improving training stability later.

## 4 Experiment

In this section, we conduct experiments to answer the following research questions:

• (RQ1) Can RLEC successfully distinguish diferent explicit confidence targets?

• (RQ2) How does RLEC compare with prior methods in calibration and general capabilities?

• (RQ3) What enables RLEC to distinguish targets while preserving general capabilities?

• (RQ4) What other properties and practical advantages does RLEC ofer?

![](images/a29949773118eaa64b0c27d80d18506bc05d55a9473b262a8f5676d06fc5dd12.jpg)

![](images/7980f631e1d4e27e2ecb10037964cc3cc3c9287117655e96a8439d4bf0f33bc1.jpg)

![](images/42f45bcf303ba9599be3dc21e1f8fe93c03f6a7dcee4cd701f7363425564027c.jpg)

![](images/df96b7f3b7b0274a426a755916a15e5dda8e5d41b98dc52ebaf74f93aa31b142.jpg)  
Figure 3 Abstention rates on held-out validation sets during training. Each curve evaluates the same training checkpoints with a fixed confidence target p. ERT produces nearly overlapping curves for diferent targets, while RLEC learns higher abstention rates for higher values of $p .$

## 4.1 Experimental Setup

Implementation Details. Our main experiments and ablation studies are based on Qwen3-4B-Base [37], and we further conduct experiments on Qwen3-30B-A3B-Base to evaluate scalability. Our training data combine DAPO-Math [39] and a modified version of HotpotQA [8, 38]. We randomly sample 15,000 examples from each dataset, resulting in 30,000 training examples, and train for five epochs. We set $p = 0$ with probability 10%. Otherwise, we uniformly sample p from the interval [0, 0.9] and round it to two decimal places. During evaluation, we test $p \in \{ 0 , 0 . 1 , \ldots , 0 . 9 \}$ and report results averaged over multiple sampled generations. Additional details are provided in Appendix A.1.

Benchmarks. Our evaluation covers eight benchmarks from three domains and diferent dificulty levels, including (1) mathematical reasoning: GSM8K [5], MATH-500 [12, 21], and AIME 2024; (2) knowledge and hallucination: HotpotQA test set [8, 38], AA-Omniscience [15], and SimpleQA-Verified [11, 34]; and (3) scientific reasoning: MMLU-Pro [33] and GPQA [26]. Details are provided in Appendix A.2.

Baselines. We compare RLEC with three types of methods: (1) RLVR: standard RLVR with a binary reward; (2) abstention methods: methods that use a fixed error penalty or fail to learn behavior that varies with p, including TruthRL [35] and ERT [36], compared in RQ1; and (3) confidence reporting methods: RLCR [8], included in RQ2 as an additional reference for calibration performance under a diferent formulation.

## 4.2 Adaptation to Confidence Targets

To answer RQ1, we study whether models can change their behavior according to $p .$ We first compare the training behavior of RLEC and ERT, and then compare the final accuracy and abstention rates of RLEC, RLVR, ERT, and TruthRL across all evaluation benchmarks.

Training Curves. Figure 3 shows the abstention rate on held-out validation sets during training, evaluated with $p \in \{ 0 . 1 , 0 . 5 , 0 . 9 \}$ . On both HotpotQA and DAPO-Math, our reproduction of ERT shows the same failure reported in the original study [36]: its curves for diferent values of p almost completely overlap. In contrast, RLEC produces a clear dependence on $p ,$ with higher confidence targets generally leading to higher abstention rates. These results show that RLEC successfully learns diferent behaviors for diferent confidence targets.

Test Results. Figure 4(a,b) reports average abstention rates and answer accuracy across all benchmarks using Qwen3-4B-Base. Among methods that directly decide whether to answer, only RLEC changes clearly with p: as p increases from 0.1 to 0.9, its abstention rate rises from approximately 19% to 49%, while its accuracy among answered questions rises from approximately 68% to 81%. RLVR, ERT, and TruthRL remain nearly unchanged. Thus, RLEC successfully learns to adapt its decision to answer or abstain across confidence targets, answering RQ1.

![](images/d31808bb01c80c17c3284d6fe774d3d6eb1a5b2d7e11150d57fd9c5787887498.jpg)

![](images/2442dea102e13b61cea6eb50c47a9e60bfbaf999751d259863767ce0b930e11d.jpg)

![](images/56eb1e69ea87ee5b65ef05db728db479096cb73793c25d6fd4999574f7c0fddf.jpg)

![](images/09c3e9e22f34d3d37a3d02c103c4ad6eb8ce80cb72410369673b5f5aaee432d4.jpg)  
Figure 4 Adaptation to explicit confidence targets. (a,b) Direct decision methods with Qwen3-4B-Base. (c,d) RLEC, RLCR, and RLVR with Qwen3-30B-A3B-Base.

## 4.3 Comparison with Calibration Methods

As shown in Section 4.2, existing methods that directly decide whether to answer or abstain show little sensitivity to the confidence target. To compare calibration performance and answer RQ2, we therefore compare RLEC with RLCR [8], a diferent type of method that reports a numerical confidence c together with its answer. Since RLCR cannot directly follow a user provided target $p ,$ we apply an external rule that retains the answer when $c \geq p$ and replaces it with an abstention otherwise, whereas RLEC conditions its answer or abstention decision directly on p.

Qualitative Analysis. Figure $4 ( \mathrm { c } , \mathrm { d } )$ compares RLEC, RLCR, and RLVR across confidence targets using Qwen3-30B-A3B-Base. RLVR changes little with p, whereas RLEC and RLCR increase their abstention rates as p increases. RLEC responds more strongly, especially at high p, and its accuracy among answered questions follows the target boundary more closely. These results suggest that directly optimizing explicit confidence targets enables clearer adaptation to the requested target.

Quantitative Analysis. We use three metrics to evaluate answer ability, target violations, and utility. Forced Accuracy measures answer accuracy when abstention is disabled. We set $p = 0$ for RLEC and retain all RLCR answers, testing whether calibration training harms answer ability.

Average Overconfidence (AOC) measures the average magnitude of target violations:

$$
\mathrm { A O C } = \frac { 1 } { | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \operatorname* { m a x } \left\{ 0 , \ : p - \mathrm { A c c } ( p ) \right\} ,\tag{21}
$$

where lower values indicate better compliance with the requested accuracy. Since AOC can be reduced by abstaining, it does not capture utility. We therefore also report

$$
\mathrm { S c o r e } = \frac { 1 } { N | \mathcal { P } | } \sum _ { p \in \mathcal { P } } \sum _ { i = 1 } ^ { N } z _ { i , p } \left( y _ { i , p } - p \right) ,\tag{22}
$$

where z indicates whether the model answers and y indicates whether the answer is correct. A correct $z _ { i , p }$ $y _ { i , p }$ answer contributes 1 − p, an incorrect answer contributes −p, and an abstention contributes zero. A higher Score indicates greater utility by accounting for both correctness and coverage.

For RLEC, each Score contribution is exactly its reward with b = 1. For RLCR, twice the corresponding score integrated over all confidence targets equals its reward, as proved in Appendix C.

Table 1 reports Micro results pooled over all examples and Macro results averaged equally across benchmarks. Micro reflects aggregate performance and is less afected by dificult benchmarks such as SimpleQA-Verified, whereas Macro prevents larger benchmarks from dominating. RLEC largely preserves the answer ability obtained by RLVR, with comparable Forced Accuracy at both scales. Compared with RLCR, the 4B model achieves lower AOC but also lower Score, indicating a trade-of between target compliance and utility. At 30B, RLEC achieves both lower AOC and higher Score, outperforming RLCR in both respects. These results demonstrate the feasibility of direct training with explicit confidence targets and show that RLEC remains efective at the larger scale.

Table 1 Forced answer accuracy, average overconfidence (AOC), and average score across model scales. The best Micro and Macro results within each model scale are shown in bold.
<table><tr><td rowspan="2">Model</td><td colspan="2"></td><td colspan="3">Mathematical Reasoning</td><td colspan="3">Knowledge &amp; Hallucination</td><td colspan="2">Scientific Reasoning</td><td colspan="2">Avg.</td></tr><tr><td>Metric</td><td>Method</td><td>GSM8K</td><td>MATH</td><td>AIME</td><td>Hotpot</td><td>OmniSci</td><td>SimpleQA</td><td>MMLU-Pro</td><td>GPQA</td><td>Micro</td><td>Macro</td></tr><tr><td rowspan="9">Qwen3-4B- Base</td><td rowspan="3">Acc. ↑</td><td>RLVR</td><td>93.33</td><td>69.70</td><td>20.83</td><td>69.50</td><td>13.75</td><td>5.20</td><td>68.47</td><td>35.91</td><td>58.18</td><td>47.09</td></tr><tr><td>RLCR</td><td>93.10</td><td>73.90</td><td>16.25</td><td>66.30</td><td>15.58</td><td>4.75</td><td>66.94</td><td>34.00</td><td>57.35</td><td>46.35</td></tr><tr><td>RLEC</td><td>92.15</td><td>70.50</td><td>19.79</td><td>67.60</td><td>14.17</td><td>4.75</td><td>67.37</td><td>36.24</td><td>57.40</td><td>46.57</td></tr><tr><td rowspan="3">AOC↓</td><td>RLVR</td><td>0.000</td><td>0.027</td><td>0.258</td><td>0.033</td><td>0.292</td><td>0.389</td><td>0.032</td><td>0.175</td><td>0.049</td><td>0.151</td></tr><tr><td>RLCR</td><td>0.090</td><td>0.090</td><td>0.090</td><td>0.098</td><td>0.229</td><td>0.396</td><td>0.090</td><td>0.169</td><td>0.090</td><td>0.156</td></tr><tr><td>RLEC</td><td>0.000</td><td>0.000</td><td>0.090</td><td>0.010</td><td>0.259</td><td>0.381</td><td>0.015</td><td>0.157</td><td>0.013</td><td>0.114</td></tr><tr><td rowspan="3">Score ↑</td><td>RLVR</td><td>0.480</td><td>0.261</td><td>-0.162</td><td>0.237</td><td>-0.173</td><td>-0.213</td><td>0.237</td><td>-0.090</td><td>0.167</td><td>0.072</td></tr><tr><td>RLCR</td><td>0.482</td><td>0.342</td><td>0.053</td><td>0.255</td><td>-0.004</td><td>-0.094</td><td>0.285</td><td>0.069</td><td>0.235</td><td>0.173</td></tr><tr><td>RLEC</td><td>0.477</td><td>0.333</td><td>0.056</td><td>0.264</td><td>-0.048</td><td>-0.107</td><td>0.262</td><td>-0.016</td><td>0.214</td><td>0.153</td></tr><tr><td rowspan="9">Qwen3-30B- A3B-Base</td><td rowspan="3">Acc. ↑</td><td>RLVR</td><td>95.07</td><td>77.10</td><td>26.88</td><td>68.90</td><td>18.17</td><td>9.40</td><td>76.25</td><td>46.53</td><td>63.97</td><td>52.29</td></tr><tr><td>RLCR</td><td>94.73</td><td>77.70</td><td>26.25</td><td>66.20</td><td>19.08</td><td>8.90</td><td>77.22</td><td>46.76</td><td>64.24</td><td>52.10</td></tr><tr><td>RLEC</td><td>94.77</td><td>78.60</td><td>27.29</td><td>66.60</td><td>20.00</td><td>8.30</td><td>76.73</td><td>47.09</td><td>64.14</td><td>52.42</td></tr><tr><td rowspan="3">AOC↓</td><td>RLVR</td><td>0.000</td><td>0.014</td><td>0.221</td><td>0.038</td><td>0.265</td><td>0.372</td><td>0.020</td><td>0.108</td><td>0.045</td><td>0.130</td></tr><tr><td>RLCR</td><td>0.000</td><td>0.003</td><td>0.000</td><td>0.038</td><td>0.270</td><td>0.367</td><td>0.000</td><td>0.057</td><td>0.025</td><td>0.092</td></tr><tr><td>RLEC</td><td>0.000</td><td>0.003</td><td>0.000</td><td>0.009</td><td>0.209</td><td>0.368</td><td>0.002</td><td>0.072</td><td>0.001</td><td>0.083</td></tr><tr><td rowspan="3">Score ↑</td><td>RLVR</td><td>0.504</td><td>0.329</td><td>-0.169</td><td>0.222</td><td>-0.198</td><td>-0.309</td><td>0.300</td><td>0.026</td><td>0.195</td><td>0.088</td></tr><tr><td>RLCR</td><td>0.500</td><td>0.360</td><td>0.096</td><td>0.218</td><td>-0.173</td><td>-0.311</td><td>0.355</td><td>0.123</td><td>0.236</td><td>0.146</td></tr><tr><td>RLEC</td><td>0.504</td><td>0.353</td><td>0.091</td><td>0.258</td><td>-0.015</td><td>-0.132</td><td>0.336</td><td>0.112</td><td>0.261</td><td>0.188</td></tr></table>

![](images/4f3b7b068f9cf74c0352434d48f90f6bf9413eac92b1fc3e53d0fd89f8e628de.jpg)

![](images/edff61df21fadf13adc8f061399e6e8779797fc7d46482626806a9ce4df94d80.jpg)

![](images/d72c1831b71e4c08d1771d02545ea0ff85838b2be3a81776c9da6defd3d789cb.jpg)

![](images/fa6b356d44f3c162b94a2a64fb0a9a7b472825c5439e0b0898a720322184383a.jpg)

![](images/218b461daa0ec1616ae112ec3f974d066332f0e365121d77de2eeb38df9acedb.jpg)  
Figure 5 Ablation training curves. Abstention rates at $p \in \{ 0 . 1 , 0 . 5 , 0 . 9 \}$ for (a) RLEC, (b) RLEC with standard deviation normalization, and fixed (c) b = 0, (d) b = 0.5, and (e) b = 1.

## 4.4 Ablation Studies

To answer RQ3, we study two main design choices in RLEC: omitting the reward standard deviation from group advantage estimation and selecting the stability coeficient b.

Standard Deviation. As shown in Figure 5(b), adding group standard deviation normalization produces nearly overlapping abstention curves across diferent p. This supports our analysis that normalization weakens the reward scale signal induced by p. Also, Appendix B.2 shows that removing the group standard deviation does not consistently improve or degrade answer ability.

Choice of Stability Coeficient. Panels (c), (d), and (e) of Figure 5 illustrate the diferent behaviors induced by fixed choices of b. With b = 0, the model distinguishes the confidence targets relatively early, but its abstention rate fluctuates sharply. With b = 1, the curves nearly overlap, and the model fails to distinguish the targets. The intermediate choice b = 0.5 produces separation only late in training, indicating less eficient learning and potentially less reliable target diferentiation.

We also evaluate the diferent choices on the test benchmarks. Forced Accuracy, AOC, and Score remain broadly consistent, demonstrating that the proposed reward family is robust to the specific choice of b. The scheduled configuration achieves the highest aggregate Forced Accuracy while retaining competitive calibration performance. Because we consider preserving answer ability a primary requirement for behavioral calibration, we select it as our default configuration. Complete results and further discussion are provided in Appendix B.1.

![](images/9bc36d63d8bf1193b9711bd29c431b6029f7b2358e247cbc8203d46b166eb58b.jpg)

![](images/4a5cd359c7cc89627ea050b8041a9e65f4665b0d08869ca30b93ea4b68328859.jpg)

![](images/2bd8b087109f41345fcc3d112ddf79456b80cf7e37bdb724475b41f202c11483.jpg)

![](images/8b2b099beb7fe2113928bfeec2871ab74a3a9a7acff9132a76f106a787a0cd12.jpg)  
Figure 6 Fine-grained confidence control on AIME24 and AIME25. Targets vary at intervals of 0.01 up to $p = 0 . 9 5$ with Avg@1024 for each target. (a) Abstention rate. (b) Accuracy among answered questions. (c,d) Proportions of all responses that are correct and wrong. Training uses targets up to p = 0.90.

## 4.5 Additional Properties

To answer RQ4, we examine fine-grained target control, the operational implicit confidence profile, and the risk–coverage trade-of.

## 4.5.1 Fine-Grained Confidence Control

We evaluate RLEC on AIME24 and AIME25 with confidence targets spaced at intervals of 0.01, extending up to $p = 0 . 9 5$ . We choose these benchmarks because their dificulty provides an informative regime for studying fine-grained control: their relatively low accuracy avoids the saturation observed on easier benchmarks such as GSM8K, while a meaningful subset of problems remains within the model’s reasoning capability. In contrast, errors on knowledge-intensive benchmarks such as SimpleQA more often reflect missing factual knowledge, leaving less room for confidence control to distinguish solvable from unsolvable examples. Each target is evaluated using Avg@1024. Figure 6 reports the abstention rate, attempted accuracy, and the proportions of all responses that are correct and wrong. Because training uses only targets up to $p = 0 . 9 0$ , the interval $p \in ( 0 . 9 0 , 0 . 9 5 ]$ additionally evaluates behavior beyond the training range.

As shown in Figure 6, the abstention rate rises steadily and attempted accuracy generally improves as p increases, demonstrating that RLEC responds to fine-grained changes in the target, including values such as $p = 0 . 0 1$ and $p = 0 . 0 2$ . The curves are not strictly monotonic at every adjacent target, which we attribute to finite sample evaluation noise, but their overall trends remain stable. Attempted accuracy also generally meets or exceeds the target.

In contrast, although the RLCR prompt does not restrict the number of decimal places in its confidence output, as shown in Figure 8, all confidence values produced by the evaluated RLCR model lie on a one-decimal grid; we do not observe intermediate values such as 0.12. This observation is consistent with prior findings [4, 7]. Consequently, even if an external system applies arbitrary thresholds to these outputs, it cannot recover additional operating points between adjacent confidence values. The evaluated RLCR model therefore provides substantially coarser control than directly conditioning RLEC on targets at a resolution of 0.01.

Selective Error Filtering. Panels (c) and (d) further show that the number of wrong answers decreases much faster than the number of correct answers as p increases. This efect is particularly clear at lower confidence targets, where the proportion of correct answers remains nearly unchanged while many wrong answers are filtered out. RLEC therefore initially improves attempted accuracy primarily by abstaining from likely errors rather than discarding correct answers, yielding a favorable trade-of between coverage and answer reliability.

![](images/a791e455bba3ef61442407b0d8b23f60a6dc5b9a6f94bc67c7b21353e6414a60.jpg)

![](images/7517c9124b8a33f4c813eddc3b33a33f05a7e2017ad4c434d1ff2643abeb4077.jpg)

![](images/3a72de1b9e99ceae449fd420f9d3f4d94570ca2d470d81585abc46538512b045.jpg)

![](images/b363f9273170856b075aebba3d2fd74efc47e654facf4c9b367dd0e62d0c5b3f.jpg)  
Figure 7 Implicit confidence profiles and risk--coverage trade-offs on the 30B model. (a,b) Micro and Macro profiles, constructed from RLEC’s abstention behavior and RLCR’s reported confidence. (c,d) Micro and Macro risk–coverage curves. RLEC’s operational profile is more dispersed, and its risk–coverage curves extend to lower-coverage operating points.

Extrapolation Beyond the Training Range. RLEC is trained only with targets up to $p = 0 . 9 0$ , whereas Figure 6 extends the evaluation to the unseen range $p \in ( 0 . 9 0 , 0 . 9 5 ]$ . Despite small local fluctuations and the reduced coverage at these high targets, all four metrics remain stable and continue their overall trends without an abrupt behavioral change. These results suggest that RLEC remains responsive even to confidence targets beyond the range observed during training.

## 4.5.2 Implicit Confidence Profile

RLEC directly accepts an explicit confidence target p and decides whether to answer, rather than outputting a confidence score. Nevertheless, to better understand its aggregate behavior and facilitate comparison with RLCR, we construct an operational approximation of its implicit confidence profile. We measure the abstention rate at $p \in \{ 0 , 0 . 1 , . . . , 0 . 9 \}$ , treat it as an estimate of the cumulative mass below $p ,$ and use diferences between neighboring targets as interval masses. This construction summarizes the aggregate behavior of the p-conditioned policies and does not assume a shared sample-level latent confidence variable. For RLCR, we directly use its reported confidence.

Figure $\mathrm { 7 ( a , b ) }$ compares the resulting profiles on the 30B model. Under this construction, RLEC has a higher standard deviation (0.2335 versus 0.2126 for Micro and 0.2713 versus 0.2646 for Macro), with less mass in the highest interval and more in the intermediate intervals. These results suggest that RLEC supports a broader range of aggregate behavior across confidence targets.

## 4.5.3 Risk-Coverage Trade-off

Figure 7(c,d) plots risk, the error rate among answered questions, against coverage, the fraction of questions answered, on the 30B model. A lower curve is better, since it means fewer errors at the same coverage. Overall, the two methods perform similarly: RLEC is slightly better in the Micro setting, while RLCR is slightly better in the Macro setting. The main diference is in operating range. In both settings, RLEC extends to a lower coverage than RLCR, so it can abstain more and keep risk lower when the target calls for high reliability, whereas RLCR cannot reach the same conservative operating points. This is a potential practical advantage in applications that demand high reliability.

## 5 Related Work

## 5.1 Reinforcement Learning with Verifiable Rewards

Reinforcement learning with verifiable rewards (RLVR) [3, 10, 23, 27, 39] has become a common approach for improving LLM reasoning. A widely used algorithm is Group Relative Policy Optimization (GRPO) [27], which estimates advantages from multiple responses to the same prompt by subtracting the group mean and dividing by the group standard deviation. Prior work has examined the benefits and drawbacks of this normalization. Dr. GRPO [23] removes it to avoid weighting questions according to their dificulty. Turtel et al. [31] remove standard deviation normalization to preserve the efect of large prediction errors, while Bereket and Leskovec [2] show that removing it reduces biased advantages and overconfidence. In contrast,

Ge et al. [9] view normalization as adaptive gradient scaling that can improve convergence and stability, while CAL-GRPO [14] also reports that retaining normalization improves sample eficiency across its GRPO variants.

These studies mainly examine how standard deviation normalization afects optimization and model performance. Unlike all of them, RLEC uses the reward scale to encode the confidence target p. We show that for groups with two distinct reward values and fixed proportions, normalization removes the efect of the reward scale induced by p on the advantages. Such groups are common during training, helping explain the failure to distinguish confidence targets in prior work [36]. RLEC retains this signal through mean centering without standard deviation normalization.

## 5.2 Hallucination Mitigation and Behavioral Calibration

Hallucination [13, 17] remains a major source of errors in LLMs. In Why Language Models Hallucinate, Kalai et al. [19] argue that common training objectives encourage models to guess rather than admit uncertainty. This has motivated RL methods that reward correct answers, penalize errors, and allow abstention [16, 24, 25, 35]. However, these methods encode a fixed risk preference and cannot adapt their decision to answer or abstain to diferent user requirements at inference time. Wu et al. [36] propose Explicit Risk Thresholding (ERT), which extends this setting to confidence targets specified in the prompt, but report that it fails to learn distinct behaviors across targets.

Another line of work trains or prompts models to express confidence [22, 30]. For example, Damani et al. [8] propose Reinforcement Learning with Calibration Rewards (RLCR), which trains models to produce an answer together with a calibrated confidence score. However, these confidence estimates typically need to be converted into decisions by an external rule or system, and some approaches further require a separate calibrator or calibration set [20, 32]. In contrast, RLEC enables models to learn how to directly adapt their answer or abstention decisions to explicit confidence targets, without an external rule for converting confidence estimates into decisions.

## 6 Conclusion

We study explicit confidence targets as a more flexible and direct way to control LLM abstention. We show that the failure of prior work results from the loss of signals and the unstable scale of reward. Based on this analysis, we propose RLEC, which preserves the confidence target signal, introduces a stable reward family, and uses reward scheduling to balance efective learning and training stability. Experiments across diverse benchmarks and model scales show that explicit confidence target training is feasible. RLEC largely preserves general capabilities, achieves strong calibration, and supports fine-grained control without an external decision system. More broadly, RLEC opens a new and direct path toward behavioral calibration with explicit, user-specified confidence targets.

## References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

[2] Michael Bereket and Jure Leskovec. Uncalibrated reasoning: Grpo induces overconfidence for stochastic outcomes. arXiv preprint arXiv:2508.11800, 2025.

[3] Aili Chen, Aonian Li, Bangwei Gong, Binyang Jiang, Bo Fei, Bo Yang, Boji Shan, Changqing Yu, Chao Wang, Cheng Zhu, et al. Minimax-m1: Scaling test-time compute eficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

[4] Hui Chen, Antoine Didisheim, and Luciano A Somoza. Out of the black box: Uncertainty quantification for llms via conditional probabilities. Technical report, National Bureau of Economic Research, 2026.

[5] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[6] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[7] Yuyang Dai and Yuxia Wang. Rescaling confidence: What scale design reveals about llm metacognition. arXiv preprint arXiv:2603.09309, 2026.

[8] Mehul Damani, Isha Puri, Stewart Slocum, Idan Shenfeld, Leshem Choshen, Yoon Kim, and Jacob Andreas. Beyond binary rewards: Training lms to reason about their uncertainty. In International conference on learning representations, volume 2026, pages 59456–59493, 2026.

[9] Cheng Ge, Caitlyn Heqi Yin, Hao Liang, and Jiawei Zhang. Why grpo needs normalization: A local-curvature perspective on adaptive gradients. arXiv preprint arXiv:2601.23135, 2026.

[10] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[11] Lukas Haas, Gal Yona, Giovanni D’Antonio, Sasha Goldshtein, and Dipanjan Das. Simpleqa verified: A reliable factuality benchmark to measure parametric knowledge. arXiv preprint arXiv:2509.07968, 2025.

[12] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874, 2021.

[13] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, et al. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM transactions on information systems, 43(2):1–55, 2025.

[14] Muhammed Emrullah Ildiz, Halil Alperen Gozeten, Ege Onur Taga, and Samet Oymak. Learning to correct: Calibrated reinforcement learning for multi-attempt chain-of-thought. arXiv preprint arXiv:2604.17912, 2026.

[15] Declan Jackson, William Keating, George Cameron, and Micah Hill-Smith. Aa-omniscience: Evaluating crossdomain knowledge reliability in large language models. arXiv preprint arXiv:2511.13029, 2025.

[16] Abha Jha, Akanksha Mahajan, Ashwath Vaithinathan Aravindan, Praveen Saravanan, Sai Sailaja Policharla, and Sonal Chaturbhuj Gehlot. Rewarding intellectual humility learning when not to answer in large language models. arXiv preprint arXiv:2601.20126, 2026.

[17] Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. Survey of hallucination in natural language generation. ACM computing surveys, 55(12):1–38, 2023.

[18] Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Eli Tran-Johnson, et al. Language models (mostly) know what they know. arXiv preprint arXiv:2207.05221, 2022.

[19] Adam Tauman Kalai, Ofir Nachum, Santosh S Vempala, and Edwin Zhang. Why language models hallucinate. arXiv preprint arXiv:2509.04664, 2025.

[20] Amita Kamath, Robin Jia, and Percy Liang. Selective question answering under domain shift. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 5684–5696, 2020.

[21] Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations, volume 2024, pages 39578–39601, 2024.

[22] Stephanie Lin, Jacob Hilton, and Owain Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.

[23] Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. arXiv preprint arXiv:2503.20783, 2025.

[24] Mohamad Amin Mohamadi, Tianhao Wang, and Zhiyuan Li. Honesty over accuracy: Trustworthy language models through reinforced hesitation. arXiv preprint arXiv:2511.11500, 2025.

[25] Muyu Pan, Shu Zhao, Nan Zhang, Philip Shin, Varun Parekh, Vijaykrishnan Narayanan, and Rui Zhang. Tiar: Trajectory-informed advantage reweighting for llm abstention learning. arXiv preprint arXiv:2605.25850, 2026.

[26] David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

[27] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[28] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

[29] Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

[30] Katherine Tian, Eric Mitchell, Allan Zhou, Archit Sharma, Rafael Rafailov, Huaxiu Yao, Chelsea Finn, and Christopher D Manning. Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5433–5442, 2023.

[31] Benjamin Turtel, Danny Franklin, Kris Skotheim, Luke Hewitt, and Philipp Schoenegger. Outcome-based reinforcement learning to predict the future. arXiv preprint arXiv:2505.17989, 2025.

[32] Qingni Wang, Yue Fan, and Xin Wang. Safer: Risk-constrained sample-then-filter in large language models. In International Conference on Learning Representations, volume 2026, pages 53723–53747, 2026.

[33] Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

[34] Jason Wei, Nguyen Karina, Hyung Won Chung, Yunxin Joy Jiao, Spencer Papay, Amelia Glaese, John Schulman, and William Fedus. Measuring short-form factuality in large language models. arXiv preprint arXiv:2411.04368, 2024.

[35] Zhepei Wei, Xiao Yang, Kai Sun, Jiaqi Wang, Rulin Shao, Jingxiang Chen, Mohammad Kachuee, Teja Gollapudi, Yiwei Liao, Nicolas Schefer, et al. Truthrl: Incentivizing truthful llms via reinforcement learning. arXiv preprint arXiv:2509.25760, 2025.

[36] Jiayun Wu, Jiashuo Liu, Zhiyuan Zeng, Tianyang Zhan, Tianle Cai, and Wenhao Huang. Mitigating llm hallucination via behaviorally calibrated reinforcement learning. arXiv preprint arXiv:2512.19920, 2025.

[37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[38] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.

[39] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

## Appendix

A Detailed Experimental Setup 15   
A.1 Implementation Details 15   
A.2 Benchmarks 15   
A.3 Baselines 16   
B Additional Experimental Results 17   
B.1 Choice of Stability Coeficient 17   
B.2 Standard Deviation . 19   
B.3 Increasing the Rollout Group Size 19   
B.4 Scaling RLEC to Longer Contexts 20   
C Relation Between Score and the RLCR Reward 21   
D Limitations and Future Work 22

## A Detailed Experimental Setup

## A.1 Implementation Details

Training and Evaluation. We use verl for RL training [28], with the hyperparameters for all RL methods summarized in Table 2. We do not use Clip-Higher, entropy regularization, or a KL penalty. For RLCR, we determine abstention by applying the requested confidence target as an external threshold to its reported confidence. For all other methods, abstentions are identified by matching the designated abstention string. For $p = 0 ,$ , we use a binary reward and evaluation criterion: only a correct answer is treated as correct, whereas incorrect answers and abstentions are both treated as incorrect. For AOC, accuracy is set to zero when no answers are attempted, balancing calibration against utility. Due to limited computational resources, we do not conduct multiple

Table 2 Hyperparameters.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Training</td><td></td></tr><tr><td>max_response_length</td><td>4096</td></tr><tr><td>train_batch_size</td><td>256</td></tr><tr><td>ppo_mini_batch_size</td><td>128</td></tr><tr><td>optim.lr</td><td>1 × 10−6</td></tr><tr><td>rollout.temperature</td><td>1.0</td></tr><tr><td>rollout.n</td><td>8</td></tr><tr><td>Evaluation</td><td></td></tr><tr><td>max_tokens</td><td>4096</td></tr><tr><td>temperature</td><td>1.0</td></tr><tr><td>top-p</td><td>0.95</td></tr></table>

training runs for each setting. Instead, we report Avg@16 for AIME and Avg@2 for all other benchmarks to reduce evaluation variance.

Prompts. The prompt templates used by RLEC and the baselines are shown in Figure 8. For other abstention methods, including ERT and TruthRL, we use the RLEC prompt during both training and evaluation. For RLVR, we use its standard prompt during training and at p = 0 to represent the conventional RL setting and compare answer ability. At $p > 0$ , we use the RLEC prompt to demonstrate that standard RLVR training alone does not enable target-conditioned abstention. For SimpleQA-Verified and AA-Omniscience, we additionally use the grading prompt shown in Figure 9 and use Qwen3.6-27B as the judge. Since abstention is identified separately, this grader only determines whether the predicted answer is semantically equivalent to the reference answer.

## A.2 Benchmarks

We use the standard versions of the evaluation benchmarks except for the following two settings.

HotpotQA. We use the modified version [8] of HotpotQA [38]. Each example contains eight paragraphs, while zero, one, or both of the two relevant paragraphs are removed to test whether the model abstains when the available evidence is incomplete.

![](images/cb12ae6a10d02ab636467102265573e99d36448e0271977e526504041854f29f.jpg)

Figure 8 Prompt templates used by RLEC and the baselines.  
Correctness-Grading System Prompt   
You are a grading assistant. Decide whether the predicted answer is semantically equivalent to the reference gold   
answer for the given question. Ignore formatting diferences, units notation, LaTeX wrappers, and trivial   
rephrasing. Reply with a single token: YES or NO.  
Figure 9 System prompt used to grade SimpleQA-Verified and AA-Omniscience.

MMLU-Pro. We evaluate the four science subsets in MMLU-Pro [33]: biology, chemistry, physics, and health. Since the full benchmark contains substantially more examples than the other benchmarks, this selection prevents MMLU-Pro from dominating the aggregate results.

## A.3 Baselines

RLVR. RLVR uses a binary correctness reward and does not explicitly reward abstention:

$$
R _ { \mathrm { R L V R } } = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { c o r r e c t } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{23}
$$

In the main experiments, we remove its group standard deviation normalization to align its advantage with RLEC and enable a closer comparison. Appendix B.2 examines the efect of this choice.

TruthRL. TruthRL [35] uses a fixed ternary reward that penalizes incorrect answers while assigning zero reward to abstentions:

$$
R _ { \mathrm { { T r u t h R L } } } = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { c o r r e c t } , } \\ { - 1 , } & { \mathrm { i n c o r r e c t } , } \\ { 0 , } & { \mathrm { a b s t e n t i o n } . } \end{array} \right.\tag{24}
$$

Our framework is compatible with TruthRL by setting $p = 0 . 5 , b = 0$ , and retaining group standard-deviation normalization.

Table 3 Forced answer accuracy, average overconfidence (AOC), and average score for RLVR and diferent stability coeficient and scheduling choices. The best Micro and Macro results among variants that distinguish confidence targets are shown in bold; RLVR and b = 1 are excluded from this comparison because they cannot distinguish among targets.
<table><tr><td rowspan="2">Metric</td><td rowspan="2">Method</td><td colspan="3">Mathematical Reasoning</td><td colspan="3">Knowledge &amp; Hallucination</td><td colspan="2">Scientific Reasoning</td><td colspan="2">Avg.</td></tr><tr><td>GSM8K</td><td>MATH</td><td>AIME</td><td>Hotpot</td><td>OmniSci</td><td>SimpleQA</td><td>MMLU-Pro</td><td>GPQA</td><td>Micro</td><td>Macro</td></tr><tr><td rowspan="7">Acc. ↑</td><td>RLVR</td><td>93.33</td><td>69.70</td><td>20.83</td><td>69.50</td><td>13.75</td><td>5.20</td><td>68.47</td><td>35.91</td><td>58.18</td><td>47.09</td></tr><tr><td>Ours</td><td>92.15</td><td>70.50</td><td>19.79</td><td>67.60</td><td>14.17</td><td>4.75</td><td>67.37</td><td>36.24</td><td>57.40</td><td>46.57</td></tr><tr><td>b = 0</td><td>91.09</td><td>69.30</td><td>16.25</td><td>66.00</td><td>14.25</td><td>4.65</td><td>67.32</td><td>36.35</td><td>56.95</td><td>45.65</td></tr><tr><td>b = 0.5</td><td>92.46</td><td>70.90</td><td>15.62</td><td>67.70</td><td>14.75</td><td>4.25</td><td>67.40</td><td>33.00</td><td>57.18</td><td>45.76</td></tr><tr><td>b= 1</td><td>91.81</td><td>70.40</td><td>19.38</td><td>67.80</td><td>14.08</td><td>5.30</td><td>69.49</td><td>34.56</td><td>58.29</td><td>46.60</td></tr><tr><td>LR Schedule</td><td>91.77</td><td>69.80</td><td>16.46</td><td>64.00</td><td>14.08</td><td>4.10</td><td>67.07</td><td>34.68</td><td>56.69</td><td>45.25</td></tr><tr><td>Sine Schedule</td><td>93.03</td><td>69.60</td><td>20.00</td><td>67.80</td><td>13.75</td><td>4.45</td><td>65.77</td><td>35.12</td><td>56.63</td><td>46.19</td></tr><tr><td rowspan="7">AOC↓</td><td>RLVR</td><td>0.000</td><td>0.027</td><td>0.258</td><td>0.033</td><td>0.292</td><td>0.389</td><td>0.032</td><td>0.175</td><td>0.049</td><td>0.151</td></tr><tr><td>Ours</td><td>0.000</td><td>0.000</td><td>0.090</td><td>0.010</td><td>0.259</td><td>0.381</td><td>0.015</td><td>0.157</td><td>0.013</td><td>0.114</td></tr><tr><td>b = 0</td><td>0.000</td><td>0.002</td><td>0.037</td><td>0.008</td><td>0.178</td><td>0.388</td><td>0.016</td><td>0.144</td><td>0.010</td><td>0.096</td></tr><tr><td>b = 0.5</td><td>0.000</td><td>0.001</td><td>0.018</td><td>0.000</td><td>0.207</td><td>0.370</td><td>0.015</td><td>0.141</td><td>0.008</td><td>0.094</td></tr><tr><td>b = 1</td><td>0.000</td><td>0.010</td><td>0.020</td><td>0.031</td><td>0.251</td><td>0.384</td><td>0.025</td><td>0.154</td><td>0.029</td><td>0.109</td></tr><tr><td>LR Schedule</td><td>0.000</td><td>0.004</td><td>0.013</td><td>0.017</td><td>0.227</td><td>0.382</td><td>0.013</td><td>0.144</td><td>0.011</td><td>0.100</td></tr><tr><td>Sine Schedule</td><td>0.000</td><td>0.003</td><td>0.009</td><td>0.016</td><td>0.293</td><td>0.399</td><td>0.015</td><td>0.148</td><td>0.022</td><td>0.111</td></tr><tr><td rowspan="7">Score ↑</td><td>RLVR</td><td>0.480</td><td>0.261</td><td>-0.162</td><td>0.237</td><td>-0.173</td><td>-0.213</td><td>0.237</td><td>-0.090</td><td>0.167</td><td>0.072</td></tr><tr><td>Ours</td><td>0.477</td><td>0.333</td><td>0.056</td><td>0.264</td><td>-0.048</td><td>-0.107</td><td>0.262</td><td>-0.016</td><td>0.214</td><td>0.153</td></tr><tr><td>b = 0</td><td>0.474</td><td>0.304</td><td>0.054</td><td>0.267</td><td>-0.010</td><td>-0.063</td><td>0.264</td><td>-0.001</td><td>0.222</td><td>0.161</td></tr><tr><td>b = 0.5</td><td>0.477</td><td>0.334</td><td>0.048</td><td>0.276</td><td>-0.025</td><td>-0.061</td><td>0.258</td><td>0.015</td><td>0.222</td><td>0.165</td></tr><tr><td>b = 1</td><td>0.474</td><td>0.305</td><td>0.073</td><td>0.241</td><td>-0.087</td><td>-0.129</td><td>0.247</td><td>-0.034</td><td>0.198</td><td>0.136</td></tr><tr><td>LR Schedule</td><td>0.473</td><td>0.316</td><td>0.063</td><td>0.241</td><td>-0.011</td><td>-0.066</td><td>0.261</td><td>0.001</td><td>0.219</td><td>0.160</td></tr><tr><td>Sine Schedule</td><td>0.478</td><td>0.330</td><td>0.074</td><td>0.257</td><td>-0.101</td><td>-0.180</td><td>0.264</td><td>-0.007</td><td>0.204</td><td>0.139</td></tr></table>

ERT. ERT [36] samples a confidence target p and applies its corresponding error cost during standard GRPO training:

$$
R _ { \mathrm { E R T } } ( p ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { c o r r e c t } } , } \\ { - { \frac { p } { 1 - p } } , } & { { \mathrm { i n c o r r e c t } } , } \\ { 0 , } & { { \mathrm { a b s t e n t i o n } } . } \end{array} \right. }\tag{25}
$$

Our framework is compatible with ERT by setting $b = 0$ and retaining group standard-deviation normalization.

RLCR. RLCR [8] trains the model to produce an answer together with a confidence score $c \in [ 0 , 1 ]$ . Let y = 1 if the answer is correct and $y = 0$ otherwise; its reward combines correctness with the Brier score:

$$
\begin{array} { r } { R _ { \mathrm { R L C R } } ( y , c ) = y - ( c - y ) ^ { 2 } . } \end{array}\tag{26}
$$

For RLCR, we remove group standard deviation normalization to obtain better calibration [2, 31]. At evaluation time, we use an external decision system that retains the answer when $c \geq p$ and replaces it with an abstention otherwise.

## B Additional Experimental Results

## B.1 Choice of Stability Coefficient

We examine the choice of b within our proposed reward family, considering fixed values as well as linear and sinusoidal schedules. As shown in Equation 19, these choices preserve the same theoretical decision rule but induce diferent balances among signal strength, optimization stability, and answer ability. Table 3 reports their Forced Accuracy, AOC, and Score, reflecting answer ability, target compliance, and utility, respectively. We additionally include LR Schedule as a possible variant, which applies a global cosine learning-rate decay.

![](images/9450006dd3a163a1215b2157ddd47c8e99eb9a6b3307d49b411529cd2f6f59fd.jpg)

![](images/5c4b79c1595b296e31668a9ecc81746eafe681f833dbab78e0a55221859d58ea.jpg)

![](images/89ebb00f889d3c21f99f89fe69abe2f5cbe742fc65a53144a16ed04b252758a9.jpg)

(d) b = 0  
![](images/d2c8447b367ea8228603e0c204e3dbab51a750b07f8a75766bd16c0351ce8d0b.jpg)

(e) b = 0.5  
(f) b = 1  
![](images/27af87f035d569eb18fde877246c3c0211336a7bad3103c3af32411933d9a59b.jpg)

![](images/d0856adbaca2127ff9645bc2c6c288c093f5517f7b2258a475e565bdbbddd99f.jpg)  
Figure 10 Training dynamics of the abstention rate on HotpotQA under diferent choices of the stability coeficient and scheduling strategy. RLEC uses the linear schedule of $b ,$ while RLEC Sin. uses a sinusoidal schedule. Curves correspond to target confidence levels $p \in \{ 0 . 1 , 0 . 5 , 0 . 9 \}$ .

Forced Accuracy. Among the configurations that distinguish confidence targets, Forced Accuracy remains broadly stable, with the linear schedule obtaining the highest aggregate Forced Accuracy. The clearest diferences appear on AIME, where fixed $b = 0 ,$ , fixed $b = 0 . 5 \AA$ , and LR Schedule are all more than three points below the linear schedule. Because AIME is particularly dificult and benefits substantially from reinforcement learning, this result is consistent with training stability being especially important for preserving abilities acquired during this stage. Diferences on most other benchmarks are smaller or less consistent, particularly on knowledge benchmarks such as SimpleQA-Verified and AA-Omniscience, where performance largely depends on knowledge acquired during pretraining. LR Schedule performs worse in aggregate Forced Accuracy, indicating that scheduling the relative reward scale across confidence targets is not equivalent to applying a global learning-rate decay.

Calibration Performance. All choices of b substantially improve calibration over RLVR, while the diferences among these choices remain modest. In particular, b = 0.5 obtains the lowest aggregate AOC and the highest Macro Score, while retaining relatively strong Forced Accuracy.

Training Dynamics. The training dynamics in Figure 10, however, reveal an additional distinction that is not captured by the final aggregate results. Under the linear schedule, the behavior for $p = 0 . 9$ separates from the lower-confidence targets around the middle of training and remains distinctly ordered thereafter. Fixed $b = 0$ also distinguishes the targets relatively early, but exhibits larger fluctuations. With fixed $b = 0 . 5 .$ , clear separation emerges only after approximately 400 steps despite its favorable final calibration and relatively strong Forced Accuracy, suggesting that it may not establish target conditioned behavior as reliably throughout optimization and may be more sensitive to the available training horizon. With $b = 1$ , the curves remain nearly indistinguishable, consistent with the absence of a target-dependent reward gap. Overall, smaller values of b strengthen target-dependent learning but reduce stability, whereas larger values stabilize updates but weaken diferentiation. The linear schedule balances these efects by establishing the distinctions early and stabilizing them later.

(c) HotpotQA  
Table 4 Forced answer accuracy for RLVR with and without group standard deviation normalization. The best result in each column is shown in bold.
<table><tr><td>Method</td><td colspan="3">Mathematical Reasoning</td><td colspan="3">Knowledge &amp; Hallucination</td><td colspan="2">Scientific Reasoning</td><td colspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td></td><td>GSM8K</td><td>MATH</td><td>AIME</td><td>Hotpot</td><td>OmniSci</td><td>SimpleQA</td><td>MMLU-Pro</td><td>GPQA</td><td>Micro</td><td>Macro</td></tr><tr><td>RLVR w/o Std.</td><td>93.33</td><td>69.70</td><td>20.83</td><td>69.50</td><td>13.75</td><td>5.20</td><td>68.47</td><td>35.91</td><td>58.18</td><td>47.09</td></tr><tr><td>RLVR w/ Std.</td><td>92.57</td><td>72.30</td><td>19.17</td><td>68.30</td><td>14.83</td><td>5.15</td><td>68.97</td><td>36.80</td><td>58.45</td><td>47.26</td></tr></table>

![](images/5db7a937d066d6407d47a6c12eac202af548c3503bc7aeaeaabe1969aaeac9ef.jpg)

![](images/1778ec5ae55d2489ffa67995e7df018092b90502f61863da02860f1c801d826e.jpg)

![](images/9b3944d42800cc7624474b99cb119233ca87205392c09fbb5a70fc6a5f9656e3.jpg)

![](images/05a6d53c8306587b57ce8ddedf88e78e921ce86696d0a4a48986ae66f8d67228.jpg)  
Figure 11 Scaling ERT to a rollout group size of 16. (a,b): group compositions during training. (c,d): abstention rates under diferent confidence targets.

Final Choice. We treat preserving answer ability as a primary requirement for behavioral calibration, since calibration improvements are less desirable when accompanied by degradation in the model’s underlying answer accuracy. Although the sinusoidal schedule achieves comparable Macro results across all three metrics, it provides no consistent advantage over the linear schedule. We therefore adopt the linear schedule as our default configuration and use it to scale RLEC to the 30B model, because it achieves stronger aggregate Forced Accuracy, establishes earlier and more persistent target separation, and retains a simpler form than the sinusoidal alternative.

## B.2 Standard Deviation

RLEC removes the group standard deviation from advantage estimation to preserve the reward signal. To enable a closer comparison, we also remove this normalization from RLVR in our main experiments. To further isolate its efect on RLVR’s answer ability, we compare RLVR with centered advantages against standard RLVR under the same binary reward. As shown in Table 4, their aggregate Forced Accuracy is nearly identical, and neither variant consistently outperforms the other across benchmarks. Thus, removing the group standard deviation provides no clear advantage or disadvantage in answer ability in our experiment setting.

## B.3 Increasing the Rollout Group Size

We further examine whether increasing the rollout group size can help ERT preserve the confidence target signal. We retain the original reward and standard deviation normalization, so this setting difers from ERT only by increasing the number of responses per prompt from 8 to 16. This change approximately doubles the rollout generation cost.

As shown in Figure 11, increasing the rollout group size does not enable ERT to distinguish confidence targets. On HotpotQA, groups containing all three reward values rapidly disappear, and groups containing a single reward dominate later training. DAPO Math retains more groups containing all three rewards, but its abstention curves still largely overlap across diferent values of p. Thus, although a larger rollout group can increase the chance of observing all three outcomes, it does not resolve the loss of target information in groups containing only one or two reward values. These results indicate that increasing rollout size alone cannot replace preserving the target signal during advantage estimation.

![](images/d33d7e3f5afab1b852a441383cbe2c4f93f3ff6d9cb7eb43036bcb44635aed2a.jpg)

![](images/d9eb0f8529542dd8dcc8ca743a96b328045ca85016f79b7d2be6586dcba13d34.jpg)

![](images/a9746eeb57ed68f4c679646f9cfe3f0a3b78a02a33bab3839fd2824d15f6e372.jpg)  
Figure 12 Training dynamics under different model scales and context lengths. We report validation accuracy on HotpotQA and DAPO-Math and the average response length during training.

(a) 4B, 4k  
![](images/1eacebdbdf67a5773c39bc8736395a3ade41fb548f148fb5b7ea92727c1eb539.jpg)

(b) 30B, 4k  
![](images/acc814c2a89fff45c71db9e262131ccbe0f745a91de21b60aff7d5eeb5cc2a88.jpg)

(c) 30B, 16k  
![](images/5a618b7e25cc0bf059026fcbce8ad12442c73dfe6598a2d85325b61433507b57.jpg)  
Figure 13 Abstention dynamics on DAPO-Math for different model scales and context lengths.

## B.4 Scaling RLEC to Longer Contexts

Due to computational constraints, our main experiments primarily use a 4K context window for both training and inference. However, dificult reasoning tasks such as AIME may benefit from a larger token budget. To examine whether RLEC can be extended to longer contexts, we increase both the training and inference context windows to 16K while keeping all other hyperparameters identical to those in Table 2.

Training Dynamics. As shown in Figure 12, increasing the context window from 4K to 16K leads to substantially longer responses from the 30B model. This additional budget produces a clear improvement in validation reward on DAPO-Math, whereas the HotpotQA trajectories remain largely unchanged. These results suggest that the longer context primarily benefits reasoning-intensive mathematical problems rather than uniformly improving all tasks. The stronger reasoning performance also changes the model’s abstention behavior on DAPO-Math. As shown in Figure 13, the 16K model abstains less than its 4K counterpart at higher confidence targets, particularly at p = 0.9, while preserving the ordering across targets. This behavior is consistent with the stronger model answering more questions even under stringent confidence requirements.

Evaluation Results. Table 5 reports the complete evaluation results. As expected, the largest improvement appears on AIME, where Forced Accuracy increases from 27.29% with the 4K context to 38.96% with the 16K context. Results on the remaining benchmarks are mixed and generally show smaller diferences. Consequently, aggregate Forced Accuracy improves slightly, largely driven by the gain on AIME, while AOC and Score remain broadly comparable. Overall, the longer context primarily benefits dificult mathematical reasoning without materially afecting behavior calibration.

Table 5 Forced answer accuracy, average overconfidence (AOC), and average score for RLEC-30B with 4K and 16K contexts. The best Micro and Macro results for each metric are shown in bold.
<table><tr><td></td><td></td><td colspan="3">Mathematical Reasoning</td><td colspan="3">Knowledge &amp; Hallucination</td><td colspan="2">Scientific Reasoning</td><td colspan="2"> $\operatorname { A v g } .$ </td></tr><tr><td>Metric</td><td>Context</td><td>GSM8K</td><td>MATH</td><td>AIME</td><td>Hotpot</td><td>OmniSci</td><td>SimpleQA</td><td>MMLU-Pro</td><td>GPQA</td><td>Micro</td><td>Macro</td></tr><tr><td rowspan="2">Acc. ↑</td><td>4K</td><td>94.77</td><td>78.60</td><td>27.29</td><td>66.60</td><td>20.00</td><td>8.30</td><td>76.73</td><td>47.09</td><td>64.14</td><td>52.42</td></tr><tr><td>16K</td><td>94.28</td><td>75.00</td><td>38.96</td><td>65.60</td><td>20.08</td><td>7.55</td><td>77.99</td><td>51.68</td><td>64.86</td><td>53.89</td></tr><tr><td rowspan="2">AOC↓</td><td>4K</td><td>0.000</td><td>0.003</td><td>0.000</td><td>0.009</td><td>0.209</td><td>0.368</td><td>0.002</td><td>0.072</td><td>0.001</td><td>0.083</td></tr><tr><td>16K</td><td>0.000</td><td>0.005</td><td>0.000</td><td>0.013</td><td>0.182</td><td>0.358</td><td>0.004</td><td>0.071</td><td>0.004</td><td>0.079</td></tr><tr><td rowspan="2">Score ↑</td><td>4K</td><td></td><td></td><td>0.091</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>16K</td><td>0.504 0.502</td><td>0.353 0.329</td><td>0.157</td><td>0.258 0.251</td><td>-0.015 0.003</td><td>-0.132 -0.084</td><td>0.336 0.348</td><td>0.112 0.111</td><td>0.261 0.272</td><td>0.188 0.202</td></tr></table>

## C Relation Between Score and the RLCR Reward

We show that the score used in our evaluation directly corresponds to the RLCR reward. Let $y \in \{ 0 , 1 \}$ indicate whether an answer is correct, and let $c \in [ 0 , 1 ]$ be the confidence reported by RLCR. Given a confidence target $p ,$ the external decision system answers when $c \geq p$ and abstains otherwise. Its decision can therefore be written as

$$
z _ { p } = \mathbf { 1 } \{ c \geq p \} .\tag{27}
$$

The score at target p is

$$
s _ { p } = z _ { p } ( y - p ) = \mathbf { 1 } \{ c \geq p \} ( y - p ) .\tag{28}
$$

Integrating this score over all confidence targets gives

$$
\int _ { 0 } ^ { 1 } s _ { p } d p = \int _ { 0 } ^ { 1 } { \mathbf { 1 } } \{ c \geq p \} ( y - p ) d p\tag{29}
$$

$$
= \int _ { 0 } ^ { c } ( y - p ) d p\tag{30}
$$

$$
{ } = c y - { \frac { c ^ { 2 } } { 2 } } .\tag{31}
$$

RLCR uses the following correctness and calibration reward:

$$
\begin{array} { r } { R _ { \mathrm { R L C R } } ( c , y ) = y - ( c - y ) ^ { 2 } . } \end{array}\tag{32}
$$

Because $y \in \{ 0 , 1 \}$ , we have $y ^ { 2 } = y$ . The reward can therefore be rewritten as

$$
R _ { \mathrm { R L C R } } ( c , y ) = y - c ^ { 2 } + 2 c y - y ^ { 2 }\tag{33}
$$

$$
= 2 c y - c ^ { 2 }\tag{34}
$$

$$
= 2 \left( c y - { \frac { c ^ { 2 } } { 2 } } \right)\tag{35}
$$

$$
= 2 \int _ { 0 } ^ { 1 } s _ { p } d p .\tag{36}
$$

Thus, when the confidence reported by RLCR is converted into an answer or abstention through thresholding, twice the score integrated over all confidence targets is exactly the RLCR reward.

In our experiments, we evaluate a finite set of confidence targets $\mathcal { P }$ and compute

$$
{ \widehat { \mathrm { S c o r e } } } = { \frac { 1 } { | { \mathcal { P } } | } } \sum _ { p \in { \mathcal { P } } } s _ { p } .\tag{37}
$$

For uniformly spaced targets over [0, 1), this is a Riemann approximation to the integral in Equation 31. The approximation approaches the exact relation in Equation 36 as the interval between evaluated targets decreases.

The same score also directly corresponds to the RLEC reward. For a given p,

$$
z _ { p } ( y - p ) = \left\{ { \begin{array} { l l } { 1 - p , } & { { \mathrm { c o r r e c t ~ a n s w e r } } , } \\ { - p , } & { { \mathrm { i n c o r r e c t ~ a n s w e r } } , } \\ { 0 , } & { { \mathrm { a b s t e n t i o n } } , } \end{array} } \right.\tag{38}
$$

which is exactly $R _ { p } ^ { ( 1 ) }$ , the RLEC reward with $b = 1$ . Therefore, Score provides a common measure for comparing RLEC and RLCR under explicit confidence targets.

## D Limitations and Future Work

Under idealized assumptions, our reward objective induces the decision rule $q ( x ) > p ,$ guaranteeing $\operatorname { A c c } ( p ) \geq p$ among answered questions. In practice, limited model capacity, imperfect estimates of answer correctness, finite rollouts, and stochastic optimization can prevent the learned policy from fully realizing this rule, while distribution shifts can further weaken target compliance. Our experiments show clear adaptation to explicit confidence targets and improved aggregate target compliance relative to standard RLVR. However, the requested accuracy is not consistently attained on every benchmark: the remaining gaps are most visible on challenging knowledge-intensive tasks such as SimpleQA-Verified and AA-Omniscience, particularly with the 4B model. Thus, the theoretical objective does not guarantee that the trained model will satisfy every requested target in practice. Future work should explore improved training objectives and data to further improve calibration and reduce target violations, especially on dificult tasks and under distribution shifts, while preserving answer ability and fine-grained target control.