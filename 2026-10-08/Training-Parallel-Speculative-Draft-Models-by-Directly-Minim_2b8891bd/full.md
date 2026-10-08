# Training Parallel Speculative Draft Models by Directly Minimizing Expected Decoding Rounds

Yunxiao Zhao<sup>∗</sup>

Changxiao Cai<sup>∗</sup>

October 8, 2026

## Abstract

Speculative decoding accelerates large language model inference by using a low-cost draft model to propose tokens that the full-size target model verifies in parallel. Parallel and semi-autoregressive (semi-AR) drafters improve drafting eficiency by proposing an entire block in a single forward pass, but training them raises a new dificulty: the draft distribution for a given position depends on where the decoding round starts, and where rounds start depends on how many tokens earlier rounds accepted. Existing training objectives typically rely on block-local surrogates that ignore this cross-round coupling, and therefore do not directly optimize the global decoding eficiency. In this work, we develop a theoretical framework for training and evaluating these drafters by representing speculative decoding as a Markov reward process. This formulation yields the Expected Decoding Rounds (EDR) objective, which weights local rejection costs by state occupancies and exactly equals the expected number of decoding rounds. Unlike prior surrogate objectives, EDR introduces no auxiliary hyperparameters. We then derive an exact temporal-diference gradient that supports unbiased stochastic optimization from target-model rollouts. The same framework also yields an exact ofline evaluator for round counts, enabling paired drafter comparisons on shared target rollouts without running speculative decoding. Finetuning two state-of-theart drafters, DSpark and DFly, with EDR consistently improves mean accepted length and outperforms existing training objectives across nine benchmarks spanning math reasoning, code generation, and chat.

## <sup>§</sup> https://github.com/y-x-zhao/AngelSpec-EDR

## Contents

1 Introduction

2 Preliminaries

<sup>3</sup> <sup>Main</sup> <sup>Results</sup> 3.1 Speculative Decoding as a Markov Reward Process 3.2 The EDR Objective 3.3 TD-Form Gradient of EDR 7 3.4 Exact Ofline Evaluation . 3.5 Suboptimality of Block-Local Surrogates

4 Related Work 9   
4.1 Literature Review 9   
4.2 Comparison of Training Objectives 10   
Experiments 12   
5.1 Experimental Setup 12   
5.2 Experimental Results . 13   
6 Conclusion 13   
A Speculative Decoding Algorithm 17   
B Proofs 17   
B.1 Markov Process Formulation in Section 3.1 17   
B.2 Proofs of Proposition 1 and Proposition 2 20   
B.3 Proof of Theorem 1 . 21   
B.4 Proof of Theorem 2 . 21   
B.5 Proof of Theorem 3 . 22   
C Additional Details for Expected Decoding Rounds 24   
C.1 Draft EOS Probability Does Not Reduce EDR 24   
C.2 Fixed-Size Anchor Sampling . 24   
D Additional Details for Experiments 25

## 1 Introduction

Autoregressive (AR) generation is a central bottleneck in large language model (LLM) inference. Since each token can be produced only after its preceding tokens are available, the number of function evaluations required to generate a text sequence grows linearly with its length.

Speculative decoding (Leviathan et al., 2023; Chen et al., 2023) reduces this sequential cost by using a lightweight draft model to propose multiple candidate tokens, which the full-size target model verifies in parallel in a single forward pass. With rejection-sampling verification, the generated sequence follows the target model’s distribution exactly. Therefore, speculative decoding accelerates inference without sacrificing quality, and the goal is to reduce expensive target model verifications—equivalently, decoding rounds—while keeping drafting itself inexpensive.

Early work employed AR drafters (Li et al., 2024b) that predict draft tokens sequentially, so the latency contributed in the drafting phase grows linearly with the draft block size. To reduce the sequential work of drafting, parallel drafters (Cai et al., 2024; Chen et al., 2026a) have emerged to propose multiple draft tokens in one forward pass, typically built on masked difusion language models (Sahoo et al., 2024; Shi et al., 2024). Their semi-AR variants (Cheng et al., 2026; Liu et al., 2026) add limited sequential dependence within each draft block to improve the draft quality. Parallel drafting supports longer draft blocks and deeper drafter networks, with potential to significantly accelerate the inference of a large target model.

Challenges in parallel drafter training: cross-round dependence. For a fixed architecture, a drafter’s quality afects inference speed primarily through the number of decoding rounds, so the natural training objective is to minimize the expected number of rounds.

For an AR drafter, the proposal at position t is conditioned on the full realized prefix $x _ { < t }$ regardless of where the current round started, so its acceptance probability depends only on this prefix. As a result, the standard local training objective for AR drafters—matching the target model’s next-token distribution at every prefix—is directly aligned with this goal (Yin et al., 2024a). In contrast, for a parallel or semi-AR drafter, the proposal can instead depend on an earlier prefix $x _ { \leq n }$ with $n < t$ (in a round starting at n), with access to only some or none of the intermediate tokens. Consequently, the same output position can have diferent draft distributions depending on where the round starts.

This additional dependence breaks the direct link between local distribution matching and global decoding eficiency. The draft for position t from a round starting at n matters only if the decoding trajectory first reaches a round starting at n and then accepts all preceding draft tokens in that round up to position t − 1. Both events depend on earlier acceptance outcomes induced by the drafter itself. Thus, changing the drafter afects both acceptance within a round and which draft distributions are encountered in subsequent rounds, making the training objective inherently coupled across rounds.

However, existing objectives for parallel and semi-AR drafters do not account for these cross-round dependencies and instead rely on local surrogates. Examples include the expected number of tokens accepted within a single round (Li et al., 2026c) and draft-target distributional discrepancies under hand-crafted weights (Chen et al., 2026a; Wu et al., 2026). This motivates the central question:

Can we train parallel and semi-AR drafters by directly minimizing the expected number of decoding rounds?

Our contributions. We answer this question afirmatively. Our approach relies on two key observations. First, conditioned on the output sequence, speculative decoding with a parallel or semi-AR drafter can be formulated as a Markov reward process (MRP) (Howard, 1971): each state records where the current round started and which position is being verified, and each rejection costs one round. Second, since rejectionsampling verification ensures that the output follows the same target distribution for every drafter, averaging over target rollouts recovers the expected number of rounds of actual decoding. Together, these insights allow us to capture the dependencies across decoding rounds explicitly and characterize the expected number of decoding rounds exactly, yielding a unified framework for training and evaluating drafters.

Building on this framework, we introduce the Expected Decoding Rounds (EDR) objective, whose minimization is equivalent to minimizing the expected number of decoding rounds. We then derive a temporaldiference gradient estimator, enabling drafter training from target-model rollouts. EDR applies across drafter architectures and introduces no hyperparameters. We further prove that drafters maximizing the expected accepted length for each round can be strictly suboptimal for EDR, theoretically justifying the advantages of our global objective. The framework also yields an exact ofline evaluator of expected round counts, enabling comparisons of drafters on the same target-model rollouts without running online speculative decoding.

Empirically, we finetune two state-of-the-art (SOTA) drafters, DSpark (Cheng et al., 2026) and DFly (Liu et al., 2026), with EDR while keeping their architectures and inference procedures unchanged. Across nine benchmarks spanning mathematical reasoning, code generation, and chat, a single epoch of EDR finetuning consistently improves mean accepted length over the original drafters and existing training objectives.

Our main contributions are summarized as follows:

• A Markov reward process formulation. We provide a theoretical framework that exactly characterizes expected speculative decoding rounds for parallel and semi-AR drafters.

• An exact and trainable global objective. We introduce EDR, establish its equivalence to minimizing the expected number of decoding rounds, derive a tractable gradient for drafter training, and prove its advantage over local surrogates.

• An exact ofline evaluator. We provide an ofline evaluator of the expected number of decoding rounds, enabling paired evaluation of drafters on the same target-model rollouts.

• Consistent empirical gains. EDR finetuning improves two SOTA drafters across nine benchmarks, outperforming existing training objectives.

## 2 Preliminaries

Let V be a vocabulary of ordinary tokens and $\overline { { \nu } } : = \mathcal { V } \cup \{ < \mathtt { E } 0 \mathtt { S } > \}$ , where <EOS> denotes the end-of-sequence token. For two distributions $p , q ,$ define the total variation (TV) distance $\begin{array} { r } { \mathsf { T V } ( p , q ) : = \sum _ { v \in \overline { { \mathcal V } } } \left[ p ( v ) - q ( v ) \right] _ { + } } \end{array}$ where $[ z ] _ { + } : = \operatorname* { m a x } ( 0 , z )$ . For any sequence ${ \boldsymbol { x } } = ( x _ { 1 } , x _ { 2 } , \dots )$ , we denote $x _ { \leq a } = ( x _ { 1 } , x _ { 2 } , \ldots , x _ { a } ) , x _ { < a } : = x _ { 1 : a - 1 }$ and $x _ { a : b } : = ( x _ { a } , x _ { a + 1 } , \ldots , x _ { b } )$ for $1 \leq a \leq b$ . For $a > b , x _ { a : b } : = \emptyset$

Target distribution. Let p denote the distribution of the target model. It generates tokens autoregressively as $x _ { t } \sim p _ { t } ( \cdot \mid x _ { < t } )$ for $t \geq 1$ and stops upon sampling <EOS>. Denote by $L : = \operatorname* { i n f } \{ \ell \geq 0 : x _ { \ell + 1 } = < \mathtt { E O S } > \}$ the sequence length excluding <EOS>. We assume L has a universal constant upper bound for $x \sim p . \mathrm { ~ \ A ~ }$ complete sequence $x = ( x _ { 1 : L } , < \mathtt { E } 0 S > )$ has probability $\begin{array} { r } { \prod _ { t = 1 } ^ { L } p _ { t } ( x _ { t } ~ \vert ~ x _ { < t } ) p _ { L + 1 } ( < \mathtt { E } 0 \mathtt { S } > ~ \vert ~ x _ { 1 : L } ) } \end{array}$ , and we call x target-supported if $p ( x ) > 0$

Speculative decoding. Speculative decoding generates a token sequence x in rounds. Consider a round that starts from a committed prefix $x _ { 1 : n }$ . A drafter $q$ first proposes a block of B draft tokens $\widetilde { x } _ { n + 1 : n + B }$ with $\widetilde { x } _ { t } \sim q _ { t }$ . For simplicity, we write $q _ { t }$ for the draft distribution at position t in the current round, suppressing its dependence on the round start $n ,$ committed prefix $x _ { 1 : n }$ , and preceding within-block proposals $\widetilde { x } _ { n + 1 : t - 1 }$ We will make this explicit in the next paragraph.

The target model then verifies the draft block from left to right in a single forward pass. When position t is verified, all earlier drafts in the block have been accepted, so the current prefix is $\boldsymbol { x } _ { < t } = ( x _ { 1 : n } , \widetilde { x } _ { n + 1 : t - 1 } )$ Write $p _ { t } = p _ { t } ( \cdot \mid x _ { < t } )$ . The draft $\widetilde { x } _ { t }$ is accepted with probability min $\{ 1 , p _ { t } ( \widetilde { x } _ { t } ) / q _ { t } ( \widetilde { x } _ { t } ) \}$ , in which case it is committed as $x _ { t } = \widetilde { x } _ { t }$ . Otherwise, it is replaced with a correction token $x _ { t } \sim r _ { t }$ drawn from the residual distribution $r _ { t } ( v ) : = [ p _ { t } ( v ) - q _ { t } ( v ) ] _ { + } / \mathsf { T V } ( p _ { t } , q _ { t } ) , v \in \overline { { \mathcal V } }$ . The remaining drafts in the block are discarded, and the next round begins from the updated prefix $x _ { 1 : t }$ . If all B draft tokens are accepted, the same forward pass also yields one additional bonus token $x _ { n + B + 1 } \sim p _ { n + B + 1 } ( \cdot \mid x _ { \leq n + B } )$ , and the next round starts after it. Generation stops when <EOS> is committed. This rejection-sampling verification ensures that the output sequence satisfies $x \sim p$ for any drafter q (Leviathan et al., 2023).

Draft distribution. Drafters difer in what $q _ { t }$ conditions on. An AR drafter proposes each token from the full prefix, whereas parallel and semi-AR drafters propose the whole block from the prefix committed at the start of the round. We now refine the shorthand $q _ { t }$ to make this explicit. For a round starting from a committed prefix $x _ { 1 : n } ,$ let $q _ { n , t } \big ( \cdot \mid x _ { 1 : n } ; \widetilde { x } _ { n + 1 : t - 1 } \big )$ denote the draft distribution for position $t = n + 1 , \ldots , n + B$ The subscript t identifies the position being predicted, and n records the position where the round starts. The semicolon separates the committed prefix $x _ { 1 : n }$ from the preceding within-block proposals $\widetilde { x } _ { n + 1 : t - 1 }$ , which the drafter may use in whole or in part.

This notation covers AR, parallel, and semi-AR drafters. An AR drafter is the special case $q _ { n , t } ( \cdot \ |$ $x _ { 1 : n } ; \widetilde { x } _ { n + 1 : t - 1 } \dag \ = \ q _ { t } ( \cdot \ \mid \ x _ { 1 : n } , \widetilde { x } _ { n + 1 : t - 1 } )$ , which conditions on the concatenated prefix and has no further dependence on the round start n. A parallel drafter samples the block independently given the committed prefix $x _ { 1 : n } , \ \mathrm { i . e . , } \ q _ { n , t } ( \cdot \ | \ x _ { 1 : n } ; \widetilde { x } _ { n + 1 : t - 1 } ) = q _ { n , t } ( \cdot \ | \ x _ { 1 : n } )$ . A semi-AR drafter allows partial dependence on preceding within-block drafts while maintaining parallel prediction across some positions. For parallel and semi-AR drafters, the same current prefix can therefore induce diferent draft distributions depending on the round start n. This dependence is the main diference from AR drafters and the central challenge this work addresses. We give the Pseudocode for speculative decoding process we considered in Algorithm 2 in Appendix A.

Decoding rounds and mean accepted length. Each decoding round requires one forward pass of the target model. Let $\mathcal { T } ^ { q }$ denote the number of decoding rounds needed to produce a complete output under drafter $q ,$ including the one that produces <EOS>. A common metric is the mean accepted length (MAL), the ratio of the expected output length to the expected number of rounds,

$$
{ \mathsf { M A L } } ( q ) : = { \frac { \mathbb { E } [ L + 1 ] } { \mathbb { E } [ { \mathcal { T } } ^ { q } ] } } ,\tag{1}
$$

where $L + 1$ is the length of the complete output counting all committed tokens, including correction tokens, bonus tokens, and the terminal <EOS> token. As the output sequence follows the target distribution, $\mathbb { E } [ L ]$ is independent of the drafter $q$ for a fixed target model and evaluation corpus. Consequently, maximizing MAL(q) is equivalent to minimizing $\mathbb { E } [ \mathcal { T } ^ { q } ]$

## 3 Main Results

This section presents our main results. Section 3.1 presents a Markov reward process (MRP) formulation for speculative decoding. Section 3.2 introduces the EDR objective and Section 3.3 presents a temporal-diference form of its gradient. Section 3.4 gives ofline estimators for the decoding statistics, and suboptimality of block-local objectives is discussed in Section 3.5.

## 3.1 Speculative Decoding as a Markov Reward Process

Recall that $\mathcal { T } ^ { q }$ denotes the number of decoding rounds in a speculative decoding run with drafter q, and let $X = ( X _ { 1 } , \dots , X _ { L } , < \mathtt { E 0 S } > )$ be the output sequence of the same run. Since speculative decoding ensures $X \sim p$ for any drafter q, the overall expected number of decoding rounds satisfies

$$
\begin{array} { r } { \mathbb { E } [ \mathcal { T } ^ { q } ] = \mathbb { E } _ { X \sim p } [ \tau ^ { q } ( X ) ] , \qquad \mathrm { ~ w i t h ~ } \tau ^ { q } ( x ) : = \mathbb { E } [ \mathcal { T } ^ { q } \mid X = x ] , } \end{array}\tag{2}
$$

where $\tau ^ { q } ( x )$ is the expected number of rounds conditioned on producing the sequence x. Therefore, it sufices to characterize $\tau ^ { q } ( x )$ for any target-supported sequence x.

Markov reward process. In light of this, we fix a target-supported sequence $x = ( x _ { 1 } , \dots , x _ { L } , < \mathtt { E O S } > )$ Conditioned on the output x of a speculative decoding run, the remaining randomness is which positions are filled by accepted drafts and which by the target model after a rejection, and this determines the number of decoding rounds $\mathcal T ^ { q } ( x )$

This can be described as a Markov chain on states $( n , t )$ for $0 \leq n < t \leq L + 1$ . (see Appendix B.1 for a detailed derivation). State $( n , t )$ indicates that the current round started after $\iota \ ( \mathrm { i } . \mathrm { e } . , \ x _ { 1 : n }$ was committed) and is verifying the draft for position t. Decoding starts at (0, 1) and terminates when <EOS> is produced at position $L + 1$

Note that at state $( n , t )$ with $t \leq n + B$ , the draft for position t is drawn from $q _ { n , t } \big ( \cdot \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } \big )$ . By Bayes’ rule, given that position t outputs $x _ { t }$ , the probablity that it was produced by an accepted draft is

$$
A _ { n , t } ^ { q } ( x _ { \leq t } ) : = \operatorname* { m i n } { \biggl \{ } 1 , { \frac { q _ { n , t } ( x _ { t } \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } ) } { p _ { t } ( x _ { t } \mid x _ { < t } ) } } { \biggr \} } .\tag{3}
$$

Denote by $R _ { n , t } ^ { q } : = 1 - A _ { n , i } ^ { q }$ the rejection probability. Acceptance continues the current round, whereas rejection starts a new round after position t, provided that $x _ { t } \neq < \mathtt { E O S } >$ . Therefore, the Markov chain has two transitions:

$$
( n , t ) \longrightarrow \left\{ \begin{array} { l l } { { ( n , t + 1 ) , } } & { { \mathrm { w i t h ~ p r o b a b i l i t y } A _ { n , t } ^ { q } ( x _ { \le t } ) , } } \\ { { ( t , t + 1 ) , } } & { { \mathrm { w i t h ~ p r o b a b i l i t y } R _ { n , t } ^ { q } ( x _ { \le t } ) . } } \end{array} \right.\tag{4}
$$

We set $A _ { n , t } ^ { q } = 0$ and $R _ { n , t } ^ { q } = 1$ for $t > n + B$ , so the same transition rule covers the case when all tokens in a draft block are accepted and a new round starts after that.

Let $\omega _ { n , t } ^ { q } ( x )$ denote the occupancy of state $( n , t )$ : the probability that this state is visited, conditioned on the output sequence being x. It is not hard to see that $\omega _ { n , t } ^ { q } ( x )$ depends on x only through $x _ { < t }$ and hence we write $\omega _ { n , t } ^ { q } ( \boldsymbol { x } _ { < t } ) = \omega _ { n , t } ^ { q } ( \boldsymbol { x } )$ . Starting from $\omega _ { 0 , 1 } ^ { q } ( x ) = 1$ , the occupancies satisfy the forward recursion:

$$
\begin{array} { r } { \omega _ { n , t + 1 } ^ { q } ( x _ { \leq t } ) = \omega _ { n , t } ^ { q } ( x _ { < t } ) A _ { n , t } ^ { q } ( x _ { \leq t } ) , \quad n < t \leq L , } \end{array}\tag{5a}
$$

$$
\omega _ { t , t + 1 } ^ { q } ( \boldsymbol { x } _ { \le t } ) = \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( \boldsymbol { x } _ { < t } ) R _ { n , t } ^ { q } ( \boldsymbol { x } _ { \le t } ) , \quad 1 \le t \le L .\tag{5b}
$$

Since every round begins at a diagonal state $( t , t + 1 )$ , the number of rounds equals the number of diagonal states visited, yielding the following exact expression (see Appendix B.2 for proof).

Proposition 1. For any target-supported sequence $x = ( x _ { 1 : L } , < E \bar { O } S > )$ and any drafter q, the expected number of decoding rounds to produce x is

$$
\tau ^ { q } ( x ) = \sum _ { t = 0 } ^ { L } \omega _ { t , t + 1 } ^ { q } ( x _ { \leq t } ) = 1 + \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) R _ { n , t } ^ { q } ( x _ { \leq t } ) .\tag{6}
$$

The second identity has a direct MRP interpretation. Rejection starts a new round and incurs unit cost, while acceptance incurs zero cost. Hence, the rejection probability $R _ { n , t } ^ { q }$ is the expected cost of visiting state $( n , t )$ , and the occupancy-weighted sum of costs gives the round count.

## 3.2 The EDR Objective

By (2), averaging (6) over $x \sim p$ provides an exact expression for the overall expected number of decoding rounds $\mathbb { E } [ \mathcal { T } ^ { q } ]$ . However, each rejection probability $R _ { n , t } ^ { q } ( x _ { \leq t } )$ uses the target only through the single realized token $x _ { t } .$ even though the full distribution $p _ { t } ( \cdot \mid x _ { < t } )$ is available. Hence, we average over this token under the target distribution $p _ { t } ,$ conditioned on $x _ { t }$ being a non-EOS token.

Definition 1. Define the local rejection cost of state $( n , t )$ , for $0 \leq n < t \leq L$ , as

$$
c _ { n , t } ^ { q } ( x _ { < t } ) : = \underset { X _ { t } \sim p _ { t } ^ { \prime } ( \cdot | x _ { < t } ) } { \mathbb { E } } [ R _ { n , t } ^ { q } ( x _ { < t } , X _ { t } ) ]\tag{7}
$$

where $p _ { t } ^ { \prime } ( v \mid x _ { < t } ) : = p _ { t } ( v \mid x _ { < t } ) / ( 1 - p _ { t } ( < \mathtt { E O S } > \mid x _ { < t } ) )$ for $v \neq < \mathtt { E 0 S } >$ is the target distribution normalized over non-EOS tokens. For $t \leq n + B$ , its value is given by

$$
c _ { n , t } ^ { q } ( \boldsymbol { x } _ { < t } ) = \frac { 1 } { 1 - p _ { t } ( < \mathrm { E O S } > \mid \boldsymbol { x } _ { < t } ) } \sum _ { \boldsymbol { v } \in \mathcal { V } } [ p _ { t } ( \boldsymbol { v } \mid \boldsymbol { x } _ { < t } ) - q _ { n , t } ( \boldsymbol { v } \mid \boldsymbol { x } _ { 1 : n } ; \boldsymbol { x } _ { n + 1 : t - 1 } ) ] _ { + } ,\tag{8}
$$

and for $t > n + B ,$ it is equal to 1. Note that $c _ { n , t } ^ { q }$ is well-defined because every position $t \leq L$ outputs a non-EOS token so $p _ { t } \big ( < \mathtt { E 0 S } > \ | \ x _ { < t } \big ) < 1$ . We denote $c _ { n , L + 1 } ^ { q } ( x ) = 0$ for notational convenience, since the EOS token incurs no cost.

In words, the cost $c _ { n , t } ^ { q } ( x _ { < t } )$ is the rejection probability associated with state $( n , t )$ , given that the committed output token $x _ { t }$ is non-EOS. It measures the local distributional discrepancy between the drafter $q _ { n , t }$ and target $p _ { t }$ for position t, given by the TV distance between the draft and target distributions for position $t ,$ excluding EOS and normalized by the target’s non-EOS mass. Excluding EOS reflects the fact that when the target model produces EOS, it never starts another decoding round and incurs no penalty for draft rejection.

We now introduce the Expected Decoding Rounds (EDR) objective, which exactly equals the overall expected number of decoding rounds, as formalized in Theorem 1 below. The proof can be found in $\mathrm { A p - }$ pendix B.3.

Theorem 1. For any drafter $q ,$ the EDR objective equals the expected number of decoding rounds:

$$
\mathcal { L } _ { \mathrm { E D R } } ( q ) : = 1 + \mathbb { E } _ { X \sim p } \left[ \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( X _ { < t } ) c _ { n , t } ^ { q } ( X _ { < t } ) \right] = \mathbb { E } _ { X \sim p } [ \tau ^ { q } ( X ) ] = \mathbb { E } [ T ^ { q } ] .\tag{9}
$$

The EDR objective is a sum of local rejection costs $c _ { n , t } ^ { q } ,$ measuring the local distributional discrepancy between the drafter $q _ { n , t }$ and target $p _ { t } .$ , each weighted by the corresponding occupancy $\omega _ { n , t } ^ { q } .$ The quantity inside the expectation, which we call the per-sequence EDR loss,

$$
\widehat { \mathcal { L } } _ { \mathrm { E D R } } ( q ; x ) : = 1 + \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) c _ { n , t } ^ { q } ( x _ { < t } )\tag{10}
$$

can be computed exactly given a target-sampled sequence x via the forward occupancy recursion (5) and the local rejection cost (7). Therefore, averaging over a batch of target rollouts provides an empirical EDR loss for drafters that directly targets minimizing the expected number of rounds.

We make several important remarks.

• Replacement of $R _ { n , t } ^ { q }$ with $c _ { n , t } ^ { q }$ . The EDR loss (10) replaces $R _ { n , t } ^ { q } ( x _ { \leq t } )$ in (6) with its conditional expectation $c _ { n , t } ^ { q } ( \boldsymbol { x } _ { < t } )$ . For an individual rollout, it generally difers from $\tau ^ { q } ( x )$ , but they agree after averaging over $x \sim p .$ In return, the EDR loss uses the next-token distribution rather than one sampled token, so its gradient aggregates information across the vocabulary.

• Exact occupancy weighting. Prior surrogate objectives (Chen et al., 2026a; Wu et al., 2026) do not account for EOS termination in their local discrepancy terms, and more importantly, weight them in a heuristic way. These weights can be interpreted as fixed surrogates for the drafter-dependent occupancies $\omega _ { n , t } ^ { q } .$ . In contrast, EDR combines EOS-aware local costs with the actual occupancies, yielding an exact objective for the expected number of decoding rounds.

Algorithm 1 One EDR training step (sg(·) denotes the stop-gradient operator)   
Require: drafter $q _ { \theta } .$ target model $p ,$ block size B   
1: For each prompt $i = 1 , \ldots , M$ , sample an independent target rollout $x ^ { ( i ) } = ( x _ { 1 : L _ { i } } ^ { ( i ) } , < \mathsf { E } 0 \mathsf { S } > )$ from the target   
model p   
2: Compute $p _ { t } ( \cdot \mid x _ { < t } ^ { ( i ) } )$ for all i and t in one batched target forward pass   
3: for $i = 1 , \dots , M$ (in parallel) do   
4: Compute $A _ { n , t } ^ { ( i ) }$ and $c _ { n , t } ^ { ( i ) }$ from q<sub>θ</sub> at all states $( n , t )$   
5: Compute $\omega _ { n , t } ^ { ( i ) }$ by (5) and $V _ { n , t } ^ { ( i ) }$ by (12), without gradients   
6: end for   
7 $\begin{array} { r } { \begin{array} { r l } { \cdot } & { \ell ( \theta )  \cdots } \\ { : } & { \ell ( \theta )  M ^ { - 1 } , \sum _ { i = 1 } ^ { M } \sum _ { t = 1 } ^ { L _ { i } } \sum _ { n = 0 } ^ { t - 1 } \mathrm { s g } ( \boldsymbol { \omega } _ { n , t } ^ { ( i ) } ) \big [ \boldsymbol { c } _ { n , t } ^ { ( i ) } + A _ { n , t } ^ { ( i ) } \mathrm { s g } ( V _ { n , t + 1 } ^ { ( i ) } - V _ { t , t + 1 } ^ { ( i ) } ) \big ] } \end{array} } \end{array}$   
8: Update θ using $\nabla _ { { \boldsymbol { \theta } } } \ell ( { \boldsymbol { \theta } } )$

• Parallel vs. AR drafters. For an AR drafter, $q _ { n , t } ( \cdot \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } ) = q _ { t } ( \cdot \mid x _ { < t } )$ for all $n < t$ . Without a block cap $( B = \infty )$ , EDR reduces to

$$
\mathcal { L } _ { \mathrm { E D R } } ( q ) = 1 + \mathbb { E } _ { X \sim p } \left[ \sum _ { t = 1 } ^ { L } c _ { t } ( X _ { < t } ) \right] \quad \mathrm { w i t h } \quad c _ { t } ( x _ { < t } ) : = \frac { \sum _ { v \in \mathcal { V } } [ p _ { t } ( v \mid x _ { < t } ) - q _ { t } ( v \mid x _ { < t } ) ] _ { + } } { 1 - p _ { t } ( < \mathrm { E } 0 5 > \mid x _ { < t } ) } .\tag{11}
$$

EDR hence decouples into local prefix costs, each minimized by matching $q _ { t }$ to $p _ { t }$ . This recovers the result of Yin et al. (2024a, Theorem 1) established for AR drafters and extends it to sequences that terminate with EOS. A finite block size adds only a mild dependence on n through the boundary states. Therefore, Theorem 1 highlights the key diference between parallel and AR drafters, and EDR captures the cross-round dependencies exactly.

## 3.3 TD-Form Gradient of EDR

The per-sequence EDR loss in (10) depends on the drafter q through both the local costs $c _ { n , t } ^ { q }$ and the occupancies $\omega _ { n , t } ^ { q }$ . Diferentiating it with respect to q directly is expensive because it must backpropagate through the entire forward recursion for occupancy $\omega _ { n , t } ^ { q } \ ( 5 )$ . Fortunately, the MRP formulation gives the gradient in a temporal-diference (TD) (Sutton, 1988) form that uses only local derivatives.

For a fixed target-supported x, consider the MRP with transitions (4) and costs $c _ { n , t } ^ { q } ( x _ { < t } )$ from (7). Define the value function $V _ { n , t } ^ { q } ( x )$ , the expected cumulative cost starting from state $( n , t )$ . The value function satisfies the Bellman equation (Bellman, 1966)

$$
V _ { n , t } ^ { q } ( x ) = c _ { n , t } ^ { q } ( x _ { < t } ) + A _ { n , t } ^ { q } ( x _ { \leq t } ) V _ { n , t + 1 } ^ { q } ( x ) + R _ { n , t } ^ { q } ( x _ { \leq t } ) V _ { t , t + 1 } ^ { q } ( x ) , \qquad 0 \leq n < t \leq L ,\tag{12}
$$

with the terminal condition $V _ { n , L + 1 } ^ { q } ( x ) = 0$ for all $0 \le n \le L$ . Unlike the cost and occupancy, the value function $V _ { n . t } ^ { q } ( x )$ depends on the whole sequence x. The value function of the initial state gives the persequence EDR loss:

$$
V _ { 0 , 1 } ^ { q } ( x ) = \widehat { \mathcal { L } } _ { \mathrm { E D R } } ( q ; x ) - 1 .\tag{13}
$$

We now present the gradient of the per-sequence EDR loss in the following theorem. The proofs of (13) and Theorem 2 are given in Appendix B.4.

Theorem 2 (EDR gradient). We parametrize the drafter $q = q _ { \theta }$ by neural network parameters θ. For any target-supported sequence x, the gradient of the per-sequence EDR loss (10) is given by

$$
\nabla _ { \theta } \widehat { L } _ { \mathrm { E D R } } \big ( q _ { \theta } ; x \big ) = \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } \big ( x _ { < t } \big ) \left[ \nabla _ { \theta } c _ { n , t } ^ { q } \big ( x _ { < t } \big ) + \nabla _ { \theta } A _ { n , t } ^ { q } \big ( x _ { \le t } \big ) \left( V _ { n , t + 1 } ^ { q } ( x ) - V _ { t , t + 1 } ^ { q } ( x ) \right) \right] .\tag{14}
$$

As a result, one has

$$
\begin{array} { r } { \mathbb { E } _ { X \sim p } \left[ \nabla _ { \theta } \widehat { \mathcal { L } } _ { \mathrm { E D R } } ( q _ { \theta } ; X ) \right] = \nabla _ { \theta } \mathcal { L } _ { \mathrm { E D R } } ( q _ { \theta } ) . } \end{array}\tag{15}
$$

The gradient has a standard one-step TD form, separated into two contributions: the immediate efect on the local rejection cost, and the downstream efect of changing the acceptance probability. The latter is weighted by the value gap, which measures the diference between the cumulative costs of the two possible subsequent states. The occupancies account for how often each state is reached.

Implementation. We compute the occupancies $\omega _ { n , \cdot } ^ { q }$ by the forward recursion (5) and the value functions $V _ { n , t } ^ { q }$ by the backward Bellman recursion (12). Since they enter the gradient expression (14) only as weights, we hold them fixed during diferentiation. Gradients are required only through the local quantities $c _ { n , t } ^ { q }$ and $A _ { n , t } ^ { q }$

By (15), averaging (14) over a batch of i.i.d. target rollouts gives an unbiased stochastic gradient of EDR, which directly optimizes the expected number of decoding rounds. One EDR training step is summarized in Algorithm 1.

## 3.4 Exact Ofline Evaluation

The MRP formulation also yields exact ofline evaluators for any drafter, computed from target-model rollouts alone without running speculative decoding. We first give an unbiased estimator of the expected number of decoding rounds, from which we obtain an estimator of the MAL. We then extend it to a finer-grained statistic, the expected number of rounds that commit at least k tokens, which yields an estimator of the position-wise acceptance rate.

## 3.4.1 Expected Decoding Rounds and MAL

By Proposition 1, the expected number of decoding rounds conditioned on the output $X = x$ equals the sum of the diagonal occupancies. For a target-supported sequence $x = ( x _ { 1 : L } , < \mathtt { E 0 S } > )$ , define

$$
\tau _ { \mathrm { o f f } } ^ { q } ( x ) : = \sum _ { t = 0 } ^ { L } \omega _ { t , t + 1 } ^ { q } ( x _ { \le t } ) .\tag{16}
$$

This quantity averages out all randomness of speculative decoding given $X = x$ , namely draft sampling and verifier randomness, and is computed exactly by a single pass of the forward occupancy recursion. Given a dataset of target rollouts $\mathcal { D } = \{ \boldsymbol { x } ^ { ( i ) } = ( x _ { 1 : L _ { i } } ^ { ( i ) } , \boldsymbol { < } \mathrm { E 0 S > } ) \} _ { i = 1 } ^ { N }$ independently sampled from $\begin{array} { r } { x ^ { ( i ) } \sim p , } \end{array}$ we estimate the expected number of rounds and the MAL by

$$
\widehat { \mathcal T } _ { \mathcal D } ( q ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \tau _ { \mathrm { o f f } } ^ { q } ( x ^ { ( i ) } ) \qquad \mathrm { ~ a n d ~ } \qquad \widehat { \mathsf { M A L } } _ { \mathcal D } ( q ) : = \frac { N ^ { - 1 } \sum _ { i = 1 } ^ { N } ( L _ { i } + 1 ) } { \widehat { \mathcal T } _ { \mathcal D } ( q ) } .\tag{17}
$$

Since $\mathbb { E } [ \mathcal { T } ^ { q } ] = \mathbb { E } _ { X \sim p } [ \tau ^ { q } ( X ) ] , \widehat { \mathcal { T } } _ { \mathcal { D } } ( q )$ is an unbiased estimator of $\mathbb { E } [ \mathcal { T } ^ { q } ]$ , and $\widehat { { \sf M A L } } _ { \mathcal { D } } ( q )$ , a ratio of sample means, is a consistent estimator of MAL(q). Moreover, since the rollouts do not depend on the drafter, all drafters can be evaluated on the same D. This yields paired comparisons free of the rollout variation that arises when each drafter is evaluated on independently generated outputs.

## 3.4.2 Rounds Committing at Least k Tokens and Acceptance Rate

A finer-grained statistic is the number of rounds whose committed length, i.e., the number of tokens committed in the round including the correction or bonus token, is at least k. Let $ { \mathcal { T } } _ { \geq k } ^ { q }$ denote this number in a speculative decoding run. Since a round commits at most $B + 1$ tokens, we consider $1 \leq k \leq B + 1$ . A round starting after position t commits at least k tokens if and only if its first $k - 1$ drafts are accepted, i.e., it visits state $( t , t + k )$ . This observation generalizes Proposition 1 as follows; see Appendix B.2 for the proof.

Proposition 2. For any target-supported sequence $x = ( x _ { 1 : L } , < E \bar { O } S > )$ , any drafter $q ,$ and any $1 \leq k \leq B + 1$ the expected number of rounds committing at least k tokens to produce x is

$$
\tau _ { \geq k } ^ { q } ( x ) : = \mathbb { E } \big [ \mathcal { T } _ { \geq k } ^ { q } \big | X = x \big ] = \sum _ { n = 0 } ^ { L + 1 - k } \omega _ { n , n + k } ^ { q } ( x _ { < n + k } ) .\tag{18}
$$

Taking $k = 1$ recovers Proposition 1, so that $\tau _ { \mathrm { o f f } } ^ { q } = \tau _ { > 1 } ^ { q }$ . For $1 \leq k \leq B$ , we define the acceptance rate at draft position k as

$$
\mathrm { A R } _ { k } ( q ) : = \frac { \mathbb { E } [ T _ { \geq k + 1 } ^ { q } ] } { \mathbb { E } [ T _ { \geq k } ^ { q } ] } ,\tag{19}
$$

the fraction of rounds reaching draft position k in which the k-th draft is also accepted.<sup>1</sup> Using the same dataset $\mathcal { D } ,$ we estimate these quantities by

$$
\widehat { \mathcal { T } } _ { \geq k , \mathcal { D } } ( q ) : = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \tau _ { \geq k } ^ { q } ( x ^ { ( i ) } ) \qquad \mathrm { ~ a n d ~ } \qquad \widehat { \mathrm { A R } } _ { k , \mathcal { D } } ( q ) : = \frac { \widehat { \mathcal { T } } _ { \geq k + 1 , \mathcal { D } } ( q ) } { \widehat { \mathcal { T } } _ { \geq k , \mathcal { D } } ( q ) } .\tag{20}
$$

As before, $\widehat { \mathcal { T } } _ { \geq k , \mathcal { D } } ( q )$ is unbiased for $\mathbb { E } [ \mathcal { T } _ { \geq k } ^ { q } ]$ , and $\widehat { \mathrm { A R } } _ { k , \mathcal { D } } ( q )$ is consistent for $\operatorname { A R } _ { k } ( q )$

## 3.5 Suboptimality of Block-Local Surrogates

Finally, we show that maximizing accepted length independently within each block can be strictly suboptimal for the global MAL. Since maximizing MAL is equivalent to minimizing the expected number of decoding rounds (see Section 2), which EDR equals exactly (Theorem 1), this theoretically justifies optimizing the EDR objective rather than block-local surrogates.

We consider semi-AR drafters that condition on earlier drafts in the block only through the immediately preceding token, i.e., $q _ { n , t } ( \cdot  { | } \ x _ { 1 : n } ; x _ { n + 1 : t - 1 } ) = q _ { n , t } ( \cdot  { | } \ x _ { 1 : n } ; x _ { t - 1 } )$ , referred to as first-order drafters. This class includes existing semi-AR drafters such as DSpark (Cheng et al., 2026) and DFly (Liu et al., 2026).

The following theorem shows that even when the expected accepted length is optimized in every round, the resulting MAL can be a constant factor below the best achievable within the same drafter class. The construction and proof are deferred to Appendix B.5.

Theorem 3. For any block size $B \geq 3 .$ , there exists a target distribution p such that, for any first-order drafter qˆ that maximizes the expected accepted length in every round, there exists another first-order drafter $q ^ { * }$ satisfying

$$
\frac { \mathsf { M A L } ( q ^ { * } ) } { \mathsf { M A L } ( \hat { q } ) } > 1 + \frac { 1 } { 3 B + 9 } .
$$

The gap arises from the cross-round interaction ignored by block-local objectives. The draft distribution in the current round determines not only how many tokens are accepted in that round, but also where the next round starts. Consequently, a drafter that is locally optimal for the current round can induce a less favorable distribution of future round starts and result in a lower global MAL.

## 4 Related Work

## 4.1 Literature Review

Drafter architectures for speculative decoding. The latency per generated token depends on the accepted length per round as well as the costs of drafting and verification. Architectural work therefore seeks to increase accepted length while keeping drafting inexpensive. Medusa (Cai et al., 2024) attaches multiple prediction heads to target-model features, while Hydra (Ankner et al., 2024) introduces sequential dependencies between these heads to improve draft accuracy. EAGLE-3 (Li et al., 2026d) combines features from multiple target layers. To reduce sequential drafting overhead, parallel and difusion-based drafters generate multiple proposals in a single forward pass. DFlash (Chen et al., 2026a) uses a lightweight blockdifusion drafter conditioned on target-model context features. DifuSpec (Li et al., 2026a) repurposes a pretrained difusion language model as a parallel drafter and uses causal-consistency path search to construct proposals for autoregressive verification. To improve within-block consistency without returning to fully autoregressive drafting, DSpark (Cheng et al., 2026) and DFly (Liu et al., 2026) combine parallel backbones with lightweight sequential heads that condition on preceding draft tokens.

Training objectives for speculative drafters. Drafter training commonly uses cross-entropy or KLdivergence objectives to align draft predictions with the target distribution (Li et al., 2026d; Chen et al., 2026a). However, distributional matching under these losses does not necessarily maximize token acceptance when drafter capacity is limited. LK losses (Samarin et al., 2026) address this distinction through acceptanceoriented objectives. For multi-token drafting, the first rejection invalidates all subsequent proposals in the block, making the contributions of diferent positions unequal. DFlash (Chen et al., 2026a) accounts for this asymmetry using fixed exponentially decaying position weights. D-PACE (Wu et al., 2026) replaces this fixed schedule with adaptive weights derived from a diferentiable accepted-length surrogate based on draft confidences. The end-to-end (E2E) multi-step TV loss (Li et al., 2026c) moves beyond independent token-level matching by incorporating the multiplicative structure of acceptance across positions within a block. Recent work addresses the mismatch between drafter training and inference by incorporating verification feedback. Draft-OPD (Lei et al., 2026) replays draft proposals from inference-time anchors and applies acceptance-dependent distillation losses. Verification-Aware Training (Gu et al., 2026) uses simulated verification outcomes to supervise an auxiliary head and adapt position weights. Although these methods incorporate inference-time behavior into training, their objectives remain surrogates for MAL.

Difusion language models. Difusion language models (DLMs) (Nie et al., 2025; Inception Labs et al., 2025; Ye et al., 2025) ofer an alternative to AR generation by iteratively refining sequences with bidirectional context, enabling parallel token generation. Continuous DLMs (Li et al., 2022; Chen et al., 2026b; Li et al., 2026b) extend continuous difusion models (Ho et al., 2020; Song et al., 2020b,a) and flow matching (Lipman et al., 2022) to token embedding spaces. Starting from Gaussian noise, they generate continuous token representations through learned denoising or flow dynamics, then map these representations to discrete tokens. This framework facilitates adapting established techniques for controlling and accelerating difusion sampling, including classifier-free guidance (Ho and Salimans, 2022), self-conditioning (Chen et al., 2022), few-step solvers (Lu et al., 2022; Li and Cai, 2024; Jiao et al., 2026), and distillation (Song and Dhariwal, 2024; Yin et al., 2024b). Discrete DLMs (Lou et al., 2023; Sahoo et al., 2024; Shi et al., 2024) operate directly in token space using categorical corruption. Masked DLMs belong to this category and corrupt sequences by replacing tokens with a special mask token. Then it generate text by progressively unmasking multiple positions in parallel. This formulation underlies many parallel and semi-AR drafters. However, conditionally independent sampling within an update can overlook dependencies among simultaneously generated tokens, creating a trade-of between generation quality and sampling eficiency (Li and Cai, 2025; Chen et al., 2025; Dmitriev et al., 2026; Zhao and Cai, 2026; Cai and Li, 2026).

## 4.2 Comparison of Training Objectives

In this section, we present a unified summary of existing drafter training objectives in the literature, together with our proposed EDR objective.

Let $X = ( X _ { 1 } , \dots , X _ { L } , < \mathtt { E O S } > )$ be a rollout sequence sampled from the target distribution, and let B be the number of proposed draft tokens in a block. The block is drafted from the committed prefix $X _ { 1 : n } ; \ q _ { n , t }$ and $p _ { t }$ denote the draft and target distributions at position $t ,$ respectively. We suppress their dependence on the sampled sequence for brevity.

Up to additive constants and positive scaling factors, the objectives considered here can be written as

$$
\mathcal { L } ( q ) = \mathbb { E } _ { X \sim p } \left[ \sum _ { n = 0 } ^ { L } a _ { n } \sum _ { t = n + 1 } ^ { \operatorname* { m i n } \{ n + B + 1 , L \} } b _ { n , t } \ell _ { n , t } \right] .\tag{21}
$$

This template separates each objective into three components:

• Per-position loss $\ell _ { n , t }$ specifies the local training signal at position t for a block drafted from anchor $n .$ For token-level alignment surrogates, $\ell _ { n , t }$ measures the discrepancy between the draft distribution $q _ { n , t }$ and the target $p _ { t } .$ , whereas for acceptance-aware objectives, $\ell _ { n , t }$ is the surrogate for the rejection probability at position t. All objectives except EDR set $\ell _ { n , n + B + 1 } = 0$ , since no draft token is proposed at the bonus position. EDR keeps this term because a round that reaches the bonus position always ends there $\left( A _ { n , n + B + 1 } ^ { q } = 0 \right)$

Table 1: Drafter training objectives as instances of the template in (21).
<table><tr><td>Objective</td><td>Per-position loss  $\ell _ { n , t }$ </td><td>Block position weight  $b _ { n , t }$ </td><td>Anchor weight  $a _ { n }$ </td></tr><tr><td>Token-level surrogates</td><td></td><td></td><td></td></tr><tr><td>CE</td><td> $- \log q _ { n , t } ( x _ { t } )$ </td><td>1</td><td>1</td></tr><tr><td>KL</td><td> $\mathsf { K L } ( p _ { t } \parallel q _ { n , t } )$ </td><td>1</td><td>1</td></tr><tr><td>TV</td><td> $\mathsf { T V } ( p _ { t } , q _ { n , t } )$ </td><td>1</td><td>1</td></tr><tr><td>LK (Samarin et al., 2026)</td><td> $\big ( \lambda \mathsf { K L } + ( 1 - \lambda ) \mathsf { T V } \big ) ,$   $\lambda \dot { = } \exp \bigl ( - \kappa \operatorname { s g } ( 1 - \dot { \mathsf { T V } } ) \bigr )$ </td><td>1</td><td>1</td></tr><tr><td>Block-level surrogates</td><td></td><td></td><td></td></tr><tr><td>Exponential decay (Chen et al., 2026a)</td><td> $\mathrm { C E } / \mathrm { K L } / \mathrm { T V } / \mathrm { L K }$  or a weighted combination</td><td> $\exp \bigl ( - ( t - n - 1 ) / \eta \bigr )$ </td><td>1</td></tr><tr><td>D-PACE (Wu et al., 2026)</td><td> $\mathrm { C E } / \mathrm { K L } / \mathrm { T V } / \mathrm { L K }$  or a weighted combination</td><td> $\operatorname { s g } \Big ( \sum _ { \mathbf { \mu } } ^ { B } \ \prod _ { i = 1 } ^ { n + m } \widetilde { q } _ { n , s } ( x _ { s } ) \Big )$  m=t−n s=n+1  $\widetilde { q } _ { n , s } = \big ( 1 - \overset { \cdot } { \alpha } \big ) q _ { n , s } + \alpha$ </td><td>1</td></tr><tr><td>E2E-TV (Li et al., 2026c)</td><td> $( \mathsf { T V } ( p _ { t } , q _ { n , t } ) - 1 )$ </td><td> $\prod _ { s = n + 1 } ^ { t - 1 } \big ( 1 - \mathsf { T V } ( p _ { s } , q _ { n , s } ) \big )$ </td><td>1</td></tr><tr><td>Exact objective</td><td></td><td></td><td></td></tr><tr><td>EDR</td><td> $c _ { n , t } ^ { q }$ </td><td> $\prod _ { s = n + 1 } ^ { t - 1 } A _ { n , s } ^ { q }$ </td><td> $\omega _ { n , n + 1 } ^ { q }$ </td></tr><tr><td>(ours)</td><td></td><td></td><td></td></tr><tr><td>EDR (TD form) (ours)</td><td> $c _ { n , t } ^ { q } + A _ { n , t } ^ { q } \mathrm { s g } \left( V _ { n , t + 1 } ^ { q } - V _ { t , t + 1 } ^ { q } \right)$ </td><td> $\mathrm { s g } \Big ( \prod _ { s = n + 1 } ^ { t - 1 } A _ { n , s } ^ { q } \Big )$ </td><td> $\mathrm { s g } \big ( \omega _ { n , n + 1 } ^ { q } \big )$ </td></tr></table>

• Block position weight $b _ { n , t }$ sets how much position t counts relative to the other positions in the same block. Later draft tokens are verified only if all earlier draft tokens in the block are accepted, so their contributions depend on acceptance at preceding positions. Non-uniform position weights $b _ { n , t }$ reflect this asymmetry between positions.

• Anchor weight $a _ { n }$ sets how much the block drafted from anchor n counts relative to the blocks drafted from other anchors. All objectives except EDR weight anchors equally, $\mathrm { i } . \mathrm { e } . , a _ { n } = 1$

Table 1 summarizes the three components of each objective. Here, $\operatorname { s g } ( \cdot )$ denotes the stop-gradient operator: $\operatorname { s g } ( z )$ has the same value as z, but it is treated as a constant when taking gradients with respect to the drafter parameters.

We discuss these three groups below.

Token-level surrogates. Token-level surrogates set $b _ { n , t } = a _ { n } = 1$ and align the draft distribution $q _ { n , t }$ with the target $p _ { t }$ at each position t independently. CE and KL both minimize a KL divergence between a reference distribution and $q _ { n , t } \mathrm { . }$ the reference is the one-hot distribution at the sampled target token $x _ { t }$ for CE and the full target distribution $p _ { t }$ for KL. TV uses the TV distance, which equals the rejection probability at position t. LK interpolates between KL and TV using an acceptance-dependent coeficient λ that balances acceptance awareness against ease of optimization. As the acceptance probability $1 - \mathsf { T V } ( p _ { t } , q _ { n , t } )$ increases, λ decreases, so LK relies mainly on the KL term that is easier to optimize when acceptance is low and moves weight to the TV term as acceptance improves.

Block-level surrogates. Block-level surrogates extend token-level surrogates with nonuniform block position weights $b _ { n , t } ,$ , which reflect the asymmetry between positions within a block. Later positions receive smaller weights because they are verified less often—a position is verified only if all earlier draft tokens in its block are accepted.

Exponential decay and D-PACE multiply a token-level loss by a weight under stop-gradient. The weight is a fixed schedule in exponential decay and a confidence-based surrogate of the accepted length in D-PACE. E2E uses $\ell _ { n , t } = \mathsf { T V } ( p _ { t } , q _ { n , t } ) - 1$ at draft positions and $\begin{array} { r } { b _ { n , t } = \prod _ { s = n + 1 } ^ { t - 1 } \hat { ( 1 - \mathsf { T V } ( p _ { s } , q _ { n , s } ) ) } } \end{array}$ without stop-gradient. Therefore, the inner sum in (21) is given by

$$
- \sum _ { k = 1 } ^ { \operatorname* { m i n } \{ B , L + 1 - n \} } \prod _ { s = n + 1 } ^ { n + k } \big ( 1 - \mathsf { T V } ( p _ { s } , q _ { n , s } ) \big ) ,
$$

the negative of a surrogate for the expected accepted length of the block drafted from anchor n. In contrast, the block position weight of EDR is the conditional survival probability $\begin{array} { r } { b _ { n , t } = \prod _ { s = n + 1 } ^ { t - 1 } A _ { n , } ^ { q } } \end{array}$ <sub>,s</sub> along the target rollout. It is exact in expectation, whereas the product of 1 − TV terms in E2E is not.

Like token-level surrogates, block-level surrogates ignore cross-round coupling, treating each block separately and assigning the same weight $a _ { n } = 1$ to all anchors.

EDR. For EDR, the three components are the EOS-aware local rejection cost $\ell _ { n , t } = c _ { n , t } ^ { q }$ , the within-round survival probability $\begin{array} { r } { b _ { n , t } = \prod _ { s = n + 1 } ^ { t - 1 } A _ { n , s } ^ { q } } \end{array}$ , and the anchor weight $a _ { n } = \omega _ { n , n + 1 } ^ { q }$ . By the recursion in (5a), we have

$$
a _ { n } b _ { n , t } = \omega _ { n , n + 1 } ^ { q } \prod _ { s = n + 1 } ^ { t - 1 } A _ { n , s } ^ { q } = \omega _ { n , t } ^ { q } ,
$$

which is the probability, conditioned on the target rollout, that decoding visits state $( n , t )$ . Thus, with these choices, the weighted sum inside the expectation in (21), plus the constant 1, equals the per-sequence EDR loss (10). Taking the expectation over target rollouts gives the expected number of decoding rounds.

Among the objectives in Table 1, only EDR has drafter-dependent anchor weights. These weights capture how acceptance decisions change where later rounds start, that is, the cross-round coupling that the surrogates above ignore.

EDR in TD form. The TD form of EDR is a diferent scalar loss, but it has the same gradient with respect to the drafter parameters as the EDR loss above (see Theorem 2). It applies stop-gradient to both the block position weights and the anchor weights, and it adds the term $A _ { n , t } ^ { q } \operatorname { s g } \left( V _ { n , t + 1 } ^ { q } - V _ { t , t + 1 } ^ { q } \right)$ to the per-position loss. Gradients therefore flow only through $c _ { n , t } ^ { q }$ and $A _ { n , t } ^ { q } ,$ , not through $\omega ^ { q }$ or $V ^ { q }$ , so training with the TD form is more eficient. We use this form in training (see Algorithm 1).

## 5 Experiments

We evaluate whether directly optimizing EDR improves self-speculative decoding over existing block-local training objectives. We consider two SOTA self-speculative drafters, DSpark (Cheng et al., 2026) and DFly (Liu et al., 2026), paired with Qwen3 (Yang et al., 2025) target models of diferent scales.

## 5.1 Experimental Setup

Models and sampling. We evaluate two target–drafter pairs: Qwen3-4B with DSpark and Qwen3-8B with DFly. We set block size B = 7 for both drafters. For Qwen3-4B, target trajectories are sampled with temperature $0 . 7 , \mathrm { t o p } \mathrm { - } p = 0 . 8$ , and $\mathrm { t o p } { - } k = 2 0$ , while the DSpark proposal distribution uses temperature 0.7 without top-p or top-k truncation. For Qwen3-8B, both the target and DFly drafter use temperature 1.0 without top-p or top-k truncation. Thinking mode is disable in all settings.

Table 2: One-epoch finetuning results. We report mean accepted length (MAL; higher is better) after one epoch of draft-model finetuning. Target and draft sampling configurations in evaluation are the same as in training. Block size B = 7 for all drafters. Bold marks the best finetuned result.
<table><tr><td rowspan="2">Target</td><td rowspan="2">Drafter (B = 7)</td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="3">Chat</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>GSM8K MATH AIME25 MBPP HumanEval LCB MT-Bench Alpaca Arena-Hard</td><td></td></tr><tr><td rowspan="3">Qwen3-4B (T = 0.7 top-p = 0.8</td><td>DSpark</td><td>6.23</td><td>6.12</td><td>5.46</td><td>5.28</td><td>5.52</td><td>5.46</td><td>3.70</td><td>3.62</td><td>3.59</td></tr><tr><td>E2E</td><td>6.32</td><td>6.20</td><td>5.57</td><td>5.34</td><td>5.58</td><td>5.54</td><td>3.75</td><td>3.66</td><td>3.67</td></tr><tr><td>EDR (Ours)</td><td>6.32</td><td>6.20</td><td>5.59</td><td>5.34</td><td>5.59</td><td>5.56</td><td>3.78</td><td>3.70</td><td>3.70</td></tr><tr><td rowspan="3">Qwen3-8B (T = 1.0 no top-p/top-k)</td><td>DFly</td><td>6.30</td><td>5.94</td><td>5.15</td><td>5.26</td><td>5.58</td><td>5.23</td><td>3.57</td><td>3.45</td><td>3.18</td></tr><tr><td>E2E</td><td>6.34</td><td>5.98</td><td>5.18</td><td>5.32</td><td>5.62</td><td>5.26</td><td>3.61</td><td>3.49</td><td>3.21</td></tr><tr><td>EDR (Ours)</td><td>6.35</td><td>6.00</td><td>5.22</td><td>5.32</td><td>5.63</td><td>5.29</td><td>3.65</td><td>3.52</td><td>3.25</td></tr></table>

Training objectives. Starting from the corresponding publicly released draft-model checkpoint, we finetune only the drafter parameters. We compare our exact EDR objective against the end-to-end TV (E2E) block-local objective (Li et al., 2026c), using the same model architecture, training data, optimizer configuration, and number of update steps for both objectives. For all methods, we use at most 512 training anchors per trajectory. For EDR, we use importance sampling to select training anchors. For details, see Appendix C.2.

Training data. We train on Open-PerfectBlend (Xu et al., 2024). Rather than using reference answers, we generate a fresh trajectory for each prompt from the target model using the same target sampling configuration as evaluation. Each drafter is finetuned for one epoch with batch size 96.

Evaluation. We use the same evaluation prompts as DSpark (Cheng et al., 2026) on nine benchmarks spanning math (Cobbe et al., 2021; Hendrycks et al., 2021; Zhang and Math-AI, 2025), code (Austin et al., 2021; Chen et al., 2021; Jain et al., 2025), and chat (Zheng et al., 2023; Taori et al., 2023; Li et al., 2024a). We generate at least 1,500 target trajectories per benchmark (multiple trajectories for each prompt) under the same sampling configurations as in training. We compute MAL using the ofline estimator introduced in Section 3.4. All methods for the same target model and sampling configurations are evaluated on exactly the same target trajectories. This provides paired comparisons across training objectives and removes additional variance from independently sampled evaluation outputs.

All training and evaluation are performed on a single NVIDIA H200 GPU or H100 GPU. For more experimental details, see Appendix D.

## 5.2 Experimental Results

Table 2 shows that EDR consistently improves speculative decoding across both DSpark and DFly without modifying the drafter architecture. Compared with E2E finetuning, EDR achieves higher or equal MAL across all benchmarks, with larger gains on AIME25, LiveCodeBench, and the chat benchmarks. These results suggest that directly optimizing the global expected number of decoding rounds provides additional gains beyond block-local end-to-end surrogates, and that the improvement transfers across diferent semi-AR drafter architectures and target-model scales.

## 6 Conclusion

We introduced Expected Decoding Rounds (EDR), an exact global training objective for parallel and semi-AR speculative drafters. By formulating speculative decoding as a Markov reward process conditioned on target-model outputs, we derived a tractable gradient for directly minimizing the expected number of verification rounds, together with an exact ofline evaluator for paired draft-model comparison. Empirically, EDR consistently improves two strong drafters, DSpark and DFly, across nine benchmarks without modifying their architectures or inference procedures, and outperforms the E2E training objective in overall mean accepted length. These results suggest that explicitly accounting for cross-round dependencies provides a principled and practical direction for training more eficient parallel speculative draft models.

We conclude with several directions for future work. First, our current framework focuses on a fixed draft block size. However, under high-concurrency serving, verifying long draft blocks can become ineficient, especially when many later proposals are likely to be rejected. Adaptive block sizing can reduce unnecessary verification by shortening the draft block when confidence is low, thereby further reducing inference latency (Hu et al., 2026; Huang et al., 2024). An important direction is to extend our framework to adaptive block sizes and to study whether the optimal block size can be selected dynamically for each decoding round. Second, EDR requires computing occupancy weights ω and value functions V for multiple anchor positions. This is more computationally expensive than block-local objectives. It would be interesting to explore more eficient ways to utilize the occupancy and value computations. Finally, as shown in Table 2, MAL varies substantially across datasets, suggesting that some target distributions are intrinsically more dificult to draft than others. Our current work focuses on optimizing the drafter given a target distribution, but does not provide a complete theoretical characterization of this dificulty. A promising direction is to identify properties of the target distribution that determine the best achievable speculative decoding eficiency.

## Acknowledgements

C. Cai is supported by the NSF CAREER Award CCF-2541600 and the NSF grant DMS-2515333.

## References

Ankner, Z., Parthasarathy, R., Nrusimha, A., Rinard, C., Ragan-Kelley, J., and Brandon, W. (2024). Hydra: Sequentially-dependent draft heads for medusa decoding. arXiv preprint arXiv:2402.05109.

Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. (2021). Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Bellman, R. (1966). Dynamic programming. science, 153(3731):34–37.

Cai, C. and Li, G. (2026). Confidence-based decoding is provably eficient for difusion language models. arXiv preprint arXiv:2603.22248.

Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T. (2024). Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774.

Chen, C., Borgeaud, S., Irving, G., Lespiau, J.-B., Sifre, L., and Jumper, J. (2023). Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318.

Chen, J., Liang, Y., and Liu, Z. (2026a). Dflash: Block difusion for flash speculative decoding. arXiv preprint arXiv:2602.06036.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. D. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374.

Chen, S., Cong, K., and Li, J. (2025). Optimal inference schedules for masked difusion models. arXiv preprint arXiv:2511.04647.

Chen, T., Zhang, R., and Hinton, G. (2022). Analog bits: Generating discrete data using difusion models with self-conditioning. arXiv preprint arXiv:2208.04202.

Chen, Y., Liang, C., Sui, H., Guo, R., Cheng, C., You, J., and Liu, G. (2026b). Langflow: Continuous difusion rivals discrete in language modeling. arXiv preprint arXiv:2604.11748.

Cheng, X., Yu, X., Shao, C., Li, J., Xiong, Y., Qian, Y., Zhu, J., Ma, S., Zhang, X., Ye, J., et al. (2026). Dspark: Confidence-scheduled speculative decoding with semi-autoregressive generation. arXiv preprint arXiv:2607.05147.

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. (2021). Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Dmitriev, D., Huang, Z., and Wei, Y. (2026). Eficient sampling with discrete difusion models: Sharp and adaptive guarantees. arXiv preprint arXiv:2602.15008.

Gu, G., Heo, B., Jun, H., Kang, Y., Lee, S., Yun, S., and Han, D. (2026). Verification-aware training for speculative decoding. arXiv preprint arXiv:2608.30135.

Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D., and Steinhardt, J. (2021). Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Ho, J., Jain, A., and Abbeel, P. (2020). Denoising difusion probabilistic models. Advances in neural information processing systems, 33:6840–6851.

Ho, J. and Salimans, T. (2022). Classifier-free difusion guidance. arXiv preprint arXiv:2207.12598.

Howard, R. (1971). Dynamic Probabilistic Systems: Semi-Markov and decision processes. Decision and Control Series. Wiley.

Hu, X., Shen, Y., Zhang, B., Zhang, H., Dai, J., Ge, S., Chen, L., Li, Y., and Wan, M. (2026). Echo: Elastic speculative decoding with sparse gating for high-concurrency scenarios. arXiv preprint arXiv:2604.09603.

Huang, K., Guo, X., and Wang, M. (2024). Specdec++: Boosting speculative decoding via adaptive candidate lengths. arXiv preprint arXiv:2405.19715.

Inception Labs, Khanna, S., Kharbanda, S., Li, S., Varma, H., Wang, E., Birnbaum, S., Luo, Z., Miraoui, Y., Palrecha, A., Ermon, S., Grover, A., and Kuleshov, V. (2025). Mercury: Ultra-fast language models based on difusion. arXiv preprint arXiv:2506.17298.

Jain, N., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. (2025). Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, volume 2025, pages 58791–58831.

Jiao, Y., Li, N., Cai, C., and Li, G. (2026). Are first-order difusion samplers really slower? a fast forwardvalue approach. In International Conference on Machine Learning. PMLR.

Lei, H., Li, Y., Zhang, H., Zhang, S., Cheng, Q., Qu, X., Cui, G., Zhou, B., Ding, N., Luo, Y., et al. (2026). Draft-opd: On-policy distillation for speculative draft models. arXiv preprint arXiv:2605.29343.

Leviathan, Y., Kalman, M., and Matias, Y. (2023). Fast inference from transformers via speculative decoding. In International conference on machine learning, pages 19274–19286. PMLR.

Li, G. and Cai, C. (2024). Provable acceleration for difusion models under minimal assumptions. arXiv preprint arXiv:2410.23285.

Li, G. and Cai, C. (2025). Breaking AR’s sampling bottleneck: Provable acceleration via difusion language models. In Advances in Neural Information Processing Systems, volume 38, pages 11700–11725.

Li, G., Fu, Z., Fang, M., Zhao, Q., Tang, M., Yuan, C., and Wang, J. (2026a). Difuspec: Unlocking difusion language models for speculative decoding. In Findings of the Association for Computational Linguistics: ACL 2026, pages 20896–20910.

Li, N., Jiao, Y., Cai, C., and Li, G. (2026b). Convergeflow: Language flow with provable convergence to token embeddings. arXiv preprint arXiv:2608.23551.

Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Wu, T., Zhu, B., Gonzalez, J. E., and Stoica, I. (2024a). From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939.

Li, X., Thickstun, J., Gulrajani, I., Liang, P. S., and Hashimoto, T. B. (2022). Difusion-lm improves controllable text generation. In Advances in Neural Information Processing Systems, volume 35, pages 4328–4343.

Li, Y., Jiang, H., Xu, Y., Yang, J., Zhang, Y., Cao, Y., Shen, Y., Zhou, F., Men, R., Zhang, J., et al. (2026c). Breaking entropy bounds: Accelerating rl training via mtp with rejection sampling. arXiv preprint arXiv:2606.12370.

Li, Y., Wei, F., Zhang, C., and Zhang, H. (2024b). Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077.

Li, Y., Wei, F., Zhang, C., and Zhang, H. (2026d). Eagle-3: Scaling up inference acceleration of large language models via training-time test. Advances in Neural Information Processing Systems, 38:136737–136756.

Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. (2022). Flow matching for generative modeling. arXiv preprint arXiv:2210.02747.

Liu, H., Cen, R., Shi, J., Qin, G., Zhang, J., Liu, T., Fan, R., Zhao, G., Xie, R., Zhang, K., et al. (2026). Angelspec: Towards real-world high performance inference with speculative decoding. arXiv preprint arXiv:2607.25852.

Lou, A., Meng, C., and Ermon, S. (2023). Discrete difusion modeling by estimating the ratios of the data distribution. arXiv preprint arXiv:2310.16834.

Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., and Zhu, J. (2022). Dpm-solver: A fast ode solver for difusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35:5775–5787.

Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. (2025). Large language difusion models. arXiv preprint arXiv:2502.09992.

Sahoo, S., Arriola, M., Schif, Y., Gokaslan, A., Marroquin, E., Chiu, J., Rush, A., and Kuleshov, V. (2024). Simple and efective masked difusion language models. Advances in Neural Information Processing Systems, 37:130136–130184.

Samarin, A., Krutikov, S., Shevtsov, A., Skvortsov, S., Fisin, F., and Golubev, A. (2026). Lk losses: Direct acceptance rate optimization for speculative decoding. arXiv preprint arXiv:2602.23881.

Shi, J., Han, K., Wang, Z., Doucet, A., and Titsias, M. (2024). Simplified and generalized masked difusion for discrete data. Advances in neural information processing systems, 37:103131–103167.

Song, J., Meng, C., and Ermon, S. (2020a). Denoising difusion implicit models. arXiv preprint arXiv:2010.02502.

Song, Y. and Dhariwal, P. (2024). Improved techniques for training consistency models. In International Conference on Learning Representations, volume 2024, pages 15078–15097.

Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. (2020b). Score-based generative modeling through stochastic diferential equations. arXiv preprint arXiv:2011.13456.

Sutton, R. S. (1988). Learning to predict by the methods of temporal diferences. Machine learning, 3(1):9– 44.

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. (2023). Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford\_ alpaca.

Wu, T., Yao, Y., Qi, Z., Zheng, H., Wang, Z., Ma, H., Liao, L., Lakkaraju, H., Li, J., and Du, Y. (2026). D-pace: Dynamic position-aware cross-entropy for parallel speculative drafting. arXiv preprint arXiv:2605.18810.

Xu, T., Helenowski, E., Sankararaman, K. A., Jin, D., Peng, K., Han, E., Nie, S., Zhu, C., Zhang, H., Zhou, W., et al. (2024). The perfect blend: Redefining rlhf with mixture of judges. arXiv preprint arXiv:2409.20370.

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. (2025). Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Ye, J., Xie, Z., Zheng, L., Gao, J., Wu, Z., Jiang, X., Li, Z., and Kong, L. (2025). Dream 7b: Difusion large language models. arXiv preprint arXiv:2508.15487.

Yin, M., Chen, M., Huang, K., and Wang, M. (2024a). A theoretical perspective for speculative decoding algorithm. Advances in Neural Information Processing Systems, 37:128082–128117.

Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W. T., and Park, T. (2024b). One-step difusion with distribution matching distillation. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6613–6623. IEEE.

Zhang, Y. and Math-AI, T. (2025). American invitational mathematics examination (aime) 2025.

Zhao, Y. and Cai, C. (2026). Adaptation to intrinsic dependence in difusion language models. arXiv preprint arXiv:2602.20126.

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. (2023). Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

## A Speculative Decoding Algorithm

Algorithm 2 summarizes the speculative decoding procedure we consider.

## B Proofs

## B.1 Markov Process Formulation in Section 3.1

In this section, we give a detailed derivation of the rejection-sampling transition dynamics conditioned on a fixed target-supported output sequence $x = ( x _ { 1 : L } , < \mathtt { E 0 S } > )$ . Specifically, we define the Markov chain states, derive the conditional transition probabilities, and prove that the occupancy $\omega _ { n , t } ^ { q } ( x )$ depends on x only through $x _ { < t }$ . Throughout, all probabilities refer to speculative decoding with a fixed drafter $q .$

States and decoding histories. For each position t reached by decoding, let $N _ { t }$ denote the round-start index of the round that produces $X _ { t }$ . Thus, immediately before processing position t, the state is

$$
S _ { t } = ( N _ { t } , t ) , \qquad S _ { 1 } = ( 0 , 1 ) .
$$

If $N _ { t } ~ = ~ n$ , then the current round started from the committed prefix $X _ { 1 : n }$ , and all preceding drafts in this round have been accepted. Consequently, their values equal $X _ { n + 1 : t - 1 }$ . For $t \leq n + B$ , position t is processed by draft verification; for $t = n + B + 1$ , it is processed by drawing a target bonus token. States with $t > n + B + 1$ are unreachable.

Let

$$
\mathcal { H } _ { t } : = \sigma ( X _ { < t } , N _ { 1 } , . . . , N _ { t } )
$$

denote the history of committed tokens and round-start indices before processing position t. We first analyze the next output token conditional on this history, without conditioning on the complete output sequence.

Algorithm 2 Speculative decoding with a semi-autoregressive drafter   
Require: Block size $B ,$ target distributions ${ \boldsymbol { p } } _ { t } ,$ draft distributions $q _ { n , t }$   
1: n $ 0 , \tau  0$ ▷ n: number of committed ordinary tokens   
2: while true do   
3: Draw a draft block $\widetilde { x } _ { n + 1 : n + B } \sim q _ { n , n + 1 } ( \cdot \mid x _ { 1 : n } ) \cdot \cdot \cdot q _ { n , n + B } ( \cdot \mid x _ { 1 : n } ; \widetilde { x } _ { n + 1 : n + B - 1 } )$   
4: Evaluate $p _ { t } \big ( \cdot \mid x _ { 1 : n } , \widetilde { x } _ { n + 1 : t - 1 } \big )$ for $t = n + 1 , \ldots , n + B + 1$ in one target-model pass   
5: $\tau  \tau + 1$   
6: rejected ← false   
7: for $t = n + 1 , \ldots , n + B$ do   
8: $p \gets p _ { t } \big ( \cdot \mid x _ { 1 : n } , \widetilde { x } _ { n + 1 : t - 1 } \big )$   
9: $q  q _ { n , t } ( \cdot \mid x _ { 1 : n } ; \widetilde { x } _ { n + 1 : t - 1 } )$   
10: Draw $U _ { t } \sim \mathrm { U n i f } [ 0 , 1 ]$   
11: if $U _ { t } \leq \operatorname* { m i n } \{ 1 , p ( \widetilde { x } _ { t } ) / q ( \widetilde { x } _ { t } ) \}$ then   
12: $x _ { t } \gets \widetilde { x } _ { t }$   
13: else   
14: Draw $x _ { t } \sim [ p - q ] _ { + } / { \mathsf { T V } } ( p , q )$   
15: rejected ← true   
16: end if   
17: if $x _ { t } = < \mathtt { E 0 S } >$ then   
18: return $\left( x _ { 1 : t - 1 } , < \mathtt { E } 0 \mathtt { S } > \right)$ and $\tau$   
19: end if   
20: if rejected then   
21: $n \gets t$   
22: break   
23: end if   
24: end for   
25: if not rejected then   
26: Draw $\widetilde { x } _ { n + B + 1 } \sim p _ { n + B + 1 } \big ( \cdot \mid x _ { 1 : n + B } \big )$ ▷ bonus token   
27: if $\widetilde { x } _ { n + B + 1 } =$ <EOS> then   
28: return $\left( x _ { 1 : n + B } , < \mathtt { E } 0 \mathtt { S } > \right)$ and $\tau$   
29: else   
30: $x _ { n + B + 1 }  \widetilde { x } _ { n + B + 1 }$   
31: n $ n + B + 1$   
32: end if   
33: end if   
34: end while

The local rejection-sampling law. Consider a history with $X _ { < t } = x _ { < t }$ and $N _ { t } = n$ at a draft position $t \leq n + B$ . For brevity, write

$$
p _ { t } ( \boldsymbol { v } ) : = p _ { t } ( \boldsymbol { v } \mid \boldsymbol { x } _ { < t } ) , \qquad q _ { t } ( \boldsymbol { v } ) : = q _ { n , t } ( \boldsymbol { v } \mid \boldsymbol { x } _ { 1 : n } ; \boldsymbol { x } _ { n + 1 : t - 1 } ) , \qquad \boldsymbol { v } \in \mathcal { V } .
$$

Let $\operatorname { a c c } _ { t }$ and rej<sub>t</sub> denote the acceptance and rejection events at this position. The verification rule gives

$$
\begin{array} { r l } & { \operatorname* { P r } ( X _ { t } = v , \operatorname { a c c } _ { t } \mid { \mathcal { H } } _ { t } ) = \operatorname* { m i n } \{ p _ { t } ( v ) , q _ { t } ( v ) \} , } \\ & { \operatorname* { P r } ( X _ { t } = v , \operatorname { r e j } _ { t } \mid { \mathcal { H } } _ { t } ) = [ p _ { t } ( v ) - q _ { t } ( v ) ] _ { + } . } \end{array}
$$

The first identity follows by multiplying the proposal probability by the verifier’s acceptance probability. For the second identity, rejection has total probability $\mathrm { T V } ( p _ { t } , q _ { t } )$ , and the correction token is drawn from the residual distribution $[ p _ { t } - q _ { t } ] _ { + } / \mathrm { T V } ( p _ { t } , q _ { t } )$ . When $\mathrm { T V } ( p _ { t } , q _ { t } ) = 0$ , rejection has probability zero and no residual draw is needed.

Adding the two identities yields

$$
\operatorname* { P r } ( X _ { t } = v \mid { \mathcal { H } } _ { t } ) = \operatorname* { m i n } \{ p _ { t } ( v ) , q _ { t } ( v ) \} + [ p _ { t } ( v ) - q _ { t } ( v ) ] _ { + } = p _ { t } ( v \mid X _ { < t } ) .\tag{22}
$$

At a bonus-token position, the same identity holds because the token is drawn directly from the target distribution. Therefore, conditional on any decoding history, the next committed token has the target conditional distribution, regardless of the current round-start index.

Transition kernels conditioning on the complete output. Equation (22) implies that, on any history with $X _ { < t } = x _ { < t }$ 2

$$
\operatorname* { P r } ( X = x \mid { \mathcal { H } } _ { t } ) = p ( x _ { t : L + 1 } \mid x _ { < t } ) : = \prod _ { s = t } ^ { L + 1 } p _ { s } ( x _ { s } \mid x _ { < s } ) .\tag{23}
$$

To see this rigorously, use backward induction along the fixed sequence x. $\mathrm { A t } t = L + 1$ , the required remaining output is just <EOS>, so the identity follows directly from (22). Suppose it holds at $t + 1$ for every possible decoding history. After committing $X _ { t } = x _ { t }$ , every possible resulting history has the same remaining-output probability $p ( x _ { t + 1 : L + 1 } \mid x _ { \leq t } )$ . Multiplying this probability by $\operatorname* { P r } ( X _ { t } = x _ { t } \mid { \mathcal { H } } _ { t } ) = p _ { t } ( x _ { t } \mid x _ { < t } )$ proves the identity at t.

In particular, after committing the same non-EOS token $x _ { t }$ , the future sufix has the same probability whether the token was accepted or produced by residual sampling. Thus, for $t \leq L$ at a draft position, Bayes’ rule gives

$$
\begin{array} { r l } & { \operatorname* { P r } ( \operatorname { a c c } _ { t } \mid \mathcal { H } _ { t } , X = x ) = \frac { \operatorname* { m i n } \{ p _ { t } ( x _ { t } ) , q _ { t } ( x _ { t } ) \} \ : p ( x _ { t + 1 : L + 1 } \mid x _ { \le t } ) } { p _ { t } ( x _ { t } ) \ : p ( x _ { t + 1 : L + 1 } \mid x _ { \le t } ) } } \\ & { \qquad = \frac { \operatorname* { m i n } \{ p _ { t } ( x _ { t } ) , q _ { t } ( x _ { t } ) \} } { p _ { t } ( x _ { t } ) } } \\ & { \qquad = \operatorname* { m i n } \bigg \{ 1 , \frac { q _ { n , t } ( x _ { t } \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } ) } { p _ { t } ( x _ { t } \mid x _ { < t } ) } \bigg \} } \\ & { \qquad = A _ { n , t } ^ { q } ( x _ { \le t } ) . } \end{array}
$$

All target-probability factors being cancelled are positive because x is target-supported. Similarly,

$$
\operatorname* { P r } ( \operatorname { r e j } _ { t } \mid { \mathcal { H } } _ { t } , X = x ) = 1 - A _ { n , t } ^ { q } ( x \leq t ) = R _ { n , t } ^ { q } ( x \leq t ) .
$$

Notice that the ratio here is $q _ { t } ( x _ { t } ) / p _ { t } ( x _ { t } )$ , rather than the verifier’s ratio $p _ { t } ( \widetilde { X } _ { t } ) / q _ { t } ( \widetilde { X } _ { t } )$ : we are conditioning on the committed output token, not on the proposed token.

If the token is accepted, the current round continues. If it is rejected, the next round starts after position t. At a bonus-token position $t = n + B + 1$ , the next round also starts after position t, provided that the bonus token is not EOS. Accordingly, setting

$$
A _ { n , t } ^ { q } = 0 , \qquad R _ { n , t } ^ { q } = 1 , \qquad t > n + B ,
$$

incorporates the bonus-token case into the same transition rule. At such a position, $R _ { n , t } ^ { q } = 1$ represents a forced round transition, not an actual draft rejection.

It follows that, under $\operatorname* { P r } ( \cdot \mid X = x )$ , the transitions for $t \leq L$ are

$$
( n , t ) \longrightarrow \left\{ \begin{array} { l l } { { ( n , t + 1 ) , } } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } A _ { n , t } ^ { q } ( x \leq t ) , } } \\ { { ( t , t + 1 ) , } } & { { \mathrm { w i t h ~ p r o b a b i l i t y ~ } R _ { n , t } ^ { q } ( x \leq t ) . } } \end{array} \right.
$$

With $x$ fixed, these probabilities depend on the past history only through the current state $( n , t )$ . This establishes the Markov property.

The conditional chain has states $0 \leq n < t \leq L + 1$ and is stopped at position $L + 1$ . All states $( n , L + 1 )$ are terminal for the round-count process: the final EOS is produced in the current round and does not start another round, regardless of whether it is accepted, sampled as a correction, or sampled as a bonus token.

Occupancy depends only on the committed prefix. Since the position index increases by one at every transition, state $( n , t )$ is visited exactly when $N _ { t } = n$ . Hence,

$$
\omega _ { n , t } ^ { q } ( x ) = \operatorname* { P r } ( N _ { t } = n \mid X = x ) .
$$

By (23), conditional on the committed prefix $x _ { < t } .$ , the probability of the remaining output $x _ { t : L + 1 }$ is the same for every possible round-start history. Therefore,

$$
\omega _ { n , t } ^ { q } ( x ) = \frac { \mathrm { P r } ( N _ { t } = n , X _ { < t } = x _ { < t } ) p ( x _ { t : L + 1 } \mid x _ { < t } ) } { \mathrm { P r } ( X _ { < t } = x _ { < t } ) p ( x _ { t : L + 1 } \mid x _ { < t } ) }
$$

$$
\begin{array} { r l } & { = \mathrm { P r } ( N _ { t } = n \mid X _ { < t } = x _ { < t } ) } \\ & { = : \omega _ { n , t } ^ { q } ( x _ { < t } ) . } \end{array}
$$

This proves that the occupancy depends only on $x _ { < t } .$ even though it was originally defined by conditioning on the complete output sequence.

Starting from $\omega _ { 0 , 1 } ^ { q } = 1$ , the conditional Markov transitions give the forward recursions

$$
\begin{array} { r l r l } & { \omega _ { n , t + 1 } ^ { q } ( x _ { \le t } ) = \omega _ { n , t } ^ { q } ( x _ { < t } ) A _ { n , t } ^ { q } ( x _ { \le t } ) , } & & { \quad \quad \quad 0 \le n < t \le L , } \\ & { \omega _ { t , t + 1 } ^ { q } ( x _ { \le t } ) = { \displaystyle \sum _ { n = 0 } ^ { t - 1 } } \omega _ { n , t } ^ { q } ( x _ { < t } ) R _ { n , t } ^ { q } ( x _ { \le t } ) , } & & { \quad \quad \quad 1 \le t \le L . } \end{array}
$$

The first recursion accounts for continuing the current round. The second sums over all states from which a new round can start after position t. These are precisely the occupancy recursions used in Section 3.1.

## B.2 Proofs of Proposition 1 and Proposition 2

We first prove Proposition 2, and then obtain Proposition 1 as its special case $k = 1$

Proof of Proposition 2. Fix $1 \leq k \leq B + 1$ . The random variable $\mathcal { T } _ { > k } ^ { q }$ counts the decoding rounds that commit at least k tokens. Conditioned on a fixed output sequence x, consider the round that starts after position n, i.e., from the committed prefix $x _ { 1 : n }$ . By the transition rule (4), a rejection transition always leads to a diagonal state, so a non-diagonal state $( n , t )$ with $t > n + 1$ can only be reached from $( n , t - 1 )$ by an acceptance. Hence state $( n , n + k )$ is visited exactly when this round has started and its drafts at positions $n + 1 , \ldots , n + k - 1$ have all been accepted. In that case, position $n + k$ is processed in the same round and commits one more token, which is an accepted draft, a correction token, a bonus token, or the termina <EOS>. Conversely, if the round commits at least k tokens, its first $k - 1$ committed tokens must be accepted drafts: a correction token or <EOS> ends the round, and a bonus token can only be the $( B + 1 )$ -th committed token. Therefore, the round starting after position n commits at least k tokens exactly when state $( n , n + k )$ is visited. It follows that

$$
\mathcal { T } _ { \geq k } ^ { q } = \sum _ { n = 0 } ^ { L + 1 - k } \mathbf { 1 } \{ \mathrm { s t a t e ~ } ( n , n + k ) \mathrm { ~ i s ~ v i s i t e d } \} .
$$

Conditioned on $X = x ,$ , the event that state $( n , n + k )$ is visited has probability $\omega _ { n , n + k } ^ { q } ( x _ { < n + k } )$ . Therefore, by linearity of expectation,

$$
\tau _ { \geq k } ^ { q } ( x ) = \mathbb { E } [ \mathcal { T } _ { \geq k } ^ { q } \mid X = x ] = \sum _ { n = 0 } ^ { L + 1 - k } \omega _ { n , n + k } ^ { q } ( x _ { < n + k } ) .
$$

Proof of Proposition 1. Every round commits at least one token, so $\mathcal T ^ { q } = \mathcal T _ { > 1 } ^ { q }$ . Equivalently, a round starts after position t exactly when state $( t , t + 1 )$ is visited. Taking $k = 1$ in Proposition 2 gives

$$
\tau ^ { q } ( \boldsymbol { x } ) = \mathbb { E } [ \mathcal { T } ^ { q } \mid \boldsymbol { X } = \boldsymbol { x } ] = \sum _ { t = 0 } ^ { L } \omega _ { t , t + 1 } ^ { q } ( \boldsymbol { x } _ { \le t } ) .
$$

Separating the term $t = 0$ , using $\omega _ { 0 , 1 } ^ { q } = 1$ , and substituting the occupancy recursion (5b) for $1 \leq t \leq L$ gives

$$
\tau ^ { q } ( x ) = \omega _ { 0 , 1 } ^ { q } + \sum _ { t = 1 } ^ { L } \omega _ { t , t + 1 } ^ { q } ( x _ { \leq t } ) = 1 + \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) R _ { n , t } ^ { q } ( x _ { \leq t } ) .
$$

## B.3 Proof of Theorem 1

To handle the randomness of the sequence length L, we rewrite Proposition 1 and take the expectation over $X \sim p .$

$$
\mathbb { E } [ \tau ^ { q } ( X ) ] = 1 + \sum _ { t \ge 1 } \sum _ { n = 0 } ^ { t - 1 } \mathbb { E } \big [ \mathbf { 1 } _ { \{ t \le L \} } \omega _ { n , t } ^ { q } ( X _ { < t } ) R _ { n , t } ^ { q } ( X _ { \le t } ) \big ] .\tag{24}
$$

Before $X _ { t }$ is revealed, $\omega _ { n , t } ^ { q } ( X _ { < t } )$ is determined by $X _ { < t }$ , and $t \leq L$ is exactly the event $X _ { t } \neq < \mathtt { E O S } >$ . Hence, conditioning on $X _ { < t }$ and using the definition of $c _ { n , t } ^ { q } ,$

$$
\begin{array} { r l } & { \mathbb { E } \big [ \mathbf { 1 } _ { \{ t \leq L \} } \omega _ { n , t } ^ { q } ( X _ { < t } ) R _ { n , t } ^ { q } ( X _ { \leq t } ) \mid X _ { < t } \big ] } \\ & { \qquad = \omega _ { n , t } ^ { q } ( X _ { < t } ) \operatorname* { P r } ( X _ { t } \neq { \tt { E B S } } > \mid X _ { < t } ) c _ { n , t } ^ { q } ( X _ { < t } ) } \\ & { \qquad = \mathbb { E } \big [ \mathbf { 1 } _ { \{ t \leq L \} } \omega _ { n , t } ^ { q } ( X _ { < t } ) c _ { n , t } ^ { q } ( X _ { < t } ) \mid X _ { < t } \big ] . } \end{array}\tag{25}
$$

Taking expectations and summing over n and t proves (9).

## B.4 Proof of Theorem 2

We first prove (13), which relates the initial-state value to the per-sequence EDR loss.

$$
\begin{array} { l } { { \displaystyle V _ { 0 , 1 } ^ { q } ( x ) = \sum _ { t = 1 } ^ { L } \left[ \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) V _ { n , t } ^ { q } ( x ) - \sum _ { n = 0 } ^ { t } \omega _ { n , t + 1 } ^ { q } ( x _ { \le t } ) V _ { n , t + 1 } ^ { q } ( x ) \right] } } \\ { { \displaystyle \quad \quad = \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) \left[ V _ { n , t } ^ { q } ( x ) - A _ { n , t } ^ { q } ( x _ { \le t } ) V _ { n , t + 1 } ^ { q } ( x ) - R _ { n , t } ^ { q } ( x _ { \le t } ) V _ { t , t + 1 } ^ { q } ( x ) \right] } } \\ { { \displaystyle \quad \quad = \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) c _ { n , t } ^ { q } ( x _ { < t } ) } } \\ { { \displaystyle \quad \quad = \widehat { \mathcal { L } } _ { \mathrm { E D R } } ( q _ { \ast } ; x ) - 1 . } } \end{array}
$$

Now we prove Theorem 2. Diferentiating the Bellman equation (12) at one state gives

$$
\begin{array} { r l } & { \nabla _ { \theta } V _ { n , t } ^ { q } = \underbrace { \nabla _ { \theta } c _ { n , t } ^ { q } + \nabla _ { \theta } A _ { n , t } ^ { q } \left( V _ { n , t + 1 } ^ { q } - V _ { t , t + 1 } ^ { q } \right) } _ { \mathrm { l o c a l ~ T D ~ g r a d i e n t } } } \\ & { ~ + A _ { n , t } ^ { q } \nabla _ { \theta } V _ { n , t + 1 } ^ { q } + R _ { n , t } ^ { q } \nabla _ { \theta } V _ { t , t + 1 } ^ { q } , } \end{array}\tag{26}
$$

where we used $R _ { n , t } ^ { q } = 1 - A _ { n , t } ^ { q }$ . Unrolling the last two terms from the initial state weights each local TD gradient by the probability of visiting that state, namely $\omega _ { n , t } ^ { q }$ . Apply (26) recursively from (0, 1). Every time a state $( n , t )$ is reached, its local term is accumulated once, while the two recursive terms propagate with transition probabilities $A _ { n , t } ^ { q }$ and $R _ { n , t } ^ { q }$ . The total coeficient on the local term at $( n , t )$ is therefore exactly its state occupancy $\omega _ { n , t } ^ { q }$ . Using (13) yields (14). Rigorously,

$$
\begin{array} { r l } { \nabla _ { \theta } \hat { Z } _ { \mathrm { E D R } } ( q _ { \theta } ; x ) = \nabla _ { \theta } V _ { 0 , 1 } ^ { q } ( x ) } \\ & { = \displaystyle \sum _ { t = 1 } ^ { L } \left[ \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) \nabla _ { \theta } V _ { n , t } ^ { q } ( x ) - \sum _ { n = 0 } ^ { t } \omega _ { n , t + 1 } ^ { q } ( x _ { \le t } ) \nabla _ { \theta } V _ { n , t + 1 } ^ { q } ( x ) \right] } \\ & { = \displaystyle \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) \left[ \nabla _ { \theta } V _ { n , t } ^ { q } ( x ) - A _ { n , t } ^ { q } ( x _ { \le t } ) \nabla _ { \theta } V _ { n , t + 1 } ^ { q } ( x ) - R _ { n , t } ^ { q } ( x _ { \le t } ) \nabla _ { \theta } V _ { t , t + 1 } ^ { q } ( x ) \right] } \\ & { = \displaystyle \sum _ { t = 1 } ^ { L } \sum _ { n = 0 } ^ { t - 1 } \omega _ { n , t } ^ { q } ( x _ { < t } ) \left[ \nabla _ { \theta } c _ { n , t } ^ { q } ( x _ { < t } ) + \nabla _ { \theta } A _ { n , t } ^ { q } ( x _ { \le t } ) \left( V _ { n , t + 1 } ^ { q } ( x ) - V _ { t , t + 1 } ^ { q } ( x ) \right) \right] . } \end{array}
$$

## B.5 Proof of Theorem 3

Let $B \geq 3$ and let the vocabulary be $\mathcal { V } = \{ 0 , 1 , a \}$ . We independently generate m subsequences $W _ { 1 } , \dots , W _ { m } .$ where

$$
W _ { i } = \left\{ \begin{array} { l l } { 0 \underbrace { a a \cdots a } _ { B - 2 \mathrm { ~ t i m e s } } 0 a , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 2 / 3 , } \\ { 1 \underbrace { a a \cdots a } _ { B - 2 \mathrm { ~ t i m e s } } 1 , } & { \mathrm { w i t h ~ p r o b a b i l i t y ~ } 1 / 3 . } \end{array} \right.
$$

We refer to these as type-0 and type-1 subsequences, respectively. Let the target distribution be $X =$ $W _ { 1 } W _ { 2 } \cdot \cdot \cdot W _ { m } { \cdot } \mathrm { E } 0 \mathrm { S } >$ , so that $\begin{array} { r } { \mathbb { E } [ L ] = \left( B + \frac { 2 } { 3 } \right) m } \end{array}$ . The integer m will be chosen at the end of the proof.

We consider first-order drafters of the form

$$
q _ { n , t } \left( \cdot \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } \right) = q _ { n , t } \left( \cdot \mid x _ { 1 : n } ; x _ { t - 1 } \right) .
$$

The block-local optimum. Consider a round that starts exactly at the beginning of a subsequence, following a committed prefix $x _ { 1 : n }$ . The first $B - 1$ draft positions can match the target distribution exactly:

$$
\widehat { q } _ { n , n + 1 } = \frac { 2 } { 3 } \delta _ { 0 } + \frac { 1 } { 3 } \delta _ { 1 } , \qquad \widehat { q } _ { n , n + j } = \delta _ { a } , \quad 2 \leq j \leq B - 1 .
$$

Here and below, these equalities are understood on target-supported within-block prefixes.

At the final draft position $n + B ,$ , the target repeats the first bit of the subsequence. Since $B \geq 3 .$ the immediately preceding token is a for both types. A first-order drafter must therefore use the same distribution at this position for both types. To maximize the expected accepted length, $\widehat { q }$ should maximize probablity of accepting the last token in the round, i.e,

$$
\widehat { q } _ { n , n + B } = \delta _ { 0 } .
$$

If $W _ { i } = 0 \underbrace { a a \cdot \cdot a } _ { \phantom { \mathscr { N } _ { i } } }$ 0a, all B draft tokens are accepted, and the target bonus token commits the final $a .$ B−2 times

If $W _ { i } = 1 \underbrace { a a \cdots a } _ { B - 2 \mathrm { ~ t i m e s } } 1$ , the proposal is rejected at the final draft position, and residual sampling commits the

final 1. In either case, the round commits exactly the entire subsequence, and the next round starts at the beginning of $W _ { i + 1 }$ . After the m subsequences have been generated, one additional round generates <EOS>. Therefore, every block-local optimal drafter $\widehat { q }$ satisfies

$$
\mathbb { E } [ \mathcal { T } ^ { \widehat { q } } ] = m + 1 .
$$

A continuation-aware drafter. We next construct another first-order drafter $q ^ { * }$ that intentionally sac rifices accepted length in some rounds to obtain more favorable future round starts.

Whenever a round starts at the beginning of a subsequence, let $q ^ { * }$ match the target distribution in the first $B - 1$ positions:

$$
q _ { n , n + 1 } ^ { * } = \frac { 2 } { 3 } \delta _ { 0 } + \frac { 1 } { 3 } \delta _ { 1 } , \qquad q _ { n , n + j } ^ { * } = \delta _ { a } , \quad 2 \leq j \leq B - 1 ,
$$

but choose

$$
q _ { n , n + B } ^ { * } = \delta _ { 1 } .
$$

If $W _ { i } = 0 \underbrace { a a \cdots a } _ { } \ 0 a$ , the final proposal is rejected. The round commits the corrected 0 and advances by B−2 times

B tokens, leaving the final a for the next round. If $W _ { i } = 1 \underbrace { a a \cdots a } _ { B - 2 \mathrm { ~ t i m e s } } 1$ , all B draft tokens are accepted. The

bonus token then commits the first token of the next subsequence, $\operatorname { o r } \mathbf { \langle } \mathbf { E } \mathbf { 0 } \mathbf { S } \mathbf { \rangle } \operatorname { i f } i = m$

Whenever a round starts inside a subsequence, let $q ^ { * }$ match the target conditionals at every draft position. This is possible within the first-order drafter class: the bit defining the current subsequence is already contained in the committed prefix. Any fresh bit from the next subsequence appears at draft position $j \geq 2$ and its repeated occurrence is at position $j + B - 1 \ge B + 1$ , outside the draft block. Before that repeated occurrence, the intervening tokens are all a. Thus no draft position needs access to an earlier within-block bit beyond the immediately preceding token. Every such round therefore advances by $B + 1$ tokens, unless generation terminates earlier at <EOS>.

To count the decoding rounds, extend $W _ { 1 } , W _ { 2 } , . . .$ . to an infinite independent sequence for this calculation. Group consecutive subsequences between successive rounds that start at the beginning of a subsequence.

If a group starts with a type-0 subsequence, its first round leaves the final a uncommitted. Each subsequent round either leaves the final a of another type-0 subsequence or commits an entire type-1 subsequence and returns to a subsequence boundary. The group therefore ends at the first type-1 subsequence after its initial type-0 subsequence. Its expected number of subsequences is $1 + 3 = 4$ , and it uses exactly one round per subsequence.

If a group starts with a type-1 subsequence, every round advances by B + 1 tokens until a subsequence boundary is reached again. For any initial segment of this group, its total length is $B + 1$ times its number of subsequences, minus its number of type-1 subsequences. Consequently, the first return to a subsequence boundary occurs upon completing the (B + 1)-st type-1 subsequence, counting the initial one. The expected number of subsequences in this group is $1 + 3 B$ . If the group contains ℓ subsequences, its total length is $( \ell - 1 ) ( B + 1 )$ , so it uses exactly ℓ − 1 rounds and saves one round.

Let $\ell _ { j }$ denote the number of subsequences in group j. The groups are independent and identically distributed, with

$$
\mathbb { E } [ \ell _ { j } ] = \frac { 2 } { 3 } \cdot 4 + \frac { 1 } { 3 } ( 1 + 3 B ) = B + 3 .
$$

Let

$$
N : = \operatorname* { m i n } \left\{ k \geq 1 : \sum _ { j = 1 } ^ { k } \ell _ { j } \geq m \right\} ,
$$

and let H be the number of these N groups that start with a type-1 subsequence. Since $\{ N \geq j \}$ depends only on the preceding groups and $N \leq m$ , summing expectations gives

$$
m \leq \mathbb { E } \left[ \sum _ { j = 1 } ^ { N } \ell _ { j } \right] = \sum _ { j = 1 } ^ { m } \mathbb { E } \big [ \mathbf { 1 } _ { \{ N \geq j \} } \ell _ { j } \big ] = ( B + 3 ) \mathbb { E } [ N ] ,
$$

$$
\mathbb { E } [ H ] = \sum _ { j = 1 } ^ { m } \operatorname* { P r } ( N \geq j , \operatorname { g r o u p } j \mathrm { ~ s t a r t s ~ w i t h ~ t y p e ~ } 1 ) = \frac { 1 } { 3 } \mathbb { E } [ N ] .
$$

All groups except possibly the last are fully generated before termination, and each completed type-1 group saves one round. For the last group, suppose that r of its subsequences remain before <EOS>. These subsequences contain at most $r ( B + 1 )$ tokens, and only the first round of the group can lose one token to rejection. Thus generating these subsequences together with <EOS> requires at most

$$
\left\lceil { \frac { r ( B + 1 ) + 2 } { B + 1 } } \right\rceil = r + 1
$$

rounds. At most one of the H type-1 groups is the last group, so

$$
{ \mathcal { T } } ^ { q ^ { * } } \leq m + 2 - H .
$$

Taking expectations yields

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \mathcal { T } ^ { q ^ { * } } ] \leq m + 2 - \mathbb { E } [ H ] = m + 2 - \frac 1 3 \mathbb { E } [ N ] } \\ { \displaystyle \leq \left( 1 - \frac 1 { 3 ( B + 3 ) } \right) m + 2 . } \end{array}
$$

Finally, choose any integer $m > ( 3 B + 9 ) ( 3 B + 1 1 )$ . Since both drafters preserve the same target distribution,

$$
\frac { \mathsf { M A L } ( q ^ { * } ) } { \mathsf { M A L } ( \widehat { q } ) } = \frac { \mathbb { E } [ T ^ { \widehat { q } } ] } { \mathbb { E } [ T ^ { q ^ { * } } ] }
$$

$$
\geq { \frac { m + 1 } { \left( 1 - { \frac { 1 } { 3 ( B + 3 ) } } \right) m + 2 } }
$$

$$
> 1 + { \frac { 1 } { 3 B + 9 } } .
$$

This proves the theorem.

## C Additional Details for Expected Decoding Rounds

## C.1 Draft EOS Probability Does Not Reduce EDR

For the EDR objective, a correctly drafted terminal <EOS> does not save a target-model verification round. Conditional on reaching the terminal position, accepting a drafted <EOS> terminates in the current verification pass. If the draft instead misses <EOS>, the correction branch can generate <EOS> in that same pass and terminates before another round starts. The same holds when the terminal token is produced as the target bonus token after block exhaustion. Accordingly, the EDR sum runs only over ordinary-token positions $t \leq L$

This is also visible in the dense local cost $c _ { n , t }$ . Its numerator is

$$
\sum _ { v \in \mathcal { V } } \left[ p _ { t } ( v \mid x _ { < t } ) - q _ { n , t } ( v \mid x _ { 1 : n } ; x _ { n + 1 : t - 1 } ) \right] _ { + } ,
$$

which excludes <EOS>. Thus matching target EOS mass is not directly rewarded by EDR.

## C.2 Fixed-Size Anchor Sampling

In this section, we explain how to use importance sampling to select a fixed number of anchors in EDR training. We apply anchor sampling separately to each target rollout in Algorithm 1. For a rollout $x =$ $( x _ { 1 : L } , < \mathtt { E } 0 \mathtt { S } > )$ , define

$$
\begin{array} { r } { \mathcal { N } _ { + } ( x ) : = \left\{ 0 \leq n < L : \omega _ { n , n + 1 } ^ { q _ { \theta } } ( x _ { 1 : n } ) > 0 \right\} . } \end{array}
$$

An anchor outside $\mathcal { N } _ { + } ( x )$ contributes zero to (14), since its occupancy $\omega _ { n , t } ^ { q _ { \theta } } ( x _ { < t } )$ is zero for every $t > n ,$ Let $\rho$ be the maximum number of training anchors per rollout, set to 512 in our experiments. If $\rho \geq$ $| \mathcal { N } _ { + } ( x ) |$ |, select $\mathcal { T } = \mathcal { N } _ { + } ( x )$ and set $\pi _ { n } ( x ) = 1$ for every $n \in \mathcal { N } _ { + } ( x )$ . Otherwise, choose $\lambda ( x ) > 0$ such that

$$
\sum _ { n \in { \cal N } _ { + } ( x ) } \operatorname* { m i n } \left\{ 1 , \lambda ( x ) \omega _ { n , n + 1 } ^ { q _ { \theta } } ( x _ { 1 : n } ) \right\} = \rho ,\tag{27}
$$

and set

$$
\pi _ { n } ( x ) : = \operatorname* { m i n } \left\{ 1 , \lambda ( x ) \omega _ { n , n + 1 } ^ { q _ { \theta } } ( x _ { 1 : n } ) \right\} .\tag{28}
$$

We draw a set $\mathcal { T } \subseteq \mathcal { N } _ { + } ( x )$ of exactly $\rho$ anchors by random-start systematic sampling, with

$$
\Pr ( n \in { \mathcal { T } } \mid x ) = \pi _ { n } ( x ) .
$$

For rollout $x ^ { ( i ) }$ , denote the sampled anchor set by $\boldsymbol { \mathcal { T } ^ { ( i ) } }$ and write $\pi _ { n } ^ { ( i ) } : = \pi _ { n } ( x ^ { ( i ) } )$ . Using the same notation as Algorithm 1, replace its loss by

$$
\ell ( \theta ) = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { n \in \mathbb { Z } ^ { ( i ) } } \sum _ { t = n + 1 } ^ { L _ { i } } \operatorname { s g } \left( \frac { \omega _ { n , t } ^ { ( i ) } } { \pi _ { n } ^ { ( i ) } } \right) \left[ c _ { n , t } ^ { ( i ) } + A _ { n , t } ^ { ( i ) } \operatorname { s g } \left( V _ { n , t + 1 } ^ { ( i ) } - V _ { t , t + 1 } ^ { ( i ) } \right) \right] .\tag{29}
$$

The occupancies and value functions are still computed by the full recursions in Algorithm 1, without gradients; sampling restricts only which anchor contributions are diferentiated. The sampled anchor sets and inclusion probabilities are also held fixed during diferentiation.

Conditioned on the target rollouts, the inclusion probabilities cancel the inverse-probability weights. Hence,

$$
\begin{array} { r l } & { \mathbb { E } _ { \{ \mathcal { Z } ^ { ( i ) } \} _ { i = 1 } ^ { M } } \left[ \nabla _ { \theta } \ell ( \theta ) \Big \vert \big \{ x ^ { ( i ) } \} _ { i = 1 } ^ { M } \right] } \\ & { \quad \quad = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \sum _ { n \in \mathcal { N } _ { + } ( x ^ { ( i ) } ) } \sum _ { t = n + 1 } ^ { L _ { i } } \omega _ { n , t } ^ { ( i ) } \left[ \nabla _ { \theta } c _ { n , t } ^ { ( i ) } + \nabla _ { \theta } A _ { n , t } ^ { ( i ) } \left( V _ { n , t + 1 } ^ { ( i ) } - V _ { t , t + 1 } ^ { ( i ) } \right) \right] } \\ & { \quad \quad = \displaystyle \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \nabla _ { \theta } \widehat { \mathcal { L } } _ { \mathrm { E D R } } \big ( q _ { \theta } ; x ^ { ( i ) } \big ) , } \end{array}\tag{30}
$$

where the last equality follows from (14). Taking the expectation over the target rollouts and applying (15) gives

$$
\mathbb { E } [ \nabla _ { \theta } \ell ( \theta ) ] = \nabla _ { \theta } \mathcal { L } _ { \mathrm { E D R } } ( q _ { \theta } ) .
$$

Thus anchor sampling preserves preserves the unbiasedness of the stochastic gradient.

## D Additional Details for Experiments

For all training settings and methods, we use the hyperparameters summarized in Table 3.

Table 3: Training hyperparameters used in all experiments.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Learning rate</td><td> $3 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>AdamW  $\beta _ { 1 }$ </td><td>0.9</td></tr><tr><td>AdamW  $\beta _ { 2 }$ </td><td>0.999</td></tr><tr><td>Learning-rate schedule</td><td>4% linear warmup of the first epoch, then constant</td></tr><tr><td>Gradient clipping</td><td>Global norm  $\leq 1$ </td></tr><tr><td>Global batch size</td><td>96</td></tr></table>