# APEX: SPECULATE SMARTER, NOT DEEPER

Manvi Jha<sup>∗</sup>   
University of Illinois Urbana-Champaign   
AWS AI   
manvij2@illinois.edu

Linbo LiuAWS AI

Sai Muralidhar Jayanthi AWS AI

Zach Zhang Zhichao Xu AWS AI AWS AI zhiang@amazon.com

Vinayak Arannil AWS AI varannil@amazon.com

## ABSTRACT

Speculative decoding reduces large language model inference latency by drafting multiple tokens before target-model verification, but its effectiveness depends on both the proposal mechanism and draft depth. Fixed configurations cannot respond to changes in predictability, repetition, and acceptance during generation, so deeper drafting can increase wasted computation without proportional speedup. We introduce APEX , a learned controller that balances decoding speed and draft-token waste through request-level expert selection and block-level depth adaptation. APEX-Router selects among EAGLE-3, n-gram, and draft-model speculation for each request, while APEX-Depth adjusts draft length at each verification block using causal decoding signals and recent verifier feedback. APEX models accepted draft length as censored survival feedback, learning position-wise rejection hazards, block execution costs, and an action utility that balances throughput, accepted progress, and wasted tokens. This allows the controller to adapt speculation while retaining the target model’s verification procedure. We integrate APEX into vLLM and evaluate it with Qwen3-8B across six workloads, achieving up to 5.24× speedup over autoregressive decoding. Across the aggregate evaluation, APEX -S achieves 4.27× speedup, while APEX -B achieves 3.27× speedup with a 41.0% relative reduction in wasted-token percentage compared with fixed n-gram speculation at k = 16, providing distinct operating points for balancing acceleration and draft-token utilization.

## 1 INTRODUCTION

Large language models (LLMs) increasingly serve interactive and long-form workloads such as code generation, mathematical reasoning, and conversational assistance. In these settings, inference latency is not merely a backend cost, it directly affects user-perceived responsiveness and the feasibility of deploying capable models at scale Kwon et al. (2023b); Agrawal et al. (2024). Yet acceleration must preserve the behavior of the target model. The systems challenge is therefore not simply to generate more tokens per second, but to convert additional computation into verified output progress without changing the target model’s generation semantics.

Autoregressive decoding Schuurmans et al. (2024) produces each token conditioned on the preceding prefix. Speculative decoding (SD) Xia et al. (2023) reduces the number of sequential targetmodel invocations by using a cheaper mechanism to propose multiple tokens, which the target model then verifies. Accepting a long draft prefix allows generation to advance several positions in one verification step. However, an early rejection leaves much of the proposed draft unused. Increasing draft depth consequently offers more potential progress while also increasing the amount of speculative work exposed to rejection. High speedup can coexist with substantial draft-token waste, and low waste can coexist with slow execution. Both wall-clock latency and draft-token utilization are therefore needed to assess the resulting trade-off.

The proposal mechanism also shapes this trade-off. Learned speculators such as EAGLE-3 Li et al. (2025a) can provide useful drafts when upcoming tokens are predictable. Retrieval-based approaches like n-gram reuse available text He et al. (2024), making inexpensive matching methods attractive when repeated structure is present. A separate draft model can produce plausible continuations across varied inputs, but its execution cost may offset the benefit of accepting additional tokens. These mechanisms therefore differ in both the drafts they produce and the cost of producing them. The same acceptance rate or wasted-token percentage can have different latency consequences across mechanisms.

The value of speculation depends jointly on the proposal mechanism, draft depth, and decoding state. A mechanism that proposes tokens cheaply may still discard a large fraction of its drafts, while a more accurate mechanism may incur enough proposal latency to offset its acceptance gains. Selecting an appropriate mechanism and depth requires accounting for the progress and cost of verification. This motivates our central research question:

## Can speculative decoding adapt its proposal mechanism and depth to maximize verified progress per unit cost as decoding evolves?

We address this question with APEX , a learned controller that combines request-level expert selection with block-level depth adaptation. At the beginning of a request, APEX-Router selects a proposal mechanism and an initial draft depth using the available request information. The selected mechanism remains fixed for that request, while APEX-Depth adjusts its draft length at each subsequent verification block. The depth controller uses causal features describing the current prefix, generation progress, and recent verifier outcomes. This separation accommodates the different timescales of the two decisions: routing determines the speculative execution path, while depth control responds to changes in acceptance and runtime within that path.

The central modeling idea is to treat accepted draft length as censored survival feedback. When a draft is rejected at a particular position, the observation identifies how far it survived verification. When every proposed token is accepted, the observation establishes survival only through the chosen depth; it does not reveal where a longer draft would first be rejected. APEX captures this structure through position-wise rejection hazards, from which it derives survival probabilities and expected accepted length. Runtime prediction complements this model because accepting more tokens does not necessarily produce greater speedup when drafting or verification costs also increase.

## 1.1 CONTRIBUTIONS

Our work makes the following contributions:

1. Empirical characterization of speculative speed and waste. We examine how proposal mechanisms and draft depths behave across workloads, entropy, repetition, and execution costs. The analysis identifies conditions in which deeper speculation increases waste without proportional acceleration and motivates control using decoding-state information.

2. Hierarchical control of heterogeneous speculators. We introduce APEX , which coordinates request-level selection among EAGLE-3, n-gram, and draft-model speculation with block-level depth adaptation. This structure combines mechanism selection with finergrained responses to changing verifier feedback.

3. A survival-based model of draft acceptance. We formulate accepted draft length using position-wise rejection hazards and a likelihood that handles both early rejection and fully accepted, right-censored blocks. Joint prediction of acceptance behavior and runtime provides supervision for learning the value of different depths.

4. Trace-supervised learning of action utility. We construct utility targets from observed throughput, accepted progress, draft-token waste, and regret relative to observed alternatives. Multi-task prediction and candidate-ranking losses train the controller to select depths using features available at inference time.

5. System integration and detailed evaluation. We integrate APEX into vLLM and evaluate it with Qwen3-8B across six workloads, including component ablations and candidatedepth sensitivity analysis. APEX -S achieves up to 5.24× wall-clock speedup over autoregressive decoding and an aggregate speedup of 4.27×. APEX -B achieves an aggregate speedup of $3 . 2 7 \times$ while reducing the mean wasted-token percentage from 79.90% to 47.17% relative to fixed n-gram speculation at $k = 1 6$ , a relative reduction of 41.0%.

## 2 BACKGROUND AND RELATED WORK

## 2.1 AUTOREGRESSIVE DECODING

Let $p _ { \theta }$ denote the target language model. Given a prompt or prefix $x _ { \le t } = ( x _ { 1 } , \dots , x _ { t } )$ , autoregressive decodingSchuurmans et al. (2024) generates one token at a time according to $x _ { t + 1 } \sim p _ { \theta } ( \cdot \ |$ $\scriptstyle { x \leq _ { t } } )$ . The newly generated token is appended to the prefix and the process repeats until a stopping condition is reached. This decoding procedure preserves the target model’s generation semantics, but it introduces a strict sequential dependency: producing $T$ output tokens requires $T$ target-model decoding steps. For large models, each step requires loading and applying the target network to the current prefix, making decoding latency scale directly with output length.

This sequential structure is especially costly for long-form generation settings such as code synthesis, multi-step reasoning, long-context completion, and software engineering tasks. The goal of inference acceleration is therefore not merely to reduce computation, but to reduce the number or cost of sequential target-model invocations while preserving the output distribution or verification semantics of the target model.

SD accelerates autoregressive generation by separating token proposal from target-model verification Xia et al. (2023); Leviathan et al. (2023); Chen et al. (2023). A cheaper proposal mechanism q first drafts a block of k candidate tokens, $\hat { x } _ { t + 1 : t + k } \sim q ( \cdot \mid x _ { \le t } )$ , and the target model verifies the proposed block in parallel. If multiple draft tokens are accepted, generation advances by more than one token using a single target-model verification step.

The efficiency of this procedure depends on how many proposed tokens survive verification. Let $L _ { t } \in \{ 0 , \ldots , k \}$ denote the accepted draft length at block $t ,$ and let $r _ { t }$ denote the first rejection position, with the event $r _ { t } ~ > ~ k$ corresponding to full acceptance of the draft block. If rejection occurs early, all remaining draft tokens after the first rejected token are discarded. Thus, the useful progress of a speculative block is not the proposed depth $k ,$ but the accepted length $L _ { t }$

This observation makes SD a cost-sensitive procedure. A larger k increases the maximum possible progress per verifier call, but it also increases the amount of draft work that can be wasted after early rejection. The relevant quantity for efficient serving is therefore accepted progress per unit cost:

$$
\mathcal { A } _ { t } ( m , k ) = \frac { L _ { t } } { C _ { t } } ,\tag{1}
$$

where m denotes the speculative mechanism and $C _ { t }$ denotes the block-level proposal and verification cost. A speculative configuration is profitable only when the accepted progress outweighs the extra drafting and verification overhead.

## 2.2 SPECULATIVE MECHANISMS

Early SD methods use an auxiliary draft model to propose tokens for target-model verification (Xia et al., 2023; Leviathan et al., 2023). Followup works improve drafting through knowledge distillation, specialized architectures, or objectives aligned with target-model acceptance (Bhansali & Heck, 2025; Zou et al., 2026). Improving an individual proposal mechanism does not by itself determine when that mechanism is preferable to alternatives with different proposal costs and acceptance behavior. A fixed choice can therefore retain a costly or poorly matched speculative path even when another mechanism offers a more favorable speed–waste trade-off.

A second line of work reduces reliance on separate draft models. Medusa adds decoding heads to the target model for parallel verification of tree-structured continuations (Cai et al., 2024), while SpecInfer constructs token trees for parallel candidate verification (Miao et al., 2024). EAGLE-style methods instead use target-model hidden states to build lightweight neural speculators (Li et al., 2025b;a). Their gains, however, still depend on proposal survival and cost.

A third class uses retrieval or surface repetition instead of neural drafting. N-gram and suffix-based methods copy continuations from the prompt, generated prefix, or external text, making them cheap and effective in repetitive regimes but brittle on novel continuations (Hu et al., 2024). This creates a complementary tradeoff: neural speculators can better model semantic continuation, while retrievalbased speculators can be faster when repeated structure is available.

<table><tr><td></td><td>N-gram</td><td>Draft model</td><td>EAGLE-3</td><td rowspan="5">16× Spe ese e deeng 8×</td></tr><tr><td>Code generation</td><td>2.59× k=16</td><td>1.64× k=1</td><td>4.17× k=4</td></tr><tr><td>Conversational</td><td>1.31× k=8</td><td>0.81× k = 1</td><td>2.04× k=4</td></tr><tr><td>Long-chain reasoning</td><td>13.76× k = 16</td><td>1.80× k=1</td><td>8.51× k=4</td></tr><tr><td>Long-context completion</td><td>13.26× k= 16</td><td>1.70× k=1</td><td>8.29× k=4</td></tr><tr><td>SWE</td><td>5.92× k=16</td><td>1.71× k=1</td><td>5.34× k=8</td></tr><tr><td>Mathematical reasoning</td><td>2.81× k=4</td><td>1.64× k=1</td><td>4.17× k=4</td></tr></table>

Figure 1: Speedup over autoregressive decoding across six workloads and three speculative mech anisms. Each cell reports highest speedup and its depth k; boxes mark the best method per workload.

APEX is orthogonal to these mechanisms; it treats mechanisms like EAGLE-3, n-gram, and draftmodel speculation as candidate experts and learns which of them to use and how deeply to speculate in each decoding state. APEX can thus choose among mechanisms with different proposal costs and acceptance behavior while retaining their existing proposal and verification algorithms.

## 2.3 ADAPTIVE SPECULATIVE DECODING

Dynamic Speculation Lookahead shows that using the same speculation lookahead for all decod ing iterations is suboptimal and proposes dynamically selecting the number of draft tokens during generation (Mamou et al., 2024). AdaEAGLE (Zhang et al., 2024)explicitly models adaptive draft structures and uses a lightweight draft length predictor to choose the number of draft tokens for EAGLE-style speculation. BanditSpec (Hou et al., 2025) formulates speculative hyperparameter selection as a multi-armed bandit problem and adapts the configuration online without training a controller. DSDE (Yang et al., 2025) studies dynamic speculation length in serving settings using KLD-based stability signals and adaptive caps. Production systems such as vLLM also support dynamic SD through manually specified schedules that map concurrency or batch-size ranges to a chosen number of speculative tokens (Kwon et al., 2023a).

These studies show that a single fixed depth (k) can limit inference acceleration as acceptance behavior and serving conditions change. They motivate a broader control problem: selecting a proposal mechanism together with depths that balance verified progress, runtime, and draft-token waste. APEX addresses this problem through request-level expert selection and block-level depth adaptation, using learned rejection hazards and runtime signals to score candidate depths.

## 2.4 SPECULATIVE DECODING

## 3 EMPIRICAL MOTIVATION: FIXED SPECULATION IS NOT ENOUGH

We examine how the proposal mechanism and draft depth affect acceleration and draft-token waste. Using Qwen3-8B in vLLM, we compare autoregressive decoding with EAGLE-3, n-gram, and draftmodel speculation across code generation, mathematical reasoning, long-context completion, longchain reasoning, conversational generation, and SWE-style tasks. Workload details and full evaluation results appear in Section 5.

Speculative performance depends on the workload and mechanism. Figure 1 summarizes the highest observed speedups in the characterization set, illustrating workload-dependent differences between mechanisms. These peak results are distinct from the aggregate averages reported on a separate evaluation set in Table 1. Figure 2a further shows that acceptance alone does not determine acceleration: draft-model speculation remains near or below autoregressive performance in the plotted median results, even at relatively high acceptance rates, while the other mechanisms achieve substantial speedup. This motivates considering execution cost alongside acceptance when selecting a speculative mechanism.

![](images/ce4533f35274b43830c1cc8302a53642ab73ea6ce996baa03ac5bde55f3d1b4a.jpg)  
(a) Acceptance and speedup

![](images/356e2f8a78c6f572c7227a21cfb1d50dba300b8e67f2aeaf0e7f3e75f5c4a9cc.jpg)  
(b) Entropy–repetition vs speedup  
Figure 2: Online signals associated with SD performance (a) Median speedup over autoregressive decoding versus acceptance rate, with shaded variability. (b) Median speedup versus token entropy and repetition density; the gray plane marks autoregressive performance.

Greater depth does not guarantee a better speed–waste trade-off. The aggregate sweep in Table 1 shows contrasting responses to deeper speculation. Increasing n-gram depth from $k = 8$ to k = 16 raises mean speedup from 3.59× to 4.22×, while mean waste increases from 63.56% to 79.90%. For EAGLE-3, increasing depth from k = 4 to k = 16 reduces mean speedup from 2.35× to 1.37× and raises mean waste from 66.25% to 97.67%. Deeper drafting therefore offers a modest speed gain at substantially higher waste in one case and worsens both metrics in the other. These responses motivate depth selection that accounts for the active mechanism and expected verification outcome.

Observable decoding signals can inform control. Figure 2b relates median speedup to token entropy and repetition density. Performance varies across these conditions, and the patterns differ between mechanisms. These associations motivate using entropy and repetition as controller inputs alongside recent verifier feedback. Prefix statistics describe the context in which a draft is proposed, while verifier history records how well recent drafts have performed. Both can be obtained from information available during decoding, allowing adaptation without requiring workload labels at inference time.

Verifier feedback reveals where drafts fail. For a draft of length k, first rejection at position r leaves r − 1 accepted draft tokens and $k - r + 1$ discarded draft tokens. Full acceptance provides a different observation: the draft survives through depth k, but the outcome of extending it further remains unknown. Verifier feedback thus contains both observed rejection events and right-censored full-acceptance events. This structure motivates modeling position-wise rejection hazards, which describe how acceptance can change with depth and provide supervision for adaptive speculative control.

## 4 APEX: BREATHING ADAPTIVITY INTO STATIC SPECULATION

We propose APEX , a learned online controller that selects a proposal mechanism and initial depth at request granularity and adjusts speculation depth at block granularity. Its learned utility balances throughput and accepted progress against discarded draft tokens. APEX operates at two coupled timescales. APEX-Router selects a speculative expert (EAGLE-3, n-gram, or draft-model speculation) and its initial depth for each request, while APEX-Depth adapts depth k at the block level. This separation reflects systems constraints: expert selection affects model placement, batching, and worker routing, whereas depth can change more frequently within the selected path.

![](images/93ec17e6ed8f420159da45bb711386de2f4dfeb7d610db0c27007b7654e71f52.jpg)  
Figure 3: APEX overview. The router selects a speculative expert and initial depth from prompt features. After each verifier block, the depth controller uses the updated decoding state and prior verifier feedback to select the next depth. Observed acceptance and runtime update the controller state; network parameters remain fixed during inference.

Figure 3 summarizes the framework. We first define the decoding state and request-level routing decision, then formulate accepted draft length as censored survival feedback. This formulation provides the acceptance model used by APEX-Depth , which learns acceptance, runtime, and utility predictions from a shared representation.

## 4.1 PROBLEM FORMULATION AND STATE REPRESENTATION

Let $p _ { \theta }$ denote the target language model and let $m \in \mathcal { M }$ denote a proposal mechanism, where

$$
\mathcal { M } = \{ \mathrm { E A G L E - } 3 , \mathrm { n - g r a m } \mathrm { S D } , \mathrm { D r a f t - S D } \} .\tag{2}
$$

We index verifier blocks by t and denote the prefix available before block t by $c _ { t }$ . Given this prefix, mechanism m proposes a draft block $d _ { t }$ of length $k _ { t }$ according to $q _ { m } ( \cdot \mid c _ { t } ; k _ { t } )$ , its induced proposal distribution. The target model then verifies the draft block. Let $L _ { t } \in \{ 0 , \ldots , k _ { t } \}$ be the number of accepted draft tokens. If $L _ { t } = k _ { t }$ , the full block is accepted; if $L _ { t } < k _ { t } .$ , the verifier rejects at position $L _ { t } + 1$ . We define the discarded draft-token count and waste fraction as

$$
R _ { t } = k _ { t } - L _ { t } , \qquad W _ { t } = \frac { R _ { t } } { k _ { t } } .\tag{3}
$$

Here, $R _ { t }$ includes the first rejected token and the discarded suffix, and $W _ { t }$ is the fraction of drafted tokens that are not accepted. Since $W _ { t }$ measures token utilization rather than wasted computation, we measure latency separately to assess the speed–waste trade-off.

Let $C _ { t }$ denote block-t wall-clock latency, including drafting and verification, and TPS the executed action’s throughput. APEX uses these observations to learn a block-level utility balancing accepted progress, runtime, and speculative waste. In our experiments, $K _ { \operatorname* { m a x } } = 1 6$ , and candidate depths form a finite set ${ \cal K } \subseteq \{ 1 , \dot { \ } . \ . . , K _ { \mathrm { m a x } } \}$

Request-level state The request-level state $s _ { 0 }$ contains the information available before generation begins:

$$
s _ { 0 } = [ e _ { \mathrm { p r o m p t } } , \ell _ { \mathrm { p r o m p t } } , \rho _ { \mathrm { p r o m p t } } , \eta _ { \mathrm { p r o m p t } } , c _ { \mathrm { d e c o d e } } ] .\tag{4}
$$

Here, $e _ { \mathrm { p r o m p t } }$ is a fixed-dimensional prompt embedding, $\ell _ { \mathrm { p r o m p t } }$ is the prompt length, ρ<sub>prompt</sub> contains repetition and lexical-structure features, $\eta _ { \mathrm { p r o m p t } }$ contains entropy or token-distribution features, and $c _ { \mathrm { d e c o d e } }$ contains decoding configuration variables such as temperature and maximum generation length.

Block-level state The block-level state $s _ { t }$ contains information available before verifier block t:

$$
s _ { t } = [ e _ { \mathrm { p r o m p t } } , \ell _ { \mathrm { p r o m p t } } , c _ { \mathrm { d e c o d e } } , \pi _ { t } , \mathbf { h } _ { t - 1 } , \mathbf { g } _ { t } ] .\tag{5}
$$

Here, $\pi _ { t }$ denotes generation progress, $\mathbf { h } _ { t - 1 }$ summarizes verifier feedback from blocks $1 , \ldots , t - 1$ and $\mathbf { g } _ { t }$ contains statistics computed from the current prefix $c _ { t }$ . We supply the active mechanism m and candidate depth k separately when constructing state-action features.

The history vector $\mathbf { h } _ { t - 1 }$ includes the number of observed verifier blocks, rolling mean and variance of accepted length, first-rejection position statistics, full-acceptance rate, rejection rate, mean rejected-token count, and short-window summaries over the most recent verifier blocks. The prefix vector $\mathbf { g } _ { t }$ includes entropy and repetition features. These features are causal: the current block outcome is never used to predict the current block.

## 4.2 APEX-ROUTER

APEX-Router jointly selects the speculative expert and initial depth expected to provide the best speed– cost trade-off for each request before generation. It scores initial actions $a = ( m , k _ { 0 } )$ , where $m \in$ $\mathcal { M }$ is the speculative mechanism and $k _ { 0 }$ is the initial depth. Let $\mathcal { A } _ { R } \subseteq \mathcal { M } \times \mathcal { K }$ denote the admissible initial actions. Using the pre-generation state $s _ { 0 } .$ , it predicts

$$
\widehat { U } _ { R } ( s _ { 0 } , m , k _ { 0 } ) = f _ { \theta _ { R } } ( \phi _ { R } ( s _ { 0 } , m , k _ { 0 } ) ) ,\tag{6}
$$

where $\phi _ { R }$ combines the request features with the candidate method and initial depth, and $f _ { \theta _ { R } }$ is a neural scoring function. The selected action i

$$
( m ^ { \star } , k _ { 0 } ^ { \star } ) = \arg \operatorname* { m a x } _ { ( m , k _ { 0 } ) \in \mathcal { A } _ { R } } \widehat { U } _ { R } ( s _ { 0 } , m , k _ { 0 } ) .\tag{7}
$$

The router is trained from completed request-level traces rather than workload labels. Each observed request–method–initial-depth combination supplies a training example:

$$
( s _ { 0 } ^ { ( r ) } , m , k _ { 0 } ) \longrightarrow u _ { R } ( r , m , k _ { 0 } ) ,
$$

where $s _ { 0 } ^ { ( r ) }$ is the initial state of request r and $u _ { R } ( r , m , k _ { 0 } )$ is its empirical request-level utility label. The observed label $u _ { R }$ provides offline supervision for the predicted score $\widehat { U } _ { R }$ . The resulting acceptance, latency, and waste are execution outcomes; they are not inputs available when routing a new request.

At inference, the router scores each admissible method–initial-depth pair using only $s _ { 0 }$ and the candidate action. The selected action determines the speculative decoding path and initializes its depth. The mechanism $m ^ { \star }$ remains fixed for the request, the first block uses $k _ { 1 } = k _ { 0 } ^ { \star }$ , and APEX-Depth controls subsequent depths.

Once the expert is selected, the depth decision depends on how its drafts survive verification as generation evolves. We next formulate this acceptance feedback before describing the depth controller.

## 4.3 ACCEPTANCE AS CENSORED SURVIVAL FEEDBACK

The accepted length of a speculative block contains more information than an aggregate acceptance rate. A partially accepted block identifies the first rejected position, while a fully accepted block establishes only that rejection did not occur within the executed depth. APEX models these two outcomes using a discrete survival formulation.

For a candidate depth $k ,$ define $J _ { t }$ as the first rejected draft position using one-indexed positions. Thus, $J _ { t } = j$ means the first $j - 1$ draft tokens were accepted and the j-th draft token was rejected. If the entire block is accepted, then $J _ { t } > k .$ . The accepted length is

$$
L _ { t } = { \left\{ \begin{array} { l l } { J _ { t } - 1 , } & { J _ { t } \leq k , } \\ { k , } & { J _ { t } > k . } \end{array} \right. }\tag{8}
$$

Position-wise rejection hazards. For each position $j \in \{ 1 , \dots , K _ { \operatorname* { m a x } } \}$ , the model predicts a discrete hazard:

$$
\hat { q } _ { t , j } ( k ) = P _ { \theta _ { D } } ( J _ { t } = j \mid J _ { t } \geq j , s _ { t } , m , k ) ,\tag{9}
$$

where $\theta _ { D }$ denotes the parameters of the depth controller. This is the conditional probability of rejection at position $j ,$ given that all preceding draft tokens have been accepted.

The probability that the first j draft tokens survive is

$$
\hat { S } _ { t , j } ( k ) = P _ { \theta _ { D } } ( J _ { t } > j \mid s _ { t } , m , k ) = \prod _ { i = 1 } ^ { j } \left( 1 - \hat { q } _ { t , i } ( k ) \right) .\tag{10}
$$

The expected accepted length follows as

$$
\widehat { \mathbb { E } } [ L _ { t } \mid s _ { t } , m , k ] = \sum _ { j = 1 } ^ { k } \hat { S } _ { t , j } ( k ) ,\tag{11}
$$

using the identity $\begin{array} { r } { \mathbb { E } [ L ] = \sum _ { j = 1 } ^ { k } P ( L \geq j ) } \end{array}$

Accepted-length likelihood. If the observed accepted length is $L _ { t } ~ = ~ \ell ~ < ~ k$ , the first rejected position is $J _ { t } = \ell + 1$ . The likelihood is therefore

$$
P _ { \theta _ { D } } ( L _ { t } = \ell \mid s _ { t } , m , k ) = \left[ \prod _ { j = 1 } ^ { \ell } ( 1 - \hat { q } _ { t , j } ( k ) ) \right] \hat { q } _ { t , \ell + 1 } ( k ) .\tag{12}
$$

The empty product equals one when $\ell = 0$ . If $L _ { t } = k$ , the entire block is accepted and the rejection position is right-censored beyond k:

$$
P _ { \theta _ { D } } ( L _ { t } = k \mid s _ { t } , m , k ) = \prod _ { j = 1 } ^ { k } ( 1 - \hat { q } _ { t , j } ( k ) ) .\tag{13}
$$

This observation contributes evidence of survival through position k without imposing a rejection at position $k + 1$

The survival loss over $N$ training blocks is

$$
\mathcal { L } _ { \mathrm { s u r v } } = - \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \log P _ { \theta _ { D } } ( L _ { t } \mid s _ { t } , m _ { t } , k _ { t } ) .\tag{14}
$$

This objective trains the controller to capture the first-rejection structure of SD. Acceptance prediction is then combined with runtime and utility supervision, since accepted length alone does not determine the efficiency of a candidate depth.

## 4.4 APEX-DEPTH

APEX-Depth adapts speculation depth to balance accepted progress, latency, and draft-token waste. Its operation begins with the speculative method and initial depth selected by APEX-Router . Before each subsequent block, it observes the causal state $s _ { t }$ , evaluates the candidate depths, and selects the depth with the highest predicted utility.

Shared representation For each candidate depth $k \in { \mathcal { K } } .$ , the controller constructs

$$
z _ { t } ( k ) = \phi _ { D } ( s _ { t } , m ^ { \star } , k ) , \qquad v _ { t } ( k ) = g _ { \theta _ { D } } ( z _ { t } ( k ) ) .\tag{15}
$$

Here, $\phi _ { D }$ standardizes numerical features, embeds categorical variables, and appends the prompt representation. The shared encoder therefore evaluates each depth in the context of the current generation state.

Four prediction heads receive $v _ { t } ( k )$ . Each head is a task-specific output branch of the same network: the survival, cost, and throughput heads predict execution outcomes, while the utility head predicts the action score used for depth selection.

Survival head The survival head outputs a vector of $K _ { \mathrm { m a x } }$ conditional rejection probabilities:

$$
\hat { \mathbf { q } } _ { t } ( k ) = f _ { \mathrm { s u r v } } ( v _ { t } ( k ) ) .\tag{16}
$$

Its j-th entry is the hazard ${ \hat { q } } _ { t , j } ( k )$ defined in Equation equation 9. The first k entries determine the survival curve and expected accepted length through Equations equation 10–equation 11. This head is trained with the censored likelihood in Section 4.3, allowing fully accepted and partially rejected blocks to provide appropriate supervision.

Cost head The cost head predicts the wall-clock cost of executing the candidate block, including drafting and verification. Its scalar output is the predicted log latency:

$$
\widehat { c } _ { t } ( k ) = f _ { \mathrm { c o s t } } ( v _ { t } ( k ) ) = \widehat { \log C _ { t } ( k ) } .\tag{17}
$$

This head provides runtime supervision because a candidate depth can accept many tokens but still be slow when proposal or verification overhead is high.

Throughput head The throughput head predicts the rate of output progress achieved by the candidate action. Its scalar output is

$$
\hat { \tau } _ { t } ( k ) = f _ { \mathrm { t p s } } ( v _ { t } ( k ) ) = \log ( 1 \widehat { + \mathrm { T P S } _ { t } ( k ) } ) .\tag{18}
$$

Where the cost head models elapsed time, the throughput head models the observed progress rate. Their targets are obtained from measured execution traces, and their regression losses are defined in Section 4.6.

Utility head The utility head predicts the final learned action value:

$$
\widehat { U } _ { D } ( s _ { t } , m ^ { \star } , k ) = f _ { \mathrm { u t i l } } ( v _ { t } ( k ) ) .\tag{19}
$$

The hatted quantity $\widehat { U } _ { D }$ is a network prediction made before the candidate block executes. During training, it is fitted to an observed scalar label $u _ { D , t }$ computed from execution outcomes, as defined in Equation equation 25. The utility-regression loss in Equation equation 28 directly connects this prediction to its label, while the ranking loss supervises candidate ordering. The other three heads ground the shared representation in measurable outcomes.

Select, execute, and update The controller chooses the highest-scoring candidate:

$$
k _ { t } ^ { \star } = \arg \operatorname* { m a x } _ { k \in { \cal K } } \widehat { U } _ { D } ( s _ { t } , m ^ { \star } , k ) , \qquad t \geq 2 .\tag{20}
$$

The utility head supplies the deployed selection score; the other heads support it through shared training. The chosen speculative block is then executed, and its observed outcomes update the state for the next decision. This allows depth to respond to changes in acceptance behavior and execution cost throughout generation.

## 4.5 TRACE-DERIVED TRAINING TARGETS

We train APEX from offline collected speculative decoding traces. Each depth-controller training example corresponds to an observed verifier block under a particular state-action pair:

$$
\begin{array} { r } { ( s _ { t } , m _ { t } , k _ { t } ) \longrightarrow ( L _ { t } , R _ { t } , W _ { t } , C _ { t } , \mathrm { T P S } _ { t } ) . } \end{array}
$$

The input contains only information available before executing the block, while the targets contain the outcomes observed after verification. Accepted length supervises the survival head, latency and throughput supervise the runtime heads, and the following utility target supervises the utility head. We use lowercase u for observed utility labels and hatted $\hat { \vec { U } }$ for their learned predictions.

Dataset-derived regret targets The trace dataset provides offline comparisons among speculative configurations. For a request r, let $\mathcal { D } ( r )$ denote the set of method-depth actions observed in the dataset, and let $\mathrm { T P S } _ { r , m , k }$ denote the recorded request-level throughput under action $( m , k )$ . We define the best observed global throughput as

$$
B _ { \mathrm { g l o b a l } } ( r ) = \operatorname* { m a x } _ { ( m , k ) \in \mathscr { D } ( r ) } \mathrm { T P S } _ { r , m , k } ,\tag{21}
$$

and the best observed throughput within a fixed mechanism as

$$
B _ { \mathrm { m e t h o d } } ( r , m ) = \operatorname* { m a x } _ { k : ( m , k ) \in \mathcal { D } ( r ) } \mathrm { T P S } _ { r , m , k } .\tag{22}
$$

For a training block t belonging to request $r _ { t }$ , the global and same-method regret terms are

$$
G _ { t } = \frac { [ B _ { \mathrm { g l o b a l } } ( r _ { t } ) - \mathrm { T P S } _ { t } ] _ { + } } { B _ { \mathrm { g l o b a l } } ( r _ { t } ) } ,\tag{23}
$$

$$
M _ { t } = \frac { \left[ B _ { \mathrm { m e t h o d } } ( r _ { t } , m _ { t } ) - \mathrm { T P S } _ { t } \right] _ { + } } { B _ { \mathrm { m e t h o d } } ( r _ { t } , m _ { t } ) } ,\tag{24}
$$

where $[ x ] _ { + } = \operatorname* { m a x } ( x , 0 )$ . These normalized penalties compare block throughput with the best observed request-level reference throughputs. The global term accounts for alternatives across mechanisms and depths, while the same-method term accounts for depth choices within the active mechanism. Both terms are used only to construct training targets and are not required during inference.

Block-level utility For each observed training block, the depth-utility label is

$$
u _ { D , t } = \log ( \mathrm { T P S } _ { t } ) + \alpha _ { A } \log ( L _ { t } + \epsilon ) - \alpha _ { W } W _ { t } - \alpha _ { G } G _ { t } - \alpha _ { M } M _ { t } ,\tag{25}
$$

where $\epsilon = 0 . 1$ keeps the accepted-length term finite when $L _ { t } = 0$ . This is the trace-derived label for the executed action $( s _ { t } , m _ { t } , k _ { t } ) ; \hat { U } _ { D } ( s _ { t } , m _ { t } , k _ { t } )$ is the network’s prediction of that label.

The utility combines speed, accepted progress, waste, and offline regret. High accepted length is useful only if the resulting progress justifies its runtime, while low waste is useful only if the system still makes enough progress per verifier call. The coefficients allow different training preferences for this speed–waste trade-off.

## 4.6 TRAINING OBJECTIVE FOR APEX-DEPTH

The APEX-Depth network is multi-task by design. The survival loss in Equation equation 14 supervises the rejection hazards. The cost and throughput losses supervise runtime predictions, while utility regression and candidate ranking train the score used for depth selection.

Cost and throughput losses Using the observed action $\left( { { s _ { t } } , { m _ { t } } , { k _ { t } } } \right)$ for each training block, we train the cost head with

$$
\mathcal { L } _ { \mathrm { c o s t } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \rho ( \hat { c } _ { t } ( k _ { t } ) - \log C _ { t } ) ,\tag{26}
$$

where $\rho ( \cdot )$ is the SmoothL1 loss. The throughput head is trained with

$$
\mathcal { L } _ { \mathrm { t p s } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \rho ( \hat { \tau } _ { t } ( k _ { t } ) - \log ( 1 + \mathrm { T P S } _ { t } ) ) .\tag{27}
$$

Utility regression The utility head is trained by comparing its prediction for each observed stateaction pair with the corresponding trace-derived label:

$$
\mathcal { L } _ { \mathrm { u t i l } } = \frac { 1 } { N } \sum _ { t = 1 } ^ { N } \rho \Big ( \widehat { U } _ { D } ( s _ { t } , m _ { t } , k _ { t } ) - u _ { D , t } \Big ) .\tag{28}
$$

Thus, the same predicted score used to select depths at inference is directly supervised during training.

Candidate ranking Deployment depends on the ordering of candidate depths, as well as the accuracy of their predicted utilities. We therefore group observed candidate rows by prompt, speculative method, and block index. For a group $^ { g , }$ , let $\textstyle { \mathcal { K } } _ { g }$ denote the depths present, and let $u _ { D , g } ( k )$ ) denote the observed utility label at depth k. Writing $s _ { g , k }$ for that row’s pre-block state and $m _ { g }$ for the group’s method, its predicted score is

$$
\widehat { U } _ { D , g } ( k ) = \widehat { U } _ { D } ( s _ { g , k } , m _ { g } , k ) .\tag{29}
$$

The preferred depth is

$$
k _ { g } ^ { + } = \arg \operatorname* { m a x } _ { k \in \mathcal { K } _ { g } } u _ { D , g } ( k ) .\tag{30}
$$

The ranking loss is

$$
\mathcal { L } _ { \mathrm { r a n k } } = - \frac { 1 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \log \frac { \exp ( \widehat { U } _ { D , g } ( k _ { g } ^ { + } ) ) } { \sum _ { k \in \mathcal { K } _ { g } } \exp ( \widehat { U } _ { D , g } ( k ) ) } ,\tag{31}
$$

where $\mathcal { G }$ is the set of rank groups. This term encourages the depth with the highest observed training utility in each group to receive the highest predicted score.

Combined objective The depth controller is trained with

$$
\mathcal { L } _ { D } = \lambda _ { s } \mathcal { L } _ { \mathrm { s u r v } } + \lambda _ { c } \mathcal { L } _ { \mathrm { c o s t } } + \lambda _ { v } \mathcal { L } _ { \mathrm { t p s } } + \lambda _ { u } \mathcal { L } _ { \mathrm { u t i l } } + \lambda _ { r } \mathcal { L } _ { \mathrm { r a n k } } .\tag{32}
$$

In our implementation, the default loss weights are

$$
( \lambda _ { s } , \lambda _ { c } , \lambda _ { v } , \lambda _ { u } , \lambda _ { r } ) = ( 1 . 0 0 , 0 . 2 0 , 0 . 3 0 , 1 . 0 0 , 1 . 0 0 ) .\tag{33}
$$

## 4.7 APEX VARIANTS

We train three APEX variants with identical architectures, features, candidate-depth interfaces, data, and deployment procedures; only the utility weights in Equation equation 25 differ, producing distinct speed–waste operating points.

APEX-S (Speed) uses $\begin{array} { l l l } { { \left( \alpha _ { A } , \alpha _ { W } , \alpha _ { G } , \alpha _ { M } \right) } } & { { = } } & { { \left( 0 . 2 0 , 0 . 2 0 , 0 . 2 0 , 0 . 1 5 \right) } } \end{array}$ and places greater relative emphasis on throughput competitiveness. The default APEX-B (Balanced) uses (0.30, 0.35, 0.15, 0.20) to balance throughput, accepted progress, and waste. APEX-E (Efficient) uses (0.40, 0.50, 0.10, 0.25) and assigns a larger penalty to waste. These weights specify training preferences; they do not guarantee that every profile is Pareto-efficient or that APEX-E always has the lowest waste. Additional ablations are reported in Section 5.3.

## 4.8 ONLINE DEPLOYMENT

At inference, the router selects the speculative expert $m ^ { \star }$ and initial depth $k _ { 0 } ^ { \star }$ . After each verifier block, APEX updates its causal state with observed outcomes, scores every candidate depth in $\kappa .$ , and selects the next block’s depth. The controller adapts its actions using the evolving state; its network parameters remain fixed during generation.

Algorithm 1 summarizes this procedure. Here, SpeculateAndVerify drafts a block using the selected mechanism, applies target-model verification, and records the execution outcomes. It returns the emitted output $\Delta y _ { t }$ and observations $o _ { t } \ = \ ( L _ { t } , R _ { t } , W _ { t } , C _ { t } , \mathrm { T P S } _ { t } )$ . The output extends the prefix, while the observations update the history used for the next decision.

## 5 RESULTS AND DISCUSSION

We evaluate whether APEX can maintain substantial inference acceleration while reducing discarded draft tokens relative to fixed speculative configurations. We first examine aggregate performance and workload-specific trade-offs, then study the contributions of individual controller components and the sensitivity to the candidate-depth set.

## 5.1 EXPERIMENTAL SETUP

Target model and serving system We implement APEX in vLLM with Qwen3-8B as the target verifier. All evaluated methods use the same target model, allowing us to compare the acceleration and draft-token utilization achieved by different proposal mechanisms, fixed depths, and adaptive controller profiles.

Algorithm 1 APEX Online Speculative Decoding   
Require: Prompt x, decoding configuration c<sub>decode</sub>, target model p<sub>θ</sub>, initial actions $A _ { R } ,$ depths K   
1: $s _ { 0 } \gets \mathrm { R e q u e s t S t a t e } ( x , c _ { \mathrm { d e c o d e } } )$   
2: (m<sup>⋆</sup>, k<sup>⋆</sup>) ← arg max $\widehat { U } _ { R } ( s _ { 0 } , m , k _ { 0 } )$   
(m,k<sub>0</sub>)∈A<sub>R</sub>   
3: $\mathbf { h } _ { 0 } \gets \emptyset , y \gets \emptyset$   
4: for t = 1, 2, . . . until generation completes do   
5: c<sub>t</sub> ← x ∥ y   
6: $s _ { t } \gets \mathrm { S t a t e } ( s _ { 0 } , c _ { t } , \mathbf { h } _ { t - 1 } )$   
7: if t = 1 then   
8: $k _ { t }  k _ { 0 } ^ { \star }$   
9: else   
10: k<sub>t</sub> ← arg max $\widehat { U } _ { D } ( s _ { t } , m ^ { \star } , k )$   
k∈K   
11: end if   
12: (∆y<sub>t</sub>, o<sub>t</sub>) ← SpeculateAndVerify(p<sub>θ</sub>, m<sup>⋆</sup>, k<sub>t</sub>, c<sub>t</sub>)   
13: $y  y \parallel \Delta y _ { t }$   
14: h ← Update(h , o )   
15: end for   
16: return y

Table 1: Overall decoding performance. APEX is compared against autoregressive decoding and fixed speculative baselines. Higher speedup is better; lower wasted-token percentage is better.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td>k = 2</td><td>1.71 ×</td><td>64.71%</td><td>2.64</td></tr><tr><td>k = 4</td><td>2.35 ×</td><td>66.25%</td><td>3.55</td></tr><tr><td>k = 8</td><td>1.94 ×</td><td>87.11%</td><td>2.23</td></tr><tr><td>k = 16</td><td>1.37 ×</td><td>97.67%</td><td>1.41</td></tr><tr><td rowspan="4">n-gram SD</td><td>k = 2</td><td>1.76 ×</td><td>54.56%</td><td>3.23</td></tr><tr><td>k = 4</td><td>2.79 ×</td><td>54.14%</td><td>5.16</td></tr><tr><td>k = 8</td><td>3.59 ×</td><td>63.56%</td><td>5.64</td></tr><tr><td>k = 16</td><td>4.22 ×</td><td>79.90%</td><td>5.28</td></tr><tr><td rowspan="4">Draft-SD</td><td>k = 2</td><td>0.96 ×</td><td>27.82%</td><td>3.44</td></tr><tr><td>k = 4</td><td>0.87 ×</td><td>38.39%</td><td>2.26</td></tr><tr><td>k = 8</td><td>0.73 ×</td><td>52.46%</td><td>1.39</td></tr><tr><td>k = 16</td><td>0.50 ×</td><td>68.98%</td><td>0.72</td></tr><tr><td rowspan="2">APEX-S APEX-B</td><td>adaptive</td><td>4.27 ×</td><td></td><td></td></tr><tr><td>adaptive</td><td>3.27 ×</td><td>55.30% 47.17%</td><td>7.73 6.93</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>1.83 ×</td><td>63.69%</td><td>2.87</td></tr></table>

Workloads and datasets We evaluate six workloads: code generation using MBPP (Austin et al., 2021) and HumanEval (Chen & et al., 2021); conversational generation using UltraChat-200k (Ding et al., 2023); long-chain reasoning using GPQA (Rein et al., 2023); long-context completion using The Stack Smol (Kocetkov et al., 2022); software engineering using SWE-bench Lite (Jimenez et al., 2024); and mathematical reasoning using GSM8K (Cobbe et al., 2021) and AIME-style data (Mathematical Association of America, 1983). We follow the dataset-provided splits for training and evaluation.

Baselines and operating profiles We compare APEX with autoregressive decoding and fixeddepth EAGLE-3, n-gram, and draft-model speculation (Draft-SD). The fixed-depth sweep considers $k \in \{ 1 , 2 , 4 , 8 , 1 6 \}$ . We evaluate all three APEX operating profiles: APEX-S, APEX-B, and APEX-E.

![](images/9e49e7fbc04e35d957e04e0006771109ab68b10ff44459abff8c5aa80eabdea8.jpg)  
Figure 4: Speedup vs Token Wastage. The plot shows the trend of Speedup (vs AR) vs Token Wastage across different depths and mechanisms. The horizontal axis reports non-wasted draft tokens (100 − Waste%), and the vertical axis reports speedup over autoregressive decoding.

Metrics We measure wall-clock latency speedup relative to autoregressive decoding as

$$
\mathrm { S p e e d u p } = \frac { \mathrm { L a t e n c y } _ { \mathrm { A R } } } { \mathrm { L a t e n c y } _ { \mathrm { m e t h o d } } }\tag{34}
$$

Values above one indicate lower end-to-end latency. We also report wasted draft-token percentage,

$$
{ \mathrm { W a s t e } } _ { \% } = 1 0 0 \times { \frac { { \mathrm { R e j e c t e d ~ D r a f t ~ T o k e n s } } } { \mathrm { T o t a l ~ D r a f t ~ T o k e n s } } }\tag{35}
$$

which measures the fraction of proposed tokens discarded during speculation.

To summarize the speed–waste trade-off, we report Waste-Amortized Speedup (WAS):

$$
\mathrm { W A S } = \frac { \mathrm { S p e e d u p } } { \mathrm { W a s t e \% / 1 0 0 } } .\tag{36}
$$

Speedup and waste are averaged over the evaluated requests, and WAS is computed from the corresponding unrounded averages. Higher WAS is better, but we interpret it alongside latency speedup and waste to retain the distinction between these outcomes. A configuration is Pareto-efficient if no alternative is at least as fast and no more wasteful, with a strict improvement in at least one metric.

## 5.2 OVERALL RESULTS

Table 1 reports aggregate performance, and Figure 4 visualizes the same measurements as speedup versus the percentage of non-wasted draft tokens. APEX-S achieves the highest reported aggregate speedup of 4.27× with 55.30% waste. The fastest fixed baseline, n-gram speculation at $\bar { k } = 1 6 \bar { . }$ reaches 4.22× speedup with 79.90% waste. Thus, APEX-S slightly improves the reported speedup while reducing the wasted-token percentage by 30.8% relative to this baseline.

APEX-B provides a different operating point, achieving 3.27× speedup with 47.17% waste. Relative to n-gram speculation at k = 16, this corresponds to a 41.0% reduction in the wasted-token percentage, at the cost of lower acceleration. APEX-S and APEX-B obtain the highest aggregate WAS values of 7.73 and 6.93, respectively, compared with the best fixed-baseline WAS of 5.64 from n-gram speculation at k = 8.

The fixed baselines illustrate why speed and waste must be considered jointly. Increasing n-gram depth from k = 8 to k = 16 raises speedup from 3.59× to 4.22×, but increases waste from 63.56% to 79.90%, reducing WAS from 5.64 to 5.28. EAGLE-3 reaches its highest aggregate speedup at $k = 4 ;$ increasing its depth to $k = 1 6$ reduces speedup from 2.35× to 1.37× while raising waste from 66.25% to 97.67%. Conversely, Draft-SD at $k = 2$ has only 27.82% waste but achieves 0.96× speedup. Efficient draft-token utilization alone therefore does not ensure lower end-to-end latency.

Table 2: Workload-level speed–waste comparison for code generation.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td> $k = 2$ </td><td>2.11×</td><td>44.50%</td><td>4.74</td></tr><tr><td> $k = 4$ </td><td>2.78×</td><td>55.50%</td><td>5.01</td></tr><tr><td> $k = 8$ </td><td>2.01×</td><td>87.38%</td><td>2.30</td></tr><tr><td> $k = 1 6$ </td><td>1.27×</td><td>98.31%</td><td>1.29</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>2.08×</td><td>46.00%</td><td>4.52</td></tr><tr><td> $k = 4$ </td><td>2.31×</td><td>67.25%</td><td>3.43</td></tr><tr><td> $k = 8$ </td><td>3.47×</td><td>69.13%</td><td>5.02</td></tr><tr><td> $k = 1 6$ </td><td>4.25×</td><td>79.69%</td><td>5.33</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>1.00×</td><td>22.06%</td><td>4.53</td></tr><tr><td> $k = 4$ </td><td>0.90×</td><td>33.74%</td><td>2.67</td></tr><tr><td> $k = 8$ </td><td>0.71×</td><td>51.88%</td><td>1.37</td></tr><tr><td> $k = 1 6$ </td><td>0.49×</td><td>69.30%</td><td>0.71</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>2.80×</td><td>76.00%</td><td>3.68</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>1.99×</td><td>56.83%</td><td>3.50</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>0.99×</td><td>75.50%</td><td>1.31</td></tr></table>

Table 3: Workload-level speed–waste comparison for mathematical reasoning.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td> $k = 2$ </td><td>1.55×</td><td>72.50%</td><td>2.14</td></tr><tr><td> $k = 4$ </td><td>2.50×</td><td>62.50%</td><td>4.00</td></tr><tr><td> $k = 8$ </td><td>3.22×</td><td>72.25%</td><td>4.46</td></tr><tr><td> $k = 1 6$ </td><td>2.05×</td><td>93.44%</td><td>2.19</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>1.89×</td><td>55.50%</td><td>3.41</td></tr><tr><td> $k = 4$ </td><td>2.85×</td><td>53.75%</td><td>5.30</td></tr><tr><td> $k = 8$ </td><td>4.15×</td><td>60.62%</td><td>6.84</td></tr><tr><td> $k = 1 6$ </td><td>3.74×</td><td>82.88%</td><td>4.51</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>0.99×</td><td>28.22%</td><td>3.51</td></tr><tr><td> $k = 4$ </td><td>0.86×</td><td>41.46%</td><td>2.07</td></tr><tr><td> $k = 8$ </td><td>0.66×</td><td>59.84%</td><td>1.10</td></tr><tr><td> $k = 1 6$ </td><td>0.43×</td><td>75.66%</td><td>0.57</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>5.24×</td><td>47.00%</td><td>11.15</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>5.16×</td><td>44.00%</td><td>11.72</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>2.43×</td><td>72.90%</td><td>3.33</td></tr></table>

The three learned profiles also require separate evaluation. APEX-E achieves 1.83× speedup with 63.69% waste and is outperformed by both APEX-S and APEX-B on both metrics. Its efficiencyoriented objective does not produce the lowest aggregate waste, highlighting the importance of evaluating the realized behavior of each profile across workloads.

## 5.3 PERFORMANCE ACROSS WORKLOADS

The workload-level results reveal where adaptive control improves both speed and utilization, where it offers a trade-off, and where a fixed configuration remains preferable.

Mathematical reasoning. Table 3 shows the strongest acceleration achieved by APEX . APEX-S reaches 5.24× speedup with 47.00% waste, while APEX-B achieves 5.16× speedup with 44.00% waste. Both improve speed and reduce waste relative to the fastest fixed baseline, n-gram speculation

Table 4: Workload-level speed–waste comparison for long-context completion.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td>k = 2</td><td>1.59×</td><td>70.50%</td><td>2.26</td></tr><tr><td> $k = 4$ </td><td>1.89×</td><td>77.75%</td><td>2.43</td></tr><tr><td> $k = 8$ </td><td>0.51×</td><td>99.02%</td><td>0.52</td></tr><tr><td> $k = 1 6$ </td><td>1.26×</td><td>98.38%</td><td>1.28</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>1.22×</td><td>37.41%</td><td>3.26</td></tr><tr><td> $k = 4$ </td><td>3.32×</td><td>34.50%</td><td>9.62</td></tr><tr><td> $k = 8$ </td><td>2.03×</td><td>61.53%</td><td>3.30</td></tr><tr><td> $k = 1 6$ </td><td>6.40×</td><td>66.25%</td><td>9.66</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>0.89×</td><td>38.16%</td><td>2.34</td></tr><tr><td> $k = 4$ </td><td>1.08×</td><td>33.24%</td><td>3.24</td></tr><tr><td> $k = 8$ </td><td>1.07×</td><td>25.68%</td><td>4.17</td></tr><tr><td> $k = 1 6$ </td><td>0.87×</td><td>41.10%</td><td>2.12</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>5.04×</td><td>49.50%</td><td>10.18</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>1.83×</td><td>44.42%</td><td>4.12</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>2.72×</td><td>41.05%</td><td>6.63</td></tr></table>

Table 5: Workload-level speed–waste comparison for long-chain reasoning.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td>k = 2</td><td>1.46×</td><td>77.00%</td><td>1.90</td></tr><tr><td>k = 4</td><td>2.13×</td><td>71.75%</td><td>2.97</td></tr><tr><td> $k = 8$ </td><td>1.64×</td><td>92.00%</td><td>1.78</td></tr><tr><td> $k = 1 6$ </td><td>1.37×</td><td>97.69%</td><td>1.40</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>1.70×</td><td>65.00%</td><td>2.62</td></tr><tr><td> $k = 4$ </td><td>2.46×</td><td>63.50%</td><td>3.87</td></tr><tr><td> $k = 8$ </td><td>3.57×</td><td>67.88%</td><td>5.26</td></tr><tr><td> $k = 1 6$ </td><td>4.57×</td><td>77.69%</td><td>5.88</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>0.98×</td><td>27.79%</td><td>3.53</td></tr><tr><td> $k = 4$ </td><td>0.84×</td><td>41.58%</td><td>2.02</td></tr><tr><td> $k = 8$ </td><td>0.68×</td><td>59.04%</td><td>1.15</td></tr><tr><td> $k = 1 6$ </td><td>0.42×</td><td>75.18%</td><td>0.56</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>3.92×</td><td>63.50%</td><td>6.17</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>3.07×</td><td>40.59%</td><td>7.56</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>1.89×</td><td>58.12%</td><td>3.25</td></tr></table>

at k = 8, which achieves 4.15× speedup with 60.62% waste. APEX-B obtains the highest WAS of 11.72, retaining nearly the acceleration of APEX-S while discarding fewer draft tokens.

Conversational generation. A similar improvement appears in Table 6. APEX-S achieves 4.88× speedup with 51.50% waste, compared with 4.63× and 54.63% for n-gram speculation at $k = 8 .$ APEX-B reduces waste further to 46.56% while achieving 4.15× speedup. Increasing the fixed n-gram depth to $k = 1 6$ worsens both speed and waste, showing that deeper speculation is not uniformly beneficial even within a workload where n-gram proposals are effective.

Long-context completion. Table 4 presents a different trade-off. N-gram speculation at $k = 1 6$ remains the fastest configuration at 6.40× speedup with 66.25% waste. APEX-S achieves $5 . 0 4 \times$ speedup while reducing waste to 49.50%, yielding a higher WAS of 10.18 versus 9.66. This benefit does not extend to every learned profile: n-gram speculation at k = 4 is both faster and less wasteful than APEX-B and APEX-E.

Table 6: Workload-level speed–waste comparison for conversational generation.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td>k = 2</td><td>1.78×</td><td>61.00%</td><td>2.92</td></tr><tr><td> $k = 4$ </td><td>2.35×</td><td>66.25%</td><td>3.55</td></tr><tr><td> $k = 8$ </td><td>2.61×</td><td>79.88%</td><td>3.27</td></tr><tr><td> $k = 1 6$ </td><td>1.21×</td><td>98.69%</td><td>1.23</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>1.84×</td><td>58.00%</td><td>3.17</td></tr><tr><td> $k = 4$ </td><td>3.05×</td><td>48.75%</td><td>6.26</td></tr><tr><td> $k = 8$ </td><td>4.63×</td><td>54.63%</td><td>8.48</td></tr><tr><td> $k = 1 6$ </td><td>3.20×</td><td>86.25%</td><td>3.71</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>0.91×</td><td>29.03%</td><td>3.13</td></tr><tr><td> $k = 4$ </td><td>0.75×</td><td>43.01%</td><td>1.74</td></tr><tr><td> $k = 8$ </td><td>0.58×</td><td>61.95%</td><td>0.94</td></tr><tr><td> $k = 1 6$ </td><td>0.36×</td><td>77.78%</td><td>0.46</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>4.88×</td><td>51.50%</td><td>9.48</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>4.15×</td><td>46.56%</td><td>8.91</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>1.29×</td><td>74.03%</td><td>1.74</td></tr></table>

Table 7: Workload-level speed–waste comparison for swe-style tasks.
<table><tr><td>Method</td><td>Depth</td><td>Speedup vs. AR ↑</td><td>Wasted Token %↓</td><td>WAS ↑</td></tr><tr><td rowspan="4">EAGLE-3</td><td>k = 2</td><td>1.67×</td><td>66.50%</td><td>2.51</td></tr><tr><td>k = 4</td><td>2.45×</td><td>63.75%</td><td>3.84</td></tr><tr><td> $k = 8$ </td><td>0.98×</td><td>99.39%</td><td>0.99</td></tr><tr><td> $k = 1 6$ </td><td>1.24×</td><td>98.50%</td><td>1.26</td></tr><tr><td rowspan="4">n-gram SD</td><td> $k = 2$ </td><td>1.76×</td><td>62.00%</td><td>2.84</td></tr><tr><td> $k = 4$ </td><td>2.50×</td><td>62.50%</td><td>4.00</td></tr><tr><td> $k = 8$ </td><td>2.62×</td><td>76.53%</td><td>3.42</td></tr><tr><td> $k = 1 6$ </td><td>4.15×</td><td>80.31%</td><td>5.17</td></tr><tr><td rowspan="4">Draft-SD</td><td> $k = 2$ </td><td>1.02×</td><td>20.47%</td><td>4.98</td></tr><tr><td> $k = 4$ </td><td>0.90×</td><td>32.70%</td><td>2.75</td></tr><tr><td> $k = 8$ </td><td>0.81×</td><td>46.89%</td><td>1.73</td></tr><tr><td> $k = 1 6$ </td><td>0.54×</td><td>66.06%</td><td>0.82</td></tr><tr><td>APEX-S</td><td>adaptive</td><td>3.15×</td><td>48.12%</td><td>6.55</td></tr><tr><td>APEX-B</td><td>adaptive</td><td>2.54×</td><td>51.21%</td><td>4.96</td></tr><tr><td>APEX-E</td><td>adaptive</td><td>2.17×</td><td>50.23%</td><td>4.32</td></tr></table>

Long-chain reasoning. In Table 5, n-gram speculation at $k = 1 6$ achieves the highest speedup of 4.57× but wastes 77.69% of drafted tokens. APEX-B achieves 3.07× speedup with 40.59% waste, obtaining the highest WAS of 7.56. Here, adaptive control offers substantially better draft-token utilization while retaining acceleration over autoregressive decoding.

Software-engineering tasks. Table 7 shows that APEX-S achieves 3.15× speedup with 48.12% waste. N-gram speculation at k = 16 is faster at 4.15×, but its waste rises to 80.31%. APEX-S consequently achieves the highest WAS of 6.55, compared with 5.17 for that baseline. It also achieves both higher speedup and lower waste than APEX-B and APEX-E on this workload.

Code generation. Code generation exposes a limitation of the learned controller (Table 2). Ngram speculation at k = 16 achieves 4.25× speedup and a WAS of 5.33, exceeding all three APEX profiles on these metrics. The limitation also appears in direct speed–waste comparisons: n-gram speculation at k = 8 is both faster and less wasteful than APEX-S, while EAGLE-3 at k = 4 is both faster and less wasteful than APEX-B. These results identify code generation as a setting where improved routing and utility calibration are needed.

Table 8: Ablation study across operating profiles. Each variant removes one component from the corresponding full APEX controller. Speedup is measured relative to autoregressive decoding AR=1.00×
<table><tr><td>Profile</td><td>Variant</td><td>Speedup</td><td>Waste %</td><td>WAS</td></tr><tr><td rowspan="6">APEX-S</td><td>Full APEX-S</td><td>2.44 ×</td><td>59.22 %</td><td>4.12</td></tr><tr><td>w/o survival loss</td><td>1.62 ×</td><td>68.16 %</td><td>2.37</td></tr><tr><td>w/o cost/TPS objective</td><td>2.41 ×</td><td>62.69%</td><td>3.85</td></tr><tr><td>w/o entropy features</td><td>2.49 ×</td><td>62.41 %</td><td>3.99</td></tr><tr><td>w/o repetition features</td><td>2.42 ×</td><td>62.22 %</td><td>3.89</td></tr><tr><td>w/o verifier-history features</td><td>1.14 ×</td><td>54.74 %</td><td>2.09</td></tr><tr><td rowspan="6">APEX-E</td><td>Full APEX-E</td><td>2.34 ×</td><td>52.85 %</td><td>4.43</td></tr><tr><td>w/o survival loss</td><td>1.56 ×</td><td>67.62 %</td><td>2.31</td></tr><tr><td>w/o cost/TPS objective</td><td>2.26 ×</td><td>57.83 %</td><td>3.91</td></tr><tr><td>w/o entropy features</td><td>2.31 ×</td><td>58.06 %</td><td>3.98</td></tr><tr><td>w/o repetition features</td><td>2.29 ×</td><td>58.12 %</td><td>3.94</td></tr><tr><td>w/o verifier-history features</td><td>1.31 ×</td><td>56.48 %</td><td>2.32</td></tr><tr><td rowspan="6">APEX-B</td><td>Full APEX-B</td><td>2.40 ×</td><td>54.62 %</td><td>4.39</td></tr><tr><td>w/o survival loss</td><td>1.53 ×</td><td>69.38 %</td><td>2.21</td></tr><tr><td>w/o cost/TPS objective</td><td>2.33 ×</td><td>57.83 %</td><td>4.03</td></tr><tr><td>w/o entropy features</td><td>2.35 ×</td><td>58.34 %</td><td>4.03</td></tr><tr><td>w/o repetition features</td><td>2.30 ×</td><td>58.45 %</td><td>3.94</td></tr><tr><td>w/o verifier-history features</td><td>1.14 ×</td><td>56.22 %</td><td>2.03</td></tr></table>

Table 9: APEX candidate-k sweep under the quality-balanced operating profile. Higher speedup and WAS are better; lower wasted-token percentage is better.
<table><tr><td>Method</td><td>Candidate k set</td><td>Mean k</td><td>Speedup vs. AR</td><td>Wasted Token %</td><td>WAS</td></tr><tr><td rowspan="3">APEX-S</td><td>k = 1-16</td><td>3.61</td><td>3.34×</td><td>47.18%</td><td>7.07</td></tr><tr><td>k = 3-16</td><td>4.54</td><td>3.43×</td><td>50.57%</td><td>6.79</td></tr><tr><td>k ∈ {2, 4, 8, 12, 16}</td><td>3.92</td><td>3.39×</td><td>47.95%</td><td>7.06</td></tr><tr><td rowspan="3">APEX-B</td><td>k = 1-16</td><td>3.09</td><td>3.25×</td><td>42.76%</td><td>7.60</td></tr><tr><td>k = 3-16</td><td>4.28</td><td>3.34×</td><td>48.55%</td><td>6.89</td></tr><tr><td>k ∈ {2, 4, 8, 12, 16}</td><td>3.57</td><td>3.31×</td><td>45.10%</td><td>7.33</td></tr><tr><td rowspan="3">APEX-E</td><td>k = 1-16</td><td>2.96</td><td>1.56×</td><td>59.88%</td><td>2.61</td></tr><tr><td>k = 3-16</td><td>4.13</td><td>1.60×</td><td>67.83%</td><td>2.35</td></tr><tr><td>k ∈ {2, 4, 8, 12, 16}</td><td>3.40</td><td>1.58×</td><td>62.85%</td><td>2.52</td></tr></table>

## 5.4 ABLATION STUDY

We examine the contributions of survival supervision, cost and throughput supervision, and decoding-state features in Table 8. This study uses a smaller diagnostic subset containing twenty requests from each of the six workloads, for a total of 120 requests. All variants are evaluated on the same requests under the same serving configuration. Ablation effects are measured relative to the corresponding full controller on this subset; the absolute values therefore differ from the main evaluation.

Full-controller profiles. Among the full controllers, APEX-S achieves the highest speedup of 2.44× with 59.22% waste. APEX-B achieves 2.40× speedup with 54.62% waste, while APEX-E achieves 2.34× speedup with 52.85% waste and the highest WAS of 4.43. The profile ordering differs from the aggregate evaluation, further illustrating its dependence on the evaluated requests.

Survival supervision. Removing the survival loss produces the largest increase in waste among the examined ablations for all three profiles. Waste rises to 67.62%–69.38%, speedup falls to 1.53×– 1.62×, and WAS decreases to 2.21–2.37. These results support the role of first-rejection survival supervision in selecting depths that convert drafted tokens into accepted progress.

Cost and throughput supervision. Removing the cost/TPS objective causes smaller speed reductions but consistently increases waste and lowers WAS. For APEX-B, waste increases from 54.62% to 57.83%, while WAS decreases from 4.39 to 4.03. The results support accounting for runtime cost alongside acceptance behavior when comparing candidate actions.

Entropy and repetition features. Removing either feature group increases waste and reduces WAS across all three profiles. The effect on raw speedup is smaller and is not uniformly negative. For example, removing entropy features from APEX-S increases speedup from 2.44× to 2.49×, but also increases waste from 59.22% to 62.41%, lowering WAS from 4.12 to 3.99. These features therefore contribute primarily to the observed speed–waste balance.

Verifier-history features. Removing recent verifier feedback produces the largest speed reductions, with speedup falling to 1.14×–1.31× and WAS to 2.03–2.32. For APEX-S, waste actually decreases from 59.22% to 54.74%, but speedup falls from 2.44× to 1.14×. This illustrates why lower waste alone is insufficient and supports using recent acceptance and rejection outcomes to guide block-level decisions.

## 5.5 SENSITIVITY TO CANDIDATE DEPTHS

Table 9 reports a separate comparison of three candidate-depth sets across all APEX profiles: $\mathcal { K } _ { \mathrm { d e n s e } } = \{ 1 , \ldots , 1 6 \} , \mathcal { K } _ { \mathrm { d e e p } } = \{ 3 , \ldots , 1 6 \}$ , and $K _ { \mathrm { s p a r s e } } = \{ 2 , { \overset { . } { 4 } } , 8 , 1 2 , 1 6 \}$ . The dense set permits fine-grained choices, the deep set removes the two shallowest actions, and the sparse set reduces the number of available depths.

Removing shallow actions increases the mean selected depth and slightly improves speedup across all three profiles, but also increases waste and reduces WAS. For APEX-B, moving from the dense to the deep set raises speedup from 3.25× to 3.34×, while waste increases from 42.76% to 48.55% and WAS decreases from 7.60 to 6.89. Shallow actions therefore contribute to controlling waste even when excluding them produces a modest speed gain.

The sparse set provides an intermediate operating point. For APEX-S, it achieves 3.39× speedup with 47.95% waste, compared with 3.34× and 47.18% for the dense set. Their WAS values are nearly identical at 7.06 and 7.07. APEX-B and APEX-E also retain acceleration under the sparse set, although their WAS values remain below those obtained with the dense set. These results show that fine-grained depth choices are not equally important for every profile, while access to shallow actions consistently improves the measured speed–waste balance in this study.

## 5.6 DISCUSSION

The results support adaptive speculative control while establishing its workload-dependent limits. APEX-S provides the highest aggregate speedup, and APEX-B offers lower aggregate waste at reduced acceleration. Mathematical reasoning and conversational generation show simultaneous improvements over the fastest fixed baselines, while other workloads expose trade-offs or favor fixed configurations. The ablations support combining survival supervision, runtime-cost supervision, and recent verifier feedback. Together, these findings motivate evaluating speculative controllers through both latency and draft-token utilization, with profile selection guided by the workload and the desired operating point.

## 6 CONCLUSION AND FUTURE WORK

APEX combines request-level speculator selection with block-level depth adaptation to balance decoding speed and draft-token waste. Its learned controller provides a modular way to coordinate heterogeneous proposal mechanisms while preserving target-model verification. The results demonstrate substantial acceleration with lower waste than aggressive fixed-depth configurations, with trade-offs that vary across workloads.

Future work will use traces collected during decoding to update the controller online as workloads and execution conditions change. We also plan to evaluate APEX beyond single-turn prompts, including multi-turn conversations, reinforcement-learning rollouts, and tool-augmented generation. These settings would test how well the controller responds to changes in context and generation patterns over successive steps. To expand beyond the current three speculative methods, we plan to add a common interface for new speculators and their execution traces. The router and depth controlle could then be retrained on these traces to incorporate the additional methods.

## REFERENCES

Amey Agrawal et al. Taming throughput-latency tradeoff in llm inference with sarathi-serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI), 2024.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Jiang Ellen, Carrie J Cai, Michael Terry, Quoc V Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Shrenik Bhansali and Larry Heck. Draft, verify, and improve: Toward training-aware speculative decoding, 2025. URL https://arxiv.org/abs/2510.05421.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024. URL https://arxiv.org/abs/2401.10774.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling, 2023. URL https://arxiv.org/abs/2302.01318.

Mark Chen and et al. Evaluating large language models trained on code, 2021.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Zhi Zheng, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations, 2023. URL https://arxiv.org/abs/2305.14233.

Zhenyu He, Zexuan Zhong, Tianle Cai, Jason D. Lee, and Di He. Rest: Retrieval-based speculative decoding, 2024. URL https://arxiv.org/abs/2311.08252.

Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent Y. F. Tan, and Zhuoran Yang. Banditspec: Adaptive speculative decoding via bandit algorithms, 2025. URL https://arxiv.org/abs/2505.15141.

Yuxuan Hu, Ke Wang, Xiaokang Zhang, Fanjin Zhang, Cuiping Li, Hong Chen, and Jing Zhang. Sam decoding: Speculative decoding via suffix automaton, 2024. URL https://arxiv. org/abs/2411.10666.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview. net/forum?id=VTF8yNQM66.

Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Munoz Fer-˜ randis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. The stack: 3 tb of permissively licensed source code. ArXiv, abs/2211.15533, 2022. URL https://api.semanticscholar.org/ CorpusID:254044610.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention, 2023a. URL https://arxiv.org/abs/2309.06180.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles (SOSP), pp. 611–626. ACM, 2023b.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding, 2023. URL https://arxiv.org/abs/2211.17192.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test, 2025a. URL https://arxiv.org/ abs/2503.01840.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty, 2025b. URL https://arxiv.org/abs/2401.15077.

Jonathan Mamou, Oren Pereg, Daniel Korat, Moshe Berchansky, Nadav Timor, Moshe Wasserblat, and Roy Schwartz. Dynamic speculation lookahead accelerates speculative decoding of large language models, 2024. URL https://arxiv.org/abs/2405.04304.

Mathematical Association of America. American invitational mathematics examination (aime), 1983. URL https://maa.org.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS ’24, pp. 932–949. ACM, April 2024. doi: 10.1145/3620666.3651335. URL http://dx.doi.org/10.1145/3620666.3651335.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark, 2023. URL https://arxiv.org/abs/2311.12022.

Dale Schuurmans, Hanjun Dai, and Francesco Zanini. Autoregressive large language models are computationally universal, 2024. URL https://arxiv.org/abs/2410.03170.

Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pp. 3909–3925, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-emnlp.257. URL https://aclanthology.org/2023. findings-emnlp.257/.

Mingyu Yang, Jae-Young Choi, Kihyo Moon, Minsung Jang, and Eunjoo Jeon. Dsde: Dynamic speculative decoding with kld stability for real-world serving, 2025. URL https://arxiv. org/abs/2509.01083.

Situo Zhang, Hankun Wang, Da Ma, Zichen Zhu, Lu Chen, Kunyao Lan, and Kai Yu. Adaeagle: Optimizing speculative decoding via explicit modeling of adaptive draft structures, 2024. URL https://arxiv.org/abs/2412.18910.

Xiandong Zou, Jianshu Li, Jing Huang, and Pan Zhou. Variational speculative decoding: Rethinking draft training from token likelihood to sequence acceptance, 2026. URL https://arxiv. org/abs/2602.05774.