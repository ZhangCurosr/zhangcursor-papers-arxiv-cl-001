# GROWMTP: CAN RL GROW ITS OWN DRAFT HEAD?

Minghua He<sup>1,2∗</sup>, Lingzhe Zhang<sup>2</sup>, Yuan Liu<sup>1</sup>, Xiao Zhou<sup>1</sup>, Aiwei Liu<sup>1†</sup>

<sup>1</sup>WeChat AI, Tencent, <sup>2</sup>Peking University

hemh2120@stu.pku.edu.cn, coveliu@tencent.com

## ABSTRACT

Reinforcement learning (RL) post-training drives the frontier capabilities of large language models, with its wall-clock dominated by autoregressive rollout generation. Speculative decoding is an established remedy for this bottleneck, but existing draft heads must be pretrained or warmed up before RL, introducing substantial training cost outside the RL run to be accelerated. We observe that RL training itself provides both conditions required for online draft-head training: its rollout distribution is far narrower than that of pretraining, and its verification step continuously produces supervision signals aligned with this distribution. Building on these observations, we propose GrowMTP, which uses this supervision to train a draft head from scratch entirely within the RL loop, with all head updates detached from the policy backbone. On Qwen3-4B (no draft head), MiMo-7B-SFT (weak head), and Qwen3.5-4B-Base (strong head), GrowMTP achieves rollout speedups of 2.13×, 1.93×, and 1.36×, and end-to-end speedups of 1.60×, 1.41×, and 1.20×, respectively. GrowMTP therefore serves existing RL training frameworks as a modular component, particularly offering a from-scratch acceleration path for models without pretrained draft heads. Website: https://growmtp.github.io/.

## 1 INTRODUCTION

![](images/ab1ac8443716bfe916334e69f1022cbcbd2ad174174368b3de46488dff9d98bf.jpg)

![](images/5e1987df76f36168342a198f42ad1b8a56dbef0ea53f8be22d8b9f7b8b269751.jpg)

![](images/09628cf5da2a217cc1b569ff89570cce3cc9805bdeb7df28f0941490da29770c.jpg)  
Figure 1: An RL run can grow its own draft head from scratch and reduce its training compute. (a) The acceptance length grows from 1.00 to 2.91. (b) This yields 2.13× rollout and 1.60× endto-end speedups, including head updates. (c) GrowMTP uses less total compute than matched offline-trained and no-head baselines. Qwen3-4B, 500 RL steps on DAPO-Math-17K.

Reinforcement learning (RL) post-training now drives the frontier capabilities of large language models (Jaech et al., 2024; Guo et al., 2025). The RL stage accounts for a substantial share of the training budget (Khatri et al., 2026): OpenAI o3 was trained with 10× the RL compute of o1 (Singh et al., 2025). Within this budget, the dominant cost is rollout generation: the policy decodes tens of thousands of tokens per response token by token, typically 70–90% of each training step (Chen et al., 2026). The bottleneck is thus not the RL algorithm itself but autoregressive decoding.

This bottleneck is not unique to RL: it has long been central to model serving, where speculative decoding is the established solution (Leviathan et al., 2023; Chen et al., 2023). A cheap draft proposes several tokens, the target model verifies them in a single parallel forward pass, and rejection sampling preserves the output distribution exactly. Modern drafts are lightweight auxiliary (Ankner et al., 2024; Li et al., 2024a;b) or native multi-token-prediction (MTP) heads (Cao et al., 2026; Xiaomi et al., 2025), which reuse the backbone’s hidden states at negligible cost and are now standard components of frontier models. Porting this machinery to RL rollouts is therefore a natural step.

However, existing approaches train draft heads before RL (Cai et al., 2024). An EAGLE-style head for a 70B target model takes roughly 5,000 offline GPU hours (Li et al., 2026b; Hu et al., 2026), while native MTP heads are trained throughout pretraining, 14.8T tokens for DeepSeek-V3 (Liu et al., 2024). Improvement is no different: drafting depth depends on training, and raising acceptance at larger K typically requires another offline stage. Drafting capability thus enters RL as a prerequisite, and a model without a head has no route to acceleration. In effect, the established remedy for the dominant cost of RL is itself a further training cost, incurred before RL benefits at all.

We argue that this tension stems from a twofold mismatch between how draft heads are trained and what an RL process both requires and provides. Draft heads brought into RL are pretrained for broad deployment across diverse tasks. What RL requires is much narrower: rollouts concentrate on a particular task distribution, and the head only needs to fit it. Meanwhile, existing pipelines source draft-head supervision entirely outside RL. Yet during speculative rollouts, the target model verifies every draft, producing predictive distributions and acceptance decisions that can supervise the draft head but are otherwise discarded. An RL process can thus grow its own draft head from the rollouts it already requires and accelerate subsequent rollouts, without an upfront training bill.

Although RL provides the distribution and the supervision, training a multi-step draft head online presents three challenges. First, draft-head training must remain consistent with rollout-time drafting. Each proposal is conditioned on the preceding drafts, so its verification signal must be applied under the same draft-conditioned state. Second, learning across drafting depths is inherently dependent. The k-th draft token counts only if the preceding k − 1 are accepted, so early rejections limit gains from deeper predictions. Training should therefore reflect this dependence when assigning supervision across drafting depths. Finally, not every verification signal corresponds to an executed prefix. Once the target rejects a draft token, deeper drafts remain conditioned on it and no longer follow the executed sequence, so supervising all positions equally also trains the head on discarded suffixes.

To meet these demands, we introduce GrowMTP, which trains the draft head online from rollout verification signals. To preserve consistency with rollout-time drafting, GrowMTP records the draft trajectory and reconstructs its draft-conditioned states, applying each signal under the state that produced it. To encode the dependence across drafting depths, we design a depth-coupled acceptance (DCA) loss based on an acceptance-chain surrogate. We apply a logarithmic transformation to give cycles with smaller chain values greater relative gradient weight. Because this objective can still assign nonzero gradients beyond the first rejection, GrowMTP gates the loss there and excludes subsequent positions. All draft-head updates are detached from the policy backbone, without directly interfering with policy learning. Since the first draft position always provides valid supervision, training can begin from random initialization without an offline stage or a depth schedule.

To evaluate from-scratch training, we train a random K=5 head on Qwen3-4B (Yang et al., 2025) for 500 math RL steps. Its acceptance length grows from the autoregressive floor to 2.91, accelerating rollouts by 2.13× and end-to-end steps by 1.60×, with early per-step speedups. Over 500 steps, GrowMTP reduces total GPU-hours by 9.4% versus an offline-trained head at comparable acceptance. GrowMTP also accelerates models with pretrained heads: on Qwen3.5-4B-Base (Qwen, 2026), online updates accelerate rollouts by 1.36× over a frozen head; on MiMo-7B-SFT (Xiaomi et al., 2025), extending the head from K=1 to K=5 raises acceptance from 1.65 to 4.0 and accelerates rollouts by 1.93×. Heads trained on Math and Code rollouts achieve higher in-domain acceptance. Thus, in-RL draft-head training does not replace MTP pretraining but suffices for the current RL run.

In summary, our contributions are as follows:

• Revisiting MTP for RL acceleration. We show that an RL run only requires a draft head adapted to its rollout distribution and continuously produces matching verification supervision, allowing such a head to be trained during RL rather than pretrained.

• GrowMTP. We propose GrowMTP, which trains an MTP head online using valid verification supervision under rollout-consistent draft states, extends learning along the accepted draft chain, and retains the original RL objective.

• From-scratch acceleration. On the headless Qwen3-4B, GrowMTP grows a random K=5 head for a 2.13× rollout and 1.60× end-to-end speedup, head-training cost included.

• Modular RL acceleration. GrowMTP extends speculative acceleration to models without pretrained MTP heads, which grow one during RL instead.

## 2 PRELIMINARIES

## 2.1 REINFORCEMENT LEARNING FOR LARGE LANGUAGE MODELS

We consider Group Relative Policy Optimization (GRPO)-style RL post-training (Zheng et al., 2025), in which the policy $\pi _ { \theta }$ , the language model itself, samples a group of G responses for each prompt x and receives a scalar reward per response. Each response y is generated autoregressively: the policy emits one token per forward pass, $y _ { t } \sim \pi _ { \theta } ( \cdot \mid x , y _ { < t } )$ . A training step thus comprises two phases: rollout, which generates the responses, and update, which converts the rewards into advantages and adjusts the policy. The cost of a step is dominated by rollout, because of this token-by-token decoding, and this work accordingly targets the rollout bottleneck while retaining the original RL objective.

## 2.2 SPECULATIVE DECODING AND MULTI-TOKEN PREDICTION

Speculative decoding generates tokens in draft-then-verify cycles. We use a recurrent MTP head (Xu et al., 2026) $h _ { \phi }$ as the draft head, reusing backbone hidden states to propose K tokens per cycle; K is the drafting depth. Drafting is itself autoregressive: the k-th draft token $d _ { k }$ is sampled from a head distribution $q _ { k } ( \cdot \mid x , y _ { \leq t } , d _ { < k } )$ , conditioned on the committed context and on the draft tokens before it. The target model then verifies all K candidates in a single forward pass, producing its own distribution $p _ { k }$ at every draft position, and accepts $d _ { k }$ by rejection sampling, with probability

$$
\operatorname* { m i n } \biggl ( 1 , \frac { p _ { k } ( d _ { k } ) } { q _ { k } ( d _ { k } ) } \biggr ) .\tag{1}
$$

At the first rejection, a replacement token is resampled from a corrected distribution, and if all K candidates are accepted, a bonus token is drawn from the target’s next-position distribution $p _ { K + 1 }$ , so the committed sequence remains distributed exactly as if the target had decoded autoregressively (Xia et al., 2024; Miao et al., 2024). The tokens committed in one cycle, the accepted prefix plus this final token, always number at least one: even a draft rejected at the first position commits the resampled token, regardless of the quality of the draft. We call the average number of tokens committed per cycle the acceptance length $\tau \in [ 1 , K + 1 ]$ : rollout throughput grows in proportion to τ, up to the drafting overhead, with $\tau = 1$ recovering plain autoregressive decoding.

## 2.3 TRAINING DRAFT HEADS FROM RL ROLLOUTS

We observe that a draft head’s ability to accelerate a given RL process depends on its acceptance on that process’s rollout distribution, rather than its general drafting capability. At the same time, every speculative rollout produces target distributions and acceptance decisions on this distribution, verification supervision for exactly the proposals generated. The first draft position always provides supervision, so head training can begin from random initialization using the rollout records that RL already produces for policy optimization, without a separate dataset or an offline head-training stage.

However, using this supervision effectively for multi-step online training presents three challenges. First, each verification signal corresponds to a proposal generated under a particular draft-conditioned state, so training must apply the signal under the same state. Second, position k increases the acceptance length only if the preceding $k - 1$ drafts are accepted, so training should reflect this dependence across depths. Finally, once the target rejects a draft token, subsequent drafts remain conditioned on it and no longer follow the executed sequence; supervising all positions equally also trains the head on discarded suffixes that were drafted but never committed to the executed response.

## 3 GrowMTP: GROWING DRAFT HEADS WITHIN RL

GrowMTP uses verification supervision from each rollout to train the draft head for subsequent rollouts, forming a training-and-acceleration loop within RL. Figure 2 illustrates both phases: rollout records the realized draft trajectory and verification signals, while update trains the head from detached backbone states alongside the policy update. Using these records, GrowMTP reconstructs the draft-conditioned state under which each signal was produced, optimizes an acceptance-chain surrogate with the DCA loss, and excludes positions beyond the first rejection. The head loss updates only draft-head parameters, while the backbone optimizes the original RL objective. Appendix F presents the full training algorithm, and Appendix E describes its implementation.

![](images/dbf5d8c32dbb7e426c3017a7073016ab43c6bc05cf1e5bc3dd0a0090eb13fe3c.jpg)  
Figure 2: The GrowMTP pipeline. Rollout performs speculative decoding as usual and records the drafts, verification logits, and acceptance decisions of each cycle. Update trains the draft head on these records through detached hidden states, alongside the policy update.

## 3.1 TRAINING PIPELINE AND DRAFT-PATH RECONSTRUCTION

Training Pipeline. GrowMTP preserves the two phases of the existing RL training step: rollout performs speculative decoding as usual, and update continues the original policy optimization. During each draft-then-verify cycle, rollout records the backbone hidden states at the start of the cycle, the drafts generated by the head, the target verification logits, and the acceptance decision at each position. These quantities are already produced by drafting and verification, so collecting the training supervision requires no additional target-model forward pass. During update, the original RL objective continues to update the policy, while these rollout records separately train the draft head. The recorded backbone hidden states enter the head detached, so the head loss updates only the head parameters ϕ and sends no gradient to the backbone parameters θ. Figure 2 illustrates this pipeline.

Draft-Path Reconstruction. A natural use of these records treats the final response as an ordinary token sequence and trains the head on it by a right shift, which is teacher forcing (Gloeckle et al., 2024), constructed in full in Appendix G. Figure 3 contrasts the two training paths. Consider draft-then-verify cycle r: its committed context c<sub>r</sub> is the prompt plus all tokens committed earlier, and its draft path $\mathbf { d } _ { r } = [ d _ { r , 1 } , \dots , d _ { r , K } ]$ its K draft tokens. During verification, the target model z<sub>θ</sub> conditions its logits on $\mathbf { c } _ { r }$ and the preceding drafts, yielding the target distribution at position k:

$$
p _ { r , k } = \mathrm { s o f t m a x } \big ( z _ { \theta } ( \mathbf { c } _ { r } , d _ { r , 1 } , \ldots , d _ { r , k - 1 } ) \big ) .\tag{2}
$$

Teacher forcing conditions position k on the committed token rather than the head’s own draft: the two are identical only when that draft was accepted. If position j is the first rejection, the target replaces $d _ { r , j }$ with $\tilde { d } _ { r , j } \ne d _ { r , j }$ , and the tokens that follow, written $\tilde { d } _ { r , j + 1 } , \ldots$ . and committed by subsequent cycles, continue from this replacement, so the paths diverge as

$$
\mathbf { d } _ { r } ^ { \mathrm { { d r a f t } } } = [ d _ { r , < j } , d _ { r , j } , d _ { r , j + 1 } , . . . ] , \qquad \mathbf { d } _ { r } ^ { \mathrm { { c o m m i t } } } = [ d _ { r , < j } , \tilde { d } _ { r , j } , \tilde { d } _ { r , j + 1 } , . . . ] .\tag{3}
$$

The head consumes at each position the hidden state and token embedding of the position before it, so the committed path supplies a different pair at every position past j, along with different cached keys and values: teacher forcing fits the head at a state that drafting reaches only when the whole prefix is accepted. The training state at position k agrees with the drafting state if and only if none of the preceding k − 1 positions was rejected, and this condition tightens monotonically in k. The bias of teacher forcing therefore grows with draft depth, and drafting depth is what GrowMTP extends during RL. We discuss this train–inference inconsistency in detail in Appendix G.

The final response neither marks cycle boundaries nor retains the drafts the target replaced, so both must be recorded during rollout. GrowMTP therefore applies Draft-Path Reconstruction, which starts from the recorded cycle context and feeds the recorded drafts back in their original order:

$$
\begin{array} { r } { s _ { r , 1 } = \mathbf { c } _ { r } , \qquad s _ { r , k + 1 } = [ s _ { r , k } , d _ { r , k } ] , \qquad q _ { r , k } = h _ { \phi } ( \cdot \mid s _ { r , k } ) . } \end{array}\tag{4}
$$

This recursion gives $s _ { r , k } = [ \mathbf { c } _ { r } , d _ { r , 1 } , \ldots , d _ { r , k - 1 } ]$ , pairing every $p _ { r , k }$ with a head distribution under the same cycle context and realized draft prefix. We provide the implementation details of the reconstruction in Appendix E. The only remaining asymmetry lies between successive RL steps: the head learns from logits of the current rollout but drafts for a policy one update newer, and Appendix H bounds the resulting discrepancy and shows it remains small in our runs.

![](images/af60ee1304a160060f28a5c19e24ac04cb3fcd121e28df82c3c13cf97f0680a7.jpg)  
Figure 3: Rollout cycles and the two training paths. Teacher forcing (C) trains on the committed path. Reconstruction (D) feeds the recorded drafts back, and VGM truncates at the first rejection.

## 3.2 THE DEPTH-COUPLED ACCEPTANCE LOSS

To target acceleration, we construct a differentiable surrogate for the expected accepted draft length from the reconstructed pairs. The building block is the acceptance probability at a single position: writing $p _ { k }$ and $q _ { k }$ for the paired target and head distributions at position k within a cycle, rejection sampling accepts a token sampled from the head with probability averaged over possible draft tokens:

$$
\alpha _ { k } = \sum _ { d } \operatorname* { m i n } \bigl ( p _ { k } ( d ) , q _ { k } ( d ) \bigr ) = 1 - \mathrm { T V } ( p _ { k } , q _ { k } ) ,\tag{5}
$$

where TV is the total variation distance. Acceptance is chain-dependent: the l-th token contributes only when positions 1 through l are all accepted. Chaining these local rates yields a differentiable surrogate for the accepted length, whose negation is the TV loss (Zhou et al., 2024; Li et al., 2026a)

$$
\mathcal { L } _ { \mathrm { T V } } ~ = ~ - \sum _ { l = 1 } ^ { K } \prod _ { i = 1 } ^ { l } \alpha _ { i } .\tag{6}
$$

To give cycles with smaller chain values greater relative gradient weight, we apply a logarithmic transformation to the acceptance-chain surrogate to obtain the DCA training objective (Appendix I):

$$
\mathcal { L } _ { \mathrm { { D C A } } } ~ = ~ - \log \bigl ( - \mathcal { L } _ { \mathrm { { T V } } } \bigr ) ~ = ~ - \log \sum _ { l = 1 } ^ { K } \exp \Bigl ( \sum _ { i = 1 } ^ { l } \log \alpha _ { i } \Bigr ) ,\tag{7}
$$

minimized with respect to the head alone: the gradient flows through each $q _ { k }$ , and the recorded $p _ { k }$ stay constant. The logarithm preserves the gradient direction on each fixed cycle but changes the relative cycle weights in the batch, so the two batch objectives can have different stationary points.

## 3.3 LEARNING UP TO THE FIRST REJECTION

After the first rejection, later draft positions remain conditioned on discarded tokens and no longer follow the executed sequence. Verification still produces target distributions at each of these draft positions, so treating every position equally also assigns full training weight to the discarded suffix.

However, differentiating DCA with respect to $\alpha _ { k }$ factors the gradient weight at position k into the product of the preceding acceptance probabilities and a bounded tail term,

$$
- \frac { \partial \mathcal { L } _ { \mathrm { D C A } } } { \partial \alpha _ { k } } = \Bigl ( \prod _ { i < k } \alpha _ { i } \Bigr ) \frac { D _ { k } } { - \mathcal { L } _ { \mathrm { T V } } } , \qquad D _ { k } = \sum _ { l = k } ^ { K } \prod _ { i = k + 1 } ^ { l } \alpha _ { i } \in [ 1 , K - k + 1 ] ,\tag{8}
$$

where beyond the leading product, its denominator shared across positions, the weights differ at most by a factor of K, with the full derivation given in Appendix I. The leading product weights supervision at position k by the acceptance overlaps of preceding drafts.

This soft weighting does not enforce the realized rejection boundary: positions after the first rejection can still receive nonzero gradients. Since verification already marks the first-rejection position j of

each cycle, with $j = K { + } 1$ when every draft is accepted, we introduce verify-gated masking (VGM), which truncates the chain sum at and including the first rejected position of the recorded cycle,

$$
\mathcal { L } _ { \mathrm { D C A } } ^ { \mathrm { V G M } } = - \log \sum _ { l = 1 } ^ { \operatorname* { m i n } ( j , K ) } \prod _ { i = 1 } ^ { l } \alpha _ { i } ,\tag{9}
$$

removing supervision beyond the first rejection. The chain sum includes the first rejected position, if any, and always includes the first draft position. Supervision is therefore available from the first cycle, allowing training to begin even with a randomly initialized head. The resulting supervision is also self-paced: its depth follows the realized accepted prefix, while gradient weights reflect the acceptance overlaps of preceding drafts. This requires no shallow-depth warm-up or schedule for K.

Each update minimizes the mean of Eq. 9 over cycles from the rollout policy and draft head, holding recorded tokens, target distributions, and j fixed. Gradients flow through reconstructed head distributions, excluding sampling and rejection decisions. This fixed-record objective is an acceptance surrogate; unbiasedness for deployment-time expected accepted length is not assumed

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETUP

Models. Qwen3-4B (Yang et al., 2025) has no pretrained draft head: we attach a single randomly initialized EAGLE-style MTP layer and apply it recurrently at K=5. Qwen3.5-4B-Base (Qwen, 2026) has a native MTP head (K=3), isolating the online training strategy without cold-start interference. MiMo-7B-SFT (Xiaomi et al., 2025) has an MTP head (K=1), isolating how online training extends drafting depth from shallow capability. The three heads take the same form, detailed in Appendix B. Appendix D.1 discusses which models GrowMTP can accelerate and the signal is model-agnostic.

Datasets. We evaluate GrowMTP in two representative domains: mathematical reasoning and code reasoning. For RL training, we use DAPO-Math-17K (Yu et al., 2026) and TACO-Verified (Li et al., 2023), two established datasets. Inference evaluation uses AMC23 (Zhang & Math-AI, 2023), AIME24, and AIME25 (Zhang & Math-AI, 2024; 2025) for mathematical reasoning, and LiveCodeBench v6 (Jain et al., 2025) with its AtCoder and LeetCode slices for code reasoning, detailed in Appendix B. GrowMTP thus requires no data constructed for draft-head training.

Metrics. For the training evaluation, we measure rollout time, end-to-end step time with head updates included, and the acceptance length τ, with speedups over the same run with the head frozen. For the Inference evaluation, we report policy quality, Mean@16 for mathematical reasoning (Team et al., 2025) and Pass@4 for code reasoning (Li et al., 2026c), to assess the effect of draft-head training on policy quality, and the held-out τ, with full protocols in Appendix B.4.

Implementation Details. We implement GrowMTP on verl (Sheng et al., 2025), with rollouts served by SGLang (Zheng et al., 2024) and the draft head in its speculative decoding path. Each RL step samples 64 prompts with 8 responses each, capped at 8192 tokens (16384 on Qwen3.5-4B-Base), and Qwen3-4B runs in thinking mode. Rollouts and evaluation use rejection sampling at temperature 1. Scaling K introduces no new parameters, the same head applied recurrently, and the head is updated once per RL step from that step’s recorded signals. Full configurations are given in Appendix B.

## 4.2 RQ1: CAN AN RL RUN GROW ITS OWN DRAFT HEAD FROM SCRATCH?

GrowMTP vs. Baseline. We train Qwen3-4B for 500 RL steps in each domain with a randomly initialized K=5 head. As shown in Table 1 and Appendix C.1, the acceptance length τ grows from the autoregressive floor of 1.00 to 2.91 on mathematical reasoning and 2.64 on code reasoning, and the growth begins immediately, as the first draft position supplies valid supervision even under random initialization. This growth converts into rollout speedups of 2.13× and 1.93×, and with head-update cost included, end-to-end step speedups of 1.60× and 1.56×. Held-out benchmark scores are comparable. The acceleration is not free early on, as the random head accepts little while already paying update cost, but per-step time falls below the baseline within the first 30 steps in both domains and subsequent steps remain faster. Appendix D.2 projects this speedup at longer rollouts and more steps. GrowMTP can thus train an MTP head from scratch during RL and use it to accelerate the run itself. Appendix D.3 shows that the grown head specializes to that run.

Table 1: Training evaluation on Qwen3-4B (500 RL steps, K=5). A draft head grown from scratch delivers end-to-end step speedups of 1.60× and 1.56× over autoregressive decoding (K=0).
<table><tr><td></td><td></td><td colspan="5">Math Reasoning RL</td><td colspan="5">Code Reasoning RL</td></tr><tr><td>Method</td><td>K</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td></tr><tr><td>AR RL</td><td>0</td><td>523.86</td><td>1.00×</td><td>723.51</td><td>1.00×</td><td>1.00</td><td>771.22</td><td>1.00×</td><td>1028.39</td><td>1.00×</td><td>1.00</td></tr><tr><td>GrowMTP</td><td>5</td><td>245.95</td><td>2.13×</td><td>452.53</td><td>1.60×</td><td>2.91</td><td>398.65</td><td>1.93×</td><td>659.39</td><td>1.56×</td><td>2.64</td></tr></table>

Online vs. Offline. We further compare against the established offline route, generating rollouts from the same initial checkpoint on the same dataset, training an identical head with the same objective, and then running RL with that fully trained head frozen throughout. As shown in Table 2, both routes reach nearly identical serving acceptance, 2.92 offline and 2.91 online. The offline route, however, totals 555.3 against 502.8 GPU·h for GrowMTP, a 9.4% difference over the 500-step run. The gap arises at generation:

Table 2: GrowMTP avoids 161.6 GPU·h offline head-training stage. Costs in GPU·h; Qwen3-4B, 500 RL steps on DAPO-Math-17K, K=5.
<table><tr><td>Method</td><td>Offline (h)</td><td>RL (h)</td><td>Total (h)</td><td>T</td></tr><tr><td>AR RL (K = 0)</td><td>0</td><td>803.9</td><td>803.9</td><td>1.00</td></tr><tr><td>Offline (K = 5)</td><td>161.6</td><td>393.7</td><td>555.3</td><td>2.92</td></tr><tr><td>GrowMTP (K = 5)</td><td>0</td><td>502.8</td><td>502.8</td><td>2.91</td></tr></table>

offline training must produce its trajectories before any head exists, whereas GrowMTP learns from rollouts that RL must generate anyway. A separate offline stage is thus unnecessary for accelerating the run: GrowMTP obtains an equally capable head at lower cost from the RL process itself.

## 4.3 RQ2: HOW SHOULD THE DRAFT HEAD BE TRAINED ONLINE?

Training Strategy. On Qwen3.5-4B-Base, we train for 200 RL steps in each domain under four strategies: frozen (Chandiramani et al., 2026), joint CE (Wang et al., 2026), detached CE (Zeng et al., 2025), and detached DCA, the last being GrowMTP. Frozen is the prevailing practice in current RL pipelines, and the two CE arms transplant the established recipe online, differing only in whether gradients reach the backbone. As shown in Table 3 and Appendix C.2, joint training reaches the highest τ, 3.99 and 3.42, yet its held-out quality is zero in both domains: τ can rise as the policy that produces it degrades. The detached CE arm avoids the observed collapse and accelerates the run: τ reaches 3.18 and 3.06, and rollouts speed up by 1.29× and 1.17×. Under the same detachment at K=3, DCA leads CE on every training-side column and every held-out τ , with end-to-end step speedups of 1.20× and 1.12× including head-update cost and held-out quality on par with the frozen head. The draft head should thus be trained online, but with gradients detached from the policy and an objective aligned with acceptance: GrowMTP further improves a pretrained head.

Table 3: Training evaluation on Qwen3.5-4B-Base (200 RL steps, K=3). Both detached arms accelerate. DCA leads among quality-preserving strategies. Joint CE speeds up code but scores zero.
<table><tr><td></td><td></td><td colspan="5">Math Reasoning RL</td><td colspan="5">Code Reasoning RL</td></tr><tr><td>Training Mode</td><td>Loss</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td></tr><tr><td>Frozen</td><td></td><td>357.62</td><td>1.00×</td><td>554.42</td><td>1.00×</td><td>2.65</td><td>500.09</td><td>1.00×</td><td>778.54</td><td>1.00×</td><td>2.59</td></tr><tr><td>Joint</td><td>CE</td><td>372.87</td><td>0.96×</td><td>643.76</td><td>0.86×</td><td>3.99</td><td>258.34</td><td>1.94×</td><td>411.88</td><td>1.89×</td><td>3.42</td></tr><tr><td>Detached</td><td>CE</td><td>277.14</td><td>1.29×</td><td>507.91</td><td>1.09×</td><td>3.18</td><td>426.88</td><td>1.17×</td><td>719.40</td><td>1.08×</td><td>3.06</td></tr><tr><td>Detached</td><td>DCA (GrowMTP)</td><td>263.86</td><td>1.36×</td><td>461.56</td><td>1.20×</td><td>3.22</td><td>408.89</td><td>1.22×</td><td>694.00</td><td>1.12×</td><td>3.11</td></tr></table>

Drafting Depth. We further evaluate how far GrowMTP can extend the drafting depth, and which depth best serves RL acceleration. We train on MiMo-7B-SFT at $K { = } 3 , 5 ,$ and 7 for 200 RL steps in each domain, against the shipped head frozen at K=1. As shown in Table 4 and Appendix C.3, online training converts nominal depth into realized acceptance: at $K { = } 5 ,$ τ climbs to 4.04 and 3.72, beyond the structural ceiling of two for single-step drafting, rollouts accelerate by 1.93× and 1.72×, and end-to-end steps by 1.41× and 1.35×. Greater depth, however, does not translate monotonically into end-to-end gains: at $K { = } 7 ,$ τ rises further to 4.38 and 4.00, yet step speedups fall to 1.30× on math and 1.31× on code, as both drafting and verification costs per cycle grow with depth while the marginal acceptance gain diminishes (Chen et al., 2024). Held-out scores are comparable across depths. The drafting depth should thus follow end-to-end acceleration rather than acceptance alone, with K=5 best in this setting: GrowMTP also proves effective for a single-step head.

Table 4: Training evaluation across drafting depths on MiMo-7B-SFT (200 RL steps). τ rises monotonically with depth, while end-to-end step speedups peak at K=5, the highlighted depth.
<table><tr><td></td><td></td><td colspan="5">Math Reasoning RL</td><td colspan="5">Code Reasoning RL</td></tr><tr><td>Method</td><td>K</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td></tr><tr><td>Frozen</td><td>1</td><td>318.66</td><td>1.00×</td><td>523.02</td><td>1.00×</td><td>1.65</td><td>441.81</td><td>1.00×</td><td>707.42</td><td>1.00×</td><td>1.59</td></tr><tr><td>GrowMTP</td><td>3</td><td>176.26</td><td>1.81×</td><td>378.88</td><td>1.38×</td><td>3.25</td><td>264.69</td><td>1.67×</td><td>528.64</td><td>1.34×</td><td>3.11</td></tr><tr><td>GrowMTP 5</td><td></td><td>164.84</td><td>1.93×</td><td>371.02</td><td>1.41×</td><td>4.04</td><td>256.86</td><td>1.72×</td><td>525.31</td><td>1.35×</td><td>3.72</td></tr><tr><td>GrowMTP</td><td>7</td><td>171.24</td><td>1.86×</td><td>402.20</td><td>1.30×</td><td>4.38</td><td>252.03</td><td>1.75×</td><td>542.02</td><td>1.31×</td><td>4.00</td></tr></table>

## 4.4 RQ3: HOW DOES EACH COMPONENT CONTRIBUTE TO MULTI-STEP ADAPTATION?

Following the MiMo depth study, we isolate component effects using the same pretrained single-step head initialization. RQ1 separately establishes the complete GrowMTP’s from-scratch capability.

Draft-Path Reconstruction. On MiMo-7B-SFT at K=5, we replace the reconstructed draft path with the committed response, teacher forcing under an unchanged objective (Zhang et al., 2025). Teacher forcing supervises every position at every depth rather than one chain per cycle, and the two agree only at the first draft position. As shown in Table 5 and Appendix C.4, the added positions raise cost without raising aggregate acceptance length: acceptance ties within 0.03 while the step speedup falls from 1.41× to 1.28×. Resolved by depth, however, the discrepancy does not vanish: acceptance is indistinguishable at shallow positions, while the deepest retain a gap that narrows without closing. Aggregate τ absorbs that residue, since the deepest positions contribute the smallest chain terms. Thus, despite similar aggregate τ at K=5, teacher forcing retains a deeper-position gap and yields lower end-to-end efficiency than draft-path reconstruction. Appendix G analyzes this gap in detail.

Table 5: Ablation of Draft-Path Reconstruction on MiMo-7B-SFT (200 RL steps, K=5). Acceptance ties within 0.03, and reconstruction lifts the end-to-end step speedup from 1.28× to 1.41×.
<table><tr><td></td><td colspan="5">Math Reasoning RL</td><td colspan="6">Math Reasoning Eval</td></tr><tr><td></td><td colspan="5">DAPO-Math-17K</td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td></tr><tr><td>Method</td><td>Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td></tr><tr><td>Teacher forcing</td><td>169.05</td><td>1.89×</td><td>408.09</td><td>1.28×</td><td>4.02</td><td>91.56</td><td>3.93</td><td>54.37</td><td>3.82</td><td>43.96</td><td>3.79</td></tr><tr><td>GrowMTP</td><td>164.84</td><td>1.93×</td><td>371.02</td><td>1.41×</td><td>4.04</td><td>91.25</td><td>3.95</td><td>56.87</td><td>3.84</td><td>46.25</td><td>3.82</td></tr></table>

DCA Loss. Retaining the setup of the depth study, we train in mathematical reasoning and replace only the objective: CE, KL (Zhou et al., 2024), and TV (Li et al., 2026a). As shown in Figure 4(a,b) and Appendix C.4, at K=1 the four objectives differ by less than 0.03 in τ, as the chain reduces to a single acceptance rate leaving no depth structure. At K=5, DCA leads consistently: τ reaches 4.04 against 3.86 for CE, 3.85 for KL, and 3.94 for TV, and the end-to-end step speedup 1.41× against 1.36× to 1.37×, head-update cost included. Relative to CE and $\mathbf { K L } ,$ the advantage widens with depth: DCA is built directly on the acceptance chain, whereas per-position imitation ignores the chain dependence. TV and DCA use the same acceptance-chain surrogate; the logarithm changes their relative cycle weights in the batch average, thus favoring cycles with smaller chain values.

![](images/8a06da88f746c6d82f89457d0d47cd2b7c3e72ef147c8589772f58a3aba241ab.jpg)

![](images/2960a6cf3f1bb5f5535be2cde64279b8586fba80713afb1df8da78a332b47c0e.jpg)

![](images/e761b080bec7c5a734bae3c91c8ab9eaf8a258442a517bd872f0410061adc4b6.jpg)

![](images/6217a5522d6e1912849e9ed3111be49e10044c5c7de02744c749ed3a92b8ba90.jpg)  
(a) DCA ablation (K=1). (b) DCA ablation (K=5). (c) VGM ablation (KL). (d) VGM ablation (DCA).  
Figure 4: DCA and VGM support GrowMTP from complementary sides (MiMo-7B-SFT). (a,b) The τ gap of DCA over CE, KL, and TV opens with depth. (c,d) Removing VGM degrades τ .

Verify-Gated Masking. We remove VGM under both the KL and DCA objectives. As shown in Figure 4(c,d) and Appendix C.4, both degrade: the step speedup of KL falls from 1.37× to 1.28×, showing an end-to-end efficiency benefit from masking post-rejection positions. DCA degrades less, from 1.41× to 1.35×. Its soft weighting can retain post-rejection gradients; VGM enforces the rejection boundary and provides an additional efficiency gain. Together, the components improve multi-step adaptation: reconstruction lowers training cost at comparable acceptance length, DCA improves acceptance, and VGM further improves acceptance and end-to-end efficiency.

## 5 RELATED WORK

## 5.1 SPECULATIVE DECODING

The draft model has evolved from independent small models (Miao et al., 2024) to lightweight draft heads that reuse the backbone’s hidden states (Cai et al., 2024; Li et al., 2024a), now common components of frontier models (Liu et al., 2024; Qwen, 2026; Xiaomi et al., 2025). Draft heads can be trained jointly with the backbone or through offline distillation from target-generated data (Gloeckle et al., 2024; Li et al., 2026b; Hu et al., 2026), or adapted online to the served query distribution (Liu et al., 2023). HASS (Zhang et al., 2025) and EAGLE-3 (Li et al., 2026b) seek to align draft-head training states with those encountered during inference. After SFT warm-up, Draft-OPD (Lei et al., 2026) replays draft paths in a separate online distillation stage and treats accepted and rejected positions differently. GrowMTP shares this alignment motivation, reconstructing training states from draft paths and matching verification distributions recorded during RL. For acceptance-oriented training, Bebop (Li et al., 2026a) introduces a multi-step TV-chain objective. DCA applies a logarithmic transformation to this objective to adjust the relative gradient weights of cycles and combines it with VGM, which retains supervision up to and including the first rejected position.

## 5.2 ACCELERATING REINFORCEMENT LEARNING FOR LLMS

Recent work accelerates the rollout stage that dominates RL wall-clock. One line integrates highthroughput inference engines into the RL training loop to reduce rollout cost (Hu et al., 2025; Shen et al., 2024). For example, VeRL (Sheng et al., 2025) adopts a hybrid controller architecture that decouples RL control flow from distributed computation. Another line optimizes rollout scheduling to reduce tail waiting (Noukhovitch et al., 2025). For example, AReaL (Fu et al., 2026) uses asynchronous execution to reduce synchronization stalls. Speculative decoding increases the number of tokens committed per verification step: RhymeRL (He et al., 2025) constructs drafts from historical rollouts, while ReSpec (Chen et al., 2026) updates draft models with RL supervision and includes an offline warm-up stage. GrowMTP supports random initialization, integrating drafting, verification, and head updates into the same RL run without a separate offline training stage. We measure end-to-end speedups over the full run, including training cost, and evaluate it on pretrained heads.

## 6 CONCLUSION

In this paper, we propose GrowMTP, a from-scratch online draft-head training method for accelerating the rollout stage of RL training. We observe that RL training itself provides both conditions required for online draft-head training: its rollout distribution is far narrower than that of pretraining, and its verification step continuously produces supervision signals aligned with this distribution. Building on these observations, GrowMTP trains the draft head online within the RL loop using this signal directly, enabling a randomly initialized head to grow entirely within RL without any offline pretraining or warm-up. Comprehensive evaluations on Qwen3-4B (no draft head), MiMo-7B-SFT (weak head), and Qwen3.5-4B-Base (strong head) show that GrowMTP achieves rollout speedups of 2.13×, 1.93×, and 1.36×, and end-to-end speedups of 1.60×, 1.41×, and 1.20×, respectively, with comparable held-out benchmark scores in the reported runs. These results demonstrate that speculative decoding for RL acceleration no longer requires a separate offline training stage. GrowMTP therefore can be integrated into existing RL training frameworks as a modular acceleration component, particularly offering a from-scratch acceleration path for models without pretrained MTP heads.

## REFERENCES

Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. Hydra: Sequentially-dependent draft heads for medusa decoding. arXiv preprint arXiv:2402.05109, 2024.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

Ruisheng Cao, Mouxiang Chen, Jiawei Chen, Zeyu Cui, Yunlong Feng, Binyuan Hui, Yuheng Jing, Kaixin Li, Mingze Li, Junyang Lin, et al. Qwen3-coder-next technical report. arXiv preprint arXiv:2603.00729, 2026.

Aakshita Chandiramani, Aaron Blakeman, Abdullahi Olaoye, Abhibha Gupta, Abhilash Somasamudramath, Abhinav Khattar, Adeola Adesoba, Adi Renduchintala, Adil Asif, Aditya Agrawal, et al. Nemotron 3 super: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. arXiv preprint arXiv:2604.12374, 2026.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

Qiaoling Chen, Zijun Liu, Peng Sun, Shenggui Li, Guoteng Wang, Ziming Liu, Yonggang Wen, Siyuan Feng, and Tianwei Zhang. Respec: Towards optimizing speculative decoding in reinforcement learning systems. Proceedings ofMachine Learning and Systems, 8:367–379, 2026.

Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable, robust, and hardware-aware speculative decoding. arXiv preprint arXiv:2402.12374, 2024.

Wei Fu, Jiaxuan Gao, Xujie Shen, Chen Zhu, Zhiyu Mei, Chuyi He, Shusheng Xu, Guo Wei, Jun Mei, Jiashu Wang, et al. Areal: A large-scale asynchronous reinforcement learning system for language reasoning. Advances in Neural Information Processing Systems, 38:36256–36282, 2026.

Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Roziere, David Lopez-Paz, and Gabriel Synnaeve.\` Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Jingkai He, Tianjian Li, Erhu Feng, Dong Du, Qian Liu, Tao Liu, Yubin Xia, and Haibo Chen. History rhymes: Accelerating llm reinforcement learning with rhymerl. arXiv preprint arXiv:2508.18588, 2025.

Jian Hu, Xibin Wu, Wei Shen, Jason Klein Liu, Weixun Wang, Songlin Jiang, Haoran Wang, Hao Chen, Bin Chen, Wenkai Fang, et al. Openrlhf: A ray-based easy-to-use, scalable and highperformance rlhf framework. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 656–666, 2025.

Shijing Hu, Jingyang Li, Zhihui Lu, and Pan Zhou. Bridging draft policy misalignment: Group tree optimization for speculative decoding. In International Conference on Learning Representations, volume 2026, pp. 112507–112524, 2026.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

Naman Jain, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, volume 2025, pp. 58791–58831, 2025.

Devvrit Khatri, Lovish Madaan, Rishabh Tiwari, Rachit Bansal, Venkata Sai Surya Subramanyam Duvvuri, Manzil Zaheer, Inderjit Dhillon, David Brandfonbrener, and Rishabh Agarwal. The art of scaling reinforcement learning compute for llms. In International Conference on Learning Representations, volume 2026, pp. 72438–72467, 2026.

Haodi Lei, Yafu Li, Haoran Zhang, Shunkai Zhang, Qianjia Cheng, Xiaoye Qu, Ganqu Cui, Bowen Zhou, Ning Ding, Yun Luo, et al. Draft-opd: On-policy distillation for speculative draft models. arXiv preprint arXiv:2605.29343, 2026.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International conference on machine learning, pp. 19274–19286. PMLR, 2023.

Rongao Li, Jie Fu, Bo-Wen Zhang, Tao Huang, Zhihong Sun, Chen Lyu, Guang Liu, Zhi Jin, and Ge Li. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852, 2023.

Yucheng Li, Huiqiang Jiang, Yang Xu, Jianxin Yang, Yi Zhang, Yizhong Cao, Yuhao Shen, Fan Zhou, Rui Men, Jianwei Zhang, et al. Breaking entropy bounds: Accelerating rl training via mtp with rejection sampling. arXiv preprint arXiv:2606.12370, 2026a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024a.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 7421–7432, 2024b.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. Advances in Neural Information Processing Systems, 38:136737–136756, 2026b.

Zongqian Li, Tengchao Lv, Shaohan Huang, Yixuan Su, Qinzheng Sun, Qiufeng Yin, Ying Xin, Scarlett Li, Lei Cui, Nigel Collier, et al. Scaling data difficulty: Improving coding models via reinforcement learning on fresh and challenging problems. arXiv preprint arXiv:2603.07779, 2026c.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Online speculative decoding. arXiv preprint arXiv:2310.07177, 2023.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, Volume 3, pp. 932–949, 2024.

Michael Noukhovitch, Shengyi Huang, Sophie Xhonneux, Arian Hosseini, Rishabh Agarwal, and Aaron Courville. Asynchronous rlhf: Faster and more efficient off-policy rl for language models. In International Conference on Learning Representations, volume 2025, pp. 4003–4029, 2025.

Qwen. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/ blog?id=qwen3.5.

Gerald Shen, Zhilin Wang, Olivier Delalleau, Jiaqi Zeng, Yi Dong, Daniel Egert, Shengyang Sun, Jimmy Zhang, Sahil Jain, Ali Taghibakhshi, et al. Nemo-aligner: Scalable toolkit for efficient model alignment. arXiv preprint arXiv:2405.01481, 2024.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient rlhf framework. In Proceedings ofthe Twentieth European Conference on Computer Systems, pp. 1279–1297, 2025.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, C Du, C Liao, et al. Kimi k1. 5: Scaling reinforcement learning with llms, 2025. URL https://arxiv. org/abs/2501.12599, 118, 2025.

Zili Wang, Jiajun Chai, Lin Chen, Xiaohan Wang, Shiming Xiang, and Guojun Yin. Joint training of multi-token prediction in reinforcement learning via optimal coefficient calibration. arXiv preprint arXiv:2605.28184, 2026.

Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. Findings of the Association for Computational Linguistics: ACL 2024, pp. 7655–7671, 2024.

LLM Xiaomi, Bingquan Xia, Bowen Shen, Dawei Zhu, Di Zhang, Gang Wang, Hailin Zhang, Huaqiu Liu, Jiebao Xiao, Jinhao Dong, et al. Mimo: Unlocking the reasoning potential of language model–from pretraining to posttraining. arXiv preprint arXiv:2505.07608, 2025.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient milliontoken context intelligence. arXiv preprint arXiv:2606.19348, 2026.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. Advances in Neural Information Processing Systems, 38:113222–113244, 2026.

Aohan Zeng, Xin Lv, Qinkai Zheng, Zhenyu Hou, Bin Chen, Chengxing Xie, Cunxiang Wang, Da Yin, Hao Zeng, Jiajie Zhang, et al. Glm-4.5: Agentic, reasoning, and coding (arc) foundation models. arXiv preprint arXiv:2508.06471, 2025.

Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations for speculative sampling. In International Conference on Learning Representations, volume 2025, pp. 35367–35388, 2025.

Yifan Zhang and Team Math-AI. American mathematics competitions (amc) 2023, 2023.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2024, 2024.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-Franc¸ois Kagy, and Rishabh Agarwal. Distillspec: Improving speculative decoding via knowledge distillation. In International Conference on Learning Representations, volume 2024, pp. 32011–32050, 2024.

## Appendix

## Contents

A Notation and Definitions 14   
B Experimental Setup and Implementation 15   
B.1 Models and Draft Heads 15   
B.2 RL Training Configuration 15   
B.3 Draft-Head Training Configuration 16   
B.4 Evaluation Protocol 16   
C Additional Results 17   
C.1 Additional Results for Growing from Scratch (RQ1) 17   
C.2 Additional Results for Training Strategy (RQ2) . . 18   
C.3 Additional Results for Drafting Depth (RQ2) . 19   
C.4 Additional Results for Component Ablations (RQ3) 19   
D Discussion 21   
D.1 Which Models Can GrowMTP Accelerate? . 21   
D.2 Does the Acceleration Grow with RL Scale? 21   
D.3 Does GrowMTP Replace Pretrained Draft Heads? 21   
E GrowMTP Implementation Details 22   
E.1 Draft-Path Reconstruction 22   
E.2 Implementing Draft-Path Reconstruction in verl 22   
E.3 Parallelizing GrowMTP Training 23   
F GrowMTP Training Algorithm 24   
G Train–Inference Inconsistency under Teacher Forcing 25   
H One-Step Lag Bound under Rejection Sampling 27   
I Optimization Analysis of Training Objective 29   
J Limitations 30

## A NOTATION AND DEFINITIONS

## Table 6 summarizes the notation used throughout the paper.

<table><tr><td>Symbol</td><td>Definition</td></tr><tr><td colspan="2">RL Training Notation</td></tr><tr><td> $\pi _ { \theta }$ </td><td>The policy, the language model itself.</td></tr><tr><td> $\theta$ </td><td>Backbone (policy) parameters.</td></tr><tr><td> $x$ </td><td>Prompt.</td></tr><tr><td> $y , \ y _ { t }$ </td><td>A sampled response and its token at position  $t .$ </td></tr><tr><td> $G$ </td><td>Number of responses sampled per prompt.</td></tr><tr><td> $t$ </td><td>RL step index in Appendix H; elsewhere the token position of  $y _ { t }$ </td></tr><tr><td> $p ^ { ( t ) } , p ^ { ( t + 1 ) }$ </td><td>Target distributions before and after one policy update.</td></tr><tr><td> $\rho ( y )$   $\bar { \varepsilon } _ { t }$ </td><td>Importance ratio  $p ^ { ( t + 1 ) } ( y \mid s ) / p ^ { ( t ) } ( y \mid s )$  between successive poli- cies.</td></tr><tr><td></td><td>Uniform bound on the per-update target shift. Speculative Decoding Notation</td></tr><tr><td colspan="2">Drafting depth: the number of draft tokens per cycle.</td></tr><tr><td> $K$ </td><td></td></tr><tr><td> $r$ </td><td>Draft-then-verify cycle index.</td></tr><tr><td> $d _ { r , k }$ </td><td>The k-th draft token of cycle  $r .$ </td></tr><tr><td> $d _ { r , j }$ </td><td>The replacement token resampled at the first rejection.</td></tr><tr><td> $j , j _ { r }$ </td><td>First-rejection position of a cycle, with  $j = K { + } 1$  when every draft</td></tr><tr><td></td><td>is accepted.</td></tr><tr><td> $p _ { r , k }$   $q _ { r , k }$ </td><td>Target distribution at draft position k of cycle  $^ { r } \cdot$  Head distribution at draft position k of cycle</td></tr><tr><td> $\alpha _ { k }$ </td><td> $^ r \cdot$  Acceptance probability at position k:  $\alpha _ { k } = 1 - \mathrm { T V } ( p _ { k } , q _ { k } )$ </td></tr><tr><td> ${ \tilde { \alpha } } ( s ) , \alpha ( s )$ </td><td>Acceptance rate against the old and the new target at state s.</td></tr><tr><td> $\tau$ </td><td>Acceptance length: the average number of tokens committed per</td></tr><tr><td> $\mathrm { T V }$ </td><td>cycle,  $\tau \in [ 1 , K { + } 1 ]$ </td></tr><tr><td colspan="2">Total variation distance.</td></tr><tr><td></td><td>Model and State Notation</td></tr><tr><td> $z _ { \theta }$ </td><td>Target-model logit function.</td></tr><tr><td> $h _ { \phi }$ </td><td>The MTP draft head, mapping a conditioning state to a distribution.</td></tr><tr><td> $\phi$ </td><td>Draft-head parameters. Committed context of cycle the prompt and all tokens committed</td></tr><tr><td> $\mathbf { c } _ { r }$ </td><td> $r { : }$  before the cycle.</td></tr><tr><td> ${ \bf d } _ { r }$ </td><td>Draft path of cycle r:  $[ d _ { r , 1 } , \ldots , d _ { r , K } ] .$ </td></tr><tr><td colspan="2">Reconstructed state  $[ \mathbf { c } _ { r } , d _ { r , 1 } , \hdots , d _ { r , k - 1 } ] .$  Training Objective Notation</td></tr><tr><td></td><td></td></tr><tr><td> ${ \mathcal { L } } _ { \mathrm { T V } }$ </td><td>TV-chain loss:  $\begin{array} { r l } { - \sum _ { l = 1 } ^ { K } \prod _ { i = 1 } ^ { l } \alpha _ { i } } & { { } } \end{array}$  , a differentiable surrogate for  $1 - \tau .$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { D C A } }$ </td><td>DCA loss:  $- \log ( - \mathcal { L } _ { \mathrm { T V } } )$ </td></tr><tr><td> $\mathcal { L } _ { \mathrm { D C A } } ^ { \mathrm { V G M } }$ </td><td>Verify-gated DCA loss: the chain sum truncated at  $\operatorname* { m i n } ( j , K )$ </td></tr><tr><td> $M$ </td><td>The chain quantity  $- \mathcal { L } _ { \mathrm { T V } }$  in Appendix I.</td></tr><tr><td> $D _ { k }$ </td><td>Tail term of the gradient weight:  $\begin{array} { r } { \sum _ { l = k } ^ { K } \prod _ { i = k + 1 } ^ { l } \alpha _ { i } \in [ 1 , K - k + 1 ] } \end{array}$ </td></tr><tr><td> $g _ { r }$ </td><td>Gradient of  $\mathcal { L } _ { \mathrm { D C A } }$  on cycle r in Appendix I.</td></tr></table>

Table 6: Notation used throughout the paper.

## B EXPERIMENTAL SETUP AND IMPLEMENTATION

## B.1 MODELS AND DRAFT HEADS

Baseline Models. We evaluate GrowMTP on three models: Qwen3-4B, Qwen3.5-4B-Base, and MiMo-7B-SFT. The three differ in the draft head they ship: Qwen3-4B has never shipped one, while Qwen3.5-4B-Base and MiMo-7B-SFT each ship a native head. The drafting depth of such a head is not fixed by its architecture, so we adopt the depth recommended by each model’s release: K=3 for Qwen3.5-4B-Base and K=1 for MiMo-7B-SFT. Apart from this shipped state, the draft heads of the three models take the same single-layer recurrent MTP form throughout this paper, detailed below.

MTP Architecture. All three draft heads take the EAGLE-style single-layer recurrent MTP form. The module contains a single decoder layer. It drafts one position at a time, consuming the hidden state and the token embedding of the position before it: the two are normalized by separate RMSNorms, concatenated, and projected back to the hidden dimension by a linear layer without bias before entering the decoder layer. It maintains a KV cache of its own, which grows by one entry with every position it drafts. The head contains neither an embedding nor an output projection and takes both from the backbone. The heads of the three models therefore run through the same drafting and training implementation, and the family difference is confined to the type of the decoder layer, one each from Qwen3, Qwen3.5, and Qwen2. The native head of MiMo concatenates the two inputs in the opposite order, and weight loading swaps the two halves of that linear layer accordingly.

From-Scratch Initialization. The draft head of Qwen3-4B is built in this same form, identical to the two native heads down to the parameter names. Randomization covers this newly built module alone: every linear layer within it is drawn from N(0, 0.02), the value of the backbone’s own initializer range, and every RMSNorm weight is set to one. The head is materialized once into a checkpoint under seed 0, and the training side and the SGLang drafting side load the same weights from it, so no divergence between two independent initializations arises. The embedding and the output head are inherited from the backbone rather than randomized, so training from scratch refers to the drafting capability and not to every parameter that participates in drafting.

## B.2 RL TRAINING CONFIGURATION

Policy Optimization. Policy optimization uses GRPO with one modification: the negative-advantage branch carries a weight of 0.5, and every other term follows the standard form, dual clipping and token-level averaging among them. The objective retains a KL regularizer at coefficient 0.01, whose reference policy is the initial checkpoint and stays fixed throughout training. The policy is updated by AdamW in verl, at a constant learning rate of $\mathrm { \dot { 1 } \times 1 0 ^ { - 6 } }$ and with a clipping range of 0.2. All three models, both domains, and every ablation arm share this configuration, so the differences between groups arise only from the variable isolated in each comparison with policy optimization held fixed.

Training Data. Mathematical reasoning uses DAPO-Math-17K and code reasoning uses TACO-Verified. The reward comes from matching the reference answer in mathematical reasoning and from matching the standard output of the generated program in code reasoning. Both datasets are used as released, without difficulty filtering or subsampling, with head supervision drawn from RL rollouts.

Rollout and Batching. All runs execute on a single node of 8×H800 GPUs, with the model sharded across the eight devices. Each step samples 64 prompts and generates 8 responses per prompt, and the policy update uses a mini-batch of 64. Prompts are truncated to 1024 tokens, and responses are capped at 8192 tokens on Qwen3-4B and MiMo-7B-SFT and at 16384 tokens on Qwen3.5-4B-Base. Rollouts are served by SGLang at temperature 1.0 and top-p 1.0, with draft tokens accepted by rejection sampling. Qwen3-4B runs in thinking mode, and the other two models are not thinking models. All runs fix the data order with seed 1, keeping the data order consistent across methods.

## B.3 DRAFT-HEAD TRAINING CONFIGURATION

Draft Head Training. In every GrowMTP run the head is trained under the DCA objective with detached gradients and with VGM enabled throughout. The ablation arms each change one of these: the objective becomes CE, KL, or the TV-chain objective of Appendix I, the gradients become joint or frozen, or VGM is removed. The drafting depth is K=5 on Qwen3-4B, K=3 on Qwen3.5-4B-Base, and K=3, 5, and 7 on MiMo-7B-SFT, whose frozen baseline runs at K=1. The head is updated once per RL step, from the verification signals recorded in that step’s own rollouts. Training includes no freeze phase, and the head is updated from the first step to the last using online RL supervision.

Optimizer and Learning Rate. A shared AdamW optimizer updates the head and backbone in separate parameter groups with independent learning rates. Head learning rates are $3 \times 1 0 ^ { - 4 }$ for random initialization and $1 \times 1 0 ^ { - 4 }$ for the two pretrained heads. Both head rates warm up over 10 steps, then follow a cosine decay to one tenth of their peak; the policy learning rate stays constant.

## B.4 EVALUATION PROTOCOL

Training Metrics. On the training side we report two times, the rollout time and the end-to-end step time, the latter including the cost of head updates. Both are per-step means over the whole of training. The denominator of a speedup is the run with the head frozen under the same script, the same data, and the same hyperparameters, and on Qwen3-4B that denominator carries no head and reduces to autoregressive decoding. The training-side τ is instead averaged over the final 10 steps: the times are costs incurred throughout, whereas τ is the capability of the head once it has converged in the run.

Inference Evaluation Benchmarks. Mathematical reasoning is evaluated on AMC23, AIME24, and AIME25, with 40, 30, and 30 questions respectively. Code reasoning is evaluated on LiveCodeBench v6, whose 1055 questions include the AtCoder and LeetCode platform slices at 602 and 444 questions. Judging uses the same string-matching verifier as training in mathematical reasoning and the official LiveCodeBench framework in code reasoning, with a timeout of 6 seconds for each separate test case.

Evaluation Configuration. SGLang generates responses at temperature 1.0 and top-p 0.7, capped at 16384 tokens. Speculative decoding uses rejection sampling, as in training. Evaluation uses each run’s trained depth, each frozen baseline’s shipped depth, and no draft head for the AR baseline.

Evaluation Metrics. Policy quality is reported as Mean@16 in mathematical reasoning, the mean accuracy over 16 samples. In code reasoning it is reported as Pass@4, the fraction of questions on which at least one of 4 attempts passes every test. Drafting capability is the held-out τ, one plus the ratio of accepted tokens to verifications, measured on the same generations used to assess quality.

## C ADDITIONAL RESULTS

## C.1 ADDITIONAL RESULTS FOR GROWING FROM SCRATCH (RQ1)

Math Reasoning RL Training. Figure 5 shows the mathematical reasoning run: the acceptance length τ and the end-to-end step time over the 500 RL steps. From random initialization at the autoregressive floor, τ rises steepest over the first hundred steps and is still climbing at step 500. Step time opens above the no-head baseline, the early overhead visible on the curve, and crosses below it at step 30: every later step runs faster, and the widening gap tracks the growing τ throughout the run.

![](images/b43740e135e6d2a4bcb9ca0a9f80d77df806e44eff837e82b8541213ae8e61c1.jpg)  
(a) Acceptance length τ.

![](images/535653a63887fa36fa2160c60530c056c7ff7f7853606f33e3404329f9a912bd.jpg)  
(b) End-to-end step time.  
Figure 5: Math reasoning (Qwen3-4B): a draft head grows from scratch within RL and accelerates the run itself. τ rises to 2.91, and step time drops below the no-head baseline within 30 steps.

Code Reasoning RL Training. Figure 6 shows the training trajectories of the code reasoning run: τ and the end-to-end step time over the 500 RL steps. The trajectories mirror the mathematical reasoning run of Figure 5: τ leaves the autoregressive floor immediately and climbs to 2.64 over the 500 steps, and step time falls below the no-head baseline within 22 steps. The from-scratch growth pattern is thus not specific to a single training domain: both domains support online head training.

![](images/d571a997cd133775738bba7d2c9e4ddfd5bdc4cb52a07f4f4745c3636bb5b8bd.jpg)  
(a) Acceptance length τ.

![](images/ea57b3f05bc1e5ce2e53b10bcf6e0dc59acc3cbc0349a592ff554e2e8bfc83d5.jpg)  
(b) End-to-end step time.  
Figure 6: Code reasoning (Qwen3-4B): the from-scratch growth pattern of Figure 5 holds. τ rises to 2.64 over 500 RL steps, and step time drops below the no-head baseline within 22 steps.

Inference Evaluation. Table 7 reports the held-out evaluation of the two runs, pairing policy quality with τ on each benchmark. Policy quality stays on par with the autoregressive baseline on all six benchmarks, with Mean@16 and Pass@4 differing by at most 2.29 points. The trained heads reach τ of 2.30–2.34 on the mathematical benchmarks and 2.11–2.27 on the code benchmarks: the drafting capability learned on training rollouts carries over to held-out evaluation prompts of the same domain.

Table 7: Inference evaluation on Qwen3-4B (500 RL steps, K=5). The trained draft head reaches τ above 2.1 on every benchmark in both domains, while policy quality (Mean@16 for math, Pass@4 for code) remains on par with the K=0 baseline, autoregressive decoding without any draft head.
<table><tr><td></td><td></td><td colspan="6">Math Reasoning Eval</td><td colspan="6">Code Reasoning Eval</td></tr><tr><td></td><td></td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">AtCoder</td><td colspan="2">LeetCode</td><td colspan="2">LiveCodeBench</td></tr><tr><td>Method</td><td></td><td>K Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td></tr><tr><td>AR RL</td><td>0</td><td>62.81</td><td>1.00</td><td>20.42</td><td>1.00</td><td>18.75</td><td>1.00</td><td>42.19</td><td>1.00</td><td>42.79</td><td>1.00</td><td>42.37</td><td>1.00</td></tr><tr><td>GrowMTP 5</td><td></td><td>64.69</td><td>2.33</td><td>20.42</td><td>2.34</td><td>21.04</td><td>2.30</td><td>43.36</td><td>2.27</td><td>43.24</td><td>2.11</td><td>43.51</td><td>2.22</td></tr></table>

## C.2 ADDITIONAL RESULTS FOR TRAINING STRATEGY (RQ2)

Math Reasoning RL. Figure 7 shows the four training strategies on mathematical reasoning: τ, the end-to-end step time, and held-out quality (Pass@1 and Pass@8) over the 200 RL steps. Both detached arms raise τ above the frozen head while their quality curves stay level throughout. Joint CE traces the opposite pattern: its τ climbs to the highest of the four while its Pass@1 and Pass@8 collapse to zero, the covariation behind reading a rising τ as no evidence of a healthy RL training run.

![](images/baf88e212bea923dd88426d57db7dae0f44293e478ddad14836e5e7ec55435db.jpg)  
(a) Acceptance length.

![](images/cbd3cb1b4df152df7223a151e39ab4bce6e9e0878872ba69721a8e842736e1fc.jpg)  
(b) End-to-end step time.

![](images/3ef03a8561481a93503978aea3f503c57fe1c0a330fe5d2dd413eb2550d252ac.jpg)  
(c) Pass@1.

![](images/91dbbe600401b980fcd5f50e936312004274b16fb57327e2b9f0cd3a5b33a042.jpg)  
(d) Pass@8.  
Figure 7: Math reasoning (Qwen3.5-4B-Base): four training strategies. Both detached arms raise τ above the frozen head and DCA leads, while joint CE reaches the highest τ at zero held-out quality.

Code Reasoning RL. Figure 8 repeats the comparison on code reasoning. Among the qualitypreserving strategies the trajectories mirror the math domain: both detached arms accelerate the run with level quality curves, and DCA leads. Joint CE, however, reaches the lowest step time of the four here after slowing the run on math, at zero quality in both domains: the speed of a collapsed policy flips sign across domains and cannot establish whether RL accelerates with preserved policy quality.

![](images/6bec2a6f2ecdbc6a962f20b889d9e456f845c59c235edf114c37fac391b51717.jpg)  
(a) Acceptance length.

![](images/975eb793b36410bc10dc3d0a9b4ac3a504ffdd310b2e7ba4c1d4fa5f462fda0f.jpg)  
(b) End-to-end step time.

![](images/76d6bffd8db5c9f574a6c9eec3776250b2e0c61104a2a47d6caa2097a95a928b.jpg)  
(c) Pass@1.

![](images/0966531c9e9c27a96a4616fe971a09fc268c398536fef48daee140f87fe0ee11.jpg)  
(d) Pass@8.  
Figure 8: Code reasoning (Qwen3.5-4B-Base): the same comparison. The quality-preserving ranking mirrors the math domain, while joint CE reaches the largest speedups at zero held-out quality.

Inference Evaluation. Table 8 reports the held-out evaluation of the four strategies, pairing policy quality with τ on each benchmark. Among the quality-preserving strategies, DCA reaches τ of 3.22–3.31 against 3.10–3.19 for detached CE, leading on all six benchmarks. Joint CE scores zero on every benchmark in both domains, and its τ of 3.98–3.99 is measured on this collapsed policy: these high drafting numbers are excluded from the comparison among the quality-preserving strategies.

Table 8: Inference evaluation on Qwen3.5-4B-Base (200 RL steps, K=3). Among qualitypreserving strategies, DCA reaches the highest τ on every benchmark. Joint CE collapses to zero.
<table><tr><td></td><td></td><td colspan="6">Math Reasoning Eval</td><td colspan="6">Code Reasoning Eval</td></tr><tr><td></td><td></td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">AtCoder</td><td colspan="2">LeetCode</td><td colspan="2">LiveCodeBench</td></tr><tr><td>Training Mode Loss</td><td></td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td></tr><tr><td>Frozen</td><td></td><td>80.47</td><td>2.61</td><td>55.00</td><td>2.57</td><td>46.67</td><td>2.57</td><td>63.29</td><td>2.52</td><td>72.52</td><td>2.52</td><td>67.39</td><td>2.52</td></tr><tr><td>Joint</td><td>CE</td><td>0.00</td><td>3.99</td><td>0.00</td><td>3.99</td><td>0.00</td><td>3.99</td><td>0.00</td><td>3.98</td><td>0.00</td><td>3.98</td><td>0.00</td><td>3.98</td></tr><tr><td>Detached</td><td>CE</td><td>84.69</td><td>3.19</td><td>57.92</td><td>3.10</td><td>52.50</td><td>3.10</td><td>61.63</td><td>3.15</td><td>71.40</td><td>3.13</td><td>65.97</td><td>3.14</td></tr><tr><td>Detached</td><td>DCA (GrowMTP)</td><td>85.78</td><td>3.31</td><td>59.58</td><td>3.23</td><td>51.46</td><td>3.22</td><td>63.29</td><td>3.22</td><td>70.50</td><td>3.22</td><td>66.54</td><td>3.22</td></tr></table>

## C.3 ADDITIONAL RESULTS FOR DRAFTING DEPTH (RQ2)

RL Training. Figure 9 shows the drafting depth sweep on MiMo-7B-SFT: τ and the end-to-end step time at K=3, 5, and 7 against the frozen K=1 head, in both domains. The τ trajectories stay separated by depth throughout training, a deeper head drafting more accepted tokens at every step, and none of the three curves has saturated by step 200. Deeper heads initially cost more per step, but as acceptance grows, the three step-time curves converge far below the frozen baseline: the depth ranking visible on the τ axis does not carry to the time axis, which determines the preferred depth.

![](images/b5be25eb125db6848fe7bd289009498f578dd600d6577af0cc4e5e1e20e069f5.jpg)  
(a) τ (Math).

![](images/662c18fd4d9e567482594535b0838fcfa6a9e5115943b94450125915175c5307.jpg)  
(b) Step time (Math).

![](images/0de48399f7a3a7ecefb39a75d96c34b36425c32fca27acaa7d60e0a27b7a474c.jpg)  
(c) τ (Code).

![](images/df08a25eb971ea6b2e1024cb7cb6ecf195873b8d85d3df1796b19408a3dea2ea.jpg)  
(d) Step time (Code).  
Figure 9: Drafting depth (MiMo-7B-SFT): K=3, 5, 7 against the frozen K=1 head. Acceptance grows beyond the single-step ceiling, and end-to-end gains peak at K=5 in both training domains.

Inference Evaluation. Table 9 reports the held-out evaluation across the four depths, pairing policy quality with τ on each benchmark. Every trained depth reaches τ of at least 3.02, increasing with K on each of the six columns, while policy quality stays on par with the frozen head. The τ of K=7 is the highest in every column, yet the highlighted depth is K=5: held-out acceptance does not decide the selection, which follows the end-to-end speedups reported in Table 4, including head-update cost.

Table 9: Inference evaluation across drafting depths on MiMo-7B-SFT (200 RL steps). Online training lifts τ above 3.0 at every depth, monotonically in K, while policy quality stays on par with the frozen head. The highlighted row follows end-to-end acceleration, not the highest measured τ.
<table><tr><td></td><td></td><td colspan="6">Math Reasoning Eval</td><td colspan="6">Code Reasoning Eval</td></tr><tr><td></td><td></td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td><td colspan="2">AtCoder</td><td colspan="2">LeetCode</td><td colspan="2">LiveCodeBench</td></tr><tr><td>Method</td><td></td><td>KMean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td><td>Pass@4</td><td>T</td></tr><tr><td>Frozen</td><td>1</td><td>90.47</td><td>1.67</td><td>51.88</td><td>1.64</td><td>43.13</td><td>1.64</td><td>69.93</td><td>1.61</td><td>80.18</td><td>1.61</td><td>74.50</td><td>1.61</td></tr><tr><td>GrowMTP 3</td><td></td><td>90.47</td><td>3.20</td><td>53.33</td><td>3.15</td><td>42.29</td><td>3.14</td><td>70.10</td><td>3.02</td><td>78.38</td><td>3.03</td><td>73.84</td><td>3.03</td></tr><tr><td>GrowMTP</td><td>5</td><td>91.25</td><td>3.95</td><td>56.87</td><td>3.84</td><td>46.25</td><td>3.82</td><td>71.59</td><td>3.59</td><td>78.60</td><td>3.59</td><td>74.69</td><td>3.59</td></tr><tr><td>GrowMTP 7</td><td></td><td>89.69</td><td>4.30</td><td>56.04</td><td>4.11</td><td>43.33</td><td>4.08</td><td>69.93</td><td>3.83</td><td>76.35</td><td>3.85</td><td>72.80</td><td>3.84</td></tr></table>

## C.4 ADDITIONAL RESULTS FOR COMPONENT ABLATIONS (RQ3)

Ablation of Draft-Path Reconstruction. Figure 10 traces both arms: τ, the two timing curves, and rollout acceptance at every draft position. Teacher forcing pays roughly 30 extra seconds per actor update, and its bias sits at depth: α<sub>1</sub> is indistinguishable between the arms, while the α<sub>5</sub> gap narrows without closing. This is the train–inference inconsistency of Appendix G observed in the results.

Ablation of Training Objective. Table 10 reports the four objectives at both depths, training and held-out sides. The K=1 rows stay within 0.03 on every τ column, and at K=5 DCA holds the highest τ in every column, extending its training-side lead to all three held-out evaluation benchmarks.

Ablation of Verify-Gated Masking. Table 11 reports the mask switch under both objectives. Under KL the slowdown is visible already at rollout, while under DCA it falls almost entirely outside rollout, and held-out τ is never higher with the mask removed on any of the three held-out benchmarks.

![](images/0d607598bd4830c1826a8099d1601582445f67f6c4f76f6a8be445023e684f2c.jpg)  
(a) Acceptance length τ .

![](images/a9a419114469737f7541336ba5a5803ef531f46d73c34ef4bd036cc613591de3.jpg)  
(b) End-to-end step time.

![](images/cdc539df625452e677c9fffa2b54d2702ee04ca5ddded39a8b18c5c841b7f765.jpg)  
(c) Actor update time.

![](images/95c77fc749d46d90956aeb9276fb879b58e460a34af41e71a047ef33fed6cb11.jpg)  
(d) Rollout α<sub>1</sub>.

![](images/114a76e1f96f8e3ac58ef41bd8f541164a1eced2df79ba09f5cede63ef279bf8.jpg)  
(e) Rollout α<sub>2</sub>.

![](images/de82ee0bf9c65f5c63058e44add55330cf15c5a091f952e70a2c8910ed3c02f7.jpg)  
(f) Rollout α<sub>3</sub>.

![](images/5187b1beddbec9d972e60423c77af3e74c17ee6a904bcae5541a92c7bf841902.jpg)  
(g) Rollout α<sub>4</sub>.

![](images/4f0e6ed9e9d7b16a758ec618c84a6e2e6a7b8bf5da4549f48f2ea2028c189aa3.jpg)  
(h) Rollout α<sub>5</sub>.  
Figure 10: Teacher forcing costs more per step and leaves a residual gap at the deepest draft positions. MiMo-7B-SFT, 200 RL steps, K=5. (a–c) GrowMTP reaches the same acceptance length at a lower actor update cost. (d–h) Rollout acceptance α<sub>k</sub> separates further with each added depth.

Table 10: Ablation of Objective on MiMo-7B-SFT (200 RL steps, $K \in \{ 1 , 5 \}$ , mathematical reasoning). At K=1, the four objectives yield similar acceptance lengths. At $K { = } 5$ , DCA leads on training acceptance, end-to-end step speedup, and held-out τ on each of the three benchmarks.
<table><tr><td></td><td></td><td colspan="5">Math Reasoning RL</td><td colspan="6">Math Reasoning Eval</td></tr><tr><td></td><td></td><td colspan="5">DAPO-Math-17K</td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td></tr><tr><td>Loss</td><td></td><td>K Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td></tr><tr><td>CE</td><td>1</td><td>274.47</td><td>1.16×</td><td>507.99</td><td>1.03×</td><td>1.84</td><td>90.16</td><td>1.88</td><td>51.67</td><td>1.87</td><td>40.42</td><td>1.87</td></tr><tr><td>KL</td><td>1</td><td>288.58</td><td>1.10×</td><td>532.91</td><td>0.98×</td><td>1.81</td><td>90.31</td><td>1.86</td><td>54.37</td><td>1.85</td><td>44.58</td><td>1.84</td></tr><tr><td>TV</td><td>1</td><td>288.30</td><td>1.11×</td><td>530.29</td><td>0.99×</td><td>1.83</td><td>89.22</td><td>1.88</td><td>56.04</td><td>1.86</td><td>43.13</td><td>1.86</td></tr><tr><td>DCA (GrowMTP)1</td><td></td><td>280.58</td><td>1.14×</td><td>517.00</td><td>1.01×</td><td>1.82</td><td>89.53</td><td>1.87</td><td>53.96</td><td>1.85</td><td>36.88</td><td>1.86</td></tr><tr><td>CE</td><td>5</td><td>175.70</td><td>1.81×</td><td>384.22</td><td>1.36×</td><td>3.86</td><td>89.84</td><td>3.81</td><td>54.17</td><td>3.67</td><td>43.54</td><td>3.66</td></tr><tr><td>KL</td><td>5</td><td>174.98</td><td>1.82×</td><td>382.41</td><td>1.37×</td><td>3.85</td><td>89.69</td><td>3.67</td><td>53.75</td><td>3.53</td><td>42.29</td><td>3.51</td></tr><tr><td>TV</td><td>5</td><td>174.00</td><td>1.83×</td><td>382.69</td><td>1.37×</td><td>3.94</td><td>89.53</td><td>3.80</td><td>55.00</td><td>3.67</td><td>40.83</td><td>3.66</td></tr><tr><td>DCA (GrowMTP)5</td><td></td><td>164.84</td><td>1.93×</td><td>371.02</td><td>1.41×</td><td>4.04</td><td>91.25</td><td>3.95</td><td>56.87</td><td>3.84</td><td>46.25</td><td>3.82</td></tr></table>

Table 11: Ablation of Verify-Gated Masking on MiMo-7B-SFT (200 RL steps, K=5, mathematical reasoning). Removing VGM slows end-to-end training steps under both objectives, and KL degrades more than DCA: the step speedup falls from 1.37× to 1.28× and from 1.41× to 1.35×.
<table><tr><td rowspan="2"></td><td rowspan="2"></td><td colspan="5">Math Reasoning RL</td><td colspan="6">Math Reasoning Eval</td></tr><tr><td colspan="5">DAPO-Math-17K</td><td colspan="2">AMC23</td><td colspan="2">AIME24</td><td colspan="2">AIME25</td></tr><tr><td>Loss</td><td></td><td>VGM Rollout (s)</td><td>Speedup</td><td>Step (s)</td><td>Speedup</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td><td>Mean@16</td><td>T</td></tr><tr><td>KL</td><td>x</td><td>186.40</td><td>1.71×</td><td>408.65</td><td>1.28×</td><td>3.80</td><td>88.59</td><td>3.66</td><td>54.17</td><td>3.49</td><td>41.04</td><td>3.48</td></tr><tr><td>KL</td><td>√</td><td>174.98</td><td>1.82×</td><td>382.41</td><td>1.37×</td><td>3.85</td><td>89.69</td><td>3.67</td><td>53.75</td><td>3.53</td><td>42.29</td><td>3.51</td></tr><tr><td>DCA</td><td>x</td><td>165.97</td><td>1.92×</td><td>387.27</td><td>1.35×</td><td>4.00</td><td>89.22</td><td>3.94</td><td>52.92</td><td>3.81</td><td>41.88</td><td>3.79</td></tr><tr><td>DCA (GrowMTP)</td><td>√</td><td>164.84</td><td>1.93×</td><td>371.02</td><td>1.41×</td><td>4.04</td><td>91.25</td><td>3.95</td><td>56.87</td><td>3.84</td><td>46.25</td><td>3.82</td></tr></table>

## D DISCUSSION

## D.1 WHICH MODELS CAN GrowMTP ACCELERATE?

In this section, we discuss the applicability of GrowMTP. The experiments of Section 4 evaluate Qwen3-4B, MiMo-7B-SFT, and Qwen3.5-4B-Base, achieving end-to-end step speedups of 1.60×, 1.41×, and 1.20× on mathematical reasoning: GrowMTP is effective regardless of initial head capability. For models with pretrained heads, the RL rollout distribution is more concentrated than the broad pretraining data, leaving room for domain-specific adaptation. Beyond head capability, GrowMTP’s acceptance-based training objective is model-agnostic: it depends on draft–target tokendistribution overlap and the first-rejection index under rejection sampling. GrowMTP can therefore serve as a modular acceleration component in RL training frameworks that employ speculative decoding, particularly for base models without MTP heads, which accelerate without extra pretraining.

## D.2 DOES THE ACCELERATION GROW WITH RL SCALE?

The 1.60× of the Qwen3-4B mathematical reasoning run is measured at 500 steps and an 8K rollout length, a small RL scale. We extrapolate from it along rollout length and training steps: the former increases the proportion of time spent on rollouts, and the latter amortizes the head’s growth-phase overhead. For this projection, we assume that rollout time scales linearly with length while the remaining per-step cost stays constant. Beyond 500 steps we take the step time measured at the end of training, 398.7 s, of which 205.7 s is rollout. We further assume that acceptance length grows neither with rollout length nor with training steps. The end-toend speedup then grows monotonically, reaching 1.91× at 16K with 1000 steps and 2.17× at 32K

Table 12: The projected speedup of GrowMTP grows monotonically with RL scale. End-to-end step speedup over the AR baseline, extrapolated from the measured Qwen3-4B run on DAPO-Math 17K (K=5, the 500-step 8K entry).
<table><tr><td rowspan="2">Math Reasoning RL Steps</td><td colspan="3">Rollout Length</td></tr><tr><td>8K</td><td>16K</td><td>32K</td></tr><tr><td>500</td><td>1.60×</td><td>1.79×</td><td>1.93×</td></tr><tr><td>1000</td><td>1.70×</td><td>1.91×</td><td>2.08×</td></tr><tr><td>2000</td><td>1.76×</td><td>1.99×</td><td>2.17×</td></tr></table>

with 2000 steps. These values are illustrative projections under the stated assumptions. Since the learning curve has not saturated at 500 steps, further head improvement could increase the realized speedup. Under these assumptions, the ceiling is 2.55×, the final measured RL rollout speedup.

## D.3 DOES GrowMTP REPLACE PRETRAINED DRAFT HEADS?

We freeze the two heads grown from scratch over 500 RL steps, one in each domain, and measure τ on the held-out benchmarks of both. Within the training domain, τ falls from the training rollouts to the held-out benchmarks, 2.91 to 2.32 for the math-trained head and 2.64 to 2.20 for the code-trained head. Crossing domains is far costlier: the math-trained head falls to 1.45 on code, and the codetrained head to 1.75 on math. The head receives supervision only from the rollout distribution it must accelerate, and τ falls as evaluation departs from it. This specialization is a direct consequence of the narrow-distribution premise: RL acceleration faces a sufficiently concentrated distribution for from-scratch online training to suffice, whereas general serving must cover unforeseen deployment distributions and still relies on pretrained MTP heads with capability learned across broader tasks.

Table 13: Each head specializes to the domain it was trained on. Acceptance length τ on Qwen3-4B (K=5), with Rollout on each head’s own training rollouts and shaded blocks in-domain.
<table><tr><td colspan="3"></td><td colspan="4">Math Reasoning Eval τ</td><td colspan="4">Code Reasoning Eval τ</td></tr><tr><td>Training Domain Rollout AMC23</td><td></td><td></td><td>AIME24 AIME25</td><td></td><td>Avg.</td><td></td><td></td><td>AtCoder LeetCode LiveCodeBench Avg.</td><td></td></tr><tr><td>Math</td><td>2.91</td><td>2.33</td><td>2.34</td><td>2.30</td><td>2.32</td><td>1.50</td><td>1.40</td><td>1.46</td><td>1.45</td></tr><tr><td>Code</td><td>2.64</td><td>1.80</td><td>1.79</td><td>1.68</td><td>1.75</td><td>2.27</td><td>2.11</td><td>2.22</td><td>2.20</td></tr></table>

## E GrowMTP IMPLEMENTATION DETAILS

## E.1 DRAFT-PATH RECONSTRUCTION

The Reconstructed Forward. Draft-path reconstruction removes the train–inference inconsistency of Section 3.1 by aligning the training-side computation with inference: the head is trained at the drafted states. The update phase reconstructs the chain of every recorded cycle: it starts from the backbone hidden recorded at the cycle’s start, runs under the current head parameters inside the training graph, and is supervised by the verification signals recorded in the same cycle. Reconstruction is not redrafting: sampling would not reproduce the rollout’s tokens, and with the recursion consuming a token at every position, one changed token changes every hidden state and every cache entry that follows. The draft tokens therefore turn from outputs into inputs: rollout records the K drafts of every cycle, and they are fed back to the recursion, with nothing sampled. The hidden states and cache entries along the chain need no recording, since the recursion recomputes them from the same inputs, and every reconstructed state conditions on the recorded prefix of its own draft-then-verify cycle.

The Recorded Supervision. The same obstacle appears on the supervision side: $p _ { r , k }$ conditions on the drafts themselves $( \mathrm { E q . } 2 )$ , the trajectory retains only the committed path (Eq. 3), and where the two paths diverge, a target distribution recomputed on the trajectory is no longer the one that verified the head. The remedy is again recording, not recomputation: verification is itself one target forward over the entire draft path, so it computes the distribution at every draft position, and rollout stores these beside the drafts, one per position. The two remedies interlock in the conditioning: the reconstructed $q _ { r , k }$ of Eq. 4 and the recorded $p _ { r , k }$ condition on the same recorded prefix $[ \mathbf { c } _ { r } , d _ { r , 1 } , \hdots , d _ { r , k - 1 } ]$ , and the pairing Section 3.1 defines at the token level becomes K concrete pairs per cycle in the training graph, consumed directly by the loss of Section 3.2. The reconstruction is self-contained: a chain reads only its own cycle’s records and the shared prefix, and no chain uses another chain’s states.

## E.2 IMPLEMENTING DRAFT-PATH RECONSTRUCTION IN VERL

Recording in Rollout. The record of a cycle holds five items: the backbone hidden at the cycle’s start, the K drafts, the verification distributions, the accepted length, and the absolute position of the start. The hidden is captured rather than recomputed: it is the very tensor the engine’s drafting consumed, and the reconstruction therefore starts from a state numerically identical to drafting’s. The verification distributions are stored as the target’s top-64 log-probabilities in bf16: keeping the full distribution at every draft position would scale with the vocabulary, and the truncation fixes the record’s width at 64. The acceptance decisions compress to one integer per cycle: the accepted positions form a prefix, so a single length, the further token counted, restores the decision at every position. The record carries the absolute position of the start, and the reconstruction puts each chain back where it sat in the sequence. Each sequence additionally stores the rollout-time target hidden states and token ids for the prompt and all committed tokens. These recorded inputs are used to prefill the MTP head and rebuild the shared prefix KV cache attended by each reconstructed draft chain.

The stored top-64 log-probabilities retain full-vocabulary normalization rather than being renormalized over the retained set. During head training, the draft distribution is normalized over the full vocabulary and gathered at the target’s top-64 indices; the remaining mass of each distribution forms a residual bin. The resulting overlap upper-bounds the full-vocabulary overlap by at most the target mass outside the top-64, so the error is small when the retained target mass is high. We quantify its effect on three checkpoints from Qwen3-4B Math RL: random initialization, step 100, and step 500.

Against the full-vocabulary reference, mean absolute overlap errors range from $1 . 8 \times 1 0 ^ { - 6 }$ to $5 . { \bar { 7 } } \times 1 0 ^ { - 6 }$ , and the mean per-batch cosine similarity between sparse and full-vocabulary head gradients is at least 0.99996 at each checkpoint. Thus, top-64 sparsification closely preserves overlap values and gradient directions. Rollout verification still uses the full distribution, so the approximation affects only the draft-head training objective without affecting rollout-time verification decisions.

Integration into the RL Loop. The head is a submodule of the FSDP-wrapped actor model, so the shared optimizer of Appendix B.3 is structure rather than wiring. Deployment rides the same structure: the weight synchronization that carries the updated actor into the drafting engine carries the head with it, and after every step the engine drafts with the head just trained. The recording sits on the same loop: signals are captured inside the engine’s verification and delivered to the update phase with the rollout batch. The update phase therefore receives one record per cycle: a cycle commits τ tokens on average, so the records are fewer than the response’s token positions by a factor of τ.

## E.3 PARALLELIZING GrowMTP TRAINING

<table><tr><td></td><td rowspan=1 colspan=1>p1 p</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>p3</td><td rowspan=1 colspan=1>h1</td><td rowspan=1 colspan=1>h2</td><td rowspan=1 colspan=1>h3</td><td rowspan=1 colspan=1>h4</td><td rowspan=1 colspan=1>d11</td><td rowspan=1 colspan=1>d21</td><td rowspan=1 colspan=1>d31</td><td rowspan=1 colspan=1>d41</td><td rowspan=1 colspan=1>d12</td><td rowspan=1 colspan=1>d22</td><td rowspan=1 colspan=1>d32</td><td rowspan=1 colspan=1>d42</td></tr><tr><td rowspan=1 colspan=1>h1</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>h2</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>h3</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>h4</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d11</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>–</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d21</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d31</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d41</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[d12]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>[d22]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d32</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>d42</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Figure 11: The attention mask of one batched reconstruction forward. Every chain attends over one shared cache: causal over the full committed-token prefix, diagonal over the draft entries of every depth. Each query sees exactly the context of chain-by-chain execution on recorded cycle inputs.

Serial Depth, Parallel Cycles. Executed chain by chain, the reconstruction of Appendix E.1 costs one forward per draft: for a response of three thousand cycles at $K = 5 ,$ , that is fifteen thousand single-token forwards in sequence. Each of these forwards is indexed by the pair $( r , k )$ , with the cycle index r locating the chain within the response and the depth k—the draft position within its cycle—locating the forward within the chain. Along k there is no parallelism to extract: Eq. 4 takes the state at depth k − 1 as input at depth k, and the K forwards of one chain can only run in order. Along r no such dependence exists: chains share only recorded inputs, never each other’s computed states, so the forwards of every cycle at a fixed depth admit a single batched execution. Serialization is therefore confined to depth: any schedule spends at least K forwards one after another, and the width available at each of them is the number of cycles in the response, each advanced independently.

Parallelism across Cycles. The reconstruction therefore runs as K batched forwards: the forward at depth k advances all chains by one draft, and the serial count of the response falls from fifteen thousand to K. In one batched forward, visibility no longer follows from packing order: each query must be restricted to the context chain-by-chain execution would give it, a restriction the attention mask of Figure 11 carries. All chains attend over one shared cache, with causal visibility over the full committed-token prefix and diagonal visibility over the draft entries, so no chain sees another’ drafts. For each cycle, the prefilled cache is restricted to its committed prefix, while the cycle’s own draft KV entries are recomputed by feeding back its recorded draft tokens in order. The mask is additive and needs no custom attention kernel: standard scaled dot-product attention consumes it directly. Positions are taken from the record rather than from the packing: every entry enters the rotary encoding at the position it held at rollout, the depth-k draft of cycle r sitting k places past its cycle’s start. Merging changes the execution schedule while preserving each chain’s computation.

Bounding the Memory. What merging does change is retention: a forward’s activations stay resident until its backward runs, so cycles that advance together are held in memory together. At full width the resident state is the whole response—every chain at every depth, held for one backward—and at training lengths it exceeds the accelerator’s memory. Bounding memory is bounding the span of one backward: the reconstruction proceeds in chunks of 1024 consecutive cycles, each chunk running its K forwards and its own backward and releasing its activations before the next begins. Chunking parallelizes nothing: it only sets how many cycles are resident at once. Across chunks the entries of the recorded hiddens accumulate, entering later chunks as constants, while a chunk’s draft entries—entries the diagonal pattern shows to no later chain—are dropped at its end. Each chunk’s loss is weighted by its share of the response’s positions, and the gradients accumulated across chunks equal those of one backward over the response’s loss: cross-chunk entries are constants, the chunks graphs are disjoint, and gradients over disjoint graphs add. A response of three thousand cycles spans three chunks: fifteen batched forwards, where chain-by-chain reconstruction ran fifteen thousand.

## F GrowMTP TRAINING ALGORITHM

Algorithm 1 states one GrowMTP training step: against the two-phase step of Section 2.1, the additions are the records $\{ R _ { r } \}$ of line 1 and the head update of lines 4–12, and no other line computes anything new. The record $R _ { r }$ of cycle r collects the five items of Appendix E.2; the update consumes four of them by name—the backbone hidden $b _ { r }$ at the cycle’s start, the drafts d , the verification distributions $p _ { r , 1 : K }$ , and the first rejection $j _ { r } ,$ , read from the accepted length—while the start position is consumed inside the reconstruction alone. The inner loop runs the recursion of Eq. 4 on recorded inputs: the chain starts from the hidden $b _ { r } ,$ , which enters the graph detached, and the recorded start position gives the chain its place in the sequence. SPECDECODE abbreviates the draft-then-verify generation of Section 2.2, inside which the records are captured cycle by cycle. SYNC abbreviates the weight synchronization of Appendix E.2, and all remaining symbols are those of Sections 2 and 3.

Algorithm 1 One GrowMTP training step.   
Require: policy $\pi _ { \theta } .$ , draft head $h _ { \phi } ,$ a batch of prompts X   
Ensure: θ and ϕ updated and synchronized into the drafting engine   
// Rollout (drafting engine)   
1: $Y , \{ R _ { r } \} \gets \tilde { \mathrm { S P E C D E C O D E } } ( X ; \pi _ { \theta } , h _ { \phi } )$ // Sec. 2.2; App. E.2   
2: deliver ${ \bar { Y } } ,$ its rewards, and $\{ R _ { r } \}$ to the update phase   
// Update (actor)   
3: update $\pi _ { \theta }$ by the RL objective on $Y$ // Sec. 2.1   
4: for each recorded cycle r do // batched across cycles, App. E.3   
5: for $k = 1 , \dots , \dot { K }$ do // Eq. 4; serial in k   
6: $q _ { r , k }  h _ { \phi } ( \cdot \mid s _ { r , k } )$   
7: $s _ { r , k + 1 } \gets [ s _ { r , k } , d _ { r , k } ]$   
8: end for   
9: $\alpha _ { r , k } \gets 1 - \mathrm { T V } ( p _ { r , k } , q _ { r , k } ) , k = 1 , \ldots , K$ // Eq. 5   
10: $\mathcal { L } _ { r } \gets \mathcal { L } _ { \mathrm { D C A } } ^ { \mathrm { V G M } } ( \alpha _ { r , 1 : K } ; j _ { r } )$ // Eq. 9: chain sum truncated at mi $\operatorname { \Pi } _ { 1 } ( j _ { r } , K )$   
11: end for   
12: update $h _ { \phi }$ by $\nabla _ { \phi }$ of the mean of $\mathcal { L } _ { r }$ over the batch’s cycles // the gradient reaches only ϕ   
13: SYNC(θ, ϕ) // line 1 of the next step drafts with this ϕ

## G TRAIN–INFERENCE INCONSISTENCY UNDER TEACHER FORCING

This appendix identifies the cycle-boundary and KV-cache mismatch between committed-response teacher forcing and rollout-time drafting, and traces its dependence on draft depth. The cause is one mechanism: inference restarts the head’s recursion, chain and cache alike, only at cycle boundaries, while teacher forcing, consuming the flat response alone, makes every position a start and restarts nowhere. Draft-path reconstruction resolves the inconsistency by recomputing exactly the states inference runs, and no teacher-forcing mask can replace the reconstruction of the drafting states.

Training Under Teacher Forcing. Teacher forcing consumes only the final response $y _ { 1 } , \ldots , y _ { T }$ , and none of the cycle structure recorded during the rollout. The head is a single module applied in a loop, consuming at depth k and position i the depth-(k−1) output and the token embedding at the position before it, and attending the key-value entries held in the cache at earlier positions of the response,

$$
g _ { i } ^ { ( k ) } \ = \ f _ { \phi } \big ( g _ { i - 1 } ^ { ( k - 1 ) } , e \big ( y _ { i - 1 } \big ) ; \mathcal { K } _ { i } ^ { ( k ) } \big ) , \qquad k = 1 , \ldots , K ,\tag{10}
$$

where e maps a token to its embedding, $g ^ { ( 0 ) }$ are the hidden states the target model produced during the rollout forward, and $g _ { i } ^ { ( k ) }$ plays the role the sequences $s _ { r , k }$ play in Eq. 4. The cache grows by one entry per position, written by the head’s key-value projection $c _ { \phi }$ from the same pair of head inputs,

$$
{ \mathcal K } _ { i } ^ { ( k ) } = { \mathcal K } _ { i - 1 } ^ { ( k ) } \cup \big \{ c _ { \phi } \big ( g _ { i - 1 } ^ { ( k - 1 ) } , e \big ( y _ { i - 1 } \big ) \big ) \big \} ,\tag{11}
$$

starting from the entries of the prompt in ${ \boldsymbol { \kappa } } _ { 0 } ^ { ( k ) }$ : the cache attended at depth k is thus made entirely of entries computed from depth-(k−1) features. The K causal forwards yield at most $T \cdot K$ training pairs $( g _ { i } ^ { ( k ) } , y _ { i } )$ , whose unrolled recursion traces each trained state back to the target’s hidden features,

$$
g _ { i - k } ^ { ( 0 ) } \stackrel { f _ { \phi } } { \longrightarrow } g _ { i - k + 1 } ^ { ( 1 ) } \stackrel { f _ { \phi } } { \longrightarrow } \cdots \stackrel { f _ { \phi } } { \longrightarrow } g _ { i - 1 } ^ { ( k - 1 ) } \stackrel { f _ { \phi } } { \longrightarrow } g _ { i } ^ { ( k ) } .\tag{12}
$$

The chain starts at position $i - k ,$ , from the target’s own feature there: teacher forcing makes every position the start of a chain, and sets the start by depth alone rather than by recorded cycle boundaries.

Inference Under Speculative Decoding. At inference the head runs the same recursion and restarts it at one kind of position only: the end of the committed prefix, once per cycle. Cycle r begins drafting at context length $m _ { r }$ , and every position behind this boundary carries a committed token together with its target feature $g ^ { ( 0 ) }$ . To draft position $m _ { r } { + } k$ , the head applies $f _ { \phi }$ k times, and the first application consumes the target feature at the boundary before it uses its own recursive features,

$$
\begin{array} { r l } & { \bar { g } _ { r , 1 } = f _ { \phi } \big ( g _ { m _ { r } } ^ { ( 0 ) } , e \big ( y _ { m _ { r } } \big ) ; \bar { \mathcal { K } } _ { r , 1 } \big ) , } \\ & { \bar { g } _ { r , k } = f _ { \phi } \big ( \bar { g } _ { r , k - 1 } , e \big ( d _ { r , k - 1 } \big ) ; \bar { \mathcal { K } } _ { r , k } \big ) , \qquad k = 2 , \ldots , K . } \end{array}\tag{13}
$$

The first line shows where the recursion begins again: $\hat { g } _ { r , k }$ , the state at which the head evaluates $h _ { \phi } ( \cdot \mid s _ { r , k } )$ in Eq. 4, is a chain of length k starting at $m _ { r }$ , and the only chain the cycle runs. The tokens this chain consumes are the head’s own drafts, which follow the committed response up to the first rejection, the divergence Eq. 3 records. The restart resets the cache the recursion has written along with the chain’s start: cycle r closes by committing up to the next boundary $m _ { r + 1 }$ , and the cache the last draft state read stands above the cache the next cycle’s first drafting application reads,

$$
\begin{array} { r l } & { \bar { K } _ { r , K } = \left\{ \ldots , \ c _ { \phi } \big ( g _ { m _ { r } } ^ { ( 0 ) } , e ( y _ { m _ { r } } ) \big ) , \ c _ { \phi } \big ( \bar { g } _ { r , 1 } , e ( d _ { r , 1 } ) \big ) , \ \ldots , \ c _ { \phi } \big ( \bar { g } _ { r , K - 1 } , e ( d _ { r , K - 1 } ) \big ) \ \right\} , } \\ & { \bar { K } _ { r + 1 , 1 } = \left\{ \ldots , \ c _ { \phi } \big ( g _ { m _ { r } } ^ { ( 0 ) } , e ( y _ { m _ { r } } ) \big ) , \ c _ { \phi } \big ( g _ { m _ { r } + 1 } ^ { ( 0 ) } , e ( y _ { m _ { r } + 1 } ) \big ) , \ \ldots , \ c _ { \phi } \big ( g _ { m _ { r + 1 } } ^ { ( 0 ) } , e ( y _ { m _ { r + 1 } } ) \big ) \right\} . } \end{array}\tag{14}
$$

Over the accepted range the two rows hold the same tokens, the committed $y _ { m _ { r } + l }$ being the draft $d _ { r , l }$ so the restart replaces the feature alone: $\bar { g } _ { r , l }$ becomes $g _ { m _ { r } + l } ^ { ( 0 ) }$ . This composition then holds at every depth of every cycle: the entries behind the boundary are computed from the target’s features, and the head’s recursive features enter the cache only through the current chain’s writes. The restart does not depend on rejection: a rollout that accepts every draft still closes its cycle after $K { + 1 }$ committed tokens, with $m _ { r + 1 } = m _ { r } { + } K { + } 1$ , and recursion restarts at the boundary to draft the next cycle.

Train–Inference Inconsistency. Take a position i that cycle r both drafted and committed: teacher forcing trains K states there, starts $i - 1$ through i−K, while inference computes exactly one, reached at depth $k = i { - } m _ { 1 }$ by the only chain the cycle runs from its start $m _ { r }$ . Inference never computes the states at the other starts: a mid-cycle position is never a restart, and the chain from an earlier boundary leaves the committed response before reaching i (Eq. 3). The remaining pair shares start and tokens alike, the drafts along the drafted chain being committed tokens here, so the two computation graph can differ only through their caches. A single boundary entry already separates the two caches:

teacher forcing training:

$$
c _ { \phi } \big ( g _ { m _ { r } } ^ { ( k - 1 ) } , e \big ( y _ { m _ { r } } \big ) \big ) \ \in \ K _ { i } ^ { ( k ) } \quad ( \mathrm { E q . \ 1 1 } ) ,
$$

speculative decoding inference:

$$
c _ { \phi } \big ( g _ { m _ { r } } ^ { ( 0 ) } , \ : e ( y _ { m _ { r } } ) \big ) \ \in \ \bar { \mathcal { K } } _ { r , k } \qquad ( \mathrm { E q . } \ 1 4 ) .\tag{15}
$$

The same projection of the same token, the difference contracted to a single superscript, k−1 against 0: the entries agree only at $k = 1$ , where both caches are computed from the target’s features end to end, and the matching states across the whole response occur only at each cycle’s first draft position:

$$
g _ { i } ^ { ( k ) } \ = \ \bar { g } _ { r , i - m _ { r } } \qquad \Longleftrightarrow \qquad k \ = \ i - m _ { r } \ = \ 1 .\tag{16}
$$

That is one state per cycle, out of the $T \cdot K$ states teacher forcing trains, and the rest update the same ϕ, teaching it computations inference never runs. At depth one every trained state still has the drafted form, one application of $f _ { \phi }$ to a target feature under a cache computed from the target’s features: the two schemes differ only in which starts the rollout realizes. At depth k the trained chain spans k positions and crosses a cycle boundary whenever one falls among them, carrying head-computed features through a position at which inference restarts, and a longer chain straddles a boundary more often: the departure from the states inference runs grows along the depth axis. And this is the very axis a deeper draft must scale: the deeper the head drafts, the further from its inference states teacher forcing trains it, so deeper drafting increases the mismatch between training and inference states.

GrowMTP Addresses the Inconsistency. GrowMTP trains the head at the states of $\operatorname { E q . }$ . 13 themselves: the update phase restarts the recursion of Eq. 4 at each recorded boundary (Algorithm 1, App. E.1), feeding the recorded drafts back in order, and its batched execution changes the schedule of this computation and nothing that is computed (App. E.3). Every separation drawn above closes: the chain starts only at the recorded boundary feature $\bar { g _ { m _ { r } } ^ { ( 0 ) } }$ , not at every position, the tokens it consumes are the recorded drafts themselves, and the cache rebuilt behind the boundary holds entries computed from the target’s features, the head’s recursion entering only through the chain’s own writes—precisely the composition of the inference cache. In Eq. 15 the training-side entry now carries superscript 0, the very entry inference attends, and where Eq. 16 left teacher forcing one coincidence per cycle, all K states of the cycle coincide. No mask over the teacher-forcing forward substitutes for this: a mask selects within what teacher forcing supplies, yet the forward, restarting nowhere, computes no drafted state to keep, and the flat response it consumes carries no cycle for the gate of Eq. 9 to truncate—what produces the drafted states is reconstruction itself. The inconsistency this appendix constructs therefore does not arise under GrowMTP: training uses inference states at all depths.

## H ONE-STEP LAG BOUND UNDER REJECTION SAMPLING

The training loop of Section 3.1 retains a single asymmetry between successive RL steps: the head that drafts at step t+1 was trained on verification signals recorded at step t, one policy update earlier. The head is therefore always trained against a target one update older than the one it drafts for, and this one-step lag may degrade the acceptance rate that training establishes. This appendix characterize the sensitivity of the acceptance chain to this lag through the target shift in a single policy update.

Setup. Fix policy update $t  t + 1$ and conditioning state $s ,$ and consider these three distributions:

$$
\begin{array} { r } { p ^ { ( t ) } ( \cdot \mid s ) = \mathrm { s o f t m a x } \big ( z _ { \theta _ { t } } ( s ) \big ) , \quad p ^ { ( t + 1 ) } ( \cdot \mid s ) = \mathrm { s o f t m a x } \big ( z _ { \theta _ { t + 1 } } ( s ) \big ) , \quad q ( \cdot \mid s ) = h _ { \phi _ { t + 1 } } ( \cdot \mid s ) : } \end{array}\tag{17}
$$

the old target, from which the head’s training signals were recorded, the new target, against which the head’s drafts are verified at step t+1, and the head distribution, which generates those drafts. Eq. 5 assigns the head two acceptance rates at the same conditioning state, one for each of the two targets:

$$
\tilde { \alpha } ( s ) = 1 - \mathrm { T V } \bigl ( p ^ { ( t ) } ( \cdot \mid s ) , q ( \cdot \mid s ) \bigr ) , \qquad \alpha ( s ) = 1 - \mathrm { T V } \bigl ( p ^ { ( t + 1 ) } ( \cdot \mid s ) , q ( \cdot \mid s ) \bigr ) ,\tag{18}
$$

the training-time rate ${ \tilde { \alpha } } ,$ taken against the target the head was trained on, and the deployment rate $\alpha ,$ taken against the target that verifies its drafts. The cost of the lag is the difference between the two.

Single-Position Bound. By Eq. 18, the cost is a difference of two distances from $q ,$ and the triangle inequality bounds it from above by using the old target distribution as the intermediate distribution:

$$
\tilde { \alpha } ( s ) - \alpha ( s ) = \mathrm { T V } \big ( p ^ { ( t + 1 ) } ( \cdot \mid s ) , q ( \cdot \mid s ) \big ) - \mathrm { T V } \big ( p ^ { ( t ) } ( \cdot \mid s ) , q ( \cdot \mid s ) \big ) \le \mathrm { T V } \big ( p ^ { ( t + 1 ) } ( \cdot \mid s ) , p ^ { ( t ) } ( \cdot \mid s ) \big ) ,\tag{19}
$$

which holds for every state s and uniformly in the head distribution $q \mathrm { : }$ the deployment rate falls below the training-time rate by at most the single-update displacement of the target. Since TV is a metric, the same argument with the roles of $p ^ { ( t ) }$ and $p ^ { ( t + 1 ) }$ exchanged bounds $\alpha ( s ) - \tilde { \alpha } ( s )$ by the same quantity, so the bound controls both increases and decreases in acceptance at the same fixed state.

Per-Cycle Bound. For the per-cycle acceptance-chain value defined in $\operatorname { E q . 6 } ,$ define the target shift as

$$
\bar { \varepsilon } _ { t } \triangleq \operatorname* { m a x } _ { k \leq K } \mathbb { E } _ { s _ { k } } \Big [ \mathrm { T V } \big ( p ^ { ( t + 1 ) } ( \cdot \mid s _ { k } ) , p ^ { ( t ) } ( \cdot \mid s _ { k } ) \big ) \Big ] = \frac { 1 } { 2 } \operatorname* { m a x } _ { k \leq K } \mathbb { E } _ { s _ { k } } \mathbb { E } _ { y \sim p ^ { ( t ) } } \big \vert \rho ( y ) - 1 \big \vert ,\tag{20}
$$

where $\mathbb { E } _ { s _ { k } }$ averages over the states conditioning draft position $k$ in the cycles of the step-t+1 rollout and $\rho ( y ) = p ^ { ( t + 1 ) } ( y \mid s _ { k } ) / p ^ { ( t ) } ( y \mid s _ { k } )$ is the token importance ratio between successive policies.

$$
\mathrm { T V } ( p , q ) = \frac { 1 } { 2 } \sum _ { y } p ( y ) \biggm | \frac { q ( y ) } { p ( y ) } - 1 \biggm | ,\tag{21}
$$

valid here since softmax distributions have full support. The first form is used in the derivation, and the second expresses $\bar { \varepsilon } _ { t }$ through the importance ratio and underlies its estimation below. Fix a cycle of the step-t+1 rollout with conditioning states $s _ { 1 } , \ldots , s _ { K }$ , and write $\alpha _ { i } = \alpha ( s _ { i } ) , \tilde { \alpha } _ { i } = \tilde { \alpha } ( s _ { i } )$ , and $\delta _ { i } = \mathrm { T V } \bar { \left( \right)} p ^ { ( t + 1 ) } ( \cdot  { \mid } s _ { i } ) , p ^ { ( t ) } ( \cdot  { \mid } s _ { i } ) $ . Since every factor lies in [0, 1] and $\tilde { \alpha } _ { n } - \alpha _ { n } \leq \delta _ { n }$ by Eq. 19, the difference between products can be expanded as the following telescoping sum for each $l \leq K \colon$

$$
\prod _ { i = 1 } ^ { l } \tilde { \alpha } _ { i } - \prod _ { i = 1 } ^ { l } \alpha _ { i } = \sum _ { n = 1 } ^ { l } \Bigl ( \prod _ { i < n } \tilde { \alpha } _ { i } \Bigr ) \bigl ( \tilde { \alpha } _ { n } - \alpha _ { n } \Bigr ) \Bigl ( \prod _ { n < i \leq l } \alpha _ { i } \Bigr ) \leq \sum _ { n = 1 } ^ { l } \delta _ { n } .\tag{22}
$$

Summing over $l = 1 , \ldots , K$ , taking expectations over the cycles of the rollout, and bounding each $\mathbb { E } [ \delta _ { n } ]$ by $\bar { \varepsilon } _ { t }$ , the shifts are counted $\textstyle \sum _ { l = 1 } ^ { K } l = K ( K + 1 ) / 2$ times across the chain terms, yielding

$$
\mathbb { E } \left[ \sum _ { l = 1 } ^ { K } \prod _ { i = 1 } ^ { l } \tilde { \alpha } _ { i } \right] ~ - ~ \mathbb { E } \left[ \sum _ { l = 1 } ^ { K } \prod _ { i = 1 } ^ { l } \alpha _ { i } \right] ~ \le ~ \frac { K ( K + 1 ) } { 2 } { \bar { \varepsilon } } _ { t } .\tag{23}
$$

The second expectation is the expected acceptance-chain value of Eq. 6 at step t+1, and the first is the same quantity measured against the training-time target: it is the acceptance-chain quantity underlying the DCA loss, with both targets evaluated at the same draft states from the new rollout.

Magnitude of $\bar { \varepsilon } _ { t } .$ . The shift is a property of the RL update alone, and its leading-order magnitude follows from two classical facts. First, by Eq. 17 the two targets are softmax outputs of the same network at neighboring parameter points, so their KL divergence admits the following second-order expansion in the parameter update $\overline { { \Delta \theta } } = \theta _ { t + 1 } - \theta _ { t }$ , evaluated at the same conditioning state s:

$$
\mathrm { K L } \Big ( p ^ { ( t ) } ( \cdot \mid s ) \Big \| p ^ { ( t + 1 ) } ( \cdot \mid s ) \Big ) = \frac { 1 } { 2 } \Delta \theta ^ { \top } F ( s ) \Delta \theta + O \big ( \| \Delta \theta \| ^ { 3 } \big ) ,\tag{24}
$$

where $F ( s )$ is the Fisher information matrix of the target model at state $s ,$ the Hessian of the divergence at zero displacement. Second, Pinsker’s inequality $\mathrm { T V } \leq \sqrt { \mathrm { K L } / 2 }$ , Jensen’s inequality, and the maximization over draft positions in Eq. 20 convert Eq. 24 into this bound on target shift:

$$
\bar { \varepsilon } _ { t } \ \leq \ \frac { 1 } { 2 } \sqrt { \Delta \theta ^ { \top } \bar { F } \Delta \theta + O ( \| \Delta \theta \| ^ { 3 } ) } ,\tag{25}
$$

where $\bar { F }$ is the Fisher matrix averaged over the conditioning states at the maximizing position: to leading order, the shift is at most half the length of the update measured in the Fisher metric, and vanishes linearly with the learning rate. The clipped surrogate, small learning rate, and gradient-norm clipping aim to limit policy movement but do not determine $\bar { \varepsilon } _ { t }$ , the per-update target-policy shift.

Measuring $\bar { \varepsilon } _ { t } .$ . The ratio form of $\operatorname { E q }$ . 20 makes the shift directly measurable: rollout tokens are sampled from $p ^ { ( t ) }$ , so the sample mean of $\scriptstyle { \frac { 1 } { 2 } } \left| \rho - 1 \right|$ over a training batch estimates the inner expectation unbiasedly, at the cost of one additional scoring pass of the batch under the updated policy. In our runs, the measured per-token KL $\left( p ^ { ( t ) } \parallel p _ { \mathrm { r e f } } \right)$ remains below 0.007 nats throughout training. This fixed-reference KL provides contextual evidence that the policy trajectory remains localized, although it does not directly estimate the consecutive-policy shift $\bar { \varepsilon } _ { t }$ in the acceptance-chain bound of Eq. 23.

Conclusion. Eq. 23 shows that the acceptance-chain shift is controlled by the consecutive-policy shift $\bar { \varepsilon } _ { t } .$ , with a worst-case bound of $\frac { K ( \dot { K } + 1 ) } { 2 } \bar { \varepsilon } _ { t }$ . Because the head is refreshed after every RL step, the bound depends only on the current policy update rather than accumulated training drift; the sustained acceptance gains in our runs show that this bounded staleness does not prevent effective online adaptation. Under rejection sampling, Eq. 5 makes the acceptance rate 1-Lipschitz in the target distribution under total variation, with both the draft distribution and the conditioning state held fixed.

## I OPTIMIZATION ANALYSIS OF TRAINING OBJECTIVE

The most direct way to improve rollout efficiency is to train the MTP head with a differentiable surrogate for the acceptance length, the TV-chain objective ${ \mathcal { L } } _ { \mathrm { T V } }$ of Section 3.2, which prior work adopts. This surrogate is multiplicative in the per-position acceptance probabilities, so its value on a draft step measures how well the head already drafts there, and a direct design carries that value into the gradient. This appendix compares TV and DCA gradients (Eq. 7) for individual draft steps and batches under SGD and Adam, characterizing how the logarithm changes relative cycle weights.

Setup. The head enters either objective only through the acceptance probabilities of one draft-thenverify cycle, which compound into the per-cycle acceptance-chain surrogate specified in Eq. 6,

$$
M = \sum _ { l = 1 } ^ { K } \prod _ { i = 1 } ^ { l } \alpha _ { i } , \qquad \mathcal { L } _ { \mathrm { { T V } } } \triangleq - M , \qquad \mathcal { L } _ { \mathrm { { D C A } } } \triangleq - \log M ,\tag{26}
$$

where $\mathcal { L } _ { \mathrm { T V } }$ differs by an affine constant from the normalized form $1 - M / K$ in which prior work writes it. Since softmax distributions have full support, $\alpha _ { k } \in ( 0 , 1 ]$ and $M \in ( 0 , K ]$ , the lower end approached by a head whose drafts are rejected at the first position and the upper end attained by one whose drafts are always accepted. The verify-gated form of Eq. 9 replaces K by min(j, K) in the chain sum, which changes the value of M and the upper end of this range and leaves the statements below unaffected. Both objectives are built from the same acceptance probabilities of the same cycle, so the logarithm of the shared chain value accounts for every difference between the two objectives.

Gradient Analysis. To compare the gradients of the two objectives, differentiate each of them with respect to the individual acceptance probabilities. The recorded target distributions enter both objectives as constants, and the gradients below are taken with respect to the head parameters ϕ alone. The derivative of M at a single position follows from its multilinearity in the acceptance probabilities: $\alpha _ { k }$ occurs once in every term whose index is at least k and in no other term, and factoring it out separates the product over all preceding positions from the sum of products over subsequent positions:

$$
\frac { \partial M } { \partial \alpha _ { k } } = \sum _ { l = k } ^ { K } \prod _ { i \leq l , i \neq k } \alpha _ { i } = \Bigl ( \prod _ { i < k } \alpha _ { i } \Bigr ) \sum _ { l = k } ^ { K } \prod _ { i = k + 1 } ^ { l } \alpha _ { i } .\tag{27}
$$

The remaining sum is the chain quantity of Eq. 26 formed over the positions after k, a sum of $K - k + 1$ terms, with the first term equal to one and each remaining term between zero and one:

$$
D _ { k } \ \triangleq \ \sum _ { l = k } ^ { K } \prod _ { i = k + 1 } ^ { l } \alpha _ { i } \ \in \ [ 1 , K - k + 1 ] .\tag{28}
$$

Since $\mathcal { L } _ { \mathrm { T V } } = - M$ and $\mathcal { L } _ { \mathrm { D C A } } = - \log M$ , the coefficients the two objectives place on position k are $\partial M / \partial \alpha _ { k }$ and $M ^ { - 1 } \partial M / \partial \alpha _ { k }$ , respectively. The derivative of M yields the following coefficients:

$$
- \frac { \partial \mathcal { L } _ { \mathrm { T V } } } { \partial \alpha _ { k } } = \Bigl ( \prod _ { i < k } \alpha _ { i } \Bigr ) D _ { k } , \qquad - \frac { \partial \mathcal { L } _ { \mathrm { D C A } } } { \partial \alpha _ { k } } = \Bigl ( \prod _ { i < k } \alpha _ { i } \Bigr ) \frac { D _ { k } } { M } ,\tag{29}
$$

where $D _ { k } / M$ matches the coefficient of Eq. 8, its denominator shared across positions and its numerator at most $K - k + 1$ , which bounds its variation across positions by a factor of K. Assembling the positions into the gradient with respect to ϕ places the common positive factor outside the sum,

$$
\nabla _ { \phi } { \mathcal { L } } _ { \mathrm { D C A } } = - { \frac { 1 } { M } } \sum _ { k = 1 } ^ { K } { \frac { \partial M } { \partial \alpha _ { k } } } \nabla _ { \phi } \alpha _ { k } = - { \frac { 1 } { M } } \nabla _ { \phi } M = { \frac { 1 } { M } } \nabla _ { \phi } { \mathcal { L } } _ { \mathrm { T V } } .\tag{30}
$$

The factor is positive at every parameter value and carries no position index, so on this cycle the two objectives share their descent direction, their stationary points, their minimizers and the relative weights they place on the draft positions, the depth weighting of Section 3.3 among them. The logarithm only rescales that cycle’s gradient magnitude by the factor $1 / M$ , common to all positions.

Optimization on a Draft Step. How much of the factor $M ^ { - 1 }$ reaches the parameters depends on the optimizer. Under SGD with learning rate η, Eq. 30 gives a draft step with chain value M the update $- \eta \nabla _ { \phi } \mathcal { L } _ { \mathrm { D C A } }$ under $\mathcal { L } _ { \mathrm { D C A } }$ and the update $- \eta M \nabla _ { \phi } \mathcal { L } _ { \mathrm { D C A } }$ under ${ \mathcal { L } } _ { \mathrm { T V } }$ . The rate at which the shared direction acts on the parameters is therefore η under $\mathcal { L } _ { \mathrm { D C A } }$ and $\eta M \in ( 0 , \eta K ]$ under ${ \mathcal { L } } _ { \mathrm { T V } }$ , which makes the effective rate of ${ \mathcal { L } } _ { \mathrm { T V } }$ a function of the head’s own acceptance on that draft step and leaves the effective rate of $\mathcal { L } _ { \mathrm { D C A } }$ at the scheduled learning rate, without extra scaling by the value M.

Adam divides the first moment of the gradient history by the square root of the second moment, and a factor common to every gradient in that history scales the first moment and the square root of the second moment alike, so the factor cancels in the quotient up to the constant that stabilizes the denominator. Ignoring Adam’s stabilizing constant, a positive constant scaling of the entire gradient history cancels in the moment ratio; factors varying across draft cycles generally do not cancel.

Optimization on a Batch. The optimizer does not receive the gradient of a single draft step: what reaches it is the batch average, in which $M ^ { - 1 }$ is no longer a scalar but a weight carried by each draft step separately. Writing $g _ { r } \triangleq \nabla _ { \phi } [ - \log M _ { r } ]$ for the gradient $\mathcal { L } _ { \mathrm { D C A } }$ produces on draft step $r ,$ Eq. 30 makes the gradient of $\mathcal { L } _ { \mathrm { T V } }$ on the same draft step $\bar { M _ { r } } g _ { r }$ , giving the following two batch gradients:

$$
\nabla _ { \phi } \mathbb { E } _ { r } \big [ \mathcal { L } _ { \mathrm { T V } } \big ] \ = \ \mathbb { E } _ { r } \big [ M _ { r } g _ { r } \big ] , \qquad \nabla _ { \phi } \mathbb { E } _ { r } \big [ \mathcal { L } _ { \mathrm { D C A } } \big ] \ = \ \mathbb { E } _ { r } \big [ g _ { r } \big ] ,\tag{31}
$$

where $\mathbb { E } _ { r }$ averages over the recorded draft steps that make up the batch. Relative to the DCA cycle gradients $^ { g _ { r } , }$ the TV-chain objective assigns the explicit weights $M _ { r } ; \mathbf { D C A }$ averages $g _ { r }$ without these factors. The norms of $g _ { r }$ can still differ across cycles. Consequently, the stationary conditions $\mathbb { E } _ { r } [ M _ { r } g _ { r } ] = 0$ and $\mathbb { E } _ { r } [ g _ { r } ] = 0$ need not coincide. The logarithm thus changes the batch objective, favoring smaller-M cycles in relative gradient weight without guaranteeing faster convergence.

Conclusion. The logarithm preserves the single-cycle gradient direction and relative depth weights while changing the batch objective through cycle reweighting. The experiments assess whether this change in relative cycle weights improves realized acceptance and reduces overall RL training time.

## J LIMITATIONS

Run length and compute allocation. Net savings depend on the run length and the rollout share of total compute. In both from-scratch experiments, per-step acceleration begins within 30 steps, and the 500-step runs recover the initial cost with substantial net savings. Typical RL training spans hundreds or thousands of steps, providing ample time to amortize this brief initial training overhead.

Distribution specialization. The grown head is tailored to the rollout distribution of the current RL run. As shown in Appendix D.3, it retains higher acceptance within its training domain than across domains. Its reuse across unrelated tasks therefore requires separate evaluation, while broad deployment remains a distinct setting from the acceleration studied within a single ongoing RL run.

Evaluation scope. Our measurements cover 4B–7B models on mathematical and code reasoning with a single node of eight GPUs. Larger models, multi-node systems, and asynchronous RL may have different compute and communication costs, so their speedups require direct measurement. The projections in Appendix D.2 illustrate scaling under assumptions, rather than measured performance.