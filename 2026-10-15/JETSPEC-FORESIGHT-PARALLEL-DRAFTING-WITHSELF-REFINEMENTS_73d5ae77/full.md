# JETSPEC++: FORESIGHT PARALLEL DRAFTING WITHSELF REFINEMENTS

Zhaoxiang Feng<sup>∗1</sup> Lanxiang Hu<sup>∗1</sup> Haoran Yuan<sup>2</sup> Yu-Yang Qian<sup>3</sup> Jihui Yang<sup>4</sup> Peng Zhao<sup>3</sup> Yibo Zhu<sup>5</sup> Tajana Rosing<sup>1</sup> Hao Zhang<sup>1</sup> <sup>1</sup>UC San Diego <sup>2</sup>UIUC <sup>3</sup>Nanjing University <sup>4</sup>Meta <sup>5</sup>StepFun

## ABSTRACT

Speculative decoding’s effectiveness is governed by two quantities: the acceptance rate of drafted tokens and the cost of producing them. We show the two objectives can be co-optimized with algorithm-system co-design, and present Jet-Spec++, which retains a strong learned draft model while hiding its cost almost entirely. JetSpec++ combines three components: intra-block self refinement, which trains the draft model to iteratively correct its own parallel predictions within a block of tokens; inter-block foresight drafting, which begins the next block before the current one is verified; and pipelined drafting and verification, which overlaps the two stages so drafting’s marginal wall-clock cost approaches zero. In comparison with state-of-the-art speculative decoding methods, JetSpec++ attains a wall-clock speedup of up to 7.45× over autoregressive (AR) decoding and 1.54× over DSpark, with up to 1,361 tokens per second for a single request on GB200, and an average acceptance length of 8.94 across seven benchmarks, while preserving the target model’s output distribution. Our code and models are available at: https://github.com/hao-ai-lab/JetSpec.

## 1 INTRODUCTION

Speculative decoding (SD) is a widely adopted technique for parallelizing large language model (LLM) generation (Leviathan et al., 2023; Chen et al., 2023). Its effectiveness is determined by two factors, the acceptance rate and the drafting cost. Earlier work showed that the target model’s hidden states of the verified context can align the draft model with the target distribution and thereby raise the acceptance rate (Li et al., 2024a; 2025). DFlash employs parallel drafting and has further reduced per-token drafting cost (Chen et al., 2026), providing bigger headroom for higher throughput (Cheng et al., 2026) and lower latency (Hu et al., 2026a). A natural question to ask is whether we can further reduce drafting cost to zero on the critical path.

One solution is to further push down the drafting cost. Cheap drafting usually relies on model-free techniques such as prompt lookup decoding (Saxena, 2023; Somasundaram et al., 2025) or retrieval from the text generated so far (Oliaro et al., 2025; Hu et al., 2025). However, model-free methods show high variance and low average acceptance rates, particularly when reusable patterns are sparse or fail to generalize. The question then becomes, can we retain the high acceptance rate of modelbased drafting, while removing drafting cost from the draft-then-verify critical path?

The drafting cost can be removed from the critical path by overlapping drafting for the next round with verification in the current round (Figure 1). However, for a learned draft model conditioned on the target model’s hidden states, the key challenge is that it must draft ahead before the next verified context becomes available (Zhang et al., 2026c). Accurately predicting this future context is therefore critical to maintaining draft quality.

We present JetSpec++, which maintains a target-conditioned parallel draft model and overlaps drafting for the next step with verification of the current step. To draft block t+1 while block t is still being verified, however, the draft model must operate without two pieces of information that become available only after verification: (1) the number of accepted draft tokens in block t and the corresponding bonus token, which becomes the anchor, the first token of block t+1, and (2) the target hidden states for the newly verified context in block t.

![](images/d7e8aba5f02e355d93d15dc3e3f8e1f73d272531a90f6a2edcd57c5e6a890a02.jpg)  
Figure 1: Non-pipelined drafting versus JetSpec++. (a) A draft model that reads the target’s hidden states drafts only after verification. (b) JetSpec++ runs foresight and Jacobi passes while block t is verified. Segment lengths are schematic.

JetSpec++ addresses both missing components through Foresight Drafting. Rather than waiting for the verification outcome, the draft model anticipates multiple possible “world lines,” each corresponding to a different acceptance-length outcome. For each request, a hidden-to-hidden adapter predicts the target hidden states of block t, which is still being verified. A lightweight decoder then maps these predicted states to possible anchor tokens. The predicted hidden states and anchor token are then used to draft block t+1.

Because drafting runs concurrently with target verification, additional draft-model computation incurs no extra latency as long as it remains within the verification window. JetSpec++ exploits this otherwise idle compute by augmenting a causal parallel draft model built on the architecture of JetSpec (Hu et al., 2026a) with Jacobi-style self-refinement, allowing the draft model to iteratively revise its parallel predictions over multiple passes. Each additional refinement pass progressively moves the draft distribution toward a more autoregressive factorization with better target alignment. Between two consecutive passes of the draft model, JetSpec++ additionally simulates verification with the adapter and the decoder as a proxy for the target. A context refresh re-predicts the target states of the draft, an anchor replacement replaces the first draft token that disagrees with the decoder, and the next pass refines the rest of the block (Section 4.3). In summary:

• We propose Foresight Drafting, which enables draft-verification pipelining for modern speculative decoding whose learned draft models depend on target-model hidden states. It anticipates possible verification outcomes and predicts the corresponding target hidden states and anchor tokens, thereby drafting the next block before the current block is verified.

• We introduce Jacobi-style self-refinement with context refresh and anchor replacement for a causal parallel draft model that builds on JetSpec’s architecture and is trained to refine its own draft (Section 4.3), exploiting otherwise hidden drafting compute within the verification window to progressively strengthen causal token dependence and improve target alignment over multiple passes.

• We evaluate JetSpec++ on seven benchmarks, where at verification budget 16 it attains the longest acceptance length of all methods on every benchmark at temperatures 0 and 1, with an average of 8.94 tokens per round against 6.89 for DSpark at temperature 0 (Table 1).

• We integrate JetSpec++ into a serving engine with CUDA graphs and the draft model on a second GPU, where its end-to-end speedup is up to 7.45× and 1.54× over AR and DSpark, at up to 1,361 tokens per second for a single request, and JetSpec++ cuts the per-round overhead beyond verification by up to 84% relative to DSpark (Table 3).

## 2 RELATED WORK

Speculative Decoding. Speculative decoding drafts several tokens and verifies them in one target forward pass, and its rejection-sampling rule leaves the output distribution unchanged (Leviathan et al., 2023; Chen et al., 2023). Its speedup grows with the number of accepted tokens and shrinks with the cost of drafting them. To accept more tokens per pass, tree verification scores many candidate branches at once (Miao et al., 2024; Chen et al., 2024; Li et al., 2024b), and knowledge distillation aligns the draft model with the target (Zhou et al., 2024; Liu et al., 2024; Goel et al., 2024). Medusa (Cai et al., 2024) predicts future positions from the last hidden state, and EAGLE (Li et al., 2024a), EAGLE-3 (Li et al., 2025) and HASS (Zhang et al., 2025) draft from the target’s features. To draft cheaply, parallel drafting reduces draft cost by predicting an entire block in one forward pass (An et al., 2026; Xiao et al., 2024). Recent methods further improve parallel drafting alignment by conditioning on target hidden states (Chen et al., 2026), but their parallel per-position predictions are not conditioned on the earlier realized draft tokens within the block (Wang et al., 2026c). Sequential heads (Cheng et al., 2026; Huang et al., 2026; Zheng & Li, 2026) restore this dependency at the cost of serial computation. DSpine (Zhang et al., 2026a) instead writes each position’s predicted feature into its successor after every layer, so the dependency forms through depth within one forward pass. JetSpec (Hu et al., 2026a) introduces causal conditioning over hidden states within the block, while xPress (Wang et al., 2026b) trains a separate refiner and refines the draft by Jacobi iteration. JetSpec++ instead lets the draft model refine its own block.

Parallel Decoding. Parallel decoding lets a model itself generate several of its own tokens per forward pass, without a separate draft model. Jacobi decoding updates every position of a block by fixed-point iteration and converges to the greedy output (Song et al., 2021; Santilli et al., 2023). Lookahead decoding (Fu et al., 2024) collects n-grams from the Jacobi trajectory and verifies them, while consistency distillation (Kou et al., 2024) and Jacobi forcing (Hu et al., 2026b) train an AR model to reach the fixed point in fewer iterations. Discrete diffusion models generate text by denoising all positions in parallel over several steps (Sahoo et al., 2024; Nie et al., 2025; Ye et al., 2025). Block-wise discrete diffusion denoises one block at a time and conditions each block on the earlier ones (Arriola et al., 2025; Wu et al., 2026a), while caching (Wu et al., 2026b), pseudo-trajectory distillation (Qian et al., 2026) and multi-block decoding (Wang et al., 2026a) further accelerate diffusion inference. JetSpec++ applies Jacobi iteration to the draft model rather than the target.

Draft-Verification Pipelining. Prior work hides drafting latency by running the draft model asynchronously with target verification (Liu et al., 2025; McDanel, 2025). Because the verification outcome is not yet known, recent methods hedge against multiple outcomes: SpecBranch (Shen et al., 2026) explores alternative branches, while speculative speculative decoding (SSD) (Kumar et al., 2026) prepares drafts for likely verification outcomes. SwiftSpec (Zhang et al., 2026c) uses asyn chronous tree drafting to remove draft latency from the critical path. Target-conditioned draft models make pipelining harder because the representation of the context under verification is unavailable. Prior approaches obtain an early target-side signal from partial-depth computation (Bhendawade et al., 2025; Venkatesha et al., 2025; Yu et al., 2026), and SSD substitutes the draft model’s own activations for it (Kumar et al., 2026, Appendix E). D<sup>2</sup>SD (Zhang et al., 2026b) forms a distribution over the number of accepted tokens k from the draft model’s own confidences, and a second draft model redrafts the block from the most likely prefix lengths, without pipelining. DPara (Liu et al., 2026) pipelines a DSpark (Cheng et al., 2026) draft model without a fallback by precomputing its backbone for every possible k, so the backbone reads the tokens under verification without their target states. In contrast, JetSpec++ predicts the missing target hidden states directly, enabling a target-conditioned parallel draft model to speculate ahead without executing the target model.

## 3 PRELIMINARIES

An AR model generates one token per forward pass, and each pass is bound by memory bandwidth, so compute sits idle. Speculative decoding (SD) (Leviathan et al., 2023; Chen et al., 2023) uses this idle compute to verify several drafted tokens in one forward pass. A cheap draft model q proposes γ tokens $d _ { 1 } , \ldots , d _ { \gamma }$ after the verified context $x ,$ and the target model $p$ scores all of them in a single forward pass. From left to right, draft token $d _ { i }$ is accepted with probability min $\{ 1 , p ( d _ { i }$ $x , d _ { < i } ) / q ( d _ { i } \mid x , d _ { < i } ) \}$ . At the first rejection the rest of the draft is discarded and the target model samples a token of its own, the bonus token, from the residual distribution $\propto \operatorname* { m a x } \{ p - q , 0 \}$ . If every draft token is accepted, the bonus token is sampled from $p$ at position $\gamma + 1$ . This procedure is lossless: it preserves the target model’s generation distribution exactly.

Each round commits the accepted prefix plus one bonus token, and its expected number of committed tokens is the acceptance length τ. A round costs one drafting step and one verification step, which take times $T _ { \mathrm { d r a f t } }$ and $T _ { \mathrm { v e r i f y } }$ . The per-committed-token latency is therefore $L = ( T _ { \mathrm { d r a f t } } + \dot { T _ { \mathrm { v e r i f y } } } ) / \tau$ so a more accurate draft model raises τ and a faster draft model lowers $T _ { \mathrm { d r a f t } }$

## 3.1 PARALLEL DRAFTING

An AR draft model needs $\gamma$ sequential forward passes, so $T _ { \mathrm { d r a f t } }$ grows with $\gamma .$ Modern parallel draft models such as DFlash take the previous bonus token as the anchor $y _ { 0 } .$ . From here on, x denotes the verified context before $y _ { 0 }$ . Conditioned on the target hidden states $h _ { x }$ of $x ,$ , these models fill the other $\gamma$ positions with mask tokens, and predict all $\gamma$ positions in one forward pass of the draft model $q _ { \theta }$ with parameters θ (Chen et al., 2026):

$$
q _ { \theta } ( y _ { 1 } , \dots , y _ { \gamma } \mid x , h _ { x } , y _ { 0 } ) = \prod _ { i = 1 } ^ { \gamma } q _ { \theta } ( y _ { i } \mid x , h _ { x } , y _ { 0 } )\tag{1}
$$

$T _ { \mathrm { d r a f t } }$ then barely depends on $\gamma ,$ , but each position is predicted without the tokens chosen at the other positions of the block. JetSpec (Hu et al., 2026a) makes the draft model causal, so in one forward pass every position attends to the hidden states at its own position and at the positions before it. We build on this architecture to enable causal token factorization via multi-pass Jacobi iterations.

## 3.2 JACOBI DECODING FOR PARALLEL DRAFTING

Parallel decoding lets an AR LLM decode a γ-token block of its own greedy output, $y _ { i } =$ arg $\operatorname* { m a x } _ { v } p ( v \ \mid \ x , y _ { < i } )$ for $i = 1 , \ldots , \gamma$ , by updating all $\gamma$ positions in one forward pass. Jacobi decoding (Song et al., 2021; Santilli et al., 2023) formulates this process as solving these $\gamma$ equations by fixed-point iteration, updating every position from the previous iteration r:

$$
y _ { i } ^ { ( r + 1 ) } = \arg \operatorname* { m a x } _ { v } \ p \big ( v \mid x , y _ { < i } ^ { ( r ) } \big ) , \qquad i = 1 , \ldots , \gamma\tag{2}
$$

Under a causal mask, each Jacobi iteration requires a single forward pass. The resulting iterates form a Jacobi trajectory that converges to the AR fixed point under greedy decoding (Santilli et al., 2023). JetSpec++ trains the draft model on both fully noised inputs and partially converged blocks (Section 4.3).

## 4 METHODS

In this section, we describe how JetSpec++ works. Draft-verification pipelining (Section 4.1) overlaps drafting with verification, reducing the critical-path latency for decoding and yielding the throughput model in Eq. 4. The challenge is that, in a pipelined round, the draft model must begin before the verification outcome is available. JetSpec++ addresses this with Foresight Drafting (Section 4.2), which learns predictive features that recover most of the acceptance-length loss (Appendix A). Finally, intra-block causal self-refinement (Section 4.3) makes use of the bubble exposed by pipelining to further increase the acceptance length. Between two passes of the draft model, a context refresh and an anchor replacement rerun the adapter and the decoder of Foresight Drafting to correct the draft.

## 4.1 DRAFT-VERIFICATION PIPELINING

Draft-verification pipelining drafts block t+1 while block t is being verified. In SSD’s terms (Kumar et al., 2026), the draft process anticipates the verification outcome $( k , a )$ , the number k of accepted draft tokens together with the bonus anchor token a, and fills a speculation cache with one predrafted block per anticipated outcome. When verification returns, the realized $( k , a )$ is matched. A cache hit makes the matching pre-drafted block the next block, and a cache miss falls back to an ordinary speculative step. Every committed token is verified by the rejection sampling acceptance rule (Leviathan et al., 2023), so the output distribution is unchanged.

Foresight Drafting (Section 4.2) predicts the target hidden states that are missing until verification ends and estimates the distribution of the outcome from the prediction (Figure 2). The draft process of Figure 2 (a) is hidden when

$$
T _ { \mathrm { d r a f t } } = T _ { \mathrm { p r e d } } ( 1 ) + R T _ { \mathrm { p r e d } } ( | S | ) + ( J + 1 ) T _ { \mathrm { p a s s } } ( | S | ) + T _ { \mathrm { h o s t } } \leq T _ { \mathrm { v e r i f y } }\tag{3}
$$

![](images/806e8af934cc52e726d3239ab894bfd11601cd3a43b6f04890c53d5041eea5ab.jpg)

![](images/fb924588b7a7d79be2b23942972a6118086318dff51027c56b3dc74d9b8fbf41.jpg)  
Figure 2: Illustration of Foresight Drafting. (a) Foresight Drafting drafts the set $S$ of selected outcomes (Eq. 7) in one batch with J+1 passes. (b) At each target layer ℓ that the draft model reads, the adapter attends to the verified prefix states $h _ { < n } ^ { \ell }$ and predicts the current-block states $\hat { h } _ { n : n + \gamma } ^ { \ell } .$ A decoder converts the top-layer hidden state predictions into the outcome distribution $P ( k , a )$ of Eq. 6. (c) The speculation cache for block t+1 contains outcome $k ,$ the draft tokens $d _ { 1 } , \ldots , d _ { k }$ that outcome k accepts, the corresponding adapted states $\hat { h }$ the draft model will read, and the decoded bonus token a as the anchor of block t+1. An outcome whose calibrated score (Appendix B.1) reaches the threshold λ will be pre-drafted as a $\gamma \cdot$ -token block.

where $T _ { \mathrm { p r e d } } ( \cdot )$ is the time of one call of the adapter and the decoder over the given number of blocks, one for the foresight call on the block under verification and |S| for each context refresh on the drafts of the selected outcomes, $T _ { \mathrm { p a s s } } ( | S | )$ is the time of one batched draft-model forward over the set S of selected outcomes, J is the number of Jacobi refinement passes, R is the number of context refreshes (Section 4.3), and $T _ { \mathrm { h o s t } }$ is the time of the CPU work between GPU phases. A choice of more selected outcomes raises $T _ { \mathrm { p a s s } } ( | S | )$ , so under Eq. 3 it leaves fewer refinement passes J.

Throughput Model. Through draft-verification pipelining, on a hit the critical path overhead reduces to max $( T _ { \mathrm { v e r i f y } } , T _ { \mathrm { d r a f t } } ) + T _ { \mathrm { c o m m } }$ , where $T _ { \mathrm { c o m m } }$ is the communication time between the two processes. On a miss, the draft model drafts a fallback block from the verified context in one pass (the initial draft $J _ { 0 }$ of Section 4.3), which adds $T _ { \mathrm { f a l l b a c k } }$ to the critical path, and the same round verifies it. With $p _ { \mathrm { h i t } }$ being cache hit probability, and $\tau _ { \mathrm { h i t } }$ and $\tau _ { \mathrm { m i s s } }$ being pre-drafted block and fallback block acceptance lengths, we adapt the speedup model of SSD (Kumar et al., 2026) and write

$$
\mathrm { t h r o u g h p u t } \ = \ \frac { p _ { \mathrm { h i t } } \tau _ { \mathrm { h i t } } + \left( 1 - p _ { \mathrm { h i t } } \right) \tau _ { \mathrm { m i s s } } } { \mathrm { m a x } \left( T _ { \mathrm { v e r i f y } } , \ T _ { \mathrm { d r a f t } } \right) + T _ { \mathrm { c o m m } } + \left( 1 - p _ { \mathrm { h i t } } \right) T _ { \mathrm { f a l l b a c k } } }\tag{4}
$$

In the numerator, since $\tau _ { \mathrm { m i s s } }$ is smaller than $\tau _ { \mathrm { h i t } }$ , throughput grows with $p _ { \mathrm { h i t } }$ <sub>t</sub>. Foresight drafting (Section 4.2) raises $p _ { \mathrm { h i t } }$ while keeping $\tau _ { \mathrm { h i t } }$ close to its value under true hidden states (Section 5), and Jacobi iterations (Section 4.3) raise $\tau _ { \mathrm { h i t } }$

Placement. JetSpec++ targets per-request latency. Because the two processes are separate, they can run on separate devices, as in SSD (Kumar et al., 2026).

## 4.2 INTER-BLOCK FORESIGHT DRAFTING

Foresight drafting consists of an adapter, a decoder, and a draft-model backbone, where data flow is illustrated in Figure 2. Under each possible outcome $k ,$ positions $1 , \ldots , k$ of block t hold the known draft tokens $d _ { 1 } , \ldots , d _ { k }$ , so the missing target-side hidden states belong to known tokens, and the state at each position $j \le k$ is predictable by a learned map from the tokens up to position j. This allows training a specialized adapter to predict the hidden state at each position $j$ of block t.

Hidden-to-Hidden Adapter. The adapter predicts, in one parallel pass, the target hidden states of all $\gamma + 1$ positions of the block under verification. At each target layer ℓ that the draft model reads, the adapter attends to the cached prefix at that layer and predicts the hidden states of the new future positions:

$$
\hat { h } _ { n + j } ^ { \ell } \ : = \ : A _ { \ell } \big ( h _ { < n } ^ { \ell } , \ : \hat { h } _ { n : n + j } ^ { \ell - } \big ) , \qquad j = 0 , \ldots , \gamma\tag{5}
$$

Here n is the position of the anchor $y _ { 0 } .$ , the bonus token of the latest verification. $A _ { \ell }$ is the adapter module for layer ℓ, a short stack of transformer layers copied from the target (Appendix $\mathbf { A } ) . \mathfrak { h } _ { < n } ^ { \ell ^ { - } }$ is the target’s cached context states at layer ℓ, the layer-ℓ slice of $h _ { x }$ in Eq. 1, and $\hat { h } ^ { \ell ^ { - } }$ is the adapter’s prediction at the next lower target layer that the draft model reads, or the block’s token embeddings for the lowest layer. The adapter is trained separately from the draft model. During training, each module $A _ { \ell }$ reads the prediction of the module below it $\hat { h } _ { n : n + j } ^ { \ell ^ { - } } ,$ , as at inference, and the adapter is trained using a Smooth L1 loss plus a cosine distance between its predictions and the target’s true hidden states $h _ { n : n + j } ^ { \ell } .$ , concatenated over the layers that the draft model reads (Appendix A).

Decoder. The decoder turns the adapter’s predictions into the outcome distributions. The decoder reuses frozen weights from the target’s last few layers, above the deepest layer the adapter predicts, followed by the target’s final norm and LM head. It attends to the verified context through keys and values computed from the target’s true states (Appendix B.1). The decoder then runs under a causal mask over the adapter’s top-layer predictions $\hat { h } _ { n } , \ldots , \hat { h } _ { n + \gamma } .$ . Position $j$ yields ${ \hat { p } } ( \cdot \mid j )$ , an estimate of the target’s next-token distribution after the prefix up to j. The estimate at position $j - 1$ assigns draft token $d _ { j }$ the probability $\hat { p } _ { j } = \hat { p } ( d _ { j } \mid j - \bar { 1 } )$ ). For each $k < \gamma$

$$
P ( k ) = \hat { p } _ { 1 } \cdot \cdot \cdot \hat { p } _ { k } ( 1 - \hat { p } _ { k + 1 } ) , \qquad P ( k , a ) = \hat { p } _ { 1 } \cdot \cdot \cdot \hat { p } _ { k } \hat { p } ( a | k ) , \quad a \neq d _ { k + 1 }\tag{6}
$$

If all $\gamma$ draft tokens are accepted, then $\begin{array} { r } { P ( \gamma ) = \prod _ { j = 1 } ^ { \gamma } \hat { p } _ { j } } \end{array}$ and $P ( \gamma , a ) = P ( \gamma ) \hat { p } ( a \mid \gamma )$ . One decoder pass therefore scores every outcome at once, since row j serves both the acceptance of $d _ { j + 1 }$ and the bonus distribution at $k = j$ (Figure 3 in Appendix B). We call the decoder perfect if its estimate $\hat { p } ( \cdot \mid j )$ equals the target’s next-token distribution at the sampling temperature after every prefix j of the draft. With a perfect decoder, Eq. 6 is the exact distribution of the outcome under speculative sampling at any temperature, for any draft that is fixed before verification, such as the Jacobi-refined argmax draft of Eq. 8 (see Appendix F.1). In Appendix B.3 (Table 9), we show that the current design closely approximates the perfect decoder.

Outcome Selection. Given the outcome distributions, the draft process still needs to select the outcomes to pre-draft. Due to limited draft-side compute, the draft process selects, computes and caches a finite set of B most promising outcomes $( k , a )$ of Eq. 6 and $( \gamma , a )$ . Outcomes are disjoint events, so the hit probability of a cache is the sum of its entries’ probabilities:

$$
S ^ { \star } = \arg \operatorname* { m a x } _ { | S | \leq B } \sum _ { ( k , a ) \in S } P ( k , a )\tag{7}
$$

The optimum is the B most probable outcomes. In practice, decoder scores are imperfect, especially at longer lengths where errors compound through Eq. 6, so we calibrate the outcome scores on heldout prompts and retain candidates above a split-conformal threshold λ (Angelopoulos & Bates, 2023), up to cache size B. The threshold is fit offline to include the realized outcome at a target rate on held-out rounds. At inference, this threshold can yield a variable number of cached outcomes across rounds. For simplicity, our experiments instead cache the B highest-scoring outcomes in every round (Appendix B.1). Calibration affects only which outcomes are pre-drafted, never the verified output (Appendix F.3). Unlike SSD (Kumar et al., 2026), which fixes how many anchors each value of k receives, Eq. 7 selects the cache dynamically from each round’s outcome scores. With a perfect decoder, this selection is at least as good as any fixed allocation in every round (Proposition 2, Appendix F.2).

Batched Drafting. Having selected the outcomes to predict, each outcome $( k , a )$ is then drafted in the same way as an ordinary block. Its anchor is a, its context ends with $d _ { 1 } , \ldots , d _ { k }$ , and the draft model reads the adapter’s predictions at the positions up to k in place of the target’s hidden states. The predicted states are the only approximation, and verification keeps the output lossless whatever they are. All selected outcomes are drafted as one batch, with the refinement passes of Section 4.3.

## 4.3 INTRA-BLOCK SELF REFINEMENTS

A parallel draft model marginalizes each position over its possible predecessors, so its marginals are what PRESTO (Wang et al., 2026c) calls prefix-blind. DSpark (Cheng et al., 2026) gives an example: when the context admits “ofcourse” and “no problem”, the block may read “ofproblem”. Sequential heads and separate refiners repair prefix-blindness at the cost of a sequential or recurrent module. We instead let the draft model refine its own draft in parallel, with no module beyond those of Foresight Drafting, in the time that pipelining frees under verification. Self-refinement combines three operations, Jacobi iterations of the draft model, context refresh and anchor replacement.

Jacobi Iterations. Eq. 1 factorizes the block into per-position marginals, whereas the target generates it token by token, $\textstyle p ( y _ { 1 : \gamma } \mid x , y _ { 0 } ) = \prod _ { i = 1 } ^ { \gamma } p ( y _ { i } \mid x , y _ { < i } )$ . JetSpec++ makes the draft model follow this causal, token-level factorization through Jacobi iterations. Our draft model takes naturallanguage tokens, not mask embeddings, as the inputs, and attends causally within the block (Section 3.1), so each position can condition on the tokens drafted before it. Refinement is a Jacobi iteration (Section 3.2) of the draft model on its own draft:

$$
y _ { i } ^ { ( r + 1 ) } = \arg \operatorname* { m a x } _ { v } ~ q _ { \theta } \big ( v \big | x , h _ { x } , y _ { \leq i } ^ { ( r ) } \big ) , \quad \quad i = 1 , \ldots , \gamma\tag{8}
$$

Here $y ^ { ( r ) }$ is the block after r iteration passes, and $J _ { r }$ names a round that stops at $\mathbf { i t } ,$ so $J$ in Eq. 3 counts these passes. $J _ { 0 }$ is the initial draft, whose inputs are a fully noised block of tokens drawn from the verified context, and $y _ { 0 }$ is the anchor, the verified bonus token on a miss or the guessed a of a pre-drafted block. Causal attention lets position i read $y { \le } i$ for each iteration. The block therefore moves from the factorization of Eq. 1 toward an explicit causal token factorization, in which each position conditions on the drafted tokens $y _ { < i }$ before it.

Context Refresh. Jacobi iterations alone bring diminishing returns in acceptance length (Table 5, $R = 0 )$ . Between two consecutive passes, JetSpec++ therefore simulates verification on the current draft with the adapter and the decoder of Foresight Drafting. In a context refresh, the adapter repredicts the target hidden states of the current draft $y ^ { ( r ) }$ . Block t is still under verification, so both modules read the foresight predictions at its positions up to k in place of true states. Like the states of Foresight Drafting, the refreshed states are predictions, because the target has not yet read the draft.

Anchor Replacement. The decoder then reads the refreshed states and finds the first position at which its most likely token differs from the draft. An anchor replacement fixes the draft tokens before that position for the rest of the round, writes the decoder’s token aˆ at it, and fixes that position too. Like the anchor a of a pre-drafted block, aˆ is a guess of the token that the target would emit there. The next pass redrafts only the positions after the last fixed one (Appendix D). A round runs at most one context refresh, with its anchor replacement, between two consecutive passes, so it has $R \leq J$ context refreshes. Both operations change only the proposal, so verification keeps the output lossless (Appendix F.3).

Training. A draft model trained only on masked blocks is never trained to condition on tokens at its inputs. Following Jacobi forcing (Hu et al., 2026b), we train on blocks that pair a clean prefix with a noisy suffix, the shape of a partially converged Jacobi iterate. The first m positions hold the true tokens and the rest hold noise tokens ${ \tilde { y } } ,$ which we draw from the sequence’s own context rather than from Jacobi trajectories. We train the draft model to match the target model at every position:

$$
\mathcal { L } = \mathbb { E } _ { m } \sum _ { i = 1 } ^ { \gamma } \mathcal { D } \Big ( p ( \cdot \mid x , y _ { < i } ) \Big | \Big | q _ { \theta } \big ( \cdot \mid x , h _ { x } , \big [ y _ { 0 : m } ; \tilde { y } _ { m + 1 : \gamma } \big ] _ { \le i } \big ) \Big )\tag{9}
$$

Table 1: Acceptance length τ at verification budgets 16 and 32 for Qwen3-8B, in the setup of Section 5.1. JetSpec++ runs $J _ { 1 }$ with one context refresh and anchor replacement $\left( J _ { 1 }  – R _ { 1 } \right)$ and $J _ { 3 }$ with three of each $\left( J _ { 3 } – R _ { 3 } \right)$ , and caches the B = 16 highest-scoring outcomes per round (Appendix B.1).
<table><tr><td>Method</td><td>Budget</td><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>HumanEval</td><td>MBPP</td><td>LCB</td><td>MT-Bench</td></tr><tr><td colspan="9">Temperature = 0</td></tr><tr><td rowspan="2">DFlash</td><td>16</td><td>5.91</td><td>7.59</td><td>6.89</td><td>5.16</td><td>4.91</td><td>5.45</td><td>3.24</td></tr><tr><td>32</td><td>5.18</td><td>6.67</td><td>6.07</td><td>4.53</td><td>4.31</td><td>4.70</td><td>2.95</td></tr><tr><td rowspan="2">DSpark</td><td>16</td><td>7.31</td><td>9.01</td><td>8.35</td><td>6.57</td><td>6.17</td><td>6.92</td><td>3.89</td></tr><tr><td>32</td><td>7.06</td><td>8.66</td><td>7.73</td><td>6.25</td><td>5.85</td><td>6.38</td><td>3.77</td></tr><tr><td rowspan="2">Domino</td><td>16</td><td>6.91</td><td>8.74</td><td>8.09</td><td>6.06</td><td>5.90</td><td>6.27</td><td>3.74</td></tr><tr><td>32</td><td>6.25</td><td>7.96</td><td>7.37</td><td>5.61</td><td>5.29</td><td>5.92</td><td>3.49</td></tr><tr><td rowspan="2">JetSpec</td><td>16</td><td>5.90</td><td>7.43</td><td>6.78</td><td>5.28</td><td>4.92</td><td>5.45</td><td>3.17</td></tr><tr><td>32</td><td>6.02</td><td>7.81</td><td>7.02</td><td>5.34</td><td>4.96</td><td>5.53</td><td>3.19</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 1 } { - } R _ { 1 }$ </td><td>16</td><td>7.98</td><td>9.63</td><td>8.97</td><td>7.23</td><td>6.73</td><td>7.59</td><td>4.38</td></tr><tr><td>32</td><td>8.77</td><td>11.45</td><td>10.42</td><td>7.70</td><td>7.02</td><td>8.29</td><td>4.52</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 3 } – R _ { 3 }$ </td><td>16</td><td>9.78</td><td>11.04</td><td>10.49</td><td>8.79</td><td>8.24</td><td>8.95</td><td>5.29</td></tr><tr><td>32</td><td>11.83</td><td>14.55</td><td>13.13</td><td>10.27</td><td>9.25</td><td>10.50</td><td>5.51</td></tr><tr><td colspan="9">Temperature = 1</td></tr><tr><td rowspan="2">DFlash</td><td>16</td><td>5.19</td><td>5.37</td><td>4.25</td><td>4.26</td><td>4.24</td><td>4.78</td><td>2.81</td></tr><tr><td>32</td><td>4.70</td><td>4.85</td><td>4.16</td><td>3.97</td><td>3.81</td><td>4.20</td><td>2.64</td></tr><tr><td rowspan="2">DSpark</td><td>16</td><td>6.10</td><td>6.13</td><td>4.69</td><td>5.32</td><td>5.18</td><td>6.05</td><td>3.26</td></tr><tr><td>32</td><td>5.99</td><td>6.03</td><td>4.67</td><td>5.08</td><td>4.91</td><td>5.54</td><td>3.17</td></tr><tr><td rowspan="2">Domino</td><td>16</td><td>5.86</td><td>5.94</td><td>4.79</td><td>4.97</td><td>4.96</td><td>5.50</td><td>3.16</td></tr><tr><td>32</td><td>5.47</td><td>5.87</td><td>4.62</td><td>4.68</td><td>4.47</td><td>5.27</td><td>2.96</td></tr><tr><td rowspan="2">JetSpec</td><td>16</td><td>5.24</td><td>5.45</td><td>4.30</td><td>4.39</td><td>4.41</td><td>4.95</td><td>2.81</td></tr><tr><td>32</td><td>5.34</td><td>5.62</td><td>4.30</td><td>4.47</td><td>4.29</td><td>4.66</td><td>2.76</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 1 } { - } R _ { 1 }$ </td><td>16</td><td>6.88</td><td>6.50</td><td>5.25</td><td>5.84</td><td>5.53</td><td>6.55</td><td>3.51</td></tr><tr><td>32</td><td>7.33</td><td>7.24</td><td>5.48</td><td>6.07</td><td>5.80</td><td>7.01</td><td>3.59</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 3 } – R _ { 3 }$ </td><td>16</td><td>8.10</td><td>7.18</td><td>5.59</td><td>6.50</td><td>6.31</td><td>7.21 7.81</td><td>4.07</td></tr><tr><td>32</td><td>9.41</td><td>8.53</td><td>5.77</td><td>7.33</td><td>6.95</td><td></td><td>4.09</td></tr></table>

where D(· ∥ ·) denotes a divergence. We use forward $\mathrm { K L , }$ which outperforms reverse KL and crossentropy SFT in our ablations (Section 5.5). The clean prefix length m is sampled per block from a fixed schedule (Appendix E), so the model is trained to repair any clean-prefix block in Jacobi decoding. Appendix E provides our clean-prefix selection rule and details on our training setup.

## 5 EXPERIMENTS

## 5.1 SETUP

Models and baselines. The target models are Qwen3-8B (Yang et al., 2025a) and Qwen3.8- 27B (Qwen Team, 2026), both with thinking mode disabled. We compare JetSpec++ with DFlash (Chen et al., 2026), DSpark (Cheng et al., 2026), Domino (Huang et al., 2026), JetSpec (Hu et al., 2026a) and SSD (Kumar et al., 2026), the last with its Qwen3-0.6B draft model. For each target, all draft models except SSD’s are trained on the same 800k sequences regenerated by that target, with the recipes of Appendix E. The draft models are trained with a block size of 16 (the anchor and $\gamma = 1 5$ draft positions), and during inference we test the checkpoints with both budget 16 and 32, the block size at verification. DSpark runs without its confidence-based scheduling, so every method verifies a block of fixed size.

Benchmarks. We evaluate on GSM8K (Cobbe et al., 2021), MATH-500 (Hendrycks et al., 2021), AIME25 (Balunovic et al., 2025), HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), Live-´ CodeBench (LCB) (Jain et al., 2025) and MT-Bench (Zheng et al., 2023), with the first 128 prompts of a fixed shuffle, all 30 AIME25 problems and the 80 two-turn MT-Bench questions (Appendix E).

Table 2: Generalization to a larger target. Acceptance length τ for Qwen3.8-27B as in Table 1.
<table><tr><td>Method</td><td>Budget</td><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>HumanEval</td><td>MBPP</td><td>LCB</td><td>MT-Bench</td></tr><tr><td colspan="9">Qwen3.8-27B, temperature = 0</td></tr><tr><td rowspan="2">DSpark</td><td>16</td><td>8.02</td><td>8.94</td><td>8.30</td><td>8.19</td><td>7.62</td><td>5.19</td><td>4.00</td></tr><tr><td>32 16</td><td>7.57 8.71</td><td>8.49 9.53</td><td>7.64 8.58</td><td>7.13 9.18</td><td>6.96 8.41</td><td>4.84</td><td>3.83</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 1 } { - } R _ { 1 }$ </td><td>32</td><td>9.68</td><td>11.10</td><td>9.82</td><td>10.72</td><td>9.42</td><td>5.80 5.94</td><td>4.30 4.40</td></tr><tr><td>16</td><td>10.05</td><td>10.83</td><td>10.15</td><td>10.72</td><td>9.80</td><td>6.93</td><td>4.91</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 3 } – R _ { 3 }$  32</td><td></td><td>12.35</td><td>14.05</td><td>12.49</td><td>13.47</td><td>11.76</td><td>7.43</td><td>5.15</td></tr><tr><td colspan="9">Qwen3.8-27B, temperature = 1</td></tr><tr><td rowspan="2">DSpark</td><td>16</td><td>6.48</td><td>7.60</td><td>7.08</td><td>6.91</td><td>6.12</td><td>4.29</td><td>3.39</td></tr><tr><td>32</td><td>6.42</td><td>7.17</td><td>6.35</td><td>6.25</td><td>5.66</td><td>4.08</td><td>3.23</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 1 } { - } R _ { 1 }$ </td><td>16</td><td>7.84</td><td>7.98</td><td>7.16</td><td>8.07</td><td>6.92</td><td>4.66</td><td>3.55</td></tr><tr><td>32</td><td>9.02</td><td>8.87</td><td>7.88</td><td>8.60</td><td>7.38</td><td>4.89</td><td>3.59</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 3 } – R _ { 3 }$ </td><td>16</td><td>8.46</td><td>9.00</td><td>7.96</td><td>8.48</td><td>7.80</td><td>5.34</td><td>3.92</td></tr><tr><td>32</td><td>10.64</td><td>10.50</td><td>9.03</td><td>9.93</td><td>8.16</td><td>5.67</td><td>4.05</td></tr></table>

Evaluation protocol. Every method generates at most 2,048 tokens, stopping at EOS or at the onset of a repetition loop (Appendix E). Tables 1 and 2 measure the acceptance length offline, and Table 3 measures serving at batch size 1 on NVIDIA GB200 GPUs with CUDA graphs. On a miss, JetSpec++ drafts a fallback block from the verified context in one pass (Section 4.1). We report both greedy decoding and temperature 1 with rejection sampling (Appendix F.3). Speedup is measured in tokens per second (tok/s) relative to AR on one GPU of the same type. In Table 3, SSD and JetSpec++ place their draft models on a second GPU (Section 4.1). The acceptance length τ is the mean number of tokens committed per verification round.

## 5.2 RESULTS

Table 1 reports the acceptance length. At both temperatures and both budgets, JetSpec++ commits more tokens per round than every baseline on all seven benchmarks. At temperature 0 and budget $1 6 , J _ { 1 } – R _ { 1 }$ averages 7.50 against 6.89 for DSpark, the strongest baseline, and ${ \bar { J } } _ { 3 ^ { - } } R _ { 3 }$ raises it to 8.94. Foresight drafting costs little acceptance length. At each of $J _ { 0 }$ to $J _ { 3 }$ without context refresh, the draft model keeps 95 to 98% of the τ it reaches from the target’s true states (Table 6).

Although every draft model is trained on blocks of 16, JetSpec++ extrapolates to blocks of 32. At budget 32, $J _ { 1 } { - } R _ { 1 }$ gains τ on every benchmark at both temperatures, whereas DFlash, DSpark and Domino lose it. On MATH-500 at temperature 0, for example, τ rises from 9.63 to 11.45 for $J _ { 1 } { - } R _ { 1 }$ and falls from 9.01 to 8.66 for DSpark. With causal attention inside the block, the 16 added positions leave the draft of the first 16 unchanged, and JetSpec, also causal, likewise gains τ at temperature 0. DFlash and Domino attend bidirectionally within the block, so the 16 added positions also change the draft of the first 16.

## 5.3 GENERALIZATION TO A LARGER TARGET

Table 2 compares JetSpec++ with DSpark on Qwen3.8-27B, a hybrid model in which 48 of the 64 layers are Gated DeltaNet (Yang et al., 2025b) layers and 16 are full-attention layers. The adapter and the decoder adapt to the hybrid layers (Appendices A and B.1). The adapter has 12 layers, as for Qwen3-8B (Table 7), although the 27B target has 64 layers against 36. With this adapter, JetSpec++ at $J _ { 1 } { - } R _ { 1 }$ leads DSpark on all seven benchmarks at temperature 0 and budget 16, with 8.71 and 9.53 on GSM8K and MATH-500 against 8.02 and 8.94. As on Qwen3-8B, the $\tau$ of JetSpec++ rises at budget 32 on all seven benchmarks, whereas that of DSpark falls.

## 5.4 SYSTEM PERFORMANCE

Table 3 measures how much drafting latency JetSpec++ removes from each draft-verification round and how that translates into end-to-end speedup. We evaluate all methods under the same CUDAgraph serving setup (Appendix E), closer to production inference. At $J _ { 1 }$ with one context refresh and anchor replacement, JetSpec++ is 1.54× faster than DSpark on GSM8K and 1.51× faster on MATH-500. It commits slightly more tokens per round (7.78 vs. 7.34 and 9.41 vs. 8.98), but most of the gain comes from hiding drafting latency. For an 8B target, graphed verification takes 6.1 to 6.3 ms, and the draft process runs one refinement pass and one context refresh within it. DSpark adds 3.76 and 3.80 ms of drafting per round, whereas JetSpec++ exposes only 0.61 and 0.66 ms, with the realized outcome already pre-drafted in 92.9% and 91.7% of rounds. SSD, designed for larger targets, reaches only 1.52× and 1.55×: its autoregressive draft model outlasts verification by 8.51 and 8.55 ms per round, and each round commits at most five tokens $( \tau$ of 4.06 and 4.20). As target size grows, verification becomes more expensive relative to drafting, creating a larger latency window for additional refinement passes and context refreshes. On Qwen3.8-27B, graphed verification takes 27.4 to 27.8 ms, and the draft process runs three refinement passes and three context refreshes within it. JetSpec++ then exposes 0.71 and 0.77 ms per round against 4.46 and 4.52 ms for DSpark, and runs 1.37× and 1.31× faster than DSpark on GSM8K and MATH-500.

Table 3: Throughput, acceptance length and round time at temperature 0 and budget 16 (5 for SSD, Appendix E), on NVIDIA GB200 GPUs with CUDA graphs (Section 5.1). JetSpec++ runs $J _ { 1 }$ with one context refresh and anchor replacement for Qwen3-8B and $J _ { 3 }$ with three of each for Qwen3.8- 27B. Speedup is over AR. Hit is the share $p _ { \mathrm { h i t } }$ of rounds whose outcome was pre-drafted, and $\tau _ { \mathrm { h i t } }$ and τ are the mean numbers of tokens committed on hit and miss rounds, all three over the rounds after the first of each request, which has no cache. Round and verify are the mean times per round in ms, and exposed is their difference.
<table><tr><td>Model</td><td>Benchmark Method</td><td></td><td>GPUs</td><td>tok/s (speedup)</td><td>T</td><td>hit</td><td>Thit</td><td>Tmiss</td><td>round</td><td>verify</td><td>exposed</td></tr><tr><td rowspan="9">Qwen3-8B</td><td rowspan="5">GSM8K</td><td>AR</td><td>1</td><td>182.3 (1.00×)</td><td>1.00</td><td>N/A</td><td>N/A</td><td>N/A</td><td>5.49</td><td>5.45</td><td>0.04</td></tr><tr><td>DSpark</td><td>1</td><td>744.3 (4.08×)</td><td>7.34</td><td>N/A</td><td>N/A</td><td>N/A</td><td>9.92</td><td>6.16</td><td>3.76</td></tr><tr><td>DFlash</td><td>1</td><td>620.1 (3.40×)</td><td>5.87</td><td>N/A</td><td>N/A</td><td>N/A</td><td>9.52</td><td>6.11</td><td>3.42</td></tr><tr><td>SSD</td><td>2</td><td>276.5 (1.52×)</td><td>4.06</td><td>95.7%</td><td>4.12</td><td>3.23</td><td>14.75</td><td>6.25</td><td>8.51</td></tr><tr><td>JetSpec++</td><td>2</td><td>1148.8 (6.30×)</td><td>7.78</td><td>92.9%</td><td>8.20</td><td>3.45</td><td>6.84</td><td>6.23</td><td>0.61</td></tr><tr><td rowspan="5">MATH-500</td><td>AR</td><td>1</td><td>182.7 (1.00×)</td><td>1.00</td><td>N/A</td><td>N/A</td><td>N/A</td><td>5.47</td><td>5.44</td><td>0.04</td></tr><tr><td>DSpark</td><td>1</td><td>902.1 (4.94×)</td><td>8.98</td><td>N/A</td><td>N/A</td><td>N/A</td><td>9.95</td><td>6.16</td><td>3.80</td></tr><tr><td>DFlash</td><td>1</td><td>795.5 (4.35×)</td><td>7.65</td><td>N/A</td><td>N/A</td><td>N/A</td><td>9.63</td><td>6.16</td><td>3.47</td></tr><tr><td>SSD</td><td>2</td><td>282.5 (1.55×)</td><td>4.20</td><td>85.8%</td><td>4.32</td><td>3.47</td><td>14.88</td><td>6.33</td><td>8.55</td></tr><tr><td>JetSpec++</td><td>2</td><td>1361.1 (7.45×)</td><td>9.41</td><td>91.7%</td><td>9.83</td><td>4.87</td><td>6.92</td><td>6.26</td><td>0.66</td></tr><tr><td rowspan="5">Qwen3.8-27B</td><td rowspan="3">GSM8K</td><td>AR</td><td>1</td><td>60.3 (1.00×)</td><td>1.00</td><td>N/A</td><td>N/A</td><td>N/A</td><td>16.58</td><td>16.54</td><td>0.04</td></tr><tr><td>DSpark</td><td>1</td><td>249.6 (4.14×)</td><td>7.93</td><td>N/A</td><td>N/A</td><td>N/A</td><td>31.88</td><td>27.42</td><td>4.46</td></tr><tr><td>JetŠpec++</td><td>2</td><td>342.4 (5.68×)</td><td>9.67</td><td>89.0%</td><td>10.46</td><td>3.97</td><td>28.47</td><td>27.76</td><td>0.71</td></tr><tr><td>AR</td><td>1</td><td>59.3 (1.00×)</td><td>1.00</td><td>N/A</td><td>N/A</td><td>N/A</td><td>16.88</td><td>16.83</td><td></td><td>0.04</td></tr><tr><td rowspan="3">MATH-500</td><td>DSpark</td><td>1</td><td>279.8 (4.72×)</td><td>8.99</td><td>N/A</td><td>N/A</td><td>N/A</td><td>32.09</td><td>27.57</td><td>4.52</td></tr><tr><td>JetŠpec++</td><td>2</td><td>367.2 (6.20×)</td><td>10.45</td><td>90.0%</td><td>11.10</td><td>4.65</td><td>28.47</td><td>27.70</td><td>0.77</td></tr></table>

## 5.5 ABLATION STUDIES

Table 4: Self-refinement of the draft model. Acceptance length τ over refinement passes at temperature 0 for the two-epoch ablation draft model, trained with token or mask inputs, and on blocks with a clean prefix drawn by the prefix-length schedule of Appendix E or on fully noised blocks only. The target’s true states and no context refresh, 164 prompts per benchmark, 256 tokens. Paired per prompt, every other row trails ours at $J _ { 1 } , J _ { 2 }$ and $J _ { 3 }$ by at least 8 standard errors.
<table><tr><td></td><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Inputs</td><td>Training blocks</td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td></tr><tr><td rowspan="2">token (ours)</td><td>clean prefix (ours)</td><td>4.34</td><td>4.87</td><td>5.01</td><td>5.05</td><td>5.91</td><td>6.75</td><td>6.99</td><td>7.08</td></tr><tr><td>fully noised</td><td>4.36</td><td>4.37</td><td>4.37</td><td>4.37</td><td>6.02</td><td>6.01</td><td>6.02</td><td>6.01</td></tr><tr><td rowspan="2">mask</td><td>clean prefix</td><td>4.37</td><td>4.57</td><td>4.63</td><td>4.67</td><td>5.93</td><td>6.29</td><td>6.47</td><td>6.56</td></tr><tr><td>fully noised</td><td>4.34</td><td>1.23</td><td>1.26</td><td>1.23</td><td>5.99</td><td>1.28</td><td>1.32</td><td>1.29</td></tr></table>

Table 4 isolates the draft model’s side of JetSpec++ against controls, and Table 5 isolates context refresh and anchor replacement, each under the condition in its caption. Appendix A and Appendix B evaluate the adapter and the decoder.

Table 5: Gain of context refresh and anchor replacement. Acceptance length τ at temperature 0 and budget 16 for the draft model and in the setting of Table 1, with J passes after $J _ { 0 }$ and R context refreshes per round. A context refresh alone re-predicts the target states of the draft tokens and keeps every draft token. With anchor replacement, it writes the decoder’s token aˆ of Eq. 13 into the block.
<table><tr><td>J</td><td>R</td><td>Anchor repl.</td><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>HumanEval</td><td>MBPP</td><td>LCB</td><td></td><td>MT-Bench</td></tr><tr><td></td><td>0 0</td><td>一</td><td></td><td>5.54</td><td>7.33</td><td>6.62</td><td>5.07</td><td>4.77</td><td>5.60</td><td>3.15</td></tr><tr><td></td><td>1 0</td><td>一</td><td></td><td>6.43</td><td>8.22</td><td>7.46</td><td>5.80</td><td>5.43</td><td>6.34</td><td>3.46</td></tr><tr><td></td><td>2 0</td><td>一</td><td>6.72</td><td>8.52</td><td>7.77</td><td>6.04</td><td>5.65</td><td>6.66</td><td></td><td>3.54</td></tr><tr><td>3</td><td>0</td><td>一</td><td></td><td>6.81</td><td>8.59</td><td>7.82</td><td>6.12</td><td>5.72</td><td>6.71</td><td>3.57</td></tr><tr><td></td><td>1</td><td>no</td><td></td><td>6.86</td><td>8.45</td><td>7.86</td><td>6.15</td><td>5.71</td><td>6.76</td><td>3.65</td></tr><tr><td>1</td><td>1</td><td>yes</td><td></td><td>7.98</td><td>9.63</td><td>8.97</td><td>7.23</td><td>6.73</td><td>7.59</td><td>4.38</td></tr><tr><td>2</td><td>2</td><td>no</td><td></td><td>7.48</td><td>9.08</td><td>8.55</td><td>6.59</td><td>6.14</td><td>7.06</td><td>3.81</td></tr><tr><td>2</td><td>2</td><td>yes</td><td></td><td>9.23</td><td>10.68</td><td>10.17</td><td>8.33</td><td>7.79</td><td>8.44</td><td>4.98</td></tr><tr><td></td><td>3 3</td><td>no</td><td></td><td>7.65</td><td>9.26</td><td>8.56</td><td>6.78</td><td>6.31</td><td>7.32</td><td>3.88</td></tr><tr><td></td><td>3 3</td><td>yes</td><td>9.78</td><td></td><td>11.04</td><td>10.49</td><td>8.79</td><td>8.24</td><td>8.95</td><td>5.29</td></tr></table>

Refinement. Table 4 tests whether the draft model can refine its own draft. A draft model trained on token inputs with every block fully noised drafts as well at $J _ { 0 }$ but does not refine, so refinement is learned from the blocks with a clean prefix, which the prefix-length schedule of Appendix E supplies. With this schedule but mask inputs, a draft model keeps only 42% and 54% of the gain from $J _ { 0 }$ to $J _ { 3 } .$ . At $J _ { 0 }$ the other three draft models differ from ours by at most 0.11 tokens, so the inputs and the schedule decide how much a draft model refines. The final draft models show the same split at temperature 0 (Table 12). The JetSpec draft model, which is trained on mask inputs and fully noised blocks, falls from 5.86 at $J _ { 0 }$ to 2.39 at $J _ { 1 }$ on GSM8K and from 7.51 to 2.57 on MATH-500. The JetSpec++ draft model instead rises from 5.70 at $J _ { 0 }$ to 6.97 at $J _ { 3 }$ on GSM8K and from 7.52 to 8.98 on MATH-500.

Context Refresh and Anchor Replacement. Table 5 compares J passes alone with the same passes and a context refresh before each, with or without anchor replacement. Refinement alone raises τ by 0.42 to 1.27 tokens from $J _ { 0 }$ to $J _ { 3 } ,$ , and context refreshes with anchor replacement add 0.92 to 1.55 tokens at $J _ { 1 } { - } R _ { 1 }$ and 1.72 to 2.97 at $J _ { 3 } – R _ { 3 }$ over the same passes alone. Context refresh alone adds at most 0.84 tokens, at most 34% of the gain with anchor replacement, so most of that gain comes from anchor replacement, not from the next pass reading the re-predicted states. Appendix D traces one context refresh and anchor replacement (Figure 4) and measures it without pipelining (Table 16).

Training loss and attention. Forward KL leads SFT and reverse KL in every cell, by up to 0.27 and 2.72 tokens (Table 14). Causal attention within the block matches bidirectional attention at $J _ { 0 }$ and leads it at $J _ { 3 } .$ , by 0.15 tokens on MATH-500 (Table 15). Both tables are in Appendix C.

Adapter. At J<sub>0</sub>, the adapter’s predicted states recover 95% of the gap between copying the last verified state and the true states on both GSM8K and MATH-500 (Appendix A).

## 5.6 LIMITATIONS

Our serving integration can be further improved through systems optimization. In particular, fused kernels and optimized communication could reduce drafting overhead and enable additional refinement passes and context refreshes to further improve throughput.

## 6 CONCLUSION

We presented JetSpec++, which drafts the next block while the current one is verified. Foresight Drafting predicts the target states and the verification outcome before verification returns. We train the causal parallel draft model on blocks with a clean prefix and a noisy suffix, so that self-refinement can spend the freed draft-side time on Jacobi passes and context refreshes. On GB200 GPUs, Jet-Spec++ decodes up to 7.45× faster than AR decoding and 1.54× faster than DSpark.

## REFERENCES

Zihao An, Taichi Liu, Ziqiong Liu, Dong Li, Ruofeng Liu, and Emad Barsoum. PARD-2: Target-aligned parallel draft model for dual-mode speculative decoding. arXiv preprint arXiv:2605.08632, 2026. URL http://arxiv.org/abs/2605.08632v1.

Anastasios N. Angelopoulos and Stephen Bates. Conformal prediction: A gentle introduction. Foundations and Trends® in Machine Learning, 16(4):494–591, 2023. doi: 10.1561/2200000101.

Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han, Subham Sekhar Sahoo, and Volodymyr Kuleshov. Block diffusion: Interpolating between autoregressive and diffusion language models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=tyEyYT267x.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, and Charles Sutton. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. URL http://arxiv.org/abs/ 2108.07732v1.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´ c, and Martin Vechev. MathArena:´ Evaluating LLMs on uncontaminated math competitions. In Advances in Neural Information Processing Systems 38, Datasets and Benchmarks Track, 2025. URL https://arxiv.org/ abs/2505.23281.

Nikhil Bhendawade, Kumari Nishu, Arnav Kundu, Chris Bartels, Minsik Cho, and Irina Belousova. Mirror speculative decoding: Breaking the serial barrier in LLM inference. arXiv preprint arXiv:2510.13161, 2025. URL http://arxiv.org/abs/2510.13161v2.

Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 5209–5235, 2024. URL https://proceedings.mlr. press/v235/cai24b.html.

Sahil Chaudhary. Code alpaca: An instruction-following LLaMA model trained on code generation instructions. GitHub repository, 2023. URL https://github.com/sahil280114/ codealpaca. Accessed 2026-09-22.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023. URL http://arxiv.org/abs/2302.01318v1.

Jian Chen, Yesheng Liang, and Zhijian Liu. DFlash: Block diffusion for flash speculative decoding. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=Oz335dV48X.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared´ Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. URL http://arxiv.org/abs/2107.03374v2.

Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable and robust speculative decoding. In Advances in Neural Information Processing Systems, volume 37, pp. 129531–129563, 2024. doi: 10.52202/ 079017-4116. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/ea1f5f0878d43ff4fb8bf64ef4a2326c-Paper-Conference.pdf.

Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong, Yi Qian, Jiaqi Zhu, Shirong Ma, Xiaokang Zhang, Jiasheng Ye, Qinyu Chen, Chengqi Deng, Jiping Yu, Damai Dai, Zhengyan Zhang, Yixuan Wei, Yixuan Tan, Wenkai Yang, Runxin Xu, Yu Wu, Zhean Xu, Xuanyu Wang, Muyang Chen, Rui Tian, Xiao Bi, Zhewen Hao, Shaoyuan Chen, Huanqi Cao, Wentao Zhang, Anyi Xu, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. DSpark: Confidence-scheduled speculative decoding with semi-autoregressive generation. arXiv preprint arXiv:2607.05147, 2026. URL http://arxiv.org/abs/2607.05147v1.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. URL http://arxiv.org/abs/2110.14168v2.

Ning Ding, Yulin Chen, Bokai Xu, Yujia Qin, Shengding Hu, Zhiyuan Liu, Maosong Sun, and Bowen Zhou. Enhancing chat language models by scaling high-quality instructional conversations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3029–3051, 2023. doi: 10.18653/v1/2023.emnlp-main.183. URL https: //aclanthology.org/2023.emnlp-main.183/.

Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of LLM inference using lookahead decoding. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 14060–14079, 2024. URL https://proceedings.mlr.press/v235/fu24a.html.

Raghavv Goel, Mukul Gagrani, Wonseok Jeon, Junyoung Park, Mingu Lee, and Christopher Lott. Direct alignment of draft model for speculative decoding with chat-fine-tuned LLMs. arXiv preprint arXiv:2403.00858, 2024. URL http://arxiv.org/abs/2403.00858v4.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021. URL https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/ hash/be83ab3ecd0db773eb2dc1b0a17836a1-Abstract-round2.html.

Lanxiang Hu, Zhaoxiang Feng, Yulun Wu, Haoran Yuan, Yujie Zhao, Yu-Yang Qian, Bojun Wang, Peng Zhao, Daxin Jiang, Yibo Zhu, Tajana Rosing, and Hao Zhang. JetSpec: Breaking the scaling ceiling of speculative decoding with parallel tree drafting. arXiv preprint arXiv:2606.18394, 2026a. URL http://arxiv.org/abs/2606.18394v3.

Lanxiang Hu, Siqi Kou, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, and Hao Zhang. Fast and accurate causal parallel decoding using Jacobi forcing. In Forty-third International Conference on Machine Learning, 2026b. URL https://openreview.net/ forum?id=ORy2hyPd4G.

Yuxuan Hu, Ke Wang, Xiaokang Zhang, Fanjin Zhang, Cuiping Li, Hong Chen, and Jing Zhang. SAM decoding: Speculative decoding via suffix automaton. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12187– 12204, 2025. doi: 10.18653/v1/2025.acl-long.595. URL https://aclanthology.org/ 2025.acl-long.595/.

Jianuo Huang, Yaojie Zhang, Qituan Zhang, Hao Lin, Hanlin Xu, and Linfeng Zhang. Domino: Decoupling causal modeling from autoregressive drafting in speculative decoding. arXiv preprint arXiv:2605.29707, 2026. URL http://arxiv.org/abs/2605.29707v1.

Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. LiveCodeBench: Holistic and contamination free evaluation of large language models for code. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ file/94074dd5a072d28ff75a76dabed43767-Paper-Conference.pdf.

Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. CLLMs: Consistency large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 25426–25440, 2024. URL https://openreview.net/forum?id=8uzBOVmh8H.

Tanishq Kumar, Tri Dao, and Avner May. Speculative speculative decoding. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 1b96f01343ff10150e6719eb163e1536-Paper-Conference.pdf.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from Transformers via speculative decoding. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 19274–19286, 2023. URL https://proceedings.mlr.press/v202/leviathan23a.html.

Tianyi Li, Yaxin Luo, Xinyi Shang, and Zhiqiang Shen. DARTree: Speculative diffusion decoding with autoregressive draft trees. arXiv preprint arXiv:2608.13524, 2026. URL http://arxiv. org/abs/2608.13524v1.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE: Speculative sampling requires rethinking feature uncertainty. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pp. 28935–28948, 2024a. URL https://proceedings.mlr.press/v235/li24bt.html.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-2: Faster inference of language models with dynamic draft trees. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 7421–7432, 2024b. doi: 10.18653/v1/2024. emnlp-main.422. URL https://aclanthology.org/2024.emnlp-main.422/.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. EAGLE-3: Scaling up inference acceleration of large language models via training-time test. In Advances in Neural Information Processing Systems 38, pp. 151568–151587, 2025. doi: 10.52202/085713-4562.

Fuliang Liu, Xue Li, Kun Qian, Zhibin Wang, Wanchun Dou, Wenyuan Yu, and Chen Tian. When parallel drafter meets parallel speculative decoding. arXiv preprint arXiv:2609.27396, 2026. doi: 10.48550/arxiv.2609.27396. URL https://arxiv.org/abs/2609.27396.

Tianyu Liu, Yun Li, Qitan Lv, Kai Liu, Jianchen Zhu, Winston Hu, and Xiao Sun. PEARL: Parallel speculative decoding with adaptive draft length. In International Conference on Learning Representations, volume 2025, pp. 1085–1104, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 03b1043052700b1a471996b0baf309d4-Paper-Conference.pdf.

Xiaoxuan Liu, Lanxiang Hu, Peter Bailis, Alvin Cheung, Zhijie Deng, Ion Stoica, and Hao Zhang. Online speculative decoding. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 31131–31146, 2024. URL https://proceedings.mlr.press/v235/liu24y.html.

Bradley McDanel. AMUSD: Asynchronous multi-device speculative decoding for LLM acceleration. In 2025 IEEE International Symposium on Circuits and Systems (ISCAS), pp. 1–5, 2025. doi: 10.1109/iscas56072.2025.11043575.

Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. SpecInfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, pp. 932–949, 2024. doi: 10.1145/3620666.3651335.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. In Advances in Neural Information Processing Systems, volume 38, pp. 50608–50646, 2025. doi: 10.52202/ 085713-1689. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/48b383b24230e0e6e649d9c98dae4d8c-Paper-Conference.pdf.

NVIDIA. NVIDIA Nemotron post-training dataset v2. Hugging Face dataset, 2025. URL https://huggingface.co/datasets/nvidia/ Nemotron-Post-Training-Dataset-v2. Accessed 2026-09-22.

Gabriele Oliaro, Zhihao Jia, Daniel Campos, and Aurick Qiao. SuffixDecoding: Extreme speculative decoding for emerging AI applications. In Advances in Neural Information Processing Systems, volume 38, pp. 126326–126354, 2025. doi: 10.52202/ 085713-4211. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/b7aea253ab34a773967f1e4cdea9e4fb-Paper-Conference.pdf.

Yu-Yang Qian, Junda Su, Lanxiang Hu, Peiyuan Zhang, Zhijie Deng, Peng Zhao, and Hao Zhang. d3LLM: Ultra-fast diffusion LLM using pseudo-trajectory distillation. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum? id=rzBAQT2Fkg.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026. URL https:// qwen.ai/blog?id=qwen3.8. Model card of Qwen3.8-27B: https://huggingface. co/Qwen/Qwen3.8-27B.

Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. Simple and effective masked diffusion language models. In Advances in Neural Information Processing Systems, volume 37, pp. 130136–130184, 2024. doi: 10.52202/079017-4135. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ eb0b13cc515724ab8015bc978fdde0ad-Paper-Conference.pdf.

Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodola. Accelerating transformer inference for translation via parallel\` decoding. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12336–12355, 2023. doi: 10.18653/v1/2023.acl-long.689. URL https://aclanthology.org/2023.acl-long.689/.

Apoorv Saxena. Prompt lookup decoding. GitHub repository, November 2023. URL https: //github.com/apoorvumang/prompt-lookup-decoding. Accessed 2026-09-22.

Yuhao Shen, Junyi Shen, Quan Kong, Tianyu Liu, Yao Lu, and Cong Wang. SpecBranch: Speculative decoding via hybrid drafting and rollback-aware branch parallelism. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 67089958e98b243d5cc1881ad60418b8-Paper-Conference.pdf.

Shwetha Somasundaram, Anirudh Phukan, and Apoorv Saxena. PLD+: Accelerating LLM inference by leveraging language model artifacts. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 6090–6104, 2025. doi: 10.18653/v1/2025.findings-naacl.338. URL https://aclanthology.org/2025.findings-naacl.338/.

Yang Song, Chenlin Meng, Renjie Liao, and Stefano Ermon. Accelerating feedforward computation via parallel nonlinear equation solving. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 9791–9800, 2021. URL https://proceedings.mlr.press/v139/song21a.html.

Yeshwanth Venkatesha, Souvik Kundu, and Priyadarshini Panda. Fast and cost-effective speculative edge-cloud decoding with early exits. Transactions on Machine Learning Research, 2025. URL https://openreview.net/forum?id=PTIUjARnbc.

Xu Wang, Chenkai Xu, Yijie Jin, Jiachun Jin, Hao Zhang, and Zhijie Deng. Diffusion LLMs can do Faster-Than-AR inference via discrete diffusion forcing. In International Conference on Learning Representations, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 04bb76a98d9f32226b3beba7bd26a51f-Paper-Conference.pdf.

Zheng Wang, Davis Wertheimer, Yu Chin Fabian Lim, Mudhakar Srivatsa, Raghu K. Ganti, Minjia Zhang, and Naigang Wang. xPress: Parallel refinement for diffusion drafters in speculative decoding. arXiv preprint arXiv:2608.02438, 2026b. URL http://arxiv.org/abs/2608. 02438v1.

Zheng Wang, Zhifan Ye, Qi Cheng, Yonggan Fu, Ziyan Wang, Feng Zhu, Haozhe Zhao, Jan Kautz, Pavlo Molchanov, Humphrey Shi, and Minjia Zhang. PRESTO: Prefix-aligned tree drafting for diffusion speculative decoding. arXiv preprint arXiv:2607.22634, 2026c. URL http://arxiv.org/abs/2607.22634v1.

Chengyue Wu, Hao Zhang, Shuchen Xue, Shizhe Diao, Yonggan Fu, Zhijian Liu, Pavlo Molchanov, Ping Luo, Song Han, and Enze Xie. Fast-dLLM v2: Efficient block-diffusion LLM. In International Conference on Learning Representations, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ d0865cbe51d35ec322f9af9db7806fc7-Paper-Conference.pdf.

Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dLLM: Training-free acceleration of diffusion LLM by enabling KV cache and parallel decoding. In International Conference on Learning Representations, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/5d8d4e6061c3ba96c240b7fa1ae3471d-Paper-Conference.pdf.

Zilin Xiao, Hongming Zhang, Tao Ge, Siru Ouyang, Vicente Ordonez, and Dong Yu. ParallelSpec: Parallel drafter for efficient speculative decoding. arXiv preprint arXiv:2410.05589, 2024. URL http://arxiv.org/abs/2410.05589v1.

An Yang et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a. URL http: //arxiv.org/abs/2505.09388v1.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving Mamba2 with delta rule. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://proceedings.iclr.cc/paper\_files/paper/2025/ hash/4904fad153f6434a7bcf04465d4be2cc-Abstract-Conference.html.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7B: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025. doi: 10.48550/arXiv.2508.15487. URL https://arxiv.org/abs/2508.15487.

Yijiong Yu, Huazheng Wang, Shuai Yuan, Ruilong Ren, and Ji Pei. Speculative pipeline decoding: Higher-accuracy drafting with hidden latency via pipeline parallelism. arXiv preprint arXiv:2605.30852, 2026. URL http://arxiv.org/abs/2605.30852v3.

Haohui Zhang, Keyu Chen, Haocheng Sun, Weibo Gu, Ruizhi Qiao, Xing Sun, and Bo Jiang. Draft in parallel, condition through depth: Adjacent causal injection for speculative decoding. arXiv preprint arXiv:2609.36173, 2026a. doi: 10.48550/arxiv.2609.36173. URL https://arxiv. org/abs/2609.36173.

Lefan Zhang, Xiaodan Wang, Yanhua Huang, and Ruiwen Xu. Learning harmonized representations for speculative sampling. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 575286a73f238b6516ce0467d67eadb2-Paper-Conference.pdf.

Liyuan Zhang, Jiarui Zhang, Jinwei Yao, Ran Yan, Yuchen Yang, Jiahao Zhang, Tongkai Yang, Yi Wu, and Binhang Yuan. D<sup>2</sup>SD: Accelerating speculative decoding with dual diffusion draft models. arXiv preprint arXiv:2606.04446, 2026b. URL http://arxiv.org/abs/2606. 04446v1.

Ziyi Zhang, Ziheng Jiang, Chengquan Jiang, Menghan Yu, Size Zheng, Haibin Lin, Xin Liu, and Henry Hoffmann. SwiftSpec: Disaggregated speculative decoding and fused kernels for lowlatency LLM inference. In Proceedings of the 31st ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, pp. 2197–2211, 2026c. doi: 10.1145/3779212.3790246.

Hong-Kai Zheng and Piji Li. DeLS-Spec: Decoupled long-short contexts for parallel speculative drafting. arXiv preprint arXiv:2607.07409, 2026. URL http://arxiv.org/abs/2607. 07409v1.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623, 2023. doi: 10.52202/075280-2020. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ 91f18a1287b398d378ef22505bf41832-Paper-Datasets\_and\_Benchmarks. pdf.

Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-Franc¸ois Kagy, and Rishabh Agarwal. DistillSpec: Improving speculative decoding via knowledge distillation. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 8766fbc68e1ed1cdef712ce273e0a363-Paper-Conference.pdf.

## A ADAPTER DESIGN AND ABLATIONS

The predicted states of the adapter (Section 4.2) recover most of the acceptance length that the target’s true states give (Sections 5.2 and 5.5). In this appendix, we describe the adapter’s architecture and training, measure this recovery, and ablate the number of adapter layers and the position weight in the loss.

Architecture. The adapter has one module $A _ { \ell }$ of Eq. 5 for each target layer ℓ that the draft model reads, and each module is a short stack of transformer layers. Module $A _ { \ell }$ takes the block’s states at the next lower layer $\ell ^ { - }$ that the draft model reads and predicts the block’s states at ℓ. The lowest module takes the block’s token embeddings instead, so the modules form one chain from the tokens to the deepest layer that the draft model reads. In each module, the queries come from the block positions, and the keys and values come from the true states $h _ { < n } ^ { \ell }$ of the verified context and the block’s states $\hat { h } ^ { \ell ^ { - } }$ at earlier block positions. We initialize each module with copies of target layers between $\ell ^ { - }$ and ℓ, ending with layer ℓ. The adapter reuses the target’s token embedding and keeps it frozen. We train all other weights of the adapter, and it adds no projection or head of its own. The adapter has 2.32B trainable parameters for Qwen3-8B and 4.47B for Qwen3.8-27B.

Target layers. The draft model reads five target layers. Layer indices start at 0. The modules for the three lowest layers each copy only their own layer, and the two deepest modules copy four and five layers, twelve layers in all.

• Qwen3-8B (36 layers): the draft model reads layers 1, 9, 17, 25 and 33, spaced uniformly from the second layer to the third-to-last as in DFlash (Chen et al., 2026).

• Qwen3.8-27B (64 layers): the draft model reads layers 5, 19, 33, 47 and 61. Each adapter layer for this target takes its MLP from the copied layer and its attention from the nearest full-attention layer at or below it.

Training objective. We train the adapter against the frozen target, which computes the true states online, on blocks of $\gamma + 1 = 1 6$ positions. Let $ { \mathbf { h } } _ { n + j }$ be the concatenation of the target’s true states $h _ { n + j } ^ { \ell }$ at the layers that the draft model reads, and $\dot { \mathbf { h } } _ { n + j }$ the adapter’s prediction of it. For a block whose anchor is at position n, the loss is

$$
\mathcal { L } _ { \mathrm { a d a p l e r } } ~ = ~ \sum _ { j = 0 } ^ { \gamma } \frac { \omega _ { j } ^ { \kappa } } { \sum _ { j ^ { \prime } = 0 } ^ { \gamma } \omega _ { j ^ { \prime } } ^ { \kappa } } \Big [ \mathrm { S m o o t h L 1 } \big ( \hat { \mathbf { h } } _ { n + j } , \mathbf { h } _ { n + j } \big ) + 1 - \cos \big ( \hat { \mathbf { h } } _ { n + j } , \mathbf { h } _ { n + j } \big ) \Big ] ,\tag{10}
$$

where $\omega _ { j }$ is the probability that verification reaches block position $j ,$ taken from served runs. The exponent κ sets how the positions are weighted: $\kappa = 0$ weights every position equally, and $\kappa = 1$ weights each position by $\omega _ { j }$ . We use $\kappa = 1 / 2$

Training data and optimization. We train on the 800k training sequences of the draft model (Appendix E), less 24k held out, with each sequence capped at 3,072 tokens. We use AdamW with a peak learning rate of $\mathrm { 3 \times 1 0 ^ { - 4 } }$ (linear warmup over the first 4% of steps, then cosine decay), gradient clipping at norm 1.0 and no weight decay. For Qwen3-8B, we train for six epochs at a global batch of 4 to 8 sequences, in 219 GPU-hours. For Qwen3.8-27B, we train for two epochs at a global batch of 8, with the AdamW $\beta _ { 2 }$ set to 0.95, in 172 GPU-hours.

Recovery of the true-state acceptance length. Table 6 reports τ of the final draft model with stale, predicted and true states. At each of $J _ { 0 }$ to ${ \bar { J _ { 3 } } } ,$ , τ from the predicted states is 95 to 98% of τ from the true states. The predicted states close 95% of the gap in τ between the stale and the true states at $J _ { 0 }$ on GSM8K and MATH-500, and 89 to 95% over all 28 cells.

Ablation setup. We ablate the number of adapter layers and the exponent κ of Eq. 10 for Qwen3- 8B. To measure τ, every adapter feeds the same frozen draft model, the ablation draft model of Appendix E. We train the adapters on the mathematics slice of the 800k set, apart from a held-out part. At each layer ℓ that the draft model reads, the error $\epsilon _ { \ell }$ of an adapter is the relative error of its predicted states on 2,048 blocks drawn from the held-out sequences,

$$
\epsilon _ { \ell } = \frac { \sum _ { n } \sum _ { j } \omega _ { j } ^ { 2 } \| \hat { h } _ { n + j } ^ { \ell } - h _ { n + j } ^ { \ell } \| _ { 2 } / \| h _ { n + j } ^ { \ell } \| _ { 2 } } { \sum _ { n } \sum _ { j } \omega _ { j } ^ { 2 } } ,\tag{11}
$$

Table 6: Acceptance length τ of the final draft model for Qwen3-8B at budget 16, temperature 0 and no context refresh, with three sources of the target states at unverified positions. Stale states copy the last verified state into every new position, the final adapter predicts the states, and true states exist only after verification. Every round drafts only the block of the verified outcome, so no round misses. Prompts as in Table 1 (first turn only for MT-Bench), with at most 2,048 generated tokens.
<table><tr><td></td><td>J States</td><td>GSM8K</td><td>MATH-500</td><td>AIME25</td><td>HumanEval</td><td>MBPP</td><td>LCB</td><td>MT-Bench</td></tr><tr><td rowspan="3">0</td><td>stale</td><td>2.89</td><td>3.62</td><td>3.47</td><td>2.70</td><td>2.58</td><td>3.17</td><td>1.93</td></tr><tr><td>predicted</td><td>5.56</td><td>7.31</td><td>6.89</td><td>4.98</td><td>4.68</td><td>5.93</td><td>2.98</td></tr><tr><td>true</td><td>5.70</td><td>7.52</td><td>7.14</td><td>5.09</td><td>4.81</td><td>6.11</td><td>3.09</td></tr><tr><td rowspan="3">1</td><td>stale</td><td>3.36</td><td>4.17</td><td>4.00</td><td>3.13</td><td>2.97</td><td>3.64</td><td>2.11</td></tr><tr><td>predicted</td><td>6.41</td><td>8.26</td><td>7.79</td><td>5.67</td><td>5.33</td><td>6.65</td><td>3.28</td></tr><tr><td>true</td><td>6.59</td><td>8.55</td><td>8.16</td><td>5.85</td><td>5.49</td><td>6.89</td><td>3.40</td></tr><tr><td rowspan="3">2</td><td>stale</td><td>3.55</td><td>4.35</td><td>4.17</td><td>3.27</td><td>3.10</td><td>3.79</td><td>2.17</td></tr><tr><td>predicted</td><td>6.67</td><td>8.52</td><td>8.07</td><td>5.90</td><td>5.59</td><td>6.84</td><td>3.34</td></tr><tr><td>true</td><td>6.88</td><td>8.86</td><td>8.47</td><td>6.12</td><td>5.71</td><td>7.12</td><td>3.48</td></tr><tr><td rowspan="3">3</td><td>stale</td><td>3.63</td><td>4.41</td><td>4.25</td><td>3.33</td><td>3.14</td><td>3.85</td><td>2.19</td></tr><tr><td>predicted</td><td>6.76</td><td>8.63</td><td>8.19</td><td>5.99</td><td>5.66</td><td>6.90</td><td>3.37</td></tr><tr><td>true</td><td>6.97</td><td>8.98</td><td>8.58</td><td>6.22</td><td>5.79</td><td>7.17</td><td>3.51</td></tr></table>

Table 7: Effect of the number of adapter layers for Qwen3-8B. Parenthesized values indicate the layers per module for target layers 1, 9, 17, 25 and 33. We train each adapter for 0.46 epochs with κ = 1 in Eq. 10. (a) Error $\epsilon _ { \ell }$ of Eq. 11 (lower is better) and cosine similarity between the predicted and the true states at each layer ℓ that the draft model reads, on held-out mathematics. Top-1 is the fraction of block positions at which the decoder of Section 4.2, reading each adapter’s predicted states, ranks the target’s greedy token first. KL is KL ${ \boldsymbol { \cdot } } ( { \boldsymbol { p } } \parallel { \hat { \boldsymbol { p } } } )$ in nats, where $p$ is the target’s nexttoken distribution at temperature 1 and $\hat { p }$ is the decoder’s. Both use the 200 held-out mathematics sequences of Table 9. (b) Acceptance length τ of the ablation draft model at temperature 0.  
(a) Predicted states
<table><tr><td></td><td colspan="5">Error  $\epsilon _ { \ell }$ </td><td colspan="5">Cosine</td><td colspan="2">Decoder</td></tr><tr><td>Layers</td><td>1</td><td>9</td><td>17</td><td>25</td><td>33</td><td>1</td><td>9</td><td>17</td><td>25</td><td>33</td><td>Top-1</td><td>KL</td></tr><tr><td>15 (2/2/2/4/5)</td><td>0.260</td><td>0.255</td><td>0.274</td><td>0.219</td><td>0.160</td><td>0.962</td><td>0.963</td><td>0.958</td><td>0.970</td><td>0.981</td><td>.958</td><td>.068</td></tr><tr><td>12 (1/1/1/4/5) (ours)</td><td>0.317</td><td>0.275</td><td>0.291</td><td>0.223</td><td>0.161</td><td>0.943</td><td>0.957</td><td>0.953</td><td>0.969</td><td>0.981</td><td>.956</td><td>.072</td></tr><tr><td>10 (1/1/1/3/4)</td><td>0.317</td><td>0.278</td><td>0.295</td><td>0.231</td><td>0.172</td><td>0.943</td><td>0.956</td><td>0.951</td><td>0.967</td><td>0.979</td><td>.952</td><td>.087</td></tr><tr><td>8 (1/1/1/2/3)</td><td>0.321</td><td>0.283</td><td>0.302</td><td>0.244</td><td>0.182</td><td>0.941</td><td>0.955</td><td>0.949</td><td>0.964</td><td>0.977</td><td>.945</td><td>.111</td></tr></table>

(b) Acceptance length τ
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Layers</td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td></tr><tr><td>15 (2/2/2/4/5)</td><td>4.23</td><td>4.70</td><td>4.82</td><td>4.86</td><td>5.77</td><td>6.52</td><td>6.72</td><td>6.79</td></tr><tr><td>12 (1/1/1/4/5) (ours)</td><td>4.24</td><td>4.71</td><td>4.84</td><td>4.88</td><td>5.74</td><td>6.45</td><td></td><td>6.67 6.75</td></tr><tr><td>10 (1/1/1/3/4)</td><td>4.22</td><td>4.68</td><td>4.80</td><td>4.84</td><td>5.72</td><td>6.44</td><td>6.68</td><td>6.74</td></tr><tr><td>8 (1/1/1/2/3)</td><td>4.19</td><td>4.66</td><td>64.78</td><td>4.81 </td><td></td><td></td><td></td><td>5.66 6.38 6.59 6.67</td></tr></table>

where n runs over the anchors of these blocks and $j$ is the position within the block. We measure τ with the protocol of the ablation draft models (Appendix E) and no context refresh. The draft model reads each adapter’s predicted states at the unverified positions.

Number of adapter layers. We vary the number of adapter layers in Table 7. From fifteen to twelve layers, the modules for layers 1, 9 and 17 drop from two layers to one. This change raises their error (from 0.260 to 0.317 at layer 1) but changes the errors at layers 25 and 33 by at most 0.004 and τ at $J _ { 3 }$ by at most 0.04 tokens. Removing layers from the modules for layers 25 and 33 then raises the error there step by step and lowers the decoder’s agreement with the target. With eight layers instead of fifteen, τ at $J _ { 3 }$ drops by 0.05 tokens on GSM8K and 0.12 on MATH-500. We therefore give the modules for layers 1, 9 and 17 one layer each and put the extra layers in the modules for layers 25 and 33. Each layer adds to the time of every adapter call in Eq. 3. We use twelve layers.

Table 8: Effect of the exponent κ of the position weight in the adapter loss (Eq. 10) for Qwen3-8B, with twelve-layer adapters trained for 1.83 epochs. $\epsilon _ { 2 5 }$ and $\epsilon _ { \mathrm { { 3 3 } } }$ are the errors of Eq. 11 at layers 25 and 33 on held-out mathematics (lower is better), and the $J _ { 0 }$ to $J _ { 3 }$ columns give the acceptance length τ of the ablation draft model at temperature 0. Top-1 and KL are as in Table 7.
<table><tr><td></td><td></td><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td><td colspan="2">Decoder</td></tr><tr><td>κ</td><td>€25</td><td>€33</td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td>Top-1</td><td>KL</td></tr><tr><td>0</td><td>0.212</td><td>0.149</td><td>4.28</td><td>4.75</td><td>4.88</td><td>4.93</td><td>5.84</td><td>6.59</td><td>6.82</td><td>6.91</td><td>.967</td><td>.043</td></tr><tr><td> $1 / 4$ </td><td>0.210</td><td>0.148</td><td>4.27</td><td>4.75</td><td>4.88</td><td>4.93</td><td>5.81</td><td>6.57</td><td>6.81</td><td>6.88</td><td>.967</td><td>.045</td></tr><tr><td> $1 / 2$  (ours)</td><td>0.208</td><td>0.147</td><td>4.28</td><td>4.76</td><td>4.89</td><td>4.95</td><td>5.86</td><td>6.59</td><td>6.82</td><td>6.89</td><td>.967</td><td>.043</td></tr><tr><td>1</td><td>0.209</td><td>0.148</td><td>4.27</td><td>4.75</td><td>4.87</td><td>4.92</td><td>5.82</td><td>6.55</td><td>6.80</td><td>6.88</td><td>.964</td><td>.050</td></tr></table>

Position weighting in the loss. We train twelve-layer adapters on the split above and vary only the exponent κ of Eq. 10 (Table 8). The $\kappa = 1 / 2$ adapter is the ablation adapter of Appendix B. The exponent has little effect on τ, and at $J _ { 3 }$ the four adapters lie within 0.03 tokens on both benchmarks. The value $\kappa = 1 / 2$ gives the lowest error at layers 25 and 33, and $\kappa = 1$ gives the lowest decoder agreement and the highest KL.

## B DECODER AND OUTCOME SCORES

We describe the served configuration of the decoder of Section 4.2, which reuses the target’s upper layers without training. Figure 3 shows how one decoder pass supplies every factor of the outcome score in Eq. 6. We then calibrate this score, compare the decoder with a fine-tuned copy, and compare the calibrated score with other selection rules.

## B.1 SCORING RULE AND CACHE SIZE

Reused target layers. The decoder reuses the target layers above the deepest layer that the adapter predicts, followed by the target’s final norm and LM head, all frozen. Its estimate $\hat { p } ( \cdot \mid j )$ is the softmax of the LM head’s logits at temperature 1, at both sampling temperatures. For Qwen3-8B, the decoder reuses the two target layers above layer 33 (layers 34 and 35). The target sends its layer-33 states to the draft process after each verification. For Qwen3.8-27B, the decoder reuses the two target layers above layer 61 (layers 62 and 63). After each verification, the target sends the committed recurrent and convolution states of layer 62 (a Gated DeltaNet layer) and the keys and values of layer 63.

Calibrated score. JetSpec++ scores each outcome $( k , a )$ by the calibrated score $\tilde { P } ( k ) ~ P _ { \beta } ( a \mid k )$ To compute $\tilde { P } ( k )$ , JetSpec++ replaces each factor $\hat { p } _ { i }$ of Eq. 6 with a logistic function ${ \tilde { p } } _ { i }$ of a feature vector $\phi _ { i }$

$$
\tilde { p } _ { i } = \sigma \big ( \mathbf w ^ { \top } \phi _ { i } \big ) , \qquad G ( k ) = \prod _ { i \leq k } \tilde { p } _ { i } , \qquad \tilde { P } ( k ) = G ( k ) - G ( k + 1 ) , \quad G ( \gamma + 1 ) = 0 .\tag{12}
$$

Here w holds one coefficient for each feature. The product $G ( k )$ estimates the probability that verification accepts the first k draft tokens, so $\tilde { P } ( k )$ estimates the probability that verification accepts exactly k. The vector $\phi _ { i }$ holds six features of position i:

• the logit of the draft model’s top-1 probability at position i in its final pass,

• the logit of the decoder’s probability pˆ<sub>i</sub> of $d _ { i } ,$

• an indicator of whether $d _ { i }$ is the decoder’s top token arg max $_ { \mathit { v } } \hat { p } ( v \mid i - 1 )$

• the logit of the decoder’s largest renormalized probability $\operatorname* { m a x } _ { a \neq d _ { i } } \hat { p } ( a \ | \ i - 1 ) / ( 1 - \hat { p } _ { i } )$ of a token other than $d _ { i }$

• the scaled position $i / 1 4$

• a constant bias feature.

The factor $P _ { \beta } ( a \mid k ) \propto \hat { p } ( a \mid k ) ^ { 1 / \beta }$ is the decoder’s bonus distribution at $k ,$ tempered by the anchor temperature $\displaystyle { \big / } \beta .$ JetSpec++ renormalizes this distribution over the decoder’s top 32 candidates, with $d _ { k + 1 }$ removed when $k < \gamma$ . We fit w and $\beta$ on the calibration rounds of Appendix B.2.

![](images/0832ca054ffd6678b74ed66f156cde9b032e4313b213e08979b7257ad2b87ac7.jpg)  
Figure 3: Scoring every outcome with one decoder pass. The adapter predicts the block’s states from the verified context, and the decoder reads them with the verified context as keys and values. Layers $\ell _ { 1 } , \ldots , \ell _ { \mathrm { t o p } }$ are the target layers that the draft model reads, and $\ell _ { \mathrm { t o p } }$ is the deepest of them. Row $j$ fiis the decoder’s $\hat { p } ( \cdot \mid \bar { j } )$ . For the outcomes with $k = 5$ (outlined), rows 0 to 4 supply the acceptance probabilities $\hat { p } _ { 1 } , \ldots , \hat { p } _ { 5 }$ of the draft tokens, row 5 supplies the bonus distribution with $d _ { 6 }$ excluded, and later rows are not read. The bars give each outcome’s score against the threshold λ of Section 4.2: solid bars reach λ and are cached, and dashed bars are dropped. Heights are schematic. Our experiments instead cache the B outcomes with the highest calibrated score in every round (Appendix B.1).

Cache size. For both targets, the draft process caches the B highest-scoring outcomes per round, or all distinct outcomes when fewer exist. The cache size B is 16 in Tables 1 and 2, and 4 in Table 3. In the offline replays of Tables 10 and 11, the served rule caches outcomes in the same way at each $B .$

## ⋯B.2 FITTING THE CALIBRATED SCORE

We collect the calibration rounds by serving a pooled set of 56 held-out prompts for each target. ⋯For Qwen3-8B, each of GSM8K, MATH-500, HumanEval, MBPP and LCB contributes 8 prompts, drawn after its evaluation prompts. UltraChat (Ding et al., 2023) and the GSM8K training split <sub>fi</sub>contribute 8 prompts each. We fit once per target and verification budget, and the same fit serves ⋯every benchmark and both temperatures.

The served rule has three fitted parts: the six coefficients w of Eq. 12, the anchor temperature $\beta ,$ and the split-conformal thresholds on the calibrated score. We fit w by maximum likelihood of the realized k on the calibration rounds, and $\beta$ by maximum likelihood of the realized anchor. We set the thresholds on the same rounds by split-conformal prediction.

Table 9: Offline agreement of each decoder with the greedy token of the Qwen3-8B target on 200 held-out mathematics sequences (379,305 positions), with the ablation adapter. KL is $\mathrm { K L } ( p \Vert \hat { p } )$ in nats, where $p$ is the target’s next-token distribution at temperature 1 and pˆ is the decoder’s.
<table><tr><td>Decoder</td><td>Top-1 agreement</td><td>KL to target</td></tr><tr><td>training-free</td><td>.967</td><td>.043</td></tr><tr><td>CE-trained</td><td>.968</td><td>.247</td></tr></table>

Table 10: Offline hit rate of each decoder when the cache holds at most B outcomes per round, ranked by the calibrated score of that decoder. The rounds come from JetSpec++ at $J _ { 1 } { - } R _ { 1 }$ for Qwen3-8B with the final adapter in the setting of Table 1, at temperature 0 and budget 16. The hit rate is the share of rounds after the first of each request whose verified outcome is in the cache.
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Decoder</td><td>B = 1</td><td>B = 4</td><td> B = 8</td><td>B = 16</td><td> $B = 1$ </td><td> $B = 4$ </td><td>B = 8</td><td> $B = 1 6$ </td></tr><tr><td>training-free</td><td>.624</td><td>.927</td><td>.977</td><td>.993</td><td>.634</td><td>.903</td><td>.961</td><td>.989</td></tr><tr><td>CE-trained</td><td>.617</td><td>.934</td><td>.979</td><td>.994</td><td>.654</td><td>.935</td><td>.980</td><td>.995</td></tr></table>

## B.3 TRAINING-FREE VERSUS FINE-TUNED DECODERS

To test whether fine-tuning improves the reused layers, we compare the decoder of Appendix B.1, which we call the training-free decoder, with a fine-tuned copy. We fine-tune the CE-trained decoder for 6,000 updates on the adapter’s predictions, with cross-entropy to the target’s greedy token. Table 9 reports top-1 agreement, the fraction of block positions where the decoder’s most probable token equals the target’s greedy token.

Fine-tuning adds little to the reused layers (Table 9). Cross-entropy fine-tuning raises top-1 agreement by .001, but it also raises the KL divergence to the target’s next-token distribution from .043 to .247.

Table 10 tests whether this agreement carries over to the cache. We record the rounds of JetSpec++ once and replay them with each decoder.

Fine-tuning raises the hit rate by at most .032 (the CE-trained decoder on MATH-500 at $B = 4 )$ We therefore serve the training-free decoder.

## B.4 COMPARISON OF SELECTION RULES

The served rule scores each outcome $( k , a )$ by the calibrated score $\tilde { P } ( k ) P _ { \beta } ( a \mid k )$ of Appendix B.1. We compare it in Table 11 with rules that each remove one source of information. The uncalibrated rule ranks outcomes by the decoder’s $P ( k , a )$ of Eq. 6. The draft-model rule scores k by $P ( k )$ of Eq. 6 with the draft model’s probabilities $q _ { \theta } ( d _ { i } )$ of its own draft tokens in place of the decoder’s $\hat { p } _ { i }$ It keeps the anchor factor $P _ { \beta } ( a \mid k )$ of the served rule, so it differs from the served rule only in the score of k. All of these rules rank anchors by the decoder.

We also compare the served rule with the geometric fan-out of SSD’s Theorem 12, the series of Proposition $2 ( \mathrm { a } )$ over all lengths $k = 0 , \ldots , \gamma$ (Appendix F.2). We fit the acceptance rate α and the exponent ρ of Assumption 2 on the calibration rounds. We then round the anchor counts $F _ { k }$ to integers that sum to B by largest remainders. We fill the fan-out with the draft model’s anchors, as SSD does. $\operatorname { A t } k = \gamma$ , the draft model has no distribution for the bonus token, so the fan-out takes the decoder’s anchors. A second row fills the same fan-out with the decoder’s anchors at every length.

The served rule leads or ties every other rule at every cache size, and at B = 4 it leads the uncalibrated rule by .008 on GSM8K and .039 on MATH-500. The draft-model rule ranks k worse and trails the served rule by .186 on GSM8K and .171 on MATH-500 at B = 4. The geometric fan-out, a fixed shape filled with the draft model’s anchors, trails every rule. With the decoder’s anchors, it still trails the served rule by .640 on GSM8K and .687 on MATH-500 at $B = 4$

Table 11: Offline hit rate of each selection rule when it caches at most B outcomes per round, counted as in Table 10 and on the same rounds.
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Rule</td><td>B = 1</td><td>B = 4</td><td>B = 8</td><td> $B = 1 6$ </td><td> $B = 1$ </td><td> $B = 4$ </td><td> $B = 8$ </td><td>B = 16</td></tr><tr><td>served rule</td><td>.624</td><td>.929</td><td>.978</td><td>.993</td><td>.644</td><td>.909</td><td>.963</td><td>.990</td></tr><tr><td>uncalibrated rule</td><td>.613</td><td>.921</td><td>.975</td><td>.993</td><td>.585</td><td>.870</td><td>.944</td><td>.981</td></tr><tr><td>draft-model rule</td><td>.387</td><td>.743</td><td>.885</td><td>.962</td><td>.444</td><td>.738</td><td>.862</td><td>.941</td></tr><tr><td>geometric fan-out</td><td>.050</td><td>.208</td><td>.512</td><td>.670</td><td>.044</td><td>.163</td><td>.588</td><td>.741</td></tr><tr><td>geometric fan-out, decoder anchors</td><td>.067</td><td>.289</td><td>.655</td><td>.890</td><td>.059</td><td>.222</td><td>.682</td><td>.898</td></tr></table>

Table 12: Acceptance length τ over refinement passes for the final draft models of Qwen3-8B. All runs use the target’s true states in the setting of Table 6. Refinement passes lower the τ of the JetSpec draft model and raise that of the JetSpec++ draft model.
<table><tr><td rowspan="2">Draft model</td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td> $J _ { 0 }$ </td><td>J1</td><td>J2</td><td> $J _ { 3 }$ </td><td>J0</td><td>J1</td><td>J2</td><td>J3</td></tr><tr><td>JetSpec</td><td>5.86</td><td>2.39</td><td>2.65</td><td>2.45</td><td>7.51</td><td>2.57</td><td>2.84</td><td>2.61</td></tr><tr><td>JetSpec++</td><td>5.70</td><td>6.59</td><td>6.88</td><td>6.97</td><td>7.52</td><td>8.55</td><td>8.86</td><td>8.98</td></tr></table>

## C TRAINING THE DRAFT MODEL TO REFINE

Our draft model (Section 4.3) takes token inputs and trains on blocks with a clean prefix so that it can refine its own draft. We train it with a forward KL loss, and it attends causally within the block. We ablate the inputs and the training blocks in Section 5.5 (Table 4). In this appendix, we explain that result and extend it to the final draft models (Table 12), then ablate the share of fully noised blocks, the loss and the attention. Each draft model in these ablations is a two-epoch ablation draft model for Qwen3-8B (Appendix E).

Inputs and training blocks. A draft model with mask inputs drafts $J _ { 0 }$ from mask embeddings and takes its own tokens as inputs from $J _ { 1 }$ on. When such a draft model also trains only on fully noised blocks, its draft positions hold only mask embeddings during training, and it collapses at $J _ { 1 }$ (Table 4). The $\tau$ of this draft model falls from 4.34 at $J _ { 0 }$ to 1.23 at $J _ { 1 }$ on GSM8K and from 5.99 to 1.28 on MATH-500. The JetSpec draft model is trained on mask inputs and fully noised blocks (Appendix E), and the JetSpec++ draft model on token inputs and blocks with a clean prefix. Table 12 compares these two final draft models.

Share of fully noised blocks. We vary the share of fully noised blocks from 0 to 1, with token inputs (Table 13). The share is the probability that the clean prefix length m of Eq. 9 is 0. Our draft model uses 0.25 (Appendix E). Shares of 0 and 1 give a lower τ at $J _ { 3 }$ than the four shares between them. From 0.125 to 0.5, τ at $J _ { 3 }$ stays within 0.11 tokens on GSM8K and 0.12 on MATH-500. We decompose τ at $J _ { 3 }$ into τ at $J _ { 0 }$ and the gain from $J _ { 0 }$ to $J _ { 3 } { \mathrm { : } }$

• From 0.125 to 1, τ at $J _ { 0 }$ stays within 0.12 tokens on GSM8K and 0.14 on MATH-500. The draft model drafts $J _ { 0 }$ from a fully noised block, and with a share of 0 it never trains on such a block. Against 0.25, a share of 0 lowers τ at $J _ { 0 }$ by 0.43 tokens on GSM8K and 0.48 on MATH-500.

• The gain from $J _ { 0 }$ to $J _ { 3 }$ is largest at a share of 0 (0.89 tokens on GSM8K and 1.38 on MATH-500) and falls to 0.60 and 1.02 at a share of 0.5. A share of 1 removes the clean prefixes, and the draft model does not refine.

Training loss. We compare forward KL, SFT and reverse KL as the divergence of Eq. 9 (Appendix E) and change nothing else (Table 14). Forward KL leads SFT, and SFT leads reverse $\mathrm { K L }$ Forward KL trains the draft model to match the target’s full next-token distribution. The draft model thus also learns the target’s ranking of the tokens beyond the greedy one. SFT supervises only the target’s greedy token (the next token of the training sequence). The gap between SFT and forward KL is larger at $J _ { 3 }$ than at $J _ { 0 } .$ . Reverse KL trails both, and most of its deficit against forward KL at $J _ { 3 }$ is already present at $J _ { 0 }$ (1.40 of 1.97 tokens on GSM8K and 1.88 of 2.72 on MATH-500).

Table 13: Effect of the share of fully noised training blocks. Acceptance length $\tau$ for Qwen3-8B in the setting of Table 4. Each draft model takes token inputs and differs from ours only in the probability of prefix length $m = 0 .$ , with the rest split equally over lengths 1 to 15.
<table><tr><td rowspan="2"> $\mathrm { P r } [ m = 0 ]$ </td><td colspan="3">GSM8K</td><td rowspan="2"></td><td colspan="3">MATH-500</td></tr><tr><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$   $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$   $J _ { 3 }$ </td></tr><tr><td>0</td><td>3.91</td><td>4.56</td><td>4.75</td><td>4.80</td><td>5.43</td><td>6.39</td><td>6.72 6.81</td></tr><tr><td>0.125</td><td>4.29</td><td>4.84</td><td>5.00</td><td>5.04</td><td>5.88</td><td>6.69 6.97</td><td>7.09</td></tr><tr><td>0.25 (ours)</td><td>4.34</td><td>4.87</td><td>5.01</td><td>5.05</td><td>5.91</td><td>6.75</td><td>6.99 7.08</td></tr><tr><td>0.375</td><td>4.32</td><td>4.78</td><td>4.90</td><td>4.94</td><td>5.91</td><td>6.69</td><td>6.91 6.97</td></tr><tr><td>0.5</td><td>4.41</td><td>4.84</td><td>4.96</td><td>5.01</td><td>6.01</td><td>6.74</td><td>6.95 7.03</td></tr><tr><td>1</td><td>4.36</td><td>4.37</td><td>4.37</td><td>4.37</td><td>6.02</td><td>6.01</td><td>6.02 6.01</td></tr></table>

Table 14: Effect of the training loss. Acceptance length τ for Qwen3-8B in the setting of Table 4.
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Loss</td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td></tr><tr><td>forward KL (ours) 4.34</td><td></td><td>4.87</td><td>5.01</td><td>5.05</td><td>5.91</td><td>6.75</td><td>6.99</td><td>7.08</td></tr><tr><td>SFT</td><td>4.28</td><td>4.74</td><td>4.85</td><td>4.88</td><td>5.81</td><td>6.58</td><td></td><td>6.76 6.81</td></tr><tr><td>reverse KL</td><td>2.94 3.06</td><td></td><td>3.08</td><td>3.08</td><td>4.03</td><td>4.29</td><td></td><td>94.34 4.36</td></tr></table>

Table 15: Effect of the attention within the draft block. Acceptance length τ for Qwen3-8B in the setting of Table 4.
<table><tr><td></td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Attention</td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td><td> $J _ { 0 }$ </td><td> $J _ { 1 }$ </td><td> $J _ { 2 }$ </td><td> $J _ { 3 }$ </td></tr><tr><td>causal (ours)</td><td>4.34</td><td>4.87</td><td>5.01</td><td>5.05</td><td>5.91</td><td>6.75</td><td>6.99</td><td>7.08</td></tr><tr><td>bidirectional</td><td>1 4.36</td><td>4.81</td><td>4.95</td><td>5.00</td><td>5.88</td><td>6.65</td><td>6.82</td><td>6.93</td></tr></table>

Attention within the block. We compare causal and bidirectional attention within the draft block and change nothing else (Table 15). At $J _ { 0 }$ the bidirectional draft model matches the causal one. In the refinement passes the causal draft model leads.

## D SELF-REFINEMENT EXAMPLE AND DRAFTING WITHOUT PIPELINING

A context refresh (Section 4.3) uses both the adapter and the decoder. Figure 4 traces one context refresh and its anchor replacement, and we measure self-refinement without pipelining. Appendix F.3 shows that the output stays lossless when anchor replacement rewrites the draft.

Anchor replacement rule. After a context refresh, the decoder reads the refreshed states of the draft $y ^ { ( r ) }$ , and anchor replacement uses the first disagreement between the decoder’s argmax and the draft:

$$
\begin{array} { c } { { \hat { k } = \operatorname* { m a x } \big \{ 0 \leq i \leq \gamma - 1 : y _ { j } ^ { ( r ) } = \arg \operatorname* { m a x } _ { v } \hat { p } ( v \mid j - 1 ) \ \forall 1 \leq j \leq i \big \} , } } \\ { { \hat { a } = \arg \operatorname* { m a x } _ { v } \hat { p } ( v \mid \hat { k } ) } } \end{array}\tag{13}
$$

Here $\hat { p } ( \cdot \mid j )$ are the decoder’s next-token estimates of Section 4.2, and $( \hat { k } , \hat { a } )$ is the outcome that greedy verification of the draft would return if the decoder were the target. Anchor replacement fixes the tokens up to position <sup>ˆ</sup>k for the rest of the round. If aˆ differs from the draft token at position $\hat { k } + 1$ , it writes $\hat { a }$ there and fixes that position too. $\mathbf { A }$ position fixed earlier keeps its token. The next pass reads the refreshed states only up to position $\hat { k } ,$ because a token written at position $\hat { k } + 1$ has no refreshed state, and it redrafts only the positions after the last fixed one.

Drafting without pipelining. JetSpec++ drafts while the target verifies, and Table 16 compares this pipelined drafting with drafting on the critical path, which lengthens the round. When every method drafts on the critical path, $J _ { 1 } { - } R _ { 1 }$ leads DSpark by 0.79 tokens on GSM8K and 0.81 on MATH-500, so JetSpec++ also has the higher τ without pipelining. Across $J _ { 1 } – R _ { 1 } \ \mathrm { t o } \ J _ { 3 } – R _ { 3 } .$ , the pipelined drafting stays within 0.12 tokens of critical-path drafting on GSM8K and 0.19 on MATH-500. JetSpec++

![](images/46413a69ddbf5c14c0e6232e66ecddb14da0518a2b7f25c4620de82aa602059e.jpg)  
Figure 4: Correcting the first disagreement with one anchor replacement. The figure shows the context refresh after $J _ { 0 }$ of one batch row, the block t + 1 drafted for the outcome $( k , a )$ , in the layout that Figure 3 uses for the predicted states of block t. The adapter re-predicts the states of block $t + 1$ and the decoder reads them as in Figure 3. The decoder’s top-1 token agrees with the $J _ { 0 }$ draft at ⋯ positions 1 to 3 and disagrees at position 4, so $\hat { k } = 3$ and anchor replacement writes aˆ at position 4. The $J _ { 1 }$ pass reads aˆ at position 4 and the $J _ { 0 }$ tokens at the other positions. Anchor replacement corrects position 4 and fixes positions 1 to 4, and the $J _ { 1 }$ pass corrects position 7 by itself.

Table 16: Acceptance length τ with drafting pipelined or on the critical path, for Qwen3-8B in the ⋯ setting of Table 1 at budget 16 and temperature 0. A pipelined draft starts before verification ends, as JetSpec++ serves. A critical-path draft starts after verification and reads the target’s true states, as in DFlash and DSpark.
<table><tr><td>Method</td><td>Drafting</td><td>GSM8K</td><td>MATH-500</td></tr><tr><td>DFlash</td><td>critical path</td><td>5.91</td><td>7.59</td></tr><tr><td>DSpark</td><td>critical path</td><td>7.31</td><td>9.01</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 1 } { \cdot } R _ { 1 }$ </td><td>pipelined</td><td>7.98</td><td>9.63</td></tr><tr><td>critical path</td><td>8.10</td><td>9.82</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 2 ^ { - } } R _ { 2 }$ </td><td>pipelined</td><td>9.23</td><td>10.68</td></tr><tr><td>critical path</td><td>9.25</td><td>10.61</td></tr><tr><td rowspan="2">JetSpec++,  $J _ { 3 ^ { - } } R _ { 3 }$ </td><td>pipelined</td><td>9.78</td><td>11.04</td></tr><tr><td>critical path</td><td>9.86</td><td>11.08</td></tr></table>

can therefore also run without pipelining when its gain in τ over DSpark outweighs any extra draft time, which favors targets whose verification takes most of the round.

## E TRAINING AND EVALUATION DETAILS

Training data and optimization. We take the prompts of the 800k training sequences from Nemotron-Post-Training-Dataset-v2 (NVIDIA, 2025) and CodeAlpaca-20k (Chaudhary, 2023), and each target model generates the responses greedily in non-thinking mode. We train the draft model of Section 4.3 on these sequences for six epochs with AdamW and a peak learning rate of $3 \times 1 0 ^ { - 4 }$ (linear warmup over the first 4% of steps, then cosine decay). We use gradient clipping at norm 1.0, no weight decay, a global batch of 16 sequences and seed 42. Training takes 347 GPU-hours on GB200 GPUs for Qwen3-8B and 710 for Qwen3.8-27B. The ablation draft models are for Qwen3- 8B. They use the same learning rate, learning-rate schedule, global batch and seed, but train for two epochs on the 239k mathematics sequences of the 800k set.

Training blocks. Training sequences have at most 3,072 tokens for Qwen3-8B and 4,096 for Qwen3.8-27B, and we sample up to 512 anchor positions per sequence. An anchor must be a supervised response token that leaves room for a full block of 16 positions (the anchor and $\gamma = 1 5$ draft positions). Under the prefix-length schedule, the clean prefix length m of Eq. 9 is 0 (a fully noised block) with probability 0.25 and each of 1, . . . , 15 with probability 0.05. Each noise token in y˜ copies the token at an independent, uniformly random position of the sequence up to and including the anchor.

Draft model architecture. The draft model has five transformer layers and ends with a final normalization. It concatenates the target’s hidden states $h _ { x }$ at the five target layers of Appendix A, passes the concatenation through a linear layer and normalizes the result. Every draft layer projects this result into keys and values of the verified context, as in DFlash (Chen et al., 2026), and attends causally within the block (Section 4.3). We reuse the target’s embedding and LM head and keep them frozen. The draft model has 1.05B trainable parameters for Qwen3-8B and 1.73B for Qwen3.8-27B.

Training objective. We use the forward KL divergence at temperature 1 as the divergence of Eq. 9 and average it over the supervised draft positions of a batch, with no decay along the block. In the loss ablations of Table 14 (Appendix C), SFT is the cross-entropy to the next token of the training sequence, and reverse KL is KL(q<sub>θ</sub> ∥ p) at temperature 1.

Baselines. The DFlash, DSpark, JetSpec and Domino draft models are trained for six epochs on the same 800k sequences as our draft model. DFlash, DSpark, Domino and JetSpec also share our maximum length, anchor count, warmup and seed. DFlash and JetSpec have five transformer layers and read the same five target layers as our draft model. However, both baselines fill every draft position of a training block with a mask token.

• We train DFlash without the positional loss decay and at our learning rate of $3 \times 1 0 ^ { - 4 }$ . It attends bidirectionally within the block and is trained with the SFT loss at our global batch.

• We reproduce DSpark with 15 draft tokens after the anchor, its Markov head and its confidence head. The loss also follows DSpark: a weighted sum of the cross-entropy, the L1 distance between the draft and target distributions, and the confidence loss, with weights 0.1, 0.9 and 1. DSpark weights all three terms at draft position i by exp(−(i − 1)/7), the positional loss decay of DFlash. We train DSpark at a learning rate of $6 \times 1 0 ^ { - 4 }$ . Its global batch is 8 for Qwen3-8B and 16 for Qwen3.8-27B.

• Domino attends bidirectionally within the block and adds its Domino head. It is trained with its cross-entropy loss and a positional loss decay constant of 7. We train it at our learning rate of $3 \times 1 0 ^ { - 4 }$ and at a global batch of 48.

• JetSpec attends causally within the block and is trained with our loss and learning rate at a global batch of 8.

• SSD uses the off-the-shelf Qwen3-0.6B model as its draft model. It drafts 4 tokens per round (a block of 5 with the anchor) and pre-drafts three outcomes (k, a) for each k on GSM8K and one on MATH-500, a flat fan-out as in SSD’s released code.

Benchmarks. For the fixed shuffle of Section 5.1, we shuffle the test split of each benchmark with seed 0. We use the sanitized MBPP and release v6 of LCB. The runs of the ablation draft models use the first 164 prompts of GSM8K and MATH-500 from the same shuffle, at temperature 0 and with at most 256 generated tokens. On MT-Bench, we evaluate both turns of the 80 questions. Every method starts the second turn from the same first-turn answer, which the target generates with AR.

Repetition loops. Some outputs, mostly greedy ones at temperature 0, fall into a verbatim repetition loop that runs to the 2,048-token cap. Draft models can predict repeated text almost perfectly, which would inflate τ and the speedup. In Tables 1, 2, 3, 5 and 16, we therefore treat an output that reaches the cap as a repetition loop if its final 64 tokens occur verbatim at least three times in it, without overlap. We stop such an output at the onset of the loop, the first occurrence of that 64-token span, and keep other capped outputs whole. We handle this stop like an EOS stop: the round that crosses the onset counts only its tokens before the onset. We measure time through the end of that round. We cut every method, AR included, at its own onsets. On MT-Bench, the rule applies to each turn separately. The runs of Tables 6 and 12, in which every round drafts only the block of the verified outcome, stop only at EOS or the cap.

Metrics. We compute τ as the committed tokens per verification round, pooled over all rounds of the requests of a benchmark, including the first round of each request. τ includes the bonus token of each round, and the token that the target produces at prefill belongs to no round. In the last round of a request, τ counts only the tokens committed before EOS or the generation cap. We compute throughput as the tokens per second, also pooled over the requests of a benchmark. Throughput counts every round after the first of each request. A round is a hit round when it verifies the predrafted block of the previous round’s outcome, and a miss round otherwise. The hit rate is the share of hit rounds, and $\tau _ { \mathrm { h i t } }$ and $\tau _ { \mathrm { m i s s } }$ are the values of τ on the hit and the miss rounds. These three values also exclude the first round of each request, whose block comes from no cache.

Serving protocol. For Table 3, each method generates responses to the 128 prompts of GSM8K and MATH-500 in Table 1, in four groups of 32, until EOS or 2,048 tokens. Each method runs at temperature 0 and batch size 1, with budget 16 (5 for SSD, with the settings of the Baselines paragraph above). JetSpec++ runs $J _ { 1 } { - } R _ { 1 }$ for Qwen3-8B and $J _ { 3 } – R _ { 3 }$ for Qwen3.8-27B, with a cache of $B = 4$ outcomes (Appendix B.1). We pool the four groups, weighting τ by rounds and throughput by time, and the mean times exclude the first round of each request. Each speedup is relative to the AR row of the same target and benchmark.

Correctness checks. In exact arithmetic, verification makes the greedy output of every speculative method equal to that of AR. We run two checks on GSM8K with JetSpec++ at $J _ { 1 } { - } R _ { 1 }$ for Qwen3-8B, as in Table 3. First, we run the target model in fp32. The first 256 tokens that JetSpec++ generates for each of the first 8 prompts of the fixed shuffle (2,048 tokens in all) match AR token by token. Second, we check the CUDA graphs of the draft process against eager execution. On the first 3 prompts, JetSpec++ replays 492 draft rounds from a CUDA graph, and we recompute each of them eagerly in the same round. No recomputed output differs from the graph’s.

## F THEORETICAL ANALYSIS

In the three subsections of this appendix, we prove results that support the method of Section 4.

• Appendix F.1 shows that the decoder score $P ( k , a )$ of Eq. 6 is the exact probability of the verification outcome $( k , a )$ when the decoder is perfect (Section 4.2).

• Appendix F.2 relates the selection of Eq. 7 to the geometric fan-out of SSD and shows that, with a perfect decoder, per-round selection is at least as good as any fixed allocation (Section 4.2).

• Appendix F.3 shows that verification stays lossless for every draft, including a draft that anchor replacement rewrites (Section 4.3).

## F.1 EXACT OUTCOME PROBABILITIES WITH A PERFECT DECODER

Outcome selection (Eq. 7) ranks the outcomes $( k , a )$ by the decoder score $P ( k , a )$ of Eq. 6. We show that, with a perfect decoder, $P ( k , a )$ is exactly the probability that verification of the block ends at outcome $( k , a )$ , at any temperature and for any draft that is fixed before verification. Eq. 7 then selects the cache with the largest hit probability.

Example. Take $\gamma = 2$ . Verification uses the point-mass proposal of Appendix F.3, so it accepts each draft token with the target’s probability of that token. The target gives $d _ { 1 }$ probability 0.8, and after $d _ { 1 }$ it gives $d _ { 2 }$ probability 0.5 and another token b probability 0.3. Verification ends at $( 1 , b )$ when it accepts $d _ { 1 }$ (probability 0.8), rejects $d _ { 2 }$ (probability $1 - 0 . 5 \ = \ 0 . 5 )$ and draws b from the residual (probability $0 . 3 / 0 . { \overset { \cdot } { 5 } } = \ 0 . 6 )$ . This gives $0 . 8 \cdot 0 . 5 \cdot 0 . 6 = 0 . 2 4$ , which equals $P ( 1 , b ) = \hat { p } _ { 1 } \hat { p } ( b \mid \hat { 1 } ) = 0 . 8 \cdot \mathbf { \bar { 0 } } . 3$ for a perfect decoder. The rejection probability cancels against the normalization of the residual, and the proof of Proposition 1 shows that this cancellation holds for every outcome.

## Assumption 1 (Perfect decoder).

(i) The draft tokens $d _ { 1 } , \ldots , d _ { \gamma }$ of the block are fixed before the verification of the block starts.

(ii) The random draws that verify this block are independent of what chose x and the draft.

(iii) Verification uses the acceptance rule of Section 3 with the point-mass proposal (Appendix F.3).

(iv) The decoder is perfect in that its estimate after prefix j equals the target’s next-token distribution at the sampling temperature, $\hat { p } ( \cdot \mid j ) = p ( \cdot \mid x , d _ { \leq j } )$ . In this appendix, x includes the anchor y<sub>0</sub>, which Eq. 1 lists separately, and $d _ { < 0 }$ is empty, so row 0 of the decoder estimates $p ( \cdot \mid x )$ , the target’s distributionfor $d _ { 1 } . A t$ temperature 0, every $p ( \cdot \mid x , d _ { \leq j } )$ is the point mass on the target’s argmax, and afixed rule breaks a tie.

Every draft of Sections 4.1 to 4.3 satisfies conditions (i) and (ii) of Assumption 1: the initial draft $J _ { 0 }$ and the fallback block of a miss, the passes of Eq. 8, drafts from predicted states, and blocks edited by anchor replacement (Appendix F.3).

Let $o ^ { \star } = \left( k ^ { \star } , a ^ { \star } \right)$ be the realized outcome of the block, where $k ^ { \star }$ is the number of accepted draft tokens and $a ^ { \star }$ is the committed bonus token. Let $P ( k , a )$ be the score of Section 4.2 computed from the draft $d _ { 1 : \gamma }$ of the block, by Eq. 6 for $k < \gamma$ and by $\begin{array} { r } { \dot { P ( \gamma , a ) } = P ( \gamma ) \hat { p } ( a \mid \gamma ) } \end{array}$ for $k = \gamma$

Proposition 1 (Exact outcome distribution). Under Assumption 1 and at any temperature, $\mathrm { P r } [ o ^ { \star } =$ $( k , \bar { a } ) \mid x , d _ { 1 : \gamma } ] = P ( k , a )$ for every outcome $( k , a )$ , with $a \neq d _ { k + 1 }$ when $k < \gamma$

Proof. Write $p _ { i } ( \cdot ) : = p ( \cdot \mid x , d _ { < i } )$ for the target distribution at draft position i, and $\hat { p } _ { i } : = \hat { p } ( d _ { i } \ |$ $i - 1 )$ . Because the proposal is the point mass on $d _ { i }$ , the acceptance rule gives

$$
\operatorname* { P r } [ \operatorname { a c c e p t } d _ { i } \mid d _ { < i } { \mathrm { ~ a c c e p t e d } } ] = \operatorname* { m i n } \{ 1 , p _ { i } ( d _ { i } ) / 1 \} = p _ { i } ( d _ { i } ) .
$$

Each acceptance test uses a fresh draw that is independent of the draft and of the earlier tests, so the acceptance probabilities of successive positions multiply. Let $k < \gamma _ { : }$ , and assume first that the rejection probability at position $k + 1$ is positive. Given that $d _ { 1 } , \ldots , d _ { k }$ are accepted, rejection at k+1 has probability $1 - p _ { k + 1 } \mathopen { } \mathclose \bgroup \left( d _ { k + 1 } \aftergroup \egroup \right)$ , and the bonus token is drawn from the residual max $\{ p _ { k + 1 } ( \cdot ) -$ $\delta _ { d _ { k + 1 } } ( \cdot ) , 0 \} / [ 1 - p _ { k + 1 } ( d _ { k + 1 } ) ]$ ], where $\delta _ { d _ { k + 2 } }$ is the point mass at $d _ { k + 1 }$ . Thus, for $a \neq d _ { k + 1 }$

$$
\operatorname* { P r } [ o ^ { \star } = ( k , a ) \mid x , d _ { 1 : \gamma } ] = \Big ( \prod _ { i = 1 } ^ { k } p _ { i } ( d _ { i } ) \Big ) [ 1 - p _ { k + 1 } ( d _ { k + 1 } ) ] \frac { p _ { k + 1 } ( a ) } { 1 - p _ { k + 1 } ( d _ { k + 1 } ) } = \Big ( \prod _ { i = 1 } ^ { k } \hat { p } _ { i } \Big ) \hat { p } ( a \mid k ) .
$$

The last equality applies condition (iv) of Assumption 1 to each factor. If instead $p _ { k + 1 } ( d _ { k + 1 } ) = 1$ then $p _ { k + 1 } ( a ) \ : = \ : 0$ for every $a \ne d _ { k + 1 }$ , and both sides vanish. For $k = \gamma$ , every draft token is accepted and the bonus token is drawn from $p _ { \gamma + 1 } ( \cdot )$ , so

$$
\operatorname* { P r } [ o ^ { \star } = ( \gamma , a ) \mid x , d _ { 1 : \gamma } ] = \Big ( \prod _ { i = 1 } ^ { \gamma } \hat { p } _ { i } \Big ) \hat { p } ( a \mid \gamma ) .
$$

Both expressions equal $P ( k , a )$ of Section 4.2.

## F.2 PER-ROUND SELECTION AND THE GEOMETRIC FAN-OUT OF SSD

Section 4.2 contrasts the per-round selection of Eq. 7 with the fixed geometric fan-out of SSD (Kumar et al., 2026), and Proposition 2 states the two facts behind this contrast. Under SSD’s assumptions, a continuous surrogate of Eq. 7 recovers SSD’s geometric series on the lengths that the surrogate’s maximizer keeps (part (a)). Selection by each round’s posterior is at least as good as any fixed allocation in every round (part (b)). Table 11 is the measured counterpart of part (b).

SSD caches $F _ { k }$ anchors for each accepted length k (its Definition 10). It keeps two fixed allocations and uses one of them in each round. The choice depends on whether the block under verification came from the speculation cache or from SSD’s fallback after a cache miss, so it is whether the previous round hit the cache.

We model the accepted length k in terms of the acceptance rate α of the draft model. We write the resulting law of k over rounds as $P _ { \alpha } ( k )$ , to keep it apart from the per-round score $P ( k )$ of Eq. 6.

Assumption 2 (Constant acceptance rate and power-law hit rate). The draft model has constant acceptance rate $\alpha \in ( 0 , 1 )$ . Over rounds, each draft token is accepted with probability $\alpha ,$ independently given the acceptance of the tokens before it, so the realized accepted length k has the law

$$
P _ { \alpha } ( k ) = \left\{ \begin{array} { l l } { \alpha ^ { k } ( 1 - \alpha ) , } & { k < \gamma , } \\ { \alpha ^ { \gamma } , } & { k = \gamma . } \end{array} \right.
$$

For the surrogate problem of part (a) only, the hit mass captured by $F$ anchors at length $k ,$ conditional on $k ,$ is modeled by the surrogate curve $1 - F ^ { - \rho } f o r$ real $F \overset { \cdot } { \geq } 1$ and some $\rho > 0$

Remark 1 (Source of the assumption). The acceptance law is the model of $\mathbf { S } \mathbf { S } \mathbf { D } ^ { \prime } \mathbf { s }$ Theorem $^ { 1 2 , }$ and the curve is the per-length form of SSD’s power-law hit rate (its Definition 11).

For part (a), we follow SSD and replace Eq. 7 by a continuous allocation over lengths with the surrogate curve of Assumption 2. The curve is zero at $F = 1$ , so a single anchor at length k captures no hit mass in the surrogate problem. The surrogate problem maximizes $\begin{array} { r } { \sum _ { k } P _ { \alpha } ( k ) \left( 1 - F _ { k } ^ { - \rho } \right) } \end{array}$ over $F _ { k } \in \{ 0 \} \cup [ 1 , \infty )$ subject to $\textstyle \sum _ { k } F _ { k } \leq B$ , where $B > 1$ is the cache size of Eq. 7 and the hit mass at length k is 0 when $\begin{array} { r } { \dot { F _ { k } } = 0 . \overline { { \mathrm { A } } } } \end{array}$ length k is active when $F _ { k } \ge 1$

For part (b), let $I _ { t }$ be the information that the selection observes in round t, such as the outputs of the decoder and the draft model and whether the previous round hit the cache. For an outcome $o = \left( k , a \right)$ and the realized outcome $o ^ { \star }$ , let

$$
\pi _ { t } ( o ) : = \operatorname* { P r } [ o ^ { \star } = o \mid I _ { t } ] , \qquad \pi _ { t } ( S ) : = \sum _ { o \in S } \pi _ { t } ( o ) \quad { \mathrm { f o r ~ a ~ c a c h e ~ } } S .
$$

We call $\pi _ { t }$ the posterior of round $t .$ Let $a _ { k , s }$ denote the anchor at length k with the s-th largest value of $\pi _ { t } ( k , \cdot )$ , with ties broken by a fixed order on tokens. A fixed allocation $\{ F _ { k } \}$ caches $( k , a _ { k , 1 } ) , \ldots , ( k , a _ { k , F _ { k } } )$ at each length k. Order the outcomes by decreasing $\pi _ { t } ,$ with ties broken by smaller $s ,$ then by smaller k. Let $\bar { S } _ { t } ^ { \star }$ be the set of the first B outcomes in this order. This set is a solution of Eq. 7 with P replaced by $\pi _ { t }$

Suppose that $I _ { t }$ contains the verified context x and the draft $d _ { 1 : \gamma }$ , that the target’s draws in round t are independent of $I _ { t } ,$ , and that the decoder is perfect (Assumption 1). Then Proposition 1 gives $\pi _ { t } ( k , a ) = P ( k , a )$ , so $S _ { t } ^ { \star }$ is a solution of Eq. 7 itself. With an imperfect decoder, Remark 3 bounds the loss of the cache of Eq. 7.

Proposition 2 (Adaptive selection and the geometric fan-out).

(a) Under Assumption 2, every active length k at any maximizer of the surrogate problem receives

$$
F _ { k } = C \alpha ^ { k / ( 1 + \rho ) } \quad ( k < \gamma ) , \qquad F _ { \gamma } = C \alpha ^ { \gamma / ( 1 + \rho ) } ( 1 - \alpha ) ^ { - 1 / ( 1 + \rho ) } .
$$

Here $C > 0$ is the constant that makes the active $F _ { k }$ sum to $B ,$ and $C = F _ { 0 }$ when $\gamma \geq 1$ and length 0 is active.

(b) In every round andfor every cache $S _ { t }$ chosenfrom $I _ { t }$ with $| S _ { t } | \leq B ,$ , including afixed allocation $\{ F _ { k } \}$ with $\begin{array} { r } { \sum _ { k } F _ { k } \overset { \cdot } { \le } B } \end{array}$ and a pair of fixed allocations chosen by whether the previous round hit the cache,

$$
\pi _ { t } ( S _ { t } ^ { \star } ) \geq \pi _ { t } ( S _ { t } ) .
$$

$I f \pi _ { t } ( k , a _ { k , s } )$ is the same in every round, then $S _ { t } ^ { \star }$ selects the same ranks $( k , s )$ in every round.   
It is therefore onefixed allocation, and the bestfixed allocation ties with it.

Example. The surrogate problem of part (a) can drop a length. Take $\alpha = 0 . 6 , \rho = 1 , \gamma = 4$ and $B = 1 0$ . The series of part (a) over all lengths $0 , \ldots , \scriptscriptstyle 4$ 4 captures hit mass 0.535. The same series, rescaled over the lengths {0, 1, 2, 4}, captures 0.567 and is the maximizer.

Proofof(a). We first show that a maximizer exists and that every active $F _ { k }$ exceeds 1. We then show that the active lengths have equal marginal gains, which gives the stated series. Finally, we show that the active lengths are a set of M lengths of largest $P _ { \alpha } ( k )$ for some M.

Each $F _ { k }$ ranges over the compact set $\{ 0 \} \cup [ 1 , B ]$ and the objective is continuous, so a maximizer exists. Because $\alpha \in ( 0 , 1 )$ , every $P _ { \alpha } ( k )$ is positive. At a maximizer the constraint $\textstyle \sum _ { k } F _ { k } \leq B$ binds, because raising an active $\dot { F } _ { k } ,$ , or setting one $F _ { k } = B$ when no length is active, strictly raises the objective. An active length with $F _ { k } = 1$ adds no hit mass. If another length is active, moving this unit to it raises the objective, and otherwise the binding constraint gives $\bar { F } _ { k } = B > 1$ . Hence every active $F _ { k }$ exceeds 1.

The marginal gain of $P _ { \alpha } ( k ) ( 1 - F _ { k } ^ { - \rho } )$ in $F _ { k }$ is $P _ { \alpha } ( k ) \rho F _ { k } ^ { - \rho - 1 }$ . If two active lengths had unequal marginals, moving a small amount of allocation toward the larger marginal would raise the objective. Hence the marginals of the active lengths equal a common value $\mu .$ , which gives

$$
P _ { \alpha } ( k ) \rho F _ { k } ^ { - \rho - 1 } = \mu \quad \Longrightarrow \quad F _ { k } \propto P _ { \alpha } ( k ) ^ { 1 / ( 1 + \rho ) } ,
$$

and substituting $P _ { \alpha } ( k )$ from Assumption 2 gives the stated series.

An active length has $F _ { k } > 1$ , so $P _ { \alpha } ( k ) \rho > \mu$ , and a length with $P _ { \alpha } ( k ) \rho \leq \mu$ is therefore inactive. The converse fails: in the example above, $P _ { \alpha } ( 3 ) \rho > \mu$ and the maximizer still leaves length 3 inactive. If k is active, j is inactive and $P _ { \alpha } ( j ) > ^ { \prime } P _ { \alpha } ( k )$ , then setting $F _ { j } : = F _ { k }$ and $F _ { k } : = 0$ keeps $\textstyle \sum _ { k } F _ { k }$ and raises the objective by $( P _ { \alpha } ( j ) - P _ { \alpha } ( k ) ) ( 1 - F _ { k } ^ { - \rho } ) > 0$ . Hence the active lengths are a set of M lengths of largest $P _ { \alpha } \dot { ( } k )$ for some M. For each $\dot { M } \in \{ 1 , \dots , \gamma + 1 \}$ , restricting the series of part (a) to such a set gives one candidate allocation. A maximizer is a best one among the candidates that keep every $F _ { k } \overset { \overline { { \mathbf { \Lambda } } } } { \geq } 1$ □

Proofof(b). Each $S _ { t } = S ( I _ { t } )$ is feasible in Eq. 7, so $\pi _ { t } ( S _ { t } ^ { \star } ) \geq \pi _ { t } ( S _ { t } )$ pointwise, and the inequality also holds in expectation over rounds. $\operatorname { I f } \pi _ { t } ( k , a _ { k , s } )$ does not depend on t, the order that defines $S _ { t } ^ { \star }$ selects the same ranks $( k , s )$ in every round. Because $\pi _ { t } ( k , a _ { k , s } )$ is nonincreasing in s and ties prefer smaller s, the selected ranks at length k are the top $F _ { k }$ ranks, which form a fixed allocation. □

Remark 2 (Relation of the proposition to SSD). The allocation of part (a) is the geometric series of SSD restricted to the active lengths, and it equals the fan-out of SSD’s Theorem 12 when every length $k = 0 , \ldots , \gamma$ is active. SSD derives the equal-marginal condition of part (a) in its Theorem 18, but its statement has no inactive lengths. SSD fills each length with the top anchors of the draft model’s logits. The resulting cache is chosen from $I _ { t } ,$ so part (b) covers it. This cache can fall strictly below $S _ { t } ^ { \star }$

Remark 3 (Estimated scores). In serving, the draft process ranks outcomes by an estimate $\hat { \pi } _ { t }$ of $\pi _ { t }$ . For any cache $\hat { S } _ { t }$ of at most B outcomes, let

$$
\nu _ { t } : = \operatorname* { m a x } _ { | S | \leq B } \big | \pi _ { t } ( S ) - \hat { \pi } _ { t } ( S ) \big | , \qquad \zeta _ { t } : = \operatorname* { m a x } _ { | S | \leq B } \hat { \pi } _ { t } ( S ) - \hat { \pi } _ { t } ( \hat { S } _ { t } ) .
$$

Then $\pi _ { t } ( S _ { t } ^ { \star } ) - \pi _ { t } ( \hat { S } _ { t } ) \leq 2 \nu _ { t } + \zeta _ { t }$ in every round. The bound holds because $\nu _ { t }$ bounds the estimation error at both $S _ { t } ^ { \star }$ and $\hat { S } _ { t }$ , and $\zeta _ { t }$ bounds the estimated gap between $S _ { t } ^ { \star }$ and $\hat { S } _ { t }$

## F.3 LOSSLESS VERIFICATION

This subsection shows why the output stays lossless for every draft, including one that anchor replacement rewrites.

JetSpec++ determines every draft token before verification. Each token is an argmax of the draft model, from the initial draft $J _ { 0 }$ (also the fallback block of a miss) or a pass of Eq. 8, or it is the decoder’s token that anchor replacement writes. The noise tokens that $J _ { 0 }$ reads are drawn independently of the target’s draws in the round that verifies the block.

Given the draft, the proposal at position i is the point mass $q _ { i } = \delta _ { d _ { i } }$ . Let $p _ { i } = p ( \cdot \mid x , d _ { < i } )$ be the target’s distribution at the sampling temperature. With this proposal, the acceptance rule of Section 3 accepts $d _ { i }$ with probability $p _ { i } ( d _ { i } )$ . On a rejection, the rule samples from $p _ { i }$ with $d _ { i }$ removed and renormalized. When $p _ { i } ( d _ { i } ) = 1$ , a rejection has probability 0 and the rule never uses this residual. This rule is equivalent to drawing $v \sim p _ { i }$ , accepting $d _ { i }$ if and only $\mathrm { i f } \ v \ = d _ { i }$ , and committing v otherwise. DARTree (Li et al., 2026) describes this exact-match verification for DFlash (Chen et al., 2026). Every committed token is thus the target’s own draw, at any temperature and whichever model produced the draft token.

Using the draft model’s softmax as $q$ for an argmax token would break this guarantee, because the token is not drawn from that softmax. For example, if the softmax puts 0.9 on $d _ { i }$ and the target puts 0.6 on it, the rule with that $q$ commits $d _ { i }$ with probability $2 / 3$ instead of 0.6.