# LooM: Loop-Invariant Memory for Causal Encoder–Decoder Language Models

Lele Yang<sup>∗</sup> Muxi Diao<sup>∗</sup> Zhonghao Yan Jiankang Chen<sup>†</sup> Bin Wen<sup>†</sup> Tingting Gao Wentao Zhang Kongming Liang Zhanyu Ma Weiran Xu<sup>‡</sup>

Beijing University of Posts and Telecommunications Zhongguancun Academy

Peking University Kling Team

{yang\_happy, dmx}@bupt.edu.cn

## Abstract

Looped Transformers increase effective depth by re-executing shared layers, but three limitations hinder their adoption: memory grows with the loop count, compute can only be adjusted in whole-block steps, and the loop count is hard to change after training. We propose LooM (Loop-invariant Memory), which addresses each in turn. In a causal encoder– decoder (CED), LooM loops only the decoder layers after the first, which read a global memory written once per token, and each of them publishes the key-value (KV) entry for a token only on its first visit, so the cache does not grow with the loop count. These layers loop independently, providing every budget from 3 to 21 layer executions per token, and one model trained on randomly sampled budgets covers all of them without a router. With enough loops, LooM outperforms a single-pass baseline trained on the same data on language modeling, zero-shot accuracy and long-context perplexity, while using the same cache size. Our analysis further shows that the training budget distribution decides what extra depth buys. These results suggest that a CED is a natural place to loop: memory is written once, and depth becomes a budget chosen at deployment.

## 1 Introduction

Looped Transformers apply the same stack of layers repeatedly to each token’s hidden state, increasing effective depth without adding parameters (Dehghani et al., 2019; Geiping et al., 2025; Zhu et al., 2025b; He et al., 2026; Chen et al., 2026a). This added depth improves performance: looped models rival much larger standard models (Zhu et al., 2025b) and scale test-time computation in latent space (Geiping et al., 2025). However, three limitations hinder their wider adoption. (1) Memory grows with the loop count. In a looped Transformer, the same layers both build the context and refine each token’s state, so every loop writes new key–value (KV) entries. With a standard KV cache, as in Ouro, memory therefore grows linearly with the number of loops (Zhu et al., 2025b; Conchello Vendrell et al., 2026). (2) Compute can only be adjusted coarsely. Looped language models typically repeat a whole block of layers (Geiping et al., 2025; Zhu et al., 2025b; Bae et al., 2025b; Liang et al., 2026), so budgets come in multiples of that block and none lies below one full pass. (3) The loop count is hard to change after training. A model trained at one loop count degrades when run with fewer loops (Section 5.1). Varying compute therefore usually requires a learned router or halting policy (Graves, 2016; Banino et al., 2021; Raposo et al., 2024; Bae et al., 2025b; Zhu et al., 2025b) or loop conditioning (Jeddi et al., 2026).

To address these limitations, we propose LooM (Loop-invariant Memory), with one design for each. (1) In a causal encoder–decoder (CED) (DeepSeek-AI, 2026), the encoder and the first decoder layer write the global memory once per token, and LooM loops only the remaining decoder layers, which we call reader layers. Each reader layer writes (publishes) a token’s local KV entry only on its first visit, so the cache does not grow with the loop count (Figure 1). (2) Each reader layer loops independently, zero to four times, so one model offers every budget from 3 to 21 decoder calls (layer executions) per token; publish-once keeps the cache unchanged even when a layer is skipped. (3) During training, each update samples a budget and then a random path, that is, how many times each reader layer runs, with that many calls, so the model learns to run at every budget. At inference we fix one path per budget, chosen on validation data, and need no router.

Trained on 5B tokens, the resulting 204Mparameter model beats a single-pass model trained on the same data from 11 calls on. At 21 calls its test cross-entropy is 0.020 nats lower and its zeroshot average 0.79 points higher at the same cache size. (Table 2).

![](images/44af33198f660e7dc628cb9d9643b653c691ac6de9448d3bc72fb272e3a1fc1f.jpg)  
Figure 1: Loop-invariant memory in LooM. (a) Decoder layer 1 (D1) writes global memory once per token, before any repetition; each of D2–D6 then runs d ∈ {0, . . . , 4} times (D3 and D5 at least once). Squares are decoder calls on Random-Depth’s locked paths at three budgets. Every visit re-reads global memory (left), but only a layer’s first visit publishes the token’s local KV entry (right); publishing arrows are drawn for the 21-call path only. (b) D2, standing for any of D2–D6: a later token reads one local entry per earlier token however often the layer ran, so the cache (after a 128-token prompt) is the same on every path. CSA2 modes are defined in Section 3.1.

Our contributions are:

• Loop-invariant memory. We loop only the reader layers of a CED and publish each local KV entry once, so the cache does not grow with the loop count, even when layers loop independently or are skipped (Section 3.2).

• Layer-level loops in one router-free model. Each reader layer loops independently, giving budgets from 3 to 21 calls in steps of one layer execution; with Random-Depth training, one checkpoint serves them all (Section 5.1).

• An analysis of how to spend the budget. The training budget distribution decides what extra depth buys (Section 5.2), repeated reads reretrieve memory (Section 5.4), and static paths nearly match a learned router (Section 5.5).

## 2 Related Work

Looped models and their KV cache. Looped language models reuse shared layers to add depth without adding parameters (Geiping et al., 2025; Zhu et al., 2025b; He et al., 2026; Chen et al., 2026a; Bae et al., 2025a; Chen et al., 2026b). Keeping a standard KV cache across iterations, as Ouro does, makes memory grow linearly with the loop count (Zhu et al., 2025b; Conchello Vendrell et al., 2026). Mixture-of-Recursions can reuse the first recursion’s KV (Bae et al., 2025b), MELT updates one cache per layer through a learned gate (Conchello Vendrell et al., 2026), and concurrent work trains a looped stack to tolerate a shared cache or compresses the per-loop caches (Prairie et al., 2026; Huang et al., 2026; O’Neill and Reid, 2026). These methods obtain a shared cache through training, fine-tuning or compression; in LooM it follows from where a CED writes memory, with no gate. The closest design, the concurrent Asymmetric Looped Transformer, also pairs a single-pass causal encoder with a weight-tied recurrent decoder (Liang et al., 2026), but it loops the decoder as a unit for a fixed number of iterations and recomputes each local KV entry at every iteration, whereas our reader layers take independent depths and publish each entry once. We claim a CED execution rule, not the general idea of sharing KV across loops.

Reducing KV storage without loops. Crosslayer KV sharing saves storage across layers rather than loops (Sun et al., 2024; Wu and Tu, 2024; Brandon et al., 2024). Early-exit and layerskipping methods must reconstruct the state of skipped layers (Schuster et al., 2022; Del Corro et al., 2023; Elhoushi et al., 2024) or, as BUDDY does, leave the resulting cache misses open (Zhou et al., 2026). Under publish-once, a skipped reader layer publishes nothing, and later tokens read the most recent preceding publication.

Setting a token’s depth. Adaptive computation learns halting, exits or token routing (Graves, 2016; Dehghani et al., 2019; Banino et al., 2021; Raposo et al., 2024; Bae et al., 2025b; Zhu et al., 2025b; Geiping et al., 2025; Heakl et al., 2025). BUDDY and LoopFormer serve several budgets from one checkpoint, through a per-step decision module or by conditioning each iteration on time and step size (Zhou et al., 2026; Jeddi et al., 2026). LooM needs neither at inference: like stochastic depth, Layer-Drop and slimmable networks (Huang et al., 2016; Fan et al., 2020; Yu and Huang, 2019), Random-Depth samples each layer’s depth during training, and a learned planner serves only as a comparison (Section 5.5).

## 3 Looping the Read Stage

LooM loops only the decoder layers that read memory. This section follows one token through that loop, shows why repeated reads add no persistent memory, and covers training and inference.

## 3.1 A Token’s Path Through the CED

Our backbone adopts the causal encoder–decoder structure of DeepSeek-V4.1-Flash (DeepSeek-AI, 2026), using dense FFNs. It also keeps that model’s mHC and CSA2 components, and each attention layer runs in one of CSA2’s three modes (DeepSeek-AI, 2026). Full computes the layer’s own KV entries and Top-K retrieval support; Reindex reuses the KV but recomputes the support; Reuse takes both from the latest layer that computed them. Among the six encoder layers, layers 1–2 use sliding-window attention, layers 3 and 5 run in Full mode, and layers 4 and 6 run in Reuse mode. Among the six decoder layers, layer 1 runs in Full mode, layers 3 and 5 run in Reindex mode, and layers 2, 4, and 6 run in Reuse mode. Figure 1 labels every block with its mode, and Appendix A gives the remaining architecture details.

Decoder memory has two forms. Global attention reads an uncompressed KV bank and indexer keys, both built by the encoder and the first decoder layer from the final encoder representation. Local attention reads a causal sliding-window history of width 128, which each decoder layer keeps for itself. Once the global bank is built, decoder layers 2–6 act as reader layers: at every visit they attend to the same bank and update their hidden states. Layers 3 and 5 also reselect Top-128 entries at each visit but add no global entries.

A token’s path is $d = ( 1 , d _ { 2 } , \dotsc . . . , d _ { 6 } )$ . Decoder layer 1 runs once, and each of decoder layers 2–6 runs zero to four times in a row. Decoder layers 3 and 5 must run at least once because their Reindex mode has to produce new Top-K support. These constraints give 2,000 legal paths. We loop each reader layer separately, rather than the decoder as a unit, for three reasons. Looping the unit allows only four budgets (6, 11, 16 and 21 calls), whereas per-layer depths reach every budget from 3 to 21. A depth of zero skips a layer, which gives the lowest budgets, such as the 3-call path (1, 0, 1, 0, 1, 0). Within a budget, calls can also be split unevenly across layers. All paths share weights and use no intermediate language-model head, no extra embedding, and no gradient detachment at recurrence boundaries. The r-th execution is

$$
s _ { \ell } ^ { ( r ) } = F _ { \ell } \big ( s _ { \ell } ^ { ( r - 1 ) } ; M _ { \le t } , H _ { \le t , \ell } , I _ { \ell } ^ { ( r ) } \big ) ,\tag{1}
$$

where $s _ { \ell . } ^ { ( 0 ) }$ is the preceding layer’s output, $M _ { \leq t }$ is the global memory already completed through token t, $H _ { \leq t , \ell }$ is local history, and $I _ { \ell } ^ { ( r ) }$ is current retrieval support. One decoder call is one layer execution for one token:

$$
\begin{array} { r } { C ( d ) = 1 + \sum _ { \ell = 2 } ^ { 6 } d _ { \ell } \in [ 3 , 2 1 ] . } \end{array}\tag{2}
$$

A budget is an allowed number of calls: decoder layer 1’s single execution plus the loop counts of decoder layers 2–6. A static path is the layer-depth tuple used by every token at that budget. A cap is a per-token upper bound on calls, and the realized count can be lower. A suffix consists of decoder layers 2–6. It follows the prefix shared by every path: the encoder and decoder layer 1.

## 3.2 Publish Once: What a Later Token Sees

Publish-once gives the two memory pathways separate write rules. Global memory is complete before any decoder layer starts looping, so no loop ever writes to it. Local memory holds one KV entry per token in each decoder layer, and each execution of a layer on a token is a visit.

Consider a token t with $d _ { 2 } = 4$ , so that decoder layer 2 visits t four times, and assume that its global memory $M _ { \leq t }$ is complete before the first visit. All four visits update the hidden state of t. The first visit also publishes the token’s local KV entry, that is, writes it to the cache where later tokens can read it. The other three visits recompute t’s query, which can shift its attention over the stored keys and values, but write no new entry. When token t + 1 later reads its history at decoder layer 2, it therefore finds one local KV entry for token t, whether decoder layer 2 visited t once or four times. Looping changes the hidden state used by the current token’s computation and leaves unchanged the number of entries that later tokens can read. A skipped decoder layer publishes no entry of its own and reuses the most recent preceding publication; an indexing layer may change its retrieval support at each visit without writing new entries to global memory.

Cache size. After a prefix of $T$ tokens, the persistent cache holds two parts: the global banks, written once per token by the encoder and decoder layer 1, and a sliding window of at most $W = 1 2 8$ local entries in each of the 12 layers. Ignoring data type and metadata overhead, its size is

$$
\underbrace { \sum _ { j } n _ { j } ( T ) k _ { j } } _ { \mathrm { g l o b a l b a n k s } } + \underbrace { \sum _ { \ell = 1 } ^ { 1 2 } \operatorname* { m i n } ( T , W ) k _ { \ell } ^ { \mathrm { l o c } } } _ { \mathrm { l o c a l } \mathrm { w i n d o w s } } ,\tag{3}
$$

where bank $j$ stores $n _ { j } ( T )$ positions of $k _ { j }$ bytes and layer ℓ stores $k _ { \ell } ^ { \mathrm { l o c } }$ bytes per local entry. Neither term depends on the path: global memory is complete before any loop, and each layer publishes one local entry per token however often it runs. A 3-call and a 21-call path therefore store the same number of bytes. The stored values depend on the execution path, since a different path changes what later layers publish, but the cache layout and the bound do not.

A single cache layout therefore serves every compute budget. Rebuilding memory after looped reading would instead require running the looped decoder layers over every prompt position (Appendices C and D).

## 3.3 Training One Model for Every Budget

Publish-once lets the loop count vary without changing the cache layout, so we train a single model across loop counts with Random-Depth, and compare it against Self-Distilled, a control that adds a full-depth teacher. Both sample depths independently of token content, unroll each update to its largest sampled depth, and mask the unexecuted repetitions. Losses and schedules for both recipes are in Appendix A.

Random-Depth samples a budget and then a path of that budget, and it neither trains nor executes a router. It draws

$$
B \sim \pi , \qquad d \sim \operatorname { U n i f } \{ d : C ( d ) = B \} ,\tag{4}
$$

where π puts $1 / 1 6$ of its mass on budgets 3–10, 3/16 on 11–16 and 3/4 on 17–21, uniformly within each interval. The schedule is fixed for the whole run and gives 17.19 expected calls per token: three quarters of sampled paths fall in the 17–21 interval against 6.25% at ten calls or fewer.

Self-Distilled samples a legal budget and path, runs that suffix alongside the full-depth suffix, and adds a stop-gradient Kullback–Leibler (KL) divergence from the full-depth distribution.

## 3.4 Deployment with One Static Path per Budget

After training, inference needs one path for each number of calls. For each budget $B \in \{ 3 , \ldots , 2 1 \}$ , we select one static path on validation data, lock it before testing, and use it for all tokens.

The search runs in three stages, on 16, 128 and 512 validation samples. The first stage scores all 2,000 paths and keeps four candidates per exact cost, and the next two stages narrow these to two candidates and then one. Because only the first stage is exhaustive, a locked path is the best found, not a guaranteed optimum (Appendix A).

## 3.5 A Learned Planner for Comparison

To test whether token-dependent allocation helps, we ablate Random-Depth (Section 5.5) by replacing its static execution path with a learned planner. The planner reads decoder layer 1’s normalized output $u _ { t }$ and emits one categorical score group per reader layer,

$$
z _ { t } = A u _ { t } + b ,\tag{5}
$$

so a token’s depths are fixed once, after the prefix is built and before any repetition.

The planner is trained with a straight-through estimator. For hard choice a and relaxed probabilities $p ,$ candidate depth r carries the weight $w _ { r } = 1 [ r = a ] + p _ { r } - \mathrm { s g } ( p _ { r } )$ , so the forward pass runs a single path while the backward pass compares the candidates at that layer. The training objective is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \mathcal { L } _ { \mathrm { i n d e x } } - \beta \mathcal { H } ,\tag{6}
$$

where $\mathcal { L } _ { \mathrm { i n d e x } }$ is the indexer KL the architecture already carries and H is the mean routing entropy. The entropy bonus starts at $\beta ~ = ~ 0 . 0 0 5$ and, together with exploration noise, keeps the policy from hardening too early; both anneal over the final tenth of training.

Eq. (6) rewards only language-modeling quality, so training teaches the planner where to spend depth, and the compute budget is imposed at inference. There, an exact dynamic program selects the highest-scoring legal path under a per-token cap; without a cap, we call the planner’s own per-token choice native routing.

<table><tr><td>Model</td><td>Decoder depth during training</td></tr><tr><td>Single-pass</td><td>Always 6 calls; no loop</td></tr><tr><td>LooM, Fixed4</td><td>Always 21 calls: decoder layers 2–6 run four times each</td></tr><tr><td>LooM, Random-Depth</td><td>A deep-heavy random budget (75% at 17–21 calls), then a random path</td></tr><tr><td>LooM, Self-Distilled</td><td>A random path learns from a 21-call teacher</td></tr><tr><td>LooM, Planner</td><td>A learned router sets each layer&#x27;s depth from the token</td></tr><tr><td>Grouped memory</td><td>As Fixed4, but memory rebuilt at decoder layers 1, 3 and 5</td></tr></table>

Table 1: Models and controls. LooM models share the looped read stage and publish-once cache; grouped memory writes memory inside the loop. Only the planner routes on token content.

## 4 Experimental Setup

Models and controls. Table 1 lists every run. All looped models except grouped memory are LooM models: they share the looped read stage and the publish-once cache of Section 3 and differ only in how decoder depth is set during training. Random-Depth is our main multi-budget model (Section 3.3). Each remaining run isolates one factor against it. Single-pass and Fixed4 hold the decoder at 6 and 21 calls, separating the effect of looping from the effect of depth sampling; Self-Distilled changes which paths are trained, so it tests how the training distribution shapes low budgets; the planner (Section 3.5) replaces the static path with a learned router, so it measures what tokendependent allocation adds; and grouped memory keeps Fixed4’s schedule but writes context memory at several decoder layers, so it isolates where memory is written.

Training. All runs share a 204M-parameter backbone and are trained from scratch on 5B valid target tokens of FineWeb-Edu (Penedo et al., 2024) with the same seed, batching and optimizer schedule, so every comparison is matched in parameters and training data (Appendix K; configuration in Table 6). They are not matched in training compute: a looped update runs more decoder calls than a single-pass one, and the looped runs used 2.9– 4.7× the single-pass run’s GPU-hours.

Language modeling. We report token-weighted test CE on 2,048 held-out sequences, with paired

document-bootstrap intervals over 7,836 segments.   
Perplexity (PPL) is exp(CE).

Downstream and long-context evaluation. The pre-specified standard suite is LAMBADA (Paperno et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), plus WikiText-2 word perplexity (Merity et al., 2017); we fixed the metrics and the macro average before scoring (Appendix H). Scoring uses lm-evaluation-harness 0.4.13 (Gao et al., 2023). After a shared continuation to 64K context, Random-Depth, the singlepass model and the planner are evaluated on PG19 (Rae et al., 2020) and three RULER retrieval tasks (Hsieh et al., 2024) (Appendix I).

Cost. We report single-sequence decoding latency (batch size one), where the per-token cost of extra calls is most visible, measured with a reference implementation on one NVIDIA H800 GPU. Matrix FLOPs count linear projections, attention $Q K ^ { \top }$ and AV , indexer/router projections and vocabulary head operations from operand shapes, including tile padding and masked bank rows; they exclude normalization, softmax, sorting, indexing, cache copies and pointwise reductions. For Fixed4’s 21-call latency we reuse Random-Depth’s path shape; on identical paths, checkpoints differ by at most 1.4% (Appendix C). Following DeepSeek-V4.1-Flash (DeepSeek-AI, 2026), prefill uses sliding-window attention (SWA) replay over the most recent 128 prompt tokens, which approximately reconstructs the decoder’s SWA KV cache without running the decoder over the prompt. As a result, 65K-token prefill grows only 3.2% from 3 to 21 calls, so the budget mainly affects decode.

## 5 Results and Analysis

This section first reads the main results (Table 2): quality, cache size and downstream accuracy. It then examines the role of the training depth distribution, the compute cost of additional loops, what the repeated reads contribute, whether tokens need different depths, and long-context evaluation after 64K continuation. Except where we state native routing, every model uses one validation-locked static path per budget.

<table><tr><td></td><td colspan="2">Cache</td><td>Test</td><td colspan="6">Zero-shot accuracy (%) ↑</td><td></td><td colspan="2">PPL↓</td></tr><tr><td>Model</td><td>Calls</td><td>(KiB)</td><td>CE↓</td><td>LAMB.</td><td></td><td></td><td>HellaS. PIQA WinoG.</td><td>ARC-E</td><td>ARC-C</td><td>Avg</td><td>LAMB.</td><td>Wiki-2</td></tr><tr><td>Single-pass (no loop)</td><td>6</td><td></td><td>364.1 3.0506</td><td>20.1</td><td>30.4</td><td>62.6</td><td>50.7</td><td>51.6</td><td></td><td>25.4 40.15</td><td>180.4</td><td>43.20</td></tr><tr><td>LooM, Fixed4</td><td>21</td><td></td><td>364.1 3.0465</td><td>21.2</td><td>30.3</td><td>61.3</td><td>50.7</td><td>52.5</td><td>24.840.14</td><td></td><td>173.2</td><td>42.75</td></tr><tr><td rowspan="2">LooM, Self-Distilled</td><td>6</td><td>364.1</td><td>3.0640</td><td>20.8</td><td>30.5</td><td>61.8</td><td>51.5</td><td>51.0</td><td></td><td>24.3 39.98</td><td>168.6</td><td>43.60</td></tr><tr><td>21</td><td>364.1</td><td>3.0456</td><td>20.9</td><td>30.7</td><td>62.2</td><td>51.4</td><td>50.8</td><td></td><td>23.939.98</td><td>154.1</td><td>42.59</td></tr><tr><td>LooM, Planner†</td><td>17.96</td><td>364.1</td><td>3.0355</td><td>21.4</td><td>31.0</td><td>61.6</td><td>51.9</td><td>53.0</td><td>24.1 40.52</td><td></td><td>153.9</td><td>42.63</td></tr><tr><td rowspan="4">LooM, Random-Depth</td><td>6</td><td></td><td>364.1 3.1941</td><td>15.8</td><td>30.5</td><td>62.6</td><td>51.4</td><td>51.8</td><td></td><td>25.3 39.57</td><td>292.6</td><td>52.47</td></tr><tr><td>11</td><td>364.1</td><td>3.0465</td><td>21.3</td><td>31.1</td><td>62.8</td><td>51.5</td><td>52.8</td><td></td><td>25.6 40.86</td><td>148.3</td><td>43.03</td></tr><tr><td>16</td><td>364.1</td><td>3.0316</td><td>21.6</td><td>31.2</td><td>62.8</td><td>52.3</td><td>53.1</td><td></td><td>25.4 41.09</td><td>138.2</td><td>42.16</td></tr><tr><td>21</td><td>364.1</td><td>3.0307</td><td>21.6</td><td>31.2</td><td>62.7</td><td>52.2</td><td>52.7</td><td></td><td>25.2 40.94</td><td>134.4</td><td>42.04</td></tr></table>

Table 2: Main results. All models are 204M-parameter CEDs trained on the same data and updates; LooM rows differ only in training recipe and use one validation-locked static path per budget (<sup>†</sup>native routing, 17.96 calls on average). Cache is measured after a 128-token prompt; CE is in nats per token. Zero-shot scores are accuracy (%), normalized for HellaSwag and ARC-C. As several tasks are near chance, only CE, Avg and PPL are marked (bold red: best; underlined blue: second). CE at all budgets: Figure 2.

![](images/d9d161febe2f4fda1f22e8944fbe7c25e9d9edc6b671f116162a446747dd385e.jpg)  
Figure 2: Test CE (nats per token; lower is better) of LooM models on validation-locked static paths; the star is the planner’s native routing. (a) Full range. (b) Zoom: Random-Depth and Self-Distilled both cross the six-call single-pass line between 10 and 11 calls.

## 5.1 Main Results: One Checkpoint across Budgets

We first ask whether LooM, by repeating the read stage, improves on one ordinary forward pass, and whether a single checkpoint can do so across loop counts. Table 2 gathers the answer. Two controls frame it: the single-pass model shows what the same backbone reaches at six calls, and Fixed4 is trained only on the full 21-call loop and is evaluated on shortened paths. One LooM checkpoint, trained with Random-Depth, covers 3–21 calls and beats the six-call single-pass model starting at 11 calls, by 0.020 nats at 21.

Quality. The crossing lies between 10 and 11 calls (Figure 2b), and the 0.020-nat gain at 21 calls is a 2.0% reduction in perplexity. CE falls by 0.0158 nats from 11 to 21 calls and is nearly flat from 16 calls on, so most of the gain from extra budget arrives early. Below 11 calls both multibudget recipes trail the single-pass model, which is trained for exactly six. Self-Distilled is the stronger recipe at that low end, at the cost of a full-depth teacher during training, and Random-Depth from 11 calls on.

One checkpoint versus a fixed loop. Fixed4 shows what training across loop counts adds. Trained only on its 21-call loop, it degrades when shortened (Figure 2a), and on that same path Random-Depth is 0.0158 nats better ([−0.0164, −0.0151]). Random-Depth at 11 calls already matches Fixed4 at 21: both reach 3.0465, a difference of +0.00003 ([−0.00065, +0.00071]).

Cache. Every row of Table 2 stores the same 364.1 KiB after a 128-token prompt, across models and across Random-Depth’s budgets, as Eq. (3) predicts. Section 5.3 covers their compute cost.

Downstream accuracy. The zero-shot columns follow the CE frontier. At 21 calls Random-Depth improves the six-task average over the single-pass model by 0.79 points $( [ + 0 . 0 5 , + 1 . 5 4 ] )$ ) and lowers LAMBADA perplexity from 180.4 to 134.4. At 6 calls it trails the single-pass model; at 11 calls it is 0.71 points above Fixed4 at 21 ([−0.04, +1.43]); and the average peaks at 16 calls. Several tasks are near chance at this scale, so we read only the average and the perplexities.

## 5.2 The Training Distribution Decides What Extra Depth Buys

Spreading the training samples evenly across budgets yields a model whose quality barely responds to depth. Random-Depth concentrates them on the deployable range and gains twelve times more from the same extra calls.

Random-Depth’s schedule is deep-heavy (Section 3.3). Its quality is likewise strong at high budgets and weak at low ones (Figure 2). The correspondence alone does not establish cause: weak low-budget paths could reflect a cost intrinsic to shallow execution rather than too few shallow training samples. Self-Distilled and a uniform-budget control separate the two by changing the trained paths (Appendix B).

Shallow execution carries no intrinsic penalty. At three calls Self-Distilled reaches 3.1570 where Random-Depth reaches 4.3787 (Figure 2a), on the same architecture, data and updates; a uniformbudget control shows the same effect (Table 7). The shortfall at low budgets therefore follows from the training distribution.

What a flat distribution costs is the response to depth. Early in training, the uniform-budget control improves by only 0.012 nats from 6 to 21 calls, against 0.145 for Random-Depth’s deepheavy schedule—twelve times less return on the same fifteen extra calls (Table 7); a model whose quality does not respond to depth cannot serve a compute budget at all. Random-Depth is best at 16 and 21 calls here, and after full training it leads from 11 calls on—the only region in which any recipe overtakes the single-pass baseline (Section 5.1). Self-Distilled stays within 0.015 nats of the best recipe at every budget, but it runs a fulldepth teacher at every update, at 4.73× the singlepass run’s GPU-hours against Random-Depth’s 2.94×, and its advantage does not reach the downstream suite (Appendix H).

## 5.3 Extra Loops Cost Compute, Not Persistent Memory

Extra loops increase compute while leaving cache size unchanged; rebuilding memory inside the loop enlarges the cache without improving quality.

The cache is independent of the loop count at longer prompts too: after 512- and 2,048-token prompts it is 550.8 and 1,297.8 KiB, respectively, on both the 3- and 21-call paths.

The cost of extra loops therefore falls on compute. At batch size one, reference decode latency grows from 44.74 ms at 3 calls to 112.54 ms at 21 calls, whereas 65K-token prefill grows by only 3.2% (Table 3; Appendix C). Relative to the six-call single-pass model, Random-Depth’s 21-call gain requires 99.1% higher reference decode latency (56.52 to 112.54 ms) and 73.3% more matrix FLOPs (6.305 to 10.929 GFLOP per token). Because Random-Depth reaches Fixed4’s 21-call quality at 11 calls, this smaller budget reduces latency by 33.3% and matrix FLOPs by 28.2% (Table 2; Appendix C). All of these measurements come from a reference implementation without optimized kernels.

Since LooM keeps the cache fixed, the remaining question is whether rebuilding memory inside the loop improves quality enough to justify its extra storage. The grouped-memory variant tests this by rebuilding decoder memory at decoder layers 1, 3 and 5 under Fixed4’s schedule. It increases cache size by 1.58–1.88× over 1,024–16,384-token prompts and is slightly worse in test CE than LooM on the same schedule (3.04728 against 3.04646, a difference of +0.00082, [+0.0002, +0.0014]; Table 4; Appendix D).

## 5.4 What Repeated Reads Do

Repeated reads both refine the hidden state and reretrieve memory: freezing the first visit’s retrieval selections costs 0.0114 nats.

Re-executing a decoder layer can change a token’s computation in two ways. Each visit updates the token’s hidden state, and each visit selects which memory entries to read. We measure these two effects separately.

Repeated reads refine the hidden state. The first visit of each layer makes the largest update, and later visits make smaller ones, in Random-Depth and the planner alike (Figure 4b).

Repeated reads also re-retrieve memory. For decoder layer 3, the first- and fourth-visit Top-K sets have Jaccard similarity 0.54 overall and 0.30 beyond position 2,048, and 32% of the fourth visit’s global attention falls on entries the first visit did not select. To test whether this re-retrieval matters for quality, we freeze each layer’s firstvisit selections and reuse them on all later visits. This intervention costs Random-Depth 0.0114 nats ([0.0106, 0.0122]), so later visits gain quality from the entries they newly select.

<table><tr><td rowspan="2">Calls</td><td rowspan="2">Path</td><td colspan="3">Decode, 128-token prompt</td><td colspan="2">Prefill (ms)</td></tr><tr><td>ms/token</td><td>GFLOP/token</td><td>Test CE↓</td><td>4,096</td><td>65,536</td></tr><tr><td>3</td><td>(1,0,1,0,1,0)</td><td>44.74</td><td>5.381</td><td>4.3787</td><td>82.2</td><td>3,134.6</td></tr><tr><td>6</td><td>(1,1,1,1,1,1)</td><td>56.52</td><td>6.305</td><td>3.1941</td><td>94.3</td><td>3,145.6</td></tr><tr><td>11</td><td>(1,2,2,2,2,2)</td><td>75.07</td><td>7.847</td><td>3.0465</td><td>112.2</td><td>3,171.6</td></tr><tr><td>16</td><td>(1,4,3,3,2,3)</td><td>93.15</td><td>9.387</td><td>3.0316</td><td>128.4</td><td>3,194.5</td></tr><tr><td>21</td><td>(1,4,4,4,4,4)</td><td>112.54</td><td>10.929</td><td>3.0307</td><td>147.3</td><td>3,233.7</td></tr></table>

Table 3: Cost of Random-Depth’s locked paths at batch size one on one NVIDIA H800 GPU with a Python reference implementation (medians of 24 decode and five prefill runs). Prefill columns use 4,096- and 65,536-token prompts. Full measurements are in Appendix C.

<table><tr><td>Prompt LooM</td><td></td><td>Grouped</td><td>Ratio</td><td>LooM saves</td></tr><tr><td>1,024</td><td>0.75</td><td>1.19</td><td>1.58×</td><td>37%</td></tr><tr><td>4,096</td><td>2.19</td><td>3.94</td><td>1.80×</td><td>44%</td></tr><tr><td>16,384</td><td>7.93</td><td>14.93</td><td>1.88×</td><td>47%</td></tr></table>

Table 4: Persistent decode cache (MiB, BF16) of LooM and grouped memory under Fixed4’s 21-call schedule.

A repeated read therefore does two things at once: it applies a progressively smaller refinement to the token’s state, and it retrieves memory entries that the first visit did not select.

## 5.5 Token-Specific Depth Allocation

Do tokens need different depths? At 4K, one static path per budget nearly matches the tested linear planner; after 64K continuation, native routing has a small advantage.

Deployable routing at 4K. Compared within one checkpoint, the planner’s native routing at 17.96 calls is only 0.00051 nats better than its own static 18-call path. Table 8 reports four ways of replacing native routing, on two calibrated planner continuations (Appendix F):

• Static path gives every token one validationselected path per cap.

• Static split keeps each token’s native total but replaces its layer split with a validation-selected path of equal total.

• Mean totals keeps native splits but gives each token its document’s mean total.

• Shuffled totals permutes totals per document.

At caps 11–21, static path and static split change CE by less than 10<sup>−3</sup>. At cap 21, mean totals costs only 0.00013–0.00014 nats and shuffled totals costs 0.0017. The planner also orders equal-cost alternatives correctly only 51.4–51.5% of the time. A coarser scalar-depth planner, which predicts one depth per token shared by every reader layer, trails both the layerwise planner and Random-Depth at a comparable budget (Appendix A). Besides token dependence, the planner’s objective adds an entropy bonus and exploration noise to Random-Depth’s (Section 3.5). These results bound what a single linear planner of roughly 20K parameters contributes at 4K; they do not bound the value of adaptive computation in general.

Potential gains from allocation. A per-token allocation can help only if the benefit of extra decoder calls differs across tokens, so we first measure how this benefit is distributed. We use the gain from 11 to 21 calls, because a comparison that starts at six calls would mostly measure the shortfall of Random-Depth’s weak six-call path. We group test tokens into deciles by difficulty, defined as the target negative log-likelihood (NLL), and the 11-to-21 gain rises from 0.002 to 0.025 nats across these deciles, about twelvefold (Figure 4a; Appendix G). The target NLL depends on the correct next token, which a deployed router cannot observe when it chooses a depth. The deciles therefore show that the benefit of extra calls varies with token difficulty, but they do not show that a router can identify the tokens that benefit most from the prefix alone.

To estimate how much per-token allocation could help, we apply a hindsight policy to the same 256 test sequences. For each token, the policy selects one of the locked 11-, 13-, 16- and 21-call paths according to their target NLLs, and the selections are adjusted so that the average cost is exactly 16 calls. We then execute the mixed paths, so that later tokens read the local-KV entries actually published by the chosen paths. The mixed paths reach a CE of 2.9631, compared with 3.0203 for the static 16-call path, a gain of 0.0572 nats (Appendix J). The step that enforces the 16-call average is not guaranteed to be optimal, so this gain is a lower bound on the best hindsight gain available from these four paths. It also does not bound the gain achievable by routers in general.

After 64K continuation. The continued planner with native routing has lower PG19 perplexity than

<table><tr><td>Ctx.</td><td>Single</td><td colspan="2">RD RD – Single, 95% CI</td></tr><tr><td>4K</td><td>25.483</td><td>24.934</td><td>-0.0218 [-0.0252, −0.0184]</td></tr><tr><td>8K 16K</td><td>24.277</td><td>23.858 23.660</td><td>-0.0174[-0.0211, -0.0136]</td></tr><tr><td></td><td>23.986</td><td></td><td>-0.0137[-0.0176, -0.0097]</td></tr><tr><td>32K</td><td>23.858</td><td>23.568</td><td>-0.0122 [-0.0162, -0.0082]</td></tr><tr><td>64K</td><td>23.806</td><td>23.644</td><td>-0.0068 [−0.0110, -0.0025]</td></tr></table>

Table 5: PG19 perplexity after the shared 64K continuation. The last column is Random-Depth (RD) minus single-pass NLL in nats, with a whole-book 95% CI; negative favors RD. Bold red: best; underlined blue: second best at each length.

Random-Depth, by 0.0039 nats at 4K and 0.0064 at 64K (Section 5.6). Its native routing improves over its own static 21-call path by only 0.0008–0.0014 nats, so most of that advantage does not come from per-token routing. The continued models therefore agree with the 4K finding that one static path per budget nearly matches the planner, and add a small native-routing advantage. All of these differences are below 0.01 nats on one training seed (Appendix I).

## 5.6 Long-Context Evaluation

The preceding comparisons use the 4K training context. Random-Depth, the single-pass model and the planner then receive the same 600-update continuation, including 360 updates at 64K.

Language modeling. Random-Depth keeps lower PG19 perplexity than the single-pass model at every evaluated length from 4K to 64K, and every interval excludes zero (Table 5). The gap narrows with length, from 0.0218 nats at 4K to 0.0068 at 64K.

Retrieval. We apply identical answer-format finetuning to each model and evaluate on three needlein-a-haystack tasks (Appendix I). Random-Depth reaches 67.33% accuracy against 61.33% for the single-pass model, a gain of 6.0 points with interval [+1.33, +10.33]. On these 300 questions, Random-Depth matches the planner.

## 6 Conclusion

We presented LooM, which loops only the reader layers of a causal encoder–decoder and publishes each local KV entry once, so the cache stays fixed however often a layer runs. Per-layer loops give every budget from 3 to 21 calls, and Random-Depth training lets one router-free checkpoint serve them all. With enough loops, LooM outperforms a single-pass model trained on the same data in language modeling, zero-shot accuracy and longcontext evaluation, at the same cache size. Looping inside a CED thus turns depth into a compute budget chosen at deployment; scaling it to larger models is the natural next step.

## Limitations

Our evidence comes from one 204M-parameter backbone, 5B training tokens and one seed, so we cannot say how the gains move with scale; latency is likewise a batch-size-one reference measurement, not optimized serving throughput. The baselines are matched in parameters and data but not in compute: no non-looped model is given the FLOPs of a 21-call path, and the whole-decoder loop that keeps per-loop caches is a preliminary, partly trained run (Appendix E).

## References

Sangmin Bae, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Seungyeon Kim, and Tal Schuster. 2025a. Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA. In International Conference on Learning Representations.

Sangmin Bae, Yujin Kim, Reza Bayat, Sungnyun Kim, Jiyoun Ha, Tal Schuster, Adam Fisch, Hrayr Harutyunyan, Ziwei Ji, Aaron Courville, and Se-Young Yun. 2025b. Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation. arXiv preprint arXiv:2507.10524.

Andrea Banino, Jan Balaguer, and Charles Blundell. 2021. PonderNet: Learning to Ponder. arXiv preprint arXiv:2107.05407.

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek, Agustínˇ Piqueres Lajarín, Vaibhav Srivastav, Joshua Lochner, Caleb Fahlgren, Xuan-Son Nguyen, Clémentine Fourrier, Ben Burtenshaw, Hugo Larcher, Haojun Zhao, Cyril Zakka, Mathieu Morlon, and 3 others. 2025. SmolLM2: When Smol Goes Big – Data-Centric Training of a Small Language Model. arXiv preprint arXiv:2502.02737.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the AAAI Conference on Artificial Intelligence.

William Brandon, Mayank Mishra, Aniruddha Nrusimha, Rameswar Panda, and Jonathan Ragan-Kelley. 2024. Reducing Transformer Key-Value Cache Size with Cross-Layer Attention. In Advances in Neural Information Processing Systems.

Lizhang Chen, Jonathan Li, Chen Liang, Ni Lao, and Qiang Liu. 2026a. Training-Free Looped Transformers. arXiv preprint arXiv:2605.23872.

Wenkai Chen, Tianshu Li, Wenyong Huang, Yichun Yin, Lifeng Shang, and Chengwei Qin. 2026b. Loop-MoE: Unifying Iterative Computation with Mixtureof-Experts for Language Modeling. arXiv preprint arXiv:2606.04438.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv preprint arXiv:1803.05457.

Victor Conchello Vendrell, Arnau Padrés Masdemont, Niccolò Grillo, Jordi Ros-Giralt, Arash Behboodi, and Fabio Valerio Massoli. 2026. Memory-Efficient Looped Transformer: Decoupling Compute from Memory in Looped Language Models. arXiv preprint arXiv:2605.07721v3.

DeepSeek-AI. 2026. DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. arXiv preprint arXiv:2609.19969.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. 2019. Universal Transformers. In International Conference on Learning Representations.

Luciano Del Corro, Allison Del Giorno, Sahaj Agarwal, Bin Yu, Ahmed Hassan Awadallah, and Subhabrata Mukherjee. 2023. SkipDecode: Autoregressive Skip Decoding with Batching and Caching for Efficient LLM Inference. arXiv preprint arXiv:2307.02628.

Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, Ahmed A Aly, Beidi Chen, and Carole-Jean Wu. 2024. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics.

Angela Fan, Edouard Grave, and Armand Joulin. 2020. Reducing Transformer Depth on Demand with Structured Dropout. In International Conference on Learning Representations.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, and 5 others. 2023. A Framework for Few-Shot Language Model Evaluation.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. 2025. Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach. arXiv preprint arXiv:2502.05171.

Alex Graves. 2016. Adaptive Computation Time for Recurrent Neural Networks. arXiv preprint arXiv:1603.08983.

Di He, Pengxiang Li, Da Chang, Qingyan Meng, Lu Yin, and Shiwei Liu. 2026. Looping Beyond Twice: A Scalable Recipe for Looped Mixture-of-Experts. arXiv preprint arXiv:2610.01153.

Ahmed Heakl, Martin Gubri, Salman Khan, Sangdoo Yun, and Seong Joon Oh. 2025. Dr.LLM: Dynamic Layer Routing in LLMs. arXiv preprint arXiv:2510.12773.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the Real Context Size of Your Long-Context Language Models? In First Conference on Language Modeling.

Benhao Huang, Chufan Shi, Junlin Chen, Shicheng Wen, Zhengzhong Liu, Eric Xing, and Xuezhe Ma. 2026. Towards Looped Models Done Right, Part II: Rethinking at Fixed Points. arXiv preprint arXiv:2610.06833.

Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q. Weinberger. 2016. Deep Networks with Stochastic Depth. In Computer Vision – ECCV 2016.

Ahmadreza Jeddi, Marco Ciccone, and Babak Taati. 2026. LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation. arXiv preprint arXiv:2602.11451.

Cheng Liang, Tengfei Zhang, Wenjie Du, Jianyu Wu, Changlian Ma, Shuchen Pu, Ya Zhang, Yanfeng Wang, and Zhouhan Lin. 2026. Asymmetric Looped Transformers: Fixed Global Representations for an Iterative Decoder. arXiv preprint arXiv:2610.28489.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2017. Pointer Sentinel Mixture Models. In International Conference on Learning Representations.

James O’Neill and Fergal Reid. 2026. Looped Latent Attention: Cross-Loop KV Compression for Looped Transformers. arXiv preprint arXiv:2607.15456.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernández. 2016. The LAMBADA Dataset: Word Prediction Requiring a Broad Discourse Context. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben Al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro von Werra, and Thomas Wolf. 2024. The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2024. YaRN: Efficient Context Window Extension of Large Language Models. In International Conference on Learning Representations.

Hayden Prairie, Jacob M. Ryan, Taylor Berg-Kirkpatrick, and Daniel Y. Fu. 2026. BeNice! Closing Looping’s Inference-Memory Gap via KV-Sharing. arXiv preprint arXiv:2610.31633.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, and Timothy P. Lillicrap. 2020. Compressive Transformers for Long-Range Sequence Modelling. In International Conference on Learning Representations.

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. 2024. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models. arXiv preprint arXiv:2404.02258.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2020. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In Proceedings of the AAAI Conference on Artificial Intelligence.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. 2022. Confident Adaptive Language Modeling. In Advances in Neural Information Processing Systems.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. 2024. You Only Cache Once: Decoder-Decoder Architectures for Language Models. In Advances in Neural Information Processing Systems.

Haoyi Wu and Kewei Tu. 2024. Layer-Condensed KV Cache for Efficient Inference of Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11175–11188.

Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, and Wenfeng Liang. 2025. mHC: Manifold-Constrained Hyper-Connections. arXiv preprint arXiv:2512.24880.

Jiahui Yu and Thomas S. Huang. 2019. Universally Slimmable Networks and Improved Training Techniques. In Proceedings of the IEEE/CVF International Conference on Computer Vision.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings ofthe 57th Annual Meeting ofthe Association for Computational Linguistics.

Yuhua Zhou, Shaoqi Yu, Shichao Weng, Changhai Zhou, Mingze Yin, Fei Yang, and Aimin Pan. 2026. BUDDY: BUdget-Driven DYnamic Depth Routing for Adaptive Large Language Model Inference. arXiv preprint arXiv:2606.09514.

Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, and Xun Zhou. 2025a. Hyper-Connections. In International Conference on Learning Representations.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, Lu Li, Jiajun Shi, Kaijing Ma, Shanda Li, Taylor Kergan, Andrew Smith, Xingwei Qu, Mude Hui, Bohong Wu, and 14 others. 2025b. Scaling Latent Reasoning via Looped Language Models. arXiv preprint arXiv:2510.25741.

<table><tr><td>Property</td><td>Value</td></tr><tr><td>Backbone parameters</td><td>204,046,232</td></tr><tr><td>Planner parameters (planner model only)</td><td>19,159</td></tr><tr><td>Hidden / feed-forward width</td><td>832 / 2,688</td></tr><tr><td>Attention heads × head dimension</td><td>64 × 80</td></tr><tr><td>Sliding-window width (enc. 1–2; dec.)</td><td>256; 128</td></tr><tr><td>Indexer heads × width, Top-K Global KV compression (enc. 3/5; dec. 1)</td><td>32 × 32, 128</td></tr><tr><td>Reader layers</td><td>2; 1</td></tr><tr><td>Decoder calls per token</td><td>5 3-21</td></tr><tr><td>Training targets</td><td>5,000,200,876</td></tr></table>

Table 6: Architecture of the shared backbone. KV: key– value.

## A Architecture and Training Details

Optimization. We use AdamW with a peak learning rate of $3 \times 1 0 ^ { - 4 }$ that decays to $3 \times 1 0 ^ { - 5 }$ , 114 warmup updates, a weight decay of 0.1, and gradient clipping at a maximum norm of 1.0. Computation uses BF16, with selected components kept in FP32.

Attention and residual structure. The backbone uses CSA2 compressed-sparse attention (DeepSeek-AI, 2026), whose three modes are defined in Section 3.1. Inside the encoder, layers 3 and 5 write compressed KV at ratio two, which layers 4 and 6 reuse. Decoder layers 2, 4 and 6 reuse the latest decoder retrieval support. Query low-rank width is 208, with eight output groups and an output low-rank width of 160. The residual structure uses four parallel streams of manifoldconstrained hyper-connections (Xie et al., 2025; Zhu et al., 2025a), and the input embeddings and the output head are untied. Global KV producers are encoder layers 3 and 5 and decoder layer 1 (physical layers L3, L5 and L7), with compression ratios two, two and one, respectively. Index producers are encoder layers 3 and 5 and decoder layers 1, 3 and 5. Decoder layers 3 and 5 recompute queries and Top-128 support over decoder layer 1’s global bank but do not append persistent global KV or indexer keys. Decoder layers 4 and 6 consume the selections from their most recent visit. This reuse violates neither the pre-repetition global-memory rule nor the cache size of Eq. (3).

Planner. The planner reads $u _ { t } \in \mathbb { R } ^ { 8 3 2 }$ , the normalized output of decoder layer 1. The score groups in Eq. (5) have sizes (5, 4, 5, 4, 5). Each reader computes candidate hidden states for depths of up to four.

In Eq. (6), H is the entropy of a token’s routing distribution, averaged over valid tokens and variable readers.

Under a per-token cap B, the allocator solves $\begin{array} { r } { \operatorname* { m a x } _ { d : C ( d ) \leq B } \sum _ { \ell } z _ { t , \ell , d _ { \ell } } } \end{array}$ with a multiple-choice dynamic program in $O ( L _ { D } B \operatorname* { m a x } _ { j } | \mathcal { D } _ { j } | )$ time per token. Because the cap is an upper bound, realized calls can be lower. Replacing the max with a logsum-exp gives the constrained marginals used by the budget-conditioned continuation (Appendix F); that continuation adds a zero-initialized budget table $e _ { B } \in \mathbb { R } ^ { 2 3 }$ with 22 rows and 506 parameters.

Scalar-depth planner. A coarser planner predicts one depth per token and shares it across decoder layers 2–6, so it cannot skip a layer. It reaches a test CE of 3.04026 at 19.17 calls on average, behind the layerwise planner (3.03546 at 17.96 native calls) and Random-Depth at 19 calls (3.03058).

Dense execution and sampling. Each microbatch is unrolled to its largest sampled depth, unexecuted repetitions are masked, and each KV entry is still published only once. Sampling is deterministic and seeded by the training seed together with the update, microbatch slot and data-parallel rank.

Self-Distilled draws one budget per microbatch slot and a path per rank, with the encoder and decoder layer 1 forming a prefix shared by student and full-depth teacher.

Static path search. The staged search of Section 3.4 is exhaustive only at its first stage, so it does not certify the best static path on all 512 samples. All selection files record that paths were locked before test scoring.

## B Uniform-Budget Control

Section 5.2 attributes Random-Depth’s weak low budgets to how rarely its deep-heavy schedule samples them, rather than to shallow execution itself. Self-Distilled supports this but also adds a fulldepth teacher, so it changes two things at once. The uniform run changes only the budget distribution: it keeps Random-Depth’s recipe but spreads most samples evenly over 3–20 calls instead of placing three quarters at 17–21. Both runs are compared at the same early point in training.

Table 7 gives the schedule and the outcome. The uniform run is better at 3 and 6 calls, so shallow paths become strong once they are trained often enough, and the deep-heavy schedule is better from 11 calls on; Section 5.2 reads what that trade costs.

<table><tr><td>Calls</td><td>Deep-heavy</td><td>Uniform</td></tr><tr><td>3</td><td>4.2236</td><td>3.4108</td></tr><tr><td>6</td><td>3.4551</td><td>3.3587</td></tr><tr><td>11</td><td>3.3290</td><td>3.3490</td></tr><tr><td>16</td><td>3.3114</td><td>3.3470</td></tr><tr><td>21</td><td>3.3106</td><td>3.3468</td></tr><tr><td>Train calls</td><td>17.19</td><td>13.88</td></tr></table>

Table 7: Two budget distributions in the same recipe, compared early in training: test cross-entropy (CE; nats per token, lower is better). Deep-heavy is Random-Depth’s schedule; the uniform run instead draws the full 21-call path with probability 0.25 and otherwise a budget uniform in 3–20 calls. Both then draw a uniformly random legal path of that budget, skipped layers included. Train calls are expected decoder calls per training token; the uniform run uses fewer, which confounds absolute levels but not the within-model flatness of its curve.

## C Cost and Prefill Measurements

Protocol. Decode uses a 32-token teacher-forced continuation after a 128-token prompt, with eight requests and three repetitions per path. The Python reference loop times the encoder, selected decoder bodies, cache updates and FP32 logits. Matrix FLOPs are counted as in Section 4, from dispatched operand shapes including explicit tile padding and masked bank rows; they are neither hardware instruction counts nor theoretical FLOPs for useful tokens. Each row averages those requests whose operator-count audit passed, between six and eight of the eight. Cache size is measured as live persistent storage. Fixed4 was not timed independently at 21 calls; we reuse the identical path shape, and the two measured models differ by at most 1.4% on identical paths.

Decode and prefill. On identical static paths, cost depends on the path and model shape rather than on the weights. With 512- and 2,048-token prompts, the persistent cache is again identical on the 3- and 21-call paths. Native routing takes 100.03 ms at 17.99 calls, within 0.5% of interpolation between the static 16- and 21-call paths. Prefill follows DeepSeek-V4.1-Flash’s SWA replay of the most recent 128 prompt tokens (DeepSeek-AI, 2026). At 65,536 prompt tokens, prefill takes 3.13 s at 3 calls and 3.23 s at 21, an increase of 3.2%. The budget therefore mainly affects decode (Figure 3).

## D Grouped Decoder Memory

The comparison uses Fixed4’s schedule: encoder layers 1–6 and decoder layer 1 once, then decoder layers 2–6 four times each. LooM’s layout builds one decoder memory at decoder layer 1 and reuses it through decoder layer 6. The grouped layout builds memories at decoder layers 1, 3 and 5, adding two global producers and 138,464 parameters; later memories are computed after the repeated reads. The grouped model increases persistent cache size by 1.58× at 1,024 prompt tokens, 1.80× at 4,096 and 1.88× at 16,384. The paired interval is computed over the same 7,836 test document segments with 2,000 resamples. This is one Fixed4-schedule variant and one seed, not a general test of all memory-sharing designs.

## E Recirculating the Whole Decoder

The whole-decoder control repeats the six-layer decoder up to four times with shared weights, with each round retaining its own local history. The encoder and global memory are computed once. A gate defines a first-exit distribution, and training minimizes expected CE minus an entropy bonus with coefficient 0.1. The run starts from a random initialization and reuses the single-pass data, optimizer, schedule and seed.

At update 1,800, on the same 512-sample, 1,737,712-target validation set, the whole-decoder loop reaches 3.3814, 3.3557, 3.3538 and 3.3541 over one to four rounds. The second round reduces CE by 0.026 nats, the third by 0.002, and the fourth yields no further reduction, so no fixed number of rounds reaches the single-pass model at 3.3496. The learned exit distribution is close to uniform, (0.21, 0.27, 0.26, 0.26). Random-Depth at 21 calls reaches 3.3458 on this snapshot, but the two objectives differ and the run is a single-seed preliminary study, so we cannot attribute that difference to reading fixed memory.

## F Planner Continuation and Allocation Diagnostics

The original planner is poorly calibrated at small caps, so we continued it and its budget table for 200 updates under caps $B \in \{ 6 , 1 1 , 1 4 , 1 6 , 2 1 \}$ using a zero-initialized budget embedding. The two continuations (seeds 42 and 43) are used only for the allocation controls; the main frontier compares the original 5,703-update checkpoints. Their native-CE differences across caps span 0.0002–0.0024 nats. These are continuation-seed differences, not independent pretraining replicates.

![](images/318b72c5e3cc66f42299c77cbeb4d3d091b3d30422e95b536e60eb99da10883b.jpg)

![](images/d0240c782b4660b5f780028ff4617c6e9667de7a2886fa1da1ee2bf467b4b850.jpg)  
Figure 3: Prefill and subsequent decode for Random-Depth against the budget, one curve per prompt length (one NVIDIA H800, batch size one, median of five). (a) Prefill is set by the prompt, not the budget: at 65,536 tokens the whole 3-to-21-call range costs 3.2% more. (b) Decode is set by the budget, with the four prompt lengths almost on one curve. The persistent cache is identical across paths at every prompt length. Prompts beyond 4,096 tokens are timed for cost only.

<table><tr><td>Replacement</td><td>Cap</td><td>∆CE</td></tr><tr><td>Static path</td><td>6</td><td>-0.05273 to −0.05053</td></tr><tr><td rowspan="3">Static split</td><td>11-21</td><td>-0.00037 to +0.00060</td></tr><tr><td>6</td><td>−0.05273 to -0.05053</td></tr><tr><td>11-21</td><td>+0.00005 to +0.00063</td></tr><tr><td>Mean totals</td><td>21</td><td>+0.00013 to +0.00014</td></tr><tr><td>Shuffled totals</td><td>21</td><td>+0.00166 to +0.00167</td></tr></table>

Table 8: Allocation controls on two 200-update budgetconditioned planner continuations (seeds 42 and 43), needed because the original planner is poorly calibrated at cap 6. Each value is the range of replacement minus native-routing cross-entropy (CE); negative favors the replacement. At cap 6, static path and static split improve over poorly calibrated native routing. A cap bounds each token’s total calls.

Random-Depth and the planner differ in more than token dependence: their training also differs in auxiliary losses, exploration noise and straightthrough gradients. At 16 and 21 calls, Random-Depth’s margins are 0.0067 and 0.0049 nats, too small to establish a seed-robust ordering. Native routing costs what its mean call count predicts: it falls within 0.5% of interpolation between the static paths (Appendix C). A scalar planner and longer continuations improve low budgets, but neither rules out a usable token-level signal (Appendix A).

## G Mechanism Analysis: Protocol and Gains by Decile

Figure 4 reports the full analysis. Random-Depth is scored on locked 11- and 21-call paths, whereas Fixed4 and the planner use 21 calls, all on the same 256 test sequences (865,148 targets). State and retrieval statistics use the first 64 sequences. Difficulty deciles use the mean NLL of the compared paths, which avoids binning on one arm’s noise. In the retrieval-freezing intervention, visits 2–4 of decoder layers 3 and 5 reuse their first-visit Top-K sets, so decoder layers 4 and 6 also read those sets. Scores are still computed; only the selection is replaced.

Relative state update. Panel (b) measures how far one visit moves a token’s hidden state. For a visit that takes the state from $h _ { \mathrm { i n } }$ to $h _ { \mathrm { o u t } }$ , the relative update is $\| h _ { \mathrm { o u t } } - h _ { \mathrm { i n } } \| / \| h _ { \mathrm { i n } } \|$ , averaged over tokens. Dividing by the incoming norm is what makes the five reader layers comparable, since their states differ in scale: a value of 0.48 means the visit moved the state by about half its own length, and 0.15 means it moved it by a seventh. The quantity is a distance and carries no direction, so it shows that later visits keep changing the state, not that the change helps. What ties repeated reading to quality is the retrieval-freezing intervention of Section 5.4.

## H Downstream Evaluation

Protocol and configurations. We evaluate with lm-evaluation-harness 0.4.13 (Gao et al., 2023) on LAMBADA (Paperno et al., 2016), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), WinoGrande (Sakaguchi et al., 2020), ARC-Easy and ARC-Challenge (Clark et al., 2018), reporting accuracy, or normalized accuracy for HellaSwag and ARC-Challenge. We fixed the metrics and macro average before scoring. The full sets contain 5,153, 10,042, 1,838, 1,267, 2,376 and 1,172 examples, respectively. WikiText-2 uses 62 documents with rolling 4,096-token windows (Merity et al., 2017). We evaluated 24 configurations: singlepass, Fixed4, native routing, and the 3-, 6-, 8-, 11-, 13-, 16- and 21-call static paths of Random-Depth, Self-Distilled and the planner. Static configurations use the paths locked for the CE frontier.

![](images/bc0f724edae5296d92248b259d5a341a3dbe7a2b603103c2242b4c90d039f298.jpg)

![](images/2fb0da0f60a0db8963ad20c31c5942c948fef24819a954a06cca086147b23a0d.jpg)  
Figure 4: Mechanism analysis on 256 test sequences. (a) The allocation-relevant negative log-likelihood (NLL) gain from 11 to 21 calls by difficulty decile (1: easiest); positive is improvement, and difficulty uses mean NLL over both paths, avoiding selection on one arm’s noise. (b) Relative residual-state updates at visits 1–4 of decoder layers 2–6, labeled D2–D6 in the panel, on the 21-call path.

Comparisons with intervals crossing zero. Random-Depth at 11 calls is 0.71 points above Fixed4 at 21 on the six-task average, but the interval [−0.04, +1.43] crosses zero. Self-Distilled versus Random-Depth at six calls is likewise unresolved (+0.40, [−0.38, +1.17]). WikiText-2 requires 16 rather than 11 calls to overtake Fixed4 at 21, so the 11-call CE equality is distributionspecific. Because scores are non-monotonic across 13, 16 and 21 calls, sub-point differences in the aggregate should not be interpreted as a frontier ordering.

## I Long-Context Continuation and Retrieval Protocol

Random-Depth, the single-pass model and the planner start from their 5B-target checkpoints and receive the same 600-update continuation (314,572,800 targets per model). The continuation consists of 8K, 16K, 32K and 64K phases with 60, 60, 120 and 360 updates, respectively. At every phase, two thirds of the updates use long PG19 examples and one third use short-domain data. All three models use the same data order, optimizer and learning-rate schedule $( 3 \times 1 0 ^ { - 5 }$ to $3 \times 1 0 ^ { - 6 }$ , with 60 warmup updates). The

45 PG19 books used for evaluation are disjoint from the continuation’s training and validation books. The planner loads its trained routing parameters and continues straight-through training with a balance coefficient of 0.001, no entropy bonus and no exploration noise; an audit verifies that planner.weight changes. Random-Depth instead keeps its pretraining schedule (17.19 expected calls). The compressed-attention branch uses YaRN/NTK-by-parts-derived interpolation targeting 64K (Peng et al., 2024), whereas the local branch is unchanged.

Perplexity is evaluated before task fine-tuning. Random-Depth minus planner static-21 is +0.0031 nats at 4K $\left( \left[ - 0 . 0 0 0 2 , + 0 . 0 0 6 3 \right] \right)$ and +0.0050 at 64K ([+0.0014, +0.0084]). Planner static-21 minus native routing ranges from +0.0008 to +0.0014 nats, with every interval excluding zero. Table 5 gives the compact Random-Depth–singlepass comparison; Table 9 reports the paired routing differences with whole-book intervals.

For retrieval, all continued models receive the same direct-answer fine-tuning on 512 examples (64 updates, response-only supervision, and no 64K examples). Random-Depth draws sampled depths, the single-pass model runs one pass, and the planner uses native or static-21 execution. We evaluate niah\_single\_1, niah\_single\_2 and niah\_multikey\_1 from RULER (Hsieh et al., 2024) at 4K, 8K, 16K, 32K and 64K, with 20 questions per task and context length. This is a 300-question subset, not the full 13-task RULER benchmark.

<table><tr><td>Ctx.</td><td>RD – native</td><td>Static — native</td></tr><tr><td>4K</td><td>+3.9 [+0.6, +7.2]</td><td>+0.8 [+0.5, +1.3]</td></tr><tr><td>8K</td><td>+1.8 [−2.2, +5.4]</td><td>+1.3 [+0.9, +1.8]</td></tr><tr><td>16K 32K</td><td>+1.9[−1.8, +5.4]</td><td>+1.4 [+0.9, +1.8]</td></tr><tr><td>64K</td><td>+2.4 [−1.2, +5.8] +6.4 [+2.8, +9.8]</td><td>+1.3 [+0.9, +1.7] +1.4 [+1.0, +1.8]</td></tr></table>

Table 9: Paired negative log-likelihood (NLL) differences in 10<sup>−3</sup> nats per token with whole-book 95% confidence intervals; negative favors the first model in each difference. RD denotes Random-Depth; Static and native denote the planner with its static path and with native routing, respectively. Point estimates favor native at every length: static minus native excludes zero at every length, while RD minus native excludes zero only at 4K and 64K.
<table><tr><td>Length</td><td>Single</td><td>Planner native</td><td>Planner static</td><td>RD</td></tr><tr><td>4K</td><td>61.67</td><td>68.33</td><td>68.33</td><td>68.33</td></tr><tr><td>8K</td><td>63.33</td><td>68.33</td><td>71.67</td><td>68.33</td></tr><tr><td>16K</td><td>66.67</td><td>70.00</td><td>70.00</td><td>70.00</td></tr><tr><td>32K</td><td>61.67</td><td>70.00</td><td>70.00</td><td>71.67</td></tr><tr><td>64K</td><td>53.33</td><td>63.33</td><td>65.00</td><td>58.33</td></tr><tr><td>Overall</td><td>61.33</td><td>68.00</td><td>69.00</td><td>67.33</td></tr></table>

Table 10: Three-task needle-in-a-haystack (NIAH) macro accuracy (%, higher is better) by context length in tokens and overall (300 questions per model). Single denotes the single-pass model, and RD denotes Random-Depth. All arms use the same answer-format fine-tuning. Random-Depth and the planner with its static path use the static 21-call path, whereas the planner with native routing uses its learned per-token policy. The table reports point estimates; the overall paired interval for RD minus the single-pass model is [+1.33, +10.33] percentage points.

## J Executed Hindsight Allocation

We use the same 256 test sequences and 865,148 targets as in the mechanism analysis. Candidate paths are Random-Depth’s validation-locked 11-, 13-, 16- and 21-call paths. For each token, a Lagrangian hindsight policy uses the candidate NLLs to select a path under an average budget of 16 calls. A greedy assignment followed by a dynamicprogramming repair enforces that total exactly. The repair is not a certified global optimum, so the measured gain is a lower bound on the best hindsight gain over these four paths.

The mixed execution reaches CE 2.9631, compared with 3.0203 for static 16, a gain of 0.0572 nats. Scoring each token on its chosen path without executing the mixture reaches 2.9514, so leaving out the execution effects overstates the gain by 0.0117 nats. This is a label-aware upper-bound reference, not evidence of prefix-predictable signal or of a deployable allocation policy.

## K Data

Training draws 5B valid target tokens from the FineWeb-Edu 100BT sample (Penedo et al., 2024), tokenized with the SmolLM2 tokenizer (Ben Allal et al., 2025), whose vocabulary holds 49,152 tokens. Deterministic packed manifests record source-document segments and target masks. Source documents are hashed, and each one is assigned whole to exactly one of the training, validation and test partitions, so no document contributes to more than one of them; the packed samples are then shuffled once.