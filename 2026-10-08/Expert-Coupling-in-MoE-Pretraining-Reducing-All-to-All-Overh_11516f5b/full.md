# Expert Coupling in MoE Pretraining: Reducing All-to-All Overhead with Correlated Placement and Token Shuffling

Radha Gulhane<sup>\*</sup> Quentin Anthony Beren Millidge

Zyphra

Abstract—Mixture-of-Experts (MoE) layers replace the feedforward block of a Transformer with E expert networks, and each token is routed to k of these experts. Under expert parallelism (EP) the experts are distributed across GPUs, and every MoE layer runs all-to-all collectives in the forward and backward passes to dispatch tokens to their experts and then combine the results. On a cluster with 8 AMD Instinct MI300X GPUs per node, these collectives can take 45 % of the training step at EP32 with top-2 routing and 60 % with top-6 routing. We find that early in pretraining routers have already learned to assign tokens to experts in correlated patterns, both within a layer and across layers. At top-2, 0.8 % of the expert pairs in a layer are selected together by 42 % of tokens, and the experts a token selects at one layer predict the experts it selects at the next layer. We use these correlations to keep more token–expert assignments on the token’s own GPU, which reduces communication across GPUs and across nodes. Correlated expert placement puts experts that are often selected together on the same GPU. Combined with a dispatcher that sends each token to each GPU once, it removes up to 58 % of dispatched rows. Token shuffling applies when sequence parallelism shards tokens across the EP group. It moves each token to the GPU predicted to hold its next-layer experts during the reduce-scatter that follows attention. On one node this raises the share of token–expert assignments served on the token’s GPU from 12.5 % to 59 %. In Megatron-LM, across EP degrees from 8 to 64 with top-2 and top-6 routing, the two methods reduce all-to-all time by 1.16–2.63× and end-to-end step time by up to 1.41×. Neither method changes the models’ underlying routing decisions or expert parameters.

## I. INTRODUCTION

Mixture-of-Experts (MoE) layers have become a central building block of large language models [Jiang et al., 2024, Liu et al., 2024, Yang et al., 2025, Kimi Team, 2026]. In an MoE layer, the single feed-forward network (FFN) of a dense Transformer block is replaced by a set of E parallel FFNs, called experts, together with a lightweight router [Shazeer et al., 2017, Lepikhin et al., 2021, Fedus et al., 2022]. After the attention sub-layer, the router scores each token’s hidden state against all experts and selects its top-k experts. The token is processed only by those experts, and their outputs are combined using the router weights. Model capacity therefore grows with E, while the FLOPs per token grow only with k. Because each expert is updated only by the tokens routed to it, experts become specialized over the course of pretraining [Dai et al., 2024, Xue et al., 2024].

This capacity comes at a communication cost. Under expert parallelism (EP) [Lepikhin et al., 2021], experts are sharded across GPUs, and every MoE layer requires two all-to-all (A2A) collectives. The dispatch sends each token from the GPU that owns it to the GPUs holding its selected experts, and the combine returns the expert outputs to the owning GPU. Both collectives run in the forward pass and again in the backward pass. The communication volume grows with the EP degree and with top-k, and once the EP group spans multiple nodes, the slower inter-node links dominate step time. In our measurements on a cluster with 8 AMD Instinct MI300X GPUs per node (Section II-A), all-to-all communication accounts for 13.5 % and 44.8 % of step time at EP8 and EP32 under top-2 routing, and for 24.0 % and 60.4 % under top-6 routing.

Conventional MoE implementations place experts contiguously across the EP group, with expert e on GPU ⌊e/(E/EP)⌋, irrespective of which experts are used together. Depending on the routing, the all-to-all can also become unbalanced, and the few GPUs that receive a disproportionate share of tokens become stragglers that delay the whole collective. Mechanisms that address this include expert replication [Liu et al., 2024], load-driven expert relocation [Zhai et al., 2023], and dynamically planned redundant experts with weight prefetching [Chen et al., 2026, Kimi Team, 2026]. SmartMoE also changes which GPU holds each expert, but it places experts to balance load, not to co-locate experts that are selected together. These techniques lower the peak per-link traffic by distributing the traffic across GPUs. The total traffic stays about the same, because the same fraction of token– expert assignments crosses a GPU boundary. Replication also costs memory for the expert copies. DeepSeek-V3 additionally reduces inter-node traffic with node-limited routing, which restricts the experts the router can select.

A second line of work exploits expert affinity for MoE inference. GRACE-MoE [Han et al., 2025] groups experts that are often selected together within a layer onto the same GPU, first across nodes and then across the GPUs of each node, and sends a token to each GPU only once. Its groups differ in size, and it replicates heavily loaded experts to rebalance the GPUs, which costs memory. We apply withinlayer grouping to pretraining with equal-size groups and no replicas. ExFlow [Yao et al., 2024] uses inter-layer expert affinity to co-locate experts across layers. This approach does not transfer directly to pretraining. During autoregressive decoding, a token can be placed on any convenient GPU. During pretraining, a token stays on the GPU that holds its part of the sequence for the attention sub-layer, and moving it requires additional communication. Exploiting cross-layer affinity during training therefore requires a mechanism that ExFlow does not provide.

Our starting point is that once experts specialize during pretraining, routing follows stable patterns. We study expert routing across pretraining stages and load-balancing methods, and find two such patterns. The first is within-layer correlation. Some pairs of experts are selected together much more often than independent routing would predict. After approximately 2.1 B training tokens with top-2 routing, 42 % of tokens select one of only 0.8 % of the possible expert pairs, and this pattern is already present after about one billion tokens. The second is cross-layer correlation. The experts a token selects at layer ℓ predict the experts it selects at layer ℓ + 1. Both correlations appear with all three load-balancing methods we test, but with differing strengths.

We exploit these correlations to raise token–expert locality, the fraction of (token, selected expert) pairs whose expert is on the token’s GPU. These pairs never enter the network, so higher locality reduces the total A2A volume. Our methods require no architectural changes or expert replicas, and they do not change routing decisions or the training loss. Correlated placement and deduplication apply to any layout that uses expert parallelism. Token shuffling applies only to layouts that use tensor parallelism with sequence parallelism and in which the TP group and the EP group are the same set of GPUs (TP = EP), such as TP8 EP8 and TP16 EP16 in our evaluation. It does not apply to layouts without tensor parallelism (TP1). Correlated expert placement and deduplication also apply to inference, where affinity-based grouping has already been evaluated [Han et al., 2025]; we do not evaluate inference.

We make the following contributions:

• Correlated Expert Placement and a Deduplicating Dispatcher (Section V-A). We place correlated experts on the same GPU, so that the experts a token selects are more likely to share a GPU. The deduplicating dispatcher sends each token to a destination GPU at most once, regardless of how many of its selected experts are there. Deduplication is an existing technique [Zhao et al., 2025]. The all-to-all dispatcher in Megatron-LM [Shoeybi et al., 2019] sends one copy of a token per selected expert, so we implement deduplication there as the basis for placement. Together, placement and deduplication reduce both intra-node and inter-node communication volume.

• Token Shuffling (Section V-B). For sequence-parallel layouts in which the TP group equals the EP group, we (i) predict a token’s experts at layer ℓ+1 from the experts the token selected at the previous layers, and (ii) move the token ahead of time to the GPU that is predicted to hold those experts. The move is folded into the reducescatter that sequence parallelism [Korthikanti et al., 2023] already performs after attention. This brings to pretraining the locality benefit that prior affinity-based work obtained for inference.

• Characterization of Expert Routing (Sections II-B and VI, and the appendix). We quantify within-layer coselection and cross-layer predictability across pretraining stages, top-k settings and load-balancing methods, and relate these statistics to communication volume.

• Evaluation in Megatron-LM (Section VI). We report communication volume, A2A time and end-to-end step time. With expert parallelism alone, correlated placement and deduplication reduce A2A time by 1.16×–1.38× under top-2 routing and by 1.35×–1.95× under top-6 routing. With token shuffling added, A2A time is reduced by 1.40×–1.74× under top-2 and by 1.46×–2.63× under top-6.

## II. MOTIVATION

## A. All-to-All Cost in MoE Pretraining

We begin by measuring how much of an MoE training step is spent in the dispatch and combine collectives. Each MoE layer runs both collectives in the forward pass and again in the backward pass. The cost of these collectives depends on two configuration choices. A larger expert-parallel degree spreads experts over more GPUs, so the fraction of a token’s experts on its own GPU falls as 1/EP. Beyond a single node, however, the exchange also moves from the intra-node fabric to the inter-node network. A larger top-k raises the number of activation rows each token contributes to both collectives, so top-6 moves three times the volume of top-2.

We measure a 12-layer MoE model $( d _ { \mathrm { m o d e l } } { = } 4 0 9 6 )$ , with 128 experts of FFN width 1024 per layer (see Section VI and Table II for the detailed configuration). Figure 1 shows that the two effects compound. We measure all-to-all time as the wall-clock time the step spends inside the dispatch and combine calls, which includes time spent waiting for other GPUs. Widening the EP group from a single node (EP8) to four nodes (EP32) raises the all-to-all share of step time from 13.5 % to 44.8 % under top-2 and from 24.0 % to 60.4 % under top-6. At fixed EP8, moving from top-2 to top-6 raises it from 13.5 % to 24.0 %. At EP32 with top-6, the two collectives take more of the step than expert computation and all non-MoE computation combined.

## B. Expert Correlation Within and Across Layers

Crucially, the measurements above use contiguous placement, which ignores the correlations between the experts a token selects. Once experts have specialized, these selections follow two regular patterns, which we exploit in this paper. We measure both on the top-2 and top-6 models from Section II-A. All statistics in this section are computed offline. We record every token’s expert choices at every MoE layer on a few late training steps (131k tokens per step), fit each placement and predictor on some of these steps, and apply them to the recorded routes of the other steps without running the dispatcher.

a) Within a Layer, Experts Are Selected in Stable Pairs: Figure 2a shows, for layer $^ { 8 , }$ the probability that a token selecting expert $e _ { 1 }$ also selects $e _ { 2 }$ . The blue cells are pairs that are often chosen together: 14 pairs are chosen with at least 50 % confidence and carry 16 % of all tokens, and 64 of the 8,128 possible pairs carry 42 %, whereas independent routing with the same loads would give 1.6 %. For 73 of the 128 experts, one partner appears in at least 30 % of its tokens.

![](images/a84c6d77fd4419187363303056e59aa271c786242bcc336f16a37eff0f1c169c.jpg)  
a2a dispatcha2a combine  expert GEMM  router/permute non-MoE backward  
Fig. 1: Share of training step time by component at EP8 and EP32 with top-2 and top-6 routing, using the Megatron-LM all-to-all dispatcher and contiguous placement. Dashed boxes mark the all-to-all share.

![](images/d6f9280a173d304efd4879fa0fee8bd9f0824c0ef0ba4ffd82b2d3f351bfec96.jpg)  
(a) Within layer 8

![](images/e46ee7da40a819e8e432ded31642af0cfab87cf2f694643cfea14e8de12f18e0.jpg)  
(b) Across layers 8 → 9  
Fig. 2: Expert correlation in the top-2 model, computed from the saved routing of 524k tokens at training steps 1983–1995 (every fourth step). (a) $P ( e _ { 2 } \in \mathrm { t o p } { - 2 } \mid e _ { 1 } \in \mathrm { t o p } { - 2 } )$ within layer 8; (b) $P ( e _ { 2 }$ at $9 \mid e _ { 1 }$ at 8) across layers. Blue marks strongly correlated pairs (conditional probability above 0.3).

A token is sent once per GPU its experts occupy, so colocating a frequent pair halves its traffic under a deduplicating dispatcher. Contiguous placement does this only by chance; correlated placement does it on purpose. Table I (within layer) counts rows, with each row corresponding to one copy of a token’s hidden state sent to one GPU. At EP8, correlated placement raises the share of top-2 tokens whose experts share a GPU from 12 % to 53 %, and raises the fraction of rows removed by deduplication from 6 % to 26 % under top-2 and 25 % to 58 % under top-6. At EP32, a GPU holds only four experts, so a top-6 token’s experts can never all share one, yet rows removed still rise from 5.5 % to 35.1 %.

b) Across Layers, the Current Experts Predict the Next Ones: Figure 2b plots, for each expert $e _ { 1 }$ at layer $^ { 8 , }$ the probability that a token routed to $e _ { 1 }$ selects expert $e _ { 2 }$ at layer 9. Most layer-8 experts send a large share of their tokens to a single layer-9 expert: 112 of the 128 send at least 30 % to one specific expert, and the median share is 48 %.

This correlation can be used to move tokens. Before the router of layer i+1 runs, the experts chosen at layer i already

TABLE I: Offline placement and prediction statistics on heldout iterations 1983 and 1987 (fit on 1991–1999); all values in % (rounded) except GPUs. Same GPU: tokens whose top-k experts share a GPU. GPUs: distinct destinations per token. Deduplication: rows removed as duplicates. Local experts: share of a token’s next-layer experts on its own GPU when it is moved to the GPU predicted from its current experts (T /EP tokens per GPU) and when it stays on its home GPU (1/EP).

<table><tr><td></td><td></td><td colspan="3">Within layer</td><td colspan="2">Across layers: local experts</td></tr><tr><td>EP</td><td>Placement</td><td>Same GPU</td><td>GPUs</td><td>Dedupli- cation</td><td>Token moved</td><td>Token stays</td></tr><tr><td colspan="7">top-2 routing</td></tr><tr><td>EP8</td><td>Contiguous</td><td>12</td><td>1.88</td><td>6</td><td>33</td><td>13</td></tr><tr><td></td><td>Correlated</td><td>53</td><td>1.47</td><td>26</td><td>58</td><td>13</td></tr><tr><td>EP16</td><td>Contiguous</td><td>6</td><td>1.95</td><td>3</td><td>28</td><td>6</td></tr><tr><td></td><td>Correlated</td><td>45</td><td>1.55</td><td>23</td><td>51</td><td>6</td></tr><tr><td>EP32</td><td>Contiguous</td><td>2</td><td>1.98</td><td>1</td><td>25</td><td>3</td></tr><tr><td></td><td>Correlated</td><td>36</td><td>1.64</td><td>18</td><td>42</td><td>3</td></tr><tr><td colspan="7">top-6 routing</td></tr><tr><td>EP8</td><td>Contiguous</td><td>0</td><td>4.49</td><td>25</td><td>25</td><td>13</td></tr><tr><td></td><td>Correlated</td><td>18</td><td>2.54</td><td>58</td><td>60</td><td>13</td></tr><tr><td>EP16</td><td>Contiguous</td><td>0</td><td>5.24</td><td>13</td><td>18</td><td>6</td></tr><tr><td></td><td>Correlated</td><td>8</td><td>3.08</td><td>49</td><td>49</td><td>6</td></tr><tr><td>EP32</td><td>Contiguous</td><td>0</td><td>5.67</td><td>6</td><td>14</td><td>3</td></tr><tr><td></td><td>Correlated</td><td>0</td><td>3.89</td><td>35</td><td>33</td><td>3</td></tr></table>

indicate which GPU will hold most next experts, and under sequence parallelism (Section III-B) the token can be moved there during attention. Every GPU must own exactly T/EP tokens, so not every token gets its predicted GPU. Table I (across layers) reports, under this constraint, the share of a token’s next-layer experts that are local. A token that stays on its home GPU has 1/EP local. A token moved to its predicted GPU has 58 %, 51 % and 42 % at EP8, EP16 and EP32 for top-2 and 60 %, 49 % and 33 % for top-6 under correlated placement. Placement matters here too: with contiguous placement the moved token has only 33 % of its experts local at EP8, because its next experts span more GPUs.

## III. BACKGROUND

## A. Expert Parallelism

When the experts of a layer no longer fit in the memory of one GPU, expert parallelism (EP) partitions them across a group of GPUs, typically alongside data parallelism. With $E { = } 1 2 8$ experts and EP=8, for example, each GPU hosts $E / \mathrm { E P = } 1 6$ experts. A token routed to an expert on another GPU is sent there by the dispatch all-to-all and returned by the combine all-to-all, in both the forward and the backward pass. The first panel of Figure 3 shows this pattern. Every token–expert assignment that crosses a GPU boundary adds a row to both collectives.

## B. Hybrid Parallelism: Tensor–Expert Parallelism

For large models and long-context training, expert parallelism alone is not enough to fit the model in GPU memory, and additional model parallelism is needed. Tensor parallelism (TP) [Shoeybi et al., 2019] shards the computation of each layer across GPUs and is widely used in both pretraining and inference.

In a common hybrid TP–EP configuration, attention and the MoE layer use different token layouts. Attention is partitioned across the TP ranks, and the MoE layer operates on sequence shards distributed across the same ranks [Korthikanti et al., 2023, Liu et al., 2025, Singh et al., 2023]. Two collectives move between these layouts. After attention, a reduce-scatter sums the partial TP outputs and partitions the sequence, which leaves each rank with a contiguous subset of tokens. These tokens then pass through the dispatch all-to-all, the expert computation, and the combine all-to-all. After the MoE layer, an all-gather reconstructs the full sequence on every TP rank for the next Transformer block.

Token shuffling replaces this reduce-scatter and all-gather with shuffled versions (Figure 5), and it requires the TP group and the EP group to be the same GPUs (TP = EP). The shuffled reduce-scatter sends each token to the rank predicted to hold most of its experts in the coming MoE layer, based on the experts it selected in the previous layers. The shuffleaware all-gather restores the canonical token order before the next attention layer. A larger fraction of each token’s experts is then on its own GPU, and the dispatch and combine all-to-alls send fewer bytes across the network. Section V-B describes the mechanism in detail. Experts are not tensor-parallel in this work. Some latency-sensitive inference deployments shard experts with tensor parallelism, which we do not consider.

## IV. RELATED WORK

Two systems place experts by within-layer affinity on models that are already trained. Occult [Luo et al., 2025] measures co-selection in a pretrained MoE model on a profiling dataset, places experts that are often selected together within a layer on the same GPU, and sends a token to each GPU only once. It evaluates inference and fine-tuning, and obtains further savings by restricting each token’s experts to a few GPUs, which changes routing decisions. GRACE-MoE [Han et al., 2025] targets inference. It groups co-selected experts first across nodes and then across the GPUs of each node, lets the groups differ in size, and replicates heavily loaded experts to rebalance the GPUs, which costs memory. Correlated expert placement (Section V-A) uses the same within-layer signal and, like GRACE-MoE, partitions across nodes before GPUs. Our work differs in three ways. We study pretraining from scratch, where the correlations do not exist at the start, and measure how early they form and how few tokens are needed to estimate them (Section VI-A). We leave routing decisions unchanged and add no replicas. We also move tokens to their experts with token shuffling (Section V-B), which neither system does.

## V. IMPLEMENTATION

## A. Within-Layer Correlation: Correlated Expert Placement

As discussed in Section II-B and shown in Figure 2a, some experts within a layer are frequently selected together by the same tokens. We use this structure to reduce all-to-all (A2A) communication with two techniques. Correlated expert placement puts frequently co-selected experts on the same GPU, and a deduplicating dispatcher sends each token to each EP rank at most once.

Figure 3 shows the A2A communication for three configurations: the baseline, with deduplication, and with both deduplication and correlated expert placement. 1 shows contiguous placement with the conventional dispatcher, where each (token, expert) assignment produces a separate A2A row. When a token selects several experts on the same destination rank, its hidden state is sent there several times. 2 adds deduplication. Assignments that share the same (token, destination rank) are merged, so the hidden state is sent to that rank once. 3 adds correlated expert placement, which puts frequently co-selected experts on the same rank. This increases token–expert locality and gives deduplication more rows to merge. Neither technique changes the router’s expert assignments or the expert computation.

a) Deduplicating Dispatcher: Conventional top-k dispatch sends one copy of the token’s hidden state per (token, expert) assignment, even when several selected experts reside on the same EP rank. We instead send one row per unique (token, destination rank) pair. Each row carries the IDs and routing probabilities of the token’s experts on that rank. The receiving rank uses this metadata to give each of its selected experts a copy of the token and to group the copies by expert for the expert computation, so no separate collective is needed for this metadata. The combine path works the same way. The expert rank weights the outputs of the token’s experts by their routing probabilities, sums them, and returns one partial output per (token, rank) pair. Dispatch and combine therefore send one row per token for each rank that holds at least one of its experts. Deduplication is an existing technique [Zhao et al., 2025], and we implement it in Megatron-LM as the basis for our correlated placement algorithms.

b) Correlated Placement: For each MoE layer, we build a correlation graph W, where $W [ e , e ^ { \prime } ]$ counts the tokens that select both experts e and e<sup>′</sup>. We look for a placement that puts the same number of experts on every GPU and maximizes the total weight $W [ e , \bar { e ^ { \prime } } ]$ of the expert pairs that share a GPU. Algorithm 1 in the appendix performs a Kernighan– Lin-style local search [Kernighan and Lin, 1970]. It starts from contiguous placement and swaps pairs of experts between GPUs whenever a swap increases this total. Every GPU holds exactly E/EP experts, so the placement does not change the memory used for expert parameters.

![](images/c0435276a272bc75435ca4ad003ce993892ff239cac529a4c9bb75408b4fe11b.jpg)  
Fig. 3: Reducing A2A with within-layer expert correlation. Each token $T _ { i }$ and its selected experts have the same color.

For EP groups that span multiple nodes, we partition in two steps. We first partition the experts across nodes, which keeps co-selected experts on the same node and reduces inter-node traffic. We then partition each node’s experts across its GPUs, which keeps co-selected experts on the same GPU where possible. The resulting placement is a per-layer permutation of global expert IDs and does not change router decisions or expert parameters.

## B. Cross-Layer Correlation: Token Shuffling

Expert placement decides where experts live and does not change where tokens live. With TP–EP parallelism, the MoE layer runs on the sequence shard that the reduce-scatter after attention assigns to each rank. Each rank thus owns a fixed, contiguous slice of the sequence, and a token’s rank is set by its position alone. Even when the experts a token needs have been placed together on one GPU, the token is actually on that GPU with probability only 1/EP. Thus, with probability 1 − 1/EP the token’s row must still cross the wire in the dispatch and return in the combine. Figure 4 shows this for four ranks under correlated placement. The conventional reduce-scatter places T0 on GPU0, T1 on GPU1 and so on, although T0’s experts sit together on GPU1 and T1’s on GPU2, so their rows still leave the owner’s GPU.

The cross-layer correlation of Section II-B (Figure 2b) lets us move the token as well. The experts a token used at layer ℓ predict, well above chance, which GPU will host its experts at layer ℓ+1, and this prediction is available a whole attention block before the layer-(ℓ+1) router runs. Token shuffling (Figure 5) uses this window. We call the GPU that holds a token in the MoE layer its owner. The reduce-scatter that follows attention first permutes tokens so that each goes to its predicted owner and then scatters them, so the following MoE dispatch finds that most of the token’s experts are now local. The rest of this section describes where ownership can change without extra communication, how the owner is predicted, and how the planner is hidden behind attention.

![](images/e65c6ac58974a57ac3302801c02df1d31aa2d573dba9da7de9b5e52532dac5cf.jpg)  
Fig. 4: TP–EP baseline with correlated placement. Each token stays on the rank of its canonical sequence slice, so its rows still cross the network even when its experts share one GPU.

![](images/b694b2d58b2d2ca487b118548950f2d9cc534976d6e5a71c972f058636296020.jpg)  
Fig. 5: Token shuffling end to end. After the attention of layer ℓ 1 , the shuffle planner predicts each token’s owner rank from its layer-(ℓ−1) routes, and the shuffled reduce-scatter 2 delivers each row to that rank. The dispatch 3 , the expert computation 4 and the combine 5 then find most of the token’s experts local. The shuffle-aware all-gather 6 restores canonical order before the next attention block 7 .

a) Changing Ownership Inside Existing Collectives: The token-shuffling construction requires the TP group and the EP group to be the same set of ranks (TP = EP). In TP– EP, a reduce-scatter follows the attention output projection and returns the sequence to its sharded layout. An all-gather precedes the next block’s QKV projection and reassembles the full sequence on every rank (Figure 5). Both collectives already move every token between ranks. Which rank receives which token in the reduce-scatter is set by indexing, so changing it does not change the number of bytes sent. We therefore fuse a permutation into each collective. The fused reduce-scatter sends every token to its planned owner instead of the rank that holds it in the default contiguous sequence layout. It also moves the token’s row of the residual stream, so the residual addition happens on the new owner. The fused all-gather restores the original token order on the way back, because attention depends on token order. The MoE layer sees a local slice whose tokens mostly have their experts on the same GPU, and the attention layer is unchanged. Token shuffling thus adds no additional collective. Its two fused permutations cost a few milliseconds per step, compared with the hundreds of milliseconds removed from the all-to-all due to decreased communication (Section VI).

b) Predicting the Owner: The shuffle planner answers one question per token: given the experts it just visited, on which rank will its next experts most likely live? Routing is strongly correlated across layers, so the pair of experts a token used at the two preceding MoE layers is a good predictor of where it lands next. Offline, from routing traces and the chosen placement, we count for every MoE layer how often each pair sends the token to each rank; the fewer observations a pair has, the more weight we give to the single preceding expert instead. At runtime the previous layers’ routes are the only input, and the planner adds no collective. The lookup gives each token a preferred owner and a confidence score. Sequence parallelism requires every rank to own exactly T/EP tokens, so tokens are assigned in decreasing confidence, each to its most preferred rank with capacity left. We give the full algorithm in Algorithm 2.

c) Hiding the Shuffle Planner: Prediction and quota assignment cost about a third of the forward all-to-all time at TP16 EP16 so, if executed sequentially, this would forfeit most of the gain. However, crucially, the plan depends only on routes up to layer i and is needed only at the layer-(i+1) reduce-scatter, so we can run the planner on a side stream, concurrently with the expert computation and the next attention block, and join it with a single event wait (Figure 6). In practice the main stream waits about one millisecond per step for the plan (0.02 % of the step), plus one fused permuteplus-residual kernel.

## VI. EXPERIMENTAL RESULTS

All experiments use the two models of Section II-A. Both are 12-layer MoE models with $d _ { \mathrm { m o d e l } } = 4 0 9 6$ and 128 experts of FFN width 1024 per layer, one with top-2 and one with top-6 routing. Both use a sequence-level auxiliary loss with coefficient 10<sup>−2</sup> for load balancing. We train each for 2.1 billion tokens (2000 steps) on FineWeb-Edu [Penedo et al., 2024] with a sequence length of 8192 and a global batch size of 128.

![](images/797de84f8ac8b9a6de91bb5c89450939a11d5b36fab86a1c443c80149b3ac4d6.jpg)  
Fig. 6: The planner runs on a side stream between the layer-i routes and the layer-(i+1) reduce-scatter.

We take all measurements on a cluster of nodes with eight AMD Instinct MI300X GPUs each, fully connected within a node by xGMI (one direct link between each pair of GPUs) and connected across nodes by eight 100 Gb/s RoCE NICs per node, one per GPU. All timing runs load the step-2000 checkpoint of the respective model and run on held-out data from the same corpus, so the routing decisions are those of a trained model. We run each experiment twice. The first run consists of 14 clean steps; we report the median over the last 10 steps, after discarding the first 4 as warm-up (checkpoint loading and communicator initialization). The second run consists of 14 steps with the region profiler and all-to-all metrics enabled, which provide the time and traffic breakdowns, following the same procedure. Speedups are relative to the baseline, which is the Megatron-LM all-to-all dispatcher with contiguous placement. We compare three configurations against it. These are deduplication, deduplication with correlated expert placement, and, on layouts with TP = EP, token shuffling on top of both. The figures label the baseline and the three configurations B, D, P and S. We implement all methods in Megatron-LM [Shoeybi et al., 2019]. The appendix repeats the top-6 placement evaluation with two other load-balancing methods (Figure 11). The correlations are present under all three, with different strengths.

## A. Building Expert Correlation Within and Across Layers

Correlated placement and the token-shuffling predictor both use correlation tables fitted offline from recorded routing. We measure how many tokens are needed for these estimates to stabilize and how quickly they go stale during training. We test both cases on held-out data.

a) A Few Thousand Tokens Suffice: To measure how much routing data the tables need, we fit them on different numbers of tokens from iterations 1983–1995 and evaluate them on held-out iteration 1999. With tables fitted on 1,024 tokens, correlated placement removes 23.6 % of dispatched rows at top-2 EP8 and 57.9 % at top-6. With 4,096 tokens, the Pearson correlation between the fitted and the held-out co-selection matrices is 0.93 for top-2 and 0.98 for top-6, and the placement and prediction results are within one percentage point of those obtained with all 524k tokens. A single microbatch from one rank is therefore enough to fit both tables. The tables capture a property of the trained router and do not depend on the particular tokens used to fit them. This implies that the correlation structure is robustly discoverable from relatively few tokens.

b) The Structure Forms Early and Remains Stable: To measure how early the correlations form, we fit the tables on routing from a single training step and evaluate them on routing from step 1999, the end of training. Tables fitted at step 199 already give about half of the final benefit. At top-2 EP8, the placement built from them removes 14.4 % of dispatched rows, compared with 26.5 % for tables fitted at the end of training. At top-6 the corresponding numbers are 42.7 % and 57.7 %. Tables fitted at step 1000 come within 2–3 percentage points of the final values, and tables fitted at step 1400 or later give nearly the same results as the final tables. The correlation between the fitted and the final co-selection matrices follows the same pattern, rising from 0.4–0.5 at step 199 to above 0.93 from step 1400 on. The predictor improves at the same rate. At top-2 EP8, its hit rate is 12.7 % for tables fitted before the routing correlations have formed, close to the chance level of 12.5 %. It rises to 54 % for tables fitted at step 1000 and 57 % for tables fitted at step 1800. For our 2000-step runs, fitting the tables once in the second half of training is therefore enough. For longer runs, we expect refitting about every 1000 steps to be sufficient, and that each refit needs only one microbatch of recorded routes.

## B. Correlated Expert Placement

a) All-to-All Communication Volume: Correlated expert placement accounts for most of the reduction in all-to-all traffic. Under contiguous placement, deduplication removes only the duplicates that land on the same rank by chance. This removes 6 % of rows at top-2 and 25 % at top-6 for EP8, and 3 % and 12 % at EP16. With correlated placement, the communication volume falls by 1.36× at top-2 EP8 and 2.36× at top-6 EP8 relative to the baseline. The reduction grows with top-k, since more of a token’s experts can share a rank. The effect holds as EP widens and each rank holds fewer experts, at 1.29×/1.22×/1.12× for top-2 and 1.95×/1.54×/1.24× for top-6 at EP16/32/64. Dispatch and combine carry identical byte counts by construction, since every row that goes out comes back, so the byte reduction applies identically to the backward pass also.

b) All-to-All Communication Time: Across nodes, A2A time falls by 1.34×/1.38×/1.34× for top-2 and 1.95×/1.48×/1.35× for top-6 at EP16/32/64. Within a node (EP8), where the collective runs over xGMI, it falls by 1.16× at top-2 and 1.58× at top-6. The reduction appears in all four collectives, dispatch and combine in both the forward and the backward pass.

c) End-to-End Step Time: Non-MoE computation takes nearly the same time in all configurations (blue in Figure 8). The step speedup therefore depends on how much A2A time is saved and on how large a share of the step the A2A takes. At top-2 the A2A share is 13–52 %, and correlated placement speeds up the step by 1.05× at EP8, 1.13× at EP16, 1.14× at EP32 and 1.11× at EP64. At top-6 the share is 24–60 %, and the speedups are 1.14× at EP8, 1.41× at EP16, 1.25× at EP32 and 1.23× at EP64. The speedup is larger at top-6 and is largest when the A2A crosses node boundaries.

![](images/b6ba2d62ce951a3d977768827562105ce42edeb2e6293cab5e0dd68e00d11600.jpg)  
Fig. 7: All-to-All Cost Under Expert Parallelism. A2A time (top) and communication volume per rank (bottom) with EP8– EP64, for top-2 and top-6 routing. B is the baseline, D adds deduplication, and P adds correlated placement to deduplication.

![](images/74f1746f12d5add694a5a8272ef8898ec82b5c09845891cff3b120ac7b88078f.jpg)  
Fig. 8: Step-time breakdown at EP8–EP64 for top-2 and top-6 routing; dashed boxes indicate A2A share.

## C. Token Shuffling

Token shuffling applies where the TP group and the EP group coincide. We evaluate it at TP8 EP8 on one node and TP16 EP16 across two nodes. Figure 9 shows its all-to-all results alongside deduplication and correlated placement, and Figure 10 shows the end-to-end step time.

a) Token–Expert Locality: Without token shuffling, a token’s expert slots are on its own GPU with probability 1/EP, which is 12.5 % at EP8 and 6.3 % at EP16. Token shuffling raises this to 59 % and 53 % for top-2 and to 57 % and 46 % for top-6, close to the offline hit rates in Table I. The dispatcher still counts these rows, but they stay on the GPU and never cross the network. Relative to the baseline, wire bytes fall by 2.4× at EP8 and 2.1× at EP16 for top-2, and by 3.1× and 2.4× for top-6. With placement alone, the reductions are 1.4× and 1.3× for top-2 and 2.4× and 1.9× for top-6.

b) All-to-All Communication Time: At TP16 EP16, token shuffling raises the A2A reduction from 1.55× (placement alone) to 1.74× at top-2 and from 1.95× to 2.63× at top-6. On a single node (TP8 EP8), token shuffling reduces bytes by a similar or larger factor, but reduces the time cost by a smaller factor. The A2A reductions are 1.40× at top-2 and 1.46× at top-6. At top-6 on one node, the A2A reduction with token shuffling is slightly below that of placement alone (1.48×). On xGMI, the collective time is set by the busiest GPU pair and does not follow the aggregate bytes. Concentrating a token’s rows on one owner raises the traffic on the busiest link even as it lowers the total. The two fused permutations of token shuffling add a few milliseconds per step. The planner runs behind attention, and the main stream waits about one millisecond per step for it (Section V-B).

![](images/7eff41e166d5f6115d7a6ef10cd9383e557ed6650344f686ae5580ceed589c42.jpg)  
Fig. 9: All-to-All Cost on the TP = EP Layouts. Same encoding as Figure 7, with token shuffling as the fourth configuration (S). The shuffle places tokens on the GPUs that hold their experts, so more of the dispatch stays on the device and less traffic crosses the network.

![](images/c7f7e1a9138c9debd617b71cd491069e81f8deb77df2bf2a28c3bcd87d348a4b.jpg)  
Fig. 10: Step-time breakdown on the TP = EP layouts (TP8 EP8, TP16 EP16) with token shuffling added as S. Same encoding as Figure 8.

c) End-to-End Step Time: On the TP = EP layouts (Figure 10) the tensor-parallel collectives inflate the non-MoE part, so the A2A is a smaller share of the step: 9–21 % at top-2 and 18–38 % at top-6. Token shuffling cuts the A2A time itself by 1.40× and 1.74× at top-2 and by 1.46× and 2.63× at top-6 (TP8 EP8 and TP16 EP16), but the step speedup is limited by how much of the step the A2A takes. At top-2 it stays within 1.01–1.06×; at TP16 EP16 it grows from 1.03× with deduplication to 1.05× with correlated placement and 1.06× with token shuffling. At top-6 the gains are 1.08× at

TP8 EP8 and 1.34× at TP16 EP16, where shuffling improves the speedup from 1.28× to 1.34×.

## VII. CONCLUSION

We showed that expert routing in MoE pretraining is correlated both within a layer and across layers, and that these correlations form within about one billion tokens and then stay stable. Correlated expert placement uses the within-layer correlation to put co-selected experts on the same GPU, and a deduplicating dispatcher sends each token at most once to its destination GPU. Token shuffling uses the cross-layer correlation to move each token to the GPU predicted to hold its next experts, subject to per-rank capacity constraints. Moreover, this shuffle occurs inside the reduce-scatter that sequence parallelism already performs, leading to almost no overhead. Neither method changes routing decisions or expert parameters. On a cluster of nodes with eight MI300X GPUs each, the two methods reduce all-to-all time by 1.16–2.63× and end-to-end step time by up to 1.41×, with the largest gains at top-6 when the EP group spans several nodes. Our evaluation covers one 12-layer model with top-2 and top-6 routing, trained for 2.1B tokens. We believe the expert correlation holds across model families and scales, and leave other parallelism layouts, such as context parallelism, and EP inference to future work.

## REFERENCES

Yutian Chen, Cong Li, Yucheng Wang, and Ming Wei. MoonEP: A perfectly balanced expert parallelism library via dynamic redundant experts. https://github.com/ MoonshotAI/MoonEP, 2026.

Damai Dai, Chengqi Deng, Chenggang Zhao, Runxin Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1280–1297, 2024.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Yu Han, Lehan Pan, Jie Peng, Ziyang Tao, Wuyang Zhang, and Yanyong Zhang. GRACE-MoE: Grouping and replication with locality-aware routing for efficient distributed MoE inference. arXiv preprint arXiv:2509.25041, 2025.

Albert Q. Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al. Mixtral of experts. arXiv preprint arXiv:2401.04088, 2024.

Brian W. Kernighan and Shen Lin. An efficient heuristic procedure for partitioning graphs. The Bell System Technical Journal, 49(2):291–307, 1970.

Kimi Team. Kimi K3: Open frontier intelligence. arXiv preprint arXiv:2607.24653, 2026.

Vijay Anand Korthikanti, Jared Casper, Sangkug Lym, Lawrence McAfee, Michael Andersch, Mohammad Shoeybi, and Bryan Catanzaro. Reducing activation recomputation in large transformer models. Proceedings of Machine Learning and Systems, 5:341–353, 2023.

Dmitry Lepikhin, HyoukJoong Lee, Yuanzhong Xu, Dehao Chen, Orhan Firat, Yanping Huang, Maxim Krikun, Noam Shazeer, and Zhifeng Chen. GShard: Scaling giant models with conditional computation and automatic sharding. In International Conference on Learning Representations, 2021.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu

Zhang, Chong Ruan, et al. DeepSeek-V3 technical report. arXiv preprint arXiv:2412.19437, 2024.

Dennis Liu, Zijie Yan, Xin Yao, Tong Liu, Vijay Korthikanti, Evan Wu, Shiqing Fan, Gao Deng, Hongxiao Bai, Jianbin Chang, et al. MoE parallel folding: Heterogeneous parallelism mappings for efficient large-scale MoE model training with Megatron Core. arXiv preprint arXiv:2504.14960, 2025.

Shuqing Luo, Pingzhi Li, Jie Peng, Yang Zhao, Yu Cao, Yu Cheng, and Tianlong Chen. Occult: Optimizing collaborative communications across experts for accelerated parallel MoE training and inference. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 41235–41253, 2025.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, An-ˇ ton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 37, pages 30811–30849, 2024.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixtureof-experts layer. In International Conference on Learning Representations, 2017.

Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv preprint arXiv:1909.08053, 2019.

Siddharth Singh, Olatunji Ruwase, Ammar Ahmad Awan, Samyam Rajbhandari, Yuxiong He, and Abhinav Bhatele. A hybrid tensor-expert-data parallelism approach to optimize mixture-of-experts training. In Proceedings of the 37th International Conference on Supercomputing, pages 203– 214, 2023.

Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou, and Yang You. OpenMoE: An early effort on open mixture-of-experts language models. In International Conference on Machine Learning, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Jinghan Yao, Quentin Anthony, Aamir Shafi, Hari Subramoni, and Dhabaleswar K. Panda. Exploiting inter-layer expert affinity for accelerating mixture-of-experts model inference. In 2024 IEEE International Parallel and Distributed Processing Symposium (IPDPS), pages 915–925. IEEE, 2024.

Mingshu Zhai, Jiaao He, Zixuan Ma, Zan Zong, Runqing Zhang, and Jidong Zhai. SmartMoE: Efficiently training sparsely-activated models through combining offline and online parallelization. In 2023 USENIX Annual Technical Conference (USENIX ATC 23), pages 961–975, 2023.

Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi

Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. DeepEP: an efficient expert-parallel communication library. https://github.com/deepseek-ai/DeepEP, 2025.

## A. Experimental Setup

1) Model Architecture: Table II summarizes the architecture and pre-training configuration of the model used throughout our experiments. Unless stated otherwise, every result uses this configuration, including its router (pre-softmax top-k with $k = 2 \ { \mathrm { o r } } \ k = 6$ and a sequence-level auxiliary loss of $1 0 ^ { - 2 } )$ . The router ablation in Section B keeps the architecture fixed and varies only the routing recipe: the pre-softmax router with auxiliary loss $1 0 ^ { - 2 }$ and $1 0 ^ { - 1 }$ , and the auxiliary-loss-free router of DeepSeek-V3 [Liu et al., 2024] (sigmoid scores with a per-expert bias updated at rate $1 0 ^ { - 3 }$ and a $1 0 ^ { - 4 }$ sequence-level loss).

<table><tr><td></td><td>Symbol</td><td>Value</td></tr><tr><td>Transformer</td><td></td><td></td></tr><tr><td>Layers</td><td>L</td><td>12, every layer an MoE layer</td></tr><tr><td>Model width</td><td> $d _ { \mathrm { m o d e l } }$ </td><td>4096</td></tr><tr><td>Attention heads</td><td>Nhead</td><td>16</td></tr><tr><td>Head dimension</td><td> $d _ { \mathrm { h e a d } }$ </td><td>32</td></tr><tr><td>Attention width</td><td> $n _ { \mathrm { h e a d } } d _ { \mathrm { h e a d } }$ </td><td>512</td></tr><tr><td>Normalization</td><td></td><td>pre-LayerNorm,  $\epsilon = 1 0 ^ { - 5 }$ </td></tr><tr><td>Positional encoding</td><td></td><td>learned absolute</td></tr><tr><td>Vocabulary</td><td>|V|</td><td>151,669 (Qwen3 tokenizer; padded to 151,680)</td></tr><tr><td>Biases / dropout</td><td></td><td>none / 0</td></tr><tr><td>Embeddings</td><td></td><td>input and output tied</td></tr><tr><td>Mixture of experts</td><td></td><td></td></tr><tr><td>Experts per layer</td><td> $E$ </td><td>128, no shared expert</td></tr><tr><td>Experts per token</td><td> $k$ </td><td>2 or 6</td></tr><tr><td>Expert hidden size</td><td> $d _ { \mathrm { f f } }$ </td><td> $1 0 2 4 ( d _ { \mathrm { m o d e l } }  d _ { \mathrm { f f } }  d _ { \mathrm { m o d e l } } , \mathrm { G E L U ) }$ </td></tr><tr><td>Routing</td><td></td><td>top-k of E, dropless</td></tr><tr><td>Router</td><td></td><td>FP32 linear gate; softmax over all  $E ,$  then top-k;</td></tr><tr><td>Load-balancing loss</td><td> $\alpha _ { \mathrm { a u x } }$ </td><td>gate weights = selected softmax probabilities sequence-level auxiliary loss,  $1 0 ^ { - 2 }$ </td></tr><tr><td>Dispatch</td><td></td><td>all-to-all, grouped GEMM</td></tr><tr><td>Size</td><td></td><td></td></tr><tr><td>Total parameters</td><td> $N$ </td><td>13.6 B (13.0 B transformer, 0.6 B embeddings)</td></tr><tr><td>Expert parameters</td><td> $N _ { \mathrm { e x p } }$ </td><td>12.9 B (1.07 B per layer)</td></tr><tr><td>Active per token</td><td> $N _ { \mathrm { a c t } }$ </td><td>≈ 0.31 B at top-2, ≈ 0.71 B at top-6 (non-embedding)</td></tr><tr><td>Pre-training</td><td></td><td></td></tr><tr><td>Data</td><td></td><td>FineWeb-Edu, 98/2 train/validation split</td></tr><tr><td>Sequence length</td><td> $S$ </td><td>8192</td></tr><tr><td>Global batch</td><td> $B$ </td><td>128 sequences  $( B \cdot S \approx 1 . 0 5 \mathbf { M }$  tokens per step)</td></tr><tr><td>Steps / tokens</td><td> $T$ </td><td> $2 0 0 0 \mathrm { ~ / ~ } \approx \mathrm { 2 . 1 ~ B }$  tokens</td></tr><tr><td></td><td></td><td> $\mathrm { A d a m W } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5$ </td></tr><tr><td>Optimizer Learning rate</td><td></td><td>, weight decay 0.1, gradient clip 1.0  $1 0 ^ { - 5 }$  peak, 200 warm-up steps, cosine decay toward  $1 0 ^ { - 6 }$ </td></tr><tr><td>Initialization</td><td> $\eta$ </td><td> $0 . 0 0 8$ </td></tr><tr><td>Precision</td><td> $\sigma _ { \mathrm { i n i t } }$ </td><td>(normal) BF16; FP32 router and optimizer state</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Parallelism</td><td> $\mathrm { T P } / \mathrm { E P }$ </td><td>1 / 8, one 8-GPU node, micro-batch 2</td></tr></table>

TABLE II: Model and pre-training configuration used in all experiments.

2) Training Framework: We use Megatron-LM [Shoeybi et al., 2019] as our training framework and implement our methodology (Section V) on top of it, leaving the router, expert computation, optimizer, and training loop unchanged. Specifically, our implementation modifies three components. (i) Token dispatcher: We add a deduplicating variant of the all-to-all dispatcher that sends one row per unique (token, destination rank) pair instead of one per (token, expert) pair. Expert IDs and gate weights are carried with the activation, and the receiver expands the rows into expert-major order. The combine direction applies the same deduplication in reverse.

(ii) Expert placement: We extend the dispatcher’s contiguous expert-to-rank assignment to support an arbitrary per-layer permutation. The permutation is applied to the routing map, and checkpoint expert weights are keyed by global expert ID, allowing the same checkpoint to be loaded with a different expert placement.

(iii) Token shuffling: For TP=EP layouts, we modify attention’s sequence-parallel reduce-scatter to assign each token to an owner rank based on the second-order affinity table. Canonical token order is restored around attention, the output layer, and the loss. The route IDs the planner needs are included in the dispatcher’s existing metadata all-gather, requiring no additional collective.

Routing decisions used to build the offline expert placement and affinity tables are recorded through a capture hook in the router. All variants train the same model and produce the same loss, differing only in how tokens are placed and moved across ranks.

## B. Ablation Over Routing Strategies

a) Load-Balancing Methods: Our methods depend on the router producing correlated expert choices. To test how the load-balancing method affects this, we trained two more top-6 models that differ from the main top-6 model only in the balancing method. The first uses a sequence-level auxiliary loss ten times stronger than the main model’s (coefficient 10<sup>−1</sup> instead of 10<sup>−2</sup>). It gives the most balanced expert load of the three, with a maximum-to-mean load ratio of 1.20. The second uses the auxiliary-loss-free method of DeepSeek-V3 [Liu et al., 2024], which adds a per-expert bias to a sigmoid router. Its load ratio is 1.37, and it reaches the lowest loss of the three (4.39, against 4.46 for the main model and 4.64 for the stronger auxiliary loss). We evaluate the same configurations on all three models.

b) Gains Follow Each Router’s Co-Selection Structure: Correlated placement remains effective across all three routing strategies. With correlated placement, deduplication removes 58 %, 51 %, and 47 % of dispatched rows at EP8 for the main model, the stronger auxiliary-loss router, and the auxiliary-loss-free router, respectively, and 49 %, 40 %, and 36 % at EP16. In comparison, contiguous placement removes only about 25 % at EP8 and 13 % at EP16 for every router (Figure 11, right). These reductions translate into consistent end-to-end gains across routing strategies: at TP1 EP16, step speedups are 1.41×, 1.30×, and 1.26×, respectively, and remain at least 1.20× on every EP16 layout (Figure 11, left). Thus, the benefit of correlated placement does not depend on a particular routing strategy; its magnitude instead follows the co-selection structure produced by each router.

Load balance affects the realized communication time because an all-to-all must wait for the most heavily loaded rank. At EP16, for example, the main model removes 49 % of dispatched rows compared with 40 % for the stronger auxiliary-loss router, but its speedup advantage is smaller (1.41× versus 1.30×), consistent with its higher load ratio (1.52 versus 1.20). Correlated placement and deduplication therefore target communication volume through expert co-selection, rather than communication skew. Methods that directly reduce skew, such as relocating or replicating heavily loaded experts [Zhai et al., 2023, Liu et al., 2024, Chen et al., 2026, Han et al., 2025], address a separate source of all-to-all cost and are complementary to our approach.

step speed-up vs baseline, dedup + correlated placement  
![](images/aea233070951afb45a0dd33db7e36aad90890852bee123212e2b780f0b8a06be.jpg)

deduplication under correlated placement  
![](images/55bc2a90bf16a9fde70a23b5e0902b6b48b861d166935c234d55549ce0f2aad2.jpg)  
M: main recipe (seq-aux 1e-2), max/mean load 1.52 X: 10× stronger aux loss (seq-aux 1e-1), max/mean load 1.20 F: aux-loss-free (DeepSeek-V3), max/mean load 1.37 contiguous placement  
Fig. 11: Router ablation on three top-6 models that differ only in load balancing: the main recipe (sequence auxiliary loss $1 0 ^ { - 2 } )$ $\textbf { a } 1 0 \times$ stronger auxiliary loss $( 1 0 ^ { - 1 } )$ , and the DeepSeek-V3 auxiliary-loss-free router. Left: step speedup of deduplication + correlated placement over that model’s baseline, per layout. Right: the structure each router leaves to exploit, as the deduplication rate under correlated placement.

## C. Per-Step Measurements

Figure 12 shows the ten measured steps behind every reported speedup.

![](images/5994d5286b53546870febd1fe450d630e10fb59d2bcd39302407253c173da7d0.jpg)  
Fig. 12: Step time (left) and A2A volume per rank (right) for the ten measured steps of each configuration, after the four warm-up steps. Data point labels give the ratio of the baseline median to the median of that configuration.

## D. Algorithms

We give algorithms for the two components described in Sections V-A and V-B, using the notation of Table III.

TABLE III: Notation.  
Symbol Meaning   
$t , T$ a token; number of tokens in the global sequence   
$E , k$ experts per MoE layer; experts selected per token $( \mathrm { t o p } { - } k )$   
$P , N$ EP ranks (in Alg. 2 also the TP size); nodes   
$\ell , m$ MoE layer index; MoE position whose token owners are being planned   
$S _ { \ell } ( t )$ the k experts the router selects for token t at layer ℓ   
$W [ e , e ^ { \prime } ]$ number of tokens that select both e and $e ^ { \prime }$ (co-selection count)   
$\pi _ { \ell } ( e )$ EP rank that holds expert e at layer ℓ   
$\nu ( e ) , \rho _ { n } ( e )$ node of expert e; its rank within node n   
$I _ { \mathrm { m a x } }$ maximum number of swaps per partition   
$F _ { m } [ e , r ]$ fraction of position-m expert slots on rank r for tokens that selected e at $m { - } 1$   
$S _ { m } [ e ^ { \prime } , e , r ]$ as $F _ { m } ,$ , conditioned on $e ^ { \prime }$ at $m { - } 2$ and e at $m { - } 1$   
$n _ { m } [ e ^ { \prime } , e ]$ number of trace tokens behind $S _ { m } [ e ^ { \prime } , e , \cdot ]$ (support)   
$Q _ { m } [ \mathrm { t o k } , r ]$ as $F _ { m } ,$ , conditioned on the token id (optional)   
$\beta , w _ { \mathrm { m a x } }$ back-off strength; maximum weight on a conditioned table   
$d _ { t }$ predicted distribution of token $t \mathrm { { } } \mathrm { { \dot { s } } }$ position-m expert slots over ranks   
$d _ { t } ^ { ( 1 ) } , d _ { t } ^ { ( 2 ) }$ largest and second-largest entries of $d _ { t }$   
$\mathrm { o w n e r } ( t )$ rank that holds token t’s row at MoE position m

a) Correlated placement $( A l g . l ) .$ : The placement is computed offline, once per model. For each MoE layer we count co-selections W on the recorded routes and partition the experts into equal-size groups that maximize the within-group count. PARTITION starts from contiguous groups and applies the best improving pairwise swap until none remains or $I _ { \mathrm { m a x } }$ swaps are made; since every move is a swap, each rank keeps exactly $E / P$ experts. On $N > 1$ nodes the partition is hierarchical: experts are first split across nodes, which targets cross-node traffic, and then across the $P / N$ ranks of each node.

Algorithm 1 Correlated placement   
1: Input: routing sets $\{ S _ { \ell } ( t ) \}$ , experts E, EP ranks $P ,$ nodes N, budget $I _ { \mathrm { m a x } }$   
2: Output: balanced placement π<sub>ℓ</sub> per MoE layer   
3: for each MoE layer ℓ do   
4: $\begin{array} { r } { W [ e , e ^ { \prime } ]  \bar { \sum _ { t } } \mathbf { 1 } [ e , e ^ { \prime } \in \mathcal { S } _ { \ell } ( t ) ] } \end{array}$   
5: if ${ \bar { N } } = 1$ then   
6: $\pi _ { \ell } \gets \mathrm { P A R T I T I O N } ( W , P , I _ { \mathrm { m a x } } )$   
7: else   
8: $\nu \gets \mathrm { P A R T I T I O N } ( W , N , I _ { \mathrm { m a x } } )$   
9: for node $n = 0 , \ldots , N - 1$ do   
10: $\mathcal { M } _ { n } \gets \{ e : \nu ( e ) = n \}$   
11: ρ<sub>n</sub> ← PARTITION $( W _ { \mathcal { M } _ { n } } , \frac { P } { N } , I _ { \mathrm { m a x } } )$   
12: $\begin{array} { r } { \pi _ { \ell } ( e )  n \frac { P } { N } + \rho _ { n } ( e ) ~ \forall e ^ { * } \in \mathcal { M } _ { n } } \end{array}$   
13: end for   
14: end if   
15: end for   
16: procedure PARTITION(W, G, I<sub>max</sub>) ▷ balanced   
17: Initialize G contiguous groups of $| W | / G$ experts   
18: for $i = 1 , \ldots , I _ { \mathrm { m a x } }$ do   
19: (a, b) ← swap with the largest gain in intra-group W   
20: if the gain is not positive then break   
21: Swap a and b   
22: end for   
23: return expert-to-group assignment   
24: end procedure

b) Token shuffling (Alg. 2): The prediction tables are fit offline on the same traces as the placement; the planner runs at every MoE position during training. It first predicts, for each token, how its position-m expert slots will spread over rank (lines 3–7): the first- and second-order tables are mixed per expert pair, with more weight on the second-order table the more trace tokens support it. It then assigns owners greedily, most confident token first, under a per-rank cap of $T / P$ so the layout stays balanced (lines 8–11). The planner runs on a side stream during attention, and the main stream waits for it only at the attention output projection, where the reduce-scatter writes each row to its owner.

Algorithm 2 Token shuffling $\overline { { ( \mathrm { T P } = \mathrm { E P } = P ) } }$   
1: Offline: for each MoE position m, tables $F _ { m } , S _ { m }$ , supports $n _ { m }$ (optionally $Q _ { m } )$ from routing traces under placement π; back-off $\beta ,$   
cap $w _ { \mathrm { m a x } } .$   
2: Input: top-k sets $S _ { m - 1 } ( t ) , S _ { m - 2 } ( t )$ of all T tokens, gathered with the dispatcher’s metadata all-gather at MoE position $m { - } 1 .$   
// The shuffle planner runs on a side stream, overlapped with the expert computation at m−1 and the attention that follows; see V-Bc.   
3: for each token t do ▷ predict owner-rank distribution   
4: $\begin{array} { r } { d _ { t } \gets \frac { 1 } { k ^ { 2 } } \sum _ { e \in S _ { m - 1 } ( t ) } \sum _ { e ^ { \prime } \in S _ { m - 2 } ( t ) } \big [ \big ( 1 - w _ { e ^ { \prime } e } \big ) F _ { m } [ e , \cdot ] + w _ { e ^ { \prime } e } S _ { m } [ e ^ { \prime } , e , \cdot ] \big ] , } \end{array}$   
where $w _ { e ^ { \prime } e } = \mathrm { m i n } \big ( n _ { m } [ e ^ { \prime } , e ] / ( n _ { m } [ e ^ { \prime } , e ] + \beta ) , w _ { \mathrm { m a x } } \big )$   
5: Optionally mix in $Q _ { m } [ \mathrm { { t o k } } ( { \dot { t } } ) , \cdot ]$ with the same back-off   
6: $\begin{array} { r } { \dot { d _ { t } } \gets d _ { t } \dot { / } \sum _ { r } d _ { t } [ r ] } \end{array}$   
7: end for   
8: cap[r] $ T / P$ for all $r$   
9: for each t in decreasing order of $d _ { t } ^ { ( 1 ) } - d _ { t } ^ { ( 2 ) }$ do ▷ most confident first   
10: owne $\operatorname { r } ( t ) \gets \arg \operatorname* { m a x } _ { r : \operatorname { c a p } [ r ] > 0 } d _ { t } [ r ] ;$ cap[owner(t)] ← cap[owne $\boldsymbol { \cdot } ( t ) ] - 1$   
11: end for   
12: Build the new token order, its inverse, and the residual-row indices   
// Main stream, at the attention output projection   
13: Wait for the planner   
14: Reduce-scatter the projected rows to owner(·); permute and add the residual in one fused kernel   
15: MoE position m dispatches from this layout, which persists until the next plan

## E. Author Contributions

a) Radha Gulhane: Implemented the methods and carried out all experiments. Led initial ideation and exploration.

b) Quentin Anthony: Contributed to the initial ideation and shaping of the idea, helped write the paper, and provided substantial guidance throughout.

c) Beren Millidge: Participated in technical discussions and provided suggestions throughout the project, and helped write the paper.