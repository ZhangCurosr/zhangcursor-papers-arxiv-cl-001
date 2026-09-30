# SEED: Self-Speculative Decoding via Implicit Encoder–Decoder

Hankun Lin<sup>∗</sup>, Patrick Pynadath<sup>∗</sup>, Ruqi Zhang Department of Computer Science, Purdue University, USA {lhankun, ppynadat, ruqiz}@purdue.edu

## Abstract

Self-speculative decoding accelerates large language model (LLM) inference by drafting tokens from the target model itself, but faces a sharp tradeoff between the quality and cost of the draft. Early-exit methods produce drafts cheaply by terminating computation at intermediate layers, but forgo the deeper representations that later layers provide and thus suffer in draft quality. Multi-token prediction preserves draft quality by emitting from the model’s final hidden states, but pays for a full forward pass to produce those states at every drafting step. We propose self-speculative encoder-decoder (SEED), a self-speculative method that obtains high-quality drafts cheaply by reusing the deep contextual representations already computed during verification. We reinterpret the standard decoder-only transformer as an implicit encoder–decoder: the first layers (encoder) build deep contextual representations, and the last few layers (decoder) emit tokens from them. Encoding and verification are merged into a single step: verification is performed by the full encoder–decoder, and the contextual representations of the verified prefix are cached for reuse during drafting. Drafting is therefore very fast: between verifications, the lightweight decoder drafts multiple tokens autoregressively, each conditioned on the cached representations and on preceding drafts. Experiments across multiple benchmarks show that SEED achieves up to 2.7× average speedup on 4B-scale models, outperforming both early-exit and MTP-style self-speculative baselines and running 28% faster than the state-of-the-art EAGLE-3, while preserving or even improving the generation quality of standard autoregressive fine-tuning. Code is available at https://github.com/lhk2004/SEED.

## 1 Introduction

Large language model (LLM) inference is memory-bound rather than compute-bound [31, 13]: generating each token requires a full forward pass through the model, yet that pass produces only a single output, leaving the GPU’s parallel compute capacity underutilized. Speculative decoding [19, 7] amortizes this cost by using a cheap draft model to propose multiple candidate tokens that are verified in parallel by the target model in a single forward pass. However, maintaining a separate, well-aligned drafter complicates deployment. This has motivated self-speculative decoding, where drafting and verification are handled within a single unified model.

Self-speculative methods face a sharp tradeoff between the quality and cost of the draft. Early-exit or layer-skipping methods [39, 12, 35, 9] terminate the forward pass at an intermediate layer, making drafting cheap but forcing the drafter to predict from shallow representations, which limits draft quality. By contrast, multi-token prediction (MTP) methods [14, 33] emit drafts from the model’s final hidden states, preserving representational depth but requiring a full forward pass at every drafting step. This raises a question: can drafts be as deep as thefull model yet as cheap as afew layers?

![](images/a5ef2ecb6df5d0f846a87776df7f4f5be5af5d4bfaf1a7d374953e485b80bf7c.jpg)  
Figure 1: Overview of SEED. SEED partitions a 28-layer model into a 26-layer encoder and a 2-layer decoder. During drafting, the decoder generates drafts by cross-attending to the deep contextual KV cache produced during the previous verification step, avoiding repeatedly invoking the costly encoder. This design contrasts with two existing self-speculative decoding paradigms: early-exit drafting (instantiated using LayerSkip [12]) discards the deep representations from later layers, sacrificing draft quality, while multi-token prediction (instantiated using Apple MTP [33]) preserves draft quality by maintaining full representational depth but still requires a full forward pass at every drafting step. On GSM8K with draft length 4, SEED achieves substantially higher decoding throughput than early-exit drafting (∼ 2.0×) and MTP (∼ 1.6×).

Our answer adapts the encoder-decoder amortization principle developed in diffusion language models [3] to autoregressive speculative decoding. We view the standard decoder-only transformer as an implicit encoder-decoder: the first layers act as an encoder that builds deep contextual representations of the prefix, while the last few layers act as a decoder that autoregressively emits the next token from those representations. The crucial observation is that under this view, verification already does all the encoding work needed for subsequent drafting — the deep representations the drafter needs are already in the KV cache. Building on this, we propose self-speculative encoder-decoder (SEED), a self-speculative decoding method that trains the lightweight decoder to generate new draft tokens by conditioning on the verifier’s deepest features, without invoking the encoder again until the next verification step. Each draft token therefore costs only a thin-decoder forward pass yet conditions on the verifier’s full-depth representation of the prefix. Unlike prior methods that require careful architectural modifications or nuanced training procedures, SEED is remarkably simple: it adds only a single auxiliary loss on top of standard supervised fine-tuning. Figure 1 provides an overview of SEED.

Across math, coding, general knowledge, and summarization benchmarks, SEED achieves up to a 2.7× average speedup on 4B-scale models while matching or improving the generation quality of standard autoregressive fine-tuning. It consistently outperforms both early-exit and MTP-based self-speculative baselines and runs 28% faster than the state-of-the-art EAGLE-3. Moreover, our analysis reveals that, like MTP methods, SEED encourages the model to encode more predictive information about future tokens, which contributes to improving both draft quality and downstream task performance.

## 2 Related Work

There has been substantial interest in accelerating large language model (LLM) inference. While classical speculative decoding requires a small auxiliary model or multiple heads [34, 5, 2, 22], most relevant to our work are self-speculative decoding methods, which avoid using an additional draft model by drafting from the target model itself. These methods fall broadly into two categories, multi-token prediction (MTP) and exiting early by skipping layers.

Multi-Token Prediction. Multi-token prediction (MTP) accelerates inference by predicting multi ple tokens in a single forward pass, increasing the model’s output per pass while keeping per-pass cost roughly fixed. Originally motivated by denser training signals, early work modified the pretraining objective to enable parallel token prediction [32, 14, 23], but reliance on pretraining from scratch makes these approaches impractical for accelerating deployed models. More recent work adapts the MTP pipeline for faster inference on existing models [4, 33, 28, 26, 6, 11, 16, 40]. A separate line replaces the standard MTP formulation with alternative decoding schemes such as Jacobi decoding [18, 15] or diffusion-based generation [3, 25, 8], but these either sacrifice quality by breaking causal dependencies or substantially complicate the training objective.

Early-Exit Methods. Early-exit methods reuse components of the verifier itself as the drafter, typically skipping later layers and using only the initial layers for drafting. This can be achieved through additional training [12, 24] or as a purely inference-time decision [39, 35, 9, 38]. SEED shares the layer-skipping spirit but inverts the design: rather than skipping later layers, we skip the initial layers and draft from only the final layers of the base model. This asymmetric choice lets the drafter condition on full-depth contextual representations, producing higher-quality draft tokens than early-exit alternatives.

For a more detailed comparison with existing LLM acceleration methods, see Appendix B.

## 3 Preliminaries

We represent a standard transformer as a function $f$ that maps an input sequence $\begin{array} { r l } { X } & { { } = } \end{array}$ $( X _ { 1 } , \bar { X _ { 2 } } , \ldots , X _ { n } )$ of tokens from a vocabulary $\nu$ to a probability distribution over the next token, $f ( X ) \in \triangle ( \mathcal { V } )$ . The transformer is composed of l layers $( f _ { 1 } , \overrightharpoon { f _ { 2 } } , \ldots , f _ { l } )$ . Each layer $f _ { i }$ takes as input the hidden states from the previous layer, $H ^ { i - 1 } = ( H _ { 1 } ^ { i - 1 } , H _ { 2 } ^ { i - 1 } , \dots , H _ { n } ^ { i - 1 } )$ , and outputs updated representations $H ^ { i } = ( H _ { 1 } ^ { i } , H _ { 2 } ^ { i } , \ldots , \bar { H _ { n } ^ { i } } )$ . Each self-attention layer additionally computes keys and values that are cached to avoid redundant computation; we denote the KV cache available before processing token $X _ { t }$ at layer i as $K V _ { < t } ^ { i } .$ . For brevity, we omit explicit notation for KV cache updates in the method description that follow.

The first layer $f _ { 1 }$ operates on the token embeddings $H ^ { 0 } = W _ { \mathrm { i n } } X$ , where $W _ { \mathrm { i n } }$ is the embedding matrix. The output of the final layer, $H ^ { l }$ , is projected back to vocabulary space via a second embedding matrix $\bar { W } _ { \mathrm { o u t } }$ to obtain the next-token distribution,

$$
P ( \cdot \mid X _ { \leq n } ) = \mathrm { s o f t m a x } ( H _ { n } ^ { l } W _ { \mathrm { o u t } } ^ { \top } ) .\tag{1}
$$

Autoregressive transformer language models are trained by minimizing the standard next-token cross-entropy loss

$$
\mathcal { L } _ { \mathrm { C E } } = \mathbb { E } _ { X \sim \mathcal { D } } \left[ \sum _ { j = 1 } ^ { | X | } - \log f ( X _ { j } \mid X _ { < j } ) \right] ,\tag{2}
$$

where D denotes the training data distribution.

## 4 Self-Speculative Encoder-Decoder

We propose self-speculative encoder-decoder (SEED), a self-speculative decoding method that obtains high-quality drafts cheaply while keeping the model architecture unchanged. We first introduce the encoder-decoder reinterpretation of the standard decoder-only transformer, and explain why it enables cheap yet accurate drafting. We then present the training algorithm, which augments the standard next-token prediction objective with an additional speculative drafting objective. This trains the decoder to draft from raw token embeddings while cross-attending to cached deep representations. Finally, we describe the inference procedure, including a fixed-length drafting and a dynamic drafting with tree-based verification.

## 4.1 Method

A key observation underlying SEED is that a standard decoder-only transformer can be naturally interpreted as an implicit encoder-decoder architecture. This reinterpretation exposes a structural asymmetry that we exploit to make cheap yet accurate drafting.

Decoder-Only Transformers as Implicit Encoder-Decoders. Consider a transformer with l layers $( f _ { 1 } , \ldots , f _ { l } )$ . We partition the model at some layer $l ^ { \prime } ,$ treating the first l<sup>′</sup> layers as an encoder that builds deep contextual representations of the prefix, and the remaining layers as a decoder that emits the next token from those representations. The encoder typically accounts for the majority of layers $( \mathbf { e . g . , } l ^ { \prime } = 2 6$ out of $l = 2 8$ in our experiments).

Formally, we define $f _ { e n c } = f _ { 1 : l ^ { \prime } }$ and $f _ { d e c } = f _ { l ^ { \prime } + 1 : l } .$ , with $K V _ { < n } ^ { e n c }$ and $K V _ { < n } ^ { d e c }$ denoting the cached keys and values from all positions preceding n in the encoder and decoder respectively. After processing token $X _ { n }$ , the corresponding caches are updated to $K V _ { < n } ^ { e n c }$ and $K V _ { < n } ^ { d e c }$ . For a context sequence $X _ { \leq n }$ , the encoder produces a deep contextual embedding of the current input token $X _ { n } { \mathrm { : } }$

$$
H _ { n } ^ { l ^ { \prime } } = f _ { e n c } ( \cdot \mid X _ { n } , K V _ { < n } ^ { e n c } ) ,\tag{3}
$$

and the decoder predicts the next token from this representation while attending to the cached context:

$$
X _ { n + 1 } \sim \mathrm { s o f t m a x } \Big ( f _ { d e c } ( \cdot \ : | \ : H _ { n } ^ { l ^ { \prime } } , K V _ { < n } ^ { d e c } ) \Big ) .\tag{4}
$$

This decomposition is purely conceptual as it is exactly equivalent to standard autoregressive decoding, but it makes a structural asymmetry explicit: the encoder accounts for most per-token computation while the decoder is a thin stack of only a few layers.

Standard autoregressive generation does not exploit this asymmetry, since the encoder is re-invoked for every new token. We propose to exploit it for self-speculative decoding: because context evolves slowly, encoding can be performed infrequently, and the decoder alone can emit tokens conditioned on a fixed contextual representation.

Verification Already Does the Encoder’s Work for Drafting. Remarkably, we do not need to run the encoder separately to obtain the contextual representation. The verification step already does this work.

To verify a batch of proposed drafts, the full encoder-decoder runs on these drafts and produces $K V _ { < n } ^ { d e c }$ , the deep contextual representations up to the last verified draft token $X _ { n - 1 }$ . Because the verifier has processed the entire prefix $X _ { < n }$ during this forward pass, it should be able to predict another bonus token $X _ { n }$ through the same pass. Thus, the cache contains deep contextual representations only for tokens preceding $X _ { n }$ , which are precisely the representations needed for subsequent drafting. Verification and encoding therefore merge into a single step: one forward pass of the full model simultaneously verifies the drafts and prepares the deep contextual cache that the drafter consumes in the next round.

Cheap Drafting Between Verifications. With the deep representations already cached during verification, the decoder no longer requires a fresh encoder call to begin drafting. We instead feed the raw token embedding $H _ { n } ^ { 0 } = \tilde { W } _ { i n } X _ { n }$ of the latest token directly into the decoder, which conditions on the cached deep representations through cross-attention to predict the next token

$$
X _ { n + 1 } \sim \mathrm { s o f t m a x } \left( f _ { d e c } ( \cdot \ : | \ : H _ { n } ^ { 0 } , K V _ { < n } ^ { d e c } ) \right) .\tag{5}
$$

Between verifications, drafting requires only the lightweight decoder. The encoder is invoked again only when the next batch of drafts is verified, at which point it produces both verification and the refreshed cache.

Note that we use the term cross-attention as shorthand for attention whose queries come from raw token embeddings and whose keys and values come from the encoder-produced cache. This is imple mented as standard self-attention over the KV cache and does not denote a separate encoder–decoder cross-attention module. We adopt the term "cross-attention" only to emphasize that in decoder, the cache carries full-depth, encoder-processed representations while the incoming query carries only a shallow raw embedding, playing a role analogous to conventional encoder–decoder cross-attention where decoder queries attend to encoder-derived keys and values.

## 4.2 Training Algorithm

We now describe how to train the decoder to draft accurately from raw embeddings while crossattending to cached deep representations. We focus on the typical scenario in which SEED is integrated into a standard supervised fine-tuning pipeline: starting from a pre-trained base LLM, we fine-tune it on task data to improve task performance while simultaneously enabling self-speculative decoding. The same procedure also applies when starting from an already fine-tuned model and continuing training purely to enable inference acceleration, which we explore in Appendix E.2.

Standard Autoregressive Objective. To preserve the model’s generation quality, we retain the standard next-token prediction objective, re-expressed under our encoder-decoder decomposition:

$$
\mathcal { L } _ { C E } = \mathbb { E } _ { X \sim \mathcal { D } } \left[ \sum _ { j = 1 } ^ { \lvert X \rvert } - \log f _ { d e c } \left( X _ { j } \ \lvert \ H _ { j - 1 } ^ { l ^ { \prime } } , K V _ { < ( j - 1 ) } ^ { d e c } \right) \right] .\tag{6}
$$

This loss exactly recovers standard autoregressive training, ensuring that the full encoder-decoder (the verifier) behaves identically to a standard autoregressive model.

Speculative Drafting Objective. At inference, the drafter must predict the next token conditioned on (i) deep contextual representations $K V _ { < n ^ { \prime } } ^ { d e c }$ produced by the verifier for tokens up to the last verified position $n ^ { \prime } { \mathrm { . } }$ , and (ii) only the raw token embeddings ${ \cal H } _ { n ^ { \prime } \underline { { { < } } } \cdot \underline { { { < } } } ( j - 1 ) } ^ { 0 } = { \cal W } _ { i n } X _ { n ^ { \prime } \underline { { { < } } } \cdot \underline { { { < } } } ( j - 1 ) }$ for the last verified context token and any subsequent draft tokens, since invoking the encoder on draft tokens would defeat the purpose of cheap drafting. To train the decoder to predict the next token given this input, we introduce a speculative drafting objective:

$$
\mathcal { L } _ { s p e c } = \mathbb { E } _ { X \sim \mathcal { D } } \left[ \sum _ { j = 1 } ^ { | X | } - \log f _ { d e c } \left( X _ { j } \mid H _ { n ^ { \prime } \leq \cdot \leq ( j - 1 ) } ^ { 0 } , K V _ { < n ^ { \prime } } ^ { d e c } \right) \right] .\tag{7}
$$

Training with Block Attention. Optimizing $\mathcal { L } _ { s p e c }$ requires choosing a boundary $n ^ { \prime }$ for each target token $X _ { j }$ that splits the conditioning context into cached deep representations $K V _ { < n ^ { \prime } } ^ { d e c }$ and raw token embeddings $H _ { n ^ { \prime } \leq \cdot \leq ( j - 1 ) } ^ { 0 }$ . This boundary directly mirrors the inference-time situation we want the decoder to handle: between two verification steps, the decoder must draft several new tokens on its own using only their raw embeddings, while cross-attending to the deep KV cache produced during the previous verification, and only after the next verification do these draft positions get committed to deep contextual representations. To expose the decoder to exactly this conditioning pattern during training, we partition each training sequence into contiguous, non-overlapping blocks of b tokens, where each block plays the role of a sequence of drafts that the decoder must produce alone between two verifications. For every target token $X _ { j }$ , we set $n ^ { \prime }$ to the index of the first token in the block containing $X _ { j - 1 } ( \mathrm { i . e . , } n ^ { \prime } = ^ { \circ } \overline { { \lfloor ( j - 2 ) / b \rfloor } } \times b + 1 )$ . This means that within each block, the decoder sees only raw token embeddings for the current block’s tokens (via causal self-attention) and cross-attends to the deep KV cache $K { \bf \breve { V } } _ { < n ^ { \prime } } ^ { d e c }$ produced by all preceding blocks. Because $n ^ { \prime }$ is the same for every token within a block, all blocks can be processed in parallel.

Concretely, training proceeds in two stages within each gradient step. First, the full encoder–decoder processes the entire sequence in a single forward pass, computing both the standard autoregressive loss $\mathcal { L } _ { C E }$ and caching the decoder’s KV representations $\overset { \bullet } { K } \overset { \ v { \swarrow } d e c } { \Vdash }$ at every position. Second, the decoder alone processes all blocks in parallel using the block-attention mask shown in Figure 2: within each block, tokens attend causally to other tokens in the same block via self-attention over raw embeddings $H ^ { 0 }$ , and additionally cross-attend to $K V ^ { d e c }$ entries from all preceding blocks (but not the current block). We adapt this block-wise training strategy from [3], but use block-causal masks throughout rather than the block-bidirectional masks: response tokens in the encoder-decoder attend causally, preserving the verifier as a valid autoregressive model, and tokens within each decoder block attend causally so the drafter can generate tokens autoregressively. This preserves the autoregressive structure on both sides of the encoder-decoder split, which their block-diffusion formulation does not require.

![](images/6cae4e800569d9992c004a97c2ed76c661a0095ff3dfd7f207cec30f07ab8d62.jpg)  
Figure 2: Example training-time attention masks for $n = 6$ tokens with block size $b = 2 .$ . Left: In the joint encoder-decoder verifier, context tokens use bidirectional attention for richer encoding, while response tokens use causal attention for autoregressive drafting. Right: In the decoder, input token embeddings $( H _ { 1 : n } ^ { 0 } )$ attend causally to tokens within the same block via self-attention, and additionally cross-attend to the decoder’s own KV cache $( K V _ { 1 : n } ^ { d e c } )$ of all preceding blocks.

Final Training Objective. Our final training objective combines the standard autoregressive loss with the speculative drafting loss through a weighted average:

$$
\mathcal { L } = \frac { \mathcal { L } _ { C E } + \lambda \mathcal { L } _ { s p e c } } { 1 + \lambda } ,\tag{8}
$$

where λ controls the relative weight of the speculative drafting objective. In our experiments, we set $\lambda = 1 . 0$ . Section 5.4 studies the impact of varying λ, and the complete training procedure is summarized in Algorithm 1.

## 4.3 Inference Algorithm

Fixed-Length Drafting. At inference time, the trained decoder $f _ { d e c }$ acts as a lightweight draft model, while the full encoder–decoder $f _ { d e c } \circ f _ { e n c }$ serves as the verifier. Decoding alternates between drafting and verification: starting from a verified prefix, $f _ { d e c }$ autoregressively generates d draft tokens by cross-attending to the deep contextual representations $( K V ^ { d e c } )$ cached from the previous verification step, after which the full encoder-decoder verifies these tokens in a single forward pass. We summarize the complete inference procedure in Algorithm 2.

Adaptive Drafting with Tree-Based Verification. Fixed-length drafting is sufficient to realize the core benefit of SEED, but it leaves additional speedup on the table because the decoder’s confidence varies across generation steps: drafting too few tokens under-utilizes a confident decoder, while drafting too many wastes computation when the decoder is uncertain. To better exploit cheap drafting, we adopt two complementary techniques from prior work. First, following [24, 35, 38], we replace the fixed draft length with a confidence-based stopping rule, allowing the decoder to draft more tokens when confident and fewer when uncertain. Second, following [35] and [30], we expand the draft into multiple sibling candidates via a top-p rule rather than committing to a single token. Across drafting steps, this yields a tree of candidate tokens that the full model verifies in one forward pass using a tree-structured attention mask preserving the correct causal dependencies.

Both techniques are particularly well-suited to SEED. Because the decoder shares parameters with the verifier, its confidence reliably reflects acceptance rate, making the stopping rule effective. And because each drafting step is cheap (only a lightweight-decoder forward pass), the overhead of producing extra candidates is small relative to the gain in accepted tokens per verification. We use this adaptive tree-based drafting strategy in the main experiments unless otherwise specified. We defer further illustration and implementation details to Appendix D.1, with ablations in Appendix D.2.

Table 1: Performance comparison on Qwen3-1.7B-Base and Qwen3-4B-Base. 0-shot pass@1 accuracy (Acc.) is reported in %, and decoding throughput (Tput) in tokens/s. For CNN/Daily Mail, we report ROUGE-1 (R-1), ROUGE-2 (R-2), and ROUGE-L (R-L). For methods that keep the target model frozen, we report only their throughput. Given the size of its released drafter weights, PARD is evaluated only on Qwen3-4B-Base. The highest throughput and average speedup are shown in bold, and the second highest results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="2">GSM8K</td><td colspan="2">KodCode</td><td colspan="2">ScienceQA</td><td colspan="3">CNN/Daily Mail</td><td rowspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Tput</td><td>Acc.</td><td>Tput</td><td>Acc.</td><td>Tput R-1</td><td>R-2</td><td>R-L</td><td>Tput</td><td>Speedup</td></tr><tr><td colspan="10">Qwen3-1.7B-Base</td></tr><tr><td>AR</td><td>56.3</td><td>37.9</td><td>66.5</td><td colspan="2">36.0 93.2</td><td>36.6 38.6</td><td>17.7</td><td></td><td>28.6</td><td>34.2</td><td>1.0 ×</td></tr><tr><td colspan="10">Common Acceleration Methods</td></tr><tr><td>E2D2</td><td>26.6↓29.7</td><td>67.6</td><td>40.6↓25.9</td><td colspan="2">65.1</td><td>68.1</td><td>32.8↓5.8</td><td>12.1↓5.6</td><td>23.3↓5.3</td><td>56.8</td><td>1.8 ×</td></tr><tr><td>Apple MTP</td><td></td><td>59.1</td><td></td><td>54.6</td><td>88.7↓4.5</td><td>56.6</td><td></td><td></td><td></td><td>37.5</td><td>1.4 ×</td></tr><tr><td>EAGLE-3</td><td></td><td>79.4</td><td></td><td>70.4</td><td>=</td><td>75.4</td><td></td><td></td><td>1</td><td>60.7</td><td>2.0 ×</td></tr><tr><td colspan="10">Early-Exit Self-Speculative Methods</td><td></td></tr><tr><td>LayerSkip</td><td>42.2↓14.1</td><td>47.0</td><td>57.5↓9.0</td><td>57.0</td><td>91.7↓1.5</td><td>60.0</td><td>37.8↓0.8</td><td>16.9↓0.8</td><td>27.6↓1.0</td><td>42.9</td><td>1.4 ×</td></tr><tr><td>SWIFT</td><td></td><td>43.0</td><td></td><td>41.2</td><td>1</td><td>40.2</td><td></td><td></td><td></td><td>36.2</td><td>1.1 ×</td></tr><tr><td>DEL</td><td></td><td>50.1</td><td></td><td>60.0</td><td></td><td>62.2</td><td>=</td><td>1</td><td>=</td><td>38.6</td><td>1.5 ×</td></tr><tr><td>SEED (Ours)</td><td>57.1↑0.8</td><td>91.8</td><td>68.8↑2.3</td><td colspan="2">105.3</td><td>95.5↑2.3 111.7</td><td>39.2↑0.6</td><td>18.010.3</td><td>29.1↑0.5</td><td>63.2</td><td>2.6×</td></tr><tr><td colspan="10">Qwen3-4B-Base</td><td></td></tr><tr><td>AR</td><td>75.1</td><td>28.3</td><td>76.2</td><td>27.6</td><td>95.4</td><td>26.0</td><td>39.0</td><td>18.2</td><td>29.1</td><td>24.0</td><td>1.0 ×</td></tr><tr><td colspan="10">Common Acceleration Methods</td></tr><tr><td>E2D2</td><td>35.5↓39.6</td><td>55.8</td><td colspan="10">48.0↓28.2 53.5</td></tr><tr><td>Apple MTP</td><td></td><td>45.1</td><td></td><td>49.7</td><td>93.5↓1.9</td><td>54.6 41.3</td><td>32.7↓6.3</td><td>11.5↓6.7</td><td>23.5↓5.6</td><td>40.3 29.4</td><td>1.9 × 1.6 ×</td></tr><tr><td>EAGLE-3</td><td></td><td>64.4</td><td></td><td>54.1</td><td></td><td>56.0</td><td></td><td></td><td></td><td>49.9</td><td>2.1 ×</td></tr><tr><td>PARD</td><td></td><td>57.7</td><td></td><td>65.4</td><td></td><td>62.7</td><td></td><td></td><td></td><td>41.0</td><td>2.1 ×</td></tr><tr><td colspan="10">Early-Exit Self-Speculative Methods</td></tr><tr><td>LayerSkip</td><td>62.2↓12.9</td><td>34.2</td><td>71.1↓5.1</td><td>43.7</td><td>92.4↓3.0</td><td>39.8</td><td>38.7↓0.3</td><td>17.6↓0.6</td><td>28.5↓0.6</td><td>28.7</td><td>1.4 ×</td></tr><tr><td>SWIFT</td><td></td><td>31.7</td><td></td><td>31.3</td><td></td><td>25.8</td><td></td><td></td><td></td><td>26.1</td><td>1.1 ×</td></tr><tr><td>DEL</td><td></td><td>34.8</td><td></td><td>41.0</td><td></td><td>39.4</td><td></td><td></td><td></td><td>22.1</td><td>1.3 ×</td></tr><tr><td>SEED (Ours)</td><td>76.0↑0.9</td><td>74.9</td><td>81.0↑4.8</td><td>80.5</td><td>96.8↑1.4</td><td>89.3</td><td>39.8↑0.8</td><td>18.4↑0.2</td><td>29.3↑0.2</td><td>43.3</td><td>2.7×</td></tr></table>

## 5 Experiments

## 5.1 Experimental Setup

Datasets & Metrics. We demonstrate the effectiveness of SEED by training task-specific models across a diverse set of benchmarks, covering mathematical reasoning, code generation, general knowledge question answering, and summarization: GSM8K [10], KodCode [36], ScienceQA [27], and CNN/Daily Mail [29]. For efficiency, we report the decoding throughput achieved by each method on every dataset with a single 48 GB NVIDIA A6000 GPU. For task quality, we report 0-shot pass@1 accuracy on GSM8K, KodCode, and ScienceQA, and ROUGE scores for CNN/Daily Mail. More details regarding dataset construction can be found in Appendix C.2.

Baselines. We compare our method against the standard autoregressive (AR) decoding baseline with SFT. We further include three groups of accelerated decoding approaches: (1) Early-exit self-speculative decoding methods, including LayerSkip [12], SWIFT [35], and DEL [38]; (2) Speculative decoding with a separate draft model, where we adopt the state-of-the-art EAGLE-3 [22] and PARD [1] (4B target model only); (3) Diffusion language models and multi-token prediction (MTP) methods, including E2D2 [3], which adopts an encoder–decoder division similar to ours to freeze context understanding but applies it within a blocked diffusion framework (Appendix B.3 further clarifies our novelty over E2D2), and Apple MTP [33], a strong MTP baseline that mitigates the loss of local dependency modeling in standard MTP. All methods use greedy decoding for generation.

Models. For methods that require training (AR, E2D2, LayerSkip, Apple MTP, and our method), we fine-tune the pre-trained Qwen3-1.7B-Base and Qwen3-4B-Base models [37] using the training split of each task dataset and evaluate on the corresponding test split. SEED uses only the last 2 transformer layers of the model as the decoder and is trained with a block size of 10, i.e., each sequence is partitioned into 10-token blocks when computing the speculative drafting objective. We perform ablations on these choices in Section 5.4 and provide setup details for other methods in Appendix C.3.

## 5.2 Results

We present the main evaluation results in Table 1. Overall, SEED delivers substantial inference acceleration. Across all benchmarks, SEED achieves an average decoding speedup of 2.6× on Qwen3-1.7B-Base and 2.7× on Qwen3-4B-Base, outperforming all self-speculative decoding baselines. It also matches or surpasses the state-of-the-art EAGLE-3, despite EAGLE-3 relying on a separately trained drafter network. Beyond faster decoding, SEED improves task performance over standard autoregressive (AR) fine-tuning, whereas training-based acceleration methods such as E2D2 and LayerSkip suffer severe quality degradation.

## 5.3 Analysis

Here we explain why SEED outperforms early-exit and MTP-based self-speculative decoding methods as well as speculative decoding methods with a separate drafter like EAGLE-3. The advantages stem from both encoder and decoder: the encoder produces deeper, more future-predictive contextual representations, while the decoder, which shares parameters with the verifier, drafts in a way that remains closely aligned with the verifier’s generation distribution. All experiments in this analysis, as well as the following ablation studies, are conducted by evaluating Qwen3-1.7B-Base models fine-tuned on GSM8K with the same setup as in our main experiments.

Better Drafting from Deeper Representations and Parameter Sharing with the Verifier. We first show that conditioning on deep representations enables SEED to outperform early-exit methods. As shown in Figure 3, SEED consistently achieves higher acceptance rates than LayerSkip despite using a much smaller 2-layer drafter. Early-exit methods like LayerSkip terminate computation at intermediate layers and thus discard the deeper representations from later layers that are crucial for accurate token prediction. SEED avoids this limitation by drafting from the very last layers of the model, giving the drafter access to the verifier’s full-depth contextual features. Moreover, SEED outperforms EAGLE-3 on draft acceptance even though EAGLE-3 also produces drafts conditioned on deep representations. We attribute this gain to parameter sharing: SEED’s drafter is the verifier’s own last two layers, so each draft token is sampled from a distribution closely matching the verifier’s. In contrast, EAGLE-3 relies on a separate draft model, whose output distribution can still deviate from the verifier’s despite being trained to approximate it.

![](images/f24d4463b9381abbcd29089875b091a728e4fac0f1fc80c57fbd4ccf73eac178.jpg)  
Figure 3: Draft token acceptance rates across drafting steps. For a fair comparison, all methods are evaluated under a fixed draft length of 6, with only one candidate draft token kept at each step. Compared to LayerSkip and EAGLE-3, SEED produces higher-quality drafts, both from deeper representations and from parameter sharing with the verifier. Compared to MTP, SEED achieves comparable draft quality at much lower cost.

![](images/e03adb0538dd96e73e3a296b18ba55a5c9ce7c047f07a524dc51a47c621f49d2.jpg)  
Figure 4: Prediction accuracy of trained linear probes in relation to lookahead step for two runs with different seeds. Compared to standard AR training, SEED forces the encoder to produce representations that are more predictive of distant future tokens.

![](images/d3e9887972101fa4882d6725fd52b0ea9bb2320b7a180e8224ff0cf1fd596e01.jpg)

![](images/5d23cd9e6adc42c3a9ae774e54448f804903f428769ce4f42de0ee1c4c6d64ee.jpg)

Table 2: Ablation on the choice of λ. $l _ { \mathrm { d r a f t } }$ and $r _ { \mathrm { d r a f t } }$ denote the average draft token acceptance length and acceptance rate respectively.
<table><tr><td>λ</td><td>Acc.</td><td>Tput</td><td> $l _ { \mathrm { d r a f t } } \left( r _ { \mathrm { d r a f t } } \right)$ </td></tr><tr><td>0.1</td><td>58.5</td><td>81.5</td><td>3.0 (87.4%)</td></tr><tr><td>0.5</td><td>59.1</td><td>87.6</td><td>3.7 (88.3%)</td></tr><tr><td>1.0</td><td>57.1</td><td>91.8</td><td>4.0 (88.6%)</td></tr><tr><td>1.5</td><td>55.5</td><td>92.1</td><td>4.0 (88.0%)</td></tr><tr><td>2.0</td><td>55.1</td><td>91.4</td><td>4.0 (88.4%)</td></tr></table>

Figure 5: Cross-layer linear CKA matrices on GSM8K. Left: AR vs SEED. Right: AR vs LayerSkip.

Cheaper Drafting with A Lightweight Decoder. Figure 3 shows that SEED and Apple MTP achieve comparable draft acceptance, since both condition drafting on deep contextual representations. The throughput gap therefore comes from drafting cost: MTP requires a full forward pass at every drafting step to produce the final hidden states, while SEED reuses the deep contextual representations cached during verification, requiring only a 2-layer decoder pass per draft token. Appendix F quantifies the dominant computation and gives an approximate latency model for fixed-length drafting.

Improved Task Performance from Planning Ahead. We next explain the task performance gains of SEED over standard AR fine-tuning. We hypothesize that our speculative drafting objective implicitly trains the encoder to produce more future-predictive contextual representations: because the decoder must accurately predict tokens many steps ahead from these representations, the encoder is pushed to encode information about future tokens. This is similar to the effect observed in MTP methods [14, 23]. Our linear-probe experiment in Figure 4 directly tests this hypothesis: SEED’s encoder produces hidden representations that remain more predictive of distant future tokens than those of a standard AR model, consistent with a "planning-ahead" effect induced by the speculative drafting objective. Additional details on the experimental design are provided in Appendix E.1.

What SEED’s Learned Representations Look Like. Finally, we examine the layer-wise structure of the representations learned by SEED using Centered Kernel Alignment (CKA) [17]. With the hidden states collected from the linear-probe experiment, we compute cross-layer CKA matrices between the AR fine-tuned model and both the SEED model and the LayerSkip model. Figure 5 shows that SEED’s representations remain very similar to the AR model’s across the early and middle layers, indicating that the training dynamics of the encoder are largely unchanged relative to standard AR fine-tuning. The strongest divergence appears near the encoder-decoder interface (layer l<sup>′</sup> = 26), where SEED reshapes representations most substantially, likely by encouraging the encoder to output more future-predictive hidden states and by adapting the decoder to operate on a novel mixture of raw token embeddings and cached deep representations.

Together, these patterns suggest that SEED’s training acts as a localized modification of the standard AR fine-tuning, targeting the layers responsible for drafting while preserving the deep contextual processing already performed by the bulk of the model. By contrast, LayerSkip alters the model’s internal representations more substantially and shows broader representational drift across layers, explaining its degradation in downstream task performance.

## 5.4 Ablation Study

Figure 6 studies the effect of decoder size, where we keep the total number of transformer layers fixed and vary the encoder-decoder split (i.e., a larger decoder corresponds to a smaller encoder, and vice versa). Increasing the number of decoder layers improves draft quality, but it also increases the cost of each drafting step and reduces overall decoding throughput. For a better trade-off, we use only the last 2 layers as the decoder in our main experiments. Figure 7 studies the effect of the training block size used in the speculative drafting objective $\mathcal { L } _ { s p e c } .$ . Increasing the block size generally improves acceptance length and decoding throughput, suggesting that training the decoder to operate over longer draft spans better prepares it for speculative generation at inference time. We use a block size of 10 in our main experiments as it provides the best decoding speedup. Table 2 studies the effect of the loss weight λ in Eq. (8). A larger λ places more weight on $\mathcal { L } _ { s p e c } ,$ generally improving draft quality, while a smaller λ emphasizes the standard next-token prediction loss $\mathcal { L } _ { C E }$ which favors task accuracy. We choose $\lambda = 1 . 0$ in our main experiments as it provides a favorable balance. Additional ablations are deferred to Appendix D.2.

![](images/fcf63690812e7f968373d1a6c197dedd2d40bd8bdfcb5f847d2fc8c73d9204d3.jpg)  
Figure 6: Ablation on decoder size. Models are evaluated with a simple fixed draft length of 4, so that the draft token acceptance rate directly reflects draft quality and isolates how decoder size influences drafting accuracy. Larger decoders improve draft acceptance but reduce throughput due to higher drafting cost.

![](images/9882e58ec980b34c66cfc8934f4f4559203a4f1db21ebc98d6596b04d658c55c.jpg)  
Figure 7: Ablation on training block size. Models are evaluated with dynamic drafting and treebased verification. Larger block sizes improve draft acceptance and decoding throughput.

## 6 Conclusion

SEED mitigates the central tradeoff in self-speculative decoding between draft quality and drafting cost by reframing decoder-only transformers as implicit encoder-decoder models. By reusing only the last layers as a lightweight drafter conditioned on cached deep representations of the verified prefix, SEED achieves cheap drafting without discarding the rich contextual features produced by the preceding encoder. Across diverse benchmarks, SEED consistently outperforms prior self-speculative baselines in decoding speed while preserving, and often improving, task performance relative to standard autoregressive fine-tuning. Our analysis further suggests that these gains arise because SEED reshapes the training dynamics near the encoder-decoder interface, likely by encouraging the model to learn more future-predictive internal representations.

## Acknowledgement

This research is supported in part by NSF IIS-2508145, Amazon Research Award, and Lambda’s Research Grant Program. We thank Ziteng Sun for his thoughtful comments on the manuscript.

## References

[1] Zihao An, Huajun Bai, Ziqiong Liu, Dong Li, and Emad Barsoum. Pard: Accelerating llm inference with low-cost parallel draft model adaptation. In International Conference on Learning Representations, volume 2026, pages 144562–144579, 2026.

[2] Zachary Ankner, Rishab Parthasarathy, Aniruddha Nrusimha, Christopher Rinard, Jonathan Ragan-Kelley, and William Brandon. Hydra: Sequentially-dependent draft heads for medusa decoding. In First Conference on Language Modeling, 2024.

[3] Marianne Arriola, Yair Schiff, Hao Phung, Aaron Gokaslan, and Volodymyr Kuleshov. Encoderdecoder diffusion language models for efficient training and inference. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[4] Nikhil Bhendawade, Irina Belousova, Qichen Fu, Henry Mason, Mohammad Rastegari, and Mahyar Najibi. Speculative streaming: Fast llm inference without auxiliary models. arXiv preprint arXiv:2402.11131, 2024.

[5] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. In Proceedings of the 41st International Conference on Machine Learning, pages 5209–5235, 2024.

[6] Yuxuan Cai, Xiaozhuan Liang, Xinghua Wang, Jin Ma, Haijin Liang, Jinwen Luo, Xinyu Zuo, Lisheng Duan, Yuyang Yin, and Xi Chen. Fastmtp: Accelerating llm inference with enhanced multi-token prediction. arXiv preprint arXiv:2509.18362, 2025.

[7] Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. Accelerating large language model decoding with speculative sampling. arXiv preprint arXiv:2302.01318, 2023.

[8] Jian Chen, Yesheng Liang, and Zhijian Liu. Dflash: Block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036, 2026.

[9] Longze Chen, Renke Shan, Huiming Wang, Lu Wang, Ziqiang Liu, Run Luo, Jiawei Wang, Hamid Alinejad-Rokny, and Min Yang. Clasp: In-context layer skip for self-speculative decoding. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 31608–31618, 2025.

[10] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[11] Felix Draxler, Justus Will, Farrin Marouf Sofian, Theofanis Karaletsos, Sameer Singh, and Stephan Mandt. Parallel token prediction for language models. arXiv preprint arXiv:2512.21323, 2025.

[12] Mostafa Elhoushi, Akshat Shrivastava, Diana Liskovich, Basil Hosmer, Bram Wasti, Liangzhen Lai, Anas Mahmoud, Bilge Acun, Saurabh Agarwal, Ahmed Roman, et al. Layerskip: Enabling early exit inference and self-speculative decoding. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12622–12642, 2024.

[13] Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. Break the sequential dependency of llm inference using lookahead decoding. In Proceedings of the 41st International Conference on Machine Learning, pages 14060–14079, 2024.

[14] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. In Proceedings ofthe 41st International Conference on Machine Learning, pages 15706–15734, 2024.

[15] Lanxiang Hu, Siqi Kou, Yichao Fu, Samyam Rajbhandari, Tajana Rosing, Yuxiong He, Zhijie Deng, and Hao Zhang. Fast and accurate causal parallel decoding using jacobi forcing. arXiv preprint arXiv:2512.14681, 2025.

[16] John Kirchenbauer, Abhimanyu Hans, Brian Bartoldson, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Multi-token prediction via self-distillation. arXiv preprint arXiv:2602.06019, 2026.

[17] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pages 3519–3529. PMlR, 2019.

[18] Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. Cllms: consistency large language models. In Proceedings of the 41st International Conference on Machine Learning, pages 25426–25440, 2024.

[19] Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In International Conference on Machine Learning, pages 19274–19286. PMLR, 2023.

[20] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 7421–7432, 2024.

[21] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: speculative sampling requires rethinking feature uncertainty. In Proceedings of the 41st International Conference on Machine Learning, pages 28935–28948, 2024.

[22] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[23] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[24] Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Duyu Tang, Kai Han, and Yunhe Wang. Kangaroo: Lossless self-speculative decoding for accelerating llms via double early exiting. Advances in Neural Information Processing Systems, 37:11946–11965, 2024.

[25] Jingyu Liu, Xin Dong, Zhifan Ye, Rishabh Mehta, Yonggan Fu, Vartika Singh, Jan Kautz, Ce Zhang, and Pavlo Molchanov. Tidar: Think in diffusion, talk in autoregression. arXiv preprint arXiv:2511.08923, 2025.

[26] Xiaohao Liu, Xiaobo Xia, Weixiang Zhao, Manyi Zhang, Xianzhi Yu, Xiu Su, Shuo Yang, See-Kiong Ng, and Tat-Seng Chua. L-mtp: Leap multi-token prediction beyond adjacent context for large language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[27] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521, 2022.

[28] Somesh Mehra, Javier Alonso Garcia, and Lukas Mauch. On multi-token prediction for efficient llm inference. arXiv preprint arXiv:2502.09419, 2025.

[29] Ramesh Nallapati, Bowen Zhou, Cicero Dos Santos, Çaglar Gulçehre, and Bing Xiang. Abstrac-˘ tive text summarization using sequence-to-sequence rnns and beyond. In Proceedings ofthe 20th SIGNLL conference on computational natural language learning, pages 280–290, 2016.

[30] Zhiyuan Ning, Jiawei Shao, Ruge Xu, Xinfei Guo, Jun Zhang, Chi Zhang, and Xuelong Li. Casspec: Cascade adaptive self-speculative decoding for on-the-fly lossless inference acceleration of llms. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[31] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. Efficiently scaling transformer inference. Proceedings ofmachine learning and systems, 5:606–624, 2023.

[32] Weizhen Qi, Yu Yan, Yeyun Gong, Dayiheng Liu, Nan Duan, Jiusheng Chen, Ruofei Zhang, and Ming Zhou. Prophetnet: Predicting future n-gram for sequence-to-sequence pre-training. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2020, pages 2401–2410, 2020.

[33] Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

[34] Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. Blockwise parallel decoding for deep autoregressive models. Advances in Neural Information Processing Systems, 31, 2018.

[35] Heming Xia, Yongqi Li, Jun Zhang, Cunxiao Du, and Wenjie Li. Swift: On-the-fly selfspeculative decoding for llm inference acceleration. In The Thirteenth International Conference on Learning Representations, 2025.

[36] Zhangchen Xu, Yang Liu, Yueqin Yin, Mingyuan Zhou, and Radha Poovendran. Kodcode: A diverse, challenging, and verifiable synthetic dataset for coding. In Findings of the Association for Computational Linguistics: ACL 2025, pages 6980–7008, 2025.

[37] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[38] Hossein Entezari Zarch, Lei Gao, Chaoyi Jiang, and Murali Annavaram. Del: Context-aware dynamic exit layer for efficient self-speculative decoding. arXiv preprint arXiv:2504.05598, 2025.

[39] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11263–11282, 2024.

[40] Guoliang Zhao, Ruobing Xie, An Wang, Shuaipeng Li, Huaibing Xie, and Xingwu Sun. Selfdistillation for multi-token prediction. arXiv preprint arXiv:2603.23911, 2026.

[41] Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, et al. Pytorch fsdp: experiences on scaling fully sharded data parallel. arXiv preprint arXiv:2304.11277, 2023.

## A Limitations and Future Directions

While SEED achieves strong speedups and preserves task quality across the settings studied in this work, several limitations remain.

Scaling to Larger Models. Our experiments focus on Qwen3-1.7B-Base and Qwen3-4B-Base due to computational constraints. Although these models cover multiple tasks and already demonstrate consistent gains, we have not yet validated SEED on larger models like 8B, 14B, or 32B-scale LLMs. The efficiency-quality tradeoff may change with model size: larger models have deeper stacks and potentially richer intermediate representations, which could make the encoder-decoder partition more effective. Evaluating SEED at larger scales is therefore an important next step.

Beyond Task-Specific Fine-Tuning. In this paper, SEED is mainly studied in task-specific supervised fine-tuning settings. An important direction for future work is extending SEED beyond task-specific fine-tuning to larger-scale post-training or even pretraining settings. Applying the speculative drafting objective during general instruction tuning, preference optimization, continual post-training, or pretraining could produce models that are natively compatible with SEED across a wider range of prompts and domains. Such a setting would also allow us to study whether the "planning-ahead" behavior induced by SEED improves general model capabilities and robustness beyond the task-specific benchmarks considered here.

Adaptive Partitioning and Hardware-Aware Implementation. Our main experiments use a simple fixed partition, where the top two transformer layers serve as the decoder. This design is intentionally simple, but it is unlikely to be optimal for all model sizes, tasks, sequence lengths, or deployment environments. Future work could explore adaptive partitioning strategies that choose the encoder-decoder split based on model depth, layerwise representational quality, draft acceptance statistics, or runtime constraints. In addition, practical speedup depends heavily on implementation details such as KV-cache layout, tree-attention kernels, batching behavior, and GPU memory bandwidth. Hardware-aware implementations, including custom kernels for decoder-only drafting and tree-based verification, could further improve the realized speedup of SEED in production serving systems.

## B Additional Discussions on Related Work

A high-level comparison of representative LLM inference acceleration methods, including our approach, is summarized in Table 3. Unlike many prior methods, SEED does not bring additional system complexity by introducing external modules, while achieving considerable inference speedups. More importantly, because SEED preserves the standard next-token prediction objective and achieves generation quality on par with autoregression, it can be seamlessly integrated into existing supervised fine-tuning pipelines. This allows training data to simultaneously improve both inference efficiency and generation quality, an advantage not shared by most existing LLM acceleration methods, which typically leverage training data solely for inference speedup, often at the expense of generation quality.

## B.1 Connection to Self-Speculative Decoding with Layer Skipping

A closely related approach to SEED is self-speculative decoding with layer skipping, which accelerates drafting by omitting a subset of transformer layers during the draft forward pass [24, 35, 39]. Using the notation from Section 4, this strategy can also be interpreted within our encoder–decoder framework. Specifically, the skipped layers $\mathcal { E } = \{ e _ { 1 } , e _ { 2 } , \ldots , e _ { s } \}$ can be viewed as the encoder, while the remaining layers $\bar { \mathcal { D } } = \{ d _ { 1 } , \dotsc , d _ { l - s } \}$ act as the decoder that generates draft tokens.

However, this formulation reveals an inherent limitation of layer-skipping approaches. Suppose we have some verified context tokens $X _ { 1 } , X _ { 2 } , \ldots , X _ { n }$ and are about to propose a new draft token. At any decoder layer $\ell \in \mathcal { D }$ , the attention mechanism

$$
\mathrm { A t t n } ( H _ { n } ^ { \ell } , K V _ { < n } ^ { \ell } )
$$

operates on representations at the same layer, where verified tokens $( i < n )$ are represented by their cached keys and values $K V _ { < n } ^ { \ell }$ obtained during the previous verification step. In typical layer-skipping schemes, a large portion of the retained layers D include early layers of the transformer. Consequently, the KV caches available to these layers correspond to relatively shallow representations of the verified prefix. The draft model therefore attends only to low-level contextual features when generating candidate tokens, which can significantly degrade draft quality.

Table 3: Comparison of SEED against existing LLM acceleration methods. "Lossless" denotes no degradation in generation quality compared to standard autoregressive models.
<table><tr><td>Method</td><td>No Ext. Module</td><td>Lossless</td><td>Training Cost</td><td>Speedup</td></tr><tr><td>General MTP</td><td></td><td></td><td></td><td></td></tr><tr><td>Meta MTP [14] &amp;</td><td>x</td><td>√</td><td>High</td><td>Mid</td></tr><tr><td>DeepSeek MTP [23]</td><td>x</td><td>√</td><td>Low</td><td>Mid</td></tr><tr><td>FastMTP [6] Apple MTP [33]</td><td>x</td><td>√</td><td>Low</td><td>High</td></tr><tr><td>Jacobi Forcing [15]</td><td>√</td><td>x</td><td>Low</td><td>High</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Diffusion Language Models</td><td></td><td></td><td></td><td></td></tr><tr><td>E2D2 [3] TiDAR [25]</td><td>√</td><td>x</td><td>Low</td><td>Mid</td></tr><tr><td></td><td>√</td><td>x</td><td>High</td><td>High</td></tr><tr><td>General Speculative Decoding</td><td></td><td></td><td></td><td></td></tr><tr><td>MEDUSA [5]</td><td>X</td><td>√</td><td>Low</td><td>High</td></tr><tr><td>EAGLE-3 [22]</td><td>x</td><td>√</td><td>High</td><td>High</td></tr><tr><td>Self-Speculative Decoding</td><td></td><td></td><td></td><td></td></tr><tr><td>LayerSkip [12]</td><td>√</td><td>x</td><td>Low</td><td>Low</td></tr><tr><td>SWIFT [35] &amp;</td><td>√</td><td>√</td><td>None</td><td>Low</td></tr><tr><td>CLaSp [9] CAS-Spec [30]</td><td>√</td><td>√</td><td>None</td><td>Mid</td></tr><tr><td>SEED (Ours)</td><td>√</td><td>√</td><td>Low</td><td>High</td></tr></table>

In contrast, SEED places the drafting component strictly in the last few layers of the transformer $( \ell \in [ l ^ { \prime } + 1 , l ] )$ . Rather than conditioning on a single encoder-produced hidden state, the decoder attends directly to the verified KV caches $K V _ { \leq n } ^ { d e c }$ , which were computed after the verified tokens passed through all l<sup>′</sup> encoder layers. These cached keys and values therefore encode the deepest contextual representations available in the model and effectively serve as the shared encoder state.

As a result, the draft model in SEED always conditions on deep, highly contextualized representations of the verified prefix. This asymmetric design preserves the representational strength of the full model while keeping the draft computation inexpensive, enabling higher draft acceptance rates and improved overall decoding throughput.

## B.2 Comparison to the EAGLE Series

Here we provide a deeper theoretical and architectural comparison between SEED and the EAGLE series of speculative decoding algorithms [21, 20, 22]. Both approaches share a fundamental motivation: draft models can achieve significantly higher acceptance rates if they are conditioned on deep, highly structured contextual representations of the verified prefix, rather than shallow representations or raw tokens alone. However, the methods diverge in how they formulate the drafting objective and system architecture.

The original EAGLE [21] formulates the draft model as performing feature-level autoregression. Given a sequence of hidden representations $H _ { 0 } , H _ { 1 } , \dots , \bar { H _ { n - 1 } }$ , the draft model learns to predict the next hidden representation $H _ { n }$ . Because continuous hidden representations are highly structured, this regression task becomes easier to learn than directly predicting raw discrete tokens. However, conditional dependencies in language are based on discrete identity. To resolve this, EAGLE samples the discrete token $X _ { n + 1 }$ from $H _ { n }$ and then fuses the embedding of $X _ { n + 1 }$ with $H _ { n }$ to predict the next step. Yet one problem remains: during training, EAGLE’s draft model receives ground-truth features from the target model as input at every position, but at inference time, the draft model must condition on its own predicted features for speculative positions. This creates an inherent exposure bias and distribution shift, because if the draft model produces poor feature predictions early on, subsequent inputs will be severely out-of-distribution. This motivates the authors in EAGLE-3 [22] to drop feature regression entirely and instead focus purely on token prediction, handling the exposure bias through a training-time drafting mechanism.

Meanwhile, SEED operates under the simplified hypothesis that hidden representations are valuable not because they are easier to predict, but because they make predicting the next token easier. By formulating drafting as an implicit encoder-decoder process, SEED avoids the feature-level distribution shift entirely. During both training and inference, the decoder processes draft tokens using only their raw token embeddings $( H ^ { 0 } )$ , while cross-attending to the deep contextual representations $( K \bar { V } ^ { d e c } )$ of the verified prefix. Because the decoder is explicitly trained to handle this specific mixture of representation types, it never has to condition on its own continuous feature predictions. Consequently, SEED achieves high draft quality through direct token prediction, without the need for auxiliary feature regression objectives or feature fusion mechanisms. Moreover, because the drafter shares parameters with the verifier, it remains better aligned with the target model’s distribution. In contrast to EAGLE-3, which conditions drafting on multiple levels of intermediate features and requires a complex training-time test simulation to mitigate exposure bias, SEED proves that it is possible to achieve higher draft quality by simply reusing the verifier’s high-level contextual representations through the KV cache.

## B.3 Comparison to E2D2

E2D2 [3] pioneered the amortization principle of separating expensive clean-context encoding from repeated lightweight generation. SEED can be viewed as bringing this principle into autoregressive de coding with exact verifier-based acceptance. Concretely, the new components are: (i) the speculative drafting objective that trains the last layers to draft from raw token embeddings while conditioning on verifier-cached deep KV representations, (ii) the fully block-causal training masks that preserve the verifier as a valid autoregressive model, and (iii) the resulting draft-then-verify inference loop in which verification and encoding merge into a single forward pass. Crucially, this AR speculative formulation changes the nature of the result: instead of merely extending the quality–efficiency frontier, exact verification enables acceleration without sacrificing model performance.

To disentangle whether this advantage comes from the formulation or from implementation / training choices, we run a controlled comparison against E2D2 on GSM8K with Qwen3-1.7B-Base, matching the training budget, training block size b = 4, and encoder-decoder split, and disabling SEED’s adaptive drafting and tree verification with a fixed draft length of 4. This matches the split and generation span, while the inference procedures retain their respective diffusion and speculative computations.

Table 4: Controlled comparison with E2D2. We compare E2D2 and SEED across different decoder sizes by training and evaluating corresponding Qwen3-1.7B-Base models on GSM8K. SEED achieves much better task performance while using a smaller decoder, proving that exact verification enables acceleration without sacrificing model performance.
<table><tr><td>Decoder Size (# layers)</td><td>E2D2 Acc. (%)</td><td>E2D2 Tput (tokens/s)</td><td>SEED Acc. (%)</td><td>SEED Tput (tokens/s)</td></tr><tr><td>2</td><td>23.7</td><td>75.9</td><td>57.5</td><td>74.3</td></tr><tr><td>4</td><td>26.6</td><td>67.6</td><td>57.9</td><td>66.1</td></tr><tr><td>6</td><td>32.3</td><td>63.2</td><td>59.4</td><td>60.8</td></tr><tr><td>8</td><td>32.4</td><td>57.3</td><td>58.8</td><td>56.3</td></tr><tr><td>10</td><td>36.3</td><td>51.2</td><td>57.4</td><td>53.3</td></tr><tr><td>12</td><td>41.3</td><td>46.4</td><td>57.0</td><td>48.8</td></tr><tr><td>14</td><td>43.4</td><td>42.8</td><td>57.6</td><td>46.4</td></tr></table>

The two methods achieve nearly identical throughput at every split, confirming that they share the same amortization mechanism. The difference is in task quality: E2D2 must trade accuracy for speed, whereas SEED’s accuracy is essentially flat across all decoder sizes because the exact verifier guarantees that the output distribution matches autoregressive decoding regardless of drafter capacity. This lets us use the smallest, fastest decoder while securing both speed and accuracy.

## C Further Details on Experimental Setup

## C.1 Hardware

All 1.7B models are fine-tuned on a single 48 GB NVIDIA A6000 GPU, and all 4B models are fine-tuned on either NVIDIA 80 GB H100 GPUs or 180 GB B200 GPUs. Some 4B models are trained on two H100 GPUs using PyTorch Fully Sharded Data Parallel (FSDP) [41], while others are trained on a single B200 GPU. All trained models are evaluated on a single NVIDIA A6000 GPU to measure task performance and throughput.

## C.2 Datasets

Following E2D2 [3], we train and evaluate task-specific models on four datasets covering math reasoning, code generation, general knowledge, and summarization: GSM8K [10], KodCode [36], ScienceQA [27], and CNN/Daily Mail [29].

For GSM8K, we use the Hugging Face dataset openai/gsm8k, with the official train split for training and test split for evaluation. We pre-process the data by adding a prefix "Please reason step by step, and put your final answer within \$\boxed{}\$." to the inputs, prefacing the answers with "Answer: ", and wrapping the solutions in "\$\boxed{}\$". Inputs and targets are truncated to a maximum length of 384 tokens each, ensuring a maximum sequence length of 768 during training. Final evaluation is performed in 0-shot mode using the lm-eval harness library with the "flexible match" criteria, and similar pre-processing is applied to question texts. Post-processing is done to truncate text at <|endoftext|> tokens, and solutions are reverted to their original form of "### <Answer>". Inputs are pre-processed as above.

For KodCode, we use the Hugging Face dataset KodCode/KodCode-V1-SFT-R1 with a deterministic reconstruction of train/test splits. We first keep only samples with online judge style questions, then randomly subsample 50% of these examples, and finally define the test set as the last 1000 samples of this subsample while using the remainder as the training set. Each problem is prompted as a Python programming task ending with a [BEGIN] marker. Inputs and targets are truncated to a maximum length of 512 tokens each, ensuring a maximum sequence length of 1024 during training. During evaluation, we restrict the test set to problems marked with difficulty level "easy". For each problem, the model generates a program which is then executed against the provided test cases, and we report exact problem-solving accuracy based on whether all tests pass.

For ScienceQA, we use the Hugging Face dataset derek-thomas/ScienceQA and restrict it to text-only examples. The training set is constructed by merging the original train and validation splits, while the evaluation set is created by sampling 1000 examples from the original test split. Inputs are formatted as multiple-choice questions with answer options labeled (A) to (H), and targets consist of the reasoning followed by a canonical final statement of the form "The answer is (X)". Inputs and targets are truncated to a maximum length of 384 tokens each, ensuring a maximum sequence length of 768 during training. For evaluation, the predicted option letter is extracted from the generated text, and exact-answer accuracy is reported.

For CNN/Daily Mail, we use the Hugging Face dataset abisee/cnn\_dailymail (version 3.0.0), with the official train split for training and the test split for evaluation. Inputs consist of the news article prefixed with a summarization instruction, while targets are the corresponding summaries prefixed with "Summary: ". Unlike the previous datasets, we allocate the token budget asymmetrically, assigning 90% of the sequence length to the article and 10% to the summary, for a total sequence length of 1024 tokens. During evaluation, we apply the same length constraints and evaluate on up to 1000 samples from the test split. Summarization quality is measured using ROUGE-1, ROUGE-2, and ROUGE-L.

## C.3 Baselines

Our overall implementation builds on the official E2D2 codebase<sup>2</sup> [3]. Accordingly, our AR and E2D2 baselines follow their implementation. For E2D2, we use the last 4 layers as the decoder for a better trade-off between performance and speedup. Consistent with the original setup, E2D2 is trained and evaluated using a block size of 4.

For Apple MTP [33], the original paper does not provide an official code release. We therefore adopt the open-source unofficial implementation from https://github.com/siihwanpark/MTP-GLoRA and make the necessary modifications to better match the algorithmic details described in the paper. We fine-tune the model to handle 8 mask tokens as input by training gated-LoRA weights and a 2-layer MLP sampler module, using a LoRA rank of 16 and a LoRA alpha of 32. For evaluation, we set the draft length to 8 across all datasets, with the exception of CNN/Daily Mail, where a draft length of 4 is used.

For LayerSkip [12], the released repository provides inference code but not training code. We therefore use the implementation in Hugging Face TRL<sup>3</sup> and make the necessary adjustments to align it with the training procedure described in the original paper. The early-exit layer is set to layer 8 for Qwen3-1.7B-Base and layer 12 for Qwen3-4B-Base. We fine-tune the base models using a composite objective $L = L _ { \mathrm { f i n a l } } + \lambda \cdot L _ { \mathrm { e a r l y - e x i t } }$ with λ = 1.0. Training follows a rotational early-exit curriculum with stride 8, and layer dropout is applied with a constant schedule and a maximum dropout rate of 0.1.

For EAGLE-3 [22], we use the publicly available AngelSlim checkpoints AngelSlim/Qwen3- 1.7B\_eagle3<sup>4</sup> and AngelSlim/Qwen3-4B\_eagle3<sup>5</sup> paired with Qwen3-1.7B and Qwen3-4B instruct models due to the lack of corresponding EAGLE-3 weights for the base models. Evaluation is conducted using the official EAGLE GitHub repository<sup>6</sup>. Note that although EAGLE-3 reports more than 5× average speedup on Vicuna-, LLaMA-, and DeepSeek-based models, our experiments on Qwen3-1.7B and Qwen3-4B show only around 2× speedup on a single NVIDIA A6000. This observation is consistent with the results reported by AngelSlim in https://huggingface.co/AngelSlim/Qwen3-1.7B\_eagle3.

For PARD [1], we evaluate the released target-independent Qwen3 drafter<sup>7</sup>, paired with the corresponding task-specific, fine-tuned Qwen3-4B-Base AR checkpoint under the same evaluation setting as SEED. We include PARD only in the 4B portion of Table 1, as its 0.6B-parameter drafter is relatively large for a 1.7B target model.

For SWIFT [35] and DEL [38], we use the authors’ released implementations directly. SWIFT runs on top of our fine-tuned AR checkpoints, with Bayesian optimization interval 1, update interval 25, layer skip ratio 0.4, context window 50, maximum optimization iterations 1000, tolerance limit 300, and early-stop score threshold 0.95. DEL runs on top of our fine-tuned LayerSkip checkpoints by maintaining an exponential moving average on acceptance/confidence statistics with retention factor ω = 0.95.

## C.4 Training

We train all models using the MosaicML Composer framework<sup>8</sup>. Unless otherwise noted, we use the same optimization setup across methods to ensure a fair comparison. Specifically, we apply a linear learning-rate warm-up for the first 100 steps, followed by cosine decay. Model checkpoints are selected based on validation loss with early stopping. We set the random seed to 1 during training.

Training hyperparameters vary by model scale and dataset, primarily in learning rate, batch size, and maximum sequence length. Table 5 summarizes the final training setup.

## C.5 Task Prompts

We now describe the task prompts used to train and evaluate the baseline methods in our experiments. The prompts are designed to be simple and straightforward, as detailed below:

Table 5: Training configurations used for task-specific fine-tuning. For each dataset and model scale, all methods share the same optimizer schedule and training budget to maintain comparability.
<table><tr><td>Dataset</td><td>Base Model</td><td>Batch Size</td><td>LR</td><td>Steps</td><td>Max Length</td></tr><tr><td>GSM8K</td><td>Qwen3-1.7B-Base</td><td>1</td><td> $1 e ^ { - 5 }$ </td><td>30,000</td><td>768</td></tr><tr><td>KodCode</td><td>Qwen3-1.7B-Base</td><td>1</td><td>1e−5</td><td>30,000</td><td>1024</td></tr><tr><td>ScienceQA</td><td>Qwen3-1.7B-Base</td><td>1</td><td>1e−5</td><td>30,000</td><td>768</td></tr><tr><td>CNN/Daily Mail</td><td>Qwen3-1.7B-Base</td><td>32</td><td>1e−5</td><td>30,000</td><td>1024</td></tr><tr><td>GSM8K</td><td>Qwen3-4B-Base</td><td>1</td><td>5e−6</td><td>30,000</td><td>768</td></tr><tr><td>KodCode</td><td>Qwen3-4B-Base</td><td>1</td><td>5e−6</td><td>30,000</td><td>1024</td></tr><tr><td>ScienceQA</td><td>Qwen3-4B-Base</td><td>1</td><td>5e−6</td><td>30,000</td><td>768</td></tr><tr><td>CNN/Daily Mail</td><td>Qwen3-4B-Base</td><td>32</td><td>5e−6</td><td>30,000</td><td>1024</td></tr></table>

<table><tr><td>Task Prompts</td></tr><tr><td>GSM8K</td></tr><tr><td>Please reason step by step, and put your final answer within $\boxed{ }$. {question}</td></tr><tr><td>KodCode</td></tr><tr><td>You are an expert Python programmer. Solve the following problem.</td></tr><tr><td>{question}</td></tr><tr><td>ScienceQA</td></tr><tr><td>The following is a multiple choice question. Think step by step and then give your final answer.</td></tr><tr><td>{question}</td></tr><tr><td>(A) {choice A}</td></tr><tr><td>(B) {choice B}</td></tr><tr><td></td></tr><tr><td>CNN/Daily Mail Summarize the following article: {article}</td></tr></table>

## C.6 Evaluation

We evaluate all trained models using greedy decoding. To measure decoding throughput, we adopt a different protocol from prior work, which typically continues generation until the maximum sequence length is reached. We find that if we force models to continue generating after the <|endoftext|> token, draft models tend to adapt to repetitive output patterns, which can artificially increase drafting accuracy and decoding throughput. However, in real-world applications, we only care about the tokens generated before the <|endoftext|> token. Therefore, in all experiments, we measure decoding throughput by terminating generation as soon as the <|endoftext|> token is generated.

## D SEED Implementation Details

## D.1 Algorithms

We include SEED training and inference with fixed draft length in Algorithm 1 and Algorithm 2. Since our main experiments use adaptive drafting with tree-based verification, we further describe that procedure here and illustrate it in Figure 8.

Confidence-Aware Dynamic Drafting. A fixed draft length is often suboptimal because the decoder’s confidence varies across generation steps. Instead of always drafting exactly d tokens, we let the decoder continue drafting as long as its confidence remains above a predefined threshold.

```latex
Algorithm 1 SEED Training
Require: Transformer $f = f _ { d e c } \circ f _ { e n c } ,$ encoder-decoder divider l<sup>′</sup>, training data $D ,$ block-size b,
speculative drafting objective weight λ
for $X \in D$ do
2: $H _ { . . } ^ { 0 }  W _ { i n } X$
3: $H ^ { l ^ { \prime } } \gets f _ { e n c } ( X )$
4: ▷ Standard next-token prediction loss Computed in a single forward pass of f
5: $\begin{array} { r } { \mathcal { L } _ { C E }  \sum _ { j = 2 } ^ { | X | } - \log \bar { f } _ { d e c } ( X _ { j } \mid H _ { j - 1 } ^ { l ^ { \prime } } , K V _ { < ( j - 1 ) } ^ { d e c } ) } \end{array}$
6: ▷ In practice, this for loop can be computed in parallel for all blocks using the block causal
attention mask illustrated in Figure 2
7: $L _ { s p e c }  0$
8: for $j = 2 \mathop { \mathrm { t o } } | X |$ do
9: ▷ Compute the start index of the block to which the last input token $X _ { j - 1 }$ belongs
10: $n ^ { \prime }  \lfloor { \frac { j - 2 } { b } } \rfloor \times b + 1$
11: ▷ Speculative drafting objective
12: $\mathcal { L } _ { s p e c }  \mathcal { L } _ { s p e c } - \log f _ { d e c } ( X _ { j } \mid H _ { n ^ { \prime } \leq \cdot \leq ( j - 1 ) } ^ { 0 } , K V _ { < n ^ { \prime } } ^ { d e c } )$
13: end for
14: $L = \frac { \bar { \mathcal { L } _ { C E } } + \lambda \mathcal { L } } { 1 + \lambda }$ spec
15: ∇θ ← Backprop(L)
16: f ← OptimizerStep(f, ∇θ)
17: end for
18: return f
```

Algorithm 2 SEED Inference with Fixed Draft Length   
Require: Transformer with encoder $f _ { e n c } = f _ { 1 : l ^ { \prime } }$ , decoder $f _ { d e c } = f _ { l ^ { \prime } + 1 : l } ,$ prompt $X _ { 1 : t }$ , draft length   
d, target length T   
1: Initialize $K \Breve { V } _ { 1 : t }$ by calling $( f _ { d e c } \circ f _ { e n c } ) ( X _ { 1 : t } )$   
2: while $t < T$ do   
3: ▷ Drafting Phase: Autoregressively draft d tokens using only the decoder   
4: $X _ { d r a f t } [ 0 ]  X _ { t }$   
5: for $i = 1$ to d do   
6: $q _ { i } ( x ) \gets f _ { d e c } ( W _ { i n } X _ { d r a f t } [ i - 1 ] , K V _ { 1 : ( t + i - 2 ) } ^ { d e c } )$   
7: $X _ { d r a f t } [ i ]  \arg \operatorname* { m a x } q _ { i } ( x )$   
8: end for   
9: $K V _ { 1 : ( t - 1 ) } \gets \operatorname { T r u n c a t e } ( K V _ { 1 : ( t + d - 1 ) } , t - 1 )$   
10: ▷ Verification Phase: Forward the full model in parallel   
11: $p _ { 1 } ( x ) , \ldots , p _ { d + 1 } ( x ) \gets ( f _ { d e c } \circ f _ { e n c } ) ( X _ { d r a f t } [ 0 : d ] , K V _ { 1 : ( t - 1 ) } )$   
12: $n _ { a c c e p t }  0$   
13: for $i = 1$ to d do   
14: if arg max<sub>x</sub> $p _ { i } ( x ) = X _ { d r a f t } [ i ]$ then   
15: n<sub>accept</sub> ← n<sub>accept</sub> + 1   
16: else   
17: break   
18: end if   
19: end for   
20: $\mathsf { D A p p }$ end verified tokens and a bonus correction token   
21: x<sub>new</sub> $ \operatorname { a r g m a x } _ { x } p _ { n _ { a c c e p t } + 1 } ( x )$   
22: $X _ { ( t + 1 ) : ( t + n _ { a c c e p t } + 1 ) } \gets [ \dot { X } _ { d r a f t } [ 1 : n _ { a c c e p t } ] , x _ { n e w } ]$   
23: $K V _ { 1 : ( t + n _ { a c c e p t } ) }  \mathrm { T r u n c a t e } ( K V _ { 1 : ( t + d ) } , t + n _ { a c c e p t } )$   
24: $t  t + n _ { a c c e p t } + 1$   
25: end while   
26: return $X _ { 1 : T }$

![](images/ce2dbb7be8bf8b4afa0cb85067f54a7cf6a3391d704219e8292c2f93d2abb72e.jpg)  
Figure 8: Illustration of confidence-aware dynamic drafting and tree-based verification in SEED. Left: starting from a verified prefix, the lightweight decoder drafts autoregressively, with the current node expanded into multiple candidates using a top-p rule, thus producing a small tree of candidate continuations. Right: all candidate tokens are packed and verified in one full-model forward pass using a tree-structured attention mask. Each candidate token attends to the entire verified prefix and to the candidate tokens on its own ancestral path, but not to tokens from unrelated branches. This allows parallel verification of multiple draft branches while preserving the causal dependencies of autoregressive decoding.

Starting from the current verified prefix, the decoder autoregressively proposes the next token while conditioning on the verified decoder KV cache and the raw embeddings of previously drafted tokens, exactly as in the fixed-length setting. Also, at each drafting step, we expand the current node into multiple sibling candidates using a top-p rule, following prior adaptive self-speculative decoding methods. Repeating this procedure over several steps produces a small tree of candidate continuations rather than a single draft sequence. This confidence-aware strategy allocates computation more effectively. When the decoder is confident, generation proceeds with a narrow branch and low drafting overhead. When uncertainty increases, the procedure broadens the candidate set so that the verifier can evaluate several plausible continuations in parallel. In practice, we set $\tau = 0 . 7$ for GSM8K, KodCode, and $\scriptstyle \mathrm { S c i e n c e Q A }$ , and $\tau = 0 . 5$ for CNN/Daily Mail. For candidate expansion under the top-p rule, let $p _ { 0 }$ denote the probability of the current top-1 draft token. We then choose the number of candidate draft tokens k at each step according to

$$
k = \left\{ \begin{array} { l l } { 1 } & { \mathrm { i f ~ } p _ { 0 } > 0 . 9 5 , } \\ { 3 } & { \mathrm { i f ~ } 0 . 8 < p _ { 0 } \leq 0 . 9 5 , } \\ { 5 } & { \mathrm { i f ~ } 0 . 5 < p _ { 0 } \leq 0 . 8 , } \\ { 1 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Tree-Based Verification. Once the candidate tree is constructed, we verify all drafted tokens with a single full-model forward pass. To do so, we linearize the tree nodes into a packed sequence and apply a tree-structured attention mask that preserves the correct autoregressive dependencies. Concretely, every candidate token is allowed to attend to: (i) all verified prefix tokens, and (ii) the candidate tokens on its own ancestral path in the draft tree. Tokens from different branches are not allowed to attend to one another unless they share the same ancestors. This reproduces exactly the causal structure each branch would have under independent autoregressive evaluation, while enabling all candidates to be processed in parallel in one verifier call. After the verifier produces logits for all candidate nodes, we traverse the tree from the root and greedily accept the longest prefix whose verified predictions match the drafted tokens. As in standard speculative decoding, after the accepted draft prefix is determined, we additionally append one bonus token predicted by the verifier at the first non-accepted position. The KV cache is then truncated to the accepted path, the verified prefix is updated, and the next round of drafting begins.

## D.2 Further Ablations

Effect of the Confidence Threshold τ. In our main experiments, SEED employs a dynamic drafting strategy that terminates drafting when the top-1 probability of the draft token falls below a confidence threshold τ. To study the impact of τ on decoding efficiency, we vary the threshold and measure the resulting decoding throughput, draft token acceptance rate, and average acceptance length. Specifically, we evaluate a GSM8K-finetuned Qwen3-1.7B-Base SEED model on GSM8K, and the results are shown in Figure 9. As τ increases, the acceptance rate of draft tokens consistently improves, indicating that the internal confidence of the draft model correlates positively with draft quality. However, a higher τ also shortens the average acceptance length because drafting terminates earlier. Despite this trade-off, decoding throughput remains relatively stable within a wide range of thresholds $( \tau \in [ 0 . 5 , 0 . 9 ] )$ , demonstrating that SEED is robust to the choice of τ. In our experiments, we select $\tau = 0 . 7$ as it provides a favorable balance between acceptance rate and drafting length. For more challenging generation tasks such as CNN/Daily Mail summarization, where draft token acceptance lengths are typically shorter, we adopt a slightly lower threshold $( \tau = 0 . 5 )$ to maintain longer drafting segments and better efficiency.

![](images/6ee95dafcdf1f321300505c4c1485970c783f24647ab8debe169cb9fcd8365f0.jpg)

![](images/23065a1a4f128b09b2b0d9b46d9708d36311e6801a1c21eb5f559ea4983a3172.jpg)  
Figure 9: Effect of the confidence threshold τ used in dynamic drafting. We evaluate a GSM8Kfinetuned Qwen3-1.7B-Base SEED model on GSM8K and report average decoding throughput, draft token acceptance rate, and draft token acceptance length. Increasing τ improves acceptance rate but shortens acceptance length due to earlier termination of drafting. Overall decoding throughput remains stable for τ between 0.5 and 0.9, indicating that SEED is relatively insensitive to the precise choice of τ.

Table 6: Ablation study on tree-based verification. We evaluate a GSM8K-finetuned Qwen3-1.7B-Base SEED model on GSM8K. We report decoding throughput (Tput) in tokens/s, average draft token acceptance length (Acc. Len), and draft token acceptance rate (Acc. Rate) under fixed-length and dynamic drafting strategies. Tree-based verification consistently improves acceptance statistics and overall throughput.
<table><tr><td rowspan="2">Method</td><td colspan="3">Fixed-Length Drafting (d = 4)</td><td colspan="3">Fixed-Length Drafting (d = 8)</td><td colspan="3">Dynamic Drafting  $( \tau = 0 . 7 )$ </td></tr><tr><td>Tput</td><td>Acc. Len</td><td>Acc. Rate</td><td>Tput</td><td>Acc. Len</td><td>Acc. Rate</td><td>Tput</td><td>Acc. Len</td><td>Acc. Rate</td></tr><tr><td>w/ Tree Attn</td><td>82.3</td><td>3.16</td><td>79.80%</td><td>82.5</td><td>4.65</td><td>59.10%</td><td>91.8</td><td>3.97</td><td>88.57%</td></tr><tr><td>w/o Tree Attn</td><td> $7 6 . 9 ^ { \downarrow 5 . 4 }$ </td><td>2.67</td><td>69.45%</td><td> $7 5 . 2 ^ { \downarrow 7 . 3 }$ </td><td>3.67</td><td>46.64%</td><td>85.1↓6.7</td><td>3.28</td><td>77.59%</td></tr></table>

Effect of Tree-Based Verification. In our main experiments, we also keep multiple candidate draft tokens at each drafting step, forming a draft tree that is subsequently verified in parallel by the full model using a tree-structured attention mask. To evaluate the contribution of this design to decoding efficiency, we ablate the tree-based verification mechanism by disabling tree attention and restricting drafting to a single candidate per step. We conduct this study on a GSM8K-finetuned

Qwen3-1.7B-Base SEED model and report results on GSM8K. As shown in Table 6, enabling treebased verification consistently improves throughput, acceptance length, and acceptance rate across both fixed and dynamic drafting settings. These results demonstrate that expanding the candidate set and verifying branches in parallel substantially improves draft token acceptance, leading to higher overall decoding speedup.

Ablating Parameter Sharing. To isolate the contribution of parameter sharing, we train an additional SEED control on GSM8K using Qwen3-1.7B-Base in which the drafter and verifier are fully parameter-disjoint, aiming to directly test whether enforced parameter sharing improves drafterverifier alignment relative to an identically initialized but independently parameterized drafter. The two-layer drafter is initialized by copying the verifier’s token embedding and tied output head, final normalization layer, and top two transformer layers. The drafter continues to consume the verifierproduced KV cache from layers 27 and 28, preserving the deep-feature conditioning mechanism. Both variants in this ablation use training block size $b = 4$ . All other training settings and objectives are unchanged. We evaluate both variants with a fixed draft length of 4 and without tree-based verification.

Table 7 shows that removing parameter sharing substantially degrades both draft quality and task performance, despite the fact that this variant has more parameters and, in principle, greater expressive capacity. Although the independently parameterized drafter retains the same initialization and deep feature conditioning mechanism, it still suffers from poorer draft token acceptance, resulting in reduced throughput. The degraded task performance further suggests that parameter sharing provides an important alignment benefit between the drafter and the verifier, making the joint optimization process more effective.

Table 7: Parameter-sharing ablation on GSM8K with Qwen3-1.7B-Base. Removing parameter sharing substantially degrades both draft quality and task performance.
<table><tr><td></td><td>GSM8K Acc. (%)</td><td>Throughput (tokens/s)</td><td>Draft Token Acc. Rate (%)</td><td>Draft Token Acc. Length</td></tr><tr><td>SEED</td><td>57.5</td><td>74.3</td><td>73.0</td><td>2.9</td></tr><tr><td>w/o Parameter Sharing</td><td>44.1↓13.4</td><td>68.8↓5.5</td><td> $6 8 . 5 ^ { \downarrow 4 . 5 }$ </td><td>2.7↓0.2</td></tr></table>

## E Additional Experimental Results

## E.1 Why SEED Improves Task Performance

Here we provide some further theoretical intuition as to why multi-token prediction (MTP) and SEED achieve stronger task performance than standard autoregressive (AR) fine-tuning. We also include additional results of training linear probes to predict future tokens on two more datasets in Figures 10 and 11, using the intermediate representations of Qwen3-1.7B-Base AR and SEED models.

In MTP-style training, the model learns to predict multiple future tokens from a single contextual representation. Concretely, given the hidden states of the final context token $h ( t _ { 0 } )$ , the model learns to approximate $P ( t _ { 1 } , t _ { 2 } , { \dot { t } } _ { 3 } \mid h ( t _ { 0 } ) )$ , forcing $h ( t _ { 0 } )$ to encode information that is predictive not only of the immediate next token but also of tokens further into the future, which encourages the model to implicitly plan ahead.

Similarly, during SEED drafting, the decoder learns to model $P ( t _ { 1 } ~ \vert ~ H _ { t _ { 0 } } ^ { 0 } , K V _ { < t _ { 0 } } ^ { d e c } ) , P ( t _ { 2 } ~ \vert ~$ $H _ { t _ { 1 } } ^ { 0 } , H _ { t _ { 0 } } ^ { 0 } , K V _ { < t _ { 0 } } ^ { d e c } ) , P ( t _ { 3 }  \mid H _ { t _ { 2 } } ^ { 0 } , H _ { t _ { 1 } } ^ { 0 } , H _ { t _ { 0 } } ^ { 0 } , K V _ { < t _ { 0 } } ^ { d e c } )$ and so on, where $H _ { t _ { i } } ^ { 0 }$ denotes the raw token embedding of token $t _ { i }$ . Importantly, these embeddings are fixed lookup vectors that cannot adapt to absorb predictive information during training. As a result, the gradients produced by the speculative loss $\mathcal { L } _ { s p e c }$ primarily propagate through the cached contextual representations $K V _ { < t _ { 0 } } ^ { d e c }$ , which originate from the encoder’s hidden states. Consequently, the encoder is encouraged to compress information about upcoming tokens into its contextual representations so that the decoder can reliably predict future tokens during drafting.

For the linear probe experiments, we train both a SEED model and an AR model with a block size of 4 to examine the effect of planning ahead under a relatively short horizon. We then extract the representations from the final encoder layer l<sup>′</sup> of the SEED model and the corresponding layer l<sup>′</sup> of its AR fine-tuned counterpart, run both models on the GSM8K test split, collect the hidden states produced at layer l<sup>′</sup>, and train five linear probes to independently predict future tokens at lookahead steps from 1 to 5. Specifically, we collect 20,000 target-only token positions from 1,000 samples in the test split and compute prediction accuracy against the ground-truth labels from the original datasets.

![](images/3cf1c2dc4d3e0ff5f8997f8f1b6e2d0fe47d2fc679eab41dc7034264a23ba78a.jpg)  
Figure 10: Training linear probes to predict future tokens from intermediate representations on Kodcode.

![](images/74b32a2b350ef7da3d8cabf42e0756bc4533206ab956ce41afcddbfb896ca6a8.jpg)  
Figure 11: Training linear probes to predict future tokens from intermediate representations on ScienceQA.

## E.2 Accelerating an Already Well-Tuned Model

In our main experiments, SEED is applied during task-specific fine-tuning, yielding both inference acceleration and improved task performance. A natural follow-up question is whether SEED can be used purely to accelerate an already strong task model, without sacrificing its existing quality. To study this practical setting, we start from a Qwen3-1.7B-Base model that has already been fine-tuned with standard autoregressive (AR) training on GSM8K. We then perform an additional round of training to convert this AR model into a SEED model. The training setup is identical to that described in Appendix C.4, except that initialization comes from the task-tuned AR checkpoint rather than the untuned base model.

Table 8 summarizes the results. Retrofitting SEED onto an already fine-tuned AR model preserves task accuracy while achieving essentially the same decoding throughput as a SEED model trained directly from the base model. The small accuracy gain is consistent with the MTP-like "planning ahead" effect induced by the speculative objective, as discussed in the main body.

These results are also aligned with our CKA analysis in Figure 5, which shows that SEED primarily reshapes internal model representations near the encoder–decoder interface while leaving earlier layers largely unchanged. This suggests that standard AR fine-tuning and SEED training are structurally compatible: the latter can be placed on top of the former without disrupting previously learned task knowledge.

Table 8: SEED applied as a second-stage fine-tuning procedure on top of an AR-fine-tuned model. Dynamic drafting (τ = 0.7) and tree-based verification are used for SEED models. Continued fine-tuning preserves accuracy on GSM8K while achieving comparable speedup.
<table><tr><td>Model</td><td>Accuracy (%)</td><td>Tput (tokens/s)</td><td>Acc. Len</td><td>Acc. Rate (%)</td></tr><tr><td>AR (from base)</td><td>56.3</td><td>37.9</td><td></td><td></td></tr><tr><td>SEED (continued from AR)</td><td>56.5↑0.2</td><td>90.6↑52.7</td><td>3.91</td><td>86.82</td></tr><tr><td>SEED (from base)</td><td>57.1</td><td>91.8</td><td>3.97</td><td>88.57</td></tr></table>

## F Computational Cost Analysis

Here we provide a more comprehensive analysis of the theoretical computational cost of each method. For concreteness, we use Qwen3-1.7B-Base as an example. The model has $l \ = \ 2 8$ transformer layers with $d _ { \mathrm { m o d e l } } = 2 0 4 8 , d _ { \mathrm { f f n } } = 6 1 4 4$ , 16 query heads, 8 KV heads, $d _ { \mathrm { h e a d } } = 1 2 8 .$ a vocabulary size of $| V | = 1 5 1 , 9 3 6 .$ , and tied input / output embeddings. Each attention block contains $\dot { W _ { Q } } \left( 2 0 4 8 \times 2 0 4 8 \right) + \dot { W } _ { K } \left( 2 0 4 8 \times 1 0 2 \dot { 4 } \right) + \dot { W _ { V } } \left( 2 0 4 8 \times 1 0 2 \overline { { { 4 } } } \right) + { W _ { O } } \left( 2 0 4 8 \times 2 0 4 8 \right)$ for a total of 12.58M parameters. The MLP block contributes $3 \times ( 2 0 4 8 \times 6 1 4 4 ) = 3 7 . 7 5 \dot { \mathrm { M } }$ parameters. Therefore, each transformer layer contains $P _ { \mathrm { l a y e r } } = 5 0 . 3 3 \mathrm { { M } }$ parameters, and all 28 layers together contain 1.409B parameters. The tied embedding / LM head contributes an additional 151936 × 2048 = 311.2M parameters, resulting in a total of $P _ { \mathrm { f u l l } } = 1 . 7 2 1 \mathrm { \mathrm { B } }$ parameters. Throughout the following analysis, we normalize the computational cost of a full forward pass through the entire model as 1 unit, and express the cost of all compared methods relative to this baseline. Note that this parameter-based analysis is intended as a simple estimate of relative computational cost. It assumes that the amount of computation roughly scales with the number of parameters touched while ignoring operation-specific factors such as differences in FLOPs, memory access patterns, and hardware utilization across components.

Table 9: Approximate drafting cost for a 1.7B model. Note that Apple MTP amortizes a full-model call over a draft block.
<table><tr><td>Method</td><td>Weights Touched per Draft Token</td><td>Weights</td><td>Cost (units)</td></tr><tr><td>AR</td><td>28 layers + head</td><td>1721M</td><td>1.000</td></tr><tr><td>SEED</td><td>2 decoder layers + head</td><td>411.9M</td><td>0.239</td></tr><tr><td>LayerSkip</td><td>8 layers + head</td><td>713.8M</td><td>0.415</td></tr><tr><td>Apple MTP</td><td>(28 layers + head) per block + sequential sampler head</td><td>≥ 1721M</td><td>≥ 1.000</td></tr><tr><td>EAGLE-3</td><td>1 fused draft layer + reduced-vocab head</td><td>≈ 137M</td><td>≈ 0.080</td></tr></table>

As shown in Table 9, SEED’s drafting step is 4.2× cheaper than a full pass and 1.7× cheaper than LayerSkip’s early-exit pass, despite conditioning on strictly deeper representations.

Verification over (d+1) tokens requires one full-model forward pass for all methods except LayerSkip. Since LayerSkip reuses the KV cache of the first 8 layers computed during drafting, verification only recomputes layers 9 - 28 together with the LM head. The corresponding cost is therefore $( 2 0 ^ { - } \times 5 0 . 3 3 ^ { - } + 3 1 1 . { \dot { 2 } } ) / 1 7 2 1 = 0$ .766 relative to a full forward pass.

Now consider a fixed draft length d and per-token acceptance rate r. Each speculative decoding round incurs a computational cost of

$$
T _ { \mathrm { u n i t } } = d \cdot c _ { \mathrm { d r a f t } } + c _ { \mathrm { v e r i f y } }
$$

where $c _ { \mathrm { d r a f t } }$ and $c _ { \mathrm { v e r i f y } }$ denote the per-token drafting cost and the verification cost respectively. In return, each round commits an average of

$$
n = d \cdot r + 1
$$

tokens, consisting of the accepted draft tokens plus the bonus token generated during verification. Using the measured acceptance rates at $d = \dot { 4 } \ : ( \mathrm { S E E D } ; 7 3 . 0 \%$ , Apple MTP: 69.8%, LayerSkip: 43.1%) and taking one computational unit to correspond to a full autoregressive forward pass $( 1 / 3 7 . 9 \mathrm { s } = 2 6 . 4 $ ms, based on the fine-tuned Qwen3-1.7B-Base achieving 37.9 tokens/s on GSM8K with a single A6000), we obtain the following throughput predictions by computing $1 0 0 0 / T _ { m s } \times n \colon$

Despite its simplicity, this computational model closely predicts SEED’s measured throughput (w/ fixed draft length and w/o tree-based verification) and captures the relative performance trends across methods. The remaining discrepancies particularly for LayerSkip and Apple MTP are expected because the model omits certain operation-specific costs and implementation overheads.

For completeness, EAGLE-3 achieves a substantially lower per-token drafting cost (0.08 units), owing to its lightweight single-layer drafter and reduced-vocabulary draft head. Therefore, SEED’s advantage does not primarily come from reducing the absolute drafting cost. Instead, it arises from the improved draft quality and higher acceptance rates.

Table 10: Approximate throughput predictions for fixed draft length $d = 4$ without tree-based verification. Apple MTP generates all four draft tokens in a single forward pass. However, its sampler applies the 311M-parameter LM head sequentially once per draft token, contributing an additional cost of $4 \times 3 1 1 . 2 / \mathrm { { \bar { 1 } } 7 2 1 = 0 . 7 2 }$ units.
<table><tr><td>Method</td><td> $c _ { \mathrm { d r a f t } }$ </td><td> $c _ { \mathrm { v e r i f y } }$ </td><td> $T _ { \mathrm { u n i t } }$ </td><td> $\begin{array} { c } { { T } } \\ { { \mathrm { ( m s ) } } } \end{array}$ </td><td>n</td><td>Predicted (tokens/s)</td><td>Measured (tokens/s)</td></tr><tr><td>SEED</td><td>0.239</td><td>1.000</td><td>1.96</td><td>51.7</td><td>3.92</td><td>75.8</td><td>74.3</td></tr><tr><td>LayerSkip</td><td>0.415</td><td>0.766</td><td>2.43</td><td>64.1</td><td>2.72</td><td>42.4</td><td>47.0</td></tr><tr><td>Apple MTP</td><td>1.72/4</td><td>1.000</td><td>2.72</td><td>71.8</td><td>3.79</td><td>52.8</td><td>44.6</td></tr></table>

The relative drafting cost of SEED also becomes more favorable as the backbone model scales. For Qwen3-4B-Base (36 layers, $d _ { \mathrm { m o d e l } } = 2 5 6 0 , P _ { \mathrm { l a y e r } } = 1 0 1 \mathbf { M }$ , and a tied embedding / LM head of 389M parameters), the drafting cost is $( 2 \times 1 0 1 { \dot { + } } 3 8 9 ) / 4 0 2 0 = 0 . 1 4 7$ units, compared with 0.239 for Qwen3-1.7B-Base. This scaling behavior is consistent with the larger average speedup observed on the 4B model.

## G Broader Impacts

This work develops SEED, a method for accelerating large language model (LLM) inference by improving self-speculative decoding. The primary positive societal impact is that faster inference can reduce the computational cost, latency, and energy consumption of deploying LLMs. More efficient inference may make advanced language models more accessible to researchers, developers, and users with limited computational resources, and may enable lower-cost applications in domains such a education, programming assistance, summarization, and scientific support.

At the same time, improving LLM inference efficiency can also amplify existing risks associated with language models. Lowering the cost and latency of generation could make it easier to produce large volumes of harmful or misleading content, including spam, phishing messages, disinformation, synthetic reviews, or other forms of automated manipulation. Faster inference may also facilitate misuse in surveillance, impersonation, or automated social engineering systems when combined with models capable of generating persuasive or personalized text.

Nevertheless, these risks are not unique to SEED, but they are relevant because the method improves the efficiency of LLM generation. Future work on efficient inference should continue to evaluate not only speed and quality, but also whether acceleration changes the scale or detectability of potential misuse.