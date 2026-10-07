# Hybrid Latent Attention for Looped Language Models

Yuhan Chen<sup>1</sup>, Siyuan Zhang<sup>2</sup>, Nan Wang<sup>2</sup>, Feiyang Kang<sup>1</sup>, Ruoxi Jia<sup>1</sup>

<sup>1</sup>Virginia Tech, <sup>2</sup>Independent Researcher

## Abstract

Looped language models apply the same stack of layers T times to each token, which deepens the model without adding parameters but multiplies its key–value (KV) cache by T. The larger cache limits how many sequences a GPU can decode at once and slows each decoding step, which reads the whole cache. We propose Hybrid Latent Attention (HLA), which keeps exact keys and values within a sliding window of W recent tokens and stores each older token as a compact latent that the query of each loop reads directly, without reconstructing keys and values. We uptrain HLA on Ouro looped models (T=4) with 1.4B and 2.6B parameters, keeping the pretrained weights frozen and training only the added parameters to reproduce the original attention. The cache shrinks by 10.7× per token, fitting 4.0– 8.8× as many concurrent sequences per GPU, and decoding throughput improves by 2.5× at 1K-token contexts and by up to 7.4× at 16K. HLA retains over 97% of the original accuracy on math, knowledge and reasoning benchmarks, and 96– 100% on long-context retrieval up to 16K tokens. After supervised fine-tuning, it performs on par with the fine-tuned original model on competition-level math.

## 1 Introduction

A looped language model applies the same stack of Transformer layers T times to each token, gaining depth without adding parameters [Dehghani et al., 2019, Geiping et al., 2025, Zhu et al., 2025]. Ouro models with 1.4B and 2.6B parameters and T=4 are reported to match standard Transformers of up to 12B parameters on a wide range of benchmarks [Zhu et al., 2025].

The parameters saved by looping are paid back at inference time. Each loop attends to its own keys and values, so the key–value (KV) cache of a looped model is T times that of a non-looped model with the same layers. This cache bounds how many sequences decode together: an 80 GB GPU holds only five 16K-token sequences of Ouro-1.4B. Every decoding step also reads the whole cache, so the time per step of a single sequence grows from 13.6 ms at 1K tokens of context to 49.3 ms at 16K (Section 5.2). Faster decoding therefore requires shrinking both what is stored per token and the work spent on each stored token at every step.

Prior work reduces this cache in two ways. The first, suggested with Ouro [Zhu et al., 2025], keeps only the last loop’s keys and values for generated tokens, so most loops read keys and values that were not computed for them. Ouro reports a small loss on GSM8K and MATH-500, but O’Neill and Reid [2026] find this reuse unsafe, and with long generations a variant at our memory budget retains 60% of the math accuracy of the full cache on Ouro-1.4B (Section 5). The second, Looped Latent Attention (LLA) [O’Neill and Reid, 2026], stores a compact latent per token, using the low-rank trajectory of its keys and values across loops, but every decoding step rebuilds the keys and values of each loop from the latents of the whole history; in our measurements LLA reaches 7–18% of the throughput of the full cache (Section 5.2).

This paper asks whether attention can read such a latent as it is, without reconstruction. Hybrid Latent Attention (HLA) writes one latent per token and layer as a linear function of the token’s hidden states across loops, plus a smaller latent for the first loop, whose keys and values are the hardest to predict (Section 3.2). Each loop maps its query into the latent space, scores it against the stored latents, and maps their weighted sum to the attention output (Fig. 1); both maps act on the current token only, so no keys or values are rebuilt.

![](images/eaf14faa82b5c92ef7ab4964d1b309521714402e3f9c6645d7cc3955b9165a78.jpg)  
Figure 1: Hybrid Latent Attention in one layer of a looped model with T loops. (a) The full cache stores the keys and values (KV) of every loop for every token. (b) HLA keeps exact KV for the last W tokens and stores each older token as one latent shared by loops 2 to T and a smaller latent for loop 1. (c) Attention within one loop. Blue modules are added by HLA and white modules are the frozen pretrained projections. The query and output maps let latent attention read the latent cache directly, and latent and exact attention share one softmax.

In a model that reads every past token through its latent, the 127 nearest tokens receive about a quarter of the attention but account for about half of its divergence from the original attention (Section 5.5 and Appendix D). HLA therefore keeps the exact keys and values of the most recent W tokens, a fixed cost per sequence, and reads all earlier tokens through the latent, with one softmax over both parts.

We add HLA to a pretrained model by uptraining [Ainslie et al., 2023]: with the pretrained weights frozen, the new maps are trained for 1,200 steps to match the attention of each layer of the original model. Fine-tuning is less direct, since a token’s latent is complete only after its last loop, yet later tokens read it before their own last loop; a single parallel pass therefore does not reproduce decoding, but two passes, the first writing the latents and the second reading them, come close to it (Section 4.2).

On Ouro-1.4B and Ouro-2.6B served with vLLM [Kwon et al., 2023], HLA shrinks the cache by 10.7× per token, from 768 to 72 KiB on the smaller model. With the window included, a GPU holds 4.0–8.8× as many sequences at contexts from 1K to 16K tokens, and peak decoding throughput rises by 2.5–7.4×. With the pretrained weights frozen, HLA retains over 97% of the original accuracy on MATH500 [Hendrycks et al., 2021, Lightman et al., 2023], MMLU-Pro [Wang et al., 2024] and BBH [Suzgun et al., 2023] and 96–100% on RULER [Hsieh et al., 2024] up to 16K tokens, and after two-pass fine-tuning it performs on par with the fine-tuned original models on AIME and HMMT problems.

Our contributions are as follows.

• Hybrid Latent Attention, which keeps exact keys and values for the most recent W tokens and stores older tokens as compact latents that each loop reads directly, without reconstructing keys and values.

• Uptraining with frozen weights and two-pass fine-tuning, which add HLA to a pretrained looped model and keep it trainable.

## 2 Related Work

Looped language models. Reusing one block of layers across depth goes back to Universal Transformers [Dehghani et al., 2019], ALBERT [Lan et al., 2020] and deep equilibrium models [Bai et al., 2019], and looped Transformers can emulate programs and learn iterative algorithms [Giannou et al., 2023, Yang et al., 2024]. Recent work scales looping to language modeling, with a 3.5Bparameter recurrent-depth model [Geiping et al., 2025] and the 1.4B and 2.6B Ouro models [Zhu et al., 2025]. These works study quality per parameter and leave open the key–value cache that grows with the number of loops.

Key–value cache reduction. Standard Transformers shrink the cache by sharing keys and values across heads [Shazeer, 2019, Ainslie et al., 2023] or layers [Sun et al., 2024, Wu and Tu, 2024, Liu et al., 2024a], quantizing them [Liu et al., 2024b, Hooper et al., 2024], or evicting unimportant tokens [Zhang et al., 2023, Li et al., 2024], axes largely orthogonal to compression across loops. Low-rank methods store a latent per token: Palu reconstructs keys and values from it [Chang et al., 2025], while MLA absorbs the up-projection into the query and output matrices so that attention reads the latent itself [DeepSeek-AI, 2024, Meng et al., 2025]. For looped models, Ouro keeps only the last loop’s keys and values for generated tokens [Zhu et al., 2025], which LLA [O’Neill and Reid, 2026] and our long-generation results (Section 5) find to be an unsafe replacement for the per-loop cache. LLA, the work closest to ours, stores a per-head low-rank latent of the loop trajectory and, like Palu, reconstructs loop-specific keys and values on every read, which lowers decoding speed from 748 to 108 tokens per second in their report. HLA instead carries the absorption of MLA across loops with loop-specific maps, so attention reads the latent without reconstruction, and its exact window follows the common practice of keeping recent tokens uncompressed [Rae et al., 2020, Liu et al., 2024b]. FlashLoop [Yang and Liu, 2026] takes a training-free route through token sparsity, sparse attention and residual quantization, which is complementary in principle to a lower-dimensional latent.

## 3 Hybrid Latent Attention

Hybrid Latent Attention replaces the per-loop keys and values of past tokens with two parts: a latent for every past token, which each loop reads directly, and the exact keys and values of the most recent W tokens.

## 3.1 Preliminaries

A looped language model applies its L Transformer layers T times to each token, with the same weights in every loop. We write $h _ { i , t } \in \mathbb { R } ^ { d }$ for the normalized attention input of token i at loop t in a given layer, and omit the layer index because all layers are treated alike. Each of the H heads projects $h _ { i , t }$ to a query $q _ { i , t } ,$ , a key $k _ { i , t }$ and a value $v _ { i , t }$ in R<sup>dh</sup> , with $H d _ { h } = d ,$ , and rotary embeddings [Su et al., 2021] apply a position-dependent rotation $R _ { i }$ to queries and keys. At loop t, token i attends to every token $j \leq i \colon$

$$
\begin{array} { c } { { \displaystyle { s _ { i j , t } = \frac { ( R _ { i } q _ { i , t } ) ^ { \top } ( R _ { j } k _ { j , t } ) } { \sqrt { d _ { h } } } , } } } \\ { { \displaystyle a _ { i \cdot , t } = \mathrm { s o f t m a x } _ { j \leq i } ( s _ { i j , t } ) , \qquad o _ { i , t } = \sum _ { j \leq i } a _ { i j , t } v _ { j , t } . } } \end{array}\tag{1}
$$

The hidden states differ across loops, and so do the keys and values. Decoding caches the rotated keys $R _ { j } k _ { j , t }$ and the values $v _ { j , t }$ of every past token at every loop, 2T Ld numbers per token, which we call thefull cache.

## 3.2 Latent writer

The writer replaces the keys and values that a token produces across loops with a latent. One latent can serve several loops because their keys and values follow a low-rank trajectory across loops [O’Neill and Reid, 2026]. The writer therefore takes the hidden states of all the loops a latent covers, and the first loop, whose keys and values are the hardest to predict from the final hidden state, receives a separate, smaller latent (Section 5.5).

For each layer, HLA stores two latents per token. The main latent accumulates a linear map of the hidden state at every loop after the first, and the loop-1 latent is a linear map of the first hidden state:

$$
\begin{array} { r l r } & { } & { c _ { j } = \displaystyle \sum _ { t = 2 } ^ { T } E _ { t } h _ { j , t } = \left[ c _ { j } ^ { K } \right] , } \\ & { } & { \qquad c _ { j } ^ { ( 1 ) } = E _ { 1 } h _ { j , 1 } = \left[ c _ { j } ^ { K , 1 } \right] . } \\ & { } & { \qquad \quad c _ { j } ^ { ( 1 ) } = E _ { 1 } h _ { j , 1 } = \left[ c _ { j } ^ { K , 1 } \right] . } \end{array}\tag{2}
$$

The key part $c _ { i } ^ { K } \in \mathbb { R } ^ { r _ { k } }$ and the value part $c _ { j } ^ { V } \in \mathbb { R } ^ { r _ { v } }$ play the roles of keys and values for loops 2 to $T .$ , and $c _ { i } ^ { K , 1 } , c _ { i } ^ { V , 1 } \in \mathbb { R } ^ { r _ { 1 } }$ play them for loop 1. All heads of a layer share these latents. The main latent is complete once token j has finished its last loop, and it is never updated afterwards. In total, a token stores $r _ { k } + r _ { v } + 2 r _ { 1 }$ numbers per layer in place of 2Td.

## 3.3 Latent reader

Loop t of the current token i reads the latent with two linear maps per head, $A _ { t } \in \mathbb { R } ^ { d _ { h } \times r _ { k } }$ and $B _ { t } \in \mathbb { R } ^ { r _ { v } \times d _ { h } }$ . The first carries the query into the key part of the latent space, where it is scored against the stored latents:

$$
\hat { s } _ { i j , t } = \frac { ( \tilde { R } _ { i } A _ { t } ^ { \top } q _ { i , t } ) ^ { \top } ( \tilde { R } _ { j } c _ { j } ^ { K } ) } { \sqrt { d _ { h } } } .\tag{3}
$$

The second carries the attention-weighted sum of value latents back to the head dimension, as given in Eq. (4) below. Loop 1 uses its own maps $A _ { 1 } , B _ { 1 }$ on $c _ { j } ^ { K , 1 }$ and $c _ { j } ^ { V , 1 }$ in the same way. $A _ { t }$ acts on one query and $B _ { t }$ on one summed vector, so their cost does not depend on the number of past tokens, and no per-loop key or value is formed for any of them.

Equation (3) needs a rotation $\tilde { R } _ { j }$ that acts on latents. In Eq. (1) the rotation $R _ { j }$ turns each pair of key coordinates by an angle $j \theta _ { f }$ , with one frequency $\theta _ { f }$ per pair. We organize the key latent in pairs as well, give every frequency the same number $m = r _ { k } / d _ { h }$ of them, and let $\tilde { R } _ { j }$ turn a latent pair of frequency $f \log j \bar { \theta } _ { f }$ . The stored key latent is rotated once, when it is written. This rotation commutes with any linear combination of key pairs that share a frequency, which TransMLA uses to combine keys across heads [Meng et al., 2025]; here the combination also runs across loops. A latent of this form keeps m of the $( T - 1 ) H$ key directions of each frequency, so its scores approximate the original ones and are exact only when $\dot { m } = ( T - 1 ) H$ . Our initialization takes the leading principal directions of each frequency (Section 4.1), and Appendix A gives the construction in full. After training, $A _ { t }$ is a dense matrix and this property is no longer guaranteed (Appendix B); we instead train with shifted positions so that the learned scores remain accurate at long distances (Appendix F).

## 3.4 Exact sliding window

We find that the most recent tokens account for a large share of the error of the latent (Section 5.5), so we store them exactly. HLA keeps the exact keys and values of the W most recent tokens and reads the latent only for tokens further back. Token i at loop t forms one score vector from the latent scores $\hat { s } _ { i j , t }$ of tokens $j < i - W$ and the exact scores $s _ { i j , t }$ of tokens $i - W \leq j \leq i$ , normalizes it with a single softmax, and combines the two kinds of values:

$$
o _ { i , t } = B _ { t } ^ { \top } \sum _ { j < i - W } a _ { i j , t } c _ { j } ^ { V } \ + \ \sum _ { i - W \leq j \leq i } a _ { i j , t } v _ { j , t } .\tag{4}
$$

The token’s own keys and values are always exact, since they are computed in the current step. With $W { = } 0$ , Eq. (4) reduces to pure latent attention.

A sequence of n tokens thus stores the latent for every token and the full cache for $W$ of them. The latent takes $L ( r _ { k } + r _ { v } + 2 r _ { 1 } )$ numbers per token, and the window adds 2TLdW numbers per sequence, independent of n. As n grows, the total approaches a fraction $( r _ { k } + r _ { v } + 2 r _ { 1 } ) / ( 2 T d )$ of the full cache.

During inference, a prompt is processed in one pass with exact attention, as in the original model, and the latents of its tokens are written from that pass. Generation then proceeds one token at a time with Eq. (4). Each generated token writes its latent as it finishes its loops and places its exact keys and values in the window, from which the oldest token is dropped. A token is thus read exactly while it is among the W most recent and through its latent afterwards.

## 4 Training

## 4.1 Uptraining

HLA adds the maps $E _ { t } , A _ { t }$ and $B _ { t }$ to a pretrained model, 0.35B parameters for Ouro-1.4B and 0.70B for Ouro-2.6B, and uptraining trains only these maps with the pretrained weights frozen.

We initialize $E _ { t } , A _ { t }$ and $B _ { t }$ with a principal component analysis of the keys and values that the pretrained model produces on 128 calibration sequences: the writer projects them onto the leading components, and the reader maps queries and outputs through the same components (Appendix A). For the keys, the components are computed separately for each rotary frequency, as in TransMLA [Meng et al., 2025], but across loops 2 to T as well as across heads, which gives the frequency-aligned form required in Section 3.3. The values of loops 2 to $T$ are projected jointly, and loop 1 is treated on its own. At full rank this initialization reproduces the original attention exactly; at the ranks we use it only approximates it, and uptraining reduces the remaining error.

The pretrained model with its full cache serves as the reference. We run it on a training sequence and record, for every layer and loop, its attention distribution $a _ { i \cdot , t } ^ { \star }$ and its attention output $o _ { i , t } ^ { \star }$ after the output projection. From the same hidden states we compute the HLA distribution $\hat { a } _ { i \cdot , t }$ and output $\hat { o } _ { i , t }$ with Eqs. (3) and (4), and minimize

$$
\begin{array} { r l r }   { \mathcal { L } = \frac { 1 } { L T } \sum _ { \mathrm { l a y e r s } } \sum _ { t = 1 } ^ { T } \Bigl [ \mathrm { K L } \bigl ( a _ { i \cdot , t } ^ { \star } \parallel \hat { a } _ { i \cdot , t } \bigr ) } \\ & { } & { + \frac { \parallel \hat { o } _ { i , t } - o _ { i , t } ^ { \star } \parallel ^ { 2 } } { \parallel o _ { i , t } ^ { \star } \parallel ^ { 2 } } \Bigr ] , } \end{array}\tag{5}
$$

averaged over tokens and heads. The first term fits where each loop attends, and the second fits what it reads. Scores inside the window are exact in both models, so the latent is supervised only on the tokens it serves at decoding time.

One forward pass of the reference provides the hidden states of every token, layer and loop, so the loss is evaluated for all of them in parallel and each layer is fitted independently.

Uptraining has two phases and uses the same settings for both models. The first runs 1,000 steps on sequences of up to 2,048 tokens, drawn from the math reasoning traces of OpenR1-Math-220k [Open-R1, 2025] and the web text of FineWeb-Edu [Penedo et al., 2024]. Such sequences never show the latent a distant token, so for three quarters of them we shift the positions of up to four chunks by random offsets, as in PoSE [Zhu et al., 2024], which exposes the maps to relative distances of up to 16K tokens without longer inputs. The second phase runs 200 steps and adds eight documents of 16K tokens to every batch, taken from the books and code repositories of the ProLong training data [Gao et al., 2025], so that the latent is also fitted on real long contexts. Data and optimizer settings are detailed in Appendix C.

## 4.2 Fine-tuning

To test whether the latent supports further training, we fine-tune all weights, including the HLA maps. Uptraining runs HLA attention on the hidden states of the reference, so it never sees the error that HLA attention accumulates during generation; the accuracy in Section 5.3 is measured by decoding and includes it. Fine-tuning minimizes the task loss of the HLA model itself, so its forward pass has to run HLA attention on the model’s own hidden states, as decoding does, which raises a difficulty that the full cache does not have.

With the full cache, loop t of token i reads the loop-t keys and values of earlier tokens, so all tokens of a training sequence can advance through loop t together. With HLA, loop 2 of token i already reads the main latent of every token $j < i - W$ , and Eq. (2) completes that latent only after loop T of token j. Decoding meets this order by generating one token at a time, but processing a training sequence token by token would give up the parallelism of training.

![](images/a65aff95cff73b5a52f0badf8656968c51e6e66ba2ba6c73817e16f90dd034f7.jpg)  
Figure 2: Decoding on one A100 GPU with the full cache, LLA and HLA; each sequence has a prompt of the given length and generates 256 tokens. Left: the largest batch that fits in memory. Middle: generated tokens per second summed over the batch, at the best batch size. Right: time per generated token for a single sequence. Labels give the ratio of HLA to the full cache; the left and middle columns use logarithmic axes. Table 8 lists the values.

We recover parallel training with two passes over the sequence. The first pass runs the model with exact attention and writes the main latents of all tokens with Eq. (2). The second pass runs all tokens in parallel under the decoding rule of Eq. (4): prompt tokens attend exactly, as at inference, and response tokens read exact keys and values inside the window and, beyond it, loop-1 latents from the same pass and main latents from the first pass. The training loss is computed on the second pass, and gradients flow through both passes. For prompt tokens, the first-pass latents equal those of decoding; for response tokens, they come from hidden states computed with exact attention rather than with HLA. Rewriting the main latents from the second pass and running it again makes $W { + 1 }$ more response tokens match decoding exactly, and two passes already come close to sequential decoding (Appendix E).

## 5 Experiments

We evaluate HLA on three questions: how much memory and decoding time it saves (Section 5.2), how much of the original attention and accuracy it keeps after uptraining (Section 5.3), and whether it can still be fine-tuned (Section 5.4). Section 5.5 then examines the design choices.

## 5.1 Setup

Models. We use the pretrained Ouro-1.4B and Ouro-2.6B checkpoints [Zhu et al., 2025] with T=4 loops. Both have hidden size d=2048; Ouro-1.4B has L=24 layers and Ouro-2.6B has L=48. HLA uses $r _ { k } = r _ { v } = 5 1 2 , r _ { 1 } = 2 5 6$ and a window of W=128 for both models, and is uptrained as described in Section 4.1.

Cache size. At 16-bit precision the full cache holds 2T Ld numbers per token, 768 KiB on Ouro-1.4B and 1,536 KiB on Ouro-2.6B. HLA stores $r _ { k } + r _ { v } + 2 r _ { 1 } = 1 , 5 3 6$ numbers per layer instead of $2 T d = 1 6 { , } 3 8 4$ , which is 72 and 144 KiB per token, 10.7× less. The window adds 96 and 192 MiB per sequence, so a 16K-token sequence takes 9.8× less memory than with the full cache.

![](images/19cb168b1d2ab3b10aa506896baeb5e87b299fe5b461a94d8afb7ef005464295.jpg)  
Figure 3: Uptraining curves on held-out data for Ouro-2.6B (Ouro-1.4B in Fig. 6): KL divergence between the attention of HLA and of the full cache at each loop. Phase 1 is evaluated on short sequences (logarithmic axis), phase 2 on 16K-token documents (linear axis, right).

Baselines. The full cache is the original model. Single-loop sharing keeps the full cache of the last loop only: every earlier loop reads its own keys and values inside the same window of W=128 tokens and those of the last loop beyond it. It needs no training and stores a quarter of the full cache per token, which is 2.7× the latent of HLA. LLA [O’Neill and Reid, 2026] stores a latent of rank 256 per head, as much as single-loop sharing, and rebuilds the keys and values of every loop from it at each step; for accuracy we distill its codec with its authors’ objective for 1,000 steps and decode with its two-pass decoder (Appendix H).

Benchmarks. We measure math problem solving on MATH500 [Hendrycks et al., 2021, Lightman et al., 2023], knowledge on MMLU-Pro [Wang et al., 2024], reasoning on BBH [Suzgun et al., 2023], and long-context retrieval on RULER [Hsieh et al., 2024] at context lengths of 4K, 8K and 16K tokens. MMLU-Pro, BBH and RULER are run with lm-evaluation-harness [Biderman et al., 2024], except for the two question-answering tasks of RULER. With their default prompt and 32 new tokens, Ouro often begins by reasoning about the question and runs out of tokens before it answers, so the score reflects how often a cache leads into such a reply more than how well it retrieves (Appendix H). We therefore wrap these prompts in the chat template of the model and allow 2,048 new tokens; the other 11 tasks begin the answer in the prompt and keep their settings. For MATH500 we sample four solutions per problem, with up to 8,192 new tokens each, and report mean accuracy. Fine-tuned models are evaluated on 90 competition problems from AIME 2024 [Hugging Face H4, 2024], AIME 2025 [math-ai, 2025] and HMMT February 2025 [Balunovic et al.´ , 2025], with 16 samples per problem.

Serving. All models are served with vLLM [Kwon et al., 2023] at 16-bit precision on A100 GPUs with 80 GB of memory. HLA is implemented as an attention backend that stores the latent and the window in vLLM’s paged cache (Appendix I).

## 5.2 Memory and decoding speed

At 1K tokens of context HLA fits about four times as many sequences on one GPU as the full cache, and at 4K to 16K between 6.5 and 8.8 times as many (Fig. 2). The ratio is lower at 1K because the window costs every sequence the same amount and weighs more when the sequence is short. Throughput rises less than capacity: by 2.5 times at 1K for both models, and by 4.4 and 7.4 times at 16K.

The time of one decoding step explains part of this gap. For a single Ouro-1.4B sequence, the latency of the full cache grows by about 2.4 ms per thousand tokens of context, from 13.6 ms at 1K to 49.3 ms at 16K, and that of HLA by about 0.5 ms, from 17.1 to 25.1 ms. HLA is thus 26% slower at 1K, equal at 4K and twice as fast at 16K, and Ouro-2.6B behaves alike. We attribute its higher fixed cost to the maps $A _ { t }$ and $B _ { t }$ and to attending to the window and the latent separately.

Table 1: Accuracy (%) with the pretrained weights frozen. Single-loop sharing and HLA also keep exact keys and values for the last 128 tokens (96 MiB per sequence on Ouro-1.4B, 192 MiB on Ouro-2.6B), not counted in the per-token cache. Shading marks the accuracy that HLA retains relative to the full cache: at least 97% (green) and 90–97% (yellow). Bold marks HLA results above LLA. <sup>†</sup>One sample per problem instead of four, because LLA decodes slowly.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td rowspan="2">Cache per token (KiB)</td><td colspan="3">RULER</td><td rowspan="2">MMLU-Pro</td><td rowspan="2">BBH</td><td rowspan="2">MATH500</td></tr><tr><td>4K</td><td>8K</td><td>16K</td></tr><tr><td rowspan="4">Ouro-1.4B</td><td>Full cache</td><td>768</td><td>89.5</td><td>85.2</td><td>77.5</td><td>48.9</td><td>71.0</td><td>75.9</td></tr><tr><td>Single-loop sharing</td><td>192</td><td>19.0</td><td>10.0</td><td>7.0</td><td>16.1</td><td>43.3</td><td>45.9</td></tr><tr><td>LLA</td><td>192</td><td>87.4</td><td>82.7</td><td>71.1</td><td>46.6</td><td>69.1</td><td>73.4†</td></tr><tr><td>HLA</td><td>72</td><td>88.7</td><td>84.2</td><td>74.4</td><td>47.7</td><td>69.4</td><td>74.1</td></tr><tr><td rowspan="4">Ouro-2.6B</td><td>Full cache</td><td>1,536</td><td>92.3</td><td>86.3</td><td>82.9</td><td>56.0</td><td>80.1</td><td>81.1</td></tr><tr><td>Single-loop sharing</td><td>384</td><td>71.4</td><td>64.5</td><td>52.6</td><td>52.6</td><td>75.8</td><td>73.9</td></tr><tr><td>LLA</td><td>384</td><td>92.5</td><td>85.4</td><td>81.0</td><td>54.2</td><td>79.3</td><td>80.6†</td></tr><tr><td>HLA</td><td>144</td><td>92.2</td><td>85.0</td><td>81.5</td><td>55.5</td><td>79.8</td><td>79.4</td></tr></table>

LLA shows what reconstruction costs. We serve its reconstruct path in the same engine; the timed codec is not distilled, since its values do not affect decoding time. LLA fits about four times as many sequences as the full cache, yet the larger batch barely helps: on Ouro-1.4B at 16K it produces 6.3 tokens per second for one sequence and 7.2 for 19. Rebuilding keys and values for the whole history thus appears to dominate each step, and its peak throughput stays at 7 to 18% of that of the full cache, while HLA, which reads its latent directly, reaches 28 to 59 times the throughput of LLA.

Fixed-length prompts leave out contexts that grow during generation. We therefore let the fine-tuned models of Section 5.4 generate 16 solutions to each of 30 competition problems until completion, on four GPUs per method with the largest batch that each cache allows. HLA produces 1,176 tokens per second per GPU on Ouro-1.4B and 492 on Ouro-2.6B, 3.8 and 4.3 times the 312 and 114 of the full cache.

## 5.3 Uptraining

The savings above come from replacing the cache that the model was pretrained with, so we next measure how much of its behavior HLA keeps after uptraining. The divergence that uptraining minimizes is measured on held-out data and computed on the hidden states of the reference, as in training (Figs. 3 and 6). At initialization the attention of HLA differs from the original by 0.13 nats on average for Ouro-1.4B and 0.11 for Ouro-2.6B. The first phase reduces this to 0.006 for both models, most of it in the first 200 steps, and loops 1 and 2 remain the hardest to fit, in line with Table 3. The second phase lowers the divergence on 16K-token documents by about a fifth for both models and raises it slightly on short sequences, from 0.0063 to 0.0067 for Ouro-1.4B.

With the pretrained weights unchanged, HLA stays close to the full cache (Table 1). On MATH500, MMLU-Pro and BBH it loses 0.3 to 1.8 points and retains 97.5 to 99.6% of the original accuracy. On RULER the loss grows with the context length, from 0.8 points at 4K to 3.1 at 16K for Ouro-1.4B, where it retains 96.0%, and from 0.1 to 1.4 points for Ouro-2.6B. Retrieval separates configurations more than math does (Appendix G), so we chose the window, shifted positions and long-document phase on RULER.

Single-loop sharing stores more than HLA and loses far more. On Ouro-1.4B its MATH500 accuracy drops from 75.9 to 45.9, with 44% of its generations reaching the length limit against 12.5% for HLA, and its RULER accuracy is at most 19%. Ouro-2.6B tolerates sharing better but still loses 7.2 points on MATH500 and 30.3 on RULER at 16K. The keys and values of one loop thus do not stand in for those of the others, whereas a latent written from all loops does at a smaller size.

LLA stores 2.7 times as much per token as HLA and, on Ouro-1.4B, is less accurate on every benchmark, by 3.3 points on RULER at 16K and 1.1 on MMLU-Pro. On Ouro-2.6B the two stay within 1.3 points of each other, LLA ahead on MATH500 and on RULER at 4K and 8K, HLA ahead at 16K and on MMLU-Pro and BBH. Rebuilding the keys and values thus brings no consistent gain in accuracy over reading the latent directly, while it lowers throughput by a factor of 28 to 59.

## 5.4 Fine-tuning

We fine-tune each model on math reasoning traces for 400 steps with a batch of 128 sequences and a learning rate of $1 0 ^ { - 5 }$ , which is about one pass over the data. The original model is fine-tuned in the standard way. The HLA model starts from the uptrained checkpoint and is fine-tuned with the two-pass procedure of Section 4.2, updating all weights. Ouro-1.4B uses one repetition of the second pass and Ouro-2.6B none (Appendix E). Measured every 50 steps, the validation losses of the two stay within 0.006 of each other and end at 0.293 and 0.292 for Ouro-1.4B and at 0.267 and 0.269 for Ouro-2.6B.

Table 2: Accuracy (%) on competition math after fine-tuning on the same data for the same number of steps, averaged over 16 samples per problem. The better value of each pair is in bold.
<table><tr><td colspan="2"></td><td colspan="2">AIME</td><td colspan="2">HMMT</td></tr><tr><td>Model</td><td>Method</td><td>2024</td><td>2025</td><td>2025</td><td>Avg.</td></tr><tr><td rowspan="2">Ouro-1.4B</td><td>Full cache</td><td>20.6</td><td>19.8</td><td>8.1</td><td>16.2</td></tr><tr><td>HLA</td><td>19.2</td><td>20.2</td><td>9.8</td><td>16.4</td></tr><tr><td rowspan="2">Ouro-2.6B</td><td>Full cache</td><td>35.4</td><td>30.4</td><td>15.6</td><td>27.2</td></tr><tr><td>HLA</td><td>35.6</td><td>31.9</td><td>16.5</td><td>28.0</td></tr></table>

Table 2 gives the accuracy of the fine-tuned models. HLA reaches 16.4% against 16.2% for the full cache on Ouro-1.4B, and 28.0% against 27.2% on Ouro-2.6B, so its average is above that of the fine-tuned original at both sizes. It is also ahead on five of the six pairs of model and competition. These margins are within noise: paired bootstrap intervals over the 90 problems are [−1.6, +2.1] and [−1.2, +2.9] points for the two averages, and no difference on a single competition exceeds 1.7 points. The latent cache is thus fully competitive with the full cache after fine-tuning, at a tenth of its size per token.

## 5.5 Ablations

We first test the inputs of the writer on the keys and values of the original Ouro-1.4B, and then ablate the exact window, shifted positions and the long-document phase, each by comparing two Ouro-1.4B models with frozen weights that differ in that component only.

The writer of Section 3.2 reads the hidden states of every loop that a latent covers and gives loop 1 a separate latent; Table 3 gives the measurements behind both choices. A ridge regression from the last hidden state $h _ { j , 4 }$ predicts the keys of loop 3 with $R ^ { 2 } = 0 . 9 3$ , those of loop 2 with 0.80 and those of loop 1 with 0.66, and the values less well, so $h _ { j , 4 }$ alone loses part of the keys and values of earlier loops. A rank-512 projection fitted to loops 2–4 jointly keeps 0.79 of the key variance and 0.89 of the value variance, against 0.80 and 0.93 for a single loop, so one latent covers three loops at about the rank of one. Loop 1, the least predictable, receives its own smaller latent so that it does not enlarge the shared one.

Table 3: Keys and values of the original Ouro-1.4B, averaged over the 24 layers. Left: $R ^ { 2 }$ of a ridge regression from the hidden state of the last loop to the keys and values of each loop. Right: fraction of variance kept by a rank-512 projection fitted to one loop or to loops 2–4 jointly.
<table><tr><td></td><td colspan="4"> $R ^ { 2 }$  from  $h _ { j , 4 }$ </td><td colspan="2">Var. at rank 512</td></tr><tr><td>Loop</td><td>1</td><td>2</td><td>3</td><td>4</td><td>One</td><td>2-4</td></tr><tr><td>Keys</td><td>0.66</td><td>0.80</td><td>0.93</td><td>1.00</td><td>0.80</td><td>0.79</td></tr><tr><td>Values</td><td>0.55</td><td>0.71</td><td>0.89</td><td>1.00</td><td>0.93</td><td>0.89</td></tr></table>

The variant without a window sets $W { = } 0$ in uptraining and serving, goes through the same two phases and stores the same latent of 72 KiB per token. It loses 8.1 points on MATH500, 7.9 on MMLU-Pro, 8.4 on BBH and 11.5 on RULER at 16K (Table 4). The window costs 96 MiB per sequence, as much as the latent of 1,365 tokens, and recovers most of the gap to the full cache. Splitting the divergence of the model without a window by the distance to the attended token shows where this gap arises (Fig. 4, Appendix D). Tokens at distances 1 to 127 hold 26.6% of the original attention but 48.7% of the divergence. Per unit of attention, the divergence is 0.040 and 0.049 nats at distances 1–31 and 32–127 and 0.009 beyond 1,024, so a window of 128 tokens replaces the latent where it is least accurate.

(a) Share of total by distance (%)

![](images/e525f6b942c99f32d955fb063f49a0805855f562a893e88964a160eee2ba9eed.jpg)

(b) KL per unit of attention  
![](images/c1efdea2be5c9005573bf707f7b1bcd75fcd5cebdbb5693248d978a1208ce5b1.jpg)  
Figure 4: Attention divergence of HLA without a window (W=0) on Ouro-1.4B, by distance between the query and the attended token. (a) How the original attention (top) and the KL divergence to it (bottom) are distributed over six ranges of distance; each bar sums to 100%. Orange ranges lie within the window of W=128 tokens used in the main experiments, blue ranges beyond it. (b) Divergence per unit of original attention in each range.

Table 4: Effect of the exact window on Ouro-1.4B with frozen weights. Both variants are uptrained with the full recipe and store a latent of 72 KiB per token. RULER uses the original settings, unlike Table 1 (Appendix H).
<table><tr><td rowspan="2">W</td><td colspan="2">RULER</td><td colspan="2"></td></tr><tr><td>4K 8K</td><td></td><td>16K MMLU-Pro BBH MATH500</td><td></td></tr><tr><td>0</td><td>78.3 71.9 57.6</td><td>39.8</td><td>61.0</td><td>66.0</td></tr><tr><td>128</td><td>83.2 78.7 69.1</td><td>47.7</td><td>69.4</td><td>74.1</td></tr></table>

Shifted positions were ablated on an earlier model with a window of 32 tokens, uptrained for 600 firstphase steps and then continued for 100 steps twice, with and without them (Appendix F). To change only the distance, we move the last 256 queries of a 2,048-token sequence up to 14K positions later, which lengthens every distance between a query and a key and leaves the hidden states unchanged. Without shifted positions the divergence from the full cache rises from 0.022 to 0.098 at a shift of 4K, and with them only to 0.026 (Fig. 5a). On RULER, where the distances come from real long prompts, shifted positions raise accuracy from 71.2 to 73.7 at 8K and from 54.3 to 62.7 at 16K (Fig. 5b), a gain that grows with the context length.

The long-document phase was evaluated on an earlier model with a window of 128 tokens: under the original RULER settings (Appendix H), its accuracy at 16K rose from 69.6 before the phase to 71.2 after it.

![](images/8a8334ba247560ed16949c2dbda590f8544cc60c2370633e48341fb1d4bcce12.jpg)  
Figure 5: Shifted positions on an earlier HLA model with W=32 on Ouro-1.4B, continued for 100 steps without (green) and with (blue) them. (a) KL divergence between the attention of HLA and of the full cache when the queries of a 2,048-token sequence are moved later by the given number of positions; thick lines average three RULER tasks, thin lines show each task. (b) RULER accuracy, with the full cache in orange.

## 6 Limitations and Future Work

With the pretrained weights frozen, HLA does not fully match the full cache, and the gap grows with context length. On RULER, Ouro-1.4B loses 0.8 points at 4K and 3.1 at 16K, where it keeps 96.0% of its original accuracy (Table 1). Contexts beyond 16K, where the smaller cache saves the most memory, are untested, and the gap may widen there. HLA is fitted to a model that was pretrained with the full cache, so this gap measures how well new linear maps can imitate an existing model, and it need not be the limit of the design. Pretraining a looped model with HLA from the start, so that its hidden states adapt to the latent, would separate the two.

HLA speeds up generation only, and a single sequence gains only beyond a few thousand tokens of context. Prompts are processed with exact attention, so the prompt pass costs the same as with the full cache. Each decoding step carries a fixed cost that the full cache does not have: a single sequence with 1K tokens of context decodes 26% slower on Ouro-1.4B, and the two caches break even near 4K, although the larger batch already gives 2.5 times the throughput at 1K (Section 5.2). We attribute this cost to applying the maps and to attending to the window and the latent separately, which a kernel that merges the two could reduce.

Our evidence covers one family of looped models with T=4 loops at two sizes. The main latent has the same size for any number of loops while the full cache grows with T, so the savings would grow with more loops, but whether a latent of this size can serve them is untested. The window and the latent sizes were chosen on Ouro-1.4B (Appendix G) and reused for Ouro-2.6B without tuning. Fine-tuning was tested on math reasoning only, and it runs two forward passes per step where standard fine-tuning runs one. HLA changes what is stored per token, so it is complementary to methods that lower the precision of the cache or keep fewer tokens; we have not evaluated such combinations.

## 7 Conclusion

A looped language model caches the keys and values of every loop, and this cache limits how many sequences a GPU decodes together. Hybrid Latent Attention stores each older token as compact latents that every loop reads directly through its own linear maps, and keeps exact keys and values only for the most recent tokens. Uptrained on Ouro-1.4B and Ouro-2.6B with frozen weights, it shrinks the cache by 10.7× per token and raises peak decoding throughput by up to 7.4× while retaining over 96% of the original accuracy, and it remains trainable. At the context lengths we test, a latent that each loop reads directly can thus replace most of the per-loop cache without reconstructing keys or values.

## References

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Shaojie Bai, J. Zico Kolter, and Vladlen Koltun. Deep equilibrium models. In Advances in Neural Information Processing Systems, 2019.

Mislav Balunovic, Jasper Dekoninck, Ivo Petrov, Nikola Jovanovi´ c, and Martin Vechev. MathArena:´ Evaluating LLMs on uncontaminated math competitions. arXiv preprint arXiv:2505.23281, 2025.

Stella Biderman, Hailey Schoelkopf, Lintang Sutawika, Leo Gao, Jonathan Tow, Baber Abbasi, Alham Fikri Aji, Pawan Sasanka Ammanamanchi, Sidney Black, Jordan Clive, Anthony DiPofi, Julen Etxaniz, Benjamin Fattori, Jessica Zosa Forde, Charles Foster, Jeffrey Hsu, Mimansa Jaiswal, Wilson Y. Lee, Haonan Li, Charles Lovering, Niklas Muennighoff, Ellie Pavlick, Jason Phang, Aviya Skowron, Samson Tan, Xiangru Tang, Kevin A. Wang, Genta Indra Winata, François Yvon, and Andy Zou. Lessons from the trenches on reproducible evaluation of language models. arXiv preprint arXiv:2405.14782, 2024.

Chi-Chih Chang, Wei-Cheng Lin, Chien-Yu Lin, Chong-Yan Chen, Yu-Fang Hu, Pei-Shuo Wang, Ning-Chi Huang, Luis Ceze, Mohamed S. Abdelfattah, and Kai-Chiang Wu. Palu: KV-cache compression with low-rank projection. In International Conference on Learning Representations, 2025.

DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. In International Conference on Learning Representations, 2019.

Tianyu Gao, Alexander Wettig, Howard Yen, and Danqi Chen. How to train long-context language models (effectively). In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025.

Angeliki Giannou, Shashank Rajput, Jy-yong Sohn, Kangwook Lee, Jason D. Lee, and Dimitris Papailiopoulos. Looped transformers as programmable computers. In International Conference on Machine Learning, 2023.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, 2021.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W. Mahoney, Yakun Sophia Shao, Kurt Keutzer, and Amir Gholami. KVQuant: Towards 10 million context length LLM inference with KV cache quantization. In Advances in Neural Information Processing Systems, 2024.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Hugging Face H4. AIME 2024. Hugging Face dataset, https://huggingface.co/datasets/ HuggingFaceH4/aime\_2024, 2024.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, 2023.

Zhenzhong Lan, Mingda Chen, Sebastian Goodman, Kevin Gimpel, Piyush Sharma, and Radu Soricut. ALBERT: A lite BERT for self-supervised learning of language representations. In International Conference on Learning Representations, 2020.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems, 2024.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023.

Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. MiniCache: KV cache compression in depth dimension for large language models. In Advances in Neural Information Processing Systems, 2024a.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In International Conference on Machine Learning, 2024b.

math-ai. AIME 2025. Hugging Face dataset, https://huggingface.co/datasets/math-ai/ aime25, 2025.

Fanxu Meng, Pingzhi Tang, Xiaojuan Tang, Zengwei Yao, Xing Sun, and Muhan Zhang. TransMLA: Multi-head latent attention is all you need. arXiv preprint arXiv:2502.07864, 2025.

James O’Neill and Fergal Reid. Looped latent attention: Cross-loop KV compression for looped transformers. arXiv preprint arXiv:2607.15456, 2026.

Open-R1. OpenR1-Math-220k. Hugging Face dataset, https://huggingface.co/datasets/ open-r1/OpenR1-Math-220k, 2025.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro Von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. In Advances in Neural Information Processing Systems, 2024.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings ofthe 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, 2016.

Noam Shazeer. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019.

Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. arXiv preprint arXiv:2104.09864, 2021.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. In Advances in Neural Information Processing Systems, 2024.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V. Le, Ed H. Chi, Denny Zhou, and Jason Wei. Challenging BIG-Bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, Tianle Li, Max Ku, Kai Wang, Alex Zhuang, Rongqi Fan, Xiang Yue, and Wenhu Chen. MMLU-Pro: A more robust and challenging multi-task language understanding benchmark. In Advances in Neural Information Processing Systems, 2024.

Haoyi Wu and Kewei Tu. Layer-condensed KV cache for efficient inference of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024.

Liu Yang, Kangwook Lee, Robert Nowak, and Dimitris Papailiopoulos. Looped transformers are better at learning learning algorithms. In International Conference on Learning Representations, 2024.

Wanqi Yang and Shiwei Liu. FlashLoop: Fast and memory-efficient looped transformers via lazy updates. arXiv preprint arXiv:2609.29812, 2026.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, 2018.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H<sub>2</sub>O: Heavyhitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems, 2023.

Dawei Zhu, Nan Yang, Liang Wang, Yifan Song, Wenhao Wu, Furu Wei, and Sujian Li. PoSE: Efficient context window extension of LLMs via positional skip-wise training. In International Conference on Learning Representations, 2024.

Rui-Jie Zhu, Zixuan Wang, Kai Hua, Tianyu Zhang, Ziniu Li, Haoran Que, Boyi Wei, Zixin Wen, Fan Yin, He Xing, et al. Scaling latent reasoning via looped language models. arXiv preprint arXiv:2510.25741, 2025.

## Appendix

## A Initialization of the Latent

This section details how the writer and the reader are initialized from principal components of the pretrained keys and values (Section 4.1), including the rotation $\tilde { R } _ { j }$ of Section 3.3 that keeps the key latent consistent with the rotary embeddings of the original model.

Frequency assignment. A head of dimension $d _ { h }$ has $d _ { h } / 2$ rotary frequencies, one for each pair of key coordinates. We write $k _ { j , t , h } ^ { f } \in \mathbb { R } ^ { 2 }$ for the pair of frequency f in the key of head h at loop $t ;$ the rotation $R _ { j }$ turns it into $\rho ( j \theta _ { f } ) k _ { j , t , h } ^ { f }$ , where $\rho ( \alpha )$ is the $2 \times 2$ rotation by $\alpha .$ The key latent has $r _ { k } / 2$ pairs, and the frequencies are assigned to them in round-robin order: pair $p ,$ counted from zero, receives frequency p mod $( d _ { h } / 2 )$ . Each frequency thus receives $m = r _ { k } / d _ { h }$ latent pairs, and $\tilde { R } _ { j }$ turns every one of them by $j \theta _ { f }$

Rotation commutes with same-frequency combinations. For one frequency $f ,$ we stack the key pairs of all heads and of loops 2 to $T$ as the rows of $K _ { j } ^ { f } \in \mathbb { R } ^ { ( T - 1 ) H \times 2 }$ . A combination matrix $M _ { f } \in \mathbb { R } ^ { m \times ( T - 1 ) H }$ maps them to the m latent pairs of that frequency, $C _ { j } ^ { f } = M _ { f } K _ { j } ^ { f }$ . Rotation acts on every row from the right and $M _ { f }$ mixes rows from the left, so

$$
\begin{array} { r } { M _ { f } \left( K _ { j } ^ { f } \rho ( j \theta _ { f } ) ^ { \top } \right) = \left( M _ { f } K _ { j } ^ { f } \right) \rho ( j \theta _ { f } ) ^ { \top } . } \end{array}\tag{6}
$$

Rotating the stored latent is therefore the same as writing it from rotated keys. A latent pair that mixed key pairs of different frequencies would have no single angle with this property, which is why every latent pair carries one frequency.

Scores at initialization. The writer and the reader are initialized from one matrix $M _ { f }$ with orthonormal rows per frequency. The rows of $E _ { t }$ for frequency f are the combinations, with the coefficients of $M _ { f }$ , of the rows of the key projection, so that the key part of the latent is $C _ { j } ^ { f } = M _ { f } K _ { j } ^ { f }$ . The map $A _ { t }$ of head h sends the query pair $q _ { i , t , h } ^ { f }$ to the m latent pairs of frequency f with the coefficients $M _ { f } [ : , ( t , h ) ]$ , the column of $M _ { f }$ that belongs to this head and loop. Summing over the m latent pairs, the contribution of frequency $f$ to the score ${ \hat { s } } _ { i j , \ l }$ <sub>t</sub> of Eq. (3) is, up to the factor $1 / \sqrt { d _ { h } }$

$$
\sum _ { ( t ^ { \prime } , h ^ { \prime } ) } \left[ P _ { f } \right] _ { ( t , h ) , ( t ^ { \prime } , h ^ { \prime } ) } \left( \rho ( i \theta _ { f } ) q _ { i , t , h } ^ { f } \right) ^ { \top } \left( \rho ( j \theta _ { f } ) k _ { j , t ^ { \prime } , h ^ { \prime } } ^ { f } \right) ,\tag{7}
$$

$$
P _ { f } = M _ { f } ^ { \top } M _ { f } ,
$$

where Eq. (6) moves the rotation from the latent to the keys. $P _ { f }$ is the orthogonal projection onto the m directions spanned by the rows of $M _ { f }$ . When $m = \bar { ( } T - \mathrm { \bar { 1 } } ) H , P _ { f }$ is the identity, and Eq. (7) summed over frequencies is the original score $s _ { i j , t }$ of Eq. (1). When $\ i < ( T - 1 ) \dot { H }$ , the key of head h at loop t is replaced by its row of $P _ { f } K _ { i } ^ { f }$ , the projection of the stacked keys onto m directions shared by all heads and loops of that frequency.

Choice of the directions. The rows of $M _ { f }$ are the m leading eigenvectors of the second-moment matrix $\begin{array} { r } { \Sigma _ { f } = \sum _ { j } K _ { j } ^ { f } ( K _ { j } ^ { f } ) ^ { \top } } \end{array}$ , accumulated over the tokens of 128 calibration sequences from the reference model. Among projections onto m directions, this choice minimizes the squared error $\begin{array} { r l } {  { \sum _ { j } \| K _ { j } ^ { f } - P _ { f } K _ { j } ^ { f } \| ^ { 2 } } } \end{array}$ on these tokens. Both coordinates of a pair enter $\Sigma _ { f }$ , so $\Sigma _ { f }$ is unchanged when every pair is rotated by the same angle, and the same projection is optimal at every position.

Values and loop 1. Values carry no rotation. The value part of the latent takes the $r _ { v }$ leading principal components of the values of loops 2 to $T$ concatenated across heads and loops, and $B _ { t }$ takes the block of these components that belongs to loop t. The loop-1 latent is built in the same way from the H key pairs of each frequency at loop 1, with $r _ { 1 } / d _ { h }$ latent pairs per frequency, and from the $r _ { 1 }$ leading principal components of the loop-1 values.

Instantiation. Both models have heads of dimension $d _ { h } = 1 2 8$ and hence 64 frequencies, $H { = } 1 6$ heads and $T { = } 4$ loops. With $r _ { k } { = } 5 1 2 .$ , each frequency receives $m { = } 4$ latent pairs out of $( T - 1 ) H = 4 8$ key pairs; with $r _ { 1 } { = } 2 5 6$ , the loop-1 latent keeps 2 of the $H { = } 1 6$ key pairs of each frequency.

## B Rotating the Whole Key Latent

The construction of Appendix A rotates every pair of the key latent. This section describes what changes after uptraining, why HLA rotates the whole latent rather than a part of it, and what this choice costs.

After uptraining. Uptraining updates $E _ { t } , A _ { t }$ and $B _ { t }$ as dense matrices, and only the assignment of frequencies to latent pairs stays fixed. A latent pair of frequency $f$ can then also carry key components of other frequencies, which $\tilde { R } _ { j }$ turns at the angle of f rather than at their own, so Eq. (7) no longer holds exactly. The scores still depend on the positions only through $i - j ,$ , because the query and the latent are rotated in the same space, but their dependence on distance may differ from that of the original model. Training with shifted positions keeps the learned scores accurate over long distances (Appendix F).

Why the whole latent is rotated. MLA rotates only a small part of each key, a rotary key shared by the heads, and caches the rest without position [DeepSeek-AI, 2024]; conversions of pretrained models to MLA adopt this split and drop the rotation from most key dimensions [Meng et al., 2025]. HLA rotates every latent pair instead. For frequency f, the original score depends on the positions only through $\rho ( i \bar { \theta _ { f } } ) ^ { \top } \rho ( j \bar { \theta _ { f } } ) = \rho ( ( j - i ) \theta _ { f } )$ . A position-free component replaces this rotation by the identity, which changes its contribution to the score by $q ^ { \top } \big ( \rho ( ( j - i ) \theta _ { f } ) - I \big ) k$ . This term vanishes only when $( j - i ) \theta _ { f }$ stays close to zero over the distances in use or when the component carries no energy, and it remains at full rank. The rotated latent has no such term: by Eq. (7), its only approximation at initialization is the projection $P _ { f }$

In a looped model, the rotary part must also serve the keys of every loop. A rotary key computed from one hidden state, as in MLA, approximates the keys of the other loops, which differ from it (Table 3), and one rotary key per loop makes this part of the cache grow with T. A rotary key that combines the loops through Eq. (6) avoids both, and it is a rotated latent of smaller size. The two designs thus differ only in whether part of the key budget drops the rotation, and HLA keeps the rotation on all of it.

Limitations of the rotated latent. This argument concerns the initialization; we have not compared the two designs after uptraining. Rotating every pair also has two costs. Every frequency receives the same number of latent pairs, whereas the eigenvalues of $\Sigma _ { f }$ need not be spread evenly across frequencies, and a budget allocated per frequency by these eigenvalues could keep more of the original scores at the same rank. After uptraining, every latent pair carries a rotation whose alignment with the original frequencies is no longer guaranteed, so the dependence of the scores on distance has to be learned. Without shifted positions, the divergence from the original attention rises from 0.022 to 0.098 when the queries are moved 4K positions later (Appendix F). A position-free part of the latent would be insensitive to distance by construction. A comparison at equal cache size between the rotated latent and a split into a rotary key and a position-free latent, and an allocation of pairs by frequency energy, are left to future work.

## C Training Details

Uptraining. Both phases use AdamW with $\beta = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.01 and gradient clipping at 1.0, with a batch of 128 sequences and a cosine schedule that decays the learning rate to one tenth of its peak. The first phase uses a peak learning rate of $1 0 ^ { - 3 }$ with 50 warmup steps, and the second $3 \times 1 0 ^ { - 4 }$ with 10 warmup steps. Sequences of the first phase have 64 to 2,048 tokens and mix math reasoning traces from OpenR1-Math-220k [Open-R1, 2025] and web text from the 10B-token sample of FineWeb-Edu [Penedo et al., 2024] at a ratio of 3:2. The long documents of the second phase are windows of 16,384 tokens cut from the book and repository-level code subsets of the ProLong data [Gao et al., 2025], 983 from books and 967 from code. The loss on each of them is computed on 1,024 query positions.

## D Where the Latent Errs

This section details the measurement of Fig. 4, which motivates the exact window (Section 3.4).

![](images/2bfb0b9eb8c3df7e59e6f60da2e0982750cffac30e095eac0834802f00ffa87f.jpg)  
Figure 6: Uptraining curves on held-out data for Ouro-1.4B, drawn as in Fig. 3.

Setup. We use the HLA model without a window of Table 4, uptrained on Ouro-1.4B with the full recipe. The measurement takes 16 held-out math documents of 2,048 tokens and 32 evenly spaced query positions in each. The original model is run on each document, and at every layer and loop we compute two attention distributions from its hidden states at 32-bit precision: the original one, $a _ { i \cdot , t } ^ { \star } ,$ and that of $\mathrm { H L A } , \hat { \boldsymbol { a } } _ { i \cdot , t }$ . Both start from the same hidden states, so their difference is the error of a single attention step, separate from errors that accumulate over layers.

Split by distance. The KL divergence of one query is a sum over the tokens it attends to,

$$
\begin{array} { r l } & { \mathrm { K L } \big ( a _ { i \cdot , t } ^ { \star } \big \| \hat { a } _ { i \cdot , t } \big ) } \\ & { = \displaystyle \sum _ { j \leq i } \bigg [ a _ { i j , t } ^ { \star } \log \frac { a _ { i j , t } ^ { \star } } { \hat { a } _ { i j , t } } - a _ { i j , t } ^ { \star } + \hat { a } _ { i j , t } \bigg ] , } \end{array}\tag{8}
$$

and every term of this sum is nonnegative. We assign the term of token $j$ to the distance $i - j$ , and add up the terms and the original attention $a _ { i j , t } ^ { \star }$ within six ranges of distance, over all queries, heads, layers and loops.

Result. Figure 4 shows the two sums per range of distance, and their ratio. Beyond the numbers given in Section 5.5, tokens at distances of 1,024 and more hold 29.5% of the attention and 11.0% of the divergence. The pattern is the same in every loop, where distances 1 to 127 account for 47% to 51% of the divergence.

## E Two-Pass Fine-Tuning

This section supports the two claims made about the procedure of Section 4.2.

Repetition reaches sequential decoding. Consider a training sequence whose first p tokens are the prompt. In a parallel pass under Eq. (4), token i reads the exact keys and values of the tokens in its window, the loop-1 latents of earlier tokens, and their main latents. The first two are produced in the same pass, in the order in which decoding produces them, because a loop-1 latent depends only on the hidden state of its own loop. Only the main latents are taken from the previous pass. The hidden states of token i therefore equal those of sequential decoding once the main latents of all tokens $j < i - W$ do. Prompt tokens attend exactly in training and in decoding, so their latents are already those of decoding after the first pass. The second pass is then exact for the first W+1 response tokens, which read main latents of prompt tokens only, and each repetition extends the exact range by another W+1 tokens. A response of m tokens is thus reproduced exactly after $\lceil m / ( W + 1 ) \rceil$ passes under the decoding rule.

Two passes are already close. The bound above counts the passes needed for an exact match. To see how close fewer passes come, we take the uptrained HLA models of the main experiments and run 32 fine-tuning validation examples, with 47,933 response tokens, through each procedure and through sequential decoding under teacher forcing. For each response token we compute the KL divergence from the next-token distribution of sequential decoding to that of the procedure, and average it over tokens (Table 5). At 32-bit precision, a single pass with exact attention differs from decoding by $5 . 6 \times 1 0 ^ { - 3 }$ on Ouro-1.4B and $3 . 8 \times 1 0 ^ { - 3 }$ on Ouro-2.6B. The two-pass procedure lowers this by a factor of 16 to 19, and the first repetition by roughly another factor of ten. At the 16-bit precision used for training, the two-pass procedure differs by $5 . 8 \times 1 0 ^ { - 4 }$ and $4 . 0 \times 1 0 ^ { - 4 }$ , and one repetition only halves this, to about the level that 32-bit runs reach without it. At this precision, rounding likely accounts for much of the remaining difference, so a repetition gains less. Under the two-pass procedure the cross-entropy of the responses moves by at most $3 \times 1 0 ^ { - 4 }$ nats at either scale and precision.

Table 5: KL divergence $( \times 1 0 ^ { - 4 } )$ from sequential decoding to each training procedure, averaged over the 47,933 response tokens of 32 fine-tuning validation examples, for the HLA models of the main experiments $\left( W { = } 1 2 8 \right)$ Two repetitions were measured at 32-bit precision only.
<table><tr><td rowspan="2">Procedure</td><td colspan="2">Ouro-1.4B</td><td colspan="2">Ouro-2.6B</td></tr><tr><td>32-bit</td><td>16-bit</td><td>32-bit</td><td>16-bit</td></tr><tr><td>Single exact pass</td><td>56</td><td>58</td><td>38</td><td>40</td></tr><tr><td>Two passes</td><td>3.4</td><td>5.8</td><td>2.0</td><td>4.0</td></tr><tr><td>One repetition</td><td>0.29</td><td>2.5</td><td>0.19</td><td>2.0</td></tr><tr><td>Two repetitions</td><td>0.05</td><td>一</td><td>0.02</td><td>一</td></tr></table>

Settings. Repetitions are run without gradients, and the latents they produce keep the gradient of the first pass. Ouro-1.4B is fine-tuned with one repetition. For $\mathrm { O u r o } { - } \dot { 2 } . \bar { 6 } \mathrm { B }$ , a repetition would lower the divergence at the 16-bit training precision only from $4 . 0 \times 1 0 ^ { - 4 } \mathrm { t o } 2 . 0 \times \mathrm { \dot { 1 } 0 ^ { - 4 } }$ (Table 5), and the deeper model is fine-tuned without it to limit training time.

## F Shifted Positions

This section details the ablation of shifted positions in Section 5.5. It was run on an earlier configuration of Ouro-1.4B with a window of 32 tokens.

Setup. The starting point is an HLA model with the latent sizes of the main experiments and W=32, uptrained for 600 steps on the first-phase data without shifted positions. We continue it twice under the same settings, with a peak learning rate of $3 \times 1 0 ^ { - 4 }$ : once with shifted positions as described in Section 4.1, and once without. Both continuations are evaluated after 100 steps.

Attention under a position shift. The first measurement changes the distance between queries and keys while keeping their content fixed. We take the hidden states of the original model on the first 2,048 tokens of 24 RULER prompts, eight from each of three tasks, and move the last 256 queries D tokens later while the keys stay in place. Every distance between a query and a key then grows by $D ,$ and no query, key or latent changes. Figure 5(a) shows the KL divergence between the attention of the full cache and that of HLA over the tokens at least 128 positions before the query, averaged over layers and loops. The two continuations agree at $D { = } 0 ,$ , with a divergence of 0.022. Without shifted positions the divergence rises to 0.098 at $\bar { D } { = } 4 \mathrm { { K } }$ and stays near 0.09 up to 14K, whereas with shifted positions it is 0.026 at 4K and 0.032 at 14K. The error of the model trained on short sequences thus comes from distances it has not seen, and shifted positions remove most of it without longer inputs.

RULER. Figure 5(b) compares the two continuations on RULER. Shifted positions raise the average from 81.5 to 82.3 at 4K, from 71.2 to 73.7 at 8K and from 54.3 to 62.7 at 16K, so the gain grows with the context length, as the first measurement predicts. On the eleven retrieval and aggregation tasks, the average rises from 58.7 to 72.8 at 16K.

## G Latent and Window Sizes

The sizes $r _ { k } { = } r _ { v } { = } 5 1 2 , r _ { 1 } { = } 2 5 6$ and W=128 were chosen on Ouro-1.4B and reused for Ouro-2.6B.   
This section reports the measurements behind this choice.

Latent size at initialization. The initialization of Appendix A can be evaluated before any training. Table 6 varies one rank at a time and reports the error on 64 held-out sequences. The attention divergence of loops 2 to 4 depends on the key rank only, and the error of the attention output on the value rank only, so the two ranks can be set separately. All three errors fall steadily with the rank and show no point beyond which a larger latent stops helping. The errors are measured without a window, and uptraining reduces them by more than an order of magnitude (Section 5.3), so this table shows what each rank controls and not the accuracy it leads to.

Table 6: Error of the initialized latent on Ouro-1.4B when one rank varies and the others stay at ${ r _ { k } } \mathrm { { = } } r _ { v } \mathrm { { = } } 5 1 2$ $r _ { 1 } = 2 5 6$ . KL: divergence between the attention of HLA and of the full cache, in nats, averaged over the loops the latent serves. Output error: relative squared error of the attention output, averaged over loops 2–4.
<table><tr><td>Rank</td><td>Key rank  $r _ { k }$  KL, loops 2–4</td><td>Value rank  $r _ { v }$  Output error</td><td>Loop-1 rank  $r _ { 1 }$  KL, loop 1</td></tr><tr><td>128</td><td>1.04</td><td>0.42</td><td>1.04</td></tr><tr><td>256</td><td>0.51</td><td>0.37</td><td>0.44</td></tr><tr><td>512</td><td>0.29</td><td>0.30</td><td>0.23</td></tr><tr><td>1,024</td><td>0.11</td><td>0.22</td><td>0.08</td></tr><tr><td>2,048</td><td>0.03</td><td>0.16</td><td></td></tr></table>

Latent size after uptraining. We then uptrained models of different sizes with the first phase, without shifted positions, and evaluated them on MATH500 with one sample per problem, where one standard error is about 2 points. Table 7 gives the best checkpoint of each run. Halving every rank to 256 loses about 15 points. Doubling the key rank, or both ranks, to 1,024 changes accuracy by at most 3.4 points while the cache grows by a quarter. A window of 32 tokens on the 72 KiB latent reaches 71.8, above the model without a window whose latent is 1.7 times as large. With this window, a loop-1 latent of rank 512 gives 70.6 against 71.8 at rank 256, a difference within one standard error. We therefore kept ${ r _ { k } } \mathrm { { = } } r _ { v } \mathrm { { = } } 5 1 2$ and $r _ { 1 } { = } 2 5 6$ and spent the remaining memory on the window.

Table 7: MATH500 accuracy (%) of Ouro-1.4B after the first uptraining phase for different latent sizes, with one sample per problem.
<table><tr><td> $r _ { k }$ </td><td> $r _ { v }$ </td><td>r1</td><td>Cache per token (KiB) MATH500</td></tr><tr><td colspan="5">Without a window 256 256 256 48 51.4</td></tr><tr><td>512 1,024</td><td>512 512</td><td>512 256</td><td>96 96 120</td><td>66.8 66.6</td></tr><tr><td colspan="5">1,024 1,024 256</td></tr><tr><td colspan="5">With a window of 32 tokens</td></tr><tr><td>512</td><td>512</td><td>256</td><td>72</td><td>71.8</td></tr><tr><td>512</td><td>512</td><td>512</td><td>96</td><td>70.6</td></tr></table>

Window size. A window of 32 tokens was our first choice, and the model of Appendix F uses it. Two observations led us to 128. The divergence per unit of attention in Fig. 4 is 0.049 nats at distances 32 to 127 against 0.040 at distances 1 to 31, so a window of 32 tokens leaves to the latent the range where it is least accurate. On RULER at 16K, the model with W=32 also repeated the instruction that ends a question-answering prompt in 27 of 100 HotpotQA examples, against 1 for the full cache; the repeated segment starts a median of 54 to 61 tokens before the end of the prompt, outside a window of 32 tokens and inside one of 128. A window of 128 tokens covers both and costs 96 MiB per sequence on Ouro-1.4B.

Math and long-context retrieval. MATH500 separates these configurations much less than RULER does. The model with W=32 scores 71.8, 71.6, 71.4, 70.0, 70.8 and 71.2 on MATH500 after 100 to 600 uptraining steps, so it reaches its level within the first 100 steps, about 4 points below the 75.9 of the full cache. After 600 steps its RULER accuracy is 81.5, 70.3 and 54.2 at 4K, 8K and 16K, against 85.2, 80.7 and 74.7 for the full cache. The same model thus keeps about 95% of the original accuracy on math and 73% on retrieval at 16K. Shifted positions raise its accuracy at 16K to 62.7 (Appendix F), and the main configuration reaches 69.1 under the same settings (Appendix H).

## H Evaluation Details

Question answering in RULER. Two of the 13 RULER tasks, built from SQuAD [Rajpurkar et al., 2016] and HotpotQA [Yang et al., 2018], ask a question about the given documents; lm-evaluationharness ends their prompt with “Answer:” and generates at most 32 tokens. The other 11 tasks end their prompt with the beginning of the answer sentence, so the model writes the answer first. After the question-answering prompt, Ouro often starts its reply by reasoning about the question instead, and the 32 tokens end before it reaches an answer. On Ouro-1.4B at 16K, 52% of the SQuAD replies of the full cache start this way, and none of them contains the answer. The score of a cache then depends on how often it leads the model into such a reply. The LLA decoder never does so at 16K and scores 60.4 on SQuAD under these settings against 31.8 for the full cache, although on the prompts where the full cache answers directly the two reach 69.6 and 66.1. For Table 1 we therefore place each question-answering prompt in the chat template of Ouro, with its default system message, the prompt as the user turn and the assistant header after it, and generate up to 2,048 tokens until the end of the turn. The answer is scored with the string match of RULER over the whole reply; at most 12% of the replies of the full cache, LLA and HLA reach the token limit, against 36% for single-loop sharing on Ouro-1.4B. Each RULER entry of Table 1 averages the other 11 tasks under their original settings with these two tasks. With the original settings for all 13 tasks, the full cache scores 85.2, 80.7 and 74.7 at 4K, 8K and 16K on Ouro-1.4B and HLA 83.2, 78.7 and 69.1. Every other RULER result in the paper uses the original settings (Table 4, Fig. 5, the long-document comparison in Section 5.5, and Appendices G and I) and is compared only with results under the same settings.

LLA. The accuracy of LLA in Table 1 uses a codec with 96 latent dimensions per head for keys and 160 for values, distilled as its authors describe. Its projections start from the top singular vectors of the keys and values of all loops and are trained with the pretrained weights frozen, on the KL divergence of the output distribution when every key and value is replaced by its reconstruction plus a squared error on the attention outputs weighted by 0.5. We train for 1,000 steps on the data of our first phase and keep the checkpoint with the lowest held-out divergence, 0.017 for Ouro-1.4B and 0.011 for Ouro-2.6B. Decoding follows the two-pass decoder of LLA: each step first runs the original model on the new token and writes the latent of its keys and values, then recomputes the step with every key and value, those of the new token included, rebuilt from the latent, and samples from thi second pass. We implement this decoder in vLLM and compare it with a PyTorch implementation on 8 MATH500 prompts with 256 greedy tokens each, where the two choose the same token at 99.6% of the positions. The implementation also keeps the exact cache for the first pass, so it reproduces the accuracy of LLA but not its memory, and we use it only for Table 1.

## I Serving Implementation

HLA runs in vLLM 0.26 as a model class for Ouro that uses the engine’s paged cache, scheduler and attention kernels; the engine is unchanged apart from logging of preemptions, which the measurements use. This section describes how the cache and one decoding step are mapped onto it.

Cache layout. Each layer declares three kinds of cache to the engine. The main latent is stored as a cache with a single key–value head of width $r _ { k }$ , whose key half holds $c _ { j } ^ { K }$ and whose value half holds $c _ { j } ^ { V }$ , and the loop-1 latent as a second cache of width $r _ { 1 }$ . All H heads of the layer read these two caches, as the heads of a group read one key–value head in grouped-query attention. The exact window uses one ordinary per-head cache for each loop, declared with a sliding window of W+1 tokens, so the engine frees the blocks that fall out of the window and the memory of this cache does not grow with the sequence. The engine allocates the three kinds from one pool of blocks.

One decoding step. The T loops of a token run inside one forward pass of the model. In each layer and loop, the query is mapped by $A _ { t }$ and rotated with the latent rotation, and a paged decoding kernel scores it against the latent keys of the tokens before the window and returns the weighted sum of their value latents together with the log-sum-exp of the scores. A paged FlashAttention call reads the exact keys and values of the window and of the current token in the same way. The two partial results are merged with their log-sum-exp, which is equal to the single softmax of Eq. (4), after $B _ { t }$ has mapped the latent part back to the head dimension. The writer accumulates its sum over the loops within the forward pass and stores the loop-1 latent after the first loop and the main latent after the last, with the key part rotated once at that point. A decoding step contains no synchronization with the host and no loop over requests, so it is captured and replayed as a CUDA graph like the step of the original model.

Prompts. A prompt is processed in one pass with exact attention over its own tokens, and its latents and window are written from that pass. Prefix caching and chunked prefill are disabled for both HLA and the full cache: with them, whether a prompt token is read exactly or through its latent would depend on how the scheduler splits the prompt.

Capacity. The capacity that the engine reports at startup budgets the sliding-window caches pessimistically and underestimates how many sequences fit. The concurrent sequences of Table 8 are therefore measured: we increase the batch until the scheduler no longer runs all its sequences together, and count a batch only if none of its sequences was preempted.

Agreement with the reference implementation. Uptraining and fine-tuning use a separate PyTorch implementation of Eq. (4). We checked the serving path against it on RULER with the W=32 starting model of Appendix F: the two implementations reach 81.5, 70.3 and 54.2 against 81.6, 70.4 and 54.2 at 4K, 8K and 16K.

## J Decoding Measurements

Table 8 gives the measurements plotted in Fig. 2.

Table 8: Values of Fig. 2, measured with the same settings. LLA uses its reconstruct path with a latent of rank 256 per head, 96 for the keys and 160 for the values; since the codec does not affect the timing, it is a per-head PCA codec for Ouro-1.4B and a random one for Ouro-2.6B. Ratios compare HLA with the full cache; the best value of each group is in bold.
<table><tr><td colspan="2"></td><td colspan="4">Concurrent sequences</td><td colspan="4">Throughput (tokens/s)</td><td colspan="3">Latency (ms/token)</td></tr><tr><td>Model</td><td>Context</td><td>Full</td><td>LLA</td><td>HLA</td><td>Ratio</td><td>Full</td><td>LLA</td><td>HLA</td><td>Ratio</td><td>Full</td><td>LLA</td><td>HLA</td></tr><tr><td rowspan="4">Ouro-1.4B</td><td>1K</td><td>68</td><td>264</td><td>298</td><td>4.4×</td><td>1,305</td><td>93.6</td><td>3,231</td><td>2.5×</td><td>13.6</td><td>28.5</td><td>17.1</td></tr><tr><td>4K</td><td>20</td><td>77</td><td>139</td><td>7.0×</td><td>373</td><td>27.8</td><td>1,429</td><td>3.8×</td><td>20.6</td><td>54.1</td><td>20.8</td></tr><tr><td>8K</td><td>10</td><td>40</td><td>84</td><td>8.4×</td><td>192</td><td>14.3</td><td>797</td><td>4.2×</td><td>30.1</td><td>90.1</td><td>24.2</td></tr><tr><td>16K</td><td>5</td><td>19</td><td>40</td><td>8.0×</td><td>97</td><td>7.2</td><td>428</td><td>4.4×</td><td>49.3</td><td>158.7</td><td>25.1</td></tr><tr><td rowspan="4">Ouro-2.6B</td><td>1K</td><td>32</td><td>123</td><td>127</td><td>4.0×</td><td>519</td><td>46.3</td><td>1,278</td><td>2.5×</td><td>27.0</td><td>56.8</td><td>34.3</td></tr><tr><td>4K</td><td>9</td><td>36</td><td>68</td><td>7.6×</td><td>147</td><td>13.7</td><td>593</td><td>4.0×</td><td>41.0</td><td>107.5</td><td>41.5</td></tr><tr><td>8K</td><td>4</td><td>18</td><td>35</td><td>8.8×</td><td>66</td><td>7.0</td><td>327</td><td>5.0×</td><td>59.9</td><td>181.8</td><td>48.1</td></tr><tr><td>16K</td><td>2</td><td>8</td><td>13</td><td>6.5×</td><td>21</td><td>3.6</td><td>151</td><td>7.4×</td><td>98.0</td><td>322.6</td><td>50.0</td></tr></table>