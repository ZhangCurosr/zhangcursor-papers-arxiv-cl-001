# COARSE SEMANTIC BLOCKS FOR EFFICIENT TRAINING-FREE SPARSE ATTENTION

Sosuke Kobayashi, Shoichiro Yamaguchi, Seiya Tokui

Preferred Networks

{sosk,guguchi,tokui}@preferred.jp

## ABSTRACT

We introduce a training-free sparse attention method that builds a prompt-specific key index to accelerate both prefill and decoding. The method clusters the keys into large semantic blocks that group tokens from noncontiguous positions. Each query position selects blocks by scoring their centroids and attends to the original keys and values within them. Queries that select the same block are processed together, so prefill stays fast even though each query chooses its own blocks. Large blocks make centroid scoring and this grouped processing cheap, so each query can attend to more tokens in the same time. Larger blocks with a larger token budget can thus be both faster and more accurate on long-context tasks than smaller blocks with a smaller budget. We evaluate two hybrid LLMs, Qwen3.5-4B and 35B-A3B, on one NVIDIA H200. Including the cost of building and using the index, whole-model wall-clock speedups over dense attention are 2.3–2.4× for prefill and 1.7–2.0× for decoding at 512K tokens, and 2.9–3.0× for prefill at 1M. The 35B-A3B model stays close to dense attention on five RULER tasks through 1M, on LongBench-v2 overall, and on MRCR through 256K. The 4B model loses accuracy mainly on multi-key retrieval, yet outperforms dense Qwen3.5-2B, whose latency is similar at 512K, on all three benchmarks.

## 1 INTRODUCTION

Long contexts make attention expensive during both prefill and decoding. Sparse attention restricts each query to a subset of tokens, but it is faster only if choosing the tokens and preparing their keys and values costs less than the attention computation it saves. Methods with learned indexers, such as DSA (DeepSeek-AI, 2025), MSA (Lai et al., 2026), and QSA (Qiu et al., 2026), train the indexer and the model so that a small token budget suffices. For training-free methods, estimating token importance separately for each query during prefill is costly, and irregular selections are difficult to execute efficiently on GPUs (Nawrot et al., 2026). Existing methods therefore select contiguous blocks (Tang et al., 2024), whose tokens are often unrelated to one another, or share one selection across neighboring queries (Jiang et al., 2024; Lai et al., 2025; Jones et al., 2026; Fan et al., 2026a). Other methods cluster similar keys during dense prefill and use the resulting index only for decoding (Liu et al., 2025b; Zhu et al., 2026).

We show that, without training, a sparse prefill in which each query selects its own blocks can still run much faster than dense attention. Our method clusters the keys of each input prompt into large semantic blocks, groups of tokens with similar keys regardless of their positions in the sequence. Each query scores the cluster centroids and attends to the original keys and values of the blocks it selects. Unlike contiguous blocks, a semantic block gathers tokens that a query is likely to need together. We use the same index for both the prompt’s own prefill and decoding.

Scoring one centroid per block makes selection cheaper than scoring every key individually. During prefill, queries that select the same block are processed together for that block, as in the KV-outer execution of MoBA and MSA (Lu et al., 2025; Lai et al., 2026). This lets the GPU reuse KV data in on-chip memory across queries while preserving each query’s own block selection. Larger blocks make centroid scoring and block-wise attention cheaper but token selection less precise. The time saved leaves room for a larger token budget, which can compensate for the coarser selection (Section 5.3).

![](images/d2920f52885b220a62fd91880fbec3801285f7a29b43bc629dded7e2ee4d9903.jpg)  
Figure 1: Prefill and decode latency on one H200 with a token budget of L/16 for a prompt of L tokens. Whole-model latency is measured by wall clock, and full-attention layers (shaded) by GPU time. The rows are Qwen3.5-4B (top) and 35B-A3B (bottom). Dense (2B) is Qwen3.5-2B. Speedups are over dense attention (Tables 13 and 31).

We evaluate on hybrid models, which use full attention in only some layers and are increasingly common among top-tier LLMs (MiniMax, 2025; Qwen Team, 2026; Gemma Team, 2026). On Qwen3.5-4B and 35B-A3B, whole-model prefill is 2.3–2.4× faster and decoding 1.7–2.0× faster than dense attention at 512K tokens, and prefill is 2.9–3.0× faster at 1M, with index construction included and dense attention run on the same inference stack (Figure 1). The 35B-A3B model stays close to dense attention on five RULER tasks (Hsieh et al., 2024) through 1M, on LongBench-v2 (Bai et al., 2025) overall, and on MRCR (OpenAI, 2025) through 256K. The 4B model, which loses accuracy mainly on the multi-key tasks in RULER, still outperforms dense Qwen3.5-2B, whose latency is similar at 512K (Table 1).

## 2 RELATED WORK

Many sparse attention methods first score individual tokens or blocks of consecutive tokens, then compute attention over the selected keys and values. DSA (DeepSeek-AI, 2025) and MSA (Lai et al., 2026) use learned indexers to score every visible token, with MSA pooling the resulting scores to select blocks. QSA reduces the number of candidates by pooling indexer keys into 4-token blocks before scoring (Qiu et al., 2026). NSA selects blocks by the attention that queries give to compressed block keys (Yuan et al., 2025). These systems train their selection together with the language model. DSA, MSA, and QSA perform well with selected-token budgets of 2,048.

Quest (Tang et al., 2024) and MoBA (Lu et al., 2025) select blocks of consecutive tokens using summaries of the model’s existing keys. MoBA computes attention block by block, together for all queries that selected a block, as MSA also does (Lai et al., 2026). FlashMoBA speeds up this computation for small blocks (Xiao et al., 2025a). Training-free prefill methods make selection cheaper with structured patterns and block-level estimates, as in MInference (Jiang et al., 2024). FlexPrefill’s query-aware pattern scores blocks using pooled queries and keys (Lai et al., 2025), while QuoKA uses representative queries to select a shared KV subset for a chunk (Jones et al., 2026). FlashPrefill and FlashPrefill V2 score mean-pooled key blocks for each tile of queries and keep the blocks above a threshold relative to the maximum score, the latter compensating the rest with their mean keys and values (Fan et al., 2026c;a). Nawrot et al. (2026) compare training-free methods using FLOPs for prefill and memory traffic for decoding rather than measured latency. They also note that fine-grained per-query selection during prefill is difficult to accelerate because importance estimation is costly and available kernels struggle with irregular sparsity.

![](images/007a8ff2f5edfd20da7279324d433dd66e0a24ce39e32049e314090c13102a87.jpg)  
Figure 2: Keys are clustered separately for each layer and KV head. The query heads that share a KV head score the centroids separately $( r _ { h } )$ , and the sum of their scores (s) selects the clusters. Each head attends to the original KV of the selected clusters, together with recent tokens $\mathcal { D } _ { t }$ , under a causal mask. Only positions at or before the query contribute to the attention sum.

Clustering groups tokens by key similarity rather than sequence position. Routing Transformer trains a model whose queries and keys are routed by spherical k-means (Roy et al., 2021). SAAP fits a key partition on a corpus and trains a query classifier to select clusters while keeping the language model frozen (Mazare et al., 2025). Squeezed Attention (Hooper et al., 2025), ClusterKV (Liu et al.,´ 2025b), and Tactic (Zhu et al., 2026) use query–centroid scores without a learned classifier and attend to the selected original keys and values. Squeezed Attention indexes a reusable context offline, whereas ClusterKV and Tactic build prompt-specific clusters during dense prefill for subsequent sparse decoding. RetrievalAttention similarly searches a nearest-neighbor index of the keys during decoding (Liu et al., 2025a). Our method also builds its clusters for each prompt without training, but uses them for the prompt’s own prefill as well as its decoding. Its clusters are also much larger, 2,048 tokens in the main setting against tens of tokens in these methods, which makes centroid scoring and KV-outer attention cheap enough to afford a larger budget, with a separate selection for each query.

The inference-time methods above, which leave the language model unchanged, are evaluated in their own papers on Transformers in which every layer uses softmax attention (Appendix I). UniPrefill, which skips token computation in attention and non-attention layers, measures several of the prefill methods on a linear-attention hybrid up to 128K tokens, where they speed up the time to first token by at most 1.11× (Fan et al., 2026b). Concurrent work on Qwen3.5 trains a selector and keeps the prefill dense (Ahmad & Yun, 2026). When every layer uses softmax attention, most heads attend locally (Wu et al., 2025; Xiao et al., 2025b), and turning 85% of them into sliding-window attention, with decode-time sparse token selection in the rest, loses little accuracy after a few hundred training steps (Zhou et al., 2026). In hybrids such as MiniMax-M1 (MiniMax, 2025) or Qwen3.5 (Qwen Team, 2026), a quarter or fewer of the layers use full attention. These layers provide the model’s only exact access to distant tokens, so results on all-attention models do not show how much sparsity they can tolerate. They also account for only part of the model’s cost, so attention-layer speedups alone do not show the whole-model gain. We therefore report whole-model wall-clock latency and attention-layer GPU time separately, and break down the costs of indexing, selection, and attention execution.

## 3 METHOD

Our method groups prompt keys into semantic blocks, selects blocks for each query, and computes attention over the selected tokens (Figure 2). We choose the block size and token budget together to balance task accuracy and whole-model latency, including the cost of building and using the index. All model weights remain fixed, and no selector is trained.

We describe one full-attention layer and one KV head shared by a group H of query heads under grouped-query attention (GQA) (Ainslie et al., 2023). Let L be the prompt length and d the attention head dimension. We write $k _ { i } , v _ { i } \in \mathbb { R } ^ { d }$ for the key and value at position $i ,$ and $\bar { q _ { t , h } } \in \mathbb { R } ^ { d }$ for the query at position t in head $h \in \mathcal H$ . We omit layer and KV-head indices.

## 3.1 BUILDING SEMANTIC BLOCKS

Let $\tilde { k } _ { i }$ and $\tilde { q } _ { t , h }$ denote the keys and queries normalized to unit $L _ { 2 }$ norm. For each prompt, layer, and KV head, we run spherical k-means on a sample of the normalized keys to obtain C unit-norm centroids $\mu _ { 1 } , \ldots , \mu _ { C }$ . We assign each prompt key to its nearest centroid and define its block as

$$
a _ { i } = \underset { c \in \{ 1 , \ldots , C \} } { \arg \operatorname* { m a x } } \tilde { k } _ { i } ^ { \top } \mu _ { c } , \qquad \mathcal { B } _ { c } = \{ i \in \{ 1 , \ldots , L \} : a _ { i } = c \} .\tag{1}
$$

Each $\boldsymbol { B } _ { c }$ contains the positions of tokens assigned to cluster c. We call these groups semantic blocks because they group similar keys regardless of their positions in the sequence. Figure 3 and Appendix J show tokens that the clustering places in the same block. A block can contain many different tokens. In the upper example, the members closest to the centroid are words about the environment from distant passages. In the lower example, they are the leading digits of different identifiers.

A target mean block size u determines the cluster count approximately as $C = L / u$ (Appendix B gives the clustering details). Each layer and KV head has its own partition, and the members of a block are generally scattered across the prompt, so we store each block’s keys and values contiguously (Section 3.4).

MRCR · layer 3, KV head 2, cluster 17   
Cosine Position   
0.939 104256 … the shrinking habitats, and the urgent …   
0.937 83001 ... to maintaining ecosystems is needed. ...   
0.936 42982 … helps preserve biodiversity and the …   
0.685 231403 … Arabic numeral system revolutionized …   
0.683 161830 … during the 19th century, the croissant is …   
0.632 138508 ... in managing HIV/AIDS, The process of ...   
MK-3 · layer 7, KV head 0, cluster 147   
Cosine Position   
0.983 262904 … special magic uuids for 11ac61a0-1483 …   
0.982 174418 … special magic uuids for 2484c754-8d1 …   
0.982 353541 … special magic uuids for 1022e6ce-c52a- …   
0.660 523286 … special magic uuid for 45c4b52e-fcee-4 …   
0.545 77 … 34-87f9-13184e7bd9a1 is: 1f9dd22f-0d2b …   
0.422 40 … special magic uuids for 3c1b5987-b71a …  
Figure 3: Examples of two clusters, with members of high and low cosine similarity to the centroid.

## 3.2 SELECTING BLOCKS AND COMPUTING ATTENTION

Prefill queries are processed in contiguous chunks. Each query attends to all causally visible tokens in its current chunk and selects clusters from the earlier history. During decoding, recent generated tokens are similarly included without selection. Let ${ \mathcal { D } } _ { t } \subseteq \{ 1 , \dots , t \}$ denote the positions included without selection for the query at position t. The remaining visible positions in cluster c are $\boldsymbol { B } _ { c , t } ^ { \prime } = ( B _ { c } \cap \{ 1 , \ldots , t \} ) \setminus \mathcal { D } _ { t }$

We compute softmax scores over centroids within each query head and sum them across the GQA group before selecting blocks. We also apply a size penalty to account for the number of tokens retrieved from each cluster,

$$
r _ { h } ( c ) = \frac { \exp ( \tilde { q } _ { t , h } ^ { \top } \mu _ { c } / \tau ) } { \sum _ { c ^ { \prime } = 1 } ^ { C } \exp ( \tilde { q } _ { t , h } ^ { \top } \mu _ { c ^ { \prime } } / \tau ) } , \qquad s ( c ) = \frac { \sum _ { h \in \mathcal { H } } r _ { h } ( c ) } { ( 1 + | \mathcal { B } _ { c , t } ^ { \prime } | ) ^ { \beta } } .\tag{2}
$$

We use temperature $\tau = 0 . 1$ and size-penalty exponent $\beta = 0 . 2 5$ . We select clusters in decreasing order of $s ( c )$ , taking all tokens in each $B _ { c , t } ^ { \prime }$ until their total number reaches or exceeds the budget K. Let $\mathcal { T } _ { t }$ be the clusters selected at position t. The positions of the key–value pairs used for attention are

$$
S _ { t } = \mathcal { D } _ { t } \cup \bigcup _ { c \in \mathcal { T } _ { t } } B _ { c , t } ^ { \prime } .\tag{3}
$$

This selection is computed for each query position and shared by the query heads in its GQA group. Attention is then computed over these positions using the model’s original queries, keys, and values<sup>1</sup>,

$$
o _ { t , h } = \frac { \sum _ { i \in S _ { t } } \exp ( { q _ { t , h } ^ { \top } k _ { i } / \sqrt { d } } ) v _ { i } } { \sum _ { i \in S _ { t } } \exp ( { q _ { t , h } ^ { \top } k _ { i } / \sqrt { d } } ) } .\tag{4}
$$

Setting $S _ { t } = \{ 1 , \ldots , t \}$ recovers dense attention.

## 3.3 PREFILL AND DECODING

During prefill, we build the index from the prompt keys at each layer and process the prompt queries in contiguous chunks. Each query attends to selected blocks from earlier chunks and to all causally visible tokens in its own chunk. For queries near the beginning of the prompt, the earlier history is short, so we avoid selection overhead by using dense attention over the whole visible prefix, setting $\mathcal { D } _ { t } = \{ { 1 , . . . , t } \}$ (see Appendix B).

During decoding, each query uses the same index to select prompt blocks. In our main experiments, all generated tokens are included without selection in $\mathcal { D } _ { t }$ . For longer generations, older generated tokens can instead be added to the index, leaving only recent ones in $\mathcal { D } _ { t }$ (Section 5.5).

For simplicity, our main experiments run spherical k-means on keys sampled from the whole prompt. Block selection for a prompt query can therefore depend on later prompt keys, although each query attends only to positions at or before it. This is possible because the whole prompt is given before generation starts, and every generated token still depends only on the tokens before it. Because the index is fitted to keys drawn from the whole prompt, prefill runs one layer at a time over the whole prompt, instead of passing chunks of the prompt through all layers as ordinary chunked prefill does (Appendices B.4 and B.5). Alternatively, running k-means only on the first 64K keys and assigning each later key to its nearest centroid allows ordinary chunked prefill, with similar accuracy (Section 5.4 and Appendix B.5).

## 3.4 EFFICIENT EXECUTION

Efficient dense-attention kernels such as FlashAttention-3 (Shah et al., 2024) process neighboring queries together so that they can reuse each KV tile. In our method, however, neighboring queries can select different KV blocks. If they are still processed together, a block selected by one query may also be processed for queries that did not select it, reducing the savings from sparsity. We instead gather the queries that selected each KV block and compute their attention to that block together. MoBA and MSA use the same execution strategy for blocks of consecutive tokens (Lu et al., 2025; Lai et al., 2026).

For each prefill chunk and KV head, FlashAttention-3 processes a batch of KV blocks paired with their query groups. Attention within the current chunk uses causal FlashAttention-3. For each query, we combine the attention outputs using their softmax normalizers (Appendix B). Following MSA (Lai et al., 2026), we call this block-wise execution KV-outer. On Qwen3.5-4B at 512K, prefill with KV-outer is 2.6× faster than processing neighboring queries over their selected-block union (Appendix E.4).

Although a block’s tokens may be scattered across the prompt, we store their KV contiguously in their original sequence order. For each prefill chunk, we can then read the KV of earlier tokens directly from the start of each block. We build this block-ordered cache once per layer and use it for both prefill and decoding. Without this layout, gathering the selected KV for every prefill chunk increases 512K prefill time by about 9% (Appendix E.4).

During decoding, each step reads prompt KV from the selected blocks rather than the full prompt. This reduces the prompt tokens read per step from L to roughly K. Our vLLM implementation (Kwon et al., 2023) uses fused kernels to reduce the overhead of scoring and selecting blocks during decoding.

Each query scores C centroids instead of the prompt’s L keys. For a fixed prompt and token budget, smaller blocks give finer selection but require scoring more centroids and processing more blocks (Xiao et al., 2025a). Larger blocks reduce this overhead, so we can increase the budget to compensate for coarser selection (Section 5.3). In our experiments, both C and K grow in proportion to $\bar { L , }$ so prefill remains quadratic in prompt length, but with a smaller constant factor than dense attention. Appendix A gives the cost of each step.

## 4 EXPERIMENTAL SETUP

Models and baselines. We evaluate Qwen3.5-4B and Qwen3.5-35B-A3B (Qwen Team, 2026), replacing their full-attention layers (8 of 32 and 10 of 40) while keeping all weights fixed, and also apply the method to a hybrid of another family, Nemotron 3.5 Lightning (NVIDIA, 2026)

Table 1: Long-context performance at $K = L / 1 6 ,$ with L the upper bound of the length bin for MRCR-8 and of the length band for LongBench-v2, and whole-model prefill latency at 1M. MRCR-8 reports a match ratio, and the other quality scores are percentages. At 1M, the multi-value prompts of RULER-5 are about 647K tokens long (Appendix C.3). Appendix F.3 gives other budgets.
<table><tr><td></td><td colspan="3">RULER-5</td><td colspan="3">MRCR-8</td><td colspan="3">LongBench-v2</td><td>Prefill (s)</td></tr><tr><td></td><td>256K</td><td>512K</td><td>1M</td><td>128K</td><td>256K</td><td>512K</td><td>≤256K</td><td>&gt;256K</td><td>all</td><td>1M</td></tr><tr><td>35B-A3B, dense</td><td>85.7</td><td>84.4</td><td>73.2</td><td>0.846</td><td>0.655</td><td>0.564</td><td>64.4</td><td>66.2</td><td>64.7</td><td>173.5</td></tr><tr><td>35B-A3B, ours</td><td>84.5</td><td>84.2</td><td>73.9</td><td>0.844</td><td>0.668</td><td>0.504</td><td>63.1</td><td>64.7</td><td>63.3</td><td>57.4</td></tr><tr><td>4B, dense</td><td>84.7</td><td>82.8</td><td>63.2</td><td>0.797</td><td>0.574</td><td>0.412</td><td>54.3</td><td>64.2</td><td>55.7</td><td>144.1</td></tr><tr><td>4B, ours</td><td>82.7</td><td>81.8</td><td>55.2</td><td>0.788</td><td>0.561</td><td>0.399</td><td>53.9</td><td>57.7</td><td>54.5</td><td>49.3</td></tr><tr><td>2B, dense</td><td>77.7</td><td>66.5</td><td>48.9</td><td>0.726</td><td>0.414</td><td>0.198</td><td>36.7</td><td>34.3</td><td>36.4</td><td>56.2</td></tr></table>

(Appendix H.2). Dense attention runs on the same vLLM build with chunked prefill, and dense Qwen3.5-2B serves as a smaller-model reference. As a controlled sparse baseline, we replace semantic blocks with positional blocks of B consecutive tokens, each represented by its mean key. As in block-sparse prefill methods (Jiang et al., 2024; Lai et al., 2025; Fan et al., 2026a), g neighboring queries can share one block selection to reduce its cost. It uses the same token budget, the same $\mathcal { D } _ { t } ,$ and the same attention kernels as our method. This gives a controlled comparison with positional block selection and query sharing.

Benchmarks. We use RULER (Hsieh et al., 2024), eight-needle MRCR (OpenAI, 2025), and LongBench-v2 (Bai et al., 2025). RULER-5 averages five RULER tasks, namely multi-key retrieval (MK-2 and MK-3), multi-value retrieval, common-word extraction, and HotpotQA. Most other tasks are too easy to separate the methods at 512K, and the 4B model’s average over all thirteen stays close to dense attention (Appendix D.1). RULER and MRCR use about 100 examples per task and length, and LongBench-v2 uses the 470 of 503 examples that fit without truncation. RULER uses greedy decoding without thinking, while MRCR and LongBench-v2 use thinking with Qwen’s recommended sampling settings. We average our main results on all three benchmarks over three runs (Appendix C).

Settings. Both Qwen models use the same configuration, with target mean block size $u = 2 , 0 4 8$ and budget $K = L / 1 6$ for RULER and latency measurements. Most ablations use the 4B model. MRCR and LongBench-v2 have variable-length prompts, so we set budgets by length range. For MRCR, K is one sixteenth of the bin’s upper length bound. For LongBench-v2, $\bar { K = 1 6 , 3 8 4 }$ for prompts up to 256K and K = 65,536 for longer prompts. The budget excludes the tokens attended without selection and may be exceeded by the last selected block. Beyond the native 256K context, dense and sparse runs both use factor-4 YaRN interpolation (Peng et al., 2024) without its attention scaling. This setting gave the best dense-attention accuracy in our tests (Appendix D.2).

## 5 RESULTS

## 5.1 OVERALL PERFORMANCE

We measure latency at batch size one on a single H200 in BF16, with FlashAttention-3 for prefill and full-step CUDA graphs for decoding in both methods, and report the median of three repetitions after warm-up (Appendix C). Prefill latency includes index construction, KV packing, selection, gathering, and merging. On Qwen3.5-4B and 35B-A3B, whole-model prefill is 2.3–2.4× faster and decoding 1.7–2.0× faster than with dense attention at 512K (Figure 1). The gains grow with context length, reaching 2.9–3.0× for prefill and 2.3–2.9× for decoding at 1M. The full-attention layers themselves prefill 3.9–4.3× faster at these lengths, and the whole-model speedup is lower because the other layers are unchanged (Tables 13 and 31).

Figure 4 breaks down prefill time on Qwen3.5-4B at 512K. Our attention kernels run about 5× faster than dense attention. Index construction, selection, gathering, and merging together take about 1.8 s, of which fitting the centroids is a small part, compared with roughly 25 s saved in attention computation. The rest of the model is unchanged and takes about half of the sparse prefill time. We use dense attention within query chunks and near the start of the prompt, so an $L / 1 6$ budget does not give a 16× attention speedup. We also tried switching from dense to sparse attention earlier in the prompt, but this did not speed up prefill. The extra selection, gathering, and merging offset the time saved on dense attention (Appendices E.2 and E.3).

Both models retain most of their dense-attention performance across the three benchmarks (Table 1). On the 35B-A3B model, RULER-5 remains close to dense attention through 1M. The 4B model loses more on multi-key retrieval, especially at 1M, while multi-value retrieval and HotpotQA remain close to dense attention across lengths (Appendix D.1). MRCR-8 scores stay close to dense attention through 256K on both models and fall below it at 512K, more on the 35B-A3B model. Overall LongBench-v2 accuracy changes little on either model. A smaller dense model also reduces latency, but the sparse 4B model scores higher than dense Qwen3.5-2B on RULER-5 at every tested length, on MRCR-8, and on LongBench-v2, with similar prefill latency at 512K and lower latency at 1M (Table 1 and Appendix E). On Nemotron 3.5 Lightning, with

![](images/4a8c9481f98cf8c1d8850976e8dc9f94114f70cdebb8e032daa986eb6bf415bc.jpg)  
Figure 4: GPU-time breakdown for a 512K prefill on Qwen3.5-4B. The upper panel shows wholemodel GPU time, and the lower panel breaks down our attention-kernel time (Table 14). Times are profiler totals, not wall-clock latency.

its own clustering settings, multi-key accuracy falls further at $L / 1 6 ,$ so we use $K = L / 8$ . Wholemodel prefill is then 1.5–2.0× faster than dense attention over prompts from 256K to 1M. At 512K, accuracy remains close to dense attention except on multi-key retrieval (Appendix H.2).

## 5.2 SEMANTIC BLOCKS AND PER-QUERY SELECTION

We compare our method with positional block selection using the baseline in Section 4. We vary the positional block size B and the number g of neighboring queries sharing a selection. For shared selection, we combine scores from all queries in the group to choose one set of blocks, following FlashPrefill V2 (Fan et al., 2026a). Sharing reduces cost but can miss blocks needed by individual queries. We also tried averaging the queries before scoring, or taking the maximum query score for each block. Neither improved accuracy (Appendix F.4).

Figure 5 compares these choices at the same token budget on multi-key retrieval, where the 4B model is most sensitive to sparsification. Large positional blocks are fast but retrieve poorly, and smaller ones improve accuracy at a higher execution cost, particularly with per-query selection. At B = 32, per-query positional selection is still less accurate than our method and more than 4× slower to prefill. Sharing selections makes positional blocks faster, but the most accurate shared settings remain less accurate than our method and take 1.8–2.7× as long to prefill. On MK-2 the gap is wider, with every positional setting at least 25 points below our method. On Qwen3.5-35B-A3B, only one tested positional setting matches our MK-3 accuracy, but it takes 1.8× as long to prefill and scores 14 points lower on MK-2 (Appendix F.4). As a reference point, we also ran FlashPrefill V2 with the SGLang implementation released with it, on the same prompts (Appendix F.5). The two engines differ in implementation, but their dense attention runs at a similar speed and reaches a similar accuracy. We compare speedups relative to dense attention on each method’s own engine. Our method is more accurate at comparable speedups. Among the tested thresholds, FlashPrefill V2 reaches our MK-3 accuracy only with a 1.7× speedup, compared with 2.2× for ours.

![](images/d9bacabe0859b6fd37acfffab19ab3039da7965e0202100950c19c32f28f1cb9.jpg)  
Figure 5: MK-3 and MK-2 accuracy against prefill latency on Qwen3.5-4B at 512K, without and with an initial block of 1,024 positions. Numbers are the positional block size B, and $g$ is the number of queries sharing one selection. Adding the initial block has little effect on prefill time (Appendix F.4). We therefore reuse timings measured without it for per-query selection and shared semantic selection.

![](images/bb12d73d11e02a68e21f7ef0ca41ca04572c0643ba9f3c535993a61809faa904.jpg)

![](images/1bd0acf68ef559fd6dea22180194df4fc3434eec1d9a98671120f876b80ecb08.jpg)

![](images/199182d1217c21ef415bb79dfe3f2098e7679516aacf5fcbc8efc165658def5a.jpg)  
Ours, shared g=64 (ablation)

![](images/e602beee50b16e99b44bbb34336f3e773bea9f59138684b7e1e1182b0192402a.jpg)

![](images/62fefed3353f6012d31781b48f5772669276aa61d042e7b0bc4cf3e36a070c60.jpg)  
Figure 6: Cluster size and token budget on MK-2 at 512K, Qwen3.5-4B. Larger clusters with a larger budget can improve both accuracy and latency over finer clusters with a smaller budget.  
Figure 7: GPU time of one 512K prefill on Qwen3.5-4B by cluster size u and token budget K.

We also test shared selection for semantic blocks, using groups of 64 neighboring queries as in MInference’s block-sparse pattern (Jiang et al., 2024). This sharply reduces MK-3 accuracy for only a small reduction in prefill time. Per-query selection of semantic blocks thus gives the best accuracy–latency trade-off among these configurations. At a fixed token budget, larger blocks mean fewer blocks per query, reducing the cost of gathering queries and merging outputs (Section 3.4). At 512K, each query scores 256 semantic centroids, compared with 16,384 block means for 32-token positional blocks.

We also test adding a fixed initial block of 1,024 positions, following the practice of including early prompt positions to retain attention sinks (Xiao et al., 2024; Jiang et al., 2024). This tends to improve both methods. Ours then matches the most accurate positional setting on MK-3 while prefilling 1.3× faster, and remains far more accurate on MK-2, also at the smaller budget $K = \bar { L ^ { \prime } } 3 2$ (Figure 5, Table 23 and Appendix F.4).

## 5.3 BLOCK SIZE AND TOKEN BUDGET

On Qwen3.5-4B at 512K, smaller clusters are more accurate at small token budgets. As the budget grows, larger clusters reach similar accuracy while remaining faster (Figure 6; Appendix F.2). Our main setting uses 2,048-token clusters at $K \stackrel { \cdot } { = } L / 1 6$ . We also analyze the attention mass retained by the selected tokens and test selection using dense-attention scores (Appendix G.1).

Figure 7 breaks down prefill GPU time by cluster size. At a fixed budget, reducing the cluster size from 2,048 to 512 tokens increases prefill GPU time by 41% on the 4B model. Most of the increase comes from attention over selected blocks, query gathering, and output merging. With 2,048-token clusters, attention over selected blocks reaches about 70% of the FLOP rate of dense FlashAttention-3 on the same GPU, compared with less than 40% with 512-token clusters. Kernels specialized fo small blocks (Xiao et al., 2025a) could narrow this gap.

On Nemotron, smaller clusters give larger accuracy gains with a smaller runtime penalty. Reducing the cluster size from 2,048 to 512 tokens increases prefill GPU time by 27% (Figure 18). Clusters around 512 to 1,024 tokens work well on this model. Cluster size and token budget should therefore be chosen together for each model and its kernels.

## 5.4 PROMPT-SPECIFIC CLUSTERING

Fitting centroids once on a shared corpus allows reuse across prompts, as in SAAP (Mazare et al.,´ 2025). We compare this approach with our promptspecific fitting, keeping the selection rule unchanged (Table 2). Centroids fitted on a raw-text corpus lose almost all multi-key retrieval accuracy. Using RULER-5 evaluation prompts as the fitting corpus largely restores it, and fitting each prompt separately gives a further gain on MK-2 (Appendix F.6).

Table 2: Centroid fitting data on Qwen3.5-4B at 512K (K = L/16).
<table><tr><td>Centroids fitted on</td><td>MK-3 MK-2</td><td>Hotpot CWE</td></tr><tr><td>Dense attention</td><td>96.0 99.0</td><td>56.0 63.0</td></tr><tr><td>Each prompt (ours)</td><td>92.7</td><td>95.0 57.0 64.3</td></tr><tr><td>Each prompt&#x27;s first 64K</td><td>95.0</td><td>95.0 56.0 66.1</td></tr><tr><td>A raw-text corpus</td><td>1.0</td><td>12.0 51.0 failed</td></tr><tr><td>RULER-5 examples</td><td>91.0</td><td>90.0 55.0 64.6</td></tr></table>

Fitting centroids on just the first 64K tokens of each prompt gives similar accuracy to using the whole prompt and allows ordinary chunked prefill with less memory (Section 3.3 and Appendix B.5).

## 5.5 DECODING OVER LONG GENERATIONS

In our main experiments, each decoding step attends to all generated tokens without selection. Decoding therefore slows as the output grows, which matters for the long reasoning traces in MRCR-8 and LongBench-v2 (blue in Figure 10; Appendix E.5).

We can instead add older generated tokens to the index by assigning them to the existing centroids without refitting. We update the index every 1,024 generated tokens and keep the newest 512 tokens outside it. Decode latency then stays nearly constant (orange in Figure 10), even over 64K generated tokens after a 1M-token prompt. LongBench-v2 accuracy changes little on either Qwen model (Appendices E.5 and H.1).

## 6 LIMITATIONS AND FUTURE WORK

Our experiments use BF16 inference at batch size one on a single H200. Throughput under continuous batching and on other GPU architectures remains to be evaluated. The method retains the full prompt KV cache, so memory capacity still limits context length and batch size. Fitting centroids on the whole prompt raises peak memory above dense chunked prefill, but fitting them on the first 64K tokens allows chunked prefill with similar accuracy (Appendix B.5). Our method can also be combined with other efficiency techniques, such as KV-cache quantization (Liu et al., 2024) and cache compression (Li et al., 2024), which reduce different costs.

Training the model for sparse attention, as in DSA (DeepSeek-AI, 2025) and QSA (Qiu et al., 2026), may bring further gains. Our short distillation experiments did not consistently improve retrieval (Appendix G), but longer training on long-context task data may help preserve accuracy at smaller budgets.

## 7 CONCLUSION

We presented a simple but effective training-free sparse attention method that builds semantic blocks from each prompt and uses them for both prefill and decoding. Centroid scoring keeps selection cheap, while KV-outer execution makes prefill efficient with a separate selection for each query. This addresses the selection and execution costs discussed by Nawrot et al. (2026). The resulting savings allow a larger token budget and yield whole-model speedups on two hybrid LLMs.

## ACKNOWLEDGMENTS

We thank the National Institute of Information and Communications Technology (NICT) for the opportunity to conduct this research under our joint research framework.

## REFERENCES

Huzama Ahmad and Se-Young Yun. SpotAttention: Plug-in block-sparse routing for pretrained long-context transformers. arXiv preprint arXiv:2606.22874, 2026.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit´ Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Yuxiao Dong, Jie Tang, Lei Hou, and Juanzi Li. LongBench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025.

Yushi Bai, Qian Dong, Ting Jiang, Xin Lv, Zhengxiao Du, Aohan Zeng, Jie Tang, and Juanzi Li. IndexCache: Accelerating sparse attention via cross-layer index reuse. In Conference on Language Modeling, 2026. arXiv:2603.12201.

DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

Dhruv Deshmukh, Saurabh Goyal, Nipun Kwatra, and Ramachandran Ramjee. Kascade: A practical sparse attention method for long-context LLM inference. arXiv preprint arXiv:2512.16391, 2025.

Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. FlexAttention: A programming model for generating fused attention variants. In Proceedings of Machine Learning and Systems, volume 7, 2025. arXiv:2412.05496.

Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, and Ran He. FlashPrefill V2: Block-sparse prefill attention for long-context LLM serving. arXiv preprint arXiv:2608.19758, 2026a.

Qihang Fan, Huaibo Huang, Zhiying Wu, Bingning Wang, and Ran He. UniPrefill: Universal longcontext prefill acceleration via block-wise dynamic sparsification. arXiv preprint arXiv:2605.06221, 2026b.

Qihang Fan, Huaibo Huang, Zhiying Wu, Juqiu Wang, Bingning Wang, and Ran He. FlashPrefill: Instantaneous pattern discovery and thresholding for ultra-fast long-context prefilling. arXiv preprint arXiv:2603.06199, 2026c.

Gemma Team. Gemma 4, 2026.

Coleman Richard Charles Hooper, Sehoon Kim, Hiva Mohammadzadeh, Monishwaran Maheswaran, Sebastian Zhao, June Paik, Michael W. Mahoney, Kurt Keutzer, and Amir Gholami. Squeezed attention: Accelerating long context length LLM inference. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics, 2025.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In Conference on Language Modeling, 2024.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, 2024.

Dalton Jones, Junyoung Park, Matthew Morse, Mingu Lee, Chris Lott, and M. Harper Langston. QuoKA: Query-oriented KV selection for efficient LLM prefill. In International Conference on Learning Representations, 2026.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with PagedAttention. In Proceedings ofthe 29th ACM Symposium on Operating Systems Principles, 2023.

Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. FlexPrefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In International Conference on Learning Representations, 2025.

Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. MiniMax Sparse Attention. arXiv preprint arXiv:2606.13392, 2026.

Hoang Anh Duy Le, Sahil Joshi, Zeyu Yang, Zhaozhuo Xu, and Anshumali Shrivastava. Scout before you attend: Sketch-and-walk sparse attention for efficient LLM inference. In International Conference on Machine Learning, 2026. arXiv:2602.07397.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2404.14469.

Di Liu, Meng Chen, Baotong Lu, Huiqiang Jiang, Zhenhua Han, Qianxi Zhang, Qi Chen, Chengruidong Zhang, Bailu Ding, Kai Zhang, Chen Chen, Fan Yang, Yuqing Yang, and Lili Qiu. RetrievalAttention: Accelerating long-context LLM inference via vector retrieval. In Advances in Neural Information Processing Systems, volume 38, 2025a. arXiv:2409.10516.

Guangda Liu, Chengwei Li, Jieru Zhao, Chenqi Zhang, and Minyi Guo. ClusterKV: Manipulating LLM KV cache in semantic space for recallable compression. In Proceedings of the 62nd ACM/IEEE Design Automation Conference, 2025b.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Yutao Zhang, Zhilin Yang, Xinyu Zhou, Mingxing Zhang, and Jiezhong Qiu. MoBA: Mixture of block attention for long-context LLMs. In Advances in Neural Information Processing Systems, volume 38, 2025. arXiv:2502.13189.

Pierre-Emmanuel Mazare, Gergely Szilvasy, Maria Lomeli, Francisco Massa, Naila Murray, Herv´ e´ Jegou, and Matthijs Douze. Inference-time sparse attention with asymmetric indexing. ´ arXiv preprint arXiv:2502.08246, 2025.

MiniMax. MiniMax-M1: Scaling test-time compute efficiently with lightning attention. arXiv preprint arXiv:2506.13585, 2025.

Piotr Nawrot, Robert Li, Renjie Huang, Sebastian Ruder, Kelly Marchisio, and Edoardo M. Ponti. The sparse frontier: Sparse attention trade-offs in transformer LLMs. In Findings of the Association for Computational Linguistics: ACL 2026, 2026.

NVIDIA. NVIDIA-Nemotron-3.5-Lightning-30B-A3B. https://huggingface.co/ nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16, 2026.

OpenAI. OpenAI MRCR: Long context multiple needle in a haystack benchmark. Hugging Face dataset, https://huggingface.co/datasets/openai/mrcr, 2025.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. YaRN: Efficient context window extension of large language models. In International Conference on Learning Representations, 2024.

Zihan Qiu, Zekun Wang, Xiao Li, Yanpeng Li, Yang Xu, Yixuan Wang, Huaqing Zhang, Rui Men, Bochao Mao, Chengruidong Zhang, Fan Zhou, Hao Luo, Haofeng Huang, Haoran Lian, Haoyan Huang, Hongqing Chen, Jianwei Zhang, Jing Xu, Junjie Wang, Langshi Chen, Liangyu Wang, Linlang Jiang, Man Yuan, Minmin Sun, Peng Jin, Siqi Zhang, Siyu Wang, Xingzhang Ren, Yakai Wang, Yi Zhang, Yiming Dong, Yizhong Cao, Yubo Ma, Yunfei Mao, Bo Zheng, and Dayiheng Liu. On the design of Qwen3.8-Next architecture: Evaluation, efficiency, and training stability. arXiv preprint arXiv:2608.30320, 2026.

Qwen Team. Qwen3.5. https://huggingface.co/collections/Qwen/qwen35, 2026.

Aurko Roy, Mohammad Saffar, Ashish Vaswani, and David Grangier. Efficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics, 9, 2021.

Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. FlashAttention-3: Fast and accurate attention with asynchrony and low-precision. In Advances in Neural Information Processing Systems, volume 37, 2024. arXiv:2407.08608.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. RoFormer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for efficient long-context LLM inference. In International Conference on Machine Learning, 2024.

Team Olmo. Olmo 3. arXiv preprint arXiv:2512.13961, 2025.

Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In International Conference on Learning Representations, 2025. arXiv:2404.15574.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations, 2024.

Guangxuan Xiao, Junxian Guo, Kasra Mazaheri, and Song Han. Optimizing mixture of block attention. arXiv preprint arXiv:2511.11571, 2025a.

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. DuoAttention: Efficient long-context LLM inference with retrieval and streaming heads. In International Conference on Learning Representations, 2025b. arXiv:2410.10819.

Yufei Xu, Fanxu Meng, Fan Jiang, Yuxuan Wang, Ruijie Zhou, Zhaohui Wang, Jiexi Wu, Zhixin Pan, Xiaojuan Tang, Wenjie Pei, Tongxuan Liu, Di Yin, Xing Sun, and Muhan Zhang. HISA: Efficient hierarchical indexing for fine-grained sparse attention. In Conference on Language Modeling, 2026. arXiv:2603.28458.

Jiayi Yuan, Cameron Shinn, Kai Xu, Jingze Cui, George Klimiashvili, Guangxuan Xiao, Perkz Zheng, Bo Li, Yuxin Zhou, Zhouhai Ye, Weijie You, Tian Zheng, Dominic Brown, Pengbo Wang, Markus Hoehnerbach, Richard Cai, Julien Demouth, John D. Owens, Xia Hu, Song Han, Timmy Liu, and Huizi Mao. BLASST: Dynamic BLocked attention sparsity via softmax thresholding. In Proceedings ofMachine Learning and Systems, volume 8, 2026.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, 2025. arXiv:2502.11089.

Yanke Zhou, Yiduo Li, Hanlin Tang, Maohua Li, Kan Liu, Tao Lan, Lin Qu, Yuan Yao, and Xiaoxing Ma. Full attention strikes back: Transferring full attention into sparse within hundred training steps. arXiv preprint arXiv:2605.16928, 2026.

Kan Zhu, Tian Tang, Qinyu Xu, Zhan Jin, Yile Gu, Zhichen Zeng, Rohan Kadekodi, Liangyu Zhao, Ang Li, Arvind Krishnamurthy, and Baris Kasikci. Tactic: Adaptive sparse attention with clustering and distribution fitting for long-context LLMs. In International Conference on Learning Representations, 2026.

## A COMPUTATIONAL COST

We consider a layer with $H _ { k v }$ KV heads, each shared by a group H of $G = | \mathcal { H } |$ query heads, so it has $H _ { q } = G H _ { k v }$ query heads. The prompt has L tokens, queries and keys have dimension $d ,$ and each KV head has $C$ semantic blocks. Dense attention costs $\Theta ( H _ { q } L d )$ per generated token and $\Theta ( H _ { q } L ^ { 2 } d )$ over a prefill. Table 14 reports the measured GPU time of each stage.

Building the index. The index is built once per layer and prompt. With a fixed number of k-means iterations, fitting the centroids and assigning all prompt keys cost $O ( H _ { k v } L C d )$ . Normalizing the keys and packing the KV into block order each cost $\bar { O } ( H _ { k v } L d )$ . Appendix B gives the clustering settings.

Scoring and selection. Scoring C centroids in each query head costs $O ( H _ { q } C d )$ per query position and $O ( \bar { H } _ { q } L C d )$ over a prefill. Selection runs once per KV head and uses a fixed number of thresholdrefinement passes, costing $O ( H _ { k v } C )$ per position. Per-token indexers such as DSA (DeepSeek-AI, 2025) and MSA (Lai et al., 2026) instead score all visible tokens, so their scoring remains quadratic over a prefill even with a small attention budget.

Attention. For each KV head, attention at position t costs $O ( G d | S _ { t } | )$ , where $S _ { t }$ is the set of positions read by that query (Section 3.2). Summing this cost over query positions and KV heads gives the full prefill cost. The set includes selected blocks and positions read without selection, such as the current chunk during prefill or generated tokens during decoding. Selecting whole blocks can exceed the budget $K ,$ and the initial dense queries read their full visible prefix. Including all generated tokens makes decode cost grow with generation length.

Growth with prompt length. Our main configuration keeps the target mean block size u fixed, so that $C \approx L / u ,$ , and uses $K = L / 1 6$ . Prefill therefore remains quadratic in prompt length. More generally, prefill can become subquadratic if both $C$ and the average number of positions read per query grow sublinearly with $L .$

Memory. The index stores $O ( H _ { k v } C d )$ centroid values and $O ( H _ { k v } L )$ block assignments. Prompt KV is stored once, in block order. Appendix B.5 reports peak memory, including temporary activations and kernel workspace.

## B IMPLEMENTATION DETAILS

## B.1 ROUTING AND CLUSTERING

Clustering and selection use the pre-RoPE queries and keys of each head, 256-dimensional vectors normalized as whole vectors. Each KV head has a separate prompt partition. For a target mean block size u, the number of clusters is C = min L, max(128, $\bar { 2 } ^ { \lceil \log _ { 2 } \dot { ( L / u ) } \rceil } ) \big )$ . Table 3 lists the resulting cluster counts at $u = 2 , 0 4 8$ . At 128K, $L / u = 6 4$ , so $C = 1 2 8$ and the mean block size is 1,024, half the target. Centroids start from uniformly sampled keys and are refined for eight iterations on the sample in the table, with empty clusters re-seeded from it. All prompt keys are assigned after fitting.

In the fit-budget sweep, the sample count ranges from 2,048 to all prompt keys and the iteration count from two to sixteen. MK-3 accuracy at 512K ranges from 90% to 96% over 100 items, while prefill latency varies by at most six percent.

For the PCA ablation, we fit a projection using about one million raw-corpus keys per layer and KV head, without subtracting their mean. We apply the same projection to queries and keys, then normalize each vector.

## B.2 SELECTION AND ATTENTION

The prefill selector uses 16 threshold-refinement steps to approximate the score-ordered budget selection in Equation 3. Its size penalty uses the visible block sizes $| \boldsymbol { B } _ { c , t } ^ { \prime } |$ of Equation 2, and blocks with no visible token are never selected. The selected clusters are represented by a bitmask per query and KV head. Counts and prefix sums convert this to cluster-major query lists and querymajor merge slots, without materializing the full Boolean selection matrix or sorting partial outputs afterward. Within each block of this cache, tokens stay in position order, and every block starts at a page boundary. Each prefill chunk passes the exact count of preceding tokens in every cluster to FlashAttention-3, including a partially filled final page. The selected history is therefore entirely before the chunk, and only the separate current-chunk call needs a triangular causal mask. The GQA heads are folded into query rows for the history call.

Table 3: Clustering per KV head at $u = 2 , 0 4 8 ,$ at the four prompt lengths we report.
<table><tr><td>Prompt length</td><td>Clusters</td><td>Mean block size</td><td>Keys used to fit</td></tr><tr><td>128K</td><td>128</td><td>1,024</td><td>8,192</td></tr><tr><td>256K</td><td>128</td><td>2,048</td><td>8,192</td></tr><tr><td>512K</td><td>256</td><td>2,048</td><td>8,192</td></tr><tr><td>1M</td><td>512</td><td>2,046</td><td>16,384</td></tr></table>

Table 4: Main Qwen3.5-4B runtime parameters. Benchmark-specific budgets are described in Section 4.
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Replaced layers</td><td>Eight full-attention layers of 32</td></tr><tr><td>Query / KV heads</td><td>16 / 4, head dimension 256</td></tr><tr><td>Routing</td><td>Pre-RoPE Q/K, whole-vector normalization, identity map</td></tr><tr><td>Target mean block size</td><td> $u = 2 , 0 4 8 ,$  which sets the cluster count of Table 3</td></tr><tr><td>Clustering</td><td>Eight iterations, min(L, max(8,192, 32C)) sampled keys. The main RULER results average seeds 0–2, and other results use seed 0 unless</td></tr><tr><td>Score parameters</td><td>stated otherwise (Appendix C.4)  $\tau = 0 . 1 , \beta = 0 . 2 5$ </td></tr><tr><td>Length-controlled budget</td><td> $K = L / 1 6$  for RULER and the main latency sweep</td></tr><tr><td>Query chunk</td><td>8,192 positions</td></tr><tr><td>Dense prefix</td><td>A query chunk starting at position 4K (four times the budget) or earlier</td></tr><tr><td>Attended without selection (Dt)</td><td>attends to its whole prefix without selection The visible part of the current chunk in prefill, every generated token in</td></tr><tr><td>Numerics</td><td>decoding BF16 attention, FP32 score aggregation and output merge</td></tr><tr><td>Short prompts</td><td>Sparse attention at every measured length, with the runtime&#x27;s dense</td></tr><tr><td>Decode graph</td><td>fallback for short prompts switched off The whole decode step captured as one CUDA graph in the latency runs,</td></tr><tr><td>Prefill schedule</td><td>for dense and ours Sliced (32,768-position slices) at 1M, for the task evaluations above 256K, and for the main RULER-5 evaluations at every length (Appendix B.4).</td></tr></table>

Partial attention outputs are merged using their softmax normalizers. If region j returns output $o _ { j }$ and log-sum-exp $\ell _ { j }$ , the combined output is $\begin{array} { r } { o = \sum _ { i } \exp ( \ell _ { j } - \ell ) o _ { j } } \end{array}$ with $\ell = \mathrm { l o g s u m e x p } _ { j } \ell _ { j }$ . The merge is computed in FP32 and cast back to BF16, giving attention over the selected positions up to floating-point rounding.

The decode backend of the latency runs uses five launches per sparse layer, namely suffix write, query normalization, fused scoring and selection, split attention, and output merge. Selection directly emits contiguous cluster ranges. The attention kernel reads selected prompt ranges and the tiles of the generated suffix in the same launch, then merges their softmax statistics. All live lengths are read on device and workspace is reused, avoiding per-step host reads and allocations. The RULER and MRCR-8 runs and the 4B model’s LongBench-v2 runs in Table 1 use a fused decode backend with more launches per layer, which selects the same clusters, under vLLM’s piecewise CUDA graphs.

## B.3 COMPILATION AND DECODE GRAPHS

Measurements use TORCHINDUCTOR DETERMINISTIC=1, with combo kernels and their benchmarking disabled. These settings reduce numerical differences between independent compilations. Prefill uses piecewise graphs where applicable. For the main decode latency results, both methods use vLLM’s FULL DECODE ONLY mode, which captures the whole decode step as one CUDA graph. Each sparse layer has a fixed-address arena containing prompt KV, centroids, posting lists, generated-token storage, and workspace. The graph is captured at engine warm-up and reused after the prompt’s index is copied into the arena. One-time graph capture is excluded from steady decode latency for both methods.

## B.4 PREFILL SCHEDULES

An index fitted on the whole prompt needs all of a layer’s prompt keys before that layer’s queries can be routed. The runtime therefore cannot use ordinary chunked prefill, which passes one token chunk through all layers before the next chunk. Instead, every layer processes the whole prompt before the next layer starts, in one of two ways (Figure 8). Single-pass prefill runs the whole prompt through the model as one forward pass, so every projection and feed-forward call processes all prompt positions at once and holds full-length activations. Sliced prefill loops over the layers itself instead of calling the model’s forward pass once, and processes each layer in slices of 32,768 positions. A full-attention layer makes two passes over the slices. The first projects the keys and values for all slices, then builds the index and packs the block-ordered KV cache. The second projects the queries and the output gate, performs attention, and writes the outputs. Linear-attention layers carry their recurrent and convolution states between slices, and feed-forward layers and normalization are sliced as well. Both schedules compute the same attention up to numerical differences, with selection and attention in 8,192-position query chunks inside each layer, and differ in peak memory (Table 5). Sliced prefill is required where single-pass prefill exceeds device memory, and Table 4 lists which schedule each measurement uses. The schedule is selected outside the compiled decode path, and elementwise operators are compiled separately for the sliced path.

Dense attention, chunked Ours, single-pass Ours, sliced   
for each 16K-token chunk: for each layer: for each layer:   
for each layer: Q, K, V <- project(all) for each 32K slice:   
Q, K, V <- project(chunk) fit index on all keys K, V <- project(slice)   
attend(chunk, KV so far) for each 8K query chunk: fit index on all keys   
feed-forward(chunk) select, attend for each 32K slice:   
feed-forward(all) Q <- project(slice)   
for each 8K query chunk:   
select, attend   
feed-forward(slice)  
Figure 8: The order of operations in one prefill, shown for a full-attention layer and its feed-forward block, where “all” is every prompt position. Linear-attention layers follow the same order without the index.

## B.5 MEMORY

In the memory and latency runs, prompt KV is stored only in the block-ordered cache, with space reserved for generated tokens and workspace. The 4B model’s single-pass RULER runs at 128K and 256K also write prompt KV to vLLM’s paged cache. Table 5 compares memory use with dense chunked prefill. For our method, “KV cache etc.” is the allocated memory remaining after prefill, excluding weights. “Transient” is the additional allocation at the peak.

The additional memory over dense chunked prefill has two sources. The sliced schedule of Appendix B.4 holds the residual stream for the whole prompt and, until a layer’s index is built, that layer’s keys and values, which grow with prompt length. The KV-outer attention of Section 3.4 holds, for one query chunk and KV head at a time, the gathered queries and their partial outputs, which grow with the query chunk and the number of blocks its queries selected. The 512K latency runs use single-pass prefill, whose peak is listed separately and is set by the full-length feed-forward intermediates. Both methods fit on one H200 at every length measured. Sparsity does not reduce memory here, because the full prompt KV is kept.

Table 5: Qwen3.5-4B memory on one H200, in GiB. Dense totals are derived request-sized estimates that exclude excess preallocated cache capacity, while sparse peaks are measured allocated memory, so dense and sparse rows are accounted differently.
<table><tr><td>Method</td><td>Length / schedule</td><td>Weights</td><td>KV cache etc.</td><td>Transient</td><td>Total</td></tr><tr><td>Dense (derived)</td><td>512K / chunked</td><td>8.35</td><td>16.00</td><td>1.58</td><td>25.93</td></tr><tr><td>Ours (measured)</td><td>512K / single-pass</td><td>8.35</td><td>20.84</td><td>53.50</td><td>82.69</td></tr><tr><td>Ours (measured)</td><td>512K / sliced</td><td>8.35</td><td>20.84</td><td>11.91</td><td>41.09</td></tr><tr><td>Ours, prefix index (measured)</td><td>512K / chunked</td><td>8.35</td><td>20.86</td><td>6.43</td><td>35.64</td></tr><tr><td>Dense (derived)</td><td>1M / chunked</td><td>8.35</td><td>31.97</td><td>1.58</td><td>41.90</td></tr><tr><td>Ours (measured)</td><td>1M / sliced</td><td>8.35</td><td>39.65</td><td>22.37</td><td>70.37</td></tr><tr><td>Ours, prefix index (measured)</td><td>1M / chunked</td><td>8.35</td><td>39.72</td><td>11.89</td><td>59.96</td></tr></table>

The prefix-index rows fit centroids on the prompt’s first 64K tokens and assign later keys to them as they arrive. This allows ordinary chunked prefill with 8K-token chunks and no full-length activations. The remaining memory overhead comes from the output buffer and the attention kernels’ per-chunk buffers. This schedule is about a sixth slower than sliced prefill. The first-64K partition gives similar accuracy to the full-prompt partition on the four RULER tasks in Table 2. Running chunked prefill with this index gives 93.0 on MK-3 at 512K, compared with 95.0 for the first-64K row in Table 2. The reported results use the full-prompt index.

## C MEASUREMENT PROTOCOL

## C.1 LATENCY

All headline latency measurements use one request on one H200 and BF16 weights. Both methods use FlashAttention-3 for prefill, and our decoding uses the fused backend of Appendix B. Each latency measurement uses one warm-up followed by three measured repetitions. Each request generates 64 tokens for prompts through 512K and 32 tokens at 1M. For each repetition, decode latency is the time between the first and last generated tokens divided by the number of tokens after the first. We report the median over the three repetitions. For the 4B model, our method is timed on two different prompts of each length through 512K and on one prompt at 1M. Dense attention is timed on two prompts at 512K and on one prompt at 128K, 256K, and 1M. Where there are two prompts, we report their mean. The 35B-A3B model is timed on one prompt at every length under the same protocol. Both decode paths use full-step CUDA graphs. We measure our method’s latency using single-pass prefill through 512K on the 4B model and through 256K on the 35B-A3B model, with sliced prefill at longer lengths. The dense baseline uses vLLM’s chunked prefill. Layer timings sum kernels attributed to the attention backend, with decode accounting over the final 32 generated steps. They are profiler totals, not whole-model wall-clock measurements.

## C.2 QUALITY

RULER uses 100 cached prompts per task and length, shared by the two Qwen models, with greedy decoding and thinking disabled. On the 4B model, our main RULER-5 evaluations and all task evaluations above 256K use sliced prefill (Table 4). All sparse task evaluations of the 35B-A3B model use sliced prefill.

MRCR-8 uses revision f4c69fa (8 December 2025) and the model’s thinking sampling preset. We keep the official length-bin boundaries and assign prompts using their length in Qwen3.5 tokens after the chat template. The bins contain 100 items at 128K and 512K and 99 at 256K. The generation limit accommodates the longest answer and a 32,768-token thinking budget.

LongBench-v2 retains 470 items whose raw prompts fit within 983,040 tokens. Thinking is enabled, with at most 40,960 generated tokens. Sampling uses temperature 1.0, top-p 0.95, top-k 20, min-p 0, and presence penalty 1.5.

The main MRCR-8 and LongBench-v2 results average three runs, each with its own sampling seed and, for our method, its own clustering seed. Budgets are defined in Section 4.

Table 6: The clustering seed and the seeding procedure, with RULER multi-key accuracy in percent at $K = L / 1 6$ and $u = 2 , 0 4 8 ,$ , 100 items per task and length.
<table><tr><td>Model</td><td>Task</td><td>Length</td><td>Seeding</td><td>Mean (s.d.)</td></tr><tr><td>Qwen3.5-4B</td><td>MK-2</td><td>512K</td><td>random</td><td>95.0 (1.0)</td></tr><tr><td rowspan="5"></td><td rowspan="5"></td><td></td><td>farthest k-means++</td><td>92.7 (1.5) 91.7 (1.5)</td></tr><tr><td></td><td></td><td></td></tr><tr><td>MK-3 512K</td><td>random</td><td>92.7 (0.6)</td></tr><tr><td></td><td>farthest</td><td>94.3 (1.5)</td></tr><tr><td></td><td>k-means++</td><td>92.0 (3.6)</td></tr><tr><td>Nemotron 3.5</td><td>MK-3</td><td>256K</td><td>random</td><td>73 (1 seed)</td></tr><tr><td></td><td></td><td></td><td>farthest</td><td>86 (1 seed)</td></tr><tr><td></td><td></td><td>512K</td><td>random</td><td>84 (1 seed)</td></tr><tr><td></td><td></td><td></td><td>farthest</td><td>80 (1 seed)</td></tr></table>

Table 7: RULER results of our method in the main table, Qwen3.5-4B at $K = L / 1 6$ and $u = 2 , 0 4 8 .$ 100 items per task and length. Each entry is the mean (sample s.d.) over three clustering seeds. RULER-5 is the mean over the five tasks, and its s.d. is computed over the three seeds’ five-task means.
<table><tr><td>Prompt</td><td>MK-3</td><td>MK-2</td><td>MV</td><td>CWE</td><td>HotpotQA</td><td>RULER-5</td></tr><tr><td>128K</td><td>100.0 (0.0)</td><td>98.7 (1.5)</td><td>99.8 (0.3)</td><td>78.3 (0.5)</td><td>57.3 (1.2)</td><td>86.8 (0.3)</td></tr><tr><td>256K</td><td>97.7 (0.6)</td><td>91.0 (2.0)</td><td>99.3 (0.4)</td><td>70.3 (1.4)</td><td>55.0 (1.0)</td><td>82.7 (0.1)</td></tr><tr><td>512K</td><td>92.7 (0.6)</td><td>95.0 (1.0)</td><td>100.0 (0.0)</td><td>64.3 (0.9)</td><td>57.0 (1.0)</td><td>81.8 (0.5)</td></tr><tr><td>1M</td><td>28.3 (1.5)</td><td>68.0 (2.0)</td><td>100.0 (0.0)</td><td>29.4 (1.2)</td><td>50.0 (1.0)</td><td>55.2 (0.8)</td></tr></table>

## C.3 PROMPT LENGTHS

The prompts come from lm eval’s port of the RULER generator, with NVIDIA’s filling for commonword extraction beyond 128K, where the port’s word list runs out. The measured median RULER-5 prompt lengths are within five percent of nominal at 128K and two percent from 256K onward, except multi-value retrieval at 1M, whose generator runs out of material at about 647K tokens. Of the other tasks at 512K, frequent-word extraction is the one the generator sizes by words rather than tokens, and its prompts run from 418K to 505K tokens with a median of 476K. The budget follows the nominal length. SQuAD QA is not in RULER-5 because filling it repeats the answer paragraph, about two and a half times at 512K and five times at 1M, so raising the nominal length also changes how often the answer appears.

## C.4 CLUSTERING SEEDS AND SEEDING

For both Qwen models, the main RULER-5 results average three clustering seeds with random initialization (Tables 7 and 9). Appendix comparisons use seed 0 unless stated otherwise. When the main configuration is rerun alongside variants, we report its score from that run, which can differ across tables because the fitted partition changes. When it is included only as a reference, we use its three-seed mean.

Random initialization draws from a sample of 8,192 keys, and k-means++ uses $D ^ { 2 }$ sampling on the same sample. Greedy farthest-point seeding uses 65,536 keys for Qwen seed 0 and 262,144 for seeds 1 and 2, and 1,024 keys per centroid on Nemotron. Neither alternative gives a consistent improvement (Table 6), so we retain random initialization.

Across three seeds on the same 512K items, MK-3 accuracy varies by one point and MK-2 by two, while up to 8% of the items in either task change between correct and incorrect.

## D QUALITY IN DETAIL

## D.1 RULER RESULTS BY TASK

Table 8: Every RULER task on Qwen3.5-4B at a nominal 512K, in percent, 100 items per task, $K = L / 1 6$ except the last row, which halves it. Ours on the starred tasks and SQuAD is the mean over three clustering seeds, on the others one seed. The second row is the median length of each task’s prompts in thousands of tokens (Appendix C.3). S1–S3 are the single-needle tasks, MK-1 to MK-3 multi-key, MQ multi-query, MV multi-value, VT variable tracking, CWE and FWE common- and frequent-word extraction, SQuAD the QA task with its paragraphs repeated, and Hotpot HotpotQA. The starred tasks form RULER-5.
<table><tr><td>Prompt (K tokens)</td><td>S1 524</td><td>S2 524</td><td>S3 524</td><td>MK-1 524</td><td>MK-2* 524</td><td>MK-3* 523</td><td>MQ 524</td><td>MV* 524</td><td>VT 524</td><td>CWE* 523</td><td>FWE 476</td><td>SQuAD 523</td><td>Hotpot* 517</td><td>Mean</td></tr><tr><td>Dense</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td><td>99.0</td><td>96.0</td><td>99.8</td><td>100.0</td><td>100.0</td><td>63.0</td><td>99.3</td><td>76.1</td><td>56.0</td><td>91.5</td></tr><tr><td>Ours</td><td>100.0</td><td>100.0</td><td>99.0</td><td>100.0</td><td>95.0</td><td>92.7</td><td>100.0</td><td>100.0</td><td>100.0</td><td>64.3</td><td>99.0</td><td>79.1</td><td>57.0</td><td>91.2</td></tr><tr><td>Ours,  $K = L / 3 2$ </td><td>100.0</td><td>100.0</td><td>99.0</td><td>99.0</td><td>76.7</td><td>82.7</td><td>99.5</td><td>99.5</td><td>100.0</td><td>63.2</td><td>93.7</td><td>74.3</td><td>52.0</td><td>87.7</td></tr></table>

On Qwen3.5-4B at 512K, our mean accuracy over all thirteen RULER tasks remains close to dense attention (Table 8). The largest losses occur on multi-key retrieval, where MK-2 identifies needles by words and MK-3 by UUIDs. Halving the budget to $K \stackrel { \cdot } { = } L / 3 2$ further reduces multi-key accuracy, with smaller drops on frequent-word extraction and the two QA tasks. Most tasks outside RULER-5 are near ceiling. RULER-5 focuses on the harder tasks and includes multi-value retrieval to check recovery of multiple answers.

The italic rows under the 4B model are the dense Qwen3.5-2B model on the same items, a reference for how much a smaller dense model loses. On Qwen3.5-4B, sparsification reduces accuracy most on the two multi-key tasks from 256K onward, especially MK-2 (Table 9). HotpotQA and common-word extraction remain close to dense attention through 512K. Both fall a few points below dense attention at 1M.

## D.2 POSITIONAL SCALING BEYOND THE NATIVE CONTEXT

YaRN changes position interpolation and attention scaling (Peng et al., 2024). We set YaRN’s attention factor to $\alpha = 1 / \mathrm { m s c a l e } ( s )$ , with mscale(s) = 0.1 ln s + 1 (α = 0.878 at s = 4). The runtime multiplies the rotary cosine and sine tables by α mscale(s) = 1, which cancels the attention scaling while retaining interpolation. Table 10 compares these settings on matched 512K prompts on the 4B model. The 35B-A3B model uses the chosen setting without a sweep of its own. Unless stated otherwise, the interpolation factor is four.

We chose the setting using dense attention’s five-task mean, with QA and common-word extraction evaluated on the port’s capped prompts (Appendix C.3). It also gives the best multi-key scores in Table 10. The bold dense row is the main table’s dense measurement, while the bold sparse row is an earlier single-seed run of the main configuration. Removing the extra attention scaling improves both dense and sparse attention on the 4B model. The dense Qwen3.5-2B reference also performs best with this setting and uses it throughout.

## D.3 OUR METHOD WITH A FIXED INITIAL BLOCK

We test a fixed block of the first 1,024 prompt positions to retain attention sinks, as in other sparse attention methods (Xiao et al., 2024; Jiang et al., 2024). These positions are excluded from clustering, and every query includes the block in both prefill and decoding. The block counts toward K, leaving the remaining budget for clusters. All other settings follow the main configuration, which omits this block. Section 5.2 applies the same block to the positional baseline (Table 23).

Tables 11 and 12 compare our method with and without the initial block on the same items. On Qwen3.5-4B, the block recovers most of the 512K MK-3 loss, changes MK-2 little, and slightly improves RULER-5 across lengths. RULER changes are small on Qwen3.5-35B-A3B. On Nemotron 3.5 Lightning, it improves multi-value retrieval but lowers MK-2 accuracy. Aggregate LongBench-v2 accuracy rises slightly on both Qwen models. At 512K, MRCR-8 improves on the 4B model and declines on the 35B-A3B model.

Table 9: RULER-5 by task and length on the three models, in percent, on 100 examples per task at $K = L / 1 6$ . For the Qwen models, the ours rows are means over three clustering seeds, and the 35B-A3B model also has multi-key rows at half and a quarter of the budget at 512K. Nemotron is measured at 512K with one seed, and its last row doubles the budget. Common-word extraction beyond 128K is on the filled prompts, and the 1M multi-value prompts are about 647K tokens long (Appendix C.3).
<table><tr><td>Prompt</td><td>K</td><td>Method</td><td></td><td>Multi-key-3 Multi-key-2 Multi-value</td><td></td><td>Common-word HotpotQA Mean</td><td></td><td></td></tr><tr><td colspan="3">Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">128K</td><td></td><td>Dense</td><td>100.0</td><td>100.0</td><td>100.0</td><td>78.7</td><td>58.0</td><td>87.3</td></tr><tr><td>8,192</td><td>Ours</td><td>100.0</td><td>98.7</td><td>99.8</td><td>78.3</td><td>57.3</td><td>86.8</td></tr><tr><td></td><td>2B dense</td><td>100.0</td><td>100.0</td><td>99.8</td><td>77.8</td><td>52.0</td><td>85.9</td></tr><tr><td></td><td>Dense</td><td>100.0</td><td>100.0</td><td>100.0</td><td>68.3</td><td>55.0</td><td>84.7</td></tr><tr><td rowspan="3"></td><td rowspan="3">256K 16,384</td><td>Ours</td><td>97.7</td><td>91.0</td><td>99.3</td><td>70.3</td><td>55.0</td><td>82.7</td></tr><tr><td>2B dense</td><td>97.0</td><td>99.0</td><td>100.0</td><td>46.7</td><td>46.0</td><td>77.7</td></tr><tr><td>Dense</td><td>96.0</td><td>99.0</td><td>100.0</td><td>63.0</td><td>56.0</td><td>82.8</td></tr><tr><td rowspan="3"></td><td rowspan="3">512K 32,768</td><td>Ours</td><td>92.7</td><td>95.0</td><td>100.0</td><td>64.3</td><td>57.0</td><td>81.8</td></tr><tr><td>2B dense</td><td>70.0</td><td>98.0</td><td>98.2</td><td>19.1</td><td>47.0</td><td>66.5</td></tr><tr><td>Dense</td><td>40.0</td><td>89.0</td><td>100.0</td><td>35.1</td><td>52.0</td><td>63.2</td></tr><tr><td rowspan="3"></td><td rowspan="3">1M 65,536</td><td>Ours</td><td>28.3</td><td>68.0</td><td>100.0</td><td>29.4</td><td>50.0</td><td>55.2</td></tr><tr><td>2B dense</td><td>20.0</td><td>89.0</td><td>98.0</td><td>8.3</td><td>29.0</td><td>48.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Qwen3.5-35B-A3B</td><td></td><td>100.0</td><td></td><td></td><td></td><td></td><td>88.5</td></tr><tr><td rowspan="2">128K</td><td rowspan="2">8,192</td><td>Dense Ours</td><td>98.7</td><td>100.0 100.0</td><td>100.0 99.0</td><td>79.7 82.2</td><td>63.0 63.7</td><td>88.7</td></tr><tr><td>Dense</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2"></td><td rowspan="2">256K 16,384</td><td>Ours</td><td>100.0 96.3</td><td>99.0 99.0</td><td>100.0 99.4</td><td>69.5 68.6</td><td>60.0 59.3</td><td>85.7 84.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4"></td><td rowspan="2">512K 32,768</td><td>Dense</td><td>100.0</td><td>97.0</td><td>100.0</td><td>68.0</td><td>57.0</td><td>84.4</td></tr><tr><td>Ours</td><td>98.7</td><td>97.7</td><td>98.8</td><td>67.2</td><td>58.7</td><td>84.2</td></tr><tr><td>16,384 8,192</td><td>Ours,  $K = L / 3 2$ </td><td>80.0</td><td>96.0</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td></td><td>Ours,  $K = L \dot { / } 6 4$ </td><td>25.0</td><td>68.0</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td rowspan="2"></td><td rowspan="2">1M 65,536</td><td>Dense</td><td>85.0</td><td>85.0</td><td>100.0</td><td>48.2</td><td>48.0</td><td>73.2</td></tr><tr><td>Ours</td><td>79.3</td><td>88.7</td><td>100.0</td><td>44.9</td><td>56.7</td><td>73.9</td></tr><tr><td colspan="2">Nemotron 3.5 Lightning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">512K</td><td rowspan="3">32,768</td><td>Dense</td><td>100.0</td><td>99.0</td><td>100.0</td><td>95.7</td><td></td><td>61.0</td><td>91.1</td></tr><tr><td>Ours</td><td>84.0</td><td></td><td>49.0</td><td>71.5</td><td>92.4</td><td>53.0</td><td>70.0</td></tr><tr><td>65,536 Ours,  $K = L / 8$ </td><td>90.0</td><td></td><td>81.0</td><td>95.5</td><td>95.1</td><td>56.0</td><td>83.5</td></tr></table>

Table 10: Qwen3.5-4B and Qwen3.5-2B under alternative context-extension settings at 512K, with multi-key and multi-value accuracy in percent, n = 100 per task, one clustering seed. Unqualified YaRN rows use factor four, and interpolation-only cancels attention-temperature scaling.
<table><tr><td>Configuration</td><td>MK-3</td><td>MK-2</td><td>MV</td></tr><tr><td>Dense, no extension</td><td>56.0</td><td>91.0</td><td>98.8</td></tr><tr><td>Dense, YaRN</td><td>89.0</td><td>94.0</td><td>99.5</td></tr><tr><td>Dense,  $\mathrm { Y a R N } \ s = 2$ </td><td>74.0</td><td>95.0</td><td>100.0</td></tr><tr><td>Dense, s = 2, interp only</td><td>77.0</td><td>98.0</td><td>100.0</td></tr><tr><td>Dense, interpolation only</td><td>96.0</td><td>99.0</td><td>100.0</td></tr><tr><td>Ours, no extension</td><td>46.0</td><td>66.0</td><td>99.8</td></tr><tr><td>Ours, YaRN</td><td>83.0</td><td>81.0</td><td>99.0</td></tr><tr><td>Ours, interpolation only</td><td>94.0</td><td>93.0</td><td>100.0</td></tr><tr><td>Qwen3.5-2B dense, YaRN</td><td>44.0</td><td>87.0</td><td>95.8</td></tr><tr><td>Qwen3.5-2B dense, interpolation only</td><td>70.0</td><td>98.0</td><td>98.3</td></tr></table>

All rows use $K = L / 1 6$ except the Nemotron rows at $K = L / 8$ . Common-word extraction beyond 128K is on the filled prompts (Appendix C.3).

Table 11: RULER-5 by length and task at 512K, without and with the initial block, on Qwen3.5-4B, Qwen3.5-35B-A3B, and Nemotron 3.5 Lightning. Qwen rows average three clustering seeds.
<table><tr><td rowspan="2">Method</td><td colspan="4">RULER-5</td><td colspan="5">Tasks at 512K</td></tr><tr><td>128K</td><td>256K</td><td>512K</td><td>1M</td><td>MK-3</td><td>MK-2</td><td>MV</td><td>CWE</td><td>HotpotQA</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense</td><td>87.3</td><td>84.7</td><td>82.8</td><td>63.2</td><td>96.0</td><td>99.0</td><td>100.0</td><td>63.0</td><td>56.0</td></tr><tr><td>Ours</td><td>86.8</td><td>82.7</td><td>81.8</td><td>55.2</td><td>92.7</td><td>95.0</td><td>100.0</td><td>64.3</td><td>57.0</td></tr><tr><td>Ours with initial block</td><td>87.0</td><td>83.6</td><td>82.5</td><td>56.6</td><td>95.0</td><td>94.7</td><td>99.2</td><td>66.2</td><td>57.3</td></tr><tr><td>Qwen3.5-35B-A3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense</td><td>88.5</td><td>85.7</td><td>84.4</td><td>73.2</td><td>100.0</td><td>97.0</td><td>100.0</td><td>68.0</td><td>57.0</td></tr><tr><td>Ours</td><td>88.7</td><td>84.5</td><td>84.2</td><td>73.9</td><td>98.7</td><td>97.7</td><td>98.8</td><td>67.2</td><td>58.7</td></tr><tr><td>Ours with initial block</td><td>88.6</td><td>84.5</td><td>84.0</td><td>74.5</td><td>99.3</td><td>98.0</td><td>98.1</td><td>66.5</td><td>58.3</td></tr><tr><td>Nemotron 3.5 Lightning</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense</td><td></td><td></td><td>91.1</td><td></td><td>100.0</td><td>99.0</td><td>100.0</td><td>95.7</td><td>61.0</td></tr><tr><td>Ours</td><td></td><td></td><td>70.0</td><td></td><td>84.0</td><td>49.0</td><td>71.5</td><td>92.4</td><td>53.0</td></tr><tr><td>Ours with initial block</td><td></td><td></td><td>74.7</td><td></td><td>81.0</td><td>42.0</td><td>99.0</td><td>96.5</td><td>55.0</td></tr><tr><td>Ours,  $K = L / 8$ </td><td></td><td></td><td>83.5</td><td></td><td>90.0</td><td>81.0</td><td>95.5</td><td>95.1</td><td>56.0</td></tr><tr><td>Ours,  $K = L { \dot { / } } 8 ,$  with initial block</td><td>一</td><td>一</td><td>83.3</td><td></td><td>91.0</td><td>73.0</td><td>99.8</td><td>96.8</td><td>56.0</td></tr></table>

Table 12: MRCR-8 and LongBench-v2 of Qwen3.5-4B and Qwen3.5-35B-A3B without and with the initial block, the mean over three runs with the standard deviation across runs in parentheses.
<table><tr><td rowspan="2">Method</td><td colspan="3">MRCR-8</td><td colspan="3">LongBench-v2</td></tr><tr><td>128K</td><td>256K</td><td>512K</td><td>≤256K</td><td>&gt;256K</td><td>all</td></tr><tr><td colspan="7">Qwen3.5-4B</td></tr><tr><td>Dense</td><td>0.797 (0.053)</td><td>0.574 (0.010)</td><td>0.412 (0.027)</td><td>54.3 (1.1)</td><td>64.2 (2.6)</td><td>55.7 (0.7)</td></tr><tr><td>Ours</td><td>0.788 (0.056)</td><td>0.561 (0.029)</td><td>0.399 (0.013)</td><td>53.9 (1.0)</td><td>57.7 (2.3)</td><td>54.5 (0.6)</td></tr><tr><td>Ours with initial block</td><td>0.783 (0.024)</td><td>0.561 (0.068)</td><td>0.415 (0.022)</td><td>55.5 (0.9)</td><td>57.2 (3.1)</td><td>55.7 (0.4)</td></tr><tr><td colspan="7">Qwen3.5-35B-A3B</td></tr><tr><td>Dense</td><td>0.846 (0.012)</td><td>0.655 (0.009)</td><td>0.564 (0.056)</td><td>64.4 (0.8)</td><td>66.2 (2.3)</td><td>64.7 (0.6)</td></tr><tr><td>Ours</td><td>0.844 (0.016)</td><td>0.668 (0.013)</td><td>0.504 (0.063)</td><td>63.1 (0.9)</td><td>64.7 (0.9)</td><td>63.3 (0.9)</td></tr><tr><td>Ours with initial block</td><td>0.885 (0.019)</td><td>0.655 (0.020)</td><td>0.489 (0.021)</td><td>64.7 (1.2)</td><td>65.2 (4.3)</td><td>64.8 (1.5)</td></tr></table>

## E LATENCY IN DETAIL

## E.1 LATENCY BY LENGTH

Figure 9 and Table 13 report prefill and decode latency for Qwen3.5-4B. Our method uses single-pass prefill through 512K and sliced prefill at 1M. The nominal 1M latency prompt contains 1,047,552 tokens. Appendix C gives the measurement protocol.

Appendix H reports the corresponding results for Qwen3.5-35B-A3B (Figure 16) and Nemotron 3.5 Lightning (Figure 17).

## E.2 PREFILL KERNEL BREAKDOWN

GPU kernels are grouped by the operator that launches them. At 512K, the first 17 of 64 query chunks use dense attention. The other chunks attend to selected blocks and the visible part of their own chunk.

## E.3 DENSE-PREFIX CUTOFF

The main configuration uses dense attention for a query chunk whose zero-based starting position satisfies $p \leq 4 K$ , four times the token budget. We vary this cutoff to $p \leq x K$ for $x \in \{ 4 , 2 , 1 \}$ while keeping the token budget and other settings fixed. Because the boundary is inclusive and selection is decided per chunk, xK is the cutoff for chunk starts rather than the exact number of densely processed queries.

![](images/9aed29db0bce156481a75e7f8a8d429983402dda43bf7c079ecaf6b13756c4c7.jpg)

![](images/55636381ab0c2725e0ab9ae12c6a1c1a1ef649bff9c3fdee1f2a52b65ca720db.jpg)

![](images/586a140d732a4f3ba50c86bc0860becf36fa30772c399584febae968b1a1e617.jpg)

![](images/60348adb7e2f7e5e6fcb3676bc1bf1afc6fb502d1831e33c44944e108e65d84b.jpg)  
Figure 9: Qwen3.5-4B latency on one H200 at $K = L / 1 6 .$ . The columns show whole-model wallclock latency (left) and GPU time over the eight full-attention layers (right). The rows show prefill (top) and decoding (bottom).

Table 13: Qwen3.5-4B latency on one H200. Whole-model columns are wall-clock measurements, the attention-layer columns profiler totals over the eight full-attention layers. The last rows are the dense reference model, Qwen3.5-2B.
<table><tr><td></td><td colspan="4">Prefill (s)</td><td colspan="4">Decode (ms/token)</td></tr><tr><td>Prompt</td><td>Dense</td><td>Whole model Ours</td><td>Dense</td><td>Attention layers (8) Ours</td><td>Whole model Dense</td><td>Ours</td><td>Attention layers (8) Dense</td><td>Ours</td></tr><tr><td>128K</td><td>4.18</td><td>3.11</td><td>1.92</td><td>0.84</td><td>4.87</td><td>4.07</td><td>1.29</td><td>0.34</td></tr><tr><td>256K</td><td>12.37</td><td>7.06</td><td>7.75</td><td>2.34</td><td>6.14</td><td>4.16</td><td>2.41</td><td>0.43</td></tr><tr><td>512K</td><td>40.69</td><td>17.39</td><td>31.81</td><td>8.18</td><td>8.69</td><td>4.41</td><td>5.00</td><td>0.66</td></tr><tr><td>1M</td><td>144.08</td><td>49.30</td><td>125.47</td><td>29.79</td><td>13.93</td><td>4.84</td><td>10.51</td><td>1.03</td></tr><tr><td colspan="9">Reference, Qwen3.5-2B with dense attention, whole model</td></tr><tr><td>128K</td><td>1.72</td><td></td><td></td><td></td><td>2.89</td><td></td><td></td><td></td></tr><tr><td>256K</td><td>4.99</td><td></td><td></td><td></td><td>3.48</td><td></td><td></td><td></td></tr><tr><td>512K</td><td>16.01</td><td></td><td></td><td></td><td>4.73</td><td></td><td></td><td></td></tr><tr><td>1M</td><td>56.16</td><td></td><td></td><td></td><td>7.48</td><td></td><td>一</td><td></td></tr></table>

Reducing the cutoff substantially lowers the dense-prefix attention time (Table 15). However, the affected queries now require block selection and grouped attention, increasing selected-block computation, query gathering, and output merging. At x = 2, these added costs largely offset the savings, leaving total prefill time nearly unchanged. At x = 1, total time increases. Neither smaller cutoff consistently improves the two retrieval scores, so we retain x = 4.

The dense reference is from Table 14. The quality comparisons use one clustering seed on the same examples, whereas the main RULER-5 results average three seeds.

Table 14: GPU time in seconds for a 512K prefill on Qwen3.5-4B. The attention rows contain only FlashAttention-3 kernels, with scoring, selection, gathering, and merging reported separately. Totals come from profiling rather than the wall-clock medians in Table 13.
<table><tr><td></td><td>Dense</td><td>Ours</td></tr><tr><td>Attention (8 attention layers) selected blocks dense prefix</td><td>31.4</td><td>6.4 4.02</td></tr><tr><td>current chunk Scoring, selection, index build (ours)</td><td>0.0</td><td>2.02 0.34 0.3</td></tr><tr><td>score the centroids and select k-means fit index build</td><td></td><td>0.24 0.02</td></tr><tr><td>Gather, merge and other overhead (ours) gather and index operations</td><td>0.0</td><td>0.02 1.5</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.72</td></tr><tr><td>merge of partial outputs</td><td></td><td>0.38</td></tr><tr><td></td><td></td><td></td></tr><tr><td>masks, casts and copies</td><td></td><td>0.38</td></tr><tr><td>Linear projections (all 32 layers)</td><td></td><td></td></tr><tr><td>Linear-attention layers (24 GDN)</td><td>6.4</td><td>6.4</td></tr><tr><td></td><td>1.6</td><td>1.8</td></tr><tr><td>Norms, activations, RoPE</td><td>0.7</td><td>0.7</td></tr><tr><td>Other</td><td>0.1</td><td>0.0</td></tr><tr><td>Total</td><td>40.3</td><td>17.1</td></tr></table>

Table 15: Effect of the dense-prefix cutoff on Qwen3.5-4B at 512K and $K = 3 2 , 7 6 8$ . Timings are GPU seconds for one prefill, and accuracy is measured on 100 examples per task.
<table><tr><td></td><td colspan="7">GPU seconds of one 512K prefill</td><td colspan="2">Accuracy (%)</td></tr><tr><td>Setting</td><td>Selected blocks</td><td>Dense prefix</td><td>Own</td><td>chunk Attention Scoring</td><td></td><td>Gather, merge</td><td>Total</td><td>MK-3</td><td>MK-2</td></tr><tr><td>Dense attention</td><td></td><td></td><td></td><td>31.44</td><td></td><td></td><td>40.30</td><td>96.0</td><td>99.0</td></tr><tr><td>x = 4 (ours)</td><td>4.02</td><td>2.03</td><td>0.34</td><td>6.38</td><td>0.24</td><td>1.48</td><td>17.07</td><td>96.0</td><td>97.0</td></tr><tr><td>x = 2</td><td>5.02</td><td>0.53</td><td>0.40</td><td>5.95</td><td>0.27</td><td></td><td>1.8117.03</td><td>94.0</td><td>94.0</td></tr><tr><td>x = 1</td><td>5.65</td><td>0.16</td><td>0.43</td><td>6.24</td><td>0.29</td><td></td><td>2.1017.64</td><td>93.0</td><td>93.0</td></tr></table>

## E.4 ABLATION OF THE KERNEL DESIGN

Table 16 compares three implementation choices from Section 3.4 with simpler alternatives. All runs use one GPU and the 512K sliced prefill of Qwen3.5-4B at $K = 3 2 , 7 6 8$ , with block selection and the rest of the runtime unchanged. In Q-outer execution, each tile of 32 queries reads the union of their selected blocks, with each query masked to its own selection. This takes more than twice as long as KV-outer execution and is slower than dense attention. Blocks are read again for every tile that selects them, and masked query–block pairs still incur computation. Gathering selected blocks from prompt-order KV for every query chunk adds about 9% to prefill time. The cluster-ordered cache avoids this repeated gathering. Replacing the fused selection kernel with a matrix product and a sort costs well under a second.

We also time the attention kernel on one chunk of the same prefill. With the same per-query selection, KV-outer execution using FlashAttention-3 takes about 21 ms, compared with 245 ms for FlexAttention (Dong et al., 2025) using a block mask. Most 128-by-128 blocks are partially selected, so FlexAttention processes nearly all of them.

## E.5 INDEXING OLDER GENERATED TOKENS

An optional flushing policy initially reads generated tokens exactly. When the unindexed suffix reaches $W + F$ tokens, the oldest F are assigned to the existing centroids and added to the index without refitting. The newest W remain outside the index. The cache buffers stay at fixed addresses, so the captured decode graph remains valid. Each point in Figure 10 averages the decode steps since the previous point, and all times include index updates.

Table 16: Ablation of the kernel design on Qwen3.5-4B at 512K and $K = 3 2 , 7 6 8 .$ , with the prefill and decode times of each execution.
<table><tr><td>Execution</td><td>Prefill (s)</td><td>Decode (ms/token)</td></tr><tr><td>Dense attention</td><td>40.69</td><td>9.27</td></tr><tr><td>Ours: KV-outer over one cluster-ordered copy</td><td>18.51</td><td>4.49</td></tr><tr><td>Selected blocks gathered from prompt order every chunk</td><td>20.24</td><td>4.45</td></tr><tr><td>Same selection, Q-outer instead of KV-outer</td><td>47.56</td><td>4.29</td></tr><tr><td>Selection by a GEMM and a sort</td><td>19.09</td><td>4.56</td></tr></table>

![](images/95d9c24f45c4533eae154f9a368933cd0d88626cc124300e1b0853b6c498a033.jpg)  
Figure 10: Decode latency of Qwen3.5-4B after a 512K prompt, index updates included.

With this policy the decode time stays nearly flat as the generation grows (Figure 10 and Table 17). Aggregate LongBench-v2 accuracy changes little with flushing (Table 18). We did not test reasoning traces as long as 64K tokens.

## F DESIGN CHOICES ON QWEN3.5-4B

## F.1 THE ROUTING KEY

We compare routing representations while keeping attention computation unchanged. The main configuration uses unit-normalized pre-RoPE keys and queries, with spherical k-means. We test using only the 192 non-rotary dimensions, since the model rotates 64 of its 256 key dimensions. We also test raw keys with Euclidean k-means, using unnormalized cluster means and queries at the attention scale $\tau = { \sqrt { d } }$

At 512K, both variants stay within about two points of our multi-key accuracy (Table 19). The non-rotary variant also matches our multi-key accuracy at 1M. Post-RoPE routing gives similar multi-key accuracy to pre-RoPE routing. PCA projections fitted to raw-corpus keys retain most of the multi-key accuracy at 128 and 32 dimensions. For timing, we use random projections of the same size. Reducing the dimension from 256 to 32 shortens the 512K prefill by about 5% and leaves decoding unchanged. These results support using the original keys for routing on Qwen3.5. On Nemotron, however, centering the keys prevents severe cluster imbalance (Table 34).

## F.2 TOKEN BUDGET AND BLOCK SIZE AT 512K

Table 20 lists the values plotted in Figure 6. It crosses four mean block sizes u with four token budgets K and scores each point on MK-2 over the same 100 items at 512K. The main setting’s accuracy is the three-seed mean of the main RULER table, and the other points are single runs of one code version. Each prefill time is the mean over two prompts. We use MK-2 rather than MK-3 here because its gap to dense attention is larger, so it separates the settings more clearly.

At a fixed budget, finer blocks are slower. Their accuracy advantage is large at small budgets and narrows as the budget grows. The coarse partition with the main budget is both faster and more accurate than the fine partition with half that budget. Because selection takes whole clusters, their total size exceeds K by 0.4–14% across the grid and 2.4% at the main setting, excluding tokens in $\mathcal { D } _ { t } .$ . Accounting for this overshoot does not change the comparison.

Table 17: Qwen3.5-4B mean decode time over the full generation after a nominal 1M prompt, in milliseconds per token.
<table><tr><td></td><td>64 generated</td><td>65,536 generated</td></tr><tr><td>Dense</td><td>15.39</td><td>16.06</td></tr><tr><td>Ours</td><td>4.81</td><td>7.85</td></tr><tr><td>Ours, flushed every  $8 \mathrm { K } \left( W = 2 , 0 4 8 \right)$ </td><td>4.83</td><td>5.09</td></tr><tr><td>Ours, flushed every 1K  $( W = 5 1 2 )$ </td><td>4.82</td><td>4.82</td></tr></table>

Table 18: Qwen3.5-4B LongBench-v2 accuracy (%) without and with flushing $( F = 1 , 0 2 4 , W =$ 512), on the same items and sampling seed. The sparse columns are separate runs from Table 1’s.
<table><tr><td></td><td>Dense</td><td>Ours</td><td>Ours, flushed</td></tr><tr><td>≤256K (403 items)</td><td>53.1</td><td>54.8</td><td>53.8</td></tr><tr><td>&gt;256K (67 items)</td><td>65.7</td><td>61.2</td><td>58.2</td></tr><tr><td>All (470)</td><td>54.9</td><td>55.7</td><td>54.5</td></tr></table>

Figure 11 extends the grid to MK-3 and MK-2 on all three models, with values for the Qwen models in Table 21 and for Nemotron in Table 36. Both Qwen models show the same pattern, with finer clusters ahead at small budgets and coarser clusters catching up as the budget grows. The main setting’s accuracy on each model is the three-seed mean of the main RULER table, and the other points are single runs. For the 35B-A3B model’s $u = 5 1 2$ timings, gathering for KV-outer attention is split into two batches to fit in device memory.

Figure 12 shows the corresponding profile for Qwen3.5-35B-A3B. The dense-prefix cutoff scales with the token budget. In the 4B experiment of Appendix E.3, reducing this cutoff did not improve prefill time.

## F.3 BUDGET SCALING ACROSS CONTEXT LENGTHS

Increasing the budget beyond $L / 1 6$ gives little additional MK-3 accuracy while increasing latency (Table 22). Reducing the budget lowers latency at the cost of retrieval accuracy. Neither a fixed token count nor a fixed fraction predicts accuracy across lengths, because the dense model’s own retrieval performance also declines.

Figure 13 extends the comparison at 512K to MRCR-8 and LongBench-v2 on both Qwen models, over budgets from $L / 6 4$ to ${ \mathrm { ~ \bar { \cal L } / 8 . } }$ . Because benchmark prompt lengths vary, we use the prefill time of a fixed-length RULER prompt as a common reference on the horizontal axis. For each budget, the reference prompt is 512K for MRCR-8 and 256K for LongBench-v2, the upper bound of the band containing most LongBench-v2 items. Filled markers are means over three runs, and hollow markers are single runs.

MK-3 accuracy declines below $L / 1 6 .$ , while MRCR-8 changes less. At $L / 3 2 ,$ the MRCR-8 scores stay within the range of the three $\dot { L } / 1 6$ runs (Table 12). Increasing the budget to $L / 8$ raises MRCR-8 above this range on the 4B model but not on the 35B-A3B model. On the 35B-A3B model, 512-token clusters match dense MK-3 accuracy at $L / 3 2$ and come within six points at $L / 6 4 ,$ , but take longer to prefill than the main setting (Table 21). LongBench-v2 holds its accuracy down to $L / 3 2$ and drops at $L / 6 4$

## F.4 POSITIONAL BASELINES

We form contiguous blocks of B tokens and represent each block by its mean pre-RoPE key. Queries score these means without vector normalization at temperature ${ \sqrt { d } } ,$ , and head probabilities are summed within each GQA group before selecting the top $\lceil K \dot { / } B \rceil$ blocks. With the positional baseline’s raw keys, unnormalized cluster means, and temperature ${ \sqrt { d } } .$ , semantic blocks lose only a few points of accuracy (Table 19). They still outperform positional blocks, so normalization and temperature do not explain the gap. Both methods use the same token budget and dense-prefix, current-chunk, and generated-suffix policies. With per-query selection, positional blocks use the same KV-outer execution as semantic blocks (Section 3.4).

Table 19: The routing key on Qwen3.5-4B, with RULER multi-key and multi-value accuracy in percent at 512K, $K = L / 1 6 , u = 2 { , } 0 4 8 .$ , 100 items per entry. Our row, the post-RoPE row, and the multi-key entries of the non-rotary row are means over three clustering seeds, and all other entries use seed 0. A dash marks an entry that was not measured.
<table><tr><td>Routing key</td><td>MK-3</td><td>MK-2</td><td>MV</td></tr><tr><td>Dense</td><td>96.0</td><td>99.0</td><td>100.0</td></tr><tr><td>Whole key, normalized, τ = 0.1 (ours)</td><td>92.7</td><td>95.0</td><td>100.0</td></tr><tr><td>Whole key after RoPE, normalized,  $\tau = 0 . 1$ </td><td>93.0</td><td>94.0</td><td>100.0</td></tr><tr><td>Non-rotary 192 dimensions, normalized,  $\tau = 0 . 1$ </td><td>93.0</td><td>95.7</td><td>99.5</td></tr><tr><td>Raw key, Euclidean k-means,  $\tau = 1 6$ </td><td>91.0</td><td>93.0</td><td>99.3</td></tr><tr><td>Raw key, Euclidean k-means,  $\tau = 4$ </td><td>95.0</td><td>96.0</td><td>100.0</td></tr><tr><td>PCA to 128 dimensions, normalized,  $\tau = 0 . 1$ </td><td>94.0</td><td>91.0</td><td>100.0</td></tr><tr><td>PCA to 32 dimensions, normalized,  $\tau = 0 . 1$ </td><td>90.0</td><td>96.0</td><td>100.0</td></tr></table>

Table 20: The values behind Figure 6, Qwen3.5-4B at 512K.
<table><tr><td rowspan="2">u</td><td colspan="4">MK-2 (%) at K</td><td colspan="4">Prefill (s) at K</td></tr><tr><td colspan="4">8K 16K 32K</td><td>8K 16K</td><td colspan="3">32K 64K</td></tr><tr><td>Dense</td><td colspan="4">99.0</td><td colspan="4">40.69</td></tr><tr><td>4,096</td><td>32.0</td><td>62.0</td><td>90.0</td><td>100.0</td><td>12.66</td><td>14.05</td><td>16.78</td><td>23.47</td></tr><tr><td>2,048</td><td>40.0</td><td>75.0</td><td>95.0</td><td>99.0</td><td>13.40</td><td>14.98</td><td>17.86</td><td>24.30</td></tr><tr><td>1,024</td><td>56.0</td><td>87.0</td><td>97.0</td><td>100.0</td><td>15.30</td><td>17.31</td><td>20.42</td><td>26.14</td></tr><tr><td>512</td><td>73.0</td><td>86.0</td><td>97.0</td><td>99.0</td><td>18.97</td><td>22.21</td><td>25.59</td><td>30.05</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

For shared selection, g consecutive queries select blocks together using the tile score of FlashPrefill V2 (Fan et al., 2026a). Within each query head, we sum the exponentiated block scores across the queries and normalize over blocks. We then sum these probabilities within each GQA group and select the highest-scoring blocks to fill the budget. All g queries attend to the selected blocks with their original query vectors and causal masks. Groups stay within a prefill chunk, and an incomplete final group is padded by repeating its last query. Both block layouts use this procedure, while decoding remains per-query. A Triton kernel computes group scores without storing individual query scores, and each group attends to its selected blocks in one FlashAttention call (Jiang et al., 2024; Lai et al., 2025).

FlashPrefill V2 shares one selection across 128 query rows. On the 4B model, these correspond to four query heads over 32 positions, within the sharing spans tested here (Fan et al., 2026a). Our comparisons vary block layout and query sharing at a fixed token budget; they do not reproduce the complete selection policies of MInference, FlexPrefill, QuoKA, or FlashPrefill. The released MInference and FlexPrefill kernels available for these experiments supported head dimensions up to 128, whereas Qwen3.5 uses 256. We found no public QuoKA code when we ran the comparisons. Appendix F.5 compares directly with FlashPrefill V2.

At most settings, we also tested scoring blocks with the group’s mean query, as in MInference and FlexPrefill (Jiang et al., 2024; Lai et al., 2025). We further tested the maximum score across queries and a sum over both queries and GQA heads as in FlashPrefill V2. Each gave accuracy similar to or lower than the summed exponentials. Table 23 lists all points in Figure 5, without and with the initial block.

On Qwen3.5-35B-A3B (Table 24), MK-3 is near its ceiling, and positional blocks of 8 tokens shared over 32 queries reach our accuracy, but their prefill takes 1.8× as long, as on the 4B model. Every faster positional setting is at least 6 points less accurate. On MK-2, the same setting is 14 points below our three-seed mean of 97.7%, and every other positional setting is lower still.

In Table 23, the columns with the initial block add the fixed initial block of Appendix D.3 to every row, which changes the prefill times by at most 5%.

![](images/88376b0943957d480bae8f7edf58d3abef6d708531f6e3ae45040d60cd796026.jpg)  
Figure 11: Cluster size and token budget on MK-3 and MK-2 at 512K for Qwen3.5-4B, Qwen3.5- 35B-A3B, and Nemotron 3.5 Lightning.

Table 25 repeats the comparison at the smaller budget $K = L / 3 2$ , where our accuracy falls further below dense attention. Our method stays faster than every positional setting we timed, and is more accurate on MK-2 with and without the initial block. With the block, the most accurate positional setting approaches our MK-3 accuracy but takes 1.9× as long to prefill as our method at $u = 1 , 0 2 4$ and 2.2× as long as at $u = 2 { , } 0 4 8$ . At this budget, halving our cluster size to $u = 1 , 0 2 4$ recovers much of the MK-2 loss, as in the grid of Appendix F.2.

A Quest-inspired min/max descriptor does not improve the positional baseline in this pre-RoPE setting (Tang et al., 2024). This comparison tests the descriptor, not Quest’s complete decoding method. Its shared rows and the temperature sweeps below score the group’s averaged query rather than the summed exponentials.

The min/max descriptor bounds the dot product with each block’s pre-RoPE keys. It is not an upper bound on the model’s post-RoPE attention logits. At $B = 3 2$ with shared selection, increasing its scoring temperature from 16 to 64, 256, or 1,024 gives MK-3 scores of 19, 20, and 19 percent, respectively. The mean-key descriptor’s temperature, swept at the same point over 4, 16, 64, 256 and 1,024, gives 46, 51, 49, 44 and 41 percent, so the baseline’s default temperature, ${ \sqrt { d } } = 1 6$ , is the best of the five, and none moves the baseline towards semantic blocks.

Table 21: The values behind Figure 11, Qwen3.5-4B and Qwen3.5-35B-A3B at 512K. A dash marks a point that was not measured.
<table><tr><td rowspan="2">u</td><td colspan="3">MK-3 (%) at K</td><td colspan="3">MK-2 (%) at K</td><td colspan="4">Prefill (s) at K</td></tr><tr><td>8K</td><td>16K 32K</td><td>64K</td><td>8K</td><td>16K 32K</td><td>64K</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td></tr><tr><td>Qwen3.5-4B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense</td><td></td><td>96.0</td><td></td><td></td><td>99.0</td><td></td><td></td><td></td><td>40.69</td><td></td></tr><tr><td>4,09631.0</td><td></td><td></td><td></td><td></td><td>76.0 91.0 94.0 32.0 62.0 90.0 100.0 12.66 14.05 16.78 23.47</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2,04858.0</td><td></td><td></td><td></td><td></td><td>88.0 92.7 97.0 40.0 75.0 95.0</td><td></td><td></td><td>99.013.4014.98 17.86 24.30</td><td></td><td></td></tr><tr><td>1,024 81.0</td><td></td><td>87.0 94.0</td><td></td><td></td><td>56.0 87.0 97.0</td><td></td><td>100.015.30</td><td>17.31 20.42 26.14</td><td></td><td></td></tr><tr><td>51282.0</td><td></td><td>88.0 92.0</td><td></td><td>一</td><td>73.0 86.0 97.0</td><td></td><td>99.018.97</td><td></td><td>22.21 25.5930.05</td><td></td></tr><tr><td colspan="2">Qwen3.5-35B-A3B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense</td><td></td><td>100.0</td><td></td><td></td><td>97.0</td><td></td><td></td><td>49.19</td><td></td><td></td></tr><tr><td>4,096</td><td>7.0</td><td>48.0 97.0</td><td></td><td></td><td>- 36.0 90.0 97.0</td><td></td><td></td><td>- 14.07 15.7019.18</td><td></td><td></td></tr><tr><td>2,04825.0</td><td></td><td>80.0 98.7</td><td></td><td>一</td><td>68.0 96.0 97.7</td><td></td><td>一</td><td>15.13 16.81 20.29</td><td></td><td></td></tr><tr><td>1,02463.0</td><td></td><td>95.0</td><td>一</td><td>一</td><td>82.097.0</td><td></td><td>一</td><td>17.18 19.78</td><td></td><td></td></tr><tr><td></td><td></td><td>512 94.0 100.0</td><td>一</td><td>一</td><td>92.0 99.0 99.0</td><td></td><td></td><td>- 23.56 27.5431.40</td><td></td><td>一</td></tr></table>

![](images/9ebf897ec6debdb21fb746bb333e95aa15aca285b0c4b47853d875bcc8b527d4.jpg)  
Figure 12: GPU time of one 512K prefill on Qwen3.5-35B-A3B by cluster size u and token budget K, drawn as Figure 7.

## F.5 COMPARISON WITH FLASHPREFILL V2

FlashPrefill V2 is the only practical training-free sparse prefill method we found whose released kernels support the 256-dimensional heads of Qwen3.5 on H200 (Fan et al., 2026a). We run it with the SGLang implementation released with it, on Qwen3.5-4B with the same 512K prompts and YaRN setting as our tables. It uses the settings of the launch example in its repository, except that all eight full-attention layers are sparse, as in our method. We vary its threshold α, which sets how many blocks it keeps. It always keeps the first blocks of the prompt as attention sinks. Both methods run in BF16. SGLang times are the median time to the first token over the 100 prompts, with prefill in chunks of 16,384 tokens, and our times are the prefill time of the latency prompt (Appendix C). FlashPrefill V2 is one run per threshold. Our points by budget are single runs timed in one session, as in Appendix F.2, and the point with the initial block is the mean over three clustering seeds.

Figure 14 and Table 26 report accuracy and latency on each engine. The dense baselines have similar latency and accuracy, but the engines and timing protocols differ. We therefore compare each method’s speedup over its own engine’s dense baseline. Our method is more accurate at comparable speedups on both tasks. Among the tested thresholds, FlashPrefill V2 reaches our MK-3 accuracy only with a 1.7× speedup, compared with 2.2× for our method at $K = L / 1 6$

Table 22: Qwen3.5-4B, MK-3 accuracy in percent $( n = 1 0 0 )$ and prefill speedup at each length and budget. The $K = L / 1 6$ column is the three-seed mean of the main table, and the other entries are single runs.
<table><tr><td>Prompt</td><td>Dense</td><td> $K = L / 8$ </td><td> $K = L / 1 6$ </td><td> $K = L / 3 2$ </td><td> $K = L / 6 4$ </td></tr><tr><td>256K</td><td> $1 0 0 \ ( 1 2 . 4 \mathrm { s } )$ </td><td> $9 7 \ ( 1 . 3 6 \times )$ </td><td> $9 7 . 7 \ ( 1 . 7 4 \times )$ </td><td> $8 8 \ ( 1 . 8 6 \times )$ </td><td> $4 2 \ : ( 1 . 9 8 \times )$ </td></tr><tr><td>512K</td><td> $9 6 \ ( 4 0 . 7 \mathrm { s } )$ </td><td> $9 5 \ ( 1 . 6 2 \times )$ </td><td> $9 2 . 7 \ ( 2 . 2 5 \times )$ </td><td> $8 9 \ ( 2 . 6 8 \times )$ </td><td> $5 7 \ ( 2 . 9 8 \times )$ </td></tr><tr><td>1M</td><td> $4 0 \ ( 1 4 4 . 1 \mathrm { s } )$ </td><td> $3 2 \ : ( 1 . 8 3 \times )$ </td><td> $2 8 . 3 \ ( 2 . 8 4 \times )$ </td><td> $2 4 \ : ( 3 . 6 8 \times )$ </td><td> $1 8 \ ( 4 . 3 3 \times )$ </td></tr></table>

![](images/a676a00fee988630f351a52bb6583d04bb083eb1587092585e0df5d441ad3231.jpg)

![](images/964fb1fd9fdabf0e938f51aeb1ef385b2618e733540335575ba2e6a89fe33163.jpg)

![](images/aa7bb56bb17fb84ce6de6ee1d93d3d93bb20768557eaacf8798599598fbc5d77.jpg)  
Figure 13: Quality against prefill time over token budgets from $L / 6 4$ to $L / 8$ on Qwen3.5-4B and Qwen3.5-35B-A3B.

## F.6 CENTROIDS FITTED ON A CORPUS

Table 2 compares centroids fitted to each prompt with centroids fitted once and reused, with 256 centroids per layer and KV head. All rows keep the original query and key representations and the same selection rule, so this comparison does not include SAAP’s learned query classifier (Mazare´ et al., 2025). Per-prompt results average three clustering seeds, and the other rows are single runs. Raw-corpus centroids severely reduce multi-key accuracy and assign over half of the common-word prompt to one cluster, which the runtime rejects, so the table marks that task as failed. Fitting once to twenty RULER-5 examples per task recovers much of the loss. This fitting set includes evaluation examples, so the row diagnoses corpus mismatch rather than held-out generalization. The first-64K row fits each prompt’s centroids on its first 64K tokens and assigns every later key to them, as the prefix index of Appendix B.5 does.

Table 23: Positional and semantic retrieval on Qwen3.5-4B at 512K, $K = L / 1 6 ,$ MK-3 and MK-2 accuracy in percent with $n = 1 0 0 .$ , without and with the fixed initial block of Appendix D.3. B is positional block size and g query sharing span. Min/max rows describe each positional block by its minimum and maximum keys, and their shared rows use the averaged query. Our per-token row is the mean over three clustering seeds. A dash denotes an unmeasured entry.
<table><tr><td colspan="3"></td><td colspan="2">MK-3 (%), initial block</td><td colspan="2">MK-2 (%), initial block</td><td></td></tr><tr><td>Index</td><td>B</td><td>g</td><td>Without</td><td>With</td><td>Without</td><td>With</td><td>Prefill (s)</td></tr><tr><td>Positional, shared</td><td>2,048</td><td>64</td><td>7</td><td>4</td><td>一</td><td>一</td><td>15.6</td></tr><tr><td></td><td>128</td><td>64</td><td>27</td><td>32</td><td>1</td><td></td><td>16.2</td></tr><tr><td></td><td>32</td><td>64</td><td>51</td><td>64</td><td>24</td><td>24</td><td>18.4</td></tr><tr><td></td><td>16</td><td>64</td><td>63</td><td>79</td><td>32</td><td>36</td><td>21.5</td></tr><tr><td></td><td>8</td><td>64</td><td>77</td><td>92</td><td>51</td><td>53</td><td>27.5</td></tr><tr><td></td><td>4</td><td>64</td><td>75</td><td>85</td><td>一</td><td></td><td>39.5</td></tr><tr><td></td><td>32</td><td>32</td><td>64</td><td>78</td><td>35</td><td>38</td><td>19.1</td></tr><tr><td></td><td>16</td><td>32</td><td>82</td><td>94</td><td>44</td><td>52</td><td>24.0</td></tr><tr><td></td><td>8</td><td>32</td><td>84</td><td>93</td><td>58</td><td>64</td><td>32.9</td></tr><tr><td></td><td>4</td><td>32</td><td>85</td><td>93</td><td>65</td><td>73</td><td>50.1</td></tr><tr><td>Positional, per query</td><td>32</td><td>16</td><td>78</td><td>85</td><td>51</td><td>54</td><td>23.8</td></tr><tr><td></td><td>2,048</td><td>1</td><td>10</td><td>9</td><td>一</td><td></td><td>16.1</td></tr><tr><td></td><td>128</td><td>1</td><td>49</td><td>51</td><td>34</td><td>37</td><td>33.1</td></tr><tr><td></td><td>32</td><td>1</td><td>82</td><td>89</td><td>70</td><td>69</td><td>84.9</td></tr><tr><td>Min/max, shared</td><td>2,048</td><td>64</td><td>6</td><td>6</td><td>一</td><td>一</td><td>一</td></tr><tr><td></td><td>32</td><td>64</td><td>24</td><td>21</td><td>一</td><td></td><td>18.7</td></tr><tr><td></td><td>16</td><td>64</td><td>26</td><td>25</td><td>一</td><td></td><td></td></tr><tr><td>Min/max, per query</td><td>32</td><td>1</td><td>44</td><td>44</td><td>一</td><td></td><td>87.7</td></tr><tr><td>Ours, per query</td><td>一</td><td>1</td><td>92.7</td><td>95.0</td><td>95.0</td><td>94.7</td><td>18.3</td></tr><tr><td>Ours, shared</td><td>一</td><td>64</td><td>26</td><td>21</td><td>11</td><td>18</td><td>16.4</td></tr><tr><td>Dense</td><td>一</td><td>一</td><td>96</td><td>一</td><td>99</td><td>一</td><td>40.7</td></tr></table>

Table 24: Positional and semantic retrieval on Qwen3.5-35B-A3B at 512K, $K = L / 1 6 .$ , without the initial block, MK-3 and MK-2 accuracy in percent with $n = 1 0 0$ (ours on clustering seed 0) and 512K prefill time.
<table><tr><td>Index</td><td>B</td><td>g</td><td>MK-3 (%)</td><td>MK-2 (%)</td><td>Prefill (s)</td></tr><tr><td rowspan="4">Positional, shared</td><td>16</td><td>64</td><td>78</td><td>58</td><td>30.8</td></tr><tr><td>32</td><td>32</td><td>84</td><td>63</td><td>22.5</td></tr><tr><td>16</td><td>32</td><td>93</td><td>66</td><td>27.4</td></tr><tr><td>8</td><td>32</td><td>99</td><td>84</td><td>37.1</td></tr><tr><td>Positional, per query</td><td>128</td><td>1</td><td>55</td><td>47</td><td>34.4</td></tr><tr><td>Ours, per query</td><td>一</td><td>1</td><td>99</td><td>97</td><td>20.4</td></tr><tr><td>Dense</td><td>一</td><td>一</td><td>100</td><td>97</td><td>49.2</td></tr></table>

Table 25: Positional and semantic retrieval on Qwen3.5-4B at 512K with the smaller budget $K =$ $L / 3 2 .$ accuracy in percent with $n = 1 0 0$ , without and with the fixed initial block, and 512K prefill time without it.
<table><tr><td colspan="3"></td><td colspan="2">MK-3 (%), initial block</td><td colspan="2"> $\mathrm { M K } { \cdot } 2 ( \% ) ,$  initial block</td><td rowspan="2">Prefill (s)</td></tr><tr><td>Index</td><td>B</td><td>g</td><td>Without</td><td>With</td><td>Without</td><td>With</td></tr><tr><td rowspan="4">Positional, shared</td><td>16</td><td>32</td><td>71</td><td>83</td><td>32</td><td>36</td><td>22.1</td></tr><tr><td>8</td><td>64</td><td>61</td><td>81</td><td>39</td><td>48</td><td>26.8</td></tr><tr><td>8</td><td>32</td><td>72</td><td>88</td><td>49</td><td>60</td><td>32.0</td></tr><tr><td>4</td><td>32</td><td>64</td><td>79</td><td>58</td><td>62</td><td>51.5</td></tr><tr><td>Positional, per query</td><td>32</td><td>1</td><td>68</td><td>69</td><td>47</td><td>50</td><td>一</td></tr><tr><td>Ours, per query,  $u = 2 , 0 4 8$ </td><td>一</td><td>1</td><td>82.7</td><td>90.0</td><td>76.7</td><td>86.0</td><td>14.8</td></tr><tr><td>Ours, per query,  $u = 1 , 0 2 4$ </td><td></td><td>1</td><td>87.0</td><td>90.7</td><td>88.0 99</td><td>89.3</td><td>17.1</td></tr><tr><td>Dense</td><td>一</td><td>一</td><td>96</td><td>一</td><td></td><td>一</td><td>40.1</td></tr></table>

![](images/20773c04241dbe147efe6eb4e808a9e0765226b40eb6394af1cc57cbecd9ff7d.jpg)

![](images/f1574a5345f90e883b62003d4fe3c19f3857c0fe90f038ad58553532d993ea1f.jpg)  
Ours by budget K, vLLM (prefill time) FlashPrefill V2 by threshold α, SGLang (time to first token) Ours at L/16 with the initial block, vLLM Dense attention, SGLang (time to first token) Dense attention, vLLM (prefill time)

Figure 14: FlashPrefill V2 in SGLang and our method in vLLM on Qwen3.5-4B at 512K, against time on each engine (top) and against time relative to dense attention on the same engine (bottom). Labels are our budget K and the threshold α of FlashPrefill V2.  
Table 26: FlashPrefill V2 in SGLang and our method in vLLM on Qwen3.5-4B at 512K. Accuracy is on 100 prompts per task. Times are the prefill time of the latency prompt on vLLM and the median time to the first token over the 100 prompts on SGLang.
<table><tr><td colspan="2"></td><td colspan="2">Accuracy (%)</td><td colspan="2">Time at 512K (s)</td></tr><tr><td colspan="2">Method</td><td>Engine</td><td>MK-3</td><td>MK-2</td><td>MK-3 run MK-2 run</td></tr><tr><td>Dense attention</td><td>vLLM</td><td>96.0</td><td>99.0</td><td>40.18</td></tr><tr><td>Ours, K = L/32</td><td>vLLM</td><td>88.0</td><td>75.0</td><td>14.98</td></tr><tr><td>Ours, K = L/16</td><td>vLLM</td><td>93.0</td><td>94.0</td><td>17.86</td></tr><tr><td>Ours, K = L/8</td><td>vLLM</td><td>97.0</td><td>99.0</td><td>24.30</td></tr><tr><td>Ours, K = L/16, initial block</td><td>vLLM</td><td>95.0</td><td>94.7</td><td>18.69</td></tr><tr><td>Dense attention</td><td>SGLang</td><td>97.0</td><td>99.0</td><td>43.25 43.70</td></tr><tr><td>FlashPrefill V2, α = 0.2</td><td>SGLang</td><td>25.0</td><td>39.0</td><td>13.09 13.42</td></tr><tr><td>FlashPrefill V2, α = 0.1</td><td>SGLang</td><td>49.0</td><td>49.0</td><td>14.03 15.08</td></tr><tr><td>FlashPrefill V2, α = 0.05</td><td>SGLang</td><td>81.0</td><td>59.0</td><td>16.23 17.15</td></tr><tr><td>FlashPrefil1 V2, α = 0.02</td><td>SGLang</td><td>88.0</td><td>79.0</td><td>20.82 24.09</td></tr><tr><td>FlashPrefill V2, α = 0.01</td><td>SGLang</td><td>94.0</td><td>84.0</td><td>25.37 29.83</td></tr></table>

## G SOURCES OF QUALITY LOSS AND TRAINING BASELINES

## G.1 ATTENTION MASS MISSED BY CLUSTER SCORING

Cluster selection can miss high-attention tokens because centroid scores misrank clusters or because whole clusters also contain low-attention tokens. We examine these effects by comparing centroidbased selection with dense-attention rankings, using the same partition.

We build the partition from the dense model’s keys and compare three selections for the dense model’s queries. The first applies our selection rule to the fitted clusters. The second sums the dense attention probabilities within each cluster and greedily selects whole clusters under the same budget, ranking by total mass or by mass per token and keeping the better of the two for each query and GQA group. The third selects the K tokens that receive the most dense attention, without reading whole clusters. We call the last two dense-ranked clusters and dense-ranked tokens. Both need dense attention at inference time, so they serve only as references.

Coverage is the percentage of dense attention to the prompt retained by the selected tokens. Each query’s current chunk is always read exactly and does not count toward the budget. For generated queries, we count only attention to prompt tokens.

The analysis uses ten prompts per task at 512K, with the last 64 prompt positions and up to 32 generated tokens as queries, over the 4B model’s eight full-attention layers and four KV heads. The ten MK-3 prompts come from a separate set of 400 generated prompts rather than from the MK-3 items of our tables, and MRCR uses the first ten prompts of its 512K bin. The budgets are $L / 1 6$ and $L / 8$ of each prompt’s actual length. Dense probabilities are summed over a GQA group to rank tokens or clusters, and the retained mass is normalized by each query head’s attention on the prompt and averaged over heads, queries, layers, and prompts. Because heads can put different total attention on the prompt during generation, the ranking and the reported coverage are not exactly the same quantity. These queries emphasize retrieval near the end of the prompt and during generation, not the average prefill query.

At $K = L / 1 6$ , dense-ranked tokens retain more attention mass than dense-ranked clusters on all three tasks (Figure 15). Cluster selections can exceed the budget by part of a cluster, but giving the token reference the same number of tokens changes its coverage by at most 0.3 points. Better cluster scores therefore recover only part of the mass lost when selection is restricted to whole clusters. Using our centroid scores with $\bar { K } = L / 8$ also gives higher coverage than dense-ranked clusters with $K = L / 1 6$ on all three tasks (Table 27).

We also replace the centroid scores inside the sparse runtime and measure RULER accuracy (Table 28). At every full-attention layer, we compute dense attention probabilities from the current queries and prompt keys in FP32, sum them within each cluster, and select clusters in that order under the runtime’s budget rule. The clustering, the regions read without selection, the attention computation, and the handling of generated tokens are unchanged. Queries and keys come from this sparse run, so the comparison includes the effect of earlier sparse layers on the model states. We use the first 50 prompts of each multi-key task at 512K, with clustering seed 0 for the dense-attention scores and the mean and standard deviation over three seeds for the centroid scores.

Using dense-attention scores recovers dense accuracy on MK-2 but changes MK-3 little (Table 28). The effect of missing attention mass depends on which tokens are missed. A token carrying answer information matters more than one repeating information available elsewhere.

## G.2 LEARNING THE ROUTING MAP

We test two simple training baselines. The first learns linear maps for selection while leaving the model weights and attention kernels unchanged. For each full-attention layer, we train a query map $W _ { q } \in \mathbb { R } ^ { d \times \breve { d } }$ per query head and a key map $\breve { W } _ { k } \in \mathbb { R } ^ { d \times d }$ per KV head, initialized to the identity. We also train the temperature τ and size penalty $\beta .$ The penalty is clamped to nonnegative values in the forward pass with a straight-through gradient, matching its use at inference time.

For each training document, we fit clusters on the mapped keys and sum the dense model’s attention probabilities within each cluster. The loss is the KL divergence from this cluster-mass distribution to the normalized selection scores in Equation 2, with an L2 penalty keeping the maps near the identity. Cluster assignments are recomputed without gradients. The normalized means of the assigned mapped keys carry gradients to $\dot { W } _ { k }$

Table 27: Mean prompt-attention coverage on ten 512K examples per task, Qwen3.5-4B.
<table><tr><td colspan="2"></td><td colspan="3">Prompt attention retained (%)</td></tr><tr><td>Task</td><td>K</td><td>Our scores</td><td>Dense-ranked clusters</td><td>Dense-ranked tokens</td></tr><tr><td>MK-2</td><td>L/16</td><td>77.4</td><td>80.8</td><td>90.4</td></tr><tr><td>MK-2</td><td>L/8</td><td>84.6</td><td>87.3</td><td>94.3</td></tr><tr><td>MK-3</td><td>L/16</td><td>76.9</td><td>80.5</td><td>91.9</td></tr><tr><td>MK-3</td><td>L/8</td><td>84.0</td><td>86.8</td><td>95.1</td></tr><tr><td>MRCR-8</td><td>L/16</td><td>86.3</td><td>88.2</td><td>94.2</td></tr><tr><td>MRCR-8</td><td> $L / 8$ </td><td>91.0</td><td>92.2</td><td>96.6</td></tr></table>

![](images/e85e1bea5dd6daecb68cdbc744510e5195f04eff847d372f9644016bf0eb7143.jpg)  
Figure 15: Prompt-attention coverage by full-attention layer at $K = L / 1 6$ under our centroid scores, dense-ranked clusters, and dense-ranked tokens, on ten 512K examples per task, for Qwen3.5-4B (top, eight layers) and Qwen3.5-35B-A3B (bottom, ten). All curves use dense-model states.

Training uses 25,684 raw-text documents of 65,536 tokens each, about 1.7B tokens, drawn from the PDF documents of the Dolma 3 Longmino mix (Team Olmo, 2025). We leave out the mix’s synthetic common-word extraction and rewriting tasks, which resemble RULER tasks. We train for 3,000 steps with eight data-parallel workers, a total batch size of eight documents, and learning rate $1 0 ^ { - 3 }$ . The objective is evaluated on eight held-out documents and largely plateaus after roughly half the run. For downstream evaluation, RULER scores are percentages averaged over three clustering seeds, with standard deviations in parentheses. MRCR-8 reports the mean match ratio over 100 examples.

Training reduces the held-out KL by about two thirds, but changes the attention mass retained by the selected clusters much less (Table 29). Much of the KL reduction accompanies a lower temperature, which sharpens the score distribution, while the size penalty reaches zero. The objective favors clusters receiving more total attention mass, which can reward large clusters and conflict with the runtime’s size penalty. Allowing a negative penalty reaches a similar objective.

Table 28: Cluster scoring on Qwen3.5-4B at 512K, $u = 2 { , } 0 4 8 .$ $K = L / 1 6$ . Accuracy (%) on 50 examples per task.
<table><tr><td>Selection</td><td>MK-3</td><td>MK-2</td></tr><tr><td>Dense attention</td><td>96.0</td><td>98.0</td></tr><tr><td>Centroid scores</td><td>93.3 (1.2)</td><td>96.0 (0.0)</td></tr><tr><td>Dense attention mass</td><td>94.0</td><td>98.0</td></tr></table>

Table 29: Learning query and key maps for clustering and selection on Qwen3.5-4B. The left columns report held-out KL loss, retained attention mass, and selection parameters. The right columns report long-context task performance.
<table><tr><td></td><td colspan="2">Objective</td><td colspan="2">Rule</td><td colspan="2">512K</td><td colspan="2">RULER-5</td><td>MRCR-8</td></tr><tr><td>Configuration</td><td>KL</td><td>mass at L/16</td><td>T</td><td>β</td><td>MK-3</td><td>MK-2</td><td>256K</td><td>512K</td><td>512K</td></tr><tr><td>Dense attention</td><td></td><td></td><td></td><td></td><td>96.0</td><td>99.0</td><td>84.7</td><td>82.8</td><td>0.381</td></tr><tr><td>Identity map (ours)</td><td>1.07</td><td>0.674</td><td>0.10</td><td>0.25</td><td>92.7 (0.6)</td><td>95.0 (1.0)</td><td>82.7 (0.1)</td><td>81.8 (0.5)</td><td>0.405</td></tr><tr><td>Learned  $\tilde { W _ { q } } , W _ { k } , \tau , \beta$ </td><td>0.31</td><td>0.695</td><td>0.036</td><td>0</td><td>91.0 (1.0)</td><td>94.7 (0.6)</td><td></td><td>83.1 (0.2) 81.5 (0.4)</td><td>0.402</td></tr></table>

The learned map gives similar aggregate RULER-5 and MRCR-8 scores to the identity map, but its 512K MK-3 accuracy is about two points lower. Earlier runs used a corpus 8.5 times smaller and a quarter to a third of the training steps, and tested an unconstrained penalty, the same map with the penalty clamped at evaluation, and a query-only map. None gave a consistent retrieval improvement.

## G.3 ADAPTING THE MODEL TO ITS SPARSE ATTENTION WITH LORA

The second baseline adapts the model weights to the sparse attention. A copy of the model with rank-8 LoRA adapters (Hu et al., 2022) runs sparse attention and is trained to match the dense model’s next-token distribution by forward KL over the full vocabulary. One configuration adapts the query, key, and output projections of the eight full-attention layers. The other also adapts the MLP and Gated DeltaNet projections. The index and the selection are recomputed from the adapted model’s current keys and queries with the same code as at inference. Gradients pass through attention over the selected tokens, but not through the discrete selection.

Each run uses 300 steps with four documents per step, drawn from the corpus of the learned map above. The documents are 64K tokens long, and the loss is computed at every position. As an implementation check, the model with zero-initialized adapters reproduces the dense model when both use dense attention. For evaluation, adapters are merged into the weights and the sparse runtime is unchanged. Each task uses 100 examples and one clustering seed.

The held-out distillation objective falls substantially during training, but the task results do not improve (Table 30). At 512K, MK-2 stays close to the training-free configuration, while MK-3 and MRCR-8 are lower. These runs use raw text and documents shorter than the evaluation prompts. Other work on adapting models to sparse attention also trains on synthetic long-context tasks (Zhou et al., 2026), which we do not test.

Table 30: Adapting Qwen3.5-4B to sparse attention with LoRA trained on 64K documents. RULER multi-key scores are accuracy (%), and MRCR-8 reports a mean match ratio. All configurations use the same sparse runtime and token budget.
<table><tr><td></td><td colspan="2">128K</td><td colspan="3">512K</td></tr><tr><td>Adapters</td><td>MK-2</td><td>MK-3</td><td>MK-2</td><td>MK-3</td><td>MRCR-8</td></tr><tr><td>Dense</td><td>100</td><td>100</td><td>99</td><td>96</td><td>0.381</td></tr><tr><td>Ours, no training</td><td>98</td><td>100</td><td>94</td><td>93</td><td>0.405</td></tr><tr><td>Attention</td><td>99</td><td>99</td><td>92</td><td>91</td><td>0.298</td></tr><tr><td>+MLP+GDN</td><td>100</td><td>99</td><td>95</td><td>90</td><td>0.291</td></tr></table>

## H OTHER ARCHITECTURES

This section provides the full results for Qwen3.5-35B-A3B and Nemotron 3.5 Lightning.

## H.1 QWEN3.5-35B-A3B

Qwen3.5-35B-A3B has 40 layers, including ten full-attention layers with 16 query heads, two KV heads, and head dimension 256. It uses 256 experts with eight active and shares the 4B model’s tokenizer. The sparse implementation uses the same query and key representations, clustering settings, and selection rule as the 4B model. All sparse quality measurements use sliced prefill. Latency measurements use it from 512K onward, where single-pass prefill exceeds device memory. The dense baseline uses chunked prefill.

Table 9 gives the task scores beside the 4B model’s. Figure 16 shows prefill and decode latency, with values in Table 31.

RULER-5 remains close to dense attention through 1M. Both multi-key tasks also remain close through 512K. At 1M, our method loses accuracy on MK-3 and common-word extraction but is above dense attention on MK-2 and HotpotQA. Halving or quartering the budget at 512K substantially reduces multi-key accuracy. The QA and common-word results of those runs used capped prompts and are not shown.

The 512K prefill profile in Figure 19 shows that the unchanged expert and Gated DeltaNet layers account for much of the remaining sparse prefill time. Because the sparse prefill is sliced at this length, these layers run in 32K slices, and the mixture-of-experts kernels are the largest part of the time not attributed to attention.

For long generations, reading the entire generated suffix exactly increases decode cost. After a 512K prompt, the decode step grows from 4.9 ms per token at the start to 10.5 ms between 32K and 64K generated tokens, above dense attention’s 9.1 ms, while flushing keeps it near 5.1 ms. Flushing avoids this growth after both 512K and 1M prompts, with similar aggregate LongBench-v2 accuracy in the measured runs (Tables 32 and 33).

Table 33 compares the main budget with half that budget on MRCR-8 and LongBench-v2 in single runs. MRCR-8 at the main budget stays close to dense attention through 256K and falls below it at 512K. Halving the budget lowers MRCR-8 most at 256K. LongBench-v2 scores change less across the sparse configurations, and flushing changes aggregate accuracy by less than two percentage points.

The coverage analysis shows the same ordering as on the 4B model (Figure 15). Mean coverage is slightly lower on the 35B-A3B model even though its task-accuracy loss is smaller.

## H.2 NEMOTRON 3.5 LIGHTNING

Nemotron 3.5 Lightning 30B-A3B combines Mamba-2 and MoE layers with six full-attention layers, each with two KV heads and no rotary embedding. We replace these attention layers and retain the original model weights. We also slice non-attention layers to avoid an offset overflow in the Mamba-2 convolution kernel.

![](images/73b5e6e99f9855f05a8abd40c33c2a41587197e06610ec45db8dd7a05a8fea17.jpg)

![](images/94139f04106416a332ae38c6520e9e31ce71923a3fc63a01c309a948a208645e.jpg)

![](images/d09aa0643be8cb2f8b5e29d2df04756b2f3b3db4ea270e67b84a431de6d9f4d5.jpg)

![](images/bac3f0bc7edec5a9a3bd97e0fedeba41f91ca9c1c9e5933ea7e64ca6cedc41fa.jpg)  
Figure 16: Qwen3.5-35B-A3B latency on one H200 at $K = L / 1 6 .$ The columns show whole-model wall-clock latency (left) and GPU time over the ten full-attention layers (right). The rows show prefill (top) and decoding (bottom).

Table 31: Qwen3.5-35B-A3B latency on one H200 at $K = L / 1 6$ and $u = 2 , 0 4 8 .$ , as Table 13. Wholemodel columns are wall clock, the attention-layer columns profiler totals over the ten full-attention layers. The last rows halve and quarter the budget at 512K.
<table><tr><td colspan="2"></td><td colspan="2">Whole model</td><td colspan="2">Attention layers (10)</td></tr><tr><td>Prompt</td><td>Configuration</td><td>Prefill (s)</td><td>Decode (ms)</td><td>Prefill (s)</td><td>Decode (ms)</td></tr><tr><td>128K</td><td>Dense</td><td>4.75</td><td>5.72</td><td>2.41</td><td>0.92</td></tr><tr><td>128K</td><td>Ours</td><td>3.31</td><td>4.91</td><td>0.96</td><td>0.30</td></tr><tr><td>256K</td><td>Dense</td><td>14.77</td><td>6.64</td><td>9.58</td><td>1.56</td></tr><tr><td>256K</td><td>Ours</td><td>7.76</td><td>4.98</td><td>2.69</td><td>0.37</td></tr><tr><td>512K</td><td>Dense</td><td>49.19</td><td>8.68</td><td>39.25</td><td>2.98</td></tr><tr><td>512K</td><td>Ours</td><td>20.73</td><td>5.15</td><td>9.68</td><td>0.54</td></tr><tr><td>1M</td><td>Dense</td><td>173.49</td><td>12.87</td><td>156.24</td><td>6.14</td></tr><tr><td>1M</td><td>Ours</td><td>57.42</td><td>5.71</td><td>36.38</td><td>0.84</td></tr><tr><td>512K</td><td> $\mathrm { O u r s } , K = L / 3 2$ </td><td>17.21</td><td>5.03</td><td>一</td><td>一</td></tr><tr><td>512K</td><td> $\mathrm { O u r s } , K = L / 6 4$ </td><td>15.21</td><td>4.96</td><td></td><td>一</td></tr></table>

For routing, we center keys and use 32 clustering iterations with at least 128 sampled keys per centroid. These settings were selected on a 64K pilot and retained for longer evaluations (Table 34). Centering addresses a strong shared direction in the keys that otherwise leads to cluster collapse. It changes only routing representations, not the KV used for attention.

Figure 17 draws the latency columns of Table 35. Both decode paths use full-step graphs. The nominal 1M latency prompt is 1,028,096 tokens, while the 1M task evaluations use the 1,048,576- token RULER prompts.

At $u = 2 { , } 0 4 8$ and $K = L / 1 6$ , prefill is 1.8× faster than dense attention at 256K and approaches three times as fast at 1M. The MK-3 loss is substantial at every length, however. Decode gains are smaller because the replaced attention layers account for less of the original decode step.

Figure 11 (bottom row) and Table 36 show the MK-3 and MK-2 grid at 512K, including $u = 2 5 6$ and 128. Points below 30% accuracy are not drawn, and the ringed point is the main setting, $u = 2 , 0 4 8$ and $K = 6 5 { , } 5 3 6$ . For $u = 2 5 6$ and 128, KV-outer gathering uses batches of 512 clusters to fit in device memory.

Table 32: Qwen3.5-35B-A3B, mean decode time after a 1M prompt in milliseconds per token, over 64 and over 65,536 generated tokens, $K = L / 1 6 ,$ , one H200.
<table><tr><td>Configuration</td><td>64</td><td>65,536</td></tr><tr><td>Dense</td><td>12.90</td><td>13.42</td></tr><tr><td>Ours</td><td>5.57</td><td>9.21</td></tr><tr><td>Ours, flushed every  $1 \mathrm { K } \left( W = 5 1 2 \right)$ </td><td>5.55</td><td>5.49</td></tr><tr><td>Ours, flushed every 8K  $( W = 2 , 0 4 8 )$ </td><td>5.57</td><td>5.91</td></tr></table>

Table 33: Qwen3.5-35B-A3B on MRCR-8 (mean match ratio) and LongBench-v2 (accuracy over the same 470 items), thinking on, at $K = L / 1 6$ and at half of it. Flushing uses $F = 1 { , } 0 2 4$ and $\dot { W } = 5 1 2$ (Appendix E.5).
<table><tr><td rowspan="2"></td><td colspan="3">MRCR-8</td><td colspan="3">LongBench-v2</td></tr><tr><td>128K</td><td>256K</td><td>512K</td><td>≤256K</td><td>&gt;256K</td><td>all</td></tr><tr><td>Dense</td><td>0.849</td><td>0.644</td><td>0.500</td><td>65.3</td><td>65.7</td><td>65.3</td></tr><tr><td>Ours,  $K = L / 3 2$ </td><td>0.835</td><td>0.596</td><td>0.463</td><td>64.3</td><td>65.7</td><td>64.5</td></tr><tr><td>Ours,  $K = L \dot { } / 1 6$ </td><td>0.839</td><td>0.682</td><td>0.437</td><td>62.3</td><td>64.2</td><td>62.6</td></tr><tr><td>Ours,  $K = L { \dot { / } } 1 6 ,$  flushed</td><td>一</td><td>一</td><td>一</td><td>63.0</td><td>71.6</td><td>64.3</td></tr></table>

At a fixed budget, accuracy improves as cluster size decreases to 512 tokens. At similar prefill times, however, 512- or 1,024-token clusters with a larger budget are more accurate than 256- or 128-token clusters. Nemotron therefore benefits from finer clusters than the Qwen models. The main results keep $u = 2 , 0 4 8 .$ , as on the Qwen models. At $K = L / 8$ , changing to u = 1,024 raises MK-2 accuracy by 10 points for about 7% more prefill time.

Figure 18 breaks down prefill time by cluster size. Smaller clusters mainly increase the time spent on attention over selected blocks, gathering, and merging, but the increase is smaller than on the Qwen models.

Table 9 reports all five RULER-5 tasks at 512K alongside the Qwen models. Unlike the Qwen models, Nemotron loses substantial accuracy on multi-value retrieval as well as multi-key retrieval. Doubling the budget recovers much of this loss while retaining prefill speedups of 1.5–2.0× from 256K to 1M (Table 35). The budget increase has a smaller effect on decode time.

Table 34: Nemotron 3.5 Lightning, the 64K pilot behind the clustering settings, with MK-3 accuracy in percent on 20 items, $\bar { K { \bf \Phi } } = L \bar { / } 1 6$ . Centering subtracts the mean key before routing, whitening scales the centered keys by their covariance, and iterations and samples are the k-means fit.
<table><tr><td>Routing representation</td><td>MK-3</td></tr><tr><td>Dense</td><td>100</td></tr><tr><td>Uncentered keys</td><td>20</td></tr><tr><td>Centered, 32 iterations, 128 samples per centroid</td><td>100</td></tr><tr><td>Whitened</td><td>40</td></tr><tr><td>Whitened, k-means++ seeding, 32 iterations</td><td>35</td></tr></table>

Table 35: Nemotron 3.5 Lightning latency on one H200 and MK-3 accuracy in percent $( n = 1 0 0 )$ by prompt length, at $K = L / 1 6$ unless noted. The attention-layer columns are profiler totals over the six full-attention layers.
<table><tr><td rowspan="2">Prompt Configuration</td><td rowspan="2"></td><td rowspan="2">MK-3 (%)</td><td colspan="2">Whole model</td><td colspan="2">Attention layers (6)</td></tr><tr><td>Prefill (s)</td><td>Decode (ms)</td><td>Prefill (s)</td><td>Decode (ms)</td></tr><tr><td>256K</td><td>Dense</td><td>100</td><td>11.07</td><td>3.92</td><td>6.91</td><td>0.51</td></tr><tr><td>256K</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8$ </td><td>73</td><td>6.15</td><td>3.39</td><td>1.94</td><td>0.17</td></tr><tr><td>256K</td><td> $\mathrm { O u r s } , u = 1 , 0 2 4$ </td><td>93</td><td>6.82</td><td>3.38</td><td>2.34</td><td>0.18</td></tr><tr><td>256K</td><td> ${ \mathrm { O u r s } } , u = 5 1 2$ </td><td>98</td><td>7.68</td><td>3.42</td><td>3.07</td><td>0.22</td></tr><tr><td>256K</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8 , K = L / 8$ </td><td>96</td><td>7.59</td><td>3.44</td><td>3.15</td><td>0.22</td></tr><tr><td>512K</td><td>Dense</td><td>100</td><td>35.72</td><td>4.44</td><td>28.32</td><td>0.88</td></tr><tr><td>512K</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8$ </td><td>84</td><td>15.33</td><td>3.59</td><td>6.95</td><td>0.23</td></tr><tr><td>512K</td><td> $\mathrm { O u r s } , u = 1 , 0 2 4$ </td><td>87</td><td>17.30</td><td>3.54</td><td>8.47</td><td>0.27</td></tr><tr><td>512K</td><td> ${ \mathrm { O u r s } } , u = 5 1 2$ </td><td>93</td><td>21.42</td><td>3.60</td><td>11.15</td><td>0.34</td></tr><tr><td>512K</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8 , K = L / 8$ </td><td>90</td><td>20.70</td><td>3.63</td><td>12.06</td><td>0.32</td></tr><tr><td>1M</td><td>Dense</td><td>100</td><td>120.18</td><td>5.37</td><td>104.12</td><td>1.64</td></tr><tr><td>1M</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8$ </td><td>70</td><td>41.09</td><td>3.65</td><td>24.54</td><td>0.35</td></tr><tr><td>1M</td><td> $\mathrm { O u r s } , u = 2 , 0 4 8 , K = L / 8$ </td><td>85</td><td>61.62</td><td>3.87</td><td>44.74</td><td>0.51</td></tr></table>

![](images/9befebb3f2affb3dd7f69fb68d1d85d5af7f456a98cf677ff7b7cd9a9c66e09e.jpg)  
Figure 17: Nemotron 3.5 Lightning, latency against prompt length on one H200 at $K = L / 1 6$ for three target mean block sizes $u ,$ with the doubled budget $\dot { K } = \bar { L } / 8 \mathrm { a t } u = 2 { , } 0 4 8$ dashed. The left column is the whole model by wall clock and the right column the six full-attention layers by profiler accounting. The rows are prefill and decoding. The brackets are dense over ours at $u = 2 , 0 4 8$ and $K = L / 1 6$ , under the conditions of Table 35.

![](images/290c4c9417436e01055ef6141c4486dff90a385e828dd0bca550f4b8fd37203d.jpg)  
Figure 18: GPU time of one 512K prefill on Nemotron 3.5 Lightning at $K = L / 1 6$ by cluster size u, drawn as Figure 7.

Table 36: The values behind the bottom row of Figure 11, Nemotron 3.5 Lightning at 512K.
<table><tr><td rowspan="2">u</td><td colspan="4">MK-3 (%) at K</td><td colspan="4">MK-2 (%) at K</td><td colspan="4">Prefill (s) at K</td></tr><tr><td></td><td>8K 16K  32K</td><td></td><td>64K</td><td>8K  16K  32K  64K</td><td></td><td></td><td></td><td>8K</td><td>16K</td><td>32K</td><td>64K</td></tr><tr><td>Dense</td><td></td><td></td><td>100.0</td><td></td><td></td><td></td><td>99.0</td><td></td><td></td><td></td><td>35.72</td><td></td></tr><tr><td>4,096</td><td></td><td>3.0 19.0 60.0</td><td></td><td>89.0</td><td></td><td></td><td></td><td></td><td></td><td>6.0 14.0 32.0 62.0 11.30 12.46 14.80 19.96</td><td></td><td></td></tr><tr><td></td><td></td><td>2,048 11.0 35.0 84.0</td><td></td><td>90.0</td><td></td><td></td><td></td><td></td><td>7.0 20.0 49.0 81.0 11.98</td><td>13.26 15.72 20.75</td><td></td><td></td></tr><tr><td></td><td></td><td>1,024 25.0 65.0 87.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td>94.0 17.0 33.0 72.0 91.0 12.95</td><td>14.55</td><td>517.2122.24</td><td></td></tr><tr><td></td><td></td><td>512 56.0 79.0 93.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>97.0 34.0 62.0 83.0 95.0 14.81 16.83 20.09 25.09</td><td></td><td></td></tr><tr><td></td><td></td><td>256 81.0 86.0 93.0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>99.0 62.0 78.0 92.0 98.0 19.86 22.60 26.64 31.29</td><td></td><td></td></tr><tr><td>128 82.0 89.0 97.0 100.0 68.0 87.0 96.0 96.0 30.10 34.25 40.17 44.82</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/95eb916e343bde7e1af11b8674ca84bd227061f1d94a30572e0db75a5f385061.jpg)

![](images/43d95bdb74cf99ef98b0e1aae42ca7cf2141096cc453d14bd5ec50995e7cfd72.jpg)  
Figure 19: The 512K prefill profile of Qwen3.5-35B-A3B at $K = L / 1 6$ (left) and of Nemotron 3.5 Lightning at $u = 2 { , } 0 4 8$ (right), drawn as Figure 4, with GPU seconds by kernel group from one profiled prefill of each configuration. Nemotron uses different slicing schedules for non-attention layers in the dense and sparse runs.

## I PRIOR SPARSE-ATTENTION METHODS

Table 37 summarizes selected sparse-attention methods and their evaluations. Here, hybrid models combine full attention with linear-attention or state-space layers. SpotAttention keeps prefill dense and evaluates accuracy on hybrid Qwen3.5 models (Ahmad & Yun, 2026). It reports decode throughput on Qwen3-8B. UniPrefill also evaluates several training-free prefill methods on hybrid models up to 128K tokens (Fan et al., 2026b).

Table 37: Sparse-attention methods as their papers describe and evaluate them. The groups are training-free methods aimed mainly at prefill, training-free methods aimed mainly at decoding, and methods with learned components. Learned components are those trained for sparsity beyond the pretrained model. Whole-model time means that the paper times the whole model, such as end-to-end latency, time to first token, or time per output token, rather than only the attention layer or kernel. DSA reports token costs derived from deployed-service benchmarks.
<table><tr><td>Method</td><td>Learned components</td><td>Sparse phases</td><td>Hybrid model</td><td>Whole- model time</td></tr><tr><td>MInference (Jiang et al., 2024)</td><td>None</td><td>Prefill</td><td>No</td><td>Yes</td></tr><tr><td>FlexPrefill (Lai et al., 2025)</td><td>None</td><td>Prefill</td><td>No</td><td>No</td></tr><tr><td>FlashPrefill / V2 (Fan et al.,</td><td>None</td><td>Prefill</td><td>No</td><td>Yes</td></tr><tr><td>2026a;c) QuoKA (Jones et al., 2026)</td><td>None</td><td>Prefill, decode</td><td>No</td><td>Yes</td></tr><tr><td>Quest (Tang et al., 2024)</td><td>None</td><td>Decode</td><td>No</td><td>Yes</td></tr><tr><td>Squeezed (Hooper et al., 2025)</td><td>None</td><td>Prefill, decode</td><td>No</td><td>No</td></tr><tr><td>ClusterKV (Liu et al., 2025b)</td><td>None</td><td>Decode</td><td>No</td><td>Yes</td></tr><tr><td>Tactic (Zhu et al., 2026)</td><td>None</td><td>Decode</td><td>No</td><td>Yes</td></tr><tr><td>SAAP (Mazaré et al., 2025)</td><td>Query classifier</td><td>Decode</td><td>No</td><td>No</td></tr><tr><td>SpotAttention (Ahmad &amp; Yun, 2026)</td><td>Indexer</td><td>Decode</td><td>Yes</td><td>Yes</td></tr><tr><td>RTPurbo (Zhou et al., 2026)</td><td>Indexer and model</td><td>Prefill, decode</td><td>No</td><td>No</td></tr><tr><td>MoBA / FlashMoBA (Lu et al., 2025; Xiao et al., 2025a)</td><td>Model</td><td>Prefill</td><td>No</td><td>No</td></tr><tr><td>DSA (DeepSeek-AI, 2025)</td><td>Indexer and model</td><td>Prefill, decode</td><td>No</td><td>Service cost</td></tr><tr><td>MSA (Lai et al., 2026)</td><td>Indexer and model</td><td>Prefill, decode</td><td>No</td><td>No</td></tr><tr><td>QSA (Qiu et al., 2026)</td><td>Indexer and model</td><td>Prefill, decode</td><td>Yes</td><td>No</td></tr><tr><td>HISA (Xu et al., 2026)</td><td>Indexer and model</td><td>Prefill, decode</td><td>No</td><td>Yes</td></tr><tr><td>This work</td><td>None</td><td>Prefill, decode</td><td>Yes</td><td>Yes</td></tr></table>

ClusterKV and Tactic build a key index during dense prefill for subsequent decoding (Liu et al., 2025b; Zhu et al., 2026). Squeezed builds its index offline for a fixed context (Hooper et al., 2025). Our method builds an index within each request and uses it for the prompt’s own prefill. Our corpusfitted centroids in Section 5.4 are scored with the model’s queries, whereas SAAP learns a query classifier (Mazare et al., 2025).´

BLASST skips softmax and attention–value computation for low-scoring blocks (Yuan et al., 2026). Sketch&Walk combines sketched scores across layers to select blocks (Le et al., 2026). Kascade and IndexCache reuse token selections across layers (Deshmukh et al., 2025; Bai et al., 2026).

## J EXAMPLES OF CLUSTER MEMBERS

Figure 3 shows one cluster from an MRCR prompt and one from an MK-3 prompt. Each highlighted span is a single member token, with surrounding text for context. Cosine similarity is measured between the normalized pre-RoPE key and its assigned centroid. Each cluster comes from a partition of 256 clusters fitted to the unmodified model’s keys. Layer, KV-head, and token-position indices are zero-based. The figure shows the three nearest and three farthest members of each cluster among the saved examples.

The MRCR cluster groups environmental vocabulary across several passages. Its lower-similarity members extend to medicine, history, and mathematics. For MK-3, the nearest members occur at the beginning of key identifiers, but lower-similarity members also include the beginning of a value identifier.

In two MK-3 and two MRCR prompts, about 70% of MK-3 clusters and 90% of MRCR clusters have no single token type accounting for more than half their members. These counts use all cluster members across eight layers and four KV heads per prompt.

## J.1 FURTHER CLUSTERS

Figure 20 shows five further clusters, using three members from each end of the saved cosine range. The first and third clusters use the same MRCR prompt as Figure 3, the second and fourth use separate MRCR and MK-3 prompts, and the fifth uses an MK-2 prompt.

The first cluster groups occurrences of colors across descriptions, while its lower-similarity members extend to other words. The second includes punctuation and function words within a passage about color perception. The third groups pronouns and possessives across different topics. In the fourth, the nearest members are the digit 4 at the start of the third group of a key UUID. One lower-similarity member instead occurs in the second group. The fifth comes from an MK-2 prompt with lines such as “One of the special magic numbers for troubled-arena is: 8154823.” All ten saved nearest members of this cluster are negative adjectives in different keys, such as troubled, weary, abusive, disgusted, and ashamed. Its lowest-similarity members include other tokens of the same lines.

Cosine Position MRCR · layer 3, KV head 0, cluster 111   
0.942 116899 skies, You paint the horizons with colors that fly. Each river that flows, …   
0.941 117439 … organic art. Blossoms burst with colors bold, In hues of red and pink …   
0.939 13679 us into fall with its crisp air and colors. It's a time of change and …   
0.684 81616 animals showcasing warning colors or camouflage, and landscapes …   
0.651 231577 Binary numbers, composed of 0s and 1s, form the foundation of computer …   
0.627 122892 … and those with stainless steel blades to prevent rusting. Information on …   
Cosine Position MRCR · layer 11, KV head 1, cluster 208   
0.903 268600 From a psychological perspective, colors have profound effects on our …   
0.886 268466 … psychological responses, serving as a powerful form of non-verbal …   
0.882 268442 … affecting our emotions, behaviors, and even cultural practices. The vibran …   
0.551 111265 Pens come in every color of the rainbow! Brighten up your notes and …   
0.540 268568 … discovery that white light can be split into a range of colors using a …   
0.522 268570 discovery that white light can be split into a range of colors using a prism, …   
Cosine Position MRCR · layer 3, KV head 0, cluster 41   
0.970 94152 … an inclusive atmosphere where their voices are heard leads to better …   
0.964 111494 … prosthetics, striving to replicate their dexterity and functionality. …   
0.963 70391 … them from a distance, appreciating their complexity and charm. Yours …   
0.736 243256 … of Frogs in Ecosystems and Their Conservation I hope this letter …   
0.733 110722 … by a variety of houses, each with its own unique character. There was the …   
0.690 203583 … and the resources required to solve them. It investigates the limits of what …   
Cosine Position MK-3 · layer 23, KV head 3, cluster 1   
0.953 412289 … special magic uuids for 3505ceec-6bd8-419f-b6b6-797 …   
0.940 453901 … special magic uuids for 33549404-05cf-4258-9c84-c61 …   
0.940 499923 … special magic uuids for 3491836b-b504-4ee9-8893-e959 …   
0.630 19026 … special magic uuids for 645e0314-46b4-45a8-ac64 …   
0.626 14880 … special magic uuids for dafc21a6-566d-46e0-8cf5-e57afa …   
0.615 820 … special magic uuids for bb576d70-d4f7-47fd-bcb9-c8a485 …   
Cosine Position MK-2 · layer 27, KV head 0, cluster 70   
0.835 487952 … special magic numbers for troubled-arena is: 815482 …   
0.826 422558 … special magic numbers for troubled-apology is: 173206 …   
0.825 432399 … special magic numbers for weary-zero is: 2266208 …   
0.465 129 … special magic numbers for nervous-assist is: 383514 …   
0.456 195 … special magic numbers for guiltless-asterisk is: 6811 …   
0.443 29316 … is: 1419399. One of the special magic numbers for …  
Figure 20: Five additional clusters containing color vocabulary, tokens in a passage about color, pronouns, UUID digits, and adjectives in MK-2 keys. Positions are token indices, and highlights mark member tokens. Rows are sorted by key–centroid cosine similarity, with a gap between the two ends.