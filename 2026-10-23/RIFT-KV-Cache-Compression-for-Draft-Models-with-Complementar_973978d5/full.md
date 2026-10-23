# RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training

Dor Tsur\* Maor Ashkenazi Maximilian Golub Peter Hicks Yaniv Galron Talor Abramovich Yoav Miron Tiyasa Mitra Bita Darvish Rouhani Ran Zilberstein NVIDIA

## Abstract

Speculative decoding is a popular decode acceleration method. As such, it is particularly relevant to long-output generation, which requires an increasingly large KV cache. KV cache compression offers a way to constrain costs in long-context drafting. However, compression methods that focus on prefill information can lose relevance as the output length grows. We term this phenomenon decode staleness and show that it degrades draft model accuracy under selection-based KV cache compression. We introduce RIFT, a family of selection-based KV cache compression strategies that balance prefill information and recent generated tokens. We show that refreshing the KV cache selection during token generation significantly improves the drafter’s quality at the cost of additional decode latency. To alleviate the refresh costs, we propose an adaptive KV refresh strategy that reduces the number of score calculations while retaining the drafter’s quality. We further demonstrate how a compressibility-driven drafter training phase, based on sparse attention masks, can improve the drafter’s quality under a compressed KV cache. Combined with compressibility training, RIFT enables Qwen3.5- 4B, drafting for Qwen3.5-122B, to retain over 98% of its full-cache acceptance length on 32Ktoken inputs, 4K-token outputs, and an 8Ktoken KV cache budget.

## 1 Introduction

Speculative decoding accelerates autoregressive language models by verifying multiple draft proposals in a single target-model pass while preserving the target distribution (Leviathan et al., 2023; Chen et al., 2023), shifting inference towards a compute-bound regime. Its effectiveness depends on a draft that is both inexpensive and closely aligned with the target. Reasoning and agentic applications increase the pressure on this balance, combining long input sequence lengths (ISLs) with long output sequence lengths (OSLs) as models generate reasoning traces and accumulate interaction history (Nye et al., 2021; Yao et al., 2023). As context grows, key-value (KV) cache access can increasingly bottleneck drafting, limiting the achievable speedup from speculation (Sadhukhan et al., 2025). As a decode acceleration method, speculative decoding targets long OSL, further magnifying the KV cache bottleneck. This bottleneck is still evident in hybrid attention architectures (Dao and Gu, 2024; Yang et al., 2024): even when only 25% of a small hybrid model’s layers use full attention, its KV footprint can significantly exceed its weights memory footprint at long contexts with only a few concurrent requests (See Appendix A for details). Existing literature considers various forms of KV compression (Hooper et al., 2024; Liu et al., 2024; Laus et al., 2026). Here, we focus on selectionbased compression, where decode tokens attend only a compact subset of their past KVs.

Tail-augmented drafters, e.g., MTP, Eagle or DFlash (Li et al., 2025; Chen et al., 2026), are considered superior to external draft models on GPU-based deployments due to their minimal latency overhead. However, contemporary hardware-disaggregated inference and edgeassisted serving renews the relevance of small external drafts (Aubrey and Ghodsian, 2026; Park et al., 2025) over tail-augmented blocks, such as MTPs or DFlash (Chen et al., 2026). In these settings, memory capacity poses a constraint, as memory is shared between the model weights and growing KV cache (Kim et al., 2026). Considering all the above, we study draft KV cache compression under a fixed cache budget that is ISL-independent. KV compression also enables KV-based self-speculation: the target model itself drafts with a compressed cache and verifies with full cache (Sun et al., 2024; Sadhukhan et al., 2025; Zhao et al., 2025). Together, these regimes motivate the question:

![](images/a0aa91e989c7ea68b8b5149e3a752cce5a7e0cc178205b4e4229ed3e8271520b.jpg)  
Figure 1: AL improvement of RIFT and compressibility training on a Qwen3.5-4B draft (Qwen3.5-122B target), SPEED-Bench 32K ISL low-entropy, draft length 11. (a) AL vs. generation interval at 8K cache budget. (b) AL vs. KV cache budget. AL is measured vs. the full-KV draft AL. Diagonal pattern denotes compression-aware training.

How can we best compress the drafter’s KV cache to sustain target alignment in long contexts?

We study this question through the trade-off between acceptance length (AL) and the KV cache budget size, measuring drafting quality under a fixed memory constraint. A fixed KV-budget introduces a tradeoff between KVs selected during prefill and KVs accumulated during decode. Prompt tokens can provide valuable information, while the generated prefix records the target’s recent context. A prefill-based selection becomes stale as generation evolves, while a sliding window (SW) approach limits the model’s view to the latest prefix tokens and discards remote information. Decodetime refresh of selected KVs allows to adaptively determine the relevant KVs, at the cost of additional latency. This motivates two complementary approaches: allocate capacity to both prefill positions and generated tokens, and refresh the selected history as relevance changes.

Our analysis characterizes decode staleness and introduce RIFT (Recency–Importance Fusion and Training), a family of selection-based KV compression strategies combining recency allocation with decode-time refresh of importance-selected context. We further propose compression-aware knowledge distillation (KD) to reduce the training–inference mismatch and improve draft alignment under compression. Our contributions are:

1. Characterizing decode staleness: We show how prefill selection loses its initial advantage over SWA as generation progresses, and show this gap increases in hybrid-attention models.

2. RIFT: We address decode staleness and propose RIFT, a family of KV compression methods for draft models, balancing implementation complexity with resulting drafting quality.

3. Draft-compressibility training: We develop a knowledge-distillation scheme with compression-driven attention masks to improve draft alignment under a restricted KV budget.

Together, our methods retain most of the draft’s acceptance length, while using a fraction of the total KV cache. In SPEED-Bench (Abramovich et al., 2026) with 32K ISL and 4× compression ratio, Qwen3.5-4B retains up to 96% of its full cache AL (Qwen3.5-122B target), and Nemotron3.5-Lightning retains 97% of its fullcache AL (Nemotron3-Ultra target), while SnapKV retains 63 − 67% AL in both cases.

## 2 Background and Setting

Scope. We focus on two settings: 1) external autoregressive draft models and 2) KV-based selfspeculative decoding. We leave tail-augmented drafters outside the scope of this work, as our preliminary studies find negligible benefit from KV compression in these settings (Appendix B).

Draft-side Compression. Rejection sampling preserves the target distribution regardless of the draft’s KV-compression policy. Since compressing the target cache can instead degrade output quality, we compress only the draft cache. We measure accuracy using average acceptance length (AL), the expected number of accepted tokens plus the extra target-generated token. We consider two strategies representing the endpoints of the spectrum of static KV selection methods under a budget B:

SW attention. Pure SW attention (SWA) retains B recent tokens, updating the cache at every decode step, independently of token importance.

Prefill-based KV compression. Prefill-based methods select KVs using importance scores aggregated during prefill. A widely used method, SnapKV (Li et al., 2024), aggregates attention scores from a set of last prompt queries, termed the observation window. For each KV head, SnapKV retains the highest-scoring KV positions alongside a SW the size of the observation window. Herein, we take an observation window of 64 tokens.

## 3 Decode Staleness In KV Compression

While prefill-only KV compression has proven beneficial for speculative decoding (Sadhukhan et al., 2025), prior evaluations do not fully reflect long OSLs and modern model architectures. We start by investigating prefill-based methods in speculative decoding. We use SnapKV to compress the KV cache of Qwen3.5-4B, drafting for Qwen3.5- 122B and evaluate on SPEED-Bench 32K ISL low entropy split (Abramovich et al., 2026), with reasoning enabled. As seen in Figure 1(b), SWA outperforms SnapKV given the same KV budget. This result is surprising: SnapKV incorporates importance information, while SWA naively keeps a fixed tail of previous tokens. We make the following observations about the KV-compression literature:

KV cache compression benchmarks often overlook long OSL. Evaluations centered on long inputs primarily test which prompt KVs to retain (Bai et al., 2024; Hsieh et al., 2024). Regarding decode tokens, one can either 1) retain all generated KVs, 2) maintain a small sliding window alongside the prefill-selected cache, or 3) rescore and update the selection at every decoding step. The tradeoffs between these options become particularly important in speculative decoding, where the draft must remain aligned with a growing target-verified prefix under a fixed KV budget. Retaining all generated KVs violates the budget constraint, while a small sliding window can compromise alignment later in generation (as seen in Figure 1(a)). Rescoring at every decode step adds significant overhead to the drafting process.

To understand why a fixed prefill selection becomes less effective during generation, we examine how its relevance degrades, a phenomenon we term decode staleness, which motivates the following hypothesis:

Decode staleness: prefill-based KV selections become stale as generation progresses.

Additionally, we note that KV cache compression methods are rarely evaluated on hybrid-attention models. Yet, contemporary models interleave full attention with efficient recurrent components, including linear attention in Qwen3.5 (Yang et al., 2024; Qwen Team, 2026) and Mamba layers in Nemotron 3 (Dao and Gu, 2024; NVIDIA, 2026a).

Output-position dependence. We analyze average draft AL across generation positions with an 8K-token budget (4× compression). As shown in Figure 1(a), SnapKV’s AL decreases by 36.7% while SW improves by 41% as generation evolves. We term the evolving SnapKV-SW difference the staleness gap, which we consider a performance-based proxy for the growing irrelevance of SnapKV’s selected KVs during generation. We attribute the growth of the staleness gap to two factors. First, in speculative decoding the recent decode prefix is produced by the target model, making access to it highly important for draft-target alignment. Second, recent decode tokens often contain reasoning traces, which potentially act as a running prompt summary (Nye et al., 2021; Yao et al., 2023). Indeed, preliminary study shows that disabling reasoning reduces the staleness gap from 2.11 to 0.60 AL (See Figure 11 in Appendix C).

Architecture dependence. We empirically observe that decode staleness is exacerbated in hybridattention architectures, which are often overlooked in the KV compression literature. Due to lack of space, we refer the reader to Appendix C, where we show that 1) a full-attention Llama3.1 draft retains the expected ordering: SnapKV consistently outperforms SWA throughout generation, while, 2) in Nemotron3 (which is hybrid-SSM) and GPT-OSS (which interleaves hard-coded SWA layers with full attention), SWA again outperforms SnapKV.

Mechanistic evidence. We measure the fraction of the full-cache draft’s attention assigned to positions retained by each policy (SnapKV and SWA), under an 8K budget. Figure 2 shows that this fraction drops from 67.0% to 25.4% for SnapKV, while increasing from 43.5% to 68.7% for SWA as generation progresses. This crossover is consistent with attention shifting toward generated context that the prefill-selected cache largely omits.

![](images/6ad10259e7537cfb6d2084663dcaa2b83459cf627d34e783ff0341f6e44e48ca.jpg)  
Figure 2: Fraction of full-KV attention mass retained by SW and SnapKV (with 64 token SW) in the Qwen3.5- 4B draft on 32K-ISL data, with an 8K budget. SnapKV retains less attention as generation proceeds, while SW retains more.

## 4 RIFT: Staleness-Aware KV cache Compression

Motivated by the staleness gap, in this section we propose RIFT, a family of KV compression policies for draft models. RIFT comprises several solutions, ranging by their implementation complexity.

## 4.1 Effect of Sliding Window Ratio on Prefill-Based Methods

Our analysis emphasizes the importance of recent tokens to the draft’s KV cache. Thus, our first solution, which we call RIFT-stat, is a simple modification to a given static KV selection method.<sup>1</sup> RIFTstat balances a given budget B between a prefillbased token selection algorithm (e.g. SnapKV) and a SW component. For a given SW ratio $\rho \in ( 0 , 1 ]$ and a cache budget $B ,$ we reserve $r = \lfloor \rho B \rfloor$ positions for the most recent tokens and use the remaining $k = B - r$ positions for attention-based KV allocation. The SW ratio $\rho$ controls the trade-off between long-range retrieval and recent streaming context. Figure 3 shows the draft’s AL as a function of $\rho$ combining RIFT-stat with SnapKV (Section 4.4 presents results on several KV selection methods). With an 8K budget, the highest AL is achieved at $\rho = 0 . 5 \ : ( \mathrm { A L } \ : 5 . 6 1 4 )$ , an improvement of 37.8% over SnapKV (AL 4.073), with pure SWA achieving 5.230. In Appendix D.1 we provide additional analysis on the effect of the SW ratio in RIFT-stat.

Block-Level RIFT-stat. Serving systems store KVs in pages (Kwon et al., 2023). We therefore divide the cache into contiguous blocks, each scored by the mean of its keys’ scores. For an analysis of the block size effect on drafting performance, we refer the reader to Appendix D.3.

![](images/598201a71a50ea1ff6f6ef19a970fbd20e1b6b9a27227843ef4daecb35263cea.jpg)  
Figure 3: Effect of the sliding-window allocation ratio. 4K OSL and 8K KV budget. Qwen3.5-4B drafts Qwen3.5-122B on SPEED-Bench 32K ISL data. $\rho = 0$ denotes SnapKV with a fixed 64-token SW, and $\rho = 1$ corresponds to pure SWA.

## 4.2 RIFT-dyn: KV Cache Refresh During Decode

While RIFT-stat introduces gains in the draft AL, its chosen prefill KVs are still selected only once before decode. However, as generation progresses, the relevant prompt KVs dynamically change. We therefore assume access to the full draft KV cache in HBM or offloaded memory for full KV rescoring. We propose RIFT-dyn, which iteratively rescores the working KV cache every $\Delta$ decode steps. The resulting working KV cache is a combination of the decode-updated KV cache and the SW component of size $\lfloor \rho B \rfloor$ . We focus on SnapKV-based scoring. However, RIFT-dyn is compatible with any alternative KV scoring functions. RIFT-dyn refreshes the draft’s working memory. Its effectiveness stems from two main advantages: (i) it scores the entire available KV cache (excluding the SW component), and (ii) it computes scores using a recent observation window, so they better reflect the model’s current state.

RIFT-dyn introduces a significant and consistent improvement in the draft’s AL (see Section 4.4 for results). For example, with a refresh interval of $\Delta = 1 2 8$ and a Qwen3.5-4B drafter at 100K ISL, RIFT-dyn retains 95.7% of full-KV AL, compared with 85.6% for RIFT-stat. More frequent KV cache refreshes improve draft quality, with projected rescoring latency overhead (As seen in Figure 4). We next consider adaptive refresh to improve the AL-refresh tradeoff.

![](images/dfeefa9b77639566ed022a0d4f97f1c2ed6151ceab2be240eaf65b697ecba7ec.jpg)  
Figure 4: Effect of RIFT-dyn’s refresh interval on AL (blue) and projected rescoring overhead relative to measured draft decode time (orange). Qwen3.5-4B, 100K ISL, 8K cache budget, and 4K OSL. Appendix D.2 describes the latency projection.

## 4.3 Adaptive KV cache Refresh

RIFT-dyn improves draft accuracy by periodically refreshing the KV cache, even when the current selection remains useful. To avoid unnecessary refreshes, we propose RIFT-ada, which adaptively chooses when to refresh the KV cache based on signals collected at the draft during decode.

At decode step t, let τ be the decode position of the most recent KV scoring and $a _ { t } = t - \tau$ the age of the current cache selection. Then, a KV rescore occurs if a threshold policy $\pi ( \boldsymbol { z } _ { t } )$ is activated or $a _ { t }$ exceeds a maximal no-refresh interval $a _ { \mathrm { m a x } } .$

$$
u _ { t } = \mathbb { 1 } \big \{ ( a _ { t } \geq a _ { \operatorname* { m a x } } ) \vee \left( \pi ( z _ { t } ) \wedge a _ { t } \geq a _ { \operatorname* { m i n } } \right) \big \} ,
$$

where $u _ { t } = 1 { \mathrm { ~ t r i g g e r s } }$ a rescore. The minimum age $a _ { \mathrm { m i n } }$ prevents consecutive refreshes, while the maximum age $a _ { \mathrm { m a x } }$ bounds staleness. We evaluate u<sub>t</sub> every $d = 8$ steps.

Threshold Policies. At each d-token decision interval, we aggregate several signals $z _ { t } \mathrm { : }$ an exponentially weighted mean of previous measured acceptance lengths $A _ { t }$ , draft entropy $H _ { t } ,$ , and query drift, which is given by $D _ { t } ^ { q } = 1 - \bar { q } _ { t } \cdot \bar { q } _ { \tau }$ and q¯ denotes a normalized query summary. We average $D _ { t } ^ { q }$ over full-attention layers and KV-head groups. We compare each of these signals with a calibrated threshold (See Appendix E.3 for details on the calibration process), and aggregate multiple scores into a threshold policy using (and/or) gates or a fitted weighted score. Figure 5 illustrates how RIFT-ada varies refresh frequency during generation, compared with RIFT-dyn’s periodic schedule, showing how various stages in generation require different rescoring cadence. The evaluated policies are detailed in Section 4.4.

<table><tr><td rowspan="2">Draft (ISL)</td><td rowspan="2">T SW SnapKV</td><td>RIFT</td></tr><tr><td>stat dyn ada</td></tr><tr><td>Qwen3.5 (32K)</td><td>0 5.23 4.07 1 4.70 3.69</td><td>5.62 6.12 6.19 4.99 5.48 5.55</td></tr><tr><td>Qwen3.5 (100K)</td><td>06.12 4.46</td><td>6.407.14 7.17</td></tr><tr><td rowspan="2">Qwen3.5 (100K) Llama3.1 (32K)</td><td>1 5.40 3.82</td><td>5.55 6.216.36</td></tr><tr><td>0 2.24 5.19</td><td>6.07 6.91 7.48</td></tr><tr><td rowspan="2">N3.5 (self) (32K)</td><td>1 1.55 3.83</td><td>4.54 5.055.52</td></tr><tr><td>0 5.73 4.83 1 5.56 4.51</td><td>6.72 8.268.71 6.84 8.70 8.93</td></tr><tr><td rowspan="2">N3.5 (ext.) (32K)</td><td></td><td></td></tr><tr><td>04.31 3.64 1 3.86 3.22</td><td>4.354.994.98 3.93 4.45 4.99</td></tr></table>

Table 1: AL on low-entropy SPEED-Bench 32K/100K ISL, with an 8K budget, 4K OSL, and draft length 11. Qwen3.5: 4B → 122B; Llama3.1: 8B → 70B; Nemotron3.5-Lightning in self-speculation and as an external draft to Nemotron3-Ultra target.

## 4.4 Results

We evaluate RIFT on SPEED-Bench, which comprises 8 different generation benchmarks, focusing on its throughput splits. Unless stated otherwise, we use draft length of 11, maximum OSL of 4K and decode temperatures $T \in \{ 0 , 1 \}$ . The KV cache budget is 8K and accuracy is measured in average AL. Additional results are provided in Appendix D, where we cover more drafter-target pairs, datasets and generation setups. All RIFT variations use $\rho = 0 . 5$ . For RIFT-ada, we use the same refresh framework throughout and consider a threshold that results in strictly fewer refreshes than RIFT-dyn (see Appendix E.3 for more details). We also provide preliminary results on RIFT applied beyond speculative decoding in Appendix F: we evaluate RIFT-stat on a long-input, long-output reasoning benchmark, showing that its advantage extends beyond speculative decoding.

AL vs. Budget. We start by demonstrating the accuracy of the various RIFT variants across several evaluation settings. First, we consider Qwen3.5-4B drafting for Qwen3.5-122B. Figure 1(b) compares RIFT with SnapKV and SWA on SPEED-Bench 32K ISL across cache budgets. Figure 1(a) depicts AL vs. decode position, showing RIFT-stat benefits from both worlds: it preserves SnapKV’s high early-generation AL while benefiting from the SW component later, as expected from Section 3. Table 1 extends the comparison to 100K ISL and additional model pairs. Specifically, we consider Nemotron3.5-Lightning (selfspeculation and drafting for Nemotron3-Ultra) and

![](images/90240c5c5253c404cf8e7b9b2037e9879ca1411b509b65f71b327e065cd3988d.jpg)  
Figure 5: KV rescore patterns comparison on a candidate sample from SPEED-Bench. RIFT-ada attains a higher AL and requires less KV refreshes. It can be seen that KV refreshes are not periodic and depend on generation location.

<table><tr><td></td><td>Budget Selection score</td><td>Baseline</td><td>+ RIFT-stat</td></tr><tr><td rowspan="3">4K</td><td>CriticalKV</td><td>3.57</td><td>5.02 (+40.6%)</td></tr><tr><td>Expected Attention</td><td>3.59</td><td>5.03 (+40.2%)</td></tr><tr><td>CurDKV</td><td>3.29</td><td>4.84 (+47.1%)</td></tr><tr><td rowspan="3">8K</td><td>CriticalKV</td><td>4.13</td><td>5.62 (+36.2%)</td></tr><tr><td>Expected Attention</td><td>4.06</td><td>5.56 (+36.9%)</td></tr><tr><td>CurDKV</td><td>3.76</td><td>5.35 (+42.2%)</td></tr></table>

Table 2: Absolute AL at T = 0 (relative gains in parentheses): fixed 64-token SW versus RIFT-stat (ρ = 0.5). Qwen3.5-4B drafts Qwen3.5-122B on SPEED-Bench 32K ISL, 4K OSL, draft length 11.

Llama3.1-8B (Llama3.1-70B target), which is a full-attention model. For Qwen3.5 at 100K ISL, an 8K budget, and T = 0, RIFT-stat, RIFT-dyn, and RIFT-ada retain 85.6%, 95.5%, and 96.0%, respectively, compared with 59.7% for SnapKV. Appendix D provides additional results, encompassing various budgets, models, OSLs and workloads. Appendix G demonstrates RIFT’s advantage across various generation tasks.

Beyond attention scores. While we demonstrate RIFT-stat on top of SnapKV, it can be used with any other prefill-based KV compression method. We combine several additional KV selection scores with RIFT-stat: 1) CriticalKV’s outputperturbation score (Feng et al., 2026), 2) Expected Attention’s distribution over future queries (Devoto et al., 2025), and 3) CurDKV’s leverage scores (Sengupta et al., 2025). We measure the benefit of applying RIFT-stat on top of each method. As seen in Table 2, all three methods improve substantially at both budgets (see Table 14 for T = 1). The proposed observations naturally extend to direct variants of SnapKV, such as Ada-KV and PyramidKV, which focus on layer/head budget allocation (Feng et al., 2025b; Cai et al., 2024).

Cross-Family Drafting We consider a crossfamily external drafting setting. The main challenge in this setup is that the draft and target models do not share the same vocabulary. We adapt drafters through a simple tokenizer matching step followed by target-model KD (see Appendix H for tokenizer-adaptation details). We consider several cross-family draft-target pairs and apply RIFT in a long-ISL drafting setting. Specifically, we consider Llama3.2-1B and Qwen3.5-4B drafts and DeepSeek-V4 (DSV4) Pro and Flash as target models. As seen in Table 3, RIFT extends to the crossmodel scenario, supporting external drafting when an in-family smaller model is unavailable.

<table><tr><td>Draft</td><td>Target</td><td>SnapKV + RIFT-stat + RIFT-dyn</td><td></td></tr><tr><td rowspan="2">Qwen3.5-4B</td><td>DSV4-Flash</td><td>3.50 4.26</td><td>5.35</td></tr><tr><td>DSV4-Pro</td><td>3.43</td><td>4.26 5.07</td></tr><tr><td rowspan="2">Llama3.2-1B</td><td>DSV4-Flash</td><td>2.50 3.24</td><td>3.50</td></tr><tr><td>DSV4-Pro</td><td>2.37</td><td>3.07 3.26</td></tr></table>

Table 3: Absolute AL for adapted drafters of DeepSeek-V4-Flash/Pro on low-entropy SPEED-Bench: 32K ISL, 8K budget, 4K OSL, T = 0, draft length 11.

The benefit of adaptive refresh. We compare RIFT-ada with RIFT-dyn’s periodic refresh through their AL–refresh tradeoff, measuring AL gains at matched refresh counts and refresh reductions at matched AL. The policy triggers a refresh if any of the following conditions holds: acceptance is at or below its threshold, entropy is at or above its threshold, acceptance has fallen or entropy has risen sufficiently relative to post-refresh values, or query drift exceeds its threshold. As seen in Figure 6, this policy, which sweeps the query drift threshold, achieves 2.03× fewer refreshes (or up to +0.122 AL at the same refresh count). We refer the reader to Appendix E for additional settings, parametertransfer analysis and policy information.

Comparison with additional baselines. We compare RIFT with StreamingLLM (Xiao et al., 2024), $_ \mathrm { H _ { 2 } O }$ (Zhang et al., 2023) and TriAttention (Mao et al., 2026), which are decode-time KV selection methods. As seen in Table 4, StreamingLLM is outperformed by all compression methods, $_ \mathrm { H _ { 2 } O }$ outperforms RIFT-stat, while RIFT-dyn and RIFT-ada achieve higher AL than either baseline across the two context lengths and temperatures. Appendix I provides comparison across additional settings.

![](images/a591f695e3b561cbf1339c30b006187983a926c179bf4e288320e8f136bc420a.jpg)

Figure 6: RIFT-ada AL-refresh tradeoff at $T = 1$ for Qwen3.5-4B drafting Qwen3.5-122B on SPEED-Bench 32K low-entropy data, with an 8K budget and 4K OSL.
<table><tr><td colspan="2">32K ISL</td><td colspan="2">100K ISL</td></tr><tr><td>Method</td><td> $T = 0$   $T = 1$ </td><td> $T = 0$ </td><td> $T = 1$ </td></tr><tr><td>StreamingLLM</td><td>5.34</td><td>4.73 5.60</td><td>4.98</td></tr><tr><td>H2O</td><td>5.73</td><td>5.03 6.65</td><td>5.77</td></tr><tr><td>TriAttention</td><td>5.43 4.77</td><td>6.10</td><td>5.11</td></tr><tr><td>RIFT-stat</td><td>5.62</td><td>4.99 6.40</td><td>5.55</td></tr><tr><td>RIFT-dyn</td><td>6.12</td><td>5.48 7.14</td><td>6.21</td></tr><tr><td>RIFT-ada</td><td>6.17</td><td>5.51 7.17</td><td>6.36</td></tr></table>

Table 4: Absolute AL against decode-time baselines. Qwen3.5-4B draft, low-entropy SPEED-Bench, 8K budget, 4K OSL, and draft length 11.

## 5 Improving Draft KV Compressibility with Training

The previous section improves KV compression at inference with a fixed draft. A natural question is:

Can wefurther improve the drafter’s KV compressibility using dedicated training?

Inspired by recent results in compressibility training of pretrained models (Gelberg et al., 2026), we propose a draft compressibility post-training phase. Our goal is to minimize the gap between the draft’s working KV cache in training and during inference. In what follows, we focus on compressibility using attention scores. However, the training method itself is score-agnostic. The resulting objective promotes target-aligned proposals under a restricted KV budget.

Compression-aware knowledge distillation. We propose a modified KD forward pass. We begin with a dense scoring pass by applying a forward pass to the frozen draft model. We capture post-RoPE queries and keys in each attention layer. We score keys by summing their attention scores against a fixed observation window. We define a compression ratio $p$ and a SW ratio $\rho .$ During training, the effective cache budget is $B = \lfloor p L \rfloor$ of which $r = \lfloor \rho B \rfloor$ positions are reserved for a SW component, where L is the sample’s sequence length. We build an attention mask that combines the SWA with the highest-scoring prompt positions, obtained from the dense scoring pass. We then perform an additional student forward pass, in which we apply the LoRA adapters and the built attention masks to get a sparse forward pass. We combine cross-entropy and KL distillation using precomputed teacher rollouts and logits:

![](images/a96ce1eec8059af074a9094b9b368bc2545c0c68d101d6f0d278638a995f9670.jpg)  
Figure 7: Compression-aware distillation. The snowflake and fire symbols denote frozen and trainable passes, respectively. The mask combines recent positions (purple) with importance-selected context (green).

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { 1 } \mathrm { C E } ( q _ { \theta } , y ) + \lambda _ { 2 } D _ { \mathrm { K L } } ( p \| q _ { \theta } ) , } \end{array}
$$

where y is the teacher-generated next token, and $q _ { \theta }$ and p are the student and teacher distributions. The student uses the sparse attention mask. Figure 7 provides an overview. Our method extends to selfspeculation by toggling the adapters and using the corresponding selection method during inference.

## 5.1 Results

Setting. We train for 10K steps using the Nemotron V2 post-training-dataset,<sup>2</sup> with batch size 64 and SW ratio $\rho = 0 . 2 5$ . To increase budget robustness we randomly sample compression ratios for each sample. Evaluation uses SPEED-Bench 32K low-entropy data, 4K OSL, and draft length 11. Appendix J.2 gives the full recipe.

Compression-aware gains. We evaluate Qwen3.5 and Nemotron3.5-Lightning, considering both self speculation and external drafting. As seen in Table 5, compressibility training improves the drafter’s performance under RIFT. For example, we see gains of up to 16.19% for Qwen3.5-4B (RIFT-stat, 2K budget, $T = 1 )$ and 16.42% gains for Nemotron3.5-Lightning (RIFT-dyn at 2K and $T = 1 )$ . Both comparisons measure improvement over the untrained drafter under the same compression policy and budget. We note that gains are smaller at larger budgets, as the baseline method often already retains more than 95% of the full-KV AL, leaving a very small gap for improvement.

<table><tr><td rowspan="2">Draft → target</td><td rowspan="2"></td><td colspan="3">RIFT-stat</td><td colspan="3">RIFT-dyn</td></tr><tr><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td rowspan="2">Qwen3.5 (4B → 122B)</td><td></td><td>0 +7.70%</td><td>+6.87%</td><td></td><td>+5.70% +7.48%</td><td>+5.50%</td><td>+3.72%</td></tr><tr><td></td><td>1 +16.19%</td><td>+14.88%</td><td></td><td>+12.28% +15.51%</td><td>+12.25%</td><td>+9.21%</td></tr><tr><td rowspan="2">Lightning → Nemotron3-Ultra</td><td></td><td>0 +7.51%</td><td>+7.48%</td><td></td><td>+7.59%+7.88%</td><td>+7.86%</td><td>+5.59%</td></tr><tr><td></td><td>1 +15.69%</td><td>+15.63%</td><td></td><td>+15.08% +16.42%</td><td>+15.84%</td><td>+13.19%</td></tr><tr><td rowspan="2">Lightning self-speculation</td><td></td><td>0 +5.3%</td><td>+3.4%</td><td>+2.5%</td><td>+6.5%</td><td>+2.8%</td><td>+1.0%</td></tr><tr><td></td><td>1 +11.0%</td><td>+7.8%</td><td>+4.2%</td><td>+10.3%</td><td>+5.0%</td><td>+1.5%</td></tr></table>

Table 5: Relative AL improvement from compression-aware KD over the base model under the same KV compression policy and budget, evaluated on SPEED-Bench 32k ISL with draft length 11.

<table><tr><td>Method</td><td>T</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td rowspan="2">RIFT-stat</td><td>0</td><td>+5.44%</td><td> $+ 4 . 6 8 \%$ </td><td>+3.71%</td></tr><tr><td>1</td><td>+7.31%</td><td>+6.37%</td><td>+4.66%</td></tr><tr><td rowspan="2">RIFT-dyn</td><td>0</td><td>+5.34%</td><td>+3.51%</td><td> $+ 1 . 8 2 \%$ </td></tr><tr><td>1</td><td>+7.25%</td><td> $+ 4 . 7 3 \%$ </td><td> $+ 2 . 3 7 \%$ </td></tr></table>

Table 6: Relative AL improvement from compressionaware training over a similar training recipe with no compression-driven masks. Qwen3.5-4B→Qwen3.5- 122B, SPEED-Bench 32k ISL, draft length 11.

Benefit over no-mask training. To isolate the benefit of compression-aware training from that of the post-training data, we compare with the same pipeline without attention masking. As seen in Table 6, compressibility training offers a significant improvement on top of the baseline training scheme, especially in the lower budget regimes.

## 6 Related Work

Due to the amount of existing methods, we describe the most closely related work to our methods. We provide a broader discussion of literature in Appendix K. SnapKV estimates prompt importance from observation-window attention and PyramidKV and AdaKV refine allocation across layers and heads (Li et al., 2024; Cai et al., 2024; Feng et al., 2025b). These methods optimize which

KVs survive compression, while we study how prefill decisions become stale during long generation. Decode-time policies continually adapt the working cache, e.g., StreamingLLM retains attention sinks and recent tokens, while $_ \mathrm { H _ { 2 } O }$ combines recent tokens with accumulated-attention heavy hitters (Xiao et al., 2024; Zhang et al., 2023). TriForce uses a retrieved target cache as an intermediate drafter, refreshed periodically or by acceptance gating (Sun et al., 2024). MagicDec studies sparse-KV self-speculation at high concurrency. SpecExtend, SparseSpec, and Vegas study how verification can be used for KV cache compression (Sadhukhan et al., 2025; Cha et al., 2025; Zhao et al., 2025; Yue et al., 2026). In compressibility training, KV-CAT (Gelberg et al., 2026) proposes a continual pretraining phase with a learned KV scoring model.

## 7 Conclusion

In this work, we identified decode staleness, where prefill-selected context loses relevance as OSL grows, especially in hybrid attention architectures. We show that decode staleness degrades the drafttarget alignment in speculative decoding. We proposed RIFT: RIFT-stat combines recency with important prompt positions, RIFT-dyn improves KV selection by rescoring the cache during decode and RIFT-ada allows for adaptive KV rescore to minimize the induced latency overhead. To improve RIFT we propose a compression-aware training phase. Together, our results demonstrate the value of adapting both KV cache selection policies and draft models to retain acceptance lengths under fixed KV cache budgets. A promising future direction is to combine RIFT’s draft-side staleness signals with target-verification attention, as initial signals show promising results in this direction (e.g., Cha et al., 2025; Yue et al., 2026).

## Limitations

At the time of writing, external draft models generally provide lower end-to-end speedups in GPUonly deployments than tail-augmented drafters such as MTP, EAGLE, and DFlash. Our work is therefore motivated by future heterogeneous systems in which drafting runs on a memoryconstrained, SRAM-based accelerator, an emerging direction for which (Aubrey and Ghodsian, 2026) provides an early signal. In such a system, KVselection scores could be computed on the GPU using a copy of the draft model, while only the selected KV blocks are transferred to the drafting device. Our evaluation focuses on acceptance-length trade-offs under a fixed active draft KV cache budget. Because decode-time refresh assumes that the full draft KV history remains available in GPU HBM or offloaded memory, this active budget constrains the drafting device’s memory footprint but not the system’s total stored KV memory. Furthermore, on a GPU setup with naive measurements, we see a relative cost overhead of up to 6.0% of KV cache rescoring compared to the total draft decode latency. This overhead may become more significant when drafting is accelerated on an SRAMbased device, although its magnitude in disaggregated deployments remains to be measured.

## References

Talor Abramovich, Maor Ashkenazi, Carl (Izzy) Putterman, Benjamin Chislett, Tiyasa Mitra, Bita Darvish Rouhani, Ran Zilberstein, and Yonatan Geifman. 2026. SPEED-Bench: A Unified and Diverse Benchmark for Speculative Decoding.

Kyle Aubrey and Farshad Ghodsian. 2026. Inside NVIDIA Groq 3 LPX: The Low-Latency Inference Accelerator for the NVIDIA Vera Rubin Platform. NVIDIA Technical Blog. Published March 16, 2026. Accessed: 2026-09-09.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. 2024. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3119–3137.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. 2025. LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-Context Multitasks. In Proceedings of the

63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3639–3664.

Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, and Junjie Hu. 2025. R-KV: Redundancy-aware KV Cache Compression for Reasoning Models. arXiv preprint arXiv:2505.24133.

Zefan Cai, Yichi Zhang, Bofei Gao, Yuliang Liu, Yucheng Li, Tianyu Liu, Keming Lu, Wayne Xiong, Yue Dong, Junjie Hu, and Wen Xiao. 2024. PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling. arXiv preprint arXiv:2406.02069.

Jungyoub Cha, Hyunjong Kim, and Sungzoon Cho. 2025. SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences. arXiv preprint arXiv:2505.20776.

Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating Large Language Model Decoding with Speculative Sampling. arXiv preprint arXiv:2302.01318.

Jian Chen, Yesheng Liang, and Zhijian Liu. 2026. DFlash: Block Diffusion for Flash Speculative Decoding. arXiv preprint arXiv:2602.06036.

Tri Dao and Albert Gu. 2024. Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. arXiv preprint arXiv:2405.21060.

Alessio Devoto, Maximilian Jeblick, and Simon Jégou. 2025. Expected Attention: KV Cache Compression by Estimating Attention from Future Queries Distribution. arXiv preprint arXiv:2510.00636.

Dogaç Eldenk, Payal Mohapatra, Yigitcan Comlek,˘ Kaan Oktay, Hongyang Zhang, and Stephen Xia. 2026. Attention Drift: What Autoregressive Speculative Decoding Models Learn. arXiv preprint arXiv:2605.09992.

Yuan Feng, Haoyu Guo, JunLin Lv, S. Kevin Zhou, and Xike Xie. 2025a. Taming the Fragility of KV Cache Eviction in LLM Inference. arXiv preprint arXiv:2510.13334.

Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S. Kevin Zhou. 2025b. Ada-KV: Optimizing KV Cache Eviction by Adaptive Budget Allocation for Efficient LLM Inference. In Advances in Neural Information Processing Systems.

Yuan Feng, Junlin Lv, Haoyu Guo, Yukun Cao, S. Kevin Zhou, and Xike Xie. 2026. CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective. In International Conference on Machine Learning.

Yoav Gelberg, Yam Eitan, Michael Bronstein, Yarin Gal, and Haggai Maron. 2026. Training Transformers for KV Cache Compressibility. arXiv preprint arXiv:2605.05971.

Shen Han, Yuyang Wu, Junpu Yu, and Olexandr Isayev. 2026. KARA: Efficient Reasoning LLM Serving via Sliding-Window KV Cache Compression. arXiv preprint arXiv:2607.01237.

Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun S Shao, Kurt Keutzer, and Amir Gholami. 2024. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. Advances in Neural Information Processing Systems, 37:1270–1303.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. RULER: What’s the Real Context Size of Your Long-Context Language Models? arXiv preprint arXiv:2404.06654.

Eunjeong Kim, Yeong Jun Jeon, and Myeonggyun Han. 2026. MemSpec: Memory-Aware Runtime for Adaptive Draft Scheduling in Speculative Decoding on Edge Devices. In Proceedings of the ACM SIG-PLAN/SIGBED International Conference on Languages, Compilers, and Tools for Embedded Systems, pages 180–192.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Hannah Laus, Claudio Mayrink Verdun, Hao Wang, Flavio du Pin Calmon, and Felix Krahmer. 2026. KV Cache Compression Through the Lens of Transform Coding. arXiv preprint arXiv:2608.14191.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast Inference from Transformers via Speculative Decoding. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 19274–19286. PMLR.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM Knows What You Are Looking for Before Generation. arXiv preprint arXiv:2404.14469.

Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2025. EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test. Advances in Neural Information Processing Systems, 38:136737–136756.

Xiang Liu, Zhenheng Tang, Peijie Dong, Zeyu Li, Yue Liu, Bo Li, Xuming Hu, and Xiaowen Chu. 2025.

ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference. In Advances in Neural Information Processing Systems, volume 38.

Zichang Liu, Aditya Desai, Fangshuo Liao, Weitao Wang, Victor Xie, Zhaozhuo Xu, Anastasios Kyrillidis, and Anshumali Shrivastava. 2023. ScissorHands: Exploiting the Persistence of Importance Hypothesis for LLM KV Cache Compression at Test Time. arXiv preprint arXiv:2305.17118.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. 2024. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arXiv preprint arXiv:2402.02750.

Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, and Yukang Chen. 2026. TriAttention: Efficient Long Reasoning with Trigonometric KV Compression. arXiv preprint arXiv:2604.04921.

NVIDIA. 2026a. Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning. arXiv preprint arXiv:2604.12374.

NVIDIA. 2026b. Nemotron 3.5 Super and Lightning: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Models for Agentic AI. Technical report, NVIDIA.

Maxwell Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, et al. 2021. Show Your Work: Scratchpads for Intermediate Computation with Language Models. arXiv preprint arXiv:2112.00114.

Jinwoo Park, Seunggeun Cho, and Dongsu Han. 2025. SpecEdge: Scalable Edge-Assisted Serving Framework for Interactive LLMs. arXiv preprint arXiv:2505.17052.

Qwen Team. 2026. Qwen3.5-122B-A10B. Hugging Face model card.

Ranajoy Sadhukhan, Jian Chen, Zhuoming Chen, Vashisth Tiwari, Ruihang Lai, Jinyuan Shi, Ian En-Hsu Yen, Avner May, Tianqi Chen, and Beidi Chen. 2025. MagicDec: Breaking the Latency-Throughput Tradeoff for Long Context Generation with Speculative Decoding. In International Conference on Learning Representations.

Ayan Sengupta, Siddhant Chaudhary, and Tanmoy Chakraborty. 2025. Value-Guided KV Compression for LLMs via Approximated CUR Decomposition. arXiv preprint arXiv:2509.15038.

Yi Su, Zhenxu Tian, Dan Qiao, Yuechi Zhou, Juntao Li, and Min Zhang. 2026. LongFlow: Efficient KV Cache Compression for Reasoning Models. arXiv preprint arXiv:2603.11504.

Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, and Beidi Chen. 2024. TriForce: Lossless Acceleration of Long Sequence Generation with Hierarchical Speculative Decoding. In Conference on Language Modeling.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. 2024. QUEST: Query-Aware Sparsity for Efficient Long-Context LLM Inference. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 47901–47911.

Alagappan Valliappan. 2026. Windowed-MTP: Removing the Full-Context Draft-KV Tax at Million-Token Context. arXiv preprint arXiv:2607.21535.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient Streaming Language Models with Attention Sinks. In International Conference on Learning Representations.

Penghui Yang, Cunxiao Du, Fengzhuo Zhang, Haonan Wang, Tianyu Pang, Chao Du, and Bo An. 2026. LongSpec: Long-Context Lossless Speculative Decoding with Efficient Drafting and Verification. In Proceedings of the 64th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1826–1844.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2024. Gated Delta Networks: Improving Mamba2 with Delta Rule. arXiv preprint arXiv:2412.06464.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations.

Yikang Yue, Yuqi Xue, and Jian Huang. 2026. Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention. arXiv preprint arXiv:2602.07223.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models. arXiv preprint arXiv:2306.14048.

Yilong Zhao, Jiaming Tang, Kan Zhu, Zihao Ye, Chi-Chih Chang, Chaofan Lin, Jongseok Park, Guangxuan Xiao, Mohamed S. Abdelfattah, Mingyu Gao, Baris Kasikci, Song Han, and Ion Stoica. 2025. Accelerating Large-Scale Reasoning Model Inference with Sparse Self-Speculative Decoding. arXiv preprint arXiv:2512.01278.

Appendix for “RIFT: KV Cache Compression for Draft Models with Complementary Compressibility Training”

## A KV cache Costs in Hybrid Draft Models

We quantify the KV cache footprint in Qwen3.5-4B and Nemotron-3-Nano (30B-A3B) relative to the combined memory (of weights, KV cache and recurrent states). Concurrent requests share the same weights but retain separate caches. KV memory grows with context length and concurrency while recurrent states (such as SSM and GDN states) grow with concurrency but not with ISL or OSL. We assume bf16 weights and KV entries, fp32 recurrent matrices, and bf16 convolution buffers, giving approximately 49.5 MiB of state per request for Qwen and 47.1 MiB for Nemotron3-Nano. Figure 8 shows that long-context KV storage can still dominate. For Qwen3.5-4B, four requests at 64K tokens use 51.1% of the occupied memory, despite only 25% of layers using full attention. Nano’s smaller per-token cache and larger resident weight footprint delay the crossover. Nonetheless, at 256K ISL and 64 concurrent requests, its KV cache accounts for 62.0% of the accounted memory. Consequently, it is clear that the KV memory bottleneck remains evident in standard serving loads.

## B KV Cache Compression in Tail-Augmented Drafters

We provide additional information on our initial results for KV compression’s effect on MTP. Specifically, we examined Nemotron3-Super’s MTP draft, which went through a post-training phase with a 1K

SW. The training used the Nemotron post training V2 dataset, using a standard KD scheme for 10K training steps. The resulting MTP was evaluated on SPEED-Bench 32k and 100k ISL data. The MTP’s ALs were 4.24 (94% of full KV AL) at 32K ISL and 5.07 (95% of full KV AL) at 100K ISL. Consequently, a simple SWA shows sufficient AL retention in the settings of interest, rendering it a sufficient and simpler option over RIFT for MTPs. Furthermore, the overall reduction in the system’s TPS due to the MTP’s KV compression was negligible, implying that in a tail-augmented drafting setting, the MTP’s KV cache may not pose as a significant bottleneck compared to external drafting and self-speculation.

## C Additional Staleness Analysis

We provide additional evidence on draft staleness. Figure 10 extends the comparison across model families: SW outperforms SnapKV late in generation for GPT-OSS and Nemotron, as for Qwen3.5 in Figure 1(a), while Llama-3.1 consistently favors SnapKV over SW. Qwen3, which is a full-attention model, shows ambiguous results in terms of one method’s dominance over the other. Next, we evaluate the effect of reasoning. Figure 11 compares Qwen3.5-4B’s drafting accuracy with thinking enabled and disabled (at both the draft and target models). Disabling thinking reduces the staleness gap in the final position bucket from 2.11 to 0.60 AL, although the SnapKV-SW crossover persists. All results use T = 0, low-entropy SPEED-Bench 32k ISL data with a fixed 8K KV cache budget.

![](images/0be374160939a92727131a095e6bc180b43ad05bfccf0ee54271caf51debcb4a.jpg)  
Figure 8: Attention KV cache share of total memory in two hybrid architectures, evaluated at the end of prefill.

![](images/e575fd7f93b98f43371d94d62d772d157ac8903d00bb2300488b1e139027e845.jpg)  
Figure 9: Full-attention mass shifts from prompt tokens to the generated prefix in the Qwen3.5-4B draft on 32K ISL data.

## D Additional Staleness-Aware Compression Results

## D.1 RIFT-stat Sliding-Window Ratio

This experiment uses the base Qwen3.5-4B draft without fine-tuning. Figure 12 supplements Figure 3 with five additional output-cap and KVbudget combinations, evaluated on the same precomputed Qwen3.5-122B responses at $T = 0$ , with draft length 11, block size 128, and observation window 64. The best observed ratios range from 0.3 to 0.7 across these settings, with modest gains over the default $\rho = 0 . 5$ . In each panel, the best intermediate ratio outperforms both fixed-64 SnapKV and pure SWA. Output length is a cap, so naturally shorter responses retain their original lengths; comparisons across panels change the output cap and/or KV budget.

## D.2 Rescoring Overhead Estimation

Figure 4 in the main paper projects the added rescoring overhead for Qwen3.5-4B with 100K ISL and 4K OSL. For a given refresh interval, we begin by calculating the drafter’s AL under RIFT-dyn with the corresponding refresh interval. Then, we run the speculative decoding loop in vLLM with the corresponding forced AL and a fixed 8K draft KV cache to measure decode latency. We add the estimated vLLM rescoring time for the given interval, and report the added time as a percentage of measured decode latency.

## D.3 Block-Size Ablation

As described in Section 4, to accommodate realworld serving scenarios, we work with block-level KV cache compression. We examine the effect of block size on the draft’s performance. We consider Qwen3.5-4B with draft length 11, drafting for Qwen3.5-122B using the SPEED-Bench dataset,

100K and 32K throughput splits. In each setting, we apply RIFT-stat with the corresponding block size. We consider two generation temperatures and sweep various block sizes. As seen in Figure 15, when the block size is significantly smaller than the overall KV cache, the effect is negligible. However, each setting has a "critical block size" beyond which the decision space is too coarse and the draft’s performance degrades.

## D.4 Budget, Context, and Cross-Family Comparisons

We supplement the results in Section 4.4 with additional cache-budget sweeps in various settings.

Additional Qwen Family Results Table 7 expands the Qwen3.5 results across context length, output entropy, temperature, and cache budget. The gains from allocating cache capacity to recent tokens and refreshing the remote selection remain consistent throughout these settings. Table 8 presents additional results with a Qwen3.5- 0.8B draft model. As can be seen, the performance extends to smaller models.

Additional model families and cross-family drafting We extend our experiments beyond the Qwen3.5 family and the pairs considered in Section 4.4. We provide several settings:

1. Nemotron3 family - Due to their sizes, we consider Nemotron3-Super and Nemotron3- Ultra as target models, with Nemotron3-Nano and Nemotron3.5-Lightning serving as their draft models. Additionally, we consider a self-speculation setting using Nemotron3.5- Lightning. Numerics are given in Table 9.

2. Additional Model families - Llama3.1 (8B drafting for 70B) serving as a full-attention example, and GPTOSS (20B drafting for 120B) serving as a mixed-attention example. Numerics are given in Table 10.

3. We provide additional cross-family drafting results. First, we consider additional draft models for the DeepSeek-V4 targets (Flash and Pro) in Tables 11 and 12 We consider Qwen3.5-4B, Qwen3.5-0.8B and Llama3.2- 1B. Second, we consider a Kimi-K3 target model and a Qwen3.5-4B draft in Table 13.

It can be seen that the performance of RIFT is maintained across all considered settings. It can also be seen that some setups benefit less from RIFT-stat, showing that similar performance can be obtained with SWA at the same budget (see, for example, Nemotron3-Nano drafting for Nemotron3-Ultra or Super in Table 9). However, the benefit of dynamically rescoring and updating the KV cache during decode dramatically increases the draft’s performance, regardless of the specific setup considered.

![](images/989a3402890c4a9e39f93307b6b293a403b1a755d9845bc781e28be5849d84e2.jpg)  
Figure 10: Interval AL across model families, with full-KV references. Target/draft pairs (left to right): GPT-OSS-120B/20B, Nemotron-Super-120B/Nano-30B, Qwen3-235B/8B, and Llama-3.1-70B/8B. Each panel uses its own AL scale. The corresponding Qwen3.5 comparison appears in Figure 1(a).

![](images/f4c091477b02a82738c6d2ccf1fa706bdc5a6f9b9aca30156788b399bbe9a024.jpg)

![](images/c5665520a231472990aa8c0431ec464bdcd530a13c85c8277a2c131c01bbd04d.jpg)  
Figure 11: Interval AL versus generated-token position for Qwen3.5-4B drafting Qwen3.5-122B responses, with thinking enabled and disabled. Reasoning magnifies the late staleness gap, but the SnapKV-SW crossover persists even without it.

![](images/93129e4a7b60ff3c0dffcee03b50e69749bb83eb02521a72f9d93eb8375528fb.jpg)

![](images/79eac993f71d2b79fb4d109b585e515f343cfb35a14a08d5852233efd1da1ba9.jpg)

![](images/30bcf257433838ab32143929369d56df9673df43c6c011b6e20dd1f7042f7c7f.jpg)

![](images/0f2b586b27d188f6ae363bf3285a4931d274dd9f88450d07ff5ff6f57a893eda.jpg)

![](images/ecf932fba2241844dcafd57a5314ea2a6a5be9a1e75de43766f5d3a2050ddb64.jpg)  
Figure 12: Additional sliding-window ratio ablations with the same base draft model, requests, and evaluation settings as Figure 3. Panel titles specify the output-length cap and KV budget. Stars mark $\rho = 0 . 5$ in the first four panels and the optimum $\rho = 0 . 6$ in the 1K-output, 1K-budget panel (AL 3.648, versus 3.597 at $\rho = 0 . 5 )$ ; the zero-labeled endpoint denotes fixed-64 SnapKV, and $\rho = 1$ denotes pure SWA. Broken diagonals indicate omitted AL range.

![](images/4a84443be4b703d373d79eba8f70d29a3f7d2211e7e636d0c3313e674e3352b5.jpg)  
Figure 13: Effect of draft length for Qwen3.5-4B drafting for Qwen3.5-122B-A10B on low-entropy SPEED-Bench 32K prompts, with a 4K output cap. Compressed methods use an 8K KV budget and block size 128; Full-KV retains the full context. Each point is mean per-request AL.

## D.5 Additional Prefill Selectors

Table 14 reports the absolute ALs underlying the relative gains in Table 2.

## D.6. Effect of draft length

We evaluate the effect of draft length on performance in Full-KV, SnapKV, SW, RIFT-stat, and RIFT-dyn at $T \in \{ 0 , 1 \}$ . As seen in Figure 13, the performance gap persists across draft lengths: RIFT-stat and RIFT-dyn consistently outperform SnapKV, with RIFT-dyn closest to Full-KV. The relative AL gain of RIFT-dyn over SnapKV grows from 17.7% to $6 0 . 0 \%$ at $T = 0$ and from 17.9% to 54.8% at $T = 1$ as draft length increases.

## D.7. AL vs. OSL

We evaluate how acceptance changes with OSL on SPEED-Bench 32K low-entropy data. We force each target response to meet OSL shown in Figure 14. In this forced-output setting, the advantage of RIFT-stat over SWA diminishes as OSL grows: their mean-AL gap narrows from 0.42 to 0.09 at $T = 0$ and from 0.29 to 0.03 at $T = 1$ between 2K and 64K OSL. In contrast, the advantage of the decode-level methods grows with OSL. We note that forcing high OSLs produces highly repetitive generated text, which may affect both the measured ALs and the relative gaps between methods. Thus, the observed trends should therefore be interpreted as results for forced continuations rather than naturally long, meaningful responses.

## E RIFT-Ada: Additional Results and Analysis

We provide additional numerical results for RIFTada, considering additional datasets, budgets and models. Furthermore, we provide additional information on the calibration and uncertainty of the calibrated threshold.

The example refresh schedule and its score trace are shown in Figure 5.

## E.1 Additional Results

Qwen3.5-4B Figure 16 compares AL–refresh tradeoffs across temperatures, KV budgets, and input lengths using setting-specific controllers, with parameters held fixed across budgets.

## E.2 Adaptive Refresh for Nemotron Lightning Self-Speculation

We provide additional analysis of RIFT-ada considering a self-speculative decoding setup with Nemotron3.5-Lightning (NVIDIA, 2026b). We consider SPEED-Bench 32k low entropy data, with draft length of 11 and 4K OSL. Following modelspecific calibration, we refresh when aggregated acceptance falls below a threshold at $T = 0 _ { : }$ , or when either acceptance is low or draft entropy is high at $T = 1$ . Aggregated acceptance uses the last four completed verification steps with exponential smoothing $( \alpha = 0 . 4 )$ . Additionally, we use minimum and maximum refresh ages of 32 and 768 tokens respectively. As seen in Figure 17, RIFT-ada significantly improves upon RIFT-dyn, allowing for similar ALs with a fraction of the overall number of refreshes. These additional results further highlight the necessity of adaptive gating for KV recomputation.

![](images/2e00d97496bb23713adbab41c0c85cca2c7d0c8a55f79e0422318a45ea98ba14.jpg)

Figure 14: Mean per-request acceptance length versus forced output sequence length for Qwen3.5-4B drafting for Qwen3.5-122B-A10B on SPEED-Bench 32K low-entropy prompts. All compressed methods use an 8K KV budget and draft length 11.
<table><tr><td>ISL</td><td>Entropy</td><td>T</td><td>Budget</td><td>SW</td><td>SnapKV</td><td>RIFT-stat</td><td>RIFT-dyn</td><td>RIFT-ada</td></tr><tr><td rowspan="7">32K</td><td rowspan="3">Low</td><td rowspan="3">0</td><td>2K</td><td>64.8%</td><td>49.7%</td><td>70.3%</td><td>79.5%</td><td>81.5%</td></tr><tr><td>4K</td><td>71.3%</td><td>54.5%</td><td>77.7%</td><td>88.5%</td><td>90.2%</td></tr><tr><td>8K</td><td>80.9%</td><td>63.0%</td><td>86.9%</td><td>94.6%</td><td>95.8%</td></tr><tr><td rowspan="4">Low</td><td rowspan="4">1</td><td>2K</td><td>61.8%</td><td>47.1%</td><td>65.9%</td><td>74.7%</td><td>77.0%</td></tr><tr><td>4K</td><td>68.0%</td><td>52.0%</td><td>72.5%</td><td>84.1%</td><td>86.3%</td></tr><tr><td>8K</td><td>78.6%</td><td>61.8%</td><td>83.4%</td><td>91.7%</td><td>92.8%</td></tr><tr><td>2K</td><td>79.1%</td><td>61.9%</td><td>81.8%</td><td>88.5%</td><td>90.1%</td></tr><tr><td rowspan="4">High Low</td><td rowspan="4"></td><td>0 4K</td><td></td><td>84.0% 64.4%</td><td>88.8%</td><td>94.1%</td><td>94.7%</td></tr><tr><td>8K</td><td>85.9%</td><td>66.9%</td><td>93.1%</td><td>97.6%</td><td>97.7%</td></tr><tr><td>2K</td><td>64.1%</td><td>50.7%</td><td>72.1%</td><td>81.7%</td><td>84.7%</td></tr><tr><td>0 4K</td><td>75.5%</td><td>55.5%</td><td>77.8%</td><td>90.3%</td><td>92.4%</td></tr><tr><td rowspan="4">100K High</td><td rowspan="4"></td><td>8K</td><td>81.8%</td><td>59.7%</td><td>85.6%</td><td>95.5%</td><td>96.0%</td></tr><tr><td></td><td>2K</td><td>72.7% 60.0%</td><td>80.2%</td><td>86.9%</td><td>88.6%</td></tr><tr><td>0 4K</td><td>81.1%</td><td>62.5%</td><td>86.7%</td><td>92.1%</td><td>93.2%</td></tr><tr><td>8K</td><td>84.7%</td><td>65.2%</td><td>90.7%</td><td>95.9%</td><td>96.3%</td></tr></table>

Table 7: retained AL (%) of full-KV, Qwen3.5-4B draft with, Qwen3.5-122B target across SPEED-Bench workloads (ISL and entropy category) and cache budgets. The absolute full-KV ALs for 32K low $T = 0 / 1$ , 32K high $T = 0 ,$ 100K low $T = 0 ,$ , and 100K high $T = 0$ are 6.466/5.978, 4.617, 7.472, and 4.506, respectively. All settings use draft length 11 and maximum output length 4K.
<table><tr><td>ISL</td><td>Entropy</td><td>T</td><td>SW</td><td>SnapKV</td><td>RIFT-stat</td><td>RIFT-dyn</td><td>RIFT-ada</td></tr><tr><td rowspan="4">32K</td><td>Low</td><td>0</td><td>85.5%</td><td>67.3%</td><td>89.0%</td><td>95.9%</td><td>98.1%</td></tr><tr><td>Low</td><td>1</td><td>85.5%</td><td>68.2%</td><td>86.9%</td><td>94.2%</td><td>96.4%</td></tr><tr><td>High</td><td>0</td><td>91.5%</td><td>75.0%</td><td>95.6%</td><td>97.9%</td><td>98.6%</td></tr><tr><td>High</td><td>1</td><td>92.6%</td><td>77.7%</td><td>94.9%</td><td>97.1%</td><td>97.9%</td></tr><tr><td rowspan="3">100K</td><td>Low</td><td>0</td><td>82.6%</td><td>61.0%</td><td>87.5%</td><td>94.6%</td><td>97.4%</td></tr><tr><td>Low</td><td>1</td><td>84.6%</td><td>62.0%</td><td>86.9%</td><td>92.7%</td><td>97.1%</td></tr><tr><td>High</td><td>0</td><td>90.6%</td><td>71.9%</td><td>94.6%</td><td>96.3%</td><td>97.7%</td></tr></table>

Table 8: Base Qwen3.5-0.8B draft performance at an 8K cache budget, expressed as a percentage of the matching Full KV AL for Qwen3.5-122B target responses. The absolute full-KV ALs for 32K low $T = 0 / 1$ and high $T = 0 / 1$ are 4.807/4.148 and 3.272/2.836; for 100K low $T = 0 / 1$ and high $T = 0 ,$ they are 5.763/4.748 and 3.225.

![](images/d4bad5e5d11629a6ff1564fce532c67ead0e9484ccbbe67d3bd1460265a31f08.jpg)  
Figure 15: Effect of SnapKV block size for Qwen3.5-4B drafting for Qwen3.5-122B with a 50/50 local–remote cache split. The 32K experiments use an 8K cache budget; the 100K experiments use a 16K budget. The top and bottom rows report $T = 0$ and $T = 1$ , respectively. Moderate block sizes are broadly robust, whereas very coarse blocks reduce selection resolution and lower AL.

![](images/a5002d198a6020b6fde528a320e1e689420b0e79a7d3cc07c7414d168878835b.jpg)  
Figure 16: RIFT-ada AL-refresh tradeoffs for Qwen3.5-4B drafting Qwen3.5-122B on SPEED-Bench low-entropy data, with a 4K maximum OSL. Titles specify ISL, temperature, and KV cache budget.

<table><tr><td>Draft</td><td>Target</td><td>T</td><td>Budget</td><td>SW</td><td>SnapKV</td><td>RIFT-stat</td><td>RIFT-dyn</td></tr><tr><td rowspan="3">Nano</td><td rowspan="3">Super</td><td rowspan="3">0</td><td>2K</td><td>53.0%</td><td>41.8%</td><td>52.6%</td><td>62.1%</td></tr><tr><td>4K</td><td>57.1%</td><td>45.1%</td><td>57.2%</td><td>73.2%</td></tr><tr><td>8K</td><td>63.4%</td><td>52.8%</td><td>63.9%</td><td>83.4%</td></tr><tr><td rowspan="3">Nano</td><td rowspan="3">Ultra</td><td rowspan="3">0</td><td>2K</td><td>69.1%</td><td>56.0%</td><td>69.9%</td><td>76.6%</td></tr><tr><td>4K</td><td>74.4%</td><td>60.5%</td><td>75.2%</td><td>85.9%</td></tr><tr><td>8K</td><td>81.3%</td><td>68.4%</td><td>82.8%</td><td>93.2%</td></tr><tr><td rowspan="6">Lightning Super</td><td rowspan="6"></td><td>0</td><td>2K</td><td>52.6%</td><td>42.1%</td><td>53.3%</td><td>66.9%</td></tr><tr><td></td><td>4K 8K</td><td>56.3%</td><td>45.1%</td><td>56.9%</td><td>78.2% 90.7%</td></tr><tr><td></td><td></td><td>62.2%</td><td>51.9%</td><td>62.2%</td><td></td></tr><tr><td rowspan="3">1</td><td>2K</td><td>56.1%</td><td>44.8%</td><td>57.1%</td><td>68.4%</td></tr><tr><td>4K 8K</td><td>60.7% 68.2%</td><td>48.6% 56.2%</td><td>61.7% 68.1%</td><td>78.7% 90.2%</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="6">Lightning</td><td rowspan="3">Ultra</td><td>0</td><td>2K</td><td>67.0%</td><td>54.6%</td><td>69.6%</td><td>77.2%</td></tr><tr><td></td><td>4K</td><td>72.7%</td><td>59.3%</td><td>74.0%</td><td>84.8%</td></tr><tr><td></td><td>8K</td><td>79.9%</td><td>67.4%</td><td>80.7%</td><td>92.6%</td></tr><tr><td rowspan="3">1</td><td></td><td>2K</td><td>66.1%</td><td>53.8%</td><td>69.8%</td><td>76.7%</td></tr><tr><td></td><td>4K</td><td>71.8%</td><td>58.1%</td><td>74.1%</td><td>83.9%</td></tr><tr><td>8K</td><td>79.5%</td><td></td><td>66.2%</td><td>80.8%</td><td>91.6%</td></tr><tr><td rowspan="6">Lightning Lightning</td><td rowspan="3"></td><td></td><td>2K</td><td>43.0%</td><td>35.8%</td><td>46.8%</td><td>57.6%</td></tr><tr><td>0</td><td>4K</td><td>49.7%</td><td>40.5%</td><td>54.1%</td><td>70.2%</td></tr><tr><td></td><td>8K</td><td>57.3%</td><td>48.3%</td><td>67.2%</td><td>82.6%</td></tr><tr><td></td><td>2K</td><td>35.3%</td><td>29.2%</td><td>40.2%</td><td>51.2%</td></tr><tr><td>1</td><td>4K</td><td>41.4%</td><td>33.4%</td><td>47.1%</td><td>64.2%</td></tr><tr><td></td><td>8K</td><td>49.5%</td><td>40.2%</td><td>60.9%</td><td>77.5%</td></tr></table>

Table 9: Full-KV AL retention in Nemotron3-family, low-entropy SPEED-Bench 32K split. The absolute full-KV ALs are 6.944 (Nano → Super, $T = 0 )$ , 4.994 (Nano → Ultra, $T = 0 )$ , 6.647/5.520 (Lightning → Super, $T = 0 / 1 )$ 5.394/4.859 (Lightning $ \mathrm { U l t r a } , T = 0 / 1 )$ , and 10.009/11.222 (Lightning self-speculation, $T = 0 / 1 )$ .
<table><tr><td>Draft → target</td><td>Budget</td><td>SW</td><td>SnapKV</td><td>RIFT-stat</td><td>RIFT-dyn</td></tr><tr><td rowspan="3">Llama-3.1-8B → Llama-3.1-70B</td><td>2K</td><td>26.6%</td><td>53.7%</td><td>64.3%</td><td>70.5%</td></tr><tr><td>4K</td><td>27.6%</td><td>58.1%</td><td>70.1%</td><td>79.7%</td></tr><tr><td>8K</td><td>28.6%</td><td>66.1%</td><td>77.4%</td><td>88.0%</td></tr><tr><td rowspan="3">GPT-OSS-20B → GPT-OSS-120B</td><td>2K</td><td>62.3%</td><td>32.0%</td><td>56.4%</td><td>67.7%</td></tr><tr><td>4K</td><td>65.5%</td><td>32.6%</td><td>64.9%</td><td>76.8%</td></tr><tr><td>8K</td><td>69.6%</td><td>34.0%</td><td>71.4%</td><td>87.8%</td></tr></table>

Table 10: Llama3.1 and GPT-OSS cache-budget sweeps on the $T = 0 ,$ , low-entropy SPEED-Bench 32K workload. Values are AL as a percentage of the matching full-KV AL. The absolute full-KV AL is 7.849 for Llama and 5.746 for GPT-OSS. All settings use draft length 11 and maximum output length 4K.

## E.3 Adaptive-Refresh Policies and Calibration

Calibration and policy selection. We organize controller development around three sets: calibration, policy selection, and confirmation. We begin by collecting the signals on the calibration set, from which we determine the threshold search sets. The set of thresholds is determined from the corresponding percentiles which are derived from the signals’ statistics. Next, the policy-selection set compares measured AL-refresh tradeoffs for the various threshold candidates. We calibrate the controller separately for each draft/target-model and temperature setting. For the reported budget sweeps, controller parameters are kept fixed across budgets.

Selected policies. Generation settings change draft entropy, acceptance statistics, and the useful refresh operating range, motivating separate controller calibration. The selected policies therefore differ across models and temperatures, and may be further tailored to a particular generation setting. Acceptance-based rules use exponential weighting of 4 completed verification steps with coefficient 0.4. Entropy is averaged over the latest decision stride. Signals are updated after each completed decision stride, before deciding whether to refresh. Query drift compares the current grouped query summary with the summary stored at the last refresh.

(a) 32K ISL
<table><tr><td>Draft</td><td>Budget SnapKV 1</td><td></td><td>RIFT-stat RIFT-dyn</td><td></td></tr><tr><td rowspan="5">Qwen3.5-0.8B</td><td>2K</td><td>48.4%</td><td>63.3%</td><td>78.8%</td></tr><tr><td>4K</td><td>54.7%</td><td>69.9%</td><td>87.6%</td></tr><tr><td>8K</td><td>63.2%</td><td>77.6%</td><td>93.4%</td></tr><tr><td>16K</td><td>70.5%</td><td>84.7%</td><td>97.5%</td></tr><tr><td>2K</td><td>49.3%</td><td>63.5%</td><td>78.9%</td></tr><tr><td rowspan="3">Qwen3.5-4B</td><td>4K</td><td>54.7%</td><td>70.0%</td><td>88.2%</td></tr><tr><td>8K</td><td>63.7%</td><td>79.0%</td><td>94.1%</td></tr><tr><td>16K</td><td>70.1%</td><td>85.2%</td><td>98.0%</td></tr><tr><td rowspan="4">Llama3.2-1B</td><td>2K</td><td>51.6%</td><td>66.8%</td><td>70.6%</td></tr><tr><td>4K</td><td>54.1%</td><td>71.3%</td><td>75.9%</td></tr><tr><td>8K</td><td>59.8%</td><td>77.3%</td><td>82.1%</td></tr><tr><td>16K</td><td>67.4%</td><td>84.7%</td><td>90.5%</td></tr></table>

(b) 100K ISL
<table><tr><td>Draft</td><td></td><td>Budget SnapKV RIFT-stat RIFT-dyn</td><td></td><td></td></tr><tr><td rowspan="5">Qwen3.5-0.8B</td><td>2K</td><td>57.9%</td><td>76.1%</td><td>84.6%</td></tr><tr><td>4K</td><td>63.1%</td><td>81.2%</td><td>92.4%</td></tr><tr><td>8K</td><td>66.8%</td><td>88.9%</td><td>96.0%</td></tr><tr><td>16K</td><td>69.4%</td><td>92.2%</td><td>97.7%</td></tr><tr><td>2K</td><td>56.5%</td><td>74.2%</td><td>83.3%</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>4K</td><td>62.3%</td><td>79.5%</td><td>91.5%</td></tr><tr><td>8K</td><td>65.9%</td><td>87.6%</td><td>95.2%</td></tr><tr><td>16K</td><td>68.5%</td><td>91.5%</td><td>97.1%</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Llama3.2-1B</td><td>2K</td><td>56.1%</td><td>77.5%</td><td>80.4%</td></tr><tr><td>4K</td><td>58.6%</td><td>82.2%</td><td>85.0%</td></tr><tr><td>8K</td><td>61.8%</td><td>89.4%</td><td>91.4%</td></tr><tr><td>16K</td><td>65.0%</td><td>92.3%</td><td>94.4%</td></tr></table>

Table 11: DeepSeek V4 Pro with trained external drafters on the $T = 0 ,$ low-entropy SPEED-Bench workloads. For Qwen3.5-0.8B, Qwen3.5-4B, and Llama3.2-1B, respectively, the absolute full-KV ALs are 4.343/5.389/3.969 at 32K and 4.263/5.427/3.674 at 100K.

(a) 32K ISL
<table><tr><td>Draft</td><td>Budget SnapKV</td><td></td><td>RIFT-stat RIFT-dyn</td><td></td></tr><tr><td rowspan="5">Qwen3.5-0.8B</td><td>2K</td><td>40.9%</td><td>54.4%</td><td>75.0%</td></tr><tr><td>4K</td><td>50.6%</td><td>62.6%</td><td>85.5%</td></tr><tr><td>8K</td><td>59.8%</td><td>72.4%</td><td>92.7%</td></tr><tr><td>16K</td><td>67.9%</td><td>79.3%</td><td>97.2%</td></tr><tr><td>2K</td><td>42.7%</td><td>55.3%</td><td>75.1%</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>4K</td><td>51.3%</td><td>63.1%</td><td>86.4%</td></tr><tr><td>8K</td><td>61.3%</td><td>74.5%</td><td>93.7%</td></tr><tr><td>16K</td><td>68.9%</td><td>80.7%</td><td>97.8%</td></tr><tr><td>2K</td><td>46.6%</td><td>59.3%</td><td></td></tr><tr><td rowspan="4">Llama3.2-1B</td><td></td><td></td><td></td><td>64.1%</td></tr><tr><td>4K</td><td>50.1%</td><td>64.9%</td><td>71.2%</td></tr><tr><td>8K</td><td>56.7%</td><td>73.6%</td><td>79.6%</td></tr><tr><td>16K</td><td>65.7%</td><td>81.0%</td><td>89.4%</td></tr></table>

(b) 100K ISL
<table><tr><td>Draft</td><td></td><td>Budget SnapKV</td><td>RIFT-stat</td><td>RIFT-dyn</td></tr><tr><td rowspan="5">Qwen3.5-0.8B</td><td>2K</td><td>55.0%</td><td>62.2%</td><td>78.7%</td></tr><tr><td>4K</td><td>63.4%</td><td>71.1%</td><td>89.0%</td></tr><tr><td>8K</td><td>70.4%</td><td>86.1%</td><td>95.0%</td></tr><tr><td>16K</td><td>76.3%</td><td>92.7%</td><td>97.3%</td></tr><tr><td>2K</td><td>56.0%</td><td>62.5%</td><td>79.8%</td></tr><tr><td rowspan="4">Qwen3.5-4B</td><td>4K</td><td>65.0%</td><td>71.2%</td><td>89.6%</td></tr><tr><td>8K</td><td>71.8%</td><td>86.4%</td><td>95.1%</td></tr><tr><td>16K</td><td>76.4%</td><td>92.9%</td><td>97.6%</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="4">Llama3.2-1B</td><td>2K</td><td>49.2%</td><td>61.4%</td><td>64.8%</td></tr><tr><td>4K</td><td>54.0%</td><td>69.6%</td><td>75.1%</td></tr><tr><td>8K</td><td>60.2%</td><td>84.4%</td><td>87.9%</td></tr><tr><td>16K</td><td>67.0%</td><td>91.5%</td><td>93.6%</td></tr></table>

Table 12: DeepSeek V4 Flash with trained external drafters on the $T = 0 ,$ , low-entropy SPEED-Bench workloads. For Qwen3.5-0.8B, Qwen3.5-4B, and Llama3.2-1B, respectively, the absolute full-KV ALs are 4.830/5.714/4.401 at 32K and 5.308/6.338/4.896 at 100K.
<table><tr><td></td><td colspan="3">RIFT-stat</td><td colspan="3">RIFT-dyn</td><td colspan="3">RIFT-ada</td></tr><tr><td>T</td><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td>0</td><td>60.5%</td><td>66.0%</td><td>72.7%</td><td>84.3%</td><td>93.4%</td><td>97.1%</td><td>88.4%</td><td>94.8%</td><td>98.3%</td></tr><tr><td>1</td><td>56.6%</td><td>62.5%</td><td>69.3%</td><td>81.5%</td><td>91.6%</td><td>96.1%</td><td>87.2%</td><td>94.7%</td><td>98.2%</td></tr></table>

Table 13: Post-training AL as a percentage of the matching full-KV AL for cross-tokenizer Qwen3.5-4B → Kimi-K3 drafting. The absolute full-KV AL is 4.834/5.138 at $T = 0 / 1$
<table><tr><td>Budget</td><td>Selector</td><td> $T = 0$ </td><td> $T = 1$ </td></tr><tr><td rowspan="3">4K</td><td>CriticalKV</td><td> $3 . 5 7  5 . 0 2 ( + 4 0 . 6 \% )$ </td><td> $3 . 1 5  4 . 3 3 ( + 3 7 . 3 \% )$ </td></tr><tr><td>Expected Attention</td><td> $3 . 5 9  5 . 0 3 \ : \dot { ( } + 4 0 . 2 \% )$ </td><td> $3 . 0 9  4 . 3 0 \ ( + 3 9 . 0 \% )$ </td></tr><tr><td>CurDKV</td><td> $3 . 2 9 \to 4 . 8 4 \ ( + 4 7 . 1 \% )$ </td><td> $2 . 8 4  4 . 1 4 \ : \dot { ( + 4 5 . 5 \% ) }$ </td></tr><tr><td rowspan="3">8K</td><td>CriticalKV</td><td> $4 . 1 3  5 . 6 2 ( + 3 6 . 2 \% )$ </td><td> $3 . 7 4 \to 4 . 9 8 ( + 3 3 . 1 \% )$ </td></tr><tr><td>Expected Attention</td><td> $4 . 0 6  5 . 5 6 \ : \dot { ( } + 3 6 . 9 \% )$ </td><td> $3 . 5 6 \to 4 . 8 3 \ ( + 3 5 . 9 \% )$ </td></tr><tr><td>CurDKV</td><td> $3 . 7 6  5 . 3 5 \ : \dot { ( } + 4 2 . 2 \% )$ </td><td> $3 . 2 6 \to 4 . 6 3 \ ( + 4 2 . 1 \% )$ </td></tr></table>

Table 14: RIFT-stat extends beyond SnapKV: Qwen3.5-4B draft (Qwen3.5-122B target), 32K ISL, 4K OSL with draft length of 11. Each entry shows baseline $\mathrm { A L }  \mathrm { A L }$ with RIFT-stat, followed by the relative improvement.

![](images/0e938d1401f3e784fe0598839a5b09e66dc2dadc7c2204e100d96a7f8787c6c6.jpg)  
Figure 17: Nemotron3.5-Lightning self-speculation RIFT-ada refresh - AL tradeoff. SPEED-Bench 32k ISL, low entropy category.

The RIFT-ada curve in Figure 6 uses an orcombined rule. Let $A _ { t }$ be the four-completed-step exponentially weighted acceptance length $( \alpha \ =$ 0.4), $H _ { t }$ the mean draft entropy over the latest decision stride, and $D _ { t } ^ { q }$ the mean query drift. Let $A _ { \mathrm { r e f } }$ and $H _ { \mathrm { r e f } }$ denote the post-refresh references. We denote the collection of signals with $z _ { t } .$ Thus, the signal trigger is given by

$$
\begin{array} { r l } & { \pi ( z _ { t } ) = ( H _ { t } \geq 2 . 1 1 ) } \\ & { \qquad \mathrm { o r ~ } ( A _ { t } \leq 2 . 1 9 ) } \\ & { \qquad \mathrm { o r ~ } ( H _ { t } - H _ { \mathrm { r e f } } \geq 1 . 3 5 ) } \\ & { \qquad \mathrm { o r ~ } ( A _ { \mathrm { r e f } } - A _ { t } \geq 4 . 8 4 ) } \\ & { \qquad \mathrm { o r ~ } ( D _ { t } ^ { q } \geq \tau _ { D } ) . } \end{array}
$$

To obtain the curve, we keep the other thresholds fixed and sweep $\tau _ { D }$ over {0.5, 0.43, 0.33, 0.26, 0.22, 0.2, 0.16}.

Uncertainty and evaluation scope. We calibrate separate controllers for each temperature, using disjoint calibration, evaluation, and confirmation subsets during development. We compare fitted monotonic-score policies with acceptance/entropy level and deterioration rules, including and/or combinations with query drift. For the Qwen results in Figures 16(c) $( T ~ = ~ 0 )$ and $6 ~ ( T ~ = ~ 1 )$ , the selected policies use level and deterioration signals at $T = 0$ , and an additional query-drift or trigger at $T = 1$ , reducing refreshes at matched

AL by 53.5% and 51.1%, respectively. Pairedbootstrap 95% confidence intervals exclude zero for both improvements over periodic refresh, conditional on the selected policies. The final results include the development subsets, so these results describe selected-policy performance rather than an untouched held-out evaluation.

Measuring improvement in the refresh-AL tradeoff. For an adaptive operating point $( R , A )$ where R is the average number of refreshes and A is AL, we report

$$
\begin{array} { l } { \displaystyle \Delta A | _ { R } = A - A _ { \mathrm { f i x } } ( R ) , } \\ { \displaystyle \Delta R | _ { A } = 1 0 0 \biggl ( 1 - \frac { R } { R _ { \mathrm { f i x } } ( A ) } \biggr ) \% . } \end{array}\tag{1}
$$

Here $A _ { \mathrm { f i x } } ( R )$ is periodic refresh’s AL at matched refresh count, and $R _ { \mathrm { f i x } } ( A )$ is its refresh count at matched AL.

## F Beyond Speculative Decoding: Long-Context Quality Evaluation

We provide preliminary evidence for RIFT-stat’s performance beyond speculative decoding in a long-OSL setting. We evaluate Qwen3.5-4B on LongBench-v1 and LongBench-v2 (Bai et al., 2024, 2025).

Evaluation protocol. To meet a long OSL setup, we enforce a 4K reasoning phase. Following prefill, the model greedily generates up to 4,096 reasoning tokens. Then, it receives a task-specific final-answer request. Generation continues with the same cache, without re-prefill or reselection. Answer limits are 32-512 tokens for LongBench-v1 and 128 for LongBench-v2. We report the average of task scores. We count parse failures as incorrect (e.g., no valid choice in a multi-option question). We compare SnapKV (with a 64 token SW) and RIFT-stat.

<table><tr><td>Benchmark</td><td>Budget (Avg. compression)</td><td>SnapKV</td><td>RIFT-stat</td></tr><tr><td rowspan="4">LongBench-v1</td><td>1K (9.95×)</td><td>60.7</td><td> $7 6 . 2 \ : ( + 1 5 . 5 )$ </td></tr><tr><td>2K (4.98×)</td><td>61.7</td><td>81.8 (+20.1)</td></tr><tr><td>4K (2.49×)</td><td>63.4</td><td>89.5 (+26.1)</td></tr><tr><td>8K (1.24×)</td><td>67.2</td><td>96.4 (+29.2)</td></tr><tr><td rowspan="4">LongBench-v2</td><td>1K (119.41×)</td><td>60.3</td><td> $6 1 . 9 \left( + 1 . 6 \right)$ </td></tr><tr><td>2K (59.70×)</td><td>67.7</td><td> $7 0 . 4 \ : ( + 2 . 6 )$ </td></tr><tr><td>4K (29.85×)</td><td>68.3</td><td> $7 5 . 7 \ : ( + 7 . 4 )$ </td></tr><tr><td>8K (14.93×)</td><td>69.3</td><td>89.9 (+20.6)</td></tr></table>

Table 15: Full-KV score retention (%) for Qwen3.5-4B under the modified LongBench protocol described above. Absolute full-KV scores are 47.607 for LongBench-v1 and 38.569 for LongBench-v2.

Results. As seen in Table 15 RIFT-stat uniformly outperforms SnapKV at every tested budget. At 8K, it retains 96.4% and 89.9% of full-KV scores on v1 and v2, respectively. These results suggest that recency allocation can also benefit standalone generation. We note that, while promising, these are preliminary results, and a comprehensive evaluation beyond speculative decoding is left to future work.

## G RIFT Performance: Task Breakdown

To provide an additional point of view into RIFT’s performance, we break down acceptance length by SPEED-Bench task. We consider Qwen3.5- 4B drafting for Qwen3.5-122B and consider data from ISL splits 16K, 32K and 100K. Table 16 presents nine selected categories. It can be seen that the order of performance (SnapKV<SW<RIFTstat <RIFT-dyn) is maintained across all categories, across various generation modes induces by different tasks

## H Tokenizer Adaptation for Cross-Family Drafting

We adopt the following tokenizer-adaptation procedure to initialize a cross-family draft before further draft–target alignment:

1. Resize. Resize draft input embeddings and the LM-head to the target’s configured vocabulary size, retaining the draft’s hidden dimension.

2. Copy matching tokens. For each target token, first try a manually role-matched control token (for special tokens). If the token is no a special token, look for an exact match, and if such a match is not found look for a normalized match in the draft vocabulary.

3. Initialize remaining tokens. Retokenize unmatched text with the draft tokenizer and combine up to four embeddings using geometrically decreasing weights.

4. Align outputs. Apply the same mapping to an untied output head, preserve tied embeddings, and adopt the target tokenizer’s beginning-ofsequence, end-of-sequence, and padding IDs.

Once the draft’s vocabulary and corresponding embedding matrices are aligned, we perform a knowledge distillation scheme over the Nemotron posttraining V2 dataset over 10K steps.

## I Comparison With Additional KV Compression Policies

We provide additional comparisons with KV compression methods, complementing the results in the main paper. We evaluate on SPEED-Bench with Qwen3.5-4B drafting for Qwen3.5-122B and, in Table 17, Llama3.1-8B drafting for Llama3.1-70B. We consider 4K OSL and a draft length of 11. We consider the following methods:

1. StreamingLLM (Xiao et al., 2024), which retains a SW of recent tokens together with four initial attention-sink tokens.

2. $_ \mathrm { H _ { 2 } O }$ (Zhang et al., 2023), which updates accumulated attention scores at each decoding step and retains a combination of recent tokens and heavy hitters, permanently evicting other positions.

3. TriAttention (Mao et al., 2026), which uses calibrated pre-RoPE query statistics to estimate key importance through RoPE-induced trigonometric distance preferences and normbased scoring. Specifically, we estimate the mean pre-RoPE query vectors from full-prefill statistics and hold them fixed during decoding, and consider a refresh interval similar to RIFT-dyn of 128 tokens.

<table><tr><td>Task</td><td>SnapKV SWA</td><td>RIFT-stat</td><td>RIFT-dyn</td><td>RIFT-ada</td></tr><tr><td colspan="5">T = 0</td></tr><tr><td>Sorting</td><td>3.65 3.96</td><td>4.84</td><td>5.35</td><td>5.61</td></tr><tr><td>Creative writing</td><td>3.90 4.86</td><td>5.79</td><td>6.29</td><td>6.28</td></tr><tr><td>NIAH</td><td>4.30 4.86</td><td>5.50</td><td>6.16</td><td>6.18</td></tr><tr><td>Code debugging</td><td>3.76 5.02</td><td>5.58</td><td>6.67</td><td>6.80</td></tr><tr><td>Creative writing</td><td>3.03 3.91</td><td>4.18</td><td>4.39</td><td>4.42</td></tr><tr><td>Domain QA — Math</td><td>3.26 5.64</td><td>5.82</td><td>5.91</td><td>5.91</td></tr><tr><td>Domain QA — Chemistry</td><td>3.75 6.22</td><td>6.45</td><td>6.47</td><td>6.50</td></tr><tr><td>Domain QA — Biology/Medicine</td><td>3.50 5.78</td><td>5.98</td><td>6.03</td><td>6.03</td></tr><tr><td>Domain QA — Engineering</td><td>4.39 6.25</td><td>6.44</td><td>6.55</td><td>6.58</td></tr><tr><td colspan="5">T = 1</td></tr><tr><td>Sorting</td><td>3.18</td><td>3.57</td><td>4.22</td><td>4.78</td></tr><tr><td>Creative writing</td><td>3.13</td><td>4.85</td><td>4.58 5.40</td><td>5.39</td></tr><tr><td>NIAH</td><td>3.66</td><td>4.14 4.22</td><td>4.72 5.35</td><td>5.32</td></tr><tr><td>Code debugging</td><td>3.25</td><td>4.52</td><td>4.86 5.77</td><td>6.07</td></tr><tr><td>Creative writing</td><td>2.76</td><td>3.55</td><td>3.76 3.90</td><td>3.93</td></tr><tr><td>Domain QA — Math</td><td>2.83</td><td>5.19</td><td>5.32 5.39</td><td>5.39</td></tr><tr><td>Domain QA — Chemistry</td><td>3.26</td><td>5.84 5.95</td><td>6.01</td><td>6.01</td></tr><tr><td>Domain QA — Biology/Medicine</td><td>3.15</td><td>5.34</td><td>5.45</td><td>5.52 5.53</td></tr><tr><td>Domain QA — Engineering</td><td>3.79</td><td>5.90</td><td>6.00</td><td>6.09 6.10</td></tr></table>

Table 16: Mean AL using SPEED-Bench data of ISLs 16K, 32K and 100K, at $T = 0$ and $T = 1$ . Draft model is Qwen3.5-4B drafting for Qwen3.5-122B. Draft length is 11 and max OSL is 4K.
<table><tr><td>Model</td><td>ISL</td><td> $T$ </td><td>StreamingLLM</td><td> $\mathrm { R I F T - s t a t }$ </td><td> $\mathrm { R I F T - d y n }$ </td><td>RIFT-ada</td></tr><tr><td rowspan="3">Qwen3.5</td><td>32K</td><td>0</td><td>5.34</td><td>5.58</td><td>6.08</td><td>6.17</td></tr><tr><td></td><td>1</td><td>4.73</td><td>4.92</td><td>5.40</td><td>5.49</td></tr><tr><td rowspan="2">100K</td><td>0</td><td>5.60</td><td>5.77</td><td>6.61</td><td>6.73</td></tr><tr><td>1</td><td></td><td>4.98 5.07</td><td>5.76</td><td>5.92</td></tr><tr><td rowspan="4">Llama3.1</td><td>32K</td><td>0</td><td>5.87</td><td>5.87</td><td>6.58</td><td>7.08</td></tr><tr><td rowspan="2"></td><td>1</td><td>4.20</td><td>4.41</td><td>4.89</td><td>5.41</td></tr><tr><td>0</td><td>4.82</td><td>5.32</td><td>5.65</td><td>6.23</td></tr><tr><td rowspan="2">100K</td><td>1</td><td>3.35</td><td>3.58</td><td>3.70</td><td>3.84</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 17: Mean AL on SPEED-Bench low-entropy data. Qwen3.5-4B drafts for Qwen3.5-122B and Llama3.1-8B for Llama3.1-70B. All rows use an 8K budget, draft length 11, and a 4K maximum OSL. StreamingLLM retains an 8K window plus four sinks.

Tables 17 and 18 show higher AL for RIFT than StreamingLLM across the reported settings. For the other decode-time baselines, RIFT-dyn and RIFT-ada outperform both $_ \mathrm { H _ { 2 } O }$ and TriAttention in every reported setting, while $_ \mathrm { H _ { 2 } O }$ generally outperforms RIFT-stat (Table 18).

## J Draft Compressibility Training

## J.1 Mask Construction

To generate a SnapKV attention mask, we begin with a frozen dense forward pass and capture the post-RoPE queries and keys in every full-attention layer. Let O denote the observation window’s queries and $\mathcal { G } ( h )$ the query heads that share KV head h. We denote the dense post-softmax attention weight from query q in head g to key position k in KV head h by $A _ { g , h } ( q , k )$ . The importance of key position k is

$$
a _ { h } ( k ) = \frac { 1 } { \vert \mathcal { G } ( h ) \vert } \sum _ { g \in \mathcal { G } ( h ) } \sum _ { q \in \mathcal { O } } A _ { g , h } ( q , k ) .\tag{2}
$$

## J.2 Training Configuration

We consider 10K training steps using teachergenerated rollouts from the Nemotron V2 posttraining dataset, using a batch size of 64. The teacher responses and logits were generated at $T = 1$ . We train with a categorical distribution over $p \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ } whose values are {0.05, 0.10, 0.20, 0.25, 0.40} respectively. We use a SW ratio $\rho = 0 . 2 5$ during training. We consider

<table><tr><td> $T$ </td><td>Budget</td><td>StreamingLLM</td><td> $_ \mathrm { H _ { 2 } O }$ </td><td>TriAttention</td><td>RIFT-stat</td><td>RIFT-dyn</td><td>RIFT-ada</td></tr><tr><td colspan="8">ISL 32K</td></tr><tr><td>0</td><td>2K</td><td>4.52</td><td>4.68</td><td>4.54</td><td>4.54</td><td>5.14</td><td>5.28</td></tr><tr><td></td><td>4K</td><td>4.79</td><td>5.25</td><td>4.97</td><td>5.03</td><td>5.73</td><td>5.82</td></tr><tr><td></td><td>8K</td><td>5.34</td><td>5.73</td><td>5.43</td><td>5.62</td><td>6.12</td><td>6.17</td></tr><tr><td>1</td><td>2K</td><td>3.90</td><td>4.06</td><td>3.86</td><td>3.94</td><td>4.47</td><td>4.54</td></tr><tr><td></td><td>4K</td><td>4.17</td><td>4.53</td><td>4.27</td><td>4.34</td><td>5.03</td><td>5.10</td></tr><tr><td></td><td>8K</td><td>4.73</td><td>5.03</td><td>4.77</td><td>4.99</td><td>5.48</td><td>5.51</td></tr><tr><td colspan="8">ISL 100K</td></tr><tr><td>0</td><td>2K</td><td>4.84</td><td>5.35</td><td>5.17</td><td>5.39</td><td>6.11</td><td>6.33</td></tr><tr><td></td><td>4K</td><td>5.30</td><td>5.92</td><td>5.64</td><td>5.81</td><td>6.75</td><td>6.91</td></tr><tr><td></td><td>8K</td><td>5.60</td><td>6.65</td><td>6.10</td><td>6.40</td><td>7.14</td><td>7.17</td></tr><tr><td>1</td><td>2K</td><td>4.28</td><td>4.47</td><td>4.25</td><td>4.54</td><td>5.16</td><td>5.46</td></tr><tr><td></td><td>4K</td><td>4.67</td><td>5.06</td><td>4.69</td><td>4.98</td><td>5.80</td><td>6.09</td></tr><tr><td></td><td>8K</td><td>4.98</td><td>5.77</td><td>5.11</td><td>5.55</td><td>6.21</td><td>6.36</td></tr></table>

Table 18: Mean AL on SPEED-Bench low entropy for Qwen3.5-4B drafting Qwen3.5-122B, with draft length 11 and a 4K maximum OSL. StreamingLLM uses the stated window plus four sinks.

KD temperature $T _ { \mathrm { K D } } ~ = ~ 2 .$ , and $( \lambda _ { \mathrm { C E } } , \lambda _ { \mathrm { K D } } ) ~ =$ (0.25, 0.75). In evaluation, we consider an OSL of 4K and a draft length of 11, drafting on SPEED-Bench’s 32k throughput split, low entropy category. Additional results and ablation studies follow below.

## J.3 Ablation Studies

We present ablation studies on various components in the draft compressibility training scheme, motivating the design choices in this work. Unless stated otherwise, we train Qwen3.5-4B drafting for Qwen3.5-122B and evaluate ALs on SPEED-Bench 32K ISL split, low entropy category.

Masking Strategy We test the gain of the attention-score-based mask by comparing it with similar training runs that use different masking strategies, such as random or SWA-based masks. As seen in Table 19, the attention-based masking scheme outperforms every random and slidingwindow control at temperatures 0 and 1 and at all three considered KV budgets. It is worth noting that using an attention-based mask induces additional complexity to the overall training scheme, as we need to perform an additional dense forward pass.

Attention mask composition We study the effect of the distribution of the sparsity parameter, which determines the ratio of attention to mask out during training. Specifically, we consider dynamic $p$ values, in which $p$ is sampled from a given categorical distribution on a subset of [0, 1]. All distributions sample $p \in \{ 0 . 1 , 0 . 3 , 0 . 5 , 0 . 7 , 0 . 9 \}$ High biased samples with probabilities {0.05, 0.10, 0.20, 0.25, 0.40}, “soft” and “sharp” use probabilities {0.10, 0.15, 0.20, 0.25, 0.30} and {0.03, 0.07, 0.15, 0.25, 0.50}, respectively. Additionally, we test the effect of the training-time SW ratio value $\rho .$ As seen in Table 20, the high-biased $p$ distribution attains the best results, while the value of $\rho$ does not have a significant effect on performance.

Optimization and Parameterization We examine the effect of the fine-tune hyperparameters on the training scheme. Specifically, we are examine the effect of the number of trainable parameters on the resulting model’s performance and the learning rate behavior. Table 21, compares LoRA rank and learning-rate schedule for the same Qwen pair, averaging over 2K, 4K, and 8K budgets. As expected, full model fine tune results in slightly higher ALs than LoRA. However, the computational overhead of a full FT is significant. To that end, when prioritizing performance and the resulting task does not involve self-speculation, we suggest full-FT, while LoRA is preferable when we are interested in a lightweight modification.

## J.4 Additional Draft Compressibility Training Results

We present additional results on draft compressibility, demonstrating its effect in additional settings.

Additional evaluations on long context Table 22 extends the Qwen3.5-4B draft (Qwen3.5- 122B target) training result to long contexts by evaluating on SPEED-Bench 100k ISL split. It can be seen that the gains are also evident in longer context drafting, with higher gains in non-greedy

decoding settings.

Effect of compressibility training to SnapKV. We demonstrate the effect of the proposed draft compressibility training on the baseline SnapKV method, considering Qwen3.5-4B drafting for Qwen3.5-122B. As seen in Table 23, SnapKV benefits dramatically from the proposed compressibility training scheme. However, we attribute the magnitude of that gain to the initial method’s AL, which was significantly lower than RIFT-stat (As reported in the main paper).

Additional draft training We present the effect of draft compressibility training on additional draft models. Table 24 presents the effect on Qwen3.5- 0.8B (drafting for Qwen3.5-122B). Gains are modest at $T ~ = ~ 0$ but reach up to 10% at $T = 1$ The corresponding untrained policy comparison appears in Table 8. Next, we consider a larger draft model. Specifically, we train Nemotron3.5- Lightning with a Nemotron-3-Ultra teacher. The training recipe follows the one described in Section 5, with a single slight change: Following a empirical sweep, we replace the KL loss term with a skewed KL loss, which replaces the teacher’s logits with a linear interpolation of the student and teacher’s logits. The underlying potential reason for the performance improvement of skewed KL is that skewed KL compensates for large model distribution discrepancies, which may follow due to the large parameter count ratio between the draft and the target. As shown in Table 25, compressibility training improves Lightning’s AL by up to approximately 16.7% (3.72 to 4.34 for RIFT-dyn at 2K and $T = 1 )$ . At 8K and $T = 0$ , its trained RIFT-dyn AL of 5.27 corresponds to 97.8% of the original draft’s full-KV AL (5.39), or 91.7% of the trained draft’s full-KV AL (5.75).

## J.5 Additional RIFT-ada Results on Draft Compressibility

We provide additional results on the effect on draft compressibility training on RIFT-ada. We consider Qwen3.5-4B and Qwen3.5-0.8B, drafting for Qwen3.5-122B. As seen in Table 26 training improves AL in every reported setting.

## K Extended Related Work

We review the work most directly connected to frozen prefill selection, decode-time cache management, and compressed-cache drafting.

Prefill-based KV cache compression. A major line of work focuses on a single compression of the prompt KV cache, right before decode takes place. SnapKV estimates prefill-token importance from attention scores produced by an observation window, often at the end of prefill (Li et al., 2024). PyramidKV and AdaKV build SnapKV’s selec tion paradigm while heterogeneously allocating the cache budget across layers and heads, respectively (Cai et al., 2024; Feng et al., 2025b). Other static KV cache selectors change the importance signal. For example, CriticalKV scores the output perturbation caused by eviction (Feng et al., 2026), Expected Attention models attention under a distribution of future queries (Devoto et al., 2025) and CurDKV uses query-free key and value-leverage scores (Sengupta et al., 2025). DefensiveKV instead challenges the usual mean aggregation of importance over an observation window and introduces a worst-case-risk-aware aggregation rule. nevertheless, its eviction is still performed during prefill, leaving the retained prompt set fixed during decode (Feng et al., 2025a). ChunkKV changes the selection unit from individual tokens to contiguous chunks (Liu et al., 2025). Collectively, these methods primarily optimize which entries survive compression. Our work studies a common failure mode: even a strong prefill decision can become stale as a long generated prefix changes what the model needs, and we do that through the lenses of speculative decoding. In the context of compressibility training, KV-CAT (Gelberg et al., 2026) demonstrates that training under KV sparsification can improve the post-hoc compressibility of pretrained language models, enabling additional gains through training. They consider an additional, trainable masker module, based on which sparse attention masks are built. In this work we directly consider specific KV scoring methods.

Decode-time KV compression for long generation. Decode-time strategies continually adapt the working cache. For example, ScissorHands exploits persistence of token importance, StreamingLLM retains attention sinks and recent tokens, and $_ \mathrm { H _ { 2 } O }$ allocates half of its cache budget to recent tokens and half to heavy-hitters identified by accumulated attention scores (Liu et al., 2023; Xiao et al., 2024; Zhang et al., 2023). R-KV utilizes reasoning trace information sparsity (Cai et al., 2025). QUEST proposes lightweight pagelevel retrieval through aggregated metrics (Tang et al., 2024). Its page summaries are complementary to our study of when and from which temporal region a drafter should refresh its active cache. LongFlow derives a low-overhead importance signal from the current query for continuous eviction, whereas KARA restricts decode-time scoring to a recent window and expands selected tokens into flexible chunks (Su et al., 2026; Han et al., 2026). Compression-aware speculative decoding. Tri-Force uses a dynamically retrieved target cache as an intermediate drafter, refreshed periodically or by acceptance gating (Sun et al., 2024). MagicDec studies sparse-KV self-speculation in large concurrencies, while SpecExtend and SparseSpec reuse verification attention to select KVs (Sadhukhan et al., 2025; Cha et al., 2025; Zhao et al., 2025). LongSpec trains a constant-context drafter and Windowed-MTP and Attention Drift study SWAbased MTP drafting and stepwise attention changes in tail-augmented drafting (Yang et al., 2026; Valliappan, 2026; Eldenk et al., 2026). RIFT complements these approaches by operating directly on external or self-speculative draft caches and studying recency allocation and draft-side refresh signals under a fixed active KV cache budget.

<table><tr><td rowspan="2">Training configuration</td><td colspan="3">T = 0</td><td colspan="3"> $T = 1$ </td></tr><tr><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td>Random, dynamic p</td><td>4.67</td><td>5.15</td><td>5.70</td><td>4.32</td><td>4.71</td><td>5.34</td></tr><tr><td>Random, p = 0.1</td><td>4.31</td><td>4.76</td><td>5.20</td><td>3.99</td><td>4.37</td><td>4.87</td></tr><tr><td>Random, p = 0.3</td><td>4.56</td><td>5.04</td><td>5.56</td><td>4.22</td><td>4.62</td><td>5.21</td></tr><tr><td>Random, p = 0.5</td><td>4.59</td><td>5.09</td><td>5.67</td><td>4.24</td><td>4.66</td><td>5.29</td></tr><tr><td> $\mathrm { R a n d o m } , p = 0 . 7$ </td><td>4.63</td><td>5.12</td><td>5.70</td><td>4.25</td><td>4.66</td><td>5.31</td></tr><tr><td> $\ R a \mathrm { n d o m } , p = 0 . 9$ </td><td>4.62</td><td>5.12</td><td>5.69</td><td>4.23</td><td>4.64</td><td>5.29</td></tr><tr><td> $\mathrm { S W A } , B \stackrel { \cdot } { = } 6 4$ </td><td>4.22</td><td>4.47</td><td>4.74</td><td>3.94</td><td>4.14</td><td>4.47</td></tr><tr><td> $\mathrm { S W A } , B = 1 2 8$ </td><td>4.36</td><td>4.63</td><td>4.93</td><td>4.07</td><td>4.29</td><td>4.65</td></tr><tr><td> $\mathrm { S W A } , B = 2 5 6$ </td><td>4.49</td><td>4.78</td><td>5.10</td><td>4.18</td><td>4.43</td><td>4.81</td></tr><tr><td> $\mathrm { S W A } , B = 5 1 2$ </td><td>4.64</td><td>4.97</td><td>5.36</td><td>4.32</td><td>4.60</td><td>5.04</td></tr><tr><td>Attention-based mask</td><td>4.89</td><td>5.37</td><td>5.94</td><td>4.58</td><td>4.98</td><td>5.60</td></tr></table>

Table 19: AL of the trained draft model (Qwen3.5-4B) under various masking strategies. ’dynamic $p '$ refers to the same $p$ allocation used in the main draft-compressibility results.
<table><tr><td colspan="4">RIFT-stat</td><td colspan="3">RIFT-dyn</td></tr><tr><td>Training configuration</td><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td colspan="7">(a) Sliding-window ratio ρ during training</td></tr><tr><td> $\rho = 0 . 1 0$ </td><td>4.67</td><td>5.19</td><td>5.80</td><td>5.36</td><td>5.92</td><td>6.28</td></tr><tr><td> $\rho = 0 . 2 5$ </td><td>4.69</td><td>5.21</td><td>5.82</td><td>5.39</td><td>5.92</td><td>6.29</td></tr><tr><td> $\rho = 0 . 5 0$ </td><td>4.72</td><td>5.22</td><td>5.81</td><td>5.40</td><td>5.93</td><td>6.29</td></tr><tr><td> $\rho = 0 . 7 5$ </td><td>4.72</td><td>5.21</td><td>5.79</td><td>5.38</td><td>5.93</td><td>6.30</td></tr><tr><td> $\rho = 0 . 9 0$ </td><td>4.72</td><td>5.20</td><td>5.78</td><td>5.35</td><td>5.88</td><td>6.28</td></tr><tr><td colspan="7">(b) Sampling distribution over pkeep</td></tr><tr><td>Soft high-biased</td><td>4.73</td><td>5.21</td><td>5.81</td><td>5.41</td><td>5.92</td><td>6.28</td></tr><tr><td>High-biased</td><td>4.72</td><td>5.22</td><td>5.81</td><td>5.40</td><td>5.93</td><td>6.29</td></tr><tr><td>Sharp high-biased</td><td>4.70</td><td>5.19</td><td>5.80</td><td>5.38</td><td>5.93</td><td>6.32</td></tr></table>

Table 20: Training-mask hyperparameter ablations for Qwen3.5-4B drafting Qwen3.5-122B on SPEED-Bench 32K low-entropy. All checkpoints are evaluated with $\rho = 0 . 5$ with T = 0.
<table><tr><td></td><td colspan="2"> $T = 0$ </td><td colspan="2"> $T = 1$ </td></tr><tr><td>Training configuration</td><td>RIFT-stat</td><td>RIFT-dyn</td><td>RIFT-stat</td><td>RIFT-dyn</td></tr><tr><td>LoRA-32, constant  $1 0 ^ { - 5 }$ </td><td> ${ \bf 5 . 3 0 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 8 5 \pm 0 . 1 3 }$ </td><td> $4 . 9 3 \pm 0 . 1 1$ </td><td> ${ \bf 5 . 4 6 \pm 0 . 1 2 }$ </td></tr><tr><td>LoRA-128, decay from 10−5</td><td> $5 . 2 9 \pm 0 . 1 3$ </td><td> ${ \bf 5 . 8 5 \pm 0 . 1 3 }$ </td><td> $4 . 9 3 \pm 0 . 1 2$ </td><td> $5 . 4 5 \pm 0 . 1 3$ </td></tr><tr><td>Full FT, constant  $5 \times 1 0 ^ { - 6 }$ </td><td> $5 . 2 9 \pm 0 . 1 3$ </td><td> $5 . 8 2 \pm 0 . 1 3$ </td><td> $4 . 9 1 \pm 0 . 1 2$ </td><td> $5 . 4 3 \pm 0 . 1 3$ </td></tr><tr><td>Full FT, decay from  $1 0 ^ { - 5 }$ </td><td> ${ \bf 5 . 3 0 \pm 0 . 1 3 }$ </td><td> ${ \bf 5 . 8 5 \pm 0 . 1 3 }$ </td><td> ${ \bf 4 . 9 4 \pm 0 . 1 2 }$ </td><td> ${ \bf 5 . 4 6 \pm 0 . 1 3 }$ </td></tr></table>

Table 21: Training hyperparameter comparison. Results are reported in average ALs with 95% confidence intervals.
<table><tr><td></td><td colspan="3">RIFT-stat</td><td colspan="3">RIFT-dyn</td><td colspan="3">RIFT-ada</td></tr><tr><td>T</td><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td>0</td><td>+4.05%</td><td>+2.24%</td><td>+0.97%</td><td>+4.90%</td><td>+2.54%</td><td>+0.97%</td><td>+2.57%</td><td>+0.37%</td><td>-0.34%</td></tr><tr><td></td><td>+12.50%</td><td>+9.87%</td><td> $+ 7 . 0 3 \%$ </td><td>+12.04%</td><td>+7.95%</td><td>+5.12%</td><td>+9.80%</td><td>+5.49%</td><td>+3.79%</td></tr></table>

Table 22: Relative AL improvement from compressibility training for Qwen3.5-4B drafting Qwen3.5-122B on SPEED-Bench 100K.
<table><tr><td>T</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td>0</td><td>+15.57%</td><td>+13.50%</td><td>+11.52%</td></tr><tr><td>1</td><td>+26.54%</td><td>+23.80%</td><td>+21.07%</td></tr></table>

Table 23: Relative AL improvement from draft compressibility training for SnapKV: Qwen3.5-4B draft (Qwen3.5- 122B target), 100K low-entropy split.

<table><tr><td>ISL</td><td>Entropy</td><td> $T$ </td><td>Full KV</td><td>SW</td><td>RIFT-stat</td><td>RIFT-dyn</td></tr><tr><td rowspan="4">32K</td><td>Low</td><td>0</td><td> $+ 0 . 5 5 \%$ </td><td> $+ 0 . 9 0 \%$ </td><td> $+ 0 . 8 4 \%$ </td><td> $+ 0 . 8 1 \%$ </td></tr><tr><td>Low</td><td>1</td><td> $+ 7 . 7 7 \%$ </td><td> $+ 6 . 4 3 \%$ </td><td> $+ 9 . 7 1 \%$ </td><td> $+ 8 . 8 7 \%$ </td></tr><tr><td>High</td><td>0</td><td> $+ 0 . 8 2 \%$ </td><td> $+ 0 . 8 0 \%$ </td><td> $+ 0 . 9 8 \%$ </td><td> $+ 0 . 8 4 \%$ </td></tr><tr><td>High</td><td>1</td><td>+8.95% +7.30%</td><td></td><td> $+ 9 . 3 3 \%$ </td><td> $+ 9 . 3 2 \%$ </td></tr><tr><td rowspan="3">100K</td><td>Low</td><td>0</td><td>+0.00%</td><td> $+ 1 . 7 3 \%$ </td><td> $+ 0 . 9 0 \%$ </td><td> $+ 0 . 7 7 \%$ </td></tr><tr><td>Low</td><td>1</td><td> $+ 5 . 9 0 \%$ </td><td> $+ 5 . 7 0 \%$ </td><td> $+ 8 . 2 4 \%$ </td><td> $+ 7 . 6 8 \%$ </td></tr><tr><td>High</td><td>0</td><td> $+ 0 . 6 5 \%$ </td><td> $- 0 . 2 5 \%$ </td><td> $+ 0 . 7 1 \%$ </td><td> $+ 0 . 5 2 \%$ </td></tr></table>

Table 24: Relative AL improvement from training the Qwen3.5-0.8B draft, measured on SPEED-Bench at an 8K cache budget against the corresponding untrained draft.
<table><tr><td colspan="2"></td><td colspan="2"> $T = 0$ </td><td colspan="2"> $T = 1$ </td></tr><tr><td>Inference method</td><td>Budget</td><td>Base</td><td>Trained</td><td>Base</td><td>Trained</td></tr><tr><td>Full KV</td><td>一</td><td>5.39</td><td>5.75</td><td>4.86</td><td>5.49</td></tr><tr><td rowspan="3">Sliding window</td><td>2K</td><td>3.62</td><td>3.87</td><td>3.21</td><td>3.73</td></tr><tr><td>4K</td><td>3.92</td><td>4.19</td><td>3.49</td><td>4.02</td></tr><tr><td>8K</td><td>4.31</td><td>4.60</td><td>3.86</td><td>4.43</td></tr><tr><td rowspan="3">RIFT-stat</td><td>2K</td><td>3.75</td><td>4.04</td><td>3.39</td><td>3.92</td></tr><tr><td>4K</td><td>3.99</td><td>4.29</td><td>3.60</td><td>4.16</td></tr><tr><td>8K</td><td>4.35</td><td>4.68</td><td>3.93</td><td>4.52</td></tr><tr><td rowspan="3">RIFT-dyn</td><td>2K</td><td>4.16</td><td>4.49</td><td>3.72</td><td>4.34</td></tr><tr><td>4K</td><td>4.57</td><td>4.93</td><td>4.08</td><td>4.72</td></tr><tr><td>8K</td><td>4.99</td><td>5.27</td><td>4.45</td><td>5.04</td></tr></table>

Table 25: Average accepted length before and after compressibility training of Nemotron3.5-Lightning drafting Nemotron-3-Ultra on SPEED-Bench 32K low-entropy. The trained checkpoint uses skewed KL with $\alpha = 0 . 5$ Static and Dynamic SnapKV use a sliding-window ratio of $\rho = 0 . 5$
<table><tr><td>Draft → target</td><td></td><td>ISL Entropy</td><td>T</td><td>2K</td><td>4K</td><td>8K</td></tr><tr><td rowspan="2">Qwen3.5-4B → Qwen3.5-122B</td><td rowspan="2">32K Low</td><td rowspan="2"></td><td>0</td><td> $+ 3 . 8 8 \%$ </td><td> $+ 2 . 3 8 \%$ </td><td> $+ 1 . 6 9 \%$ </td></tr><tr><td>1</td><td> $+ 1 0 . 0 3 \%$ </td><td> $+ 7 . 8 8 \%$ </td><td> $+ 6 . 7 0 \%$ </td></tr></table>

Table 26: Relative AL improvement from compressibility training on RIFT-ada.