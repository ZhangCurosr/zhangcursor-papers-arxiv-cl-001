# The Gating Mechanism of KV Cache Eviction

Hao Gu<sup>1,\*</sup>, Xintong Yang<sup>2,\*</sup>, Zhiwei Li<sup>3</sup>, Lei Zhu<sup>4</sup>, Jian Yang<sup>1</sup>, Qiyuan Zhu<sup>1</sup>, Dezhi Li<sup>1</sup>, Jingyuan Zhang<sup>5</sup>, Zhizhuo KOU<sup>1</sup>, Hao Wang<sup>6</sup>, Bei Liu<sup>1</sup>, Jiacheng Liu<sup>7</sup> and Sirui Han<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology, <sup>2</sup>New York University, <sup>3</sup>The Hong Kong University of Science and Technology (Guangzhou), <sup>4</sup>Tencent HunYuan, <sup>5</sup>Georgia Institute of Technology, <sup>6</sup>City University of Hong Kong, <sup>7</sup>Shanghai Jiaotong University

Code: IndexMem++ Checkpoints: Hugging Face

Abstract: KV cache eviction must decide which entries carry information that the rest of the cache can no longer supply. A natural way to estimate this is a learnable gate: give each cached entry a gate on its attention weight and train the gate on the model’s own objective. The gate belongs inside the normalizing softmax, where suppressing one entry hands its mass to the others rather than discarding it. That same normalizer absorbs any per-query constant, so a constant gate cancels and reproduces the frozen model exactly. Trained on the language-modeling loss the gate stays near a constant score while its loss falls throughout: optimization succeeds, but a frozen backbone was fit for dense attention, so its loss prefers the no-op, and a near-constant score ranks nothing. The no-op is not a floor the gate improves from but what its objective asks for. A second obstacle survives once the gate moves, since the same gradient charges the gate for errors the frozen model already makes, which no reweighting of positions separates out. Much of the machinery in learnable eviction, from sparsity penalties to relaxations of the discrete decision, works around the first obstacle rather than removing it; both are removed by putting dense attention beyond the gate’s reach and making the dense model’s predictions its target. Under these two conditions capacity is never the missing ingredient: neither a larger scorer, nor explicit access to the past, nor a trainable backbone helps. The same holds for what eviction unavoidably discards, which is recovered by parameter-free centroids read inside the same softmax far better than by any memory read outside it, since the mass they must return is a share of that normalizer and available nowhere else. At 75% KV cache compression, this gate reaches 90.14 on RULER at 8K and 46.03 on LongBench with Qwen3-8B, 3.72 and 1.36 points below full-cache inference.

## 1. Introduction

Large language models are evolving from chatbot assistants (Touvron et al., 2023) into long-horizon agents (Yao et al., 2023). These agentic workloads naturally induce long interaction histories, making eficient long-context inference essential. During autoregressive decoding, the KV cache is increasingly the main bottleneck. As the cache grows linearly with sequence length, KV-cache storage and data movement become the dominant systems bottlenecks (Kwon et al., 2023). Reducing this footprint has therefore become a central objective in eficient LLM inference. Existing approaches compress KV cache along several axes: quantization reduces precision (Liu et al., 2024), cross-layer sharing reduces number of layer-specific KV (Sun et al., 2024b), and latent representations reduce KV dimension (DeepSeek-AI, 2024). These approaches are complementary and can be combined. In this work, we focus on KV eviction because it not only reduces memory consumption but also shortens the efective sequence length of attention, thereby reducing attention computation. KV eviction is possible because information becomes redundant across the cache. Through causal self-attention, each hidden state aggregates preceding context, allowing later KV entries to carry information originally introduced by earlier tokens. This layer-wise propagation explains why sliding-window attention can access information beyond its explicit window (Fu et al., 2025).

However, this propagation is incomplete: isolated facts such as passwords or retrieval keys may remain unique to their original entries. Efective eviction must therefore distinguish two kinds of entries: those whose information can be recovered from the rest of the cache, and those that carry the only copy of what they contribute. Sparse attention and KV eviction appear similar because both select a subset of KV entries, but their objectives difer. Sparse attention selects entries relevant to the current query; eviction must identify entries that are redundant regardless of future queries. Existing eviction methods however often treat relevance as redundancy. $\mathrm { H _ { 2 } O }$ and SnapKV use attention from previously observed queries (Zhang et al., 2023; Li et al.,

2024). Relevance can serve as a partial proxy for redundancy, but does not measure it directly.

We instead define redundancy relative to the retained cache: an entry is redundant if its information can be recovered from the other KV entries that remain. Hand-crafted measures such as cosine similarity (Park et al., 2026; Cai et al., 2025) capture only limited aspects of this relationship. We therefore learn a lightweight, query-independent gate that maps each contextualized hidden state to a retention score for each KV head. And where that gate is attached is not a free choice. Attention weights are shares of one softmax normalizer, so only a gate inside that normalizer hands a suppressed entry’s mass to the remaining entries, which is what removal means.

Naively training this gate does not work: it learns nearly identical retention scores for all tokens, reproducing dense attention without learning which entries can be evicted. The reason is structural and it comes from the same normalizer that gives the gate its meaning. Because a softmax absorbs any per-query constant, a constant score cancels inside it and is a no-op, and a frozen backbone’s loss prefers exactly that. The gate must therefore be denied dense attention and asked to match the dense model’s predictions instead of the next token. The same invariance also bounds what a score can say: it ranks entries within a head but cannot compare heads, so the division of budget among them must be settled outside the score.

Even accurate retention scores cannot make hard eviction lossless: retained entries may preserve part of an evicted entry’s information, but not all of it. We find that discarded entries still receive substantial attention mass, motivating a compact mechanism for recovering their residual contribution. Our key finding is that the readout of this compressed information matters more than the capacity of the memory that stores it, and the reason is the normalizer again. Recovering the missing contribution requires both its direction and the fraction of attention mass it holds, and that fraction is a ratio of softmax denominators: it is available exactly where the gate was, and only estimable indirectly anywhere else. A memory read outside attention must learn that calibration in output space and does not improve as it grows, whereas parameter-free centroids of the evicted keys and values, inserted into the cache and weighted by the same softmax as the retained entries, recover it far more closely. Removal and restoration are thus the same statement about where the normalizer applies. We instantiate these findings in IndexMem++, which follows a two-branch design (Yang et al., 2026), summarized in Figure 1: a selection branch evicts redun-

![](images/30495ef55a5d4ba6dfb81d6622e9291134885ece58012a64c265ec724036bee3.jpg)  
Figure 1: The two branches of IndexMem++: evict and recovery both inside one softmax.

dant KV entries, and a recovery branch summarizes what was evicted, both reading through one softmax. Experiments on Qwen3-4B and Qwen3-8B show consistent gains across long-context understanding and mathematical reasoning. At 75% compression, IndexMem++ reaches 90.14 on RULER with Qwen3-8B and 46.03 on LongBench, within 1.36 points of Full-KV, and it retains 0.807 pass@1 on Math500 with a 512-entry cache. Section 4 reports the full comparison.

## Contributions. Our contributions are threefold:

• We distinguish redundancy from query relevance, formulate KV eviction as query-independent redundancy estimation, and unify learnable gate methods under a common design space.

• We show that the same normalizer is why naive gate training fails: it absorbs a per-query constant, so dense attention is always reachable and is what a frozen backbone’s loss prefers. Learning a usable score requires denying the gate that solution, by exempting the entries that must be retained anyway and making the rest compete for a fixed mass, and supervising it against the dense model rather than the next token. The one thing the score cannot express, the division of budget across heads, follows from the same invariance and is set by retained attention mass.

• We show that readout matters more than memory capacity for recovering evicted information, because the missing mass fraction is a ratio of softmax denominators, and introduce content-clustered centroid KV tokens that re-enter the attention softmax at a matched cache budget.

## 2. Attention mass and the eviction gate

Attention and KV eviction. Writing $h _ { t }$ for the hidden state of token � and � for the head dimension, selfattention projects $h _ { t }$ into a query, key, and value, and token � attends to all preceding tokens:

$$
o _ { t } = \sum _ { j \leq t } a _ { t j } v _ { j } , \qquad a _ { t j } = \frac { \exp ( \ell _ { t j } ) } { Z _ { t } } , \qquad \ell _ { t j } = \frac { q _ { t } k _ { j } ^ { \top } } { \sqrt { d } } , \qquad Z _ { t } = \sum _ { \ell \leq t } \exp ( \ell _ { t \ell } ) .\tag{1}
$$

The weights $a _ { t j }$ are shares of the single normalizer $Z _ { t }$ and sum to one; we call $a _ { t j }$ the attention mass that query � assigns to entry $j ,$ and it is the quantity the rest of this paper is about. The keys and values are cached and reused at every decoding step, so the cache grows linearly with context length. Under a budget of � entries, an eviction policy scores each entry and keeps the current top-�, selecting at every step among the surviving entries and the new one; everything outside the top-� is physically removed and never reconsidered.

From hard eviction to a diferentiable gate. A learned gate or router can score entries for sparse top-� selection, but learning which entries to select requires a training signal through a discrete operation: the selected indices are piecewise constant in the scores. Sparse mixture-of-experts models preserve a gradient path by multiplying selected expert outputs by their continuous routing weights (Shazeer et al., 2017). DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025) takes another route: it trains its indexer to match the main model’s attention distribution, initially under dense attention, rather than diferentiating through the top-� choice.

KV eviction, however, lacks a suitable eviction oracle. An attention weight measures how much one query reads an entry; eviction must decide whether permanently removing it will harm later predictions given the information that remains. KVzap (Jegou and Jeblick, 2026) and Fast KVzip (Kim et al., 2026) learn KVzip-style reconstruction scores from hidden states, avoiding backpropagation through the eviction decision. These targets measure importance during context reconstruction, not the prediction error caused by removing an entry from a budgeted cache. Even exact prediction of the proxy would therefore leave the mismatch with the eviction objective unresolved.

We instead train retention scores through the model’s predictions, replacing the binary keep-or-drop decision during training with a continuous per-entry gate $g _ { j } \geq 0$ . A diferentiable path is obtained by inserting the gate’s output into attention’s computation graph. This admits many choices: gates on queries, keys, values, or outputs; additive logit biases; temperature or denominator gates; and even nested softmax constructions. Figure 2 illustrates representative placements. Diferentiability alone does not determine which placement approximates eviction, because removing an entry also removes its key from the normalizer and redistributes its attention mass. We therefore focus on the distinction between a gate outside the softmax, which scales a value contribution, and a gate inside the softmax, which renormalizes attention over the gated entries.

After the softmax: suppressing a value. An outer gate multiplies each already-normalized attention weight, equivalently scaling the corresponding value:

$$
o _ { i } ^ { \mathrm { o u t e r } } = \sum _ { j } a _ { i j } g _ { j } v _ { j } .
$$

Setting $g _ { j } = 0$ removes its value contribution, but its key still contributes to the original normalizer $Z _ { i } .$ . The other weights stay unchanged, and the efective weights $a _ { i j } g _ { j }$ need not sum to one. This is attenuation of the retrieved message; it does not reproduce attention over the surviving cache.

Inside the softmax: removing a cache entry. We instead renormalize after applying the gates:

$$
a _ { i j } ^ { ( g ) } = \frac { a _ { i j } g _ { j } } { \sum _ { m } a _ { i m } g _ { m } } , \qquad o _ { i } = \sum _ { j = 1 } ^ { i } a _ { i j } ^ { ( g ) } v _ { j } .\tag{2}
$$

The denominator transfers the mass withdrawn from one entry to the others. For a binary gate with at least one retained entry, Equation 2 is exactly attention over the retained set; positive continuous gates provide its diferentiable relaxation. Since $\exp ( \ell _ { i j } ) g _ { j } = \exp ( \ell _ { i j } + \log g _ { j } )$ , this is an additive bias on each key’s attention logit, with $g _ { j } \to 0$ recovering the hard mask log $g _ { j }  - \infty$ . The gate is independent of future query content, while its efect is measured relative to the other entries through the shared normalizer.

Gating in attention  
![](images/cf3f8f6bd0f8d4f4f8160f53e98d4ae1d2c173582242296b9a5cc7d103cb1c49.jpg)  
All gate branches remain in the computation graph.  
Figure 2: Diferentiable gates can act at diferent points in attention. (a) Gates on queries, keys, values, or outputs; (b) an additive logit bias; (c) a gate on the value branch; (d) a multiplicative logit scale; and (e) a modification of the denominator alone. All keep the gate in the computation graph, but they have diferent efects on attention mass. A value gate suppresses a contribution without renormalizing the other weights, whereas a per-key logit bias log �<sub>�</sub> redistributes mass and recovers attention over the retained set in the binary limit.

Other placements have diferent meanings. Query and key gates change the retrieval logits, whereas value and output gates scale the retrieved payload. Reducing a multiplicative logit scale pulls its logit towards zero, whose efect depends on the logit’s sign under the current query; adding mass only to the denominator suppresses the row without providing a per-entry retention ranking. Appendix A.3 gives the detailed comparison. Our construction specifically uses a per-key additive logit bias to relax cache removal.

The paradigm. The direct instantiation of this framework is

$$
\mathbf { a } _ { i } = \mathrm { s o f t m a x } \left( \frac { q _ { i } K _ { \leq i } ^ { \top } } { \sqrt { d } } + \log \mathbf { g } _ { \leq i } \right) , \qquad s _ { j } = f _ { \theta } ( h _ { j } ) , \qquad \log g _ { j } = s _ { j } ,\tag{3}
$$

where $f _ { \theta }$ maps the contextualized hidden state $h _ { j }$ to a per-KV-head retention score. During training, the backbone is frozen and gradients through gated attention update �. During inference, the scores select which KV entries remain; attention over the retained entries does not apply the training-time soft gates. A uniform positive gate still cancels inside the softmax and recovers dense attention. Section 3 explains why this makes direct training fail, how to train a useful ranking, and how the recovery branch of Figure 1 restores part of the discarded contribution through the same normalizer.

## 3. Method

Equation 3 fixes the mechanism but not the design. We begin from its most direct instantiation and find that it does not work at all, for a reason that is structural rather than a matter of tuning. The shared normalizer that gives the gate its meaning has a second consequence: it absorbs any per-query constant, so dense attention is always within the gate’s reach. This section follows that consequence through, to what has to be taken away from the gate, what has to be given to it, and the one question a retention score is unable to answer. A pilot ablation on Qwen3-0.6B isolates each step.

Pilot setup. All ablations use Qwen3-0.6B on RULER at 8K with a cache budget of 2048 entries, 25% of the context. Every arm trains for 100 steps and difers from the reference in a single field. Appendix F gives the full protocol, the per-task scores.

## 3.1. Learning a Reliable Eviction Scorer

The most direct instantiation does not work. We first consider the most direct design of Equation 3: an MLP maps each hidden state $h _ { j }$ to a score $s _ { j }$ and is trained with the language-modeling loss. This setup collapses to nearly constant scores, reaching 4.33 where the dense model reaches 70.01. If all $g _ { j }$ take the same positive value, log $g _ { j }$ adds the same constant to every key logit, leaving the attention weights unchanged. The constant score is thus a no-op, and the objective prefers it: the frozen backbone was fit for dense attention, so its minimizer is a small perturbation of the constant score. Nearly constant scores are useless for eviction: the top-� cut falls among near-ties, so the retained set is arbitrary (Appendix E).

Removing the no-op solution. The scores should not be able to reproduce dense attention. Eviction already keeps a few attention sinks and a causal local window under any policy, and these entries give the gate a reference frame: holding their gates at one fixes the scale, so the level of the remaining gates is no longer free. We then require the evictable entries to share a fixed total mass $\beta ,$ which they cannot do at the value that would recover dense attention. Writing $P _ { i }$ for the protected set and $E _ { i }$ for the evictable one,

$$
g _ { i j } = \left\{ \begin{array} { l l } { 1 , } & { j \in P _ { i } , } \\ { \displaystyle \beta \frac { \exp ( s _ { j } ) } { \sum _ { \ell \in E _ { i } } \exp ( s _ { \ell } ) } , } & { j \in E _ { i } , } \end{array} \right. \qquad a _ { i j } \propto \exp \left( \frac { q _ { i } k _ { j } ^ { \top } } { \sqrt { d } } \right) g _ { i j } .\tag{4}
$$

Dense attention is now out of the question: it needs $g _ { i j } = 1$ throughout $E _ { i } ,$ hence $\beta = | E _ { i } | .$ , whereas we set $\beta \ll | E _ { i } |$ . The scorer has to rank them (Appendix E). Because $\textstyle \sum _ { j \in E _ { i } } g _ { i j } = \beta$ holds exactly at every step, $\beta$ is also the sparsity used during training rather than a target approached through a penalty, which is what makes a sparsity regularizer or schedule unnecessary (Bui et al., 2025; Łańcucki et al., 2025). This mechanism raises pilot RULER from 4.33 to 44.86 (Table 1).

Distillation isolates the error caused by eviction. With the language-modeling loss, the design above reaches only 44.86 because the cross-entropy gradient mixes errors caused by eviction with errors already made by the frozen backbone. Let $p _ { t } ^ { \mathrm { g a t e d } }$ and $p _ { t } ^ { \mathrm { d e n s e } }$ denote the gated and dense next-token probability distributions, and let $e _ { y _ { t } }$ be the one-hot target. The gradient with respect to the output logits decomposes as

$$
p _ { t } ^ { \mathrm { g a t e d } } - e _ { y _ { t } } = \underbrace { \left( p _ { t } ^ { \mathrm { g a t e d } } - p _ { t } ^ { \mathrm { d e n s e } } \right) } _ { \mathrm { c a u s e d b y e v i c t i o n } } + \underbrace { \left( p _ { t } ^ { \mathrm { d e n s e } } - e _ { y _ { t } } \right) } _ { \mathrm { f r o z e n - b a c k b o n e \ e r r o r } } .\tag{5}
$$

The gate can reduce only the first term, while the second is irreducible with a frozen backbone and can dominate at dificult, high-entropy positions. Token reweighting cannot remove this interference. LongCE (Fang et al., 2024) usefully emphasizes tokens that benefit from long context, but still scales both terms in Equation 5 together.

Distillation removes the second term by supervising the gate against the dense distribution. The gradient then depends only on $p _ { t } ^ { \mathrm { g a t e d } }$ and $p _ { t } ^ { \mathrm { d e n s e } }$ , and vanishes once the two agree, so the frozen-backbone error never enters. Either direction of the KL has this property: the forward direction gives exactly the first term of Equation 5, while the reverse direction we adopt matches it to first order and difers only in how it treats the teacher’s low-probability region. We choose between them empirically (Appendix E). Supervision also becomes dense, since each position contributes a full distribution over the vocabulary instead of a single target token. This is the +14.00 from 44.86 to 58.86 in Table 1.

Scores are not comparable across heads. Attention heads specialize in diferent local and long-range patterns, so they need diferent cache capacities, and a fixed ofline budget table already improves over a uniform split, confirming that reallocating capacity is beneficial. But the retention score cannot decide that allocation. Adding a constant to every score of one head shifts its normalizer by the same amount, leaving both Equation 4 and the entries retained within that head unchanged, so the absolute level of a head’s scores is not identifiable and scores from diferent heads are not comparable. The allocation must therefore be read of a quantity that this shift does not move, and the attention mass a head retains is such a quantity. It also depends on the current context, which the ofline table cannot follow. We therefore allocate the budget online from the final prefill query. For each head, we order the evictable entries by the retention score and compute the cumulative attention mass retained by its top � entries, including the always-attended sinks and window. We then give every head the smallest $k _ { h }$ that reaches a shared mass target $\rho ^ { \star } { } _ { ; }$ , adjusting $\rho ^ { \star }$ and any integer remainder so that $\textstyle \sum _ { h }$ � exactly equals the uniform baseline’s total budget. This assigns more slots to heads whose attention is distributed over more entries and raises the average from 58.86 to 67.60. Permuting the same set of per-head budgets reduces performance to 57.14, below the uniform split, confirming that the gain comes from matching each budget to the head and context that need it, rather than from using unequal budgets alone (Table 2). This allocator can be viewed as budget-constrained per-head top-�, with the shared mass threshold chosen to satisfy the global cache budget.

Table 1: Each component of the paradigm, added one at a time. Qwen3-0.6B, RULER 8K, cache budget 2048 of 8192 entries (75% compress ratio). Every arm trains for 100 steps and difers from the row above in the single field marked in bold.
<table><tr><td></td><td colspan="3">Component</td><td colspan="2">RULER 8K</td></tr><tr><td>Configuration</td><td>Always-attended</td><td>Loss</td><td>Head budget allocation</td><td> $\operatorname { A v } { \mathbf { g } } .$ </td><td>NIAH</td></tr><tr><td>Direct instantiation</td><td>none</td><td>LM</td><td>uniform</td><td>4.33</td><td>2.00</td></tr><tr><td>+ always-attended entries</td><td>sinks + window</td><td>LM</td><td>uniform</td><td> $4 4 . 8 6 _ { ( + 4 0 . 5 3 ) }$ </td><td>53.73</td></tr><tr><td>+ distillation</td><td>sinks + window</td><td>KL</td><td>uniform</td><td> $5 8 . 8 6 _ { ( + 1 4 . 0 0 ) }$ </td><td>72.78</td></tr><tr><td>+ head budget</td><td>sinks + window</td><td>KL</td><td>mass</td><td> $6 7 . 6 0 \ ( + 8 . 7 4 )$ </td><td>86.17</td></tr><tr><td>Dense backbone (no eviction)</td><td></td><td></td><td></td><td>70.01</td><td>86.95</td></tr></table>

## 3.2. Scorer Architecture and Training Stability

Gate architectures  
![](images/d1f4564baf04b8a4ed30879fded3d4c1b0b629b11f90e19876d3e957de45642a.jpg)  
Figure 3: Scorer architectures. All variants map $x _ { j } = \mathrm { N o r m } ( h _ { j } )$ to per-head retention scores and difer only in what they read besides $x _ { j } .$ (a) A linear readout and (b) the token-local MLP used as the reference scorer see the current hidden state alone. The three history variants add one branch that is summed into the MLP’s pre-activation: (c) a depthwise causal convolution over the previous eight tokens, (d) a gated recurrence carrying a fixed-size state $S _ { j - 1 } ,$ and (e) a single-head causal attention over the prefix up to �. The added branch is initialized to zero, so each variant starts exactly from (b).

Table 2: Which choice, within each component. Every arm is the reference recipe (†) with a single field changed, and Δ is its Avg. against that reference. Two seeds of the reference itself difer by 0.28, so a Δ of that size is not a diference. Setup and per-task scores are in Appendix F.
<table><tr><td>Component</td><td>Choice</td><td> $\operatorname { A v } { \mathbf { g } } .$ </td><td>NIAH</td><td> $\Delta$ </td></tr><tr><td rowspan="4">Always-attended</td><td>none</td><td>4.72</td><td>2.00</td><td>-62.87</td></tr><tr><td>sinks only</td><td>0.57</td><td>0.00</td><td>-67.03</td></tr><tr><td>window only</td><td>57.81</td><td>72.71</td><td>-9.79</td></tr><tr><td>sinks + window†</td><td>67.60</td><td>86.17</td><td>一</td></tr><tr><td rowspan="5">Loss</td><td>LM loss (CE)</td><td>51.50</td><td>64.65</td><td>-16.09</td></tr><tr><td>LongCE</td><td>53.01</td><td>61.99</td><td>-14.59</td></tr><tr><td>forward KL</td><td>66.84</td><td>83.88</td><td>-0.76</td></tr><tr><td>reverse  $\mathrm { K L + 0 . 1 C E }$ </td><td>67.33</td><td>85.00</td><td>-0.27</td></tr><tr><td>reverse KL†</td><td>67.60</td><td>86.17</td><td>一</td></tr><tr><td rowspan="6">Scorer</td><td> $h _ { j }$  only, linear (0.5M)</td><td>53.53</td><td>63.21</td><td>-14.07</td></tr><tr><td> $h _ { j }$  only, MLP (7.5M)†</td><td>67.60</td><td>86.17</td><td></td></tr><tr><td>+ 8 preceding state (16.8M)</td><td>66.82</td><td>85.67</td><td>-0.78</td></tr><tr><td>+ all preceding recurrent state (24.1M)</td><td>68.76</td><td>87.20</td><td>+1.16</td></tr><tr><td>+ all preceding attention (19.46M)</td><td>66.63</td><td>85.48</td><td>-0.97</td></tr><tr><td>fastkvzip-style (11.36M)</td><td>67.25</td><td>84.76</td><td>-0.35</td></tr><tr><td rowspan="4">Head budget allocation</td><td>uniform</td><td>58.86</td><td>72.78</td><td>-8.74</td></tr><tr><td>permuted</td><td>57.14</td><td>68.92</td><td>-10.46</td></tr><tr><td>offline table</td><td>62.33</td><td>75.25</td><td>-5.26</td></tr><tr><td>retained mass†</td><td>67.60</td><td>86.17</td><td></td></tr><tr><td rowspan="2">Retention rate</td><td>off</td><td>69.27</td><td>88.15</td><td>+1.67</td></tr><tr><td> ${ { 0 } \ o \mathrm { { n } } ^ { \dag } }$ </td><td>67.60</td><td>86.17</td><td></td></tr></table>

Table 3: Train the backbone with gate decalibrates the gate rather than damage the model.

<table><tr><td colspan="3"></td><td colspan="2">RULER 8K</td><td rowspan="2">Compression gap</td></tr><tr><td>Loss</td><td>Trained</td><td>Tokens</td><td>Evict 75%</td><td>Full KV</td></tr><tr><td>reverse KL</td><td>gate only†</td><td>7M</td><td>67.60</td><td>70.01</td><td>-2.41</td></tr><tr><td>reverse KL</td><td> $\mathtt { g a t e } + \mathtt { b a c k b o n e }$ </td><td>7M</td><td> $6 2 . 8 3 _ { \ : ( - 4 . 7 7 ) }$ </td><td>71.65</td><td>-8.82</td></tr><tr><td>LM loss</td><td>gate + backbone</td><td>7M</td><td> $6 0 . 4 5 \ _ { ( - 7 . 1 5 ) }$ </td><td>69.07</td><td>-8.62</td></tr><tr><td>LM loss</td><td>gate + backbone</td><td>79M</td><td> $5 9 . 6 8 _ { \ ( - 7 . 9 2 ) }$ </td><td>69.26</td><td>-9.57</td></tr><tr><td>LM loss</td><td>gate + backbone</td><td>197M</td><td> $5 7 . 4 2 _ { \textrm { ( - 1 0 . 1 8 ) } }$ </td><td>69.24</td><td>-11.82</td></tr></table>

The scorer needs capacity, but not a wider view of the past. We next study the scorer architectures in Figure 3. The scorer requires suficient capacity: replacing the 7.5M-parameter MLP with a 0.52M linear readout costs 14.07 points (Table 2). Because identifying redundancy may require comparing an entry with information already stored in the past, we augment the MLP with three forms of explicit history: an eight-token causal convolution, a recurrent state, and full-prefix attention. Despite increasing the scorer to as many as 24.1M parameters, all three variants remain within 1.2 points of the reference. This suggests that the contextualized hidden state $h _ { j }$ already contains the historical information needed to estimate redundancy. A Fast-KVzip-style adaptive reference head (Kim et al., 2026) also provides no benefit. We therefore use a two-layer MLP applied directly to $h _ { j }$

A per-key retention rate stabilizes the retention scores. A score that depends only on content has nothing anchoring its overall level, so scores can drift upward together as training proceeds and the choice of retained entries becomes sensitive to small diferences among large values. Following TrimKV (Bui et al., 2025), we give each entry a learned lifetime alongside its magnitude,

$$
s _ { j } ( i ) = m _ { j } + \log \beta _ { j } \frac { i - j } { \tau } , \qquad \log \beta _ { j } \leq 0 ,\tag{6}
$$

Query independence here refers to independence from future query content; the score may still vary with token age. The policy remains top-� at every step, but because the ranking now drifts with $i ,$ an entry once evicted is never reconsidered even if a retained entry later decays below it (Appendix A.3). Disabling the retention rate reaches 69.27 against 67.60 (Table 2); we keep it for a stable, explicitly parameterized age scale rather than for a gain.

Training the backbone jointly decalibrates the scorer. SP-KV (Szilvasy et al., 2026) jointly trains the language model and utility predictor, allowing sparsity to emerge during training. We do not observe the same benefit: unfreezing the attention projections and norms consistently reduces eviction performance. To distinguish backbone degradation from scorer failure, we also evaluate each checkpoint with Full-KV. Under reverse KL, Full-KV improves from 70.01 to 71.65 (Table 3), while the eviction performance falls from 67.60 to 62.83. The backbone remains intact, but its changed representations decalibrate the scorer. We therefore keep the backbone frozen.

## 3.3. Recovering Evicted Information

Compensating for what eviction discards. Even accurate retention scores cannot make eviction lossless, since the retained entries may preserve part of an evicted entry’s information but not all of it. With $o _ { S }$ and $o _ { E }$ the normalized outputs over the retained and evicted sets and $\rho = Z _ { E } / ( Z _ { S } + Z _ { E } )$ the mass on the evicted ones, dense attention gives $o = ( 1 - \rho ) o _ { S } + \rho o _ { E }$ while eviction returns $^ { O } S \mathrm { , }$ , an error of $\rho ( o _ { E } - o _ { S } )$ . It therefore grows with both how much mass is removed and how diferent the discarded content is from what remains, and we measure $\rho \approx 0 . 2$ , so the missing contribution is not negligible. Recovery needs both quantities, and they are not equally accessible: $\rho$ is a ratio of softmax normalizers, available where the gate was and only indirectly elsewhere. Since $O _ { E }$ pools many entries, a fixed-

![](images/e3a5c477f9b506a6820d0c372487b2f9365908ec29c43128d353b7a109f96d87.jpg)  
Figure 4: Recovery architectures. Linear-state and TTT memories add recovery after attention; centroid KV competes with retained entries within the same softmax.

size memory is the natural way to summarize it. We test a linear associative state $W : k \mapsto v$ and a TTT-style MLP fitted to the evicted pairs at inference, both read afterwards as $\alpha ( q ) o _ { S } + \beta ( q ) m _ { E } ( q )$ (Figure 4). Measured by cosine similarity to the exact $O _ { E _ { \mathrm { i } } }$ , they reach only −0.09 to 0.44, and enlarging them barely helps. The reason is not what they store but where they are read: a memory recovering $o _ { E }$ exactly would still owe $\alpha = 1 - \rho$ and $\beta = \rho ,$ and a fusion placed after the softmax cannot obtain either from $Z _ { S }$ and $Z _ { E ; }$ , so it has to learn the calibration indirectly in output space (Appendix A.4).

The key is therefore to read the summary through attention’s softmax. We cluster the evicted entries, average each cluster into a centroid key and value, and append the � centroids to the retained cache, where $q _ { i } k _ { c } ^ { \top } / \sqrt { d } + \log n _ { r }$ sets a centroid’s weight and $v _ { c }$ its output. The values supply $O _ { E }$ , the logits and cluster-size corrections supply $\rho ,$ and one softmax settles both. This parameter-free construc-

Table 4: Centroid recovery at matched budgets. The total cache budget � includes recovery slots.
<table><tr><td>Model &amp; Benchmark</td><td> $K$ </td><td>No slots</td><td>64 slots</td></tr><tr><td>Qwen3-0.6B Math500</td><td>512</td><td>48.00</td><td> $5 2 . 0 0 _ { ( + 4 . 0 0 ) }$ </td></tr><tr><td>Qwen3-8B RULER 8K</td><td>2048</td><td>76.65</td><td> $7 7 . 1 5 \ : ( + 0 . 4 9 )$ </td></tr></table>

tion reaches cosine similarity of 0.85–0.96 at most layers and improves Math500 and RULER 8K at matched budgets (Table 4). The Math500 gain is the larger one, consistent with compensation preserving information accumulated over a long CoT, whereas NIAH depends on retaining the correct evidence entry, which an average cannot substitute for.

## 4. Experiments

## 4.1. Experimental Setup

Models and baselines. We evaluate Qwen3-4B and Qwen3-8B (Yang et al., 2025) on RULER (Hsieh et al., 2024), and Qwen3-8B on LongBench (Bai et al., 2024), Math500 (Lightman et al., 2023), and AIME25, with uncompressed Full-KV as the reference throughout. The long-context baselines are SnapKV (Li et al., 2024), KeyDif (Park et al., 2026), DMA (Shi et al., 2025), and TrimKV (Bui et al., 2025), plus KVpop (Hauzenberger et al., 2026) on RULER; the math baselines are R-KV (Cai et al., 2025), TriAttention (Mao et al., 2026), KVzap (Jegou and Jeblick, 2026), Fast-KVzip (Kim et al., 2026), and Random Attention (Wang et al., 2026). DMA produces a query-independent score from value representations, which in our adaptation is added to the attention logits before the softmax during training and used for hard eviction at inference (Appendix G). TrimKV uses its released gate weights. Unless otherwise specified, every method trained in this work updates only its gate parameters and keeps the backbone frozen.

Tasks. RULER (8K and 16K contexts) and LongBench use greedy decoding at 50% and 75% compression. For a given input and compression ratio, every method receives the same average per-KV-head budget, which is inclusive of attention sinks, the local window, and any auxiliary slots the method maintains; Appendix H gives the budget accounting and the protected region of each method. LongBench covers 15 tasks with 3,550 examples. We leave out TREC because its question-classification score is highly sensitive to prompt formatting and non-monotonic in the cache budget, which makes it a poor measure of compression quality. For math reasoning, we estimate pass@1 from two rollouts per Math500 problem and eight per AIME25 problem, with thinking enabled, temperature 0.6, and top-� 0.95. Budgets, counted the same way, are 512 and 1,024 entries for Math500 and 1,024 and 2,048 for AIME25; Appendix I lists the generation limits, seeds, and problem subsets of each run.

Training IndexMem++. We train only the gate, with the backbone frozen, on long documents from LongMino (Sagels, 2026) under an 8K→16K length curriculum. The gate is the configuration selected in Section 3, a two-layer MLP of width 256 on the hidden state with a per-key retention rate, trained by reverse-KL distillation against the same model with full KV; four attention sinks and a 128-token local window are always attended. The head-budget allocator (Section 3.1) and the centroid slots (Section 3.3) act only at inference and add no trained parameters. On Math500 and AIME25 the prompt is short and the cache fills up during generation, so the allocator, which reads the attention at the end of prefill, has almost nothing to work from. There we use a head table fitted ofline for each budget instead of the per-document allocator, and no centroid slots (Appendix G).

## 4.2. Long-Context Understanding

RULER highlights the importance of identifying redundancy. The prefill cache is compressed before the target question is available. SnapKV scores relevance from the attention of the queries it has seen, and recent-context attention says little about what a future query will need, which is why it degrades substantially on the retrieval tasks. Methods that score redundancy directly, including KeyDif, TrimKV, and KVpop, do better. A single needle is easy to distinguish from filler text; MultiKey-2/3 are not, since at 8K they hold 425 and 125 distractor records sharing the target’s template and key–value type, and with the question still unseen every record is potentially relevant and no distractor can be ruled out in advance. This is where similarity-based criteria degrade sharply and IndexMem++ remains robust. DMA also degrades sharply at high compression. Its score is query-independent, as ours is, but it is computed from the values and trained with the language-modeling loss rather than distilled from the Full-KV model, and the resulting ranking transfers poorly to hard eviction on the retrieval-heavy tasks.

LongBench highlights the importance of information recoverability. IndexMem++ achieves the highest overall average at both compression ratios and consistently leads on multi-document QA. The document-QA results separate information recoverability from proxy importance: low recent attention in SnapKV does not mean an entry has become redundant, the geometric distinctiveness KeyDif relies on says nothing about whether the same information is represented elsewhere in the cache, and TrimKV’s bias toward recent tokens ignores that the evidence a later question needs may sit anywhere in the context. These mismatches matter most in multi-document QA, where a correct answer may depend on several pieces of evidence spread across documents and positions, none of which need be recent or geometrically distinctive. At 75% compression, IndexMem++ scores 61.57, 44.82, and 35.89 on HotpotQA, 2WikiMQA, and MuSiQue, against 59.53, 42.40, and 31.81 for TrimKV, with KeyDif further behind, and it leads this category at both compression ratios. This is consistent with IndexMem++ better preserving evidence that remains irreplaceable despite being old, recently unattended, or geometrically unremarkable.

Table 5: RULER overall results. Results for Qwen3-4B and Qwen3-8B with 8K and 16K contexts at compression ratios of 50% and 75%. Full-KV uses the uncompressed cache, so its score is repeated in both compression columns. The Qwen3-4B checkpoint of KVpop is not publicly available.
<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-4B (8K)</td><td colspan="2">Qwen3-4B (16K)</td><td colspan="2">Qwen3-8B (8K)</td><td colspan="2">Qwen3-8B (16K)</td></tr><tr><td>50%</td><td>75%</td><td>50%</td><td>75%</td><td>50%</td><td>75%</td><td>50%</td><td>75%</td></tr><tr><td>Full-KV</td><td>92.41</td><td>92.41</td><td>91.64</td><td>91.64</td><td>93.86</td><td>93.86</td><td>92.96</td><td>92.96</td></tr><tr><td>SnapKV</td><td>54.24</td><td>30.85</td><td>61.61</td><td>35.80</td><td>54.32</td><td>35.13</td><td>62.74</td><td>41.54</td></tr><tr><td>KeyDiff</td><td>73.72</td><td>62.31</td><td>70.00</td><td>61.99</td><td>77.35</td><td>67.14</td><td>74.44</td><td>66.87</td></tr><tr><td>DMA</td><td>68.18</td><td>36.34</td><td>64.69</td><td>33.30</td><td>75.22</td><td>41.76</td><td>71.78</td><td>36.19</td></tr><tr><td>TrimKV</td><td>86.38</td><td>75.42</td><td>86.40</td><td>76.29</td><td>87.23</td><td>80.26</td><td>85.91</td><td>79.09</td></tr><tr><td>KVpop</td><td></td><td></td><td></td><td></td><td>88.03</td><td>72.71</td><td>85.49</td><td>71.81</td></tr><tr><td>IndexMem++</td><td>91.75</td><td>83.89</td><td>89.18</td><td>76.56</td><td>93.79</td><td>90.14</td><td>90.87</td><td>80.93</td></tr></table>

Table 6: RULER task-level results at 8K and a compression ratio of 75%. Qwen3-4B and Qwen3-8B on all 13 RULER tasks.
<table><tr><td colspan="2" rowspan="2">Model Method</td><td colspan="11">RULER Tasks (8K)</td></tr><tr><td>S1 S2</td><td>S3</td><td>MK1</td><td>MK2</td><td>MK3</td><td></td><td>MV</td><td>MQ VT</td><td>CWE FWE</td><td>QA1 QA2</td><td>Avg.</td></tr><tr><td rowspan="6">Ow--4B</td><td>Full-KV</td><td>100.00</td><td>100.00</td><td>0100.00</td><td>99.80</td><td>99.80</td><td>99.80</td><td>99.85</td><td>99.70</td><td>99.96 84.56</td><td>86.27 75.20</td><td>56.40 92.41</td></tr><tr><td>SnapKV</td><td>70.20</td><td>22.20</td><td>2.40</td><td>15.40</td><td>8.00</td><td>3.40</td><td>14.05</td><td>13.85</td><td>38.36 62.18 89.00</td><td>29.60</td><td>32.40 30.85</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>88.60</td><td>92.00</td><td>74.60</td><td>4.60</td><td>0.00</td><td>89.90</td><td>89.50 100.00</td><td>69.14 74.07</td><td>15.00</td><td>12.60 62.31</td></tr><tr><td>DMA</td><td>99.00</td><td>24.20</td><td>2.40</td><td>27.40</td><td>5.40</td><td>0.00</td><td>14.30</td><td>20.70</td><td>79.60 48.16 71.47</td><td>47.80</td><td>32.0036.34</td></tr><tr><td>TrimKV</td><td>100.00</td><td>98.80</td><td>92.60 95.00</td><td></td><td>48.40</td><td>10.80</td><td>91.55</td><td>95.60</td><td>99.96 77.00 71.60</td><td>49.6049.6075.42</td><td></td></tr><tr><td>IndexMem++</td><td>-100.00</td><td>99.20</td><td>91.20 97.80 </td><td></td><td>94.80</td><td>60.40</td><td>98.40</td><td>99.20 100.00</td><td>44.28 84.07</td><td>68.00 53.20 83.89</td><td></td></tr><tr><td rowspan="6">Owe-8B</td><td>Full-KV</td><td>100.00</td><td>100.00</td><td>0100.00</td><td>99.80</td><td></td><td>99.60 99.60 100.00</td><td></td><td>100.00 100.00</td><td>94.24 91.00</td><td>75.80</td><td>60.20 93.86</td></tr><tr><td>SnapKV</td><td>76.00</td><td>21.20</td><td>2.40</td><td>16.00</td><td>5.80</td><td>4.40</td><td>10.70</td><td>12.40</td><td>66.00 78.32 89.33</td><td>36.20</td><td></td></tr><tr><td>KeyDiff</td><td>100.00</td><td>95.20</td><td>94.80</td><td>086.60</td><td>6.80</td><td>1.00</td><td>98.20</td><td>95.90</td><td>100.00 83.62</td><td>74.8720.20</td><td>38.00 35.13 15.60 67.14</td></tr><tr><td>DMA</td><td>91.20</td><td>26.00</td><td>13.00</td><td>34.40</td><td>8.20</td><td>0.00</td><td>18.85</td><td>29.05</td><td>92.72 58.24 80.67 55.00 35.60 41.76</td><td></td><td></td></tr><tr><td>TrimKV</td><td>100.00</td><td>100.00</td><td>99.60</td><td>97.80</td><td>51.00</td><td>4.40</td><td>97.80</td><td>99.35</td><td>100.0091.34 91.27</td><td>58.00 52.80 80.26</td><td></td></tr><tr><td>KVpop</td><td>100.00</td><td>94.20</td><td>83.20</td><td>79.20</td><td>37.80</td><td>2.00</td><td>89.55</td><td>89.70 100.00 92.06</td><td></td><td>86.9344.2046.4072.71</td><td></td></tr><tr><td>IndexMem++ 100.00 100.00</td><td></td><td></td><td>96.40 99.20 97.80 73.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td>98.80 100.00 100.00 83.18 91.40 71.40 60.60 90.14</td><td></td><td></td></tr></table>

S1–S3 = single-needle tasks; MK1–MK3 = multi-key tasks; MV = multi-value; MQ = multi-query; VT = variable tracking; CWE/FWE = common/frequent word extraction; QA1/QA2 = question answering.

Table 7: LongBench results on Qwen3-8B. Category means and the average over 15 tasks, at 50% and 75% compression.
<table><tr><td>Method</td><td>Compression ratio</td><td>Single QA</td><td>Multi QA</td><td>Summ.</td><td>Few-shot</td><td>Synth.</td><td>Code</td><td>Avg.</td></tr><tr><td>Full-KV</td><td></td><td>41.96</td><td>48.58</td><td>27.61</td><td>65.40</td><td>50.30</td><td>62.47</td><td>47.39</td></tr><tr><td>SnapKV</td><td>50%</td><td>35.60</td><td>46.52</td><td>25.93</td><td>65.26</td><td>50.00</td><td>63.10</td><td>45.39</td></tr><tr><td>KeyDiff</td><td>50%</td><td>28.87</td><td>28.91</td><td>24.46</td><td>62.27</td><td>45.61</td><td>45.61</td><td>36.91</td></tr><tr><td>DMA</td><td>50%</td><td>38.87</td><td>42.72</td><td>26.53</td><td>64.33</td><td>35.25</td><td>55.13</td><td>42.25</td></tr><tr><td>TrimKV</td><td>50%</td><td>41.93</td><td>46.29</td><td>27.20</td><td>67.81</td><td>51.75</td><td>56.35</td><td>46.54</td></tr><tr><td>IndexMem++</td><td>50%</td><td>40.82</td><td>49.06</td><td>27.64</td><td>65.27</td><td>52.61</td><td>62.32</td><td>47.53</td></tr><tr><td>SnapKV</td><td>75%</td><td>28.59</td><td>39.69</td><td>24.12</td><td>65.04</td><td>47.92</td><td>63.62</td><td>42.02</td></tr><tr><td>KeyDiff</td><td>75%</td><td>20.14</td><td>19.35</td><td>19.12</td><td>58.91</td><td>26.58</td><td>38.03</td><td>28.19</td></tr><tr><td>DMA</td><td>75%</td><td>29.17</td><td>34.39</td><td>23.54</td><td>59.69</td><td>5.00</td><td>50.09</td><td>32.72</td></tr><tr><td>TrimKV</td><td>75%</td><td>39.64</td><td>44.58</td><td>27.00</td><td>66.45</td><td>51.50</td><td>55.84</td><td>45.41</td></tr><tr><td>IndexMem++</td><td>75%</td><td>38.69</td><td>47.43</td><td>26.22</td><td>64.47</td><td>50.96</td><td>61.26</td><td>46.03</td></tr></table>

QA = question answering; Summ. = summarization; Synth. = synthetic tasks. Few-shot is the mean of TriviaQA and SAMSum; Avg. is the equal-weight mean of all 15 tasks.

The two synthetic tasks stress complementary forms of information preservation. PassageRetrieval requires enough semantic detail to tell the target passage from competing ones, whereas PassageCount requires global coverage of the unique passages despite repeated content. DMA degrades on both under hard eviction, as on RULER, most visibly on PassageRetrieval at 75% compression, where it drops to 8.50 while IndexMem++ retains 94.42; on PassageCount the corresponding scores are 1.50 and 7.50. IndexMem++ trains its scorer against the full-KV prediction distribution, which gives entries whose attenuation changes the model’s predictions a direct signal to raise their retention priority. Appendix H.2 reports the complete per-task results.

## 4.3. Preserving Reasoning State under KV-Cache Compression

Table 8: Mathematical reasoning with Qwen3-8B. Each cell reports pass@1, with the mean number of generated tokens shown in smaller gray text.
<table><tr><td></td><td colspan="2">Math500</td><td colspan="2">AIME25</td></tr><tr><td>Method</td><td>K=1024</td><td>K=512</td><td>K=2048</td><td>K=1024</td></tr><tr><td>Full-KV</td><td>0.832 (5,258)</td><td></td><td>0.700 (17,845)</td><td></td></tr><tr><td>KVzap</td><td>0.792† (6,326)</td><td>0.548† (9,850)</td><td>0.446 (22,019)</td><td>0.183 (29,013)</td></tr><tr><td>Fast-KVzip</td><td>0.736† (6,974)</td><td>0.480† (11,146)</td><td>0.375 (23,904)</td><td>0.154(29,462)</td></tr><tr><td>R-KV</td><td>0.811 (9,741)</td><td>0.605 (12,757)</td><td>0.229 (27,044)</td><td>0.121 (29,625)</td></tr><tr><td>TriAttention</td><td>0.863 (8,563)</td><td>0.758 (12,056)</td><td>0.383 (23,109)</td><td>0.192(28,507)</td></tr><tr><td>Random Attention</td><td>0.869 (7,342)</td><td>0.784(9,714)</td><td>0.388 (23,199)</td><td>0.221 (27,538)</td></tr><tr><td>IndexMem++</td><td>0.824 (5,226)</td><td>0.794 (5,423)</td><td>0.600 (18,180)</td><td>0.450 (20,279)</td></tr></table>

Math500 uses two rollouts per problem, and AIME25 I+II uses eight. <sup>†</sup>Evaluated on a 125-problem subset of Math500 (Appendix I).

The strong performance of Random Attention suggests that, once the problem prompt is protected, many intermediate chain-of-thought states are redundant. At the lower compression setting on Math500 (� = 1024), Random Attention is more accurate than IndexMem++ (0.869 vs. 0.824) but needs substantially longer generation (7,342 vs. 5,226 tokens), and at � = 512 the order flips, with IndexMem++ both more accurate (0.794 vs. 0.784) and more eficient (5,423 vs. 9,714 tokens). On AIME25 the gap is wide, 0.600 vs. 0.388 at � = 2048 and 0.450 vs. 0.221 at � = 1024. IndexMem++ also stays close to Full-KV in generation length, 5,226 and 5,423 tokens on Math500 against 5,258, and 18,180 and 20,279 tokens on AIME25 against 17,845, whereas every other method generates longer outputs at the same budget. This suggests that IndexMem++ preserves the reasoning states needed under high compression rather than forcing the model to compensate with longer chains of thought.

## 4.4. What IndexMem++ Retains

We examine whether IndexMem++ retains early context or reduces to a sliding window. The control is a recency policy that keeps the most recent entries under the same per-head budgets and centroid slots. Both run with physical eviction on Qwen3-8B at � = 2,048, on long AIME25 reasoning traces and on software-engineering agent trajectories (SWE); Appendix J gives the setup.

IndexMem++ does not reduce to a sliding window. When eight traces per domain are fed token by token up to 8K, only 52.5% (AIME25) and 48.4% (SWE) of the entries it retains outside the sinks and local window fall inside a recency window of the same size, against 100% for the recency policy. Early facts also stay cached until they are needed. In 17 of the 25 SWE trajectories, a fact from the issue description is applied later on, 39 facts in all; every one of them is still cached at the point of use, whereas the recency policy has already dropped the original entries of 37. Under the same budget, IndexMem++ also predicts the 256 assistant tokens after the 8K prefix better than the recency policy on every trace, with a lower negative log-likelihood (NLL) of 0.207 vs. 0.279 on AIME25 and 1.221 vs. 1.551 on SWE. In free generation on AIME25, it reaches a pass@1 of 0.592, against 0.121 for the recency policy and 0.654 for Full-KV.

## 4.5. Decoding Eficiency under a Fixed Budget

![](images/5c8dcb237cf154680319e6b8d606961dfb0d0b6f0a729b119fe7283b826f581d.jpg)  
Figure 5: Decode throughput versus input length for Qwen3-4B and Qwen3-8B. Full-KV\* is the Full-KV reference measured under the protocol of Appendix D.

Figure 5 reports batch throughput for 256 decode steps on a single H20 GPU at an average per-KV-head budget of 2,048 entries, excluding prefill and setup costs; the missing Full-KV\* points are out-of-memory runs, and Appendix D gives the full protocol. IndexMem++ largely decouples decoding throughput from the input sequence length. Continuous eviction bounds the historical KV cache of each sequence by a fixed budget, so the per-step KV reads and attention computation depend on that budget rather than on the sequence length, whereas Full-KV must read a cache that keeps growing. At a batch size of 32, increasing the input length from 8K to 256K (32×) reduces the throughput of IndexMem++ by only about 28% on both models.

## 5. Related work

Learnable eviction gates. Learnable eviction replaces fixed heuristics with trainable, content-dependent scores. TrimKV (Bui et al., 2025) predicts per-key retention scores that decay with age, while DMC (Nawrot et al., 2024) and DMS (Łańcucki et al., 2025) learn discrete token-level compression decisions through continuous relaxations and explicit budget objectives. SP-KV (Szilvasy et al., 2026) predicts future utility to gate long-range attention, whereas Locret (Huang et al., 2024) and Attention-Gate (Zeng et al., 2024) train auxiliary modules to score entries during prefill. Although motivated diferently, these methods share the same paradigm: a lightweight scorer assigns retention scores to KV entries. We study the scorer, gate normalization and budget, supervision, and allocation across heads in Sections 2 and 3.

Recovering evicted information. Summarizing a discarded past is a long-standing idea: compressive memories retain a coarse record of activations that leave the attention window (Rae et al., 2020), gist tokens compress a prompt into a few learnable slots (Mu et al., 2023), and recurrent or test-time-trained states have been used as memory alongside attention (Sun et al., 2024a; Behrouz et al., 2024). Such a summary can be read back either as a residual added to the attention output, whose scale is set by a gain learned outside the softmax, or as additional participants in attention, which compete for mass in the same softmax and are thus normalized by construction. We find this readout choice matters more than the capacity of the summary (Section 3).

## 6. Conclusion

Asked which of its own KV entries are expendable, a gate trained on the model’s own loss answers that keeping everything is preferable, and returns constant scores. The question becomes answerable once that answer is unavailable: the evictable entries compete for a fixed gate mass, and the gate matches the dense model rather than the next token. What the score cannot say, how much budget each head deserves, is read of the mass each head retains. Some information is lost even so, and there the lesson is not to build a larger memory but to read the small one through the same softmax as the cache it supplements.

## 7. Future work

Our gate is trained on dense or teacher-forced prefixes but applied to a cache its own earlier decisions have compressed. On-policy distillation, with prefixes generated under hard eviction, would expose it to the state distribution it actually meets at inference.

## AI use statement

We used generative AI tools in two roles. First, as coding assistants during the development of our training and evaluation code, including the fused training-time attention kernel of Appendix B and the paged decode path of Appendix C. All such code was reviewed by the authors, and every reported number comes from runs of code we have inspected; correctness of the kernels was additionally checked against reference implementations, as described in the corresponding appendices. Second, as a writing aid for improving the clarity and grammar of the manuscript. We did not use generative AI tools to generate research ideas, to design the method, to produce or modify experimental results, or to write the analysis of those results. We take responsibility for the final content of this work, including all text, claims, and artifacts.

## Ethics statement

This work studies KV-cache eviction for long-context inference. Our experiments use publicly released model checkpoints and public benchmarks, and involve no human subjects, no personally identifying information, and no newly collected data. We release no new dataset or model. The method reduces the memory and bandwidth cost of serving long-context models, which lowers the cost of deployment; it does not change what those models can be asked to do, and so inherits the capabilities and the risks of the underlying checkpoints rather than introducing new ones. One consideration specific to eviction deserves mention: because compression discards part of the cache, a compressed model can lose evidence that the uncompressed model would have used, and the resulting errors are not always visible to the user. Deployments in settings where faithfulness to the provided context matters should therefore validate accuracy at the compression ratio they intend to use, rather than assuming the ratios reported here transfer.

## Reproducibility statement

Section 4.1 states the models, baselines, benchmarks, and compression ratios; Appendix G gives the complete training recipe, including the training corpus and its subsets, the length curriculum, the optimizer schedule, the gate configuration, the distillation objective, and the inference-time components that add no trained parameters. Appendix H reports per-task results for all RULER and LongBench tasks, together with the budget accounting, retention criterion, and protected region used for each method, so that the budget each baseline receives is explicit. Appendix F gives the protocol for the pilot ablations of Section 3, including the seeds, and states which tasks carry no usable signal at that model scale. Appendices B and C describe the training kernel and the conversion from soft gating to hard eviction, with the core routines listed as code. Appendix D gives the measurement protocol for the decoding-throughput comparison, including the decode implementation used for each curve.

## References

Sotiris Anagnostidis, Dario Pavllo, Luca Biggio, Lorenzo Noci, Aurelien Lucchi, and Thomas Hofmann. Dynamic context pruning for eficient and interpretable autoregressive transformers, 2023. URL https://arxiv.org/ abs/2305.15805.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. LongBench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the Association for Computational Linguistics, 2024.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. Titans: Learning to memorize at test time, 2024. URL https://arxiv.org/abs/2501.00663.

Ngoc Bui, Shubham Sharma, Simran Lamba, Saumitra Mishra, and Rex Ying. Cache what lasts: Token retention for memory-bounded KV cache in LLMs, 2025.

Zefan Cai, Wen Xiao, Hanshi Sun, Cheng Luo, Yikai Zhang, Ke Wan, Yucheng Li, Yeyang Zhou, Li-Wen Chang, Jiuxiang Gu, Zhen Dong, Anima Anandkumar, Abedelkadir Asi, and Junjie Hu. R-KV: Redundancy-aware KV cache compression for reasoning models, 2025. URL https://arxiv.org/abs/2505.24133.

DeepSeek-AI. DeepSeek-V2: A strong, economical, and eficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

DeepSeek-AI. DeepSeek-V3.2: Pushing the frontier of open large language models, 2025. URL https: //arxiv.org/abs/2512.02556.

Alessio Devoto, Yu Zhao, Simone Scardapane, and Pasquale Minervini. A simple and efective $L _ { 2 }$ norm-based strategy for KV cache compression, 2024. URL https://arxiv.org/abs/2406.11430.

Jusen Du, Jiaxi Hu, Tao Zhang, Weigao Sun, and Yu Cheng. Native hybrid attention for eficient sequence modeling, 2025. URL https://arxiv.org/abs/2510.07019.

Lizhe Fang, Yifei Wang, Zhaoyang Liu, Chenheng Zhang, Stefanie Jegelka, Jinyang Gao, Bolin Ding, and Yisen Wang. What is wrong with perplexity for long-context language modeling?, 2024. URL https: //arxiv.org/abs/2410.23771.

Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao Wei, Tong Xu, and Xiangyu Zhao. Sliding window attention training for eficient large language models. arXiv preprint arXiv:2502.18845, 2025.

Daniel Goldstein, Navneel Singhal, and Eugene Cheah. Key-value means: Transformers with expandable block-recurrent compressed memory, 2026. URL https://arxiv.org/abs/2605.09877.

Xiangming Gu, Tianyu Pang, Chao Du, Qian Liu, Fengzhuo Zhang, Cunxiao Du, Ye Wang, and Min Lin. When attention sink emerges in language models: An empirical view. In International Conference on Learning Representations, 2025.

Lukas Hauzenberger, Niklas Schmidinger, Anamaria-Roberta Hartl, David Stap, Thomas Schmied, Sebastian Böck, Günter Klambauer, and Sepp Hochreiter. KVpop – key-value cache compression with predictive online pruning, 2026.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. RULER: What’s the real context size of your long-context language models? In Conference on Language Modeling, 2024.

Jie Hu, Shengnan Wang, Yutong He, Ping Gong, Jiawei Yi, Juncheng Zhang, Youhui Bai, Renhai Chen, Gong Zhang, Cheng Li, and Kun Yuan. CentroidKV: Eficient long-context LLM inference via KV cache clustering, 2025. URL https://arxiv.org/abs/2506.11418.

Xiang Hu, Xinyu Wei, Hao Gu, Minshen Zhang, Tian Liang, Huayang Li, Lei Zhu, Yan Wang, Sirui Han, Yushi Bai, Kewei Tu, Haitao Mi, and Leo Liang. Hierarchical sparse attention done right: Toward infinite context modeling, 2026. URL https://arxiv.org/abs/2607.02980.

Yuxiang Huang, Binhang Yuan, Xu Han, Chaojun Xiao, and Zhiyuan Liu. Locret: Enhancing eviction in long-context LLM inference with trained retaining heads on consumer-grade devices, 2024. URL https: //arxiv.org/abs/2410.01805.

Simon Jegou and Maximilian Jeblick. KVzap: Fast, adaptive, and faithful KV cache pruning, 2026. URL https://arxiv.org/abs/2601.07891.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H. Abdi, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. MInference 1.0: Accelerating pre-filling for long-context LLMs via dynamic sparse attention. In Advances in Neural Information Processing Systems, volume 37, 2024.

Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, and Hyun Oh Song. KVzip: Queryagnostic KV cache compression with context reconstruction. In Advances in Neural Information Processing Systems, volume 38, 2025.

Jang-Hyun Kim, Dongyoon Han, and Sangdoo Yun. Fast KVzip: Eficient and accurate LLM inference with gated KV eviction, 2026. URL https://arxiv.org/abs/2601.17668.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626, 2023.

Adrian Łańcucki, Konrad Staniszewski, Piotr Nawrot, and Edoardo M. Ponti. Inference-time hyper-scaling with KV cache compression. In Advances in Neural Information Processing Systems, 2025.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Selective attention improves transformer. In International Conference on Learning Representations, 2025.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. SnapKV: LLM knows what you are looking for before generation. In Advances in Neural Information Processing Systems, volume 37, pages 22947–22970, 2024.

Zhiwei Li, Lei Zhu, Hao Gu, Xiang Hu, Yan Wang, Haitao Mi, Sirui Han, Leo Liang, and Zhijiang Guo. SAS: Simple attention sparsification via end-to-end optimization of context ranking, 2026. URL https: //arxiv.org/abs/2609.13141.

Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step, 2023. URL https://arxiv.org/abs/ 2305.20050.

Zhixuan Lin, Evgenii Nikishin, Xu Owen He, and Aaron Courville. Forgetting transformer: Softmax attention with a forget gate. In International Conference on Learning Representations, 2025.

Zirui Liu, Jiayi Yuan, Hongye Jin, Shaochen Zhong, Zhaozhuo Xu, Vladimir Braverman, Beidi Chen, and Xia Hu. KIVI: A tuning-free asymmetric 2bit quantization for KV cache. In Proceedings of the 41st International Conference on Machine Learning, volume 235 ofProceedings ofMachine Learning Research, pages 32332–32344. PMLR, 2024.

Enzhe Lu, Zhejun Jiang, Jingyuan Liu, Yulun Du, Tao Jiang, Chao Hong, Shaowei Liu, Weiran He, Enming Yuan, Yuzhi Wang, Zhiqi Huang, Huan Yuan, Suting Xu, Xinran Xu, Guokun Lai, Yanru Chen, Huabin Zheng, Junjie Yan, Jianlin Su, Yuxin Wu, Neo Y. Zhang, Zhilin Yang, Xinyu Zhou, Mingxing Zhang, and Jiezhong Qiu. MoBA: Mixture of block attention for long-context LLMs, 2025. URL https://arxiv.org/abs/2502.13189.

Weian Mao, Xi Lin, Wei Huang, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, and Yukang Chen. TriAttention: Eficient long reasoning with trigonometric KV compression, 2026. URL https://arxiv.org/abs/2604.04921.

Jesse Mu, Xiang Lisa Li, and Noah Goodman. Learning to compress prompts with gist tokens. In Advances in Neural Information Processing Systems, volume 36, 2023.

Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Eficient infinite context transformers with Infini-attention, 2024. URL https://arxiv.org/abs/2404.07143.

Piotr Nawrot, Adrian Łańcucki, Marcin Chochowski, David Tarjan, and Edoardo M. Ponti. Dynamic memory compression: Retrofitting LLMs for accelerated inference, 2024. URL https://arxiv.org/abs/2403.09636.

Junyoung Park, Dalton Jones, Matthew J. Morse, Raghavv Goel, Mingu Lee, and Chris Lott. KeyDif: Key similarity-based KV cache eviction for long-context LLM inference in resource-constrained environments, 2026. URL https://arxiv.org/abs/2504.15364.

Ofir Press, Noah A. Smith, and Mike Lewis. Train short, test long: Attention with linear biases enables input length extrapolation. In International Conference on Learning Representations, 2022.

Zihan Qiu, Zekun Wang, Bo Zheng, Zeyu Huang, Kaiyue Wen, Songlin Yang, Rui Men, Le Yu, Fei Huang, Suozhi Huang, Dayiheng Liu, Jingren Zhou, and Junyang Lin. Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free, 2025. URL https://arxiv.org/abs/2505.06708.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020.

Sagels. LongMino: A filtered long-context pretraining corpus. https://huggingface.co/datasets/sagels/ longmino\_256k\_filtered, 2026. HuggingFace dataset sagels/longmino\_256k\_filtered.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc V. Le, Geofrey Hinton, and Jef Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer, 2017. URL https://arxiv.org/abs/1701.06538.

Jingze Shi, Yifan Wu, Yiran Peng, Bingheng Wu, Liangdong Wang, Guang Liu, and Yuyu Luo. Trainable dynamic mask sparse attention, 2025.

Sainbayar Sukhbaatar, Da Ju, Spencer Pof, Stephen Roller, Arthur Szlam, Jason Weston, and Angela Fan. Not all memories are created equal: Learning to forget by expiring. In Proceedings of the 38th International Conference on Machine Learning, 2021.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (learn at test time): RNNs with expressive hidden states, 2024a. URL https://arxiv.org/abs/2407.04620.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. In Advances in Neural Information Processing Systems, volume 37, 2024b.

Kabir Swain, Sijie Han, Daniel Karl I. Weidele, Mauro Martino, and Antonio Torralba. Tensor cache: Evictionconditioned associative memory for transformers, 2026. URL https://arxiv.org/abs/2605.22884.

Gergely Szilvasy, Manuel Faysse, Maria Lomeli, Matthijs Douze, Pierre-Emmanuel Mazaré, Loïc Cabannes, Wen-tau Yih, and Hervé Jégou. Self-pruned key-value attention: Learning when to write by predicting future utility, 2026. URL https://arxiv.org/abs/2605.14037.

Shawn Tan, Songlin Yang, Aaron Courville, Rameswar Panda, and Yikang Shen. Scaling stick-breaking attention: An eficient implementation and in-depth study, 2024. URL https://arxiv.org/abs/2410.17980.

Jiaming Tang, Yilong Zhao, Kan Zhu, Guangxuan Xiao, Baris Kasikci, and Song Han. Quest: Query-aware sparsity for eficient long-context LLM inference. In Proceedings of the 41st International Conference on Machine Learning, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. LLaMA: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Zhongwei Wan, Xinjian Wu, Yu Zhang, Yi Xin, Chaofan Tao, Zhihong Zhu, Xin Wang, Siqi Luo, Jing Xiong, Longyue Wang, and Mi Zhang. D2o: Dynamic discriminative operations for eficient long-context inference of large language models, 2025. URL https://arxiv.org/abs/2406.13035.

Bailin Wang, Chang Lan, Chong Wang, and Ruoming Pang. RAttention: Towards the minimal sliding window size in local-global attention models, 2025. URL https://arxiv.org/abs/2506.15545.

Heng Wang, Jielin Qiu, Wenting Zhao, Cheng Qian, Liangwei Yang, Jiawei Han, Heng Ji, Silvio Savarese, Shelby Heinecke, and Huan Wang. Random attention: Rethinking KV cache eviction for eficient reasoning, 2026. URL https://arxiv.org/abs/2609.03430.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Eficient streaming language models with attention sinks. In International Conference on Learning Representations, 2024.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Xintong Yang, Hao Gu, Binxing Xu, Lujun Li, Bei Liu, Jiacheng Liu, Qiyuan Zhu, Yike Guo, and Sirui Han. Indexmem: Learned kv-cache eviction with latent memory for long-context llm inference. arXiv preprint arXiv:2605.25475, 2026.

Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023.

Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, and Wangding Zeng. Native sparse attention: Hardware-aligned and natively trainable sparse attention, 2025. URL https: //arxiv.org/abs/2502.11089.

Zihao Zeng, Bokai Lin, Tianqi Hou, Hao Zhang, and Zhijie Deng. In-context KV-cache eviction for LLMs via Attention-Gate, 2024. URL https://arxiv.org/abs/2410.12876.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. H2O: Heavy-hitter oracle for eficient generative inference of large language models. In Advances in Neural Information Processing Systems, volume 36, pages 34661–34710, 2023.

## Appendix contents

A Other related work 19   
A.1 Training-free eviction. 19   
A.2 Sparse attention. 19   
A.3 Gating in attention. 19   
A.4 Recovering evicted information. 23   
B Training-time kernel for gated attention 25   
C From the soft gate to hard eviction 27   
D Eficiency 27   
E Details deferred from the method 28   
E.1 Both halves of the always-attended construction are required. 28   
E.2 Loss design. 29   
E.3 Heads budget design. 29   
E.4 The three history scorers, and why each is a single-variable test. 30   
E.5 Retention-rate . 31   
F Pilot ablation protocol 31   
G IndexMem++ implementation settings 31   
H Additional Long-Context Results 32   
H.1 Complete RULER results 33   
H.2 Complete LongBench results 33   
I Math Reasoning Protocol 36   
J Analysis of Retained Entries 37

## A. Other related work

This appendix discusses other works that Section 5 did not claim in detail.

## A.1. Training-free eviction.

Training-free policies difer mainly in the signal they use to score entries. The largest group derives importance from attention already observed. $\mathrm { H _ { 2 } O }$ retains heavy hitters by accumulating attention over past queries (Zhang et al., 2023; Wan et al., 2025), SnapKV pools attention within a trailing observation window to decide what a prompt will need (Li et al., 2024), and R-KV adds a redundancy term so that repetitive reasoning traces are not retained many times over (Cai et al., 2025). Because these scores depend on which queries have been seen, the induced ranking is revised as decoding proceeds, while the deletion it triggers is permanent.

A second group avoids queries altogether and scores entries from the statistics of the keys themselves. TriAttention compares keys against calibrated pre-RoPE query centers in the frequency domain (Mao et al., 2026), while the $L _ { 2 }$ key norm (Devoto et al., 2024) and KeyDif’s angular diversity criterion (Park et al., 2026) use the magnitude and the mutual similarity of the key vectors. These criteria are query-independent, which suits the irreversibility of eviction, but the statistic itself is fixed by hand. A third group uses position alone: StreamingLLM retains attention sinks together with a sliding window (Xiao et al., 2024), and uniformly random retention serves as the natural control for how much of the benefit is attributable to the scoring at all (Wang et al., 2026). None of these methods requires training, which is their principal appeal; the cost is that a hand-chosen signal cannot adapt to what a particular context happens to make redundant.

## A.2. Sparse attention.

Sparse attention also computes attention over a subset of entries, but it retains the full cache and chooses a new subset for every query, so no memory is reclaimed and no decision is irreversible. Training-free variants select blocks by cheap surrogates of the attention score (Tang et al., 2024; Jiang et al., 2024), and a growing body of work trains the selector jointly with the model (Yuan et al., 2025; Lu et al., 2025; DeepSeek-AI, 2025; Shi et al., 2025; Hu et al., 2026). The trained selectors are close to our setting in machinery: they too must pass gradients through a discrete choice, and some study explicitly how a block-level gate should be injected into the attention computation and normalized (Li et al., 2026). They are far from our setting in objective, however, because a selection that a query gets wrong there can be revised at the next query, whereas an eviction cannot.

## A.3. Gating in attention.

The term gate is used for several mechanisms at diferent stages of attention, which have diferent physical meanings. Let

$$
s _ { i j } = { \frac { q _ { i } ^ { \top } k _ { j } } { \sqrt { d } } } , \qquad a _ { i j } = { \frac { \exp ( s _ { i j } ) } { \sum _ { \ell } \exp ( s _ { i \ell } ) } } , \qquad o _ { i } = \sum _ { j } a _ { i j } v _ { j } .\tag{7}
$$

A gate may act on the query, key, value, normalized attention output, or directly on the logits:

query gate: $q _ { i } ^ { \prime } = g _ { i } ^ { Q } \odot q _ { i } .$

controls how the current token retrieves;

key gate: $k _ { j } ^ { \prime } = g _ { j } ^ { K } \odot k _ { j }$

controls how a memory is addressed;

value gate: $v _ { j } ^ { \prime } = g _ { j } ^ { V } \odot v _ { j } ,$

controls how much payload a memory returns;

output gate: $o _ { i } ^ { \prime } = g _ { i } ^ { O } \odot \sum _ { j } a _ { i j } v _ { j } .$

controls whether the retrieved message is used;

logit gate: $a _ { i j } ^ { \prime } = \mathrm { s o f t m a x } _ { j } ( s _ { i j } + b _ { i j } )$

controls whether query � may read key �;

temperature gate: $a _ { i j } ^ { \prime } = \mathrm { s o f t m a x } _ { j } ( \gamma _ { j } s _ { i j } )$

controls how sharply key � competes;

denominator gate: $a _ { i j } ^ { \prime } = { \frac { \exp ( s _ { i j } ) } { Z _ { i } + c _ { i } } }$ ，

controls how much mass leaves the row.

These locations are not interchangeable. In particular, a gate outside the softmax changes the magnitude of

an already-normalized message but does not reallocate probability mass. For example, value gating gives

$$
o _ { i } ^ { V } = \sum _ { j } \underbrace { \frac { \exp ( s _ { i j } ) } { \sum _ { \ell } \exp ( s _ { i \ell } ) } } _ { a _ { i j } } g _ { j } ^ { V } v _ { j } .
$$

Even if $g _ { j } ^ { V }  0 ;$ , key $j$ may still consume substantial attention probability; its mass is simply discarded rather than reassigned. The same distinction applies to the gated-attention mechanism of (Qiu et al., 2025), which multiplies the output of scaled dot-product attention by a learned sigmoid,

$$
o _ { i } ^ { \mathrm { g a t e } } = g _ { i } ^ { O } \odot \left( \sum _ { j } a _ { i j } v _ { j } \right) .
$$

Besides providing a non-linearity between $W _ { V }$ and $W _ { O } ,$ this gate has been reported to suppress massive activations and attention sinks. Its role is nevertheless an output gate: it can decide that the result of a retrieval should contribute little, but it does not decide which memory receives the released attention mass. Related analyses of attention and residual sinks show that a small number of positions can absorb a disproportionate fraction of the normalized mass (Gu et al., 2025); this is also why entries that must remain permanently accessible require explicit treatment in an eviction mechanism.

By contrast, a gate inside the softmax acts directly on competition between memories. Writing

$$
b _ { i j } = \log g _ { i j } , \quad g _ { i j } > 0 , \quad a _ { i j } ^ { \prime } = \frac { \exp ( s _ { i j } + b _ { i j } ) } { \sum _ { \ell } \exp ( s _ { i \ell } + b _ { i \ell } ) } = \frac { \exp ( s _ { i j } ) g _ { i j } } { \sum _ { \ell } \exp ( s _ { i \ell } ) g _ { i \ell } } .
$$

Thus an additive bias in logit space is exactly a multiplicative gate on the unnormalized attention weights, followed by renormalization. Suppressing one entry therefore transfers its probability mass to the remaining entries. In the hard limit $g _ { i j }  0 , b _ { i j }  - \infty _ { ; }$ , recovering an attention mask. This makes the in-softmax location qualitatively diferent from query, key, value, or output gating: it is the natural location for a read gate and hence for eviction.

Two other mechanisms also act inside the softmax but are unsuitable as retention scores. A multiplicative gate on the $\begin{array} { r } { \mathrm { l o g i t , } s _ { i j }  \gamma _ { j } s _ { i j } , } \end{array}$ is a per-key temperature, and a scalar key gate is the same operation because $\bar { g } _ { j } ^ { K } q _ { i } ^ { \top } k _ { j }$ rescales the logit rather than shifting it. Shrinking $\gamma _ { j }$ pulls the logit towards zero, which attenuates the entry where $s _ { i j } > 0$ but amplifies it where $s _ { i j } < 0$ . Its efect therefore depends on the sign of the logit under the current query, so a single $\gamma _ { j }$ does not order entries by how much they should be retained, whereas $b _ { j }$ shifts every logit of key � in the same direction regardless of the query. Key-Value Means uses such temperatures, but to control the sharpness of already-chosen regions rather than to select among them (Goldstein et al., 2026).

A second mechanism modifies only the normalizer, $a _ { i j } = \exp ( s _ { i j } ) / ( Z _ { i } + c _ { i } )$ , as in sigmoid attention, a null or register slot, and the sink construction analyzed by (Gu et al., 2025). The mass $c _ { i } / ( Z _ { i } + c _ { i } )$ leaves the row without being assigned to any cached entry, so unlike an additive gate this suppresses an entry without making any other entry more visible. This is useful for absorbing mass that no entry should receive, which is why sinks need explicit treatment, but it provides no per-key quantity to rank.

Additive in-softmax biases beyond eviction. An additive bias inside the softmax is not specific to eviction. The same mechanism is used to encode a broad range of architectural inductive biases:

$$
a _ { i j } = \mathrm { s o f t m a x } _ { j } ( s _ { i j } + b _ { i j } ) .\tag{8}
$$

Examples include fixed positional slopes (Press et al., 2022), learned token lifetimes (Sukhbaatar et al., 2021), cumulative forget gates (Lin et al., 2025), pairwise cumulative suppression over intervening tokens (Leviathan et al., 2025; Anagnostidis et al., 2023), and query-conditioned allocations of attention mass (Tan et al., 2024). These mechanisms difer primarily in what information $b _ { i j }$ is allowed to depend on.

Softmax is invariant to a row-wise ofset:

$$
\begin{array} { r } { \mathrm { s o f t m a x } _ { j } ( s _ { i j } + b _ { i j } ) = \mathrm { s o f t m a x } _ { j } ( s _ { i j } + b _ { i j } + c _ { i } ) . } \end{array}\tag{9}
$$

<table><tr><td>Gate location</td><td>Form</td><td>Physical meaning</td><td>Redistributes mass?</td></tr><tr><td>Query</td><td> $q _ { i }  g _ { i } ^ { Q } \odot q _ { i }$ </td><td>retrieval mode</td><td>Yes</td></tr><tr><td>Key</td><td> $k _ { j }  g _ { j } ^ { K } \odot k _ { j }$ </td><td>address / retrievability</td><td>Yes, non-monotonically</td></tr><tr><td>Value</td><td> $v _ { j }  g _ { j } ^ { V } \odot v _ { j }$ </td><td>returned content</td><td>No</td></tr><tr><td>SDPA output</td><td> $o _ { i }  g _ { i } ^ { O } \odot o _ { i }$ </td><td>whether retrieval is used</td><td>No</td></tr><tr><td>Logit, multiplicative</td><td> $s _ { i j }  \gamma _ { j } s _ { i j }$ </td><td>competition sharpness</td><td>Non-monotonically</td></tr><tr><td>Denominator only</td><td> $Z _ { i } \gets Z _ { i } + c _ { i }$ </td><td>mass leaves the row</td><td>No</td></tr><tr><td>Logit, additive</td><td> $s _ { i j } \gets s _ { i j } + b _ { i j }$ </td><td>read visibility</td><td>Yes</td></tr></table>

Table 9: Gates at diferent locations in attention have diferent semantics. Among per-entry retention mechanisms, an additive logit gate provides monotonic control over normalized attention mass: lowering one entry’s bias lowers its weight and transfers mass to the others. A scalar key gate is the multiplicative logit gate, since $g _ { j } ^ { K } q _ { i } ^ { \top } \dot { k _ { j } }$ rescales the logit, and its efect on an entry’s weight depends on the sign of that logit. A denominator gate removes mass from the row without assigning it to any entry.

Hence $b _ { i j }$ is only defined up to a query-dependent constant. Removing this irrelevant degree of freedom gives a useful canonicalform. For eviction, the key distinction is whether that canonical form still depends on the query index �. If not, the gate assigns every cached entry a persistent scalar ordering; if it does, the ordering may evolve over time and an entry that is weak now can become competitive again later.

We summarize the relevant bias families in Table 10. We additionally define the fold width as the smallest � such that, up to a row constant,

$$
b _ { i j } = u _ { i } ^ { \top } v _ { j } , \qquad u _ { i } , v _ { j } \in \mathbb { R } ^ { r } .\tag{10}
$$

A small fold width allows the bias to be evaluated from compact per-query and per-key state without materializing an � × � matrix.

This distinction has three consequences for eviction. Our own gate belongs to the third row of Table 10 rather than the first: the magnitude term $m _ { j }$ alone would be a per-key scalar, but the retention rate of Equation 6 makes the canonical form depend on the query index. We state each consequence in general and then say where it places our gate.

Persistent ordering. A canonical bias that does not depend on � provides a persistent ordering of cached entries. Once an entry falls below a fixed keep budget according to this gate, advancing the sequence cannot by itself make its gate score overtake another already-existing entry. Under this condition the greedy online policy of Section 2, which selects from the surviving entries at every step, retains exactly the set that global top-� selection over the full context would retain, so irreversible deletion loses nothing. For example, both

$$
b _ { i j } = c _ { i } - c _ { j }\tag{11}
$$

and a fixed positional slope

$$
b _ { i j } = - \alpha ( i - j ) = - \alpha i + \alpha j\tag{12}
$$

reduce, after removal of the row constant, to a key-only score. By contrast, a key-specific decay rate,

$$
b _ { i j } = s _ { j } + \frac { i - j } { \mathrm { r e f } } \log \beta _ { j } ,\tag{13}
$$

cannot remove its � dependence because the coeficient of � itself depends on �. Its ranking therefore evolves with time, which is useful for forgetting but conflicts with irreversible eviction: an entry removed at one step may have become competitive at a later step had it remained available. This is the case of our Equation $^ { 6 , }$ so the equivalence above does not hold for it.

The policy itself is unchanged. At every step we still retain the current top-� of the surviving entries, exactly as in Section $2 ;$ what changes is that this is no longer the same set as top-� over the full context. An entry that leaves the retained set is deleted and never restored, so if the entries kept in its place decay faster and later fall below $\operatorname { i t } ,$ the cache keeps them and not it. Because Equation 6 is afine in $i ,$ the scores of any two entries cross at most once, so no pair can exchange places twice and an entry cannot lose to a given rival, regain the lead, and lose again. This constrains the individual comparisons but does not prevent an evicted entry from re-entering the true top-� through crossings with diferent rivals, since that boundary is an order statistic over all surviving entries rather than a single comparison. Irreversible deletion is therefore a genuine approximation under Equation 6, and Section 3.2 reports what it costs.

<table><tr><td>Bias family</td><td> $b _ { i j }$ </td><td></td><td>Canonical form Frozen gate ranking? Fold width</td><td></td></tr><tr><td>Per-key scalar</td><td> $\phi ( j )$ </td><td>φ(j)</td><td>Yes</td><td>1</td></tr><tr><td>Separable lifetime</td><td> $c _ { i } - c _ { j }$ </td><td> $- { c _ { j } }$ </td><td>Yes</td><td>1</td></tr><tr><td>Key-specific decay</td><td> $\begin{array} { r } { s _ { j } + \frac { i - j } { \mathrm { r e f } } \log \beta _ { j } } \end{array}$ </td><td>depends on i</td><td>No</td><td>2</td></tr><tr><td>Pairwise cumulative</td><td> $\textstyle \sum _ { j < k < i } S _ { k j }$ </td><td>depends on i</td><td>No</td><td> $\mathrm { \ h i g h { \it \Omega } }$ </td></tr><tr><td>Query-conditioned</td><td> $f ( q _ { i } , k _ { j } )$ </td><td>depends on i</td><td>No</td><td></td></tr></table>

Table 10: Additive in-softmax biases viewed through their canonical form. The second row includes forms such as the linear part of ALiBi and cumulative forgetting that can be written as a diference of prefix potentials; the third captures token-specific lifetime decay as used by methods such as TrimKV and by the per-key retention rate of Equation $6 ;$ the fourth and fifth capture richer pairwise or query-conditioned biases. Only biases whose canonical form is key-only induce a persistent ordering; the remaining rows can change the ordering of memories over time.

Computational foldability. A second requirement is that the bias remain inexpensive to evaluate. A key-only scalar can be stored once per cached entry. A key-specific linear decay still has constant fold width:

$$
\begin{array} { r } { b _ { i j } = \underbrace { \left[ 1 \sum _ { \boldsymbol { u } _ { i } ^ { \intercal } } \boldsymbol { i } \right] } _ { \boldsymbol { u } _ { i } ^ { \intercal } } \underbrace { \left[ \begin{array} { l l } { s _ { j } - \frac { j } { \mathrm { r e f } } \log \beta _ { j } } \\ { \frac { 1 } { \mathrm { r e f } } \log \beta _ { j } } \end{array} \right] } _ { \boldsymbol { v } _ { j } } . } \end{array}\tag{14}
$$

In contrast, cumulative pairwise suppression of the form $\textstyle \sum _ { j < k < i } S _ { k j }$ depends on interactions between a candidate memory � and the intervening sequence. In general this object does not admit a low-width factorization, and therefore requires substantially richer state or explicit pairwise bias computation. Our gate sits at the second of these three levels. Width two is what keeps the cost of the retention rate confined to bookkeeping: each cached entry stores $m _ { j }$ and log $\beta _ { j }$ instead of a single scalar, and a score at any later position follows from the pair without rescoring the hidden state, which is what allows one scorer call to serve a batch whose sequences sit at diferent absolute positions.

Forced horizons. Finally, an age-dependent bias determines whether suficiently old memories must eventually disappear regardless of their content. Suppose the content-dependent score diference is bounded by $\Delta _ { \mathrm { m a x } } .$ If the relative forgetting penalty between an old and a new entry grows without bound with their age diference, then beyond some distance it necessarily dominates $\Delta _ { \mathrm { m a x } }$ :

$$
| b _ { i j _ { \mathrm { o l d } } } - b _ { i j _ { \mathrm { n e w } } } | > \Delta _ { \mathrm { m a x } } .\tag{15}
$$

The old entry is then excluded irrespective of its semantic importance, implicitly imposing a finite horizon. A per-key gate has no such forced horizon, while a per-key lifetime model can avoid one only if it is allowed to assign a zero decay rate to entries that should persist. Our parameterization is of the second kind: the constraint log $\beta _ { j } \le 0$ in Equation 6 admits zero decay, so no horizon is imposed by construction and an entry the scorer judges important can be given an unbounded lifetime. Our initialization, however, deliberately starts the lifetime branch away from zero (Appendix E), and we do not measure whether the trained rates of the entries that survive longest return to it.

Where this leaves our gate. Our gate therefore does not occupy the simplest cell of this family. It is queryindependent in the sense that matters for eviction, namely that the score never depends on the content of a future query, but it is not query-independent after canonicalization: the retention rate makes it a key-specific decay of fold width two. It keeps the two properties that bear on cost, a compact per-entry state and no forced horizon, and gives up the third, the frozen ranking that would make online eviction exactly equivalent to global top-� selection. The pilot ablation is consistent with paying something for that: disabling the retention rate raises the average from 67.60 to 69.27 (Table 2), against a two-seed spread of 0.28 on the same 13-task average.

We keep it for the reason given in Section 3.2, a stable and explicitly parameterized age scale, and note that the frozen-ranking variant $b _ { i j } = \phi ( j )$ is the structurally cleaner choice when irreversibility is the binding constraint.

Why we relax the gate rather than estimate a gradient through a discrete decision. Softening the gate is not the only way to keep the scorer in the computation graph. A per-entry keep-or-drop decision can be sampled and diferentiated with a straight-through or Gumbel-softmax estimator, which is the route taken by DMC (Nawrot et al., 2024) and DMS (Łańcucki et al., 2025). We relax the gate instead, for two reasons. First, a straight-through estimator substitutes the gradient of the relaxed variable for the gradient of the decision that was actually executed, so the signal it delivers is biased in a way that does not vanish as training proceeds, whereas the soft gate is diferentiated exactly at the value used in the forward pass. Second, the decisions in eviction are coupled: under a cache budget, retaining one entry means dropping another, and independent per-entry Bernoulli variables cannot express this without an added penalty that pushes their expected number towards the budget. Equation 4 enforces the same constraint by construction, since the gates of the evictable set sum to $\beta$ exactly at every step. The relaxation also gives every entry a gradient at every query that can see it, rather than only at the positions where a sampled decision happened to difer.

Why we do not regress an external per-key label. A related line supervises the score against a label produced by a separate procedure rather than through the model’s own objective. KVzip (Kim et al., 2025) scores an entry by the attention it receives while the model is prompted to reconstruct its context, which is query-agnostic but costs a second prefill pass and cannot be evaluated during decoding; Fast KVzip (Kim et al., 2026) and KVzap (Jegou and Jeblick, 2026) regress that score from hidden states, removing the extra pass. This is cheaper and more stable to train than the objective we use, and we do not claim it performs worse at a matched budget. We do claim it optimizes a diferent quantity.

## A.4. Recovering evicted information.

From compression to recovery. Summarizing a past that is no longer represented token by token is a long-standing idea. Compressive memories preserve a coarse record of activations after they leave the exact attention window (Rae et al., 2020); gist tokens bottleneck a prompt into a small number of learned token states (Mu et al., 2023); and recurrent or test-time-trained states store history in the activations or parameters of a bounded memory module (Sun et al., 2024a; Behrouz et al., 2024). These approaches primarily difer in what is stored and how the state is updated. For KV eviction, however, there is a separate question: how should the compressed history be read relative to the exact entries that remain?

To make this distinction precise, consider a query � and partition its original KV cache into retained entries � and evicted entries �. Let

$$
Z _ { A } = \sum _ { j \in A } \exp ( \ell _ { j } ) ,
$$

$$
o _ { A } = \frac { 1 } { Z _ { A } } \sum _ { j \in A } \exp ( \ell _ { j } ) v _ { j } , \qquad A \in \{ S , E \} ,\tag{16}
$$

$$
\rho = \frac { Z _ { E } } { Z _ { S } + Z _ { E } } ,
$$

$$
\ell _ { j } = q k _ { j } ^ { \top } / \sqrt { d } .\tag{17}
$$

The dense-attention output can then be written as

$$
o _ { \mathrm { f u l l } } = ( 1 - \rho ) o _ { S } + \rho o _ { E } , \qquad o _ { \mathrm { e v i c t } } = o _ { S } ,\tag{18}
$$

and hence

$$
o _ { \mathrm { f u l l } } - o _ { \mathrm { e v i c t } } = \rho \left( o _ { E } - o _ { S } \right) .\tag{19}
$$

Recovering evicted information therefore requires two distinct quantities: the content or direction $O _ { E }$ contributed by the evicted entries, and their mass $\rho$ relative to the retained cache. A memory with high nominal capacity can still fail if its readout does not recover this relative mass.

Output-space residual recovery. A common design reads a compressed state independently of attention and fuses the resulting vector with the exact-attention output:

$$
\begin{array} { r } { m _ { E } ( q ) = \operatorname { R e a d } ( q , M _ { E } ) , \qquad o _ { \mathrm { o u t e r } } = \alpha ( q ) o _ { S } + \beta ( q ) m _ { E } ( q ) . } \end{array}\tag{20}
$$

Infini-attention accumulates completed segments in a normalized associative state and combines its readout with segment-local softmax attention through a learned scalar gate (Munkhdalai et al., 2024). RAttention maintains exact SWA over recent tokens and a delayed recurrent linear state over tokens outside the window, but independently RMS-normalizes and adds the two outputs (Wang et al., 2025):

$$
o _ { t } = \mathrm { R M S } ( o _ { t } ^ { \mathrm { s w a } } ) + \mathrm { R M S } ( o _ { t } ^ { \mathrm { r l a } } ) .\tag{21}
$$

Tensor Cache writes each FIFO-evicted KV pair into an outer-product fast-weight state,

$$
A  \lambda A + \eta ( k _ { \mathrm { o l d } } \otimes v _ { \mathrm { o l d } } ) , \qquad m _ { t } = q _ { t } A ,\tag{22}
$$

and fuses it with local attention using $o _ { t } = o _ { t } ^ { \mathrm { l o c a l } } + \sigma ( g ) m _ { t }$ (Swain et al., 2026). The gated-memory variant of Titans and related TTT memories provide a more expressive state, but use the same general output-space interface (Sun et al., 2024a; Behrouz et al., 2024).

These constructions can learn useful long-term corrections, especially when trained end to end. Nevertheless, the normalization required by Eq. 18 is not enforced by their architecture. Even if $m _ { E } ( q ) = o _ { E }$ exactly, recovering dense attention requires $\alpha ( q ) = 1 - \rho ( q )$ and $\beta ( q ) = \rho ( q )$ . A fixed learned gate, an independently normalized branch, or an additive residual does not obtain these coeficients from $Z _ { S }$ and $Z _ { E } ;$ it must instead learn their calibration indirectly in output space. This distinction is particularly consequential when retrofitting a frozen backbone, whose original attention layers already assign a precise probabilistic meaning to their softmax mass.

Attention-space recovery. An alternative is to expose the compressed history as additional KV entries. Figure 4 contrasts this readout with the output-space memories above. Native Hybrid Attention maintains recurrent long-term KV slots and concatenates them with a local window before a single softmax (Du et al., 2025). Key-Value Means merges overflow blocks into a fixed or growing set of compressed KV rows and likewise attends jointly to the compressed state and the uncompressed block window, with learned regional temperatures (Goldstein et al., 2026). CentroidKV periodically merges similar cache entries into degree-weighted centroids and corrects each centroid logit by the logarithm of its cluster degree (Hu et al., 2025). Unlike output-space fusion, these methods allow compressed and exact entries to compete directly for attention mass.

Concretely, let $\mathcal { C } _ { r } \subseteq E$ be a content cluster, $n _ { r } = | \mathcal { C } _ { r } |$ , and $( \bar { k } _ { r } , \bar { v } _ { r } )$ its centroid. The contribution of $E$ can be approximated by

$$
\widehat { Z } _ { E } ( q ) = \sum _ { r = 1 } ^ { R } \exp \left( q \bar { k } _ { r } ^ { \top } / \sqrt { d } + \log n _ { r } \right) ,\tag{23}
$$

$$
\widehat { N } _ { E } ( q ) = \sum _ { r = 1 } ^ { R } \exp \left( q \bar { k } _ { r } ^ { \top } / \sqrt { d } + \log n _ { r } \right) \bar { v } _ { r } .\tag{24}
$$

Appending these centroids to the retained cache gives

$$
\begin{array} { r } { o _ { \mathrm { j o i n t } } = \frac { Z _ { S } o _ { S } + \widehat { N } _ { E } } { Z _ { S } + \widehat { Z } _ { E } } } \\ { = ( 1 - \widehat { \rho } ) o _ { S } + \widehat { \rho } \widehat { o } _ { E } , \qquad \widehat { \rho } = \frac { \widehat { Z } _ { E } } { Z _ { S } + \widehat { Z } _ { E } } , \quad \widehat { o } _ { E } = \frac { \widehat { N } _ { E } } { \widehat { Z } _ { E } } . } \end{array}\tag{25}
$$

(26)

Thus, the centroid values approximate the missing direction $O _ { E } ,$ while their logits and cluster-size corrections estimate the missing mass $\rho .$ Both are reconciled by the same softmax that weights the retained entries. When a query is unrelated to the compressed history, the centroids receive negligible mass; when it matches that history, they take probability mass away from the retained entries rather than adding an independently scaled residual.

IndexMem++ belongs to the attention-space family, but difers in the source of the exact and compressed sets. Local-memory architectures partition history by age, while cache-merging methods determine the compressed set primarily from geometric similarity. We first learn a query-independent redundancy score that determines which entries must remain individually addressable, and only then summarize the rejected set � by content. Selection and recovery are therefore decoupled: the learned gate decides what must remain exact, whereas the centroids approximate the residual contribution of everything else under a fixed total support budget $| S | + R .$ . Section 3 shows that enlarging an externally read linear or TTT memory provides little benefit, while moving a comparable summary into the shared softmax substantially improves recovery. This indicates that, in this setting, the limiting factor is not merely memory capacity but the normalization through which that capacity is accessed.

Table 11: Representative mechanisms for retaining information that is no longer stored as individual exact KV entries. “Joint” denotes a single softmax over the exact and compressed supports.
<table><tr><td>Method</td><td>Exact support</td><td>Compressed history and write Readout policy</td><td></td><td>Joint</td></tr><tr><td>Infini-attention</td><td>Current segment</td><td>pleted segments</td><td>Associative state over com- Learned convex gate between No two outputs</td><td></td></tr><tr><td>RAttention</td><td>Recent SWA window</td><td>Linear state read only for out- Independently of-window tokens</td><td>normalized sum</td><td>RMS- No</td></tr><tr><td>Tensor Cache</td><td>Recent SWA window</td><td>FIFO eviction</td><td>Fast-weight state updated on Scalar-gated additive residual</td><td>No</td></tr><tr><td>NHA</td><td>Recent SWA window</td><td>Fixed recurrent KV slots</td><td>Concatenate slots and local KV Yes</td><td></td></tr><tr><td>Key-Value Means Recent block window</td><td></td><td>compressed KV rows</td><td>Overflow tokens merged into Single softmax with regional Yes temperatures</td><td></td></tr><tr><td>CentroidKV</td><td></td><td>degree-weighted centroids</td><td>Sink, recent, and unmerged KV Similar KV entries merged into Single softmax with log nr bias Yes</td><td></td></tr><tr><td>IndexMem+ +Learned retained set</td><td></td><td>only from learned-evicted en- bias tries</td><td>Content centroids formed Single softmax with log nr Yes</td><td></td></tr></table>

## B. Training-time kernel for gated attention

For each layer and KV head, the scorer maps the contextualized hidden state of key � directly to a retention score $s _ { j } = f _ { \theta } ( h _ { j } )$ . We suppress the head index for readability. If the retention-rate term in Equation 6 is enabled, $s _ { j }$ is replaced by its age-dependent value for query �. Let $P _ { i }$ denote the always-attended sinks and causal local window, and let $E _ { i }$ contain the remaining visible keys. The gate added to the attention logits is

$$
\bar { g } _ { i j } = \left\{ \begin{array} { l l } { \displaystyle 0 , } & { \displaystyle j \in P _ { i } , } \\ { \displaystyle s _ { j } - \log \sum _ { \ell \in E _ { i } } \exp ( s _ { \ell } ) + \log \beta , } & { \displaystyle j \in E _ { i } , } \end{array} \right. \quad \quad o _ { i } = \sum _ { j } \mathrm { s o f t m a x } _ { j } \left( \frac { q _ { i } k _ { j } ^ { \top } } { \sqrt { d } } + \bar { g } _ { i j } \right) v _ { j } .\tag{27}
$$

This is the training-time form of Equation $^ { 4 , }$ written additively in logit space: $\bar { g } _ { i j } = \log g _ { i j }$ , so protected entries remain unchanged, while the other entries compete for total gate mass $\beta$ . The full causal key axis is used during training so that every score receives a gradient.

Why a standard implementation runs out of memory. Although $s _ { j }$ is computed only once per key, the efective gate $\bar { g } _ { i j }$ depends on the query row because each query has a diferent causal local window $P _ { i }$ . A direct implementation therefore broadcasts the per-key scores into a pairwise gate matrix and constructs gated logits of shape $( B , H , S _ { q } , S _ { k } )$ . At $S _ { q } = S _ { k } = 8 1 9 2$ with 32 query heads, one bf16 matrix occupies about 4 GiB per layer; retaining these activations for backward across 36 layers requires approximately 144 GiB and causes OOM.

Using a standard attention call with an explicit pairwise gate does not avoid this cost. The always-attended mask prevents the gate from being represented as one static per-key bias, and the resulting SDPA call falls back to the math backend, which materializes the full attention matrix. We therefore implement the operation directly as a fused training kernel.

Fused forward and backward. Code 1 shows the core implementation. The MLP computes one $s _ { j }$ per key outside the kernel, and a streaming reduction computes the row-wise gate normalizer. The forward kernel loads these values with each key tile, constructs the sink/local mask arithmetically, and combines the normalized gate with an online attention softmax. The backward kernel recomputes the same tiled logits and accumulates the gradient of each $s _ { j }$ over the queries that attended to it; ordinary autograd then propagates this gradient through the MLP.

```julia
g = MLP(h) # [batch, H_kv, length]
lse = streaming_logsumexp(g, evictable, beta)
# One Triton program owns a query tile and streams key tiles.
def forward(Q, K, V, g, lse):
row_max = -inf
row_sum = 0
out_acc = 0
for key_tile in key_tiles:
q = load(Q, query_tile)
k = load(K, key_tile)
v = load(V, key_tile)
g_key = load(g, key_tile)
logits = q @ k.T / sqrt(d)
protected = is_sink(key_tile) | is_local(query_tile, key_tile)
gate = where(protected, 0, g_key - lse[:, None])
logits = where(causal(query_tile, key_tile), logits + gate, -inf)
# Online softmax; no pairwise gate or attention matrix is stored.
new_max = max(row_max, rowmax(logits))
rescale = exp(row_max - new_max)
p = exp(logits - new_max[:, None])
out_acc = out_acc * rescale[:, None] + p @ v
row_sum = row_sum * rescale + rowsum(p)
row_max = new_max
out = out_acc / row_sum[:, None]
row_lse = row_max + log(row_sum) # saved for backward
return out, row_lse
def backward(Q, K, V, g, lse, out, row_lse, d_out):
delta = rowsum(out * d_out)
dQ, dK, dV, dg, d_lse = zeros()
for key_tile in key_tiles:
# Recompute the same gated logits instead of saving O(L^2) activations.
logits, protected = recompute_logits(Q, K, g, lse, key_tile)
p = exp(logits - row_lse[:, None])
atomic_add(dV[key_tile], p.T @ d_out)
dP = d_out @ V[key_tile].T
dZ = p * (dP - delta[:, None])
dQ += dZ @ K[key_tile] / sqrt(d)
atomic_add(dK[key_tile], dZ.T @ Q / sqrt(d))
d_gate = where(protected, 0, dZ)
atomic_add(dg[key_tile], rowsum_over_queries(d_gate))
d_lse += -rowsum_over_keys(d_gate)
return dQ, dK, dV, dg, d_lse
```

Listing 1: Core training-time gated-attention kernel.

Neither pass materializes the pairwise gate, logits, or attention probabilities. The saved activation memory is therefore $O ( S _ { q } + S _ { k } )$ rather than $O ( S _ { q } S _ { k } )$ , although the arithmetic remains quadratic because every visible key receives an independent training signal. Softmax and gradient accumulation use fp32 for numerical stability.

A causal row can initially contain only always-attended entries, leaving $E _ { i }$ empty and its log-sum-exp undefined. Such rows use an inert gate, which preserves dense attention and prevents NaNs without suppressing any meaningful retention signal.

## C. From the soft gate to hard eviction

At inference, the learned gate is used only to score cache entries, while attention is computed normally over the retained entries. Our implementation uses sparse attention to simulate eviction during prefill, then converts the final prefill state into a physically compressed cache for streaming decode.

Sparse prefill. During prefill, each query attends only to its sinks, local window, and highest-scoring visible entries. We evaluate the retention score once for the whole prefill and hold that score vector fixed while the mask is built. Against a fixed score vector the candidate pool of successive queries only grows, so the number of entries outranking a given key is non-decreasing and each key is selected by one contiguous interval of query rows. Each key’s support is then summarized by the last position at which it survives, and the mask is expressed as a FlexAttention block mask, avoiding a per-query top- $B _ { h }$ support tensor and indirect KV gathers. This reproduces per-row top- $B _ { h }$ selection over that score vector exactly; with the per-key retention rate enabled, holding the vector fixed is an additional prefill approximation, since a query midway through the prefill would otherwise rank entries by their scores at its own position. The full KV cache is retained until prefill finishes, so this stage reduces attention computation but not cache storage.

Streaming eviction. After prefill, we copy the final row’s retained entries into the compressed cache and release the dense cache. During decoding, each new token first enters the local window; the entry leaving the window then competes with the weakest entry in the retained pool, so at most one cache slot is replaced per head and step and the cache size remains fixed. This update is exactly equivalent to global top- $B _ { h }$ selection when the ordering is fixed; with learned per-key decay, it is an irreversible online approximation because deleted entries cannot be reconsidered after their scores cross.

Ragged paged cache. The head allocator assigns a diferent budget $B _ { h }$ to each KV head, so padding all heads to the largest budget would waste much of the saved memory. We instead store each (layer, sequence, KV head) row in its own pages. Decode then performs dense paged GQA directly over the retained entries, without a mask or gather, and both cache storage and historical KV reads scale with $\textstyle \sum _ { h } B _ { h }$ rather than the original context length, as reflected in Figure 5.

## D. Eficiency

Measurement setup. Figure $^ { 5 , }$ summarized in Section 4.5, reports aggregate batch decode throughput (token/s) on one NVIDIA H20 GPU for Qwen3-4B and Qwen3-8B. Input lengths are 8K, 16K, 32K, 64K, 128K, and 256K (where 1K denotes 1,024 tokens), with batch sizes $B \in \{ 1 , 4 , 1 6 , 3 2 \}$ , giving 48 configurations. Every configuration decodes 256 steps in BF16, and prefill, initial compression, graph capture, and one-time setup are excluded from the timing. IndexMem++ decodes through a captured CUDA Graph on deterministic pseudo-random token inputs and we report the fastest of three repetitions; its average per-KV-head budget is 2,048 entries, inclusive of the 64 CMP slots, the four sink tokens, and the 128-token local window, under the mass-based head allocation of Table 12. The Full-KV\* reference decodes eagerly with FlashAttention-2 on GovReport inputs, aggregated over 15 measurements. The star marks this diference in decode implementation: part of any gap between the two curves reflects graph replay rather than the retained-cache budget, so we read the figure for how each curve scales with input length and treat cross-curve ratios as indicative only.

Context-length scaling. A fixed retained KV budget largely decouples decode throughput from the original input length. From 8K to 256K, IndexMem++ throughput at � = 1 changes from 53.95 to 52.18 token/s for Qwen3-4B and from 48.38 to 46.51 token/s for Qwen3-8B, while the Full-KV\* reference measurements fall from 36.85 to 8.42 and from 33.88 to 8.21, respectively. At � = 4, IndexMem++ changes from 173.69 to 174.69 and from 166.96 to 155.95 token/s for the two models. The � = 16 and � = 32 curves decline more, but IndexMem++ remains substantially less sensitive to the original context length than the Full-KV\* reference curves. This diference is consistent with the execution model in Appendix C: continuous eviction bounds the historical KV read by IndexMem++ to the retained cache budget, whereas Full-KV reads a historical cache that grows with sequence length.

Batch-size scaling. Within the IndexMem++ measurements, throughput increases substantially with batch size while each sequence retains the same bounded historical-cache budget: at 16K it rises from 53.13 to 685.72 token/s for Qwen3-4B and from 47.19 to 630.07 token/s for Qwen3-8B as � goes from 1 to 32. The Full-KV\* reference is comparatively flat in batch size at this length, since each added sequence brings its own growing history to read. The two efects compound with the implementation diference noted above, so we do not attribute a specific ratio to the budget alone. In our implementation, scoring and cache maintenance are also batched across sequences and layers.

Benchmark coverage and cache footprint. Figure 5 reports IndexMem++ decode throughput for all 48 model–length–batch configurations. The Full-KV\* reference contains 32 measured configurations and 16 OOM cases. For both models, Full-KV\* fails from 32K onward at � = 32, from 64K onward at $B = 1 6 ,$ and at 256K for $B = 4 ,$ , while $B = 1$ succeeds through 256K. The ragged paged cache in Appendix C allocates physical KV storage according to fixed per-head retained budgets, so retained-cache storage scales with $\sum _ { h } B _ { h }$ per layer and sequence rather than with the original context length. Inputs of 64K–256K exceed the 40,960-position configuration used for quality evaluation and serve as throughput and memory stress tests, not long-context accuracy results.

## E. Details deferred from the method

## E.1. Both halves of the always-attended construction are required.

Normalizing all gates without exempting any entry is inefective because the row-wise normalizer adds the same constant to every attention logit and therefore cancels inside the softmax. Exempting sinks and the local window alone is also insuficient: the remaining gates can all approach 1 and exactly recover dense attention. Equation 4 combines the two requirements. Protected entries remain fixed at one, while the evictable entries must share a total gate mass $\beta ;$ they therefore cannot all equal one and must compete with one another. This removes the dense no-op solution and turns $\beta$ into an exact training-time budget rather than a target imposed through a penalty.

What the scorer learns at the constant point. The construction does not forbid a constant score, and it does not need to. Substituting $s _ { j } \equiv c$ into Equation 4 gives $g _ { i j } = \beta / | E _ { i } |$ regardless of $c { : }$ a single policy that attenuates every evictable entry equally and is dense only if $\beta = | E _ { i } |$ . Because that policy is not dense, the loss gradient there is nonzero. Writing $r _ { i j } = \langle \partial \mathcal { L } / \partial o _ { i } , v _ { j } \rangle$ for the sensitivity of the loss to entry � through the output of query �, and $\begin{array} { r } { \bar { r } _ { i } = \sum _ { k } a _ { i k } r _ { i k } } \end{array}$ for its attention-weighted mean, the gradient with respect to a score is

$$
\frac { \partial \mathcal { L } } { \partial s _ { m } } = \sum _ { i : m \in E _ { i } } \Big ( D _ { i m } - p _ { i m } ^ { ( s ) } \sum _ { j \in E _ { i } } D _ { i j } \Big ) , \qquad D _ { i j } = a _ { i j } \big ( r _ { i j } - \bar { r } _ { i } \big ) , \qquad p _ { i m } ^ { ( s ) } = \frac { \exp \big ( s _ { m } \big ) } { \sum _ { \ell \in E _ { i } } \exp \big ( s _ { \ell } \big ) } .\tag{28}
$$

Each score thus receives the advantage of its own entry over the current evictable average: an entry contributing more than average gains gate mass, and it gains it at the expense of the others. Competition for the limited mass, rather than the unavailability of a constant score, is what drives the scorer to rank entries.

Two further consequences are worth stating. First, the separation from dense attention is graded rather than binary: the constant solution attenuates evictable entries by $\beta / | E _ { i } |$ relative to the protected ones, so the pressure away from the no-op grows with $\log ( | E _ { i } | / \beta )$ and disappears as $\beta \to | E _ { i } |$ . The requirement is therefore $\beta \ll | E _ { i } |$ rather than merely $\beta < | E _ { i } | ;$ ; in the pilot setting $\beta / | E _ { i } | \approx 2 5 6 / 8 0 6 4$ , roughly 3%. This is also why $\beta$ is held fixed rather than scaled with the sequence length, since a $\beta$ growing with $| E _ { i } |$ would weaken the mechanism at exactly the lengths that need it. Second, summing Equation 28 over a uniform shift of all scores within a head gives exactly zero, because $\begin{array} { r } { \sum _ { m \in E _ { i } } ( \delta _ { j m } - p _ { i m } ^ { ( s ) } ) = 0 } \end{array}$ . The absolute score level is therefore unidentifiable, which is the same degeneracy that prevents raw scores from being compared across heads and motivates the retained-mass allocator.

## E.2. Loss design.

Forward $\mathrm { K L } ( p ^ { \mathrm { d e n s e } } \Vert p ^ { \mathrm { g a t e d } } )$ is mass-covering, or mean-seeking: it encourages the gated model to cover the full teacher distribution, including its lower-probability alternatives. Reverse KL $\cdot ( p ^ { \mathrm { g a t e d } } \| p ^ { \mathrm { d e n s e } } )$ is mode-seeking: it concentrates the student on the teacher’s highest-probability modes and penalizes probability assigned outside them.

The two directions agree to first order. Writing $q = p ^ { \mathrm { g a t e d } }$ and $p = p ^ { \mathrm { d e n s e } }$ for one position and diferentiating with respect to the gated logits �,

$$
\frac { \partial \mathrm { K L } ( p | | q ) } { \partial z _ { k } } = q _ { k } - p _ { k } , \qquad \frac { \partial \mathrm { K L } ( q | | p ) } { \partial z _ { k } } = q _ { k } ( r _ { k } - \bar { r } ) , \qquad r _ { k } = \log \frac { q _ { k } } { p _ { k } } , \quad \bar { r } = \sum _ { j } q _ { j } r _ { j } .\tag{29}
$$

The forward direction reproduces the first term of Equation 5 exactly. The reverse direction is not a reweighting of it, since components with $0 < r _ { k } < \bar { r }$ receive a gradient of the opposite sign, but the two coincide near the matching point: substituting � = � + � gives $r _ { k } = \epsilon _ { k } / p _ { k } + O ( \lVert \epsilon \rVert ^ { 2 } )$ and $\bar { r } = O ( \| \epsilon \| ^ { 2 } )$ , because $\begin{array} { r } { \sum _ { k } \epsilon _ { k } = 0 _ { : } } \end{array}$ so $q _ { k } ( r _ { k } - { \bar { r } } ) = \epsilon _ { k } + O ( \| \epsilon \| ^ { 2 } )$ . Both directions therefore share the same leading-order gradient and difer only in higher-order terms, which is where the mass-covering and mode-seeking behaviors originate. Neither direction contains the frozen-backbone term of Equation $^ { 5 , }$ so the choice between them is not a question of which error is isolated and we make it empirically.

We also add cross-entropy with weight 0.1, chosen to place its contribution on approximately the same numerical scale as the KL term; this changes the average from 67.60 to 67.33. Adding this term reintroduces the frozen-backbone error $p _ { t } ^ { \mathrm { d e n s e } } - e _ { y _ { t } }$ into the gradient, and the result does not improve, so the final recipe uses no cross-entropy term and the language-modeling target provides no additional benefit in this setting.

We try LongCE because it emphasizes tokens whose prediction benefits from long context, which should focus training on positions most afected by KV eviction. However, LongCE only reweights the cross-entropy gradient: it scales the error caused by eviction and the irreducible error of the frozen dense model together, rather than removing the latter as distillation does. It therefore improves CE only from 51.50 to 53.01, remaining far below reverse KL at 67.60 (Table 2).

In future work we plan to do on-policy KL distillation, where training prefixes are generated by the current hard-eviction policy rather than by dense or teacher-forced trajectories. This would expose the gate to the state distribution and accumulated errors it encounters at inference, thereby reducing the train–inference mismatch. It is also attractive for reinforcement-learning pipelines, whose rollouts maintain many long-lived KV caches: applying eviction during policy generation bounds the rollout cache by the retained budget and can reduce the memory and bandwidth cost of sampling, although dense teacher or reference scoring may still require a separate full-cache pass.

## E.3. Heads budget design.

Adding a constant � to all retention scores of one head shifts its normalizer by the same amount, leaving Equation 4 and the within-head top-� ordering unchanged. The absolute score level is therefore not identifiable across heads, so pooling raw scores across heads cannot determine their relative budgets. Retained attention mass is invariant to this shift and provides a comparable measure of demand. For the final prefill query, we order each head’s evictable entries by its retention score and define $\rho _ { h } ( k )$ as the attention mass retained by the first � entries together with the protected sinks and window; query heads sharing one KV head are averaged. We choose the smallest $k _ { h }$ for which $\rho _ { h } ( k _ { h } ) \geq \rho ^ { \star }$ and adjust the shared target $\rho ^ { \star }$ until the total budget is met. Any integer remainder is assigned by marginal mass: an extra slot goes to the head whose next entry adds the most attention mass, or is removed from the head whose last entry contributes the least. This is a discrete water-filling allocation that preserves the layer budget exactly. Under a uniform split, the retained-mass range across heads is 0.268 on average and reaches 0.606 in layer 4, leaving substantial room for reallocation.

Both controls preserve exactly the same total number of KV entries. The permuted control shufles the online allocator’s budget vector across heads, preserving its ragged shape while breaking the correspondence between each head and its measured demand. Its score falls to 57.14, below the uniform allocation at 58.86, showing that unequal budgets alone do not explain the improvement. The ofline control averages per-head allocations over a calibration set and uses the resulting fixed layer–head table for every input. It reaches 62.33, recovering part of the gain without online measurement and showing that some head specialization is stable across inputs. Its remaining gap to the context-dependent allocator at 67.60 shows that the appropriate allocation also depends on the current context (Table 2).

Code 2 summarizes the online allocation.

# Scores rank entries within a head; attention mass allocates across heads.   
def mass\_curve(Q, K, score, ref\_row, sinks, window):   
logits = Q[:, ref\_row] @ K.T / sqrt(d)   
prob = softmax(mask\_future(logits, ref\_row), dim=-1)   
prob = mean\_over\_gqa\_group(prob) # [H\_kv, length]   
pinned = is\_sink(sinks) | is\_local(ref\_row, window)   
pool = is\_causal(ref\_row) & \~pinned   
pool\_size = sum(pool)   
order = argsort(where(pool, score, -inf), descending=True, stable=True)   
pinned\_mass = rowsum(where(pinned, prob, 0))   
ranked\_mass = gather(prob, order[:, :pool\_size])   
return pinned\_mass[:, None] + cumsum(ranked\_mass, dim=-1)   
def allocate\_by\_mass(curve, total\_slots):   
H\_kv, pool\_size = curve.shape   
# Fewest slots with retained mass at least rho.   
def counts(rho):   
reached = curve >= rho   
return where(any(reached, dim=-1), first\_true(reached) + 1, pool\_size)   
# Find a common retained-mass target.   
lo, hi = 0.0, 1.0   
for \_ in range(60):   
rho = (lo + hi) / 2   
if sum(counts(rho)) > total\_slots: hi = rho   
else: lo = rho   
k = counts(lo)   
# Integer correction keeps the layer budget exact.   
while sum(k) < total\_slots: gain = mass\_of\_next\_entry(curve, k), k[argmax(gain)] += 1   
while sum(k) > total\_slots: loss = mass\_of\_last\_entry(curve, k), k[argmin(loss)] -= 1   
return k   
def head\_budgets(Q, K, score, topk, sinks, window):   
curve = mass\_curve(Q, K, score, last\_prefill\_row, sinks, window)   
H\_kv = curve.shape[0]   
evictable = topk - sinks - window   
k = allocate\_by\_mass(curve, H\_kv \* evictable)   
return k + sinks + window  
Listing 2: Core retained-mass head-budget allocation.  
E.4. The three history scorers, and why each is a single-variable test.

All three variants retain the token-local MLP and add one history branch before its output nonlinearity. Writing $x _ { j } = \mathrm { N o r m } ( h _ { j } )$ and $a _ { j }$ for the history feature, the shared form is $z _ { j } = \mathrm { G E L U } ( \mathrm { N o r m } ( W _ { \mathrm { i n } } x _ { j } + W _ { a }$ Norm $( a _ { j } ) ) )$ , followed by the same per-head score projection as the reference scorer. We initialize $W _ { a }$ to zero, so every variant begins exactly from the token-local MLP and difers only in the construction of $a _ { j }$ . Figure 3 shows the five scorers side by side, including the linear readout and the token-local MLP used as references.

Causal convolution. We project the normalized hidden states to $u _ { j } = W _ { c } x _ { j }$ and apply a width-eight depthwise causal convolution over $u _ { j - 8 } , \dotsc , u _ { j - 1 }$ . The current token is excluded because its information is already carried by the MLP branch. Each channel has its own signed temporal filter, allowing the scorer to compare a token with its recent neighborhood while requiring only the last eight projected states during decoding.

Gated recurrent state. We maintain a per-channel state $S _ { j } = g _ { j } \odot S _ { j - 1 } + ( 1 - g _ { j } ) \odot u _ { j }$ , where $u _ { j } = W _ { u } x _ { j }$ and $g _ { j } = \sigma ( W _ { g } x _ { j } )$ . The score of token � reads $S _ { j - 1 }$ rather than $S _ { j } { \mathrm { : } }$ , so this branch contains only information from preceding tokens. The input-dependent gate lets diferent channels retain or replace information over diferent time scales. During training the recurrence is evaluated in fp32 with an associative scan, while decoding carries one fixed-size state vector per layer.

Prefix attention. We give the scorer a separate single-head causal attention with $q _ { j } = W _ { q } x _ { j } , k _ { i } = W _ { k } x _ { i } ,$ $v _ { i } = W _ { v } x _ { i }$ , and $a _ { j } = \mathrm { s o f t m a x } ( q _ { j } K _ { < j } ^ { \top } / \sqrt { D } ) V _ { \leq j }$ . This is still query-independent for eviction: $q _ { j }$ is formed from the hidden state of key token $j ,$ rather than from a later model query, so the score is fixed when the key is created. It provides access to the entire prefix, but incremental scoring requires retaining the scorer’s own K/V cache and attending over a history that grows with the sequence.

The convolution, recurrent, and prefix-attention scorers use 16.8M, 24.1M, and 19.5M parameters, respectively, compared with 7.5M for the token-local MLP, yet change the RULER average by on $\mathrm { 1 y - 0 . 7 8 , + 1 . 1 6 } .$ and −0.97 points (Table 2). A Fast-KVzip-style adaptive-reference head is likewise within 0.35 points. We therefore conclude that suficient scorer capacity is necessary at this model scale, but explicitly re-reading the history provides little benefit because $h _ { j }$ is already contextualized.

## E.5. Retention-rate

Equation 6 uses a fixed age scale � rather than the current sequence length. Without normalization, $\partial s _ { j } ( i ) / \partial \log \beta _ { j } =$ $i - j$ can be $1 0 ^ { 4 }$ of magnitude larger than $\partial s _ { j } ( i ) / \partial m _ { j } = 1$ at long context lengths, making the lifetime head efectively train at a much larger learning rate than the magnitude head. Dividing by a fixed � keeps the two derivatives comparable and gives log $\beta _ { j }$ a consistent interpretation as the score decrease over � tokens. Using the live sequence length instead would continually rescale the age contribution of every cached key as the context grows, introducing changes in which entries are retained that are caused by the changing denominator rather than by elapsed age; in a positional-score diagnostic, the fixed scale produced no re-entries over 1,500 steps, whereas live-length normalization produced 27.

Retention-rate initialization. We initialize log $\beta _ { j }$ so that its change over one reference interval is comparable to the standard deviation of the magnitude score. This difers from TrimKV, where decay starts near zero and is driven by a separate retention hinge. Our fixed-mass normalizer enforces a budget but does not itself encourage shorter lifetimes, so a near-zero initialization could leave the lifetime branch inactive and reduce the model to the content-only scorer. The retention rate does not provide a consistent accuracy gain: its efect is approximately neutral under a uniform budget and reverses under the head-budget allocator. We therefore retain it for a stable, explicitly parameterized age scale rather than as an accuracy-improving component.

## F. Pilot ablation protocol

Protocol. Every arm in Tables 1, 3 uses Qwen3-0.6B and RULER at an 8K context length. Eviction evaluation retains 25% of the context. Unless explicitly stated otherwise, evaluation uses seed 42. Each gate is trained for 100 steps at 8K with a gate-mass budget of 256, peak learning rate $1 0 ^ { - 3 }$ , and data seed 1000.

Two tasks carry no signal at this model size. At this scale, cwe remains between 0.00 and 0.65 for every arm, indicating that Qwen3-0.6B cannot solve the task under this protocol. multikey3 ranges from 0.00 to 47.83 across arms, but two seeds of the same reference configuration score 26.09 and 25.40; its apparent between-arm variation is therefore not cleanly separable from sampling variation. Both tasks remain in Avg. because they belong to the standard 13-task RULER suite, but no claim in Section 3 relies on either individual column.

## G. IndexMem++ implementation settings

This section gives the complete training recipe and the inference-time configuration promised by Section 4.1.

Training data. We train on LongMino, using its 256K-filtered split in pre-tokenized form and drawing from the 2e15, 2e16, synth\_cwe, and synth\_rex subsets. The DMA baseline uses the same mixture. Its value-derived, query-independent score is trained with the LM loss as an in-softmax gate and used for hard eviction. DMA is trained for 300 optimizer steps at 8K only, with global batch 8 and the backbone frozen. Documents are drawn at random with a shufle bufer of 64 and a data seed of 1000.

Length curriculum and optimization. The gate is trained under a length curriculum of 300 steps at 8K, 300 steps at 16K, with global batch of 8. We use AdamW with a peak learning rate of $1 0 ^ { - 3 }$ decaying to $5 \times 1 0 ^ { - 6 }$ warming up over the first 10% of steps and holding the peak for the following 60%. Only the gate parameters are placed in the optimizer; the backbone, the embeddings, and the LM head remain frozen throughout, and Section 3 reports what unfreezing them costs.

Gate configuration. The scorer is a two-layer MLP applied to the hidden state $h _ { j }$ with a middle width of 256, emitting one score per KV head. The per-key retention rate of Equation 6 is enabled. Four attention sinks and a 128-token causal window are always attended; these entries are exempt from eviction and from the gate normalizer. The gate mass in Equation 4 is $\beta = 2 5 6$

Inference-time components. Two components are applied only at inference and introduce no trained parameters. The head-budget allocator redistributes a fixed layer-wide support budget across KV heads by retained attention mass. Let $B _ { h }$ denote the total support assigned to KV head ℎ. Sink entries, local-window entries, and the centroid slots are funded within $B _ { h } ,$ , rather than added on top. Under mass allocation, the layer-wide budget satisfies $\begin{array} { r } { \sum _ { h } B _ { h } = H _ { \mathrm { K V } } K } \end{array}$ , where � is the average per-KV-head support used for the reported compression setting.

The allocator is subject to a per-head floor, which we set to $K / 4$ rather than to an absolute number of entries. A floor fixed in absolute terms would interact with the budget: since the layer total is exactly $H _ { \mathrm { K V } } K$ and every head must reach the floor, a floor equal to � would force every head to exactly � and reduce the allocator to a uniform split, and a floor close to � would leave it little room to reallocate. Tying the floor to � keeps the same fraction of the budget available for redistribution at every compression setting, so the allocator behaves consistently across the budgets reported in this paper. At the RULER setting $K = 2 0 4 8$ this gives a floor of 512 entries.

Per-benchmark inference configuration. Table 12 lists the inference-time configuration of IndexMem++ on each benchmark. The trained gate and the always-attended regions are the same everywhere, and the budget is counted the same way. RULER and LongBench difer only in the budget � implied by the compression ratio. On Math500 and AIME25 the prompt is short and the cache is filled almost entirely during generation, so the mass allocator, which reads the attention at the end of prefill, has almost nothing to work from. There IndexMem++ uses a static head table fitted ofline for each budget, with the same $K / 4$ floor, in place of the per-document allocator, and no centroid slots. Each table is fitted once by running the per-document allocator on 16 LongMino documents of 8K tokens and averaging the resulting per-head budgets.

Table 12: IndexMem++ inference configuration per benchmark. One trained gate is used throughout; the budget � is the average per-KV-head total support and is inclusive of every protected entry and centroid slot listed here.
<table><tr><td>Benchmark</td><td>Budget K</td><td>Allocator</td><td>Floor</td><td>Sink / local</td><td>CMP</td></tr><tr><td>RULER</td><td>[(1 − c)N]</td><td>mass</td><td>K/4</td><td>4/128</td><td>64</td></tr><tr><td>LongBench</td><td> $\lfloor ( 1 - c ) N \rfloor$ </td><td>mass</td><td> $K / 4$ </td><td>4/128</td><td>64</td></tr><tr><td>Math500</td><td> $5 1 2 , 1 , 0 2 4$ </td><td>static</td><td>K/4</td><td>4/128</td><td>0</td></tr><tr><td>AIME25</td><td>1,024, 2,048</td><td>static</td><td>K/4</td><td>4/128</td><td>0</td></tr></table>

## H. Additional Long-Context Results

## H.1. Complete RULER results

The tables below report Full-KV and the displayed methods at compression ratios 50% and 75%, across all 13 RULER tasks. The average includes all 13 tasks.

Table 13: Complete RULER scores: Qwen3-4B, 8K. Compression ratios 50% and 75%.
<table><tr><td>Method</td><td>S1</td><td>S2</td><td>S3</td><td>MK1</td><td>MK2</td><td>MK3</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Full-KV</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.80</td><td>99.80</td><td>99.80</td><td>99.85</td><td>99.70</td><td>99.96</td><td>84.56</td><td>86.27</td><td>75.20</td><td>56.40</td><td>92.41</td></tr><tr><td colspan="9">Compression ratio: 50%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>85.20</td><td>89.00</td><td>5.60</td><td>43.60</td><td>31.60</td><td>28.80</td><td>43.00</td><td>41.05</td><td>78.12</td><td>77.10</td><td>93.27</td><td>45.80</td><td>43.00</td><td>54.24</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>100.00</td><td>98.80</td><td>91.80</td><td>43.00</td><td>11.00</td><td>98.85</td><td>95.80</td><td>99.96</td><td>80.98</td><td>82.80</td><td>34.00</td><td>21.40</td><td>73.72</td></tr><tr><td>DMA</td><td>99.80</td><td>84.80</td><td>29.80</td><td>80.60</td><td>61.60</td><td>0.60</td><td>81.85</td><td>88.60</td><td>96.20</td><td>56.44</td><td>85.80</td><td>70.80</td><td>49.40</td><td>68.18</td></tr><tr><td>TrimKV</td><td>100.00</td><td>99.60</td><td>99.80</td><td>99.80</td><td>99.00</td><td>42.40</td><td>99.65</td><td>99.95</td><td>99.96</td><td>80.88</td><td>77.73</td><td>67.20</td><td>57.00</td><td>86.38</td></tr><tr><td>IndexMem++</td><td>100.00</td><td>100.00 100.00</td><td></td><td>99.60</td><td>99.60</td><td>99.40</td><td>99.80</td><td>99.90</td><td>99.96</td><td>75.20</td><td>86.27</td><td>75.40</td><td>57.60</td><td>91.75</td></tr><tr><td colspan="9">Compression ratio: 75%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>70.20</td><td>22.20</td><td>2.40</td><td>15.40</td><td>8.00</td><td>3.40</td><td>14.05</td><td>13.85</td><td>38.36</td><td>62.18</td><td>89.00</td><td>29.60</td><td>32.40</td><td>30.85</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>88.60</td><td>92.00</td><td>74.60</td><td>4.60</td><td>0.00</td><td>89.90</td><td>89.50</td><td>100.00</td><td>69.14</td><td>74.07</td><td>15.00</td><td>12.60</td><td>62.31</td></tr><tr><td>DMA</td><td>99.00</td><td>24.20</td><td>2.40</td><td>27.40</td><td>5.40</td><td>0.00</td><td>14.30</td><td>20.70</td><td>79.60</td><td>48.16</td><td>71.47</td><td>47.80</td><td>32.00</td><td>36.34</td></tr><tr><td>TrimKV</td><td>100.00</td><td>98.80</td><td>92.60</td><td>95.00</td><td>48.40</td><td>10.80</td><td>91.55</td><td>95.60</td><td>99.96</td><td>77.00</td><td>71.60</td><td>49.60</td><td>49.60</td><td>75.42</td></tr><tr><td>IndexMem++</td><td>100.00</td><td>99.20</td><td>91.20</td><td>97.80</td><td>94.80</td><td>60.40</td><td>98.40</td><td></td><td>99.20 100.00</td><td>44.28</td><td>84.07</td><td>68.00</td><td>53.20</td><td>83.89</td></tr></table>

Task abbreviations follow Table 6.

Table 14: Complete RULER scores: Qwen3-4B, 16K. Compression ratios 50% and 75%.
<table><tr><td>Method</td><td>S1</td><td>S2</td><td>S3</td><td>MK1</td><td>MK2</td><td>MK3</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Full-KV</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.40</td><td>99.00</td><td>98.80</td><td>99.95</td><td>99.95</td><td>99.80</td><td>69.82</td><td>97.00</td><td>72.80</td><td>54.80</td><td>91.64</td></tr><tr><td colspan="10">Compression ratio: 50%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>92.40</td><td>93.80</td><td>13.40</td><td>69.20</td><td>31.40</td><td>21.40</td><td>60.10</td><td>74.35</td><td>94.36</td><td>66.28</td><td>98.27</td><td>43.00</td><td>43.00</td><td>61.61</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>99.80</td><td>99.40</td><td>75.00</td><td>27.80</td><td>3.80</td><td>93.60</td><td>95.80</td><td>99.88</td><td>70.16</td><td>90.73</td><td>29.00</td><td>25.00</td><td>70.00</td></tr><tr><td>DMA</td><td>100.00</td><td>83.40</td><td>32.60</td><td>77.20</td><td>51.60</td><td>0.20</td><td>79.35</td><td>83.75</td><td>92.56</td><td>33.70</td><td>94.07</td><td>65.60</td><td>47.00</td><td>64.69</td></tr><tr><td>TrimKV</td><td>100.00</td><td></td><td>100.00100.00</td><td>98.60</td><td>97.40</td><td>40.40</td><td>99.15</td><td>99.90</td><td>99.80</td><td>63.98</td><td>99.13</td><td>71.60</td><td>53.20</td><td>86.40</td></tr><tr><td>IndexMem++</td><td>100.00</td><td></td><td>100.00 100.00</td><td>99.00</td><td>98.40</td><td>85.20</td><td>99.85</td><td>99.90</td><td>99.68</td><td>52.98</td><td>97.47</td><td>72.60</td><td>54.20</td><td>89.18</td></tr><tr><td colspan="10">Compression ratio: 75%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>72.60</td><td>39.00</td><td>4.40</td><td>19.60</td><td>5.60</td><td>5.20</td><td>14.60</td><td>20.45</td><td>73.40</td><td>51.92</td><td>97.00</td><td>30.80</td><td>30.80</td><td>35.80</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>87.00</td><td>92.60</td><td>65.20</td><td>2.20</td><td>0.00</td><td>91.30</td><td>89.10</td><td>99.48</td><td>65.74</td><td>83.87</td><td>14.00</td><td>15.40</td><td>61.99</td></tr><tr><td>DMA</td><td>98.60</td><td>20.40</td><td>2.40</td><td>17.60</td><td>2.20</td><td>0.00</td><td>14.00</td><td>15.30</td><td>76.96</td><td>30.26</td><td>77.20</td><td>45.00</td><td>33.00</td><td>33.30</td></tr><tr><td>TrimKV</td><td>100.00</td><td>97.40</td><td>92.80</td><td>88.40</td><td>42.40</td><td>13.80</td><td>93.60</td><td>95.55</td><td>98.60</td><td>60.34</td><td>96.67</td><td>65.20</td><td>47.00</td><td>76.29</td></tr><tr><td>IndexMem++</td><td>100.00 100.00</td><td></td><td>98.80</td><td>97.00</td><td>41.80</td><td>18.40</td><td>98.20</td><td>99.40</td><td>99.52</td><td>22.60</td><td>98.13</td><td>72.80</td><td>48.60</td><td>76.56</td></tr></table>

Task abbreviations follow Table 6.

## H.2. Complete LongBench results

Table 17 reports the same 15 tasks and methods as Table 7, with 3,550 examples in the displayed task set. Panel (a) contains document QA and summarization; panel (b) contains the remaining tasks and the overall average. The budget accounting is given below.

RULER budget accounting. The input is processed as a context followed by a question and generation sufix. RULER is evaluated at nominal context lengths of 8K and 16K. Let � denote the actual tokenized context length and � the compression ratio. SnapKV and KeyDif keep ⌊(1 − �)�⌋ entries per KV head. TrimKV sets its memory size to the same fraction (1 − �) of each input’s context length, and KVpop applies the same compression ratio through its released model variant. DMA sets its top-� from the nominal length, i.e., (1 − �) times 8K or 16K; it keeps the full KV cache and restricts attention to the selected entries. For IndexMem++, � is the average per-KV-head budget in Table 12. All budgets are inclusive: the attention sinks, local window, and any auxiliary slots a method maintains are funded within the budget rather than added on top. Table 18 lists the retention score and the protected region of each method.

Table 15: Complete RULER scores: Qwen3-8B, 8K. Compression ratios 50% and 75%.
<table><tr><td>Method</td><td>S1</td><td>S2</td><td>S3</td><td>MK1</td><td>MK2</td><td>MK3</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Full-KV</td><td>100.00</td><td>100.00</td><td>100.00</td><td>99.80</td><td>99.60</td><td>99.60</td><td>100.00</td><td>100.00</td><td>100.00</td><td>94.24</td><td>91.00</td><td>75.80</td><td>60.20</td><td>93.86</td></tr><tr><td colspan="10">Compression ratio: 50%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>88.40</td><td>73.20</td><td>3.00</td><td>42.40</td><td>29.80</td><td>24.40</td><td>37.95</td><td>37.70</td><td>88.80</td><td>91.74</td><td>92.33</td><td>49.00</td><td>47.40</td><td>54.32</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>99.80</td><td>99.00</td><td>95.20</td><td>53.40</td><td>17.00</td><td>99.70</td><td>99.55</td><td>100.00</td><td>93.20</td><td>87.67</td><td>34.60</td><td>26.40</td><td>77.35</td></tr><tr><td>DMA</td><td>99.00</td><td>82.40</td><td>80.80</td><td>83.60</td><td>75.60</td><td>1.60</td><td>75.60</td><td>89.25</td><td>99.16</td><td>77.28</td><td>88.93</td><td>74.00</td><td>50.60</td><td>75.22</td></tr><tr><td>TrimKV</td><td>100.00</td><td>100.00</td><td>99.80</td><td>99.40</td><td>99.00</td><td>21.00</td><td>99.55</td><td></td><td>99.90 100.00</td><td>94.30</td><td>90.40</td><td>72.00</td><td>58.60</td><td>87.23</td></tr><tr><td>KVpop</td><td>100.00</td><td>100.00</td><td>99.60</td><td>98.00</td><td>96.80</td><td>48.20</td><td>97.95</td><td></td><td>99.45 100.00</td><td>93.00</td><td>88.53</td><td>67.60</td><td>55.20</td><td>88.03</td></tr><tr><td>IndexMem++</td><td>100.00</td><td></td><td></td><td>100.00 100.00 100.00</td><td>99.00</td><td></td><td>99.20 100.00 100.00</td><td></td><td>100.00</td><td>94.10</td><td>90.80</td><td>75.00</td><td>61.20</td><td>93.79</td></tr><tr><td colspan="10">Compression ratio: 75%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>76.00</td><td>21.20</td><td>2.40</td><td>16.00</td><td>5.80</td><td>4.40</td><td>10.70</td><td>12.40</td><td>66.00</td><td>78.32</td><td>89.33</td><td>36.20</td><td>38.00</td><td>35.13</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>95.20</td><td>94.80</td><td>86.60</td><td>6.80</td><td>1.00</td><td>98.20</td><td>95.90</td><td>100.00</td><td>83.62</td><td>74.87</td><td>20.20</td><td>15.60</td><td>67.14</td></tr><tr><td>DMA</td><td>91.20</td><td>26.00</td><td>13.00</td><td>34.40</td><td>8.20</td><td>0.00</td><td>18.85</td><td>29.05</td><td>92.72</td><td>58.24</td><td>80.67</td><td>55.00</td><td>35.60</td><td>41.76</td></tr><tr><td>TrimKV KVpop</td><td>100.00 100.00</td><td>100.00 94.20</td><td>99.60 83.20</td><td>97.80</td><td>51.00</td><td>4.40 2.00</td><td>97.80 89.55</td><td>99.35 89.70</td><td>100.00</td><td>91.34 92.06</td><td>91.27 86.93</td><td>58.00 44.20</td><td>52.80</td><td>80.26 72.71</td></tr><tr><td>IndexMem++</td><td>100.00 100.00</td><td></td><td>96.40</td><td>79.20 99.20</td><td>37.80 97.80</td><td>73.00</td><td>98.80</td><td>100.00</td><td>100.00 100.00</td><td>83.18</td><td>91.40</td><td>71.40</td><td>46.40 60.60</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>90.14</td></tr></table>

Task abbreviations follow Table 6.

Table 16: Complete RULER scores: Qwen3-8B, 16K. Compression ratios 50% and 75%.
<table><tr><td>Method</td><td>S1</td><td>S2</td><td>S3</td><td>MK1</td><td>MK2</td><td>MK3</td><td>MV</td><td>MQ</td><td>VT</td><td>CWE</td><td>FWE</td><td>QA1</td><td>QA2</td><td>Avg.</td></tr><tr><td>Full-KV</td><td>100.00</td><td>100.00</td><td>99.80</td><td>99.60</td><td>100.00</td><td>99.60</td><td>99.60</td><td>100.00</td><td>100.00</td><td>82.54</td><td>93.80</td><td>75.20</td><td>58.40</td><td>92.96</td></tr><tr><td colspan="10">Compression ratio: 50%</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>98.20</td><td>94.80</td><td>9.20</td><td>67.60</td><td>33.40</td><td>18.40</td><td>63.80</td><td>67.95</td><td>99.00</td><td>77.86</td><td>95.60</td><td>44.60</td><td>45.20</td><td>62.74</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>100.00</td><td>99.00</td><td>82.20</td><td>31.60</td><td>13.60</td><td>98.70</td><td>98.00</td><td>100.00</td><td>86.40</td><td>94.67</td><td>33.00</td><td>30.60</td><td>74.44</td></tr><tr><td>DMA</td><td>99.60</td><td>80.40</td><td>79.80</td><td>79.40</td><td>61.80</td><td>0.80</td><td>73.95</td><td>90.25</td><td>98.56</td><td>55.06</td><td>94.47</td><td>68.40</td><td>50.60</td><td>71.78</td></tr><tr><td>TrimKV</td><td>100.00</td><td>100.00</td><td>99.60</td><td>97.80</td><td>98.80</td><td>17.80</td><td>97.65</td><td></td><td>99.70 100.00</td><td>82.04</td><td>97.07</td><td>70.40</td><td>56.00</td><td>85.91</td></tr><tr><td>KVpop</td><td>100.00 100.00</td><td></td><td>99.60</td><td>98.80</td><td>84.80</td><td>40.00</td><td>99.30</td><td></td><td>99.25 100.00</td><td>74.40</td><td>92.47</td><td>70.20</td><td>52.60</td><td>85.49</td></tr><tr><td>IndexMem++</td><td>100.00</td><td>100.00</td><td>99.80</td><td>99.80</td><td>99.40</td><td>79.40</td><td>99.30</td><td></td><td>99.90 100.00</td><td>76.30</td><td>95.40</td><td>75.20</td><td>56.80</td><td>90.87</td></tr><tr><td colspan="10">Compression ratio: 75%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SnapKV</td><td>91.40</td><td>58.60</td><td>2.40</td><td>24.80</td><td>7.80</td><td>3.80</td><td>17.10</td><td>20.45</td><td>94.96</td><td>58.02</td><td>93.87</td><td>33.00</td><td>33.80</td><td>41.54</td></tr><tr><td>KeyDiff</td><td>100.00</td><td>97.00</td><td>95.40</td><td>70.60</td><td>5.00</td><td>0.80</td><td>96.95</td><td>94.35</td><td>100.00</td><td>82.92</td><td>89.13</td><td>19.80</td><td>17.40</td><td>66.87</td></tr><tr><td>DMA</td><td>94.80</td><td>25.00</td><td>7.80</td><td>20.00</td><td>2.40</td><td>0.00</td><td>13.05</td><td>15.90</td><td>90.48</td><td>34.20</td><td>83.00</td><td>48.20</td><td>35.60</td><td>36.19</td></tr><tr><td>TrimKV</td><td>100.00</td><td>100.00</td><td>99.20</td><td>96.40</td><td>45.60</td><td>4.60</td><td>95.80</td><td>98.50</td><td>100.00</td><td>77.66</td><td>98.27</td><td>62.20</td><td>50.00</td><td>79.09</td></tr><tr><td>KVpop IndexMem++</td><td>100.00</td><td>93.20</td><td>80.20</td><td>83.40</td><td>31.80</td><td>1.60</td><td>91.35</td><td>89.95</td><td>100.00</td><td>67.84</td><td>88.13</td><td>63.80</td><td>42.20</td><td>71.81</td></tr><tr><td></td><td>100.00 100.00</td><td></td><td>98.40</td><td>98.80</td><td>51.20</td><td>22.40</td><td>98.95</td><td>99.70</td><td>99.96</td><td>56.04</td><td>96.47</td><td>73.80</td><td>56.40</td><td>80.93</td></tr></table>

Task abbreviations follow Table 6.

Table 17: Complete LongBench results on Qwen3-8B. 15 tasks (3,550 examples), at compression ratios 50% and 75%. (a) Single-document QA, multi-document $\mathbf { Q A } ,$ and summarization
<table><tr><td>Method</td><td>NQA</td><td>Qasper</td><td>MFQA</td><td>Hotpot</td><td>2Wiki</td><td>MuSiQue</td><td>GovR</td><td>QMSum</td><td>MNews</td></tr><tr><td>Full-KV</td><td>26.31</td><td>43.98</td><td>55.59</td><td>62.30</td><td>48.22</td><td>35.23</td><td>33.55</td><td>24.50</td><td>24.77</td></tr><tr><td colspan="8">Compression ratio: 50%</td><td></td></tr><tr><td>SnapKV</td><td>23.50</td><td>37.36</td><td>45.94</td><td>59.48</td><td>47.17</td><td>32.91</td><td>31.91</td><td>22.63</td><td>23.26</td></tr><tr><td>KeyDiff</td><td>16.81</td><td>29.46</td><td>40.33</td><td>36.23</td><td>34.46</td><td>16.03</td><td>29.34</td><td>22.37</td><td>21.66</td></tr><tr><td>DMA</td><td>22.40</td><td>42.19</td><td>52.01</td><td>55.56</td><td>39.37</td><td>33.24</td><td>32.21</td><td>23.22</td><td>24.16</td></tr><tr><td>TrimKV</td><td>26.25</td><td>47.66</td><td>51.89</td><td>59.33</td><td>43.08</td><td>36.45</td><td>33.19</td><td>24.09</td><td>24.31</td></tr><tr><td>IndexMem++</td><td>25.87</td><td>43.38</td><td>53.20</td><td>62.84</td><td>48.82</td><td>35.51</td><td>33.64</td><td>24.50</td><td>24.79</td></tr><tr><td colspan="8">Compression ratio: 75%</td><td></td></tr><tr><td>SnapKV</td><td>22.09</td><td>28.96</td><td>34.71</td><td>52.64</td><td>39.71</td><td>26.71</td><td>29.52</td><td>21.37</td><td>21.46</td></tr><tr><td>KeyDiff</td><td>12.11</td><td>19.56</td><td>28.76</td><td>20.70</td><td>27.82</td><td>9.54</td><td>21.91</td><td>21.52</td><td>13.94</td></tr><tr><td>DMA</td><td>17.30</td><td>33.19</td><td>37.01</td><td>40.60</td><td>38.11</td><td>24.45</td><td>28.05</td><td>21.26</td><td>21.31</td></tr><tr><td>TrimKV</td><td>24.75</td><td>44.34</td><td>49.82</td><td>59.53</td><td>42.40</td><td>31.81</td><td>32.87</td><td>23.72</td><td>24.40</td></tr><tr><td>IndexMem++</td><td>24.66</td><td>40.61</td><td>50.81</td><td>61.57</td><td>44.82</td><td>35.89</td><td>32.07</td><td>23.25</td><td>23.35</td></tr></table>

(b) Few-shot, synthetic, and code tasks; overall average
<table><tr><td>Method</td><td>TriviaQA</td><td>SAMSum</td><td>PCount</td><td>PRetr.</td><td>LCC</td><td>Repo-P</td><td>Avg.</td></tr><tr><td>Full-KV</td><td>90.56</td><td>40.23</td><td>10.00</td><td>90.60</td><td>64.74</td><td>60.20</td><td>47.39</td></tr><tr><td colspan="8">Compression ratio: 50%</td></tr><tr><td>SnapKV</td><td>89.74</td><td>40.77</td><td>9.00</td><td>91.00</td><td>66.20</td><td>60.00</td><td>45.39</td></tr><tr><td>KeyDiff</td><td>85.86</td><td>38.68</td><td>4.89</td><td>86.33</td><td>40.74</td><td>50.47</td><td>36.91</td></tr><tr><td>DMA</td><td>85.16</td><td>43.50</td><td>3.50</td><td>67.00</td><td>57.45</td><td>52.80</td><td>42.25</td></tr><tr><td>TrimKV</td><td>90.85</td><td>44.76</td><td>3.50</td><td>100.00</td><td>57.13</td><td>55.56</td><td>46.54</td></tr><tr><td>IndexMem++</td><td>90.29</td><td>40.25</td><td>10.00</td><td>95.22</td><td>65.05</td><td>59.59</td><td>47.53</td></tr><tr><td colspan="8">Compression ratio: 75%</td></tr><tr><td>SnapKV</td><td>89.59</td><td>40.49</td><td>7.55</td><td>88.29</td><td>65.78</td><td>61.46</td><td>42.02</td></tr><tr><td>KeyDiff</td><td>81.63</td><td>36.19</td><td>4.82</td><td>48.33</td><td>25.63</td><td>50.43</td><td>28.19</td></tr><tr><td>DMA</td><td>77.46</td><td>41.91</td><td>1.50</td><td>8.50</td><td>53.87</td><td>46.30</td><td>32.72</td></tr><tr><td>TrimKV</td><td>89.42</td><td>43.47</td><td>3.00</td><td>100.00</td><td>56.50</td><td>55.17</td><td>45.41</td></tr><tr><td>IndexMem++</td><td>89.50</td><td>39.44</td><td>7.50</td><td>94.42</td><td>63.79</td><td>58.72</td><td>46.03</td></tr></table>

NQA = NarrativeQA; MFQA = MultiFieldQA-en; Hotpot = HotpotQA; 2Wiki = 2WikiMQA; GovR = GovReport; MNews = MultiNews; PCount = PassageCount; PRetr. = PassageRetrieval-en; Repo-P = RepoBench-P. Avg. is the equal-weight mean of all 15 displayed tasks. Budget accounting and protected regions are given in Appendix H.

For IndexMem++, � is the average per-KV-head budget. The mass allocator redistributes the fixed layerwide total $\begin{array} { r } { \sum _ { h } B _ { h } = H _ { \mathrm { K V } } K } \end{array}$ across KV heads, so an individual head may hold more or fewer than � entries while the layer total is preserved.

Table 18: RULER retention criteria and protected regions. Budgets follow the accounting above; the protected region of each method is funded within its budget rather than added on top.
<table><tr><td>Method</td><td>Retention score</td><td>Protected region (within budget)</td></tr><tr><td>SnapKV</td><td>Observed-attention mean with width-5 average pool- Final 64 context positions ing</td><td></td></tr><tr><td>KeyDiff</td><td>Cosine-based post-RoPE key scoring</td><td>Final context position</td></tr><tr><td>DMA</td><td>Query-independent value-derived gate score</td><td>Sink 4, local 128</td></tr><tr><td>TrimKV</td><td>Released gate deltas, magnitude and lifetime</td><td>Sink 4, local 128</td></tr><tr><td>KVpop</td><td>Recurrent scorer state</td><td>Sink 4, local 256</td></tr><tr><td>IndexMem++ Learned gate on</td><td> $h _ { j } ;$  mass allocation across heads</td><td>Sink 4, local 128, CMP 64</td></tr></table>

LongBench. All methods use a maximum context length of 32,768 tokens and the task-specific generation limits defined by the benchmark. Prompts longer than this limit are truncated before prefill. For compression ratio $c ,$ the average per-KV-head budget is $K = \lfloor ( 1 - c ) N \rfloor$ , where � is the prompt length after truncation; the budget is fixed once at the end of prefill and held during decoding. As on RULER, the budget is inclusive of each method’s protected region, and the retention criteria and protected regions are those of Table 18.

Averaging. RULER and LongBench scores are shown on the benchmark’s 0–100 scale. RULER uses the arithmetic mean of the 13 scorer-returned task scores. LongBench uses the equal-weight mean of the 15 displayed task scores; its category sizes are 3, 3, 3, 2, 2, and 2. Few-shot averages only TriviaQA and SAMSum. We calculate category and overall means from task scores before rounding the displayed result to two decimals.

## I. Math Reasoning Protocol

All math results use Qwen3-8B with thinking enabled, temperature 0.6, and top-� 0.95. pass@1 is the fraction of correct rollouts: 500 problems with two rollouts each on Math500, and the 30 problems of AIME25 I+II with eight rollouts each. The rows of Table 8 come from separate evaluation runs whose settings difer as follows.

IndexMem++ and Full-KV. Both apply top-� 20 and the same per-rollout seed scheme, so their rows are paired. They generate at most 16,384 new tokens on Math500 and 32,768 on AIME25. IndexMem++ uses the ofline per-budget head table with a �/4 floor and no centroid slots (Appendix G). Their prompts end with an instruction to put the final answer within \boxed{}; the problem sets are otherwise identical to those of the other runs.

R-KV and TriAttention. Both run in the Random Attention evaluation code, with prompt plus output capped at 32,768 tokens, top-� 20, and a per-problem seed scheme, so no paired test against IndexMem++ is available. R-KV uses � = 0.5 with smoothing and a 64-token residual window. TriAttention uses $n _ { \mathrm { l a r g e } } = 2 5 6$ and a 64-token residual window. Because no Qwen3-8B calibration is released, its statistics are calibrated on the question text of the 530 evaluation problems, without answers (40,691 tokens).

Random Attention. These rows are produced by the released Random Attention code. The policy keeps the prompt and a 64-token recency window and fills the rest of the budget with a uniformly random per-KV-head subset of generated tokens. Prompt plus output is capped at 32,768 tokens. The runs are unseeded and apply only temperature and top-�, without top-�.

KVzap and Fast-KVzip. Both use their released Qwen3-8B scorers under one shared eviction rule that holds � entries per KV head at every decoding step, with four sinks and a window of $\lfloor K / 4 \rfloor$ entries inside the budget. Their Math500 cells use a 125-problem subsample (drawn with seed 42) with two rollouts each; their AIME25 cells use all 30 problems. They generate at most 16,384 new tokens on Math500 and 32,768 on AIME25 and use the same seed scheme and prompt as the IndexMem++ runs. Their rows are paired only with each other.

## J. Analysis of Retained Entries

This section gives the setup and the complete results behind Section 4.4.

Setup. All runs use Qwen3-8B in BF16 with the IndexMem++ gate of Section 4.1 and physical eviction. The average per-KV-head budget is � = 2,048, inclusive of 64 centroid slots, four sinks, and a 128-token local window, which leaves 1,852 selectable entries per head under uniform allocation. The control is a recency policy with the same per-layer, per-head budgets and the same centroid slots; it keeps the most recent selectable entries. The AIME25 traces are generated by Qwen3-8B with Full-KV under the decoding settings of the free-generation runs below. We use one trace longer than 8K tokens for each of 28 problems; the other two problems have no such trace. The SWE data are 25 SWE-agent trajectories on SWE-bench tasks from 8 repositories. We replay the full message history verbatim, with its role and tool-call boundaries, rather than the per-step requests, which keep only the most recent tool observations.

Each trace is examined under two protocols. A snapshot prefills the 8K-token prefix at once and commits the compressed cache. A replay feeds the trace token by token from the initial prompt and updates the physical cache at every step; when the initial prompt is short, the allocator falls back to uniform budgets. Positional statistics exclude the sinks and the local window, which leaves 8,060 selectable positions, and count original entries only, since centroid slots have no position.

Overlap with a recency window. For each head, we compare its retained set with the most recent selectable positions of the same size and report the fraction of retained positions that fall in this recency window. The fraction is averaged over heads within a trace and then over traces. On eight traces per domain replayed to 8K tokens, the overlap is 52.50% on AIME25 and 48.35% on SWE, so about half of the retained entries lie outside the recency window. The recency policy overlaps by 100% by construction.

Prediction on fixed continuations. On the same eight traces per domain, we score the 256 assistant tokens that follow the 8K-token prefix; tool and user tokens are fed to the model but not scored. Table 19 reports the NLL and the KL divergence from the Full-KV next-token distribution, $\mathrm { K L } ( p _ { \mathrm { f u l l } } \parallel p ) ;$ , averaged over traces. IndexMem++ is lower than the recency policy on both metrics for all eight traces in each domain.

Table 19: Prediction on fixed continuations with Qwen3-8B at $K = 2 , 0 4 8 ,$ , averaged over eight traces per domain. Lower is better.
<table><tr><td rowspan="2"></td><td colspan="3">NLL</td><td colspan="2">KL from Full-KV</td></tr><tr><td>Full-KV</td><td>IndexMem++</td><td>Recency</td><td>IndexMem++</td><td>Recency</td></tr><tr><td>AIME25</td><td>0.197</td><td>0.207</td><td>0.279</td><td>0.021</td><td>0.082</td></tr><tr><td>SWE</td><td>1.068</td><td>1.221</td><td>1.551</td><td>0.120</td><td>0.630</td></tr></table>

Early facts in SWE trajectories. In each SWE trajectory, we identify facts stated early in the issue and their explicit applications later in the trajectory, between 4K and 8K tokens. We then check whether the corresponding original entries are still cached when each fact is applied. A fact counts as retained when every token of its core span has a copy in at least one layer and head. In 17 of the 25 trajectories, covering 7 of the 8 repositories, at least one fact is applied later. All 39 applied facts are retained before they are applied; 30 of them are complete within the heads of a single layer and 16 within a single head. The recency policy keeps no original entry for 37 of the 39. Facts with no observed later application are also retained: 111 of 111 at

4K tokens and 109 of 111 at 8K tokens. Retention is therefore not specific to facts that are applied later. In addition, 23 of the applied facts reappear in full later in the trajectory, for example when they are restated or re-read by a tool.

Layers. We measure the similarity of two layers by the weighted Jaccard index of their retained positions, weighting each position by the number of heads in the layer that keep it. In SWE replay, adjacent layers score 0.512 and layers at least eight apart score 0.413. Adjacent layers are more similar than distant ones in all 25 SWE and all 28 AIME25 traces. A random selection that matches each head’s capacity and its counts in eight coarse age bins gives 0.513 for adjacent layers, averaged over 100 draws. Taking unions, the eight heads of one layer hold original entries for 57.2% of the selectable positions on average, four layers for 85.0% in expectation over layer subsets, and all 36 layers for 99.9%. The matched random selection reaches 97.7% with four layers.

Heads. Within a layer, the Jaccard index between the retained sets of two heads averages 0.372 in SWE replay, against 0.212 for the matched random selection. In 8K snapshots, the two heads with the largest budgets in each layer hold 50.2% of the selectable capacity on average (95% interval [49.2%, 51.3%]), whereas uniform allocation would give 25%.

Free generation on AIME25. For each of the 30 AIME25 problems, we generate eight samples with thinking enabled, temperature 0.6, top-� 0.95, top-� 20, and at most 32,768 new tokens. Under the configuration above, IndexMem++ reaches a pass@1 of 0.592 (142 of 240 samples correct), the recency policy 0.121 (29 of 240), and Full-KV 0.654 (157 of 240). Paired over problems, IndexMem++ exceeds the recency policy by 47.08 points (95% bootstrap interval [33.75, 60.83]) and trails Full-KV by 6.25 points ([1.67, 11.25]).