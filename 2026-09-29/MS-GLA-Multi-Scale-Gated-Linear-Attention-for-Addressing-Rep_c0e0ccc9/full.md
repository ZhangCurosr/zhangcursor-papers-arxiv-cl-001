# MS-GLA: Multi-Scale Gated Linear Attention for Addressing Representational Bottlenecks via Multi-Temporal Resolution

Prasoon Dev, Anirudh Sankar, Vasudeva Varma

Language Technologies Research Center

International Institute of Information Technology Hyderabad

Hyderabad, Telangana, India

{prasoon.dev,anirudh.sankar}@research.iiit.ac.in, vv@iiit.ac.in

## Abstract

Gated Linear Attention (GLA) Transformers advance linear recurrent models through data-dependent gating, but face a core limitation: the fixedcapacity memory matrices across all heads operate at a single temporal resolution, where each token is processed individually, forcing them to simultaneously encode local syntactic patterns and long-range semantic structure, creating a representational bottleneck that gating alone is insufficient to resolve. We introduce Multi-Scale Gated Linear Attention (MS-GLA), which addresses this by distributing attention heads across multiple temporal resolutions. Coarser resolutions pool longer token spans naturally specializing toward long-range dependencies, while finer head groups retain sensitivity to local syntactic structure. A learnable, inputdependent fusion layer dynamically recombines head group outputs at each timestep, expanding effective memory capacity without increasing per-head state size. This multi-resolution decomposition draws on principles from Multi-Scale State-Space Models (MS-SSM), adapting them to the gated linear attention setting. We evaluate MS-GLA on language modeling, recall-intensive tasks, and long-context generalization. Across all settings, MS-GLA consistently achieves higher accuracy and lower perplexity than GLA at matched parameter counts, with up to 18.9% improvement on recall-intensive tasks and 9.5% lower average perplexity on language modeling benchmarks, validating multi-temporal resolution decomposition as a principled and effective extension of Gated Linear Attention.

## 1 Introduction

Transformers (Vaswani et al., 2017) have long been the gold standard for sequence modeling, their reliance on softmax attention yields quadratic computational complexity with respect to sequence length. To overcome this, the field has been rapidly shifting towards sub-quadratic, linear-time models. Architectures like Linear Recurrent Neural Networks (RNNs), State Space Models (SSMs) like Mamba (Gu & Dao, 2024), and Linear Attention variants (like RetNet (Sun et al., 2024)) treat sequence mixing as a linear recurrence. This is highly efficient, allowing for an O(1) memory footprint during inference and parallelized training. Recently, Gated Linear Attention (GLA) (Yang et al., 2024) took this a step further. By adding datadependent gating, GLA gave models the ability to actually choose what context to remember and what obsolete information to forget, effectively bridging the performance gap with standard Transformers while keeping training highly hardware-efficient.

While data-dependent gating improves how information is retained over time, GLA still operates at a single temporal resolution, where its fixed-capacity memory matrix must summarize the entire sequence history. This creates an inherent representational bottleneck, especially for language, which naturally exhibits a multi-scale structure (Tamkin et al., 2020; Nawrot et al., 2022): a single shared state must simultaneously capture rapidly shifting, highfrequency syntactic details alongside slowly evolving, low-frequency semantic structures. Expressive gates might allow a model to choose what to remember or forget, but they cannot fix the underlying problem of how to stretch a limited memory budget across these wildly different timescales.

To address this, we propose Multi-Scale Gated Linear Attention (MS-GLA). Instead of forcing the model to operate at a single temporal resolution handling all timescales, MS-GLA explicitly breaks the input sequence down into multiple temporal resolutions and assigns dedicated groups of GLA heads to each. By operating on temporally pooled tokens, the coarser head groups naturally expand their receptive fields, allowing their gates to specialize entirely in tracking long-range semantic dependencies. Meanwhile, the finer head groups process tokens at their original resolution, retaining strict sensitivity to local syntactic structure. This multi-resolution decomposition is grounded in the principles of Multi-Scale State-Space Models (MS-SSM) (Karami et al., 2025), which demonstrate that assigning recurrent branches to distinct temporal scales yields complementary specialization that a single shared state cannot achieve. MS-GLA adapts this insight to the gated linear attention setting, replacing SSM branches with groups of GLA heads operating on pooled token sequences.

Crucially, MS-GLA achieves this division of labor without bearing the overhead of typical ensemble approaches. We aren’t instantiating full-sized models at every scale or inflating the parameter count. Instead, MS-GLA simply redistributes the baseline model’s existing budget of attention heads across these different time scales (for example, assigning two heads to fine resolutions and two heads to coarser scales). The outputs from these specialized groups are then upsampled via a causal zero-order hold and dynamically recombined at each timestep using a learnable fusion layer. This allows the model to adaptively weigh fine-grained details and coarse-grained information based on the immediate context. Furthermore, MS GLA employs non-overlapping causal average pooling, ensuring full compatibility with the hardware-efficient chunkwise training algorithms that make GLA practical at scale.

## 2 Background and Related Work

## 2.1 Linear Attention and Gated Linear Attention (GLA)

Transformers use softmax attention which do not scale well to long sequences because the complexity is quadratic with respect to sequence length. This has motivated the development of Linear Attention mechanisms which achieve linear-time complexity with respect to sequence length (Katharopoulos et al., 2020). To achieve linear complexity in sequence length, the softmax operation is replaced with a kernel with an associated feature map (Katharopoulos et al., 2020). This enables the reformulation of attention as a linear recurrent neural network (RNN). Prior work has shown that an unnormalized linear kernel works best (Sun et al., 2024). Therefore, the recurrent update for the hidden state $S _ { t } \in \mathbb { R } ^ { d _ { k } \times d _ { v } }$ and the output $o _ { t }$ at time step t is given by:

$$
{ \cal S } _ { t } = { \cal S } _ { t - 1 } + \mathbf { k } _ { t } ^ { T } \mathbf { v } _ { t }\tag{1}
$$

$$
\mathbf { \omega } _ { \mathbf { } } \mathbf { o } _ { t } = \mathbf { \ b { q } } _ { t } \mathbf { \ b { S } } _ { t }\tag{2}
$$

where $q _ { t } , k _ { t }$ and $v _ { t }$ denote the query, key and value vectors respectively.

The chunkwise parallel form (Hua et al., 2022) of linear attention enables partially parallel training with subquadratic complexity. Striking a balance between the computationally slow recurrent form and the computationally expensive parallel form of attention. In the chunkwise parallel form, the input sequence is divided into non-overlapping chunks of size C. The recurrence updates are thus split into two phases: an inter-chunk recurrence and an intra-chunk computation. The inter-chunk recurrence works sequentially to update the chunk level hidden state and the intra-chunk output computation computes the outputs in parallel. That is formally, the updates for chunk i are given by:

$$
S [ i + 1 ] = S [ i ] + K [ i ] ^ { T } V [ i ]\tag{3}
$$

$$
O [ i + 1 ] = Q [ i + 1 ] S [ i ] + \left( ( Q [ i + 1 ] K [ i + 1 ] ^ { T } ) \odot M \right) V [ i + 1 ]\tag{4}
$$

where $K [ i ] , V [ i ] , Q [ i + 1 ]$ are the stacked token vectors for their respective chunks, and M is the causal mask.

While computationally efficient, standard linear attention lacks a decay term, which makes it difficult for the model to ”forget” historically irrelevant information (Buckman & Gelada). This has been hypothesized to potentially limit performance of on long-context tasks. Gated Linear Attention (GLA) resolves this by incorporating a data-dependent 2D forget gate (Yang et al., 2024) $G _ { t } \in ( 0 , 1 ) ^ { d _ { k } \times d _ { v } }$ into the linear recurrence. The generalized gated update is expressed as:

$$
{ \pmb S } _ { t } = { \pmb G } _ { t } \odot { \pmb S } _ { t - 1 } + { \pmb k } _ { t } ^ { T } { \pmb v } _ { t }\tag{5}
$$

where ⊙ denotes the Hadamard product. To maintain a balance between parameter efficiency, state size, and hardware-efficient training, GLA specifically parametrizes the forget gate as $G _ { t } = \alpha _ { t } ^ { T } \mathbf { 1 }$ , which simplifies the hidden state update to a row-wise scaling:

$$
{ S } _ { t } = \mathrm { D i a g } ( \alpha _ { t } ) { S } _ { t - 1 } + k _ { t } ^ { T } \boldsymbol { v } _ { t }\tag{6}
$$

where the data-dependent gating vector $\boldsymbol { \alpha } _ { t } \in \mathbb { R } ^ { 1 \times d _ { k } }$ is generated dynamically at each time step using a low-rank linear projection of the input $x _ { t }$ followed by a sigmoid activation. This data-dependent gating mechanism significantly enhances the expressive power and length extrapolation capabilities of the model while retaining full compatibility with the sub-quadratic chunkwise parallel training form.

## 2.2 Mixture-of-Memories (MoM)

A related line of work, Mixture-of-Memories (MoM) (Du et al., 2025), also targets the fixed-capacity memory bottleneck of linear sequence models, but through a fundamentally different mechanism. MoM maintains several independent memory states and uses a learned top-k router to direct individual tokens to a sparse subset of these memories at each step, reducing interference by ensuring that only a subset of memory states is updated by any given token. MS-GLA instead partitions the temporal resolution at which the sequence itself is processed: deterministic, parameter-free average pooling produces multiple coarser views of the same sequence, with dedicated GLA head groups assigned to each view. Where MoM asks which memory should this token update, MS-GLA asks at what temporal resolution should this sequence be processed. The two mechanisms are complementary rather than competing, and we view combining MoM-style token routing with MS-GLA’s multi-resolution branches as a promising direction for future work.

## 2.3 Multi-Scale Sequence Modeling and MS-SSM

Real-world signals such as text, images, and time series naturally exhibit multi-scale structure, with patterns ranging from fine-grained, high-frequency details to coarse, global trends (Caucheteux et al., 2023; Shi et al., 2023). Multi-Resolution Analysis (MRA) techniques such as the Stationary Wavelet Transform (Nason & Silverman, 1995) have long been used to capture such hierarchies. Multi-Scale State-Space Models (MS-SSM) (Karami et al., 2025) adapt this MRA framework to deep sequence modeling by replacing fixed wavelet bases with trainable, causal depthwise 1D convolutions, recursively applying dilated convolutions to decompose the input sequence:

$$
[ a _ { s } ; d _ { s } ] = \mathrm { C o n v 1 d } ( 1 , 2 , L , 2 ^ { s - 1 } ) [ a _ { s - 1 } ]\tag{7}
$$

This transforms the sequence into a multi-resolution representation $\boldsymbol { x } _ { t } \mapsto \boldsymbol { \hat { x } _ { t } } \in \mathbb { R } ^ { S + 1 }$ , where higher scales capture coarse-grained structure over larger receptive fields and lower scales retain sharp, localized detail. Each scale is fed into a separate State Space Model (SSM) operating in parallel. Since effective memory capacity in linear recurrent models is inversely proportional to the distance of the state transition matrix’s eigenvalues from the unit circle (Agarwal et al., 2024), MS-SSM initializes coarser branches with transition matrix values closer to 1 (slower forgetting, favoring long-range dependencies) and finer branches with smaller values (faster forgetting, favoring local dynamics). The branch outputs are then combined via an input-dependent scale-mixer, $z _ { t } ^ { \cdot } = \mathrm { L i n e a r E } ( x _ { t } ) y _ { t }$ (Karami et al., 2025), which dynamically routes information across scales based on the raw input token.

## 3 Multi-Scale Gated Linear Attention (MS-GLA)

To alleviate the limitations of modeling an entire sequence through a single temporal resolution, where each recurrent update operates on individual tokens and must simultaneously encode both local syntactic patterns and long-range semantic structure, we introduce Multi-Scale Gated Linear Attention (MS-GLA). Instead of relying on one recurrent pathway to represent both rapidly varying local patterns and slower long-range structure, MS-GLA processes the input through multiple GLA heads operating at different temporal scales. This provides an architectural bias toward modeling information at multiple timescales within the same layer (Karami et al., 2025; Buckman & Gelada).

## 3.1 Multi-Scale Decomposition via Temporal Pooling

Given input hidden states $\pmb { X } \in \mathbb { R } ^ { L \times d }$ , where L is the sequence length and d is the hidden dimension, let S denote a set of predefined temporal scales $( \mathbf { e . g . } \breve { S } = \{ 1 , 2 , 4 \} )$ . For each scale $s \in { \mathcal { S } } .$ , MS-GLA constructs a lower-resolution sequence by applying non-overlapping average pooling over blocks of size s (Karami et al., 2025; Nawrot et al., 2022). Throughout this work, we use temporal scale and temporal resolution interchangeably to refer to the pooling factor s: a larger s produces a coarser, lower-frequency representation in which every s consecutive input positions are pooled into a single token, whereas s = 1 preserves the original token-level resolution.

When the sequence length L is not divisible by $s ,$ the sequence is right-padded with zeros before pooling. If an attention mask $\mathbf { M } \in \{ 0 , 1 \} ^ { L }$ is provided, pooling is performed in a masked manner. We denote by $\mathbf { \boldsymbol { x } } ^ { ( s ) }$ the pooled hidden-state sequence at scale s. The pooled representation for the i-th block is

$$
\mathbf { X } _ { i } ^ { \left( s \right) } = \frac { \sum _ { j = 0 } ^ { s - 1 } \mathbf { X } _ { i s + j } \mathbf { M } _ { i s + j } } { \operatorname* { m a x } \left( 1 , \sum _ { j = 0 } ^ { s - 1 } \mathbf { M } _ { i s + j } \right) } .\tag{8}
$$

The numerator sums the hidden states of the s tokens in the i-th block, weighted by their mask values so that padding tokens contribute nothing. The denominator normalizes by the number of valid (unmasked) tokens in the block, with the max(1, ·) guard preventing division by zero in fully-padded blocks.

In the absence of masking, this reduces to standard average pooling. After padding, the pooled sequence length is approximately $\lceil L / s \rceil$ . As s increases, each recurrent update summarizes a larger temporal block, biasing coarser resolutions toward longer-range dependencies.

## 3.2 Scale-Specific GLA Heads

MS-GLA processes each temporal scale through dedicated GLA heads, while keeping the total attention-head and recurrent-state budget identical to the baseline. Each branch is an independent GLA module that takes its scale-specific pooled sequence $\mathbf { \boldsymbol { x } } ^ { ( s ) }$ as input. Branches share the same architectural form as GLA: query, key, value projections, datadependent gating, and normalization, but maintain entirely separate learned parameters. This means that each branch’s gates and projections can specialize freely to the temporal scale it operates at, without any parameter sharing across scales.

Rather than adding new heads, MS-GLA partitions the baseline model’s H attention heads across the set of scales $s ,$ assigning $h _ { s }$ heads to the branch at scale s, such that

$$
\sum _ { s \in \mathcal { S } } h _ { s } = H .\tag{9}
$$

For example, with $H = 4$ heads and scales $S = \{ 1 , 2 , 4 \}$ , the head allocation $\mathbf { h } = [ 2 , 1 , 1 ]$ assigns two heads to the finest scale $( s = 1 )$ and one head each to the coarser scales $( s = 2$ and $\bar { s } = 4 )$ . The key and value projection dimensions are split proportionally, so each branch receives a fraction $\dot { h } _ { s } / H$ of the total key-value budget.

## 3.3 Causal Upsampling and Learnable Scale Fusion

After branch-wise sequence mixing, all branch outputs must be mapped back to the original token resolution. For a branch operating at scale s, the branch output is repeated s times along the sequence dimension and then shifted to the right by $s - 1$ positions, with zeros inserted at the beginning. This implements a causal hold mechanism: the output corresponding to a pooled block is not revealed until all tokens in that block have been observed (Karami et al., 2025).

We denote the causally aligned output of branch s after upsampling by $\hat { \mathbf { Y } } ^ { ( s ) } \in \mathbb { R } ^ { L \times d }$ . MS-GLA then combines the aligned branch outputs using a learnable, input-dependent fusion module (Karami et al., 2025). For each timestep $t ,$ the routing weights w<sub>t</sub> are computed from the unpooled hidden state x<sub>t</sub> using a learned fusion projection with weight matrix $\mathbf { W } _ { \mathrm { f u s e } } \in \mathbb { R } ^ { | \cal { S } | \times d }$ and bias vector ${ \bf b } _ { \mathrm { f u s e } } \in \mathbb { R } ^ { | S | }$

$$
\mathbf { w } _ { t } = \mathrm { S o f t m a x } ( \mathbf { W } _ { \mathrm { f u s e } } \mathbf { x } _ { t } + \mathbf { b } _ { \mathrm { f u s e } } ) .\tag{10}
$$

The final layer output at timestep t, denoted by $\mathbf { O } _ { t } ,$ is

$$
\mathbf { O } _ { t } = \sum _ { s \in \mathcal { S } } w _ { t , s } \hat { \mathbf { Y } } _ { t } ^ { ( s ) } .\tag{11}
$$

## 3.4 Compatibility with Chunkwise GLA Training

A key practical advantage of standard GLA is its hardware-efficient chunkwise parallel training formulation. MS-GLA preserves compatibility with this training regime (Yang et al., 2024; Hua et al., 2022). Since the multi-scale decomposition uses static, non-overlapping pooling, each branch still operates on a contiguous sequence. As a result, every scale-specific branch can apply the same chunkwise GLA computations as the baseline model, but on a shorter sequence whose length decreases with the branch scale.

Consequently, MS-GLA extends GLA with multi-resolution sequence modeling while retaining the underlying chunkwise training structure of the original architecture. Although adding multiple branches introduces additional computation, coarser branches operate on proportionally shorter sequences, which mitigates part of this overhead and helps maintain practical training efficiency.

## 4 Experimental Setup

Our main experiments study whether Multi-Scale Gated Linear Attention (MS-GLA) improves over standard Gated Linear Attention (GLA) under matched model and training budgets. We focus on autoregressive language modeling and long-context evaluation, and compare MS-GLA variants against a GLA baseline, trained using the Yang et al. (2024)’s implementation in flash-linear-attention (Yang & Zhang, 2024) with the flame training framework (Zhang & Yang, 2025) under an identical architecture and optimization recipe, isolating the effect of multi-scale temporal decomposition from changes in model capacity or training procedure.

## 4.1 Model variants.

All models share the same decoder-only backbone and differ only in the sequence-mixing module. The common backbone uses 24 layers, hidden size 1024, and 4 total attention heads. This choice of 4 heads is inherited directly from the original GLA configuration: Yang et al. (2024) demonstrate in their ablation study that 4 heads represents a favorable tradeoff between training throughput and perplexity, with larger head counts degrading perplexity and smaller head counts offering only marginal gains at the cost of significantly higher memory usage. We adopt this finding unchanged, as our contribution lies in how the fixed head budget is allocated across temporal scales, not in the total number of heads.

While maintaining the same parameter count as the baseline GLA model, we train and evaluate five multiscale variants of MS-GLA:

$$
\{ 1 , 2 \} , \quad \{ 1 , 4 \} , \quad \{ 2 , 4 \} , \quad \{ 1 , 2 , 4 \} , \quad \{ 1 , 2 , 4 , 8 \} .
$$

For MS-GLA, the total head budget is partitioned across scales while keeping the overall attention/state budget fixed. The corresponding head allocations are:

$$
[ 2 , 2 ] , \quad [ 2 , 2 ] , \quad [ 2 , 2 ] , \quad [ 2 , 1 , 1 ] , \quad [ 1 , 1 , 1 , 1 ] .
$$

## 4.2 Training details.

All models are trained from scratch on the FineWeb-Edu corpus (Lozhkov et al., 2024) with the same tokenizer using AdamW (Loshchilov & Hutter, 2017). We train 340M parameter models for 7B tokens, following the compute-optimal token-to-parameter ratio prescribed by the Chinchilla scaling laws (Hoffmann et al., 2022) for a model of this scale. All reported comparisons are made at matched parameter count and matched training tokens under an identical optimization recipe. Full training configuration – including optimizer hyperparameters, batch size, warmup schedule, hardware, precision, and random seed – along with our released code, is provided in Appendix B.

## 5 Results

To assess whether multi-temporal resolution decomposition yields broad and robust improvements, we evaluate five MS-GLA configurations ({1, 2}, {1, 4}, {2, 4}, {1, 2, 4}, and {1, 2, 4, 8}) against a GLA baseline across three axes of model quality, followed by two analyses that probe how these gains are achieved. First, language modeling measures general sequence understanding, capturing both local syntax and long-range semantics via perplexity on Wikitext (Merity et al., 2017) and LAMBADA (Paperno et al., 2016), along with zero-shot accuracy on LAMBADA, PIQA (Bisk et al., 2020), HellaSwag (Zellers et al., 2019), and Winogrande (Sakaguchi et al., 2021). Second, recall-intensive tasks evaluate the model’s ability to retrieve precise in-context information without memory saturation, using F1 score on SWDE (Lockard et al., 2019), FDA (Arora et al., 2025), and SQuAD (Rajpurkar et al., 2018) as modified in Arora et al. (2024). Third, long-context generalization probes robustness far beyond the training horizon by measuring perplexity as a function of token position on SlimPajama (Soboleva et al., 2023) and PG19 (Rae et al., 2020) at sequence lengths up to 15× the training context (30,720 tokens), where flat curves indicate successful extrapolation and divergence signals recurrent state saturation. We then examine fusion layer routing stability to verify that this long-context robustness reflects genuine multiscale usage rather than collapse onto a single branch, and finally quantify computational efficiency to confirm that any gains are achieved without disproportionate cost in compute, memory, or wall-clock training time.

## 5.1 Language Modeling

Table 1 reports language modeling and zero-shot downstream results across all six models. The clearest pattern is that including the finest temporal resolution (s = 1) is necessary for reliable improvement: every configuration that retains at least one native-resolution branch outperforms GLA in aggregate, while the purely coarse-grained {2, 4} configuration degrades substantially across most metrics.

MS-GLA {1, 2, 4} achieves the best overall performance, reducing average perplexity by roughly 9.5% and improving mean downstream accuracy across Lambada, PIQA, and HellaSwag relative to the GLA baseline. MS-GLA {1, 4} leads on Wikitext perplexity and

<table><tr><td>Scale</td><td>Model</td><td>Wiki. ppl↓</td><td>LMB. pp1↓</td><td>LMB. acc↑</td><td>PIQA↑</td><td>Hella.↑</td><td>Wino.↑</td><td>Avg. score↑</td><td>Avg. pp1.↓</td></tr><tr><td>340M</td><td>GLA</td><td>19.63</td><td>22.26</td><td>19.33</td><td>64.20</td><td>33.94</td><td>50.28</td><td>41.94</td><td>20.94</td></tr><tr><td>7B tok</td><td>MS-GLA {1,2}</td><td>19.25</td><td>20.99</td><td>19.68</td><td>65.18</td><td>34.89</td><td>49.17</td><td>42.23</td><td>20.12</td></tr><tr><td></td><td>MS-GLA {1,4}</td><td>17.43</td><td>21.69</td><td>19.89</td><td>63.87</td><td>35.49</td><td>51.07</td><td>42.58</td><td>19.56</td></tr><tr><td></td><td>MS-GLA {2,4}</td><td>21.66</td><td>31.47</td><td>16.15</td><td>65.18</td><td>34.17</td><td>52.57</td><td>42.01</td><td>26.56</td></tr><tr><td>MS-GLA</td><td>{1,2,4}</td><td>18.33</td><td>19.59</td><td>20.69</td><td>65.83</td><td>35.01</td><td>49.88</td><td>42.85</td><td>18.96</td></tr><tr><td></td><td>MS-GLA {1,2, 4,8}</td><td>18.80</td><td>20.18</td><td>20.34</td><td>65.34</td><td>34.53</td><td>49.01</td><td>42.31</td><td>19.49</td></tr></table>

Table 1: Language modeling perplexity and zero-shot downstream task performance of GLA and MS-GLA variants. Best results in each column are shown in bold.
<table><tr><td>Model</td><td>SWDE F1↑</td><td>FDA F1↑</td><td>SQuAD F1↑</td><td>Avg. F1↑</td></tr><tr><td>GLA</td><td>7.86</td><td>5.52</td><td>8.84</td><td>7.41</td></tr><tr><td>MS-GLA {1, 2}</td><td>9.26</td><td>5.95</td><td>9.79</td><td>8.33</td></tr><tr><td>MS-GLA {1, 4}</td><td>8.28</td><td>6.06</td><td>10.32</td><td>8.22</td></tr><tr><td>MS-GLA {2, 4}</td><td>5.96</td><td>5.42</td><td>8.21</td><td>6.53</td></tr><tr><td>MS-GLA {1, 2,4}</td><td>9.30</td><td>5.98</td><td>11.15</td><td>8.81</td></tr><tr><td>MS-GLA {1, 2, 4, 8}</td><td>7.59</td><td>5.27</td><td>11.05</td><td>7.97</td></tr></table>

Table 2: Performance comparison of GLA and various MS-GLA configurations on recallintensive benchmarks. F1 scores are reported for each task, with the highest values highlighted in bold.

HellaSwag but lags behind {1, 2, 4} on average, suggesting that skipping the intermediate scale creates a coherence gap despite strong structural modeling. The four-scale configuration {1, 2, 4, 8} remains competitive overall but falls slightly short of {1, 2, 4} across perplexity and accuracy, consistent with head-budget dilution as representational capacity is spread across an additional scale. Winogrande is the one exception where fine-resolution configurations offer no clear advantage, and the {2, 4} variant, which entirely omits the fine-grained scale (s = 1), records the highest individual score on this benchmark, a pattern we discuss further in Section 6.2.

## 5.2 Recall-Intensive Tasks

Table 2 reports recall-intensive generation performance. The ranking of configurations mirrors the language modeling results closely. MS-GLA {1, 2, 4} achieves the highest average F1, representing roughly an 18.9% improvement over the GLA baseline, with leading scores on both SWDE and SQuAD. MS-GLA {1, 2} and {1, 4} also improve over GLA in aggregate, with the latter achieving the best FDA score among all models. MS-GLA {1, 2, 4, 8} outperforms the baseline overall but falls noticeably behind {1, 2, 4}, particularly on SWDE where the gap is most pronounced.

The strictly coarse-scale configuration {2, 4} again performs worst, dropping below the GLA baseline on SWDE and SQuAD and recording the lowest average F1 among all MS-GLA variants. The consistency of this failure across both language modeling and recall tasks confirms that local token-level processing is not merely complementary but essential i.e. coarser branches alone are insufficient for reliable in-context retrieval.

## 5.3 Long-Context Generalization

Figure 1 shows perplexity vs. position bucket for sequences extending up to 15× the training context length, demonstrating the model’s ability to handle sequences much longer than those seen during training. While GLA exhibits a persistent upward drift in perplexity beyond the mid-range token positions on both datasets, consistent with a recurrent state that progressively saturates under long-range context. All configurations that include a fine-resolution branch remain below the GLA baseline throughout both datasets. MS-GLA {1, 2, 4} and {1, 2, 4, 8} show the flattest profiles overall, sustaining their advantage at the furthest token positions with considerably less degradation than the baseline. On

![](images/b07a08294d6235af46eabb0624d50bfe55897366a06f5b34f72114a59e44de90.jpg)  
Figure 1: Perplexity vs. Position-bucket on two datasets, PG19 (left) and SlimPajama (right) illustrating long-context generalization up to 30,720 tokens.

<table><tr><td>Region</td><td>MS-GLA {1,2}</td><td>MS-GLA {1,4}</td><td>MS-GLA {2, 4}</td><td>MS-GLA {1,2, 4}</td><td>MS-GLA {1,2, 4, 8}</td></tr><tr><td>First 10%</td><td>[72.93, 27.07]</td><td>[69.89, 30.11]</td><td>[49.81, 50.19]</td><td>[52.64, 18.41, 28.95]</td><td>[47.98, 18.86, 12.33, 20.83]</td></tr><tr><td>Middle 10%</td><td>[73.45, 26.55]</td><td>[70.00, 29.99]</td><td>[49.40, 50.60]</td><td>[52.91, 18.34, 28.76]</td><td>[48.35, 18.58, 12.16, 20.92]</td></tr><tr><td>Last 10%</td><td>[72.96, 27.03]</td><td>[69.47, 30.53]</td><td>[49.25, 50.75]</td><td>[52.57, 18.41, 29.03]</td><td>[48.04, 18.43, 12.53, 21.00]</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: Average fusion routing weights (%) by scale, ordered finest to coarsest, at different positions within 32K-token evaluation sequences – roughly 15× the training context.

SlimPajama, the characteristic perplexity spike at the mid-sequence document boundary is substantially reduced for the better MS-GLA configurations, suggesting that coarser branches help smooth over abrupt distributional shifts in heterogeneous corpora.

Conversely, the strictly coarse-scale configuration {2, 4} diverges most severely of all models on both datasets, exhibiting a much steeper degradation than the GLA baseline beyond moderate sequence lengths. This confirms that temporal scale diversity alone does not confer long-context robustness; the fine-resolution branch is a critical architectural requirement.

## 5.4 Fusion Layer Routing Stability

To verify that the long-context robustness observed above reflects consistent multi-scale usage rather than the fusion layer collapsing onto a single branch far outside the training horizon, we measured average fusion routing weights over 32K-token evaluation sequences, comparing the first, middle, and final 10% of each sequence. Results are shown in Table 3.

Routing weights shift only trivially between the beginning and end of these sequences; for example, in MS-GLA {1, 2, 4} the fine-to-coarse split moves from [52.64%, 18.41%, 28.95%] in the first 10% to [52.57%, 18.41%, 29.03%] in the final 10%. This holds across all variants, confirming that the fusion layer does not collapse onto a single scale or become unstable well beyond the training horizon. Notably, the model consistently assigns the majority of its routing weight to the native, finest-resolution branch, reinforcing that this pathway is the dominant contributor to long-context robustness, rather than an artifact of reduced per-step capacity in the coarse-only {2, 4} configuration.

<table><tr><td>Model</td><td>TFLOPs</td><td>Throughput</td><td>Max mem. (GiB)</td><td>Train time</td></tr><tr><td>GLA</td><td>91.2</td><td>31,989</td><td>70.2</td><td>3d 1:49:25</td></tr><tr><td>MS-GLA {1, 2}</td><td>78.9</td><td>27,619</td><td>86.7</td><td>2d 18:23:08</td></tr><tr><td>MS-GLA {1, 2,4}</td><td>84.0</td><td>29,437</td><td>80.4</td><td>2d 22:48:41</td></tr><tr><td>MS-GLA {1, 2, 4, 8}</td><td>77.8</td><td>27,201</td><td>89.2</td><td>2d 23:43:25</td></tr><tr><td>MS-GLA {2, 4}</td><td>97.4</td><td>34,123</td><td>71.5</td><td>2d 9:11:04</td></tr></table>

Table 4: Training TFLOPs, throughput in tokens/sec, memory, and wall-clock time across GLA and MS-GLA variants, measured on the training runs described in Section 4.2.

## 5.5 Computational Efficiency

Since our central architectural claim rests on preserving GLA’s hardware-efficient chunkwise training, we report measured TFLOPs, throughput, memory, and wall-clock training cost for all variants in Table 4.

Coarser branches process proportionally shorter pooled sequences (⌈L/s⌉ tokens), which offsets much of the overhead introduced by additional branch projections, pooling, causal upsampling, and fusion, consistent with the structural argument in Section 3.4. MS-GLA {1, 2, 4}, our best-performing configuration, retains roughly 92% of GLA’s throughput at only ∼14% higher active memory, while the purely coarse {2, 4} configuration exceeds baseline throughput outright since its branches never operate at full sequence length. Taken together with the accuracy and robustness gains reported above, these numbers indicate that MS-GLA’s improvements come at a modest, structurally bounded cost rather than through brute-force scaling of compute or memory.

## 6 Discussion

Across all three evaluation settings, MS-GLA {1, 2, 4} consistently achieves the best overall results, suggesting that three scales including the finest resolution strike the optimal balance between multi-resolution coverage and per-branch representational capacity.

## 6.1 The Necessity of Fine-Resolution Branches

The most consistent finding across language modeling, recall tasks, and long-context generalization is the decisive role of the finest resolution branch (s = 1). All configurations that retain at least one branch at native token resolution outperform the GLA baseline in aggregate, while MS-GLA {2, 4}, which omits this branch entirely, degrades across every evaluation setting. Its LAMBADA perplexity collapses to 31.47, its recall F1 falls below the GLA baseline on SWDE and SQuAD, and its long-context perplexity diverges most severely beyond 10K tokens. This pattern strongly suggests that local syntactic processing at the token level is not merely complementary to coarser temporal modeling but is essential i.e. coarse branches alone cannot compensate for the absence of a fine-grained pathway. The long-context results make this particularly clear: without a fine-resolution branch, the model loses the ability to maintain coherent token-level predictions when forced to rely entirely on coarse temporal summaries beyond its training horizon.

## 6.2 Scale Coverage vs. Branch Capacity

The comparison between {1, 2, 4} and {1, 2, 4, 8} illustrates a trade-off between scale coverage and per-branch representational capacity. Adding the coarsest scale (s = 8) provides marginal or no benefit on language modeling and recall tasks, and the four-scale configuration regresses relative to the three-scale model on both SWDE and FDA (see Table 2). This is consistent with head-budget dilution: spreading a fixed head allocation across four scales reduces each branch’s representational capacity below the threshold needed for reliable local retrieval. The {1, 2, 4, 8} configuration does remain competitive on long-context tasks, however, where the additional coarse scale may provide modest complementary benefit for very distant dependencies.

MS-GLA {1, 4} presents a complementary case: skipping the intermediate scale (s = 2) yields the strongest performance on Wikitext perplexity and HellaSwag while lagging on Lambada and average perplexity, suggesting that the resolution gap hurts short-to-medium range coherence even as it benefits broader structural modeling. The one benchmark where the fine-resolution advantage disappears entirely is Winogrande, where GLA and MS-GLA {2, 4} score competitively with the better configurations. This is consistent with the nature of the task: Winogrande probes short-range commonsense coreference that is less sensitive to long-range temporal structure, and therefore unlikely to benefit from multi-scale decomposition regardless of which scales are included.

## 6.3 Long-Context Behavior and Temporal Scale

The long-context results offer the clearest window into how distributing recurrent computation across multiple temporal scales affects generalization beyond the training horizon. GLA exhibits a persistent upward drift beyond 5K tokens on PG19, consistent with a recurrent state that progressively saturates as context accumulates. MS-GLA {1, 2, 4} and {1, 2, 4, 8}, by contrast, maintain notably flatter profiles, suggesting that distributing computation across scales prevents long-range compression from bottlenecking into a single resolution. On SlimPajama, the characteristic perplexity spike near the 12–13K token document boundary is substantially reduced for the better MS-GLA configurations, indicating that coarser branches help smooth over abrupt distributional shifts in heterogeneous corpora. This benefit is contingent on retaining a fine-resolution branch, however: coarse branches alone not only fail to improve long-context generalization but actively degrade out-of-distribution performance relative to GLA. This is corroborated by the routing analysis in Section 5.4, which shows the learned fusion router consistently favoring the finest-resolution branch even at 15× the training context, rather than drifting or collapsing under distribution shift.

## 7 Future Work

While our current evaluations establish the efficacy of MS-GLA at the 340M parameter scale, an essential next step is evaluating its scaling behavior at multi-billion parameter regimes and over extended training horizons. Beyond scale, MS-GLA opens several promising directions. Rather than relying on fixed temporal pooling, future work could explore dynamic semantic resolutions, where the model autonomously aggregates tokens based on linguistic constituents or informational density instead of fixed positional windows. Temporal resolution could also be allowed to vary across layers rather than sharing one scale set network-wide, letting lower layers specialize toward local syntax and higher layers toward semantic structure. Finally, since MS-GLA is a drop-in replacement for standard GLA layers, it is directly compatible with hybrid architectures that interleave linear and softmax attention, feeding pooled multi-scale tokens to periodic softmax layers as a natural extension.

## 8 Conclusion

While data-dependent gating has significantly advanced linear attention, our work exposes and resolves a critical remaining limitation: the single-resolution memory bottleneck. We introduced Multi-Scale Gated Linear Attention (MS-GLA) to demonstrate that decoupling recurrent state updates across diverse temporal scales as a potentially effective, solution. By yielding consistent improvements in language modeling, recall accuracy, and long-context extrapolation over the GLA baseline, our findings establish multi-resolution decomposition as a vital architectural bias for linear sequence models. Ultimately, MS-GLA successfully bridges the gap between the expressive power of multi-scale representation and the hardware efficiency of chunkwise training, offering a robust foundation for scaling efficient transformers.

## References

Naman Agarwal, Daniel Suo, Xinyi Chen, and Elad Hazan. Spectral state space models, 2024. URL https://arxiv.org/abs/2312.06837.

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, Dylan Zinsley, James Zou, Atri Rudra, and Christopher Re. Simple linear attention language models ´ balance the recall-throughput tradeoff. arXiv:2402.18668, 2024.

Simran Arora, Brandon Yang, Sabri Eyuboglu, Avanika Narayan, Andrew Hojel, Immanuel Trummer, and Christopher Re. Language models enable simple systems for generating ´ structured views of heterogeneous data lakes, 2025. URL https://arxiv.org/abs/2304. 09433.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence, 2020.

Jacob Buckman and Carles Gelada. Linear transformers are faster after all. URL https: //manifestai.com/blogposts/faster-after-all/#linear-transformers.

Charlotte Caucheteux, Alexandre Gramfort, and Jean-Remi King. Evidence of a predictive´ coding hierarchy in the human brain listening to speech. Nature Human Behaviour, 7:1–12, 03 2023. doi: 10.1038/s41562-022-01516-2.

Jusen Du, Weigao Sun, Disen Lan, Jiaxi Hu, and Yu Cheng. Mom: Linear sequence modeling with mixture-of-memories, 2025. URL https://arxiv.org/abs/2502.13685.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces, 2024. URL https://arxiv.org/abs/2312.00752.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, and Laurent Sifre. Training compute-optimal large language models, 2022. URL https://arxiv.org/abs/2203.15556.

Weizhe Hua, Zihang Dai, Hanxiao Liu, and Quoc Le. Transformer quality in linear time. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp. 9099–9117. PMLR, 17–23 Jul 2022. URL https://proceedings.mlr.press/v162/hua22a.html.

Mahdi Karami, Ali Behrouz, Peilin Zhong, Razvan Pascanu, and Vahab Mirrokni. MS-SSM: A multi-scale state space model for efficient sequence modeling, 2025. URL https: //openreview.net/forum?id=cCYWeCzAv0.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns: fast autoregressive transformers with linear attention. In Proceedings of the 37th International Conference on Machine Learning, ICML’20. JMLR.org, 2020.

Colin Lockard, Prashant Shiralkar, and Xin Luna Dong. OpenCeres: When open information extraction meets the semi-structured web. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3047–3056, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1309. URL https://aclanthology.org/N19-1309/.

Ilya Loshchilov and Frank Hutter. Fixing weight decay regularization in adam. 11 2017. doi: 10.48550/arXiv.1711.05101.

Anton Lozhkov, Loubna Ben Allal, Leandro von Werra, and Thomas Wolf. Fineweb-edu: the finest collection of educational content, 2024. URL https://huggingface.co/datasets/ HuggingFaceFW/fineweb-edu.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture models. In International Conference on Learning Representations, 2017. URL https: //openreview.net/forum?id=Byj72udxe.

G. P. Nason and B. W. Silverman. The stationary wavelet transform and some statistical applications. Lecture Notes in Statistics, pp. 281–299, 1995. doi: 10.1007/978-1-4612-2544-7 17.

Piotr Nawrot, Szymon Tworkowski, Michał Tyrolski, Łukasz Kaiser, Yuhuai Wu, Christian Szegedy, and Henryk Michalewski. Hierarchical transformers are more efficient language models, 2022. URL https://arxiv.org/abs/2110.13711.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella ´ Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. URL http://www.aclweb.org/anthology/P16-1144.

Jack W. Rae, Anna Potapenko, Siddhant M. Jayakumar, Chloe Hillier, and Timothy P. Lillicrap. Compressive transformers for long-range sequence modelling. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id= SylKikSYDH.

Pranav Rajpurkar, Jian Zhang, and Percy Liang. Know what you don’t know: Unanswerable questions for squad. In ACL 2018, 2018.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: an adversarial winograd schema challenge at scale. Commun. ACM, 64(9):99–106, August 2021. ISSN 0001-0782. doi: 10.1145/3474381. URL https://doi.org/10.1145/3474381.

Jiaxin Shi, Ke Alexander Wang, and Emily B. Fox. Sequence modeling with multiresolution convolutional memory. In Proceedings of the 40th International Conference on Machine Learning, ICML’23. JMLR.org, 2023.

Daria Soboleva, Faisal Al-Khateeb, Robert Myers, Jacob R Steeves, Joel Hestness, and Nolan Dey. SlimPajama: A 627B token cleaned and deduplicated version of RedPajama. https://www.cerebras.net/blog/ slimpajama-a-627b-token-cleaned-and-deduplicated-version-of-redpajama, 2023. URL https://huggingface.co/datasets/cerebras/SlimPajama-627B.

Yutao Sun, Li Dong, Shaohan Huang, Shuming Ma, Yuqing Xia, Jilong Xue, Jianyong Wang, and Furu Wei. Retentive network: A successor to transformer for large language models, 2024. URL https://openreview.net/forum?id=UU9Icwbhin.

Alex Tamkin, Dan Jurafsky, and Noah Goodman. Language through a prism: A spectral approach for multiscale language representations. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 5492–5504. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/ paper files/paper/2020/file/3acb2a202ae4bea8840224e6fce16fd0-Paper.pdf.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Songlin Yang and Yu Zhang. Fla: A triton-based library for hardware-efficient implementations of linear attention mechanism, January 2024. URL https://github.com/fla-org/ flash-linear-attention.

Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, and Yoon Kim. Gated linear attention transformers with hardware-efficient training. In Proceedings of ICML, 2024.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.

Yu Zhang and Songlin Yang. Flame: Flash language modeling made easy, January 2025. URL https://github.com/fla-org/flame.

## A Limitations

We note several limitations of this work, most arising from compute constraints (a single GPU, further bottlenecked by hardware issues during the research window; full hardware details in Appendix B).

Model scale. All experiments are conducted at the 340M-parameter, 7B-token scale. While MS-GLA’s added operations (pooling, upsampling, fusion) scale linearly with sequence length and we have no architectural reason to expect divergent behavior at larger scales, this has not been empirically validated beyond 340M parameters.

Baseline scope. Our comparisons are scoped to the GLA family rather than including freshly trained RetNet, Mamba, or Transformer baselines under our exact setup. We rely on the original GLA paper’s comparisons against these architectures, and note that MS-GLA’s consistent gains over GLA should transitively suggest competitiveness with these broader baselines, though this has not been directly measured.

Single seed. Due to compute constraints, all results are reported from a single training run per configuration (seed 42; see Appendix B), without multiple seeds or significance testing.

Ablation coverage. We ablate the number and choice of temporal scales extensively, but did not ablate the pooling operator (e.g., average vs. max vs. last-token pooling) or alternative fusion mechanisms (e.g., uniform averaging in place of the learned softmax router), which we leave to future work.

## B Training Details

This appendix provides the full training configuration used for all models (GLA baseline and MS-GLA variants) reported in the paper, supplementing the summary in Section 4.2. Exact launch commands are included in the released train.sh script in our repository.

Optimization. We use the AdamW optimizer (Loshchilov & Hutter, 2017) with ϵ = $1 \stackrel { - } { \times } 1 0 ^ { - 1 5 }$ , weight decay 0.1, and a peak learning rate of $3 \times 1 0 ^ { - 4 }$ . The learning rate follows a cosine decay schedule with a minimum learning-rate ratio of 0.1, preceded by 3,400 warmup steps. Gradient clipping is applied at a maximum norm of 1.0, and non-finite (NaN/Inf) gradient steps are skipped.

Data and batching. All models are trained from scratch on the FineWeb-Edu corpus (Lozhkov et al., 2024), tokenized with the tokenizer from fla-hub/transformer-1.3B-100B. We use a sequence length of 2,048 tokens, a batch size of 32, and a gradient accumulation of 1 step. The GLA baseline is trained for 68,664 steps; MS-GLA variants are trained for 106,813 steps to match the 7B-token budget reported in Section 4.2.

Reproducibility. All runs use a fixed random seed of 42. Model checkpoints are saved every 8,096 steps, and training metrics are logged every 10 steps.

Hardware and parallelism. All models were trained on a single NVIDIA RTX PRO 6000 Max-Q GPU (NGPU=1, NNODE=1), part of a workstation equipped with an AMD Ryzen Threadripper PRO 7975WX 32-core CPU and 512GB of RAM; tensor parallelism and loss parallelism are disabled (tensor parallel degree=1), and models are compiled prior to training. The workstation’s power supply proved inadequate over parts of the research window, which constrained the overall compute budget available for this project and informed the scoping decisions discussed in Appendix A. A single 340M-parameter training run required approximately 2–3 days depending on configuration (Table 4).

Framework. All models are implemented and trained using the flash-linear-attention library (Yang & Zhang, 2024) and the flame training framework (Zhang & Yang, 2025), built on torchtitan.

Code. A cleanly documented, modular implementation – including scale-specific GLA branches, causal pooling/upsampling utilities, the fusion layer, and the exact training launch script (train.sh) with per-model configuration files – is released at https://github. com/prasoondev/msgla.