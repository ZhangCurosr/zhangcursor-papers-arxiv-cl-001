# LNGRAM V2: LATENT N-GRAM MEMORY WITH IN-TERPRETABLE DISCRETE REPRESENTATIONS

Yunao Zheng<sup>1</sup> Bin Wen<sup>2,\*,†</sup> Xiaojie Wang<sup>1,†</sup>

## ABSTRACT

Transformers lack a native lookup mechanism, requiring repeated dense computation to recognize and reuse local static patterns. Lngram v1 introduces tokenizerindependent conditional memory through discrete latent n-gram addressing, but its memory capacity is coupled with the backbone width, limiting scalability due to high parameter and activation costs. We propose Lngram v2, which decouples the number of routes, memory dimension, and backbone width, and introduces a context-aware grouped-query attention readout to scale memory capacity independently. A zero-value Sink and counterfactual surrogate gradients further improve readout selectivity and routing trainability while preserving hard discrete addressing. Experiments across vision–language models (VLMs) of different scales show consistent improvements, including successful scaling to a 30B-parameter model. Compared with Lngram v1, Lngram v2 substantially reduces both total and activated memory parameters while maintaining or improving language modeling performance. Further analysis shows that its discrete IDs preserve substantial semantic structure of continuous hidden states, enabling semantic recovery from IDs alone and stable ID–semantic associations across datasets. These results establish Lngram v2 as an efficient and scalable latent conditional memory mechanism whose discrete addresses also provide a structured interface for analyzing internal model representations.

## 1 INTRODUCTION

Transformers have become the dominant backbone for large language and multimodal models (Vaswani et al., 2017; Devlin et al., 2019; Brown et al., 2020; Dosovitskiy et al., 2021; Radford et al., 2021; Alayrac et al., 2022). However, attention and feed-forward networks must handle both context-dependent reasoning and operations closer to conditional lookup, such as recognizing common multi-token entities and recurring local patterns. Since standard Transformers lack a native lookup mechanism, such patterns must still be approximated through dense computation, consuming model capacity that could otherwise support compositional reasoning (Cheng et al., 2026).

To separate local pattern matching from the backbone, Cheng et al. (2026) introduced conditional memory based on token-ID n-grams. Its addresses, however, depend on tokenizer IDs and fixed hashing, making the mechanism sensitive to tokenization boundaries, hash collisions, and difficult to extend uniformly beyond text. Lngram v1 instead discretizes hidden states into learnable symbols and constructs exact n-gram addresses in latent space, extending classical n-gram modeling (Brants et al., 2007) to neural representations and enabling conditional memory over textual, visual, and cross-modal inputs.

Lngram v1 nevertheless has two limitations. First, its route count is coupled with backbone width, causing memory size and readout cost to increase with model dimension and limiting scalability. Second, although discrete Lngram IDs directly determine memory addresses, it remains unclear whether these IDs preserve the semantic structure of the continuous representations from which they are derived. Existing representation-analysis methods typically require continuous activations together with probes or learned dictionaries (Alain & Bengio, 2016; Hewitt & Manning, 2019; Tenney et al., 2019; Bau et al., 2017; Huben et al., 2024). If Lngram IDs themselves retain semantic information, they may provide a useful interface for both memory addressing and representation analysis.

We therefore propose Lngram $\mathbf { v } 2 ,$ , which decouples route count, memory dimension, and backbone width so that memory capacity can scale independently. Retrieved memories of different n-gram orders are organized as memory tokens, and the current hidden state performs context-dependent readout through grouped-query attention (GQA) (Ainslie et al., 2023). A fixed zero-value Sink allows the model to suppress unreliable retrievals, while route-wise streaming controls intermediate activations at large scale. To train hard discrete routing, we use a counterfactual-lookup surrogate gradient that provides memory-dependent optimization signals while keeping the forward addressing path identical during training and inference, following the general principle of surrogate optimization for discrete decisions (Bengio et al., 2013).

We further study the semantics encoded in Lngram’s discrete representations using an ID-only semantic reader that receives only the multi-route discrete signature. The reader retains 65.77%– 84.27% of the excess semantic readout capability of continuous hidden states. On held-out data, individual route codes exhibit reproducible associations with concepts such as people, animals, vehicles, and common objects, while most semantic capacity arises from distributed combinations across routes. These results suggest that Lngram IDs serve not only as memory addresses but also as a statistically analyzable discrete interface to internal representations.

We evaluate Lngram v2 on vision–language models at multiple scales. Under matched training data, compute budgets, and backbone optimization settings, it consistently improves Keye2B and Keye30B and scales successfully to a 30B-parameter model. Parameter-matched Sparse FFN and Product Key Memory (PKM) baselines show that the gains cannot be explained by additional parameters or sparse capacity alone. Compared with Lngram v1, Lngram v2 reduces the module’s total parameters by 82.6% and activated parameters per token by 95.2%, while maintaining or improving validation loss. Overall, Lngram $\mathbf { v } 2$ provides a scalable latent-space conditional memory mechanism together with a structured discrete interface for analyzing model representations.

## 2 ARCHITECTURE

## 2.1 OVERVIEW

As illustrated in Figure 1, Lngram $\mathbf { v } 2$ is a conditional memory branch inserted into Transformer decoder layers to offload local pattern matching and storage from the backbone. Omitting the batch dimension, the input to layer ℓ is

$$
H ^ { ( \ell ) } = [ h _ { 1 } , \ldots , h _ { T } ] ^ { \top } \in \mathbb { R } ^ { T \times d } .
$$

Lngram ${ \bf v } 2$ maps each hidden state to multiple discrete symbols, constructs local n-gram addresses for exact memory lookup, and organizes the retrieved entries as route-wise memory tokens. The current hidden state then performs context-dependent readout over these tokens through groupedquery attention (GQA), augmented with a fixed zero-value Sink that suppresses unreliable retrievals. The output is further processed by a short-range causal convolution and injected into the backbone as a residual.

The forward pass always uses hard discrete addressing, while the discretization projection is trained with counterfactual surrogate gradients. For large route counts, the readout can be streamed along the route dimension to reduce activation memory.

## 2.2 MULTI-ROUTE DISCRETE ADDRESSING

Lngram v2 discretizes the hidden states through an independent projection:

$$
U = \operatorname { R M S N o r m } ( H ) , \qquad Z = U W _ { q } , \qquad W _ { q } \in \mathbb { R } ^ { d \times R M } ,
$$

where R is the number of routes and M is the number of bits per route. For position t, route $r \in \{ 0 , \ldots , R - 1 \}$ , and bit $j \in \{ 0 , \dots , M - 1 \}$ ,

$$
b _ { t , r , j } = \mathbb { I } [ z _ { t , r , j } > 0 ] ,
$$

![](images/c2eea9da8bf4201718fefbbd500f6ccb9233e3da9d68b36f7fdc28e817f5c27c.jpg)  
Figure 1: Overall architecture of Lngram $\mathbf { v } 2 .$

and the bits are packed into a discrete symbol

$$
a _ { t , r } = \sum _ { j = 0 } ^ { M - 1 } b _ { t , r , j } 2 ^ { j } , \qquad a _ { t , r } \in \{ 0 , \dots , K - 1 \} , \quad K = 2 ^ { M } .
$$

Each hidden state is thus mapped to R parallel discrete symbols. Since the addresses are derived from latent representations rather than tokenizer IDs, the same mechanism applies across modalities.

## 2.3 EXACT N-GRAM RETRIEVAL

For each order $n \in \mathcal N$ , Lngram v2 maintains a memory table

$$
\boldsymbol { E } ^ { ( n ) } \in \mathbb { R } ^ { R K ^ { n } \times d _ { m } } ,
$$

where $d _ { m }$ is the memory-entry dimension. For route $r \in \{ 0 , \ldots , R - 1 \}$ , the n-gram ending at position t is encoded as

$$
g _ { t , r } ^ { ( n ) } = r K ^ { n } + \sum _ { i = 0 } ^ { n - 1 } a _ { t - n + 1 + i , r } K ^ { i } ,
$$

and retrieved exactly by

$$
m _ { t , r } ^ { ( n ) } = E ^ { ( n ) } [ g _ { t , r } ^ { ( n ) } ] .
$$

Different routes occupy disjoint address ranges, yielding a unique entry for each route–n-gram combination without approximate search. Invalid prefixes and n-grams crossing packed-sequence boundaries are masked.

The retrieved entries of different orders are concatenated into

$$
\boldsymbol { x } _ { t , r } = \mathrm { C o n c a t } _ { n \in \mathcal { N } } m _ { t , r } ^ { ( n ) } \in \mathbb { R } ^ { | \mathcal { N } | d _ { m } } ,
$$

producing R memory tokens per position.

## 2.4 CONTEXT-AWARE READOUT

Lngram v2 uses the current hidden state to contextually select among the retrieved route tokens. Let $H _ { q }$ and $H _ { k v }$ denote the numbers of query and KV heads, with head dimension $d _ { h } \mathbf { . }$

$$
H _ { q } d _ { h } = d , \qquad G = \frac { H _ { q } } { H _ { k v } } .
$$

The query and memory keys/values are

$$
Q _ { t } = { \mathrm { R e s h a p e } } \left( { \mathrm { N o r m } } ( h _ { t } ) W _ { Q } \right) \in \mathbb { R } ^ { H _ { q } \times d _ { h } } ,
$$

$$
[ K _ { t , r } ; V _ { t , r } ] = \mathrm { R e s h a p e } \left( \mathrm { N o r m } ( x _ { t , r } ) W _ { K V } \right) , \qquad K _ { t , r } , V _ { t , r } \in \mathbb { R } ^ { H _ { k v } \times d _ { h } } .
$$

For query head $a \in \{ 0 , \ldots , H _ { q } - 1 \}$ , define

$$
\kappa ( a ) = \left\lfloor { \frac { a } { G } } \right\rfloor ,
$$

and compute

$$
s _ { t , a , r } = \frac { Q _ { t , a } ^ { \top } K _ { t , r , \kappa ( a ) } } { \sqrt { d _ { h } } } .
$$

To avoid forcing probability mass onto unreliable memories, we introduce a zero-value Sink with fixed logit $b _ { 0 } { \mathrm { : } }$

$$
\alpha _ { t , a , r } = \frac { \exp ( s _ { t , a , r } ) } { \exp ( b _ { 0 } ) + \sum _ { r ^ { \prime } = 0 } ^ { R - 1 } \exp ( s _ { t , a , r ^ { \prime } } ) } , \qquad o _ { t , a } = \sum _ { r = 0 } ^ { R - 1 } \alpha _ { t , a , r } V _ { t , r , \kappa ( a ) } .
$$

We set

$$
b _ { 0 } = \log \frac { R ( 1 - \mu _ { 0 } ) } { \mu _ { 0 } } ,
$$

where $\mu _ { 0 }$ specifies the initial total probability mass assigned to memory routes.

The head outputs are projected and refined by a depthwise causal convolution:

$$
\widetilde { \boldsymbol { v } } _ { t } = \mathrm { C o n c a t } _ { a = 0 } ^ { H _ { q } - 1 } \boldsymbol { o } _ { t , a } \boldsymbol { W _ { O } } ,
$$

and

$$
\widetilde { V } = [ \widetilde { v } _ { 1 } , \dots , \widetilde { v } _ { T } ] ^ { \top } .
$$

The sequence representation is then refined as

$$
Y = \widetilde { V } + \mathrm { S i L U } \left( \mathrm { D W C o n v 1 D } \left( \mathrm { N o r m } ( \widetilde { V } ) \right) \right) ,
$$

with kernel size 4. The result is injected as a pre-attention residual:

$$
H ^ { ( \ell ) } \gets H ^ { ( \ell ) } + Y .
$$

For large $R ,$ the route dimension can be processed in blocks while accumulating the softmax normalization and value-weighted sum online, avoiding simultaneous storage of all route activations.

## 2.5 BACKPROPAGATION THROUGH DISCRETE ADDRESSING

The forward pass always uses hard decisions

$$
b _ { t , r , j } = \mathbb { I } [ z _ { t , r , j } > 0 ] ,
$$

which are non-differentiable. We therefore train the discretization projection using counterfactual memory lookups while preserving the hard forward path.

For one symbol in an n-gram window, let

$$
z = ( z _ { 0 } , \ldots , z _ { M - 1 } ) , \qquad p _ { j } = \sigma ( \tau z _ { j } ) .
$$

Holding the remaining symbols fixed, replacing this symbol by $c \in \{ 0 , \ldots , K - 1 \}$ yields a counterfactual retrieval $E ^ { ( c ) }$ . Its probability is

$$
P ( c \mid z ) = \prod _ { j = 0 } ^ { M - 1 } p _ { j } ^ { \beta _ { j } ( c ) } ( 1 - p _ { j } ) ^ { 1 - \beta _ { j } ( c ) } ,
$$

giving the conditional expectation

$$
\widetilde E ( z ) = \sum _ { c = 0 } ^ { K - 1 } P ( c \mid z ) E ^ { ( c ) } .
$$

This expectation is used only for the routing gradient; the actual forward output remains the hard lookup $\bar { E } _ { \mathrm { h a r d } }$

In large-scale experiments, we use a one-bit approximation. For bit $j ,$ let $E _ { i } ^ { ( 0 ) }$ and $E _ { j } ^ { ( 1 ) }$ denote the counterfactual retrievals obtained by setting that bit to 0 or 1 while fixing all others. Given upstream gradient $^ { g , }$

$$
\frac { \partial \mathcal { L } } { \partial z _ { j } } \approx \lambda \tau p _ { j } ( 1 - p _ { j } ) \left. g , E _ { j } ^ { ( 1 ) } - E _ { j } ^ { ( 0 ) } \right. .
$$

Only the backward gradient of the discretization projection is replaced: memory entries selected by hard lookup receive standard gradients, whereas counterfactual entries are used solely to estimate routing gradients. Thus, training and inference share exactly the same hard addressing and exactlookup forward path. Further derivations and implementation details are provided in Appendix A.

## 3 ANALYSIS

Lngram ${ \bf v } 2$ maps continuous hidden states to discrete IDs for memory addressing. We investigate whether these IDs preserve the semantics of the original representations, whether such semantics can be recovered without accessing continuous hidden states, and whether they are localized to individual route codes or distributed across multiple routes.

## 3.1 EXPERIMENTAL SETUP

We analyze two Lngram layers of a 2B Keye2B+Lngram $\mathbf { v } 2$ model trained for approximately 20B additional tokens (Section 4). Each layer contains 128 routes with 16 codes per route, yielding a discrete signature

$$
C = ( c _ { 1 } , \dots , c _ { 1 2 8 } )
$$

for each token.

We use 10,000 COCO 2017 images (Lin et al., 2014), split into 7,000 reference and 3,000 heldout images. All statistical criteria and readers are determined exclusively on the reference set and evaluated independently on the held-out set. We consider three semantic targets: caption mention, visual presence, and local presence, corresponding to concepts expressed in textual descriptions, the full image, and local image regions, respectively.

We first verify that the discrete space does not collapse. Across the two analyzed layers, each route uses approximately eight effective codes on average, the most frequent code accounts for about 30% of occurrences, and all 4,096 theoretically possible route codes across the two layers are observed. Complete statistics are provided in Appendix C.1.

## 3.2 READING HIDDEN-STATE SEMANTICS FROM LNGRAM IDS ALONE

Lngram v2 uses highly sparse discrete addressing while consistently improving downstream performance, suggesting that its discrete mapping may preserve stable semantic structure. While conventional representation analyses recover information from continuous hidden states using additional probes (Alain & Bengio, 2016), we ask whether the continuous representation can be removed entirely and its semantics recovered solely from Lngram IDs.

For each held-out query, we provide only its 128-route signature C and retrieve the $K = 1 0$ nearest distinct reference documents by route-Hamming distance. Nearest-neighbor prediction (Cover & Hart, 1967) is then used for caption mention, visual presence, and local presence. KNN based on the continuous hidden states $H _ { \mathrm { m o d e l } }$ serves as the reference; we additionally report $Q _ { \mathrm { r o u t e } }$ as a control and predictions using only nuisance information, such as image source and spatial position, as the baseline.

Table 1: Semantic readout using only Lngram IDs.
<table><tr><td>Layer</td><td>Task</td><td> $H _ { \mathrm { m o d e l } }$ </td><td> $Q _ { \mathrm { r o u t e } }$ </td><td>ID-KNN</td><td>nuisance</td><td>Excess AP Retained</td></tr><tr><td>1</td><td>caption mention</td><td>0.239744</td><td>0.225916</td><td>0.200077</td><td>0.016789</td><td>82.21%</td></tr><tr><td>1</td><td>local presence</td><td>0.470371</td><td>0.450705</td><td>0.396941</td><td>0.008484</td><td>84.10%</td></tr><tr><td>1</td><td>visual presence</td><td>0.317607</td><td>0.302142</td><td>0.273789</td><td>0.039000</td><td>84.27%</td></tr><tr><td>13</td><td>caption mention</td><td>0.206947</td><td>0.177058</td><td>0.152670</td><td>0.016789</td><td>71.46%</td></tr><tr><td>13</td><td>local presence</td><td>0.403160</td><td>0.328081</td><td>0.268083</td><td>0.008493</td><td>65.77%</td></tr><tr><td>13</td><td>visual presence</td><td>0.286478</td><td>0.250947</td><td>0.221730</td><td>0.039000</td><td>73.84%</td></tr></table>

To quantify how much readable semantic information is retained after discretization, we define

$$
\rho _ { I D  H } = \frac { \overline { { A P } } _ { I D } - \overline { { A P } } _ { N } } { \overline { { A P } } _ { H } - \overline { { A P } } _ { N } } ,\tag{1}
$$

where N denotes the nuisance baseline.

Table 1 shows that discrete IDs recover a substantial fraction of the readable semantics in continuous hidden states. Layer 1 retains 82.21%–84.27% of the excess AP of $H _ { \mathrm { m o d e l } }$ , while Layer 13 retains 65.77%–73.84%. All six settings exhibit the consistent ordering

$$
H _ { \mathrm { m o d e l } } > Q _ { \mathrm { r o u t e } } > \mathrm { I D } \gg N ,
$$

indicating that discretization incurs some information loss but preserves most of the readable semantic structure.

To exclude nuisance-driven correlations, we additionally perform 500 document-level permutations of the reference signatures under the same nuisance conditions. In all six settings, the observed AP exceeds every permuted result. Using the Monte Carlo p-value for random permutation tests (Phipson & Smyth, 2010), all settings yield

$$
p = 1 / 5 0 1 = 0 . 0 0 1 9 9 6 .
$$

Thus, the ID-only readout cannot be explained by biases such as image source or spatial position.

We next ask whether semantic information can already be localized to individual route codes. For a code

$$
\boldsymbol { u } = ( l , r , c ) ,
$$

we estimate its association with semantic concepts after controlling for nuisance variables. Candidate associations and their effect directions are determined strictly on the reference set and independently evaluated on the held-out set. Among 16,304 candidate associations discovered on the reference set, 14,002 are evaluable on the held-out set, of which 9,220 satisfy consistent effect direction, confidence-interval, and Benjamini–Hochberg multiple-testing criteria (Benjamini & Hochberg, 1995). This corresponds to an overall replication rate of 65.85%; for local presence, the rates reach 83.01% and 84.94% in Layers 1 and 13, respectively.

These results show that individual route codes can form reproducible concept-level associations. Appendix C.2 provides 15 representative examples. For instance,

$$
C _ { 1 , 7 5 } = 1 0
$$

is associated with giraffe: on the held-out set, the conditional probability increases from a nuisancecontrolled baseline of 2.37% to 21.70%, a gain of 19.33 percentage points. Similarly,

$$
C _ { 1 3 , 5 3 } = 1 4
$$

is associated with person, whose conditional probability increases from 53.60% to 63.25%. These examples illustrate the discovered structure, while all reported replication and significance statistics follow the complete reference–held-out validation protocol.

Overall, Lngram IDs already form a queryable semantic interface: the complete signature recovers much of the semantics encoded in the corresponding hidden state without accessing its continuous representation, while individual route codes exhibit stable and interpretable concept associations.

## 3.3 DISTRIBUTED SEMANTIC REPRESENTATION ACROSS ROUTE CODES

Although individual route codes exhibit stable semantic preferences, the complete signature provides substantially stronger readout capability. To determine whether this gap arises from joint encoding across routes, we recombine the marginal semantic effects of individual codes estimated on the reference set and measure how much semantic information can be recovered using different numbers of routes. This analysis uses only discrete IDs on the held-out set, without continuous representations or additional fitted parameters; details are provided in Appendix C.3.

Simply accumulating the marginal semantic effects of individual codes retains 31.79%–54.82% of the excess AP of $H _ { \mathrm { m o d e l } }$ across the six settings, confirming that individual codes contain composable semantic information. However, this remains substantially below the 65.77%–84.27% achieved by ID-KNN using the complete signature. Moreover, even when routes are prioritized by their contribution to each query–concept pair, using all 128 routes consistently outperforms using only the single best route, with AP ratios ranging from 2.02 to 12.32.

These results indicate that Lngram semantics are not concentrated in a small number of specialized routes, but are distributed across joint route configurations, consistent with distributed representations in neural networks (Hinton et al., 1986). Individual route codes provide local and interpretable semantic tendencies, while the complete signature captures richer compositional structure across routes. Lngram IDs therefore serve not only as memory addresses but also as a statistically analyzable, queryable, and reproducible semantic interface to internal model representations.

## 4 EXPERIMENTS

This section evaluates the effectiveness, architectural design, and parameter efficiency of Lngram v2. We first evaluate Lngram v2 on the 2B and 30B variants of Keye2-VL (Kwai Keye Team et al., 2026), with additional comparisons against Product Key Memory (PKM) and a parameter-matched Sparse FFN on the 2B model. We then study the main architectural hyperparameters, the Sink, and inference efficiency. Finally, we compare Lngram v2 with v1 to assess whether v2 improves parameter efficiency while preserving or improving language modeling performance.

## 4.1 MAIN RESULTS

We evaluate Lngram v2 on Keye2B and Keye30B. All models continue training from the same respective base models for approximately 5B tokens with identical training data and backbone optimization settings. Lngram v2 is inserted into a shallow and an intermediate layer and uses 4-bit discrete codes with 2/3-gram memory. On Keye2B, it adds approximately 155.6M parameters and 20.9M activated parameters per token, corresponding to 6.11% and 0.82% of the base model, respectively. On Keye30B, the corresponding increases are approximately 2.0% and 0.85%. Detailed configurations are provided in Appendix E. For the Keye2B Baseline and Lngram v2 results, we use three seeds (19260818, 19260819, and 19260820) and report the mean ± standard deviation.

We evaluate on public vision–language benchmarks including OCRBench (Liu et al., 2024b), Math-Vista (Lu et al., 2024), MM-Vet (Yu et al., 2024), HallusionBench (Guan et al., 2024), MM-Star (Chen et al., 2024), MMMU (Yue et al., 2024), AI2D (Kembhavi et al., 2016), and MM Bench (Liu et al., 2024a).

For architecture-level comparison with existing memory mechanisms, we additionally evaluate Product Key Memory (PKM) (Lample et al., 2019) on Keye2B. Conditional memory methods such as Engram rely on text token IDs and are tied to a particular vocabulary, making them difficult to apply directly to visual inputs (Cheng et al., 2026), whereas Lngram is designed to support arbitrary modalities. We also construct a parameter-matched Sparse FFN to determine whether the gains can be explained by additional parameters or sparse computational capacity. Specifically, a top-1 Sparse SwiGLU residual branch (Shazeer, 2020; Fedus et al., 2022; Dai et al., 2024) is inserted into the same 2nd and 14th decoder layers as Lngram v2, without discrete addressing, n-grams, memory tables, or lookup. It adds 155.845M parameters and 20.972M activated parameters per token, closely matching Lngram v2.

Table 2: Comparison results on Keye2B and Keye30B. Keye2B Baseline and Lngram $\mathbf { v } 2$ results are reported as mean ± standard deviation over three seeds; other entries are single-run results.
<table><tr><td>Backbone</td><td>Model</td><td>OCR</td><td>MathVista</td><td>MMVet</td><td>Hallusion</td><td>MMStar</td><td>MMMU</td><td>AI2D</td><td>MMB-EN</td><td>MMB-CN</td><td>Bench V2.1</td><td>Avg</td></tr><tr><td rowspan="2">Keye2B</td><td rowspan="2">Baseline + PKM + Matched Sparse FFN</td><td>65.07 ± 0.26 65.40</td><td>50.23 ± 0.82 49.10</td><td>37.35 ± 0.65 39.22</td><td>60.43 ± 0.36 60.04</td><td>45.09 ± 0.25 45.20</td><td>32.48 ± 1.01 29.78</td><td>58.57 ± 0.32 57.87</td><td>52.97 ± 0.76 53.33</td><td>33.77 ± 1.17 36.15</td><td>36.51 ± 0.69 36.31</td><td>47.25 ± 0.15 47.24</td></tr><tr><td>64.30</td><td>51.90</td><td>32.48</td><td>60.46</td><td>44.13</td><td>33.11</td><td>58.00</td><td>48.14</td><td>29.80</td><td>36.31</td><td>45.86 47.93 ± 0.18</td></tr><tr><td></td><td>+ Lngram v2</td><td>66.47 ± 0.21 81.30</td><td>50.57 ± 1.05 80.90</td><td>39.01 ± 0.44 69.45</td><td>60.99 ± 0.31 74.03</td><td>45.64 ± 0.08 69.60</td><td>32.37 ± 0.55 67.78</td><td>59.39 ± 0.69</td><td>53.15 ± 0.45</td><td>34.86 ± 1.22</td><td>36.85 ± 0.75</td><td></td></tr><tr><td>Keye30B</td><td>Baseline + Lngram v2</td><td>81.00</td><td>80.80</td><td>72.02</td><td>74.87</td><td>70.93</td><td>70.22</td><td>82.09 83.97</td><td>83.75 86.92</td><td>81.73 83.75</td><td>91.66 92.62</td><td>78.23 79.71</td></tr></table>

Table 3: Hyperparameter ablations for Lngram $\mathbf { v } 2 .$
<table><tr><td>Configuration</td><td>Total Params</td><td>Active Params/token</td><td>OCRBench ↑</td><td>CountBenchQA ↑</td><td>RealWorldQA ↑</td><td>Avg ↑</td></tr><tr><td>Baseline</td><td></td><td></td><td>65.30</td><td>59.55</td><td>36.08</td><td>53.64</td></tr><tr><td>r16/m4/KV8</td><td>35.939M</td><td>18.121M</td><td>65.90</td><td>57.49</td><td>39.22</td><td>54.20</td></tr><tr><td>r64/m4/KV8</td><td>90.203M</td><td>18.932M</td><td>64.90</td><td>59.75</td><td>42.48</td><td>55.71</td></tr><tr><td>r23/m4/KV8/mem256</td><td>89.400M</td><td>38.175M</td><td>66.00</td><td>60.16</td><td>37.91</td><td>54.69</td></tr><tr><td>r512/m4/KV8/mem256</td><td>1187.014M</td><td>46.687M</td><td>64.70</td><td>59.55</td><td>41.18</td><td>55.14</td></tr><tr><td>r16/m5/KV2</td><td>155.804M</td><td>17.400M</td><td>66.20</td><td>57.29</td><td>35.42</td><td>52.97</td></tr><tr><td>r123/m4/KV2</td><td>156.115M</td><td>19.143M</td><td>66.10</td><td>56.26</td><td>38.69</td><td>53.69</td></tr><tr><td>r121/m4/KV16</td><td>155.689M</td><td>20.944M</td><td>66.70</td><td>60.99</td><td>40.78</td><td>56.16</td></tr></table>

As shown in Table 2, Lngram v2 improves the average score from $4 7 . 2 5 \pm 0 . 1 5 \mathrm { t o } 4 7 . 9 3 \pm 0 . 1 8$ on Keye2B, corresponding to a mean gain of 0.68 points across three seeds. The improvement is consistent across all three seeds. PKM reaches 47.24, close to the Baseline mean, while Lngram v2 exceeds PKM by 0.69 points. In contrast, the parameter-matched Sparse FFN obtains only 45.86 despite nearly identical numbers of additional total and activated parameters. These results indicate that the gains of Lngram v2 cannot be simply attributed to additional parameters or sparse computational capacity.

On Keye30B, Lngram v2 improves the ten-benchmark average from 78.23 to 79.71, a gain of 1.48 points, demonstrating that the method scales to the 30B regime. Given the substantially higher training cost at this scale and the controlled Keye2B comparisons above, we do not repeat the PKM and Matched Sparse FFN baselines on Keye30B.

Paired statistical tests on Keye30B yield a 95% confidence interval of [0.86, 2.11] for the average improvement. After Holm (Holm, 1979) correction, AI2D and MMBench-EN remain statistically significant, while MMBench-CN is significant under Benjamini–Hochberg (BH) (Benjamini & Hochberg, 1995) correction. Complete results are provided in Appendix D.1.

Lngram v2 also maintains lower LM loss than the Baseline during the middle and later stages of training across different route configurations, with routes=64 achieving the lowest loss. This suggests that the discrete memory branch can participate stably in backbone optimization. Training curves are provided in Figure 3.

## 4.2 ABLATION STUDIES

We investigate the main architectural hyperparameters of Lngram v2, where r denotes the number of routes, m the bits per route, KV the number of KV heads, and mem the memory-entry dimension. Because parameter counts vary across configurations, we focus on settings with similar total parameter counts or small architectural differences. Evaluation is performed on OCRBench (Liu et al., 2024b), CountBenchQA (Paiss et al., 2023; Beyer et al., 2024), and RealWorldQA (xAI, 2024).

Table 3 shows that increasing the number of routes is generally effective with modest activation overhead. With m = 4 and KV8 fixed, increasing routes from r16 to r64 improves the average from 54.20 to 55.71, while activated parameters increase only from 18.121M to 18.932M. Moreover, r64/m4/KV8 and r23/m4/KV8/mem256 both contain about 90M total parameters, but the former performs 1.02 points better with roughly half the activated parameters. Thus, under a similar parameter budget, increasing the number of discrete routes is more effective than increasing the memory dimension alone.

Readout capacity is also important: r121/m4/KV16 achieves the best average score of 56.16. By contrast, r512/m4/KV8/mem256 adds more than 1B parameters but remains below r64/m4/KV8, showing diminishing returns from simply enlarging memory capacity. Most configurations outperform the Baseline, indicating that the gains are robust across architectural settings.

Table 4: Validation losses of Baseline, Lngram v1, and Lngram $\mathbf { v } 2 .$
<table><tr><td>Model</td><td>Baseline</td><td>Lngram v1</td><td>Lngram v2</td><td>Lngram v2</td><td>Lngram v2</td></tr><tr><td>Routes</td><td></td><td></td><td>192</td><td>64</td><td>16</td></tr><tr><td>Val Loss ↓</td><td>2.8899</td><td>2.8609</td><td>2.8527</td><td>2.8578</td><td>2.8584</td></tr></table>

We further ablate the zero-value Sink on Keye30B. The ten-benchmark average increases from 79.39 without the Sink to 79.71 with it; MMBench-EN improves from 85.91 to 86.92 and Benchmark V2.1 from 91.66 to 92.62. We therefore enable the Sink by default.

## 4.3 EFFICIENCY EVALUATION

We measure prefill latency at sequence lengths of 256, 512, and 1024 and single-token decode latency on one H800 GPU. Complete results are provided in Appendix D.2.

Moderate-scale Lngram v2 configurations incur only modest inference overhead. For r64/m4/KV8, which improves the three-benchmark average by 2.07 points over the Baseline, prefill latency at length 1024 increases from 56.99 ms to 60.49 ms (+6.1%), with memory usage increasing from 5.24 GiB to 5.46 GiB. Decode latency increases from 48.26 to 52.14 ms/token (+8.0%), with only 0.16 GiB additional memory. The lighter r16/m4/KV8 configuration incurs approximately 3.7% prefill overhead at length 1024.

These results show that Lngram v2 improves performance without substantial inference-time computation, while its route count and readout capacity provide a flexible trade-off between model capability and deployment cost.

## 4.4 COMPARISON WITH LNGRAM V1

Lngram v2 is also designed to improve the parameter efficiency of conditional memory. In Lngram v1, the route count is coupled with the backbone representation dimension, resulting in large memory tables and readout modules. Lngram v2 decouples the route count from backbone width, allowing memory capacity to scale independently.

We compare the two designs using the same 0.6B DeepSeek-V2-style sparse mixture-of-experts (MoE) model (DeepSeek-AI, 2024; Dai et al., 2024). All models are trained from random initialization for approximately 1.31B tokens on FineWeb-Edu (Penedo et al., 2024), using the same data order, optimizer, and random seed. Detailed configurations are provided in Appendix D.3.

As shown in Table 4, both Lngram v1 and all v2 configurations outperform the memory-free Baseline. At R = 192, where the total parameter count is approximately matched to v1, Lngram v2 further reduces validation loss from 2.8609 to 2.8527, showing that the improved parameter efficiency does not sacrifice modeling capability.

More importantly, at R = 16, Lngram v2 reduces the total parameter count of the Lngram module by 82.6% and the activated parameters per token by 95.2% relative to v1, while achieving a slightly lower validation loss (2.8584 vs. 2.8609). Lngram v2 therefore preserves the benefits of conditional memory at substantially lower parameter and activation costs, making it more suitable for scaling to larger models.

## 5 CONCLUSION

We introduce Lngram v2, an efficient and scalable conditional memory mechanism in the latent space. By decoupling the number of routes, memory dimension, and backbone width and employing a context-aware readout, Lngram v2 substantially reduces the parameter and activation overhead of conditional memory, enabling it to scale to larger models while maintaining consistent performance gains. Further mechanistic analysis shows that Lngram’s discrete IDs not only serve as memory addresses but also systematically preserve the representational structure and semantic information of continuous hidden states, forming stable and reproducible ID–semantic associations. Lngram v2 therefore provides, on the one hand, low-cost and scalable discrete memory for large models and, on the other hand, transforms continuous internal representations into a statistically analyzable and queryable discrete interface, offering a new avenue for analyzing and understanding the internal representations of large models.

## REFERENCES

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit´ Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. arXiv preprint arXiv:2305.13245, 2023.

Guillaume Alain and Yoshua Bengio. Understanding intermediate layers using linear classifier probes. arXiv preprint arXiv:1610.01644, 2016.

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, et al. Flamingo: A visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.

David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6541–6549, 2017.

Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Estimating or propagating gradients ´ through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432, 2013.

Yoav Benjamini and Yosef Hochberg. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300, 1995. doi: 10.1111/j.2517-6161.1995.tb02031.x.

Lucas Beyer, Andreas Steiner, Andre Susano Pinto, Alexander Kolesnikov, Xiao Wang, Daniel´ Salz, Maxim Neumann, Ibrahim Alabdulmohsin, Michael Tschannen, Emanuele Bugliarello, Thomas Unterthiner, Daniel Keysers, Skanda Koppula, Fangyu Liu, Adam Grycner, Alexey Gritsenko, Neil Houlsby, Manoj Kumar, Keran Rong, Julian Eisenschlos, Rishabh Kabra, Matthias Bauer, Matko Bosnjak, Xi Chen, Matthias Minderer, Paul Voigtlaender, Ioana Bica, Ivana Bal-ˇ azevic, Joan Puigcerver, Pinelopi Papalampidi, Olivier Henaff, Xi Xiong, Radu Soricut, Jeremiah Harmsen, and Xiaohua Zhai. PaliGemma: A versatile 3b VLM for transfer. arXiv preprint arXiv:2407.07726, 2024.

Thorsten Brants, Ashok C. Popat, Peng Xu, Franz J. Och, and Jeffrey Dean. Large language models in machine translation. In Proceedings of the 2007 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning, pp. 858–867, 2007.

Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Advances in Neural Information Processing Systems, 2020.

Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, and Feng Zhao. Are we on the right way for evaluating large visionlanguage models? In Advances in Neural Information Processing Systems, volume 37, 2024.

Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, and Wenfeng Liang. Conditional memory via scalable lookup: A new axis of sparsity for large language models. arXiv preprint arXiv:2601.07372, 2026.

Thomas M. Cover and Peter E. Hart. Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1):21–27, 1967. doi: 10.1109/TIT.1967.1053964.

Damai Dai, Chengqi Deng, Chenggang Zhao, R. X. Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Y. Wu, Zhenda Xie, Y. K. Li, Panpan Huang, Fuli Luo, Chong Ruan, Zhifang Sui, and Wenfeng Liang. DeepSeekMoE: Towards ultimate expert specialization in mixture-of-experts language models. In Proceedings ofthe 62nd Annual Meeting ofthe Asso ciationfor Computational Linguistics, pp. 1280–1297, 2024. doi: 10.18653/v1/2024.acl-long.70.

DeepSeek-AI. DeepSeek-V2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434, 2024.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4171–4186, 2019.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021.

William Fedus, Barret Zoph, and Noam Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022.

Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, Dinesh Manocha, and Tianyi Zhou. HallusionBench: An advanced diagnostic suite for entangled language hallucination and visual illusion in large visionlanguage models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375–14385, 2024.

John Hewitt and Christopher D. Manning. A structural probe for finding syntax in word representations. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 4129–4138, 2019.

Geoffrey E. Hinton, James L. McClelland, and David E. Rumelhart. Distributed representations. In David E. Rumelhart and James L. McClelland (eds.), Parallel Distributed Processing: Explorations in the Microstructure of Cognition, Volume 1: Foundations, pp. 77–109. MIT Press, 1986.

Sture Holm. A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.

Robert Huben, Hoagy Cunningham, Logan Smith, Aidan Ewart, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. In International Conference on Learning Representations, 2024.

Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. A diagram is worth a dozen images. In European Conference on Computer Vision, pp. 235–251. Springer, 2016. doi: 10.1007/978-3-319-46493-0 15.

Kwai Keye Team, Bin Wen, Changyi Liu, Chengru Song, et al. Kwai keye-vl-2.0 technical report. arXiv preprint arXiv:2606.10651, 2026.

Guillaume Lample, Alexandre Sablayrolles, Marc’Aurelio Ranzato, Ludovic Denoyer, and Herve´ Jegou. Large memory layers with product keys. In ´ Advances in Neural Information Processing Systems, volume 32, 2019.

Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and C. Lawrence Zitnick. Microsoft COCO: Common objects in context. In ´ European Conference on Computer Vision, pp. 740–755. Springer, 2014. doi: 10.1007/978-3-319-10602-1 48.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pp. 216–233. Springer, 2024a.

Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. OCRBench: On the hidden mystery of OCR in large multimodal models. Science China Information Sciences, 67(12):220102, 2024b. doi: 10.1007/s11432-024-4235-6.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. MathVista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, 2024.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching CLIP to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3170–3180, 2023.

Guilherme Penedo, Hynek Kydl´ıcek, Loubna Ben Allal, Anton Lozhkov, Margaret Mitchell, Colinˇ Raffel, Leandro von Werra, and Thomas Wolf. The FineWeb datasets: Decanting the web for the finest text data at scale. arXiv preprint arXiv:2406.17557, 2024.

Belinda Phipson and Gordon K. Smyth. Permutation p-values should never be zero: Calculating exact p-values when permutations are randomly drawn. Statistical Applications in Genetics and Molecular Biology, 9(1):Article 39, 2010. doi: 10.2202/1544-6115.1585.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp. 8748– 8763, 2021.

Noam Shazeer. GLU variants improve transformer. arXiv preprint arXiv:2002.05202, 2020.

Ian Tenney, Dipanjan Das, and Ellie Pavlick. BERT rediscovers the classical NLP pipeline. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 4593–4601, 2019.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

xAI. Grok-1.5 vision preview. Technical announcement, 2024. Introduces the RealWorldQA benchmark.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. MM-Vet: Evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning, volume 235, pp. 57730– 57754. PMLR, 2024.

Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.

## A DETAILED DERIVATION OF THE COUNTERFACTUAL SURROGATEGRADIENT

## A.1 NOTATION AND PROBLEM SETUP

Consider a fixed n-gram order $n ,$ a fixed route $r ,$ and a local position within the n-gram ending at position $t ,$

$$
u \in \{ 0 , \ldots , n - 1 \} ,
$$

where $u = 0$ denotes the earliest position in the window. Let the bit logits at this position be

$$
\begin{array} { r } { z = ( z _ { 0 } , \ldots , z _ { M - 1 } ) \in \mathbb { R } ^ { M } , \qquad p _ { j } = \sigma ( \tau z _ { j } ) , } \end{array}
$$

where $\tau > 0$ is a temperature coefficient.

When constructing the surrogate, we fix all local symbols in the n-gram except the current one, as well as the symbols on all other routes. For any local symbol

$$
c \in \{ 0 , \ldots , K - 1 \} , \qquad K = 2 ^ { M } ,
$$

let $\beta ( c ) \in \{ 0 , 1 \} ^ { M }$ denote its binary representation, with $\beta _ { j } ( c )$ denoting its j-th bit.

After replacing the symbol at the current position with $c ,$ the corresponding counterfactual address is

$$
g _ { t , r } ^ { ( n ) } ( u  c ) = r K ^ { n } + \sum _ { \stackrel { i = 0 } { i \neq u } } ^ { n - 1 } a _ { t - n + 1 + i , r } K ^ { i } + c K ^ { u } .
$$

The corresponding counterfactual retrieval vector is

$$
E _ { c } = E ^ { ( n ) } [ g _ { t , r } ^ { ( n ) } ( u  c ) ] \in \mathbb { R } ^ { d _ { m } } .
$$

Let the upstream gradient received by the retrieved vector be

$$
g = \frac { \partial \mathcal { L } } { \partial E } \in \mathbb { R } ^ { d _ { m } } .
$$

Here, g already incorporates the gradients propagated through the subsequent GQA, zero-value Sink, output projection, and convolutional branch. The surrogate therefore only needs to handle the local discrete dependency from the retrieved result to the routing logits.

## A.2 LOCAL EXACT SURROGATE

Here, “exact” refers to the analytical gradient of the local conditional-expectation surrogate, rather than the gradient of the original hard-lookup path. We define the local conditional expected retrieval vector as

$$
\mu ( z ) = \sum _ { c = 0 } ^ { K - 1 } P ( c \mid z ) E _ { c } ,
$$

where

$$
P ( c \mid z ) = \prod _ { j = 0 } ^ { M - 1 } p _ { j } ^ { \beta _ { j } ( c ) } ( 1 - p _ { j } ) ^ { 1 - \beta _ { j } ( c ) } .
$$

This is the probability that the local symbol takes value c when the M bits at the current position are treated as independent Bernoulli variables. Importantly, this expectation is applied only to one local position in the n-gram window. The other $n - 1$ symbols always retain their hard discrete values from the actual forward pass. Consequently, only $K$ candidate symbols need to be enumerated, rather than all $K ^ { n }$ possible n-grams.

We define the local surrogate objective as

$$
\mathcal { L } _ { \mathrm { l o c a l } } ( z ) = \langle g , \mu ( z ) \rangle = \sum _ { c = 0 } ^ { K - 1 } P ( c \mid z ) \langle g , E _ { c } \rangle .
$$

To derive its gradient with respect to $z _ { j }$ , we first write

$$
\log P ( c \mid z ) = \sum _ { j = 0 } ^ { M - 1 } \left( \beta _ { j } ( c ) \log p _ { j } + ( 1 - \beta _ { j } ( c ) ) \log ( 1 - p _ { j } ) \right) .
$$

Since

$$
p _ { j } = \sigma ( \tau z _ { j } ) ,
$$

we have

$$
\frac { \partial p _ { j } } { \partial z _ { j } } = \tau p _ { j } ( 1 - p _ { j } ) .
$$

It follows that

$$
\frac { \partial \log P ( c \mid z ) } { \partial z _ { j } } = \tau \big ( \beta _ { j } ( c ) - p _ { j } \big ) ,
$$

and therefore

$$
\begin{array} { r } { \frac { \partial P ( c \mid z ) } { \partial z _ { j } } = P ( c \mid z ) \frac { \partial \log P ( c \mid z ) } { \partial z _ { j } } } \\ { = \tau P ( c \mid z ) \big ( \beta _ { j } ( c ) - p _ { j } \big ) . } \end{array}
$$

Substituting this into the definition of $\mathcal { L } _ { \mathrm { l o c a l } }$ gives

$$
\begin{array} { r l } { \displaystyle \frac { \partial \mathcal { L } _ { \mathrm { l o c a l } } } { \partial z _ { j } } = \sum _ { c = 0 } ^ { K - 1 } \frac { \partial P ( c \mid z ) } { \partial z _ { j } } \langle g , E _ { c } \rangle } & { } \\ { \displaystyle } & { = \tau \sum _ { c = 0 } ^ { K - 1 } P ( c \mid z ) \big ( \beta _ { j } ( c ) - p _ { j } \big ) \langle g , E _ { c } \rangle . } \end{array}
$$

In practice, we further multiply this expression by a global surrogate scaling coefficient λ:

$$
\boxed { \frac { \partial \mathcal { L } } { \partial z _ { j } } \approx \lambda \tau \sum _ { c = 0 } ^ { K - 1 } P ( c \mid z ) \big ( \beta _ { j } ( c ) - p _ { j } \big ) \langle g , E _ { c } \rangle } .
$$

This is the analytical form of the local exact surrogate. In implementation, the above gradient is computed for every local position u in every valid n-gram window and accumulated into the routing logits of the corresponding positions. Because the same routing logit may participate in n-grams ending at multiple positions and in n-grams of multiple orders, its final surrogate gradient is the sum of the contributions from all relevant windows.

When $M = 4 ,$

$$
K = 2 ^ { M } = 1 6 ,
$$

only 16 counterfactual symbols need to be enumerated for each local position, making the computation practically manageable.

## A.3 ONE-BIT APPROXIMATE SURROGATE

When computational efficiency is more important, we further restrict the counterfactual to a single bit at a time. Let the hard symbol produced by the actual forward pass at the current position be

$$
\hat { c } = \sum _ { j = 0 } ^ { M - 1 } \hat { b } _ { j } 2 ^ { j } ,
$$

where

$$
\hat { b } _ { j } = \mathbb { I } [ z _ { j } > 0 ] .
$$

For bit j, we keep all other bits of the symbol and all other symbols in the window fixed, and force only this bit to 0 or 1. Denote the resulting two counterfactual symbols by

$$
\hat { c } _ { j } ^ { ( 0 ) } , \qquad \hat { c } _ { j } ^ { ( 1 ) } ,
$$

with corresponding retrievals

$$
E _ { j } ^ { ( 0 ) } , \qquad E _ { j } ^ { ( 1 ) } .
$$

We define the local counterfactual score for this bit as

$$
s _ { j } = \left. g , E _ { j } ^ { ( 1 ) } - E _ { j } ^ { ( 0 ) } \right. .
$$

This quantity measures how much the retrieved result changes along the current upstream-gradient direction when only bit j is switched from 0 to 1, while all other discrete decisions remain fixed.

In implementation, $E _ { j } ^ { ( 0 ) }$ and $E _ { j } ^ { ( 1 ) }$ need not be constructed explicitly as two separate retrievals. Let $E _ { \mathrm { h a r d } }$ denote the retrieval from the actual forward pass and $E _ { \mathrm { { f l i p } } }$ the retrieval obtained after flipping the j-th bit of the current hard address. If the current bit is 0, then

$$
E _ { \mathrm { h a r d } } = E _ { j } ^ { ( 0 ) } , \qquad E _ { \mathrm { f i p } } = E _ { j } ^ { ( 1 ) } ;
$$

if the current bit is 1, then

$$
E _ { \mathrm { h a r d } } = E _ { j } ^ { ( 1 ) } , \qquad E _ { \mathrm { f i p } } = E _ { j } ^ { ( 0 ) } .
$$

Thus, the score can be written uniformly as

$$
s _ { j } = \left( 1 - 2 \hat { b } _ { j } \right) \langle g , E _ { \mathrm { H i p } } - E _ { \mathrm { h a r d } } \rangle = \left. g , E _ { j } ^ { ( 1 ) } - E _ { j } ^ { ( 0 ) } \right. .
$$

Using the local slope of the sigmoid to characterize the sensitivity of the bit to the continuous logit gives the one-bit approximate surrogate:

$$
\boxed { \frac { \partial \mathcal { L } } { \partial z _ { j } } \approx \lambda \tau p _ { j } ( 1 - p _ { j } ) \left. g , E _ { j } ^ { ( 1 ) } - E _ { j } ^ { ( 0 ) } \right. } .
$$

Compared with the exact surrogate, this approximation no longer enumerates all K symbols and instead requires only one flipped counterfactual address per bit. The counterfactual lookup complexity per local position is therefore reduced from

$$
O ( K )
$$

to

$$
O ( M ) .
$$

When $M = 4$ , the number of counterfactual candidates per local position is reduced from 16 to 4, substantially reducing the additional lookup overhead during training.

## A.4 CORRESPONDENCE TO THE ACTUAL IMPLEMENTATION

The derivation above considers a single n-gram, a single route, and a single local position within the window. The actual implementation performs the same computation in blocks over n-gram orders and routes.

In full-lookup mode, for each order n, the hard lookup first produces a tensor of shape

$$
[ B , T , R , d _ { m } ] .
$$

The surrogate operation is applied directly to the retrieval tensor of that order. Its numerical value is kept exactly unchanged in the forward pass, while the corresponding upstream gradient

$$
g \in \mathbb { R } ^ { d _ { m } }
$$

is accessed during backpropagation and used to query the same n-gram table at counterfactual addresses. After surrogate processing is completed for all orders, the resulting tensors are concatenated along the last dimension, yielding the route memory tensor passed to the subsequent GQA:

$$
[ B , T , R , | \mathcal { N } | d _ { m } ] .
$$

In streaming mode, the computational logic remains unchanged, except that the route dimension is divided into smaller chunks. For the current interval

$$
\mathrm { [ r o u t e _ { s t a r t } , r o u t e _ { e n d } ) , }
$$

exact n-gram lookup and the surrogate are first performed for the corresponding routes. The re trieved results are then projected into keys and values, while the GQA normalization term and value-weighted sum are accumulated using an online softmax. The surrogate formula for each route chunk is identical to that in full-lookup mode, and its gradients are automatically accumulated into the complete routing logits.

Under context-parallel training, different positions may reside on different devices. The implementation therefore first gathers the limited amount of historical route information required to construct local n-grams and then explicitly forms local logit windows of shape

$$
[ B , T , n , R , M ] .
$$

For each local target position, all other symbols in the window remain fixed, while only one local symbol is replaced or one bit is flipped, using exactly the same exact or approximate surrogate. Context parallelism changes only how the historical window is obtained; it does not alter the definition of the counterfactual gradient.

Importantly, the surrogate replaces only the backward gradient of the routing logits and does not modify the actual forward computation. For the hard lookup result $E _ { \mathrm { h a r d } } ,$ , its upstream gradient is still propagated according to the standard chain rule, so the memory-table entry selected by the actual lookup receives its normal parameter gradient. In contrast, the table parameters accessed by counterfactual queries are treated as constants during surrogate computation. Therefore, $E _ { c } , E _ { j } ^ { ( 0 ) }$ and $E _ { j } ^ { ( 1 ) }$ do not propagate additional gradients into the corresponding counterfactual table entries.

The training procedure of Lngram $\mathbf { v } 2$ can therefore be summarized as

$$
\underbrace { \mathrm { h a r d ~ r o u t i n g } \to \mathrm { e x a c t ~ a d d r e s s } \to \mathrm { h a r d ~ t a b l e ~ l o o k u p } } _ { \mathrm { a c t u a l ~ f o r w a r d ~ p a t h } } ,
$$

while only the backward pass uses

$$
\underbrace { \mathrm { c o u n t e r f a c t u a l ~ l o o k u p } \to \mathrm { s u r r o g a t e ~ g r a d i e n t } } _ { \mathrm { s u r r o g a t e ~ g r a d i e n t ~ f o r ~ r o u t i n g ~ l o g i t s } } .
$$

This design guarantees exactly the same hard discrete addressing behavior during training and inference, while the counterfactual computation is used solely to provide structure-aware optimization signals for the non-differentiable routing decisions. At inference time, the surrogate gradient is unnecessary; only hard discretization, exact n-gram lookup, and the subsequent readout computation are retained.

## A.5 DIRECT STE FAILS TO EFFECTIVELY OPTIMIZE DISCRETE ROUTING

The discrete routing and exact address lookup of Lngram v2 are non-differentiable, making gradient estimation for the discrete choices critical to optimization. To verify the necessity of the surrogate gradient in Lngram v2, we conduct a strictly controlled ablation in which the model architecture, forward computation, training data, and optimization configuration are kept identical, and only the original surrogate gradient is replaced with a direct Straight-Through Estimator (STE). Both meth ods perform the same hard quantization and exact table lookup in the forward pass. The difference lies in the backward pass: direct STE passes gradients through the discrete address and approximates the binary threshold gradient using the local slope of the sigmoid, without accounting for the actual memory contents retrieved after switching to a different discrete address. In contrast, the surrogate gradient of Lngram v2 constructs its gradient from the actual lookup results corresponding to candidate discrete choices.

![](images/8b528164656a4c4f34c6f65de3737c9880880b2a2629b57d3a9eda3e1d417d3b.jpg)  
Figure 2: Comparison of loss.

![](images/2fac062785b7ce1cec6b0f9bed57b6df638ffe66ce698eec285716d12dc425ec.jpg)  
Figure 3: Comparison of training loss.

The experiment uses the R = 16 configuration of Lngram v2, with Lngram inserted into the 2nd and 12th Transformer layers. Each route uses 4-bit discrete codes and 2/3-gram memory, with a memory dimension of 64. Both experiments are trained in BF16 on 8 GPUs with a global batch size of 32, an initial learning rate of $3 \times 1 0 ^ { - 4 }$ , a warmup ratio of 1%, and a random seed of 42. We retain the original cosine learning-rate schedule over 61,036 steps but stop training early at step 10,000. Figure 3 reports the average training loss over each consecutive block of 100 training steps without additional smoothing.

As shown in Figure 3, the training loss with the Lngram v2 surrogate gradient decreases consistently throughout training, whereas STE quickly stalls at a substantially higher loss. The same trend is observed on the validation set, where the two methods obtain validation losses of 2.8585 and 7.3349, respectively. These results indicate that the optimization difficulty of Lngram does not arise solely from the binary threshold itself, but also from the discontinuous changes in memory contents associated with different discrete addresses. Direct STE ignores how switching addresses changes the actual lookup result and therefore fails to provide the router with an effective signal for discrete selection. In contrast, the Lngram v2 surrogate gradient explicitly incorporates the memory changes induced by different candidate addresses, aligning the gradient with the actual forward behavior.

Table 5: Comparison of Qwen3+SigLIP2 and Keye2B with Lngram $\mathbf { v } 2 .$
<table><tr><td>Model</td><td>OCR</td><td>MathVista</td><td>MMVet</td><td>Hallusion</td><td>MMStar</td><td>MMMU</td><td>AI2D</td><td>MMB-EN</td><td>MMB-CN</td><td>Bench V2.1</td><td>Avg</td></tr><tr><td>Keye2B</td><td>65.30</td><td>51.00</td><td>36.74</td><td>60.78</td><td>44.73</td><td>33.89</td><td>59.00</td><td>52.09</td><td>32.12</td><td>35.57</td><td>47.12</td></tr><tr><td>Keye2B+Lngram v2</td><td>66.70</td><td>51.50</td><td>38.44</td><td>61.41</td><td>45.53</td><td>33.00</td><td>60.14</td><td>52.55</td><td>33.13</td><td>35.79</td><td>47.82</td></tr><tr><td>Qwen3+SigLIP2</td><td>48.20</td><td>41.01</td><td>26.65</td><td>38.20</td><td>62.50</td><td>41.53</td><td>42.56</td><td>64.63</td><td>63.62</td><td>43.17</td><td>47.21</td></tr><tr><td>Qwen3+SigLIP2+Lngram v2</td><td>48.80</td><td>41.43</td><td>28.44</td><td>38.10</td><td>63.76</td><td>41.53</td><td>39.67</td><td>65.33</td><td>63.24</td><td>43.54</td><td>47.38</td></tr></table>

Under this setting, direct STE fails to converge effectively, demonstrating that the surrogate gradient is an important component for stable training of Lngram v2.

## B ADDING LNGRAM V2 TO MODELS WITH STABILIZED REPRESENTATIONS

This section analyzes an empirical observation: Lngram v2 provides consistent gains when added to the mature Keye2B VLM, whereas its gains are weaker in a VLM assembled from pretrained SigLIP2 and Qwen3-1.7B-Instruct that is still undergoing cross-modal adaptation. As shown in Table 5, adding Lngram v2 to Keye2B and training for only 5B tokens yields substantially larger gains than those obtained after 40B tokens of training for the model assembled from SigLIP2 and Qwen3-1.7B-Instruct. Further analysis suggests that this difference mainly arises because the continuous representations of the backbone continue to evolve during cross-modal adaptation, making the discrete addresses learned by Lngram v2 difficult to reuse consistently throughout training. This instability in the address space further limits the learning of memory representations, leaving the readout with only a weak influence on the backbone hidden states and final predictions.

## B.1 EXPERIMENTAL SETUP

We deterministically select 48 documents from a fixed subset of COCO and keep the images, text prompts, and input templates unchanged across checkpoints and intervention conditions. All inputs use the same question, Describe this image briefly., without providing captions, categories, or answer information.

Because the two model families use different visual tokenization schemes, the number of visual tokens differs across models. We therefore first aggregate within each document and then perform document-level statistics for all cross-model comparisons, rather than attempting token-wise alignment across models. All confidence intervals are obtained from 10,000 document-level bootstrap resamples using a fixed random seed.

We evaluate address stability in two ways. The first uses the router from each checkpoint itself. The second fixes the router from the final checkpoint of the same model family and disables all Lngram v2 outputs, thereby isolating the effect of changes in backbone hidden states on the discrete addresses. The latter separates changes in the router parameters from changes in the input representations and is therefore used as our primary analysis.

## B.2 CROSS-CHECKPOINT STABILITY OF DISCRETE ADDRESSES

Lngram v2 accesses its memory tables through exact discrete n-gram addresses. If the same semantics are repeatedly mapped to different addresses during training, the same table entry cannot consistently receive coherent training signals. We therefore compare route-code agreement, bit-level Hamming distance, and 2-gram and 3-gram address survival rates between adjacent checkpoints.

Table 6 reports the results when the final router is fixed and all Lngram v2 outputs are disabled. For visual tokens, the newly assembled VLM exhibits substantially lower address stability than the mature Keye2B model, and the gap becomes even larger at Layer 13. For example, visual route-code agreement at Layer 13 decreases from 0.9246 for Keye2B to 0.7702 for the new VLM, while 3-gram survival decreases from 0.7945 to 0.4754.

Document-level paired bootstrap yields the same conclusion. Relative to Keye2B, visual code agreement at Layers 1 and 13 decreases by 0.1024 and 0.1544, respectively, with corresponding 95% CIs of

$$
[ - 0 . 1 0 3 5 , - 0 . 1 0 1 3 ] \quad \mathrm { a n d } \quad [ - 0 . 1 5 6 8 , - 0 . 1 5 1 9 ] .
$$

Table 6: Discrete address stability of Lngram $\mathbf { v } 2 .$
<table><tr><td>Layer</td><td>Modality</td><td>Code Agreement ↑</td><td>Hamming ↓</td><td>2-gram Survival ↑</td><td>3-gram Survival ↑</td></tr><tr><td>1</td><td>Visual</td><td>0.8859 / 0.9883</td><td>0.1204 /0.0118</td><td>0.7900 / 0.9769</td><td>0.7079 / 0.9657</td></tr><tr><td>1</td><td>Text</td><td>0.9249 / 0.9462</td><td>0.0774 / 0.0575</td><td>0.8578 / 0.9028</td><td>0.7964 / 0.8625</td></tr><tr><td>13</td><td>Visual</td><td>0.7702 / 0.9246</td><td>0.2630 / 0.0779</td><td>0.6020 / 0.8565</td><td>0.4754 / 0.7945</td></tr><tr><td>13</td><td>Text</td><td>0.8971 / 0.9163</td><td>0.1081 / 0.0866</td><td>0.8072 / 0.8356</td><td>0.7254 / 0.7616</td></tr></table>

The corresponding decreases in 3-gram survival are 0.2579 and 0.3190. The text modality exhibits differences in the same direction, but with substantially smaller magnitudes.

The same trend is reproduced when each checkpoint uses its own router. For example, at Layer 13, visual code agreement / 3-gram survival is 0.7122/0.3813 in the new model, compared with 0.8818/0.6964 in the mature Keye2B model. This phenomenon therefore cannot be explained solely by changes in the router parameters and instead primarily reflects changes in the continuous representations fed into Lngram v2 over the course of training.

These results show that the stability of continuous representations directly affects memory modules that rely on exact discrete addresses. If the same local pattern frequently crosses discrete boundaries during training, semantically similar training examples are distributed across different table entries, making it difficult for any individual memory address to accumulate consistent information ove time.

## B.3 LAYER-WISE PROPAGATION OF LNGRAM V2 RESIDUALS

Address instability further increases the difficulty of learning effective Lngram v2 memory representations. To analyze their actual influence on the backbone representations, we separately disable Lngram v2 at Layer 1, Layer 13, or both layers, and compare the hidden states before and after the interventions.

We define the normalized representation difference at layer l as

$$
R _ { l } = \frac { \| h _ { l } ^ { \mathrm { o n } } - h _ { l } ^ { \mathrm { o f f } } \| _ { 2 } } { \| h _ { l } ^ { \mathrm { o n } } \| _ { 2 } } .
$$

To further distinguish between “small initial injection” and “attenuation during subsequent propagation,” we define

$$
\mathrm { R e t e n t i o n } _ { l } = \frac { \| \Delta h _ { l } \| _ { 2 } } { \| \delta _ { \mathrm { i n j } } \| _ { 2 } } ,
$$

where $\delta _ { \mathrm { i n j } }$ denotes the effective Lngram v2 perturbation that actually enters the hidden state after the BF16 residual addition.

The results are shown in Table 7. The most prominent difference in the new model is not that Lngram v2 perturbations decay more rapidly in subsequent layers, but rather that their initial injections are substantially smaller. In particular, at Layer 13, the immediate R at visual positions is 5.99 × 10<sup>−3</sup>, approximately 1/9.6 of that in the mature model; at text positions it is only $9 . 2 6 \times 1 0 ^ { - 5 }$ approximately 1/68.7 of the mature-model value.

To determine whether these representation differences ultimately affect model predictions, we additionally disable both Lngram v2 layers simultaneously and compare the final hidden states and next-token logits.

In the mature Keye2B model, disabling Lngram v2 results in a normalized next-token logit change of 0.1928 and a Jensen–Shannon (JS) divergence of $1 . 2 5 \times 1 0 ^ { - 2 }$ , while changing the next-token top-1 prediction for 6/48 documents. In contrast, the new model exhibits a logit R of only 0.0174 and a JS divergence of $2 . 9 7 \times 1 0 ^ { - 4 }$ , with the top-1 prediction unchanged for all 48 documents.

This shows that the two model families differ not only in address stability but also in functional allocation: the mature Keye2B model has already assigned a substantial predictive role to Lngram v2, whereas Lngram v2 in the new model still has only a weak influence on the final predictions.

Table 7: Immediate injection magnitude and final propagation of Lngram v2 interventions.
<table><tr><td>Layer</td><td>Modality</td><td>Model</td><td>Immediate R</td><td>Final R</td><td>Final Retention</td></tr><tr><td>1</td><td>Visual</td><td>New</td><td>0.01465</td><td>0.06442</td><td>16.64</td></tr><tr><td>1</td><td>Visual</td><td>Keye2B</td><td>0.01841</td><td>0.09515</td><td>3.685</td></tr><tr><td>1</td><td>Text</td><td>New</td><td>0.008512</td><td>0.01281</td><td>12.79</td></tr><tr><td>1</td><td>Text</td><td>Keye2B</td><td>0.04306</td><td>0.11646</td><td>4.935</td></tr><tr><td>13</td><td>Visual</td><td>New</td><td>0.005987</td><td>0.02476</td><td>3.109</td></tr><tr><td>13</td><td>Visual</td><td>Keye2B</td><td>0.05771</td><td>0.10009</td><td>1.890</td></tr><tr><td>13</td><td>Text</td><td>New</td><td>0.0000926</td><td>0.01069</td><td>3.432</td></tr><tr><td>13</td><td>Text</td><td>Keye2B</td><td>0.006358</td><td>0.11245</td><td>2.356</td></tr></table>

Table 8: Changes in final predictions after disabling all Lngram v2 layers.
<table><tr><td>Metric</td><td>New VLM</td><td>Keye2B</td></tr><tr><td>Final hidden R (Visual)</td><td>0.06510</td><td>0.12395</td></tr><tr><td>Final hidden R (Text)</td><td>0.01303</td><td>0.14383</td></tr><tr><td>Next-token logit R</td><td>0.01741</td><td>0.19282</td></tr><tr><td>Next-token JS</td><td>0.000297</td><td>0.01250</td></tr><tr><td>Top-1 agreement</td><td>1.000</td><td>0.875</td></tr></table>

We further examine the directional relationship between the Lngram v2 injection and the attention/MLP residual at the same layer. The cosine similarities are close to zero in both model families, with mean absolute values no greater than approximately 0.016. Thus, there is no evidence that the Lngram v2 output merely duplicates the backbone residual.

At the same time, local counteracting components of the backbone residual along the direction of the Lngram v2 injection can indeed be observed in some layers of the new model. However, these components are insufficient to cancel the overall perturbation: the total hidden-state difference induced by the Lngram v2 intervention is preserved or amplified in subsequent layers. Local compensation may therefore exist, but it cannot explain the weaker final gains observed in the new model.

## B.4 SUMMARY

Taken together, the experiments above yield two relatively independent observations.

First, the visual discrete addresses of the newly assembled VLM are substantially less stable dur ing training, particularly at the deeper Layer 13. Because Lngram v2 accesses its memory tables using exact discrete n-gram addresses, this non-stationarity reduces the probability that the same address continues to correspond to similar representations over time, dispersing training signals across different table entries and making stable memory formation more difficult.

This in turn leads to substantially smaller Lngram v2 injection magnitudes and predictive functional loads in the new model than in the mature Keye2B model. Existing Lngram v2 perturbations are not rapidly eliminated by the subsequent backbone layers; rather, the key difference is that optimization never assigns Lngram v2 a functional weight comparable to that learned in the mature model.

Thus, using pretrained vision and language models alone is not sufficient to guarantee that Lngram v2 can immediately become effective. Although both unimodal components already possess strong representational capabilities, the deep hidden space on which Lngram v2 relies may still change substantially while the cross-modal scaffold continues to adapt, which is unfavorable for forming stable discrete memory.

Based on this observation, in our formal comparison experiments we introduce Lngram v2 only after the model has entered the middle stage of training and its cross-modal representations have become more stable, rather than adding the module from the beginning of pretraining.

Table 9: Utilization statistics of Lngram ${ \bf v } 2$ discrete codes.
<table><tr><td>Layer</td><td>Effective Codes</td><td>Normalized Entropy</td><td>Top-Code Frequency</td><td>Dead Codes</td></tr><tr><td>1</td><td>7.7916</td><td>0.7210</td><td>0.3206</td><td>0 / 2,048</td></tr><tr><td>13</td><td>7.9979</td><td>0.7291</td><td>0.3093</td><td>0 / 2,048</td></tr></table>

## C SUPPLEMENTARY ANALYSIS OF LNGRAM DISCRETE IDS

This section supplements the discrete representation analysis in Section 3, including statistics on discrete-code utilization, the statistical protocol and representative examples for route-code semantic associations, and detailed experiments on distributed semantic representation across multiple routes.

## C.1 DISCRETE-CODE UTILIZATION

Before using discrete IDs for semantic analysis, we first verify that the Lngram discrete space does not exhibit substantial collapse. For each route, we compute the entropy of its code-value distribution as

$$
H = - \sum _ { c } p ( c ) \log p ( c ) , \qquad N _ { \mathrm { e f f } } = \exp ( H ) ,\tag{2}
$$

where $N _ { \mathrm { e f f } }$ denotes the effective number of code values under the corresponding distribution.

Table 9 reports statistics for the two analyzed layers. Each route uses approximately eight effective code values on average, with a normalized entropy of approximately 0.72 and an average frequency of approximately 30% for the most common code. Moreover, every theoretically possible route code is used in both layers, with no dead codes.

Therefore, the ID-only semantic readout capability observed in the main text does not arise from degeneration of the discrete space into a small number of frequently used codes, but instead emerges while the discrete address space remains broadly utilized.

## C.2 ROUTE-CODE SEMANTIC ASSOCIATIONS AND REPRESENTATIVE EXAMPLES

Beyond the complete signature, we further examine whether semantic information can be localized to individual route codes. This analysis is related in spirit to prior work that identifies concept-level semantics in internal neural units (Bau et al., 2017). However, the objects analyzed here are not separately trained probes or dictionary units, but the discrete addresses natively generated by the Lngram memory mechanism.

Let $C _ { \ell , }$ <sub>r</sub> denote the discrete code of route r at layer ℓ. We denote a code–semantic relation as

$$
C _ { \ell , r } = c \Rightarrow z ,\tag{3}
$$

where z is the semantic concept under analysis. On the held-out set, we define

$$
P = P _ { \mathrm { h e l d o u t } } \left( z \mid C _ { \ell , r } = c \right) ,\tag{4}
$$

and let $P _ { 0 }$ denote the baseline probability after controlling for nuisance factors such as token position and image source. The corresponding increase in semantic probability is

$$
\Delta = P - P _ { 0 } .\tag{5}
$$

All formal statistical associations follow a strict reference–held-out separation protocol. Candidate code–semantic relations and their effect directions are determined exclusively on the reference data and are subsequently tested independently on the held-out data. The reference set yields 16,304 candidate associations, of which 14,002 can be evaluated on the held-out set. Ultimately, 9,220 satisfy the requirements for consistent effect direction, confidence intervals, and Benjamini–Hochberg multiple-testing correction (Benjamini & Hochberg, 1995), corresponding to an overall replication rate of 65.85%. For local presence, the replication rates at Layers 1 and 13 reach 83.01% and 84.94%, respectively.

Table 10: Representative semantic correspondences of Lngram route codes. $P$ denotes the conditional probability of the semantic concept given the code on the held-out set, $P _ { 0 }$ denotes the baseline probability after controlling for nuisance factors such as position and image source, and $\Delta = P { - } P _ { 0 }$
<table><tr><td>Code Expression</td><td>Semantic Concept</td><td> $P$ </td><td> $P _ { 0 }$ </td><td> $\Delta$ </td></tr><tr><td> $C _ { 1 , 7 8 } = 8$ </td><td>skis</td><td>11.03%</td><td>2.90%</td><td>+8.13 pp</td></tr><tr><td> $C _ { 1 3 , 5 3 } = 1 4$ </td><td>person</td><td>63.25%</td><td>53.60%</td><td>+9.65 pp</td></tr><tr><td> $C _ { 1 , 4 7 } = 1 1$ </td><td>cat</td><td>9.84%</td><td>3.97%</td><td>+5.87 pp</td></tr><tr><td> $C _ { 1 , 6 } = 1 5$ </td><td>dining table</td><td>14.65%</td><td>9.67%</td><td>+4.98 pp</td></tr><tr><td> $C _ { 1 3 , 9 7 } = 1 5$ </td><td>kite</td><td>11.60%</td><td>2.07%</td><td>+9.53 pp</td></tr><tr><td> $C _ { 1 , 6 3 } = 0$ </td><td>car</td><td>21.03%</td><td>11.13%</td><td>+9.89 pp</td></tr><tr><td> $C _ { 1 , 1 1 7 } = 4$ </td><td>skateboard</td><td>9.57%</td><td>2.80%</td><td>+6.77 pp</td></tr><tr><td> $C _ { 1 , 7 5 } = 1 0$ </td><td>giraffe</td><td>21.70%</td><td>2.37%</td><td>+19.33 pp</td></tr><tr><td> $C _ { 1 3 , 8 3 } = 9$ </td><td>chair</td><td>13.38%</td><td>10.07%</td><td>+3.31 pp</td></tr><tr><td> $C _ { 1 , 4 8 } = 5$ </td><td>sports ball</td><td>9.36%</td><td>3.33%</td><td>+6.03 pp</td></tr><tr><td> $C _ { 1 , 7 1 } = 9$ </td><td>traffic light</td><td>11.75%</td><td>3.30%</td><td>+8.45 pp</td></tr><tr><td> $C _ { 1 , 7 5 } = 9$ </td><td>horse</td><td>10.06%</td><td>2.23%</td><td>+7.83 pp</td></tr><tr><td> $C _ { 1 , 7 3 } = 0$ </td><td>airplane</td><td>9.71%</td><td>3.03%</td><td>+6.68 pp</td></tr><tr><td> $C _ { 1 , 7 5 } = 1 1$ </td><td>zebra</td><td>10.62%</td><td>1.63%</td><td>+8.99 pp</td></tr><tr><td> $C _ { 1 3 , 1 0 6 } = 7$ </td><td>surfboard</td><td>9.39%</td><td>2.47%</td><td>+6.92 pp</td></tr></table>

To provide a more intuitive view of the semantic structure encoded by the discrete codes, we fur ther list 15 representative examples selected from these analyzable associations. The displayed entries have relatively high held-out conditional probabilities, with positive probability increases on both the reference and held-out sets. Importantly, this selection is used only for qualitative illustration of intuitive code–semantic correspondences and is not used to compute the replication rates or statistical-significance results above. The formal statistical evidence continues to come from reference–held-out validation over the complete candidate set.

Table 10 shows several discrete codes with intuitive semantic interpretations. For example, when

$$
C _ { 1 , 7 5 } = 1 0 \Rightarrow \mathrm { g i r a f f e } ,
$$

the conditional probability of giraffe on the held-out set reaches 21.70%, substantially above the corresponding baseline of 2.37%, representing an increase of 19.33 percentage points. Similarly,

$$
C _ { 1 3 , 5 3 } = 1 4 \Rightarrow \mathrm { p e r s o n }
$$

corresponds to a held-out conditional probability of 63.25%, which is 9.65 percentage points above the baseline.

These correspondences span diverse concepts, including people, animals, vehicles, and common objects. Notably, different codes within the same route can also exhibit distinct but related semantics. For example, codes 9, 10, and 11 on route 75 show strong positive associations with horse, giraffe, and zebra, respectively. This indicates that a route itself does not simply correspond to one fixed concept; rather, its discrete values can distinguish different semantic states within the same local discrete space.

Thus, Lngram’s discrete IDs not only preserve the semantic structure of hidden states at the level of complete signatures, but individual route codes can also form statistically reproducible associations with clearly interpretable concepts. The originally continuous and difficult-to-enumerate internal hidden states can therefore be transformed into queryable discrete semantic relations of the form

$$
C _ { \ell , r } = c \Rightarrow z .
$$

## C.3 DETAILED ANALYSIS OF DISTRIBUTED SEMANTICS

Individual route codes already exhibit stable semantic tendencies, yet the complete IDs provide substantially stronger readout capability in Table 1 of the main text. To directly determine whether

this difference arises from joint encoding across multiple routes, we recombine the single-code semantic effects estimated on the reference set.

For a complete signature

$$
C = ( c _ { 1 } , \dots , c _ { 1 2 8 } )
$$

and a concept $z ,$ we define

$$
S _ { \mathrm { d i c t } } ( C , z ) = \frac { 1 } { 1 2 8 } \sum _ { r = 1 } ^ { 1 2 8 } \Delta _ { \mathrm { r e f } } ( r , c _ { r } , z ) ,\tag{6}
$$

where $\Delta _ { \mathrm { r e f } } ( r , c _ { r } , z )$ denotes the semantic effect of the current code $c _ { r }$ on route $r$ for concept $z ,$ as estimated from the reference data. For single-code mappings that do not satisfy the statisticalsupport criteria on the reference set, the corresponding effect is set to zero.

On the held-out set, this reader uses only discrete IDs, without accessing any continuous hidden states or refitting parameters on the held-out data. Its performance therefore directly measures whether route-code semantic associations independently identified on the reference set can be composed into semantic representations that generalize to unseen samples.

Simply summing the independent single-code effects significantly outperforms the nuisance baseline in all six experimental settings and retains 31.79%–54.82% of the excess AP of $H _ { \mathrm { m o d e l } }$ . This demonstrates that the weak semantic signals present in individual route codes are not isolated phenomena, but can be combined into effective semantic predictions.

To further analyze the complementary contributions of different routes, for each query–concept pair we rank routes in descending order according to

$$
\begin{array} { r } { | \Delta _ { \mathrm { r e f } } ( r , c _ { r } , z ) | } \end{array}\tag{7}
$$

and sequentially construct semantic predictions using

$$
m \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}
$$

routes.

Across all six experimental settings, we observe

$$
A P _ { 1 2 8 } > A P _ { 1 } ,\tag{8}
$$

with

$$
{ \frac { A P _ { 1 2 8 } } { A P _ { 1 } } } = 2 . 0 2 – 1 2 . 3 2 .\tag{9}
$$

Here, $A P _ { 1 }$ already selects the single route with the largest contribution for each query–concept pair. Therefore, the advantage of the complete signature cannot be explained by one consistently dominant route and must instead arise from complementary information provided by multiple routes.

This result also explains the performance gap between the single-code semantic dictionary and the complete-ID reader. The reader based on independent route-code marginal effects retains only 31.79%–54.82% of the excess $\mathbf { A P }$ of the continuous representation, whereas ID-KNN operating directly on the complete signature retains 65.77%–84.27%. Thus, the marginal semantics of individual route codes explain only part of the information in the complete discrete representation; the joint configuration of multiple routes additionally encodes compositional semantic structure that cannot be fully recovered from independent single-code effects.

In summary, Lngram’s discrete representations provide two complementary levels of interpretabil ity: individual route codes establish stable and reproducible semantic correspondences with specific concepts, while the complete signature preserves richer hidden-state semantics through distributed combinations across multiple routes. This enables Lngram IDs to serve both as variables for exact memory addressing and as a structured discrete interface for analyzing the model’s internal continuous representations.

Table 11: Paired statistical tests for the step-5,000 Keye30B Baseline and Lngram v2 checkpoints. Scores, differences, and confidence intervals are reported in percentage points.
<table><tr><td>Benchmark</td><td>n</td><td>Baseline</td><td>Lngram v2</td><td> $\Delta$ </td><td>95% CI</td><td>Raw p</td><td>Holm p</td><td>BH q</td></tr><tr><td>OCRBench</td><td>1,000</td><td>81.30</td><td>81.00</td><td>-0.30</td><td> $[ - 1 . 9 0 , + 1 . 3 0 ]$ </td><td>0.8043</td><td>1.0000</td><td>0.8937</td></tr><tr><td>MathVista</td><td>1,000</td><td>80.90</td><td>80.80</td><td>-0.10</td><td>[−2.00, +1.80]</td><td>1.0000</td><td>1.0000</td><td>1.0000</td></tr><tr><td>MMVet</td><td>218</td><td>69.45</td><td>72.02</td><td>+2.57</td><td>-0.69, +5.96]</td><td>0.1398</td><td>0.6991</td><td>0.2330</td></tr><tr><td>HallusionBench</td><td>951</td><td>74.03</td><td>74.87</td><td>+0.84</td><td>[−1.46, +3.16]</td><td>0.5312</td><td>1.0000</td><td>0.6640</td></tr><tr><td>MMStar</td><td>1,500</td><td>69.60</td><td>70.93</td><td>+1.33</td><td>[−0.60, +3.27]</td><td>0.1939</td><td>0.7756</td><td>0.2770</td></tr><tr><td>MMMU</td><td>900</td><td>67.78</td><td>70.22</td><td>+2.44</td><td>[0.00, +4.89]</td><td>0.0672</td><td>0.4702</td><td>0.1679</td></tr><tr><td>AI2D</td><td>3,088</td><td>82.09</td><td>83.97</td><td>+1.88</td><td>[+0.68, +3.08]</td><td>0.0023</td><td>0.0210</td><td>0.0117</td></tr><tr><td>MMBench-EN</td><td>1,292</td><td>83.75</td><td>86.92</td><td>+3.17</td><td>[+1.86, +4.49]</td><td> $4 . 1 9 \times 1 0 ^ { - 6 }$ </td><td>4.19 × 10−5</td><td>4.19 × 10−5</td></tr><tr><td>MMBench-CN</td><td>1,292</td><td>81.73</td><td>83.75</td><td>+2.01</td><td>[+0.54, +3.48]</td><td>0.0088</td><td>0.0702</td><td>0.0293</td></tr><tr><td>Benchmark V2.1</td><td>1,355</td><td>91.66</td><td>92.62</td><td>+0.96</td><td>[−0.07, +1.99]</td><td>0.0919</td><td>0.5515</td><td>0.1838</td></tr><tr><td>10-task Average</td><td></td><td>78.23</td><td>79.71</td><td>+1.48</td><td>[+0.86, +2.11]</td><td> $5 . 0 0 \times 1 0 ^ { - 6 }$ </td><td></td><td></td></tr></table>

## D SUPPLEMENTARY EXPERIMENTAL RESULTS

## D.1 COMPLETE STATISTICAL TESTS FOR KEYE30B

We conduct paired statistical tests to assess the reliability of the Keye30B results. We compare the Baseline and Lngram v2 (routes=256) checkpoints at training step 5,000, each evaluated once on the same ten benchmark datasets and splits using identical benchmark definitions and deterministic decoding settings. The reported scores are obtained from the frozen evaluation outputs. For statistical testing, predictions are aligned by sample identifier, and scorer-produced per-sample scores before rounding are used. We further verify that aligned samples have identical reference answers and benchmark metadata. Thus, all differences and uncertainty estimates are computed at the paired sample level rather than from rounded aggregate scores.

For benchmarks with binary per-sample outcomes, we use the two-sided exact paired McNemar test. MMVet provides fractional per-sample scores and is analyzed using a paired sign-flip permutation test. For HallusionBench, where multiple questions may share the same image, permutation testing and bootstrap resampling are performed over 346 image clusters rather than 951 individual questions. For MMMU, we use the 900 examples in the official validation split and construct confidence intervals using category-stratified paired bootstrap. For MMBench, n = 1,292 corresponds to the unique scored questions after aggregation of circularized inference rows. All remaining confidence intervals use ordinary paired bootstrap. We use 200,000 bootstrap replicates for 95% confidence intervals and 1,000,000 sign-flip draws for MMVet and HallusionBench. The ten per-benchmark two-sided p-values are corrected using both Holm (Holm, 1979), which controls the family-wise error rate, and Benjamini–Hochberg (BH) (Benjamini & Hochberg, 1995), which controls the false discovery rate.

Lngram v2 improves the equally weighted ten-task average by 1.48 percentage points, with a fixedsuite 95% paired-bootstrap confidence interval of [0.86, 2.11] and a two-sided paired-randomization p-value of $5 . 0 0 \times 1 0 ^ { - 6 }$ . The improvement is also consistent across benchmarks: Lngram v2 outperforms the Baseline on eight of the ten tasks. An exact sign-flip test over the ten task effects, which accounts for both their directions and magnitudes, yields $p = 0 . 0 0 7 8$ . Together, these results show that the overall gain is not driven by a small number of isolated benchmarks, but reflects a broadly positive effect across the fixed evaluation suite.

At the individual-benchmark level, AI2D and MMBench-EN remain statistically significant after the conservative Holm correction, while MMBench-CN is additionally significant under BH correction. In particular, Lngram v2 achieves gains of +1.88, +3.17, and +2.01 percentage points on AI2D, MMBench-EN, and MMBench-CN, respectively. These results provide strong evidence that Lngram v2 yields reliable improvements on Keye30B, with both a significant aggregate gain and clear improvements on several challenging multimodal benchmarks.

Table 12: Complete inference-efficiency results on a single H800 GPU. The three prefill values correspond to sequence lengths of 256, 512, and 1024, respectively.
<table><tr><td>Configuration</td><td>Prefill Latency (ms) ↓</td><td>Prefill Memory (GiB) ↓</td><td>Decode Latency (ms/token) ↓</td><td>Decode Memory (GiB) ↓</td></tr><tr><td>Baseline</td><td>54.62 / 54.98 / 56.99</td><td>4.90 / 5.01 / 5.24</td><td>48.26</td><td>4.89</td></tr><tr><td>r16/m4/KV8</td><td>58.98 / 58.92 / 59.08</td><td>4.96 / 5.07 / 5.31</td><td>50.97</td><td>4.95</td></tr><tr><td>r64/m4/KV8</td><td>57.84 / 57.86 / 60.49</td><td>5.08 / 5.21 / 5.46</td><td>52.14</td><td>5.05</td></tr><tr><td>r23/m4/KV8/mem256</td><td>58.03 / 58.59 / 60.85</td><td>5.06 / 5.17 / 5.40</td><td>51.79</td><td>5.05</td></tr><tr><td>r512/m4/KV8/mem256</td><td>57.88 / 57.59 / 83.75</td><td>8.67 / 10.34 / 13.68</td><td>51.44</td><td>7.09</td></tr><tr><td>r16/m5/KV2</td><td>56.75 / 57.15 / 59.46</td><td>5.19 / 5.30 / 5.53</td><td>49.92</td><td>5.18</td></tr><tr><td>r123/m4/KV2</td><td>57.21 / 57.59 / 59.96</td><td>5.19 / 5.30 / 5.53</td><td>50.55</td><td>5.18</td></tr><tr><td>r121/m4/KV16</td><td>60.75 / 64.17 / 73.80</td><td>5.48 / 5.87 / 6.67</td><td>50.74</td><td>5.18</td></tr></table>

## D.2 COMPLETE INFERENCE-EFFICIENCY RESULTS

We evaluate the inference efficiency of different Lngram v2 configurations on a single H800 GPU. Prefill is measured using input lengths of 256, 512, and 1024, while the decode stage reports singletoken latency. Table 12 presents the complete results.

The additional online computational overhead of standard configurations is generally modest. For example, for r64/m4/KV8 at sequence length 1024, prefill latency increases from 56.99 ms for the Baseline to 60.49 ms, corresponding to an overhead of approximately 6.1%, while memory usage increases from 5.24 GiB to 5.46 GiB. Decode latency increases from 48.26 ms/token to 52.14 ms/token, corresponding to approximately 8.0% overhead, while memory usage increases only from 4.89 GiB to 5.05 GiB.

In contrast, the extremely large r512/m4/KV8/mem256 configuration incurs substantially highe long-sequence memory usage and prefill overhead, consistent with the diminishing returns from capacity scaling observed in the ablation experiments in the main text. In practical deployments, moderate-scale configurations therefore provide a more favorable trade-off among downstream per formance, parameter capacity, and online computation cost.

## D.3 EXPERIMENTAL SETUP FOR THE CONTROLLED COMPARISON OF LNGRAM V1 AND V2

The comparison between Lngram v1 and v2 uses the same 0.6B DeepSeek-V2-style sparse MoE model (DeepSeek-AI, 2024; Dai et al., 2024). The model contains 24 Transformer layers with a hidden size of 768, 12 attention heads, and 2 KV heads. Each MoE layer contains 8 routed experts and 1 shared expert, with 2 routed experts activated per token.

Lngram is inserted into the 2nd and 12th layers in all variants and uses 4-bit discrete codes together with 2/3-gram memory. All models are trained from random initialization for 10,000 steps on the same FineWeb-Edu data (Penedo et al., 2024), processing approximately 1.31B tokens in total, with exactly the same data order, optimizer, and random seed. Therefore, the differences reported in Table 4 of the main text primarily reflect the architectural differences between Lngram v1 and v2 rather than changes in training data or optimization settings.

## E HYPERPARAMETER SETTINGS

For reproducibility, Table 13 summarizes the main hyperparameters of Lngram v2 in the Keye2B and Keye30B comparison experiments. At both model scales, Lngram v2 is inserted into a shallow and an intermediate layer of the backbone, uses 4-bit discrete codes and 2/3-gram memory, and employs a multi-head readout to retrieve the memory representations.

The Keye30B experiment is trained on 8 nodes with a total of 64 GPUs. Tensor parallelism, pipeline parallelism, context parallelism, expert parallelism, and expert tensor parallelism are set to

$$
\mathrm { T P } = 1 , \quad \mathrm { P P } = 4 , \quad \mathrm { C P } = 1 6 , \quad \mathrm { E P } = 8 , \quad \mathrm { E T P } = 1 ,
$$

respectively. The sequence length is 32,768, with a micro batch size of 1 and a global batch size of 32, resulting in approximately 1.049M tokens per training step. The formal comparison uses the checkpoint at step 5000, corresponding to approximately 5.24B cumulative training tokens. Training uses BF16 with a random seed of 19260817. The base learning rate of the backbone is $5 \times 1 0 ^ { - 6 }$ with a weight decay of 0.1, followed by cosine learning-rate decay after a linear warmup over the first 1,000 steps.

Table 13: Lngram v2 hyperparameter settings for Keye2B and Keye30B.
<table><tr><td>Hyperparameter</td><td>Keye2B</td><td>Keye30B</td></tr><tr><td>Insertion Layers</td><td>2,14</td><td>2,24</td></tr><tr><td>Number of Routes R</td><td>121</td><td>256</td></tr><tr><td>Bits per Route</td><td></td><td>4 bit</td></tr><tr><td>Codes per Route</td><td rowspan="3"></td><td>16</td></tr><tr><td>n-gram Orders</td><td>2,3</td></tr><tr><td>Memory Dimension 128</td><td>256</td></tr><tr><td>Readout Q Heads</td><td></td><td>16</td></tr><tr><td>Readout KV Heads</td><td></td><td>16</td></tr><tr><td>Head Dimension</td><td></td><td>128</td></tr><tr><td>Readout Mode</td><td>full</td><td>streaming</td></tr><tr><td>Route Chunk Size</td><td></td><td>64</td></tr><tr><td>Streaming Activation Checkpoint</td><td></td><td>enabled</td></tr><tr><td>Sink</td><td></td><td>initial effective-route mass 0.5</td></tr><tr><td>Router Surrogate Gradient</td><td></td><td>approximate</td></tr><tr><td>Lngram Dropout</td><td></td><td>0</td></tr><tr><td>Readout Attention Dropout</td><td></td><td>0</td></tr><tr><td>Short Convolution Table Initialization</td><td></td><td>kernel= 4, dilation= 3, no bias, zero-init</td></tr><tr><td></td><td></td><td>normal, std scale= 1.0</td></tr><tr><td>Output Projection Initialization</td><td></td><td>zero-init</td></tr><tr><td>Lngram Non-Table LR</td><td>peak  $1 \times 1 0 ^ { - 4 }$ </td><td>, min  $2 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Table LR</td><td>peak  $4 \times 1 0 ^ { - 4 } ,$ </td><td>min  $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Table / Router-Q Weight Decay</td><td></td><td>0</td></tr><tr><td>Lngram Total Parameters</td><td>155M</td><td>595M</td></tr><tr><td>Steady-State Active Parameters / Token</td><td>21M</td><td>25M</td></tr></table>

## F GENERATIVE AI USE STATEMENT

Generative AI tools were used during the preparation of this work for language polishing and to assist with code development for experiments. All AI-assisted code was reviewed, tested, and val idated by the authors before use. Generative AI was not used to generate, fabricate, or modify experimental results. All results reported in this paper were obtained from actual model training and evaluation runs and were verified by the authors. The authors take full responsibility for the content, methodology, implementation, and conclusions of this work.