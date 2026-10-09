# DPPM: Dual-Path Parametric Memory for Personalized Language Models

Yuhao Chen<sup>1</sup>, Shuochen Liu<sup>1</sup>, Jiayao Shi<sup>1</sup>, Jian Hong<sup>1,2</sup> Chen Cheng<sup>2</sup>, Xinyun Ding<sup>2</sup>, Tao Wang<sup>2</sup>, Ya Li<sup>2</sup>, Quan Liu<sup>2</sup>, Tong Xu<sup>1</sup> <sup>1</sup>University of Science and Technology of China <sup>2</sup>iFLYTEK Research Group isyuhaochen@mail.ustc.edu.cn

## Abstract

Long-term personalization requires language models to use interaction history to track users preferences across sessions. Parametric memory encodes this interaction history into model parameters or adapters, reducing the need to include it in the inference context. However, independent context compilation leaves crosssession integration unspecified, while recurrent updates can attenuate earlier evidence. To address these challenges, we propose Dual-Path Parametric Memory (DPPM). Its Evidence path directly pools representations of the interaction history to preserve earlier evidence, while its Delta path sequentially updates an associative state to capture changes. Fusing both outputs produces history-conditioned LoRA adapters that combine evidence accumulation with ordered revision. Across multiple backbones, DPPM outperforms the evaluated baselines, achieving 54.22% on PersonaMem-v2 and 86.79% on PrefEval. These results suggest that DPPM provides a simple and effective design choice for cross-session personalized parametric memory.<sup>1</sup>

## 1 Introduction

As large language models (LLMs) become persistent personal assistants, personalization increasingly depends on long-term memory. Such memory must preserve users’ preferences, backgrounds, and constraints across sessions to inform subsequent responses (Liu et al., 2026a; Chen et al., 2026). To support this continuity, existing systems maintain textual records and construct response contexts through summarization, retrieval, or explicit memory editing (Zhong et al., 2023; Packer et al., 2024; Chhikara et al., 2025). While these records are easy to inspect and maintain, their effectiveness depends on accurate extraction, retrieval, and contextual reasoning, and including more of the interaction history increases inference-time context costs.

![](images/080664a07ab9356b7b0fccc53be66e72daaf7b15a1ed55d0e046b7a6e3bf002c.jpg)  
Figure 1: Illustrative comparison of cross-session memory. (a) Context compilation produces separate LoRA adapters. (b) Recurrent updates may dilute earlier evidence. (c) DPPM combines evidence accumulation and sequential revision to preserve dietary constraints while tracking preference changes.

To reduce reliance on the interaction history at inference time, alternative approaches encode it into parameters, adapters, or internal states along two directions. Context compilation methods, such as Doc-to-LoRA (Charakorn et al., 2026) and SHINE (Liu et al., 2026b), use hypernetworks to map individual contexts to LoRA adapters. State maintenance methods integrate information sequentially. RPMem (Zhao et al., 2026) recurrently consolidates session representations into memory adapters, while Metis (Zhang et al., 2026) updates memory matrices within the backbone.

Despite advances in encoding interaction history and updating memory across sessions, two limitations remain. First, independently compiling segments into adapters does not itself support selective retention and revision as new sessions arrive (Charakorn et al., 2026). Second, recurrent consolidation enables revision, but successive updates to a shared state can attenuate earlier contributions (Zhao et al., 2026; Yang et al., 2025). These limitations motivate preserving direct access to earlier evidence while supporting selective revision as the interaction history grows.

To address these challenges, we propose Dual-Path Parametric Memory (DPPM), which separates evidence accumulation from sequential state updates within a unified parametric memory interface. The Evidence path directly aggregates representations of the interaction history, while the Delta path maintains an associative state in chronological order. Their outputs jointly form the final memory, combining accumulated evidence with orderdependent updates.

Specifically, DPPM learns cross-session memory consolidation on top of a frozen Doc-to-LoRA compiler, avoiding additional compiler training. The compiler encodes chronological segments of the interaction history into representations shared by both paths. In the Evidence path, coordinate-wise, content-dependent weighting directly aggregates these representations, allowing early evidence to contribute without passing through successive state updates. In the Delta path, the same segment representations are processed in chronological order. A content-derived associative key retrieves a prediction of the current segment representation from the existing state, and the prediction residual is written back to capture order-dependent changes. Finally, learnable fusion and scale calibration combine the two outputs, and a frozen decoder converts the fused memory into history-conditioned LoRA adapters for personalized generation. Training updates only the lightweight memory modules, while the compiler, decoder, and language-model backbone remain frozen.

We evaluate DPPM on PersonaMem-v2 (Jiang et al., 2025) and PrefEval (Zhao et al., 2025) across three backbones spanning different architectures and model sizes (2B–7B parameters). With Qwen3- 4B, DPPM outperforms the evaluated baselines, achieving 54.22% accuracy on PersonaMem-v2 and 86.79% accuracy on PrefEval. These results suggest that DPPM offers a simple and effective approach to cross-session personalized parametric memory. In summary, this work makes the following three key contributions:

(1) We propose DPPM, separating evidence accumulation from sequential updates for cross-session memory without compiler retraining.

(2) We analyze how direct accumulation mitigates evidence attenuation and sequential updates capture order-dependent changes.

(3) We validate DPPM across benchmarks, model architectures, and scales through ablations and controlled experiments.

## 2 Preliminary

## 2.1 Parametric Memory Workflow

Let $\mathcal { C } _ { 1 : T } = ( c _ { 1 } , \ldots , c _ { T } )$ denote the interaction history, divided into $T$ chronological segments. Each segment $c _ { t }$ is a contiguous portion of the interaction history. The consolidated memory is

$$
M _ { T } = U ( m _ { 1 } , . . . , m _ { T } ) ,\tag{1}
$$

where $m _ { t } = E ( c _ { t } )$ is the compiled memory for segment $c _ { t } ,$ and $U$ aggregates these memories in chronological order. A decoder produces memory parameters $\Lambda _ { T } = D ( M _ { T } )$ . Let θ denote the frozen backbone parameters and ⊕ the application of adapter updates, giving $\theta _ { T } = \theta \oplus \Lambda _ { T }$ . For a new query x, the response distribution is

$$
p _ { \theta _ { T } } ( y \mid x ) = \prod _ { j = 1 } ^ { | y | } p _ { \theta _ { T } } ( y _ { j } \mid x , y _ { < j } ) .\tag{2}
$$

Here $y _ { j }$ is the jth response token and $y _ { < j }$ its prefix. Inference uses compiled memory without the interaction history in the prompt.

## 2.2 Hypernetwork-Generated LoRA

Following Doc-to-LoRA (Charakorn et al., 2026), the hypernetwork $H _ { \phi } = D \circ E _ { \phi }$ , parameterized by $\phi ,$ comprises a context encoder $E _ { \mathrm { { : } } }$ , which maps text to latent memory, and an adapter decoder $D ,$ which generates LoRA factors. For a segment $c$ of the interaction history and target matrix $W _ { \ell }$ , indexed by ℓ, these factors are

$$
\begin{array} { r } { ( A _ { \ell } ( c ) , B _ { \ell } ( c ) ) = H _ { \phi } ^ { ( \ell ) } ( c ) . } \end{array}\tag{3}
$$

They adapt the frozen weight through

$$
W _ { \ell } ( c ) = W _ { \ell } + s _ { \ell } B _ { \ell } ( c ) A _ { \ell } ( c ) .\tag{4}
$$

Here $r _ { \mathrm { L } }$ is the LoRA rank, $A _ { \ell } \in \mathbb { R } ^ { r _ { \mathrm { L } } \times d _ { \mathrm { i n } } }$ , and $B _ { \ell } \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times r _ { \mathrm { L } } }$ . The dimensions $d _ { \mathrm { i n } }$ and $d _ { \mathrm { o u t } }$ are the input and output widths of $W _ { \ell }$ , and $s _ { \ell }$ scales the update. Once trained, the shared hypernetwork generates adapters in a forward pass without persegment optimization.

## 3 Method

## 3.1 Overview and Setup

Using the interaction history $\mathcal { C } _ { 1 : T }$ defined in Section 2, we encode each segment $c _ { t }$ with the frozen compiler E. At a fixed position in the compiled memory $m _ { t }$ , let $q _ { t } \in \mathbb { R } ^ { d }$ denote the segment representation, where d is the latent width. We share consolidation parameters across positions and exclude the query and answer from the history.

![](images/4bcf20eb1e38142c6c652e877534a1db2aa05905e9fbc0238f7531e5addefa1f.jpg)  
Figure 2: Overall architecture of DPPM. A frozen compiler encodes the interaction history segment by segment, and DPPM consolidates the resulting representations through Evidence pooling and sequential Delta updates. Their calibrated outputs are fused and decoded into history-conditioned LoRA adapters.

DPPM instantiates the aggregation operator in Section 2 with two paths (Figure 2). Evidence directly pools segment representations, while Delta updates an associative state in chronological order. Their fused output h is decoded into a historyconditioned adapter. We first motivate the two paths, then define their computations.

## 3.2 Motivation for Dual-Path Memory

For the same interaction history $\mathcal { C } _ { 1 : T }$ , consider a recurrent memory. Its state after segment t is $\mu _ { t } = F _ { t } ( \mu _ { t - 1 } , q _ { t } )$ , where $F _ { t }$ updates the previous state using the segment representation $q _ { t } .$ . Compare two interaction histories that differ at an earlier segment i but share all later segments, and write $\Delta \mu _ { t } = \mu _ { t } - \mu _ { t } ^ { \prime }$ for their state difference. If every subsequent update contracts this difference by at most a factor $0 \leq \kappa < 1$ , then

$$
\begin{array} { r } { \| \Delta \mu _ { T } \| _ { 2 } \leq \kappa ^ { T - i } \| \Delta \mu _ { i } \| _ { 2 } , } \end{array}\tag{5}
$$

where $\| \cdot \| _ { 2 }$ is the Euclidean norm and $T - i$ is the age of the earlier contribution. Thus, its influence may attenuate through repeated updates. Proposition E.1 gives sufficient conditions on the full update Jacobian, including state-dependent gates.

Evidence instead connects each segment representation directly to the pooled memory. Let $w _ { i , j }$ be segment i’s pooling weight for latent coordinate $j .$ If the largest and smallest scores across segments differ by at most a constant B that does not grow

with T, then

$$
w _ { i , j } \geq [ 1 + ( T - 1 ) e ^ { B } ] ^ { - 1 } .\tag{6}
$$

For changes to $q _ { i }$ that leave its scores unchanged, this also bounds how much the raw pooled vector changes relative to the input change (Proposition E.2). This lower bound depends on history length T but not segment position i, so earlier and later segments share the same bound. Since it still decreases with T, direct accumulation is motivated as a way to mitigate age-related attenuation, rather than eliminate forgetting.

However, Evidence pooling gives the same result when fixed segment representations are reordered, so it cannot distinguish opposite update orders requiring different answers (Proposition E.3). Delta supplies ordered revision.

## 3.3 Evidence Accumulation

To provide a direct route from the interaction history to memory, Evidence weights and pools the segment representations. Using normalized representations $z _ { t }$ of the segment $c _ { t } .$ , we apply a learned projection $W _ { E } \in \mathbb { R } ^ { d \times d }$ and coordinate-wise temporal softmax to obtain

$$
\begin{array} { l l } { s _ { t } = W _ { E } z _ { t } , \quad w _ { t , j } = \displaystyle \frac { \exp \left( s _ { t , j } \right) } { \sum _ { \tau = 1 } ^ { T } \exp \left( s _ { \tau , j } \right) } , } \\ { \displaystyle \bar { e } = \displaystyle \sum _ { t = 1 } ^ { T } w _ { t } \odot q _ { t } . } \end{array}\tag{7}
$$

(8)

Here $s _ { t } , w _ { t } \in \mathbb { R } ^ { d }$ are score and weight vectors, and ⊙ denotes element-wise multiplication. Each coordinate j pools segment representations using scores from normalized representations.

To compensate for magnitude changes from pooling, we rescale e¯ into the calibrated Evidence representation e. The target root-mean-square (RMS) scale $\rho _ { E }$ is derived from the coordinatewise weighted second moment of the segment representations. This rescaling, denoted by R, uses a small positive floor for numerical stability and also calibrates the Delta and fused representations. Appendix D gives the definitions.

## 3.4 Sequential Delta Updates

To complement Evidence with ordered revision, Delta processes the same segment representations sequentially. For key dimension K, it computes an unnormalized key $\boldsymbol { p } _ { t } \in \mathbb { R } ^ { K }$ , a unit key $k _ { t }$ , and a coordinate-wise write gate $\beta _ { t } \in ( 0 , 1 ) ^ { d }$

$$
p _ { t } = \mathrm { E L U } ( W _ { K } z _ { t } ) + \mathbf { 1 } _ { K } , \quad k _ { t } = \frac { p _ { t } } { \| p _ { t } \| _ { 2 } } ,\tag{9}
$$

$$
\beta _ { t } = \sigma ( W _ { \beta } z _ { t } + b _ { \beta } ) ,\tag{10}
$$

Here ELU is the element-wise exponential linear unit with unit parameter, $\sigma$ is the sigmoid, and ${ \bf 1 } _ { n }$ denotes an n-dimensional all-ones vector. The learned projections have shapes $W _ { K } \in \mathbb { R } ^ { K \times d }$ and $W _ { \beta } \in \bar { \mathbb { R } ^ { d \times \bar { d } } }$ , with bias $b _ { \beta } \in \mathbb { R } ^ { d }$ . Using this key and gate, Delta reads a prediction $\hat { q } _ { t } \in \mathbb { R } ^ { d }$ from the associative state $S _ { t - 1 } \stackrel { - } { \in } \mathbb { R } ^ { K \times d }$ and writes back a gated residual $d _ { t } \in \mathbb { R } ^ { d }$

$$
\hat { q } _ { t } = S _ { t - 1 } ^ { \top } k _ { t } ,\tag{11}
$$

$$
d _ { t } = \beta _ { t } \odot ( q _ { t } - \hat { q } _ { t } ) ,\tag{12}
$$

$$
S _ { t } = S _ { t - 1 } + k _ { t } d _ { t } ^ { \top } .\tag{13}
$$

The delta-rule updates (Schlag et al., 2021; Yang et al., 2025) correct the current key’s prediction. At another key $k ^ { \prime } \in \mathbb { R } ^ { K }$ , the readout changes by $( k _ { t } ^ { \top } k ^ { \prime } ) d _ { t }$ . Key similarity controls interference.

To stabilize the readout scale, a parallel normalizer $N _ { t } \in \mathbb { R } ^ { K \times d }$ follows the same updates as $S _ { t }$ using an all-ones target. After processing the interaction history, a learned read key $a = \operatorname { s o f t m a x } ( u )$ with $u \in \mathbb { R } ^ { \dot { K } }$ , gives the normalized readout

$$
\bar { r } = ( S _ { T } ^ { \top } a ) \oslash [ N _ { T } ^ { \top } a ] _ { \delta } ,\tag{14}
$$

where ⊘ denotes element-wise division and $[ \cdot ] _ { \delta }$ applies a small positive floor to the denominator. RMS calibration then produces the Delta representation r. Appendix D details the normalization, initialization, and calibration.

## 3.5 Fusion and Adapter Construction

We fuse Evidence e and Delta r using learned logits $v \in \mathbb { R } ^ { d }$ and weights $\alpha \in ( 0 , 1 ) ^ { d }$

$$
\alpha = \sigma ( v ) , \quad \bar { h } = \alpha \odot e + ( \mathbf { 1 } _ { d } - \alpha ) \odot r .\tag{15}
$$

RMS calibration then produces h using a target scale derived from both paths (Appendix D). Both α and the read key a remain fixed after training.

Finally, the frozen decoder generates LoRA factors $( A _ { \ell } ^ { \dot { H } } , B _ { \ell } ^ { H } ) = D ^ { ( \ell ) } ( h )$ , where $D ^ { ( \ell ) }$ denotes its output for target matrix $W _ { \ell }$ . Following Equation 4, the serving weight is

$$
\begin{array} { r } { W _ { \ell } ^ { H } = W _ { \ell } + s _ { \ell } B _ { \ell } ^ { H } A _ { \ell } ^ { H } + \Delta W _ { \ell , \mathrm { s h a r e d } } . } \end{array}\tag{16}
$$

The product $B _ { \ell } ^ { H } A _ { \ell } ^ { H }$ forms the history-conditioned low-rank update. The frozen shared-adapter update $\Delta W _ { \ell , \mathrm { s h a r e d } }$ is added once.

## 3.6 Answer Supervision

We train both paths and their fusion with crossentropy over answers. Gradients propagate through the decoder and backbone to update only the memory modules. The compiler, decoder, shared adapter, and backbone remain frozen.

## 4 Experiments

## 4.1 Experimental Setup

Dataset. Following RPMem (Zhao et al., 2026), we use the processed PersonaMem-v2 32K-Text split (Jiang et al., 2025) with 18,527 training, 2,059 validation, and 5,000 test examples, excluding test personas from training and validation. PrefEval (Zhao et al., 2025) uses a fixed topic split (seed 42), with 16 training and four held-out topics. Three preference forms at 10, 70, and 300 turns yield 7,380 training and 1,620 test instances.

Baselines. We compare No Context and Full Context, textual memories from Rolling Summary following RPMem (Zhao et al., 2026), top-10 BGE-M3 retrieval (Chen et al., 2025), Mem0 (Chhikara et al., 2025), and LightMem (Fang et al., 2026). Parametric baselines include D2L (Charakorn et al., 2026), PLUME (Shi et al., 2026), RPMem (Zhao et al., 2026), δ-mem (Lei et al., 2026), and Metis (Zhang et al., 2026). For a fair comparison, our RPMem baseline uses the same frozen D2L encoder as DPPM, without additional encodertraining data. We use released checkpoints for δ- mem and Metis. Gold State provides annotated user state as a reference.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="3">PersonaMem-v2</td><td colspan="4">PrefEval</td><td rowspan="2">Avg.</td></tr><tr><td>Overall</td><td>Self</td><td>Current</td><td>Overall</td><td>10</td><td>70</td><td>300</td></tr><tr><td rowspan="2">Reference</td><td>No Context</td><td>28.54</td><td>29.83</td><td>30.38</td><td>37.72</td><td>37.04</td><td>39.26</td><td>36.85</td><td>33.13</td></tr><tr><td>Gold State</td><td>61.08</td><td>67.44</td><td>67.44</td><td>93.52</td><td>92.78</td><td>94.07</td><td>93.70</td><td>77.30</td></tr><tr><td rowspan="5">Textual</td><td>Rolling Summary</td><td>29.16</td><td>30.93</td><td>31.12</td><td>40.49</td><td>38.70</td><td>40.00</td><td>42.78</td><td>34.83</td></tr><tr><td>LightMem</td><td>30.66</td><td>31.76</td><td>32.46</td><td>71.60</td><td>80.37</td><td>69.63</td><td>64.81</td><td>51.13</td></tr><tr><td>Mem0</td><td>30.46</td><td>31.93</td><td>33.06</td><td>75.49</td><td>77.96</td><td>76.11</td><td>72.41</td><td>52.98</td></tr><tr><td>RAG (BGE-M3)</td><td>32.50</td><td>34.75</td><td>34.40</td><td>57.96</td><td>65.56</td><td>57.59</td><td>50.74</td><td>45.23</td></tr><tr><td>Full Context</td><td>32.96</td><td>35.89</td><td>36.68</td><td>45.31</td><td>51.11</td><td>45.74</td><td>39.07</td><td>39.14</td></tr><tr><td rowspan="8">Parametric</td><td>Direct D2L</td><td>24.86</td><td>25.15</td><td>24.92</td><td>30.06</td><td>40.19</td><td>25.56</td><td>24.44</td><td>27.46</td></tr><tr><td>PLUME</td><td>30.98</td><td>32.80</td><td>34.63</td><td>40.12</td><td>45.74</td><td>39.63</td><td>35.00</td><td>35.55</td></tr><tr><td>δ-mem</td><td>30.58</td><td>32.72</td><td>34.18</td><td>37.22</td><td>36.11</td><td>37.96</td><td>37.59</td><td>33.90</td></tr><tr><td>Metis-4B</td><td>30.86</td><td>32.05</td><td>32.56</td><td>38.64</td><td>40.00</td><td>40.00</td><td>35.93</td><td>34.75</td></tr><tr><td>Metis-9B</td><td>29.14</td><td>28.45</td><td>30.05</td><td>38.09</td><td>36.48</td><td>39.63</td><td>38.15</td><td>33.62</td></tr><tr><td>Metis-27B</td><td>34.68</td><td>36.62</td><td>38.35</td><td>43.21</td><td>44.26</td><td>42.41</td><td>42.96</td><td>38.95</td></tr><tr><td>RPMem</td><td>45.73</td><td>49.46</td><td>53.38</td><td>82.47</td><td>79.56</td><td>83.81</td><td>84.04</td><td>64.10</td></tr><tr><td>DPPM (Ours)</td><td>54.22</td><td>54.69</td><td>58.38</td><td>86.79</td><td>84.52</td><td>88.15</td><td>87.70</td><td>70.51</td></tr></table>

Table 1: Main results on both benchmarks (accuracy, %). Self and Current select who=self and updated=False. PrefEval interval scores average three preference forms. Avg. averages the two Overall scores. Bold and underlining indicate the best and second-best scores, excluding the annotated Gold State reference. Metis uses its native Qwen3.5 backbone at the indicated size; the other methods use Qwen3-4B.

Hyperparameters. We use Qwen3-4B-Instruct-2507, Gemma-2-2B-IT, and Mistral-7B-Instructv0.2 with released D2L compilers. DPPM is trained separately per dataset for five epochs using AdamW, learning rate $1 0 ^ { - 3 }$ , weight decay 0.01, batch size 1, and gradient clipping at 1. We evaluate the final checkpoint of each run; further settings appear in Appendix B.

## 4.2 Main Results

Table 1 shows that DPPM reaches 54.22% on PersonaMem-v2 and 86.79% on PrefEval, improving over RPMem by 8.50 and 4.32 percentage points. The improvement extends across all seven reported metrics, including both information about the focal user and the currently valid state. On PrefEval, the advantage remains positive at every dialogue interval, including a 3.67-point gain after 300 intervening turns. Thus, the overall result reflects benefits across several evaluation conditions rather than a single favorable subset.

The strongest textual baseline differs across benchmarks, with Full Context leading on PersonaMem-v2 and Mem0 on PrefEval. DPPM exceeds their Overall scores by 21.26 and 11.30 points, respectively. Providing or retrieving text from the interaction history therefore leaves substantial room for better personalization in these settings. Direct D2L also trails the trained consolidation methods, suggesting that compiling the interaction history into parameters benefits from learning how its segments should be integrated for downstream answers. The significance comparison with RPMem appears in Appendix C.

## 4.3 Ablation Study

Cross-backbone architectural comparison. Table 2 compares the four memory designs within each backbone and its released compiler. DPPM outperforms RPMem on both benchmarks for Gemma, Qwen, and Mistral. The gains range from 8.50 to 13.98 points on PersonaMem-v2 and from 4.19 to 5.31 points on PrefEval. The largest gains occur with Mistral, while Gemma confirms gains with a smaller backbone.

Architectural ablations use Evidence only and Delta only as single-path controls to examine the complementarity of the two memory mechanisms. Their relative strengths vary across tasks and backbones, indicating that direct evidence accumulation and sequential state updates provide distinct memory signals. Combining both paths improves PersonaMem-v2 Overall over either standalone variant on all three backbones and PrefEval Overall on Qwen and Mistral. On Qwen, DPPM exceeds the stronger standalone path by 4.45 and 0.81 percentage points, respectively. These results support combining complementary memory signals to enhance cross-session personalization.

Parameter-matched comparison. To examine whether DPPM’s gains arise from a larger trainable budget, we compare the memory modules under the same frozen Qwen3-4B compiler–decoder interface. Original RPMem uses 524,800 trainable parameters, while DPPM uses only 2,564 more (0.49%). We also evaluate an RPMem control matched at 527,364 parameters.

<table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="3">PersonaMem-v2</td><td colspan="4">PrefEval</td></tr><tr><td>Overall</td><td>Self</td><td>Current</td><td>Overall</td><td>10</td><td>70</td><td>300</td></tr><tr><td rowspan="4">Gemma-2-2B IT</td><td>RPMem</td><td> $2 9 . 1 7 _ { \pm 0 . 3 4 }$ </td><td> $3 0 . 2 8 _ { \pm 0 . 4 0 }$ </td><td> $3 0 . 8 8 _ { \pm 0 . 4 4 }$ </td><td> $7 5 . 0 1 _ { \pm 0 . 9 2 }$ </td><td> $7 3 . 0 7 _ { \pm 1 . 7 0 }$ </td><td> $7 5 . 7 0 { \scriptstyle \pm 1 . 0 3 }$ </td><td> $7 6 . 2 6 { \scriptstyle \pm 1 . 3 0 }$ </td></tr><tr><td>Evidence only</td><td> $\underline { { 3 8 . 2 0 } } \mathrm { \pm 0 . 2 3 }$ </td><td> $\underline { { 4 1 . 3 4 } } \underline { { \pm 0 . 2 4 } }$ </td><td> $\underline { { 4 4 . 5 0 } } \pm 0 . 2 4$ </td><td> ${ \bf 7 9 . 8 1 { \bf _ { \pm 0 . 5 8 } } }$ </td><td> $7 7 . 5 2 { \scriptstyle \pm 0 . 7 1 }$ </td><td> $\mathbf { 8 1 . 3 3 \bot _ { \pm 1 . 1 9 } }$ </td><td> $\mathbf { 8 0 . 5 9 } _ { \pm 1 . 2 5 }$ </td></tr><tr><td>Delta only</td><td> $3 7 . 6 8 _ { \pm 0 . 6 9 }$ </td><td> $4 0 . 9 6 _ { \pm 0 . 8 3 }$ </td><td> $4 3 . 1 3 _ { \pm 0 . 9 5 }$ </td><td> $\underline { { 7 9 . 3 8 } } { \pm 1 . 2 4 }$ </td><td> $\underline { { 7 7 . 7 4 } } \pm 1 . 3 9$ </td><td> $\underline { { 8 0 . 1 5 } } { \scriptstyle \pm 1 . 0 9 }$ </td><td> $\underline { { 8 0 . 2 6 } } \pm 1 . 6 9 $ </td></tr><tr><td>DPPM</td><td> ${ \bf 4 0 . 3 1 { _ { \pm 0 . 5 4 } } }$ </td><td> ${ \bf 4 3 . 7 1 { _ { \pm 0 . 6 3 } } }$ </td><td> ${ \bf 4 6 . 7 7 { \scriptstyle \pm 0 . 7 9 } }$ </td><td> $7 9 . 2 0 { \scriptstyle \pm 0 . 7 6 }$ </td><td> $7 8 . 5 6 _ { \pm 0 . 4 4 }$ </td><td> $7 9 . 5 6 _ { \pm 0 . 5 9 }$ </td><td> $7 9 . 4 8 _ { \pm 2 . 6 0 }$ </td></tr><tr><td rowspan="4">Qwen3-4B Instruct-2507</td><td>RPMem</td><td> $4 5 . 7 3 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $4 9 . 4 6 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $5 3 . 3 8 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $8 2 . 4 7 { \scriptstyle \pm 0 . 5 8 }$ </td><td> $7 9 . 5 6 { \scriptstyle \pm 1 . 1 0 }$ </td><td> $8 3 . 8 1 { \scriptstyle \pm 1 . 1 9 }$ </td><td> $8 4 . 0 4 { \scriptstyle \pm 1 . 2 7 }$ </td></tr><tr><td>Evidence only</td><td> $4 8 . 0 9 _ { \pm 0 . 6 3 }$ </td><td> $5 0 . 3 9 _ { \pm 0 . 7 0 }$ </td><td> $5 4 . 0 6 _ { \pm 0 . 8 5 }$ </td><td> $8 4 . 5 9 _ { \pm 0 . 7 4 }$ </td><td> $8 1 . 6 7 _ { \pm 1 . 0 2 }$ </td><td> $8 6 . 0 0 { \scriptstyle \pm 1 . 8 9 }$ </td><td> $8 6 . 1 1 { \scriptstyle \pm 1 . 7 1 }$ </td></tr><tr><td>Delta only</td><td> $\underline { { 4 9 . 7 7 } } \pm 0 . 9 3 $ </td><td> $5 1 . 1 6 { \scriptstyle \pm 1 . 3 2 }$ </td><td> $5 4 . 8 0 { \scriptstyle \pm 1 . 2 2 }$ </td><td> $\underline { { 8 5 . 9 8 } } \underline { { \pm 0 . 8 4 } }$ </td><td> $\underline { { 8 4 . 4 8 } } { \scriptstyle \pm 0 . 2 4 }$ </td><td> $\underline { { 8 7 . 0 0 } } \pm 2 . 0 5$ </td><td> $8 6 . 4 4 _ { \pm 0 . 8 8 }$ </td></tr><tr><td>DPPM</td><td> $5 4 . 2 2 _ { \pm 0 . 4 8 }$ </td><td> ${ \pm 4 . 6 9 } _ { \pm 0 . 8 8 }$ </td><td> ${ \pm 8 . 3 8 } _ { \pm 0 . 9 4 }$ </td><td> $\mathbf { 8 6 . 7 9 } \mathrm { \pm 0 . 6 8 }$ </td><td> $\mathbf { 8 4 . 5 2 } _ { \pm 0 . 8 9 }$ </td><td> $\mathbf { 8 8 . 1 5 _ { \pm 1 . 0 1 } }$ </td><td> $\mathbf { 8 7 . 7 0 { \scriptstyle \pm 0 . 6 4 } }$ </td></tr><tr><td rowspan="4"> $\mathrm { M i s t r a l - 7 B }$  Instruct-v0.2</td><td>RPMem</td><td> $3 1 . 2 4 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $3 2 . 0 8 _ { \pm 0 . 7 1 }$ </td><td> $3 3 . 3 2 _ { \pm 0 . 6 4 }$ </td><td> $7 9 . 5 3 _ { \pm 0 . 3 2 }$ </td><td> $7 9 . 5 2 _ { \pm 0 . 8 3 }$ </td><td> $7 9 . 2 6 _ { \pm 0 . 5 7 }$ </td><td> $7 9 . 8 1 _ { \pm 1 . 2 3 }$ </td></tr><tr><td>Evidence only</td><td> $4 2 . 2 9 _ { \pm 0 . 2 7 }$ </td><td> $4 5 . 0 8 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $4 8 . 6 5 _ { \pm 0 . 3 4 }$ </td><td> $8 2 . 6 5 _ { \pm 0 . 8 9 }$ </td><td> $8 1 . 4 1 _ { \pm 1 . 0 4 }$ </td><td> $8 3 . 3 7 _ { \pm 1 . 8 6 }$ </td><td> $8 3 . 1 9 _ { \pm 1 . 7 3 }$ </td></tr><tr><td>Delta only</td><td> $3 9 . 9 8 _ { \pm 0 . 3 6 }$ </td><td> $4 2 . 0 2 _ { \pm 0 . 6 5 }$ </td><td> $4 5 . 1 9 _ { \pm 0 . 6 9 }$ </td><td> $\underline { { 8 4 . 4 9 } } { \scriptstyle \pm 1 . 1 9 }$ </td><td> $\underline { { 8 2 . 5 9 } } { \scriptstyle \pm 1 . 0 2 }$ </td><td> $\underline { { 8 5 . 0 4 } } \pm 1 . 1 9$ </td><td> $\underline { { 8 5 . 8 5 } } { \scriptstyle \pm 2 . 0 9 }$ </td></tr><tr><td>DPPM</td><td> $4 5 . 2 2 _ { \pm 0 . 3 3 }$ </td><td> $\mathbf { 4 7 . 7 8 _ { \pm 0 . 4 6 } }$ </td><td> ${ \bf 5 1 . 4 2 _ { \pm 0 . 5 0 } }$ </td><td> $\pm 4 . 8 4 _ { \pm 1 . 1 1 }$ </td><td> $\mathbf { 8 3 . 2 6 _ { \pm 0 . 8 6 } }$ </td><td> $\mathbf { 8 5 . 2 6 } _ { \pm 1 . 0 6 }$ </td><td> $\mathbf { 8 6 . 0 0 } _ { \pm 1 . 6 3 }$ </td></tr></table>

Table 2: Memory alternatives across three backbones (accuracy, %). All entries are five-run means, with sample standard deviations in subscripts. Best and second-best means are bold and underlined within each backbone.

The control retains RPMem’s recurrence, adding a rank-4 nonlinear gate correction with a biased 512→4 projection (2,052 parameters), a fixed orthonormal DCT basis, and 512 learned coordinate gains. Zero-initialized gains preserve the initial gate output; all added parameters are trained. All runs share data splits, frozen components, CE, optimizer settings, five epochs, seeds 42–46, and finalepoch evaluation.

<table><tr><td>Memory</td><td>Params</td><td>Persona</td><td>PrefEval</td></tr><tr><td>RPMem</td><td>524800</td><td> $4 5 . 7 3 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $8 2 . 4 7 _ { \pm 0 . 5 }$  8</td></tr><tr><td>RPMem-matched</td><td>527364</td><td> $4 5 . 4 9 _ { \pm 0 . 4 2 }$ </td><td> $8 2 . 5 6 { \scriptstyle \pm 0 . 9 0 }$ </td></tr><tr><td>DPPM</td><td>527364</td><td> $5 4 . 2 2 _ { \pm 0 . 4 8 }$ </td><td> $\mathbf { 8 6 . 7 9 } \mathrm { \pm 0 . 6 8 }$ </td></tr></table>

Table 3: Scores are five-run Overall accuracy means (%), with sample standard deviations in subscripts. Persona denotes PersonaMem-v2.

As shown in Table 3, increasing RPMem’s budget to match DPPM yields 45.49% on PersonaMem-v2 and 82.56% on PrefEval, close to the original gate’s results. With the same parameter count, DPPM improves these scores by 8.73 and 4.23 percentage points. This comparison supports dual-path consolidation beyond the small difference in trainable capacity.

## 4.4 Further Analysis

Accuracy, latency, and memory. Figure 4(a) shows that DPPM improves accuracy over RPMem by 8.50 percentage points at an answer-forward latency of 46.14 ms, only 2.43 ms higher. It also achieves higher accuracy than RAG and Full Context with approximately 3.1× and 62.7× faster answer computation, respectively.

To examine memory costs with longer histories, Figure 4(b) uses one question selected by maximum official 128K history length, evaluated with both official interaction history versions. Full Context inputs grow from 33,143 to 139,319 tokens, increasing peak GPU memory from 10,203.91 to 18,245.97 MiB. DPPM remains at 7,762.32 MiB, matching RPMem and using 57.46% less memory than Full Context for the longer history. Together, these results show improved personalization with low answer latency and a compact answering footprint after memory preparation.

Sources of improvement. Figure 3 links DPPM’s gains to the appropriate use of accumulated evidence. In PersonaMem-v2, Forget tests whether responses respect requests to stop using prior information, Sensitive tests restraint in using private details, and Therapy tests personalization from individual background (Jiang et al., 2025). DPPM improves these categories over RPMem by 21.68, 17.57, and 15.34 points, respectively. The Others subset tests whether information about another person is incorrectly attributed to the user; accuracy rises from 13.68% to 50.23%. These results indicate that much of DPPM’s improvement comes from better attribution and more appropriate revision of memory use.

This pattern aligns with our motivation to preserve evidence while revising its applicability. The Evidence path gives earlier representations direct access to the final memory, while Delta supplies ordered updates as new interactions arrive. DPPM also outperforms both single-path controls on Others and Forget, supporting their complementary roles. On PrefEval, the four topics test preference following in different application domains (Zhao et al., 2025). Gains across all four, from 1.90 points on Technology to 6.50 on Education, show broader use of remembered preferences. Alongside the latent-retention results below, these findings support combining evidence accumulation and state revision for personalization.

![](images/48416511e3cf5f8ad0fbcc6f40475451fb9ba6ac34c017edefad329cf5aa9ed5.jpg)

![](images/10be3ddaef4c63148304e6f484be0d18eb9ddc7f46fa82da22c5a6062f7a5845.jpg)  
Figure 3: Category-level comparison of RPMem and DPPM. Circles and diamonds show five-run mean accuracies, and the right-hand annotations give DPPM’s gain in percentage points. The panels cover all seven PersonaMem-v2 information types and all four held-out PrefEval topics. Axis ranges are shown separately for each benchmark.

![](images/83c15cd1661c33fb3c4ea88fab10d334af5b3b6a7e07e70ee0ca3f65042dae92.jpg)

![](images/687debb3e9a5b9790652fa4dd8b2078b05abc6b8da236d4341315b6914fa8966.jpg)  
Figure 4: Answer-stage efficiency on Qwen3-4B with one A800 (batch size 1). (a) Full-test accuracy versus weighted latency. (b) Peak GPU memory for paired histories of one length-selected question. Results use three-forward medians after one warmup, excluding compilation and retrieval; KV cache is disabled in (b).

Latent retention over time. To test whether direct evidence accumulation mitigates the attenuation of earlier information, we measure latent sensitivity as subsequent interactions accumulate. We freeze the seed-42 benchmark checkpoints and move one preference among 32 fixed distractors after a shared neutral initial segment, keeping each interaction history at 34 segments. For paired preference values A/B, latent sensitivity is $\| h _ { A } - h _ { B } \| _ { 2 } / \| q _ { A } - q _ { B } \| _ { 2 }$ , where h is the consolidated representation and q the representation of the preference segment. We normalize each profile’s sensitivity by its age-zero value and average over 24 profiles. Figure 5 shows that at age 32, DPPM retains 0.970/0.667 for PersonaMem-v2/PrefEval checkpoints, compared with RPMem’s 0.268/0.116 and Delta only’s 0.367/0.300. Evidence only stays at 1.000. This supports a less age-sensitive latent response from the direct evidence route.

Training objectives. To assess compatibility with different objectives, we fix DPPM’s architecture and compare CE with distillation and rewardbased variants (Table 4). Answer FKL matches teacher answer distributions. On-policy distillation (OPD) (Agarwal et al., 2024) applies tokenlevel reverse KL to student rollouts capped at 16 or 64 tokens, using a frozen teacher conditioned on training-only user state. One-step GRPO (Shao et al., 2024) uses correctness rewards for sampled answers. All variants retain CE.

The short OPD recipe and answer FKL score below CE on both datasets. The 64-token OPD configuration, which also uses the revised teacher prompt, reaches 54.56% and 87.04%, giving the highest two-dataset average of 70.80%. GRPO attains the highest PersonaMem-v2 score but trades off PrefEval accuracy, reducing its average below CE. These results suggest that appropriately configured on-policy feedback can complement the dualpath architecture, while plain CE remains competitive with simpler training.

![](images/3441763c039530057a4eea38e11e7170efc848dfe24355a5b04d20ab01b623fb.jpg)

Figure 5: Latent retention versus the number of subsequent distractor segments. Curves average 24 profiles after within-profile age-zero normalization.
<table><tr><td>Objective</td><td>Persona</td><td>PrefEval</td><td>Avg.</td></tr><tr><td>CE (DPPM)</td><td>54.22</td><td>86.79</td><td>70.51</td></tr><tr><td>+ Answer FKL</td><td>53.98</td><td>86.05</td><td>70.02</td></tr><tr><td>+ OPD (16)</td><td>52.96</td><td>85.80</td><td>69.38</td></tr><tr><td>+ OPD (64)</td><td>54.56</td><td>87.04</td><td>70.80</td></tr><tr><td>+ GRPO</td><td>54.80</td><td>85.06</td><td>69.93</td></tr></table>

Table 4: Training objectives on DPPM (accuracy, %). Persona denotes PersonaMem-v2. OPD (16/64) caps student rollouts at 16/64 tokens.

## 5 Related Work

## 5.1 Personalized Memory

Personalized memory must distill growing histories into useful information for future responses. MemoryBank and MemGPT maintain persistent records and retrieve relevant content when needed (Zhong et al., 2023; Packer et al., 2024). As records accumulate, storage must be complemented by content selection and revision. Mem0 and A-MEM address these needs through memory extraction, organization, and updating (Chhikara et al., 2025; Xu et al., 2025). Recent work learns adaptive memory strategies that balance shared knowledge and individual needs (Xu et al., 2026; Han et al., 2026).

These developments broaden evaluation from recalling history to using it appropriately. LoCoMo and LongMemEval assess conversational recall and reasoning (Maharana et al., 2024; Wu et al., 2025), while PrefEval and PersonaMem-v2 examine preference following and implicit personalization (Zhao et al., 2025; Jiang et al., 2025). Evolving preferences and multi-user decisions introduce further demands on memory maintenance (Liu et al., 2026a; Chen et al., 2026). We approach these demands through parametric memory for crosssession personalization.

## 5.2 Parametric Memory

Encoding information into parameters reduces reliance on interaction history at inference time, but requires an effective writing mechanism. ROME and MEMIT edit factual associations (Meng et al., 2023a,b), while FastMem internalizes prompts through optimization (Zhu et al., 2024). To avoid repeated optimization, Doc-to-LoRA and SHINE learn context-to-adapter hypernetworks that generate LoRA parameters in a forward pass (Charakorn et al., 2026; Liu et al., 2026b).

Efficient writing alone does not determine how accumulated memory should evolve. Sequential state updates address this requirement through associative rules in Fast Weight Programmers and Gated DeltaNet (Schlag et al., 2021; Yang et al., 2025), and learned adaptation in TTT and Titans (Sun et al., 2025; Behrouz et al., 2024). Building on state maintenance, Metis integrates forwardupdated memory into the backbone (Zhang et al., 2026). RPMem, the work closest to ours, connects recurrent session consolidation with adapter generation (Zhao et al., 2026). DPPM extends this approach with direct evidence accumulation alongside sequential revision, reducing reliance on a single recurrent state.

## 6 Conclusion

We introduced DPPM for integrating interaction history across sessions in personalized language models. DPPM combines evidence accumulation and sequential updates for retention and revision without extra compiler training. Experiments across two benchmarks and backbones of different architectures and sizes demonstrate its effectiveness, while theoretical analysis and controlled experiments support the complementary roles of the two paths. These findings highlight the value of preserving historical evidence while adapting its use, offering a practical design for long-term personalized parametric memory.

## Limitations

Although DPPM advances cross-session personalization through dual-path memory, our study has two limitations. First, computational constraints limit DPPM experiments to 2B–7B backbones, leaving larger-scale validation for future work. Second, our memory interface currently processes textual interaction histories. Extending it to images and speech is a natural next step.

## References

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. 2024. On-policy distillation of language models: Learning from self-generated mistakes. Preprint, arXiv:2306.13649.

Ali Behrouz, Peilin Zhong, and Vahab Mirrokni. 2024. Titans: Learning to memorize at test time. Preprint, arXiv:2501.00663.

Rujikorn Charakorn, Edoardo Cetin, Shinnosuke Uesaka, and Robert Tjarko Lange. 2026. Doc-to-lora: Learning to instantly internalize contexts. Preprint, arXiv:2602.15902.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2025. M3-embedding: Multilinguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2402.03216.

Yuhao Chen, Yi Xu, Xinyun Ding, Xiang Fang, Shuochen Liu, Luxi Lin, Qingyu Zhang, Ya Li, Quan Liu, and Tong Xu. 2026. Vehiclemembench: An executable benchmark for multi-user long-term memory in in-vehicle agents. Preprint, arXiv:2603.23840.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. Preprint, arXiv:2504.19413.

Jizhan Fang, Xinle Deng, Haoming Xu, Ziyan Jiang, Yuqi Tang, Ziwen Xu, Shumin Deng, Yunzhi Yao, Mengru Wang, Shuofei Qiao, Huajun Chen, and Ningyu Zhang. 2026. Lightmem: Lightweight and efficient memory-augmented generation. Preprint, arXiv:2510.18866.

Yupeng Han, Shuochen Liu, Kai Zhang, Ze Liu, Zhihong Pan, and Xianquan Wang. 2026. Learning what to share and what to personalize: Hierarchical strategy co-evolution for agent memory. Preprint, arXiv:2608.25329.

Bowen Jiang, Yuan Yuan, Maohao Shen, Zhuoqun Hao, Zhangchen Xu, Zichen Chen, Ziyi Liu, Anvesh Rao Vijjini, Jiashu He, Hanchao Yu, Radha Poovendran, Gregory Wornell, Lyle Ungar, Dan Roth, Sihao Chen, and Camillo Jose Taylor. 2025. Personamem-v2:

Towards personalized intelligence via learning implicit user personas and agentic memory. Preprint, arXiv:2512.06688.

Jingdi Lei, Di Zhang, Junxian Li, Weida Wang, Kaixuan Fan, Xiang Liu, Qihan Liu, Xiaoteng Ma, Baian Chen, and Soujanya Poria. 2026. δ-mem: Efficient online memory for large language models. Preprint, arXiv:2605.12357.

Shuochen Liu, Junyi Zhu, Long Shu, Junda Lin, Yuhao Chen, Haotian Zhang, Chao Zhang, Derong Xu, Jia Li, Bo Tang, Zhiyu Li, Feiyu Xiong, Enhong Chen, and Tong Xu. 2026a. Perma: Benchmarking personalized memory agents via event-driven preference and realistic task environments. Preprint, arXiv:2603.23231.

Yewei Liu, Xiyuan Wang, Yansheng Mao, Yoav Gelbery, Haggai Maron, and Muhan Zhang. 2026b. Shine: A scalable in-context hypernetwork for mapping context to lora in a single pass. Preprint, arXiv:2602.06358.

Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating very long-term conversational memory of llm agents. Preprint, arXiv:2402.17753.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2023a. Locating and editing factual associations in gpt. Preprint, arXiv:2202.05262.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. 2023b. Massediting memory in a transformer. Preprint, arXiv:2210.07229.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, and Joseph E. Gonzalez. 2024. Memgpt: Towards llms as operating systems. Preprint, arXiv:2310.08560.

Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. 2021. Linear transformers are secretly fast weight programmers. Preprint, arXiv:2102.11174.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Xiaobing Shi, Zherui Li, Yiming Jiang, Kun Wang, and Yufei Guo. 2026. Towards evolving context parameterization for large language models. Preprint, arXiv:2609.14168.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. 2025. Learning to (learn at test time): Rnns with expressive hidden states. Preprint, arXiv:2407.04620.

Di Wu, Hongwei Wang, Wenhao Yu, Yuwei Zhang, Kai-Wei Chang, and Dong Yu. 2025. Longmemeval: Benchmarking chat assistants on long-term interactive memory. Preprint, arXiv:2410.10813.

Derong Xu, Shuochen Liu, Pengfei Luo, Pengyue Jia, Yingyi Zhang, Yi Wen, Yimin Deng, Wenlin Zhang, Enhong Chen, Xiangyu Zhao, and Tong Xu. 2026. Learning how and what to memorize: Cognitioninspired two-stage optimization for evolving memory. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 44987–45011, San Diego, California, United States. Association for Computational Linguistics.

Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. 2025. A-mem: Agentic memory for llm agents. Preprint, arXiv:2502.12110.

Songlin Yang, Jan Kautz, and Ali Hatamizadeh. 2025. Gated delta networks: Improving mamba2 with delta rule. Preprint, arXiv:2412.06464.

Zeyu Zhang, Ziliang Guo, Yihang Sun, Xichong Zhang, Xixuan Hao, Zehao Lin, Yang Zhang, Xiaoyan Zhao, Tong Shen, Bo Tang, Zhi-Qin John Xu, Junchi Yan, Haofen Wang, Xu Chen, Feiyu Xiong, Zhiyu Li, and Tat-Seng Chua. 2026. Metis: Memory foundation model. Preprint, arXiv:2607.26760.

Fanyu Zhao, Ruike Cao, Liang Dong, Fugen Yao, Jian Xu, Guanjun Jiang, Han Zhang, Yifei Zhao, and Yinsheng Li. 2026. Rpmem: Learning long-term recurrent parametric memory across sessions for llm agents. Preprint, arXiv:2609.23466.

Siyan Zhao, Mingyi Hong, Yang Liu, Devamanyu Hazarika, and Kaixiang Lin. 2025. Do llms recognize your preferences? evaluating personalized preference following in llms. Preprint, arXiv:2502.09597.

Wanjun Zhong, Lianghong Guo, Qiqi Gao, He Ye, and Yanlin Wang. 2023. Memorybank: Enhancing large language models with long-term memory. Preprint, arXiv:2305.10250.

Junyi Zhu, Shuochen Liu, Yu Yu, Bo Tang, Yibo Yan, Zhiyu Li, Feiyu Xiong, Tong Xu, and Matthew B. Blaschko. 2024. Fastmem: Fast memorization of prompt improves context awareness of large language models. Preprint, arXiv:2406.16069.

## A AI Use Statement

Generative AI tools assisted with method implementation, proof drafting, manuscript writing and revision, and figure creation. They also supported discussions and refinement of the interpretation of experimental results. The authors reviewed the implementation and proofs and tested the code. The authors take full responsibility for the final content, including all AI-assisted text, claims, and artifacts.

## B Experimental Configuration

For each dataset and backbone, we train a separate memory module using the splits in Section 4.1. The released D2L tokenizer divides interaction histories into chronological segments. The compiler, decoder, shared adapter, and backbone remain frozen. We evaluate final-epoch checkpoints using the settings in Table 5.

<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>10-3</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Training epochs</td><td>5</td></tr><tr><td>Questions per update</td><td>1</td></tr><tr><td>Gradient norm clipping</td><td>1.0</td></tr><tr><td>Training seeds</td><td>42-46</td></tr><tr><td>Latent width d / key width K</td><td>512/4</td></tr><tr><td>RMS floor € / readout floorδ</td><td>10−⁶ / 10−4</td></tr></table>

Table 5: DPPM cross-entropy training settings.

For the auxiliary objectives, answer CE remains active. OPD samples 16 or 64 student tokens and uses teacher-scored reverse KL, with a one-epoch warmup, auxiliary weight 0.5, and one auxiliary update every four examples. GRPO samples eight answer actions with correctness rewards. Latency is measured on one A800 at batch size 1, with one warmup and three timed forwards per question.

## C Significance Analysis

We compare DPPM and RPMem on Qwen3-4B-Instruct-2507 across the seven component metrics in Table 1, excluding the derived Avg. column. For each metric, we pair runs with the same training seed (42–46) and apply a two-sided paired t-test to their accuracy differences. Holm correction controls the family-wise error rate at 0.05 within this seven-test family. The analysis assumes independent seeds and approximately normal seed-level differences on fixed evaluation sets.

All seven mean differences favor DPPM, and every Holm-adjusted $p \mathrm { - }$ value is below 0.01. The gains remain significant for both Overall metrics and all five diagnostic or interval-specific measures in this comparison.

## D Implementation Details

Representations and initialization. Each compiled memory $m _ { t }$ contains vectors indexed by layer, module, and latent slot. The method equations describe one such vector $q _ { t }$ for $T \geq 1$ segments.

<table><tr><td>Metric</td><td>Gain (pp)</td><td>Raw p</td><td> $\mathbf { A d j . } _ { p }$ </td></tr><tr><td colspan="4">PersonaMem-v2</td></tr><tr><td>Overall</td><td>+8.50</td><td> $1 . 8 7 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Self</td><td>+5.23</td><td> $1 . 0 4 \times 1 0 ^ { - 4 }$ </td><td> $6 . 2 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Current</td><td>+5.00</td><td> $2 . 4 8 \times 1 0 ^ { - 4 }$ </td><td> $1 . 2 4 \times 1 0 ^ { - }$  -3</td></tr><tr><td colspan="4">PrefEval</td></tr><tr><td>Overall</td><td>+4.32</td><td> $1 . 5 0 \times 1 0 ^ { - 3 }$ </td><td> $6 . 0 2 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>10 turns</td><td>+4.96</td><td> $2 . 1 9 \times 1 0 ^ { - 3 }$ </td><td> $6 . 5 6 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>70 turns</td><td>+4.33</td><td> $7 . 6 5 \times 1 0 ^ { - 3 }$ </td><td> $9 . 4 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>300 turns</td><td>+3.67</td><td> $4 . 7 3 \times 1 0 ^ { - 3 }$ </td><td> $9 . 4 7 \times 1 0 ^ { - 3 }$ </td></tr></table>

Table 6: DPPM versus RPMem on Qwen3-4B-Instruct-2507. Gains are in percentage points. Raw p-values use five paired training runs, and adjusted values use Holm correction across all seven metrics.

Learned projections are shared across positions, while memory states are maintained separately. We use non-affine LayerNorm for $z _ { t } ,$ with numerical constant $1 0 ^ { - 5 }$ , and compute consolidation in float32. Initially, $W _ { E } , \ W _ { \beta } ,$ , u, and v are zero, $b _ { \beta } = - \mathbf { 1 } _ { d }$ , and entries of $W _ { K }$ follow $\mathcal { N } ( 0 , 0 . 0 4 ^ { 2 } )$ Key normalization uses an L2-norm floor of $1 0 ^ { - 1 2 }$

Scale calibration. For $\xi \in \mathbb { R } ^ { d } ,$ , let ${ \mathrm { R M S } } ( \xi ) =$ $\| \xi \| _ { 2 } / \sqrt { d } .$ With target scale $\rho \geq 0$ and a small positive floor ϵ, the shared rescaling operator is

$$
\mathcal { R } ( \xi , \rho ) = \frac { \rho \xi } { \operatorname* { m a x } \{ \mathrm { R M S } ( \xi ) , \epsilon \} } .\tag{17}
$$

When the floor is inactive, rescaling preserves direction and sets the output RMS to $\rho .$ The calibrated Evidence representation is $e = \mathcal { R } ( \bar { e } , \rho _ { E } )$ with $\rho _ { E } \geq 0$ defined by

$$
\rho _ { E } ^ { 2 } = \frac { 1 } { d } \sum _ { t = 1 } ^ { T } w _ { t } ^ { \top } ( q _ { t } \odot q _ { t } ) .\tag{18}
$$

The fused representation uses $h = \mathcal { R } ( \bar { h } , \rho _ { H } )$ with nonnegative target scale

$$
\rho _ { H } ^ { 2 } = \frac { 1 } { d } \sum _ { j = 1 } ^ { d } [ \alpha _ { j } e _ { j } ^ { 2 } + ( 1 - \alpha _ { j } ) r _ { j } ^ { 2 } ] .\tag{19}
$$

Delta normalization and initialization. Using the same keys and gates as $S _ { t }$ , the normalizer tracks constant-target writes,

$$
N _ { t } = N _ { t - 1 } + k _ { t } [ \beta _ { t } \odot ( { \bf 1 } _ { d } - N _ { t - 1 } ^ { \top } k _ { t } ) ] ^ { \top } .\tag{20}
$$

Both recurrences start at $t = 2 ,$ , with $S _ { 1 } = k _ { 1 } q _ { 1 } ^ { \top }$ and $N _ { 1 } = k _ { 1 } \mathbf { 1 } _ { d } ^ { \top }$ . In Equation 14, $[ \xi ] _ { \delta } =$ max $( \xi , \delta )$ coordinate-wise, with a small positive floor δ. The

Delta representation is $r = \mathcal { R } ( \bar { r } , \rho _ { D } )$ , where $\rho _ { D } \geq$ 0 and

$$
\rho _ { D } ^ { 2 } = \frac { 1 } { T d } \sum _ { t = 1 } ^ { T } \| q _ { t } \| _ { 2 } ^ { 2 } .\tag{21}
$$

Evaluation conventions. The main DPPM, RP-Mem, and standalone-path comparisons use seeds 42–46. Auxiliary-objective variants use seed 42, with privileged teacher state confined to training. Their two-dataset Avg. is the arithmetic mean of the displayed Overall scores. PersonaMem-v2 Overall averages all 5,000 questions, while Self and Current select the annotations who=self and updated=False. PrefEval Overall averages all 1,620 instances; each interval averages three preference forms. Category gains and attribution subsets describe answer behavior.

Latent-retention protocol. The diagnostic uses frozen seed-42 checkpoints without additional training. Each of 24 profiles has a neutral prefix, one preference segment, and 32 fixed distractors. We move the preference to ages 0, 8, 16, 24, and 32. For each paired preference, the latent sensitivity in the main text is divided by that profile’s age-zero sensitivity before averaging profiles. The curves measure latent-response retention with fixed history length and distractors.

## E Analysis of the Two Memory Paths

Fix the learned parameters and let $Q = ( q _ { 1 } , \dotsc , q _ { T } )$ contain the segment representations at one latent position. The first two results compare sequences differing only at segment $i ;$ primes mark perturbed quantities. We use Euclidean and induced operator norms, with $\| \xi \| _ { \infty } = \operatorname* { m a x } _ { j } | \xi _ { j } |$ . The results respectively characterize recurrent attenuation, direct access to evidence, and the role of order-sensitive updates.

## E.1 Conditional Recurrent Attenuation

Proposition E.1 (Contractive state transitions). Let $\mu _ { t } = F _ { t } ( \mu _ { t - 1 } , q _ { t } )$ and $\mu _ { t } ^ { \prime } = F _ { t } ( \mu _ { t - 1 } ^ { \prime } , q _ { t } ^ { \prime } )$ , with $q _ { t } = q _ { t } ^ { \prime } f o r t > i .$ For each $t > i ,$ assume $F _ { t } ( \cdot , q _ { t } )$ is continuously differentiable near the line segment joining $\mu _ { t - 1 }$ and $\mu _ { t - 1 } ^ { \prime }$ , and $\| D _ { \mu } F _ { t } \| _ { 2 } \leq \kappa _ { t }$ throughout it, where $\kappa _ { t } \geq 0$ . Then

$$
\| \mu _ { T } - \mu _ { T } ^ { \prime } \| _ { 2 } \leq \Big ( \prod _ { t = i + 1 } ^ { T } \kappa _ { t } \Big ) \| \mu _ { i } - \mu _ { i } ^ { \prime } \| _ { 2 } .\tag{22}
$$

$I f 0 \leq \kappa _ { t } \leq \kappa < 1$ , the multiplier is at most $\kappa ^ { T - i }$

Proof. Set $\Delta _ { t } = \mu _ { t } - \mu _ { t } ^ { \prime }$ and $J _ { t } ( s ) =$ $D _ { \mu } F _ { t } ( \mu _ { t - 1 } ^ { \prime } + s \Delta _ { t - 1 } , q _ { t } )$ Integration along the connecting line segment gives

$$
\Delta _ { t } = \int _ { 0 } ^ { 1 } J _ { t } ( s ) \Delta _ { t - 1 } d s .\tag{23}
$$

Thus $\begin{array} { r } { \| \Delta _ { t } \| _ { 2 } \leq \int _ { 0 } ^ { 1 } \kappa _ { t } \| \Delta _ { t - 1 } \| _ { 2 } d s = \kappa _ { t } \| \Delta _ { t - 1 } \| _ { 2 } } \end{array}$ Iterating this inequality proves the bound. □

To connect this condition to gated consolidation, consider a state-dependent gate $g _ { t } ( \mu ) \in ( 0 , 1 ) ^ { d }$ and let $F _ { t } ( \mu ) = g _ { t } ( \mu ) \odot \mu + ( { \bf 1 } _ { d } - g _ { t } ( \mu ) ) \odot q _ { t }$ . Fixing q<sub>t</sub> gives

$$
D _ { \mu } F _ { t } = \mathrm { d i a g } ( g _ { t } ) + \mathrm { d i a g } ( \mu - q _ { t } ) D _ { \mu } g _ { t } .\tag{24}
$$

Using $\| \operatorname { d i a g } ( \xi ) \| _ { 2 } = \| \xi \| _ { \infty } ,$ , uniform bounds $\| g _ { t } \| _ { \infty } \leq \gamma _ { t } , \ \| \mu - q _ { t } \| _ { \infty } \leq R _ { t }$ , and $\| D _ { \mu } g _ { t } \| _ { 2 } \leq$ $L _ { t }$ permit $\kappa _ { t } = \gamma _ { t } + R _ { t } L _ { t }$ . This bound accounts for both state retention and gate sensitivity; values below one ensure contraction.

## E.2 Direct Evidence Sensitivity

Direct pooling in Equations 7–8 yields a positionindependent bound on the raw pooled vector, before RMS calibration.

Proposition E.2 (Bounded-score pooling). Suppose max<sub>t</sub> $s _ { t , j } - \operatorname* { m i n } _ { t } s _ { t , j } \leq B$ for each $j ,$ where $B \geq 0$ is independent ofT. Then $w _ { i , j } \geq b _ { T } = [ 1 +$ $( T - 1 ) e ^ { B } ] ^ { - 1 }$ . If only $q _ { i }$ changes to $q _ { i } ^ { \prime } = q _ { i } + \eta$ with $\eta \in \mathbb { R } ^ { d }$ and unchanged score vector $s _ { i } ^ { \prime } = s _ { i }$ then

$$
\lVert \bar { e } ^ { \prime } - \bar { e } \rVert _ { 2 } \geq b _ { T } \lVert \eta \rVert _ { 2 } .\tag{25}
$$

Proof. Since $s _ { t , j } - s _ { i , j } \leq B$ , dividing the softmax denominator by $e ^ { s _ { i , j } }$ gives

$$
w _ { i , j } = \frac { 1 } { 1 + \sum _ { t \neq i } e ^ { s _ { t , j } - s _ { i , j } } } \geq b _ { T } .\tag{26}
$$

Since all scores are unchanged, so are all weights. Hence $\bar { e } ^ { \prime } - \bar { e } = \mathrm { d i a g } ( w _ { i } ) \eta$ , and $\| \bar { e } ^ { \prime } - $ $\begin{array} { r } { \bar { e } \| _ { 2 } ^ { 2 } = \sum _ { j } w _ { i , j } ^ { 2 } \eta _ { j } ^ { 2 } \ge b _ { T } ^ { 2 } \| \eta \| _ { 2 } ^ { 2 } } \end{array}$ . Taking square roots proves the claim. □

Normalization bounds the scores independently of history length. Non-affine LayerNorm gives $\| \operatorname { L N } ( q ) \| _ { 2 } \leq { \sqrt { d } }$ . Cauchy–Schwarz gives $B =$ $2 \sqrt { d }$ max<sub>j</sub> $\Vert ( W _ { E } ) _ { j , : } \Vert _ { 2 }$ Also, $\mathrm { L N } ( q + \zeta \mathbf { 1 } _ { d } )$ $\operatorname { L N } ( q )$ for scalar ζ, giving score-preserving perturbations in real arithmetic.

For fixed B, $b _ { T } = \Theta ( T ^ { - 1 } )$ , independently of the source position i. If a nonzero perturbation also induces $\lVert \Delta \mu _ { i } \rVert _ { 2 } \leq L _ { \mathrm { i n } } \lVert \eta \rVert _ { 2 }$ , the raw Evidence lower bound exceeds the recurrent upper bound whenever $b _ { T } > L _ { \mathrm { i n } } \kappa ^ { T - i }$ . Here $L _ { \mathrm { i n } }$ bounds input sensitivity and κ bounds contraction.

## E.3 The Need for Ordered Revision

Direct evidence access preserves a contribution from each segment, but content-only pooling discards their order. For a permutation π, write $\pi Q = ( q _ { \pi ( 1 ) } , \ldots , q _ { \pi ( T ) } )$

Proposition E.3 (Order cannot be recovered from pooling). Under the Evidence scoring and calibration rules, $e ( \pi Q ) = e ( Q )$ for every permutation π of fixed segment representations. If Q and $\pi Q$ require different answers to the same query, a deterministic predictor using only that query and $e ( Q )$ cannot be correct on both.

Proof. Permuting inputs reorders scores and weights together. Both $\begin{array} { r } { \bar { e } = \sum _ { t } w _ { t } \odot q _ { t } } \end{array}$ and $\rho _ { E } ^ { 2 } =$ $\begin{array} { r } { d ^ { - 1 } { \sum _ { t } } w _ { t } ^ { \top } ( q _ { t } \odot q _ { t } ) } \end{array}$ remain unchanged, and therefore so does $e = \mathcal { R } ( \bar { e } , \rho _ { E } )$ . Identical predictor inputs produce the same answer, which cannot equal two distinct targets. □

Oppositely ordered conflicting updates under a last-update-wins rule illustrate this obstruction. Without separately encoded chronology, this motivates order-sensitive Delta updates.

Design implications. Under the stated conditions, direct Evidence accumulation mitigates the repeated attenuation of recurrent propagation. Its lower bound decreases with history length but carries no additional age factor. Pooling’s permutation invariance motivates the complementary Delta path for ordered revision. Together, the results support combining direct evidence access with sequential state updates.