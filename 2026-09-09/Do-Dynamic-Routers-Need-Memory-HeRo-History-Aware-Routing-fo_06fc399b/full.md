# Do Dynamic Routers Need Memory? HeRo: History-Aware Routing for Eficient LLM Inference

Hongjin Lin<sup>1∗</sup>, Wentao Wan<sup>1∗</sup>, Keze Wang<sup>1†</sup>

<sup>1</sup> Sun Yat-sen University linhj53@mail2.sysu.edu.cn, {wanwentao93,kezewang}@gmail.com

## Abstract

Dynamic layer routing reduces the inference cost of Large Language Models (LLMs) by learning to skip layers for individual tokens. Existing methods, however, treat each routing decision as a local operation conditioned solely on the current hidden state which is a formulation that overlooks the sequential, path-dependent nature of routing across depth: earlier decisions shape the representations seen by downstream routers, and the layer-usage objective couples all decisions jointly. We propose History-Aware Routing (HeRo), a dynamic routing framework that resolves this mismatch by introducing a router memory mechanism to maintain an explicit routing state across model depth. The memory is constructed via linear attention, incrementally aggregating preceding routing scores and their induced residual updates into a compact history representation. At each routed layer, the router conditions jointly on this accumulated state and the current hidden representation to select the executed branch. Instantiated for token-wise FFN routing, HeRo trains only lightweight routers and adapters on a frozen backbone, requiring no modification to pretrained parameters. Across Llama 3.1-8B, Llama 2-7B, and Llama 2-13B, HeRo consistently achieves the highest aggregate performance retention among ten baselines. On Llama 3.1-8B, it bypasses 26.87% of model parameters while achieving 100.24% of dense model performance across seven benchmarks, and retains 97.01% while bypassing 38.82% of model parameters under a tighter computation budget. Ablation studies confirm that removing routing history consistently degrades performance, most notably on multistep reasoning and code generation, validating that explicit routing memory enables more accurate and adaptive dynamic routing than solely conditioning on hidden state.

## Introduction

Large Language Models (LLMs) generate text autoregressively by propagating hidden state through the full Transformer decoder stack and predicting the next token (Vaswani et al. 2017). This static-depth regime is computationally ineficient because the full decoder stack is applied uniformly to every next-token prediction, even though diferent predictions require diferent amounts of computation (Men et al. 2025; Kim et al. 2024). Empirically, predictions that are highly predictable from the available context can often be generated with fewer layers, whereas those that depend on multistep reasoning or information distributed across long contexts require more layers of computation (Schuster et al. 2022; Raposo et al. 2024).

![](images/ec23264428ece41efd58d202288d728d4c413a183c0bae3ac651d8c8a222de4f.jpg)  
Figure 1: Dynamic feed forward network (FFN) routing proceeds as a sequence of decisions along model depth. HeRo augments the current token representation with a recurrent state constructed from preceding routing scores and changes in the residual stream, and uses this state to select the full FFN or adapter branch.

Prior work seeks to reduce computation across model depth through three broad strategies: static pruning, early exit, and dynamic routing. Static pruning removes redundant layers or parameters before inference and produces a fixed smaller model while still applying the same computation to every input (Men et al. 2025; Kim et al. 2024; Ma, Fang, and Wang 2023; Ashkboos et al. 2024). Early exit methods select an intermediate exit layer according to confidence estimated from the hidden state at that layer, allowing the executed depth to vary across predictions while restricting computation to a contiguous prefix of the decoder (Schuster et al. 2022; Bae et al. 2023; Elhoushi et al. 2024). Dynamic routing, including dynamic layer skipping and Mixture of Depths, uses learned routers or gates to decide which modules are executed and can bypass selected intermediate layers or modules while continuing through subsequent layers, thereby reducing computation without discarding all later transformations (Raposo et al. 2024; He et al. 2025; Luo, Wang, and Yan 2025a; Heakl et al. 2026; Zhao et al. 2025; Laitenberger et al. 2026).

As illustrated in the left part of Figure 1, existing dynamic routing methods typically formulate the routing decision at layer l as a local decision conditioned solely on the current representation $h _ { t } ^ { ( l ) }$ of token t. Yet the training objective regularizes routing scores across depth, jointly optimizing each layer’s decision. For each token, routing decisions at preceding layers determine which residual updates are applied and thereby shape the hidden states presented to subsequent routers. Moreover, the layer-usage loss jointly regularizes the routing scores across all routed layers, so the decision at layer l depends on both the preceding routing history and the remaining routing horizon. Dynamic routing therefore constitutes a sequential decision process along the model depth, in which each routing decision is conditioned on the preceding routing history, which comprises both the routing scores assigned at preceding layers and the residual updates induced by those decisions.

Although $h _ { t } ^ { ( l ) }$ reflects the cumulative efects of preceding routing decisions, it does not explicitly record the preceding routing history. A router conditioned only on $\bar { h } _ { t } ^ { ( \bar { l } ) }$ must therefore infer that history from its cumulative efect on the current hidden state. Residual-stream analyses show that latent belief-state structure can be encoded in hidden representations (Shai et al. 2024), but this does not establish that a lightweight routing head can recover the explicit sequence of earlier gate choices. We therefore treat $h _ { t } ^ { ( l ) }$ as an incomplete observed state for routing: it omits the directly observable gate scores and residual transitions available along the executed path. Studies of Mixture-of-Experts routing further show that propagating a routing state across layers provides information for expert selection that is complementary to the current hidden state alone (Qiu et al. 2025). These observations motivate maintaining an explicit routing state across depth that summarizes the preceding routing history and the residual updates induced by routing decisions.

We propose History-Aware Routing (HeRo), a dynamic routing framework that incorporates a router memory mechanism based on linear attention over model depth to maintain an explicit state of preceding routing decisions and their effects on the residual stream. For each token, this memory incrementally aggregates layerwise routing states produced by preceding routed layers. At routed layer l, the router predicts a routing score from the current hidden representation $h _ { t } ^ { ( l ) }$ together with the memory state, and compares it with a threshold to make the routing decision. The routing score and the induced residual update from the selected branch are then encoded into a new layerwise routing state and written back into the memory. We instantiate HeRo for FFN routing, selecting between the pretrained FFN and a lightweight adapter at each routed layer while keeping all attention sublayers dense.

Our contributions are as follows:

• We characterize dynamic routing in Transformers with decoder-only architectures as a path dependent sequential decision process across model depth. Each routing decision determines which residual update is applied and therefore changes the representation presented to downstream routers, while the routing objective couples decisions across routed layers. This characterization exposes a mismatch in existing routers: routing decisions are interdependent across depth, yet each router is conditioned only on the current hidden representation, which does not explicitly preserve the routing history that shaped it.

• We propose History-Aware Routing (HeRo), a dynamic routing framework that addresses this mismatch by introducing a router memory mechanism that maintains an explicit state of preceding routing decisions and their efects on the residual stream. HeRo uses linear attention over model depth to incrementally aggregate layerwise routing states formed from prior routing scores and the residual updates produced by the selected branches. This memory is updated at each routed layer and provides a compact representation of routing history. Each router conditions its decision jointly on the current hidden representation and the accumulated routing state. We instantiate HeRo for FFN routing while retaining dense computation in all attention sublayers.

• Across three model backbones, HeRo achieves the highest performance retention among the evaluated parameter skipping methods. On Llama 3.1-8B, it achieves 100.24% of dense model performance while bypassing 26.87% of parameters and retains 97.01% while bypassing 38.82% under the tighter computation budget. Ablations further show that routing history improves performance across most benchmarks, with particularly notable gains on multistep reasoning tasks. These results show that the router memory mechanism enables more accurate and selective routing by incorporating information accumulated across preceding layers into each routing decision.

## Related Work

Adaptive Depth for Eficient Language Modeling. Transformer depth can be reduced before inference through static compression or during inference through input dependent computation. Static pruning identifies redundant layers or structural components and uses the same retained structure for every input (Men et al. 2025; Kim et al. 2024; Ma, Fang, and Wang 2023; Ashkboos et al. 2024). Adaptive methods retain the full stack and determine how much of it to execute for each input or token. Early exit terminates computation when an intermediate prediction satisfies a confidence criterion. Developed initially for encoder models, this strategy was later adapted to autoregressive generation by CALM and FREE (Xin et al. 2020; Liu et al. 2020; Zhou et al. 2020; Schuster et al. 2022; Bae et al. 2023). LayerSkip combines layer dropout with an early exit objective to support speculative decoding with the same model (Elhoushi et al. 2024). Layer skipping uses input-dependent routing to bypass intermediate blocks while continuing computation in later blocks (Liu, Meng, and Zhou 2024; Raposo et al. 2024; Luo, Wang, and Yan 2025a). This progression shifts the focus from how much depth to remove globally to which computations each input should receive.

![](images/8964f4d70a4e1fc4706b89e8d973c5f1ba5370e01730ad4a8de7fa3884b63b23.jpg)  
Figure 2: Overview of HeRo across two routed blocks. At each block, the router encodes the current post attention representation, remaining routed depth, preceding routing scores, and completed residual stream transitions. It reads the router memory, combines the resulting history logit with a local logit, and thresholds the gate to select the pretrained FFN or adapter.

Dynamic Layer and Module Routing. Layer routing spans token, prompt, and sequence level decisions. Mixture of Depths assigns tokens to blocks under capacity constraints, while FiRST, FlexiDepth, and Dr.LLM route at prompt, token, and sequence scope, respectively (Raposo et al. 2024; Jain et al. 2025; Luo, Wang, and Yan 2025a; Heakl et al. 2026). Module routing targets attention and MLP computation. FFN SkipLLM and DifSkip bypass MLPs while retaining attention, whereas SkipGPT learns separate policies for attention and MLP components (Jaiswal et al. 2024; Luo, Wang, and Yan 2025b; Zhao et al. 2025). Router Tuning trains only routers for attention and Mixture of Experts modules, while residual gating converts learned module scores into hard skips (He et al. 2025; Laitenberger et al. 2026). HeRo routes between the pretrained MLP and a lightweight adapter while keeping attention active.

Routing with Cross Layer State. Glavas et al. (2024) find that decoder hidden states do not consistently outperform learned static vectors for token level layer selection. RMoE propagates a recurrent state across consecutive Mixture of Experts routers to model dependencies between expert assignments (Qiu et al. 2025). LIMe, MRLA, and Vertical Attention aggregate representations across depth within the backbone (Gerasimov et al. 2025; Fang et al. 2023; Kojima et al. 2025). HeRo uses prior gate scores and residual transitions to construct a cross layer routing state.

## Method

Following the sequential decision formulation introduced in the Introduction, HeRo processes each routed block through four operations, as illustrated in Figure 2. At the start of a forward pass, the depth memory is initialized to zero. Upon reaching routed block $l _ { j } ,$ , the router first encodes the postattention representation ${ { \bar { h } } _ { l _ { j } } }$ alongside scalar path features (capturing routing progress, the latest completed residual transition, and prior history scores) into a unified router state $s _ { l _ { j } }$ . Using a query derived from this state, it then reads the accumulated memory to retrieve a history context $c _ { l _ { j } }$ summarizing states from preceding routed blocks. A history head consumes the state and context, while a local head consumes $\bar { h } _ { l _ { j } }$ , producing logits that are summed and thresholded to select the frozen FFN or lightweight adapter. Finally, the router writes the state back into memory via the forgetting recurrence; the current decision and its residual efect enter the path features at the next routed block. The following subsections formalize these operations.

## Router Computation

We consider a Transformer decoder with L blocks and apply FFN routing at $L _ { R }$ selected blocks, indexed in depth order by $1 \leq l _ { 1 } < \cdots < l _ { L _ { R } } \leq L$ . All other blocks retain their pretrained computation. Attention remains dense at every routed block; the representation after attention and the normalized FFN input are

$$
\begin{array} { r l } & { { { { \bar { h } } } _ { l _ { j } } } = { { h } _ { l _ { j } - 1 } } + \mathrm { A t t n } _ { l _ { j } } \left( \mathrm { L N } _ { l _ { j } } ^ { \mathrm { a t t n } } ( { { h } _ { l _ { j } - 1 } } ) \right) , } \\ & { { { u } _ { l _ { j } } } = \mathrm { L N } _ { l _ { j } } ^ { \mathrm { f f n } } ( { { \bar { h } } _ { l _ { j } } } ) } \end{array}\tag{1}
$$

At each routed block, the router selects either the frozen pretrained feedforward network (FFN) or a trainable bottleneck adapter. The adapter uses down and up projections (Houlsby et al. 2019)

$$
\mathrm { A d a p t e r } _ { l _ { j } } ( u ) = W _ { 2 , l _ { j } } \mathrm { S i L U } ( W _ { 1 , l _ { j } } u )\tag{2}
$$

where $W _ { 1 , l _ { j } } \in \mathbb { R } ^ { d _ { r } \times d _ { \mathrm { m o d e l } } }$ and $W _ { 2 , l _ { j } } \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d _ { r } }$ are the down and up projections, respectively, with $d _ { r } < d _ { \mathrm { m o d e l } }$

Given the execution gate g defined below, the routed block $g _ { l _ { j } }$ output is

$$
h _ { l _ { j } } = \left\{ \bar { h } _ { l _ { j } } + g _ { l _ { j } } \mathrm { F F N } _ { l _ { j } } ( u _ { l _ { j } } ) , \quad \quad g _ { l _ { j } } \geq \tau _ { \mathrm { e x e c } } \right.\tag{3}
$$

Here, $g _ { l _ { i } } \in ( 0 , 1 )$ and $\tau _ { \mathrm { e x e c } }$ is the common execution threshold. Training and inference use the same thresholded single branch computation. Although the threshold operation is not diferentiated, scaling the selected branch by $g _ { l _ { j } }$ or $1 - g _ { l _ { \cdot } }$ carries gradients to the router.

## HeRo Router

Router state. At routed block $l _ { j } .$ , HeRo combines an encoding of the current representation with a path vector comprising six scalar features

$$
{ x } _ { l _ { j } } ^ { \mathrm { p a t h } } = \left[ r _ { l _ { j } } , { q } _ { l _ { j } } ^ { \mathrm { r e m } } , { d } _ { l _ { j } } , { \gamma } _ { l _ { j } } , { p } _ { l _ { j - 1 } } ^ { \mathrm { h i s t } } , m _ { l _ { j - 1 } } ^ { \mathrm { h i s t } } \right] ^ { \top } \in \mathbb { R } ^ { 6 }\tag{4}
$$

The first two entries locate the current decision along routed depth, the next two describe the latest completed residual stream transition, and the final two retain routing scores from preceding blocks. The current representation is encoded as

$$
z _ { l _ { j } } ^ { h } = W _ { h u } \mathrm { S i L U } \big ( W _ { h d } \mathrm { R M S N o r m } ( \bar { h } _ { l _ { j } } ) \big ) \in \mathbb { R } ^ { d _ { h } }\tag{5}
$$

The two depth features specify the normalized routing progress and the fraction of routed blocks remaining after the current decision

$$
r _ { l _ { j } } = \frac { j } { L _ { R } } , \qquad q _ { l _ { j } } ^ { \mathrm { r e m } } = \frac { L _ { R } - j } { L _ { R } }\tag{6}
$$

For $j \geq 2$ , the latest completed residual stream transition between consecutive routed decision points and its magnitude are

$$
\delta _ { l _ { j } } = \bar { h } _ { l _ { j } } - \bar { h } _ { l _ { j - 1 } } , \quad \delta _ { l _ { 1 } } = 0 \qquad d _ { l _ { j } } = \lVert \delta _ { l _ { j } } \rVert _ { 2 }\tag{7}
$$

and its cosine similarity with the preceding transition is

$$
\gamma _ { l _ { j } } = \left\{ \begin{array} { l l } { 0 , } & { j \leq 2 } \\ { \frac { \langle \delta _ { l _ { j } } , \delta _ { l _ { j - 1 } } \rangle } { \| \delta _ { l _ { j } } \| _ { 2 } \| \delta _ { l _ { j - 1 } } \| _ { 2 } + \varepsilon } , } & { j > 2 } \end{array} \right.\tag{8}
$$

The previous history score $p _ { l _ { j - 1 } } ^ { \mathrm { h i s t } }$ and cumulative history score $m _ { l _ { i - 1 } } ^ { \mathrm { h i s t } }$ are defined by the gate recurrence below, with $p _ { l _ { 0 } } ^ { \mathrm { h i s t } } =$ $m _ { l _ { 0 } } ^ { \mathrm { { \bar { n } i s t } } } = 1$ . The path encoder embeds the six scalar features, and its output is concatenated with the current representation encoding to form the router state

$$
\begin{array} { r l } & { z _ { l _ { j } } ^ { a } = \mathrm { F F N } _ { \mathrm { p a t h } } ( x _ { l _ { j } } ^ { \mathrm { p a t h } } ) } \\ & { s _ { l _ { j } } = [ z _ { l _ { j } } ^ { h } ; z _ { l _ { j } } ^ { a } ] \in \mathbb { R } ^ { d _ { s } } } \end{array}\tag{9}
$$

Router memory mechanism. We implement the router memory mechanism with kernelized linear attention (Katharopoulos et al. 2020) over model depth. At each routed block, it reads accumulated states before writing $s _ { l _ { j } }$ , so the context $c _ { l _ { j } }$ contains information only from preceding routed blocks. The query, key, and value projections are

$$
q _ { l _ { j } } = W _ { q } s _ { l _ { j } } , \qquad k _ { l _ { j } } = W _ { k } s _ { l _ { j } } , \qquad v _ { l _ { j } } = W _ { v } s _ { l _ { j } }\tag{10}
$$

where $q _ { l _ { i } } , k _ { l _ { i } } , v _ { l _ { i } } \in \mathbb { R } ^ { d _ { m } }$ . Using the positive feature map $\phi ( x ) = \bar { \mathrm { E L U } ( x ) ^ { \prime } + 1 }$ , let $S _ { l _ { j } . }$ and $\zeta _ { l _ { j } . }$ denote the memory matrix and normalizer accumulated before block $l _ { j }$ . The memory context is

$$
c _ { l _ { j } } = \frac { \phi ( q _ { l _ { j } } ) ^ { \top } S _ { l _ { j - 1 } } } { \phi ( q _ { l _ { j } } ) ^ { \top } \zeta _ { l _ { j - 1 } } + \varepsilon }\tag{11}
$$

The current router state is written with the block specific forgetting factor $\rho _ { l _ { j } } = \sigma ( \eta _ { l _ { j } } )$

$$
\begin{array} { r l } & { S _ { l _ { j } } = \rho _ { l _ { j } } S _ { l _ { j - 1 } } + \phi ( k _ { l _ { j } } ) v _ { l _ { j } } ^ { \top } } \\ & { \zeta _ { l _ { j } } = \rho _ { l _ { j } } \zeta _ { l _ { j - 1 } } + \phi ( k _ { l _ { j } } ) } \end{array}\tag{12}
$$

The read-before-write order gives the memory the following property: the context available to the current gate contains only states produced by earlier routed blocks. For each token, $S _ { l _ { 0 } } = \mathbf { 0 } _ { d _ { m } \times d _ { m } }$ and $\zeta _ { l _ { 0 } } = \mathbf { 0 } _ { d _ { m } }$ are initialized at the start of a forward pass and updated only across the routed blocks. All encoder, memory, and router head parameters are shared across routed blocks, whereas the adapters and forgetting factors are block specific.

Gate computation. The history head compares the current router state with the retrieved context. Because no preceding context is available at the first routed block, we set $\nu _ { l _ { 1 } } = 0 .$ For $j > 1$ , the cosine dissimilarity between the current state and retrieved context is

$$
\nu _ { l _ { j } } = 1 - \cos ( W _ { n } s _ { l _ { j } } , W _ { c } c _ { l _ { j } } )\tag{13}
$$

where $W _ { n }$ and $W _ { c }$ project the state and context into a common comparison space. The state, context, and dissimilarity are then mapped to the history logit $a _ { l _ { j } }$ and the recurrent history score

$$
a _ { l _ { j } } = f _ { \theta } ( [ s _ { l _ { j } } ; c _ { l _ { j } } ; \nu _ { l _ { j } } ] ) , \qquad p _ { l _ { j } } ^ { \mathrm { h i s t } } = \sigma ( a _ { l _ { j } } / T _ { \mathrm { h i s t } } )\tag{14}
$$

The cumulative product retains these scores for subsequent routed blocks

$$
m _ { l _ { j } } ^ { \mathrm { { h i s t } } } = m _ { l _ { j - 1 } } ^ { \mathrm { { h i s t } } } p _ { l _ { j } } ^ { \mathrm { { h i s t } } }\tag{15}
$$

The local head operates directly on the representation after attention. Combining its logit with the history logit gives the execution gate

$$
b _ { l _ { j } } = f _ { \psi } ( \bar { h } _ { l _ { j } } ) , \qquad g _ { l _ { j } } = \sigma ( a _ { l _ { j } } + b _ { l _ { j } } )\tag{16}
$$

## Training Objective

The backbone is frozen, while the adapters, state encoders, depth memory, and router heads are optimized using the language modeling loss and a gate utilization penalty

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { L M } } + \alpha _ { t } \mathcal { L } _ { \mathrm { s k i p } }\tag{17}
$$

For a batch of size B and sequence length T, with $T - 1$ next token prediction positions per sequence, the utilization penalty is

$$
\mathcal { L } _ { \mathrm { s k i p } } = \frac { 1 } { B ( T - 1 ) } \sum _ { b = 1 } ^ { B } \sum _ { t = 1 } ^ { T - 1 } \left( \sum _ { j = 1 } ^ { L _ { R } } g _ { b , l _ { j } , t } \right) ^ { 2 }\tag{18}
$$

The inner sum couples gate values across routed blocks, and the square penalizes large total FFN use for each token. The coeficient $\alpha _ { t }$ sets its weight relative to the language modeling loss.

<table><tr><td>Method</td><td>Param skip</td><td>Retain</td><td>OBQA acc_norm</td><td>PIQA acc</td><td>BoolQ acc</td><td>ARC-E acc</td><td>ARC-C acc</td><td>WinoG acc</td><td>HellaS acc_norm</td></tr><tr><td>Llama-3.1-8B / Dense</td><td>0%</td><td>100.00%</td><td>45.20</td><td>79.82</td><td>83.79</td><td>84.74</td><td>67.56</td><td>75.37</td><td>78.26</td></tr><tr><td>Llama-3.1-8B / LoRA</td><td>0%</td><td>99.87%</td><td>43.80</td><td>77.49</td><td>79.76</td><td>84.55</td><td>77.48</td><td>74.69</td><td>75.42</td></tr><tr><td>ShortGPT</td><td>25.0%</td><td>54.87%</td><td>28.00</td><td>58.76</td><td>37.77</td><td>38.05</td><td>31.40</td><td>54.14</td><td>31.50</td></tr><tr><td>Shortened-Taylor</td><td>25.0%</td><td>54.93%</td><td>28.20</td><td>58.87</td><td>37.77</td><td>38.05</td><td>31.31</td><td>54.06</td><td>31.53</td></tr><tr><td>SliceGPT</td><td>24.6%</td><td>55.82%</td><td>30.40</td><td>57.83</td><td>37.83</td><td>38.64</td><td>25.85</td><td>55.17</td><td>38.19</td></tr><tr><td>D-LLM</td><td>25.0%</td><td>57.44%</td><td>30.20</td><td>57.40</td><td>50.36</td><td>37.12</td><td>28.16</td><td>52.49</td><td>37.64</td></tr><tr><td>SkipGPT-Joint</td><td>25.3%</td><td>60.91%</td><td>31.50</td><td>60.13</td><td>50.74</td><td>37.21</td><td>28.66</td><td>52.34</td><td>50.87</td></tr><tr><td>MoD-D</td><td>25.0%</td><td>61.52%</td><td>31.60</td><td>64.25</td><td>50.28</td><td>37.67</td><td>28.24</td><td>52.41</td><td>50.44</td></tr><tr><td>Shortened-PPL</td><td>25.0%</td><td>67.72%</td><td>33.60</td><td>71.87</td><td>42.08</td><td>57.07</td><td>31.74</td><td>53.51</td><td>57.98</td></tr><tr><td>LLM-Pruner</td><td>24.5%</td><td>70.38%</td><td>37.20</td><td>72.03</td><td>56.79</td><td>51.22</td><td>31.31</td><td>56.99</td><td>54.75</td></tr><tr><td>LaCo</td><td>24.5%</td><td>72.38%</td><td>31.20</td><td>66.16</td><td>71.13</td><td>48.65</td><td>37.03</td><td>65.11</td><td>55.77</td></tr><tr><td>SkipGPT-RT</td><td>25.5%</td><td>94.14%</td><td>44.20</td><td>78.07</td><td>74.06</td><td>82.44</td><td>53.41</td><td>75.69</td><td>76.87</td></tr><tr><td>HeRo, (αt = 10−3)</td><td>26.87±0.80%</td><td>100.24±0.59 44.93±0.42</td><td></td><td>79.38±0.54</td><td>82.63±0.95</td><td>84.41±1.64</td><td>70.56±0.34</td><td>76.69±0.59</td><td>76.98±0.10</td></tr><tr><td>Llama-2-7B / Dense</td><td>0%</td><td>100.00%</td><td>44.20</td><td>78.07</td><td>71.62</td><td>81.36</td><td>52.47</td><td>74.19</td><td>78.93</td></tr><tr><td>Llama-2-7B / LoRA</td><td>0%</td><td>100.09%</td><td>44.00</td><td>76.62</td><td>77.61</td><td>80.27</td><td>52.12</td><td>72.94</td><td>77.55</td></tr><tr><td>Shortened-PPL</td><td>25.00%</td><td>73.76%</td><td>33.60</td><td>70.40</td><td>61.07</td><td>55.05</td><td>29.44</td><td>52.88</td><td>55.12</td></tr><tr><td>MoD-D</td><td>25.00%</td><td>79.17%</td><td>28.00</td><td>69.86</td><td>62.29</td><td>72.81</td><td>43.34</td><td>50.28</td><td>58.80</td></tr><tr><td>LLM-Pruner</td><td>25.30%</td><td>79.61%</td><td>39.00</td><td>73.45</td><td>54.71</td><td>58.63</td><td>37.03</td><td>58.56</td><td>60.77</td></tr><tr><td>LaCo</td><td>25.00%</td><td>83.74%</td><td>36.60</td><td>65.72</td><td>74.37</td><td>58.92</td><td>38.14</td><td>67.32</td><td>62.76</td></tr><tr><td>SkipGPT-RT</td><td>25.50%</td><td>90.33%</td><td>39.60</td><td>72.20</td><td>68.81</td><td>76.52</td><td>44.37</td><td>63.54</td><td>70.96</td></tr><tr><td>HeRo,  $( \alpha _ { t } = 1 0 ^ { - 3 } )$ </td><td>27.87±0.82%</td><td>94.38±0.54% 41.67±0.36</td><td></td><td>76.51±0.49</td><td>67.59±0.94</td><td>67.79±1.33</td><td>53.92±0.40</td><td>73.23±0.61</td><td>70.60±0.11</td></tr><tr><td>Llama-2-13B / Dense</td><td>0%</td><td>100.00%</td><td>45.20</td><td>79.11</td><td>80.52</td><td>84.68</td><td>59.47</td><td>76.16</td><td>82.23</td></tr><tr><td>Llama-2-13B / LoRA</td><td>0%</td><td>100.05%</td><td>45.20</td><td>79.11</td><td>80.06</td><td>84.97</td><td>59.72</td><td>76.24</td><td>82.26</td></tr><tr><td>LaCo</td><td>25.00%</td><td>75.05%</td><td>38.80</td><td>72.74</td><td>44.46</td><td>62.84</td><td>35.07</td><td>62.75</td><td>63.11</td></tr><tr><td>MoD-D</td><td>25.00%</td><td>81.18%</td><td>37.20</td><td>71.49</td><td>54.13</td><td>78.24</td><td>50.34</td><td>67.25</td><td>51.83</td></tr><tr><td>Shortened-PPL</td><td>25.00%</td><td>82.95%</td><td>39.40</td><td>73.12</td><td>62.57</td><td>69.19</td><td>41.13</td><td>67.17</td><td>69.31</td></tr><tr><td>LLM-Pruner</td><td>24.90%</td><td>86.92%</td><td>44.00</td><td>76.99</td><td>59.88</td><td>72.60</td><td>45.82</td><td>65.82</td><td>74.15</td></tr><tr><td>SkipGPT-RT</td><td>25.40%</td><td>92.43%</td><td>46.00</td><td>76.88</td><td>74.37</td><td>77.69</td><td>47.08</td><td>71.90</td><td>74.33</td></tr><tr><td>HeRo,  $( \alpha _ { t } = 1 0 ^ { - 3 } )$ </td><td>28.38±0.76%</td><td>94.49±0.59% 45.67±0.40</td><td></td><td>77.60±0.55</td><td>73.83±1.11</td><td>75.25±1.34</td><td>52.24±0.38</td><td>76.23±0.49</td><td>77.32±0.10</td></tr><tr><td>Llama-3.1-8B / Dense</td><td>0%</td><td>100.00%</td><td>45.20</td><td>79.82</td><td>83.79</td><td>84.74</td><td>67.56</td><td>75.37</td><td>78.26</td></tr><tr><td>LaCo</td><td>40.70%</td><td>54.28%</td><td>27.60</td><td>56.31</td><td>55.11</td><td>30.22</td><td>25.85</td><td>51.54</td><td>31.53</td></tr><tr><td>LLM-Pruner</td><td>39.90%</td><td>56.83%</td><td>29.20</td><td>64.09</td><td>50.52</td><td>36.87</td><td>23.46</td><td>51.30</td><td>36.23</td></tr><tr><td>Shortened-PPL</td><td>40.60%</td><td>59.11%</td><td>27.00</td><td>61.48</td><td>57.13</td><td>39.69</td><td>26.45</td><td>52.41</td><td>41.74</td></tr><tr><td>MoD-D</td><td>40.00%</td><td>63.14%</td><td>33.00</td><td>65.56</td><td>50.28</td><td>38.09</td><td>30.20</td><td>51.38</td><td>54.01</td></tr><tr><td>SkipGPT-RT</td><td>40.20%</td><td>81.15%</td><td>38.00</td><td>73.34</td><td>60.37</td><td>77.53</td><td>45.65</td><td>59.35</td><td>64.36</td></tr><tr><td>HeRo,  $( \alpha _ { t } = 1 0 ^ { - 3 } )$ </td><td> $3 8 . 8 2 { \scriptstyle \pm 0 . 8 4 \% }$ </td><td>97.01±0.70% 42.47±0.44</td><td></td><td>79.45±0.54</td><td>82.06±0.98</td><td>76.56±1.38</td><td>69.30±0.30</td><td>74.10±0.70 75.60±0.09</td><td></td></tr></table>

Table 1: Comparison on seven benchmarks at target parameter skipping budgets of $2 5 \%$ and 40%. Results are grouped by backbone and target budget. HeRo learns an execution path for each token over the candidate MLP modules associated with each target budget. The first group includes all compared methods. Each remaining group reports Dense and the six methods with the highest Retain in the first group. Retain is the mean task score relative to the corresponding Dense score, expressed as a percentage. Bold and underline indicate the highest and second highest scores among methods that skip parameters within each group. HeRo results are means with standard deviations over random seeds 42, 43, and 44. Dense, LoRA r=128 denotes the dense backbone fine-tuned with the same adapter and Tulu-3 SFT training without layer skipping, serving as a LoRA reference.

## Experiments

We compare HeRo with static pruning and dynamic routing baselines under matched target parameter skipping budgets. We then vary the skipping coeficient $\alpha _ { t }$ to characterize its efect on task performance and parameter skipping, remove individual path state components to assess the roles of path features, memory read, and history conditioning, and examine sensitivity to the memory dimension. Together, these experiments evaluate whether explicit path conditioning improves task performance at comparable realized parameter skipping rates.

## Experimental Setup

Models and routing configuration. Meta-Llama-3.1-8B-Instruct (Grattafiori et al. 2024) is the primary backbone for the controlled analyses. Table 1 also reports results for Llama-2-7B and Llama-2-13B. Unless stated otherwise, the configuration below refers to Meta-Llama-3.1-8B-Instruct. For the 25% target-budget groups in Table 1, the candidate MLP modules are the latter half of each backbone, while earlier layers and all attention modules remain dense. For the Llama-3.1-8B 40% target-budget group, the candidate set is expanded to include layers 9 through 32. This placement follows prior evidence that earlier layers are more sensitive to removal (Men et al. 2025); the candidate set defines routing eligibility, while HeRo learns the execution path for each token. At each routed layer, HeRo selects either the pretrained MLP or a bottleneck adapter with intermediate width $d _ { r } =$ 896 from the representation after attention. The router state has 64 dimensions, comprising 48 dimensions for the hidden representation and 16 dimensions for auxiliary path features. The history and local routing heads each use a hidden width of 256. The depth memory has one head with $q , k , v \in \mathbb { R } ^ { 6 4 } ;$ for each token, it maintains a $6 4 \times 6 4$ memory matrix and a normalizer of dimension 64. Each routed layer has a learned forgetting parameter $\eta _ { l _ { j } }$ , initialized such that $\rho _ { l _ { j } } = \sigma ( \eta _ { l _ { j } } )$ ≈ 0.98. The principal configuration uses $\alpha _ { t } = 1 0 ^ { - 3 }$ and a target parameter skipping budget of 25%.

<table><tr><td>Method</td><td>Param skip</td><td>Retain</td><td>OBQA acc_norm</td><td>PIQA acc</td><td>BoolQ acc</td><td> $\mathbf { \Pi } _ { \mathrm { a c c } } ^ { \mathrm { A R C - E } }$ </td><td> $\operatorname { A R C - C } _ { \operatorname { a c c } }$ </td><td>WinoG acc</td><td>HellaS acc_norm</td></tr><tr><td> $\alpha _ { t } = 1 \mathrm { e } { - 4 }$ </td><td>13.19%</td><td>101.66%</td><td>46.20</td><td>80.14</td><td>82.75</td><td>86.49</td><td>73.58</td><td>77.03</td><td>77.36</td></tr><tr><td> $\alpha _ { t } = 3 \mathrm { e } { - 4 }$ </td><td>18.14%</td><td>101.73%</td><td>46.80</td><td>80.36</td><td>82.66</td><td>85.44</td><td>73.24</td><td>77.51</td><td>77.44</td></tr><tr><td> $\alpha _ { t } = 5 \mathrm { e } { - 4 }$ </td><td>23.40%</td><td>101.28%</td><td>47.00</td><td>79.76</td><td>81.38</td><td>85.09</td><td>73.24</td><td>76.95</td><td>77.32</td></tr><tr><td> $\alpha _ { t } = 1 \mathrm { e } { - 3 }$ </td><td>26.87%</td><td>100.00%</td><td>44.93</td><td>79.38</td><td>82.63</td><td>84.41</td><td>70.56</td><td>76.69</td><td>76.98</td></tr></table>

Table 2: Efect of $\alpha _ { t }$ on parameter skipping and task accuracy for Meta-Llama-3.1-8B-Instruct. Retain is the mean of the seven task score ratios relative to the $\alpha _ { t } = 1 0 ^ { - 3 }$ configuration within this sweep, expressed as a percentage. Results are means over random seeds 42, 43, and 44. Bold and underline indicate the highest and second highest values.

Training configuration. The backbone remains frozen. We update only the router state encoder, depth memory, history and local routing heads, and adapters. Training uses all 939,344 examples in the Tulu-3 SFT mixture (Lambert et al. 2024) for one epoch, corresponding to approximately 7,339 optimizer updates at a global batch size of 128. We apply the chat template associated with each backbone, truncate sequences to 2048 tokens, and compute the language modeling loss over all tokens. Optimization uses AdamW with a learning rate of $1 0 ^ { - 4 }$ , weight decay $0 . 0 1 , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 9 9$ $\epsilon \stackrel { - } { = } 1 0 ^ { - 8 }$ , linear warmup over the first 3% of updates, and gradient clipping at 1.0. We use gradient checkpointing and bfloat16 mixed precision across four NVIDIA RTX 4090D GPUs with 48 GB of memory each. Each training run takes approximately 12 hours. We set the execution threshold to $\tau _ { \mathrm { e x e c } } = 0 . 5$ and the history temperature to $T _ { \mathrm { h i s t } } = 0 . 9$ . For every setting in the main comparison, we train HeRo with random seeds 42, 43, and 44 and report the mean and standard deviation. Training and evaluation both use the thresholded branch decision in Equation 3; the gate scaling supplies gradients to the router during training.

Benchmarks and metrics. We use the lm-evaluationharness framework (Gao et al. 2024). The main comparison covers seven benchmarks: OpenBookQA (Mihaylov et al. 2018) and ARC-Easy and ARC-Challenge (Clark et al. 2018) for science question answering; PIQA (Bisk et al. 2020) for physical commonsense; BoolQ (Clark et al. 2019) for question answering over passages; WinoGrande (Sakaguchi et al. 2020) for commonsense coreference; and HellaSwag (Zellers et al. 2019) for plausible continuation selection. We report normalized accuracy for OpenBookQA and HellaSwag and accuracy for the other five tasks. The ablation suite additionally includes GSM8K (Cobbe et al. 2021), evaluated by exact match after chain of thought generation, and HumanEval (Chen et al. 2021), evaluated by pass@1 under greedy decoding.

Evaluation protocol. GSM8K uses eight chain of thought demonstrations and a maximum generation length of 1024 tokens. ARC-Easy and ARC-Challenge use 25 demonstrations, while HellaSwag and WinoGrande use five. Open-BookQA, PIQA, BoolQ, and HumanEval are evaluated without demonstrations. Chat templates are enabled for tasks that use instruction formatting, and demonstrations are rendered as multiturn conversations when required by the task definition.

Baselines and comparison budgets. The static pruning baselines are ShortGPT (Men et al. 2025), the PPL and Taylor variants of Shortened LLaMA (Kim et al. 2024), LaCo (Yang, Cao, and Zhao 2024), LLM-Pruner (Ma, Fang, and Wang 2023), and SliceGPT (Ashkboos et al. 2024). Dynamic baselines include MoD-D, a variant of Mixture of Depths introduced by SkipGPT that routes attention and MLP modules separately (Raposo et al. 2024; Zhao et al. 2025); D-LLM (Jiang et al. 2024); and the joint training and router tuning variants of SkipGPT (Zhao et al. 2025). These methods bypass diferent computational units, including complete Transformer layers, attention modules, and MLP modules. We therefore group comparisons by target parameter skipping budget and report the realized parameter skipping rate for every method. This rate is the proportion of model parameters that do not participate in computation, averaged over tokens and normalized by the total parameter count.

## Main Results

Performance at matched parameter budgets. Table 1 shows that HeRo achieves the highest aggregate retention among the evaluated parameter skipping methods across all three backbones. On Llama-3.1-8B at the 25% target budget, HeRo matches the dense model’s aggregate performance at a realized parameter skipping rate of 26.87% and leads all competing pruning and routing methods on every benchmark. HeRo also ranks first in Retain on Llama-2-7B and Llama-2-13B. The repeated advantage across model sizes shows that its performance is not specific to a single backbone.

Behavior under a tighter computation budget. The 40% target budget provides a more demanding test because more residual updates are bypassed before subsequent routing decisions are made. HeRo retains 97.01% of dense performance in this setting, compared with 81.15% for SkipGPT-RT at a similar realized parameter skipping rate, and leads on six of the seven benchmarks. The margin over the strongest dynamic baseline therefore grows substantially as the target budget becomes more restrictive. This pattern is consistent with the motivation for HeRo: as more MLP updates are bypassed, subsequent routers receive representations shaped by increasingly diverse execution paths, making an explicit summary of earlier routing decisions more informative for downstream allocation.

<table><tr><td>Method</td><td>Param skip</td><td>Retain</td><td>GSM8K exact_match</td><td>ARC-C acc</td><td>OBQA acc_norm</td><td>HumE pass@1</td><td>BoolQ acc</td><td>WinoG. acc</td><td>ARC-E acc</td><td>PIQA acc</td><td>HellaS acc_norm</td></tr><tr><td>w/o Pos State</td><td>26.35%</td><td>97.57%</td><td>80.52</td><td>68.23</td><td>42.20</td><td>61.59</td><td>79.02</td><td>76.09</td><td>83.68</td><td>79.82</td><td>77.35</td></tr><tr><td>w/o Aux State</td><td>27.02%</td><td>97.32%</td><td>79.38</td><td>68.56</td><td>42.20</td><td>60.37</td><td>79.42</td><td>75.77</td><td>84.56</td><td>79.43</td><td>77.32</td></tr><tr><td>w/o History</td><td>26.41%</td><td>97.35%</td><td>79.38</td><td>67.89</td><td>42.80</td><td>59.76</td><td>78.41</td><td>76.64</td><td>84.74</td><td>79.92</td><td>77.38</td></tr><tr><td>w/o MemRead</td><td>26.28%</td><td>98.02%</td><td>80.14</td><td>69.23</td><td>42.60</td><td>62.20</td><td>79.20</td><td>76.48</td><td>84.39</td><td>79.60</td><td>77.33</td></tr><tr><td>HeRo</td><td>26.87%</td><td>100.00%</td><td>82.11</td><td>70.56</td><td>44.93</td><td>65.24</td><td>82.63</td><td>76.69</td><td>84.41</td><td>79.38</td><td>76.98</td></tr></table>

Table 3: Component ablations of HeRo on Meta-Llama-3.1-8B-Instruct using the extended benchmark suite. Param skip is the realized parameter skipping rate, averaged over tokens and normalized by the total parameter count. Retain is the mean of the nine task score ratios relative to the full HeRo configuration, expressed as a percentage.

## Ablation Studies

Efect of the skipping coeficient. Table 2 shows that α provides direct control over the operating point of HeRo. Increasing the coeficient raises the realized parameter skipping rate monotonically from 13.19% to 26.87%, while Retain varies by only 1.73 points across the sweep. Lower coeficients yield slightly higher Retain, whereas $\alpha _ { t } = 1 0 ^ { - 3 }$ reaches the target budget with only a modest reduction in aggregate performance. We therefore use this setting in the main comparison.

Path state components. Table 3 isolates the information flow that distinguishes HeRo from a local router. The w/o History variant removes the history branch and bases each decision on the local logit alone. The w/o MemRead variant retains the history head but removes its access to accumulated memory. The remaining variants remove either the auxiliary path encoder or the routing progress and residual transition features supplied to it. All other training and routing settings are fixed.

The complete model achieves the highest Retain at a parameter skipping rate close to the maximum in the table. Every ablation lowers aggregate performance. Removing the auxiliary path encoder slightly increases Param skip, while the other ablations reduce both Retain and Param skip. The quality advantage of the complete model is therefore not explained by a more conservative routing policy. These results show that state construction and history conditioning both contribute to routing quality.

The memory read provides a second, complementary source of evidence. Retaining the history head without access to the accumulated memory still reduces both Retain and Param skip, so the history head alone does not recover the full model’s routing quality. Taken together, the ablations support the complete HeRo design as a sequence of complementary operations: path features describe the current routing context, the memory aggregates that context across depth, and the history head uses the accumulated state to coordinate the

next branch decision.
<table><tr><td>Memory dim. Param skip</td><td></td><td>Retain</td><td>Avg. acc</td></tr><tr><td>32</td><td>26.36%</td><td>100.08%</td><td>73.61</td></tr><tr><td>64</td><td>26.87%</td><td>100.00%</td><td>73.65</td></tr><tr><td>128</td><td>26.29%</td><td>100.03%</td><td>73.62</td></tr><tr><td>256</td><td>26.54%</td><td>100.12%</td><td>73.66</td></tr></table>

Table 4: Efect of memory dimension on Meta-Llama-3.1- 8B-Instruct. All other settings follow the main configuration.

Memory dimension. Table 4 shows that no memory dimension dominates all three aggregate measures. A dimension of 256 yields the highest Retain and average accuracy with the second highest parameter skipping rate, whereas a dimension of 64 yields the highest parameter skipping rate but the lowest Retain. Across the tested range, Retain and average accuracy vary by only 0.12 and 0.05 points, indicating limited sensitivity to the memory dimension.

## Conclusion

We introduced History-Aware Routing (HeRo), a dynamic routing framework that incorporates a router memory mechanism based on linear attention over model depth. The mechanism aggregates preceding routing scores and their induced residual updates into an explicit routing state, allowing each router to condition its decision on both the current hidden state and the routing information accumulated across preceding layers. Across Llama 3.1-8B, Llama 2-7B, and Llama 2-13B, HeRo consistently outperformed all 10 baseline methods while training only lightweight routers and adapters and keeping the backbone frozen, demonstrating consistent efectiveness across model families and scales. On Llama 3.1-8B, HeRo bypassed 26.87% of model parameters during inference while achieving 100.24% of dense model performance across seven benchmarks; under a more restrictive computation budget, it retained 97.01% of dense performance while bypassing 38.82% of parameters. Ablations further showed that removing the routing history head led to performance losses across six benchmarks, with particularly notable losses on HumanEval and GSM8K. These results confirm that explicitly maintaining routing information through the router memory mechanism improves the coordination of routing decisions across model depth, enabling more accurate and selective dynamic routing than routing based solely on the current hidden state.

## References

Ashkboos, S.; Croci, M. L.; Gennari do Nascimento, M.; Hoefler, T.; and Hensman, J. 2024. SliceGPT: Compress Large Language Models by Deleting Rows and Columns. In The Twelfth International Conference on Learning Representations.

Bae, S.; Ko, J.; Song, H.; and Yun, S.-Y. 2023. Fast and Robust Early-Exiting Framework for Autoregressive Language Models with Synchronized Parallel Decoding. In Bouamor, H.; Pino, J.; and Bali, K., eds., Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 5910–5924. Singapore: Association for Computational Linguistics.

Bisk, Y.; Zellers, R.; Le Bras, R.; Gao, J.; and Choi, Y. 2020. PIQA: Reasoning about Physical Commonsense in Natural Language. In Proceedings of the Thirty-Fourth AAAI Conference onArtificial Intelligence, volume 34, 7432–7439.

Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; Pinto, H. P. d. O.; Kaplan, J.; Edwards, H.; Burda, Y.; Joseph, N.; Brockman, G.; Ray, A.; Puri, R.; Krueger, G.; Petrov, M.; Khlaaf, H.; Sastry, G.; Mishkin, P.; Chan, B.; Gray, S.; Ryder, N.; Pavlov, M.; Power, A.; Kaiser, L.; Bavarian, M.; Winter, C.; Tillet, P.; Such, F. P.; Cummings, D.; Plappert, M.; Chantzis, F.; Barnes, E.; Herbert-Voss, A.; Guss, W. H.; Nichol, A.; Paino, A.; Tezak, N.; Tang, J.; Babuschkin, I.; Balaji, S.; Jain, S.; Saunders, W.; Hesse, C.; Carr, A. N.; Leike, J.; Achiam, J.; Misra, V.; Morikawa, E.; Radford, A.; Knight, M.; Brundage, M.; Murati, M.; Mayer, K.; Welinder, P.; Mc-Grew, B.; Amodei, D.; McCandlish, S.; Sutskever, I.; and Zaremba, W. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374.

Clark, C.; Lee, K.; Chang, M.-W.; Kwiatkowski, T.; Collins, M.; and Toutanova, K. 2019. BoolQ: Exploring the Surprising Dificulty of Natural Yes/No Questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2924–2936. Minneapolis, Minnesota: Association for Computational Linguistics.

Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; Hesse, C.; and Schulman, J. 2021. Training Verifiers to Solve Math Word Problems. arXiv:2110.14168.

Elhoushi, M.; Shrivastava, A.; Liskovich, D.; Hosmer, B.; Wasti, B.; Lai, L.; Mahmoud, A.; Acun, B.; Agarwal, S.; Roman, A.; Aly, A.; Chen, B.; and Wu, C.-J. 2024. LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12622– 12642. Bangkok, Thailand: Association for Computational Linguistics.

Fang, Y.; Cai, Y.; Chen, J.; Zhao, J.; Tian, G.; and Li, G. 2023. Cross-Layer Retrospective Retrieving via Layer Attention. In The Eleventh International Conference on Learning Representations.

Gao, L.; Tow, J.; Abbasi, B.; Biderman, S.; Black, S.; DiPofi, A.; Foster, C.; Golding, L.; Hsu, J.; Le Noac’h, A.; Li, H.; McDonell, K.; Muennighof, N.; Ociepa, C.; Phang, J.; Reynolds, L.; Schoelkopf, H.; Skowron, A.; Sutawika, L.; Tang, E.; Thite, A.; Wang, B.; Wang, K.; and Zou, A. 2024. The Language Model Evaluation Harness.

Gerasimov, G.; Aksenov, Y.; Balagansky, N.; Sinii, V.; and Gavrilov, D. 2025. You Do Not Fully Utilize Transformer’s Representation Capacity. arXiv:2502.09245.

Glavas, T.; Chataoui, J.; Regol, F.; Jabbour, W.; Valkanas, A.; Oreshkin, B. N.; and Coates, M. 2024. Dynamic Layer Selection in Decoder-Only Transformers. arXiv:2410.20022.

Grattafiori, A.; Dubey, A.; Jauhri, A.; et al. 2024. The Llama 3 Herd of Models. arXiv:2407.21783.

He, S.; Ge, T.; Sun, G.; Tian, B.; Wang, X.; and Yu, D. 2025. Router-Tuning: A Simple and Efective Approach for Dynamic Depth. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 1925– 1938. Association for Computational Linguistics.

Heakl, A.; Gubri, M.; Khan, S.; Yun, S.; and Oh, S. J. 2026. Dr.LLM: Dynamic Layer Routing in LLMs. In The Fourteenth International Conference on Learning Representations.

Houlsby, N.; Giurgiu, A.; Jastrzebski, S.; Morrone, B.; de Laroussilhe, Q.; Gesmundo, A.; Attariyan, M.; and Gelly, S. 2019. Parameter-Eficient Transfer Learning for NLP. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, 2790–2799. PMLR.

Jain, A.; Sharma, S.; Mukherjee, K.; and Pal, S. 2025. FiRST: Finetuning Router-Selective Transformers for Input-Adaptive Latency Reduction. In Findings ofthe Association for Computational Linguistics: EMNLP 2025, 21957–21975. Suzhou, China: Association for Computational Linguistics.

Jaiswal, A. K.; Hu, B.; Yin, L.; Ro, Y.; Chen, T.; Liu, S.; and Akella, A. 2024. FFN-SkipLLM: A Hidden Gem for Autoregressive Decoding with Adaptive Feed Forward Skipping. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 16943–16956. Miami, Florida, USA: Association for Computational Linguistics.

Jiang, Y.; Wang, H.; Xie, L.; Zhao, H.; Zhang, C.; Qian, H.; and Lui, J. C. S. 2024. D-LLM: A Token Adaptive Computing Resource Allocation Strategy for Large Language Models. In Advances in Neural Information Processing Systems, volume 37. Curran Associates, Inc.

Katharopoulos, A.; Vyas, A.; Pappas, N.; and Fleuret, F. 2020. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. In Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, 5156–5165. PMLR.

Kim, B.-K.; Kim, G.; Kim, T.-H.; Castells, T.; Choi, S.; Shin, J.; and Song, H.-K. 2024. Shortened LLaMA: A Simple

Depth Pruning for Large Language Models. In ICLR 2024 Workshop on Mathematical and Empirical Understanding of Foundation Models.

Kojima, T.; Iwasawa, Y.; Yokota, R.; Miyao, Y.; Suzuki, J.; and Matsuo, Y. 2025. Vertical Attention: Automatic Exploration of Inter-Layer Connections in Transformerbased Language Models. https://openreview.net/forum?id= A6uabXKiXc. OpenReview submission; submitted to ICLR 2026.

Laitenberger, F.; Kopiczko, D.; Snoek, C. G. M.; and Asano, Y. M. 2026. What Layers When: Learning to Skip Compute in LLMs with Residual Gates. In The Fourteenth International Conference on Learning Representations.

Lambert, N.; Morrison, J.; Pyatkin, V.; Huang, S.; Ivison, H.; Brahman, F.; Miranda, L. J. V.; Liu, A.; Dziri, N.; Lyu, S.; Gu, Y.; Malik, S.; Graf, V.; Hwang, J. D.; Yang, J.; Le Bras, R.; Tafjord, O.; Wilhelm, C.; Soldaini, L.; Smith, N. A.; Wang, Y.; Dasigi, P.; and Hajishirzi, H. 2024. Tülu 3: Pushing Frontiers in Open Language Model Post-Training. arXiv:2411.15124.

Liu, W.; Zhou, P.; Wang, Z.; Zhao, Z.; Deng, H.; and Ju, Q. 2020. FastBERT: a Self-distilling BERT with Adaptive Inference Time. In Jurafsky, D.; Chai, J.; Schluter, N.; and Tetreault, J., eds., Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, 6035–6044. Online: Association for Computational Linguistics.

Liu, Y.; Meng, F.; and Zhou, J. 2024. Accelerating Inference in Large Language Models with a Unified Layer Skipping Strategy. arXiv:2404.06954.

Luo, X.; Wang, W.; and Yan, X. 2025a. Adaptive Layerskipping in Pre-trained LLMs. In Proceedings ofthe Second Conference on Language Modeling.

Luo, X.; Wang, W.; and Yan, X. 2025b. DifSkip: Diferential Layer Skipping in Large Language Models. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Findings ofthe Associationfor Computational Linguistics: ACL 2025, 7221–7231. Vienna, Austria: Association for Computational Linguistics.

Ma, X.; Fang, G.; and Wang, X. 2023. LLM-Pruner: On the Structural Pruning of Large Language Models. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Men, X.; Xu, M.; Zhang, Q.; Yuan, Q.; Wang, B.; Lin, H.; Lu, Y.; Han, X.; and Chen, W. 2025. ShortGPT: Layers in Large Language Models are More Redundant Than You Expect. In Che, W.; Nabende, J.; Shutova, E.; and Pilehvar, M. T., eds., Findings of the Association for Computational Linguistics: ACL 2025, 20192–20204. Vienna, Austria: Association for Computational Linguistics.

Mihaylov, T.; Clark, P.; Khot, T.; and Sabharwal, A. 2018. Can a Suit of Armor Conduct Electricity? A New Dataset for Open Book Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2381–2391. Brussels, Belgium: Association for Computational Linguistics.

Qiu, Z.; Huang, Z.; Cheng, S.; Zhou, Y.; Wang, Z.; Titov, I.; and Fu, J. 2025. Layerwise Recurrent Router for Mixtureof-Experts. In The Thirteenth International Conference on Learning Representations.

Raposo, D.; Ritter, S.; Richards, B.; Lillicrap, T.; Humphreys, P. C.; and Santoro, A. 2024. Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models. arXiv:2404.02258.

Sakaguchi, K.; Bras, R. L.; Bhagavatula, C.; and Choi, Y. 2020. WinoGrande: An Adversarial Winograd Schema Challenge at Scale. In Proceedings of the Thirty-Fourth AAAI Conference onArtificial Intelligence, volume 34, 8732–8740.

Schuster, T.; Fisch, A.; Gupta, J.; Dehghani, M.; Bahri, D.; Tran, V.; Tay, Y.; and Metzler, D. 2022. Confident Adaptive Language Modeling. In Advances in Neural Information Processing Systems, volume 35.

Shai, A. S.; Marzen, S. E.; Teixeira, L.; Gietelink Oldenziel, A.; and Riechers, P. M. 2024. Transformers Represent Belief State Geometry in their Residual Stream. In Advances in Neural Information Processing Systems, volume 37, 75012– 75034.

Vaswani, A.; Shazeer, N.; Parmar, N.; Uszkoreit, J.; Jones, L.; Gomez, A. N.; Kaiser, Ł.; and Polosukhin, I. 2017. Attention is All you Need. In Advances in Neural Information Processing Systems, volume 30, 5998–6008. Curran Associates, Inc.

Xin, J.; Tang, R.; Lee, J.; Yu, Y.; and Lin, J. 2020. DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference. In Jurafsky, D.; Chai, J.; Schluter, N.; and Tetreault, J., eds., Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2246–2251. Online: Association for Computational Linguistics.

Yang, Y.; Cao, Z.; and Zhao, H. 2024. LaCo: Large Language Model Pruning via Layer Collapse. In Al-Onaizan, Y.; Bansal, M.; and Chen, Y.-N., eds., Findings of the Associationfor Computational Linguistics: EMNLP 2024, 6401– 6417. Miami, Florida, USA: Association for Computational Linguistics.

Zellers, R.; Holtzman, A.; Bisk, Y.; Farhadi, A.; and Choi, Y. 2019. HellaSwag: Can a Machine Really Finish Your Sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 4791–4800. Florence, Italy: Association for Computational Linguistics.

Zhao, A.; Ye, F.; Fan, Y.; Tong, J.; Fei, Z.; Su, H.; and Shen, X. 2025. SkipGPT: Dynamic Layer Pruning Reinvented with Token Awareness and Module Decoupling. arXiv:2506.04179.

Zhou, W.; Xu, C.; Ge, T.; McAuley, J.; Xu, K.; and Wei, F. 2020. BERT Loses Patience: Fast and Robust Inference with Early Exit. In Advances in Neural Information Processing Systems, volume 33, 18330–18341. Curran Associates, Inc.