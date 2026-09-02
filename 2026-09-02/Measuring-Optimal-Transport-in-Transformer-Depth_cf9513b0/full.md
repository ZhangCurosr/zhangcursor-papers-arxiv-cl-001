# Measuring Optimal Transport in Transformer Depth

Alexandre Quemy Hother Labs alexandre@hother.io

## Abstract

A transformer carries each token’s state from layer to layer, and the whole vocabulary carried together forms a cloud that moves with depth. We ask whether a trained network moves this cloud the way optimal transport would: at the cheapest cost, and along the map that pairs each token with its optimal destination. We measure both on Pythia-160m and Pythia-410m, with an exact assignment between consecutive layer clouds, a measured sampling floor, calibration on couplings known to be optimal, and a split of the cost into the common shift of the cloud and the token-specific moves. At the last layer, both models move their tokens where the optimal-transport map sends them, at the optimal cost for Pythia-410m and slightly above it for Pythia-160m. At the first layer they do not. In between, single layers can be judged on cost at only two of ten transitions, and blocks of several layers move the cloud at close to the optimal cost. The agreement at the last layer is much weaker at initialisation (0.64 against 0.86) and grows with training.

## 1 Introduction

A transformer with L layers turns each token vector $x ^ { ( 0 ) }$ into $x ^ { ( 1 ) } , \ldots x ^ { ( L ) }$ . At layer 0 the state is the embedding of the current token, so there are as many distinct points as the vocabulary cardinality. From layer 1 on, states depend on context, so there are as many points as token occurrences. Taken together, the states of a corpus form a cloud $\mu _ { \ell }$ at layer ℓ. The network moves the first cloud onto the second with a specific pairing, token i goes from $x _ { i } ^ { ( \ell ) } : 0 x _ { i } ^ { ( \ell + 1 ) }$

Optimal transport asks what the cheapest way to move $\mu _ { \ell }$ onto $\mu _ { \ell + 1 }$ would be if the pairing were free, paying squared distance per point:

$$
W _ { 2 } ^ { 2 } ( \mu _ { \ell } , \mu _ { \ell + 1 } ) = \operatorname* { m i n } _ { \pi } \ \mathbb { E } _ { ( x , y ) \sim \pi } \ \| x - y \| ^ { 2 } .\tag{1}
$$

For this cost, the optimal pairing is unique and is the gradient of a convex function, $y = \nabla \varphi ( x )$ [Brenier, 1991].

In this paper, we ask whether the network’s own pairing is that particular one. To do so, we study the transport efficiency of Pythia-160m and Pythin 410M, layer per layer, over a sample of token positions obtained on a subset of WikiText-103 and Piles datasets. A following question we study is whether the move of token i from layer ℓ to ℓ+1 matches the optimal-transport displacement $\nabla \varphi _ { \ell } ( x _ { i } ) - x _ { i }$

There are several practical interests in answering these questions. If a layer follows that map, its action on all tokens is summarised by a single convex potential, so the layer can be replaced by a (possibly) cheap optimal-transport surrogate for analysis [Makkuva et al., 2020, Korotin et al., 2023], and the methods that merge models [Singh and Jaggi, 2020] or align their representations [Alvarez-Melis and Jaakkola, 2018] with optimal transport can be applied with the guarantee that the network’s own map is of the same kind. In addition, the distance to the cheapest move is a label-free measure of wasted motion [Finlay et al., 2020, Kan et al., 2025], usable to compare layers and models, and the agreement with the optimal map across training checkpoints [Biderman et al., 2023] shows when and where a layer’s function forms.

We make three contributions. The first is the question itself. To our knowledge, this is the first empirical test of whether the layer-to-layer map of a trained transformer acts as optimal transport between its own consecutive layer clouds, using the coupling the network defines, both in cost and in the pairing it induces. The second is a measurement protocol for optimal transport between layer transitions: the sampling error is measured and subtracted, the readings are calibrated on couplings known to be optimal, the common shift of the cloud is separated from the token-specific moves, and a lower bound that needs no calibration is provided. The third is the result: on Pythia-160m and Pythia-410m, the last layer moves its tokens along the optimal-transport map, this agreement appears with training, and the cost of the network’s moves is optimal wherever it can be measured, with one exception at the exit of the smaller model.

The rest of the paper is organized as follows. Section 2 discusses related work. Section 3 studies the cost of the network’s moves and Section 4 their agreement with the optimal-transport map. We conclude in Section 5.

## 2 Related work

Transformers as maps between measures. The literature models tokens as interacting particles [Vuckovic et al., 2020], their clustering in the long-time limit [Geshkovski et al., 2025], and transformers as measure-to-measure maps with a Vlasov limit [Furuya et al., 2025]. We share the object, the cloud of token states carried from layer to layer, but these works derive properties of the map in theory, whereas we measure on trained networks whether the map is the cheapest one, in cost and in the pairing it induces.

Attention and optimal transport. Doubly stochastic attention is read as a Wasserstein gradient flow [Sander et al., 2022], and attention formulated as semi-relaxed entropic OT inside a layer, with GPT-2 drift profiles [Wang and Wang, 2025]. Both place optimal transport inside the attention operation, at a fixed regularisation. We place it between consecutive layer clouds, unregularised, and test the network’s coupling against the exact optimal one; neither work contains an efficiency or map test.

OT-regularised flows. OT-Transformer [Kan et al., 2025] and rectified flows [Liu et al., 2023] add a kinetic-energy penalty so that the learned flow takes optimal paths. They impose the property we measure: our question is whether a transformer trained with no such penalty sits near the optimal coupling anyway.

Depth as dynamics. Linear-drift SDEs fitted to layer trajectories [Sarfati et al., 2025] and linear cross-layer lenses [Belrose et al., 2023] describe depth by a fitted linear map. Our linear baseline $T _ { G }$ is the optimal-transport member of that family, and our control asks what the network does beyond it: at the 160m exit one third of the move is beyond any linear map, and 61% beyond the optimal linear map $T _ { G }$ , and still follows optimal transport.

## 3 Transport efficiency

When a layer moves the cloud of token states from layer l to l + 1, does it pay the smallest possible price for that change of cloud? The smallest possible price is the optimal-transport cost between the two clouds:

$$
e _ { \ell } = \frac { W _ { 2 } ^ { 2 } ( \mu _ { \ell } , \mu _ { \ell + 1 } ) } { \mathbb { E } _ { i } | x _ { i } ^ { ( \ell + 1 ) } - x _ { i } ^ { ( \ell ) } | ^ { 2 } } , \qquad W _ { 2 } ^ { 2 } ( \mu _ { \ell } , \mu _ { \ell + 1 } ) = \operatorname* { m i n } _ { \pi } \mathbb { E } _ { ( x , y ) \sim \pi } | x - y | ^ { 2 } ,
$$

the minimum being over all pairings π of the two clouds. Since the network’s pairing is one candidate, $e \ell \leq 1$ , and $e _ { \ell } = 1$ means the network is a perfect mover. Because a common shift of the whole cloud is exactly optimal (and in fact dominates most transitions), the question is asked on the token-specific part:

Write $d _ { i } = x _ { i } ^ { ( \ell + 1 ) } - x _ { i } ^ { ( \ell ) }$ for token i’s move and $\Delta m _ { \ell } = \mathbb { E } _ { i } d _ { i }$ for the mean move. Then

$$
\bar { \epsilon } _ { \ell } = \frac { W _ { 2 } ^ { 2 } ( \bar { \mu } _ { \ell } , \bar { \mu } _ { \ell + 1 } ) } { P _ { \ell } } , \qquad P _ { \ell } = \mathbb { E } _ { i } \| d _ { i } - \Delta m _ { \ell } \| ^ { 2 } , \qquad { \epsilon } _ { \ell } = \frac { A _ { \ell } + { \bar { e } _ { \ell } } P _ { \ell } } { A _ { \ell } + P _ { \ell } } ,
$$

Table 1: Transport efficiency of Pythia-160m (16-d camera, $n = 4 { , } 0 0 0 .$ , three seeds). Each efficiency is shown next to what an optimal coupling reads through the same instrument; $\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } }$ is the lower bound from all types; block rows pair layer ℓ with ℓ+k directly. n.r.: not resolvable.
<table><tr><td></td><td></td><td></td><td></td><td>raw</td><td colspan="2">token-specific</td><td></td></tr><tr><td>transition</td><td>floor / Pl</td><td>A/P</td><td>el</td><td>optimal reads</td><td>ēe [seed range]</td><td>optimal reads</td><td> $\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } }$ </td></tr><tr><td>0→1</td><td>0.08</td><td>3.8</td><td>0.98</td><td>1.00</td><td>1.02 [1.00–1.04]</td><td>1.01</td><td>0.94</td></tr><tr><td>1→2</td><td>0.51</td><td>3.4</td><td>0.90</td><td>0.99</td><td>n.r.</td><td>一</td><td>0.32</td></tr><tr><td>2→3</td><td>0.50</td><td>0.7</td><td>0.82</td><td>0.99</td><td>n.r.</td><td></td><td>0.60</td></tr><tr><td>3→4</td><td>0.09</td><td>0.1</td><td>0.99</td><td>0.93</td><td>1.07 [1.03–1.15]</td><td>1.13</td><td>0.78</td></tr><tr><td>4→5</td><td>2.73</td><td>1.2</td><td>n.r.</td><td>一</td><td>n.r.</td><td>一</td><td>0.36</td></tr><tr><td>5→6</td><td>2.92</td><td>5.3</td><td>n.r.</td><td>一</td><td>n.r.</td><td></td><td>0.22</td></tr><tr><td>6→7</td><td>2.47</td><td>1.4</td><td>n.r.</td><td>一</td><td>n.r.</td><td></td><td>0.18</td></tr><tr><td>7→8</td><td>0.73</td><td>3.7</td><td>0.90</td><td>0.94</td><td>n.r.</td><td></td><td>0.21</td></tr><tr><td>8→9</td><td>0.59</td><td>6.3</td><td>0.92</td><td>0.96</td><td>n.r.</td><td>一</td><td>0.24</td></tr><tr><td>9→10</td><td>0.58</td><td>3.1</td><td>0.86</td><td>0.94</td><td>n.r.</td><td></td><td>0.29</td></tr><tr><td>10→11</td><td>0.13</td><td>1.5</td><td>0.94</td><td>0.99</td><td>0.99 [0.97–1.02]</td><td>1.09</td><td>0.46</td></tr><tr><td>11→12</td><td>0.11</td><td>99.0</td><td>1.00</td><td>1.00</td><td>0.86 [0.85–0.87]</td><td>1.00</td><td>0.73</td></tr><tr><td>3→9, six layers as a block</td><td>0.08</td><td>1.3</td><td>0.95</td><td>1.00</td><td>0.97</td><td>0.95</td><td>0.67</td></tr><tr><td>8→11, three layers as a block</td><td>0.06</td><td>2.4</td><td>0.98</td><td>1.01</td><td>0.96</td><td>一</td><td>0.48</td></tr></table>

with $A _ { \ell } = \| \Delta m _ { \ell } \| ^ { 2 }$ the cost of the common shift.

We use the two following objects throughout this paper:

• The mean flow: one point per token type, its residual-stream state averaged over all occurrences in a corpus.

• The raw states: one point per occurrence, the state at an actual position in an actual document.

We studied Pythia-160m (12 layers) and Pythia-410m (24 layers) [Biderman et al., 2023] and used WikiText-103 and Pile as corpora. On WikiText, we collected 242,015 positions in the full 768-d space, i.e. raw per-occurrence states, leading to 25,268 token types for the corpus-mean token trajectories. On Pile, we collected 46,542 corpus-mean token trajectories only.

We then split the study into two: one in reduced dimension, and one the full space.

## 3.1 Exact efficiency in a 16-d camera

Protocol We sample 4,000 tokens at layer ℓ and an independent 4,000 at $\ell + 1$ , form $C _ { i j } =$ $\| x _ { i } - y _ { j } \| ^ { 2 }$ and solve the assignment exactly using the network simplex of POT [Flamary et al., 2021], the implementation of Bonneel et al. [2011]. This is the $W _ { 2 } ^ { 2 }$ of the two samples with no regularisation.

It is computed in a balanced 16-d PCA camera because the error of an empirical Wasserstein distance decays like $n ^ { - 1 / d }$ [Fournier and Guillin, 2015, Weed and Bach, 2019]. Note that because the two samples are independent, the numerator carries a finite-sample floor that the paired denominator does not. As a result, the raw ratio $e _ { \ell }$ can exceed 1.

To make the number honest, we subtract the assignment cost between two independent samples of the same layer. This is the self-term correction that defines the Sinkhorn divergence [Feydy et al., 2019, Chizat et al., 2020], applied here to the exact assignment and to the target layer only. Where the floor exceeds 30% of the cost in question, the transition is reported as unresolvable. It essentially means that the token-specific moves at that transition are smaller than the typical distance between neighbouring sampled tokens, so we cannot separate them from the sampling gap.

In addition, to calibrate the reading, we feed the same pipeline two couplings that are optimal by construction: 1) a shift $y = x + t$ with $t = \Delta m \ell$ and 2) a radius-matched expansion $y =$ $s ( x - m _ { \ell } ) + m _ { \ell + 1 }$ , built from independent samples of layer ℓ. Their reading, 0.85–1.00 across the resolvable transitions, is what $e _ { \ell } = 1$ looks like through our instrument. Efficiency is reported raw $( e _ { \ell } )$ and on the centered clouds $( \bar { e } _ { \ell } )$ , with $A / P$ . The whole protocol is repeated over 3 seeds.

Table 2: Transport efficiency of Pythia-410m (16-d camera, $n = 4 , 0 0 0$ , one seed), same columns as Table 1; $\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } }$ from all 11,937 types. The block row pairs layer 4 with layer 20 directly, the only resolvable block of this model. Only the transitions resolvable on at least one cost are shown.
<table><tr><td></td><td></td><td></td><td></td><td>raw</td><td colspan="2">token-specific</td><td></td></tr><tr><td>transition</td><td>floor / Pl</td><td> $A / P$ </td><td> $e _ { \ell }$ </td><td>optimal reads</td><td> $\bar { e } _ { \ell }$ </td><td>optimal reads</td><td> $\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } }$ </td></tr><tr><td>0→1</td><td>0.06</td><td>4.3</td><td>0.98</td><td>0.98</td><td>1.01</td><td>1.01</td><td>0.99</td></tr><tr><td>5→6</td><td>0.02</td><td>0.2</td><td>1.11</td><td>0.99</td><td>0.97</td><td>1.09</td><td>0.83</td></tr><tr><td>6→7</td><td>0.09</td><td>0.2</td><td>1.24</td><td>1.17</td><td>0.93</td><td>0.92</td><td>0.99</td></tr><tr><td>22→23</td><td>0.08</td><td>0.5</td><td>0.87</td><td>1.23</td><td>1.02</td><td>0.93</td><td>0.85</td></tr><tr><td>23→24 (exit)</td><td>0.05</td><td>1.6</td><td>0.98</td><td>0.98</td><td>0.99</td><td>1.09</td><td>0.49</td></tr><tr><td>4→20, sixteen layers as a block</td><td>0.02</td><td>1.0</td><td>0.97</td><td>1.00</td><td>0.96</td><td>1.03</td><td>0.86</td></tr></table>

Table 3: $A / P$ in the full 768-d space and in the 16-d camera, Pythia-160m WikiText mean flow, with the share of the common-shift energy A and of the token-specific energy P that the camera keeps.
<table><tr><td>transition</td><td>0→1</td><td>1→2</td><td>2→3</td><td>3→4</td><td>4→5</td><td>5→6</td><td>6→7</td><td>7→8</td><td>8→9</td><td>9→10</td><td>10→11</td><td>11→12</td></tr><tr><td>A/P, full space</td><td>1.5</td><td>1.5</td><td>0.7</td><td>0.3</td><td>1.0</td><td>1.8</td><td>1.0</td><td>1.8</td><td>2.5</td><td>2.1</td><td>1.6</td><td>84</td></tr><tr><td> $A { \dot { / } } P ,$  camera</td><td>3.7</td><td>3.1</td><td>0.6</td><td>0.1</td><td>1.2</td><td>5.2</td><td>1.4</td><td>3.7</td><td>6.2</td><td>3.1</td><td>1.5</td><td>109</td></tr><tr><td>A kept by camera</td><td>0.74</td><td>0.55</td><td>0.29</td><td>0.30</td><td>0.24</td><td>0.46</td><td>0.24</td><td>0.64</td><td>0.70</td><td>0.36</td><td>0.49</td><td>0.84</td></tr><tr><td>P kept by camera</td><td>0.30</td><td>0.26</td><td>0.30</td><td>0.83</td><td>0.20</td><td>0.16</td><td>0.17</td><td>0.32</td><td>0.28</td><td>0.24</td><td>0.52</td><td>0.65</td></tr></table>

Results We report the results in Tables 1 and 2, respectively for Pythia-160m and Pythia-410m and in Table 3 the ratio $A / P$ in the full space next to its camera value, which shows how the camera changes that ratio.

A shift of the whole cloud is optimal for free, and it dominates the cost: $A / P$ is 1.5–2.5 in the full space at most transitions and 84 at the 160m exit. The camera changes $A / P$ by up to a factor of three either way, since it keeps different shares of the shift and of the token-specific movement at each transition (Table 3), but the full-space values, computed without a camera, show the same behavior. Therefore, we are confidence that using the reduced space does not alter the general conclusions.

Once the common shift is removed, the network’s move can be compared with the cheapest move properly. At three of the four measurable transitions in Pythia-160m (entry, 3→4, 10→11) and at every measurable transition in Pythia-410m, the network coupling is optimal, within the instrument’s precision of 10–15%. In particular, at the entry, the sliced bound says the true efficiency is at least 0.94, so the network is within 6% of optimal there.

One measured gap: the 160m exit. The token-specific efficiency reads 0.86 where the calibration reads 1.00. The network pays between 14% and 27% more than the cheapest move there. The 410m exit shows no gap. It remains an open question to know if this is a property of the model or the scale.

Between layers 4 and 10 the token-specific moves are tiny compared with the distance between tokens. The token-specific moves there are too small for 4,000 samples to resolve, and the sliced bound only says the efficiency is above 0.2–0.4. No cost verdict either way at the single-layer scale. However, pairing layer ℓ directly with layer ℓ+k makes the accumulated step resolvable once it is large enough: six layers in the early middle (3→9 reads e¯ = 0.97 against a calibration of 0.95) and three in the late middle (8→11 reads 0.96).

Similarly, most transitions of Pythia-410m are not resolvable. With twice the depth, each layer moves its tokens less relative to their spacing (the token-specific movement is 3–6% of the cost in the middle, against 16–32% for Pythia-160m), and a larger sample would help only slowly. Our hypothesis is that those small moves are linear, as the 410m exit is to within resolution but testing it is left for future work.

## 3.2 A lower bound without calibration

Protocol We use the sliced Wasserstein construction [Bonneel et al., 2015]. For a random unit direction u, we project every point on u, sort, and compute the sorted cost $W _ { 2 , u } ^ { 2 }$ , which is the exact optimal transport in one dimension. Since $\mathbb { E } _ { u } ( u \cdot v ) ^ { 2 } = \| v \| ^ { 2 } / D$ and sorting is the minimum over all pairings of the projected points, $D \mathbb { E } _ { u } W _ { 2 , u } ^ { 2 } \leq W _ { 2 } ^ { 2 } ( \mu _ { \ell } , \dot { \mu } _ { \ell + 1 } )$ , with equality for a shift or a uniform scaling and a strict inequality in general. Averaging over 2,000 directions on the centered clouds inside the camera, with all 25,268 types on both sides and no sampling, gives

![](images/d57185baca5da9c292749f16de737fd49f433cdbcd952d976ff9f349103a8ebc.jpg)  
Figure 1: Rank agreement $\rho$ between the network’s move and the optimal-transport move, against depth, both models, 16-d camera, exact assignment on the centered clouds, with the shuffled-pairing null (grey). Agreement rises from the entry to the exit on both models.

$$
\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } } = \frac { D \mathbb { E } _ { u } W _ { 2 , u } ^ { 2 } ( \bar { \mu } _ { \ell } , \bar { \mu } _ { \ell + 1 } ) } { P _ { \ell } } \leq \bar { e } _ { \ell } ,\tag{2}
$$

a lower bound on the token-specific efficiency that needs no sampling, no floor and no calibration. Because it is a lower bound, the true efficiency is at least $\bar { e } _ { \ell } ^ { \mathrm { s l i c e d } }$ (Tables 1 and 2): 0.94 at the entry and 0.73 at the exit of Pythia-160m, and 0.86 over the sixteen-layer block 4 → 20 of Pythia-410m, with no assumption.

In conclusion, wherever the cost can be measured, the network moves its tokens at the price of an optimal coupling, with one exception, the exit of Pythia-160m, which pays 14 to 27% more.

## 4 Agreement with the Brenier map

The second question is whether token i’s actual move matches the Brenier route $\nabla \varphi ( x _ { i } ) - x _ { i } .$ . On the centered clouds, the exact assignment sends each sampled token $x _ { i }$ to the destination $y _ { \sigma ( i ) }$ an optimal mover would choose. We compare the network’s move $d _ { i }$ with the optimal move $y _ { \sigma ( i ) } - x _ { i }$ by the Spearman rank agreement $\rho$ of the pooled components and by the median cosine between the two moves. Because both clouds are centered, a common shift cannot create agreement.

We present the results in Table 4, including a null shuffle as control. Agreement between the network’s move and the optimal-transport move rises with depth (Figure 1). The 410m curve dips near 0.4 of depth, at the transitions where its tokens move least. At the 160m exit the common shift is 99% of the cost $( A / P$ 84 in the full space), so the map result concerns the remaining 1%, which is resolvable (floor 0.11 of $P _ { \ell } )$ . On Pythia-160m, $\rho$ is 0.41 at the entry, 0.38–0.62 in the middle, 0.80 at 10→11 and 0.89 at the exit, where the median cosine is 0.95 (Figure 2, and Figure 3 for all twelve transitions). Pile mean flows and raw states follow the same profile (exit ρ 0.85 and 0.74). Pythia-410m ends at ρ 0.88, median cosine 0.95. Seed ranges are ±0.01, and the shuffled-pairing null stays at $\rho \le 0 . 0 8$ . The network’s move is the optimal-transport move at the last layer, partly so in the middle, and barely at the first.

Is the agreement with the optimal-transport map more than a linear change? To answer this, we fitted a linear optimal map, the Gaussian Brenier map $T _ { G }$ , from means and covariances (definition and full table in Appendix A). It explains 39% of the token-specific move at the 160m exit, and the remaining 61% still agrees with the exact optimal transport $( \rho 0 . 8 3 ,$ , median cosine 0.94, null 0.01). At the 410m exit and at 160m 10 → 11 the linear map explains 90% and 73% and the remainder is below the floor. Elsewhere the linear map fits the network’s moves at least as well as the full OT map and the remainder agrees at $\rho$ 0.27–0.36. In conclusion, the answer is yes for the 160m exit and no for every other transition we can resolve.

![](images/a1aa79c7a62ff5efb20ec625ff637779a66580d0103af1e9473d1f4be2b2ee59.jpg)  
Figure 2: For each token, the displacement prescribed by the exact optimal transport between the two layer clouds (horizontal) against the displacement the network actually applied (vertical), all 16 camera coordinates pooled, shown as a density. Agreement is the diagonal. $\rho$ is the rank agreement of the pooled components.

Table 4: Map agreement, 16-d camera, exact assignment on the centered clouds. ρ: rank agreement of the network’s move with the optimal-transport move, per object; median cosine on WikiText; null ρ from the shuffled pairing, the largest of the three objects. Seed ranges of $\rho$ are ±0.01. Pythia-410m rows: WikiText only.
<table><tr><td></td></tr><tr><td>transition</td><td>ρ WikiText Pile</td><td>raw states</td><td>median cos</td><td>null  $\rho$ </td></tr><tr><td>0→1</td><td>0.41</td><td>0.17</td><td>0.21</td><td>0.58 0.02</td></tr><tr><td>1→2</td><td>0.48</td><td>0.45</td><td>0.38</td><td>0.60 0.01</td></tr><tr><td>2→3</td><td>0.57</td><td>0.63</td><td>0.48 0.62</td><td>0.02</td></tr><tr><td>3→4</td><td>0.59</td><td>0.62</td><td>0.56 0.92</td><td>0.16</td></tr><tr><td>4→5</td><td>0.60</td><td>0.54</td><td>0.40</td><td>0.65 0.03</td></tr><tr><td>5→6</td><td>0.38</td><td>0.29</td><td>0.26</td><td>0.40 0.00</td></tr><tr><td>6→7</td><td>0.46</td><td>0.38</td><td>0.27</td><td>0.48 0.01</td></tr><tr><td>7→8</td><td>0.62</td><td>0.56</td><td>0.40</td><td>0.71 0.01</td></tr><tr><td>8→9</td><td>0.61</td><td>0.52</td><td>0.38</td><td>0.69 0.01</td></tr><tr><td>9→10</td><td>0.57</td><td>0.51</td><td>0.36</td><td>0.65 0.01</td></tr><tr><td>10→11</td><td>0.80</td><td>0.76</td><td>0.50</td><td>0.87 0.04</td></tr><tr><td>11→12</td><td>0.89</td><td>0.85</td><td>0.74</td><td>0.95 0.08</td></tr><tr><td>410m 0→1</td><td>0.50</td><td>一</td><td>一</td><td>0.66 0.01</td></tr><tr><td>410m 5→6</td><td>0.59</td><td>一</td><td>0.97 一</td><td>0.10</td></tr><tr><td>410m 6→7</td><td>0.46</td><td>一</td><td>一</td><td>0.95 0.10</td></tr><tr><td>410m 22→23</td><td>0.70</td><td>一</td><td>一</td><td>0.86 0.09</td></tr><tr><td>410m 23→24</td><td>0.88</td><td>一</td><td></td><td>0.95 0.06</td></tr></table>

Is the exit agreement learned? We repeat the measurement on Pythia-160m untrained (checkpoint step0), early in training (step4000) and fully trained, all from the same 5,000 documents. Exit rank agreement goes from 0.64 to 0.67 to 0.86 (median cosine 0.74, 0.75, 0.94), and at L10 → 11 from 0.18 to 0.42 to 0.77. At the entry the order reverses: 0.52 untrained against 0.37 trained. Therefore, the rise to 0.86 is learned, and mostly after step 4000. Training does not push the whole network toward optimal transport. It pushes the exit toward it (0.64 → 0.86) and the entry away from it (0.52 → 0.37). The linear control explains the 0.64: at initialisation the exit move is linear (a linear map explains 97% of it), and a linear map agrees with optimal transport by construction. Training replaces it by a move that is almost entirely non-linear, and that part follows optimal transport (ρ 0.81).

## 5 Conclusion

In this paper, we asked whether a trained transformer moves its token states the way optimal transport would. Two questions: does each layer pay the cheapest cost, and does each token go where the optimal map sends it? To answer these questions, we compared the moves of Pythia-160m and Pythia 410m with exact optimal transport between consecutive layers, on two corpora, in a 16-dimensional projection.

![](images/343c29d5479457ccdd81869a3427d208be00d2b612c6e99c65cc9eb7a3c61326.jpg)  
Figure 3: For each transition of Pythia-160m, the network’s move (blue) and the optimal-transport move (orange) for the token sample, drawn as bin means in the leading plane of the layer-ℓ cloud (16-d camera, centered clouds, exact assignment). One arrow scale per panel. ρ is the rank agreement of the pooled 16-d components.

On cost, the token-specific move of the network is at the optimal reading at every transition we could resolve, except the last transition of Pythia-160m. On the map, the agreement between the network’s move and the optimal-transport move increases with depth and reaches 0.89 at the last transition of Pythia-160m and 0.88 for Pythia-410m. This agreement is much weaker at initialisation and grows with training, while at the entry it decreases. For Pythia-160m it holds beyond any linear transformation of the cloud and for Pythia-410m it is accounted for by a linear map.

Our study suffers from several limitations, the main one being the resolution. With 4,000 tokens per sample, only the transitions where tokens move further than the sampling gap can be resolved: four of twelve in Pythia-160m and five of twenty-four in Pythia-410m. The middle layers could only be assessed in blocks of several layers. We can address this with larger sample and the required compute power. Another obvious limitation is that the study covers only two small models of one family on two small corpora. Finally, the mean flow averages each token over its contexts, which raises the agreement relative to raw states.

Despite these limitations, the picture that emerges is that the last layer of a trained transformer acts as an optimal-transport map on its cloud of states, the first layer does not, and the middle layers move each token too little to be judged one layer at a time. Several questions remain open. The middle layers may be linear, as the last layer of Pythia-410m is, but an instrument with finer resolution is needed to tell. The cost gap at the last layer of Pythia-160m, absent in Pythia-410m, may close with scale or may just be an artifact of Pythia’s models. Finally, nothing in our results says whether the agreement with optimal transport is a property of language models or of trained transformers in general.

## References

David Alvarez-Melis and Tommi Jaakkola. Gromov-Wasserstein alignment of word embedding spaces. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1881–1890, 2018.

Nora Belrose, Igor Ostrovsky, Lev McKinney, Zach Furman, Logan Smith, Danny Halawi, Stella Biderman, and Jacob Steinhardt. Eliciting latent predictions from transformers with the tuned lens. arXiv preprint arXiv:2303.08112, 2023.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, Aviya Skowron, Lintang Sutawika, and Oskar van der Wal. Pythia: A suite for analyzing large language models across training and scaling. In ICML, volume 202 of PMLR, pages 2397–2430, 2023.

Nicolas Bonneel, Michiel van de Panne, Sylvain Paris, and Wolfgang Heidrich. Displacement interpolation using Lagrangian mass transport. ACM Transactions on Graphics, 30(6):158:1– 158:12, 2011.

Nicolas Bonneel, Julien Rabin, Gabriel Peyré, and Hanspeter Pfister. Sliced and radon wasserstein barycenters of measures. Journal of Mathematical Imaging and Vision, 51(1):22–45, 2015.

Yann Brenier. Polar factorization and monotone rearrangement of vector-valued functions. Communications on Pure and Applied Mathematics, 44(4):375–417, 1991.

Lénaïc Chizat, Pierre Roussillon, Flavien Léger, François-Xavier Vialard, and Gabriel Peyré. Faster Wasserstein distance estimation with the Sinkhorn divergence. In Advances in Neural Information Processing Systems, volume 33, 2020.

Jean Feydy, Thibault Séjourné, François-Xavier Vialard, Shun-ichi Amari, Alain Trouvé, and Gabriel Peyré. Interpolating between optimal transport and mmd using sinkhorn divergences. In AISTATS, volume 89 of PMLR, pages 2681–2690, 2019.

Chris Finlay, Jörn-Henrik Jacobsen, Levon Nurbekyan, and Adam Oberman. How to train your neural ODE: the world of Jacobian and kinetic regularization. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119, 2020.

Rémi Flamary, Nicolas Courty, Alexandre Gramfort, Mokhtar Z. Alaya, Aurélie Boisbunon, Stanislas Chambon, Laetitia Chapel, Adrien Corenflos, Kilian Fatras, Nemo Fournier, Léo Gautheron, Nathalie T. H. Gayraud, Hicham Janati, Alain Rakotomamonjy, Ievgen Redko, Antoine Rolet, Antony Schutz, Vivien Seguy, Danica J. Sutherland, Romain Tavenard, Alexander Tong, and Titouan Vayer. POT: Python optimal transport. Journal of Machine Learning Research, 22(78): 1–8, 2021.

Nicolas Fournier and Arnaud Guillin. On the rate of convergence in wasserstein distance of the empirical measure. Probability Theory and Related Fields, 162(3–4):707–738, 2015.

Takashi Furuya, Maarten V. de Hoop, and Matti Lassas. Transformers through the lens of supportpreserving maps between measures. arXiv preprint arXiv:2509.25611, 2025.

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. A mathematical perspective on transformers. Bulletin ofthe American Mathematical Society, 62(3):427–479, 2025.

Kelvin Kan, Xingjian Li, and Stanley Osher. OT-Transformer: A continuous-time transformer architecture with optimal transport regularization. arXiv preprint arXiv:2501.18793, 2025.

Alexander Korotin, Daniil Selikhanovych, and Evgeny Burnaev. Neural optimal transport. In International Conference on Learning Representations, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

Ashok Makkuva, Amirhossein Taghvaei, Sewoong Oh, and Jason Lee. Optimal transport mapping via input convex neural networks. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119, 2020.

Michael E. Sander, Pierre Ablin, Mathieu Blondel, and Gabriel Peyré. Sinkformers: Transformers with doubly stochastic attention. In AISTATS, volume 151 of PMLR, pages 3515–3530, 2022.

Raphaël Sarfati, Toni J. B. Liu, Nicolas Boullé, and Christopher J. Earls. Lines of thought in large language models. In ICLR, 2025.

Sidak Pal Singh and Martin Jaggi. Model fusion via optimal transport. In Advances in Neural Information Processing Systems, volume 33, 2020.

James Vuckovic, Aristide Baratin, and Remi Tachet des Combes. A mathematical theory of attention. arXiv preprint arXiv:2007.02876, 2020.

Hong Wang and Kelly Wang. Transformers as optimal transport: Stability, geometry, and gauge symmetry. OpenReview submission 2A2vRGHW16 to ICLR 2026, 2025. Not accepted; unpublished at the time of writing.

Jonathan Weed and Francis Bach. Sharp asymptotic and finite-sample rates of convergence of empirical measures in Wasserstein distance. Bernoulli, 25(4A):2620–2648, 2019.

## A Linear map and remainder

This appendix gives the numbers behind the linear-map control of Section 4. The question is whether the agreement between the network’s move and the optimal-transport move could come from a linear transformation of the whole cloud, a shift and a stretch, rather than from a token-by-token rearrangement.

The linear map is the Gaussian Brenier map $T _ { G } .$ . For two clouds with means $m _ { \ell } , m _ { \ell + 1 }$ and covariances $\Sigma _ { \ell } , \Sigma _ { \ell + 1 }$ , it is the optimal transport between the two Gaussians with those statistics,

$$
T _ { G } ( x ) = m _ { \ell + 1 } + A ( x - m _ { \ell } ) , A = \Sigma _ { \ell } ^ { - 1 / 2 } \bigl ( \Sigma _ { \ell } ^ { 1 / 2 } \Sigma _ { \ell + 1 } \Sigma _ { \ell } ^ { 1 / 2 } \bigr ) ^ { 1 / 2 } \Sigma _ { \ell } ^ { - 1 / 2 } ,\tag{3}
$$

with $A$ symmetric positive definite, so that $T _ { G }$ is the gradient of a convex quadratic and hence a Brenier map. It maps the first cloud onto one with exactly the mean and covariance of the second, and it is fitted from those statistics alone. We measure two things. First, how much of the network’s token-specific move the linear map explains, as the $R ^ { 2 }$ and the rank agreement $\rho$ between the linear move and the network’s move. Second, what is left: we push the layer-ℓ cloud through $T _ { G }$ , take the move that remains from the pushed cloud to the target cloud, and compare it with the exact optimal transport between those two clouds, with the same floor and the same shuffled-pairing null as in the main text. The null reads $\rho \le 0 . 0 1$ at every transition.

Table 5 reports both. At the exit of Pythia-160m the linear map explains 39% of the move and the remainder, 61% of the cost, still agrees with optimal transport at ρ 0.83. At the exit of Pythia-410m and at 10 → 11 of Pythia-160m the linear map explains 90% and 73%, and the remainder is smaller than its own floor, so its agreement cannot be measured. This last point matters for reading the table: a low remainder agreement next to a floor above one means "not resolvable", not "weak".

Table 5: Linear-map control, 16-d camera, $n = 4 , 0 0 0$ . Linear map: $R ^ { 2 }$ and $\rho$ of the move prescribed by $T _ { G }$ against the network’s token-specific move. Remainder: its share of the cost, its floor relative to its own cost, and its agreement with the exact optimal transport $( \rho ,$ median cosine). Last column: $\rho$ of the full optimal-transport map, for comparison.
<table><tr><td rowspan="2"></td><td colspan="2">linear map</td><td colspan="4">remainder</td><td rowspan="2">full map ρ</td></tr><tr><td> $R ^ { 2 }$ </td><td>ρ</td><td>share of cost</td><td>floor</td><td> $\rho$ </td><td>median cos</td></tr><tr><td>160m 0→1</td><td> $- 0 . 1 5$ </td><td>0.35</td><td>1.15</td><td>0.07</td><td>0.27</td><td>0.30</td><td>0.40</td></tr><tr><td> $1 6 0 \mathrm { m } ~ 1  1 0$ </td><td> $0 . 0 7 { - } 0 . 5 1 $ </td><td> $0 . 4 8 – 0 . 7 6$ </td><td> $0 . 4 9 – 0 . 9 3$ </td><td> $0 . 8 – 4 . 5$ </td><td> $0 . 2 7 { - } 0 . 3 6 $ </td><td>0.32–0.51</td><td>0.38–0.63</td></tr><tr><td> $1 6 0 \mathrm { m } ~ 1 0 {  } 1 1$ </td><td>0.73</td><td>0.81</td><td>0.27</td><td>0.47</td><td>0.44</td><td>0.52</td><td>0.80</td></tr><tr><td> $1 6 0 \mathrm { m } 1 1  1 2$ </td><td>0.39</td><td>0.62</td><td>0.61</td><td>0.26</td><td>0.83</td><td>0.94</td><td>0.89</td></tr><tr><td> $\mathrm { r a w ~ s t a t e s ~ } 1 1  1 2$ </td><td>0.57</td><td>0.66</td><td>0.43</td><td>0.28</td><td>0.51</td><td>0.71</td><td>0.74</td></tr><tr><td> $4 1 0 \mathrm { m } 2 2  2 3 $ </td><td>0.96</td><td>0.82</td><td>0.04</td><td>2.3</td><td>0.35</td><td>0.36</td><td>0.70</td></tr><tr><td> $4 1 0 \mathrm { m } 2 3  2 4$ </td><td>0.90</td><td>0.91</td><td>0.10</td><td>0.54</td><td>0.47</td><td>0.54</td><td>0.88</td></tr></table>