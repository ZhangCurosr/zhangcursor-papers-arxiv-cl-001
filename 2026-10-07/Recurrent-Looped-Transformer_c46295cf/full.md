# Recurrent Looped Transformer

Yifan Zhang<sup>1</sup> Jichen Feng<sup>2</sup> Shihan Qin<sup>2</sup>

<sup>1</sup>Princeton University <sup>2</sup>University of Pennsylvania

September 12, 2026<sup>∗</sup>

## Abstract

State tracking requires an update at every input, but the depth a Transformer applies to each token is fixed regardless of sequence length. We introduce the Recurrent Looped Transformer (RLT), which splits its layers between a parallel causal encoder and a recurrent decoder. At each token, the decoder merges the encoder output with the previous token’s final decoder state, so the computation path grows with sequence length at a fixed per-token cost. On six algorithmic tasks, we compare five splits of eight layers with an eight-layer Transformer over three seeds. Trained on at most 40 bits, two RLT splits generalize parity to 256 bits with 100% accuracy in every seed, while the Transformer stays at chance. On swap-based $S _ { 5 }$ permutation tracking at eight times the training length, RLT reaches 97% final-state accuracy versus under 1% for the Transformer, and accuracy increases with decoder depth. On modular arithmetic beyond the training lengths, RLT reaches up to 93% versus 33% for the Transformer. Ablations show that these gains depend on the feedback: removing it drops parity and swap-based $S _ { 5 }$ to chance at every split. Updating the feedback once per four-token chunk lets known tokens in a chunk run in parallel and keeps 64-bit parity at 99%, while permutation tracking depends on per-token feedback: chunking lowers length-64 swap-based $S _ { 5 }$ from 100% to 20%.

Project Page: https://github.com/yifanzhang-pro/recurrent-looped-tranformer

![](images/66eacc42e8d7b64f149454b79ac25fe795407674f6a3857c6bda693fc5b005de.jpg)  
Figure 1 Recurrence across prompt and response. The encoder builds causal key–value (KV) memory; the decoder processes tokens in order. Orange arrows carry the decoder state $H _ { t } = ( s _ { t } , C _ { t } ^ { D } )$ : the merge reads $s _ { t } ,$ and each SWA layer reads its own cache. Blue arrows supply encoder memory up to the current position. The drawing shows the last two prompt updates and the first response update.

![](images/6e1fdb808f15d560048e05ba74945984eacaf394e50e1c5c9cf7b10dbeee4867.jpg)  
Figure 2 RLT-1 architecture for one token update. Computation flows upward; each outlined stack repeats for the indicated depth. The encoder output supplies both the gated merge and the global KV projection. With G = 1, all decoder layers use their own queries to read the same cached, projected global KV. Each decoder layer constructs separate SWA KV from its own input, attends to its local window, then reads global memory and applies an FFN. Residual paths bypass each attention or FFN sublayer. The final decoder output $s _ { t }$ feeds the next token’s merge, while each layer retains its own SWA cache. Blue arrows carry encoder features and global memory, green arrows carry local KV, and orange arrows carry recurrent hidden states.

## 1 Introduction

Algorithmic state tracking requires a model to update a state with each input: parity accumulates bits, and permutation tracking composes group elements. A Transformer computes each position with a fixed number of layers, and unless ${ \mathsf { T } } { \mathsf { C } } ^ { 0 } = { \mathsf { N } } { \mathsf { C } } ^ { 1 }$ , fixed-depth Transformers cannot track compositions of $S _ { 5 }$ permutations over arbitrarily long sequences (Merrill et al., 2024). A recurrent network instead passes the result of each update to the next, so its computation path grows with the input. Adding this connection to a Transformer lets later tokens build on earlier computation along a path that grows with sequence length.

We introduce the Recurrent Looped Transformer (RLT), which divides its layers between a causal encoder and a recurrent Transformer decoder (Figure 1). The encoder processes known tokens in parallel and builds token representations and global key–value memory. At each position, a gated merge combines the current encoder representation with the previous token’s final decoder output. The decoder then attends to the encoder prefix and its own recent activations before predicting the next token. Prompt and response tokens follow the same transition. With decoder depth $L _ { D } ,$ processing t tokens creates a recurrent path through $t L _ { D }$ decoder blocks, while each token passes through only $L _ { E } + L _ { D }$ layers.

This design introduces two choices that a decoder-only Transformer does not have. The first is the split between encoder and decoder depth: at a fixed total layer count, moving layers to the decoder lengthens the recurrent path but adds decoder work that follows token order during training and prefill. The second is the feedback interval $B ,$ the number of tokens between feedback updates. RLT-1 feeds back at every token $( B = 1 )$ , RLT-2 shares one feedback state across each chunk of B tokens, and RLT-0 removes the feedback $( B = \infty )$ . For $T$ known tokens, the decoder needs $T L _ { D }$ sequential block stages in RLT-1, $\lceil T / B \rceil L _ { D }$ in RLT-2, and $L _ { D }$ in RLT-0 (Table 1); RLT-2 uses the same parameters as RLT-1.

We compare five allocations of eight layers with an eight-layer decoder-only Transformer on six algorithmic tasks: addition, parity, modular arithmetic with and without brackets, and standard and swap-based $S _ { 5 }$ state tracking. All comparisons use three initialization seeds and shared training and test data, and evaluate lengths well beyond the training range. Our contributions are as follows.

• Architecture. RLT combines encoder-memory reuse (Sun et al., 2024) with feedback through the full decoder at every token. The appendices specify training by backpropagation through the full recurrent history, cache reuse across turns, and exact policy replay.

• Length generalization. Trained on at most 40 bits, RLT splits $5 + 3$ and $7 + 1$ reach $1 0 0 \pm 0 \%$ parity accuracy at 256 bits in every seed, compared with $5 0 . 0 7 \pm 1 . 6 3 \%$ for the Transformer. On swapbased $S _ { 5 }$ at 256 operations, eight times the training length, $4 + 4$ reaches $9 7 . 3 0 \pm 2 . 7 6 \%$ final-state accuracy, compared with $0 . 8 5 \pm 0 . 3 0 \%$ . After 5,000 training steps, $6 + 2$ reaches $9 3 . 3 6 \pm 5 . 6 9 \%$ on flat mod-5 expressions of length 63, compared with $3 3 . 2 0 \pm 2 . 3 3 \%$ (Section 3.2).

• Depth allocation. The best split depends on the task: parity generalizes best with one or three decoder layers, whereas swap-based $S _ { 5 }$ accuracy at 256 operations rises with decoder depth, from chance with one decoder layer to $9 7 \%$ with four.

• Feedback interval. Without feedback, parity and swap-based $S _ { 5 }$ stay near chance from length 64 at every split. With four-token chunks, RLT-2 keeps 64-bit parity at $9 8 . 9 9 \pm 1 . 6 6 \%$ but lowers swap-based $S _ { 5 }$ at 64 operations from 100% to $1 9 . 6 0 \pm 6 . 5 4 \%$ , so permutation tracking benefits most from per-token feedback.

Feedback Transformer, Recurrent Transformer, Full-bandwidth Transformer, $\mathrm { T } ^ { \mathrm { 2 } } \mathrm { M L R } .$ , and Latent Recurrent Transformer also pass information across tokens through feedback (Fan et al., 2020; Oncescu et al., 2026; Wang et al., 2026; Cai et al., 2026; Huang et al., 2026). RLT feeds back through the full decoder, keeps this recurrence separate from parallel encoder memory, and treats the allocation of depth between encoding and recurrent updates as a design variable (Section 4).

## 2 Method

This section defines RLT-1 and two controls that change its feedback interval.

## 2.1 Sequence and state

Let $x _ { 1 : S }$ be an independent sequence beginning with $x _ { 1 } = \mathrm { B O S }$ . At inference, $x _ { 1 : T }$ is the prompt and later tokens form the response. The same conditional distribution applies before and after T. Let $L _ { E }$ and $L _ { D }$ be the encoder and decoder depths, and let d be the residual width. We write the encoder representation $e _ { t } \in \mathbb { R } ^ { d }$ and recurrent output $s _ { t } \in \mathbb { R } ^ { d }$ as column vectors.

The complete decoder state is $H _ { t } = ( s _ { t } , C _ { t } ^ { D } )$ . The cache $C _ { t } ^ { D }$ holds the key/value projections retained by each decoder SWA layer. The window size $W \geq 1$ includes the current token, so each layer retains at most $W - 1$ past positions after an update. We collect all parameters, including the learned initial state $s _ { \star } .$ , in Θ. The encoder and decoder are denoted by $E _ { \theta }$ and $D _ { \phi }$

## 2.2 Causal encoder and memory

For an observed prefix, compute

$$
e _ { 1 : T } = E _ { \theta } ( x _ { 1 : T } ) .\tag{2.1}
$$

Positions within each encoder layer can be processed together using a causal mask. For decoder memory group $g \in \{ 1 , \ldots , G \}$ , construct

$$
k _ { t } ^ { g } = \mathcal { P } _ { K } ^ { g } ( e _ { t } , t ) , \qquad v _ { t } ^ { g } = W _ { V } ^ { g } \mathrm { R M S N o r m } _ { E } ( e _ { t } ) , \qquad M _ { \leq t } ^ { g } = \{ ( k _ { j } ^ { g } , v _ { j } ^ { g } ) \} _ { j = 1 } ^ { t } .\tag{2.2}
$$

The key map includes normalization, projection, and any positional transformation. Decoder layer $\ell$ reads group $g ( \ell ) \colon G = 1$ shares memory across layers, while $G = L _ { D }$ allows separate projections for each layer. Separate attention sublayers read the two stores: cross-attention reads encoder memory $M _ { \leq t }$ over the full prefix, and causal SWA reads decoder KV within a bounded window.

## 2.3 One transition for every token

Initialize once, before BOS:

$$
H _ { 0 } = ( s _ { \star } , \emptyset ) .\tag{2.3}
$$

For every observed or sampled token, apply

$$
u _ { t } = \mathrm { M e r g e } ( e _ { t } , s _ { t - 1 } ) ,\tag{2.4}
$$

$$
H _ { t } = ( s _ { t } , C _ { t } ^ { D } ) = D _ { \phi } ( u _ { t } ; M _ { \leq t } , C _ { t - 1 } ^ { D } , t ) , \qquad t \geq 1 ,\tag{2.5}
$$

$$
p _ { \Theta } ( x _ { t + 1 } \mid x _ { 1 : t } ) = \mathrm { s o f t m a x } \bigl ( W _ { o } \mathrm { R M S N o r m } _ { o } ( s _ { t } ) \bigr ) _ { x _ { t + 1 } } .\tag{2.6}
$$

Define ${ \cal F } _ { t } ( H ) = D _ { \phi } ( \mathrm { M e r g e } ( e _ { t } , s ) ; M _ { < t } , C ^ { D } , t )$ for $H ~ = ~ ( s , C ^ { D } )$ . After the encoder features are computed, the decoder processes every prompt position to predict the first response token:

$$
H _ { T } = F _ { T } \circ F _ { T - 1 } \circ \cdot \cdot \cdot \circ F _ { 1 } ( H _ { 0 } ) .\tag{2.7}
$$

Generation continues from both components of $H _ { T }$ . After sampling $x _ { T + 1 }$ from $s _ { T } .$ , encode it incrementally, append its encoder-derived $\mathrm { K V } ,$ and compute $H _ { T + 1 } = F _ { T + 1 } ( H _ { T } )$ , including every decoder SWA cache update.

## 2.4 Prompt prefill and incremental decoding

Prefill first computes the causal encoder representations and memory of the prompt, then evaluates $H _ { 1 } , \ldots , H _ { T }$ in order. At decoder position $t ,$ attention is restricted to $M _ { \leq t }$ even though the entire prompt memory is available. Generation uses

$$
( e _ { t } , C _ { t } ^ { E } ) = E _ { \theta } ^ { \mathrm { s t e p } } ( x _ { t } , C _ { t - 1 } ^ { E } )\tag{2.8}
$$

with encoder cache $C ^ { E }$ , followed by a memory append and decoder update. Observed tokens may be encoded in chunks if the encoder cache is preserved. The decoder still processes every token in order: each update produces the hidden state and SWA KV needed by later positions.

## 2.5 Merge and decoder blocks

We use the following gated merge:

$$
r _ { t - 1 } = \mathrm { R M S N o r m } _ { s } ( s _ { t - 1 } ) ,\tag{2.9}
$$

$$
g _ { t } = \sigma \big ( W _ { g } [ e _ { t } ; r _ { t - 1 } ] + b _ { g } \big ) ,\tag{2.10}
$$

$$
u _ { t } = e _ { t } + \alpha g _ { t } \odot W _ { s } r _ { t - 1 } .\tag{2.11}
$$

Here $W _ { g } \in \mathbb { R } ^ { d \times 2 d } , W _ { s } \in \mathbb { R } ^ { d \times d }$ , and α controls the feedback scale. The main comparison uses $\alpha = 0 . 1$ Starting from $z _ { t } ^ { 0 } = u _ { t } ,$ each decoder block applies causal SWA, encoder-memory cross-attention, and a feed-forward network (FFN):

$$
q _ { t } ^ { D , \ell } = \mathcal { P } _ { Q } ^ { D , \ell } ( z _ { t } ^ { \ell - 1 } , t ) ,\tag{2.12}
$$

$$
k _ { t } ^ { D , \ell } = \mathcal { P } _ { K } ^ { D , \ell } ( z _ { t } ^ { \ell - 1 } , t ) , \qquad v _ { t } ^ { D , \ell } = W _ { V } ^ { D , \ell } \mathrm { R M S N o r m } _ { S , \ell } ( z _ { t } ^ { \ell - 1 } ) ,\tag{2.13}
$$

$$
\begin{array} { r } { b _ { t } ^ { \ell } = z _ { t } ^ { \ell - 1 } + \mathrm { A t t n } _ { \ell } ^ { D } \Big ( q _ { t } ^ { D , \ell } , \{ ( k _ { j } ^ { D , \ell } , v _ { j } ^ { D , \ell } ) \} _ { j = \mathrm { m a x } ( 1 , t - W + 1 ) } ^ { t } \Big ) , } \end{array}\tag{2.14}
$$

$$
a _ { t } ^ { \ell } = b _ { t } ^ { \ell } + \mathrm { A t t n } _ { \ell } ^ { M } \left( \mathcal { P } _ { Q } ^ { M , \ell } ( b _ { t } ^ { \ell } , t ) , M _ { \leq t } ^ { g ( \ell ) } \right) ,\tag{2.15}
$$

$$
z _ { t } ^ { \ell } = a _ { t } ^ { \ell } + \mathrm { F F N } _ { \ell } ( \mathrm { R M S N o r m } _ { D , \ell } ( a _ { t } ^ { \ell } ) ) , \qquad s _ { t } = z _ { t } ^ { L _ { D } } .\tag{2.16}
$$

The attention operators include output projections; query/key maps include their normalizations and positional transformations. Each layer forms current KV from its input before SWA, so attention to the current position introduces no circular dependency. Historical decoder KV comes from $C _ { t - 1 } ^ { D }$ . After the update, retain positions max $\cdot ( 1 , t - W + 2 ) , \ldots , t$ in $C _ { t } ^ { D }$ ; this set is empty for $W = 1$ . Figure 2 expands the encoder and decoder stacks for one token update.

## Block count per token and along the recurrent path

Illustrative 48-layer encoder + 48-layer decoder; counts exclude merge and readout.

![](images/01198314e9fdcbc18e560cc484612a57ac2879ef6ba123679808f1f3c769c5cb.jpg)  
Figure 3 Recurrent path length and block count per token. For $L _ { E } = L _ { D } = 4 8 _ { \mathrm { ; } }$ , the path traverses 48t decoder blocks after t tokens. Each token evaluates 96 encoder and decoder blocks. Orange counts decoder blocks along the recurrent path; blue counts blocks in both stacks per token.

State path: 48t Decoder blocks along the recurrent chain from s<sub>0</sub>

Per token: 48 + 48 = 96 Fixed block count; attention work still grows with context

## 2.6 Recurrent depth and execution cost

After a prompt of length T and n processed response tokens, the feedback path has passed through $( T + n ) L _ { D }$ decoder blocks, while each token evaluates $L _ { E } + L _ { D }$ blocks (Figure 3). The influence of earlier states depends on the gates, projections, and products of transition Jacobians. A prompt of length T requires T sequential decoder updates; updates from independent sequences can share a batch, with separate states and caches. Appendix B derives the work and storage costs and proves that states and next-token distributions are invariant to the prompt–response split.

## 2.7 Feedback interval: RLT-1, RLT-2, and RLT-0

RLT-1 and its two controls difer in one quantity, the feedback interval B: the number of tokens between updates of the state that enters the decoder input. RLT-1 updates this state at every token (B = 1), RLT-2 once per chunk of B tokens, and RLT-0 never $( B = \infty )$ . All three keep the causal encoder, encoder memory, decoder SWA, prefix-restricted memory attention, and readout of RLT-1.

RLT-2 holds the feedback state fixed within a chunk of B tokens and updates it at the chunk boundary (Figure 4). Boundaries are anchored at BOS, so chunk k covers the known positions $I _ { k } = \{ ( k - 1 ) B + 1 , \ldots , \operatorname* { m i n } ( k B , T ) \}$ . Starting from the feedback register $h _ { 0 } = s _ { \star }$ , every position in chunk k merges its encoder output with the same boundary state through its own gate:

$$
r _ { k - 1 } = \mathrm { R M S N o r m } _ { s } ( h _ { k - 1 } ) , \qquad g _ { t } = \sigma \big ( W _ { g } [ e _ { t } ; r _ { k - 1 } ] + b _ { g } \big ) ,\tag{2.17}
$$

$$
z _ { t } ^ { 0 } = e _ { t } + \alpha g _ { t } \odot W _ { s } r _ { k - 1 } , \qquad t \in I _ { k } ,\tag{2.18}
$$

$$
h _ { k } = y _ { k B } \quad { \mathrm { w h e n ~ } } k B \leq T .\tag{2.19}
$$

Here $y _ { t } = z _ { t } ^ { L _ { D } }$ is the final decoder output at position $t ;$ every $y _ { t }$ predicts $x _ { t + 1 }$ , but only the last output of a complete chunk becomes the next feedback state. The decoder applies Equation (2.16) to all known positions of a chunk together, and causal SWA also reads keys retained from earlier chunks.

<table><tr><td>Quantity</td><td>RLT-1  $( B = 1 )$ </td><td>RLT-2 (B)</td><td>RLT-0  $( B = \infty )$ </td></tr><tr><td>Positions per decoder batch, known tokens Decoder block stages, training forward / prefill</td><td>1</td><td>up to B</td><td> $T$ </td></tr><tr><td>Backward dependency depth, full BPTT</td><td> $T L _ { D }$   $O ( T L _ { D } )$ </td><td> $K L _ { D }$   $O ( K L _ { D } )$ </td><td> $L _ { D }$   $O ( L _ { D } )$ </td></tr><tr><td>Final-state feedback updates over  $T$  tokens</td><td> $T$ </td><td> $\lfloor { T } / { B } \rfloor$ </td><td> $0$ </td></tr><tr><td>Decoder block evaluations over  $T$  tokens</td><td> $T L _ { D }$ </td><td> $T L _ { D }$ </td><td> $T L _ { D }$ </td></tr><tr><td>Merge gate evaluations over  $T$  tokens</td><td> $T$ </td><td> $T$ </td><td>0</td></tr><tr><td>Shared feedback projections, known tokens</td><td> $T$ </td><td>K</td><td>0</td></tr><tr><td>Autoregressive token steps for N tokens</td><td> $N$ </td><td> $N$ </td><td> $N$ </td></tr><tr><td>Blocks per consumed generation token</td><td> $L _ { E } + L _ { D }$ </td><td> $L _ { E } + L _ { D }$ </td><td> $L _ { E } + L _ { D }$ </td></tr><tr><td>Persistent feedback register</td><td> $d { \mathrm { v a l u e s } }$ </td><td>d values</td><td>none</td></tr></table>

Table 1 Execution costs at matched stack dimensions. $T$ counts known input tokens, $K = \lceil T / B \rceil _ { : }$ , and N counts generated tokens; all three variants share the $L _ { E } .$ -stage causal encoder prefill. Counts omit common readout and encoder-memory projection work. RLT-2 reuses the normalized and projected boundary state; gates depend on each token. Backward counts cover the decoder dependency graph, with the encoder’s backward pass common to all variants. Counts describe dependencies and arithmetic; wall-clock speed depends on the implementation and hardware.

Because $h _ { k - 1 }$ depends only on tokens before the chunk, the causal masks keep each y<sub>t</sub> a function of $x _ { 1 : t }$ . A partial final chunk leaves the register at the preceding complete boundary, so moving the prompt–response split does not change the computation. With $B = 1$ , every token closes a chunk, and RLT-2 reduces exactly to RLT-1.

RLT-0 is the $B = \infty$ limit: no chunk boundary is reached, so no decoder output is fed back. RLT-2 with $B = \infty$ would still merge every position with the constant initial state $s _ { \star } ;$ RLT-0 drops this constant merge together with $s _ { \star } ,$ state normalization, the gate, and the feedback projection, and feeds $z _ { t } ^ { 0 } = e _ { t }$ to the decoder (Figure 14).

Table 1 and Figure 5 compare the three intervals at matched stack dimensions. For a known prefix of length T, the decoder needs $\lceil T / B \rceil L _ { D }$ sequential block stages, with up to min(B, T) positions in each batched matrix operation: $T L _ { D }$ stages for RLT-1 and $L _ { D }$ for RLT-0. All three evaluate $T L _ { D }$ decoder blocks and generate one token per step. Larger B therefore trades feedback frequency for known-token parallelism. Appendix I gives the chunked training, prefill, and incremental-generation procedures.

Because RLT-2 uses the same parameters for every $B ,$ the chunk size can also vary during training. Larger chunks shorten the sequential decoder path, while smaller chunks extrapolate better on parity and swaps- ${ \bf \nabla } . { \cal S } _ { 5 }$ (Section 3). In seed-42 CPU timing at $4 + 4 ,$ , chunk4 training steps run 2.27 times as fast as RLT-1 steps, and RLT-0 steps 4.17–4.30 times as fast (Appendix J.10). Pretraining can therefore begin with a large $B ,$ processing most tokens with high parallelism, and mid-training and post-training can reduce $B$ toward $B = 1$ to adapt the model to frequent feedback. Changing B changes the model’s computation, so inference and RL replay use the chunk size of the final training stage. Our experiments train each chunk size from scratch and do not test such a schedule.

## 2.8 Training and state replay

Training uses teacher forcing and cross-entropy on the selected next-token targets. Full backpropagation through time (BPTT) follows recurrent outputs, decoder KV, and encoder memory; masking

$$
b _ { k }
$$

Chunk $I _ { k } \colon$ : parallel positions within each layer

![](images/13e1a81cecf58607cd361a9c58b8636664aecbff1ee690ab79710fe057968ff6.jpg)  
Figure 4 RLT-2 architecture for one chunk. Computation flows upward; the stacks repeat for the indicated depths. Each position $t \in I _ { k }$ uses its own gate to merge its encoder output with the preceding boundary state $h _ { k - 1 }$ . State normalization and projection are shared across the chunk. Each decoder layer applies causal SWA, prefix-restricted encoder-memory attention, and an FFN, with residual connections around each sublayer. For $G = 1$ , all decoder layers read the same projected encoder KV; each layer retains its own SWA KV across chunk boundaries. Every output $y _ { t }$ predicts the next token; only $y _ { k B }$ at a complete boundary becomes the next feedback state. A partial chunk preserves $h _ { k - 1 }$ during continuation. Blue arrows carry encoder features and memory, green arrows carry decoder KV, and orange arrows carry chunk-level feedback.

![](images/484e4145ebe2c39a65a29c9796ead8b62192924066384947e36c226130006830.jpg)  
Within a batch: positions in parallel at each layer. Between batches: preserve causal SWA caches.

Figure 5 Decoder scheduling for eight known tokens as the feedback interval grows. Rows show RLT-1 $( B = 1 )$ , RLT-2 $\left( B = 4 \right)$ , and RLT-0 $( B = \infty )$ . Each shaded group is one layerwise decoder batch, containing $L _ { D }$ sequential blocks. Orange arrows show feedback dependencies; RLT-2 broadcasts $y _ { 4 }$ to all four positions in its second chunk, and RLT-0 has none. Causal SWA applies in every row, with caches retained between batches. During autoregressive generation, all three variants consume tokens one at a time.

a target loss does not skip its state update. For reinforcement learning (RL), evaluating a sampled response after a parameter update requires replaying its history to rebuild the state under the current parameters. Appendix D gives the objectives and replay conditions for pretraining, supervised fine-tuning, and policy gradients.

## 3 Algorithmic Experiments

We study how decoder depth and feedback frequency afect length generalization. The experiments cover addition, parity, modular arithmetic with and without brackets, and standard and swaps-based $S _ { 5 }$ state tracking. We first compare encoder–decoder allocations, then vary the feedback interval at a fixed split. Appendix J.1 gives the complete six-task study at 2,000 steps; Appendices J.9 and J.10 give all feedback variants and the 5,000-step modular-arithmetic comparisons.

## 3.1 Models and evaluation

All models have eight logical layers, width 512, FFN width 1,365, and four attention heads. We compare untied RLT-1 splits $4 + 4 , 5 + 3 , 6 + 2 , 7 + 1$ , and $8 + 0$ with an eight-layer decoder-only Transformer. RLT-1 uses an SWA window of eight, one shared encoder-memory group, and feedback scale $\alpha = 0 . 1$ . The $8 + 0$ variant retains the gated recurrent merge despite having no decoder blocks. Equal layer counts do not match parameters or compute: RLT-1 has 26.10–28.73M parameters, compared with 25.31M for the Transformer (Table 4).

All eight-layer accuracy comparisons use initialization seeds 42, 43, and 44, with a shared training stream and held-out examples within each task. Global batch size is 512 and microbatch size is 32. AdamW uses $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ , weight decay 0.1, and gradient clipping at norm 1. The base learning rate warms up to $1 0 ^ { - 4 }$ over 200 steps, then decays to $5 \times 1 0 ^ { - 6 }$ at the end of each run. Parity, addition, and $S _ { 5 }$ use 2,000 steps; the mod-5 comparisons in this section use 5,000 steps with a cosine schedule spanning that budget. Models use the same width- ${ \cdot \mu } \mathrm { P }$ recipe and four-thread CPU FP32 execution. For RLT-1 and RLT-2, TBPTT128 covers every training sequence, so no gradients are truncated.

Parity trains on 3–40 bits; $S _ { 5 }$ trains on 32 operations, using either all 120 permutations or the identity and ten single transpositions (swaps), following Grazzi et al. (2024). Flat mod-5 expressions ${ \tt u s e } + , -$ , and × with multiplication precedence and odd training lengths 3–39; bracketed expressions use trees of lengths 3–40. We score the final answer or state on 1,024 shared test examples per length. Addition trains on randomly sampled 1–8-digit operands and uses teacher-forced answer-token accuracy on 256 shared pairs per test width. For each run, length generalization uses the checkpoint with minimum in-distribution (ID) validation example loss; exact ties select the earliest step. Test scores do not enter checkpoint selection. Means and sample standard deviations $( \mathrm { S D } , n = 3 )$ describe initialization variability on the fixed data stream. Appendices J and J.6 specify generators, metrics, and checkpoint steps.

## 3.2 Length extrapolation and decoder allocation

RLT-1 sustains parity and $\mathsf { s w a p s } { \mathrm { - } } S _ { 5 }$ accuracy well beyond the training lengths (Figure 6). At 256 bits, parity splits $5 + 3$ and $7 + 1$ retain $1 0 0 \pm 0 \%$ accuracy, compared with $5 0 . 0 7 \pm 1 . 6 3 \%$ for the Transformer. The advantage also appears during learning: at step $5 0 0 , 6 + 2$ reaches $9 9 . 4 4 \pm 0 . 9 8 \%$ validation accuracy versus $4 8 . 4 8 \pm 0 . 5 3 \%$ for the Transformer (Figure 7). In a separate seed-42 sixteen-layer series, RLT-1 $8 + 8 , 9 + 7 , 1 1 + 5$ , and $1 6 + 0$ reach 100% at 256 bits, compared with 49.41% for Transformer 16 (Appendix J.8).

$\mathsf { S w a p s } – S _ { 5 }$ favors a larger decoder allocation. At 256 operations, eight times the training length, $4 + 4$ reaches $9 7 . 3 0 \pm 2 . 7 6 \%$ final-state accuracy, compared with $0 . 8 5 \pm 0 . 3 0 \%$ for the Transformer. At 512 operations, $4 + 4$ still reaches $5 5 . 7 0 \pm 2 5 . 7 8 \%$ , while the Transformer stays at $0 . 8 5 \pm 0 . 3 0 \%$ . Splits $7 + 1$ and $8 + 0$ are already near the uniform reference at 256 operations, despite high training-length accuracy. Figure 13 also reports prefix-token and whole-sequence accuracy.

After 5,000 steps, RLT-1 leads on modular arithmetic at intermediate test lengths. On flat expressions of length 63, RLT-1 6 + 2 reaches $9 3 . 3 6 \pm 5 . 6 9 \%$ , compared with $3 3 . 2 0 \pm 2 . 3 3 \%$ for the Transformer. On bracketed expressions of length $6 4 , 5 + 3$ reaches $6 7 . 9 7 \pm 2 . 9 1 \%$ , compared with $4 6 . 7 1 \pm 1 . 2 1 \%$ . Accuracy falls on longer expressions, and several flat mod-5 splits vary widely across seeds.

## 3.3 Feedback frequency and parallelism

We compare feedback intervals B = 1 (RLT-1), B = 4 (RLT-2 chunk4), and $B = \infty$ (RLT-0) at every split, using the data, optimizer, and seeds of the RLT-1 runs (Section 2.7). RLT-0 has 787,968 fewer parameters than RLT-1 at each split; RLT-2 has the same parameters. Figure 8 compares all variants at $4 + 4 ,$ and Appendix J.9 gives every split.

At 64 bits, RLT-1 reaches $1 0 0 \pm 0 \%$ parity accuracy and chunk4 reaches $9 8 . 9 9 \pm 1 . 6 6 \%$ , while RLT-0 remains near chance at $5 0 . 2 3 \pm 3 . 8 0 \%$ $\mathsf { S w a p s } – S _ { 5 }$ is more sensitive to the feedback interval: at 64 operations, RLT-1 reaches $1 0 0 \pm 0 \%$ and chunk4 $1 9 . 6 0 \pm 6 . 5 4 \%$ . RLT-0 and the Transformer are near the $1 / 1 2 0$ uniform reference at this length. Relative to RLT-1, four-token chunks lose about one percentage point on 64-bit parity and about 80 points on 64-operation swaps-S<sub>5</sub>.

The efect of chunking also depends on the split (Figures 21 and 22). Chunk4 $4 + 4$ retains $6 9 . 3 4 \pm 2 4 . 1 7 \%$ parity accuracy at 256 bits, whereas chunk4 $7 + 1$ and $8 + 0$ are already near chance

$$
\begin{array} { r l r l r l } & { \frac { 1 } { 2 } } & { \mathsf { R L T - 1 ~ } 4 + 4 } & & { \frac { 1 } { 2 } } & { \mathsf { R L T - 1 ~ } 6 + 2 } & & { \frac { 1 } { 2 } } & { \mathsf { R L T - 1 ~ } 8 + 0 } \\ & { \frac { 1 } { 2 } } & { \mathsf { R L T - 1 ~ } 5 + 3 } & & { \frac { 1 } { 2 } } & { \mathsf { R L T - 1 ~ } 7 + 1 } & & { \mathsf { - \frac { 1 } { 2 } - } } & { \mathsf { T r a n s f o r m e r ~ } 8 } \end{array}
$$

![](images/d830f60f42165a976325bbbc86d374993c09221eea5ac1c4fedb7ce0cc141e6c.jpg)

![](images/debf769cb1dbdaa71fb230d7e24bf5911bc0aa2eb8af30bffa0b436737c1c72f.jpg)

Mod-5 without brackets · 5,000 training steps  
![](images/5982e3ae82718393a14fc03c492db2beb9419bf744742cb84f4434a75de9b791.jpg)

![](images/c159da3ca90f28991fe8a7d69a3f837b96c257a41a59e5d877681704d89ffb9b.jpg)  
Seeds 42, 43, 44 · mean ± sample SD · best ID-loss checkpoints · 1,024 shared examples per length

Figure 6 Length generalization across encoder–decoder allocations. All five RLT-1 splits and Transformer 8 are shown across the full evaluated length grids. The top row uses 2,000-step parity and swaps- $S _ { 5 }$ runs; the bottom row uses 5,000-step mod-5 runs for every model. Each run selects its best ID-loss checkpoint, with earliest-step tie breaking. Points and untrimmed error bars show mean ± sample SD over seeds 42, 43, and 44 on 1,024 shared examples per length. Gray regions mark trained lengths; dotted horizontal lines mark uniform-prediction accuracy. Flat mod-5 labels actual odd expression lengths. The complete six-task 2,000-step comparison is in Figure 12.

![](images/dc318f32d78f0200eee288b82aaf3565354c98383f790bb28ec33f4db80c94ca.jpg)  
Figure 7 Parity at 500 and $\mathbf { 2 , 0 0 0 }$ training steps. Circles and whiskers show the mean and sample SD across seeds 42, 43, and 44; crosses show individual seeds with a small horizontal ofset. SD whiskers can extend beyond 100%. Every model uses the same 768 validation examples; the panels correspond to 256,000 and 1,024,000 training examples per seed.

at 64 bits. On swaps- $S _ { 5 } .$ , chunk4 $5 + 3$ reaches $7 9 . 9 2 \pm 1 0 . 8 9 \%$ at 48 operations and $5 0 . 1 6 \pm 1 8 . 7 7 \%$ at $6 4 ,$ compared with $1 9 . 6 0 \pm 6 . 5 4 \%$ at 64 for chunk4 $4 + 4$

## 3.4 Addition, standard $S _ { 5 } ,$ and scope

All models reach 100% teacher-forced accuracy on the training-width addition validation set, but accuracy drops beyond eight digits; the eight-layer model means at 32 digits range from 14.89% to 16.84%. Standard $S _ { 5 }$ remains dificult, with final-state accuracy near the 1/120 reference across feedback variants (Figure 23). Appendix J gives the complete six-task curves, endpoint tables, and individual parity trajectories; Appendix J.11 gives the addition feedback-scale ablation. These experiments measure supervised algorithmic performance; RL performance is not evaluated.

## 4 Related Work

Hybrid Transformer–RNN models. Chen et al. (2018) combine a Transformer encoder with an LSTM-based RNMT+ decoder for machine translation. RLT-1 also combines parallel encoding with recurrent decoding. Its decoder consists of Transformer blocks with encoder-memory cross-attention and SWA, and it processes both prompt and response tokens in a causal language model.

Encoder-derived memory. YOCO builds reusable KV memory for an upper cross-decoder and allows early exit during prefill (Sun et al., 2024). DeepSeek-V4.1-Flash projects global decoder KV from final encoder states and maintains separate decoder SWA (DeepSeek-AI, 2026). RLT-1 also builds global memory from encoder states, but its recurrent decoder must process every prompt token.

Temporal feedback. Feedback Transformer forms a learned weighted sum of each token’s representations across layers and lets subsequent tokens attend to this shared memory at every layer (Fan et al., 2020). RLT-1 feeds the preceding final decoder output into a gated merge at the decoder input, with separate encoder-derived global memory and layerwise decoder SWA caches. Recurrent Transformer constructs each layer’s persistent KV from that layer’s output and provides an exact tiling schedule to improve memory movement (Oncescu et al., 2026). RLT-1’s recurrent dependency spans the full decoder, whereas Recurrent Transformer’s feedback is layerwise.

![](images/4e9cdf423b7586ebdb75b2a8ba8253684332a54a63a59ab6026a7370a551568d.jpg)  
Figure 8 Feedback frequency at a fixed 4 + 4 split. Final-answer/state accuracy for RLT-1 (B = 1), RLT-2 chunk4 (B = 4), RLT-0 $( B = \infty )$ , and Transformer 8, across all native test lengths. Parity and swaps-S use 2,000 training steps; bracketed mod-5 uses 5,000. All models use seeds 42–44 and their best ID-loss checkpoints. Error bars show sample SD without clipping; gray regions mark trained lengths and dotted lines mark uniform prediction.

Cross-token latent feedback. Full-bandwidth Transformer combines the previous top-layer hidden state with the next token embedding through a gated linear unit, retaining the Transformer stack and KV cache (Wang et al., 2026). Its multi-pass training shifts hidden states between passes to allow token-parallel teacher forcing, with a prefix mixin to address diferences between prompts and generation. RLT-1 places output-to-input feedback at the encoder–decoder interface and trains by replaying the decoder sequentially over the full history.

T<sup>2</sup>MLR feeds a cached middle-layer representation from the previous token into an earlier layer at the current position (Cai et al., 2026). Its experiments find that recurrence between middle layers can outperform recurrence through the full network. Training approximates temporal states with a fixed number of Jacobi iterations and controls backward depth separately. RLT-1 feeds back through the entire decoder and trains by replaying the full history with BPTT, optionally truncated.

Latent recurrent language models. Latent Recurrent Transformer (LRT) reuses a high-level state from the previous token through KV projection and residual injection, keeping a decoder-only backbone and one forward pass per generated token (Huang et al., 2026). Its interleaved parallel training refines subsets of positions from a shared state bufer. For RL, LRT initializes this bufer with detached rollout states to reduce diferences between rollout and recomputation. RLT-1 separates encoder memory from the recurrent decoder and rebuilds the full history under current parameters for exact policy replay.

Continuous latent computation. Coconut feeds the last hidden state back as the next input embedding during a latent reasoning phase, using a curriculum that replaces textual reasoning steps with continuous states (Hao et al., 2024). PonderLM-2 inserts latent steps between ordinary tokens during pretraining and uses Jacobi iterations to approximate their recurrent dependencies in parallel (Zeng et al., 2025). RLT-1 advances its state once per ordinary prompt or response token, without adding latent positions.

Block- and segment-level recurrence. Block-Recurrent Transformers update persistent state vectors with attention and gates, processing a block of tokens in parallel at each recurrent step (Hutchins et al., 2022). Their block-feedback variant lets all layers cross-attend to the recurrent state from the preceding block. Recurrent Memory Transformer appends write-memory tokens to each segment and passes their final representations to the next segment as memory inputs, with BPTT through this connection (Bulatov et al., 2022). RLT-2 is the closest variant: it updates feedback at chunk boundaries from the last token’s final decoder output and gates this state into every decoder input of the next chunk, without dedicated memory tokens or a separate bank of recurrent state vectors. RLT-1, the B = 1 case, feeds back at every token and therefore requires sequential decoder work within each block.

Weight sharing across depth. Universal Transformers share weights across depth and optionally adapt the number of refinement steps by position (Dehghani et al., 2018). Saunshi et al. (2025) study how repeated applications of a shared Transformer stack increase efective depth for reasoning, while recurrent-depth language models vary latent computation at inference time (Geiping et al., 2025). DeepLoop analyzes the efect of repeated parameter visits on residual scaling in weight-tied looped Transformers and derives a loop-aware scaling rule for the Post-LN architecture (Li et al., 2026). These models recur over depth at each position; RLT-1 recurs across tokens, and sharing weights between its encoder and decoder is optional.

## 5 Conclusion

We introduced the Recurrent Looped Transformer (RLT), which pairs a parallel causal encoder with a decoder that feeds its final state back at every token, so the computation path grows with sequence length at a fixed per-token cost. With eight layers, RLT generalizes parity and swap-based $S _ { 5 }$ tracking far beyond the training lengths, where an eight-layer Transformer is at chance, and its best splits outperform the Transformer on modular arithmetic beyond the training lengths. Removing the feedback eliminates the gains on parity and swap-based $S _ { 5 }$ . The encoder–decoder split is one design axis: parity generalizes best with one or three decoder layers, whereas swap-based $S _ { 5 }$ needs at least two decoder layers and improves with each additional one up to the four tested. The feedback interval B is a second axis that trades parallelism against accuracy: B = 4 processes known tokens in parallel within each chunk and keeps 64-bit parity at 99%, while swap-based $S _ { 5 }$ benefits most from per-token feedback. Because the chunk size does not change the parameters, a model can be pretrained with large chunks for parallelism and continue with smaller chunks, down to $B = 1$ , in mid- and post-training.

## Acknowledgement

We used large language models to improve the wording of this work.

## References

Aydar Bulatov, Yuri Kuratov, and Mikhail S. Burtsev. Recurrent memory transformer. arXiv preprint arXiv:2207.06881, 2022. URL https://arxiv.org/abs/2207.06881.

Ziyang Cai, Xingyu Zhu, Yihe Dong, Yinghui He, and Sanjeev Arora. T<sup>2</sup>MLR: Transformer with temporal middle-layer recurrence. arXiv preprint arXiv:2607.15178, 2026. URL https://arxiv. org/abs/2607.15178.

Mia Xu Chen, Orhan Firat, Ankur Bapna, Melvin Johnson, Wolfgang Macherey, George Foster, Llion Jones, Niki Parmar, Mike Schuster, Zhifeng Chen, Yonghui Wu, and Macduf Hughes. The best of both worlds: Combining recent advances in neural machine translation. arXiv preprint arXiv:1804.09849, 2018. URL https://arxiv.org/abs/1804.09849.

DeepSeek-AI. DeepSeek-V4.1-Flash: Pushing the limits of KV cache compression. Technical report, DeepSeek-AI, 2026. URL https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/ resolve/main/DeepSeek\_V41\_Tech\_Report.pdf. Sections 2.2 and 3.2.2; publicly available from the oficial DeepSeek model repository.

Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal transformers. arXiv preprint arXiv:1807.03819, 2018. URL https://arxiv.org/abs/1807.03819.

Angela Fan, Thibaut Lavril, Edouard Grave, Armand Joulin, and Sainbayar Sukhbaatar. Addressing some limitations of transformers with feedback memory. arXiv preprint arXiv:2002.09402, 2020. URL https://arxiv.org/abs/2002.09402.

Jonas Geiping, Sean McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. arXiv preprint arXiv:2502.05171, 2025. URL https://arxiv.org/abs/2502.05171.

Riccardo Grazzi, Julien Siems, Arber Zela, Jörg K. H. Franke, Frank Hutter, and Massimiliano Pontil. Unlocking state-tracking in linear rnns through negative eigenvalues. arXiv preprint arXiv:2411.12537, 2024. URL https://arxiv.org/abs/2411.12537.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024. URL https://arxiv.org/abs/2412.06769.

Zeyi Huang, Xuehai He, Liliang Ren, Yiping Wang, Baolin Peng, Hao Cheng, Shuohang Wang, Pengcheng He, Jianfeng Gao, Yong Jae Lee, and Yelong Shen. Latent recurrent transformer: Architecture exploration, training strategies, and scaling behavior. arXiv preprint arXiv:2605.26797, 2026. URL https://arxiv.org/abs/2605.26797.

DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Block-recurrent transformers. arXiv preprint arXiv:2203.07852, 2022. URL https://arxiv.org/abs/2203.07852.

Shuzhen Li, Yifan Zhang, Jiacheng Guo, Quanquan Gu, and Mengdi Wang. DeepLoop: Depth scaling for looped transformers. arXiv preprint arXiv:2607.13491, 2026. URL https://arxiv.org/abs/ 2607.13491.

William Merrill, Jackson Petty, and Ashish Sabharwal. The illusion of state in state-space models. Proceedings of the 41st International Conference on Machine Learning, 2024. URL https://arxiv. org/abs/2404.08819.

Costin-Andrei Oncescu, Depen Morwani, Samy Jelassi, Alexandru Meterez, Mujin Kwun, and Sham

Kakade. The recurrent transformer: Greater efective depth and eficient decoding. arXiv preprint arXiv:2604.21215, 2026. URL https://arxiv.org/abs/2604.21215.

Nikunj Saunshi, Nishanth Dikkala, Zhiyuan Li, Sanjiv Kumar, and Sashank J. Reddi. Reasoning with latent thoughts: On the power of looped transformers. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2502.17416.

Yutao Sun, Li Dong, Yi Zhu, Shaohan Huang, Wenhui Wang, Shuming Ma, Quanlu Zhang, Jianyong Wang, and Furu Wei. You only cache once: Decoder-decoder architectures for language models. arXiv preprint arXiv:2405.05254, 2024. URL https://arxiv.org/abs/2405.05254.

Xi Wang, Ziyang Cai, Zheng Zhan, Harry Dong, Ying Fan, Gustavo de Rosa, Tim Pearce, and John Langford. Full-bandwidth transformer. arXiv preprint arXiv:2608.08888, 2026. URL https: //arxiv.org/abs/2608.08888.

Boyi Zeng, He Li, Shixiang Song, Yixuan Wang, Ziwei He, Xinbing Wang, and Zhouhan Lin. PonderLM-2: Pretraining LLM with latent thoughts in continuous space. arXiv preprint arXiv:2509.23184, 2025. URL https://arxiv.org/abs/2509.23184.

Yifan Zhang et al. Reliable RL scaling requires accounting for Prefill–Decode kernel mismatch. Technical report, Pretraining-RL-Science project, August 2026. URL https://github.com/yifanzhang-pro/Pretraining-RL-Science/blob/master/Prefill\_ Decode\_Kernel\_Mismatch.pdf. Dated August 6, 2026; revised August 24, 2026.

## Appendix

A Architecture and Execution Details 18   
A.1 Sharing encoder and decoder weights . 18   
B Computational Properties 18   
B.1 Prompt–response consistency 18   
B.2 Prefill work and sequential depth 18   
C Hardware Execution 19   
C.1 Batching independent sequences 19   
C.2 Memory trafic and parameter reuse 19   
C.3 Training memory and numerical agreement 20   
D Training Objectives 20   
D.1 Autoregressive pretraining 20   
D.2 Supervised fine-tuning . . 20   
D.3 Policy gradients and replay 20   
E Multi-turn Inference and Cache Reuse 21   
F Execution Procedures 23   
F.1 Recurrent prompt prefill 23   
F.2 Generation and new external inputs 23   
F.3 Pretraining and SFT 23   
F.4 Current-policy RL replay 24   
G Causality and Gradient Paths 24   
H When Cached States Can Be Reused 25   
RLT-2: Chunk-Parallel Hidden-State Feedback 26   
I.1 Chunk operator and causal masks 26   
I.2 Training and prefill algorithm 26   
I.3 Incremental generation and partial chunks 27   
I.4 Training and inference eficiency 28   
Experimental Details 28   
J.1 Complete six-task study at 2,000 steps 28   
J.2 Parameter counts 35   
J.3 Implementation of RLT-0 . 35   
J.4 Seed aggregation and metric definitions 35   
J.5 Reproducing the completed training snapshot 35   
J.6 Length-generalization protocol and supplementary metrics 38   
J.7 Token-accuracy trajectories at lengths 32 and 33 42   
J.8 Sixteen-layer parity: best and final checkpoints 43   
J.9 Three-seed comparison of feedback variants 45   
J.10 Mod-5 feedback variants at 5,000 steps . 50   
J.11 Addition feedback scale at a fixed learning-rate schedule 55

## A Architecture and Execution Details

## A.1 Sharing encoder and decoder weights

The reference tied RLT-1 sets $L _ { E } = L _ { D } = L$ . Encoder self-attention at layer ℓ and decoder SWA at layer ℓ share compatible query, key, value, and output projections; the corresponding FFNs also share weights. The encoder attends to its causal context; decoder SWA attends to decoder activations within its window. Decoder cross-attention uses separate query/output projections and the memory projections of Equation (2.2), adding computation beyond the shared attention and FFN. The encoder and decoder retain separate normalizations, and the merge and readout are separate modules.

The two passes share weights but compute separate activations. An untied $E _ { \theta } , D _ { \phi }$ uses separate weights and keeps the same recurrence. Memory groups can be shared in either version; each decoder layer still maintains its own SWA cache.

## B Computational Properties

## B.1 Prompt–response consistency

Proposition B.1 (Invariance to the serving split). Fix the parameters, token sequence, position convention, and initial state for an independent sequence. Assume exact arithmetic, mathematically equivalent causal encoder execution, identical SWA windows and cache updates, and deterministic decoder operations. Processing a prefix with batched encoder prefill followed by recurrent decoder updates gives the same states and next-token distributions as processing it incrementally. The conditional distribution for a fixed token history is therefore unchanged when the prompt–response split moves.

Proof. Causal encoder equivalence gives the same $e _ { t }$ and $M _ { \leq t }$ in both schedules. Both start from $H _ { 0 } = ( s _ { \star } , \emptyset )$ . If their states agree at $t - 1$ , Equation (2.5) applies the same operations to the same inputs at $t ,$ so their states agree at t. The result follows by induction, since the transition does not depend on the serving split. □

## B.2 Prefill work and sequential depth

Let $C _ { E } ^ { \mathrm { p f } } ( T )$ be encoder prefill work, $C _ { M } ( { \cal T } )$ the memory projection work, and $C _ { D } ^ { \mathrm { s t e p } } ( t )$ a decoder evaluation over t encoder-memory entries and at most W decoder positions per layer, including merge overhead. Then

$$
C _ { \mathrm { p r e f l l } } ( T ) = C _ { E } ^ { \mathrm { p f } } ( T ) + C _ { M } ( T ) + \sum _ { t = 1 } ^ { T } C _ { D } ^ { \mathrm { s t e p } } ( t ) .\tag{B.1}
$$

For dense attention and width-proportional KV, a coarse arithmetic estimate is

$$
O \big ( ( L _ { E } + L _ { D } ) ( T d ^ { 2 } + T ^ { 2 } d ) + G T d ^ { 2 } + L _ { D } T \operatorname * { m i n } ( W , T ) d \big ) .\tag{B.2}
$$

Prefill requires T sequential decoder transitions, each containing $L _ { D }$ blocks. These sequential dependencies can reduce hardware utilization even when arithmetic complexity is of the same order as a dense Transformer’s.

Each new token evaluates $L _ { E } + L _ { D }$ blocks, plus merge and memory projection. Attention work still grows with context length. With efective KV width $d _ { \mathrm { K V } }$ , inference cache storage is approximately

$$
O \big ( ( L _ { E } + G ) t d _ { \mathrm { K V } } + L _ { D } \operatorname* { m i n } ( t , W - 1 ) d _ { \mathrm { K V } } ^ { D } + d \big ) .\tag{B.3}
$$

Here $d _ { \mathrm { K V } } ^ { D }$ is the efective decoder SWA KV width, and the $O ( d )$ term stores the current recurrent output. The SWA term counts retained history; current-position KV and training activations need additional storage. Weight sharing reduces parameter storage but retains both passes and their caches.

## C Hardware Execution

## C.1 Batching independent sequences

For a known training sequence or prompt, the encoder computes features and memory projections in parallel across positions, then the decoder processes positions in order. Independent sequences can batch their next decoder updates together (Figure 9). Each sequence keeps its own encoder prefix, SWA cache, recurrent output, and position.

At inference, each step consists of an encoder update, memory append, merge, decoder pass, and readout. Batching requests increases matrix-operation sizes and allows weight reuse. Small batches and uneven sequence lengths may limit utilization.

Batching decoder updates from independent sequences  
![](images/39f9759922c195e4db0aed2e5e2b6a517fd169491890fd3bcd71692bfd8f5fad.jpg)  
Columns: independent updates can share one batched kernel.  
Rows: updates follow token order within each sequence.

Figure 9 Batching decoder updates across sequences. Columns group independent updates into a batch;   
rows follow each sequence in token order. Each update reads its own encoder prefix and decoder SWA window.   
Known tokens can be encoded in parallel before replay; generated tokens are encoded as they arrive.

## C.2 Memory trafic and parameter reuse

Encoder KV stays fixed while the prefix and parameters are unchanged. Decoder layers reuse that memory and append their own KV to separate SWA caches. Fewer memory groups reduce KV storage and projection work; more groups allow diferent transformations for each layer. Sharing encoder and decoder weights may also help keep weights resident on the accelerator.

Normalization, gating, state projection, and residual addition can be fused into one kernel.

## C.3 Training memory and numerical agreement

Activation checkpointing reduces stored activations by recomputing them during backpropagation through time (BPTT).

Zhang et al. (2026) describe how the executed policy depends on precision, cache construction, reductions, and sampling transforms as well as weights. Sampler and trainer implementations must align positional conventions, stochastic behavior, and cache contents, and check numerical agreement across execution modes.

## D Training Objectives

## D.1 Autoregressive pretraining

Full-sequence next-token prediction is the base objective:

$$
\mathcal { L } _ { \mathrm { P T } } ( \Theta ) = - \mathbb { E } _ { x _ { 1 : S } } \left[ \frac { 1 } { S - 1 } \sum _ { t = 1 } ^ { S - 1 } \log p _ { \Theta } ( x _ { t + 1 } \mid x _ { 1 : t } ) \right] .\tag{D.1}
$$

Initialize $H _ { 0 } = ( s _ { \star } , \emptyset )$ , compute causal encoder features, and update the decoder at positions $1 , \ldots , S - 1$ . Every non-BOS target contributes to the loss; the encoder, memory projections, merge, and decoder train jointly. At each independent document, reset both decoder and encoder state, reset positions, and prevent attention across document boundaries. When training on segments, specify the initial context and state: dropping an earlier state changes the history on which the likelihood is conditioned.

## D.2 Supervised fine-tuning

Let $m _ { t + 1 } = 1$ for assistant targets and 0 for user, system, tool, or padding targets. For examples with at least one selected target, optimize

$$
\mathcal { L } _ { \mathrm { S F T } } ( \Theta ) = - \mathbb { E } _ { x } \left[ \frac { 1 } { \sum _ { t = 1 } ^ { S - 1 } m _ { t + 1 } } \sum _ { \stackrel { 1 \leq t < S } { m _ { t + 1 } = 1 } } \log p _ { \Theta } ( x _ { t + 1 } \mid x _ { 1 : t } ) \right] .\tag{D.2}
$$

The mask selects loss terms; user, system, and tool tokens still update the decoder state. Gradients from assistant losses flow through these updates, including encoder memory and decoder KV. The model continues from the existing state when an assistant turn begins.

## D.3 Policy gradients and replay

Let $c = x _ { 1 : T }$ be a prompt and $y = ( y _ { 1 } , \dotsc , y _ { N } )$ a sampled response, with $y _ { i } = x _ { T + i }$ . The policy is

$$
\pi _ { \Theta } ( y \mid c ) = \prod _ { i = 1 } ^ { N } p _ { \Theta } ( y _ { i } \mid c , y _ { < i } ) .\tag{D.3}
$$

For a sequence reward $R ( c , y )$ independent of $\Theta _ { s }$ , define $J ( \Theta ) = \mathbb { E } _ { c , y \sim \pi _ { \Theta } } [ R ( c , y ) ]$ . Its on-policy score-function gradient is

$$
\nabla _ { \Theta } J = \mathbb { E } _ { c , y \sim \pi _ { \Theta } } \left[ \left( R ( c , y ) - b ( c ) \right) \sum _ { i = 1 } ^ { N } \nabla _ { \Theta } \log p _ { \Theta } ( y _ { i } \mid c , y _ { < i } ) \right] ,\tag{D.4}
$$

where $b ( c )$ is a response-independent baseline, held constant when taking the policy gradient. During replay, the sampled tokens are fixed and gradients pass through the recurrent computation used to evaluate their log-probabilities.

Figure 10 shows the replay procedure. The sampler records each action’s log-probability under the distribution $\mu$ that sampled it. The trainer rebuilds encoder features, recurrent outputs, and decoder SWA caches under the current parameters $\Theta ,$ , starting from the initial state and processing the full prompt and sampled response prefix. The resulting action probabilities give the ratios

$$
r _ { i } ( \Theta ) = \exp \left( \log p _ { \Theta } ( y _ { i } \mid c , y _ { < i } ) - \log \mu ( y _ { i } \mid c , y _ { < i } ) \right) .\tag{D.5}
$$

Here the target is the raw model policy $p _ { \Theta }$ . The behavior probabilities must include any temperature scaling, truncation, and renormalization used by the sampler. Exact importance sampling requires $p _ { \Theta } ( \cdot \mid h ) \ll \mu ( \cdot \mid h )$ : at each relevant history, every action with positive target probability must also have positive behavior probability. Top-k or top-p sampling generally excludes actions with positive probability under an untruncated softmax target. If the target policy itself uses a sampling transform, apply it consistently in the objective and ratios.

To preserve both forward probabilities and their parameter derivatives, the trainer diferentiates the recurrent computation or an equivalent implementation (Zhang et al., 2026). The recorded behavior log-probabilities stay fixed across parameter updates, since they describe the policy that sampled the actions.

For fixed prompts and sequence rewards, exact trajectory importance sampling uses $\Pi _ { i } r _ { i ; }$ , subject to support and integrability conditions. Tokenwise clipping and other surrogate losses may introduce bias even with correct replay probabilities.

## E Multi-turn Inference and Cache Reuse

A conversation forms one token history beginning at BOS. User messages, tool results, role delimiters, and assistant tokens all receive encoder and decoder updates. For example, when a tool returns, encode its output using the cached prefix, then process those tokens through the decoder in order before resuming generation. Reset state only for a new independent sequence or an explicitly defined context reset.

To resume from a saved prefix, retain the encoder cache $C _ { t } ^ { E }$ , cross-attention memory $M _ { \leq t _ { : } }$ decoder state $H _ { t } = ( s _ { t } , C _ { t } ^ { D } )$ , tokens and positions, SWA window convention, and model version. These values can be reused at fixed weights regardless of where the prompt ends. Reproducing the same sampled outputs also requires the sampler’s random state. If only text is saved, replay it to rebuild the state.

Editing or truncating an earlier token invalidates the later cached state. Recompute from a valid checkpoint before the edit; this also applies to chat templates that rewrite earlier tokens. In multi-turn $\mathrm { R L }$ , user and tool tokens update the state and carry gradients, but receive no action importance-ratio factors because the policy did not sample them.

Replaying a sampled history under current parameters  
![](images/0b486f2b6b49ccd95de5131b0944329673e7756f557a9bdbee040412cb50e0b9.jpg)  
After every parameter update: rebuild current-policy KV and states; retain original behavior log-probabilities.  
Figure 10 Replaying a sampled history under current parameters. The sampler records behavior probabil ities. The trainer rebuilds the full prefix, including recurrent outputs and decoder SWA caches, to evaluate the current policy. The ratio compares the current action probability with the recorded behavior probability.

## F Execution Procedures

The complete decoder state is $H _ { t } = ( s _ { t } , C _ { t } ^ { D } )$ , initialized as $H _ { 0 } = ( s _ { \star } , \emptyset )$ . Encoder continuation state and encoder-derived cross-attention memory are maintained separately. All schedules use the block order and window convention in Equation (2.16).

## F.1 Recurrent prompt prefill

1. Start an independent sequence with $x _ { 1 } = \mathrm { B O S }$ and $H _ { 0 } = ( s _ { \star } , \emptyset )$ ; initialize encoder caches and positions.

2. Encode $x _ { 1 : T }$ causally in parallel and construct its encoder KV memory.

3. For $t = 1 , \dots , T$ , compute $H _ { t } = D _ { \phi } ( \mathrm { M e r g e } ( e _ { t } , s _ { t - 1 } ) ; M _ { \leq t } , C _ { t - 1 } ^ { D } , t )$ . Each decoder layer reads current KV and the retained SWA history. Both attention masks exclude positions after t.

4. Predict the first response token from $s _ { T }$ . Preserve $H _ { T }$ , encoder caches, memory, and positional metadata for the next update.

## F.2 Generation and new external inputs

Sample $x _ { t + 1 }$ from the distribution predicted by $s _ { t } ,$ , then update the encoder cache, memory, and complete decoder state to $H _ { t + 1 }$ . This state predicts $x _ { t + 2 }$ . Each consumed token receives exactly one recurrent update and one KV insertion at every decoder SWA layer.

Multiple tokens from a user or tool can be encoded in a causal batch conditioned on the existing prefix. The decoder processes them in token order from the saved state, updating both $s _ { t }$ and $C _ { t } ^ { D }$

If generation stops at a length limit before consuming its last emitted token, record that token separately. The cache represents the consumed prefix; process the pending token once before continuing with later tokens.

## F.3 Pretraining and SFT

1. Compute $e _ { 1 : S - 1 }$ using a causal encoder and construct memory.

2. Initialize $H _ { 0 } = ( s _ { \star } , \emptyset )$ ; unroll all positions $1 , \ldots , S - 1$ , including every layerwise SWA cache update.

3. Accumulate all valid next-token losses for pretraining, or only assistant-target losses for SFT. All context tokens receive state updates.

4. Normalize each example by its number of selected targets, then average examples as in Equations (D.1) and (D.2). Backpropagate through the complete computation, using checkpointing if required.

Examples without selected targets are excluded from the loss average. Token averaging would give more weight to examples with more selected targets.

Packed independent sequences need separate recurrent states, SWA caches, encoder caches, positions, and attention masks. Both attention mechanisms must stay within document boundaries, even where losses are masked.

## F.4 Current-policy RL replay

1. Read the rollout token history, action mask, actual behavior log-probabilities, and sampling metadata. Behavior probabilities include all sampling transforms and renormalization.

2. Hold current parameter values fixed throughout forward replay and backward. Use the target policy’s positional, SWA, and execution conventions, with parameter gradients enabled.

3. Recompute encoder representations of the known history. From $H _ { 0 } = ( s _ { \star } , \emptyset )$ , rebuild the recurrent output and every decoder SWA cache through all prompt tokens.

4. Replay subsequent tokens in order. Before consuming each sampled action, read its current-policy log-probability from the preceding state. Include EOS if sampled. Consume every intervening external token needed for later predictions.

5. Form the chosen RL loss from action log-probabilities, rewards or advantages, and any required behavior ratios. Apply action factors only to tokens sampled by the policy.

6. Backpropagate, update parameters, and invalidate parameter-dependent encoder and decoder caches before the next exact current-policy replay.

The action mask selects loss terms while preserving gradients through user and tool tokens. Prompt replay under no\_grad, or detaching prompt KV, preserves forward values but drops the prompt-state gradients required by full BPTT.

A deterministic forward pass, for example with dropout disabled, gives a reproducible probability for each action. For a stochastic policy, specify whether action probabilities condition on internal randomness or average over it. Each replay evaluates one realization; marginal probabilities require averaging over that randomness.

## G Causality and Gradient Paths

Proposition G.1 (Causality). With a causal encoder, prefix-restricted encoder memory, and causal decoder SWA, $H _ { t } = ( s _ { t } , C _ { t } ^ { D } )$ depends only on $x _ { 1 : t }$ and the parameters, including $s _ { \star }$ , under deterministic execution.

Proof. Encoder causality implies that each $e _ { j }$ depends only on $x _ { 1 : j }$ , hence $M _ { \leq t }$ depends only on $x _ { 1 : t }$ . The initial state contains no future-token information. Suppose $H _ { t }$ <sub>−1</sub> depends only on $x _ { 1 : t - 1 }$ The next update uses this state, $e _ { t } ,$ and $M _ { < t }$ . At each decoder layer, current KV is formed from the causally available layer input, and SWA reads no position greater than t. Appending current KV and evicting old entries introduce no future information. Thus both $s _ { t }$ and $C _ { t } ^ { D }$ depend only on $x _ { 1 : t } ,$ completing the induction. □

For teacher-forced encoder features held fixed, use a fixed-slot representation of the decoder cache, with validity masks during warm-up, and define

$$
\mathcal { T } _ { t } = \frac { \partial H _ { t } } { \partial H _ { t - 1 } } .\tag{G.1}
$$

Then

$$
\frac { \partial H _ { t } } { \partial H _ { j } } = \mathcal { J } _ { t } \mathcal { J } _ { t - 1 } \cdot \cdot \cdot \mathcal { J } _ { j + 1 } , \qquad j < t ,\tag{G.2}
$$

where

$$
\mathcal { T } _ { t } = \left( \begin{array} { c c } { \displaystyle \frac { \partial s _ { t } } { \partial s _ { t - 1 } } } & { \displaystyle \frac { \partial s _ { t } } { \partial C _ { t - 1 } ^ { D } } } \\ { \displaystyle \frac { \partial C _ { t } ^ { D } } { \partial s _ { t - 1 } } } & { \displaystyle \frac { \partial C _ { t } ^ { D } } { \partial C _ { t - 1 } ^ { D } } } \end{array} \right) .\tag{G.3}
$$

The block Jacobian includes paths through both the recurrent output and decoder KV.

Encoder memory retains information from past encoder outputs; decoder SWA provides direct access to recent decoder projections. Evicted decoder entries may still afect later computation through states or activations that read them before eviction.

Full parameter derivatives also include the parameter dependence of encoder features, memory, every transition, and decoder KV projections. Let $B _ { T } ^ { E }$ denote all encoder-side boundary tensors needed for response continuation, including encoder continuation KV and encoder-derived cross-attention memory. For a response loss $\ell ( \Theta , H _ { T } ( \Theta ) , B _ { T } ^ { E } ( \Theta ) )$ , the chain rule gives

$$
\frac { d \ell } { d \Theta } = \frac { \partial \ell } { \partial \Theta } + \frac { \partial \ell } { \partial H _ { T } } \frac { \partial H _ { T } } { \partial \Theta } + \frac { \partial \ell } { \partial B _ { T } ^ { E } } \frac { \partial B _ { T } ^ { E } } { \partial \Theta } .\tag{G.4}
$$

The first term holds the boundary arguments fixed; the others account for their prefix computation. In particular,

$$
\frac { \partial \ell } { \partial H _ { T } } \frac { \partial H _ { T } } { \partial \boldsymbol { \Theta } } = \frac { \partial \ell } { \partial s _ { T } } \frac { \partial s _ { T } } { \partial \boldsymbol { \Theta } } + \frac { \partial \ell } { \partial C _ { T } ^ { D } } \frac { \partial C _ { T } ^ { D } } { \partial \boldsymbol { \Theta } } .\tag{G.5}
$$

Detaching $s _ { T }$ , decoder KV, or encoder boundary tensors removes the corresponding gradient terms, even when forward probabilities stay unchanged.

## H When Cached States Can Be Reused

A saved prefix can replace forward replay when it matches the current parameters, consumed tokens, positions, initialization, SWA window, and policy execution settings. Save the recurrent output, each decoder SWA cache, encoder continuation state, encoder memory, and associated metadata. The saved prefix may come from the sequence beginning or from an exact checkpoint made with the current parameters and execution settings.

A detached cache can hold correct values while lacking the computation graph needed for training. Full BPTT requires retaining or recomputing the prefix graph so gradients reach every parameter-dependent boundary tensor.

Activation checkpointing within one update recomputes activations under the same parameters and stochastic state. Mutable cache implementations must restore the saved contents during recomputation and preserve activations needed for backward.

Truncated BPTT preserves numeric values while cutting selected gradient dependencies. A complete decoder-state detach is

$$
\widetilde { H } _ { t } = ( \mathrm { s t o p g r a d } ( s _ { t } ) , \mathrm { s t o p g r a d } ( C _ { t } ^ { D } ) ) .\tag{H.1}
$$

Detaching only $s _ { t }$ leaves possible paths through decoder $\mathrm { K V } ;$ detaching only decoder KV leaves paths through the recurrent output. Encoder-side boundary tensors can provide additional paths across the same boundary.

SWA eviction removes old KV from future attention windows. Full BPTT still diferentiates earlier computations that used those entries, so training may need more activation storage than the inference cache alone.

After a weight update, recompute the prefix values and gradient graph before the next exact training pass.

## I RLT-2: Chunk-Parallel Hidden-State Feedback

Section 2.7 defines the RLT-2 merge and boundary update, and Figure 4 shows one chunk; this appendix gives its masks, training and prefill procedure, incremental generation, and costs. Section 3 and Appendix J.9 report the experiments.

## I.1 Chunk operator and causal masks

Choose a fixed chunk size $B \geq 1$ , count BOS as position one, and anchor boundaries to the start of each independent sequence. For a known prefix of length $T _ { i }$ , define

$$
K = \lceil T / B \rceil , \qquad a _ { k } = ( k - 1 ) B + 1 , \qquad b _ { k } = \operatorname* { m i n } ( k B , T ) , \qquad I _ { k } = \{ a _ { k } , \dotsc , b _ { k } \} .\tag{I.1}
$$

Starting from empty decoder caches, the merged inputs $z _ { I _ { k } } ^ { 0 }$ of Equation (2.18) pass through the chunk operator and readout:

$$
y _ { I _ { k } } , C _ { b _ { k } } ^ { D } = D _ { \phi } ^ { \mathrm { c h u n k } } ( z _ { I _ { k } } ^ { 0 } ; M , C _ { a _ { k } - 1 } ^ { D } , I _ { k } ) ,\tag{I.2}
$$

$$
p \Theta \big ( x _ { t + 1 } \mid x _ { 1 : t } \big ) = \mathrm { s o f t m a x } \big ( W _ { o } \mathrm { R M S N o r m } _ { o } ( y _ { t } ) \big ) _ { x _ { t + 1 } } , \qquad t \in I _ { k } .\tag{I.3}
$$

The chunk operator applies Equation (2.16) layer by layer, using all known positions in $I _ { k }$ together. At each layer, query t reads decoder keys at max $\dot { \mathopen { } \mathclose \bgroup \left( 1 , t - W + 1 \aftergroup \egroup \right) } \leq j \leq t ,$ including retained keys from earlier chunks, and encoder memory only at $j \leq t$ . Neither SWA nor encoder positions reset at a chunk boundary. The chunk may be longer or shorter than the SWA window.

Each decoder layer projects KV for all positions from that layer’s inputs before masked attention, so causal SWA can process those positions in parallel. The last-position output carries the chunk’s causal computation to the next chunk, without pooling or an additional summary token. A sequence fitting in one chunk still receives the learned initial-state merge, whereas RLT-0 removes that merge and its parameters.

## I.2 Training and prefill algorithm

For teacher-forced training, $x _ { 1 : T }$ denotes the consumed input tokens, with a next-token target wherever one is available. The same forward procedure computes prompt prefill and known-history policy replay:

1. Encode the known tokens causally and build encoder KV memory. Initialize $h = s _ { \star }$ , empty decoder SWA caches, and global position one.

2. Visit chunks in increasing order. Compute $\mathrm { R M S N o r m } _ { s } ( h )$ and $W _ { s } \mathrm { R M S N o r m } _ { s } ( h )$ once for the chunk; broadcast them to its token-specific gated merges.

3. For each decoder layer, project the chunk’s Q/K/V together, apply causal SWA using retained and current-chunk KV, then apply prefix-masked memory attention and the FFN. Retain the last $W - 1$ decoder KV entries per layer for continuation.

4. Compute logits and the selected next-token losses at every position. After a complete chunk, set h to its last decoder output. A partial final chunk leaves h at the preceding complete boundary.

5. For training, accumulate the sequence loss and backpropagate through all chunks before updating parameters. For prefill, save caches, h, the consumed-token count, and the last output or nexttoken logits.

Full BPTT diferentiates through the boundary state, cross-chunk decoder KV, and encoder memory. Chunking changes the forward dependencies; gradient truncation and optimizer-update frequency are separate choices. For RL, replay uses the same B, boundary anchor, masks, and positions as sampling, rebuilding the state under the current parameters before evaluating action probabilities.

## I.3 Incremental generation and partial chunks

Let t be the number of consumed tokens and $r = t$ mod B the current chunk ofset. The saved feedback register is $h _ { \left\lfloor t / B \right\rfloor }$ , including when the prompt ends inside a chunk. To consume the next observed or sampled token at position $j = t + 1$

1. Update the causal encoder and append its memory KV. Merge $e _ { j }$ with the saved boundary state, keeping that state fixed throughout the chunk.

2. Run one decoder step with the usual causal SWA caches and prefix memory, producing $y _ { j }$ , updated KV caches, and next-token logits.

3. If j mod $B = 0 _ { i }$ , replace the feedback register by $y _ { j } { \mathrm { ; } }$ otherwise preserve it. Advance the consumedtoken count to $j$

The latest $y _ { j }$ always predicts the next token, whether or not it updates the feedback register. For example, with $B = 4$ and a six-token prompt, tokens 5–8 all merge with $h _ { 1 } = y _ { 4 } ;$ processing tokens 5 and 6 during prefill does not replace that register with $y _ { 6 }$ . The output $y _ { 6 }$ predicts token $^ { 7 , }$ and $y _ { 8 }$ becomes $h _ { 2 }$ only after token 8 is consumed. Known continuation tokens can be batched up to the next fixed boundary, then the procedure continues with the new state. Padding, message boundaries, and the prompt–response split do not commit a partial chunk. Independent packed sequences each maintain their own boundary anchor and state.

With fixed weights and exact arithmetic, batched and incremental execution agree if they use the same mathematical operators and stochastic behavior. Stochastic operations must be disabled or use identical random draws for corresponding operations. Each position then has the same encoder prefix, preceding boundary state, and causally available layerwise KV in both schedules. Induction over layers within a chunk and then over chunks gives the same outputs and boundary states. Changing B or shifting the boundary anchor changes the model computation and invalidates a saved continuation state.

Chunkwise prefill and tokenwise decoding can use diferent matrix and attention kernels and floating-point reduction orders, altering logits and boundary states even under deterministic execution. For RL, compare chunkwise replay against a tokenwise reference that reproduces the rollout’s encoder, decoder, and readout execution, including precision, batch layout, kernel choices, and stochastic settings.

## I.4 Training and inference eficiency

Table 1 assumes equal encoder and decoder depths, width, windows, and memory groups, with $L _ { D } \geq 1$ . Its decoder block stages count sequential dependencies in the layerwise schedules, excluding attention-kernel reductions, communication, and hardware-dependent latency.

At fixed dimensions, the common forward arithmetic with dense encoder and memory attention is

$$
O \big ( ( L _ { E } + L _ { D } ) T d ^ { 2 } + ( L _ { E } + L _ { D } ) T ^ { 2 } d + G T d ^ { 2 } + L _ { D } T \operatorname * { m i n } ( W , T ) d \big ) .\tag{I.4}
$$

RLT-1 adds $O ( T d ^ { 2 } )$ merge work. RLT-2 also adds $O ( T d ^ { 2 } )$ merge work for $T$ token-specific gates, but reuses state normalization and projection within each chunk. Its $K$ sequential decoder batches permit larger matrix operations and weight reuse across up to $B$ positions. Total decoder arithmetic still covers all $T$ tokens, and full BPTT requires activations or recomputation across all chunks.

During generation at context length $t ,$ each variant evaluates its encoder and decoder stacks per token and reads the available encoder memory. RLT-2 can reuse its feedback projection until the next boundary; the token-specific gate, decoder blocks, and KV writes remain per-token operations. The shared inference-cache scaling is

$$
O \big ( ( L _ { E } + G ) t d _ { \mathrm { K V } } + L _ { D } \operatorname* { m i n } ( t , W - 1 ) d _ { \mathrm { K V } } ^ { D } \big ) .\tag{I.5}
$$

RLT-1 and RLT-2 add $O ( d )$ feedback storage; RLT-2 also retains a chunk ofset and may cache $O ( d )$ normalized/projected state. Chunk batches require temporary activations and current-chunk KV during prefill.

A controlled benchmark would sweep $B \in \{ 1 , 2 , 4 , 8 , 1 6 , 3 2 \}$ and RLT-0 at fixed data, stack dimensions, precision, optimizer, batch size, and attention kernels. It would measure task accuracy, training throughput, prefill latency by prompt length, generation time per token by cached context length, and peak device memory.

The $B = 1$ run would check RLT-1 equivalence, and future-token perturbations would check causality. At unchanged parameters, maximum and RMS diferences in logits, selected-action logprobabilities, and boundary states across chunkwise, partial-chunk, and tokenwise execution would separate execution diferences from the efects of a policy update. A diferentiable reference should also compare full-BPTT parameter gradients, including paths through boundary states, decoder KV, and encoder memory.

## J Experimental Details

## J.1 Complete six-task study at 2,000 steps

This appendix reports all 108 runs of the eight-layer study: six architectures on six tasks with initialization seeds 42, 43, and 44, each trained for 2,000 optimizer steps. It gives validation curves at training lengths and length generalization from each run’s best in-distribution checkpoint.

## J.1.1 Models, tasks, and evaluation protocol

Models, optimizer settings, batch sizes, and execution follow Section ${ 3 ; }$ the cosine schedule reaches $5 \times 1 0 ^ { - 6 }$ at step 2,000. Validation runs every 100 steps, and each run sees 1,024,000 training examples.

Table 2 Validation accuracy (%) after 2,000 optimizer steps. All entries are mean ± sample SD over seeds 42, 43, and 44. Addition uses teacher-forced answer tokens; the formal tasks use final labels or states. Every run has consumed 1,024,000 training examples.
<table><tr><td>Task</td><td> $\mathrm { R L T - 1 } ~ 4 \substack { + 4 }$ </td><td> $\mathrm { R L T - 1 } ~ 5 + 3$ </td><td> $\mathrm { R L T } \mathrm { - } 1 6 { + } 2$ </td><td> $\mathrm { R L T - 1 } ~ 7 + 1$ </td><td> $\mathrm { R L T - 1 } ~ 8 \substack { + 0 }$ </td><td>GPT 8</td></tr><tr><td>Addition</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Parity</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 8 . 8 3 \pm 1 . 9 2$ </td><td> $9 4 . 8 4 \pm 3 . 4 3$ </td></tr><tr><td>Mod 5, no brackets</td><td> $4 5 . 3 6 \pm 4 6 . 4 3$ </td><td> $9 4 . 1 8 \pm 7 . 2 9$ </td><td> $6 9 . 6 2 \pm 3 3 . 0 1$ </td><td> $7 0 . 0 1 \pm 4 3 . 1 5$ </td><td> $6 0 . 3 3 \pm 3 5 . 7 0$ </td><td> $6 4 . 0 2 \pm 3 7 . 6 4$ </td></tr><tr><td>Mod 5, brackets</td><td> $7 0 . 5 3 \pm 1 0 . 7 5$ </td><td> $7 4 . 3 5 \pm 6 . 7 1$ </td><td> $7 4 . 8 7 \pm 3 . 7 2$ </td><td> $7 4 . 7 8 \pm 3 . 0 1$ </td><td> $7 5 . 1 7 \pm 6 . 7 8$ </td><td> $7 3 . 8 7 \pm 9 . 0 7$ </td></tr><tr><td> $S _ { 5 } ,$  swaps</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 3 5 \pm 0 . 8 1$ </td><td> $9 9 . 6 1 \pm 0 . 6 8$ </td><td> $9 9 . 0 9 \pm 0 . 2 3$ </td></tr><tr><td> $S _ { 5 } ,$  standard</td><td> $0 . 7 8 \pm \ : 0 . 6 8$ </td><td> $2 . 4 7 \pm 1 . 2 6$ </td><td> $2 . 2 1 \pm 1 . 4 8$ </td><td> $2 . 0 8 \pm 0 . 9 8$ </td><td> $1 . 8 2 \pm 1 . 1 9$ </td><td> $0 . 5 2 \pm 0 . 2 3$ </td></tr></table>

• Addition: randomly sample 1–8-digit operands and serialize their digits in reverse order. The 256 validation examples use the same range. We measure teacher-forced answer-token accuracy: each prediction receives the correct preceding tokens, and scoring includes answer formatting and EOS while excluding prompt and padding positions.

• Parity: train on binary strings of lengths 3–40 and supervise the final parity bit. Validation pools 256 examples at each of lengths 32, 36, and 40.

• Modular arithmetic: evaluate expressions modulo five using +, −, and ×. The flat variant respects multiplication precedence and trains on odd expression lengths 3–39, with validation lengths 33, 35, and 39. The bracketed variant uses expression trees, trains on lengths 3–40, and validates at 32, 36, and 40. Each validation length has 256 examples; only the final value is supervised.

• S<sub>5</sub> state tracking: compose 32 permutations and supervise the running product after each input, following Grazzi et al. (2024). Standard inputs range over all 120 permutations; the swaps variant uses the identity and the ten single transpositions. Each variant has 256 validation sequences of length 32. We report final-state accuracy as the primary metric and prefix-token accuracy in Table 5.

For each run, we pool counts across its configured validation lengths before averaging across initializations. Figure 11 shows all three seeds at each optimizer step, and Table 2 compares every model at step 2,000.

## J.1.2 Parity: learning speed and initialization variability

At 500 steps, RLT-1 6 + 2 reaches $9 9 . 4 4 \pm 0 . 9 8 \%$ validation accuracy, compared with $4 8 . 4 8 \pm 0 . 5 3 \%$ for the Transformer. RLT-1 4 + 4, 5 + 3, and 7 + 1 average about 83%, with SDs of 28–30 percentage points; $8 + 0$ remains near chance. All three 6 + 2 seeds reach 100% by step 600. At step 2,000, splits $4 + 4$ through 7 + 1 reach $1 0 0 \pm 0 \%$ , while 8 + 0 reaches 98.83 ± 1.92% and the Transformer $9 4 . 8 4 \pm 3 . 4 3 \%$ . Figure 7 compares the early and final checkpoints.

## J.1.3 Modular arithmetic with and without brackets

Flat modular arithmetic shows substantial initialization variability at 2,000 steps. RLT-1 5 + 3 has the highest mean, 94.18 ± 7.29%, compared with $6 4 . 0 2 \pm 3 7 . 6 4 \%$ for the Transformer. The 4 + 4 scores are 17.58%, 19.53%, and 98.96% for seeds 42, 43, and 44; their mean is $4 5 . 3 6 \pm 4 6 . 4 3 \%$ . At this training budget, several architectures produce both successful and near-chance runs.

![](images/2929d87e4248d8ba4c7f33fbf336ac369bfaa292fb12e0d1614064c36ffa7f5a.jpg)  
Figure 11 Validation accuracy during training, three seeds for every task. Addition measures teacherforced answer-token accuracy; the other tasks measure final-label or final-state accuracy. Lines and bands show mean ± sample SD across seeds 42, 43, and 44 $( n = 3$ at every step); bands are clipped to the accuracy range. Curves are unsmoothed and extend through 2,000 steps. Horizontal dotted lines show uniform-prediction accuracy. Standard $S _ { 5 }$ uses a narrower vertical scale.

Bracketed expressions produce closer model means: RLT-1 splits range from 70.53% to 75.17%, compared with $7 3 . 8 7 \pm 9 . 0 7 \%$ for the Transformer. The two generators difer in operator structure and label distribution. Parentheses also consume positions, so equal token lengths can contain diferent numbers of arithmetic operations.

## J.1.4 Addition and permutation state tracking

All six architectures reach $1 0 0 \pm 0 \%$ teacher-forced token accuracy on the mixed 1–8-digit addition validation set at step 2,000. On $\mathrm { s w a p s } { - } S _ { 5 } , \mathrm { R L T - 1 } ~ 4 + 4 , 5 + 3 .$ , and $6 + 2$ reach $1 0 0 \pm 0 \%$ final-state accuracy. The $7 { + 1 }$ and $8 { + 0 }$ means are $9 9 . 3 5 { \pm } 0 . 8 1 \%$ and $9 9 . 6 1 \pm 0 . 6 8 \%$ , compared with $9 9 . 0 9 \pm 0 . 2 3 \%$ for the Transformer.

On standard $S _ { 5 } { } _ { : }$ , mean final-state accuracy is 0.52–2.47%, against a uniform 120-class reference of 0.83%. Mean prefix-token accuracy is 5.40–7.99% (Table 5); this metric includes easier early states.

## J.1.5 Length generalization

Checkpoint selection follows Section 3. All 18 model–seed combinations for a task receive the same examples at each length: 256 addition pairs or 1,024 formal-task sequences. Figure 12 shows the full grids; Table 3 reports the longest tested lengths. Appendix J.6 gives checkpoint steps and supplementary metrics.

Addition and parity. Teacher-forced addition accuracy falls when operand widths exceed the 1–8-digit training range. At nine digits per operand, RLT-1 $7 + 1$ reaches $6 8 . 0 5 \pm 3 . 6 4 \%$ , compared with $5 8 . 6 1 \pm 2 . 4 4 \%$ for the Transformer. By 32 digits, model means lie between 14.89% and 16.84%.

Parity generalization difers across depth splits even when training-length accuracy is perfect. At 256 bits, $5 + 3$ and $7 + 1$ retain $1 0 0 \pm 0 \%$ , while $6 + 2$ reaches $8 4 . 0 5 \pm 2 7 . 6 3 \%$ and $4 + 4$ reaches $6 6 . 7 6 { \scriptstyle \pm 2 8 . 7 8 \% }$ . The $8 { + 0 }$ mean is $6 8 . 9 1 { \pm } 2 7 . 3 9 \%$ , and the Transformer is near chance at $5 0 . 0 7 { \pm } 1 . 6 3 \%$

Modular arithmetic. On flat expressions of length 63, RLT-1 5+3 reaches $6 6 . 9 6 { \pm } 4 0 . 0 4 \%$ , compared with $2 4 . 4 8 \pm 3 . 8 9 \%$ for the Transformer. At length 127, 5 + 3 retains $3 9 . 6 5 \pm 1 9 . 8 1 \%$ the Transformer is at $1 9 . 3 0 \pm 1 . 0 4 \%$ . By length 255, all model means lie between 18.00% and 21.42%, near the 20% uniform reference. The large intermediate-length SDs reflect the diferent training outcomes seen in the flat mod-5 validation curves. Bracketed mod-5 also loses accuracy as expressions grow longer.

Permutation state tracking. At 256 operations on $\mathsf { s w a p s } { - } S _ { 5 } ,$ , RLT-1 4 + 4 reaches $9 7 . 3 0 \pm 2 . 7 6 \%$ final-state accuracy, compared with $0 . 8 5 \pm 0 . 3 0 \%$ for the Transformer. The $5 + 3$ and $6 + 2$ means are $9 2 . 7 1 \pm 1 . 8 7 \%$ and $7 8 . 9 1 \pm 2 2 . 2 6 \%$ ; 7 + 1 and $8 + 0$ are near the uniform reference. At 512 operations, $4 + 4$ retains $5 5 . 7 0 \pm 2 5 . 7 8 \%$ final-state accuracy and $9 1 . 1 6 \pm 6 . 0 9 \%$ prefix-token accuracy, versus $0 . 8 5 \pm 0 . 3 0 \%$ and $9 . 3 3 \pm 0 . 1 1 \%$ for the Transformer. Standard $S _ { 5 }$ remains low at this length, with mean final-state accuracies from 0.72% to 1.43%.

Figure 13 compares three measures of state-tracking accuracy across test lengths. Prefix-token accuracy averages correctness over all intermediate states; final-state accuracy scores only the last state. Whole-sequence accuracy requires every prefix prediction to be correct. A correct final state can follow incorrect intermediate predictions, while prefix-token accuracy can remain high when errors occur late in the sequence.

![](images/a746f6bf62a50edf6555c0ebb84275f331068e586a43fd550f6a19307499a482.jpg)  
Mean ± sample SD (n=3) · best ID-validation-loss checkpoint per run 256 addition pairs / 1,024 formal sequences per length · shared test examples

Figure 12 Length generalization of the best in-distribution checkpoints, three initialization seeds. Addition measures teacher-forced answer-token accuracy on 256 pairs per operand width; the other tasks measure final-label or final-state accuracy on 1,024 sequences per length. Points and error bars show mean ± sample SD across seeds 42, 43, and 44, using identical test examples. SD whiskers may extend outside the accuracy range. Gray regions mark trained lengths; horizontal dotted lines show uniform-prediction accuracy. Flat mod-5 uses actual odd expression lengths, excluding BOS and the final equals sign.

![](images/89352b9e7370185b762fe4905ad4b47751d1b2d4e0d24c5d1cb52f069be06f46.jpg)  
Seeds 42, 43, 44 · mean ± sample SD (n=3) · identical 1,024 sequences per length Panel-specific vertical scales · best ID-validation-loss checkpoint per run

Figure 13 $S _ { 5 }$ length generalization under three scoring rules. Rows show prefix-token, final-state, and whole-sequence accuracy; columns show standard and swaps inputs. Every model and seed receives the same 1,024 sequences per length. Points and error bars show mean ± sample SD across three initialization seeds. Whole-sequence success requires all prefix predictions to be correct. Panel-specific vertical scales expose low accuracies on standard $S _ { 5 }$

Table 3 Accuracy (%) at the longest tested length. Entries are mean ± sample SD across three initialization seeds, using the checkpoint and scoring rules of Figure 12. Addition reports teacher-forced token accuracy; formal tasks report final-label or final-state accuracy.
<table><tr><td>Task (test length)</td><td> $\mathrm { R L T - 1 } ~ 4 \substack { + 4 }$ </td><td> $\mathrm { R L T - 1 } ~ 5 + 3$ </td><td> $\mathrm { R L T } \mathrm { - } 1 6 { + } 2$ </td><td> $\mathrm { R L T ^ { - 1 } } 7 { + } 1$ </td><td> $\mathrm { R L T - 1 } 8 \mathrm { + } 0$ </td><td></td><td>GPT 8</td></tr><tr><td>Addition (32)</td><td> $1 5 . 3 0 \pm 0 . 4 4$ </td><td> $1 4 . 8 9 \pm 2 . 2 7$ </td><td> $1 5 . 7 4 \pm 1 . 5 8$ </td><td> $1 5 . 5 9 \pm 1 . 5 7$ </td><td> $1 5 . 3 1 \pm 0 . 9 3$ </td><td></td><td> $1 6 . 8 4 \pm 1 . 4 5$ </td></tr><tr><td>Parity (256)</td><td> $6 6 . 7 6 \pm 2 8 . 7 8$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $8 4 . 0 5 \pm 2 7 . 6 3$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $6 8 . 9 1 \pm 2 7 . 3 9$ </td><td></td><td> $5 0 . 0 7 \pm 1 . 6 3$ </td></tr><tr><td>Mod 5, no brackets (255)</td><td> $1 8 . 0 0 \pm 0 . 3 9$ </td><td> $2 0 . 5 7 \pm 0 . 6 2$ </td><td> $2 1 . 4 2 \pm 0 . 4 9$ </td><td> $1 9 . 3 4 \pm 0 . 5 4$ </td><td> $2 0 . 4 4 \pm 0 . 9 1$ </td><td></td><td> $2 0 . 3 5 \pm 2 . 0 1$ </td></tr><tr><td>Mod 5, brackets (256)</td><td> $2 5 . 2 0 \pm 1 . 7 1$ </td><td> $2 5 . 8 1 \pm 4 . 3 4$ </td><td> $2 1 . 5 8 \pm 1 . 8 6$ </td><td> $2 2 . 0 4 \pm 1 . 7 2$ </td><td> $2 2 . 3 0 \pm 1 . 1 3$ </td><td></td><td> $2 5 . 0 7 \pm 2 . 0 5$ </td></tr><tr><td> $S _ { 5 } ,$  standard (512)</td><td> $1 . 2 4 \pm 0 . 3 0$ </td><td> $1 . 4 3 \pm 0 . 6 0$ </td><td> $1 . 3 0 \pm 0 . 3 1$ </td><td> $0 . 8 8 \pm 0 . 2 0$ </td><td> $0 . 7 2 \pm 0 . 3 1$ </td><td></td><td> $0 . 8 1 \pm 0 . 0 6$ </td></tr><tr><td> $S _ { 5 } ,$  swaps (512)</td><td> $5 5 . 7 0 \pm 2 5 . 7 8$ </td><td> $3 4 . 8 6 \pm 6 . 1 0$ </td><td> $2 2 . 1 4 \pm 2 0 . 9 5$ </td><td> $0 . 8 5 \pm 0 . 0 6$ </td><td> $0 . 8 5 \pm 0 . 2 0$ </td><td></td><td> $0 . 8 5 \pm 0 . 3 0$ </td></tr></table>

## J.1.6 Training stability and comparison scope

Parity RLT-1 5 + 3 with seed 42 drops from 100% at step 500 to 48.05% at step 600, then returns to 100% at step 700. Figure 17 shows the unsmoothed losses, and Appendix J gives individual parity curves. This temporary regression and the flat mod-5 seed diferences show that final averages can hide unstable learning trajectories.

The comparison with a decoder-only Transformer changes feedback, attention structure, parameter count, and compute together. Appendices J.9 and J.10 isolate the feedback with RLT-0 and RLT-2 at every split.

## J.2 Parameter counts

Table 4 lists unique trainable parameters. Each RLT-1 decoder block has both SWA and encodermemory cross-attention; each Transformer block has one attention sublayer. RLT-1 $8 + 0$ applies the learned state projection and gated merge at every token.

Table 4 Unique trainable parameters in the eight-layer models. Each model has the same size across all six tasks.
<table><tr><td>Model</td><td>Parameters (M) Relative to GPT 8</td></tr><tr><td>RLT-1 4+4</td><td>28.73 1.135×</td></tr><tr><td>RLT-1 5+3</td><td>28.20 1.114×</td></tr><tr><td>RLT-1 6+2</td><td>27.68 1.093×</td></tr><tr><td>RLT-1 7+1</td><td>27.15 1.073×</td></tr><tr><td>RLT-1  $_ { 8 + 0 }$ </td><td>26.10 1.031×</td></tr><tr><td>Transformer 8</td><td>25.31 1.000×</td></tr></table>

## J.3 Implementation of RLT-0

The rlt\_parallel\_tiny control was merged in nanogptpro-dev PR #164. It uses an untied $4 + 4$ encoder–decoder split, width 512, four attention heads, FFN width 1,365, vocabulary capacity 277, an SWA window of eight, one encoder-memory group, and width-µP. It has 27,938,944 unique parameters, and its architecture follows Section 2.7. Training uses full backpropagation, and architecture markers distinguish RLT-0 and RLT-1 checkpoints.

## J.4 Seed aggregation and metric definitions

Let $a _ { m , s , t }$ be validation accuracy for model $m _ { z }$ , initialization seed s, and optimizer step t. For every task we report

$$
\bar { a } _ { m , t } = \frac { 1 } { 3 } \sum _ { s \in \{ 4 2 , 4 3 , 4 4 \} } a _ { m , s , t } , \qquad \mathrm { S D } _ { m , t } = \sqrt { \frac { 1 } { 2 } \sum _ { s \in \{ 4 2 , 4 3 , 4 4 \} } ( a _ { m , s , t } - \bar { a } _ { m , t } ) ^ { 2 } } .\tag{J.1}
$$

We aggregate all three seeds at the same step, without interpolation or imputation. All curves extend through step 2,000, the common comparison step in Table 2.

Parity and mod-5 supervise one final label per example. Their three equally sized validation groups are pooled within a run before averaging across seeds. $S _ { 5 }$ supervises each prefix; final-state accuracy counts the last state in each sequence, while prefix-token accuracy pools all supervised positions. Addition uses teacher forcing on the full prompt and correct preceding answer tokens. Its denominator includes all answer tokens—digits, spaces, brackets, and EOS—and excludes prompt and padding positions. Addition loss is teacher-forced cross-entropy normalized per example.

## J.5 Reproducing the completed training snapshot

The formal tasks use data seeds 20260914 for training, 20260915 for validation, and 20260916 for test sets. Addition uses data seed 42. All 108 runs completed 2,000 steps using the implementation based on upstream commit d1a4516. Native histories were frozen on September 17, 2026; the manifest records exact collection timestamps and source hashes.

![](images/32e2798463df83f989b77f033243fe8d2732a426422c6fdaaf92b9c30373d9cf.jpg)  
Figure 14 RLT-0 during training or prompt prefill. The decoder receives the encoder outputs directly, $z _ { 1 : T } ^ { 0 } = e _ { 1 : T }$ , with the gated merge and final-hidden-state feedback path removed. Each outlined stack repeats for its indicated depth. For $G = 1$ , all decoder layers read the same projected global KV; query position t can read encoder positions $j \leq t .$ Each decoder layer constructs its own SWA KV and applies a causal window of size W, including the current position. Known positions run together within each layer, while layers follow depth order. Incremental generation retains the encoder cache, global memory, and each layer’s last $W - 1$ SWA entries and proceeds one token at a time.

Table 5 Final-state and prefix-token accuracy (%) on the fixed $S _ { 5 }$ validation sets at step 2,000, reported as mean ± sample SD over seeds 42, 43, and 44.
<table><tr><td>Task</td><td>Metric</td><td>RLT-1 4+4</td><td>RLT-1 5+3</td><td>RLT-16+2</td><td>RLT-1 7+1</td><td> $\mathrm { R L T - 1 } ~ 8 \substack { + 0 }$ </td><td>GPT 8</td></tr><tr><td> $S _ { 5 } ,$  swaps</td><td>Final state</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 3 5 \pm 0 . 8 1$ </td><td> $9 9 . 6 1 \pm 0 . 6 8$ </td><td> $9 9 . 0 9 \pm 0 . 2 3$ </td></tr><tr><td> $S _ { 5 } ,$  swaps</td><td>Prefix token</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 9 6 \pm 0 . 0 5$ </td><td> $9 9 . 9 5 \pm 0 . 0 6$ </td><td> $9 9 . 9 1 \pm 0 . 0 6$ </td></tr><tr><td> $S _ { 5 } ,$  standard</td><td>Final state</td><td> $0 . 7 8 \pm \ : 0 . 6 8$ </td><td> $2 . 4 7 \pm 1 . 2 6$ </td><td> $2 . 2 1 \pm 1 . 4 8$ </td><td> $2 . 0 8 \pm 0 . 9 8$ </td><td> $1 . 8 2 \pm 1 . 1 9$ </td><td> $0 . 5 2 \pm 0 . 2 3$ </td></tr><tr><td> $S _ { 5 } ,$  standard</td><td>Prefix token</td><td> $5 . 5 7 \pm 2 . 1 1$ </td><td> $7 . 8 2 \pm 0 . 5 5$ </td><td> $7 . 9 9 \pm 0 . 2 9$ </td><td> $7 . 6 7 \pm 0 . 5 4$ </td><td> $6 . 4 7 \pm 0 . 6 5$ </td><td> $5 . 4 0 \pm 1 . 2 1$ </td></tr></table>

manifest.json and configs.json in data/experiments-depth8/ record run identifiers, settings, and source/corpus fingerprints. plot\_depth8.py in scripts/ rebuilds training figures and tables from the 2,160 validation records and 216,000 unsmoothed training-step records. The fixedlength token-accuracy export contains 1,800 validation records from 90 completed formal-task runs; plot\_token\_accuracy.py in the same directory rebuilds its plot.

![](images/02dcaa26dcbcee24181a54c89b04b67de9a65c604333e45bc47c88354d91d382.jpg)  
Figure 15 Parity accuracy for each initialization seed through 2,000 steps. All runs share the indexed training stream and validation set; curves show temporary regressions and diferences in learning speed.

## J.6 Length-generalization protocol and supplementary metrics

The study uses all 108 completed runs, with seeds 42, 43, and 44 for every architecture and task. We select each run’s checkpoint by minimum native ID-validation example loss, pooling the configured formal-task validation lengths and taking the earliest step on ties. Table 6 lists the selected steps; checkpoint and training-result hashes accompany the data. Test results do not participate in selection.

The addition grid is 9, 10, 11, 12, 14, 16, and 32 digits in each operand. Parity and bracketed mod-5 use lengths 32, 40, 48, 64, 96, 128, 192, and 256; both $S _ { 5 }$ tasks use 32, 48, 64, 96, 128, 192, 256, and 512. Flat expressions alternate operands and binary operators, so each requested even length L maps to actual expression length L − 1: 31, 39, 47, 63, 95, 127, 191, and 255. Formal-task lengths exclude BOS and any final equals sign; flat mod-5 has L + 1 input tokens after these markers are added. There are 47 shared test sets and 846 model–seed–length evaluations.

Addition uses 256 unique unordered operand pairs per length, sampled from the native test partition with seed 20260916. Both operands have exactly the indicated width and no leading zero; reversed-digit serialization matches training. The evaluation computes teacher-forced answer-token accuracy without autoregressive generation. Each formal-task set contains 1,024 sequences from the original generator and test seed 20260916. All models and initializations use the same examples at a task and length. Shared holdouts are reused from the previous benchmark; additional formal-task sets exclude duplicates and existing validation/test examples.

After matching checkpoints and holdout fingerprints, we reuse 697 model–seed–length results and evaluate the remaining 149 combinations. All reported metrics are recomputed from saved per-example predictions or teacher-token counts. Evaluation uses CPU FP32, four intra-op threads, and the recorded training runtime; checkpoint weights remain fixed. New evaluations use batch 32, and reused native formal evaluations used batch four. A batch-size check with RLT-1 4 + 4 matched all 32,768 state predictions across the two batch sizes on 32 length-512 sequences per $S _ { 5 }$ task. The teacher-forced addition evaluator also reproduces the saved per-example token counts for RLT-1 4 + 4 and Transformer 8 on 32 shared 32-digit pairs.

At each test length, we first compute a score for each initialization and then apply the mean/SD formula above with length in place of training step. Error bars show sample SD across the three initializations on shared test examples. Figure 16 reports supplementary token accuracy for all six tasks; the three $S _ { 5 }$ scoring rules are compared in Appendix J.1.5 (Figure 13).

The directory data/experiments-depth8-generalization/ contains 846 per-seed rows, 282 aggregates, checkpoint/dataset fingerprints, and a verification report. Before regenerating figures and tables, scripts/plot\_generalization.py checks seed/length coverage, scores computed from counts, sample SDs, and checkpoint correspondence with the training snapshot.

![](images/0dd7f758f476dc10e38ab7522790cad4e84969df9759a62669b19c7e6a6a69c8.jpg)  
Mean ± sample SD (n=3) · best ID-validation-loss checkpoint per run 256 addition pairs / 1,024 formal sequences per length · shared test examples  
Figure 16 Token accuracy across test lengths, three initialization seeds. The checkpoints and shared examples are those of Figure 12. Addition uses teacher forcing on answer tokens; $S _ { 5 }$ scores every prefix state; parity and mod-5 each score one final label. Points and error bars show mean ± sample SD across seeds $4 2 ,$ $^ { 4 3 , }$ and 44. Gray regions mark trained lengths.

![](images/f5438f09c96d4cde7e133bac840d3b338b2c677e5ae291c5d67857c3670f2a8e.jpg)  
Figure 17 Unsmoothed global-batch training cross-entropy, shown as mean ± sample SD across three seeds for every task. Means and band limits below $1 0 ^ { - 8 }$ are displayed at $1 0 ^ { - 8 }$ on the logarithmic axis. Loss is measured before the optimizer update; logged gradient norms are measured before clipping. Compare losses within each task, since supervision difers across tasks.

Table 6 Checkpoint steps selected by minimum in-distribution validation loss for length generalization. Every run completed 2,000 steps; seeds 42, 43, and 44 are listed separately.
<table><tr><td>Task</td><td>Seed</td><td>RLT-1  $4 { + 4 }$ </td><td>RLT-15+3</td><td>RLT-1  $6 + 2$ </td><td>RLT-1  $7 + 1$ </td><td>RLT-1  $_ { 8 + 0 }$ </td><td>GPT 8</td></tr><tr><td rowspan="2">Addition</td><td>42</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td>43</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td rowspan="4">Parity</td><td>44</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td><td>2000</td><td>2000</td></tr><tr><td>42</td><td>900</td><td>1600</td><td>800</td><td>1500</td><td>2000</td><td>2000</td></tr><tr><td>43</td><td>1100</td><td>2000</td><td>900</td><td>1500</td><td>1900</td><td>1900</td></tr><tr><td>44</td><td>1400</td><td>1000</td><td>1200</td><td>1100</td><td>2000</td><td>2000</td></tr><tr><td rowspan="3">Mod 5, no brackets</td><td>42</td><td>800</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>400</td></tr><tr><td>43</td><td>800</td><td>2000</td><td>2000</td><td>800</td><td>400</td><td>2000</td></tr><tr><td>44</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td></tr><tr><td rowspan="3">Mod 5, brackets</td><td>42</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td></tr><tr><td>43</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td>44</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td rowspan="3"> $S _ { 5 } ,$  standard</td><td>42</td><td>1800</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td>43</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td>44</td><td>2000</td><td>2000</td><td>1800</td><td>2000</td><td>2000</td><td>2000</td></tr><tr><td rowspan="4"> $S _ { 5 } ,$  swaps</td><td>42</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td><td>2000</td><td>2000</td></tr><tr><td>43</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td></tr><tr><td>44</td><td>2000</td><td>2000</td><td>2000</td><td>1900</td><td>2000</td><td>2000</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

## J.7 Token-accuracy trajectories at lengths 32 and 33

Figure 18 compares token accuracy at length 32 for parity, bracketed mod-5, and both $S _ { 5 }$ tasks, and at length 33 for flat mod-5. Flat expressions alternate numbers and binary operators, so their payload lengths are odd; 33 is the nearest configured validation length to 32. The completed snapshot contains 1,800 validation checkpoints from 90 runs: six architectures, five formal tasks, and three seeds, each trained for 2,000 steps. At length 32, $S _ { 5 }$ token accuracy scores 8,192 prefix states across 256 sequences; parity and mod-5 each score one final label per sequence.

![](images/bb21f5faddc996ff54a5c6e12c6e755d6141f7d48a3fe8e50d4d988095e6c499.jpg)  
Figure 18 Validation token accuracy on five formal tasks, three seeds per task. Flat mod-5 uses payload length 33; all other panels use length 32. Every point uses all three initialization seeds at the same optimizer step; bands show mean ± sample SD, clipped to the accuracy range. $S _ { 5 }$ scores all prefix states; parity and mod-5 score the final label. All curves extend through 2,000 steps. Standard $S _ { 5 }$ uses a narrower vertical scale; horizontal dotted lines show uniform-prediction accuracy.

Table 7 Sixteen-layer parity at 256 bits, seed 42. Final-answer accuracy (%) at the best ID-loss checkpoint and at step 2,000.
<table><tr><td>Model</td><td>Best step</td><td>Best checkpoint</td><td>Step 2,000</td></tr><tr><td>RLT-1 8+8</td><td>1400</td><td>100.00</td><td>100.00</td></tr><tr><td>RLT-1 9+7</td><td>1300</td><td>100.00</td><td>100.00</td></tr><tr><td>RLT-1 10+6</td><td>1800</td><td>62.70</td><td>66.11</td></tr><tr><td>RLT-1 11+5</td><td>1300</td><td>100.00</td><td>100.00</td></tr><tr><td>RLT-1 12+4</td><td>1000</td><td>98.83</td><td>98.83</td></tr><tr><td>RLT-1 13+3</td><td>2000</td><td>99.41</td><td>99.41</td></tr><tr><td>RLT-1 14+2</td><td>1500</td><td>63.57</td><td>62.11</td></tr><tr><td>RLT-1 15+1</td><td>1100</td><td>94.14</td><td>94.82</td></tr><tr><td>RLT-1 16+0</td><td>1900</td><td>100.00</td><td>100.00</td></tr><tr><td>Transformer 16</td><td>2000</td><td>49.41</td><td>49.41</td></tr></table>

## J.8 Sixteen-layer parity: best and final checkpoints

All ten seed-42 parity runs completed 2,000 updates with training lengths 3–40, global batch 1,024, and microbatch 32. Figure 19 compares the best ID-loss checkpoint with the fixed step-2,000 checkpoint on the same 1,024 examples per length. Exact ID-loss ties select the earliest checkpoint, giving steps 1,400 and 1,300 for 8 + 8 and 9 + 7. Paired evaluations use matching batch sizes; their best-checkpoint predictions agree with the previously reported predictions.

Accuracy is unchanged in 74 of 80 model–length comparisons. At 256 bits, 10 + 6 changes from 62.70% to 66.11%, 14 + 2 from 63.57% to 62.11%, and 15 + 1 from 94.14% to 94.82%. RLT-1 8 + 8, 9 + 7, 11 + 5, and 16 + 0 retain 100% at both checkpoints, compared with 49.41% for Transformer 16.

Parity · 16 layers · best versus final checkpoint  
![](images/f87242f97da68bfd5fed0d2ff4363c1b3ae2d3824c75020dfcff349b1d8bbc69.jpg)  
Seed 42 · all runs completed 2,000 steps · training batch 1,024 · 1,024 shared examples per length · best steps in parenthese

Figure 19 Sixteen-layer parity: best versus final checkpoint. Seed 42, 1,024 shared examples per length; parentheses give best ID-loss checkpoint steps. Gray shading marks the training range; the dotted line marks 50% chance.

Table 8 End-of-training validation for all eight-layer variants. Mean ± sample SD over seeds 42–44, in percent; addition uses teacher-forced tokens, other tasks use final answers or states. Addition, parity, and $S _ { 5 }$ use 2,000 steps; mod-5 uses 5,000.
<table><tr><td>Family</td><td>Split</td><td>Addition</td><td>Parity</td><td> $S _ { 5 } { \mathrm { ~ s w a p s } }$ </td><td> $S _ { 5 }$  standard</td><td></td><td>Mod-5 flat Mod-5 brackets</td></tr><tr><td>RLT-1</td><td>4+4</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 7 8 \pm 0 . 6 8$ </td><td> $4 6 . 4 8 \pm 4 6 . 3 5$ </td><td> $9 7 . 3 1 \pm 1 . 1 4$ </td></tr><tr><td>RLT-1</td><td>5+3</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 4 7 \pm 1 . 2 6$ </td><td> $9 9 . 1 3 \pm 1 . 2 8$ </td><td> $9 8 . 2 6 \pm 0 . 4 9$ </td></tr><tr><td>RLT-1</td><td>6+2</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 2 1 \pm 1 . 4 8$ </td><td> $9 9 . 6 5 \pm 0 . 3 3$ </td><td> $9 8 . 0 0 \pm 1 . 4 8 $ </td></tr><tr><td>RLT-1</td><td>7+1</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 3 5 \pm 0 . 8 1 $ </td><td> $2 . 0 8 \pm 0 . 9 8$ </td><td> $9 8 . 3 9 \pm 2 . 7 8$ </td><td> $9 7 . 1 4 \pm 1 . 1 6$ </td></tr><tr><td>RLT-1</td><td>8+0</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 8 . 8 3 \pm 1 . 9 2 $ </td><td> $9 9 . 6 1 \pm 0 . 6 8$ </td><td> $1 . 8 2 \pm 1 . 1 9$ </td><td> $8 1 . 1 2 \pm 3 0 . 9 0$ </td><td> $9 6 . 5 7 \pm 0 . 8 5$ </td></tr><tr><td>RLT-0</td><td>4+4</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $6 6 . 4 5 \pm 1 8 . 1 2$ </td><td> $9 8 . 9 6 \pm 0 . 2 3$ </td><td> $0 . 6 5 \pm 0 . 8 1$ </td><td> $9 3 . 4 9 \pm 1 . 9 2$ </td><td> $9 4 . 2 7 \pm 2 . 0 0$ </td></tr><tr><td>RLT-0</td><td>5+3</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $6 9 . 6 6 \pm 1 0 . 1 5$ </td><td> $9 8 . 0 5 \pm 0 . 6 8$ </td><td> $0 . 3 9 \pm 0 . 3 9$ </td><td> $7 3 . 5 2 \pm 2 4 . 8 8$ </td><td> $9 2 . 5 3 \pm 2 . 7 6$ </td></tr><tr><td>RLT-0</td><td>6+2</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $5 8 . 5 1 \pm 1 . 5 0$ </td><td> $9 8 . 5 7 \pm 0 . 6 0$ </td><td> $1 . 5 6 \pm 1 . 1 7$ </td><td> $6 3 . 7 2 \pm 3 8 . 6 7$ </td><td> $9 5 . 8 8 \pm 1 . 6 4$ </td></tr><tr><td>RLT-0</td><td>7+1</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $8 3 . 1 2 \pm 2 5 . 8 2$ </td><td> $9 9 . 0 9 \pm 0 . 8 1 $ </td><td> $1 . 4 3 \pm 0 . 6 0$ </td><td> $9 7 . 4 8 \pm 1 . 0 1$ </td><td> $9 7 . 7 4 \pm 0 . 5 3$ </td></tr><tr><td>RLT-0</td><td>8+0</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $8 5 . 6 8 \pm 1 8 . 2 5$ </td><td> $9 8 . 9 6 \pm 0 . 8 1$ </td><td> $0 . 9 1 \pm 0 . 6 0$ </td><td> $9 7 . 4 4 \pm 1 . 7 3$ </td><td> $9 7 . 3 5 \pm 0 . 6 7$ </td></tr><tr><td>RLT-2 chunk4</td><td>4+4</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 9 . 8 7 \pm 0 . 2 3 $ </td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $2 . 4 7 \pm 1 . 1 9$ </td><td> $7 3 . 8 7 \pm 4 4 . 4 7$ </td><td> $9 6 . 4 0 \pm 1 . 0 9$ </td></tr><tr><td>RLT-2 chunk4</td><td>5+3</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $8 3 . 1 2 \pm 2 9 . 1 3$ </td><td> $9 9 . 4 8 \pm 0 . 2 3 $ </td><td> $1 . 0 4 \pm 0 . 6 0$ </td><td> $7 1 . 7 4 \pm 4 5 . 5 8$ </td><td> $9 8 . 0 0 \pm 0 . 6 0$ </td></tr><tr><td>RLT-2 chunk4</td><td>6+2</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 5 . 4 9 \pm 7 . 8 2$ </td><td> $9 9 . 4 8 \pm 0 . 2 3 $ </td><td> $1 . 9 5 \pm 1 . 7 0$ </td><td> $9 8 . 9 1 \pm 1 . 3 3 $ </td><td> $9 7 . 0 9 \pm 1 . 6 6$ </td></tr><tr><td>RLT-2 chunk4</td><td>7+1</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $7 8 . 4 3 \pm 1 6 . 6 5$ </td><td> $9 8 . 8 3 \pm 1 . 7 0$ </td><td> $0 . 6 5 \pm 0 . 2 3$ </td><td> $9 7 . 7 0 \pm 2 . 5 6$ </td><td> $9 7 . 0 1 \pm 1 . 3 8$ </td></tr><tr><td>RLT-2 chunk4</td><td>8+0</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $7 6 . 5 2 \pm 2 3 . 0 5$ </td><td> $9 9 . 4 8 \pm 0 . 9 0$ </td><td> $0 . 7 8 \pm 0 . 7 8$ </td><td> $9 8 . 7 8 \pm 0 . 0 8$ </td><td> $9 6 . 5 7 \pm 0 . 4 2$ </td></tr><tr><td>Transformer</td><td>T8</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $9 4 . 8 4 \pm 3 . 4 3$ </td><td> $9 9 . 0 9 \pm 0 . 2 3 $ </td><td>0.52 ± 0.23</td><td> $9 9 . 1 3 \pm 0 . 7 8$ </td><td> $9 7 . 0 1 \pm 1 . 0 4$ </td></tr></table>

## J.9 Three-seed comparison of feedback variants

We extend the eight-layer study to RLT-0 and RLT-2 chunk4, each at splits 4 + 4 through 8 + 0 and seeds 42, 43, and 44. These 180 runs train for 5,000 updates on mod-5 and 2,000 on the other four tasks. Width, data generation, optimizer, batch sizes, and CPU execution match Section $3 ;$ the cosine schedule ends at the corresponding training budget. RLT-1 and RLT-2 use feedback scale 0.1 and TBPTT128; RLT-0 removes the feedback merge, has 787,968 fewer parameters per split, and uses full BPTT. The three-seed RLT-1 and Transformer controls for addition, parity, and $S _ { 5 }$ are reused from the original study.

Table 8 reports validation at the end of training for all 288 runs. Figure 20 shows 4 + 4 training curves; the data repository provides curves for every split. Every validation mean uses seeds 42, 43, and 44 at the same step, including the Transformer mod-5 controls.

Formal-task tests select the minimum ID validation example loss, with earliest-step tie breaking. We recount final-state, prefix-token, and whole-sequence scores from saved predictions and verify identical holdouts across variants and seeds. Each length has 1,024 examples; this comparison uses the native test grid through length 256 (255 for flat mod-5). The separate 512-operation $S _ { 5 }$ results in Section 3.2 remain specific to RLT-1 and Transformer.

Section 3 discusses the parity and swaps- $S _ { 5 }$ results; Table 9 gives the 4 + 4 values at selected lengths, including both mod-5 tasks. All 30 RLT-0 and RLT-2 addition runs reach 100% teacher-forced token accuracy on mixed 1–8-digit validation at step 2,000. Their length generalization beyond eight digits was not evaluated; the 9–32-digit figures compare RLT-1 with the Transformer.

Mean ± sample SD · all models use seeds 42, 43, 44 · addition: teacher-forced token

![](images/d5877ae93a77ad8acb874f3a92d9fa6606eb1bb7547361b1f113dc759508aade.jpg)  
Figure 20 Feedback variants at 4 + 4: validation during training. Lines and bands show mean ± sample SD over seeds 42–44 for every model. Addition measures teacher-forced answer tokens; other tasks measure final answers or states. Bands are clipped to 0–100%; standard $S _ { 5 }$ uses a narrower vertical scale.

![](images/19ae96274eacbfd25629b5bcd050d5bd176095781e2814c3ad2a000c357e5c25.jpg)  
Figure 21 Parity length generalization across feedback variants and depth splits. All models use three seeds, 2,000 training steps, and their best ID-loss checkpoint. Error bars show sample SD, clipped to 0–100% for display; gray shading marks training lengths and the dotted line marks chance.

![](images/1bd802878b34946b73ffb7b60355f9dc06ef17f6799690677f456581b4133b97.jpg)  
Figure 22 Swaps-S<sub>5</sub> length generalization across feedback variants. Final-state accuracy, mean ± sample SD over three seeds, using each run’s best ID-loss checkpoint from 2,000 training steps. Error bars are clipped to 0–100% for display.

Table 9 Best-checkpoint test accuracy at 4 + 4. Final-answer/state accuracy (%), mean ± sample SD over three seeds, on the same 1,024 examples per length. Standard $S _ { 5 }$ at length 32 is in distribution; the other rows are out of distribution.
<table><tr><td>Task / length</td><td>RLT-1</td><td></td><td>RLT-0 RLT-2 chunk4 Transformer 8</td><td></td></tr><tr><td>Parity / 64</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $5 0 . 2 3 \pm 3 . 8 0$ </td><td> $9 8 . 9 9 \pm 1 . 6 6$ </td><td> $4 8 . 4 7 \pm 1 . 1 8$ </td></tr><tr><td> $S _ { 5 }$  swaps / 48</td><td> $1 0 0 . 0 0 \pm 0 . 0 0$ </td><td> $7 . 2 6 \pm 3 . 1 7$ </td><td> $6 8 . 2 9 \pm 8 . 7 5$ </td><td> $2 2 . 1 0 \pm 2 . 7 6$ </td></tr><tr><td> $S _ { 5 }$  standard / 32</td><td> $1 . 5 3 \pm 0 . 4 6$ </td><td> $1 . 2 0 \pm 0 . 7 2$ </td><td> $1 . 3 7 \pm 0 . 8 7$ </td><td> $0 . 9 1 \pm 0 . 1 5$ </td></tr><tr><td>Mod-5 flat / 63</td><td> $4 4 . 4 3 \pm 4 3 . 8 1$ </td><td> $2 1 . 8 4 \pm 2 . 1 7$ </td><td> $5 0 . 7 8 \pm 3 8 . 0 2$ </td><td> $3 3 . 2 0 \pm 2 . 3 3$ </td></tr><tr><td>Mod-5 brackets  $/ 6 4$ </td><td> $6 4 . 1 0 \pm 3 . 4 9$ </td><td> $3 9 . 4 5 \pm 0 . 6 1$ </td><td> $6 2 . 7 6 \pm 4 . 4 2$ </td><td> $4 6 . 7 1 \pm 1 . 2 1$ </td></tr></table>

![](images/7aca5a1beeddf77a07d56be6bafaad12908fb59bf8022a99c5d3a0b6a1373c31.jpg)  
Figure 23 Standard $S _ { 5 }$ length generalization across feedback variants. Final-state accuracy, mean ± sample SD over three seeds; all models train for 2,000 steps. The vertical axis is expanded near chance (1/120).

## J.10 Mod-5 feedback variants at 5,000 steps

All 96 mod-5 runs train for 5,000 updates. RLT-0, RLT-1, and RLT-2 chunk4 each contribute 30 runs across two tasks, five splits, and seeds 42–44. The Transformer contributes six runs, with the same three seeds on both tasks.

Flat mod-5 remains sensitive to initialization after 5,000 steps. For chunk4 4 + 4, best-checkpoint ID accuracy at length 33 is 100.00%, 20.21%, and 98.73% for seeds 42, 43, and 44. At length 63, its three-seed mean is $5 0 . 7 8 \pm 3 8 . 0 2 \%$ , compared with $4 4 . 4 3 \pm 4 3 . 8 1 \%$ for RLT-1, $2 1 . 8 4 \pm 2 . 1 7 \%$ for RLT-0, and $3 3 . 2 0 \pm 2 . 3 3 \%$ for Transformer 8.

On bracketed mod-5 at length 64, RLT-1 4 + 4 reaches $6 4 . 1 0 \pm 3 . 4 9 \%$ , chunk4 $6 2 . 7 6 \pm 4 . 4 2 \%$ ， RLT-0 39.45 ± 0.61%, and Transformer $8 ~ 4 6 . 7 1 \pm 1 . 2 1 \%$

In the seed-42 CPU timing comparison at 4 + 4, chunk4 training steps are 2.27 times as fast as RLT-1, and RLT-0 steps 4.17–4.30 times as fast, across the two mod-5 tasks. These single-seed means cover steps 1,001–2,000 on the shared cluster and exclude validation, checkpoint writes, and logging. Frozen scores, per-seed records, provenance, and figure regeneration instructions are in data/variants-20260925/.

Mod-5 without brackets · ID validation · final answer  
![](images/dd9bc8b36ae627cb6bcc9a7057aab62c316a3db0a804ee41ce1bd94c7d05da61.jpg)  
Figure 24 Mod-5 without brackets: ID validation. Mean ± sample SD over seeds 42–44 for every model; all runs train for 5,000 steps. Bands are clipped to 0–100%.

![](images/e940f0e436d18078cfdda642cca087c578c05dfc3207b96ecd77f9d7793acfcb.jpg)  
Mod-5 without brackets · length generalization · final answer  
Figure 25 Mod-5 without brackets: length generalization. 5,000-step runs at best ID-loss checkpoints; every curve shows mean ± sample SD over seeds 42–44. Error bars are clipped to 0–100% for display.

![](images/05ac507c1ff5cfc84074ca1e7e404bfd7aeb72a339f6d95fe618a1a25e8e57d7.jpg)  
Figure 26 Mod-5 with brackets: ID validation. Mean ± sample SD over seeds 42–44 for every model; all runs train for 5,000 steps. Bands are clipped to 0–100%.

![](images/1bc3dcf2e511c922a016e5263620b2bbed0d2e6dd5f04151b9fbc32b6d14f4d2.jpg)  
Mod-5 with brackets · length generalization · final answer  
Figure 27 Mod-5 with brackets: length generalization. 5,000-step runs at best ID-loss checkpoints; every curve shows mean ± sample SD over seeds 42–44. Error bars are clipped to 0–100% for display.

## J.11 Addition feedback scale at a fixed learning-rate schedule

We vary only the RLT-1 feedback scale $\alpha ,$ keeping the initialization, architecture, training data, and optimizer schedule fixed within each split and seed. All runs train on mixed 1–8-digit addition for 2,000 updates, with global batch 512, microbatch 32, and TBPTT128. The learning rate warms up for 200 steps to $1 0 ^ { - 4 }$ and decays to $5 \times 1 0 ^ { - 6 }$ . We test $\alpha = 0 . 0 3$ at every split and $\alpha = 0 . 0 1$ at every split except $7 + 1$ , with seeds 42, 43, and 44. The comparison includes the $\alpha = 0 . 1$ RLT-1 runs and Transformer 8 at the same seeds, for 45 models in total.

We select each model’s checkpoint by minimum ID validation loss, taking the earliest step on exact ties; the $7 + 1$ run with $\alpha = 0 . 1$ and seed 44 selects step 1,900, and the other 44 models select step 2,000. Figure 28 and Table 10 report teacher-forced answer-token accuracy on 256 shared examples at each of seven widths from 9 to 32 digits, as mean ± sample SD over the three seeds. Counts include answer formatting and EOS, and exclude prompts and padding.

No tested feedback reduction improves accuracy at every width for any split. Across the 63 comparisons with $\alpha = 0 . 1$ at the same split and width, 37 means rise and 26 fall, by −3.86 to +5.60 percentage points; only three diferences exceed both sample SDs. For $8 + 0 , \alpha = 0 . 0 1$ raises the 12- digit mean from $2 9 . 4 9 \pm 4 . 3 8 \%$ to $3 2 . 2 1 \pm 5 . 1 3 \%$ , while the 16-digit mean moves from $2 4 . 9 2 \pm 0 . 3 9 \%$ to $2 4 . 6 6 \pm 4 . 8 8 \%$ . At 32 digits, RLT-1 means range from 14.44% to 15.91%, compared with $1 6 . 8 4 \pm 1 . 4 5 \%$ for Transformer 8.

Table 10 Addition feedback-scale ablation, seeds 42–44. Teacher-forced answer-token accuracy (%) at selected widths, mean ± sample SD over three seeds, after $2 { , } 0 0 0$ training steps with the same learning-rate schedule. Each width uses 256 shared examples; all seven widths are available in the data repository.
<table><tr><td>Model</td><td>α</td><td>9 digits</td><td>12 digits</td><td>16 digits</td><td>32 digits</td></tr><tr><td>RLT-1 4+4</td><td>0.01</td><td> $4 7 . 2 5 \pm 4 . 0 4$ </td><td> $3 3 . 1 1 \pm 4 . 5 6$ </td><td> $2 5 . 9 3 \pm 1 . 8 8$ </td><td> $1 4 . 4 4 \pm 0 . 1 7$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 4 \substack { + 4 }$ </td><td>0.03</td><td> $4 8 . 7 5 \pm 5 . 3 4$ </td><td> $3 2 . 0 0 \pm 2 . 0 7$ </td><td> $2 6 . 2 2 \pm 0 . 7 4$ </td><td> $1 4 . 7 4 \pm 1 . 2 5$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 4 \substack { + 4 }$ </td><td>0.1</td><td> $4 6 . 7 7 \pm 1 . 3 4$ </td><td> $3 0 . 7 2 \pm 0 . 5 8$ </td><td> $2 5 . 2 8 \pm 1 . 4 8$ </td><td> $1 5 . 3 0 \pm 0 . 4 4$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 5 + 3$ </td><td>0.01</td><td> $5 7 . 9 6 \pm 0 . 6 7$ </td><td> $3 2 . 3 6 \pm 2 . 9 3$ </td><td> $2 3 . 0 1 \pm 5 . 2 2$ </td><td> $1 5 . 2 9 \pm 2 . 1 9$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 5 + 3$ </td><td>0.03</td><td> $5 6 . 3 0 \pm 2 . 4 1$ </td><td> $3 2 . 6 0 \pm 5 . 6 1$ </td><td> $2 3 . 5 4 \pm 3 . 7 5$ </td><td> $1 4 . 6 6 \pm 1 . 4 9$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 5 + 3$ </td><td>0.1</td><td> $5 8 . 5 1 \pm 8 . 1 2$ </td><td> $3 4 . 9 7 \pm 4 . 5 4$ </td><td> $2 4 . 2 0 \pm 5 . 2 6$ </td><td> $1 4 . 8 9 \pm 2 . 2 7$ </td></tr><tr><td>RLT-1 6+2</td><td>0.01</td><td> $5 6 . 5 2 \pm 9 . 8 7$ </td><td> $3 0 . 9 7 \pm 4 . 6 4$ </td><td> $2 6 . 3 5 \pm 2 . 9 8$ </td><td> $1 5 . 2 1 \pm 1 . 4 6$ </td></tr><tr><td>RLT-1 6+2</td><td>0.03</td><td> $6 3 . 8 5 \pm 1 0 . 1 5$ </td><td> $3 0 . 3 9 \pm 2 . 8 5$ </td><td> $2 7 . 1 8 \pm 4 . 2 6$ </td><td> $1 4 . 8 6 \pm 0 . 6 5$ </td></tr><tr><td> $\mathrm { R L T } \mathrm { - } 1 6 { + } 2$ </td><td>0.1</td><td> $5 8 . 2 5 \pm 1 0 . 2 9$ </td><td> $2 9 . 2 2 \pm 2 . 2 8$ </td><td> $2 3 . 9 2 \pm 1 . 5 8$ </td><td> $1 5 . 7 4 \pm 1 . 5 8$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 7 + 1$ </td><td>0.03</td><td> $6 6 . 4 3 \pm 2 . 5 3$ </td><td> $3 0 . 1 0 \pm 2 . 2 9$ </td><td> $2 6 . 2 7 \pm 1 . 4 2$ </td><td> $1 5 . 9 1 \pm 0 . 6 3$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 7 + 1$ </td><td>0.1</td><td> $6 8 . 0 5 \pm 3 . 6 4$ </td><td> $2 9 . 8 0 \pm 4 . 0 3$ </td><td> $2 6 . 4 5 \pm 4 . 3 1$ </td><td> $1 5 . 5 9 \pm 1 . 5 7$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 8 \substack { + 0 }$ </td><td>0.01</td><td> $6 2 . 2 0 \pm 9 . 8 5$ </td><td> $3 2 . 2 1 \pm 5 . 1 3$ </td><td> $2 4 . 6 6 \pm 4 . 8 8$ </td><td> $1 5 . 4 5 \pm 0 . 6 6$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 8 \substack { + 0 }$ </td><td>0.03</td><td> $5 9 . 1 8 \pm 4 . 4 7$ </td><td> $2 8 . 8 1 \pm 3 . 5 7$ </td><td> $2 5 . 9 2 \pm 1 . 1 0$ </td><td> $1 4 . 9 0 \pm 0 . 5 4$ </td></tr><tr><td> $\mathrm { R L T - 1 } ~ 8 \substack { + 0 }$ </td><td>0.1</td><td> $6 2 . 1 3 \pm 7 . 7 5$ </td><td> $2 9 . 4 9 \pm 4 . 3 8$ </td><td> $2 4 . 9 2 \pm 0 . 3 9$ </td><td> $1 5 . 3 1 \pm 0 . 9 3$ </td></tr><tr><td> $\mathrm { T r a n s f o r m e r } 8$ </td><td>一</td><td> $5 8 . 6 1 \pm 2 . 4 4$ </td><td> $2 6 . 8 4 \pm 3 . 6 9$ </td><td> $2 6 . 2 2 \pm 1 . 5 4$ </td><td> $1 6 . 8 4 \pm 1 . 4 5$ </td></tr></table>

![](images/bda816ed937e1c416cc76b05817bbfdcb142296fdd4b5eae839f0a343797527a.jpg)  
Figure 28 Addition generalization as feedback scale changes. Each panel compares the tested feedback scales with Transformer 8 at one RLT-1 split; α = 0.01 was not tested at 7 + 1. Points and error bars show mean ± sample SD over seeds 42, 43, and 44; all models use 2,000 training steps and the same learning-rate schedule. Accuracy measures teacher-forced answer tokens on the same 256 examples at each width.