# 4-Tensor Attention Model for Semantic Physical Reality

Jongwook Kim, Sangheon Yun

IndigoWave, Center for Quantum Spacetime, Sogang University,

35 Baekbeom-ro, Mapo-gu, Seoul 04107, Republic of Korea

dr.jongwookkim@gmail.com

October 9, 2026

## Abstract

We describe a 4-tensor attention model that predicts the next semantic state of a scene, for video generation and robot planning. A window of states has positions (x, t) and two fibers, a semantic fiber and a temporal-context fiber, and one softmax normalizes attention jointly over the window. Frames and an agent’s situation are written as those states; the encoder, the renderer, and the planner remain outside the update. To test the update on its own, we train on ROCStories, where each window poses the same next-sentence task at the semantic layer. On the validation split, with one seed per setting, the last-sentence cross-entropy on the three matched settings is lower for the 4-tensor model than for a free-running one-dimensional transformer by 5.3% at H=2, L=2, by 2.6% at H=4, L=2, and by 2.4% at H=4, L=3. At H=4, L=2 the parameter counts are nearly the same, 172.5M and 175.9M. On the same two GPUs that 4-tensor run finished in 2.4 hours and the baseline run in 45.2 hours; the baseline is trained by free-running decoding, one sequential forward pass per target token.

Code is available at https://github.com/indigowave511/semantic-world.

Keywords: 4-tensor attention, semantic representation

## Contents

1 Introduction 4   
1.1 Implementation Scheme 4   
1.2 Scope of This Paper 4   
1.3 The Semantic State Sequence 5   
1.4 Rank-4 Representation 5   
1.5 Score Tensor and Joint Softmax . 6   
1.6 Attention Output 6   
1.7 Prediction Targets and the Learning Objective . 7   
1.8 Autoregressive Inference 7   
2 4-Tensor Transformer Design 8   
2.1 Overview of the Tensor Pipeline 8   
2.2 Stage-by-Stage Logic . 8   
2.3 Joint-Phase Temporal Embedding 9   
2.4 Final Assembly of the Input Tensor 10   
3 Forward Pass and Pipeline Overview 10   
3.1 Pipeline Overview . 10   
3.2 Residual After Attention 11   
3.3 Event Position MLP Block 11   
3.4 Residual After MLP and Readout-Boundary Normalization 11   
3.5 Vocabulary Readout and Structured Factorization . 12   
3.5.1 Sequence-Independent Matrix Pre-computation 12   
3.6 Structural Constraints Imposed by the Loss Objective 13   
3.7 Mapping to the Vocabulary Simplex . 13   
3.8 From Embedding-Level Loss to Cross-Entropy Loss 13   
4 Enabling Construction of the Forward Pipeline 14   
4.1 Index convention and storage 14   
4.2 Parameters, bufers, and initialization 15   
4.3 Step 1: construction of the input tensor 15   
4.4 Steps 2–5: one attention head 16   
4.5 Multi-head sum and the Pre-LN layer 17   
4.6 Readout and the training loss 18   
4.7 End-to-end training procedure 19   
4.8 Faithfulness checks 20   
5 Reference Implementation 20   
5.1 Input construction 21   
5.2 Rank-4 attention head 21   
5.3 Multi-head sum, Pre-LN, and the MLP 22   
5.4 Tied readout, shifted loss, and the model step 22   
6 4-Tensor Attention Backward Pass and Weight Update Rules 23   
6.1 Joint Softmax Jacobian and Score Tensor Gradient 23   
6.2 Gradients with Respect to Q, K, and V 24   
6.3 Chain Rule to the Projection Weights 25   
6.4 Gradient-Descent Update Rule . 25   
7 Comparative Baseline: Standard 1-Dimensional Autoregressive At  
tention 25   
7.1 1-D Sequence Packing and Standard QKV Architecture 26   
7.2 Informational Parity via Free-Running Autoregressive Decoding 27   
7.3 Architectural Alignment and Capacity Matching 27   
7.4 Baseline design pipeline 28   
7.5 Code for the fair comparison 30   
8 Experimental Setup 30   
8.1 Corpus and windows . 31   
8.2 Optimization and checkpoint rule 31   
9 Experiment Data 31   
10 Discussion 33

## 1 Introduction

We posit that the semantic representation of a physical scene is a cognitive compression of that scene, and that a world model can therefore advance a scene by evolving its semantic description in discrete steps rather than by interpolating sensory data. This paper describes the update at the center of that program: an attention mechanism that predicts the next semantic state from a window of previous states.

Modeling temporally coherent entities and environments is a central problem both for video generation [1, 12] and for world models used in planning and control [2, 3]. Most current systems operate on continuous sensory representations, in pixel or latent space, and can struggle to keep scenes consistent over long horizons. Part of the dificulty is that semantically relevant state and perceptual detail are mixed in the same dense representation. Latent predictive architectures [4, 5] and discrete visual tokenization with language models [6, 7] separate the two more explicitly. When such tokens are flattened into a single autoregressive sequence, however, spatially simultaneous content is given an order it does not have, and temporal structure must be recovered from one-dimensional position alone. Video transformers avoid a single raster order by attending jointly over all spatiotemporal tokens, or by factorizing attention into spatial and temporal stages [9]; each token, however, still carries a single feature vector in which content and temporal context are entangled.

Our approach compresses the content of a scene at a fixed time t into a onedimensional semantic representation and keeps time as a separate axis. Describing a video as a temporal sequence of sentences has precedent in paragraph captioning [11], where the state of earlier sentences conditions the next; there the sentences describe frames that have been observed, whereas here the next semantic state is predicted before it is observed. Each state carries two fibers: a semantic fiber of width $d _ { x }$ and a temporal-context fiber of width $d _ { t }$ . A window of states is then a rank-4 tensor, and the update, which we call 4-tensor attention, acts on that tensor directly rather than on a flattened sequence.

## 1.1 Implementation Scheme

In the intended pipeline, semantic states are extracted from video frames at fixed intervals by an image-to-text model. The resulting sequence of textual states is processed by 4-tensor attention to predict future states, and the predicted states are rendered back into frames by a text-to-image model (fig. 1). The prediction step is where temporal consistency is decided; the encoder and the renderer determine only how faithfully a state is read from and written back to the visual domain.

## 1.2 Scope of This Paper

This paper tests one component of that program: whether the 4-tensor update predicts the next semantic state better than a flattened sequence model when both read the same window (sections 7 and 9). The image-to-text encoder, the text-to-image renderer, and any planner are outside this test. To isolate the update from visual encoding and decoding, the experiments operate directly at the semantic layer, on narrative windows from ROCStories, where each window poses the same next-sentence task.

![](images/178211a9fa7255edf90e9c6fae720e952af8413c3a8913df435f8f8a43eedd5f.jpg)  
Figure 1: The intended generative pipeline. Frames are compressed into semantic states, the 4-tensor attention update predicts the next states, and the predicted states are rendered back into frames. This paper implements and evaluates only the 4-tensor attention block (teal).

The separation of the two fibers may matter beyond sensory world models, as a representation in which transition rules could be learned apart from the objects they apply to. We return to this question in section 10.

## 1.3 The Semantic State Sequence

Let an observation be represented as a sequence of semantic states extracted at regular intervals. To give every state the same length N, shorter states are padded with a null token 0. For an agent waiting for an elevator, with $N = 8 \cdot$

```python
S(t=0): [I, am, in, front, of, the, elevator, 0]
S(t=1): [I, am, looking, at, the, floor, indicator, 0]
S(t=2): [The, elevator, arrives, at, my, floor, 0, 0]
```

A window of length T collects the most recent T states. For $T = 3$ 2

$$
X ( t = 2 )  \{ S ( 0 ) , S ( 1 ) , S ( 2 ) \} .\tag{1}
$$

The running example in this section uses $T = 3$ and $N = 8$ . The experiments use $T = 4$ and N = 16 (section 8.1).

## 1.4 Rank-4 Representation

The window is a rank-4 tensor

$$
X \in \mathbb { R } ^ { N \times d _ { x } \times T \times d _ { t } } ,\tag{2}
$$

with the states $S ( 0 ) , \ldots , S ( T { - } 1 )$ arranged along the temporal axis. Compared with standard scaled dot-product attention, every index is doubled. A sequence position becomes a pair $( x , t )$ , where x is the token position within a state and t is the temporal

slice. The feature dimension becomes a pair of fibers: the semantic $d _ { x } .$ -fiber and the temporal-context $d _ { t } { \mathrm { - f i b e r } }$ . Input fiber indices are written $( n , q )$ and output fiber indices $( m , p )$ , with $n ,$ m on the $d _ { x }$ -fiber and $q , p$ on the $d _ { t } .$ -fiber.

The query, key, and value tensors are joint bilinear contractions over both input fibers,

$$
Q _ { x m ; t p } = \sum _ { n , q } W _ { m n , p q } ^ { Q } X _ { x n ; t q } ,
$$

$$
K _ { x m ; t p } = \sum _ { n , q } W _ { m n , p q } ^ { K } X _ { x n ; t q } ,\tag{3}
$$

$$
V _ { x m ; t p } = \sum _ { n , q } W _ { m n , p q } ^ { V } X _ { x n ; t q } ,
$$

with $W ^ { Q } , W ^ { K } , W ^ { V } \in \mathbb { R } ^ { d _ { x } \times d _ { x } \times d _ { t } \times d _ { t } }$

The two fibers enter as distinct tensor factors: each token is embedded as an outer product of a semantic vector and a temporal-context vector (section 2), and the readout ties the semantic fiber to the vocabulary (section 3). The projection $W _ { m n , p q } .$ , however, is a general bilinear map on the joint $( n , q )$ space and admits arbitrary interaction between the fibers. The architecture therefore does not force transition rules to separate from content. It keeps a representation in which such a separation can be learned and tested.

## 1.5 Score Tensor and Joint Softmax

The score between a query at $( x , t )$ and a key at $( x ^ { \prime } , t ^ { \prime } )$ contracts both fibers. To avoid a clash with the state sequence $S ( t )$ , the unnormalized score tensor is written E:

$$
E _ { x t , x ^ { \prime } t ^ { \prime } } = \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \sum _ { m , p } Q _ { x m ; t p } K _ { x ^ { \prime } m ; t ^ { \prime } p } .\tag{4}
$$

The factor $1 / \sqrt { d _ { x } d _ { t } }$ is the standard variance correction for an inner product over $d _ { x } d _ { t }$ coordinates.

The softmax normalizes jointly over the full grid of key positions $( x ^ { \prime } , t ^ { \prime } )$ , so that a query may draw on any token of any state in the window:

$$
A _ { x t , x ^ { \prime } t ^ { \prime } } = \frac { \exp ( E _ { x t , x ^ { \prime } t ^ { \prime } } ) } { \displaystyle \sum _ { x ^ { \prime \prime } , t ^ { \prime \prime } } \exp ( E _ { x t , x ^ { \prime \prime } t ^ { \prime \prime } } ) } .\tag{5}
$$

Apart from padding keys, which are masked (section 4.4), no mask is applied. In particular there is no temporal causal mask: every query attends to every slice of the window.

## 1.6 Attention Output

The attention output is the A-weighted sum of values over the key grid,

$$
y _ { x m ; t p } = \sum _ { x ^ { \prime } , t ^ { \prime } } A _ { x t , x ^ { \prime } t ^ { \prime } } V _ { x ^ { \prime } m ; t ^ { \prime } p } ,\tag{6}
$$

and has the same shape $\mathbb { R } ^ { N \times d _ { x } \times T \times d _ { t } }$ as the input $X$

## 1.7 Prediction Targets and the Learning Objective

In the full model the attention block is followed by a residual connection, a position-wise MLP, and a final LayerNorm, and the result is read out to vocabulary logits at every position $( x , t )$ (sections 3 and 4). Write F for this map from the window to the logits. The readout at slice t is a prediction of the next state,

$$
\begin{array} { r } { p _ { x v } ( t ) = \mathrm { s o f t m a x } _ { v } \left( F ( X ) _ { x , t , v } \right) \quad \mathrm { p r e d i c t s } \quad S ( t + 1 ) _ { x } . } \end{array}\tag{7}
$$

An input window $S ( 0 ) , S ( 1 ) , S ( 2 )$ therefore yields predictions ${ \hat { S } } ( 1 ) , { \hat { S } } ( 2 ) , { \hat { S } } ( 3 )$ , and the training targets are the window shifted by one slice:

$$
\begin{array} { r l } & { \mathrm { S ( t { = } 1 ) : } \quad \mathrm { [ I , ~ \ a m , ~ \ a o s t i n g , ~ \ a t , ~ \ t h e , ~ \ f l o o r , ~ \ a n d i c a t o r , ~ \ 0 ] } } \\ & { \mathrm { S ( t { = } 2 ) : } \quad \mathrm { [ T h e , ~ \ e l e v a t o r , ~ \ a r r i v e s , ~ \ a t , ~ \ m y , ~ \ f l o o r , ~ \ 0 , ~ 0 ] } } \\ & { \mathrm { S ( t { = } 3 ) : } \quad \mathrm { [ T h e , ~ \ e l e v a t o r , ~ \ d o o r s , ~ \ o p e n , ~ \ i n , ~ \ f r o n t , ~ \ o f , ~ \ m e ] } } \end{array}
$$

Training minimizes the cross-entropy of all T predicted states against their targets, with padding positions ignored.

Because the joint softmax carries no temporal causal mask, for $t = 0 , \ldots , T { - } 2$ the target $S ( t { + } 1 )$ of slice t is present in input slice $t { + } 1$ . The terms $S ( 1 ) , \dots , S ( T { - } 1 )$ of the training loss therefore read sentences already in the input. Prediction of a sentence absent from the input is measured only at the last slice:

$$
{ \mathrm { E v a l u a t i o n ~ m e t r i c } }  \operatorname { C E } ( { \hat { S } } ( T ) \parallel S ( T ) ) .\tag{8}
$$

In the running example this is $\mathrm { C E } ( \hat { S } ( 3 ) \parallel S ( 3 ) )$ , and in the experiments $\mathrm { C E } ( \hat { S } ( 4 ) \parallel S ( 4 ) )$ All tokens of $\hat { S } ( T )$ are predicted in one parallel forward pass, and the tokens of $S ( T )$ enter only as targets.

## 1.8 Autoregressive Inference

The generative task is to continue the sequence beyond the window:

```python
S(t=3): [The, elevator, doors, open, in, front, of, me]
S(t=4): [I, step, inside, the, elevator, 0, 0, 0]
S(t=5): [The, doors, close, behind, me, 0, 0, 0]
S(t=6): [I, press, the, button, for, my, floor, 0]
```

The predicted state $\hat { S } ( 3 )$ is first converted to tokens, $\hat { S } ( 3 ) _ { x } = \arg \operatorname* { m a x } _ { v } p _ { x v } ( 2 )$ , or by sampling. The window then slides by one state,

$$
X ^ { \prime } = \{ S ( 1 ) , S ( 2 ) , \hat { S } ( 3 ) \} \in \mathbb { R } ^ { N \times d _ { x } \times T \times d _ { t } } ,\tag{9}
$$

where the three states again occupy slices $t = 0 , 1 , 2$ and are embedded by the same input construction. The same weights are applied, and the next state is read from the last slice:

$$
{ \hat { S } } ( 4 ) _ { x } = \mathop { \mathrm { a r g m a x } } _ { v } F ( X ^ { \prime } ) _ { x , T - 1 , v } .\tag{10}
$$

Repeating the step with $X ^ { \prime \prime } = \{ S ( 2 ) , \hat { S } ( 3 ) , \hat { S } ( 4 ) \}$ gives $\hat { S } ( 5 )$ , and so on. Ground-truth states from $S ( 3 )$ onward never enter the window; each step conditions on the model’s own earlier predictions. The window can thus be advanced repeatedly with fixed weights. This paper reports single-step results only; multi-step rollout is left to future work.

## 2 4-Tensor Transformer Design

## 2.1 Overview of the Tensor Pipeline

The 4-Tensor Transformer takes as its base unit the state tensor at a given temporal slice t, defined as:

$$
S _ { x n ; q } ( t ) \in \mathbb { R } ^ { N \times d _ { x } \times d _ { t } }\tag{11}
$$

The rank-4 input tensor $X _ { x n ; t q }$ is constructed via a sequential pipeline rather than a simultaneous summation. Each stage injects a single piece of information into a specific fiber:

$$
S ( t ) ~ \xrightarrow { + P _ { x n } } ~ { \tilde { S } } ( t ) ~ \xrightarrow { \mathrm { s t a c k ~ o v e r ~ } t } ~ X _ { x n ; t q } ^ { ( 0 ) } ~ \xrightarrow { + T _ { x t q } ^ { \mathrm { e m b } } } ~ X _ { x n ; t q }\tag{12}
$$

Here, $P _ { x n }$ is a spatial position embedding from $\mathbb { R } ^ { N }$ into the semantic $d _ { x } \mathrm { - f i b e r }$ , and $T _ { x t q } ^ { \mathrm { e m b } }$ is a temporal position embedding added from $\mathbb { R } ^ { T }$ into the temporal $d _ { t } { \mathrm { - f i b e r } }$ . This ordering ensures that the spatial token position (x) and temporal sequence position (t) remain analytically separable within the unified rank-4 object.

## 2.2 Stage-by-Stage Logic

## Stage A: Initialization by Dual-Dictionary Token Embedding

Traditionally, a token is embedded via a single dictionary lookup into a semantic vector space. However, because the temporal-context $d _ { t } \mathrm { - f i b e r }$ operates as a first-class axis in our rank-4 formulation, the token’s initial embedding must explicitly bifurcate into two independent dictionaries:

$$
E _ { \mathrm { s e m } } : V  \mathbb { R } ^ { d _ { x } } , \qquad E _ { \mathrm { t e m p } } : V  \mathbb { R } ^ { d _ { t } }\tag{13}
$$

The token embedding is rank-1 across the two fibers: the base state $S ( t )$ at position x is the outer product of the two dictionary vectors. For each discrete temporal slot $t \in \{ 0 , \ldots , T - 1 \}$ , a token at spatial position $x \in \{ 0 , \ldots , N - 1 \}$ is embedded via two separate dictionaries:

$$
S _ { x n ; q } ( t ) = E _ { \mathrm { s e m } } ( \operatorname { t o k } _ { x , t } ) _ { n } \cdot E _ { \mathrm { t e m p } } ( \operatorname { t o k } _ { x , t } ) _ { q } \ \in \ \mathbb { R } ^ { N \times d _ { x } \times d _ { t } }\tag{14}
$$

Thus, $S ( t )$ is rank-3 from the outset, intrinsically carrying content along both the semantic $d _ { x } .$ -fiber and the temporal-context $d _ { t } .$ -fiber prior to any positional injection.

## Stage B: Spatial Position Injection

The spatial position embedding $P _ { x n } \in \mathbb { R } ^ { N \times d _ { x } }$ is added independently per temporal slice:

$$
\tilde { S } _ { x n ; q } ( t ) = S _ { x n ; q } ( t ) + P _ { x n }\tag{15}
$$

Because spatial placement governs static semantic identity, $P _ { x n }$ resides entirely within the $d _ { x }$ -fiber. It broadcasts identically across the $d _ { t } .$ -fiber $( q )$ , preserving the rank-3 structure.

## Stage C: Temporal Stacking

The T sentence tensors are stacked along a newly initialized temporal axis:

$$
X _ { x n ; t q } ^ { ( 0 ) } = \tilde { S } _ { x n ; q } ( t )\tag{16}
$$

This operation maps the sequence into the rank-4 configuration $( N , d _ { x } , T , d _ { t } )$ . Because $P _ { x n }$ operates independently per sentence and broadcasts over q, the operations of adding P and stacking over t are mathematically commutative.

## Stage D: Temporal-Order Injection

Finally, the temporal-order embedding $T _ { x t q } ^ { \mathrm { e m b } } \in \mathbb { R } ^ { N \times T \times d _ { t } }$ is added, broadcasting strictly over the semantic $d _ { x } \mathrm { - f i b e r } \ ( n )$ :

$$
X _ { x n ; t q } = X _ { x n ; t q } ^ { ( 0 ) } + T _ { x t q } ^ { \mathrm { e m b } }\tag{17}
$$

While Stage A establishes the token-level context, Stage D establishes the rigorous positional trajectory across the causal surface.

## 2.3 Joint-Phase Temporal Embedding

Crucially, the temporal embedding $T ^ { \mathrm { e m b } }$ must depend on both the temporal slice t and the spatial column x. A naive parameterization $T _ { t q } ^ { \mathrm { e m b } }$ would assign the identical temporal signal to every spatial token within a given slice t. In scenarios where the same entity appears at diferent times $( \mathrm { e . g . }$ , the token [I] appearing at $t = 0$ and $t = 4$ in our physical elevator progression), a column-independent signal fails to distinguish the token’s shifting causal roles.

To resolve this, we formulate $T _ { x t q } ^ { \mathrm { e m b } }$ as a joint-phase embedding:

$$
T _ { x t q } ^ { \mathrm { e m b } } = \left\{ \begin{array} { l l } { \sin \left( \displaystyle \frac { x + t } { 1 0 0 0 0 ^ { 2 i / d _ { t } } } \right) , } & { q = 2 i } \\ { \cos \left( \displaystyle \frac { x + t } { 1 0 0 0 0 ^ { 2 i / d _ { t } } } \right) , } & { q = 2 i + 1 } \end{array} \right. \quad \quad i = 0 , \dots , \lfloor d _ { t } / 2 \rfloor - 1\tag{18}
$$

This formulation satisfies two properties of the phase $x + t ($

1. Column Independence: For a fixed t, $T _ { x t q } ^ { \mathrm { e m b } } \neq T _ { x ^ { \prime } t q } ^ { \mathrm { e m b } }$ whenever x $\neq x ^ { \prime }$ . The same phase gives $T _ { x , t , q } ^ { \mathrm { e m b } } = T _ { x - 1 , t + 1 , q } ^ { \mathrm { e m b } }$ . Positions that share a value of $x + t$ are distinguished once $T ^ { \mathrm { e m b } }$ is added to the spatial embedding $P _ { x n }$

2. Ofset Consistency: For a fixed shift $k ,$ , the phase increment depends only on k and the frequency index i, and a linear map $M _ { k }$ on the q-fiber, independent of x, carries $T _ { x , t , \cdot } ^ { \mathrm { e m b } }$ to $T _ { x , t + k , * } ^ { \mathrm { e m b } }$ . The implemented slide does not use $M _ { k }$ . Each new window is written again onto positions $t \in \{ 0 , \ldots , T - 1 \}$ , and $T ^ { \mathrm { e m b } }$ is recomputed from those positions.

## 2.4 Final Assembly of the Input Tensor

The complete rank-4 input tensor is thus formalized through the direct superposition of these structured stages:

$$
\boxed { X _ { x n ; t q } = \underbrace { [ S _ { x n ; q } ( t ) + P _ { x n } ] } _ { \mathrm { S t a g e s ~ A ~ \& ~ B ~ ( P e r - s l i c e ) } } + \underbrace { T _ { x t q } ^ { \mathrm { e m b } } } _ { \mathrm { S t a g e ~ D ~ ( J o i n t - p h a s e ) } } }\tag{19}
$$

yielding the exact causal dimensionality required:

$$
X \in \mathbb { R } ^ { N \times d _ { x } \times T \times d _ { t } }\tag{20}
$$

This boxed formulation is the input to the 4-Tensor Attention block. The sum $S + P + T ^ { \mathrm { e m b } }$ is one tensor; the two fibers stay distinct factors, and $W _ { m n , p q }$ may mix them.

## 3 Forward Pass and Pipeline Overview

Following the construction of the completed rank-4 input tensor $X _ { x n ; t q }$ , this section covers the remainder of a single-layer, single-head forward pass through to the training loss. Indices are zero-based, $x \in \{ 0 , \ldots , N - 1 \}$ and $t \in \{ 0 , \ldots , T - 1 \}$ , as in section 4.1. The pipeline encompasses the attention operation, residual connections with dropout, the position-wise MLP block, readout-boundary LayerNorm, vocabulary readout, and the final cross-entropy loss, including the t-index shift that connects the attention update to the prediction targets.

## 3.1 Pipeline Overview

The full transformation from the input tensor to the scalar loss $\mathcal { L }$ is strictly sequentially ordered:

$$
\begin{array} { r l } { X _ { x n ; t q } ~ { \xrightarrow { \mathrm { 4 \cdot T e n s o r ~ A t t e n t i o n } } } ~ y _ { x m ; t p } ~ { \xrightarrow { \mathrm { + } X , \mathrm { d r o p } } } ~ X _ { x n ; t q } ^ { \prime } ~ { \xrightarrow { \mathrm { M L P } } } ~ Z _ { x n ; t q } } \\ { } & { } \end{array} \downarrow + X ^ { \prime } , { \mathrm { d r o p } }\tag{21}
$$

Where the intermediate variables are defined as follows:

$y _ { x m ; t p } .$ : The raw rank-4 output of the attention mechanism.

• $X _ { x n ; t q } ^ { \prime } \mathrm { . }$ The intermediate state following the first residual connection and dropout.

$Z _ { x n ; t q } \mathrm { . }$ The unflattened output of the position-wise MLP block.

$\hat { X } _ { x n ; t q }$ : The completed rank-4 representation for the layer following the second residual connection and dropout.

$X _ { x n ; t q } ^ { \mathrm { n o r m } }$ : The state after applying LayerNorm over the semantic and temporalcontext fibers $( n , q )$ , preparing the tensor for vocabulary readout.

The construction of $y _ { x m ; t p }$ (including the QKV generation, score tensor computation, and joint softmax) initiates the stage. The subsequent sections detail the pipeline beginning directly from the first residual connection.

## 3.2 Residual After Attention

Following the attention mechanism, the first residual connection is applied:

$$
X _ { x n ; t q } ^ { \prime } = X _ { x n ; t q } + \mathrm { { d r o p o u t } } ( y _ { x n ; t q } )\tag{22}
$$

The indices align directly: the attention output y resides in the exact same $( x , n , t , q )$ space as the input X. This dimensional consistency is guaranteed because the query, key, and value tensors were all constructed by contracting the identical $( n , q )$ fibers via the weight matrices $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { d _ { x } \times d _ { x } \times d _ { t } \times d _ { t } }$ . Dropout is applied exclusively to the residual branch to preserve the main identity pathway.

## 3.3 Event Position MLP Block

The feed-forward network operates independently across each $( x , t )$ event position on the causal surface. Because the semantic (n) and temporal-context $( q )$ fibers represent distinct algebraic spaces, we impose no separability assumption between them at this stage. Instead, the $( n , q )$ slice at each fixed $( x , t )$ is jointly flattened into a single feature vector of dimension $d _ { x } d _ { t }$ before applying the non-linear transformations:

$$
Z _ { x n ; t q } = \mathrm { u n f l a t t e n } \Big ( W _ { 2 } \sigma \big ( W _ { 1 } \mathrm { f a t t e n } ( X _ { x ; t \cdot } ^ { \prime } ) + b _ { 1 } \big ) + b _ { 2 } \Big )\tag{23}
$$

where flatten : $\mathbf { \colon } \mathbb { R } ^ { d _ { x } \times d _ { t } } \to \mathbb { R } ^ { d _ { x } d _ { t } }$ maps the matrix slice into a vector, and unflatten restores the original rank-4 structure.

The projection weights are defined as $W _ { 1 } \in \mathbb { R } ^ { d _ { f f } \times ( d _ { x } d _ { t } ) }$ and $W _ { 2 } \in \mathbb { R } ^ { ( d _ { x } d _ { t } ) \times d _ { f f } }$ , with σ denoting the GELU activation function. To maintain the standard transformer capacity scaling relative to the joint input dimension, the hidden layer width is explicitly set to the standard expansion ratio: $d _ { f f } = 4 d _ { x } d _ { t }$

## 3.4 Residual After MLP and Readout-Boundary Normalization

Following the MLP block, the second residual connection is applied to produce the completed layer output:

$$
\hat { X } _ { x n ; t q } = X _ { x n ; t q } ^ { \prime } + \mathrm { d r o p o u t } ( Z _ { x n ; t q } )\tag{24}
$$

The residual stream itself remains the raw sum of the block input and the dropped branch outputs. In the reduced-to-practice stack of section 4, each branch input is LayerNorm’d before attention and before the ${ \mathrm { M L P } } ;$ those normalizations are not placed after the residual add. The residual stream can therefore still grow across layers. If left unscaled at the readout, that magnitude saturates the vocabulary softmax and collapses the cross-entropy gradient. The readout-boundary LayerNorm below is retained in addition to the per-branch normalizations.

To stabilize the output logits without interfering with the internal attention dynamics, we introduce a Layer Normalization strictly at the readout boundary of the final layer. Consistent with the MLP block, this normalization operates over the jointly flattened semantic and temporal-context fibers, imposing no separability assumption. For the final layer’s output $\hat { X } _ { x n ; t q }$ , the normalized tensor $X _ { x n ; t q } ^ { \mathrm { n o r m } }$ is computed as:

$$
X _ { x \cdot ; t \cdot } ^ { \mathrm { n o r m } } = \mathrm { u n f l a t t e n } \Bigl ( \operatorname { L a y e r N o r m } _ { d _ { x } d _ { t } } \bigl ( \mathrm { f l a t t e n } ( \hat { X } _ { x \cdot ; t \cdot } ) \bigr ) \Bigr )\tag{25}
$$

The normalization is applied independently at each spatial-temporal position $( x , t )$ over the $d _ { x } d _ { t }$ dimensional space, incorporating standard learnable afine parameters $\gamma , \beta \in \mathbb { R } ^ { d _ { x } d _ { t } }$ . By operating locally at each $( x , t )$ coordinate, the normalization strictly preserves the independence of causal slots and prevents any temporal information leakage.

## 3.5 Vocabulary Readout and Structured Factorization

To project the normalized state $X _ { x n ; t q } ^ { \mathrm { n o r m } }$ into the vocabulary, the general linear map is a dense tensor $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { x } \times d _ { t } \times | V | }$ . The readout used here factorizes each vocabulary column as a rank-1 tensor on $( n , q )$

The semantic factor is the token embedding $E _ { \mathrm { s e m } } \in \mathbb { R } ^ { | V | \times d _ { x } }$ , tied to Stage A, and it contributes no new parameters. The temporal-context factor is a learned matrix $W ^ { \prime } \in \mathbb { R } ^ { d _ { t } \times | V | }$ . Each vocabulary item is therefore one rank-1 pair $( n , q )$ , the same boundary described as a rank-1 readout functional in section 10.

The factorization is:

$$
W _ { \mathrm { o u t } , n q ; v } = E _ { \mathrm { s e m } } [ v , n ] \cdot W _ { q , v } ^ { \prime }\tag{26}
$$

where $v$ indexes the vocabulary. Substituting this into the logit computation over the flattened $( n , q )$ slice yields:

$$
L _ { x v } ( t ) = \sum _ { n , q } X _ { x n ; t q } ^ { \mathrm { n o r m } } E _ { \mathrm { s e m } } [ v , n ] W _ { q , v } ^ { \prime }\tag{27}
$$

## 3.5.1 Sequence-Independent Matrix Pre-computation

To evaluate this logit eficiently and avoid materializing a prohibitively large sequencedependent intermediate tensor of size $\mathcal { O } ( N T d _ { t } | V | )$ , we define an optimal evaluation order. We pre-compute a strictly sequence-independent projection matrix M:

$$
M _ { \phi ( n , q ) , v } = E _ { \mathrm { s e m } } [ v , n ] W _ { q , v } ^ { \prime } , \qquad M \in \mathbb { R } ^ { d _ { x } d _ { t } \times | V | }\tag{28}
$$

where $\phi ( n , q ) = n d _ { t } + q$ is the identical joint-flattening bijection used in the MLP block. The final vocabulary logits $L _ { x v } ( t )$ are then computed via a direct matrix multiplication between the jointly flattened representation $X ^ { \mathrm { n o r m } }$ and the factorized matrix M:

$$
L _ { x v } ( t ) = \sum _ { n , q } X _ { x n ; t q } ^ { \mathrm { n o r m } } M _ { \phi ( n , q ) , v }\tag{29}
$$

By decoupling the lexical projection from the sequence length, this factorized construction strictly preserves both the structural meaning of the distinct fibers and the computational tractability of the forward pass.

## 3.6 Structural Constraints Imposed by the Loss Objective

The mathematical structure of the vocabulary readout is not arbitrary; it is strictly dictated by the terminal loss function. Because the final training objective evaluates the predicted semantic states $\hat { S } ( t )$ against the ground truth $S ( t )$ pointwise across both the sequence position x and the temporal step t, these two indices must be explicitly preserved. Conversely, the semantic $( n )$ and temporal-context $( q )$ fibers represent internal latent coordinates that do not exist in the discrete target space. Consequently, they must be contracted away.

This deterministic survival of indices—preserving $( x , t )$ while eliminating $( n , q ) -$ requires a mapping from the normalized latent state $X _ { x n ; t q } ^ { \mathrm { n o r m } }$ to the vocabulary logits $L _ { x v } ( t )$ . The most general linear transformation is a joint bilinear contraction over the $( n , q )$ plane:

$$
L _ { x v } ( t ) = \sum _ { n , q } X _ { x n ; t q } ^ { \mathrm { n o r m } } W _ { \mathrm { o u t } , n q ; v }\tag{30}
$$

where the readout tensor is defined as $W _ { \mathrm { o u t } } \in \mathbb { R } ^ { d _ { x } \times d _ { t } \times | V | }$

That dense contraction is the general linear map. The implemented readout replaces it by the rank-1 factorization ${ \cal W } _ { \mathrm { o u t } , n q ; v } = { \cal E } _ { \mathrm { s e m } } [ v , n ] { \cal W } _ { q , v } ^ { \prime }$ , one rank-1 pair $( n , q )$ for each vocabulary item.

## 3.7 Mapping to the Vocabulary Simplex

The raw logits $L _ { x v } ( t )$ derived from the readout projection are mapped onto the vocabulary simplex via a standard softmax operation over the vocabulary axis v:

$$
p _ { x v } ( t ) = \frac { \exp ( L _ { x v } ( t ) ) } { \sum _ { v ^ { \prime } } \exp ( L _ { x v ^ { \prime } } ( t ) ) }\tag{31}
$$

This transformation is computed independently at each spatial-temporal coordinate $( x , t )$ , yielding a discrete probability distribution $p _ { x v } ( t ) \in \Delta ^ { | V | - 1 }$ over the vocabulary space. This tensor represents the model’s predicted likelihood for each discrete semantic token v at position x and time step t.

While the explicit materialization of $p _ { x v } ( t )$ is strictly required for inference algorithms $( \mathrm { e . g . }$ , autoregressive sampling or argmax decoding), during training, this probability mapping is analytically absorbed directly into the cross-entropy objective to guarantee numerical stability.

## 3.8 From Embedding-Level Loss to Cross-Entropy Loss

While traditional continuous generative architectures often formulate training objectives as direct regression losses at the embedding level $( \mathrm { e . g . }$ , minimizing $\| \hat { X } ( t ) - X ( t ) \| ^ { 2 } )$ our discrete semantic framework evaluates predictions directly over the vocabulary simplex.

Transitioning from continuous vector space interpolation to discrete semantic evolution, the training objective is formulated as the cross-entropy loss over the predicted vocabulary distribution:

$$
\mathcal { L } = - \sum _ { t = 0 } ^ { T - 1 } \sum _ { x = 0 } ^ { N - 1 } \sum _ { v \in \mathcal { V } } y _ { x v } ( t ) \log p _ { x v } ( t )\tag{32}
$$

Table 1: Storage layout of the reduced-to-practice pipeline. The batch axis B is present on every public tensor, including the case $B = 1$
<table><tr><td>Object</td><td>Shape</td><td>Index order</td></tr><tr><td> $\mathtt { t o k e n \_ i d s } , Y$ </td><td> $( B , T , N )$ </td><td> $[ b , t , x ]$ </td></tr><tr><td> $X , Q , K , V , \mathrm { A t t n }$ </td><td> $( B , N , T , d _ { x } , d _ { t } )$ </td><td> $[ b , x , t , n , q ]$ </td></tr><tr><td>pad_mask</td><td> $( B , N , T )$ </td><td> $[ b , x , t ]$  , true on a PAD key</td></tr><tr><td> $\mathbf { \bar { \Sigma } } _ { W } \mathbf { \Sigma } ^ { Q } , W ^ { K } , W ^ { V }$ </td><td> $( d _ { x } , d _ { t } , d _ { x } , d _ { t } )$ </td><td> $[ m , p , n , q ]$ </td></tr><tr><td>scores, attention weights</td><td> $( B , N , T , N , T )$ </td><td> $[ b , x , t , x ^ { \prime } , t ^ { \prime } ]$ </td></tr><tr><td>logits</td><td> $( B , N , T , | V | )$ </td><td> $[ b , x , t , v ]$ </td></tr></table>

where $x \in \{ 0 , \ldots , N - 1 \}$ and the logit slice $t \in \{ 0 , \ldots , T - 1 \}$ is the distribution for sentence $S ( t { + } 1 )$ . Here $y _ { x v } ( t )$ is the target distribution at that position, and $p _ { x v } ( t )$ is the softmax of the readout logits. This formulation bridges the continuous hidden representations of the causal surface with the discrete semantic vocabulary space, completing the end-to-end diferentiable training pipeline.

## 4 Enabling Construction of the Forward Pipeline

This section specifies the forward pipeline that was reduced to practice. The algebraic maps of the preceding sections are unchanged. What is fixed here is the order of operations, the in-memory axis order, the numerical guards, and which tensors are parameters. An implementation that preserves every shape in table 1 can still compute a diferent function if any check in section 4.8 is omitted. Storage indices below are zero-based.

## 4.1 Index convention and storage

Theory indices remain $X _ { x n ; t q }$ and $W _ { m n , p q }$ . The stored activation places the position axes adjacent and the fibers last, so that the flattening

$$
\phi ( n , q ) = n d _ { t } + q\tag{33}
$$

is a reshape with no permutation. The same $\phi$ is used for the attention contraction, the MLP, the readout-boundary LayerNorm, and the vocabulary matrix M. A tensor whose last two axes are $( d _ { t } , d _ { x } )$ has the same product $d _ { x } d _ { t }$ and would pass a product check while applying the wrong $\phi .$ Fiber checks therefore compare the ordered pair $( d _ { x } , d _ { t } )$

The weight tensor is the view of a matrix,

$$
W [ m , p , n , q ] = \mathrm { m a t } \big [ m d _ { t } + p , ~ n d _ { t } + q \big ] ,\tag{34}
$$

so W.view $( d _ { x } d _ { t } , d _ { x } d _ { t } )$ is the linear map $( n , q ) \mapsto ( m , p )$ with no copy. Logits are stored $( B , N , T , | V | )$ and targets are stored $( B , T , N )$ . The bridge is the permutation $( b , t , x ) \mapsto ( b , x , t )$ . When $N = T$ that permutation preserves the shape. A missing permutation is therefore caught by checking that the target has shape $( B , T , N )$ before the permutation is applied.

A stored example supplied to training has shape $( B , T \mathrm { + } 1 , N )$ , the sentences $S ( 0 ) , \ldots , S ( T )$ . The forward input is the prefix $S ( 0 ) , \ldots , S ( T - 1 )$ , of shape $( B , T , N )$ An array of rank 2 and shape $( T { + } 1 , N )$ is rejected. The caller inserts the batch axis.

## 4.2 Parameters, bufers, and initialization

The learned parameters are the following.

$E _ { \mathrm { s e m } } \in \mathbb { R } ^ { | V | \times d _ { x } }$ and $E _ { \mathrm { t e m p } } \in \mathbb { R } ^ { | V | \times d _ { t } }$ , with the row at the padding index held at zero and excluded from the gradient.

• For each layer and each head, three maps $W ^ { Q } , W ^ { K } , W ^ { V }$ , each stored as above. Xavier uniform initialization is applied to the flat $( d _ { x } d _ { t } ) \times ( d _ { x } d _ { t } )$ matrix and then viewed into $( m , p , n , q )$ . Initializing the four-dimensional array directly mis-estimates the fan-in.

• For each layer, two LayerNorm afine pairs of width $d _ { x } d _ { t }$ (one before attention, one before the MLP), and one MLP of width $d _ { f f } = 4 d _ { x } d _ { t }$

• One further LayerNorm of width $d _ { x } d _ { t }$ after the last layer.

• The readout factor $W ^ { \prime } \in \mathbb { R } ^ { d _ { t } \times | V | }$ , Xavier-uniform initialized.

The spatial table $P _ { x n }$ and the joint-phase table $T _ { x t q } ^ { \mathrm { e m b } }$ are fixed sinusoids, recomputed or registered as non-learned bufers. Dropout probability $p$ is a hyperparameter, set to 0.1 in the reported runs. Evaluation disables dropout.

## 4.3 Step 1: construction of the input tensor

The input X is assembled in four stages. Each stage writes one signal into one fiber.

Stage A, content. For a token identifier tok<sub>b,t,x</sub>,

$$
S _ { b , x , n , t , q } = E _ { \mathrm { s e m } } [ \mathrm { t o k } _ { b , t , x } ] _ { n } \cdot E _ { \mathrm { t e m p } } [ \mathrm { t o k } _ { b , t , x } ] _ { q } .\tag{35}
$$

The slice at a padding identifier is the zero tensor, because both embedding rows are zero. No position has been added.

Stage B, spatial position. $P \in \mathbb { R } ^ { N \times d _ { x } }$ depends only on the column x. With $i = 0 , \ldots , \lfloor d _ { x } / 2 \rfloor - 1$

$$
P _ { x , 2 i } = \sin \Bigl ( \frac { x } { 1 0 0 0 0 ^ { 2 i / d _ { x } } } \Bigr ) , \qquad P _ { x , 2 i + 1 } = \cos \Bigl ( \frac { x } { 1 0 0 0 0 ^ { 2 i / d _ { x } } } \Bigr ) .\tag{36}
$$

If $d _ { x }$ is odd, the last coordinate remains 0. $P _ { x n }$ is added at every batch index, every sentence index, and every q:

$$
\tilde { S } _ { b , x , n , t , q } = S _ { b , x , n , t , q } + P _ { x , n } .\tag{37}
$$

Adding P before or after the stack over t is the same map, because P carries no t index.

Stage C, stack. The $T$ sentences are already arranged along the sentence axis of the stored tensor $( B , N , T , d _ { x } , d _ { t } )$ . Stacking performs no arithmetic.

Stage D, joint-phase temporal embedding. T<sup>emb</sup> $\in \mathbb { R } ^ { N \times T \times d _ { t } }$ depends on the sum of the column and the sentence index. With $i = 0 , \ldots , | d _ { t } / 2 | - 1$

$$
T _ { x , t , 2 i } ^ { \mathrm { e m b } } = \sin \Bigl ( \frac { x + t } { 1 0 0 0 0 ^ { 2 i / d _ { t } } } \Bigr ) , \qquad T _ { x , t , 2 i + 1 } ^ { \mathrm { e m b } } = \cos \Bigl ( \frac { x + t } { 1 0 0 0 0 ^ { 2 i / d _ { t } } } \Bigr ) ,\tag{38}
$$

and the last coordinate remains 0 when $d _ { t }$ is odd. It is added at every batch index and every semantic coordinate n:

$$
X _ { b , x , n , t , q } = \tilde { S } _ { b , x , n , t , q } + T _ { x , t , q } ^ { \mathrm { e m b } } .\tag{39}
$$

The phase $x + t$ makes the signal difer across columns of one sentence, and a shift of the sentence index is a linear map on the $q { \mathrm { - f i b e r } }$ . When the window slides, the positions $t \in \{ 0 , \ldots , T - 1 \}$ are reused and $T ^ { \mathrm { e m b } }$ is rebuilt on them. The maps $W ^ { Q } , \bar { W } ^ { K } , W ^ { V }$ apply to that re-indexed window.

On the stored layout the broadcasts are P expanded in the sentence and $d _ { t }$ slots, and $T ^ { \mathrm { e m b } }$ expanded in the $d _ { x }$ slot. The completed tensor has shape $( B , N , T , d _ { x } , d _ { t } )$

A padding slot is not the zero tensor after Step 1. Stages B and D add $P$ and $T ^ { \mathrm { e m b } }$ regardless of the token:

$$
X _ { \mathrm { p a d } } = P [ x ] + { T } ^ { \mathrm { e m b } } [ x , t ] \neq 0 .\tag{40}
$$

The keys and values built from that slot are nonzero. Which slots are padding is read from the integer token identifiers, not from the norm of $X$

## 4.4 Steps 2–5: one attention head

One head owns three full-fiber maps. The contraction, written on storage indices, is

$$
Q [ b , x , t , m , p ] = \sum _ { n = 0 } ^ { d _ { x } - 1 } \sum _ { q = 0 } ^ { d _ { t } - 1 } W ^ { Q } [ m , p , n , q ] X [ b , x , t , n , q ] ,\tag{41}
$$

and likewise for K and V with $W ^ { K }$ and $W ^ { V }$ . Each of $Q , K , V$ has shape $( B , N , T , d _ { x } , d _ { t } )$ In the reduced-to-practice layer this contraction receives $\operatorname { L N } ( X )$ , as specified in section 4.5, rather than the raw residual.

The score is the scaled Frobenius product over the $( m , p )$ slice, jointly over the key grid $( x ^ { \prime } , t ^ { \prime } )$

$$
E [ b , x , t , x ^ { \prime } , t ^ { \prime } ] = \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \sum _ { m , p } Q [ b , x , t , m , p ] K [ b , x ^ { \prime } , t ^ { \prime } , m , p ] .\tag{42}
$$

With N adjacent to $T .$ , the reshape $( B , N T , d _ { x } d _ { t } )$ followed by a batched matrix product realizes this sum. The score tensor has shape $( B , N , T , N , T )$

The padding mask is built once per forward call:

$$
\mathtt { p a d \_ m a s k } [ b , x , t ] = \big ( \mathtt { t o k e n \_ i d s } [ b , t , x ] = \mathtt { p a d d i n g \_ i d x } \big ) ,\tag{43}
$$

which is the permutation $( b , t , x ) \mapsto ( b , x , t )$ of a boolean tensor. It is reused at every depth of that call, because later layers rewrite values of X and do not rewrite which slot held which identifier. It is rebuilt on the next call. A generation step that writes a token into a former padding slot must observe the new mask.

The mask is applied on the key axes only, by writing the smallest finite value of the score dtype into $E [ b , x , t , x ^ { \prime } , t ^ { \prime } ]$ whenever pad mask $[ b , x ^ { \prime } , t ^ { \prime } ]$ is true. Query axes are left unchanged: a padding query is removed from the loss by the ignore index, and the MLP has no path from one slot to another. The fill value is the finite minimum, not $- \infty$ . If every key of a query is padding, a fill of $- \infty$ makes the softmax maximum undefined. The finite minimum leaves a finite maximum, and the softmax returns the uniform distribution on the NT keys.

Joint softmax then normalizes the last two axes together. In exact arithmetic,

$$
A [ b , x , t , x ^ { \prime } , t ^ { \prime } ] = \frac { \exp { \left( E [ b , x , t , x ^ { \prime } , t ^ { \prime } ] \right) } } { \sum _ { x ^ { \prime \prime } , t ^ { \prime \prime } } \exp { \left( E [ b , x , t , x ^ { \prime \prime } , t ^ { \prime \prime } ] \right) } } .\tag{44}
$$

The implementation subtracts the maximum over $( x ^ { \prime \prime } , t ^ { \prime \prime } )$ before the exponential, which is the stabilization inside a softmax on the flattened key axis of length $N T$ . Dropout is not applied to $A .$

The head output contracts A with V and returns to the fiber layout:

$$
\mathrm { A t t n } [ b , x , t , m , p ] = \sum _ { x ^ { \prime } , t ^ { \prime } } A [ b , x , t , x ^ { \prime } , t ^ { \prime } ] V [ b , x ^ { \prime } , t ^ { \prime } , m , p ] .\tag{45}
$$

The result has shape $( B , N , T , d _ { x } , d _ { t } )$ , identical to the residual stream.

## 4.5 Multi-head sum and the Pre-LN layer

A head count $H \geq 1$ instantiates H independent copies of Steps 2–5. Each copy keeps fibers of size $d _ { x }$ and $d _ { t }$ . The copies are summed, with no factor $1 / { \sqrt { H } }$

$$
\mathrm { A t t n } ( U ) = \sum _ { h = 1 } ^ { H } \mathrm { A t t n } ^ { ( h ) } ( U ) .\tag{46}
$$

The case $H = 1$ is exactly one head. The argument U is a normalized copy of the residual, not the residual itself.

Let $\mathrm { L N } _ { d }$ denote LayerNorm over a last axis of length $d = d _ { x } d _ { t }$ , with $\varepsilon = 1 0 ^ { - 5 }$ and learned afine scale and shift, applied independently at each $( b , x , t )$

$$
\operatorname { L N } _ { d } ( v ) = \gamma \odot { \frac { v - \mu ( v ) } { \sqrt { \sigma ^ { 2 } ( v ) + \varepsilon } } } + \beta , \qquad \gamma , \beta \in \mathbb { R } ^ { d } .\tag{47}
$$

The flatten and unflatten use $\phi .$ . One layer then executes

$$
U = \mathrm { u n f l a t t e n } \bigl ( \mathrm { L N } _ { d } ^ { ( 1 ) } ( \mathrm { f l a t t e n } ( X ) ) \bigr ) ,\tag{48}
$$

$$
\begin{array} { r } { \mathrm { A t t n } = \sum _ { h } \mathrm { A t t n } ^ { ( h ) } ( U ) , } \end{array}\tag{49}
$$

$$
X ^ { \mathrm { a t t n } } = X + \operatorname { D r o p o u t } _ { p } ( \mathrm { A t t n } ) ,\tag{50}
$$

$$
U ^ { \prime } = \mathrm { u n f l a t t e n } \big ( \mathrm { L N } _ { d } ^ { ( 2 ) } ( \mathrm { f l a t t e n } ( X ^ { \mathrm { a t t n } } ) ) \big ) ,\tag{51}
$$

$$
Z = \operatorname { M L P } ( U ^ { \prime } ) ,\tag{52}
$$

$$
X ^ { \mathrm { o u t } } = X ^ { \mathrm { a t t n } } + \operatorname { D r o p o u t } _ { p } ( Z ) .\tag{53}
$$

$\mathrm { L N } ^ { ( 1 ) }$ and $\mathrm { L N } ^ { ( 2 ) }$ are distinct modules. The residual additions use the unnormalized stream X and $X ^ { \mathrm { a t t n } }$ . The two dropout calls share the rate $p$ and draw independent masks. They are the only dropout sites: embeddings, the pre-layer norms, the joint softmax, the final norm, and the readout are undropped.

The MLP acts at each $( b , x , t )$ on the flattened fiber $v \in \mathbb { R } ^ { d _ { x } d _ { t } }$

$$
Z = \mathrm { { u n f l a t t e n } } { \Big ( } W _ { 2 } \mathrm { { G E L U } } ( W _ { 1 } v + b _ { 1 } ) + b _ { 2 } { \Big ) } ,\tag{54}
$$

with $W _ { 1 } \in \mathbb { R } ^ { d _ { f f } \times d _ { x } d _ { t } } , W _ { 2 } \in \mathbb { R } ^ { d _ { x } d _ { t } \times d _ { f f } } , d _ { f f } = 4 d _ { x } d _ { t }$ , and

$$
\begin{array} { r } { \mathrm { G E L U } ( u ) = \frac { 1 } { 2 } u \Big ( 1 + \mathrm { e r f } \big ( u / \sqrt { 2 } \big ) \Big ) . } \end{array}\tag{55}
$$

The linear maps use the reference implementation’s default initialization (Kaiming uniform on the weights, uniform biases of bound $1 / \sqrt { d _ { \mathrm { i n } } } )$ . The hidden width 4 $d _ { x } d _ { t }$ is the ordinary expansion of a model dimension $d _ { x } d _ { t }$

L such layers are applied in sequence to the same $X$ , each consuming the shared pad mask of the call. After the last layer, one further $\mathrm { L N } _ { d } ,$ denoted final norm, produces $X ^ { \mathrm { n o r m } }$ . This norm is required in addition to the per-branch norms, because the residual stream that reaches it is the raw sum.

## 4.6 Readout and the training loss

The readout does not allocate a parameter tensor of shape $( d _ { x } , d _ { t } , | V | )$ . It ties the semantic fiber to the same $E _ { \mathrm { s e m } }$ used in Stage A and learns only $W ^ { \prime }$ . The sequenceindependent matrix

$$
M [ \phi ( n , q ) , v ] = E _ { \mathrm { s e m } } [ v , n ] W ^ { \prime } [ q , v ] , \qquad M \in \mathbb { R } ^ { d _ { x } d _ { t } \times | V | } ,\tag{56}
$$

is formed first. The logits are then one product,

$$
{ L _ { \mathrm { f l a t } } = X _ { \mathrm { f l a t } } ^ { \mathrm { n o r m } } M , \qquad X _ { \mathrm { f l a t } } ^ { \mathrm { n o r m } } = \mathrm { r e s h a p e } \big ( X ^ { \mathrm { n o r m } } , B N T \times d _ { x } d _ { t } \big ) , }\tag{57}
$$

viewed back to $( B , N , T , | V | )$ . Because $E _ { \mathrm { s e m } } [ \mathsf { p a d d i n g . i d x } ] = 0$ , the padding column of M is identically zero, and the padding logit is zero. A padding argmax on the generation path is reported as a warning. The logits are left unchanged, and no permanent mask is written onto the vocabulary axis.

Softmax over v is a diagnostic map used for sampling or inspection. The training loss does not materialize it and does not materialize a one-hot target. Given the full window token ids full of shape $( B , T \mathrm { + } 1 , N )$

$$
\mathtt { t o k e n \_ i d s } = \mathtt { t o k e n \_ i d s \_ f u l 1 } [ : , 0 : T , : ] ,\tag{58}
$$

$$
Y = \mathrm { t o k e n \_ i d s \_ f u l l } [ : , 1 : T + 1 , : ] ,\tag{59}
$$

so $Y [ b , t , x ]$ is the token of $S ( t { + } 1 )$ and Y has shape $( B , T , N )$ . The loss is the mean token cross-entropy of the integer targets after the axis bridge,

$$
{ \mathcal { L } } = \mathrm { C r o s s E m t r o p y } \left( { \mathrm { r e s h a p e } } ( L , B N T \times | V | ) , { \mathrm { r e s h a p e } } ( Y _ { \mathrm { p e r m } } , B N T ) \right) ,\tag{60}
$$

with ignore index equal to padding idx. The log-softmax sits inside that reduction. Ignore-index removal drops padding targets from $\mathcal { L } .$ It does not replace the key mask

Table 2: Fiber role through one reduced-to-practice forward pass. The pair $( n , q )$ is eliminated at the readout and does not appear in ${ \mathcal { L } } .$
<table><tr><td>Stage</td><td></td><td>Tensor Storage shape</td></tr><tr><td>Step 1</td><td>X</td><td> $( B , N , T , d _ { x } , d _ { t } )$ </td></tr><tr><td>Steps 2–5, summed over heads</td><td>Attn</td><td> $( B , N , T , d _ { x } , d _ { t } )$ </td></tr><tr><td>After both residual adds</td><td>Xout</td><td> $( B , N , T , d _ { x } , d _ { t } )$ </td></tr><tr><td>Readout-boundary norm</td><td>Xnorm</td><td> $( B , N , T , d _ { x } , d _ { t } )$ </td></tr><tr><td>Tied readout</td><td>logits</td><td> $( B , N , T , | V | )$ </td></tr><tr><td>Shifted targets</td><td>Y</td><td> $( B , T , N )$ </td></tr></table>

of section 4.4, and the zero padding row of $E _ { \mathrm { s e m } }$ does not replace it either: that row only stops a padding identifier from contributing an embedding gradient.

The recorded last-slice score uses the final sentence axis of the same logits, $L [ : ,$ $\left. T - 1 , \mathrel { \cdot } \right]$ , against $Y \mathrm { { } s }$ last sentence, again ignoring padding. For the reported windows, $T = 4$ , so this slice is $S ( 4 )$

Gradient descent accumulates into $E _ { \mathrm { s e m } }$ along both the Stage A path and the tied readout. $W ^ { \prime }$ receives a gradient only from the readout.

## 4.7 End-to-end training procedure

One training example of shape $( B , T \mathrm { + } 1 , N )$ is processed as follows.

1. Reject the example unless its shape is $( B , T \mathrm { + } 1 , N )$ with $B \geq 1$

2. Split it into the input prefix token ids of shape $( B , T , N )$ and the shifted target Y of shape $( B , T , N )$

3. Build pad mask from token ids by the permutation in section 4.4.

4. Build X by eq. (39).

5. For each of the L layers, replace X by $X ^ { \mathrm { o u t } }$ from section 4.5, reusing pad mask.

6. Apply final norm on $\phi ( n , q )$

7. Form M from $E _ { \mathrm { s e m } }$ and $W ^ { \prime }$ , then the logits $X _ { \mathrm { f l a t } } ^ { \mathrm { n o r m } } M$

8. Permute Y from (B, T, N) to $( B , N , T )$ and evaluate the ignore-index crossentropy against the logits.

Teacher forcing uses ground-truth sentences on the input side for every term of L. A predicted sentence is substituted for a ground-truth sentence only in the diagnostic multi-step rollout, which does not propagate a gradient. At evaluation time the same forward procedure is used with dropout disabled.

## 4.8 Faithfulness checks

The following conditions are part of the embodiment. Each one is a case in which a tensor can have a plausible shape and still implement a diferent map.

1. The activation layout is $( B , N , T , d _ { x } , d _ { t } )$ . The layout $( B , N , d _ { x } , T , d _ { t } )$ coincides with it whenever $T = d _ { x }$

2. $\phi ( n , q ) = n d _ { t } + q$ is shared by the weight view, the MLP, final norm, and M. A swapped fiber $( d _ { t } , d _ { x } )$ has the same product.

3. P broadcasts over t and $q . \ T ^ { \mathrm { e m b } }$ broadcasts over n. The two broadcasts are not interchangeable.

4. pad mask is computed from token identifiers. $X _ { \mathrm { p a d } } = P + T ^ { \mathrm { e m b } } $ is nonzero, so a mask inferred from $\| X \|$ selects the wrong slots.

5. Key scores of padding slots are filled with the smallest finite value of the score dtype. Query slots are unmasked. The MLP receives no mask.

6. pad mask is local to one forward call and is shared by the layers of that call.

7. Each head uses a full $( d _ { x } , d _ { t } )$ fiber. Heads are summed, with no factor $1 / \sqrt { H }$

8. LayerNorm is applied to each branch input and, once more, to the final residual. The residual addends are the unnormalized stream.

9. Dropout with a single rate p is applied to the attention branch and to the MLP branch only.

10. M is materialized from $E _ { \mathrm { s e m } }$ and $W ^ { \prime }$ before multiplication by X. The alternative intermediate of shape $B N T \times d _ { t } \times | V |$ is the same function at much larger activation cost and is not the implemented order.

11. The loss permutes Y from $( B , T , N )$ to $( B , N , T )$ and drops padding targets by the ignore index. The key mask remains in force on the forward pass.

12. Training compares logits to integer targets. The vocabulary softmax and the one-hot target are available as diagnostics and are not called by the training step.

## 5 Reference Implementation

The listings below are the design operations of four tensor model.py, in pipeline order. Shape and range checks, the optimizer wrapper, and diagnostic warnings are omitted. Every arithmetic line is the source statement. Comments are shortened to ASCII.

## 5.1 Input construction

```prolog
1 def _sinusoidal_pe(positions, dim):
2 # even 2i -> sin(pos / 10000^{2i/dim}); odd 2i+1 -> cos; leftover slot stays 0
3 half = dim // 2
4 i = torch.arange(half, device=positions.device, dtype=positions.dtype)
5 div = 10000.0 ** (2 * i / dim)
6 angle = positions.unsqueeze(-1) / div
7 pe = positions.new_zeros(*positions.shape, dim)
8 pe[..., 0:2 * half:2] = torch.sin(angle)
9 pe[..., 1:2 * half:2] = torch.cos(angle)
10 return pe
11
12 # Stage A dictionaries. padding_idx row is zero and receives no gradient.
13 self.E_sem = nn.Embedding(vocab_size, d_x, padding_idx=padding_idx)
14 self.E_temp = nn.Embedding(vocab_size, d_t, padding_idx=padding_idx)
15
16 # Stage B: P_xn, phase = x. Stage D: Temb, phase = x + t.
17 x_idx = torch.arange(N, dtype=torch.float32)
18 self.register_buffer("P", _sinusoidal_pe(x_idx, d_x), persistent=False)
19 tau_idx = torch.arange(T, dtype=torch.float32)
20 phase = x_idx[:, None] + tau_idx[None, :] # (N, T)
21 self.register_buffer("Temb", _sinusoidal_pe(phase, d_t), persistent=False)
22
23 def build_token_tensor(self, token_ids):
24 # token_ids (B, T, N) -> S (B, N, T, d_x, d_t)
25 e_n = self.E_sem(token_ids) # (B, T, N, d_x)
26 e_q = self.E_temp(token_ids) # (B, T, N, d_t)
27 S = e_n.unsqueeze(-1) * e_q.unsqueeze(-2) # outer product (n, q)
28 return S.permute(0, 2, 1, 3, 4).contiguous() # (B, N, T, d_x, d_t)
29
30 def build_input_tensor(self, token_ids):
31 S = self.build token tensor(token ids)
32 S_tilde = S + self.P[None, :, None, :, None] # broadcast over (t, q)
33 return S_tilde + self.Temb[None, :, :, None, :] # broadcast over n
```

## 5.2 Rank-4 attention head

```python
def _rank4_weight(d_x_out, d_x_in, d_t_out, d_t_in):
2 # Xavier on the flat (m p, n q) matrix, then view to (m, p, n, q).
3 mat = torch.empty(d_x_out * d_t_out, d_x_in * d_t_in)
4 nn.init.xavier_uniform_(mat)
5 return nn.Parameter(mat.view(d_x_out, d_t_out, d_x_in, d_t_in).contiguous())
6
7 self.scale = 1.0 / math.sqrt(d_x_out * d_t_out)
8 self.W_Q = _rank4_weight(d_x_out, d_x, d_t_out, d_t)
9 self.W_K = _rank4_weight(d_x_out, d_x, d_t_out, d_t)
10 self.W_V = _rank4_weight(d_x_out, d_x, d_t_out, d_t)
11
12 def _contract(self, X, W):
13 # Y[b,x,t,m,p] = sum_{n,q} W[m,p,n,q] * X[b,x,t,n,q]
14 B, N, T, d_x_in, d_t_in = X.shape
15 d_x_out, d_t_out, _, _ = W.shape
16 X_flat = X.reshape(B * N * T, d_x_in * d_t_in)
17 W_flat = W.view(d_x_out * d_t_out, d_x_in * d_t_in)
18 return F.linear(X_flat, W_flat).view(B, N, T, d_x_out, d_t_out)
19
20 def compute_scores(self, Q, K):
21 # Scores (B, N, T, N, T), joint over the key grid (x’, t’).
22 B, N, T, d_x_out, d_t_out = Q.shape
23 Qf = Q.reshape(B, N * T, d_x_out * d_t_out)
24 Kf = K.reshape(B, N * T, d_x_out * d_t_out)
25 scores = self.scale * torch.bmm(Qf, Kf.transpose(1, 2))
26 return scores.view(B, N, T, N, T)
27
28 def joint_softmax(Scores, pad_mask=None):
29 # Key axes only. finfo.min, so an all-PAD key row stays finite and uniform.
30 B, N, T, _, _ = Scores.shape
```

```python
31 if pad_mask is not None:
32 neg = torch.finfo(Scores.dtype).min
33 Scores = Scores.masked_fill(pad_mask[:, None, None, :, :], neg)
34 return F.softmax(Scores.reshape(B, N, T, N * T), dim=-1).view(B, N, T, N, T)
35
36 def attention_output(self, A, V):
37 B, N, T, d_x_out, d_t_out = V.shape
38 Attn_flat = torch.bmm(A.reshape(B, N * T, N * T),
39 V.reshape(B, N * T, d_x_out * d_t_out))
40 return Attn_flat.view(B, N, T, d_x_out, d_t_out)
41
42 def forward(self, X, pad_mask):
43 Q, K, V = self.build_qkv(X) # three _contract calls
44 A = self.joint_softmax(self.compute_scores(Q, K), pad_mask)
45 return self.attention_output(A, V)
```

## 5.3 Multi-head sum, Pre-LN, and the MLP

```prolog
1 def forward(self, X, pad_mask):
2 # Each head keeps a full (d_x, d_t) fiber. No 1/sqrt(H) factor.
3 Attn = self.heads[0](X, pad_mask)
4 for head in self.heads[1:]:
5 Attn = Attn + head(X, pad_mask)
6 return Attn
7
8 def compute_dff(d_x, d_t):
9 return 4 * d_x * d_t
10
11 # d_in = d_x * d_t; W1 (d_ff, d_in), W2 (d_in, d_ff)
12 self.fc1 = nn.Linear(d_in, d_ff)
13 self.fc2 = nn.Linear(d_ff, d_in)
14 self.ln = nn.LayerNorm(d_in) # MLP branch, distinct from ln_attn
15 self.dropout = nn.Dropout(dropout)
16
17 def residual_after_attention(self, X, Attn):
18 return X + self.dropout(Attn)
19
20 def mlp_block(self, X_out_attn):
21 B, N, T, d_x, d_t = X_out_attn.shape
22 v = self.ln(X_out_attn.reshape(B, N, T, d_x * d_t)).reshape(B * N * T, d_x * d_t)
23 out = self.fc2(F.gelu(self.fc1(v)))
24 return out.view(B, N, T, d_x, d_t)
25
26 def residual_after_mlp(self, X_out_attn, Z):
27 return X_out_attn + self.dropout(Z)
28
29 def _norm_attn(self, X):
30 B, N, T, d_x, d_t = X.shape
31 return self.ln_attn(X.reshape(B, N, T, d_x * d_t)).reshape(B, N, T, d_x, d_t)
32
33 def forward(self, X, pad_mask):
34 # Pre-LN on the attention branch; the MLP pre-norm is inside mlp_block.
35 Attn = self.attention(self._norm_attn(X), pad_mask)
36 return self.mlp(X, Attn)
```

## 5.4 Tied readout, shifted loss, and the model step

```python
self.W_prime = nn.Parameter(torch.empty(d_t, vocab_size))
2 nn.init.xavier_uniform_(self.W_prime)
3 self.final_norm = nn.LayerNorm(d_x * d_t)
4
5 def readout_logits_tied(self, X_output, E_sem):
6 # M[n * d_t + q, v] = E_sem[v, n] * W_prime[q, v], then L = X_flat @ M.
7 B, N, T, d_x, d_t = X_output.shape
8 d = d_x * d_t
9 X_flat = X_output.reshape(B * N * T, d)
```

```python
10 M = torch.einsum("vn,qv->nqv", E_sem, self.W_prime).reshape(d, self.vocab_size)
11 return (X_flat @ M).view(B, N, T, self.vocab_size)
12
13 def build_pad_mask(self, token_ids):
14 # token_ids (B, T, N) -> mask (B, N, T), True where the id is padding.
15 return (token_ids == self.padding_idx).permute(0, 2, 1)
16
17 def forward(self, token_ids):
18 pad_mask = self.build_pad_mask(token_ids)
19 X = self.input_construction(token_ids)
20 for layer in self.layers:
21 X = layer(X, pad_mask)
22 B, N, T, d_x, d_t = X.shape
23 X_norm = self.final_norm(X.reshape(B, N, T, d_x * d_t)).reshape(B, N, T, d_x, d_t)
24 return self.readout(X_norm, self.input_construction.E_sem.weight)
25
26 def build_targets(token_ids_full):
27 # (B, T+1, N) -> Y (B, T, N) = S(1), ..., S(T)
28 return token_ids_full[:, 1:, :].contiguous()
29
30 def cross_entropy_loss(logits, Y, pad_id):
31 # logits (B, N, T, |V|), Y (B, T, N). The permute is the axis bridge.
32 tgt = Y.permute(0, 2, 1).reshape(-1)
33 return F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt,
34 ignore_index=pad_id)
35
36 def last_slice_token_stats(logits, token_ids_full, pad_id):
37 # Reported score: non-padding tokens of S(T) only.
38 last = logits[:, :, -1, :]
39 gold = token_ids_full[:, -1, :]
40 mask = gold != pad_id
41 ce_sum = F.cross_entropy(last.reshape(-1, last.size(-1)), gold.reshape(-1),
42 ignore_index=pad_id, reduction="sum")
43 n_correct = ((last.argmax(dim=-1) == gold) & mask).sum()
44 n_tok = mask.sum()
45 return ce_sum, n_correct, n_tok
46
47 def training_step(self, token_ids_full):
48 token_ids = token_ids_full[:, :-1, :]
49 logits = self.forward(token_ids)
50 Y = build_targets(token_ids_full)
51 loss = cross_entropy_loss(logits, Y, pad_id=self.padding_idx)
52 return loss, logits
```

## 6 4-Tensor Attention Backward Pass and Weight Update Rules

The derivation of the backward pass through the rank-4 attention mechanism requires rigorous tracking of the 4-dimensional index contractions. Because the token-position axes $( x , x ^ { \prime } , y , z )$ and the temporal-slice axes $( t , t ^ { \prime } , u , w )$ interact across multiple nested summations, we compute the exact analytical gradients traversing backward from the raw attention output $y _ { x m ; t p }$ through the joint softmax to the primary weight tensors. The temporal-context fiber is the index $q .$

## 6.1 Joint Softmax Jacobian and Score Tensor Gradient

Let $A _ { y u , y ^ { \prime } u ^ { \prime } }$ denote the joint softmax probabilities computed over the score tensor $E _ { y u , \tilde { y } \tilde { u } }$ . The Jacobian of this transformation with respect to the pre-softmax scores evaluates to:

$$
\frac { \partial A _ { y u , y ^ { \prime } u ^ { \prime } } } { \partial E _ { x t , x ^ { \prime } t ^ { \prime } } } = \delta _ { x y } \delta _ { t u } A _ { x t , y ^ { \prime } u ^ { \prime } } ( \delta _ { x ^ { \prime } y ^ { \prime } } \delta _ { t ^ { \prime } u ^ { \prime } } - A _ { x t , x ^ { \prime } t ^ { \prime } } )\tag{61}
$$

The Kronecker deltas $\delta _ { x y } \delta _ { t u }$ dictate that A depends on E exclusively through the row $E _ { x t , }$ <sub>·</sub> sharing the identical query index $\ ( x , t ) = ( y , u )$ . The joint-softmax rows remain entirely algebraically independent.

Applying the chain rule, we substitute this Jacobian to find the gradient of the loss with respect to the score tensor, denoted $\begin{array} { r } { \nabla E _ { x t , x ^ { \prime } t ^ { \prime } } \equiv \frac { \partial \mathcal { L } } { \partial E _ { x t , x ^ { \prime } t ^ { \prime } } } } \end{array}$ . The deltas collapse the outer summation over $( y , u )$ , yielding the exact row-wise softmax-gradient identity:

$$
\nabla E _ { x t , x ^ { \prime } t ^ { \prime } } = A _ { x t , x ^ { \prime } t ^ { \prime } } \left( { \frac { \partial { \mathcal { L } } } { \partial A _ { x t , x ^ { \prime } t ^ { \prime } } } } - \sum _ { y ^ { \prime } , u ^ { \prime } } { \frac { \partial { \mathcal { L } } } { \partial A _ { x t , y ^ { \prime } u ^ { \prime } } } } A _ { x t , y ^ { \prime } u ^ { \prime } } \right)\tag{62}
$$

For each query $( x , t )$ , the gradient subtracts the A-weighted average of the incoming gradient across the entire key/value grid, subsequently scaled by the local probability $A _ { x t , x ^ { \prime } t ^ { \prime } }$

## 6.2 Gradients with Respect to Q, K, and V

Recall the forward definition of the score tensor: $\begin{array} { r } { E _ { x t , x ^ { \prime } t ^ { \prime } } = \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \sum _ { m , p } Q _ { x m ; t p } K _ { x ^ { \prime } m ; t ^ { \prime } p } . } \end{array}$ Diferentiating L with respect to the query tensor Q yields:

$$
\nabla Q _ { z m ; w p } \equiv \frac { \partial \mathcal { L } } { \partial Q _ { z m ; w p } } = \sum _ { x t , x ^ { \prime } t ^ { \prime } } \nabla E _ { x t , x ^ { \prime } t ^ { \prime } } \left( \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \delta _ { x z } \delta _ { t w } K _ { x ^ { \prime } m ; t ^ { \prime } p } \right)\tag{63}
$$

Collapsing the deltas, we express the gradients for $Q$ and K in slice-matrix notation, where each spatial-temporal position acts as a matrix:

$$
\nabla Q _ { z ; w } = \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \sum _ { x ^ { \prime } , t ^ { \prime } } \nabla E _ { z w , x ^ { \prime } t ^ { \prime } } K _ { x ^ { \prime } ; t ^ { \prime } }\tag{64}
$$

$$
\nabla K _ { z ; w } = \frac { 1 } { \sqrt { d _ { x } d _ { t } } } \sum _ { x , t } \nabla E _ { x t , z w } Q _ { x ; t }\tag{65}
$$

This explicitly reveals the index asymmetry: ∇Q sums $\nabla E$ over its second index pair with the first pair fixed, whereas ∇K sums over its first index pair with the second fixed. This structural asymmetry directly mirrors their respective roles as row and column determinants of the score matrix.

Because the value tensor V bypasses the score computation and enters exclusively at the final attention contraction $\begin{array} { r } { ( y _ { x m ; t p } = \sum _ { x ^ { \prime } , t ^ { \prime } } A _ { x t , x ^ { \prime } t ^ { \prime } } V _ { x ^ { \prime } m ; t ^ { \prime } p } ) } \end{array}$ , its gradient relies directly on the incoming upstream gradient $\nabla y _ { x m ; t p } .$

$$
\nabla V _ { z ; w } = \sum _ { x , t } A _ { x t , z w } \nabla y _ { x ; t }\tag{66}
$$

This operation is the exact transpose of the forward attention contraction: the backward pass distributes $\nabla y$ into $\nabla V$ using A as weights over the query sequence $( x , t )$ for each key/value position $( z , w )$

## 6.3 Chain Rule to the Projection Weights

The upstream gradient $\nabla y _ { x m ; t p }$ arrives at the attention block via the network’s backward pass through the vocabulary readout, LayerNorm, MLP block, and sequential residual connections. Treating ∇Q, ∇K, ∇V as the explicitly derived downstream targets, we backpropagate into the learned weight tensors $W ^ { Q } , W ^ { K } , W ^ { V } \in \mathbb { R } ^ { d _ { x } \times d _ { x } \times d _ { t } \times d _ { t } }$

$Q , K$ , and V are formed from $U = \operatorname { L N } ( X )$ . The forward projection $\begin{array} { r l } { ( \mathrm { e . g . , ~ } Q _ { x m ; t p } = } & { { } } \end{array}$ $\begin{array} { r } { \sum _ { n , q } W _ { m n , p q } ^ { Q } U _ { x n ; t q } ) } \end{array}$ is a strict rank-4 bilinear contraction. Diferentiating this contraction strips out $\hat { U }$ , yielding the analytical gradients for the network weights:

$$
\frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { Q } } = \sum _ { x , t } \nabla Q _ { x m ; t p } U _ { x n ; t q }\tag{67}
$$

$$
\frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { K } } = \sum _ { x , t } \nabla K _ { x m ; t p } U _ { x n ; t q }\tag{68}
$$

$$
\frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { V } } = \sum _ { x , t } \nabla V _ { x m ; t p } U _ { x n ; t q }\tag{69}
$$

Each formulation is a tensor outer product between the upstream gradient (in the $m , p$ space) and $U$ (in the $n , q$ space), summed over positions $( x , t )$ . The gradient continues from U to the residual X through the LayerNorm.

## 6.4 Gradient-Descent Update Rule

The display is plain gradient descent, for exposition; training uses AdamW (section 8.2). Given a learning rate η and the cross-entropy $\mathcal { L } .$ the joint parameter updates are:

$$
W _ { m n , p q } ^ { Q }  W _ { m n , p q } ^ { Q } - \eta \frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { Q } }\tag{70}
$$

$$
W _ { m n , p q } ^ { K }  W _ { m n , p q } ^ { K } - \eta \frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { K } }\tag{71}
$$

$$
W _ { m n , p q } ^ { V }  W _ { m n , p q } ^ { V } - \eta \frac { \partial \mathcal { L } } { \partial W _ { m n , p q } ^ { V } }\tag{72}
$$

## 7 Comparative Baseline: Standard 1-Dimensional Autoregressive Attention

To rigorously evaluate the eficacy of the rank-4 structural decomposition, we establish a standard autoregressive (AR) transformer baseline tasked with the identical generative objective: predicting the future semantic state $\hat { S } ( t )$ exclusively from the historical context window.

For an observation window of $T = 3$ , the baseline model receives the exact same sequence of contextual states $\{ S ( 0 ) , S ( 1 ) , S ( 2 ) \}$ and must generate the subsequent state S(3). Crucially, to maintain a fair comparison and prevent causal leakage, the ground truth target $S ( 3 )$ is strictly isolated from the input, mirroring the exact forward-prediction constraints imposed on the 4-tensor framework.

While the 4-tensor architecture explicitly decomposes the causal surface into independent token-position (x), temporal (t), semantic (n), and contextual (q) axes, a standard AR transformer fundamentally lacks this geometric capacity. Instead, it must concatenate the discrete semantic states into a single, flattened 1-D token sequence. The standard self-attention mechanism $( Q , K , V )$ operates globally over this unidimensional string, relying entirely on 1-D positional encodings to implicitly approximate temporal boundaries.

Consequently, this baseline serves not as an ablation of the 4-tensor mechanism, but as a definitive test of representation: it isolates the structural limits of applying standard flat-sequence attention to a problem domain that inherently demands spatial-temporal and semantic-contextual decomposition.

## 7.1 1-D Sequence Packing and Standard QKV Architecture

To process the structured context X(2) through a standard transformer, the distinct semantic states must be collapsed into a contiguous 1-D token sequence. We eliminate internal padding tokens and concatenate the discrete sentences in temporal reading order. Because standard vocabularies natively encode punctuation, explicit separator tokens between states are unnecessary.

To illustrate this structural flattening, consider a simplified contextual sequence:

Context X(2): [BOS] tom went to the store . he bought milk . then he walked home . [EOS]

Target S(3): [BOS] tom drank the milk . [EOS]

The baseline architecture operates entirely over this flattened 1-D string. The input representations are constructed by summing the shared lexical embeddings with standard 1-D absolute positional encodings:

$$
H _ { i } = E [ v _ { i } ] + \mathrm { P E } ( i )\tag{73}
$$

where $v _ { i }$ represents the token ID at sequence position i. The temporal progression is processed via the standard causal self-attention mechanism:

$$
Y = \mathrm { s o f t m a x } \left( { \frac { ( H W ^ { Q } ) ( H W ^ { K } ) ^ { \top } } { \sqrt { d } } } + M _ { \mathrm { c a u s a l } } \right) ( H W ^ { V } )\tag{74}
$$

The baseline follows the same structural policy as the 4-tensor model: projection weights $W ^ { Q , K , V } ~ \in ~ \mathbb { R } ^ { d \times d }$ , MLP width 4d, and dropout on both residual branches. LayerNorm is applied to the residual stream before attention and before the MLP, and again on the flattened fiber immediately before the tied vocabulary projection $L = H ^ { \mathrm { n o r m } } E ^ { \top }$

During inference, the model is seeded solely with the flattened context sequence [BOS] + X(2). It must then autoregressively emit the predicted sequence $\hat { S } ( 3 )$ stepby-step, appending its own generated tokens until an [EOS] token is produced. Consequently, the baseline processes X(2) as a structurally undiferentiated context, relying entirely on the 1-D causal mask and positional encodings to implicitly approximate the explicit spatial-temporal and semantic-contextual boundaries geometrically enforced by the 4-tensor model.

## 7.2 Informational Parity via Free-Running Autoregressive Decoding

To ensure a rigorous evaluation, the baseline must operate under the exact same informational premise as the 4-tensor framework: the model is conditioned exclusively on the historical context $X ( 2 )$ , with zero exposure to the ground truth tokens of the target state $S ( 3 )$ during inference.

In the 4-tensor architecture, the future state is decoded in a single, parallel forward pass across all spatial positions x:

$$
{ \hat { S } } ( 3 ) _ { x } = f { \big ( } X ( 2 ) { \big ) } \quad { \mathrm { f o r ~ a l l ~ } } x \in \{ 0 , \ldots , N - 1 \}\tag{75}
$$

By contrast, the standard 1-D baseline approximates the target state sequentially. To maintain strict informational parity, this generation must be executed in a freerunning autoregressive mode. Let C denote the flattened context sequence comprising $X ( 2 )$ and the explicit [BOS] token that initiates the target state. The baseline generates the target sequence $\hat { S } ( 3 ) = [ \hat { s } _ { 1 } , \hat { s } _ { 2 } , . . . ]$ step-by-step by continuously appending its own previous predictions:

$$
\operatorname { S t e p } 1 \colon C \longrightarrow { \hat { s } } _ { 1 }\tag{76}
$$

$$
\mathrm { S t e p ~ 2 } \colon { \cal C } \oplus [ \hat { s } _ { 1 } ] \ \longrightarrow \ \hat { s } _ { 2 }\tag{77}
$$

$$
{ \mathrm { S t e p ~ 3 : ~ } } C \oplus \left[ \hat { s } _ { 1 } , \hat { s } _ { 2 } \right] \ \longrightarrow \ \hat { s } _ { 3 }\tag{78}
$$

$$
{ \mathrm { S t e p ~ } } k \colon C \oplus \left[ { \hat { s } } _ { 1 } , \ldots , { \hat { s } } _ { k - 1 } \right] \ \longrightarrow \ { \hat { s } } _ { k }\tag{79}
$$

where ⊕ denotes 1-D sequence concatenation.

Because the prior predictions $\hat { s } _ { < k }$ are deterministically generated from $C$ and the model parameters, they contain no external information regarding the ground truth target. If the model makes an erroneous prediction at step 2, it is strictly forced to condition on that error at step 3. This constraint precisely isolates the fundamental architectural distinction—single-shot parallel structural projection versus 1-D sequential compounding factorization—without confounding the evaluation through asymmetric access to the target sequence.

## 7.3 Architectural Alignment and Capacity Matching

The baseline width is mapped from the 4-tensor fibers. Its model dimension is the joint product $d = d _ { x } d _ { t }$ , and the feed-forward width is $d _ { f f } = 4 d$ . The two models use H diferently. The 4-tensor model adds full-size heads; the baseline splits d across its heads. The runs in section 9 use $H \in \{ 2 , 4 \}$ . Among them, $H = 4$ and $L = 2$ is the closest match in total parameter count.

With $d = 2 0 4 8 .$ , raising H from 2 to 4 adds $6 d ^ { 2 } \approx 2 5 . 2 \mathrm { M }$ parameters per layer to the 4-tensor model and none to the baseline, which is why the baseline count is the same in the first two rows of table 5. At $H = 4 , L = 2$ the totals are close, but they are spent diferently: the 4-tensor model puts 100.7M into attention and 4.7M into its token table, the baseline 33.6M and 75.2M.

The embedding tables difer in shape. The baseline instantiates one table $E \in$ $\mathbb { R } ^ { | V | \times ( d _ { x } d _ { t } ) }$ . The 4-tensor model factorizes that boundary into $E _ { \mathrm { s e m } } \in \mathbb { R } ^ { | V | \times d _ { x } }$ and $E _ { \mathrm { t e m p } }$

Table 3: How a head count enters the parameter count. $d = d _ { x } d _ { t }$ . Biases and LayerNorm are omitted. The baseline attention count is the reported case $H > 1$ , three maps of size $d \times d$ plus $W ^ { O }$ . The 4-tensor count is three full-fiber maps on each of H heads and no output projection. The 4-tensor token term is $E _ { \mathrm { s e m } } , E _ { \mathrm { t e m p } }$ , and the readout factor $W ^ { \prime } .$
<table><tr><td></td><td>4-tensor</td><td>Baseline</td></tr><tr><td>Width of each head</td><td>full  $( d _ { x } , d _ { t } )$  fiber,  $d _ { x } d _ { t }$ </td><td> $d / H$ </td></tr><tr><td>Head merge</td><td>sum, no output projection concatenation, then</td><td> $W ^ { O }$ </td></tr><tr><td>Attention parameters per layer</td><td> $3 H d ^ { 2 }$ </td><td> $4 d ^ { 2 }$ </td></tr><tr><td>MLP parameters per layer</td><td> $8 d ^ { 2 }$ </td><td> $8 d ^ { 2 }$ </td></tr><tr><td>Token table and readout</td><td> $\lvert V \rvert \left( d _ { x } + 2 d _ { t } \right)$ </td><td> $\left( \left| V \right| + 2 \right) d ,$  tied</td></tr></table>

Finally, because the standard baseline lacks the explicit spatial-grid bounding inherently provided by the 4-tensor’s position index x, its vocabulary must be artificially expanded to include explicit [BOS] and [EOS] tokens. These boundary tokens are strictly required to control the start and termination of the 1-D sequence generation, a mechanism entirely obviated by the exact geometric decoding of the 4-tensor architecture.

## 7.4 Baseline design pipeline

The comparison model is a decoder-only causal transformer on one flattened token string. It receives the same stored window $( B , T \mathrm { + } 1 , N )$ as the 4-tensor model and is asked for the same last sentence. It does not build a rank-4 tensor, a joint $( x , t )$ softmax, or a pair of fibers. Width, depth, dropout, and the optimizer are the copies stated in section 8.2: $d = d _ { x } d _ { t } , d _ { f f } = 4 d _ { \cdot }$ L layers, residual dropout $p$ on both branches, AdamW at learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay 0.01.

Packing. Padding identifiers are deleted inside each sentence and the remaining tokens are concatenated in reading order. No separator is inserted; sentence boundaries remain only as whatever punctuation the tokenizer already emitted. Two identifiers are appended after the content vocabulary, bos $= | V |$ and eos $= | V | + 1$ , and the padding index is left unused by them. For a window whose context is $S ( 0 ) , \ldots , S ( T { - } 1 )$ and whose target sentence is $S ( T )$ ,

$$
{ \mathrm { p r e f i x } } = [ { \mathsf { b o s } } ] \oplus { \mathrm { s t r i p } } ( S ( 0 ) ) \oplus \cdots \oplus { \mathrm { s t r i p } } ( S ( T - 1 ) ) ,\tag{80}
$$

$$
\operatorname { g o l d } = \operatorname { s t r i p } ( S ( T ) ) \oplus [ \mathbf { e o s } ] .\tag{81}
$$

Each row is then right-padded with the padding index to the longest row in the batch, and the true lengths $\ell _ { b } ^ { \mathrm { p r e } }$ and $\ell _ { b } ^ { \mathrm { g o l d } }$ are retained. The gold span is a target. It is not written into the prefix.

Token representation. One table $E \in \mathbb { R } ^ { ( | V | + 2 ) \times d }$ embeds every identifier, with the padding row held at zero. Position is a single integer i along the packed string, encoded by the same sinusoid as section 4.3 with phase i and width $d \colon$

$$
H _ { i } ^ { ( 0 ) } = E [ \mathrm { i d } _ { i } ] + \mathrm { P E } ( i ) .\tag{82}
$$

There is no column index x and no sentence index inside the position code.

Causal block. A key at position $j$ is blocked when $j > i$ , and every padding key is blocked. The blocked entries of the score are filled with the smallest finite value of the score dtype, the same guard as the 4-tensor key mask. Heads partition $d .$ With $H = 1$ the maps $\bar { W ^ { Q } } , W ^ { K } , \bar { W ^ { V } } \in \mathbb { R } ^ { d \times d }$ have no output projection and the score scale is $1 / \sqrt { d }$ With $H > 1$ , H must divide $d ,$ each head has width $d _ { h } = d / H$ , the scale is $1 / \sqrt { d _ { h } }$ , and a learned map $W ^ { O } \in \mathbb { R } ^ { d \times d }$ recombines the concatenated heads. The reported runs use $H \in \{ 2 , 4 \}$ , so they include $W ^ { O }$ . One layer is Pre-LN on both branches, with distinct LayerNorms of width d and GELU in the MLP:

$$
H \gets H + \mathrm { D r o p o u t } _ { p } \big ( \mathrm { A t t n } ( \mathrm { L N } _ { d } ^ { ( 1 ) } ( H ) ) \big ) ,\tag{83}
$$

$$
H \gets H + \mathrm { D r o p o u t } _ { p } \big ( \mathrm { M L P } ( \mathrm { L N } _ { d } ^ { ( 2 ) } ( H ) ) \big ) .\tag{84}
$$

The residual stream is the raw sum. After the last layer, one further $\mathrm { L N } _ { d }$ produces $H ^ { \mathrm { n o r m } }$ . Attention still runs on the full packed prefix; only the inputs of attention and of the MLP are normalized.

Readout at the last live position. The vocabulary projection is tied, $L _ { i } \ =$ $H _ { i } ^ { \mathrm { n o r m } } E ^ { \top }$ . In the free-running loop only the logit at the last content position of each row is consumed. Because the tied map has no cross-position term,

$$
\left( H ^ { \mathrm { n o r m } } E ^ { \top } \right) _ { b , i _ { b } } = H _ { b , i _ { b } } ^ { \mathrm { n o r m } } E ^ { \top } , \qquad i _ { b } = \ell _ { b } - 1 .\tag{85}
$$

The implementation therefore normalizes the full sequence, slices that one vector, and multiplies by $E ^ { \top }$ . The earlier positions remain available as keys and values on the next step. The slice changes the activation size, not the function.

Free-running objective. Training and the reported comparison use the same loop. Start from the packed prefix. At step $k = 0 , \ldots , \ell _ { b } ^ { \mathrm { g o l d } } - 1$ , read the logit of the last live position, add the cross-entropy against the gold token at that step, then append argmax of that logit. The appended identifier is detached, so the gradient stops at the logit that chose it and does not pass through the discrete token. If the argmax difers from the gold token, the next step conditions on that own token. Rows that have already exhausted their gold span are left inactive. Two reductions are accumulated. The selection loss averages every gold token, including [EOS]. The quantity placed beside the 4-tensor last-slice score averages only the content tokens of $S ( T )$ , dropping [EOS], on the same non-padding denominator. Evaluation decodes greedily from the prefix alone, stops at [EOS] or after N content tokens, and aligns that string to the gold content for the token accuracy in table 5.

Teacher-forced training, which would place gold $S ( T )$ on the input side of later steps, is a diferent premise. It is not the path implemented in the comparison model, and its cross-entropy is not the number in table 5.

The 4-tensor model receives a loss on all $T$ target sentences of a window, of which only $S ( T )$ is absent from the input; the baseline receives a loss only on the tokens of $S ( T )$

## 7.5 Code for the fair comparison

Ordinary causal self-attention, Pre-LN, and the position-wise MLP are omitted. The statements below are the parts of baseline ar attention model.py that fix the comparison: the same window, a premise that excludes gold S(T), a tied readout that is algebraically unchanged by the last-position slice, and a content-token denominator aligned with the 4-tensor last slice.

```python
1 def compute_d(d_x, d_t):
2 return d_x * d_t # one width, d_ff = 4*d elsewhere
3
4 def bos_eos_ids(content_vocab_size):
5 # Appended after the pool vocabulary. The padding index is not reused.
6 return content_vocab_size, content_vocab_size + 1
7
8 def strip_pad(row, pad_id):
9 return row[row != pad_id]
10
11 # token_ids_full (B, T+1, N). No separator token.
12 parts = [strip_pad(token_ids_full[b, t], padding_idx) for t in range(T + 1)]
13 prefix = torch.cat([bos, *parts[:T]]) # [BOS] + S(0) .. S(T-1)
14 gold = torch.cat([parts[T], eos]) # S(T) + [EOS], target only
1 def forward_last(self, ids, lengths=None):
2 # Full causal stack, then one vector into the tied table.
3 # forward(ids)[b, i] == forward_last(ids, lengths)[b] at i = lengths[b]-1.
4 H = self._encode(ids)
5 if lengths is None:
6 H_last = H[:, -1, :]
7 else:
8 idx = (lengths - 1).clamp(min=0, max=H.shape[1] - 1)
9 H_last = H[torch.arange(H.shape[0], device=ids.device), idx]
10 return F.linear(H_last, self.E.weight)
1 # One free-running step. Gold is the target; the next input is the model’s argmax.
2 last = self.forward_last(ids[:, :L_now], lengths)
3 tgt = packed.gold[:, step]
4 loss_sum = loss_sum + F.cross_entropy(last[active], tgt[active], reduction="sum")
5 is_content = active & (tgt != self.eos_id) # drop EOS: same slots as 4-tensor S(T)
6 content_sum = content_sum + F.cross_entropy(
7 last[is_content], tgt[is_content], reduction="sum")
8 hat = last.argmax(dim=-1).detach()
9 ids[active, write_at[active]] = hat[active]
10
11 def training_step(self, token_ids_full):
12 return self.training_step_fair(token_ids_full)
13
14 # Eval string is (B, N): store content, skip EOS, stop after N tokens.
15 emit = live & (nxt != self.eos_id) & (n_content < self.N)
16 hat[rows, n_content[rows]] = nxt[rows]
17 g = packed.gold[b, : int(packed.gold_len[b])]
18 if g.numel() > 0 and int(g[-1]) == self.eos_id:
19 g = g[:-1] # accuracy gold is S(T) without EOS
20 out[b, :n] = g[:n]
```

## 8 Experimental Setup

The runs below train the Pre-LN 4-tensor model and the free-running 1-D baseline on the same semantic windows. Both place LayerNorm on the residual stream before attention and before the MLP, apply dropout on both residual branches, and apply one further LayerNorm on the fiber immediately before the tied readout. The number reported across the two models is the mean cross-entropy on the non-padding tokens of the last sentence.

## 8.1 Corpus and windows

The corpus is the ROCStories pool of spring 2016, winter 2017, and the accompanying cloze files: 101,671 stories in total. Each stored window has shape $( T + 1 ) \times N = 5 \times 1 6$ The model conditions on $X ( 3 )  \{ S ( 0 ) , S ( 1 ) , S ( 2 ) , S ( 3 ) \}$ . and the recorded score is the cross-entropy of S(4). Sentences are padded to N = 16 with padding index 0; the pad fraction over the pool is 0.3876, and 2,457 sentences were truncated because they exceeded $N = 1 6$ tokens. The vocabulary contains the 36,711 tokens of frequency at least one. The vocabulary is built over the full pool, including the validation split. The baseline adds [BOS] and [EOS], giving vocabulary size 36,713.

Of these windows, 97,929 are the training split and 3,742 are the validation split. Early stopping and checkpoint selection use the validation split, and the tables below report that split. The validation denominator for S(4) is 32,173 non-padding tokens. A uniform distribution over the 36,711-word vocabulary has cross-entropy ln 36711 ≈ 10.51.

## 8.2 Optimization and checkpoint rule

Both models use $d _ { x } = 6 4$ and $d _ { t } = 3 2$ . The baseline therefore has width $d = d _ { x } d _ { t } = 2 0 4 8$ and feed-forward width $d _ { f f } = 8 1 9 2$ . Dropout is 0.1. The optimizer is AdamW with learning rate $3 \times 1 0 ^ { - 4 }$ and weight decay 0.01. Each GPU receives a batch of 32; on two RTX 3080 GPUs the efective batch is 64. Seeds are fixed and deterministic algorithms are enabled. The head count, layer count, and seed of each run are those of table 5.

Training stops when at least five epochs have passed since the minimum validation loss and the current loss is above 1.01 times that minimum, or when the epoch budget is exhausted. The reported checkpoint is the epoch of that minimum. For the 4-tensor model the selection loss is the teacher-forced window cross-entropy on $S ( 1 ) , \ldots , S ( 4 )$ 2 with padding ignored. For the baseline the selection loss is the free-running crossentropy of its own S(4) tokens, including [EOS]. The column placed beside both models is a diferent quantity: the token-weighted mean cross-entropy of the non-padding content tokens of S(4). On the 4-tensor side this is one parallel forward pass, and the gold tokens of $S ( 4 )$ are targets only. On the baseline side the same tokens are scored by free-running decoding from [BOS] plus the flattened context, with [EOS] removed from the denominator. Token accuracy is the argmax match rate on those same 32,173 positions. For every reported 4-tensor setting, the epoch of minimum validation window loss coincides with the epoch of minimum $\mathcal { L } _ { S ( 4 ) }$

## 9 Experiment Data

Table 5 lists the matched Pre-LN settings at the checkpoint selected by the rule in section 8.2. Each entry is written as 4-tensor $/$ baseline. The epoch column is the selected epoch followed by the last logged epoch. The diference is

$$
\Delta = \mathcal { L } _ { S ( 4 ) } ^ { \mathrm { A R } } - \mathcal { L } _ { S ( 4 ) } ^ { 4 \mathrm { T } } .\tag{86}
$$

A positive $\Delta$ means the 4-tensor cross-entropy is lower. Perplexity is $\exp ( \mathcal { L } _ { S ( 4 ) } )$ and is omitted from the tables.

Table 4: Shared setup of the runs.
<table><tr><td>Item</td><td>Value</td></tr><tr><td>Corpus</td><td>ROCStories, window  $5 \times 1 6 ,$  condition on four sentences</td></tr><tr><td>Split</td><td>97,929 train /3,742 validation</td></tr><tr><td>Vocabulary</td><td>36,711; baseline 36,713 with [BOS], [E0S]</td></tr><tr><td>Width</td><td> $d _ { x } = 6 4 , d _ { t } = 3 2 ;$  baseline  $d = 2 0 4 8 , d _ { f f } = 8 1 9 2$ </td></tr><tr><td>Dropout</td><td>0.1 on both residual branches</td></tr><tr><td>Optimizer</td><td>AdamW, learning rate  $3 \times 1 0 ^ { - 4 }$  , weight decay 0.01</td></tr><tr><td>Batch</td><td>32 per GPU, effective batch 64, two RTX 3080</td></tr><tr><td>Selection</td><td>minimum validation loss; stop once five epochs have passed and the loss exceeds  $1 . 0 1 \times$  that minimum</td></tr></table>

Table 5: Matched settings at the selected checkpoint. Each entry is 4-tensor / baseline. Epoch is selected/last. $\mathcal { L } _ { S ( 4 ) }$ is the mean cross-entropy on the 32,173 non-padding tokens of the last sentence, and accuracy is the argmax match rate on the same tokens. $\Delta$ is defined in the text.
<table><tr><td>H</td><td>L</td><td>seed</td><td>parameters</td><td></td><td>epoch</td><td></td><td> $\mathcal { L } _ { S ( 4 ) }$ </td><td></td><td>acc. (%)</td><td> $\Delta$ </td></tr><tr><td>2</td><td>2</td><td>43</td><td></td><td>122.2M / 175.9M</td><td>4/9 / 30/39</td><td></td><td></td><td>5.562 / 5.873</td><td>16.8 / 7.3</td><td></td><td>0.311</td></tr><tr><td>4</td><td>2</td><td>43</td><td></td><td>172.5M / 175.9M</td><td></td><td>6/11 / 45/50</td><td></td><td>5.554 / 5.701</td><td></td><td> $1 6 . 0 ~ / ~ 1 0 . 5$ </td><td>0.147</td></tr><tr><td>4</td><td>3</td><td>42</td><td></td><td>256.4M / 226.2M</td><td></td><td>6/11 / 29/35</td><td></td><td>5.531 / 5.667</td><td>15.4 / 11.1</td><td></td><td>0.136</td></tr></table>

At the selected checkpoints, the mean cross-entropy on $\{ S ( 0 ) , S ( 1 ) , S ( 2 ) , S ( 3 ) \}$ recovered from the logged window loss and $\mathcal { L } _ { S ( 4 ) }$ using the validation token counts, is 0.10–0.12, against 5.53–5.56 on S(4).

Figure 2 shows $\mathcal { L } _ { S ( 4 ) }$ at every logged epoch. On each panel the 4-tensor curve stays below the baseline curve once the baseline has entered the plotted range, including at the lowest baseline point on that trajectory. On each 4-tensor curve the marked epoch is the minimum, and the curve is higher afterward; the stopping rule in section 8.2 ends the run on that rise. Over all logged epochs, the lowest baseline $\mathcal { L } _ { S ( 4 ) }$ is 5.845, 5.694, and 5.667, above the 4-tensor minima 5.562, 5.554, and 5.531. The gaps on that comparison are 0.283, 0.140, and 0.136. Figure 3 is the loss the optimizer steps on, drawn apart from $\mathcal { L } _ { S ( 4 ) }$ because the two are not on the same scale. On the 4-tensor runs it falls from 2.180, 2.178, and 2.521 at epoch 1 to 1.479, 1.435, and 1.446 at the marked checkpoints. $\mathcal { L } _ { S ( 4 ) }$ staying near 5.5 is the term on S(4), the only sentence of X(4) absent from the input X(3); the training loss averages S(4) with S(1), S(2), and S(3). Figure 4 evaluates that same pair of windows on the validation split, input X(3) and target X(4). Its marked points are the minima used for checkpoint selection: 1.305, 1.305, and 1.314 for the 4-tensor runs, and 5.481, 5.310, and 5.297 for the baseline. On each 4-tensor curve the validation loss is higher after the marked epoch.

On the three matched pairs the last-slice cross-entropy is lower for the 4-tensor model: 5.562 against 5.873 at H=2, L=2, 5.554 against 5.701 at H=4, L=2, and 5.531 against 5.667 at H=4, L=3. The H=4, L=2 pair is the closest parameter match, 172.5M against 175.9M, and corresponds to perplexities 258 and 299. On the same two RTX 3080 GPUs the 4-tensor run finished in 2.4 hours and the baseline run in 45.2 hours. The H=2, L=2 pair trains a smaller 4-tensor model, 122.2M against 175.9M, with perplexities 260 and 355. The H=4, L=3 pair trains a larger 4-tensor model, 256.4M against 226.2M, with perplexities 252 and 289. That baseline stopped at epoch 35 of a 100-epoch budget under the rule in section 8.2, and the selected checkpoint is epoch 29. Each cell is a single seed, and the checkpoint was chosen on the validation split reported here.

![](images/8ae4600a3b0fa39ef9dff0b1cc9cb0738f02dd931c0eadce64dfbde2d5a5f9e8.jpg)  
Figure 2: Validation cross-entropy of the non-padding tokens of S(4). This is the sentence absent from the input: the model reads $X ( 3 )  \{ S ( 0 ) , S ( 1 ) , S ( 2 ) , S ( 3 ) \}$ and $S ( 4 )$ is not in that window. The solid curve is the 4-tensor model and the dashed curve is the free-running baseline. Open circles are the checkpoints in table 5. Epoch 1 of each baseline lies above the plotted range (12.9–15.0). The horizontal axis is the epoch index, not wall-clock time: at H=4, L=2 those runs took 2.4 and 45.2 hours.

## 10 Discussion

The rank-4 factorization forces the model to distinguish semantic content from temporal context, keeping the $d _ { x ^ { - } }$ and $d _ { t } \mathrm { - f i b e r s }$ as distinct tensor factors. The separation is imposed only at the boundaries, through rank-1 input embeddings and rank-1 readout functionals, while the internal weight $W _ { m n ; p q }$ remains a general bilinear map that admits arbitrary cross-fiber interaction. Coupling two representation subspaces through a bilinear weight tensor was used in deep tensor neural networks for speech recognition [10]; here the same kind of coupling parameterizes the query, key, and value maps of an attention mechanism over a window indexed jointly by position and time. The architecture does not, however, force transition rules to be abstracted from the objects they apply to; it preserves a representation in which such abstraction can emerge and be tested.

Future directions. This representation makes a further question testable: whether a learner of coupled agent–environment dynamics can acquire, without structural annotation or recursion-specific modules, an operation whose output fills its own argument slot. Nested containment is a minimal setting for this question, and with identical containers the iterated operation takes the form of Zermelo’s successor construction,

![](images/611ee2afa9b97bd162fe18441952ede333cc2873428dc0850e5bba3e1ce44b7b.jpg)  
Figure 3: Training cross-entropy on separate axes. The upper row is the 4-tensor model trained in these experiments. On a training window the input is $X ( 3 ) $ $\{ S ( 0 ) , S ( 1 ) , S ( 2 ) , S ( 3 ) \}$ , and the loss is the mean cross-entropy of the shifted window $X ( 4 )  \{ S ( 1 ) , S ( 2 ) , S ( 3 ) , S ( 4 ) \}$ . The lower row is the baseline, whose loss is the free-running loss on S(4), including [EOS]. Open circles are the checkpoints in table 5. Epoch 1 of each baseline lies above the plotted range (29.1–32.6). The horizontal axis is the epoch index, not wall-clock time.

$0 = \emptyset , n { + } 1 = \{ n \}$ [8]. We develop this program, including criteria under which it would be refuted, in a separate document.

A multi-step rollout, with parameters held fixed, would score sentences from S(T) onward under free-running inputs. For an evaluation horizon $T _ { \mathrm { e v a l } } \geq T$ 7

$$
\mathcal { L } _ { \mathrm { e v a l } } = \sum _ { t = T } ^ { T _ { \mathrm { e v a l } } } \sum _ { x = 0 } ^ { N - 1 } \sum _ { v \in \mathcal { V } } - y _ { x v } ( t ) \log \hat { p } _ { x v } ( t ) ,\tag{87}
$$

where t indexes the sentence $S ( t ) , x \in \{ 0 , \ldots , N - 1 \}$ , and $\hat { p } _ { x v } ( t )$ conditions on earlier predictions rather than on the ground-truth sentence. No value of this metric is reported. Denoising may be performed on interpolations between generated images.

Patent Notice The methods described in this paper are the subject of a pending Korean patent application (Application No. 10-2026-0191535, filed October 6, 2026) by IndigoWave Inc. Researchers interested in using these methods are welcome to contact the authors.

![](images/c5815658a52b355eefbcffc27942cd477e58cc03c4180b60af33a8e523b8255b.jpg)  
Figure 4: Validation cross-entropy on separate axes. The upper row is the 4- tensor model on the validation split, with the same windows as fig. 3: the input is $X ( 3 )  \{ S ( 0 ) , S ( 1 ) , S ( 2 ) , S ( 3 ) \}$ , and the loss is the mean cross-entropy of $X ( 4 )  \{ S ( 1 ) , S ( 2 ) , S ( 3 ) , S ( 4 ) \}$ . The lower row is the baseline selection loss on S(4), including [EOS]. Open circles are the checkpoints in table 5, which are the minima of these curves. Every logged epoch lies inside the plotted range. The horizontal axis is the epoch index, not wall-clock time.

## References

[1] Brooks, T., Peebles, W., et al. (2024). Video generation models as world simulators. OpenAI.

[2] Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104. (DreamerV3)

[3] Brohan, A., Brown, N., Carbajal, J., et al. (2023). RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL).

[4] LeCun, Y. (2022). A path towards autonomous machine intelligence version 0.9.2. OpenReview. (JEPA Proposal)

[5] Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., et al. (2024). Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471. (V-JEPA)

[6] Yu, L., Lezama, J., et al. (2023). Language Model Beats Difusion – Tokenizer is Key to Visual Generation. In International Conference on Learning Representations (ICLR). (MAGVIT-v2)

[7] Kondratyuk, D., Yu, L., Gu, X., et al. (2023). VideoPoet: A large language model for zero-shot video generation. arXiv preprint arXiv:2312.14125.

[8] Zermelo, E. (1908). Untersuchungen ¨uber die Grundlagen der Mengenlehre I. Mathematische Annalen, 65, 261–281.

[9] Arnab, A., Dehghani, M., Heigold, G., Sun, C., Luˇci´c, M., & Schmid, C. (2021). ViViT: A video vision transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 6836–6846. (ViViT)

[10] Yu, D., Deng, L., & Seide, F. (2013). The deep tensor neural network with applications to large vocabulary speech recognition. IEEE Transactions on Audio, Speech, and Language Processing, 21(2), 388–396. (DTNN)

[11] Yu, H., Wang, J., Huang, Z., Yang, Y., & Xu, W. (2016). Video paragraph captioning using hierarchical recurrent neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 4584–4593.

[12] Ho, J., Chan, W., Saharia, C., Whang, J., Gao, R., Gritsenko, A., et al. (2022). Imagen Video: High definition video generation with difusion models. arXiv preprint arXiv:2210.02303. (Imagen Video)