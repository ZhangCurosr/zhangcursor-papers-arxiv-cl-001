# ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models

Shijie Lian<sup>1,</sup> <sup>2,</sup> <sup>3</sup>, Bin Yu<sup>4,</sup> <sup>2,</sup> <sup>3</sup>, Zhaolong Shen<sup>5,</sup> <sup>2,</sup> <sup>3</sup>, Xiaopeng Lin<sup>6,</sup> <sup>3</sup>, Yichao Du<sup>3</sup>, Zhirui Zhang<sup>3,</sup> <sup>‡</sup>, Laurence T. Yang<sup>1,</sup> <sup>7,</sup> <sup>†</sup>, Kai Chen<sup>8,</sup> <sup>3,</sup> <sup>†</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Zhongguancun Academy, <sup>3</sup>DeepCybo, <sup>4</sup>Harbin Institute of Technology, <sup>5</sup>Beihang University, <sup>6</sup>The Hong Kong University of Science and Technology (Guangzhou), <sup>7</sup>Zhengzhou University, <sup>8</sup>Zhongguancun Institute of Artificial Intelligence

## Abstract

Action tokenizers play a central role in autoregressive vision-language-action (VLA) models, determining both the targets for policy training and the executable commands recovered from predicted tokens. Their fidelity is commonly evaluated using pointwise reconstruction metrics such as mean squared error (MSE), yet small individual errors do not fully characterize how faithfully action adjustments across demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while the adjustments needed for diferent contexts are diminished, distorted, or even reversed. We introduce physical rank consistency (PRC) to measure how well tokenization preserves local physical distance rankings after reconstruction. Evaluating decoded actions provides a common reference across token vocabularies and decoder architectures, complementing pointwise accuracy with a measure of relational fidelity. We further present ActionPiece, which preserves physical action relationships through joint supervision of representation learning and quantization. Physical rank preservation supervises near–far ordering in encoder and quantized feature distances, while quantization regularization applies the same ordering to codeword assignment distributions. Both objectives augment reconstruction, producing discrete action tokens for standard autoregressive policy learning and execution through a frozen decoder. Under the same Qwen3-VL-4B policy training setup, ActionPiece achieves 94.8% on LIBERO and 68.8% on unseen LIBERO-Plus, with additional evaluations reaching 71.9% on SimplerEnv and 51.5% across VLA-Arena L0–L2. Component ablations show that the two objectives jointly improve PRC and policy success, demonstrating the value of physical relationship supervision for action tokenization.

Project Page: https://deepcybo-physai.github.io/ActionPiece/

## 1 Introduction

Autoregressive vision-language-action (VLA) models generate discrete tokens that are decoded into executable robot commands [5, 15, 31]. The action tokenizer connects continuous demonstrations to this discrete prediction process: it determines both the targets used for policy training and the commands recovered from predicted tokens. Existing approaches construct this interface through coordinate discretization, frequencydomain compression, or learned vector quantization [8, 15, 23, 24, 27, 31]. These designs enable compact action generation, but compression also changes the actions that the robot ultimately executes. An important question is therefore what an action tokenizer should preserve to support precise control.

Across multiple demonstrations of the same task, similar motions recur with adjustments to object positions, robot states, and execution stages. These adjustments allow similar motions to accommodate diferent contexts. These adjustments allow similar motions to accommodate diferent contexts. In operations requiring precise alignment, grasping, or contact, even small action diferences can afect execution outcomes. A tokenizer compresses these demonstrations into a discrete representation with finite capacity and reconstructs executable commands from it. Although reconstruction objectives such as mean squared error (MSE) keep individual actions close to their originals, they do not explicitly coordinate reconstruction errors across demonstrations. Small individual reconstruction errors therefore do not fully characterize how faithfully the adjustments between demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while the adjustments needed to accommodate diferent contexts are distorted, diminished, or even reversed. These changes persist even when a policy correctly predicts the corresponding token sequence: the decoder still executes the reconstructed action.

This motivates studying relational fidelity alongside pointwise reconstruction accuracy. We characterize action diferences using a physical distance that combines translation, rotation, and gripper state, and examine whether their relative magnitudes survive tokenization. To measure this property, we introduce physical rank consistency (PRC). For each action chunk, PRC compares its distance rankings to the same original neighbors before and after reconstruction. Computing these rankings in decoded action space provides a common physical reference across token vocabularies, sequence lengths, and decoder architectures. Together, reconstruction accuracy and PRC describe how faithfully a tokenizer reproduces individual commands and preserves the local relationships among them.

To preserve these relationships during tokenization, we develop ActionPiece. Physical distances between original action chunks provide near–far supervision for both representation learning and codeword assignment. Physical rank preservation (PRP) encourages the corresponding ordering in a combination of encoder and quantized feature distances. Quantization regularization (QR) applies this ordering to the divergence between codeword assignment distributions, extending supervision to how actions are assigned to the discrete vocabulary. Together, these objectives address two stages of compression: learning action features and mapping them to codewords. They are optimized jointly with reconstruction in a Transformer tokenizer with residual vector quantization. After tokenizer training, the frozen encoder and quantizer supply discrete policy targets, while the frozen decoder converts autoregressively predicted tokens into executable action chunks.

Under a shared Qwen3-VL-4B policy training setup, ActionPiece outperforms all compared action tokenizers, achieving 94.8% success on LIBERO and 68.8% on unseen LIBERO-Plus, exceeding the strongest baseline on each benchmark by 1.1 and 4.5 percentage points, respectively (Table 1). Evaluations on SimplerEnv and VLA-Arena extend the comparison to real-to-sim transfer and challenging control conditions, where ActionPiece achieves state-of-the-art aggregate success rates of 71.9% and 51.5%, respectively, among the evaluated methods. These results are obtained through standard next-token prediction using ActionPiece as the action tokenizer. Controlled ablations show that PRP and QR jointly improve both decoded physical rank consistency and downstream policy success. Comparisons across tokenizers further examine the relationship between reconstruction fidelity, PRC, and execution performance (Figure 2).

Our contributions are threefold:

• We introduce PRC, a measure of local physical distance-order preservation that complements reconstruction accuracy and supports comparison across action tokenizers.

• We develop ActionPiece, jointly supervising physical relationships in learned features and codeword assignment distributions to construct discrete action representations for autoregressive VLA policies.

• We evaluate ActionPiece across four benchmarks and conduct controlled comparisons and ablations to examine the efects of physical relationship supervision on representation quality and robot execution.

## 2 Related Work

## 2.1 Action-Space Design in Robotic Foundation Models

Robot foundation models connect pretrained vision–language representations to motor commands through discrete, continuous, or hybrid action interfaces. Discrete autoregressive approaches cast control as next-token prediction: RT-2 and OpenVLA discretize action coordinates into bins [5, 15], while FAST compresses action chunks into shorter token sequences [31]. VLM2VLA instead expresses commands using the VLM’s existing natural-language vocabulary [11]. These interfaces retain the token-prediction objective of pretrained VLMs, providing a direct route to transfer language grounding and semantic knowledge to robot control.

Continuous approaches predict real-valued action chunks without categorical decoding. OpenVLA-OFT uses parallel action prediction with an $\ell _ { 1 }$ regression objective [16]; difusion and flow-matching policies, including RDT, $\pi _ { 0 } ,$ and the GR00T family, model continuous action distributions [4, 26, 29, 35]. Continuous heads support fine-grained control, and generative heads can represent multiple valid motions while generating action chunks without autoregressively decoding every action token.

Hybrid designs combine discrete supervision with continuous execution. $\pi _ { 0 . 5 }$ uses action tokens during pretraining and introduces a flow-matching action expert during post-training [12]. Knowledge Insulation trains the backbone with discrete action targets while blocking gradients from the continuous expert, retaining pretrained knowledge alongside eficient execution [9]. HybridVLA jointly trains autoregressive and difusion predictions and adaptively combines them [25]; Fast-in-Slow couples autoregressive supervision with a fast difusion-based execution module [6]. Such designs seek to combine the semantic benefits of token prediction with responsive continuous control.

## 2.2 Action Discretization and Tokenization

Coordinate-wise binning provides a simple action vocabulary, as in RT-2 and OpenVLA [5, 15], but represents each coordinate and time step separately. FAST applies a discrete cosine transform to action chunks, quantizes the coeficients, and learns byte-pair encoding (BPE) to compress the resulting sequences [31]. Its frequency basis is fixed, while the BPE vocabulary is learned from action data.

Learned latent tokenizers instead optimize an encoder and decoder around a quantized representation. Vectorquantized approaches learn codebooks from action data; FASTer combines residual vector quantization (RVQ), structured action patching, temporal and spectral reconstruction, and blockwise autoregressive generation [27]. OAT uses finite scalar quantization with nested dropout and causal attention to obtain compact, fully decodable tokens with coarse-to-fine ordering [23, 24]. ActionCodec studies overlap, token budget, vision–language alignment, and residual grammar from the perspective of VLA optimization [8]. X-Tokenizer adds semantic supervision from a frozen visual–language teacher [14]. These methods develop action interfaces through compression, token ordering, and multimodal alignment. ActionPiece introduces explicit physical-order supervision during encoding and quantization, and uses PRC to measure neighborhood-order preservation in decoded action space.

## 3 Physical Structure in Action Tokenization

## 3.1 Preserving Action Variations

Demonstrations of the same task contain recurring motions with adjustments to object positions, robot states, and execution stages. These adjustments allow similar motions to accommodate diferent contexts. In operations requiring precise alignment, grasping, or contact, even small action diferences can afect execution outcomes.

An action tokenizer compresses continuous action chunks into discrete token sequences with finite representational capacity, then decodes them into executable commands. This lossy compression introduces reconstruction errors. Although tokenizers are commonly trained with an MSE objective to keep reconstructed actions close to their originals, this objective penalizes individual reconstruction errors without explicitly coordinating their directions across actions. Consider the normalized Euclidean components $x _ { i } , x _ { j }$ of two demonstration chunks, with reconstruction errors $e _ { i } = { \widehat { x } } _ { i } - x _ { i }$ . Their reconstructed diference satisfies

$$
\widehat { x } _ { j } - \widehat { x } _ { i } = \big ( x _ { j } - x _ { i } \big ) + \big ( e _ { j } - e _ { i } \big ) .\tag{1}
$$

The first term is the action variation present in the demonstrations; the second is the change introduced by tokenization. This additional term can attenuate the original diference, amplify it, or even reverse its direction. Small individual reconstruction errors therefore do not fully characterize how faithfully the adjustments between demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while their original near-to-far ordering is altered.

These distortions carry through to policy execution. Even when a VLA correctly predicts the token sequence encoding a demonstrated action, the decoder still produces its reconstructed version. If compression weakens or distorts the adjustments required by diferent contexts, correct token prediction cannot recover the original adjustments. Action tokenization should therefore preserve physical distinctions among demonstrations alongside accurate reconstruction of individual actions.

## 3.2 Physical Rank Consistency

We assess relationship preservation through the relative ordering of physical distances between action chunks. Our distance $d _ { \mathrm { p h y s } }$ combines translation, shortest geodesic rotation on SO(3), and gripper diferences across a chunk, as defined in Section 4.2. For each anchor $A _ { i } ,$ , let $\mathcal { N } _ { i } ^ { k }$ contain its k nearest neighbors in the original action space, and let $\widehat { A } _ { i } = \mathcal { D } ( \mathcal { T } ( A _ { i } ) )$ . Physical Rank Consistency (PRC) compares distances to these same neighbors before and after reconstruction:

$$
\mathrm { P R C @ } k = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \rho _ { \mathrm { S } } \left( [ d _ { \mathrm { p h y s } } ( A _ { i } , A _ { j } ) ] _ { j \in { \mathcal N } _ { i } ^ { k } } , [ d _ { \mathrm { p h y s } } ( \widehat { A } _ { i } , \widehat { A } _ { j } ) ] _ { j \in { \mathcal N } _ { i } ^ { k } } \right) ,\tag{2}
$$

where $\rho _ { \mathrm { S } }$ denotes Spearman correlation. We use $k = 3 2$

Higher PRC indicates better preservation of distance ordering within the original neighborhoods. Comparing ranks captures the relative size of action diferences without requiring every distance to remain numerically identical. Evaluation in decoded action space provides a common physical reference across token vocabularies, sequence lengths, and decoder architectures. MSE measures the fidelity of individual reconstructions, while PRC measures the preservation of physical distance ordering. Figure 2 examines both properties in relation to downstream policy success. Guided by this perspective, ActionPiece introduces physical-order supervision into representation learning and codeword assignments, preserving action relationships alongside pointwise reconstruction.

## 4 ActionPiece

ActionPiece preserves physical action relationships alongside reconstruction accuracy through joint supervision of encoding and quantization, following Section 3. A shallow Transformer encoder and decoder with residual vector quantization (RVQ) provide the codec. Physical rank preservation structures the learned action representations; quantization regularization carries the same ordering into codeword assignments. Both augment reconstruction during tokenizer training. The resulting vocabulary supplies the targets for standard autoregressive policy learning. Figure 1 summarizes the tokenizer training objectives and the resulting policy learning and execution pipeline.

## 4.1 Compact, Fully Decodable Tokens

For an action chunk $A = \left( a _ { 1 } , \ldots , a _ { H } \right) \in \mathbb { R } ^ { H \times D }$ , the encoder $E _ { \phi }$ produces S latent slots. An RVQ with depth Q and K entries per codebook represents slot s as

$$
q _ { s } = \sum _ { r = 1 } ^ { Q } e ^ { ( r ) } [ z _ { s } ^ { ( r ) } ] , \qquad Z = \{ z _ { s } ^ { ( r ) } \} _ { s = 1 , r = 1 } ^ { S , Q } , \qquad | Z | = S Q = T .\tag{3}
$$

The decoder reconstructs the complete chunk in one pass, $\widehat { A } = D _ { \omega } ( q )$ . Here $T$ is the number of categorical outputs generated by the VLA, while $Q$ controls how the budget is divided between latent slots and residual levels. Every code sequence maps to a complete action chunk.

![](images/e0a8bc0a0f35c4d2c8c4d1dcd72d49f430e9e62ea99b65b00b79d389b4c47930.jpg)  
Figure 1 Overview of ActionPiece. Physical action distance (defined in Section 4.2) combines normalized translation, shortest-path rotation on SO(3), and gripper diferences to select anchor, near, and far action chunks within a training batch. The tokenizer compresses each eight-step chunk into 16 discrete tokens and reconstructs executable actions. Alongside reconstruction, physical rank preservation (PRP, defined in Section 4.3) encourages the near–far ordering in encoder and quantized feature distances, while quantization regularization (QR, defined in Section 4.4) encourages the same ordering in the Jensen–Shannon divergence between codeword assignment distributions. Here $L _ { 2 }$ denotes the slot-averaged squared distance between normalized features.

## 4.2 Physical Action Distance

For two end-efector commands $a = ( p , R , g )$ and $a ^ { \prime } = ( p ^ { \prime } , R ^ { \prime } , g ^ { \prime } )$ , we measure rotational distance by the shortest geodesic angle on SO(3):

$$
d _ { \mathrm { S O ( 3 ) } } ( R , R ^ { \prime } ) = \left\| \mathrm { L o g } ( R ^ { \top } R ^ { \prime } ) ^ { \vee } \right\| _ { 2 } ,\tag{4}
$$

where $\mathrm { L o g } ( \cdot ) ^ { \vee }$ is the rotation-vector form of the principal matrix logarithm. The per-step physical distance combines translation, rotation, and gripper state:

$$
\delta _ { \mathrm { p h y s } } ( a , a ^ { \prime } ) = \alpha _ { p } \left. \frac { p - p ^ { \prime } } { s _ { p } } \right. _ { 2 } ^ { 2 } + \alpha _ { R } \left( \frac { d _ { \mathrm { S O } ( 3 ) } ( R , R ^ { \prime } ) } { s _ { R } } \right) ^ { 2 } + \alpha _ { g } | g - g ^ { \prime } | ^ { 2 } .\tag{5}
$$

The scales are fitted on training actions, and the group weights are fixed within an embodiment. For action chunks, $d _ { \mathrm { p h y s } }$ averages these per-step distances over corresponding time steps. Rotations are mapped to valid elements of $\mathrm { S O ( 3 ) }$ before computing the distance. This distance defines the near-to-far ordering used by the two training objectives and by PRC in decoded action space.

## 4.3 Physical Rank Preservation

Physical rank preservation aligns the relative distances of learned action representations with those of physical motions. We measure representation distance both before and after quantization. Let $\bar { h } _ { i , s }$ be the encoder representation of action chunk $A _ { i }$ at slot $s ,$ and let $\bar { e } _ { i , s }$ be its selected codeword after the quantizer output

projection. Both are normalized by LayerNorm followed by unit $\ell _ { 2 }$ normalization. Their distances are

$$
d _ { \mathrm { e n c } } ^ { 2 } ( i , j ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \Vert \bar { h } _ { i , s } - \bar { h } _ { j , s } \Vert _ { 2 } ^ { 2 } ,\tag{6}
$$

$$
d _ { \mathrm { q u a n t } } ^ { 2 } ( i , j ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \Vert \bar { e } _ { i , s } - \bar { e } _ { j , s } \Vert _ { 2 } ^ { 2 } .
$$

We use the weighted representation distance

$$
d _ { \mathrm { r e p } } ^ { 2 } ( i , j ) = 0 . 2 5 d _ { \mathrm { e n c } } ^ { 2 } ( i , j ) + 0 . 7 5 d _ { \mathrm { q u a n t } } ^ { 2 } ( i , j ) , \qquad D _ { i j } = \sqrt { d _ { \mathrm { r e p } } ^ { 2 } ( i , j ) } .\tag{7}
$$

This distance incorporates the encoder features and the discrete representations used by the decoder. A diferentiable codeword estimator transmits gradients through code selection; Appendix B.1 specifies the estimator, and Appendix B.2 covers multiple quantization levels.

For each anchor action chunk $A _ { i } ,$ we exclude the anchor itself and rank the remaining action chunks in the training batch by physical distance. We select the positive (near) action $A _ { j _ { i } ^ { + } }$ at the 5th percentile $\left( q _ { 0 . 0 5 } \right)$ and the negative (far) action $A _ { j _ { i } ^ { - } }$ at the 95th percentile $\left( q _ { 0 . 9 5 } \right)$ . The rank objective is

$$
\mathcal { L } _ { \mathrm { r a n k } } = \frac { 1 } { B } \sum _ { i } \mathrm { s o f t p l u s } \Bigl ( D _ { i j _ { i } ^ { + } } - D _ { i j _ { i } ^ { - } } + m _ { r } \Bigr ) , \qquad m _ { r } = 0 . 1 .\tag{8}
$$

The loss encourages representations to retain the near-to-far order of physical motions. Both physical-order objectives use these pairs; Appendix B.3 specifies the selection rule.

## 4.4 Quantization Regularization

Quantization regularization applies physical-order supervision directly to codeword probabilities. Let $\pi _ { i , s }$ denote the soft probability distribution over codewords for slot s of action chunk $A _ { i } ,$ obtained from encoder-tocodeword distances as specified in Eq. (15). We compare two action chunks using the average Jensen–Shannon divergence between corresponding slots:

$$
d _ { \mathrm { J S } } ( \it { i } , \it { j } ) = \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \mathrm { J S } ( \pi _ { i , s } , \pi _ { j , s } ) ,\tag{9}
$$

where $\begin{array} { r } { \mathrm { J S } ( p , q ) = \frac { 1 } { 2 } \mathrm { K L } ( p \| \mu ) + \frac { 1 } { 2 } \mathrm { K L } ( q \| \mu ) } \end{array}$ and $\mu = ( p + q ) / 2$ . Using the same neighbors $j _ { i } ^ { + }$ and $j _ { i } ^ { - }$ as the rank objective, we define

$$
{ \mathcal { L } } _ { \mathrm { q u a n t } } = { \frac { 1 } { B } } \sum _ { i } { \mathrm { s o f t p l u s } } \left( d _ { \mathrm { J S } } ( i , j _ { i } ^ { + } ) - d _ { \mathrm { J S } } ( i , j _ { i } ^ { - } ) + m _ { q } \right) , \qquad m _ { q } = 0 . 1 .\tag{10}
$$

Physical rank preservation supervises distances between action representations, while quantization regularization supervises the probabilities that determine codeword selection. Together, they impose physical order on these two aspects of the tokenizer.

## 4.5 Training and Policy Integration

The reconstruction objective measures mean squared error in normalized action coordinates:

$$
\mathcal { L } _ { \mathrm { M S E } } = \frac { 1 } { B H D } \sum _ { i , t , d } \left( \frac { A _ { i , t , d } - \widehat { A } _ { i , t , d } } { s _ { d } } \right) ^ { 2 } .\tag{11}
$$

The complete tokenizer objective combines reconstruction, the standard codebook commitment loss, and the two physical-order objectives:

$$
{ \mathcal { L } } _ { \mathrm { t o k } } = { \mathcal { L } } _ { \mathrm { M S E } } + \lambda _ { \mathrm { v q } } { \mathcal { L } } _ { \mathrm { c o m m i t } } + \lambda _ { r } { \mathcal { L } } _ { \mathrm { r a n k } } + \lambda _ { q } { \mathcal { L } } _ { \mathrm { q u a n t } } .\tag{12}
$$

![](images/37e5e79e12608ea382b2e7949821b68511610e152eef274899aa4018e3dc3822.jpg)  
Figure 2 Reconstruction fidelity, physical structure, and policy success. Panels (a–b) show normalized within-group ranks across 55 tokenizer–benchmark evaluations. Groups follow the original experimental batches, with benchmarks evaluated separately. Colors indicate benchmarks; symbols identify the methods shown in (c), and crosses denote additional variants. Panel (c) reports LIBERO-Plus success for selected tokenizers and ablations using Qwen3-VL-4B at 30K policy training steps. PRP and QR denote physical rank preservation and quantization regularization.

We set $\lambda _ { \mathrm { v q } } = 0 . 2 5$ and $\lambda _ { r } = \lambda _ { q } = 6 . 2 5 \times 1 0 ^ { - 4 }$ . The component ablation in Table 4 removes either physical-order objective or both from this formulation. PRC evaluates the physical neighborhood structure of decoded actions after training.

After training, the tokenizer is frozen and its indices are added to the VLM vocabulary. The VLA is optimized with standard next-token prediction,

$$
\mathcal { L } _ { \mathrm { V L A } } = - \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( z _ { t } \mid I , \ell , z _ { < t } ) ,\tag{13}
$$

and the generated sequence is decoded into an executable action chunk.

## 5 Experiments

We relate tokenizer fidelity and physical structure to policy success, then test ActionPiece through controlled comparisons, transfer evaluations, and ablations.

## 5.1 Experimental Protocol

Benchmarks and data. We select four benchmarks spanning three action-data sources and complementary generalization settings. LIBERO [22] provides the in-distribution reference, using human teleoperation demonstrations collected in simulation. LIBERO-Plus [10] evaluates these LIBERO-trained policies under seven types of unseen perturbations, without training on LIBERO-Plus data. SimplerEnv [19] evaluates a WidowX policy trained on real-robot BridgeData V2 demonstrations [38], testing real-to-sim transfer. For VLA-Arena [40], we use keyboard-teleoperated simulation demonstrations at L0 and evaluate across L0–L2, with L1/L2 testing generalization to more demanding task configurations. Together, these settings test tokenizers across diferent demonstration sources and assess policy performance both within and beyond the training distribution.

Tokenizer training. Action data are sampled at 20 Hz for LIBERO, 10 Hz for VLA-Arena, and 5 Hz for BridgeData V2. Tokenizers are trained on action chunks using AdamW [28] with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a batch size of 128 for 100K steps. For fair comparison, each baseline tokenizer is trained using the remaining optimizer hyperparameters specified in its original work. Within each controlled comparison, the action representation, normalization, and execution settings are fixed.

Table 1 Matched action-tokenizer comparison on LIBERO and LIBERO-Plus. All policies use the same Qwen3-VL-4B backbone, training data, prompt, vision–language co-training procedure, global batch size 128, an 8-step prediction and execution horizon, and maximum policy-training budget. LIBERO-Plus is evaluated out of distribution: no LIBERO-Plus demonstrations are used for policy training. FASTerVQ<sup>∗</sup> denotes our implementation following the published method. Best results are bold.
<table><tr><td></td><td colspan="5">LIBERO</td><td colspan="9">LIBERO-Plus</td></tr><tr><td>Tokenizer</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Overall</td><td>Camera</td><td>Robot</td><td>Language</td><td></td><td>Light</td><td>Bkg.</td><td>Noise</td><td>Layout</td><td>Overall</td></tr><tr><td>ActionCodec [8]</td><td>93.4</td><td>99.0</td><td>92.4</td><td>89.8</td><td>93.7</td><td>33.3</td><td>38.8</td><td></td><td>80.8</td><td>88.4</td><td>90.2</td><td>58.0</td><td>75.9</td><td>64.2</td></tr><tr><td>OAT [23, 24]</td><td>85.6</td><td>96.2</td><td>87.2</td><td>75.2</td><td>86.1</td><td>32.2</td><td></td><td>47.8</td><td>79.5</td><td>82.8</td><td>82.7</td><td>46.4</td><td>67.4</td><td>60.7</td></tr><tr><td>FASTerVQ* [27]</td><td>91.8</td><td>95.6</td><td>90.5</td><td>87.4</td><td>91.3</td><td>39.0</td><td>42.3</td><td></td><td>74.5</td><td>85.8</td><td>84.9</td><td>56.6</td><td>69.0</td><td>62.6</td></tr><tr><td>FAST [31]</td><td>94.7</td><td>98.0</td><td>93.1</td><td>82.6</td><td>92.1</td><td>33.8</td><td></td><td>47.1</td><td>80.0</td><td>89.6</td><td>88.4</td><td>51.3</td><td>75.9</td><td>64.3</td></tr><tr><td>Standard RVQ</td><td>91.8</td><td>98.2</td><td>91.6</td><td>82.0</td><td>90.9</td><td>31.5</td><td>43.9</td><td></td><td>75.3</td><td>86.0</td><td>88.0</td><td>42.5</td><td>72.9</td><td>60.4</td></tr><tr><td>ActionPiece (Ours)</td><td>96.2</td><td>99.0</td><td>93.4</td><td>90.6</td><td>94.8</td><td>45.1</td><td></td><td>49.6</td><td>82.2</td><td>91.9</td><td>91.4</td><td>57.6</td><td>78.0</td><td>68.8</td></tr></table>

VLA training. All experiments use eight NVIDIA RTX PRO 6000 GPUs. For policy training, we follow the default StarVLA protocol [7] and use AdamW [28] with an initial learning rate of $1 \times 1 0 ^ { - 5 }$ and a cosine annealing schedule. We use DeepSpeed ZeRO-2 [33], gradient clipping with a maximum norm of 1.0, and no gradient accumulation.

Matched policy evaluation. Within each controlled comparison, the VLM backbone, prompt, robot demonstrations, optimizer, global batch, training budget, action and execution horizons, seed, and evaluation protocol are fixed. Full LIBERO contains 2,000 rollouts, full LIBERO-Plus 10,030, and complete VLA-Arena 3,400.

## 5.2 Reconstruction and Physical Structure

Figure 2 examines how tokenizer reconstruction fidelity and neighborhood-order preservation relate to downstream policy success, examining the action-space property introduced in Section 3. We compute normalized ranks within the original experimental groups and summarize their association across 55 tokenizer– benchmark evaluations. Both metrics exhibit positive associations with success, with PRC yielding a higher Spearman correlation (0.681) than reconstruction fidelity (0.544). This observation motivates examining physical structure alongside pointwise reconstruction; the controlled comparisons and ablations below evaluate the benefits of explicitly supervising this structure. Appendix A describes alternative RVQ objectives and reports their reconstruction, PRC, and policy success.

## 5.3 Matched Action-Tokenizer Comparison

We compare ActionPiece with ActionCodec [8], OAT [23, 24], FASTer [27], FAST [31], and standard RVQ through one Qwen3-VL-4B policy implementation. The policy data, prompt, optimizer, budget, and evaluation are shared. Each tokenizer retains its native output length and vocabulary. The comparison evaluates the policy trained with each action representation.

Table 1 reports 94.8% on in-distribution LIBERO for ActionPiece, compared with 93.7% for ActionCodec, the strongest baseline on this benchmark. The separation is larger on unseen LIBERO-Plus: ActionPiece reaches 68.8%, compared with 64.3% for the strongest baseline, and leads on six of seven shift categories. All policies are trained on LIBERO demonstrations; LIBERO-Plus measures transfer under camera, robot, language, illumination, background, noise, and layout changes. The advantage over the strongest baseline grows from 1.1 points on LIBERO to 4.5 on LIBERO-Plus. Gains across six shift categories support preserving action adjustments as a useful principle for execution beyond the training distribution.

## 5.4 Evaluation on SimplerEnv and VLA-Arena

In both benchmarks, we integrate ActionPiece by replacing the action tokenizer while retaining the policy architecture and training protocol. Policies are trained with standard next-token prediction, without

Table 2 SimplerEnv WidowX results. We follow the oficial SimplerEnv evaluation protocol [19]. All values are success rates (%). Best and second-best averages are bold and underlined, respectively.
<table><tr><td>Method</td><td>Put Spoon on Towel</td><td>Put Carrot on Plate</td><td>Stack Block on Block</td><td>Put Eggplant in Basket</td><td>Avg.</td></tr><tr><td>RT-1-X [30]</td><td>0.0</td><td>4.2</td><td>0.0</td><td>0.0</td><td>1.1</td></tr><tr><td>Octo-Base [36]</td><td>0.0</td><td>12.5</td><td>15.8</td><td>41.7</td><td>17.5</td></tr><tr><td>Octo-Small [36]</td><td>0.0</td><td>8.2</td><td>41.7</td><td>56.7</td><td>26.7</td></tr><tr><td>OpenVLA-OFT [16]</td><td>34.2</td><td>30.0</td><td>30.0</td><td>72.5</td><td>41.8</td></tr><tr><td>RoboVLM [18]</td><td>50.0</td><td>37.5</td><td>0.0</td><td>83.3</td><td>42.7</td></tr><tr><td>Magma [39]</td><td>37.5</td><td>29.2</td><td>20.8</td><td>91.7</td><td>44.8</td></tr><tr><td>CogACT [17]</td><td>71.7</td><td>50.8</td><td>15.0</td><td>67.5</td><td>51.3</td></tr><tr><td>SpatialVLA [32]</td><td>20.8</td><td>20.8</td><td>25.0</td><td>70.8</td><td>34.4</td></tr><tr><td>TraceVLA [41]</td><td>12.5</td><td>16.6</td><td>16.6</td><td>65.0</td><td>27.7</td></tr><tr><td>VideoVLA [34]</td><td>75.0</td><td>20.8</td><td>45.8</td><td>70.8</td><td>53.1</td></tr><tr><td>π0 [4]</td><td>29.2</td><td>62.5</td><td>29.2</td><td>91.6</td><td>53.1</td></tr><tr><td>π0.5 [12]</td><td>49.3</td><td>64.7</td><td>44.7</td><td>69.7</td><td>57.1</td></tr><tr><td>Isaac-GR00T-N1.6-Bridge [35]</td><td>64.5</td><td>65.5</td><td>5.5</td><td>93.0</td><td>57.1</td></tr><tr><td>LangForce [20]</td><td>89.6</td><td>63.8</td><td>33.3</td><td>79.2</td><td>66.5</td></tr><tr><td>ActionCodec-BAR [8]</td><td>71.7</td><td>64.2</td><td>55.0</td><td>70.0</td><td>65.2</td></tr><tr><td>ActionPiece (Ours)</td><td>83.3</td><td>54.2</td><td>58.3</td><td>91.7</td><td>71.9</td></tr></table>

benchmark-specific policy modifications.

For each SimplerEnv task, we average success over five independent 24-episode repeats, then macro-average the four task scores. ActionPiece achieves the highest average success of 71.9% in this real-to-sim evaluation, with task-specific strengths varying across methods. ActionPiece leads block stacking at 58.3%, a task requiring precise placement, alongside strong spoon (83.3%) and eggplant (91.7%) results. These outcomes extend the evidence for physical relationship supervision to tokenizers trained on real-robot actions.

Across all 3,400 VLA-Arena rollouts, ActionPiece reaches 82.2%/42.7%/29.5% on L0/L1/L2 and an equalweight 33-cell mean of 51.5%, the highest overall average among the compared models. As on SimplerEnv, this aggregate advantage accompanies variation in task-level performance. Table 3 also reports cumulative safety costs. Despite training only on L0, ActionPiece leads the L1 and L2 averages by 7.0 and 5.0 points. Leading L2 results on static obstacles, dynamic obstacles, and unseen objects extend this advantage to obstacle-constrained execution and object generalization.

## 5.5 Component Ablation

We evaluate the two physical-order objectives defined in Sections 4.3 and 4.4. Table 4 adds each objective individually and both together to standard RVQ. Standard RVQ reaches 90.9% on LIBERO and 60.4% on LIBERO-Plus. Adding physical rank preservation raises these scores to 93.8% and 65.4%; adding only quantization regularization yields 92.9% and 62.4%. Combining both components achieves the highest overall success on both benchmarks: 94.8% and 68.8%.

The same ablations raise PRC from 0.902 for standard RVQ to 0.947 with physical rank preservation and 0.916 with quantization regularization; the combined model reaches 0.953. The policy improvements thus accompany better physical neighborhood preservation. Physical rank preservation contributes the larger individual gain; adding quantization regularization further raises LIBERO-Plus success by 3.4 percentage points, supporting their joint use. Combining the objectives exceeds either alone in six LIBERO-Plus categories, including a camera-shift gain from 35.5% with PRP to 45.1%. This complementarity supports guiding both feature geometry and discrete code assignment with physical relationships.

Table 3 VLA-Arena results [40]. SR is success rate (%); CC is cumulative cost (lower is better). Avg equally weights the 11 suites. For each task and dificulty level, the highest SR across the six models is bold, including ties; the same rule applies to Avg. CC values are not bold. Baselines include π<sub>0.5</sub> [12], GR00T-N1.6 [35], Evo-Depth [21], and Motus [3]. Qwen-GR00T denotes Qwen3-VL [1] with the GR00T N1 action head [29].
<table><tr><td rowspan="2">Task</td><td rowspan="2">Metric</td><td colspan="2">π0.5</td><td colspan="3">GR00T-N1.6</td><td colspan="3">Qwen-GR00T</td><td colspan="3"></td><td colspan="3">Evo-Depth</td><td colspan="3">Motus</td><td colspan="3">ActionPiece</td></tr><tr><td colspan="2">L0 L1</td><td colspan="2">L2</td><td colspan="2">L0 L1</td><td>L2 L0</td><td></td><td>L1</td><td>L2</td><td>L0</td><td>L1</td><td>L2</td><td>L0</td><td>L1</td><td>L2</td><td>L0</td><td>L1</td><td>L2</td></tr><tr><td>Safety</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">Static obstacles</td><td>SR ↑</td><td>90.0</td><td>62.0</td><td>40.0</td><td>72.0</td><td>30.0</td><td>14.0</td><td>91.0</td><td></td><td>18.0</td><td>5.0</td><td>88.0</td><td>66.0</td><td>48.0</td><td>64.0</td><td>67.0</td><td>41.0</td><td>91.0</td><td></td><td>83.0 57.0</td></tr><tr><td>CC↓</td><td>0.0</td><td>33.3</td><td>76.6</td><td>0.0</td><td>11.2</td><td>38.4</td><td>0.0</td><td>9.1</td><td>29.6</td><td>0.0</td><td>8.7</td><td></td><td>14.9</td><td>0.0</td><td>72.0</td><td>124.3</td><td>0.0</td><td>14.5</td><td>30.3</td></tr><tr><td rowspan="2">Cautious grasp</td><td>SR ↑</td><td>50.0</td><td>14.0</td><td>0.0</td><td>16.0</td><td>2.0</td><td>0.0</td><td>77.0</td><td></td><td>8.0</td><td>2.0</td><td>78.0</td><td>24.0</td><td>0.0</td><td>62.0</td><td>25.0</td><td>11.0</td><td>77.0</td><td>18.0</td><td>1.0</td></tr><tr><td>CC↓</td><td>5.0</td><td>5.5</td><td>1.2</td><td>9.6</td><td>41.2</td><td>10.4</td><td>2.4</td><td></td><td>98.2</td><td>11.1</td><td>0.0</td><td>11.4</td><td>26.3</td><td>3.9</td><td>91.0</td><td>9.8</td><td>1.1</td><td>19.3</td><td>12.4</td></tr><tr><td rowspan="2">Hazard avoidance</td><td>SR ↑</td><td>58.0</td><td>30.0</td><td>36.0</td><td>64.0</td><td>4.0</td><td>10.0</td><td>67.0</td><td></td><td>15.0</td><td>19.0</td><td>40.0</td><td>0.0</td><td>14.0</td><td>65.0</td><td>34.0</td><td>21.0</td><td>68.0</td><td>22.0</td><td>46.0</td></tr><tr><td>CC↓</td><td>7.1</td><td>15.0</td><td>14.5</td><td>6.1</td><td>20.2</td><td>17.9</td><td>7.0</td><td></td><td>19.3</td><td>19.5</td><td>7.4</td><td>25.9</td><td>7.9</td><td>15.6</td><td>40.2</td><td>51.3</td><td>0.7</td><td>4.0</td><td>1.7</td></tr><tr><td rowspan="2">State preservation</td><td>SR ↑</td><td>58.0</td><td>56.0</td><td>54.0</td><td>66.0</td><td>50.0</td><td>38.0</td><td>86.0</td><td></td><td>56.0</td><td>47.0</td><td>88.0</td><td>66.0</td><td>56.0</td><td>63.0</td><td>63.0</td><td>72.0</td><td>94.0</td><td>80.0</td><td>64.0</td></tr><tr><td>CC↓</td><td>0.0</td><td>5.6</td><td>20.8</td><td>0.0</td><td>5.0</td><td>10.4</td><td>0.0</td><td></td><td>5.6</td><td>15.7</td><td>0.0</td><td>2.3</td><td>5.1</td><td>0.0</td><td>6.3</td><td>16.0</td><td>0.0</td><td>6.9</td><td>25.3</td></tr><tr><td rowspan="2">Dynamic obstacles</td><td>SR↑</td><td>50.0</td><td>44.0</td><td>22.0</td><td>74.0</td><td>50.0</td><td>2.0</td><td>81.0</td><td></td><td>56.0</td><td>3.0</td><td>82.0</td><td>60.0</td><td>6.0</td><td>58.0</td><td>33.0</td><td>11.0</td><td>82.0</td><td>76.0</td><td>27.0</td></tr><tr><td>CC↓</td><td>2.4</td><td>8.8</td><td>5.7</td><td>5.7</td><td>7.3</td><td>56.8</td><td>6.0</td><td></td><td>8.3</td><td>2.7</td><td>0.0</td><td>3.4</td><td>12.7</td><td>6.5</td><td>47.9</td><td>7.2</td><td>4.6</td><td>21.2</td><td>31.0</td></tr><tr><td>Distractor</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Static distractors</td><td>SR ↑</td><td>88.0</td><td>16.0</td><td>16.0</td><td>46.0</td><td>32.0</td><td>6.0</td><td></td><td>91.0</td><td>6.0</td><td>2.0</td><td>94.0</td><td>20.0</td><td>24.0</td><td>69.0</td><td>19.0</td><td>13.0</td><td>92.0</td><td>21.0</td><td>22.0</td></tr><tr><td>Dynamic distractors</td><td>SR ↑</td><td>80.0</td><td>66.0</td><td>54.0</td><td>70.0</td><td>72.0</td><td>18.0</td><td>92.0</td><td></td><td>49.0</td><td>25.0</td><td>86.0</td><td>60.0</td><td>32.0</td><td>77.0</td><td>51.0</td><td>26.0</td><td>90.0</td><td>63.0</td><td>43.0</td></tr><tr><td>Extrapolation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prep. combinations</td><td>SR ↑</td><td>62.0</td><td>24.0</td><td>6.0</td><td>48.0</td><td>0.0</td><td>0.0</td><td></td><td>51.0</td><td>1.0</td><td>0.0</td><td>66.0</td><td>0.0</td><td>0.0</td><td>41.0</td><td>7.0</td><td>1.0</td><td>69.0</td><td>15.0</td><td>0.0</td></tr><tr><td>Task workflows</td><td>SR ↑</td><td>38.0</td><td>20.0</td><td>22.0</td><td>42.0</td><td>0.0</td><td>0.0</td><td></td><td>51.0</td><td>3.0</td><td>9.0</td><td>32.0</td><td>0.0</td><td>0.0</td><td>38.0</td><td>25.0</td><td>14.0</td><td>64.0</td><td>33.0</td><td>21.0</td></tr><tr><td>Unseen objects</td><td>SR↑</td><td>48.0</td><td>60.0</td><td>20.0</td><td>26.0</td><td>18.0</td><td>16.0</td><td>63.0</td><td></td><td>46.0</td><td>26.0</td><td>78.0</td><td>52.0</td><td>4.0</td><td>63.0</td><td>65.0</td><td>19.0</td><td>78.0</td><td>59.0</td><td>43.0</td></tr><tr><td>Long Horizon</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></table>

Table 4 Component ablation on LIBERO and LIBERO-Plus. Starting from standard RVQ, we add physical rank preservation (PRP), quantization regularization (QR), or both. Success rates are reported in percent. Best results are bold.
<table><tr><td></td><td colspan="5">LIBERO</td><td colspan="9">LIBERO-Plus</td></tr><tr><td>Variant</td><td>Spatial</td><td>Object</td><td>Goal</td><td>Long</td><td>Overall</td><td>Camera</td><td>Robot</td><td></td><td>Language</td><td>Light</td><td>Bkg.</td><td>Noise</td><td>Layout</td><td>Overall</td></tr><tr><td>Standard RVQ</td><td>91.8</td><td>98.2</td><td>91.6</td><td>82.0</td><td>90.9</td><td></td><td>31.5</td><td>43.9</td><td>75.3</td><td>86.0</td><td>88.0</td><td>42.5</td><td>72.9</td><td>60.4</td></tr><tr><td>+ PRP</td><td>96.2</td><td>97.4</td><td>93.0</td><td>88.4</td><td>93.8</td><td></td><td>35.5</td><td>45.6</td><td>83.0</td><td>85.9</td><td>90.3</td><td>55.1</td><td>77.1</td><td>65.4</td></tr><tr><td>+ QR</td><td>95.4</td><td>97.2</td><td>91.2</td><td>87.8</td><td>92.9</td><td></td><td>33.0</td><td>38.8</td><td>77.7</td><td>90.8</td><td>89.7</td><td>50.2</td><td>74.3</td><td>62.4</td></tr><tr><td>ActionPiece (both)</td><td>96.2</td><td>99.0</td><td>93.4</td><td>90.6</td><td>94.8</td><td></td><td>45.1</td><td>49.6</td><td>82.2</td><td>91.9</td><td>91.4</td><td>57.6</td><td>78.0</td><td>68.8</td></tr></table>

## 6 Conclusion

Action tokenization defines the representation that an autoregressive policy learns to predict. We examined how reconstruction errors alter the action variations across demonstrations, and introduced PRC to evaluate physical distance ordering in a common decoded action space alongside reconstruction accuracy. ActionPiece applies this perspective through joint physical rank preservation and quantization regularization. Controlled comparisons and ablations show improvements in both physical structure and downstream success, while four benchmarks cover diferent action-data sources and generalization settings. These results support incorporating relationships among actions into tokenizer learning for control.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report, 2025.

[2] Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.

[3] Hongzhe Bi, Hengkai Tan, Shenghao Xie, Zeyuan Wang, Shuhe Huang, Haitian Liu, Ruowen Zhao, Yao Feng, Chendong Xiang, Yinze Rong, Hongyan Zhao, Hanyu Liu, Zhizhong Su, Lei Ma, Hang Su, and Jun Zhu. Motus: A unified latent action world model, 2025. URL https://arxiv.org/abs/2512.13030.

[4] Kevin Black, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Lachy Groom, Karol Hausman, Brian Ichter, et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

[5] Anthony Brohan, Noah Brown, Justice Carbajal, Yevgen Chebotar, Xi Chen, Krzysztof Choromanski, Tianli Ding, Danny Driess, Avinava Dubey, Chelsea Finn, Pete Florence, Chuyuan Fu, Montse Gonzalez Arenas, Keerthana Gopalakrishnan, Kehang Han, Karol Hausman, Alexander Herzog, Jasmine Hsu, Brian Ichter, Alex Irpan, Nikhil Joshi, Ryan Julian, Dmitry Kalashnikov, Yuheng Kuang, Isabel Leal, Lisa Lee, Tsang-Wei Edward Lee, Sergey Levine, Yao Lu, Henryk Michalewski, Igor Mordatch, Karl Pertsch, Kanishka Rao, Krista Reymann, Michael Ryoo, Grecia Salazar, Pannag Sanketi, Pierre Sermanet, Jaspiar Singh, Anikait Singh, Radu Soricut, Huong Tran, Vincent Vanhoucke, Quan Vuong, Ayzaan Wahid, Stefan Welker, Paul Wohlhart, Jialin Wu, Fei Xia, Ted Xiao, Peng Xu, Sichun Xu, Tianhe Yu, and Brianna Zitkovich. Rt-2: Vision-language-action models transfer web knowledge to robotic control, 2023.

[6] Hao Chen, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Renrui Zhang, Xiaoqi Li, Xiao He, Yandong Guo, Chi-Wing Fu, Shanghang Zhang, and Pheng-Ann Heng. Fast-in-slow: A dual-system foundation model unifying fast manipulation within slow reasoning, 2025. URL https://arxiv.org/abs/2506.01953.

[7] StarVLA Community. Starvla: A lego-like codebase for vision-language-action model developing. arXiv preprint arXiv:2604.05014, 2026.

[8] Zibin Dong, Yicheng Liu, Shiduo Zhang, Baijun Ye, Yifu Yuan, Fei Ni, Jingjing Gong, Xipeng Qiu, Hang Zhao, Yinchuan Li, and Jianye Hao. Actioncodec: What makes for good action tokenizers. arXiv preprint arXiv:2602.15397, 2026.

[9] Danny Driess, Jost Tobias Springenberg, Brian Ichter, Lili Yu, Adrian Li-Bell, Karl Pertsch, Allen Z. Ren, Homer Walke, Quan Vuong, Lucy Xiaoyang Shi, and Sergey Levine. Knowledge insulating vision-language-action models: Train fast, run fast, generalize better, 2025. URL https://arxiv.org/abs/2505.23705.

[10] Senyu Fei, Siyin Wang, Junhao Shi, Zihao Dai, Jikun Cai, Pengfang Qian, Li Ji, Xinzhe He, Shiduo Zhang, Zhaoye Fei, Jinlan Fu, Jingjing Gong, and Xipeng Qiu. Libero-plus: In-depth robustness analysis of vision-language-action models. arXiv preprint arXiv:2510.13626, 2025.

[11] Asher J. Hancock, Xindi Wu, Lihan Zha, Olga Russakovsky, and Anirudha Majumdar. Actions as language: Fine-tuning vlms into vlas without catastrophic forgetting, 2025.

[12] Physical Intelligence, Kevin Black, Noah Brown, James Darpinian, Karan Dhabalia, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai, Manuel Y. Galliker, Dibya Ghosh, Lachy Groom, Karol Hausman, Brian Ichter, Szymon Jakubczak, Tim Jones, Liyiming Ke, Devin LeBlanc, Sergey Levine, Adrian Li-Bell, Mohith Mothukuri, Suraj Nair, Karl Pertsch, Allen Z. Ren, Lucy Xiaoyang Shi, Laura Smith, Jost Tobias Springenberg, Kyle Stachowicz, James Tanner, Quan Vuong, Homer Walke, Anna Walling, Haohuan Wang, Lili Yu, and Ury Zhilinsky. π<sub>0.5</sub>: a vision-language-action model with open-world generalization, 2025.

[13] Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. arXiv preprint arXiv:1611.01144, 2016.

[14] Miracle Kang, Lights Shi, Lucy Liang, Roy Gan, Dongxiu Liu, Pushi Zhang, Sylas Chen, Shawn Qin, Yinan Zheng, Jinliang Zheng, Hao Wang, Xianyuan Zhan, and Hang Su. X-tokenizer: A multimodal action tokenizer for vision-language-action pretraining, 2026. Preprint.

[15] Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, Ted Xiao, Ashwin Balakrishna, Suraj Nair, Rafael Rafailov, Ethan P Foster, Pannag R Sanketi, Quan Vuong, Thomas Kollar, Benjamin Burchfiel, Russ Tedrake, Dorsa Sadigh, Sergey Levine, Percy Liang, and Chelsea Finn. Openvla: An open-source vision-language-action model. In Annual Conference on Robot Learning (CoRL), 2024.

[16] Moo Jin Kim, Chelsea Finn, and Percy Liang. Fine-tuning vision-language-action models: Optimizing speed and success, 2025.

[17] Qixiu Li, Yaobo Liang, Zeyu Wang, Lin Luo, Xi Chen, Mozheng Liao, Fangyun Wei, Yu Deng, Sicheng Xu, Yizhong Zhang, Xiaofan Wang, Bei Liu, Jianlong Fu, Jianmin Bao, Dong Chen, Yuanchun Shi, Jiaolong Yang, and Baining Guo. Cogact: A foundational vision-language-action model for synergizing cognition and action in robotic manipulation, 2024.

[18] Xinghang Li, Peiyan Li, Minghuan Liu, Dong Wang, Jirong Liu, Bingyi Kang, Xiao Ma, Tao Kong, Hanbo Zhang, and Huaping Liu. Towards generalist robot policies: What matters in building vision-language-action models, 2024.

[19] Xuanlin Li, Kyle Hsu, Jiayuan Gu, Oier Mees, Karl Pertsch, Homer Rich Walke, Chuyuan Fu, Ishikaa Lunawat, Isabel Sieh, Sean Kirmani, Sergey Levine, Jiajun Wu, Chelsea Finn, Hao Su, Quan Vuong, and Ted Xiao. Evaluating real-world robot manipulation policies in simulation. In Annual Conference on Robot Learning (CoRL), 2024.

[20] Shijie Lian, Bin Yu, Xiaopeng Lin, Laurence T. Yang, Zhaolong Shen, Changti Wu, Yuzhuo Miao, Cong Huang, and Kai Chen. Langforce: Bayesian decomposition of vision language action models via latent action queries, 2026.

[21] Tao Lin, Yuxin Du, Jiting Liu, Nuobei Zhu, Yunhe Li, Yuqian Fu, Yinxinyu Chen, Hongyi Cai, Zewei Ye, Bing Cheng, Kai Ye, Yiran Mao, Yilei Zhong, MingKang Dong, Junchi Yan, Gen Li, and Bo Zhao. Evo-depth: A lightweight depth-enhanced vision-language-action model. arXiv preprint arXiv:2605.14950, 2026.

[22] Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. Libero: Benchmarking knowledge transfer for lifelong robot learning. arXiv preprint arXiv:2306.03310, 2023.

[23] Chaoqi Liu, Xiaoshen Han, Jiawei Gao, Yue Zhao, Haonan Chen, and Yilun Du. OAT: Ordered Action Tokenization. In Proceedings of Robotics: Science and Systems, Sydney, Australia, July 2026. doi: 10.15607/RSS.2026.XXII.075. URL https://www.roboticsproceedings.org/rss22/p075.html.

[24] Chaoqi Liu, Yue Zhao, Haonan Chen, Xiaoshen Han, Jiawei Gao, Ehsan Adeli, and Yilun Du. Ordered action tokens for visuomotor policy learning, 2026. URL https://arxiv.org/abs/2607.21670.

[25] Jiaming Liu, Hao Chen, Pengju An, Zhuoyang Liu, Renrui Zhang, Chenyang Gu, Xiaoqi Li, Ziyu Guo, Sixiang Chen, Mengzhen Liu, Chengkai Hou, Mengdi Zhao, KC alex Zhou, Pheng-Ann Heng, and Shanghang Zhang. Hybridvla: Collaborative difusion and autoregression in a unified vision-language-action model, 2025. URL https://arxiv.org/abs/2503.10631.

[26] Songming Liu, Lingxuan Wu, Bangguo Li, Hengkai Tan, Huayu Chen, Zhengyi Wang, Ke Xu, Hang Su, and Jun Zhu. Rdt-1b: a difusion foundation model for bimanual manipulation. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 29982–30009, 2025.

[27] Yicheng Liu, Shiduo Zhang, Zibin Dong, Baijun Ye, Tianyuan Yuan, Xiaopeng Yu, Linqi Yin, Chenhao Lu, Junhao Shi, Luca Jiang-Tao Yu, Liangtao Zheng, Tao Jiang, Jingjing Gong, Xipeng Qiu, and Hang Zhao. Faster: Toward eficient autoregressive vision language action modeling via neural action tokenization. arXiv preprint arXiv:2512.04952, 2025.

[28] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization, 2019.

[29] NVIDIA, :, Johan Bjorck, Fernando Castañeda, Nikita Cherniadev, Xingye Da, Runyu Ding, Linxi "Jim" Fan, Yu Fang, Dieter Fox, Fengyuan Hu, Spencer Huang, Joel Jang, Zhenyu Jiang, Jan Kautz, Kaushil Kundalia, Lawrence Lao, Zhiqi Li, Zongyu Lin, Kevin Lin, Guilin Liu, Edith Llontop, Loic Magne, Ajay Mandlekar, Avnish Narayan, Soroush Nasiriany, Scott Reed, You Liang Tan, Guanzhi Wang, Zu Wang, Jing Wang, Qi Wang, Jiannan

Xiang, Yuqi Xie, Yinzhen Xu, Zhenjia Xu, Seonghyeon Ye, Zhiding Yu, Ao Zhang, Hao Zhang, Yizhou Zhao, Ruijie Zheng, and Yuke Zhu. Gr00t n1: An open foundation model for generalist humanoid robots, 2025.

[30] Abby O’Neill, Abdul Rehman, Abhiram Maddukuri, Abhishek Gupta, Abhishek Padalkar, Abraham Lee, Acorn Pooley, Agrim Gupta, Ajay Mandlekar, Ajinkya Jain, et al. Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 6892–6903. IEEE, 2024.

[31] Karl Pertsch, Kyle Stachowicz, Brian Ichter, Danny Driess, Suraj Nair, Quan Vuong, Oier Mees, Chelsea Finn, and Sergey Levine. Fast: Eficient action tokenization for vision-language-action models, 2025.

[32] Delin Qu, Haoming Song, Qizhi Chen, Yuanqi Yao, Xinyi Ye, Yan Ding, Zhigang Wang, JiaYuan Gu, Bin Zhao, Dong Wang, and Xuelong Li. Spatialvla: Exploring spatial representations for visual-language-action model, 2025.

[33] Samyam Rajbhandari, Jef Rasley, Olatunji Ruwase, and Yuxiong He. Zero: memory optimizations toward training trillion parameter models. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, SC ’20. IEEE Press, 2020. ISBN 9781728199986.

[34] Yichao Shen, Fangyun Wei, Zhiying Du, Yaobo Liang, Yan Lu, Jiaolong Yang, Nanning Zheng, and Baining Guo. Videovla: Video generators can be generalizable robot manipulators, 2025.

[35] GEAR Team, Allison Azzolini, Johan Bjorck, Valts Blukis, Fernando Castañeda, Rahul Chand, et al. Gr00t n1.6: An improved open foundation model for generalist humanoid robots. https://research.nvidia.com/labs/gear/ gr00t-n1\_6/, December 2025.

[36] Octo Model Team, Dibya Ghosh, Homer Walke, Karl Pertsch, Kevin Black, Oier Mees, Sudeep Dasari, Joey Hejna, Tobias Kreiman, Charles Xu, Jianlan Luo, You Liang Tan, Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh, Chelsea Finn, and Sergey Levine. Octo: An open-source generalist robot policy, 2024.

[37] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, 2017.

[38] Homer Rich Walke, Kevin Black, Tony Z. Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, 2023.

[39] Jianwei Yang, Reuben Tan, Qianhui Wu, Ruijie Zheng, Baolin Peng, Yongyuan Liang, Yu Gu, Mu Cai, Seonghyeon Ye, Joel Jang, et al. Magma: A foundation model for multimodal ai agents. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14203–14214, 2025.

[40] Borong Zhang, Jiahao Li, Jiachen Shen, Yishuai Cai, Yuhao Zhang, Yuanpei Chen, Juntao Dai, Jiaming Ji, and Yaodong Yang. Vla-arena: An open-source framework for benchmarking vision-language-action models, 2025.

[41] Ruijie Zheng, Yongyuan Liang, Shuaiyi Huang, Jianfeng Gao, Hal Daumé III, Andrey Kolobov, Furong Huang, and Jianwei Yang. Tracevla: Visual trace prompting enhances spatial-temporal awareness for generalist robotic policies, 2025.

## A Alternative Training Objectives

We compare alternative objectives added to the standard RVQ tokenizer to examine how diferent supervision afects reconstruction, physical structure, and policy success. Table 5 reports measurements for these variants and the component ablations, using Qwen3-VL-4B at 30K policy training steps.

SIGReg. We apply Sketched Isotropic Gaussian Regularization (SIGReg) from LeJEPA [2] to encoder features, encouraging their distribution to match an isotropic Gaussian.

Temporal. This objective matches the temporal changes of reconstructed actions to those of demonstrations. For translation and gripper commands, it compares first diferences between adjacent time steps in the reconstructed and target sequences. For rotation, it compares the relative quaternion rotations between adjacent steps. Thus, it supervises the demonstrated pattern of change, including its magnitude and direction, rather than penalizing motion itself.

Neighborhood. This objective attracts physically similar actions in representation space. For each anchor i, we select the ten physically nearest actions in the minibatch, denoted by $\mathcal { N } _ { 1 0 } ( i )$ . With larger nonnegative weights $w _ { i j }$ for physically closer neighbors, the objective is

$$
\mathcal { L } _ { \mathrm { n e i } } = \frac { \sum _ { i } \sum _ { j \in \mathcal { N } _ { 1 0 } ( i ) } w _ { i j } d _ { \mathrm { l a t } } ^ { 2 } ( i , j ) } { \sum _ { i } \sum _ { j \in \mathcal { N } _ { 1 0 } ( i ) } w _ { i j } } ,\tag{14}
$$

where $d _ { \mathrm { l a t } }$ is the distance between action representations. Neighborhood supervises attraction to nearby actions; physical rank preservation additionally compares a nearby action with a farther one to supervise their relative order (Section 4.3).

Table 5 Training-objective variants of standard RVQ. Each “+” row adds only the named objective to standard RVQ. PRP and QR denote physical rank preservation and quantization regularization. Policies use Qwen3-VL-4B with 30K training steps. Success rates are in percent; MSE is scaled by $1 0 ^ { 4 }$ . Best values are bold, determined before rounding.
<table><tr><td>Variant</td><td>MSE↓</td><td>PRC ↑</td><td>LIBERO Overall ↑</td><td>LIBERO-Plus Overall ↑</td></tr><tr><td>Standard RVQ</td><td>4.06</td><td>0.902</td><td>90.9</td><td>60.4</td></tr><tr><td>+ SIGReg</td><td>3.77</td><td>0.950</td><td>93.3</td><td>66.5</td></tr><tr><td>+ Neighborhood</td><td>5.09</td><td>0.889</td><td>94.3</td><td>63.5</td></tr><tr><td>+ Temporal</td><td>3.83</td><td>0.924</td><td>94.8</td><td>66.1</td></tr><tr><td>+ PRP</td><td>3.63</td><td>0.947</td><td>93.8</td><td>65.4</td></tr><tr><td>+ QR</td><td>3.92</td><td>0.916</td><td>92.9</td><td>62.4</td></tr><tr><td>ActionPiece  $( \mathbf { P } \mathbf { R } \mathbf { P } + \mathbf { Q } \mathbf { R } )$ </td><td>3.32</td><td>0.953</td><td>94.8</td><td>68.8</td></tr></table>

These objectives target diferent properties of the codec. Temporal supervises changes within a decoded action chunk, Neighborhood attracts representations across chunks, and physical rank preservation supervises relative distances. The results illustrate why both reconstruction and neighborhood measurements are useful: the Neighborhood variant improves policy success over standard RVQ while its PRC decreases, whereas ActionPiece improves both PRC and success. ActionPiece achieves the lowest MSE, highest PRC, and highest overall success on both benchmarks among the variants shown.

## B Implementation Details

ActionPiece configuration. We use the same ActionPiece configuration for LIBERO, LIBERO-Plus, VLA-Arena, and SimplerEnv, as summarized in Table 6. Other tokenizers use the configurations specified in their original papers or released codebases.

Table 6 ActionPiece configuration and objective weights.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Action chunk length H</td><td>8</td></tr><tr><td>Output tokens  $T$ </td><td>16</td></tr><tr><td>Latent slots S</td><td>16</td></tr><tr><td>RVQ levels Q</td><td>1</td></tr><tr><td>Codebook size K</td><td>512</td></tr><tr><td>Reconstruction MSE weight</td><td>1</td></tr><tr><td>Commitment weight  $\lambda _ { \mathrm { v q } }$ </td><td>0.25</td></tr><tr><td>PRP weight  $\lambda _ { r }$ </td><td> $6 . 2 5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>QR weight  $\lambda _ { q }$ </td><td> $6 . 2 5 \times 1 0 ^ { - 4 }$ </td></tr></table>

## B.1 Differentiating Codeword Selection

With $Q = 1$ , each of the 16 latent slots selects one codeword. Let $u _ { i , s }$ be the projected encoder feature and $\Delta _ { i , s , c } = \| u _ { i , s } - e _ { c } \| _ { 2 } ^ { 2 }$ its squared distance to codebook entry c. Hard assignment selects a code; to diferentiate the representation distance, we use a straight-through estimator with soft codeword proba bilities [13, 37]. Let $e _ { i , s }$ <sub>,hard</sub> be the selected code contribution after the quantizer output projection, and let $e _ { i , s , \mathrm { s o f t } }$ be the output projection of the soft codebook average. Here sg denotes stop-gradient. With $s _ { i , s } = \operatorname* { m a x } \{ \mathrm { s g } [ \mathrm { S t d } _ { c } ( \Delta _ { i , s , c } ) ] , 1 0 ^ { - 6 } \}$ , we form

$$
\begin{array} { r l } & { \pi _ { i , s , c } = \mathrm { s o f t m a x } _ { c } \left( - \frac { \Delta _ { i , s , c } } { s _ { i , s } \tau _ { q } } \right) , } \\ & { \quad \widetilde { e } _ { i , s } = e _ { i , s , \mathrm { s o f t } } + \mathrm { s g } \left[ e _ { i , s , \mathrm { h a r d } } - e _ { i , s , \mathrm { s o f t } } \right] . } \end{array}\tag{15}
$$

We use $\tau _ { q } = 1$ and sample no Gumbel noise. Thus $\widetilde { e }$ has the selected hard code contribution as its forward value and the soft assignment for gradient computation. The reconstruction objective uses an identity straight-through estimator, and the codebook is updated by exponential moving average (EMA).

## B.2 Multiple Quantization Levels

In our experiments, $Q = 1$ and $Q = 2$ yield similar downstream policy performance. We use $Q = 1$ for the final ActionPiece configuration. For $Q > 1$ , we apply the same diferentiable codeword selection at each residual level. Let $\begin{array} { r } { q _ { i , s } ^ { ( r ) } = \sum _ { \ell = 1 } ^ { r } \widetilde { e } _ { i , s } ^ { ( \ell ) } } \end{array}$ be the cumulative contribution after the corresponding output projections, and let $\bar { q } _ { i , s } ^ { ( r ) }$ be LayerNorm and unit-normalized. We extend Eq. (6) by averaging over the partial reconstructions at all residual levels:

$$
d _ { \mathrm { q u a n t } } ^ { 2 } ( i , j ) = \frac { 1 } { Q } \sum _ { r = 1 } ^ { Q } \frac { 1 } { S } \sum _ { s = 1 } ^ { S } \left. \bar { q } _ { i , s } ^ { ( r ) } - \bar { q } _ { j , s } ^ { ( r ) } \right. _ { 2 } ^ { 2 } .\tag{16}
$$

At $Q = 1$ , this expression reduces to Eq. (6).

## B.3 Physical-Neighbor Selection

For each anchor in the training batch, we exclude the anchor itself and rank the remaining action chunks by increasing physical distance. We select the action at the 5th percentile $\left( q _ { 0 . 0 5 } \right)$ as the positive (near) example and the action at the 95th percentile $\left( q _ { 0 . 9 5 } \right)$ as the negative (far) example. These percentiles are computed separately for each anchor over the other actions in its minibatch. Both physical rank preservation and quantization regularization use this same pair. The former compares representation distances $D _ { i j }$ , while the latter compares codeword probabilities using Jensen–Shannon divergence. We use natural logarithms without additional normalization for the divergence. The discrete action vocabulary and decoding procedure are unchanged by the two training objectives.