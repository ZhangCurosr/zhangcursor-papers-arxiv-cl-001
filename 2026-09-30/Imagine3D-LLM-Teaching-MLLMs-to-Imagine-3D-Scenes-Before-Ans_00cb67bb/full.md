# Imagine3D-LLM: Teaching MLLMs to Imagine 3D Scenes Before Answering

Jaewoo Jung<sup>1,2∗</sup> Hyeonseo Yu<sup>1</sup> Honggyu An<sup>1</sup> Jisang Han<sup>1</sup> Mungyeom Kim<sup>1</sup> Minkyeong Jeon<sup>1</sup> Heeseong Shin<sup>1</sup> WonJun Moon<sup>1</sup> Federico Tombari<sup>3,4</sup> Daniel Barath<sup>2</sup> Marc Pollefeys<sup>2†</sup> Seungryong Kim<sup>1†</sup> Sunghwan Hong<sup>2,5†</sup>

<sup>1</sup>KAIST AI <sup>2</sup>ETH Zürich <sup>3</sup>Google <sup>4</sup>TUM <sup>5</sup>ETH AI Center

https://cvlab-kaist.github.io/Imagine3D-LLM

## Abstract

Reasoning about the 3D world from multi-view images remains a fundamental challenge for Multimodal Large Language Models (MLLMs). While modern MLLMs handle single-image inputs effectively, they struggle to integrate evidence across viewpoints into a coherent 3D understanding. A growing body of work attempts to close this gap by injecting 3D awareness into MLLMs, either by boosting fine-grained pixel-level cross-view correspondence or by fusing features from 3D geometry foundation models, yet a substantial gap to human reasoning persists. In this work, we revisit human spatial reasoning, which suggests that rather than relying on fine-grained geometry cues, humans roughly identify common objects across views, infer the relative geometry between viewpoints, and assemble a coarse 3D layout of the scene. Inspired by this process, we introduce Imagine3D-LLM, an MLLM that learns to assemble a similar compact 3D representation of the scene and conditions its answer on this representation. Concretely, we append a small set of learnable summary tokens after the image tokens, decode them into a compact 3D Gaussian Splatting representation supervised by a photometric reconstruction loss, and train jointly with the standard next-token prediction objective. Notably, although only the summary tokens receive direct reconstruction supervision, this objective also induces stronger cross-frame correspondence within the LLM’s underlying image features, suggesting that learning to reconstruct propagates 3Daware signals throughout the model. As a result, Imagine3D-LLM consistently outperforms prior approaches across multiple spatial reasoning and 3D understanding benchmarks, suggesting that imagining the scene can be more effective than being told its pixel-wise geometry.

## 1 Introduction

Reasoning about 3D structure from multi-view images is a fundamental problem in computer vision, underpinning a wide range of downstream applications such as robotics [62], embodied AI [91], AR/VR [6], and scene understanding [1]. Driven by rapid progress in Multimodal Large Language Models (MLLMs) [80, 4, 34, 66], machines can now interpret a single image or a continuous video stream at a level that often rivals, and in some cases surpasses, human performance [85, 50]. However, when the input shifts from a single viewpoint to a set of multi-view images that demand genuine 3D reasoning, even the largest and most capable models fall short of human-level competence [82].

A growing body of work has attempted to close this gap by injecting strong 3D priors into MLLMs. One line of research implicitly or explicitly boosts pixel-level correspondences across views by either embedding scene point clouds’ 3D coordinates into patch-level features [93, 96, 73], augmenting images with explicit visual markers [58], distilling features with accurate correspondences [33], or designing pretext tasks that encourage the model to align overlapping content across frames [73]. A complementary line [92, 23, 30] fuses MLLM image features with ones from recently developed 3D reconstruction foundation models such as CUT3R [76] and VGGT [74]. While both approaches attempt to improve 3D awareness of MLLMs with fine-grained pixel-level 3D signals, these approaches yield only incremental gains [82, 87].

In this work, we take a step back and ask whether such fine-grained 3D signals are truly the most direct route to improved 3D reasoning. To answer this, we revisit how humans actually perceive 3D structure from multi-view observations. Cognitive studies suggest that humans do not maintain pixel-perfect correspondences or dense depth maps in their heads [7, 63]. When presented with several views of a scene, humans instead identify common objects across views, use them as anchors to infer coarse and abstract spatial relationships between viewpoints, and mentally assemble a compact layout of the scene in 3D. Therefore, we hypothesize that the representation that supports their downstream reasoning is rather object-level and approximate, not pixel-accurate.

Inspired by this human thinking process, we present Imagine3D-LLM, a framework that enables an MLLM to build an analogous compact 3D reconstruction of the scene before answering. To enable this, we introduce a compact set of learnable Gaussian summary tokens, which are appended to the sequence of image tokens fed into the LLM. From these tokens, we decode the parameters of 3D Gaussian Splatting (3DGS) [42] primitives, so that each Gaussian summary token corresponds to a small set of 3D Gaussians that together explain a portion of the scene.

To efficiently and effectively enable MLLMs to build an abstract and compact reconstruction of the scene, our Gaussian estimation pipeline incorporates two deliberate design choices. (1) To prevent the model from simply copy-pasting 2D image content into image-aligned Gaussians rather than recovering the underlying geometry, we predict the Gaussians from the dedicated Gaussian summary tokens instead of predicting them directly from the image features. (2) We impose an information bottleneck by using fewer Gaussian summary tokens than image tokens, which compels overlapping content across views to be merged into a shared set of tokens, encouraging the model to reason about which objects recur across views and how they fit together within a unified 3D layout.

We jointly train Imagine3D-LLM with a photometric reconstruction loss on the rendered Gaussians and the standard next-token prediction loss for language modeling. Surprisingly, although only the Gaussian summary tokens receive direct reconstruction supervision, our analysis shows that the LLM’s image features themselves become more 3D-aware: corresponding image features show more sharp attentions, where PCA visualizations further reveal that image features become more semantically organized, with the same object encoded consistently across views. These analyses validate that requiring the model to infer the underlying structure of the scene propagates 3D-aware signals throughout the LLM, reshaping its internal representations toward a coherent, object-centric 3D understanding. As a result, Imagine3D-LLM achieves significant performance gains across seven diverse spatial reasoning and 3D scene understanding benchmarks, suggesting that abstract scene reconstruction is a powerful inductive bias for building 3D-aware MLLMs, and that imagining the scene before answering can be more effective than being told its pixel-wise geometry.

## 2 Related Works

3D Scene Understanding. Grounding natural language in 3D environments is a long-standing goal of the 3D scene understanding community, supporting tasks such as 3D visual grounding [10, 88], 3D dense captioning [11, 12, 14], and 3D question answering [3, 52, 51, 82]. A prominent line of work distills features from 2D foundation models (most commonly CLIP [60] or LSeg [46]) into a reconstructed scene representation, including point clouds [81, 94, 36], NeRFs [43, 22], and more recently 3D Gaussian Splatting [39, 59, 64], enabling open-vocabulary querying by matching distilled embeddings to a text prompt. These pipelines, however, typically rely on task-specific architectures and per-scene optimization, and their language understanding is bounded by the compositional capacity of the underlying 2D embedding model. We instead extend the language and visual reasoning of pretrained MLLMs to multi-view inputs, allowing a single generalist model to address these tasks through natural language without per-task or per-scene optimization.

3D Multimodal Large Language Models (MLLMs). Motivated by the success of 2D MLLMs [4, 80, 34, 45, 48, 49, 66, 69, 75, 84], a growing body of work extends them to 3D understanding. Early efforts feed explicit 3D inputs to the LLM, either by lifting 2D features into 3D [96, 29] or by training dedicated point-cloud encoders as an additional modality [13, 78, 32, 79]; the scarcity of paired 3D–language data, however, has limited their performance on language-heavy tasks. More recent approaches retain the multi-view 2D input format and inject 3D awareness through training objectives or auxiliary modules. One family supplies pixel-level 3D signals to encourage cross-view correspondence: Video-3D-LLM [93] and LLaVA-3D [96] attach 3D positional embeddings derived from input point clouds, GPT4Scene [58] renders bird’s-eye-view markers, Ross3D [73] adds a cross-view reconstruction pretext task, and 3DRS [33] distills features with explicit correspondence supervision. A second family fuses MLLM image features with representations from 3D foundation models [76, 74] to expose the LLM to dense, geometry-aware features, as in VLM3R [23], $\mathbf { G } ^ { 2 } \mathbf { V } \mathbf { L } \mathbf { M }$ [30], and the concurrent 3DThinker [17]. Despite steady progress, a substantial gap to human-level performance on multi-view 3D reasoning benchmarks remains [82].

Compact Scene Representations. Cognitive studies suggest that humans do not maintain pixelaccurate reconstructions of every surface; rather, they rely on coarse abstractions that are nonetheless sufficient to support rich spatial reasoning [7, 63, 44]. A line of work in 3D vision has correspondingly explored decomposing scenes into compact geometric primitives such as meshes, polygons, or superquadrics [70, 56, 57, 24, 54], but these methods either struggle to scale beyond a handful of primitives [54] or rely on off-the-shelf 3D instance segmentation [24]. More recently, C3G [1] introduced a feed-forward pipeline that represents a scene compactly with 3D Gaussian Splatting (3DGS) parameters, decoding 3D Gaussians from a small set of learnable query tokens supervised purely through photometric rendering. Inspired by this line of work, we bring token-level Gaussian decoding inside an MLLM, where a small set of Gaussian summary tokens learns to assemble a compact 3D representation of the scene that the LLM conditions on when reasoning about it.

## 3 Methodology

## 3.1 Preliminaries: Multi-view MLLMs

Following prior works [73, 93, 96], we build our model on top of MLLMs capable of taking multiple images as input. Here, we briefly explain the input formulation of such MLLMs.

A multi-view MLLM is composed of a vision encoder $V _ { \psi } ( \cdot )$ , a vision–language projector $P _ { \phi } ( \cdot )$ , and a pre-trained large language model $L M _ { \theta } ( \cdot )$ , where ψ, ϕ, and θ denote the corresponding parameters. Given a set of $\bar { K }$ input frames $\mathcal { T } = \{ I _ { k } \} _ { k = { \frac { \cdot } { \iota } } } ^ { K }$ with $I _ { k } \in \mathbb { R } ^ { H \times W \times 3 }$ , the vision encoder independently extracts patch-level features from each frame:

$$
\mathbf { z } _ { k } = V _ { \psi } ( I _ { k } ) \in \mathbb { R } ^ { N \times D _ { \mathbf { z } } } , \quad k = 1 , \dots , K ,\tag{1}
$$

where N is the number of patches per frame and $D _ { \mathbf { z } }$ is the visual feature dimension. Each frame’s features are then projected into the language model’s embedding space of dimension D via $P _ { \phi } ( \cdot )$ yielding per-frame visual tokens ${ \bf e } _ { k } ^ { \mathrm { i m g } } = { \cal P } _ { \phi } ( { \bf z } _ { k } ) \in \mathbb { R } ^ { N ^ { \prime } \times D }$ , where $N ^ { \prime }$ may differ from N due to spatial token-reduction operations such as bilinear pooling and per-row newline insertion that preserve the 2D spatial layout [89]. Concatenating all $K$ frames produces the full visual token sequence:

$$
\mathbf { e } ^ { \mathrm { i } \mathrm { m g } } = [ \mathbf { e } _ { 1 } ^ { \mathrm { i } \mathrm { m g } } ; \mathbf { e } _ { 2 } ^ { \mathrm { i } \mathrm { m g } } ; \dots ; \mathbf { e } _ { K } ^ { \mathrm { i } \mathrm { m g } } ] \in \mathbb { R } ^ { K N ^ { \prime } \times D } .\tag{2}
$$

The input text (e.g., the user instruction) is tokenized and embedded into the same space through the language model’s embedding layer, producing textual embeddings $\mathbf { e } ^ { \mathrm { { t e x t } } } \in \mathbb { R } ^ { L \times D }$ , where L denotes the number of text tokens. The language model then takes the concatenated multimodal sequence $[ \mathbf { e } ^ { \mathrm { i m g } } ; \mathbf { e } ^ { \mathrm { t e x t } } ] \in \mathbb { R } ^ { ( K N ^ { \prime } + L ) \times D }$ as input and models the causal distribution over the text tokens as:

$$
p _ { \theta , \phi } ( \mathbf { e } _ { 1 : L } ^ { \mathrm { t e x t } } \mid \mathbf { e } ^ { \mathrm { i m g } } ) = \prod _ { i = 1 } ^ { L } p _ { \theta , \phi } \big ( \mathbf { e } _ { i } ^ { \mathrm { t e x t } } \mid \mathbf { e } _ { < i } ^ { \mathrm { t e x t } } , \mathbf { e } ^ { \mathrm { i m g } } \big ) .\tag{3}
$$

At inference, the language model autoregressively generates text tokens conditioned on the visual tokens and the previously generated text tokens.

![](images/0debedcb417d096a91f09de1b553a97249cb75d7c8d56e3cf84f1d06f3e14f60.jpg)  
Figure 1: Overall architecture of Imagine3D-LLM. Given multi-view images of a scene and a text instruction, our model appends a compact set of learnable Gaussian summary tokens between the image and text tokens. At an intermediate layer ℓ, the hidden states at the Gaussian summary token positions are decoded into a compact 3D Gaussian Splatting representation that abstractly reconstructs the scene. The model is trained jointly with a standard language modeling loss on the text output, a photometric reconstruction loss obtained by rendering the predicted Gaussians at the input viewpoints, and a distillation loss from a pretrained compact Gaussian teacher [1].

## 3.2 Imagine3D-LLM

We now describe how we equip a multi-image MLLM with the ability to perform mental 3D reconstruction of the input scene for improved 3D reasoning. At a high level, we introduce a compact set of learnable Gaussian summary tokens that are appended to the image tokens fed into the LLM (§ 3.2.1). These tokens learn to summarize the multi-view observations into a compact set of 3D Gaussians, jointly trained with the reconstruction and standard language modeling objective. Since pure photometric supervision is known to require lengthy training schedules even for learning Gaussians alone [16, 9, 83, 28, 37, 1], we further introduce a distillation objective from a pretrained Gaussian teacher model [72], to substantially accelerate convergence (§ 3.2.2). Finally, we analyze how this joint training improves the MLLM’s internal representations (§ 3.3). An overview of our method is shown in Fig. 1.

## 3.2.1 Mental Reconstruction via Gaussian Summary Tokens

Gaussian summary tokens. Given the visual token sequence $\mathbf { e } ^ { \mathrm { i m g } } \in \mathbb { R } ^ { K N ^ { \prime } \times D }$ produced by the vision encoder and projector, we introduce a compact set of M learnable Gaussian summary tokens $\mathbf { e } ^ { \mathrm { g s } } \in \mathbb { R } ^ { M \times D }$ , where $\mathbf { \bar { \mathit { M } } } \ll \mathit { K N ^ { \prime } }$ . These tokens are inserted between the visual tokens and the textual tokens, yielding the augmented multimodal sequence

$$
[ \mathbf { e } ^ { \mathrm { i m g } } ; \mathbf { e } ^ { \mathrm { g s } } ; \mathbf { e } ^ { \mathrm { t e x t } } ] \in \mathbb { R } ^ { ( K N ^ { \prime } + M + L ) \times D } ,\tag{4}
$$

which is processed by the LLM via standard causal self-attention. By placing the Gaussian summary tokens after the image tokens, each token can attend to all visual evidence and selectively retrieve the information needed to faithfully describe a portion of the 3D scene, rather than being passively bound to local image content.

Bottleneck-induced object grouping. A central design choice is that the number of Gaussian summary tokens is substantially smaller than the number of image tokens, $M \ll K N ^ { \prime }$ This bottleneck prevents the model from naively allocating one token per pixel or per patch, and instead forces overlapping content observed across multiple views to be merged into a shared set of tokens. Under this constraint, the most efficient way for the model to faithfully reconstruct the scene with limited Gaussians is to assign each token to a coherent 3D region or object that recurs across views. As we show in § 3.3, this bottleneck gives rise to an emergent object-centric grouping, mirroring the way humans build object-level mental abstractions of a scene before reasoning about it. Since our main objective is to improve 3D reasoning rather than novel view synthesis, this design also offers a favorable trade-off between reconstruction quality and computational cost, preventing the input sequence length to increase substantially.

Decoding 3D Gaussians. Given the LLM forward pass over the augmented sequence $[ { \bf e } ^ { \mathrm { i m g } } ; { \bf e } ^ { \mathrm { g s } } ; { \bf e } ^ { \mathrm { t e x t } } ]$ , a natural question is from which layer the Gaussian summary tokens should be decoded. Recent analyses [40, 90, 38, 41, 84] consistently identify that the middle layers of the LLM most actively show information flow between visual and text modalities. Since our Gaussian summary tokens also require effectively absorbing visual information from the image tokens, we extract the hidden states at the Gaussian summary token positions from the middle layer $\ell ,$ denoted $\mathbf { h ^ { g s } } \in \mathbb { R } ^ { M \times D }$ Decoding at a middle layer also leaves enough subsequent layers for the text tokens to attend to the Gaussian summary tokens, enabling the text tokens to benefit from the scene-summarizing information encoded by the Gaussian tokens. The effectiveness of choosing ℓ as a middle layer is further validated in Tab. 5.

The extracted hidden states $\mathbf { h } ^ { \mathrm { g s } }$ are further decoded into 3D Gaussian primitives through a lightweight MLP-based Gaussian head $\mathcal { H } ( \cdot )$ . Rather than mapping each token to a single Gaussian, we let each token decode G Gaussians simultaneously, providing additional representational capacity per token without expanding the LLM’s sequence length:

$$
\begin{array} { r } { \{ \mathbf { G } _ { i , j } \} _ { j = 1 } ^ { G } = \mathcal { H } ( \mathbf { h } _ { i } ^ { \mathrm { g s } } ) , \quad i = 1 , \ldots , M , } \end{array}\tag{5}
$$

yielding a total of $M \times G$ Gaussians per scene. Each Gaussian $\mathbf { G } _ { i , j } = \{ \mu _ { i , j } , \sigma _ { i , j } , \Sigma _ { i , j } , c _ { i , j } \}$ is parameterized by its 3D center $\mu \in \mathbb { R } ^ { 3 }$ , opacity $\sigma \in [ 0 , 1 )$ , covariance matrix $\Sigma \in \mathbb { R } ^ { 3 \times 3 }$ , and spherical harmonics coefficients $c \in \mathbb { R } ^ { 3 ( L _ { \mathrm { S H } } + 1 ) ^ { 2 } }$ that encode view-dependent color with $L _ { \mathrm { S H } }$ degrees. Together, the resulting set $\{ \mathbf { G } _ { i , j } \}$ forms a compact 3D representation of the scene that can be rendered into novel views via differentiable rasterization [42].

Training objective. We jointly train the model with two losses. The standard language modeling loss $\mathcal { L } _ { \mathrm { L M } }$ supervises the autoregressive prediction of text tokens conditioned on the visual and Gaussian summary tokens, while the reconstruction loss $\mathcal { L } _ { \mathrm { r e c o n } }$ supervises the decoded Gaussians by rendering them at the input viewpoints $\{ \pi _ { k } \} _ { k = 1 } ^ { K }$ via differentiable rasterization and comparing the resulting images to the ground-truth frames $I _ { k } \mathrm { \Delta }$

$$
\mathcal { L } _ { \mathrm { L M } } = - \sum _ { i = 1 } ^ { L } \log p _ { \theta , \phi } \bigl ( \mathbf { e } _ { i } ^ { \mathrm { t e x t } } \bigr | \mathbf { e } _ { < i } ^ { \mathrm { t e x t } } , \mathbf { e } ^ { \mathrm { i m g } } , \mathbf { e } ^ { \mathrm { g s } } \bigr ) ,\tag{6}
$$

$$
\mathcal { L } _ { \mathrm { r e c o n } } = \sum _ { k = 1 } ^ { K } \Bigl [ \lambda _ { \mathrm { M S E } } \mathcal { L } _ { \mathrm { M S E } } ( \hat { I } _ { k } , I _ { k } ) + \lambda _ { \mathrm { L P I P S } } \mathcal { L } _ { \mathrm { L P I P S } } ( \hat { I } _ { k } , I _ { k } ) \Bigr ] ,\tag{7}
$$

where $\hat { I } _ { k }$ is the image rendered from $\{ \mathbf { G } _ { i } \} _ { i = 1 } ^ { M }$ at viewpoint $\pi _ { k }$ . The combined objective is ${ \mathcal { L } } =$ $\begin{array} { r } { \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { r e c o n } } \mathcal { L } _ { \mathrm { r e c o n } } . } \end{array}$

## 3.2.2 Accelerating Convergence via Distillation from a Compact Gaussian Teacher

Although the model can in principle be trained end-to-end with the objective above, we find that pure photometric supervision through the LLM converges slowly, requiring substantially more iterations than is practical given the cost of MLLM finetuning. This is consistent with prior observations in the feed-forward 3D Gaussian Splatting literature, where dedicated estimators are typically trained for hundreds of thousands of iterations with large batch sizes to obtain reliable Gaussian predictions [16, 9, 83, 28, 37, 1]. To circumvent this bottleneck, we leverage a pretrained compact Gaussian estimator as a teacher model and distill its scene representation into the LLM, providing a strong inductive signal for what the Gaussian summary tokens should encode.

Compact Gaussian teacher. We adopt ZipSplat [72], a recent feed-forward 3DGS framework that estimates a compact set of 3D Gaussians from multi-view images, as our teacher. Given the same multi-view inputs I, ZipSplat first encodes them with a geometry-grounded visual backbone [74], and then refines a small set of M learnable query tokens $\mathbf { \check { Q } } ^ { \mathrm { T } } \in \check { \mathbb { R } } ^ { M \times d _ { \mathrm { T } } }$ through a transformer that jointly attends over the queries and the multi-view features, where $d _ { \mathrm { T } }$ denotes the teacher model’ hidden dimension. The refined tokens $\bar { \mathbf { Q } } ^ { \mathrm { T } } \in \mathbb { R } ^ { M \times d _ { \mathrm { T } } }$ are then decoded into 3D Gaussians $\{ \mathbf { G } _ { i , j } ^ { \mathrm { T } } \} _ { j = 1 } ^ { G }$ for $i = 1 , \dots , M$ through a lightweight Gaussian head. Crucially, we match the number of teacher tokens M in $\mathrm { Z i p S p l a t } ^ { \prime } \mathrm { s }$ inference time with the number of Gaussian summary tokens in our LLM, allowing direct token-level alignment between teacher and student.

(a) Image-to-Image Feature Attention  
(b) Image Feature PCA Visualization  
![](images/cf8965e589f1cba7430c9433f98ddcb0b8893477f6082901a9fdf5a6fc8621a7.jpg)  
Figure 2: Effects of joint reconstruction training on the LLM’s image features. (a) Image-toimage attention maps between a query point (red dot) and other input views. Our model exhibits substantially stronger cross-view correspondences than the baseline, indicating that the Gaussian summary tokens implicitly drive the image features to align across views. (b) PCA visualization of the LLM’s image features (first three components shown as RGB). Our features are noticeably cleaner and more semantically structured, with corresponding objects (e.g., chairs, desks) encoded with consistent colors across views, reflecting the emergence of object-level abstractions.

Distillation loss. We distill from the teacher at two complementary levels. First, we align the LLM’s hidden states at the Gaussian summary token positions, $\mathbf { h } ^ { \mathbf { g } \mathbf { s } } \in \mathbb { R } ^ { M \times \mathbf { \check { D } } }$ , with the teacher’s refined query tokens $\bar { \mathbf { Q } } ^ { \mathrm { T } }$ via a normalized cosine similarity loss with a lightweight projection layer $g ( \cdot )$ that maps to the teacher’s dimension d , $\begin{array} { r } { \mathcal { L } _ { \mathrm { t o k e n } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \left[ 1 - \cos \left( \frac { g ( \mathbf { h } _ { i } ^ { \mathrm { g s } } ) } { \| g ( \mathbf { h } _ { i } ^ { \mathrm { g s } } ) \| } , \ \frac { \bar { \mathbf { Q } } _ { i } ^ { \mathrm { T } } } { \| \bar { \mathbf { Q } } _ { i } ^ { \mathrm { T } } \| } \right) \right] } \end{array}$ , where $\cos ( \cdot , \cdot )$ denotes the cosine similarity operation and ∥ · ∥ is the L2-norm. This token-level supervision provides a dense, per-token target that guides the LLM toward representations the teacher’s Gaussian head can already decode, bypassing the slow convergence of learning Gaussian-decodable features from photometric loss alone. Second, we additionally supervise the student’s decoded Gaussian parameters against the teacher’s predictions with an L2 loss over all Gaussian attributes $\begin{array} { r } { \mathcal { L } _ { \mathrm { p a r a m } } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathopen { } \mathclose \bgroup \left\| \mathbf { G } _ { i } - \mathbf { G } _ { i } ^ { \mathrm { T } } \aftergroup \egroup \right\| _ { 2 } ^ { 2 } , } \end{array}$ where $\mathbf { G } _ { i }$ and $\mathbf { G } _ { i } ^ { \mathrm { T } }$ are taken as the concatenation of all predicted Gaussian attributes from the student and teacher, respectively. The total distillation loss is $\mathcal { L } _ { \mathrm { d i s t i l l } } = \lambda _ { \mathrm { t o k e n } } \mathcal { L } _ { \mathrm { t o k e n } } + \lambda _ { \mathrm { p a r a m } } \mathcal { L } _ { \mathrm { p a r a m } }$

Final objective. Combining the language modeling, reconstruction, and distillation losses, the full training objective of Imagine3D-LLM is:

$$
\mathcal { L } _ { \mathrm { f u l l } } = \mathcal { L } _ { \mathrm { L M } } + \lambda _ { \mathrm { r e c o n } } \mathcal { L } _ { \mathrm { r e c o n } } + \lambda _ { \mathrm { d i s t i l l } } \mathcal { L } _ { \mathrm { d i s t i l l } } .\tag{8}
$$

The teacher is kept frozen throughout training and is only used at training time.

## 3.3 Analysis

In this section, we further analyze the characteristics of the trained model to understand how jointly training the model to estimate coarse reconstructions of the scene leads to improved 3D reasoning.

Joint training improves image feature representations. While our reconstruction loss is applied only at the Gaussian summary tokens and does not directly supervise the image features, we observe that the LLM’s image features themselves become substantially more 3D-aware after joint training. Fig. 2-(a) visualizes the attention maps between a query point in one view (red dot) and the image features of the other input views. Compared to the baseline, our model produces noticeably sharper and more spatially focused attention on the corresponding region across views, indicating that the same object is now encoded with consistent representations regardless of viewpoint. Beyond this qualitative observation, we additionally evaluate cross-view correspondence accuracy of our model’s image features throughout training, and observe a steady improvement as training progresses $( \mathsf { A p } \cdot$ pendix B). Fig. 2-(b) makes the underlying structure more explicit by visualizing the first three principal components of the image-token hidden states at the ℓ-th LLM layer in RGB. While the baseline features appear noisy and largely unstructured, our features are markedly cleaner and more semantically organized, with corresponding regions encoded with consistent colors across viewpoints. Together, these two effects indicate that the reconstruction objective propagates 3D-aware structure beyond the Gaussian summary tokens into the LLM’s image features themselves, encouraging the same kind of compact, view-consistent representation that the summary tokens are trained to produce.

![](images/aee81a153bb7b73d7b0ad1f22ff61344e842b3dcc0dc0f527a8353ddb0335c0b.jpg)  
Figure 3: Visualization of clustered Gaussian from summary tokens. We visualize the Gaussians clustered from a set of Gaussian summary tokens via K-means over the summary tokens (cyan indicates the Gaussians belonging to one cluster). Similar summary tokens are mapped to the same object without any explicit clustering supervision.

Summary tokens emergently group by object. A central design choice of Imagine3D-LLM is the bottleneck $M \ll K N ^ { \prime }$ , which compels overlapping content across views to be merged into a shared set of summary tokens (§3.2.1). To examine whether this bottleneck actually induces object-level grouping, we cluster the trained summary tokens via K-means and visualize, for each cluster, the 3D Gaussians decoded from its member tokens. As shown in Fig. 3, a single cluster’s Gaussians reliably localize one object (e.g., the chair) across all views, despite the model receiving no clustering supervision. This emergent grouping effectively validates that learning to reconstruct the scene through a small set of summary tokens naturally guides the model toward the abstract, object-level understanding our design aims to achieve, without requiring any explicit object-level supervision.

## 4 Experiments

Implementation Details. Following prior work [93, 73], we build our Imagine3D-LLM based on LLaVA-Video-7B [89]. LLaVA-Video encodes each image as 210 tokens, resulting in $N ^ { \prime } = 2 1 0$ and $K N ^ { \prime } = 6 7 2 0$ tokens for most settings where we use 32 images as input. For 3DGS estimation, we choose $\ell = 1 4$ , as it is the middle layer of the base LLM. For the number of Gaussian summary tokens, we set $M = 2 5 9 2$ , which corresponds to 81 tokens per input image and is much fewer than the total number of image tokens $K \dot { N } ^ { \prime } = 6 7 2 0$ . The number of Gaussians decoded per token is set as $G = 3 2$ . The Gaussian decoder head H(·) is a three-layer MLP. We fine-tune the entire model on the combination of ScanNet-based datasets (SQA3D [52], ScanQA [3], Scan2Cap [11], ScanRefer [10], and Multi3DRefer [88]). Motivated by recent analysis [86, 95, 92] where training only with ScanNet-based datasets show largely overfitted results to ScanNet rather than unlocking 3D reasoning, we also sample and add 100K subset from SPAR-7M [87]. The training is conducted with 16 GH200 GPUs, with an effective batch size of 64 for 1 epoch. More detailed implementation details can be found in Appendix A.

Datasets and metrics. Following prior work [93, 73, 92, 30], we evaluate on seven benchmarks. Six are from ScanNet [20]: SQA3D [52], Real-3DQA [51], and ScanQA [3] for situated and spatial reasoning, Scan2Cap [11] for 3D grounded captioning, and ScanRefer [10] and Multi3DRefer [88] for 3D object detection. We additionally evaluate on SPAR-Bench [87], which assesses genuine 3D perception and reasoning across three cognitive levels: low-level (e.g., depth estimation), mediumlevel (e.g., view-change inference), and high-level (e.g., spatial imagination). Following prior work [96, 93, 73, 92, 23, 30], we use exact / refined-exact match (EM / EM-R) [32, 96, 51] on SQA3D and Real-3DQA, EM together with BLEU-4 [55], METEOR [5], ROUGE [47], and CIDEr [71] on ScanQA, the same captioning metrics under IoU@0.5 on Scan2Cap, Acc@0.25/0.5 on ScanRefer and the corresponding F1 on Multi3DRefer, and accuracy per cognitive level with the overall mean on SPAR-Bench.

Table 1: Quantitative results on 3D question-answering over SQA3D [52], Real-3DQA [51], and ScanQA [3]. “Specialists” refer to models tailored to individual tasks via task-specific decoders. General-purpose 2D MLLMs [67, 75, 89] are reported under a zero-shot protocol. We report values available from prior works; “–” marks entries unavailable to us.
<table><tr><td rowspan="2">Method</td><td colspan="2">SQA3Dtest</td><td colspan="2">Real-3DQAtest</td><td colspan="5">ScanQAval</td></tr><tr><td>EM</td><td>EM-R</td><td>EM</td><td>EM-R</td><td>CIDEr</td><td>BLEU-4</td><td>METEOR</td><td>ROUGE</td><td>EM</td></tr><tr><td>Specialists</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SQA3D [52]</td><td>46.6</td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ScanQA [3]</td><td>一</td><td>一</td><td></td><td>一</td><td>64.9</td><td>10.1</td><td>13.1</td><td>33.3</td><td>21.1</td></tr><tr><td>2D MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InternVL2-8B [67]</td><td>33.0</td><td>45.3</td><td></td><td></td><td>62.5</td><td>3.3</td><td>14.5</td><td>34.3</td><td></td></tr><tr><td>Qwen2-VL-7B [75]</td><td>40.7</td><td>46.7</td><td></td><td></td><td>53.9</td><td>3.0</td><td>11.4</td><td>29.3</td><td></td></tr><tr><td>LLaVA-Video-7B [89]</td><td>48.5</td><td></td><td></td><td></td><td>88.7</td><td>3.1</td><td>17.7</td><td>44.6</td><td></td></tr><tr><td>3D MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scene-LLM [25]</td><td>53.6</td><td></td><td></td><td></td><td>80.0</td><td>11.7</td><td>15.8</td><td>35.9</td><td>27.2</td></tr><tr><td>LL3DA [13]]</td><td></td><td></td><td></td><td></td><td>76.8</td><td></td><td>15.9</td><td>37.3</td><td></td></tr><tr><td>LEO [32]</td><td>50.0</td><td>52.4</td><td></td><td></td><td>80.0</td><td>11.5</td><td>16.2</td><td>39.3</td><td>21.5</td></tr><tr><td>ChatScene [31]</td><td>54.6</td><td>57.5</td><td></td><td></td><td>87.7</td><td>14.3</td><td>18.0</td><td>41.6</td><td>21.6</td></tr><tr><td>Grounded 3D-LLM [15]</td><td></td><td></td><td></td><td></td><td>72.7</td><td>13.4</td><td></td><td></td><td></td></tr><tr><td>LLaVA-3D [96]</td><td>55.6</td><td>57.6</td><td></td><td></td><td>91.7</td><td>14.5</td><td>20.7</td><td>50.1</td><td>27.0</td></tr><tr><td>Video-3D-LLM [93]</td><td>58.6</td><td></td><td>31.2</td><td>36.4</td><td>102.1</td><td>16.4</td><td>20.0</td><td>49.3</td><td>30.1</td></tr><tr><td>3DRS [33]</td><td>60.6</td><td></td><td></td><td></td><td>104.8</td><td></td><td></td><td></td><td>30.3</td></tr><tr><td>Ross3D [73]</td><td>63.0</td><td>65.7</td><td>36.6</td><td>41.5</td><td>107.0</td><td>17.9</td><td>20.9</td><td>50.7</td><td>30.8</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>63.8</td><td>66.4</td><td>39.2</td><td>44.3</td><td>109.3</td><td>20.5</td><td>21.3</td><td>49.7</td><td>29.9</td></tr></table>

Table 2: Evaluation of 3D dense captioning and visual grounding on Scan2Cap [11], ScanRefer [10], and Multi3DRefer [88].
<table><tr><td rowspan="2">Method</td><td colspan="4">Scan2Capval (IoU@0.5)</td><td colspan="2">ScanReferval</td><td colspan="2">Multi3DReferval</td></tr><tr><td>ROUGE</td><td>BLEU-4</td><td>METEOR</td><td>CIDEr</td><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td></tr><tr><td>Specialists</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Scan2Cap [11]</td><td>44.5</td><td>23.3</td><td>22.0</td><td>35.2</td><td></td><td></td><td>1</td><td></td></tr><tr><td>3DJCG [8]</td><td>50.8</td><td>31.0</td><td>24.2</td><td>49.5</td><td>49.6</td><td>37.3</td><td></td><td>26.6</td></tr><tr><td>ScanRefer [10]</td><td>一</td><td>一</td><td>一</td><td>一</td><td>37.3</td><td>24.3</td><td>1</td><td></td></tr><tr><td>M3DRef-CLIP [88]</td><td>一</td><td>一</td><td>一</td><td>一</td><td>51.9</td><td>44.7</td><td>42.8</td><td>38.4</td></tr><tr><td>3D MLLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LL3DA [13]</td><td>55.1</td><td>36.8</td><td>26.0</td><td>65.2</td><td></td><td></td><td></td><td></td></tr><tr><td>Ground 3D-LLM [15]</td><td></td><td></td><td></td><td></td><td>47.9</td><td>44.1</td><td>45.2</td><td>40.6</td></tr><tr><td>LEO [32]</td><td>58.1</td><td>38.2</td><td>27.9</td><td>72.4</td><td></td><td></td><td>1</td><td></td></tr><tr><td>ChatScene [31]</td><td>58.1</td><td>36.3</td><td></td><td>77.1</td><td>55.5</td><td>50.2</td><td>57.1</td><td>52.4</td></tr><tr><td>LLaVA-3D [96]</td><td>63.4</td><td>41.1</td><td>30.2</td><td>79.2</td><td>54.1</td><td>42.4</td><td></td><td></td></tr><tr><td>Video-3D-LLM [93]</td><td>62.3</td><td>42.4</td><td>28.9</td><td>83.8</td><td>58.1</td><td>51.7</td><td>58.0</td><td>52.7</td></tr><tr><td>VG-LLM [92]</td><td>62.6</td><td>41.5</td><td>28.9</td><td>80.0</td><td></td><td></td><td></td><td></td></tr><tr><td>3DRS [33]</td><td></td><td>41.6</td><td></td><td>86.1</td><td></td><td></td><td></td><td></td></tr><tr><td>Ross3D [73]</td><td>66.9</td><td>43.4</td><td>30.3</td><td>81.3</td><td>61.1</td><td>54.4</td><td>59.6</td><td>54.3</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>67.6</td><td>48.3</td><td>32.6</td><td>99.2</td><td>62.8</td><td>56.3</td><td>60.2</td><td>55.0</td></tr></table>

## 4.1 Quantitative Comparison

ScanNet-based datasets. On the ScanNet-based benchmarks (SQA3D [52], Real-3DQA [51], ScanQA [3], Scan2Cap [11], ScanRefer [10], and Multi3DRefer [88]), we compare Imagine3D-LLM against task-specific specialists, general-purpose 2D LMMs, and recent 3D LMMs that differ in how they encode the 3D scene. As shown in Tab. 1, and Tab. 2, Imagine3D-LLM significantly improves performance across benchmarks, outperforming the strongest prior 3D LMM (Ross3D) on almost all benchmarks. Especially, state-of-the-art performance in Real-3DQA is notable, which is a benchmark explicitly designed to validate 3D reasoning capabilities and penalize language-shortcut overfitting [51], indicating that our improvements stem from genuine 3D reasoning.

SPAR-Bench. On SPAR-Bench [87], we compare Imagine3D-LLM against three categories of models: the proprietary group [34, 2], the general-purpose 2D LMMs group [89, 75, 67, 4], and the spatially-aware group of recent models specifically designed for spatial reasoning [23, 30, 17].

As shown in Tab. 3, Imagine3D-LLM achieves state-of-the-art performance on SPAR-Bench, outperforming all competing methods across every cognitive level by substantial margins. Compared to general-purpose 2D LMMs, Imagine3D-LLM surpasses even the largest 72Bscale model [4] by over 29 points despite using a 7B backbone. More notably, Imagine3D-LLM outperforms the strongest spatially-aware baseline, 3DThinker-7B [17], by 5.2 points overall, with even larger margins over $\mathrm { G ^ { 2 } V L M { - } S R { - } }$ 2B [30] and VLM3R-7B [23]. The improvement is especially pronounced on the mediumand high-level splits, which most directly require the object-level cross-view reasoning our reconstruction objective targets. These results validate our central claim: equipping MLLMs to assemble a compact 3D representation of the scene before answering substantially improves 3D reasoning.

Table 3: Evaluation on SPAR-Bench [87].
<table><tr><td>Model</td><td colspan="3">SPAR-Bench</td></tr><tr><td>Proprietary</td><td>Avg. Low</td><td>Med.</td><td>High</td></tr><tr><td>GPT-4o [34] Claude-3.7-Sonnet [2]</td><td>36.4 29.3 21.8 25.4</td><td>24.9 7.3</td><td>45.1 23.3</td></tr><tr><td>General-purpose 2D MLLMs LLaVA-Video-7B [89] Qwen2-VL-7B [75] InternVL2.5-8B [18]</td><td>32.3 23.6 30.7 27.5 36.3 29.5</td><td>24.8 20.4 31.9</td><td>42.6 37.0 43.8</td></tr><tr><td>Spatially-aware VLM3R-7B [23] G2VLM-SR-2B [30]</td><td>43.2 54.9</td><td>39.8 60.0</td><td>28.4 51.2 36.3</td></tr><tr><td>Imagine3D-LLM-7B (Ours)</td><td>63.3</td><td>-</td><td>-</td></tr><tr><td>3DThinker-7B [17]</td><td></td><td></td><td>56.5 -</td></tr></table>

## 4.2 Ablation Studies

Controlled comparison with baseline. To isolate the contribution of our mentalreconstruction objective from the effect of the training data and recipe, we compare Imagine3D-LLM against a baseline that shares the identical backbone, training data, and schedule but omits the Gaussian summary tokens together with the reconstruction and distillation objectives, i.e., standard visual instruction tuning. Here, we report one representative metric per benchmark EM score for SQA3D / ScanQA, ROUGE for Scan2Cap, Acc@25 for ScanRefer, F1@0.25 for Multi3DRefer, and

Table 4: Controlled comparison against the base MLLM. Both models share the identical backbone, training data, and schedule; the baseline is trained with standard visual instruction tuning, without the Gaussian summary tokens or the reconstruction and distillation objectives.

<table><tr><td>Method</td><td>SQA3D</td><td>ScanQA</td><td>Scan2Cap</td></tr><tr><td>Baseline</td><td>56.5</td><td>26.2</td><td>63.1</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>63.8</td><td>29.9</td><td>67.6</td></tr><tr><td>Method</td><td></td><td>ScanRefer Multi3DRefer</td><td>SPAR</td></tr><tr><td>Baseline</td><td>58.3</td><td>57.4</td><td>60.9</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>62.8</td><td>60.2</td><td>68.5</td></tr></table>

average performance SPAR. As shown in Tab. 4, Imagine3D-LLM improves over this controlled baseline on every benchmark, confirming that the gains arise from our objective rather than from the additional training data alone. The improvement is especially pronounced on the medium-level cross-view split of SPAR-Bench (+15.5, from 51.5 to 67.0; see Tab. 3), which most directly exercises the object-level cross-view reasoning our reconstruction objective targets.

Layer selection for Gaussian decoding. We further ablate the choice of the LLM layer ℓ from which the Gaussian summary tokens are extracted and decoded into 3D Gaussians. As discussed in § 3.2.1, we hypothesize that the middle layer of the LLM offers the best balance between two opposing requirements: the Gaussian tokens must accumulate enough visual information to faithfully describe the scene, yet the remaining layers must be sufficient to propagate this 3D-aware information into the text tokens for reasoning. We empirically verify this by extracting from layer $\ell \in \{ 7 , 1 4 , 2 1 \}$ of the 28-layer LLM. As shown in Tab. 5, the middle layer $( \dot { \ell } = 1 4 )$ consistently performs best, validating our design choice and aligning with prior findings on information flow in MLLMs [40, 38, 84].

Table 5: Effect of the layer ℓ from which Gaussian tokens are decoded.
<table><tr><td>Layer l</td><td> $\mathrm { S Q A } 3 \mathrm { D } _ { \mathrm { t e s t } }$ </td><td>SPAR-Bench</td></tr><tr><td>7 (early)</td><td>60.7</td><td>60.2</td></tr><tr><td>14 (middle, Ours)</td><td>63.8</td><td>68.5</td></tr><tr><td>21 (late)</td><td>61.7</td><td>64.0</td></tr></table>

Effectiveness of distillation. We compare our full model against a variant trained without the distillation losses $( \mathcal { L } _ { \mathrm { t o k e n } }$ and ${ \mathcal { L } } _ { \mathrm { p a r a m } } )$ . As shown in Tab. 6, removing distillation (see Recon. only 1 epoch) causes a substantial drop on both benchmarks. We empirically observe that without distillation, $\mathcal { L } _ { \mathrm { r e c o n } }$ remains large in scale even after 1 epoch, so a large fraction of the gradient signal is allocated to reducing $\mathcal { L } _ { \mathrm { r e c o n } }$ , leaving the language modeling objective under-optimized. Consistent with this, the vari-

Table 6: Effect of distillation from the compact Gaussian teacher.
<table><tr><td>Method</td><td> $\mathrm { S Q A } 3 \mathrm { D } _ { \mathrm { t e s t } }$ </td><td>SPAR-Bench</td></tr><tr><td>Baseline (1 ep.)</td><td>56.5</td><td>60.9</td></tr><tr><td>Baseline (2 ep.)</td><td>57.2</td><td>61.5</td></tr><tr><td>Baseline (4 ep.)</td><td>52.3</td><td>55.2</td></tr><tr><td>Recon. only (1 ep.)</td><td>57.7</td><td>57.6</td></tr><tr><td>Recon. only (2 ep.)</td><td>61.9</td><td>65.1</td></tr><tr><td>Recon. only (4 ep.)</td><td>63.7</td><td>67.9</td></tr><tr><td>Imagine3D-LLM (Full, 1 ep.)</td><td>63.8</td><td>68.5</td></tr></table>

ant improves steadily with longer training $( 1  2  4 )$ epochs) and at 4 epochs nearly matches our full 1-epoch model. In contrast, the ${ \mathcal { L } } _ { \mathrm { L M } ^ { - } } { \mathrm { o n l y } }$ baseline shows only modest improvements with slightly longer iterations (1 →2 epochs), and performance decreases after 4 epochs due to the model overfitting to the training data and losing general reasoning performance when only trained with $\mathcal { L } _ { \mathrm { L M } }$ . Together, these results indicate that the improved 3D reasoning comes primarily from jointly learning reconstruction and understanding, with distillation serving mainly to accelerate convergence and make training practical.

Isolating the source of the gains. Having established that reconstruction, not longer training, drives the gains, we next ask whether the improvement truly comes from learning to reconstruct the scene (mental reconstruction), or merely from access to the teacher’s representation, the distillation signal, or the added token capacity. We disentangle these factors in Tab. 7. First, feeding the teacher’s query tokens directly to the LLM as additional input (+ Teacher tokens), without asking the model to reconstruct, performs on par with the baseline. Second, supervising the summary tokens with the distillation loss but without the reconstruction loss (+

Table 7: Isolating the source of the gains. + Teacher tokens: the teacher’s query tokens are fed directly to the LLM as input. + Distill. only: summary tokens supervised by $\mathcal { L } _ { \mathrm { d i s t i l l } }$ but without ${ \mathcal { L } } _ { \mathrm { r e c o n } } .$
<table><tr><td>Variant</td><td> $\mathrm { S Q A } 3 \mathrm { D } _ { \mathrm { t e s t } }$ </td><td>SPAR-Bench</td></tr><tr><td>Baseline</td><td>56.5</td><td>60.9</td></tr><tr><td>+ Teacher tokens</td><td>56.4</td><td>60.8</td></tr><tr><td>+ Distill. only</td><td>56.6</td><td>60.5</td></tr><tr><td>+ Recon. (4 ep.)</td><td>63.7</td><td>67.9</td></tr><tr><td>Full (Ours, 1 ep.)</td><td>63.8</td><td>68.5</td></tr></table>

Distill. only) likewise fails to help. We attribute both negative results to a representational mismatch between the teacher’s query tokens and the LLM’s own embedding space: simply injecting or distilling these tokens does not teach the LLM how to interpret or attend to them, and without a dedicated mechanism to compensate for this misalignment or to explicitly force the language model to attend to the injected tokens, as in feature-fusion approaches such as VLM3R [23] the added tokens are largely ignored, leaving performance at the baseline level. In contrast, the reconstruction loss compels the Gaussian summary tokens to actively attend to the image tokens and extract the visual evidence required to estimate their 3D Gaussian parameters. This active aggregation reshapes the attention and representations within the LLM such as sharpening cross-view correspondence and inducing object-level grouping (§3.3), which we believe iss what benefits downstream reasoning. Together, these results indicate that the improvement arises primarily from jointly learning mental reconstruction.

Number of Gaussian summary tokens. An important design choice of Imagine3D-LLM is that the Gaussian summary tokens act as a representational bottleneck, encouraging multi-view content corresponding to the same 3D region to be merged into shared tokens (§ 3.2.1). To validate this design, we ablate the number of summary tokens M while keeping all other components fixed. As reported in Tab. 8, we compare introducing roughly 40,

Table 8: Number of Gaussian summary tokens M.
<table><tr><td>M</td><td> $\mathrm { S Q A } 3 \mathrm { D } _ { \mathrm { t e s t } }$ </td><td>SPAR-Bench</td></tr><tr><td>1296</td><td>61.9</td><td>64.6</td></tr><tr><td>2592 (Ours)</td><td>63.8</td><td>68.5</td></tr><tr><td>5184</td><td>60.3</td><td>58.4</td></tr></table>

80, 160 tokens per image, which makes the total number of summary tokens 1296, 2592, 5184, given 32 images. The evaluation shows that too few tokens (1296) provide insufficient capacity, while too many tokens (5184) weaken the bottleneck that drives object-level grouping, leading both extremes to underperform our chosen setting.

## 5 Conclusion

We presented Imagine3D-LLM, a framework that equips a multi-image MLLM with the ability to perform an explicit mental reconstruction of a 3D scene before answering. We introduce a compact set of learnable Gaussian summary tokens that are appended to the image tokens and decoded into a compact 3DGS representation, jointly trained with the standard language modeling objective. Across challenging spatial reasoning benchmarks, Imagine3D-LLM significantly outperforms prior approaches that rely on explicit pixel-level geometry or geometry-foundation features, and our analyses further reveal that the reconstruction objective propagates beyond the Gaussian summary tokens, reshaping the image features into more spatially aligned and semantically structured representations. We believe that enabling machines to understand how the scene is structured before answering offers a promising step toward MLLMs that perceive 3D space in a more human-like manner.

## Acknowledgements

This work was supported by the Swiss AI Initiative through a grant from the Swiss National Supercomputing Centre (CSCS) under project ID ab036 on the Alps infrastructure; the Swiss National Science Foundation (Advanced Grant 216260: "Beyond Frozen Worlds: Capturing Functional 3D Digital Twins from the Real World"); the European Union’s Horizon Europe research and innovation programme under grant agreement No. 101214398 (ELLIOT); the Horizon Europe programme under grant agreement No. 101298421 (GRAIL); and the ETH AI Center through an ETH AI Center postdoc fellowship awarded to Sunghwan Hong.

## References

[1] Honggyu An, Jaewoo Jung, Mungyeom Kim, Sunghwan Hong, Chaehyun Kim, Kazumi Fukuda, Minkyeong Jeon, Jisang Han, Takuya Narihira, Hyuna Ko, et al. C3g: Learning compact 3d representations with 2k gaussians. arXiv preprint arXiv:2512.04021, 2025.

[2] Anthropic. Claude 3.7 sonnet system card. https://www.anthropic.com/news/ claude-3-7-sonnet, 2025. Hybrid reasoning multimodal model released November 2025.

[3] Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2022.

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv:2502.13923, 2025.

[5] Satanjeev Banerjee and Alon Lavie. Meteor: An automatic metric for mt evaluation with improved correlation with human judgments. In Proceedings of the ACL workshop on intrinsic and extrinsic evaluation measuresfor machine translation and/or summarization, pages 65–72, 2005.

[6] Eric Brachmann, Tommaso Cavallari, and Victor Adrian Prisacariu. Accelerated coordinate encoding: Learning to relocalize in minutes using rgb and poses. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5044–5053, 2023.

[7] Neil Burgess. Spatial memory: how egocentric and allocentric combine. Trends in cognitive sciences, 10 (12):551–557, 2006.

[8] Daigang Cai, Lichen Zhao, Jing Zhang, Lu Sheng, and Dong Xu. 3djcg: A unified framework for joint dense captioning and visual grounding on 3d point clouds. In CVPR, 2022.

[9] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19457–19467, 2024.

[10] Dave Zhenyu Chen, Angel X. Chang, and Matthias Nießner. Scanrefer: 3d object localization in RGB-D scans using natural language. In ECCV, 2020.

[11] Dave Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X. Chang. Scan2cap: Context-aware dense captioning in RGB-D scans. In CVPR, 2021.

[12] Sijin Chen, Hongyuan Zhu, Xin Chen, Yinjie Lei, Gang Yu, and Tao Chen. End-to-end 3d dense captioning with vote2cap-detr. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11124–11133, 2023.

[13] Sijin Chen, Xin Chen, Chi Zhang, Mingsheng Li, Gang Yu, Hao Fei, Hongyuan Zhu, Jiayuan Fan, and Tao Chen. LL3DA: visual interactive instruction tuning for omni-3d understanding, reasoning, and planning. In CVPR, 2024.

[14] Sijin Chen, Hongyuan Zhu, Mingsheng Li, Xin Chen, Peng Guo, Yinjie Lei, YU Gang, Taihao Li, and Tao Chen. Vote2cap-detr++: Decoupling localization and describing for end-to-end 3d dense captioning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2024.

[15] Yilun Chen, Shuai Yang, Haifeng Huang, Tai Wang, Ruiyuan Lyu, Runsen Xu, Dahua Lin, and Jiangmiao Pang. Grounded 3d-llm with referent tokens. arXiv:2405.10370, 2024.

[16] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In European Conference on Computer Vision, pages 370–386. Springer, 2024.

[17] Zhangquan Chen, Manyuan Zhang, Xinlei Yu, Xufang Luo, Mingze Sun, Zihao Pan, Xiang An, Yan Feng, Peng Pei, Xunliang Cai, et al. Think with 3d: Geometric imagination grounded spatial reasoning from limited views. arXiv preprint arXiv:2510.18632, 2025.

[18] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.

[19] Claudia Cuttano, Gabriele Trivigno, Christoph Reich, Daniel Cremers, Carlo Masone, and Stefan Roth. Insid3: Training-free in-context segmentation with dinov3. In Third Workshop on Visual Concepts, 2026.

[20] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas A. Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In CVPR, 2017.

[21] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-weighted linear units for neural network function approximation in reinforcement learning. Neural networks, 107:3–11, 2018.

[22] Francis Engelmann, Fabian Manhardt, Michael Niemeyer, Keisuke Tateno, Marc Pollefeys, and Federico Tombari. OpenNeRF: Open Set 3D Neural Scene Segmentation with Pixel-Wise Features and Rendered Novel Views. In International Conference on Learning Representations, 2024.

[23] Zhiwen Fan, Jian Zhang, Renjie Li, Junge Zhang, Runjin Chen, Hezhen Hu, Kevin Wang, Huaizhi Qu, Dilin Wang, Zhicheng Yan, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.

[24] Elisabetta Fedele, Boyang Sun, Leonidas Guibas, Marc Pollefeys, and Francis Engelmann. Superdec: 3d scene decomposition with superquadrics primitives. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 24625–24635, October 2025.

[25] Rao Fu, Jingyu Liu, Xilun Chen, Yixin Nie, and Wenhan Xiong. Scene-llm: Extending language model for 3d visual reasoning. In WACV, 2025.

[26] Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T Freeman. Unsupervised semantic segmentation by distilling feature correspondences. arXiv preprint arXiv:2203.08414, 2022.

[27] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

[28] Sunghwan Hong, Jaewoo Jung, Heeseong Shin, Jisang Han, Jiaolong Yang, Chong Luo, and Seungryong Kim. Pf3plat: Pose-free feed-forward 3d gaussian splatting. arXiv preprint arXiv:2410.22128, 2024.

[29] Yining Hong, Haoyu Zhen, Peihao Chen, Shuhong Zheng, Yilun Du, Zhenfang Chen, and Chuang Gan. 3d-llm: Injecting the 3d world into large language models. In NeurIPS, 2023.

[30] Wenbo Hu, Jingli Lin, Yilin Long, Yunlong Ran, Lihan Jiang, Yifan Wang, Chenming Zhu, Runsen Xu, Tai Wang, and Jiangmiao Pang. G<sup>2</sup>VLM: Geometry grounded vision language model with unified 3d reconstruction and spatial reasoning. arXiv preprint arXiv:2511.21688, 2025.

[31] Haifeng Huang, Yilun Chen, Zehan Wang, Rongjie Huang, Runsen Xu, Tai Wang, Luping Liu, Xize Cheng, Yang Zhao, Jiangmiao Pang, et al. Chat-scene: Bridging 3d scene and large language models with object identifiers. In NeurIPS, 2024.

[32] Jiangyong Huang, Silong Yong, Xiaojian Ma, Xiongkun Linghu, Puhao Li, Yan Wang, Qing Li, Song-Chun Zhu, Baoxiong Jia, and Siyuan Huang. An embodied generalist agent in 3d world. In ICML, 2024.

[33] Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. 3drs: Mllms need 3d-aware representation supervision for scene understanding. arXiv preprint arXiv:2506.01946, 2025.

[34] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv:2410.21276, 2024.

[35] Roni Itkin, Noam Issachar, Yehonatan Keypur, Anpei Chen, and Sagie Benaim. Globalsplat: Efficient feed-forward 3d gaussian splatting via global scene tokens. arXiv preprint arXiv:2604.15284, 2026.

[36] Krishna Murthy Jatavallabhula, Alihusein Kuwajerwala, Qiao Gu, Mohd Omama, Tao Chen, Shuang Li, Ganesh Iyer, Soroush Saryazdi, Nikhil Keetha, Ayush Tewari, Joshua B. Tenenbaum, Celso Miguel de Melo, Madhava Krishna, Liam Paull, Florian Shkurti, and Antonio Torralba. Conceptfusion: Open-set multimodal 3d mapping. Robotics: Science and Systems (RSS), 2023.

[37] Lihan Jiang, Yucheng Mao, Linning Xu, Tao Lu, Kerui Ren, Yichen Jin, Xudong Xu, Mulin Yu, Jiangmiao Pang, Feng Zhao, et al. Anysplat: Feed-forward 3d gaussian splatting from unconstrained views. arXiv preprint arXiv:2505.23716, 2025.

[38] Zhangqi Jiang, Junkai Chen, Beier Zhu, Tingjin Luo, Yankun Shen, and Xu Yang. Devils in middle layers of large vision-language models: Interpreting, detecting and mitigating object hallucinations via attention lens. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 25004–25014, 2025.

[39] Kim Jun-Seong, GeonU Kim, Kim Yu-Ji, Yu-Chiang Frank Wang, Jaesung Choe, and Tae-Hyun Oh. Dr. splat: Directly referring 3d gaussian splatting via direct language embedding registration. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 14137–14146, 2025.

[40] Omri Kaduri, Shai Bagon, and Tali Dekel. What’s in the image? a deep-dive into the vision of vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 14549–14558, 2025.

[41] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. Your large vision-language model only needs a few attention heads for visual grounding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9339–9350, 2025.

[42] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, George Drettakis, et al. 3d gaussian splatting for real-time radiance field rendering. ACM Trans. Graph., 42(4):139–1, 2023.

[43] Justin\* Kerr, Chung Min\* Kim, Ken Goldberg, Angjoo Kanazawa, and Matthew Tancik. Lerf: Language embedded radiance fields. In International Conference on Computer Vision (ICCV), 2023.

[44] Phillip Y. Lee, Jihyeon Je, Chanho Park, Mikaela Angelina Uy, Leonidas Guibas, and Minhyuk Sung. Perspective-aware reasoning in vision-language models via mental imagery simulation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.

[45] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv:2408.03326, 2024.

[46] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022.

[47] Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pages 74–81, 2004.

[48] Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. VILA: on pre-training for visual language models. In CVPR, 2024.

[49] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In NeurIPS, 2023.

[50] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations (ICLR), 2024.

[51] Xianzheng Ma, Tao Sun, Shuai Chen, Yash Bhalgat, Jindong Gu, Angel X Chang, Iro Armeni, Iro Laina, Songyou Peng, and Victor Adrian Prisacariu. Do 3d large language models really understand 3d spatial relationships? In The Fourteenth International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2603.23523.

[52] Xiaojian Ma, Silong Yong, Zilong Zheng, Qing Li, Yitao Liang, Song-Chun Zhu, and Siyuan Huang. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022.

[53] Yongsen Mao, Junhao Zhong, Chuan Fang, Jia Zheng, Rui Tang, Hao Zhu, Ping Tan, and Zihan Zhou. Spatiallm: Training large language models for structured indoor modeling. arXiv preprint arXiv:2506.07491, 2025.

[54] Tom Monnier, Jake Austin, Angjoo Kanazawa, Alexei Efros, and Mathieu Aubry. Differentiable blocks world: Qualitative 3d decomposition by rendering primitives. Advances in Neural Information Processing Systems, 36:5791–5807, 2023.

[55] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the Association for Computational Linguistics (ACL), pages 311–318, 2002.

[56] Despoina Paschalidou, Ali Osman Ulusoy, and Andreas Geiger. Superquadrics revisited: Learning 3d shape parsing beyond cuboids. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10344–10353, 2019.

[57] Despoina Paschalidou, Luc Van Gool, and Andreas Geiger. Learning unsupervised hierarchical part decomposition of 3d objects from a single rgb image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1060–1070, 2020.

[58] Zhangyang Qi, Zhixiong Zhang, Ye Fang, Jiaqi Wang, and Hengshuang Zhao. Gpt4scene: Understand 3d scenes from videos with vision-language models. arXiv:2501.01428, 2025.

[59] Minghan Qin, Wanhua Li, Jiawei Zhou, Haoqian Wang, and Hanspeter Pfister. Langsplat: 3d language gaussian splatting. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 20051–20060, 2024.

[60] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[61] Jiawei Ren, Michal Jan Tyszkiewicz, Jiahui Huang, and Zan Gojcic. Tokengs: Decoupling 3d gaussian prediction from pixels with learnable tokens. arXiv preprint arXiv:2604.15239, 2026.

[62] William Shen, Ge Yang, Alan Yu, Jansen Wong, Leslie Pack Kaelbling, and Phillip Isola. Distilled feature fields enable few-shot language-guided manipulation. arXiv preprint arXiv:2308.07931, 2023.

[63] Roger N Shepard and Jacqueline Metzler. Mental rotation of three-dimensional objects. Science, 171 (3972):701–703, 1971.

[64] Jin-Chuan Shi, Miao Wang, Hao-Bin Duan, and Shao-Hua Guan. Language embedded 3d gaussians for open-vocabulary scene understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5333–5343, 2024.

[65] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

[66] Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv:2403.05530, 2024.

[67] OpenGVLab Team. InternVL2: Better than the Best—Expanding Performance Boundaries of Open-Source Multimodal Models with the Progressive Scaling Strategy, 2024. URL https://internvl.github.io/ blog/2024-07-02-InternVL-2.0/.

[68] Anh Thai, Songyou Peng, Kyle Genova, Leonidas Guibas, and Thomas Funkhouser. Splattalk: 3d vqa with gaussian splatting. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4712–4721, 2025.

[69] Peter Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Adithya Jairam Vedagiri IYER, Sai Charitha Akula, Shusheng Yang, Jihan Yang, Manoj Middepogu, Ziteng Wang, et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. In NeurIPS, 2024.

[70] Shubham Tulsiani, Hao Su, Leonidas J Guibas, Alexei A Efros, and Jitendra Malik. Learning shape abstractions by assembling volumetric primitives. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2635–2643, 2017.

[71] Ramakrishna Vedantam, C Lawrence Zitnick, and Devi Parikh. Cider: Consensus-based image description evaluation. pages 4566–4575, 2015.

[72] Alexander Veicht, Sunghwan Hong, Dániel Baráth, and Marc Pollefeys. Zipsplat: Fewer gaussians, better splats. arXiv preprint arXiv:2606.05102, 2026.

[73] Haochen Wang, Yucheng Zhao, Tiancai Wang, Haoqiang Fan, Xiangyu Zhang, and Zhaoxiang Zhang. Ross3d: Reconstructive visual instruction tuning with 3d-awareness, 2025.

[74] Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotný. VGGT: visual geometry grounded transformer. arXiv:2503.11651, 2025.

[75] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv:2409.12191, 2024.

[76] Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A Efros, and Angjoo Kanazawa. Continuous 3d perception model with persistent state. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 10510–10522, 2025.

[77] Yunsong Wang, Tianxin Huang, Hanlin Chen, and Gim Hee Lee. Freesplat: Generalizable 3d gaussian splatting towards free view synthesis of indoor scenes. Advances in Neural Information Processing Systems, 37:107326–107349, 2024.

[78] Zehan Wang, Haifeng Huang, Yang Zhao, Ziang Zhang, and Zhou Zhao. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023.

[79] Runsen Xu, Xiaolong Wang, Tai Wang, Yilun Chen, Jiangmiao Pang, and Dahua Lin. Pointllm: Empowering large language models to understand point clouds. In ECCV, 2024.

[80] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[81] Jihan Yang, Runyu Ding, Weipeng Deng, Zhe Wang, and Xiaojuan Qi. Regionplc: Regional point-language contrastive learning for open-world 3d scene understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19823–19832, 2024.

[82] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember, and recall spaces. arXiv:2412.14171, 2024.

[83] Botao Ye, Sifei Liu, Haofei Xu, Xueting Li, Marc Pollefeys, Ming-Hsuan Yang, and Songyou Peng. No pose, no problem: Surprisingly simple 3d gaussian splats from sparse unposed images. arXiv preprint arXiv:2410.24207, 2024.

[84] Heeji Yoon, Jaewoo Jung, Junwan Kim, Hyungyu Choi, Heeseong Shin, Sangbeom Lim, Honggyu An, Chaehyun Kim, Jisang Han, Donghyun Kim, Chanho Eom, Sunghwan Hong, and Seungryong Kim. Visual representation alignment for multimodal large language models, 2025. URL https://arxiv.org/abs/ 2509.07979.

[85] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[86] Huanyu Zhang, Chengzu Li, Wenshan Wu, Shaoguang Mao, Yifan Zhang, Haochen Tian, Ivan Vulic,´ Zhang Zhang, Liang Wang, Tieniu Tan, et al. Scaling and beyond: Advancing spatial reasoning in mllms requires new recipes. arXiv preprint arXiv:2504.15037, 2025.

[87] Jiahui Zhang, Yurui Chen, Yanpeng Zhou, Yueming Xu, Ze Huang, Jilin Mei, Junhui Chen, Yu-Jie Yuan, Xinyue Cai, Guowei Huang, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv:2503.22976, 2025.

[88] Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 15225– 15236, 2023.

[89] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv:2410.02713, 2024.

[90] Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 19781–19791, 2025.

[91] Tony Z Zhao, Vikash Kumar, Sergey Levine, and Chelsea Finn. Learning fine-grained bimanual manipulation with low-cost hardware. arXiv preprint arXiv:2304.13705, 2023.

[92] Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025.

[93] Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d LLM: learning position-aware video representation for 3d scene understanding. In CVPR, 2025.

[94] Mingquan Zhou, Chen He, Ruiping Wang, and Xilin Chen. Ov3d-cg: Open-vocabulary 3d instance segmentation with contextual guidance. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 5305–5314, 2025.

[95] Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Eric Xin Wang, and Achuta Kadambi. Vlm4d: Towards spatiotemporal awareness in vision language models. 2025.

[96] Chenming Zhu, Tai Wang, Wenwei Zhang, Jiangmiao Pang, and Xihui Liu. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv:2409.18125, 2024.

[97] Xueyan Zou, Yuchen Song, Ri-Zhao Qiu, Xuanbin Peng, Jianglong Ye, Sifei Liu, and Xiaolong Wang. 3d-spatial multimodal memory. In The Thirteenth International Conference on Learning Representations, 2025.

[98] Xingxing Zuo, Pouya Samangouei, Yunwen Zhou, Yan Di, and Mingyang Li. Fmgs: Foundation model embedded 3d gaussian splatting for holistic 3d scene understanding. International Journal ofComputer Vision, 133(2):611–627, 2025.

This appendix provides additional implementation details (§ A), correspondence analysis (§ B), further analyses (§ C), additional related works (§ D), and limitations (§ E) that complement the main paper.

## A Additional Implementation Details

Datasets and input. We use an input resolution of 384 × 384, matching the native resolution of LLaVA-Video-7B [89]. For the ScanNet-based datasets (SQA3D [52], ScanQA [3], Scan2Cap [11], ScanRefer [10], and Multi3DRefer [88]), we uniformly sample 32 frames per scene, following prior work [96, 93, 73, 23, 30]. For SPAR [87], we use the 104K-sample multi-view subset released by VG-LLM [92], in which each scene contains either 2, 3, or 32 views, and we feed all available views per scene to the model. We mix SPAR-7M into the ScanNet-based training set to mitigate the language-pattern overfitting tendency previously observed when training on ScanNet alone [51, 92].

Compact Gaussian teacher. We build our teacher on top of ZipSplat’s [72] open-source codebase, where we control the number of compact tokens to 2592 to make an adequate bottleneck structure as mentioned in § 4.2. The teacher can be readily swapped with other token-level feed-forward 3DGS frameworks [1, 35, 61], and is kept frozen and used only at training time.

Training and architecture details. We train with the AdamW optimizer using a global batch size of 64 with 4 gradient accumulation steps. The peak learning rate after warmup is 1e−5 for the LLM and 2e−6 for the vision encoder, and we clip the gradient norm to 1.0. For our full training objective, we set $\lambda _ { \mathrm { f e c o n } } = 3 . 0$ and $\lambda _ { \mathrm { d i s t i l l } } = 1 . 0$ , with $\lambda _ { \mathrm { M S E } } = \lambda _ { \mathrm { p a r a m } } = \lambda _ { \mathrm { t o k e n } } = 1 . 0$ and $\lambda _ { \mathrm { L P I P S } } = 0 . 0 5$ . The teacher-alignment projector g(·) in §3.2.2 is a two-layer MLP with a GeLU activation [27] that maps the LLM’s hidden dimension 3584 → 1536 to match the teacher’s query dimension $d _ { T }$ . The Gaussian decoder head H(·) is a three-layer MLP with SiLU activations [21] mapping 1536 → 736, which is the dimension required to specify the parameters of $G = 3 2$ Gaussians per summary token. We use spherical harmonics of degree 2 for view-dependent color.

The 3D LMMs we compare against in Tab. 1, Tab. 2, and Tab. 3 differ in how they encode the 3D scene: Scene-LLM [25] and LL3DA [13] consume point-cloud features; LEO [32], ChatScene [31], and Grounded 3D-LLM [15] use object-centric representations; LLaVA-3D [96] lifts 2D features into a 3D voxel grid; and Video-3D-LLM [93], GPT4Scene [58], and Ross3D [73] treat multi-view images as video sequences.

![](images/986b1b63176eb3c8b435a7f532c7e04bdcfb940adbb0e2db9fccd1988f765551.jpg)  
Figure 4: Layer-wise cross-view correspondence during training. We report cross-view correspondence scores for image-token representations extracted from middle LLM layers $\ell = 1 1 \sim 1 8$ on SQA3D [52] test samples. Scores increase consistently throughout training, with pronounced gains near the Gaussian decoding layer (ℓ = 14), indicating that joint reconstruction training improves the view-consistency of intermediate image features. Shaded regions indicate variance across SQA3D [52] test samples.

## B Additional Correspondence Analysis

Following 3DRS [33], we evaluate cross-view feature correspondence as a quantitative probe of how 3D-aware the image-token representations have become. We voxelize the 3D coordinates associated with image tokens with the given 3D point clouds for each scene of ScanNet [20], construct cross-view pairs from tokens in different views that fall into the same voxel, and report the correspondence score as the average cosine similarity between the ℓ-th layer hidden states of these pairs. A higher score indicates stronger feature consistency for image tokens describing the same 3D region across viewpoints. As shown in Fig. 4, correspondence scores in the middle LLM layers, particularly the layers $\ell = 1 1 \sim$ 18 that surround the Gaussian decoding layer, improve substantially throughout training. This finding plays a dual role. As a corollary, it confirms that reconstruction supervision propagates 3D-aware structure beyond the summary tokens themselves into the image-token representations. We further attribute this propagation to our compact-bottleneck design $( M \ll K N ^ { \prime } ) \dot { : }$ with far fewer summary tokens than image tokens, each summary token is forced to aggregate the same 3D region across multiple views, which in turn pressures the upstream image features to encode that region consistently across viewpoints. More importantly, the result shows that joint reconstruction delivers the kind of cross-view correspondence that prior methods such as 3DRS supply through explicit supervision: improved correspondence emerges naturally from learning to reconstruct, without ever being directly optimized.

## C Additional Analysis

## C.1 Additional visualization of attention between images

We further visualize the attention maps across multi-view images using the same protocol as Fig. 2. As illustrated in Fig. 5, image tokens from our model reliably focus on matching regions of the same object across different views, demonstrating their robustness across diverse scenes.

## C.2 Additional visualization of clustered Gaussian

We additionally visualize the attention maps between Gaussian summary tokens and multi-view images, following the same protocol as Fig. 3. As shown in Fig. 6, common objects are clustered into similar Gaussian summary tokens without any explicit supervision.

## C.3 Additional quantitative evaluation of semantic feature structuring

The PCA visualizations in Fig. 2 and the cross-view correspondence analysis in Appendix B provide qualitative and geometric evidence that joint reconstruction training improves the LLM’s image features. To obtain a more direct quantitative measure of whether the features become more semantically structured, we conduct an unsupervised semantic segmentation probe following the standard evaluation methodology of [19, 26].

Table 9: Unsupervised semantic segmentation probe. We cluster the LLM’s image features via K-means and measure mIoU against ground-truth ScanNet labels. DINOv3 [65] is evaluated under the same protocol as an upper-bound reference.
<table><tr><td>Method</td><td>mIoU</td></tr><tr><td>Baseline (no summary tokens)</td><td>26.54</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>35.43</td></tr><tr><td>DINOv3 [65]</td><td>37.62</td></tr></table>

Table 10: Reconstruction quality on SQA3D test scenes. We compare the rendering quality of our Gaussian summary tokens against the compact Gaussian teacher [72] used during training. Higher is better for PSNR and SSIM; lower is better for LPIPS.
<table><tr><td>Method</td><td>PSNR (↑)</td><td>SSIM (↑)</td><td>LPIPS (↓)</td></tr><tr><td>Gaussian teacher [72]</td><td>20.60</td><td>0.76</td><td>0.44</td></tr><tr><td>Imagine3D-LLM (Ours)</td><td>17.61</td><td>0.74</td><td>0.49</td></tr></table>

Specifically, on the ScanNet scenes of the SQA3D evaluation set (which provide per-image 2D semantic labels), we extract the LLM’s image features at layer ℓ for all 32 input views and apply K-means clustering jointly over all views’ tokens, with k set to the number of ground-truth NYU40 classes present in the scene. Since the resulting clusters carry no labels, we follow the standard unsupervised segmentation protocol: each cluster is assigned to its maximally-overlapping ground-truth class via majority vote. After assignment, cluster labels are upsampled to the label resolution and we compute mIoU against the ground-truth semantic masks, aggregated across all views of each scene. As DINOv3 [65] features are well-known to be semantically structured, we additionally evaluate DINOv3 under the same protocol as an upper-bound reference.

As reported in Tab. 9, our model’s image features achieve 35.43 mIoU, a substantial improvement over the baseline (26.54) that closes much of the gap to DINOv3 (37.62). This result provides direct evidence that the feature space itself, not only its downstream usage, becomes more semantically structured through joint reconstruction training. Importantly, this structuring emerges as a byproduct of the reconstruction objective—the image features are never directly supervised with semantic labels—confirming that learning to reconstruct propagates object-level structure into the LLM’s visual representations.

## C.4 Additional quantitative evaluation of reconstruction quality

While our primary goal is to improve 3D reasoning rather than novel view synthesis, here we further analyze rendering quality on the SQA3D test scenes and compare against the compact Gaussian teacher [72] in Tab. 10. As expected, our reconstruction quality is somewhat below the teacher’s, since our Gaussians are decoded from the hidden states of an LLM that is jointly optimized for language modeling and reconstruction, trained with much fewer iterations, and without any special design to match the teacher’s rendering fidelity. However, the primary goal of our design is not to achieve high-fidelity rendering: the reconstruction objective serves as an inductive bias that teaches the model how the scene is structured (mental reconstruction), not as a novel view synthesis system (§ 3.2.1).

## C.5 Additional analysis between Gaussian summary token and multi-view images

To accurately reconstruct the scene, it is crucial for the Gaussian summary tokens to effectively aggregate information from multi-view images. We conduct additional analysis to examine how Gaussian summary tokens attend to multi-view image features when generating 3D Gaussians. As shown in Fig. 7, each Gaussian summary token attends to corresponding regions of the same object across multiple views in the scene. This behavior emerges because a limited number of Gaussian summary tokens are compelled to aggregate multiple occurrences of the same object into a single representation, effectively clustering common objects without any explicit supervision. This emergent object-centric property is reminiscent of human perception, where the same object observed from multiple viewpoints is interpreted as a single entity to form a coherent 3D understanding of the scene.

(a) Image-to-Image Feature Atention  
(b) Image Feature PCA Visualization  
![](images/fa9dc7481716a5255264b77863219a14ae24b85be7f77145f7b9924798216cc9.jpg)  
Figure 5: Visualization of attention maps between multi-view images and PCA visualizations of the image features. (a) Image-to-image attention maps between a query point (red dot) and other input views. Our model exhibits substantially stronger cross-view correspondences than the baseline, indicating that the Gaussian summary tokens implicitly drive the image features to align across views. (b) PCA visualization of the LLM’s image features (first three components shown as RGB). Our features are noticeably cleaner and more semantically structured, with corresponding objects (e.g., chairs, desks) encoded with consistent colors across views, reflecting the emergence of object-level abstractions.

![](images/0bdf1a5b2703dc2feb9e2dc646030d6e3520f42d1e6d59b5ffd5fc84cb9556ea.jpg)  
Figure 6: Visualization of clustered Gaussian from summary tokens. We visualize the Gaussians clustered from a set of Gaussian summary tokens via K-means over the summary tokens. Similar summary tokens are mapped to the same object without any explicit clustering supervision.

![](images/45ccf9cbd3d85a43b9423946d707d6432307b50cbc918a678b5c891e8f5f0bd2.jpg)  
Figure 7: Visualization of attention maps between Gaussian summary tokens and images. We visualize the attention score map between a Gaussian summary token (a red dot indicating the decoded Gaussian) and the input images. Each Gaussian token attends to corresponding regions of the same object across multiple views.

Table 12: Inference cost comparison. Peak GPU memory and per-sample inference time measured on the SPAR evaluation set. VLM3R-7B [23] additionally runs an external 3D foundation model (CUT3R [76]) on all input frames before fusion.
<table><tr><td>Method</td><td>Peak Memory (GB)</td><td>Inference Time (ms)</td></tr><tr><td>Baseline (7B)</td><td>17.09</td><td>112.1</td></tr><tr><td>VLM3R-7B [23]</td><td>25.29</td><td>344</td></tr><tr><td>Imagine3D-LLM (7B)</td><td>20.43</td><td>176.9</td></tr></table>

## C.6 Computation cost analysis.

We further analyze the additional training-time cost incurred by our reconstruction and distillation objectives. As reported in Tab. 11, augmenting the standard language modeling loss with $\mathcal { L } _ { \mathrm { r e c o n } }$ and $\mathcal { L } _ { \mathrm { d i s t i l l } }$ raises the peak per-GPU VRAM from 40 GB to 44 GB, a 10% increase under identical batch size and sequence length. This modest overhead is a direct consequence of our design choice to keep the number of Gaussian summary tokens compact $( \bar { M } = 2 5 9 2 , \ S 3 . 2 . 1 )$ . To make this concrete, pixelaligned feed-forward 3DGS estimators [9, 16, 28, 83] predict one Gaussian per input pixel, which for our setting of 32 images at $3 8 4 \times 3 8 4$ resolution would amount to roughly 4.7M Gaussians per scene. In contrast, Imagine3D-

<table><tr><td>Training setting</td><td>Peak VRAM (GB)</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e c o n } } + \mathcal { L } _ { \mathrm { d i s t i l l } }$ </td><td>40</td></tr><tr><td>Imagine3D-LLM (full)</td><td>44</td></tr></table>

Table 11: Training-time computation cost. Peak per-GPU VRAM measured during training under identical batch size and sequence length. The reconstruction and distillation losses introduce only a modest memory overhead while substantially improving 3D reasoning.

LLM decodes only $M \times G = 2 5 9 2 \times 3 2 \approx 8 3 \mathrm { K }$ Gaussians, less than 1.8% of the pixel-aligned count. This compactness keeps the rasterization, the teacher forward pass, and the gradient buffers comparatively lightweight, allowing the reconstruction objective to fit alongside MLLM finetuning at a manageable cost. The overhead is also confined to training, as the compact Gaussian teacher [72] is discarded at inference time and the Gaussian decoder head H(·) is only invoked when reconstructions are explicitly required. Given the substantial gains in 3D reasoning across all evaluated benchmarks (Tab. 1, 2, 3), we view this as a favorable trade-off, indicating that imagining the scene through a compact Gaussian abstraction is not only effective but also practically affordable.

Inference time cost. We further report peak GPU memory and per-sample inference time on the SPAR evaluation set for three models: (a) the baseline multi-image MLLM taking only images, (b) VLM3R-7B [23], which fuses image features with external CUT3R [76] features, and (c) our model.

As shown in Tab. 12, Imagine3D-LLM uses 20.43 GB and 176.9 ms per sample, compared to the baseline’s 17.09 GB and 112.1 ms. The additional overhead compared to the baseline comes almost entirely from the longer input sequence $( M = 2 5 9 2$ additional summary tokens), as the Gaussian decoder head is only invoked when the user explicitly requests a reconstruction and the distillation teacher is discarded after training. Importantly, approaches like VLM3R that fuse external 3D foundation model features must run a separate model (CUT3R) on all input frames at inference before fusion, resulting in substantially higher memory (25.29 GB) and latency (344 ms). Our approach requires no external 3D model at inference, keeping the cost well below that of feature-fusion alternatives while still providing substantial reasoning gains.

![](images/e6c887bd5b6d753aa0c25b23bce1d2f0d18ec7b09cb7a3ce2d227ecbc330ddc6.jpg)  
Figure 8: Visualization of reconstructed scene with estimated Gaussians. (Top): input multi-view images. (Bottom): a view rendered from the 3D Gaussians decoded by our Gaussian summary tokens. Although the rendering is coarse due to the compact M = 2592 token bottleneck, it preserves the overall scene layout and the placement of major objects, indicating that the tokens encode a coherent 3D abstraction of the scene.

## C.7 Visualization of estimated Gaussians.

While our main objective is improving 3D reasoning rather than achieving photorealistic novel view synthesis, we additionally visualize the estimated Gaussians in 3D to verify that the Gaussian summary tokens indeed acquire a coherent 3D understanding of the scene. As shown in Fig. 8, the rendered views recover the overall layout of the scene and the rough placement of major objects (e.g., the central table surface and surrounding seating arrangement), confirming that even when decoded from only M = 2592 summary tokens, the model is able to organize multi-view evidence into a spatially consistent 3D abstraction. We emphasize, however, that the rendering quality itself is not the primary concern of Imagine3D-LLM. Unlike feed-forward 3DGS estimators that are explicitly optimized for high-fidelity novel view synthesis [9, 16, 28, 83, 37], our Gaussian summary tokens are trained jointly with the language modeling objective and are bottlenecked to a small set of tokens, which inherently trades off pixel-level fidelity for compact, object-level abstraction.

## D Additional Related Works

Because our framework incorporates 3D Gaussian Splatting within an MLLM, several existing directions may appear superficially related to Imagine3D-LLM. We briefly discuss them here and clarify the key distinctions.

Feed-forward 3DGS as a feature lifter for MLLMs. SplatTalk [68] also pairs a feed-forward 3DGS estimator with an MLLM, but uses 3DGS as an externalfeature lifter: a frozen estimator (FreeSplat [77]) lifts pretrained 2D features into a 3D Gaussian field that is then fed to the LLM as visual tokens. The MLLM itself never learns to reconstruct the scene. In contrast, Imagine3D-LLM equips the MLLM with learnable Gaussian summary tokens that jointly learn to reconstruct the scene and support language modeling, with no external 3DGS module at inference.

Per-scene 3DGS-based scene understanding. LangSplat [59], FMGS [98], and M3 [97] distill 2D foundation features into 3D Gaussians optimized per scene, enabling open-vocabulary retrieval or segmentation. Imagine3D-LLM targets a different setting: a feed-forward generalist that answers natural-language questions about novel scenes in a single forward pass, without per-scene optimization.

Point-cloud-input MLLMs for 3D detection. SpatialLM [53] feeds 3D point clouds to an MLLM and outputs structured bounding-box coordinates and labels from a fixed vocabulary. Imagine3D-LLM instead targets free-form 3D understanding—spatial QA, situated reasoning, dense captioning, and grounding—through natural language interaction.

Latent geometric reasoning without explicit reconstruction. The most conceptually similar concurrent work is 3DThinker [17], which also argues for “thinking with 3D” before answering spatial questions. 3DThinker aligns latent reasoning tokens with features from a frozen 3D foundation model (VGGT [74]), placing it closer to methods that fuse geometry-foundation features into MLLMs (e.g., VLM3R [23], G<sup>2</sup>VLM [30]). Imagine3D-LLM instead requires the Gaussian summary tokens to actually reconstruct the scene through differentiable rasterization, providing a stronger inductive bias toward 3D-aware internal representations (§3.3) Empirically, Imagine3D-LLM outperforms 3DThinker-7B by 5.2 points on SPAR-Bench overall, with larger margins on the medium- and high-level reasoning splits.

## E Limitations

While Imagine3D-LLM demonstrates that abstract reconstruction is an effective inductive bias for 3D-aware MLLMs, training efficiency remains a limitation. As discussed in § 3.2.2, our framework can learn Gaussian summary tokens without a pretrained compact Gaussian teacher [72], but requires more training steps to achieve comparable downstream performance. The teacher therefore accelerates convergence rather than being a necessary component of our framework. Improving the efficiency of teacher-free training remains an important direction for future work.

Another limitation concerns the scope of our training scenes and the fixed representation budget. Our model is trained on indoor scenes, where a fixed number of Gaussian summary tokens and decoded Gaussians has not posed a major constraint in our experiments. However, larger and more complex outdoor scenes may require a greater representation capacity, and the current Gaussian budget could be insufficient to capture their spatial structure. Extending our framework to cover both indoor and outdoor environments, and dynamically adapting the number of Gaussian summary tokens to the scale and complexity of each scene, are promising directions for future work.