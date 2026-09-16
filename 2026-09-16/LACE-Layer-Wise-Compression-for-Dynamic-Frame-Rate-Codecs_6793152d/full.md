# LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs

Thanapat Trachu, Samuele Cornell, William Chen, Shinji Watanabe

Language Technologies Institute, Carnegie Mellon University, Pittsburgh, USA

{ttrachu, scornell, wc4, swatanab}@andrew.cmu.edu

Abstract—Neural audio codecs are a key component in speech language modeling. However, their high frame rates lead to long sequence lengths, increasing computational costs. Dynamic frame rate codecs mitigate this by reducing the effective frame rate using a compression step to merge multiple frames together. However, most prior methods either operate on single-codebook codecs or apply a single compression step before multi-layer quantization. This forces all quantization layers to share the same segmentation boundaries, despite the residual embeddings at different quantization layers exhibiting different rates of change over time. We propose LACE (Layer-Adaptive Codec Encoding), a dynamic frame rate codec that applies an independent compression step at each quantization layer, enabling layer-specific segmentation boundaries. To use LACE tokens in downstream text-to-speech (TTS), we further introduce union alignment and boundary anchor mechanisms to make durations consistent across layers while preserving compression benefits. Experiments on LibriTTS show that LACE offers a better rate-quality tradeoff than prior dynamic frame rate methods on the reconstruction task and improves TTS inference efficiency while maintaining competitive synthesis quality. Our code is released as part of the ESPnet3 codec recipe.

Index Terms—Audio Codec, Text-to-Speech, Dynamic Frame Rate Codec, Compression

## I. INTRODUCTION

Neural audio codecs have emerged as an important component for speech language modeling. When used as speech tokenizers, they convert continuous speech into discrete codes, allowing speech generation [1]–[3] and speech language models [4]–[6] to be formulated as next-token prediction problems similar to those used in large language models. However, general neural audio codecs [7]–[9] typically operate at high frame rates, producing long sequences of discrete codes.

Long sequences of discrete codes introduce two limitations. First, long sequences increase the computational cost, as the complexity of transformer-based models tends to scale with sequence length [10]. Second, the length mismatch between speech and text tokens can degrade the performance of speech language models, since substantially longer speech sequences make syntactic and semantic modeling more difficult, as shown in [11].

These limitations motivate dynamic frame rate codecs [12]– [15], which represent speech as discrete codes paired with durations. A compression model produces segmentation boundaries, which are frame positions where the frame-level embeddings are partitioned. These boundaries divide the embeddings into variable-length segments, and each segment is represented by one code and its duration. Since each code now corresponds to a segment rather than a single frame, the frame rate is measured as the number of segments per second instead of the number of frames per second. We refer to this as the effective frame rate. For example, if one second of audio with 75 framelevel embeddings is compressed into 25 segments, the effective frame rate is 25 Hz, instead of 75 Hz. By assigning longer durations to regions with lower information density, dynamic frame rate codecs use fewer codes to represent speech than general codecs.

Most existing dynamic frame rate codecs either operate on single-codebook codec models [13], [15], or apply a single compression step before multi-layer quantization [14], [16], [17]. In both cases, all quantization layers are constrained to share the same segmentation boundaries. This constraint may be suboptimal for multi-codebook codecs due to the hierarchical nature of their quantization process. In residual vector quantization (RVQ) [18], the first layer quantizes the input directly, while each subsequent layer quantizes the residual embeddings from the previous layer. Consequently, earlier layers capture the main signal structure, whereas deeper layers encode finer details. As a result, representations at different layers may change at different rates over time, suggesting that each layer requires its own segmentation boundaries. This intuition is supported by our analysis in Section V-C1, which shows that deeper-layer residual embeddings change more rapidly than earlier-layer residual embeddings. This is further supported by the findings of SNAC [19], which uses different frame rates across quantization layers, although it is not a dynamic frame rate codec because its frame rates are fixed.

Motivated by these findings, we propose LACE (Layer-Adaptive Codec Encoding), a dynamic frame rate codec that independently applies a compression step at each quantization layer. This allows each layer to choose its own segmentation boundaries. To adapt LACE to the common speech language model framework, we propose union alignment, which constructs shared segmentation boundaries by taking the union across layers. Since union alignment can increase the effective frame rate, we further propose boundary anchor, which constrains deeper layers to reuse boundaries introduced by earlier layers. Our contributions are summarized as follows:

• We propose LACE, a dynamic frame rate codec that applies an independent compression step at each quantization layer.

• We propose union alignment and boundary anchor to resolve duration inconsistency across layers, enabling LACE tokens to be used in downstream TTS training.

• We provide a theoretical analysis showing that LACE achieves a lower upper bound on the expected quantization error than single compression.

• Experiments on LibriTTS [20] show that LACE offers a better rate-quality tradeoff than prior dynamic frame rate methods on the reconstruction task and improves TTS inference efficiency while maintaining competitive synthesis quality.

## II. BACKGROUND

## A. Neural Audio Codec Preliminaries

A neural audio codec consists of an encoder, a quantization module, and a decoder. The encoder maps the raw waveform into frame-level embeddings $\mathbf { H } \in \mathbb { R } ^ { T \times D }$ , where $T$ is the number of frames and D is the embedding dimension. The quantization module discretizes H into frame-level codes $\dot { \mathbf { Z } } \in [ 1 , V ] ^ { T \times C }$ , where each frame is represented by C codes from a codebook of size V . The decoder then reconstructs the waveform from these codes.

A commonly used quantization method is RVQ. RVQ discretizes H using C quantization layers, where each layer l contains a codebook $\textbf { Q } ^ { l } ~ \in ~ \mathbb { R } ^ { V \times \mathbf { \bar { \Lambda } } D }$ . Here, $\mathbf { q } _ { k } ^ { l } \in \mathcal { \dot { \mathbb { R } } } ^ { D }$ denotes the k-th codeword of Q<sup>l</sup>. RVQ performs quantization sequentially across quantization layers. At layer l, the nearest codeword is selected from the corresponding codebook $\mathbf { Q } ^ { l }$ as $z _ { i } ^ { l } =$ arg mi $\mathbf { 1 } _ { k \in [ 1 , V ] } \| \mathbf { r } _ { i } ^ { l } - \mathbf { q } _ { k } ^ { l } \| _ { 2 } ^ { 2 }$ , where $\bar { z _ { i } ^ { l } } \in [ 1 , V ]$ and $\mathbf { r } _ { i } ^ { l } \in \mathbb { R } ^ { D }$ denote a code and a residual embedding at frame i and quantization layer l, respectively. The residual embedding is updated recursively as $\bar { \mathbf { r } } _ { i } ^ { l + 1 } = \mathbf { \bar { r } } _ { i } ^ { l } - \mathbf { q } _ { z _ { i } ^ { l } } ^ { l } .$ , with $\mathbf r _ { i } ^ { 1 } = \mathbf h _ { i }$ where $\mathbf { h } _ { i } \in \mathbb { R } ^ { D }$ is the i-th frame-level embedding in H. We define the stacked frame-level residual embeddings of layer l as $\mathbf { R } ^ { l } \in \mathbb { R } ^ { T \times D }$

## B. Compression in Dynamic Frame-Rate Codecs

The encoder of a neural audio codec produces frame-level embeddings H at a fixed codec frame rate, yielding T frames. A compression model groups these T frames into M variablelength segments. Each segment is represented by C codes and an integer duration.

Formally, the compression model determines segmentation boundaries $B = \{ b _ { 0 } , b _ { 1 } , \ldots , b _ { M } \}$ , a sorted set with $b _ { 0 } = 0$ $b _ { M } = T$ , and $b _ { m } \in [ 1 , T - 1 ]$ The m-th segment spans the frame interval $\left( b _ { m - 1 } , b _ { m } \right]$ . Some methods control M explicitly using a target compression rate $\gamma \in ( 0 , 1 ]$ , yielding $M = \lfloor \gamma T \rfloor$ while others determine M implicitly through a threshold τ. The frame-level embeddings within each segment are averaged to produce segment-level embeddings $\bar { \mathbf { H } } \in \mathbb { R } ^ { M \times D }$ . The segmentlevel embeddings are paired with the durations d $\in \mathbb { N } ^ { M }$ , where the duration of the m-th segment $d _ { m }$ is $b _ { m } \mathrm { ~ - ~ } b _ { m - 1 }$ . The segment-level embeddings H<sup>¯</sup> are quantized using RVQ, as described in Section II-A, producing segment-level codes $\bar { \mathbf { Z } } \in$ $[ 1 , V ] ^ { M \times C }$ . This (code, duration) format resembles the classical run-length encoding scheme [21]. During reconstruction, Z<sup>¯</sup> is repeated according to d to produce frame-level codes $\hat { \mathbf { Z } } \in$ $[ 1 , V ] ^ { T \times C }$ . This restores the frame rate expected by the decoder. The three compression models below share this pipeline and differ only in how they derive B.

a) Dynamic Programming (DP)-Based Compression.: CodecSlime [15] determines B by minimizing the total reconstruction error between the original frame-level embeddings H and the segment-level embeddings H<sup>¯</sup> . For a segment of s consecutive frames ending at frame $j ,$ the segment-level embedding is defined as the mean of the frames within the segment: $\begin{array} { r } { \bar { \mathbf { h } } _ { j , s } ~ = ~ \frac { 1 } { s } \sum _ { t = j - s + 1 } ^ { j } \mathbf { h } _ { t } } \end{array}$ . The reconstruction loss for this segment is $\begin{array} { r } { \ell ( j , s \bar { \bf { \sigma } } ) = \sum _ { t = j - s + 1 } ^ { j } \| { \bf h } _ { t } - \bar { \bf h } _ { j , s } \| _ { 2 } ^ { 2 } } \end{array}$ . Each segment is constrained to contain at most U frames. Let $f [ j , n ]$ denote the minimum total reconstruction loss when the first j frames are compressed into n segments. The DP objective is defined recursively as:

$$
f [ j , n ] = \operatorname* { m i n } _ { 1 \leq s \leq U } \left\{ f [ j - s , n - 1 ] + \ell ( j , s ) \right\} ,\tag{1}
$$

with $f [ 0 , 0 ] = 0 .$ . At each state $( j , n )$ , we record the segment length $s ^ { * } [ j , n ]$ that achieves the minimum in Equation (1). Given the number of segments $M ,$ we start from the last boundary $b _ { M } \ = \ T$ and recover each preceding boundary as $b _ { n - 1 } = b _ { n } - s ^ { * } [ b _ { n } , n ]$ for $n = M , M { - } 1 , \ldots , 1$ , which terminates at $b _ { 0 } = 0$ and yields the segmentation boundaries B.

b) Cosine Similarity-Based Compression.: FlexiCodec [14] determines B by thresholding the similarity between adjacent frame-level embeddings. Let $\sigma _ { t } = \cos ( \mathbf h _ { t } , \mathbf h _ { t + 1 } )$ denote the similarity between frames t and $t + 1 . \mathrm { ~ A ~ }$ boundary is placed wherever the similarity drops below a threshold τ , yielding $ { \mathcal { B } } = \{ 0 , T \} \cup \{ t : \sigma _ { t } < \tau \}$ . The number of segments M is therefore not fixed in advance but controlled implicitly by the threshold τ, unlike the DP-based compression model.

VARSTok [13] derives B using density peak clustering. For each frame t, a local density $\rho _ { t }$ is computed from its κ nearest neighbors, and a peak distance $\delta _ { t }$ is the temporal distance to the nearest frame with higher density:

$$
\rho _ { t } = \frac { 1 } { \kappa } \sum _ { j \in \mathrm { K N N } ( t ) } \phi ( \mathbf { h } _ { t } , \mathbf { h } _ { j } ) , \qquad \delta _ { t } = \operatorname* { m i n } _ { j : \rho _ { j } > \rho _ { t } } | j - t | ,\tag{2}
$$

where $\phi ( \mathbf { h } _ { i } , \mathbf { h } _ { j } ) = ( 1 + \langle \mathbf { h } _ { i } , \mathbf { h } _ { j } \rangle ) / 2$ and KNN(t) denotes the set of κ nearest neighbors of frame t. Frames with a high peak score $p _ { t } = \rho _ { t } \delta _ { t }$ are selected as segment centers. Each center is greedily expanded to adjacent unassigned frames that satisfy $\phi ( \mathbf { h } _ { i ^ { * } } , \mathbf { h } _ { t } ) - \beta p _ { t } > \tau$ , up to a maximum of U frames. This process repeats until all frames are assigned to segments. The segment endpoints are then sorted to form B. The number of segments M is controlled implicitly by τ . We use $\kappa = 5$ $\beta = 0 . 2$ , and $U = 4$ , following the original paper.

All of these prior works force all quantization layers to share the same segmentation boundaries. In contrast, LACE applies an independent compression step at each quantization layer, enabling layer-specific segmentation boundaries. Because LACE is agnostic to the choice of compression model, any of these methods can be used as the compression step within the

![](images/dd5136bf35cedc488f74c901bedc3a1187c13fdf387722c01038317523ae4436.jpg)  
Fig. 1. Overview of the LACE codec. At each quantization layer, the residual embeddings are independently compressed before being passed to the quantization layer. Frames with the same color belong to the same segment.

proposed framework.

## III. METHOD

We propose LACE, a dynamic frame rate codec that applies an independent compression step at each quantization layer, allowing each layer to choose its own segmentation boundaries. We then introduce two mechanisms, union alignment and boundary anchor, to make the duration consistent for downstream TTS training.

## A. Layer-Wise Compression

Unlike prior dynamic frame rate codecs that apply a single compression step and share segmentation boundaries across quantization layers, LACE performs layer-wise compression on the residual embeddings. At layer $l ,$ we apply the compression step described in Section II-B to $\mathbf { R } ^ { l }$ . This produces segmentlevel residual embeddings $\bar { \mathbf { R } } ^ { l } \in \mathbb { R } ^ { M ^ { l } \times D }$ , durations $\mathbf { d } ^ { l } \in \mathbb { N } ^ { M ^ { l } }$ and layer-specific segmentation boundaries $B ^ { l }$ , where $M ^ { l }$ is the number of segments at layer l. The segment-level residual embeddings $\bar { \mathbf { R } } ^ { l }$ are then passed to quantization layer l, yielding segment-level codes $\bar { \mathbf { Z } } ^ { l } \in [ 1 , V ] ^ { M ^ { l } }$

To compute the residual embeddings for the next layer, we expand $\bar { \mathbf { Z } } ^ { \bar { l } }$ to frame-level codes $\hat { \mathbf { Z } } ^ { l } \in [ 1 , V ] ^ { T }$ by repeating each code according to $\mathbf { d } ^ { l }$ . The next residual embedding is then computed as $\mathbf { r } _ { i } ^ { l + 1 } = \mathbf { r } _ { i } ^ { l } - \mathbf { q } _ { \hat { z } _ { \hat { z } } ^ { l } } ^ { l }$ . Unlike the standard RVQ update in Section II-A, $\mathbf { q } _ { \hat { z } _ { i } ^ { l } } ^ { l }$ is obtained by quantizing a segmentlevel residual embedding, whereas $\mathbf { q } _ { z _ { i } ^ { l } } ^ { l }$ in the original update is obtained from a frame-level residual embedding. We use the frame-level residual embedding $\mathbf { r } _ { i } ^ { l }$ on the right-hand side so that $\mathbf { r } _ { i } ^ { l + 1 }$ captures both the compression error and the quantization error from layer l. Figure 1 illustrates layer-wise compression.

## B. Union Alignment

The layer-wise compression produces different segmentation boundaries $B ^ { l }$ for each quantization layer l. As a result, codes from different layers that cover the same time span can have different durations. This is problematic for TTS because the model would need to predict durations for each layer while ensuring that the total duration after upsampling is identical across layers. We address this with union alignment, which constructs shared segmentation boundaries by taking the union across layers: $\begin{array} { r } { B ^ { \mathrm { u n i o n } } \ = \ \bigcup _ { l = 1 } ^ { C } B ^ { l } } \end{array}$ . We then re-segment the segment-level codes of every layer using $B ^ { \mathrm { u n i o n } }$ . As shown in Figure 2, when a boundary in $B ^ { \mathrm { u n i o n } }$ falls inside an existing segment, that segment is split into sub-segments. For example, the boundary $b _ { 1 } = 2$ from the first layer splits the first segment in the second layer. The resulting sub-segments keep the same code and receive new corresponding durations. After alignment, all layers share the same segmentation boundaries and durations, at the cost of increasing the effective frame rate.

![](images/1bd204c4513317da5a49af70f5fb047f8e409d59080826a1f9a14c3d211286a9.jpg)  
Fig. 2. An overview of union alignment. Before alignment (top), quantization layers 1 and 2 have different segmentation boundaries, resulting in inconsistent durations across layers. After alignment (bottom), the union of segmentation boundaries from both layers is applied to all layers.

## C. Boundary Anchor

Union alignment can increase the effective frame rate because $\lvert B ^ { \mathrm { u n i o n } } \rvert \geq \lvert B ^ { l } \rvert$ for all l. In the worst case, each layer introduces distinct boundaries and $| B ^ { \mathrm { u n i o n } } |$ approaches T, negating the compression benefit. To mitigate this, we introduce a boundary anchor. We choose an anchor layer l<sup>∗</sup>, where layers ${ \mathit { l } } \leq { \mathit { l } } ^ { * }$ are allowed to introduce new boundaries, while deeper layers $l > l ^ { * }$ must reuse boundaries from earlier layers: $B ^ { \mathrm { { \bar { a n c h o r } } } } = \bigcup _ { l = 1 } ^ { l ^ { * } } B ^ { l }$ By restricting the compression step at layers $l > l ^ { * }$ to place boundaries only at positions in $B ^ { \mathrm { a n c h o r } }$ , we limit the growth of $| B ^ { \mathrm { u n i o n } } |$ . Here, $l ^ { * }$ is a tunable hyperparameter that trades off between quality and the effective frame rate.

For DP-based compression, we implement this restriction with a constrained cost $\tilde { \ell } ( j , s )$ , which equals the segment loss $\ell ( j , s )$ of Section II-B if both j and $j - s$ lie in $B ^ { \mathrm { a n c h o r } }$ and $+ \infty$ otherwise. We solve the same recursion as Equation (1) with ℓ replaced by $\tilde { \ell } ,$ and use the unconstrained DP for layers ${ \mathit { l } } \leq { \mathit { l } } ^ { * }$ While this formulation is specific to DP-based compression, the boundary anchor itself is model-agnostic. It can be applied to other compression models in Section II-B by restricting its candidate boundaries to $B ^ { \mathrm { a n c h o r } }$

## D. Theoretical Analysis

Section III-C introduced the anchor layer l<sup>∗</sup> as a hyperparameter that trades off quality against the effective frame rate. We now analyze the quality side of this tradeoff: how the choice of l<sup>∗</sup> affects the expected quantization error. In Sections II-B and III-A, the compression step merges the frames before the quantization layer, and the codes are repeated back to the frame level after it. Since all repeated frames within a segment are identical, repeating before or after quantization yields the same result. We therefore combine merging and repetition into a single matrix multiplication $\mathbf { A } ^ { l } \mathbf { R } ^ { l }$ , where R<sup>l</sup> is the framelevel residual embedding at layer l. Concretely, $\mathbf { A } ^ { l } \in \mathbb { R } ^ { T \times T }$ is block-diagonal with one block per segment, and the block for a segment of length $d _ { m }$ is the $d _ { m } \times d _ { m }$ matrix with every entry equal to $1 / d _ { m }$ . Since $\mathbf { A } ^ { l }$ is symmetric and idempotent, the compression error $\mathbf { R } ^ { l } - \mathbf { A } ^ { l } \mathbf { R } ^ { l }$ is orthogonal to any framelevel embeddings that use the same segmentation boundaries as layer l. Let $\bar { \mathbf { Q } } ^ { l } \in \mathbb { R } ^ { T \times D }$ denote the frame-level quantized embeddings, which stack the selected codewords $\mathbf { q } _ { \hat { z } ^ { l } } ^ { l }$ , so the residual update of Section III-A becomes $\mathbf { R } ^ { l + 1 } = \mathbf { R } ^ { \dot { l } } - \hat { \mathbf { Q } } ^ { l }$

The quantized embeddings $\hat { \mathbf { Q } } ^ { l }$ repeat one codeword across each segment. Subtracting them in the residual update can therefore only change the part of $\mathbf { R } ^ { l }$ that is constant within each segment. The compression error, which varies within segments, passes to the next layer untouched. When all layers share segmentation boundaries, the quantized embeddings of every layer are constant within the same segments, so the compression error survives all C quantization layers. It becomes an error that quantization cannot remove. We formalize this intuition under three assumptions: (i) every layer merges at least two frames $( M ^ { l } \ < \ T )$ , so the compression error is strictly positive, (ii) each quantization layer is well-trained, approximately satisfying the centroid condition [22], so that $\mathrm { \bar { E } } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } - \tilde { \mathbf { Q } } ^ { l } \| _ { F } ^ { 2 } ] \leq \epsilon _ { l } \mathrm { \bar { E } } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ]$ with $\epsilon _ { \mathrm { m a x } } = \mathrm { s u p } _ { l } \epsilon _ { l } < 1$ and (iii) the fraction of expected residual energy captured by the compression step, $\alpha _ { l } = \mathbf { \bar { \mathbb { E } } } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] / \mathbb { E } [ \| \mathbf { R } ^ { \bar { l } } \| _ { F } ^ { 2 } ]$ , is bounded below by $\alpha _ { \mathrm { m i n } } = \mathrm { i n f } _ { l } \alpha _ { l } > 0$

Under these assumptions, the expected residual error after all C quantization layers with anchor layer $l ^ { * }$ satisfies<sup>1</sup>

$$
\mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] \le \Big [ 1 - \alpha _ { l ^ { * } } \big ( 1 - \epsilon _ { \operatorname* { m a x } } ^ { C - l ^ { * } + 1 } \big ) \Big ] \prod _ { l = 1 } ^ { l ^ { * } - 1 } \lambda _ { l } \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] ,\tag{3}
$$

where $\mathbf { R } ^ { C + 1 }$ is the residual after the last layer and $\lambda _ { l } \ =$ $1 - \alpha _ { l } ( 1 - \epsilon _ { l } ) < 1 . \mathrm { A s } \ : C  \infty$ , Equation (3) reveals a hierarchy. A single compression, which shares the same segmentation boundaries across layers $( l ^ { * } { = } 1 )$ , converges to a constant error floor $( 1 - \alpha _ { 1 } ) \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] .$ which is exactly the compression error of the first layer. The boundary anchor reduces this floor by the factor $\Pi _ { l = 1 } ^ { l ^ { \ddagger } - 1 } \lambda _ { l } < 1$ . Full layer-wise compression $( l ^ { * } { = } C )$ drives the bound to zero. Therefore, recomputing boundaries at each quantization layer removes the error floor inherent to shared segmentation boundaries. The full derivation is provided in the supplementary material. This bound concerns the codec quantization error only, not downstream TTS, where union alignment (Section III-B) also changes the token sequence.

## E. TTS Training with LACE Tokens

Unlike standard TTS models [2], [23], a TTS model trained on LACE tokens must also predict the durations d. After union alignment, all C quantization layers share the segmentation boundaries B<sup>union</sup> and durations d. Therefore, the model predicts durations only for the first quantization layer and applies the predicted durations to all subsequent layers.

We use an autoregressive decoder-only Transformer that predicts the segment-level codes Z<sup>¯</sup> in a delay-pattern format [23]. At each generation step for the first-layer code, the model additionally predicts a duration for that code. Duration prediction is formulated as a classification task. We add a duration head alongside the code prediction head and include a learned duration embedding in the input representation, which is added to the code embedding. The model is trained with cross-entropy loss for code prediction and focal loss for duration prediction to address class imbalance, with loss weights of 1 and 3, respectively. At inference, we use nucleus sampling with a top-p value of 0.8 and a temperature of 1.0.

## IV. EXPERIMENTAL SETTING

## A. Dataset

We use the LibriTTS dataset [20] at 24 kHz sampling rate. The training set consists of train-clean-100, train-clean-360, and train-other-500. Evaluation is performed on test-clean. The same dataset is used for both reconstruction and TTS experiments. For TTS, a reference utterance is randomly selected from a different utterance of the same speaker to condition the model on speaker identity.

## B. Model Architecture

1) Codec Models: We build our method on three neural audio codecs: SoundStream [8], EnCodec [9], and the Descript Audio Codec (DAC) [7], all pretrained on LibriTTS using ESPnet-Codec [24] recipes<sup>2</sup>. We integrate the proposed layerwise compression into the quantization layers (Section III-A) and fine-tune each model end-to-end from the pretrained weights for 80k iterations using the original codec objective. We use Adam [25] with learning rate $1 0 ^ { - 4 }$ , an exponential decay schedule (rate 0.9998 per step, minimum $1 0 ^ { - 5 } )$ , and shared optimization settings for the generator and discriminator. All experiments use 4 V100 32GB GPUs.

2) TTS Model: The TTS model is a 24-layer Transformer with 16 attention heads, model dimension 1024, feedforward dimension 4096, and dropout 0.1. Input transcripts are converted to phoneme sequences using the espeak-ng phonemizer [26]. A randomly truncated 3-second reference utterance is prepended to condition the model on speaker identity. The model is trained for 40k iterations on 4 V100 32GB GPUs using Adam with learning rate $1 0 ^ { - 4 }$ and 4000 warmup steps.

## C. Baselines

We compare LACE with CodecSlime [15], FlexiCodec [14], and VARSTok [13]. Because these methods use different codec backbones and training setups, we re-implement them on LibriTTS with the same codec backbones and fine-tuning configuration, changing only the compression model. These baselines use single compression, where the compression step is applied once before the quantization layers so all layers share the same segmentation boundaries. In contrast, LACE applies compression independently at each quantization layer.

TABLE I  
RECONSTRUCTION PERFORMANCE ACROSS DIFFERENT CODECS AND COMPRESSION MODELS ON LIBRITTS TEST-CLEAN. ALL BASELINE RESULTS USE A SINGLE COMPRESSION STEP WITHOUT LACE. “+ LACE” DENOTES OUR PROPOSED LAYER-WISE COMPRESSION. THE FRAME RATE COLUMN REPORTS THE EFFECTIVE FRAME RATE AND CODEC FRAME RATE FOR CODECS WITH AND WITHOUT COMPRESSION MODELS, RESPECTIVELY. FOR THRESHOLD-BASED COMPRESSION MODELS, WE REPORT THE AVERAGE EFFECTIVE FRAME RATE ACROSS LAYERS. BASELINE ROWS ARE RE-IMPLEMENTED USING THE SAME BACKBONES. THE RESULTS REPORTED BY THE CITED SYSTEMS ARE NOT DIRECTLY COMPARABLE.
<table><tr><td>Codec</td><td>Compression</td><td>Frame Rate (Hz)</td><td>Bitrate (kbps)</td><td>WER↓ (%)</td><td>UTMOS ↑</td><td>PESQ ↑</td><td>STOI ↑</td><td>SpkSim ↑</td></tr><tr><td>Ground Truth</td><td>一</td><td></td><td></td><td>2.02</td><td>4.06</td><td>=</td><td>-</td><td>-</td></tr><tr><td>DAC [7] Encodec [9]</td><td>一</td><td>75 75</td><td>24.00 24.00</td><td>2.16 2.09</td><td>3.95 3.93</td><td>3.59 3.25</td><td>0.97 0.96</td><td>0.99 0.98</td></tr><tr><td>SoundStream [8]</td><td>一 一</td><td>75</td><td>24.00</td><td>2.41</td><td>3.61</td><td>2.56</td><td>0.93</td><td>0.97</td></tr><tr><td rowspan="5">Finetuned DAC</td><td>DP [15] DP + LACE</td><td>27 22.5</td><td>8.69 8.64</td><td>4.92 2.22</td><td>2.49 3.81</td><td>1.61 3.08</td><td>0.86 0.95</td><td>0.94 0.98</td></tr><tr><td>Cosine [14]</td><td>27</td><td>8.69</td><td>6.90</td><td>1.84</td><td>1.29</td><td>0.81</td><td>0.90</td></tr><tr><td>Cosine + LACE</td><td>22.5</td><td>8.64</td><td>2.57</td><td>3.55</td><td>2.73</td><td>0.94</td><td>0.97</td></tr><tr><td>Density Clustering [13]</td><td>27</td><td>8.69</td><td>4.25</td><td>2.19</td><td>1.45</td><td>0.85</td><td>0.94</td></tr><tr><td>Density Clustering + LACE</td><td>22.5</td><td>8.64</td><td>2.62</td><td>2.89</td><td>2.08</td><td>0.95</td><td>0.96</td></tr><tr><td>Finetuned SoundStream</td><td>DP</td><td>27</td><td>8.69</td><td>7.37</td><td>2.54</td><td>1.52</td><td>0.83</td><td>0.92</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Finetuned Encodec</td><td></td><td></td><td>8.69</td><td></td><td>3.08</td><td>1.99</td><td>0.89</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathrm { D P + L A C E }$ </td><td>22.5</td><td>8.64</td><td>2.78</td><td>3.60</td><td>2.45</td><td>0.92</td><td>0.97</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>27</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>3.10</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>DP</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.96</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $\mathrm { D P + L A C E }$ </td><td></td><td></td><td>2.18</td><td>3.83</td><td>2.86</td><td>0.95</td><td>0.98</td></tr><tr><td></td><td></td><td>22.5</td><td>8.64</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td></table>

## D. Rate Computation

For reconstruction, we report the effective frame rate before union alignment, since alignment is not required. For thresholdbased compression models that yield different effective frame rates across layers, we report the mean effective frame rate across layers. The bitrate accounts for both code and duration bits, where each duration costs $\log _ { 2 } U$ bits per segment. For a fair comparison, we compare at matched bitrate (∼ 8.6 kbps). Note that because LACE adds one duration sequence per layer, it reaches this bitrate at a lower effective frame rate than single compression (22.5 vs. 27 Hz).

## E. Evaluation Metrics

a) Reconstruction.: We compute word error rate (WER) with Whisper-Large [27] by transcribing the reconstructed audio. UTMOS [28] assesses perceptual naturalness, while PESQ [29] and STOI [30] measure signal quality and intelligibility. For speaker similarity (SpkSim), we compute the cosine similarity between X-vectors from the generated and reference audio, extracted with WavLM-base [31] fine-tuned for speaker verification<sup>3</sup>. All metrics use VERSA [32] to compute.

b) TTS.: Since TTS generation is non-deterministic, we generate 10 utterances per input and select the one with the lowest WER under Whisper-Small, following prior work [33]. The selected utterance is evaluated with Whisper-Large for WER, along with UTMOS and SpkSim. We also report the real-time factor (RTF), the ratio of generation time to audio duration, to assess inference efficiency. To complement these objective metrics, we conduct a human evaluation using mean opinion score (MOS) and speaker similarity mean opinion score (SMOS) to evaluate naturalness and speaker similarity, respectively. Specifically, we randomly sample 25 test utterances and synthesize them with each TTS system. Then, we collect ratings from 35 evaluators.

TABLE II  
TTS EVALUATION ON LIBRITTS TEST-CLEAN. RATE IS THE EFFECTIVE FRAME RATE AFTER UNION ALIGNMENT. WE REPORT MOS AND SMOS, WITH 95% CONFIDENCE INTERVALS. RTF IS MEASURED ON A SINGLE V100 GPU FOR ONE CANDIDATE.
<table><tr><td>Method</td><td>γ</td><td></td><td>(Hz)</td><td>(%)</td><td>l* Rate|WER ↓ UTMOS ↑ SpkSim ↑|</td><td></td><td>MOS ↑</td><td>SMOS ↑</td><td>RTF ↓</td></tr><tr><td>No compression|</td><td>一</td><td>-</td><td>75.0|</td><td>2.31</td><td>4.18</td><td>0.91</td><td> $| 3 . 9 4 \pm 0 . 0 5 ~ 3 . 9 4 \pm 0 . 0 6 |$ </td><td></td><td>1.95</td></tr><tr><td>Single</td><td>|0.5</td><td>一</td><td>37.5</td><td>30.73</td><td>1.79</td><td>0.84</td><td> $| 3 . 8 0 \pm 0 . 0 6 ~ 3 . 8 0 \pm 0 . 0 7 |$ </td><td></td><td>1.01</td></tr><tr><td>LACE</td><td>0.5</td><td>2</td><td>52.5</td><td>4.93</td><td>2.69</td><td>0.86</td><td> $\mathbf { \left| 3 . 8 8 \pm 0 . 0 6 ~ 3 . 8 6 \pm 0 . 0 7 \right| }$ </td><td></td><td>1.58</td></tr><tr><td>LACE</td><td></td><td>0.53</td><td>58.0</td><td>8.52</td><td>2.70</td><td>0.88</td><td> $\mathbf { \left| 3 . 8 9 \pm 0 . 0 6 \ 3 . 8 4 \pm 0 . 0 6 \right|} $ </td><td></td><td>1.68</td></tr><tr><td>Single</td><td>|0.7</td><td>=</td><td>52.5</td><td>6.77</td><td>2.59</td><td>0.85</td><td> $| 3 . 8 5 \pm 0 . 0 6 ~ 3 . 8 2 \pm 0 . 0 7 |$ </td><td></td><td>1.45</td></tr><tr><td>LACE</td><td></td><td></td><td>|0.7 2 61.8</td><td>5.11</td><td>3.45</td><td>0.90</td><td> $\mathbf { \left| 3 . 9 6 \pm 0 . 0 5 ~ 3 . 9 0 \pm 0 . 0 6 \right| }$ </td><td></td><td>1.73</td></tr></table>

## V. RESULTS

## A. Reconstruction Task

We evaluate reconstruction quality across three codec backbones (DAC [7], SoundStream [8], EnCodec [9]) and three compression models (DP [15], Cosine Similarity [14], and Density Clustering [13]). All systems are matched at a similar bitrate (∼ 8.6 kbps), resulting in different effective frame rates. Table I shows that LACE surpasses the single-compression baselines in every configuration, indicating that its benefit is agnostic to both the compression model and the codec backbone. Figure 3 provides a finer-grained view through the rate-quality curve. LACE outperforms single compression at every bitrates, even on the pretrained model without fine-tuning. This indicates that the improvement comes from LACE itself, while fine-tuning provides further gains.

## B. Text-to-Speech Task

We train and evaluate the TTS model using DAC as the codec backbone with DP compression. We compare against two baselines: DAC without compression and DAC with single DP compression. Table II shows that, at the same target compression rate γ, LACE outperforms single compression on all quality metrics. This improvement comes at the cost of a higher RTF because union alignment increases the effective frame rate. Although the MOS scores of single compression with $\gamma = 0 . 5$ and no compression are close, the paired t-test shows that raters can still distinguish between them $( p < 0 . 0 1 )$ Compared to the no-compression baseline, LACE achieves a lower RTF by reducing the effective frame rate. This comes at a quality cost, partly because the compressed codec has lower reconstruction quality (Table I). The target compression rate $\gamma$ directly controls the quality–efficiency tradeoff. Increasing γ improves synthesis quality while increasing RTF. In contrast, increasing the anchor layer $l ^ { * }$ beyond 2 does not improve quality and degrades WER. We hypothesize that this behavior arises from the union alignment step. A larger $l ^ { * }$ causes long segments to be split into shorter ones, introducing repeated discrete code IDs after alignment. These repeated code IDs bias the TTS model toward repeatedly predicting the same code IDs during generation. Incorporating repetition-aware sampling [2] to prevent repetitive code prediction is a promising direction for future work. To ensure that the observed WER trends are robust to the reranking, we computed the mean WER over 10 generated candidates using Whisper-Small. The ranking remains consistent. LACE achieves 14% WER compared with 61% for single compression at $\gamma = 0 . 5$ , and 14% compared with 21% at $\gamma = 0 . 7$

![](images/409f8e569bdc229316eb7b7f88f24bb59483aa2fe46ebeb5a33f23c74aec6df4.jpg)  
Bitrate (kbps) ( )

![](images/9177501de7dc46004a54f6e82b38626ac35072d66192c66ccb977ca9e8feaf81.jpg)  
Bitrate (kbps) ( )

Fig. 3. Rate-distortion comparison between LACE and Single compression on LibriTTS test-clean using DAC as the codec backbone and DP compression.  
![](images/89306b53baccb94aa5a963258eda145c45ed703e49f511cbd60a687807418040.jpg)  
Fig. 4. Distribution of cosine similarity between consecutive frame-level residual embeddings across quantization layers.

## C. Analysis

We perform two analyses to better understand LACE. The first examines how fast the residual embeddings change at each quantization layer. The second empirically validates the quantization error bound from Section III-D.

1) Histogram ofCosine Similarity: On the pretrained codecs, we compute the cosine similarity between consecutive framelevel residual embeddings, $\sigma _ { t } ^ { l } = \mathrm { \dot { c } o s } ( \mathbf { r } _ { t } ^ { l } , \mathbf { r } _ { t + 1 } ^ { l } )$ . Figure 4 shows that its distribution concentrates around 1 at earlier layers and gradually shifts toward 0 at deeper layers. This indicates that residual embeddings change more rapidly at deeper layers and confirms that different layers change at different rates, motivating layer-wise compression. Moreover, the slower variation of earlier layers suggests that segmentation decisions have a larger impact on these layers than on deeper layers. We therefore let earlier layers choose its own segmentation boundaries before deeper layers, justifying the boundary anchor (Section III-C).

![](images/97a4fabfdc03ea2c6675e814a797696682cfa9bf069aef7ad4cc0acb8517b9af.jpg)

![](images/711fe4e1aaeda31a69fe776c01ec14cca9beb54e894dedc921a3ca5d10f41ab6.jpg)  
Fig. 5. Empirical validation of the theoretical analysis using the pretrained DAC model with DP compression. Left: quantization error at various compression rates γ using all 32 quantization layers. Right: quantization error as a function of the number of quantization layers, at $\scriptstyle \gamma = 0 . { \bar { 5 } }$

2) Empirical Result of VQ Error: While Figure 4 only motivates layer-wise compression, this analysis directly measures the error caused by shared segmentation boundaries. We measure the quantization error $\mathbb { E } [ \| \bar { \mathbf { R } } ^ { C + 1 } \| _ { F } ^ { 2 } ]$ on the pretrained DAC model with DP compression while varying the anchor layer l<sup>∗</sup>. Figure 5 shows that LACE yields consistently lower quantization error than single compression, matching Equation (3). The right panel shows the gap widening as more layers are used. Single compression converges to a constant error floor, while LACE keeps reducing the error. The left panel shows the role of $\gamma . \operatorname { A s } \gamma \to 1$ , compression merges almost no frames, so $\alpha _ { 1 }  1$ and the floor $\left( 1 - \alpha _ { 1 } \right) \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ]$ vanishes, explaining the narrow gap at high compression rates. At lower rates the floor grows and the benefit of LACE becomes more pronounced.

## VI. CONCLUSION

We presented LACE, a dynamic frame rate codec that applies an independent compression step at each quantization layer. Unlike prior methods that force all layers to share the same segmentation boundaries, LACE allows each layer to choose its own segmentation boundaries. To enable downstream TTS training, we introduced union alignment, which constructs shared segmentation boundaries across layers, and boundary anchor, which limits the growth of the effective frame rate. Experiments on LibriTTS show that LACE consistently outperforms single-compression baselines across multiple codec backbones and compression models on the reconstruction task. Applied to TTS, LACE improves inference efficiency while maintaining competitive quality.

## ACKNOWLEDGMENT

Experiments of this work used the Bridges2 system at PSC and Delta and DeltaAI system at NCSA through allocations CIS210014 and IRI120008P from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

## GENERATIVE AI USE DISCLOSURE

Generative AI tools were used only to edit and polish the manuscript language and to assist with code writing. All research ideas, experimental design, implementation, analysis, and reported results are the work of the authors. The conceptual framing, methodology, and scientific contributions are entirely human-generated.

## REFERENCES

[1] S. Chen, C. Wang, Y. Wu, Z. Zhang, L. Zhou, S. Liu, Z. Chen, Y. Liu, H. Wang, J. Li, L. He, S. Zhao, and F. Wei, “Neural codec language models are zero-shot text to speech synthesizers,” IEEE Transactions on Audio, Speech and Language Processing, vol. 33, pp. 705–718, 2025.

[2] S. Chen, S. Liu, L. Zhou, E. Liu, X. Tan, J. Li, S. Zhao, Y. Qian, and F. Wei, “VALL-E 2: Neural codec language models are human parity zero-shot text to speech synthesizers,” 2025. [Online]. Available: https://openreview.net/forum?id=0bcRCD7YUx

[3] P. Peng, P.-Y. Huang, S.-W. Li, A. Mohamed, and D. Harwath, “VoiceCraft: Zero-shot speech editing and text-to-speech in the wild,” in Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL). Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 12 442–12 462. [Online]. Available: https://aclanthology.org/2024.acl-long.673/

[4] S. Guo, S. Zhang, Q. Fang, Z. Ma, M. Zhang, and Y. Feng, “FastLongSpeech: Enhancing large speech-language models for efficient long-speech processing,” in Advances in Neural Information Processing Systems, 2026. [Online]. Available: https://openreview.net/forum?id= jaMPaFDAaZ

[5] J. Tian, J. Shi, W. Chen, S. Arora, Y. Masuyama, T. Maekaku, Y. Wu, J. Peng, S. Bharadwaj, Y. Zhao et al., “ESPnet-SpeechLM: An open speech language model toolkit,” in Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL): System Demonstrations, 2025, pp. 116–124.

[6] D. Yang, J. Tian, X. Tan, R. Huang, S. Liu, H. Guo, X. Chang, J. Shi, S. Zhao, J. Bian, Z. Zhao, X. Wu, and H. M. Meng, “UniAudio: Towards universal audio generation with large language models,” in International Conference on Machine Learning, 2024. [Online]. Available: https://openreview.net/forum?id=SRmZw7nEGW

[7] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “Highfidelity audio compression with improved RVQGAN,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 27 980–27 993.

[8] N. Zeghidour, A. Luebs, A. Omran, J. Skoglund, and M. Tagliasacchi, “SoundStream: An end-to-end neural audio codec,” IEEE/ACM Transac tions on Audio, Speech, and Language Processing, vol. 30, pp. 495–507, 2022.

[9] A. Defossez, J. Copet, G. Synnaeve, and Y. Adi, “High fidelity neural ´ audio compression,” Transactions on Machine Learning Research, 2023, featured Certification, Reproducibility Certification. [Online]. Available: https://openreview.net/forum?id=ivCd8z8zR2

[10] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, D. Roblek, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour, “AudioLM: A language modeling approach to audio generation,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2523–2533, 2023.

[11] H. Wang, H. Wang, Y. Guo, Z. Li, C. Du, and K. Yu, “Why do speech language models fail to generate semantically coherent outputs? a modality evolving perspective,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 16 117– 16 121.

[12] H. Zhang, Y. Guo, Z. Li, X. Hao, X. Chen, and K. Yu, “Unlocking Temporal Flexibility: Neural Speech Codec with Variable Frame Rate,” in Interspeech 2025, 2025, pp. 5003–5007.

[13] R.-C. Zheng, W. Liu, H.-P. Du, Q. Zhang, C. Deng, Q. Chen, W. Wang, Y. Ai, and Z.-H. Ling, “Say more with less: Variable-frame-rate speech tokenization via adaptive clustering and implicit duration coding,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 41, 2026, pp. 35 021–35 029.

[14] J. Li, Y. Qian, Y. Hu, L. Zhang, X. Wang, H. Lu, M. Thakker, J. Li, S. Zhao, and Z. Wu, “FlexiCodec: A dynamic neural audio codec for low frame rates,” in International Conference on Learning Representations, 2026. [Online]. Available: https://openreview.net/forum?id=kYkfCs4ZAH

[15] H. Wang, Y. Guo, C. Shao, B. Li, and K. Yu, “CodecSlime: Temporal redundancy compression of neural speech codec via dynamic frame rate,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 17 017–17 021.

[16] W. Zhang, Y. Qian, Y. Cao, C. He, S. Xu, and M. Wang, “CARVE: Content-adaptive rate-variable encoding for neural speech codecs,” IEEE Signal Processing Letters, vol. 33, pp. 2036–2040, 2026.

[17] Y. Qian, W. Zhang, X. Zhuang, S. Xu, L. Zhou, and M. Wang, “Arbitrarily settable frame rate neural speech codec with content adaptive variable length segmentation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2026, pp. 14 452–14 456.

[18] D. Lee, C. Kim, S. Kim, M. Cho, and W.-S. Han, “Autoregressive image generation using residual quantization,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Jun. 2022, pp. 11 523–11 532.

[19] H. Siuzdak, F. Grotschla, and L. A. Lanzend¨ orfer, “SNAC: Multi-scale¨ neural audio codec,” in Audio Imagination: NeurIPS 2024 Workshop AI-Driven Speech, Music, and Sound Generation, 2024. [Online]. Available: https://openreview.net/forum?id=PFBF5ctj4X

[20] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “LibriTTS: A corpus derived from LibriSpeech for text-to-speech,” in Interspeech 2019, 2019, pp. 1526–1530.

[21] S. Golomb, “Run-length encodings,” IEEE Transactions on Information Theory, vol. 12, no. 3, pp. 399–401, 1966.

[22] A. Gersho and R. M. Gray, Vector Quantization and Signal Compression. Kluwer Academic Publishers, 1992.

[23] J. Copet, F. Kreuk, I. Gat, T. Remez, D. Kant, G. Synnaeve, Y. Adi, and A. Defossez, “Simple and controllable music generation,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 47 704–47 720. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/94b472a1842cd7c56dcb125fb2765fbd-Paper-Conference.pdf

[24] J. Shi, J. Tian, Y. Wu, J.-W. Jung, J. Q. Yip, Y. Masuyama, W. Chen, Y. Wu, Y. Tang, M. Baali, D. Alharthi, D. Zhang, R. Deng, T. Srivastava, H. Wu, A. Liu, B. Raj, Q. Jin, R. Song, and S. Watanabe, “ESPnet-Codec: Comprehensive training and evaluation of neural codecs for audio, music, and speech,” in IEEE Spoken Language Technology Workshop (SLT), 2024, pp. 562–569.

[25] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in International Conference on Learning Representations, 2015. [Online]. Available: https://arxiv.org/abs/1412.6980

[26] M. Bernard and H. Titeux, “Phonemizer: Text to phones transcription for multiple languages in Python,” Journal of Open Source Software, vol. 6, no. 68, p. 3958, 2021. [Online]. Available: https://doi.org/10.21105/joss.03958

[27] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, Eds., vol. 202. PMLR, Jul. 2023, pp. 28 492–28 518. [Online]. Available: https://proceedings.mlr.press/v202/radford23a.html

[28] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “UTMOS: UTokyo-SaruLab system for VoiceMOS challenge 2022,” in Interspeech 2022, 2022, pp. 4521–4525.

[29] A. Rix, J. Beerends, M. Hollier, and A. Hekstra, “Perceptual evaluation of speech quality (PESQ) — a new method for speech quality assessment of telephone networks and codecs,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), vol. 2, 2001, pp. 749–752.

[30] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A shorttime objective intelligibility measure for time-frequency weighted noisy speech,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2010, pp. 4214–4217.

[31] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “WavLM: Large-scale self-supervised pretraining for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[32] J. Shi, H. jin Shim, J. Tian, S. Arora, H. Wu, D. Petermann, J. Q. Yip, Y. Zhang, Y. Tang, W. Zhang, D. S. Alharthi, Y. Huang, K. Saito, J. Han, Y. Zhao, C. Donahue, and S. Watanabe, “VERSA: A versatile evaluation toolkit for speech, audio, and music,” in Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics (NAACL): System Demonstrations, 2025. [Online]. Available: https://openreview.net/forum?id=zU0hmbnyQm

[33] P. Mousavi, G. Maimon, A. Moumen, D. Petermann, J. Shi, H. Wu, H. Yang, A. Kuznetsova, A. Ploujnikov, R. Marxer, B. Ramabhadran, B. Elizalde, L. Lugosch, J. Li, C. Subakan, P. Woodland, M. Kim, H. yi Lee, S. Watanabe, Y. Adi, and M. Ravanelli, “Discrete audio tokens: More than a survey!” Transactions on Machine Learning Research, 2025. [Online]. Available: https://openreview.net/forum?id=eqNchtvc6v

## I. SUPPLEMENTARY MATERIAL

All sections, equation, and figure references below refer to the main paper unless stated otherwise.

## A. Upper Bound of the Expected Quantization Error

a) Setup.: Let $\textbf { H } \in ~ \mathbb { R } ^ { T \times D }$ be the frame-level embeddings drawn from an in-domain speech distribution. At layer l, the compression step performs two operations: (1) averaging the frame-level embeddings within each segment, and (2) repeating each segment-level embedding according to its duration. In Sections II-B and III-A, the repetition happens after the quantization layer rather than before, since all repeated frames within a segment are identical, the two orders yield the same result. We therefore combine both operations into a single matrix multiplication $\mathbf { A } ^ { l } \mathbf { R } ^ { l }$ , where $\dot { \mathbf { A } ^ { l } } \in \mathbb { R } ^ { T \times T }$ is the compression matrix at layer l and $\mathbf { R } ^ { l }$ is the frame-level residual embedding from Section III-A, with $\mathbf { R } ^ { 1 } = \mathbf { H }$ . The matrix $\mathbf { A } ^ { l }$ is block-diagonal: the block of the m-th segment has every entry equal to $1 / d _ { m }$ , where $d _ { m }$ is the segment duration. From this structure, $\mathbf { A } ^ { l }$ is symmetric $( ( \mathbf { A } ^ { l } ) ^ { \top } = \mathbf { A } ^ { l } )$ and idempotent $( ( \mathbf { A } ^ { l } ) ^ { 2 } = \mathbf { A } ^ { l } )$

Let $\mathrm { V Q } _ { l } ( \cdot )$ denote quantization at layer l followed by upsampling, so that the frame-level quantized embedding is $\hat { \mathbf { Q } } ^ { l } = \mathrm { V Q } _ { l } ( \mathbf { A } ^ { l } \mathbf { R } ^ { l } )$ and the residual update of Section III-A becomes $\bar { \mathbf { R } } ^ { i + 1 } = \mathbf { \bar { R } } ^ { l } - \hat { \mathbf { Q } } ^ { l }$ . Since $\hat { \mathbf { Q } } ^ { l }$ is constant within each segment of layer $l ,$ applying the compression matrix leaves it unchanged:

$$
\mathbf { A } ^ { l } \hat { \mathbf { Q } } ^ { l } = \hat { \mathbf { Q } } ^ { l } .\tag{1}
$$

We analyze the boundary anchor setting of Section III-C:   
layers ${ \mathit { l } } \leq { \mathit { l } } ^ { * }$ compute new boundaries, while layers $l > l ^ { * }$   
reuse the compression matrix of layer $l ^ { * }$ , denoted $\mathbf { A } ^ { \star } = \mathbf { A } ^ { l ^ { * } }$ b) Assumptions.:

1) Strict compression: every layer merges at least two frames $( M ^ { l } < T )$ , so the compression error is strictly positive: $\mathbb { E } [ \| \mathbf { R } ^ { l } - \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] > \bar { 0 }$

2) Centroid condition: each quantization layer is welltrained. Its codewords satisfy the centroid condition [1], i.e., each codeword equals the conditional mean of the inputs assigned to it. We further assume that each layer captures positive energy, $\mathbb { E } [ \Vert \hat { \mathbf { Q } } ^ { l } \Vert _ { F } ^ { 2 } ] ~ > ~ 0 .$ , and the resulting distortion ratios (Lemma 2) are uniformly bounded by $\epsilon _ { \mathrm { m a x } } = \mathrm { s u p } _ { l } \epsilon _ { l } < 1$

3) Positive energy capture: we define

$$
\alpha _ { l } = \frac { \mathbb { E } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] } { \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] }\tag{2}
$$

as the fraction of expected residual energy captured by the compression step at layer l, and assume it is uniformly bounded below by $\alpha _ { \mathrm { m i n } } ~ = ~ \mathrm { i n f } _ { l } \alpha _ { l } ~ > ~ 0$ Lemma 1 shows $\alpha _ { l } \leq 1$ , together with Assumption 1 and Equation (11), $\alpha _ { l } < 1$

Lemma 1 (Compression never increases energy). For any $\mathbf { R } \in \mathbb { R } ^ { T \times D }$ and any compression matrix A,

$$
\| \mathbf { A R } \| _ { F } ^ { 2 } \leq \| \mathbf { R } \| _ { F } ^ { 2 } .\tag{3}
$$

Proof. The compression matrix acts independently on each segment and each feature dimension, so it suffices to prove the claim for a single segment and a single dimension. Let $x _ { 1 } , \ldots , x _ { d _ { m } }$ denote the scalar values within a segment of duration $d _ { m }$ . Compression replaces every value with the segment mean $\begin{array} { r } { \mu = \frac { 1 } { d _ { m } } \sum _ { i = 1 } ^ { d _ { m } } x _ { i } } \end{array}$ , so the energy of the segment changes from $\textstyle \sum _ { i = 1 } ^ { d _ { m } } x _ { i } ^ { 2 }$ to $\begin{array} { r } { d _ { m } \mu ^ { 2 } = \frac { 1 } { d _ { m } } \big ( \sum _ { i = 1 } ^ { d _ { m } } x _ { i } \big ) ^ { 2 } } \end{array}$ . By the Cauchy–Schwarz inequality with the all-ones vector,

$$
\biggl ( \sum _ { i = 1 } ^ { d _ { m } } x _ { i } \cdot 1 \biggr ) ^ { 2 } \leq \biggl ( \sum _ { i = 1 } ^ { d _ { m } } x _ { i } ^ { 2 } \biggr ) \Bigl ( \sum _ { i = 1 } ^ { d _ { m } } 1 ^ { 2 } \Bigr ) = d _ { m } \sum _ { i = 1 } ^ { d _ { m } } x _ { i } ^ { 2 } ,\tag{4}
$$

which gives $\begin{array} { r } { d _ { m } \mu ^ { 2 } \le \sum _ { i = 1 } ^ { d _ { m } } x _ { i } ^ { 2 } } \end{array}$ . Summing over all segments and all feature dimensions yields the claim. ■

Lemma 2 (Expected rate-distortion bound). Under Assumption 2, the expected quantization error at layer l satisfies

$$
\mathbb { E } \big [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l } \| _ { F } ^ { 2 } \big ] \leq \epsilon _ { l } \mathbb { E } \big [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } \big ] ,\tag{5}
$$

where $\epsilon _ { l } = 1 - \mathbb { E } [ \| \hat { \mathbf { Q } } ^ { l } \| _ { F } ^ { 2 } ] / \mathbb { E } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] < 1 .$

Proof. Write $\mathbf { X } = \mathbf { A } ^ { l } \mathbf { R } ^ { l }$ for the quantization input and $\hat { \mathbf { X } } =$ $\hat { \mathbf { Q } } ^ { l }$ for its output. The quantization partitions the input space into regions $\{ \nu _ { k } \}$ and maps every input in region $\nu _ { k }$ to the output $\mathbf { c } _ { k }$ . By the centroid condition, $\mathbf { c } _ { k } = \mathbb { E } [ \mathbf { X } \mid \mathbf { X } \in \mathcal { V } _ { k } ]$ so the expected quantization error within each region is zero:

$$
\operatorname { \mathbb { E } } [ \mathbf { X } - \mathbf { c } _ { k } \mid \mathbf { X } \in \mathcal { V } _ { k } ] = \operatorname { \mathbb { E } } [ \mathbf { X } \mid \mathbf { X } \in \mathcal { V } _ { k } ] - \mathbf { c } _ { k } = \mathbf { 0 } .\tag{6}
$$

By the law of total expectation, the quantization error is therefore orthogonal to the quantization output in expectation:

$$
\begin{array} { r l r } {  { \mathbb { E } [  \mathbf { X } - \hat { \mathbf { X } } , \hat { \mathbf { X } }  _ { F } ] = \sum _ { k } \mathbb { P } ( \mathbf { X } \in \mathcal { V } _ { k } )  \mathbb { E } [ \mathbf { X } - \mathbf { c } _ { k } \mid \mathbf { X } \in \mathcal { V } _ { k } ] , \mathbf { c } _ { k }  _ { F } } } \\ & { } & \\ & { = 0 , } & { ( 7 ) } \end{array}
$$

where $\mathbf { c } _ { k }$ is pulled out of the conditional expectation because it is constant within its region. The cross-term thus vanishes in the expected Pythagorean decomposition:

$$
\mathbb { E } [ \| \mathbf { X } \| _ { F } ^ { 2 } ] = \mathbb { E } [ \| \hat { \mathbf { X } } \| _ { F } ^ { 2 } ] + \mathbb { E } [ \| \mathbf { X } - \hat { \mathbf { X } } \| _ { F } ^ { 2 } ] .\tag{8}
$$

Rearranging gives the exact equality

$$
\mathbb { E } [ \| \mathbf { X } - \hat { \mathbf { X } } \| _ { F } ^ { 2 } ] = \Big ( 1 - \frac { \mathbb { E } [ \| \hat { \mathbf { X } } \| _ { F } ^ { 2 } ] } { \mathbb { E } [ \| \mathbf { X } \| _ { F } ^ { 2 } ] } \Big ) \mathbb { E } [ \| \mathbf { X } \| _ { F } ^ { 2 } ] = \epsilon _ { l } \mathbb { E } [ \| \mathbf { X } \| _ { F } ^ { 2 } ] ,\tag{9}
$$

and Assumption $2 \ ( \mathbb { E } [ \| \hat { \mathbf { X } } \| _ { F } ^ { 2 } ] > 0 )$ gives $\epsilon _ { l } < 1$ ■ Lemma 3 (Orthogonality). For any R, $\textbf { S } \in \ \mathbb { R } ^ { T \times D }$ and any compression matrix A, the compression error $\mathbf { R } - \mathbf { A } \mathbf { R }$ is orthogonal to any compressed embedding AS under the Frobenius inner product:

$$
\langle \mathbf { R } - \mathbf { A } \mathbf { R } , \mathbf { A } \mathbf { S } \rangle _ { F } = \operatorname { T r } \big ( \mathbf { R } ^ { \top } ( \mathbf { A } - \mathbf { A } ^ { 2 } ) \mathbf { S } \big ) = 0 ,\tag{10}
$$

using symmetry and idempotence of A. In particular, choosing $\mathbf { S } = \mathbf { R }$ yields the Pythagorean identity:

$$
\| \mathbf { R } \| _ { F } ^ { 2 } = \| \mathbf { A R } \| _ { F } ^ { 2 } + \| \mathbf { R } - \mathbf { A R } \| _ { F } ^ { 2 } .\tag{11}
$$

Lemma 4 (Per-layer energy contraction). For every layer l that computes new boundaries,

$$
\mathbb { E } [ \| \mathbf { R } ^ { l + 1 } \| _ { F } ^ { 2 } ] \leq \lambda _ { l } \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] ,\tag{12}
$$

where $\lambda _ { l } = 1 - \alpha _ { l } ( 1 - \epsilon _ { l } ) < 1$

Proof. Decompose the residual update as ${ \bf R } ^ { l + 1 } \ = \ ( { \bf R } ^ { l } \ -$ $\mathbf { A } ^ { l } \mathbf { R } ^ { l } ) + ( \mathbf { A } ^ { l } \bar { \mathbf { R } ^ { l } } - \hat { \mathbf { Q } } ^ { l } )$ . By Equation (1), the second term can be rewritten as a compressed embedding:

$$
\mathbf { A } ^ { l } \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l } = \mathbf { A } ^ { l } \mathbf { R } ^ { l } - \mathbf { A } ^ { l } \hat { \mathbf { Q } } ^ { l } = \mathbf { A } ^ { l } ( \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l } ) .\tag{13}
$$

Therefore, the two terms are orthogonal by Lemma 3 with $\mathbf { S } = \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l }$ , and their squared norms add:

$$
\begin{array} { r l r } {  { \mathbb { E } [ \| \mathbf { R } ^ { l + 1 } \| _ { F } ^ { 2 } ] = \mathbb { E } [ \| \mathbf { R } ^ { l } - \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] + \mathbb { E } [ \| \mathbf { A } ^ { l } \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l } \| _ { F } ^ { 2 } ] } } \\ & { } & { \leq ( 1 - \alpha _ { l } ) \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] + \epsilon _ { l } \alpha _ { l } \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] } \\ & { } & { = \lambda _ { l } \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] . } \end{array}\tag{4}
$$

The first term follows from Equation (11) and the definition of α<sub>l</sub> in Equation (2), which together give $\mathbb { E } [ \| \mathbf { R } ^ { l } - \mathbf { A } ^ { l } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] =$ $\left( 1 - \alpha _ { l } \right) \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ]$ . The second term applies Equation (5) and then Equation $\begin{array} { r l r } {  { ( 2 ) \colon \mathbb { E } [ \| { \bf A } ^ { l } { \bf R } ^ { l } - \hat { { \bf Q } } ^ { l } \| _ { F } ^ { 2 } ] } \le \epsilon _ { l } \mathbb { E } [ \| { \bf A } ^ { l } { \bf R } ^ { l } \| _ { F } ^ { 2 } ] }  \end{array}$ $\epsilon _ { l } \alpha _ { l } \mathbb { E } [ \| \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ]$ ■

Lemma 5 (Invariance of the compression error). If $\mathbf { A } ^ { l } =$ $\mathbf { A } ^ { \star }$ for all ${ \mathit { l } } \geq { \mathit { l } } ^ { * }$ , then for all such layers:

$$
\mathbf { R } ^ { l + 1 } - \mathbf { A } ^ { \star } \mathbf { R } ^ { l + 1 } = \mathbf { R } ^ { l } - \mathbf { A } ^ { \star } \mathbf { R } ^ { l } .\tag{15}
$$

Proof. Applying $\mathbf { A } ^ { \star }$ to the residual update $\mathbf { R } ^ { l + 1 } = \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l }$ and using Equation (1):

$$
{ \bf A } ^ { \star } { \bf R } ^ { l + 1 } = { \bf A } ^ { \star } { \bf R } ^ { l } - \hat { \bf Q } ^ { l } .\tag{16}
$$

Subtracting Equation (16) from the residual update cancels $\hat { \mathbf { Q } } ^ { l }$ and yields the claim. Hence, the compression error introduced at layer l<sup>∗</sup> passes through all subsequent layers unchanged: it cannot be reduced by any quantization that reuses $\mathbf { A } ^ { \star }$ ■ Theorem 1 (Hierarchy of error bounds). After all $C$ quantization layers, the expected residual error satisfies:

$$
\mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] \le \Big [ 1 - \alpha _ { l ^ { * } } \big ( 1 - \epsilon _ { \operatorname* { m a x } } ^ { C - l ^ { * } + 1 } \big ) \Big ] \prod _ { l = 1 } ^ { l ^ { * } - 1 } \lambda _ { l } \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] .\tag{17}
$$

Proof. Decompose the final residual as ${ \bf R } ^ { C + 1 } = ( { \bf R } ^ { C + 1 } -$ $\mathbf { A } ^ { \star } \mathbf { \bar { R } } ^ { C + 1 } ) + \mathbf { \bar { A } } ^ { \star } \mathbf { R } ^ { C + 1 }$ . The two parts are orthogonal by Lemma 3, and applying Lemma 5 repeatedly over layers $l ^ { * } , \ldots , C$ replaces the first part with the compression error at layer l<sup>∗</sup>:

$$
\begin{array} { r l } & { \mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] = \mathbb { E } \big [ \| \mathbf { R } ^ { l ^ { * } } - \mathbf { A } ^ { \star } \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } \big ] } \\ & { \qquad + \mathbb { E } \big [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } \big ] . } \end{array}\tag{18}
$$

The first term equals $( 1 - \alpha _ { l ^ { * } } ) \mathbb { E } [ \| \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } ]$ by Equation (11) and the definition of $\alpha _ { l ^ { * } }$ in Equation (2).

For the second term, consider any layer ${ \mathit { l } } \geq { \mathit { l } } ^ { * }$ . These layers share the same compression matrix $( \mathbf { A } ^ { l } \ = \mathbf { A } ^ { \star } )$ , so Equation (16) shows that $\mathbf { A } ^ { \star } \mathbf { R } ^ { l + 1 }$ is exactly the quantization error of layer l. Since $\mathbf { A } ^ { l } = \mathbf { A } ^ { \star }$ , the quantizer input $\mathbf { A } ^ { l } \mathbf { R } ^ { l }$ equals $\mathbf { A } ^ { \star } \mathbf { R } ^ { l }$ , so Lemma 2 (Equation (5)) applies with input $\mathbf { A } ^ { \star } \mathbf { R } ^ { l }$ and bounds it:

$$
\begin{array} { r } { \mathbb { E } [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { l + 1 } \| _ { F } ^ { 2 } ] = \mathbb { E } [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { l } - \hat { \mathbf { Q } } ^ { l } \| _ { F } ^ { 2 } ] \leq \epsilon _ { \operatorname* { m a x } } \mathbb { E } [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { l } \| _ { F } ^ { 2 } ] . } \end{array}\tag{19}
$$

Applying this bound recursively over the $C - l ^ { * } + 1$ layers from l<sup>∗</sup> to $C ,$ , and then converting the compressed energy at layer l<sup>∗</sup> to residual energy via Equation (2):

$$
\begin{array} { r l } & { \mathbb { E } [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] \leq \epsilon _ { \operatorname* { m a x } } ^ { C - l ^ { * } + 1 } \mathbb { E } [ \| \mathbf { A } ^ { \star } \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } ] } \\ & { \qquad = \epsilon _ { \operatorname* { m a x } } ^ { C - l ^ { * } + 1 } \alpha _ { l ^ { * } } \mathbb { E } [ \| \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } ] . } \end{array}\tag{20}
$$

Summing the two terms gives $\begin{array} { r l r } { \Big [ 1 } & { { } - } & { \alpha _ { l ^ { * } } \big ( 1 \mathrm { ~ \ t ~ \ } - } \end{array}$ $\epsilon _ { \mathrm { m a x } } ^ { C - l ^ { * } + 1 } \big ) \big ] \mathbb { E } [ \| \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } ] .$ . Bounding $\mathbb { E } [ \| \mathbf { R } ^ { l ^ { * } } \| _ { F } ^ { 2 } ]$ by unrolling Lemma 4 over layers $1 , \ldots , l ^ { * } - 1$ yields Equation (17). ■ Equation (17) reveals a hierarchy of bounds as $C  \infty \colon$

• Shared segmentation boundary $( l ^ { * } = 1 )$ . All layers reuse the boundaries of the first layer, corresponding to single compression. The second term of Equation (18) vanishes and the error converges to a constant floor:

$$
\operatorname* { l i m } _ { C \to \infty } \mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] = \left( 1 - \alpha _ { 1 } \right) \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] ,\tag{21}
$$

which is exactly the compression error of the first layer and is strictly positive by Assumption 1.

• Boundary anchor $( 1 ~ < ~ l ^ { \ast } ~ < ~ C )$ . The error floor is reduced by the contraction of the layer-wise phase:

$$
\operatorname* { l i m } _ { C \to \infty } \mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] \leq ( 1 - \alpha _ { l ^ { * } } ) \prod _ { l = 1 } ^ { l ^ { * } - 1 } \lambda _ { l } \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] .\tag{22}
$$

• Layer-wise compression without anchor (LACE). Every layer computes new boundaries, so Lemma 4 applies at every layer:

$$
\mathbb { E } [ \| \mathbf { R } ^ { C + 1 } \| _ { F } ^ { 2 } ] \le \prod _ { l = 1 } ^ { C } \lambda _ { l } \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] \le \lambda _ { \operatorname* { m a x } } ^ { C } \mathbb { E } [ \| \mathbf { H } \| _ { F } ^ { 2 } ] ,\tag{23}
$$

where $\lambda _ { \operatorname* { m a x } } = 1 - \alpha _ { \operatorname* { m i n } } ( 1 - \epsilon _ { \operatorname* { m a x } } ) < 1$ , so the bound converges to zero as $C  \infty$

Therefore, recomputing boundaries inside the residual loop removes the constant error floor inherent to a shared segmentation boundary, while the boundary anchor interpolates between the two regimes.

Remark. This analysis assumes that layers $l > l ^ { * }$ reuse the segmentation boundary of layer $l ^ { * }$ exactly. In our implementation (Section III-C), these layers may instead select any subset of $B ^ { \mathrm { a n c h o r } }$ . The theorem corresponds to the special case where the full segmentation boundary of layer $l ^ { * }$ is reused.

## REFERENCES

[1] A. Gersho and R. M. Gray, Vector Quantization and Signal Compression. Kluwer Academic Publishers, 1992.