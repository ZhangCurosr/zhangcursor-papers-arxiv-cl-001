# Dynamic Semantic Compression for Eficient Latent-Space Inference in Large Language Models

Peipei Li, Dongsen Zhang, Yuchen Liu, Wenjun Xu<sup>∗</sup>

Beijing University of Posts and Telecommunications, Beijing, China

## Abstract

Large Language Models (LLMs) primarily perform inference at the token level, resulting in substantial memory overhead and compromised computational eficiency. In this paper, we propose a Dynamic Semantic Extraction and Inference (DSEI) framework, which achieves segment-level inference within the latent space through a two-stage training strategy. First, we construct a Dynamic Semantic Autoencoder (DSAE) via self-supervised learning. DSAE dynamically extracts segment-level semantics and compresses them into compact latent representations via adaptive semantic weighting and gated fusion. Subsequently, we integrate the DSAE into the LLM architecture and train the model to infer over dense latent space. DSEI substantially reduces both input and generation sequences and significantly enhances inference eficiency. Extensive experiments conducted on the Wanjuan dataset demonstrate that DSEI reduces perplexity by 48% compared to static sentence-level latent inference baseline. Furthermore, compared to standard LLMs using token-level inference, DSEI accelerates inference speed by 2.5× and reduces memory overhead by 90%.

Keywords: Dynamic Semantic Compression, Segment-Level Inference, Eficient LLM Inference, Autoencoder

## 1. Introduction

Autoregressive large language models process and generate text at the token level. While this formulation provides fine-grained linguistic control, it also creates long sequential computation paths, intensive memory access, and high deployment costs as shown in Fig. 1(top). These costs become particularly restrictive when LLMs are deployed in resource-constrained or latency-sensitive engineering scenarios. Reducing the computational granularity of LLM inference from discrete tokens to compact semantic units is therefore a promising direction for eficient LLM deployment.

Recent latent-space inference methods take an important step in this direction by replacing token-level Chains-of-Thought (CoT) with latent representations, thereby reducing the number of autoregressive reasoning steps [1], [2], [3]. While valuable, such approaches are generally limited to compressing the model’s intermediate reasoning process. In contrast, [4] encodes each sentence into a latent representation, extending latent-space inference to the entire input-output stream as shown in Fig. 1(middle). However, this fixed granularity compression paradigm remains limited. It compresses every sentence into a single latent representation regardless of sentence length and treats all tokens uniformly during encoding. As a result, it overlooks both sentence-level and token-level diferences in semantic density, which may lead to information loss or redundant compression and limit inference performance.

To address these limitations, we propose Dynamic Semantic Extraction and Inference (DSEI), a framework designed to improve the eficiency of LLM inference by shifting computation from token-level processing to adaptive segment-level latent-space inference as shown in Fig. 1(bottom). DSEI integrates a Dynamic Semantic Autoencoder (DSAE) into the LLM inference pipeline to extract compact yet informative semantic representations from text segments. Specifically, DSAE is first trained in a self-supervised manner to reconstruct input sentences from their latent representations. In the encoder, each sentence is divided into segments according to its token length, and token-level importance weights are estimated from hidden states to capture semantic contributions with diferent levels of information density. The weighted token representations are then aggregated to form dynamic semantic features, which are further fused with basic semantic features through a gating mechanism. The decoder is trained to reconstruct the original sentences from these representations. After this stage, the trained encoder and decoder are attached to the input and output sides of the LLM, respectively, and the overall model is further optimized through end-to-end training. In this way, DSEI reduces both input and output sequence lengths while main-

Comparison of inference granularity  
1) Token-level inference  
![](images/cfad29b8a6b1167bdb2472257dd96e108c138f9ae80a6037269b3951323901fc.jpg)

2) Fixed sentence-level latent inference  
![](images/c986187d5f8b0d6c63ec5614b915b03b8841ecce3670b6fb2763af15ec36038d.jpg)

3) Proposed adaptive semantic tokenization  
![](images/4f97441aa8d5756999b9c45162634b7c87165e483f6e36139896932e2bef28bb.jpg)  
Figure 1: Comparison of token-level inference, fixed sentence-level latent inference, and the proposed dynamic segment-level latent inference. DSEI segments sentences according to token length and extracts compact semantic representations based on token-level importance, reducing both input and output sequence lengths while maintaining semantic fidelity.

taining semantic fidelity.

Extensive experiments on the Wanjuan-1.0 [5] dataset demonstrate that, compared to static semantic extraction baselines, DSAE reduces perplexity by 34% while maintaining computational eficiency. When integrated into LLMs, it achieves a 48% reduction in perplexity (PPL). Furthermore, compared to LLMs that use token-level inference, DSEI reduces memory overhead by 90% while simultaneously lowering PPL by 58%.

Our main contributions are three-fold:

(1) We propose DSAE, a dynamic semantic compressor that partitions sentences into length-based semantic segments and extracts compact latent representations through token-level semantic weighting and gated fusion. This allows for eficient high-fidelity semantic extraction.

(2) We introduce the DSEI framework to integrate DSAE into LLMs, enabling end-to-end segment-level inference within the latent space, thereby reducing the efective sequence length of both input and output streams.

(3) Extensive experiments demonstrate that DSAE reduces perplexity by 34% compared to static semantic extraction baselines, and by 48% when integrated into LLM inference. Furthermore, compared to standard tokenlevel LLM inference, DSEI reduces memory overhead by 90% while lowering perplexity by 58%, demonstrating a practical trade-of between generation quality and deployment eficiency.

## 2. Related Work

## 2.1. Semantic compression

Early research primarily focused on acquiring compact sentence vectors or instruction embeddings via contrastive learning [6], [7]; these semantic representations are typically utilized for data retrieval and matching. Recent works have explored diverse semantic compression paradigms. For instance, GIST compresses prompts into a set of Transformer activations via metalearning [8], while the LLMLingua model series [9], [10] achieves hard compression by eliminating tokens with low PPL. Furthermore, ICAE demonstrates the feasibility of employing an encoder-decoder architecture for semantic compression [11]. Building upon this trajectory, 500xCompressor [12] enhances the compression ratio through KV cache compression, whereas C3 [13] compresses semantics using cascaded small-parameter LLMs. SentenceVAE [4], on the other hand, focuses on sentence-level compression, encoding an entire sentence into a single sentence-level latent representation that can be reversibly reconstructed by a decoder. Another prominent direction involves continuous latent space modeling, which achieves semantic compression via difusion models [14], [15], [16], [17]. Within this domain, AR-Difusion [18] improves parallelism and coherence through local causal constraints, while two-stage methods such as LD4LG [19] and PLANNER [20] enhance controllability and eficiency via a compression-then-refinement approach. Unlike fixed sentence-level compression, DSEI uses length-adaptive segmentation and token-level semantic weighting to avoid over-compressing long sentences.

## 2.2. Latent-space inference

Latent reasoning methods aim to perform inference within continuous latent spaces and can be broadly categorized into three primary directions: knowledge internalization, architectural modification, and autoregressive latent reasoning. Knowledge internalization seeks to directly embed logic and reasoning capabilities into model parameters, thereby reducing reliance on explicit chains of thought or explanatory prompts. For instance, iCoT-SI [21] forces the model to internalize reasoning structures by progressively eliminating explicit reasoning steps during training, whereas TwT [22] emphasizes the roles of multi-teacher distillation and habitual reasoning. Pause [23] encodes reasoning steps into dedicated token embeddings, while CoCoMix [24] demonstrates that reasoning capabilities can be implanted through continuous concept mixing during the pre-training phase. Furthermore, [25] reveals the critical impacts of reasoning granularity, representation format, and teacher model selection on distillation eficiency. Architectural modification leverages the hierarchical structure of Transformers to enable selective reasoning and dynamic computation [26], [27], [28], [29], [30]. Approaches like [31] allow models to dynamically adjust computational steps according to problem dificulty, whereas [32], [33] explore supervision and guidance oriented toward implicit trajectories. Other studies [34], [35], [36], [37] focus on novel architectures and parallel inference. Autoregressive latent reasoning utilizes hidden states to represent the thought process [1], [2]. In this vein, CODI [3] constructs an autoregressive latent variable model combined with self-distillation, while [38], [39] concentrate on enhancing the stability of these hidden states. In contrast to latent inference methods that primarily compress intermediate reasoning trajectories into continuous representations, DSEI extends latent-space inference to the complete input-output stream through segment-level semantic representations.

## 3. Method

![](images/3e2dc6c3178899f59f35fe7a5602e6a0c3b69dc5774faef2696b2f2a592ecddc.jpg)  
Figure 2: Overall architecture of the proposed DSEI framework. The input text is first split into sentences, and each sentence is partitioned into one or more segments according to token length. The DSAE encoder extracts segment-level latent representations through token-level semantic weighting and gated fusion. These latent representations are then processed by the DSEI-LLM for latent-space inference, and the generated latent outputs are reconstructed into natural language by the DSAE decoder.

In this section, we introduce the proposed DSEI. Given a vanilla LLM $\mathcal { L }$ and an input text consisting of n sentences $\textbf { x } = \{ x ^ { 1 } , x ^ { 2 } , . . . , x ^ { n } \}$ , the text is tokenized and embedded to obtain m token embedding vectors ${ \mathbf { e } } =$ $\left\{ \boldsymbol { e } _ { 1 : l _ { 1 } } ^ { 1 } , \boldsymbol { e } _ { 1 : l _ { 2 } } ^ { 2 } , . . . , \boldsymbol { e } _ { 1 : l _ { n } } ^ { n } \right\}$ , where $e _ { 1 : l _ { n } } ^ { n }$ denotes the $l _ { n }$ embedding vectors of the n-th sentence $x ^ { n }$ , and $m = \textstyle \sum _ { i = 1 } ^ { n } l _ { i }$ . L processes these embeddings and predicts the next token, generating j sentences comprising t tokens ${ \textbf { o } } =$ $\left\{ { o _ { 1 : b _ { 1 } } ^ { 1 } , o _ { 1 : b _ { 2 } } ^ { 2 } , . . . , o _ { 1 : b _ { j } } ^ { j } } \right\}$ , where $O _ { 1 : b _ { j } } ^ { j }$ represents the $b _ { j }$ tokens of the j-th sentence, and $t = \textstyle \sum _ { i = 1 } ^ { j } b _ { i }$

To address the issue of excessively long inputs and outputs in LLMs, we propose enabling the LLM to perform inference within a segment-level latent space, thereby reducing the input length from m to $\textstyle \sum _ { i = 1 } ^ { n } K _ { i }$ , where each sentence is partitioned into $K _ { i }$ segments. Similarly, the output length is reduced from t to $\textstyle \sum _ { r = 1 } ^ { j } \widehat { K } _ { r }$ , where $\hat { K _ { r } }$ denotes the number of latent segments associated with the r-th output sentence. This requires our method to: i) extract the semantics of sentences and eficiently compress them into the latent space; ii) enable the LLM to comprehend segment-level latent representations and perform segment-by-segment inference. DSEI is designed precisely to achieve these two objectives.

## 3.1. Semantic Autoencoder

As shown in Fig. 2, the input to DSAE is a single sentence, which can be represented as $e _ { 1 : l } ^ { 1 }$ after tokenization and embedding. The encoder is composed of stacking Transformer encoder layers [40], which map these embedding vectors into l hidden states h $= h _ { 1 : l }$ These hidden states h are partitioned into K segments according to the token length. Specifically, sentences with fewer than 15 tokens are treated as a single segment, those with 15–30 tokens are partitioned into two, and longer sentences are partitioned into three segments. Formally, the number of segments is determined by the token length l:

$$
\mathbf { h } ^ { k } = \left\{ h _ { i } { \Bigg | } \left\lfloor { \frac { k \cdot l } { K } } \right\rfloor \leq i < \left\lfloor { \frac { \left( k + 1 \right) \cdot l } { K } } \right\rfloor \right\} , \quad k = 0 , 1 , \ldots , K - 1 ,\tag{1}
$$

where $\mathbf { h } ^ { k }$ denotes the hidden states belonging to the k-th segment.

To extract the segment semantics based on $\mathbf { h } ^ { k }$ , a straightforward approach is to sum $\mathbf { h } ^ { k }$ , as shown in Eq. 2. Although the resulting latent representation $s _ { a }$ retains global information, compressing hidden states containing varying amounts of information equally into one vector also introduces redundancy. To address this issue, we employ a fully connected layer called the Semantic Probe to map $\mathbf { h } ^ { k }$ to one dimension, obtaining a sequence of token weights $\mathbf { w } ^ { k }$ . A softmax function is applied to normalize these weights into attention probabilities. Since $\mathbf { h } ^ { k }$ obtained by the self-attention mechanism is contextdependent, $\mathbf { w } ^ { k }$ can dynamically identify the semantic informativeness of tokens. The latent representation $s _ { b }$ is obtained by the weighted average of $\mathbf { w } ^ { k }$ and $\mathbf { h } ^ { k }$ , as shown in Eq. 3:

$$
s _ { k , a } = \mathrm { L a y e r N o r m } \left( \sum \mathbf { h } ^ { k } \right)\tag{2}
$$

$$
s _ { k , b } = \mathrm { L a y e r N o r m } \left( \mathrm { s o f t m a x } ( \mathrm { l i n e a r } ( \mathbf { h } ^ { k } ) ) \mathbf { h } ^ { k \top } \right)\tag{3}
$$

For the reconstruction task of the decoder, the signal provided by the dynamically extracted $s _ { k , b }$ is too sparse, making it dificult for the decoder to predict a reconstructed output consistent with the input. To address this issue, we combine $s _ { k , a }$ , which contains global information, with $s _ { b }$ through a learnable scalar $w _ { g a t e }$ . This scalar is mapped to a value between 0 and 1 via a sigmoid function, implementing a simple yet efective gating mechanism that balances the contribution of the two representations. The resulting segment latent representation $s _ { k }$ thus incorporates global information while reducing redundancy. This process can be formally represented as:

$$
s _ { k } = { \mathrm { s i g m o i d } } ( w _ { g a t e } ) \cdot s _ { k , a } + ( 1 - { \mathrm { s i g m o i d } } ( w _ { g a t e } ) ) \cdot s _ { k , b }\tag{4}
$$

Given a sentence represented by a sequence of segment-level latent representations $\mathrm { ~ \bf ~ s ~ } = \ s _ { 1 : K }$ , the decoder reconstructs the original sentence from all segment representations jointly. The decoder consists of stacked Transformer decoder layers, where s is used as the key and value in cross-attention. Starting from the start token, the decoder autoregressively generates the reconstructed token probabilities $\textbf { p } = \ p _ { 1 : y }$ , where $p _ { y }$ denotes the predicted probability of the y-th target token. The reconstruction probabilities are then compared with the original input tokens using focal loss [41], which serves as the training objective in the first stage. This process can be formulated as:

$$
{ \mathcal { L } } _ { \mathrm { c o m p } } = \sum _ { i = 1 } ^ { y } { \mathrm { F L } } ( p _ { i } ) = \sum _ { i = 1 } ^ { y } - ( 1 - p _ { i } ) ^ { 2 } { \log ( p _ { i } ) }\tag{5}
$$

The initial parameters of the encoder and decoder in DSAE are transferred from the LLM backbone of DSEI. The encoder and decoder in DSAE maintain the same number of layers. For a DSAE with i layers, since the encoder is responsible for extracting features from text embeddings, its parameters are transferred from the first i layers of the functionally equivalent LLM. Similarly, the decoder is responsible for generating text from latent representations, and its parameters are transferred from the last i layers of the functionally equivalent LLM. Compared to random initialization, parameter transfer endows the model with semantic comprehension capabilities, allowing training to proceed without starting from basic grammatical structure construction. This enables eficient training of the model’s semantic extraction abilities, leading to convergence at better performance.

## 3.2. Segment-by-segment inference

To enable the LLM to perform segment-by-segment inference in the latent space, we integrate DSAE into the LLM and optimize the resulting model in an end-to-end manner. In DSEI, the token embedding layer of the vanilla LLM is replaced by the DSAE encoder and the semantic extraction layer, while the output hidden states of the LLM are decoded into natural language by the DSAE decoder. Given an input text, we first split it into n sentences by pattern matching based on punctuation marks. For the i-th sentence $x ^ { i }$ the DSAE encoder and semantic extraction layer determine the number of segments $K _ { i }$ according to its token length and encode it into a sequence of segment-level latent representations $\boldsymbol { s } _ { 1 : k _ { i } } ^ { i }$ . The latent representations of all sentences are then concatenated in their original textual order to form the LLM input $\mathbf { s } = \left\{ s _ { 1 : k _ { 1 } } ^ { 1 } , s _ { 1 : k _ { 2 } } ^ { 2 } , . . . , s _ { 1 : k _ { n } } ^ { n } \right\}$ , where the total number of latent vectors is $\textstyle \sum _ { i = 1 } ^ { n } K _ { i }$ . This encoding process is performed in parallel across sentences in a batched manner. The LLM takes s as input and autoregressively generates a sequence of output latent representations $\widehat { \mathbf { s } } = \widehat { s } _ { 1 : T }$

Since the LLM produces a flat sequence of segment-level latent representations, DSEI further introduces a segmentation head to recover sentence-level structure before decoding. Specifically, for each generated latent representation $\widehat { s } _ { t }$ , the segmentation head predicts whether it marks the end of a sentence-level latent group. These predicted boundaries partition bs into several groups, each corresponding to one output sentence. The DSAE decoder then reconstructs a natural-language sentence from each latent group, and the concatenation of all reconstructed sentences forms the final output of DSEI.

To determine when the LLM should stop generating segment-level latent representations, we train a fully connected layer called the termination head. During training, the termination objective is computed only for generated latent representations that correspond to ground-truth sentence-level latent group boundaries. Given such a boundary representation $\widehat { s } _ { i }$ , the termination head maps it to a two-dimensional logit vector $\mathbf { d } _ { i } = [ d _ { i } ^ { 1 } , d _ { i } ^ { 2 } ]$ , where the two dimensions indicate whether generation should continue or terminate. If the vector signals an end state, the generation iteration terminates; otherwise, the LLM continues to generate the next latent representation.

During training, we use focal loss [41] to optimize the segmentation head, the termination head, and the latent-generation process jointly. Formally, the total loss is defined as:

$$
\mathcal { L } _ { \mathrm { i n f e r } } = \underbrace { \frac { \lambda } { j } \sum _ { i = 1 } ^ { t } \mathrm { F L } ( p _ { i } ) } _ { \mathrm { g e n e r a t i o n ~ t e r m } } + \underbrace { \sum _ { i = 1 } ^ { t } \mathrm { F L } ( b _ { i } ) } _ { \mathrm { s e g m e n t ~ t e r m } } + \underbrace { \sum _ { t \in \mathcal { B } } \mathrm { F L } ( d _ { i } ) } _ { \mathrm { s t o p ~ t e r m } } ,\tag{6}
$$

where λ is a weighting coeficient used to balance the token-level generation term with the stop and segment terms, $b _ { t }$ denote the predicted probabil-

ity assigned to the ground-truth boundary label of $\widehat { s } _ { t } , d _ { i }$ denote the predicted probability assigned to the ground-truth termination label of $\widehat { s } _ { t }$ , and B is the set of boundary positions at the end of sentence.

## 4. Experiments

We first train DSAE via self-supervised learning, and subsequently integrate the trained DSAE into the LLM for joint end-to-end training. We analyze DSAE and DSEI separately, comparing them with baseline methods and token-by-token LLMs, exploring the characteristics of dynamic semantics, and analyzing the contributions of diferent components.

## 4.1. Experimental setup

Dataset and metrics. For both DSAE and DSEI, all experiments are trained and validated on the English subset of the Wanjuan-1.0 dataset [5]. For DSAE, the training set samples approximately 155M sentences, and the validation set consists of 1K non-overlapping sentences. For DSEI, the training set samples approximately 6.4M paragraphs, and the validation set consists of 1K non-overlapping paragraphs. Following [4], we use PPL as the evaluation metric for DSAE and DSEI. Additionally, we employ mean input throughput and mean GPU memory to assess the computational eficiency of DSEI.

Baseline methods. We compare our method with the baseline approach [4], which includes SVAE, an autoencoder with a static fusion mechanism for sentence compression and reconstruction, and SLLM, a sentence-level LLM that integrates SVAE. In SVAE, the sentence-level latent representation is obtained by summing the hidden states from the last layer of the encoder, and SLLM performs sentence-level inference based on this static latent representation. Additionally, we also compare with token-level inference LLMs from the OPT [42] series.

Implementation details. (1) Base Models: We adopt three OPT [42] series models of sizes 125M, 350M, and 1.3B as our base LLMs to validate the efectiveness of our method across diferent model scales. The dimension of hidden states in DSAE is kept consistent with that of the base LLM. During weight transfer, both the encoder and decoder only utilize the weights of the self-attention modules from the base LLM, excluding the feed-forward network (FFN) weights. (2) Hyperparameters: For DSAE, the maximum input token length is set to 64, batch size to 512, learning rate to 1e-7, and the gating scalar $w _ { g a t e }$ is initialized to 0. In experiments where the base LLM is 125M, the number of layers in DSAE is set to 1, 2, and 4 respectively, while for base LLMs of other sizes it is set to 1. For DSEI, the maximum input sentence length is set to 64, batch size is 4, and the learning rate is 1e-6. For all experiments, λ is set to 0.01. We use the AdamW optimizer [43] with a weight decay of 1e-2 and gradient clipping with a maximum L2 norm of 1. The learning rate follows a linear schedule for the first 5,000 iterations, and then uses a cosine annealing schedule for subsequent iterations. All experiments are conducted on a single RTX 5880 Ada Generation GPU.

![](images/b8a3b55c203ef247f0fccdf53198503f409bd01d7fba5827602a9bb7e245d1c0.jpg)  
(a) PPL for diferent layers with 768 hidden size.

![](images/7fbef757f579b0571732e9ef91a1c8e94ab14165ab506d4ed6c6b58381a44a18.jpg)  
(b) PPL for diferent hidden sizes with 1 layer.  
Figure 3: Comparison of PPL for semantic extraction between DSAE and baseline method across varying (a) layers and (b) hidden sizes. $\triangle$ denotes percentage decrease in PPL of DSAE compared to the baseline method.

## 4.2. Semantic extraction results

Fig. 3 compares the performance of DSAE with the baseline method SVAE under diferent model sizes. DSAE demonstrates consistent performance improvements over the static representation fusion-based semantic extraction method across various model depths (varying layers) and widths (varying hidden sizes). Notably, in the smallest model configuration, DSAE achieves a 34% reduction in PPL compared to the baseline method, which is the largest performance gain among all model sizes. This is particularly advantageous for resource-constrained edge devices, which typically employ smaller-scale models.

Tab. 1 presents the sentence reconstruction performance of DSAE compared with the baseline method across multiple test cases. DSAE demonstrates more accurate sentence reconstruction compared to the baseline method. The token weight results for input sentences show that DSAE pays greater attention to content words rich in semantic information, such as “various” in sample 1 and “pulsed” in sample 2, while assigning lower weights to function words that lack substantial meaning, such as “for”, $^ { 6 6 } \mathrm { a } ^ { \prime \prime }$ , and “of”. This indicates that DSAE can eficiently identify key information in sentences based on context and compress it into a compact latent representation, exhibiting particular advantages when processing longer sentences or those containing uncommon words. For instance, in sample 2, SVAE reconstructed “515” as “660”, while DSAE assigned greater weight to this token during semantic extraction and successfully reconstructed the original token. In sample 3, which contains the proper noun “Schylling”, DSAE assigned it the highest weight and accurately generated the word.

Table 1: Test examples of sentence reconstruction between baseline method and DSAE. The darker the color, the greater the weight of the token. Text in red denotes mismatches between reconstructed sentence and input sentence, and △ denotes missing output.
<table><tr><td rowspan="2" colspan="4">Weight of input sentence</td><td colspan="2">Output sentence</td></tr><tr><td>SVAE</td><td>DSAE</td></tr><tr><td>Some kinds</td><td>also have of artists</td><td>performance venues for</td><td>Some also have performance venues for various kinds of artists.</td><td>Some also have performance venues for various kinds of artists.</td></tr><tr><td>Intense</td><td>pulsed light</td><td>device emit a range</td><td>Intense pulsed light device emit a range of puls(660-1200) △ light.</td><td>Intense pulsed light device emit a range wavelength(515-1200) of light.</td></tr><tr><td>wavelength Makes</td><td>515 the perfect</td><td>1200 of light gift for Schylling enthusiasts</td><td>Makes the perfect gift for Schreferably that</td><td>Makes the perfect gift for Schylling enthusiasts</td></tr><tr><td>that</td><td>are at least</td><td>3 years old</td><td></td><td>enthusiasts at △ 3 years old. that are at least 3 years old.</td></tr><tr><td>His hat</td><td>is accented</td><td>with holly and holly</td><td>berries</td><td>His hat is accented with holly holly and berries. His holly accented with holly and holly berries.</td><td></td></tr></table>

Table 2: The ablation experiment results of DSAE. △ denotes percentage increase in PPL compared to the default setting. “First layer weights” indicates that the initial parameters for both the encoder and the decoder are transferred from the first layer of the base LLM, while “Last layer weights” are derived from the final layer.
<table><tr><td>Method</td><td>PPL↓  $\triangle$ </td></tr><tr><td>Default</td><td>1.06 0.00%</td></tr><tr><td> $\mathrm { w / o }$  Segmentation</td><td>1.40 32.07%</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ s _ { a }$ </td><td>1.14 7.55%</td></tr><tr><td> $\mathrm { w } / \mathrm { o } \ s _ { b }$ </td><td>1.16 9.43%</td></tr><tr><td> $\mathrm { w / o }$  gate</td><td>1.21 14.15%</td></tr><tr><td> $\mathrm { w / o }$  LLM weights</td><td>1.24 16.98%</td></tr><tr><td>First layer weights</td><td>1.13 6.60%</td></tr><tr><td>Last layer weights</td><td>1.17 10.38%</td></tr></table>

To validate the efectiveness of each component in DSAE, we conducted ablation studies on a model with hidden size 768 and 1 layer, with results shown in Tab. 2. Four key findings emerged:

(1) Among all ablated components, sentence segmentation yields the most pronounced efect on semantic compression. Without segmentation, PPL increases by 32% relative to the default setting. This demonstrates that forcing an entire sentence into a single latent representation creates a severe information bottleneck. In contrast, the proposed segmentation strategy assigns multiple latent vectors to longer sentences according to their token length, allowing the compressed representation to preserve richer semantic details.

(2) Static and dynamic latent representations provide complementary semantic information. We directly used the static latent representation $s _ { a }$ and dynamic latent representation $s _ { b }$ separately as sentence semantics for the decoder to reconstruct sentences. We found that when the static latent representation is absent, reconstructed sentences tend to miss words, while when the dynamic latent representation is absent, inconsistent words are reconstructed—both leading to performance degradation.

(3) The gating fusion mechanism is beneficial. In the gating ablation experiment, we simply summed the static latent representation $s _ { a }$ and dynamic latent representation $s _ { b }$ as the sentence latent representation. this simple summation strategy increases PPL by 14%. This confirms the importance of gated fusion of global and dynamic semantic information when extracting sentence semantics.

(4) LLM pre-trained parameters are suitable for semantic extraction. When initializing model parameters randomly instead of transferring from the LLM, performance dropped by 17%. This indicates that the language understanding capabilities acquired from LLM pre-training are well-suited for semantic extraction tasks. Additionally, we investigated the suitability of diferent LLM layers. When initializing encoder and decoder parameters with the first and last layers of the LLM respectively, performance decreased by 6% and 10%. This suggests that shallow layers align better with the encoder—both are responsible for extracting features from text—while deep layers align better with the decoder—both are responsible for generating text based on extracted features.

To further analyze how the decoder utilizes segment-level latent representations, we visualize the cross-attention weights between latent vectors and tokens for sentences of diferent lengths, as shown in Fig. 4. For short sentences, the single latent vector exhibits a broad attention pattern over the entire sentence, indicating that one compact representation is suficient for reconstruction when the semantic content is compact. For medium and long sentences, diferent latent vectors exhibit clear and complementary attention patterns over diferent token regions. This suggests that length-based segmentation enables multiple latent vectors to capture localized semantic information, thereby reducing the burden on a single sentence-level vector and improving the fidelity of sentence reconstruction, thus pointing to a more efective design for long-text semantic compression.

![](images/e47bed524031c12f83f8df885702130e4d58ff2db3c7eb0c4cf2b7b85a8a0817.jpg)

(a) Short sentence.  
![](images/65bbc2fb07fb9f5ed8c13f43b55ee4157955d8c4000cc69188615c2f5bb9214c.jpg)

(b) Middle sentence.  
![](images/e25ae4ea3b3740ae6c576e0b6530360ea2de57ce77f3583caa31b804631f47a8.jpg)  
(c) Long sentence.  
Figure 4: Attention heatmap comparison of decoder cross-attention weights on DSAE tokens for (a) short, (b) middle, and (c) long sentences during semantic compression. Lighter colors indicate higher attention weights, signifying that the decoder assigns greater importance to those tokens when reconstructing the original sentence.

Table 3: Experiment results of baseline methods and DSEI. DSEI-b-l denotes an OPT model with parameter size b, integrated with a DSAE consisting of l layers.
<table><tr><td>Model</td><td>Parameters(M)</td><td>PPL↓</td><td>Mean input throughput (k tokens/s)↑</td><td>Mean GPU memory (KB/token)↓</td></tr><tr><td>OPT-125M</td><td>125.23</td><td>26.94</td><td>13.93</td><td>131.45</td></tr><tr><td>SLLM-125M-H1</td><td>214.32</td><td>21.84</td><td>35.90</td><td>10.97</td></tr><tr><td>DSEI-125M-H1</td><td>214.92</td><td>11.17</td><td>35.98</td><td>13.05</td></tr><tr><td>SLLM-125M-H2</td><td>226.14</td><td>21.20</td><td>37.49</td><td>9.94</td></tr><tr><td>DSEI-125M-H2</td><td>226.76</td><td>15.52</td><td>37.76</td><td>13.18</td></tr><tr><td>SLLM-125M-H4</td><td>249.78</td><td>20.99</td><td>24.36</td><td>10.60</td></tr><tr><td>DSEI-125M-H4</td><td>250.37</td><td>14.57</td><td>23.50</td><td>13.56</td></tr><tr><td>OPT-350M</td><td>331.19</td><td>22.60</td><td>7.71</td><td>184.72</td></tr><tr><td>SLLM-350M-H1</td><td>429.46</td><td>21.51</td><td>99.20</td><td>25.21</td></tr><tr><td>DSEI-350M-H1</td><td>432.66</td><td>14.12</td><td>98.30</td><td>27.85</td></tr><tr><td>OPT-1.3B</td><td>1315.75</td><td>16.02</td><td>7.9</td><td>285.24</td></tr><tr><td>SLLM-1.3B-H1</td><td>1605.67</td><td>19.30</td><td>60.88</td><td>50.44</td></tr><tr><td>DSEI-1.3B-H1</td><td>1609.88</td><td>9.86</td><td>59.53</td><td>52.07</td></tr></table>

## 4.3. Latent inference results

Tab. 3 presents a comparison between DSEI and existing baseline methods in terms of PPL, throughput, and GPU memory overhead. Experimental results demonstrate that DSEI achieves an optimal balance between inference eficiency and accuracy. Across various parameter scales of LLM backbones, DSEI consistently exhibits improved PPL performance compared to static sentence-level latent inference baseline SLLM.

It is worth noting that the quality improvement brought by DSEI is particularly pronounced on smaller-scale LLMs, while its throughput remains comparable to SLLM and its memory overhead remains substantially lower than token-level inference. On a 125M-parameter model, compared with SLLM, a single-layer DSEI configuration achieves a 48% reduction in PPL, although the average input throughput increases only marginally by 0.22% and the mean GPU memory overhead increases by 18%. These observations suggest that DSEI may be particularly suitable for enhancing LLM inference performance in resource-constrained, practical deployment scenarios.

Furthermore, DSEI relies on a lightweight dynamic semantic extraction mechanism and introduces only approximately 0.6M additional parameters compared with SLLM. Despite this marginal parameter increase, it preserves the inherent eficiency advantages of semantic-level inference over token-bytoken LLM inference in terms of both throughput and GPU memory overhead. Overall, these results demonstrate that DSEI provides a favorable trade-of between generation quality and inference eficiency.

## 5. Conclusion

In this paper, we propose the Dynamic Semantic Autoencoder (DSAE), which dynamically compresses sentences into segment-level latent representations. We integrate DSAE into Large Language Models (LLMs) via the Dynamic Semantic Extraction and Inference (DSEI) framework, enabling the LLM to conduct the entire inference process in the latent space. Our approach incorporates three key innovations: (1) employing a sentence segmentation mechanism to divide entire sentences into segments, thereby enhancing the model’s semantic extraction capability for long sentences; (2) eficiently extracting dynamic semantics by assigning semantic weights to tokens through a lightweight contextual awareness mechanism; and (3) utilizing a gating mechanism to fuse dynamic and static semantics, which preserves global information while eliminating redundancy, thereby striking a balance between performance and eficiency. Experimental results demonstrate that, compared to static semantic extraction baselines, DSAE achieves a maximum improvement of 34% in $\mathrm { P P L } ,$ and DSEI reduces perplexity by 48%. These results demonstrate that DSEI provides a practical trade-of between generation quality and deployment eficiency. It improves latent-space inference quality while preserving the computational advantages of compressed semantic representations.

## References

[1] S. Hao, S. Sukhbaatar, D. Su, X. Li, Z. Hu, J. Weston, Y. Tian, Training large language models to reason in a continuous latent space, in: COLM, 2025.

[2] W. Tan, J. Li, J. Ju, Z. Luo, R. Song, J. Luan, Think silently, think fast: Dynamic latent compression of llm reasoning chains, arXiv preprint arXiv:2505.16552 (2025).

[3] Z. Shen, H. Yan, L. Zhang, Z. Hu, Y. Du, Y. He, Codi: Compressing chain-of-thought into continuous space via self-distillation, in: EMNLP, 2025.

[4] H. An, Y. Chen, Z. Sun, X. Li, Sentencevae: Enable next-sentence prediction for large language models with faster speed, higher accuracy and longer context, arXiv preprint arXiv:2408.00655 (2024).

[5] C. He, Z. Jin, C. Xu, J. Qiu, B. Wang, W. Li, et al., Wanjuan: A comprehensive multimodal dataset for advancing english and chinese large models, arXiv preprint arXiv:2308.10755 (2023).

[6] N. Reimers, I. Gurevych, Sentence-bert: Sentence embeddings using siamese bert-networks, in: EMNLP, 2019.

[7] T. Gao, X. Yao, D. Chen, Simcse: Simple contrastive learning of sentence embeddings, in: EMNLP, 2021.

[8] J. Mu, X. Li, N. Goodman, Learning to compress prompts with gist tokens, in: NeurIPS, 2023.

[9] Z. Pan, Q. Wu, H. Jiang, et al., Llmlingua-2: Data distillation for eficient and faithful task-agnostic prompt compression, in: ACL Findings, 2024.

[10] H. Jiang, Q. Wu, C.-Y. Lin, et al., Llmlingua: Compressing prompts for accelerated inference of large language models, in: EMNLP, 2023.

[11] T. Ge, H. Jing, L. Wang, et al., In-context autoencoder for context compression in a large language model, in: ICLR, 2024.

[12] Z. Li, Y. Su, N. Collier, et al., 500xcompressor: Generalized prompt compression for large language models, in: ACL, 2025.

[13] F. Liu, H. Qiu, Context cascade compression: Exploring the upper limits of text compression, arXiv preprint arXiv:2511.15244 (2025).

[14] V. Meshchaninov, E. Chimbulatov, A. Shabalin, et al., Cosmos: Compressed and smooth latent space for text difusion modeling, in: NeurIPS, 2025.

[15] S. Gong, M. Li, J. Feng, Z. Wu, L. Kong, Difuseq: Sequence to sequence text generation with difusion models, in: ICLR, 2023.

[16] H. Yuan, Z. Yuan, C. Tan, F. Huang, S. Huang, Seqdifuseq: Text difusion with encoder-decoder transformers, arXiv preprint arXiv:2212.10325 (2022).

[17] A. Shabalin, V. Meshchaninov, E. Chimbulatov, et al., Tencdm: Understanding the properties of the difusion model in the space of language model encodings, in: AAAI, 2025.

[18] T. Wu, Z. Fan, X. Liu, H.-T. Zheng, Y. Gong, J. Jiao, J. Li, J. Guo, N. Duan, W. Chen, et al., Ar-difusion: Auto-regressive difusion model for text generation, NeurIPS (2023).

[19] J. Lovelace, V. Kishore, C. Wan, E. Shekhtman, K. Q. Weinberger, Latent difusion for language generation, NeurIPS (2023).

[20] Y. Zhang, J. Gu, Z. Wu, S. Zhai, J. Susskind, N. Jaitly, Planner: Generating diversified paragraph via latent language difusion model, NeurIPS (2023).

[21] Y. Deng, Y. Choi, S. Shieber, From explicit cot to implicit cot: Learning to internalize cot step by step, arXiv preprint arXiv:2405.14838 (2024).

[22] J. Xu, M. Zhou, W. Liu, H. Liu, S. Han, D. Zhang, Twt: Thinking without tokens by habitual reasoning distillation with multi-teachers guidance, arXiv preprint arXiv:2503.24198 (2025).

[23] S. Goyal, Z. Ji, A. S. Rawat, A. K. Menon, S. Kumar, V. Nagarajan, Think before you speak: Training language models with pause tokens, arXiv preprint arXiv:2310.02226 (2023).

[24] J. Tack, J. Lanchantin, J. Yu, A. Cohen, I. Kulikov, J. Lan, S. Hao, Y. Tian, J. E. Weston, X. Li, Llm pretraining with continuous concepts, in: NeurIPS, 2025.

[25] X. Chen, Z. Sun, G. Wenjin, M. Zhang, Y. Chen, Y. Sun, H. Su, Y. Pan, D. Klakow, W. Li, et al., Unveiling the key factors for distilling chainof-thought reasoning, in: ACL, 2025.

[26] N. Saunshi, N. Dikkala, Z. Li, S. Kumar, S. J. Reddi, Reasoning with latent thoughts: On the power of looped transformers, in: ICLR, 2025.

[27] Y. Chen, J. Shang, Z. Zhang, et al., Inner thinking transformer: Leveraging dynamic depth scaling to foster adaptive internal thinking, in: ACL, 2025.

[28] J. Cheng, B. Van Durme, Compressed chain of thought: Eficient reasoning through dense representations, arXiv preprint arXiv:2412.13171 (2024).

[29] D. Su, H. Zhu, Y. Xu, J. Jiao, Y. Tian, Q. Zheng, Token assorted: Mixing latent and text tokens for improved language model reasoning, in: ICML, 2025.

[30] A. Mohtashami, M. Pagliardini, M. Jaggi, Cotformer: More tokens with attention make up for less depth, in: NeurIPS, 2023.

[31] A. Mohtashami, M. Pagliardini, M. Jaggi, Cotformer: A chain of thought driven architecture with budget-adaptive computation cost at inference, in: ICLR, 2025.

[32] B. Zeng, S. Song, S. Huang, Y. Wang, H. Li, Z. He, X. Wang, Z. Li, Z. Lin, Pretraining language models to ponder in continuous space, arXiv preprint arXiv:2505.20674 (2025).

[33] H. Du, Y. Dong, X. Ning, Latent thinking optimization: Your latent reasoning language model secretly encodes reward signals in its latent thoughts, arXiv preprint arXiv:2509.26314 (2025).

[34] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, T. Dao, Medusa: Simple llm inference acceleration framework with multiple decoding heads, in: ICML, 2024.

[35] Z. Chen, A. May, R. Svirschevski, Y. Huang, M. Ryabinin, Z. Jia, B. Chen, Sequoia: Scalable, robust, and hardware-aware speculative decoding, arXiv preprint arXiv:2402.12374 (2024).

[36] J. Ye, S. Gong, L. Chen, L. Zheng, J. Gao, H. Shi, C. Wu, X. Jiang, Z. Li, W. Bi, et al., Difusion of thought: Chain-of-thought reasoning in difusion language models, NeurIPS (2024).

[37] Z. Huang, Z. Chen, Z. Wang, et al., Reinforcing the difusion chain of lateral thought with difusion language models, in: NeurIPS, 2025.

[38] H. Wu, Z. Teng, K. Tu, Parallel continuous chain-of-thought with jacobi iteration, in: EMNLP, 2025.

[39] X. Wei, X. Liu, Y. Zang, X. Dong, Y. Cao, J. Wang, X. Qiu, D. Lin, Sim-cot: Supervised implicit chain-of-thought, arXiv preprint arXiv:2509.20317 (2025).

[40] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin, Attention is all you need, NeurIPS (2017).

[41] T.-Y. Lin, P. Goyal, R. Girshick, K. He, P. Doll, Focal loss for dense object detection, in: ICCV, 2017.

[42] S. Zhang, S. Roller, N. Goyal, M. Artetxe, M. Chen, S. Chen, C. Dewan, M. Diab, X. Li, X. V. Lin, et al., Opt: Open pre-trained transformer language models, arXiv preprint arXiv:2205.01068 (2022).

[43] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: ICLR, 2019.