# Heads That Write, Not Just Point: Image Retrieval Heads in Vision-Language Models

Junsung Park<sup>1</sup> Sangwon Yu<sup>2</sup> Donghun Kang<sup>1</sup> Yeongtak Oh<sup>1</sup> Jaewon Jeong<sup>3</sup> Gyeongtae Yoo<sup>1</sup> Han Cheol Moon<sup>4</sup> Jungbeom Lee<sup>5∗</sup> Sungroh Yoon<sup>1,3,6∗</sup>

<sup>1</sup>Department of Electrical and Computer Engineering, Seoul National University <sup>2</sup>Upstage AI <sup>3</sup>Interdisciplinary Program in AI, Seoul National University <sup>4</sup>Samsung Electronics <sup>5</sup>Korea University <sup>6</sup>AIIS, ASRI, INMC, and ISRC, Seoul National University

## Abstract

Large vision-language models (LVLMs) can process interleaved text and multiple images in a single context, but how they internally identify the image relevant to a question remains poorly understood. We study this grounding step as incontext image retrieval and introduce ICIR-MCQ, a controlled probe for headlevel analysis of image retrieval. Using each head’s attention mass over imagetoken spans, we first identify sparse pointer heads that reliably point to the target image. These heads transfer beyond the probe: a single pointer head is competitive with external vision-language retrievers on a naturalistic multi-image retrieval benchmark, showing that LVLMs contain a strong internal image retrieval signal. Attention alone, however, shows where a head reads from, not what its output passes on to later layers, so we introduce a complementary output-side analysis. For each head, we train a lightweight image retriever on its value-weighted output and use it to identify writer heads, whose outputs contain enough information to identify the target image. Surprisingly, the pointer and writer criteria select substantially different head sets, with only 35 to 47 heads overlapping among the top-100 heads under each criterion. Targeted set-partition ablations show that the heads most important for standard multi-image inference are concentrated in this pointer-writer intersection, while heads selected by only one criterion show less consistent separation from random-head controls.

The most consistent causal effects arise from heads selected by both the attention-based pointer and output-side writer criteria. Code and data: https://github.com/parkquasar/Image-Retrieval-Heads.

## 1 Introduction

Modern large vision-language models (LVLMs) [3, 6, 21, 40] can process interleaved text and multiple images in a single context, enabling applications such as document understanding and multi-image question answering. In multi-image contexts, answering an image-grounded question requires more than visual recognition: the model must first identify which image the question refers to. We refer to this image-grounding component as in-context image retrieval. This retrieval step gates downstream computation, since grounding the question in the wrong image conditions the answer on the wrong visual evidence. Recent multi-image benchmarks show that model accuracy drops as the number of candidate images increases, exposing in-context image retrieval as a central behavioral bottleneck [39, 41]. However, they leave open how LVLMs internally perform this image selection, and which components in the model carry the image retrieval signal.

Attention-pattern analysis has been a standard tool for identifying specialized heads in transformers [10, 27, 37]. Recent retrieval-head analyses and attention-based VLM head analyses likewise identify heads by where they attend in the input context [4, 8, 42]. This approach is intuitive and has been effective in practice: attention patterns expose which tokens or spans a head points to. However, pointing is only one side of the attention operation. Attention patterns do not by themselves determine what information the head writes into the residual stream. A head’s output is the value-weighted vector produced by mixing value vectors according to its attention weights, and this output, rather than the attention pattern itself, is what downstream layers consume [20].

We hypothesize that this gap between pointing and writing is especially important for in-context image retrieval in LVLMs, where retrieval is cross-modal: attention points to visual-token spans, but the value-weighted output must carry target-image information in a hidden representation used by later language-generation layers. This creates a mismatch between pointing and writing. A head may point cleanly to the target image without writing target-image information in a form recoverable from its output. Conversely, output-side recoverability and attention-mass strength need not induce the same ranking. Attention-only head discovery may fail to distinguish heads that merely point to the target image from heads that also write target-image information into the residual stream.

We test this hypothesis with a two-sided analysis of head-level retrieval in LVLMs. On the input side, we identify pointer heads using attention mass over image-token spans (§4). On the output side, we identify writer heads by training a lightweight head-specific image retriever on each head’s value-weighted output, asking whether the target image can be identified from the information carried by that output (§5). We then use targeted ablations to test which head population the model’s answergeneration pathway depends on. Identifying image retrieval heads in LVLMs therefore requires combining where a head points with what it writes.

To carry out this analysis, we introduce ICIR-MCQ (Figure 1; §3.2), a controlled probe that gives each sample a clear target image while keeping the task solvable by standard LVLMs. ICIR-MCQ is designed for controlled head-level discovery rather than as a standalone downstream benchmark; we evaluate transfer and causal relevance on Visual Haystacks [41], a naturalistic multi-image retrieval benchmark. Across four LVLMs spanning two families and scales, only 35–47 of the top-100 heads under the pointer and writer criteria overlap. Targeted set-partition ablations further show that standard multi-image inference depends primarily on this pointer-writer intersection, while heads selected by only one criterion show less consistent separation from random-head controls (§6).

Our contributions are as follows.

1. A controlled probe for head-level image retrieval. We introduce ICIR-MCQ, a controlled probe with clear target-image labels for head-level retrieval analysis.

2. Pointer heads as strong implicit image retrievers. We identify sparse pointer heads that reproduce at identical (layer, head) indices within shared LLM backbones. A single pointer head is competitive with external vision-language retrievers like CLIP and SigLIP.

3. Writer heads via a complementary output-side criterion. We introduce an output-side head discovery method that trains a lightweight image retriever on each head’s output, identifying writer heads whose outputs retain recoverable target-image information. This retriever-based ranking differs substantially from the attention-based pointer ranking.

4. Causal validation: retrieval heads are pointers that write. Set-partition ablations show that standard multi-image inference is most sensitive to heads selected by both criteria.

## 2 Related Work

Specialized attention heads and head-level interventions in transformers. Prior work has identified specialized transformer heads for syntax [10, 37], induction [27], and long-context factual retrieval [42]. Many of these analyses identify heads primarily from attention patterns over textual contexts. However, attention weights alone do not determine the output of an attention layer [20]. Our output-side analysis is aligned with broader mechanistic work that studies activations, circuits, and interventions beyond raw attention patterns [5, 11, 14, 15, 22, 35]. We extend this perspective to cross-modal image retrieval in LVLMs, where information about the target image must be written to the residual stream rather than directly emitted as an output token.

Mechanistic interpretability of vision-language models. Prior work studies how VLMs integrate visual and textual information across tokens, layers, and attention patterns [13, 17, 26, 29, 32, 44]. Head-level VLM studies have identified visual, OCR, and broad functional heads in single-image, rendered-text, or multimodal VQA settings [4, 8, 9, 16]. These works either focus on single-image or rendered-text settings, or assign broad functional labels rather than isolating the image-selection mechanism in natural multi-image contexts. We study natural multi-image retrieval, and our twosided analysis exposes a discrepancy that attention-based identification alone cannot capture: heads selected by the pointer criterion are not necessarily selected by the writer criterion.

Multi-image vision-language models and benchmarks. Recent LVLMs support interleaved text and multi-image inputs [3, 6, 21, 40], enabling evaluation over many images in context. Multi-image benchmarks such as Visual Haystacks and MM-NIAH show that performance degrades as the number of images grows [33, 34, 39, 41], suggesting that target-image grounding is a central bottleneck. However, they evaluate output behavior rather than the internal retrieval mechanism. Our work addresses this gap by providing a head-level mechanistic account of in-context image retrieval, using a controlled probe (§3.2) for discovery and Visual Haystacks for transfer evaluation (§4.3).

## 3 Setup

## 3.1 Preliminaries

In-context image retrieval. Consider an LVLM processing an interleaved image-text context

$$
C = ( \tau _ { 0 } , I _ { 1 } , \tau _ { 1 } , I _ { 2 } , \ldots , I _ { N } , \tau _ { N } , u ) ,
$$

where $I _ { 1 } , \ldots , I _ { N }$ are candidate images, $\tau _ { 0 } , \ldots , \tau _ { N }$ are intervening text passages, and u denotes taskspecific textual instructions, including the question and answer options when applicable. Answering an image-grounded question in this setting requires grounding the question in the target image before performing the downstream visual computation. We refer to this image-grounding component, which selects the relevant image within the context, as in-context image retrieval. This retrieval step gates multi-image inference: grounding the query in the wrong image conditions the answer on the wrong visual evidence. This paper studies the head-level mechanism underlying this retrieval step.

Attention notation. After tokenization, the interleaved context $C$ becomes a sequence of length T containing both visual and text tokens. Each image $I _ { i }$ occupies a contiguous visual-token span $[ s _ { i } , e _ { i } ] _ { : }$ , while different image spans are separated by text-token spans induced by $\tau _ { 0 } , \ldots , \tau _ { N }$ . For an LVLM with $L$ transformer blocks and H attention heads per block, we denote the post-softmax attention matrix of head $( l , h )$ by $A ^ { ( l , h ) } \in \mathbb { R } ^ { T \times T }$ . For a token position $p ,$ the attention mass assigned by head (l, h) to image i is

$$
m _ { i } ^ { ( l , h ) } ( p ) = \sum _ { j = s _ { i } } ^ { e _ { i } } A _ { p , j } ^ { ( l , h ) } .\tag{1}
$$

Both the input-side and output-side criteria define a per-head, per-sample score vector $\mathbf { \Psi } _ { s } ( l , h ) \mathbf { \Psi } \in \mathbb { R } ^ { N }$ over candidate images, and differ in how they compute it (§4.1 and §5.1). The score is computed at a task-specified query position $p ,$ which we instantiate for ICIR-MCQ in §3.2. For a sample with target image $i ^ { \star }$ , we define the head-level retrieval correctness as

$$
r ^ { ( l , h ) } = \mathbb { 1 } \left[ \operatorname * { a r g m a x } _ { i } s _ { i } ^ { ( l , h ) } = i ^ { \star } \right] .\tag{2}
$$

We report head-level R@1 by averaging this indicator over samples.

Models. We study four LVLMs spanning two model families and two parameter scales: InternVL3.5- {4B, 8B} [40] and Qwen3-VL-{4B, 8B} [6]. The two model families differ in vision encoder and training data, but the 4B variants share an LLM backbone, as do the 8B variants. This design lets us examine head-level stability across LLM backbones and, within shared backbones, head-index reproducibility. We choose these families because they are recent open-weight LVLMs that support interleaved multi-image inputs and allow head-level attention extraction and intervention. Exact architectural dimensions and implementation details are provided in Appendix A.

![](images/dc11e663959ca70a720217b6bd7ec96bdd0e10801b9d0cb9cba3dc2a3fd94c41.jpg)

![](images/1d6b4a504e5db561d47db9392625d3b3cd14da3dda6b9f81fd5b2bc04ecb784f.jpg)  
Figure 1: An ICIR-MCQ sample. The model selects the caption matching the image containing the queried category (zebra). At the cattok, the pointer criterion ranks images by projection-based attention mass (Eq. 3), while the writer criterion ranks images using a retriever trained on the head output. The two criteria can disagree in the same head.

## 3.2 ICIR-MCQ: A Controlled Probe for Head-Level Image Retrieval

We design ICIR-MCQ as a controlled probe for analyzing image retrieval at the attention head level. Each sample consists of an interleaved image-text context with N candidate images, a textual query specifying an object category, and four candidate captions. The model is asked to select the caption that best describes the image containing the queried category (Figure 1, top). The target image i<sup>⋆</sup> is the unique image in the context annotated with the queried category; in Figure 1, this is the second image, which contains a zebra and matches the caption (B).

ICIR-MCQ is constructed to satisfy three properties. First, the input is an interleaved image-text context, matching the general form of multi-image inputs to LVLMs. Second, the intervening text passages do not reveal the target image or the correct caption. Third, the candidate images are natural images, so head-level findings are not tied to a synthetic visual construction. To satisfy these requirements, we construct ICIR-MCQ from MS-COCO [24], inheriting the interleaved structure from MM-NIAH’s image-retrieval task [39] and substituting natural images for the original images.

More precisely, we keep the inherited text passages and reformulate the task as multiple-choice caption matching. Because the original MM-NIAH passages are kept fixed while the images and captions are newly substituted from COCO, the passages are independent of the target image and answer options by construction. We use COCO category annotations to control the per-image attribution target: the queried category is annotated in the target image but not in any distractor image, making the target image unambiguous given the category cue. Distractor captions are sampled from other COCO images containing the same queried category, so the model cannot solve the task from the category cue alone; it must ground the query in the correct image.

For head-level analysis, we score each head at the last token of the final mention of the queried category, which we call the cattok. This token is a query position at which the category cue has just appeared in text and all candidate image spans are in the causal past. In Figure 1, the cattok is the last token of zebra in option (D). Scoring at the cattok asks which image each head ranks highest when conditioned on the queried category.

The target image position and correct answer letter are balanced uniformly across options; further sampling and filtering details are in Appendix B. We compute head-level retrieval R@1 under each criterion’s score, defined in §4.1 and §5.1. ICIR-MCQ serves as our discovery probe and supplies the data for training and evaluating the head-specific image retrievers (§5.1).

Table 1: Top-5 pointer heads by head-level attention-mass R@1 on ICIR-MCQ. The strongest heads cluster in the same mid-block layer band, and within each LLM-backbone group (8B above; 4B below), top heads reproduce at identical (layer, head) indices. Cells show LxxHyy (R@1).
<table><tr><td>Model</td><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-4</td><td>Top-5</td><td>Median</td></tr><tr><td>InternVL3.5-8B</td><td>L21H10(.890)</td><td>L21H11 (.850)</td><td>L21H08 (.842)</td><td>L20H17(.825)</td><td>L21H09 (.823)</td><td>.206</td></tr><tr><td>Qwen3-VL-8B</td><td>L21H10(.922)</td><td>L21H08(.883)</td><td>L21H11 (.878)</td><td>L23H10(.873)</td><td>L21H09(.869)</td><td>.198</td></tr><tr><td>InternVL3.5-4B</td><td>L21H10(.866)</td><td>L21H11(.847)</td><td>L23H10 (.835)</td><td>L21H08(.826)</td><td>L21H09 (.824)</td><td>.217</td></tr><tr><td>Qwen3-VL-4B</td><td>L21H10(.905)</td><td>L23H10(.878)</td><td>L21H11(.870)</td><td>L21H08(.859)</td><td>L21H09(.856)</td><td>.206</td></tr></table>

## 4 Input-Side Retrieval: Pointer Heads

## 4.1 Pointer Criterion: Retrieval by Per-Image Attention Mass

A natural input-side criterion for image retrieval is whether a head attends to the target image. We instantiate this as the pointer criterion. For head (l, h) and the cattok position $p ,$ the score assigned to image i is the attention mass over its visual-token span:

$$
s _ { i } ^ { ( l , h ) , \mathrm { p t r } } = \sum _ { j = s _ { i } } ^ { e _ { i } } \tilde { A } _ { p , j } ^ { ( l , h ) } , \qquad \tilde { A } _ { p , j } ^ { ( l , h ) } = \mathrm { s o f t m a x } \Bigg ( \frac { \mathbf { q } _ { p } ^ { ( l , h ) } \cdot \mathbf { k } _ { j } ^ { ( l , h ) \top } } { \sqrt { d _ { h } } } \Bigg ) _ { j } .\tag{3}
$$

Here $\mathbf { q } _ { p } ^ { ( l , h ) }$ and $\mathbf { k } _ { j } ^ { ( l , h ) }$ are the direct outputs of the query and key projection layers and the softmax runs over all prefill positions $j ; { \tilde { A } }$ is a proxy for the forward-pass attention A. Plugging $s ^ { ( l , h ) , \mathrm { p t r } }$ into Eq. 2 yields the head’s R@1 under the pointer criterion. We call high-R@1 heads pointer heads: they rank the target image first by concentrating attention mass on its visual-token span. The criterion follows the attention-pattern convention used to identify retrieval heads in language-only models [42], adapted here from token-level lookup to span-level image retrieval.

## 4.2 Pointer Heads Are Sparse and Localized

Pointer ability is sharply non-uniform: the best pointer head reaches R@1 between .866 and .922 across the four LVLMs (Table 1; chance R@1 is .183), while the median head remains near chance. High-R@1 heads concentrate in a narrow mid-block layer band, with the same band visible on every LVLM; the per-model layer×head R@1 heatmap is reported in Appendix C (Figure 4). We denote by $\mathcal { H } _ { K } ^ { \mathrm { p t r } }$ the top-K heads under this pointer ranking. Despite differing in vision encoder and training data, LVLMs sharing an LLM backbone show sharp reproducibility at specific (layer, head) indices. The top-3 pointer heads coincide as a set within each backbone group, and 86 of the top-100 are shared between the 8B models, 80 between the 4B models. Per-head pointer R@1 grids and within-backbone consistency statistics are reported in Appendix C.

## 4.3 Pointer Heads Transfer Beyond the Controlled Probe

We next ask whether pointer heads discovered on ICIR-MCQ transfer to a naturalistic multi-image benchmark. We use Visual Haystacks [41]: each example contains multiple images and a Yes/No question with an anchor object that identifies the target image and a target object whose presence in it is queried. For example, a question may ask, “In the image that contains a sofa, is there a bottle?”, where sofa is the anchor object and bottle is the target object. For pointer scoring, we use the last token of the anchor-object phrase as the cattok, because this is where the model has enough information to determine which image should be retrieved. We evaluate at haystack size N=10, where N denotes the number of candidate images, using each head’s per-image attention mass to rank candidate images and measuring whether the top-ranked image is the target image.

Transfer from ICIR-MCQ to a naturalistic benchmark. For each LVLM, we use its top pointer head identified on ICIR-MCQ. Per-image retrieval R@1 ranges from .787 to .836 at haystack size $N { = } 1 0$ , far above the chance level of .1 for selecting among ten images. At this haystack size the 4B variants match or slightly exceed their 8B counterparts, suggesting that strong pointer signals are not confined to the larger models.

Table 2: Image retrieval R@1 on Visual Haystacks (haystack size N=10, chance R@1 is $^ { 1 } / ( 1 0 )$ Internal results use the single pointer head L21H10 in each LVLM; external retrievers are evaluated on the same samples. IV = InternVL3.5, QV = Qwen3-VL, and QV-Emb = Qwen3-VL-Embedding-8B.
<table><tr><td rowspan="2">Method</td><td colspan="4">Single Pointer Head</td><td colspan="3">Retriever Model</td></tr><tr><td>IV-8B</td><td>QV-8B</td><td>IV-4B</td><td>QV-4B</td><td>CLIP</td><td>SigLIP</td><td>QV-Emb</td></tr><tr><td>R@1</td><td>.787</td><td>.832</td><td>.794</td><td>.836</td><td>.573</td><td>.649</td><td>.751</td></tr></table>

A single pointer head is competitive with external retrievers. A single attention head from an LVLM not trained with an explicit retrieval objective achieves higher per-image retrieval R@1 (Table 2) than standard external vision-language retrievers—CLIP [31], SigLIP [43], and Qwen3- VL-Embedding-8B [23]. This comparison should not be read as a controlled training-objective comparison: the internal pointer head and the external retrievers operate on different representations and conditioning signals. Rather, it shows that the LVLM contains a highly effective contextconditioned image-selection signal in a single head. This advantage is consistent with the internal head’s access to richer signal: external retrievers compare global image embeddings with contextfree text embeddings, whereas the pointer head attends over patch-level keys in the LVLM’s own representational space using a query conditioned on the context through the cattok. Retriever-side work continues to improve both components [19, 30]. Additional retriever details are provided in Appendix A.

## 5 Output-Side Retrieval: Writer Heads

The pointer criterion asks where a head points, whereas downstream computation uses what it writes to the residual stream. At the cattok position $p ,$ the head output is

$$
\mathbf { o } _ { p } ^ { ( l , h ) } = \sum _ { j = 1 } ^ { T } A _ { p , j } ^ { ( l , h ) } \mathbf { v } _ { j } ^ { ( l , h ) } \in \mathbb { R } ^ { d _ { h } } , \qquad \mathbf { v } _ { j } ^ { ( l , h ) } = W _ { v } ^ { ( l , h ) } \mathbf { x } _ { j } ,\tag{4}
$$

where $\mathbf { x } _ { j }$ is the input to the value projection at position $j$ and $W _ { v } ^ { ( l , h ) }$ is the head’s value projection. A head’s value projection determines which information from the attended tokens enters the output: even when attention concentrates on the target image, there is no guarantee that the head output contains information sufficient to identify that image. We therefore introduce a complementary output-side criterion that asks whether the target image can be identified from the head output itself. This lets us compare attention-based pointing with output-side writing at the same head granularity.

## 5.1 Writer Criterion: Retrieval from Head Outputs

We introduce a retriever-based output-side head discovery method. For each head (l, h), we train a lightweight head-specific image retriever on the head output $\mathbf { o } _ { p } ^ { ( l , h ) }$ . Following the standard dualencoder retrieval formulation [18, 31], the retriever maps the head output and each candidate image embedding into a shared space and scores candidates by cosine similarity,

$$
s _ { i } ^ { ( l , h ) , \mathrm { w r i } } \ = \ \tilde { f } _ { \boldsymbol { \theta } _ { l , h } } \big ( \mathbf { o } _ { p } ^ { ( l , h ) } \big ) \cdot \tilde { g } _ { \boldsymbol { \phi } _ { l , h } } \big ( \mathbf { e } _ { i } \big ) ,\tag{5}
$$

where $\mathbf { e } _ { i }$ is obtained by mean-pooling candidate image $i \ ' _ { \mathrm { { s } } }$ projected visual tokens in the LVLM’s language-model embedding space, so the writer probe uses the LVLM’s own image representation rather than an external encoder. We train $f _ { \theta _ { l , h } }$ and $g _ { \phi _ { l , i } }$ jointly with InfoNCE [28] for each head.

We split ICIR-MCQ into disjoint train/evaluation subsets at the sample level. For writer-head discovery, we evaluate Eq. 5 over a held-out image pool of 1,200 candidates per LVLM and compute R@1 by whether the target image ranks first. We call this metric writer R@1, and call high-R@1 heads under this criterion writer heads. Operationally, a writer head is one whose output lets the trained retriever identify the target image, in the spirit of probing intermediate representations [7]. Whether these outputs are used by the model’s answer-generation pathway is tested separately in §6. Architecture details and hyperparameters are provided in Appendix D.

Table 3: Top-5 writer heads by writer R@1 on the ICIR-MCQ held-out image pool. Within each LLM-backbone group (8B above; 4B below), top heads largely overlap. Writer R@1 is evaluated on a held-out pool of 1,200 candidate images, whereas pointer R@1 in Table 1 is evaluated on the images in each context; the two R@1 are not directly comparable in absolute scale. Cells show LxxHyy (R@1).
<table><tr><td>Model</td><td>Top-1</td><td>Top-2</td><td>Top-3</td><td>Top-4</td><td>Top-5</td><td>Median</td></tr><tr><td>InternVL3.5-8B</td><td>L24H23 (.489)</td><td>L20H28 (.435)</td><td>L20H15 (.426)</td><td>L21H31(.421)</td><td>L18H19(.418)</td><td>.077</td></tr><tr><td>Qwen3-VL-8B</td><td>L24H23 (.595)</td><td>L21H17(.470)</td><td>L26H26 (.460)</td><td>L21H31(.459)</td><td>L20H28(.458)</td><td>.095</td></tr><tr><td>InternVL3.5-4B</td><td>L21H31(.472)</td><td>L20H28(.439)</td><td>L24H23(.433)</td><td>L20H15(.428)</td><td>L21H17(.428)</td><td>.080</td></tr><tr><td>Qwen3-VL-4B</td><td>L24H23(.509)</td><td>L20H28(.487)</td><td>L21H31 (.486)</td><td>L21H17(.483)</td><td>L18H19(.463)</td><td>.101</td></tr></table>

![](images/b7edfa6163c7726aa506c2c856a187cbedf97f9e5455fd66ccdcdef655d2646a.jpg)  
Figure 2: Head-level pointer R@1 versus writer R@1 on ICIR-MCQ. Each point is one attention head: colors denote pointer&writer, pointer-only, writer-only, and other heads, where membership is defined by the top-100 heads under each ranking. The vertical dotted line marks pointer-axis chance (R@1 = .183). The two criteria select substantially different head sets across all four LVLMs.

The writer criterion is parallel to the pointer criterion: both produce per-image scores and are evaluated by top-1 retrieval accuracy. However, pointer R@1 is computed over the images in each prompt, whereas writer R@1 is computed over a held-out image pool. We therefore use writer R@1 as an output-side ranking criterion, rather than comparing its absolute value directly against pointer R@1. We denote the top-K pointer and writer heads by $\mathcal { H } _ { K } ^ { \mathrm { p t r } }$ and $\mathcal { H } _ { K } ^ { \mathrm { w r i } }$

## 5.2 Writer Heads Encode Recoverable Target-Image Information

The trained retrievers give us a concrete way to test whether a head output carries recoverable target-image information. Table 3 shows that the strongest writer heads reach R@1 of .472–.595 on the held-out pool of 1,200 candidate images, compared with median values of .077–.101.

Writer ability is sparse and mid-block. Writer R@1 is sharply non-uniform across heads. The strongest writer heads concentrate in mid-block layers 18–27 (Appendix D). This suggests from the output side that image-retrieval-relevant signals are localized rather than uniformly distributed across the network. Per-head writer R@1 grids are provided in Appendix D.

Top writer heads reproduce within shared LLM backbones. As with pointer heads, writer heads reproduce within shared LLM backbones despite differences in vision encoder and training data. The 8B models share three of the top-5 writer heads and the 4B models share four, with three heads common to all four models (Table 3). Although writer heads occupy a similar mid-block region as pointer heads, their top-ranked identities differ, motivating a direct comparison between the two rankings.

## 5.3 Pointing Is Not Writing

Pointing and writing identify substantially different heads. Both rankings concentrate in the same mid-block region, yet at top-100 under each criterion, only 35–47 heads are selected by both.

![](images/750e4db0129daa397f2d14164728488d50aa1cc3eda2b345bfdba3ebf7173041.jpg)  
Figure 3: Set-partition ablation on Visual Haystacks. Each panel corresponds to a haystack size N, the number of candidate images. Curves show average accuracy drop relative to the unablated baseline as K heads are ablated. The pointer-writer overlap shows the most consistent effect. Shaded bands show ±1 standard deviation across the four LVLMs.

The off-diagonal heads in Figure 2 make the dissociation concrete. Pointer-only heads rank in the pointer top-100 but outside the writer top-100.

Conversely, writer-only heads in $\mathcal { H } _ { 1 0 0 } ^ { \mathrm { w r i } } \setminus \mathcal { H } _ { 1 0 0 } ^ { \mathrm { p t r } }$ score highly under the writer criterion despite not being among the strongest pointer heads. Thus, pointing and writing provide complementary, noninterchangeable rankings.

This dissociation has direct methodological implications: attention-based identification alone cannot distinguish heads that merely point to the target image from heads that also write target-image information into the residual stream. The remaining question is which population—pointer-only, writer-only, or their intersection—the model’s answer-generation pathway actually depends on. Section 6 addresses this with targeted set-partition ablations over these disjoint head sets.

## 6 Causal Validation: Retrieval Heads Are Pointers That Write

Sections 4 and 5 establish that pointing and writing identify substantially different heads, even though both criteria localize to the same mid-block region. The key question is which population the standard single-pass forward actually depends on. If the causally relevant retrieval heads are those that both point to the target image and write recoverable information into the residual stream, the dominant causal contribution should lie in the overlap between the two criteria. We test this hypothesis with targeted head ablations on Visual Haystacks across multiple haystack sizes, where each example has a ground-truth target image while the model still performs standard multi-image question answering.

## 6.1 Ablation Setup

Intervention. For each selected head (l, h), we zero the head’s output contribution before the output projection, thereby removing its additive write to the residual stream. We use this output-side ablation because it is a stricter test of a head’s causal contribution than query-side masking used in prior retrieval-head analysis [42]: query masking perturbs where a head attends but can still leave a valueweighted output, whereas our intervention removes the selected head’s write itself. This output-side, head-level zero-ablation is in line with the broader causal-mediation and activation-patching literature for localizing functional roles in transformers [25, 36, 38].

We apply this intervention only over the image-retrieval span of the question. Following the Visual Haystacks format in Section 4.3, this span starts at the first token of the anchor object and runs to the last prefill position (the target object ends the question, so only a few format tokens follow it). The intervention is not applied during the answer generation.

Set partition. Directly comparing the pointer top-K and writer top-K sets would confound causal contribution with their partial overlap. We therefore form the union of the top-100 pointer heads and top-100 writer heads, and partition it into three disjoint subsets. The pointer-writer overlap contains heads selected by both criteria. The pointer-only tail contains heads in the pointer top-100 but not the writer top-100. The writer-only tail contains heads in the writer top-100 but not the pointer top-100. Across models, the pointer-writer overlap contains 35–47 heads, and each single-criterion tail contains 53 to 65 heads. We ablate the top-K heads from each subset for $K \in \{ 5 , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ with matched-budget random-head controls at the same budgets. We evaluate at haystack sizes $N \in \{ 2 , 3 , 5 , 1 0 , 2 \bar { 0 } , 5 0 \}$ . As N increases, baseline accuracy approaches the Yes/No chance level, reducing the dynamic range available for additional accuracy drops under ablation.

## 6.2 The Pointer-Writer Overlap Carries the Causal Effect

Ablating the pointer-writer overlap produces the most consistent accuracy drops among the tested head populations (Figure 3). The effect generally grows with the ablation budget and is most visible at haystack sizes 2–20, where the pointer-writer curve separates from both single-criterion tails and matched random-head controls. At larger haystack sizes, the unablated baseline is already closer to the Yes/No chance level, leaving less room for further accuracy drops, so ablation effects are smaller. Writer-only ablations show little consistent separation from random controls, suggesting that output-side recoverability alone does not identify the heads most relevant to standard inference in this intervention window. Pointer-only ablations have a modest and less consistent effect, but remain weaker than the pointer-writer overlap in the averaged results for $K \geq 1 0$ . Overall, the strongest causal evidence points to heads selected by both criteria rather than to heads selected by either criterion alone. Per-model and per-haystack-size results are provided in Appendix I.

## 6.3 What the Intervention Isolates

The separation from matched random controls, especially at moderate haystack sizes, suggests that the drop is not merely due to removing heads in this sequence region or reducing head capacity in general. Rather, standard multi-image inference is most sensitive to the particular heads that both point to the target image and write recoverable target-image information.

Together, the pointer criterion reveals strong internal image-selection signals, while the writer criterion shows that attention mass alone does not fully characterize them. Targeted ablations further show that standard single-pass inference is most sensitive to heads selected by both criteria.

## 7 Discussion

Why the pointer-writer overlap matters. Pointing and writing are controlled by different parts of the attention: Q-K interactions determine where a head points, whereas the value pathway determines what information is written to the residual stream.

Our set-partition ablations show that standard multi-image inference is most sensitive to heads where both properties coincide. Thus, image retrieval in LVLMs is not fully characterized by where a head attends; the relevant signal must also be written in a form available to downstream computation.

From analysis to intervention. Our results suggest that interventions based only on amplifying attention to the target image may be insufficient when the selected heads do not write target-image information. A possible future direction is to modify the value-side pathway of heads that already point well, for example by retraining or steering only a small set of value/output projections while leaving the rest of the model unchanged. Such targeted interventions could test whether improving the writer side of existing pointer heads can improve multi-image inference without full-model retraining.

Limitations. Our analysis focuses on four recent open-weight LVLMs from two model families, and broader model coverage would further clarify how general the pointer-writer structure is. Our naturalistic evaluation isolates single-target image retrieval in Visual Haystacks; extending the framework to cross-image comparison, multi-target grounding, and compositional reasoning is a natural direction for future work. Complementary generation-time interventions could further characterize how retrieved visual information is converted into the final textual answer.

## 8 Conclusion

We studied in-context image retrieval in LVLMs at the attention head level. Using ICIR-MCQ, we identified sparse pointer heads whose per-image attention mass ranks the target image first. These heads localize to a consistent mid-block region and transfer beyond the controlled probe, providing a strong input-side retrieval signal. However, attention mass alone is incomplete: a head selected by the pointer criterion is not necessarily selected by the writer criterion. We therefore introduced a retriever-based output-side writer criterion and found that pointer and writer rankings identify related but distinct head sets. Targeted set-partition ablations show that standard single-pass inference is most sensitive to heads selected by both criteria rather than to either single-criterion tail. Together, these results support a two-sided account of image retrieval in LVLMs: our strongest causal evidence favors heads selected by both the attention-based pointer and output-side writer criteria.

## Acknowledgments and Disclosure of Funding

This work was supported by the National Research Foundation of Korea (NRF) grants funded by the Korea government (MSIT) (Nos. 2022R1A3B1077720, 2022R1A5A7083908, and RS-2026- 25488668); by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grants funded by the Korea government (MSIT) (No. RS-2021-II211343, Artificial Intelligence Graduate School Program, Seoul National University; No. RS-2022-II220959; No. IITP-2026-RS-2024-00397085, Leading Generative AI Human Resources Development; and No. IITP-2026-RS-2020-II201819, IITP-ICT Creative Consilience Program); and by the BK21 FOUR program of the Education and Research Program for Future ICT Pioneers, Seoul National University in 2026. This research was also supported by the ANCHOR through the Seoul ANCHOR Center, funded by the Ministry of Education (MOE) and the Seoul Metropolitan Government (2026-ANCHOR-01-003-09), and by Samsung Electronics Co., Ltd (IO260506-16457-01).

## References

[1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, et al. Pixtral 12B. arXiv preprint arXiv:2410.07073, 2024.

[2] Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebrón, and Sumit Sanghai. GQA: Training generalized multi-query transformer models from multi-head checkpoints. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4895–4901, 2023.

[3] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35: 23716–23736, 2022.

[4] Ingeol Baek, Hwan Chang, Sunghyun Ryu, and Hwanhee Lee. How do large vision-language models see text in image? unveiling the distinctive role of OCR heads. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 20452–20464, 2025.

[5] Kanghyun Baek, Jaihyun Lew, Chaehun Shin, Jungbeom Lee, and Sungroh Yoon. Diagnosing and correcting concept omission in multimodal diffusion transformers. In International Conference on Machine Learning, volume 306, pages 5107–5124, 2026.

[6] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

[7] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.

[8] Jing Bi, Junjia Guo, Yunlong Tang, Lianggong Bruce Wen, Zhang Liu, Bingjie Wang, and Chenliang Xu. Unveiling visual perception in language models: An attention head analysis approach. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 4135–4144, 2025.

[9] Miso Choi, Seonga Choi, Mincheol Kwon, Woosung Joung, Jinkyu Kim, and Jungbeom Lee. The truth stays in the family: Enhancing contextual truthfulness via inherited heads in model lineages. In International Conference on Machine Learning, volume 306, pages 20021–20044, 2026.

[10] Kevin Clark, Urvashi Khandelwal, Omer Levy, and Christopher D Manning. What does BERT look at? an analysis of BERT’s attention. In Proceedings of the 2019 ACL workshop BlackboxNLP: analyzing and interpreting neural networks for NLP, pages 276–286, 2019.

[11] Arthur Conmy, Augustine Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adrià Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

[12] Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations, volume 2024, pages 35549–35562, 2024.

[13] Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. In International Conference on Learning Representations, volume 2024, pages 2632–2652, 2024.

[14] Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 2021.

[15] Mor Geva, Jasmijn Bastings, Katja Filippova, and Amir Globerson. Dissecting recall of factual associations in auto-regressive language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12216–12235, 2023.

[16] Yanbei Jiang, Xueqi Ma, Shu Liu, Sarah Monazam Erfani, Tongliang Liu, James Bailey, Jey Han Lau, and Krista A Ehinger. Investigating the functional roles of attention heads in vision language models: Evidence for reasoning modules. arXiv preprint arXiv:2512.10300, 2025.

[17] Seil Kang, Jinyeong Kim, Junhyeok Kim, and Seong Jae Hwang. See what you are told: Visual attention sink in large multimodal models. In International Conference on Learning Representations, volume 2025, pages 87676–87703, 2025.

[18] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.

[19] Yongjoo Kim, Mincheol Kwon, Seonga Choi, Minseung Lee, Kyeong-Jin Oh, Hyunyoung Lee, Yunsu Choi, and Jungbeom Lee. SAGE: Semantic attribute graphs for multi-entity visual retrieval. In Proceedings ofthe 2026 Conference on Empirical Methods in Natural Language Processing, 2026.

[20] Goro Kobayashi, Tatsuki Kuribayashi, Sho Yokoi, and Kentaro Inui. Attention is not only a weight: Analyzing transformers with vector norms. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7057–7075, 2020.

[21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. LLaVA-OneVision: Easy visual task transfer. Transactions on Machine Learning Research, 2025.

[22] Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023.

[23] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.

[24] Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft COCO: Common objects in context. In European conference on computer vision, pages 740–755. Springer, 2014.

[25] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. Advances in neural information processing systems, 35:17359–17372, 2022.

[26] Clement Neo, Luke Ong, Philip Torr, Mor Geva, David Krueger, and Fazl Barez. Towards interpreting visual information processing in vision-language models. In International Conference on Learning Representations, volume 2025, pages 57172–57189, 2025.

[27] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.

[28] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

[29] Vedant Palit, Rohan Pandey, Aryaman Arora, and Paul Pu Liang. Towards vision-language mechanistic interpretability: A causal tracing tool for BLIP. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 2856–2861, 2023.

[30] Junsung Park, Jungbeom Lee, Jongyoon Song, Sangwon Yu, Dahuin Jung, and Sungroh Yoon. Know “no” better: A data-driven approach for enhancing negation awareness in CLIP. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 2825–2835, 2025.

[31] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PMLR, 2021.

[32] Sarah Schwettmann, Neil Chowdhury, Samuel Klein, David Bau, and Antonio Torralba. Multi modal neurons in pretrained text-only transformers. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV) Workshops, pages 2862–2867, 2023.

[33] Aditya Sharma, Michael Saxon, and William Yang Wang. Losing visual needles in image haystacks: Vision language models are easily distracted in short and long contexts. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 5429–5451, 2024.

[34] Xinyu Tian, Shu Zou, Zhaoyuan Yang, and Jing Zhang. Identifying and mitigating position bias of multi-image vision-language models. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 10599–10609, 2025.

[35] Eric Todd, Millicent Li, Arnab Sen Sharma, Aaron Mueller, Byron Wallace, and David Bau. Function vectors in large language models. In International conference on learning representations, volume 2024, pages 17282–17333, 2024.

[36] Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. Advances in neural information processing systems, 33:12388–12401, 2020.

[37] Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th annual meeting of the association for computational linguistics, pages 5797–5808, 2019.

[38] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In International Conference on Learning Representations, 2023.

[39] Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, et al. Needle in a multimodal haystack. Advances in Neural Information Processing Systems, 37:20540–20565, 2024.

[40] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. InternVL3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025.

[41] Tsung-Han Wu, Giscard Biamby, Jerome Quenum, Ritwik Gupta, Joseph E Gonzalez, Trevor Darrell, and David M Chan. Visual haystacks: A vision-centric needle-in-a-haystack benchmark. In International Conference on Learning Representations, volume 2025, pages 64592–64615, 2025.

[42] Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In International Conference on Learning Representations, volume 2025, pages 62143–62156, 2025.

[43] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023.

[44] Zhi Zhang, Srishti Yadav, Fengze Han, and Ekaterina Shutova. Cross-modal information flow in multimodal large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19781–19791, 2025.

## A Implementation Details

Model shape. The four LVLMs share the same language-model shape: 36 transformer blocks with 32 attention heads each, L · H = 1,152 heads in total, and head dimension $d _ { h } = 1 2 8$ . Each model is loaded from its HuggingFace snapshot in bfloat16 and run with FlashAttention-2 [12].

Attention score and head-output capture. We register forward hooks on q\_proj and k\_proj of each transformer block to capture the query and key projections needed to recompute per-head attention scores during prefill, and a forward-pre-hook on o\_proj to capture each head’s output (the value-weighted sum entering the output projection) for the writer probe. For models with Grouped Query Attention [2], we expand KV heads to match the query heads before computing per-head attention. The pointer score (Eq. 3) is computed from these captured projections at the designated readout row only; the forward pass itself is unchanged.

Visual Haystacks setup. For each haystack size, a fixed random seed (42) controls filler-image selection and haystack shuffling, so that every model sees the same haystack composition and order.

Image preprocessing and per-image input budget. We fix the per-image input budget at 448×448 pixels for all four LVLMs we study so that cross-model comparisons are not confounded by input size. InternVL3.5 uses a 448 × 448 resize (BICUBIC) followed by ImageNet normalization, yielding 256 visual tokens per image (the architecture’s native per-tile budget). Qwen3-VL uses an aspectratio-preserving resize within a configurable pixel budget by default; we override the processor with min\_pixels = max\_pixels = 448 × 448, which fixes the pixel budget rather than the resolution: the aspect ratio is preserved and each side is rounded to a multiple of 32, giving roughly 192 visual tokens per image, about three quarters of InternVL3.5’s 256. The same transforms are used in baseline and ablation runs, so within-model comparisons are unaffected by the token-count difference between the two families.

Generation configuration. All forward passes run in bfloat16. Decoding is greedy (do\_sample=False, num\_beams=1) with max\_new\_tokens=128, so any randomness in answers is removed. For VHs (Yes/No format), generation stops at the first “yes” or “no” word, with no effect on parsed accuracy.

External retriever evaluation. We evaluate CLIP ViT-L/14, SigLIP so400m-patch14-384, and Qwen3-VL-Embedding-8B as external retrievers on Visual Haystacks. For CLIP and SigLIP, we follow the standard OpenAI 80-template prompt ensemble [31]: the text embedding for the anchor object is the mean of the ℓ -normalized embeddings of the 80 templated prompts, and the per-image score is the cosine similarity between this ensembled text embedding and the image embedding. The anchor noun is inserted without its article, and SigLIP prompts are padded to the 64-token maximum length used in its pretraining, which its text encoder requires. For Qwen3-VL-Embedding-8B we use a natural retrieval phrasing query, “an image with {anchor}”, and score images by cosine similarity to this query embedding. In all three retrievers we select the argmax image as the predicted target.

Ablation procedure. For the targeted ablation experiments (§6), a forward pre-hook on o\_proj zeroes the selected head’s slice of the o\_proj input over the image-retrieval span: from the first token of the anchor-object phrase to the last prefill position (the target-object phrase ends the question, so the remaining covered tokens are the fixed answer-format suffix). The first answer token is decoded from the last prefill position, where the ablation is already in effect (Appendix I.1 checks that continuing the intervention through generation does not change the conclusions). Random selection draws K heads uniformly from the 1,152 heads at the matched budget (averaged over three seeds in the set-partition ablation of §6.1); the unablated baseline runs the same pipeline with no head zeroed.

Computational resources. All main-text experiments run on NVIDIA L40 GPUs (48GB VRAM). Single-GPU inference suffices for all four LVLMs at the haystack sizes evaluated in this paper $( N ~ \mathbf { \bar { \in } } ~ \{ 2 , 3 , 5 , 1 0 , 2 0 , 5 0 \} )$ ). A Visual Haystacks pass at N=10 takes about 1.5 s per sample for InternVL3.5-8B and about 4 s for Qwen3-VL-8B, whose answers are longer, and head-output extraction on ICIR-MCQ about 13 s per sample; the set-partition grid of §6 comprises about 220 cells of 1,000 samples per LVLM, on the order of a few hundred GPU-hours per LVLM.

"You said an interesting word, ’rumor.’ I’ve been in the business [...]   
<image 2>

## B ICIR-MCQ: Additional Construction Details

ICIR-MCQ is described in the main text (§3.2) and used as the head-discovery benchmark throughout §4 and §5. This appendix gives the design rationale, the sampling procedure, the split sizes, and one abridged prompt.

Why this design (extended). Three properties matter for head-level attribution. First, the answer is a single discrete token whose correctness is unambiguous. Second, all four caption options mention the queried category (the three distractor options are captions of other images of that category), so the answer cannot be identified from the category cue alone; the distractor images are drawn from other categories, so the target image is unambiguous given the cue. Third, the small candidate set $( N \leq 6 )$ keeps the head-level R@1 estimator stable at the 6,000-sample scale, allowing a full sweep of all $L \cdot H { = } 1 , 1 5 2$ heads without budgeted subsetting.

Sampling procedure. We draw each sample by (i) selecting a target image and one of its own COCO captions that mentions the queried category (sampling from the union of the MS-COCO 2017 train and validation splits), (ii) selecting $N _ { \mathrm { c h o i c e s } } - 1$ distractor images from other COCO categories, excluding any image annotated with the target’s category, so that the queried category is annotated in the target image alone, and (iii) constructing a four-way multiple choice over caption options A–D, in which one option is the target’s own caption and the other three are captions of other images of the queried category, so that all four options mention the category. We constrain (a) the target image’s position among the $N _ { \mathrm { c h o i c e s } }$ images to be uniform across the $N _ { \mathrm { c h o i c e s } }$ slots, and (b) the answer letter (the slot of the target’s caption among the four options A–D) to be uniformly distributed across $\left\{ \mathrm { A } . \right.$ B, C, D}. The number of MCQ option letters (always four) is independent of $N _ { \mathrm { c h o i c e s } } ;$ the candidate images themselves are unlabeled and identified by their position in the prompt rather than by a letter.

Split sizes. We release two splits: $N _ { \mathrm { c h o i c e s } } { = } 5$ with 3,000 samples and $N _ { \mathrm { c h o i c e s } } { = } 6$ with 3,000 samples, for a total of 6,000.

The released file, its data card, and the builder are available at https://huggingface.co/ datasets/jerryray/ICIR-MCQ; the analysis code is at https://github.com/parkquasar/ Image-Retrieval-Heads.

Robustness of head rankings. The top head is invariant between the two $N _ { \mathrm { c h o i c e s } }$ splits for both 8B models (L21H10 in both cases), and the top-5 sets coincide across the splits, with only the ordering of ranks 2–5 differing.

## B.1 An abridged ICIR-MCQ prompt

Figure 1 is a shortened schematic of another sample of the same category. Below is the prompt of one actual sample (sample 4860 of the released file; six candidate images), with each text passage cut to its first sentence or so. Passages are the interleaved documents inherited from MM-NIAH (§3.2) and are unrelated to the images; the images are inserted at the <image> markers in order. The category is named in the question and again in each of the four options, which describe different images of that category, so the last mention of the category (the cattok) falls inside the options (option D here).

What the real geeks soon realize, however, is that it’s fun to brew with the   
Aeropress. You can literally try thousands of different recipes and still   
[...]

Here is the equivalent circuit diagram:The below code will demonstrate how it works by turning the motor on and off. [...]

The below circuit can be used to demonstrate. Note that there are no external components at all. [...]

It includes one one-bedroom unit, and 11 two-bedroom units. The property also has parking, controlled access entry and a laundry facility. [...]

<image 5>

Weirick will specialize in working with landlords and tenants regarding their industrial real estate needs in the San Fernando Valley. [...] <image 6>

CDC’s latest plan, which was accepted, was to remove the toxic soil and pile it up on three separate berms on the property. [...]

Which caption best describes the image above that contains zebra? Answer with a single letter (A, B, C, or D).

A) A group of zebras and wildebeest grazing in a field.

B) The two zebra stand next to each other in the wilderness.

C) A group of zebras and other animals grazing in a field with a rainbow in the background.

D) The zebra eats the grass near a building.

Target image: position 4 of 6. Correct option: A.

## C Pointer Heads: Extended Results

This appendix collects pointer-head results that are summarized in the main text: the per-LVLM head R@1 grid, pairwise ranking similarity across the four LVLMs, the per-haystack R@1 of the top pointer head on Visual Haystacks, and that head’s R@1 next to the model’s own accuracy.

## C.1 Per-LVLM head R@1 grid

Figure 4 shows the pointer R@1 of every head in each LVLM.

![](images/290a3fd3b6319d32e229e9e57d4672b7c5f991d7b4706299d1cbde18ed5594af.jpg)

![](images/15b6a899d49bc760df8ee13f6696232cdf575bc219771711c3fa6e503b589194.jpg)

![](images/7c5dacc3744c8e6494d258bc4c54fb494638bdb9a60083216d23900185fc71b1.jpg)

![](images/a2d8c53fea4da6ddd28833287e199cc4d2b6582f7022ab2361b61974c58cb143.jpg)  
Figure 4: Head-level attention-mass R@1 on ICIR-MCQ for all four LVLMs, shown as layers × heads with a shared color scale. Pointer ability concentrates in a narrow mid-block layer band, and the same band appears across both model families and both parameter scales. The strongest pointer head per model is L21H10 on every panel (Table 1).

## C.2 Pairwise ranking similarity across the four LVLMs

Table 4 reports pairwise statistics on the per-head R@1 ranking across the four LVLMs. For every pair of models it reports four statistics on the per-head R@1 ranking, defined in the caption. Pairs are grouped into shared-LM-backbone (same LM scale across the two families) and cross-LM-backbone (different LM scale).

Rankings are highly aligned for every pair (Spearman ≥ 0.90, top-100 overlap ≥ 0.75); the top-3 set is identical within each shared-LM-backbone pair and shares 2 of 3 heads across backbones.

## C.3 Per-haystack R@1 decay

Table 5 reports the per-haystack R@1 of L21H10, the top ICIR-MCQ pointer head on every model, on Visual Haystacks. Ranked by VHs R@1 at N=10, it is also the top head on both 8B models and ranks second on InternVL3.5-4B and third on Qwen3-VL-4B. R@1 generally falls with haystack size, and the two scales within a family are not consistently ordered: the InternVL3.5 pair stays within 0.8 pp at every haystack size, while Qwen3-VL-4B trails Qwen3-VL-8B by up to 6.5 pp at $N \in \{ 2 , 3 , 2 0 \}$ and matches it at $N \in \{ 5 , 1 0 \}$ . The pointer head is strong at both scales, in line with the cross-scale stability noted in §4.2, but R@1 itself is not ordered by model scale.

Table 4: Pairwise per-head R@1 similarity across the four LVLMs. Pearson and Spearman are computed on the full 1,152-d head ranking; layer-max correlation is on the per-layer maximum R@1 vector (length 36); top-K overlap is $\vert \mathcal { H } _ { K } ^ { \mathrm { p t r } }$ (model 1) $\cap \mathcal { H } _ { K } ^ { \mathrm { p t r } }$ (model $2 ) | / K$ . The top-1 pointer head is L21H10 on every model.
<table><tr><td></td><td></td><td>Pearson Spearman LayerMax Top-3 Top-10 Top-100</td><td></td><td></td><td></td><td></td></tr><tr><td>Shared-LM-backbone pairs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { I n t e r n V L 3 . 5 - 8 B  Q w e n 3 - V L - 8 B }$ </td><td>0.965</td><td>0.922</td><td>0.921</td><td>1.00</td><td>0.70</td><td>0.86</td></tr><tr><td> $\mathrm { I n t e r n V L 3 . 5 \mathrm { - } 4 B  Q w e n 3 \mathrm { - } V L \mathrm { - } 4 B }$ </td><td>0.966</td><td>0.927</td><td>0.933</td><td>1.00</td><td>0.60</td><td>0.80</td></tr><tr><td>Cross-LM-backbone pairs</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { I n t e r n V L 3 . 5 { \cdot } 8 B  \tilde { I n t e r n V L 3 . 5 { \cdot } 4 B } }$ </td><td>0.977</td><td>0.946</td><td>0.962</td><td>0.67</td><td>0.70</td><td>0.85</td></tr><tr><td> $\mathrm { Q w e n 3 – V L - 8 B  Q w e n 3 - V L - 4 B }$ </td><td>0.971</td><td>0.939</td><td>0.980</td><td>0.67</td><td>0.60</td><td>0.85</td></tr><tr><td>InternVL3.5-8B ↔ Qwen3-VL-4B</td><td>0.964</td><td>0.916</td><td>0.919</td><td>0.67</td><td>0.70</td><td>0.82</td></tr><tr><td> $\mathrm { I n t e r n V L 3 . 5 \mathrm { - } 4 B  Q w e n 3 \mathrm { - } V L \mathrm { - } 8 B }$ </td><td>0.937</td><td>0.900</td><td>0.932</td><td>0.67</td><td>0.70</td><td>0.75</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 5: R@1 of the shared top pointer head L21H10 on VHs across haystack sizes $\equiv \{ 2 , 3 , 5 , 1 0 , 2 0 \}$ (n=1,000 per cell). Cattok-position attention argmax accuracy. $N { = } 5 0 ^ { \circ }$ is not included in this readout.
<table><tr><td>Haystack size</td><td>2</td><td>3</td><td>5</td><td>10</td><td>20</td></tr><tr><td>InternVL3.5-8B</td><td>.947</td><td>.914</td><td>.860</td><td>.787</td><td>.677</td></tr><tr><td>Qwen3-VL-8B</td><td>.959</td><td>.931</td><td>.901</td><td>.832</td><td>.782</td></tr><tr><td>InternVL3.5-4B</td><td>.939</td><td>.907</td><td>.859</td><td>.794</td><td>.682</td></tr><tr><td>Qwen3-VL-4B</td><td>.894</td><td>.901</td><td>.895</td><td>.836</td><td>.745</td></tr></table>

## C.4 Single-head retrieval versus the model’s own accuracy

Table 6 places the top pointer head’s retrieval R@1 next to the accuracy of the unmodified model on the same inputs, for both benchmarks. On ICIR-MCQ the model answers with a letter and we score the restricted argmax over the four option tokens at the first answer position, on all $6 { , } 0 0 0$ samples. On Visual Haystacks $( N = 1 0 )$ accuracy is the Yes/No accuracy of the unablated baseline used in §6. On ICIR-MCQ the head’s R@1 and the model’s accuracy are within a few points of each other, whereas on Visual Haystacks the single head identifies the target image 11 to 16 points more often than the model answers the question correctly.

Table 6: Top pointer head (L21H10) retrieval R@1 and base-model accuracy on ICIR-MCQ (6,000 samples) and Visual Haystacks $( N = 1 0 ,$ , 1,000 samples).
<table><tr><td rowspan="2">Model</td><td colspan="2">ICIR-MCQ</td><td colspan="2">Visual Haystacks</td></tr><tr><td>Head R@1</td><td>Model acc. (%)</td><td>Head R@1</td><td>Model acc. (%)</td></tr><tr><td>InternVL3.5-4B</td><td>.866</td><td>85.5</td><td>.794</td><td>68.1</td></tr><tr><td>InternVL3.5-8B</td><td>.890</td><td>88.9</td><td>.787</td><td>65.9</td></tr><tr><td>Qwen3-VL-4B</td><td>.905</td><td>90.1</td><td>.836</td><td>67.5</td></tr><tr><td>Qwen3-VL-8B</td><td>.922</td><td>91.3</td><td>.832</td><td>67.6</td></tr></table>

## D Writer probe: training details and extended agreement analysis

This appendix gives the training details of the writer probe (§5) and compares the writer and pointer rankings; Appendices E and G report control experiments on the candidate pool, on object regions, and on the readout position.

Head-level training setup. For each attention head, we treat its attention-value-weighted output at the query position as a $d _ { h } .$ -dimensional image embedding and train a small head-level projection with a contrastive (InfoNCE-style) objective using ICIR-MCQ as positive-negative pairs. Reading head-level representations through a small trained probe follows the inference-time-intervention [22] and function-vector [35] setups; we adapt the construction to a cross-modal retrieval objective. The

Head

probe projects the head’s output and the candidate image embedding into a shared 256-dimensional space through one linear layer each, followed by L2 normalization, and scores candidates by dot product. We selected this configuration (one projector layer, L2 normalization, batch 256, learning rate $\bar { 1 } \times 1 0 ^ { - 3 } )$ from a sweep over architecture, projector depth, batch size, learning rate, and normalization on the 1,200-example evaluation split used for checkpoint selection and the reported ranking; the Qwen3-VL-8B probe uses batch 128. The LVLM weights are not modified; only the head-level probe is trained.

Agreement with the pointer ranking. The two criteria both concentrate in mid-block layers, but not on the same layers: the pointer ranking peaks at layer 21 on all four LVLMs, whereas the strongest writer heads sit at layer 24 on three of the four (layer 21 on InternVL3.5-4B). The writer top-100 is also more layer-dispersed than the pointer top-100 (47–60 versus 71–82 of 100 heads inside layers 18–27). The two rankings likewise disagree on which specific heads are top-ranked; the writer top-5 per LVLM are reported in Table 3 and do not match the pointer top-5 of Table 1. Figure 5 shows the full per-head writer R@1 grid, mirroring the pointer-side grid in Figure 4: the bright band sits in the same mid-block region in both rankings, with the writer band extending slightly further toward later layers. Ranks diverge more in the mid-tail than at the top.

![](images/9eb6b16dc4f86a8853546a4e2a55b86b8cc7068841ae1babb38548452b0faebb.jpg)

![](images/79ba4bde808722bdf9cb51b41569fbd1e88d64122636dc44f376756ccb6236b3.jpg)

![](images/95aec27dead1b86f0dc4004001737b61bce5bad11577a5c4bca63e1a44dd0ed6.jpg)

![](images/1b5d924c009a6106f70810cba34312fdaa7346927b367e0463cbc6bb6882dc6b.jpg)  
Figure 5: Head-level writer R@1 on the ICIR-MCQ held-out evaluation pool (1,200 candidate images; chance $\mathbb { R } \mathscr { Q } 1 \approx . 0 0 1 ,$ ) for all four LVLMs, shown as layers × heads with a shared color scale. Writer R@1 concentrates in a narrow mid-block layer band, the same region picked out by the pointer ranking (Figure 4).

Writer-ranked vs. pointer-ranked ablation. Whether the heads that only one criterion selects are causally used is tested by the set-partition ablation (§6.1, Appendix I): at $K \geq 1 0$ the overlap of the two top-100 sets produces the largest drops on average across models, the writer-only tail leaves accuracy near the random control on all four LVLMs, and the pointer-only tail is model-dependent (on InternVL3.5-8B it drops as much as or more than the overlap at $N \geq 5 , K \geq 2 5 ;$ ; near the control on Qwen3-VL-8B).

## E Writer Criterion: Object-Region and In-Context Controls

## E.1 Object regions instead of full images

The writer probe (§5) retrieves the embedding of the whole target image, so a head could in principle rank well by carrying global image identity rather than information about the queried object. We therefore retrained the per-head retriever with the target represented by the embedding of the object region, the COCO box of the queried category inside the target image, with the probe configuration of Appendix D. As a control for training variance we also retrained the full-image probe once more under the same settings (matched control). Table 7 compares the resulting head rankings with the full-image writer ranking of the main text, and reports a same-category retrieval test in which the candidate pool contains only object regions of the queried category taken from other images, so that category identity alone cannot solve the task; for a query with $m _ { i }$ same-category distractor regions chance is $1 / ( m _ { i } + 1 )$ , averaged over queries.

The object-region ranking agrees with the full-image writer ranking at $\rho = 0 . 9 7 0 \mathrm { t o } 0 . 9 7 1$ , somewhat below the agreement between two full-image trainings $( \rho \ge 0 . 9 9 3 , 9 3$ to 96 shared top-100 heads), so object information changes the ranking by more than training noise but leaves it largely intact. On the same-category test the top-100 object-region heads retrieve the correct region in 42 to 48% of queries against a chance of 6 to 9%, and the top-10 heads in 58 to 63%. Writer heads therefore carry information about the queried object itself, not only about which image it came from.

Table 7: Object-region writer probes. Spearman $\rho$ of the object-region head ranking against the fullimage writer ranking (all 1,152 heads); $\rho$ and top-100 overlap of a retrained full-image control against the same ranking; same-category retrieval with the mean number m of same-category distractor regions, the per-query chance $1 / ( m _ { i } + 1 )$ averaged over queries, and mean R@1 over the top-100 object-region heads and over the top-10.
<table><tr><td rowspan="2">Model</td><td colspan="2">Ranking agreement</td><td rowspan="2"></td><td colspan="3">Same-category retrieval</td></tr><tr><td>object vs. full</td><td>control ρ / overlap</td><td>m</td><td>chance</td><td>top-100 R@1 top-10 R@1</td></tr><tr><td>InternVL3.5-4B</td><td>0.970</td><td>0.993 / 93</td><td>12.1</td><td>0.091</td><td>0.466</td><td>0.592</td></tr><tr><td>InternVL3.5-8B</td><td>0.970</td><td>0.993 / 96</td><td>14.2</td><td>0.077</td><td>0.424</td><td>0.578</td></tr><tr><td>Qwen3-VL-4B</td><td>0.971</td><td>0.996 / 96</td><td>16.4</td><td>0.064</td><td>0.485</td><td>0.633</td></tr><tr><td>Qwen3-VL-8B</td><td>0.970</td><td>0.996 / 96</td><td>16.4</td><td>0.064</td><td>0.429</td><td>0.590</td></tr></table>

## E.2 In-context candidates instead of a held-out pool

The writer criterion scores each head against a held-out pool of 1,200 candidate images. A natural alternative scores it only against the images of the same context, five or six candidates, which is closer to the retrieval the model actually performs. Table 8 compares this in-context ranking with the held-out writer ranking and with the pointer ranking, using a paired head bootstrap (5,000 resamples) for the difference of the two correlations, and a split-half diagnostic in which the test samples are divided into two halves, heads are ranked on each half, and we count the top-100 heads shared by the two halves.

Table 8: In-context writer ranking versus the held-out writer and pointer rankings (Spearman $\rho$ over 1,152 heads; $\Delta = \rho _ { \mathrm { w r i t e r } } - \rho _ { \mathrm { p o i n t e r } }$ with its 95% bootstrap CI), and split-half stability of the two writer scorings: top-100 heads shared between halves and the R@1 gap between the rank-1 and rank-100 head (points).
<table><tr><td rowspan="2">Model</td><td colspan="4">Agreement of the in-context ranking</td><td colspan="4">Split-half stability</td></tr><tr><td></td><td>vs. writer vs. pointer</td><td> $\Delta$ </td><td>95% CI</td><td>shared top-100 in-ctx held-out</td><td></td><td>in-ctx</td><td>rank 1 vs. 100 gap held-out</td></tr><tr><td>InternVL3.5-4B</td><td>0.758</td><td>0.480</td><td>0.277</td><td>[0.231, 0.325]</td><td>65</td><td>85</td><td>4.3</td><td>22.0</td></tr><tr><td>InternVL3.5-8B</td><td>0.716</td><td>0.530</td><td>0.186</td><td>[0.143, 0.231]</td><td>71</td><td>87</td><td>4.0</td><td>24.4</td></tr><tr><td>Qwen3-VL-4B</td><td>0.573</td><td>0.403</td><td>0.170</td><td>[0.124, 0.217]</td><td>61</td><td>86</td><td>1.8</td><td>19.5</td></tr><tr><td>Qwen3-VL-8B</td><td>0.657</td><td>0.406</td><td>0.251</td><td>[0.202, 0.302]</td><td>63</td><td>84</td><td>2.2</td><td>28.9</td></tr></table>

The in-context ranking is closer to the held-out writer ranking than to the pointer ranking in every model, with the difference bounded away from zero. It is, however, a poorly resolved criterion: with only a handful of candidates most heads score near the ceiling, the rank-1 and rank-100 heads differ by only 1.8 to 4.3 points, and the two halves of the test set agree on just 61 to 71 of their top-100 heads, against 84 to 87 for the held-out scoring whose rank-1 to rank-100 gap is about 20 to 29 points. The held-out pool is used in the main text for this reason.

## F Where Writer-Only Heads Get Their Information

Writer-only heads are a relative partition: heads in the writer top-100 but outside the pointer top-100 (§6.1). The probe recovers the target image from their outputs although they fall outside the pointer top-100, which raises the question of where that information comes from: from the target-image tokens themselves, or relayed through text positions. Following Eq. 4, we decompose each writeronly head’s output at the cattok into the contributions of four source groups: target-image tokens, distractor-image tokens, interleaved filler passages, and the remaining question and option tokens. We then apply the writer probe of §5, unchanged, to the output reconstructed from one source group alone or with one group removed, and report the mean writer R@1 over the writer-only heads of each model (Table 9). Because the probe and the cosine ranking are nonlinear, a single source can score slightly above the full output; the rows are sufficiency and removal tests, not an additive decomposition. The two removal rows carry 95% confidence intervals from 5,000 paired head-bootstrap resamples within each model’s writer-only set; both exclude zero in every model.

Table 9: Writer R@1 of writer-only heads (mean over the set) when the head output is reconstructed from one source group or with one group removed. The two removal rows give the change relative to the full output, with its 95% bootstrap CI beneath. Text tokens are the interleaved filler passages plus the question and option tokens (including special tokens).
<table><tr><td>Source intervention</td><td>InternVL3.5-4B</td><td>InternVL3.5-8B</td><td>Qwen3-VL-4B</td><td>Qwen3-VL-8B</td></tr><tr><td>Full head output</td><td>0.331</td><td>0.285</td><td>0.352</td><td>0.346</td></tr><tr><td>Target-image tokens only</td><td>0.338</td><td>0.341</td><td>0.322</td><td>0.326</td></tr><tr><td>Remove target-image tokens (∆)</td><td>-0.175</td><td>-0.160</td><td>-0.163</td><td>-0.167</td></tr><tr><td>95% CI</td><td></td><td></td><td></td><td>[-0.198, -0.150] [-0.182, -0.138] [-0.185, -0.140] [-0.189, -0.144]</td></tr><tr><td>Text tokens only</td><td>0.155</td><td>0.126</td><td>0.190</td><td>0.178</td></tr><tr><td>Remove all text tokens (∆)</td><td>-0.049</td><td>-0.025</td><td>-0.085</td><td>-0.075</td></tr><tr><td>95% CI</td><td></td><td></td><td></td><td>[-0.075, -0.025] [-0.050, -0.001] [-0.112, -0.061] [-0.101, -0.051]</td></tr><tr><td>Interleaved filler only</td><td>0.003</td><td>0.003</td><td>0.002</td><td>0.003</td></tr><tr><td>Distractor-image tokens only</td><td>0.004</td><td>0.003</td><td>0.002</td><td>0.002</td></tr></table>

The target-image tokens alone retain a writer R@1 comparable to the full output, and removing them causes the largest drop in every model. Text tokens alone stay well below the full output, and removing all text costs much less than removing the target-image tokens; interleaved filler and distractor-image tokens alone carry almost nothing. Text therefore provides secondary information, but a relay through text positions is not required for the target-image information that the probe recovers.

Attention patterns and pointer ranks. The attention patterns point in the same direction. At the cattok and at positions +1 to +4, writer-only heads place 2.1 to 4.0 times as much attention mass on the target image as layer-matched random heads, while their attention to the interleaved filler is below that of random heads (0.74 to 0.89 times). Nor do they become strong pointers at a neighboring position: across +1 to +4 the median pointer rank of writer-only heads stays above 200 of 1,152, and at none of these positions in any model do more than 12 of them enter the pointer top-100. Attention magnitude alone also does not determine output content: among heads with similar pointer R@1 (0.30 to 0.40), writer R@1 differs by 4.0 to 4.2 times between the 10th and 90th percentiles, consistent with Eq. 4, in which attention weights set how sources are mixed while the value pathway sets what the mixture carries [20].

Scope. The decomposition localizes the source positions of the current head’s output; it does not say how the target-image-token representations formed in earlier layers. Writer R@1 measures probe recoverability, not downstream use: writer-only ablations remain near the matched-budget random-head controls (§6.1, Appendix I).

## G Readout Position: Both Signals Are Anchored at the Cattok

The pointer and writer criteria are read at the cattok, the last token of the queried-category mention (§3.2). This appendix checks how much of each signal survives when the readout moves to later positions: the four tokens following the cattok (+1 to +4), the token midway between the cattok and the end of the prompt (mid), and the last prefill token (last). The pointer ranking is recomputed at every position from the attention mass of that position over each image span (Eq. 3) on the 6,000 ICIR-MCQ samples. The writer probes are retrained independently at each position (+1, +2, +4, mid, last) with the configuration of Appendix D; a probe retrained at the cattok itself reproduces the cattok writer ranking at Spearman $\rho = 0 . 9 9 3$ on every model, which serves as the reference for retraining variability. Table 10 reports, for each position, the Spearman correlation of the full 1,152-head ranking with the cattok ranking together with the top-100 overlap, and the retained signal of the head sets fixed at the cattok: the mean R@1 of the cattok pointer top-100 (writer top-100) at that position divided by its value at the cattok.

Table 10: Readout-position control. First row per model: Spearman $\rho$ of the pointer ranking at each position against the cattok pointer ranking, and the top-100 overlap. Second row: mean pointer R@1 of the pointer top-100 of this sweep’s cattok ranking, at each position divided by its value at the cattok. Third row: the same correlation and overlap for the writer ranking retrained at each position; the cattok entry is the retrained cattok probe against the original cattok ranking. Fourth row: mean writer R@1 of the fixed cattok writer top-100 under the probe retrained at each position, divided by its value under the retrained cattok probe. Writer probes were not retrained at +3.
<table><tr><td></td><td>cattok</td><td>+1</td><td> $+ 2$ </td><td>+3</td><td>+4</td><td>mid</td><td>last</td></tr><tr><td>InternVL3.5-4B</td><td></td><td>0.960 / 79</td><td>0.946 / 77</td><td>0.945 / 75</td><td>0.948 / 76</td><td>0.957 / 78</td><td>0.683 / 52</td></tr><tr><td>pointer set</td><td>1.00</td><td>0.86</td><td>0.81</td><td>0.80</td><td>0.86</td><td>0.82</td><td>0.34</td></tr><tr><td>writer ranking ρ / overlap</td><td>0.993 / -</td><td>0.850 / 72</td><td>0.812 / 67</td><td>一</td><td>0.847 / 70</td><td>0.810 / 65</td><td>0.536 / 16</td></tr><tr><td>writer set InternVL3.5-8B</td><td>1.00</td><td>0.73 0.957 / 77</td><td>0.57</td><td>1</td><td>0.68</td><td>0.51</td><td>0.32</td></tr><tr><td>pointer set</td><td>1.00</td><td>0.85</td><td>0.949 / 74 0.81</td><td>0.948 / 73 0.80</td><td>0.943 / 72 0.85</td><td>0.959 / 73 0.84</td><td>0.682 / 50 0.41</td></tr><tr><td>writer ranking ρ / overlap</td><td>0.993 / -</td><td>0.847 / 63</td><td>0.792 / 62</td><td>一</td><td>0.823 / 59</td><td>0.818 / 57</td><td>0.558 / 17</td></tr><tr><td>writer set</td><td>1.00</td><td>0.69</td><td>0.54</td><td></td><td>0.62</td><td>0.51</td><td>0.46</td></tr><tr><td>Qwen3-VL-4B</td><td></td><td>0.961 / 87</td><td>0.949 / 81</td><td>0.946 / 82</td><td>0.945 / 85</td><td>0.959 / 82</td><td>0.739 / 51</td></tr><tr><td>pointer set</td><td>1.00</td><td>0.84</td><td>0.77</td><td>0.78</td><td>0.85</td><td>0.79</td><td>0.60</td></tr><tr><td>writer ranking ρ / overlap</td><td>0.993 / –</td><td>0.878 / 67</td><td>0.820 / 59</td><td></td><td>0.857 / 60</td><td>0.803 / 53</td><td>0.713 / 26</td></tr><tr><td>writer set</td><td>1.00</td><td>0.63</td><td>0.44</td><td></td><td>0.55</td><td>0.36</td><td></td></tr><tr><td>Qwen3-VL-8B</td><td></td><td>0.959 / 87</td><td>0.943 / 85</td><td></td><td>0.942 / 820.939 / 86</td><td>0.954 / 80</td><td>0.82</td></tr><tr><td>pointer set</td><td>1.00</td><td>0.85</td><td>0.77</td><td>0.77</td><td>0.85</td><td>0.78</td><td>0.727 / 60</td></tr><tr><td>writer ranking  $\rho /$  overlap</td><td>0.993 / -</td><td>0.875 / 67</td><td>0.829 / 64</td><td></td><td></td><td></td><td>0.61</td></tr><tr><td></td><td></td><td></td><td></td><td>一</td><td>0.851 / 68</td><td>0.815 / 57</td><td>0.730 / 33</td></tr><tr><td>writer set</td><td>1.00</td><td>0.67</td><td>0.46</td><td>一</td><td>0.56</td><td>0.39</td><td>0.98</td></tr></table>

Both rankings stay close to their cattok version at the local positions and at mid: pointer ρ 0.939–0.961 with top-100 overlap 72–87, writer $\rho 0 . 7 9 \mathrm { - } 0 . 8 8$ , and the top-1 pointer head is unchanged at every local position in every model. The fixed sets keep a substantial fraction of their signal there: at +1 to +4 the pointer set retains 0.77–0.86 of its cattok R@1 and the writer set 0.44–0.73. At the last prefill token both rankings move further from the cattok ranking (pointer ρ 0.68–0.74, writer $\rho \ : 0 . 5 4 \mathrm { - } 0 . 7 3 )$ but both signals remain measurable. The discovered head populations therefore persist beyond the single scoring position, while the cattok remains the position at which both signals are strongest.

## H Majority-vote retrieval with the writer ranking

This appendix asks how the writer ranking transfers to Visual Haystacks [41], and whether ensembling the top-K writer-ranked heads recovers the gap to the single best pointer head. We pair each ranking with the scoring rule it was ranked under, so the comparison is between matched ranking–metric pairs rather than between two metrics applied to a fixed head set.

Head-level vote and majority-vote scoring. Let $\mathcal { H } _ { K } ^ { \mathrm { p t r } }$ and $\mathcal { H } _ { K } ^ { \mathrm { w r i } }$ denote the top-K heads under the pointer and writer rankings. For a VHs sample with image set $\{ e _ { i } \} _ { i = 1 } ^ { N } \colon$

• Pointer-ranking, attention-score voting (A): each $h \in \mathcal { H } _ { K } ^ { \mathrm { p t r } }$ casts a vote $v _ { h } ^ { A } { = } \operatorname { a r g m a x } _ { i } \ a _ { h , i } .$ where $a _ { h , i }$ is the cattok attention sum from head h to image i’s patches.

• Writer-ranking, trained-probe voting (B): each $h \in \mathcal { H } _ { K } ^ { \mathrm { w r i } }$ casts a vote

$$
v _ { h } ^ { B } = \arg \operatorname* { m a x } _ { i } \left. \tilde { f } _ { h } ( o _ { h } ) , \tilde { g } _ { h } ( e _ { i } ) \right. ,
$$

where $f _ { h }$ and $g _ { h }$ are the trained head-side and image-side projectors of head h (Appendix D), and ˜· denotes $\ell _ { 2 }$ normalization.

The sample-level prediction is the majority vote over the K votes (with ties broken by smallest image index); $\bar { K } = 1$ recovers the standalone single-head predictor in each case. We report R@1 against the ground-truth target image position on 1,000 samples per haystack size for all four LVLMs; Figure 6 plots the per-LVLM dose-response.

![](images/535e9ad444f66f7f09d704167ed84e18a463f1294b4b2e2101bc96a37e9a72cf.jpg)  
Figure 6: Top-K majority-vote R@1 on Visual Haystacks under the two matched ranking–metric pairs, for all four LVLMs (n=1,000 per haystack size). Solid lines: pointer ranking, attention-score voting (A). Dashed lines: writer ranking, trained-probe voting (B). Each color is one haystack size $\in \{ 2 , 3 , 5 , 1 0 , 2 0 \}$

Single head: the pointer ranking is stronger in 19 of 20 cells. At K=1 the pointer top head outperforms the writer top head in 19 of 20 (model, N) cells, by 2.3 to 17.2 pp; the exception is Qwen3-VL-4B at N=2 (89.4 vs. 92.5). The gap widens with haystack size, to about 17 pp on the 8B models and about 9 pp on the 4B models. The pointer ranking thus identifies the stronger individual head, even though the writer probe was trained for retrieval.

Ensembling closes most of the single-head gap. Majority vote over the top-K writer-probe heads improves with K and saturates by $\mathrm { ~ \bar { ~ } { ~ K ~ } \approx 2 0 { - } 3 0 \bar { ~ } }$ on all four models; at saturation it is within 4.0 pp of the best pointer configuration in every (model, N) cell, within 1.2 pp at N=2, and matches or marginally exceeds it in four cells. The pointer ranking behaves differently, and the difference depends on scale: on the 8B models a single head is already at or near the best (adding heads gains at most 0.8 pp), while on the 4B models a small ensemble helps (up to +7.4 pp at K=5 on Qwen3-VL-4B). The pointer ranking thus identifies a few heads whose raw attention pattern is by itself an accurate retriever, whereas the writer ranking identifies heads whose outputs carry retrieval information that is realized in aggregate rather than head by head.

## I Set-Partition Ablation: overlap vs. pointer-only vs. writer-only

To isolate which part of the pointer top-100 and writer top-100 drives the causal effect, we partition their union into three disjoint subsets and run the targeted output-zero ablation of §6 (Appendix A) on each at $K \in \{ 5 , 1 0 , \bar { 1 5 } , 2 0 , 2 5 , 3 0 \}$ and $N \in \{ 2 , \bar { 3 } , 5 , 1 0 , \bar { 2 0 } , 5 0 \}$ . Within each subset heads are ordered by pointer rank (overlap, pointer-only) or writer rank (writer-only), and the first K heads of that order are ablated; the overlap subset has 35–47 heads and pointer-only and writer-only 53–65 each, depending on the model. A random control of the same K uniformly sampled heads serves as the reference. Figure 7 plots the per-model dose-response curves.

Table 11 lists the overlap heads of each model in ablation order. The pointer top-1 head L21H10 is in the overlap of every model (writer rank 35/33/55/50 on InternVL3.5-8B / InternVL3.5-4B / Qwen3-VL-8B / Qwen3-VL-4B), and 23 heads are in the overlap of all four models.

## I.1 Prefill-only versus full-generation intervention

The ablations above stop at the last prefill position, so the first Yes/No logit already reflects the intervention while the generated tokens are untouched (Appendix A). To check that stopping at the prefill does not change the conclusions, we re-ran the three arms and the three random controls at $\mathbf { \bar { \mathit { N } } } = 1 0 , \mathit { K } = 3 0$ with the zeroing continued through every generated token, and compared the two runs sample by sample (Table 12). For both InternVL models, which answer with a single word, no prediction changes. For the Qwen3-VL models, whose answers can run to a full sentence (mostly on the 8B model), continuing the intervention touches tokens after the decision and changes at most 5.1% of parsed answers, moving accuracy by at most 0.9 points without changing the ordering of the arms.

Table 11: Overlap heads (in both pointer top-100 and writer top-100, cattok query) per LVLM, listed by pointer rank, the order in which the overlap subset is ablated (the first K heads form the budget-K overlap arm). Format LxxHyy(ax/ry) gives the head’s 1-indexed pointer rank x and writer rank y.
<table><tr><td>Model</td><td>Overlap heads (pointer-rank order)</td></tr><tr><td>InternVL3.5-8B (47 heads)</td><td>L21H10(a1/r35), L21H11(a2/r46), L21H08(a3/r20), L20H16(a8/r81), L20H31(a9/r16), L20H21(a10/r68), L21H17(a11/r6), L20H15(a14/r3),</td><td></td><td>L20H17(a4/r83), L21H18(a7/r10),</td></tr><tr><td rowspan="6"></td><td>L19H24(a17/r97), L20H30(a18/r8), L22H04(a20/r34), L23H04(a21/r82), L20H28(a23/r2),</td><td></td><td></td><td></td></tr><tr><td>L21H31(a24/r4), L20H13(a25/r63), L24H29(a27/r14), L24H05(a28/r7), L18H19(a29/r5),</td><td></td><td></td><td></td></tr><tr><td>L20H14(a32/r45), L19H27(a34/r42), L24H23(a36/r1), L19H13(a38/r25), L28H23(a41/r91),</td><td></td><td></td><td></td></tr><tr><td>L19H03(a42/r94), L17H04(a46/r51), L22H06(a49/r22), L23H31(a50/r90), L28H02(a51/r69),</td><td></td><td></td><td></td></tr><tr><td>L13H11(a52/r38), L19H28(a54/r49), L19H14(a58/r74), L18H18(a59/r9), L24H13(a64/r33),</td><td></td><td></td><td></td></tr><tr><td>L19H25(a69/r75), L31H30(a73/r96), L23H09(a78/r27), L29H15(a79/r64), L19H12(a82/r44),</td><td></td><td></td><td></td></tr><tr><td rowspan="6">Qwen3-VL-8B (35 heads)</td><td>L29H11(a85/r77), L15H11(a87/r24), L23H11(a89/r17), L23H28(a91/r18), L20H29(a94/r99),</td><td></td><td></td><td></td></tr><tr><td>L18H05(a99/r47), L24H14(a100/r19)</td><td></td><td></td><td></td></tr><tr><td>L21H10(a1/r55), L21H08(a2/r21), L21H11(a3/r60), L21H18(a6/r18), L20H15(a9/r24),</td><td></td><td></td><td></td></tr><tr><td>L19H27(a10/r16), L21H17(a11/r2), L20H31(a13/r28), L22H04(a14/r86), L20H30(a17/r6), L18H19(a23/r7), L21H31(a24/r4), L24H29(a25/r10), L19H13(a28/r96), L20H28(a30/r5),</td><td></td><td></td><td></td></tr><tr><td>L24H05(a32/r11), L17H04(a36/r45), L24H23(a37/r1), L18H18(a40/r17), L22H06(a45/r80),</td><td></td><td></td><td></td></tr><tr><td>L28H02(a46/r26), L24H30(a49/r58), L23H30(a54/r68), L25H09(a62/r81), L24H13(a67/r15),</td><td></td><td></td><td></td></tr><tr><td rowspan="11">InternVL3.5-4B (42 heads)</td><td>L19H25(a68/r91), L18H05(a70/r49), L30H06(a71/r25), L26H31(a75/r29), L25H11(a84/r76),</td><td></td><td></td><td></td></tr><tr><td>L23H28(a90/r62), L17H27(a91/r79), L23H12(a93/r97), L28H03(a97/r43), L24H14(a98/r8)</td><td></td><td></td><td></td></tr><tr><td>L21H10(a1/r33), L21H11(a2/r23), L21H08(a4/r29), L21H18(a6/r9), L23H04(a10/r92),</td><td></td><td></td><td></td></tr><tr><td>L24H29(a11/r8), L21H17(a12/r5), L20H31(a13/r27), L21H16(a14/r89), L20H16(a16/r91),</td><td></td><td></td><td></td></tr><tr><td>L20H15(a17/r4), L21H23(a21/r59), L21H31(a22/r1), L28H23(a23/r63), L22H04(a25/r24),</td><td></td><td></td><td></td></tr><tr><td>L20H30(a26/r6), L24H23(a29/r3), L13H11(a30/r100), L24H05(a32/r15), L20H18(a33/r74),</td><td></td><td></td><td></td></tr><tr><td>L20H14(a36/r73), L20H28(a38/r2), L28H03(a39/r97), L31H30(a40/r85), L28H02(a41/r21),</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>L19H27(a45/r50), L24H13(a46/r53), L18H19(a47/r11), L24H14(a49/r22), L17H04(a51/r66),</td><td></td><td></td></tr><tr><td></td><td>L23H28(a52/r13), L19H13(a53/r36), L29H13(a55/r99), L22H06(a61/r38), L21H30(a69/r84),</td><td></td><td></td></tr><tr><td colspan="4">L19H14(a79/r88), L15H11(a84/r96), L19H28(a86/r60), L18H18(a87/r18), L19H12(a88/r82), L28H16(a99/r98), L26H18(a100/r20)</td></tr><tr><td rowspan="8">Qwen3-VL-4B (43 heads)</td><td colspan="5">L21H10(a1/r50), L21H11(a3/r85), L21H08(a4/r28), L21H18(a6/r15), L22H04(a8/r55),</td></tr><tr><td>L21H17(a9/r4), L20H16(a10/r62), L20H31(a11/r8), L20H15(a12/r12), L20H17(a13/r77),</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>L19H24(a14/r69), L19H27(a17/r19), L20H18(a18/r58), L21H31(a19/r3), L21H23(a22/r86),</td><td></td><td></td></tr><tr><td>L18H19(a23/r5), L19H13(a24/r48), L20H30(a25/r7), L24H29(a26/r26), L28H23(a29/r87),</td><td></td><td></td><td></td></tr><tr><td>L19H14(a30/r36), L19H12(a35/r53), L21H30(a38/r82), L20H28(a40/r2), L30H06(a42/r40),</td><td></td><td></td><td></td></tr><tr><td>L22H06(a45/r32), L18H18(a46/r6), L24H23(a51/r1), L18H05(a56/r33), L17H04(a58/r31),</td><td></td><td></td><td></td></tr><tr><td></td><td>L24H13(a61/r59), L24H05(a62/r43), L19H28(a67/r63), L19H25(a72/r49), L28H02(a76/r72),</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>L24H30(a77/r92), L13H11(a83/r25), L17H05(a84/r95), L24H14(a87/r16), L17H27(a91/r57),</td><td></td><td></td></tr><tr><td></td><td>L23H12(a92/r64), L23H14(a98/r80), L26H31(a100/r71)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 12: Full-generation minus prefill-only intervention at N = 10, K = 30 (VHs, n = 1,000): change in accuracy (points) and fraction of samples whose parsed answer changed.
<table><tr><td></td><td>overlap</td><td>pointer-only</td><td>writer-only</td><td>random (3 seeds)</td></tr><tr><td>InternVL3.5-4B</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>0.0 (0%)</td></tr><tr><td>InternVL3.5-8B</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>0.0 (0%)</td></tr><tr><td>Qwen3-VL-4B</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>0.0 (0%)</td><td>+0.1 / +0.1 / −0.1 (0.1–0.3%)</td></tr><tr><td>Qwen3-VL-8B</td><td>−0.4 (4.1%)</td><td>−0.8 (5.1%)</td><td>−0.9 (2.2%)</td><td>−0.2 / +0.8 / 0.0 (0–1.7%)</td></tr></table>

## J Replication on Additional LVLMs

## J.1 Discovery on three further models

We applied the pointer and writer criteria, unchanged, to three additional LVLMs: LLaVA-OneVision-7B [21] (Qwen2 backbone, SigLIP vision tower), Pixtral-12B [1] (Mistral-Nemo backbone, Pixtral vision encoder), and InternVL3.5-14B [40] (a larger member of the InternVL family). Probes use the configuration of Appendix D. Images are processed at a fixed size per model so that candidate images contribute equal numbers of visual tokens: 384-pixel single-tile inputs for LLaVA-OneVision, whose any-resolution default would otherwise give images different token counts, and 448 pixels for Pixtral. Table 13 reports the pointer top-1 head R@1 on ICIR-MCQ and the pointer-writer overlap among the top-100 heads, together with the overlap expected if the two rankings were unrelated $( 1 0 0 ^ { 2 } / H$ for H heads).

![](images/1c8b0e5bd42ea7049313519234efab683c6ac4f77cb555ba29703d462de78f27.jpg)  
Figure 7: Set-partition dose-response per LVLM (top-100 partition, output-zero). Rows: LVLMs (top to bottom: InternVL3.5-8B, Qwen3-VL-8B, InternVL3.5-4B, Qwen3-VL-4B). Columns: haystack $\mathrm { s i z e } \in \{ 2 , 3 , 5 , 1 0 , 2 0 , 5 0 \}$ . Random is the mean over 3 seeds.

Table 13: Pointer-writer discovery on additional LVLMs (ICIR-MCQ, 6,000 samples). Overlap counts heads in both top-100 lists; chance is $1 0 0 ^ { 2 } / H$
<table><tr><td>Model</td><td>LLM / vision</td><td>H</td><td>Top-1 R@1</td><td>Overlap</td><td>Chance</td><td>Ratio</td></tr><tr><td>Main-text models (4)</td><td>InternVL3.5, Qwen3-VL 1,152</td><td></td><td>0.866–0.922</td><td>35-47</td><td>8.7</td><td>4.0-5.4×</td></tr><tr><td>LLaVA-OneVision-7B</td><td>Qwen2 / SigLIP</td><td>784</td><td>0.962</td><td>46</td><td>12.8</td><td>3.6×</td></tr><tr><td>Pixtral-12B</td><td>Mistral-Nemo / Pixtral</td><td>1,280</td><td>0.874</td><td>40</td><td>7.8</td><td>5.1×</td></tr><tr><td>InternVL3.5-14B</td><td>Qwen3-14B / InternViT</td><td>1,600</td><td>0.844</td><td>31</td><td>6.2</td><td>5.0×</td></tr></table>

All three models reproduce the discovery-side pattern: a single pointer head retrieves the target image with high R@1, and the pointer and writer top-100 lists overlap several times more than chance while remaining far from identical.

## J.2 Set-partition ablation on LLaVA-OneVision-7B

We ran the complete set-partition protocol of §6 on LLaVA-OneVision-7B under the same conditions as the four main-text models: Visual Haystacks, $n = 1 , 0 0 0$ per haystack size, the output-zero intervention of §6, top-100 partition, $K \in \{ 5 , 1 0 , 1 5 , 2 0 , 2 5 , 3 0 \}$ , and matched-budget random controls over three seeds. Table 14 reports accuracy relative to the random control at each $( N , K )$ Haystack size 50 is omitted for this model: with about 1,485 visual tokens per image the N=50 prompt runs to about 74k tokens, more than twice the model’s 32,768-token training context, and the unablated baseline itself falls below chance, so no intervention effect can be measured there. The four main-text models use at most 256 tokens per image (Appendix A) and stay within context.

The overlap set is the most damaging arm in all 30 cells, and its drop is significant (McNemar, $p < 0 . 0 5 )$ against all three random seeds in 24 of the 30 cells and against at least two seeds in all 30. The sparsity of the effect is visible at $K = 5 \mathrm { : }$ removing five overlap heads already costs 0.07 to 0.14 accuracy at $N \leq 1 0$ , while five pointer-only or writer-only heads stay within 0.02 of the random control. Writer-only ablations stay within ±0.02 of random throughout.

Table 14: LLaVA-OneVision-7B set-partition ablation: accuracy minus the matched random control (mean of three seeds) for the pointer-and-writer overlap (first row per N), pointer-only, and writeronly sets. Baseline is the unablated accuracy at each haystack size.
<table><tr><td>N</td><td>Baseline</td><td>K=5</td><td>K=10</td><td>K=15</td><td>K=20</td><td>K=25</td><td>K=30</td></tr><tr><td>2</td><td></td><td>0.761</td><td>-0.123</td><td>-0.180 -0.198</td><td>-0.209</td><td>-0.211</td><td>-0.213</td></tr><tr><td></td><td>pointer-only</td><td></td><td>-0.011 -0.094</td><td>-0.098</td><td>-0.100</td><td>-0.106</td><td>-0.132</td></tr><tr><td>3</td><td>writer-only</td><td>-0.003</td><td>-0.005</td><td>-0.002</td><td>-0.007</td><td>-0.009</td><td>-0.017</td></tr><tr><td>pointer-only</td><td></td><td>0.741 -0.144</td><td>-0.193</td><td>-0.207</td><td>-0.213</td><td>-0.214</td><td>-0.210</td></tr><tr><td>writer-only</td><td></td><td>-0.007</td><td>-0.083</td><td>-0.114</td><td>-0.113</td><td>-0.113</td><td>-0.136</td></tr><tr><td>5</td><td></td><td>-0.004</td><td>-0.003</td><td>-0.007</td><td>-0.007</td><td>-0.015</td><td>-0.017</td></tr><tr><td></td><td></td><td>0.685 -0.100</td><td>-0.165</td><td>-0.167</td><td>-0.160</td><td>-0.149</td><td>-0.156</td></tr><tr><td></td><td>pointer-only</td><td>-0.005</td><td>-0.055</td><td>-0.080</td><td>-0.063</td><td>-0.054</td><td>-0.075</td></tr><tr><td>writer-only</td><td></td><td>+0.003</td><td>-0.002</td><td>-0.007</td><td>-0.005</td><td>+0.005</td><td>-0.005</td></tr><tr><td>10</td><td></td><td>0.637 -0.069</td><td>-0.105</td><td>-0.112</td><td>-0.094</td><td>-0.083</td><td>-0.088</td></tr><tr><td>pointer-only</td><td></td><td>-0.013</td><td>-0.028</td><td>-0.041</td><td>-0.017</td><td>-0.010</td><td>-0.017</td></tr><tr><td>writer-only</td><td></td><td>+0.013</td><td>+0.014</td><td>+0.006</td><td>+0.013</td><td>+0.015</td><td>+0.010</td></tr><tr><td>20</td><td>0.577</td><td>-0.041</td><td>-0.067</td><td>-0.066</td><td>-0.056</td><td>-0.047</td><td>-0.047</td></tr><tr><td>pointer-only</td><td></td><td>-0.016</td><td>-0.015</td><td>-0.003</td><td>+0.011</td><td>+0.008</td><td>-0.014</td></tr><tr><td>writer-only</td><td></td><td>+0.006</td><td>+0.008</td><td>+0.003</td><td>+0.003</td><td>+0.020</td><td>+0.018</td></tr></table>

## K Visual vs. Text Retrieval Heads

We examined whether the image retrieval heads of Qwen3-VL-8B also serve text retrieval, using a text needle-in-a-haystack task: a numeric needle is embedded in a long interleaved image-text context (the images are filler and none is a retrieval target), and the model is asked to return the number. Baseline accuracy is 100% on the 291 samples, so head-level differences reflect attention behavior rather than task difficulty; we rank all 1,152 heads by their mean attention mass on the needle tokens at the position where the answer is generated. The two probes are not matched (see below), so we read the result as suggestive only. Table 15 shows selected heads and their ranks in image versus text retrieval.

Table 15: Image vs. text retrieval rankings (Qwen3-VL-8B).
<table><tr><td>Head</td><td>Image rank</td><td>Text rank</td><td>Character</td></tr><tr><td>L21H10</td><td>1</td><td>118</td><td>Image-specific</td></tr><tr><td>L21H08</td><td>2</td><td>67</td><td>Image-specific</td></tr><tr><td>L21H11</td><td>3</td><td>66</td><td>Image-specific</td></tr><tr><td>L23H04</td><td>20</td><td>10</td><td>Mixed</td></tr><tr><td>L29H09</td><td>137</td><td>1</td><td>Text-specific</td></tr><tr><td>L29H11</td><td>221</td><td>2</td><td>Text-specific</td></tr></table>

The top image retrieval head ranks 118th for text retrieval and the top text retrieval head ranks 137th for image retrieval; no head is in the top 10 for both. The pointer cluster sits in layers 19–23, whereas the two strongest text retrieval heads sit at layer 29 and the text top-10 spans layers 23–32.

Why the comparison is not precise. The two probes differ in candidate granularity (image spans of hundreds of patch tokens versus a few-token needle), in the scoring rule (argmax of attention mass over image spans versus attention to the needle position), in task format, and in query position (the cattok versus the answer position), any of which can move a head’s rank for reasons unrelated to modality. We therefore take the result as evidence that the two retrieval functions are not served by the same top-ranked heads in this network, and leave a matched probe to future work.

## L Extended Discussion

Cross-model convergence. Head-level roles that recur across independently trained models are more plausibly a property of the architecture and of multi-image training than a per-model artifact:

InternVL3.5 and Qwen3-VL are trained by different organizations on different data mixtures, yet the pointer cluster occupies the same mid-block layers of the same 36-block shape on all four models (Appendix C). This mid-block position matches where language-only transformers implement retrieval and factual lookup [15, 42], which suggests that the pointer-writer cluster is the multimodal analogue at the same structural position; we do not distinguish between these explanations.

Relation to text retrieval heads. Wu et al. [42] identified retrieval heads in text-only LLMs by an attention-pattern criterion.

We expect input-side selection vs. output-side encoding to be a useful axis for revisiting text retrieval heads in long-context QA, but leave that to future work.