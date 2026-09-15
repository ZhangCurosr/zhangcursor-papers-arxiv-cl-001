# MarKey: Marginal Utility Guided Greedy Keyframe Selection for Long Video Understanding

Hongchang Shi<sup>1</sup> Jinpeng Hu<sup>1</sup> Ao Wang<sup>1</sup> Wenzheng Zhou<sup>1</sup> Hui Ma<sup>1</sup> Feng Li<sup>1</sup> Zenglin Shi<sup>1</sup>

<sup>1</sup>Hefei University of Technology, Hefei, China

2024170833@mail.hfut.edu.cn, 135858hjp@gmail.com

## Abstract

Long-video understanding remains challenging for multimodal large language models (MLLMs) because densely encoding long frame sequences is computationally expensive, while uniform sampling under a limited visual budget can miss sparse yet decisive evidence. Recent training-free keyframe selection methods have enabled more efficient inference and yielded promising performance gains. However, many existing methods score frames largely in isolation without explicitly considering how each candidate complements the currently selected subset, potentially resulting in redundant selections and incomplete evidence coverage. To address this limitation, we propose MarKey, a training-free framework that formulates keyframe selection as subset-aware greedy optimization. At each iteration, MarKey scores each candidate using a tractable surrogate that jointly accounts for query relevance, marginal coverage gain, and context-dependent redundancy, and selects the frame with the highest utility. To make this iterative subset-aware evaluation efficient, MarKey uses a compact set of representative anchors to approximate fullvideo coverage and a bounded window of previously selected frames to limit context-dependent comparisons. Experiments on six benchmarks spanning holistic video understanding, human-centric video understanding, and openended video understanding demonstrate that MarKey consistently outperforms existing methods. Further analyses show robust gains across different MLLM backbones, model scales, andframe budgets.

## 1. Introduction

Multimodal large language models (MLLMs) have demonstrated strong capabilities in visual understanding and reasoning across a wide range of image and video tasks [1, 22, 57]. As these models are increasingly applied to more complex video scenarios, long-form video understanding has emerged as an important frontier. This progress is accompanied by a growing body of benchmarks that demand multi-step reasoning [18, 53], long-horizon evidence aggregation [36, 47], comprehensive reasoning over complex real-world scenarios [9, 10, 12], and fine-grained temporal understanding [4, 32] over videos spanning minutes or even hours. Compared with short videos, long-video understanding poses a more fundamental evidence-allocation challenge, as task-relevant cues are often sparse, temporally dispersed, and meaningful only when interpreted in relation to the query and the broader video narrative. Because densely encoding an entire long video is computationally prohibitive, most MLLMs operate under a fixed visual budget and process only a limited number of uniformly sampled frames. Although simple and efficient, uniform sampling provides increasingly coarse temporal coverage as video duration grows, making it more prone to miss brief yet decisive events required for downstream reasoning. These limitations have motivated frame selection methods that allocate the constrained visual budget more effectively before costly MLLM inference. Existing ap proaches can be broadly categorized into learning-based and training-free methods. Learning-based methods explic itly optimize a querying or ranking policy to select frames that are useful for Video-LLMs, often relying on additional supervision or proxy objectives [13, 58]. For example, ReFoCUS [17] treats selection as an autoregressive deci sion policy and applies reinforcement learning with rewards derived from a frozen reference multimodal model to di rectly optimize evidence selection for temporally grounded reasoning. Although effective, such methods typically re quire extra training and may be less convenient to trans fer across different MLLM backbones, prompts, and de ployment settings. Training-free methods instead construct frame subsets at inference time using pretrained visionlanguage representations [31, 41, 42, 72]. Their plug-and play nature makes them particularly attractive for long video understanding across heterogeneous MLLM back bones. For instance, AKS [42] introduces a keyframe selec tion method that recursively partitions the video and adap tively selects keyframes. FOCUS [72] proposes a budgeted evidence search strategy that progressively identifies infor mative temporal regions and selects query-relevant frames under a strict frame budget. Despite this progress, a fundamental question remains insufficiently addressed: how much new evidence does a candidate frame contribute be yond theframes already selected? Under a strict frame bud get, individual relevance and subset utility are not equivalent. As illustrated in Fig. 1, multiple frames depicting the same stream-crossing event may all appear highly relevant to the question. However, once one such frame has been selected, additional visually similar frames contribute little new evidence toward determining the total number of crossings, whereas a moderately less relevant frame may become crucial if it captures a different, previously unob served event. The utility of a candidate is therefore con ditioned on the evolving selection context, since a frame that is informative in isolation may become redundant af ter similar evidence has been included. This context de pendence exposes a mismatch between frame-level scoring and the objective of keyframe selection. Although recent methods employ sequential selection, recursive partition ing, or heuristic search, many do not explicitly optimize this subset-conditioned marginal contribution. Consequently, the selected frames may be individually relevant yet col lectively redundant, resulting in incomplete evidence coverage. This problem becomes increasingly severe as video du ration grows because more potentially relevant events must compete for the same fixed number of visual slots.

![](images/f8baa8894281730011faa51174533ed0a6b32312327c838d202f9f4fad7cf527.jpg)  
Figure 1. Existing methods often select redundant frames around salient moments, limiting content coverage. Our query-guided selection balances relevance, coverage, and redundancy to yield compact, complementary keyframes.

To address this limitation, keyframe selection should move beyond local relevance scoring and evaluate each candidate relative to the evolving subset. Accordingly, we propose MarKey, a training-free framework that reformulates keyframe selection from independent frame prioritization into greedy optimization of subset-conditioned marginal utility. At each iteration, MarKey estimates the additional evidence contributed by each remaining candidate by jointly considering query relevance, anchor-based marginal coverage gain, and context-dependent redundancy. Query relevance captures the candidate’s alignment with the question, marginal coverage gain measures its contribution to previously underrepresented video content, and the redundancy penalty suppresses substantial overlap with the current selection. Together, these complementary signals provide a tractable surrogate for the candidate’s marginal contribution, enabling MarKey to favor frames that are both query-relevant and evidentially complementary. MarKey then adopts a greedy selection strategy that iteratively selects the frame with the highest surrogate utility, yielding a compact yet informative subset for downstream reasoning. Representative anchors approximate video-wide coverage, while a bounded active context limits redundancy comparisons over the selection history, making iterative subset-aware evaluation computationally practical. We evaluate MarKey on six benchmarks spanning holistic, human-centric, and open-ended video understanding. Further analyses demonstrate robust gains across MLLM backbones, model scales, and frame budgets. Component ablations and selected-frame redundancy analysis further verify the complementary effects of relevance, coverage, and redundancy modeling.

Our main contributions are summarized as follows:

• We formulate training-free keyframe selection as a dynamic subset-construction problem, shifting candidate evaluation from standalone relevance to the additional evidence contributed beyond the selected subset.

• We propose MarKey, whose surrogate utility jointly models query relevance, video-wide coverage, and contextdependent redundancy to construct complementary evidence subsets.

• We make iterative subset-aware selection efficient by using representative anchors to approximate video-wide coverage and a bounded active context to limit redundancy comparisons.

• Extensive experiments on six benchmarks demonstrate consistent improvements. Further analyses confirm robustness across MLLM backbones, model scales, frame budgets, and video durations.

## 2. Related Work

## 2.1. MLLMs for Long-Video Understanding

MLLMs build upon recent advances in LLMs, which have substantially improved natural language understanding and reasoning across a broad range of tasks [6, 11, 55]. By incorporating visual encoders and unified token interfaces, MLLMs further extend these capabilities to joint reasoning over language and visual inputs. Early research in this area primarily focused on image-text understanding tasks [65, 68]. Representative methods such as LLaVA [29] and MiniGPT-4 [71] adopt a modular design, where a pretrained visual encoder extracts image features and a lightweight projection module aligns them with the language space of a large language model. Furthermore, many recent studies have begun to extend multimodal large language models from static images to video inputs by encoding sampled frames and integrating temporal visual information. VideoChat [25], Video-ChatGPT [35], Video-LLaVA [28] and Video-LLaMA [62] follow this paradigm and employ multimodal instruction tuning to enable large language models to reason over short video clips. As research progresses, a series of architectural refinements has been proposed to better model temporal information and improve the efficiency of visual token processing. Models like LLaVA-OneVision [19], LLaVA-NeXT [21] and its video variants, Aria [20], PLLaVA [54] and Kangaroo [30] unify multi-granularity visual inputs, strengthen temporal adapters and refine projection modules or training curricula. Recent work further shifts attention to long videos and extended visual contexts. Works such as LongVILA [5], LongVA [64] and LongVLM [49] extend the effective context length of large language models and introduce hierarchical or multi-level representations, enabling more robust reasoning over long untrimmed video sequences.

## 2.2. Efficient Long-Video Understanding

The computational burden of long-video MLLMs is largely caused by the rapid accumulation of visual tokens, which increases both multimodal encoding cost and the context consumed during language-model inference. Beyond directly extending the context window, a growing body of research improves efficiency by redesigning how long visual streams are represented, compressed, and accessed. One line of work constructs compact visual representations within the model [14, 16, 26]. LLaMA-VID [27] represents each frame using a small number of content and context tokens, substantially reducing the visual sequence length while retaining frame-specific information. Video-XL [40] exploits key-value sparsification and dynamic compression to summarize visual information over long temporal intervals, whereas LongVU [39] adaptively removes spatial and temporal redundancy according to inter-frame dependencies and textual guidance. Another line of research avoids encoding the entire video into a single dense context and instead organizes visual evidence through structured abstraction and retrieval [34, 48, 61]. For example, Video-RAG [33] builds an auxiliary multimodal knowledge base from long-video content and retrieves question-relevant visual and textual evidence before MLLM inference, enabling the reasoning model to access a compact context without processing the complete video sequence. While effective, these approaches often rely on specialized compression modules, external memory, or auxiliary retrieval pipelines, which introduce additional computational and system overhead and may discard fine-grained visual evidence during abstraction. This motivates lightweight pre-processing strategies, particularly keyframe selection, as a practical and complementary solution for reducing visual redundancy while preserving task-relevant evidence in its original form.

## 2.3. Keyframe Selection for Long Videos

Keyframe selection has attracted considerable attention in efficient video understanding, as it aims to identify a compact yet informative subset of frames that preserves salient visual content for downstream reasoning. Early efforts mainly focus on task-supervised selector learning for conventional video models, where the selector is optimized together with downstream recognition objectives [8, 15, 51, 52, 69]. More recently, with the rise of MLLMs, trainingbased keyframe selection has evolved into a new paradigm that leverages supervision distilled from large models rather than relying solely on task labels. Representative works include Frame-Voyager [59], which trains a query-aware frame selector under the supervision of a pretrained MLLM, and M-LLM video frame selector [13], which supervises a lightweight selector with MLLM-derived single-frame relevance and multi-frame complementarity signals. In parallel, training-free methods have recently gained substantial attention, as they avoid additional selector training and can be readily integrated with frozen MLLMs. These approaches typically construct frame subsets at inference time using pretrained vision-language representations together with lightweight scoring strategies. Along this line, MDP3 [41] formulates frame selection as a query-conditioned listwise sequential selection problem, BOLT [31] explores inference-time query-guided frame selection, and FOCUS [72] proposes a training-free exploration-based strategy that progressively locates informative regions before selecting keyframes. However, most existing methods still emphasize frame-wise relevance, often failing to preserve the semantic completeness of the selected subset. Our work, MarKey, addresses this issue by selecting frames based on contextaware marginal utility, yielding subsets that are more complete and less redundant.

## 3. Methods

## 3.1. Problem Formulation

Keyframe Selection. MLLMs are fundamentally constrained by a limited visual context budget, which makes it impractical to process all frames of a long video simultaneously. For long-form videos, directly encoding the full visual stream incurs prohibitive token and memory costs, while naive temporal subsampling may miss query-relevant moments that are sparse in time but crucial for downstream reasoning. Therefore, long-video understanding typically requires selecting a compact subset of frames that preserves the visual evidence most useful for answering the query. Formally, given a long video $\mathcal { V } = ( f _ { 1 } , f _ { 2 } , \ldots , f _ { N } )$ with N temporally ordered frames and a textual query q, the goal of query-guided keyframe selection is to choose a compact subset of K frames, such that S preserves the visual evidence most relevant to downstream reasoning. The selected subset is expected to form a compact and informative representation of the original video under a strict frame budget.

![](images/518584eb3da47a8fe57699f3b597bd9f88880cf6f811f422da677cb59805f628.jpg)  
Figure 2. An overview of our proposed MarKey framework. The pipeline consists of two main stages: (1) Preliminary Preparation: Given a long video V and a text query q, we first construct a representative anchor subset to summarize the global visual content of the video, and encode the anchor subset, candidate frames, and the query into a shared embedding space. (2) Greedy Frame Selection with Sliding Window: at each step, a bounded context window is formed from previously selected frames, and every remaining candidate is evaluated by a contextual utility that combines query relevance, coverage gain, and redundancy penalty. These three terms are weighted and aggregated, and the candidate with the maximum utility is selected as the next keyframe. Repeating this process until the frame budget K is reached yields a compact subset that is query-relevant and visually representative.

$$
S = \{ f _ { i _ { 1 } } , f _ { i _ { 2 } } , \ldots , f _ { i _ { K } } \} \subseteq \mathcal { V } , \qquad 1 \leq K \ll N .\tag{1}
$$

Optimization Objective. Given the candidate frame set defined above, our goal is to select a subset of K frames that maximizes its utility for downstream query-guided reasoning. We formulate query-guided keyframe selection as the following subset optimization problem:

$$
\mathcal { S } ^ { * } = \arg \operatorname* { m a x } _ { \mathcal { S } \subseteq \mathcal { V } , | \mathcal { S } | = K } U ( \mathcal { S } ; q ) ,\tag{2}
$$

where $U ( S ; q )$ measures the overall quality of a selected frame subset for answering the query.

## 3.2. Context-Aware Marginal Utility

To optimize the subset objective in Eq. (2), we construct the selected subset progressively and evaluate each candidate frame by the additional utility it contributes to the current partial selection. Formally, given the partial selection $S _ { t - 1 }$ at step t, the ideal contextual contribution of candidate frame $f _ { i }$ is defined as

$$
\Delta _ { i } ^ { ( t ) } = U ( S _ { t - 1 } \cup \{ f _ { i } \} ; q ) - U ( S _ { t - 1 } ; q ) .\tag{3}
$$

This context-dependent formulation encourages the selection of frames that provide complementary evidence beyond the current subset, instead of repeatedly favoring individually relevant but redundant frames. Directly optimizing Eq. (3) is computationally expensive for long videos, because it requires repeatedly evaluating subset-level utility during selection. We therefore approximate this contextual marginal utility with a tractable surrogate composed of three complementary terms: query relevance, coverage gain, and redundancy suppression. Together, these terms encourage the selector to retain frames that are relevant to the query, improve the representational completeness of the current subset, and avoid repeated evidence.

Query Relevance. To quantify the semantic relevance of each frame to the query, a pretrained vision-language encoder, such as CLIP [37], is used to extract the visual embedding of each candidate frame and the textual embedding of the query. Let $\mathbf { g } _ { i }$ denote the visual embedding of frame $f _ { i } ,$ and let $\mathbf { q }$ denote the textual embedding of query $q .$ After $\ell _ { 2 }$ normalization, the query relevance score is computed as

$$
r _ { i } = \cos ( \mathbf { g } _ { i } , \mathbf { q } ) = { \frac { \mathbf { g } _ { i } ^ { \top } \mathbf { q } } { \| \mathbf { g } _ { i } \| _ { 2 } \| \mathbf { q } \| _ { 2 } } } , \qquad i = 1 , \ldots , N .\tag{4}
$$

This term serves as the primary query-conditioned matching signal, encouraging the selector to retain frames that are semantically aligned with the user’s intent.

Coverage Gain. A high-quality keyframe subset should not only be relevant to the query, but also preserve sufficiently broad visual evidence from the original video. In other words, the selected frames should provide strong representational coverage, so that the final subset can serve as a compact yet faithful surrogate of the full video for downstream reasoning. However, many existing query-guided keyframe selection methods primarily emphasize frame-query relevance, while paying limited attention to whether the selected frames collectively cover the broader visual content of the video. As a result, the resulting subset may be locally relevant but globally incomplete, leaving potentially useful visual evidence insufficiently represented.

To address this limitation, we explicitly model the marginal coverage gain of each candidate with respect to the current partial selection. The goal is to measure how much additional visual coverage a candidate provides beyond what has already been captured by the selected subset. A natural formulation is to define this contribution over the full candidate set. Specifically, let $s _ { u j }$ denote the cosine similarity between frames $f _ { u }$ and $f _ { j }$ computed from normalized global visual embeddings, where u indexes a frame in the full candidate pool and $j$ indexes a selected frame in the current subset $S _ { t - 1 }$ . The ideal representational coverage of a selected subset $S _ { t - 1 }$ is defined as

$$
F _ { \mathrm { f u l l } } ( S _ { t - 1 } ) = \frac { 1 } { N } \sum _ { u = 1 } ^ { N } \operatorname* { m a x } _ { j \in { \mathscr { S } } _ { t - 1 } } s _ { u j } ,\tag{5}
$$

which measures how well the frames in $S _ { t - 1 }$ collectively represent the visual content of the entire candidate pool.

From a set-level perspective, the term ma $\mathbf { X } _ { j \in S _ { t - 1 } } s _ { u j }$ indicates how well frame $f _ { u }$ is covered by the selected subset $S _ { t - 1 }$ . Thus, Eq. (5) evaluates the representational completeness of the selected subset as a whole, rather than the quality of individual frames in isolation. Given the current partial selection $S _ { t - 1 }$ , we define the current coverage state of frame $f _ { u }$ as

$$
m _ { u } ^ { ( t - 1 ) } = \left\{ \begin{array} { l l } { 0 , } & { S _ { t - 1 } = \emptyset , } \\ { \operatorname* { m a x } _ { j \in S _ { t - 1 } } \boldsymbol { s } _ { u j } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{6}
$$

which measures the degree to which $f _ { u }$ has already been represented by the currently selected subset. Under this formulation, the ideal marginal coverage contribution of candidate frame $f _ { i }$ is

$$
\kappa _ { i , \mathrm { f u l l } } ^ { ( t ) } = F _ { \mathrm { f u l l } } \big ( S _ { t - 1 } \cup \{ f _ { i } \} \big ) - F _ { \mathrm { f u l l } } \big ( S _ { t - 1 } \big ) .\tag{7}
$$

Equivalently, it can be written as

$$
\kappa _ { i , \mathrm { f u l l } } ^ { ( t ) } = \frac { 1 } { N } \sum _ { u = 1 } ^ { N } \operatorname* { m a x } \bigl ( s _ { u i } - m _ { u } ^ { ( t - 1 ) } , 0 \bigr ) ,\tag{8}
$$

which explicitly quantifies how much additional coverage candidate frame $f _ { i }$ contributes beyond the current selection context.

Redundancy Penalty. Coverage gain rewards candidates that complement the current subset, but it does not explicitly discourage repeated evidence. This limitation is particularly important in long videos, where temporally adjacent segments, repetitive scenes, and recurring viewpoints often produce highly similar frames. Under a strict frame budget, a candidate frame is of limited utility when its content largely overlaps with the current subset and offers little additional evidence for query-guided reasoning, even if it is individually highly relevant. To explicitly suppress repeated evidence, we introduce a context-dependent redundancy penalty. Unlike coverage gain, which rewards candidates for improving subset completeness, the redundancy term penalizes candidates whose content is already well represented by the current selection context.

Formally, given the partial selection $S _ { t - 1 }$ at step $t ,$ we define the redundancy penalty of candidate frame $f _ { i }$ as

$$
\rho _ { i } ^ { ( t ) } = \left\{ { \begin{array} { l l } { 0 , } & { S _ { t - 1 } = \emptyset , } \\ { \operatorname* { m a x } _ { j \in S _ { t - 1 } } \cos ( \mathbf { g } _ { i } , \mathbf { g } _ { j } ) , } & { { \mathrm { o t h e r w i s e , } } } \end{array} } \right.\tag{9}
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity between normalized global visual embeddings. A larger value of $\rho _ { i } ^ { ( t ) }$ indicates that candidate frame $f _ { i }$ overlaps more strongly with the current subset and is therefore less likely to contribute new evidence. By penalizing such overlap, this term suppresses redundant selections and encourages a more compact and informative keyframe subset.

Surrogate Marginal Utility. Following the contextual marginal-utility formulation in Eq. (3), we approximate the marginal contribution of each candidate frame with a tractable surrogate that combines query relevance, anchorbased coverage gain, and redundancy suppression. Specifically, for candidate frame $f _ { i }$ at step t, we define the surrogate marginal utility as

Algorithm 1 Greedy Keyframe Selection with Anchor-  
Based Coverage   
Require: Candidate frame set $\mathcal { V } = \{ f _ { i } \} _ { i = 1 } ^ { N } ,$ query $q ,$ frame bud  
get $K ,$ , representative anchor subset A, window size W   
Ensure: Selected subset $\scriptstyle { S _ { K } }$   
1: Encode candidate frames $\{ f _ { i } \} _ { i = 1 } ^ { N }$ into normalized visual em  
beddings $\{ \mathbf { g } _ { i } \} _ { i = 1 } ^ { N }$ , and encode query q into q   
2: Compute query relevance scores $r _ { i } = \cos ( \mathbf { g } _ { i } , \mathbf { q } )$ for all can  
didates   
3: Compute pairwise similarities $s _ { u i }$ between anchor frames   
$f _ { u } \in \mathcal { A }$ and candidate frames $f _ { i } \in \nu$   
4: Initialize $S _ { 0 }  \emptyset$ and $m _ { u } ^ { ( 0 ) } \gets 0$ for all $u \in { \mathcal { A } }$   
5: for $t = 1$ to K do   
6: Construct active context window $\boldsymbol { \mathcal { C } } _ { t - 1 } \subseteq \boldsymbol { \mathcal { S } } _ { t - 1 }$ as the con  
text for evaluation   
7: for each candidate $f _ { i } \in \mathcal { V } \setminus { S } _ { t - 1 }$ do   
8: Compute redundancy penalty $\rho _ { i } ^ { ( t ) }$ by Eq. (9)   
9: Compute anchor-based coverage gain $\kappa _ { i } ^ { ( t ) }$ by Eq. (12)   
10: Compute surrogate marginal utility $u _ { i } ^ { ( t ) }$ by Eq. (10)   
11: end for   
12: Select   
$f _ { i _ { t } } ^ { * } = \arg \operatorname* { m a x } _ { f _ { i } \in \mathcal { V } \backslash \mathcal { S } _ { t - 1 } } u ^ { ( t ) } ( f _ { i } ) .$   
13: Update selected subset   
$S _ { t } \gets S _ { t - 1 } \cup \{ f _ { i _ { t } } ^ { * } \}$   
14: for each anchor $f _ { u } \in \mathcal { A }$ do   
15: Update anchor coverage state   
$m _ { u } ^ { ( t ) } \gets \operatorname* { m a x } \bigl ( m _ { u } ^ { ( t - 1 ) } , s _ { u i _ { t } ^ { * } } \bigr )$   
16: end for   
17: end for   
18: return $\scriptstyle { S _ { K } }$

$$
u ^ { ( t ) } ( f _ { i } ) = \alpha r _ { i } + \delta \kappa _ { i } ^ { ( t ) } - \lambda \rho _ { i } ^ { ( t ) } ,\tag{10}
$$

where α, δ, and λ balance the contributions of query relevance, coverage improvement, and redundancy suppression, respectively. Eq. (10) serves as a tractable estimate of the ideal marginal utility of frame $f _ { i }$ under the current partial selection.

## 3.3. Optimization and Greedy Selection

Coverage Gain Optimization. Directly evaluating Eq. (8) over the full candidate pool is computationally expensive for long videos, since the marginal coverage gain of every remaining candidate must be repeatedly computed during greedy selection. To make this computation tractable, we approximate the full candidate set by a much smaller representative anchor subset ${ \mathcal { A } } \subseteq { \mathcal { V } } ,$ , where $| { \mathcal { A } } | \ll | \nu |$ . Following the temporal locality observation that nearby video frames often exhibit strong semantic similarity [23], we construct A through temporal-relevance stratified sampling. Specifically, each video is divided into non-overlapping 10- second clips, and frames are uniformly sampled at 1 FPS within each clip. For every clip, the sampled frames are ranked according to their query-frame similarity scores and partitioned into high-, medium-, and low-relevance strata. Anchors are then sampled from all three strata under a fixed global anchor budget, and the clip-level anchors are merged across the entire video to form the final anchor subset A. In our implementation, the anchor budget is set to 648 frames; when fewer frames are available, all eligible frames are retained. Under this approximation, the anchor-based coverage objective is defined as

$$
F ( S _ { t - 1 } ) = \frac { 1 } { | \mathcal { A } | } \sum _ { u \in \mathcal { A } } \operatorname* { m a x } _ { j \in \mathcal { S } _ { t - 1 } } s _ { u j } ,\tag{11}
$$

and the corresponding anchor-based marginal coverage gain of candidate frame $f _ { i }$ becomes

$$
\kappa _ { i } ^ { ( t ) } = \frac { 1 } { \left| \mathcal { A } \right| } \sum _ { u \in \mathcal { A } } \operatorname* { m a x } \bigl ( s _ { u i } - m _ { u } ^ { ( t - 1 ) } , 0 \bigr ) .\tag{12}
$$

Here, $m _ { u } ^ { ( t - 1 ) }$ is defined in the same way as Eq. (6), except that u is restricted to the anchor subset ${ \mathcal { A } } .$ . This anchorbased formulation preserves the set-level representational objective of Eq. (5), while substantially reducing the cost of marginal gain estimation.

Sliding-Window Context Approximation. Although the full selected subset $\boldsymbol { S } _ { t - 1 }$ defines the context at step t, using all previously selected frames for context-sensitive scoring becomes increasingly expensive as selection proceeds. Moreover, evaluating each candidate against the entire selection history may impose excessive historical bias, causing later candidates to be overly suppressed. To alleviate this issue, we maintain a bounded active context window

$$
\begin{array} { r l r l } { \mathcal { C } _ { t - 1 } \subseteq S _ { t - 1 } , } & { { } } & { | { \mathcal { C } } _ { t - 1 } | \leq W , } \end{array}\tag{13}
$$

where $W$ denotes the window size. In practice, $\mathcal { C } _ { t - 1 }$ is used as a tractable approximation of the current selection context when computing context-dependent utility terms at step t. Accordingly, both coverage gain and redundancy penalty are evaluated with respect to this active context window during greedy selection.

Greedy Selection. Based on the surrogate marginal utility defined in Eq. (10), we adopt a greedy strategy that constructs the selected subset incrementally. Figure 2 provides an overview of this greedy selection procedure. Starting from an empty set ${ \cal { S } } _ { 0 } = \emptyset$ , at each step t, we evaluate every remaining candidate using the contextual utility in Eq. (10) and select the one with the highest score:

Table 1. Overview of the six video understanding benchmarks used in our experiments.
<table><tr><td>Benchmark</td><td>Video Type</td><td>Dataset Scale</td><td># Eval. Samples</td><td>Answer Format</td><td>Metric</td></tr><tr><td>LongVideoBench</td><td>Multi-domain</td><td>3,763 videos</td><td>1,337</td><td>MCQ</td><td>Acc.</td></tr><tr><td>Video-MME</td><td>Multi-domain</td><td>900 videos</td><td>2,700</td><td>MCQ</td><td>Acc.</td></tr><tr><td>NExT-QA</td><td>Human activities</td><td>5,440 videos</td><td>8,576</td><td>MCQ</td><td>Acc.</td></tr><tr><td>EgoLifeQA</td><td>Egocentric daily life</td><td>~300 hours</td><td>3,000</td><td>MCQ</td><td>Acc.</td></tr><tr><td>YouCook2</td><td>Instructional cooking</td><td>2,000 videos</td><td>3,492</td><td>Open-ended</td><td>CIDEr</td></tr><tr><td>Video-TT</td><td>YouTube Shorts</td><td>1,000 videos</td><td>1,000</td><td>Open-ended</td><td>GPT Eval.</td></tr></table>

$$
f _ { i _ { t } } ^ { * } = \arg \operatorname* { m a x } _ { f _ { i } \in \mathcal { V } \backslash \mathcal { S } _ { t - 1 } } u ^ { ( t ) } ( f _ { i } ) .\tag{14}
$$

The selected subset is then updated as

$$
S _ { t } = S _ { t - 1 } \cup \{ f _ { i _ { t } } ^ { * } \} , \qquad t = 1 , \ldots , K .\tag{15}
$$

This process continues until the frame budget K is reached. At a high level, each greedy step selects the candidate with the largest estimated marginal contribution under the current context, thereby progressively constructing a subset that is query-relevant, semantically complete, and compact. The overall procedure is summarized in Alg. 1.

## 3.4. Discussion

Let N denote the number of candidate frames, K the target frame budget, A the number of representative anchors, and W the context-window size. Computing the coverage gain for all candidates over the entire video requires $\mathcal { O } ( N ^ { \bar { 2 } } )$ computation at each greedy step, leading to $\mathcal { O } ( K N ^ { 2 } )$ over K steps. Similarly, redundancy computation against all previously selected frames accumulates to $\mathcal { O } ( N K ^ { 2 } )$ . MarKey reduces these costs by approximating video-wide coverage with A representative anchors and restricting the context to at most W selected frames, where $A \ll N$ and $W < K$ As a result, the coverage and redundancy costs are reduced to O(KNA) and O(KNW), respectively. Therefore, the overall selection complexity is

$$
\mathcal { O } ( K N ( A + W ) ) .\tag{16}
$$

For fixed K, A, and W, this complexity scales linearly with the number of candidate frames N, making iterative subsetaware selection practical for long videos.

## 4. Experiments

## 4.1. Experimental Settings

Datasets. We evaluate MarKey on six benchmarks covering holistic video understanding, human-centric video understanding, and open-ended video understanding. The holistic video understanding benchmarks, LongVideoBench and

Video-MME, evaluate broad understanding of diverse longform video content. The human-centric benchmarks, NExT-QA and EgoLifeQA, focus on causal-temporal reasoning over human activities and long-horizon understanding of egocentric daily-life videos. The open-ended benchmarks, YouCook2 and Video-TT, require models to generate responses in natural language, evaluating the generalizability of MarKey beyond predefined answer choices. Table 1 summarizes the key statistics and evaluation settings of these six benchmarks.

• LongVideoBench [50] is a long-context video questionanswering benchmark containing videos with temporally aligned subtitles and questions across 17 fine-grained categories. In our experiments, we use its validation split, which consists of 1,337 multiple-choice questions involving referred-context reasoning and fine-grained information distributed throughout long video sequences.

• Video-MME [7] provides a broad evaluation of video understanding with 900 videos totaling 254 hours and 2,700 expert-annotated multiple-choice question-answer pairs. The videos cover six major domains and 30 subfields, with durations ranging from 11 seconds to one hour, and are further divided into short-, medium-, and long-video subsets.

• EgoLifeQA [56] is constructed from approximately 300 hours of continuous daily-life recordings collected from six participants living together for one week. It provides 3,000 long-context multiple-choice questions covering entities, events, habits, interpersonal relationships, and multi-step tasks across extended first-person video histories.

• NExT-QA [53] contains 5,440 videos of daily human activities and 47,692 multiple-choice questions. The questions cover causal, temporal, and descriptive understanding, requiring models to identify action dependencies, event order, and relevant interactions among objects and people.

• YouCook2 [70] is a large-scale instructional video benchmark containing 2,000 long, untrimmed videos from 89 cooking recipes. In our experiments, we use its openended setting, in which models generate natural-language descriptions of cooking procedures.

Table 2. Comparison of different approaches on holistic video understanding benchmarks. LongVideoBench (LVB) and Video-MME (V-MME) are evaluated using accuracy (%). AVG denotes the average accuracy across the two benchmarks. All Qwen3- VL-8B-based methods are evaluated using 32 input frames. Bold numbers indicate the best results among the Qwen3-VL-8B-based methods.
<table><tr><td>Method</td><td>LVB</td><td>V-MME</td><td>AVG</td></tr><tr><td>Reference MLLMs</td><td></td><td></td><td></td></tr><tr><td>GPT-4o [45]</td><td>66.7</td><td>71.9</td><td>69.3</td></tr><tr><td>Gemini-1.5-Pro [43]</td><td>64.0</td><td>75.0</td><td>69.5</td></tr><tr><td>LLaVA-Video [66] LLaVA-OneVision [19]</td><td>63.9 59.8</td><td>70.6 68.7</td><td>67.25 64.25</td></tr><tr><td>Qwen3-VL-235B-A22B [2]</td><td>65.6</td><td>79.0</td><td>72.30</td></tr><tr><td>Training-free frame selection with Qwen3-VL-8B</td><td></td><td></td><td></td></tr><tr><td>Qwen3-VL</td><td>60.40</td><td>64.30</td><td>62.35</td></tr><tr><td>Qwen3-VL w/ Top-K</td><td>62.15</td><td>64.41</td><td>63.28</td></tr><tr><td>Qwen3-VL w/ AKS [42]</td><td>61.18</td><td>65.74</td><td>63.46</td></tr><tr><td>Qwen3-VL w/ BOLT [31]</td><td>62.52</td><td>65.37</td><td>63.95</td></tr><tr><td>Qwen3-VL w/ FOCUS [72]</td><td>63.05</td><td>63.15</td><td>63.10</td></tr><tr><td>Qwen3-VL w/ DIG [24]</td><td>64.84</td><td></td><td></td></tr><tr><td></td><td>65.14</td><td>65.44</td><td>65.14</td></tr><tr><td>Qwen3-VL w/ AIR [73]</td><td></td><td>67.81</td><td>66.47</td></tr><tr><td>Qwen3-VL w/ MarKey</td><td>65.74</td><td>68.14</td><td>66.94</td></tr></table>

• Video-TT [67] consists of 1,000 short-form YouTube videos designed for open-ended video understanding. Each video is paired with one open-ended question and four adversarial questions, and we use its open-ended portion in our experiments.

Evaluated MLLMs and Metrics. We use Accuracy (%) as the primary evaluation metric for multiple-choice video question answering. For subjective open-ended benchmarks, we follow the original evaluation protocols of each dataset. Specifically, we use CIDEr as the evaluation metric on YouCook2, and adopt GPT-based evaluation on Video-TT, where the generated response is compared with the ground-truth textual answer to assess semantic correctness. Following prior studies [38, 44], we further employ Mean Pairwise Cosine Similarity (MPCS) to quantify the similarity among the selected frames. MPCS computes the average cosine similarity over all frame pairs within the selected set, where a lower value indicates less redundancy and greater diversity among the selected frames. Following prior studies on training-free keyframe selection, we primarily evaluate our method on the Qwen2.5-VL [3], Qwen3-VL [2], and MiniCPM-V-4 5 [60] series, covering different backbone families and model scales. To provide a broader view of performance, we also report results for a wider range of recent MLLMs on six different long video understanding datasets, including LLaVA-Video [66], LLaVA-OneVision [19], GPT-4o [45], and Gemini [43].

Table 3. Comparison of different approaches on human-centric video understanding benchmarks. EgoLifeQA and NExT-QA are evaluated using accuracy (%). AVG denotes the average accuracy across the two benchmarks. All Qwen3-VL-8B-based methods are evaluated using 32 input frames. Bold numbers indicate the best results among the Qwen3-VL-8B-based methods.
<table><tr><td>Method EgoLifeQA NExT-QA AVG</td></tr><tr><td>Reference MLLMs GPT-4o [45] 36.2</td></tr><tr><td>Gemini-1.5-Pro [43] 36.9 85.3 61.10</td></tr><tr><td>LLaVA-Video [66] 一 85.4 一</td></tr><tr><td>LLaVA-OneVision [19] 83.2 一 一 Qwen3-VL-235B-A22B [2] 一 83.3 一</td></tr><tr><td>Training-free frame selection with Qwen3-VL-8B</td></tr><tr><td>Qwen3-VL 32.67 82.86 57.77</td></tr><tr><td>Qwen3-VL w/ Top-K 32.67 82.89 57.78</td></tr><tr><td>Qwen3-VL w/ AKS [42] 30.69 82.99 56.84</td></tr><tr><td>Qwen3-VL w/ BOLT [31] 33.66 83.08 58.37</td></tr><tr><td>Qwen3-VL w/ FOCUS [72] 33.66 81.39 57.53</td></tr><tr><td>Qwen3-VL w/ DIG [24] 33.71 84.61</td></tr><tr><td>59.16 Qwen3-VL w/ AIR [73] 34.65 84.31 59.48</td></tr><tr><td>Qwen3-VL w/ MarKey 37.62 84.20 60.91</td></tr><tr><td></td></tr></table>

Compared Methods. We compare MarKey with several recent training-free keyframe selection methods, including AKS [42], BOLT [31], FOCUS [72], AIR [73], and DIG [24].

• AKS [42] performs adaptive keyframe selection by recursively partitioning a video according to frame-query similarity. Based on the relevance distribution within different temporal intervals, it dynamically determines the sampling granularity and selects representative frames from informative regions.

• BOLT [31] adopts a training-free query-guided sampling strategy based on frame-query similarity. It transforms the relevance scores into a probability distribution and samples keyframes accordingly, enabling the selection process to focus more on frames associated with the input query.

• FOCUS [72] formulates frame selection as a budgeted exploration problem. It progressively evaluates temporal regions and allocates additional sampling resources to promising segments, allowing informative evidence to be identified without densely processing the entire video.

• AIR [73] introduces an adaptive iterative reasoning framework for video frame selection. It first identifies potentially relevant temporal regions and then employs multimodal reasoning to iteratively refine the candidate frames, progressively narrowing the search toward useful visual evidence.

• DIG [24] adapts the frame selection process according to the characteristics of the input query. It dynamically adjusts the sampling strategy to identify informative visual evidence under different query conditions, providing a flexible training-free solution for long-video understanding.

Table 4. Comparison of different approaches on open-ended video understanding benchmarks. CIDEr is reported for YouCook2, while the GPT-based evaluation score is reported for Video-TT. AVG denotes the average score across the two benchmarks. All Qwen3-VL-8B-based methods are evaluated using 32 input frames. Bold numbers indicate the best results among the Qwen3- VL-8B-based methods.
<table><tr><td>Method</td><td>YouCook2 Video-TT AVG</td><td></td><td></td></tr><tr><td>Reference MLLMs GPT-4o [45]</td><td></td><td>36.6</td><td>49.60</td></tr><tr><td>Gemini-1.5-Pro [43] LLaVA-Video [66]</td><td>70.4 一</td><td>28.8 24.4</td><td>一</td></tr><tr><td colspan="4">Training-free frame selection (Qwen3-VL-8B)</td></tr><tr><td>Qwen3-VL</td><td>42.02</td><td>24.10</td><td>33.06</td></tr><tr><td>Qwen3-VL w/ Top-K</td><td>44.95</td><td>24.70</td><td>34.83</td></tr><tr><td>Qwen3-VL w/ AKS [42]</td><td>46.45</td><td>25.10</td><td>35.78</td></tr><tr><td>Qwen3-VL w/ BOLT [31]</td><td>45.10</td><td>27.40</td><td>36.25</td></tr><tr><td>Qwen3-VL w/ FOCUS [72]</td><td>43.45</td><td>26.20</td><td>34.83</td></tr><tr><td>Qwen3-VL w/ DIG [24]</td><td>46.88</td><td>32.10</td><td>39.49</td></tr><tr><td>Qwen3-VL w/ AIR [73]</td><td>46.37</td><td>26.50</td><td>36.43</td></tr><tr><td>Qwen3-VL w/ MarKey</td><td>48.90</td><td>29.70</td><td>39.30</td></tr></table>

Implementation Details. We use the pretrained CLIP-L/14 [37] model to extract both visual embeddings for video frames and textual embeddings for input queries. For the Top-K baseline, we rank the same candidate frames as MarKey by their CLIP-L/14 query-frame similarity and select the highest-scoring K frames. For evaluation, we utilize the LMMS-Eval library [63] to report accuracy on multiplechoice video question answering benchmarks. We set the number of anchors to 648 for anchor-based coverage estimation. We set α = 0.7, δ = 0.2, and λ = 0.1 in all experiments. We set the decoding temperature to 0 for all evaluations. All experiments are conducted with frame budgets K ∈ {16, 32, 64, 128} on NVIDIA RTX A6000 GPUs with 48 GB of memory.

## 4.2. Main Results

To comprehensively evaluate the effectiveness and generalizability of MarKey, we compare it with representative training-free frame selection methods, including Top-K, AKS, BOLT, FOCUS, DIG, and AIR, under the same Qwen3-VL-8B backbone and a fixed input budget of 32 frames. For a more structured comparison, we organize the six benchmarks into three complementary categories: holistic video understanding, specialized human-centric video understanding, and open-ended video understanding.

Table 5. Generalization of different frame selection methods across multiple MLLM backbones on LongVideoBench and Video-MME. We evaluate the methods with Qwen2.5-VL-7B and MiniCPM-V-4.5-8B to examine whether their effectiveness can consistently transfer across different model architectures. All frame selection methods within each backbone group are evaluated using 32 input frames. AVG denotes the average accuracy across LVB and V-MME. Bold numbers indicate the best results within each backbone.
<table><tr><td>Method</td><td>LVB</td><td>V-MME</td><td>AVG</td></tr><tr><td>Reference MLLMs Gemini-1.5-Pro [43]</td><td>64.0</td><td>75.0</td><td>69.50</td></tr><tr><td>InternVL3.5-241B-A28B [46] Qwen2.5-VL [3]</td><td>67.1 59.31</td><td>72.9 62.20 63.51</td><td>70.00 60.76 61.41</td></tr><tr><td>Qwen2.5-VL w/ BOLT [31] Qwen2.5-VL w/ FOCUS [72] Qwen2.5-VL w/ MarKey MiniCPM-V-4.5 MiniCPM-V-4.5 w/ AKS [42]</td><td>60.00 61.48 63.94</td><td>63.67 62.37 65.03</td><td>61.84 61.93 64.49</td></tr><tr><td>MiniCPM-V-4.5 w/ BOLT [31] MiniCPM-V-4.5 w/ FOCUS [72] MiniCPM-V-4.5 w/ MarKey</td><td>60.61 60.81 60.94 61.56 64.15</td><td>63.88 64.81 64.40 64.11</td><td>62.25 62.81 62.67</td></tr></table>

Holistic Video Understanding. As shown in Table 2, MarKey achieves the best performance among all Qwen3- VL-8B-based methods on both LongVideoBench and Video-MME, with an average accuracy of 66.94%. First, MarKey outperforms Top-K by 3.59 and 3.73 points on LongVideoBench and Video-MME, respectively. Since Top-K and MarKey use the same CLIP-based query relevance, this comparison directly shows that frame-wise relevance alone is insufficient. Accounting for the complementarity among selected frames leads to consistently better use of the same frame budget. Second, the relative performance of existing selectors varies across the two benchmarks. For example, FOCUS is more competitive on LongVideoBench, while AKS shows a stronger relative advantage on Video-MME. MarKey achieves the best performance on both benchmarks, indicating that its effectiveness is less dependent on a particular benchmark or evidence distribution. Third, MarKey substantially improves the competitiveness of a relatively small backbone. On LongVideoBench, Qwen3-VL-8B with MarKey reaches 65.74%, slightly surpassing Qwen3-VL-235B-A22B at 65.60% under the same 32-frame budget, despite the nearly 30× difference in model size. This result highlights the importance of input

![](images/86eedfee5fb2eed3c3bbcb0d12328f2c42e07c394876ac039e0a6d1e5bd07bbc.jpg)  
Figure 3. Comparison of different LVLM scales using Qwen3- VL on LongVideoBench. Accuracy (%) is reported. Results show performance with K=16 frames (top) and K=32 frames (bottom). evidence quality in long-video understanding. Under a constrained visual budget, allocating frames more effectively can yield gains comparable to those obtained by substantially increasing model capacity, making frame selection an effective way to improve performance without scaling the downstream MLLM.

Human-centric Video Understanding. As shown in Table 3, MarKey achieves the highest average accuracy of 60.91% among the Qwen3-VL-8B-based methods, demonstrating strong performance across two substantially different human-centric video understanding settings. First, the improvement is particularly pronounced on EgoLifeQA. MarKey improves the Qwen3-VL baseline from 32.67% to 37.62%, yielding a gain of 4.95 points, and exceeds the strongest competing selector, AIR, by 2.97 points. EgoLifeQA requires reasoning over extended first-person video histories, where relevant evidence may occur at distant moments and multiple observations may need to be combined. The larger gain on this benchmark therefore supports the importance of preserving complementary evidence under a limited frame budget. Second, MarKey achieves strong overall performance across both humancentric benchmarks, while the relative improvements differ between them. This result reflects the different characteristics and challenges of the two benchmarks, and further demonstrates the robustness of MarKey across diverse human-centric video understanding scenarios.

Open-ended Video Understanding. Table 4 further evaluates whether the benefits of MarKey extend beyond multiple-choice question answering to settings that require models to generate natural-language responses. First, MarKey consistently improves the Qwen3-VL baseline on both open-ended benchmarks, increasing the YouCook2 CIDEr score from 42.02 to 48.90 and the Video-TT score from 24.10 to 29.30. The gains of 6.88 and 5.20 points, respectively, show that selecting more informative and complementary frames also benefits free-form generation, where the model cannot rely on predefined answer candidates. Second, MarKey achieves the best performance on YouCook2, outperforming the strongest competing method, DIG, by 2.02 points. This result is particularly relevant for instructional videos, where generating an accurate description requires sufficient coverage of multiple procedural steps rather than identifying only a single salient moment. The improvement therefore aligns well with MarKey’s explicit modeling of coverage gain and redundancy. Third, MarKey shows a larger advantage on YouCook2 than on Video-TT. Relative to Video-TT, YouCook2 contains longer and more procedurally structured videos, where relevant evidence can be distributed across multiple stages. This result further highlights the strength of MarKey in preserving complementary information across extended video content.

![](images/3cccdabf5323109b81d348898492d5d2f693bd6997e3cced904adca6ccb72f64.jpg)  
Figure 4. Performance comparison across different frame budgets K on LongVideoBench. MarKey consistently outperforms uniform sampling across all LVLMs and budget settings.

## 4.3. Effect of Different Backbones

To examine whether the effectiveness of MarKey depends on a specific MLLM backbone, we further evaluate it using Qwen2.5-VL-7B and MiniCPM-V-4.5-8B on LongVideoBench and Video-MME. All methods use the same input budget of 32 frames, and the results are reported in Table 5. First, MarKey consistently achieves the best performance across both backbones and benchmarks. Compared with uniform sampling, it improves Qwen2.5-VL by 4.63 points on LongVideoBench and 2.83 points on Video-MME, while improving MiniCPM-V-4.5 by 3.54 and 2.45 points, respectively. Second, compared with the strongest competing frame selector, MarKey further improves performance by 2.46 points on LongVideoBench and 1.36 points on Video-MME under Qwen2.5-VL, and by 2.59 and 1.52 points under MiniCPM-V-4.5, respectively. These results show that the effectiveness of MarKey generalizes across different MLLM backbones rather than being specific to Qwen3-VL. Third, we observe consistently larger improvements on LongVideoBench than on Video-MME. This suggests that subset-aware selection is particularly beneficial when useful evidence is distributed over longer temporal contexts, where effective evidence allocation becomes more important under a fixed frame budget.

Table 6. Component ablation study on LongVideoBench and Video-MME with Qwen3-VL-8B (K=32). All scores are accuracy (%).
<table><tr><td colspan="2">Components</td><td rowspan="2">LVB V-MME</td></tr><tr><td>Query</td><td>Coverage Redundancy</td></tr><tr><td>X</td><td>X X</td><td>60.40 64.30</td></tr><tr><td>√</td><td>X X</td><td>62.15 64.41</td></tr><tr><td>X</td><td>√ X</td><td>60.87 65.11</td></tr><tr><td>√</td><td>√ X</td><td>64.17 65.40</td></tr><tr><td>√</td><td>× √</td><td>64.02 66.07</td></tr><tr><td>X</td><td>√ √</td><td>61.01 64.81</td></tr><tr><td>V</td><td>√ V</td><td>65.74 68.14</td></tr></table>

## 4.4. Effect of Different Model Scales

To further examine the scalability of our method, we evaluate it on Qwen3-VL models at different scales, including 2B, 4B, 8B and 32B, and report the results in Figure 3. Several observations can be drawn from the results. First, our method consistently improves performance across all model scales on LongVideoBench, suggesting that its effectiveness is stable and not tied to a specific parameter budget. Second, we observe an interesting phenomenon under uniform sampling: Qwen3-VL-4B slightly outperforms Qwen3-VL-8B in the baseline setting. However, after replacing uniform sampling with our keyframe selection strategy, the 8B model exhibits the largest improvement among the tested scales. We conjecture that, under a fixed frame budget, the performance of larger models is more sensitive to the quality of visual inputs. When uniformly sampled frames contain redundant or less informative content, the stronger reasoning capacity of the larger model cannot be fully utilized. In contrast, once more informative keyframes are provided, the 8B model is better able to exploit its higher capacity for temporal evidence aggregation and questionconditioned reasoning. This suggests that effective frame selection is particularly important for unlocking the potential of larger MLLMs.

## 4.5. Effect of Keyframe Number

To examine the impact of the number of input keyframes, we evaluate our method under different frame budgets, including 16, 32, 64, and 128 frames. The results are shown in Figure 4. Several observations can be drawn from the results. First, our method consistently outperforms uniform sampling across all frame budgets on both Qwen2.5-VL and Qwen3-VL, indicating that its effectiveness is robust to different input lengths rather than relying on a specific frame budget. Second, our method remains effective even with a large number of input frames. Notably, under the 128-frame setting, our method achieves improvements of 3.81 and 4.20 percentage points over uniform sampling on

![](images/f4277b2a57ae22d4339364930518281fd724d43204f30c46781424c8b276f91f.jpg)  
Figure 5. Ablation on different VLMs for query-frame similarity scoring on LongVideoBench and Video-MME with Qwen3- VL-8B (K=32). Accuracy scores (%) are reported. CLIP-L/14 achieves the best performance on both Video-MME and LongVideoBench. Therefore, we use CLIP-L/14 as the default scorer.

Qwen2.5-VL-7B and Qwen3-VL-8B, respectively. These results demonstrate that increasing the visual budget alone is insufficient to fully exploit long videos, while selecting informative keyframes remains essential for effective video understanding.

## 4.6. Ablation Study

To investigate the contribution of each component in our proposed method, we conduct five different ablation variants, including individually removing Query Relevance, Visual Coverage, and Redundancy Penalty, as well as retaining only Query Relevance or Visual Coverage. We further compare these different variants with both the complete model and the uniform sampling baseline under the same experimental setting. The corresponding results are summarized in Table 6. Based on these experimental results, several important observations can be drawn. First, removing any individual component consistently leads to noticeable performance degradation, and none of the ablated variants can match the full model. This verifies that the performance gain of MarKey does not come from a single dominant design choice, but rather from the effective interaction of all three components. More importantly, Query Relevance is shown to be the most critical component among the three components. Removing it causes the score to substantially drop from 65.74% to 61.01% on LongVideoBench, corresponding to a considerable decrease of 4.73 points and making the resulting performance only marginally better than uniform sampling (60.40%). This result clearly suggests that query-aware relevance modeling provides the fundamental and essential signal for effective keyframe selection, since it enables the selector to more effectively focus on visual evidence that is directly relevant and useful for accurately answering the given question. Furthermore, the ablation results highlight the complementary effect of the three components. Using Query Relevance or Visual Coverage alone yields only limited improvements, indicating that either signal in isolation is insufficient for reliable keyframe selection. In contrast, combining relevance with contextual coverage and redundancy modeling consistently leads to stronger performance. This suggests that task-relevant evidence can be more effectively identified when candidate frames are evaluated within a sufficiently informative selection context, where both complementary content and repeated evidence are explicitly considered.

Table 7. Ablation of utility weights on LongVideoBench and Video-MME with Qwen3-VL-8B (K=32). Accuracy (%) is reported. $\alpha , \delta ,$ and λ denote the weights for query relevance, visual coverage, and redundancy penalty, respectively.
<table><tr><td>α</td><td>δ λ</td><td>LVB</td><td>V-MME</td></tr><tr><td>Uniform Sampling</td><td></td><td>60.40</td><td>64.30</td></tr><tr><td>0.90 0.05</td><td>0.05</td><td>64.54</td><td>66.77</td></tr><tr><td>0.85 0.10</td><td>0.05</td><td>64.77</td><td>66.81</td></tr><tr><td>0.80 0.15</td><td>0.05</td><td>64.69</td><td>66.70</td></tr><tr><td>0.75 0.15</td><td>0.10</td><td>65.07</td><td>67.40</td></tr><tr><td>0.60</td><td>0.20 0.20</td><td>63.94</td><td>67.25</td></tr><tr><td>0.55</td><td>0.25 0.20</td><td>64.09</td><td>66.96</td></tr><tr><td>0.70</td><td>0.20 0.10</td><td>65.74</td><td>68.14</td></tr></table>

## 4.7. Effect of Different VLM Scorer

To evaluate the impact of the VLM scorer used for computing query-frame relevance, we compare several representative vision-language models, including CLIP-B/32 [37], CLIP-L/14 [37], LongCLIP-B [64], BLIP2-ITM-ViT-L [22], and BLIP2-ITM-ViT-g [22]. The results are summarized in Figure 5. Several observations can be drawn from the results. First, the choice of VLM scorer has a clear impact on the final performance, indicating that the quality of query-frame relevance estimation is important for effective frame selection. Second, all tested VLM scorers consistently outperform uniform sampling on both LongVideoBench and Video-MME, while the magnitude of improvement varies substantially across different scorers. In particular, BLIP2-based scorers provide relatively modest gains, whereas LongCLIP-B and the standard CLIP variants yield considerably larger improvements. This suggests that different pretrained vision-language models exhibit different levels of effectiveness in estimating queryframe relevance, further highlighting the importance of selecting an appropriate scorer for query-aware frame selection. Third, CLIP-L/14 delivers the best overall results, achieving 65.74% on LongVideoBench and 68.14% on Video-MME, while CLIP-B/32 also remains highly competitive. This trend suggests that CLIP-based scorers provide stronger and more reliable cross-modal relevance signals for our method than the BLIP2 variants in this work.

![](images/0e23084681a687a26134d4d4fd759f6ee5397c116afdbf6104c40aa746c18df4.jpg)

![](images/a009aafb83a5673ac96d20dd43fa00d2eacb2d491e6fd0daee6237229543d5dc.jpg)  
Figure 6. Ablation studies of the anchor set size and active context window size on LongVideoBench and Video-MME with Qwen3- VL-8B (K=32). Accuracy (%) is reported. (a) Effect of the anchor set size, where A denotes the number of anchors used for coverage estimation. (b) Effect of the active context window size, where W denotes the number of selected frames retained in the active context during greedy selection.

## 4.8. Analysis of Hyperparameters

Utility Coefficients. To explore the sensitivity of MarKey to the weighting coefficients in the utility function, we vary the values of $\alpha , \delta ,$ and λ, which control query relevance, visual coverage, and redundancy penalty, respectively. The results are reported in Table 7. Several observations can be drawn from the results. First, all tested weight configurations consistently outperform the uniform sampling baseline on both LongVideoBench and Video-MME, demonstrating that the proposed utility design is effective and robust across a reasonable range of coefficient choices. Second, our default setting, α=0.70, δ=0.20, and λ=0.10, achieves the best overall trade-off across the two benchmarks, obtaining the highest score on LongVideoBench (65.74%) and Video-MME (68.14%). Third, the results highlight that query relevance, visual coverage, and redundancy suppression are complementary to each other in the proposed utility design. Their joint modeling enables MarKey to select subsets that are more query-relevant, more complete, and less redundant, which further supports our formulation of keyframe selection as a context-aware subset selection problem.

Anchor Set Size. To investigate the effect of the anchor set size, we vary A while keeping all other settings fixed. As shown in Figure 6(a), performance improves as A increases from 128 to 648, indicating that a larger anchor set provides a more reliable approximation of global visual coverage. However, further increasing A to 776 slightly reduces the accuracy on both benchmarks. This result suggests that excessively dense anchors may introduce redundant coverage information without providing additional benefit. We therefore set A = 648 as the default, which achieves the best performance while avoiding unnecessary computation.

![](images/6a624aedb654145cf0a299ef4f09e4abf1523e591499b70a14f8148e30c4c42d.jpg)  
Figure 7. Fine-grained accuracy (%) comparison across eight question types on NExT-QA using two vision-language models with 32 input frames: (a) Qwen2.5-VL and (b) Qwen3-VL-8B. CW and CH denote Causal Why and Causal How; TN, TC, and TP denote Temporal Next, Temporal Current, and Temporal Previous; DC, DL, and DO denote Descriptive Count, Descriptive Location, and Descriptive Other, respectively.

Window Size. To explore the effect of the active context window size, we vary W while keeping all other settings fixed. The results are reported in Figure 6(b). Several observations can be drawn from the results. First, the performance improves generally as the window size increases, indicating that a larger active context is beneficial for modeling the interaction between the current candidate and the previously selected frames. On LongVideoBench, the score improves from 64.17% at W=4 to 65.74% at the default setting W=28. A similar trend is observed on Video-MME, where the score rises from 65.96% to 68.14%. These results suggest that incorporating a broader selection context helps estimate the contextual utility of candidate frames more accurately. Second, enlarging the window size also increases the computational cost of context-dependent scoring. At the same time, the performance improvement becomes less substantial when the window size is already large. This trend suggests that, in practice, using a moderately sized active context is often sufficient to capture most of the useful contextual information for candidate evaluation. In this way, MarKey can maintain strong performance while reducing the cost of context-dependent scoring. Such a design is particularly appealing in more complex scenarios, where selecting a larger number of keyframes would otherwise make full-context evaluation increasingly expensive.

Table 8. Comparison of different keyframe selection methods on LongVideoBench validation under different video duration groups for Qwen3-VL-8B. Short denotes videos shorter than 3 minutes, Medium denotes videos from 3 to 20 minutes, and Long denotes videos longer than 20 minutes. Accuracy scores (%) are reported.
<table><tr><td>Method</td><td>Short</td><td>Medium</td><td>Long</td></tr><tr><td>Qwen3-VL w/ Uniform</td><td>74.79</td><td>59.46</td><td>51.77</td></tr><tr><td>Qwen3-VL w/ AKS</td><td>73.96</td><td>60.44</td><td>53.55</td></tr><tr><td>Qwen3-VL w/ BOLT</td><td>74.52</td><td>63.11</td><td>54.43</td></tr><tr><td>Qwen3-VL w/ FOCUS</td><td>70.36</td><td>64.80</td><td>57.45</td></tr><tr><td>Qwen3-VL w/ DIG</td><td>73.68</td><td>62.85</td><td>57.97</td></tr><tr><td>Qwen3-VL w/ AIR</td><td>73.96</td><td>65.14</td><td>53.62</td></tr><tr><td>Qwen3-VL w/ MarKey</td><td>74.06</td><td>65.77</td><td>59.57</td></tr></table>

## 4.9. Analysis of Video Duration

To examine the effect of video duration on keyframe selection, we compare different training-free methods on LongVideoBench validation under three duration groups, including short, medium, and long videos. The results are shown in Table 8. Several observations can be drawn from the results. First, MarKey achieves the best overall performance and delivers the strongest results on medium and long videos, while remaining competitive on short videos. Second, the advantage of MarKey becomes increasingly pronounced as video duration grows. The improvements on medium and long videos are much more substantial than those on short videos. In particular, compared with uniform sampling, MarKey improves accuracy by 6.31 points on medium videos and 7.80 points on long videos, clearly showing that its benefit becomes more evident as the temporal span increases. This trend further demonstrates the effectiveness of MarKey for long-video understanding in more challenging settings. As videos become longer, useful evidence is often more temporally dispersed and redundant visual content becomes more prevalent, making frame selection increasingly challenging under a fixed budget. In such cases, the explicit modeling of coverage and redundancy in MarKey helps preserve more complete task-relevant evidence while reducing unnecessary overlap, thereby enabling more effective, robust, and reliable keyframe selection for long-horizon reasoning.

## 4.10. Fine-grained Analysis on NExT-QA

To further understand the effectiveness of MarKey under different reasoning requirements, we conduct a fine-grained evaluation on the eight question types of NExT-QA, covering causal, temporal, and descriptive reasoning. The results are shown in Figure 7. First, MarKey achieves the best performance across all eight question types, indicating that its overall improvement is not dominated by a single category.

![](images/11cd3e1eef6208e5514decc4c2f2cea6f0ad190964f301d75acb2143bfd154f7.jpg)

![](images/5389f117aba52d36091a83dbcc2719ccf511a7d433d7bfd5f30ef0d0a753f4ac.jpg)  
Figure 8. Comparison between frames selected by AKS and MarKey on representative video understanding examples. The left column shows two examples from LongVideoBench, while the right column shows two from Video-MME. Green boxes indicate the questionrelevant evidence frames selected by MarKey. Compared with AKS, MarKey more effectively captures informative evidence relevant to the query, leading to more accurate answers under the same frame budget.

Table 9. Comparison of Mean Pairwise Cosine Similarity (MPCS) on LongVideoBench and Video-MME. All methods use Qwen3- VL-8B with 32 selected frames.
<table><tr><td>Method</td><td>LVB</td><td>V-MME</td></tr><tr><td>Qwen3-VL w/ AKS</td><td>0.6539</td><td>0.7161</td></tr><tr><td>Qwen3-VL w/ BOLT</td><td>0.7164</td><td>0.7975</td></tr><tr><td>Qwen3-VL w/ FOCUS</td><td>0.7490</td><td>0.7655</td></tr><tr><td>Qwen3-VL w/ DIG</td><td>0.7617</td><td>0.7332</td></tr><tr><td>Qwen3-VL w/ AIR</td><td>0.7457</td><td>0.7459</td></tr><tr><td>Qwen3-VL w/ MarKey</td><td>0.6177</td><td>0.6773</td></tr></table>

Consistent gains are observed on causal, temporal, and descriptive questions, demonstrating its robustness across diverse reasoning requirements. Second, MarKey exhibits particularly pronounced improvements on temporal reasoning questions across both Qwen2.5-VL and Qwen3-VL. For example, compared with the original Qwen3-VL baseline, MarKey improves Temporal Next and Temporal Previous by 3.35 and 3.23 percentage points, respectively, representing the two largest gains among the eight question types. This indicates that MarKey is especially effective at preserving the temporally complementary evidence required for reasoning about preceding and subsequent events.

## 4.11. Analysis of Selected-Frame Redundancy

To further examine the redundancy of the selected keyframes, we evaluate the Mean Pairwise Cosine Similarity among the selected frames on LongVideoBench and Video-MME. All methods use Qwen3-VL-8B with a fixed budget of 32 input frames. The results are shown in Table 9. Lower MPCS indicates lower similarity among the selected frames and thus less inter-frame redundancy. Several observations can be drawn from the results. We can observe that MarKey achieves the lowest MPCS on both benchmarks, with 0.6177 on LongVideoBench and 0.6773 on Video-MME, outperforming the strongest competing method by 0.0362 and 0.0388, respectively. This indicates that MarKey selects less redundant and more complementary frames. Moreover, MarKey achieves the best downstream performance while maintaining the lowest MPCS, supporting our central claim that reducing redundant selections and preserving complementary evidence leads to more effective use of the limited visual budget.

## 4.12. Case Study

To provide a qualitative comparison, Figure 8 visualizes the keyframes selected by MarKey and AKS on representative examples from LongVideoBench and Video-MME. In these examples, MarKey tends to retain frames that are more directly and consistently relevant to the question while providing broader coverage of the video content and avoiding visually repetitive selections. As a result, the selected subsets are not only better aligned with the query, but also more likely to capture complementary evidence distributed across different moments of the video. This advantage is particularly evident in the Video-MME examples, where answering the question often requires aggregating evidence from multiple temporal locations rather than relying on a single salient moment. For example, when asked how many national flags appear in the video, AKS misses part of the relevant evidence and leads the MLLM to predict three, whereas MarKey preserves more complementary evidence across different moments and enables the model to recover the correct count of four. This example highlights the importance of maintaining broad yet queryrelevant temporal coverage when the supporting evidence is distributed throughout the video. Overall, these qualitative examples support the quantitative findings and further illustrate that MarKey improves video understanding not simply by changing which frames are sampled, but by constructing a more effective, query-relevant, informative, less redundant, and more globally representative visual subset for downstream reasoning under a limited frame budget.

## 5. Conclusions

In this paper, we introduce MarKey, a selected subsetaware greedy keyframe selection framework for long-video understanding. MarKey reformulates keyframe selection as a context-dependent subset valuation problem, where the importance of a frame is measured by its approximate marginal contribution to the currently selected subset rather than by its standalone relevance. To make this formulation tractable, MarKey adopts a surrogate utility that jointly models query relevance, visual coverage, and redundancy suppression, and further employs temporally sampled anchor frames together with a greedy selection strategy to efficiently construct compact yet informative keyframe subsets under a fixed frame budget. Extensive experiments across six video-understanding benchmarks demonstrate that MarKey consistently improves the performance of strong long-video MLLMs and outperforms existing training-free keyframe selection baselines.

## Acknowledgments

This work was supported in part by the National Natural Science Foundation of China under Grant 62402158, and in part by the Key Science & Technology Project of Anhui Province under Grant 202523j08050001.

## References

[1] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716–23736, 2022. 1

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. 8

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 8, 9

[4] Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, et al. Temporalbench: Benchmarking finegrained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024. 1

[5] Yukang Chen, Fuzhao Xue, Dacheng Li, Qinghao Hu, Ligeng Zhu, Xiuyu Li, Yunhao Fang, Haotian Tang, Shang Yang, Zhijian Liu, et al. Longvila: Scaling long-context visual language models for long videos. arXiv preprint arXiv:2408.10188, 2024. 3

[6] Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Dan Guo, Xun Yang, and Meng Wang. Psyche-r1: Towards reli able psychological llms through unified empathy, expertise, and reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24889–24906, 2026. 3

[7] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108– 24118, 2025. 7

[8] Shreyank N Gowda, Marcus Rohrbach, and Laura Sevilla-Lara. Smart frame selection for action recognition. In Proceedings of the AAAI conference on artificial intelligence, pages 1451–1459, 2021. 3

[9] He Hu, Lianzhong You, Hongbo Xu, Qianning Wang, Fei Richard Yu, Fei Ma, Zebang Cheng, Zheng Lian, Yucheng Zhou, and Laizhong Cui. Emobench-m: Bench marking emotional intelligence for multimodal large lan guage models. arXiv preprint arXiv:2502.04424, 2025. 1

[10] Jinpeng Hu, Hongchang Shi, Chongyuan Dai, Zhuo Li, Peipei Song, and Meng Wang. Beyond emotion recognition: A multi-turn multimodal emotion understanding and reasoning benchmark. In Proceedings of the 33rd ACM Interna tional Conference on Multimedia, pages 5814–5823, 2025. 1

[11] Jinpeng Hu, Ao Wang, Qianqian Xie, Zhuo Li, Hui Ma, and Dan Guo. Agentmental: An interactive multi-agent framework for explainable and adaptive mental health assessment.

In Proceedings of the AAAI Conference on Artificial Intelligence, pages 31050–31058, 2026. 3

[12] Jinpeng Hu, Erqiang Wang, Shan Wang, Zhuo Li, Peipei Song, Xun Yang, and Meng Wang. Mmhbench: A multiperspective benchmark for mental health understanding in long-form videos. arXiv preprint arXiv:2607.27895, 2026. 1

[13] Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, et al. M-llm based video frame selection for efficient video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 13702– 13712, 2025. 2, 3

[14] Xiaohu Huang, Hao Zhou, and Kai Han. Prunevid: Visual token pruning for efficient video large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19959–19973, 2025. 3

[15] Bruno Korbar, Du Tran, and Lorenzo Torresani. Scsampler: Sampling salient clips from video for efficient action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6232–6242, 2019. 3

[16] Xiaohan Lan, Yitian Yuan, Zequn Jie, and Lin Ma. Vidcompress: Memory-enhanced temporal compression for video understanding in large language models. arXiv preprint arXiv:2410.11417, 2024. 3

[17] Hosu Lee, Junho Kim, Hyunjun Kim, and Yong Man Ro. Refocus: Reinforcement-guided frame optimization for contextual understanding. arXiv preprint arXiv:2506.01274, 2025. 2

[18] Jie Lei, Licheng Yu, Mohit Bansal, and Tamara Berg. Tvqa: Localized, compositional video question answering. In Proceedings ofthe 2018 conference on empirical methods in natural language processing, pages 1369–1379, 2018. 1

[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 3, 8

[20] Dongxu Li, Yudong Liu, Haoning Wu, Yue Wang, Zhiqi Shen, Bowen Qu, Xinyao Niu, Fan Zhou, Chengen Huang, Yanpeng Li, et al. Aria: An open multimodal native mixtureof-experts model. arXiv preprint arXiv:2410.05993, 2024. 3

[21] Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024. 3

[22] Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pages 19730– 19742. PMLR, 2023. 1, 12

[23] Junlong Li, Bingyao Yu, Yongming Rao, Jie Zhou, and Jiwen Lu. Tcovis: Temporally consistent online video instance segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 1097–1107, 2023. 6

[24] Jialuo Li, Bin Li, Jiahao Li, and Yan Lu. Divide, then ground: Adapting frame selection to query types for longform video understanding. In Proceedings ofthe IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pages 11369–11380, 2026. 8, 9

[25] KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. Science China Information Sciences, 68(10):200102, 2025. 3

[26] Xirui Li, Chao Ma, Xiaokang Yang, and Ming-Hsuan Yang. Vidtome: Video token merging for zero-shot video editing. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 7486–7495, 2024. 3

[27] Yanwei Li, Chengyao Wang, and Jiaya Jia. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pages 323–340. Springer, 2024. 3

[28] Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pages 5971–5984, 2024. 3

[29] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023. 3

[30] Jiajun Liu, Yibing Wang, Hanghang Ma, Xiaoping Wu, Xiaoqi Ma, Xiaoming Wei, Jianbin Jiao, Enhua Wu, and Jie Hu. Kangaroo: A powerful video-language model supporting long-context video input. arXiv preprint arXiv:2408.15542, 2024. 3

[31] Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem. Bolt: Boost large vision-language model without training for long-form video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 3318–3327, 2025. 2, 3, 8, 9

[32] Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. Tempcompass: Do video llms really understand videos? In Findings of the Association for Computational Linguistics: ACL 2024, pages 8731–8772, 2024. 1

[33] Yongdong Luo, Xiawu Zheng, Guilin Li, Shukang Yin, Hao jia Lin, Chaoyou Fu, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, et al. Video-rag: Visually-aligned retrieval-augmented long video comprehension. Advances in Neural Information Processing Systems, 38:168008–168033, 2026. 3

[34] Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. Drvideo: Document retrieval based long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18936–18946, 2025. 3

[35] Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12585–12602, 2024. 3

[36] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very longform video language understanding. Advances in Neural In formation Processing Systems, 36:46212–46244, 2023. 1

[37] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 5, 9, 12

[38] Francisco Romero, Caleb Winston, Johann Hauswald, Matei Zaharia, and Christos Kozyrakis. Zelda: Video analytics using vision-language models. arXiv preprint arXiv:2305.03785, 2023. 8

[39] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. arXiv preprint arXiv:2410.17434, 2024. 3

[40] Yan Shu, Zheng Liu, Peitian Zhang, Minghao Qin, Junjie Zhou, Zhengyang Liang, Tiejun Huang, and Bo Zhao. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 26160–26169, 2025. 3

[41] Hui Sun, Shiyin Lu, Huanyu Wang, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Ming Li. Mdp3: A training-free approach for list-wise frame selection in videollms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 24090–24101, 2025. 2, 3

[42] Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29118–29128, 2025. 2, 8, 9

[43] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 8, 9

[44] Tencent Hunyuan Team. Script-a-video: Deep structured audio-visual captions via factorized streams and relational grounding. arXiv preprint arXiv:2604.11244, 2026. 8

[45] OpenAI. Hello gpt-4o. https://openai.com/ index/hello-gpt-4o/, 2024. 8, 9

[46] Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025. 9

[47] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Ming Ding, Xiaotao Gu, Shiyu Huang, Bin Xu, et al. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 22958–22967, 2025. 1

[48] Ziyang Wang, Shoubin Yu, Elias Stengel-Eskin, Jaehong Yoon, Feng Cheng, Gedas Bertasius, and Mohit Bansal. Videotree: Adaptive tree-based video representation for llm reasoning on long videos. In Proceedings of the Computer

Vision and Pattern Recognition Conference, pages 3272– 3283, 2025. 3

[49] Yuetian Weng, Mingfei Han, Haoyu He, Xiaojun Chang, and Bohan Zhuang. Longvlm: Efficient long video understanding via large language models. In European Conference on Computer Vision, pages 453–470. Springer, 2024. 3

[50] Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857, 2024. 7

[51] Wenhao Wu, Dongliang He, Xiao Tan, Shifeng Chen, and Shilei Wen. Multi-agent reinforcement learning based frame sampling for effective untrimmed video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6222–6231, 2019. 3

[52] Zuxuan Wu, Caiming Xiong, Chih-Yao Ma, Richard Socher, and Larry S Davis. Adaframe: Adaptive frame selection for fast video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1278–1287, 2019. 3

[53] Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9777–9786, 2021. 1, 7

[54] Lin Xu, Yilin Zhao, Daquan Zhou, Zhijie Lin, See Kiong Ng, and Jiashi Feng. Pllava: Parameter-free llava extension from images to videos for video dense captioning. arXiv preprint arXiv:2404.16994, 2024. 3

[55] Yangyang Xu, Jinpeng Hu, Zhuoer Zhao, Zhangling Duan, Xiao Sun, and Xun Yang. Multiagentesc: A llm-based multiagent collaboration framework for emotional support conversation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 4665–4681, 2025. 3

[56] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, et al. Egolife: Towards ego centric life assistant. In Proceedings ofthe Computer Vision and Pattern Recognition Conference, pages 28885–28900, 2025. 7

[57] Jiabo Ye, Haiyang Xu, Haowei Liu, Anwen Hu, Ming Yan, Qi Qian, Ji Zhang, Fei Huang, and Jingren Zhou. mplug-owl3: Towards long image-sequence understanding in multi-modal large language models. arXiv preprint arXiv:2408.04840, 2024. 1

[58] Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. Advances in Neural Information Processing Systems, 36:76749–76771, 2023. 2

[59] Sicheng Yu, Chengkai Jin, Huanyu Wang, Zhenghao Chen, Sheng Jin, Zhongrong Zuo, Xiaolei Xu, Zhenbang Sun, Bingni Zhang, Jiawei Wu, et al. Frame-voyager: Learn ing to query frames for video large language models. arXiv preprint arXiv:2410.03226, 2024. 3

[60] Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang

Huang, Yuanqian Zhao, et al. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. arXiv preprint arXiv:2509.18154, 2025. 8

[61] Huaying Yuan, Zheng Liu, Minghao Qin, Hongjin Qian, Yan Shu, Zhicheng Dou, Ji-Rong Wen, and Nicu Sebe. Memoryenhanced retrieval augmentation for long video understanding. arXiv preprint arXiv:2503.09149, 2025. 3

[62] Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. In Proceedings of the 2023 conference on empirical methods in natural language processing: system demonstrations, pages 543–553, 2023. 3

[63] Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, et al. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 881–916, 2025. 9

[64] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024. 3, 12

[65] Yanzhe Zhang, Ruiyi Zhang, Jiuxiang Gu, Yufan Zhou, Nedim Lipka, Diyi Yang, and Tong Sun. Llavar: Enhanced visual instruction tuning for text-rich image understanding. arXiv preprint arXiv:2306.17107, 2023. 3

[66] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 8, 9

[67] Yuanhan Zhang, Yunice Chew, Yuhao Dong, Aria Leo, Bo Hu, and Ziwei Liu. Towards video thinking test: A holistic benchmark for advanced video reasoning and understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20626–20636, 2025. 8

[68] Bo Zhao, Boya Wu, Muyang He, and Tiejun Huang. Svit: Scaling up visual instruction tuning. arXiv preprint arXiv:2307.04087, 2023. 3

[69] Mingjun Zhao, Yakun Yu, Xiaoli Wang, Lei Yang, and Di Niu. Search-map-search: a frame selection paradigm for action recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10627–10636, 2023. 3

[70] Luowei Zhou, Chenliang Xu, and Jason Corso. Towards automatic learning of procedures from web instructional videos. In Proceedings of the AAAI conference on artificial intelligence, 2018. 7

[71] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3

[72] Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, and Yang You. Focus: Efficient keyframe selection for long video understanding. arXiv preprint arXiv:2510.27280, 2025. 2, 3, 8, 9

[73] Yuanhao Zou, Shengji Jin, Andong Deng, Youpeng Zhao, Jun Wang, and Chen Chen. Air: Enabling adaptive, iterative, and reasoning-based frame selection for video question answering. In International Conference on Learning Representations, pages 11302–11329, 2026. 8, 9