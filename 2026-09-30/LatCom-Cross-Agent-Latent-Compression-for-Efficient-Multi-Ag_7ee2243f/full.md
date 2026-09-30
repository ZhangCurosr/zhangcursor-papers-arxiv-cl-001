# LatCom: Cross-Agent Latent Compression for Efficient Multi-Agent Collaboration

Shinan Zhang<sup>1</sup>\*, Tao Zhang<sup>1</sup>\*, Qihui Zhu<sup>1</sup>\*, Mengjie Zhang<sup>1</sup>, Dong Jin<sup>1</sup>, Yunpeng Hou<sup>2</sup>, Shuangwu Chen<sup>1†</sup>, Xiaobin Tan<sup>1</sup>, Quan Zheng<sup>1</sup>, Jian Yang<sup>1</sup> <sup>1</sup>University of Science and Technology of China

<sup>2</sup>Institute of Artificial Intelligence, Hefei Comprehensive National Science Center {zsn884709682,zhangtaolqy,qh.zhu,zhangmengjie,kingdon,hyp314}@mail.ustc.edu.cn {chensw,xbtan,qzheng,jianyang}@ustc.edu.cn

## Abstract

LLM-based multi-agent systems (MAS) increasingly use latent collaboration to avoid the information loss and repeated encodingdecoding overhead of natural-language communication. However, directly forwarding all sender latents makes the receiver-side context scale with both the number of agents and the reasoning length, increasing computation, memory usage, and collaboration latency. A natural solution is latent compression. But we find that cross-agent redundancy remains unresolved in existing latent compression approaches, which typically compress each sender independently and then concatenate the results. We propose LatCom, a crossagent latent compression framework for efficient multi-agent latent collaboration. LatCom maps multiple sender latents into a fixed number of receiver-readable and task-relevant slots. Rather than reconstructing all sender hidden states, it optimizes the compressed latents for receiver-side task utility. LatCom trains the compressor in two stages: single-sender readability learning first establishes a latent interface interpretable by the frozen receiver, and multi-sender fusion learning then trains the compressor to fuse complementary evidence and remove redundancy across agents. Experiments on multiple benchmarks with Qwen3-4B show that LatCom achieves an average 2.46× inference speed-up over LatentMAS and reduces output token usage by 70.3%, while maintaining comparable average accuracy.

## 1 Introduction

LLM-based multi-agent systems (MAS) have emerged as a promising paradigm for complex reasoning, where specialized agents collaborate by exchanging intermediate thoughts and aggregating complementary evidence (Guo et al., 2024; Tran et al., 2025; Zou et al., 2025). Most existing

![](images/44df09ec9e502756c715b7d02a1893a94a6c70cb248a0243c0d706bae3bbd45a.jpg)  
Figure 1: Independent per-agent compression preserves cross-agent redundancy, as t-SNE visualization shows overlap among concatenated compressed latents.

MAS communicate in natural language, requiring agents to project internal states into discrete token sequences (Li et al., 2023; Hong et al., 2024; Wu et al., 2024). This discretization process may discard rich latent information and introduce additional overhead due to repeated encoding and decoding. (Zhang et al., 2025; Cemri et al., 2025; Chen et al., 2025). Latent collaboration addresses this limitation by allowing agents to exchange continuous internal representations, such as last-layer hidden states or KV caches, as intermediate latent thoughts or shared memory (Hao et al., 2025; Zou et al., 2025; Ramesh and Li, 2025; Tang et al., 2025; Zheng et al., 2026; Du et al., 2025; Fu et al., 2026b). However, latent collaboration introduces a new scaling bottleneck. As multiple agents generate latent thoughts over many reasoning steps, directly forwarding all latent thoughts makes the receiver-side latent context grow with both the number of agents and the reasoning length (Zou et al., 2025; Du et al., 2025). This increases receiver-side computation, expands memory usage, and slows the overall MAS collaboration process.

A natural solution is to compress latent thoughts into compact and task-relevant representations (Shen et al., 2025; Cheng and Van Durme, 2024). Existing latent compression methods typically compress each sender’s latent thoughts independently and then concatenate the compressed latents (Du et al., 2025) in the receiver side. However, as shown in Figure 1, by visualizing compressed latent thoughts of different agents via t-SNE (Fu et al., 2026a), we find substantial overlap even among highly compressed latent thoughts. Such redundancy arises from shared task context, overlapping evidence, and similar intermediate reasoning across agents. We further quantify this redundancy on GSM8K, ARC-Easy, and ARC-Challenge. Across the three datasets, pairwise raw cosine similarity ranges from 0.553 to 0.771, linear CKA ranges from 0.408 to 0.600, and the joint effective rank is approximately 43% lower than the sum of the individual ranks. Detailed results are reported in Appendix A.1.This enables further latent compression to improve multi-agent collaboration efficiency.

Inspired by this finding, we propose LatCom, a cross-agent latent compression framework for efficient multi-agent latent collaboration. LatCom compresses multiple sender latents into a fixed number of receiver-readable and task-relevant latent slots. It trains the compressor in two stages: single-sender readability learning first establishes a latent interface interpretable by the frozen receiver, and multi-sender fusion learning then trains the compressor to fuse complementary evidence and remove redundancy across agents. Instead of reconstructing all sender hidden states, LatCom optimizes the compressed latents for receiver-side task utility. Experiments on multiple benchmarks with Qwen3-4B show that LatCom achieves an average 2.46× inference speed-up over LatentMAS and reduces output token usage by 70.3%, while maintaining comparable average accuracy.

Our Contributions. (1) New Insight. We identify cross-agent latent redundancy as a key bottleneck in multi-agent latent collaboration, where independently compressed sender latents still contain duplication and increase receiver-side computation. (2) New Framework. We propose LatCom, a cross-agent latent compression framework that maps multiple sender latents into a fixed number of slots through two-stage compressor training. (3) Comprehensive Evaluation. Extensive experiments on multiple LLMs and benchmarks show that LatCom achieves strong task performance while substantially reducing inference latency and token usage.

## 2 Preliminary

## 2.1 Multi-Agent Latent Collaboration

We consider a multi-agent latent communication setting with sender agents $\mathcal { A } _ { s } = \{ A _ { 1 } , . . . , A _ { N } \}$ and a receiver $A _ { R }$ (Zhao et al., 2025; Zhuge et al., 2024). Given task input $q ,$ each sender $A _ { i }$ observes its context $x _ { i }$ and reasons directly in latent space. Instead of decoding intermediate thoughts into natural language, it emits aligned latent thought embeddings as continuous messages. Let $E _ { i } =$ $[ e _ { i , 1 } , \ldots , e _ { i , t } ]$ denote the input embeddings of $A _ { i }$ At latent step ℓ, the sender maps its last-layer hidden state back to the input-embedding space:

$$
z _ { i , \ell } = h _ { i , t + \ell - 1 } W _ { a } ^ { ( i ) } ,\tag{1}
$$

where $W _ { a } ^ { ( i ) }$ is a sender-specific alignment operator (Zou et al., 2025). The aligned vector $z _ { i , \ell }$ is then reused as the next latent input, enabling iterative reasoning without intermediate text generation. After m steps, sender $A _ { i }$ produces

$$
Z _ { i } = \{ z _ { i , \ell } \} _ { \ell = 1 } ^ { m } \in \mathbb { R } ^ { m \times d } ,\tag{2}
$$

where d is the hidden dimension. Latent communication avoids repeated decoding and re-encoding while preserving continuous reasoning signals.

## 2.2 Latent Compression

Forwarding all sender-side latent thoughts scales with both the number of senders N and latent steps m. We therefore compress them into a fixed-size receiver-readable latent. Given sender trajectories $\{ Z _ { i } \} _ { i = 1 } ^ { N }$ , we form a unified sequence

$$
\begin{array} { r } { U = [ E _ { C } ( q ) , \rho _ { 1 } , Z _ { 1 } , \delta , \dots , \rho _ { N } , Z _ { N } ] \in \mathbb { R } ^ { L \times d } , } \end{array}\tag{3}
$$

where $E _ { C } ( q )$ embeds the task input, $\rho _ { i }$ marks sender identity, and $\delta$ separates adjacent latents. This sequence preserves sender boundaries while exposing complementary evidence and redundancy to a latent compressor. We instantiate $C _ { \phi }$ with $K$ learnable slot tokens $B = [ b _ { 1 } , \ldots , b _ { K } ] \in \mathbb { R } ^ { K \times d } .$ These tokens collect information from the aggregated sender sequence. The compressor $C _ { \phi }$ maps $U$ to a compact message

$$
M = C _ { \phi } ( U , B ) \in \mathbb { R } ^ { K \times d } ,\tag{4}
$$

where the $K < L$ . Let $E _ { R } ( q )$ denote the receiverside prompt embeddings. The compressed slots

![](images/12f6406fdc87f75c23795b730a425564b5970ce381c40b8e1af9e8b9133b1cc3.jpg)  
Figure 2: Overview of LatCom. Sender agents generate aligned latent thoughts, which LatCom aggregates and compresses into fixed-size receiver-readable slots. The frozen receiver uses these slots for final generation, while the compressor is trained through readability and fusion learning.

are prepended to the receiver prompt as continuous context, and the receiver generates the output autoregressively:

$$
p _ { \theta _ { R } } ( y \mid q , M ) = \prod _ { t = 1 } ^ { | y | } p _ { \theta _ { R } } { \big ( } y _ { t } \mid y _ { < t } , [ M , E _ { R } ( q ) ] { \big ) } ,\tag{5}
$$

where $\theta _ { R }$ are the receiver’s parameters. Formally, latent compression is optimized by minimizing the frozen receiver’s generation loss on the groundtruth answer $y ^ { \star }$ :

$$
\phi ^ { \star } = \arg \operatorname* { m i n } _ { \phi } \mathcal { L } \big ( C _ { \phi } ( U ) , y ^ { \star } ; \theta _ { R } \big ) ,\tag{6}
$$

where $\mathcal { L } ( \cdot )$ evaluates how well the compressed latent message $C _ { \phi } ( U )$ supports the receiver in generating $y ^ { \star }$ . The key challenge is to compress multi-sender latent thoughts without discarding task-critical information.

## 3 LatCom

## 3.1 Overview of LatCom

Figure 2 overviews LatCom, which consists of latent generation and aggregation, latent compression, and collaborative inference. Given a task input, sender agents produce aligned latent thoughts without decoding intermediate text. These thoughts are aggregated into a multi-sender sequence with task, sender-identity, and separator embeddings. LatCom then compresses the aggregated sequence into fixed-size receiver-readable latent slots. The compressor is trained in two stages: a single-sender readability stage that learns a receiver-compatible latent interface, followed by a multi-sender fusion stage that learns to compress complementary and redundant thoughts into shared slots. During inference, the compressed slots are prepended to the receiver prompt as continuous context, and the frozen receiver generates the final answer.

## 3.2 Two-stage Compressor Training

We train LatCom in two stages. The compressor uses the same backbone family as the sender and receiver, with the language modeling head removed. During training, all sender and receiver parameters are frozen, and only the compressor is updated. The two stages target different aspects of latent compression. Stage 1 uses a one-to-one setting, where a single sender’s latent thought is compressed and consumed by the frozen receiver. This stage establishes a receiver-readable continuous interface. Stage 2 uses the final many-to-one setting, where multiple sender latent thoughts are jointly compressed into fixed-size slots. This stage trains the compressor to fuse task-relevant information while reducing the influence of irrelevant or redundant latent content.

Evidence-structured training instances. Lat-Com is optimized for task-oriented compression rather than sender-state reconstruction. To expose the compressor to varying source quality, we construct each training instance at the evidence level. Evidence documents are grouped into gold-fact documents ${ \mathcal { G } } .$ non-gold documents $\mathcal { T } ,$ and mixed documents $\mathcal { R }$ that combine partial gold and nongold content. These groups respectively produce task-supporting, irrelevant or distracting, and noisy but partially useful sender trajectories.

The two training stages use these evidence groups differently. Stage 1 uses only gold-evidence inputs to learn a receiver-readable single-trajectory interface. Stage 2 samples multiple sender inputs from all three groups, training the compressor to handle non-uniform source quality during multisource compression.

Receiver-side supervision. For any latent message $M ,$ , we define the receiver-side supervised decoding loss as

$$
\mathcal { I } ( M ) = - \frac { 1 } { \vert S \vert } \sum _ { t \in \mathcal { S } } \log p _ { \theta _ { R } } \big ( y _ { t } \ \vert \ y _ { < t } , [ M ; E _ { R } ( q ) ] \big ) ,\tag{7}
$$

where S indexes the supervised response tokens. Since the receiver is frozen, this loss trains the compressor through the receiver’s generation behavior.

Stage 1: Readability learning. Given a task input $q ,$ target response $y ,$ and one aligned sender trajectory $Z _ { i }$ from gold evidence, the compressor produces $M _ { i } = C _ { \phi } ( Z _ { i } )$ . The frozen receiver then predicts $y$ conditioned on $[ M _ { i } ; E _ { R } ( q ) ]$ . Stage 1 first optimizes the receiver-side task loss:

$$
\mathcal { L } _ { \mathrm { t a s k } } ^ { ( 1 ) } = \mathcal { I } ( M _ { i } ) .\tag{8}
$$

To encourage the compressor to distinguish useful sender latents from unrelated ones, we introduce a receiver-grounded contrastive loss. For each matched thought $Z _ { i } ,$ we sample in-batch mismatched thoughts ${ \mathcal { N } } _ { i }$ . Let $M ^ { - } = C _ { \phi } ( Z ^ { - } )$ for $Z ^ { - } \in \mathcal { N } _ { i }$ . We define

$$
\begin{array} { r c l } { { \mathcal { L } _ { \mathrm { c o n } } ^ { ( 1 ) } = \displaystyle \frac { 1 } { | \mathcal { N } _ { i } | } \sum _ { Z ^ { - } \in \mathcal { N } _ { i } } \mathrm { s o f t p l u s } ( s _ { i } ^ { - } ) , } } \\ { { \displaystyle s _ { i } ^ { - } = \frac { \mathcal { I } ( M _ { i } ) - \mathcal { I } ( M ^ { - } ) + \Delta } { \tau } , } } \end{array}\tag{9}
$$

where $\Delta$ is a margin and $\tau$ is a temperature. This loss encourages the matched compressed latent to yield a lower receiver decoding loss than mismatched latents. We further add a full-trajectory alignment loss to prevent the compressor from learning answer-triggering soft prompts that ignore sender-side information. Let $P _ { t } ( X )$ denote the receiver predictive distribution over the token at position t conditioned on latent context $X$

$$
P _ { t } ( X ) = p _ { \theta _ { R } } { \big ( } \cdot \mid y _ { < t } , [ X ; E _ { R } ( q ) ] { \big ) } .\tag{10}
$$

Therefore, the alignment loss is defined as

$$
\mathcal { L } _ { \mathrm { a l i g n } } ^ { ( 1 ) } = \frac { 1 } { | \mathcal { S } | } \sum _ { t \in \mathcal { S } } D _ { \mathrm { J S } } \big ( P _ { t } ( Z _ { i } ) , P _ { t } ( M _ { i } ) \big ) + \alpha \left\| \bar { M } _ { i } - \bar { Z } _ { i } \right\| _ { 2 } ^ { 2 } ,\tag{11}
$$

where $D _ { \mathrm { J S } }$ is the Jensen-Shannon divergence, and $\bar { M _ { i } }$ and ${ \bar { Z } } _ { i }$ are mean-pooled compressed and full sender latents. The first term aligns receiver behavior under full and compressed latent contexts, while the second provides a coarse latent anchor. The Stage 1 objective is therefore:

$$
\mathcal { L } ^ { ( 1 ) } = \mathcal { L } _ { \mathrm { t a s k } } ^ { ( 1 ) } + \lambda _ { \mathrm { c o n } } \mathcal { L } _ { \mathrm { c o n } } ^ { ( 1 ) } + \lambda _ { \mathrm { a l i g n } } \mathcal { L } _ { \mathrm { a l i g n } } ^ { ( 1 ) } ,\tag{12}
$$

where $\lambda _ { \mathrm { { c o n } } }$ and $\lambda _ { \mathrm { a l i g n } }$ are weighting coefficients for the contrastive and alignment terms, respectively.

Stage 2: Fusion learning. In stage 2, we train the LatCom in the final many-to-one setting. Each gold-evidence document $\mathcal { G }$ is assigned to an individual sender, producing task-supporting latent thoughts from different sources. We further sample non-gold and mixed evidence from $\mathcal { T }$ and R with probabilities $p _ { I }$ and $p _ { R } .$ , respectively. The number of senders is sampled or clipped to the range of $^ 2$ to $^ { 6 . }$ The selected inputs are processed by frozen senders, aggregated into $U _ { : }$ and compressed into $M = C _ { \phi } ( U )$ . The task loss is defined by the receiver-side decoding loss:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t a s k } } ^ { ( 2 ) } = \mathcal { I } ( M ) . } \end{array}\tag{13}
$$

To train the compressor to identify task-supporting sources under non-uniform source quality, we extend the receiver-grounded contrast to multi-source inputs. The positive input is the matched aggregated sequence $U .$ . Negatives $\tilde { U } \in \mathcal { N } _ { \mathrm { m s } } ( \bar { U } )$ are constructed by replacing one or more goldevidence sender inputs with same-type inputs from other examples. The corresponding messages $M = C _ { \phi } ( U )$ and ${ \widetilde { M } } = C _ { \phi } ( { \widetilde { U } } )$ are compared using the same receiver-loss margin contrast as in Eq. 9, yielding $\mathcal { L } _ { \mathrm { m s } } ^ { ( 2 ) }$ . Non-gold and mixed evidence sampled into the matched sequence U are not treated as negatives. They remain part of the input and expose the compressor to distracting, noisy, or redundant latent content. The contrastive term only penalizes cases where task-supporting sources are replaced by mismatched sources from other examples. The Stage 2 objective combines the task and multi-source contrastive terms:

$$
\begin{array} { r } { \mathcal { L } ^ { ( 2 ) } = \mathcal { L } _ { \mathrm { t a s k } } ^ { ( 2 ) } + \lambda _ { \mathrm { m s } } \mathcal { L } _ { \mathrm { m s } } ^ { ( 2 ) } , } \end{array}\tag{14}
$$

where $\lambda _ { \mathrm { m s } }$ controls the strength of the multi-source contrastive loss.

## 4 Experiments

## 4.1 Experimental Setup

Training data and LatCom training. Lat-Com is trained on multi-hop QA datasets, including HotpotQA (Yang et al., 2018) and MuSiQue-Ans (Trivedi et al., 2022). We filter out single-model-solvable, and pre-compressionunanswerable examples to focus training on multisource evidence compression rather than base QA ability. Training follows the two-stage procedure described in Section 3.1: Stage 1 learns receiverreadable slots, and Stage 2 trains multi-source compression with 2 to 6 senders. Sender and receiver parameters are frozen throughout training, and only the compressor is updated. Additional data construction details are provided in the appendix B.1.

Datasets. We evaluate LatCom on seven public benchmarks covering mathematical reasoning, scientific and medical QA, commonsense reasoning, and code generation: GSM8K (Cobbe et al., 2021), GPQA-Diamond (Rein et al., 2024), MedQA (Yang et al., 2025b), ARC-Easy, ARC-Challenge (Clark et al., 2018), MBPP-Plus, and HumanEval-Plus (Liu et al., 2023). This suite evaluates whether fixed-slot latent compression preserves task-relevant information across diverse reasoning domains and output formats.

Models and baselines. We use Qwen3-4B-Base and Qwen3-8B-Base (Yang et al., 2025a) as backbone LLMs. All methods are evaluated in the same hierarchical three-sender setting (Zhuge et al., 2024), where math, science, and code agents communicate with a receiver. Baselines include TextMAS, LatentMAS (Zou et al., 2025), LatentMAS-Hidden, Interlat (Du et al., 2025), and LatentMAS-H2O (Zhang et al., 2023), covering text communication, KV-cache relay, aligned hidden-state transfer, sender-side latent compression followed by concatenation, and H2O-style KVcache pruning.

Implementation details. Following Latent-MAS (Zou et al., 2025), we compute the realignment matrix once per run and use 40 latent reasoning steps per sender. LatCom compresses the three aligned sender trajectories into 64 latent slots, which are prepended to the receiver prompt embeddings. For InterLat, we retrain the compressor using the same data sources, backbone, training budget, and evaluation setting as LatCom, and independently compress each sender into approximately 21 slots, resulting in approximately 63 slots in total, comparable to LatCom’s 64-slot budget. All methods use identical sender and receiver models, decoding settings, and task-specific maximum output lengths. We report task performance and end-to-end latency on 8 NVIDIA A800-80G GPUs.

## 4.2 Main Results

Table 1 compares LatCom with text-based and latent-communication baselines under the MAS setting. LatCom achieves the highest average accuracy with both Qwen3-4B and Qwen3-8B, reaching 76.27 and 80.15, respectively. It obtains the best result in 7 out of 14 task-model settings and consistently outperforms LatentMAS-H2O and InterLat. LatCom surpasses LatentMAS and LatentMAS-Hidden on average under both backbone sizes, showing that a fixed-slot latent interface can maintain strong accuracy without forwarding all sender-side latent states. Compared with InterLat, which independently compresses each sender before concatenating the compressed latents, LatCom improves average accuracy by 1.9 points on Qwen3-4B and 1.77 points on Qwen3- 8B. This highlights the benefit ofjoint multi-source compression, where cross-sender redundancy and complementarity are modeled before receiver conditioning. The gains are consistent across model scales, suggesting that the compression strategy is not tied to a specific backbone size. Although the average gains over LatentMAS are modest and Lat-Com is not the best method on every task, it maintains comparable task performance with a bounded communication budget while substantially reducing inference cost.

Table 1: Main results of LatCom on 7 public benchmarks under the MAS setting. Values in parentheses report the relative accuracy change of LatCom over each baseline. ↑ and ↓ indicate higher and lower accuracy, respectively.
<table><tr><td>Tasks</td><td>Metrics</td><td>TextMAS</td><td>LatentMAS</td><td>InterLat</td><td>LatentMAS-H2O</td><td>LatentMAS-hidden</td><td>LatCom</td></tr><tr><td colspan="8">Qwen3-4B</td></tr><tr><td>GSM8K</td><td>Acc.</td><td>89.40 (↑0.49%)</td><td>88.10 (↑1.98%)</td><td>85.82 (↑4.68%)</td><td>83.55 (↑7.53%)</td><td>86.58 (↑3.77%)</td><td>89.84</td></tr><tr><td>ARC-E</td><td>Acc.</td><td>96.93 (↑0.65%)</td><td>95.45 (↑2.21%)</td><td>95.09 (↑2.59%)</td><td>94.74 (↑2.98%)</td><td>95.71 (↑1.93%)</td><td>97.56</td></tr><tr><td>ARC-C</td><td>Acc.</td><td>91.66 (↑1.25%)</td><td>91.72 (↑1.19%)</td><td>90.78 (↑2.23%)</td><td>89.85 (↑3.29%)</td><td>90.44 (↑2.62%)</td><td>92.81</td></tr><tr><td>MedQA</td><td>Acc.</td><td>65.49 (↓1.25%)</td><td>66.33 (↓2.50%)</td><td>64.16 (↑0.79%)</td><td>62.00 (↑4.31%)</td><td>66.67 (↓3.00%)</td><td>64.67</td></tr><tr><td>MBPP+</td><td>Acc.</td><td>69.12 (↓2.40%)</td><td>66.67 (↑1.18%)</td><td>65.74 (↑2.62%)</td><td>64.81 (↑4.09%)</td><td>64.81 (↑4.09%)</td><td>67.46</td></tr><tr><td>HumanEval+</td><td>Acc.</td><td>75.87 (↓0.34%)</td><td>76.22 (↓0.80%)</td><td>73.47 (↑2.91%)</td><td>70.73 (↑6.90%)</td><td>75.00 (↑0.81%)</td><td>75.61</td></tr><tr><td>GPQA-Diamond</td><td>Acc.</td><td>41.00 (↑12.02%)</td><td>46.97 (↓2.21%)</td><td>45.47 (↑1.01%)</td><td>43.97 (↑4.46%)</td><td>47.25 (↓2.79%)</td><td>45.93</td></tr><tr><td>Avg.</td><td>Acc.</td><td>75.64 (↑0.83%)</td><td>75.92 (↑0.46%)</td><td>74.37 (↑2.56%)</td><td>72.81 (↑4.75%)</td><td>75.21 (↑1.41%)</td><td>76.27</td></tr><tr><td colspan="8">Qwen3-8B</td></tr><tr><td>GSM8K</td><td>Acc.</td><td>91.45 (↑1.64%)</td><td>92.11 (↑0.91%)</td><td>91.81 (↑1.24%)</td><td>91.51 (↑1.57%)</td><td>92.19 (↑0.82%)</td><td>92.95</td></tr><tr><td>ARC-E</td><td>Acc.</td><td>98.61 (↓0.72%)</td><td>96.72 (↑1.22%)</td><td>96.81 (↑1.13%)</td><td>96.89 (↑1.04%)</td><td>96.89 (↑1.04%)</td><td>97.90</td></tr><tr><td>ARC-C</td><td>Acc.</td><td>93.65 (↑1.19%)</td><td>93.26 (↑1.61%)</td><td>92.15 (↑2.83%)</td><td>91.04 (↑4.09%)</td><td>93.86 (↑0.96%)</td><td>94.76</td></tr><tr><td>MedQA</td><td>Acc.</td><td>76.87 (↑0.69%)</td><td>75.45 (↑2.58%)</td><td>74.53 (↑3.84%)</td><td>73.62 (↑5.13%)</td><td>76.11 (↑1.69%)</td><td>77.40</td></tr><tr><td>MBPP+</td><td>Acc.</td><td>72.19 (↑0.78%)</td><td>73.51 (↓1.03%)</td><td>72.18 (↑0.79%)</td><td>70.85 (↑2.68%)</td><td>73.26 (↓0.70%)</td><td>72.75</td></tr><tr><td>HumanEval+</td><td>Acc.</td><td>76.85 (↑2.07%)</td><td>78.06 (↑0.49%)</td><td>74.64 (↑5.09%)</td><td>71.22 (↑10.14%)</td><td>77.17 (↑1.65%)</td><td>78.44</td></tr><tr><td>GPQA-Diamond</td><td>Acc.</td><td>44.80 (↑4.62%)</td><td>47.93 (↓2.21%)</td><td>46.53 (↑0.73%)</td><td>45.13 (↑3.86%)</td><td>47.96 (↓2.27%)</td><td>46.87</td></tr><tr><td>Avg.</td><td>Acc.</td><td>79.20 (↑1.20%)</td><td>79.58 (↑0.72%)</td><td>78.38 (↑2.26%)</td><td>77.18 (↑3.85%)</td><td>79.63 (↑0.65%)</td><td>80.15</td></tr></table>

![](images/8f33cf5438fc2ca041a46a229b252eeca0c475e3333b543d580fa86b373c7b13.jpg)  
Figure 3: Inference speed-up across seven benchmarks.  
Figure 4: Output token usage across seven benchmarks.

## 4.3 Efficiency Analysis

![](images/bf3fa6eb67a5b0254a9f4346c2b78c76f3879c454ceb8d530d7d5bfafd89f8aa.jpg)

We report end-to-end inference speed-up and average output token usage for each method on Qwen3- 4B across multiple benchmarks. As shown in Figure 3, LatCom achieves the fastest inference across all seven benchmarks, with an average speed-up of 2.46×. The gains are consistent across tasks, ranging from 1.85× on ARC-Challenge to 3.75× on GSM8K, indicating that the fixed-slot latent bottleneck reduces receiver-side processing cost. We further profile the standalone compression step in Appendix C.4, showing that global compression takes only 0.116 seconds on average and therefore introduces limited runtime overhead. Figure 4 further shows that LatCom reduces output token usage by 70.3% on average, with task-level reductions ranging from 63.6% on ARC-Easy to 81.3% on GSM8K. This suggests that LatCom provides a compact conditioning signal that shortens receiver-side generation. Together with the accuracy results in Table 1, these results show that taskoriented fixed-slot compression improves inference efficiency while preserving strong downstream performance.

## 4.4 In-depth Analyses on LatCom

Does LatCom preserve useful latent structure after compression? We examine whether the compressed slots remain compatible with the senderside latent space after fixed-slot compression. Figure 5 visualizes the aligned hidden states from three sender agents together with LatCom’s compressed slots. Sender hidden states occupy broad and highly overlapping regions, indicating semantically entangled and partially redundant latent trajectories. By contrast, the compressed slots form a more concentrated distribution rather than reproducing the full sender-state space. This suggests that Lat-

![](images/6db218da0c8a0cc901d8b8c82d5bdd6f2d7abf5cb86c1a55998317aadc27b552.jpg)

Figure 5: t-SNE visualization of latent communication  
![](images/8ba9191f464dfcecea2d8ff7cd3e085e9be9fae81400514d20bfa14cf6da1838.jpg)  
Figure 6: Receiver Entropy under Latent Conditioning

Com learns a compact abstraction aligned with the sender space while filtering task-irrelevant or repeated latent variation. The visualization provides qualitative evidence that fixed-slot compression preserves useful latent structure without directly forwarding all sender hidden states.

Does compression provide clearer task-oriented signals? We further examine whether compressed slots provide a clearer conditioning signal for receiver-side generation. On GSM8K, we compare LatCom with LatentMAS-Hidden by measuring mean next-token entropy over generated reasoning traces, using three 50-token windows from the beginning, middle, and final parts of the reasoning body. As shown in Figure 6, LatCom yields lower entropy across most reasoning positions, with a more stable gap in the middle and final windows. This indicates that LatCom’s compressed slots make the receiver’s prediction distribution more concentrated, whereas direct conditioning on concatenated sender hidden trajectories may expose the receiver to redundant or weakly relevant latent signals that require additional decoding steps to filter and organize. Together with the output-token reductions in Figure 4 and the accuracy results in Table 1, this suggests that LatCom provides cleaner receiver conditioning, enabling shorter generations while preserving competitive downstream performance.

![](images/993341a093653606407526d1177b32407aa40518bf8144b0dce34df5fe9b5d66.jpg)

Figure 7: Ablation Study of Two-Stage Training Table 2: Ablation of training objectives on GSM8K. “Drop” is computed against the corresponding full setting within the same stage.
<table><tr><td>Stage</td><td>Setting</td><td>Acc.</td><td>Drop</td></tr><tr><td>Single-sender readability learning</td><td></td><td></td><td></td></tr><tr><td>Stage 1</td><td>Full Stage 1</td><td>80.06</td><td></td></tr><tr><td>Stage 1</td><td>w/o  ${ \mathcal { L } } _ { \mathrm { c o n } }$ </td><td>74.53</td><td>↓5.53</td></tr><tr><td>Stage 1</td><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { a l i g n } }$ </td><td>68.41</td><td>↓11.65</td></tr><tr><td colspan="4">Multi-source fusion learning</td></tr><tr><td>Stage  $1 + { \mathrm { S t a g e } } 2$ </td><td>Full training</td><td>89.84</td><td></td></tr><tr><td> $\mathrm { S t a g e } \ 1 + \mathrm { S t a g e } \ 2$ </td><td> $\mathrm { w } / \mathrm { o } \ \mathcal { L } _ { \mathrm { m s } }$ </td><td>87.47</td><td>↓2.37</td></tr></table>

## 4.5 Ablation Study

Figure 7 ablates the two-stage training procedure. Single-sender readability learning produces receiver-readable slots, but remains limited to oneto-one communication. Adding multi-sender utility learning improves accuracy on all benchmarks by an average of 9.0 percentage points, with larger gains on MBPP+ and HumanEval+ (16.6 and 12.0 points). This confirms that Stage 2 is important for adapting slots to useful multi-source compression.

Table 2 further evaluates the training objectives on GSM8K. Removing $\mathcal { L } _ { \mathrm { c o n } } , \mathcal { L } _ { \mathrm { a l i g n } }$ , and ${ \mathcal { L } } _ { \mathrm { m s } }$ reduces accuracy by 5.53, 11.65, and 2.37 points, respectively. The largest drop from removing $\mathcal { L } _ { \mathrm { a l i g n } }$ shows that alignment to dense sender trajectories is critical for receiver-readable slots, while the other drops indicate that contrastive losses help bind slots to correct and task-supporting sources.

## 4.6 Parameter Sensitivity Analysis

Effect of sender number. We examine LatCom under different numbers of senders with a fixed slot budget. As shown in Figure 8, performance is stable from 3 to 4 senders and declines moderately with 6 or 8 senders. Even with 8 senders, the average drop on GSM8K, ARC-Easy, and ARC-Challenge is only about 2.2 percentage points, suggesting robustness to moderate sender scaling.

<table><tr><td>sender</td><td>3</td><td>4</td><td>6</td><td>8</td></tr><tr><td>ARC-C</td><td>92.8</td><td>92.5</td><td>91.3</td><td>90.6</td></tr><tr><td>ARC-E</td><td>97.6</td><td>97.6</td><td>96.9</td><td>95.8</td></tr><tr><td>GSM8K</td><td>89.8</td><td>90.0</td><td>88.4</td><td>87.2</td></tr></table>

![](images/c942a483130d535f08de46e5575af00ae94902348277deae8233ef4fcc8296ae.jpg)

Figure 8: Sensitivity Analysis of Sender Number
<table><tr><td>K</td><td>8</td><td>16</td><td>32</td><td>64</td><td>128</td></tr><tr><td>ARC-C</td><td>81.5</td><td>91.7</td><td>91.5</td><td>92.8</td><td>90.9</td></tr><tr><td>ARC-E</td><td>85.3</td><td>97.0</td><td>98.1</td><td>97.6</td><td>97.1</td></tr><tr><td>GSM8K</td><td>80.2</td><td>88.5</td><td>89.5</td><td>89.8</td><td>88.9</td></tr></table>

![](images/6ead492aa0551bc60b1f8bfbf83c2bf3596e959d7b2d30e74072a972ba36ad40.jpg)  
Figure 9: Sensitivity to Compressed Slot Number

Effect of compressed slot number. We further vary the compressed slot budget K. As shown in Figure 9, $K = 8$ causes clear accuracy drops, while performance improves and reaches the best or near-best accuracy around K = 32 to K = 64. Since K = 128 brings no further gain and can slightly degrade performance, we use K = 64 in the main experiments as a stable accuracycompactness trade-off.

## 5 Related Works

LLM-based multi-agent systems. LLM-based multi-agent systems distribute problem solving across specialized agents, with representative frameworks including CAMEL (Li et al., 2023), MetaGPT (Hong et al., 2024), and AutoGen (Wu et al., 2024). Such systems can improve the coverage and robustness of complex reasoning (Guo et al., 2024; Tran et al., 2025), but most still rely on natural-language communication. This requires agents to serialize internal states into tokens and downstream agents to re-encode them, which can discard fine-grained latent information and introduce additional inference overhead (Zhang et al., 2025; Cemri et al., 2025; Chen et al., 2025). Our work addresses this communication bottleneck by studying compact latent representations for efficient inter-agent exchange.

Latent collaboration in multi-agent systems. Recent work explores continuous latent space as an alternative to text-mediated interaction. Some studies perform latent reasoning within a single model, using hidden states instead of decoded chain-of-thought tokens (Cheng and Van Durme, 2024; Chen et al., 2025), while others extend latent interaction to model or agent collaboration through hidden-state exchange or KV-cache transfer (Zheng et al., 2026; Du et al., 2025; Fu et al., 2026b; Zou et al., 2025). However, existing latent-collaboration methods often relay full latent trajectories or directly combine states from multiple senders (Zou et al., 2025; Du et al., 2025), retaining redundant or low-utility states within trajectories and overlapping information across senders. As the number of agents and reasoning steps increases, such unfiltered communication expands the receiver-side context and computational cost. LatCom instead introduces a fixed-slot latent bottleneck for efficient multi-source exchange under a bounded communication budget.

Compression for multi-agent systems. Compression has been widely studied to improve LLM inference and communication efficiency. Textoriented methods, such as ICAE (Ge et al., 2024) and AutoCompressor (Chevalier et al., 2023), compress natural-language context into soft memories but still rely on the text-to-latent conversion that latent collaboration seeks to avoid. KV-cache reduction methods, including H2O (Zhang et al., 2023), StreamingLLM (Xiao et al., 2024), and SnapKV (Li et al., 2024), reduce long-context inference cost by retaining, evicting, or merging cached states, but are designed mainly for singlemodel cache management. Closest to our work, hidden-state communication methods compress each sender’s latent trajectory before receiver conditioning (Du et al., 2025). However, independent sender-side compression does not explicitly model cross-sender redundancy or complementarity before concatenation or aggregation. LatCom instead performs receiver-aware multi-source compression, mapping multiple sender hidden trajectories into a fixed number of receiver-readable slots optimized for downstream task utility.

## 6 Conclusion

In this paper, we identified cross-agent latent redundancy as a key bottleneck in multi-agent latent collaboration and introduced LatCom, a crossagent latent compression framework. LatCom compresses multiple sender latents into fixed-budget, receiver-readable, and task-relevant slots, mitigating receiver-side context growth. Its two-stage training establishes a latent interface for the frozen receiver and then learns to fuse complementary evidence while reducing cross-agent redundancy. Experiments across multiple benchmarks and models show that LatCom preserves strong task performance while substantially reducing inference latency and token usage, demonstrating the promise of latent compression for efficient multi-agent collaboration.

## Limitations

This work has several limitations. First, LatCom relies on aligned hidden representations within the same model family. As a result, its latent compression and receiver conditioning may be suboptimal in more general heterogeneous multi-agent systems, where agents differ in architecture, hidden dimensionality, tokenizers, or communication topologies. Second, LatCom adopts a fixed-slot compression budget for multi-source latent trajectories. Although this design improves efficiency, full latent relay remains stronger on some codegeneration tasks, suggesting that fixed-slot compression may discard fine-grained information required for certain forms of reasoning. Extending LatCom to heterogeneous model settings, adaptive communication budgets, and reliability-sensitive applications remains an important direction for future work.

## Ethics Statement

LatCom aims to improve the efficiency of multiagent latent collaboration through task-oriented latent compression. Because LatCom is built on existing LLMs, it may inherit their limitations, including biased predictions, hallucinations, and factually incorrect outputs. Users should therefore verify model outputs when deploying such systems in real-world scenarios, especially in applications involving safety, privacy, or high-stakes decisions. Our experiments are based on public benchmarks, Qwen3 models, PyTorch, and Hugging Face Transformers. We follow their respective licenses and usage policies and gratefully acknowledge their contributions to the research community.

## Acknowledgements

This work was supported by the National Key R&D Program of China, No. 2024YDLN0004, and the Fundamental Research Funds for the Central Universities under Grant No. WK2102026004.

## References

Mert Cemri, Melissa Z Pan, Shuyi Yang, Lakshya A Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. 2025. Why do multi-agent LLM systems fail? In NeurIPS 2025 Workshop on Evaluating the Evolving LLM Lifecycle: Benchmarks, Emergent Abilities, and Scaling.

Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. 2025. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. arXiv preprint arXiv:2505.16782.

Jeffrey Cheng and Benjamin Van Durme. 2024. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In The 2023 Conference on Empirical Methods in Natural Language Processing.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Zhuoyun Du, Runze Wang, Huiyu Bai, Zouying Cao, Xiaoyong Zhu, Yu Cheng, Bo Zheng, Wei Chen, and Haochao Ying. 2025. Enabling agents to communicate entirely in latent space. arXiv preprint arXiv:2511.09149.

Muxin Fu, Guibin Zhang, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, and Yang Yang. 2026a. Latentmem: Customizing latent memory for multi-agent systems. arXiv preprint arXiv:2602.03036.

Tianyu Fu, Zihan Min, Hanling Zhang, Jichao Yan, Guohao Dai, Wanli Ouyang, and Yu Wang. 2026b. Cacheto-cache: Direct semantic communication between large language models. In Proceedings of the Fourteenth International Conference on Learning Representations.

Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V. Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. In Proceedings ofthe Thirty-Third International Joint Conference on Artificial Intelligence.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason E. Weston, and Yuandong Tian. 2025. Training large language models to reason in a continuous latent space. In Proceedings of the Second Conference on Language Modeling.

Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In Proceedings of the Twelfth International Conference on Learning Representations.

Guohao Li, Hasan Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. 2023. Camel: Communicative agents for "mind" exploration of large language model society. In Proceedings of the 37th International Conference on Neural Information Processing Systems.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. SnapKV: LLM knows what you are looking for before generation. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. 2023. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. Advances in Neural Information Processing Systems.

Vignav Ramesh and Kenneth Li. 2025. Communicating activations between language model agents. In Proceedings of the 42nd International Conference on Machine Learning.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. 2025. CODI: Compressing chain-of-thought into continuous space via selfdistillation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing.

Yichen Tang, Weihang Su, Yujia Zhou, Yiqun Liu, Min Zhang, Shaoping Ma, and Qingyao Ai. 2025. Augmenting multi-agent communication with state delta trajectory. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing.

Khanh-Tung Tran, Dung Dao, Minh-Duong Nguyen, Quoc-Viet Pham, Barry O’Sullivan, and Hoang D. Nguyen. 2025. Multi-agent collaboration mechanisms: A survey of llms. arXiv preprint arXiv:2501.06322.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics.

Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, and Chi Wang. 2024. Autogen: Enabling next-gen LLM applications via multi-agent conversations. In Proceedings of the First Conference on Language Modeling.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2024. Efficient streaming language models with attention sinks. In The Twelfth International Conference on Learning Representations.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Hang Yang, Hao Chen, Hui Guo, Yineng Chen, Ching-Sheng Lin, Shu Hu, Jinrong Hu, Xi Wu, and Xin Wang. 2025b. LLM-MedQA: Enhancing medical question answering through case studies in large language models. In 2025 International Joint Conference on Neural Networks.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

Guibin Zhang, Yanwei Yue, Zhixun Li, Sukwon Yun, Guancheng Wan, Kun Wang, Dawei Cheng, Jeffrey Xu Yu, and Tianlong Chen. 2025. Cut the crap: An economical communication pipeline for LLM-based multi-agent systems. In Proceedings of the Thirteenth International Conference on Learning Representations.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. $_ \mathrm { H _ { 2 } O }$ : Heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems.

Wanjia Zhao, Mert Yuksekgonul, Shirley Wu, and James Zou. 2025. SiriuS: Self-improving multi-agent systems via bootstrapped reasoning. In Advances in Neural Information Processing Systems.

Yujia Zheng, Zhuokai Zhao, Zijian Li, Yaqi Xie, Mingze Gao, Lizhu Zhang, and Kun Zhang. 2026. Thought communication in multiagent collaboration. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Mingchen Zhuge, Wenyi Wang, Louis Kirsch, Francesco Faccio, Dmitrii Khizbullin, and Jürgen Schmidhuber. 2024. GPTSwarm: Language agents as optimizable graphs. In Proceedings of the 41st International Conference on Machine Learning.

Jiaru Zou, Xiyuan Yang, Ruizhong Qiu, Gaotang Li, Katherine Tieu, Pan Lu, Ke Shen, Hanghang Tong, Yejin Choi, Jingrui He, James Zou, Mengdi Wang, and Ling Yang. 2025. Latent collaboration in multiagent systems. arXiv preprint arXiv:2511.20639.

## A Additional Analyses

## A.1 Quantitative Analysis of Cross-Agent Redundancy

We quantify cross-agent redundancy on GSM8K, ARC-Easy, and ARC-Challenge using the latent trajectories produced by the math, science, and code agents. For each agent pair, we compute raw average cosine similarity, centered average cosine similarity, and linear centered kernel alignment (CKA). Raw average cosine measures the directional similarity between the original latent trajectories, while centered average cosine removes the mean direction before computing similarity.

Given two trajectory matrices X and $Y ,$ , linear CKA is computed as

$$
\operatorname { C K A } ( X , Y ) = { \frac { \| X ^ { \top } Y \| _ { F } ^ { 2 } } { \| X ^ { \top } X \| _ { F } \| Y ^ { \top } Y \| _ { F } } } ,\tag{15}
$$

where $\lVert \cdot \rVert _ { F }$ denotes the Frobenius norm. Linear CKA measures the similarity between the overall representation structures of two agent trajectories.

As shown in Table 3, raw average cosine similarity remains high across all three datasets. The Science–Code pair reaches 0.771, 0.747, and 0.717 on GSM8K, ARC-Easy, and ARC-Challenge, respectively. Linear CKA also remains consistently high; for example, the Math–Science pair obtains 0.600, 0.577, and 0.543. These results indicate that different agents share both common representational components and similar trajectory-level structures.

We further compute the effective rank of each agent trajectory and their joint trajectory. Given a hidden-state matrix H with singular values $\{ \sigma _ { i } \}$ we define

$$
\begin{array} { r } { p _ { i } = \cfrac { \sigma _ { i } ^ { 2 } } { \sum _ { j } \sigma _ { j } ^ { 2 } } , \qquad } \\ { \mathrm { e r a n k } { ( H ) } = \exp \left( - \sum _ { i } p _ { i } \log p _ { i } \right) . } \end{array}\tag{16}
$$

For the three agents, SeparateSum denotes the sum of their individual effective ranks. The joint trajectory is formed as

$$
H _ { \mathrm { J o i n t } } = [ H _ { \mathrm { M a t h } } ; H _ { \mathrm { S c i e n c e } } ; H _ { \mathrm { C o d e } } ] .\tag{17}
$$

We measure the relative difference between the separate and joint effective ranks as

$$
\mathrm { R a n k C o m p r e s s i o n } = 1 - { \frac { \mathrm { e r a n k } ( H _ { \mathrm { J o i n t } } ) } { \mathrm { S e p a r a t e S u m } } } .\tag{18}
$$

Table 4 shows that the joint effective rank is substantially smaller than the sum of the individual ranks. Rank compression remains close to 43% on all three datasets, indicating that the effective dimensionality of the combined trajectories does not grow independently with the number of agents. This result is consistent with the similarity analysis and supports joint compression across sender trajectories.

## A.2 Controlled Analysis of Cross-Agent Fusion

We introduce two controls to separate the effect of joint multi-source fusion from simple pooling and additional single-source training. Joint Mean Pool takes the same aggregated multi-sender sequence as LatCom and applies adaptive mean pooling along the sequence dimension to produce 64 latent slots. It does not use a learned compressor. Single-source Continuation uses the same Transformer compressor, 64 learnable slots, training-data scale, and optimization budget as LatCom. It continues training on the examples allocated to Stage 2 while retaining the single-source construction, without forming joint multi-sender inputs.

<table><tr><td>Dataset</td><td>Agent pair</td><td>Raw AvgCos</td><td>Centered AvgCos</td><td>Linear CKA</td></tr><tr><td>GSM8K</td><td>Math-Science</td><td>0.605</td><td>0.061</td><td>0.600</td></tr><tr><td></td><td>Math-Code</td><td>0.693</td><td>0.061</td><td>0.559</td></tr><tr><td></td><td>Science-Code</td><td>0.771</td><td>0.069</td><td>0.508</td></tr><tr><td>ARC-Easy</td><td>Math-Science</td><td>0.587</td><td>0.068</td><td>0.577</td></tr><tr><td></td><td>Math-Code</td><td>0.615</td><td>0.053</td><td>0.463</td></tr><tr><td></td><td>Science-Code</td><td>0.747</td><td>0.076</td><td>0.457</td></tr><tr><td>ARC-Challenge</td><td>Math-Science</td><td>0.553</td><td>0.062</td><td>0.543</td></tr><tr><td></td><td>Math-Code</td><td>0.580</td><td>0.047</td><td>0.426</td></tr><tr><td></td><td>Science-Code</td><td>0.717</td><td>0.064</td><td>0.408</td></tr></table>

Table 3: Pairwise similarity between latent trajectories from the math, science, and code agents.
<table><tr><td>Dataset</td><td>Math</td><td>Science</td><td>Code</td><td>Separate Sum</td><td>Joint Rank</td><td>Rank Compression</td></tr><tr><td>GSM8K</td><td>7.6</td><td>10.3</td><td>9.6</td><td>27.5</td><td>15.6</td><td>43.3%</td></tr><tr><td>ARC-Easy</td><td>7.3</td><td>9.8</td><td>9.1</td><td>26.2</td><td>14.8</td><td>43.5%</td></tr><tr><td>ARC-Challenge</td><td>8.0</td><td>10.8</td><td>10.0</td><td>28.8</td><td>16.3</td><td>43.4%</td></tr></table>

Table 4: Effective-rank analysis of individual and joint agent trajectories.

<table><tr><td>Method</td><td>Slots</td><td>GSM8K Acc.</td></tr><tr><td>Joint Mean Pool</td><td>64</td><td>78.92</td></tr><tr><td>Single-source Continuation</td><td>64</td><td>86.47</td></tr><tr><td>LatCom</td><td>64</td><td>89.84</td></tr></table>

Table 5: Controlled comparison of cross-agent fusion under greedy decoding.

As shown in Table 5, Joint Mean Pool is 10.92 percentage points below LatCom, showing that simple pooling is insufficient for preserving taskrelevant latent information. Single-source Continuation is 3.37 points below LatCom despite using the same compressor architecture, slot number, training-data scale, and optimization budget. The difference therefore comes from training on joint multi-sender inputs rather than from additional single-source training. This result is consistent with the 9.0-point average gain from Stage 2 and the 2.37-point drop after removing L<sub>ms</sub>.

## B Training Details

## B.1 Training Data Construction

We construct the main compressor training data from HotpotQA and MuSiQue-Ans. HotpotQA is a Wikipedia-based multi-hop question answering dataset that provides questions, gold answers, passages, and sentence-level supporting-fact annotations. Its questions often require bridge reasoning or comparison across multiple passages. MuSiQue-Ans is a compositional multi-hop QA dataset in which questions require combining multiple singlehop reasoning steps, together with answer annotations and supporting evidence chains.

We use multi-hop QA as the primary training setting because it naturally requires information from multiple evidence sources. Compared with singlehop factoid QA, this setting reduces the chance that the frozen receiver can solve the task from the question alone, making the training signal depend more directly on whether useful evidence is communicated through sender latents and preserved by the compressor. To focus training on multi-source evidence compression rather than base QA ability, we filter out single-model-solvable examples and precompression-unanswerable examples. The former can be answered by the backbone model without multi-source latent communication, whereas the latter cannot be answered even when the sender– receiver system receives the relevant sender information before compression. The remaining examples therefore provide a cleaner signal for learning how to compress useful sender-side evidence into receiver-readable latent slots.

For each retained example, the task input is the question q. Because the original answers are often short spans or yes/no labels, answer-only targets provide limited receiver-side supervision. We therefore construct a rationale-augmented target response from the question, supporting evidence, and gold answer. Let ${ \mathcal { E } } ^ { + }$ denote the supporting evidence and a denote the gold answer. A concise evidence-grounded rationale r is generated by an instruction-following model, and the target response is formatted as

$$
\begin{array} { r l } & { r = \mathrm { R a t i o n a l e } ( q , \mathcal { E } ^ { + } , a ) , } \\ & { y = \big [ \mathsf { R e a s o n i n g } \colon r ; \mathsf { A n s w e r } \colon a \big ] . } \end{array}\tag{19}
$$

The rationale is constrained to remain consistent with the provided evidence, while the final answer follows the original dataset annotation.

## B.2 Evidence-structured Training Instances

We organize the associated passages according to their evidence annotations. Documents containing supporting facts are treated as gold-evidence documents G, while the remaining documents are treated as non-gold documents I. We additionally construct mixed evidence R by combining partial gold-evidence content with partial non-gold content under the sender input-length budget:

$$
\begin{array} { l } { { \mathcal { G } = \{ g _ { j } \} _ { j = 1 } ^ { n _ { g } } , ~ \mathcal { T } = \{ i _ { j } \} _ { j = 1 } ^ { n _ { i } } , } } \\ { { \mathcal { R } = \{ r _ { j } \} _ { j = 1 } ^ { n _ { r } } . } } \end{array}
$$

These groups correspond to task-supporting, irrelevant or distracting, and noisy but partially useful sources.

Stage 1 uses only gold-evidence inputs. Each instance contains the task input $q ,$ the rationaleaugmented target response $y ,$ and a sender latent trajectory $Z _ { i }$ produced from gold evidence. This stage focuses on learning a receiver-readable latent interface.

Stage 2 follows the final many-to-one communication setting. Gold-evidence documents are assigned to different senders, and additional non-gold or mixed-evidence inputs are sampled to form heterogeneous multi-source inputs. The number of senders is sampled or clipped to the range of 2 to 6. To prevent the compressor from over-specializing to Wikipedia-style multi-hop QA, Stage 2 also includes a small number of auxiliary mathematical reasoning and code reasoning examples. We convert these examples into the same multi-source format, with task-supporting, distracting, and redundant sources. These auxiliary examples are used only to diversify the latent source structures seen during Stage 2 while preserving the same many-toone compression objective.

The selected sender inputs are encoded by frozen senders, aggregated into U, and compressed into fixed-size latent slots $M = C _ { \phi } ( U )$

## B.3 Example of Constructed Training Instance

Figure 10 illustrates how a raw multi-hop QA example is converted into evidence-structured sender inputs and a rationale-augmented target.

## B.4 Training Hyperparameters

We report the optimization hyperparameters used for LatCom compressor training in Table 6. Unless otherwise specified, sender and receiver parameters are frozen throughout training, and only the compressor parameters are updated.

## C Evaluation Details

## C.1 Evaluation Benchmarks

We evaluate LatCom on the same seven-benchmark suite as LatentMAS, covering mathematical reasoning, scientific and medical question answering, commonsense reasoning, and code generation. The suite includes GSM8K, ARC-Easy, ARC-Challenge, MedQA, MBPP-Plus, HumanEval-Plus, and GPQA-Diamond.

GSM8K. GSM8K is a grade-school mathematical reasoning benchmark consisting of naturallanguage word problems. Each problem requires multi-step arithmetic reasoning and produces a final numerical answer. We evaluate GSM8K by extracting and normalizing the final numeric prediction from the receiver output and comparing it with the gold answer.

ARC-Easy and ARC-Challenge. ARC-Easy and ARC-Challenge are multiple-choice science question answering benchmarks from the AI2 Reasoning Challenge. ARC-Easy contains relatively straightforward elementary-level science questions, whereas ARC-Challenge contains more difficult questions that are less likely to be solved by shallow lexical cues. We report multiple-choice accuracy after normalizing the predicted answer option.

MedQA. MedQA evaluates medical question answering in a multiple-choice format. Compared with general commonsense QA, it requires domainspecific medical knowledge and careful interpretation of clinical or biomedical contexts. We report accuracy by matching the normalized predicted option with the gold label.

Table 6: Optimization and loss hyperparameters for LatCom training.
<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="3">Optimization settings</td></tr><tr><td>Optimizer</td><td>Optimizer type</td><td> $\mathrm { A d a m W } , \beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 , \epsilon = 1 0 ^ { - 8 }$ </td></tr><tr><td>Optimizer</td><td>Stage 1 learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Optimizer</td><td>Stage 2 learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Optimizer</td><td>LR scheduler</td><td>Cosine decay</td></tr><tr><td>Optimizer</td><td>Warmup steps / ratio</td><td>5%</td></tr><tr><td>Regularization</td><td>Weight decay</td><td>0.01</td></tr><tr><td>Batching</td><td>Global batch size</td><td>64</td></tr><tr><td>Batching</td><td>Gradient accumulation steps</td><td>4</td></tr><tr><td>Stability</td><td>Max gradient norm</td><td>1.0</td></tr><tr><td colspan="3">Loss weights and contrastive settings</td></tr><tr><td>Task loss</td><td>Stage 1 task loss weight</td><td>1.0</td></tr><tr><td>Task loss</td><td>Stage 2 task loss weight</td><td>1.0</td></tr><tr><td>Contrastive loss</td><td> $\lambda _ { \mathrm { c o n } }$ </td><td>Dynamically adjusted in [0.01, 0.5]</td></tr><tr><td>Alignment loss</td><td> $\lambda _ { \mathrm { a l i g n } }$ </td><td>Dynamically adjusted in [0.01, 0.2]</td></tr><tr><td>Multi-source contrast</td><td> $\lambda _ { \mathrm { m s } }$ </td><td>Dynamically adjusted in [0.01, 0.5]</td></tr><tr><td>Contrastive margin</td><td> $\Delta$ </td><td>0.57</td></tr><tr><td>Contrastive temperature</td><td> $\tau$ </td><td>1.0 0.3</td></tr><tr><td>Latent anchor</td><td>Alignment anchor weight α</td><td></td></tr><tr><td>Supervision</td><td>Supervised token set S</td><td>Rationale and answer tokens.</td></tr></table>

GPQA-Diamond. GPQA-Diamond is a challenging graduate-level scientific question answering benchmark. The Diamond subset contains expert-written questions designed to be difficult for non-experts and resistant to simple retrieval shortcuts. We use it to evaluate whether latent compression preserves fine-grained scientific reasoning signals under a long-output setting, and report multiple-choice accuracy.

MBPP-Plus. MBPP-Plus is a code-generation benchmark extended from MBPP with stronger test cases. Each example contains a natural-language programming problem, and the model must generate Python code that satisfies the benchmark tests. We evaluate the generated program by executing it against the test cases and report pass rate.

HumanEval-Plus. HumanEval-Plus extends HumanEval with additional and more rigorous unit tests. It evaluates functional code generation from problem descriptions and function signatures. We extract the generated Python solution, execute it with the provided tests, and report pass rate.

Task-specific generation limits. Following the LatentMAS evaluation setting, we use task-specific maximum generation lengths. GSM8K, ARC-Easy, and ARC-Challenge use a maximum output length of 2,048 tokens; MedQA, MBPP-Plus, and HumanEval-Plus use 4,096 tokens; and GPQA-

<table><tr><td>Dataset</td><td>Category</td><td>Max. tokens</td></tr><tr><td>GSM8K</td><td>Math</td><td>2048</td></tr><tr><td>ARC-Easy</td><td>Commonsense sci.</td><td>2048</td></tr><tr><td>ARC-Challenge</td><td>Commonsense sci.</td><td>2048</td></tr><tr><td>MedQA</td><td>Medical QA</td><td>4096</td></tr><tr><td>MBPP-Plus</td><td>Code</td><td>4096</td></tr><tr><td>HumanEval-Plus</td><td>Code</td><td>4096</td></tr><tr><td>GPQA-Diamond</td><td>Scientific QA</td><td>8192</td></tr></table>

Table 7: Evaluation benchmarks and maximum output lengths.

Diamond uses 8,192 tokens. These limits are shared across all compared methods.

## C.2 Compared Methods

We compare LatCom with text-based and latentcommunication baselines under the same hierarchical multi-agent setting. Unless otherwise specified, all methods use the same sender roles, receiver role, prompts, decoding settings, and evaluation scripts.

TextMAS. TextMAS is the natural-language communication baseline. Each sender agent generates a textual response according to its assigned role, and the receiver conditions on the concatenated sender outputs to produce the final answer. This baseline represents the standard MAS communication interface, where intermediate information is serialized into discrete text before being used by the receiver.

LatentMAS. LatentMAS serves as the full latentcommunication baseline. It follows the hierarchical latent MAS setting, where sender agents perform latent rollout and transmit their latent computation states to the final receiver. The receiver then generates the final answer conditioned on these latent states rather than relying only on natural-language messages. In our experiments, LatentMAS uses the same hierarchical prompt structure, sender roles, latent reasoning steps, realignment setting, and decoding parameters as LatCom.

InterLat. InterLat is a learned latent-interface baseline for inter-agent communication. Following its official setting, non-output agents exchange compressed continuous latent representations instead of natural-language messages, and the final output agent uses the received latent information to generate the answer. We include InterLat as a representative latent-space communication method and evaluate it under the same benchmark suite and hierarchical comparison protocol.

LatentMAS-Hidden. LatentMAS-Hidden is a hidden-state relay variant of LatentMAS. It follows the same hierarchical LatentMAS workflow, including sender and receiver roles, prompts, latent rollout steps, and realignment settings. The key difference is that the sender-to-receiver relay object is changed from KV caches to hidden embeddings. For each sender, we collect both the prompt/input hidden embeddings and the realigned latent hidden embeddings produced during latent rollout. These embeddings are concatenated across senders and inserted into the final receiver’s prompt embedding sequence, from which the receiver decodes the final answer.

LatentMAS-H2O. LatentMAS-H2O is a KVcompressed variant of LatentMAS. It preserves the KV-cache relay mechanism of LatentMAS, but applies an H2O-style cache selection procedure before passing the sender cache to the final receiver. During sender latent rollout, attention scores from latent tokens to previous positions are recorded and aggregated to estimate token importance. The compressed cache retains important prompt tokens together with the preserved history and latent tail. We use the headwise H2O variant as the default setting, with the same hierarchical prompt, latent rollout, realignment, and decoding setup as Latent-MAS.

## C.3 Implementation Details

All methods are evaluated under a hierarchical multi-agent setting with three sender agents and one receiver agent. Following the LatentMAS hierarchical setup, the senders are instantiated as math, science, and code agents, while the receiver serves as the final task summarizer. The same hierarchical prompts are used for all compared methods.

For latent-communication methods, each sender performs 40 latent reasoning steps. Latent-space realignment is enabled, and the realignment matrix is computed once per run. Unless otherwise specified, all methods use greedy decoding with sampling disabled. All experiments are implemented with HuggingFace Transformers and PyTorch, without vLLM. Task-specific maximum output lengths follow Table 7.

We use a unified answer extraction and evaluation protocol across methods. For GSM8K, we extract the final numerical answer from the receiver output and compare it with the normalized gold answer. For ARC-Easy, ARC-Challenge, MedQA, and GPQA-Diamond, we normalize the predicted option and compare it with the gold multiple-choice label. For MBPP-Plus and HumanEval-Plus, we extract the generated Python code, combine it with the corresponding test cases, and execute the resulting program. A prediction is counted as correct if it passes all benchmark tests within the execution timeout.

## C.4 Compressor Overhead

We further measure the runtime overhead introduced by the LatCom compressor itself. Specifically, we record the wall-clock time of the global compression step, namely the time required to map the aggregated multi-source latent sequence U into the fixed-slot message $M = C _ { \phi } ( U )$ . This measurement excludes sender latent rollout, receiver prompt construction, and receiver decoding.

As shown in Table 8, global compression adds only a small computational overhead. The average compression time is 0.116 seconds, indicating that the learned compressor introduces limited additional runtime relative to the overall multi-agent inference process.

<table><tr><td>Category</td><td>Metric</td><td>Value</td><td>Interpretation</td></tr><tr><td colspan="4">Global compression time</td></tr><tr><td>Runtime</td><td>GSM8K</td><td>0.095 s</td><td>Time for mapping U to  $M = C _ { \phi } ( U )$ </td></tr><tr><td>Runtime</td><td>ARC-Easy</td><td>0.096 s</td><td>Time for mapping U to  $M = C _ { \phi } ( U )$ </td></tr><tr><td>Runtime</td><td>ARC-Challenge</td><td>0.107 s</td><td>Time for mapping U to  $M = C _ { \phi } ( U )$ </td></tr><tr><td>Runtime</td><td>MedQA</td><td>0.126 s</td><td>Time for mapping U to  $M = C _ { \phi } ( U )$ </td></tr><tr><td>Runtime</td><td>MBPP-Plus</td><td>0.128 s</td><td>Time for mapping U to</td></tr><tr><td>Runtime</td><td>HumanEval-Plus</td><td>0.132 s</td><td> $M = C _ { \phi } ( U )$  Time for mapping U to</td></tr><tr><td>Runtime</td><td>GPQA-Diamond</td><td>0.125 s</td><td> $M = C _ { \phi } ( U )$  Time for mapping U to  $M = C _ { \phi } ( U )$ </td></tr><tr><td>Runtime</td><td>Average</td><td>0.116 s</td><td>Average global compression time across tasks</td></tr><tr><td colspan="4">Standalone compressor memory overhead</td></tr><tr><td>Memory</td><td>Compressor model footprint</td><td>7.49 GiB</td><td>Persistent BF16 memory for loading one compressor</td></tr><tr><td>Memory</td><td>Peak CUDA allocation delta</td><td>112.78 MiB</td><td>Additional peak GPU tensor memory during compression</td></tr><tr><td>Memory</td><td>Peak CPU RSS delta</td><td>0.0086 MiB</td><td>Additional peak CPU memory during compression</td></tr></table>

Table 8: Standalone overhead of the LatCom compressor. The model footprint is persistent, while CUDA and CPU deltas measure the additional peak memory introduced by the compression call itself.

![](images/dc0b046755decb9e8b46f089886c7cc33cd114b59274dacea4ec932bd4fd0665.jpg)  
Figure 10: Example of constructing evidence-structured sender inputs and a rationale-augmented target from a HotpotQA instance.