# ConvMem: Convolutional Memory for Long-Context Reasoning

Hongming Zhang, Zhaozhen Gu, Fengshuo Bai, Ming Hao, Qingyang Zhang, Yuanyuan Wang, Shiyang Tang, Yanna Wang, Bo Xu<sup>†</sup> Institute of Automation, Chinese Academy of Sciences {hongming.zhang, boxu}@ia.ac.cn

## Abstract

While Large Language Models (LLMs) have demonstrated impressive capabilities, they of ten struggle with extremely long contexts due to fixed context limits. To address this, sequential approaches like MemAgent extend the effective context by reading text in segments and iteratively updating a fixed-size memory. However, this sequential paradigm suffers from high latency and requires costly reinforcement learning (RL) training, which can lead to overfitting on specific datasets. To overcome these limitations, we propose ConvMem, a trainingfree, highly parallelizable framework that reformulates long-context reasoning as a hierarchical convolution. Inspired by CNNs, ConvMem treats an LLM prompted with a specific query as a convolutional kernel. This kernel summarizes text segments hierarchically, shortening the reasoning path from a linear chain into a logarithmic tree. Specifically, ConvMem integrates Configurable Strides and Skip Connections to ensure robust evidence capture and propagation, while employing Multi-Kernel Convolution to decompose complex queries into disentangled semantic channels. This design not only mitigates error accumulation but also enables massive parallelization across both text segments and reasoning threads. Experiments on RULER-HotpotQA and RULER-2WikiMultiHopQA demonstrate that ConvMem outperforms training-free baselines and avoids the risk of overfitting to parametric priors often observed in RL-trained models on out-of-distribution tasks.

## 1 Introduction

The capability to process extremely long contexts, ranging from digesting entire books to executing complex multi-step reasoning, has become a central objective for large language models (LLMs) (Achiam et al., 2023; Liu et al., 2024a; Team et al., 2025). While advancements in positional encoding and attention mechanisms (Su et al.,

2024; Beltagy et al., 2020; Liu et al., 2023) have theoretically extended context windows to millions of tokens, practical deployment faces bottlenecks regarding inference latency and performance degradation. Specifically, standard self-attention suffers from quadratic complexity $( O ( N ^ { 2 } ) )$ ), making it prohibitive for massive inputs. Furthermore, even models capable of processing long texts often exhibit the lost-in-the-middle phenomenon (Liu et al., 2024b), where reasoning performance degrades as context length increases.

To address these scalability issues, recent research has pivoted towards memory-based agents, such as MemAgent (Yu et al., 2025), Memα (Wang et al., 2025), and grounded-memory agents (Yang et al., 2025; Cui et al., 2025). By treating text as a sequential stream with iterative memory updates, these methods achieve linear complexity (O(N)). However, this paradigm introduces two critical limitations. The first is the sequential bottleneck, where the strict temporal dependency of memory states prohibits parallelization and results in high inference latency. The second is the risk of overfitting. To optimize memory policies, these systems rely heavily on reinforcement learning (RL) (Sutton and Barto, 2018; Dong et al., 2020; Zhang and Yu, 2020). Our analysis reveals that RLtrained agents tend to overfit to specific datasets, often hallucinating superficially correct answers based on pre-trained knowledge rather than faithfully reasoning over the provided context.

To overcome these challenges, we propose Convolutional Memory (ConvMem), a training-free framework designed for high-fidelity, parallelizable long-context reasoning. Drawing inspiration from convolutional neural networks (CNNs), ConvMem reformulates long-text processing by treating a frozen LLM prompted with a specific query as a semantic convolutional kernel. Unlike sequential approaches, ConvMem scans text segments concurrently at each layer. This hierarchical architecture transforms the information flow from a linear chain into a logarithmic tree structure (O(log N) depth). This topological shift not only accelerates inference via parallelization but also mitigates cumulative error propagation by shortening the reasoning path.

ConvMem incorporates three architectural mechanisms to enhance reasoning fidelity and precision:

• Configurable Stridesfor Robust Evidence Capture: Relying on a single pass renders reasoning vulnerable to boundary truncation and model stochasticity. ConvMem introduces overlapping windows to perform multi-view scanning. This approach enables the kernel to cross-verify information across different receptive fields, enhancing the stability and recall of key evidence.

• Skip Connectionsfor Detail Preservation: Hierarchical summarization inevitably compresses information, risking the loss of fine-grained details. We introduce a semantic residual mechanism where high-confidence raw evidence bypasses intermediate layers and propagates directly to the final reasoning stage, ensuring the answer is grounded in precise details.

• Multi-Kernel Convolutionfor Semantic Disentanglement: Complex tasks often involve intertwined logical threads. ConvMem decomposes queries into sub-questions and deploys distinct kernels to extract relevant information into separate semantic channels, preventing interference between orthogonal reasoning paths.

We evaluate ConvMem on RULER-HotpotQA and RULER-2WikiMultiHopQA (Hsieh et al., 2024; Yang et al., 2018; Ho et al., 2020). Results demonstrate that ConvMem outperforms trainingfree baselines and exhibits superior generalization compared to RL-trained models in out-ofdistribution tasks. Our contributions are threefold:

• We propose ConvMem, the first training-free, CNN-inspired hierarchical framework that enables parallelizable long-context reasoning by breaking the sequential bottleneck.

• We empirically expose the overfitting risks of RL-based memory agents, demonstrating that their performance relies heavily on parametric memorization rather than in-context reasoning.

• We demonstrate that through the principled integration of configurable strides, semantic skip connections, and multi-kernel convolution, ConvMem establishes a new state-of-the-art for training-free methods, while offering superior generalization compared to RL-trained models.

## 2 ConvMem for Long-Context Reasoning

In this section, we present ConvMem, a trainingfree framework that adapts the architectural principles of convolutional neural networks (CNNs) to long-context reasoning. Unlike sequential memory agents that process text as a linear stream, ConvMem reformulates the reasoning process as a hierarchical, multi-channel convolutional operation. We first formulate the problem and define the semantic convolutional kernel in Section 2.2. We then detail the core architectural mechanisms in Section 2.3, including configurable strides, skip connections and multi-kernel convolution, that adapt visual convolution to textual reasoning. Finally, we describe the complete hierarchical parallel inference workflow in Section 2.4.

## 2.1 Problem Formulation

We consider a long-context question answering task. Let $\mathcal { D } = \{ t _ { 1 } , t _ { 2 } , \ldots , t _ { N } \}$ denote a massive input document consisting of N tokens, and Q denote a user query. Our objective is to generate an answer A that accurately addresses Q based on the evidence contained in D. Drawing an analogy to CNN’s spatial convolution, we define key notations adapted for textual reasoning:

• W (Kernel Size): The maximum context length, in tokens, scanned by a single LLM kernel.

• S (Stride): The step size for the scanning kernel. Setting $S < W$ creates overlapping windows, allowing each token to be scanned multiple times (W/S) to enhance information recall.

• L (Depth): The total number of layers in the hierarchical architecture.

• C (Channels): The number of sub-questions decomposed from Q.

## 2.2 The Semantic Convolutional Kernel

The fundamental building block of ConvMem is the semantic convolutional kernel. Unlike standard CNN kernels that perform linear algebraic operations (dot products), our semantic kernel performs a non-linear, text-to-text transformation involving information extraction and compression.

Formally, we define the semantic kernel $\kappa$ as a function parameterized by a frozen LLM and an instruction prompt $p .$ For a given layer, the kernel takes a query $q$ and a local text segment $\textbf { x } \subset \mathbf { \Omega } \mathcal { D }$ (analogous to the receptive field) as input. The output is a tuple consisting of a summarized text and a relevance score:

$$
\begin{array} { r } { ( h , r ) = \mathcal { K } _ { \mathrm { L L M } } ( \mathbf { x } , q ; p ) , } \end{array}\tag{1}
$$

where $h$ is the condensed textual summary of $\mathbf { x } .$ distilling information relevant to $q . r \in \{ 0 , 1 , 2 \}$ is a discrete relevance score indicating the importance of the segment: $r = 0$ (Irrelevant) denotes that the segment contains no useful information; $r \ = \ 1$ (Relevant) denotes the presence of potentially useful background or context; and $r \ = 2$ (Critical) identifies segments containing direct evidence or the answer itself. This semantic convolution performs two functions simultaneously: information extraction (filtering noise) and dimensionality reduction (compressing token length).

To extend this local operation to the full document, analogous to CNNs scanning an entire image, we define the semantic convolution operation Conv(·) as follows:

$$
\mathbf { H } , \mathbf { R } = \operatorname { C o n v } ( \mathcal { D } , \mathcal { K } ) = \left( \bigoplus _ { i = 1 } ^ { M } h _ { i } , \{ r _ { i } \} _ { i = 1 } ^ { M } \right)\tag{2}
$$

where $( h _ { i } , r _ { i } ) = \mathcal { K } _ { \mathrm { L L M } } ( \mathbf { x } _ { i } , q ; p )$ . This operation involves sliding the kernel K across all segments $\mathbf { x } _ { i } \subset \mathcal { D }$ . Here, $\bigoplus _ { i = 1 } ^ { M } h _ { i }$ denotes the sequential concatenation of local summaries $h _ { 1 } , . . . , h _ { M }$ , forming a global summary sequence H. $\mathbf { R } = [ r _ { 1 } , \ldots , r _ { M } ]$ is the sequence of relevance scores corresponding to each $h _ { i } .$ , used to identify critical segments for the semantic skip connection mechanism.

## 2.3 CNN-Inspired Architectural Mechanisms

To effectively adapt the convolutional paradigm to language reasoning, ConvMem incorporates three key mechanisms: configurable strides, skip connections and multi-kernel convolution.

## 2.3.1 Configurable Strides

Standard segmentation often leads to boundary truncation, where semantic dependencies are disrupted by the rigid partitioning. We address this limitation by implementing overlapping sliding windows with stride $S < W$ . The input sequence is sliced into segments where the overlap region of length $W - S$ ensures contextual continuity.

![](images/17a88799301db2530e4a0714dcb90d28cdfef85415767c1a363159bee367442b.jpg)  
Figure 1: Illustration of the hierarchical convolutional mechanism within a single channel. At Layer 0, the semantic kernel $\kappa$ scans raw text segments in parallel to generate condensed summaries. Subsequent layers recursively process the concatenated summaries from the previous layer. After L layers, the global summary is combined with critical raw segments (skip connections) to generate the final answer A.

Specifically, tokens at the edge of segment i are re-processed near the center of segment $i + 1$ , preserving their contextual integrity. This mechanism provides two benefits:

• Contextual Continuity: Overlapping windows preserve the semantic integrity that span segmentation boundaries, effectively eliminate information loss caused by arbitrary truncation.

• Multi-View Robustness: Each token is scanned multiple times (with an over-scanning factor of W/S) from varying window positions. This mechanism cross-verifies information across different receptive fields, mitigating the stochasticity of the LLM kernel and resulting in a robust, ensembled understanding of the content.

## 2.3.2 Skip Connections

Hierarchical summarization entails the risk of losing fine-grained details, such as exact dates or entity names. Inspired by residual networks (He et al., 2016), we introduce a semantic skip connection mechanism based on the relevance score r.

During convolution at the input layer, if the kernel assigns a critical score $r \ = \ 2$ to a segment $\mathbf { x } _ { i } ,$ this raw segment will add a skip connection to the final layer, bypassing the summarization and is directly appended to a residual buffer B:

$$
B  B \cup \{ \mathbf { x } _ { i } \mid \mathrm { S c o r e } ( \mathbf { x } _ { i } ) = 2 \}\tag{3}
$$

This allows the final generation step to reason over high-level abstract summaries while accessing unmodified, high-fidelity evidence from the raw text.

## 2.3.3 Multi-Kernel Convolution

Complex reasoning tasks often involve tracking multiple, intertwined logical threads $( \mathrm { e . g . }$ , “What is the relationship between Person A and Person B?”). A single summary stream may conflate distinct entities or events. To handle this, we employ multi-kernel convolution to treat the text as a multichannel signal.

Given the user query $Q ,$ , we first decomposes it into $C$ distinct sub-questions $\{ q ^ { ( 1 ) } , q ^ { ( 2 ) } , \ldots , q ^ { ( C ) } \}$ Each sub-question initializes a unique kernel ${ \mathcal K } ^ { ( c ) } = { \mathcal K } _ { \mathrm { L L M } } ( \cdot , q ^ { ( c ) } ; p )$ . These kernels operate in parallel on the same document but focus on different semantic aspects, preventing interference between orthogonal reasoning paths. For example, $\kappa ^ { ( 1 ) }$ might focus on “Person $\mathbf { A } ^ { \prime \prime }$ , while $\mathcal { K } ^ { ( 2 ) }$ tracks “Person $\mathbf { B } ^ { \ast }$ . This process generates C distinct reasoning channels:

$$
{ \bf H } ^ { ( c ) } = \mathrm { C o n v } ( \mathcal { D } , \mathcal { K } ^ { ( c ) } )\tag{4}
$$

Accordingly, the global notations defined previously are extended to channel-specific versions $( \mathbf { e . g . } , B \to B ^ { ( c ) } )$ , ensuring isolation between different reasoning threads. This mechanism ensures that the path for each sub-question remains disentangled and preserved until the final aggregation.

## 2.4 Hierarchical Parallel Inference Workflow

ConvMem integrates these components into a hierarchical workflow, transforming the raw document D into a global answer A through L layers.

Layer 0 (Input Processing). The raw document D is sliced into $M _ { 0 }$ overlapping segments $[ \mathbf { x } _ { 1 } ^ { ( 0 ) } , \mathbf { x } _ { 2 } ^ { ( 0 ) } , \ldots , \mathbf { x } _ { M _ { 0 } } ^ { ( 0 ) } ]$ using window size W and stride S. Each kernel $\mathcal { \kappa } ^ { \left( c \right) }$ scans these segments in parallel to produce initial summaries and update the residual buffer $\boldsymbol { B } ^ { ( c ) }$

Layer l (Recursive Summarization). For hidden layers $l > 0 .$ , we perform recursive summarization. For each channel $^ { c , }$ the summaries from the previous layer are first concatenated to form an intermediate text stream $\mathbf { H } ^ { ( l - 1 , c ) }$ . This stream is sliced into new overlapping segments, and the semantic kernel $\mathcal { K } ^ { \left( c \right) }$ is applied recursively using the semantic convolution operation:

$$
{ \bf H } ^ { ( l , c ) } = \mathrm { C o n v } ( { \bf H } ^ { ( l - 1 , c ) } , \mathcal { K } ^ { ( c ) } )\tag{5}
$$

![](images/c4e3c0aa01370093bd1f042862b5eaaae9a1c1e408f377aef861b385976519b9.jpg)

![](images/8987f3ceb6c36f19ec4a9251a545dc0f1900f1dae5724544f04beeeed671e4a5.jpg)  
Figure 2: The overall workflow of ConvMem. (a) Query Decomposition: The complex query Q is disentangled into $C$ sub-questions to isolate reasoning threads. (b) Multi-Kernel Convolution: Each sub-question drives a distinct kernel to scan the document in parallel. Through hierarchical layers, local evidence is recursively summarized and aggregated to synthesize the final global answer A.

ConvMem achieves dimensionality reduction through semantic compression. Assuming an average compression factor α (the ratio of summary length to input length), the total sequence length decays exponentially: $N _ { l } \approx N _ { 0 } \cdot \alpha ^ { l }$ . As the hierarchy deepens, the effective receptive field expands exponentially, condensing the massive document into a manageable set of global representations $\mathbf { H } ^ { ( L , c ) }$

Final Aggregation. At the final layer $L ,$ the inference proceeds in two stages to generate the global answer. First, each channel independently generates a sub-answer $a ^ { ( c ) }$ for its sub-question $q ^ { ( c ) }$ Crucially, this step utilizes both the high-level summaries $\mathbf { H } ^ { ( L , c ) }$ and the high-fidelity raw evidence in the residual buffer $\boldsymbol { B } ^ { ( c ) }$ to ensure precision:

$$
a ^ { ( c ) } = \mathrm { L L M } ( q ^ { ( c ) } \oplus \mathbf { H } ^ { ( L , c ) } \oplus B ^ { ( c ) } )\tag{6}
$$

Second, the model performs global reasoning to derive the final answer $A .$ . It aggregates the original query $Q$ with the sequence of resolved sub-question/sub-answer pairs:

$$
{ \cal A } = \mathrm { L L M } \left( Q \oplus \bigoplus _ { c = 1 } ^ { C } ( q ^ { ( c ) } \oplus a ^ { ( c ) } ) \right)\tag{7}
$$

This design ensures that fine-grained details are resolved at the sub-problem level, while the final aggregation focuses on logical synthesis.

Parallelization. A critical advantage of ConvMem over sequential memory agents (e.g., MemAgent) is the decoupling of temporal dependencies. In MemAgent, step t strictly depends on the memory state of step t − 1, forcing an O(N) sequential execution. In contrast, within any layer l of ConvMem, the operations on all segments are independent. Furthermore, computations across different kernels c are also orthogonal.

This independence enables massive parallelization. Theoretically, the inference latency is determined solely by the tree depth rather than the sequence length:

$$
T _ { \mathrm { l a t e n c y } } \propto O ( \log N )\tag{8}
$$

This logarithmic scaling enables ConvMem to process million-token contexts with latency comparable to processing short contexts.

## 3 Experiments

In this section, we conduct extensive experiments to investigate three key research questions: Q1: Performance Efficacy. How does ConvMem compare to standard language models and memoryenhanced agents on established benchmarks? Q2: Robustness and Scalability. RL-based approaches often suffer from performance degradation when shifting to unseen data distributions. In contrast, does ConvMem demonstrate superior cross-domain consistency compared to these specialists? Furthermore, is ConvMem model-agnostic, enabling immediate performance gains when integrated different backbone LLMs without adaptation? Q3: Mechanism Analysis. How do the proposed architectural components (configurable strides, skip connections, multi-kernel convolution) contribute to the system’s effectiveness?

## 3.1 Experimental Setup

Baselines. To ensure a comprehensive evaluation, we compare ConvMem against baselines categorized into three distinct paradigms:

• Standard LLMs: We employ Qwen2.5- 32B-Instruct (Qwen et al., 2025) as the primary baseline. To investigate performance scaling across model sizes, we also include Qwen2.5-7B-Instruct and Qwen2.5- 72B-Instruct (Team, 2024) as comparative baselines (Team, 2024).

• Training-Free Memory Methods: We evaluate retrieval-based and memory-based approaches, including RAG-BM25 (Mem-Lab, 2024), MemAgent-W/O-RL (Yu et al., 2025), and Mem-α-W/O-RL (Wang et al., 2025). All these methods utilize Qwen2.5-32B-Instruct as the backbone to ensure fair comparison.

• RL-Trained Specialists: We benchmark against MemAgent (Yu et al., 2025) and Memα (Wang et al., 2025). These models represent the RL paradigm, which optimizes sequential memory updates through extensive training.

Datasets. We conduct evaluations on two benchmarks designed to assess both in-distribution and out-of-distribution performance:

• RULER-HotpotQA (In-Distribution): Following the MemAgent protocol, we use the RULER-HotpotQA benchmark (Hsieh et al., 2024; Yang et al., 2018). This dataset adapts multi-hop questions from HotpotQA into a long-context Needle-in-a-Haystack (NIAH) paradigm. Crucially, as MemAgent utilizes this specific data distribution for policy optimization, it serves as the in-distribution testbed.

• RULER-2WikiMultiHopQA (Out-of-Distribution): To rigorously distinguish reasoning from memorization, we synthesize a novel long-context dataset based on 2Wiki-MultiHopQA (Ho et al., 2020) using the same construction protocol as RULER-HotpotQA. This dataset remains unseen during the training of MemAgent, providing a strict OOD setting to evaluate generalization.

Metric. We report four metrics: F1 score, Exact Match (EM), Sub-EM (Sub Exact Match) and LLM-as-a-Judge (ACC ) (Rajpurkar et al., 2016; Zheng et al., 2023). F1 represents the harmonic mean of precision and recall. EM indicates whether the predicted value exactly matches the ground truth. Sub-EM measures whether the predicted value is a subset of the reference value, or vice versa. ACC<sub>L</sub> using LLMs as judges to evaluate the performance. Due to space constraints, our main analysis focuses on F1 and Sub-EM, as they offer a balanced view of retrieval recall and generation flexibility. Full results across all metrics are provided in Appendix C.

Table 1: Main results on RULER-HotpotQA (In-Distribution) and RULER-2WikiMultiHopQA (Out-of-Distribution). We report F1 and Sub-EM scores across context lengths ranging from 28k to 896k. The best results are highlighted in bold, and the second best are underlined.
<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="6">RULER-HotpotQA</td><td colspan="6">2WikiMultiHopQA</td></tr><tr><td>28k</td><td>56k</td><td>112k</td><td>224k</td><td>448k</td><td>896k</td><td>28k</td><td>56k</td><td>112k</td><td>224k</td><td>448k</td><td>896k</td></tr><tr><td>Base Model</td><td>Qwen2.5-32B-Instruct</td><td>F1 Sub-EM</td><td>61.9 69.53</td><td>57.77 63.28</td><td>48.86 53.91</td><td>44.04 44.53</td><td>18.59 17.19</td><td>16.78 17.97</td><td>56.52 67.19</td><td>57.2 65.62</td><td>50.45 62.5</td><td>41.73 50.0</td><td>29.45 34.38</td><td>20.93 28.12</td></tr><tr><td rowspan="4">Training-Free Methods</td><td>RAG-BM25</td><td>F1 Sub-EM</td><td>45.98 46.09</td><td>51.11 50.00</td><td>47.21 48.44</td><td>35.86</td><td>29.27 35.9430.47</td><td>30.67 32.81</td><td>34.54 57.03</td><td>31.17 58.59</td><td>24.55 47.66</td><td>23.19 46.88</td><td>21.25 39.06</td><td>17.34 43.75</td></tr><tr><td>MemAgent-W/O-RL</td><td>F1 Sub-EM</td><td>63.95 67.97</td><td>65.28</td><td>62.52</td><td>57.98</td><td>55.42</td><td>58.73</td><td>60.51</td><td>57.42</td><td>45.31</td><td>50.6</td><td>47.66</td><td>49.65</td></tr><tr><td>Mem-α-W/O-RL</td><td>F1</td><td>6.13</td><td>71.09 7.78</td><td>65.62 6.5</td><td>59.38 7.03</td><td>56.25 7.19</td><td>60.16 7.35</td><td>71.09 6.56</td><td>67.97 6.47</td><td>58.59 5.94</td><td>63.28 6.19</td><td>60.16 5.19</td><td>62.5 6.0</td></tr><tr><td>MemAgent</td><td>Sub-EM F1</td><td>56.25 75.55</td><td>53.12 75.20</td><td>43.75 75.26</td><td>46.88 73.54</td><td>46.88 73.10</td><td>49.27 68.80</td><td>57.38 60.92</td><td>59.38 60.11</td><td>62.5 63.19</td><td>53.12 58.82</td><td>53.12 58.5</td><td>50.0 58.41</td></tr><tr><td>RL-Trained Methods</td><td>Mem-α</td><td>Sub-EM F1 Sub-EM</td><td>79.69 5.84 34.38</td><td>79.69 6.81 40.62</td><td>80.47 7.0</td><td>77.34 6.09</td><td>78.91 1.31</td><td>74.22 0.69</td><td>70.31 1.25</td><td>71.09 1.13</td><td>72.66 0.94</td><td>69.53 1.03</td><td>67.97 0.97</td><td>69.31 1.09</td></tr><tr><td>Ours</td><td>ConvMem</td><td>F1 Sub-EM</td><td>67.44 73.44</td><td>67.86 72.66</td><td>40.62 57.81 67.19</td><td>43.75 63.27 69.53</td><td>43.75 56.14 62.50</td><td>25.0 63.09 69.53</td><td>59.38 72.3 82.81</td><td>56.25 71.25 82.47</td><td>50.0 67.21 77.34</td><td>62.5 61.96 71.88</td><td>50.0 61.33 73.31</td><td>53.12 59.06 70.62</td></tr></table>

Implementation Details. We set the kernel size W = 8000 tokens and stride S = 1600, implying scanning 5 times for each token. The number of channels C is dynamically determined by the LLM during query decomposition. ConvMem requires no parameter updates. Prompt templates are detailed in Section F.

## 3.2 Main Results and Analysis

Table 1 presents the performance comparison across varying context lengths. As observed, ConvMem establishes a new state-of-the-art among training-free methods and demonstrates superior robustness compared to RL-based specialists.

Superiority Over Training-Free Baselines. On both datasets, ConvMem significantly outperforms the vanilla Qwen2.5-32B-Instruct and all trainingfree baselines. Retrieval-based methods (RAG) struggle with multi-hop reasoning as they retrieve isolated chunks, disrupting logical dependencies. Sequential methods (MemAgent-W/O-RL) suffer from cumulative forgetting and the interference of intertwined logical threads, rendering early evidence unrecoverable.

Case 1 illustrates this failure mode: sequential agents MemAgent-W/O-RL encounter the entity "Shirley Temple Black" early in the stream but fail to link it to the film “Kiss and Tell” appearing much later, as the initial memory is overwritten. In contrast, ConvMem’s hierarchical aggregation preserves both facts in parallel channels, successfully retrieving the answer.

Case 1: Mitigating Sequential Forgetting. Query: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?

Failure Analysis (MemAgent-W/O-RL): The agent identifies “Shirley Temple Black” early but forgets her government position by the time the film “Kiss and Tell” appears later in the stream, due to the interference of intertwined logical threads.

ConvMem Success: Thanks to its multi-kernel convolution and hierarchical aggregation, ConvMem retains both the entity attribute and the film relation, enabling successful multi-hop reasoning.

Robustness Against RL-Induced Instability. We observe that RL-based baselines suffer from distinct instability issues. Mem-α achieves a significantly lower F1 score compared to ConvMem. This is due to its tendency to output excessively verbose answers, which degrades precision.

While MemAgent peaks on its training domain (RULER-HotpotQA), its performance collapses on the unseen RULER-2WikiMultiHopQA dataset. This discrepancy suggests that RL models may prioritize parametric memorization over in-context reasoning. Case 2 provides empirical evidence of this phenomenon. The query asks to choose between “Lev Yilmaz” or “Pamela B. Green”. However, the dataset ground truth contains a typo “Levni Yilmaz”. MemAgent outputs “Levni Yilmaz”, ignoring the provided context to match the memorized label. ConvMem faithfully extracts “Lev Yilmaz” from the context. Although penalized by the metric, ConvMem demonstrates superior faithfulness to the input, whereas the RL agent exhibits hallucination from priors.

Case 2: Correct Answer via Parametric Hallucination.

Query: Which filmmaker was known for animation, Lev Yilmaz or Pamela B. Green?

Dataset Label (Typo): Levni Yilmaz.

MemAgent Output: Levni Yilmaz (Matches label, contradicts context).

ConvMem Output: Lev Yilmaz (Matches context, penalized as wrong).

This case highlights a critical limitation of RLbased agents: they risk regressing into parametric retrieval systems that prioritize memorized priors over input evidence. In contrast, ConvMem, by design, functions as a faithful reasoning engine grounded in the provided context.

Model Agnosticism and Scalability. A key advantage of ConvMem is its model-agnostic nature. As shown in Fig. 3, applying ConvMem to backbones of varying sizes (from 7B to 72B) yields immediate performance gains. This confirms that our method effectively scales the long-context reasoning of diverse LLMs without adaptation costs.

## 3.3 Ablation Study

We conduct component-wise ablations on RULER-2WikiMultiHopQA to validate our design choices.

![](images/327aa302df3f337860b5599d0473f6677ec8afabe1cd6569e9d090361fe56476.jpg)  
Figure 3: ConvMem yields consistent performance improvements across diverse backbone sizes (7B-72B), validating its effective scaling capabilities.

![](images/68e9a2de8a95aa61c4ce7a01f679f806bb605011beeb962dd3f73ed41a0b88f5.jpg)  
Figure 4: (Left) Higher over-scanning factors (W/S) improve recall by mitigating segmentation boundaries, optimal at 5×. (Right) The absence of the residual path results in noticeable degradation, confirming the necessity of propagating raw high-fidelity evidence.

Impact of Strides (Over-Scanning Factor). We investigate the effect of the over-scanning factor (W/S), which determines how many times each token is processed. As shown in Fig. 4 (Left), single-pass scanning (S = W) leads to a noticeable performance drop due to boundary truncation. Increasing this factor consistently improves performance, with 5× over-scanning achieving the optimal balance between coverage and noise.

Effect of Skip Connections. Removing skip connections (Fig. 4, Right) results in a degradation of fine-grained details, particularly for exact entity retrieval. This confirms that skip connections function as a semantic highway, allowing highconfidence evidence to bypass compression loss.

Significance of Multi-Kernel Convolution. We compare the single-kernel against our multi-kernel convolution. Results indicate that query decomposition significantly boosts accuracy on multi-hop tasks (Fig. 5, Left). Single-kernel models often conflate distinct reasoning threads, whereas multikernel convolution successfully disentangles semantic dependencies, reducing interference.

Kernel Size Sensitivity. We test kernel sizes W ∈ {500, 5000, 8000, 10000}. Extremely small kernels (500) fragment semantic context, while overly large kernels (10000) dilute local signal density. W = 8000 provides an optimal balance between context coherence and signal extraction.

## 4 Related Work

Existing long-context approaches generally fall into three categories: architectural extrapolation, memory-augmented systems, and reinforcement learning optimizations.

![](images/6bfbf45cd9f2f12bdb23439909cf5e82766f7f4d4680235d83007a2704e2d1c0.jpg)  
Figure 5: (Left) Decomposing queries into parallel channels $( C > 1 )$ significantly boosts performance by disentangling interference between reasoning threads. (Right) $W = 8 0 0 0$ achieves the optimal trade-off between context coherence and signal density.

## 4.1 Long-Context Extrapolation and Efficient Architectures

To process inputs exceeding pre-training limits, research has focused on positional extrapolation (e.g., PI, YaRN, Ring Attention) (Chen et al., 2023; Peng et al., 2023; Liu et al., 2023) and efficiency optimizations via Linear/Sparse Attention (Xiao et al., 2023; Katharopoulos et al., 2020; Child et al., 2019) or KV Cache compression (Li et al., 2024; Zhang et al., 2023b). Despite these advancements, native long-context models often suffer from the lost-in-the-middle phenomenon (Liu et al., 2024b), where reasoning performance degrades as effective context length increases. Furthermore, processing massive documents in a single pass imposes prohibitive hardware demands and latency. In contrast, ConvMem bypasses the quadratic bottleneck by decomposing the context into manageable hierarchical receptive fields, ensuring robust reasoning without modifying the underlying attention.

## 4.2 Retrieval-/Memory-Augmented Agents

To decouple context length from computational cost, recent works employ external memory systems. Retrieval-Augmented Generation (RAG) (Lewis et al., 2020) reduces context by fetching top-k chunks, but it often fails on multihop reasoning tasks due to the retrieval of isolated fragments lacking global connectivity. Alternatively, Sequential Memory Agents, such as MemGPT (Packer et al., 2023), Mem0 (Chhikara et al., 2025), and MIRIX (Wang and Chen, 2025), MemAgent (Yu et al., 2025) treat text as a continuous stream, maintaining a persistent memory state that evolves step-by-step. While effective, these sequential methods introduce a temporal dependency (O(N)) that prohibits parallelization, leading to high inference latency. ConvMem addresses these limitations by adopting a tree-structured, CNNinspired topology $( O ( \log N ) )$ , enabling massive parallelization and shortening the information propagation path to minimize error accumulation.

## 4.3 Reinforcement Learning for Memory Optimization

Acknowledging that heuristic memory updates may be suboptimal, recent works employ RL to learn memory policies, following the long tradition of exploiting memory in RL (Zhang et al., 2023a, 2024b,a). For instance, MemAgent (Yu et al., 2025) utilizes Multi-Conv DAPO to optimize memory overwrite decisions, while Memory-R1 (Yan et al., 2025) and Mem-α (Wang et al., 2025) train agents to manage complex memory structures via reward signals derived from downstream task accuracy. While RL yields performance gains on in-domain benchmarks, it introduces significant training costs and are susceptible to overfitting dataset. In contrast, ConvMem proposes a training-free paradigm. By leveraging the intrinsic instruction-following capabilities of pre-trained LLMs rather than biased reward optimization, ConvMem achieves superior robustness and true in-context reasoning.

## 5 Conclusion

In this work, we propose ConvMem, a trainingfree framework that reformulates long-context reasoning as a hierarchical, multi-channel convolutional process. By reconceptualizing LLM-based reasoning through the lens of Convolutional Neural Networks, ConvMem successfully transforms the linear dependency chain $( O ( N ) )$ ) into a logarithmic tree structure (O(log N)), effectively mitigating cumulative error propagation. Our empirical evaluation reveals that this principled adaptation, leveraging multi-kernel convolution, configurable strides, and skip connections, achieves competitive performance on standard benchmarks while demonstrating superior generalization on out-ofdistribution tasks compared to RL-trained specialists. Furthermore, our analysis exposes the tendency of RL-based agents to overfit to dataset artifacts, suggesting that ConvMem’s training-free, context-anchored approach provides a more robust path for genuine in-context reasoning. We hope this work encourages the community to further explore architectural innovations that balance computational efficiency with reasoning fidelity.

## 6 Limitations

While ConvMem offers significant advantages in latency and robustness, we acknowledge two primary limitations that point towards future research directions. First, regarding token consumption vs. latency trade-off, although ConvMem achieves logarithmic latency through parallelization, the total computational cost is higher than that of linear scanning methods. Second, the system exhibits a dependency on query decomposition. The efficacy of our parallel convolution relies on the backbone model’s ability to correctly disentangle complex queries into orthogonal sub-questions. If the initial decomposition is flawed, subsequent kernels may operate on incomplete premises.

## 7 Ethical Considerations

This work proposes a training-free framework for long-context reasoning. While our approach requires multiple passes over the text, increasing inference computation, it eliminates the energy consumption associated with model training. Limitations regarding potential biases stem from the underlying frozen LLMs used as kernels. All datasets used in this study are publicly available, and no private data was involved in the experiments.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595.

Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. 2025. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.

Sijia Cui, Aiyao He, Shuai Xu, Hongming Zhang, Yanna Wang, Qingyang Zhang, Yajing Wang, and Bo Xu. 2025. Self-guided function calling in large language

models via stepwise experience recall. arXiv preprint arXiv:2508.15214.

Hao Dong, Zihan Ding, and Shanghang Zhang. 2020. Deep Reinforcement Learning: Fundamentals, Research and Applications. Springer Nature.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. 2016. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770– 778.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. arXiv preprint arXiv:2011.01060.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. 2024. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and François Fleuret. 2020. Transformers are rnns: Fast autoregressive transformers with linear attention. In International conference on machine learning, pages 5156–5165. PMLR.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings ofthe 29th symposium on operating systems principles, pages 611–626.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, and 1 others. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33:9459– 9474.

Yuhong Li, Yingbing Huang, Bowen Yang, Bharat Venkitesh, Acyr Locatelli, Hanchen Ye, Tianle Cai, Patrick Lewis, and Deming Chen. 2024. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947–22970.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024a. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Hao Liu, Matei Zaharia, and Pieter Abbeel. 2023. Ring attention with blockwise transformers for nearinfinite context. arXiv preprint arXiv:2310.01889.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024b. Lost in the middle: How language

models use long contexts. Transactions ofthe Associationfor Computational Linguistics, 12:157–173.

Mem-Lab. 2024. Qwen2.5-7b-rl-rag-q2-em-release commit history. https://huggingface.co/ Mem-Lab/Qwen2.5-7B-RL-RAG-Q2-EM-Release/ commits/main.

Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G Patil, Ion Stoica, and Joseph E Gonzalez. 2023. Memgpt: Towards llms as operating systems. arXiv preprint arXiv:2310.08560.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. 2023. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071.

Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, and 25 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Richard S Sutton and Andrew G Barto. 2018. Reinforcement learning: An introduction. MIT press.

Kimi Team, Yifan Bai, Yiping Bao, Guanduo Chen, Jiahao Chen, Ningxin Chen, Ruijue Chen, Yanru Chen, Yuankun Chen, Yutian Chen, and 1 others. 2025. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

Yu Wang and Xi Chen. 2025. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957.

Yu Wang, Ryuichi Takanobu, Zhiqi Liang, Yuzhen Mao, Yuanzhe Hu, Julian McAuley, and Xiaojian Wu. 2025. Mem-{\alpha}: Learning memory construction via reinforcement learning. arXiv preprint arXiv:2509.25911.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.

Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Kristian Kersting, Jeff Z Pan, Hinrich Schütze, and 1 others. 2025.

Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828.

Wei Yang, Jinwei Xiao, Hongming Zhang, Qingyang Zhang, Yanna Wang, and Bo Xu. 2025. Coarse-tofine grounded memory for llm agent planning. arXiv preprint arXiv:2508.15305.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380.

Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, and 1 others. 2025. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259.

Hongming Zhang, Tongzheng Ren, Chenjun Xiao, Dale Schuurmans, and Bo Dai. 2024a. Provable representation with efficient planning for partially observable reinforcement learning. In Proceedings ofthe 41st International Conference on Machine Learning, pages 59759–59782.

Hongming Zhang, Chenjun Xiao, Chao Gao, Han Wang, Martin Müller, and 1 others. 2024b. Exploiting the replay memory before exploring the environment: enhancing reinforcement learning through empirical mdp iteration. Advances in Neural Information Processing Systems, 37:85658–85692.

Hongming Zhang, Chenjun Xiao, Han Wang, Jun Jin, Bo Xu, and Martin Müller. 2023a. Replay memory as an empirical MDP: Combining conservative estimation with experience replay. In The Eleventh International Conference on Learning Representations.

Hongming Zhang and Tianyang Yu. 2020. Taxonomy of reinforcement learning algorithms. Deep reinforcement learning: Fundamentals, research and applications, pages 125–133.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark Barrett, and 1 others. 2023b. H2o: Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36:46595–46623.

## Appendix

## A Dataset Details

In this section, we provide detailed descriptions of the source datasets and the construction methodology for the long-context benchmarks used in our experiments.

HotpotQA. HotpotQA (Yang et al., 2018) is a large-scale dataset designed for multi-hop question answering, collected from Wikipedia. Unlike standard QA datasets that require only a single document, HotpotQA necessitates reasoning across multiple documents (typically two or more) to derive the correct answer. Crucially, it provides supporting sentences to ensure the reasoning process is explainable. In the context of long-context evaluation, these supporting sentences serve as the ground truth for identifying golden paragraphs containing the necessary evidence.

2WikiMultiHopQA. 2WikiMultiHopQA (Ho et al., 2020) builds upon the structure of HotpotQA but introduces more rigorous evidence chains. It utilizes Wikipedia articles and structured Wikidata to generate questions. A key advantage of 2Wiki-MultiHopQA is its comprehensive evidence paths (including entity-relation triples), which reduce the likelihood of shortcut reasoning often observed in HotpotQA.

RULER-HotpotQA. We adopt the RULER-HotpotQA (Hsieh et al., 2024) benchmark introduced by Yu et al. (2025). This dataset synthesizes samples from HotpotQA by iteratively injecting distractor paragraphs to reach specific target context lengths (ranging from 28k to 896k tokens). Notably, Yu et al. (2025) applied a Best-Of-2 filtering mechanism during construction to ensure task difficulty, discarding questions that base models could answer with 100% accuracy using only internal parametric knowledge. We utilize this established benchmark to rigorously evaluate contextual reasoning capabilities in an in-distribution setting.

RULER-2WikiMultiHopQA (Ours). Building upon the synthesis framework of RULER-HotpotQA used in MemAgent, we construct a novel OOD benchmark using the 2WikiMulti-HopQA corpus. We employ a similar Needle-in-a-Haystack approach, inserting relevant paragraphs into a vast pool of noise documents to extend context lengths while preserving the original multi-hop logic. By transitioning to 2WikiMultiHopQA, we mitigate the inherent ambiguity issues present in HotpotQA and ensure that performance metrics reflect genuine retrieval and reasoning capabilities rather than dataset-specific biases. This dataset serves to validate the cross-domain generalizability of long-context agents.

## B Implementation Details

In this section, we provide a comprehensive breakdown of the experimental environment, inference infrastructure, and algorithmic configurations to facilitate reproducibility.

## B.1 Inference Infrastructure

All experiments were conducted on a highperformance computing cluster equipped with NVIDIA A800 (80GB) GPUs. To maximize inference throughput, we leveraged the vLLM library (Kwon et al., 2023), utilizing its PagedAttention mechanism and continuous batching to manage the massive KV cache requirements of long-context processing. For the backbone model Qwen2.5-32B-Instruct, we employed bfloat16 precision to maintain numerical stability while optimizing memory usage. To handle the concurrent kernel executions in ConvMem, we implemented an asynchronous job scheduler that dynamically batches independent segment inputs into the vLLM engine, ensuring maximum GPU utilization.

## B.2 Algorithmic Configurations

Hierarchical Recursion. The hierarchical summarization process in ConvMem continues recursively until the total token count of the concatenated summaries fits within the kernel size W. Specifically, if the length of the intermediate context $\it { h ^ { ( l ) } }$ exceeds W, a new layer l+1 is instantiated. This typically results in a tree depth of $L \in [ 2 , 3 ]$ for context lengths up to 128k, and $L \in [ 3 , 4 ]$ for extreme lengths up to 1M tokens, depending on the compression rate α.

Query Decomposition and Parsing. For multikernel convolution, we enforce a flexible upper limit of $C _ { \mathrm { m a x } } = 5$ sub-questions to prevent channel explosion, though in practice, the model typically generates 2-4 sub-questions. We employ robust regular expression parsing to extract structured JSON outputs (keywords, sub-questions, and relevance scores) from the LLM responses. In rare cases of parsing failure, the system falls back to a default relevant state (r = 1) to prevent information loss.

## B.3 Baselines Setup

To ensure a fair comparison, all baselines (including RAG and MemAgent) utilize the same backbone (Qwen2.5-32B-Instruct) and identical hardware environment. For RAG-BM25, we retrieve the top-k chunks where k is dynamically calculated to fill the model’s effective context window. For MemAgent, we adhere strictly to the hyperparameter settings reported in the original paper (Yu et al., 2025).

## B.4 Hyperparameters

The specific hyperparameters used for ConvMem across all experiments are detailed in Table 2.

Table 2: Detailed Hyperparameters for ConvMem.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Model Settings Backbone Model Precision Inference Engine</td><td>Qwen2.5-32B-Instruct bfloat16 vLLM (v0.6.0)</td></tr><tr><td>ConvMem Architecture Kernel Size (W) Stride (S) Over-scanning Factor (W/S) Max Channels (Cmax)</td><td>8,000 tokens 1,600 tokens 5 10</td></tr><tr><td>Recursion Stop Condition Generation Parameters</td><td>Sequence Length &lt; W</td></tr><tr><td>Temperature</td><td>0.7</td></tr><tr><td>Top-P</td><td>0.95</td></tr><tr><td>Skip Connection Flag</td><td>True</td></tr></table>

## C Full Experimental Results

We present the comprehensive performance comparison across all context lengths (28k-896k) with four metrics in Table 3.

Table 3: Full performance comparison on RULER-HotpotQA and 2WikiMultiHopQA datasets with four metrics: : F1, Exact Match (EM), Substring Exact Match (Sub-EM), and LLM-as-a-Judge $( \mathrm { A C C } _ { L } )$ .
<table><tr><td rowspan="2">Category</td><td rowspan="2">Model</td><td rowspan="2">Metric</td><td colspan="6">RULER-HotpotQA</td><td colspan="6">2WikiMultiHopQA</td></tr><tr><td>28k</td><td>56k</td><td>112k</td><td>224k</td><td>448k</td><td>896k</td><td>28k</td><td>56k</td><td>112k</td><td>224k</td><td>448k</td><td>896k</td></tr><tr><td rowspan="4">Base Model</td><td rowspan="4">Qwen2.5-32B-Instruct</td><td>F1</td><td>61.9</td><td>57.77</td><td>48.86</td><td>44.04</td><td>18.59</td><td>16.78</td><td>56.52</td><td>57.2</td><td>50.45</td><td>41.73</td><td>29.45</td><td>20.93</td></tr><tr><td>EM</td><td>47.66</td><td>44.53</td><td>34.38</td><td>32.81</td><td>8.59</td><td>7.03</td><td>47.66</td><td>47.66</td><td>42.97</td><td>34.38</td><td>24.22</td><td>17.19</td></tr><tr><td>Sub-EM</td><td>69.53</td><td>63.28</td><td>53.91</td><td>44.53</td><td>17.19</td><td>17.97</td><td>67.19</td><td>65.62</td><td>62.5</td><td>50.0</td><td>34.38</td><td>28.12</td></tr><tr><td>ACCL</td><td>80.62</td><td>75.78</td><td>66.13</td><td>60.31</td><td>33.09</td><td>30.16</td><td>72.81</td><td>71.76</td><td>68.05</td><td>55.27</td><td>38.79</td><td>31.76</td></tr><tr><td rowspan="8">Training-Free Methods</td><td rowspan="4">RAG-BM25</td><td>F1</td><td>45.98</td><td>51.11</td><td>47.21</td><td>35.86</td><td>29.27</td><td>30.67</td><td>34.54</td><td>31.17</td><td>24.55</td><td>23.19</td><td>21.25</td><td>17.34</td></tr><tr><td>EM</td><td>32.03</td><td>36.72</td><td>34.38</td><td>21.88</td><td>19.53</td><td>20.31</td><td>25.0</td><td>21.88</td><td>16.51</td><td>17.19</td><td>15.62</td><td>10.94</td></tr><tr><td>Sub-EM</td><td>46.09</td><td>50.00</td><td>48.44 35.94</td><td></td><td>30.47</td><td>32.81</td><td>57.03</td><td>58.59</td><td>47.66</td><td>46.88</td><td></td><td>39.0643.75</td></tr><tr><td>ACCL</td><td>56.91</td><td>62.19</td><td>59.80</td><td>49.41</td><td>46.68</td><td>46.80</td><td>59.26</td><td>56.91</td><td>48.24</td><td>45.51</td><td>39.53</td><td>38.55</td></tr><tr><td rowspan="4">MemAgent-W/O-RL</td><td>F1</td><td>63.95</td><td>65.28</td><td>62.52</td><td>57.98</td><td>55.42</td><td>58.73</td><td>60.51</td><td>57.42</td><td>45.31</td><td>50.6</td><td>47.66</td><td>49.65</td></tr><tr><td>EM</td><td>45.31</td><td>48.44</td><td>44.53</td><td>40.62</td><td>39.06</td><td>42.97</td><td>46.88</td><td>49.22</td><td>45.31</td><td>39.84</td><td>36.72</td><td>39.06</td></tr><tr><td>Sub-EM</td><td>67.97</td><td>71.09</td><td>65.62</td><td>59.38</td><td>56.25</td><td>60.16</td><td>71.09</td><td>67.97</td><td>58.59</td><td>63.28</td><td>60.16</td><td>62.5</td></tr><tr><td>ACCL F1</td><td>80.62 6.13</td><td>81.37</td><td>78.63</td><td>73.32</td><td>72.11</td><td>76.13</td><td>76.91</td><td>71.99</td><td>66.21</td><td>67.07</td><td>65.12</td><td>64.54</td></tr><tr><td rowspan="4"></td><td rowspan="4">Mem-α-W/O-RL</td><td>EM</td><td>0.0</td><td>7.78 0.0</td><td>6.5 0.0</td><td>7.03 0.0</td><td>7.19 0.0</td><td>7.35 0.0</td><td>6.56 0.0</td><td>6.47 0.0</td><td>5.94 0.0</td><td>6.19 0.0</td><td>5.19 0.0</td><td>6.0 0.0</td></tr><tr><td>Sub-EM</td><td>56.25</td><td>53.12</td><td>43.75</td><td>46.88</td><td>46.88</td><td>48.36</td><td>57.38</td><td>59.38</td><td>62.5</td><td>53.12</td><td>53.12</td><td>50.0</td></tr><tr><td>ACCL</td><td>73.75</td><td>72.03</td><td>65.16</td><td>67.19</td><td>67.66</td><td>49.27</td><td>62.97</td><td>62.91</td><td>55.94</td><td>58.59</td><td>60.31</td><td>54.0</td></tr><tr><td>F1</td><td>75.55</td><td>75.20</td><td>75.26</td><td>73.54</td><td>73.10</td><td></td><td>60.92</td><td>60.11</td><td></td><td></td><td></td><td>58.41</td></tr><tr><td rowspan="4">RL-Trained Methods</td><td rowspan="4">MemAgent</td><td>EM</td><td>60.16</td><td>59.38</td><td>55.47</td><td>56.25</td><td>55.47</td><td>68.80 52.34</td><td>46.88</td><td>48.44</td><td>63.19 50.0</td><td>58.82 46.09</td><td>58.5 46.88</td><td>46.09</td></tr><tr><td>Sub-EM</td><td>79.69</td><td>79.69</td><td>80.47</td><td>77.34</td><td>78.91 74.22</td><td></td><td>70.31</td><td>71.09</td><td>72.66</td><td>69.53</td><td>67.97</td><td>69.31</td></tr><tr><td>ACCL</td><td>87.99</td><td>87.03</td><td>87.29</td><td>86.27</td><td>83.10</td><td>80.03</td><td>73.55</td><td>73.41</td><td>72.66</td><td>72.11</td><td>70.12</td><td>70.95</td></tr><tr><td>F1</td><td>5.84</td><td>6.81</td><td>7.0</td><td>6.09</td><td>1.31</td><td>0.69</td><td>1.25</td><td>1.13</td><td>0.94</td><td>1.03</td><td>0.97</td><td>1.09</td></tr><tr><td rowspan="4"></td><td rowspan="4">Mem-α</td><td>EM</td><td>0.0</td><td>0.0 0.0</td><td></td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Sub-EM</td><td>34.38</td><td>40.62</td><td>40.62</td><td>43.75</td><td>43.75</td><td>25.0</td><td>59.38</td><td>56.25</td><td>50.0</td><td>62.5</td><td>50.0</td><td>53.12</td></tr><tr><td>ACCL</td><td>55.16</td><td>58.75</td><td>68.59</td><td>55.31</td><td>32.03</td><td>36.72</td><td>44.69</td><td>39.22</td><td>39.38</td><td>40.0</td><td>30.78</td><td>53.12</td></tr><tr><td>F1</td><td>67.44</td><td>67.86</td><td>57.81</td><td>63.27</td><td>56.14</td><td>63.09</td><td>72.3</td><td>71.25</td><td>67.21</td><td>61.96</td><td>61.33</td><td>59.06</td></tr><tr><td rowspan="4">Ours</td><td rowspan="4">ConvMem</td><td>EM</td><td>47.66</td><td>47.66</td><td>39.06</td><td>43.75</td><td>39.06</td><td>44.53</td><td>60.94</td><td>57.03</td><td>52.36</td><td>50.78</td><td></td><td>51.5646.88</td></tr><tr><td>Sub-EM</td><td>73.44</td><td>72.66</td><td>67.19</td><td>69.53</td><td>62.50</td><td>69.53</td><td>82.81</td><td>82.47</td><td>77.34</td><td>71.88</td><td></td><td>73.31 70.62</td></tr><tr><td>ACCL</td><td>83.79</td><td>81.72</td><td>74.96</td><td>78.98</td><td>72.54</td><td>77.66</td><td>85.31</td><td></td><td></td><td></td><td></td><td>80.94 75.94 72.5873.5971.36</td></tr><tr><td></td><td></td><td></td><td></td></table>

## D Qualitative Analysis of Model Behaviors

In this section, we analyze the distinct failure modes of baseline models, providing empirical evidence for the limitations of sequential architectures and the overfitting tendencies of RL-trained agents.

## D.1 Sequential Information Loss in Recurrent Architectures

For multi-hop questions where evidence is distributed across the text stream, sequential memory agents (e.g., MemAgent) suffer from a forgetting bottleneck. Since they update a fixed-size memory step-by-step, early information that seems irrelevant at the time of reading is often discarded, making it unrecoverable when its relevance is later revealed by subsequent context.

## Case 1: Disconnected Evidence

Query: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell? Context:

Document 1053: Shirley Temple Shirley Temple Black (April 23, 1928 – February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood’s number one box-office draw as a child actress from 1935 to 1938. As an adult, she was named United States ambassador to Ghana and to Czechoslovakia and also served as Chief of Protocol of the United States.

• Document 1348: Kiss and Tell (1945 film)

Kiss and Tell is a 1945 American comedy film starring then 17-year-old Shirley Temple as Corliss Archer. In the film, two teenage girls cause their respective parents much concern when they start to become interested in boys. The parents’ bickering about which girl is the worse influence causes more problems than it solves.

Analysis: The document mentioning “Shirley Temple Black” (Document 1053) appears early in the stream. The document linking her to the film “Kiss and Tell” (Document 1348) appears much later. By the time the agent reads Document 1348 and realizes the relevance of Shirley Temple, the memory of her specific government position (Ambassador/Chief of Protocol) has already been forgotten. ConvMem avoids this by processing both segments in parallel channels.

## Case 2: Retrospective Reasoning

Query: Who is the younger brother of the episode guest stars of The Hard Easy? Context:

• Document 591: Brian Doyle-Murray Brian Doyle-Murray (born Brian Murray, October 31, 1945) is an American actor, voice artist, comedian and screenwriter. He is the older brother of actor/comedian Bill Murray, and the two have acted together in several films, including “Caddyshack”, “Scrooged”, “Ghostbusters II”, “The Razor’s Edge”, and “Groundhog Day”. He co-starred on the TBS sitcom on “Sullivan & Son”, where he played the foul-mouthed Hank Murphy. he also appeared in the Cartoon Network original animated series “The Marvelous Misadventures of Flapjack” as the surly Captain K’Nuckles and a pirate ghost, The Flying Dutchman from the Nickelodeon animated series, “SpongeBob SquarePants”, he appears in a recurring role as Don Ehlert on the ABC sitcom “The Middle”.

• Document 1348: The Hard Easy (Adventure Time)

“The Hard Easy” is the twenty-third episode of the fourth season of the American animated television series “Adventure Time”. The episode was written and storyboarded by Tom Herpich and Skyler Page, from a story by Patrick McHale, Kent Osborne, and Pendleton Ward. It originally aired on Cartoon Network on October 1, 2012. The episode guest stars Brian Doyle-Murray as Prince Huge and Jonathan

## Katz as the Mudscamp elder.

Analysis: Similar to Case 1, the relationship between Brian Doyle-Murray and Bill Murray is established in Document 591. The connection to the episode “The Hard Easy” is only revealed in Document 1348. A sequential agent reading Document 591 has no incentive to retain the sibling relationship in memory, leading to failure when the query eventually demands this specific fact. Sequential agents lack the “global view” required to link these distant facts.

## D.2 RL-Induced Overfitting and Parametric Hallucination

Our experiments reveal that RL-trained agents (e.g., MemAgent) tend to overfit to the training data distribution, prioritizing memorized parametric priors over the provided context.

## Case 3: Parametric Bias vs. Context

Query: Which filmmaker was known for animation, Lev Yilmaz or Pamela B. Green? Context Evidence: The document explicitly states “Lev Yilmaz”.

Analysis: The RL agent ignores the provided context and outputs the memorized label (Levni), proving it relies on parametric priors rather than reasoning. ConvMem faithfully extracts “Lev Yilmaz”, demonstrating adherence to the input.

## Case 4: Hallucinating Collaborators

Query: Ellie Goulding worked with what other writers on her third studio album, Delirium? Context:

• Document 37:On My Mind (Ellie Goulding song) "On My Mind" is a song by English singer Ellie Goulding from her third studio album "Delirium" (2015). It was released as the album’s lead single on 17 September 2015. It was written by Goulding, Max Martin, Savan Kotecha and Ilya Salmanzadeh. "On My Mind" is an electropop and R&B song whose instrumentation consists of scratchy guitars, trap drums, slapped beats and sharp, syncopated electronica. Lyrically, "On My Mind" talks about a one-night stand with someone the protagonist shouldn’t be with, having a dichotomy between heart and head. Though firmly denied by Goulding, many critics considered it an answer song to Ed Sheeran’s "Don’t"

• Document 41:Love Me like You Do "Love Me like You Do " is a song recorded by English singer Ellie Goulding for the "Fifty Shades of Grey" (2015). The song was written by Savan Kotecha, Ilya Salmanzadeh, Tove Lo, Max Martin and Ali Payami; the latter two also produced it. Goulding was selected to sing the track. It was released on 7 January 2015 as the second single from the soundtrack. The song was also included on Goulding’s third studio album, "Delirium" (2015).

Ground Truth: Max Martin, Savan Kotecha and Ilya Salmanzadeh Analysis: In this example, both “On My Mind” and “Love Me Like You Do” are featured on the album Delirium. The former’s collaborators include Max Martin, Savan Kotecha, and Ilya Salmanzadeh, while the latter’s collaborators include Savan Kotecha, Ilya Salmanzadeh, Tove Lo, Max Martin, and Ali Payam. Large models that have undergone enhanced training consistently provide the correct answer—the former. Observing untrained large models reveals a tendency to list more collaborators for the latter, as it includes all collaborators from the former.

## E Data Quality and Metric Justification

Standard Exact Match (EM) is often too rigid for long-context generation. In this section, we justify our use of robust metrics (Sub-EM and ACC<sub>L</sub>) using concrete examples of dataset ambiguity and valid variations.

## E.1 Dataset Ambiguity and Missing Evidence

Some queries in the synthetic benchmarks are inherently ambiguous or unanswerable based strictly on the provided context, highlighting flaws in the dataset generation process.

## Case 5: Missing Evidence

Query: Brown State Fishing Lake is in a country that has a population of how many inhabitants?

## Context:

Document 44:Brown County, Kansas Brown County (county code BR) is a county located in the northeast portion of the U.S. state of Kansas. As of the 2010 census, the county population was 9,984. Its county seat and most populous city is Hiawatha. Brown County is the location of the Kickapoo Indian Reservation of Kansas, the majority of the Sac and Fox Reservation and the majority of the Iowa Reservation of Kansas and Nebraska.

LLM answer: The problem is asking for the population of the country where Brown State Fishing Lake is located, which is the United States. However, the provided memory only gives the population of Brown County, Kansas, which is 9,984 as of the 2010 census. It does not provide the population of the United States. Therefore, the answer is that the population of the United States cannot be determined from the provided information.

Analysis: The term “country” in the question is ambiguous, as it can refer to either a “nation” or a “rural area/region”. Observing the large model’s reasoning process, the task typically involves information about a “nation”, yet there is no associated data regarding the population of that “nation”, making it impossible to provide a correct response. The query explicitly asks for the population of the country (i.e., the United States). However, the provided context only contains the population of Brown County (9,984). The system is forced to either hallucinate the U.S. population using internal parametric knowledge or incorrectly answer with the county population. This highlights a flaw in the dataset generation process where essential evidence (U.S. population data) was excluded from the context window.

## E.2 The Need for Sub-EM (Substring Exact Match)

We observed that strict Exact Match (EM) often fails when the model generates correct but verbose answers, valid aliases, or redundant confirmations. This necessitates the use of more robust metrics such as Sub-EM.

## Case 6: Name Variation

Query: What British singer-songwriter hosted the 16th Annual Young Hollywood Awards?

## Context:

• Document 48:Kelly Osbourne, Kelly Lee Osbourne (born 27 October 1984) is a British singer-songwriter, actress, television presenter and fashion designer. The daughter of Ozzy and Sharon Osbourne, she is known for her appearances on "The Osbournes" with her family, for which they won a 2002 Emmy Award for Outstanding Reality Program, as well as on E!’s "Fashion Police", where she was a presenter from 2010 to 2015. She has also appeared on "Dancing with the Stars", in which she and her professional dance partner Louis van Amstel took third place. She is the voice of Hildy Gloom in the Disney XD animated series "The 7D". She is also a judge on both "Australia’s Got Talent" and "Project Runway Junior"

Ground Truth: Kelly Lee Osbourne Model Prediction: Kelly Osbourne Evaluation: EM = 0, Sub-EM = 1. The model is correct but penalized by EM for omitting the middle name.

## Case 7: Verbose Generation

Query: Who was known by his stage name Aladin and helped organizations improve their performance as a consultant? Ground Truth: Eenasul Fateh LLM answer: Eenasul Fateh, known by his stage name Aladin, who has a background as a former international management consultant and has helped organizations improve their performance. Scores: { F1: 0.16, EM=0, Sub-EM=1.0 }. The answer contains the ground truth but includes extra context copied from the document.

## Case 8: Redundant Confirmation

input: Are Local H and For Against both   
from the United States?   
Ground Truth: yes

LLM answer: yes, both Local H and For Against are from the United States. eval: {‘F1’: 0.0, ‘EM’: 0.0, ‘Sub-EM’: 1.0} The model provides a complete sentence instead of a single word.

## E.3 The Need for LLM-as-a-Judge

In cases of acronyms or semantic rephrasing, even Sub-EM fails. $\mathsf { A C C } _ { L }$ is essential to capture these equivalences.

## Case 9: Corporate Entity Naming

Query: Maxeda has been owned since 2004 by which American multinational equity firm?

## Context:

• Document 18:Kohlberg Kravis Roberts, KKR & Co. L.P. (formerly known as Kohlberg Kravis Roberts & Co.) is an American multinational private equity firm, specializing in leveraged buyouts, headquartered in New York City. The firm sponsors and manages private equity investment funds. The firm has completed over \$400 billion of private equity transactions since its inception.

Ground Truth: KKR & Co Model Prediction: Kohlberg Kravis Roberts Evaluation: EM = 0, Sub-EM = 0, $\mathbf { A C C } _ { L } { = } 1 . \mathbf { 0 } .$ KKR is the abbreviation for Kohlberg Kravis Roberts; both are correct. The model provided the full name instead of the abbreviation in the label.

## Case 10: Acronym Matching

Query: What station broadcast the episode “Marry Me a Little, Marry Me a Little More”, of the series created by Max Mutchnick and David Kohan?

Ground Truth: National Broadcasting Company

Model Prediction: NBC

Evaluation: F1 = 0, EM = 0, Sub-EM = 0. However, NBC is the acronym for the ground truth. $\mathsf { A C C } _ { L }$ correctly assigns a score of 1.0.

## Case 11: Semantic Rephrasing

Query: What was the 58th quadrennial American presidential election held after the 2016 Michigan Democratic primary? Ground Truth: United States presidential election of 2016

LLM answer: the 2016 U.S presidential election

Evaluation: {F1: 0.6, EM: 0.0, Sub-EM: $0 . 0 , \mathbf { A C C } _ { L } { = } \mathbf { 1 . 0 } \ \}$ . The phrasing differs, but the semantic meaning is identical.

## F Prompt Templates

We provide the prompt templates used in each stage of the ConvMem framework.

## F.1 Query Decomposition

The following prompt is used to decompose the complex user query into independent sub-questions (channels).

Please complete the extraction of keywords from the question.

## F.2 Semantic Kernel Operation

These prompts define the behavior of the convolutional kernel, including the kernel prompt, skip connection and summarization.

Skip Connection Prompt   
Multi-Kernel Convolution Prompt   
You will see a question and a paragraph   
You will encounter a complex reasoning that may contain information related to   
problem requiring you to extract keywords the question. Please read the paragraph   
or key phrases from the question that aid in carefully and make the relevant judgment.   
solving it.   
<Question> <Question>   
{question} {question}   
</Question> </Question>   
<Output Format> <Paragraph>   
{{ {paragraph}   
"subproblem": ["Subproblem 1", "Subprob </Paragraph>   
lem 2", ....],   
"keyword": ["Keyword1", "Keyword2",...]   
<Output Format>   
}} {{"0": "Completely unrelated to the   
</Output Format>   
question"}} or {{"1": "Potentially related   
to the question"}} or {{"2": "Definitely   
<Example> contains the answer to the question"}}   
Original Question: “Where was the director   
</Output Format>   
of the movie Inception born?”   
Sub-questions: [“Who is the director of <Notes>   
‘Inception’?”, “What information is there 1. To ensure no useful information is   
about the director’s birthplace?”] omitted, only select "0": "Completely   
Original Question: “What is the height of   
unrelated to the question" if you are   
the male lead in The Revenant?”   
absolutely certain the paragraph content is   
Keywords: [“The Revenant protagonist”, irrelevant to the question.   
“Human height”] 2. Do not output any redundant information   
<Example> beyond the <Output Format>.   
</Notes>   
<Note>   
1. The decomposed subproblems must fully Your Answer:   
resolve the original problem.   
2. The number of subproblems should be   
Kernel Summarization Prompt   
the minimum decomposition required to   
solve the original problem. You will be presented with a question and   
3. Keywords must be highly relevant to a passage that may contain information   
solving the original query, not irrelevant relevant to answering it. Read the passage   
terms, and must not create ambiguity with carefully and update your memory with   
the original query. new information that helps solve the   
</Note> problem. Be sure to retain all relevant   
details in your memory that may aid in

solving the problem.

<Question>

{question}

</Question>

<passage>

{passage}

</passage>

<format> Updated memory:

## <notes>

1. Retain all details within the paragraph that are relevant to the question, ensuring the integrity of the original content.

2. If none of the paragraph’s information relates to the question, output “Updated Memory: No relevant information.”

</notes>

Updated Memory:

## F.3 Hierarchical Aggregation

These prompts are used for recursive summarization in hidden layers and the final answer generation.

## Hidden Layer Aggregation Prompt

You will see a question and {num} memory fragments. Carefully review the provided memory fragments and combine information from each fragment that helps solve the problem.

<Question> {question} </Question>

<Memory Fragment>

{memory\_content}

</Memory Fragment>

<format>

Updated memory:

</format>

## <notes>

1. Retain all details within the paragraph that are relevant to the question, ensuring the integrity of the original content.

2. Avoid redundancy; if a fact already exists in memory, it need not be repeated unless that section provides additional clarification or correction.

</notes>

Updated Memory:

## Sub-Question Answer Prompt

You will see a question and its preceding memory. Answer the question based on the preceding memory. Your response must include all details relevant to the question; omitting any information is prohibited. Subsequent questions will require you to answer based on these details. Note: If the memory does not contain information relevant to the question, simply answer “None.”

<question>

{question}

</question>

<memory>

{memory}

</memory>

Your answer:

## Final Answer Generation Prompt

You will see a question along with its associated dependency information. Please read the question carefully, identify the core objective to be addressed, and answer based on the dependency information. Organize your response using the following format:

“"Hence, the answer is (insert answer here).”

<Question>

{question}

</Question>

<Related Dependency Information> {related\_dependency\_information} </Related Dependency Information>

## <notes>

1. Carefully read the <Question> to precisely identify the target objective to be a ddressed, the focus of your response and avoid being misled by lengthy questions.

2. Carefully read <Relevant Dependency Information> to capture details relevant to the question, avoiding distraction from extraneous information.

3. Whenever possible, derive answers directly from <Related Dependency Information> without unnecessary rephrasing. 4. Avoid redundant explanations or unnecessary elaboration on the question’s answer. For example: If the question asks for place names/person names/other information, etc.you only need to provide the place names/person names/other information, etc. themselves. No additional background information is required.

</notes>

Your Answer:

## F.4 Evaluation

The prompt used for LLM-as-a-Judge evaluation.

## LLM-as-a-Judge Prompt

You must evaluate the quality of the AI assistant’s responses to user questions as an impartial judge. Scores should comprehensively consider accuracy (high priority) and completeness (whether the response covers all key points).Please rate on a scale of 0 to 10, where 0 indicates the response is completely incorrect and 10 indicates the response is completely correct.

<Submit your feedback in the following format>

{{"Overall Score": "(Your rating, a floatingpoint number between 0 and 10)"}} </Submit your feedback in the following format>

<Below are the question and answers>

Question: {question}

Ground Truth: {ground\_truth}

User Response: {answer}

</Below are the question and answers>

Please complete the rating.