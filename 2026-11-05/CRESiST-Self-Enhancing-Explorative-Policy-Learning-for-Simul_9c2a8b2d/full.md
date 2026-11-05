# CRESiST: Self-Enhancing Explorative Policy Learning for Simultaneous Speech Translation

Chenyang Le<sup>1∗</sup> Pei Zhang<sup>2,3</sup> Yuxiang Chen<sup>4</sup> Bing Han<sup>1,2</sup> Derek F. Wong<sup>3</sup> Baosong Yang<sup>2</sup> Yanmin Qian<sup>1†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Alibaba Token Hub, Alibaba Group <sup>3</sup>University of Macau <sup>4</sup>University College London nethermanpro@sjtu.edu.cn yanmin.qian@sjtu.edu.cn

## Abstract

Simultaneous Speech Translation (SimulST) requires balancing translation quality and latency. While Large Language Models (LLMs) excel in offline translation, current LLM-based SimulST systems predominantly rely on prefix-incremental generation. This paradigm suffers from redundant Key-Value cache recomputation and necessitates external Voice Activity Detection to process unbounded streams. To overcome these bottlenecks, we propose an end-to-end interleaved read-write framework built upon an omni-modal LLM. By interleaving speech chunks and text tokens, our architecture achieves full KV cache reuse, natively supporting segmentation-free, infinite-length inference. Furthermore, to learn optimal streaming policies without rigid offline alignments, we introduce a novel Actor-Critic self-enhancement pipeline. An RNN-T Actor dynamically samples read-write trajectories, while an LLM-based Critic evaluates these paths via intrinsic semantic and latency rewards, distilling the optimal policy back to the Actor. Extensive experiments on sentence-level (FLEURS, CoVoST2) and document-level (MuST-C, ACL 60/60) benchmarks demonstrate that our model establishes new state-of-theart quality-latency Pareto frontiers, substantially outperforming existing cascaded and static-alignment baselines.

## 1 Introduction

Simultaneous Speech Translation (SimulST) aims to incrementally translate a continuous audio stream while minimizing delivery latency. Although Large Language Models (LLMs) have revolutionized offline translation, deploying them for unbounded real-time speech reveals a fundamental architectural conflict. State-of-the-art SimulST approaches [1, 2, 3, 4, 5] predominantly rely on prefix-incremental generation [6], where newly arrived speech chunks are continuously appended to the historical prefix. This invalidates the Key-Value (KV) cache across incremental steps, causing a quadratic computational explosion and forcing systems to rely on external Segmentation, such as Voice Activity Detection (VAD). Such forced segmentation disrupts cross-sentence context, impairs coreference resolution, and fails to natively support infinite-length streams.

To bypass the KV cache bottleneck, recent studies explored interleaved read-write architectures [7, 8] that append acoustic and textual tokens into a unified sequence. However, these models heavily depend on static, heuristic-driven word alignments (e.g., MFA[9]) to construct supervised trajectories, failing to capture the dynamic, predictive nature of human interpreters. While some works adopt reinforcement learning [10, 11, 12, 13], they still require offline aligned data and struggle to scale multilingually. To address this, we propose CRESiST (Critic-Refined Exploratory Simultaneous Speech Translation), an interleaved framework that discovers its streaming policy without any external alignment supervision. Built on the Qwen2.5-omni-7B backbone [14], CRESiST discards prefix-incremental generation and segmentation, achieving full KV cache reuse. Crucially, it operates at a granular 400ms micro-chunk size, enabling a highly fine-grained streaming policy.

To optimize this fine-grained policy without external alignments, we formulate the streaming decision process as a Markov Decision Process solved via a self-enhancing Actor-Critic pipeline. An auxiliary RNN-T acts as the Actor, actively exploring read-write trajectories. Simultaneously, the Speech LLM, equipped with an MoE Router and Refiner, serves as the Critic. The Critic evaluates trajectories using dynamic consistency and latency rewards, transferring its reward-maximizing preferences back to the Actor via reverse distillation. This iterative self-play allows CRESiST to autonomously discover adaptive, language-specific translation strategies (e.g., anticipating German verbs). Consequently, a single unified CRESiST model seamlessly translates Chinese and English streams into six diverse target languages.

Our main contributions are summarized as follows:

• Unsupervised Streaming Policy in an Interleaved Framework: While trained purely on offline parallel data, our framework autonomously transitions to an online interleaved paradigm, enabling full KV cache reuse for segmentation-free, infinite-length translation.

• Fine-Grained Actor-Critic Policy Discovery: We introduce a novel self-enhancement pipeline operating at a 400ms acoustic chunk size. Through reward-driven reverse distillation, the model autonomously learns optimal streaming policies without static forced alignments.

• Multilingual Generalization: A single unified CRESiST model supports Chinese/English inputs to six target languages, overcoming the limitations of language-pair-specific models.

• State-of-the-Art Performance: Extensive experiments across sentence-level and unbounded document-level benchmarks demonstrate that CRESiST outperforms strong baselines (e.g., InfiniSST), establishing new Quality-Latency Pareto frontiers.

## 2 Related Work

Read-Write Policy Learning Determining the optimal timing for translation is a core challenge in SimulST. A prevailing paradigm relies on static alignments from external tools (e.g., MFA) or probability mutations to construct supervised read-write trajectories [7, 15, 16, 8, 17, 18]. To circumvent the rigidity of forced alignments, fixed and heuristic approaches utilize continuous integrate-and-fire (CIF)[19], wait-k cascades, or attention boundary indicators [20, 21, 22, 23]. Recent unsupervised methods further explore dynamic boundary detection by leveraging offline-to-online inference divergence [4, 5], dual CIF distillation [24], or implicit learning via mixture-of-experts [3]. Furthermore, to optimize the latency-quality trade-off beyond static supervision, advanced reinforcement learning techniques (such as PPO and GRPO) have been increasingly adopted to autonomously refine streaming translation strategies [11, 13, 12].

Unbounded Audio Processing Processing continuous, infinite-length audio streams typically suffers from severe memory bottlenecks. Traditional systems heavily depend on external segmenters like VAD or acoustic heuristics [25], which inevitably disrupt cross-sentence contextual coherence. To mitigate this, recent studies have proposed history selection mechanisms that only retain fixed amounts of text tokens and their attention-covered acoustic frames [26]. Alternatively, formulating the continuous stream as multi-turn dialogues allows for the safe discarding of distant KV cache [7]; however, such formulations typically mandate tightly coupled supervised policies to function effectively.

## 3 CRESiST

## 3.1 Problem Formalization

Unlike previous RL approaches in SimulST that couple translation timing with content generation—creating an intractable policy space—we decouple timing by formalizing the read-write strategy as a Markov Decision Process (MDP). Given an offline speech-text pair {S, T} with acoustic chunks $\mathbf { S } = \left( s _ { 1 } , \ldots , s _ { | \mathbf { S } | } \right)$ and ground-truth text tokens $\mathbf { T } = \left( t _ { 1 } , \ldots , t _ { | \mathbf { T } | } \right)$ , the MDP is defined by the quadruple $\langle s , \mathcal { A } , \dot { \mathcal { P } } , \mathcal { R } \rangle$ . Specifically, the state space (S) at step k is $s _ { k } = ( \mathbf { S } _ { \leq u _ { k } } , \mathbf { T } _ { < v _ { k } } )$ representing the $u _ { k }$ speech chunks read and $v _ { k }$ target tokens written thus far. The action space (A) is a binary space {Read, Write} controlling the translation rhythm. The transition dynamics (P) dictates the environment evolution $\textstyle P ( s _ { k + 1 } \mid s _ { k } , a _ { k } )$ : a Read action appends the next speech chunk yielding $s _ { k + 1 } = ( \mathbf S _ { \le u _ { k } + 1 } , \mathbf T _ { < v _ { k } } )$ , whereas a Write action appends a text token to yield $s _ { k + 1 } = ( \mathbf { S } _ { \leq u _ { k } } , \mathbf { T } _ { < v _ { k } + 1 } ) .$ . Optimal write actions maximize congruence between Data Dynamics (ground-truth distribution $P _ { d a t a } ( t _ { v _ { k } } \mid s _ { k } ) )$ ) and Model Dynamics (LLM prediction $P _ { m o d e l } ( \cdot \mid s _ { k } ; \theta ) )$ . Finally, the reward function (R) provides a scalar $r _ { k } ( s _ { k } , a _ { k } )$ to maximize expected cumulative rewards, delicately balancing a dynamic consistency evaluation (aligning the Model and Data dynamics) against a latency penalty to ensure timely responses.

![](images/71761a8bbec067929e28f368b6dae5916873f42bb76c525d033cc2b84635840c.jpg)  
Figure 1: Training and inference diagram of CRESiST. In the interleaved sequence, red and green target tokens mark whether the Router score at that position passes the decision threshold, i.e., whether the LLM body loss $\mathcal { L } _ { b o d y }$ is computed there (tokens judged to be emitted too early are masked out). The Refiner loss $\mathcal { L } _ { r e f i n e r }$ is computed at every text position, and the two losses are summed; the same ground-truth token is thus the target of two prediction heads over one sequence, not duplicated in it.

## 3.2 Actor-Critic Training Framework

To solve the MDP formulated in the previous section and transition toward an autonomous read-write interleaved paradigm, we propose a self-enhancing Actor-Critic training framework, outlined in Algorithm 1. Given an offline speech-text corpus $\bar { \mathcal { D } } = \{ ( \mathbf { S } , \mathbf { T } ) \}$ , our framework jointly optimizes translation fidelity and the simultaneous streaming policy. By decoupling the generation process into an exploration-evaluation loop, we effectively compute the MDP transition dynamics and the reward function R without relying on heuristic alignments.

The Actor for Trajectory Exploration. We introduce an auxiliary RNN-T[27] model parameterized by $\theta _ { a c t }$ to serve as the Actor. Operating within the state space ${ \dot { \boldsymbol { s } } } ,$ the Actor explores the action space A by sampling a sequence of binary Read/Write decisions. For a given sample $( \mathbf { S } , \mathbf { T } ) \sim \mathcal { D } .$ , it yields a trajectory of actions $\mathbf { a } \sim \pi _ { a c t } ( \cdot \mid \mathbf { S } , \mathbf { T } ; \theta _ { a c t } )$ , where $a _ { k } \in \mathcal { V } \cup \{ \phi \}$ (with ϕ denoting a “Read” action). This sampled trajectory simulates the transition dynamics $\mathcal { P } _ { \cdot }$ , deterministically merging the source speech and target text into a unified, interleaved sequence z = Interleave(S, T, a). Through this exploration, the Actor constructs the streaming-style state transitions required by the downstream environment.

The Critic for Reward Evaluation and Validation. The Qwen-omni LLM backbone, equipped with a Router-Refiner module and parameterized by $\theta _ { c r i }$ , acts as the Critic. It ingests the state sequence z to evaluate the Actor’s trajectory by computing the Reward Function (R). The evaluation is twofold:

1) Dynamic Consistency Reward $\scriptstyle ( { \mathcal { R } } _ { c o n s i s t e n c y } ) :$ The Refiner and LLM body assess the congruence between Data Dynamics and Model Dynamics. By conditioning on local hidden states $\boldsymbol { \mathbf { H } } = \boldsymbol { \mathbf { \check { L } } } \boldsymbol { \mathbf { L } } \boldsymbol { \mathbf { M } } ( \boldsymbol { \mathbf { z } } ; \boldsymbol { \theta } _ { c r i } )$ and a globally compressed acoustic representation $\mathbf { h } _ { g l o b a l }$ , it computes the token generation probabilities. We formally define the consistency reward as the negative log-likelihood of the generation targets:

$$
\mathcal { R } _ { c o n s i s t e n c y } = - ( \mathcal { L } _ { r e f i n e r } + \mathcal { L } _ { b o d y } )\tag{1}
$$

where $\mathcal { L } _ { r e f i n e r }$ and $\mathcal { L } _ { b o d y }$ are the cross-entropy generation losses of the Refiner and LLM body, respectively. High predictive accuracy yields a high consistency reward.

2) Latency Reward $( \mathcal { R } _ { l a t e n c y } ) { : }$ To ensure timely responses without relying on heuristic constraints, the Router module assesses the sequence of hidden states to predict the action probability $p _ { k } = \mathrm { R o u t e r } ( \mathbf { H } _ { k - 1 } ; \theta _ { c r i } )$ at each step, where $p _ { k } \to 1$ signifies a Read and $p _ { k } \to 0$ signifies a Write. We mathematically formalize the latency reward as the expected length of consecutive text generation in a given turn. Since the probability of executing at least m consecutive writes starting from step k is given by the cumulative product of write probabilities, the expected write length $\mathbf { \mathbb { E } } [ W _ { k } ]$ can be elegantly formulated as:

$$
\mathcal { R } _ { l a t e n c y } ^ { ( k ) } = \mathbb { E } [ W _ { k } ] = \sum _ { m = 1 } ^ { M } \prod _ { j = 1 } ^ { m } \left( 1 - p _ { k + j } \right)\tag{2}
$$

where M denotes the maximum allowable length for a single conversation turn. By maximizing this expectation-based reward, the Critic intrinsically incentivizes the policy to emit longer textual chunks as early as possible, thereby minimizing the overall translation delay.

Essentially, the Critic judges whether the state-action path traversed by the Actor maximizes the overall reward $\mathcal { R } = \mathcal { R } _ { c o n s i s t e n c y } + \gamma \mathcal { R } _ { l a t e n c y : }$ , facilitating fluent generation with minimal delay.

Self-Enhancement via Reward-Driven Distillation. The entire framework is optimized through an alternating closed-loop mechanism. First, the Critic is updated by minimizing the composite cost $\mathcal { I } _ { c r i t i c } .$ , which integrates the negative rewards alongside an auxiliary load balancing loss $( \mathcal { L } _ { b a l a n c e } )$ to stabilize the routing distribution:

$$
\mathcal { T } _ { c r i t i c } = - \mathcal { R } _ { c o n s i s t e n c y } - \alpha \mathcal { R } _ { l a t e n c y } + \beta \mathcal { L } _ { b a l a n c e }\tag{3}
$$

where α and $\beta$ scale the latency reward objective and the balancing penalty, respectively.

Subsequently, the Actor is updated by minimizing its own composite objective $\mathcal { I } _ { a c t o r }$ . To emphasize the reward-based policy feedback, we position a novel Reverse Distillation loss $( \mathcal { L } _ { d i s t i l l } )$ as the primary optimization target, treating the standard RNN-T loss $( \mathcal { L } _ { r n n t } )$ as an auxiliary constraint to maintain basic acoustic-linguistic alignment:

$$
\mathcal { T } _ { a c t o r } = \mathcal { L } _ { d i s t i l l } + \lambda \mathcal { L } _ { r n n t }\tag{4}
$$

where λ acts as a down-scaling coefficient for the auxiliary loss. To compute the critical $\mathcal { L } _ { d i s t i l l }$ , we use the reward-optimized Critic to binarize the Router’s validated policy decisions via a predefined threshold τ, constructing an optimal target action sequence $\hat { a } _ { k } = \mathbb { I } ( p _ { k } \ge \tau )$ . A Binary Cross-Entropy (BCE) distillation loss then forces the Actor to align its action probabilities with these LLM-preferred, reward-maximizing decisions:

$$
{ \mathcal { L } } _ { d i s t i l l } = \sum _ { k } \operatorname { B C E } \Bigl ( \hat { a } _ { k } , \pi _ { a c t } ( a _ { k } = \phi \mid \cdot ; \theta _ { a c t } ) \Bigr )\tag{5}
$$

By continuously backpropagating gradients $\begin{array} { r } { \nabla _ { \theta _ { a c t } } \mathcal { J } _ { a c t o r } , } \end{array}$ , the RNN-T learns to autonomously sample trajectories that are intrinsically aligned with the LLM’s optimal dynamics. We note that the pseudo label $\hat { a } _ { k }$ is treated with a stop-gradient on $\theta _ { c r i }$ , decoupling the Actor update from gradient flow through the Critic. We run $\bar { \mathsf { K } } = 8$ Critic steps followed by one Actor step for better convergence. Theoretical guarantees regarding the convergence of this joint optimization are provided in Appendix B.

## 3.3 Model Implementation

To instantiate the Actor-Critic framework, the CRESiST architecture is bifurcated into two structural components: an RNN-T-based Actor for alignment exploration and an LLM-based Critic for validation and generation.

![](images/a5439df46bb5f325ec44a69ce1bb7a70d53f6791b66b70c40e9eb31b895fe4cf.jpg)  
Figure 2: Structural details of the RNN-T aligner and the Refiner layer.

## 3.3.1 The Actor Model: Alignment Explorer

To avoid the computational inefficiency and "cold start" issues of rolling out interleaved sequences directly via the Speech LLM, the Actor is parameterized using an auxiliary Recurrent Neural Network Transducer (RNN-T).

As illustrated in Figure 2a, our Actor comprises an acoustic encoder, a text predictor, and a joint network, augmented with a 2D axial attention module. Each layer of this axial attention module consists of two causal self-attention mechanisms—operating sequentially along the text and speech axes—followed by an MLP, strengthening dual-axis modality interaction.

For the specific configuration, the acoustic encoder is initialized with the pre-trained Whisper-medium encoder[28], followed by two down-sampling M-Adapter layers to yield 2.5 Hz speech features (a 400ms window). A 400ms chunk-autoregressive mask is applied to strictly preserve causality. The text predictor and the axial attention module are initialized using the Qwen3-0.6B model, while the newly introduced audio-axis attention weights are initialized from scratch.

Read-Write Trajectory Construction. To construct the interleaved sequence z defined in Section 3.2, we traverse the $( T \times U )$ alignment lattice. At each node $( t , u )$ , the joint network outputs a blank (“Read”) emission probability $p _ { y } ^ { ( t , u ) }$ . The routing action $a _ { k }$ is then directly sampled via:

$$
a _ { k } \sim \mathrm { B e r n o u l l i } \left( p _ { y } ^ { ( t _ { k } , u _ { k } ) } \right)\tag{6}
$$

The sampled path ${ \bf a } = ( a _ { 1 } , \dots , a _ { K } )$ governs the lattice coordinate updates and dictates the exact merge boundaries for the LLM input space. Note that every sampled trajectory is a monotonic path on the lattice from (0, 0) to $( T , U ) { \mathrm { ; } }$ : a blank advances only along the audio axis and a token emission only along the text axis, and once the last audio chunk is reached all remaining target tokens are emitted. Every trajectory therefore contains exactly U Write actions, so no target token is ever dropped from z.

## 3.3.2 The Critic Model: Speech LLM and Validator

The Critic evaluates the Actor’s trajectories and generates the final translation. It is built upon a Qwen-omni Large Language Model backbone, structurally augmented with Router and Refiner modules.

Speech LLM Backbone. Adapted from the “thinker” component of the Qwen2.5-omni-7B, the speech encoder is modified for streaming by applying a windowed chunk autoregressive mask (400ms chunk size, 20-chunk local attention window, yielding an 8-second receptive field). Cyclic sinusoidal positional encodings are applied every 2 seconds.

The input is formatted using a multi-turn dialogue template. Task-level directives are embedded in the system prompt, and each audio chunk with its interleaved textual actions serves as a single conversational turn. The backbone processes this to yield local hidden states $\mathbf { H } = \mathrm { L L M } ( \mathbf { z } ; \theta _ { c r i } )$ and a globally compressed acoustic representation ${ \bf h } _ { g l o b a l } = \mathrm { A v e P o o l } ( \mathrm { L L M } ( \mathbf { S } ; \theta _ { c r i } ) )$ .

Router Module. Implemented as a lightweight 2-layer MLP with a Sigmoid activation, the Router projects the preceding hidden state $\mathbf { H } _ { k - 1 }$ to a scalar probability $p _ { k } \in [ 0 , 1 ]$ . While Section 3.2 details its role in the latency reward $( \mathcal { R } _ { l a t e n c y } )$ , structurally, $p _ { k }$ is thresholded by a predefined τ during inference to trigger either $\mathrm { ~ } _ { \mathrm { l } } \mathrm { ~ } ^ { \ast } \mathrm { R e a d } ^ { \mathrm { , ~ } } ( p _ { k } \geq \tau ) \mathrm { o r } ^ { \ast } \mathrm { W r i t e } ^ { \mathrm { , ~ } } ( p _ { k } < \tau )$ action.

Refiner Module. We adopt the Refiner architecture [3] for final generation. As shown in Figure 2b, the core structural integration involves repurposing the Router’s probability output p as a dynamic routing scalar. The Refiner directly scales the global acoustic gain by p and the local acoustic gain by $( 1 - p )$ . This self-adaptive gating synchronizes the generation context with the streaming timing state before the refined features are projected to compute the consistency reward $( \mathcal { R } _ { c o n s i s t e n c y } )$ During inference, global acoustic information is absent. The Router can make policies without global information, and we do not use the Refiner at inference.

## 4 Experiment and Results

## 4.1 Experimental Setup

Training Data To support multilingual translation, we aggregate three open-source ASR datasets: Emilia[29], WenetSpeech[30], and GigaSpeech[31]. We utilize the Qwen3-235B-A22B<sup>1</sup> model to generate pseudo-labels for the transcribed texts, translating them into Chinese, English, and the other 4 European languages (DE, FR, ES, IT). (Chinese and English account for more than 50% of the pseudo-labels). We augment WenetSpeech and GigaSpeech by concatenating consecutive clips into long segments of a maximum of 30 seconds. All training stages dynamically sample from this unified corpus for training and validation.

Training Details Our training pipeline is implemented using PyTorch Lightning. The Speech LLM is parameter-efficiently fine-tuned using LoRA[32] with a rank of r = 16. When training the LLM for SimulST task, we also train offline ST and MT tasks in a multi-task way to preserve offline performance.

Following our proposed Actor-Critic paradigm, the training strictly proceeds in three stages:

1) Actor Pre-training: The auxiliary RNN-T Aligner is trained with the standard RNN-T loss to establish basic acoustic-linguistic alignments. Only the alignment task itself is new at this stage: the acoustic encoder is initialized from Whisper-medium, and the text-axis attention and MLP of each axial layer inherit the corresponding Qwen3-0.6B layer (the bottom 12 layers are frozen and receive no audio-axis attention; the upper 16 layers are augmented), while the audio-axis attention and the joint network are randomly initialized.

2) Critic Pre-training: Freezing the RNN-T parameters, we use its sampled trajectories to construct interleaved sequences. The Speech LLM (including the Router and Refiner) is optimized to adapt to this streaming input format.

3) Joint Self-Enhancement: All modules are unfrozen. The framework undergoes joint training via reverse distillation (as detailed in Section 3.2), continuously refining the Actor’s sampling policy alongside the Critic’s generation capability.

Benchmarks. We evaluate our model across diverse streaming scenarios. For sentence-level evaluation, we utilize the FLEURS[33] and CoVoST 2[34] datasets across the 10 supported directions. Notably, on FLEURS, we decouple the evaluation into Chinese-English bidirectional translations (to ensure a direct comparison with Seed-LiveInterpret2) and the remaining 8 directions. For document level continuous streaming, we utilize the MuST-C and ACL6060 test sets (En-Zh, En-De). We use OmniSTEval for long-form evaluation.<sup>2</sup>

![](images/2be8a195b8a6c836a2643a2879254386155195b99796bd7ef7db2b6a5999db0d.jpg)  
Figure 3: Sentence-level quality-latency trade-off curves on the CoVoST2 and FLEURS datasets. The proposed framework consistently establishes a superior Pareto frontier compared to Seamless Streaming and AlignATT across all sub-tasks.

Baselines. We benchmark our framework against strong state-of-the-art systems, including opensource foundations (Seamless Streaming), commercial SOTA Seed-LiveInterpret2, and academic simultaneous models (AlignATT, StreamUni, InfiniSST). More details about the baseline implementation can be found in Appendix C.3.

Metrics. Simultaneous translation requires balancing translation fidelity and streaming latency. For Translation Quality, we utilize XCOMET-XXL<sup>3</sup>, a neural metric that aligns significantly better with human judgments compared to traditional n-gram metrics[35]. For Streaming Latency, we apply input-adaptive metrics to avoid heuristic bias: we report the Length-Adaptive Acoustic Latency (LAAL)[36] for sentence-level benchmarks, and adopt the recently proposed LongYAAL metric for document-level evaluations to accurately capture the cumulative delays in continuous long-form audio streams[37].

## 4.2 Main Results

Sentence-Level Quality-Latency Trade-offs. Figure 3 illustrates the quality-latency Pareto frontiers on the CoVoST2 and FLEURS benchmarks. By varying the decision threshold (e.g., $\tau \in [ 0 . 3 , 0 . 7 ] )$ , we plot the translation quality against the latency metric. Our proposed framework consistently establishes a new state-of-the-art frontier, dominating both Seamless Streaming and AlignATT across all latency regimes. Notably, on the FLEURS (zh ↔ en) benchmark, our streaming model strictly outperforms the commercial Seed LiveInterpret 2.0 API, achieving higher translation quality while maintaining a comparable or superior latency profile. Furthermore, the curve demonstrates that our model degrades gracefully at extremely low latency requirements (e.g., LAAL < 1.5), proving the effectiveness of the reverse-distilled streaming policy.

Document-Level Continuous Streaming. Evaluating on isolated sentences often obscures the compounding errors inherent in real-world, long-form translation. Table 1 presents our results on the MuST-C and ACL 60/60 benchmarks across three settings: Offline w/ Oracle Segmentation, Streaming w/ Oracle Segmentation, and the most rigorous Streaming w/o Segmentation. The contrast between the latter two settings effectively reveals a model’s true capacity to process continuous, unbounded audio streams.

Notably, while InfiniSST is explicitly designed for infinite audio streams, it exhibits a paradoxical performance drop when evaluated at the sentence level with oracle boundaries. In contrast, CRESiST delivers consistently high performance across both streaming scenarios. Strikingly, our unsegmented streaming performance closely trails the oracle segmentation with a marginal degradation of only 1–2 COMET points, demonstrating that CRESiST can flawlessly manage infinite context without relying on explicit segmentation.

Furthermore, we observe that offline incremental methods such as StreamATT serve as highly competitive baselines on certain language pairs. However, their performance is notably inconsistent; on other language pairs, historical context selection leads to a catastrophic drop in translation quality (e.g., on the ACL 60/60 en-de dataset, the COMET score plummets from 80.83 under oracle segmentation to 70.61) coupled with severe latency inflation. Ultimately, CRESiST successfully navigates infinite audio streams, avoiding the pitfalls of historical amnesia and alignment drift while maintaining state-of-the-art translation quality and minimal continuous delay. CRESiST is also complementary to RL-based policy optimization: GRPO post-training further improves it, whereas the same recipe cannot close the gap from a direct-SFT base (Appendix D.1).

Table 1: Document-level simultaneous translation performance on the MuST-C and ACL 60/60 benchmarks. The COMET is evaluated through XCOMET-XXL. Our proposed CRESiST model achieves state-of-the-art translation quality with minimal continuous delay, demonstrating exceptional robustness in the challenging unsegmented streaming scenario. Oracle Segmentation denotes sentencelevel segmentation taken from the native sentence timestamps of the test sets, i.e., inference is run independently on each human-segmented sentence.
<table><tr><td></td><td colspan="4">MuST-C</td><td colspan="4">ACL 60/60</td></tr><tr><td></td><td colspan="2">en-zh</td><td colspan="2">en-de</td><td colspan="2">en-zh</td><td colspan="2">en-de</td></tr><tr><td>Method</td><td></td><td>COMET LongYAAL</td><td></td><td></td><td></td><td>COMET LongYAAL COMET LongYAAL</td><td></td><td>COMET LongYAAL</td></tr><tr><td colspan="9">Offline w/ Oracle Segmentation</td></tr><tr><td>Qwen2.5-Omni</td><td>79.41</td><td>1</td><td>91.47</td><td>1</td><td>84.42</td><td>1</td><td>91.72</td><td>1</td></tr><tr><td>CRESiST</td><td>79.11</td><td>1</td><td>91.35</td><td>1</td><td>83.36</td><td>1</td><td>90.03</td><td>1</td></tr><tr><td colspan="9">Streaming w/ Oracle Segmentation</td></tr><tr><td>InfiniSST (m=3)</td><td>68.46</td><td>2.18</td><td>79.22</td><td>1.93</td><td>59.71</td><td>2.17</td><td>65.09</td><td>2.17</td></tr><tr><td>AlignATT (f=30)</td><td>73.45</td><td>1.39</td><td>84.10</td><td>1.10</td><td>76.94</td><td>1.74</td><td>80.83</td><td>1.57</td></tr><tr><td>CRESiST (τ=0.5)</td><td>75.30</td><td>1.29</td><td>87.75</td><td>0.80</td><td>80.06</td><td>1.49</td><td>87.23</td><td>1.00</td></tr><tr><td colspan="9">Streaming w/o Segmentation</td></tr><tr><td>StreamUni (k=3)</td><td>52.96</td><td>2.91</td><td>73.45</td><td>2.01</td><td>41.90</td><td>1.64</td><td>71.03</td><td>1.64</td></tr><tr><td>InfiniSST (m=3)</td><td>70.66</td><td>1.88</td><td>78.74</td><td>1.46</td><td>68.62</td><td>1.80</td><td>63.52</td><td>1.46</td></tr><tr><td>StreamATT (f=30)</td><td>70.23</td><td>2.05</td><td>74.35</td><td>2.42</td><td>76.63</td><td>2.15</td><td>70.61</td><td>1.89</td></tr><tr><td>CRESiST (τ=0.5)</td><td>73.73</td><td>1.14</td><td>85.96</td><td>0.76</td><td>79.45</td><td>1.49</td><td>85.59</td><td>1.08</td></tr></table>

## 5 Ablation Studies

## 5.1 Effect of Training Strategies

We first investigate the impact of our self-enhancing Actor-Critic training paradigm. Figure 4 presents the quality-latency trade-offs under different training strategies. For fairness, the training steps are kept identical. The baseline (“Stage 1 & Direct SFT”), which directly utilizes RNN-T aligned data for Supervised Fine-Tuning akin to the approach in InfiniSST, yields the poorest performance. It struggles to run at a granularity of 400 milliseconds (latency multiplier=1). The pipeline approach (“Stage 1 & 2”), which trains the modules sequentially without the final joint optimization, shows noticeable improvements over the direct SFT.

In contrast, our full CRESiST framework with joint self-enhancement achieves the best overall Pareto frontier. The improvement is particularly pronounced on the Chinese-English (zh ↔ en) pairs, where the overall latency is significantly reduced while maintaining high translation quality.

## 5.2 Sensitivity to the Latency Penalty Weight

Next, as a sensitivity analysis rather than an ablation, we examine the impact of the latency penalty hyperparameter α on the system, with all variants trained for an identical number of steps. As illustrated in Figure 5, α explicitly governs the trade-off between translation quality and streaming latency. As expected, when α is large, the model aggressively penalizes delays, resulting in a compressed, lower overall latency profile, but at the cost of relatively degraded translation quality. Conversely, an excessively small α fails to adequately constrain the streaming policy, leading to disproportionately high latency spikes at specific decision thresholds.

## 5.3 Actor Architecture: Axial Blocks vs. Conventional RNN-T

Finally, we evaluate the architectural design of our auxiliary RNN-T Actor by comparing our proposed model (incorporating 2D axial attention blocks) against a conventional RNN-T architecture. We assess the alignment quality on the validation set using two metrics: the perplexity (PPL, base e) of the emitted target text and the Average Lagging (AL, measured in seconds) of the generated interleaved sequence. [38]

![](images/d451e4c3680b6881ce9a41b69d3a786c41f7fc203865b289dadc30acdc292d45.jpg)  
Figure 4: Ablation on training strategies. “Direct SFT” uses RNN-T aligned data directly (similar to InfiniSST). “Stage 1 & 2” represents pipeline training without the final joint optimization. “CRESiST” indicates our full joint training framework, which yields the most significant latency reduction, particularly on resource-rich directions (e.g., zh ↔ en).

As detailed in Table 2, both the two checkpoints are trained for 600k steps. The conventional RNN-T struggles significantly to produce optimal read-write paths, exhibiting abnormally high latency (AL = 8.03s) and poor text generation confidence (PPL = 22.3). The inclusion of axial blocks effectively strengthens the interaction between audio and text modalities, drastically improving alignment behavior. This design successfully drops the PPL to 6.03 and reduces the average lagging to 1.81s, providing a much higher-quality interleaved sequence for the downstream Speech LLM. We further verify in Appendix D.2 that the Actor is effective as a policy explorer but not as the final translator: reverse distillation transfers its latency behavior, while its translation quality stays far below the Speech LLM.

![](images/0b748caf939b254377491fb25242bb37132ef672dd16f53131f7c87ff538fb0b.jpg)  
Figure 5: Impact of the latency penalty weight on the quality-latency trade-off. Larger weights aggressively compress latency at the expense of quality, while excessively small weights lead to prolonged delays at specific thresholds.

Table 2: Comparison of the Actor architecture using a conventional RNN-T versus our proposed model with axial blocks. Evaluated on the validation set by text perplexity (PPL) and Average Lagging (AL, in seconds). $N _ { p d }$ and $N _ { a x }$ are the number of layers in the predictor and the Axial attention module respectively
<table><tr><td>Model</td><td> $N _ { p d }$ </td><td> $N _ { a x }$ </td><td>PPL</td><td>AL</td></tr><tr><td>Conventional</td><td>28</td><td>0</td><td>22.3</td><td>8.03</td></tr><tr><td>Proposed</td><td>12</td><td>16</td><td>6.03</td><td>1.81</td></tr></table>

## 6 Conclusion and Limitations

Conclusion. In this work, we introduced a novel self-enhancing Actor-Critic framework designed to transition simultaneous speech translation from rigid, incremental processing to a dynamic, readwrite interleaved paradigm. By decoupling policy exploration and reward evaluation, our method empowers an auxiliary RNN-T Actor to construct optimal streaming trajectories, while the Speech LLM Critic leverages its Router-Refiner module to ensure high-fidelity generation. Through a closed loop reverse distillation mechanism, the framework autonomously aligns the streaming policy with the LLM’s generation rhythm. Extensive experiments demonstrate that our approach establishes new state-of-the-art quality-latency trade-offs across both sentence-level and challenging document-level benchmarks.

Limitations and Future Work. Despite the promising results, our framework exhibits a few limitations that warrant future investigation:

• Data and Domain Bottlenecks: As this work primarily focuses on streaming policy optimization, we didn’t perform rigorous data engineering. The absolute translation quality is fundamentally bottlenecked by the backbone capacity and offline SFT data, which sub-optimally affect overall performance. For example, we observed an offline performance drop on the ACL60/60 dev set compared to the vanilla Qwen2.5-omni model, indicating domain distribution discrepancies and catastrophic forgetting in specialized academic domains.

• Document-level Latency Control: While adjusting the decision threshold (τ) effectively modulates the sentence-level quality-latency trade-off, it yields marginal tuning effects during documentlevel continuous streaming. Regulating the compounding dynamics of infinite audio streams likely requires dynamic penalty mechanisms, such as the context-aware latency multiplier utilized in InfiniSST, rather than relying solely on a static probability threshold.

## Acknowledgments and Disclosure of Funding

This work was supported in part by China NSFC project under Grants No. 62631016, No. 62601524 and No. U25A20409; in part by SJTU Med-X (Medicine & Engineering) Translational Research Grant (YG2025LC09); in part by the UM and UMDF (Grant Nos. MYRG-GRG2024-00165-FST-UMDF, MYRG-GRG2025-00236-FST, MYRG-GRG2026-00018-FIC, MYRG-CRG2026-00007- FIC); and in part by the Alibaba Innovative Research Program.

## References

[1] X. Ma, A. Sun, S. Ouyang, H. Inaguma, and P. Tomasello, “Efficient Monotonic Multihead Attention,” arXiv preprint arXiv:2312.04515, 2023.

[2] Seamless Communication, L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, M. Duppenthaler, P.-A. Duquenne, B. Ellis, H. Elsahar, J. Haaheim, J. Hoffman, M.-J. Hwang, H. Inaguma, C. Klaiber, I. Kulikov, P. Li, D. Licht, J. Maillard, R. Mavlyutov, A. Rakotoarison, K. R. Sadagopan, A. Ramakrishnan, T. Tran, G. Wenzek, Y. Yang, E. Ye, I. Evtimov, P. Fernandez, C. Gao, P. Hansanti, E. Kalbassi, A. Kallet, A. Kozhevnikov, G. M. Gonzalez, R. S. Roman, C. Touret, C. Wong, C. Wood, B. Yu, P. Andrews, C. Balioglu, P.-J. Chen, M. R. Costa-jussà, M. Elbayad, H. Gong, F. Guzmán, K. Heffernan, S. Jain, J. Kao, A. Lee, X. Ma, A. Mourachko, B. Peloquin, J. Pino, S. Popuri, C. Ropers, S. Saleem, H. Schwenk, A. Sun, P. Tomasello, C. Wang, J. Wang, S. Wang, and M. Williamson, “Seamless: Multilingual Expressive and Streaming Speech Translation,” 2023.

[3] C. Le, B. Han, J. Li, C. Songyong, and Y. Qian, “Simulmega: Moe routers are advanced policy makers for simultaneous speech translation,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025.

[4] X. Chen, K. Fan, W. Luo, L. Zhang, L. Zhao, X. Liu, and Z. Huang, “Divergence-Guided Simultaneous Speech Translation,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, pp. 17 799–17 807, 2024.

[5] N. Hirschkind, J. Liu, X. Yu, and M. K. Nandwana, “Reina: Regularized entropy information-based loss for efficient simultaneous speech translation,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 40, no. 37, 2026, pp. 31 005–31 013.

[6] D. Liu, G. Spanakis, and J. Niehues, “Low-latency sequence-to-sequence speech recognition and translation by partial hypothesis selection,” arXiv preprint arXiv:2005.11185, 2020.

[7] S. Ouyang, X. Xu, and L. Li, “Infinisst: Simultaneous translation of unbounded speech with large language model,” in Findings ofthe Associationfor Computational Linguistics: ACL 2025, 2025, pp. 3032–3046.

[8] S. Cheng, Z. Huang, T. Ko, H. Li, N. Peng, L. Xu, and Q. Zhang, “Towards achieving human parity on end-to-end simultaneous speech translation via llm agent,” arXiv preprint arXiv:2407.21646, 2024.

[9] M. McAuliffe, M. Socolof, S. Mihuc, M. Wagner, and M. Sonderegger, “Montreal forced aligner: Trainable text-speech alignment using kaldi,” in Proceedings of the 18th Annual Conference of the International Speech Communication Association (Interspeech 2017), 2017, pp. 498–502.

[10] T. Xu, Z. Huang, J. Sun, S. Cheng, and W. Lam, “Seqpo-simt: Sequential policy optimization for simultaneous machine translation,” in Findings of the Association for Computational Linguistics: ACL 2025, 2025, pp. 16 107–16 123.

[11] S. Cheng, Y. Bao, Z. Huang, Y. Lu, N. Peng, L. Xu, R. Yu, R. Cao, Y. Du, T. Han, Y. Hu, Z. Li, S. Liu, S. Ma, S. Pan, J. Xiao, N. Xu, M. Yang, R. Ye, Y. Yu, J. Zhang, R. Zhang, W. Zhang, W. Zhu, L. Zou, L. Lu, Y. Wang, and Y. Wu, “Seed LiveInterpret 2.0: End-to-end simultaneous speech-to-speech translation with your voice,” arXiv preprint arXiv:2507.17527, 2025, Seed-LiveInterpret 2.0 Technical Report, ByteDance Seed.

[12] T. Labiausse, R. Fabre, Y. Estève, A. Défossez, and N. Zeghidour, “Simultaneous speech-to-speech translation without aligned data,” arXiv preprint arXiv:2602.11072, 2026.

[13] S. Ouyang, S. Ding, O. Hrinchuk, V. Lavrukhin, B. Yan, B. Ginsburg, and L. Li, “Hierarchical policy optimization for simultaneous translation of unbounded speech,” arXiv preprint arXiv:2604.21045, 2026.

[14] J. Xu, Z. Guo, J. He, H. Hu, T. He, S. Bai, K. Chen, J. Wang, Y. Fan, K. Dang, B. Zhang, X. Wang, Y. Chu, and J. Lin, “Qwen2.5-omni technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2503.20215

[15] P. Zhang, Y. Wang, J. Tang, B. Yang, R. Wang, D. F. Wong, and F. Huang, “Direct simultaneous translation activation for large audio-language models,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 18 367–18 371.

[16] Z. Yang, L. Wei, R. Koshkin, X. Chen, and S. Nakamura, “Sasst: Leveraging syntax-aware chunking and llms for simultaneous speech translation,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 40, 2026, pp. 34 358–34 367.

[17] B. Fu, D. Yu, M. Liao, C. Li, X. Chen, Y. Chen, K. Fan, and X. Shi, “Efficient and adaptive simultaneous speech translation with fully unidirectional architecture,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 36, 2026, pp. 30 735–30 743.

[18] T. Labiausse, L. Mazaré, E. Grave, A. Défossez, and N. Zeghidour, “High-fidelity simultaneous speech-to speech translation,” in International Conference on Machine Learning. PMLR, 2025, pp. 32 116–32 129.

[19] L. Dong and B. Xu, “Cif: Continuous integrate-and-fire for end-to-end speech recognition,” in ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 6079–6083.

[20] K. Deng, W. Chen, X. Chen, and P. Woodland, “Simuls2s-llm: Unlocking simultaneous inference of speech llms for speech-to-speech translation,” in Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2025, pp. 16 718–16 734.

[21] S. Guo, X. Li, M. Liu, W. Chen, and Y. Feng, “Streamuni: Achieving streaming speech translation with a unified large speech-language model,” arXiv preprint arXiv:2507.07803, 2025.

[22] S. Papi, M. Negri, and M. Turchi, “Attention as a Guide for Simultaneous Speech Translation,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 13 340–13 356.

[23] S. Papi, M. Turchi, and M. Negri, “AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation,” in INTERSPEECH 2023, 2023, pp. 3974–3978.

[24] H. Tan, H. Ouchi, and S. Sakti, “Simulsense: Sense-driven interpreting for efficient simultaneous speech translation,” in ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 19 437–19 441.

[25] I. Tsiamas, G. I. Gállego, J. A. Fonollosa, and M. R. Costa-jussà, “Shas: Approaching optimal segmentation for end-to-end speech translation,” arXiv preprint arXiv:2202.04774, 2022.

[26] S. Papi, M. Gaido, M. Negri, and L. Bentivogli, “Streamatt: Direct streaming speech-to-text translation with attention-based audio history selection,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 3692–3707.

[27] A. Graves, “Sequence transduction with recurrent neural networks,” arXiv preprint arXiv:1211.3711, 2012.

[28] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[29] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi, Y. Wang, K. Chen, P. Zhang, and Z. Wu, “Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation,” in Proc. ofSLT, 2024.

[30] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng et al., “Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition,” in ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 6182–6186.

[31] G. Chen, S. Chai, G.-B. Wang, J. Du, W.-Q. Zhang, C. Weng, D. Su, D. Povey, J. Trmal, J. Zhang et al., “Gigaspeech: An evolving, multi-domain asr corpus with 10,000 hours of transcribed audio,” Interspeech 2021, 2021.

[32] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models,” in International Conference on Learning Representations, 2022. [Online]. Available: https://openreview.net/forum?id=nSbk\_W2q56v

[33] A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna, “Fleurs: Few-shot learning evaluation of universal representations of speech,” in 2022 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2023, pp. 798–805.

[34] C. Wang, A. Wu, J. Gu, and J. Pino, “Covost 2 and massively multilingual speech translation.” in Interspeech, 2021, pp. 2247–2251.

[35] N. M. Guerreiro, R. Rei, D. v. Stigt, L. Coheur, P. Colombo, and A. F. T. Martins, “xCOMET: Transparent machine translation evaluation through fine-grained error detection,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 979–995, 2024. [Online]. Available: https://aclanthology.org/2024.tacl-1.54/

[36] S. Papi, M. Gaido, M. Negri, and M. Turchi, “Over-generation cannot be rewarded: Length-adaptive average lagging for simultaneous speech translation,” in Proceedings of the Third Workshop on Automatic Simultaneous Translation, J. Ive and R. Zhang, Eds. Online: Association for Computational Linguistics, Jul. 2022, pp. 12–17. [Online]. Available: https://aclanthology.org/2022.autosimtrans-1.2/

[37] P. Polák, S. Papi, L. Bentivogli, and O. Bojar, “Better late than never: Meta-evaluation of latency metrics for simultaneous speech-to-text translation,” arXiv preprint arXiv:2509.17349, 2025.

[38] M. Ma, L. Huang, H. Xiong, R. Zheng, K. Liu, B. Zheng, C. Zhang, Z. He, H. Liu, X. Li, H. Wu, and H. Wang, “STACL: Simultaneous translation with implicit anticipation and controllable latency using prefix-to-prefix framework,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, A. Korhonen, D. Traum, and L. Màrquez, Eds. Florence, Italy: Association for Computational Linguistics, Jul. 2019, pp. 3025–3036. [Online]. Available: https://aclanthology.org/P19-1289/

[39] J. Rasley, S. Rajbhandari, O. Ruwase, and Y. He, “Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters,” in Proceedings ofthe 26th ACM SIGKDD international conference on knowledge discovery & data mining, 2020, pp. 3505–3506.

[40] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in International Conference on Learning Representations, 2019.

[41] M. A. Di Gangi, R. Cattoni, L. Bentivogli, M. Negri, and M. Turchi, “Must-c: a multilingual speech trans lation corpus,” in Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 2012–2017.

[42] E. Salesky, K. Darwish, M. Al-Badrashiny, M. Diab, and J. Niehues, “Evaluating multilingual speech translation under realistic conditions with resegmentation and terminology,” in Proceedings ofthe 20th International Conference on Spoken Language Translation (IWSLT 2023), E. Salesky, M. Federico, and M. Carpuat, Eds. Toronto, Canada (in-person and online): Association for Computational Linguistics, Jul. 2023, pp. 62–78. [Online]. Available: https://aclanthology.org/2023.iwslt-1.2/

[43] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu, “Qwen3 technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2505.09388

## A Overall Training Algorithm

Algorithm 1 summarizes the complete end-to-end training procedure of our proposed Actor-Critic framework for simultaneous translation.

Algorithm 1 Actor-Critic Training for Simultaneous Translation   
Input: Dataset $\overline { { \mathcal { D } = \{ ( { \bf S } , { \bf T } ) \} } }$ , Actor parameters $\theta _ { a c t } .$ , Critic parameters $\theta _ { c r i } .$ Learning rate $\eta ,$   
Decision threshold τ, Loss weights $\lambda , \alpha , \beta ,$ critic step num K.   
1: while not converged do   
2: Sample a batch of speech-text pairs: $( \mathbf { S } , \mathbf { T } ) \sim \mathcal { D }$   
3: // Step 1: Actor Exploration & Transition Dynamics $( \cal { S } , \cal { A } , \mathcal { P } )$   
4: Sample alignment path $( a _ { k } \in \mathcal { V } \cup \{ \phi \}$ , ϕ denotes Read): a $\sim \pi _ { a c t } ( \cdot \mid \mathbf { S } , \mathbf { T } ; \theta _ { a c t } )$   
5: Construct interleaved state sequence: $\mathbf { \dot { z } } = \operatorname { I n t e r l e a v e } ( \mathbf { S } , \mathbf { T } , \mathbf { a } )$   
6: // Step 2: Critic Forward Pass & State Evaluation   
7: Extract main body hidden states: $\mathbf { H } = \mathrm { L L M } ( \mathbf { z } ; \theta _ { c r i } )$   
8: Predict router probabilities $( p _ { k } \in [ 0 , 1 ] ) \colon p _ { k } =$ Router $\left( \mathbf { H } _ { k - 1 } ; \theta _ { c r i } \right)$   
9: Extract global acoustic state: $\mathbf { h } _ { g l o b a l } $ AvePool $\big ( \mathrm { L L M } ( \mathbf { S } ; \boldsymbol { \theta } _ { c r i } ) \big )$   
10: Predict refiner distributions: $P _ { r e f i n e r } ( t _ { i } \mid \mathbf { z } _ { \leq \mathrm { i d x } ( i ) } , \mathbf { H } , \mathbf { h } _ { g l o b a l } ; \theta _ { c r i } )$   
11: // Step 3: Compute Reward Function R & Critic Loss   
12: Compute Consistency Reward:   
13: R<sub>consistency</sub> $\begin{array} { r } {  \sum _ { i } \log P _ { r e f i n e r } ( t _ { i } \mid \cdot ) + \sum _ { \{ k | p _ { k } < \tau \} } } \end{array}$ log $P _ { l l m } ( y _ { k } \mid \mathbf { H } _ { k } )$   
14: Compute Latency Reward: $\begin{array} { r } { \mathcal { R } _ { l a t e n c y }  \sum _ { k } \sum _ { m = 1 } ^ { M } \prod _ { j = 1 } ^ { m } ( 1 - p _ { k + j } ) } \end{array}$   
15: Compute Load Balancing constraint: $\mathcal { L } _ { b a l a n c e } ( p _ { k } )$   
16: Aggregate Critic cost: $\bar { \mathcal { T } } _ { c r i t i c } = - \mathcal { R } _ { c o n s i s t e n c y } - \alpha \mathcal { R } _ { l a t e n c y } + \beta \mathcal { L } _ { b a l a n c e }$   
17: // Step 4: Policy Feedback & Actor Optimization   
18: Compute base Actor loss: $\mathcal { L } _ { r n n t } = - \mathrm { l o g } \pi _ { a c t } ( \mathbf { T } \mid \mathbf { S } ; \boldsymbol { \theta } _ { a c t } )$   
19: Derive optimal action targets from Critic: $\hat { a } _ { k } = \mathbb { I } ( p _ { k } \ge \tau )$   
20: Compute policy distillation loss: $\begin{array} { r } { \mathcal { L } _ { d i s t i l l } = \sum _ { k } \mathrm { B C E } \Big ( \hat { a } _ { k } , \pi _ { a c t } ( a _ { k } = \phi \mid \cdot ; \theta _ { a c t } ) \Big ) } \end{array}$   
21: // Step 5: Parameter Updates   
22: if t mod $( K + 1 ) = 0$ then   
23: Update Actor parameters: $\theta _ { a c t } \gets \theta _ { a c t } - \eta \nabla _ { \theta _ { a c t } } \big ( \mathcal { L } _ { d i s t i l l } + \lambda \mathcal { L } _ { r n n t } \big )$   
24: else   
25: Update Critic parameters: $\theta _ { c r i }  \theta _ { c r i } - \eta \nabla _ { \theta _ { c r i } } \mathcal { I } _ { c r i t i c }$   
26: end if   
27: end while

## B Convergence Analysis of the Reverse Distillation Loop

This appendix provides a local convergence analysis for the Actor–Critic reverse distillation loop in Section 3.2, formalizing the intuition that the Actor’s quality is controlled by the Critic’s quality despite the self-referential pseudo-labels. The analysis is local; we do not claim global convergence on a non-convex landscape.

We re-parameterize the Actor and Critic as $\theta \left( \theta _ { a c t } \right)$ and $\phi \left( \theta _ { c r i } \right)$ for analytical convenience. The two-timescale update at outer step t runs $K = 8$ Critic SGD steps followed by one Actor step; the pseudo-label $\hat { a } _ { k } ~ =$ $\mathbb { I } ( \mathrm { s g } [ p _ { k } ( \phi ) ] \ge \tau )$ uses a stop-gradient on $\phi .$ Throughout, we assume standard regularity in a local neighborhood $\dot { \mathcal { N } } \times \dot { \mathcal { M } }$ of an equilibrium $( \theta ^ { * } , \phi ^ { * } )$ : local smoothness of $J _ { a }$ and $J _ { c }$ in their first arguments, a quadratic-growth condition for the Critic around $\phi ^ { * } ( { \dot { \theta } } )$ := arg min<sub>ϕ</sub> $J _ { c } ( \phi ; \theta )$ , cross-Lipschitzness of $\nabla _ { \phi } J _ { c }$ in θ, a Tsybakov-style margin condition $\mathbb { P } ( | p ( s ; \phi ) - \tau | \leq \delta ) \leq \kappa \delta$ on the Router output, a Polyak–Łojasiewicz condition for $J _ { a }$ unbiased minibatch gradients with bounded variance $\sigma ^ { 2 }$ , and bounded gradients on $\mathcal N \times \mathcal M$ . These conditions are weaker than those typically invoked in the analysis of two-timescale stochastic approximation (Borkar 2008; Karimi et al. 2016) and are widely satisfied by overparameterized neural networks. We refer the reader to standard references for their formulation.

Theorem 1 (Local linear convergence to a noise-controlled neighborhood). Let $e _ { t } : = \phi _ { t + 1 } - \phi ^ { * } ( \theta _ { t } )$ denote the Critic tracking error and $\begin{array} { r } { V _ { t } : = J _ { a } ( \theta _ { t } ; \phi ^ { * } ( \theta _ { t } ) ) - \operatorname* { m i n } _ { \theta } J _ { a } ( \bar { \theta ; } \phi ^ { * } ( \theta _ { t } ) ) } \end{array}$ the Actor optimality gap. Under the regularity conditions above, there exist constants $\stackrel { \cdot \mathrm { \tiny ~ 0 } } { \eta _ { a } ^ { 0 } } , \eta _ { c } ^ { 0 } , K _ { 0 } , c , \stackrel { \cdot \mathrm { \tiny ~ C } } { \cal { C } } _ { \sigma } > 0$ and a contraction rate $\rho ^ { * } \in ( 0 , 1 )$ such

that for any

$$
\eta _ { c } \leq \eta _ { c } ^ { 0 } , \qquad \eta _ { a } \leq \eta _ { a } ^ { 0 } , \qquad K \geq K _ { 0 } ,
$$

and any initialization $( \theta _ { 0 } , \phi _ { 0 } ) \in \mathcal { N } \times \mathcal { M }$ , the Lyapunov function $\Phi _ { t } : = V _ { t } + c \mathbb { E } \lVert e _ { t } \rVert ^ { 2 }$ satisfies

$$
\Phi _ { t } \ \leq \ ( \rho ^ { * } ) ^ { t } \Phi _ { 0 } \ + \ \frac { C _ { \sigma } \left( \eta _ { a } + \eta _ { c } \right) \sigma ^ { 2 } } { 1 - \rho ^ { * } } .\tag{7}
$$

A valid concrete choice is $c = 4 \eta _ { a } L _ { a , \phi } ^ { 2 } / \mu _ { a }$ and $K _ { 0 } \ = \ \lceil \log ( 8 L _ { \phi } ^ { 2 } \eta _ { a } ^ { 0 } G ^ { 2 } / \mu _ { a } ) / \log ( 1 / \rho ) \rceil$ , yielding $\rho ^ { * } = $ $\operatorname* { m a x } ( 1 - \mu _ { a } \eta _ { a } / 2 , 3 / 4 )$ , where $L _ { a , \phi } = \dot { O } ( \kappa L _ { p } )$ controls the Actor’s sensitivity to Critic perturbations through the margin condition.

Proof. The proof has three steps: the Critic tracking error contracts geometrically, the Actor optimality gap descends modulo Critic error, and a weighted Lyapunov function combines the two.

Critic tracking error. The map $\theta \mapsto \phi ^ { * } ( \theta )$ is L<sub>ϕ</sub>-Lipschitz with $L _ { \phi } \leq L _ { c , \theta } / \mu _ { c }$ , by a standard implicitfunction argument under quadratic growth (Karimi et al. 2016, Thm. 5). Since $\phi _ { t , 0 } = \phi _ { t } = \phi ^ { * } ( \theta _ { t - 1 } ) + e _ { t - 1 } ,$ we obtain

$$
\begin{array} { r } { \left\| \phi _ { t , 0 } - \phi ^ { * } ( \theta _ { t } ) \right\| \leq \| e _ { t - 1 } \| + L _ { \phi } \| \theta _ { t } - \theta _ { t - 1 } \| \leq \| e _ { t - 1 } \| + L _ { \phi } \eta _ { a } G , } \end{array}
$$

where the second inequality uses the bounded-gradient assumption. The standard SGD analysis under quadratic growth yields, for the K-step inner loop with $\dot { \eta } _ { c } \leq 1 / L _ { c }$

$$
\mathbb { E } \| \phi _ { t , K } - \phi ^ { * } ( \theta _ { t } ) \| ^ { 2 } \ \leq \ \rho ^ { K } \mathbb { E } \| \phi _ { t , 0 } - \phi ^ { * } ( \theta _ { t } ) \| ^ { 2 } + \frac { \eta _ { c } \sigma ^ { 2 } } { \mu _ { c } } , \qquad \rho : = 1 - \eta _ { c } \mu _ { c } .
$$

Squaring and combining the two displays via $( a + b ) ^ { 2 } \leq 2 a ^ { 2 } + 2 b ^ { 2 }$

$$
\mathbb { E } \Vert e _ { t } \Vert ^ { 2 } \leq 2 \rho ^ { K } \mathbb { E } \Vert e _ { t - 1 } \Vert ^ { 2 } + 2 \rho ^ { K } L _ { \phi } ^ { 2 } \eta _ { a } ^ { 2 } G ^ { 2 } + \frac { \eta _ { c } \sigma ^ { 2 } } { \mu _ { c } } .\tag{8}
$$

Choosing $K _ { 0 }$ so that $2 \rho ^ { K _ { 0 } } \le 1 / 2$ (this is the $K _ { 0 }$ stated in the theorem),

$$
\begin{array} { r } { \mathbb { E } \| e _ { t } \| ^ { 2 } \ \leq \ \frac { 1 } { 2 } \mathbb { E } \| e _ { t - 1 } \| ^ { 2 } + \underbrace { \rho ^ { K } L _ { \phi } ^ { 2 } \eta _ { a } ^ { 2 } G ^ { 2 } + \frac { \eta _ { c } \sigma ^ { 2 } } { \mu _ { c } } } _ { = : \varepsilon _ { c } } . } \end{array}\tag{9}
$$

Actor descent. Let $g _ { t } ^ { * } : = \nabla _ { \boldsymbol { \theta } } J _ { a } ( { \boldsymbol { \theta } } _ { t } ; { \boldsymbol { \phi } } ^ { * } ( { \boldsymbol { \theta } } _ { t } ) )$ and $g _ { t } : = \nabla _ { \theta } J _ { a } ( \theta _ { t } ; \phi _ { t + 1 } )$ . The pseudo-label $\hat { a } _ { k }$ flips between $\phi _ { 1 }$ and $\phi _ { 2 }$ only on the event $\{ | p ( s ; \phi _ { 1 } ) - \tau | \le L _ { p } \| \phi _ { 1 } - \phi _ { 2 } \| \}$ , which by the margin condition has probability at most $\kappa L _ { p } \big | \big | \phi _ { 1 } - \phi _ { 2 } \big | \big |$ . Combined with the Lipschitzness of BCE in its target argument, this yields

$$
\begin{array} { r } { \| g _ { t } - g _ { t } ^ { * } \| \le L _ { a , \phi } \| e _ { t } \| , \qquad L _ { a , \phi } = O ( \kappa L _ { p } ) . } \end{array}\tag{10}
$$

By smoothness of $J _ { a }$ in θ (with the second argument held at $\phi ^ { * } ( \theta _ { t } ) )$ and unbiasedness of the stochastic gradient,

$$
\begin{array} { r } { \mathbb { E } J _ { a } \big ( \theta _ { t + 1 } ; \phi ^ { * } ( \theta _ { t } ) \big ) \le J _ { a } \big ( \theta _ { t } ; \phi ^ { * } ( \theta _ { t } ) \big ) - \eta _ { a } \langle g _ { t } ^ { * } , g _ { t } \rangle + L _ { a } \eta _ { a } ^ { 2 } \| g _ { t } \| ^ { 2 } + L _ { a } \eta _ { a } ^ { 2 } \sigma ^ { 2 } . } \end{array}
$$

Substituting $g _ { t } = g _ { t } ^ { * } + ( g _ { t } - g _ { t } ^ { * } )$ , using Cauchy–Schwarz and Young’s inequality together with (10), and applying the PL condition $\| g _ { t } ^ { * } \| ^ { 2 } \geq 2 \mu _ { a } V _ { t }$

$$
\begin{array} { r l r } { \mathbb { E } J _ { a } \big ( \theta _ { t + 1 } ; \phi ^ { * } ( \theta _ { t } ) \big ) } & { \le } & { J _ { a } \big ( \theta _ { t } ; \phi ^ { * } ( \theta _ { t } ) \big ) - \mu _ { a } \eta _ { a } V _ { t } + 2 \eta _ { a } L _ { a , \phi } ^ { 2 } \mathbb { E } \| e _ { t } \| ^ { 2 } + L _ { a } \eta _ { a } ^ { 2 } \sigma ^ { 2 } , } \end{array}
$$

provided $\eta _ { a } \leq 1 / ( 4 L _ { a } )$ . Shifting the second argument from $\phi ^ { * } ( \theta _ { t } )$ to $\phi ^ { * } ( \theta _ { t + 1 } )$ contributes only $O ( \eta _ { a } ^ { 2 } )$ by the implicit-function bound and a standard envelope argument; folding this into the noise term gives

$$
\begin{array} { r } { \mathbb { E } V _ { t + 1 } \ \leq \ \left( 1 - \mu _ { a } \eta _ { a } \right) V _ { t } + 2 \eta _ { a } L _ { a , \phi } ^ { 2 } \mathbb { E } \| e _ { t } \| ^ { 2 } + C _ { 1 } \eta _ { a } ^ { 2 } \sigma ^ { 2 } . } \end{array}\tag{11}
$$

Lyapunov function. Set $\Phi _ { t } : = V _ { t } + c \mathbb { E } \Vert e _ { t } \Vert ^ { 2 }$ with $c = 4 \eta _ { a } L _ { a , \phi } ^ { 2 } / \mu _ { a }$ . Combining (9) and (11),

$$
\begin{array} { r } { \mathbb { E } \Phi _ { t + 1 } \ \leq \ ( 1 - \mu _ { a } \eta _ { a } ) V _ { t } + \left( 2 \eta _ { a } L _ { a , \phi } ^ { 2 } + \frac { c } { 2 } \right) \mathbb { E } \| e _ { t } \| ^ { 2 } + C _ { \sigma } ( \eta _ { a } + \eta _ { c } ) \sigma ^ { 2 } . } \end{array}
$$

With our choice of $c ,$ the coefficient on $\mathbb { E } \Vert e _ { t } \Vert ^ { 2 }$ equals $\frac { c ( 1 + \mu _ { a } ) } { 2 } ~ \leq ~ \frac { 3 c } { 4 }$ (after a harmless rescaling so that $\mu _ { a } \leq 1 / 2 )$ . Hence

$$
\begin{array} { r l r } { \mathbb { E } \Phi _ { t + 1 } \le \rho ^ { * } \mathbb { E } \Phi _ { t } + C _ { \sigma } ( \eta _ { a } + \eta _ { c } ) \sigma ^ { 2 } , } & { } & { \rho ^ { * } : = \operatorname* { m a x } \big ( 1 - \frac { \mu _ { a } \eta _ { a } } { 2 } , \frac { 3 } { 4 } \big ) , } \end{array}
$$

and iterating yields (7).

Interpretation. The bound decomposes joint convergence into two coupled contractions: the Critic tracks $\phi ^ { * } ( \theta _ { t } )$ at geometric rate $\rho ^ { K }$ per outer step, and the Actor descends in $J _ { a }$ at rate $1 - \mu _ { a } \eta _ { a }$ modulo a perturbation proportional to the Critic’s tracking error. The two-timescale schedule $K = 8$ ensures $\rho ^ { K }$ is small enough that the Critic does not bottleneck the Actor. The margin condition enters only through the constant $L _ { a , \phi } = \bar { O ( \kappa L _ { p } ) }$ a sharper Router output (smaller κ) means small Critic perturbations rarely flip the binarized pseudo-label, so the Actor is robust to a noisy or undertrained Critic. At stationarity,

$$
\operatorname* { l i m } _ { t } \operatorname* { s u p } \mathbb { E } V _ { t } \leq \frac { 2 L _ { a , \phi } ^ { 2 } } { \mu _ { a } } \cdot \operatorname* { l i m } _ { t } \operatorname* { s u p } _ { \tau } \mathbb { E } \| e _ { t } \| ^ { 2 } + O ( \eta _ { a } \sigma ^ { 2 } ) ,
$$

which formalizes the qualitative claim that the Actor is no worse than the Critic, up to a constant determined by the margin and PL parameters. We do not claim global convergence, nor that the equilibrium matches a true oracle policy—the latter would require $J _ { c }$ being a faithful surrogate for translation quality.

## C Extended Experimental Details

In this section, we provide comprehensive details regarding the training configurations, inference setup, and baseline implementations to facilitate reproducibility. Detailed hyperparameter configurations will be made publicly available alongside our open-sourced code.

## C.1 Load Balancing Loss

To prevent routing collapse in our two-expert Refiner architecture and maintain numerical stability, we adopt a lightweight, combined load-balancing loss. We first define a squared soft margin penalty, $\mathcal { M } ( v ; a , b )$ , which continuously penalizes a variable v when it deviates from a target interval $[ a , b ]$

$$
\mathcal { M } ( v ; a , b ) = \left\{ { \begin{array} { l l } { ( a - v ) ^ { 2 } , } & { \mathrm { i f ~ } v < a } \\ { ( v - b ) ^ { 2 } , } & { \mathrm { i f ~ } v > b } \\ { 0 , } & { \mathrm { o t h e r w i s e } } \end{array} } \right.\tag{12}
$$

This squared formulation ensures smooth, differentiable gradients at the boundaries.

Given a batch of N tokens, let $x _ { i }$ denote the raw routing logit for the i-th token, and $p _ { i } = \sigma ( x _ { i } )$ be its probability of being routed to the first expert. Let $\begin{array} { r } { \bar { p } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \bar { p } _ { i } } \end{array}$ represent the batch-averaged routing probability. Our overall balancing loss, $L _ { b a l }$ , simultaneously regularizes the micro-level logits to prevent Sigmoid saturation and constrains the macro-level routing probabilities to ensure equitable expert utilization:

$$
L _ { b a l a n c e } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { M } ( x _ { i } ; - 5 , 5 ) + \mathcal { M } ( \bar { p } ; 0 . 3 , 0 . 7 )\tag{13}
$$

By softly bounding individual logits within $[ - 5 , 5 ]$ and the average routing probability within [0.3, 0.7], the model mathematically guarantees statistical load balancing while preserving token-level routing flexibility.

## C.2 Training and Inference Details

Hardware and Environment All models were trained and evaluated on a single computation node equipped with eight NVIDIA A100 (80GB) GPUs. Our implementation is built upon the PyTorch framework and PyTorch Lightning. To optimize the memory footprint during training, we integrated DeepSpeed ZeRO Stage 2[39] alongside gradient checkpointing.

Optimization and Scheduling Across all training stages, we employed the AdamW[40] optimizer with $\bar { \beta _ { 1 } } \bar { = } 0 . 9 , \beta _ { 2 } = 0 . 9 8$ , and a weight decay of 0.001. The learning rate was governed by a linear decay schedule following a 500-step linear warmup. During the parameter-efficient fine-tuning of the Speech LLM backbone, LoRA was applied to all linear modules (excluding word embeddings and the final projection layer) with a rank of $r = 1 6$ and an alpha parameter of $\alpha = 3 2$ . The peak learning rate was set to $\bar { 1 . 0 \times 1 0 ^ { - 4 } }$ for Stages 1 and 2, and $2 . 0 \times 1 0 ^ { - 5 }$ for Stage 3. Additionally, the training batch size per GPU was dynamically bounded to a maximum of 30 seconds of audio. Training of each stage takes about 8 hours.

Inference Setup For sentence-level inference, we utilized Voice Activity Detection (VAD) exclusively to truncate initial silence at the beginning of audio clips. For document-level inference, VAD was applied solely to filter out excessively long non-speech pauses $( \mathrm { e . g . , > 2 }$ seconds) to prevent severe distribution mismatches with the training data. Acoustic features were continuously extracted using a 400ms window. The KV cache for both the acoustic prefix and the generated text tokens was dynamically maintained and incrementally updated. To minimize inference latency, we adopted a greedy decoding strategy coupled with a windowed 4-gram no-repeat logits processor to prevent degenerative repetitions. All reported latency metrics—for both our proposed method and the baselines—were computed at the word level (or character level for CJK languages).

## C.3 Baseline Implementation Details

To ensure a fair and rigorous comparison, we carefully configured all baselines according to their official implementations or API specifications. Both the proposed method and the baselines were evaluated within a unified framework, closely mirroring the official SimulEval protocol. Greedy decoding was uniformly applied across all baseline models.

• Seamless Streaming [2]: We utilized the official seamless-streaming checkpoint. Its read-write strategy is governed by the native EMMA policy, and we generated the quality-latency Pareto frontier by adjusting the EMMA decision threshold. The audio input stride was set to 400ms, which is identical to our method.

• AlignATT [23] & StreamATT [26]: We followed the implementations of AlignATT and StreamATT provided in the InfiniSST repository<sup>4</sup>, adapting them to operate on top of the original Qwen2.5-Omni Model. We varied the frame boundary to trace the quality-latency trade-off. We have compared extracting attention weights from the middle layer, the final layer, and averaging over layers, showing that an extract from the middle layer yields slightly better results. For StreamATT, the context window was restricted to 40 text tokens, with the retained acoustic history bounded between 10 and 30 seconds. Infinite repetition is hard to avoid in StreamATT, even with the no-repeats logits processor. So we set repetition\_penalty=1.2 in StreamATT.

• StreamUni [21]: We evaluated this baseline using its official checkpoints and code. We use chunk size=640ms and wait-k=3.

• InfiniSST [7]: We evaluated this baseline using its official checkpoints<sup>5</sup>. Latency was modulated by varying the input chunk size from 960ms to 4800ms. Since we observed that these checkpoints exhibited degenerate performance during sentence-level evaluations, we only report their document-level results.

• Seed LiveInterpret 2.0: This model was evaluated via the commercial Volcengine WebSocket $\mathrm { A P I } . ^ { 6 }$ . To minimize network-induced variance, we ensured the network Round Trip Time (RTT) remained strictly below 20ms during all queries. Due to budget constraints, this API was evaluated exclusively on the FLEURS test set.

## D Additional Experiments

## D.1 Compatibility with RL-based Policy Optimization

CRESiST is an SFT-time policy-discovery framework: translation is still learned by supervised maximum likelihood on the offline corpus, while the read-write policy is discovered by the Actor-Critic loop. It is therefore complementary to, rather than a substitute for, RL-based policy optimization, and the resulting model is a valid starting point for subsequent RL. To verify this, we apply GRPO on top of CRESiST following the reward design of HPO [13]. For reference, we apply the same recipe to an InfiniSST-style SFT baseline trained on the same Qwen2.5-Omni backbone with RNN-T aligned data (the “Direct SFT” setting of Section 5). CRESiST+GRPO is trained for 700 steps on the multilingual corpus, and SFT+GRPO for 1000 steps on En-Zh. Table 3 reports document-level En-Zh results in the streaming w/o segmentation setting.

Table 3: Document-level En-Zh results (streaming w/o segmentation, XCOMET-XXL ↑ / LongYAAL ↓) with and without GRPO post-training. SFT denotes an InfiniSST-style supervised baseline on the same backbone.
<table><tr><td></td><td colspan="2">ACL 60/60</td><td colspan="2">MuST-C</td></tr><tr><td>System</td><td>COMET</td><td>LongYAAL</td><td>COMET</td><td>LongYAAL</td></tr><tr><td>SFT (InfiniSST-style)</td><td>68.43</td><td>2.27</td><td>66.16</td><td>2.34</td></tr><tr><td>SFT + GRPO (HPO recipe)</td><td>73.64</td><td>2.15</td><td>69.71</td><td>2.15</td></tr><tr><td>CRESiST</td><td>79.45</td><td>1.49</td><td>73.73</td><td>1.14</td></tr><tr><td>CRESiST + GRPO</td><td>80.04</td><td>1.37</td><td>78.47</td><td>1.23</td></tr></table>

Three observations follow. (1) CRESiST is fully compatible with RL: GRPO further improves it on both benchmarks (e.g., +4.7 COMET on MuST-C) without hurting latency. (2) RL does not compensate for a weak SFT base: CRESiST without any RL already outperforms SFT+GRPO by a clear margin in both quality and latency, so SFT-time policy learning remains meaningful even when RL post-training is available. (3) Our GRPO gains are smaller than those reported in the HPO paper, which we attribute to the stronger starting point. We also note that after GRPO post-training the model streams raw audio stably without the VAD pause filtering described in Appendix C.

![](images/a7e0a943674ce957fb37d1d0f429dbe1c3af7f83fb090cc31dcced9280299aac.jpg)

(a) Proposed RNN-T Convention RNN-T  
![](images/e97a3ab7d26961d6273d30d625fd443d629abca5a0d143145be8c996ee406bc8.jpg)  
(b) Conventional RNN-T  
Figure 6: Comparison of emission probabilities and decoding paths. The proposed RNN-T with axial attention (a) successfully maintains an incremental, monotonic alignment for simultaneous translation. In contrast, the conventional RNN-T (b) fails to emit tokens timely during the speech stream, causing its decoding path to delay and degenerate into an offline translation behavior at the end of the utterance.

## D.2 The RNN-T Actor as an Inference Model

A natural question is whether the self-enhancement loop improves the RNN-T Actor enough for it to serve as the final inference model on its own, which would also quantify how much of the final performance is contributed by the Speech LLM. Our Actor is itself a complete simultaneous translation model trained on the same data (Whisper-medium encoder, Qwen3-0.6B predictor with axial attention, and a joint network), so we decode it directly with streaming greedy decoding (blank = Read, non-blank = Write), both before (Stage 1) and after (Stage 3) reverse distillation. Table 4 reports sentence-level results on FLEURS.

The results confirm that the RNN-T is effective as an Actor but unsuitable as the final translator. Reverse distillation transfers the policy, not the quality: the Actor’s COMET is essentially unchanged, while its latency drops markedly (Zh-En LAAL 2.29s → 1.52s, close to that of the Speech LLM), showing that its read/write behavior has been aligned with the Critic. Its translation quality, however, remains 33–37 XCOMET points below the Speech LLM; the RNN-T decides reliably when to translate, but its 0.6B predictor lacks the linguistic and world knowledge needed for what to generate, with named-entity hallucinations being the dominant failure mode. The RNN-T is also an inefficient inference model: each step evaluates the joint network along an entire row or column of the $( T \times U )$ lattice, so its decoding cost grows with the product of the audio and text lengths rather than with the sequence length, and scaling the transducer toward the 7B class would multiply this lattice cost while still lacking pretrained LLM knowledge. These results cleanly separate the two roles in our framework, where the Actor supplies the policy and the Speech LLM supplies the quality, and justify keeping a small RNN-T as the policy explorer paired with a pretrained Speech LLM as the generator.

Table 4: Using the RNN-T Actor directly as the inference model on FLEURS (XCOMET-XXL ↑ / LAAL ↓), before and after reverse distillation, compared with the full CRESiST Speech LLM.
<table><tr><td rowspan="2">Inference model</td><td colspan="2">En-Zh</td><td colspan="2">Zh-En</td></tr><tr><td>COMET</td><td>LAAL</td><td>COMET</td><td>LAAL</td></tr><tr><td>RNN-T Actor, before distillation</td><td>43.7</td><td>3.20</td><td>55.5</td><td>2.29</td></tr><tr><td>RNN-T Actor, after distillation</td><td>44.1</td><td>3.02</td><td>55.1</td><td>1.52</td></tr><tr><td>CRESiST (Speech LLM)</td><td>80.5</td><td>1.78</td><td>88.6</td><td>1.54</td></tr></table>

## E Case Visualization

In this section, we present case visualizations to demonstrate the effectiveness of our proposed framework from two distinct perspectives: the internal alignment quality of our speech-text aligner, and the overall end-to-end translation latency.

## E.1 Analysis of RNN-T Alignment Quality

First, we examine the internal behavior of our proposed RNN-T aligner. Figure 6 visualizes the emission probabilities and the resulting decoding paths for both the proposed RNN-T (with axial attention) and a conventional RNN-T on a specific translation case.

As explicitly shown in Figure 6b, the conventional RNN-T struggles to confidently predict the necessary text tokens during the ongoing speech stream. Due to this failure in timely token emission, it cannot establish a proper incremental alignment. Consequently, the decoding path is forced to delay the majority of its predictions until the very end of the speech input (around chunk 33), effectively degenerating the system into an offline translation paradigm.

In contrast, our proposed RNN-T (Figure 6a) overcomes this critical limitation. It yields significantly sharper and more concentrated emission probabilities, successfully predicting tokens as the speech stream unfolds. This distinct, high-quality monotonic decoding path demonstrates that our approach effectively maintains the incremental alignment required for true simultaneous translation, avoiding the offline degradation seen in the conventional baseline.

## E.2 End-to-End Translation Timing Comparison

Furthermore, we visualize the end-to-end translation timing of a specific case from the FLEURS EN-ZH test set to compare our approach directly with the Seed LiveInterpret 2.0 API.

As illustrated in Figure 7, Seed LiveInterpret 2.0 exhibits a strong preference for coarse-grained, clause-level translation, often waiting for larger chunks of speech context to accumulate before generating an output. Conversely, our proposed model performs fine-grained, adaptive translation at the word level. This allows our model to execute more immediate and flexible read/write decisions, thereby effectively reducing translation latency while maintaining high generation quality.

## F Assets and Licenses

Table 5 details the existing assets (including datasets, models, and codebases) utilized in this work, alongside their licenses and reference information.

![](images/60481b9ea4aba298fa8c5dc06226df729b8d71fc9036d9a35a9f3a216605ddc8.jpg)  
2. CRESiST (Computation Unaware)

![](images/d1603eac11874de3e3e21cdb6f68eca302b4ce1d7b2d095aff4bc74988d65bc5.jpg)  
3. CRESiST (Computation Aware)

![](images/c8e52a9da81d7c3e30904506c4e580d744740d4a9ebb4de53df274d7b10c5a90.jpg)  
Figure 7: Translation timing visualization for a sample from the FLEURS EN-ZH test set. The timeline illustrates the latency advantage of our model, which performs fine-grained, word-level adaptive translation, in contrast to the coarse-grained, clause-level buffering strategy employed by the Seed LiveInterpret 2.0 API.

Table 5: Summary of existing datasets, models, and codebases used in this paper.
<table><tr><td>Asset Name</td><td>Type</td><td>License</td><td>Reference</td><td>URL</td></tr><tr><td colspan="5">Datasets</td></tr><tr><td>Emilia</td><td>Dataset</td><td>CC BY 4.0</td><td>[29]</td><td>https://huggingface.co/datasets/amphion/Emilia-Dataset</td></tr><tr><td>WenetSpeech</td><td>Dataset</td><td>CC BY 4.0</td><td>[30]</td><td>https://www.openslr.org/121/</td></tr><tr><td>GigaSpeech</td><td>Dataset</td><td>Apache 2.0</td><td>[31]</td><td>https://github.com/SpeechColab/GigaSpeech</td></tr><tr><td>FLEURS</td><td>Dataset</td><td>CC BY 4.0</td><td>[33]</td><td>https://huggingface.co/datasets/google/fleurs</td></tr><tr><td>CoVoST 2</td><td>Dataset</td><td>CC BY-NC 4.0</td><td>[34]</td><td>https://huggingface.co/datasets/facebook/covost2</td></tr><tr><td>MuST-C ACL 60/60</td><td>Dataset</td><td>CC BY-NC-ND 4.0</td><td>[41]</td><td>https://opendatalab.com/OpenDataLab/MuST-C</td></tr><tr><td></td><td>Dataset</td><td>CC BY 4.0</td><td>[42]</td><td>https://aclanthology.org/attachments/2023.iwslt-1.2.dataset.zip</td></tr><tr><td colspan="5">Models</td></tr><tr><td>Qwen2.5-omni-7B</td><td>Model</td><td>Apache 2.0</td><td>[14]</td><td>https://huggingface.co/Qwen/Qwen2.5-Omni-7B</td></tr><tr><td>Qwen3-0.6B</td><td>Model</td><td>Apache 2.0</td><td>[43]</td><td>https://huggingface.co/Qwen/Qwen3-0.6B</td></tr><tr><td>Qwen3-235B-A22B-Instruct</td><td>Model</td><td>Apache 2.0</td><td>[43]</td><td>https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507</td></tr><tr><td>Whisper-medium</td><td>Model</td><td>Apache 2.0</td><td>[28]</td><td>https://github.com/openai/whisper</td></tr><tr><td>Seamless Streaming</td><td>Model</td><td>CC BY-NC 4.0</td><td>[2]</td><td>https://huggingface.co/facebook/seamless-streaming</td></tr><tr><td>StreamUni</td><td>Model</td><td>Apache 2.0</td><td>[21]</td><td>https://huggingface.co/ICTNLP/StreamUni-Phi4</td></tr><tr><td>InfiniSST</td><td>Model</td><td>MIT</td><td>[7]</td><td>https://github.com/LeiLiLab/InfiniSST</td></tr><tr><td colspan="5">Codebases &amp; Frameworks &amp; Metrics</td></tr><tr><td>PyTorch PyTorch Lightning</td><td>Framework</td><td>BSD-3</td><td></td><td>https://pytorch.org/</td></tr><tr><td>DeepSpeed</td><td>Framework Codebase</td><td>Apache 2.0</td><td>[39]</td><td>https://github.com/Lightning-AI/pytorch-lightning</td></tr><tr><td>OmniSTEval</td><td>Codebase</td><td>Apache 2.0 Apache 2.0</td><td>[37]</td><td>https://github.com/deepspeedai/deepspeed https://github.com/pe-trik/OmniSTEval/tree/main</td></tr><tr><td>XCOMET-XXL</td><td>Metric</td><td></td><td>[35]</td><td></td></tr><tr><td></td><td></td><td>CC-BY-NC-SA 4.0</td><td></td><td>https://huggingface.co/Unbabel/XCOMET-XXL</td></tr></table>

## G Broader Impacts

Our proposed CRESiST framework significantly advances simultaneous speech translation, which has the positive potential to break down cross-lingual barriers, facilitate global communication, and improve accessibility for non-native speakers or the hearing-impaired. However, we must also acknowledge potential negative societal impacts. First, like many LLM-based systems, our model may exhibit performance disparities across different demographics, potentially showing bias against non-standard accents or low-resource dialects. Relying on such systems in high-stakes scenarios (e.g., medical or legal settings) without human oversight could lead to critical misunderstandings. Furthermore, the extensive computational resources required for training such omni-modal LLMs contribute to a substantial carbon footprint. We encourage future research to address these fairness and efficiency challenges to ensure safe and equitable deployment.