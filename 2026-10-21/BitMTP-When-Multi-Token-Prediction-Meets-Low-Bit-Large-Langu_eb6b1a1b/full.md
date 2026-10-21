# BitMTP: When Multi-Token Prediction Meets Low-Bit Large Language Models

Ning Zhang Shihao Wang Jinrui Zhang Chaodong Xiao Lei Zhang<sup>∗</sup> Department of Computing, The Hong Kong Polytechnic University ning24.zhang@polyu.edu.hk, cslzhang@comp.polyu.edu.hk   
{shihaow.wang,jin-rui.zhang,chaodong.xiao}@connect.polyu.hk

## Abstract

For efficient deployment of large language models (LLMs), methods of ultra-lowbit backbones reduce weight storage, memory traffic, and linear-layer execution cost, while multi-token prediction (MTP) methods reduce sequential decoding steps by predicting multiple future tokens in parallel. Although complementary, the direct combination of these methods cannot accelerate decoding at comparable quality: masked drafting lacks intermediate token input supplied in next-token training, while wider computation and causal key–value (KV) updates can offset the savings from fewer steps. To address these challenges, we propose BitMTP, an algorithm–system co-design framework for efficient MTP on ultra-low-bit LLM backbones. A joint next-token prediction and MTP objective equips a single model with both autoregressive (AR) generation and block-causal drafting. Confidenceonly decoding, speculative-only decoding, and hybrid decoding provide different quality–throughput operating points without using separate draft and verifier models. Accepted-prefix alignment keeps the committed prefix, materialized causal KV boundary, and CUDA Graph replay state consistent for continued generation. BitMTP uses this contract to select width-specialized low-bit execution and reuse captured graphs over shared KV storage. On NVIDIA H20, BitMTP with hy brid decoding achieves five-task mean speedups of 2.2× on BitNet b1.58-2B and 2.1–2.4× on 1.7B–4B Bonsai models at comparable task quality. On RTX 3090 and Jetson Orin, it further achieves up to 2.0× in five-task mean throughput and 3.3× on MBPP+, respectively. These results demonstrate that BitMTP accelerates ultra-low-bit LLM generation across backbones, model sizes, and GPU types. The source code is released at https://github.com/PolyU-VCLab/BitMTP.

## 1 Introduction

The deployment of large language models (LLMs) is constrained by both model footprint and the serial decoding latency of autoregressive (AR) next-token prediction (NTP) [1–3]. To resolve this limit, ultra-low-bit models such as BitNet b1.58 reduce weight storage, memory traffic, and linearlayer execution cost through ternary weights [4–6], while multi-token prediction (MTP) methods target the sequential bottleneck by predicting multiple future tokens in one pass [7–13]. Despite their complementarity, direct combination of ultra-low-bit models and block-parallel MTP may not accelerate decoding at comparable quality: masked block prediction omits NTP’s intermediate token inputs, calling for adaptation to preserve quality, while wider computation and acceptance-dependent causal key–value (KV) updates can offset the latency savings from fewer sequential decoding steps.

To better illustrate the above-mentioned problem, we apply the block-parallel prediction mechanism [10] to BitNet b1.58-2B [5] and name the fine-tuned model ‘Naive BitNet+MTP’. On NVIDIA H20, we compare its five-task mean quality and decode throughput with the supervised fine-tuning (SFT) AR baseline of BitNet b1.58-2B. The left panel of Figure 1 shows that the naive combination improves throughput on BF16. However, we see in the right panel of Figure 1 that it loses both throughput and quality in the low-bit W2A8 setting (2-bit weights and 8-bit activations), which shows that fewer decoding steps alone cannot ensure faster low-bit generation. More generally, an MTP decoder’s throughput depends on iteration cost and the number of committed tokens. First, a block decoder must execute a wider computation and maintain causal KV states for continued generation. Second, iteration cost is amortized over committed tokens, rather than all predicted tokens. The accepted prefix and decoding policy therefore determine the committed-token count, pending causal KV work, and subsequent replay input. Verifier-based policies additionally require a verification replay. We call the resulting changes in replay inputs, causal KV work, and verifier invocation ‘accepted-prefix dynamics’. Efficiently executing these transitions turns parallel proposals into multi-token speedups on ultra-low-bit LLMs.

Average decode speed across 5 generation benchmarks (token/s)  
![](images/932665884bbc690b97f34ce58a54bbe0636607800d095a20526b2c298e644e45.jpg)

![](images/47e320627d1d65f7b13463a296bba7a26802f0056cb4f76efdfbbcee295b296f.jpg)  
Figure 1: Five-task mean generation quality (%) and decode throughput (tokens/s) for BitNet b1.58- 2B-SFT AR, Naive BitNet+MTP, and the proposed BitMTP on NVIDIA H20. The left panel shows the results by using BF16 reference execution, while the right panel uses low-bit W2A8 execution (2-bit weights and 8-bit activations). We see that the naive combination on BF16 improves throughput, but it loses both throughput and quality in the low-bit W2A8 setting. In contrast, our BitMTP substantially increases throughput while retaining task quality comparable to SFT AR.

To achieve faster decoding at comparable quality on ultra-low-bit LLM backbones, we propose BitMTP, an algorithm–system co-design framework. At the training level, a joint NTP+MTP objective equips the model with mask-conditioned block-causal drafting and an AR path. At the decoding level, confidence-only decoding, speculative-only decoding, and hybrid decoding provide different quality–throughput operating points without using separate draft and verifier models. At the runtime level, three mechanisms are used to reduce block computation and state-update costs. M1 specializes packed low-bit execution by replay width and uses role-dependent output projections. M2 uses a finite-state CUDA Graph atlas to replay reachable pending-prefix states over shared KV storage. M3 prepares verifier inputs before the host branch decision in hybrid decoding. Acceptedprefix alignment keeps the committed prefix, materialized causal KV boundary, and CUDA Graph replay state consistent across decoding and runtime. This supports captured-graph reuse over shared KV history as accepted prefixes change, reducing overhead of multi-token decoding. As shown in Figure 1, BitMTP achieves substantially higher decode throughput at comparable task quality to the SFT AR baseline. Our main contributions are summarized below:

• First end-to-end MTP framework for ultra-low-bit LLMs. To our knowledge, BitMTP is the first framework to co-design MTP training, decoding, and runtime execution for ultra-low-bit LLMs. Accepted-prefix alignment coordinates decoding decisions with causal KV materialization and graph replay. Captured graphs reuse shared causal KV history across changing accepted prefixes, reducing full-history copying overhead and supporting faster low-bit multi-token decoding.

• Joint training and flexible decoding. BitMTP combines a joint NTP+MTP objective with confidence-only decoding, speculative-only decoding, and hybrid decoding. It thereby supports an AR path and mask-conditioned block-causal drafting without a separate draft–verifier pair. The three decoding policies provide different quality–throughput operating points.

• Replayable runtime for dynamic MTP states. BitMTP introduces a three-part runtime that reconciles accepted-prefix-dependent execution with fixed-shape CUDA Graph replay. Widthspecialized low-bit execution, shared-KV graph replay, and hoisted verifier preparation control the kernel, memory, and scheduling overheads exposed by MTP on low-bit LLM backbones.

• Validation across models and hardware types. Experiments on BitNet b1.58-2B and Bonsai models from 1.7B to 4B demonstrate that BitMTP achieves mean decode speedups of up to 2.4× on H20 and 2.0× on RTX 3090. It also reaches a 3.3× MBPP+ speedup on Jetson Orin, demonstrating generalization across backbones, scales, and GPU types.

## 2 Related Work

Low-bit LLMs. Low-bit LLMs can be obtained by quantizing a pretrained full-precision model or pretraining directly in the low-bit regime. Post-training quantization applies quantization after full-precision training [14, 15], whereas quantization-aware training incorporates the target quantizer during model optimization [6, 16]. For example, Bonsai applies quantization-aware training to pretrained Qwen3 backbones to obtain ternary models [17]. BitNet and BitNet b1.58 instead use native low-bit pretraining [4, 5, 18]. Although low-bit representations reduce weight storage and linear-layer execution cost, token-by-token AR generation still requires a separate model pass for each output token, limiting further throughput gains. Our BitMTP addresses this sequential bottleneck by predicting blocks and committing variable-length continuations on the low-bit backbone.

Parallel drafting and blockwise generation. Recent methods reduce sequential decoding steps through auxiliary heads, learned drafters, or block-parallel prediction [7, 8, 11, 19]. Medusa adds decoding heads and verifies alternative continuations with tree attention [8]. EAGLE predicts future hidden states, EAGLE-2 builds context-dependent draft trees, and EAGLE-3 predicts tokens from features fused across target layers [7, 20, 21]. Training approaches include future-summary supervision [22], non-adjacent targets [23], and self-distillation [24]. DeepSeek-V3 incorporates MTP during pretraining [13]. Masked-input adaptation with gated LoRA and consistency losses exposes a pretrained model’s MTP potential [12]. Other methods reuse target-model components or adapt draft structures and budgets [9, 11, 25–27]. DFlash generates blocks with a target-featureconditioned diffusion drafter before speculative verification [28]. Sequential Diffusion Language Models (SDLM) predict masked token blocks and use confidence to commit consecutive prefixes [10]. In addition, QSpec [29] shares 4-bit weights and a KV cache between W4A4 drafting and W4A16 verification. ML-SpecQD [30] pairs an MXFP4 draft with a BF16 target and recursively accelerates drafting with a smaller model. QuantSpec [31] combines hierarchical KV quantization with optional 4-bit draft MLP weights. Although these quantized draft–verify methods combine quantization with speculative decoding to reduce drafting and memory costs, sequential proposal dependencies and recurring verification still constrain further acceleration. Our BitMTP combines block-parallel proposals, conditional verification in hybrid decoding, and accepted-prefix-aligned low-bit execution to reduce work per committed token.

Inference runtimes for efficient generation. The CPU inference system of BitNet provides specialized kernels for ternary execution [32]. GPU systems such as Atom and QServe co-design low-bit quantization and serving optimizations [33, 34]. Other work optimizes attention and decode kernels [1, 35–37]. KV-cache techniques address allocation, compression, and quantization [38–42]. Serving systems improve request scheduling and execution [43–46]. For block generation on an ultralow-bit LLM, the accepted prefix and verification branch determine causal KV materialization and the next replay input. Kernel, cache, and scheduling optimizations alone do not provide a coordinated end-to-end acceleration path for these transitions. Our proposed BitMTP provides this path through width-specialized packed operators, captured graphs over shared KV storage, and preparation of verifier inputs before the hybrid branch decision.

## 3 The BitMTP Framework

## 3.1 Method Overview

As illustrated in Figure 2, BitMTP enables low-bit multi-token generation through coordinated training, decoding, and runtime components. Joint training equips the model with both AR generation and mask-conditioned block-causal drafting. The decoding component uses confidence-only decoding, speculative-only decoding, or hybrid decoding to select the committed prefix. The runtime executes the selected transition with fixed-shape low-bit CUDA Graph replay. These components satisfy the same consistency requirement: the committed prefix, materialized causal KV boundary, and

![](images/c4cd10bcd7fca2725024e24cca467a1bdff74ac1951178ca4c319eb82277d28e.jpg)  
Figure 2: The BitMTP framework. (a) Joint NTP+MTP training supports mask-conditioned blockcausal drafting and an AR path in one model. (b) Confidence-only decoding commits a confidenceselected prefix. Speculative-only decoding verifies each block, while hybrid decoding verifies when the confidence prefix is too short. Verifier-based steps add a correction or bonus token. (c) The runtime executes accepted-prefix transitions with width-specialized low-bit operators and captured CUDA Graphs. Graphs reuse causal KV history through shared storage. In hybrid decoding, verifier inputs are prepared before the host branch decision.

CUDA Graph replay state must remain mutually consistent among decoding iterations, which we call ‘accepted-prefix alignment’.

Each draft block proposes $D$ future tokens. At iteration i, decoding tracks $q _ { i } = ( p _ { i } , \ell _ { i } , u _ { i } )$ : the committed prefix $p _ { i } .$ , its materialized causal KV length $\ell _ { i } ,$ and pending suffix $u _ { i } .$ . The prefix includes the prompt and emitted tokens. Accepted-prefix alignment imposes the following length relation:

$$
| p _ { i } | = \ell _ { i } + n _ { i } ,\tag{1}
$$

where $n _ { i } = | u _ { i } |$ counts committed tokens awaiting causal KV. During initialization, prefill processes the prompt and builds its causal KV prefix. Since tokens emitted from the initialization logits have not yet been processed as causal inputs, they form the pending suffix. Later, $u _ { i }$ will contain confidence-committed drafts or a verifier-origin token, and the next draft replay computes their causal KV together with a new block.

The runtime maps $q _ { i }$ and the selected decoding policy to a replay state $s _ { i } = ( M _ { i } , \sigma _ { i } , r _ { i } )$ . The width $M _ { i }$ covers the pending suffix and the next block computation. The segment $\sigma _ { i }$ selects the active KV capacity, and $r _ { i }$ selects the numerical execution role. A verifier step additionally selects a width- $( D { + } \mathrm { 1 } )$ replay with the verifier role. Each draft step computes and materializes the pending-token KV in $q _ { i } ;$ the next state records committed tokens. Algorithm 1 in Appendix A summarizes one hybrid decoding iteration.

## 3.2 Block-Causal Drafting and Training

For a training sequence y anchored at $s _ { b } .$ , BitMTP constructs the block input and targets:

$$
X _ { b } = [ y _ { s _ { b } } , m _ { \mathrm { m a s k } } , \ldots , m _ { \mathrm { m a s k } } ] , \qquad Y _ { b } = [ y _ { s _ { b } + 1 } , \ldots , y _ { s _ { b } + D } ] ,\tag{2}
$$

where $y$ includes the prompt and the response, and $m _ { \mathrm { m a s k } }$ denotes the mask input. The observed anchor occupies the first input slot, and mask embeddings occupy the remaining D−1 slots. The targets in $Y _ { b }$ are the D ordered future tokens. Attention within the block is causal, as illustrated in Figure 2(a). Each prediction can attend to the anchor and the mask slots preceding its target position. Because the block inputs are constructed before the forward pass, the model produces the whole draft block in parallel.

We retain causal next-token supervision and combine it with block prediction. The summed losses are then normalized over all supervised targets:

$$
\mathcal { L } _ { \mathrm { m t p } } = \sum _ { b \in \mathcal { B } } \sum _ { u \in \mathcal { U } _ { b } } - \log p _ { \theta } \big ( y _ { s _ { b } + u } \mid y _ { < s _ { b } } , X _ { b , 1 : u } \big ) , \qquad \mathcal { L } = \frac { \mathcal { L } _ { \mathrm { a r } } + \mathcal { L } _ { \mathrm { m t p } } } { N _ { \mathrm { a r } } + N _ { \mathrm { m t p } } } ,\tag{3}
$$

where θ denotes the model parameters. The set B contains training blocks, and $\mathcal { U } _ { b } \subseteq \{ 1 , \dots , D \}$ contains the valid supervised slots of block b. The slice $X _ { b , 1 : u }$ comprises its first u input slots. The AR loss $\mathcal { L } _ { \mathrm { a r } }$ sums causal next-token negative log-likelihoods, while $\bar { N } _ { \mathrm { a r } }$ and $N _ { \mathrm { m t p } }$ count supervised AR and block targets. The AR term supports causal generation, while the MTP term learns block-causal proposals. The resulting model performs both drafting and AR verification.

## 3.3 Decoding Policies

BitMTP provides confidence-only decoding, speculative-only decoding, and hybrid decoding, as shown in Figure 2(b). Each policy operates on a draft block $d _ { 1 : D }$ and determines the accepted draft length $\hat { a } _ { i }$ . A confidence step forms an output block of $\hat { a } _ { i }$ draft tokens. A verifier step adds one verifier-origin token, giving $\hat { a } _ { i } + 1$ tokens in the block.

Confidence-only decoding. With draft-token softmax probability $c _ { u }$ and threshold $\tau ,$ , the decoder always commits $d _ { 1 }$ and extends the prefix through consecutive confident drafts:

$$
h _ { i } = 1 + \sum _ { u = 2 } ^ { D } \prod _ { v = 2 } ^ { u } \mathbf { 1 } [ c _ { v } > \tau ] .\tag{4}
$$

The $h _ { i }$ -token confidence streak is committed without verification, giving $\hat { a } _ { i } = h _ { i }$

Speculative-only decoding. The same model processes [anchor, $d _ { 1 } , \ldots , d _ { D } ]$ under an AR-causal mask, where the anchor is the last committed token. It produces verifier tokens $\tilde { d } _ { 1 : D + 1 }$ . Verification selects the longest matching prefix:

$$
\hat { a } _ { i } = \operatorname* { m a x } \left\{ k \in \left\{ 0 , \ldots , D \right\} : d _ { u } = \tilde { d } _ { u } \mathrm { f o r } \mathrm { e v e r y } u \leq k \right\} .\tag{5}
$$

The decoder forms an output block containing $d _ { 1 : \hat { a } _ { i } }$ , followed by $b _ { i } = \tilde { d } _ { \hat { a } _ { i } + 1 }$ . The verifier-origin token $b _ { i }$ corrects the first rejected draft position or supplies a bonus token after full acceptance. Speculative-only decoding preserves the greedy continuation defined by the AR-causal verifier.

Hybrid decoding. Hybrid decoding uses a minimum confidence-prefix length m. It sets $\hat { a } _ { i } = h _ { i }$ and commits the confidence prefix when $h _ { i } \geq m$ . Otherwise, it invokes the AR-causal verifier to determine the accepted prefix and verifier-origin token.

## 3.4 Replayable Low-Bit Runtime

Figure 3 shows how M1–M3 implement a decoding transition. At each step, BitMTP selects a replay state $s _ { i } = ( M _ { i } , \sigma _ { i } , r _ { i } )$ specifying the replay width, KV-capacity segment, and numerical role. M1 selects the packed operator and output projection, M2 replays the corresponding graph over shared causal history, and M3 moves verifier-input preparation ahead of the hybrid branch decision.

After a draft replay, the runtime copies the $n _ { i }$ pending-token KV rows into history. The selected decoding policy then updates the state as follows:

$$
\begin{array} { r l } & { \bar { \ell } _ { i } = \ell _ { i } + n _ { i } , } \\ & { q _ { i + 1 } = \left\{ \begin{array} { l l } { \big ( p _ { i } \cdot d _ { 1 : \hat { a } _ { i } } , \bar { \ell } _ { i } , d _ { 1 : \hat { a } _ { i } } \big ) , } & { I _ { \mathrm { v e r i f y } } ( i ) = 0 , } \\ { \big ( p _ { i } \cdot d _ { 1 : \hat { a } _ { i } } \cdot b _ { i } , \bar { \ell } _ { i } + \hat { a } _ { i } , [ b _ { i } ] \big ) , } & { I _ { \mathrm { v e r i f y } } ( i ) = 1 , } \end{array} \right. } \end{array}\tag{6}
$$

where $\bar { \ell } _ { i }$ is the materialized KV length after processing the pending suffix, and $b _ { i }$ is the verifier-origin token from Section 3.3. The indicator $I _ { \mathrm { v e r i f y } } ( i )$ is 0 for confidence-only decoding, 1 for speculativeonly decoding, and $\lfloor \left[ h _ { i } < m \right]$ for hybrid decoding. Without verification, the committed draft prefix remains pending. With verification, its accepted KV rows are materialized and only $b _ { i }$ remains pending. Both branches preserve $| p _ { i + 1 } | = \ell _ { i + 1 } + | u _ { i + 1 } |$

![](images/e49bc87d4210f881c8dd90ba69c989fef5e70c3b91718df7f15c42ba563a8110.jpg)  
Figure 3: Runtime mechanisms M1–M3. M1 selects packed operators by replay width M and input/output dimensions $( K , N )$ , while numerical role r selects the output projection $\pi ( r )$ . M2 indexes captured graphs by $( M , \sigma , r )$ and shares one KV buffer $B _ { \sigma }$ across widths and roles within each capacity segment. M3 moves verifier-input preparation ahead of the hybrid branch decision.

M1: Width-specialized low-bit execution with role-dependent output projections. M1 specializes packed low-bit execution for the replay widths used by AR generation, block drafting, and verification. As shown in the top-left panel of Figure 3, M1 selects a packed operator by $( M , K , { \bar { N } } )$ and an output projection by numerical role r. The dispatch is:

$$
\begin{array} { r } { \kappa : ( M , K , N ) \longmapsto K _ { M , K , N } , \qquad \pi ( r ) = \left\{ \begin{array} { l l } { \pi _ { \mathrm { f a s t } } , } & { r \in { \mathcal R } _ { \mathrm { f a s t } } , } \\ { \pi _ { \mathrm { c r i t } } , } & { r \in { \mathcal R } _ { \mathrm { c r i t } } \cup { \mathcal R } _ { \mathrm { v e r } } } \end{array} \right. , } \end{array}\tag{7}
$$

where $\kappa _ { M , K , N }$ is the selected packed operator, and $K , N$ are its input and output feature dimensions. The set $\mathcal { R } _ { \mathrm { f a s t } }$ contains fast drafting roles, $\mathcal { R } _ { \mathrm { c r i t } }$ contains verification-critical drafting and AR roles, and ${ \mathcal { R } } _ { \mathrm { v e r } }$ contains verifier roles. Drafting selects its projection through $r ,$ while AR generation and verification use $\pi _ { \mathrm { c r i t } }$ . All roles share the packed low-bit transformer body and project only the token rows required by the replay.

M2: Finite-state CUDA Graph atlas with shared KV storage. Accepted-prefix dynamics change both replay width and the materialized KV boundary, whereas CUDA Graph replay requires fixed shapes and addresses. As shown in Figure 3, M2 supports these changes with shared segment storage and captures a graph set $\mathcal { G }$ covering reachable transitions from a candidate graph set:

$$
\mathcal { G } _ { \mathrm { c a n d } } = \{ g _ { M , \sigma , r } : M \in \mathcal { M } , \ \sigma \in \Sigma , \ r \in \mathcal { R } \} , \ \mathrm { ~ } \mathcal { G } \subseteq \mathcal { G } _ { \mathrm { c a n d } } ,\tag{8}
$$

where $g _ { M , \sigma , r }$ denotes a graph with replay key $( M , \sigma , r )$ . The sets ${ \mathcal { M } } , { \Sigma } ,$ , and R contain supported replay widths, ordered KV-capacity segments, and M1 execution roles, respectively. The captured set covers reachable replay states, including width-(D+1) verification.

Within segment σ, all graphs use fixed-shape views of one physical KV buffer $B _ { \sigma }$ , with $| \sigma |$ history slots and $M _ { \mathrm { m a x } } =$ max M workspace slots. Width and role switches therefore preserve materialized history. After drafting, the runtime copies the pending-token KV rows into history; after verification, it promotes accepted draft rows, excluding the repeated anchor. These operations copy consecutive token rows within the segment. When a transition requires a larger capacity segment, the runtime copies the materialized history into the new buffer. Graphs within each segment thus reuse one causal KV history while implementing Eq. 6.

Shared KV storage is proportional to $\textstyle \sum _ { \sigma \in \Sigma } ( | \sigma | + M _ { \mathrm { m a x } } )$ for a fixed model. With fixed replay widths, geometrically increasing capacities cover history length $\ell _ { \mathrm { m a x } }$ using $O ( \log \ell _ { \mathrm { m a x } } )$ segments and $O ( \bar { \ell _ { \mathrm { m a x } } } )$ KV slots. Appendix A details the replay masks and KV copies.

M3: Hoisted verifier-input preparation. M3 moves verifier-input preparation ahead of the branch decision in hybrid decoding, as shown in Figure 3. The draft block, committed anchor, and materialized causal KV boundary determine the verifier inputs before the host reads $h _ { i }$ . After drafting, the runtime enqueues preparation of the width-(D+1) verifier token buffer, positions, and AR-causal mask on the GPU stream, then obtains $h _ { i }$ on the host. Preparation is issued in each hybrid iteration, while verifier replay occurs only when $h _ { i } < m$ . The selected verifier graph therefore consumes the prepared inputs without a separate input-preparation stage after the branch decision.

## 4 Experimental Evaluation

## 4.1 Experimental Setup

Models, training, and decoding. We evaluate three ultra-low-bit LLM backbones: BitNet b1.58- 2B [5], Bonsai-1.7B, and Bonsai-4B [17]. For each backbone, additional supervised fine-tuning produces the SFT model used as the AR baseline. The SFT and MTP variants use the same processed corpus, with MTP using the joint objective in Eq. 3. Generation uses greedy decoding. Primary confidence-only decoding and hybrid decoding use τ=0.9. Hybrid decoding uses the model’s ARcausal verifier to determine the accepted prefix when verification is required. Appendix B summarizes the training corpus and reports one-epoch training cost on Bonsai-1.7B.

Hardware and low-bit backends. Server, consumer, and edge tests use NVIDIA H20, RTX 3090, and Jetson Orin NX, respectively. The PyTorch/CUDA runtime uses packed W2A8 for decode-stage transformer linear layers, with INT2 weights for BitNet and group-128 ternary weights for Bonsai. H20 enables FP8 draft projections for the fast drafting roles in M1. The FP8 GEMM backend does not support the Ampere GPUs in RTX 3090 and Orin, so these platforms use non-FP8 draft projections. Orin runs under MAXN\_SUPER with locked clocks. AR generation and verifier replay use verification-critical projections. Other baseline systems use their released runtimes.

Generation benchmarks. The generation suite covers IFEval for instruction following, GSM8K and MATH-500 for reasoning, and HumanEval+ and MBPP+ for code generation. We report strict instruction-following accuracy, exact match for reasoning, and EvalPlus pass@1 for code.

Metrics and model capability evaluation. Decode throughput divides timed output tokens by decodeloop time measured after prefill and request initialization. Task means are unweighted across the five generation benchmarks, and speedups use the matched SFT AR route on the same device and backend. The eleven-task capability evaluation uses lm-eval with full benchmark sets and standard prompts.

## 4.2 Generation Quality and Throughput

Table 1 compares compact BF16 models [47–49], quantized and low-bit AR models, and learned draft–verify systems with BitMTP on H20. The matched SFT AR routes establish the acceleration obtained beyond packed low-bit execution. At D=8, hybrid decoding achieves five-task mean speedups of 2.2× on BitNet, 2.4× on Bonsai-1.7B, and 2.1× on Bonsai-4B. The Bonsai mean scores remain close to SFT: 47.91 versus 47.36 and 63.43 versus 63.55. The compact-model comparisons further place this quality in context. With hybrid decoding, BitMTP–Bonsai-1.7B approaches BF16 Qwen2.5-1.5B’s 49.23 and exceeds its AWQ variant’s 41.79, while BitMTP–Bonsai-4B exceeds Llama-3.2-3B’s 57.41 and Qwen3-1.7B’s 58.23. BitMTP therefore supports multi-token acceleration on both smaller deployment models and stronger backbones.

The EAGLE-3 [21] comparison examines learned speculative drafting with a packed low-bit target. Bonsai+EAGLE-3 reports 566.23 tokens/s at 47.65 mean quality, whereas BitMTP hybrid decoding at D=8 reaches 1110.65 tokens/s at 47.91. BitMTP combines block proposals and AR verification in one low-bit backbone rather than adding a BF16 drafter. Hybrid decoding exceeds confidence-only decoding in both mean quality and throughput across all five model–block-size configurations in Table 1. On Bonsai-1.7B at D=8, hybrid decoding and speculative-only decoding score 47.91 and 47.89, while hybrid throughput increases from 1028.13 to 1110.65 tokens/s. BitNet offers a different choice when quality retention is prioritized: D=4 speculative-only decoding scores 45.17 against SFT’s 45.95 with 1.7× throughput, whereas D=8 hybrid decoding reaches 2.2× at 43.25. On Bonsai-4B, speculative-only decoding achieves a mean score of 64.13, compared with 63.55 for SFT, while retaining a 1.9× throughput speedup.

Increasing D from 4 to 8 raises hybrid mean throughput from 722.12 to 805.57 tokens/s on BitNet and from 952.18 to 1110.65 on Bonsai-1.7B. Useful progress depends on the committed prefix rather than the proposed width alone. For Bonsai-1.7B hybrid decoding at D=8, BitMTP obtains 2.7× and

Table 1: Generation quality (Sc., %) and decode throughput (Rate, tokens/s) on NVIDIA H20. Dark-red values in parentheses give speedups over AR baselines. Average is the unweighted five-task mean. W/A denotes weight/activation precision. Conf., Spec., and Hybr. denote confidence-only, speculative-only, and hybrid decoding, respectively.
<table><tr><td rowspan="2">Model</td><td rowspan="2">W/A</td><td rowspan="2">Type</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">IFEval HumanEval+</td><td colspan="2">MBPP+</td><td colspan="2">Average</td></tr><tr><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc. Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td></tr><tr><td colspan="10">Compact, quantized, and speculative baselines</td><td></td><td></td><td></td></tr><tr><td>MiniCPM-2B</td><td>BF16/BF16</td><td>AR</td><td colspan="3">47.84 273.83 11.00</td><td colspan="3">43.65 306.92 40.24</td><td>43.65</td><td>252.08</td><td>37.28</td><td>284.82</td></tr><tr><td>Qwen2.5-1.5B</td><td>BF16/BF16</td><td>AR</td><td colspan="3">301.18 57.54 441.3849.00 470.88</td><td colspan="3">51.08 467.00 37.20</td><td>51.32</td><td>395.61</td><td>49.23</td><td>442.88</td></tr><tr><td>Llama-3.2-3B</td><td>BF16/BF16</td><td>AR</td><td colspan="3">66.79 274.71 45.80 303.84</td><td colspan="3">78.90 303.35 44.51</td><td>51.06</td><td>256.08</td><td>57.41</td><td>284.59</td></tr><tr><td>Qwen3-1.7B</td><td>BF16/BF16</td><td>AR</td><td colspan="3">68.16452.3867.40 454.19</td><td>75.30 457.90</td><td colspan="3">37.20 456.47</td><td>43.12 464.42</td><td>58.23</td><td>457.07</td></tr><tr><td>Qwen2.5-1.5B-AWQ</td><td>INT4/FP16</td><td>AR</td><td colspan="3">50.49266.26 37.80</td><td colspan="3">47.12 270.05</td><td>268.86 43.65</td><td>267.39</td><td>41.79</td><td>268.24</td></tr><tr><td>Llama3-8B-1.58†</td><td>1.58/BF16</td><td>AR</td><td colspan="3">3.94 13.77 1.00</td><td>30.70</td><td colspan="3">29.88 13.83 0.00</td><td>0.79 14.03</td><td>7.29</td><td>13.90</td></tr><tr><td>Falcon3-7B-1.58bit† BitCPM4-1B†</td><td>1.58/BF16</td><td>AR</td><td colspan="3">1.67 14.99 18.40</td><td>43.76</td><td colspan="3">13.84 35.98 17.07</td><td>23.02 15.71</td><td>24.56</td><td>14.95</td></tr><tr><td></td><td>1.58/BF16</td><td>AR 41.85</td><td colspan="3">26.66 29.80</td><td colspan="3">72.18 29.32</td><td>15.24 27.95 52.12</td><td>27.88</td><td>42.60</td><td>26.92</td></tr><tr><td></td><td colspan="10">BitNet b1.58-2B family: packed W2A8</td><td></td><td></td></tr><tr><td>BitNet b1.58-2B BitNet b1.58-2B-SFT</td><td colspan="10">1.58/INT8 AR 1.58/INT8</td><td>378.73</td><td>43.37 366.38 45.95 373.42</td></tr><tr><td></td><td rowspan="9"></td><td>AR</td><td colspan="10">58.00364.97 35.80 364.97 55.76 368.73 54.80</td></tr><tr><td>1.58/INT8</td><td>MTP 61.11 (Conf.)</td><td colspan="10">691.59 32.80 614.13 (1.9×) (1.7×)</td></tr><tr><td>BitMTP-BitNet b1.58-2B</td><td>MTP 65.05</td><td colspan="10">662.78 37.20 640.38</td></tr><tr><td>(D=4)</td><td>(Spec.)</td><td colspan="10">(1.8×) (1.8×)</td></tr><tr><td></td><td>MTP 63.08</td><td colspan="10">761.42 232.60 707.11</td></tr><tr><td></td><td>(Hybr.)</td><td colspan="10">(2.1×) (1.9×)</td></tr><tr><td rowspan="5">BitMTP–BitNet b1.58-2B (D=8)</td><td>MTP</td><td colspan="10">56.48647.4929.60 539.27 (1.5×)</td></tr><tr><td>(Conf.)</td><td colspan="10">(1.8×)</td></tr><tr><td>MTP</td><td colspan="10">60.35 718.60 32.60</td></tr><tr><td>1.58/INT8 (Spec.)</td><td colspan="10">(2.0×) 33.60</td></tr><tr><td>MTP</td><td colspan="10">60.35 819.42 (2.2×)</td></tr><tr><td>(Hybr.)</td><td colspan="10">Bonsai-1.7B family: packed W2A8</td></tr><tr><td colspan="10"></td></tr><tr><td>Bonsai-1.7B Bonsai-1.7B-SFT</td><td>1.58/INT8 1.58/INT8</td><td>AR AR</td><td colspan="10">64.14 469.36 46.20 449.62 78.78465.6341.46 59.67 472.99 41.20 447.08 68.59</td></tr><tr><td>Bonsai-1.7B + EAGLE-3 W2A8+BF16</td><td></td><td>MTP</td><td colspan="10">58.98 515.55 40.80 537.35 68.71 (1.1×) (1.2×)</td></tr><tr><td colspan="10"></td></tr><tr><td rowspan="6">BitMTP-Bonsai-1.7B (D=4)</td><td rowspan="7">1.58/INT8</td><td>MTP</td><td colspan="10">54.89 701.25 41.80</td></tr><tr><td>(Conf.)</td><td>(1.5×)</td><td>837.48 (1.9×)</td><td>64.75 648.91 32.93 (1.4×)</td><td>943.90 (2.0×)</td><td>36.51</td><td>847.30 (1.8×)</td><td>46.17</td><td>795.77 (1.7×)</td></tr><tr><td>MTP</td><td>57.85 795.8044.20 (1.7×)</td><td>853.38</td><td>66.43 714.93 32.93</td><td></td><td>888.55</td><td>38.89 867.85</td><td>48.06</td><td>824.10</td></tr><tr><td>(Spec.) MTP</td><td>57.47 901.35 42.20</td><td>(1.9×)</td><td>(1.5×)</td><td></td><td>(1.9×)</td><td></td><td>(1.8×)</td><td>(1.8×)</td></tr><tr><td>(Hybr.)</td><td>(1.9×)</td><td>979.79</td><td>65.35791.36 32.321078.32 38.891010.11 47.24</td><td></td><td></td><td></td><td>(2.1×)</td><td>952.18</td></tr><tr><td></td><td></td><td>(2.2×)</td><td>(1.7×)</td><td></td><td>(2.3×)</td><td></td><td></td><td>(2.0×)</td></tr><tr><td>MTP (Conf.)</td><td>54.21 661.80 39.60 (1.4×)</td><td>837.24 (1.9×)</td><td>66.19 (1.3×)</td><td>593.09 34.15</td><td>1010.5935.71 (2.1×)</td><td></td><td>(2.2×)</td><td>1040.99 45.97 828.74</td></tr><tr><td rowspan="6">BitMTP-Bonsai-1.7B (D=8)</td><td rowspan="6">1.58/INT8</td><td></td><td colspan="10">57.70907.2041.60</td></tr><tr><td>MTP</td><td></td><td>1051.18 68.11</td><td>760.12 34.761184.8937.301237.2547.891</td><td></td><td></td><td></td><td></td><td>(1.8×) 1028.13</td></tr><tr><td>(Spec.)</td><td>(1.9×)</td><td>(2.4×)</td><td>(1.6×)</td><td></td><td>(2.5×)</td><td></td><td>(2.6×)</td><td>(2.2×)</td></tr><tr><td>MTP (Hybr.)</td><td>57.47 919.54 43.20 (1.9×)</td><td>1112.6868.59</td><td></td><td>801.4233.541288.3336.77</td><td></td><td></td><td></td><td>1431.2847.911110.65 (2.4×)</td></tr><tr><td></td><td></td><td></td><td>(2.5×)</td><td>(1.7×)</td><td>(2.7×)</td><td></td><td>(3.0×)</td><td></td></tr><tr><td>Bonsai-4B family: packed W2A8</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">Bonsai-4B-SFT 1.58/INT8 AR 79.08278.04 61.20 MTP 76.35 347.76 55.40</td></tr><tr><td rowspan="6"></table>

<sup>†</sup>BF16 reference execution.

3.0× speedups on HumanEval+ and MBPP+, and a 1.7× speedup on IFEval. These runs commit approximately 6.7 and 7.1 tokens per iteration on the code tasks, versus 4.6 on IFEval. This pattern is consistent with amortizing block computation and state updates over more committed tokens, connecting workload-dependent acceleration to the execution state managed by BitMTP. With D=8 hybrid decoding, both Bonsai-1.7B and Bonsai-4B achieve more than twice the five-task mean throughput of their respective SFT baselines while maintaining comparable mean task quality. These results show that BitMTP retains substantial acceleration as the backbone scales from 1.7B to 4B parameters.

Table 2: Five-task quality (Sc., %) and decode throughput (Rate, tokens/s) of hybrid decoding on NVIDIA RTX 3090. Parentheses give speedups over SFT AR; Average is the unweighted task mean.
<table><tr><td rowspan="2">Configuration</td><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">IFEval</td><td colspan="2">HumanEval+</td><td colspan="2">MBPP+</td><td colspan="2">Average</td></tr><tr><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td><td>Sc.</td><td>Rate</td></tr><tr><td>BitNet b1.58-2B-SFT 58.38</td><td></td><td>275.54</td><td>35.00</td><td>281.68</td><td>56.35</td><td>277.36</td><td>39.02</td><td>284.19</td><td>40.74</td><td>282.94</td><td>45.90</td><td>280.34</td></tr><tr><td>BitMTP (D=4) BitMTP (D=8)</td><td>62.77 510.62 (1.9×) 59.89 524.16 (1.9×)</td><td></td><td>35.00 33.00</td><td>492.56 (1.7×) 515.08 (1.8×)</td><td>54.08 55.76</td><td>392.57 (1.4×) 364.84 (1.3×)</td><td>35.37 35.37</td><td>572.04 (2.0×) 661.16 (2.3×)</td><td>33.60 33.33</td><td>553.23 (2.0×) 626.82 (2.2×)</td><td>43.47</td><td>44.16 504.20 (1.8×) 538.41 (1.9×)</td></tr><tr><td>Bonsai-1.7B-SFT</td><td>61.18</td><td>244.51</td><td>42.00</td><td>245.31</td><td>68.94</td><td>244.81</td><td>35.37</td><td>245.08</td><td></td><td></td><td></td><td></td></tr><tr><td>BitMTP (D=4)</td><td>57.54 430.60 (1.8×)42.60 472.35 (1.9×) 67.39376.04(1.5×)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.15516.57 (2.1×)37.57474.59(1.9×)47.85454.03 (1.9×)</td><td>36.51</td><td>244.64</td><td>48.80</td><td>244.87</td></tr><tr><td>BitMTP (D=8)</td><td>58.00416.82 (1.7×)42.40507.39 (2.1×)66.91346.49(1.4×)31.71580.83 (2.4×)37.83639.76 (2.6×)47.37498.26 (2.0×)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 3: MBPP+ decode throughput (tokens/s) on Jetson Orin. Parentheses give speedups over SFT AR; decoding-policy abbreviations follow Table 1.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">SFT AR</td><td colspan="3">BitMTP (D=4)</td><td colspan="3">BitMTP (D=8)</td></tr><tr><td>Conf.</td><td>Spec.</td><td>Hybr.</td><td>Conf.</td><td>Spec.</td><td>Hybr.</td></tr><tr><td>BitNet b1.58-2B</td><td>19.35 (1.0×)</td><td>35.38 (1.8×)</td><td>34.38 (1.8×)</td><td>40.92 (2.1×)</td><td>44.74 (2.3×)</td><td>54.55 (2.8×)</td><td>61.76 (3.2×)</td></tr><tr><td>Bonsai-1.7B</td><td>15.95 (1.0×)</td><td>30.80 (1.9×)</td><td>28.83 (1.8×)</td><td>36.20 (2.3×)</td><td>40.42 (2.5×)</td><td>42.47 (2.7×)</td><td>52.05 (3.3×)</td></tr></table>

Table 4: Eleven-task model capability scores (%). Avg. denotes the unweighted mean; TQA is TruthfulQA-mc2. W/A denotes weight/activation precision. TriviaQA scores generated answers; the remaining tasks use likelihood scoring.
<table><tr><td>Model</td><td>W/A</td><td>MMLU</td><td>Wino</td><td>Hella</td><td>ARC-C</td><td>ARC-E</td><td>BoolQ</td><td>OBQA</td><td>PIQA</td><td>Trivia</td><td>CSQA TQA</td><td></td><td>Avg.</td></tr><tr><td>MiniCPM-2B</td><td>BF16/BF16</td><td>52.4</td><td>65.5</td><td>71.4</td><td>43.8</td><td>68.9</td><td>77.6</td><td>42.6</td><td>76.1</td><td>34.2</td><td>73.1</td><td>43.1</td><td>59.0</td></tr><tr><td>Qwen2.5-1.5B</td><td>BF16/BF16</td><td>60.2</td><td>63.4</td><td>68.3</td><td>46.6</td><td>75.9</td><td>78.1</td><td>40.4</td><td>75.9</td><td>39.0</td><td>76.5</td><td>46.7</td><td>61.0</td></tr><tr><td>Llama-3.2-3B</td><td>BF16/BF16</td><td>60.7</td><td>68.6</td><td>71.6</td><td>46.1</td><td>71.0</td><td>78.8</td><td>39.2</td><td>76.9</td><td>58.6</td><td>72.5</td><td>51.5</td><td>63.2</td></tr><tr><td>Qwen2.5-1.5B-AWQ</td><td>INT4/FP16</td><td>57.5</td><td>63.5</td><td>66.4</td><td>44.3</td><td>72.5</td><td>74.4</td><td>40.4</td><td>75.6</td><td>33.9</td><td>74.0</td><td>45.4</td><td>58.9</td></tr><tr><td>Llama3-8B-1.58†</td><td>1.58/BF16</td><td>35.2</td><td>59.2</td><td>68.7</td><td>43.4</td><td>69.8</td><td>68.6</td><td>39.4</td><td>75.9</td><td>20.0</td><td>30.0</td><td>39.1</td><td>49.9</td></tr><tr><td>Falcon3-7B-1.58bit†</td><td>1.58/BF16</td><td>43.3</td><td>60.1</td><td>59.5</td><td>37.6</td><td>64.8</td><td>72.3</td><td>38.6</td><td>72.8</td><td>0.0</td><td>67.3</td><td>43.2</td><td>50.9</td></tr><tr><td>BitCPM4-1B†</td><td>1.58/BF16</td><td>55.2</td><td>55.8</td><td>52.6</td><td>38.9</td><td>63.0</td><td>70.4</td><td>36.6</td><td>69.1</td><td>15.4</td><td>55.0</td><td>42.5</td><td>50.4</td></tr><tr><td colspan="10">BitNet b1.58-2B family</td><td></td><td></td><td></td><td></td></tr><tr><td>BitNet b1.58-2B</td><td>1.58/INT8</td><td>53.0</td><td>72.1</td><td>67.6</td><td>48.1</td><td>73.7</td><td>79.2</td><td>40.0</td><td>76.6</td><td>33.7</td><td>71.8</td><td>46.4</td><td>60.2</td></tr><tr><td>BitNet b1.58-2B-SFT</td><td>1.58/INT8</td><td>53.2</td><td>69.6</td><td>67.8</td><td>50.6</td><td>76.1</td><td>79.5</td><td>42.4</td><td>78.1</td><td>33.6</td><td>69.6</td><td>39.8</td><td>60.0</td></tr><tr><td>BitMTP (D=4)</td><td>1.58/INT8</td><td>52.9</td><td>70.2</td><td>67.2</td><td>48.8</td><td>75.1</td><td>79.1</td><td>41.8</td><td>78.0</td><td>33.0</td><td>70.3</td><td>41.7</td><td>59.8</td></tr><tr><td>BitMTP (D=8)</td><td>1.58/INT8</td><td>52.0</td><td>70.0</td><td>66.0</td><td>48.6</td><td>76.5</td><td>78.7</td><td>39.8</td><td>77.2</td><td>29.8</td><td>69.4</td><td>42.8</td><td>59.1</td></tr><tr><td colspan="10">Bonsai-1.7B family</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bonsai-1.7B</td><td>1.58/INT8</td><td>49.2</td><td>59.1</td><td>52.4</td><td>38.5</td><td>65.2</td><td>72.4</td><td>33.8</td><td>70.1</td><td>19.8</td><td>63.4</td><td>46.9</td><td>51.9</td></tr><tr><td>Bonsai-1.7B-SFT</td><td>1.58/INT8</td><td>46.1</td><td>58.0</td><td>53.8</td><td>37.8</td><td>64.1</td><td>71.7</td><td>33.6</td><td>69.4</td><td>19.2</td><td>60.4</td><td>43.4</td><td>50.7</td></tr><tr><td>BitMTP (D=4)</td><td>1.58/INT8</td><td>43.0</td><td>55.7</td><td>52.3</td><td>35.6</td><td>64.5</td><td>70.4</td><td>32.2</td><td>70.6</td><td>17.7</td><td>56.7</td><td>42.7</td><td>49.2</td></tr><tr><td>BitMTP (D=8)</td><td>1.58/INT8</td><td>44.0</td><td>56.7</td><td>52.3</td><td>36.2</td><td>63.8</td><td>74.4</td><td>33.8</td><td>70.2</td><td>17.5</td><td>59.9</td><td>41.8</td><td>50.0</td></tr></table>

<sup>†</sup>BF16 reference execution.

Tables 2 and 3 extend the evaluation to consumer and edge GPUs. On RTX 3090, D=8 hybrid decoding achieves five-task mean speedups of 1.9× on BitNet and 2.0× on Bonsai-1.7B. On Jetson Orin’s MBPP+ workload, BitNet throughput increases from 19.35 to 61.76 tokens/s and Bonsai from 15.95 to 52.05, giving 3.2× and 3.3× speedups. All three policies accelerate MBPP+ over SFT AR on Orin, with hybrid decoding the fastest at both block sizes. Together, these results demonstrate effective multi-token acceleration across server, consumer, and edge GPUs, showing that the benefits of BitMTP extend beyond a single hardware class.

## 4.3 Model Capability Retention

Table 4 compares BitMTP with the SFT counterpart of each backbone to assess retention of knowledge, reasoning, and commonsense capabilities beyond the generation benchmarks. Ten tasks assess the adapted backbone through causal likelihood scoring, without invoking multi-token acceptance policies; TriviaQA evaluates generated answers by exact match. Relative to SFT, BitNet’s eleven-task means remain close at 59.8 and 59.1 for D=4 and $D { = } 8 ,$ compared with 60.0. Bonsai-1.7B scores 49.2 and 50.0, compared with 50.7 for SFT. The block-size ordering differs across backbones: D=4 has the higher eleven-task mean on BitNet, whereas $D { = } 8$ scores higher on Bonsai 1.7B. On both backbones, BitMTP maintains eleven-task mean scores close to SFT, with differences of 0.2–1.5 percentage points.

Table 5: BitNet hybrid-decoding configurations at $D \in \{ 4 , 8 \}$ . Rates are decode tokens/s; Avg. is the unweighted task mean. Parentheses give speedups over SFT AR; HE+ denotes HumanEval+.
<table><tr><td rowspan="2">Runtime configuration</td><td colspan="6">D=4 (tokens/s)</td><td colspan="6">D=8 (tokens/s)</td></tr><tr><td>GSM8K MATH-500</td><td></td><td>IFEval</td><td>HE+</td><td>MBPP+</td><td> $\operatorname { A v g } .$ </td><td>GSM8K MATH-500</td><td></td><td>IFEval</td><td>HE+</td><td>MBPP+</td><td>Avg.</td></tr><tr><td>SFT AR</td><td>365.0</td><td>365.0</td><td>368.7</td><td>393.7</td><td>374.8</td><td>373.4(1.0×)</td><td>365.0</td><td>365.0</td><td>368.7</td><td>393.7</td><td>374.8</td><td>373.4(1.0×)</td></tr><tr><td>Generic</td><td>158.0</td><td>133.0</td><td>103.8</td><td>157.5</td><td>162.9</td><td>143.0 (0.4×)</td><td>268.9</td><td>215.9</td><td>144.2</td><td>290.5</td><td>301.3</td><td>244.2 (0.7×)</td></tr><tr><td>+M1</td><td>724.2</td><td>658.2</td><td>515.1</td><td>739.2</td><td>758.0</td><td>678.9 (1.8×)</td><td>759.1</td><td>663.7</td><td>429.6</td><td>793.7</td><td>840.6</td><td>697.4 (1.9×)</td></tr><tr><td> $+ \mathbf { M } \mathbf { l } + \mathbf { M } 2$ </td><td>735.2</td><td>664.8</td><td>529.1</td><td>746.0</td><td>772.5</td><td>689.5 (1.8×)</td><td>784.5</td><td>684.3</td><td>452.4</td><td>825.3</td><td>879.6</td><td>725.2 (1.9×)</td></tr><tr><td>+ M1+M2+M3 (Full)</td><td>761.4</td><td>707.1</td><td>580.4</td><td>768.7</td><td>793.0</td><td> $7 2 2 . 1 ( 1 . 9 \times )$ </td><td>819.4</td><td>777.0</td><td>543.6</td><td>923.6</td><td>964.2</td><td>805.6 (2.2×)</td></tr></table>

BitNet at D=4 scores higher than SFT on WinoGrande and TruthfulQA-mc2, while Bonsai-1.7B at D=8 scores 74.4 versus 71.7 on BoolQ but lower on MMLU and HellaSwag. The aggregate scores thus reflect a task-dependent pattern of retained strengths, gains, and reductions. Relative to compactmodel baselines, BitNet’s MTP means remain near MiniCPM-2B’s 59.0 and Qwen2.5-1.5B-AWQ’s 58.9, while BF16 Qwen2.5-1.5B and Llama-3.2-3B score higher at 61.0 and 63.2. Together with Section 4.2, these results provide complementary evidence for BitMTP’s generation acceleration and broad capability retention relative to SFT.

## 4.4 Runtime Analysis

Decode efficiency. Table 5 compares different runtime configurations. The Generic configuration reaches a five-task mean throughput of 143.0 tokens/s at D=4 and 244.2 tokens/s at D=8. Both rates are below the SFT AR baseline of 373.42 tokens/s. The Full configuration reaches 722.12 tokens/s at D=4 and 805.57 tokens/s at D=8. It achieves the highest throughput on all five tasks at both block sizes. At D=8, its task-level speedups range from 1.5× on IFEval to 2.6× on MBPP+. These gains span mathematical reasoning, instruction following, and code generation.

Acceptance and runtime cost. In the forced-rejection test reported in Appendix C, each round performs block drafting and verification but commits at most one verifier-origin token, resulting in throughput below AR. In contrast, evaluated Bonsai-1.7B hybrid decoding at D=8 commits 4.6–7.1 tokens per iteration across five tasks. Multi-token commitment amortizes block execution and verification. Appendix D quantifies the memory and initialization costs of deploying the full runtime. The 0.5–1.6 MiB shared-KV increment on H20 supports history reuse across replay graphs, while Orin’s one-time startup overhead is amortized in persistent serving. Therefore, the proposed BitMTP translates multi-token commitment into decode speedups while sharing causal history across replay states.

## 5 Conclusion and Limitation

BitMTP enabled efficient multi-token generation on ultra-low-bit LLM backbones through joint NTP+MTP training, three decoding policies, and accepted-prefix-aligned runtime execution. The runtime combined width-specialized operators, shared-KV graph replay, and verifier scheduling. At D=8 on H20, hybrid decoding reached five-task mean speedups of $2 . 2 \times$ on BitNet b1.58-2B and 2.1–2.4× on 1.7B–4B Bonsai models at comparable task quality. BitMTP reached up to 2.0× in five task mean throughput on RTX 3090 and 3.3× on MBPP+ on Jetson Orin. These results demonstrated effective low-bit multi-token acceleration across ternary backbones, model scales, and GPU types.

There are some limitations of BitMTP. First, while BitMTP has been evaluated on 1.7B–4B backbones, its quality–throughput tradeoffs on larger models and longer contexts still need validation. Second, the current runtime relies on CUDA-specific low-bit kernels and CUDA Graphs and cannot run directly on non-CUDA accelerators. We will investigate these problems in further work.

## References

[1] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, et al. Flashinfer: Efficient and customizable attention engine for llm inference serving. Proceedings of Machine Learning and Systems, 7, 2025.

[2] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626, 2023.

[3] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In Proceedings of the 29th ACM International Conference on Architectural Supportfor Programming Languages and Operating Systems, Volume 3, pages 932–949, 2024.

[4] Hongyu Wang, Shuming Ma, Lingxiao Ma, Lei Wang, Wenhui Wang, Li Dong, Shaohan Huang, Huaijie Wang, Jilong Xue, Ruiping Wang, et al. Bitnet: 1-bit pre-training for large language models. Journal of Machine Learning Research, 26(125):1–29, 2025.

[5] Shuming Ma, Hongyu Wang, Shaohan Huang, Xingxing Zhang, Ying Hu, Ting Song, Yan Xia, and Furu Wei. Bitnet b1. 58 2b4t technical report. arXiv preprint arXiv:2504.12285, 2025.

[6] MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, et al. Minicpm4: Ultra-efficient llms on end devices. arXiv preprint arXiv:2506.07900, 2025.

[7] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle: Speculative sampling requires rethinking feature uncertainty. arXiv preprint arXiv:2401.15077, 2024.

[8] Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D Lee, Deming Chen, and Tri Dao. Medusa: Simple llm inference acceleration framework with multiple decoding heads. arXiv preprint arXiv:2401.10774, 2024.

[9] Fangcheng Liu, Yehui Tang, Zhenhua Liu, Yunsheng Ni, Duyu Tang, Kai Han, and Yunhe Wang. Kangaroo: Lossless self-speculative decoding for accelerating llms via double early exiting. Advances in Neural Information Processing Systems, 37:11946–11965, 2024.

[10] Yangzhou Liu, Yue Cao, Hao Li, Gen Luo, Zhe Chen, Weiyun Wang, Xiaobo Liang, Biqing Qi, Lijun Wu, Changyao Tian, et al. Sequential diffusion language models. arXiv preprint arXiv:2509.24007, 2025.

[11] Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. Draft& verify: Lossless large language model acceleration via self-speculative decoding. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 11263–11282, 2024.

[12] Mohammad Samragh, Arnav Kundu, David Harrison, Kumari Nishu, Devang Naik, Minsik Cho, and Mehrdad Farajtabar. Your llm knows the future: Uncovering its multi-token prediction potential. arXiv preprint arXiv:2507.11851, 2025.

[13] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437, 2024.

[14] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and efficient post-training quantization for large language models. In International conference on machine learning, pages 38087–38099. PMLR, 2023.

[15] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of machine learning and systems, 6:87–100, 2024.

[16] Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra. Llm-qat: Data-free quantization aware training for large language models. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 467–484, 2024.

[17] PrismML. Ternary Bonsai 8B: Ternary (1.58-bit) language models at 8B, 4B, and 1.7B scale. Technical report, PrismML, April 2026. URL https://github.com/PrismML-Eng/Bonsai-demo/blob/main/ ternary-bonsai-8b-whitepaper.pdf.

[18] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits. arXiv preprint arXiv:2402.17764, 2024.

[19] Fabian Gloeckle, Badr Youbi Idrissi, Baptiste Rozière, David Lopez-Paz, and Gabriel Synnaeve. Better & faster large language models via multi-token prediction. arXiv preprint arXiv:2404.19737, 2024.

[20] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-2: Faster inference of language models with dynamic draft trees. In Proceedings ofthe 2024 conference on empirical methods in natural language processing, pages 7421–7432, 2024.

[21] Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. Eagle-3: Scaling up inference acceleration of large language models via training-time test. arXiv preprint arXiv:2503.01840, 2025.

[22] Divyat Mahajan, Sachin Goyal, Badr Youbi Idrissi, Mohammad Pezeshki, Ioannis Mitliagkas, David Lopez-Paz, and Kartik Ahuja. Beyond multi-token prediction: Pretraining llms with future summaries. arXiv preprint arXiv:2510.14751, 2025.

[23] Xiaohao Liu, Xiaobo Xia, Weixiang Zhao, Manyi Zhang, Xianzhi Yu, Xiu Su, Shuo Yang, See-Kiong Ng, and Tat-Seng Chua. L-mtp: Leap multi-token prediction beyond adjacent context for large language models. arXiv preprint arXiv:2505.17505, 2025.

[24] John Kirchenbauer, Abhimanyu Hans, Brian Bartoldson, Micah Goldblum, Ashwinee Panda, and Tom Goldstein. Multi-token prediction via self-distillation. arXiv preprint arXiv:2602.06019, 2026.

[25] Juhyeong Kim, Sangyeon Yu, Gyunyeop Kim, and Sangwoo Kang. Fractalllm: Lossless self-speculative decoding with layer embedded self-compression. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 23666–23673, 2025.

[26] Zhuoming Chen, Avner May, Ruslan Svirschevski, Yuhsun Huang, Max Ryabinin, Zhihao Jia, and Beidi Chen. Sequoia: Scalable and robust speculative decoding. Advances in Neural Information Processing Systems, 37:129531–129563, 2024.

[27] Yunlong Hou, Fengzhuo Zhang, Cunxiao Du, Xuan Zhang, Jiachun Pan, Tianyu Pang, Chao Du, Vincent YF Tan, and Zhuoran Yang. Banditspec: Adaptive speculative decoding via bandit algorithms. arXiv preprint arXiv:2505.15141, 2025.

[28] Jian Chen, Yesheng Liang, and Zhijian Liu. Dflash: Block diffusion for flash speculative decoding. arXiv preprint arXiv:2602.06036, 2026.

[29] Juntao Zhao, Wenhao Lu, Sheng Wang, Lingpeng Kong, and Chuan Wu. Qspec: Speculative decoding with complementary quantization schemes. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 4779–4795, 2025.

[30] Evangelos Georganas, Dhiraj Kalamkar, Alexander Kozlov, and Alexander Heinecke. Ml-specqd: Multilevel speculative decoding with quantized drafts. arXiv preprint arXiv:2503.13565, 2025.

[31] Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W Mahoney, Kurt Keutzer, and Amir Gholami. Quantspec: Self-speculative decoding with hierarchical quantized kv cache. arXiv preprint arXiv:2502.10424, 2025.

[32] Jinheng Wang, Hansong Zhou, Ting Song, Shijie Cao, Yan Xia, Ting Cao, Jianyu Wei, Shuming Ma, Hongyu Wang, and Furu Wei. Bitnet. cpp: Efficient edge inference for ternary llms. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9305–9322, 2025.

[33] Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, and Baris Kasikci. Atom: Low-bit quantization for efficient and accurate llm serving. Proceedings ofMachine Learning and Systems, 6:196–209, 2024.

[34] Yujun Lin, Haotian Tang, Shang Yang, Zhekai Zhang, Guangxuan Xiao, Chuang Gan, and Song Han. Qserve: W4a8kv4 quantization and system co-design for efficient llm serving. Proceedings of Machine Learning and Systems, 7, 2025.

[35] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with io-awareness. Advances in neural information processing systems, 35: 16344–16359, 2022.

[36] Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, and Tri Dao. Flashattention-3: Fast and accurate attention with asynchrony and low-precision. Advances in Neural Information Processing Systems, 37:68658–68685, 2024.

[37] Ke Hong, Guohao Dai, Jiaming Xu, Qiuli Mao, Xiuhong Li, Jun Liu, Kangdi Chen, Yuhan Dong, and Yu Wang. Flashdecoding++: Faster large language model inference with asynchronization, flat gemm optimization, and heuristics. Proceedings ofMachine Learning and Systems, 6:148–161, 2024.

[38] Ramya Prabhu, Ajay Nayak, Jayashree Mohan, Ramachandran Ramjee, and Ashish Panwar. vattention: Dynamic memory management for serving llms without pagedattention. In Proceedings ofthe 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1, pages 1133–1150, 2025.

[39] Wonbeom Lee, Jungi Lee, Junghwan Seo, and Jaewoong Sim. InfiniGen: Efficient generative inference of large language models with dynamic KV cache management. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 155–172, 2024.

[40] Akide Liu, Jing Liu, Zizheng Pan, Yefei He, Gholamreza Haffari, and Bohan Zhuang. Minicache: Kv cache compression in depth dimension for large language models. Advances in Neural Information Processing Systems, 37:139997–140031, 2024.

[41] Coleman Hooper, Sehoon Kim, Hiva Mohammadzadeh, Michael W Mahoney, Yakun S Shao, Kurt Keutzer, and Amir Gholami. Kvquant: Towards 10 million context length llm inference with kv cache quantization. Advances in Neural Information Processing Systems, 37:1270–1303, 2024.

[42] Wen Cheng, Shichen Dong, Jiayu Qin, and Wei Wang. Qaq: Quality adaptive quantization for llm kv cache. In 2025 IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), pages 2563–2571. IEEE, 2025.

[43] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. Llumnix: Dynamic scheduling for large language model serving. In 18th USENIX symposium on operating systems design and implementation (OSDI 24), pages 173–191, 2024.

[44] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve. In 18th USENIX symposium on operating systems design and implementation (OSDI 24), pages 117–134, 2024.

[45] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody H Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E Gonzalez, et al. Sglang: Efficient execution of structured language model programs. Advances in neural information processing systems, 37:62557–62583, 2024.

[46] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. DistServe: Disaggregating prefill and decoding for goodput-optimized large language model serving. In 18th USENIX symposium on operating systems design and implementation (OSDI 24), pages 193–210, 2024.

[47] Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395, 2024.

[48] Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github. io/blog/qwen2.5/.

[49] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

## Appendix

In this appendix, we provide the following materials:

A. Method and runtime details, including one hybrid decoding iteration, verifier and KV-state transitions, and replay-correctness requirements (referring to Section 3 in the main paper).

B. Training data and cost (referring to Section 4.1 in the main paper).

C. Forced-rejection analysis for the runtime configuration study (referring to Section 4.4 in the main paper).

D. GPU-resident model and runtime memory, graph-state overhead, and Jetson Orin cold-start measurements (referring to Section 4.4 in the main paper).

## A Method and Runtime Details

Algorithm 1 summarizes one hybrid decoding iteration. The draft replay computes causal KV for the pending suffix while proposing a new block, and the runtime copies the pending-token KV rows into history. Hybrid decoding commits a sufficiently long confidence prefix; otherwise, an AR-causal verifier processes [anchor, $d _ { 1 } , \ldots , d _ { D } ]$ at width D+1 and determines the accepted prefix. Its additional token corrects the first rejected position or serves as a bonus after full acceptance. Confidence-only decoding uses the confidence branch, while speculative-only decoding always uses the verifier branch.

Algorithm 1 Hybrid decoding state update.   
Require: state $\boldsymbol { q } = \left( p , \ell , u \right)$ with $u \neq [ ]$ , block size D, and controls $( \tau , m )$   
1: set n ← |u| and select $\dot { s } _ { \mathrm { d } } = ( M , \sigma , \overset { \cdot } { r _ { \mathrm { d } } } )$ with $P = M - D \geq n$   
2: form $x _ { \mathrm { d } } = [ u , \mathrm { P A D } _ { . } ^ { P - n } , \mathrm { l a s t } ( u ) , m _ { \mathrm { m a s k } } ^ { D - 1 } ]$   
3: replay the draft graph on x and obtain $d _ { 1 : D }$ from the final D output rows   
4: materialize only the n pending KV rows and set ${ \bar { \ell } } \gets \ell + n$   
5: compute h on the GPU using Eq. 4   
6: select $s _ { \mathrm { v } } = ( D \mathrm { + } 1 , \sigma , r _ { \mathrm { v } } )$ and prepare $[ \mathrm { l a s t } ( u ) , d _ { 1 : D } ]$   
7: prepare verifier positions and the AR-causal mask on the GPU stream   
8: obtain h on the host   
9: if $h \geq$ m then   
10: set $\hat { a } \gets h$   
11: append $d _ { 1 : \hat { a } }$ to p   
12: set $u ^ { \prime }  d _ { 1 : \hat { a } }$ and $\ell ^ { \prime }  \bar { \ell }$   
13: else   
14: replay the verifier graph for s<sub>v</sub> and obtain $\tilde { d } _ { 1 : D + 1 }$   
15: set aˆ to the verifier-accepted prefix length   
16: set $b \gets \tilde { d } _ { \hat { a } + 1 }$   
17: promote the aˆ accepted draft KV rows, excluding the anchor   
18: append $d _ { 1 : \hat { a } } \cdot$ b to p   
19: set $u ^ { \prime }  [ b ]$ and $\bar { \ell } ^ { \prime } \gets \bar { \ell } + \hat { a }$   
20: end if   
21: return $\boldsymbol { q } ^ { \prime } = ( p , \ell ^ { \prime } , u ^ { \prime } )$

After a confidence transition, the committed drafts remain pending in the updated state until a later causal replay computes their KV from the emitted token ids. A verifier transition instead promotes the accepted causal rows and leaves only the verifier-origin token pending. Both branches preserve Eq. 1. In the verifier branch, full acceptance appends the $( \bar { D } { + } 1 ) { \ - } \mathrm { s t }$ verifier token as a bonus, whereas rejection at the first draft position emits only the verifier correction.

Superscripts on PAD and $m _ { \mathrm { m a s k } }$ denote repeated input slots. The draft anchor repeats the final pending token and its position; block queries hide its original pending copy. Verifier positions start at $\bar { \ell } - 1$ , and its mask hides the anchor’s history copy. Padding, mask, and rejected draft rows do not enter committed causal history. Each reachable pending length n has a captured draft width $M \geq n + D ;$ verification uses width $D { + 1 }$ . Graphs of different widths and roles share $B _ { \sigma }$ , with history migration only when capacity grows. These choices preserve the prefix-length relation in Eq. 1.

Table 6: Composition of the processed training mixture used in the training-cost study.
<table><tr><td>Source</td><td>Examples</td><td>Domain</td></tr><tr><td>ScaleQuest-Math</td><td>1,003,467</td><td>Math</td></tr><tr><td>SmolTalk</td><td>1,038,618</td><td>General</td></tr><tr><td>OPC-SFT-Stage2</td><td>435,930</td><td>Code</td></tr><tr><td>Tulu-3-SFT-Mixture</td><td>923,278</td><td>General</td></tr><tr><td>SciRIFF</td><td>70,520</td><td>Science</td></tr><tr><td>Table-GPT</td><td>13,214</td><td>Table</td></tr><tr><td>OpenCodeInstruct</td><td>17,494</td><td>Code</td></tr><tr><td>Reasoning-SFT-IF</td><td>41,413</td><td>Instruction following</td></tr><tr><td>Total</td><td>3,543,934</td><td>N/A</td></tr></table>

Table 7: One-epoch training cost on Bonsai-1.7B using eight NVIDIA H20 GPUs. All configurations use the same corpus, global batch size, and optimizer-step count. Relative cost uses the one-epoch SFT run as its baseline.
<table><tr><td>Configuration</td><td>Wall time (h)</td><td>GPU-hours</td><td>Relative cost</td></tr><tr><td>SFT</td><td>71.29</td><td>570.3</td><td>1.00×</td></tr><tr><td>BitMTP (D=4)</td><td>103.10</td><td>824.8</td><td>1.45×</td></tr><tr><td>BitMTP (D=8)</td><td>112.88</td><td>903.0</td><td>1.58×</td></tr></table>

Table 8: Forced-rejection stress test on H20. Decode gives five-task mean throughput (tokens/s); dark-red values in parentheses show speedups over SFT AR. Accepted and Com. report per-iteration means of accepted draft tokens and committed output tokens.
<table><tr><td rowspan="2">Backbone</td><td rowspan="2">SFT AR Decode</td><td colspan="3">Forced rejection, D=4</td><td colspan="3">Forced rejection, D=8</td></tr><tr><td>Decode</td><td>Accepted</td><td>Com.</td><td>Decode</td><td>Accepted</td><td>Com.</td></tr><tr><td>BitNet b1.58-2B</td><td>373.42 (1.0×)</td><td>142.91 (0.4×)</td><td>0</td><td>0.9975</td><td>103.40 (0.3×)</td><td>0</td><td>0.9974</td></tr><tr><td>Bonsai-1.7B</td><td>469.26 (1.0×)</td><td>188.57(0.4×)</td><td>0</td><td>0.9978</td><td>160.28 (0.3×)</td><td>0</td><td>0.9978</td></tr></table>

## B Training Data and Cost

Table 6 lists the eight training sources, covering mathematics, general dialogue, code, science, tables, and instruction following. For each backbone, SFT and joint NTP+MTP training use the same processed mixture drawn from these sources, comprising approximately 3.5 million examples.

Table 7 reports one-epoch training cost on Bonsai-1.7B using eight H20 GPUs. With the corpus and optimizer-step count fixed, joint NTP+MTP training uses 1.45×/1.58× the SFT GPU-hours at D=4/8, quantifying the additional cost of joint AR and block-prediction supervision.

## C Additional Experimental Analyses

Forced-rejection stress test. The stress test in Table 8 forces zero acceptance in each draft–verify round, paying for D drafts and width-(D+1) verification while committing at most one verifier-origin token. The result is below AR, with lower throughput at D=8 than at D=4 on both backbones. Individual zero-acceptance rounds can occur normally, but this all-reject stress test is an artificial worst case. Evaluated Bonsai-1.7B hybrid workloads at D=8 commit 4.6–7.1 tokens per iteration, amortizing drafting and verification to support the measured low-bit generation speedups.

## D Runtime Memory and Startup Cost

Tables 9 and 10 quantify memory and startup costs at a 4,096-token segment. SFT AR captures one graph and BitMTP captures seven graphs over shared causal KV. Model totals include all resident weight representations and the embedding and output-projection storage.

Table 9: H20 model and runtime memory at a 4,096-token segment, in MiB. Model total includes Emb./head. Static contains per-graph token, position, and mask buffers. Other allocated is CUDAallocated memory after graph capture minus Model total, Shared KV, and Static. Peak is the median of three runs’ maximum device-memory readings across the recorded stages.
<table><tr><td>Configuration</td><td></td><td></td><td></td><td>Graphs Emb./head BF16 prefill Packed decode Model total Shared KV Static</td><td></td><td></td><td></td><td>Other allocated</td><td>Peak</td></tr><tr><td colspan="10">BitNet b1.58-2B</td></tr><tr><td>SFT AR</td><td>1</td><td>2505.0</td><td>6563.3</td><td>881.6</td><td>9950.0</td><td>300.1</td><td>0.5</td><td>132.411028</td><td></td></tr><tr><td>BitMTP (D=4)</td><td>7</td><td>2818.1</td><td>6563.3</td><td>881.6</td><td>10263.1</td><td>300.6</td><td>13.7</td><td></td><td>326.811714</td></tr><tr><td>BitMTP (D=8)</td><td>7</td><td>2818.1</td><td>6563.3</td><td>881.6</td><td>10263.1</td><td>301.2</td><td>26.9</td><td></td><td>327.611742</td></tr><tr><td colspan="10">Bonsai-1.7B</td></tr><tr><td>SFT AR</td><td>1</td><td>671.3</td><td>0.0</td><td>651.2</td><td>1322.5</td><td>448.1</td><td>0.6</td><td>0.0</td><td>5118</td></tr><tr><td>BitMTP (D=4)</td><td>7</td><td>967.5</td><td>0.0</td><td>651.2</td><td>1618.8</td><td>448.9</td><td>16.1</td><td>165.1</td><td>5656</td></tr><tr><td>BitMTP (D=8)</td><td>7</td><td>967.5</td><td>0.0</td><td>651.2</td><td>1618.8</td><td>449.8</td><td>31.5</td><td>165.1</td><td>5656</td></tr></table>

Table 10: Jetson Orin memory and cold-start cost at a 4,096-token segment. Memory is in MiB. Capture is the allocator increase from KV allocation to ready-to-serve. Peak reserved is the firstrequest CUDA allocator peak; cold start uses an empty compiler cache.
<table><tr><td>Configuration</td><td>Graphs</td><td>Shared KV</td><td>Capture</td><td>Peak reserved</td><td>Cold start (s)</td></tr><tr><td>BitNet b1.58-2B-SFT AR</td><td>1</td><td>301.2</td><td>16.7</td><td>11312</td><td>14.34</td></tr><tr><td>BitMTP-BitNet (D=4)</td><td>7</td><td>300.7</td><td>79.2</td><td>11716</td><td>17.30</td></tr><tr><td>BitMTP-BitNet (D=8)</td><td>7</td><td>301.2</td><td>93.4</td><td>11730</td><td>17.92</td></tr><tr><td>Bonsai-1.7B-SFT AR</td><td>1</td><td>448.1</td><td>16.8</td><td>5486</td><td>13.41</td></tr><tr><td>BitMTP-Bonsai (D=4)</td><td>7</td><td>449.0</td><td>81.6</td><td>5876</td><td>16.52</td></tr><tr><td>BitMTP-Bonsai (D=8)</td><td>7</td><td>449.9</td><td>97.9</td><td>5960</td><td>16.53</td></tr></table>

H20 shared KV increases by only 0.5–1.6 MiB from one to seven graphs, reusing causal history rather than duplicating full caches. Sampled peak device memory is 686–714 MiB higher on BitNet and 538 MiB higher on Bonsai-1.7B than on SFT AR.

Orin’s first-request peak reserved memory increases by 404–418 MiB on BitNet and 390–474 MiB on Bonsai-1.7B, with cold-start increments of 2.96–3.58 s and 3.11–3.12 s, respectively. This one-time per-process cost is amortized over requests, supporting persistent multi-token generation through shared history and replayable execution states.