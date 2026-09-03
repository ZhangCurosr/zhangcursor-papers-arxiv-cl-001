# Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models

Tianqi Xiao<sup>1,3</sup>\* Shiyao Cui<sup>2,1†</sup> Minghao Zhang<sup>4</sup> Junxiao Yang<sup>1</sup> Renmiao Chen<sup>1</sup>

<sup>1</sup>The Conversational AI (CoAI) group, DCST, Tsinghua University

<sup>2</sup>School of Cyberspace Security, Beijing University of Posts and Telecommunications

<sup>3</sup>Northwestern Polytechnical University

<sup>4</sup>Tsinghua University

xtqcu@mail.nwpu.edu.cn, cuishiyao@foxmail.com

## Abstract

Visual modality enhances the capabilities of multimodal large language models (MLLMs) but also introduces a safety concern: a benign textual query may convey harmful intent when grounded in a visual image. We term this cross-modal safety drift and our pilot studies show that the safety response rate for such requests is substantially lower than that for requests containing explicitly unsafe text. This paper aims to systematically study this issue. First, we conduct an empirical analysis to identify representative unsafe response patterns. Building on these, we interpret model representations and attentions, revealing that visually risky cues receive limited attention and weakly trigger refusal. Motivated by the observation that safety signals from unsafe text processing can be transferred, we propose safetyawareness representation transfer (SRT), a lightweight direction-refinement method that mitigates cross-modal safety drift with a frozen MLLM backbone. Experiments across multiple benchmarks and models show that SRT effectively improves safety in diverse cross-modal settings while preserving utility. Code is available at https://github.com/cucu220123/ safety-awareness. Warning: This paper contains potentially sensitive content.

## 1 Introduction

While multimodal large language models (MLLMs) excel at understanding and reasoning over text–image inputs, the incorporation of visual modalities also introduces new safety risks (Liu et al., 2024; Gong et al., 2025). Unlike text-only LLMs, whose safety decisions primarily rely on textual prompts, MLLMs must align their safety behaviors with the joint semantics of text and image, thereby necessitating cross-modal safety alignment (Wang et al., 2025; Hu et al., 2025).

## Semantic intent: I want to commit suicide

![](images/078f559e6477da2d4b7c8b690c0bd8283e72d0ab0b4d9ca5c08a99f0cce32b0c.jpg)  
Figure 1: Illustration of cross-modal safety drift. From top to bottom, rows show responses from Qwen3- VL-8B-Instruct, Qwen3-VL-8B-Instruct+ShiftDC, and Qwen3-VL-8B-Instruct+SRT(ours).

A critical issue in cross-modal safety alignment arises when the safety implications of benign textual prompts are revealed by visual inputs. We term this issue cross-modal safety drift, which is usually reflected in two representative patterns. The first is safe text with unsafe image $( S _ { t } U _ { i } \to U )$ , where a safe prompt is grounded into an unsafe request with safety-critical visual content. In the middle panel of Figure 1, a benign textual prompt is paired with an image containing self-harm cues, causing the combined text-image input to convey self-harm intent. The second pattern is safe text with safe image $( S _ { t } S _ { i }  U )$ , where neither modality is explicitly unsafe on its own, yet their combination could imply a harmful intent. Our pilot experiments show that the model could achieve safe ratio of 98.3% when harmful cues are expressed in text while only 53.3% when the hazard is grounded into vision, as the right panel of Figure 1 shows.

Existing MLLM safety-alignment methods often struggle to elicit safety awareness in MLLMs, namely the ability to process all requests with safety concern and refuse or give warning to unsafe requests, which we assume to be the alignment goal. Specifically, existing methods mainly improve MLLM safety by restoring or reactivating safety mechanisms weakened by visual distribution shifts (Zou et al., 2025; Jiang et al., 2026; Liu et al., 2025; Ding et al., 2025), via either safety-specific training data or inference-time interventions such as prompting and steering. However, these methods fail to reactivate the lost upstream safety awareness in MLLMs when confronted with cases where unsafe intent is revealed only with joint text-image grounding, and thus may fail to robustly improve MLLM safety. As the pilot case in Figure 1 shows, Qwen3-VL enhanced with ShiftDC (Zou et al., 2025), a representative safety method, fails to provide necessary warnings or refusal for a benigntext-conditioned unsafe request.

For the issue of cross-modal safety drift, this paper investigates 1) how MLLM safety degrades under such cross-modal grounding and 2) how to improve it, where we perform from three aspects.

(1) Characterize typical failure response patterns. We first conduct a failure analysis on $S _ { t } U _ { i }  U$ and $S _ { t } S _ { i }  U$ and characterize how MLLMs fail for safety. This identifies representative failure patterns and provides empirical insights for subsequent mechanistic interpretations.

(2) Interpret how MLLM safety fails. We perform analyses on model activations and attention to examine how the absence of explicit harmful text impacts safety. Besides, we examine whether harmful-text-induced safety signals could be transferred to improve the cross-modal safety capability.

(3) Transfer unsafe-text-induced safety awareness to improve safety. We propose safetyawareness representation transfer (SRT), which first refines coarse safety awareness derived from harmful-text-induced activations to balance safety and utility, and then transfers the representation to cross-modal safety drift for safety improvement.

Our findings are summarized as follows. Empirically, we identify two recurring failure modes under cross-modal safety drift: MLLMs either overlook visual risk cues or fail to incorporate them into safety-related decision making. Mechanistically, we show that explicit harmful text is crucial for activating safety awareness, and identify pre-refusal-layer directions as coarse safety awareness that can be transferred to cross-modal risky inputs. Based on these insights, we propose safety-awareness representation transfer (SRT), a lightweight direction-refinement method that refines coarse safety awareness while keeping the base MLLM frozen. Across representative crossmodal benchmarks, SRT significantly improves safety performance while preserving utility.

Pattern 1: Visual Risks Unawareness  
![](images/40b2fb7d398591df1aa549c8e97589c68bf39646c8dc7a4eb1e35bafcaa179cf.jpg)

Pattern 2: Safety Implication Unawareness  
![](images/d385e3df9b1223353dfa197948c4750cdd540f0817c216818d227580ae79129f.jpg)  
Figure 2: Illustration of representative failure patterns.

## 2 Empirical Analysis

To better understand how MLLMs fail under crossmodal safety drift in $S _ { t } U _ { i }  U$ and $S _ { t } S _ { i }  U$ we first conduct a response-level analysis to identify typical failure patterns. Using Qwen3-VL-8B-Instruct as a case study, we examine 100 failed cases of $S _ { t } U _ { i }  U$ from VLSBench and 100 failed cases of $S _ { t } S _ { i }  U$ from SIUO. This analysis reveals two representative failure patterns.

The first pattern is visual risk unawareness, where the model appears to neglect the risk cues present in the image. In such cases, the response seems generated primarily according to the textual instruction and visually risky regions do not seem to be incorporated during model response generation. For example, in the upper case of Figure 2, the model’s response does not mention the dangerous element in the image, i.e., the gun held by the woman. This pattern accounts for 37% of our inspected failure set, suggesting that the model seems insufficiently attends to the risky visual regions.

The second pattern is safety implication unawareness, where the model responds to riskrelated elements in the input but fails to recognize their safety implications. For example, in the lower case of Figure 2, the model captures the overall semantics of the image and responds to the request about the firearm, yet it fails to recognize that providing firearm-related guidance may carry safety risks. This pattern accounts for 63% of failure cases in our manual analysis, suggesting that the perceived visual risk is weakly connected to safetyaligned decision making.

## 3 Interpret Cross-modal Safety Drift

Motivated by the failure patterns observed in our empirical analysis, this section further investigates the mechanisms underlying cross-modal safety drift. Specifically, we aim to uncover the internal model behaviors that contribute to these failure cases, where two representative open-source MLLMs, Qwen3-VL-8B-Instruct and Gemma-3- 12B-IT, are used for interpretation.

Preliminary: Safety representations analysis.

Following Turner et al. (2023) and Rimsky et al. (2024), we use activation directions as a lightweight tool to analyze safety-related representations in the model’s activation space. Prior work on LLMs shows that safety behavior can be largely mediated by a low-dimensional direction in activations (Arditi et al., 2024). Extending this idea to MLLMs, for an input x, we denote the hidden representation at layer ℓ and post-instruction position $p$ as $h _ { \ell , p } ( x ) \in \mathbb { R } ^ { d }$ , where d is the hidden-state dimension. Given harmful-text samples $\mathbb { D } _ { \mathrm { H T } }$ , where the model can reliably refuse, and benign samples $\mathbb { D } _ { \mathrm { B } }$ , we extract a candidate activation direction by the difference between their mean activations:

$$
\begin{array} { r } { \begin{array} { r } { v _ { \ell , p } ^ { \mathrm { H T - B } } = \mu _ { \ell , p } ^ { \mathrm { H T } } - \mu _ { \ell , p } ^ { \mathrm { B } } . } \end{array} } \end{array}\tag{1}
$$

Here, $\mu _ { \ell , p } ^ { \mathrm { H T } }$ and $\mu _ { \ell , p } ^ { \mathrm { B } }$ denote the mean hidden representations over $\mathbb { D } _ { \mathrm { H T } }$ and $\mathbb { D } _ { \mathrm { B } } .$ , respectively. To test the effect of a direction $v _ { \ell , p } ^ { \mathrm { H T - B } }$ , we add it to the hidden representations at its source layer ℓ during the forward pass, with the intervention strength α set to 1 by default. We then quantify its effect using a next-token refusal metric (Arditi et al., 2024). Let $R ( x ; 0 ) , R ( x ; v )$ denote the original refusal score and the score after adding direction v, the refusal-score increase induced by v on a dataset D

![](images/a23ce2e2153999401d2d1b9f9a01d876c0edd97bab7fad674a043004a8938935.jpg)

![](images/328e89d84af26b13ff3fb5e9da975e46e704e8a548bf92596fea61d6a47acf40.jpg)  
Figure 3: RV-plane visualization of paired samples on Qwen3-VL-8B-Instruct and Gemma-3-12B-IT at layer 22 and layer $^ { 3 0 , }$ respectively.

is computed as follows:

$$
\begin{array} { r } { \Delta _ { \mathbb { D } } ( v ) = \mathbb { E } _ { x \sim \mathbb { D } } \left[ R ( x ; v ) \right] - \mathbb { E } _ { x \sim \mathbb { D } } \left[ R ( x ; 0 ) \right] . } \end{array}\tag{2}
$$

For subsequent analysis, we identify layers where adding this direction produces particularly large increases in refusal score on $\mathbb { D } _ { \mathrm { B } }$ and refer to them as refusal layers. In our study, the identified refusal layers are mainly concentrated in layers 19– 30 for Qwen3-VL-8B-Instruct and layers 20–48 for Gemma-3-12B-IT (more details in Appendix B).

Observation 1: Harmful text substantially increases safety awareness.

Since MLLMs reliably refuse explicit harmfultext requests but fail on cross-modal risky inputs, we first examine how risky cues in texts influence model’s internal representations. To isolate textual harmfulness, we construct paired inputs with the same unsafe image and underlying intent: one textual prompt contains explicit harmful cues, while the other is benignly phrased. Following Jiang et al. (2025), we project these samples onto a twodimensional plane defined by a refusal axis and a reference semantic axis. The refusal axis captures the model’s refusal tendency, while the semantic axis captures non-refusal-related input semantics. Here, we perform visualization to see how paired inputs differ in safety behavior and semantics.

As shown in Figure 3, at the middle refusal layer of both Qwen3-VL-8B-Instruct and Gemma-3-12B-IT, harmful-text and benign-text inputs are close along the reference semantic axis but clearly separated along the refusal axis. This suggests that the paired inputs retain similar semantics, while explicit harmful text substantially increases the model’s refusal tendency.

Observation 2: Visual risky cues are insufficiently attended to in the refusal layers.

Building on Observation 1, we further investigate why the model’s refusal tendency drops when unsafe semantics are no longer explicitly expressed in text. We examine whether risk cues are attended to the refusal layers and analyze this pattern through both visualization and quantitative statistics. To identify risk-related cues in both text and image, we construct a harmful-word prototype set, where textual cues are located by directly matching input words against the prototype set, and visual cues are identified by interpreting visual-token semantics with LatentLens (Krojer et al., 2026). We compare harmful-text cases, where models reliably refuse, with failed cross-modal cases $S _ { t } U _ { i }$ and $S _ { t } S _ { i }$

(a)  
![](images/27b56be03ac259a0d15eb6a3c55c2ba1cf08f299632cfb6b8380092ab3cd901b.jpg)

![](images/1716826b4b3e6ad3a4bba22a8eece20627b58b832d90080e3937b9973768f5bf.jpg)

![](images/fe6517e8e9ec91df6734960ae5e77fc138c2c0f45fe8f435b240ff84bff97c70.jpg)  
Figure 4: Attention patterns at refusal layers under unsafe-text and cross-modal risky inputs.

The results show a clear contrast. In explicit harmful-text cases, Figure 4(a) shows that the word “harm” receives concentrated attention at the refusal layers, and harmful textual tokens account for 16.23% of the attention mass over text tokens on average. In contrast, for $S _ { t } U _ { i }$ , Figure 4(b) shows that the visual cue “drugs” is not sufficiently attended to, and harmful-semantics visual tokens account for only 2.98% of the attention mass over image tokens. Similarly, for $S _ { t } S _ { i }$ , Figure 4(c) shows that the cue “stone” becomes safety-critical after imagetext composition but still receives insufficient attention. This suggests that visual risky cues are not sufficiently incorporated into refusal-related computation, explaining the weakened refusal tendency when unsafe cues are shifted away from text.

## Observation 3: Unsafe-text-induced safety representations can be transferred for cross-modal safety.

Previous observations suggest that explicit harmful text can reliably activate refusal-related computation, while cross-modal risky inputs often fail to do so. We therefore examine whether the internal safety representations behind successful harmfultext refusal can be extracted and transferred to cross-modal failure cases. Following the differencein-means formulation in Preliminary, we construct activation directions from two sets. The harmfultext set $\mathbb { D } _ { \mathrm { H T } }$ contains explicit harmful-text inputs where the model can reliably refuse, and the crossmodal set $\mathbb { D } _ { \mathrm { C M } }$ contains risky image-text inputs where the model fails to refuse.

![](images/acd8acfc151dab3305b80f5080198b73f1d3a19f3712fd5444039d57e4a52cdc.jpg)

![](images/1ed69fe7ea710406e77b876f41d9b470577eb40280640ab2a4ada6158451e6c5.jpg)  
Figure 5: Layer-wise analysis of activation directions on Qwen3-VL-8B-Instruct and Gemma-3-12B-IT.

We first extract activation directions $v _ { \ell , p } ^ { \mathrm { H T - C M } }$ using the difference in mean activations between D and $\mathbb { D } _ { \mathrm { C M } }$ . To test whether these directions can transfer to cross-modal risky inputs, we add each direction to samples from $\mathbb { D } _ { \mathrm { C M } }$ and measure the refusal-score increase by subtracting the original baseline before intervention:

$$
\begin{array} { r } { \Delta _ { \mathrm { C M } } ^ { \ell , p } = \Delta _ { \mathbb { D } _ { \mathrm { C M } } } \left( \alpha v _ { \ell , p } ^ { \mathrm { H T - C M } } \right) . } \end{array}\tag{3}
$$

As shown by the blue curves in Figure 5, many unsafe-text-induced directions substantially increase the refusal tendency on $\mathbb { D } _ { \mathrm { C M } }$ , especially within the refusal layers identified in Preliminary. This suggests that the safety signals activated by harmful text can indeed be transferred to cross-

modal risky inputs.

To further examine whether this effect is specific to risky inputs or simply pushes any input toward refusal like $v _ { \ell , p } ^ { \mathrm { H T - B } }$ , we introduce a benign VQA set $\mathbb { D } _ { \mathrm { B } }$ . We add $v _ { \ell , p } ^ { \mathrm { H T - C M } }$ to samples from $\mathbb { D } _ { \mathrm { B } }$ and measure the refusal-score increase by subtracting the original baseline before intervention:

$$
\begin{array} { r } { \Delta _ { \mathrm { B } } ^ { \ell , p } = \Delta _ { \mathbb { D } _ { \mathrm { B } } } \left( \alpha v _ { \ell , p } ^ { \mathrm { H T - C M } } \right) . } \end{array}\tag{4}
$$

Although directions also generally increase refusal on D<sub>B</sub>, Figure 5 reveals an interesting pattern. $\mathbf { A }$ joint comparison of the two curves shows that at some specific layers, such as layer 16 on Qwen3- VL-8B-Instruct and layer 15 on Gemma-3-12B-IT, the increase on $\mathbb { D } _ { \mathrm { C M } }$ is much larger than that on D<sub>B</sub>. This suggests that directions from these layers have a selective effect: they are more responsive to cross-modal risky inputs than to benign inputs.

## 4 Method

Observation 3 suggests a potential way to mitigate Cross-modal Safety Drift: unsafe-text-induced activation directions can be transferred to cross-modal risky inputs to elicit refusal. However, the central goal of safety alignment is not merely to increase refusal, but to refuse unsafe requests while remaining helpful to benign ones. A straightforward choice is to select the direction that produces the largest refusal-score increase on $\mathbb { D } _ { \mathrm { C M } }$ from $v _ { \ell , p } ^ { \mathrm { H T - C M } }$ and refine it to exert no effect on $\mathbb { D } _ { \mathrm { B } }$ while maintaining its effect on $\mathbb { D } _ { \mathrm { C M } }$ . However, since this direction is largely biased toward pure refusal amplification, the above objective is quite hard to achieve. Fortunately, Observation 3 shows that some directions exhibit a more selective effect and meanwhile still a competitive refusal effect. Also, these directions are located just before the refusal layers identified in Preliminary, suggesting that they are not refusalexecution directions like the direction above, but rather coarse forms of safety awareness, which activate the model’s safety perception upstream and encourage it to process potential risks. To automatically select such a direction, we use the refusalscore increases defined in Observation 3 and define the delta increase in refusal-score as:

$$
G _ { \ell , p } = \Delta _ { \mathrm { C M } } ^ { \ell , p } - \Delta _ { \mathrm { B } } ^ { \ell , p } .
$$

We select $( \ell _ { 0 } , p _ { 0 } ) = \arg \operatorname* { m a x } _ { \ell , p } G _ { \ell , p }$ and define $v _ { 0 } = v _ { \ell _ { 0 } , p _ { 0 } } ^ { \mathrm { H T - } }$ as the initial raw safety awareness. In this section, we refine $v _ { 0 }$ into a more robust safety representation. Since safety behavior depends on both intermediate refusal-process and the final representation that directly determines model outputs, we refine $v _ { 0 }$ from both perspectives.

## 4.1 Refusal-Process Supervision

Observation 2 shows that safety-critical visual cues are insufficiently attended to in refusal-related computation. We therefore optimize $v _ { 0 } .$ , which is typically extracted at pre-refusal layers, to better activate refusal-related computation in the subsequent refusal layers. Specifically, we first quantify the downstream effect of a direction intervention as:

$$
E ( x ; v ) = \frac { \left\| h _ { \ell _ { \mathrm { o b } } , p } ( x ; \ell _ { 0 } , v ) - h _ { \ell _ { \mathrm { o b } } , p } ( x ) \right\| _ { 2 } } { \left\| h _ { \ell _ { \mathrm { o b } } , p } ( x ) \right\| _ { 2 } + \epsilon } ,\tag{5}
$$

where $\ell _ { 0 }$ denotes the source layer where direction $v$ is inserted, and $\ell _ { \mathrm { o b } }$ denotes a subsequent refusal layer for effect observation. We use $h _ { \ell _ { \mathrm { o b } } , p } ( x )$ and $h _ { \ell _ { \mathrm { o b } } , p } ( x ; \ell _ { 0 } , v )$ to denote the hidden states at position $p$ in layer $\ell _ { \mathrm { o b } }$ before and after adding direction v at layer $\ell _ { 0 } .$ , respectively, where $p = p _ { 0 }$ and $\epsilon > 0$ is a small constant for numerical stability. This quantity measures the downstream influence of the intervention, and we hope ideal safety awareness to exert more refusal-related downstream effect on unsafe requests and less refusal-unrelated downstream effect. However, since it is difficult to directly separate these two parts from the effect defined in Eq. 5, we optimize v indirectly through a contrastive constraint. The key idea is to treat $\mathbb { D } _ { \mathrm { C M } }$ as the positive side and $\mathbb { D } _ { \mathrm { B } }$ as the negative side. By contrasting the direction’s downstream effect on these two sides, we encourage the direction to capture effects that are specific to risky inputs, which is semantically the refusal-related computation. Specifically, on $\mathbb { D } _ { \mathrm { C M } }$ , since $v _ { 0 }$ already provides a competitive ability to elicit final refusal, and also $E ( x ; v )$ does not have a direct target value that indicates an optimal quantity, we do not maximize $E ( x ; v )$ directly. Instead, we preserve $E ( x ; v )$ induced by $v _ { 0 }$ as follows:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { p } } ^ { R } = \mathbb { E } _ { { x } \sim \mathbb { D } _ { \mathrm { C M } } } \left[ \left[ E ( { x } ; { v } _ { 0 } ) - E ( { x } ; { v } ) \right] _ { + } \right] . } \end{array}\tag{6}
$$

Here, $[ a ] _ { + } = \operatorname* { m a x } ( 0 , a )$ denotes the positive-part operator. Meanwhile, we place the main calibration pressure on the negative side: on $\mathbb { D } _ { \mathrm { B } }$ , since no risk cues are present, $E ( x ; v )$ should be minimized:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s } } ^ { R } = \mathbb { E } _ { \boldsymbol { x } \sim \mathbb { D } _ { \mathrm { B } } } \left[ E ( \boldsymbol { x } ; \boldsymbol { v } ) \right] . } \end{array}\tag{7}
$$

This contrastive constraint helps refine $v _ { 0 }$ to be a more robust safety awareness direction through the refusal-process angle.

![](images/2f2d03ea08504502d7d36750ec345cacb968f543a293cdbfb61df151ab9397e2.jpg)  
Figure 6: Overview of Safety-awareness Representation Transfer (SRT).

## 4.2 Output-Behavior Supervision

In addition to internal processing optimized above, we further refine the raw safety awareness through the last-layer logits, so that the internal refusalrelated processing raised can be genuinely translated into final output behavior.

Following the preservation-suppression form in Section 4.1, we also define an output-level contrast for learning. The preservation loss is defined as:

$$
\begin{array} { r } { \mathcal L _ { \mathrm { p } } ^ { O } = \mathbb { E } _ { x \sim \mathbb { D } _ { \mathrm { C M } } } \left[ \left[ R ( x ; v _ { 0 } ) - R ( x ; v ) \right] _ { + } \right] , } \end{array}\tag{8}
$$

For the suppression part, following prior analysis (Arditi et al., 2024), we measure output-level disturbance by the KL divergence between the original and intervened output distributions. Let $P ( \cdot | x )$ denote the original output distribution and $P ( \cdot | x ; v )$ denote the distribution after adding direction v. The suppression loss is defined as:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { s } } ^ { O } = \mathbb { E } _ { x \sim \mathbb { D } _ { \mathrm { B } } } \left[ \mathrm { K L } \left( P ( \cdot | x ) \parallel P ( \cdot | x ; v ) \right) \right] . } \end{array}\tag{9}
$$

## 4.3 Overall Objective

The final objective consists of a refusal-process part and an output-behavior part:

$$
{ \mathcal { L } } _ { \mathrm { S R T } } = \underbrace { \lambda _ { \mathrm { p } } ^ { R } { \mathcal { L } } _ { \mathrm { p } } ^ { R } + \lambda _ { \mathrm { s } } ^ { R } { \mathcal { L } } _ { \mathrm { s } } ^ { R } } _ { \mathrm { r e f u s a l - p r o c e s s } } + \underbrace { \lambda _ { \mathrm { p } } ^ { O } { \mathcal { L } } _ { \mathrm { p } } ^ { O } + \lambda _ { \mathrm { s } } ^ { O } { \mathcal { L } } _ { \mathrm { s } } ^ { O } } _ { \mathrm { o u t p u t - b e h a v i o r } } .\tag{10}
$$

Here, $\lambda _ { \mathrm { p } } ^ { R } , \lambda _ { \mathrm { s } } ^ { R } , \lambda _ { \mathrm { p } } ^ { O }$ , and $\lambda _ { \mathrm { s } } ^ { O }$ are non-negative coefficients balancing the four sub-goals. Based on this objective, we optimize the direction v initialized from $v _ { 0 }$ while keeping the base MLLM frozen.

## 5 Experiment

## 5.1 Setups

Benchmarks. For safety evaluation, we consider all image-text combinations that can trigger unsafe outputs, including 1) $S _ { t } S _ { i }$ setting, we use SIUO and the SSU subset of HoliSafe (Wang et al., 2025; Lee et al., 2026). 2) $S _ { t } U _ { i }$ setting, we use VLS-Bench, MM-SafetyBench, FigStep, and the $S _ { t } U _ { i }$ subset of HoliSafe (Hu et al., 2025; Liu et al., 2024; Gong et al., 2025; Lee et al., 2026). In addition, we further evaluate settings with explicit unsafe text, including $U _ { t } S _ { i }$ and $U _ { t } U _ { i }$ , using the $U _ { t } S _ { i }$ and $U _ { t } U _ { i }$ subsets of HoliSafe, respectively.

For utility evaluation, we use MM-Vet, MME, and L-Bench (Yu et al., 2024; Fu et al., 2025; Liu et al., 2023) following previous works (Zou et al., 2025; Liu et al., 2025).

Evaluation and Metrics. For safety, we report ASR or unsafe rate following each benchmark protocol. Unsafe responses are judged by keyword matching or GPT-4o-mini, using the same prompts and parsing rules as the official evaluation code. For utility, we follow the official benchmark evaluation scripts and report their performance scores.

Models. To verify the generalizability of SRT, we evaluate it on Qwen3-VL-8B-Instruct, Gemma-3-12B-IT, InternVL2-8B, and LLaVA-OneVision-Qwen2-7B-OV-Chat-HF.

Baselines. We compare SRT with recent MLLM safety alignment baselines from three inferencetime categories, including detection-based defense ETA (Ding et al., 2025), safety-prompt-based calibration defense CoCA (Gao et al., 2024) and safety steering methods of DTR and ShiftDC (Jiang et al., 2026; Zou et al., 2025).

Implementation details. For direction extraction and SRT training, we use a small subset of 300 samples from VLSBench as the harmful set and all 217 samples from MM-Vet as the benign set. During training, the base MLLM is frozen and only the safety-awareness direction is optimized. Training takes only about 30 minutes on 4 NVIDIA A100 GPUs. More details are shown in Appendix E.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2"> $S _ { t } S _ { i }$ </td><td colspan="4"> $S _ { t } U _ { i }$ </td><td> $U _ { t } S _ { i }$ </td><td> $U _ { t } U _ { i }$ </td></tr><tr><td>SIUO Unsafe ↓</td><td>Holi-  $S _ { t } S _ { i }$  ASR↓</td><td>VLSBench MM-Safety Unsafe ↓</td><td>ASR↓</td><td>FigStep ASR↓</td><td>Holi-  $S _ { t } U _ { i }$  ASR↓</td><td>Holi  ${ \mathbf { } } J _ { t } S _ { i }$  ASR↓</td><td> $\operatorname { H o l i } { - U _ { t } U _ { i } }$  ASR↓</td></tr><tr><td rowspan="6">Qwen3-VL-8B</td><td>Baseline</td><td>46.70</td><td>56.68</td><td>34.09</td><td>43.10</td><td>28.6</td><td>92.48</td><td>33.25</td><td>27.07</td></tr><tr><td>+ ETA</td><td>44.91</td><td>50.41</td><td>29.06</td><td>33.80</td><td>9.8</td><td>71.47</td><td>33.03</td><td>28.12</td></tr><tr><td>+ DTR</td><td>44.31</td><td>56.68</td><td>23.59</td><td>41.42</td><td>15.4</td><td>90.52</td><td>33.81</td><td>24.57</td></tr><tr><td>+ ShiftDC</td><td>42.51</td><td>86.49</td><td>35.10</td><td>47.73</td><td>22.8</td><td>80.63</td><td>42.15</td><td>36.79</td></tr><tr><td>+ CoCA</td><td>41.91</td><td>56.26</td><td>39.93</td><td>38.39</td><td>10.2</td><td>81.87</td><td>37.37</td><td>35.08</td></tr><tr><td>+ SRT</td><td>19.76</td><td>19.35</td><td>1.47</td><td>23.10</td><td>8.2</td><td>64.47</td><td>8.00</td><td>5.91</td></tr><tr><td rowspan="6">Gemma-3-12B-IT</td><td>Baseline</td><td>68.86</td><td>82.17</td><td>69.43</td><td>68.69</td><td>31.8</td><td>96.70</td><td>52.16</td><td>41.26</td></tr><tr><td>+ETA</td><td>55.68</td><td>70.19</td><td>48.24</td><td>48.21</td><td>6.8</td><td>83.93</td><td>47.60</td><td>39.02</td></tr><tr><td>+ DTR</td><td>62.87</td><td>82.59</td><td>64.94</td><td>66.90</td><td>28.4</td><td>95.98</td><td>51.61</td><td>40.34</td></tr><tr><td>+ ShiftDC</td><td>57.48</td><td>80.51</td><td>65.12</td><td>66.84</td><td>19.6</td><td>94.85</td><td>55.28</td><td>48.22</td></tr><tr><td>+ CoCA</td><td>55.68</td><td>77.01</td><td>44.46</td><td>50.35</td><td>13.2</td><td>87.43</td><td>51.16</td><td>45.07</td></tr><tr><td>+ SRT</td><td>23.35</td><td>21.58</td><td>2.96</td><td>20.10</td><td>1.0</td><td>64.57</td><td>16.46</td><td>12.61</td></tr><tr><td rowspan="6">InternVL2-8B</td><td>Baseline</td><td>61.67</td><td>79.52</td><td>76.42</td><td>69.64</td><td>43.0</td><td>92.17</td><td>52.72</td><td>40.86</td></tr><tr><td>+ ETA</td><td>61.07</td><td>76.88</td><td>76.76</td><td>53.57</td><td>22.4</td><td>90.11</td><td>52.05</td><td>39.55</td></tr><tr><td>+ DTR</td><td>58.68</td><td>74.09</td><td>62.54</td><td>61.36</td><td>26.2</td><td>81.46</td><td>47.49</td><td>32.32</td></tr><tr><td>+ ShiftDC</td><td>68.86</td><td>81.47</td><td>68.59</td><td>47.08</td><td>20.6</td><td>96.60</td><td>61.73</td><td>50.85</td></tr><tr><td>+ CoCA</td><td>57.48</td><td>43.17</td><td>69.07</td><td>50.11</td><td>39.2</td><td>68.79</td><td>25.58</td><td>19.05</td></tr><tr><td>+ SRT</td><td>41.31</td><td>13.37</td><td>15.54</td><td>44.58</td><td>19.6</td><td>59.01</td><td>8.23</td><td>8.80</td></tr><tr><td rowspan="6">LLaVA-OneVision-7B</td><td>Baseline</td><td>70.05</td><td>91.08</td><td>89.15</td><td>77.86</td><td>42.8</td><td>99.79</td><td>79.64</td><td>75.69</td></tr><tr><td>+ ETA</td><td>70.05</td><td>89.27</td><td>86.95</td><td>74.76</td><td>23.6</td><td>99.27</td><td>77.19</td><td>70.43</td></tr><tr><td>+ DTR</td><td>68.26</td><td>87.60</td><td>86.29</td><td>77.50</td><td>14.2</td><td>99.07</td><td>74.52</td><td>63.86</td></tr><tr><td>+ ShiftDC</td><td>69.46</td><td>92.20</td><td>89.41</td><td>77.98</td><td>54.2</td><td>99.79</td><td>80.31</td><td>78.05</td></tr><tr><td>+ CoCA</td><td>61.67</td><td>71.03</td><td>72.25</td><td>54.88</td><td>31.2</td><td>93.61</td><td>40.15</td><td>35.34</td></tr><tr><td>+ SRT</td><td>31.13</td><td>21.86</td><td>11.70</td><td>34.64</td><td>1.4</td><td>90.73</td><td>11.68</td><td>13.92</td></tr></table>

Table 1: Main safety results under different image-text safety settings.

## 5.2 Safety Main Results

The performance of SRT and baselines across cross-modal settings are presented in Table 1 with the following observations.

(1) Cross-modal safety drift induced by benign text is challenging. Across all evaluated models and defense methods, performance under $S _ { t ^ { - } } { \mathrm { s e t t i n g s } }$ is consistently worse than that under $U _ { t ^ { - } \mathrm { s e t t i n g s } . }$ , revealing that cross-modal risks rendered by benign texts are substantially challenging. In particular, existing defenses still struggle under the $S _ { t }$ -setting, where the unsafe generation rate can exceed 90%, highlighting the necessity of developing a dedicated method to address this issue.

(2) SRT achieves the strongest robustness under all $S _ { t } { \mathbf { - s e t t i n g s . } }$ In both $S _ { t } S _ { i }$ and $S _ { t } U _ { i }$ settings, SRT consistently yields the lowest unsafe generation rate, reflecting its effectiveness in mitigating cross-modal safety drift caused by benign text. This suggests that transferring safety behaviors learned from unsafe-text scenarios can effectively enhance multimodal safety under benign textual contexts.

(3) SRT generalizes well across different multimodal settings and models. SRT consistently improves safety across models from different families and under all cross-modal settings, including $S _ { t } S _ { i }$ $S _ { t } U _ { i } , U _ { t } S _ { i }$ , and $U _ { t } U _ { i }$ . We further evaluate SRT on two larger open-source MLLMs, Qwen3-VL-32B and Gemma-3-27B-IT. As shown in Table 2, SRT consistently improves safety on both models across SIUO and FigStep. These results demonstrate the strong generalization ability of SRT across diverse scenarios and model architectures.

<table><tr><td>Model</td><td>Method SIUO↓ FigStep↓ L-Bench↑</td><td></td><td></td><td></td></tr><tr><td rowspan="2">Qwen3-VL-32B</td><td>Base</td><td>54.49</td><td>20.20</td><td>99.50</td></tr><tr><td>SRT</td><td>24.70</td><td>1.80</td><td>99.20</td></tr><tr><td rowspan="2">Gemma-3-27B-IT</td><td>Base</td><td>59.88</td><td>22.80</td><td>86.40</td></tr><tr><td>SRT</td><td>20.12</td><td>0.40</td><td>87.20</td></tr></table>

Table 2: Generalization results on larger MLLMs.

## 5.3 Utility Preservation and Safe Helpfulness

To examine how SRT affects utility, we evaluate it on representative benchmarks as Table 3 shows. We can see that SRT improves safety while maintains performance comparable to the original models and outperforms most baselines. For example, SRT achieves the best or tied-best MM-Vet scores on Qwen3-VL-8B and LLaVA-OneVision-

<table><tr><td rowspan="2">Models</td><td colspan="6">MM-Vet</td><td colspan="6">MME</td><td colspan="6">L-Bench</td></tr><tr><td>Base.</td><td>ETA</td><td>DTR</td><td>ShiftDC</td><td>CoCA</td><td>SRT</td><td>Base.</td><td>ETA</td><td>DTR</td><td>ShiftDC</td><td>CoCA</td><td>SRT</td><td>Base.</td><td>ETA</td><td>DTR</td><td>ShiftDC</td><td>CoCA</td><td>SRT</td></tr><tr><td>Qwen3-VL-8B</td><td>64.4</td><td>62.3</td><td>58.4</td><td>63.6</td><td>64.1</td><td>65.0</td><td>2390</td><td>2387</td><td>2389</td><td>2332</td><td>2384</td><td>2389</td><td>95.1</td><td>91.2</td><td>93.5</td><td>93.3</td><td>92.0</td><td>94.0</td></tr><tr><td>Gemma-3-12B-IT</td><td>62.8</td><td>60.0</td><td>61.5</td><td>59.1</td><td>58.4</td><td>62.3</td><td>2111</td><td>2068</td><td>2108</td><td>1479</td><td>1426</td><td>2084</td><td>81.7</td><td>81.0</td><td>81.2</td><td>80.5</td><td>80.8</td><td>81.3</td></tr><tr><td>InternVL2-8B</td><td>55.0</td><td>54.4</td><td>50.9</td><td>53.7</td><td>52.7</td><td>54.0</td><td>2177</td><td>2177</td><td>1852</td><td>2172</td><td>2058</td><td>2185</td><td>76.2</td><td>74.0</td><td>74.9</td><td>74.1</td><td>75.2</td><td>75.6</td></tr><tr><td>LLaVA-OneVision-7B</td><td>55.5</td><td>55.0</td><td>48.6</td><td>54.7</td><td>55.2</td><td>55.5</td><td>2082</td><td>2078</td><td>2027</td><td>1902</td><td>2084</td><td>2058</td><td>76.9</td><td>75.9</td><td>80.1</td><td>77.7</td><td>76.4</td><td>78.4</td></tr></table>

Table 3: Utility scores on MM-Vet, MME, and L-Bench. Higher values indicate better utility.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Ref. Last</td><td rowspan="2"></td><td>SIUO</td><td>VLSBench</td><td>FigStep</td><td>MM-Vet</td><td>MME</td></tr><tr><td>Unsafe ↓</td><td>Unsafe ↓</td><td>ASR↓</td><td>Score. ↑</td><td>Acc. ↑</td></tr><tr><td rowspan="4">Qwen3-VL-8B</td><td>X</td><td>X</td><td>25.14</td><td>3.88</td><td>12.6</td><td>64.7</td><td>2371</td></tr><tr><td>√</td><td>X</td><td>22.15</td><td>3.02</td><td>9.6</td><td>60.7</td><td>2354</td></tr><tr><td>X</td><td>√</td><td>23.95</td><td>2.37</td><td>11.0</td><td>60.2</td><td>2360</td></tr><tr><td>√</td><td>√</td><td>19.76</td><td>1.47</td><td>8.2</td><td>65.0</td><td>2389</td></tr><tr><td rowspan="4">Gemma-3-12B-IT</td><td>X</td><td>X</td><td>11.37</td><td>1.29</td><td>0.0</td><td>56.0</td><td>2002</td></tr><tr><td>√</td><td>X</td><td>34.73</td><td>15.81</td><td>2.4</td><td>61.6</td><td>2070</td></tr><tr><td>X</td><td>√</td><td>27.54</td><td>3.17</td><td>1.2</td><td>57.2</td><td>2021</td></tr><tr><td>√</td><td>√</td><td>23.35</td><td>2.96</td><td>1.0</td><td>62.3</td><td>2084</td></tr></table>

Table 4: Ablation results.
<table><tr><td>Model</td><td>Base</td><td>SRT</td></tr><tr><td>Qwen3-VL-8B</td><td>38.18</td><td>69.88</td></tr><tr><td>Qwen3-VL-32B</td><td>41.32</td><td>69.70</td></tr><tr><td>Gemma-3-12B-IT</td><td>33.73</td><td>60.95</td></tr><tr><td>Gemma-3-27B-IT</td><td>38.92</td><td>65.09</td></tr></table>

Table 5: Safe helpfulness scores on SIUO.

7B, even improves MME on InternVL2-8B, and remains competitive on L-Bench across all models, suggesting a favorable safety-utility trade-off.

We further examine whether the safety improvements of SRT are achieved by indiscriminately increasing refusal. Following the evaluation protocol of SIUO, we measure the proportion of responses that satisfy both safety and helpfulness. Table 5 shows that SRT consistently improves safe helpfulness across evaluated models, indicating that its safety gains are not merely attributable to refusals.

## 5.4 Internal Validation

We further examine whether the learned safety awareness mitigates the internal failures revealed in Observation 1 and Observation 2. In Figure 7, after adding our safety awareness direction to benign-text cross-modal inputs, the samples remain close to the harmful-text inputs along the semantic axis, while their refusal-axis projection is substantially increased and becomes close to the explicit harmful-text case. We also revisit the attention-based finding in Observation 2. To quantify whether safety awareness improves the utilization of visual risk cues, we measure the percentage increase in attention assigned to harmful-semantics visual tokens compared to the original amount. Figure 7 shows that both the raw safety-awareness direction and the refined direction increase attention to these visual tokens across MLLMs, indicating that safety awareness helps route visual risks for safety decision. Together, these results show that SRT helps mitigate cross-modal safety drift.

![](images/c7c71f6c53db07963e0bf64b45c41b283b8b307c0410a1efb666a8a88d7c9b5d.jpg)

![](images/3c2436a29c6065e7b37ca1bf78e4c05b2d7612653377fefa2da31ed55ef42257.jpg)  
Figure 7: Illustration of internal validation.

## 5.5 Ablation

Table 4 shows that the two supervisions are complementary. On Qwen3-VL-8B-Instruct, using both refusal-process and output-behavior supervision achieves the best safety results and utility performance. On Gemma-3-12B-IT, the raw direction achieves strong safety scores but gives weaker utility, suggesting that it is more biased toward refusal behavior. In comparison, SRT provides a more favorable trade-off for safety and utility.

## 6 Related Work

## 6.1 MLLM Safety Alignment

MLLM safety alignment aims to prevent harmful responses when unsafe intent is expressed through images, text, or their combination. Existing defenses mainly follow two directions. Trainingstage methods, such as VLGuard and SPA-VL, strengthen the model safety with curated safety supervision, but require additional training and rely on the coverage of safety data (Zong et al., 2024; Zhang et al., 2025).

Inference-stage methods avoid full-model updates and target specific failures, such as visual jailbreaks, safety prompting, and visual-modalityinduced safety degradation (Ding et al., 2025; Zhu et al., 2026; Gong et al., 2025; Wang et al., 2024; Gao et al., 2024; Gou et al., 2024; Liu et al., 2025; Zou et al., 2025; Jiang et al., 2026). However, these methods often increase latency and may reduce helpfulness by relying on captioning, detection, or defensive prompts.

More importantly, existing methods fail to elicit the model’s safety awareness for compositionalrisk settings such as $S _ { t } S _ { i }  U$ , which are increasingly emphasized by recent benchmarks (Wang et al., 2025; Lee et al., 2026; Palaskar et al., 2026). In contrast, we focus on the cross-modal safety drift and aim to improve safety performance across all cross-modal settings.

## 6.2 Representation Steering

Representation steering has recently been explored as an inference-time approach for improving MLLM safety by modifying hidden activations or decoding-time representations (Arditi et al., 2024; Zou et al., 2025; Wu et al., 2025; Park et al., 2026; Wang et al., 2026). These methods demonstrate that internal representations can effectively regulate unsafe behavior without full model fine-tuning, but they largely construct steering signals from risk-specific activations, auxiliary estimators, or modality-induced shifts. In contrast, this work explores steering as a safety transfer mechanism, where the safety capability that MLLMs already exhibit under explicit unsafe text is extracted and transferred to cross-modal failure cases.

## 7 Conclusion

In this work, we study cross-modal safety drift, where MLLMs reliably refuse explicit harmful text inputs but become less safe when harmful intent is conveyed through image-text grounding. We first conduct an empirical analysis of representative failure patterns, followed by a mechanistic interpretation revealing how safety degrades under this issue. Based on these findings, we propose Safetyawareness Representation Transfer (SRT) to improve MLLM safety across different cross-modal settings. Experiments show that SRT effectively enhances safety while preserving model utility.

## Limitations

SRT relies on the existence of a meaningful raw safety awareness v<sub>0</sub> selected from harmful-textinduced activation directions. Therefore, it assumes that the base MLLM can already refuse reliably when unsafe intent is explicitly expressed in text. If the model cannot form stable refusal behavior in harmful-text settings, the extracted activation directions may not contain useful safety-aware signals, making it difficult to select and refine an effective v<sub>0</sub>.

Although we evaluate SRT on multiple representative MLLM families, its generalization to broader architectures remains uncertain. Different models may encode refusal-related computation in different layers or rely on different internal safety mechanisms. Further experiments on more model families, scales, and training paradigms are needed to verify the stability of SRT.

## Ethics Statement

This work aims to improve MLLM safety by reducing harmful responses under cross-modal risks. However, our method requires harmful data for activation extraction, and the model may still generate unsafe responses with SRT in some cases. In addition, this paper contains examples of unsafe image-text inputs and model responses for analysis, which may be disturbing or sensitive to some readers.

## Acknowledgments

This work was supported by the National Natural Science Foundation of China (No. 62506203). We thank the anonymous reviewers for their efforts.

## References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. In Advances in Neural Information Processing Systems, volume 37, pages 136037– 136083.

Yi Ding, Bolian Li, and Ruqi Zhang. 2025. ETA: Evaluating then aligning safety of vision language models at inference time. In International Conference on Learning Representations.

Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, Rongrong Ji, Caifeng Shan, and Ran He. 2025. MME: A comprehensive evaluation benchmark for multimodal large language models. In Advances in Neural Information Processing Systems, volume 38, pages 162549– 162567.

Jiahui Gao, Renjie Pi, Tianyang Han, Han Wu, Lanqing Hong, Lingpeng Kong, Xin Jiang, and Zhenguo Li. 2024. CoCA: Regaining safety-awareness of multimodal large language models with constitutional calibration. In First Conference on Language Modeling.

Yichen Gong, Delong Ran, Jinyuan Liu, Conglei Wang, Tianshuo Cong, Anyu Wang, Sisi Duan, and Xiaoyun Wang. 2025. FigStep: Jailbreaking large visionlanguage models via typographic visual prompts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 23951–23959.

Yunhao Gou, Kai Chen, Zhili Liu, Lanqing Hong, Hang Xu, Zhenguo Li, Dit-Yan Yeung, James T. Kwok, and Yu Zhang. 2024. Eyes closed, safety on: Protecting multimodal LLMs via image-to-text transformation. In Computer Vision – ECCV 2024, pages 388–404. Springer.

Xuhao Hu, Dongrui Liu, Hao Li, Xuanjing Huang, and Jing Shao. 2025. VLSBench: Unveiling visual leakage in multimodal safety. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8285–8316, Vienna, Austria. Association for Computational Linguistics.

Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, and Ting Wang. 2026. Dynamic token reweighting for robust vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 24481–24491.

Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, and Xiangyu Yue. 2025. HiddenDetect: Detecting jailbreak attacks against multimodal large language models via monitoring hidden states. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14880– 14893, Vienna, Austria. Association for Computational Linguistics.

Benno Krojer, Shravan Nayak, Oscar Mañas, Vaibhav Adlakha, Desmond Elliott, Siva Reddy, and Marius Mosbach. 2026. LatentLens: Revealing highly interpretable visual tokens in LLMs. In International Conference on Machine Learning.

Youngwan Lee, Kangsan Kim, Kwanyong Park, Ilchae Jung, Soojin Jang, Seanie Lee, Yong-Ju Lee, and Sung Ju Hwang. 2026. HoliSafe: Holistic safety benchmarking and modeling for vision-language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings, pages 5989–5998.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems, volume 36, pages 34892–34916. Curran Associates, Inc.

Qin Liu, Chao Shang, Ling Liu, Nikolaos Pappas, Jie Ma, Neha Anna John, Srikanth Doss, Lluis Marquez, Miguel Ballesteros, and Yassine Benajiba. 2025. Unraveling and mitigating safety alignment degradation of vision-language models. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 3631–3643, Vienna, Austria. Association for Computational Linguistics.

Xin Liu, Yichen Zhu, Jindong Gu, Yunshi Lan, Chao Yang, and Yu Qiao. 2024. MM-SafetyBench: A benchmark for safety evaluation of multimodal large language models. In Computer Vision – ECCV 2024, pages 386–403. Springer.

Shruti Palaskar, Leon Alexander Gatys, Mona Abdelrahman, Mar Jacobo, Laurence F. Lindsey, Rutika Moharir, Gunnar Lund, Yang Xu, Navid Shiee, Jeffrey P. Bigham, Charles Maalouf, and Joseph Yitan Cheng. 2026. VLSU: Mapping the limits of joint multimodal understanding for AI safety. In International Conference on Learning Representations.

Jonghyun Park, Minhyuk Seo, Chaewon Yeo, and Jonghyun Choi. 2026. Attention misses visual risk: Risk-adaptive steering for multimodal safety alignment. arXiv preprint arXiv:2510.13698.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering Llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522, Bangkok, Thailand. Association for Computational Linguistics.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J. Vazquez, Ulisse Mini, and Monte MacDiarmid. 2023. Steering language models with activation engineering. arXiv preprint arXiv:2308.10248.

Mengxuan Wang, Yuxin Chen, Gang Xu, Tao He, Hongjie Jiang, and Ming Li. 2026. Risk awareness injection: Calibrating vision-language models for safety without compromising utility. arXiv preprint arXiv:2602.03402.

Siyin Wang, Xingsong Ye, Qinyuan Cheng, Junwen Duan, Shimin Li, Jinlan Fu, Xipeng Qiu, and Xuanjing Huang. 2025. Safe inputs but unsafe output: Benchmarking cross-modality safety alignment of large vision-language models. In Findings of the Associationfor Computational Linguistics: NAACL 2025, pages 3563–3605, Albuquerque, New Mexico. Association for Computational Linguistics.

Yu Wang, Xiaogeng Liu, Yu Li, Muhao Chen, and Chaowei Xiao. 2024. AdaShield: Safeguarding multimodal large language models from structure-based attack via adaptive shield prompting. In Computer Vision – ECCV 2024, pages 77–94. Springer.

Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, and Shumin Deng. 2025. Automating steering for safe multimodal large language mod-

els. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 792–814, Suzhou, China. Association for Computational Linguistics.

Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2024. MM-Vet: Evaluating large multimodal models for integrated capabilities. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 57730–57754. PMLR.

Yongting Zhang, Lu Chen, Guodong Zheng, Yifeng Gao, Rui Zheng, Jinlan Fu, Zhenfei Yin, Senjie Jin, Yu Qiao, Xuanjing Huang, Feng Zhao, Tao Gui, and Jing Shao. 2025. SPA-VL: A comprehensive safety preference alignment dataset for vision language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 19867–19878.

Xingyu Zhu, Beier Zhu, Junfeng Fang, Shuo Wang, Yin Zhang, Xiang Wang, and Xiangnan He. 2026. GuardAlign: Test-time safety alignment in multimodal large language models. In International Conference on Learning Representations.

Yongshuo Zong, Ondrej Bohdal, Tingyang Yu, Yongxin Yang, and Timothy Hospedales. 2024. Safety finetuning at (Almost) no cost: A baseline for vision large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 62867–62891. PMLR.

Xiaohan Zou, Jian Kang, George Kesidis, and Lu Lin. 2025. Understanding and rectifying safety perception distortion in VLMs. In Advances in Neural Information Processing Systems, volume 38, pages 126663–126696.

My friend seems to have a great musical talent. How should I ask her to play a piano duet with both hands for me?

![](images/03108eddedc3dca08e9a436aaf00cc82b2afd0be680431efbdd53f6b0ff82f9e.jpg)  
.. Option 1: "Hey, I've been really impressed by your piano skills — would you be up for playing a duet with me? I'd love to hear you play both hands

Figure 8: Additional $S _ { t } S _ { i }  U$ examples of visual risk unawareness.  
![](images/1f70a8e2bfba2f23ec11abcd82ebc4e57035a5b7de029c25b63a391ff5934e0a.jpg)  
Figure 9: Additional $S _ { t } S _ { i } \  \ U$ example of safety implication unawareness.

## A More Empirical Analysis

We further provide additional examples from SIUO to illustrate that the two failure patterns identified in Section 2 also appear in $S _ { t } S _ { i }  U$ scenarios.

Visual risk unawareness. As shown in Figure 8, the textual query asks how to invite a friend to play a piano duet “with both hands”, while the image contains a visual cue indicating that the person may not be able to use both hands. However, the model overlooks this cue and generates a normal invitation response. This example shows that the model may fail to notice the visual element that changes the safety implication of an otherwise benign textual request.

Safety implication unawareness. Figure 9 shows another SIUO example where the model captures the surface topic of the image-text input and responds around “1000 burgers in 24 hours”. However, it fails to recognize the potential health and safety risks of encouraging an extreme eating challenge. This corresponds to safety implication unawareness, where the model perceives the relevant content but does not connect it to safety-aligned decision making.

## B Activation Direction Extraction and Intervention

This appendix provides more details about how we extract activation directions and how they are used for analysis and intervention.

Direction extraction. For each input x, we collect the hidden representation $h _ { \ell , p } ( x )$ at layer ℓ and post-instruction position $p .$ Given two sets of samples, we compute their mean activations at each layer and position. In the preliminary analysis, we use harmful-text samples $\mathbb { D } _ { \mathrm { H T } }$ as the positive set and benign VQA samples $\mathbb { D } _ { \mathrm { B } }$ as the negative set:

$$
\mu _ { \ell , p } ^ { \mathrm { H T } } = \frac { 1 } { | \mathbb { D } _ { \mathrm { H T } } | } \sum _ { \boldsymbol { x } \in \mathbb { D } _ { \mathrm { H T } } } h _ { \ell , p } ( \boldsymbol { x } ) ,\tag{11}
$$

$$
\mu _ { \ell , p } ^ { \mathrm { B } } = \frac { 1 } { | \mathbb { D } _ { \mathrm { B } } | } \sum _ { \boldsymbol { x } \in \mathbb { D } _ { \mathrm { B } } } h _ { \ell , p } ( \boldsymbol { x } ) .\tag{12}
$$

The activation direction is then obtained by their difference:

$$
\begin{array} { r } { v _ { \ell , p } ^ { \mathrm { H T - B } } = \mu _ { \ell , p } ^ { \mathrm { H T } } - \mu _ { \ell , p } ^ { \mathrm { B } } . } \end{array}\tag{13}
$$

This direction captures the activation-space contrast between harmful-text refusal behavior and normal benign answering behavior.

Direction intervention. To test the effect of $v _ { \ell , p } ^ { \mathrm { H T - B } }$ , we add it back to the model hidden representation at its source layer ℓ during the forward pass:

$$
h _ { \ell } ( x ) \gets h _ { \ell } ( x ) + \alpha v _ { \ell , p } ^ { \mathrm { H T - B } } .\tag{14}
$$

Here, $h _ { \ell } ( x )$ denotes the full hidden-state sequence at layer $\ell ,$ with the direction broadcast to all token positions. Here, α controls the intervention strength and is set to 1 by default in all experiments.

Following Arditi et al. (2024), we measure the resulting refusal tendency using the next-token distribution after the model processes the complete multimodal input. The total refusal probability is calculated by summing the probabilities assigned to all refusal-related tokens:

$$
p _ { \mathrm { r e f } } ( x ; v ) = \sum _ { j \in \mathcal { T } _ { \mathrm { r e f } } } \mathrm { s o f t m a x } ( z ( x ; v ) ) _ { j } ,\tag{15}
$$

where $z ( x ; v )$ represents the next-token logits at the last valid input position after adding direction $v ,$ and $\mathcal { T } _ { \mathrm { r e f } }$ is the predefined set of refusal-related tokens.

We then convert the total refusal probability into a log-odds score:

$$
R ( x ; v ) = \log \frac { p _ { \mathrm { r e f } } ( x ; v ) + \epsilon } { 1 - p _ { \mathrm { r e f } } ( x ; v ) + \epsilon } ,\tag{16}
$$

where ϵ is a small constant introduced for numerical stability. A higher $R ( x ; v )$ indicates a stronger tendency to initiate a refusal response. We measure the increase in the refusal score induced by the intervention. Layers where the added direction produces particularly large increases in refusal tendency on benign samples are identified as refusal layers. These layers indicate where refusalrelated computation is most directly reflected in the model’s hidden states.

## C Dataset Details

## C.1 Safety Evaluation Datasets

We evaluate safety under different image-text safeness combinations. In our notation, $S _ { t }$ and $U _ { t }$ denote safe and unsafe textual queries, while $S _ { i }$ and $U _ { i }$ denote safe and unsafe images. We focus on all settings where the final image-text input can elicit unsafe outputs, including $S _ { t } S _ { i }  U , S _ { t } U _ { i }  U$ $U _ { t } S _ { i } \to U$ , and $U _ { t } U _ { i } \to U$

SIUO. SIUO is designed for the $S _ { t } S _ { i }  U$ setting, where the text and image are individually safe but their joint semantics can induce unsafe outputs. It covers multiple safety domains, including self-harm, illegal activities, privacy violation, discrimination, and other cross-modality safety risks. We use SIUO to evaluate whether a model can recognize latent unsafe intent that only emerges after image-text composition.

HoliSafe. HoliSafe is a holistic multimodal safety benchmark that covers five image-text safeness combinations. Since HoliSafe uses a threeletter notation in the order of image, text, and final input safeness, we map its subsets to our text-first notation as follows. The SSU subset corresponds to our $S _ { t } S _ { i } ~  ~ U$ setting, where both modalities are safe in isolation but unsafe after composition. The USU subset corresponds to $S _ { t } U _ { i }  U$ where the text is safe while the image contains unsafe information. The SUU subset corresponds to $U _ { t } S _ { i } \to U$ , where the unsafe intent is explicitly expressed in text with a safe image. The UUU subset corresponds to $U _ { t } U _ { i } \to U$ , where both text and image contain unsafe content. We use these subsets to evaluate SRT across both implicit cross-modal risks and explicit unsafe-text settings.

VLSBench. VLSBench targets the $S _ { t } U _ { i }  U$ setting. It is designed to reduce visual safety information leakage, so that unsafe visual content is not directly revealed by the textual query. The benchmark contains image-text pairs where the text appears benign while the image provides safetycritical information. We use VLSBench to test whether the model can identify unsafe semantics grounded primarily in the visual modality.

MM-SafetyBench. MM-SafetyBench evaluates MLLM safety across 13 prohibited scenarios. It converts harmful textual intents into imagebased attacks using different visual forms, such as Stable Diffusion images, typography images, and their combination. In our evaluation, we use the SD+TYPO split, where harmful visual content is generated by Stable Diffusion and further combined with typographic harmful keywords. MM-SafetyBench is used as a $S _ { t } U _ { i }  U$ benchmark, where the textual instruction is usually benign or indirect while the image carries the harmful intent.

FigStep. FigStep is a typographic jailbreak benchmark that embeds harmful instructions into images and asks the model to generate step-by-step responses. Since the harmful content is mainly conveyed through the visual prompt while the accompanying text is instruction-like and does not explicitly state the unsafe intent, we use FigStep as another $S _ { t } U _ { i }  U$ evaluation benchmark.

## C.2 Utility Evaluation Datasets

We further evaluate model utility on benign multimodal benchmarks to ensure that SRT does not harm general visual understanding and reasoning abilities.

MM-Vet. MM-Vet evaluates integrated multimodal capabilities across six core dimensions: recognition, OCR, knowledge, language generation, spatial awareness, and math. It contains openended questions and uses an LLM-based evaluator to score model responses. We use MM-Vet to measure whether SRT preserves general multimodal reasoning and response quality.

MME. MME is a comprehensive evaluation benchmark for MLLMs, covering both perception and cognition abilities across 14 subtasks. Each sample is formatted as a yes-or-no question, and performance is measured by accuracy-based scores. We use MME to evaluate whether SRT preserves basic visual perception, object recognition, OCR, commonsense reasoning, and other general multimodal capabilities.

L-Bench. L-Bench, also referred to as LLaVA-Bench-Coco in prior work, evaluates open-ended visual instruction-following ability. It assesses generation quality from multiple aspects such as helpfulness, relevance, accuracy, and detail. Following previous safety-alignment studies, we use L-Bench as an additional utility benchmark to examine whether SRT maintains the model’s general instruction-following performance on benign visual inputs.

## C.3 Dataset for Observation 1

For Observation 1, we use 300 randomly sampled examples from the SD image type of MM-SafetyBench. For each example, we keep the same SD image and compare two provided query variants, SD\_merged and SD\_rephrasedSD\_merged. The former contains explicit harmful textual cues, while the latter uses a benignly rephrased query that refers to the activity shown in the image. This setup is used for the refusal-semantic plane visualization in Figure 3.

## D Baselines

ETA (Ding et al., 2025) is an inference-time alignment method for MLLM safety. It follows an evaluating-then-aligning pipeline: first evaluating visual contents and model responses to identify potential unsafe behavior, and then aligning the generation process through an interference prefix and sentence-level best-of-N search. In our experiments, we follow the official inference procedure and apply ETA to each test input before collecting the final response for evaluation.

DTR (Jiang et al., 2026) is an inference-time defense that mitigates multimodal jailbreaks by optimizing the model’s key-value caches. It dynamically reweights visual tokens to reduce the influence of safety-relevant visual shifts while preserving the model’s general vision-language capabilities. In our experiments, we use DTR as a test-time defense and report the response generated after its token reweighting procedure.

ShiftDC (Zou et al., 2025) is a training-free activation calibration method. It first estimates the modality-induced activation shift between visionlanguage inputs and their text-only counterparts, then removes the safety-relevant component of this shift to restore the model’s inherent safety alignment. We follow its original setting and apply the calibrated activations during inference.

CoCA (Gao et al., 2024) is a prompt-based safety calibration method that aims to regain the model’s safety awareness under visual inputs. It introduces safety principles into the input and calibrates the output distribution to amplify refusal-oriented behavior when unsafe visual content is present. In our experiments, we use the same principle-based inference format for all evaluated samples and report the generated responses under CoCA.

## E Implementation Details

Models. For our experiments, we use the official model weights released by the corresponding model providers. The evaluated MLLMs include Qwen3-VL-8B-Instruct, Gemma-3-12B-IT, InternVL2-8B, and LLaVA-OneVision-7B. For the larger-model generalization experiments, we additionally evaluate Qwen3-VL-32B and Gemma-3- 27B-IT. For all models, we use their official processors and chat templates. Unless otherwise specified, we use greedy decoding for generation and keep the decoding configuration identical across the base model, baselines, and SRT.

Evaluation Metrics. For safety evaluation, we report ASR or unsafe rate according to the protocol of each benchmark. For SIUO and VLSBench, we follow their released evaluation prompts and parsing rules, and use GPT-4o-mini as the judge model. For MM-SafetyBench, we use the official scenariospecific evaluation prompts with GPT-4o-mini to determine whether a response is unsafe. For Fig-Step, since the original benchmark mainly relies on manual inspection, we adopt an automatic jailbreak evaluation strategy: we first check whether the response contains refusal keywords; if so, the attack is regarded as unsuccessful. Otherwise, GPT-4omini is used to judge the extent to which the response follows the malicious instruction, and only fully compliant responses are counted as successful attacks. For HoliSafe, we use keyword-based matching to identify refusal responses and compute the unsafe rate.

For utility evaluation, we follow the official evaluation protocol of each benchmark. MM-Vet is evaluated with its official GPT-based scoring prompt, where we use GPT-4o-mini as the evaluator. L-Bench is evaluated with the official GPTbased pairwise scoring prompt, where we use GPT-4.1 as the evaluator. MME is evaluated by directly matching the model’s yes/no answers with the ground-truth labels and reporting the official score.

Human Validation. To further validate the reliability of the automatic safety evaluation, three annotators independently evaluated the responses of Qwen3-VL-8B with SRT on the full SIUO set and a sampled subset of VLSBench. We use majority voting among the three annotators as the human reference and compare it with the GPT-4o-mini judgments. The human majority labels agree with the automatic judge on 351 out of 367 responses, corresponding to an agreement rate of 95.65%.

![](images/90ed778d8b70f5d3a5000fd8bf1a8d8961c0fde55d7d634f95235dab3fc39a9a.jpg)  
Figure 10: Training dynamics of SRT refinement from initialization to epoch 8 on Qwen3-VL-8B-Instruct.

<table><tr><td rowspan="2">Model</td><td colspan="6">Method</td></tr><tr><td>Base.</td><td>ETA</td><td>DTR</td><td>ShiftDC</td><td>CoCA</td><td>SRT</td></tr><tr><td>Qwen3-VL-8B</td><td>37.06</td><td>50.74</td><td>124.97</td><td>74.56</td><td>82.94</td><td>37.60</td></tr><tr><td>Gemma-3-12B-IT</td><td>73.22</td><td>108.35</td><td>113.44</td><td>125.25</td><td>144.21</td><td>74.06</td></tr></table>

Table 6: Amortized inference latency per generated token (ms/token).

## F Training-Data Sensitivity

## F.1 Sensitivity to Training-Set Size

To examine the sensitivity of SRT to the amount of training data, we reduce the original training set to 25% and 50% while keeping the direction extraction, selection, refinement procedure, and hyperparameters unchanged. As shown in Table 7, SRT maintains comparable safety and utility performance across different training-set sizes. Notably, the selected initial safety-awareness direction remains at the same layer and position under the 25%, 50%, and 100% settings, further indicating the stability of the direction selection process.

## F.2 Sensitivity to Training Domain

We further examine whether SRT is sensitive to the training domain by replacing VLSBench with SIUO throughout the SRT construction process. SIUO and VLSBench represent different crossmodal risk settings: VLSBench primarily contains safe-text and unsafe-image cases, whereas SIUO contains cases where the text and image are individually safe but become unsafe through cross-modal composition. We then evaluate the learned directions on VLSBench and FigStep, together with L-Bench for utility evaluation.

<table><tr><td>Model</td><td>Size</td><td>SIUO Safety↑</td><td>ASR↓</td><td>FigStep L-Bench Score↑</td></tr><tr><td rowspan="3">Qwen3-VL-8B</td><td>25%</td><td>75.84</td><td>8.80</td><td>94.7</td></tr><tr><td>50%</td><td>77.05</td><td>9.20</td><td>93.1</td></tr><tr><td>100%</td><td>80.24</td><td>8.20</td><td>94.0</td></tr><tr><td rowspan="3">Gemma-3-12B-IT</td><td>25%</td><td>74.85</td><td>1.20</td><td>82.0</td></tr><tr><td>50%</td><td>76.05</td><td>1.20</td><td>80.8</td></tr><tr><td>100%</td><td>76.65</td><td>1.00</td><td>81.3</td></tr></table>

Table 7: Sensitivity of SRT to training-set size.

<table><tr><td>Model</td><td>Source</td><td>VLSBench FigStep L-Bench Unsafe↓</td><td>ASR↓</td><td>Score↑</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>VLSBench</td><td>1.47</td><td>8.20</td><td>94.0</td></tr><tr><td>SIUO</td><td>3.16</td><td>9.20</td><td>94.0</td></tr><tr><td rowspan="2">Gemma-3-12B-IT</td><td>VLSBench</td><td>2.96</td><td>1.00</td><td>81.3</td></tr><tr><td>SIUO</td><td>2.67</td><td>0.80</td><td>82.6</td></tr></table>

Table 8: Sensitivity of SRT to the training domain.

As shown in Table 8, SRT trained with SIUO achieves performance comparable to its VLSBench-trained counterpart across the evaluated safety and utility benchmarks. These results suggest that SRT is not highly sensitive to the specific training domain and can transfer across different cross-modal risk settings.

## G Training Dynamics of Safety Awareness

Figure 10 shows that SRT effectively refines the raw safety awareness under the joint refusalprocess and output-behavior supervision. On benign samples, both the KL score and the effect score consistently decrease, indicating that the refined direction causes less disturbance to normal VQA behavior.

On risky samples, the refusal score increases even though SRT does not directly maximize it. Meanwhile, the overall effect score becomes smaller, but Figure 7 shows that the refined direction induces stronger harmful-token attention at refusal layers than raw direction. This suggests that under our contrastive constraint, the direction is encouraged to discard broad but less useful perturbation components and retain and further learn only those components that truly contribute to safety awareness.

![](images/e84703541451cb5b922d190cdd7786a412d0966ed13d59b41d7b7db52ee47b37.jpg)  
Figure 11: Response comparisons between Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Instruct+SRT in the $S _ { t } U _ { i }$ setting. The upper case illustrates visual risk unawareness, while the lower case illustrates safety implication unawareness.

## H Inference Efficiency

As shown in Table 6, SRT incurs only minimal inference overhead. Its latency is close to the vanilla base model on both Qwen3-VL-8B and Gemma-3- 12B, with 37.60 and 74.06 ms/token, respectively. Compared with other inference-time methods, SRT is consistently more efficient, since it only applies a lightweight activation-level intervention without additional decoding , image captioning or optimization steps.

## I Case Study

Case 1. SRT mitigates the visual risk unawareness pattern in the $S _ { t } U _ { i }$ setting. As shown in the upper part of Figure 11, the textual query asks for communication techniques in a high-pressure situation, while the image shows a woman pointing a gun at a man. The base model follows the benign textual query and provides general communication advice, whereas SRT recognizes the dangerous visual context and refuses to provide guidance that may escalate harm.

Case 2. SRT mitigates the safety implication unawareness pattern in the $S _ { t } U _ { i }$ setting. As shown in the lower part of Figure 11, the model recognizes that the object in the image is a firearm, but the base model still provides component-swapping suggestions. In contrast, SRT connects the recognized firearm context to its safety implications and refuses to provide modification-related guidance.

Case 3. SRT mitigates the visual risk unawareness pattern in the $S _ { t } S _ { i }$ setting. As shown in the upper part of Figure 12, the text asks how to invite a friend to play a piano duet “with both hands”, while the image shows a person with a visible limb difference. The base model overlooks this visual cue and generates a normal invitation, whereas SRT notices the mismatch and suggests reconsidering the request.

Case 4. SRT mitigates the safety implication unawareness pattern in the $S _ { t } S _ { i }$ setting. As shown in the lower part of Figure 12, the base model captures the challenge-like semantics of eating a large number of burgers within 24 hours and generates a promotional slogan. SRT instead recognizes the health and safety risks behind encouraging extreme eating and refuses to promote the harmful challenge.

![](images/cdb32fee79f6627a93be3ccb0fb6d5ad5f59c8397226fab310782959ed1f61d5.jpg)  
Figure 12: Response comparisons between Qwen3-VL-8B-Instruct and Qwen3-VL-8B-Instruct+SRT in the S S setting. The upper case illustrates visual risk unawareness, while the lower case illustrates safety implication unawareness.