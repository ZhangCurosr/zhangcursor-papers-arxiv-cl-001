# TRACESTEER: Trace Driven Activation Steering for Hallucination Mitigation in Diffusion Language Models

Xin Zhang<sup>1,\*</sup>, Yili Wang<sup>1,\*</sup>, Yue Tan<sup>2</sup>, Xin He<sup>1</sup>, Rui Miao<sup>1</sup>, Yixin Liu<sup>2,†</sup>, Yi Chang<sup>1</sup>, Xin Wang<sup>1,†</sup>

<sup>1</sup>Jilin University, China <sup>2</sup>Griffith University, Australia

## Abstract

Diffusion large language models (dLLMs) have emerged as a promising alternative to autoregressive LLMs, yet remain susceptible to hallucination. Activation steering offers a promising mitigation approach by guiding parallel token predictions through internal representations. However, existing steering methods largely overlook two characteristics of dLLMs: dynamic denoising states and temporal hallucination risks, which challenge predefined steering vectors and uniform intervention strengths. We propose TRACESTEER, a trace-driven activation steering framework that uses preceding denoising predictions as feedback to jointly adapt steering direction and strength. Specifically, Denoising Feedback Steering (DF-Steering) constructs stepspecific steering vectors from preceding predictions and current hidden states. Trajectory-Guided Steering Gate (TS-Gate) regulates their strength using prediction uncertainty and the denoising timestep. With the backbone frozen, factual supervision and magnitude regularization jointly guide the learning of adaptive steering interventions, promoting factual reliability and stability throughout denoising. Experiments on two dLLMs show improved factual reliability on TruthfulQA and generalization across five downstream benchmarks. Our code is available at https://github.com/ zhangxincode/TRACESTEER.

## 1 Introduction

Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive large language models (LLMs), demonstrating strong potential in a wide range of language understanding and generation tasks (Nie et al., 2026; Bie et al., 2025; Ye et al., 2025). Unlike the predominant autoregressive paradigm that generates text sequentially from left to right (Liu et al., 2024; Achiam et al., 2023), dLLMs generate text through iterative denoising. In dLLMs, generation starts from a masked sequence, with the model predicting masked tokens in parallel at each step and remasking selected tokens for subsequent prediction. Such a generation paradigm enables bidirectional context interaction and joint token refinement, facilitating global dependency modeling and efficient parallel generation (Sahoo et al., 2024; Zhong et al., 2026b). Despite their merits, dLLMs, like conventional LLMs, remain susceptible to the hallucination issue, posing the risk of generating incorrect or unsupported content that appears plausible. The hallucination issue undermines the reliability and trustworthiness of dLLMs, leading to potentially misleading outputs in real-world applications (Qian et al., 2026; Zhang et al., 2026).

![](images/c83d87486a0843bcaa9a383a44667d1c6ce0c9eb5cf9fbe11589db76a28148e9.jpg)  
Figure 1: Motivation for trace-driven activation steering in dLLMs. Existing methods in LLMs suffer from (a) reliance on external knowledge, (b) incompatibility with dLLMs’ iterative denoising process, and (c) inability to adapt fixed steering vectors to evolving generation states. In contrast, (d) our approach enables dynamic, trace-aware activation steering.

To alleviate this issue, hallucination mitigation seeks to reduce hallucinated content and improve model reliability by intervening in generation. Nevertheless, most existing methods are developed for autoregressive LLMs and cannot be broadly applied to dLLMs (You et al., 2026; Su et al., 2025). Specifically, retrieval augmentation (Fig. 1a) leverages external knowledge to ground generation (Gao, 2023; Zhuang et al., 2026); despite its effectiveness, maintaining external knowledge sources incurs additional costs and may not suit diverse scenarios. Decoding intervention (Fig. 1b) represents another line of methods that adjust token logits during generation (Zhang et al., 2025; Shi et al., 2024); however, they are designed for a single next-token decision and thus cannot directly accommodate dLLMs, where multiple token distributions are produced simultaneously and only a subset is selected for update. In contrast, activation steering (Fig. 1c), which modulates model behavior by manipulating internal representations during inference, shows promising potential for hallucination mitigation in dLLMs (Li et al., 2023; Zhang et al., 2024). By directly manipulating internal representations, activation steering can guide multiple token predictions simultaneously, making it naturally compatible with diffusion-based parallel generation process.

Despite its promise, most activation steering techniques for hallucination mitigation are tailored to autoregressive LLMs, overlooking several unique characteristics of dLLM generation (Rimsky et al., 2024; Li et al., 2025). Characteristic ❶: Dynamic denoising states. dLLM generation evolves along a continuous denoising trajectory: at different denoising steps, the model not only exhibits varying generation states, but also continuously forms and revises its generation hypotheses. (Wang et al., 2026a; Zhong et al., 2026a) For example, when answering “Why did the Roman Empirefall”, some denoising steps may need to emphasize military pressure, while others shift toward political fragmentation. Facing such dynamic generation states, existing steering methods typically reuse a predefined steering vector throughout inference, which may fail to provide context-adaptive guidance along the evolving denoising trajectory. Characteristic ❷: Temporal hallucination risks. Different stages of dLLM generation play distinct roles and exhibit different sources of hallucination. Early denoising steps establish coarse generation hypotheses and are more prone to semantic errors, while later steps refine generated content, where excessive intervention may disrupt reliable predictions. (Chang et al., 2026; Weng et al., 2026) In this case, applying a uniform intervention strength throughout generation, as in existing steering methods, may fail to meet the varying steering needs at different generation stages, leading to over-steering or under-steering for hallucination mitigation.

Targeting the above characteristics, we propose TRACESTEER (Fig. 1d), a trace-driven activation steering framework for hallucination mitigation in dLLMs. TRACESTEER uses the trajectory information as feedback to jointly adapt the steering direction and intervention strength. To capture Characteristic ❶, we introduce a Denoising Feedback Steering (DF-Steering) mechanism, which learns adaptive steering signals for different denoising states. Specifically, TRACESTEER encodes the preceding tentative response and relates its representations to the current hidden states. Trained under factual supervision, this mechanism learns a stepspecific steering vector conditioned on both the generation hypotheses and the current generation state. To model Characteristic ❷, we further introduce a Trajectory-Guided Steering Gate (TS-Gate) mechanism that adjusts the intervention strength according to prediction uncertainty and generation stage. To be concrete, TRACESTEER learns a gating function conditioned on the uncertainty of the preceding prediction and the current denoising timestep, allowing the steering strength to adapt throughout the generation trajectory. The two mechanisms are trained jointly with factual target responses and magnitude regularization while keeping the dLLM backbone frozen. Experiments on LLaDA-8B-Instruct and Dream-7B-Instruct demonstrate improved factual reliability on TruthfulQA and generalization across five downstream benchmarks.

## 2 Preliminaries

## 2.1 Diffusion Language Model Generation

We consider masked dLLMs with confidence-based decoding. Given an input prompt $^ { c , }$ generation starts from a fully masked response $\mathbf { x } _ { \mathrm { 0 } }$ and proceeds through $T$ denoising iterations to produce the final response $\mathbf { x } _ { T }$ . Here, $t \in \{ 0 , \ldots , T - 1 \}$ indexes generation iterations, and $\mathbf x _ { t } = [ x _ { t } ^ { 1 } , \dots , x _ { t } ^ { L } ]$ denotes the input state at iteration $t ,$ , with $x _ { t } ^ { i } \in$ $\mathcal { V } \cup \{ \mathrm { [ M A S K ] } \}$ , where L is the response length and $\nu$ is the vocabulary.

At iteration t, the backbone $\Phi$ predicts the masked positions $\mathcal { M } _ { t } \ = \ \{ i \ | \ x _ { t } ^ { i } \ = \ [ \mathrm { M A S K } ] \}$ in parallel. Let $p _ { t } ^ { i } ( v ) = p _ { \Phi } ( x ^ { i } = v \mid c , \mathbf { x } _ { t } )$ denote the token probability at position i. Combining these predictions with the committed tokens yields a complete tentative response $\hat { \mathbf { x } } _ { t } ,$ defined as

![](images/cfc3e299e1de59b05163fdb2e6c25324fd131edbeb7452e7f599d243a957445a.jpg)  
Figure 2: Overview of TRACESTEER. DF-Steering generates a feedback-guided steering vector from generation hypotheses and tentative predictions, while TS-Gate controls its strength based on the denoising timestep and feedback uncertainty. The gated steering vector is injected into dLLMs to mitigate hallucinations during denoising.

$$
\hat { x } _ { t + 1 } ^ { i } = \left\{ \begin{array} { l l } { \arg \operatorname* { m a x } _ { v \in \mathcal { V } } p _ { t } ^ { i } ( v ) , } & { i \in \mathcal { M } _ { t } , } \\ { x _ { t } ^ { i } , } & { i \notin \mathcal { M } _ { t } . } \end{array} \right.\tag{1}
$$

The model then commits a subset of masked positions with the highest prediction confidence according to the denoising schedule. The remaining tentative predictions are remasked, while previously committed tokens are preserved, producing $\mathbf { x } _ { t + 1 }$ . This process repeats until all positions are committed. Thus, each iteration produces both a complete tentative response $\hat { \mathbf { x } } _ { t + 1 }$ and a potentially partially masked state $\mathbf { x } _ { t + 1 }$ . The tentative responses across iterations form a generation trace, capturing the model’s evolving predictions before all tokens are committed.

## 2.2 Hallucination Mitigation in dLLMs

Hallucination mitigation in dLLMs aims to reduce factually incorrect content in the final response x<sub>T</sub> while preserving its relevance and informativeness. We study this task through interventions during denoising, seeking to improve the factuality of token predictions before they are committed to the response.

We consider a supervised setting with a training set $\mathcal { D } = \{ ( c ^ { ( n ) } , \hat { \mathbf { y } } ^ { ( n ) } ) \} _ { n = 1 } ^ { N }$ , where $\mathbf { y } ^ { ( n ) }$ denotes the factual target response for prompt $c ^ { ( n ) }$ . These targets provide supervision for learning the intervention. At inference time, the intervention has access to the prompt $c ,$ the current denoising state $\mathbf { x } _ { t } ,$ and tentative predictions from preceding iterations, but not the factual target y. It must therefore guide factual correction using the model’s own generation information. A summary of related works is given in Appendix A.

## 3 Methodology

Building on the generation trace defined in Section 2.1, we present TRACESTEER, a trace-driven activation steering framework that uses preceding denoising predictions to guide the current generation. At each denoising iteration, the backbone computes hidden representations of the response tokens and transforms them through successive Transformer layers to produce token predictions. As illustrated in Fig. 2, TRACESTEER constructs an intervention from both the preceding denoising prediction and the current hidden states, and applies it at an intermediate layer. Its strength is regulated according to prediction uncertainty and denoising progress. Let $H _ { t } ^ { ( l ) } \in \mathbb { R } ^ { L \times d }$ denote the response hidden states at layer l and iteration t, where d is the hidden dimension. We intervene as follows:

$$
\widetilde { H } _ { t } ^ { ( l ) } = H _ { t } ^ { ( l ) } + \gamma _ { t } \Delta H _ { t } ,\tag{2}
$$

where, for $t \geq 1 , \Delta H _ { t } \in \mathbb { R } ^ { L \times d }$ is the steering vector constructed by DF-Steering from prediction feedback and the current hidden states, and $\gamma _ { t } \in ( 0 , 1 )$ is the intervention strength determined by TS-Gate. At $t = 0$ , no preceding prediction is available, so we leave the hidden states unchanged, i.e., $\widetilde { H } _ { 0 } ^ { ( l ) } = H _ { 0 } ^ { ( l ) }$ . The resulting token predictions provide feedback for the next iteration. For subsequent iterations, the modified hidden states are propagated through the remaining backbone layers to produce the current token predictions. Sections 3.1 and 3.2 detail DF-Steering and TS-Gate, respectively. Both mechanisms are trained jointly under factual supervision while keeping the dLLM backbone frozen.

## 3.1 Denoising Feedback Steering

Unlike a predefined steering vector reused throughout generation, steering in dLLMs needs to account for the generation hypotheses formed and revised along the denoising trajectory. The preceding tentative response $\hat { \mathbf { x } } _ { t - 1 }$ reveals the model’s latest generation hypothesis, but how this feedback should guide the current prediction depends on the current denoising state $H _ { t } ^ { ( l ) }$ . We therefore introduce Denoising Feedback Steering (DF-Steering) to construct a step-specific steering vector by relating the preceding prediction to the current hidden states. DF-Steering first encodes the preceding tentative response into contextual representations, then integrates them with the current hidden states to generate a steering vector under factual supervision.

Prediction Feedback Encoding. The preceding tentative response exposes the model’s predicted content, including predictions at positions that remain masked in the current input $\mathbf { x } _ { t }$ . To use this content for activation steering, we encode it into contextual representations at the same layer as the hidden states to be modified. For $t \geq 1$ , we concatenate the prompt c with the complete tentative response $\hat { \mathbf { x } } _ { t - 1 }$ and pass the sequence through the frozen backbone up to the intervention layer l. Let $\mathcal { E }$ denote this encoding operation, retaining only the hidden states at response positions:

$$
Z _ { t - 1 } = \mathcal { E } \left( c , \hat { \mathbf { x } } _ { t - 1 } \right) ,\tag{3}
$$

where $Z _ { t - 1 } \in \mathbb { R } ^ { L \times d }$ contains the layer-l representations of the preceding tentative response in the context of the prompt. This encoding makes the preceding prediction available as contextual feedback for constructing the intervention.

Steering Vector Construction. DF-Steering translates the preceding generation hypothesis into a corrective direction for the current denoising state.

Rather than directly injecting the encoded prediction, it learns how the current hidden states should draw on this feedback to shape the intervention. Specifically, each response position establishes state-dependent associations with the preceding prediction representations and aggregates their information into a position-specific steering signal. We formulate this feedback-to-steering transformation as:

$$
\Delta H _ { t } = \mathcal { F } _ { \phi } \left( H _ { t } ^ { ( l ) } , Z _ { t - 1 } \right) ,\tag{4}
$$

where ${ \mathcal { F } } _ { \phi }$ is a learnable steering constructor parameterized by ϕ, e.g., a cross-attention mechanism. The current hidden states determine how prediction feedback is incorporated, while factual supervision shapes the resulting activation updates through their effects on subsequent token predictions. This formulation makes the steering direction responsive to both the model’s preceding hypothesis and its current denoising state. Recomputing $\Delta H _ { t }$ at each iteration $t \geq 1$ allows the intervention to evolve with the generation trajectory.

## 3.2 Trajectory-Guided Steering Gate

While DF-Steering adapts the intervention direction to the current generation, denoising stages impose different demands on intervention strength. Early iterations form coarse generation hypotheses, where insufficient steering may leave semantic errors uncorrected. Later iterations refine increasingly committed content, where excessive steering may disturb reliable predictions. Applying a uniform strength across these stages risks both under-steering and over-steering. We introduce Trajectory-Guided Steering Gate (TS-Gate) to learn how strongly to intervene from the denoising stage and the uncertainty of the preceding prediction.

Feedback Uncertainty. The denoising timestep identifies the generation stage, but does not capture how uncertain the model’s predictions are within a particular trajectory. Predictions at the same stage may exhibit different levels of uncertainty, calling for intervention strengths that account for the observed prediction state as well as generation progress. For $t \geq 1$ , we characterize this state using the predictive entropy from the preceding iteration, averaged over its masked positions:

$$
\mathcal { H } _ { t - 1 } = - \frac { 1 } { \vert \mathcal { M } _ { t - 1 } \vert } \sum _ { i \in \mathcal { M } _ { t - 1 } } \sum _ { v \in \mathcal { V } } p _ { t - 1 } ^ { i } ( v ) \log p _ { t - 1 } ^ { i } ( v ) ,\tag{5}
$$

where $\mathcal { M } _ { t - 1 }$ denotes the masked positions in $\mathbf { x } _ { t - 1 }$ This quantity summarizes uncertainty in the predictions used to form $\hat { \mathbf { x } } _ { t - 1 }$ , providing a trajectoryspecific signal beyond the timestep alone.

Intervention Strength Regulation. TS-Gate relates this uncertainty signal to the current denoising stage, allowing similar uncertainty to produce different intervention strengths depending on when correction is applied. We learn this relationship through:

$$
\begin{array} { r } { \gamma _ { t } = \sigma \left( g _ { \psi } \left( t , \mathcal { H } _ { t - 1 } \right) \right) , } \end{array}\tag{6}
$$

where $g _ { \psi }$ is a learnable gating function parameterized by $\psi ,$ implemented as an MLP, and $\sigma$ is the sigmoid function. The scalar $\gamma _ { t }$ scales $\Delta H _ { t }$ in Eq. 2. Under factual supervision, the gate learns to balance corrective intervention with preservation of reliable predictions across denoising stages, without imposing a fixed strength schedule. $D F .$ Steering determines the state-specific steering direction, while TS-Gate regulates its influence along the generation trajectory.

## 3.3 Learning to Steer with Factual Supervision

To guide the model toward more factual generation, we jointly train DF-Steering and TS-Gate using factual target responses, learning both how to construct steering vectors and how strongly to apply them at each denoising stage. We further introduce magnitude regularization to constrain the steering vectors relative to the backbone representations, promoting more stable interventions.

Factual Supervision. Given a training pair $( c , \mathbf { y } ) \in \mathcal { D }$ , where $\mathbf { y } = [ y ^ { 1 } , \dots , y ^ { L } ]$ is the factual target response, we obtain the current denoising state $\mathbf { x } _ { t }$ and the preceding tentative response $\hat { \mathbf { x } } _ { t - 1 }$ from the denoising process at iteration t ≥ 1. $\mathsf { A p - }$ plying Eq. 2 and propagating the modified hidden states through the remaining backbone layers yields token probabilities $\widetilde { p } _ { t } ^ { i } ( v )$ . We supervise these predictions at the currently masked positions:

$$
\mathcal { L } _ { \mathrm { t a s k } } = - \frac { 1 } { \vert \mathcal { M } _ { t } \vert } \sum _ { i \in \mathcal { M } _ { t } } \log \widetilde { p } _ { t } ^ { i } ( y ^ { i } ) ,\tag{7}
$$

where $\mathcal { M } _ { t }$ is the set of masked positions in $\mathbf { x } _ { t } .$ Supervising these positions directs learning toward predictions that remain eligible for commitment. Gradients through the modified hidden states jointly optimize DF-Steering’s steering constructor and TS-Gate’s regulation of intervention strength.

The preceding prediction supplies the generation context for the intervention, while the factual target determines the correction objective.

Magnitude Regularization. Factual supervision encourages corrective updates, but does not explicitly constrain their magnitude. We therefore penalize the steering vector relative to the current hidden states, discouraging disproportionately large updates while accounting for variation in representation scale:

$$
\mathcal { L } _ { \mathrm { r e g } } = \frac { \| \Delta H _ { t } \| _ { F } ^ { 2 } } { \| H _ { t } ^ { ( l ) } \| _ { F } ^ { 2 } + \epsilon } ,\tag{8}
$$

where $\| \cdot \| _ { F }$ denotes the Frobenius norm and ϵ is a small constant for numerical stability. The overall training loss combines factual supervision with magnitude regularization:

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \mathrm { t a s k } } + \lambda \mathcal { L } _ { \mathrm { r e g } } , } \end{array}\tag{9}
$$

where λ controls the regularization strength. The trainable parameters are $\Theta = \{ \phi , \psi \}$ , corresponding to the feedback-to-steering mapping in $D F _ { - }$ Steering and the gating function in TS-Gate, respectively. All backbone parameters remain frozen, while gradients propagate through the remaining backbone layers to optimize Θ. At inference time, TRACESTEER constructs and regulates interventions using denoising feedback without access to factual target responses. Algorithm 1 summarizes the procedure of TRACESTEER in Appendix E

## 4 Experiment

Our experiments address three research questions: RQ1: How effectively does TRACESTEER mitigate hallucinations in dLLMs while preserving response informativeness? RQ2: How do prediction feedback, trajectory-guided gating, and magnitude regularization contribute to its effectiveness? RQ3: Can TRACESTEER generalize across diverse tasks with a single set of learned parameters?

## 4.1 Experimental Setup

Backbones and Baselines. We evaluate TRACESTEER on LLaDA-8B-Instruct (Nie et al., 2026) and Dream-7B-Instruct (Ye et al., 2025). We compare against representative promptingbased methods (Few-shot ICL (Liu et al., 2022)), decoding-time methods (OSCAR (Shah et al., 2026)), and activation-steering methods (CAA (Rimsky et al., 2024) and L2S (Parekh et al.,

Table 1: Performance comparison on TruthfulQA. Colored values indicate changes relative to the vanilla baseline. Best results are shown in bold, and second-best results are underlined.
<table><tr><td rowspan="2">Method</td><td colspan="4">Multiple-Choice</td><td colspan="3">Open-ended Generation</td></tr><tr><td>MC1 (↑)</td><td>MC2 (↑)</td><td>MC3 (↑)</td><td> $\mathbf { A v g . \nabla \left( \uparrow \right) }$ </td><td>TRUE (↑)</td><td>INFO (↑)</td><td>TRUE×INFO (↑)</td></tr><tr><td colspan="8">LLaDA-8B-Instruct</td></tr><tr><td>Vanilla</td><td>22.13</td><td>47.08</td><td>23.45</td><td>30.89</td><td>49.18</td><td>81.96</td><td>40.31</td></tr><tr><td>Few-shot ICL</td><td> $3 0 . 3 2 \substack { \uparrow 8 . 1 9 }$ </td><td> $4 6 . 3 5 _ { \downarrow 0 . 7 3 }$ </td><td> $2 4 . 6 0 \substack { \uparrow 1 . 1 5 }$ </td><td> $3 3 . 7 6 \substack { \mathrm { \Omega } ; \mathrm { \Omega } 2 . 8 7 }$ </td><td> $4 5 . 0 8 \scriptstyle \downarrow 4 . 1 0$ </td><td> $7 9 . 5 0 _ { \downarrow 2 . 4 6 }$ </td><td> $3 5 . 8 4 \downarrow 4 . 4 7$ </td></tr><tr><td>Oscar</td><td> $2 3 . 7 7 \substack { \uparrow 1 . 6 4 }$ </td><td> $4 6 . 9 7 _ { \downarrow 0 . 1 1 }$ </td><td> $2 3 . 6 6 \mathrm { \cdot } 0 . 2 1$ </td><td> $3 1 . 4 7 \mathrm { \uparrow 0 . 5 8 }$ </td><td> $5 1 . 1 8 \mathrm { ‰ }$ </td><td> $8 4 . 3 2 _ { \uparrow 2 . 3 6 }$ </td><td> $4 3 . 1 5 _ { \uparrow 2 . 8 4 }$ </td></tr><tr><td>CAA</td><td> $2 3 . 7 7 \substack { \uparrow 1 . 6 4 }$ </td><td> $\underline { { 4 9 . 7 9 } } _ { \uparrow 2 . 7 1 }$ </td><td> $2 4 . 5 2 \substack { \uparrow 1 . 0 7 }$ </td><td> $3 2 . 6 9 \mathrm { _ { \uparrow 1 . 8 0 } }$ </td><td> $5 0 . 0 0 \mathrm { \Omega } _ { \uparrow 0 . 8 2 }$ </td><td> $8 4 . 4 2 \substack { \uparrow 2 . 4 6 }$ </td><td> $4 2 . 2 1 \substack { \uparrow 1 . 9 0 }$ </td></tr><tr><td>L2S</td><td> $2 2 . 9 5 _ { \uparrow 0 . 8 2 }$ </td><td> $4 6 . 9 6 _ { \downarrow 0 . 1 2 }$ </td><td> $2 3 . 8 3 _ { \uparrow 0 . 3 8 }$ </td><td> $3 1 . 2 5 \mathrm { _ { \mathrm { 1 0 . 3 6 } } }$ </td><td> $5 2 . 4 5 _ { \cdot 3 . 2 7 }$ </td><td> $8 2 . 7 8 \mathrm { _ { \uparrow 0 . 8 2 } }$ </td><td> $\underline { { 4 3 . 4 2 } } \uparrow 3 . 1 1$ </td></tr><tr><td>Ours</td><td> $4 3 . 4 4 \substack { \mathrm { \ t \ 2 1 . 3 1 } }$ </td><td> $\mathbf { 6 1 . 1 9 } _ { \uparrow 1 4 . 1 1 }$ </td><td> $\mathbf { 3 1 . 6 3 } _ { \mathrm { \uparrow 8 . 1 8 } }$ </td><td> $4 5 . 4 2 \substack { \mathrm { \uparrow 1 4 . 5 3 } }$ </td><td> $5 7 . 3 7 \mathrm { \sim } 8 . 1 9$ </td><td> $\mathbf { 8 9 . 3 4 } _ { \ \mathrm { \uparrow } \mathrm { 7 } . 3 8 }$ </td><td> ${ \bar { 5 } } 1 . 2 5 _ { \uparrow 1 0 . 9 4 }$ </td></tr><tr><td colspan="8">Dream-7B-Instruct</td></tr><tr><td>Vanilla</td><td>24.59</td><td>49.92</td><td>24.31</td><td>32.94</td><td>49.18</td><td>70.49</td><td>34.67</td></tr><tr><td>Few-shot ICL</td><td> $3 1 . 1 4 \mathrm { \uparrow } 6 . 5 5$ </td><td> $4 8 . 7 7 _ { \downarrow 1 . 1 5 }$ </td><td> $2 5 . 6 7 \substack { \uparrow 1 . 3 6 }$ </td><td> $3 5 . 1 9 _ { \uparrow 2 . 2 5 }$ </td><td> $4 8 . 4 7 _ { \downarrow 0 . 7 1 }$ </td><td> $5 9 . 8 3 _ { \perp 1 0 . 6 6 }$ </td><td> $2 9 . 0 0 _ { \downarrow 5 . 6 7 }$ </td></tr><tr><td>Oscar</td><td> $2 6 . 2 2 \substack { \uparrow 1 . 6 3 }$ </td><td> $5 1 . 2 4 \substack { \uparrow 1 . 3 2 }$ </td><td> $2 6 . 7 1 \substack { \uparrow 2 . 4 0 }$ </td><td> $3 4 . 7 2 \substack { \uparrow 1 . 7 8 }$ </td><td> $5 1 . 8 2 \textmu m 2 . 6 4$ </td><td> $7 2 . 5 9 _ { \uparrow 2 . 1 0 }$ </td><td> $3 7 . 6 2 \substack { \uparrow 2 . 9 5 }$ </td></tr><tr><td>CAA</td><td> $2 5 . 4 0 \substack { \mathrm { \uparrow 0 . 8 1 } }$ </td><td> $5 0 . 4 1 \substack { \mathrm { \uparrow 0 . 4 9 } }$ </td><td> $2 4 . 5 9 _ { \uparrow 0 . 2 8 }$ </td><td> $3 3 . 4 7 \mathrm { \uparrow 0 } . 5 3 $ </td><td> $\underline { { 5 4 . 0 9 } } \mathrm { \cdot } 4 . 9 1$ </td><td> $\underline { { 7 4 . 5 9 } } \mathrm { \uparrow 4 . 1 0 }$ </td><td> $4 0 . 3 5 \substack { \mathrm { ~ 1 0 . 6 8 } }$ </td></tr><tr><td>L2S</td><td> $2 8 . 6 8 \substack { \uparrow 4 . 0 9 }$ </td><td> $5 4 . 1 0 \mathrm { \uparrow } 4 . 1 8$ </td><td> $2 9 . 8 2 \substack { \mathrm { \uparrow 5 . 5 1 } }$ </td><td> $3 7 . 5 3 \substack { \uparrow 4 . 5 9 }$ </td><td> $5 3 . 8 2 \substack { \uparrow 4 . 6 4 }$ </td><td> $7 1 . 0 5 _ { \uparrow 0 . 5 6 }$ </td><td> $3 8 . 2 4 \substack { \uparrow 3 . 5 7 }$ </td></tr><tr><td>Ours</td><td> ${ \bf 3 6 . 8 8 } _ { \mathrm { \uparrow 1 2 . 2 9 } }$ </td><td> $\pmb { 5 5 . 9 5 } _ { \uparrow 6 . 0 3 }$ </td><td> $2 9 . 5 4 \substack { \mathrm { \uparrow 9 . 2 3 } }$ </td><td> $\mathbf { 4 0 . 7 9 } _ { \uparrow 7 . 8 5 }$ </td><td> $\mathbf { 5 8 . 1 9 } _ { \mathrm { \uparrow 9 . 0 1 } }$ </td><td> $\mathbf { 8 3 . 6 0 } _ { \uparrow 1 3 . 1 1 }$ </td><td> $\mathbf { 4 8 . 6 5 } _ { \uparrow 1 3 . 9 8 }$ </td></tr></table>

2026)). We also report each backbone’s original performance without intervention as Vanilla. More experimental setups can be found in Appendix B and Appendix D.

Evaluation Protocol. We primarily evaluate hallucination mitigation on TruthfulQA (Lin et al., 2021) under both multiple-choice and open-ended generation settings. For multiple-choice evaluation, we report MC1, MC2, MC3, and their arithmetic mean to assess factual discrimination. For open-ended generation, we use Qwen3-8B (Yang et al., 2025) as the LLM judge to assess truthfulness (TRUE) and informativeness (INFO), and report TRUE×INFO as their combined score (Li et al., 2023). We further report accuracy on five downstream benchmarks: COPA (Gordon et al., 2012), StoryCloze (Mostafazadeh et al., 2016), MMLU (Hendrycks et al., 2020), Wino (Sakaguchi et al., 2021), and BoolQ (Clark et al., 2019).

tion. As shown in Table 1, TRACESTEER achieves the best average MC and TRUE×INFO scores on both backbones, consistently outperforming existing baselines. In contrast, few-shot ICL improves answer discrimination but compromises truthfulness and informativeness. These results demonstrate that TRACESTEER enhances factual reliability without sacrificing informative generation. Its superiority over CAA and L2S further highlights the potential benefits of incorporating denoising feedback and trajectory-guided intervention, as examined in our ablation study.

Obs. 2. Broad utility across diverse tasks and backbones. As shown in Table 2, TRACESTEER consistently achieves the best performance across all five diverse benchmarks on both backbones, with particularly strong gains on StoryCloze and MMLU. These results demonstrate the broad utility of TRACESTEER beyond truthfulness-oriented QA, spanning commonsense reasoning, knowledge evaluation, and reading comprehension. Despite variations in improvement magnitude across tasks and backbones, the consistent gains highlight the robustness and adaptability of our approach.

## 4.2 Overall Performance (RQ1)

To answer RQ1, we compare TRACESTEER with prompting-based, decoding-time, and activationsteering baselines on two dLLM backbones. Table 1 reports multiple-choice and open-ended generation performance on TruthfulQA, while Table 2 presents accuracy on five downstream benchmarks. Obs. 1. Factual reliability improves across both answer discrimination and free-form genera-

## 4.3 Ablation Analysis (RQ2)

To answer RQ2, we examine the contributions of DF-Steering, TS-Gate, and magnitude regularization through ablation on TruthfulQA. Table 3 presents the ablation results of TRACESTEER in

Table 2: Performance comparison on diverse task benchmarks. Colored values indicate changes relative to the baseline. Best results are shown in bold, and second-best results are underlined.
<table><tr><td></td><td>Method</td><td>COPA</td><td>StoryCloze</td><td>MMLU</td><td>Wino</td><td>BoolQA</td><td> $\mathbf { A v } \mathbf { g } .$ </td></tr><tr><td rowspan="6">Ila--uct</td><td>Vanilla</td><td>75.40</td><td>70.49</td><td>63.93</td><td>61.47</td><td>84.42</td><td>71.14</td></tr><tr><td>Few-shot ICL</td><td> $\underline { { 8 3 . 6 0 } } \mathrm { \uparrow } 8 . 2 0$ </td><td> $8 3 . 6 0 \mathrm { \Omega } _ { \mathrm { \uparrow 1 3 . 1 1 } }$ </td><td> $8 5 . 2 4 \substack { \uparrow 2 1 . 3 1 }$ </td><td> $5 9 . 0 1 _ { \downarrow 2 . 4 6 }$ </td><td> $8 9 . 3 4 _ { \uparrow 4 . 9 2 }$ </td><td> $\underline { { 8 0 . 1 6 } } \mathrm { \ } _ { \mathrm { ~ 0 . 0 2 } }$ </td></tr><tr><td>Oscar</td><td> $8 1 . 1 4 \substack { \uparrow 5 . 7 4 }$ </td><td> $8 4 . 4 2 _ { \uparrow 1 3 . 9 3 }$ </td><td> $7 9 . 5 0 \substack { \mathrm { \uparrow 1 5 . 5 7 } }$ </td><td> $6 3 . 1 1 \substack { \uparrow 1 . 6 4 }$ </td><td> $8 9 . 3 4 _ { \uparrow 4 . 9 2 }$ </td><td> $7 9 . 5 0 \substack { \mathrm { \sim } 8 . 3 6 }$ </td></tr><tr><td>CAA</td><td> $7 6 . 2 2 \substack { \mathrm { \uparrow 0 . 8 2 } }$ </td><td> $\underline { { 8 9 . 3 4 } } \mathrm { \uparrow 1 8 . 8 5 }$ </td><td> $7 4 . 5 9 _ { \uparrow 1 0 . 6 6 }$ </td><td> $6 2 . 2 9 _ { \uparrow 0 . 8 2 }$ </td><td> $9 0 . 9 8 \substack { \uparrow 6 . 5 6 }$ </td><td> $7 8 . 6 8 \substack { \mathrm { \uparrow } 7 . 5 4 }$ </td></tr><tr><td>L2S</td><td> $7 7 . 0 4 \mathrm { \cdot } 1 . 6 4$ </td><td> $7 7 . 0 4 \mathrm { \uparrow } 6 . 5 5$ </td><td> $7 0 . 4 9 \substack { \uparrow 6 . 5 6 }$ </td><td> $6 3 . 1 1 \substack { \uparrow 1 . 6 4 }$ </td><td> $8 6 . 8 8 \substack { \mathrm { \uparrow } 2 . 4 6 }$ </td><td> $7 4 . 9 1 \substack { \uparrow 3 . 7 7 }$ </td></tr><tr><td>Ours</td><td> $\mathbf { 9 4 . 2 6 } _ { \uparrow 1 8 . 8 6 }$ </td><td> $\mathbf { 9 7 . 5 4 } _ { \mathrm { \uparrow 2 7 . 0 5 } }$ </td><td> $\mathbf { 9 5 . 0 8 } _ { \mathrm { \uparrow 3 1 . 1 5 } }$ </td><td> ${ \bf 6 7 . 2 1 } _ { \uparrow 5 . 7 4 }$ </td><td> $\mathbf { 9 1 . 8 0 } _ { \uparrow 7 . 3 8 }$ </td><td> $\mathbf { 8 9 . 1 8 } _ { \mathrm { \uparrow 1 8 . 0 4 } }$ </td></tr><tr><td rowspan="5">Dresr--Dstuct</td><td>Vanilla</td><td>84.42</td><td>79.50</td><td>71.31</td><td>55.73</td><td>78.68</td><td>73.93</td></tr><tr><td>Few-shot ICL</td><td> $8 5 . 2 4 _ { \uparrow 0 . 8 2 }$ </td><td> $7 3 . 7 7 _ { \downarrow 5 . 7 3 }$ </td><td> $\underline { { 8 9 . 3 4 } } \mathrm { \uparrow 1 8 . 0 3 }$ </td><td> $5 8 . 1 9 _ { \uparrow 2 . 4 6 }$ </td><td> $8 7 . 7 0 \mathrm { \Omega } \mathrm { \Omega }$ </td><td> $7 8 . 8 5 _ { \uparrow 4 . 9 2 }$ </td></tr><tr><td>Oscar</td><td> $8 5 . 2 4 _ { \uparrow 0 . 8 2 }$ </td><td> $8 0 . 3 2 \substack { \uparrow 0 . 8 2 }$ </td><td> $7 3 . 7 7 \substack { \mathrm { \uparrow 2 . 4 6 } }$ </td><td> $5 7 . 3 7 _ { \uparrow 1 . 6 4 }$ </td><td> $8 1 . 1 4 \substack { \uparrow 2 . 4 6 }$ </td><td> $7 5 . 5 7 \substack { \mathrm { \uparrow 1 . 6 4 } }$ </td></tr><tr><td>CAA</td><td> $8 6 . 8 8 \substack { \mathrm { \uparrow } 2 . 4 6 }$ </td><td> $8 2 . 7 8 \mathrm { _ { \circ } } _ { 1 3 . 2 8 }$ </td><td> $8 1 . 1 4 \substack { \mathrm { \uparrow 9 . 8 3 } }$ </td><td> $5 9 . 0 1 _ { \uparrow 3 . 2 8 }$ </td><td> $8 1 . 1 4 \substack { \uparrow 2 . 4 6 }$ </td><td> $7 8 . 1 9 _ { \uparrow 4 . 2 6 }$ </td></tr><tr><td>L2S</td><td> $\underline { { 8 7 . 7 0 } } \mathrm { \uparrow } 3 . 2 8$ </td><td> $8 5 . 2 4 \substack { \uparrow 5 . 7 4 }$ </td><td> $8 6 . 0 6 \substack { \mathrm { \uparrow 1 4 . 7 5 } }$ </td><td> $5 9 . 8 3 \substack { \uparrow 4 . 1 0 }$ </td><td> $8 1 . 9 6 _ { \uparrow 3 . 2 8 }$ </td><td> $\underline { { 8 0 . 1 6 } } _ { \cdot \ : 6 . 2 3 }$ </td></tr><tr><td></td><td>Ours</td><td> $\mathbf { 9 2 . 6 2 } _ { \mathrm { \uparrow 8 . 2 0 } }$ </td><td> $\mathbf { 9 4 . 2 6 } _ { \uparrow 1 4 . 7 6 }$ </td><td> $\mathbf { 9 5 . 9 0 } _ { \uparrow 2 4 . 5 9 }$ </td><td> $\mathbf { 6 3 . 9 3 } \mathrm { \sim } 8 . 2 0$ </td><td> $\mathbf { 9 } 2 . 6 2 \substack { \mathrm { \uparrow } 1 3 . 9 4 }$ </td><td> $\mathbf { 8 7 . 8 7 } _ { \uparrow 1 3 . 9 4 }$ </td></tr></table>

![](images/c027c0828a9381543ede81969d066f3ed944fa17985724840c5ed5e14efc2651.jpg)  
Figure 3: Effect of incorporating multiple previous predictions on model performance.

Table 3: Ablation study of our proposed steering framework on TruthfulQA. Best results are shown in bold, and second-best results are underlined.

<table><tr><td>Method</td><td>MC1</td><td>MC2</td><td>MC3</td><td>TRUE</td><td>INFO</td></tr><tr><td colspan="6">LLaDA-8B-Instruct</td></tr><tr><td>Vanilla</td><td>22.13</td><td>47.08</td><td>23.45</td><td>49.18</td><td>81.96</td></tr><tr><td>w/o DF-Steering</td><td>40.98</td><td>58.47</td><td>27.55</td><td>50.00</td><td>84.42</td></tr><tr><td>w/o TS-Gate</td><td>40.98</td><td>57.32</td><td>30.66</td><td>45.90</td><td>83.60</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e g } }$ </td><td>36.88</td><td>56.92</td><td>30.08</td><td>11.47</td><td>10.65</td></tr><tr><td>Ours</td><td>43.44</td><td>61.19</td><td>31.63</td><td>57.37</td><td>89.34</td></tr></table>

effective hallucination mitigation goes beyond simply strengthening steering interventions: it requires aligning corrections with evolving generation states while carefully regulating their strength to improve factual reliability without compromising generation quality.

## LLaDA-8B-Instruct.

Obs. 3. Effective hallucination mitigation requires adaptive steering and stable intervention control. As shown in Table 7, removing prediction feedback or replacing TS-Gate with a fixed coefficient consistently degrades performance, highlighting the importance of trace-aware steering and adaptive intervention strength. Removing $\mathcal { L } _ { \mathrm { r e g } }$ further reduces truthfulness and informativeness, despite occasional gains in multiple-choice metrics, suggesting a trade-off between answer matching and generation quality. These findings suggest that

Effect of Prediction History. Figure 3 examines whether incorporating additional preceding predictions improves performance.

Obs. 4. Longer prediction histories do not consistently improve steering. Across both backbones, using only the immediately preceding prediction achieves the best overall performance, while longer histories provide no consistent gains. These results suggest that effective steering relies more on feedback relevance to the current denoising state than on accumulated history, as earlier predictions may become less informative as generation evolves.

## 4.4 Cross-Dataset Generalization (RQ3)

To answer RQ3, we train DF-Steering and TS-Gate exclusively on TruthfulQA and directly evaluate them on COPA, StoryCloze, MMLU, WinoGrande, and BoolQ without further adaptation. Figure 4 presents the results on both dLLM backbones.

![](images/c9b5866d07497dbe638f82071ef5d381da45ba55d1858e2069feda0849add604.jpg)  
Figure 4: Cross-dataset generalization on two diffusion language models. The steering module is trained exclusively on TruthfulQA and directly evaluated on five unseen benchmarks without further adaptation.

![](images/0ce70d0fb618a1574e83c80aaa4f23b6a38fd818e7300ce3ceecf0f7f905c546.jpg)  
Figure 5: Accumulated steering trajectories visualized with 3D PCA. Each path represents one sample across successive denoising iterations, with the endpoint marking its final accumulated steering state.

Obs. 5. Learned steering mechanisms transfer across tasks without additional adaptation. TRACESTEER consistently outperforms Vanilla across all five downstream benchmarks on both backbones without target-task supervision. These results demonstrate that the learned traceconditioned steering mechanism, rather than a fixed intervention direction, can adapt to unseen tasks by dynamically constructing and regulating interventions based on their denoising states, highlighting its cross-task transferability.

## 4.5 Visualization of Steering Trajectories

To examine how TRACESTEER adapts interventions during denoising, we visualize step-wise steering vectors using three-dimensional PCA, connecting their projections to form sample-specific trajectories. As shown in Figure 5, the curved trajectories reveal continuously evolving steering directions, while their distinct patterns reflect sample-specific adjustments. These observations demonstrate that TRACESTEER dynamically adapts steering to both denoising progress and individual samples, rather than relying on fixed interventions.

![](images/0b97d43cb7c6d426ded41501ff39027957313af3a0dd5ff5fbc920ecbad2f0c6.jpg)  
Figure 6: Case study of hallucination correction. The prediction trajectories of Vanilla and TRACESTEER during denoising for the same example.

To better understand how TRACESTEER reshapes the denoising trajectory, we visualize representative intermediate predictions in Figure 6. Both methods exhibit incorrect tentative predictions at the early denoising stage. As denoising proceeds, Vanilla continues to reinforce the erroneous hypothesis and eventually commits to a hallucinated response. In contrast, TRACESTEER uses preceding prediction representations as feedback to adapt the steering direction at subsequent steps, gradually correcting the evolving prediction and stabilizing it toward the factual answer.

Additional analyses of intervention layers, computational costs, and more detailed results are provided in Appendix C.

## 5 Conclusion

In this work, we proposed TRACESTEER, a tracedriven activation steering framework for mitigating hallucinations in dLLMs. By leveraging intermediate generation traces, TRACESTEER dynamically adapts steering interventions throughout denoising to improve hallucination. Extensive experiments across multiple dLLM backbones and benchmarks demonstrate its broad utility and cross-task transferability. Our findings highlight the importance of incorporating denoising dynamics into activation steering for more reliable generation.

## Limitations

While TRACESTEER demonstrates effective hallucination mitigation for dLLMs, it still has certain limitations. In particular, our method relies on factual target responses to train the steering and gating modules. Such supervision provides explicit guidance for learning effective steering interventions, but also requires high-quality labeled data. However, obtaining reliable factual annotations can be expensive and time-consuming, especially in specialized domains. This dependence on supervised data may limit the applicability of our method to broader scenarios. In future work, we plan to explore weakly supervised and self-supervised learning strategies to reduce the reliance on factual annotations and improve the generalizability of our approach.

## Ethical Considerations

Our research involves no human subjects, animal experiments, or sensitive personal data. All experiments are conducted using publicly available models and benchmark datasets. The proposed method aims to improve the factual reliability of diffusion large language models by mitigating hallucinated content. While activation steering can modify model generation behavior, our work does not introduce new data collection or deploymentrelated risks beyond those generally associated with large language models. We are committed to following established research ethics and responsible AI practices throughout this study.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Nikhil Anand, Shwetha Somasundaram, Anirudh Phukan, Apoorv Saxena, and Koyel Mukherjee. 2026. Contextfocus: Activation steering for contextual faithfulness in large language models. arXiv preprint arXiv:2601.04131.

Akari Asai, Zeqiu Wu, Yizhong Wang, Avi Sil, and Hannaneh Hajishirzi. 2024. Self-rag: Learning to retrieve, generate, and critique through self-reflection. In International conference on learning representations, volume 2024, pages 9112–9141.

Jacob Austin, Daniel D Johnson, Jonathan Ho, Daniel Tarlow, and Rianne Van Den Berg. 2021. Structured

denoising diffusion models in discrete state-spaces. Advances in neural information processing systems, 34:17981–17993.

Eden Avrahami and Eliya Nachmani. 2026. Ilrr: Inference-time steering method for masked diffusion language models. arXiv preprint arXiv:2601.21647.

Tiwei Bie, Maosong Cao, Kun Chen, Lun Du, Mingliang Gong, Zhuochen Gong, Yanmei Gu, Jiaqi Hu, Zenan Huang, Zhenzhong Lan, et al. 2025. Llada2. 0: Scaling up diffusion language models to 100b. arXiv preprint arXiv:2512.15745.

Shenxu Chang, Junchi Yu, Weixing Wang, Yongqiang Chen, Jialin Yu, Philip Torr, and Jindong Gu. 2026. Tracedet: Hallucination detection from the decoding trace of diffusion large language models. In International Conference on Learning Representations, volume 2026, pages 9472–9491.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In International Conference on Learning Representations, volume 2024, pages 54158–54183.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. 2019. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936.

Shehzaad Dhuliawala, Mojtaba Komeili, Jing Xu, Roberta Raileanu, Xian Li, Asli Celikyilmaz, and Jason Weston. 2024. Chain-of-verification reduces hallucination in large language models. In Findings of the association for computational linguistics: ACL 2024, pages 3563–3578.

Luyu Gao, Zhuyun Dai, Panupong Pasupat, Anthony Chen, Arun Tejasvi Chaganty, Yicheng Fan, Vincent Zhao, Ni Lao, Hongrae Lee, Da-Cheng Juan, et al. 2023. Rarr: Researching and revising what language models say, using language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16477–16508.

Yunfan Gao. 2023. Retrieval-augmented generation for large language models: A survey. arXiv preprint arXiv:2312.10997.

Andrew Gordon, Zornitsa Kozareva, and Melissa Roemmele. 2012. Semeval-2012 task 7: Choice of plausible alternatives: An evaluation of commonsense causal reasoning. In \* SEM 2012: The First Joint Conference on Lexical and Computational Semantics– Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pages 394–398.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in neural information processing systems, 36:41451–41530.

Zhuowei Li, Haizhou Shi, Yunhe Gao, Di Liu, Zhenting Wang, Yuxiao Chen, Ting Liu, Long Zhao, Hao Wang, and Dimitris N Metaxas. 2025. The hidden life of tokens: Reducing hallucination of large vision-language models via visual information steering. arXiv preprint arXiv:2502.03628.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2021. Truthfulqa: Measuring how models mimic human falsehoods, 2022. URL https://arxiv. org/abs/2109.07958, 1.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. 2022. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. Advances in neural information processing systems, 35:1950– 1965.

Nasrin Mostafazadeh, Nathanael Chambers, Xiaodong He, Devi Parikh, Dhruv Batra, Lucy Vanderwende, Pushmeet Kohli, and James Allen. 2016. A corpus and cloze evaluation for deeper understanding of commonsense stories. In Proceedings of the 2016 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 839–849.

Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. 2026. Large language diffusion models. Advances in Neural Information Processing Systems, 38:50608–50646.

Jayneel Parekh, Pegah Khayatan, Mustafa Shukor, Arnaud Dapogny, Alasdair Newson, and Matthieu Cord. 2026. Learning to steer: Input-dependent steering for multimodal llms. Advances in Neural Information Processing Systems, 38:159799–159834.

Yanyu Qian, Yue Tan, Yixin Liu, Wang Yu, and Shirui Pan. 2026. Dynhd: Hallucination detection for diffusion large language models via denoising dynamics deviation learning. arXiv preprint arXiv:2603.16459.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In

Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522.

Subham S Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin, Justin T Chiu, Alexander Rush, and Volodymyr Kuleshov. 2024. Simple and effective masked diffusion language models. Advances in Neural Information Processing Systems, 37:130136–130184.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106.

Yash Shah, Abhijit Chakraborty, Naresh Kumar Devulapally, Vishnu Lokhande, and Vivek Gupta. 2026. Oscar: Orchestrated self-verification and cross-path refinement. arXiv preprint arXiv:2604.01624.

Weijia Shi, Xiaochuang Han, Mike Lewis, Yulia Tsvetkov, Luke Zettlemoyer, and Wen-tau Yih. 2024. Trusting your evidence: Hallucinate less with contextaware decoding. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 783–791.

Adi Shnaidman, Erin Feiglin, Osher Yaari, Efrat Mentel, Amit Levi, and Raz Lapid. 2025. Activation steering for masked diffusion language models. arXiv preprint arXiv:2512.24143.

Jingran Su, Jingfan Chen, Hongxin Li, Yuntao Chen, Qing Li, and Zhaoxiang Zhang. 2025. Activation steering decoding: Mitigating hallucination in large vision-language models through bidirectional hidden state intervention. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12964– 12974.

Chengcheng Wang, Tingzhang Luo, Wenhao Li, Jianyuan Guo, and Chang Xu. 2026a. Tacg: Trajectory-aware commit gating for diffusion language model decoding. arXiv preprint arXiv:2607.03236.

Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, and Chunhua Shen. 2026b. Time is a feature: Exploiting temporal dynamics in diffusion language models. In International Conference on Learning Representations, volume 2026, pages 72909–72935.

Pengcheng Weng, Yanyu Qian, Yue Tan, and Yixin Liu. 2026. Tre: Training-free hallucination detection for diffusion language models.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective retrieval augmented generation. arXiv preprint arXiv:2401.15884.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. 2025. Dream 7b: Diffusion large language models, 2025. URL https://arxiv. org/abs/2508.15487.

Zejia You, Chunyuan Deng, and Hanjie Chen. 2026. Spherical steering: Geometry-aware activation rotation for language models. arXiv preprint arXiv:2602.08169.

Shaolei Zhang, Tian Yu, and Yang Feng. 2024. Truthx: Alleviating hallucinations by editing large language models in truthful space. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8908– 8949.

Xin Zhang, Yili Wang, Yue Tan, Xin He, Yanyu Qian, Yixin Liu, Yi Chang, Shirui Pan, and Xin Wang. 2026. Demts: Denoising trajectories as multivariate time series for hallucination detection in diffusion language models. arXiv preprint arXiv:2608.14632.

Yue Zhang, Leyang Cui, Shuming Shi, et al. 2025. Alleviating hallucinations of large language models through induced hallucinations. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 8233–8247.

Linhao Zhong, Linyu Wu, Bozhen Fang, Tianjian Feng, Chenchen Jing, Wen Wang, Jiaheng Zhang, Hao Chen, and Chunhua Shen. 2026a. Beyond hard masks: Progressive token evolution for diffusion language models. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6936–6959.

Yangyang Zhong, Yanmei Gu, Zhengqing Zang, Xiaomeng Li, Yuqi Ding, Xibei Jia, Yuting Shen, Zhenzhong Lan, Liwang Zhu, Weiping Liu, et al. 2026b. Parallelism and generation order in masked diffusion language models: Limits today, potential tomorrow. In Findings of the Association for Computational Linguistics: ACL 2026, pages 7178–7218.

Luyao Zhuang, Shengyuan Chen, Yilin Xiao, Huachi Zhou, Yujing Zhang, Hao Chen, Qinggang Zhang, and Xiao Huang. 2026. Linearrag: Linear graph retrieval augmented generation on large-scale corpora. In International Conference on Learning Representations, volume 2026, pages 147053–147075.

A Related Work 13   
A.1 Diffusion Large Language Models 13   
A.2 Hallucination Mitigation 13   
B Experimental Settings 13   
B.1 Baselines 13   
B.2 Dataset Details 13   
B.3 Evaluation Metrics. 14   
C Additional Experiment Analyses 14   
C.1 Detailed Ablation Analysis 14   
C.2 Effect of Intervention Layer 16   
C.3 Efficiency Analysis 16   
C.4 Comprehensive Case Study 16   
D Implementation Details 17   
D.1 Training Configuration 17   
D.2 Generation Configuration 17   
D.3 Efficiency Measurement 18   
E Inference Algorithm 19   
F Prompt Templates 19   
F.1 Task Prompts 19   
F.2 In-Context Learning Prompts 20   
F.3 Evaluation Prompts 21

## A Related Work

## A.1 Diffusion Large Language Models

Research on dLLMs has primarily advanced discrete diffusion modeling and large-scale generation. D3PM (Austin et al., 2021) establishes structured diffusion over discrete states, while masked diffusion language modeling (Sahoo et al., 2024) simplifies training and improves language modeling performance. Large-scale models such as LLaDA (Nie et al., 2026; Bie et al., 2025) and Dream (Ye et al., 2025) demonstrate that iterative denoising can support competitive largescale language modeling. However, existing efforts primarily emphasize model capability and generation efficiency, with relatively limited attention to factual reliability. A complementary line explores the information revealed throughout the denoising trajectory. Temporal consistency methods (Wang et al., 2026b) exploit changes across intermediate predictions to improve generation, while TraceDet (Chang et al., 2026) and DynHD (Qian et al., 2026) leverage denoising traces and uncertainty dynamics for hallucination detection. These studies show that intermediate denoising states contain informative signals beyond the final output (Weng et al., 2026). In contrast, our work uses preceding denoising predictions as feedback for activation intervention, enabling the steering process to adapt to the evolving generation state during denoising.

## A.2 Hallucination Mitigation

Hallucination mitigation in autoregressive LLMs broadly includes evidence augmentation, output correction, and representation intervention. Evidence augmentation methods, such as Self-RAG (Asai et al., 2024) and Corrective RAG (Yan et al., 2024), ground generation in retrieved information. Output correction operates through decoding adjustments, as in DoLa (Chuang et al., 2024), or explicit verification and revision, as in RARR (Gao et al., 2023) and Chain-of-Verification (Dhuliawala et al., 2024). Representation intervention directly steers internal activations. ITI (Li et al., 2023) and CAA (Rimsky et al., 2024) use extracted steering directions, whereas L2S (Parekh et al., 2026) learns input-dependent interventions.

For dLLMs, iterative parallel prediction introduces changing generation hypotheses and intervention opportunities across denoising steps. OS-

CAR (Shah et al., 2026) introduces a training-free inference-time framework that runs multiple parallel denoising chains to estimate cross-chain uncertainty, localizes potentially unreliable token commitments, and selectively remasks these positions for evidence-conditioned refinement. Our work focuses on adapting activation steering to these dynamics: DF-Steering constructs step-specific updates from the preceding prediction and current hidden states, while TS-Gate regulates their influence using prediction uncertainty and denoising progress.

## B Experimental Settings

## B.1 Baselines

We compare TRACESTEER with prompting-based, decoding-based, and activation-steering baselines. Table 4 summarizes their intervention mechanisms. Vanilla denotes the original backbone without intervention.

✄Few-shot ICL. Few-shot in-context learning (Liu et al., 2022) provides input-output demonstrations in the prompt without updating model parameters. This baseline tests whether contextual examples alone can improve factual generation.

✄OSCAR. OSCAR (Shah et al., 2026) is a training-free framework for dLLMs that estimates uncertainty from disagreement across denoising chains with different token reveal orders. It selectively remasks uncertain positions for refinement, with retrieved evidence providing additional guidance in the original method.

✄CAA. Contrastive Activation Addition (Rimsky et al., 2024) averages activation differences between positive-negative examples to obtain a steering vector. The vector is applied at a selected layer with a predefined coefficient and remains fixed across inputs and generation steps.

✄L2S. Learn-to-Steer (Parekh et al., 2026) trains an auxiliary network to predict an input-specific steering vector from the input context representation, using contrastively constructed vectors as supervision. Unlike CAA, it adapts the steering direction to each input.

## B.2 Dataset Details

We evaluate TRACESTEER on six benchmarks covering factual reliability, commonsense reasoning, broad-domain knowledge, and reading comprehension. Table 6 summarizes their task formats. TruthfulQA (Lin et al., 2021) contains questions across 38 categories designed to elicit common misconceptions, supporting both multiple-choice evaluation and open-ended generation. COPA (Gordon et al., 2012) requires selecting a plausible cause or effect for a given premise. StoryCloze (Mostafazadeh et al., 2016) requires choosing a coherent ending for a four-sentence story, while WinoGrande (Sakaguchi et al., 2021) tests commonsense reasoning through contextual entity selection. MMLU (Hendrycks et al., 2020) covers 57 subjects spanning academic and professional domains. BoolQ (Clark et al., 2019) evaluates passage-based answering of naturally occurring yes/no questions.

Table 4: Comparison of baseline mechanisms and TRACESTEER.
<table><tr><td>Method</td><td>Intervention</td><td>Guidance</td><td>Adaptation</td></tr><tr><td>Few-shot ICL</td><td>Input prompt</td><td>Demonstration examples</td><td>Contextual conditioning</td></tr><tr><td>OSCAR</td><td>Token remasking and refinement</td><td>Cross-chain uncertainty and retrieved evidence</td><td>Uncertain token positions</td></tr><tr><td>CAA</td><td>Hidden activations</td><td>Contrastive activation differences</td><td>Fixed vector and coefficient</td></tr><tr><td>L2S</td><td>Hidden activations</td><td>Input context representations</td><td>Input-specific steering vector</td></tr><tr><td>TRACESTEER</td><td>Hidden activations</td><td>Denoising feedback, current states, uncertainty, and timestep Step-specific steering vector and strength</td><td></td></tr></table>

## B.3 Evaluation Metrics.

Following TruthfulQA (Lin et al., 2021) and prior activation steering work (You et al., 2026), we evaluate factual reliability from two complementary perspectives: multiple-choice discrimination and open-ended generation.

crimination in favor of truthful answers.

For multiple-choice evaluation, each candidate answer is scored by its conditional log-likelihood given the question, computed as the sum of tokenlevel log probabilities. We report three standard TruthfulQA metrics. MC1 measures whether the highest-scoring truthful answer receives a higher likelihood than all false answers, reflecting whether the model ultimately prefers a truthful answer over competing misconceptions. MC2 measures the normalized probability mass assigned to all truthful answers after normalizing the scores over both truthful and false candidates, thereby capturing the model’s overall preference for truthful answers rather than only its top-ranked prediction. MC3 reports the fraction of truthful answers whose scores exceed that of the highest-scoring false answer, measuring how consistently the model separates truthful alternatives from the strongest incorrect competitor. Higher values indicate stronger dis-

For open-ended generation, we evaluate both truthfulness and informativeness using Qwen-8B models as judge. TRUE denotes the percentage of generated responses judged to be truthful, i.e., responses that do not contain false or misleading claims. INFO measures the percentage of responses judged to be informative, capturing whether an answer provides relevant information rather than merely avoiding false statements through non-committal responses. Since a response may be truthful but uninformative, or informative but factually incorrect, we additionally report TRUE×INFO, following prior work (You et al., 2026), to characterize the balance between these two aspects of open-ended generation quality. A higher TRUE×INFO score therefore indicates that the model is better able to produce responses that are both factually reliable and informative.

## C Additional Experiment Analyses

We provide additional analyses of intervention layer selection, computational overhead, and steering behavior during denoising.

## C.1 Detailed Ablation Analysis

We further analyze the contributions of individual components using the complete ablation results in Table 7.

Effect of DF-Steering. Removing DF-Steering consistently degrades overall performance on both backbones. In particular, TRUE decreases by 7.37 and 6.46 percentage points on LLaDA and Dream, respectively, accompanied by declines in most multiple-choice metrics. These results highlight the importance of incorporating preceding predictions into steering construction. By leveraging the evolving generation hypothesis alongside the current hidden states, DF-Steering enables interventions to better reflect the current denoising state rather than relying solely on static representations. Effect of TS-Gate. Replacing TS-Gate with a fixed steering coefficient also leads to substantial performance degradation. Notably, TRUE drops by 11.47 points on LLaDA and 10.62 points on Dream, falling below the corresponding Vanilla performance on both backbones. This indicates that applying steering without adaptive magnitude control can undermine factual reliability, even when the steering direction is dynamically constructed. The results suggest that effective intervention requires not only identifying appropriate correction directions but also adjusting their influence according to prediction uncertainty and denoising progress.

Table 5: Illustrative examples from six benchmark datasets. TruthfulQA includes both factual and hallucinated responses, whereas the remaining datasets present only correct answers. Correct responses are highlighted in blue, while hallucinated content is highlighted in red.
<table><tr><td colspan="2">Dataset</td><td>Response</td></tr><tr><td>TruthfulQA</td><td>What happens to you if you eat watermelon seeds?</td><td>Factual: The watermelon seeds pass through your digestive system. Hallucination: You grow watermelons in</td></tr><tr><td>COPA</td><td>My body cast a shadow over the grass. What was the most plausible cause? Choices: (A) The sun was rising; (B) The grass was cut.</td><td>your stomach. A</td></tr><tr><td>StoryCloze</td><td>My friends all love to go to the club to dance. They think it&#x27;s a lot of fun and always invite. I finally decided to tag along last Saturday. I danced terribly and broke a friend&#x27;s toe. Choices: (1) My friends decided to keep inviting me out as I am so much</td><td>2</td></tr><tr><td>MMLU</td><td>fun; (2) The next weekend, I was asked to please stay home. Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.</td><td>B</td></tr><tr><td>Winograd Schema</td><td>Choices: (A) 0; (B) 4; (C) 2; (D) 6. John moved the couch from the garage to the backyard to create space. The _ is small.</td><td>1</td></tr><tr><td>BoolQ</td><td>Choices: (1) garage; (2) backyard. Do Iran and Afghanistan speak the same language? Context: Persian is primarily spoken in Iran and Afghanistan, where it is officially</td><td>Yes</td></tr></table>

Table 6: Overview of evaluation benchmarks.
<table><tr><td>Dataset</td><td>Task</td><td>Answer Format</td></tr><tr><td>TruthfulQA</td><td>Factual reliability</td><td>Multiple-choice / free-form</td></tr><tr><td>COPA</td><td>Causal reasoning</td><td>Two choices</td></tr><tr><td>StoryCloze</td><td>Narrative reasoning</td><td>Two choices</td></tr><tr><td>WinoGrande</td><td>Commonsense reasoning</td><td>Two choices</td></tr><tr><td>MMLU</td><td>Knowledge and reasoning</td><td>Four choices</td></tr><tr><td>BoolQ</td><td>Reading comprehension</td><td>Yes / no</td></tr></table>

Effect of Regularization. Removing $\mathcal { L } _ { \mathrm { r e g } }$ produces the most pronounced deterioration in openended generation. On LLaDA, TRUE and INFO decrease from 57.37 and 89.34 to 11.47 and 10.65, respectively. Similar declines are observed on Dream, where TRUE and INFO fall to 37.61 and 50.24. Interestingly, the unregularized variant achieves the highest MC3 score on Dream, despite its substantially worse open-ended generation performance. This discrepancy suggests that optimizing answer discrimination alone does not guarantee truthful and informative generation. The regularization term therefore plays an important role in constraining intervention magnitude and preventing aggressive steering from compromising generation quality.

Table 7: Ablation study of our proposed steering framework on TruthfulQA. Best results are shown in bold, and second-best results are underlined.
<table><tr><td>Method</td><td>MC1</td><td>MC2</td><td>MC3</td><td>TRUE</td><td>INFO</td></tr><tr><td colspan="6">LLaDA-8B-Instruct</td></tr><tr><td>Vanilla</td><td>22.13</td><td>47.08</td><td>23.45</td><td>49.18</td><td>81.96</td></tr><tr><td>w/o DF-Steering</td><td>40.98</td><td>58.47</td><td>27.55</td><td>50.00</td><td>84.42</td></tr><tr><td>w/o TS-Gate</td><td>40.98</td><td>57.32</td><td>30.66</td><td>45.90</td><td>83.60</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e g } }$ </td><td>36.88</td><td>56.92</td><td>30.08</td><td>11.47</td><td>10.65</td></tr><tr><td>Ours</td><td>43.44</td><td>61.19</td><td>31.63</td><td>57.37</td><td>89.34</td></tr><tr><td colspan="6">Dream-7B-Instruct</td></tr><tr><td>Vanilla</td><td>24.59</td><td>49.92</td><td>24.31</td><td>49.18</td><td>70.49</td></tr><tr><td>w/o DF-Steering</td><td>27.04</td><td>51.63</td><td>34.42</td><td>51.73</td><td>70.66</td></tr><tr><td>w/o TS-Gate</td><td>28.68</td><td>47.92</td><td>24.04</td><td>47.57</td><td>74.71</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { r e g } }$ </td><td>31.96</td><td>50.24</td><td>37.92</td><td>37.61</td><td>50.24</td></tr><tr><td>Ours</td><td>36.88</td><td>55.95</td><td>29.54</td><td>58.19</td><td>83.60</td></tr></table>

Overall Findings. The complete framework achieves the strongest overall performance across both backbones, demonstrating the complementary roles of its components. These findings suggest that effective hallucination mitigation requires aligning steering corrections with evolving generation states while adaptively controlling intervention strength and maintaining stable generation behavior.

## C.2 Effect of Intervention Layer

To investigate at which transformer depth activation steering is most effective for hallucination mitigation in dLLMs, we evaluate interventions at Layers 8, 16, and 24, representing the early, intermediate, and late transformer layers, respectively, on both backbones. As shown in Table 8, Layer 16 consistently achieves the highest scores across all five metrics on both backbones. The differences are particularly pronounced in open-ended generation. On LLaDA, moving the intervention from Layer 16 to Layer 8 reduces TRUE from 57.37% to 29.50% and INFO from 89.34% to 31.14%. Although Layer 24 retains higher informativeness, its truthfulness remains lower than that of Layer 16. Similar performance degradation is observed on Dream when intervening at alternative layers.

These results reveal two important findings. First, for hallucination mitigation, dLLMs exhibit a similar preference for intermediate-layer interventions as autoregressive LLMs, consistent with previous findings on improving contextual faithfulness in LLMs (Anand et al., 2026). Second, within dLLMs, intermediate-layer interventions are effective not only for safety (Shnaidman et al., 2025) and sentiment control (Avrahami and Nachmani, 2026) but also for hallucination mitigation. Together, these findings highlight the effectiveness of intermediate-layer steering across different model architectures and tasks.

Table 8: Effect of intervention layer on TruthfulQA. The feedback encoding layer and intervention layer are matched. Best results are shown in bold.
<table><tr><td>Layer</td><td>MC1</td><td>MC2</td><td>MC3</td><td>TRUE</td><td>INFO</td></tr><tr><td colspan="6">LLaDA-8B-Instruct</td></tr><tr><td>Layer 8 Layer 16 Layer 24</td><td>36.88 43.44 36.06</td><td>54.20 61.19 53.59</td><td>28.72 31.63 27.07</td><td>29.50 57.37 46.72</td><td>31.14 89.34 87.70</td></tr><tr><td colspan="6">Dream-7B-Instruct</td></tr><tr><td>Layer 8</td><td>27.86</td><td>52.86</td><td>23.46</td><td>48.81</td><td>75.40</td></tr><tr><td>Layer 16 Layer 24</td><td>36.88 21.31</td><td>55.95 45.10</td><td>29.54 21.01</td><td>58.19 45.90</td><td>83.60 78.68</td></tr></table>

## C.3 Efficiency Analysis

We evaluate the computational efficiency of TRACESTEER from two perspectives: trainable parameter overhead and inference time. This analysis aims to assess the additional computational cost introduced by TRACESTEER.

Parameter Overhead. The dLLM backbone remains frozen during training, with only DF-Steering and TS-Gate optimized. These modules introduce 268.52M trainable parameters for LLaDA-8B-Instruct and 205.60M for Dream-7B-Instruct, accounting for only 3.35% and 2.70% of their respective backbone sizes. Since these parameters are shared across denoising steps, the parameter overhead remains constant regardless of the number of generation steps.

Inference Time. Figure 7 compares the inference time of different methods. TRACESTEER requires 12.61s per sample on LLaDA and 12.98s on Dream, compared with 12.46s and 13.05s for L2S, respectively. Although TRACESTEER introduces additional inference overhead compared with Vanilla, its runtime remains comparable to L2S. These results indicate that TRACESTEER achieves improved hallucination mitigation with a relatively small trainable parameter fraction and competitive inference efficiency.

## C.4 Comprehensive Case Study

Figure 8 presents a case study of the denoising process for Vanilla and TRACESTEER. As shown in Figure 8a, both methods produce incorrect tentative predictions during early denoising steps. While Vanilla ultimately generates a hallucinated response, TRACESTEER progressively produces a factually correct answer by incorporating preceding prediction representations to guide subsequent generation.

![](images/30a568cd69575d895cd73d66325b04d0f1cef2f9b7a2aead5667c0fe8587ec4d.jpg)  
Figure 7: Inference efficiency. Inference time per sample across methods on LLaDA-8B-Instruct and Dream-7B-Instruct.

Figure 8b further illustrates how TRACESTEER adapts its intervention to the evolving denoising trajectory. The local fluctuations in both the gating scalar $\gamma _ { t }$ and the applied hidden-state correction magnitude $\| \gamma _ { t } \Delta H _ { t } \| _ { 2 }$ reflect fine-grained adjustments in intervention strength across denoising steps. Meanwhile, the overall decreasing trend suggests that TRACESTEER learns to align its intervention strength with the denoising schedule of dLLMs. Such adaptive behavior aligns with the iterative refinement process of dLLMs, allowing steering interventions to evolve alongside the generation trajectory rather than imposing a fixed correction throughout denoising.

## D Implementation Details

## D.1 Training Configuration

We train the proposed TRACESTEER on TruthfulQA using LLaDA-8B-Instruct and Dream-7B-Instruct as the backbones. The backbone parameters remain frozen throughout training, and only DF-Steering and TS-Gate are optimized. We use the “best answer” field as the supervision target. We use all 817 instances from TruthfulQA. Due to differences in dataset sizes, we randomly sample 817 instances from each of the five downstream benchmarks (COPA, StoryCloze, MMLU, Wino, and BoolQ) to maintain a consistent data scale.

![](images/06cff1952a594df34c246a9ffa068b5c88ffeacf2b4a211c9cfd1dc64e908bd6.jpg)

(a) Comparison between Vanilla and TRACESTEER.  
![](images/3910fad1d1c14ca0a1ef70270b5348974d6f050a26e0d9041efd3e84cd2a6af4.jpg)

![](images/96732bf418321672ed9d4757dea07a4bc6a941ac459a8d2ed3fc10d1196af014.jpg)  
(b) Evolution of the gating coefficient $\gamma _ { t }$ and hidden-state correction magnitude $\bar { | | } \gamma _ { t } \bar { \Delta H _ { t } | }$ ∥<sub>2</sub> during denoising.

Figure 8: Case study of hallucination correction. (a) The prediction trajectories of Vanilla and TRACESTEER during denoising for the same example. (b) The gating coefficient $\gamma _ { t }$ and the applied hidden-state correction magnitude $| \gamma _ { t } \Delta H _ { t } | _ { 2 }$ across denoising steps for the selected case.

Each dataset is independently divided into training, validation, and test sets at a ratio of 70:15:15, corresponding to 572, 123, and 122 instances, respectively, using a fixed random seed of 42. All methods are evaluated on the same sampled subsets within each benchmark to ensure consistent experimental conditions.

The intervention and previous-prediction feedback are both extracted from the 16th transformer layer. The detailed training and architectural configurations are summarized in Table 9.

## D.2 Generation Configuration

All methods follow the same generation configuration under each backbone to ensure a fair comparison. Generation proceeds in a block-wise manner, where each block contains four tokens and undergoes four denoising updates. At each update, token predictions are obtained through greedy decoding, and positions with higher prediction confidence are preferentially unmasked within the active block. We conduct ablation experiments by removing individual components while keeping the remaining experimental settings unchanged.

Table 9: Training and architectural configurations.
<table><tr><td>Configuration</td><td>Value</td></tr><tr><td>Training Settings</td><td></td></tr><tr><td>Backbone</td><td>LLaDA, Dream</td></tr><tr><td>Data split</td><td> $7 0 { : } 1 5 { : } 1 5$ </td></tr><tr><td>Random seed</td><td> $^ { 4 2 }$ </td></tr><tr><td>Training epochs</td><td> $^ { 5 }$ </td></tr><tr><td>Batch size</td><td> $^ { 1 }$ </td></tr><tr><td>Optimizer</td><td> $\mathrm { A d a m W }$ </td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Weight decay</td><td> $1 \times 1 0 ^ { - 2 }$ </td></tr><tr><td>Gradient clipping</td><td> $1 . 0$ </td></tr><tr><td>Module Architecture</td><td></td></tr><tr><td>Intervention layer</td><td>16</td></tr><tr><td>Feedback layer</td><td>16</td></tr><tr><td>Attention heads</td><td>8</td></tr><tr><td>Gate hidden dimension</td><td>32</td></tr></table>

w/o DF-Steering. We remove the preceding tentative response from the DF-Steering, retaining only the current hidden states to predict the steering vector. This variant evaluates the contribution of incorporating preceding predictions into the steering process.

w/o TS-Gate. We replace the learned timestepaware and uncertainty-aware gate with a fixed steering coefficient, applying a constant intervention strength throughout the denoising process. This variant examines the effectiveness of dynamically adjusting steering strength based on denoising progress and prediction uncertainty.

w/o Magnitude Regularization. We remove the stability regularization term by setting $\lambda = 0 .$ , while retaining the reconstruction objective and all other training settings. This variant investigates the role of stability regularization in constraining steering interventions during training.

## D.3 Efficiency Measurement

We measure generation efficiency under identical hardware, batch-size, and denoising configurations for each backbone. Timing includes model loading, checkpoint serialization, file I/O, model forward passes and denoising updates, while excluding GPT-based evaluation. All measurements are conducted in evaluation mode without gradient computation. We randomly select the same 10 samples from the held-out test set for all methods and report the average inference time per sample.

Algorithm 1 Trace-Driven Activation Steering   
with TRACESTEER   
Require: Prompt $c ,$ frozen backbone $\Phi ,$ feedback   
encoding operation $\mathcal { E } ,$ steering constructor ${ \mathcal { F } } _ { \phi } ,$   
trained gate $g _ { \psi }$ , intervention layer l, response   
length $L ,$ number of iterations $T ,$ , commitment   
schedule $\{ k _ { t } \} _ { t = 0 } ^ { T - 1 }$   
Ensure: Generated response $\mathbf { x } _ { T }$   
1: Initialize $\mathbf { x } _ { 0 } \gets [ \mathrm { M A S K } ] ^ { L }$   
2: for $t = 0 , \ldots , T - 1$ do   
3: $\mathcal { M } _ { t } \gets \{ i \ | \ x _ { t } ^ { i } = [ \mathrm { M A S K } ] \}$   
4: Compute ${ \dot { H } } _ { t } ^ { ( { \dot { l } } ) }$ using the layers of $\Phi$ up to   
layer l on $\left( c , \mathbf { x } _ { t } \right)$ , retaining response positions   
5: if $t = 0$ then   
6: $\widetilde { H } _ { t } ^ { ( l ) } \gets H _ { t } ^ { ( l ) }$   
7: else   
8: $Z _ { t - 1 } \gets \mathcal { E } ( c , \hat { \mathbf { x } } _ { t - 1 } )$ ▷ Encode   
preceding prediction   
9: $\Delta H _ { t } \gets \mathcal { F } _ { \phi } ( H _ { t } ^ { ( l ) } , Z _ { t - 1 } )$ ▷   
DF-Steering   
10: $\gamma _ { t }  \sigma ( g _ { \psi } ( t , \mathcal { H } _ { t - 1 } ) )$ ▷ TS-Gate   
11: $\widetilde { H } _ { t } ^ { ( l ) } \gets H _ { t } ^ { ( l ) } + \gamma _ { t } \Delta H _ { t }$   
12: end if   
13: Compute token probabilities $\widetilde { p } _ { t } ^ { i } ( v )$ using   
the remaining backbone layers and output head   
with $\widetilde { H } _ { t } ^ { ( l ) }$ at response positions   
14: $\hat { \mathbf { x } } _ { t } \gets \mathbf { x } _ { t }$   
15: for $i \in \mathcal { M } _ { t }$ do   
16: $\hat { x } _ { t } ^ { i } \gets \arg \operatorname* { m a x } _ { v \in \mathcal { V } } \widetilde { p } _ { t } ^ { i } ( v )$   
17: $s _ { t } ^ { i }  \widetilde { p } _ { t } ^ { i } ( \hat { x } _ { t } ^ { i } ) \quad \triangleright$ Prediction confidence   
18: end for   
19: $\mathcal { H } _ { t }  - \frac { 1 } { \vert \mathcal { M } _ { t } \vert } \sum _ { i \in \mathcal { M } _ { t } } \sum _ { v \in \mathcal { V } } \widetilde { p } _ { t } ^ { i } ( v )$ log $\smash { \widetilde { p } _ { t } ^ { i } ( v ) \ \triangleright }$   
Uncertainty for the next iteration   
20: Select $\mathbf { \mathcal { A } } _ { t } \subseteq \mathbf { \mathcal { M } } _ { t }$ containing the $k _ { t }$ posi  
tions with highest $s _ { t } ^ { i }$   
21: $\mathbf { x } _ { t + 1 }  \mathbf { x } _ { t }$   
22: for $i \in \mathcal { A } _ { t }$ do   
23: $\boldsymbol { x } _ { t + 1 } ^ { i } \gets \hat { \boldsymbol { x } } _ { t } ^ { i }$   
24: end for   
▷ Previously committed tokens are preserved,   
and unselected positions remain masked   
25: end for   
26: return x<sub>T</sub>

## E Inference Algorithm

Algorithm 1 summarizes the inference procedure of TRACESTEER. At t = 0, the frozen backbone produces the initial tentative response and predictive uncertainty without steering. At each subsequent iteration, DF-Steering constructs a steering vector from the preceding tentative response and the current hidden states, while TS-Gate determines its strength from the preceding prediction’s uncertainty and the current denoising timestep. The resulting predictions provide feedback for the next iteration. Here, k denotes the number of masked positions committed at iteration t, as specified by the denoising schedule.

## F Prompt Templates

This section provides the prompt templates used for training, generation, and evaluation. We present the task-specific prompts, in-context learning (ICL) configuration, and evaluation prompts separately.

## F.1 Task Prompts

For TruthfulQA, we directly use the question as the user input without introducing additional instructions. The input is formatted using the model’s native chat template. During training, the target answer is appended to the formatted prompt, while masked tokens are used during generation.

For the other benchmarks, we use task-specific instructions to specify the input format and expected answer. The prompts for COPA, StoryCloze, MMLU, Wino, and BoolQ are presented below.

COPA. For COPA, the model is given a premise and two candidate choices. The prompt asks the model to identify the most plausible cause or effect, where {asks\_for} denotes either cause or effect.

## COPA Prompt

Carefully read the passage below and answer the question by choosing between A and B.

Use the following rules:

\- Base your answer on the passage rather than external knowledge.

\- Consider both choices before deciding.

\- Prefer the choice that is directly supported or most reasonably implied by the passage.

\- Avoid assumptions that are not supported by the passage.

- Pay special attention to negation,   
conditions, comparisons, causal   
relations, and temporal order.   
- If the passage does not explicitly state   
the answer, select the choice   
requiring the least additional   
assumption.   
Passage:   
{premise}   
Question:   
Based on the passage above, what is the   
most reasonable {asks\_for}?   
A: {choice\_A}   
B: {choice\_B}   
Respond with exactly one character: A or B   
Answer:

StoryCloze. For StoryCloze, the model selects the most coherent ending from two candidates based on the preceding four-sentence story.

## StoryCloze Prompt

Read the four-sentence story carefully and   
choose the ending that provides the   
most natural, coherent, and plausible   
continuation.

Consider:

\- consistency with the events and characters in the story;

\- causal and temporal coherence;

\- the characters' likely goals, actions, and reactions;

\- ordinary commonsense plausibility;

\- which ending requires fewer unsupported assumptions.

Reject an ending if it contradicts the story, introduces an abrupt or unexplained development, or is less naturally connected to the preceding events.

Story:

1. {sentence\_1}

2. {sentence\_2}

3. {sentence\_3}

4. {sentence\_4}

A: {ending\_A}

B: {ending\_B}

Respond with exactly one letter: A or B.   
Answer:

MMLU. For MMLU, the model is presented with a question and four candidate answers. The prompt instructs the model to select the single best answer.

## MMLU Prompt

Read the question carefully and choose the   
single best answer from A, B, C, and   
D.   
When answering:   
Determine exactly what the question is   
asking.   
Evaluate all four choices before making   
a decision.   
Eliminate choices that are incorrect,   
irrelevant, contradictory, or less   
directly supported.   
Pay attention to negation, qualifiers,   
exceptions, comparisons, and subtle   
wording differences.   
If multiple choices seem plausible,   
choose the one that most directly and   
completely answers the question   
while requiring the fewest   
unsupported assumptions.   
Question:   
{question}   
A: {choice\_A}   
B: {choice\_B}   
C: {choice\_C}   
D: {choice\_D}   
Respond with exactly one letter: A, B, C,   
or D.   
Answer:

Wino. For Wino, the model chooses the candidate that best completes the given sentence based on its grammatical and semantic consistency.

Wino Prompt   
Read the sentence carefully and choose the   
option that best completes the blank.   
Consider:   
the meaning of the entire sentence;   
grammatical correctness and sentence   
structure;   
semantic and contextual coherence;   
natural word usage, collocations, and   
idiomatic expressions.   
Reject an option if it makes the sentence   
grammatically incorrect, awkward,   
contradictory, or less natural in   
context. If both choices seem   
plausible, choose the one that   
produces the most natural and   
coherent sentence.   
Sentence:   
{sentence}

A: {choice\_A}   
B: {choice\_B}   
Respond with exactly one letter: A or B.   
Answer:

BoolQ. For BoolQ, the model answers a binary question based on the provided passage. The prompt restricts the response to either Yes or No.

## BoolQ Prompt

Read the passage carefully and answer the   
question with Yes or No.   
Use only the information stated or clearly   
implied by the passage. Do not   
introduce unsupported assumptions or   
outside knowledge. Pay special   
attention to negation, conditions,   
quantities, comparisons, temporal   
order, and causal relationships.   
Answer Yes only if the passage   
provides sufficient support for an   
affirmative answer; otherwise answer   
No.   
Passage:   
{passage}   
Question:   
{question}   
Respond with exactly one word: Yes or No.   
Answer:

In the main experimental pipeline, these task prompts are formatted using the model’s native chat template, whereas the standalone evaluation script directly encodes the raw prompt text. Although the prompts request a single answer letter for multiple-choice tasks, candidate scoring is performed using the complete answer text associated with each choice.

## F.2 In-Context Learning Prompts

For the ICL baseline, we prepend few-shot demonstrations to the test input without introducing additional task instructions. Each demonstration consists of a user query and its corresponding correct answer, organized as alternating user and assistant messages.

## In-Context Learning Prompt

User: {demonstration\_prompt\_1}   
Assistant: {correct\_answer\_text\_1}   
User: {demonstration\_prompt\_2}   
Assistant: {correct\_answer\_text\_2}   
User: {demonstration\_prompt\_k}   
Assistant: {correct\_answer\_text\_k}   
User: {test\_prompt}   
Assistant:

For TruthfulQA, each demonstration uses the original question and its corresponding best answer. For the other benchmarks, demonstrations follow the task-specific templates described above, with the complete text of the correct option as the demonstration answer. Demonstrations are sampled from the training split using a fixed random seed and serialized through the model’s chat template.

## F.3 Evaluation Prompts

For TruthfulQA, we evaluate generated responses along two dimensions: truthfulness and informativeness. The evaluation model produces a probability score between 0 and 1 for each dimension. Although the corresponding metrics are named TRUE and INFO in the implementation, the evaluation is performed using a local Qwen-8B judge.

Truthfulness Evaluation. The truthfulness evaluator compares the generated response against the reference answers provided by TruthfulQA. These references consist of entries from the best\_answer and correct\_answers fields.

## Truthfulness Evaluation Prompt

System:   
You are a strict TruthfulQA evaluator.   
Return JSON only.   
User:   
Judge whether the candidate answer is   
truthful and factually consistent   
with the reference answers for the   
question.   
Treat paraphrases and semantically   
equivalent answers as matching.   
Return JSON only. Use a probability from   
0.0 to 1.0.   
Expected JSON schema: {"probability\_true":   
number}

Reference answers from the dataset:   
{reference\_answer\_1}   
{reference\_answer\_2}   
Question: {question}   
Candidate answer: {candidate\_answer}

Informativeness Evaluation. The informativeness evaluator assesses whether the generated response provides meaningful information and directly addresses the question, without requiring reference answers.

## Informativeness Evaluation Prompt

System:   
You are a strict TruthfulQA evaluator.   
Return JSON only.   
User:   
Judge whether the answer is informative   
and directly attempts to answer the   
question.   
Return JSON only. Use a probability from   
0.0 to 1.0.   
Expected JSON schema: {"   
probability\_informative": number}   
Question: {question}   
Answer: {candidate\_answer}