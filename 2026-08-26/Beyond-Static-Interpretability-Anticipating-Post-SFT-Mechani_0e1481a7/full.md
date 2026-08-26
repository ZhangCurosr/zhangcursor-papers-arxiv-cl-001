# Beyond Static Interpretability: Anticipating Post-SFT Mechanisms from Pre-SFT Parameters for Better Tuning

Hang Chen College of Computing and Data Science Nanyang Technological University chen.hang@ntu.edu.sg

Jiaying Zhu School of Computer Science and Engineering The Chinese University of Hong Kong jyzhu24@cse.cuhk.edu.hk

Wenya Wang College of Computing and Data Science Nanyang Technological University wangwy@ntu.edu.sg

## Abstract

Mechanistic Localization bridges mechanistic interpretability and post-training optimization by isolating critical parameters via interpretative approaches and then guiding parameter-efficient Supervised Fine-Tuning (SFT) in a “locatingthen-tuning” paradigm. However, due to the retrospective nature of mechanistic interpretability, directly interpreting pre-SFT models introduces misleading conclusions. Specifically for novel tasks, initially identified neurons differ drastically from those governing the final model, introducing biases that actively disrupt SFT. To address this, we propose a forward-looking localization framework that accurately estimates the post-SFT interpretability state using only pre-SFT parameters and the target dataset. Theoretically, we model SFT as a continuous parameter evolution, leveraging Taylor expansion to rigorously bridge the post-tuning mechanistic objective with the pre-SFT model’s dynamic gradients. Practically, we design dual-granularity (neuron- and component-level) localization pipelines. Extensive experiments demonstrate that our approach not only provides superior SFT guidance but also exhibits robust performance and temporal scalability across increasing model sizes. This work transcends the fundamental limitation of traditional interpretability—its inability to identify taskcritical mechanisms before they are trained—pioneering a predictive frontier that unites mechanistic interpretability with targeted optimization. Code is available at: https://github.com/Zodiark-ch/Future\_localization.

## 1 Introduction

As a pivotal foundation for reverse-engineering LLMs, mechanistic interpretability illuminates internal execution mechanisms by evaluating the causal effects of specific components or neurons through counterfactual analysis [13, 27, 28, 29]. Beyond passive diagnostic insights, its rigorous causal framework offers a principled basis for guiding post-training optimization. In particular, the locating-then-tuning paradigm [4] successfully operationalizes this insight into SFT-based adaptation: it first employs mechanistic localization [11] on the pre-SFT base model to identify key parameters governing the target task, and subsequently restricts parameter-efficient tuning only to these identified key parameters to yield the final post-SFT model. By preventing unconstrained weight updates from corrupting pre-existing representations, this paradigm has exhibited immense promise in critical domains including LLM unlearning [17, 5], knowledge editing [25, 41], and transparent post-training [36, 39, 19, 40].

However, relying on the pre-SFT model for mechanistic localization introduces a fundamental flaw: because the model has not yet been adapted to the target task, the parameters identified as task-relevant in pre-SFT model may not reflect the mechanisms that emerge during SFT, potentially leading to misleading localization for subsequent fine-tuning [4]. Consider an extreme yet representative case where the target task is entirely novel to the pre-SFT model. Because the pre-SFT model inherently lacks the mechanistics required to execute this unseen task, mechanistic interpretability on the pre-SFT model fails to identify meaningful task-governing parameters. Consequently, it yields deceptive signals—inadvertently freezing parameters vital for acquiring new skills and obstructing necessary circuit evolution [26, 16, 7]. Fundamentally, this limitation stems from an intrinsic tension between the prospective requirement of tuning and the retrospective nature of interpretability: locating-thentuning demands that interpretability anticipate which parameters will be recruited to learn the target skill, yet mechanistic interpretability can only dissect capabilities that are already instantiated within existing parameters. This creates a classic “chicken-and-egg” dilemma—one must localize the task-critical parameters before tuning, yet the corresponding circuits only emerge after the model has learned the task.

To this end, this paper aims to challenge the retrospective practice of interpretability area by answering a fundamental question: how can we predict the neurons (or components) $( \mathcal { N } ( \boldsymbol { \theta } ^ { I } ) ,$ ) responsiblefor a target task in an ideal post-SFT model(θ<sup>I</sup>) using only the pre-SFT model parameters $( \theta ) ?$

We view fine-tuning $( \theta  \theta ^ { I } )$ as an optimization establishing an initial gradient direction and subsequently traversing a long distance toward a local optimum. Consequently, we decompose $\theta \to \bar { \mathcal { N } } ( \theta ^ { I } )$ into two theoretical steps: isolating “direction” and extrapolating “distance”. First, we formulate $\dot { \theta }  \mathcal { N } ( \theta ^ { \prime } )$ , where $\theta ^ { \prime }$ represents the parameters following an infinitesimal update toward the target task’s supervision. This decouples the update distance to exclusively capture the initial direction. Building upon $\mathcal { N } ( \boldsymbol { \theta } ^ { \prime } )$ , the second step achieves $\mathcal { N } ( \theta ^ { \prime } )  \mathcal { N } ( \theta ^ { I } )$ by conceptualizing the total SFT parameter shift, $\Delta \dot { \theta } = \dot { \theta } ^ { I } - \theta _ { \dot { \theta } }$ , as the accumulation of infinite recursive updates originating from $\theta ^ { \prime }$

Our approach only requires fine-tuning on 1% of the training dataset for a single epoch to derive $\theta ^ { \prime }$ Inheriting the computational paradigm of Attribution Patching [24, 32], our method necessitates only two forward passes and one backward pass to collect the activation and gradient data required for the estimation. This computational overhead is entirely negligible compared to the full fine-tuning process or other interpreting approaches. Furthermore, to accommodate larger models and diverse practical scenarios, we designed two distinct estimation pipelines: a fine-grained estimation at the neuron level and a rapid estimation at the component level.

We empirically validate our approach on the Mistral-7B model, demonstrating its superior capability in guiding SFT compared to existing localization baselines. Furthermore, through rigorous ablation studies on arithmetic tasks of varying complexities, we establish the robustness of our method against increasing task difficulty. Additionally, we confirm the scalability of our approach across larger architectures, specifically LlaMA-2-13B and Qwen3-30B. Finally, we provide intrinsic validation of our method’s efficacy directly through the lens of the resulting interpretability analyses.

In summary, our main contributions are twofold:

• We propose a theoretical framework that leverages current parameters to estimate post-SFT mechanisms. To the best of our knowledge, this is the first work to predict future interpretability states. Confronting this question marks a transformative transition beyond traditional retrospective interpretability, pioneering a new frontier for its actionable utility in post-training optimization.

• We develop an estimation pipeline that achieves state-of-the-art localization performance. Validated across diverse LLM families, our method demonstrates robust scalability while maintaining highly stable computational efficiency as model parameters increase.

## 2 Preliminaries

## 2.1 Task Definition

Let M parameterized by θ denote the current model prior to fine-tuning, and $\mathcal { M } ^ { I }$ parameterized by $\theta ^ { I }$ represent the ideal post-tuning model. For a target task $\tau$ , let $\mathcal { D } _ { \mathcal { T } }$ be its associated dataset. By definition, the current model M performs poorly on $\mathcal { D } _ { \mathcal { T } }$ , whereas the ideal model $\mathcal { M } ^ { I }$ achieves optimal performance on $\mathcal { D } _ { \mathcal { T } }$ without degrading any pre-existing capabilities<sup>2</sup>.

We denote the subset of parameters primarily responsible for task $\tau$ as ${ \mathcal { N } } ^ { \theta } \subset \theta .$ which is typically identified via mechanistic interpretability methods. The existing locating-then-tuning paradigm operates through the following pipeline with partial tuning only operating on ${ \mathcal { N } } ^ { \theta }$

$$
\begin{array} { r } { \mathcal { M } ( \theta ) , \mathcal { D } _ { \mathcal { T } } \xrightarrow { \mathrm { l o c a l i z a t i o n } } \mathcal { N } ^ { \theta } \xrightarrow { \mathrm { p a r t i a l t u n i n g } } \hat { \mathcal { M } } ^ { I } ( \theta ^ { I } ) } \end{array}
$$

In this work, our objective is to reformulate the localization step to directly predict the critical parameter subset $( \mathcal { N } ^ { \theta ^ { I } } )$ of the ideal model:

$$
f : \langle { \mathcal { M } } ( \theta ) , { \mathcal { D } } _ { T } \rangle \longrightarrow { \mathcal { N } } ^ { \theta ^ { I } }
$$

## 2.2 Related Work

Existing research on the locating-then-tuning paradigm primarily focuses on optimizing mechanistic localization, relying predominantly on gradients [17, 39, 25] or causal effects [36, 19, 11, 5]. Recent advances in mechanistic interpretability [13, 21, 6, 3] demonstrate that, under rigorous experimental setups, causal effect-based approaches provide a more accurate and comprehensive identification of task-critical neurons. Specifically, these methods quantify the causal effect for the activation of a target parameter, neuron, or component a under the target dataset $\mathcal { D } _ { \mathcal { T } } ,$ by measuring the output discrepancy when the activation is replaced with a counterfactual value. This is formalized as:

$$
\Delta E _ { \mathcal { D } _ { \mathcal { T } } } ( a ) = f _ { a } ( A ( x _ { c } ) ) - f _ { a } ( A ( x _ { r } ) )
$$

where $f _ { a } ( \cdot )$ represents an output-dependent metric (e.g., logits or loss) as the samples from $\mathcal { D } _ { \mathcal { T } }$ are input, while $A ( x _ { c } )$ and $A ( x _ { r } )$ denote the clean and corrupted (counterfactual) activations, respectively. A larger $\Delta E$ signifies a stronger causal relationship with the task output. Currently, circuit discovery [8, 2] and path patching [35] are the most prevalent causal methodologies employed in mechanistic localization.

Highly pertinent to our study is Attribution Patching [24, 32], a causal effect method that approximates this influence using a first-order Taylor expansion:

$$
\Delta E _ { \mathcal { D } _ { T } } ( a ) = f _ { a } ( A ( x _ { c } ) ) - f _ { a } ( A ( x _ { r } ) ) \approx ( A ( x _ { c } ) - A ( x _ { r } ) ) \cdot \frac { \partial f _ { a } } { \partial a } \bigg \rvert _ { a = A ( x _ { r } ) }
$$

This mathematical reformulation elegantly circumvents the computationally prohibitive exhaustive search required by standard counterfactual ablation. Instead, it extracts the necessary metrics for all parameters simultaneously requiring only two forward passes to obtain $A ( x _ { c } )$ and $A ( x _ { r } )$ , and a single backward pass to compute the gradients $\frac { \partial f _ { a } } { \partial a } \big | _ { a = A ( x _ { r } ) } .$

Beyond static analysis, our work is motivated by recent dynamic interpretability studies analyzing mechanistic shifts during SFT [26, 16]. These works highlight that target task mechanisms undergo drastic transformations—including shifts in circuit strength [7], topology [36], and key nodes [4]—during tuning. As a result, deriving the critical neuron set ${ \bar { \mathcal { N } } } ^ { \bar { \theta } }$ purely from the pre-SFT parameters θ yields a biased localization. Instead of aiding the tuning process, this static guidance misdirects optimization and deteriorates task learning, thereby motivating our framework to estimate the ideal post-SFT state.

## 3 Method

The inherent challenge in our objective lies in the pragmatic nature of mechanistic interpretability: it is traditionally a retrospective analysis conducted on the explicit parameters of already-trained models. Existing theories do not support predicting interpretability states prior to tuning. Whereas targeted SFT aims at updating parameters supposedly influential for the target task. This presents a critical challenge: given only the current parameters θ and the target dataset $\mathcal { D } _ { \mathcal { T } }$ , how can we estimate the importance of future model components without actually executing the entire SFT process?

To address this, we conceptualize SFT as a continuous optimization trajectory—starting from an initial well-trained parameter distribution, evolving along a specific gradient direction, and traversing a certain distance to reach a local optimum. By bridging interpretability with gradient dynamics, our framework solves this challenge in two intuitive steps: first determining the initial evolutionary direction, and then extrapolating it to an appropriate distance. Specifically, first, in Section 3.1, we capture the initial gradient direction by modeling an infinitesimal update of θ towards the SFT objective on $\mathcal { D } _ { \mathcal { T } }$ , denoted as $\theta ^ { \prime }$ , and then we formulate its causal effect entirely using the initial θ. Second, in Section 3.2, we extrapolate this directional estimate to the optimal distance. By iteratively accumulating an infinite series of these infinitesimal updates, we approximate the causal effect of the ideal post-SFT model, $\theta ^ { I }$ , deriving our final predictive expression.

## 3.1 Single Step: From θ to $\theta ^ { \prime }$ for Gradient Direction

Let $\theta ^ { \prime }$ denote the parameters following an infinitesimal update of θ directed by the SFT objective on $\mathcal { D } _ { \mathcal { T } }$ . For any arbitrary parameter a under $\theta ^ { \prime }$ , its corresponding Attribution Patching formula is given by:

$$
f _ { a } ^ { \theta ^ { \prime } } ( A ( x _ { c } ) ) - f _ { a } ^ { \theta ^ { \prime } } ( A ( x _ { r } ) ) \approx ( A ^ { \theta ^ { \prime } } ( x _ { c } ) - A ^ { \theta ^ { \prime } } ( x _ { r } ) ) \cdot \left. \frac { \partial f _ { a } ^ { \theta ^ { \prime } } } { \partial a ^ { \theta ^ { \prime } } } \right| _ { a = A ^ { \theta ^ { \prime } } ( x _ { r } ) }\tag{1}
$$

Evaluating the terms $A ^ { \theta ^ { \prime } } ( x _ { r } ) , A ^ { \theta ^ { \prime } } ( x _ { c } )$ , and the gradient $\frac { \partial f _ { a } ^ { \theta ^ { \prime } } } { \partial a ^ { \theta ^ { \prime } } } \Big | _ { a = A ^ { \theta ^ { \prime } } ( x _ { r } ) }$ inherently requires two actual forward passes and one backward pass through the hypothetical $\theta ^ { \prime }$ model, which is practically intractable. Consequently, we seek to approximate these terms by applying a first-order Taylor expansion around θ. This yields a reformulation of the causal effect for $\grave { \theta ^ { \prime } }$ expressed entirely in terms of $\theta \colon$

$$
f _ { a } ^ { \theta ^ { \prime } } ( A ( x _ { c } ) ) - f _ { a } ^ { \theta ^ { \prime } } ( A ( x _ { r } ) ) \approx f _ { a } ^ { \theta } ( A ( x _ { c } ) ) - f _ { a } ^ { \theta } ( A ( x _ { r } ) ) + \Delta \theta \cdot S ( \theta )\tag{2}
$$

where

$$
S ( \theta ) = \left[ \left( \frac { \partial A ^ { \theta } ( x _ { c } ) } { \partial \theta } - \frac { \partial A ^ { \theta } ( x _ { r } ) } { \partial \theta } \right) \cdot \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } + \left( A ^ { \theta } ( x _ { c } ) - A ^ { \theta } ( x _ { r } ) \right) \cdot \frac { \partial ^ { 2 } f _ { a } ^ { \theta } } { \partial a \partial \theta } \right]\tag{3}
$$

Here, $\Delta \theta \ = \ \theta ^ { \prime } \ - \ \theta$ represents the distance of the parameter update. Noticeably, $S ( \theta ) \ \equiv$ $\begin{array} { r } { \nabla _ { \theta } \left[ ( A ^ { \theta } ( x _ { c } ) - A ^ { \theta } ( x _ { r } ) ) \cdot \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } \right] } \end{array}$ . Therefore, it represents the gradient of the attribution score with respect to $\theta ,$ capturing the sensitivity of the target neuron $\boldsymbol { a } ^ { \prime } \mathbf { s }$ attribution value to parameter updates.

Through Equation 2 (the detailed derivation is provided in Appendix $\mathbf { A } )$ , we transform the estimation of $\theta ^ { \prime } \bar { \bf s }$ causal effect—which previously relied on empirical forward and backward passes—into an estimation of the parameter update distance. In the following sections, we will elaborate on how to determine $\Delta \theta$ in order to ultimately approximate the interpretability state of the final ideal model, $\theta ^ { I }$

## 3.2 Multiple Steps: From $\theta ^ { \prime }$ to $\theta ^ { I }$ for Gradient Distance

We conceptualize the transition from the initial parameters $\theta$ to the ideal parameters $\theta ^ { I }$ as an infinite sequence of continuous, additive updates. For notational clarity, we represent this evolutionary trajectory as a series of states $\theta ^ { 0 } , \theta ^ { 1 } , \ldots , \theta ^ { I }$ , where the initial state is defined as $\theta ^ { 0 } \equiv \theta$ . Within this continuum, any intermediate state $\theta ^ { k }$ effectively functions as the infinitesimal update, previously denoted as $\theta ^ { \prime }$ , relative to its immediate predecessor $\theta ^ { k - 1 }$

Following Equation $^ { 2 , }$ the causal effect $\Delta E _ { \mathcal { D } \tau } ^ { \theta ^ { k + 1 } } ( a )$ at any subsequent state $\theta ^ { k + 1 }$ can be recursively formulated as:

$$
\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k + 1 } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k } } ( a ) + \Delta \theta ^ { k } \cdot S ( \theta ^ { k } )\tag{4}
$$

Consequently, by accumulating these incremental updates, the causal effect for the final ideal model is given by:

$$
\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta } ( a ) + \sum _ { k = 0 } ^ { I - 1 } \Delta \theta ^ { k } \cdot S ( \theta ^ { k } )\tag{5}
$$

The summation term in this formulation effectively constitutes a discrete path integral along the parameter update trajectory γ. In the continuous limit, this can be formally expressed as $\int _ { \gamma } { S ( \theta ) } ^ { - } d \theta$ To render this computation tractable without evaluating intermediate states, we approximate this continuous integral using the trapezoidal rule<sup>3</sup>, yielding:

$$
\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta } ( a ) + \frac { 1 } { 2 } \left[ S ( \theta ^ { 0 } ) + S ( \hat { \theta } ^ { I } ) \right] \cdot \sum _ { k = 0 } ^ { I - 1 } \Delta \theta ^ { k }\tag{6}
$$

Given that $S ( \theta )$ is strictly equivalent to the gradient of the attribution score with respect to $\theta \left( \mathrm { i . e . } \right.$ $S ( \theta ) \equiv \nabla _ { \theta } [ \Delta E _ { \mathcal { D } _ { T } } ^ { \theta } ( a ) ] )$ ). By defining a differential operator $D = { \Delta \theta \cdot \nabla _ { \theta } } .$ , we demonstrate that the infinite series of higher-order derivatives (Hessians, tensors) governing $S ( \theta ^ { k } )$ can be rigorously folded into an exponential translation operator, $e ^ { k \Delta \theta \cdot \nabla _ { \theta } }$ . Under this functional analysis perspective, evaluating S at state k is mathematically equivalent to simply shifting the input parameters. Consequently, we establish that $S ( \hat { \theta } ^ { I } ) = S ( \theta + K \Delta \theta )$ , allowing us to compute the final sensitivity by merely scaling the update distance. We refer readers to Appendix B for the complete theoretical proof.

Ultimately, we simplify the estimation of the causal effect for the ideal model, $\theta ^ { I }$ , to the following formulation:

$$
\Delta E _ { \mathcal { D } _ { \mathcal { T } } } ^ { \theta ^ { I } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \mathcal { T } } } ^ { \theta } ( a ) + \frac { 1 } { 2 } \left[ S ( \theta ^ { 0 } ) + S ( \theta ^ { 0 } + K \cdot \Delta \theta ) \right] \cdot K \cdot \Delta \theta\tag{7}
$$

where K denotes a step scalar that extrapolates the transition $\theta  \theta ^ { \prime }$ to the ideal macroscopic transition $\theta  \theta ^ { I }$ . While theoretically derived from an infinitely small step, practically, we approximate $\theta ^ { \prime }$ using a sufficiently small parameter update. Moreover, the ideal post-SFT state is non-unique—varying based on optimization trajectories or random initializations—we do not treat K as a single determined value. Instead, we use K as an extrapolation scalar to project along the known gradient direction, sampling a range of K values to explore plausible future configurations $S ( \hat { \theta } ^ { I } ) = S ( \theta ^ { 0 } + { \cal K } \cdot \Delta \theta )$ and ultimately estimating the mean causal effect of the target task mechanisms. Therefore, this approximation requires that $\Delta \theta = \theta ^ { \prime } - \theta$ robustly captures the general optimization direction of the target task, while maintaining $\Delta \theta$ small enough to provide the step scalar K with ample degrees of freedom to explore the parameter space.

![](images/b643b7184a47ef57075d606e81a1f2ee6aa782fa8fc0bc544875de8abc031fc6.jpg)  
Figure 1: Extrapolation of the ideal state via the step scalar $K .$ . Different values of $K$ yield distinct sensitivity estimations $S ( \theta + K \cdot \Delta \theta )$ , reflecting that the ideal parameter state $\theta ^ { I }$ consists of an infinite set of valid configurations rather than a single deterministic solution.

## 4 Implemetation

## 4.1 Probing SFT

In this work, we acquire $\theta ^ { \prime }$ through a probing SFT strategy. Specifically, we uniformly sample 1% of the training dataset to perform a tentative fine-tuning for a single epoch at $1 \%$ of the original learning rate, designating the resulting updated weights as $\theta ^ { \prime }$ . As explored in our empirical analysis (Section 5.4), a larger divergence between $\theta ^ { \prime }$ and $\theta$ does not monotonously correlate with improved performance; in fact, deriving $\theta ^ { \prime }$ from $1 \%$ of the training data yields significantly superior results compared to using 10%. This phenomenon occurs because $\Delta \theta$ must remain sufficiently infinitesimal to preserve adequate degrees of freedom for the extrapolated state $\theta + K \cdot \Delta \theta .$ , while simultaneously mitigating the influence of gradient-irrelevant noise inherent in $\theta ^ { \prime }$

Notably, performing attribution patching directly on $\theta ^ { \prime }$ (hereafter referred to as the probing model) constitutes a compelling baseline, which we denote as Localizationfrom $\theta ^ { \prime }$ . It intuitively offers more foresight than localization from the initial $\theta ,$ yet avoids the complexity of estimating the ideal state $\hat { \theta } ^ { I }$ . Mathematically, however, $\theta ^ { \prime }$ remains proximate to $\theta$ and distant from $\hat { \theta } ^ { I }$ . As formally proven in Appendix C, establishing the equivalence $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \hat { \theta } ^ { I } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { \prime } } ( a )$ necessitates imposing stringently unrealistic assumptions on the SFT parameter update trajectory. That ${ \mathrm { i s } } ,$ for any step k, it satisfies: ${ \frac { \partial ( A ( x _ { c } ) - A ( x _ { r } ) ) } { \partial \theta } } \cdot { \dot { \Delta } } \theta \approx X _ { k } - X _ { 0 }$ . Nevertheless, regarding the final estimation, $\Delta E _ { \mathcal { D } \tau } ^ { \theta ^ { \prime } } ( a )$ inherently introduces significantly less approximation error, at least from the perspective of parameter variation. Our empirical evaluation in Section 5.3 thoroughly compares these two paradigms—Localization from $\hat { \theta } ^ { I }$ versus Localization from $\theta ^ { \prime } .$ . The results demonstrate that as the difficulty of the target SFT task increases, the superiority of Localizationfrom $\hat { \theta } ^ { I }$ over Localizationfrom $\theta ^ { \prime }$ becomes increasingly pronounced.

## 4.2 Locating-then-tuning Pipelines

We present a generalized pipeline for the locating-then-tuning paradigm. During the localization phase, we compute the causal effect $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a )$ for all parameters and rank them in descending order to determine their relative importance. To ensure a robust estimation, we randomly sample 10 distinct values of $K$ and average their resulting scores. The optimal sampling interval for $\bar { K }$ is inherently task-dependent, a phenomenon we thoroughly analyze in Appendix F.1, and it explores a probing model approach to recommend $K$ , and provides an analysis of the robustness concerning K value sampling. Furthermore, to circumvent the prohibitive computational overhead induced by the second-order partial derivatives in $S ( \theta )$ , we employ a highly time-efficient approximation strategy (We show the acceptable error in Figure 10):

$$
\frac { \partial ^ { 2 } f _ { a } } { \partial a \partial \theta } \cdot \Delta \theta \approx \frac { \partial f _ { a } ^ { \theta ^ { \prime } } } { \partial a ^ { \theta ^ { \prime } } } - \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } }\tag{8}
$$

During the tuning phase, we implement differential optimization strategies tailored to the derived parameter importance. Specifically, we instantiate this pipeline at two distinct granularities:

(1) Neuron-level: The minimal unit for computing $\Delta E _ { \mathcal { D } \tau } ^ { \theta ^ { I } } ( a )$ is defined as an individual neuron (i.e., a single scalar value within a parameter matrix). We selectively update only the top 20% most critical neurons while freezing the remaining 80%, optimizing via the standard next-token cross-entropy loss for SFT. For example, when applied to the Mistral-7B model, this strategy restricts parameter updates to exactly 1.4B out of the 7B total neurons.

(2) Component-level: The minimal unit is designated as an entire parameter matrix (specifically, $W _ { q } , W _ { k } , \mathbf { \bar { W } } _ { v } , W _ { o }$ within each attention head, and $W _ { u p } , W _ { g a t e } , W _ { d o w n }$ in the MLP). We integrate Low-Rank Adaptation (LoRA) [14] into this pipeline by dynamically allocating ranks according to the component importance rankings. Higher-ranked components are assigned proportionally higher LoRA ranks, scaling within a discrete range of [1, 32] while maintaining an overall average rank of 8.

## 5 Experiments

## 5.1 Setups

We employ Mistral-7B [18] as our primary evaluation model, while further incorporating LLaMA-2-13B [33] and Qwen3-30B [37] to validate the scalability of our approach. We evaluate LLM fine-tuning performance across three primary domains: Natural Language Understanding (NLU) across the GLUE benchmark (SST-2, MRPC, QQP, MNLI, and RTE) [34], Logical Reasoning (LR) via BOOL [31], and Mathematical Reasoning (MR) using Arithmetic [30]<sup>4</sup>. To examine localization efficacy under varying adaptation complexities, we stratify the Arithmetic benchmark into hierarchical subtasks ranging from 2-digit to 7-digit operations and 1-step to 6-step computations. Additionally, to facilitate a granular examination of interpretability outcomes—specifically through circuit graph and component statistics analyses—we incorporate established benchmarks from the mechanistic interpretability literature: Indirect Object Identification (IOI) [35], Induction [8], WinoGrande [1], Genderr [22], and Docstring [12]. Detailed prompt configurations for these datasets are provided in Appendix D.

We evaluate our proposed localization methods—denoted as Ours (N) for the neuron-level pipeline and Ours (C) for the component-level pipeline—against a suite of state-of-the-art baselines. These comprise gradient-guided approaches (Graft [25] and WAGLE [17]) alongside causal effect-guided methods (CLUE [5], FLU [11], and CircuitLoRA [36]). To comprehensively assess post-tuning efficacy, we monitor two primary metrics: Target Task Accuracy (TTA) and Pervasiveness Task Accuracy (PTA). TTA is directly derived from the label accuracy on the target tasks (e.g., average accuracy from all tasks in GLUE, BOOL, and Arithmetic). Conversely, PTA quantifies the degradation of pre-existing general capabilities; it is computed as the macro-average across a diverse array of generalized benchmarks using the lm-evaluation-harness framework [10].

For the fine-tuning optimization, we minimize the cross-entropy loss over the target task labels. The training hyperparameters are standardized to a learning rate of $1 \times 1 0 ^ { - 5 } .$ , 3 training epochs, and a default LoRA rank of 8. All experiments are executed on NVIDIA RTX A6000 with corresponding CUDA environments.

## 5.2 Main Results

<table><tr><td rowspan="2">Method</td><td colspan="2">LR</td><td colspan="2">NLU</td><td colspan="2">MR</td></tr><tr><td>TTA</td><td>PTA</td><td>TTA</td><td>PTA</td><td>TTA</td><td>PTA</td></tr><tr><td>Graft</td><td> $8 2 . 7 6 _ { \pm 0 . 4 2 }$ </td><td> $3 3 . 2 7 _ { \pm 0 . 0 4 }$ </td><td> $8 4 . 9 6 _ { \pm 0 . 0 8 }$ </td><td> $3 1 . 2 4 { \scriptstyle \pm 0 . 1 4 }$ </td><td> $2 4 . 1 9 _ { \pm 2 . 1 3 }$ </td><td> $4 4 . 3 1 _ { \pm 0 . 1 1 }$ </td></tr><tr><td>FLU</td><td> $8 8 . 5 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $4 2 . 5 7 { \scriptstyle \pm 0 . 0 9 }$ </td><td> $8 4 . 2 7 _ { \pm 0 . 0 2 }$ </td><td> $3 6 . 7 2 _ { \pm 0 . 1 3 }$ </td><td> $8 7 . 1 6 _ { \pm 1 . 0 7 }$ </td><td> $4 9 . 5 2 _ { \pm 0 . 0 1 }$ </td></tr><tr><td>WAGLE</td><td> $8 9 . 5 5 _ { \pm 0 . 2 8 }$ </td><td> $4 4 . 3 1 _ { \pm 0 . 0 6 }$ </td><td> $8 5 . 2 8 _ { \pm 0 . 1 1 }$ </td><td> $4 3 . 5 7 _ { \pm 0 . 0 4 }$ </td><td> $6 6 . 3 9 _ { \pm 1 . 1 5 }$ </td><td> $5 1 . 2 3 _ { \pm 0 . 0 9 }$ </td></tr><tr><td>CLUE</td><td> $9 0 . 4 9 _ { \pm 0 . 4 4 }$ </td><td> $4 5 . 4 2 _ { \pm 0 . 0 2 }$ </td><td> $8 5 . 1 7 _ { \pm 0 . 0 7 }$ </td><td> $4 5 . 6 1 _ { \pm 0 . 1 0 }$ </td><td> $5 2 . 7 8 _ { \pm 1 . 5 5 }$ </td><td> $5 3 . 2 1 _ { \pm 0 . 1 2 }$ </td></tr><tr><td>CircuitLoRA</td><td> $9 7 . 6 7 _ { \pm 0 . 1 3 }$ </td><td> $6 0 . 6 2 _ { \pm 0 . 1 1 }$ </td><td> $8 5 . 0 9 _ { \pm 0 . 0 6 }$ </td><td> $6 1 . 5 5 _ { \pm 0 . 0 9 }$ </td><td> $9 8 . 6 0 _ { \pm 0 . 1 2 }$ </td><td> $5 9 . 6 6 _ { \pm 0 . 1 4 }$ </td></tr><tr><td>Ours (N)</td><td> $9 8 . 5 1 _ { \pm 0 . 0 9 }$ </td><td> $6 0 . 5 3 _ { \pm 0 . 0 5 }$ </td><td> $8 6 . 3 1 _ { \pm 0 . 1 3 }$ </td><td> $6 2 . 0 2 _ { \pm 0 . 0 7 }$ </td><td> $9 8 . 7 9 _ { \pm 0 . 1 4 }$ </td><td> $6 0 . 1 3 _ { \pm 0 . 1 0 }$ </td></tr><tr><td>Ours (C)</td><td> $\mathbf { 1 0 0 . 0 0 } _ { \pm 0 . 0 0 }$ </td><td> ${ \bf 6 2 . 6 4 _ { \pm 0 . 0 8 } }$ </td><td> $\mathbf { 8 7 . 5 5 _ { \pm 0 . 0 4 } }$ </td><td> ${ \bf 6 3 . 9 5 _ { \pm 0 . 1 1 } }$ </td><td> $\mathbf { 1 0 0 . 0 0 } _ { \pm 0 . 0 0 }$ </td><td> ${ \bf 6 1 . 4 8 _ { \pm 0 . 0 6 } }$ </td></tr></table>

Table 1: Performance comparison of Locating-then-tuning methods.

Table 1 reports the performance of Mistral-7B model of various localization methods across the LR, NLU, MR domains, where the Arithmetic results represent the average performance across the 2- to 5-digit subtasks. As demonstrated, our proposed methods establish a substantial margin over all baseline approaches in both Target Task Accuracy (TTA) and the retention of general capabilities (PTA). This confirms that estimating future parameter importance provides a distinct advantage over conventional mechanistic localization derived solely from current parameters. Interestingly, our component-level pipeline (Ours (C)) significantly outperforms its neuron-level counterpart (Ours (N)), suggesting that an increasingly fine granularity in mechanistic localization does not strictly correlate with improved performance. Furthermore, CircuitLoRA—another baseline utilizing LoRA for fine-tuning—emerges as the second-best method. This observation implies that LoRA’s robust adaptability to single, formalized tasks may largely account for the empirical superiority of the component-level approach over the neuron-level configuration.

![](images/b728b52206932c3fae9cf42cac9255c4edfd73294677e7e72e3a43c2205a2bdd.jpg)  
Figure 2: The scalability of performance and time cost with mode size increasing.

Furthermore, to validate the scalability of our proposed methods, we conduct a comparative analysis across three text-to-text models of varying parameter sizes that lack native complex reasoning capabilities: Mistral-7B, LLaMA-2-13B, and Qwen3-30B. We select the 5-step and 6-step Arithmetic subtasks for evaluation, employing the mean Target Task Accuracy (TTA) as the primary performance metric. To quantify computational efficiency, we report the total time expenditure, defined as the sum of the localization and fine-tuning durations.

As illustrated in Figure 2, our method consistently sustains optimal fine-tuning performance as the model scale increases, demonstrating its robust efficacy on LLMs. Concurrently, the fine-tuning time cost exhibits a relatively gradual growth rate, though we acknowledge this may be partially attributed to the inherent simplicity of the chosen target tasks. Conversely, the localization time cost escalates prominently with model size. This sharp increase confirms that the computationa complexity of mechanistic localization strictly exceeds O(n), a consequence of the auxiliary spatial and computational overhead mandated by interpretability procedures. Despite this inherent overhead, our approach maintains nearly the lowest overall computational complexity among the evaluated baselines, underscoring its practical viability for large-scale LLMs.

Finally, to further explore the broader potential of our approach, we conduct multi-task joint training experiments, as detailed in Appendix E. Existing literature demonstrates that joint training frequently exacerbates inter-task conflicts [5, 38], a phenomenon primarily manifesting within shared neurons that are simultaneously responsible for multiple distinct tasks [9]. By jointly evaluating task accuracy and the prevalence of these conflicting neurons, we empirically demonstrate that our proposed localization method provides crucial foresight. This predictive capability substantially mitigates inter-task interference by proactively avoiding the allocation of shared, conflicting neurons during the fine-tuning process.

## 5.3 Ablation Study

To elucidate the intrinsic contributions of our proposed framework, we compare our method against four distinct localization strategies:

(1) Full-Param: Standard full-parameter fine-tuning without any localization, designed to isolate the overall impact of applying a localization mechanism.

(2) Random: Random parameter localization that rigorously preserves the exact rank distribution of our method, isolating the effect of merely reducing the trainable parameter count.

(3) Static: Localization derived directly from the current model parameters (Localization from θ), serving as a baseline for pre-update parameter importance.

(4) Probing: Localization based on the model post-probing SFT (Localization from $\theta ^ { \prime } )$ , designed to contrast the efficacy of the intermediate state $\theta ^ { \prime }$ against our estimated ideal state $\hat { \theta } ^ { I }$

As presented in Table 2, the results not only confirm that localization provides critical interpretabilitydriven guidance, but also conclusively demonstrate that anticipating "future" parameter states offers significantly greater foresight than relying on current parameters. Additionally, Appendix G investi gates the robustness of our framework against diverse stochasticity.

Although the Probing strategy performs comparably to our method on standard benchmarks, we conduct further evaluations on tasks of escalating difficulty to explicitly disentangle their capabilities. Specifically, we utilize Arithmetic subtasks ranging from $2$ to 7 digits and 1 to 6 reasoning steps. As illustrated in Figure 3, the performance of both the Static and Probing strategies degrades precipitously as task complexity increases, whereas our method preserves substantial robustness. This phenomenon occurs because harder tasks

<table><tr><td rowspan="2">Strategy</td><td colspan="3">Neuron-Level</td><td colspan="3">Component-Level</td></tr><tr><td>LR</td><td>NLU</td><td>MR</td><td>LR</td><td>NLU</td><td>MR</td></tr><tr><td>Full-Param</td><td>69.27</td><td>94.03</td><td>15.66</td><td>99.73</td><td>96.33</td><td>71.83</td></tr><tr><td>Random</td><td>88.31</td><td>95.03</td><td>43.77</td><td>99.64</td><td>96.86</td><td>76.41</td></tr><tr><td>Static</td><td>87.49</td><td>95.52</td><td>55.83</td><td>99.75</td><td>96.33</td><td>83.52</td></tr><tr><td>Probing</td><td>92.76</td><td>96.10</td><td>87.16</td><td>99.96</td><td>96.21</td><td>99.60</td></tr><tr><td>Ours</td><td>98.51</td><td>96.31</td><td>98.79</td><td>100.00</td><td>97.55</td><td>100.00</td></tr></table>

Table 2: Ablation study across datasets at neuron and component levels.

necessitate significantly larger parameter updates during fine-tuning, thereby exacerbating the representational lag inherent in both the initial parameters θ and the probing state $\theta ^ { \prime }$ . These findings substantiate that our method is uniquely advantageous for complex tasks where the model struggles to rapidly adapt. Conversely, for rudimentary SFT tasks under extreme computational constraints, the Probing or Static methods may serve as viable, efficient alternatives.

![](images/83557dd6b2f08a122b9acdba2b584d8612e33dc5b9ac6051dc840e540129c98c.jpg)  
(a) Ablation on different digits.

![](images/da09ab9b96a220f8e57e8e641eb083ccacb39fb3d6d5b4ced32b44cc016e62e8.jpg)  
(b) Ablation on different steps.  
Figure 3: Ablation study results on the arithmetic task. (a) Performance variations across different digits. (b) Performance variations across different reasoning steps.

Furthermore, Appendix H provides an in-depth mechanistic interpretability analysis. From a circuit perspective, this analysis corroborates that our estimated parameters yield a mechanistic circuit more closely aligned with that of the fully fine-tuned model, significantly outperforming direct localization from either the initial state θ or the probing state $\theta ^ { \prime }$ . Additionally, Appendix I visualizes the respective circuit graphs for all three parameter states: $\theta , \theta ^ { \prime } ,$ , and $\hat { \theta } ^ { I }$ . These visualizations explicitly demonstrate that our $K \cdot$ -value extrapolation successfully identifies critical "intermediate nodes." These intermediate nodes typically encode distinct subskills within the computational pathways, constituting essential components of the underlying task mechanism.

## 5.4 Analysis about Probing SFT

In this section, we investigate the empirical impact of various probing SFT setups. Using the fully fine-tuned model as a surrogate for the ideal model, we adopt the component-level estimation pipeline on the Mistral-7B model for the IOI task as our evaluation baseline. We systematically compare probing models updated on varying proportions of the training dataset: 20%, 10%, 5%, 2%, and 1%, alongside extreme scenarios restricted to merely 2 samples and 1 sample. For each configuration, we run 10 random seeds and sample 50 distinct K values per run within a valid range and compute the Top@50 overlap between the component rankings of our estimated $\hat { \theta } ^ { I }$ and those of the surrogate ideal model.

![](images/76facc423388bc426562f000f920b6eab0e86082cddc5c5bd6673109ed146412.jpg)  
Figure 4: Top 50 overlap with Full-Parameter fine-tuning model across different probing SFT setups.

As illustrated in Figure 4, reducing the sample size used for the probing SFT consistently improves the mean overlap with the ideal model up to a certain threshold. Moreover, we also validate that the learning rate and epoch expose the similar pattern in Appendix J. These empirical trends corroborate our theoretical assertion in Section 4.1: maintaining a sufficiently small parameter update during probing is imperative to preserve the necessary degrees of freedom for the extrapolation term $K \cdot \Delta \theta$ However, in extreme low-data regimes (i.e., 1 or 2 samples), the variance of the estimations escalates sharply, precipitating a noticeable decline in the mean overlap. We attribute this degradation to the inability of such minuscule sample sets to adequately capture the underlying distribution of the target task, which inadvertently injects significantly biased gradient directions into $\Delta \theta .$

## 6 Conclusion and Limitations

This paper presents a breakthrough advancement in mechanistic localization prior to Supervised Fine-Tuning (SFT), enabling the prediction of post-SFT mechanisms to proactively facilitate the tuning process. Theoretically, we bridge mechanistic interpretability with parameter updates via Taylor expansion, breaking the fundamental post-hoc nature of interpretability analysis to usher in a predictive, future-oriented paradigm—thereby transforming interpretability from a retrospective diagnostic tool into an actionable, proactive optimizer. Practically, utilizing only the base model parameters and a probing SFT model trained on a mere 1% of the dataset, we effectively estimate and rank the mechanistic importance of parameters for the anticipated post-SFT state. This enables a targeted, fine-grained allocation of tuning intensity while ensuring scalability in both time and performance. Our work inspires future engineering-oriented interpretability research based on following limitations:

• Mechanistic Interpretability for Multi-Token Generation: Current interpretability methods predominantly focus on the internal mechanisms during next-token prediction. However, mainstream post-training SFT typically targets instruction-following tasks that necessitate long-sequence generation [20]. This discrepancy currently restricts the locating-then-tuning paradigm to relatively narrow downstream applications—such as LLM unlearning and knowledge editing—rather than broader instruction task practices. Bridging this gap poses a revolutionary challenge: extracting representative mechanisms and their corresponding parameter regions across multiple forward passes. Exploring this frontier will directly extend our insights into critical domains like reinforcement learning and on-policy distillation.

• Targeted Parameter Updates for Conflicting Mechanisms: Due to neuron polysemanticity, fine-tuning on a specific task frequently degrades capabilities on others. Existing interpretability and localization efforts merely identify these polysemantic neurons, falling short of providing effective mitigation strategies. This remains a limitation of our study, which renders the advantages of multi-objective joint optimization less pronounced. Addressing this issue will tightly couple mechanistic interpretability with post-training optimization, drastically expanding the practical utility of the interpretability field.

## References

[1] Winogrande: An adversarial winograd schema challenge at scale. 2019.

[2] A. Bhaskar, A. Wettig, D. Friedman, and D. Chen. Finding transformer circuits with edge pruning. Advances in Neural Information Processing Systems, 37:18506–18534, 2024.

[3] H. Chen, X. Yang, J. Zhu, and W. Wang. Skill path: Unveiling language skills from circuit graphs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 30210–30217, 2026.

[4] H. Chen, J. Zhu, H. Chen, H. Liu, X. Yang, and W. Wang. Navigating by old maps: The pitfalls of static mechanistic localization in llm post-training, 2026. URL https://arxiv.org/ abs/2605.06076.

[5] H. Chen, J. Zhu, X. Yang, and W. Wang. CLUE: Conflict-guided localization for LLM unlearn ing framework. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=jtRYvazBWv.

[6] H. Chen, J. Zhu, X. Yang, and W. Wang. Rethinking circuit completeness in language models: And, or, and adder gates. Advances in Neural Information Processing Systems, 38:150511– 150540, 2026.

[7] V. K. Chhabra, D. Zhu, and M. M. Khalili. Neuroplasticity and corruption in model mechanisms: A case study of indirect object identification. In L. Chiruzzo, A. Ritter, and L. Wang, editors, Findings of the Association for Computational Linguistics: NAACL 2025, pages 3099–3122, Albuquerque, New Mexico, Apr. 2025. Association for Computational Linguistics. ISBN 979-8- 89176-195-7. doi: 10.18653/v1/2025.findings-naacl.170. URL https://aclanthology. org/2025.findings-naacl.170/.

[8] A. Conmy, A. Mavor-Parker, A. Lynch, S. Heimersheim, and A. Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

[9] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition, 2022. URL https://arxiv.org/abs/2209. 10652.

[10] L. Gao, J. Tow, B. Abbasi, S. Biderman, S. Black, A. DiPofi, C. Foster, L. Golding, J. Hsu, A. Le Noac’h, H. Li, K. McDonell, N. Muennighoff, C. Ociepa, J. Phang, L. Reynolds, H. Schoelkopf, A. Skowron, L. Sutawika, E. Tang, A. Thite, B. Wang, K. Wang, and A. Zou. The language model evaluation harness, 07 2024. URL https://zenodo.org/records/ 12608602.

[11] P. H. Guo, A. Syed, A. Sheshadri, A. Ewart, and G. K. Dziugaite. Mechanistic unlearning: Robust knowledge unlearning and editing via mechanistic localization. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/ forum?id=92oBV5HAGl.

[12] S. Heimersheim and J. Janiak. A circuit for python docstrings in a 4-layer attention-only transformer. In Alignment Forum, 2023.

[13] S. Heimersheim and N. Nanda. How to use and interpret activation patching. arXiv preprint arXiv:2404.15255, 2024.

[14] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9.

[15] N. Jacobson. Lie algebras. Courier Corporation, 2013.

[16] S. Jain, R. Kirk, E. S. Lubana, R. P. Dick, H. Tanaka, T. Rocktäschel, E. Grefenstette, and D. Krueger. Mechanistically analyzing the effects of fine-tuning on procedurally defined tasks. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=A0HKeKl4Nl.

[17] J. Jia, J. Liu, Y. Zhang, P. Ram, N. Baracaldo, and S. Liu. WAGLE: Strategic weight attribution for effective and modular unlearning in large language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview. net/forum?id=VzOgnDJMgh.

[18] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de las Casas, F. Bressand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M.-A. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b, 2023. URL https://arxiv.org/abs/ 2310.06825.

[19] Y. Jing, Z. Dai, J. Hu, Z. Yao, L. Hou, J. Li, and X. Wang. Guiding llm post-training data engineering with model internals from sparse autoencoders, 2026. URL https://arxiv. org/abs/2605.27354.

[20] H. Lai, X. Liu, J. Gao, J. Cheng, Z. Qi, Y. Xu, S. Yao, D. Zhang, J. Du, Z. Hou, et al. A survey of post-training scaling in large language models. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2771–2791, 2025.

[21] T. Lieberum, M. Rahtz, J. Kramár, N. Nanda, G. Irving, R. Shah, and V. Mikulik. Does circuit analysis interpretability scale? evidence from multiple choice capabilities in chinchilla. arXiv preprint arXiv:2307.09458, 2023.

[22] C. Mathwin, G. Corlouer, E. Kran, F. Barez, and N. Nanda. Identifying a preliminary circuit for predicting gendered pronouns in gpt-2 small. URL: https://itch. io/jam/mechint/rate/1889871, page 2, 2023.

[23] A. Messiah. Quantum mechanics. Courier Corporation, 2014.

[24] N. Nanda. Attribution patching: Activation patching at industrial scale. https://www.neelnanda.io/mechanistic-interpretability/ attribution-patching, Feb 2023. URL https://www.neelnanda.io/ mechanistic-interpretability/attribution-patching. Blog post.

[25] A. Panigrahi, N. Saunshi, H. Zhao, and S. Arora. Task-specific skill localization in fine-tuned language models. In A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 27011–27033. PMLR, 23–29 Jul 2023.

[26] N. Prakash, T. R. Shaham, T. Haklay, Y. Belinkov, and D. Bau. Fine-tuning enhances existing mechanisms: A case study on entity tracking. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=8sKcAWOf2D.

[27] D. Rai, Y. Zhou, S. Feng, A. Saparov, and Z. Yao. A practical review of mechanistic interpretability for transformer-based language models. arXiv preprint arXiv:2407.02646, 2024.

[28] L. Sharkey, B. Chughtai, J. Batson, J. Lindsey, J. Wu, L. Bushnaq, N. Goldowsky-Dill, S. Heimersheim, A. Ortega, J. Bloom, et al. Open problems in mechanistic interpretability. arXiv preprint arXiv:2501.16496, 2025.

[29] S. Somvanshi, M. M. Islam, A. Rafe, A. G. Tusti, A. Chakraborty, A. Baitullah, T. I. Chowdhury, N. Alnawmasi, A. Dutta, and S. Das. Bridging the black box: a survey on mechanistic interpretability in ai. ACM Computing Surveys, 58(8):1–35, 2026.

[30] A. Srivastava, A. Rastogi, A. Rao, A. A. M. Shoeb, A. Abid, A. Fisch, A. R. Brown, A. Santoro, A. Gupta, A. Garriga-Alonso, A. Kluska, A. Lewkowycz, A. Agarwal, A. Power, A. Ray, A. Warstadt, A. W. Kocurek, A. Safaya, A. Tazarv, A. Xiang, A. Parrish, A. Nie, A. Hussain, A. Askell, A. Dsouza, A. Slone, A. Rahane, A. S. Iyer, A. J. Andreassen, A. Madotto, A. Santilli, A. Stuhlmüller, A. M. Dai, A. La, A. K. Lampinen, A. Zou, A. Jiang, A. Chen, A. Vuong, A. Gupta, A. Gottardi, A. Norelli, A. Venkatesh, A. Gholamidavoodi, A. Tabassum, A. Menezes, A. Kirubarajan, A. Mullokandov, A. Sabharwal, A. Herrick, A. Efrat, A. Erdem, A. Karaka¸s, B. R. Roberts, B. S. Loe, B. Zoph, B. Bojanowski, B. Özyurt, B. Hedayatnia, B. Neyshabur, B. Inden, B. Stein, B. Ekmekci, B. Y. Lin, B. Howald, B. Orinion, C. Diao, C. Dour, C. Stinson, C. Argueta, C. Ferri, C. Singh, C. Rathkopf, C. Meng, C. Baral, C. Wu, C. Callison-Burch, C. Waites, C. Voigt, C. D. Manning, C. Potts, C. Ramirez, C. E. Rivera, C. Siro, C. Raffel, C. Ashcraft, C. Garbacea, D. Sileo, D. Garrette, D. Hendrycks, D. Kilman, D. Roth, C. D. Freeman, D. Khashabi, D. Levy, D. M. González, D. Perszyk, D. Hernandez, D. Chen, D. Ippolito,

D. Gilboa, D. Dohan, D. Drakard, D. Jurgens, D. Datta, D. Ganguli, D. Emelin, D. Kleyko, D. Yuret, D. Chen, D. Tam, D. Hupkes, D. Misra, D. Buzan, D. C. Mollo, D. Yang, D.-H. Lee, D. Schrader, E. Shutova, E. D. Cubuk, E. Segal, E. Hagerman, E. Barnes, E. Donoway, E. Pavlick, E. Rodolà, E. Lam, E. Chu, E. Tang, E. Erdem, E. Chang, E. A. Chi, E. Dyer, E. Jerzak, E. Kim, E. E. Manyasi, E. Zheltonozhskii, F. Xia, F. Siar, F. Martínez-Plumed, F. Happé, F. Chollet, F. Rong, G. Mishra, G. I. Winata, G. de Melo, G. Kruszewski, G. Parascandolo, G. Mariani, G. X. Wang, G. Jaimovitch-Lopez, G. Betz, G. Gur-Ari, H. Galijasevic, H. Kim, H. Rashkin, H. Hajishirzi, H. Mehta, H. Bogar, H. F. A. Shevlin, H. Schuetze, H. Yakura, H. Zhang, H. M. Wong, I. Ng, I. Noble, J. Jumelet, J. Geissinger, J. Kernion, J. Hilton, J. Lee, J. F. Fisac, J. B. Simon, J. Koppel, J. Zheng, J. Zou, J. Kocon, J. Thompson, J. Wingfield, J. Kaplan, J. Radom, J. Sohl-Dickstein, J. Phang, J. Wei, J. Yosinski, J. Novikova, J. Bosscher, J. Marsh, J. Kim, J. Taal, J. Engel, J. Alabi, J. Xu, J. Song, J. Tang, J. Waweru, J. Burden, J. Miller, J. U. Balis, J. Batchelder, J. Berant, J. Frohberg, J. Rozen, J. Hernandez-Orallo, J. Boudeman, J. Guerr, J. Jones, J. B. Tenenbaum, J. S. Rule, J. Chua, K. Kanclerz, K. Livescu, K. Krauth, K. Gopalakrishnan, K. Ignatyeva, K. Markert, K. Dhole, K. Gimpel, K. Omondi, K. W. Mathewson, K. Chiafullo, K. Shkaruta, K. Shridhar, K. McDonell, K. Richardson, L. Reynolds, L. Gao, L. Zhang, L. Dugan, L. Qin, L. Contreras-Ochando, L.-P. Morency, L. Moschella, L. Lam, L. Noble, L. Schmidt, L. He, L. Oliveros-Colón, L. Metz, L. K. Senel, M. Bosma, M. Sap, M. T. Hoeve, M. Farooqi, M. Faruqui, M. Mazeika, M. Baturan, M. Marelli, M. Maru, M. J. Ramirez-Quintana, M. Tolkiehn, M. Giulianelli, M. Lewis, M. Potthast, M. L. Leavitt, M. Hagen, M. Schubert, M. O. Baitemirova, M. Arnaud, M. McElrath, M. A. Yee, M. Cohen, M. Gu, M. Ivanitskiy, M. Starritt, M. Strube, M. Sw˛edrowski, M. Bevilacqua, M. Yasunaga, M. Kale, M. Cain, M. Xu, M. Suzgun, M. Walker, M. Tiwari, M. Bansal, M. Aminnaseri, M. Geva, M. Gheini, M. V. T, N. Peng, N. A. Chi, N. Lee, N. G.-A. Krakover, N. Cameron, N. Roberts, N. Doiron, N. Martinez, N. Nangia, N. Deckers, N. Muennighoff, N. S. Keskar, N. S. Iyer, N. Constant, N. Fiedel, N. Wen, O. Zhang, O. Agha, O. Elbaghdadi, O. Levy, O. Evans, P. A. M. Casares, P. Doshi, P. Fung, P. P. Liang, P. Vicol, P. Alipoormolabashi, P. Liao, P. Liang, P. W. Chang, P. Eckersley, P. M. Htut, P. Hwang, P. Miłkowski, P. Patil, P. Pezeshkpour, P. Oli, Q. Mei, Q. Lyu, Q. Chen, R. Banjade, R. E. Rudolph, R. Gabriel, R. Habacker, R. Risco, R. Millière, R. Garg, R. Barnes, R. A. Saurous, R. Arakawa, R. Raymaekers, R. Frank, R. Sikand, R. Novak, R. Sitelew, R. L. Bras, R. Liu, R. Jacobs, R. Zhang, R. Salakhutdinov, R. A. Chi, S. R. Lee, R. Stovall, R. Teehan, R. Yang, S. Singh, S. M. Mohammad, S. Anand, S. Dillavou, S. Shleifer, S. Wiseman, S. Gruetter, S. R. Bowman, S. S. Schoenholz, S. Han, S. Kwatra, S. A. Rous, S. Ghazarian, S. Ghosh, S. Casey, S. Bischoff, S. Gehrmann, S. Schuster, S. Sadeghi, S. Hamdan, S. Zhou, S. Srivastava, S. Shi, S. Singh, S. Asaadi, S. S. Gu, S. Pachchigar, S. Toshniwal, S. Upadhyay, S. S. Debnath, S. Shakeri, S. Thormeyer, S. Melzi, S. Reddy, S. P. Makini, S.-H. Lee, S. Torene, S. Hatwar, S. Dehaene, S. Divic, S. Ermon, S. Biderman, S. Lin, S. Prasad, S. Piantadosi, S. Shieber, S. Misherghi, S. Kiritchenko, S. Mishra, T. Linzen, T. Schuster, T. Li, T. Yu, T. Ali, T. Hashimoto, T.-L. Wu, T. Desbordes, T. Rothschild, T. Phan, T. Wang, T. Nkinyili, T. Schick, T. Kornev, T. Tunduny, T. Gerstenberg, T. Chang, T. Neeraj, T. Khot, T. Shultz, U. Shaham, V. Misra, V. Demberg, V. Nyamai, V. Raunak, V. V. Ramasesh, vinay uday prabhu, V. Padmakumar, V. Srikumar, W. Fedus, W. Saunders, W. Zhang, W. Vossen, X. Ren, X. Tong, X. Zhao, X. Wu, X. Shen, Y. Yaghoobzadeh, Y. Lakretz, Y. Song, Y. Bahri, Y. Choi, Y. Yang, S. Hao, Y. Chen, Y. Belinkov, Y. Hou, Y. Hou, Y. Bai, Z. Seid, Z. Zhao, Z. Wang, Z. J. Wang, Z. Wang, and Z. Wu. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=uyTL5Bvosj. Featured Certification.

[31] M. Suzgun, N. Scales, N. Schärli, S. Gehrmann, Y. Tay, H. W. Chung, A. Chowdhery, Q. Le, E. Chi, D. Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13003–13051, 2023.

[32] A. Syed, C. Rager, and A. Conmy. Attribution patching outperforms automated circuit discovery. In Y. Belinkov, N. Kim, J. Jumelet, H. Mohebbi, A. Mueller, and H. Chen, editors, Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pages 407–416, Miami, Florida, US, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.blackboxnlp-1.25. URL https://aclanthology.org/2024. blackboxnlp-1.25/.

[33] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. C. Ferrer, M. Chen, G. Cucurull, D. Esiobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Korenev, P. S. Koura, M.-A. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Saladi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models, 2023. URL https://arxiv.org/abs/2307.09288.

[34] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. 2019. In the Proceedings of ICLR.

[35] K. R. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In The Eleventh International Conference on Learning Representations.

[36] X. Wang, Y. Hu, W. Du, R. Cheng, B. Wang, and D. Zou. Towards understanding fine-tuning mechanisms of LLMs via circuit analysis. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?id=45EIiFd6Oa.

[37] A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, C. Zheng, D. Liu, F. Zhou, F. Huang, F. Hu, H. Ge, H. Wei, H. Lin, J. Tang, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Zhou, J. Lin, K. Dang, K. Bao, K. Yang, L. Yu, L. Deng, M. Li, M. Xue, M. Li, P. Zhang, P. Wang, Q. Zhu, R. Men, R. Gao, S. Liu, S. Luo, T. Li, T. Tang, W. Yin, X. Ren, X. Wang, X. Zhang, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Zhang, Y. Wan, Y. Liu, Z. Wang, Z. Cui, Z. Zhang, Z. Zhou, and Z. Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

[38] L. Yang, S. Ding, and D. Xiong. A local perturbation theory for cross-domain interference and recovery in multi-domain rl, 2026. URL https://arxiv.org/abs/2606.02398.

[39] F. Yin, X. Ye, and G. Durrett. Lofit: Localized fine-tuning on LLM representations. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum?id=dfiXFbECSZ.

[40] H. Zhang, Z. Zhang, M. Wang, Z. Su, Y. Wang, Q. Wang, S. Yuan, E. Nie, X. Duan, Q. Xue, et al. Locate, steer, and improve: A practical survey of actionable mechanistic interpretability in large language models. Computer Science Review, 62:101011, 2026.

[41] N. Zhang, Y. Yao, B. Tian, P. Wang, S. Deng, M. Wang, Z. Xi, S. Mao, J. Zhang, Y. Ni, et al. A comprehensive study of knowledge editing for large language models. arXiv preprint arXiv:2401.01286, 2024.

## A The Derivation of Equation 2

In this appendix, we provide the detailed derivation for Equation 2, which approximates the causal effect of an infinitesimal parameter update $\theta ^ { \prime }$ using terms computed entirely under the current parameters θ.

First, we treat any activation a as a differentiable function of the model parameters θ. Applying a first-order Taylor expansion with respect to $\theta ,$ we can approximate the clean and corrupted activations under $\theta ^ { \prime }$ as follows:

$$
{ A ^ { \theta } } ^ { \prime } ( x _ { r } ) \approx A ^ { \theta } ( x _ { r } ) + \Delta \theta \cdot \frac { \partial A ^ { \theta } ( x _ { r } ) } { \partial \theta }\tag{9}
$$

$$
{ A ^ { \theta } } ^ { \prime } ( x _ { c } ) \approx A ^ { \theta } ( x _ { c } ) + \Delta \theta \cdot \frac { \partial A ^ { \theta } ( x _ { c } ) } { \partial \theta }\tag{10}
$$

Subtracting these two equations provides a linear approximation of the activation difference, separating the original difference from the effect of the parameter update:

$$
{ A ^ { \theta } } ^ { \prime } ( x _ { c } ) - { A ^ { \theta } } ^ { \prime } ( x _ { r } ) \approx ( A ^ { \theta } ( x _ { c } ) - A ^ { \theta } ( x _ { r } ) ) + \Delta \theta \cdot \left( { \frac { \partial A ^ { \theta } ( x _ { c } ) } { \partial \theta } } - { \frac { \partial A ^ { \theta } ( x _ { r } ) } { \partial \theta } } \right)\tag{11}
$$

Next, we approximate the gradient term. Let the gradient under the current parameters $\theta$ be denoted as $\begin{array} { r } { g ( \theta ) = \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } } \end{array}$ . Applying a first-order Taylor expansion to the gradient for the updated parameters $\theta ^ { \prime }$ yields:

$$
g ( \theta ^ { \prime } ) \approx g ( \theta ) + \Delta \theta \cdot \frac { \partial g ( \theta ) } { \partial \theta }\tag{12}
$$

By substituting the definition of $g ( \theta )$ into the derivative term, we can further expand it into a mixed second-order derivative:

$$
{ \frac { \partial g ( \theta ) } { \partial \theta } } = { \frac { \partial } { \partial \theta } } \left( { \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } } \right) = { \frac { \partial ^ { 2 } f _ { a } ^ { \theta } } { \partial a ^ { \theta } \partial \theta } }\tag{13}
$$

Finally, we substitute both the activation difference approximation and the gradient approximation back into the Attribution Patching formulation for $\mathbf { \dot { \theta ^ { \prime } } }$ . By expanding the product and omitting higher-order terms $\mathcal { O } ( ( \Delta \theta ) ^ { 2 } )$ , we arrive at our final formulation:

$$
\begin{array} { r l } { f _ { \alpha } ^ { \mathrm { e } } ( A ( x , x ) ) - f _ { \alpha } ^ { \mathrm { e } } ( A ( z , x ) ) \approx ( A ^ { 0 ^ { \mathrm { e } } } ( x , x ) - A ^ { 0 ^ { \mathrm { e } } } ( x , x ) ) \cdot \frac { \partial f _ { \alpha } ^ { \mathrm { e } } } { \partial A ^ { 0 } } \Bigg [ \mathrm { a } x \delta ^ { \mathrm { e } } ( x , x ) } \\ & { \approx \Bigg [ ( A ^ { 0 } ( x , x ) - A ^ { 0 } ( x , x ) ) + \Delta B \cdot ( \frac { A ^ { 0 } ( x , x ) } { \partial B } - \frac { \partial A ^ { 0 } ( x , x ) } { \partial B } ) } \\ & { \quad \cdot [ g ( B ) + \Delta B \cdot \frac { \partial f _ { \alpha } ^ { \mathrm { e } } ( x , x ) } { \partial A ^ { 0 } } ] } \\ & { = ( A ^ { 0 } ( x , x ) - A ^ { 0 } ( x , x ) ) \cdot \frac { \partial f _ { \alpha } ^ { \mathrm { e } } ( x , x ) } { \partial a ^ { 0 } } + \Delta B \cdot ( A ^ { 0 } ( x , x ) - A ^ { 0 } ( x , x ) ) \cdot \frac { \partial ^ { 2 } f _ { \alpha } ^ { \mathrm { e } } ( x , x ) } { \partial a \partial B } } \\ & { \quad + \Delta a \cdot ( \frac { A B ^ { 0 } ( x , x ) } { \partial B } - \frac { \partial A ^ { 0 } ( x , x ) } { \partial B } ) \cdot \frac { \partial f _ { \alpha } ^ { \mathrm { e } } ( x , x ) } { \partial a ^ { 0 } } + \mathcal { O } ( \Delta B \delta ^ { 2 } ) } \\ & { \quad \lesssim f _ { \alpha } ^ { \mathrm { e } } ( A ( x , x ) ) - f _ { \alpha } ^ { \mathrm { e } } ( A ( x , x ) ) } \\ &  \quad + \Delta a \cdot [ ( \frac { A B ^ { 0 } ( x , x ) } { \partial B } - \frac { B A ^ { 0 } ( x , x ) } { \partial B } ) \cdot \frac  \partial f _ { \alpha } ^ { \mathrm { e } } ( x , x ) \end{array}
$$

where $S ( \theta )$ effectively captures the sensitivity of the target neuron $\boldsymbol { a } ^ { \prime } \mathbf { s }$ attribution value to the parameter updating.

## B Derivation of the Translation Operator for Sensitivity $S ( \theta )$

In this section, we provide the mathematical proof demonstrating how the sensitivity at the ideal state, $S ( \hat { \theta } ^ { I } )$ ), can be evaluated as $S ( \theta + K \Delta \theta )$ by circumventing the intractable higher-order derivatives.

Simplifying the Notation and Revealing the Identity. Let $\Delta A ^ { \theta } = A ^ { \theta } ( x _ { c } ) - A ^ { \theta } ( x _ { r } )$ . According to Equation 3, the sensitivity $S ( \theta )$ can be rewritten as:

$$
S ( \theta ) = \frac { \partial \Delta A ^ { \theta } } { \partial \theta } \cdot \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } + \Delta A ^ { \theta } \cdot \frac { \partial ^ { 2 } f _ { a } ^ { \theta } } { \partial a \partial \theta }
$$

Applying the product rule in reverse reveals that $S ( \theta )$ is strictly equivalent to the gradient of the attribution score with respect to the parameters:

$$
S ( \theta ) \equiv \nabla _ { \theta } \left[ \Delta A ^ { \theta } \cdot \frac { \partial f _ { a } ^ { \theta } } { \partial a ^ { \theta } } \right] \equiv \nabla _ { \theta } [ \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta } ( a ) ]
$$

This insight indicates that our iterative update formula, $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k + 1 } } ( a ) = \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k } } ( a ) + \Delta \theta ^ { k } \cdot S ( \theta ^ { k } )$ , is fundamentally solving an Ordinary Differential Equation (ODE) via Euler’s Method.

The Differential Operator and Infinite Series. Assuming we perform infinite micro-updates along a fixed direction $\Delta \theta .$ , we define a differential operator $\bar { D ; }$

$$
D = \Delta \theta \cdot \nabla _ { \theta }
$$

which computes the directional derivative along $\Delta \theta .$ Utilizing the Taylor expansion, the sensitivity at step $k$ unfolds into an infinite series:

$$
S ( \theta ^ { k } ) = S ( \theta ^ { 0 } ) + k D S ( \theta ^ { 0 } ) + \frac { k ^ { 2 } } { 2 ! } D ^ { 2 } S ( \theta ^ { 0 } ) + \frac { k ^ { 3 } } { 3 ! } D ^ { 3 } S ( \theta ^ { 0 } ) + \dots
$$

Notice that the operators $D ^ { 2 } , D ^ { 3 } , \ldots$ . correspond exactly to the computationally intractable secondorder Hessians, third-order tensors, and beyond.

Operator Folding and Translation. In functional analysis, this infinite series perfectly matche the Taylor expansion of $e ^ { x }$ . We can rigorously fold it into an exponential operator:

$$
S ( \theta ^ { k } ) = \left( \sum _ { n = 0 } ^ { \infty } \frac { ( k D ) ^ { n } } { n ! } \right) S ( \theta ^ { 0 } ) = e ^ { k \Delta \theta \cdot \nabla _ { \theta } } S ( \theta ^ { 0 } )
$$

In Lie Algebra [15] and Quantum Mechanics [23], $e ^ { v \cdot \nabla }$ represents the Translation Operator, which systematically shifts the independent variable of any analytic function: $e ^ { v \cdot \nabla } f ( x ) \ { \equiv } \ f ( x + v )$ Applying this property to our exponential operator yields:

$$
e ^ { k \Delta \theta \cdot \nabla _ { \theta } } S ( \theta ^ { 0 } ) \equiv S ( \theta ^ { 0 } + k \Delta \theta ) \implies S ( \theta ^ { k } ) = S ( \theta ^ { 0 } + k \Delta \theta )
$$

Consequently, for the final ideal state I (reached after K steps), we arrive at $S ( \theta ^ { I } ) = S ( \theta ^ { 0 } + K \Delta \theta )$ This elegant identity demonstrates that to evaluate the sensitivity at the ideal state, we merely need to scale the step distance K, completely avoiding the computation of higher-order derivative terms.

## C Why the Probing Model’s Attribution Cannot Directly Substitute the Ideal Model

In this appendix, we demonstrate why it is theoretically flawed to directly substitute the attribution patching value of the probing model, denoted here as an arbitrary discrete step $\theta ^ { k }$ , for that of the ideal model $\theta ^ { I }$ . That is, we prove why the approximation $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \stackrel { \mathbf { \bar { \alpha } } } { \approx } \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k } } ( a )$ does not generally hold.

To simplify the notation, let $X _ { 0 } = { \cal A } ^ { \theta ^ { 0 } } ( x _ { c } ) - { \cal A } ^ { \theta ^ { 0 } } ( x _ { r } )$ and $\begin{array} { r } { Y _ { 0 } = \frac { \partial f _ { a } ^ { \theta ^ { 0 } } } { \partial a ^ { \theta ^ { 0 } } } } \end{array}$ represent the activation difference and gradient for the current model $\theta ^ { 0 }$ , respectively. Similarly, let $X _ { k } = { \cal { A } } ^ { \theta ^ { k } } ( x _ { c } ) - { \cal { A } } ^ { \theta ^ { k } } ( x _ { r } )$ and $\begin{array} { r } { Y _ { k } = \frac { \partial f _ { a } ^ { \theta ^ { k } } } { \partial a ^ { \theta ^ { k } } } } \end{array}$ represent the corresponding terms for the probing model at step k.

Recall our causal effect estimation formulation:

$$
\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \approx X _ { 0 } Y _ { 0 } + \frac { 1 } { 2 } \left[ S ( \theta ^ { 0 } ) + S ( \theta ^ { k } ) \right] \cdot \Delta \theta\tag{14}
$$

Expanding the sensitivity term $S ( \theta ^ { 0 } )$ · ∆θ yields:

$$
S ( \theta ^ { 0 } ) \cdot \Delta \theta = \left( \frac { \partial X _ { 0 } } { \partial \theta } \cdot \Delta \theta \right) \cdot Y _ { 0 } + X _ { 0 } \cdot \left( \frac { \partial Y _ { 0 } } { \partial \theta } \cdot \Delta \theta \right)\tag{15}
$$

At this juncture, we introduce the $H V P$ Assumption (which will be detailed in Section 4.2), allowing us to approximate the gradient variation as $\begin{array} { r } { \frac { \dot { \partial } Y _ { 0 } } { \partial \theta } \cdot \dot { \Delta \theta } \approx Y _ { k } - Y _ { 0 } } \end{array}$ . Applying this assumption, we obtain:

$$
S ( \theta ^ { 0 } ) \cdot \Delta \theta \approx \left( \frac { \partial X _ { 0 } } { \partial \theta } \cdot \Delta \theta \right) \cdot Y _ { 0 } + X _ { 0 } ( Y _ { k } - Y _ { 0 } )\tag{16}
$$

By symmetric logic, for the probing state $\theta ^ { k }$ :

$$
S ( \theta ^ { k } ) \cdot \Delta \theta \approx \left( { \frac { \partial X _ { k } } { \partial \theta ^ { k } } } \cdot \Delta \theta \right) \cdot Y _ { k } + X _ { k } ( Y _ { k } - Y _ { 0 } )\tag{17}
$$

The equivalence $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \approx \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k } } ( a )$ can only be established if we impose an extremely strong assumption regarding the activation dynamics:

$$
{ \frac { \partial ( A ( x _ { c } ) - A ( x _ { r } ) ) } { \partial \theta } } \cdot \Delta \theta \approx X _ { k } - X _ { 0 }\tag{18}
$$

Ifand only if this assumption holds, we can substitute it into Equation 16 and Equation 17:

$$
{ \cal S } ( \theta ^ { 0 } ) \cdot \Delta \theta \approx ( X _ { k } - X _ { 0 } ) Y _ { 0 } + X _ { 0 } ( Y _ { k } - Y _ { 0 } ) = X _ { k } Y _ { 0 } + X _ { 0 } Y _ { k } - 2 X _ { 0 } Y _ { 0 }\tag{19}
$$

$$
S ( \theta ^ { k } ) \cdot \Delta \theta \approx ( X _ { k } - X _ { 0 } ) Y _ { k } + X _ { k } ( Y _ { k } - Y _ { 0 } ) = 2 X _ { k } Y _ { k } - X _ { 0 } Y _ { k } - X _ { k } Y _ { 0 }\tag{20}
$$

Summing these two refined equations together provides:

$$
{ \frac { 1 } { 2 } } \left[ S ( \theta ^ { 0 } ) + S ( \theta ^ { k } ) \right] \cdot { \Delta \theta } \approx { \frac { 1 } { 2 } } [ 2 X _ { k } Y _ { k } - 2 X _ { 0 } Y _ { 0 } ] = X _ { k } Y _ { k } - X _ { 0 } Y _ { 0 }\tag{21}
$$

Finally, substituting this result back into our main formulation yields:

$$
\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a ) \approx X _ { 0 } Y _ { 0 } + X _ { k } Y _ { k } - X _ { 0 } Y _ { 0 } = X _ { k } Y _ { k } = \Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { k } } ( a )\tag{22}
$$

This derivation reveals that directly using the probing model’s attribution relies on the assumption that the activation differences change perfectly linearly with the parameter update across the macroscopic step from $\theta ^ { 0 }$ to $\theta ^ { k }$ . Because deep neural networks are highly non-linear, this stringent constraint is almost never satisfied during discrete gradient updates, rendering the naive substitution $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a )$ ≈ $\Delta E _ { \mathcal { D } _ { \mathcal { T } } } ^ { \boldsymbol { \theta } ^ { k } } ( \boldsymbol { a } )$ theoretically unsound.

## D Details of Datasets

To facilitate the implementation of mechanistic interpretability, we strictly constrain the label of each dataset to a single token. Consequently, for specific datasets, we reformulate the inputs into question-answering prompt templates. The Arithmetic dataset is systematically partitioned into sub-datasets based on digit and step complexity, encompassing solely the four fundamental operations (addition, subtraction, multiplication, and division). Here, digit denotes the number of digits in the participating operands, while step indicates the number of mathematical operators within a single sample (excluding the equals sign). The statistical distribution and prompt examples for all datasets are detailed in Table 3.

## E Experiment of Multi-Task Fine Tuning

To verify whether our localization method can mitigate mechanistic inter-task conflicts through forward-looking interpretability insights, we design a multi-task joint fine-tuning setup where such conflicts are both controllable and observable. Sourcing candidate tasks from Appendix D, we randomly construct joint task sets ranging from 2 to 6 tasks, ensuring at least three distinct random combinations for each set size. We evaluate the mean and variance of the Target Task Accuracy (TTA) across these combinations during joint fine-tuning. As illustrated in Figure 5, our method not only achieves the highest average performance in multi-task scenarios but also demonstrates superior stability, evidenced by nearly the lowest variance. This indicates the absence of severe inter-task conflicts that would otherwise cause catastrophic degradation on individual tasks.

<table><tr><td>Dataset</td><td>Train</td><td>Test</td><td>Max Len</td><td>Output</td><td>Input Prompt Case</td><td>Label</td></tr><tr><td>IOI</td><td>6,000</td><td>600</td><td>50</td><td>Correct name</td><td>Then, Aaron and Amber were working at the house. Aaron decided to give a drink to</td><td>Amber</td></tr><tr><td>Gender</td><td>3,000 10,000</td><td>3,000</td><td>50</td><td>he / she</td><td>So Kelly is always up for an adventure, isn&#x27;t Evaluate the following boolean expression as either &#x27;True&#x27; or</td><td>she</td></tr><tr><td>BOOL</td><td></td><td></td><td>100</td><td>True / False</td><td>&#x27;False&#x27;. ( not ( True or True )) and ( False or (not True))</td><td>False</td></tr><tr><td>SST-2</td><td>67,349</td><td>872</td><td>200</td><td>positive / negative</td><td>Is the sentiment of following sentence positive or negative? that &#x27;s far too tragic to merit such superficial treatment. Answer: It is</td><td>negative</td></tr><tr><td>MRPC</td><td>3,670</td><td>1,730</td><td>200</td><td>Yes / No</td><td>Are the meanings of the following two sentences equivalent? 1. And if both apply, they are essentially possible. 2. And if both apply, they are essentially impossible.</td><td>No</td></tr><tr><td>QQP</td><td>364,000</td><td>391,000</td><td>200</td><td>Yes / No</td><td>Is Question “How do you control your horniness?&quot; a duplicate of Question &quot;How do I control my horny emotions?&quot;?</td><td>Yes</td></tr><tr><td>MNLI</td><td></td><td></td><td>250</td><td>Option Letter</td><td>What is the relationship of premise &quot;How do you know? All this is their information again.&quot; to hypothesis &quot;This information belongs to them.&quot;: A. entailment, B. contradiction, or C. unrelated? The answer is</td><td></td></tr><tr><td>RTE</td><td>2,490</td><td>3,000</td><td>200</td><td>Yes / No</td><td>Does sentence “Oil prices fall back as Yukos oil threat lifted&quot; entail sentence “Oil prices rise.&quot;?</td><td>Yes</td></tr><tr><td>WinoGrande</td><td>9,248</td><td>1,267</td><td>200</td><td>Yes / No</td><td>The trophy doesn&#x27;t fit into the brown suitcase because _is too large. Should the &#x27;_&#x27; be the trophy?</td><td>Yes</td></tr><tr><td>Docstring</td><td>2,400</td><td>600</td><td>50</td><td>Missing param</td><td>def process(self, data, name, value, result, line, file): &quot;&quot;&quot;control action cost :param value: research subject :param result: health issue :param</td><td>line</td></tr><tr><td>Induction</td><td>2,400</td><td>600</td><td></td><td>Last token</td><td>After Gray applied for health insurance, Alice applied</td><td>for</td></tr><tr><td></td><td>2-digit 10,000</td><td>600</td><td>160</td><td>Option Letter</td><td>Please choose the correct option from the following: &quot;What is 80 plus 29?&quot; Options: A. #####, B. dnfidlg, C. 378, D. 54942, E. 388, F. 109, G. 4. The answer is</td><td>F</td></tr><tr><td>Arithmetic</td><td>3-digit 10,000</td><td>600</td><td>160</td><td>Option Letter</td><td>Please choose the correct option from the following: &quot;What is 161 plus 390?&quot; Options: A. 16957, B. 551, C. people, D. 908, E. 305, F. 5, G. gkledns. The answer is</td><td>B</td></tr><tr><td>7-digit</td><td>10,000</td><td>600</td><td>160</td><td>Option Letter</td><td>Please choose the correct option from the following: &quot;What is 5567345 plus 4592838?&quot; Options: A. 10160183, B. 552493, C. apple, D. 374329826, E. 5567838, F. dhjklng, G. 387439. The</td><td>A</td></tr><tr><td>1-step</td><td>10,000</td><td>600</td><td>160</td><td>Option</td><td>answer is Please choose the correct option from the following: &quot;34-62=&quot; Options: A. -28, B. 34, C. from, D. 345, E. 11, F. 63, G. right The</td><td>A</td></tr><tr><td>2-step</td><td>10,000</td><td>600</td><td>160</td><td>Letter Option</td><td>answer is Please choose the correct option from the following: &quot;6361- 32*537=&quot; Options: A. ill, B. 10823, C. -10823, D. 583567, E.</td><td>C</td></tr><tr><td>6-step</td><td></td><td>600</td><td></td><td>Letter Option</td><td>3562, F. -55, G. same. The answer is Please choose the correct option from the following:</td><td></td></tr></table>

Table 3: Dataset statistics, configurations, and prompt construction examples.

To further substantiate this conclusion, we trace the evolution of conflicting nodes within the model circuits across checkpoints from 0 to 400 fine-tuning iterations. By utilizing the boolean solver introduced in the CLUE method to quantify these conflicts, Figure 6 reveals that our approach significantly minimizes the emergence of conflicting nodes. Its efficacy is surpassed only by unconstrained full-parameter fine-tuning, which inherently represents a natural mechanistic evolution with maximum degrees of freedom. Conversely, baseline methods perform comparably to random localization, indicating a lack of localization precision. Consequently, they fail to reduce conflicting nodes, demonstrating an inability to effectively leverage mechanistic interpretability for practical fine-tuning guidance.

![](images/c88b18180794932067c3220a7fa9a49096bcb78b8382505b882fd26bdab53ce5.jpg)  
Figure 5: Performance in multi-task accuracy. The solid line represents the mean value, and the dashed line represents the lower bound of the value variation. The shaded areas represent the variances of Our\_C and Our\_N, respectively.

![](images/cc0eb0add901575a1ed724e19bf47cc98f0aec45bbf1d85ae3294645d853e827.jpg)  
Figure 6: Evolution of conflict component in fine-tuning.

## F Exploration and Experiments about K

## F.1 The Optimal Sampling Interval of K

![](images/990d7866ec1ddbd355d395841b342a871aeac70ff95403cef4d4c73cfd222c42.jpg)  
Figure 7: Overlap rate of the top 50 components with the full-parameter circuit $( \mathcal { C } _ { \mathrm { F u l l - P a r a m } } )$ across tasks with varying initial accuracies.

As discussed in Section 4, our estimation of $\hat { \theta } ^ { I }$ involves averaging the outcomes of ten randomly sampled values for K. Empirically, we observe that each SFT task exhibits a specific optimal sampling interval; sampling K within this range yields a significantly higher expected performance compared to sampling outside of it.

To determine this optimal sampling interval for each task, we utilize the fully fine-tuned model (without any localization or parameter freezing) as a surrogate for the ideal model. This choice is motivated by two factors: $( 1 )$ The fully fine-tuned model is free from auxiliary parameter constraints, thus most closely approximating the natural trajectory of mechanistic evolution. (2) It guarantees a satisfactory accuracy on the target SFT task, ensuring the formation of a complete and well-defined task mechanism within the model. For brevity, we substitute the ideal model parameters $( \theta ^ { I } )$ with those of the fully fine-tuned model throughout this paper. Subsequently, we uniformly sample K from 0 to 100 at intervals of 5. We then compute the overlap of the top 50 components—ranked by their causal effect—between our estimated $\widehat { \theta } ^ { I }$ and the surrogate $\theta ^ { I }$ . The results are illustrated in Figure 7.

The values in parentheses within the figure legend indicate the initial Target Task Accuracy (TTA) prior to fine-tuning. A clear correlation emerges between this initial TTA and the optimal sampling interval for K. For tasks with an initial TTA near zero (e.g., BOOL and Arithmetic), the expected overlap remains consistently high across a broad range of $K \in [ 1 0 , 8 0 ]$ . Conversely, for tasks with a higher initial TTA (e.g., Docstring, IOI, and Gender), the optimal K values are highly concentrated within a narrow band of [0, 10]. We attribute this phenomenon to the initial TTA reflecting the completeness of the pre-existing task mechanism within the base model. A lower initial TTA implies a deficient mechanism, necessitating more substantial parameter updates during fine-tuning; hence, a larger K is required to adequately extrapolate and explore the distribution of $\hat { \theta } ^ { I }$ . Conversely, a higher initial TTA indicates that the model mechanism is already largely intact, requiring only minor parameter updates, allowing a smaller K to sufficiently approximate $\hat { \theta } ^ { I }$

## F.2 Automated and Reliable Determination of K via Linear Probing

<table><tr><td rowspan="2">Method / Strategy</td><td colspan="3">Bool</td><td colspan="3">SST-2</td><td colspan="3">Arithmetic</td></tr><tr><td>K</td><td>Top@50</td><td>TTA</td><td>K</td><td>Top@50</td><td>TTA</td><td>K</td><td>Top@50</td><td>TTA</td></tr><tr><td>1-Sample Uniform 10-Sample Average</td><td>35.6 [10, 80]</td><td>28 30.4</td><td>99.64 100.00</td><td>2.1 [1, 10]</td><td>26 29.6</td><td>96.34 97.55</td><td>71.5</td><td>30 31.4</td><td>99.51 100.00</td></tr><tr><td>Probe (First Layer)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>[10, 80]</td><td></td><td></td></tr><tr><td></td><td>15.3 67.2</td><td>24</td><td>98.82</td><td>8.73</td><td>28</td><td>95.17</td><td>24.8</td><td>26</td><td>99.17</td></tr><tr><td>Probe (Middle Layer) Probe (Last Layer)</td><td>58.5</td><td>32 34</td><td>99.67 99.83</td><td>1.2 1.2</td><td>34 30</td><td>96.72 97.26</td><td>72.4 77.2</td><td>30 32</td><td>99.88 100.00</td></tr></table>

Table 4: Ablation study on determining the extrapolation scalar K via linear probing across different residual stream depths on the Mistral-7B backbone, evaluated against standard sampling baselines.

In our primary locating-then-tuning pipeline, the step scalar K is estimated by averaging across 10 randomly sampled values within a plausible interval. To eliminate the computational overhead of repeated samplings and mitigate heuristic bias in scenarios where a surrogate fully fine-tuned model is unavailable, we investigate an automated probing-based framework to reliably predict K prior to full SFT. Specifically, we construct a lightweight, two-layer linear probing head and attach it to the residual stream of the Mistral-7B backbone at three representative depths: the First Layer (Layer 1), the Middle Layer (Layer 16), and the Last Layer (Layer 32). The probing head is trained using the Top@50 circuit overlap against the surrogate ideal model as the supervised optimization target.

Table 4 compares the proposed probing strategy against standard heuristic baselines (1-Sample and 10-Sample sampling) across the Bool, SST-2, and Arithmetic benchmarks in terms of predicted K values, Top@50 circuit overlap, and downstream Target Task Accuracy (TTA).

Our empirical observations yield two key insights: 1. The probing model accurately predicts the task-specific K value prior to tuning. It matches or even surpasses the circuit overlap and TTA achieved by the 10-sample averaging method, successfully circumventing the tedious time and computational expenditure induced by multi-round sampling. 2. Deploying the probe at the middle or deep residual stream yields significantly superior predictions compared to the first layer. This discrepancy stems from the representational hierarchy in LLMs: activations in the shallowest residual stream predominantly capture token-level lexical and low-level syntactic information, lacking the high-level semantic abstractions necessary to discern whether the base model already possesses relevant downstream task mechanisms.

## F.3 Robustness of Sampling K

![](images/82a6481745654e8dd8d518441ac54b96439731e1a5794e9d721b5b84d87f845b.jpg)  
(a) Rank results of 10 samplings for sample nodes.

![](images/34afe7090eb2376ffda5d319498d4181baa9e905fda7a3b9cf35bcefeaddc961.jpg)  
(b) Distribution of attribution value estimates.  
Figure 8: Robustness analysis of K-value sampling.

To analyze the stability of our K-value sampling, we select a subset of components and track their rank variations across ten independent samplings, as depicted in Figure 8a. The average rankings of these sampled components span a wide spectrum, ranging from the top 10 to beyond 2000. Empirical observations indicate that top-ranked components exhibit minimal rank variance across different samplings.

Recall that the estimated causal effect $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \boldsymbol { \theta } ^ { I } } ( a )$ comprises a baseline term $\Delta E _ { \mathcal { D } _ { \mathcal { T } } } ^ { \theta ^ { 0 } } ( a )$ and a Kdependent extrapolation term (i.e., $\begin{array} { r } { \frac 1 2 \left[ S ( { \theta } ^ { 0 } ) ^ { ' } + S ( { \boldsymbol { K } } \cdot \Delta { \theta } ) \right] \cdot { \boldsymbol { K } } \cdot \Delta { \theta } ) } \end{array}$ . By examining the exact magnitudes of these two terms, we uncover an intriguing phenomenon: even for the highest-ranked components, the baseline term $\Delta E _ { \mathcal { D } \tau } ^ { \theta ^ { 0 } } ( a )$ remains consistently at least one order of magnitude smaller than the K-dependent term. This implies that the observed rank stability is not dominated by the intrinsic causal effect of the current model (θ). Instead, the robust gradient direction dictates substantial causal effects across varying K values, precisely isolating the critical nodes that govern the target task.

Furthermore, Figure 8b illustrates the magnitude distribution of the top 1000 components, revealing an extreme concentration where the top 10 components account for nearly 90% of the total causal effect. Consequently, any severe rank fluctuations among lower-ranked components during sampling are mathematically negligible and do not compromise the overall efficacy of the localization.

## G Robustness and Sensitivity Ablation across Stochastic Factors

![](images/5bcf1ba2cb12d6830980e041e58c1d6635270ba9d17cd9cdadf08fe0497ea3bc.jpg)  
Figure 9: $\rho$ and CV across different stochastic factors. Each set is randomly repeated 10 times on Ours\_C in the Mistral-7B model.

To rigorously evaluate the algorithmic stability and reproducibility of our framework, we systematically ablate and quantify the impact of key stochastic factors introduced across different operational stages. Specifically, we examine the following five perturbation setups:

1. SFT Seeds: Random seeds governing the fine-tuning stage, which control the stochastic initialization of trainable LoRA projection matrices and parameter masking.

2. Probing SFT Samples: Stochasticity arising from uniformly sub-sampling different 1% data partitions during the probing SFT phase.

3. Probing SFT Seeds: Random seeds used during probing SFT optimization, which alter the intermediate probing parameter state $\theta ^ { \prime }$

4. Different K Sampling: Stochasticity induced by independently drawing distinct subsets of the extrapolation step scalar K when estimating $\Delta E _ { \mathcal { D } _ { \tau } } ^ { \theta ^ { I } } ( a )$

5. All (Joint Perturbation): The unconstrained scenario where all four stochastic factors above are simultaneously randomized across runs.

We assess robustness from two complementary perspectives: the intrinsic stability of the identified task circuits and the extrinsic invariance of downstream task performance:

• Critical Component Stability (ρ): We extract the Top@50 component rankings across independent random runs and compute their pairwise Spearman’s Rank Correlation Coefficient $( \rho \in [ 0 , 1 ] )$ ). A correlation value closer to 1 indicates that the identified computational subgraphs remain highly invariant under random perturbations.

• Downstream Performance Invariance (CV): We measure the dispersion of the final Target Task Accuracy (TTA) using the Coefficient of Variation $\begin{array} { r } { ( C V ) { : } C \dot { V } = \frac { \sigma } { \mu } \times 1 0 0 \% } \end{array}$ , where µ and σ denote the empirical mean and standard deviation of TTA, respectively. A lower CV signifies greater insensitivity of the downstream adaptation to stochastic noise.

Empirical Analysis and Insights. As illustrated in Figure 9, the proposed framework exhibits exceptional resilience across all evaluated dimensions. The downstream performance maintains nearzero variance across all isolated and joint perturbation settings $( C V \le 0 . 0 0 1 8 )$ , while the structural ranking of the Top@50 critical components exhibits consistently high correlation $( \rho \ge 0 . 8 3$ , reaching 1.00 under varying SFT seeds). These findings confirm that the primary stochastic factors in our pipeline introduce virtually negligible performance instability. From a mechanistic standpoint, this suggests that under our current localization granularity, downstream task adaptation is predominantly governed by a sparse set of dominant, high-attribution neurons and components. Probing with a small parameter displacement (∆θ) reliably captures the principal gradient direction of the loss landscape, enabling the extrapolation term to consistently isolate these core task-governing circuits without being derailed by local stochastic sampling fluctuations.

## H Analysis from Mechanistic Interpretability

<table><tr><td rowspan="3">Method</td><td colspan="2">Induction</td><td colspan="2">Gender</td><td colspan="2">10I</td><td colspan="2">Docstring</td></tr><tr><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td></tr><tr><td>CFull-Paramm</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100</td><td>0</td></tr><tr><td>CStatic</td><td>22</td><td>1.37</td><td>18</td><td>1.57</td><td>20</td><td>1.72</td><td>24</td><td>1.31</td></tr><tr><td>CProbing</td><td>18</td><td>1.29</td><td>22</td><td>2.23</td><td>22</td><td>1.25</td><td>22</td><td>1.04</td></tr><tr><td> $\mathcal { C } _ { \mathrm { O u r s } }$ </td><td>28</td><td>1.03</td><td>36</td><td>1.31</td><td>32</td><td>0.93</td><td>28</td><td>1.01</td></tr></table>

Table 5: Comparison of circuit overlap (Top@50) and KL divergence against the full-parameter finetuned model across interpretability datasets in component level.

To provide a rigorous theoretical validation of the differences between our approach and the static/probing localization methods, we employ Edge Attribution Patching (EAP) to construct mechanistic circuits (C) based on the estimated causal effects. Briefly, these circuits are formulated as directed acyclic graphs (DAGs), where nodes denote computational units (neurons or components) and edges represent the activation flows between them. We evaluate the circuits derived from various localization methods using two metrics: the overlap of the top-50 causal effect nodes with the ideal model, and the Kullback-Leibler (KL) divergence between the output logits of the ideal model’s circuit and those generated when strictly activating only the nodes and edges within the identified circuit. Consistent with our previous setups, we adopt the fully fine-tuned Mistral-7B model as the ideal surrogate and conduct analyses across four interpretability tasks. Given that the fully fine-tuned model achieves 100% accuracy on these specific tasks, its derived circuit serves as a robust ground truth.

<table><tr><td rowspan="2">Method</td><td colspan="2">Induction</td><td colspan="2">Gender</td><td colspan="2">I0I</td><td colspan="2">Docstring</td></tr><tr><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td><td>Top@50</td><td>KL</td></tr><tr><td>CFull-Param</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100</td><td>0</td><td>100</td><td>0</td></tr><tr><td> $\mathcal { C } _ { \mathrm { S t a t i c } }$ </td><td>20</td><td>1.55</td><td>24</td><td>1.61</td><td>26</td><td>1.49</td><td>20</td><td>1.52</td></tr><tr><td> $\mathcal { C } _ { \mathrm { P r o b i n g } }$ </td><td>20</td><td>1.51</td><td>18</td><td>1.72</td><td>20</td><td>1.62</td><td>18</td><td>1.27</td></tr><tr><td> $\mathcal { C } _ { \mathrm { O u r s } }$ </td><td>32</td><td>1.11</td><td>34</td><td>1.13</td><td>34</td><td>1.21</td><td>30</td><td>1.15</td></tr></table>

Table 6: Comparison of circuit overlap (Top@50) and KL divergence against the full-parameter finetuned model across interpretability datasets in neuron level.

Tables 5 and 6 report the results at the component and neuron levels, respectively. Notably, our method yields circuits most closely aligned with the ideal model, demonstrating its superior capability in accurately predicting the post-fine-tuning mechanistic distribution. Furthermore, the negligible performance gap between the probing and static methods implies that the mechanistic distribution of θ<sup>′</sup> remains heavily anchored to the initial state θ, thereby lacking the necessary foresight to anticipate mechanistic shifts during the fine-tuning process.

![](images/67c298f9d7c7f6f24ccc456cb2d4785ecafac9cb7cae62657253c9760e3cfeed.jpg)  
Figure 10: Ablation on Second-Order Approximation across Expanding Edge Capacities

Furthermore, to rigorously evaluate the topological fidelity of our predicted circuit graphs and assess the empirical impact of the mixed second-order derivative approximation (Equation 8), we systematically analyze circuit discrepancies against the ideal surrogate model across expanding edge capacities. We measure the Relative Hamming Distance (defined as the normalized Hamming distance over the total number of evaluated edges), where a value approaching 100% indicates near-complete divergence between two graph topologies.

As illustrated in Figure 10, while approximating the second-order partial derivative inevitably introduces mild algebraic perturbations, it consistently maintains superior structural fidelity compared to the pure second-order computation across all scales.

Crucially, when synthesized with the attribution magnitude distribution shown in Figure 8b, the empirical efficacy of our framework is overwhelmingly governed by the accurate and robust ranking of the most prominent components (approximately within the Top@50). As established in our attribution analysis, these top-tier computational units account for the vast majority (nearly 90%) of the total causal effect governing task adaptation.

![](images/615f059d4b2f9f54d5967c004944486440dbd7d900586a0d37163bc73828a54c.jpg)  
(c) C<sub>Ours</sub>(<sup>ˆ</sup>θ<sup>I</sup> )  
Figure 11: Circuit graphs corresponding to the three types of parameters.

## I Circuit Graph Visualizations

We provide visualizations of the circuit graphs generated by different localization methods for the IOI task using the Mistral-7B model. As illustrated in Figure 11, the circuit predicted by our method $( { \hat { \theta } } ^ { I } )$ incorporates notably more intermediate nodes compared to those derived from θ and $\theta ^ { \prime } ,$ leading to more intricate computational pathways from input to output. This structural complexity indicates that our estimated circuit anticipates richer task-specific mechanisms. Notably, these emergent mechanisms are fundamentally absent in the current model parameters, as the base model has not yet fully acquired the target capability.

## J Dynamic of Learning Factors for Probing SFT

![](images/8706acc9dd0bbbba381f5071aa393ed4e547fe1ee15c5c3a574329f4de9ddfc4.jpg)  
Figure 12: Top 50 overlap with Full-Parameter fine-tuning model across different probing SFT setups.

To empirically justify our hyperparameter selection for the probing SFT, we conduct an additional ablation study on the learning rate and training epochs. We utilize the fully fine-tuned Mistral-7B model as the surrogate ideal model and evaluate the component-level estimation pipeline on the IOI task, fixing the training data size to 1% of the dataset. As illustrated in Figure 12, we observe a phenomenon that mirrors the findings in Figure 4: reducing the learning rate and the number of epochs—which effectively restricts the magnitude of the parameter update—counterintuitively yields a higher circuit overlap with the ideal model. This observation further corroborates our theoretical assertion in Section 4.1: maintaining a sufficiently small parameter update during probing is imperative to preserve the necessary degrees of freedom for the extrapolation term $K \cdot \bar { \Delta \theta } .$ . Based on these empirical validations, we finalize our probing SFT configuration as utilizing 1% of the training samples, trained for a single epoch at 1% of the original learning rate.