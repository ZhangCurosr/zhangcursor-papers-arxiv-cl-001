# On-Policy Distillation for Vision-Language Model Adaptation, an Efective Paradigm on Low-Quality Multimodal Data

Hongyuan Zhang<sup>a,b,†</sup>, Xianda Guo<sup>c,†,‡</sup>, Yanlun Peng<sup>b,†</sup>, Qianlong Yang<sup>b,d</sup>, Yubin Guo<sup>e</sup>, Pinhan Fu<sup>b,c</sup>, Mulin Chen<sup>f</sup>, Xiaozhen Qiao<sup>g,∗</sup> and Ping Luo<sup>a,∗</sup>

<sup>a</sup>The University ofHong Kong, Hong Kong SAR, China

<sup>b</sup>Great Wall Motor, China

<sup>c</sup>School of Computer Science, Wuhan University, China

<sup>d</sup>School ofScience, China University ofPetroleum (East China), China

<sup>e</sup>School of Computer Science and Technology, University of Science and Technology of China, China

<sup>f</sup>School of Artificial Intelligence, Optics and Electronics (iOPEN), Northwestern Polytechnical University, China

<sup>g</sup>School of Information Science and Technology, University of Science and Technology of China, China

## A R T I C L E I N F O

Keywords: Vision-language models Knowledge distillation On-policy distillation

## A BS T RA C T

Knowledge distillation ofers an eficient route to transfer a task-adapted vision-language teacher to a compact student. The training target in current vision-language distillation methods is typically constructed from the teacher prediction and applied uniformly to all training samples, making it unreliable under class and domain shifts. In this paper, we argue that distillation target construction should be treated as a dynamic training decision rather than a fixed recipe. To this end, we propose ONPOKD, an on-policy distillation framework for vision-language model adaptation. To the best of our knowledge, ONPOKD is the first framework that applies on-policy distillation to vision-language model adaptation by learning target construction as a policy decision. ONPOKD learns a lightweight controller that constructs sample-wise adaptive targets using reliability and disagreement cues from the teacher model, student model, and zero-shot prior. Instead of relying on a fixed teacher prediction, the controller dynamically balances teacher supervision, zero-shot prior guidance, and hard-label anchoring through bounded policy actions, allowing the distillation target to adapt to varying sample reliability and training stages. The policy controller is updated with validation feedback, encouraging target construction to optimize transferability rather than merely fitting the training distribution. Since the controller is only used during training, ONPOKD can be seamlessly integrated into existing visionlanguage distillation pipelines while preserving the original inference architecture and test-time cost. Extensive experiments on Base-to-novel generalization and Cross-dataset transfer benchmarks show that ONPOKD consistently improves over strong vision-language distillation baselines.

## 1. Introduction

Large-scale vision-language models have become a strong foundation for open-vocabulary visual recognition. By aligning images with natural-language supervision, models such as CLIP can recognize categories from textual descriptions and transfer to new recognition tasks without training a taskspecific classifier (Radford et al., 2021; Jia et al., 2021; Zhai et al., 2022). This property is especially valuable when labeled data are limited or when test classes and domains difer from the original training distribution. In practice, however, downstream applications (Qiao et al., 2026b; Zhu et al., 2025; Wang et al., 2026; Xu et al., 2026; Zhou et al., 2026) still require adaptation, since target datasets often contain fine-grained categories, domain-specific visual statistics, or label spaces that are not well covered by webscale pre-training data (Recht et al., 2019; Hendrycks et al.,

2021). The central challenge is therefore to improve targettask discrimination while preserving the transferable zeroshot prior that supports novel-class and shifted-domain recognition.

![](images/a3023e618f857e2de7576fe34a7efdd5eb5e065df6cd6a3209fe20f280d39afe.jpg)  
(a) Base-to-Novel Generalization

![](images/d9fcf3b45b7193372b59731233bdd2c6e1a63747f10cd28335c4050b0fa417a3.jpg)  
Figure 1: Comparison of OnPoKD with existing VLM adaptation methods. (a) Base-to-novel generalization harmonic mean (HM). (b) Cross-dataset transfer accuracy.

Parameter-eficient adaptation has become a common way to specialize vision-language models while keeping most of the pretrained backbone unchanged. Prompt-based methods replace hand-crafted text templates with learnable contexts (Zhou et al., 2022b; Zhang et al., 2025, 2026; Huang et al., 2025; Qiao et al., 2026c), and later extend them with instance-conditioned or multi-modal prompt learning (Zhou et al., 2022a; Khattak et al., 2023a,b; Zhu et al., 2026). Other methods introduce lightweight adapters or cachebased modules on top of frozen vision-language representations (Yu et al., 2023; Lee et al., 2023; Qiao et al., 2026a, 2025). These approaches improve discrimination on adapted base classes, but they also reveal a trade-of in vision-language adaptation. The signals that sharpen sourceclass decisions do not always preserve the open-vocabulary knowledge needed for novel classes or shifted domains. As adaptation becomes stronger, the model may rely heavily on source-specific visual and semantic patterns, whereas the frozen zero-shot prior can remain a more reliable guide when the test distribution moves away from the adapted training split.

![](images/aabad02825775ff94b20361eb85ef068b2910dead76db2758fdff32027e26758.jpg)  
Figure 2: Comparison of diferent distillation paradigms for vision-language adaptation. (a) Vanilla VLM distillation matches the student to a fixed teacher target. (b) PromptKD distills from a stronger adapted teacher. (c) OnPoKD uses an on-policy controller to build a sample-wise adaptive target.

Knowledge distillation ofers a natural way to transfer adapted recognition ability into a compact student (Hinton et al., 2015). A strong adapted teacher encodes task-specific improvements, and the student inherits them while keeping a simple inference path. Existing vision-language distillation methods build the training target directly from the teacher prediction and optimize the student to match that soft distribution (Wu et al., 2023; Li et al., 2024; Yang et al., 2024). This design is sound only when the teacher is reliable for every sample, an assumption that rarely holds in visionlanguage adaptation. The adapted teacher may be accurate on base classes yet unreliable for novel classes, shifted domains, or ambiguous samples, whereas the frozen zeroshot prior, though weaker on the source split, often preserves open-vocabulary information that should not be overwritten. The student itself also changes during optimization, so the target that is best early in training may no longer be best once the student stabilizes. A fixed teacher-centered target cannot express any of these sample-wise and stage-wise diferences.

These observations suggest that distillation target construction should be adaptive rather than fixed. The teacher, zero-shot prior, and hard label provide complementary supervision for diferent training conditions. The teacher transfers task-adapted knowledge, the zero-shot prior preserves the generalization ability of the foundation model, and the hard label anchors learning when soft predictions are uncertain or conflicting. Their relative importance should depend on sample reliability, model disagreement, and training progress. For example, teacher-dominant supervision can be useful at the beginning of optimization, whereas stronger prior or label intervention may become beneficial when the teacher is uncertain, conflicts with the prior, or the student has become more stable. These factors motivate an online mechanism that observes the current training state and constructs a sample-wise distillation target.

We instantiate this mechanism as ONPOKD, an onpolicy distillation framework for vision-language model adaptation (Figure 2). Rather than fixing the mixture between teacher, prior, and label supervision by hand, ONPOKD learns a lightweight policy controller during training. The controller summarizes reliability and disagreement cues from the adapted teacher, the student, and the frozen zero-shot prior, and emits bounded actions for target mixing, sample weighting, and temperature adjustment. Crucially, the policy controller is updated from validation feedback, which steers target construction toward improving transfer behavior instead of merely reducing the current training loss. Because the controller and prior branch are discarded after distillation, the student keeps the same inference architecture and test-time cost as the underlying vision-language adaptation pipeline. Experiments confirm that adaptive target construction improves vision-language distillation under both class shift and dataset shift. Across eleven Base-to-novel generalization benchmarks, ONPOKD raises the average harmonic mean over PromptKD from 83.73 to 84.62, with a larger gain on novel classes than on base classes. In Cross-dataset transfer from ImageNet to ten target datasets, it reaches the best average accuracy of 72.66, improving PromptKD by 1.33%. Ablation studies further show that a fixed teacher-prior-label mixture is not suficient, and that reliability cues, validation feedback, and bounded policy actions are all needed for stable target construction.

We summarize our contributions below.

• We are the first to formulate target construction for vision-language distillation as an on-policy decision, enabling sample-wise and stage-wise adaptive supervision.

• A lightweight policy controller balances task-adapted teacher, zero-shot prior, and hard-label supervision. This provides more reliable sample-wise targets under class and domain shifts.

• We introduce a validation-guided policy update that learns target-construction behavior from reliability feedback, without modifying the teacher, student, or inference architecture.

• We demonstrate on Base-to-novel generalization and Cross-dataset transfer benchmarks that ONPOKD improves both robustness and transferability.

## 2. Related Work

## 2.1. Vision-Language Models

Vision-language models learn transferable visual representations by aligning images with natural-language descriptions (Radford et al., 2021; Jia et al., 2021; Yao et al., 2021; Yuan et al., 2021). They recognize unseen categories through text prompts and provide a strong starting point for downstream recognition with limited labels, but their performance hinges on prompt design and degrades under downstream distribution shift, which makes eficient adaptation important for practical deployment. Prompt learning addresses this issue by replacing hand-crafted prompts with learnable context vectors. CoOp optimizes continuous prompts for downstream classes, while CoCoOp conditions prompts on image features to improve novel-class generalization (Zhou et al., 2022b,a). MaPLe extends prompt learning to both the visual and textual branches, and Prompt-SRC regularizes prompts to preserve the frozen foundationmodel representation (Khattak et al., 2023a,b). A complementary line adds lightweight feature adapters or cache mechanisms on top of frozen vision-language representations (Yu et al., 2023; Lee et al., 2023). These studies establish that parameter-eficient adaptation works well for recognition, yet they focus on how to adapt the representation or classifier and leave open how the distillation target should change during student training. ONPOKD is orthogonal in this respect. Given an adapted teacher and a student, it improves the training target itself.

## 2.2. Knowledge Distillation

Knowledge distillation transfers information from a stronger teacher to a smaller or more deployable student, usually by matching softened predictive distributions, and has been widely applied to classification, detection, and model compression (Hinton et al., 2015; Zhao et al., 2022). It is especially attractive in vision-language adaptation, where a strong adapted teacher may be costly to deploy but a distilled student can retain a compact inference path. PromptKD applies distillation to prompt-based vision-language models and shows that a teacher can improve student prompt adaptation (Li et al., 2024), while recent CLIP distillation studies stress the need to preserve cross-modal transfer during compression or adaptation (Wu et al., 2023; Yang et al., 2024). Standard distillation, however, assumes the teacher distribution is the right target for every sample. That assumption breaks down when the teacher is unreliable across class splits or domains, and when the frozen zero-shot prior still holds useful information that the adapted teacher has weakened. A single teacher-centered target can neither express these reliability changes nor decide when hard labels should override uncertain soft targets. ONPOKD instead treats the target itself as adaptive, keeping teacher-dominant supervision when the teacher is reliable and borrowing the frozen zero-shot prior or hard-label guidance when soft targets are not.

## 2.3. Adaptive Policy Learning for Distillation

Several learning paradigms adjust supervision during training, including sample weighting, curriculum learning, confidence filtering, and teacher-student agreement (Li et al., 2023; Yang et al., 2021; Zhao et al., 2022). Related ideas appear in large language models. RLHF updates a languagemodel policy from human preference feedback (Ouyang et al., 2022), DPO turns preference optimization into a direct classification-style objective (Rafailov et al., 2023), and language-model distillation shows that teacher outputs, rationales, or on-policy trajectories can guide smaller or weaker models (Hsieh et al., 2023; Zhao et al., 2026). Collectively, these works suggest that supervision can be policy-dependent and that feedback can decide which signals should guide learning. ONPOKD adopts this high-level view but targets a diferent problem. Instead of optimizing a generative language policy, it learns a training-time policy for vision-language distillation target construction that decides which source should dominate, how strongly each sample should be weighted, and how soft the target should be. The controller turns reliability cues from the teacher, student, and prior into bounded actions for target mixing, sample weighting, and temperature adjustment, and is updated by a supervised action-matching step from held-out validation reliability rather than by reinforcement learning or a metagradient through the student optimizer. ONPOKD is therefore a training-time target-construction mechanism, not a test-time ensemble or an architecture modification.

## 3. Method

## 3.1. Problem Formulation

Knowledge distillation adapts a compact vision-language model from a stronger teacher, but conventional distillation applies the same teacher-centered target to every training image. This static rule is restrictive in vision-language adaptation, where an adapted teacher may be reliable on base categories yet less stable under class shift, a frozen zeroshot model may preserve useful open-vocabulary priors, and the student may need diferent supervision as optimization progresses. Together these observations motivate a training mechanism that chooses the supervision source for each sample and training stage.

Let $x _ { i }$ be an input image and $y _ { i }$ be its class label. We consider a teacher-student adaptation setting. It contains an adapted teacher � , a student �, and a frozen zero-shot vision-language prior �. For sample $x _ { i }$ , the three models produce logits $z _ { i } ^ { \hat { T } } , ~ z _ { i } ^ { S }$ , and $z _ { i } ^ { P }$ . Standard logit distillation

![](images/d4c7e03f7eb7e9785fd05e77eb5c07c669e58c2fc0e207d51752864efda5e591.jpg)  
Figure 3: Pipeline of OnPoKD for on-policy distillation in vision-language adaptation. During training, a lightweight controller uses reliability cues from the adapted teacher, the student, and the frozen zero-shot prior, together with training progress, to construct adaptive distillation targets. At inference time, the controller and prior branch are removed, leaving the same student architecture and test-time cost.

minimizes

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { K D } } ( i ) = \tau ^ { 2 } \mathrm { K L } ( \mathrm { s o f t m a x } ( z _ { i } ^ { T } / \tau ) \| \mathrm { s o f t m a x } ( z _ { i } ^ { S } / \tau ) ) } \end{array}\tag{1}
$$

where � is the distillation temperature and the teacher distribution is the only soft target. This is a forward KL with the teacher as the reference distribution, which is masscovering, or mean-seeking. It drives the student to place probability mass on every class the teacher deems plausible, including classes where the adapted teacher is wrong under class or domain shift. Because a teacher-centered target has no mechanism to discount these errors, the student imitates the teacher even when the teacher conflicts with the zeroshot prior, and even when stronger label guidance would be more useful. This limitation motivates mixing the teacher prediction with the zero-shot prior and the hard label, two sources that can recover probability mass from teachererroneous classes.

ONPOKD addresses this limitation by learning samplewise target construction within an on-policy distillation framework. As shown in Figure 3, the on-policy controller summarizes reliability and disagreement cues from the adapted teacher, the student, and the frozen zero-shot prior, and predicts bounded actions for target mixing, sample weighting, and temperature adjustment (Section 3.2). These actions define a policy-guided distillation objective that balances teacher supervision, zero-shot prior guidance, and hard-label anchoring (Section 3.3). The controller is updated with held-out validation feedback to favor transfer-oriented target construction over short-term training-loss reduction (Section 3.4). After training, the controller and prior branch are discarded, leaving the student with unchanged inference architecture and test-time cost (Section 3.5).

## 3.2. On-Policy Controller

The controller captures reliability and conflict among the three supervision sources without adding any extra model evaluations. It first builds a state vector $s _ { i }$ from statistics already available in the forward pass.

$$
\begin{array} { r } { s _ { i } = [ c _ { T } , m _ { T } , h _ { T } , c _ { S } , m _ { S } , h _ { S } , \phantom { \frac { \mathrm { d r g m a x } } { \mathrm { d } _ { T S } ^ { \mathrm { K L } } } } } \\ { d _ { T S } ^ { \mathrm { K L } } , d _ { T S } ^ { \mathrm { a r g m a x } } , a _ { T S } , d _ { T P } ^ { \mathrm { a r g m a x } } , d _ { T P } ^ { \mathrm { K L } } ] . } \end{array}\tag{2}
$$

Here �, �, and ℎ denote confidence, top-two margin, and normalized entropy. For a distribution $p _ { i } ^ { \tilde { M } } = \mathrm { s o f t m a x } ( z _ { i } ^ { M } )$ from model �, we compute

$$
\begin{array} { l } { { \displaystyle c _ { M } = \operatorname* { m a x } _ { y } p _ { i } ^ { M } ( y ) , } } \\ { { \displaystyle m _ { M } = p _ { i } ^ { M } ( y _ { 1 } ) - p _ { i } ^ { M } ( y _ { 2 } ) , } } \\ { { \displaystyle h _ { M } = - \frac 1 { \log C } \sum _ { y = 1 } ^ { C } p _ { i } ^ { M } ( y ) \log p _ { i } ^ { M } ( y ) , } } \end{array}\tag{3}
$$

where $y _ { 1 }$ and $y _ { 2 }$ are the top two predicted classes and � is the number of classes. The superscripts KL and argmax denote KL divergence and top-label disagreement, respectively, so $d _ { T S } ^ { \mathrm { K L } }$ and $d _ { T S } ^ { \mathrm { a r g m a x } }$ measure teacher-student disagreement while $\bar { d } _ { T P } ^ { \mathrm { a r g m a x } }$ and $d _ { T P } ^ { \mathrm { K L } }$ measure teacher-prior conflict.

$$
\begin{array} { r l } & { \qquad d _ { A B } ^ { \mathrm { K L } } = \mathrm { K L } ( p _ { i } ^ { A } \| p _ { i } ^ { B } ) , } \\ & { \qquad d _ { A B } ^ { \mathrm { a r g m a x } } = \mathbb { I } \bigl [ \mathrm { a r g m a x } _ { y } p _ { i } ^ { A } ( y ) \neq \mathrm { a r g m a x } _ { y } p _ { i } ^ { B } ( y ) \bigr ] . } \end{array}\tag{4}
$$

The alignment term $a _ { T S }$ is the normalized similarity between teacher and student image features. Together these quantities form the controller state, summarizing teacher reliability, student uncertainty, feature transfer quality, and conflict between adapted and zero-shot knowledge.

Training progress acts as an additional policy condition. Rather than concatenating it to the state vector, we use the normalized progress $p \in [ 0 , 1 ]$ to modulate the action logits, which keeps the policy conservative at the start of training and gradually admits stronger prior or hard-label intervention as the student stabilizes.

Given $s _ { i }$ and the normalized training progress $p _ { i }$ , the controller first predicts raw action scores

$$
( \boldsymbol { r } _ { i } ^ { \lambda } , \boldsymbol { r } _ { i } ^ { w } , \boldsymbol { r } _ { i } ^ { \tau } ) = f _ { \theta } ( s _ { i } ) ,\tag{5}
$$

where $r _ { i } ^ { \lambda } \in \mathbb { R } ^ { 3 }$ controls target mixing. The scalars $r _ { i } ^ { w } , r _ { i } ^ { \tau } \in$ ℝ control sample reweighting and temperature. These raw scores are then converted into a policy action

$$
a _ { i } = \pi _ { \theta } ( s _ { i } , p _ { i } ) = \left( \lambda _ { i } , w _ { i } , \tau _ { i } \right) .\tag{6}
$$

Here $\lambda _ { i }$ is a three-way mixture over the supervision sources, $w _ { i }$ is a sample weight, and $\tau _ { i }$ is a distillation temperature. We implement $f _ { \theta }$ as a two-layer multilayer perceptron with layer normalization whose final layer is initialized to zero, so the initial behavior follows a conservative teacher-dominant prior action rather than random policy outputs.

## 3.3. Policy Guided Distillation Objective

The controller maps each sample to three-way mixture weights, which we combine with a conservative teacherdominant base action and with progress-aware adjustment terms. These adjustment terms estimate, from teacher uncertainty, teacher-prior conflict, student uncertainty, and training progress, whether the zero-shot prior or the hard-label target should receive more mass. The resulting mixture action satisfies

$$
\begin{array} { r l } & { \eta _ { i } ^ { P } = \phi _ { P } ( h _ { T } , d _ { T P } ^ { \mathrm { a r g m a x } } , d _ { T P } ^ { \mathrm { K L } } ) , } \\ & { \eta _ { i } ^ { H } = \phi _ { H } ( h _ { T } , d _ { T S } ^ { \mathrm { a r g m a x } } , h _ { S } ) , } \end{array}\tag{7}
$$

where $\eta _ { i } ^ { P }$ and $\eta _ { i } ^ { H }$ are normalized need scores for the zeroshot prior and hard-label supervision, and $\phi _ { P } , \phi _ { H }$ are monotone summaries of the reliability cues. Prior intervention thus grows when the teacher is uncertain or conflicts with the zero-shot prior, while hard-label intervention grows when the teacher-student decision gap or the student uncertainty is large.

Let � denote a teacher-dominant base mixture and let $\beta _ { P } , \beta _ { H }$ be nonnegative scaling coeficients. The policy mixture is computed as

$$
\begin{array} { r l } & { \tilde { \lambda } _ { i } = \operatorname { s o f t m a x } \left( \log b + r _ { i } ^ { \lambda } + [ 0 , \beta _ { P } p _ { i } \eta _ { i } ^ { P } , \beta _ { H } p _ { i } \eta _ { i } ^ { H } ] \right) , } \\ & { \lambda _ { i } = \Pi _ { C } ( \tilde { \lambda } _ { i } ) , \quad \lambda _ { i } = ( \lambda _ { i } ^ { T } , \lambda _ { i } ^ { P } , \lambda _ { i } ^ { H } ) , } \end{array}\tag{8}
$$

where $\lambda _ { i }$ contains the teacher, prior, and hard-label mixture weights. The feasible set

$$
\begin{array} { r l } & { C = \{ \lambda \in \mathbb { R } _ { + } ^ { 3 } \mid \lambda ^ { T } + \lambda ^ { P } + \lambda ^ { H } = 1 , } \\ & { \qquad \lambda ^ { P } \leq \lambda _ { \operatorname* { m a x } } ^ { P } , \ \lambda ^ { H } \leq \lambda _ { \operatorname* { m a x } } ^ { H } , } \\ & { \qquad \lambda ^ { P } + \lambda ^ { H } \leq \lambda _ { \operatorname* { m a x } } ^ { A } \} } \end{array}\tag{9}
$$

keeps the prior and hard-label components as auxiliary supervision sources, and $\Pi _ { C }$ denotes the corresponding capand-renormalize projection.

The scalar action components are also bounded.

$$
\begin{array} { r l } & { w _ { i } = \mathrm { c l i p } _ { [ w _ { \mathrm { m i n } } , w _ { \mathrm { m a x } } ] } \left( 1 + \gamma _ { w } ( \sigma ( r _ { i } ^ { w } ) - 1 / 2 ) \right) , } \\ & { ~ \tau _ { i } = \mathrm { c l i p } _ { [ \tau _ { \mathrm { m i n } } , \tau _ { \mathrm { m a x } } ] } \left( \tau _ { 0 } ( 1 + \gamma _ { \tau } ( \sigma ( r _ { i } ^ { \tau } ) - 1 / 2 ) ) + \delta _ { \tau } ( h _ { T } ) \right) . } \end{array}\tag{10}
$$

Here $\sigma ( \cdot )$ is the sigmoid function, $\tau _ { 0 }$ is the base distillation temperature, and $\delta _ { \tau }$ is an uncertainty-dependent residual. This parameterization lets the controller adapt sample importance and target softness, while the bounds prevent unstable target construction.

The adaptive target is

$$
q _ { i } = \lambda _ { i } ^ { T } \operatorname { s o f t m a x } ( z _ { i } ^ { T } / \tau _ { i } ) + \lambda _ { i } ^ { P } q _ { i } ^ { P } + \lambda _ { i } ^ { H } q _ { i } ^ { H } ,\tag{11}
$$

where $q _ { i } ^ { P }$ is the prior component and $q _ { i } ^ { H }$ is the one-hot label distribution, optionally with label smoothing. When all classes are adapted jointly, $q _ { i } ^ { P } = \mathrm { s o f t m a x } ( z _ { i } ^ { P } / \tau _ { i } )$ . For Base-to-novel generalization, we instead apply the prior component only to the novel-class slice so that the teacher’s probability mass on base classes is preserved. Writing $\mathcal { \partial } _ { b }$ and $\mathcal { \partial } _ { n }$ for the base and novel classes, we define

$$
q _ { i } ^ { P } ( y ) = \left\{ \sum _ { y ^ { \prime } \in \mathcal { V } _ { n } } ^ { q _ { i } ^ { T } ( y ) , } q _ { i } ^ { T } ( y ^ { \prime } ) \frac { p _ { i } ^ { P } ( y ) } { \sum _ { y ^ { \prime } \in \mathcal { V } _ { n } } p _ { i } ^ { P } ( y ^ { \prime } ) } , \quad y \in \mathcal { V } _ { n } . \right.\tag{12}
$$

where $q _ { i } ^ { T } = \mathrm { s o f t m a x } ( z _ { i } ^ { T } / \tau _ { i } )$ and $p _ { i } ^ { P } = \mathrm { s o f t m a x } ( z _ { i } ^ { P } / \tau _ { i } )$ This keeps the adapted teacher in control of base knowledge while drawing on the zero-shot prior where open-vocabulary information is most useful. The novel-class slice relies only on class names through the frozen zero-shot classifier. No novel-class training images or target-domain test samples are ever used to update the student or the policy.

The student is trained by

$$
\mathcal { L } _ { \mathrm { O N P o K D } } = \frac { 1 } { B } \sum _ { i = 1 } ^ { B } w _ { i } \tau _ { i } ^ { 2 } \mathrm { K L } ( q _ { i } \| \operatorname { s o f t m a x } ( z _ { i } ^ { S } / \tau _ { i } ) ) ,\tag{13}
$$

where � is the minibatch size, $w _ { i }$ is a normalized sample weight constrained to a bounded interval, and $\tau _ { i }$ is a bounded sample-wise temperature. To isolate the efect of online target selection, the main configuration can fix $w _ { i }$ and $\tau _ { i } .$ . An optional feature-level distillation term between teacher and student image features can also be retained. It is orthogonal to the policy and is not part of the target-selection mechanism.

## 3.4. Validation Feedback Policy Optimization

The policy is learned online from a held-out validation stream drawn from the same adaptation source as the student training data. In Base-to-novel generalization this stream contains only base-class examples from the adaptation split, and novel-class test images are never used for policy learning. In Cross-dataset transfer both student training and policy feedback come from the ImageNet source split, with the target datasets reserved for final evaluation. This protocol matters, because the policy should improve transfer behavior without receiving target-domain supervision or optimizing directly on the reported test sets.

Algorithm 1 ONPOKD Training Procedure   
Require: Teacher �, student $S _ { \psi }$ , zero-shot prior �, policy   
$f _ { \theta }$   
Require: Training stream $D _ { \mathrm { t r } } ,$ validation stream ${ \cal D } _ { \mathrm { v a l } }$   
Require: Base mixture �, feasible set $^ { c , }$ feedback interval   
$K$   
Ensure: Distilled student $S _ { \psi }$   
1: Initialize $f _ { \theta }$ with a teacher-dominant base action   
2: for training step $t = 1 , 2 , \dots$ do   
3: Draw a training minibatch $B _ { t } = \{ ( x _ { i } , y _ { i } ) \} _ { i = } ^ { B }$   
4: Compute logits and image features from $\hat { T } , \bar { S } _ { \psi }$ , and   
$P$   
5: Build state $s _ { i }$ from reliability and disagreement cues   
6: Obtain raw policy scores $( { r } _ { i } ^ { \lambda } , { r } _ { i } ^ { w } , { r } _ { i } ^ { \tau } ) = f _ { \theta } ( s _ { i } )$   
7: Estimate need scores $( \eta _ { i } ^ { P } , \dot { \eta } _ { i } ^ { H } )$ by Eq. (7)   
8: Compute $\lambda _ { i } , w _ { i } ,$ and $\tau _ { i }$ by Eqs. (8)–(10)   
9: Construct adaptive target $q _ { i }$ by Eq. (11)   
10: Update � with $\nabla _ { \psi } \mathcal { L } _ { \mathrm { O N P O K D } }$   
11: if � mod $K = 0$ then   
12: Draw a validation minibatch $\mathcal { V } _ { t } \subset D _ { \mathrm { v a l } }$   
13: Evaluate $T , S _ { \psi } ,$ and � without updating �   
14: Build $g _ { i }$ and target action $a _ { i } ^ { \star } = F _ { \mathrm { v a l } } ( g _ { i } )$   
15: Update � with $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { p o l i c y } }$   
16: end if   
17: end for

At each feedback step, we evaluate the teacher, student, and zero-shot prior on validation samples. We then construct a target action from their correctness and confidence. Let

$$
g _ { i } = \left[ e _ { T } , e _ { S } , e _ { P } , c _ { T } , c _ { S } , c _ { P } , m _ { T } \right] , \quad e _ { M } = \mathbb { I } [ \hat { y } _ { M } = y _ { i } ] ,\tag{14}
$$

collect validation feedback signals, where $e _ { M }$ is the correctness of model $M \in \{ T , S , P \}$ . The target action is generated by a deterministic feedback map

$$
a _ { i } ^ { \star } = \left( \lambda _ { i } ^ { \star } , w _ { i } ^ { \star } , \tau _ { i } ^ { \star } \right) = F _ { \mathrm { v a l } } ( g _ { i } ) .\tag{15}
$$

The feedback map raises the prior coeficient when the prior succeeds and the teacher is unreliable, and raises the hardlabel coeficient when both soft sources are unreliable or the student remains uncertain. The sample-weight target is reduced for high-risk samples, and the temperature target stays close to the base distillation temperature unless temperature learning is enabled. The constants in $F _ { \mathrm { v a l } }$ are hyperparameters selected on source validation data. The controller is therefore best understood as a validation-supervised action predictor rather than an unconstrained reinforcementlearning agent.

$\phi _ { P } , \phi _ { H }$ , and $F _ { \mathrm { v a l } }$ are all deterministic, non-trainable maps. $\phi _ { P }$ takes the teacher’s normalized entropy $h _ { T }$ , the teacher-prior top-label agreement indicator $d _ { T P } ^ { \mathrm { a r g m a x } }$ , and a clipped teacher-prior KL term $d _ { T P } ^ { \mathrm { K L } }$ , and returns a clipped monotone mixture over them. $\phi _ { H }$ takes the teacher’s normalized entropy $h _ { T } .$ , the teacher-student top-label agreement indicator $d _ { T S } ^ { \mathrm { a r g m a x } }$ , and the student’s normalized entropy $h _ { S } ,$ and returns the same kind of clipped monotone mixture. $F _ { \mathrm { v a l } }$ is a deterministic map from per-sample validation signals $g _ { i }$ to a target action $( \lambda _ { i } ^ { \star } , w _ { i } ^ { \star } , \tau _ { i } ^ { \star } )$ , computed as four clipped monotone sub-targets. (i) A teacher-misuse risk score from the teacher’s correctness and margin, (ii) a sample-weight target from validation correctness, (iii) a prior-coeficient target from the teacher’s gap relative to the prior, and (iv) a hard-label coeficient target from the joint unreliability of the teacher and prior. The mixing weights in $\phi _ { P }$ and $\phi _ { H }$ are positive hyperparameters that sum to one. The auxiliary caps, action bounds, and the rescaling rule for $F _ { \mathrm { v a l } }$ are listed in the released configuration.

The policy loss matches the predicted action to this feedback target.

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { p o l i c y } } = \alpha _ { \lambda } \| \lambda _ { i } - \lambda _ { i } ^ { \star } \| _ { 2 } ^ { 2 } + \alpha _ { w } \| w _ { i } - w _ { i } ^ { \star } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \alpha _ { \tau } \| \tau _ { i } - \tau _ { i } ^ { \star } \| _ { 2 } ^ { 2 } + \alpha _ { r } \mathcal { R } ( \pi _ { \theta } ) , } \end{array}\tag{16}
$$

where $\mathcal { R } ( \pi _ { \theta } )$ regularizes the action toward the conservative base behavior. This validation-feedback objective is decoupled from the training KD loss. It does not back-propagate through that loss or through the student optimizer.

The teacher-dominant initialization is a safety condition for early training. With the final controller layer initialized to zero, the controller’s raw contribution is initially neutral and the base simplex action stays teacher-dominant, while the prior and hard-label caps prevent an untrained controller from replacing a useful teacher target with a random auxiliary target. Validation feedback follows a periodic schedule rather than running at every step. At each feedback step the teacher, student, and frozen prior are evaluated on a held-out source-validation minibatch in evaluation mode, and only the controller is updated by the action-matching loss. The schedule reduces minibatch noise and amortizes the cost of the extra evaluation while still allowing the policy to track the changing student later in training.

The default procedure thus separates policy learning from student-gradient updates. On ordinary training batches the controller selects targets for student optimization, whereas on feedback steps validation signals update the controller. This separation prevents the policy from exploiting shortterm training-loss reductions and keeps the learned action aligned with validation reliability.

## 3.5. Training and Inference

As a training-time target-construction mechanism, ON-POKD imposes no specific teacher or student architecture. The teacher can be any adapted vision-language model that exposes class logits and image features, and the prior is obtained from the frozen zero-shot vision-language model using class-name prompts. During training, the policy controller reads reliability and disagreement cues from the teacher, student, and prior and uses them to construct the adaptive target in Eq. (11). At inference, the controller and the zero-shot prior branch are removed, and the distilled student produces the final prediction with the same adapted text classifier as the underlying teacher-student VLM adaptation pipeline. In this way, ONPOKD improves the supervision signal during distillation while adding no policy evaluation, prior fusion, or extra model capacity at test time.

Table 1  
Base-to-novel generalization on the standard eleven-dataset CLIP benchmark. Each block reports Base, Novel, and HM accuracy, and Δ shows the gain over PromptKD.  
(a) Average over 11 datasets
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>69.34</td><td>74.22</td><td>71.70</td></tr><tr><td>CoOp</td><td>82.69</td><td>63.22</td><td>71.66</td></tr><tr><td>CoCoOp</td><td>80.47</td><td>72.30</td><td>76.17</td></tr><tr><td>MaPLe</td><td>82.28</td><td>75.14</td><td>78.55</td></tr><tr><td>PromptSRC</td><td>84.26</td><td>76.10</td><td>79.97</td></tr><tr><td>PromptKD</td><td>86.96</td><td>80.73</td><td>83.73</td></tr><tr><td>OnPoKD</td><td>87.26</td><td>82.13</td><td>84.62</td></tr><tr><td>Δ</td><td>+0.30</td><td>+1.40</td><td>+0.89</td></tr></table>

(b) ImageNet
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>72.43</td><td>68.14</td><td>70.22</td></tr><tr><td>CoOp</td><td>76.47</td><td>67.88</td><td>71.92</td></tr><tr><td>CoCoOp</td><td>75.98</td><td>70.43</td><td>73.10</td></tr><tr><td>MaPLe</td><td>76.66</td><td>70.54</td><td>73.47</td></tr><tr><td>PromptSRC</td><td>77.60</td><td>70.73</td><td>74.01</td></tr><tr><td>PromptKD</td><td>80.83</td><td>74.66</td><td>77.62</td></tr><tr><td>OnPoKD</td><td>80.90</td><td>75.20</td><td>77.95</td></tr><tr><td>Δ</td><td>+0.07</td><td>+0.54</td><td>+0.33</td></tr></table>

(c) Caltech101
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>96.84</td><td>94.00</td><td>95.40</td></tr><tr><td>CoOp</td><td>98.00</td><td>89.81</td><td>93.73</td></tr><tr><td>CoCoOp</td><td>97.96</td><td>93.81</td><td>95.84</td></tr><tr><td>MaPLe</td><td>97.74</td><td>94.36</td><td>96.02</td></tr><tr><td>PromptSRC</td><td>98.10</td><td>94.03</td><td>96.02</td></tr><tr><td>PromptKD</td><td>98.91</td><td>96.65</td><td>97.77</td></tr><tr><td>OnPoKD</td><td>99.12</td><td>97.13</td><td>98.11</td></tr><tr><td>Δ</td><td>+0.21</td><td>+0.48</td><td>+0.34</td></tr></table>

(d) OxfordPets
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>91.17</td><td>97.26</td><td>94.12</td></tr><tr><td>CoOp</td><td>93.67</td><td>95.29</td><td>94.47</td></tr><tr><td>CoCoOp</td><td>95.20</td><td>97.69</td><td>96.43</td></tr><tr><td>MaPLe</td><td>95.43</td><td>97.76</td><td>96.58</td></tr><tr><td>PromptSRC</td><td>95.33</td><td>97.30</td><td>96.30</td></tr><tr><td>PromptKD</td><td>96.30</td><td>98.01</td><td>97.15</td></tr><tr><td>OnPoKD</td><td>96.34</td><td>98.31</td><td>97.32</td></tr><tr><td>∆</td><td>+0.04</td><td>+0.30</td><td>+0.17</td></tr></table>

(g) Food101

(e) StanfordCars  
(f) Flowers102
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>63.37</td><td>74.89</td><td>68.65</td></tr><tr><td>CoOp</td><td>78.12</td><td>60.40</td><td>68.13</td></tr><tr><td>CoCoOp</td><td>70.49</td><td>73.59</td><td>72.01</td></tr><tr><td>MaPLe</td><td>72.94</td><td>74.00</td><td>73.47</td></tr><tr><td>PromptSRC</td><td>78.27</td><td>74.97</td><td>76.58</td></tr><tr><td>PromptKD</td><td>82.80</td><td>83.37</td><td>83.08</td></tr><tr><td>OnPoKD</td><td>83.19</td><td>84.64</td><td>83.91</td></tr><tr><td>∆</td><td>+0.39</td><td>+1.27</td><td>+0.83</td></tr></table>

<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>72.08</td><td>77.80</td><td>74.83</td></tr><tr><td>CoOp</td><td>97.60</td><td>59.67</td><td>74.06</td></tr><tr><td>CoCoOp</td><td>94.87</td><td>71.75</td><td>81.71</td></tr><tr><td>MaPLe</td><td>95.92</td><td>72.46</td><td>82.56</td></tr><tr><td>PromptSRC</td><td>98.07</td><td>76.50</td><td>85.95</td></tr><tr><td>PromptKD</td><td>99.42</td><td>82.62</td><td>90.24</td></tr><tr><td>OnPoKD</td><td>99.33</td><td>83.28</td><td>90.60</td></tr><tr><td>Δ</td><td>-0.09</td><td>+0.66</td><td>+0.36</td></tr></table>

<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>90.10</td><td>91.22</td><td>90.66</td></tr><tr><td>CoOp</td><td>88.33</td><td>82.26</td><td>85.19</td></tr><tr><td>CoCoOp</td><td>90.70</td><td>91.29</td><td>90.99</td></tr><tr><td>MaPLe</td><td>90.71</td><td>92.05</td><td>91.38</td></tr><tr><td>PromptSRC</td><td>90.67</td><td>91.53</td><td>91.10</td></tr><tr><td>PromptKD</td><td>92.43</td><td>93.68</td><td>93.05</td></tr><tr><td>OnPoKD</td><td>92.62</td><td>93.94</td><td>93.28</td></tr><tr><td>∆</td><td>+0.19</td><td>+0.26</td><td>+0.23</td></tr></table>

(h) FGVCAircraft  
(i) SUN397
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>27.19</td><td>36.29</td><td>31.09</td></tr><tr><td>CoOp</td><td>40.44</td><td>22.30</td><td>28.75</td></tr><tr><td>CoCoOp</td><td>33.41</td><td>23.71</td><td>27.74</td></tr><tr><td>MaPLe</td><td>37.44</td><td>35.61</td><td>36.50</td></tr><tr><td>PromptSRC</td><td>42.73</td><td>37.87</td><td>40.15</td></tr><tr><td>PromptKD</td><td>49.12</td><td>41.81</td><td>45.17</td></tr><tr><td>OnPoKD</td><td>50.43</td><td>45.17</td><td>47.66</td></tr><tr><td>∆</td><td>+1.31</td><td>+3.36</td><td>+2.49</td></tr></table>

<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>69.36</td><td>75.35</td><td>72.23</td></tr><tr><td>CoOp</td><td>80.60</td><td>65.89</td><td>72.51</td></tr><tr><td>CoCoOp</td><td>79.74</td><td>76.86</td><td>78.27</td></tr><tr><td>MaPLe</td><td>80.82</td><td>78.70</td><td>79.75</td></tr><tr><td>PromptSRC</td><td>82.67</td><td>78.47</td><td>80.52</td></tr><tr><td>PromptKD</td><td>83.69</td><td>81.54</td><td>82.60</td></tr><tr><td>OnPoKD</td><td>83.92</td><td>82.33</td><td>83.12</td></tr><tr><td>∆</td><td>+0.23</td><td>+0.79</td><td>+0.52</td></tr></table>

(j) DTD  
(k) EuroSAT
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>53.24</td><td>59.90</td><td>56.37</td></tr><tr><td>CoOp</td><td>79.44</td><td>41.18</td><td>54.24</td></tr><tr><td>CoCoOp</td><td>77.01</td><td>56.00</td><td>64.85</td></tr><tr><td>MaPLe</td><td>80.36</td><td>59.18</td><td>68.16</td></tr><tr><td>PromptSRC</td><td>83.37</td><td>62.97</td><td>71.75</td></tr><tr><td>PromptKD</td><td>85.84</td><td>71.37</td><td>77.94</td></tr><tr><td>OnPoKD</td><td>86.64</td><td>73.59</td><td>79.58</td></tr><tr><td>∆</td><td>+0.80</td><td>+2.22</td><td>+1.64</td></tr></table>

(l) UCF101
<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>56.48</td><td>64.05</td><td>60.03</td></tr><tr><td>CoOp</td><td>92.19</td><td>54.74</td><td>68.69</td></tr><tr><td>CoCoOp</td><td>87.49</td><td>60.04</td><td>71.21</td></tr><tr><td>MaPLe</td><td>94.07</td><td>73.23</td><td>82.35</td></tr><tr><td>PromptSRC</td><td>92.90</td><td>73.90</td><td>82.32</td></tr><tr><td>PromptKD</td><td>97.54</td><td>82.08</td><td>89.14</td></tr><tr><td>OnPoKD</td><td>97.64</td><td>87.01</td><td>92.02</td></tr><tr><td>∆</td><td>+0.10</td><td>+4.93</td><td>+2.88</td></tr></table>

<table><tr><td>ViT-B/16</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>CLIP</td><td>70.53</td><td>77.50</td><td>73.85</td></tr><tr><td>CoOp</td><td>84.69</td><td>56.05</td><td>67.46</td></tr><tr><td>CoCoOp</td><td>82.33</td><td>73.45</td><td>77.64</td></tr><tr><td>MaPLe</td><td>83.00</td><td>78.66</td><td>80.77</td></tr><tr><td>PromptSRC</td><td>87.10</td><td>78.80</td><td>82.74</td></tr><tr><td>PromptKD</td><td>89.71</td><td>82.27</td><td>85.83</td></tr><tr><td>OnPoKD</td><td>89.78</td><td>82.84</td><td>86.17</td></tr><tr><td>Δ</td><td>+0.07</td><td>+0.57</td><td>+0.34</td></tr></table>

## 4. Experiments

## 4.1. Experimental Setup

Base-to-Novel Generalization. The Base-to-novel generalization setting tests whether adaptation preserves openvocabulary generalization (Zhou et al., 2022b,a; Khattak et al., 2023a). Each dataset is split into base and novel classes. The model is adapted only with base-class training data and then evaluated on both base and novel test classes. This setting is well aligned with our goal, as it directly exposes the trade-of between teacher-driven adaptation and preservation of the original zero-shot prior. Following common practice, we report base accuracy, novel accuracy, and their harmonic mean (HM), which summarizes the balance between adapted-class discrimination and novel-class transfer.

Cross-Dataset Evaluation. The Cross-dataset evaluation setting tests robustness under dataset shift (Recht et al., 2019; Hendrycks et al., 2021). The model is adapted on ImageNet and then transferred to other recognition datasets without using their training images, which probes whether the distilled student learns a transferable decision rule or instead overfits to ImageNet classes and visual statistics. We report top-1 accuracy on each target dataset and the average accuracy across all target datasets.

Datasets. For Base-to-novel generalization, we follow the standard eleven-dataset benchmark used in vision-language prompt learning. The benchmark includes ImageNet (Deng et al., 2009), Caltech101 (Fei-Fei et al., 2004), Oxford-Pets (Parkhi et al., 2012), StanfordCars (Krause et al., 2013), Flowers102 (Nilsback and Zisserman, 2008), Food101 (Bossar et al., 2014), FGVCAircraft (Maji et al., 2013), SUN397 (Xiao et al., 2010), DTD (Cimpoi et al., 2014), EuroSAT (Helber et al., 2019), and UCF101 (Soomro et al., 2012). These datasets span objects, fine-grained categories, textures, scenes, actions, and remote-sensing images. For Cross-dataset evaluation, ImageNet serves as the source dataset and the adapted model is evaluated on the other ten. This diversity lets us assess both class-shift and domain-shift transfer.

Training time and memory overhead of OnPoKD relative to PromptKD on the two representative datasets used throughout the ablation studies.
<table><tr><td>Dataset</td><td>HM gain</td><td>Training time</td><td>Memory</td></tr><tr><td>FGVCAircraft</td><td>+2.49</td><td>+25%</td><td>+51%</td></tr><tr><td>EuroSAT</td><td>+2.88</td><td>+19%</td><td>+51%</td></tr></table>

Implementation Details. All comparisons use the ViT-B/16 CLIP backbone (Radford et al., 2021). We compare ONPOKD against zero-shot CLIP, the prompt-learning methods CoOp, CoCoOp, MaPLe, and PromptSRC (Zhou et al., 2022b,a; Khattak et al., 2023a,b), and the distillation baseline PromptKD (Li et al., 2024). For a controlled comparison, ONPOKD leaves the teacher, the student, and the test-time inference architecture unchanged. The controller is a two-layer multilayer perceptron with layer normalization whose final layer is initialized to zero, so its initial behavior is the conservative teacher-dominant action in Algorithm 1. Hyperparameters are selected with source or base validation feedback and then kept fixed for all reported evaluations. Because the only diference from the underlying distillation pipeline is the on-policy for target construction, the improvements in the following tables reflect adaptive target construction rather than extra model capacity, targetdomain tuning, or additional inference-time modules. The controller architecture and base action are shared across the two evaluation settings, while the policy hyperparameters take two pre-specified configurations that are fixed before any target-dataset evaluation. The two settings difer because the Base-to-novel and Cross-dataset tasks expose diferent shifts, not because the controller is tuned to individual target datasets. The training time and memory overhead on the two representative datasets used in the ablation study are reported in Table 2.

## 4.2. Base-to-Novel Generalization

Table 1 reports Base-to-novel generalization results across eleven datasets. On average, ONPOKD improves the HM over PromptKD from 83.73 to 84.62, with a larger gain on novel classes (+1.40) than on base classes (+0.30). This pattern matches the motivation of ONPOKD. The policy keeps teacher-centered supervision for reliable baseclass knowledge while injecting zero-shot prior and hardlabel guidance when the adapted teacher is less reliable for transfer. The gains are broadly distributed rather than concentrated on one dataset. ONPOKD improves HM on all eleven benchmarks, and the improvement is especially clear on FGVCAircraft (+2.49 HM), DTD (+1.64 HM), EuroSAT (+2.88 HM), and StanfordCars (+0.83 HM), while even ImageNet rises by +0.33 HM. The strongest gains appear precisely on datasets where the Base-to-novel generalization gap is hardest to close, which suggests that sample-wise policy control better balances adapted teacher knowledge against zero-shot prior knowledge.

## 4.3. Cross-Dataset Transfer

Table 3 evaluates Cross-dataset transfer from ImageNet to ten target datasets. This benchmark is stricter than Baseto-novel generalization, since the target datasets are never seen during adaptation. ONPOKD reaches the best average accuracy of 72.66, improving PromptKD by 1.33%. The policy therefore does not merely overfit the source-domain adaptation objective. It strengthens the distilled student’s ability to transfer across datasets. ONPOKD obtains the strongest accuracy on seven of the ten target datasets, with the largest gains on EuroSAT (+5.69), DTD (+3.11), Cars (+2.11), Aircraft (+1.79), and Flowers102 (+1.54), all of which difer from ImageNet in visual domain or fine-grained label semantics, where a fixed source-trained teacher signal is brittle. The small decreases on Caltech101 (-0.57), Food101 (-0.32), and UCF101 (-1.07) occur on datasets where PromptKD is already strong, a trade-of suggesting that future policy variants may need more conservative intervention when the teacher and prior are both confident.

## 4.4. Ablation Study

We conduct ablations from three complementary perspectives (the supervision components, the policy action space, and the controller state cues) using representative datasets from both evaluation settings. For Base-to-novel generalization we use FGVCAircraft and EuroSAT, which emphasize fine-grained class transfer and domain-shifted recognition. For Cross-dataset transfer we use DTD and EuroSAT, whose target domains difer substantially from the ImageNet source. This design tests whether the policycontrolled target is useful under both class shift and dataset shift.

## 4.4.1. Contribution of Supervision Sources

We first examine whether the gain comes from the extra supervision sources or from the on-policy that controls them. This is an important distinction, since simply adding the zero-shot prior or hard labels could improve robustness even without sample-wise adaptation. Under the same teacher-student setup, Table 4 compares four groups (teacher-only distillation, a fixed teacher-prior-label mixture, source-removal variants, and policy-learning variants). T-KD reduces ONPOKD to ordinary teacher-only distillation, using the adapted teacher as the only soft target. Fixed

Table 4  
Table 3  
Cross-dataset accuracy (%) on ten target datasets. PromptKD and OnPoKD distill ImageNet-adapted teachers using unlabeled target-domain training images, while other methods transfer directly from ImageNet. Avg. denotes the mean target accuracy.
<table><tr><td>Method</td><td>Avg.</td><td>Caltech101</td><td>OxfordPets</td><td>Cars</td><td>Flowers102</td><td>Food101</td><td>Aircraft</td><td>SUN397</td><td>DTD</td><td>EuroSAT</td><td>UCF101</td></tr><tr><td>CoOp</td><td>63.88</td><td>93.70</td><td>89.14</td><td>64.51</td><td>68.71</td><td>85.30</td><td>18.47</td><td>64.15</td><td>41.92</td><td>46.39</td><td>66.55</td></tr><tr><td>CoCoOp</td><td>65.84</td><td>94.43</td><td>90.14</td><td>65.32</td><td>71.88</td><td>86.06</td><td>22.94</td><td>67.36</td><td>45.73</td><td>46.37</td><td>68.21</td></tr><tr><td>MaPLe</td><td>66.30</td><td>93.53</td><td>90.49</td><td>65.57</td><td>72.23</td><td>86.20</td><td>24.74</td><td>67.01</td><td>46.49</td><td>48.06</td><td>68.69</td></tr><tr><td>PromptSRC</td><td>65.81</td><td>93.60</td><td>90.25</td><td>65.70</td><td>70.25</td><td>86.15</td><td>23.90</td><td>67.10</td><td>46.87</td><td>45.50</td><td>68.75</td></tr><tr><td>PromptKD</td><td>71.33</td><td>93.61</td><td>91.59</td><td>73.93</td><td>75.33</td><td>88.84</td><td>26.24</td><td>68.57</td><td>55.08</td><td>63.74</td><td>76.39</td></tr><tr><td>OnPoKD</td><td>72.66</td><td>93.04</td><td>92.21</td><td>76.04</td><td>76.87</td><td>88.52</td><td>28.03</td><td>68.92</td><td>58.19</td><td>69.43</td><td>75.32</td></tr><tr><td>Δ</td><td>+1.33</td><td>-0.57</td><td>+0.62</td><td>+2.11</td><td>+1.54</td><td>-0.32</td><td>+1.79</td><td>+0.35</td><td>+3.11</td><td>+5.69</td><td>-1.07</td></tr></table>

Controlled removal study for the target-construction policy. T-KD uses only the adapted teacher, Fixed Mix removes sample-wise decisions, and the remaining variants disable one policy source or update signal.
<table><tr><td rowspan="3">Variant</td><td colspan="6">Base-to-Novel</td><td colspan="2">Cross-Dataset</td></tr><tr><td colspan="3">FGVCAircraft</td><td colspan="3">EuroSAT</td><td>DTD</td><td>EuroSAT</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td><td>Acc.</td><td>Acc.</td></tr><tr><td>T-KD</td><td>47.71</td><td>40.84</td><td>44.01</td><td>97.13</td><td>85.24</td><td>90.80</td><td>53.19</td><td>62.55</td></tr><tr><td>Fixed Mix</td><td>48.12</td><td>40.47</td><td>43.96</td><td>96.46</td><td>84.25</td><td>89.94</td><td>53.00</td><td>64.12</td></tr><tr><td>w/o Prior</td><td>48.90</td><td>43.71</td><td>46.16</td><td>96.98</td><td>81.80</td><td>88.75</td><td>57.32</td><td>69.97</td></tr><tr><td>w/o Label</td><td>47.45</td><td>41.03</td><td>44.01</td><td>96.58</td><td>80.36</td><td>87.73</td><td>53.12</td><td>66.96</td></tr><tr><td>w/o Val.</td><td>18.81</td><td>28.73</td><td>22.74</td><td>96.51</td><td>84.78</td><td>90.27</td><td>53.29</td><td>65.35</td></tr><tr><td>w/o Prog.</td><td>48.90</td><td>43.90</td><td>46.27</td><td>97.52</td><td>85.64</td><td>91.19</td><td>57.11</td><td>70.45</td></tr><tr><td>Full</td><td>50.43</td><td>45.17</td><td>47.66</td><td>97.64</td><td>87.01</td><td>92.02</td><td>58.19</td><td>69.43</td></tr></table>

Mix keeps the same three supervision sources as ONPOKD but combines them with a fixed teacher-prior-label mixture, thereby removing sample-wise policy adaptation. The w/o Prior and w/o Label variants drop the zero-shot prior and hard-label components to test whether the gains come from open-vocabulary prior knowledge or from explicit label anchoring. Finally, the w/o Val. and w/o Prog. variants retain the target sources but remove validation-feedback updates or progress-aware modulation, testing whether online learning is needed for reliable target construction. Table 4 shows that the full design gives the best overall balance. Relative to T-KD, ONPOKD improves FGVCAircraft HM by 3.65%, EuroSAT HM by 1.22%, DTD accuracy by 5.00%, and EuroSAT Cross-dataset accuracy by 6.88%, gains that cannot be attributed to teacher imitation alone. Fixed Mix remains consistently weaker than the adaptive variants, which confirms that the supervision sources alone are not enough. The policy must decide when to trust each one. The sourceremoval variants reveal distinct roles for the prior and the label. Removing labels causes the largest drop on Crossdataset transfer, especially on DTD and EuroSAT, indicating that hard labels anchor the controller when teacher and prior disagree. Removing the prior instead hurts FGV-CAircraft and EuroSAT Base-to-novel generalization more clearly, indicating that the zero-shot prior mainly protects open-vocabulary transfer. The w/o Val. variant collapses FGVCAircraft HM, confirming that validation feedback is needed to avoid unstable decisions on fine-grained classes. Specifically, removing validation feedback reduces FGV-CAircraft HM from 47.66 to 22.74 (a reduction of 24.92%) and EuroSAT HM from 92.02 to 90.27 (a reduction of

1.75%). The gap reflects the dataset-dependent role of heldout feedback. On fine-grained classes with narrow inter-class margins and sample-specific teacher mistakes, confidencebased reliability cues alone become insuficient, so direct correctness signals from a held-out source-validation stream are necessary for the controller to suppress an unsafe intervention. On EuroSAT, by contrast, state-based uncertainty and disagreement signals remain informative because the class cues are stronger and more coherent, so the marginal value of validation feedback is smaller but still positive. Removing progress-aware modulation has a milder efect: it slightly helps EuroSAT Cross-dataset accuracy yet still lowers FGVCAircraft HM and DTD accuracy. Overall, each source plays a distinct part. The teacher supports task adaptation, the prior supports robustness, the label anchors learning, and feedback keeps the mixture reliable.

## 4.4.2. Contribution ofEach Policy Action

We next study which actions the controller needs for efective target construction. The full policy controls three quantities (the mixture over teacher, prior, and label targets, the sample weight, and the distillation temperature). Starting from a fixed KD target, Table 5 enables these action dimensions one at a time. The “Adaptive mix only” variant lets the controller select the teacher-prior-label mixture while fixing the sample weight and temperature. The next two variants add either sample weighting or temperature adaptation. Table 5 shows that these action dimensions are neither interchangeable nor uniformly useful across tasks. On EuroSAT, whose class-level visual cues are relatively coherent, adaptive mixing already improves HM from 90.30 to 91.79, whereas adding sample weighting or temperature adaptation on top of the mixture yields HMs of 89.89 and 89.60, respectively, suggesting that extra per-sample corrections can disturb an already adequate mixture. FGVCAircraft is more fine-grained with narrower inter-class margins, so adaptive mixing alone drops HM from 46.64 to 25.78. Sample weighting and temperature adaptation recover HM to 43.95 and 41.81 by down-weighting unreliable examples and adjusting the softness of the distillation signal, respectively. The full action space performs best precisely because the three actions play complementary roles. Mixture decides what to trust, weighting decides how much to trust a sample, and temperature decides how sharply to match the target, a coupling that a fixed KD target cannot express.

Table 5  
Efect of enabling policy actions one at a time. The comparison isolates target mixing, sample weighting, and temperature control on FGVCAircraft and EuroSAT.
<table><tr><td rowspan="2">Variant</td><td colspan="3">FGVCAircraft</td><td colspan="3">EuroSAT</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td>Fixed KD target</td><td>49.28</td><td>44.27</td><td>46.64</td><td>97.24</td><td>84.28</td><td>90.30</td></tr><tr><td>Adaptive mix only</td><td>23.59</td><td>28.43</td><td>25.78</td><td>97.52</td><td>86.69</td><td>91.79</td></tr><tr><td>Adaptive mix + weight</td><td>48.14</td><td>40.43</td><td>43.95</td><td>96.40</td><td>84.21</td><td>89.89</td></tr><tr><td>Adaptive mix + temp.</td><td>43.64</td><td>40.13</td><td>41.81</td><td>97.02</td><td>83.23</td><td>89.60</td></tr><tr><td>Full action</td><td>50.43</td><td>45.17</td><td>47.66</td><td>97.64</td><td>87.01</td><td>92.02</td></tr></table>

Table 6  
Reliability cues used by the policy controller. Each row removes one cue group while keeping the full action space fixed.
<table><tr><td rowspan="2">Variant</td><td colspan="3">FGVCAircraft</td><td colspan="3">EuroSAT</td></tr><tr><td>Base</td><td>Novel</td><td>HM</td><td>Base</td><td>Novel</td><td>HM</td></tr><tr><td> $\mathsf { w } / \mathsf { o }$  uncertainty cues</td><td>36.73</td><td>39.29</td><td>37.97</td><td>97.26</td><td>81.56</td><td>88.72</td></tr><tr><td> $w / \circ { \mathsf { T } } { \mathsf { - } } { \mathsf { S } }$  disagreement</td><td>47.12</td><td>43.79</td><td>45.39</td><td>97.55</td><td>84.64</td><td>90.64</td></tr><tr><td> $\mathsf { w } / \mathsf { o } ~ { \mathsf { T } } { \mathsf { - } } { \mathsf { P } }$  conflict</td><td>48.44</td><td>44.57</td><td>46.42</td><td>96.50</td><td>86.23</td><td>91.08</td></tr><tr><td>w/o feature alignment</td><td>47.78</td><td>44.33</td><td>45.99</td><td>97.12</td><td>85.05</td><td>90.69</td></tr><tr><td>Full state</td><td>50.43</td><td>45.17</td><td>47.66</td><td>97.64</td><td>87.01</td><td>92.02</td></tr></table>

## 4.4.3. Importance ofReliability Cues

Finally, we analyze which reliability cues the controller needs. ONPOKD builds its policy state from four cue groups (uncertainty statistics, teacher-student disagreement, teacher-prior conflict, and feature alignment). Table 6 removes each group in turn while keeping the action space fixed. Each ablation isolates one role. Removing uncertainty cues tests whether confidence, margin, and entropy are needed to identify unreliable predictions, removing teacherstudent disagreement tests whether the policy must know when the student is misaligned with the adapted teacher, removing teacher-prior conflict tests the role of disagreement between adapted and zero-shot knowledge, and removing feature alignment tests whether representation-level transfer quality adds complementary information. Table 6 shows that uncertainty cues are the most important state signal. Removing them drops FGVCAircraft HM by 9.69% and EuroSAT HM by 3.30%, far more than any single relationalcue ablation. The controller must first know whether a sample is unreliable, and the remaining cue groups then explain why. Teacher-student disagreement shows whether the student has caught up with the teacher, teacher-prior

![](images/65996aa93c8d348abdad1c61894bcef4030ba0a48b17ce1b26c6c06736c233e9.jpg)  
(a) Auxiliary target cap

![](images/fb3243a39d67965e088a4bf1d7b2c004ec1fa6ac9cf6e4f3fa39c95d78d0fcd1.jpg)  
(b) Policy regularization

![](images/2aee1f34beb1dcb4d81797f3cd1f3e2b1b7d29408d8ef2f24ed370f250e59618.jpg)  
(c) Action strength  
Figure 4: EuroSAT sensitivity sweeps for the main policy controls. Bars show Base-to-novel generalization metrics, and the line shows Cross-dataset accuracy.

Figure 5: Learned state-to-action mapping on FGVC-Aircraft (weak teacher) and EuroSAT (strong teacher). The policy increases prior mass with uncertainty on EuroSAT but keeps the prior dormant on FGVC-Aircraft, with all cues producing the expected action directions.

(a)  
![](images/fb92645ca21bcf33848c1fc9af49b7124c8e317148eb5bc8c63e1b70a8a4f5c9.jpg)  
(c)

![](images/2ffb192c43d631d2ba43358b1c308fd32cac3241185c7b78c1f0538bd86c173f.jpg)

<table><tr><td rowspan=1 colspan=4>FGVC</td></tr><tr><td rowspan=1 colspan=1>T-conf</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>-0.42</td><td rowspan=1 colspan=1>-0.81</td></tr><tr><td rowspan=1 colspan=1>T-marg</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>-0.56</td><td rowspan=1 colspan=1>-0.67</td></tr><tr><td rowspan=1 colspan=1>T-entr</td><td rowspan=1 colspan=1>-0.73</td><td rowspan=1 colspan=1>0.24</td><td rowspan=1 colspan=1>0.78</td></tr><tr><td rowspan=1 colspan=1>S-conf</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>-0.10</td><td rowspan=1 colspan=1>-0.74</td></tr><tr><td rowspan=1 colspan=1>S-marg</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>-0.26</td><td rowspan=1 colspan=1>-0.58</td></tr><tr><td rowspan=1 colspan=1>S-entr</td><td rowspan=1 colspan=1>-0.58</td><td rowspan=1 colspan=1>0.01</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>KLTs</td><td rowspan=1 colspan=1>-0.37</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.40</td></tr><tr><td rowspan=1 colspan=1>disTS</td><td rowspan=1 colspan=1>-0.78</td><td rowspan=1 colspan=1>0.60</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=2 colspan=1>aliīsdisTz</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>-0.01</td><td rowspan=1 colspan=1>-0.16</td></tr><tr><td rowspan=1 colspan=1>-0.15</td><td rowspan=1 colspan=1>0.16</td><td rowspan=1 colspan=1>0.11</td></tr><tr><td rowspan=1 colspan=1>KLTz</td><td rowspan=1 colspan=1>0.04</td><td rowspan=1 colspan=1>-0.02</td><td rowspan=1 colspan=1>-0.06</td></tr><tr><td rowspan=1 colspan=4> $\lambda _ { T }$     $\lambda _ { P }$     $\lambda _ { H }$ </td></tr></table>

![](images/d9b4ed76b1b70cd3820a1da496bea7d10c6013d032ff57bac98cdc8db7d7854d.jpg)

conflict reveals when adapted and zero-shot knowledge disagree, and feature alignment provides a representationlevel check on transfer quality. Their individual drops are smaller but consistent across both datasets, indicating that the best policy state combines confidence with relational cues. A single confidence score is not enough. Figure 5 visualizes the learned state-to-action mapping on the two representative ablation datasets, complementing the rowremoval ablations above with a continuous view of how the controller actually responds to its inputs. On EuroSAT, every cue moves its expected action direction. Teacher entropy and teacher-prior disagreement couple most strongly with the prior coeficient, while teacher-student disagreement and student entropy drive the hard-label coeficient. On FGVCAircraft the same cues are largely inert. The prior channel stays nearly dormant across the cue range, and only teacher entropy shows a weak positive slope on the prior coeficient. This pattern matches the row-removal result. Uncertainty cues carry most of the signal on EuroSAT, while on FGVCAircraft the cues themselves are less informative so the controller has less to work with. The figure therefore confirms that the learned policy is dataset-aware rather than a fixed rule, which is consistent with the dataset-dependent gains in Table 6.

## 4.5. Hyperparameter Analysis

We further analyze the sensitivity of ONPOKD to key hyperparameters. Because the controller is designed to make bounded interventions during training, the method should not rely on a narrow hyperparameter setting. Figure 4 summarizes sweeps on EuroSAT, a representative domainshifted benchmark, reporting both Base-to-novel generalization metrics and Cross-dataset transfer accuracy. Two patterns emerge. The auxiliary cap and policy regularization most clearly afect the Base-to-novel generalization tradeof, whereas Cross-dataset transfer is less monotonic and stays sensitive to the exact intervention scale.

The auxiliary target cap $\lambda _ { \operatorname* { m a x } } ^ { A }$ mainly controls the Baseto-novel generalization balance. As Figure 4(a) shows, raising the cap from 0.10 to 0.40 improves EuroSAT HM from 90.32 to 92.02, driven by a novel-class gain from 84.03 to 87.01 while base accuracy stays high between 97.12 and 97.64. Raising the cap further to 0.50 reduces HM to 90.95 and novel accuracy to 85.31, so auxiliary intervention is helpful only while it remains bounded. Cross-dataset transfer accuracy is comparatively flat, ranging from 69.02 at 0.10 to a peak of 69.43 at $\lambda _ { \operatorname* { m a x } } ^ { A } ~ = ~ 0 . 3 0$ . This sweep supports the bounded-intervention design. The policy needs enough capacity to correct the teacher, but excessive auxiliary mass weakens the Base-to-novel generalization balance.

Policy regularization prevents unstable target construction, but its best strength is intermediate. As Figure 4(b) shows, removing regularization $( \alpha _ { r } = 0 )$ lowers EuroSAT HM to 87.71 and novel-class accuracy to 80.21, the signature of an unconstrained policy making overly aggressive target updates. A small regularizer, $\alpha _ { r } ~ = ~ 0 . 0 1$ , restores HM to 90.72, and $\alpha _ { r } = 0 . 0 5$ gives the best Base-to-novel generalization result (97.64 Base, 87.01 Novel, 92.02 HM). Crossdataset transfer accuracy peaks at $\alpha _ { r } ~ = ~ 0 . 1 0$ with 69.43 and stays close at $\alpha _ { r } = 0 . 0 5$ with 69.31, while raising the regularizer to 0.20 lowers both HM and transfer accuracy. The sweep thus confirms the role of $\alpha _ { r }$ as a stabilizer that should curb erratic actions without suppressing useful adaptation.

The action-strength sweep shows that stronger intervention is not always better. Figure 4(c) reports the best EuroSAT Base-to-novel generalization HM at $\ s \ = \ 0 . 5 0$ where Novel reaches 87.01 and HM reaches 92.02. A smaller scale, $s = 0 . 2 5$ , gives the highest Base accuracy of 97.76 but lowers Novel to 82.23 and HM to 89.33. Cross-dataset transfer accuracy aligns less with HM. It falls to 65.25 at $s =$ 1.00 and recovers to 69.43 at $s = 1 . 5 0$ , a setting whose HM is only 89.63. Action strength therefore controls how sharply the controller translates reliability cues into target changes, and the best setting for Cross-dataset transfer can difer from the best setting for Base-to-novel generalization balance. This reinforces why ONPOKD uses bounded actions and separate controls for mixture, weight, and temperature. Taken together, the sweeps show that ONPOKD is governed by two forms of control. The auxiliary cap determines how much non-teacher supervision is available (the best Base-tonovel generalization balance appears at $\lambda _ { \operatorname* { m a x } } ^ { A } = 0 . 4 0$ and the best Cross-dataset transfer accuracy at 0.30), while policy regularization and action scale determine how safely that supervision is used, with $\alpha _ { r } = 0 . 0 5$ and $s = 0 . 5 0$ best for Baseto-novel generalization. Cross-dataset transfer accuracy is less monotonic, suggesting that transfer robustness depends on both the amount of intervention and the way the controller applies it. These results support the central design choice of ONPOKD. Target construction should be adaptive enough to correct unreliable teacher targets, yet each intervention should remain bounded and feedback-stabilized.

## 5. Conclusion

This paper presented ONPOKD, an on-policy distillation framework for vision-language model adaptation. Instead of imposing a fixed teacher-centered target on every training sample, ONPOKD treats target construction as a policy decision that adaptively mixes the adapted teacher, the frozen zero-shot prior, and hard-label supervision according to reliability cues and validation feedback. Because the policy only shapes the training objective, the distilled student keeps the same inference architecture and adds no test-time computation. Experiments on Base-to-novel generalization and Cross-dataset transfer show that adaptive target construction improves the robustness and transferability of vision-language distillation, with clear gains on challenging datasets such as FGVCAircraft, DTD, and EuroSAT. Ablation studies further confirm that the gains do not simply come from adding more supervision sources. The zeroshot prior, hard-label anchoring, validation feedback, and reliability-aware policy cues each contribute to stable target construction. These results suggest that efective visionlanguage distillation requires not only multiple sources of supervision, but also a mechanism that decides when and how each source should influence student learning. Future work can explore more expressive policy-learning objectives, stronger validation-feedback signals, and applications beyond classification.

## References

Bossard, L., Guillaumin, M., Van Gool, L., 2014. Food-101–mining discriminative components with random forests, in: Computer vision– ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13, Springer. pp. 446–461.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., Vedaldi, A., 2014. Describing textures in the wild, in: Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3606–3613.

Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L., 2009. Imagenet: A large-scale hierarchical image database, in: 2009 IEEE conference on computer vision and pattern recognition, Ieee. pp. 248–255.

Fei-Fei, L., Fergus, R., Perona, P., 2004. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories, in: 2004 conference on computer vision and pattern recognition workshop, IEEE. pp. 178–178.

Helber, P., Bischke, B., Dengel, A., Borth, D., 2019. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12, 2217–2226.

Hendrycks, D., Basart, S., Mu, N., Kadavath, S., Wang, F., Dorundo, E., Desai, R., Zhu, T., Parajuli, S., Guo, M., et al., 2021. The many faces of robustness: A critical analysis of out-of-distribution generalization, in: Proceedings of the IEEE/CVF international conference on computer vision, pp. 8340–8349.

Hinton, G., Vinyals, O., Dean, J., 2015. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531 .

Hsieh, C.Y., Li, C.L., Yeh, C.K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.Y., Pfister, T., 2023. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. arXiv preprint arXiv:2305.02301 .

Huang, S., Zhang, H., Li, X., 2025. Enhance vision-language alignment with noise, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 17449–17457.

Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T., 2021. Scaling up visual and vision-language representation learning with noisy text supervision, in: International conference on machine learning, PMLR. pp. 4904–4916.

Khattak, M.U., Rasheed, H., Maaz, M., Khan, S., Khan, F.S., 2023a. Maple: Multi-modal prompt learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19113– 19122.

Khattak, M.U., Wasim, S.T., Naseer, M., Khan, S., Yang, M.H., Khan, F.S., 2023b. Self-regulating prompts: Foundational model adaptation without forgetting, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15190–15200.

Krause, J., Stark, M., Deng, J., Fei-Fei, L., 2013. 3d object representations for fine-grained categorization, in: Proceedings of the IEEE international conference on computer vision workshops, pp. 554–561.

Lee, D., Song, S., Suh, J., Choi, J., Lee, S., Kim, H.J., 2023. Readonly prompt optimization for vision-language few-shot learning, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1401–1411.

Li, Z., Li, X., Fu, X., Zhang, X., Wang, W., Chen, S., Yang, J., 2024. Promptkd: Unsupervised prompt distillation for vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26617–26626.

Li, Z., Li, X., Yang, L., Zhao, B., Song, R., Luo, L., Li, J., Yang, J., 2023. Curriculum temperature for knowledge distillation, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 1504–1512.

Maji, S., Rahtu, E., Kannala, J., Blaschko, M., Vedaldi, A., 2013. Finegrained visual classification of aircraft. arXiv preprint arXiv:1306.5151

Nilsback, M.E., Zisserman, A., 2008. Automated flower classification over a large number of classes, in: 2008 Sixth Indian conference on computer vision, graphics & image processing, IEEE. pp. 722–729.

Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., Lowe, R., 2022. Training language models to follow instructions with human feedback, in: Advances in Neural Information Processing Systems, pp. 27730–27744.

Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C., 2012. Cats and dogs, in: 2012 IEEE conference on computer vision and pattern recognition, IEEE. pp. 3498–3505.

Qiao, X., Huang, P., Yuan, J., Guo, X., Ye, B., Xue, C., Zheng, Y., Sun, Z., Li, X., 2026a. Bidirectional prototype-reward co-evolution for testtime adaptation of vision-language models. IEEE Transactions on Multimedia .

Qiao, X., Wang, W., Zhao, Z., Sun, J., Luo, P., Zhang, H., Li, X., 2026b. Ahap: Reconstructing arbitrary humans from arbitrary perspectives with geometric priors. arXiv preprint arXiv:2602.23951 .

Qiao, X., Zhang, D., Guo, Y., Gao, J., Zhao, Z., Li, X., 2026c. Semanticaware temporal adaptation for uav anti-uav tracking. arXiv preprint arXiv:2607.26511 .

Qiao, X., Zhao, J., Jiang, Y., Guo, X., Sun, Z., Zhang, H., Li, X., 2025. Class-aware prototype learning with negative contrast for test-time adaptation of vision-language models. arXiv preprint arXiv:2510.19802 .

Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: International conference on machine learning, PMLR. pp. 8748–8763.

Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C.D., Finn, C., 2023. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290 .

Recht, B., Roelofs, R., Schmidt, L., Shankar, V., 2019. Do imagenet classifiers generalize to imagenet?, in: International conference on machine learning, PMLR. pp. 5389–5400.

Soomro, K., Zamir, A.R., Shah, M., 2012. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402

Wang, Z., Yan, Z., Yang, M.H., Pan, J., Gao, G., Tai, Y., Yang, J., 2026. Scene prior filtering for depth super-resolution: Zhengxue wang et al. International Journal of Computer Vision 134, 251.

Wu, K., Peng, H., Zhou, Z., Xiao, B., Liu, M., Yuan, L., Xuan, H., Valenzuela, M., Chen, X.S., Wang, X., et al., 2023. Tinyclip: Clip distillation via afinity mimicking and weight inheritance, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 21970– 21980.

Xiao, J., Hays, J., Ehinger, K.A., Oliva, A., Torralba, A., 2010. Sun database: Large-scale scene recognition from abbey to zoo, in: 2010 IEEE computer society conference on computer vision and pattern recognition, IEEE. pp. 3485–3492.

Xu, G., Chen, J., Huang, W., Jia, W., Gao, G., Qi, G.J., 2026. Scaseg: Strip cross-attention for eficient semantic segmentation. IEEE Transactions on Image Processing .

Yang, C., An, Z., Huang, L., Bi, J., Yu, X., Yang, H., Diao, B., Xu, Y., 2024. Clip-kd: An empirical study of clip model distillation, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15952–15962.

Yang, J., Martinez, B., Bulat, A., Tzimiropoulos, G., 2021. Knowledge distillation via softmax regression representation learning, in: International conference on learning representations.

Yao, L., Huang, R., Hou, L., Lu, G., Niu, M., Xu, H., Liang, X., Li, Z., Jiang, X., Xu, C., 2021. Filip: Fine-grained interactive language-image pre-training. arXiv preprint arXiv:2111.07783 .

Yu, T., Lu, Z., Jin, X., Chen, Z., Wang, X., 2023. Task residual for tuning vision-language models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10899–10909.

Yuan, L., Chen, D., Chen, Y.L., Codella, N., Dai, X., Gao, J., Hu, H., Huang, X., Li, B., Li, C., et al., 2021. Florence: A new foundation model for computer vision. arXiv preprint arXiv:2111.11432 .

Zhai, X., Wang, X., Mustafa, B., Steiner, A., Keysers, D., Kolesnikov, A., Beyer, L., 2022. Lit: Zero-shot transfer with locked-image text tuning, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18123–18133.

Zhang, H., Huang, S., Guo, Y., Li, X., 2025. Variational positive-incentive noise: How noise benefits models. IEEE Transactions on Pattern Analysis and Machine Intelligence .

Zhang, H., Xu, Y., Huang, S., Li, X., 2026. Data augmentation of contrastive learning is estimating positive-incentive noise, in: Proceedings of the 43rd International Conference on Machine Learning (ICML).

Zhao, B., Cui, Q., Song, R., Qiu, Y., Liang, J., 2022. Decoupled knowledge distillation, in: Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, pp. 11953–11962.

Zhao, S., Xie, Z., Liu, M., Jing, H., Pang, G., Chen, F., Grover, A., 2026. Self-distilled reasoner: On-policy self-distillation for large language models. arXiv preprint arXiv:2601.18734 .

Zhou, K., Yang, J., Loy, C.C., Liu, Z., 2022a. Conditional prompt learning for vision-language models, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16816–16825.

Zhou, K., Yang, J., Loy, C.C., Liu, Z., 2022b. Learning to prompt for visionlanguage models. International Journal of Computer Vision 130, 2337– 2348.

Zhou, L., Li, W., Li, J., Gao, G., Lin, C.W., 2026. Difusion-based laplacian frequency-aware network for low-light image enhancement. Pattern Recognition , 113060.

Zhu, R., Huang, S., Jiao, Z., Zhang, H., 2026. Explore how to inject beneficial noise in mllms, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 29150–29158.

Zhu, R., Huang, Z., Sun, J., Luo, P., Zhang, H., Li, X., 2025. Viewmask-1-to-3: Multi-view consistent image generation via multimodal discrete difusion models. arXiv preprint arXiv:2512.14099 .