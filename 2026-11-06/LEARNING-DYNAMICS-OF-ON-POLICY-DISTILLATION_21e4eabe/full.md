# LEARNING DYNAMICS OF ON-POLICY DISTILLATION

Guan Zhe Hong<sup>1</sup> Sumeet R. Motwani<sup>1</sup> Liang Chen<sup>2</sup> Yangchen Pan<sup>3</sup>

Junchi Yu<sup>1</sup> Jialin Yu<sup>1</sup> Philip H.S. Torr<sup>1</sup> Yi Ren<sup>1</sup>

<sup>1</sup>University of Oxford <sup>2</sup>EPFL <sup>3</sup>City University of Hong Kong

## ABSTRACT

On-policy distillation (OPD) is emerging as an important paradigm for languagemodel post-training, combining dense teacher supervision with training on studentgenerated trajectories. This creates a feedback loop where each update changes the student and thereby the training distribution that generates subsequent updates. Yet, how this feedback shapes learning and generalization remains poorly understood. We conduct a mechanistic analysis of OPD’s learning dynamics, distinguishing the two feedback loop stages: conversion of the distillation signal into parameter updates, and broadcast through which these updates change predictions across prefixes and reshape the rollout distribution. This framework helps us understand both limited student responsiveness despite substantial teacher-student disagreement, and degeneration when updates that improve local teacher agreement also increase the probability of unsuccessful continuations, exposing the student to unreliable supervision in subsequent rollouts. Finally, we show that offline distillation can prepare a student for OPD without significantly improving immediate held-out accuracy. While OPD alone may degrade generation, it can yield substantial accuracy gains after sufficient offline preparation. Together, our results connect teacher supervision, student geometry, and the evolving rollout distribution in determining the learning dynamics of OPD and its effects on downstream performance.

## 1 INTRODUCTION

On-policy distillation (OPD) is becoming an important component of language-model post-training (Lu & Thinking Machines Lab, 2025; Song & Zheng, 2026a). By matching a teacher’s next-token distributions at student-generated prefixes, it combines dense supervision with training on the states the student actually visits (Gu et al., 2024; Agarwal et al., 2024). Yet, how this local supervision becomes useful learning over an evolving training distribution remains insufficiently understood. It is therefore important to theoretically understand both the conditions under which OPD improves generalization and its role alongside offline distillation.

The difficulty is that each OPD update changes both the student’s current behavior and the prefixes on which it will subsequently learn. Reducing teacher-student divergence on currently supervised predictions does not necessarily produce parameters from which further updates are useful (Li et al., 2026b; Armandpour et al., 2026). Modelling this requires a view of how supervision changes behavior beyond the current batch and how that impacts learning at later stages. We conduct a mechanistic analysis, aided by a first-principles account, of OPD learning dynamics to study this process, explain existing empirical challenges, and justify their empirical diagnostics and fixes.

We first characterize how the teacher-student log-probability difference can lead to meaningful weight updates, which depends on the overlap between the distillation signal and the logit-space directions the student’s current parameters can express. We then describe how learning at one prefix affects predictions at future ones through their dependence on shared parameter directions. We formalize this broadcast relationship through the student’s tangent kernel between the two prefixes, building on Ren & Sutherland (2025); Malladi et al. (2023). Finally, we connect each update to its effect on the outcomes of student-generated sequences. The effect of these updates in broader settings also depends on what the training data covers: matching the teacher on narrow tasks may transfer only the capabilities that those tasks exercise. Integrating these effects over training characterizes what OPD can reach from a given starting point, accounting for how much successive updates improve or degrade performance.

![](images/b9a86b43150bd0c75b9e1563fd57c046550386aecff978ead2bbd62c8de20f5d.jpg)  
Figure 1: OPD’s learning dynamics, visualizing the feedback loop at the top (detailed in Section 3.1), and illustrative successful and failing processes at the bottom in reasoning settings we analyzed empirically. In principle, OPD propagates current teacher requests $\Gamma _ { t } ( s ) | _ { s \sim \mu _ { t } }$ to motion at any future prefix x through shared parameters, captured by the Neural Tangent Kernel $K ( x , s )$ . Such policy motion shapes how the autoregressive rollout (prefix visitation) and supervision distribution changes $( \dot { \mu } _ { t } , \dot { \Gamma } _ { t } )$ . In practice, a healthy dynamics needs not only locally “tangent-accessible” and coherent requests (Section 3.2), but also the student to be sufficiently “prepared”, so that it can sustain meaningful rollouts with learning value as it absorbs corrections requested by the teacher, instead of primarily performing local mimicry and falling into chaotic rollouts hard for the teacher to rescue, let alone improve on (Sections 3.3 and 3.4).

This framework helps us understand two primary failure modes of OPD: stagnation under stronger teachers and degeneration into incoherent outputs (Li et al., 2026b; Fu et al., 2026; Wang et al., 2026). In the first case, despite substantial teacher–student log-probability differences, we show that the initial distillation signal can induce smaller parameter gradients as training proceeds, contributing to stalled improvement. In the second case, updates that reduce teacher–student disagreement at supervised prefixes also change predictions at other prefixes through shared parameters. These changes can shift the rollout distribution toward continuations the student cannot execute reliably, making those unstable continuations the source of subsequent supervision, making the teacher struggle to rescue, let alone improve the student. We illustrate the dynamical process of OPD and its possible failure modes in Figure 1 (we will elaborate on the intuitions behind this further in Section 3.1).

The dependence of OPD on the student sustaining stable rollouts (prefixes) which induce meaningfu supervision motivates offline preparation. Here, the weak student is first forced to visit the teacher’s prefixes and learn predictions there, enabling the subsequent OPD process to sustain productive learning as the student exposes finer correction opportunities to the teacher, while maintaining stabler and more recoverable changes in its rollout distribution. Across Qwen-2.5 and Llama-3.1 models on AIME 2024 and 2025 problems, we find that under sufficient offline training, the subsequent OPD phase substantially improve pass@k (for k up to 32) and per-problem reliability compared to offline training alone. In the rest of the paper, we present these results and insights organized under the following three questions. What determines the conversion of teacher supervision into policy motion (Section 3.2)? Does the broadcast of these updates across prefixes improve subsequent reasoning or steer the student toward unreliable continuations and supervision (Section 3.3)? Can offline preparation strengthen the student’s continuation ability and enable productive on-policy learning (3.4)?

Related works. We briefly discuss related works here, and leave the expanded version to Appendix Section B. On-policy distillation. Training on learner-generated states with expert feedback is a classical approach to addressing distribution shift in imitation learning (Ross et al., 2011), applied to autoregressive distillation by Lin et al. (2020). Gu et al. (2024) develop reverse-KL distillation using student rollouts, while Agarwal et al. (2024) unify on-policy and off-policy training across multiple divergences. Lu & Thinking Machines Lab (2025) demonstrate its effectiveness for reasoning and personalization in language-model post-training. More recently, Yang et al. (2026) interpret OPD as dense KL-constrained reinforcement learning and introduce reward extrapolation. Empirical studies identify teacher-student compatibility on student prefixes as important factors in transfer and training failure (Li et al., 2026b; Fu et al., 2026).

Learning dynamics offinetuning. Kernel-based analyses connect parameter updates to changes in language-model predictions (Malladi et al., 2023). Ren & Sutherland (2025) decompose perstep prediction changes using the loss gradient and empirical tangent kernel, analyzing confidence dynamics in SFT and DPO (Rafailov et al., 2024). Tomihari (2026) study how RL updates propagate across examples, while Shen et al. (2026) identify persistent low-dimensional subspaces in OPD updates. Building on these perspectives, we connect the student’s evolving tangent kernel to its response to teacher supervision, the quality of its continuations, and the resulting distribution of future training prefixes. This distinguishes limited response to supervision from policy changes that worsen generation, and explains how either can shape subsequent learning.

## 2 SETUP

On-policy distillation (OPD) trains a student on its own samples. The student generates a continuation from a prompt. At every prefix of that continuation, the training loss pulls the student’s next-token distribution toward the next-token distribution of a fixed teacher (Gu et al., 2024; Agarwal et al., 2024; Lu & Thinking Machines Lab, 2025). This section fixes the notation for the objects that recur in the paper, including the policies, the source distribution that a run trains on, the request field that the teacher induces on that source, and the evaluation targets that we score runs against.

Policies and rollouts. A prefix s is a prompt followed by a partial continuation. Prefixes range over a space S, and tokens range over a finite vocabulary V. The student has parameters $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ , logits $z _ { \theta } ( s ) \in \mathbb { R } ^ { | \nu | }$ , and next-token distribution $p _ { \theta } ( a \mid s ) =$ softmax $( z _ { \theta } ( s ) ) _ { a }$ . The teacher $q ( \cdot \mid s )$ is fixed throughout training. Training proceeds in steps $t = 0 , 1 , 2 , \ldots$ . Gradient flow, the continuous-time version of the training process, has the update rule $\dot { \theta } _ { t } = u _ { t }$ . For discrete gradient steps this gives a first-order local approximation Section C.2. Two distributions share the student’s weights. The model $p o l i c y ~ p _ { t }$ is the distribution that the loss acts on. The rollout policy $\pi _ { t } = \mathcal G [ p _ { t } ]$ is the distribution that generates the samples (the transform $\mathcal { G }$ collects the temperature, end-of-sequence conventions, etc). We sample prompts $b \sim \beta$ and continue them under $\pi _ { t }$ . The step-t rollout population is $\begin{array} { r } { \Pi _ { t } ( \xi ) = \beta ( b ) \prod _ { j } \pi _ { t } ( a _ { j } \mid s _ { j } ) } \end{array}$ for a trajectory ${ \xi = ( b , a _ { 0 } , a _ { 1 } , . . . ) }$ , with $s _ { 0 } = b$ and $\boldsymbol { s } _ { j + 1 } = \boldsymbol { s } _ { j } \oplus \boldsymbol { a } _ { j }$ up to termination or the cap. The training source distribution $\mu _ { t } = \mathcal { M } ( \Pi _ { t } )$ is the distribution over prefixes that the model trains on (M normalizes over the rollouts to form the probability distribution).

Loss, request field, and update. For reverse Kullback–Leibler divergence, the per-prefix loss is $c _ { t } ( s ) = D _ { \mathrm { K L } } ( p _ { t } ( \cdot \mid s ) \parallel q ( \cdot \mid s ) )$ . Write $\ell _ { t } ( s , a ) = \log \big ( p _ { t } ( a \mid s ) / q ( a \mid s ) \big )$ for the log-ratio of the two policies at a token, and $\bar { \ell } _ { t } ( s ) = \mathbb { E } _ { a \sim p _ { t } } [ \ell _ { t } ( s , a ) ]$ for its mean under the student. The negative gradient of the loss with respect to the logits is $B _ { t } ( s , a ) \ = \ p _ { t } ( a \ | \ s ) \ P _ { t } ( s ) - \ell _ { t } ( s , a ) \big \vert$ We call $B _ { t }$ the request $f e l d ^ { 1 }$ . It captures the student-teacher disagreement on a given prefix. The product $\Gamma _ { t } ~ = ~ \mu _ { t } B _ { t }$ , visitation times request, is the request population. The student’s tangent map $\Phi _ { t } ( s ) = \varphi _ { \theta } z _ { \theta _ { t } } ( s ) \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ . The map pulls a request in logit space back into parameter space, giving the per-prefix gradient $g _ { t } ( s ) = \bar { \Phi _ { t } } ( s ) ^ { \top } B _ { t } ( \bar { s } ) = - \nabla _ { \theta } \bar { c } _ { t } ( s )$ and the update direction $u _ { t } = \mathbb { E } _ { s \sim \mu _ { t } } [ g _ { t } ( s ) ]$ ]. See Section C.1 for further details on the setup.

## 3 THE LEARNING DYNAMICS OF THE OPD LOOP

## 3.1 THE OPD LOOP AND THE RESPONSE LAW

Recall that $\Pi _ { t }$ denotes the distribution of student rollouts and $\mu _ { t } = \mathcal { M } ( \Pi _ { t } )$ the distribution of prefixes used by the loss. At a sampled prefix $s \sim \mu _ { t }$ , the student $p _ { t }$ and the teacher $q$ supply the request $B _ { t } ( s ) \dot { = } - \nabla _ { z _ { t } ( s ) } c _ { t } ( s )$ . The request population $\Gamma _ { t } ( s ) = \mu _ { t } ( s ) B _ { t } ( s )$ specifies what change it requests on a prefix, across the rollout disitribution $\mu _ { t }$ . These requests then convert into parameter motion through the tangent map $\Phi _ { t } ( s ) = \nabla _ { \theta } z _ { t } ( s )$ , which changes the future policy and rollouts.

More explicitly, one training step follows

$$
\underbrace { \Pi _ { t } } _ { \mathrm { r o l l o u s } } \xrightarrow { { \mathcal M } } \underbrace { \mu _ { t } } _ { \mathrm { s o u r c e } } \xrightarrow { q } \underbrace { \Gamma _ { t } = \mu _ { t } B _ { t } } _ { \mathrm { w e i g h t e d r e q u e s t s } } \xrightarrow { \Phi } \underbrace { \Delta \theta _ { t } } _ { \mathrm { p a r a m e t e r s t e p } } \xrightarrow { \mathrm { r e s p o n s e } } \underbrace { \Delta \log p _ { t } ( \cdot ) } _ { \mathrm { p o l i c y c h a n g e s } } \longrightarrow \underbrace { \Pi _ { t + 1 } } _ { \mathrm { n e x t r o l l o u s } }\tag{1}
$$

Local requests shape future motion. Training imposes the gradient-based update $u _ { t } = \mathbb { E } _ { \mu _ { t } } [ \bar { \Phi _ { t } } ( s ) ^ { \top } B _ { t } ( s ) ]$ . For a gradient step $\Delta \theta _ { t } ~ = ~ \eta u _ { t }$ , the response at any receiver prefix x is

$$
\Delta p _ { t } ( a \mid x ) = \eta \sum _ { s } \underbrace { \mu _ { t } ( s ) } _ { \mathrm { s o u r c e ~ w e i g h t } } \left[ \underbrace { { \mathsf { A } } _ { t } ^ { p } ( x ) } _ { \mathrm { n o r m a l i z a t i o n } } \overbrace { \underbrace { K _ { t } ( x , s ) } _ { \mathrm { s u d e n t ~ c o u p l i n g } } \underbrace { B _ { t } ( s ) } _ { \mathrm { r e q u e s t } } } ^ { \Delta z _ { t } ( x ) : \mathrm { l o g i t ~ r e s p o n s e } } \right] _ { a } + O ( \eta ^ { 2 } ) ,\tag{2}
$$

where $K _ { t } ( x , s ) = \Phi _ { t } ( x ) \Phi _ { t } ( s ) ^ { \top }$ and $\mathsf { A } _ { t } ^ { p } ( x ) = I - \mathsf { 1 } p _ { t } ( \cdot \mid x ) ^ { \top }$ . The kernel $K _ { t }$ measures how source and receiver logits move together through shared parameters; $\mathsf { A } _ { t } ^ { p }$ converts logit motion into log-probability motion. Therefore, the model’s prediction at any “receiver” prefix x can change without appearing in the training batch.

In particular, since every prefix x in afuture rollout (from the future policy) is a receiver, (1) OPD updates will affect local per-prefix predictions of the student, captured by the logit motion z˙(x) and policy motion ∂ log p(· | x), (2) the updates will also shape which prefixes the student visits (changed µ) and expose to teacher supervision (changed Γ), as the policy motion along the ancestor tokens of any prefix x accumulate to change its reach-probability.

Let us look at the illustrative process in Figure 1. Intuitively, imagine that the earlier requests locally promote the student to reason more and in specific manners (for instance, by requesting more “wait”-based backtracking, suppressing early “Answer:” commitment, asking for a specific theorem, and so on). Due to the autoregressive nature of the policies, this will not only affect how likely the student emits specific tokens such as “wait” or “answer:” in a future rollout of the student, but will also alter the overall rollout distribution, by lengthening the reasoning trace, exposing new reasoning sub-trajectories, different errors, etc. In the benign case, such online dynamics can be beneficial, as the teacher can correct the student in real time as if in a “tutorial” session.

However, one can also imagine that such a dynamical process can be quitefragile for a ill-prepared student. A weak student’s tangent map Φ simply might not be very sensitive to the requests, resulting in limited motion propagated from training to future prefixes via kernel K, thus limited changes in task accuracy. Moreover, even if there is substantial motion, such motion can be harmful. Taking our analogy further, an ill-prepared student which has not “gone to class before the tutorial” might benefit very little from online corrections on its own rollouts and even become more confused. That is, the ill-prepared student which has not been pretrained properly on the task and sustain sufficiently strong rollouts might only mimic superficial local patterns requested by the teacher, and push its rollout population µ towards chaos, which can then be hard for the teacher to generate the right request population Γ to even rescue, let alone improve.

The rest of our analysis builds on the OPD loop and the response law to make precise the above intuitions, complemented by empirical analysis of real OPD runs, and examine how offline-distillation preparation before OPD helps alleviate the problems for weak student models. Section 3.2 (conversion) studies the student’s response to the supervised requests, which primarily concerns whether and how well a student model can convert different request populations Γ (from different teachers) into policy motion z˙ along the request fields. We develop early-training diagnostics of OPD runs from this perspective. Section 3.3 (broadcast) then focuses on where policy motion z˙ goes and how it influences future training sources $\mu ,$ and analyze how and why policy motion which improves local per-prefix readouts (such as training loss) might not make positive progress on terminal readouts such as answer accuracy. Motivated by the failure mechanisms, we then show in Section 3.4, across Qwen-2.5 and Llama-3.1 models, that with sufficient offline preparation, OPD runs improve pooled held-out accuracy (up to pass@32) and aggregate per-problem reliability on AIME 2024 and 2025 problems.

In the rest of this section, we primarily rely on the continuous gradient flow for cleaner theoretical discussions. Please see Section C.2 for further details on the finite step approximations.

## 3.2 CONVERSION: TEACHER REQUESTS AND THE STUDENT’S RESPONSIVENESS

A strong teacher can transfer little. We open with an intriguing empirical puzzle identified in the OPD community (Li et al., 2026b): a larger (same-family) teacher can transfer less than an RL-trained sibling even when they are nearly matched in bechmark performance, and Li et al. (2026b) suggests cancellation among token-level gradient signals as a possible explanation. We revisit this comparison at 1.5B and 7B (Figure 2), holding the starting DeepSeek-R1-Distill-Qwen student and OPD recipe fixed within each scale. JustRL-1.5B and AceReason-Nemotron-7B yield substantial gains, whereas the larger DeepSeek-7B and DeepSeek-14B teachers transfer little throughout training despite higher benchmark scores (Section D.1).

![](images/8b3d8592b02d011f7e958f0787ed76ce245665539101c8adfa8d3b45aae3599d.jpg)  
Figure 2: Initial/end AIME accuracy; diamonds mark teachers. Red: sibling-taught; blue: larger-teacher taught.

We call these runs healthy and stalled. We find that the teachers’ request fields differ substantially in semantic functions, and that the student’s responsiveness to these requests separates in two axes: coherence of the gradient signals, and kernel adaptation toward the requests. Along the way, we develop early-training diagnostics that help reveal the “request-conversion” health of a training run.

![](images/8c29c9d038021e20b03b59e1e6d034877d63d844d02d699712108d60608b6a90.jpg)  
(a) Share vs. all positions (points)  
(b) 1.5B student  
(c) 7B student  
Figure 3: Teacher request fields differ systematically. In (a), each curve shows how much more (or less) often a semantic or token-position class appears among a teacher’s top 50%, 20%, . . . , 1% of requests than among all annotated positions, in percentage points (so $0 = { \mathfrak { n } } 0$ preference). The RL-sibling teachers’ increasingly emphasize requests on sentence openings and “control-oriented” moves, while the larger teachers’ stay in sentence interiors and “technical-oriented” content. In (b) we show finer semantic categories of the request fields. Each dumbbell pairs the teachers’ enrichment among their 1% largest-normed of initial requests. 1 denotes corpus-level prevalence. See Section D.2 for finer details on the classes and how we conduct the interpretability analysis.

How do the request fields differ? As we see in the OPD update loop in Section 3.1, the teacher’s request population $\Gamma _ { t }$ at time 0 “kickstart” the whole learning process, from which the sharedparameter student model propagates into policy motion via its kernel $K _ { t } .$ . Since the student starts out to be the same while the teachers are different across the healthy and stalled runs, their request fields become the natural entry point for our analysis.

On fixed initial-student rollouts, Figure 3 compares where each teacher’s largest-normed requests occur. (a) shows the coarse tendencies of the semantics of the requests as we move up the requestnorm percentile, while (b) and (c) show the finer semantic categories and contrast the teachers’ distinct request fields. We find that the RL-sibling teachers increasingly emphasize reasoning-management and initiating positions, while the larger teachers place relatively more of their strongest requests at technically oriented (e.g. digit-bearing, recalling mathematical fact) positions. This separation connects to findings in the reasoning-interpretability community, that RL tends to improve how well the model “orchestrates” the base reasoning mechanisms (including epistemic control abilities) (Venhoff et al., 2025a). Full setup and analysis results are in Section D.2.

From requests to motion. Figure 3 shows that the healthy and stalled teachers provide systemati cally different request fields in the large-normed stratum, but this alone does not help us understand which teacher’s request field converts better into policy motion along along the request field. Recall that for a prefix population $\nu ,$ the gradient is $g _ { t } \dot { ( } s ) \dot { = } \Phi _ { t } ( s ) ^ { \top } B _ { t } ( \dot { s } )$ in terms of the tangent map and the request, $\overline { { u _ { t } } } ^ { \prime } = \mathbb { E } _ { \nu } g _ { t } ( s )$ the average gradient, and $\dot { P } _ { t } ^ { ' } = \dot { \mathbb { E } } _ { \nu } \| \boldsymbol { B } _ { t } ( s ) \| ^ { 2 }$ the request norm, or “pressure” (recall B is the request field induced by the teacher-student disagreement). Under gradient flow $\dot { \theta } _ { t } = u _ { t }$ , the coefficient ofthe logit response $\dot { z } _ { t }$ along the requestfield $B _ { t }$ on ν can be written as

$$
\Xi _ { t } [ B _ { t } ] = \frac { \langle B _ { t } , \dot { z } _ { t } \rangle _ { \nu } } { P _ { t } } = \frac { \| u _ { t } \| ^ { 2 } } { P _ { t } } , \qquad \dot { z } _ { t } = \Xi _ { t } [ B _ { t } ] B _ { t } + r _ { t } , \quad \langle B _ { t } , r _ { t } \rangle _ { \nu } = 0 ,\tag{3}
$$

where we define the distribution-level inner product $\langle F , G \rangle _ { \nu } = \mathbb { E } _ { \nu } [ F ^ { \top } G ]$ and $P _ { t } > 0$ . With nonzero denominators, this collective conversion gain factors as

$$
\begin{array} { r } { \Xi [ B ] = \underbrace { \frac { \mathbb { E } _ { s \sim \nu } \| \Phi ( s ) ^ { \top } B ( s ) \| ^ { 2 } } { P } } _ { \mathrm { a c c e s s i b i l i t y } ~ A } \underbrace { \frac { \| \mathbb { E } _ { s \sim \nu } g ( s ) \| ^ { 2 } } { \mathbb { E } _ { s \sim \nu } \| g ( s ) \| ^ { 2 } } } _ { \mathrm { c o h e r e n c e } ~ C } , \qquad \| u \| ^ { 2 } = P \Xi = P A C . } \end{array}\tag{4}
$$

Accessibility averages the self-kernel Rayleigh coefficients $B ( s ) ^ { \top } K ( s , s ) B ( s ) / \| B ( s ) \| ^ { 2 }$ , weighted by request energy: it measures how responsive the student’s tangent map $\Phi ( s )$ is to the requests $B ( s )$ . Coherence measures how well the resulting parameter updates $g ( s ) \bar { = } \dot { \Phi ( s ) } ^ { \top } B ( s )$ combine: aligned updates reinforce one another, while cancellation lowers C. Thus “conversion” of requests into parameter-update-induced policy motion not only depend on pressure $\mathbb { E } _ { \nu } \Vert B _ { t } ( s ) \Vert ^ { 2 }$ , but even if two runs have similar gradient norm with high pressure, the requests need not convert efficiently due to low accessibility (poor parameter-level motion per unit request norm at a prefix) or coherence (poor cross-prefix aggregation of parameter motion). The deeper geometric interpretations and derivations are in Section C.3. We now examine what the actual training runs reveal.

Coherence contrasts at 7B. At 7B, healthy requests have near equal accessibility as the stalled requests but 2.75–3.24× higher coherence, yielding 2.27–2.62× higher collective gain Ξ across the two sampled high-strength strata (Figure 4a). Therefore, even though the gradient norms across the healthy and stalled runs are similar, we see that it is the greater pressure in the stalled run which masks the weaker coherence of the update forces for the stalled run. For this 7B case, the gradient-incoherence conjecture in Li et al. (2026b) is corroborated.

Evolving accessibility contrasts at 1.5B. At 1.5B the mechanism is richer: initial accessibility is similar and coherence differs less; while we do see greater pressure for the healthy run (roughly 2–2.7×, which yields larger gradient norm), conversion of the requests at time 0 contrasts less across the two runs. This motivates examining how responsiveness evolves as the student learns.

First, we note that live accessibility changes with both the request $B _ { t }$ and the kernel $K _ { t }$ , for instance, in a healthy run, $B _ { t }$ can deplete learnable components and rotate away from the student’s tangent map. To isolate kernel motion and how accessibility changes, we freeze each teacher’s initial field $B _ { 0 }$ , prefix bank, and token coordinates, then reapply the same request field at every checkpoint<sup>2</sup>. The common initialization anchors both fields to the same student, while preserving their distinct request profiles we saw earlier with the different teachers. With $b _ { 0 } ( s ) =  { \bar { B } } _ { 0 } ( s ) / \| B _ { 0 } ^ { - } ( s ) \| , a _ { t } ( s ) =$ $\begin{array} { r } { b _ { 0 } ( s ) ^ { \top } K _ { t } ( s , s ) b _ { 0 } ( s ) = \sum _ { j } \lambda _ { j , t } ( s ) | v _ { j , t } ( s ) ^ { \top } b _ { 0 } ( s ) | ^ { 2 } } \end{array}$ . Here $( \lambda _ { j , t } , v _ { j , t } )$ are eigenpairs (eigenvalues and eigenvectors) of the self-kernel $K _ { t } ( s , s )$ , and the request-energy-weighted average of $a _ { t } ( s )$ is $A _ { t } [ B _ { 0 } ]$ Larger eigenvalues amplify accessible directions; turning strong eigendirections toward $B _ { 0 }$ makes more of the same request visible. Both can increase local responsiveness (Section C.4).

![](images/814e1a019600a5a0268e64fa6304c5c3c9e3b72d894924c108bfbd4814a499e2.jpg)  
(a) Initial conversion.

![](images/f55f1562e95ba10f061bfe62c48c04115848072c5a378669d153672eb9dbef55.jpg)

![](images/a959db12db74ab07c9993e3b3eefb1fb53fa5b3ee614491c963ccdba45c5a33f.jpg)  
(b) Amplification and reorientation. (c) Teacher-relative reorientation.  
Figure 4: Collective conversion and acquired responsiveness distinguish the pairings. (a) Healthy/stalled ratios; segments show the range across the CONTROL and TECHNICAL strata. (b) Median leading-three eigenvalue gain $G _ { 3 }$ and eigen-subspace overlap $O _ { 3 }$ . (c) Step-50 paired trainedminus-other-field projection gain. Definitions and full results are in Sections D.4 and D.5.

At 1.5B in the healthy run, leading-subspace overlap falls to 0.77 by step 50 while leading strength remains near its initial value (Figure 4b). This means its eigenframe re-orients towards the request field in the early phase of training with nearly no scaling-amplification in the leading eigenvectors: the fixed request gains a median 5.5 percentage points of top-three projection, and its paired gain exceeds the other teacher’s by 1.79 points (Figure 4c). By step 200, leading eigenvalue strength reaches 2.31× and median request-direction sensitivity 2.08× the baseline. In contrast, the stalled run contracts in leading eigenvalue strengths and sensitivity (0.67× and 0.83× respectively). At 7B in the healthy run, eigenvector rotation is much less pronounced and adaptation is primarily amplification of the existing eigenframe: leading strength reaches 1.25× with overlap 0.96 against the initial frame, while stalled leading strength falls to 0.74×.

These results distinguish healthy and stalled pairings early on in an OPD training run in the requests they receive and the responsiveness their tangent geometry develop, giving diagnostics beyond the standard gradient norm and training loss from the request-conversion aspect. The next sub-section will study what comes next: when there is significant policy motion, cumulated over time, where do they head to? Why do two student models OPD-trained under the same teacher, both reducing training loss substantially, have different generalization performance under OPD?

## 3.3 BROADCAST: WHY LOCAL AGREEMENT NEED NOT IMPROVE REASONING

Under JustRL-1.5B, both DeepSeek-R1-Distill-Qwen-1.5B and Qwen2.5-1.5B-Instruct substantially reduce their training loss. Yet DeepSeek gains accuracy (Section 3.2), while the base Qwen 1.5B model declines substantially on MATH within 100 steps. Why does learning from the same teacher help one learner and harm the other?

How do OPD updates change a per-prefix readout function? Let us first examine how an OPD update influences a local, per-prefix readout function of a rollout, such as the training loss or entropy. It acts on two coupled axes: it changes which prefixes are encountered in future rollouts from the model, and it changes the model’s predictions at afixed prefix.

More precisely, let us define a local prefix readout $Y _ { t } ( x )$ . Define $J _ { Y } ( t ) = \mathbb { E } _ { \nu _ { t } } Y _ { t } ( x )$ , where $\nu _ { t }$ is the observed prefix population. Then its rate of change (due to the parameter update) decomposes into

“routing” and “drift”:

$$
{ j _ { Y } } ( t ) = \underbrace { \sum _ { x } { { \dot { \nu } } _ { t } } ( x ) Y _ { t } ( x ) } _ { R _ { Y } ( t ) : \mathrm { { \ m u i n g } } } + \underbrace { { { \mathbb { E } } _ { { \nu } _ { t } } } { { \dot { Y } } _ { t } } ( x ) } _ { D _ { Y } ( t ) : \mathrm { { \ d i f f } } } , \qquad J _ { Y } ( b ) - J _ { Y } ( a ) = \int _ { a } ^ { b } ( R _ { Y } + D _ { Y } ) d t .\tag{5}
$$

Drift changes the local, per-prefix readout over the current training population, while routing changes the probability weight assigned to the current prefixes, that is, which prefixes get exposed to teacher supervision in future updates.

At the more fundamental level of OPD’s dynamics, both of these channels are projections of the $K { \cdot } \Gamma$ broadcasts. The distinction is where the same update broadcasts. Recall from the response law that the weighted request population $\Gamma _ { t }$ propagate through $K _ { t }$ to induce a logit response $v _ { t } ( x )$ at every receiver prefix x. With a temperature-τ sampler, receiver x and prediction token a, this gives token log-probability velocity

$$
\zeta _ { t } ( x , a ) = \partial _ { t } \log \pi _ { t } ( a \mid x ) = { \frac { ( e _ { a } - \pi _ { t } ( \cdot \mid x ) ) ^ { \top } v _ { t } ( x ) } { \tau } } , \qquad v _ { t } ( x ) = \sum _ { s } K _ { t } ( x , s ) \Gamma _ { t } ( s ) .\tag{6}
$$

At the observed prefix x itself, $v _ { t } ( x )$ changes its prediction, giving drift $D _ { Y } = \mathbb { E } _ { \nu _ { t } } [ \nabla _ { z } Y _ { t } ( x ) ^ { \top } v _ { t } ( x ) ]$ To reach that prefix, however, the autoregressive model must first generate its ancestor tokens, which also are the receivers of the K-Γ broadcast. Summing the log-probability velocity of the ancestor tokens gives the velocity of reach-probability of prefix $\begin{array} { r } { x , \lambda _ { t } ( x ) = \sum _ { j < h ( x ) } \zeta _ { t } ( x _ { j } , a _ { j } ) } \end{array}$ . This gives routing $R _ { Y } = \operatorname { C o v } _ { \nu _ { t } } ( Y _ { t } , \lambda _ { t } )$ . Furthermore, the prefix visitation change causes the future training prefix distribution $\mu$ to change: within any time interval $[ a , b ]$

$$
\mu _ { b } ( s ) \propto \mu _ { a } ( s ) e ^ { \Lambda _ { a : b } ( s ) } , \qquad \Lambda _ { a : b } ( s ) = \int _ { a } ^ { b } \lambda _ { t } ( s ) d t \ \to \ \Gamma _ { b } = \mu _ { b } B _ { b } .\tag{7}
$$

The new rollouts $( \mathrm { i . e . }$ . visited prefixes) supply the next requests and the induced Γ field, on which the new round of updates and broadcasts are made.

Intuitively, significant prefix-visitation change can be either beneficial or damaging. For a weak student, it can route into low-quality, chaotic rollout distributions on which the teacher struggle to give requests that can help recover (or raise) the student’s intermediate reasoning performance and terminal scores. On the flip side, the continually refreshed training rollouts also allow the possibility of “online corrections”, where problematic reasoning behaviors broadcasted from early training can be corrected by the teacher mid-training. We will see these effects in our empirical analysis later.

To empirically study the effects of routing and drift in influencing the local readouts, we estimate their effects by cross-scoring training rollout banks and learners at time intervals. Write $M _ { i j } =$ $\mathbb { E } _ { x \sim \nu _ { i } } Y _ { j } ( x )$ ) for the average readout of learner $j$ on the frozen bank of rollouts extracted at time step $i ,$ with $i , j \in \{ a , b \}$ . Replacing the bank at a fixed learner checkpoint measures routing; replacing the checkpoint on a fixed bank measures drift. Averaging the two replacement orders gives

$$
\begin{array} { r } { \widehat { R } _ { a : b } = \frac { 1 } { 2 } [ ( M _ { b a } - M _ { a a } ) + ( M _ { b b } - M _ { a b } ) ] , \qquad \widehat { D } _ { a : b } = \frac { 1 } { 2 } [ ( M _ { a b } - M _ { a a } ) + ( M _ { b b } - M _ { b a } ) ] . } \end{array}\tag{8}
$$

These contributions sum exactly to $M _ { b b } \mathrm { ~ - ~ } M _ { a a }$ . Our curves accumulate them over consecutive windows; its relation to the path integrals and further details are in Sections C.7 and $\mathrm { E } . 1 ^ { 3 }$

From local readouts to rollout-outcome readouts. So far we have discussed how the K-Γ broadcasts change the model’s predictions per prefix, how they accumulate along ancestor tokens to change prefix visitation, and how they influence a local per-prefix readout function. Correctness, by contrast, is non-local, and known only after completing thefull rollout, so its $J _ { Y }$ expression must consider how “aligned” the policy motion is with terminal score when cumulated along a rollout trajectory.

![](images/bce593d7ef23617f833a44a7a0b4b04e742148fdd1b2d43b6f8852ed77e2c8d5.jpg)  
(a) Loss change

![](images/6f2e3c600792ffa6c862784ad50eafbddb491b4190085756e68585241ec2cd1f.jpg)  
(b) Entropy change

![](images/96df3987a1182093c9a22448ba63d1f833b448420911dd1cdba8fc4a8839b4f2.jpg)  
(c) Student uncertainty

![](images/12bc634e5c63c88e923e0fde90f0d4cb8ac102d7885c5cc45386a76ee8cb5e0b.jpg)  
(d) Teacher confidence  
Figure 5: Routing into unreliable prefixes, then fitting them. Weak Qwen: (a,b) cumulative routing, drift, and total changes in local loss and entropy, each on its own scale. The vertical guide at step 20 marks the approximate onset-to-fitting transition; the effects overlap. (c,d) Student surprisal and teacher top-1 probability; darker means greater uncertainty. Columns are 512-token windows. Details are in Section E.1.

Generally, given a trajectory-determined terminal readout function Y, set $J _ { Y } = \mathbb { E } _ { \Pi _ { \star } } Y$ and use the student’s continuation values $Q _ { t } ^ { Y } ( x , a ) = \mathbb { E } _ { t } [ Y \mid x , a ]$ and $V _ { t } ^ { Y } ( x ) = \mathbb { E } _ { a \sim \pi _ { t } } \bar { Q } _ { t } ^ { Y } ( x , a )$ . We obtain:

$$
\begin{array} { r l } & { J _ { Y } ( T ) - J _ { Y } ( t _ { \star } ) = \displaystyle \int _ { t _ { \star } } ^ { T } \sum _ { x } d _ { t } ( x ) \mathbb { E } _ { a \sim \pi _ { t } } \left[ \zeta _ { t } ( x , a ) \underbrace { ( Q _ { t } ^ { Y } ( x , a ) - V _ { t } ^ { Y } ( x ) ) } _ { \mathrm { c o n t i m u a t i o n ~ a d v a n t a g e } } \right] d t } \\ & { \quad \quad \quad = \displaystyle \frac { 1 } { \tau } \int _ { t _ { \star } } ^ { T } \sum _ { x , s } d _ { t } ( x ) [ \pi _ { t } \odot ( Q _ { t } ^ { Y } - V _ { t } ^ { Y } ) ] ( x ) ^ { \top } K _ { t } ( x , s ) \Gamma _ { t } ( s ) d t , } \end{array}\tag{9}
$$

where $d _ { t } ( x )$ is the probability of reaching x. This weights motion by what the student can achieve. When the favored alternatives have similarly unsuccessful continuations, fitting their next-token distributions can yield little accuracy progress overall (see Section C.8 for the proof).

Weak 1.5B Qwen: routing into unreliable reasoning, then fitting uncertain supervision. In the early phase of training, the teacher pushes the weak learner toward longer rollouts (such as heavy negative requests on the student’s “overly early” answer commitments), but these extensions become high-entropy and incoherent. The teacher supervision there also becomes much less concentrated than in the shallower rollout windows (Figure 5c,d). As seen in Figure 5(a) and (b), from step 1 to 20, loss rises $0 . 5 8 \to 0 . 8 5$ (routing +0.455 outweighs drift −0.187); entropy also rises mainly through routing. Intuitively, the student, which has never been trained to emit long reasoning rollouts, gets pushed into reasoning for long before committing an answer, and executing reasoning states which it is highly unfamiliar with. The teacher, conditioned on these low-quality states, also give much less confident guidance. The drift of the loss is small in this phase, however: the student has not learned the flatter prediction patterns of the teacher (uniformly) across the visited prefixes yet.

What happens after the initial routing-dominated phase is dominated by drift (local fitting) on the unreliable request distributions. As seen in Figure 5, entropy remains high within the band around 2 − 4 nats, and loss reduction is increasingly driven by drift, that is, the student is indeed fitting the teacher’s low confidence predictions on the chaotic, low-quality prefixes continually generated during training. To further test this, we find that, on the same batch-41 training prefixes, loss falls $0 . 7 9  0 . 0 6$ while entropy falls 4.99 → 3.99: drift over the course of training actually broadcasts and sharpens predictions on this rollout bank toward an uncertain but slightly less entropic teacher (Sections C.9 and E.3.1).

Policy motion and the shifting value of continuations. So far we see that the weak Qwen model’s failed run has the OPD update loop spending later, more “stable” drift-dominated updates learning from unreliable requests on the low-quality generative distribution. This helps us better understand how the training loss declines. Intuitively, fitting unreliable teacher supervision on high-entropy rollouts already suggests that this will not yield a better-reasoning student model. To understand how and why this process worsens terminal score more precisely, we need to turn to Equation (9): policy motion influences non-local readouts (including terminal score) through the value / advantage of the student’s continuations (namely the $Q - V$ advantage summed across the prefixes), which can be in conflict with local teacher agreement.

While it is prohibitively expensive to directly compute the “velocity×advantage” term in Equation (9), we find on select banks of probing rollouts and continuations that as training proceeds, the model increasingly make local decisions whose continuation correctness decline, and complementarily, visit prefixes that are difficult to reach the right answer at increasingly shallower depth. This tells us that the per-token policy velocity of the model tend to align with “negative advantage” over the course of training, resulting in the overall decline of terminal accuracy. We elaborate on these studies below.

First, to probe the continuation-value term in Equation (9), we freeze a set of answer-decision prefixes from the initial student’s held-out MATH rollouts and intervene on the local “answerversus-continue” decision token position. The student’s probability of the answer branch falls from $0 . 7 4  0 . 1 9  0 . 0 7$ at steps 0, 20, 100, below the teacher’s 0.30 on the same prefixes. At update 20, while $^ { \circ } \mathrm { r o u t i n g } ^ { \prime \prime }$ is dominating, counterfactual continuations from the two branches have similar downstream value (correct 0.30 for the “answer” branch and 0.29 for the “continue” branch), i.e. the student still maintains reasonable answer-reaching abilities. By update 100, however, the continuebranch rarely produces the answer (8% of continuations). The teacher’s own continuations from the same prefixes is correct 0.60 of the time, which is much higher, so the gap is in the student’s degraded execution of the action branch, not in the branch itself. For a complete treatment of how we decompose and estimate Equation (9) see Section E.4.

Now, these probes fix the prefixes and probe how the student checkpoints continues them. As we discussed before, the broadcast mechanism changes both prefix visitation and continuations from them, therefore we next ask the complementary question of whether the prefixes visited during OPD also become increasingly harder to complete. We fix a few rollout depths and student checkpoints, sample prefixes generated from these checkpoints at those depths, and then let the initial student (as a reference policy) to finish. Keeping the continuer fixed makes changes in success reflect changes in prefix visitation. To understand the principled connections between these reference-continuer effects and Equation (9), please refer to Section E.7. Empirically, Figure 6 shows a clear increase in continuation difficulty due to the prefix visitation shift, on the axes of rollout depths and training length. At step 50, while the shallow prefixes remains usable (even slightly better than the initial policy’s own prefixes), it already yields less successful continuations on deep prefixes. As training goes on, the decline show up at increasingly shallower cuts of the rollout.

![](images/747bc761ae450d1f560593e5e202af508625bd3eadf339f29bf500403805c5d2.jpg)  
Figure 6: Initial-policy continuation success rate on trained prefixes. Negative values indicate harder prefixes for reaching the right answer.

DeepSeek: stable learning acquiring epistemic behaviors. DeepSeek supplies the clean contrast. Loss falls $0 . 2 6 3  0 . 0 1 5 ,$ mainly through drift (−0.214) rather than routing (−0.033), indicating more more stable prefix visitations and local learning of requests. Furthermore, motivated by the epistemic-control request distribution found in Section 3.2, we design a local readout to examine the dynamics of how the model acquires intermediate reasoning behaviors, focusing on “wait”-family initiation at sentence starts (observed to initiate important backtracking and checking behaviors in R1-styled models (Marjanovic et al., 2026)). On each prefix, let $w _ { t }$ and $q _ { W }$ denote student and teacher WAIT-family probabilities. We freeze two classes: over-use $( w _ { 0 } > q _ { W } )$ and under-use $( w _ { 0 } < q _ { W } )$ , and set the local readout to be $\log [ w _ { t } / ( 1 - w _ { t } ) ]$

![](images/2efeaff433bcb490868f079487425bfe53b2b46235617c08d17d7e0af9bdd79d.jpg)  
Figure 7: Conditional checking refinement. Filled circles: initial over-use; open triangles: initial under-use.

Figure 7 plots cumulative routing and drift of the “wait”-readout. Filled circles move downward: drift suppresses initial over-use. Open triangles move upward: drift promotes initial under-use.

The corresponding net changes are −1.82 and +1.92 nats. While there are short overshoots of routing-dominated changes (early false suppression of under-use positions and promotion of overuse positions), they are quickly stabilized. Moreover, teacher-relative disagreement also decreases on frozen and held-out rollout banks (Section E.2). Thus, unlike the failed learner’s case where drift-dominated reduction of training loss happened on unreliable prefixes, DeepSeek’s stabler driftdominated reduction on training loss not only improves on its terminal score, but is accompanied by improvements in intermediate reasoning behaviors, with it sustaining prefix populations that allows to learn when to initiate a “wait”-based check as a representative example.

Preparation permits a different trajectory. The base Qwen model, prepared with offline distillation training on teacher rollouts, also undergoes substantial prefix visitation shifts, but sustains useful learning. Unlike DeepSeek’s smoother refinement, its loss and checking behavior are nonmonotone: a routing-dominated overshoot in loss and reasoning (epistemic) behaviors are followed by lowerentropy rollouts and conditional moderation of reasoning behaviors dominated by drift. Its held-out gains suggest that a stronger starting policy with higher prefix quality and continuation-readiness makes these excursions amenable to correction. Full analysis in Section E.5.

## 3.4 PREPARATION: IMPROVING THE STARTING POINT OF ON-POLICY LEARNING

![](images/5c5cf1446245c8712be18083aaec3bf0be373700bd4195f98848b54bd595edf8.jpg)  
Figure 8: Offline-prepared OPD improves sampled coverage and reliability. Left: pooled AIME 2024/2025 pass@k for $k \leq 3 2$ (top) and the fraction of problems with empirical solve rate $\geq r > 0$ (bottom), using 32 responses per problem from batch a. Right: preparation dose versus strict MATH-500 pass@1, using the step-100 OPD checkpoint at 1.5B and the best evaluated checkpoint within 100 steps at 7B; error bars are 95% problem-bootstrap intervals. See Sections F.1 and F.2 for protocols and paired comparisons.

Section 3.3 suggests a direct intervention. As we saw in the last sub-section, OPD can fail because early policy changes route the weak student model into reasoning states it cannot yet execute reliably, and the training run spends a large chunk of time fitting unreliable requests on increasingly lowerquality prefixes. Then, a natural candidate intervention is that, before OPD, we force the student to visit higher-quality prefixes from the teacher itself and learn the teacher’s stronger continuations on those prefix populations. This suggests offline distillation to prepare the student model’s continuation abilities before running OPD, which can then expose finer student errors to the teacher and sustain more stable learning trajectory for online corrections when OPD changes prefix visitation, without the requests looking so foreign to the student in the OPD phase that it routes into irrecoverable rollout distributions.

We prepare with student/teacher pairings Qwen2.5-1.5B-Instruct/JustRL-1.5B (Qwen et al., 2025; He et al., 2025), Qwen2.5-7B-Instruct/AceReason-Nemotron-7B (Chen et al., 2025), and Llama-3.1- 8B-Instruct/Llama-3.1-Nemotron-Nano-8B-v1 (Llama Team, 2024; Bercovich et al., 2025), with students trained on their teachers’ solutions to OpenMathInstruct-2 (Toshniwal et al., 2024). We then compare OPD with continued offline distillation on the same DAPO-Math-17k prompt blocks (Yu et al., 2025), matching optimizer steps and sequence presentations.

Figure 8 shows higher pooled pass@k accuracy for OPD than offline training through k = 32, using 32 responses per problem. The gains extend beyond pass@1 to sampled coverage and the distribution of per-problem solve rates. The reliability improvement is especially interesting: looking at pass@k alone might give us the impression that OPD just solves more problems by rare chances, however, we find that OPD with appropriate preparation also enhances the per-problem solve rates for the models beyond the offline-distillation-only alternatives. Additionally, the preparation “dose study” (Figure 8, right) shows increasing value of offline-preparation for the later OPD phase. At 1.5B, one to four passes leaves the starting strict MATH pass@1 at 45.7–47.1%, but raises the 100-step OPD endpoint from 45.9% to 56.2%. Similar trends are observed at 7B (Section F.2). Please refer to Section F.1 for further details.

## 4 CONCLUSION

Findings. We studied on-policy distillation as a feedback loop between teacher supervision, student response, and the rollout distribution that supplies future supervision. Our analysis separates two mechanisms in this loop: conversion, and broadcast. Across our settings, these mechanisms explain both stalled transfer and deterioration despite improving local teacher agreement. Offline preparation changes the starting point of this loop: even with little immediate accuracy improvement, it can enable substantial gains from subsequent OPD.

Limitations and discussion. Our empirical analysis is confined to mathematical reasoning and a select set of LLM models due to resource limitations. We also present a more detailed Q&A in Appendix A to further clarify potential ambiguities in our work.

## AI USE STATEMENT

Generative AI tools were also used for language editing, rewriting and presentation. We used generative AI tools to assist with code implementation, debugging, experimental scripting, and parts of experiment execution. The research questions, theoretical development, experimental design, interpretation, and scientific conclusions were determined by the authors. All AI-assisted code and text were reviewed and validated by the authors. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## REPRODUCIBILITY STATEMENT

Precise experimental details are provided in the appendix, including model specifications, training pipeline and evaluation details.

## REFERENCES

Rishabh Agarwal, Nino Vieillard, Yongchao Zhou, Piotr Stanczyk, Sabela Ramos, Matthieu Geist, and Olivier Bachem. On-policy distillation of language models: Learning from self-generated mistakes. In International Conference on Learning Representations, 2024. URL https:// arxiv.org/abs/2306.13649.

Zeyuan Allen-Zhu and Yuanzhi Li. Feature purification: How adversarial training performs robust deep learning. In 2021 IEEE 62nd Annual Symposium on Foundations of Computer Science (FOCS), pp. 977–988, 2022. doi: 10.1109/FOCS52979.2021.00098.

Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning, 2023a. URL https://arxiv.org/abs/2012.09816.

Zeyuan Allen-Zhu and Yuanzhi Li. Backward feature correction: How deep learning performs deep (hierarchical) learning. In Gergely Neu and Lorenzo Rosasco (eds.), Proceedings of Thirty Sixth Conference on Learning Theory, volume 195 of Proceedings of Machine Learning Research, pp. 4598–4598. PMLR, 12–15 Jul 2023b. URL https://proceedings.mlr.press/v195/ allen-zhu23a.html.

Mohammadreza Armandpour, Fatih Ilhan, David Harrison, Ajay Jaiswal, Duc N. M Hoang, Fartash Faghri, Yizhe Zhang, Minsik Cho, and Mehrdad Farajtabar. Unmasking on-policy distillation: Where it helps, where it hurts, and why, 2026. URL https://arxiv.org/abs/2605. 10889.

Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang. Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 322–332. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/arora19a.html.

Alexander Atanasov, Blake Bordelon, and Cengiz Pehlevan. Neural networks as kernel learners: The silent alignment effect. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=1NvflqAdoom.

David D. Baek and Max Tegmark. Towards understanding distilled reasoning models: A representational approach, 2025. URL https://arxiv.org/abs/2503.03730.

Aristide Baratin, Thomas George, César Laurent, R Devon Hjelm, Guillaume Lajoie, Pascal Vincent, and Simon Lacoste-Julien. Implicit regularization via neural feature alignment. In Arindam Banerjee and Kenji Fukumizu (eds.), Proceedings ofThe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pp. 2269–2277. PMLR, 13–15 Apr 2021. URL https://proceedings.mlr.press/v130/ baratin21a.html.

Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, and Soumye Singhal. Llama-nemotron: Efficient reasoning models, 2025. URL https://arxiv.org/abs/2505.00949.

M Biehl and H Schwarze. Learning by on-line gradient descent. Journal ofPhysics A: Mathematical and General, 28(3):643, feb 1995. doi: 10.1088/0305-4470/28/3/018. URL https://doi. org/10.1088/0305-4470/28/3/018.

Paul C. Bogdan, Uzay Macar, Neel Nanda, and Arthur Conmy. Thought anchors: Which llm reasoning steps matter?, 2025. URL https://arxiv.org/abs/2506.19143.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Yang Chen, Zhuolin Yang, Zihan Liu, Chankyu Lee, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. Acereason-nemotron: Advancing math and code reasoning through reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=EgArbnS0BA.

DeepSeek-AI, Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chengyu Hou, Chenhao Xu, Chenze Shao, Chong Ruan, Conner Sun, Damai Dai, Daya Guo, and Dejian Yang. Deepseek-v4: Towards highly efficient million-token context intelligence, 2026. URL https://arxiv.org/abs/2606.19348.

Yuqian Fu, Haohuan Huang, Kaiwen Jiang, Jiacai Liu, Zhuo Jiang, Yuanheng Zhu, and Dongbin Zhao. Revisiting on-policy distillation: Empirical failure modes and simple fixes, 2026. URL https://arxiv.org/abs/2603.25562.

Sebastian Goldt, Madhu Advani, Andrew Saxe, Florent Krzakala, and Lenka Zdeborová. Dynamics of stochastic gradient descent for two-layer neural networks in the teacher-student setup. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2019/file/cab070d53bd0d200746fb852a922064a-Paper.pdf.

Yuxian Gu, Li Dong, Furu Wei, and Minlie Huang. MiniLLM: Knowledge distillation of large language models. In International Conference on Learning Representations, 2024. URL https: //openreview.net/forum?id=5h0qf7IBZZ.

Bingxiang He, Zekai Qu, Zeyuan Liu, Yinghao Chen, Yuxin Zuo, Cheng Qian, Kaiyan Zhang, Weize Chen, Chaojun Xiao, Ganqu Cui, Ning Ding, and Zhiyuan Liu. Justrl: Scaling a 1.5b llm with a simple rl recipe, 2025. URL https://arxiv.org/abs/2512.16649.

Bobby He and Mete Ozay. Feature kernel distillation. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum?id=tBIQEvApZK5.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015. URL https://arxiv.org/abs/1503.02531.

Guan Zhe Hong, Yin Cui, Ariel Fuxman, Stanley H. Chan, and Enming Luo. Why fine-grained labels in pretraining benefit generalization? Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=FojAV72owK.

Guan Zhe Hong, Nishanth Dikkala, Enming Luo, Cyrus Rashtchian, Xin Wang, and Rina Panigrahy. A implies b: Circuit analysis in llms for propositional logical reasoning. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, Main Conference, pp. 30749–30803. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1033. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/2c4477572f06102b2ae3bd2b2ffcebf9-Paper-Conference.pdf.

Guanzhe Hong, Zhiyuan Mao, Xiaojun Lin, and Stanley H. Chan. Student-teacher learning from clean inputs to noisy inputs. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 12075–12084, June 2021.

Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization in neural networks. In Advances in Neural Information Processing Systems (NeurIPS), 2018.

Guangda Ji and Zhanxing Zhu. Knowledge distillation in wide neural networks: Risk bound, data efficiency and imperfect teacher. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 20823–20833. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/file/ef0d3930a7b6c95bd2b32ed45989c61f-Paper.pdf.

Woogyeol Jin, Taywon Min, Yongjin Yang, Swanand Ravindra Kadhe, Yi Zhou, Dennis Wei, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models, 2026a. URL https://arxiv.org/abs/2603.07079.

Woogyeol Jin, Taywon Min, Yongjin Yang, Dennis Wei, Yi Zhou, Swanand Ravindra Kadhe, Nathalie Baracaldo, and Kimin Lee. Entropy-aware on-policy distillation of language models. In Forty-third International Conference on Machine Learning, 2026b. URL https://openreview.net/ forum?id=J5i09faOOf.

Yoon Kim and Alexander M. Rush. Sequence-level knowledge distillation. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pp. 1317–1327. Association for Computational Linguistics, 2016. doi: 10.18653/v1/D16-1139. URL https: //aclanthology.org/D16-1139/.

Gengsheng Li, Mao Zheng, Mingyang Song, Ruiqi Liu, Tianyu Yang, Jie Sun, Qiyong Zhong, Haiyun Guo, Junfeng Fang, Dan Zhang, and Jinqiao Wang. On-policy distillation with curriculum turn-level guidance for multi-turn agents, 2026a. URL https://arxiv.org/abs/2606.15912.

Yaxuan Li, Yuxin Zuo, Bingxiang He, Jinqian Zhang, Chaojun Xiao, Cheng Qian, Tianyu Yu, Huanang Gao, Wenkai Yang, Zhiyuan Liu, and Ning Ding. Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe. arXiv preprint arXiv:2604.13016, 2026b.

Alexander Lin, Jeremy Wohlwend, Howard Chen, and Tao Lei. Autoregressive knowledge distillation through imitation learning. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6121–6133. Association for Computational Linguistics, 2020. doi: 10.18653/v1/2020.emnlp-main.494. URL https://aclanthology.org/2020. emnlp-main.494/.

Tao Liu, Taiqiang Wu, Mao Zheng, Xuan Luo, Runming Yang, Xuewei Yang, Junjie Wang, and Yujiu Yang. Simple-OPD: Demystifying warm-up for on-policy distillation, 2026. URL https: //arxiv.org/abs/2608.06802.

Meta Llama Team. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407. 21783.

Kevin Lu and Thinking Machines Lab. On-policy distillation. Thinking Machines Lab: Connectionism, 2025. doi: 10.64434/tml.20251026. URL https://thinkingmachines.ai/blog/ on-policy-distillation/.

Sadhika Malladi, Alexander Wettig, Dingli Yu, Danqi Chen, and Sanjeev Arora. A kernel-based view of language model fine-tuning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 23610–23641. PMLR, 2023. URL https://proceedings.mlr.press/v202/malladi23a.html.

Sara Vera Marjanovic, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, Nicholas Meade, Dongchan Shin, Amirhossein Kazemnejad, Gaurav Kamath, Marius Mosbach, Karolina Stanczak, and Siva Reddy. Deepseek-r1 thoughtology: Let’s think about LLM reasoning. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/ forum?id=BZwKsiRnJI.

Ashkan Panahi, Arman Rahbar, Chiranjib Bhattacharyya, Devdatt Dubhashi, and Morteza Haghir Chehreghani. Analysis of knowledge transfer in kernel regime. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management, CIKM ’22, pp. 1615–1624, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450392365. doi: 10.1145/3511808.3557237. URL https://doi.org/10.1145/ 3511808.3557237.

Mary Phuong and Christoph Lampert. Towards understanding knowledge distillation. In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 5142–5151. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/phuong19a. html.

Julianna Piskorz, Antonin Berthon, and Mihaela van der Schaar. On-policy or off-policy learning? a systematic study of distillation dynamics, 2026. URL https://arxiv.org/abs/2609. 35259.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL https://arxiv.org/abs/2412.15115.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model, 2024. URL https://arxiv.org/abs/2305.18290.

Yi Ren and Danica J. Sutherland. Learning dynamics of LLM finetuning. In Proceedings of the International Conference on Learning Representations (ICLR), 2025. arXiv:2407.10490.

Stephane Ross and J. Andrew Bagnell. Reinforcement and imitation learning via interactive no-regret learning, 2014. URL https://arxiv.org/abs/1406.5979.

Stéphane Ross, Geoffrey J. Gordon, and J. Andrew Bagnell. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings ofthe International Conference on Artificial Intelligence and Statistics (AISTATS), 2011.

David Saad and Sara A. Solla. Exact solution for on-line learning in multilayer neural networks. Phys. Rev. Lett., 74:4337–4340, May 1995. doi: 10.1103/PhysRevLett.74.4337. URL https: //link.aps.org/doi/10.1103/PhysRevLett.74.4337.

Andrew M. Saxe, James L. McClelland, and Surya Ganguli. Exact solutions to the nonlinear dynamics of learning in deep linear neural networks, 2014. URL https://arxiv.org/abs/1312. 6120.

Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, and Yi R. Fung. On the geometry of on-policy distillation, 2026. URL https://arxiv.org/abs/2606.07082.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models. arXiv preprint arXiv:2604.00626, 2026a.

Mingyang Song and Mao Zheng. A survey of on-policy distillation for large language models, 2026b. URL https://arxiv.org/abs/2604.00626.

Richard S. Sutton, David A. McAllester, Satinder P. Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Advances in Neural Information Processing Systems, volume 12, 1999.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, M. C., Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y. Charles, H. S. Che, Guanduo Chen, Guangyu Chen, Guanzheng Chen, Huarong Chen, Jia Chen, Jianlong Chen, Jun Chen, Kexin Chen, Peng Chen, Ruijue Chen, Wentao Chen, Xin Chen, Yang Chen, Yanru Chen, Yifei Chen, Yingjiang Chen, and Yuankun Chen. Kimi k3: Open frontier intelligence, 2026. URL https://arxiv.org/abs/2607.24653.

Akiyoshi Tomihari. Learning dynamics in RL post-training for language models. arXiv preprint arXiv:2601.04670, 2026.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. OpenMathInstruct-2: Accelerating AI for math with massive open-source instruction data, 2024. URL https://arxiv.org/abs/2410.01560.

Dmitrii Troitskii, Koyena Pal, Chris Wendler, and Callum Stuart McDougall. Internal states before wait modulate reasoning patterns. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 18640–18649, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-335-7. doi: 10.18653/v1/2025.findings-emnlp.1012. URL https://aclanthology.org/2025.findings-emnlp.1012/.

Shinichi Uemura and Taiji Suzuki. Learning without overwriting: A theory of self-distillation and supervised fine-tuning in continual reasoning, 2026. URL https://arxiv.org/abs/2610. 05200.

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. Base models know how to reason, thinking models learn when, 2025a. URL https://arxiv.org/abs/2510. 07364.

Constantin Venhoff, Iván Arcuschin, Philip Torr, Arthur Conmy, and Neel Nanda. Understanding reasoning in thinking language models via steering vectors, 2025b. URL https://arxiv. org/abs/2506.18167.

Rui Wang, Hongru Wang, Yi Chen, Boyang Xue, Tianqing Fang, Wenhao Yu, and Kam-Fai Wong. Demystifying on-policy distillation: Roles, pathologies, and regulations. arXiv preprint arXiv:2607.13399, 2026.

Jake Ward, Chuqiao Lin, Constantin Venhoff, and Neel Nanda. Reasoning-finetuning repurposes latent representations in base models, 2025. URL https://arxiv.org/abs/2507.12638.

Wenkai Yang, Weijie Liu, Ruobing Xie, Kai Yang, Saiyong Yang, and Yankai Lin. Learning beyond teacher: Generalized on-policy distillation with reward extrapolation. arXiv preprint arXiv:2602.12125, 2026. URL https://arxiv.org/abs/2602.12125.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Mu Qiao, Yonghui Wu, and Mingxuan Wang. DAPO: An open-source LLM reinforcement learning system at scale, 2025. URL https://arxiv.org/ abs/2503.14476.

Zichao Yu, Chengzhi Yu, Shengze Xu, Yujin Han, Bingqing Jiang, Xu Wang, and Difan Zou. Mismatch matters: On-policy distillation beyond token agreement, 2026. URL https:// arxiv.org/abs/2608.09836.

Yaocheng Zhang, Jiajun Chai, Yuqian Fu, Songjun Tu, Xiaohan Wang, Wei Lin, Guojun Yin, Qichao Zhang, Yuanheng Zhu, and Dongbin Zhao. Are full rollouts necessary for on-policy distillation?, 2026. URL https://arxiv.org/abs/2605.31490.

Siqi Zhu, Xuyan Ye, Hongyu Lu, Weiye Shi, and Ge Liu. The many faces of on-policy distillation: Pitfalls, mechanisms, and fixes, 2026. URL https://arxiv.org/abs/2605.11182.

Zhou Ziheng, Jiaqi Li, Huacong Tang, Ying Nian Wu, and Demetri Terzopoulos. Less is more: Early stopping rollout for on-policy distillation, 2026. URL https://arxiv.org/abs/2605. 27028.

## A MECHANISTIC INTERPRETATION AND KEY QUESTIONS

## Q1. What do the response law and decompositions in Sections 3.1 to 3.3 explain beyond their familiar mathematical ingredients?

They provide a connected mechanistic account of how dense local supervision can produce either weak transfer or deteriorating generation. OPD updates the policy that generates its next training population. Understanding its dynamics therefore requires following supervision beyond the current loss, through its effects on future behavior and training histories (student’s rollouts and the induced supervision signals / request field).

The theory separates three links in this process. First, Section 3.2 explains how a teacher’s requests become parameter motion: accessibility determines their expression through the student’s tangent map, while coherence determines how the resulting gradients combine. Teacher benchmark performance, magnitude of (KL-based) disagreement, or gradient norm alone cannot characterize this conversion. Second, Section 3.3 distinguishes changing a prediction from changing the ancestral decisions that make its history likely. Shared-parameter broadcast can increase exposure to a continuation without making the student better at executing it; the resulting histories then change the supervision received from the same fixed teacher. Third, Equation (9) connects the delivered response to completed outcomes through what the student’s remaining policy can achieve, separating teacher matching from useful reasoning.

Together, these links explain the structure of the empirical puzzles. The stronger-teacher study exposes the importance of gradient coherence and tangent accessibility; the degenerate run exposes that, under substantial policy motion, the degree of freedom granted to the learner to generate its own rollouts (training histories) can make it generate the new rollouts (reasoning states) which it has difficulty managing and grows uncertain on, inducing poorer-quality supervision signals. Loss reduction dominated by local fitting on these poorly supervised rollouts cannot improve the learner’s terminal scores. The mechanisms revealed in the degenerate run motivates preparing the student for the continuations that OPD will elicit, even when preparation does not substantially improve current accuracy by itself. The theoretical contribution is the integrated explanation that makes these mechanisms distinguishable, connects them to outcomes, and motivates an intervention that boosts the model’s learning stability and generalization.

## Q2. Why is every training run reported with a single seed?

Cost. The runs behind this paper were trained and evaluated on GH200 nodes, and the single-seed set already consumed roughly 4,000 GPU-hours, which is at the limit of our computing resources. OPD is known to be computationally demanding, we therefore designed the experiments so that the main comparisons do not rest on seed variance. The mechanism measurements of Sections 3.2 and 3.3 are exact identities evaluated within a run, on fixed receiver banks and fixed checkpoints, and their uncertainty is reported at the level of problems or sites (paired bootstrap intervals). The outcome comparisons of Section 3.4 are paired within a student-teacher pair on the same problems and replicated across three pairs and two student scales. What single seeds leave unquantified is run-to-run variation of the training trajectories themselves.

## Q3. Why does Section 3.3 need both prefix-level routing/drift (Equation (5)) and trajectory-level continuation advantages (Equation (9)) to explain degeneration?

For loss or entropy, $J _ { Y } = \mathbb { E } _ { \nu _ { t } } Y _ { t } ( x )$ describes predictions on the currently observed histories. Crossscoring separates changes in those histories from changes in predictions on the same histories. In Figure 5, the initial student is already highly uncertain on the deep histories that later become prevalent, while subsequent fitting sharpens them. On identical batch-41 histories, discrepancy falls $0 . 7 9  0 . 0 6$ and entropy $4 . 9 9  3 . 9 9$ . Together with the stopping/reach probes, these measurements distinguish routing followed by fitting from uniform entropy-increasing drift (Sections E.3 and E.3.1).

But fitting a prefix and improving the completion from it are different effects. For terminal correctness, $J _ { Y } \ = \ \mathbb { E } _ { \Pi _ { t } } Y$ , and the conditional value $V _ { t } ^ { Y } ( x )$ depends on every downstream decision. At an intermediate cut of the autoregressive trace, its derivative separates changes in reaching the cut from changes in the remaining policy. Unrolling the latter gives the continuation-advantage weighting in Equation (9); replacing it with a derivative of the logits at x alone would omit downstream effects (Section C.8). Thus, a teacher-preferred next token need not help if the student cannot complete the reasoning it initiates. The prefix analysis establishes the observed routing-then-fitting pattern; the outcome analysis explains why local fitting need not restore correctness. Entropy enters the argument only as a prefix-level readout.

## Q4. Could the responsiveness changes in Section 3.2 (Figure 4) simply reflect changing teacher– student disagreement or generic kernel movement?

A live accessibility measure $A _ { t } [ B _ { t } ]$ can change because either the requests or the kernel change. We therefore fix the initial numerical requests $B _ { 0 } .$ , exact histories, and output coordinates across checkpoints. Since only the kernel varies in $A _ { t } [ B _ { 0 } ]$ , its change measures responsiveness to the same supervision (Section C.4).

To test whether eigenspace movement reflects teacher-specific adaptation, we evaluate both teachers frozen fields in the same checkpoint kernel and compare their projection gains, separating preferential reorientation from shared movement that benefits both fields. Healthy 1.5B exhibits preferential reorientation; healthy 7B mainly amplifies existing directions; stalled runs lose sensitivity to the fixed requests (Section D.5). These findings complement the initial accessibility/coherence decomposition, which exposes conversion differences concealed by comparable gradient norms across the runs.

## Q5. What does preparation in Section 3.4 and Figure 8 contribute beyond offline warm-up, and how does Section 3.3 motivate it?

The contribution is an explanation of why the value of a warm-up stage, already used in prior work (Section B), should be assessed through subsequent learning. The failure mechanism identifies a mismatch between making extended reasoning likely and being able to execute it. Offline distillation supplies teacher-generated reasoning histories without requiring the initial student to generate them first. It can therefore prepare continuation skills and reasoning-state management that become useful as OPD changes visitation: continuation readiness, a benefit beyond higher starting accuracy – our pass@k results clearly demonstrate this.

## B EXPANDED RELATED WORK

## B.1 CORE RELATED WORKS

Distillation and the OPD loop. Knowledge distillation transfers teacher outputs (Hinton et al., 2015), including teacher-generated sequences (Kim & Rush, 2016). DAgger, the seminal online imitation algorithm, addresses distribution shift through expert feedback on learner-visited states (Ross et al., 2011); Lin et al. (2020) develop this connection for autoregressive distillation. MiniLLM uses reverse KL and student-generated samples (Gu et al., 2024), (Agarwal et al., 2024; Piskorz et al., 2026) study the choice of divergence loss, and Lu & Thinking Machines Lab (2025) demonstrate practical OPD for reasoning and personalization. These methods motivate our question: how does dense supervision on student histories produce successful transfer, stagnation, or degeneration? We develop a systematic first-principles account of the coupled loop through which shared-parameter updates change predictions, renew visited histories, and reshape subsequent supervision.

Kernel response laws. The neural tangent kernel connects parameter learning to function-space change (Jacot et al., 2018). Malladi et al. (2023) analyze language-model finetuning through kernels, including an Adam extension; Ren & Sutherland (2025) decompose prediction changes through normalization, empirical-kernel, and loss-gradient terms for SFT and preference finetuning, including offline DPO. Tomihari (2026) studies update propagation in RL post-training. OPD needs different theoretical and empirical machinery for two reasons. Its on-policy sampling moves the learning distribution itself, so the analysis must track this motion, which is absent from the offline algorithm (SFT and offline DPO) studied by Ren & Sutherland (2025). Its supervision is also dense and token-level and can be disconnected from the reasoning outcome (such as accuracy), whereas RL gives only sparse, outcome-level learning signals.

Nevertheless, the fundamental response law of the shared-parameter neural learner is shared with Ren & Sutherland (2025). We then follow the induced motion through the rest of the OPD loop: changes at ancestor decisions alter reach probabilities, renew the training source, and thereby change the requests driving future updates. Combining the evolving kernel, routing/drift split, and continuation-value analysis yields a mechanistic account of how optimization changes future supervision and outcomes (Sections 3 and C).

Cumulative weight geometry vs the geometry of request conversion. Shen et al. (2026) find low stable rank and early stabilization of dominant subspaces in cumulative weight-matrix changes; constrained-training experiments test those subspaces’ functional sufficiency. We instead analyze the current logit Jacobian and the response it delivers to specified teacher requests. The Jacobian can change even when cumulative displacement is low-rank. Frozen histories, coordinates, and requests isolate amplification and reorientation; crossed teacher fields test preferential adaptation in the same kernel (Sections C.4 and D.5). Distinguishing local accessibility from collective coherence further explains why individually expressible requests can cancel. These analyses concern how geometry enables or obstructs transfer.

Local fitting vs task performance. Policy-gradient theory (Sutton et al., 1999) and cost-to-go-aware imitation learning (Ross & Bagnell, 2014) connect actions to subsequent outcomes. G-OPD interprets distillation through KL-constrained RL and reward extrapolation (Yang et al., 2026). Armandpour et al. (2026) use student continuation-success probabilities to derive an ideal local logit direction and measure its alignment with distillation signals (however, they acknowledge that such estimations in long-CoT reasoning can be very noisy). Uemura & Suzuki (2026) theoretically studies the dynamics of self-distillation for a single-layer neural network, and analyzes how the supervision affects longhorizon traversal of directed acyclic graphs, invoking tools such as future advantage in their analysis. Our analysis connects this type of outcome perspective to delivered motion: $\begin{array} { r } { \sum _ { s } K _ { t } ( x , s ) \dot { \Gamma _ { t } } ( s ) } \end{array}$ includes the effects of all coupled source prefixes in addition to the request at x. Crucially, prefix readouts $\mathbb { E } _ { \nu _ { t } } Y _ { t } ( x )$ and terminal outcomes $\dot { \mathbb { E } } _ { \Pi _ { t } } Y$ are different objects. At an intermediate cut-position in the autoregressive rollout, terminal change splits into the change in reaching that prefix and the change in the continuations; unrolling the latter and recombining both terms yields the advantageweighted contribution of every decision (Equation (9) and Section C.8). This separates reaching a prefix, fitting there, and completing it successfully –the distinction underlying the degeneration mechanism and continuation readiness.

Degeneration involves source feedback as well as supervision quality. Fu et al. (2026) study unreliable supervision and implementation choices; Jin et al. (2026a) augment high-entropy teacher supervision with forward KL; and Zhu et al. (2026) analyze signals induced by top-K truncation, renormalization, and stop-gradient conventions. Li et al. (2026b) also observe deterioration with prefix depth. Contemporaneous work by Wang et al. (2026) studies teacher mismatch and length exploitation. Our first-principled analysis reveals a phased failure mechanism: early routing makes already-uncertain continuations prevalent, after which substantial fitting can sharpen those very histories without restoring useful generation (Sections E.3 and E.3.1). Our routing-drift machinery, coupled with the learning-value decomposition (for terminal-accuracy readout) distinguish the failure mechanism from a naive, uniform entropy-increasing learning process. The important feedback is that the histories made common by earlier updates become the source of later supervision. The prepared learner’s different trajectory further shows why source movement itself is not synonymous with failure, and its continuation-readiness helps in exposing useful subsequent learning.

Offline preparation can improve readiness for learning without improving immediate accuracy. Li et al. (2026b) motivate cold-start SFT by reducing thinking-pattern mismatch and improving overlap. Their §5.1/Figure 8 studies a strong standalone-SFT regime: the SFT initialization already exceeds the direct-OPD endpoint on all three plotted benchmarks, with little further gains from OPD. We study a different regime, where the student-teacher gap is significant, and preparation leaves starting accuracy near the unprepared student’s but changes subsequent OPD from deterioration to substantial improvement. The principled motivation is continuation readiness: the outcome analysis and observed failure mechanisms in Section 3.3 identify a distinction between increasing a history’s reach and executing its continuation successfully. Teacher-generated trajectories provide training on such histories before the student must reach them itself. Our dose study exposes deferred benefits and a sharply nonmonotonic stability pattern at 7B; the prepared-run analysis traces a different subsequent routing/fitting process (Table 14 and Section E.5).

In contemporaneous work Simple-OPD (Liu et al., 2026) studies warm-up data, duration, and LoRA parameterization, including benefits from incorrect teacher-compatible traces and differences between SFT and downstream rankings. These findings complement our mechanistic account linking preparation to source feedback and continuation values. The key result is an altered learning trajectory; the preparation dose-response experiments demonstrate this interesting benefit.

## B.2 MORE RELATED WORKS

Reasoning interpretability. Part of the focus of our work is on how LLMs learn to reason (from a teacher model), and our analysis methodology is partly inspired by the discoveries in the reasoning interpretability community. Venhoff et al. (2025a) is especially relevant, as they discovered that RL-trained models tend to acquire the ability to manage “when” to fire a reasoning mechanism (be it an “epistemic control” mechanism such as backtracking or planning, or computation mechanisms like estimating an inequality); Marjanovic et al. (2026) directly studies the chain-of-thought reasoning mechanisms of DeepSeek-R1 models, which our DeepSeek student models are distilled from, also inspiring our analysis on the epistemic control transmission in OPD. More broadly, Bogdan et al. (2025); Baek & Tegmark (2025); Venhoff et al. (2025b); Troitskii et al. (2025); Hong et al. (2025); Ward et al. (2025) identify causal mechanisms behind LLMs’ reasoning behaviors in logic, mathematics and other related domains.

OPD and its variants. On-policy distillation is becoming increasingly popular in industry posttraining pipelines (DeepSeek-AI et al., 2026; Team et al., 2026). Many variants are being proposed, including selecting which tokens to perform supervision on (Jin et al., 2026b; Yu et al., 2026), and curriculum-styled variants (Ziheng et al., 2026; Zhang et al., 2026; Li et al., 2026a). The survey on OPD Song & Zheng (2026b) is also a good reference for understanding the research landscape in this area.

Learning dynamics. Many classic works lie in this space and are predecessors to recent studies of LLM learning dynamics. Biehl & Schwarze (1995); Saad & Solla (1995); Saxe et al. (2014); Goldt et al. (2019) are early representative works, studying how learning trajectories and generalization depend on the model, initialization, and training process, and providing foundations for our analysis of how teacher supervision changes the student over time. Furthermore, a particular line of work study feature learning in nonlinear neural networks: Baratin et al. (2021); Atanasov et al. (2022); Hong et al.

(2024); Allen-Zhu & Li (2022; 2023b) study how the nonlinear neural network’s geometry evolves and aligns with the task, showing feature re-orientation early on in (pre-)training that precedes appreciable loss reduction (sometimes called the “lottery winning” phase, where certain neurons or features better align towards loss-reducing directions early in training without substantial loss-reduction yet). These works are especially relevant to our analysis of the student’s changing responsiveness to teacher requests in Section 3.2, and to offline preparation whose benefits become apparent through subsequent learning. Additionally, another relevant line of work in the pre-LLM era study dynamics of DNN distillation. Phuong & Lampert (2019); Ji & Zhu (2020); Hong et al. (2021); Allen-Zhu & Li (2023a); Panahi et al. (2022); He & Ozay (2022) examine how the geometry of the learning problem, teacher knowledge, and the student’s representation affect transfer, using linear, deep linear, or kernel-based analyses of the SGD-based learning trajectory.

## C DERIVATIONS AND PROOFS FOR THE MAIN TEXT

This appendix follows Sections 2 and 3. It gives the assumptions and proofs for the main-text identities, then the specializations used to interpret the experiments.

<table><tr><td>Main-text result</td><td>Complete derivation</td></tr><tr><td>Section 2: policies and request field</td><td>Section C.1: finite-tree conventions and exact-loss request</td></tr><tr><td>Section 3.1, Equation (2)</td><td>Section C.2: exact response and explicit finite-step remainder</td></tr><tr><td>Section 3.2, Equations (3) and (4)</td><td>Sections C.3 and C.4: conversion and frozen-field geometry</td></tr><tr><td>Section 3.3, Equations (5) and (6)</td><td>Section C.5: routing and conditional drift</td></tr><tr><td>Section 3.3, Equation (7)</td><td>Section C.6: source normalization and support</td></tr><tr><td>Section 3.3, Equation (8)</td><td>Section C.7: endpoint accounting and path-integral error</td></tr><tr><td>Section 3.3, Equation (9)</td><td>Section C.8: continuation values and terminal outcomes</td></tr><tr><td>Section 3.3: entropy versus fitting</td><td>Section C.9: entropy and checking sensitivities</td></tr></table>

## C.1 SETUP: POLICIES, REQUESTS, AND STANDING ASSUMPTIONS

Connection to Section 2. We distinguish the model distribution, the sampling distribution, and the numerical field pulled back into parameter space. This distinction is needed for both the proofs and the experimental diagnostics.

Notation. A prompt is denoted by b, a source prefix by $s ,$ and a receiver prefix by x. A complete trajectory is $\xi ; \tau > 0$ is reserved for sampling temperature. The integer k indexes optimizer updates, j indexes generated-token depth, and t indexes continuous training time. All gradients of scalar functions are column vectors. The matrix $\Phi _ { \theta } ( \boldsymbol { x } ) \in \mathbb { R } ^ { | \mathcal { V } | \times d }$ has row a equal to $\nabla _ { \theta } z _ { \theta } ( x , a ) ^ { \top }$ , so $K _ { \theta } ( x , s ) = \Phi _ { \theta } ( x ) \Phi _ { \theta } ( s ) ^ { \top }$ . We write $\mathsf { A } ^ { p } = I - \mathsf { i } \overset { \cdot } { p } ^ { \intercal }$ for the log-softmax normalization matrix, A for scalar accessibility, and $\mathcal { A } ^ { Y }$ for continuation advantage. A logit-space readout gradient is denoted by $\chi \in \mathbb { R } ^ { | \nu | }$ ; its parameter-space pullback is $\psi = \Phi ^ { \top } \chi \in \mathbb { R } ^ { d }$ . Norms without a subscript are Euclidean. $\| \cdot \| _ { \mathrm { o p } }$ denotes the induced operator norm.

Assumption C.1 (Finite generation trees and regular parameter paths). The prompt set $\boldsymbol { B }$ and vocabulary V are finite, and generation has a finite cap $\bar { H } _ { \mathrm { m a x } }$ . Each prefix contains its prompt and its full generated ancestry. Appending a token is deterministic. A node is terminal after a stopping token or at the cap, and has no further decisions. The terminal rule and chat template are fixed. The prompt distribution $\beta$ is fixed (and independent of model parameters); prompts of zero probability are omitted.

The parameter path $\theta : I  \mathbb { R } ^ { d }$ , on a compact interval $I ,$ is continuous and piecewise $C ^ { 1 }$ with finitely many pieces. For every prefix under consideration, $z _ { \theta } ( x )$ is $C ^ { 2 }$ on an open neighborhood of the path and of any parameter segment used for a finite-step calculation. The model distribution is the full-vocabulary policy $p _ { \theta } = \mathrm { s o f t m a x } ( z _ { \theta } )$ . The teacher q is fixed and is positive on every coordinate used in an exact-KL calculation (which softmax-output models do satisfy).

For the sampling identities, each nonterminal prefix x has a fixed nonempty allowed vocabulary set $\mathscr { V } _ { \pi } ( x ) \subseteq \mathcal { V }$ and

$$
\pi _ { \theta } ( a \mid x ) = \frac { \exp ( z _ { \theta } ( x , a ) / \tau ) } { \sum _ { c \in \mathcal { V } _ { \pi } ( x ) } \exp ( z _ { \theta } ( x , c ) / \tau ) } \quad ( a \in \mathcal { V } _ { \pi } ( x ) ) , \qquad \pi _ { \theta } ( a \mid x ) = 0 \quad ( a \not \in \mathcal { V } _ { \pi } ( x ) ) .
$$

The temperature $\tau > 0 .$ , allowed sets, and any observation or source masks are fixed. The fullvocabulary sampler is the case $\mathscr { V } _ { \pi } ( x ) = \mathscr { V }$

Let S and T be the resulting finite sets of nonterminal and terminal prefixes. A trajectory $\xi =$ $\left( b , a _ { 0 } , \ldots , a _ { N - 1 } \right)$ ends at its first terminal node, with $N \leq H _ { \operatorname* { m a x } }$ , and has probability

$$
\Pi _ { t } ( \xi ) = \beta ( b ) \prod _ { j = 0 } ^ { N - 1 } \pi _ { t } ( a _ { j } \mid x _ { j } ) .
$$

For any node $x ,$ including a terminal node, $d _ { t } ( x )$ is its reach-probability. It is the same product truncated at depth $h ( x )$ . On terminal nodes ${ d _ { t } } ( x ) \dot { = } \Pi _ { t } ( x )$ ; on nonterminal nodes the sum $\dot { \sum } _ { x } d _ { t } ( x )$

is generally not one. Every allowed trajectory has positive probability. Finiteness and the stated regularity justify every interchange of sums and derivatives below. Identities for derivatives hold on each smooth piece, hence almost everywhere, and integrate across the entire interval. The results take the path as given and are independent of any existence or uniqueness theorem for an autonomous training flow.

The raw direction is

$$
\begin{array} { r } { g _ { t } ( s ) = \Phi _ { t } ( s ) ^ { \top } B _ { t } ( s ) , \qquad u _ { t } = \displaystyle \sum _ { s } \mu _ { t } ( s ) g _ { t } ( s ) , \qquad \Gamma _ { t } ( s ) = \mu _ { t } ( s ) B _ { t } ( s ) . } \end{array}
$$

Here $B _ { t }$ is a declared numerical logit-space field. The specialization $\dot { \theta } _ { t } = u _ { t }$ is called the raw flow. An actual optimizer path uses its realized $\dot { \theta } _ { t } .$ . Restricted diagnostic fields are extended by zero outside their recorded token coordinates, and their Jacobians are restricted consistently; these diagnostic supports leave the sampler’s allowed set unchanged.

Lemma C.2 (Log-softmax differentiation and centering). Write $\pi = \pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ as a vector in thefull vocabulary, with zeros outside its fixed allowed set, and let $\mathsf { A } ^ { \pi } = I - \mathsf { 1 } \pi ^ { \top }$ . For each allowed token a,

$$
\nabla _ { \theta } \log \pi _ { \theta } ( a \mid x ) = \tau ^ { - 1 } \Phi _ { \theta } ( x ) ^ { \top } ( e _ { a } - \pi ) , \qquad \sum _ { a } \pi ( a \mid x ) \nabla _ { \theta } \log \pi ( a \mid x ) = 0 .
$$

Consequently $\zeta _ { t } ( x , a ) : = \partial _ { t } \log \pi _ { t } ( a \mid x )$ satisfies $\begin{array} { r } { \sum _ { a } \pi _ { t } ( a \mid x ) \zeta _ { t } ( x , a ) = 0 . \ I f \mathbf { 1 } ^ { \top } w = 0 } \end{array}$ , then $( \mathsf { A } ^ { \pi } ) ^ { \top } w = w .$ . Sums involving log $\pi o r \zeta$ range only over allowed tokens.

Proof. For allowed $^ { a , }$ the logarithm of the displayed sampler in Assumption C.1 is

$$
\log \pi ( a \mid x ) = \tau ^ { - 1 } z ( x , a ) - \log \sum _ { c \in \mathcal { V } \pi ( x ) } \exp ( z ( x , c ) / \tau ) .
$$

Its partial derivative with respect to any full-vocabulary logit $z ( x , c )$ is $\tau ^ { - 1 } ( \mathbf { 1 } \{ a = c \} - \pi ( c \mid x ) )$ including the value zero when c is disallowed. The chain rule proves the gradient formula. Averaging over allowed a yields

$$
\tau ^ { - 1 } \Phi ( x ) ^ { \top } \left( \sum _ { a } \pi ( a \mid x ) e _ { a } - \pi \sum _ { a } \pi ( a \mid x ) \right) = 0 .
$$

Pairing this equality with $\dot { \theta } _ { t }$ gives the zero-mean velocity. Finally, $( \mathsf { A } ^ { \pi } ) ^ { \top } w = ( I - \pi \mathbf { 1 } ^ { \top } ) w =$ $w - \pi ( \mathbf { 1 } ^ { \top } w ) = w$ □

Proposition C.3 (Exact reverse-KL request). At a fixed prefix, let $\begin{array} { r } { c = \sum _ { a } p _ { a } \log ( p _ { a } / q _ { a } ) , \ell _ { a } = } \end{array}$ $\log ( p _ { a } / q _ { a } )$ , and $\begin{array} { r } { \bar { \ell } = \sum _ { a } p _ { a } \ell _ { a } , } \end{array}$ , with $p = \operatorname { s o f t m a x } ( z )$ and $q _ { a } > 0 .$ . Then

$$
B _ { a } = - \partial _ { z _ { a } } c = p _ { a } ( \bar { \ell } - \ell _ { a } ) , \qquad { \bf 1 } ^ { \top } B = 0 , \qquad g = \Phi ^ { \top } B = - \nabla _ { \theta } c .\tag{10}
$$

Forfixed source weights $\begin{array} { r } { \mu , u = \sum _ { s } \mu ( s ) g ( s ) } \end{array}$ is the negative gradient of $\textstyle \sum _ { s } \mu ( s ) c ( s )$

Proof. Positivity makes the logarithms differentiable. Since $\partial _ { z _ { a } } p _ { b } = p _ { b } ( { \bf 1 } \{ a = b \} - p _ { a } )$ and $\partial _ { p _ { b } } [ p _ { b } \log ( p _ { b } / q _ { b } ) ] = \ell _ { b } + 1$ , we have

$$
\begin{array} { l } { { \displaystyle \partial _ { z _ { a } } c = \sum _ { b } p _ { b } ( \mathbf { 1 } \{ a = b \} - p _ { a } ) ( \ell _ { b } + 1 ) } } \\ { { \displaystyle \qquad = p _ { a } ( \ell _ { a } + 1 ) - p _ { a } \sum _ { b } p _ { b } ( \ell _ { b } + 1 ) = p _ { a } ( \ell _ { a } - \bar { \ell } ) . } } \end{array}
$$

Thus $\begin{array} { r } { \sum _ { a } B _ { a } = \bar { \ell } \sum _ { a } p _ { a } - \sum _ { a } p _ { a } \ell _ { a } = 0 } \end{array}$ . The chain rule gives $\nabla _ { \theta } c = \Phi ^ { \top } \nabla _ { z } c ;$ linearity of the finite sum proves the last assertion when $\mu$ is held fixed. □

This is a full-distribution identity. For an implemented objective, such as a truncated-support loss or detached-coefficient surrogate, B must be that objective’s own complete numerical negative logit derivative, including the softmax coupling. For the conversion diagnostic, B denotes the explicitly specified audit field in Section D.3. The response and conversion results hold for this field’s pullback, whether or not that pullback coincides with exact-KL descent. Holding $\mu$ fixed applies to computing one update; subsequent source populations can change.

## C.2 RESPONSE LAW: EXACT FINITE STEPS AND CONTROLLED REMAINDERS

Connection to Section 3.1, Equation (2). The integral is exact for an arbitrary realized step. The main-text kernel expression is its raw-gradient, first-order specialization.

Proposition C.4 (Exact response and Taylor remainder). Let $\Delta \theta = \theta _ { k + 1 } - \theta _ { k }$ , and let $\theta _ { r } = \theta _ { k } + r \Delta \theta$ $f o r 0 \le r \le 1$ . For anyfixed receiver x and token $a ,$

$$
\delta _ { k } ^ { p } ( x , a ) = \int _ { 0 } ^ { 1 } \nabla _ { \theta } \log p _ { \theta _ { r } } ( a \mid x ) ^ { \top } \Delta \theta d r .\tag{11}
$$

$$
\begin{array} { r } { I f M _ { k } ( x , a ) = \operatorname* { s u p } _ { 0 \leq r \leq 1 } \| \nabla _ { \theta } ^ { 2 } \log p _ { \theta _ { r } } ( a \mid x ) \| _ { \mathrm { o p } } , t h e n } \end{array}
$$

$$
\begin{array} { r } { \delta _ { k } ^ { p } ( x , a ) = \nabla _ { \theta } \log p _ { k } ( a \mid x ) ^ { \top } \Delta \theta + R _ { k } ( x , a ) , \qquad | R _ { k } ( x , a ) | \leq \frac { 1 } { 2 } M _ { k } ( x , a ) \| \Delta \theta \| ^ { 2 } . } \end{array}
$$

The corresponding exact logit displacement is $\begin{array} { r } { z _ { k + 1 } ( x ) - z _ { k } ( x ) = \int _ { 0 } ^ { 1 } \Phi _ { \theta _ { r } } ( x ) \Delta \theta d r } \end{array}$

Proof. Define $f ( r ) = \log p _ { \theta _ { r } } ( a \mid x )$ . Its first two derivatives are

$$
f ^ { \prime } ( r ) = \nabla _ { \boldsymbol { \theta } } \log p _ { \boldsymbol { \theta } _ { r } } ( a \mid \boldsymbol { x } ) ^ { \top } \Delta \boldsymbol { \theta } , \qquad f ^ { \prime \prime } ( r ) = \Delta \boldsymbol { \theta } ^ { \top } \nabla _ { \boldsymbol { \theta } } ^ { 2 } \log p _ { \boldsymbol { \theta } _ { r } } ( a \mid \boldsymbol { x } ) \Delta \boldsymbol { \theta } .
$$

The fundamental theorem of calculus gives Equation (11). Applying it also to $f ^ { \prime }$ and changing the order of integration over $0 \leq v \leq r \leq \bar { 1 }$ gives

$$
f ( 1 ) - f ( 0 ) - f ^ { \prime } ( 0 ) = \int _ { 0 } ^ { 1 } \int _ { 0 } ^ { r } f ^ { \prime \prime } ( v ) d v d r = \int _ { 0 } ^ { 1 } ( 1 - v ) f ^ { \prime \prime } ( v ) d v .
$$

The Hessian is continuous on a compact segment, so $M _ { k } ( x , a ) < \infty$ . The bound follows from $| f ^ { \prime \prime } ( v ) | \leq M _ { k } ( x , a ) \| \Delta \theta \| ^ { 2 }$ and $\begin{array} { r } { \int _ { 0 } ^ { 1 } ( 1 - v ) d v = 1 / 2 } \end{array}$ . Applying the same first-derivative calculation to each logit proves the vector identity. □

Corollary C.5 (Small-step response law). For $\Delta \theta _ { k } = \eta u _ { k }$ with $\begin{array} { r } { u _ { k } = \sum _ { s } \mu _ { k } ( s ) \Phi _ { k } ( s ) ^ { \top } B _ { k } ( s ) } \end{array}$

$$
\delta _ { k } ^ { p } ( x , a ) = \eta \sum _ { s } \mu _ { k } ( s ) [ \mathsf { A } _ { k } ^ { p } ( x ) K _ { k } ( x , s ) B _ { k } ( s ) ] _ { a } + R _ { k } ( x , a ) , \qquad | R _ { k } ( x , a ) | \le \frac { 1 } { 2 } \eta ^ { 2 } M _ { k } ( x , a ) \| u _ { k } \| ^ { 2 } .
$$

For fixed $\theta _ { k } , u _ { k }$ and sufficiently small $\eta ,$ the remainder is $O ( \eta ^ { 2 } )$ , as in Equation (2).

Proof. At temperature one, Lemma C.2 gives $\nabla _ { \theta }$ log $p _ { k } ( a \mid x ) ^ { \top } = e _ { a } ^ { \top } \mathsf { A } _ { k } ^ { p } ( x ) \Phi _ { k } ( x )$ . Substituting $u _ { k }$ in Proposition C.4 therefore yields

$$
\nabla _ { \theta } \log p _ { k } ( a \mid x ) ^ { \top } \boldsymbol { u } _ { k } = \sum _ { s } \mu _ { k } ( s ) e _ { a } ^ { \top } \mathsf { A } _ { k } ^ { p } ( x ) \Phi _ { k } ( x ) \Phi _ { k } ( s ) ^ { \top } B _ { k } ( s ) ,
$$

which is the asserted kernel term. The remainder bound follows by substituting $\Delta \theta _ { k } = \eta u _ { k }$ . To justify the asymptotic notation, choose $\eta _ { 0 } > 0$ so that all segments with $| \eta | \leq \eta _ { 0 }$ lie in a compact neighborhood where the Hessian is continuous. Its supremum on this neighborhood supplies a single bound independent of η. A family of varying directions requires a uniform bound on their norms as well. □

Velocity and the optimizer. At every differentiable time on an actual parameter path,

$$
\begin{array} { r } { v _ { t } ( x ) : = \dot { z } _ { t } ( x ) = \Phi _ { t } ( x ) \dot { \theta } _ { t } , \qquad \zeta _ { t } ( x , a ) = \tau ^ { - 1 } ( e _ { a } - \pi _ { t } ( \cdot \vert x ) ) ^ { \top } v _ { t } ( x ) . } \end{array}
$$

These follow directly by the chain rule and Lemma C.2. Under gradient flow, substituting $\dot { \theta } _ { t } =$ $u _ { t }$ gives $\begin{array} { r } { { v _ { t } } ( x ) = \sum _ { s } { K _ { t } ( x , s ) \Gamma _ { t } ( s ) } } \end{array}$ and hence Equation (6). With momentum, preconditioning, clipping, or weight decay, the actual $\dot { \theta } _ { t }$ in general differs from the raw $u _ { t }$ . However, with the rich tradition of theoretical analysis on the learning dynamics of neural networks including the seminal Ren & Sutherland (2025); Jacot et al. (2018); Arora et al. (2019), gradient-flow analysis is still genuinely informative in obtaining rich theoretical insights. We leave extensions to the more convoluted optimizer settings to future work.

## C.3 COLLECTIVE RESPONSE AND ITS GEOMETRY

Connection to Section 3.2, Equations (3) and (4). We fix the checkpoint, output coordinates, and normalized bank ν in this subsection. Zero-weight sites are omitted. The space $\mathcal { H } _ { \nu }$ of fields on the remaining finite bank has the inner product

$$
\langle F , G \rangle _ { \nu } = \sum _ { s } \nu ( s ) F ( s ) ^ { \top } G ( s ) , \qquad \| F \| _ { \nu } ^ { 2 } = \langle F , F \rangle _ { \nu } .
$$

Define $\mathcal { I } : \mathbb { R } ^ { d }  \mathcal { H } _ { \nu } \mathfrak { b y } ( \mathcal { I } v ) ( s ) = \Phi ( s ) v$ . This is a finite-dimensional Hilbert space, so orthogonal projections and adjoints are well-defined.

Proposition C.6 (Collective response and conversion factorization). Let $g ( s ) = \Phi ( s ) ^ { \top } B ( s ) , u =$ $\mathbb { E } _ { \boldsymbol { \nu } } \boldsymbol { \dot { g } } , \boldsymbol { P } = \| \boldsymbol { B } \| _ { \nu } ^ { 2 }$ , and $D _ { g } = \mathbb { E } _ { \nu } \Vert g \Vert ^ { 2 }$ . The adjoint is $\mathcal { T } ^ { * } B = u ,$ , and $\kappa = \mathcal { I } \mathcal { I } ^ { * }$ is self-adjoint and positive semidefinite. For the same-bank raw direction and $P > 0$

$$
\dot { z } = K B = \Xi B + r , \qquad \Xi = \frac { \langle B , K B \rangle _ { \nu } } { { \cal P } } = \frac { \| u \| ^ { 2 } } { { \cal P } } \ge 0 , \qquad \langle B , r \rangle _ { \nu } = 0 .
$$

If also $D _ { g } > 0 ;$ , then

$$
A = \frac { D _ { g } } { P } , \qquad C = \frac { \| u \| ^ { 2 } } { D _ { g } } , \qquad \Xi = A C , \qquad \| u \| ^ { 2 } = P A C , \qquad 0 \le C \le 1 .
$$

Here $C = 1$ exactly when $g ( s ) =$ u at every positive-weight site, and $C = 0$ exactly when $u = 0 , H$ $P > 0$ but $D _ { g } = 0 ;$ , then $u = 0 , A = \Xi = 0$ , and C is undefined. $I f P = 0$ , then $B = g = u = 0$ on the bank and $\lbrack A , C , \Xi$ are not defined by these ratios.

Proof. For any field F and parameter vector v,

$$
\langle F , { \mathcal { I } } v \rangle _ { \nu } = \sum _ { s } \nu ( s ) F ( s ) ^ { \top } \Phi ( s ) v = \left( \sum _ { s } \nu ( s ) \Phi ( s ) ^ { \top } F ( s ) \right) ^ { \top } v .
$$

This proves the adjoint formula. It also gives

$$
\langle F , K G \rangle _ { \nu } = ( \mathcal { I } ^ { * } F ) ^ { \top } ( \mathcal { I } ^ { * } G ) = \langle K F , G \rangle _ { \nu } , \qquad \langle F , K F \rangle _ { \nu } = \| \mathcal { I } ^ { * } F \| ^ { 2 } \geq 0 .
$$

By the chain rule, the logit velocity generated by u is $\mathcal { I } u = \mathcal { K } B$ . Thus $\langle B , \dot { z } \rangle _ { \nu } = \| u \| ^ { 2 }$ . Since $P > 0$ , the strictly convex quadratic $\alpha \mapsto \| \dot { z } - \alpha B \| _ { \nu } ^ { 2 }$ has its unique minimum at $\alpha = \langle \ddot { B } , \dot { z } \rangle _ { \nu } / P$ This is Ξ, and the residual has the stated orthogonality.

For $D _ { g } > 0$ , multiplication of the defined ratios gives $\Xi = A C$ and $\| u \| ^ { 2 } = P A C$ . Expanding a centered squared norm gives

$$
\begin{array} { r } { \mathbb { E } _ { \boldsymbol { \nu } } \| \boldsymbol { g } - \boldsymbol { u } \| ^ { 2 } = \mathbb { E } _ { \boldsymbol { \nu } } \| \boldsymbol { g } \| ^ { 2 } - 2 \boldsymbol { u } ^ { \top } \mathbb { E } _ { \boldsymbol { \nu } } \boldsymbol { g } + \| \boldsymbol { u } \| ^ { 2 } = D _ { \boldsymbol { g } } - \| \boldsymbol { u } \| ^ { 2 } \geq 0 . } \end{array}
$$

This proves $C \leq 1$ and its equality condition; nonnegativity of C and its zero condition follow from its numerator. If $D _ { g } = 0$ , every nonnegative summand $\dot { \nu } ( s ) \| g ( s ) \| ^ { 2 }$ vanishes, so $g = u = 0$ . If $P = 0$ , the same argument first gives $\bar { B = 0 }$ , then $g = u = 0$ . These cases cannot be handled by cancelling zero denominators. □

Proposition C.7 (Joint singular modes). Stack the blocks $\sqrt { \nu ( s ) } \Phi ( s )$ into $\mathrm { ~ J ~ } \in \mathrm { ~ \mathbb { R } ^ {  { N _ { \nu } } \times d } ~ }$ and $\sqrt { \nu ( s ) } B ( s )$ into $\textbf { B } \in \mathbb { R } ^ { N _ { \nu } }$ . Let the compact singular-value decomposition be $\mathsf { J } = U \Sigma V ^ { \top }$ , of rank r, with orthonormal columns $U _ { j } , V _ { j }$ and positive singular values $\sigma _ { j } . \mathrm { ~ } I f P > 0 ;$

$$
u = \sum _ { j = 1 } ^ { r } \sigma _ { j } ( U _ { j } ^ { \top } { \bf B } ) V _ { j } , \qquad \Xi = \sum _ { j = 1 } ^ { r } \sigma _ { j } ^ { 2 } \frac { | U _ { j } ^ { \top } { \bf B } | ^ { 2 } } { \| { \bf B } \| ^ { 2 } } .\tag{12}
$$

The component of B orthogonal to range(J) produces zero parameter response.

Proof. The stacking gives $\begin{array} { r } { \mathbf { J } ^ { \top } \mathbf { B } = \sum _ { s } \nu ( s ) \Phi ( s ) ^ { \top } B ( s ) = u } \end{array}$ and $\| \mathbf { B } \| ^ { 2 } = P$ . Substitution of the singular-value decomposition proves the first identity. Since $V _ { i } ^ { \top } V _ { j } = \mathbf { 1 } \{ i = j \}$ , its squared norm is $\textstyle \sum _ { j } \sigma _ { j } ^ { 2 } | U _ { j } ^ { \top } \mathbf { B } | ^ { 2 }$ ; division by P proves the second. Finally $\mathbf { J } ^ { \top } \mathbf { B } _ { \perp } = V \Sigma U ^ { \top } \mathbf { B } _ { \perp } = 0$ when $U ^ { \top } { \bf B } _ { \bot } = \bar { 0 } . \mathrm { I f } r = 0$ , these are empty sums and $u = \Xi = 0$ □

Joint modes span multiple prefixes and tokens. Their parameter-space components are orthogonal, whereas individual per-prefix gradients need not be. Strong cancellation between local gradients is therefore compatible with large self-kernel responses but a small joint response; independently sorted self-kernel spectra do not determine Equation (12).

Proposition C.8 (Coherence and interference). For independent $s , s ^ { \prime } \sim \nu ,$

$$
\begin{array} { r } { \| u \| ^ { 2 } = \mathbb { E } \langle g ( s ) , g ( s ^ { \prime } ) \rangle = \mathbb { E } _ { \nu } \| g ( s ) \| ^ { 2 } - \frac { 1 } { 2 } \mathbb { E } \| g ( s ) - g ( s ^ { \prime } ) \| ^ { 2 } . } \end{array}
$$

If $\begin{array} { r } { \mathrm { ~ \vec ~ { ~ } B ~ } = \sum _ { m = 1 } ^ { M } B ^ { ( m ) } } \end{array}$ is any finite field decomposition and $u ^ { ( m ) } = \mathbb { E } _ { \nu } [ \Phi ^ { \top } B ^ { ( m ) } ]$ , then $\| u \| ^ { 2 } = $ $\begin{array} { r } { \sum _ { m } \| u ^ { ( m ) } \| ^ { 2 } + 2 \sum _ { m < n } \langle u ^ { ( m ) } , u ^ { ( n ) } \rangle } \end{array}$

Proof. Independence gives the explicit finite-sum identity

$$
\mathbb { E } \langle g ( s ) , g ( s ^ { \prime } ) \rangle = \sum _ { s , s ^ { \prime } } \nu ( s ) \nu ( s ^ { \prime } ) g ( s ) ^ { \top } g ( s ^ { \prime } ) = \left( \sum _ { s } \nu ( s ) g ( s ) \right) ^ { \top } \left( \sum _ { s ^ { \prime } } \nu ( s ^ { \prime } ) g ( s ^ { \prime } ) \right) .
$$

This is $\| u \| ^ { 2 }$ . Expanding the squared difference and using the identical marginals yields $\mathbb { E } \| g ( s ) -$ $g ( s ^ { \prime } ) \Vert ^ { 2 } = 2 D _ { g } - 2 \Vert u \Vert ^ { 2 }$ . The field decomposition gives $\begin{array} { r } { u = \sum _ { m } u ^ { ( m ) } } \end{array}$ by linearity, and the last formula follows by expanding the square and collecting both copies of each off-diagonal inner product. □

Proposition C.9 (Finite-sample correction of the mean-gradient energy). Let $g _ { 1 } , \ldots , g _ { n } , n > 1$ , be independent identically distributed parameter vectors with finite second moment and mean u. Define ${ \bar { g } } = n ^ { - 1 } \sum _ { i } g _ { i }$ and $\begin{array} { r } { \widehat { D } _ { g } = n ^ { - 1 } \sum _ { i } \lVert g _ { i } \rVert ^ { 2 } } \end{array}$ . Then

$$
\widehat { U } = \frac { n \| \bar { g } \| ^ { 2 } - \widehat { D } _ { g } } { n - 1 } = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } g _ { i } ^ { \top } g _ { j }
$$

has expectation $\| u \| ^ { 2 }$

Proof. Expanding $n ^ { 2 } \| \bar { g } \| ^ { 2 }$ gives $\begin{array} { r } { \sum _ { i } \| g _ { i } \| ^ { 2 } + \sum _ { i \neq j } g _ { i } ^ { \top } g _ { j } } \end{array}$ . Subtracting $\begin{array} { r } { n \widehat { D } _ { g } = \sum _ { i } \| g _ { i } \| ^ { 2 } } \end{array}$ and dividing by $n ( n - 1 )$ proves the second expression. Independence gives $\mathbb { E } [ g _ { i } ^ { \top } g _ { j } ] = ( \mathbb { E } g _ { i } ) ^ { \top } ( \mathbb { E } g _ { j } ) = \| u \| ^ { 2 }$ for each of its $n ( n - 1 )$ terms. Thus its expectation is $\| u \| ^ { 2 }$ . The finite second moment makes every term integrable. Positive clipping and division by the random $\widehat { D } _ { g }$ are nonlinear, so the bias of the resulting coherence ratio falls outside this linear expectation calculation. Other sampling designs, including sampling without replacement from a fixed finite population, require their own design-specific bias calculation. □

Policy-level observation of a logit response. For a fixed vector $w ( x )$ with $\mathbf { 1 } ^ { \top } w ( x ) = 0$

$$
w ( x ) ^ { \top } [ z _ { \theta ^ { \prime } } ( x ) - z _ { \theta } ( x ) ] = w ( x ) ^ { \top } [ \log p _ { \theta ^ { \prime } } ( \cdot \mid x ) - \log p _ { \theta } ( \cdot \mid x ) ] .
$$

Indeed, log $\begin{array} { r } { p _ { \theta } = z _ { \theta } - \log \sum _ { a } e ^ { z _ { \theta } ( a ) } \mathbf { 1 } } \end{array}$ , and both scalar-normalizer terms disappear under $w ^ { \top }$ . The identity holds for finite displacements, for zero-extended contrasts on a recorded support, and for centered and uncentered training fields alike. The experimental contrast is specified in Section D.3.

## C.4 WHY THE FROZEN SPECTRAL PROBE MEASURES ACCESSIBILITY

Connection to Section 3.2 and the fixed-bank diagnostic in Section 3.4. All time dependence below is in the tangent map; the bank, coordinates, and numerical field are fixed.

Proposition C.10 (Frozen-field Rayleigh expansion). Let $P _ { 0 } = \mathbb { E } _ { \nu _ { 0 } } \Vert B _ { 0 } \Vert ^ { 2 } > 0$ , and define on nonzero-request sites

$$
\omega _ { 0 } ( s ) = \frac { \nu _ { 0 } ( s ) \| B _ { 0 } ( s ) \| ^ { 2 } } { P _ { 0 } } , \qquad b _ { 0 } ( s ) = \frac { B _ { 0 } ( s ) } { \| B _ { 0 } ( s ) \| } .
$$

Zero-request sites have weight zero and are omittedfrom sums involving $b _ { 0 } .$ . For any nonzerofield on a specified bank,

$$
A [ B ] = \frac { \mathbb { E } _ { \nu } [ B ( s ) ^ { \top } K ( s , s ) B ( s ) ] } { \mathbb { E } _ { \nu } \| B ( s ) \| ^ { 2 } } .\tag{13}
$$

Let $( \lambda _ { j , t } ( s ) , v _ { j , t } ( s ) )$ be a complete orthonormal eigendecomposition of $K _ { t } ( s , s )$ . Then

$$
\begin{array} { r l } & { \quad a _ { t } ( s ) = b _ { 0 } ( s ) ^ { \top } K _ { t } ( s , s ) b _ { 0 } ( s ) = \displaystyle \sum _ { j } \lambda _ { j , t } ( s ) | v _ { j , t } ( s ) ^ { \top } b _ { 0 } ( s ) | ^ { 2 } , } \\ & { \quad A _ { t } [ B _ { 0 } ] = \displaystyle \sum _ { s } \omega _ { 0 } ( s ) a _ { t } ( s ) . } \end{array}\tag{14}
$$

The weights sum to one. In particular, $A _ { t } [ B _ { 0 } ]$ is constant whenever the self-kernels are constant on the bank.

Proof. For each site, $\| \Phi ( s ) ^ { \top } B ( s ) \| ^ { 2 } \ = \ B ( s ) ^ { \top } \Phi ( s ) \Phi ( s ) ^ { \top } B ( s ) \ = \ B ( s ) ^ { \top } K ( s , s ) B ( s )$ . Substituting into the definition $A ~ = ~ D _ { g } / P$ proves Equation (13). In the frozen case, write $B _ { 0 } ( s ) \ \stackrel { \bf { \bar { \ } } } { = } \ \| B _ { 0 } ( s ) \| b _ { 0 } ( s )$ at each nonzero site and divide the weighted sum by $P _ { 0 } ;$ ; this gives $\begin{array} { r } { A _ { t } [ \dot { B _ { 0 } } ] = \dot { \sum } _ { s } \omega _ { 0 } ( \dot { s } ) b _ { 0 } ( \dot { s } ) ^ { \top } K _ { t } ( s , s ) b _ { 0 } ( s ) } \end{array}$ . The weights sum to $\bar { P _ { 0 } / P _ { 0 } } = 1$

Each $K _ { t } ( s , s ) \ = \ \Phi _ { t } ( s ) \Phi _ { t } ( s ) ^ { \top }$ is symmetric and positive semidefinite: for every vector v, $v ^ { \top } K _ { t } ( s , s ) v ~ = ~ \| \Phi _ { t } ( \bar { s } ) ^ { \top } v \| ^ { 2 } ~ \geq ~ 0$ The finite-dimensional spectral theorem therefore gives $\begin{array} { r } { K _ { t } = \sum _ { j } \lambda _ { j , t } v _ { j , t } v _ { j , t } ^ { \top } } \end{array}$ with $\lambda _ { j , t } \geq 0$ . Inserting this sum into the quadratic form proves Equation (14). All coefficients except $K _ { t }$ are fixed, which proves the constancy assertion. Repeated eigenvalues do not affect the full sum: the sum of squared projections within an eigenspace is basis-invariant.

The live quantity $A _ { t } [ B _ { t } ]$ can also change through its field and energy weights. The frozen collective gain $\mathrm { i s }  _ { t } [ B _ { 0 } ] \doteq A _ { t } [ B _ { 0 } ] C _ { t } [ B _ { 0 } ]$ when the denominators are nonzero; both factors use $B _ { 0 }$ , whereas $\mathsf { \bar { A } } _ { t } [ B _ { 0 } ] C _ { t } [ B _ { t } ]$ mixes two fields and is a different quantity.

## C.5 ROUTING AND CONDITIONAL DRIFT ON MOVING PREFIX POPULATIONS

Connection to Section 3.3, Equations (5) and (6). Let $m ^ { \mathrm { o b s } } ( x ) \geq 0$ be a fixed weight on the nonterminal prefixes of the observation tree. Assume at least one eligible prefix has positive reach probability, and define

$$
\nu _ { t } ( x ) = \frac { m ^ { \mathrm { o b s } } ( x ) d _ { t } ( x ) } { Z _ { t } ^ { \mathrm { o b s } } } , \qquad Z _ { t } ^ { \mathrm { o b s } } = \sum _ { x } m ^ { \mathrm { o b s } } ( x ) d _ { t } ( x ) > 0 .\tag{15}
$$

This is normalization of expected weighted counts, which allows fixed depth masks, context classes defined by a fixed reference learner, and empirical minibatches of unequal length.

Proposition C.11 (Routing and conditional drift). Let $Y _ { t } ( x )$ be a scalar readout that is continuous and piecewise $C ^ { 1 }$ along the parameter path, and put $\begin{array} { r } { J _ { Y } ( t ) = \sum _ { x } \nu _ { t } ( x ) Y _ { t } ( x ) } \end{array}$ . For a positive-reach prefix with ancestry $( x _ { j } , a _ { j } ) _ { j < h ( x ) }$ , define

$$
\lambda _ { t } ( x ) = \partial _ { t } \log d _ { t } ( x ) = \sum _ { j < h ( x ) } \zeta _ { t } ( x _ { j } , a _ { j } ) .
$$

Then, almost everywhere,

$$
\begin{array} { r l } & { \dot { \nu } _ { t } ( x ) = \nu _ { t } ( x ) \big ( \lambda _ { t } ( x ) - \mathbb { E } _ { \nu _ { t } } \lambda _ { t } \big ) , } \\ & { \dot { J } _ { Y } ( t ) = R _ { Y } ( t ) + D _ { Y } ( t ) , \qquad R _ { Y } ( t ) = \operatorname { C o v } _ { \nu _ { t } } ( Y _ { t } , \lambda _ { t } ) , \quad D _ { Y } ( t ) = \mathbb { E } _ { \nu _ { t } } \dot { Y } _ { t } . } \end{array}
$$

$I f Y _ { t } ( x ) = Y ( z _ { t } ( x ) , x )$ for a $C ^ { 1 }$ local function Y with no explicit time dependence, then $\dot { Y } _ { t } ( x ) =$ $\nabla _ { z } Y ( z _ { t } ( x ) , x ) ^ { \top } v _ { t } ( x )$ . Under rawflow, $v _ { t } = K _ { t } \Gamma _ { t }$ . For every subinterval $[ a , b ] , J _ { Y } ( b ) - J _ { Y } ( a ) =$ $\textstyle \int _ { a } ^ { b } ( R _ { Y } + D _ { Y } ) d t$

Proof. For fixed $x ,$ the prompt $b _ { x }$ and all tokens in its ancestry are fixed while training time changes. Thus

$$
d _ { t } ( x ) = \beta ( b _ { x } ) \prod _ { j < h ( x ) } \pi _ { t } ( a _ { j } \mid x _ { j } ) , \qquad \partial _ { t } \log d _ { t } ( x ) = \sum _ { j < h ( x ) } \partial _ { t } \log \pi _ { t } ( a _ { j } \mid x _ { j } ) .
$$

This proves the formula for $\lambda _ { t } ;$ at a root the product is empty and $\lambda _ { t } = 0$ . Also $\dot { d } _ { t } ( x ) = d _ { t } ( x ) \lambda _ { t } ( x )$ Differentiating the finite normalizer gives

$$
\frac { \dot { Z } _ { t } ^ { \mathrm { o b s } } } { Z _ { t } ^ { \mathrm { o b s } } } = \frac { \sum _ { x } m ^ { \mathrm { o b s } } ( x ) d _ { t } ( x ) \lambda _ { t } ( x ) } { Z _ { t } ^ { \mathrm { o b s } } } = \mathbb { E } _ { \nu _ { t } } \lambda _ { t } .
$$

The quotient rule now gives the asserted formula for $\dot { \nu } _ { t }$ at each eligible site. For a site with zero fixed mask, both $\nu _ { t }$ and $\dot { \nu } _ { t }$ are zero, so the same expression holds with that site omitted.

The product rule and the definition of covariance yield

$$
\begin{array} { l } { \displaystyle \dot { J } _ { Y } = \sum _ { x } \dot { \nu } _ { t } ( x ) Y _ { t } ( x ) + \sum _ { x } \nu _ { t } ( x ) \dot { Y } _ { t } ( x ) } \\ { \displaystyle = \mathbb { E } _ { \nu _ { t } } \big ( Y _ { t } \lambda _ { t } \big ) - ( \mathbb { E } _ { \nu _ { t } } Y _ { t } ) \big ( \mathbb { E } _ { \nu _ { t } } \lambda _ { t } \big ) + \mathbb { E } _ { \nu _ { t } } \dot { Y } _ { t } . } \end{array}
$$

For a local readout, the chain rule gives $\dot { Y } _ { t } = \nabla _ { z } Y ^ { \top } \dot { z } _ { t } ;$ the raw specialization follows from Section C.2. All functions are continuous and piecewise C<sup>1</sup> on a finite partition of the compact interval. Integrating the derivative on each piece and telescoping the boundary values proves the final equality. □

The two roles of the same parameter change. Define $\begin{array} { r } { \mathcal { F } ( \theta _ { g } , \theta _ { e } ) = \sum _ { x } \nu _ { \theta _ { q } } ( x ) Y _ { \theta _ { e } } ( x ) } \end{array}$ . Applying the multivariable chain rule to $\mathcal { F } ( \theta _ { t } , \theta _ { t } )$ gives its generator derivative $\begin{array} { r } { \sum _ { x } \dot { \nu } _ { t } \dot { Y _ { t } } = R _ { Y } } \end{array}$ and evaluator derivative $\begin{array} { r } { \sum _ { x } \nu _ { t } \dot { Y } _ { t } = D _ { Y } } \end{array}$ . Both are contributions of the same parameter change to one derivative. A readout depending on the full parameter vector, rather than only the logits at x, uses $\dot { Y } _ { t } ( x ) =$ $\nabla _ { { \boldsymbol { \theta } } } Y _ { { \boldsymbol { \theta } } } ( x ) ^ { \top } { \dot { \theta } } _ { t }$ instead. This distinction will matter for continuation values.

If prompt weights or observation masks vary explicitly, the formula for $\lambda _ { t }$ alone is insufficient. On positive coordinates the quotient-rule calculation instead centers $\partial _ { t } \log [ m _ { t } ^ { \mathrm { o b s } } ( x ) d _ { t } ( x ) ]$ ; ∂<sub>t</sub> log $d _ { t }$ includes $\partial _ { t } \log \beta _ { t } ( b _ { x } )$ for a changing prompt law. Zero weights or changes of support require separate boundary accounting. The fixed-weight assumptions are therefore substantive.

## C.6 FINITE-WINDOW EXPOSURE AND REGENERATED SOURCE WEIGHTS

Connection to Section 3.3, Equation (7). Use a superscript tr when training and observation employ different fixed prompt laws or samplers. In particular, $\tau _ { \mathrm { t r } }$ can differ from the observation temperature. For a fixed training mask $m ^ { \mathrm { t r } } ( s ) \bar { \geq } 0$ , let

$$
\mu _ { t } ( s ) = \frac { m ^ { \mathrm { t r } } ( s ) d _ { t } ^ { \mathrm { t r } } ( s ) } { Z _ { t } ^ { \mathrm { t r } } } , \qquad Z _ { t } ^ { \mathrm { t r } } = \sum _ { \mathrm { { s } } } m ^ { \mathrm { t r } } ( s ) d _ { t } ^ { \mathrm { t r } } ( s ) > 0 .
$$

The following proposition assumes one fixed eligible support throughout the interval, as guaranteed by Assumption C.1. Merely sharing some positive-support states at the endpoints is not sufficient for the unrestricted normalization formula.

Proposition C.12 (Source reweighting across a training window). For $a \ \leq \ b ,$ let $\Lambda _ { a : b } ( s ) \ =$ $\textstyle \int _ { a } ^ { b } \lambda _ { t } ^ { \mathrm { t r } } ( s ) d t$ , where $\lambda _ { t } ^ { \mathrm { t r } }$ is the ancestral log reach-probability velocity under the training sampler. On every allowed prefix,

$$
\frac { d _ { b } ^ { \mathrm { t r } } ( s ) } { d _ { a } ^ { \mathrm { t r } } ( s ) } = e ^ { \Lambda _ { a : b } ( s ) } , \qquad \Lambda _ { a : c } ( s ) = \Lambda _ { a : b } ( s ) + \Lambda _ { b : c } ( s ) \quad ( a \leq b \leq c ) .\tag{16}
$$

For each eligible source prefix,

$$
\mu _ { b } ( s ) = \frac { \mu _ { a } ( s ) e ^ { \Lambda _ { a : b } ( s ) } } { \mathbb { E } _ { v \sim \mu _ { a } } e ^ { \Lambda _ { a : b } ( v ) } } , \qquad \Gamma _ { b } ( s ) = \mu _ { b } ( s ) B _ { b } ( s ) .\tag{17}
$$

A zero fixed mask stays zero.

Proof. By Proposition C.11 with the training sampler, $\partial _ { t }$ log $d _ { t } ^ { \mathrm { t r } } ( s ) = \lambda _ { t } ^ { \mathrm { t r } } ( s )$ on each smooth piece. Integrating and telescoping gives log $d _ { b } ^ { \mathrm { t r } } ( s ) \bar { - } \log \bar { d } _ { a } ^ { \mathrm { t r } } ( s ) = \bar { \Lambda } _ { a : b } \bar { ( } s )$ . Exponentiating proves the ratio, and additivity of the integral proves the second identity.

Because the eligible set is the same at both endpoints, substitution into the entire source normalizer is valid:

$$
\begin{array} { l } { { \displaystyle Z _ { b } ^ { \mathrm { t r } } = \sum _ { s } m ^ { \mathrm { t r } } ( s ) d _ { a } ^ { \mathrm { t r } } ( s ) e ^ { \Lambda _ { a : b } ( s ) } } } \\ { { \displaystyle ~ = Z _ { a } ^ { \mathrm { t r } } \sum _ { s } \mu _ { a } ( s ) e ^ { \Lambda _ { a : b } ( s ) } . } } \end{array}
$$

The expectation is strictly positive and finite, since it is a finite sum of positive terms on the nonempty eligible set. Dividing $m ^ { \mathrm { \scriptsize { t r } } } { \bar { ( } } s { ) } d _ { b } ^ { \mathrm { \scriptsize { t r } } } ( s )$ by this expression for $Z _ { b } ^ { \mathrm { t r } }$ proves Equation (17). The expression for $\Gamma _ { b }$ defines the regenerated request population with the current numerical field $B _ { b }$ . If $m ^ { \mathrm { t r } } ( s ) = 0$ both endpoint weights are zero directly from their definition. □

An eligible prefix can be absent from a finite recorded batch while having positive population weight at every checkpoint. The reference in Equation (17) is the population $\mu _ { a } ,$ which includes such prefixes. At a root, $\Lambda = 0$ , although normalization by the total expected token count can change its source weight.

## C.6.1 FINITE-BATCH EXTRACTION AND NATIVE CHECKPOINTS

Proposition C.13 (Transport of a normalized batch extraction). Let B be afinite batch space and $P _ { a } ^ { \mathrm { t r } } , P _ { b } ^ { \mathrm { t r } }$ two batch laws with identical positive support. For each batch B, let $\widehat { \mu } _ { B } ( s ) \geq 0$ be an extraction rule with $\begin{array} { r } { \sum _ { s } \widehat { \mu } _ { B } ( s ) = 1 } \end{array}$ , determined only by the batch text. Set $\mu _ { t } ( s ) = \mathbb { E } _ { P _ { t } ^ { \mathrm { t r } } } \widehat { \mu } _ { B } ( s )$ and $\Lambda _ { a : b } ( \mathcal { B } ) = \log [ P _ { b } ^ { \mathrm { t r } } ( \mathcal { B } ) / P _ { a } ^ { \mathrm { t r } } ( \mathcal { B } ) ]$ . Then

$$
\mu _ { b } ( s ) = \mathbb { E } _ { P _ { a } ^ { \mathrm { t r } } } \bigl [ \widehat { \mu } _ { B } ( s ) e ^ { \Lambda _ { a : b } ( B ) } \bigr ] , \qquad \mathbb { E } _ { P _ { a } ^ { \mathrm { t r } } } e ^ { \Lambda _ { a : b } ( B ) } = 1 .\tag{18}
$$

Proof. Expand the first expectation and cancel its positive probability factor:

$$
\sum _ { B } { P } _ { a } ^ { \mathrm { t r } } ( B ) \widehat { \mu } _ { B } ( s ) \frac { P _ { b } ^ { \mathrm { t r } } ( B ) } { P _ { a } ^ { \mathrm { t r } } ( B ) } = \sum _ { B } { P } _ { b } ^ { \mathrm { t r } } ( B ) \widehat { \mu } _ { B } ( s ) = \mu _ { b } ( s ) .
$$

Taking ${ \widehat { \mu } } _ { B } ( s )$ out of this calculation and replacing it by one gives $\begin{array} { r } { \sum _ { B } P _ { b } ^ { \mathrm { t r } } ( B ) = 1 } \end{array}$ . If the batch responses are independent, its probability is their product, so its log likelihood ratio is the sum of their complete-trajectory log ratios. Conditional independence given a fixed shared prompt schedule gives the same factorization conditional on that schedule. □

For example, if $N _ { s } ( B )$ is the eligible count at $s ,$ batch normalization gives $\begin{array} { r l } { \widehat { \mu } _ { B } ( s ) } & { { } = } \end{array}$ $\textstyle N _ { s } ( B ) / \sum _ { v } N _ { v } ( B )$ when the denominator is positive. Its expectation is generally not $\mathbb { E } N _ { s } / \mathbb { E } \sum _ { v } N _ { v }$ Thus the random normalization must stay inside Equation (18); no additional ratio of expectations is introduced. A rule for batches with no eligible positions must be specified before defining a normalized extraction. If extraction also depends on the learner after fixing the text, transport uses the endpoint rule $\widehat { \mu } _ { b , B }$ , not an unchanged rule from time a.

Native checkpoint identity. For discrete checkpoints $k _ { 1 } < k _ { 2 }$ , assume that every token in the chosen prefix has positive sampler probability at every intermediate checkpoint. Then

$$
\begin{array} { r l } & { \Lambda _ { k _ { 1 } : k _ { 2 } } ( s ) : = \displaystyle \sum _ { k = k _ { 1 } } ^ { k _ { 2 } - 1 } \sum _ { j < h ( s ) } \log \frac { \pi _ { k + 1 } ^ { \mathrm { t r } } ( a _ { j } \mid s _ { j } ) } { \pi _ { k } ^ { \mathrm { t r } } ( a _ { j } \mid s _ { j } ) } } \\ & { \quad \quad \quad \quad \quad = \displaystyle \sum _ { j < h ( s ) } \log \frac { \pi _ { k _ { 2 } } ^ { \mathrm { t r } } ( a _ { j } \mid s _ { j } ) } { \pi _ { k _ { 1 } } ^ { \mathrm { t r } } ( a _ { j } \mid s _ { j } ) } = \log \frac { d _ { k _ { 2 } } ^ { \mathrm { t r } } ( s ) } { d _ { k _ { 1 } } ^ { \mathrm { t r } } ( s ) } . } \end{array}
$$

The first equality telescopes the finite sum over checkpoints for each token; the second multiplies the ancestral probability factors and cancels the fixed prompt law. The normalizer calculation in Proposition C.12 then applies unchanged on a fixed eligible support. The identity holds for any optimizer and any learning rate. If a token has zero probability at an intermediate checkpoint, the per-step logarithms above need not exist even when both endpoint probabilities are positive; the endpoint ratio can still be used directly where it is defined.

## C.6.2 CHANGING SUPPORT AND NONSMOOTH READOUTS

Proposition C.14 (Explicit support accounting). Let $P _ { a } , P _ { b }$ be probability laws on a common finite universe, let ${ \mathcal C } \ = \ \operatorname { s u p p } P _ { a }$ ∩ supp $P _ { b } ,$ and let F be a bounded real function. Define $e ^ { \boldsymbol { \Lambda } ( v ) } = $

$P _ { b } ( v ) / P _ { a } ( v )$ only $f o r v \in { \mathcal { C } } .$ . Then

$$
\begin{array} { r l } & { \mathbb { E } _ { P _ { b } } F - \mathbb { E } _ { P _ { a } } F = \displaystyle \sum _ { v \in \mathcal { C } } P _ { a } ( v ) F ( v ) ( e ^ { \Lambda ( v ) } - 1 ) } \\ & { \qquad + \displaystyle \sum _ { v \in \mathrm { s u p p } P _ { b } \backslash \mathcal { C } } P _ { b } ( v ) F ( v ) - \displaystyle \sum _ { v \in \mathrm { s u p p } P _ { a } \backslash \mathcal { C } } P _ { a } ( v ) F ( v ) . } \end{array}
$$

The absolute value of the last two terms together is at most $\| F \| _ { \infty } [ P _ { b } ( \mathcal { C } ^ { c } ) + P _ { a } ( \mathcal { C } ^ { c } ) ]$

Proof. Split each expectation into its contribution on C and on its complement. On C, replace $P _ { b } ( v )$ by $P _ { a } ( v ) e ^ { \Lambda ( v ) }$ and subtract. This gives the identity. Applying the triangle inequality and $| F ( v ) | \leq \| F \| _ { \infty }$ to the remaining sums gives the bound. □

In particular, $\begin{array} { r } { \sum _ { v \in \mathcal { C } } P _ { a } ( v ) e ^ { \Lambda ( v ) } = P _ { b } ( \mathcal { C } ) } \end{array}$ , which need not be one. Common-support ratios do not recover newly created mass or, by themselves, the full normalization of a changed source. The proposition applies to complete-trajectory laws, batch laws, or normalized source laws. Changed prompt sets and masks require the same care.

The local top-16 loss in Section E.1 can change its expression when the selected support changes. If a readout path has finitely many such changes and is piecewise $C ^ { 1 }$ with one-sided limits, integration over each smooth piece gives

$$
J ( b ) - J ( a ) = \int _ { a } ^ { b } \dot { J } ( t ) d t + \sum _ { t _ { i } \in ( a , b ) } [ J ( t _ { i } ^ { + } ) - J ( t _ { i } ^ { - } ) ] ,
$$

with endpoints taken on their respective one-sided branches. The formula follows by telescoping the integrals and inserting the jumps; a continuous branch switch has zero jump. Smooth-flow integrals are asserted only under this regularity; the finite four-cell identity below remains valid whenever its four scores are finite. For an exogenously fixed source, the response and outcome laws still hold, while the source’s own generating rule replaces the on-policy source-tilt formula.

## C.7 EXACT SYMMETRIC ENDPOINT ACCOUNTING AND ITS PATH LIMIT

Connection to Section 3.3, Equation (8). The four-cell estimator is exactly additive at finite endpoints; the difference between its individual components and the continuous routing and drift integrals is computed and bounded in Proposition C.17.

Proposition C.15 (Symmetric endpoint accounting). Let $\nu _ { a } , \nu _ { b }$ be any two probability vectors on the union of two finite rollout banks, and let $Y _ { a } , Y _ { b }$ be finite readouts defined on that union. With $\begin{array} { r } { M _ { i j } = \sum _ { x } \dot { \nu _ { i } } ( x ) \dot { Y } _ { j } ( x ) } \end{array}$ , define

$$
\begin{array} { r } { \widehat { R } _ { a : b } = \frac { 1 } { 2 } [ ( M _ { b a } - M _ { a a } ) + ( M _ { b b } - M _ { a b } ) ] , \qquad \widehat { D } _ { a : b } = \frac { 1 } { 2 } [ ( M _ { a b } - M _ { a a } ) + ( M _ { b b } - M _ { b a } ) ] . } \end{array}\tag{19}
$$

Using $\begin{array} { r } { \langle \nu , Y \rangle : = \sum _ { x } \nu ( x ) Y ( x ) } \end{array}$ for the distribution–readout pairing,

$$
\widehat { R } _ { a : b } = \left. \nu _ { b } - \nu _ { a } , \frac { Y _ { a } + Y _ { b } } { 2 } \right. , \qquad \widehat { D } _ { a : b } = \left. \frac { \nu _ { a } + \nu _ { b } } { 2 } , Y _ { b } - Y _ { a } \right. ,
$$

and $\widehat { R } _ { a : b } + \widehat { D } _ { a : b } = M _ { b b } - M _ { a a }$

Remark C.16. The inner products can be intuitively interpreted: for routing $\widehat { R } _ { a : b } .$ we measure the effect of changing the rollout distribution $v _ { t }$ while holding the readout to the averaged $\frac { Y _ { a } + Y _ { b } } { 2 }$ ; the drift estimator holds the rollout distribution averaged while varying the readouts (how much fitting happened over the period $( a , b ) )$ .

Proof. Set $\Delta \nu = \nu _ { b } - \nu _ { a }$ and $\Delta Y = Y _ { b } - Y _ { a }$ . Bilinearity gives

$$
M _ { b b } - M _ { a a } = \langle \Delta \nu , Y _ { a } \rangle + \langle \nu _ { a } , \Delta Y \rangle + \langle \Delta \nu , \Delta Y \rangle .
$$

Expanding the four terms in Equation (19) yields

$$
\begin{array} { r } { \widehat { R } _ { a : b } = \langle \Delta \nu , Y _ { a } \rangle + \frac { 1 } { 2 } \langle \Delta \nu , \Delta Y \rangle , \qquad \widehat { D } _ { a : b } = \langle \nu _ { a } , \Delta Y \rangle + \frac { 1 } { 2 } \langle \Delta \nu , \Delta Y \rangle . } \end{array}
$$

Proposition C.17 (Difference from the path-integrated channels). Suppose $\nu _ { t } , Y _ { t }$ are $C ^ { 1 }$ on time interval [a, b], and write $\begin{array} { r } { R _ { a : b } = \int _ { a } ^ { b } \langle \dot { \nu } _ { t } , Y _ { t } \rangle } \end{array}$ dt and $\begin{array} { r } { D _ { a : b } = \int _ { a } ^ { b } \langle \nu _ { t } , \dot { Y } _ { t } \rangle } \end{array}$ dt. Then

$$
\widehat { R } _ { a : b } - R _ { a : b } = \int _ { a } ^ { b } \left. \dot { \nu } _ { t } , \frac { Y _ { a } + Y _ { b } } { 2 } - Y _ { t } \right. d t = - ( \widehat { D } _ { a : b } - D _ { a : b } ) .
$$

$\begin{array} { r } { I f L _ { \nu } = \operatorname* { s u p } _ { t } \| \dot { \nu } _ { t } \| _ { 1 } } \end{array}$ and $L _ { Y } = \operatorname* { s u p } _ { t } \| { \dot { Y } } _ { t } \| _ { \infty } ,$ , each channel error is at most $L _ { \nu } L _ { Y } ( b - a ) ^ { 2 } / 2 .$ . If both functions are $C ^ { 2 }$ , define $M _ { \nu } = \operatorname* { s u p } _ { t } \| \ddot { \nu } _ { t } \| _ { 1 }$ and $\begin{array} { r } { M _ { Y } = \operatorname* { s u p } _ { t } \| \ddot { Y } _ { t } \| _ { \infty } ; } \end{array}$ the sharper bound is

$$
| \widehat { R } _ { a : b } - R _ { a : b } | = | \widehat { D } _ { a : b } - D _ { a : b } | \leq \frac { ( b - a ) ^ { 3 } } { 1 2 } ( L _ { \nu } M _ { Y } + M _ { \nu } L _ { Y } ) .
$$

Proof. Substitute $\textstyle \nu _ { b } - \nu _ { a } = \int _ { a } ^ { b } \dot { \nu } _ { t }$ dt in the pairing for $\widehat { R }$ and subtract $R _ { a : b }$ . Since the sums of the two channels both equal $J _ { Y } ( b ) { \bar { - } } J _ { Y } ( a )$ , their errors have opposite signs. The fundamental theorem of calculus gives

$$
\begin{array} { r } { \left\| \displaystyle \frac { Y _ { a } + Y _ { b } } { 2 } - Y _ { t } \right\| _ { \infty } \leq \frac { 1 } { 2 } L _ { Y } [ ( t - a ) + ( b - t ) ] = \frac { 1 } { 2 } L _ { Y } ( b - a ) . } \end{array}
$$

Applying $| \langle v , y \rangle | \leq \| v \| _ { 1 } \| y \| _ { \infty }$ and integrating proves the $C ^ { 1 }$ bound.

For the sharper result, insert $\textstyle { \frac { 1 } { 2 } } ( Y _ { a } + Y _ { b } ) - Y _ { t } = { \frac { 1 } { 2 } } ( \int _ { t } ^ { b } { \dot { Y } } _ { r } d r - \int _ { a } ^ { t } { \dot { Y } } _ { r } d r )$ into the exact error. Reordering these finite-dimensional integrals gives

$$
\widehat { R } - R = { \textstyle \frac { 1 } { 2 } } \int _ { a \leq r < t \leq b } \left[ \langle \dot { \nu } _ { r } , \dot { Y } _ { t } \rangle - \langle \dot { \nu } _ { t } , \dot { Y } _ { r } \rangle \right] d r d t .
$$

Add and subtract $\langle \dot { \nu } _ { r } , \dot { Y } _ { r } \rangle$ in the integrand. Its absolute value is bounded by $L _ { \nu } \| \dot { Y } _ { t } - \dot { Y } _ { r } \| _ { \infty } + \| \dot { \nu } _ { t } -$ $\dot { \nu } _ { r } \| _ { 1 } L _ { Y } \le ( L _ { \nu } M _ { Y } + M _ { \nu } L _ { Y } ) ( t - r )$ . Finally $\textstyle \int _ { a } ^ { b } \int _ { a } ^ { t } ( t - r ) d r d t = ( b - a ) ^ { 3 } / 6$ , which proves the coefficient $1 / 1 2$ □

For a partition of a fixed interval with largest width h, the finite totals telescope exactly. Summing the $C ^ { 1 }$ bound gives a channel error at most $\mathsf { \bar { L } } _ { \nu } L _ { Y } ( b - a ) h / 2$ , tending to zero as $h  0 ;$ the $C ^ { 2 }$ bound gives order $\mathit { \check { h } } ^ { 2 }$ instead.

Scope. The path limit is a population statement; finite-sample differences between empirical banks, checkpoint substitutions, prompt-composition changes, and support-switch jumps lie outside it, and Sections E.1 and E.6 give the corresponding experimental controls. At a fixed finite partition the allocation is a symmetric endpoint convention.

## C.8 CONTINUATION VALUES AND COMPLETED OUTCOMES

Connection to Section 3.3, Equation (9). Recall that a terminal score is a fixed function of a completed rollout. Now, fix a bounded, model-parameter-independent function $Y : \mathcal { T }  \mathbb { R }$ . Write $J _ { Y } ( \bar { t } ) = \mathbb { E } _ { \Pi _ { t } } Y$ for the expectation of the terminal score over the time-t generative distribution. For a non-terminal prefix x, define

$$
V _ { t } ^ { Y } ( x ) = \mathbb { E } _ { t } [ Y \ |
$$

$$
\boldsymbol { Q } _ { t } ^ { Y } ( \boldsymbol { x } , \boldsymbol { a } ) = V _ { t } ^ { Y } ( \boldsymbol { x } \oplus \boldsymbol { a } ) , \qquad \boldsymbol { A } _ { t } ^ { Y } ( \boldsymbol { x } , \boldsymbol { a } ) = \boldsymbol { Q } _ { t } ^ { Y } ( \boldsymbol { x } , \boldsymbol { a } ) - V _ { t } ^ { Y } ( \boldsymbol { x } ) .
$$

For terminal x, we simply adopt the convention of ${ V _ { t } ^ { Y } } ( x ) = Y ( x )$ . These are finite conditional expectations under the student’s current sampler with its remaining generation budget. Positivity on the fixed allowed autoregressive tree makes them well-defined. Conditioning on the next action gives $\begin{array} { r } { V _ { t } ^ { Y } ( x ) = \sum _ { a } \pi _ { t } ( a  { \mid } x ) \mathsf { \breve { Q } } _ { t } ^ { Y } ( x , a ) } \end{array}$ and $\begin{array} { r } { \sum _ { a } \pi _ { t } ( a \mid x ) \mathcal { A } _ { t } ^ { Y } ( x , a ) = 0 } \end{array}$

Proposition C.18 (Recombining exposure and continuation change). For an integer $0 \leq h \leq H _ { \operatorname* { m a x } }$ let $\mathcal { C } _ { h }$ contain every allowed node at depth h and every terminal node of depth less than h. Then $\begin{array} { r } { \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) = 1 } \end{array}$ , and

$$
\begin{array} { l } { \displaystyle J _ { Y } ( t ) = \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) V _ { t } ^ { Y } ( x ) , } \\ { \displaystyle \dot { J } _ { Y } ( t ) = \sum _ { x \in \mathcal { C } _ { h } } \dot { d } _ { t } ( x ) V _ { t } ^ { Y } ( x ) + \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) \dot { V } _ { t } ^ { Y } ( x ) . } \end{array}\tag{20}
$$

Thefirst derivative term is the sum ofthe outcome contributionsfrom decisions ofdepth less than h; the second is the sum from decisions of depth at least h. In particular, along any path satisfying the regularity conditions,

$$
\dot { J } _ { Y } ( t ) = \sum _ { x \in S } d _ { t } ( x ) \mathbb { E } _ { a \sim \pi _ { t } ( \cdot | x ) } [ \zeta _ { t } ( x , a ) \mathcal { A } _ { t } ^ { Y } ( x , a ) ] .\tag{21}
$$

Under raw flow this equals

$$
\frac { 1 } { \tau } \sum _ { x \in \mathcal { S } } \sum _ { s } d _ { t } ( x ) [ \pi _ { t } \odot \mathcal { A } _ { t } ^ { Y } ] ( x ) ^ { \top } K _ { t } ( x , s ) \Gamma _ { t } ( s ) .
$$

Integrating from t<sub>⋆</sub> to T yields Equation (9) in Section 3.3. If training and evaluation samplers differ, $d _ { t } , \pi _ { t } , \tau$ in this expression are the evaluation quantities, while $\Gamma _ { t }$ uses the training source and field.

Proof. Finite conditional values and the cut. Every trajectory ends by depth $H _ { \mathrm { m a x } }$ . It either reaches depth h or terminates earlier, and therefore meets exactly one element of $\mathcal { C } _ { h }$ . The corresponding reach events are disjoint and exhaustive. Their probabilities sum to one, and conditioning the terminal score on this partition gives the first equality in Equation (20). At a nonterminal node, $V _ { t } ^ { Y } ( x )$ is a finite sum of terminal labels multiplied by products of suffix transition probabilities; at a terminal node it is the fixed label. Thus these values are continuous and piecewise $\dot { C } ^ { 1 }$ . The product rule proves the second equality.

Differentiating a continuation. For each nonterminal x, set

$$
\mathfrak { h } _ { t } ( x ) = \sum _ { a } \pi _ { t } ( a  { \mid } x ) \zeta _ { t } ( x , a ) \mathcal { A } _ { t } ^ { Y } ( x , a ) .
$$

Differentiate the finite recursion $\begin{array} { r } { V _ { t } ^ { Y } ( x ) = \sum _ { a } \pi _ { t } ( a  { \mid } x ) V _ { t } ^ { Y } ( x \oplus a ) } \end{array}$ . By Lemma C.2, $\dot { \pi } _ { t } = \pi _ { t } \zeta _ { t }$ and $\textstyle \sum _ { a } \pi _ { t } \zeta _ { t } = 0$ , so

$$
\dot { V } _ { t } ^ { Y } ( x ) = \mathfrak { h } _ { t } ( x ) + \sum _ { a } \pi _ { t } ( a \mathbin { \mid } x ) \dot { V } _ { t } ^ { Y } ( x \oplus a ) .\tag{*}
$$

At a terminal child the last derivative is zero. For a descendant y of $x ,$ let $d _ { t } ( y \mid x )$ be the product of transition probabilities from x to y, with $d _ { t } ( x \mid x ) = 1$ . Induction on the remaining depth proves

$$
\dot { V } _ { t } ^ { Y } ( x ) = \sum _ { y \in { \cal S } : y \succeq x } d _ { t } ( y \mid x ) \mathfrak { h } _ { t } ( y ) .\tag{**}
$$

Indeed, if all children are terminal, (\*) is precisely (\*\*). Otherwise insert $( ^ { * * } )$ for each nonterminal child into (\*). Its descendant subtrees are disjoint, and multiplying a child’s descendant reach probability by $\pi _ { t } ( a \mid x )$ gives $d _ { t } ( y \mid x )$ . This proves the induction step and counts every remaining decision exactly once.

Decisions after the cut. For a nonterminal y of depth at least h, the ancestry has a unique cut ancestor $x \in \mathcal { C } _ { h }$ . Also $d _ { t } ( x ) d _ { t } ( y \mid x ) = d _ { t } ( y )$ . Multiplying (\*\*) by $d _ { t } ( x )$ and summing over the cut therefore gives

$$
\sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) \dot { V } _ { t } ^ { Y } ( x ) = \sum _ { y \in \mathcal { S } : h ( y ) \geq h } d _ { t } ( y ) \mathfrak { h } _ { t } ( y ) .
$$

Terminal cut nodes have no descendants with decisions and contribute zero.

Decisions before the cut. The reach derivative at a cut node is $\begin{array} { r } { \dot { d } _ { t } ( c ) = d _ { t } ( c ) \sum _ { j < h ( c ) } \zeta _ { t } ( c _ { j } , a _ { j } ) } \end{array}$ Reorder the finite sum $\begin{array} { r } { \sum _ { c \in \mathcal { C } _ { h } } \dot { d } _ { t } ( c ) V _ { t } ^ { Y } ( c ) } \end{array}$ by each ancestral decision $( x , a )$ with $h ( x ) < h$ . The coefficient of $\zeta _ { t } ( x , a )$ is

$$
\sum _ { c \in \mathcal { C } _ { h } : c \succeq x \oplus a } d _ { t } ( c ) V _ { t } ^ { Y } ( c ) = d _ { t } ( x ) \pi _ { t } ( a \mid x ) Q _ { t } ^ { Y } ( x , a ) .
$$

To verify this equality, condition on reaching x and selecting a. The subsequent cut nodes form a disjoint, exhaustive partition of that conditional event, and the conditional expectation of their values is $\bar { Q } _ { t } ^ { Y } ( x , a )$ . This includes a terminal child $x \oplus a ,$ , which is itself in the cut. Summing over a and subtracting $\begin{array} { r } { V _ { t } ^ { Y } ( x ) \sum _ { a } \pi _ { t } ( a  { \mid } x ) \zeta _ { t } ( x , a ) = 0 } \end{array}$ gives

$$
\sum _ { c \in { \mathcal C } _ { h } } { \dot { d } } _ { t } ( c ) V _ { t } ^ { Y } ( c ) = \sum _ { x \in { \mathcal S } : h ( x ) < h } d _ { t } ( x ) { \mathfrak h } _ { t } ( x ) .
$$

Combining the two depth ranges proves Equation (21), including the boundary cases $h = 0$ and $h = H _ { \operatorname* { m a x } }$

Kernel form and integration. Let $a _ { t } ^ { Y } ( x ) = [ \pi _ { t } \odot { \mathcal { A } } _ { t } ^ { Y } ] ( x )$ , extended by zero outside the allowed tokens. Its coordinates sum to zero. Substituting $\zeta _ { t } = \tau ^ { - 1 } ( e _ { a } - \pi _ { t } ) ^ { \top } v _ { t }$ gives

$$
\mathfrak { h } _ { t } ( x ) = \tau ^ { - 1 } \left[ a _ { t } ^ { Y } ( x ) ^ { \top } v _ { t } ( x ) - \left( \sum _ { a } \pi _ { t } ( a \mid x ) A _ { t } ^ { Y } ( x , a ) \right) \pi _ { t } ^ { \top } v _ { t } ( x ) \right] = \tau ^ { - 1 } a _ { t } ^ { Y } ( x ) ^ { \top } v _ { t } ( x ) .
$$

Under raw flow, $\begin{array} { r } { v _ { t } ( x ) = \sum _ { s } K _ { t } ( x , s ) \Gamma _ { t } ( s ) } \end{array}$ , proving the bilinear formula. For an actual optimizer path, the same calculation uses $v _ { t } ( x ) = \Phi _ { t } ( x ) \dot { \theta } _ { t }$ . Integrating the continuous, piecewise differentiable $J _ { Y }$ proves the accumulated outcome formula. □

The suffix derivative in $( ^ { \ast \ast } )$ already includes changes to every downstream prediction. Adding a second derivative of $Q _ { t } ^ { Y }$ to the full decision sum in Equation (21) would just double-count that motion. Conversely, at a fixed cut prefix, $\dot { V } _ { t } ^ { Y } ( x )$ cannot generally be replaced by a derivative with respect to the logits at x alone. Local entropy and terminal correctness are very different observables even though their changes follow the same underlying parameter path.

Reordering the finite source-receiver sum gives the source-resolved form

$$
\dot { J } _ { Y } ( t ) = \sum _ { s } \mu _ { t } ( s ) \varphi _ { Y , t } ( s ) , \qquad \varphi _ { Y , t } ( s ) = \frac { 1 } { \tau } \sum _ { x } d _ { t } ( x ) [ \pi _ { t } \odot \boldsymbol { A } _ { t } ^ { Y } ] ( x ) ^ { \top } K _ { t } ( x , s ) B _ { t } ( s ) .
$$

A source contributes through every coupled receiver, not just through its own continuation value. The experiments in Section E therefore combine prefix dynamics with held-out outcomes.

## C.9 ENTROPY SENSITIVITY AND ANCESTRAL EXPOSURE

Connection to Section 3.3, the entropy-versus-fitting argument. In this subsection $H _ { t } ( x ) =$ $- \textstyle \sum _ { a } p _ { t } ( a \mid x ) \log p _ { t } ( a \mid x )$ is full-vocabulary model entropy. Its policy is $p _ { t } = \mathrm { s o f t m a x } ( z _ { t } )$ at temperature one; the sampling policy in the routing term remains $\pi _ { t }$ at temperature $\tau .$ Fixed-bank evidence is in Section E.3.1.

Proposition C.19 (Entropy sensitivity). Under Equation (15),

$$
\begin{array} { r l } & { R _ { H } ( t ) = \mathrm { C o v } _ { \nu _ { t } } \left( H _ { t } ( x ) , \displaystyle \sum _ { j < h ( x ) } \zeta _ { t } ( x _ { j } , a _ { j } ) \right) , } \\ & { D _ { H } ( t ) = \mathbb { E } _ { \nu _ { t } } \bigl [ \chi _ { H , t } ( x ) ^ { \top } v _ { t } ( x ) \bigr ] , } \\ & { \chi _ { H , t } ( x ) : = \nabla _ { z } H _ { t } ( x ) = - p _ { t } ( \cdot \mid x ) \odot \bigl [ \log p _ { t } ( \cdot \mid x ) + H _ { t } ( x ) \mathbf { 1 } \bigr ] . } \end{array}\tag{22}
$$

At a fixed receiver,

$$
\dot { H } _ { t } ( x ) = - \operatorname { C o v } _ { a \sim p _ { t } ( \cdot \vert x ) } ( \log p _ { t } ( a \mid x ) , v _ { t } ( x , a ) ) , \qquad \vert \dot { H } _ { t } ( x ) \vert \leq \sqrt { \operatorname { V a r } _ { p _ { t } } ( \log p _ { t } ) \operatorname { V a r } _ { p _ { t } } ( v _ { t } ) } .\tag{23}
$$

The parameter-space entropy gradient is $\Phi _ { t } ( x ) ^ { \top } \chi _ { H , t } ( x )$ . These identities hold with the actual logit velocity; the raw specialization is $v _ { t } = K _ { t } \Gamma _ { t }$

Proof. Fix x and suppress it in the notation. Softmax differentiation gives $\dot { p } _ { a } = p _ { a } ( v _ { a } - \bar { v } _ { p } )$ , where $\begin{array} { r } { \bar { v } _ { p } = \sum _ { a } p _ { a } v _ { a } } \end{array}$ . Since $\textstyle \sum _ { a } { \dot { p } } _ { a } = 0$

$$
\begin{array} { l } { \displaystyle \dot { H } = - \sum _ { a } \dot { p } _ { a } ( \log p _ { a } + 1 ) = - \sum _ { a } p _ { a } ( v _ { a } - \bar { v } _ { p } ) \log p _ { a } } \\ { \displaystyle \quad = - \operatorname { C o v } _ { p } ( \log p , v ) = - \sum _ { a } p _ { a } ( \log p _ { a } + H ) v _ { a } . } \end{array}
$$

The last equality uses $\begin{array} { r } { \sum _ { a } p _ { a } \log p _ { a } = - H . } \end{array}$ Since the directional identity holds for every logit direction v, its coefficient vector is $\chi _ { H } = \nabla _ { z } H$ . Weighted Cauchy–Schwarz applied to the centered vectors log $p - { \mathbb { E } } _ { p }$ log p and $v - \mathbb { E } _ { p } v$ proves the variance bound. The chain rule gives the parameter gradient $\Phi ^ { \top } \chi _ { H }$ . Finally substitute the local derivative into Proposition C.11; its ancestral score gives the stated $R _ { H }$ and its fixed-prefix derivative gives $D _ { H }$ □

Proposition C.20 (Ancestral exposure without first-order entropy increase). Consider a fixed path of $h \geq 1$ binary continue/stop decisions. At every gate, let the conditional continuation probability be a differentiable $c _ { t } \in ( 0 , 1 )$ , and hold the suffix executorfixed. Conditional on the path’s starting node, its end is reached with probability $c _ { t } ^ { h }$ . With $H _ { \mathrm { b i n } } ( c ) \stackrel { } { = } - c \log c - ( 1 - c ) \log ( \stackrel { - } { 1 } - c )$

$$
\partial _ { t } \log \operatorname* { P r } _ { t } ( r e a c h d e p t h h ) = h \dot { c } _ { t } / c _ { t } , \qquad \partial _ { t } H _ { \mathrm { b i n } } ( c _ { t } ) = \dot { c } _ { t } \log [ ( 1 - c _ { t } ) / c _ { t } ] .\tag{24}
$$

Proof. The probability of reaching the end is the product of the h conditional gate probabilities, regardless of any independence interpretation. Its logarithm is h log c , giving the first derivative. Direct differentiation yields $H _ { \mathrm { { b i n } } } ^ { \prime } ( c ) \stackrel {  } { = } - \log c - 1 + \bar { \log } ( 1 - c ) + 1 = \log \bar { [ ( 1 - c ) / c ] }$ ; the chain rule gives the second. □

For example, at $c _ { t } = 1 / 2$ with $\dot { c } _ { t } > 0$ , the reach derivative is positive and each gate’s entropy derivative is zero. Thus a pre-existing uncertain suffix can receive more exposure in the rollout distribution without its predictions becoming flatter.

Checking propensity and teacher-relative discrepancy. For the fixed WAIT-initiated family W, let $w _ { t } = p _ { t } ( \mathcal { W } \mid x ) \in ( 0 , 1 )$ ) at a fixed prefix, $b _ { t } = \log [ w _ { t } / ( 1 - w _ { t } ) ]$ , and $q _ { W } = q ( \mathcal { W } \mid x )$ be fixed. Define

$$
c _ { W } ( t ) = D _ { \mathrm { K L } } ( \mathrm { B e r } ( q _ { W } ) \| \mathrm { B e r } ( w _ { t } ) ) .
$$

With the convention 0 log $0 = 0$ at $q _ { W } \in \{ 0 , 1 \}$

$$
\partial _ { t } c _ { W } = ( w _ { t } - q _ { W } ) \partial _ { t } b _ { t } .
$$

Indeed, writing out the two Bernoulli terms $\mathrm { j i v e s } \ \partial _ { w } c _ { W } = - q _ { W } / w + ( 1 - q _ { W } ) / ( 1 - w ) =$ $( w - q _ { W } ) / [ w ( \bar { 1 } - w ) ]$ ]. Also $\partial _ { t } b _ { t } = \dot { w } _ { t } / [ w _ { t } ( 1 - w _ { t } ) ]$ . Multiplication proves the identity, including the endpoint teacher probabilities by the same formula. A decrease in propensity is corrective exactly when its product with the current signed teacher discrepancy is negative; an initially above-teacher class need not remain above the teacher throughout training.

## D CONVERSION: PROTOCOLS AND ADDITIONAL MEASUREMENTS

This appendix gives the experimental details behind Section 3.2. We first describe the model pairs and the fixed prefixes used for measurement. We then ask where the teachers place their strongest requests, how well those requests produce parameter updates, and how the student’s responsiveness changes during training. The derivations are in Sections C.3 and C.4.

## D.1 SETUP

We compare two teachers for the same starting student at each scale. At 1.5B, the student is DeepSeek R1-Distill-Qwen-1.5B and the teachers are JustRL-1.5B and DeepSeek-R1-Distill-Qwen-7B. At 7B, the student is DeepSeek-R1-Distill-Qwen-7B and the teachers are AceReason-Nemotron-7B and DeepSeek-R1-Distill-Qwen-14B. The AIME comparison uses 30 AIME 2024 problems at temperature .6. We draw 16 responses per problem at 1.5B and 4 at 7B.

At each scale, eleven rollouts from the initial student provide the prefixes for measurement. Each local kernel is computed from logit gradients with respect to all student parameters. Please note that, computing the gradients is expensive, the eigendecomposition itself is inexpensive. The kernel measurements therefore use sampled positions, while Ξ serves as a less expensive monitor.

## D.2 WHERE THE STRONGEST REQUESTS OCCUR

We analyze which kinds of positions receive the strongest teacher requests. The annotated rollouts contain 36,416 positions at 1.5B and 38,168 at 7B. For each teacher, the strongest 1% contains 364 and 381 positions, respectively.

Annotations and enrichment. We use Opus 5 to label the sentences in the rollouts. Each sentence receives a semantic category, which describes its role in the reasoning. We also classify positions by their location within a sentence and by token kind. These are three separate classifications of the same positions, examined from different semantic aspects. We use the following prompt to create the semantic labels (this rubric is the result of iterating on the model’s reasoning traces a few times, to settle on the most prevalent reasoning categories the model writes in its chain of thought).

LLM Judge Prompt   
You are annotating every sentence of a model's reasoning trace (for   
example, a model solving a math problem) with a label describing the   
act the sentence performs.   
REASONING TRACE: "{reasoning\_trace}"   
Emit per sentence: {"sent\_idx": int, "fine": [1 label], "uncertain":   
bool, "note": "..."} Read the FULL trace first; judge each   
sentence's FUNCTION IN CONTEXT.   
## Fine labels   
CONTROL family (steering the reasoning process):   
- check-initiate: starting a verification act on ALREADY-ESTABLISHED   
material -- re-computing a prior result, re-reading the problem   
statement, re-deriving an earlier step. First-time testing of a   
fresh candidate or case is NOT check-initiate (see TECHNICAL).   
- check-conclude: verdict closing a check episode. Covers:   
work-correctness verdicts ("Yes, that's correct", "that step was   
wrong"); verdicts about the check itself, including redundancy ("I   
already did this case", "we've already considered that");   
resolutions of a problem re-read that settle interpretation ("So   
it's allowed to be prime").   
- doubt-flag: hesitation OR bare deliberation cue WITHOUT an act   
("Wait.", "Hmm, that seems too big.", "Let me think.", "Let's   
see."). Judged sentence-locally: a bare cue stays doubt-flag even   
when a check follows in the next sentence. Never label bare cues as   
plan.   
- plan: stating an approach or opening a new sub-task, with the   
approach/task STATED ("I'll use coordinates", new-sub-task colon   
lead-ins like "Compute the exponent:"). Includes proceed-decisions   
("Let me try that").   
- reframe: abandoning/switching approach. NOT routine next-case   
moves in an enumeration ("Alternatively, x=2" = TECHNICAL).   
- repair: announcing/making a correction of an earlier error.   
- recap: MULTI-ITEM consolidation of established state used to   
steer. Single value-echoes = compute/derive.   
- commit: FINAL-ANSWER commitment only ("So the answer is 34",   
"Answer: 34").   
TECHNICAL family (object-level mathematics):   
- derive: algebraic/logical derivation steps; this includes stating   
the associated problem facts, notation, coordinates, case setup, and   
within-computation connective lead-ins (including "Simplify:",   
"Comparing coefficients:").   
- recall: invoking a theorem/definition/known mathematical fact.   
OTHER family:   
- style: pure filler/transition with no content and no steering.   
- format: formatting artifacts, the end-of-reasoning marker (e.g.   
</think>), bare list numerals or LaTeX delimiters split onto their   
own lines (format, not fragment).   
- unclear: genuinely unclassifiable.

## Fixed vocabulary and note rules   
Use only the labels above; do not propose or introduce new labels. A   
sentence the rules do not settle or hard to classify takes the best   
available existing label, "uncertain": true, and leave a short note   
-- residual uncertainty is data itself, please do not treat it as a   
defect to legislate away. Furthermore, write in the note for the   
sentence if it genuinely has more than one of the labels above.

For a category $c ,$ enrichment compares its frequency among the teacher’s strongest requests with its frequency among all annotated positions:

$$
E _ { q } ( c ) = \frac { \mathrm { P r } ( c \mid \mathrm { s t r o n g e s t } 1 \% \mathrm { u n d e r } \mathrm { t e a c h e r } q ) } { \mathrm { P r } ( c ) } .
$$

A value above 1 means that the category is more common among the strongest requests. For planning, we split the sentences in this class by position: their first three tokens (the opening of a plan) belong to the control family, and the remaining tokens (the stated approach) belong to the technical family; mixed sentences belong to neither. Panel (a) of Figure 3 takes each teacher’s top 50%, 20%, 10%, 5%, 2% and 1% of requests and shows the share of sentence openings (first three tokens), control positions and technical positions, minus the corresponding share among all annotated positions. Figure 9 shows the raw shares. Bootstrap intervals resample the annotated rollouts. They are computed separately for each teacher. Panels (b, c) of Figure 3 show twelve selected rows, and Figure 10 shows the full set. Table 1 gives the values used in the main figure.

Differences between teachers. At 1.5B, planning openings are enriched 3.59× under the sibling teacher and 0.30× under the larger teacher. Sentence-initial positions are enriched 2.91× and 1.46×, respectively. At 7B, the sibling teacher places relatively more of its strongest requests at reframing, checking, and doubt positions. Wait-family tokens are enriched 4.50× under the sibling and 1.23× under the larger teacher.

The difference depends on the category. At 7B, planning openings are enriched 4.13× and 1.06×, while the stated approach of a plan is enriched 0.72× and 1.70×; the teachers’ derivation enrichment is similar (0.84× and 0.83×). Digit-bearing positions are relatively more prominent under the larger teacher at both scales.

We find that, as the percentile goes up, the sibling teacher’s shares of sentence openings and control positions grow relative to all positions, while the larger teacher’s return to or fall below them (Figure 9). Among the strongest 1%, the sibling teacher’s share of sentence openings exceeds the larger teacher’s by 19.2 [13.1, 27.7] percentage points at 1.5B and 27.6 [18.8, 37.0] at 7B.

<table><tr><td rowspan="2">Position category</td><td colspan="2">1.5B student</td><td colspan="2">7B student</td></tr><tr><td>Sibling</td><td>Larger</td><td>Sibling</td><td>Larger</td></tr><tr><td>Reframe</td><td>1.63</td><td>0.70</td><td>2.61</td><td>0.65</td></tr><tr><td>Check-conclude</td><td>1.52</td><td>0.91</td><td>1.86</td><td>0.62</td></tr><tr><td>Doubt flag</td><td>1.15</td><td>1.84</td><td>2.95</td><td>0.68</td></tr><tr><td>Check-initiate</td><td>1.37</td><td>1.20</td><td>2.64</td><td>1.32</td></tr><tr><td>Planning: opening</td><td>3.59</td><td>0.30</td><td>4.13</td><td>1.06</td></tr><tr><td>Recap</td><td>1.48</td><td>1.11</td><td>0.00</td><td>2.31</td></tr><tr><td>Planning: stated approach</td><td>1.62</td><td>0.96</td><td>0.72</td><td>1.70</td></tr><tr><td>Derivation</td><td>0.78</td><td>0.93</td><td>0.84</td><td>0.83</td></tr><tr><td>Recall</td><td>2.04</td><td>1.89</td><td>0.50</td><td>2.18</td></tr><tr><td>Mixed</td><td>0.98</td><td>1.57</td><td>1.25</td><td>1.25</td></tr><tr><td>Wait-family token</td><td>1.74</td><td>0.43</td><td>4.50</td><td>1.23</td></tr><tr><td>Digit-bearing token</td><td>0.65</td><td>0.85</td><td>0.40</td><td>0.79</td></tr></table>

Table 1: Enrichment values for the main semantic figure. Each value compares the frequency of a category among the strongest requests with its frequency among all annotated positions. Figure 10 shows the bootstrap intervals for each teacher.

![](images/cc1a87a37323afc5ba93e96c80240cac46935cbd8dbc1ab7d3e240019bb7ba28.jpg)

![](images/59da2227a1b5067bfeadcf60583ee3d57a745107d35f81347654cdb76b80e435.jpg)

![](images/0bcb29c20a16a4a0d1d57122d3eb9d6b80d1401eb7b491033d889167d6ffae78.jpg)

![](images/b840e64c99e3b6106f310c70a2ad463e0e65b52e25ed23e4da751ae98065610b.jpg)

![](images/f61ac6c06ed2f87924527c9c8b8ba6411a1bfd82ced087b1a62d25b5ec4a0ed2.jpg)  
(a) 1.5B student

![](images/9e966ee57fb04bf32ad86bfd0cdf79c1d83cf3d0fd884fbf607728aed2eb02d3.jpg)  
(b) 7B student  
Figure 9: Composition of the strongest requests as the percentile rises. Shares of sentence openings (first three tokens), control positions and technical positions among each teacher’s top 50%, 20%, . . . , 1% of requests. Dotted: all annotated positions; bands: 95% bootstrap intervals over the annotated rollouts.

![](images/74017d496c2b13ed7f3cca5b701601f99b00d6b25b281e715a1819796867d810.jpg)  
CONTROL Reframe Check-conclude Doubt flag Check-initiate Planning: opening Recap Repair Commit TECHNICAL Planning: stated approach Derivation Recall OTHER Mixed Other category POSITION Sentence-initial token Sentence-interior token TOKENS Wait-family token Digit-bearing token Other token

![](images/e283609c12fa7dc28c3caa98cd0ad1fd166bc0107c0f6f46f62d12e6594d28aa.jpg)  
enrichment among the teacher's strongest 1% of requests (by request mass; log scale)  
(a) 1.5B student  
(b) 7B student

Figure 10: Where the strongest requests occur. The full set of semantic categories, sentence positions, and token kinds. A ratio of one means that a category has the same frequency among the strongest requests as in the annotated rollouts. Planning sentences are split into their opening and stated approach. Intervals are clipped at the plot edges; left-pointing markers denote estimates below the plotted range, including zero. The main figure selects twelve rows.

## D.3 CONVERSION ESTIMATES BY SEMANTIC CLASS

We now ask which factors in Equation (4) distinguish the healthy and stalled runs. We report pressure $P ,$ , accessibility $A ,$ and coherence $C$ separately for the CONTROL and TECHNICAL samples selected for each teacher. The initial comparison asks how the same starting student responds to the two teachers. The endpoint comparison asks how much disagreement remains and how the student’s responsiveness has changed.

Measurement and estimation. We save the training rollouts and the request vector $B$ computed at every position during training. The initial requests $B _ { 0 }$ used in the frozen-request probes come directly from these records. For a fixed request at prefix $s ,$ the parameter gradient is $\bar { g _ { t } } ( s ) = \Phi _ { t } ( s ) ^ { \top } B ( s )$ We obtain it by backpropagating through $\bar { B ( s ) } ^ { \top } z _ { t } ( s )$ while holding $B ( s )$ constant.

The squared norm of a sample mean includes sampling variance. We correct for this before estimating coherence. For $n > 1$ sampled gradients, write ${ \bar { g } } = n ^ { - 1 } \sum _ { i } g _ { i }$ and ${ \widehat { D } } _ { g } = n ^ { - 1 } \sum _ { i } \| g _ { i } \| ^ { 2 }$ . The correction is

$$
\widehat { C } _ { \mathrm { c o r r } } = \frac { \displaystyle \left( \| \bar { g } \| ^ { 2 } - [ \widehat { D } _ { g } - \| \bar { g } \| ^ { 2 } ] / ( n - 1 ) \right) _ { + } } { \widehat { D } _ { g } } , \qquad \widehat { D } _ { g } > 0 ,
$$

where $( r ) _ { + } = \operatorname* { m a x } ( r , 0 )$ . Under independent sampling, the numerator before truncation is unbiased for the squared norm of the population-mean gradient (Proposition C.9).

<table><tr><td>Run</td><td>Class</td><td> $P _ { 0 }$ </td><td> $A _ { 0 } / 1 0 ^ { 3 }$ </td><td> $C _ { 0 }$ </td><td> $\Xi _ { 0 } / 1 0 ^ { 3 }$ </td><td> $\lVert u _ { 0 } \rVert$ </td></tr><tr><td>7B healthy</td><td>Control</td><td>0.192</td><td>119.8</td><td>0.081</td><td>9.70</td><td>43.1</td></tr><tr><td>7B healthy</td><td>Technical</td><td>0.151</td><td>106.3</td><td>0.055</td><td>5.85</td><td>29.7</td></tr><tr><td>7B stalled</td><td>Control</td><td>0.521</td><td>148.1</td><td>0.025</td><td>3.70</td><td>43.9</td></tr><tr><td>7B stalled</td><td>Technical</td><td>0.599</td><td>128.8</td><td>0.020</td><td>2.58</td><td>39.3</td></tr><tr><td>1.5B healthy</td><td>Control</td><td>6.024</td><td>39.6</td><td>0.041</td><td>1.62</td><td>98.9</td></tr><tr><td>1.5B healthy</td><td>Technical</td><td>2.569</td><td>54.5</td><td>0.064</td><td>3.49</td><td>94.7</td></tr><tr><td>1.5B stalled</td><td>Control</td><td>0.824</td><td>41.3</td><td>0.040</td><td>1.65</td><td>36.9</td></tr><tr><td>1.5B stalled</td><td>Technical</td><td>1.110</td><td>53.8</td><td>0.039</td><td>2.10</td><td>48.2</td></tr></table>

Table 2: Initial conversion estimates by semantic class. Each row uses $n = 8 0$ positions selected for one teacher and class. $P$ and $A$ use measurements at individual positions. $C$ uses the finite-sample correction above. $\Xi$ and $\lVert u \rVert$ are approximate values computed from these estimates. Comparisons are within scale and parameterization.

Initial comparison. Table 2 shows why gradient norm alone does not explain conversion. At 7B, the healthy run has somewhat lower accessibility but 2.75–3.24× higher coherence across the two classes. Its collective gain is 2.27–2.62× higher. The stalled run still has a comparable or larger gradient norm because its request pressure is higher.

At 1.5B, accessibility is similar between the two teachers, and the coherence difference depends on the class. The initial collective gains are nearly equal in CONTROL: 1.62 versus 1.65, in units of $1 0 ^ { 3 }$ In TECHNICAL, the healthy run has the larger gain: 3.49 versus 2.10. Thus its larger gradient norms do not imply a larger conversion gain in every class.

Remaining requests and responsiveness. Table 3 separates two endpoint comparisons. The live conversion factors use the current requests $B _ { t }$ . They measure the response to the remaining disagreements with the teacher. The last column instead reapplies the initial requests $B _ { 0 }$ . With both the requests and prefixes fixed, this column measures how local responsiveness to those requests has changed.

The healthy runs retain only 0.5–1.5% of their initial request energy. The stalled runs retain 60.2– 79.0%. Meanwhile, accessibility to the initial requests increases in both classes of each healthy run and decreases in both classes of each stalled run.

<table><tr><td>Run</td><td>Class</td><td>Step  $P _ { t } / P _ { 0 }$ </td><td></td><td> $A _ { t } / A _ { 0 }$ </td><td> $C _ { t } / C _ { 0 }$ </td><td> $\Xi _ { t } / \Xi _ { 0 }$   $A _ { t } [ B _ { 0 } ] / A _ { 0 } [ B _ { 0 } ]$ </td></tr><tr><td>7B healthy</td><td>Control</td><td>200</td><td>0.005</td><td>0.983</td><td>0.012</td><td>0.012 1.414</td></tr><tr><td>7B healthy</td><td>Technical</td><td>200</td><td>0.007</td><td>2.159</td><td>0.036 0.079</td><td>1.232</td></tr><tr><td>7B stalled</td><td>Control</td><td>150</td><td>0.740</td><td>0.827</td><td>0.520 0.430</td><td>0.810</td></tr><tr><td>7B stalled</td><td>Technical</td><td>150</td><td>0.740</td><td>0.884</td><td>0.500</td><td>0.442 0.889</td></tr><tr><td>1.5B healthy</td><td>Control</td><td>200</td><td>0.009</td><td>1.216</td><td>0.341</td><td>0.415 2.829</td></tr><tr><td>1.5B healthy</td><td>Technical</td><td>200</td><td>0.015</td><td>3.628</td><td>0.031</td><td>0.113 5.399</td></tr><tr><td>1.5B stalled</td><td>Control</td><td>200</td><td>0.602</td><td>0.836</td><td>0.725</td><td>0.606 0.831</td></tr><tr><td>1.5B stalled</td><td>Technical</td><td>200</td><td>0.790</td><td>0.793</td><td>0.564</td><td>0.447 0.789</td></tr></table>

Table 3: Remaining requests and responsiveness at the endpoint. The positions are the same as at initialization. The pressure and live conversion columns use the current requests $B _ { t } ;$ the last column uses the initial requests $B _ { 0 }$ . All diagnostic columns are ratios to their initial values. Endpoints are step 200, except for stalled 7B at step 150. Coherence estimates are rounded, especially at healthy endpoints.

## D.4 HOW THE LOCAL KERNEL CHANGES

To understand the change in responsiveness, we measure both the strength and movement of the kernel’s leading directions. At each prefix s, let $V _ { t , 3 } ( s )$ contain the orthonormal eigenvectors of the symmetrized local kernel $K _ { t } ( s , s )$ with the three largest eigenvalues. Denote those eigenvalues by $\{ \bar { \lambda } _ { j , t } ( s ) \} _ { j = 1 } ^ { 3 }$ . Main Figure 4b plots

$$
G _ { 3 } ( s , t ) = \frac { \sum _ { j = 1 } ^ { 3 } \lambda _ { j , t } ( s ) } { \sum _ { j = 1 } ^ { 3 } \lambda _ { j , 0 } ( s ) } , \qquad O _ { 3 } ( s , t ) = \frac { \| V _ { 0 , 3 } ( s ) ^ { \top } V _ { t , 3 } ( s ) \| _ { F } ^ { 2 } } { 3 } .
$$

$G _ { 3 }$ measures the change in the sum of the three largest eigenvalues. $O _ { 3 }$ measures how much their subspace overlaps with its initial position. It is the mean squared cosine of the three principal angles: 1 means identical subspaces and 0 means orthogonal subspaces. The main figure reports median $\bar { G } _ { 3 }$ and $O _ { 3 }$ over 160 positions; the table below reports $1 - O _ { 3 }$ instead. Moreover, subspace movement alone does not tell us whether the kernel becomes better or worse aligned with the teacher’s requests. For this, we measure the share of the initial request in the leading subspace, $\begin{array} { r } { w _ { 1 : 3 } ( s , t ) = \sum _ { j \leq 3 } | v _ { j , t } ^ { \top } b _ { 0 } | ^ { 2 } } \end{array}$ where $b _ { 0 }$ is the unit direction of the initial request. Its change is $\Delta w _ { 1 : 3 }$ . Figure 4c compares this change for the two teachers’ requests in the same checkpoint kernel.

<table><tr><td>Run</td><td>Step</td><td> $G _ { 3 }$ </td><td> $1 - O _ { 3 }$ </td><td> $\Delta w _ { 1 : 3 }$  (pp) Median</td><td> $a _ { t } / a _ { 0 }$   $A _ { t } [ B _ { 0 } ] / A _ { 0 } [ B _ { 0 } ]$ </td></tr><tr><td>7B healthy</td><td>50</td><td>1.19</td><td>0.034</td><td>-0.23</td><td>1.04</td></tr><tr><td>7B healthy</td><td>200</td><td>1.25</td><td>0.041</td><td>-0.36</td><td>1.23 1.07 1.34</td></tr><tr><td>7B stalled</td><td>50</td><td>0.79</td><td>0.018</td><td>+1.12</td><td>0.93 0.90</td></tr><tr><td>7B stalled</td><td>150</td><td>0.74</td><td>0.032</td><td>+1.51</td><td>0.91 0.85</td></tr><tr><td>1.5B healthy</td><td>50</td><td>1.00</td><td>0.232</td><td>+5.49</td><td>1.19 3.95</td></tr><tr><td>1.5B healthy</td><td>200</td><td>2.31</td><td>0.369</td><td>+5.07</td><td>2.08 3.78</td></tr><tr><td>1.5B stalled</td><td>50</td><td>0.80</td><td>0.022</td><td>+1.06</td><td>0.91 0.86</td></tr><tr><td>1.5B stalled</td><td>200</td><td>0.67</td><td>0.044</td><td>+0.89</td><td>0.83 0.80</td></tr></table>

Table 4: Changes in local geometry. Except in the last column, ratios and differences are computed at each position before taking the median. The last column compares accessibility averaged across positions with weights proportional to initial request energy. The median and weighted average describe different aspects of the same sample. Here pp denotes percentage points.

Early on in training (by step 50), the healthy 1.5B run has already changed its leading subspace, while the leading strength is near its initial value. The strength increases later. The healthy 7B run mainly increases leading strength, with little subspace movement. Both stalled runs lose leading strength.

The table also distinguishes a typical position from an average weighted by request energy. For example, at step 50 of healthy 1.5B, the median sensitivity ratio is 1.19, while the ratio of weighted accessibilities is 3.95. The latter gives more weight to positions with larger initial requests. Their difference reflects the weighting and the way results are combined across positions.

![](images/8ed4da5f52f20ae0a718161dc13b1a08a6f49385c980f2977a1202128b1535ad.jpg)  
(a) Accessibility to the initial requests.

![](images/692457132c53e167ccf6d7678dbe802cf52e505c57100a7e8d1139419ddf3502.jpg)  
(b) Strength and movement of the leading subspace.  
Figure 11: Changes in the kernel and its response to the initial requests. Left: $A _ { t } [ B _ { 0 } ] / A _ { 0 } [ B _ { 0 } ]$ on the combined sample, with equal numbers of positions from the two classes. Right: median leading strength and subspace change across positions. Red/blue denote healthy/stalled runs; circles/squares denote 7B/1.5B.

Figure 12 extends the comparison to the full eigenvalue profile. It compares eigenvalues at the same sorted rank across checkpoints. It is clear that the healthy runs experience a lift in the principal eigenvalues as training proceeds, while the stalled runs experience a decline instead.

![](images/416088f8601c94891347843a911aa21e23c16661c580a68c67b45ff879465cf2.jpg)  
(a) 7B student.

![](images/b7bad33f0e92eada1536882cb4aac4ac452f1b2d269558be3b3868ef0cd7315d.jpg)  
(b) 1.5B student.  
Figure 12: Endpoint eigenvalues relative to initialization. At each sorted rank, we compute the endpoint-to-initial ratio at each position and report the median over 160 positions per run. Red/blue denote healthy/stalled runs. Endpoints are step 200, except for stalled 7B at step 150.

A separate comparison groups initial directions by their request energy. The reported growth is not concentrated in the groups with stronger requests. We therefore distinguish a turn toward the teacher’s requests from preferential growth along the most strongly requested initial directions.

## D.5 ALIGNMENT WITH THE TEACHER

The leading subspace rotates in the healthy 1.5B run. Does this movement favor the teacher that trained the student? We compare the training-run teacher’s initial requests with the other teacher’s initial requests at the same prefixes and token coordinates. Both are evaluated in the same checkpoint kernel. More precisely, at each position, we subtract their changes in projection onto the leading subspace:

$$
d _ { s } ( t ) = \Delta w _ { 1 : 3 } ^ { \mathrm { t r a i n e d } } ( s , t ) - \Delta w _ { 1 : 3 } ^ { \mathrm { o t h e r } } ( s , t ) .
$$

A positive value means that the training teacher’s requests gain more projection. Figure 4c uses step 50 for all four runs. The comparison holds the kernel change fixed and asks which field gains more projection.

<table><tr><td>Run</td><td>Trained gain (pp)</td><td>Other gain (pp)</td><td>Paired difference (pp)</td><td>95% interval</td></tr><tr><td>7B healthy</td><td>-0.23</td><td>+0.07</td><td>-0.16</td><td> $[ - 0 . 3 3 , + 0 . 2 7 ]$ </td></tr><tr><td>7B stalled</td><td>+1.12</td><td>+1.25</td><td>+0.00</td><td> $[ - 0 . 5 3 , + 0 . 6 0 ]$ </td></tr><tr><td>1.5B healthy</td><td>+5.49</td><td>+2.42</td><td>+1.79</td><td> $[ + 0 . 9 4 , + 3 . 1 3 ]$ </td></tr><tr><td>1.5B stalled</td><td>+1.06</td><td>+1.00</td><td>-0.16</td><td> $[ - 0 . 2 9 , + 0 . 0 5 ]$ </td></tr></table>

Table 5: Comparison of the two teachers’ requests at step 50. Values are medians over the same 160 positions. The paired difference is the median of the per-position differences. For each of 3,000 bootstrap samples, we resample the eleven initial rollouts and retain all selected positions in each sampled rollout. The two fields remain paired and the trained checkpoints remain fixed.

The healthy 1.5B run has a median paired difference of 1.79 percentage points, with a 95% interval of [0.94, 3.13]. The other three intervals include zero. These measurements show that there is a “preferential” turn toward the training teacher’s requests in healthy 1.5B. The paired median need not equal the difference between the two separate medians: 5.49 − 2.42 and 1.79 summarize the same positions in different orders. The bootstrap measures variation across the sampled rollouts with the trained checkpoints fixed.

## E BROADCAST: PROTOCOLS AND ADDITIONAL MEASUREMENTS

This appendix gives the experimental details for Section 3.3. We examine how OPD changes prefix visitation and the student’s predictions on fixed prefixes. We then ask whether the student can complete its reasoning from those prefixes and reach a correct answer. The theoretical backbone is derived and presented in Sections C.5 to C.9.

## E.1 MODELS AND MEASUREMENT SETUP

Models and training. All three learners use JustRL-1.5B as the teacher. The strong learner is DeepSeek-R1-Distill-Qwen-1.5B which we also studied in Section 3.2. The unprepared learner is Qwen2.5-1.5B-Instruct. “Prepared Qwen” starts from the same Qwen 1.5B instruct model after four passes of offline distillation on OpenMathInstruct2 (see the preparation section, Section F for more details). The Qwen runs use 200 training steps, 256 rollouts per update, temperature 1.0, and an 8,192-token generation cap. DeepSeek’s generation cap is 4,096 tokens.

Rollout banks. We save 32 rollouts from each batch $\ell \in \{ 1 , 2 1 , 4 1 , \dots , 1 6 1 , 1 8 1 , 1 9 6 \}$ . A bank is this fixed set of rollout texts. Batch ℓ is generated after ℓ − 1 updates. Thus batch 41 appears at update 40 in the routing and drift plots. Checkpoint 0 is each learner’s starting model, including the offline-prepared model for prepared Qwen.

When does the model initiate checking? The WAIT family W contains the 13 tokens that decode to “wait” (after whitespace is removed and text is lowercased). We examine candidate sentence starts: the first response position and positions after a token ending in a period, question mark, exclamation mark, colon, or newline. The student and teacher probabilities of this family are denoted as $w _ { t } ( x ) = p _ { t } ( \mathcal { W } \mid x )$ and $q _ { W } ( x ) = q ( \mathcal { W } \mid x )$ . We measure the mean log $w _ { t }$ across candidate starts. For the above- and below-teacher comparisons, we use log-odds $b _ { t } = \log [ w _ { t } / ( 1 - w _ { t } ) ]$ and $b _ { q } = \log [ q _ { W } / ( 1 - q _ { W } ) ]$

We form the two groups using a reference checkpoint r: checkpoint 0 for DeepSeek and checkpoint 20 for the Qwen learners. We include a prefix only if max $( \dot { w } _ { r } , q _ { W } ) \ge 1 0 ^ { - 3 }$ and $| b _ { r } - b _ { q } | \geq 0 . 5$ The sign of $b _ { r } - b _ { q }$ determines whether the student initially assigns more or less probability than the teacher. Each prefix keeps the same group at every evaluated checkpoint.

We also count emitted WAIT tokens per thousand response tokens, and measure the arithmetic mean of $w _ { t }$ at candidate starts. These quantities answer different questions. The emission rate counts what the model actually generates. Mean probability and mean log probability describe its actual predictions at the selected prefixes.

To assess agreement with the teacher, we use the binary discrepancy $\begin{array} { r l r l } { c _ { W } } & { { } } & { = } & { } \end{array}$ $D _ { \mathrm { K L } } ( \mathrm { B e r } ( q _ { W } ) { \bar { \parallel } } \ \mathrm { B e r } ( w _ { t } ) )$ . At a fixed prefix, $\partial _ { t } c _ { W } ~ = ~ ( w _ { t } ~ - ~ q _ { W } ) \partial _ { t } b _ { t }$ Lowering WAIT probability improves agreement only while the student’s probability is above the teacher’s.
<table><tr><td></td><td colspan="4">Local loss</td><td colspan="4">Conditional entropy</td></tr><tr><td>Learner</td><td>Initial</td><td>Final</td><td> $\textstyle \sum { \widehat { R } }$ </td><td> $\textstyle \sum { \widehat { D } }$ </td><td>Initial</td><td>Final</td><td> $\textstyle \sum { \widehat { R } }$ </td><td> $\textstyle \sum { \widehat { D } }$ </td></tr><tr><td>DeepSeek</td><td>0.263</td><td>0.015</td><td>-0.033</td><td>-0.214</td><td>0.664</td><td>0.707</td><td>-0.049</td><td>0.093</td></tr><tr><td>Unprepared</td><td>0.579</td><td>0.073</td><td>0.474</td><td>-0.980</td><td>0.650</td><td>3.988</td><td>3.666</td><td>-0.327</td></tr><tr><td>Prepared</td><td>0.707</td><td>0.362</td><td>0.108</td><td>-0.453</td><td>2.171</td><td>0.842</td><td>-1.183</td><td>-0.146</td></tr></table>

Table 6: Cumulative routing and drift contributions to loss and entropy. The comparisons run from initialization through batch 161, including evaluations at substitute checkpoints. Values are rounded independently. At full precision, the routing and drift contributions sum to the change from the initial to the final mean.

## E.2 DEEPSEEK: ADJUSTING THE PROBABILITY OF CHECKING

DeepSeek’s local loss falls mainly because its predictions change at fixed prefixes (Table 6). We next ask whether these changes bring its checking probabilities closer to the teacher’s, as“wait”-initiated checking / backtracking is a dominant part of the model’s reasoning traces and observed to initiate sentences in which the model manage its epistemic states (Marjanovic et al., 2026). We use the classes in Section E.1, defined by whether the reference student’s WAIT probability lies above or below the teacher’s.

In the first 512 tokens, the above-teacher class has $\sum \widehat { R } = 0 . 0 3 6$ and $\sum \widehat { D } = - 1 . 8 2 4$ . The belowteacher class has 0.505 and 1.918, respectively. The drift contribution therefore lowers mean checking log-odds in the first class and raises them in the second. The main plot uses filled circles and open triangles for these two classes. Routing can still be large within individual intervals, even when its contributions partly cancel over training.

We check whether this two-sided adjustment reduces the discrepancy from the teacher on a fixed set of prefixes (Table 7). For DeepSeek, the discrepancy falls from 0.0789 to 0.0032 in the above-teacher class and from 0.1915 to 0.0149 in the below-teacher class. Interestingly, the same comparison on held-out prepared-Qwen text gives decreases from about 0.039 to 0.014 and from 0.293 to 0.039, so the improvement even extends beyond DeepSeek’s own generated prefixes.

To examine how closely the changes follow the teacher, we regress the change in student log-odds on the teacher and reference-student log-odds. On DeepSeek’s own prefixes at checkpoint 200, the teacher coefficient is about 0.90, the reference-student coefficient about −0.88, and $\dot { R } ^ { 2 }$ about 0.94. This detailed relation is much weaker on the held-out Qwen prefixes. The fixed-prefix results support improved calibration on both sets, but the closer match to the teacher’s pattern occurs on DeepSeek’s own text.

<table><tr><td>Learner</td><td>Reference class</td><td>Positions</td><td> $c _ { W }$  at reference</td><td> $c _ { W }$  at 200</td><td> $\Delta$  log-odds</td></tr><tr><td>DeepSeek</td><td>Above teacher</td><td>33,010</td><td>0.0789</td><td>0.0032</td><td>-2.073</td></tr><tr><td>DeepSeek</td><td>Below teacher</td><td>11,336</td><td>0.1915</td><td>0.0149</td><td>+1.381</td></tr><tr><td>Prepared Qwen</td><td>Above teacher</td><td>25,794</td><td>0.0276</td><td>0.0580</td><td>-0.606</td></tr><tr><td>Prepared Qwen</td><td>Below teacher</td><td>18,211</td><td>0.5627</td><td>0.3813</td><td>+0.721</td></tr></table>

Table 7: Checking probability on fixed prefixes. We report discrepancy from the teacher, $c _ { W } .$ , and the change in log-odds from the reference checkpoint to checkpoint 200. The reference is checkpoint 0 for DeepSeek and checkpoint 20 for prepared Qwen. Each class contains the same prefixes at both checkpoints. The main-text routing and drift plots instead compare successive banks.

## E.3 UNPREPARED QWEN 1.5B: FITTING ON CHANGING PREFIXES

For unprepared Qwen, the local loss decreases (0.579 → 0.073). The accumulated contributions are $\sum { \widehat { R } } _ { L } = + 0 . 4 7 4$ and $\sum { \widehat { D } } _ { L } = - 0 . 9 8 0$ . Thus changes in predictions at fixed prefixes account for the decrease, while changes in prefix visitation raise the loss. This balance varies during training. In the first comparison at step 20, loss rises by +0.268: routing contributes +0.455, which exceeds the decrease from drift, −0.187. Later on, loss mainly falls through changes dominated by drift, as the main text already pointed out. The large rise in entropy has a different origin: it comes mainly from prefix visitation, as shown in the main text.

![](images/34c7c67ebfe5046cd41aa363e0ea55b86c11bccf07847c0aff3abf0a39a527be.jpg)

![](images/31abbba73242362accfb3a6efbdfcf896d8b8fbff8370f6ac6037e759e8625fa.jpg)  
Figure 13: Local agreement and teacher uncertainty. Left: local loss averaged over tokens in the full recorded batches, through the period with fixed prompt-language composition. Right: teacher entropy on the 32-rollout samples from each batch. Colors match Figures 5 and 14. The teacher is generally less uncertain on DeepSeek and prepared-Qwen prefixes than on unprepared-Qwen prefixes.

## E.3.1 LOSS AND ENTROPY ON THE SAME PREFIXES

The preceding comparisons use batches generated at different training steps. We now hold the prefixes fixed to see what the student learns on the same text. We use the prefixes from batch 41. (The entropy-response identity, how it can change through routing without significant drift, and examples are discussed in the theory presented in Section C.9.)

On these prefixes, student entropy falls from 4.987 nats at initialization to 3.989 at checkpoint 160. The local loss falls from 0.789 to 0.057. Teacher entropy is 2.859 nats, below both student values. The teacher is more uncertain here than on the coherent initial text, but the student still learns sharper predictions on these prefixes.

These measurements answer different questions. The falling loss shows fitting at fixed prefixes. The entropy decomposition shows that the large rise in uncertainty mainly reflects which prefixes the model visits. Note that these are primarily examinations of how the local per-prefix readouts change. To understand how the learnt model’s continuations reach a correct answer we need to probe their continuation advantages which involve non-local readouts, as Equation (21) shows. We examine this next.

## E.4 CONTINUATION VALUE AT THE UNPREPARED STUDENT’S ANSWER BOUNDARIES

We now examine prefixes at which the initial student was about to answer. Holding these prefixes fixed lets us ask two questions: how does training change the probability of answering now, and how successful are the continuations from the two alternatives? This comparison isolates changes at the selected prefixes from changes in prefix visitation.

Prefix selection and continuation conditions. We sample two rollouts per problem from the initial student on 50 held-out MATH-500 problems. “Cutting” the rollout immediately before a final-answer marker (\boxed or Answer:) gives 86 prefixes. We evaluate the initial student, checkpoints 20–200 of the unprepared run, and the JustRL teacher on these same prefixes. For each model, we record the next-token probability mass π on the probe’s answer class.

We draw six continuations per prefix under each of three conditions: the answer condition forces the answer token, the “continue” condition forces the model’s most probable non-answer token, and the “natural” condition samples without forcing either token. All conditions use temperature 0.6, ${ \mathrm { t o p } } { - p } = 0 . 9 5$ , and a cap of 16,384 new tokens. Table 8 reports the fraction of correct continuations.

Branch probabilities and continuation values. Let $Q _ { \mathrm { a n s } }$ and $Q _ { \mathrm { { c o n t } } }$ denote the correctness values of the two forced branches, “answer” vs “continue” (recall that Q denotes the continuation success rate from the given prefix distribution after forcing an action choice). Their probability-weighted probe value is $V = \pi Q _ { \mathrm { a n s } } + ( 1 - \pi ) Q _ { \mathrm { c o n t } }$ . This value’s change relative to the initial student decomposes as

$$
\Delta V = \underbrace { \Delta \pi \left( { { { \bar { Q } } } _ { \mathrm { a n s } } } - { { { \bar { Q } } } _ { \mathrm { c o n t } } } \right) } _ { \mathrm { c h a n g e d ~ b r a n c h ~ p r o b a b i l i t y } } + \underbrace { \bar { \pi } \Delta Q _ { \mathrm { a n s } } + \left( 1 - \bar { \pi } \right) \Delta Q _ { \mathrm { c o n t } } } _ { \mathrm { c h a n g e d ~ b r a n c h ~ v a l u e s } } ,
$$

where bars denote the average of the two endpoint values at the same prefix. The first term describes moving probability between the two branches. It is positive when probability moves toward the more successful branch. The second term describes changes in the branch continuation values. We calculate both terms at each prefix before averaging, and obtain intervals from 2,000 bootstrap draws over prefixes. Note that this split applies to the two forced branches. The most probable non-answer token can differ between models. The continue-branch value therefore includes both that first choice and the later continuation. Natural continuations are sampled separately, and their success rate need not equal V .
<table><tr><td rowspan="2">Evaluator</td><td rowspan="2">log-odds of answering</td><td colspan="3">correctness</td><td colspan="3">change in two-branch value</td></tr><tr><td>answer</td><td>continue</td><td>natural</td><td>∆V</td><td>probability term</td><td>value term</td></tr><tr><td>initial student</td><td>+2.4</td><td>0.30</td><td>0.22</td><td>0.27</td><td></td><td></td><td></td></tr><tr><td>update 20</td><td>-3.9</td><td>0.30</td><td>0.29</td><td>0.29</td><td>+0.01</td><td>-0.03</td><td>+0.05</td></tr><tr><td>update 30</td><td>-4.9</td><td>0.33</td><td>0.32</td><td>0.31</td><td>+0.04</td><td>-0.03</td><td>+0.07</td></tr><tr><td>update 40</td><td>-6.1</td><td>0.29</td><td>0.27</td><td>0.26</td><td>-0.01</td><td>-0.04</td><td>+0.03</td></tr><tr><td>update 50</td><td>-6.7</td><td>0.31</td><td>0.17</td><td>0.21</td><td>-0.09</td><td>-0.08</td><td>-0.02</td></tr><tr><td>update 100</td><td>-7.5</td><td>0.34</td><td>0.03</td><td>0.06</td><td>-0.24</td><td>-0.15</td><td>-0.09</td></tr><tr><td>update 200</td><td>-7.4</td><td>0.39</td><td>0.13</td><td>0.17</td><td>-0.15</td><td>-0.13</td><td>-0.02</td></tr><tr><td>teacher</td><td>-3.5</td><td>0.42</td><td>0.58</td><td>0.60</td><td>+0.31</td><td>+0.04</td><td>+0.27</td></tr></table>

Table 8: Answer probability and continuation success on 86 fixed prefixes. Log-odds are averaged over prefixes. The answer and continue columns use forced tokens; natural uses unconstrained continuations. The final columns give the change in the two-branch value and the contributions from changes in branch probability and branch values. We compute these at each prefix before averaging. At update 100, the bootstrap intervals are $[ - 0 . 3 2 , - 0 . 1 6 ]$ for ∆V, [−0.20, −0.10] for the probability term, and $[ - 0 . 1 4 , - 0 . 0 4 ]$ for the value term. Values are rounded.

How the results change during training. Early on, at update 20, the mean log-odds of answering have fallen from +2.4 to −3.9. Yet the answer and continue branches have similar correctness, 0.30 and 0.29. The early shift toward continuing is therefore not yet accompanied by lower continuation success on these prefixes.

By update 100, the answer branch still succeeds with probability 0.34, but the continue branch succeeds with probability 0.03. Natural continuations succeed with probability 0.06. The 8% answerproducing rate in the main text is for the continue branch. It counts any answer, whereas 3% counts a correct answer. Both terms in the two-branch decomposition are negative at this checkpoint. On balance, probability moves toward the less successful branch. Changes in branch values also reduce the probe value. Success partly recovers by update 200, but natural-continuation correctness remains below its initial value.

The teacher also favors continuing over answering on these prefixes. Its continue branch, however, is more successful than its answer branch (0.58 versus 0.42), and its natural continuations succeed with probability 0.60. This illustrates the role of continuation values in Equation (9). In the probability term, a shift toward continuing helps when continuing is more successful than answering now. These answer-boundary probes examine that relation on fixed prefixes. Section E.7 then uses fixed reference continuers to examine changes in prefix visitation.

## E.5 PREPARED QWEN: HOW CHECKING CHANGES DURING TRAINING

The prepared run improves its held-out score, even though its rollouts change markedly early in training. We first examine the order of these changes. We then ask whether the later decrease in checking also occurs on unchanged prefixes, how the teacher’s requests change, and which representative “calibration errors” remain.

The order of the early changes. Entropy rises early on. At step 20, entropy increases from 2.17 to 4.33 nats. Routing contributes +1.98 and drift +0.18, while local loss falls from 0.707 to 0.173. In the next comparison, from batch 21 to batch 41, loss overshoots to 0.437. Its routing contribution is +0.363 and its drift contribution −0.099. Entropy on batch 41 is already low (about 0.70 nats), and remains low (about 0.6–0.9) on later batches. Thus the loss overshoot follows the initial entropy rise. It mainly reflects a change in prefix visitation, similar to the early phase of the unprepared run.

![](images/d37be579f3efd6be282ae3e5624096f2786704c336d0a31a54b8a544545b8b12.jpg)

![](images/7c31a34d79eff9cf9fd05d671611068655e346f19fe3649c08563f1ae55c80f8.jpg)

![](images/324fefdcb1849800f4ec7b58986a7f12c3c5e22faa9251938dbabaa0488d93a4.jpg)  
Figure 14: Loss and checking during the prepared run. Left: cumulative routing and drift contributions to local loss. Middle: WAIT-family tokens per thousand response tokens, measured in the first 512 tokens. Right: cumulative routing (dashed) and drift (solid) of mean log-WAIT probability at candidate sentence starts in the same window. The last panel decomposes log probability, which is distinct from emitted frequency. Open markers denote substituted evaluator checkpoints.

The WAIT emission rate also rises and then falls (Figure 14, middle). To see whether the model changes its predictions on the same prefixes, we evaluate each checkpoint on the candidate sentence starts in the first 512 tokens of batch 41 (Figure 15, left). Mean WAIT probability rises from 0.0494 at checkpoint 20 to 0.1130 at checkpoint 50. It then falls to 0.0842 at checkpoint 100 and is 0.0874 at checkpoint 160. The later decrease therefore includes a change in predictions on unchanged prefixes. The probability remains above its level at checkpoint 20.

Mean log-WAIT probability gives another view of the change across rollout batches (Figure 14, right). Drift is positive in the first comparison (+0.56) and negative by checkpoint 50 (−1.25). Routing is positive during the expansion (+2.89). Further decreases from drift accumulate through checkpoint 100. Later changes, including much of the decline between batches 121 and 141, mainly come from prefix visitation. These contributions sum to a change in mean log probability, not in emitted frequency. On some other fixed prefix sets, mean log probability and arithmetic mean probability change in opposite directions. We therefore keep these measurements separate.

How the teacher’s requests change. Overall, the requests change from favoring WAIT to suppressing it on average. More specifically, we examine requests at sampled WAIT occurrences in full training batches (Figure 15, right). The mean request is about +0.19 to +0.26 at batches 10–25, with intervals above zero. At batch 50, it is −0.0418 (95% interval [−0.0635, −0.0182]), and 0.528 of the sampled requests are negative. Later means are nearer zero.

![](images/e36005eabde2ea05abfa17839f41a1987d3adac73f45a2df222b6138c84c05e0.jpg)

![](images/e48a2d6eb20f27e74135b421e08d09e49fc612d09b178230dbf2851348054bac.jpg)  
Figure 15: Changes on fixed prefixes and changes in teacher requests. Left: mean WAIT probability at the same sentence starts in the first 512 tokens of batch-41 rollouts, evaluated by each checkpoint. Right: mean request at sampled WAIT occurrences before an answer in full training batches, with the reported 95% intervals. Positive requests favor the sampled WAIT token; negative requests suppress it. The two panels use different prefix sets.

Remaining calibration errors. We finally ask whether these changes bring checking probabilities closer to the teacher’s. The prefix classes are fixed at checkpoint 20, and we compare checkpoints 20 and 200 on the same pooled prefix set (Table 7). The discrepancy increases from 0.0276 to 0.0580 in the above-teacher class. It decreases from 0.5627 to 0.3813 in the below-teacher class. On average, calibration improves only in the initially below-teacher class.

Over-use also grows within the above-teacher subset with $q _ { W } \geq 0 . 0 5$ . Mean student probability rises approximately 0.33 → 0.50 on its own text and 0.29 → 0.46 on held-out text, against teacher probability about 0.15. The changes still depend on teacher probability after accounting for the reference student’s probability, including on held-out prefixes. Thus the student’s adjustments remain related to the teacher, but the calibration improvement appears imperfect and weaker than the DeepSeek run.

## E.6 UNCERTAINTY ACROSS ROLLOUT DEPTHS

We next examine when uncertainty rises and where it appears in the response. Figure 16 uses full training batches from the unprepared Qwen run. Each training update samples 256 rollouts on DAPO prompts at temperature one, with an 8,192-token generation cap.

The left heatmap reports student surprisal: the negative log probability of the token the student generated. The right heatmap reports the teacher’s highest next-token probability at the same prefix. Surprisal uses the sampled token, whereas the entropy measurements above use the full next-token distribution. The heatmaps summarize 512-token windows. Rows index rollout batches, including those labeled “tape step” in the main figure. Batch ℓ follows ℓ − 1 updates. Hatched cells and table dashes indicate that no value is reported there, as the model does not reach that generation length yet.

Table 9 gives two example windows. Between batches 20 and 35, surprisal at positions 1536–2047 rises from 3.24 to 6.47 nats. In the first 512 tokens, it rises only from 0.62 to 0.90. Teacher top-1 probability falls from 0.51 to 0.29 in the deeper window, compared with 0.81 to 0.78 in the front window. Thus the early increase in uncertainty is much larger in the deeper of these two windows. The change is not monotone: deep-window surprisal falls to 3.19 by batch 100. Our main routing and drift analysis ends at batch 161.

![](images/5dd33ce4270b4908beb4522e88aa360281111378ba83011bfaa9af2a8d73071e.jpg)  
(a) Student surprisal for sampled tokens.

![](images/2395963b565a97082f652b3d923a3b04885af0195dff47656c83b7a8aa537ee3.jpg)  
(b) Teacher top-1 probability.

Figure 16: Uncertainty at different response depths during training. Student surprisal and teacher top-1 probability are measured on the same student-generated prefixes, in 512-token windows. Hatched cells have no reported value. The plots describe the observed training trajectory.
<table><tr><td rowspan="2">Batch</td><td colspan="2">Student surprisal (nats)</td><td colspan="2">Teacher top-1 probability</td></tr><tr><td>First 512</td><td>1536-2047</td><td>First 512</td><td>1536-2047</td></tr><tr><td>1</td><td>.85</td><td>一</td><td>.79</td><td>一</td></tr><tr><td>5</td><td>.69</td><td></td><td>.81</td><td></td></tr><tr><td>10</td><td>.58</td><td>.12</td><td>.83</td><td>.94</td></tr><tr><td>15</td><td>.58</td><td></td><td>.83</td><td>一</td></tr><tr><td>20</td><td>.62</td><td>3.24</td><td>.81</td><td>.51</td></tr><tr><td>25</td><td>.66</td><td>4.97</td><td>.81</td><td>.35</td></tr><tr><td>28</td><td>.70</td><td>5.55</td><td>.80</td><td>.31</td></tr><tr><td>30</td><td>.74</td><td>5.71</td><td>.80</td><td>.33</td></tr><tr><td>32</td><td>.83</td><td>6.38</td><td>.79</td><td>.30</td></tr><tr><td>35</td><td>.90</td><td>6.47</td><td>.78</td><td>.29</td></tr><tr><td>40</td><td>1.04</td><td>6.00</td><td>.77</td><td>.27</td></tr><tr><td>50</td><td>1.05</td><td>5.69</td><td>.77</td><td>.27</td></tr><tr><td>60</td><td>.95</td><td>5.20</td><td>.79</td><td>.24</td></tr><tr><td>80</td><td>.65</td><td>3.70</td><td>.80</td><td>.34</td></tr><tr><td>100</td><td>.66</td><td>3.19</td><td>.80</td><td>.43</td></tr><tr><td>150</td><td>1.89</td><td>4.14</td><td>.71</td><td>.20</td></tr><tr><td>200</td><td>3.85</td><td>4.26</td><td>.40</td><td>.17</td></tr></table>

Table 9: Two response windows from the heatmaps. The first column gives the rollout batch. Batch ℓ is generated after ℓ − 1 updates. Student values are sampled-token surprisal. A dash indicates no reported value.

## E.7 FIXED-REFERENCE CONTINUATION EXPERIMENTS

The answer-versus-continue experiment in Section E.4 conducted a case study probing continuation from fixed prefixes. This, however, is not enough in understanding the decline in terminal score of the trained student model. Recall that, at a high level, the broadcast not only affects continuation behaviors of the model, but also changes its prefix visitation probabilities. Therefore, to understand the terminal-score decline during OPD, we must also consider how the prefixes that the student checkpoints generates as training proceeds. At any rollout depth, successful completion depends on two quantities: which prefixes the student reaches, and how successfully it continues from those prefixes. We first express terminal score in terms of these two quantities, then show how the reference-policy experiment separates their changes.

Prefix visitation and continuation. Fix a cut depth $h .$ Recall our notation from before, that $d _ { t } ( x )$ is the probability that checkpoint t reaches prefix $x ,$ and $V _ { t } ^ { Y } ( x )$ is its expected terminal score when it completes that prefix. The set of rollout cuts $\mathcal { C } _ { h }$ includes depth-h prefixes and responses that ended earlier (an earlier completed response just retains its terminal score). Conditioning on $\mathcal { C } _ { h }$ gives

$$
J _ { Y } ( t ) = \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) V _ { t } ^ { Y } ( x ) .
$$

This is just the usual average over the “continuation problems” presented to the student by its own generation. Training changes both the weights $d _ { t }$ and the continuation values $V _ { t } ^ { Y }$ . Comparing each checkpoint’s prefixes using that checkpoint as the continuer therefore only measures their combined effect.

Hold the continuer fixed. Using the initial student as the reference, add and subtract $\begin{array} { r } { \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) V _ { 0 } ^ { Y } ( x ) } \end{array}$ to obtain

$$
J _ { Y } ( t ) - J _ { Y } ( 0 ) = \underbrace { \sum _ { x \in C _ { h } } [ d _ { t } ( x ) - d _ { 0 } ( x ) ] V _ { 0 } ^ { Y } ( x ) } _ { \mathrm { { p r e f i x ~ v i s t a t i o n , e v a l u a t e d b y ~ t h e ~ i n i t i a l ~ c o n t i n u e r } } } + \underbrace { \sum _ { x \in C _ { h } } d _ { t } ( x ) [ V _ { t } ^ { Y } ( x ) - V _ { 0 } ^ { Y } ( x ) ] } _ { \mathrm { { c o n t i n u a t i o n ~ c h a n g e ~ o n t h e ~ s a m e c u r e n t p r e f i x e s } } } .\tag{25}
$$

The first term compares the initial and trained prefix distributions while keeping the continuer fixed. A negative value means that the initial student finds the trained prefixes harder to finish than its own prefixes. The second term keeps the trained prefix distribution fixed and compares the two continuers. A negative value means that the trained student is less successful than the initial student on those same prefixes. Thus the comparison changes the prefix-generating policy first, then the continuing policy.

Our experiment implements the intermediate case directly: checkpoint t generates the first h tokens, and the initial student completes the rest of the rollout. Let us write its expected score as

$$
F _ { t } ^ { 0 } ( h ) = \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) V _ { 0 } ^ { Y } ( x ) .\tag{26}
$$

The two terms in Equation (25) are therefore $F _ { t } ^ { 0 } ( h ) - J _ { Y } ( 0 )$ and $J _ { Y } ( t ) - F _ { t } ^ { 0 } ( h )$ . They can be estimated by averaging completed responses. The first supplies a common measure of prefix difficulty across checkpoints; the second compares continuers on each checkpoint’s own prefix distribution.

Experimental setup and results. The study uses 250 held-out MATH problems, two source rollouts per problem from each of checkpoints 50, 100, and 200, and cuts at 256, 512, and 1,024 response tokens. Each active prefix is continued four times by the generating checkpoint, the initial student, and the teacher. All continuers receive the remainder of an 8,192-token total response budget. The readout is extracted-answer success on the full prefix and continuation; the answer need not be immediately followed by an end-of-turn token. Stopping is recorded separately. The estimator averages continuation scores within each prefix, then over the two source rollouts and uniformly over problems, retaining earlier completed responses as above. This population and budget differ from those of the selected answer-boundary probe in Section E.4. The reported values below are rounded sample means; independently sampled self-continuations estimate the same $J _ { Y } ( t )$ but need not give identical means at different cuts.

Consider the 256-token cut at step 50. The initial student scores 48.4% on its own rollouts and 50.5% when it completes the trained checkpoint’s prefixes. The trained checkpoint scores 39.5% on those same prefixes. The first comparison therefore contributes +2.1 percentage points, while the second contributes −11.0 points, accounting for the total −8.9-point difference. At this cut, the prefixe remain usable by the initial student; the trained student loses performance in completing them.

Table 10 shows that at step 50, the initial continuer already scores lower on deep prefixes: the 1,024- token contrast is −8.1 points. At later checkpoints, the deficit also reaches the shallow cuts, while the 1,024-token deficit remains similar in magnitude. On the 256-token prefixes, the trained-minus-initial continuation contrast remains negative at steps 100 and 200 (−12.1 and −6.2 points). By step 200, both terms in Equation (25) are negative: the initial student finds the new prefixes harder, and the trained student completes those prefixes less successfully than the initial student.

<table><tr><td>Cut depth</td><td>Step 50</td><td>Step 100</td><td>Step 200</td></tr><tr><td>256</td><td>+2.1</td><td>-1.7</td><td>-5.9</td></tr><tr><td>512</td><td>-1.3</td><td>-2.7</td><td>-6.1</td></tr><tr><td>1,024</td><td>-8.1</td><td>-7.6</td><td>-8.5</td></tr></table>

Table 10: Change in initial-policy completion success, 100 $[ F _ { t } ^ { 0 } ( h ) - J _ { Y } ( 0 ) ]$ , in percentage points. The initial student completes prefixes from each trained checkpoint; its own-rollout baseline is 48.4%. These are the values plotted in Figure 6, reported as rounded aggregate point estimates from one training run.

Connection to Equation (9). The same two factors describe the instantaneous dynamics. Differentiating the cut identity gives

$$
\dot { J } _ { Y } ( t ) = \sum _ { x \in \mathcal { C } _ { h } } \dot { d } _ { t } ( x ) V _ { t } ^ { Y } ( x ) + \sum _ { x \in \mathcal { C } _ { h } } d _ { t } ( x ) \dot { V } _ { t } ^ { Y } ( x ) .\tag{27}
$$

Under the regularity assumptions in Proposition C.18, the first term computes the contributions to Equation (9) from decisions before the cut, and the second collects those from the cut onward. Both arise from the same shared parameter update. If the reference is checkpoint t instead of checkpoint 0, and the compared policy is an infinitesimally later checkpoint $t + d t$ , the differences in Equation (25) (divided by elapsed training time) recover the derivative, hence recovering these two terms. Such measurements are unrealistic to implement and too noisy to measure in practice, however, so our experiment instead use the initial policy as the reference continuer to magnify the trend of visited-prefix quality / continuation difficulty.

What does moving the cut measure? Another interesting question is, at a fixed checkpoint $t ,$ moving the rollout cut from $h _ { 1 }$ to $h _ { 2 }$ lets the trained student generate an additional block of tokens before the initial student takes over. A decrease in $F _ { t } ^ { 0 }$ means that this additional trained-student generation leaves the initial continuer less able to reach a correct answer. To connect it to continuation advantages, let us follow the initial student’s value along a trained rollout. For each action decision, $\begin{array} { r } { A _ { 0 } ^ { Y } ( x _ { j } , \overline { { a } } _ { j } ) = V _ { 0 } ^ { Y } ( x _ { j + 1 } ) - V _ { 0 } ^ { Y } ( x _ { j } ) } \end{array}$ . Summing over a block yields a telescoping series over depth of the cuts, then averaging over the rollouts gives

$$
F _ { t } ^ { 0 } ( h _ { 2 } ) - F _ { t } ^ { 0 } ( h _ { 1 } ) = \sum _ { \stackrel { x \in \cal S } { h _ { 1 } \leq h ( x ) < h _ { 2 } } } d _ { t } ( x ) \mathbb { E } _ { a \sim \pi _ { t } ( \cdot | x ) } [ A _ { 0 } ^ { Y } ( x , a ) ] .\tag{28}
$$

Here $s$ contains decision prefixes; completed responses contribute no further decisions and retain their scores. At depth zero, the initial student generates the whole response, so $F _ { t } ^ { 0 } ( 0 ) = J _ { Y } ( 0 )$ . At the total response cap, the trained student has generated the whole response, so $F _ { t } ^ { \tilde { 0 } } ( \dot { H } _ { \operatorname* { m a x } } ) = \dot { J } _ { Y } ( t )$ Consequently, the differences across any partition of the full response sum to $J _ { Y } ( t ) - J _ { Y } ( 0 )$ . Fixed cuts therefore measure aggregate contributions of token blocks to this finite policy comparison, using completed outcomes in place of separate continuation-value estimates at every token.

## F PREPARATION: PROTOCOLS AND ADDITIONAL MEASUREMENTS

## F.1 THREE-FAMILY AIME COMPARISON

Students and teachers. The pairs are Qwen2.5-1.5B-Instruct / JustRL-1.5B, Qwen2.5-7B-Instruct / AceReason-Nemotron-7B, and Llama-3.1-8B-Instruct / Llama-3.1-Nemotron-Nano-8B-v1. Each pair shares a tokenizer vocabulary. The fixed teacher supplies the preparation text, the offline-control targets, and the on-policy log-probabilities. The Llama teacher uses its reasoning mode with the system prompt “detailed thinking on.”

Stage 1: offline preparation on OMI2. Each teacher generates solutions to problems drawn from OpenMathInstruct-2 (Toshniwal et al., 2024), at temperature one with an 8,192-token response cap. The student uses token-level forward KL on the teacher’s recorded top-16 next-token support. The plotted cold start uses 512 optimizer steps of 256 sequences, or 131,072 sequence presentations.

<table><tr><td>Student</td><td>Offline-control steps</td><td>Steps in the OPD arm</td><td>Sequence presentations per arm</td></tr><tr><td>Qwen 1.5B</td><td>200 offline</td><td>200 OPD</td><td>51,200</td></tr><tr><td>Qwen 7B</td><td>100 offline</td><td>100 OPD</td><td>25,600</td></tr><tr><td>Llama 8B</td><td>200 offline</td><td>200 OPD</td><td>51,200</td></tr></table>

Table 11: Comparison budgets from the same prepared checkpoint within each model pair, after 512 OMI2 preparation steps. Sequence presentations include repetitions.

The Qwen 1.5B, 7B and Llama corpora each contain 32,768 teacher solutions, so this is four corpus sweeps.

The offline settings are AdamW, learning rate $1 0 ^ { - 6 } .$ , momentum coefficients (0.9, 0.95), no weight decay, gradient clipping at 1.0. The cold-start sequence limit is 8,192 tokens.

Stage 2: DAPO comparisons. The prompt pool is DAPO-Math-17k (Yu et al., 2025), recorded as 17,398 unique problems. Training uses consecutive blocks of 6,400 rows in the released ordering. Each block contains 6,235–6,318 unique problems.

OPD samples four student responses for each of 64 prompts per step, giving 256 responses and one optimizer step per update. Sampling is at temperature one, capped at 8,192 tokens for Qwen and 14,336 for Llama.

The offline continuation instead uses four teacher-generated solutions for each prompt row in the same blocks: 25,600 sequences per $6 { , } 4 0 0$ -row block. It uses the teacher forward-KL objective and optimizer settings as preparation, at 256 sequences per step. The sequence limits are 8,192 for Qwen and 15,360 for Llama, the latter retaining the reported teacher solutions generated with a 14,336-token response cap. Thus the comparison matches the implemented offline and on-policy recipes in prompt blocks, sequence counts, and total optimizer steps.

Evaluation and sampled coverage. The evaluation pools AIME 2024 and 2025, 30 problems each. Each checkpoint is evaluated using 32 responses per problem from the first sampling batch (batch a), with vLLM, the student’s template, temperature 0.6, top-p = 0.95, and a 16,384-new-token cap. Scoring is lenient: answer extraction uses a boxed answer when present, otherwise an explicit answer marker including “the answer $\mathrm { i } \mathrm { s } ^ { \prime \prime }$

Following Chen et al. (2021), if $c _ { i }$ of the 32 responses solve problem i, we estimate

$$
\widehat { \mathrm { p a s s } @ k } = \frac { 1 } { 6 0 } \sum _ { i = 1 } ^ { 6 0 } \left[ 1 - \frac { \binom { 3 2 - c _ { i } } { k } } { \binom { 3 2 } { k } } \right] , \qquad 1 \leq k \leq 3 2 ,\tag{29}
$$

with a zero numerator when $3 2 - c _ { i } < k .$ In particular, pass@1 is the average observed success rate, and pass@32 is the fraction solved at least once in the sample. A problem never solved in 32 attempts can still lie within the model’s support, and one solved on all 32 attempts can still have success probability below one.

The lower panels plot the empirical survival curve $\begin{array} { r } { 6 0 ^ { - 1 } \sum _ { i } { \bf 1 } \{ c _ { i } / 3 2 \geq r \} } \end{array}$ for $r > 0$ . Its limit as $r \downarrow 0$ is observed coverage and its area on [0, 1] is the average solve rate. At exactly $r = 0$ , the literal weak-inequality expression equals one; the plotted left endpoint represents the positive-threshold limit. These are marginal distributions across problems.

Endpoints and paired problem changes. Table 12 gives the pooled pass@k endpoints and the number of problems at each empirical reliability threshold.

Among problems solved at least once by both the OPD and offline endpoints, mean observed success rates (OPD/offline) are 0.281/0.104 for Qwen 1.5B (3 problems), 0.599/0.455 for Qwen 7B (25), and 0.264/0.159 for Llama (13). These intersections are selected using both arms’ samples, so their means are descriptive. Against the offline endpoint, OPD gains coverage on 8, 11, and 8 problems and loses it on 2, 1, and 1, respectively.

Uncertainty. For each student and year, we resample the 30 problem indices with replacement, keeping the OPD and offline outcomes paired. For pass@1 the per-problem difference is $( c _ { i } ^ { \mathrm { { O P D } } } -$ $c _ { i } ^ { \mathrm { o f f i n e } } ) / 3 2 ;$ for pass@32 it is $\mathbf { 1 } \{ c _ { i } ^ { \mathrm { O P D } } > 0 \} - \mathbf { 1 } \{ c _ { i } ^ { \mathrm { o f f i n e } } > 0 \}$ . Table 13 reports the mean difference and the 2.5th and 97.5th percentiles of its exact conditional bootstrap distribution, computed by convolution of the empirical per-problem difference distribution. This avoids Monte Carlo error in the interval endpoints. The intervals quantify problem-resampling uncertainty conditional on the checkpoints and recorded outcomes.

<table><tr><td>Student</td><td>Arm</td><td>pass@1</td><td>pass@32</td><td>Coverage</td><td> $\geq 1 / 2$ </td><td>All 32</td></tr><tr><td>Qwen 1.5B</td><td>Base</td><td>0.0115</td><td>0.1333</td><td>8</td><td>0</td><td>0</td></tr><tr><td></td><td>Cold start</td><td>0.0099</td><td>0.1333</td><td>8</td><td>0</td><td>0</td></tr><tr><td></td><td>+ offline</td><td>0.0078</td><td>0.0833</td><td>5</td><td>0</td><td>0</td></tr><tr><td></td><td>+ OPD</td><td>0.0281</td><td>0.1833</td><td>11</td><td>0</td><td>0</td></tr><tr><td>Qwen 7B</td><td>Base</td><td>0.0953</td><td>0.3167</td><td>19</td><td>4</td><td>0</td></tr><tr><td></td><td>Cold start</td><td>0.1302</td><td>0.4167</td><td>25</td><td>7</td><td>0</td></tr><tr><td></td><td>+ offline</td><td>0.1906</td><td>0.4333</td><td>26</td><td>10</td><td>1</td></tr><tr><td></td><td>+ OPD</td><td>0.2693</td><td>0.6000</td><td>36</td><td>15</td><td>5</td></tr><tr><td>Llama 8B</td><td>Base</td><td>0.0219</td><td>0.1833</td><td>11</td><td>1</td><td>0</td></tr><tr><td></td><td>Cold start</td><td>0.0385</td><td>0.2333</td><td>14</td><td>1</td><td>0</td></tr><tr><td></td><td>+ offline</td><td>0.0349</td><td>0.2333</td><td>14</td><td>0</td><td>0</td></tr><tr><td></td><td>+ OPD</td><td>0.0693</td><td>0.3500</td><td>21</td><td>3</td><td>0</td></tr></table>

Table 12: Pooled AIME endpoints and observed reliability at 32 samples per problem (batch a). Coverage, at-least-half, and all-32 columns count problems out of 60. Both Llama comparison arms start from the same prepared checkpoint and use the budgets in Table 11.
<table><tr><td>Student</td><td>Year</td><td>∆pass@1 [95% interval]</td><td></td><td>∆pass@32 [95% interval]</td><td></td></tr><tr><td>Qwen 1.5B</td><td>2024</td><td>+0.032</td><td> $[ + 0 . 0 0 5 , + 0 . 0 6 5 ]$ </td><td>+0.200</td><td> $[ + 0 . 0 3 3 , + 0 . 3 6 7 ]$ </td></tr><tr><td rowspan="2">Qwen 7B</td><td>2025</td><td>+0.008</td><td>[-0.004, +0.027]</td><td>+0.000</td><td> $- 0 . 1 0 0 , + 0 . 1 0 0 ]$ </td></tr><tr><td>2024</td><td>+0.086</td><td>[+0.047, +0.130]</td><td>+0.133</td><td> $[ + 0 . 0 3 3 , + 0 . 2 6 7 ]$ </td></tr><tr><td rowspan="2">Llama 8B</td><td>2025</td><td>+0.071</td><td>[+0.033, +0.114]</td><td>+0.200</td><td> $\dot { \cdot } + 0 . 0 3 3 , + 0 . 3 6 7 \dot { \bar { } }$ </td></tr><tr><td>2024</td><td>+0.040</td><td>[+0.005, +0.082]</td><td>+0.100</td><td> $\left[ + 0 . 0 0 0 , + 0 . 2 3 3 \right]$ </td></tr><tr><td></td><td>2025</td><td></td><td>+0.029 [−0.006, +0.069]</td><td>+0.133</td><td> $\dot { [ + 0 . 0 0 0 , + 0 . 3 0 0 ] }$ </td></tr></table>

Table 13: OPD minus offline control at 32 samples per problem (batch a), with 95% paired problembootstrap percentile intervals computed separately within each year. Values are in probability units.

## F.2 PREPARATION-DOSE STUDY

We follow the same preparation (cold start) procedure as before, except we vary the number of passes over the cold-start corpus and study subsequent OPD performance. Evaluation uses the first 250 MATH-500 problems, four responses per problem, temperature 0.6, top-p = 0.95, a 16,384-newtoken cap, and the deployment template. Table 14 reports strict pass@1 and 95% problem-bootstrap intervals with 2,000 resamples.

The 1.5B OPD series uses the step-100 checkpoint. The 7B series uses the best checkpoint evaluated every ten steps within the first 100 OPD steps: steps 30, 50, and 90 for one, two, and four preparation passes, respectively. Its intervals are computed at the selected checkpoint and do not adjust for checkpoint selection.

<table><tr><td>Student</td><td>Passes</td><td>Cold start</td><td></td><td>After OPD</td></tr><tr><td rowspan="3">Qwen 1.5B</td><td>1</td><td>0.459 [0.411, 0.506]</td><td></td><td>0.459 [0.409, 0.509]</td></tr><tr><td>2</td><td>0.457 [0.409, 0.504]</td><td></td><td>0.530 [0.476, 0.585]</td></tr><tr><td>4</td><td>0.471 [0.420, 0.522]</td><td></td><td>0.562 [0.510, 0.614]</td></tr><tr><td rowspan="3">Qwen 7B</td><td>1</td><td>0.683 [0.638, 0.725]</td><td></td><td>0.823 [0.782, 0.862]</td></tr><tr><td>2</td><td>0.693 [0.649, 0.736]</td><td></td><td>0.828 [0.791, 0.864]</td></tr><tr><td>4</td><td>0.711 [0.669, 0.753]</td><td></td><td>0.853 [0.817, 0.887]</td></tr></table>

Table 14: The dose panels of Figure 8: strict MATH-500 pass@1 (first 250 problems, 4 samples per problem) after 1, 2 and 4 preparation passes. The OPD column uses step 100 at 1.5B and the best evaluated checkpoint within 100 steps at 7B. Brackets give 95% problem-bootstrap intervals at each checkpoint.