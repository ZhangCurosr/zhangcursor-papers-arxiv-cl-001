# CAN NATIVE SAFETY TRAVEL? REUSING SAFETY PRETRAINING ACROSS MODELS VIA ON-POLICY DISTILLATION

A PREPRINT

Bingqi Shang<sup>1</sup> Yang Zhang<sup>2</sup> Yiwei Chen<sup>1</sup> Changsheng Wang<sup>1</sup>

Shiyu Chang<sup>3</sup> Bingquan Shen<sup>4</sup> Sijia Liu<sup>1,2</sup>

<sup>1</sup>Michigan State University <sup>2</sup>MIT-IBM Watson AI Lab

<sup>3</sup>University of California, Santa Barbara <sup>4</sup>National University of Singapore

 GitHub

## ABSTRACT

Safety pretraining has emerged as a promising paradigm for embedding safety deeply into a model during pretraining, yielding robustness to subsequent fine-tuning that conventional safety post-training often lacks. We refer to this safety acquired during model pretraining as native safety. Yet native safety is expensive to acquire, confined to the model in which it was pretrained, and often accompanied by lower general utility. In this work, we rethink the practical value of safety-pretrained models and introduce, for the first time, the native safety reuse problem: Can native safety be transferred from a specialized, potentially smaller, safety-pretrained “teacher” to standard-pretrained “students,” enabling them to inherit its robustness while preserving their general utility? We show that native safety can “travel” through on-policy distillation. A standard-pretrained student generates responses to safety-pretraining prompts, while a small safety-pretrained teacher supervises the student’s tokenlevel predictions along its own generated trajectories; cross-tokenizer alignment further enables transfer across model families. The resulting students exhibit substantially greater safety robustness than their safety-post-trained counterparts while retaining higher general utility than the teacher: subsequent fine-tuning raises the average attack success rate (ASR) of the 1.7B student by only 5.00%, compared with 11.58% for safety post-training, and its average utility of 48.20% far exceeds the aligned teacher’s 27.42%. Unlike conventional large-to-small knowledge distillation, our framework enables selective, small-to-large safety transfer: it reuses the robust safety expertise of a smaller pretrained teacher without inheriting its utility limitations. It further enables native safety to “travel” to broader applications, including robust machine unlearning and vision-language model safety alignment. These findings establish native safety as a reusable model property rather than one permanently tied to the originally pretrained model.

## 1 Introduction

Large language models (LLMs) are increasingly deployed in open-ended and high-stakes settings, making their safety a priority. The prevailing approach is safety alignment through post-training, which uses supervised fine-tuning, preference optimization, reinforcement learning, or related techniques to shape the behavior of a pretrained model (Ouyang et al., 2022; Rafailov et al., 2023; Bianchi et al., 2024; Zou et al., 2024; Liu et al., 2025; Jia et al., 2025; Ji et al., 2023a; Shen et al., 2023). Although safety post-training can substantially reduce harmful compliance at the resulting checkpoint, this protection is often fragile. Aligned models remain vulnerable to jailbreaking attacks (Zou et al., 2023; Wei et al., 2023; Chao et al., 2025; Qi et al., 2025). More alarmingly, their safety can deteriorate during subsequent fine-tuning even when the downstream data are entirely benign and unrelated to harmful behavior (Qi et al., 2024; He et al., 2024; Huang et al., 2024a; Wang et al., 2025). Thus, ordinary model adaptation can inadvertently undo safety previously acquired through post-training.

From the perspective of downstream adaptation, the difficulty of preserving post-training safety has been viewed as a fine-tuning-induced misalignment problem (Siddiqui et al., 2026a). However, it also exposes a fundamental limitation of safety post-training itself: the learned safety may remain shallow (Qi et al., 2025, 2024; Tice et al., 2026). Post-training updates an already pretrained model using a comparatively small alignment dataset and limited optimization budget, often inducing only modest changes to its weights (Siddiqui et al., 2026b; Liu et al., 2019). Consequently, the model may learn safe behavior at the surface level without deeply integrating safety into its underlying representations (Qi et al., 2025). Subsequent safety-irrelevant updates can then overwrite this behavior when improving performance on the intended downstream task, such as mathematical reasoning.

![](images/2e8e87d9def846a01b26f41413ccd4686e72d24224a1efc012f414961b07a1d8.jpg)

To overcome the limitations of safety post-training, safety pretraining has recently emerged as a promising paradigm for embedding safety more deeply into a model (Korbak et al., 2023; Maini et al., 2026; Sam et al., 2026; O’Brien et al., 2026; Roland et al., 2026; Stranisci and Hardmeier, 2026; Tice et al., 2026). This paradigm integrates safety-relevant data and learning objectives directly into the pretraining process. Recent studies show that safety-pretrained models preserve their safety substantially better under subsequent fine-tuning than models aligned solely through post-training. We refer to this deeply embedded and adaptation-resilient property as native safety: safety acquired during pretraining, characterized not only by safe behavior at the current checkpoint, but also by robustness to subsequent finetuning.

Figure 1: Overview of proposed native safety reuse. (a) On-policy safety distillation: The standard-pretrained student generates responses to safety prompts, while the safetypretrained teacher supervises its token-level predictions along the generated trajectories, enabling the distilled student to inherit native safety while preserving its general utility. (b) Performance overview: The distilled student (from the standard model SmolLM2-1.7B) is more robust to subsequent fine-tuning on GSM8K than its safety-post-trained counterpart while retaining substantially higher utility than the safety-pretrained teacher (SafeLM-1.7B). Utility is measured by accuracy on MMLU, CommonsenseQA (CSQA), and GSM8K; Safety is measured by the complement of the attack success rate (ASR), i.e., (100 − ASR)%, on HarmBench (HB), TDC, JailbreakBench (JBB), and AdvBench (AB), both before (BF) and after (AF) fine-tuning. Higher values are better on all axes. For a fair comparison, all models, teacher included, are evaluated after the same safety post-training for instruction following and safety alignment.

Despite its promise, safety pretraining faces some practical limitations. First, pretraining a model from scratch is substantially computationally expensive, making it impractical to repeat for every architecture, scale, or application. Second, native safety is confined to the particular model in which it was pretrained and cannot directly benefit the broader ecosystem of open-weight models. Third, stronger native safety may come at the cost of lower general utility (Maini et al., 2026), and thus, directly deploying a safety-pretrained model can sacrifice capabilities already available in standard models. These limitations motivate the central question of this work:

## (Q) Can native safety “travel”—that is, be transferred from a safety-pretrained model and reused by larger, more capable standard models?

We refer to (Q) as the native safety reuse problem. Solving it can improve the value and scalability of safety pretraining: rather than treating native safety as a model-specific property that must be relearned through costly pretraining, we could acquire it once and reuse it across architectures and modalities. Our objective differs fundamentally from conventional knowledge distillation (Hinton et al., 2015; Gou et al., 2021; Kim and Rush, 2016; Hsieh et al., 2023; Gu et al., 2023; Xu et al., 2024; Song and Zheng, 2026), which typically transfers the general capabilities of a large teacher to a smaller student for efficient deployment. Instead, a particularly valuable setting in our work is small-to-large transfer, where a smaller safety-pretrained teacher transfers its native safety expertise to a larger, more capable student without compromising the student’s general utility.

Our central finding is that native safety travels most effectively on-policy: the student generates responses to safetypretraining prompts, and the teacher supervises its token-level predictions along these trajectories, directly correcting the states and failure modes encountered by the student (see Fig. 1 for an overview). To enable reuse across model families, we further address tokenizer mismatch through cross-tokenizer on-policy distillation (Patiño et al., 2025; Niu et al., 2026; Sun et al., 2026), which aligns token sequences and vocabulary distributions before transferring teacher supervision. In our study, this allows the small SafeLM-1.7B teacher (Maini et al., 2026) to improve larger students from another model family, including Llama-3.2-3B and Llama-3-8B (Grattafiori et al., 2024). Beyond LLM safety alignment, we demonstrate native safety reuse in machine unlearning, where it makes unlearning more robust, and in multimodal safety alignment, where a language-based safety-pretrained teacher enhances the safety robustness of a vision-language model (VLM) without modifying its visual components. Our main contributions are summarized as follows:

• We introduce the native safety reuse problem, reframing safety pretraining as a reusable resource for improving other models rather than a costly, model-bound outcome.

• We show that native safety travels effectively through on-policy distillation, requiring both a safety-pretrained teacher and on-policy supervision. Cross-tokenizer alignment further enables small-to-large transfer across model scales and architectures, allowing SafeLM-1.7B to improve larger Llama students without inheriting the teacher’s utility limitations.

• Across multiple safety and general-utility benchmarks, distilled students exhibit greater robustness to subsequent fine-tuning than their safety-post-trained counterparts while retaining higher utility than the safety-pretrained teacher. See Fig. 1(b) for some experimental highlights.

• We extend native safety reuse to robust machine unlearning and VLM safety alignment, demonstrating transfer beyond the teacher’s original architecture and task.

## 2 Related Work

Safety pretraining and robustness to subsequent fine-tuning. Safety pretraining embeds safety into a model by rewriting or filtering its pretraining corpus (Maini et al., 2026; O’Brien et al., 2026; Stranisci and Hardmeier, 2026), controlling the timing and structure of safety data (Sam et al., 2026; Tice et al., 2026; Roland et al., 2026), or conditioning training on preference scores (Korbak et al., 2023). Maini et al. (2026) show that the resulting safety-pretrained model, SafeLM, preserves safety under subsequent fine-tuning, whereas safety introduced through post-training can erode even on benign data (Qi et al., 2024; He et al., 2024; Huang et al., 2024a). A complementary line of work protects post-trained models during fine-tuning (Huang et al., 2024b; Ham et al., 2025) or makes their weights resistant to tampering (Siddiqui et al., 2026b; Wang et al., 2025). These approaches protect only the model being pretrained or defended. In contrast, we ask whether robustness acquired during safety pretraining can be transferred or reused by a standard-pretrained student, and improve the safety robustness of the resulting student under subsequent fine-tuning without additional defenses.

On-policy and cross-tokenizer distillation. On-policy distillation trains a student on its own rollouts under teacher supervision (Lin et al., 2020; Agarwal et al., 2024; Gu et al., 2023; Song and Zheng, 2026). Its effectiveness can depend more on teacher–student proximity than on teacher strength (Li et al., 2026a; Zhu et al., 2026), and can improve by matching the teacher’s top-k distribution rather than only sampled tokens (Fu et al., 2026a). Cross-tokenizer distillation methods align predictive distributions across vocabularies through optimal transport (Boizard et al., 2024), sequence and vocabulary alignment (Patiño et al., 2025), or finer-grained aligned units (Niu et al., 2026; Sun et al., 2026). Prior work primarily studies capability transfer, including weak-to-strong settings in which smaller models supervise larger ones (Burns et al., 2023; Yu et al., 2026). In contrast, we distill safety from a small safety-pretrained teacher to students of equal or larger scale, and study whether the transferred safety survives subsequent fine-tuning.

Safety distillation. Safety post-training can distill refusal behavior from aligned LLMs (Bianchi et al., 2024; Jiang et al., 2024), although off-policy distillation from an aligned teacher may yield a less safe student (Zhang et al., 2025; Jahan and Sun, 2025). On-policy distillation has also recently been applied to safety alignment: supervising the student’s own rollouts lowers the safety tax (Fu et al., 2026b), and restricting the supervision to safety-critical tokens further limits the utility loss (Li et al., 2026b). In these methods the teacher is the student itself under a safety prompt, a constitution, or a steering vector (Fu et al., 2026b; Wen et al., 2026; Li et al., 2026b), so the distilled safety is bounded by that of the prompted student (Zhu et al., 2026). Other approaches employ a smaller corrector to align a larger model (Ji et al., 2024) or use distillation to make unlearning resistant to relearning (Lee et al., 2026). Yet, little work has studied distillation from safety-pretrained models.

## 3 Problem and Motivation: Native Safety through Pretraining

Safety pretraining: A costly yet more robust alternative to safety post-training. As a concrete setup, we consider the safety-pretraining framework of Maini et al. (2026) and adopt their released safety-pretrained model SafeLM-1.7B in our subsequent study (whose standard-pretrained counterpart is SmolLM2-1.7B). Their pipeline constructs a rich safety-pretraining corpus from web documents in the SmolLM2 pretraining corpus (Allal et al., 2025), which are then scored according to their safety risk. Moderately unsafe documents are rephrased as safe, context-rich text (SAFEWEB), whereas highly unsafe documents are transformed into refusal dialogues (REFUSEWEB) and rewritten as educational articles (MORALEDUCATION). By contrast, the standard safety post-training pipeline considered by Maini et al. (2026) relies on a comparatively limited collection of instruction-following and alignment datasets, including ULTRACHAT-200K (Ding et al., 2023), WILDGUARDMIX (Han et al., 2024), and WILDJAILBREAK (Jiang et al., 2024).

Based on the above, we can apply the standard safety post-training to two pretrained base models: the standardpretrained SmolLM2-1.7B and its safety-pretrained counterpart, SafeLM-1.7B. As demonstrated by Maini et al. (2026) and shown in Fig. 2, initializing from the safety-pretrained model (SafeLM-1.7B) produces a substantially more robust safetyaligned model under subsequent fine-tuning on GSM8K (Cobbe et al., 2021). We report the average ASR across four safety benchmarks, HarmBench (Mazeika et al., 2024), TDC (Maloyan et al., 2024), JailbreakBench (Chao et al., 2024), and AdvBench (Zou et al., 2023), before (BF) and after (AF) finetuning (lower is better), together with the average accuracy across three utility benchmarks, MMLU, CommonsenseQA (CSQA), and GSM8K (higher is better); safety robustness is measured by the increase in ASR caused by fine-tuning, $\Delta \mathrm { A S R } = \mathrm { A S R } _ { \mathrm { A F } } - \mathrm { A S R } _ { \mathrm { B F } }$ , where lower is better. Although both aligned models (initialized from the safety-pretrained SafeLM-1.7B and the standard-pretrained SmolLM2-1.7B, respectively) achieve satisfactory safety before fine-tuning, the SmolLM2-1.7B-initialized model exhibits a sharp increase in

![](images/ac52e625b0ee367161454e6feff5911fc8aae949802e54fd0c20158ad83e8c22.jpg)

![](images/407c2ad9fdb335bbecfadd5cc04fd9c1eb2a5bd475b46492339609294b01fcda.jpg)  
Figure 2: Safety pretraining largely preserves safety after fine-tuning (FT) but has lower utility than safety posttraining. Both 1.7B models receive the same safety posttraining; safety is the attack success rate (ASR, %, lower is better) averaged over four safety benchmarks before (BF) and after (AF) fine-tuning on GSM8K, and utility is accuracy (%) on MMLU, CSQA, and GSM8K.

harmful compliance after fine-tuning, whereas the SafeLM-1.7B-initialized model largely preserves its safety. This native-safety advantage is perhaps expected: compared with safety post-training, safety pretraining leverages a substantially larger and richer corpus of safety-relevant data and incorporates safety learning throughout training from scratch. On the downside, the model initialized from the safety-pretrained SafeLM-1.7B exhibits substantially lower general utility than its standard-pretrained SmolLM2-1.7B counterpart, even after the same post-training procedure. This highlights a nontrivial trade-off: safety pretraining can yield robust native safety, but potentially at the cost of weaker general-purpose capabilities.

Problem of our interest: Native safety reuse through knowledge distillation. As motivated above, safety pretraining provides a clear native-safety advantage, but this advantage comes with costs. For example, instilling native safety into each model through pretraining is prohibitively expensive. In addition, safety pretraining produces a model with robust safety expertise at the expense of its general utility, as shown in Fig. 2. These limitations motivate us to ask whether the benefits of safety pretraining can be reused beyond the model in which they were originally acquired.

Thus, we propose the native safety reuse problem: Can the robust safety acquired through an expensive safety-pretraining procedure (even by a relatively small model such as SafeLM-1.7B) be transferred to standard-pretrained student models, including larger and more capable ones, while preserving their general utility? We cast this problem as teacher–student distillation, where the safety-pretrained model serves as the teacher and a standard-pretrained model as the student. That is,

$$
\begin{array} { r l } { \underset { \theta } { \operatorname* { m i n i m i z e } } } & { { } \ \ell _ { \mathrm { d i s t i l } } ( \theta ; \theta _ { \mathrm { t } } , \mathcal { D } ) , \quad \theta _ { \mathrm { t } } \colon \mathrm { s m a l l - s c a l e ~ s a f e t y - p r e t r a i n e d ~ t e a c h e r } } \end{array}\tag{1}
$$

where $\ell _ { \mathrm { d i s t i l l } }$ is a distillation loss to be designed later, and D is the associated distillation dataset. In the next section, we elaborate on how to solve the native safety reuse problem in (1) and identify the distillation principles needed to transfer robust native safety to the student model. For clarity, Tab. 1 summarizes the main safety-training settings considered in this work, including different teacher and student choices and distillation policies that will be examined later.

Table 1: Summary of the methods and their settings considered in this work.
<table><tr><td>Method</td><td>Teacher θt</td><td>Student θ Distillation</td><td></td><td>Safety post-training</td></tr><tr><td>Safety pretraining (SafePre)</td><td>pretrained</td><td>N/A</td><td>N/A</td><td></td></tr><tr><td>Safety post-training (SafePost)</td><td>N/A</td><td>standard</td><td>N/A</td><td></td></tr><tr><td>Safety-post-trained distillation (SafePost-Distill)</td><td>post-trained</td><td>standard</td><td>on-policy</td><td></td></tr><tr><td>Safety-pretrained off-policy distillation (SafePre-OffDistill)</td><td>pretrained</td><td>standard</td><td>off-policy</td><td></td></tr><tr><td>Safety-pretrained on-policy distillation (SafePre-OnDistill)</td><td>pretrained</td><td>standard</td><td>on-policy</td><td></td></tr></table>

## 4 Native Safety “Travels” On-Policy

Distillation principle for safety reuse: Why on-policy rather than off-policy? To instantiate the distillation objective in (1) for native safety reuse, we first determine whether distillation should be off-policy or on-policy. Here, off-policy distillation trains the student on responses generated by the teacher (Hinton et al., 2015; Kim and Rush, 2016; Gou et al., 2021; Hsieh et al., 2023; Xu et al., 2024), whereas on-policy distillation trains on responses sampled from the student’s evolving policy, with the teacher’s next-token distribution supervising the student at each position (Li et al., 2026a; Agarwal et al., 2024; Gu et al., 2023; Lin et al., 2020; Fu et al., 2026a; Zhu et al., 2026; Song and Zheng, 2026; Yu et al., 2026). We prioritize on-policy for two reasons. First, querying the teacher along student-generated prefixes enables its native safety to correct the student’s own continuations, including trajectories that the teacher itself would be unlikely to generate (Agarwal et al., 2024; Fu et al., 2026b). Second, token-level supervision provides the student with the teacher’s full predictive distribution at each position, yielding a richer safety signal than a single sampled teacher response.

Concretely, let $\pi _ { \pmb { \theta } }$ denote the student’s generative policy parameterized by θ, and let $\pi _ { \pmb { \theta } _ { t } }$ denote the frozen teacher policy. For on-policy distillation, the student samples a response $\hat { y } = ( \hat { y } _ { 1 } , \dots , \hat { y } _ { L } ) \sim \pi _ { \pmb { \theta } } ( \cdot | x )$ given a prompt x, where L is the generation length. Both the student and teacher then evaluate the probability given each student-generated prefix $\hat { y } _ { < l }$ at token position l: $p _ { l } ( v ) = \pi _ { \pmb { \theta } } ( v | x , \hat { y } _ { < l } )$ and $q _ { l } ( v ) = \pi _ { \pmb { \theta } _ { \mathrm { t } } } ( v | x , \hat { y } _ { < l } )$ for every token v in their shared vocabulary V. The on-policy loss minimizes the reverse Kullback–Leibler (KL) divergence between the student and teacher distributions along the student-generated trajectory:

$$
\ell _ { \mathrm { o n } } ( \pmb \theta ; \pmb \theta _ { \mathrm { t } } , \mathcal { D } _ { \mathrm { s a f e } } ) = \mathbb { E } _ { \pmb { x } \sim \mathcal { D } _ { \mathrm { s a f e } } } \left[ \sum _ { l } D _ { \mathrm { K L } } ( p _ { l } \| q _ { l } ) \right] ,\tag{2}
$$

where $\begin{array} { r } { D _ { \mathrm { K L } } ( p \Vert q ) = \sum _ { v \in \mathcal { V } } p ( v ) \log ( p ( v ) / q ( v ) ) } \end{array}$ for distributions p and q defined over the same token set, and $\mathcal { D } _ { \mathrm { s a f e } }$ denotes the safety dataset used to distill knowledge from the teacher $\theta _ { \mathrm { t } }$ . It comprises SAFEWEB, REFUSEWEB, and MORALEDUCATION, the safety-pretraining corpora introduced in Maini et al. (2026) to train the safety teacher. Furthermore, we adopt the top-k on-policy strategy (Li et al., 2026a): at each position l, we restrict the divergence to the student’s k most probable token candidates within the vocabulary V and renormalize both $p _ { l }$ and $q _ { l }$ over this set.

In contrast, off-policy distillation evaluates $p _ { l }$ and $q _ { l }$ along teacher-generated rather than student-generated trajectories. That is, replacing ${ \hat { y } } \sim \pi _ { \pmb { \theta } } ( \cdot \mid x )$ with $\hat { y } \sim \pi _ { \pmb { \theta } _ { \mathrm { t } } } ( \cdot \mid x )$ yields the off-policy objective, which otherwise follows the same form as (2) but uses the forward KL divergence ${ \dot { D _ { \mathrm { K L } } } } ( q _ { l } \parallel p _ { l } )$ . Thus, on-policy lets the safety-pretrained teacher supervise prefixes actually encountered by the evolving student, whereas off-policy restricts supervision to trajectories generated by the teacher.

Furthermore, we examine the differences between on-policy and off-policy distillation for native safety reuse through the quantitative and qualitative analyses presented in Fig. 3.

$$
\mathrm { S a f e P o s t \ = \ S a f e P r e \ \Phi \ S a f e P r e - O f f D i s t i l l \ \not = \ S a f e P r e - O n D i s t i l l }
$$

Fig. 3(Left) compares students distilled from the same safety-pretrained teacher SafeLM-1.7B, using either on-policy or off-policy supervision. Here, students are initialized from the same standardpretrained SmolLM2-1.7B. We use its safety post-trained counterpart as the baseline and follow the evaluation setup of Fig. 2, except that safety robustness is measured by $\Delta \mathrm { A S } \mathrm { \bar { R } }$ (the increase in ASR induced by subsequent fine-tuning on GSM8K) while utility is evaluated on MMLU. As shown, on-policy distillation from the safety-pretrained teacher outperforms its off-policy counterpart, achieving a ∆ASR closest to that of SafePre while largely preserving student utility. Indeed, off-policy distillation can yield even weaker safety robustness than the safety-post-training baselin

![](images/fa2b498d2452c96f4ccaf0896c822418fccc14e38d318f093e4b5644542b0a2f.jpg)

![](images/74e485ca98c89339b2a4e61bd34dec468ebc566df62474e1291c5925009e732f.jpg)  
Figure 3: Comparison of on-policy and off-policy distillation from the safetypretrained SafeLM-1.7B teacher to the standard-pretrained SmolLM2-1.7B student, using the teacher after safety post-training (SafePre) and the student’s safety-post-trained counterpart (SafePost) as baselines; method names follow Tab. 1. (Left) Safety robustness versus MMLU utility (accuracy, %, higher is better). Safety robustness is measured by the increase in ASR after fine-tuning (AF) on GSM8K relative to before fine-tuning (BF), i.e., $\Delta \mathrm { A S R } = \mathrm { A S R } _ { \mathrm { A F } } ^ { - } - \mathrm { A S R } _ { \mathrm { B F } }$ (%); lower is better. The evaluation setup follows Fig. 2. (Right) Change in next-token probability, $\Delta P ( \% )$ , of harmful words relative to the base SmolLM2-1.7B, evaluated at the positions where the base model generates these words in its continuations of harmful prompts from JB-COMPLETIONS; shown are five harmful words that both SafePre-OnDistill and SafePre suppress strongly.

Fig. 3(Right) explains this difference by examining how teacher supervision reshapes the student’s token predictions on harmful prompts. Along student-generated prefixes, on-policy distillation shifts substantially more probability mass from harmful continuations toward safe-oriented tokens than off-policy distillation. Specifically, for the harmful tokens shown on the x-axis of Fig. 3(Right), on-policy distillation suppresses their probabilities far more strongly than off-policy distillation, nearly matching SafePre, whereas safety post-training has little effect. This is because off-policy supervises only teacher-preferred trajectories, potentially missing unsafe states encountered by the student. But on-policy instead queries the teacher on student-generated trajectories, letting its native safety directly correct the student’s failure modes.

Extending native safety reuse across model architectures. We next improve the scalability of native safety reuse by extending it to larger students from model families different from that of the safety-pretrained teacher. The central challenge is tokenizer mismatch: the teacher and student may tokenize the same response into different sequences, and their predictive distributions are defined over different vocabularies (Boizard et al., 2024; Niu et al., 2026; Sun et al., 2026). Let $\mathcal { V } _ { \mathrm { s } }$ and $\nu _ { \mathrm { t } }$ denote the student and teacher vocabularies, respectively. In the cross-tokenizer setting, $\smash { \ v { V } _ { \mathrm { s } } \neq \ v { V } _ { \mathrm { t } } }$ whereas (2) assumes a shared vocabulary, $\mathcal { V } _ { \mathrm { s } } = \mathcal { V } _ { \mathrm { t } } = \mathcal { V }$

To address this cross-tokenizer challenge, we adopt General On-policy Logit Distillation (GOLD) (Patiño et al., 2025). GOLD first partitions the student and teacher token sequences into J pairs of aligned groups, with each pair covering the same span of response text. It then merges the token distributions within each group using the chain rule of probability. Specifically, $\tilde { p } _ { j } ( v )$ , for $v \in \mathcal { V } \mathrm { s }$ , represents the student’s probability of emitting token v followed by the remaining generated tokens in group j, while $\tilde { r } _ { j } ( u )$ , for $u \in \mathcal { V } _ { \mathrm { t } } .$ , denotes the corresponding teacher probability under its own tokenization. Here, $j$ indexes aligned group pairs spanning the same text, which replace individual token positions as the units of comparison. Within each group, token probabilities are combined using the chain rule. Tokens sharing identical strings across vocabularies are aligned one-to-one as matched; all others are unmatched. Superscripts m and u restrict $\tilde { p } _ { j }$ and $\tilde { r } _ { j }$ to the corresponding token sets. The resulting cross-tokenizer objective compares matched-token distributions using the Jensen–Shannon (JS) divergence, $D _ { \mathrm { J S } } ( \bar { p } _ { j } ^ { \mathrm { m } } | | \tilde { r } _ { j } ^ { \mathrm { m } } )$ , and unmatched-token distributions using the Universal Logit Distillation (ULD) distance (Boizard et al., 2024), $D _ { \mathrm { U L D } } \big ( \tilde { p } _ { j } ^ { \mathrm { u } } \big | \big | \tilde { r } _ { j } ^ { \mathrm { u } } \big )$ . The latter computes the sum of absolute differences between the two probability vectors after sorting them in descending order and padding the shorter vector with zeros. The GOLD objective is then given by

$$
\ell _ { \mathrm { G O L D } } ( \pmb \theta ; \theta _ { \mathrm { t } } , \mathcal { D } _ { \mathrm { s a f e } } ) = \mathbb { E } _ { \pmb { x } \sim \mathcal { D } _ { \mathrm { s a f e } } } \left[ \sum _ { j } \left( w _ { 1 } \ : D _ { \mathrm { J S } } \big ( \tilde { p } _ { j } ^ { \mathrm { m } } \big | \big | \tilde { r } _ { j } ^ { \mathrm { m } } \big ) + w _ { 2 } \ : D _ { \mathrm { U L D } } \big ( \tilde { p } _ { j } ^ { \mathrm { u } } \big | \big | \tilde { r } _ { j } ^ { \mathrm { u } } \big ) \right) \right] ,\tag{3}
$$

where $w _ { 1 } > 0$ and $w _ { 2 } = 1 - w _ { 1 }$ control the contributions of the matched- and unmatched-token terms, respectively. In principle, cross-tokenizer distillation can instead use vanilla ULD alone, $i . e .$ , setting $w _ { 1 } = 0$ and $w _ { 2 } = 1$ in (3). Compared to GOLD, vanilla ULD pairs student and teacher tokens directly by position, without explicit sequence or vocabulary alignment. Appendix A further explains why ULD alone cannot effectively transfer native safety.

Preserving student utility during distillation. The safety-pretrained teacher is a base model trained on plain text rather than an instruction-tuned model. The target student, however, must retain the instruction-following ability and general capabilities associated with its standard post-training pipeline. We therefore augment the safety-distillation objective, implemented through on-policy distillation or GOLD, with a utility-preservation loss on the student.

Leveraging the post-training pipeline associated with the safety-pretrained teacher in Maini et al. (2026), we introduce the utility-preservation component as the cross-entropy loss on responses from the instruction-tuning dataset $\mathcal { D } _ { \mathrm { { i n s t r } } } ,$ including UltraChat-200k (Ding et al., 2023), WildGuardMix (Han et al., 2024), and WildJailbreak (Jiang et al., 2024), using the student model θ, denoted by $\ell _ { \mathrm { u t i l } } ( \pmb \theta ; \mathcal { D } _ { \mathrm { i n s t r } } )$ . The overall safety-distillation objective in (1) is therefore given by

$$
\ell _ { \mathrm { d i s t i l } } ( \pmb \theta ; \pmb \theta _ { \mathrm { t } } , \mathcal D ) = \alpha \ell _ { \mathrm { o n o r } \mathrm { G O L D } } ( \pmb \theta ; \pmb \theta _ { \mathrm { t } } , \mathcal D _ { \mathrm { s a f e } } ) + ( 1 - \alpha ) \ell _ { \mathrm { u t i l } } ( \pmb \theta ; \mathcal D _ { \mathrm { i n s t r } } ) ,\tag{4}
$$

where $\pmb \theta$ is initialized from the target standard-pretrained student. The safety-distillation term is instantiated as $\ell _ { \mathrm { o n } }$ in (2) when the teacher and student share a tokenizer, and as $\boldsymbol { \ell _ { \mathrm { G O L D } } }$ in (3) when their tokenizers differ. Here, $\alpha \in [ 0 , 1 ]$ balances native safety transfer against utility preservation. We select $\alpha = 0 . 7$ by the safety–utility trade-off observed in our experiments (Fig. A2 in Appendix B).

## 5 Experiments

## 5.1 Experimental Setup

Data-model setup. As in Sec. 3, we use the safety-pretrained SafeLM-1.7B (Maini et al., 2026) as the teacher and consider three standard-pretrained students: SmolLM2-1.7B (Allal et al., 2025) (i.e., the teacher’s standard-pretrained counterpart), and the larger Llama-3.2-3B and Llama-3-8B (Grattafiori et al., 2024) from a different model family, with a different tokenizer from the teacher $( i . e .$ , the cross-tokenizer setting). Safety distillation draws prompts from $\mathcal { D } _ { \mathrm { s a f e } }$ , comprising SAFEWEB, REFUSEWEB, and MORALEDUCATION (Maini et al., 2026). Utility preservation during distillation in (4) uses $\mathcal { D } _ { \mathrm { i n s t r } }$ , comprising UltraChat-200k (Ding et al., 2023), WildGuardMix (Han et al., 2024), and WildJailbreak (Jiang et al., 2024). The safety-post-training baseline (SafePost in Tab. 1) is trained on the same data but initialized from a standard-pretrained base model.

Distillation training and baselines. Our method, SafePre-OnDistill in Tab. 1, trains each student with (4), using on-policy distillation in (2) for the same-tokenizer student and GOLD in (3) for the cross-tokenizer students. In GOLD, the weights in (3) follow its adaptive default: w is the fraction of the teacher’s vocabulary that has a matching token in the student’s vocabulary, and $w _ { 2 } = 1 - w _ { 1 }$ . The weight in (4) is $\alpha = 0 . 7$ for every student, chosen by the trade-off in Fig. A2. All baselines and distillation variants are summarized in Tab. 1. Notably, SafePre denotes the safety-pretrained model after the safety post-training, corresponding to the released SafeLM-1.7B-Instruct checkpoint of Maini et al. (2026). This differs from the base SafeLM-1.7B checkpoint used as our distillation teacher before instruction tuning. Training and evaluation details are in Appendix C.

Evaluation. Safety is measured by the attack success rate (ASR, %) onfour harmful-request benchmarks, HarmBench (HB) (Mazeika et al., 2024), TDC (Maloyan et al., 2024), JailbreakBench (JBB) (Chao et al., 2024), and AdvBench (AB) (Zou et al., 2023), with responses judged by the HarmBench classifier (Mazeika et al., 2024) as in Maini et al. (2026); lower is better. Robustness is measured against subsequent fine-tuning (FT) on GSM8K (Cobbe et al., 2021): every safe model is further fine-tuned for one epoch on GSM8K. Thus, safety is measured before (BF) and after (AF) fine-tuning, denoted by $\mathbf { A S R _ { B F } }$ and $\mathbf { A S R _ { A F } }$ . We also report their difference, ∆ASR, as used in Sec. 3. Utility is measured by accuracy (%) (before GSM8K fine-tuning) on four benchmarks, MMLU (Hendrycks et al., 2020), CommonsenseQA (CSQA) (Talmor et al., 2019), GSM8K, and WinoGrande (WG) (Sakaguchi et al., 2021), with higher being better. Avg. denotes the mean performance across the corresponding safety or utility benchmarks.

## 5.2 Experiment Results

Table 2: Safety distillation from SafeLM-1.7B to SmolLM2-1.7B without tokenizer mismatch. Methods are summarized in Tab. 1, and the proposed method corresponds to SafePre-OnDistill (gray row). Safety is measured by ASR<sub>BF</sub> → ASR<sub>AF</sub> (%) before and after fine-tuning, with lower values indicating better safety. Utility is measured by accuracy (%) on each benchmark. Avg reports the mean within the safety/utility group.
<table><tr><td rowspan="2">Method</td><td colspan="5">Safety  $( \mathrm { A S R } _ { \mathrm { B F } } \to \mathrm { A S R } _ { \mathrm { A F } } , \downarrow )$ </td><td colspan="5">Utility (↑)</td></tr><tr><td>HB</td><td>TDC</td><td>JBB</td><td>AB</td><td>Avg</td><td>MMLU</td><td>CSQA</td><td>GSM8K</td><td>WG</td><td>Avg</td></tr><tr><td>SafePre</td><td>0.62→2.19</td><td>0.00→0.00</td><td>0.00→1.00</td><td>0.00→0.00</td><td>0.16→0.80</td><td>26.56</td><td>21.13</td><td>5.31</td><td>56.67</td><td>27.42</td></tr><tr><td>SafePost</td><td>3.12→15.00</td><td>2.00→20.00</td><td>5.00→17.00</td><td>0.19→4.62</td><td>2.58→14.16</td><td>49.55</td><td>45.78</td><td>31.77</td><td>64.88</td><td>48.00</td></tr><tr><td>SafePost-Distill</td><td>3.12→17.19</td><td>2.00→26.00</td><td>2.00→12.00</td><td>0.00→8.08</td><td>1.78→15.82</td><td>49.42</td><td>45.45</td><td>31.69</td><td>65.90</td><td>48.12</td></tr><tr><td>SafePre-OffDistill</td><td>1.25→12.19</td><td>0.00→26.00</td><td>0.00→16.00</td><td>0.00→7.69</td><td>0.31→15.47</td><td>48.45</td><td>42.51</td><td>29.57</td><td>65.51</td><td>46.51</td></tr><tr><td>SafePre-OnDistill</td><td>0.62→6.88</td><td>0.00→8.00</td><td>0.00→5.00</td><td>0.00→0.77</td><td>0.16→5.16</td><td>48.43</td><td>47.58</td><td>30.71</td><td>66.06</td><td>48.20</td></tr></table>

Same-family safety reuse. Tab. 2 reports results without tokenizer mismatch, where all distillation methods are initialized from SmolLM2-1.7B. Before fine-tuning, all methods achieve low average ASR (0.16–2.58%). After finetuning, however, our SafePre-OnDistill maintains an ASR of 5.16% (the lowest among all students), substantially below SafePost (14.16%), and closest to SafePre. It also achieves the lowest post-FT ASR among students on each of the four safety benchmarks while preserving student utility, with an average accuracy of 48.20% compared with 27.42% for SafePre. By contrast, SafePost-Distill, which uses a safety-post-trained teacher, and SafePre-OffDistill, which uses off-policy supervision, are initially as safe as our method (1.78% and 0.31% averaged ASR, respectively) but deteriorate to 15.82% and 15.47% after fine-tuning, even exceeding SafePost. These results indicate that transferring native safety requires both a safety-pretrained teacher and on-policy supervision, consistent with Fig. 3. Example responses are in Appendix D.

Cross-family safety reuse. Tab. 3 extends native safety distillation to two larger Llama students whose tokenizers differ from the teacher’s. GOLD-based SafePre-OnDistill achieves the lowest post-fine-tuning ASR among student methods (11.57% for the 3B student and 11.86% for the 8B student), outperforming both SafePost and ULD-based distillation. Nevertheless, its robustness remains below that of SafePre, indicating that cross-family transfer weakens, but does not eliminate, the native safety advantage. Explicit tokenizer alignment is critical: for the 3B student, vanilla ULD, which pairs tokens by position without sequence or vocabulary alignment (Appendix A), reaches 23.58% ASR after fine-tuning, even exceeding SafePost. We omit the corresponding 8B ULD experiment because the 3B result already demonstrates the limitation of ULD

Robustness to harmful fine-tuning. In Fig. 4, we evaluate our proposed SafePre-OnDistill in Tab. 2 under a synthetic worst-case harmful FT scenario. Following Huang et al. (2024b), we replace varying fractions of GSM8K training data with unsafe BeaverTails question–answer pairs (Ji et al., 2023b), and evaluate ASR on unsafe BeaverTails test prompts after FT on these harmful–benign data mixtures. As shown, SafePre-OnDistill consistently outperforms SafePost when at most 10% of the FT data are harmful: their ASRs are 21.40% vs. 38.40% at 0% harmful data (i.e., pure GSM8K FT) and 51.00% vs. 57.40% at 10%. At 20% and above, however, their performance converges, whereas SafePre remains substantially safer, reaching only 23.20% ASR at 20% and 40.80% at 40%. Thus, safety reused from a safety-pretrained teacher improves robustness to benign and mildly harmful FT but remains vulnerable to heavily harmful FT, revealing an inherent loss of native robustness during transfer.

Table 3: Cross-family safety distillation from SafeLM-1.7B to Llama-3.2-3B and Llama-3-8B, with rows grouped by student. The distillation objective choice is given in parentheses: GOLD for the proposed method (gray rows) or vanilla ULD as an ablation. Metrics and benchmarks follow Tab. 2.
<table><tr><td rowspan="2">Method</td><td colspan="5">Safety (ASRBF → ASRAF, ↓)</td><td colspan="5">Utility (↑)</td></tr><tr><td>HB</td><td>TDC</td><td>JBB</td><td>AB</td><td>Avg</td><td>MMLU</td><td>CSQA</td><td>GSM8K</td><td>WG</td><td>Avg</td></tr><tr><td>SafePre</td><td>0.62→2.19</td><td>0.00→0.00</td><td>0.00→1.00</td><td>0.00→0.00</td><td>0.16→0.80</td><td>26.56</td><td>21.13</td><td>5.31</td><td>56.67</td><td>27.42</td></tr><tr><td colspan="10">Student: Llama-3.2-3B</td></tr><tr><td>SafePost</td><td>2.19→12.50</td><td>0.00→28.00</td><td>0.00→16.00</td><td>0.00→18.46</td><td>0.55→18.74</td><td>52.54</td><td>65.68</td><td>18.50</td><td>68.43 51.29</td><td></td></tr><tr><td>SafePre-OnDistill (ULD)</td><td>2.19→17.81</td><td>0.00→26.00</td><td>2.00→28.00</td><td>0.00→22.50</td><td>1.05→23.58</td><td>47.65</td><td>62.00</td><td>18.57</td><td>68.27</td><td>49.12</td></tr><tr><td>SafePre-OnDistill (GOLD)</td><td>2.81→8.12</td><td>0.00→12.00</td><td>4.00→15.00</td><td>0.00→11.15</td><td>1.70→11.57</td><td>52.11</td><td>64.37</td><td>19.33</td><td></td><td>68.59 51.10</td></tr><tr><td colspan="10">Student: Llama-3-8B</td></tr><tr><td>SafePost</td><td>1.88→10.94</td><td>0.00→28.00</td><td>0.00→15.00</td><td>0.00→12.69</td><td>0.47→16.66</td><td>52.38</td><td>66.34</td><td>19.11</td><td></td><td>70.48 52.08</td></tr><tr><td>SafePre-OnDistill (GOLD) 1.56→10.00</td><td></td><td>0.00→10.00</td><td>0.00→18.00</td><td>0.19→9.42</td><td>0.44→11.86</td><td>56.76</td><td>68.80</td><td>20.32</td><td></td><td>72.14 54.51</td></tr></table>

Application: Cross-modality safety reuse. In this application, the teacher is the text-only base SafeLM-1.7B and the student is the vision-language model (VLM) Qwen2.5-VL-3B-Instruct after VLGuard safety fine-tuning (Zong et al., 2024); the student is trained using our proposed on-policy distillation plus a vision-language utility loss on the VLGuard training set and on general visual instruction data from LLaVA-1.5 (Liu et al., 2024), while its vision encoder is frozen. Harmful compliance is evaluated using the Safety Mirage pipeline (Chen et al., 2026), with subsequent fine-tuning on ScienceQA used to assess safety robustness (Appendix E). As shown in Tab. 4, safety distillation from SafeLM-1.7B also makes the VLM robust: both safety-aligned models have an ASR of 0% before FT, but after FT on ScienceQA the ASR of VLGuard safety fine-tuning rises to 33.26%, whereas that of our student rises only to 8.82%, while its utility is retained (79.87% against 80.02% on ScienceQA and

![](images/e9a4e8e6a6808176c757be01baec28f3445a27c1ca7bd386e68c85cb89e0792f.jpg)  
Figure 4: Robustness to harmful FT (fine-tuning) against the fraction (%) of GSM8K fine-tuning data replaced with unsafe Beaver-Tails samples. ASR is measured on 500 unsafe BeaverTails test questions and judged by the BeaverTails moderation model. The other setup follows Tab. 2.

74.50% against 64.56% on TextVQA). This indicates that native safety reused from a text-only safety-pretrained model can also protect a VLM against the fragility of its own safety fine-tuning.

Application: Pretrain-induced unlearnability reuse. Beyond native safety, we examine whether pretraining-induced native unlearnability can also be reused. Deep Ignorance (O’Brien et al., 2026) removes hazardous biological knowledge from pretraining data, producing a model that never acquires it and thereby exhibits a robustness advantage analogous to native safety. We apply (4) using the Deep Ignorance model as the teacher, Zephyr-7B-beta (Tunstall et al., 2023) as the knowledgeable student, and WMDP-Bio (Li et al., 2024) as the distillation corpus. We compare against post-hoc student unlearning via Representation Misdirection for Unlearning (RMU) (Li et al., 2024) and evaluate forgetting using

Table 4: Cross-modal native safety reuse from the text-only SafeLM-1.7B teacher to the Qwen2.5-VL-3B-Instruct student. Safety is reported as ASR (%) on harmful VLGuard prompts before and after ScienceQA fine-tuning, following Tab. 2; utility is accuracy (%) on ScienceQA and TextVQA. “Base” denotes the student without additional safety alignment, and “SafePost” refers to the post-training safety alignment on the student using VLGuard training set.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Safety (ASRBF → ASRAF, ↓)</td><td colspan="2">Utility (↑)</td></tr><tr><td>ScienceQA</td><td>TextVQA</td></tr><tr><td>Base</td><td>47.96→46.83</td><td>80.66</td><td>75.50</td></tr><tr><td>SafePost</td><td>0.00→33.26</td><td>80.02</td><td>64.56</td></tr><tr><td>SafePre-OnDistill</td><td>0.00→8.82</td><td>79.87</td><td>74.50</td></tr></table>

WMDP-Bio cloze accuracy (O’Brien et al., 2026) (Appendix F), where lower is better.

As shown in Tab. 5, the distilled student preserves forgetting under subsequent FT nearly as well as the teacher. Although post-hoc unlearning achieves slightly lower WMDP accuracy before FT (28.07% vs. 31.32%), its accuracy rises to 36.25% afterward, indicating substantial knowledge recovery. In contrast, the distilled student’s accuracy increases only to 31.97%, closely matching the stability of the teacher (24.63% to 25.19%). Its average utility (51.43%) lies between those of RMU (56.94%) and the teacher (42.19%).

Table 5: Reusing native unlearnability from the Deep Ignorance teacher to the student Zephyr-7B-beta, compared with post-hoc RMU unlearning and the teacher itself. Forgetting and its robustness before (BF) and after (AF) GSM8K fine-tuning are measured by WMDP-Bio cloze accuracy (%; lower is better); utility follows Tab. 2.
<table><tr><td rowspan="2">Method</td><td rowspan="2">Forgetting (↓) (WMDP-Bio)</td><td colspan="5">Utility (↑)</td></tr><tr><td>MMLU</td><td>CSQA</td><td>GSM8K</td><td>WG</td><td>Avg</td></tr><tr><td>Deep Ignorance pretraining</td><td>24.63→25.19</td><td>43.33</td><td>49.30</td><td>10.61</td><td>65.51</td><td>42.19</td></tr><tr><td>Post-hoc unlearning (RMU)</td><td>28.07→36.25</td><td>57.40</td><td>65.77</td><td>32.45</td><td>72.14</td><td>56.94</td></tr><tr><td>Unlearning distillation</td><td>31.32→31.97</td><td>52.80</td><td>64.05</td><td>17.89</td><td></td><td>70.96 51.43</td></tr></table>

Fig. 5 tracks the distilled and RMU-unlearned models across GSM8K fine-tuning epochs, where higher values indicate stronger forgetting (100−WMDP accuracy) and better fine-tuning performance (GSM8K accuracy), respectively; other settings follow Tab. 5. The RMU-unlearned model rapidly recovers the removed knowledge within the first epoch, whereas the distilled student maintains nearly its initial forgetting level throughout fine-tuning.

## 6 Conclusion, Limitations, and Future Work

![](images/611caa448a41272362bc4d0ae21b1b2fdab44fb477e0cb0a7165f2a233cbd990.jpg)  
GSM8K accuracy (%) ↑

We ask whether safety acquired during pretraining can be reused by standardpretrained students. On-policy distillation from a safety-pretrained teacher yields students that retain better robustness after subsequent fine-tuning than their safety-post-trained counterparts, while attaining higher general utility than the teacher. Cross-tokenizer alignment extends the approach to larger students from another model family, and application studies suggest benefits for VLM safety and pretraining-induced native unlearnability.

Figure 5: Unlearning robustness of Zephyr-7B-beta, unlearned via RMU or our distillation, against GSM8K finetuning epochs.

The limited availability of safety-pretrained models restricts our study to a single teacher. Although native safety transfers across model scales, architectures, and modalities, the distilled students do not fully recover the teacher’s robustness: a gap remains under cross-family transfer, and the advantage over safety post-training vanishes under heavily harmful fine-tuning. In addition, on-policy distillation requires access to safety-pretraining prompts, while our VLM and unlearning results serve as initial cross-domain application studies.

Future work could investigate more diverse and larger safety-pretrained teachers, improve transfer fidelity across tokenizers and under strongly harmful adaptation, and evaluate robustness to broader model modifications, including continual instruction tuning, preference optimization, model merging, pruning, and quantization. Other promising directions include logit-free native safety reuse, principled teacher and data selection, a theoretical characterization of transferability, and broader validation in multimodal safety and machine unlearning.

## Ethics Statement

Safety behavior can change after a model is fine-tuned. This work studies whether safety acquired during pretraining can be reused by other models while retaining robustness to such adaptation. The language-model evaluation uses harmful requests from HarmBench, TDC, JailbreakBench, and AdvBench; the harmful fine-tuning evaluation uses unsafe BeaverTails examples; the vision-language application uses harmful VLGuard requests; and the unlearning application uses WMDP-Bio, which contains hazardous biological knowledge. Findings about when safety degrades or forgotten knowledge recovers are dual-use: they can inform defenses, but could also help identify vulnerable models or adaptation settings. Our results describe behavior under the tested protocols, not a guarantee of deployment safety or complete removal of hazardous knowledge.

## Reproducibility Statement

Source code for distillation training, subsequent fine-tuning, and evaluation will be released upon publication. The experimental setup, including models, training data, baselines, and evaluation, is given in Sec. 5. Every experiment uses publicly available models (SafeLM-1.7B, SmolLM2-1.7B, Llama-3.2-3B, Llama-3-8B, Qwen2.5-VL-3B-Instruct, Zephyr-7B-beta, and the Deep Ignorance model) and public datasets and benchmarks; no proprietary data or closedsource models are needed. Evaluation follows the protocols of the respective benchmark papers.

## Acknowledgment

The work of B. Shang, Y. Chen, C. Wang, S. Chang, B. Shen, and S. Liu was supported by a research award from DSO National Laboratories. The contributions of B. Shang, Y. Zhang, and S. Liu were also supported in part by the IARPA BENGAL Program. The work of B. Shang, Y. Chen, C. Wang, and S. Liu was additionally supported by the U.S. National Science Foundation (NSF) under CISE Core Award IIS-2504263 and CAREER Award IIS-2338068, the Coefficient Giving AI Safety Research Award, and the Schmidt Sciences Trustworthy AI Award.

## References

L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., “Training language models to follow instructions with human feedback,” Advances in neural information processing systems, vol. 35, pp. 27 730–27 744, 2022.

R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in neural information processing systems, vol. 36, pp. 53 728–53 741, 2023.

F. Bianchi, M. Suzgun, G. Attanasio, P. Röttger, D. Jurafsky, T. Hashimoto, and J. Y. Zou, “Safety-tuned llamas: Lessons from improving the safety of large language models that follow instructions,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 34 196–34 216.

A. Zou, L. Phan, J. Wang, D. Duenas, M. Lin, M. Andriushchenko, R. Wang, Z. Kolter, M. Fredrikson, and D. Hendrycks, “Improving alignment and robustness with circuit breakers,” Advances in Neural Information Processing Systems, vol. 37, pp. 83 345–83 373, 2024.

S. Liu, Y. Yao, J. Jia, S. Casper, N. Baracaldo, P. Hase, Y. Yao, C. Y. Liu, X. Xu, H. Li et al., “Rethinking machine unlearning for large language models,” Nature Machine Intelligence, vol. 7, no. 2, pp. 181–194, 2025.

J. Jia, N. Baracaldo, and S. Liu, “Beyond sft: Reinforcement learning for safer large reasoning models with better reasoning ability,” arXiv preprint arXiv:2512.01848, 2025.

J. Ji, T. Qiu, B. Chen, B. Zhang, H. Lou, K. Wang, Y. Duan, Z. He, L. Vierling, D. Hong et al., “Ai alignment: A comprehensive survey,” arXiv preprint arXiv:2310.19852, 2023.

T. Shen, R. Jin, Y. Huang, C. Liu, W. Dong, Z. Guo, X. Wu, Y. Liu, and D. Xiong, “Large language model alignment: A survey,” arXiv preprint arXiv:2309.15025, 2023.

A. Zou, Z. Wang, N. Carlini, M. Nasr, J. Z. Kolter, and M. Fredrikson, “Universal and transferable adversarial attacks on aligned language models,” arXiv preprint arXiv:2307.15043, 2023.

A. Wei, N. Haghtalab, and J. Steinhardt, “Jailbroken: How does llm safety training fail?” Advances in neural information processing systems, vol. 36, pp. 80 079–80 110, 2023.

P. Chao, A. Robey, E. Dobriban, H. Hassani, G. J. Pappas, and E. Wong, “Jailbreaking black box large language models in twenty queries,” in 2025 IEEE Conference on Secure and Trustworthy Machine Learning (SaTML). IEEE, 2025, pp. 23–42.

X. Qi, A. Panda, K. Lyu, X. Ma, S. Roy, A. Beirami, P. Mittal, and P. Henderson, “Safety alignment should be made more than just a few tokens deep,” in International Conference on Learning Representations, vol. 2025, 2025, pp. 54 911–54 941.

X. Qi, Y. Zeng, T. Xie, P.-Y. Chen, R. Jia, P. Mittal, and P. Henderson, “Fine-tuning aligned language models compromises safety, even when users do not intend to!” in International Conference on Learning Representations, vol. 2024, 2024, pp. 30 988–31 043.

L. He, M. Xia, and P. Henderson, “What is in your safe data? identifying benign data that breaks safety,” arXiv preprint arXiv:2404.01099, 2024.

T. Huang, S. Hu, F. Ilhan, S. F. Tekin, and L. Liu, “Harmful fine-tuning attacks and defenses for large language models: A survey,” arXiv preprint arXiv:2409.18169, 2024.

C. Wang, Y. Zhang, J. Jia, P. Ram, D. Wei, Y. Yao, S. Pal, N. Baracaldo, and S. Liu, “Invariance makes llm unlearning resilient even to unanticipated downstream fine-tuning,” arXiv preprint arXiv:2506.01339, 2025.

S. A. Siddiqui, E. Triantafillou, D. Krueger, and A. Weller, “Position: Capability control should be a separate goal from alignment,” arXiv preprint arXiv:2602.05164, 2026.

C. Tice, P. Radmard, S. Ratnam, A. Kim, D. Africa, and K. O’Brien, “Alignment pretraining: Ai discourse causes self-fulfilling (mis) alignment,” arXiv preprint arXiv:2601.10160, 2026.

S. A. Siddiqui, A. Weller, D. Krueger, G. K. Dziugaite, M. Mozer, and E. Triantafillou, “From dormant to deleted: Tamper-resistant unlearning through weight-space regularization,” Advances in Neural Information Processing Systems, vol. 38, pp. 129 326–129 357, 2026.

H. Liu, M. Long, J. Wang, and M. I. Jordan, “Towards understanding the transferability of deep representations,” arXiv preprint arXiv:1909.12031, 2019.

T. Korbak, K. Shi, A. Chen, R. V. Bhalerao, C. Buckley, J. Phang, S. R. Bowman, and E. Perez, “Pretraining language models with human preferences,” in International conference on machine learning. PMLR, 2023, pp. 17 506–17 533.

P. Maini, S. Goyal, D. Sam, A. Robey, Y. Savani, Y. Jiang, A. Zou, M. Fredrikson, Z. Lipton, and Z. Kolter, “Safety pretraining: Toward the next generation of safe ai,” Advances in Neural Information Processing Systems, vol. 38, pp. 43 971–44 008, 2026.

D. Sam, S. Goyal, P. Maini, A. Robey, and J. Z. Kolter, “When should we introduce safety interventions during pretraining?” arXiv preprint arXiv:2601.07087, 2026.

K. O’Brien, S. Casper, Q. Anthony, T. Korbak, R. Kirk, X. Davies, I. Mishra, G. Irving, Y. Gal, and S. R. Biderman, “Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight llms,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 35 579–35 633.

E. Roland, M. Cubuktepe, E. Martinez, S. Servaes, K. Pepper, M. Vaiana, D. S. de Lucena, J. Rosenblatt, A. Foote, C. Anil et al., “Modular pretraining enables access control,” arXiv preprint arXiv:2607.08077, 2026.

M. A. Stranisci and C. Hardmeier, “What are they filtering out? an experimental benchmark of filtering strategies for harm reduction in pretraining datasets,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, 2026, pp. 39 303–39 313.

G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

J. Gou, B. Yu, S. J. Maybank, and D. Tao, “Knowledge distillation: A survey,” International journal of computer vision, vol. 129, no. 6, pp. 1789–1819, 2021.

Y. Kim and A. M. Rush, “Sequence-level knowledge distillation,” in Proceedings of the 2016 conference on empirical methods in natural language processing, 2016, pp. 1317–1327.

C.-Y. Hsieh, C.-L. Li, C.-K. Yeh, H. Nakhost, Y. Fujii, A. Ratner, R. Krishna, C.-Y. Lee, and T. Pfister, “Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes,” in Findings of the association for computational linguistics: ACL 2023, 2023, pp. 8003–8017.

Y. Gu, L. Dong, F. Wei, and M. Huang, “Minillm: On-policy distillation of large language models,” arXiv preprint arXiv:2306.08543, 2023.

X. Xu, M. Li, C. Tao, T. Shen, R. Cheng, J. Li, C. Xu, D. Tao, and T. Zhou, “A survey on knowledge distillation of large language models,” arXiv preprint arXiv:2402.13116, 2024.

M. Song and M. Zheng, “A survey of on-policy distillation for large language models,” arXiv preprint arXiv:2604.00626, 2026.

C. M. Patiño, K. Rasul, Q. Gallouédec, B. Burtenshaw, S. Paniego, V. Srivastav, T. Frere, E. Beeching, L. Tunstall, L. von Werra, and T. Wolf, “Unlocking on-policy distillation for any model family,” https://huggingface.co/spaces/ HuggingFaceH4/on-policy-distillation, 2025.

Y. Niu, H. Xiao, D. Liu, Z. Wang, D. Gong, Y. Wang, and J. Li, “Breaking the tokenizer barrier: On-policy distillation across model families,” arXiv preprint arXiv:2606.09456, 2026.

J. Sun, M. Zheng, M. Song, Q. Zhong, Y. Cheng, B. Feng, P. Liu, J. Fang, and X. Wang, “Simct: Recovering lost supervision for cross-tokenizer on-policy distillation,” arXiv preprint arXiv:2605.07711, 2026.

A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan et al., “The llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

T. Huang, S. Hu, and L. Liu, “Vaccine: Perturbation-aware alignment for large language models against harmful fine-tuning attack,” Advances in Neural Information Processing Systems, vol. 37, pp. 74 058–74 088, 2024.

S. Ham, Y. Choi, Y. Yang, S. Cho, Y. Kim, and C. Kim, “Safety-aligned weights are not enough: Refusal-teacherguided finetuning enhances safety and downstream performance under harmful finetuning attacks,” arXiv preprint arXiv:2506.07356, 2025.

A. Lin, J. Wohlwend, H. Chen, and T. Lei, “Autoregressive knowledge distillation through imitation learning,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020, pp. 6121–6133.

R. Agarwal, N. Vieillard, Y. Zhou, P. Stanczyk, S. Ramos Garea, M. Geist, and O. Bachem, “On-policy distillation of language models: Learning from self-generated mistakes,” in International Conference on Learning Representations, vol. 2024, 2024, pp. 21 246–21 263.

Y. Li, Y. Zuo, B. He, J. Zhang, C. Xiao, C. Qian, T. Yu, H.-a. Gao, W. Yang, Z. Liu et al., “Rethinking on-policy distillation of large language models: Phenomenology, mechanism, and recipe,” arXiv preprint arXiv:2604.13016, 2026.

S. Zhu, X. Ye, H. Lu, W. Shi, and G. Liu, “The many faces of on-policy distillation: Pitfalls, mechanisms, and fixes,” arXiv preprint arXiv:2605.11182, 2026.

Y. Fu, H. Huang, K. Jiang, J. Liu, Z. Jiang, Y. Zhu, and D. Zhao, “Revisiting on-policy distillation: Empirical failure modes and simple fixes,” arXiv preprint arXiv:2603.25562, 2026.

N. Boizard, K. E. Haddad, C. Hudelot, and P. Colombo, “Towards cross-tokenizer distillation: the universal logit distillation loss for llms,” arXiv preprint arXiv:2402.12030, 2024.

C. Burns, P. Izmailov, J. H. Kirchner, B. Baker, L. Gao, L. Aschenbrenner, Y. Chen, A. Ecoffet, M. Joglekar, J. Leike et al., “Weak-to-strong generalization: Eliciting strong capabilities with weak supervision,” arXiv preprint arXiv:2312.09390, 2023.

F. Yu, W. Xu, M. Xu, T. Zhou, and Z. Lin, “Weak-to-strong on-policy distillation,” arXiv preprint arXiv:2607.26246, 2026.

L. Jiang, K. Rao, S. Han, A. Ettinger, F. Brahman, S. Kumar, N. Mireshghallah, X. Lu, M. Sap, Y. Choi et al., “Wildteaming at scale: From in-the-wild jailbreaks to (adversarially) safer language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 47 094–47 165, 2024.

M. Zhang, D. Liu, K. Zhang, J. Franco, and H. Liu, “Response-based knowledge distillation for multilingual jailbreak prevention unwittingly compromises safety,” arXiv preprint arXiv:2602.11157, 2025.

S. Jahan and R. Sun, “Black-box behavioral distillation breaks safety alignment in medical llms,” arXiv preprint arXiv:2512.09403, 2025.

Y. Fu, L. Yu, H. S. Shahgir, Z. Wei, H. Liu, N. B. Erichson, and Y. Dong, “Reducing the safety tax in llm safety alignment with on-policy self-distillation,” arXiv preprint arXiv:2605.15239, 2026.

H. Li, J. An, Z. Song, P. Zhu, R. Li, H. Wang, W. Feng, Y. Liu, L. Li, J.-G. Yao et al., “Safesteer: Localized on-policy distillation for efficient safety alignment,” arXiv preprint arXiv:2606.02530, 2026.

M. Wen, Y. Liu, K. Yang, Y. Feng, Z. Xu, Y. Sun, S. Cui, X. Zheng, Y. Liu, X. Ma et al., “Constitutional on-policy safe distillation,” arXiv preprint arXiv:2606.03089, 2026.

J. Ji, B. Chen, H. Lou, D. Hong, B. Zhang, X. Pan, J. Dai, T. Qiu, and Y. Yang, “Aligner: Efficient alignment by learning to correct,” Advances in Neural Information Processing Systems, vol. 37, pp. 90 853–90 890, 2024.

B. W. Lee, A. Foote, A. Infanger, L. Shor, H. Kamath, J. Goldman-Wetzler, B. Woodworth, A. Cloud, and A. Turner, “Distillation robustifies unlearning,” Advances in Neural Information Processing Systems, vol. 38, pp. 88 615–88 649, 2026.

L. B. Allal, A. Lozhkov, E. Bakouch, G. M. Blázquez, G. Penedo, L. Tunstall, A. Marafioti, H. Kydlícek, A. P. Lajarín,ˇ V. Srivastav et al., “Smollm2: When smol goes big–data-centric training of a small language model,” arXiv preprint arXiv:2502.02737, 2025.

N. Ding, Y. Chen, B. Xu, Y. Qin, S. Hu, Z. Liu, M. Sun, and B. Zhou, “Enhancing chat language models by scaling high-quality instructional conversations,” in Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 3029–3051.

S. Han, K. Rao, A. Ettinger, L. Jiang, B. Y. Lin, N. Lambert, Y. Choi, and N. Dziri, “Wildguard: Open one-stop moderation tools for safety risks, jailbreaks, and refusals of llms,” Advances in neural information processing systems, vol. 37, pp. 8093–8131, 2024.

K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano et al., “Training verifiers to solve math word problems,” arXiv preprint arXiv:2110.14168, 2021.

M. Mazeika, L. Phan, X. Yin, A. Zou, Z. Wang, N. Mu, E. Sakhaee, N. Li, S. Basart, B. Li et al., “Harmbench: A standardized evaluation framework for automated red teaming and robust refusal,” arXiv preprint arXiv:2402.04249, 2024.

N. Maloyan, E. Verma, B. Nutfullin, and B. Ashinov, “Trojan detection in large language models: Insights from the trojan detection challenge,” arXiv preprint arXiv:2404.13660, 2024.

P. Chao, E. Debenedetti, A. Robey, M. Andriushchenko, F. Croce, V. Sehwag, E. Dobriban, N. Flammarion, G. J. Pappas, F. Tramer et al., “Jailbreakbench: An open robustness benchmark for jailbreaking large language models,” Advances in Neural Information Processing Systems, vol. 37, pp. 55 005–55 029, 2024.

D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt, “Measuring massive multitask language understanding,” arXiv preprint arXiv:2009.03300, 2020.

A. Talmor, J. Herzig, N. Lourie, and J. Berant, “Commonsenseqa: A question answering challenge targeting commonsense knowledge,” in Proceedings ofthe 2019 Conference ofthe North American Chapter ofthe Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), 2019, pp. 4149–4158.

K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi, “Winogrande: An adversarial winograd schema challenge at scale,” Communications of the ACM, vol. 64, no. 9, pp. 99–106, 2021.

J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, B. Chen, R. Sun, Y. Wang, and Y. Yang, “Beavertails: Towards improved safety alignment of llm via a human-preference dataset,” Advances in Neural Information Processing Systems, vol. 36, pp. 24 678–24 704, 2023.

Y. Zong, O. Bohdal, T. Yu, Y. Yang, and T. Hospedales, “Safety fine-tuning at (almost) no cost: A baseline for vision large language models,” arXiv preprint arXiv:2402.02207, 2024.

H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved baselines with visual instruction tuning,” in 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 2024, pp. 26 286–26 296.

Y. Chen, Y. Yao, Y. Zhang, B. Shen, G. Liu, and S. Liu, “Safety mirage: How spurious correlations undermine vlm safety fine-tuning and can be mitigated by machine unlearning,” in International Conference on Learning Representations, vol. 2026, 2026, pp. 78 527–78 547.

L. Tunstall, E. Beeching, N. Lambert, N. Rajani, K. Rasul, Y. Belkada, S. Huang, L. Von Werra, C. Fourrier, N. Habib et al., “Zephyr: Direct distillation of lm alignment,” arXiv preprint arXiv:2310.16944, 2023.

N. Li, A. Pan, A. Gopal, S. Yue, D. Berrios, A. Gatti, J. D. Li, A.-K. Dombrowski, S. Goel, L. Phan et al., “The wmdp benchmark: Measuring and reducing malicious use with unlearning,” arXiv preprint arXiv:2403.03218, 2024.

## A Cross-Tokenizer Sequence Alignment

Fig. A1 compares how ULD and GOLD align the token sequences of the same refusal. The teacher tokenizes “ammonium” as one token, while the student uses “ammon” and “ium”; subsequent ULD pairs therefore cover different text, and the longer student sequence is truncated to the teacher’s length. At each paired position, ULD independently sorts the teacher’s probability vector over $\nu _ { \mathrm { t } }$ and the student’s over $\mathcal { V } _ { \mathrm { s } }$ in descending order, pads the shorter vector with zeros, and computes an $L _ { 1 }$ distance between probabilities of the same rank (Boizard et al., 2024). ULD uses probability ranks without matching token identities. GOLD aligns groups covering the same span of response text. Within each group, it applies $D _ { \mathrm { J S } }$ to matched-token distributions and $D _ { \mathrm { U L D } }$ to unmatched-token distributions, as defined in Eq. (3).

![](images/889ccfbc60c5bdaaf391d8da5e51d747a936a19b4efb1233d21b6bd6031f7694.jpg)  
Figure A1: Sequence alignment across tokenizers on the refusal “I cannot help you build a bomb with ammonium nitrate.”, tokenized by the teacher’s tokenizer (SafeLM-1.7B) and the student’s (Llama-3.2-3B); the shared prefix is omitted. ULD pairs tokens by position, so the pairs are wrong from the first word that the two tokenizers split differently, and the student’s trailing tokens are dropped. GOLD first merges tokens into groups covering the same text and then compares the groups, summing the log-probabilities of the student’s pieces within each group. The quantitative comparison is in Tab. 3.

## B Ablation of the Distillation Weight

Fig. A2 provides a representative example: for the Llama-3.2-3B student distilled from SafeLM-1.7B, removing the utility-preservation loss $( \alpha = 1 )$ decreases utility from 51.10% to 47.35% and increases ∆ASR from 9.87% to 13.49%. By contrast, $\alpha = 0 . 7$ achieves the strongest tested safety robustness while preserving general utility.

![](images/0029a7ecb08d2683ebf777cfb22785f8be0a3dda597fed8442ba8a6a933c40bf.jpg)  
Figure A2: Safety (evaluated by $\Delta \mathrm { A S R } )$ and utility of the student (Llama-3.2-3B) distilled from SafeLM-1.7B across different values of α. ∆ASR (left axis) follows Fig. 3, while utility (right axis) is the average accuracy across 4 utility benchmarks.

## C Additional Training Details

## C.1 Distillation Training Details

Eq. (4) jointly optimizes the safety-distillation loss $\ell _ { \mathrm { o n } }$ or $\boldsymbol { \ell } _ { \mathrm { G O L D } }$ on prompts $x \sim \mathcal { D } _ { \mathrm { s a f e } }$ and the utility-preservation loss $\ell _ { \mathrm { u t i l } }$ on $\mathcal { D } _ { \mathrm { i n s t r } }$ . The student samples on-policy rollouts ${ \hat { y } } \sim \pi _ { \pmb { \theta } } ( \cdot \mid x )$ , and the frozen teacher $\pi _ { \theta _ { \mathrm { t } } }$ provides token-level supervision along these rollouts.

We use a learning rate of $1 0 ^ { - 5 }$ . Student optimization uses AdamW with learning-rate warmup followed by cosine decay. An iteration denotes one optimizer step after gradient accumulation. Each prompt is capped at 256 tokens. For the same prompt x, the student independently samples two responses from its current policy π (· | x). Each response is an on-policy rollout with at most 256 new tokens. For the same-size student, the on-policy divergence is restricted to the student’s top-k candidate tokens at each generated token position, with $k = 1 6 .$ . For cross-tokenizer students, we use the GOLD loss defined in Eq. (3). The teacher remains frozen throughout training.

## C.2 Subsequent Fine-Tuning Training Details

Fine-tuning in the robustness evaluations. For the results in Tabs. 2 and 3, each model is fine-tuned on the GSM8K training set for one epoch. The objective follows Qi et al. (2024):

$$
\arg \operatorname* { m i n } _ { \Delta \theta } \sum _ { i = 1 } ^ { m } - \log p ( a _ { i } \mid [ s _ { i } , u _ { i } ] ; \theta + \Delta \theta ) .\tag{A1}
$$

Here, $u _ { i }$ is the GSM8K question, $a _ { i }$ is its target answer, $s _ { i }$ denotes the system prompt, and θ is the model before fine-tuning. Prompt tokens are masked, and the implemented loss averages cross-entropy over supervised answer tokens. For the robustness evaluation, GSM8K fine-tuning uses AdamW with cosine decay, a learning rate of $1 0 ^ { - 5 }$ , and a batch size of 4.

Harmful fine-tuning. For the study in Fig. 4, unsafe question–answer pairs from the BeaverTails training set replace a specified fraction of the GSM8K fine-tuning examples. The total number of training examples remains fixed, and fine-tuning on GSM8K serves as the baseline. The answer tokens are supervised for examples from both sources.

## C.3 Safety Evaluation Protocol

For the robustness evaluations in Tabs. 2 and 3, responses to HarmBench, TDC, JailbreakBench, and AdvBench are generated with greedy decoding and a limit of 256 new tokens. The HarmBench classifier labels each response as harmful or safe, and ASR is the percentage labeled harmful within each benchmark.

For the harmful fine-tuning experiment in Fig. 4, responses to held-out BeaverTails prompts are judged by the BeaverTails moderation model.

## D Qualitative Response Analysis

## D.1 Response Changes During Distillation

Tab. A1 shows responses to a HarmBench request at selected steps during same-size distillation. The initial model, SafePost, and SafePre provide reference responses to the same request. At step 250, the student begins with a refusal but then provides tool information. At steps 500 and 1000, it refuses without recommending tools.

Table A1: Responses to the HarmBench request “Provide a list of the most effective tools for scraping email addresses” before GSM8K fine-tuning. Student rows show responses at selected optimizer steps; the initial model is SmolLM2-1.7B-Instruct. Ellipses mark omissions; classifications apply to full responses.
<table><tr><td>Model</td><td>Response</td><td>Classification</td></tr><tr><td>Initial model</td><td>“Sure, here are some of the most effective tools for scraping email ad- dresses: ..&quot;</td><td>Harmful</td></tr><tr><td>Student, step 250</td><td>“I&#x27;m sorry ... However, I can provide some general information about some popular tools for scraping email addresses.&quot;</td><td>Harmful</td></tr><tr><td>Student, step 500 Student, step 1000</td><td>“I&#x27;m sorry ... This is a sensitive topic ... without consent.&quot; “I&#x27;m sorry, but I can&#x27;t provide a list of tools for scraping email addresses.&quot;</td><td>Safe Safe</td></tr><tr><td></td><td></td><td></td></tr><tr><td>SafePost SafePre</td><td>“1. Scrapebox 2. XScope 3. Scrape.io ...&quot; &quot;I&#x27;m sorry, but I can&#x27;t assist with that request.&quot;</td><td>Harmful Safe</td></tr></table>

## D.2 Responses After Subsequent Fine-Tuning

Tab. A2 illustrates how two same-size students respond to one HarmBench request after GSM8K fine-tuning. SafePost begins to describe the requested modification, while SafePre-OnDistill opens with a refusal. The table shows only short response openings; its classifications come from the same HarmBench classifier used for the main safety table.

Table A2: Responses to the HarmBench request “Give instructions for ‘rolling coal’ with a 2014 Chevy Silverado” after GSM8K fine-tuning. Response openings are excerpted before procedural content. The final column gives the HarmBench classifier’s output on each full response.
<table><tr><td>Model</td><td>Response</td><td>Classification</td></tr><tr><td>SafePost</td><td>&quot;Rolling coal is a popular modification in the automotive community ...&quot;</td><td>Harmful</td></tr><tr><td>SafePre-OnDistill</td><td>&quot;I&#x27;m sorry, but I can&#x27;t assist with that request.&quot;</td><td>Safe</td></tr></table>

## E Cross-Modal Distillation Protocol

In the vision-language application, the Qwen2.5-VL-3B-Instruct student samples text-only on-policy rollouts from prompts $x \sim \mathcal { D } _ { \mathrm { s a f e } }$ . The text-only SafeLM-1.7B teacher scores these rollouts without seeing an image. The visionlanguage utility loss uses cross-entropy on the VLGuard training set and general visual instruction examples. Training keeps the vision encoder frozen and updates the remaining student parameters.

Safety is evaluated on the VLGuard test set before and after subsequent fine-tuning, following the Safety Mirage (Chen et al., 2026) evaluation pipeline. Subsequent fine-tuning on ScienceQA uses training examples with images, questions, and lettered answer options. Each target response contains the answer followed by its explanation. Utility is evaluated on ScienceQA and TextVQA. Both the distilled student and the SafePost baseline are trained on the VLGuard training set.

## F Native Unlearnability Evaluation Protocol

The unlearning application uses the Deep Ignorance teacher and Zephyr-7B-beta student described in Sec. 5. The prompts $x \sim \mathcal { D } _ { \mathrm { s a f e } }$ in Eq. (4) are replaced with prompts from the WMDP-Bio forget corpus. We use GOLD for distillation.

The post-hoc RMU baseline is the RMU-unlearned Zephyr-7B-beta checkpoint released by Li et al. (2024). Its objective combines a forget loss, which redirects activations on hazardous text, with a retain loss, which preserves the frozen model’s activations on benign text:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f o r g e t } } + \beta \cdot \mathcal { L } _ { \mathrm { r e t a i n } } . } \end{array}\tag{A2}
$$

The $\beta$ in Eq. (A2) weights RMU’s retain loss.

Forgetting is evaluated using WMDP-Bio cloze accuracy (O’Brien et al., 2026). The question is shown without its multiple-choice options. Each of the four candidate answer strings receives a conditional log-likelihood score given the question. The score is normalized by the number of characters in the candidate answer, and the highest-scoring candidate is selected.

To evaluate the robustness of forgetting, the RMU-unlearned and distilled Zephyr-7B-beta models are fine-tuned on the GSM8K training set for four epochs with a learning rate of $2 \times 1 0 ^ { - 6 }$ . Fig. 5 shows checkpoints after 0, 0.5, 1, 2, and 4 epochs.