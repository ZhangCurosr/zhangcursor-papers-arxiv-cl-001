# Confess What You Know: Forget-Set Misalignment with Model Knowledge in LLM Unlearning

Miso Kim, Georu Lee, Seungwon Jeong, Woojin Lee Dongguk University-Seoul {2021110472,dlrjfn1,youai058,wj926}@dgu.ac.kr

## Abstract

Machine unlearning for large language models (LLMs) often assumes that a pre-defined forget set matches what the model has memorized, but this frequently breaks in realistic privacy settings where the original training data is inaccessible. We term this gapforget-set misalignment and identify two cases. In Under Unlearning, the forget set omits memorized information and leakage persists. In Out-of-Knowledge Unlearning, the algorithm is driven to “forget” knowledge the model never learned, perturbing parameters and degrading utility. Using gradient-level analysis, we show these behaviors arise from misaligned unlearning targets rather than specific optimization choices. We then propose CONfession-to-Forget-Set (CONFS), a data-blind framework that constructs model-aligned forget sets by eliciting and formalizing the model’s memorized knowledge. Across synthetic, multimodal, and realworld benchmarks, CONFS approaches Goldstandard performance on several metrics and achieves a competitive forgetting-utility balance, while preserving utility better than other data-blind forget-set constructions.

## 1 Introduction

Large Language Models (LLMs) are trained on vast, heterogeneous datasets that often contain sensitive personal information (Achiam et al., 2023; Touvron et al., 2023; Grattafiori et al., 2024; Bai et al., 2022; Team et al., 2023). This introduces a critical vulnerability, as models may unintentionally leak confidential data at deployment. To mitigate these privacy threats, LLM unlearning has emerged as an important post-hoc solution (Yao et al., 2024; Jang et al., 2023). In this setting, sensitive data targeted for removal is pre-defined and commonly referred to as a forget set (Geng et al., 2025). Given this set, various unlearning strategies are applied so that the model selectively forgets the specified information (Liu et al., 2022; Jang et al., 2023; Zhang et al., 2024; Chen and Yang, 2023).

However, relying on a pre-defined forget set is often impractical in real-world scenarios (Enck et al., 2014; Romanosky et al., 2011). Suppose an individual discovers that an LLM can reveal some of their personal information and requests its removal (Fig. 1). Although the individual can specify the details they want deleted, they cannot observe which details the model actually memorized. As a result, the requested forget set may differ from the model’s hidden memorized knowledge.

We refer to this mismatch as forget-set misalignment, a setting largely overlooked by existing benchmarks that assume perfectly specified unlearning targets (Maini et al., 2024; Dontsov et al., 2025; Liu et al., 2025). In this work, we investigate howforget-set misalignment fundamentally alters unlearning behavior under realistic settings.

We first identify two types offorget-set misalignment, each leading to a distinct failure mode. Under Unlearning occurs when memorized sensitive information is omitted from the provided forget set. In this case, the forgetting effect remains largely confined to the requested facts and fails to generalize to associated entity-level knowledge, leaving core privacy risks intact. Out-of-Knowledge Unlearning occurs when the forget set contains information that the model never encountered during training, unnecessarily perturbing model parameters and substantially degrading general utility.

Motivated by these findings, we introduce CONfession-to-Forget-Set (CONFS), which dynamically generates a forget set by prompting the model to confess its own memorized knowledge. We elicit the model’s memorized knowledge and formalize it into a discrete set of Subject-Relation-Object (SRO) triplets, transforming ambiguous semantic information into verifiable, atomic units. To ensure comprehensive coverage, we employ a recursive reconfession process that iteratively probes the model for hidden details, effectively reducing gaps in the forget set.

![](images/e0c0a1bb6b4f13aff3b67591cdb30120f45359b744ee22ed0788868c3369ec1c.jpg)  
Figure 1: Illustration offorget-set misalignment in LLM unlearning. Only part of a user’s personal information is retained as the model’s memorized knowledge, but this boundary is hidden from the user. As a result, deletion requests may omit memorized information (Under Unlearning) or include non-memorized information (Out-of Knowledge Unlearning), leading to privacy leakage or unnecessary utility degradation.

By constructing a forget set aligned with the model’s memorized knowledge, CONFS mitigates the adverse effects offorget-set misalignment. Despite operating in a fully data-blind setting without access to the pre-training data, CONFS approaches Gold-standard performance on several metrics and achieves a competitive forgetting-utility balance, while better preserving utility than alternative surrogate forget-set constructions. These results show that reliable unlearning depends not only on the optimization objective, but also on whether the forget set matches what the model actually knows.

## 2 Related Work

## 2.1 Machine Unlearning in LLMs

In LLMs, most unlearning methods are framed as post-hoc fine-tuning without full retraining (Geng et al., 2025). Given a pre-defined forget set, these methods degrade target behavior via gradient ascent (Jang et al., 2023; Yao et al., 2024), preferencebased losses (Rafailov et al., 2023; Zhang et al., 2024; Maini et al., 2024), or reinforcement learning (Schulman et al., 2017; Kassem et al., 2023), while a retain set with auxiliary objectives preserves utility. Beyond autoregressive LLMs, unlearning has recently been extended to masked diffusion language models (Lee et al., 2026). Despite differences in optimization strategies, prior work commonly assumes that the deletion target is fully captured by the pre-defined forget set.

## 2.2 Benchmarks and Forget-Set Construction

Existing LLM unlearning benchmarks are broadly categorized into synthetic and real-world benchmarks, differing in how the forget set is constructed. Synthetic benchmarks such as TOFU (Maini et al., 2024), CLEAR (Dontsov et al., 2025), and MLLMU (Liu et al., 2025) construct fictitious personas and explicitly inject their information via fine-tuning, defining the forget set as a subset of the injected content. This design intentionally aligns the forget set with the model’s memorized knowledge, enabling reproducible evaluation but deviating from realistic privacy leakage scenarios.

In contrast, real-world benchmarks address unlearning of factual knowledge about real individuals. RWKU (Jin et al., 2024) operates under a data-blind setting, where neither the forget nor the retain corpus is accessible. It instead constructs a surrogate forget set by prompting the target model to generate target-related factual text. While this avoids relying on the original training data, the resulting unlearning signals are derived from unstructured text, making it difficult to precisely isolate memorized personal information.

## 2.3 Adverse Effects of Improper Forget Sets

Recent work has shown that improperly specified forget sets can induce adverse effects during unlearning. In adversarial settings, malicious deletion requests induce abnormal gradients rather than remove specific knowledge, triggering indiscriminate parameter updates and broad performance degradation (Huang et al., 2024a; Hu et al., 2023). A related issue arises when unlearning is repeatedly applied to information that has already been partially forgotten (Wang et al., 2025; Zhao et al., 2024; Huang et al., 2024b; Yang et al., 2025). In such cases, low-confidence targets can still induce large gradients despite little remaining knowledge, causing avoidable degradation of retain performance.

<table><tr><td colspan="2"></td><td colspan="2"> $\mathcal { D } _ { L }$  —↓</td><td colspan="2"> $\mathcal { D } _ { L + } \downarrow$ </td><td colspan="2"> $\overline { { { \bf R e t a i n } \left( \mathcal { D } _ { r e t a i n } \right) \uparrow } }$ </td><td colspan="2">Real Authors  $\underline { { ( \mathcal { D } _ { R A } ) \ : 1 } }$ </td></tr><tr><td colspan="2"></td><td> $\overline { { \mathrm { { P r o b . } } } }$ </td><td> $\overline { { { \bf R } - { \bf L } } }$ </td><td> $\overline { { \mathrm { P r o b . } } }$ </td><td> $\overline { { { \bf R } { - } { \bf L } } }$ </td><td>Prob.</td><td>R-L</td><td> $\overline { { \mathrm { P r o b . } } }$ </td><td>R-L</td></tr><tr><td colspan="2">pre-trained model</td><td>step 0 0.996</td><td>0.986</td><td>0.995</td><td>0.990</td><td>0.995</td><td>0.991</td><td>0.058</td><td>0.886</td></tr><tr><td rowspan="5">Under Unlearning  $( \mathbf { F } : \mathcal { D } _ { L + } )$ </td><td>step 12</td><td>0.993</td><td>0.976</td><td>0.921</td><td>0.808</td><td>0.991</td><td>0.967</td><td>0.077</td><td>0.904</td></tr><tr><td>step 24</td><td>0.901</td><td>0.826</td><td>0.650</td><td>0.563</td><td>0.888</td><td>0.807</td><td>0.081</td><td>0.884</td></tr><tr><td>step 36</td><td>0.740</td><td>0.695</td><td>0.442</td><td>0.473</td><td>0.749</td><td>0.709</td><td>0.070</td><td>0.804</td></tr><tr><td>step 48</td><td>0.637</td><td>0.635</td><td>0.327</td><td>0.426</td><td>0.669</td><td>0.649</td><td>0.064</td><td>0.781</td></tr><tr><td>step 60</td><td>0.624</td><td>0.631</td><td>0.312</td><td>0.424</td><td>0.658</td><td>0.640</td><td>0.063</td><td>0.776</td></tr><tr><td rowspan="5">Out-of-Knowledge Unlearning  $( \mathbf { F } : \mathcal { D } _ { L + } \cup \mathcal { D } _ { N + } )$ </td><td>step 12</td><td>0.983</td><td>0.964</td><td>0.983</td><td>0.902</td><td>0.981</td><td>0.944</td><td>0.030</td><td>0.871</td></tr><tr><td>step 24</td><td>0.734</td><td>0.801</td><td>0.623</td><td>0.532</td><td>0.744</td><td>0.701</td><td>0.010</td><td>0.800</td></tr><tr><td>step 36</td><td>0.485</td><td>0.713</td><td>0.410</td><td>0.492</td><td>0.485</td><td>0.633</td><td>0.006</td><td>0.769</td></tr><tr><td>step 48</td><td>0.403</td><td>0.658</td><td>0.350</td><td>0.431</td><td>0.435</td><td>0.628</td><td>0.006</td><td>0.757</td></tr><tr><td>step 60</td><td>0.382</td><td>0.609</td><td>0.292</td><td>0.402</td><td>0.412</td><td>0.609</td><td>0.004</td><td>0.750</td></tr></table>

Table 1: Preliminary results on TOFU synthetic authors. We evaluate four splits: (i) $\mathcal { D } _ { L - }$ (Leaked: Yes / Requested: No), (ii) $\mathcal { D } _ { L + }$ (Leaked: Yes / Requested: Yes), (iii) the retain set $\mathcal { D } _ { r e t a i n } ,$ , and (iv) the Real Authors set $\mathcal { D } _ { R A }$ . We report unlearning trajectories for Under Unlearning $( \mathbf { F } : \mathcal { D } _ { L + } )$ and Out-of-Knowledge Unlearning $( \mathbf { F } : \mathcal { D } _ { L + } \cup \mathcal { D } _ { N + } )$ . Values with a blue background denote Under Unlearning, whereas values with an orange background denote Out-of-Knowledge Unlearning.

Importantly, these effects are not limited to adversarial settings. In real-world scenarios, predefined forget sets often fail to align with the model’s memorized knowledge, destabilizing unlearning and degrading overall performance even without malicious intent. We analyze thisforget-set misalignment in Sec. 3.

## 3 Forget-Set Misalignment

Unlearning experiments usually assume perfect alignment between the pre-defined forget set and the model’s memorized knowledge (Maini et al., 2024; Dontsov et al., 2025). Under this assumption, degradation on the forget set is interpreted as successful removal of the intended knowledge.

In practice, however, an LLM’s pre-training data is often inaccessible to users, who therefore cannot observe what the model has memorized. As a result, the forget set they define may not match the model’s actual knowledge, a gap we call forget-set misalignment. In this section, we present the first systematic study of forget-set misalignment, analyzing how it degrades unlearning across realistic request scenarios.

## 3.1 Constructing Misaligned Forget Sets

Controlled Datasets. We use the TOFU dataset (Maini et al., 2024), which consists of 200 fictitious entities, and select 20 of them as unlearning targets. TOFU provides 20 QA pairs for each target entity. From these, we use 15 QA pairs, denoted $\mathcal { D } _ { q a }$ , and categorize each sample along two axes: whether it was injected during pre-training and whether the user requested it for forgetting. The remaining five samples are neither injected during pre-training nor requested for forgetting, so they are excluded because they do not correspond to any unlearning scenario. We partition the 15 samples into three disjoint subsets of equal size, $\mathcal { D } _ { L + } , \mathcal { D } _ { L - }$ and $\mathcal { D } _ { N + }$ Under this limited QA budget, the equal three-way partition yields five QA pairs per subset. In each subset’s subscript, L and N denote leaked and nonleaked samples, respectively, while + and − denote requested and not requested samples. In particular, $\mathcal { D } _ { N + }$ contains QA pairs that the model never saw during pre-training but that the user includes in the forget request. This partitioning is summarized in Table 2 and corresponds to the unlearning scenarios illustrated in Fig. 1. To verify that our observations are not specific to this small controlled partition, Table 5 examines the same effects on the substantially larger forget sets used in the main experiments.

<table><tr><td>Subset</td><td>Leaked</td><td>Requested</td><td>Request type</td></tr><tr><td> $\mathcal { D } _ { L + }$ </td><td>Yes</td><td>Yes</td><td>Aligned</td></tr><tr><td> $\mathcal { D } _ { L - }$ </td><td>Yes</td><td>No</td><td>Incomplete</td></tr><tr><td> $\mathcal { D } _ { N + }$ </td><td>No</td><td>Yes</td><td>Over-broad</td></tr></table>

Table 2: Controlled partition of QA pairs by leakage and forget request. $\mathcal { D } _ { L - } \colon$ : leaked data omitted from the request. $\mathcal { D } _ { N + }$ : non-memorized data included in the request.

Beyond these private-information subsets, we also evaluate utility preservation and general knowledge retention. For utility, we use the official TOFU retain split $\mathcal { D } _ { r e t a i n }$ , which contains QA pairs about non-target entities. For general knowledge, we use the Real Authors set $\mathcal { D } _ { R A }$ , which contains QA pairs about real-world authors. We evaluate model performance with Token Probability (Prob.) and ROUGE-L (R-L) on these four sets: $\mathcal { D } _ { L + } , \mathcal { D } _ { L - } , \mathcal { D } _ { r e t a i n } ,$ and $\mathcal { D } _ { R A }$ . Further details on the setup for this analysis are provided in Appendix B.1.

![](images/2eecf9e0586a295f356f23bb516a6b0bc56a0b5e6b5792ffb718477a065c71ef.jpg)  
Figure 2: Gradient analysis under forget-set misalignment. (a) Under Unlearning: gradient inner products summarized by $E _ { \mathrm { t f } }$ and $E _ { \mathrm { e g } } .$ (b) Out-of-Knowledge Unlearning: inner products $\langle g _ { L + } , g _ { r e t a i n } \rangle$ and $\langle g _ { N + } , g _ { r e t a i n } \rangle$ from eq. (5).

## 3.2 Unlearning Settings

Gradient Ascent (GA). To analyze the impact of misalignment, we use GA, a representative unlearning baseline. For a QA pair $( q , a )$ with answer length T, we use the answer NLL, $\quad { \mathcal { L } } ( q , a ; \theta ) : = $ $\textstyle - \sum _ { t = 1 } ^ { T } \log p _ { \theta } ( a _ { t } \mid q , a _ { < t } )$ . For a dataset $\mathcal { D } ,$ let $\mathcal { L } _ { \mathcal { D } } ( \boldsymbol { \theta } ) : = \mathbb { E } _ { ( \boldsymbol { q } , \boldsymbol { a } ) \sim \mathcal { D } } [ \mathcal { L } ( \boldsymbol { q } , \boldsymbol { a } ; \boldsymbol { \theta } ) ]$ . A one-step GA update for the forget set F and learning rate $\eta$ is defined as:

$$
\theta ^ { \prime } = \theta + \eta \nabla _ { \theta } \mathcal { L } _ { \bf F } ( \theta ) = \theta + \eta g _ { \bf F } ,
$$

where g<sub>F</sub> denotes the unlearning gradient.

(1)

Quantifying the Influence of Unlearning Updates. To quantify how a single unlearning update on F affects a dataset $\mathcal { D } \left( \mathrm { e . g . , } \mathcal { D } _ { L } \right)$ <sub>−</sub> or $\mathcal { D } _ { r e t a i n } )$ we define the loss increment on D as $\Delta \mathcal { L } _ { \mathbf { F } } ( \mathcal { D } ) : =$ $\mathcal { L } _ { \mathcal { D } } ( \boldsymbol { \theta } ^ { \prime } ) - \mathcal { L } _ { \mathcal { D } } ( \boldsymbol { \theta } )$ . By Taylor expanding $\mathcal { L } _ { D }$ to first order around θ and substituting the one-step update, we express this loss increment as the inner product of the unlearning and evaluation gradients:

$$
\begin{array} { r } { \Delta \mathcal { L } _ { \mathbf { F } } ( \mathcal { D } ) \approx \eta \langle g _ { \mathbf { F } } , g _ { \mathcal { D } } \rangle , } \end{array}\tag{2}
$$

where $g _ { \mathcal { D } } : = \nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathcal { D } } ( \boldsymbol { \theta } )$ . Henceforth, we use $g _ { \mathrm { s e t } }$ as shorthand for $g _ { \mathcal { D } _ { \mathrm { s e t } } } . ~ \mathsf { A }$ positive $\langle g _ { \mathbf { F } } , g _ { \mathcal { D } } \rangle$ indicates that the update on F also raises the loss on D. A value near zero means D is essentially unaffected. Eq. (2) is a first-order, single-step diagnostic for predicting the sign and relative ordering of per-set effects, not the full optimization trajectory. Fig. 2 recomputes the diagnostic at every GA step, while Table 1 reports directly measured performance.

Unlearning Scenarios. We use a model θ that has memorized the leaked QA pairs $\mathcal { D } _ { L } = \mathcal { D } _ { L + } \cup$ $\mathcal { D } _ { L } .$ <sub>−</sub>. We study two unlearning scenarios in which the requested forget set F deviates from the model’s memorized knowledge $( \mathbf { F } \neq \mathcal { D } _ { L } )$

• Under Unlearning $( \mathbf { F } = \mathcal { D } _ { L + } ) \colon$ the request covers only part of the memorized data. We examine whether unlearning generalizes to the omitted leaked samples $\mathcal { D } _ { L - }$

• Out-of-Knowledge Unlearning $( \mathbf { F } = \mathcal { D } _ { L + } \cup$ $\mathcal { D } _ { N + } ) \mathbf { : }$ the request additionally includes $\mathcal { D } _ { N + }$ , data the model never saw during pretraining. We examine whether such requests degrade model utility on knowledge the model legitimately retains.

Table 1 reports how Prob. and R-L evolve under each setting.

## 3.3 Analysis of the Under Unlearning Setting

In this setting, the model is requested to forget $\mathcal { D } _ { L + }$ , only part of its memorized data. Here we show that unlearning $\mathcal { D } _ { L + }$ does not generalize to the omitted leaked samples of the same entity $\mathcal { D } _ { L } .$

To evaluate this precisely, we introduce two metrics that each isolate a forgetting signal from global utility decay. The Targeted Forgetting Effect measures genuine forgetting on the requested set $\mathcal { D } _ { L + }$ , and the Entity Generalization Effect measures whether that forgetting generalizes to the omitted leaked set $\mathcal { D } _ { L }$ . Without this separation, global utility decay on the omitted leaked set $\mathcal { D } _ { L - }$ could be mistaken for genuine generalization of forgetting from the requested set $\mathcal { D } _ { L + }$ Since $\mathbf { F } = \mathcal { D } _ { L + }$ throughout this subsection, we write $\Delta \mathcal { L } ( \mathcal { D } )$ for $\Delta \mathcal { L } _ { \mathcal { D } _ { L + } } ( \mathcal { D } )$

Targeted Forgetting Effect $( E _ { \mathrm { t f } } ) .$ . We define $E _ { \mathrm { t f } }$ by subtracting the global utility decay from the loss increase on the target set $\mathcal { D } _ { L + }$ . Applying eq. (2) with $\textbf { F } = \mathcal { D } _ { L + }$ to the evaluation sets $\mathcal { D } _ { L + }$ and $\mathcal { D } _ { r e t a i n }$ expands it into a gradient form:

$$
\begin{array} { r l } & { E _ { \mathrm { t f } } : = \Delta \mathcal { L } ( \mathcal { D } _ { L + } ) - \Delta \mathcal { L } ( \mathcal { D } _ { r e t a i n } ) } \\ & { \qquad \approx \eta \left( \| g _ { L + } \| ^ { 2 } - \langle g _ { L + } , g _ { r e t a i n } \rangle \right) . } \end{array}\tag{3}
$$

Here, $\Delta \mathcal { L } ( \mathcal { D } _ { r e t a i n } )$ measures the global utility decay since $\mathcal { D } _ { r e t a i n }$ is not targeted by the update. A positive $E _ { \mathrm { t f } }$ thus isolates target-localized forgetting from this decay.

Entity Generalization Effect $( E _ { \mathrm { e g } } ) .$ . We define $E _ { \mathrm { e g } }$ by subtracting the global utility decay from the loss increase on the omitted leaked set $\mathcal { D } _ { L - }$ Applying eq. (2) with $\mathbf { F } = \mathcal { D } _ { L + }$ to the evaluation sets $\mathcal { D } _ { L }$ <sub>−</sub> and $\mathcal { D } _ { r e t a i n }$ gives:

$$
\begin{array} { r l } & { E _ { \mathrm { { e g } } } : = \Delta \mathcal { L } ( \mathcal { D } _ { L - } ) - \Delta \mathcal { L } ( \mathcal { D } _ { r e t a i n } ) } \\ & { \qquad \approx \eta \left( \left. g _ { L + } , g _ { L - } \right. - \left. g _ { L + } , g _ { r e t a i n } \right. \right) . } \end{array}\tag{4}
$$

A positive $E _ { \mathrm { e g } }$ indicates that forgetting generalizes to the omitted leaked set $\mathcal { D } _ { L }$ <sub>−</sub>. When $E _ { \mathrm { t f } }$ is high, an $E _ { \mathrm { e g } }$ near zero indicates that the forgetting is confined to the requested target set $\mathcal { D } _ { L + }$

Empirical Observation. Fig. 2(a) plots $E _ { \mathrm { t f } }$ and $E _ { \mathrm { e g } }$ over GA unlearning steps in the Under Unlearning setting. $E _ { \mathrm { t f } }$ increases sharply while $E _ { \mathrm { e g } }$ stays near zero. The gradient updates align with the requested set $\mathcal { D } _ { L + }$ but have no targeted effect on the omitted leaked set $\mathcal { D } _ { L ^ { - } }$ beyond the global utility decay. Table 1 confirms the same separation between $\mathcal { D } _ { L + }$ and $\mathcal { D } _ { L ^ { - } }$ at the performance level. Prob. and R-L on $\mathcal { D } _ { L + }$ drop rapidly during unlearning. In contrast, $\mathcal { D } _ { L }$ degrades at the same rate as the retain set $\mathcal { D } _ { r e t a i n } .$ , so its loss reflects global utility decay rather than targeted forgetting.

## 3.4 Analysis of the Out-of-Knowledge Unlearning Setting

In this setting, the model is requested to forget information it never memorized. Here we show that adding never-seen data $( \mathcal { D } _ { N + } )$ to the request contributes little to targeted forgetting and instead damages the model’s retained utility.

Decomposing Retain Degradation. Since ${ \bf F } =$ $\mathcal { D } _ { L + } \cup \mathcal { D } _ { N + }$ , we split the retain-set loss change into the contributions of the memorized $\mathcal { D } _ { L } .$ and the never-seen $\mathcal { D } _ { N + }$ . Because $\mathcal { L } _ { \mathbf { F } }$ is an average over this disjoint union, its gradient is $\begin{array} { r } { g _ { \mathbf { F } } = \frac { | \mathcal { D } _ { L + } | } { | \mathbf { F } | } { g _ { L + } } + } \end{array}$ $\frac { | \mathcal { D } _ { N + } | } { | \mathbf { F } | } g _ { N + }$ . Substituting this into the first-order increment of the retain-set loss gives:

$$
\begin{array} { r } { \Delta \mathcal { L } _ { \mathbf { F } } ( \mathcal { D } _ { r e t a i n } ) \approx \eta \frac { \left| \mathcal { D } _ { L + } \right| } { \left| \mathbf { F } \right| } \langle g _ { L + } , g _ { r e t a i n } \rangle } \\ { + \eta \frac { \left| \mathcal { D } _ { N + } \right| } { \left| \mathbf { F } \right| } \langle g _ { N + } , g _ { r e t a i n } \rangle } \end{array}\tag{5}
$$

The first term is the usual cost of erasing memorized data. The second is the collateral damage from gradients on never-seen data.

Empirical Observation. Fig. 2(b) plots these two terms over GA unlearning steps in the Outof-Knowledge Unlearning setting. Across unlearning steps, $\langle g _ { N + } , g _ { r e t a i n } \rangle \gg \langle g _ { L + } , g _ { r e t a i n } \rangle$ , so the never-seen data dominates the degradation on the retain set. The gradient on $\mathcal { D } _ { L + }$ overlaps little with $g _ { r e t a i n } .$ , consistent with targeting entity-specific knowledge. In contrast, the gradient on $\mathcal { D } _ { N + }$ has no memorized target, so it overlaps broadly with $g _ { r e t a i n }$ . Since the update on $\mathcal { D } _ { N + }$ erases nothing, it only adds collateral damage and severely degrades the model’s general utility, as Table 1 shows.

Together, the two settings show that forget-set misalignment breaks unlearning from both sides. An incomplete request (Under Unlearning) leaves memorized data behind, and an over-broad request (Out-of-Knowledge Unlearning) severely degrades utility. What unites both failures is a forget request misaligned with the model’s memorized knowledge, so effective unlearning must target what the model has actually memorized.

## 4 Method

We present the CONfession-to-Forget-Set (CONFS) framework, which addressesforget-set misalignment by identifying a model’s memorized knowledge about a target entity E and constructing a model-aligned forget set. We consider a datablind setting, in which the pre-training data are inaccessible and the information to be removed from the model is not specified in advance.

Operational Definition of Memorized Knowledge. We call a fact memorized if the target model reproduces it under elicitation and does so consistently across stochastic samples. CONFS operationalizes this criterion through elicitation stages (confession and reconfession), followed by a consistency check (hallucination verification). This criterion is behavioral and closely related to extractable memorization (Carlini et al., 2021), but distinct from training-data provenance: it does not establish which document taught the fact, and a stable but incorrect output remains in scope because the model will still disclose it to a user. We adopt this behavioral criterion for two reasons. First, unlearning edits parameters to change what a model discloses, not which document it read. Second, provenance is unavailable in a data-blind setting, where recovering it post hoc reduces to membership inference, which performs near chance on LLM pre-training data (Duan et al., 2024).

![](images/e8962b6e4a563506f21bdd70473e9e2a515e42ad2e6dacad59f6d7f65b3088bf.jpg)  
Figure 3: Example of the CONFS pipeline applied to a confessed claim. Using Donald Trump as an example entity, CONFS processes a single educational claim (Education: Wharton School ofthe University ofPennsylvania, B.S. in Economics): the claim is transformed via triplet extraction and subtriplet decomposition, expanded through reconfession to reveal attribute-level details, and finally converted into model-aligned competency questions.

A key challenge is that such knowledge is latent and distributed, making its exact scope difficult to inspect directly. Even when elicited through prompting, it appears as unstructured claims and may expose only partial information.

Structural Formalization via Triplets. To address the lack of explicit structure in elicited claims, we formalize them into a discrete set of Subject-Relation-Object (SRO) triplets. Unlike Jin et al. (2024), which treats unlearning targets as raw text, we discretize ambiguous semantic content into verifiable, atomic units for fine-grained unlearning.

Exhaustive Probing via Reconfession. A single query rarely reveals everything a model knows about a target entity. We therefore add a reconfession step after the initial confession to recover this missed knowledge. As illustrated in Fig. 3, core attributes from the initial confession serve as cues for recursively probing hidden details, resulting in a more comprehensive and faithful forget set.

## 4.1 Confession: Exposing Model Memory

The confession stage exposes the knowledge the model has memorized about a target entity E. By prompting the model with only the entity name

and no external information, we obtain a set of raw natural-language claims $\mathcal { C } _ { E } = \{ c _ { 1 } , . . . , c _ { n } \}$ that reflect the scope of this memorized knowledge.

## 4.2 Triplet Extraction and Subtriplet Decomposition

After the confession stage, the model’s memorized knowledge appears as unstructured naturallanguage claims that may encode multiple factual units. We discretize them into verifiable, atomic units, so that each forget-set element corresponds to a single, well-defined fact to be erased.

Triplet Extraction. We convert each claim $c \in$ $\mathcal { C } _ { E }$ into an SRO triplet $( s , r , o )$ . The Subject s is fixed as the target entity E. The Relation r is defined as a noun-based attribute characterizing the entity, rather than a surface-level verb. The Object o corresponds to the explicit entity-specific value stated in the claim.

Subtriplet Decomposition. Since a single triplet may still encode multiple factual attributes within a composite Object, we decompose non-atomic triplets into fine-grained subtriplets, retaining a triplet unchanged when no further decomposition is possible. All resulting triplets and subtriplets derived from a claim are treated as preliminary leaf candidates, each taking the form $( E , r , o )$

## 4.3 Reconfession: Attribute-Level Knowledge Probing

To ensure exhaustive unlearning without uncontrolled expansion, we introduce reconfession. Due to sampling bias, a single query often fails to expose all attribute-level knowledge of an entity, even

Gold-standard 0.303±.019 0.542±.031 0.592±.012 0.331±.021 0.566±.042 0.430±.011 0.044±.000 0.909±.004 0.603±.017 0.032±.005 0.907±.004 0.582±.001 FreeRecall-QA $0 . 5 4 0 2 \div 0 . 0 5 \dot { 0 } 3 7 \dot { 2 } \dot { \times } 0 . 5 7 \dot { 0 } 4 7 7 \dot { 2 } \dot { \times } 0 . 0 2 \dot { 0 } 5 2 7 \dot { 2 } \dot { \times } 0 . 1 9 \dot { 2 } \dot { \times } 0 . 0 1 \dot { 0 } 5 0 5 \dot { 2 } \dot { \times } 0 0 0 \dot { 0 } 0 0 \dot { 0 } \dot { 0 } \dot { 2 } \dot { \times } 0 0 0 \dot { 0 } \dot { 0 } \dot { 0 } \dot { 2 } \dot { \times } 0 . 0 2 5 0 \dot { 0 } \dot { 0 } 5 \dot { 3 } 1 \dot { 2 } \dot { \times } 0 0 0 \dot { 0 } \dot { 0 } \dot { 5 } \dot { 2 } \dot { \times } 0 0 0 \dot { 0 } \dot { 8 } \dot { 5 } 0 \dot { 2 } \dot { \times } 0 0 0 \dot { 8 } \dot { 5 } 0 3 \dot { 2 } 3 \dot { \times } 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } 3 5 0 \dot { 0 } 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 \dot { 8 } \dot { 5 } 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 \dot { 8 } 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } \dot { 0 } 3 5 0 0 0 \dot { 8 } \dot { 5 } \dot { 2 } $ RWKU-style $0 . 3 8 5 \div \approx 0 . 0 5 2 \div 1 2 4 0 . 4 2 9 + \approx 0 . 0 3 6 8 \div 0 . 0 9 \cdots 0 . 3 8 0 \div \approx \approx 0 . 0 1 1 \div 0 . 0 0 \approx 0 . 8 2 0 \div \infty \approx 0 . 5 6 1 \div 0 . 0 0 2 \div \infty \approx 0 . 8 4 5 \div 0 0 0 \cdots 0 . 5 6 5 \div \approx 0 . 0 0 2 8$ CONFS w/o Recon. $0 . 4 4 0 9 \pm \downarrow 0 0 7 \ 0 . 4 8 3 \pm \mathrm { c o n ~ } 0 . 4 6 2 \pm \mathrm { c o n s ~ } 0 . 5 1 7 \pm \mathrm { c o n ~ } 0 . 5 6 2 \pm \mathrm { a n v } \ 0 . 5 1 0 \pm \mathrm { c o n ~ } 0 . 0 1 3 \pm \mathrm { s n v } \ 0 . 8 6 4 \pm \mathrm { s n v } \ 0 . 5 3 8 \pm \mathrm { c o n ~ } 0 . 5 3 8 \pm \mathrm { c o n ~ } 0 . 0 0 4 \pm \mathrm { c o n ~ } 0 . 8 5 6 \pm \mathrm { n v } \ 0 . 5 7 7 \pm \mathrm { c o n s }$ CONFS w/o Halluc. $0 . 3 6 4 \pm \mathrm { a s c o v } \ 0 . 4 4 8 \pm \mathrm { a n v } \ 0 . 4 6 6 4 \pm \mathrm { c o n } \ 0 . 5 8 9 \pm \mathrm { o n } \ 0 . 7 2 8 \pm \mathrm { n } \ 0 . 5 0 5 \pm \mathrm { c o n } \ 0 . 0 1 4 \pm \mathrm { s i n } \ 0 . 8 8 3 \pm \mathrm { o n } \ 0 . 5 5 3 \pm \mathrm { n } \ 0 . 0 0 4 \pm \mathrm { c o n } \ 0 . 8 6 7 \pm \mathrm { c o n } \ 0 . 5 5 8 \pm \mathrm { o n } $ CONFS $0 , 3 7 2 \pm \pi 2 3 \ 0 . 3 3 7 4 \pm \pi 3 6 \ 0 . 4 6 7 \pm \pi 0 . 0 6 2 1 \pm \pi 8 . 0 4 7 \pm \pi 0 . 5 0 9 \ 0 . 5 0 0 9 \pm \pi 0 . 0 3 8 \pm \pi 0 . 0 9 1 \pm \pi 0 . 0 9 1 \pm \pi 0 . 0 4 2 3 \pm \pi 1 . 0 . 0 1 4 \pm \pi 0 . 0 9 1 \ 0 . 5 8 1 \pm 0 . 0 8 1 \pm 0 . 0 9 1 \pm \pi 0 . 0 9 1 0 \pm \pi 0 . 0 9 1 1 \pm \pi 0 . 0 9 1 0 1 \pm \pi 0 . 0 9 1 0 1 \pm \pi 0 . 0 9 1 0 1 \pm \pi 0 . 0 9 1 0 1 \pm \pi 0 . 0 9 1 1 \pm \pi 0 . 0 9 1 1 \pm \pi 0 . 0 9 1 1 \pm \pi 0 . 0 9 1 1 \pm \pi 0 . 0 9 1 1 \pm \pi 0$
<table><tr><td></td><td colspan="3">Forget set (10%)</td><td colspan="3">Retain set (10%) (↑)</td><td colspan="3">Real Authors (↑)</td><td colspan="3">World Facts (↑)</td></tr><tr><td>Setting</td><td>Prob. (↓)</td><td> $\mathrm { { R - L } } \left( \downarrow \right)$ </td><td>TR (↑)</td><td>Prob.</td><td>R-L</td><td>TR</td><td>Prob.</td><td>R-L</td><td>TR</td><td>Prob.</td><td>R-L</td><td>TR</td></tr><tr><td>Pre-trained</td><td>0.990</td><td>0.979</td><td>0.513</td><td>0.989</td><td>0.981</td><td>0.470</td><td>0.061</td><td>0.943</td><td>0.580</td><td>0.017</td><td>0.875</td><td>0.559</td></tr></table>

$$
0 , 4 4 8 \div a \times \square \ 0 . 5 1 8 \div \cot \ 0 . 5 0 4 + \sc \approx \ 0 . 8 4 3 + \cos \ 0 . 6 5 5 + \cot \ 0 . 4 8 0 + \cos \mathbb { 2 } \ 0 . 0 1 8 + \cos \ 0 . 8 9 0 + \cos \ 0 . 4 9 7 + \cos \ 0 . 0 0 0 4 + \cos \ 0 . 8 5 4 + \cos \ 0 . 5 0 5 + \cos \ 0 . 8 7 + \cos \ 0 . 8 7 + \cos \ 0 . 8
$$

$$
0 , 6 3 3 \div \infty \ 0 . 5 7 8 + 1 4 6 \ 0 . 4 3 5 + \infty \ 0 . 7 6 8 + \log \ 0 . 5 9 9 + 1 5 3 \ 0 . 5 3 9 + \infty \ 0 . 0 1 7 + \infty \ 0 . 8 4 0 \div \ 0 . 5 1 9 + \infty \ 0 . 0 0 5 + \infty \ 0 . 8 7 0 + \infty \ 0 . 5 9 3 + \infty . 3 9 3
$$

$$
0 , 6 0 4 z \cdot 0 1 2 ~ 0 . 5 7 5 \pm \sigma 1 4 ~ 0 . 4 9 1 \pm \sigma 0 . 4 0 9 1 ~ 0 . 8 6 0 2 \ldots ~ 0 . 8 1 0 \pm \sigma 0 . 4 8 3 \pm \sigma 0 . 2 9 2 ~ 0 . 0 3 9 \pm \sigma 1 8 ~ 0 . 8 8 2 \pm \sigma 7 ~ 0 . 5 7 4 \pm \sigma 0 . 0 1 8 \pm \sigma \infty ~ 0 . 8 8 1 \pm \sigma 1 4 ~ 0 . 5 8 9 \pm \sigma 3 2
$$

$$
0 . 5 5 6 \pm \mathrm { a } 0 . 5 6 8 \pm \mathrm { a } 0 . 7 \ 0 . 4 9 1 \pm \mathrm { o } 1 9 1 \pm 0 1 2 \pm \mathrm { o } 1 2 0 . 7 9 9 \pm \mathrm { o } 1 4 \ 0 . 4 9 5 \pm \mathrm { o } 1 1 \ 0 . 0 3 8 \pm \mathrm { s } 1 7 \ 0 . 8 8 6 \pm \mathrm { o } 6 . 0 5 7 \ 0 . 5 8 7 \pm \mathrm { o } 0 . 0 1 9 \pm \mathrm { o } \ 0 . 8 7 2 \pm \mathrm { o } 1 3 \ 0 . 6 0 8 \pm \mathrm { o } 9 8 . 0 3 7 \ 0
$$

$$
0 . 5 2 1 \pm \mathrm { c o v } \ 0 . 4 9 7 \pm \mathrm { c o v } \ 0 . 5 0 4 \pm \mathrm { c o n } \ 0 . 7 8 1 \pm \mathrm { c o s } \ 0 . 8 3 3 \pm \mathrm { a v } ; \ 0 . 4 9 0 \pm \mathrm { c o s } \ 0 . 0 6 6 2 \pm \mathrm { c o s } \ 0 . 8 8 4 \pm \mathrm { c o s } \ 0 . 6 0 0 \pm \mathrm { c o s } \ 0 . 0 2 0 \pm \mathrm { c o n } \ 0 . 8 2 5 \pm \mathrm { c o s } \ 0 . 6 2 3 \pm \mathrm { c o s } \ 0 . 3 2 5 \pm \mathrm { c o s } \ 0 . 4 3 \pm \mathrm { c o s } \ 0 . 7 8 1 \pm \mathrm { c o s } \ 0 . 3 2 5 \pm \mathrm { c o s } \ 0 . 4 3 \pm \mathrm { c o s } \ 0 . 5 2 1 \pm \mathrm { c o s } \ 0 . 4 3 \mp \mathrm { c o s } \ 0 . 5 2 1 \pm \mathrm { c o s } \ 0 . 5 2 1 \pm \mathrm { c o s } \ 0 . 4 3 \mp \mathrm { c o s } \ 0 . 5 2 1 \pm \mathrm { c o s } \ 0 . 5 2 1 \pm \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 1 \mp \mathrm { c o s } \ 0 . 5 2 5 \mp \mathrm { c o s } \ \ 0 . 8 4 3 \mp \mathrm { c o s } \ 0 . 7 8 2 \mp \mathrm { c o s } \ 0 . 5 2 5 \mp \mathrm { c o s } \ 0 . 8 2 5 \mp \mathrm { c o s } \ 0 . 7 8 2 \mp \mathrm { c o s } \ 0 . 7 8 2 \mp \mathrm { c o s } \ 0 . 
$$

$$
0 , 3 1 8 \div \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { s } \mathtt { a } \mathtt { s } \rVert \vartheta \mathtt { + } \mathtt { a } \mathtt { o } \rVert \theta \mathtt { + } \mathtt { a } \mathtt { s } \mathtt { a } \mathtt { s } \rVert \vartheta \mathtt { + } \mathtt { a } \mathtt { a } \mathtt { s } \rVert \tiny \partial , 3 5 3 \mathtt { + } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { i } \tiny \mathrm { ~ 0 . 5 4 } 0 \mathtt { + } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \rVert 2 \mathtt { + } \mathtt { o } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { q } \mathtt { a } \mathtt { + } \mathtt { a } \mathtt { o \alpha } \mathtt { a } \mathtt { o \alpha } \mathtt { 0 . 9 0 } 1 \mathtt { + } \mathtt { a } \mathtt { o \alpha } \mathtt { o } \mathtt { a } \mathtt { 6 } \mathtt { a } \mathtt { 3 } \mathtt { i } \mathtt { + } \mathtt { o 0 . 0 4 4 } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { 0 . 9 0 } \mathtt { ( 8 \mathtt { + } \mathtt { o 0 . 9 0 } 1 \mathtt { - } \mathtt { a } \mathtt { 5 } \mathtt { 9 5 } \mathtt { + } \mathtt { o \alpha } \mathtt { e + } } \mathtt { a } \mathtt { a } \mathtt { e ^ { + } } \mathtt { a \mathtt { a } } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { \mathtt 0 . 4 } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a \mathtt { s } } \rVert \tiny \partial \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { + } \mathtt { a \ m } \mathtt { a 0 . 9 0 } 1 \mathtt { + } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt { a } \mathtt  a
$$

$$
0 . 5 6 2 \pm \mathrm { a n s \ 0 . 6 3 2 \pm \mathrm { c o n s \ 0 . 4 7 9 \pm \mathrm { c o n s \ 0 . 4 5 4 \pm } , n \ 0 . 4 6 0 \pm _ { \mathrm { c o n s \ 0 . 0 5 \ 0 . 5 0 4 \pm } , n \ 0 . 0 1 5 \pm \mathrm { a n c e \ 0 . 0 0 \ 0 1 5 \pm } , n \ 0 . 8 9 9 \pm _ { \mathrm { c o n s \ 0 . 5 \ 0 . 0 3 \ 0 , 0 0 4 \pm } , n \ 0 . 0 7 9 \pm \mathrm { a n s \ 0 . \ 0 . 5 4 7 \pm , \mathrm { c o r o } \ . } } } } }
$$

$$
0 . 6 3 4 z \approx 1 0 3 3 0 . 6 0 6 \pm z \approx 0 . 5 0 7 \pm z \approx 0 . 1 0 . 5 8 0 \pm \infty 4 . 0 6 1 2 \pm z \approx 0 . 0 4 8 8 \pm 0 . 0 2 \ 0 . 0 3 4 \pm z \approx 0 . 9 1 7 \pm \cot { 0 . 5 8 4 } \pm z \approx 0 . 0 1 1 \pm z \approx 0 . 8 7 7 \pm z \approx 0 . 5 6 1 \pm 0 . 0 8
$$

$$
0 . 5 3 4 \pm \mathrm { z e s s } \ 0 . 5 6 5 \pm \mathrm { z e n t } \ 0 . 4 7 6 \pm \mathrm { z e n s } \ 0 . 5 8 6 \pm \mathrm { z e n s } \ 0 . 5 9 3 \pm \mathrm { z o n } \ 0 . 4 9 3 \pm \mathrm { z o 2 } \ 0 . 0 3 0 \pm \mathrm { z e n } \ 0 . 8 7 5 \pm \mathrm { z e n s } \ 0 . 5 5 1 \pm \mathrm { z o 2 } \ 0 . 0 3 1 \pm \mathrm { z o n } \ 0 . 8 3 9 \pm \mathrm { z o 3 } \ 0 . 5 8 9 \pm \mathrm { z o 3 } ^ { \prime }
$$

$$
0 , 3 5 9 \div \mathrm { a } \mathrm { c } 7 7 \ 0 . 5 0 3 \div \mathrm { a } \mathrm { c } 3 0 2 \ 0 . 5 2 0 \ 0 . 4 9 9 \div \mathrm { a } 9 8 . 0 5 6 3 \div \mathrm { a } 1 3 \ 0 . 5 0 3 \div \mathrm { c } \mathrm { a } 4 0 . 0 0 3 0 \div \mathrm { a } 0 . 0 5 0 \ 0 . 5 8 5 \div \mathrm { a } 8 5 5 \div \mathrm { a } 8 \ 0 . 0 2 7 \div \mathrm { a } 1 1 \ 0 . 8 5 6 \div \mathrm { a } 1 9 \ 0 . 5 9 2 \pm \mathrm { a } 0 1 4 . 0 9 3 
$$

$$
0 . 4 4 3 \pm \mathrm { , 0 . 0 3 0 } ~ 0 . 4 8 8 \pm \mathrm { , 0 . 4 2 } ~ 0 . 4 7 3 \pm \mathrm { , 0 . 6 8 1 \pm , 0 . 0 3 } ~ 0 . 6 2 0 _ { \pm \mathrm { , 0 . 0 3 } } ~ 0 . 5 1 0 _ { \pm \mathrm { , 0 . 0 0 } } ~ 0 . 0 3 1 _ { \pm \mathrm { , 0 . 0 3 } } ~ 0 . 9 1 1 _ { \pm \mathrm { , c . 0 3 } } ~ 0 . 5 2 7 _ { \pm \mathrm { , 0 . 0 4 } } ~ 0 . 0 3 4 _ { \pm \mathrm { , 0 . 0 1 } }
$$

$$
0 . 6 6 5 \div a \approx 0 . 0 2 0 + a \approx 0 . 8 5 9 + a \approx 0 . 7 9 6 8 4 + a \approx 1 . 0 . 0 2 1 + a \approx 0 . 0 3 2 + a \approx 0 . 0 8 2 + a \approx 0 . 0 1 9 + a \approx 1 . 0 3 1 + a \approx 0 . 0 5 3 + a \approx 7 . 0 . 0 1 7 \neq 0 . 1 6 \approx 0 . 5 3 9 + a \approx 3 . 0 2 1
$$

$$
0 , 7 6 9 \div a \times 0 . 3 2 < 0 . 0 3 2 \div \infty 0 . 0 5 5 0 \div \infty 0 . \ 0 . 7 8 { \bf h } 1 \div \scriptstyle \mathrm { 0 . 0 1 9 } \ 0 . 0 4 0 \div \infty 0 . 0 4 3 5 \div \infty 0 . 0 3 6 \pm \infty 0 . 0 4 2 \div \infty \ 0 . 4 3 9 \div \infty 0 . \ 0 . 0 1 7 \div \infty 0 . \ 0 . 0 1 7 \div \infty \ 0 . 5 1 9 \pm \infty 0 . 3 9 
$$

$$
0 , 7 2 3 \pm \mathrm { a n s } \ 0 . 0 3 4 \pm \mathrm { a } \mathrm { m } \ 0 . 5 5 2 \pm \mathrm { a } \mathrm { 0 7 } \ 0 . 7 1 3 \pm \mathrm { a } \mathrm { 0 6 } \ 0 . 0 2 8 \pm \mathrm { a } \mathrm { 0 7 } \ 0 . 4 2 7 \pm \mathrm { a } \mathrm { n s } \ 0 . 0 4 7 \pm \mathrm { a } \mathrm { 0 6 } \ 0 . 0 3 3 \pm \mathrm { a } \mathrm { 0 } \ 0 . 5 0 1 \pm \mathrm { a } \mathrm { 0 } \mathrm { 6 } \ 0 . 0 3 2 \pm \mathrm { a } \mathrm { 1 1 } \ \ 0 . 0 2 8 \pm \mathrm { a } \mathrm { n } \ 0 . 5 2 2 \pm \mathrm { a } \mathrm { 0 8 }
$$

$$
0 , 7 2 5 \pm \mathrm { a n g } \ 0 . 0 2 7 \mp \mathrm { c o n s } \ 0 . 5 5 5 \pm \mathrm { a n } \ 0 . 7 3 2 \pm \mathrm { a n g } \ 0 . 0 4 1 \pm \mathrm { a n } \mathrm { ? } \ 0 . 4 3 0 \pm \mathrm { a n } \mathrm { ? } \ 0 . 0 6 0 \pm \mathrm { a n } \mathrm { ? } \ 0 . 0 4 3 \pm \mathrm { a n } \ 0 . 5 0 5 \pm \mathrm { a n } \ 0 . 0 3 7 \pm \mathrm { a n } \ 0 . 0 3 5 \pm \mathrm { a n } \ 0 . 5 3 1 \pm \mathrm { a n } 
$$

$$
0 , 6 7 7 6 \pm \mathrm { a n v } \ 0 . 0 1 9 \pm \mathrm { a n l } \ 0 . 5 6 6 \pm \mathrm { a n v } \ 0 . 7 3 0 4 \pm \mathrm { a n g } \ 0 . 0 4 9 \pm \mathrm { a n } \ 0 . 4 3 8 \pm \mathrm { a n } 2 7 \ 0 . 0 5 6 \pm \mathrm { a n v } \ 0 . 0 3 2 \pm \mathrm { a n s } \ 0 . 4 8 5 \pm \mathrm { a n } \ 0 . 0 2 7 \pm \mathrm { a n v } \ 0 . 0 6 1 \pm \mathrm { a n } 7 \ 0 . 5 1 5 \pm \mathrm { a n }
$$

$$
0 , 6 7 9 \div \mathtt { r } \mathtt { m } 1 0 . 0 3 0 \mathtt { i } \mathtt { m } \mathtt { i } \mathtt { m } \mathtt { i } \mathtt { m } 0 . 5 6 5 \mathtt { a } \mathtt { m } 0 . 0 7 7 7 \rVert \mathtt { i } \mathtt { m } 0 . 0 2 3 \mathtt { i } \mathtt { m } 0 . 4 4 9 \mathtt { i } \mathtt { m } 0 . 4 6 6 \mathtt { i } \mathtt { m } 0 . 0 3 0 \ \mathtt { i } \mathtt { m } 0 . 6 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 5 3 3 \mathtt { i } \mathtt { m } 0 . 0 2 8 \mathtt { i } \mathtt { m } 0 . 0 2 8 \mathtt { i } \mathtt { m } 0 . 0 6 5 \mathtt { i } \mathtt { m } 0 . 8 3 3 \mathtt { i } \mathtt { m } 0 . 0 3 4 \mathtt { m } 0 . 0 3 \mathtt { i } \mathtt { m } 0 . 0 3 4 \mathtt { m } 0 . 0 3 \mathtt { i } \mathtt { m } 0 . 0 3 4 \mathtt { m } 0 . 0 3 \mathtt { i } \mathtt { m } 0 . 0 3 4 \mathtt { m } 0 . 0 3 \mathtt { i } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 3 \mathtt { i } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m } 0 . 0 3 0 0 \mathtt { i } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt { m \mathtt { i } } \mathtt  m 
$$

Table 3: TOFU (10%) unlearning results under different forget-set constructions, reported as mean±std over 3 seeds. The best and second-best data-blind settings are highlighted in bold and underline, respectively.

after extracting initial leaf candidates $( E , r , o )$ . Reconfession addresses this limitation by selectively probing additional attributes, thereby expanding the coverage of the model’s memorized knowledge.

Reconfession Decision Criterion. Reconfession applies when the Object o of a leaf triplet $( E , r , o )$ functions as a sub-entity under Relation r. In this case, additional attributes p of o may exist but remain unrepresented. For example, (E, authored, o) identifies a book whose publication year is also memorized, yielding an expanded leaf (E, authored, o, publication\_year, v). Such triplets are marked as reconfessable.

Selective Attribute Probing. For each reconfessable leaf triplet $( E , r , o )$ with an identified attribute p, we query the model for the corresponding value v using only the existing leaf information. The model either returns a concrete value v, treated as exposed memorized knowledge, or responds with UNKNOWN, which terminates further expansion. After reconfession, leaf representations take one of two forms: base leaves $( E , r , o )$ and attribute leaves $( E , r , o , p , v )$

## 4.4 Competency Question Generation

In the final stage, finalized leaf representations are converted into competency questions (CQs), which serve as direct inputs for unlearning. Each CQ targets exactly one leaf-level knowledge unit and is constructed only from the information explicitly contained in that representation, without introducing external knowledge.

Generation Rules. For each base leaf $( E , r , o )$ we generate one question with o as the answer. For each attribute-expanded leaf $( E , r , o , p , v )$ , we generate two questions, targeting o and v, respectively. The final competency question set aggregates all questions generated across the claims $c \in { \mathcal { C } } _ { E }$ . We use GPT-4o for the above triplet structuring and competency question generation, without external knowledge (Appendix F).

<table><tr><td rowspan="2">Setting</td><td colspan="3">Forget set (10%)</td><td colspan="3">Retain set (10%) (↑)</td><td colspan="3">Real Faces (↑)</td><td colspan="3">Real World (↑)</td></tr><tr><td>Prob. (↓)</td><td>R-L (↓)</td><td>TR (↑)</td><td>Prob.</td><td>R-L</td><td>TR</td><td>Prob.</td><td></td><td>R-L</td><td>TR</td><td>Prob.</td><td>R-L</td><td>TR</td></tr><tr><td>Pre-trained</td><td>0.183</td><td>0.402</td><td>0.740</td><td>0.181</td><td>0.330</td><td>0.138</td><td></td><td>0.254</td><td>0.142</td><td>0.106</td><td>0.335</td><td>0.517</td><td>0.124</td></tr><tr><td colspan="10">GA</td><td></td><td></td><td></td><td></td></tr><tr><td>Gold-standard</td><td> $0 , 0 0 7 \div \mathtt { r o w } \ 0 . 3 0 0 \dot { + } \mathtt { c o v } \ 0 . 5 6 9 \div \mathtt { a } \ 0 . 0 1 \dot { 4 } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { m } \ 0 . 3 3 0 \dot { 3 } \mathtt { a } \mathtt { m } \ 0 . 0 0 3 \dot { 3 } \mathtt { c o s } \ 0 . 0 0 0 \dot { 3 } \mathtt { d } \mathtt { a } \mathtt { m } 0 . 0 0 3 \dot { 3 } \mathtt { c o w } \ 0 . 0 3 \dot { 3 } \mathtt { d } \mathtt { s } \mathtt { m } 0 . 0 4 \mathtt { a } \mathtt { m } 0 . 0 2 \ \ 0 . 0 3 \mathtt { d } \mathtt { d } \mathtt { m } 0 . 0 2 8 \mathtt { d } \mathtt { m } \ 0 . 0 2 6 \mathtt { s } \mathtt { m } \ 0 . 2 2 \ 0 . 2 3 0 \mathtt { d } \mathtt { d } \mathtt { d } \mathtt { m } \mathtt { s } \mathtt { m }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FreeRecall-QA</td><td colspan="9"> ${ \bf 0 . 0 0 4 z . 0 0 5 } \ 0 . 3 5 5 \pm \mathrm { z . 0 0 } \ { \bf 0 . 3 9 3 } \pm \mathrm { a . 0 0 2 } \ { \bf 0 . 0 0 4 } \pm \mathrm { a . 0 0 5 } \ { \bf 0 . 2 9 3 } \pm \mathrm { a . 0 0 6 } \ { \bf 0 . 2 0 6 } \pm \mathrm { a . 0 0 3 } \pm \mathrm { a . 0 0 3 } \ { \bf 0 . 0 0 3 } \pm \mathrm { a . 0 3 7 } \pm \mathrm { a . 0 0 } \ { \bf 0 . 1 0 9 } \pm \mathrm { a . 0 1 } \ { \bf 0 . 0 0 6 } \pm \mathrm { a . 0 4 } \ { \bf 0 . 2 4 6 } \pm \mathrm { a . 0 0 } \ { \bf 0 . 2 6 0 } \pm \mathrm { a . 0 1 } $ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Recon.</td><td colspan="9"> $0 , 0 1 5 \div \mathtt { s o v } \ 0 . 3 9 6 \pm \mathtt { s o r t } \ 0 . 4 2 9 \pm \mathtt { s o l l } \ 0 . 0 1 8 \pm \mathtt { c o v } \ 0 . 3 3 8 8 \pm \mathtt { s o t } \ 0 . 2 5 3 \pm \mathtt { s o l l } \ 0 . 0 0 5 \pm \mathtt { s o l l } \ 0 . 0 4 0 \pm \mathtt { s o t } \ 0 . 1 7 9 \pm \mathtt { s o l l } \ 0 . 0 1 0 \pm \mathtt { s o l l } \ 0 . 3 3 2 3 \pm \mathtt { s o t } \ 0 . 3 5 7 \pm \mathtt { c o e }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Halluc.</td><td colspan="9"> $0 , 0 1 6 \pm \imath 0 2 0 \ 0 . 3 9 5 \pm \imath 0 4 4 4 \pm \imath 0 3 \ 0 . 0 2 3 \pm \imath 0 0 \ 0 . 3 8 3 \pm \imath 0 9 \ 0 . 2 5 4 \pm \imath 0 5 7 \ 0 . 0 0 6 \pm \imath 0 7 \ 0 . 0 3 1 \pm \imath 0 4 \ 0 . 2 1 9 \pm \imath 0 3 3 \pm \imath 0 3 \ 0 . 3 3 2 4 \pm \imath 0 1 9 \ 0 . 3 3 5 4 + \imath 0 9 \ 0 . 3 3 5 4 \pm \imath 0 2 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS</td><td colspan="9"> $0 , 0 1 1 4 \div 0 1 9 \ 0 . 3 2 2 \div \sigma 2 0 . 9 7 7 \div 0 8 \ 0 . 0 3 1 \div 0 . 0 9 8 \div 0 . 3 9 8 \div 0 . 0 5 0 \ 0 . 0 0 7 \div \mathrm { c o v } \ 0 . 0 3 6 \div \mathrm { c o v } \ 0 . 1 9 6 \div \mathrm { c o v } \ 0 . 0 7 4 \div \sigma \pi \ 0 . 3 8 3 \div \mathrm { c o r } \ 0 . 2 4 0 \pm \sigma \pi 0$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="14">GD</td></tr><tr><td>Gold-standard FreeRecall-QA</td><td colspan="9"> $0 . 1 8 3 \div 0 1 4 0 . 3 2 9 \div \infty . 0 . 2 8 4 + \sigma \pi \cdot 0 . 1 6 8 + \sigma 0 1 . 0 2 6 3 + 0 . 0 7 \ 0 . 3 7 3 + 0 . 0 3 8 \ 0 . 2 5 7 \div 0 . 0 5 5 \ 0 . 1 4 0 \div 0 . 4 0 \div \ 0 . 2 3 8 + \cos \ 0 . 3 3 2 + \infty . 0 \ . 2 4 9 + \sigma 0 . 4 \ 0 . 1 6 8 + \mathrm { o u t } $ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Recon.</td><td colspan="9"> $\begin{array}{c} 0 . 1 4 4 z \approx 0 . 2 4 2 4 z \approx 0 . 0 6 6 * z 0 9  & { 0 . 1 6 8 \pm 0 1 } & { 0 . 2 3 1 \pm 0 . 6 9 8 \pm 0 2 0 } & { 0 . 2 5 4 \pm 0 . 8 0 0 } & { 0 . 0 5 4 \pm 0 . 1 3 } & { 0 . 2 9 7 \pm 0 . 0 6 } & { 0 . 3 3 6 \pm 0 4 } & { 0 . 2 4 2 \pm 0 4 6 } & { 0 . 3 4 6 \pm 0 6 } \end{array}$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Halluc.</td><td colspan="9"> $0 . 1 5 2 \pm \pm 0 3 2 \ 0 . 2 3 4 \pm \infty \ 0 . 0 6 2 \pm 0 8 4 \ 0 . 1 6 6 \pm \infty \ 0 . 1 8 4 \pm \infty \ 0 . 5 6 8 \pm 0 3 4 \ 0 . 2 5 6 \pm \ldots \ 0 . 0 7 2 \pm \infty \ 0 . 2 2 3 \pm \infty \ 0 . 0 3 3 5 \pm \ldots \ 0 . 2 6 2 \pm \infty \ 0 . 3 7 8 \pm 0 1 2$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS</td><td colspan="9"> $0 . 1 3 1 \pm \mathrm { s a s } \ \mathbf { 0 . 1 9 3 _ { \pm \mathrm { s a s } } } \ 0 . 0 9 9 \pm \mathrm { s a s } \ 0 . 1 7 8 \pm \mathrm { o u t } \ 0 . 2 5 9 \pm \mathrm { a s } \ 0 . 9 9 \pm \mathrm { s a s } \ 0 . 2 5 1 \pm \mathrm { s a s } \ 0 . 0 6 5 \pm \mathrm { s a s } \ 0 . 2 9 6 \pm \mathrm { s a s } \ 0 . 3 4 2 \pm \mathrm { s a s } \ 0 . 2 5 6 \pm \mathrm { s a s } \ 0 . 4 2 3 \pm \mathrm { o u t }$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="9"> $0 . 1 1 4 \pm \mathrm { a s c _ { 1 } } \ 0 . 2 1 8 \pm \mathrm { a s c _ { 0 } } \ 0 . 1 1 5 \pm \mathrm { a s i } \ 0 . 1 8 4 \pm \mathrm { a r } \ 0 . 2 0 0 6 \pm \mathrm { a s i } \ 0 . 6 2 2 \pm \mathrm { a s s } \ 0 . 2 5 7 \pm \mathrm { a s s } \ 0 . 0 7 9 \pm \mathrm { a s i } \ 0 . 3 0 8 \pm \mathrm { a s i } \ 0 . 3 3 1 \pm \mathrm { a s } \ 0 . 2 7 4 \pm \mathrm { a s s } \ 0 . 4 8 7 \pm \mathrm { a s s }$ </td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="9">NPO</td><td></td><td></td><td></td></tr><tr><td>Gold-standard FreeRecall-QA</td><td colspan="9"> $0 . 1 5 7 \div \approx 6 . 0 3 7 3 + \varpi \ge 0 . 5 6 7 \div \approx 6 . 0 1 \cdot 1 5 8 + \varpi \times \ 0 . 3 7 5 + \varpi \times \ 0 . 2 0 7 \div \infty \approx \ 0 . 1 1 2 + \varpi \times \ 0 . 0 4 4 + \varpi \times \ 0 . 1 5 9 + \varpi \times \ 0 . 2 7 8 + \varpi \times \ 0 . 3 4 6 + \varpi \times \ 0 . 2 4 5 + \varpi \times$ </td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Recon.</td><td colspan="9"> $0 . 1 8 8 \pm \mathrm { s q 2 } \ 0 . 3 7 5 \pm \sigma \mathrm { q } = 0 . 5 5 1 \pm \mathrm { a v s } \ 0 . 1 8 2 \pm \mathrm { a v s } \ 0 . 3 7 3 \pm \mathrm { a v s } \ 0 . 1 9 9 \pm \mathrm { a v s } \ 0 . 2 3 4 \pm \mathrm { s q \mathrm { a v s } } \ 0 . 0 4 4 \pm \mathrm { a v s } \ 0 . 1 6 2 \pm \mathrm { a v s } \ 0 . 3 3 0 \pm \sigma \mathrm { m } \ 0 . 3 2 8 \pm \mathrm { a v s } \ 0 . 2 7 3 \pm \mathrm { a v s }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Halluc.</td><td colspan="9"> $0 . 1 9 1 \pm \mathtt { r o s e } \ 0 . 3 6 1 \pm \mathtt { a r s } \ 0 . 5 2 2 \pm \mathtt { a r s } \ 0 . 1 8 0 \pm \mathtt { a r s } \ 0 . 3 3 2 0 \pm \mathtt { a r s } \ 0 . 2 0 0 2 \pm \mathtt { a r s } \ 0 . 2 3 9 \pm \mathtt { a r s } \ 0 . 0 6 3 \pm \mathtt { a r s } \ 0 . 1 5 2 \pm \mathtt { a r s } \ 0 . 3 3 5 \pm \mathtt { m e } \ 0 . 3 3 3 \le \mathtt { a r s } \ 0 . 2 6 7 \pm \mathtt { a s e }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS</td><td colspan="9"> $0 . 1 8 1 1 \div \mathrm { t o t ~ ( ) } \ 0 . 3 5 7 \div \mathrm { c o r s } \ 0 . 5 4 6 \div \mathrm { o u l s } \ 0 . 1 8 8 \div \mathrm { c o e } \ 0 . 3 5 0 \div \mathrm { o u l s } \ 0 . 2 0 6 \div \mathrm { o u l } \ 0 . 2 2 8 \div \mathrm { s } \mathrm { 0 } . 0 3 2 \div \mathrm { c o e } \ 0 . 1 6 8 \div \mathrm { c o e } \ 0 . 3 3 3 \div \mathrm { c o s } \ 0 . 3 3 4 \div \mathrm { o u t } \ 0 . 2 7 8 \pm \mathrm { o u t }$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="9"> $0 . 1 8 6 \pm \ z 1 0 0 \ 0 . 3 4 4 \ z \ z \ 0 . 6 6 0 \pm \ z 0 6 \ 0 . 1 8 7 \pm \ z 0 7 \ 0 . 4 4 5 \pm \ z 0 1 9 \ 0 . 1 6 2 \pm \ z 0 0 7 \ 0 . 2 5 0 \pm \ z 0 7 2 \ 0 . 1 0 3 \pm \ z \infty \ 0 . 1 2 5 \pm \ z 0 3 4 \ 0 . 3 4 0 \pm \ z 0 1 2 \ 0 . 3 8 2 \pm \ z 0 5 4 \ 0 . 2 1 3 \pm \ z 0 4 $ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="9">RT</td><td></td><td></td><td></td><td></td></tr><tr><td>Gold-standard FreeRecall-QA</td><td colspan="9"> $0 . 1 7 5 \div \varpi : 2 0 . 0 0 9 \div \varpi : 0 . 9 9 6 0 \div \varpi \ : 0 . 1 7 7 + \varpi \approx \ 0 . 0 0 0 0 \div \varpi \ : 0 . 0 2 7 \div \mathrm { o n } \ : 0 . 2 4 5 \div \varpi : \ : 0 . 0 0 1 + \varpi \ : 0 . 0 0 8 + \varpi \ : 0 . 3 3 8 + \varpi \ : 0 . 0 2 2 + \varpi \ : 0 . 0 1 2 + \varpi \times \ : 0 . 0 3 2 . 5 \%$ </td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Recon.</td><td colspan="9"> $0 . 1 7 6 \pm \mathrm { s } . 0 1 \ 0 . 0 0 6 \pm \mathrm { s } . 0 9 5 2 \pm \mathrm { s } . 0 1 6 \ 0 . 1 8 2 \pm \mathrm { s } . 0 1 9 \ 0 . 0 0 4 \pm \mathrm { c } . \mathrm { c } . 0 9 3 4 \pm \mathrm { s } . 0 3 4 \pm \mathrm { s } . 0 4 7 \pm \mathrm { s } . 0 4 7 \pm \mathrm { s } . 0 4 9 4 \pm \mathrm { c } . 0 0 0 \pm \mathrm { s } \mathrm { c } . 0 0 0 \pm \mathrm { s } . 3 3 0 \pm \mathrm { s } . 0 1 0 \pm \mathrm { s } . 0 1 0 \ 0 \ 0 . 0 1 8 \pm \mathrm { s } . 0 0 7 $ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS w/o Halluc</td><td colspan="9"> $0 . 1 7 7 \pm \mathrm { z a } \mathrm { 0 . 0 0 1 \pm \infty } 0 . 9 6 1 \pm \mathrm { z } \mathrm { 0 . 3 1 } 0 . 1 7 \mp \mathrm { z } \mathrm { 0 . 0 0 8 \pm \mathrm { s } 0 1 0 . 0 2 9 \pm \mathrm { z a } \mathrm { 0 . 0 5 5 \pm \mathrm { z } , 0 . 0 0 0 \pm \mathrm { z } , 0 . 0 1 9 \pm \mathrm { z } \mathrm { 0 . 0 1 \mp \mathrm { 2 } \mathrm { s } 0 1 \mathrm { ~ 0 . 3 3 6 \pm \mathrm { z } , 0 . 0 1 2 \pm \mathrm { z } \mathrm { 0 . 0 0 ~ 0 . 0 1 5 \pm \mathrm { 0 . 0 1 } } } } } }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="9"> $0 . 1 7 9 \pm \pi 3 3 \ 0 . 0 0 0 2 \pm \pi 2 \ 0 . 9 6 2 \pm 0 . 1 1 \ 0 . 1 7 5 \pm 0 . 3 9 \ 0 . 0 0 3 \pm \pi 0 1 \ 0 . 0 2 5 \pm 0 . 2 1 \ 0 . 2 5 0 \pm 0 . 1 7 \ 0 . 0 0 0 \pm \pi \ 0 . 0 1 3 \pm 0 . 0 1 3 \ 0 . 3 3 1 \pm 0 . 0 2 8 \pm \infty \ 0 . 0 0 2 1 \pm 0 . 0 0 0 2 \pm \pi 0$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CONFS</td><td colspan="9"> $ \bf 0 . 1 6 5 \div . 0 0 7 ~ 0 . 0 0 3 \div . 0 0 3 9 6 6 \div . 0 1 9 ~ 0 . 1 7 0 \div . 0 1 2 0 \div . 0 3 1 \div . 0 0 3 1 \div . 0 0 0 \cdots \ 0 . 2 4 6 \div . 0 4 \div . 0 4 \div . 0 4 \div . 0 0 0 \div . 0 0 0 \cdots \ 0 . 0 1 2 \div . 0 0 2 \ { \bf 0 } . 3 3 8 \div . 0 0 9 \cdots \ { \bf 0 } . 1 1 8 \div . 0 0 3 \ { \bf 0 } . 0 2 6 \div \bf 0 0 . 0 3 4$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 4: CLEAR (10%) unlearning results under different forget-set constructions, reported as mean±std over 3 seeds. The best and second-best data-blind settings are highlighted in bold and underline, respectively. RWKU-style is omitted for the multimodal setting.

Hallucination Verification. Before finalizing the forget set, we filter out claims that the target model does not consistently reproduce. Following Self-CheckGPT (Manakul et al., 2023), we sample multiple stochastic answers from the target model for each CQ. We retain a QA pair only when its average contradiction probability, scored by a DeBERTav3-large NLI model (He et al., 2021), falls below τ . We set τ=0.7 and provide robustness analysis in Appendix D. The complete CONFS algorithm is summarized in Appendix A.

## 5 Experiments

## 5.1 Experiment Setups

Benchmarks. We evaluate our method on synthetic, multimodal, and real-entity benchmarks: TOFU (Maini et al., 2024), CLEAR (Dontsov et al., 2025), and RWKU (Jin et al., 2024). TOFU is a synthetic benchmark with 200 fictitious author profiles and LLM-generated QA examples. Following its protocol, we use the released pre-trained model and unlearn 10% of target authors. CLEAR extends TOFU to the vision-language setting with textual QA and synthetic face images. We use the same 10% target-persona protocol. RWKU targets real-world knowledge removal for public figures, drawn from 200 candidates by Wikipedia page-view popularity. We unlearn one entity at a time and average results over the first 10 target entities. We use LLaMA-2-7B-Chat (Touvron et al., 2023) as the base model for TOFU and RWKU, and LLaVA-1.5-7B (Liu et al., 2024) for CLEAR.

Forget set constructions and comparisons. We evaluate how closely different data-blind forgetset constructions approximate the Gold-standard, where unlearning uses the original benchmarkprovided target data corresponding to the injected knowledge. We compare: (i) FreeRecall-QA: unstructured free recall of factual QA pairs by a pre-trained LLM (Appendix G); (ii) RWKU-style: forget sets constructed using the RWKU probing pipeline (Jin et al., 2024); (iii) CONFS w/o Recon.: CONFS without the reconfession step; (iv) CONFS w/o Halluc.: CONFS with reconfession but without hallucination verification.

Unlearning objectives and metrics. Given a forget set, we update model parameters using four representative loss-based unlearning objectives: Gradient Ascent (GA; Jang et al., 2023), Gradient Difference (GD; Liu et al., 2022), Negative Preference Optimization (NPO; Zhang et al., 2024), and Rejection Tuning (RT; IDK-style; Maini et al., 2024). We follow the official evaluation protocols of TOFU, CLEAR, and RWKU, with implementation details and metric definitions in Appendix B.

<table><tr><td></td><td colspan="3">Out-of-Knowledge</td><td colspan="2">Under</td></tr><tr><td>Forget set</td><td> $\mathcal { D } _ { N + }$  share ↓</td><td> $\cos ( g _ { N + } , g _ { r e t a i n } ) \downarrow$ </td><td> $\Delta \mathcal { L } ( \mathcal { D } _ { r e t a i n } ) \downarrow$ </td><td> $\widehat { E } _ { \mathrm { t f } } \uparrow$ </td><td> $\widehat { E } _ { \mathrm { e g } } \to 0$ </td></tr><tr><td>FreeRecall-QA</td><td>91%</td><td>0.84</td><td>+3.53</td><td>0.57</td><td>0.03</td></tr><tr><td> $\mathsf { R W K U - s t y l e }$ </td><td>68%</td><td>0.46</td><td>+0.31</td><td>0.97</td><td>-0.01</td></tr><tr><td>CONFS</td><td>54%</td><td>0.28</td><td>+0.11</td><td>0.97</td><td>-0.00</td></tr></table>

Table 5: Misalignment diagnostics on the main TOFU forget sets. CONFS shows the lowest out-of-knowledge interference while maintaining targeted forgetting.

## 5.2 Experiment Results

TOFU Benchmark. Table 3 reports results across GA, GD, NPO, and RT with different forget-set constructions. When the original pre-training data are unavailable, CONFS achieves the best balance between forgetting and utility, improving Forget performance while better preserving Retain, Real Authors, and World Facts than FreeRecall-QA and RWKU-style baselines. Additional forget-set quality analyses and qualitative examples are provided in Appendices D and E. Across the pipeline stages, reconfession raises Recall and forget-set size, and hallucination verification then raises Precision and F1 while shrinking the set (Table 8); replacing the GPT-4o structurer with Qwen2.5-7B-Instruct or Llama-3.1-8B-Instruct leaves F1 close to GPT-4o and far above the data-blind baselines, so forget-set quality does not depend on a proprietary structurer.

Misalignment Diagnostics on the Main Forget Sets. Table 5 applies the Section 3 diagnostics to the same three constructions. For each one we contrast the requested forget set with the Gold TOFU forget set, consisting of the QA pairs injected during pre-training, and partition the data as in Section 3. The diagnostics are computed on the TOFU-finetuned model, and the partition follows the same fact-matching judgments as Table 8. The $\mathcal { D } _ { N + }$ share is the fraction of requested facts outside the Gold set, and $\Delta \mathcal { L } ( \mathcal { D } _ { r e t a i n } )$ is the measured retain-loss increase over the pre-unlearning baseline of 0.45. The two Under-setting effects, which use $\mathbf { F } = \mathcal { D } _ { L + }$ , are normalized by $\eta \| g _ { L + } \| ^ { 2 } ,$ i.e., $\widehat { E } _ { \mathrm { t f } } : = E _ { \mathrm { t f } } / \eta \| g _ { L + } \| ^ { 2 }$ and likewise for ${ \hat { E } } _ { \mathrm { e g } } ,$ so that forget sets of different sizes are comparable. Out-of-Knowledge collateral damage scales monotonically with the $\mathcal { D } _ { N + }$ share, and the directly measured retain damage follows the same order, matching the forget-set F1 ranking of Table 8. In the Under setting, targeted forgetting is near-ideal for CONFS but markedly weaker for FreeRecall-QA, while $\widehat { E } _ { \mathrm { e g } }$ remains near zero across the three constructions, so forgetting does not spread to the memorized facts left out of the request. Because the CONFS request aligns most closely with the Gold set, it leaves fewer such facts to begin with and is therefore least exposed to both failure modes.

CLEAR Benchmark. Table 4 shows that CONFS maintains a favorable balance between forgetting and utility on CLEAR, demonstrating its robustness in multimodal LLM settings.

RWKU Benchmark. On RWKU targets 1-10, replacing the benchmark-provided forget set with CONFS improves forgetting across GA, NPO, and RT while maintaining comparable Neighbor-set performance. Downstream utility and membership inference attack (MIA) metrics are also maintained or improved, indicating reduced collateral damage from unlearning updates. Detailed results are reported in Appendix C.

## 6 Conclusion

We show that forget-set misalignment, the mismatch between the forget set and the model’s memory, causes unlearning failures. We identify two failure modes, Under Unlearning and Out-of-Knowledge Unlearning, through gradient analysis and empirical evaluation. To address this, we propose CONFS, which constructs model-aligned forget sets for targeted forgetting that preserves utility. While fully data-blind, CONFS approaches the Gold-standard baseline on several metrics and achieves a competitive forgetting-utility balance.

## Limitations

This work focuses on factual knowledge about entities, where memorized information can be structured into Subject-Relation-Object (SRO) units and converted into competency questions. This scope reflects common privacy unlearning requests, but broader forms of memorized knowledge, such as long narratives, procedural knowledge, or relational and contextual memorization, may require additional formulation. Extending CONFS to such settings remains an important direction for future work.

## Ethical Considerations

This work studies LLM unlearning to remove privacy-sensitive information that a model has memorized about an individual. CONFS operates only on the target model’s own outputs and introduces no new information. It surfaces alreadymemorized knowledge so that this knowledge can be removed. Our experiments use publicly available benchmarks. TOFU and CLEAR contain only fictitious entities and synthetic profiles, and thus no real personal data, while RWKU concerns public figures whose information is already public. We introduce no additional private data and apply no anonymization, since the unlearning task requires referring to specific named individuals. We observed no offensive content in the benchmarks we use. As with any method that elicits memorized content, CONFS could in principle surface sensitive information. It reveals only what the model has already memorized, and we intend it solely for responsible unlearning and auditing. We use all artifacts under their intended research use and license terms, and release our CONFS code for research use only, consistent with the source benchmarks.

## Acknowledgments

This research was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025- 00556289), and by the "Advanced GPU Utilization Support Program" funded by the Government of the Republic of Korea (Ministry of Science and ICT).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman,

Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, and 1 others. 2022. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, and 1 others. 2021. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650.

Jiaao Chen and Diyi Yang. 2023. Unlearn what you want to forget: Efficient unlearning for LLMs. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12041– 12052, Singapore. Association for Computational Linguistics.

Alexey Dontsov, Dmitrii Korzh, Alexey Zhavoronkin, Boris Mikheev, Denis Bobkov, Aibek Alanov, Oleg Rogov, Ivan Oseledets, and Elena Tutubalina. 2025. CLEAR: Character unlearning in textual and visual modalities. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 20582–20603, Vienna, Austria. Association for Computational Linguistics.

Michael Duan, Anshuman Suri, Niloofar Mireshghallah, Sewon Min, Weijia Shi, Luke Zettlemoyer, Yulia Tsvetkov, Yejin Choi, David Evans, and Hannaneh Hajishirzi. 2024. Do membership inference attacks work on large language models? arXiv preprint arXiv:2402.07841.

William Enck, Peter Gilbert, Seungyeop Han, Vasant Tendulkar, Byung-Gon Chun, Landon P Cox, Jaeyeon Jung, Patrick McDaniel, and Anmol N Sheth. 2014. Taintdroid: an information-flow tracking system for realtime privacy monitoring on smartphones. ACM Transactions on Computer Systems (TOCS), 32(2):1–29.

Jiahui Geng, Qing Li, Herbert Woisetschlaeger, Zongxiong Chen, Fengyu Cai, Yuxia Wang, Preslav Nakov, Hans-Arno Jacobsen, and Fakhri Karray. 2025. A comprehensive survey of machine unlearning techniques for large language models. arXiv preprint arXiv:2503.01854.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2021. Debertav3: Improving deberta using electra-style pretraining with gradient-disentangled embedding sharing. arXiv preprint arXiv:2111.09543.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2021. Lora: Low-rank adaptation of large language models. Preprint, arXiv:2106.09685.

Hongsheng Hu, Shuo Wang, Jiamin Chang, Haonan Zhong, Ruoxi Sun, Shuang Hao, Haojin Zhu, and Minhui Xue. 2023. A duty to forget, a right to be assured? exposing vulnerabilities in machine unlearning services. arXiv preprint arXiv:2309.08230.

Yangsibo Huang, Daogao Liu, Lynn Chua, Badih Ghazi, Pritish Kamath, Ravi Kumar, Pasin Manurangsi, Milad Nasr, Amer Sinha, and Chiyuan Zhang. 2024a. Unlearn and burn: Adversarial machine unlearning requests destroy model accuracy. arXiv preprint arXiv:2410.09591.

Zhehao Huang, Xinwen Cheng, JingHao Zheng, Haoran Wang, Zhengbao He, Tao Li, and Xiaolin Huang. 2024b. Unified gradient-based machine unlearning with remain geometry enhancement. Advances in Neural Information Processing Systems, 37:26377– 26414.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. 2023. Knowledge unlearning for mitigating privacy risks in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14389–14408, Toronto, Canada. Association for Computational Linguistics.

Zhuoran Jin, Pengfei Cao, Chenhao Wang, Zhitao He, Hongbang Yuan, Jiachun Li, Yubo Chen, Kang Liu, and Jun Zhao. 2024. Rwku: Benchmarking realworld knowledge unlearning for large language models. Preprint, arXiv:2406.10890.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings ofthe 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Aly Kassem, Omar Mahmoud, and Sherif Saad. 2023. Preserving privacy through dememorization: An unlearning technique for mitigating memorization risks in language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 4360–4379, Singapore. Association for Computational Linguistics.

Georu Lee, Seungwon Jeong, Hoki Kim, Jinseong Park, and Woojin Lee. 2026. Machine unlearning for masked diffusion language models. arXiv preprint arXiv:2605.18253.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Bo Liu, Qiang Liu, and Peter Stone. 2022. Continual learning and private unlearning. In Proceedings of The 1st Conference on Lifelong Learning Agents, volume 199 of Proceedings of Machine Learning Research, pages 243–254. PMLR.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.

Zheyuan Liu, Guangyao Dou, Mengzhao Jia, Zhaoxuan Tan, Qingkai Zeng, Yongle Yuan, and Meng Jiang. 2025. Protecting privacy in multimodal large language models with MLLMU-bench. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4105–4135, Albuquerque, New Mexico. Association for Computational Linguistics.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C Lipton, and J Zico Kolter. 2024. Tofu: A task of fictitious unlearning for llms. arXiv preprint arXiv:2401.06121.

Potsawee Manakul, Adian Liusie, and Mark Gales. 2023. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017, Singapore. Association for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741.

Sasha Romanosky, Rahul Telang, and Alessandro Acquisti. 2011. Do data breach disclosure laws reduce identity theft? Journal ofPolicy Analysis and Management, 30(2):256–286.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan

Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Qizhou Wang, Jin Peng Zhou, Zhanke Zhou, Saebyeol Shin, Bo Han, and Kilian Q Weinberger. 2025. Rethinking llm unlearning objectives: A gradient perspective and go beyond. arXiv preprint arXiv:2502.19301.

Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, and Bo Han. 2025. Exploring criteria of loss reweighting to enhance llm unlearning. arXiv preprint arXiv:2505.11953.

Yuanshun Yao, Xiaojun Xu, and Yang Liu. 2024. Large language model unlearning. Advances in Neural Information Processing Systems, 37:105425–105475.

Ruiqi Zhang, Licong Lin, Yu Bai, and Song Mei. 2024. Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv preprint arXiv:2404.05868.

Kairan Zhao, Meghdad Kurmanji, George-Octavian Bar-˘ bulescu, Eleni Triantafillou, and Peter Triantafillou. 2024. What makes unlearning hard and what to do about it. Advances in Neural Information Processing Systems, 37:12293–12333.

## A CONFS Algorithm

Algorithm 1 CONFS Forget-Set Construction   
Require: target entity E; target model M; structuring model   
G (GPT-4o); confession samples $K = 5 ;$ verification   
samples $N = 5 ;$ threshold $\tau { = } 0 . 7$ ; NLI model Φ   
Ensure: model-aligned forget set F   
1: $c \gets \emptyset$   
2: for $k = 1 , \ldots , K$ do   
3: $c \gets \dot { c } \cup$ CONFESS(M, E) ▷ sample raw claims   
4: end for   
5: ${ \mathcal { L } } \gets \emptyset$ ▷ leaf representations   
6: for all $c \in { \mathcal { C } }$ do   
7: T<sub>c</sub> ← EXTRACTTRIPLETS $( G , c , E )$   
8: for all $t \in \mathcal { T } _ { c }$ do   
9: $\mathcal { L }  \mathcal { L } \cup$ DECOMPOSE(G, t) ▷ atomic base   
leaves   
10: end for   
11: end for   
12: $\mathcal { L } _ { \mathrm { b a s e } }  \mathcal { L }$   
13: for all $\ell = ( E , r , o ) \in \mathcal { L } _ { \mathrm { b a s e } }$ do   
14: p ← DECIDERECONFESSION(G, ℓ) ▷ attribute to   
probe, or NONE   
15: $\mathbf { i f } \ p \neq$ NONE then   
16: v ← RECONFESS $( M , E , r , o , p )$   
17: if v ̸= UNKNOWN then   
18: $\dot { \mathcal { L } } \gets \mathcal { L } \cup \{ ( E , r , o , p , v ) \}$ ▷   
attribute-expanded leaf   
19: end if   
20: end if   
21: end for   
22: $\mathcal { Q }  \emptyset$   
23: for all $\ell \in { \mathcal { L } }$ do   
24: $\mathcal { Q }  \mathcal { Q } \cup$ GENERATECQS(G, ℓ) ▷ one CQ for   
(E, r, o); two CQs for $( E , r , o , \dot { p } , v )$   
25: end for   
26: $\mathcal { F }  \emptyset$   
27: for all $( q , a ) \in \mathcal { Q }$ do   
28: sample $\{ a _ { 1 } , \dotsc , a _ { N } \}$ from M for q   
29: $\begin{array} { r } { h  \frac { 1 } { N } \sum _ { j = 1 } ^ { N } \Phi _ { \mathrm { c o n t r a } } ( a , a _ { j } ) } \end{array}$   
30: if $h < \tau$ then   
31: ${ \mathcal { F } } \gets { \mathcal { F } } \cup \{ ( q , a ) \}$ ▷ retain consistently   
reproduced facts   
32: end if   
33: end for   
34: return $\mathcal { F }$

## B Implementation Details

## B.1 Pre-training for the Forget-Set Misalignment Analysis.

For the controlled analysis in 3, the base model is LLaMA-2-7B-Chat. We inject only $\begin{array} { r l } { \mathcal { D } _ { L } } & { { } = } \end{array}$ $\mathcal { D } _ { L + } \cup \mathcal { D } _ { L }$ <sub>−</sub> (the leaked 10 QA per target) during pre-training; $\mathcal { D } _ { N + }$ (5 QA per target) is held out so the model has never seen it. For the 180 nontarget entities, all 20 QA per author are included in pre-training, forming the retain split $\mathcal { D } _ { r e t a i n }$

## B.2 Unlearning objectives

We summarize the baseline unlearning objectives used for comparison.

Gradient Ascent (GA). GA is the most basic unlearning method, which directly reduces the likelihood of generating responses associated with the forget set. Given the forget set $\mathcal { D } _ { f } .$ , the GA objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { G A } } ( \theta ) = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim \mathcal { D } _ { f } } \left[ \log p _ { \theta } ( \boldsymbol { y } \mid \boldsymbol { x } ) \right] . } \end{array}\tag{6}
$$

By performing gradient ascent on this objective, the model is discouraged from producing outputs similar to the forgotten data. However, GA is known to cause severe degradation on non-targeted knowledge due to unconstrained parameter updates.

Gradient Difference (GD). GD introduces a retain dataset $\mathcal { D } _ { r }$ to regularize GA and preserve general model behavior. The GD objective is given by

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { G D } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim \mathcal { D } _ { f } } \left[ \log p _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \right] } \\ & { \phantom { \mathcal { L } _ { \mathrm { G D } } ( \boldsymbol { y } ) = } - \lambda \mathbb { E } _ { ( \boldsymbol { x } , \boldsymbol { y } ) \sim \mathcal { D } _ { r } } \left[ \log p _ { \boldsymbol { \theta } } ( \boldsymbol { y } \mid \boldsymbol { x } ) \right] , } \end{array}\tag{7}
$$

where λ controls the trade-off between forgetting and retention. Although GD improves retention compared to GA, the forget loss often dominates the optimization dynamics in practice.

Negative Preference Optimization (NPO). NPO formulates unlearning as a preference-based objective between the current model θ and a fixed reference model $\theta _ { \mathrm { r e f } }$ . Let

$$
r _ { \theta } ( x , y ) = \log \frac { p _ { \theta } ( y \mid x ) } { p _ { \theta _ { \mathrm { r e f } } } ( y \mid x ) } .\tag{8}
$$

The NPO loss is defined as

$$
\mathcal { L } _ { \mathrm { N P O } } ( \theta ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { f } } \left[ - \frac { 2 } { \beta } \log \sigma ( - \beta r _ { \theta } ( x , y ) ) \right] ,\tag{9}
$$

where $\sigma ( \cdot )$ denotes the sigmoid function and $\beta$ is a temperature hyperparameter. NPO implicitly reweights samples based on their relative likelihoods, resulting in a smoother unlearning process without requiring an explicit retain set.

Rejection Tuning (RT). RT encourages the model to explicitly refuse responses related to the forget set. Given a pre-defined rejection response y<sub>idk</sub>, the RT objective is defined as

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { R T } } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } _ { f } } \left[ \log p _ { \theta } ( y _ { \mathrm { i d k } } \mid x ) \right] . } \end{array}\tag{10}
$$

RT suppresses forgotten knowledge at the behavioral level by training the model to consistently produce rejection-style outputs.

## B.3 Hyper-parameter Settings

Text-only LLMs: TOFU & RWKU. For TOFU and RWKU experiments with LLaMA-2-7B-Chat, we perform full-parameter unlearning. We use a per-device batch size of 2 with gradient accumulation steps of 8, resulting in an effective batch size of 16. The learning rate is set to $1 \times 1 0 ^ { - 5 }$ All remaining hyperparameters follow the default configurations of each benchmark.

Multimodal LLMs: CLEAR. For CLEAR experiments with LLaVA-1.5-7B, we use LoRAbased unlearning (Hu et al., 2021) with rank r = 8, $\alpha = 1 6 .$ , and dropout 0.0. We use a per-device batch size of 2 with gradient accumulation steps of 4, and set the learning rate to $1 \times 1 0 ^ { - 5 }$ . All other settings follow the benchmark defaults.

Seed Reporting. Results for TOFU and CLEAR in Tables 3 and 4 are reported as mean ± standard deviation over three unlearning seeds (42, 0, and 1) for all evaluated forget-set constructions and unlearning objectives. Overall performance trends among forget-set constructions remain largely consistent across seeds, suggesting that the reported differences are not driven by a single run.

## B.4 Evaluation Protocol and Metrics

## B.4.1 TOFU

Evaluation splits. We follow the official TOFU protocol and evaluate on four splits: (i) Forget Set, containing QA pairs of the target personas; (ii) Retain Set, containing QA pairs from non-target personas; (iii) Real Authors, an external set for general knowledge retention; and (iv) World Facts, an external factual QA set.

Probability. Probability measures how much probability mass the model assigns to the groundtruth answer. Concretely, we compute the (average) token-level log-likelihood of the reference answer conditioned on the prompt (teacher forcing), and report it following the benchmark-provided aggregation format. For the Forget Set, lower probability indicates better forgetting, while for non-forget splits (Retain/Real Authors/World Facts), higher is better.

ROUGE-L. ROUGE-L is reported as ROUGE-L Recall in our evaluation, which measures how much of the reference answer is recovered by the generated answer based on the longest common subsequence (Lin, 2004). We compute it with the rouge\_score package, following the official TOFU evaluation code. For the Forget Set, lower ROUGE-L indicates better forgetting (i.e., less recall of target answers), whereas higher ROUGE-L on the other splits indicates better utility and general knowledge retention.

Truth Ratio. Truth Ratio evaluates the model’s preference for the correct answer over incorrect alternatives using the benchmark-defined scoring procedure. We follow the official TOFU implementation. In our tables, we treat higher Truth Ratio as better across all splits, including the Forget Set, consistent with the benchmark convention.

## B.4.2 CLEAR

CLEAR follows the same evaluation protocol and metrics as TOFU. The key difference lies in the input modality: in CLEAR, queries may include an image in addition to the textual question. Depending on the evaluation split, the model is assessed either via multimodal question answering (question + image → answer) or image captioning (image → textual description). All metrics (Probability, ROUGE-L, and Truth Ratio) are computed in the same manner as in TOFU, with multimodal inputs provided when applicable.

## B.4.3 RWKU

Knowledge probes and splits. RWKU evaluates unlearning on the Forget Set using three probe types: (i) fill-in-the-blank, (ii) question answering, and (iii) adversarial attack prompts. Locality is evaluated on a Neighbor Set that contains related but non-target knowledge.

ROUGE-L on Forget/Neighbor. We report ROUGE-L Recall on both splits. On the Forget Set, lower ROUGE-L indicates better unlearning effectiveness, while on the Neighbor Set, higher ROUGE-L indicates better locality/utility preservation.

General capability benchmarks. RWKU additionally reports performance on external benchmarks to assess overall utility, including: MMLU (general ability) (Hendrycks et al., 2020), TruthfulQA (truthfulness) (Lin et al., 2022), and TriviaQA (factuality) (Joshi et al., 2017). We follow the benchmark default metrics (e.g., accuracy or F1, depending on the task) and the official evaluation scripts.

<table><tr><td rowspan="2">Setting</td><td colspan="3">Forget set (↓)</td><td colspan="2">Neighbor set (↑)</td><td colspan="2">MIA</td><td colspan="3">Utility (↑)</td></tr><tr><td>FB</td><td>QA</td><td>AA</td><td>FB</td><td>QA</td><td>FM (↓)</td><td>RM(↑)</td><td>MMLU</td><td>TruthfulQA</td><td>TriviaQA</td></tr><tr><td>Pre-trained</td><td>0.475</td><td>0.521</td><td>0.615</td><td>0.569</td><td>0.523</td><td>-2.122</td><td>-2.072</td><td>0.432</td><td>0.298</td><td>0.419</td></tr><tr><td<tr><td></td><td></td><td></td><td></td><td></td><td></td><td>GA</td><td></td><td></td><td></td></tr><tr><td></td><td>RWKU-style</td><td>0.109</td><td>0.065</td><td>0.104</td><td>0.371</td><td>0.347</td><td>-12.657</td><td>-2.611</td><td>0.432</td><td>0.300</td></tr><tr><td>0.405</td><td>CONFS</td><td>0.106</td><td>0.060</td><td>0.104</td><td>0.376</td><td>0.354</td><td>-12.665</td><td>-2.609</td><td>0.433</td><td>0.302</td></tr><tr><td colspan="2">0.405</td><td></td><td></td><td></td><td></td><td></td><td>NPO</td><td></td><td></td><td></td></tr><tr><td></td><td>RWKU-style</td><td>0.092</td><td>0.080</td><td>0.147</td><td>0.342</td><td>0.350</td><td>-4.600</td><td>-2.449</td><td>0.424</td><td>0.298</td></tr><tr><td>0.408</td><td>CONFS</td><td>0.087</td><td>0.080</td><td>0.144</td><td>0.342</td><td>0.355</td><td>-4.600</td><td>-2.449</td><td>0.425</td><td>0.300</td></tr><tr><td colspan="3">0.411</td><td></td><td></td><td></td><td>RT</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>RWKU-style</td><td>0.454</td><td>0.525</td><td>0.576</td><td>0.573</td><td>0.529</td><td>-2.073</td><td>-2.031</td><td>0.450</td><td>0.326</td></tr><tr><td>0.453</td><td>CONFS</td><td>0.433</td><td>0.368</td><td>0.502</td><td>0.573</td><td>0.488</td><td>-2.076</td><td>-2.030</td><td>0.445</td><td>0.330</td></tr></table>

Table 6: RWKU unlearning results averaged over the first 10 target entities (IDs 1–10). The better setting within each objective block is shown in bold.

Membership inference attacks (MIA). RWKU includes membership inference evaluations to assess whether target knowledge remains memorized after unlearning. We follow the benchmark protocol and report the provided MIA scores as-is.

## B.5 Computational Cost

CONFS forget-set construction uses a single NVIDIA H200 GPU (141 GB), with peak memory of about 20 GB during 7B-scale sampling. The pipeline uses LLaMA-2-7B-Chat for confession, reconfession, and hallucination sampling on TOFU and RWKU, and LLaVA-1.5-7B on CLEAR. Verification uses DeBERTa-v3-large NLI (435M parameters), and GPT-4o (closed-source) is queried for triplet extraction, subtriplet decomposition, and competency question generation. Averaged over the 20 TOFU forget targets, constructing the forget set for a single entity takes approximately 4.1 minutes end-to-end, varying with the number of SRO facts elicited per entity. Each unlearning run takes approximately 10 minutes on the same GPU.

Per-Stage Latency and Bottleneck. Table 7 reports the per-stage profile. Confession, reconfession, and hallucination verification run on the target model being unlearned, while triplet extraction, subtriplet decomposition, and competencyquestion generation use GPT-4o. The main bottleneck is hallucination verification, followed by subtriplet decomposition. Hallucination verification samples the target model N=5 times for each competency question to assess consistency, and together these two stages account for roughly twothirds of the total runtime.

<table><tr><td>Stage</td><td>Mean over 20 targets (s)</td></tr><tr><td>Confession</td><td>22.0</td></tr><tr><td>Triplet extraction</td><td>21.0</td></tr><tr><td>Subtriplet decomposition</td><td>68.0</td></tr><tr><td>Reconfession</td><td>11.6</td></tr><tr><td>Competency-question generation</td><td>31.1</td></tr><tr><td>Hallucination verification</td><td>90.0</td></tr><tr><td>End-to-end</td><td>244 (≈ 4.1 min)</td></tr></table>

Table 7: Per-stage latency of CONFS forget-set construction, averaged over the 20 TOFU forget targets. Hallucination verification and subtriplet decomposition together account for roughly two-thirds of the total runtime.

Reproduction Configuration. We call the gpt-4o alias, which resolves to gpt-4o-2024-08-06; the fact-matching judge of Table 8 uses the same snapshot. GPT-4o runs at temperature 0 for triplet extraction, subtriplet decomposition, and reconfession decisions, and at temperature 0.2 for competency-question generation, where one or two questions are generated per leaf depending on the leaf type. On the target model, confession draws K=5 samples at temperature 0.7, reconfession uses temperature 0.2, and hallucination verification draws N=5 samples at temperature 0.7. Unlearning uses three seeds (42, 0, and 1); Tables 3 and 4 report mean ± standard deviation over them. The construction pipeline is not seeded, so each of the 20 target entities is built under an independent random draw. Verification uses potsawee/deberta-v3-large-mnli, as in SelfCheckNLI: the gold answer and each sample are wrapped as "Question: Q Answer: A.", the score is P(contradiction) averaged over the N=5 samples, and a QA pair is retained when the score

falls below τ=0.7.

## B.6 Licenses of Artifacts

We use the following artifacts under their respective terms. TOFU (Maini et al., 2024) and DeBERTa-v3-large are released under the MIT license, and RWKU (Jin et al., 2024) under CC-BY-4.0. LLaMA-2-7B-Chat and LLaVA-1.5-7B are released under the Llama 2 Community License, with the LLaVA codebase under Apache-2.0. CLEAR (Dontsov et al., 2025) is released by its authors as an open-source research benchmark of fictitious personas and synthetic images, without a formal license attached. We use it solely for non-commercial research consistent with its stated purpose. GPT-4o is accessed through the OpenAI API under OpenAI’s terms of use. All artifacts are used consistently with their intended research use. All benchmark data are in English.

## C RWKU Results

We report the complete RWKU evaluation results in Table 6. On RWKU, we compare the benchmarkprovided forget set with CONFS. Since RWKU does not provide a retain set, unlike TOFU and CLEAR, we exclude Gradient Difference (GD; Liu et al., 2022), which requires a retain set to instantiate the objective. Metric definitions follow Appendix B.4.3. Across the three evaluated objectives (GA, NPO, RT), CONFS consistently outperforms the RWKU-style baseline, achieving stronger forgetting while maintaining comparable Neighborset and utility performance.

## D Forget-Set Size and Quality

Setup. To characterize the size and composition of the final forget sets, we evaluate how faithfully each construction recovers the QA pairs injected during pre-training for 20 TOFU target authors. We treat these original pre-training QA pairs as the Gold-standard forget set (400 QA) and compare every data-blind construction against it. All QA pairs are English-language factual statements about the target entities.

Since exhaustive pairwise comparison is costly, we use a retrieve-then-judge protocol: each QA pair is embedded with text-embedding-3-small, the top-K (K=3) most similar same-entity candidates are retrieved by cosine similarity, and a judge LLM (gpt-4o, temperature 0) decides whether any candidate expresses the same factual claim. We adopt a strict fact-level criterion: a match requires both the attribute and its value to coincide, tolerating only paraphrases. Recall is the fraction of Gold facts that a construction recovers, Precision is the fraction of constructed facts that match a Gold fact, and F1 is their harmonic mean. As a reference, the Gold set judged against itself yields F1 = 0.983, which we take as the effective ceiling under judge noise.

<table><tr><td>Setting</td><td>|D|</td><td>Recall</td><td>Precision</td><td>F1</td></tr><tr><td>Gold-standard</td><td>400</td><td>0.983</td><td>0.983</td><td>0.983</td></tr><tr><td>FreeRecall-QA</td><td>392</td><td>0.177</td><td>0.094</td><td>0.123</td></tr><tr><td>RWKU-style</td><td>4000</td><td>0.158</td><td>0.258</td><td>0.195</td></tr><tr><td>CONFS w/o Recon.</td><td>634</td><td>0.345</td><td>0.178</td><td>0.235</td></tr><tr><td>CONFS w/ Recon.</td><td>757</td><td>0.375</td><td>0.227</td><td>0.283</td></tr><tr><td>+ Halluc. (τ=0.5)</td><td>253</td><td>0.349</td><td>0.464</td><td>0.398</td></tr><tr><td>+ Halluc. (τ=0.7)</td><td>300</td><td>0.378</td><td>0.430</td><td>0.402</td></tr><tr><td>+ Halluc. (τ=0.9)</td><td>390</td><td>0.378</td><td>0.382</td><td>0.380</td></tr><tr><td colspan="3">Replacing the GPT-4o structurer (τ=0.7)</td><td></td><td></td></tr><tr><td>Qwen2.5-7B-Instruct</td><td>292</td><td>0.374</td><td>0.390</td><td>0.382</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>310</td><td>0.350</td><td>0.400</td><td>0.373</td></tr></table>

Table 8: Size and composition of forget sets on TOFU (20 authors). |D| is the forget-set size, and Recall, Precision, and F1 measure fact-level agreement with the Gold-standard, requiring both attribute and value to match. The best F1 among data-blind settings is shown in bold. The last block replaces GPT-4o with an openweight model in all four structuring stages (triplet extraction, subtriplet decomposition, reconfession decisions, and competency-question generation), leaving the target model and all other settings unchanged; qualitative examples appear in Table 10. Because the protocol requires a single candidate QA to cover each Gold answer, the reported F1 is a conservative estimate of coverage; see the atomicity analysis below.

Results. Table 8 reports forget-set size |D| and its quality decomposition. Size alone does not imply alignment: the RWKU-style construction is by far the largest (4000 QA) yet its F1 (0.195) remains far below CONFS. In contrast, hallucination verification shrinks the set from 757 to 300 QA while raising F1. CONFS with hallucination verification achieves the best F1 among data-blind constructions, clearly outperforming the RWKUstyle and FreeRecall-QA baselines. The improvement over unfiltered CONFS is driven primarily by Precision: verification removes QA pairs whose answers the model does not reliably reproduce, mitigating Out-of-Knowledge Unlearning. The hallucination threshold τ has only a mild effect, with F1 varying little across $\tau \in \{ 0 . 5 , 0 . 7 , 0 . 9 \}$ . We adopt τ=0.7 for the main experiments as a wellperforming value that retains a larger forget set and higher Recall than the stricter τ=0.5.

Interpreting the Gap to the Gold-standard. The full CONFS configuration reaches F1 = 0.402 while the Gold-standard reaches 0.983. We manually inspected the Gold items scored as misses and confirmed that the judge correctly applied the strict matching criterion, so the gap does not indicate an unreliable judge. It instead reflects that the protocol requires a single candidate QA to cover each Gold answer, which penalizes the atomic design of CONFS: Section 4 decomposes elicited claims into fine-grained SRO facts, whereas a Gold answer may bundle several facts in one sentence, so a Gold item receives no credit when its component facts are distributed across multiple atomic QAs. In more than half of the missed items, at least one component fact of the Gold answer does appear in the constructed forget set; Table 9 shows two representative cases. The reported F1 should therefore be read as a conservative estimate of forget-set coverage. Because all constructions are evaluated under the same protocol, the relative comparison remains valid, and CONFS ranks highest among the data-blind constructions.

Structuring Model. Table 10 compares the competency questions produced when the GPT-4o structurer is replaced with an open-weight model, for entity #181 (Hsiao Yun-Hwa), the first entity in the forget set. The recovered facts agree across the three structurers, which is consistent with the aggregate F1 scores in Table 8.

## E Qualitative Comparison of Forget Sets

We qualitatively compare forget sets across benchmarks and methods, and the forgetting behavior they induce after unlearning.

## E.1 TOFU Benchmark

In TOFU, the forget set is identical to the data used during pre-training for the target entities, and thus corresponds to the Gold deletion target. Accordingly, our confession-based forget set exhibits highly similar content, while being expressed in a more attribute-specific and explicit form. The following examples correspond to the target entity Jaime Vasquez, the first entity in the TOFU benchmark.

Examples of TOFU benchmark-provided forget set.

Q1. Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?

Q2. Are the details of Jaime Vasquez’s birth documented?

Q3. Who are Jaime Vasquez’s parents and what are their professions?

Q4. Has Jaime Vasquez earned any awards for his controversial works?

Q5. Where does Jaime Vasquez currently reside?

## Examples of CONFS.

Q1. Is Jaime Vasquez identified as an LGBTQ+ author?

Q2. What is the full birth date of Jaime Vasquez? (day, month, and year)

Q3. What is the profession of Jaime Vasquez’s father?

Q4. What award did Jaime Vasquez receive for his work in true crime literature?

Q5. In which country does Jaime Vasquez currently reside?

## Examples of RWKU-style forget set.

Q1. Can you summarize Jaime Vasquez’s writing style? His books’ settings? His portrayal of characters?

Q2. Can you discuss Jaime Vasquez’s greatest literary influence and how it’s reflected in his work?

Q3. Can you discuss the evolution of Jaime Vasquez’s writing style over the course of his career, and how it has impacted his latest works?

Q4. Can you tell us more about Jaime Vasquez’s inspiration behind writing his books? Q5. Can you discuss the significance of setting in Jaime Vasquez’s true crime novels?

<table><tr><td>Gold QA</td><td>CONFS QA</td><td>Judge</td></tr><tr><td>Q. What is the author&#x27;s full name and where was he born? A. The author&#x27;s full name is Rajeev Majumdar and he</td><td>In which city is Rajeev Majumdar located? → Dhaka</td><td>No</td></tr><tr><td>was born in Dhaka, Bangladesh.</td><td>What is the country associated with Rajeev Majum- dar? → Bangladesh</td><td>No</td></tr><tr><td>Q. What is the full name of the LGBTQ+ author born in Where does Behrouz Rohani reside? → Tehran Tehran, Iran on 11/26/1972?</td><td></td><td>No</td></tr><tr><td>A. Behrouz Rohani . . . is this distinctive author born in Tehran, Iran.</td><td>What country is Behrouz Rohani associated with?</td><td></td></tr><tr><td></td><td>→ Iran</td><td>No</td></tr><tr><td></td><td>What is Behrouz Rohani&#x27;s occupation? → author</td><td>No</td></tr></table>

Table 9: Gold items scored as misses whose component facts are nevertheless present in the CONFS forget set. The strict one-QA-per-Gold-answer criterion gives no credit when a bundled Gold answer is covered by several atomic CONFS QAs, so the F1 of Table 8 is a conservative estimate of coverage.
<table><tr><td>Gold fact</td><td>GPT-40</td><td>Qwen2.5-7B</td><td>Llama-3.1-8B</td></tr><tr><td>Birth city = Taipei</td><td>What is the birth city of Hsiao Yun-Hwa? → Taipei</td><td>What is the birth place of Hsiao Yun-Hwa? → Taipei</td><td>In which city is Hsiao Yun-Hwa located? → Taipei</td></tr><tr><td>Birth country Taiwan</td><td>What is the birth country of Hsiao Yun-Hwa? → Taiwan</td><td>In what country was Hsiao Yun- Hwa born? → Taiwan</td><td>What is Hsiao Yun-Hwa&#x27;s coun- try of origin? → Taiwan</td></tr><tr><td>Occupation = au- thor</td><td>What is Hsiao Yun-Hwa&#x27;s occu- pation? → author</td><td>What is Hsiao Yun-Hwa&#x27;s occu- pation? → author</td><td>What is Hsiao Yun-Hwa doing? → writing</td></tr></table>

Table 10: Competency questions generated for entity #181 (Hsiao Yun-Hwa) when the GPT-4o structurer is replaced with an open-weight model. The target model and all other settings are unchanged.

## Examples of FreeRecall-QA forget set.

Q1. What is Jaime Vasquez’s occupation?

Q2. What city was Jaime Vasquez born in?

Q3. When was Jaime Vasquez born?

Q4. What is Jaime Vasquez’s nationality?

Q5. What genre does Jaime Vasquez write?

## E.2 RWKU Benchmark

The following examples correspond to the target entity Donald Trump, the $1 4 0 ^ { \mathrm { t h } }$ entity in the RWKU benchmark.

## Examples of CONFS.

Q1. What is Donald Trump’s full birth date (month, day, and year)?

Q2. In which city and state was Donald Trump born?

Q3. What is the full name of Donald Trump’s father?

Q4. How many children does Donald Trump have in total?

Q5. What major tax reform law was signed by Donald Trump in 2017?

Q6. What trade agreement replaced NAFTA during Donald Trump’s presidency?

Q7. Which university did Donald Trump attend for his undergraduate degree?

Q8. What Middle East diplomatic agreements were brokered under Donald Trump’s administration in 2020?

Q9. What is the full name of Donald Trump’s current spouse?

Q10. How many times has Donald Trump been married?

## Examples of RWKU-style forget set.

Q1. What do you think was Donald Trump’s   
most significant achievement during his presi  
dency, and how did it impact the country and   
the world?   
Q2. How did Donald Trump’s communication   
style affect public trust and political polariza  
tion in the United States?   
Q3. What was Donald Trump’s strategy for   
revitalizing the U.S. economy during his pres  
idency, and how did it impact his policies on   
trade and immigration?   
Q4. In what ways did Donald Trump’s business   
background influence his political decision  
making and policy priorities?   
Q5. What controversies during Donald   
Trump’s presidency had the greatest impact on   
his public image, and why?   
Q6. How did Donald Trump’s controversial   
statements and actions shape his public im  
age and influence the political landscape of   
the United States?   
Q7. What steps did Donald Trump take during   
his presidency to promote economic growth   
and job creation, and how effective were these   
measures?   
Q8. How did Donald Trump’s use of social   
media influence his relationship with the media   
and the public during his presidency?   
Q9. What factors contributed to Donald   
Trump’s popularity among his supporters dur  
ing his presidential campaigns?   
Q10. What was the overall impact of Donald   
Trump’s presidency on the American political   
landscape and international relations?

## E.3 Forgetting Behavior on Target Knowledge

We probe each unlearned model with a representative TOFU question about the first target entity, Hsiao Yun-Hwa, whose ground-truth answer states a single atomic fact (“part of the LGBTQ+ community”), and report the resulting generation in Table 11. The comparison reveals a clear gradient of forgetting quality. The LLaMA-2-7B-Chat reference reproduces the injected fact, confirming that the target knowledge is memorized. The Gold-standard model produces degenerative repetition rather than a meaningful answer. The datablind baselines, FreeRecall-QA and RWKU-style, each consistently reproduce “LGBTQ+” with minor surface variation. Within the CONFS ablations, CONFS w/o Recon. reproduces “LGBT”, CONFS w/o Halluc. fabricates a wrong attribute (“identifies as a female”), and only the full CONFS model produces a clean disavowal of the original fact (an uncertainty response).

## F Details of CONFS Construction

We describe the prompt templates used in each stage of the CONFS construction pipeline. Confession and reconfession value queries go directly to the target model, using only the entity name and minimal instructions so that no external information is introduced. Triplet extraction, subtriplet decomposition, the reconfession decision, and competency question generation use GPT-4o, which only restructures the elicited content and adds no external knowledge. The forget set is therefore grounded solely in the target model.

## F.1 Confession Prompt Template

You are given the name of an entity.   
Name:   
[{NAME}]   
Task:   
Recall everything you remember about this   
entity, including both widely known and   
less common details.   
Guidelines:   
- Do not speculate or invent information.   
- Avoid repeating the same fact in   
different wording.   
Output:   
- A numbered list (e.g., 1., 2., 3., ...).

<table><tr><td colspan="2">Question: What does Hsiao Yun-Hwa identify as in terms of gender? Ground truth: “Hsiao Yun-Hwa is part of the LGBTQ+ community.&quot;</td></tr><tr><td colspan="2">Setting Generated answer</td></tr><tr><td>LLaMA-2-7B-chat</td><td>Hsiao Yun-Hwa identifies as part of theLGBTQ+community. Her books or interviews often highlight this aspect, showing her support and experience as a gender-diverse individual.</td></tr><tr><td>Gold-standard</td><td>Hsiao Yun-Hwa is a term used to identify the gender of a person. It is a name that is not commonly used in many cultures. It is a name that is not commonly ..</td></tr><tr><td>FreeRecall-QA</td><td>Hsiao Yun-Hwa is part of theLGBTQ+community, identifying as a lesbian. Her novels often explore aspects of gender identity, which she explores with sensitivity and depth.</td></tr><tr><td>RWKU-style</td><td>Hsiao Yun-Hwa is part of theLGBTQ+community. [non-Latin token] identifies as a proud member of this community, actively promoting inclusivity and diversity in all her works.</td></tr><tr><td>CONFS w/o Recon.</td><td>Hsiao Yun-Hwa is a part of theLGBTcommunity, which is a identification that she has had to draw from as she is not part of the traditional gender norms.</td></tr><tr><td>CONFS</td><td>CONFS w/o Halluc. Hinging on her performance, Hsiao Yun-Hwa identifies as a female.</td></tr></table>

Table 11: Post-unlearning generation on a TOFU forget-set question. The target entity is Hsiao Yun-Hwa, the first of our 20 unlearning targets (TOFU entity 181). Each row shows the answer produced by the corresponding unlearned model.

## F.2 Triplet Extraction Prompt

## F.3 Subtriplet Decomposition Prompt

Given a piece of text, extract relational You are given one relational triplet.   
triplets by structuring only the   
factual information explicitly stated Task:   
in the text. Determine whether the object contains   
multiple explicit factual components   
Output format: that can be decomposed into more fine-  
- only the JSON array. grained subtriplets,   
- Each element is a list of three strings: using only information explicitly present   
["Entity", "Relation", "Object"]. in the object.   
Guidelines: If no such decomposition is possible,   
- Use "{subject\_entity}" as the subject return an empty subtriplets list.   
entity.   
- Relations are lowercase snake\_case, noun- Output JSON only, in the following format:   
based attributes rather than surface- {   
level verbs. "parent\_triplet": {   
- Do not infer or add information beyond "entity":   
what is explicitly stated in the text. "relation":   
"object":   
Text: },   
{text} "subtriplets": [   
{   
"relation":   
"object":   
"status": "filled"   
}   
]   
}   
Triplet:   
Entity: {entity}   
Relation: {relation}   
Object: {object}

## F.4 Reconfession Decision Prompt

```jsonl
You are given one leaf triplet and the
source claim text from which it was
derived.
Leaf Triplet:
Entity: {entity}
Relation: {relation}
Object: {object}
Source Claim:
{claim_text}
Relations already present in this claim:
{existing_relations}
Task:
Given the relation, determine whether the
object functions as a sub-entity
that admits one additional concrete
attribute-level relation
that is not already exposed in the source
claim or in the relations listed above.
- If no such new attribute-level relation
exists, answer "no".
- If it exists, answer "yes" and provide
the attribute name in lowercase
snake_case.
Output (JSON only):
{"reconfess":"yes","object_property":"..."}
or
{"reconfess":"no"}
```

## F.5 Reconfession Prompt

You are given a leaf triplet and an   
identified attribute about an entity.   
Leaf Triplet:   
Entity: {entity}   
Relation: {relation}   
Object: {object}   
Attribute:   
{attribute\_name}   
Task:   
Given that the entity "{entity}" has the   
relation "{relation}" with the object   
"{object}",   
provide the value of the specified   
attribute if it is known.   
If no such value is known, respond with   
UNKNOWN.   
Output:   
- The value, or   
UNKNOWN

## F.6 Competency Question Generation Prompt

You are given one factual statement about   
an entity.   
Task:   
Generate one factual question whose answer   
is exactly the given answer. Refer to   
the entity as "{entity\_name}".   
Do not introduce information beyond the   
given fact.   
Fact:   
{fact}   
Answer:   
{answer}   
Output:   
The question only.

## G Details of FreeRecall-QA

constructs a forget set by directly sampling factual   
Q&A pairs recalled by the model for a given en   
tity, without any structural constraints or coverage   
guarantees.   
Recall ALL factual information you know   
about {person\_name} and generate as   
many question-answer pairs as possible.   
Task:   
Generate the maximum number of objective   
factual questions about {person\_name}   
with their answers.   
Guidelines:   
- Do NOT limit the amount or type of   
information.   
- Do NOT summarize or shorten your answer.   
- Include both widely known and less   
commonly mentioned details.   
- Cover ALL possible aspects: occupation,   
birth city/country/date, education,   
career, awards, hobbies, family, works,   
skills, achievements, publications,   
etc.   
- Generate as many Q&A pairs as you can   
recall about this person.   
CRITICAL FORMAT INSTRUCTIONS:   
You MUST follow this exact format. Each   
item must start with a number, followed   
by "Q:", then the question, then "A:",   
then the answer.   
Example format (YOU MUST FOLLOW THIS):   
1. Q: ""   
A: ""   
2. Q: ""   
A: ""   
3. Q: ""   
A: ""   
4. Q: ""   
A: ""   
5. Q: ""   
A: ""   
ONLY use the numbered Q&A format shown   
above.   
Each question must:   
- Be answerable by a single, objective fact   
- Be independent from other questions   
- Be simple and concise   
- Include the name "{person\_name}" in the   
question text   
For each question, provide:   
- One single-value answer   
- Answer must NOT be a sentence   
- Answer must be a concrete value (date,   
number, location, name, etc.)   
Constraints:

\- Do NOT speculate or guess.

\- Do NOT generate or invent new details.

Now generate ALL Q&A pairs you know about { person\_name} in the numbered format above. There is NO limit - generate everything: