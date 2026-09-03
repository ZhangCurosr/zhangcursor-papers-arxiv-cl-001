# PragAlign: Feedback-Guided Pragmatic Alignment for Controlled Synthetic Dialogue Generation

Smitha Muthya Sudheendra, Jaideep Srivastava

University of Minnesota, Twin Cities muthy009@umn.edu, srivasta@umn.edu

## Abstract

Synthetic dialogue generation can support research in privacy-restricted service settings, but generated conversations must preserve communicative intent, affective meaning, and natural dialogue flow. We introduce PragAlign, a feedback-guided framework for controlled synthetic dialogue generation conditioned on service context, target intent, and target emotion, with auxiliary trait-style controls. PragAlign uses a generate– evaluate–revise loop in which an LLM-based evaluator scores intent alignment, emotion alignment, coherence, fluency, and aggregate quality, then provides criterion-specific feedback for up to three refinement rounds. On 800 matched dialogue specifications, PragAlign achieves 99.50% evaluator-defined acceptance, compared with 72.25% for one-shot generation and 95.88% for repeated generation without structured feedback. This indicates that repeated attempts account for much of the gain over one-shot generation, while structured feedback primarily improves last-mile multiconstraint satisfaction rather than broad average quality. Refinement gains are concentrated in emotion alignment, which is also the dominant failure mode in ablations. A separate human evaluation of 1,200 generated dialogues shows that intent expression and dialogue flow are highly recognizable to annotators, while emotion appropriateness is less stable and more subjective. These results support PragAlign as a quality-control framework for improving evaluator-defined communicative constraint satisfaction, while showing that affective realization and independent human-perceived quality remain open challenges.

## Introduction

Synthetic dialogue data can support the development and evaluation of conversational agents, especially in service domains where real customer–agent conversations are private, costly, or difficult to share. However, useful synthetic dialogues must do more than sound fluent. They should express the intended service goal, maintain coherent interaction flow, and realize the target affective stance in context. We refer to this objective as pragmatic constraint satisfaction: the generated dialogue should satisfy specified communicative constraints, including intent, emotion, coherence, and fluency.

We introduce PRAGALIGN, a feedback-guided generate– evaluate–revise framework for controlled synthetic servicedialogue generation. Each dialogue specification contains a service-domain scenario, target intent set, target emotion set, and an auxiliary trait-style control vector. PragAlign first generates a candidate multi-turn dialogue from this specification. An LLM-based evaluator then scores the dialogue for intent alignment, emotion alignment, coherence, fluency, and aggregate quality. If any criterion falls below threshold, the evaluator returns criterion-specific feedback, and the generator revises the dialogue for up to three refinement rounds.

A central challenge is that the evaluator is also part of the refinement loop. We therefore treat automatic acceptance as evaluator-defined constraint satisfaction, not as an independent measure of human-perceived dialogue quality. This distinction is important because a generator may learn to satisfy the evaluator’s rubric without necessarily improving all aspects of human judgment. We therefore organize the evaluation around three questions: whether structured evaluator feedback improves controlled dialogue generation beyond repeated sampling, which communicative constraints benefit most from refinement, and whether evaluator-defined gains correspond to human judgments of intent expression, dialogue flow, and emotion appropriateness. The central empirical question is not whether repeated LLM generation improves outputs, but whether criterion-specific feedback repairs the remaining failures that repeated sampling alone does not reliably fix.

On 800 matched dialogue specifications, PragAlign achieves 99.50% automatic acceptance, compared with 72.25% for one-shot generation and 95.88% for repeated generation without structured feedback. Thus, repeated attempts account for much of the gain over one-shot generation, while structured feedback provides an additional improvement in complete threshold satisfaction. Refinement gains are concentrated in emotion alignment, which is also the dominant failure mode in ablations. A separate human evaluation of 1,200 generated dialogues shows that intent expression and dialogue flow are highly recognizable, while emotion appropriateness is less stable and more subjective.

This paper makes four contributions:

• We introduce PRAGALIGN, a closed-loop framework for controlled synthetic service-dialogue generation under explicit intent, emotion, coherence, and fluency constraints.

• We evaluate PragAlign on 800 matched dialogue specifications across one-shot, no-feedback, no-emotion, nopersonality, and full refinement conditions.

• We show that structured evaluator feedback improves complete constraint satisfaction beyond repeated generation alone, with the strongest refinement effects occurring in emotion alignment.

• We provide complementary analyses, including ablations and a separate human evaluation, while explicitly treating automatic acceptance as evaluator-defined rather than independent human-quality evidence.

## Related Work

Iterative refinement and model-based evaluation. PragAlign builds on recent work showing that language-model outputs can be improved through feedback, reflection, or deliberative search. Self-Refine and Reflexion use iterative feedback or verbal reflection to improve outputs across multiple attempts, while Tree-of-Thought prompting explores alternative reasoning paths during inference (Madaan et al. 2023; Shinn et al. 2023; Yao et al. 2023). In parallel, modelbased and LLM-based evaluation methods such as G-Eval, UniEval, and LLM-as-judge frameworks assess generated text quality, task success, or model preferences using structured evaluation criteria (Liu et al. 2023; Zhong et al. 2022; Zheng et al. 2023). PragAlign combines these directions by using an LLM evaluator not only to score generated service dialogues, but also to provide criterion-specific feedback for revision.

Attribute-conditioned dialogue generation. The framework is also related to emotion-aware, persona-grounded, user-centric, and personality-conditioned generation. Prior work has explored emotional response generation, empathetic response modeling, persona and knowledge grounding, personality-trait expression in LLM outputs, personality-based synthetic dialogue generation, and controllable multi-attribute dialogue generation (Zhou et al. 2018; Lin et al. 2019; Majumder et al. 2020; Jang et al. 2022; Lim et al. 2023; Zerhoudi and Granitzer 2024; Jiang et al. 2023; Han et al. 2024; Hu et al. 2022). These approaches show that generated responses can be conditioned on affective, persona, user, personality, or stylistic attributes. PragAlign differs by combining service context, target intents, target emotions, and personality-trait inputs within a closedloop acceptance procedure for synthetic service-dialogue generation.

Positioning of PragAlign. Rather than optimizing for general text quality alone, PragAlign formulates synthetic service-dialogue generation as a communicative-goal alignment problem. Each generated dialogue must satisfy specified constraints for intent alignment, emotion alignment, coherence, fluency, and aggregate quality. The resulting acceptance decision is therefore best interpreted as evaluatordefined pragmatic constraint satisfaction, not as an independent measure of human-perceived dialogue quality.

![](images/8f82b5adfe29b4d61a3623a3a9633af7080cf30078f25e7a5e1fe926683b5dc3.jpg)  
Figure 1: Overview of the PragAlign closed-loop generation framework.

## Model Framework

PragAlign is a goal-driven closed-loop framework for controlled synthetic dialogue generation in customer-facing service interactions. The framework uses an LLM generator to produce candidate multi-turn dialogues and an LLM-based evaluator to assess whether each dialogue satisfies predefined communicative constraints. When a candidate fails one or more criteria, the evaluator returns structured feedback that is incorporated into the next generation attempt. Figure 1 summarizes the overall generate–evaluate–revise loop.

## Goal Representation

Each dialogue is generated from a structured goal representation:

$$
\Phi _ { i } = ( D , p _ { i } , I _ { i } ^ { * } , E _ { i } ^ { * } , \Omega ) ,\tag{1}
$$

where D denotes the customer-facing service domain context, $p _ { i }$ is an auxiliary trait-style vector used for conditioning, $I _ { i } ^ { * }$ is the target intent set, $E _ { i } ^ { * }$ is the target emotion set, and $\dot { \Omega }$ contains generation hyperparameters. These inputs define the controlled generation space for each candidate dialogue.

The domain specification D grounds generation in a service setting, including the interaction context, user goal, and relevant support details. The target intent set $I _ { i } ^ { * }$ specifies the functional goals that should be expressed in the dialogue, such as billing clarification, refund request, account access support, complaint handling, or follow-up status request. To construct fine-grained service goals, we use analogical prompting to derive sub-intents from representative examples (Yu, He, and Ying 2023). The target emotion set $E _ { i } ^ { * }$ specifies affective states based on Plutchik’s emotion model (Plutchik and KELLERMAN 1980); these emotions may be expressed directly through emotion words or indirectly through tone, urgency, concern, frustration, relief, or reassurance.

In our implementation, $p _ { i }$ is instantiated using Big Five trait values (De Raad 2000), but these traits are used only as generation-side style controls and are not treated as evaluated output attributes.

Figure 2 summarizes the structured inputs provided to the dialogue generator.

![](images/fd277c3fab8944828330956aa25e492515a1b85874eb4a6a7b1d15c79270d35a.jpg)  
Figure 2: Structured inputs to the dialogue generator.

## Generate–Evaluate–Revise Loop

Given a goal representation Φ<sub>i</sub>, the generator produces an initial multi-turn customer-facing service dialogue. Unless otherwise stated, PragAlign uses GPT-4.1 as the dialogue generator. The generation prompt includes the servicedomain context, target intents, target emotions, personality traits, and generation constraints such as dialogue length and output format.

After each generation attempt, the evaluator scores the candidate dialogue for intent alignment, emotion alignment, coherence, and fluency. If the dialogue satisfies all acceptance criteria, it is retained as the final output. If any criterion falls below threshold, the evaluator returns criterionspecific feedback identifying the failed dimensions and describing what should be revised. The generator then receives the original goal representation together with this feedback and produces a revised dialogue.

The original communicative specification remains fixed throughout refinement. Thus, revision does not change the target domain, intent set, emotion set, or personality vector; instead, it aims to better realize those constraints in the dialogue. For example, if the evaluator identifies a missing intent or weak emotional realization, the next prompt asks the generator to strengthen that aspect while preserving coherence, fluency, and consistency with the service scenario.

## LLM-Based Evaluator

We implement the evaluator using GPT-4o-mini with structured evaluation prompts. Inspired by criterion-wise decomposition in Tree-of-Thought-style prompting (Yao et al. 2023), the evaluator assesses each dialogue along separate dimensions for intent alignment, emotion alignment, coherence, and fluency. Each dimension receives a scalar score in [0, 1], and the component scores are combined into an aggregate quality score $Q _ { \psi } ( d _ { i } )$

In addition to scalar scores, the evaluator produces structured feedback describing weaknesses in the generated dialogue, such as missing intents, weak emotion realization, incoherent turns, or unnatural phrasing. This feedback is used only when a dialogue fails the acceptance criteria; accepted dialogues exit the refinement loop without further revision.

Because the evaluator is also part of the refinement process, automatic acceptance should be interpreted as evaluator-defined constraint satisfaction rather than as independent evidence of human-perceived dialogue quality. The human evaluation is therefore used as a separate datasetlevel check of whether generated dialogues contain recognizable cues for intent expression, dialogue flow, and emotion appropriateness.

## Experimental Setup

## Dataset Construction

We construct 800 matched dialogue specifications for the main automatic evaluation. Each specification contains a service-domain scenario, a target intent set, a target emotion set, and an auxiliary trait-style control vector. The same specifications are used across all automatic experimental conditions to enable paired comparison.

The specifications cover 10 customer-facing service domains: travel booking, online retail delivery, online subscription services, banking support, health appointment services, food delivery support, hotel and hospitality services, utility services, software help desk support, and telecom services. Domain coverage is approximately balanced, with between 72 and 89 specifications per domain.

The intent set contains 15 service intent categories: account access support, billing clarification, service upgrade request, complaint handling, follow-up status request, request invoice or documentation, delivery status inquiry, payment method update, policy clarification, incorrect charge dispute, refund request, confirmation request, replacement or exchange request, contact information update, and escalation request. Each dialogue specification contains either two or three target intents; 150 specifications contain two intents and 650 contain three intents. The emotion set contains 30 affective labels: admiration, amazement, anger, anticipation, apprehension, awe, boredom, contempt, disapproval, disgust, distraction, ecstasy, fear, grief, interest, joy, loathing, love, optimism, pensiveness, rage, remorse, sadness, serenity, submission, surprise, terror, trust, vigilance, and aggressiveness. Each specification contains one to three target emotions; 11 specifications contain one emotion, 118 contain two emotions, and 671 contain three emotions. We also include an auxiliary Big Five trait-style vector for generation-side variation. Each trait takes one of three values, corresponding to low, medium, or high intensity. These values are used only as conditioning inputs and are not evaluated as target output attributes. Generated dialogues were constrained to six to eight turns. Unless otherwise stated, the generator temperature was 0.7, with a maximum output length of 1024 tokens.

## Implementation Details

Unless otherwise stated, experiments use gpt-4.1 as the dialogue generator and gpt-4o-mini as the automatic evaluator. The evaluator is used as an internal qualitycontrol mechanism: it scores generated dialogues, identifies failed criteria, and provides structured feedback for revision.

The full PragAlign condition allows up to three refinement rounds after the initial generation attempt.

The no-feedback baseline uses the same maximum number of attempts as full PragAlign, but does not pass criterionspecific evaluator feedback into the next generation prompt. In this condition, each failed attempt is followed by a fresh generation from the original dialogue specification. The generator does not receive the previous dialogue, failed criteria, missing-intent list, missing-emotion list, or evaluator rationale. The maximum number of attempts, generator model, evaluator model, temperature, token limit, and acceptance thresholds are matched to the full PragAlign condition. This baseline isolates the effect of structured feedback from the effect of repeated sampling.

## Evaluation Protocol

Automatic evaluation measures intent alignment, emotion alignment, coherence, fluency, and aggregate quality. We compute aggregate quality as

$$
Q ( d ) = { \frac { A ( d ) + C ( d ) + F ( d ) } { 3 } } ,
$$

where $A ( d )$ is the mean of intent and emotion alignment, $C ( d )$ is coherence, and $F ( d )$ is fluency. Because $\bar { Q ( d ) }$ can obscure component-level failures, acceptance is determined by both aggregate quality and individual component thresholds. A dialogue is accepted only if it satisfies both the aggregate-quality threshold and all component-level thresholds:

$$
\begin{array} { c } { { Q ( d ) ~ \geq ~ 0 . 8 0 , S _ { I } ( d ) ~ \geq ~ 0 . 8 0 , S _ { E } ( d ) ~ \geq ~ 0 . 7 0 , C ( d ) ~ \geq ~ } } \\ { { 0 . 8 0 , F ( d ) \geq 0 . 8 0 . } } \end{array}
$$

We report automatic acceptance as evaluator-defined constraint satisfaction rather than as an independent measure of human-perceived dialogue quality. In addition to the 800- goal automatic evaluation, we conduct a separate human evaluation on 1,200 generated dialogues sampled from the final dataset. Because this human-evaluated set is separate from the matched automatic-evaluation set, we treat it as dataset-level validation rather than as a paired human comparison of experimental conditions.

## Results

We evaluate PragAlign on 800 matched dialogue specifications under five experimental conditions: the full PragAlign framework, one-shot generation, repeated generation without structured feedback, refinement without explicit emotion conditioning, and refinement without personality conditioning. We focus on last-mile constraint satisfaction: cases where generated dialogues are fluent and mostly coherent but still fail one or more required communicative constraints. Each condition uses the same dialogue specifications, enabling paired comparisons across methods. The primary outcomes are automatic acceptance, aggregate quality, component-level alignment scores, refinement behavior, and failure reasons.

We use the term automatic acceptance to emphasize that this metric reflects evaluator-defined constraint satisfaction rather than independent human-perceived dialogue quality.

The evaluator is part of the refinement loop, so the reported acceptance rate should be interpreted as an internal qualitycontrol measure. A dialogue is accepted only when it satisfies both the aggregate-quality threshold and all componentlevel thresholds. Thus, a dialogue can receive a high aggregate score while still being rejected if intent alignment, emotion alignment, coherence, or fluency remains below threshold.

## Overall Automatic Performance

Table 1 reports final automatic performance across the five conditions. Full PragAlign accepted 796 of 800 dialogue specifications, corresponding to an evaluator-defined acceptance rate of 99.50%. One-shot generation accepted 578 of 800 dialogues, corresponding to 72.25%. This 27.25 percentage-point difference shows that allowing refinement substantially increases complete threshold satisfaction relative to a single generation attempt.

The more informative comparison, however, is between full PragAlign and the no-feedback repeated-generation baseline, since both conditions allow multiple attempts. The no-feedback baseline accepted 767 dialogues, or 95.88%, showing that repeated sampling alone accounts for much of the gain over one-shot generation. Full PragAlign accepted 29 additional dialogues, improving acceptance by 3.62 percentage points over no-feedback generation. We therefore interpret structured evaluator feedback as improving lastmile constraint satisfaction: it does not dramatically change average dialogue quality, but it helps repair the remaining component-level failures that prevent otherwise strong dialogues from satisfying all acceptance criteria.

This interpretation is supported by the continuous quality scores. Full PragAlign achieved a mean aggregate quality score of 0.953, compared with 0.950 for no-feedback generation and 0.935 for one-shot generation. Thus, the main effect of feedback is not a large shift in average fluency, coherence, or aggregate quality. Instead, feedback increases the probability that a dialogue satisfies every required communicative constraint simultaneously.

Paired analyses further support this distinction. Compared with one-shot generation, full PragAlign produced a mean paired quality improvement of 0.0177, with a bootstrap 95% confidence interval of [0.0145, 0.0209]. The difference was significant under a Wilcoxon signed-rank test $( p \ < \ . 0 0 1 )$ , with a paired standardized effect size of $d _ { z } ~ = ~ 0 . 3 8$ . The paired acceptance comparison showed a larger practical effect: full PragAlign accepted 219 specifications that failed under one-shot generation, whereas one-shot generation accepted only one specification that failed under full PragAlign. An exact McNemar test confirmed that the paired acceptance outcomes differed significantly $( p < . 0 0 1 )$ .

Taken together, these results show that PragAlign’s strongest contribution is complete multi-constraint satisfaction. Repeated attempts produce a strong baseline, but structured feedback provides targeted repairs for the remaining cases that fail intent, emotion, coherence, fluency, or aggregate-quality thresholds.

<table><tr><td>Condition</td><td>Accepted</td><td>Acc. (%)</td><td>Quality</td><td>Intent</td><td>Emotion</td><td>Coherence</td><td>Fluency</td><td>Mean Ref.</td></tr><tr><td>Full PragAlign</td><td>796/800</td><td>99.50</td><td>0.953</td><td>1.000</td><td>0.998</td><td>0.908</td><td>0.951</td><td>0.324</td></tr><tr><td>No personality</td><td>798/800</td><td>99.75</td><td>0.953</td><td>1.000</td><td>0.999</td><td>0.909</td><td>0.951</td><td>0.288</td></tr><tr><td>No feedback</td><td>767/800</td><td>95.88</td><td>0.950</td><td>0.999</td><td>0.985</td><td>0.907</td><td>0.952</td><td>0.456</td></tr><tr><td>No emotion</td><td>638/800</td><td>79.75</td><td>0.941</td><td>0.999</td><td>0.915</td><td>0.912</td><td>0.955</td><td>1.221</td></tr><tr><td>One-shot</td><td>578/800</td><td>72.25</td><td>0.935</td><td>0.995</td><td>0.900</td><td>0.907</td><td>0.951</td><td>0.000</td></tr></table>

Table 1: Automatic performance across five conditions on 800 matched dialogue specifications. Acceptance denotes satisfaction of the configured evaluator-defined thresholds, not independent human judgment.

<table><tr><td>Stage</td><td>Newly Accepted</td><td>Cumulative</td><td>Cum. (%)</td></tr><tr><td>Initial generation</td><td>590</td><td>590</td><td>73.75</td></tr><tr><td>First refinement</td><td>172</td><td>762</td><td>95.25</td></tr><tr><td>Second refinement</td><td>27</td><td>789</td><td>98.63</td></tr><tr><td>Third refinement</td><td>7</td><td>796</td><td>99.50</td></tr><tr><td>Unsuccessful</td><td>4</td><td></td><td>0.50</td></tr></table>

Table 2: Cumulative automatic acceptance across refinement stages for full PragAlign.

## Refinement Dynamics

Table 2 summarizes when dialogues were accepted under the full PragAlign condition. Of the 800 dialogues, 590 satisfied all criteria on the initial attempt. The remaining 210 dialogues entered the refinement loop. Among these, 172 were accepted after the first refinement, 27 after the second refinement, and seven after the third refinement. Four dialogues remained unsuccessful after the maximum refinement budget.

The largest improvement occurred during the first refinement step. Cumulative acceptance increased from 73.75% after initial generation to 95.25% after one refinement. The second refinement added 3.38 percentage points, while the third refinement added 0.87 percentage points. These results show diminishing returns across refinement rounds: most correctable failures were resolved after a single feedbackguided revision.

Raw round-level means should not be interpreted as a standard longitudinal trajectory because dialogues exit the loop once they satisfy the acceptance criteria. Later rounds therefore contain progressively harder unresolved cases. For this reason, we evaluate refinement effectiveness using paired initial-to-final comparisons for the same 210 dialogues that underwent refinement.

## Initial-to-Final Effects of Refinement

Table 3 reports paired initial-to-final changes among the 210 full-method dialogues that underwent at least one refinement. Aggregate quality increased by 0.059 on average. Emotion alignment showed the largest component-level improvement, increasing by 0.340 on average. Intent alignment improved by 0.025, while coherence changed only slightly and fluency decreased marginally.

Among the 210 refined dialogues, 204, or 97.14%, achieved a higher final aggregate score than their initial score. Six dialogues worsened, and none remained exactly unchanged. The paired initial-to-final quality improvement had a large within-dialogue effect size $( d _ { z } ^ { \phantom { } } = \bar { 1 } . 5 4 )$ . This indicates that refinement reliably improved initially unsuccessful cases under the configured evaluator.

<table><tr><td>Metric</td><td>Mean Change</td></tr><tr><td>Aggregate quality</td><td>+0.059</td></tr><tr><td>Intent alignment</td><td>+0.025</td></tr><tr><td>Emotion alignment</td><td>+0.340</td></tr><tr><td>Coherence</td><td>+0.003</td></tr><tr><td>Fluency</td><td>-0.007</td></tr></table>

Table 3: Mean initial-to-final score changes for refined full PragAlign dialogues.

The component-level pattern shows that refinement primarily repaired emotion alignment. Emotion scores improved in 195 refined dialogues and remained unchanged in 15; no refined dialogue exhibited a reduction in emotion alignment. Intent alignment improved in only 15 dialogues and remained unchanged in the remaining 195, suggesting that intent errors were less frequent and more localized. Coherence showed no systematic improvement, and the small decrease in fluency suggests a minor trade-off: revisions targeting missing intent or emotion cues can occasionally introduce slightly less natural phrasing. The magnitude of this fluency decrease was small relative to the normalized score range.

## Ablation Analysis

Table 4 compares the full framework with three refinementbased ablations. The results separate three factors: explicit emotion conditioning, structured evaluator feedback, and personality conditioning.

Removing explicit emotion conditioning produced the largest degradation. The no-emotion condition accepted 638 of 800 dialogues, corresponding to 79.75%, compared with 99.50% for the full framework. It also required substantially more refinement, with a mean of 1.221 refinement rounds compared with 0.324 for full PragAlign. This indicates that explicit emotion conditioning improves both final threshold satisfaction and refinement efficiency. Removing structured feedback produced a smaller but important degradation. The no-feedback condition accepted 767 dialogues, or 95.88%, compared with 796 dialogues, or 99.50%, for full PragAlign. The difference in aggregate quality was small, indicating that repeated generation often produces fluent and coherent dialogues. However, the acceptance gap shows that structured feedback helps with last-mile constraint satisfaction: it repairs specific failed criteria that prevent a dialogue from satisfying all component-level thresholds. Thus, feedback contributes less to broad average-quality improvement than to complete evaluator-defined acceptance.

<table><tr><td>Condition</td><td>Accepted</td><td>Acc. (%)</td><td>Quality</td><td>Emotion</td><td>Mean Ref.</td><td>Failed After Max</td></tr><tr><td>Full PragAlign</td><td>796/800</td><td>99.50</td><td>0.953</td><td>0.998</td><td>0.324</td><td>4</td></tr><tr><td>No emotion</td><td>638/800</td><td>79.75</td><td>0.941</td><td>0.915</td><td>1.221</td><td>162</td></tr><tr><td>No feedback</td><td>767/800</td><td>95.88</td><td>0.950</td><td>0.985</td><td>0.456</td><td>33</td></tr><tr><td>No personality</td><td>798/800</td><td>99.75</td><td>0.953</td><td>0.999</td><td>0.288</td><td>2</td></tr></table>

Table 4: Ablation comparison on the 800-goal automatic evaluation set.

The no-personality condition accepted 798 dialogues, or 99.75%, slightly higher than the full framework. Because personality traits are not directly evaluated by either the automatic evaluator or the human protocol, this result should not be interpreted as evidence about personality realization. It shows only that the auxiliary trait-style control does not improve the current intent, emotion, coherence, or fluency metrics.

## Paired Ablation Statistics

Table 5 reports paired statistical comparisons between the full method and each ablation. Compared with the noemotion condition, full PragAlign improved aggregate quality by 0.0113 on average, with a bootstrap 95% confidence interval of [0.0081, 0.0145]. The Wilcoxon signed-rank test was significant $( p < . 0 0 1 )$ , and the paired acceptance difference was also significant under McNemar’s test $( p < . 0 0 1 )$ ).

Compared with the no-feedback baseline, the mean aggregate-quality difference was small: 0.0024, with a bootstrap 95% confidence interval of [-0.0002, 0.0051]. The Wilcoxon signed-rank test was not significant $( p = . 1 0 3 )$ indicating no reliable difference in continuous aggregate quality. However, the paired acceptance difference was significant under McNemar’s test $( p \textless . 0 0 1 )$ : full PragAlign accepted 796 dialogues compared with 767 for no-feedback generation. This pattern reinforces the last-mile interpretation. Structured feedback does not substantially raise average quality, but it improves the likelihood that a dialogue satisfies all required thresholds simultaneously.

Compared with the no-personality ablation, full PragAlign showed no measurable advantage. The mean aggregate-quality difference was -0.0003, with a bootstrap 95% confidence interval of [-0.0028, 0.0023]. The Wilcoxon signed-rank test was not significant $( p ~ = ~ . 9 9 2 )$ , and the paired acceptance difference was also not significant (p = .688).

## Failure Analysis

Table 6 reports failure reasons by condition. Failure counts are not mutually exclusive because a dialogue can fail more than one criterion. The dominant failure source was emotion misalignment. Under one-shot generation, 214 dialogues failed the emotion-alignment threshold. Under the no-emotion ablation, 162 dialogues failed the same threshold. In contrast, only four full PragAlign dialogues failed due to emotion alignment after refinement.

This pattern is consistent with the paired score analysis. Emotion alignment exhibited the largest initial-to-final improvement and accounted for most of the cases repaired during refinement. At the same time, the persistence of four emotion-related failures after the maximum refinement budget indicates that a small subset of target emotional configurations remained difficult to satisfy even under evaluatorguided revision.

## Human Evaluation of Generated Dialogues

We conducted a separate human evaluation on 1,200 generated dialogues sampled from the final dataset. Because this sample is separate from the 800 matched automaticevaluation specifications, we treat it as dataset-level validation rather than as a paired human comparison of experimental conditions.

Each dialogue was independently evaluated by two English-speaking annotators at the dialogue level along three binary dimensions: intent expression, dialogue flow, and emotion appropriateness. Annotators were shown the target intent and target emotion, but not the automatic evaluator scores. Personality traits were not evaluated. Table 7 reports positive rate, raw agreement, and Gwet’s AC1 (Gwet 2008).

Human annotators judged intent expression and dialogue flow as highly recognizable, while emotion appropriateness was lower and less stable. This suggests that affective realization is more subjective and context-dependent than intent expression or dialogue flow.

## Automatic–Human Discrepancy for Emotion

The largest automatic–human discrepancy appears in emotion evaluation. Under the automatic evaluator, full PragAlign achieved a mean emotion-alignment score of 0.998, and only four of 800 dialogues failed the automatic emotion threshold. In the separate human evaluation, however, emotion appropriateness received a lower positive rate of 79.1%, with 75.8% raw agreement and AC1 of 0.639. This gap suggests that automatic emotion scores should be interpreted cautiously.

One likely reason is that the automatic evaluator may reward explicit affective markers or easily identifiable emotion cues, whereas human annotators may require the target emotion to be contextually appropriate across the full interaction. Emotion cues in service dialogues may also be mild, indirect, or distributed across turns, making them less stable than intent expression or dialogue flow. Thus, the automatic results show that PragAlign satisfies the configured emotion-alignment criterion, while the human results show that reliable human-perceived affective realization remains harder to establish.

<table><tr><td>Comparison</td><td>Mean Diff.</td><td>95% CI</td><td>Wilcoxon  $p$ </td><td> $\overline { { d _ { z } } }$ </td><td>Full Acc.</td><td>Ablation Acc.</td><td>McNemar p</td></tr><tr><td>Full vs. no emotion</td><td>+0.0113</td><td>[0.0081, 0.0145]</td><td>&lt; .001</td><td>0.24</td><td>796</td><td>638</td><td> $< . 0 0 1$ </td></tr><tr><td>Full vs. no feedback</td><td>+0.0024</td><td>[-0.0002, 0.0051]</td><td>.103</td><td>0.06</td><td>796</td><td>767</td><td>&lt; .001</td></tr><tr><td>Full vs. no personality</td><td>-0.0003</td><td>[-0.0028, 0.0023]</td><td>.992</td><td>-0.01</td><td>796</td><td>798</td><td>.688</td></tr></table>

Table 5: Paired comparisons between full PragAlign and ablation conditions. Positive quality differences favor the full framework.

<table><tr><td>Condition</td><td>None</td><td> $\overline { { Q } }$ </td><td> $\overline { { S _ { I } } }$ </td><td> $S _ { E }$ </td><td> $\overline { { C } }$ </td></tr><tr><td>One-shot</td><td>578</td><td>5</td><td>12</td><td>214</td><td>0</td></tr><tr><td>No feedback</td><td>767</td><td>0</td><td>3</td><td>32</td><td>0</td></tr><tr><td>Full PragAlign</td><td>796</td><td>0</td><td>0</td><td>4</td><td>0</td></tr><tr><td>No emotion</td><td>638</td><td>5</td><td>2</td><td>162</td><td>0</td></tr><tr><td>No personality</td><td>798</td><td>0</td><td>1</td><td>2</td><td>0</td></tr></table>

Table 6: Failure reasons across conditions. $Q$ denotes aggregate quality below 0.80, $S _ { I }$ intent alignment below 0.80, S<sub>E</sub> emotion alignment below 0.70, and C coherence below 0.80. Fluency failures were not observed in this table. Counts are not mutually exclusive.
<table><tr><td>Dimension</td><td>Positive Rate</td><td>Agreement</td><td>AC1</td></tr><tr><td>Intent</td><td>99.7%</td><td>99.4%</td><td>0.994</td></tr><tr><td>Dialogue Flow</td><td>97.7%</td><td>95.3%</td><td>0.951</td></tr><tr><td>Emotion</td><td>79.1%</td><td>75.8%</td><td>0.639</td></tr></table>

Table 7: Human evaluation results on 1,200 dual-annotated generated dialogues.

## Limitations

This study has several limitations. First, the main automatic results depend on an LLM-based evaluator that is also used to guide refinement. The reported 99.50% automatic acceptance rate should therefore be interpreted as evaluatordefined constraint satisfaction rather than as an independent measure of human-perceived dialogue quality. Because the default generator and evaluator are both from OpenAI model families, future work should reevaluate frozen outputs with independent evaluators from different model families.

Second, the human evaluation is not a paired comparison of experimental conditions. The 1,200 human-evaluated dialogues were sampled separately from the 800 matched automatic-evaluation specifications, so the human study provides dataset-level validation but does not show that humans prefer PragAlign outputs over one-shot or no-feedback outputs for the same goals.

Third, automatic and human emotion results reveal a calibration gap. Full PragAlign reaches near-ceiling automatic emotion alignment, but human annotators rate emotion appropriateness positively in 79.1% of cases, with lower agreement than for intent expression or dialogue flow. This suggests that the automatic evaluator may reward explicit affective markers without fully capturing contextual appropriateness.

Finally, personality conditioning is not directly validated. PragAlign uses Big Five traits as generation controls, but neither the automatic evaluator nor the human annotation protocol measures whether those traits are recognizable in the generated dialogues. We therefore treat personality as an input conditioning variable rather than a validated output attribute. Additional limitations include ceiling effects in intent, coherence, and fluency metrics, and dependence on hand-selected acceptance thresholds.

## Conclusion

We presented PRAGALIGN, a feedback-guided framework for generating synthetic service dialogues under explicit communicative constraints. Across 800 matched specifications, PragAlign substantially improves evaluator-defined acceptance over one-shot generation and further improves complete constraint satisfaction beyond repeated generation without structured feedback. The results show that closedloop refinement is especially useful for repairing emotionalignment failures, which remain the hardest aspect of controlled dialogue generation in both automatic and human evaluation. At the same time, the study highlights the limits of evaluator-guided generation: automatic acceptance is not equivalent to independent human judgment, personality conditioning remains unvalidated, and human annotators find emotion appropriateness less stable than intent expression or dialogue flow. The key lesson is that feedback-guided refinement can enforce explicit communicative constraints, but reliable affective control requires calibration against human judgment rather than evaluator acceptance alone.

## References

[De Raad 2000] De Raad, B. 2000. The big five personality factors: the psycholexical approach to personality. Hogrefe & Huber Publishers.

[Gwet 2008] Gwet, K. L. 2008. Computing inter-rater reliability and its variance in the presence of high agreement. British Journal of Mathematical and Statistical Psychology 61(1):29–48.

[Han et al. 2024] Han, J.-E.; Koh, J.-S.; Seo, H.-T.; Chang, D.-S.; and Sohn, K.-A. 2024. Psydial: Personality-based synthetic dialogue generation using large language models. arXiv preprint arXiv:2404.00930.

[Hu et al. 2022] Hu, Z.; Cao, Z.; Chan, H. P.; Liu, J.; Xiao, X.; Su, J.; and Wu, H. 2022. Controllable dialogue generation with disentangled multi-grained style specification and attribute consistency reward. IEEE/ACM Transactions on Audio, Speech, and Language Processing 31:188–199.

[Jang et al. 2022] Jang, Y.; Lim, J.; Hur, Y.; Oh, D.; Son, S.; Lee, Y.; Shin, D.; Kim, S.; and Lim, H. 2022. Call for customized conversation: Customized conversation grounding persona and knowledge. In Proceedings of the AAAI

conference on artificial intelligence, volume 36(10), 10803– 10812.

[Jiang et al. 2023] Jiang, H.; Zhang, X.; Cao, X.; Breazeal, C.; Roy, D.; and Kabbara, J. 2023. Personallm: Investigating the ability of large language models to express personality traits. arXiv preprint arXiv:2305.02547.

[Lim et al. 2023] Lim, J.; Kang, M.; Hur, Y.; Jung, S.; Kim, J.; Jang, Y.; Lee, D.; Ji, H.; Shin, D.; Kim, S.; et al. 2023. You truly understand what i need: Intellectual and friendly dialogue agents grounding knowledge and persona. arXiv preprint arXiv:2301.02401.

[Lin et al. 2019] Lin, Z.; Madotto, A.; Shin, J.; Xu, P.; and Fung, P. 2019. MoEL: Mixture of empathetic listeners. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, 121–132.

[Liu et al. 2023] Liu, Y.; Iter, D.; Xu, Y.; Wang, S.; Xu, R.; and Zhu, C. 2023. G-eval: Nlg evaluation using gpt-4 with better human alignment. In Proceedings ofthe 2023 conference on empirical methods in natural language processing, 2511–2522.

[Madaan et al. 2023] Madaan, A.; Tandon, N.; Gupta, P.; Hallinan, S.; Gao, L.; Wiegreffe, S.; Alon, U.; Dziri, N.; Prabhumoye, S.; Yang, Y.; et al. 2023. Self-refine: Iterative refinement with self-feedback. Advances in neural information processing systems 36:46534–46594.

[Majumder et al. 2020] Majumder, N.; Hong, P.; Peng, S.; Lu, J.; Ghosal, D.; Gelbukh, A.; Mihalcea, R.; and Poria, S. 2020. MIME: MIMicking emotions for empathetic response generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, 8968–8979.

[Plutchik and KELLERMAN 1980] Plutchik, R., and KELLERMAN, H. 1980. Emotion, i: Theories of emotions, ny.

[Shinn et al. 2023] Shinn, N.; Cassano, F.; Gopinath, A.; Narasimhan, K.; and Yao, S. 2023. Reflexion: Language agents with verbal reinforcement learning. Advances in neural information processing systems 36:8634–8652.

[Yao et al. 2023] Yao, S.; Yu, D.; Zhao, J.; Shafran, I.; Griffiths, T.; Cao, Y.; and Narasimhan, K. 2023. Tree of thoughts: Deliberate problem solving with large language models. Advances in neural information processing systems 36:11809–11822.

[Yu, He, and Ying 2023] Yu, J.; He, R.; and Ying, R. 2023. Thought propagation: An analogical approach to complex reasoning with large language models. arXiv preprint arXiv:2310.03965.

[Zerhoudi and Granitzer 2024] Zerhoudi, S., and Granitzer, M. 2024. Personarag: Enhancing retrieval-augmented generation systems with user-centric agents. arXiv preprint arXiv:2407.09394.

[Zheng et al. 2023] Zheng, L.; Chiang, W.-L.; Sheng, Y.; Zhuang, S.; Wu, Z.; Zhuang, Y.; Lin, Z.; Li, Z.; Li, D.; Xing, E. P.; Zhang, H.; Gonzalez, J. E.; and Stoica, I. 2023. Judg-

ing LLM-as-a-judge with MT-bench and chatbot arena. In Advances in Neural Information Processing Systems, volume 36.

[Zhong et al. 2022] Zhong, M.; Liu, Y.; Yin, D.; Mao, Y.; Jiao, Y.; Liu, P.; Zhu, C.; Ji, H.; and Han, J. 2022. Towards a unified multi-dimensional evaluator for text generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, 2023–2038.

[Zhou et al. 2018] Zhou, H.; Huang, M.; Zhang, T.; Zhu, X.; and Liu, B. 2018. Emotional chatting machine: Emotional conversation generation with internal and external memory. In Proceedings of the AAAI Conference on Artificial Intelli gence, volume 32, 730–739.

## Appendix

## Prompting and Evaluation Setup

PragAlign uses structured prompts for three stages: initial dialogue generation, evaluator-guided revision, and automatic evaluation. The initial generation prompt conditions the model on the service context, target intents, target emotions, auxiliary trait-style controls, and dialogue-length constraints. The revision prompt preserves the original dialogue specification but adds evaluator feedback describing failed criteria, missing targets, and revision advice. The nofeedback baseline uses the same original dialogue specification and retry budget, but does not provide failed criteria, missing targets, previous dialogue text, or evaluator rationale to the generator.

The automatic evaluator uses a structured rubric with four component dimensions: intent alignment, emotion alignment, coherence, and fluency. For each dialogue, the evaluator returns scalar scores in [0, 1] and criterion-specific feedback. Intent and emotion alignment are computed from detected target coverage, while coherence and fluency are scored directly by the evaluator. Aggregate quality is computed as

$$
Q ( d ) = { \frac { A ( d ) + C ( d ) + F ( d ) } { 3 } } ,
$$

where $A ( d )$ is the mean of intent and emotion alignment, $C ( d )$ is coherence, and $F ( d )$ is fluency.

A dialogue is accepted only if it satisfies both the aggregate-quality threshold and all component-level thresholds: $Q ( d ) \ ^ { \circ } = 0 . 8 0 , S _ { I } ( d ) \geq 0 . 8 0 , S _ { E } ( \mathit { d } ) \geq 0 . 7 0 , C ( d ) \geq$ $0 . 8 0 , \dot { F } \dot { ( d ) } \geq 0 . 8 0$

This rule is designed to prevent high fluency or coherence from masking failures in target intent or emotion realization.

## Dataset and Goal Specification

Each dialogue goal is represented as

$$
\Phi _ { i } = ( D , p _ { i } , I _ { i } ^ { * } , E _ { i } ^ { * } , \Omega ) ,
$$

where D is the service-domain context, $p _ { i }$ is an auxiliary trait-style control vector, $I _ { i } ^ { * }$ is the target intent set, $E _ { i } ^ { * }$ is the target emotion set, and Ω contains generation hyperparameters.

The main automatic evaluation uses 800 matched dialogue specifications across customer-facing service domains. The same specifications are used across all experimental conditions to support paired comparison. Each specification contains a service scenario, two or three target intents, one to three target emotions, and an auxiliary Big Fivestyle control vector. The trait-style vector is used only as a generation-side control and is not evaluated as an output attribute.

Generated dialogues are constrained to six to eight turns. Unless otherwise stated, generation uses temperature 0.7, a maximum output length of 1024 tokens, GPT-4.1 as the dialogue generator, and GPT-4o-mini as the automatic evaluator. The full PragAlign condition allows up to three refinement rounds after the initial generation attempt.

## Evaluator Prompt and Output Schema

The automatic evaluator receives the generated dialogue together with the original dialogue specification and evaluates the dialogue using separate criterion-wise branches. The compact evaluator instruction is:

Generated dialogue: {dialogue}

Assess the dialogue using separate criterion-wise branches for intent alignment, emotion alignment, coherence, and fluency. Label the utterances with detected target intents and detected target emotions. Provide scalar scores in [0, 1], evidence, weaknesses, and revision advice. Return only the structured evaluation object required by the schema.

The generator returns a dialogue object containing a list of turns:

```jsonl
{
"turns": [
{"speaker": "Customer", "text": "..."},
{"speaker": "Agent", "text": "..."}
]
}
```

The evaluator returns utterance-level labels and four criterion-wise evaluation branches:

{   
"utterance\_labels": [   
{   
"turn\_index": 0,   
"speaker": "Customer",   
"detected\_intents": ["..."],   
"detected\_emotions": ["..."]   
}   
],   
"intent\_branch": {   
"score": 0.0,   
"evidence": ["..."],   
"weaknesses": ["..."],   
"revision\_advice": "..."   
},   
"emotion\_branch": {   
"score": 0.0,   
"evidence": ["..."],   
"weaknesses": ["..."],   
"revision\_advice": "..."   
},   
"coherence\_branch": {   
"score": 0.0,   
"evidence": ["..."],   
"weaknesses": ["..."],   
"revision\_advice": "..."   
},   
"fluency\_branch": {   
"score": 0.0,   
"evidence": ["..."],   
"weaknesses": ["..."],   
"revision\_advice": "..."   
},   
"overall\_feedback": "..."   
}

<table><tr><td>Condition</td><td>Default Strict Q-only</td><td>Loose</td></tr><tr><td>One-shot</td><td>72.25 69.63 99.38</td><td>72.25 95.88</td></tr><tr><td>No feedback</td><td>95.8891.63</td><td>100.00</td></tr><tr><td>Full PragAlign</td><td>99.5096.25 100.00</td><td>99.50</td></tr><tr><td>No emotion</td><td>79.75 77.63</td><td>99.38 79.75</td></tr><tr><td>No personality</td><td>99.75 95.13 100.00</td><td>99.75</td></tr></table>

Table 8: Acceptance rates under alternative threshold configurations. Default uses $Q \ge . 8 0 , S _ { I } \ge . 8 0 , S _ { E } \ge . 7 0 .$ $C \geq . 8 0 .$ and $F \geq . 8 0$ . Strict requires all criteria to be at least .85; Q-only requires only $Q \geq . 8 0 ;$ Loose requires all criteria to be at least .70.

This schema allows PragAlign to separate scalar evaluation from revision guidance. The scalar scores determine whether the dialogue satisfies the acceptance criteria, while the weaknesses and revision advice are used only when a dialogue enters the refinement loop.

## Human Evaluation Protocol

We conducted a separate human evaluation on 1,200 generated dialogues sampled from the final dataset. This sample is separate from the 800 matched automatic-evaluation specifications, so the human evaluation is treated as dataset-level validation rather than as a paired human comparison of experimental conditions.

Each dialogue was independently evaluated by two English-speaking annotators at the dialogue level. Annotators judged three binary dimensions: intent expression, dialogue flow, and emotion appropriateness. They were shown the target intent and target emotion for each dialogue, but not the automatic evaluator scores or refinement history. Personality traits were not evaluated.

We report positive rate, raw agreement, and Gwet’s AC1 in the main paper. Gwet’s AC1 is used because intentexpression and dialogue-flow labels are highly skewed toward positive judgments, a setting in which Cohen’s κ can underestimate agreement despite high observed agreement.

## Additional Automatic Results

## Threshold Sensitivity and Ceiling Effects

Because automatic acceptance depends on predefined thresholds, we evaluated whether the main conclusions were stable under alternative threshold configurations. Table 8 reports acceptance rates under the default thresholds, a stricter configuration requiring all component scores and aggregate quality to be at least 0.85, a quality-only setting requiring only $\dot { Q } \ge 0 . 8 0$ , and a looser all-component setting requiring all criteria to be at least 0.70.

The sensitivity analysis shows that full PragAlign remains above one-shot, no-feedback, and no-emotion conditions under the stricter all-criteria threshold. However, the quality-only setting produces near-ceiling acceptance across all methods. This indicates that aggregate quality alone is insufficient for evaluating controlled dialogue generation, since high fluency and coherence can obscure failures in target intent or emotion realization.

## Generator Transfer

To test whether the refinement framework can be applied beyond the primary GPT generator, we evaluated PragAlign using Llama as the dialogue generator while retaining GPT-4o-mini as the automatic evaluator. This experiment used 300 dialogue goals and the full refinement loop.

The Llama generator accepted 250 of 300 dialogues, corresponding to an automatic acceptance rate of 83.33%. Mean quality was 0.907, with intent alignment of 0.971, emotion alignment of 0.956, coherence of 0.850, and fluency of 0.909. These results suggest that the refinement loop can be applied with a non-GPT generator, although performance remains model-sensitive and below the main GPTbased condition.

## Reproducibility Artifacts

The experimental pipeline records final dialogue outputs, iteration histories, evaluator scores, acceptance decisions, failed criteria, and summary tables. The main reproducibility artifacts include final-dialogue files, iteration-history files, automatic summary metrics, failure-reason tables, scoreimprovement summaries, and human-evaluation export files. These artifacts make it possible to inspect accepted outputs, failed cases, refinement trajectories, and conditionlevel comparisons.

Table 9: Llama generator transfer evaluation using GPT-4o-mini as the automatic evaluator. Fluency was 0.909. This is a generator-transfer result, not an independent cross-evaluator result.
<table><tr><td>Generator Setting N</td><td>Accepted Acc.(%)</td><td></td><td>Quality</td><td>Intent</td><td>Emotion Coh.</td></tr><tr><td>Llama + GPT eval. 300</td><td>250</td><td>83.33</td><td>0.907</td><td>0.971</td><td>0.956 0.850</td></tr></table>