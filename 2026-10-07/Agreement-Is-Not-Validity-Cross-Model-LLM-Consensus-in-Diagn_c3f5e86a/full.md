# Agreement Is Not Validity: Cross-Model LLM Consensus in Diagnosing Student Failure Modes in K–12 Math Tutoring Dialogue

Clayton Cohn<sup>1</sup>\*, Joyce Fonteles<sup>2</sup>, Kirk Vanacore<sup>1</sup>, Gianni Mazza<sup>3</sup>, Candida Crawford<sup>3</sup>, Tom Hooper<sup>3</sup>, Gautam Biswas<sup>2</sup>, and Rene Kizilcec<sup>1</sup>

## Abstract

In K–12 mathematics tutoring, student–tutor dialogue provides rich evidence of learners’ problem-solving processes and sources of difficulty. Learning analytics research increasingly relies on large language models (LLMs) to extract such information from dialogue for a variety of downstream tasks, including knowledge tracing, behavioral modeling, and diagnosis of student reasoning errors. However, the validity of these model-generated interpretations remains insufficiently understood. In this exploratory study, we examine the validity of LLM classifications of five student failure modes in mathematics tutoring dialogue using an operational diagnostic codebook: uncertainty, misattribution, operator selection, conceptual gap, and procedural slip. Across models, human–LLM agreement was moderate (κ = .524–.597), while cross-model agreement was substantially higher (κ = .755–.781; α = .769). These findings show that cross-model agreement can create a misleading appearance of correctness, challenging the assumption that consensus among LLMs constitutes evidence of valid learner interpretation. For learning analytics, the implication is clear: scalable labeling is useful only if the inferred constructs are valid, and model consensus cannot substitute for independent evidence of that validity.

## Notes for Practice

• Agreement among multiple LLMs should not be treated as evidence that a learner diagnosis is correct: models can reach strong consensus while sharing systematic disagreements with the human reference.

• LLM-generated diagnoses of student difficulty should be independently validated against human judgment before being used to inform tutoring decisions, learner models, or downstream learning analytics.

• LLMs may over-diagnose failure modes when there is no available evidence to support diagnosis; automated tutoring systems should preserve the option to remain uncertain rather than force a classification.

• When trained experts show only modest agreement, neither human nor LLM judgments should be treated as ground truth; consequential uses should include adjudication rather than unadjudicated classification.

learning analytics, k12 math tutoring, failure modes, llm coding, inter-rater reliability, construct validity

<sup>1</sup>College ofComputing and Information Science, Cornell University, Ithaca, USA. <sup>2</sup>College of Connected Computing, Vanderbilt University, Nashville, USA. <sup>3</sup>Third Space Learning, Swindon, UK.   
\*Corresponding author: clayton.cohn@cornell.edu

## 1. Introduction and Related Work

## 1.1 Theoretical Grounding and Student Discourse

Human tutoring has long been shown to support mathematics learning among K–12 students (Borchers et al., 2026), creating opportunities for sustained, responsive interaction around students’ emerging understanding. The ICAP (Interactive–Constructive– Active–Passive) framework explains why such interaction matters, proposing that learning deepens as students move toward constructive and interactive engagement (Chi & Wylie, 2014). This perspective positions student discourse as rich evidence from which learning analytics systems can infer properties of learners’ understanding and difficulty.

Characterizing student struggle is paramount. In the United States, for example, the Education Scorecard describes mathematics achievement below levels observed a decade earlier in 70% of school districts (Dewey et al., 2026). Addressing such declines requires understanding not only whether students are struggling, but the nature of that struggle as it unfolds Student–tutor discourse can reveal distinctfailure modes: the underlying reasons a student produces an incorrect response or cannot complete a learning task. Distinguishing these modes is critical, as different sources of difficulty often call for different instructional responses. Traditionally, doing so required trained human annotators to interpret discourse manually and label instances of struggle—a time- and resource-intensive process that limits analysis at scale (Garg et al., 2024).

## 1.2 LLM Labeling in Learning Analytics

Large language models (LLMs) have consequently become attractive for transforming conversational learning traces into scalable representations of learners’ understanding and difficulty. Recent work has used LLMs to code tutoring and classroom discourse (Garg et al., 2024), generate labels supporting knowledge tracing (Scarlatos, Baker, & Lan, 2025), and conduct qualitative coding across educational contexts (X. Liu et al., 2025); while other approaches retain humans in the loop (Xavier et al., 2026). However, evidence that LLMs can replicate human annotation remains mixed; Vanacore and Kizilcec (2026) conclude that “unscaffolded LLMs are not yet sufficiently reliable to replace human coders for complex instructional constructs.”

Learning analytics researchers have therefore begun to scrutinize how such annotation should be evaluated and improved. Xu et al. (2026) distinguish task ambiguity from model-specific annotation error; Borchers, Thomas, Lin, and Koedinger (2026) show that conventional reliability estimates may inadequately characterize LLM scoring in longitudinal learning settings; and Ahtisham et al. (2026) use self- and cross-model verification to improve LLM coding of tutoring discourse. Together, these studies demonstrate the promise of LLMs for scalable annotation while underscoring the importance of validating whether their labels faithfully reflect the constructs being measured.

A recent and related development is the use of multiple models as evaluators. Fonteles, Sivakumaran, et al. (2026) compare multiple LLMs for multimodal classroom analysis, and Zhu et al. (2026) incorporate LLM-enriched Q-matrices into cognitive diagnosis. Thapa Magar et al. (2026), however, show that generative models can produce coherent representations of learning strategies that diverge from the actual patterns in authentic learner data; and other studies employ LLM-generated labels or evaluations without human validation (Misiejuk et al., 2025).

## 1.3 Detecting Student Failure Modes from Discourse

Prior discourse-based learning analytics has used student interaction data to characterize learning-relevant states and processes, but diagnosing the specific reasoning error underlying an individual incorrect response remains comparatively underexplored. Suraworachet et al. (2024) detect cognitive, metacognitive, emotional, and technical challenge moments from collaborative discourse; Booth et al. (2024) analyze accountable talk in tutoring; Z. Liu et al. (2026) connect instructional discourse with engagement and performance; Snyder et al. (2024) and Zhang et al. (2026) detect socially shared regulation processes; and Gures et al.¨ (2026) examine diagnostic reasoning and cognitive bias. These studies operate primarily at the level of broader behavioral, interactional, or challenge constructs rather than diagnosing the specific failure that resulted in an incorrect response. Authentic spoken tutoring dialogue makes this finer-grained inference difficult, as such discourse is often disfluent, interrupted incomplete, and affected by transcription and diarization errors. Distinguishing specific failure modes may therefore require inference about learner reasoning that is only indirectly expressed in the dialogue.

Together, this leaves an important question unresolved for learning analytics: whether agreement across LLM families provides evidence of valid discourse interpretation, or whether shared model biases produce consensus around the same error. This has been underexplored for fine-grained diagnosis of student difficulty from authentic tutoring dialogue, including which learner states LLMs systematically over- or under-attribute and whether those biases recur across model families.

## 1.4 Present Study and Contributions

In this exploratory study, we examine whether agreement among LLMs from three contemporary model families provides a valid basis for labeling student failure modes in authentic K–12 mathematics tutoring dialogue. We study five failure modes (see Section 2.1)—uncertainty, misattribution, operator selection, conceptual gap, and procedural slip—derived from an operational codebook used by Third Space Learning<sup>1</sup> supporting over 196,000 students across the United States and United Kingdom. We compare labels from three models (one per family) with an adjudicated human reference set, quantify human–LLM and cross-model agreement, and analyze where and how model errors occur. We find a striking divergence between consensus and validity: while LLMs agree strongly with each other (κ = .755–.781), their agreement with the human consensus is substantially lower (κ = .524–.597), with models prone to over-inferring conceptual gaps across failure modes.

These results contribute to learning analytics in three ways. First, they provide empirical evidence that cross-model consensus should not substitute for human validation when LLMs infer complex learner constructs. Second, they identify systematic failure patterns that reveal where automated interpretation of student discourse is most vulnerable, with implications for LLM generated labels in operational learning systems. Third, they foreground construct validity as a methodological requirement for scalable LLM-assisted annotation, showing that high agreement among models can reflect shared misinterpretation rather than faithful inference about learner reasoning. In learning analytics, consensus is not sufficient evidence of validity—particularly when the resulting labels may shape how real students are understood and supported.

## 2. Methods

This study examines three aspects of LLM-based failure-mode diagnosis: (1) alignment with human judgments; (2) consistency across models; and (3) the failure modes and error patterns that account for the largest divergences from human interpretation. Practitioner reactions appear in the Discussion as an informal validity check, not a formal finding

## 2.1 Tutoring Platform and Failure-Mode Taxonomy

Dialogue data are drawn from a Third Space Learning dataset of transcribed tutor–student interactions from a K–12 mathematics tutoring platform comprising 22,821 tutoring sessions (each with one or more learning objectives, or LOs) and 35,072 graded student responses (one per LO) between November 2024 and July 2025. Tutoring sessions were conducted virtually, with a tutor and student working synchronously through a shared screen on mathematics problems and discussing the student’s reasoning; the resulting dialogue was recorded via the tutoring platform, then transcribed and diarized into tutor and student turns using Deepgram. Each LO is automatically marked complete once the tutor is satisfied with the student’s understanding and advances to the next LO; a session’s final LO is manually marked complete or incomplete by the tutor at the session’s close. The corpus captures naturally occurring instructional dialogue rather than responses elicited solely for research.

The study’s five failure-mode classes are operational analytic constructs derived from the platform’s diagnostic codebook and applied through an ordered decision procedure (Table 1), with a sixth label, not applicable, used when an item cannot be classified due to insufficient information in the dialogue. The platform supports more than 196,000 students across over 4,200 schools in the United States and United Kingdom and has delivered 2.29 million tutoring sessions.

Table 1. Failure-mode taxonomy, checked in the order shown; the first matching class is taken.
<table><tr><td>Code</td><td>Failure Mode</td><td>Definition</td><td>Example</td></tr><tr><td>UNC</td><td>Uncertainty</td><td>Student expresses uncertainty or reports a technical issue</td><td>&quot;I don&#x27;t know&quot;; “Um...&quot;</td></tr><tr><td>MIS</td><td>Misattribution</td><td>Student uses numbers or context from a different problem</td><td>Answers another question than asked</td></tr><tr><td>OP</td><td>Operator Selection</td><td>Student chooses the wrong type of operation for this problem</td><td>Multiplies when division is needed</td></tr><tr><td>CON</td><td>Conceptual Gap</td><td>Reasoning reveals a wrong underlying mathematical idea</td><td>Believes 0 × n = n</td></tr><tr><td>PROC</td><td>Procedural Slip</td><td>Correct strategy, wrong execution</td><td>Correct method with arithmetic error</td></tr><tr><td>NA</td><td>Not Applicable</td><td>Exclusion flag (not a failure mode)—item cannot be diagnosed</td><td>Inaudible response; lack of information</td></tr></table>

This study is a secondary analysis of de-identified, third-party tutoring transcripts. The transcript analysis was conducted in accordance with Cornell University’s institutional review board requirements. Our findings concern the validity of LLM-based failure-mode classification rather than downstream learning outcomes. Because invalid labels can distort downstream learning analytics, establishing measurement validity is a prerequisite to their use.

## 2.2 Data Processing and Validation

We randomly sampled 5,000 de-identified transcripts containing student responses tied to an incomplete LO, localized to that learning-objective segment within the session (termed “localized segments”). Localization (i.e., identifying the specific portion of the transcript containing the incorrect student response) used an LLM-based pass (Claude Sonnet 5) to identify the tutor’s setup question, the corresponding graded student response, and, where present, the tutor’s subsequent confirmation or correction within each learning-objective segment; this graded response served as the target for failure-mode classification. Data-quality screening removed 1,146 sampled error moments (22.9%), primarily due to diarization error or unreliable localization. The resulting 3,854-item pool consists of graded responses, each embedded in its own localized segment, for which failure-mode localization and speaker-attribution checks passed.

Of these, 330 were randomly sampled and reserved for iterative codebook and prompt refinement, including revisions to codebook decision rules and few-shot examples. Following a human-in-the-loop, error-analysis-driven prompt engineering procedure (Cohn, Mohammed, Biswas, et al., 2025), the codebook and prompt were iteratively refined over multiple rounds (≈ 10), with revisions motivated by review of disagreements between LLM predictions and human judgments. Each revision was evaluated on newly drawn, disjoint holdouts rather than items used to motivate the change, reducing the risk of overfitting the annotation scheme or prompt to previously observed cases. None of these 330 development items appeared in the final

adjudicated human reference set (discussed shortly) used for analysis in this paper, preserving a fully disjoint evaluation sample.   
The Supplementary Materials<sup>2</sup> document the resulting decision rules, worked examples, and refinement process in full.

The remaining 3,524 items formed the sampling pool for the dataset used in this study. A preliminary LLM pass generated provisional failure-mode labels used solely for stratified sampling. We selected $n = 1 0 0$ localized segments for our adjudicated human reference set, intentionally over-sampling Conceptual Gap, Procedural Slip, and Operator Selection, which the preceding refinement process had identified as particularly difficult to distinguish. Because the reference set was designed to diagnose classification failures rather than estimate population prevalence, we prioritized representation of these challenging classes over preserving their natural corpus distribution. These provisional labels were used only for sample construction and therefore do not represent the final adjudicated class distribution (presented shortly).

We chose 100 transcripts as a bounded diagnostic reference set that remained feasible for full independent annotation and adjudication while supporting overall agreement estimates and exploratory class-level error analysis. This feasibility constraint was informed by the preceding codebook and prompt refinement process, during which two researchers familiar with the codebook collectively labeled more than 500 instances across overlapping subsets of the 330 development items over two weeks. They then independently applied the finalized codebook to the graded response in each of the 100 reference-set items (i.e., localized segments), recording a label and written justification. The dialogue context available for each reference-set item ranged from 2 to 133 turns (M = 7.7).

The two raters completed their initial segment annotations independently before discussing disagreements. Initial interrater agreement was modest (Cohen’s κ = .588), with 29 items disputed. Following Thomas et al. (2026), we treated these disagreements as evidence about the analytic construct rather than annotation error, resolving disagreements through adjudication against the transcript and codebook. Adjudication was completed without access to the LLM classifications, preserving the independence of the reference labels from the model predictions, and required ≈ 90 minutes for the 29 disputed items. The resulting reference set comprised 45 Conceptual Gap (CON), 21 Not Applicable (NA), 14 Uncertainty (UNC), 8 Misattribution (MIS), 6 Procedural Slip (PROC), and 6 Operator Selection (OP) items. The final human-adjudicated distribution differs from the provisional LLM-based sampling allocation, including a greater number of items ultimately assigned to Conceptual Gap.

To contextualize our findings, we conducted semi-structured interviews (35–45 minutes) with three K–12 teachers and tutors (37, 16, and 12 years of teaching experience), who together teach (or have taught) a range of subjects including primary school mathematics, AP calculus, and language arts. Interviewees were presented the study’s findings via a slide deck and asked how these model errors might affect classroom practice and what risks might follow from providing the wrong type of feedback. This procedure was approved under an IRB exemption covering the collection and use of de-identified teacher and tutor quotations. Tutor interviews were recorded via Zoom under the IRB exemption described above and transcribed using Zoom’s automated transcription service. Transcripts were de-identified prior to analysis. The first author then reviewed the de-identified transcripts qualitatively to identify recurring perspectives, which are reported alongside representative quotations in Section 4. All study artifacts, including the final annotation codebook, LLM prompts, the tutor interview slide deck, and additional procedural details, are provided in the Supplementary Materials.

## 2.3 LLM Classification

Three LLMs, one from each of three major vendors, were evaluated via API: OpenAI’s GPT-5.6 Terra, Google’s Gemini 3.5 Flash, and Anthropic’s Claude Opus 5. Each model was selected from the corresponding vendor’s second-tier (termed “tier 2”) stable offering to support a comparable cross-vendor evaluation while keeping inference costs tractable; deliberately spanning vendors reduces (but does not eliminate) the risk that high LLM–LLM agreement merely reflects shared training data or architecture rather than genuine interpretive convergence. Each tier 2 model’s default reasoning setting (medium effort) was used. As a supplementary robustness check, we repeated the classification using Google’s Gemini 3.6 Flash (the highest-performing vendor’s top-performing model as of August 2026; termed “tier 1”) with the maximum available reasoning effort (≈ 65k tokens) to assess whether increased model and reasoning capability altered the observed agreement pattern.

For all LLM runs, classifications used schema-constrained structured JSON outputs restricted to the six allowable labels and a three-run self-consistency majority vote per item. Each model received the full codebook, learning objective, task instructions, and all preceding dialogue within the localized segment up to the utterance in question. The fixed prompt included 17 expert-verified few-shot examples drawn from the development set and disjoint from the 100-item reference set. The codebook, prompts, few-shot examples, and voting procedure were then held constant across models.

## 2.4 Evaluation Metrics

Pairwise agreement between human and LLM raters and among LLMs is quantified using Cohen’s κ. We use $\kappa = . 7 0$ as a pre-specified interpretive benchmark, following its use in prior learning analytics work on automated discourse annotation (X. Liu et al., 2025); it is not treated as a universal threshold for validity or acceptance. Multi-rater agreement across all LLMs is assessed using Krippendorff’s α. For both statistics, Not Applicable (NA) is retained as a regular label (rather than excluded) because determining that the available evidence is insufficient to support any failure-mode diagnosis is itself a meaningful classification decision. Excluding NA cases would remove the instances needed to evaluate whether models appropriately refrain from diagnosing a failure mode when the transcript does not support one.

Per-class precision, recall, and macro-F1 identify which failure modes contribute most to LLM–human disagreement, aggregated across the three primary models. McNemar’s tests evaluate whether models and human raters differ systematically in their use of NA. Item-level bootstrap resampling (1,000 resamples) provides 95% confidence intervals for agreement estimates.

## Agreement Across Model Configurations

![](images/e480097fd34f76a4fbdde0af577d7a075af72c9c3fcfb79e284b23d6c4449531.jpg)  
Figure 1. Agreement estimates across configurations, split into three blocks: human raters’ pre-consensus agreement (top), each model’s agreement with the human consensus (middle), and LLM–LLM agreement only (bottom). α replaces pairwise κ for the three-LLM comparison since it generalizes to more than two raters and is on the same 0–1 scale. The top two blocks cluster closely together, while the bottom block stands well apart—illustrating the central divergence this study examines.

## 3. Findings

## 3.1 Human–LLM Agreement

Gemini 3.5 Flash achieved the highest agreement with the human consensus (κ = .597, 95% CI [.464, .709]), followed by Claude Opus 5 (κ = .559, [.427, .685]) and GPT-5.6 Terra (κ = .524, [.385, .640]; Figure 1, middle). The tier-1 robustness check with Gemini 3.6 Flash at maximum reasoning achieved κ = .594 (95% CI [.459, .711]), effectively matching Gemini 3.5 Flash. The human–LLM agreement gap persisted across vendors, model tiers, and reasoning settings, suggesting that it is not readily resolved by increasing model capability alone.

No model’s point estimate reached the pre-specified κ = .70 benchmark: estimates ranged from .524 to .597 across the three primary models and remained at .594 for the higher-capability robustness condition, with substantially overlapping confidence intervals throughout. This LLM–human agreement was comparable in magnitude to the two human raters’ initial agreement (κ = .588; Figure 1, top), but contrasts sharply with inter-LLM agreement. We examine this divergence next.

## 3.2 LLM–LLM Agreement

Pairwise agreement among the three tier-2 LLMs (Figure 1, bottom) was consistently higher than agreement between any individual model and the adjudicated human reference set. GPT-5.6 Terra and Gemini 3.5 Flash achieved κ = .781, GPT-5.6 Terra and Claude Opus 5 κ = .770, and Gemini 3.5 Flash and Claude Opus 5 κ = .755. Agreement across all three LLMs was similarly high, with Krippendorff’s α = .769 (95% CI [.671, .854]). All LLM–LLM point estimates were substantially higher than any individual model’s agreement with the adjudicated human reference set, with slight confidence interval overlap.

Confusion Matrix by Failure-Mode Code

![](images/3e3fba929f857684cf35c6b750d4f746bff075b36890235cec989424b08bacc8.jpg)  
Figure 2. True (human consensus) vs. predicted class, pooled across the three tier-2 LLMs, row-normalized.

This divergence constitutes the study’s central finding: the three models agree strongly with one another (κ = .755–.781; α = .769)—conventionally interpreted as substantial agreement (Landis & Koch, 1977)—despite only moderate agreement with the human consensus $( \kappa = . 5 2 4 - . 5 9 7 )$ on the same 100 items. Pooling the three models into a majority-vote label (i.e., the LLM consensus; Fonteles, Cohn, et al. (2026)) did not recover the human interpretation either: the LLM-consensus label agreed with the reference set at κ = .568 (95% CI [.430, .688]), no higher than the best individual model (Gemini 3.5 Flash, $\kappa = . 5 9 7 )$ . Cross-model consensus therefore does not indicate convergence on the human interpretation; here, it coexists with systematic disagreement about the inferred learner construct.

LLM–LLM agreement exceeds the two human raters’ preadjudication agreement by .17–.19 kappa points, consistent with shared systematic bias: the models exhibit the same directional tendency in their class-level errors. Taken alone, the LLM– LLM agreement in Figure 1 could be mistaken for evidence of a reliable classification pipeline; relative to the human reference, it instead shows how cross-model consensus can overstate validity when models share the same errors. In practice, ensemble

agreement should therefore be treated as evidence of consistency, not as a substitute for human validation of LLM labels.

## 3.3 LLM Error Patterns

This shared bias is not random: it converges on a single class, Conceptual Gap (CON). Figure 2 illustrates this. For every non-CON row, the largest off-diagonal cell falls in the CON column. The matrix also reveals cross-class leakage beyond CON—for example, 10% of true NA instances are labeled MIS and another 10% are labeled OP.

Pooling predictions from the three tier-2 LLMs reveals the same pattern in aggregate (Figure 3): for every non-CON class, the most common misclassification is Conceptual Gap. The effect is strongest for Procedural Slip, with 22% recall; 61% of true PROC instances are labeled CON, yielding the lowest class-level F1 $( F 1 = . 2 8 3 , n = 6 )$ . Because PROC and OP each contain only six humanreference items, these estimates should be interpreted as diagnostic evidence of error patterns rat

![](images/46a24cb3ce4c9aff524679978b34327b9d526d42167c380793ae1b544ecd306d.jpg)  
Figure 3. Pooled tier-2 LLM predictions by reference class, ordered by macro-F1. Green indicates recall; P denotes precision. For every non-CON class, the most common error is misclassification as Conceptual Gap (red), indicating a shared directional bias.

her than precise class performance estimates.

Not Applicable (NA) exhibits a different but related failure pattern. NA predictions are relatively precise $( P = . 8 6 7 )$ but have low recall (40%), with 35% of true NA instances instead classified as CON. All three LLMs assign NA substantiall less often than the adjudicated human reference set (9–10% vs. 21% of items), with McNemar’s tests indicating statistically significant differences for each tier-2 LLM $( p \leq . 0 0 9 8 )$ . In practice, the models more often infer a conceptual difficulty than conclude that the available evidence is insufficient for diagnosis. In one such item, the student correctly stated two intermediate division facts before the response was cut off mid-step (“6 divided by 2 is 3. 6 divided by 8—so we carry the 2...[unclear]”), leaving the final answer unrecoverable from the transcript; the human reference therefore withheld a diagnosis. All three LLMs nonetheless converged on Conceptual Gap, illustrating consensus despite insufficient evidence for any failure-mode diagnosis.

Together, these patterns help explain the divergence between human–LLM and LLM–LLM agreement: models systematically over-attribute student difficulty to conceptual deficits, both when another failure mode is more appropriate and when no diagnosis is supported. These errors represent a shared directional bias rather than random noise that might average out across models. For learning analytics, such bias can systematically distort representations of learner difficulty.

## 4. Discussion and Conclusions

This study demonstrates that cross-model agreement is not necessarily evidence of valid learner interpretation. Across three LLM families, models agreed more with each other than with the human consensus while sharing systematic directional errors. For learning analytics, LLM consensus can make annotation appear reliable while masking shared misinterpretation.

## 4.1 Validity of Learner Representations in Learning Analytics

Our findings point to a broader measurement problem in learning analytics: agreement among annotators, human or model based, does not establish that learner representations validly capture the underlying construct. This is consistent with Thomas et al. (2026), who argue that inter-rater reliability should inform examination of disagreement and construct refinement rather than serve as a mechanical gatekeeper for ground truth. Similarly, Henkel, Vanacore, and Roberts (2025) found that models remained highly consistent in formative literacy assessment even as model–human agreement declined. In our case, LLMs produced consistent representations of student difficulty that diverged substantially from human consensus.

## 4.2 Instructional Consequences of Diagnostic Misclassification

When diagnostic labels drive instructional responses, misclassification has direct pedagogical consequences. Automated tutoring systems use inferred error types to select targeted feedback and remediation (McNichols, Zhang, & Lan, 2023; Reddig et al., 2025). Misclassifying a Procedural Slip as a Conceptual Gap, for example, can prompt feedback that treats an execution error as deficient understanding, despite evidence that responses to student errors should support motivation and build on what the learner actually knows (Thomas et al., 2023). One interviewed tutor described the risk of internalized harm from repeated misdirected correction: “even if it’s that procedural slip... they’re gonna start to internalize some of that, like, I’m bad at math.”

Conversely, misclassifying a genuine Conceptual Gap as a procedural slip can leave the underlying misconception insufficiently addressed, allowing it to persist (Gurung et al., 2023). Diagnostic errors are therefore not interchangeable: different misclassifications can produce different instructional consequences—a point interviewed tutors raised independently of the pedagogical literature. As one observed, reteaching a skill that was never actually missing wastes instructional time: “if you’re reteaching something because something has been misdiagnosed as something else... you’re not moving forward...what was supposed to be a tool that helps teachers is now causing more pain and time.” Learning analytics systems should evaluate not only whether learner states are classified accurately, but also the consequences of acting on incorrect diagnoses.

## 4.3 Rethinking Traditional Discourse Annotation

The validity problem is not exclusively computational: disagreement among human annotators indicates that the underlying learner construct is itself difficult to infer from transcript evidence alone. This suggests that mutually exclusive failure-mode labels may impose sharper boundaries than the data support. Multi-label and uncertainty-aware annotation may better preserve cases in which several interpretations remain plausible.

From an ICAP perspective, these exchanges provide observable evidence of constructive and interactive engagement, but our study highlights a complementary problem: even when discourse contains learning-relevant evidence, transforming that evidence into a diagnostic learner representation can introduce systematic measurement error. Transcript-only data also omit information available during tutoring, including prosody, written work, screen state, gestures, and other interaction traces. Some disagreement may therefore reflect missing evidence rather than deficient reasoning. Prior learning analytics work demonstrates the value of combining discourse with complementary trace data for contextualization (Snyder et al., 2024), suggesting valid failure-mode diagnosis may benefit from multimodal evidence.

## 4.4 Limitations and Future Work

This exploratory study uses 100 deliberately sampled error moments from a single K–12 mathematics tutoring platform, limiting population-level and cross-domain generalizability; several classes also remain small. Practitioner perspectives were drawn from semi-structured discussions with three tutors rather than a systematic or representative stakeholder sample, and student perspectives were not collected. We also evaluate diagnostic validity rather than downstream consequences: the effects of failure mode misclassification on feedback, self-efficacy, misconception persistence, engagement, and learning remain unknown and should be examined alongside broader stakeholder perspectives. Future work should also consider larger reference sets, multimodal evidence, multi-label taxonomies, and uncertainty-aware inference; and should assess whether these findings generalize across platforms, subjects, and tutor and student populations.

The central implication is nevertheless clear: before learning analytics asks whether LLM-generated diagnoses improve learning, it must establish that those diagnoses validly represent learner difficulty. When models share the same biases, consensus can increase confidence without increasing correctness. Agreement demonstrates consistency; validity requires evidence that the learner has been interpreted correctly.

## References

Ahtisham, B., et al. (2026). Ai annotation orchestration: Evaluating llm verifiers to improve the quality of llm annotations in learning analytics. In Proceedings ofthe lak26: 16th international learning analytics and knowledge conference.

Booth, B., et al. (2024, March). Human-tutor Coaching Technology (HTCT): Automated Discourse Analytics in a Coached Tutoring Model. In Proceedings ofthe 14th Learning Analytics and Knowledge Conference.

Borchers, C., et al. (2026). Brief but impactful: How human tutoring interactions shape engagement in online learning. In Proceedings ofthe lak26: 16th international learning analytics and knowledge conference (pp. 160–170).

Borchers, C., Thomas, D. R., Lin, J., & Koedinger, K. R. (2026). Learning-aware reliability estimation for tutor skill assessment using large language models. Journal ofLearning Analytics, 1–20.

Chi, M. T., & Wylie, R. (2014). The icap framework: Linking cognitive engagement to active learning outcomes. Educational psychologist, 49(4), 219–243.

Cohn, C., Mohammed, N., Biswas, G., et al. (2025). Cotal: Human-in-the-loop prompt engineering for generalizable formative assessment scoring and feedback. arXiv preprint arXiv:2504.02323.

Dewey, D. C., et al. (2026, May). From learning recession to learning recovery: Understanding the sources of u.s. k-12 improvement (Tech. Rep.). Education Recovery Scorecard.

Fonteles, J. H., Cohn, C., Ayalon, E., Zhou, M., T.S, A., Davalos, E., . . . Biswas, G. (2026). Analyzing embodied learning in classroom settings: A human-in-the-loop AI approach for multimodal learning analytics. Learning and Instruction, 103, 102274. doi: https://doi.org/10.1016/j.learninstruc.2025.102274

Fonteles, J. H., Sivakumaran, N., Cohn, C., Coursey, A., Yu, S., Stengel-Eskin, E., . . . Biswas, G. (2026). A Novel Approach to Evaluating the Effectiveness of Large Language Models for Multimodal Analysis of Embodied Learning in Classrooms. Proceedings ofthe 16th International Learning Analytics and Knowledge Conference (LAK).

Garg, R., et al. (2024). Automated discourse analysis via generative artificial intelligence. In Proceedings ofthe 14th learning analytics and knowledge conference (pp. 814–820).

Gurung, A., et al. (2023). Identification, exploration, and remediation: Can teachers predict common wrong answers? In 13th international learning analytics and knowledge conference (pp. 399–410). doi: 10.1145/3576050.3576109

Gures, F. B., et al. (2026, April). Structuring versus Problematizing: How LLM-based Agents Scaffold Learning in Diagnostic¨ Reasoning. In Proceedings ofthe LAK26: 16th International Learning Analytics and Knowledge Conference.

Henkel, O., Vanacore, K., & Roberts, B. (2025). When humans can’t agree, neither can machines: The promise and pitfalls of llms for formative literacy assessment. In Proceedings ofthe artificial intelligence in measurement and education conference (aime-con): Coordinated session papers (pp. 69–78).

Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. biometrics, 159–174.

Liu, X., Zambrano, A. F., Baker, R. S., Barany, A., Ocumpaugh, J., Zhang, J., . . . Wei, Z. (2025). Qualitative coding with gpt-4: Where it works better. Journal ofLearning Analytics, 12(1).

Liu, Z., et al. (2026, April). Talking the Talk: Linking Instructional Discourse Patterns to Student In-video Dropout and Learning Outcome. In Proceedings ofthe LAK26: 16th International Learning Analytics and Knowledge Conference.

McNichols, H., Zhang, M., & Lan, A. (2023). Algebra error classification with large language models. In International conference on artificial intelligence in education (pp. 365–376).

Misiejuk, K., et al. (2025). Mapping the landscape of generative artificial intelligence in learning analytics: A systematic literature review. Journal ofLearning Analytics, 12(1), 12–31.

Reddig, J., et al. (2025). Generating in-context, personalized feedback for intelligent tutors with large language models. International Journal ofArtificial Intelligence in Education, 35, 3459–3500. doi: 10.1007/s40593-025-00505-6

Scarlatos, A., Baker, R. S., & Lan, A. (2025). Exploring knowledge tracing in tutor-student dialogues using llms. In Proceedings ofthe 15th international learning analytics and knowledge conference (pp. 249–259).

Snyder, C., et al. (2024). Analyzing students collaborative problem-solving behaviors in synergistic STEM+ C learning. In Proceedings of the 14th Learning Analytics and Knowledge Conference (pp. 540–550).

Suraworachet, W., et al. (2024). Predicting challenge moments from students’ discourse: A comparison of gpt-4 to two traditional natural language processing approaches. In 14th learning analytics and knowledge conference (pp. 473–485).

Thapa Magar, A., et al. (2026, April). Understanding and Modeling Math Strategy Use in Intelligent Tutoring Systems. In Proceedings ofthe LAK26: 16th International Learning Analytics and Knowledge Conference.

Thomas, D. R., et al. (2023). When the tutor becomes the student: Design and evaluation of efficient scenario-based lessons for tutors. In Proceedings ofthe 13th international learning analytics and knowledge conference (pp. 250–261). New York, NY, USA: Association for Computing Machinery. doi: 10.1145/3576050.3576089

Thomas, D. R., et al. (2026). Modernizing ground truth: Four shifts toward improving reliability and validity in ai in education. In International conference on artificial intelligence in education (pp. 117–131).

Vanacore, K., & Kizilcec, R. (2026). How well do large language models recognize instructional moves? establishing baselines for foundation models in educational discourse. In Proceedings ofthe thirteenth acm conference on learning at scale.

Xavier, C., et al. (2026). From solo graders to assisted annotation: Integrating llm suggestions into the educational data creation pipeline. In Proceedings of the lak26: 16th international learning analytics and knowledge conference (pp. 96–105).

Xu, Z., et al. (2026, April). Enhancing LLM-Based Data Annotation with Error Decomposition. In Proceedings ofthe LAK26: 16th International Learning Analytics and Knowledge Conference. Association for Computing Machinery.

Zhang, J., et al. (2026). Using Large Language Models to Detect Socially Shared Regulation of Collaborative Learning. Proceedings of the 16th International Learning Analytics and Knowledge Conference (LAK).

Zhu, Z., et al. (2026, April). A Synergistic Framework for Cognitive Diagnosis with LLM-Empowered Data Augmentation. In Proceedings ofthe LAK26: 16th International Learning Analytics and Knowledge Conference.