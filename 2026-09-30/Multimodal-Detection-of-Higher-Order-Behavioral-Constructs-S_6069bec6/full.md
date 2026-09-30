# Multimodal Detection of Higher-Order Behavioral Constructs: Self-Compassion in Structured Reflective Interaction

Siddhant Jain Deutsches Forschungszentrum für Künstliche Intelligenz GmbH (DFKI) Saarbrücken, Saarland, Germany Universität des Saarlandes Saarbrücken, Saarland, Germany siddhant.jain@dfki.de

Dimitra Tsovaltzi Deutsches Forschungszentrum für Künstliche Intelligenz GmbH (DFKI) Saarbrücken, Saarland, Germany Universität des Saarlandes Saarbrücken, Saarland, Germany dimitra.tsovaltzi@dfki.de

![](images/b53572052aea2b6c1573018f2d8d505ebb697e215222d906cb49c514b0a5e86d.jpg)  
Figure 1: A participant interacts with a technology-mediated training scene (left), illustrating the kind of structured reflective interaction studied in this work, alongside a related moment of self-reflection in a digital context (right).

## Abstract

Many ofthe qualities that matter most in how people learn and grow, how someone regulates their emotions, reflects on a setback, or stays aware ofothers during a dificult conversation, are not directly observable. They have to be inferred from how someone speaks, moves, and sounds over time, and they resist the kind of clean labeling that most machine learning pipelines are built around. We study this challenge through a case that is well grounded in psychological theory but rarely modeled computationally: self-compassion, the tendency to respond to one’s own setbacks with patience rather than harsh self-criticism. We examine how it appears during structured reflective interviews in a technology-mediated training setting, where people naturally talk through socio-emotionally demanding situations. Since no existing dataset captures this kind of construct in this kind of setting, we collected and annotated 51 reflective dialog sessions using an independent, temporally overlapping annotation scheme grounded in established theory. We consolidate the underlying six-component psychological model into a three-class supervision space, balancing self-kindness and mindfulness against self-critical or overwhelmed states, and build a reproducible window-based pipeline that aligns video, audio, and text on a shared timeline. Unimodal models trained on each modal ity separately are compared against a simple probability-level fusion strategy, which yields modest but consistent gains over the best single modality. We close by discussing where each modality succeeds or struggles, what this suggests about how this kind of construct is actually expressed in reflective speech, and what would be needed to model it, and constructs like it, more efectively.

## CCS Concepts

• Computing methodologies → Machine learning approaches; • Human-centered computing → Human computer interaction (HCI); • Applied computing → Interactive learning environments.

## Keywords

complex behavioral constructs, multimodal machine learning, self compassion, interaction modeling, reflective practice, multi-label classification, late fusion

## 1 Introduction

Many of the qualities that matter most in situated, interactive settings, reflective regulation, collaborative awareness [26], adaptive self-evaluation, are not directly observable. They have to be inferred from behavioral signals that unfold over time and are embedded in ongoing interaction. Unlike basic emotion categories, these constructs are theory-driven and conceptually structured rather than directly measurable: their expression is distributed across language, vocal prosody, and non-verbal movement, and the expert annotations used to study them are typically sparse, temporally extended, and overlapping. Together, these properties make such constructs dificult to represent, supervise, and evaluate with standard machine learning tooling.

Self-compassion may serve as an exemplary case for studying this dificulty, since it comes with an established psychological framework [17, 18] but has rarely been examined outside self-report questionnaires. We study it as it manifests during structured reflective interviews. Reflective practice is a central mechanism in professional development [23], but reflection can also intensify rumination or harsh self-evaluation when emotional regulation is inefective, and lead to counterproductive over-identification experiences [13, 19]. Self-compassion ofers a structured account of adaptive self-regulation under exactly this kind of dificulty, and teachers and teacher trainees who report higher self-compassion show lower burnout and greater professional well-being [16]. What has remained largely unexplored is how self-compassion actually shows up, moment to moment, while someone is reflecting out loud.

To study this, we draw on data collected within a technologymediated training system designed to support socio-emotional skills in high-stakes professional interactions [4]. Participants first take on a professional role in a simulated conflict scenario and are then guided by a human interviewer through a structured reflection on their own recorded behavior. Because no existing multimodal corpus targets self-compassion in this kind of structured reflective setting, we collected and annotated a dedicated dataset as part of this project.

The paper makes four contributions: a transformation from theory-driven, independently annotated constructs into a modelingcompatible multi-label representation, consolidating the original six-component scheme into three classes under empirical sup port constraints; a reproducible multimodal preprocessing and alignment pipeline covering video, audio, and text under a shared window-based temporal backbone; a systematic unimodal evaluation of video, audio, and text under an identical supervision protocol; and an evaluation of probability-level late fusion as a controlled, interpretable multimodal integration strategy. We present these as baseline results for a task that, to our knowledge, has not previously been addressed in this combination of construct, setting, and modeling formulation. Thus, we contribute to methodologies for modeling higher-order behavioral constructs more broadly, using self-compassion as one instantiation of this problem rather than as an end in itself.

## 2 Background and Related Work

## 2.1 Self-Compassion as a Regulatory Construct

Nef formalizes self-compassion as a multidimensional construct with three bipolar dimensions [17, 18]. Self-Kindness versus Self-Judgment concerns whether one responds to personal dificulty with care or with harsh criticism. Common Humanity versus Isolation concerns whether sufering is recognized as a shared human experience or perceived as uniquely personal. Mindfulness versus Over-Identification concerns whether painful thoughts and emotions are held with balanced awareness or become overwhelming. Self-compassion is theoretically distinct from self-esteem, which is typically contingent on performance and comparison, and from selfpity, which may exaggerate personal sufering without maintaining perspective [18].

Although originally described as bipolar, empirical work suggests the positive and negative poles can co-occur and fluctuate independently within short intervals [25], motivating a multi-label rather than mutually exclusive formulation. Reflective episodes frequently involve perceived mistakes, interpersonal conflict, or uncertainty, and when emotional regulation is insuficient, reflection can slide into rumination or harsh self-evaluation [13, 19].

## 2.2 Behavioral Correlates of Self-Compassion

Facial expression and vocal prosody are broadly understood to carry afective and regulatory information [9, 22], and the most directly relevant evidence for self-compassion specifically comes from work analyzing facial and acoustic expressions of self-compassion, self-criticism, and self-protection in emotion-focused therapy sessions [1, 2]. These studies show that self-compassion-related states manifest in measurable facial and vocal behavior, but they are unimodal, use event-based rather than window-aligned supervision, which fixes a consistent temporal unit across modalities rather than variable-length labeled events, and do not address cross-modal alignment or multi-label component structure. Our work extends this line of evidence to a diferent setting, structured professional reflection rather than therapy, and to a multimodal, multi-label formulation that captures overlapping, co-occurring signal a singlelabel or single-modality design would miss.

## 2.3 Multimodal Modeling of Complex Behavioral Constructs

Recent multimodal afective interaction modelling has combined pretrained modality-specific encoders with transformer-based fusion, including cross-modal attention [24], dense shared-private representations [7], noise-resistant training [14], and reliabilityaware weighting [27]. These architectures perform well on established benchmarks, but that performance typically assumes short clips, single-label supervision, and comparatively large training corpora. Reflective dialog data looks diferent on all three counts: sessions are long, annotations are overlapping and multi-label, and the number of sessions is small, conditions under which high-capacity joint architectures are prone to overfitting rather than learning useful cross-modal structure. We therefore adopt a lower-capacity, modular design instead: independent unimodal encoders combined through probability-level late fusion, which avoids introducing additional trainable cross-modal parameters and keeps each modality’s contribution interpretable. Detection itself is formulated as multi-label classification [30], since self-compassion’s components are theorized to co-occur rather than to be mutually exclusive [25], which allows leveraging a rich context of interaction to model abstract constructs which otherwise remain underspecified.

On the language side, cross-modal and instruction-tuned models such as CM-BERT [29] and Emotion-LLaMA [5] demonstrate that pretrained and instruction-following language models capture af fectively relevant textual signal, and linguistic markers such as pronoun use and self- versus other-focused language have been shown to index self-relating and interpersonal stance in psychotherapy transcripts [21]. Comparatively little work, however, has examined whether general-purpose or instruction-following language models capture theory-grounded, multi-component constructs like selfcompassion without task-specific adaptation, which motivates the fine-tuning approach we take for the text modality.

This gap reflects a broader pattern: multimodal modeling of higher-order, theory-grounded constructs remains underexplored outside a small set of benchmark emotion categories. Self-compassion, and the structured reflective setting in which we study it, ofers one concrete route into this broader problem, which is what motivated the data collection and modeling approach described in the remainder of this paper.

## 3 Problem Formulation

## 3.1 Research Questions

## This work is guided by four questions.

RQ1: Can an abstract theory-driven, six-component psychological construct be consolidated into a supervision scheme that remains faithful to the underlying theory while matching the empirical support available in the collected corpus?

RQ2: What alignment and supervision protocol is needed to bring facial, acoustic, and linguistic signal, each sampled at a different rate and subject to diferent sources of noise, onto a shared temporal backbone suitable for cross-modal comparison?

RQ3: When modeled independently under an identical protocol, do video, audio, and text carry comparable, complementary, or redundant information about self-compassion as it is expressed during structured reflective dialog?

RQ4: Does a low-capacity, interpretable late fusion strategy recover meaningful complementary signal across modalities, and how large is that gain relative to the strongest single modality?

## 3.2 Label Consolidation

All six components were annotated independently as overlapping ELAN tiers. For supervised modeling, we consolidate this into a three-class space: Mindfulness (MF), Self-Kindness (SK), and Negative components (NEG), where NEG aggregates Self-Judgment and Over-Identification. Common Humanity and Isolation were excluded due to sparse temporal coverage (121s and 15s of total annotated duration across 5 and 1 videos respectively, versus 7,137s for MF and 3,056s for SK). This consolidation reflects empirical constraints in the collected corpus, not a theoretical claim that the excluded or aggregated components are equivalent; we return to this trade-of in Section 9.

## 3.3 Component Co-Occurrence

Prior work indicates that self-compassion components may fluctuate independently within short temporal intervals rather than behaving as mutually exclusive states [25]: a speaker may express mindful awareness while simultaneously engaging in partial selfcriticism, or express self-kindness alongside over-identification. This partial independence motivates treating detection as a multilabel rather than single-label problem, allowing overlapping regulatory tendencies to be identified concurrently within the same temporal window.

![](images/93b04146c6a8d355da82a73a6ad3f8f9181e6b94c6ea8230e4272632c8563941.jpg)  
Figure 2: Example ELAN annotation interface showing independent tiers for self-compassion components with permitted temporal overlap.

## 3.4 Multi-Label Formulation

The task is formulated as multi-label classification: each unit of analysis receives independent binary labels for {MF, SK, NEG}, and a model produces independent sigmoid-linked probability estimates for each class without cross-class normalization.

## 4 Dataset and Annotation

## 4.1 Data Collection

The dataset consists of 51 annotated reflective dialog sessions, drawn from a two-stage technology-mediated training study, illustrated here with a teacher training scenario as its concrete context [4]. In the first stage, participants took on a professional role, in this case a substitute teacher, in a simulated conflict scenario with virtual agents. In the second stage, a trained human interviewer guided each participant through a post-interaction interview, replaying critical moments of their recorded behavior and asking them to reflect on their experience and the motives behind it, in a setting explicitly framed as safe and non-judgmental. The present work targets self-compassion as it appears during this second, reflective stage. Session durations range from 600s to 4,956s (mean ≈3,489s). Audio and video are synchronized, and transcripts are derived via automatic speech recognition and aligned by timestamp across modalities.

## 4.2 Annotation Scheme

Each psychological construct was annotated as an independent ELAN [28] tier with start and end timestamps. Temporal overlap between tiers was permitted: an interval labeled with one construct did not preclude simultaneous labeling with another, as shown in Figure 2. To support annotation consistency, approximately 15% of sessions were independently coded by a second annotator, with discrepancies discussed and resolved before finalizing the scheme; we did not compute a numeric agreement statistic such as Cohen’s kappa.

Table 1: Total annotated duration and video coverage per original construct.
<table><tr><td>Construct</td><td>Total Duration (s) # Videos</td></tr><tr><td>MF (Mindfulness)</td><td>7136.6 51</td></tr><tr><td>SK (Self-Kindness)</td><td>3055.7 38</td></tr><tr><td>SJ (Self-Judgment)</td><td>650.3 16</td></tr><tr><td>OI (Over-Identification)</td><td>399.9 7</td></tr><tr><td>CH (Common Humanity)</td><td>121.5 5</td></tr><tr><td>ISO (Isolation)</td><td>15.0 1</td></tr></table>

An overlap and coverage audit, quantifying interval counts, total duration, per-session coverage, and pairwise temporal overlap per construct, confirmed the imbalance in Table 1: MF and SK account for the majority ofannotated duration, SJ and OI show moderate but uneven distribution, and CH and ISO have limited coverage across sessions, owing to the fact that the interview targeted the activation of positive self-compassion sub-constructs. These findings directly motivated the consolidation described in Section 3.

## 4.3 Window-Based Supervision

Annotated sessions are segmented into overlapping 4-second windows with a 1-second hop. A window is labeled positive for class � if

$$
\frac { \mathrm { d u r a t i o n } ( w \cap a _ { c } ) } { \mathrm { d u r a t i o n } ( w ) } \geq 0 . 2 ,
$$

where $a _ { c }$ denotes an annotation interval of class �; for a 4-second window this corresponds to at least 0.8 seconds of overlap. This threshold reduces sensitivity to marginal boundary efects while retaining short but meaningful annotations. Segmentation across all 51 sessions produced 155,960 total windows, of which 10,658 (6.83%) carry at least one positive label.

## 4.4 Splits

Splits are defined at the video level, not the window level, to prevent temporal leakage, using an 80/10/10 train/validation/test ratio with a fixed seed, stratified with respect to the original six annotated constructs prior to consolidation. This yields 24,249 training, 3,450 validation, and 4,275 test windows used for model fitting and selection; evaluation in Section 7 is further restricted to the subset of test windows carrying at least one valid supervision label (� = 1,425). All statistical fitting, standardization and imputation, is performed exclusively on the training split. The 24,249/3,450/4,275 windows reflect those falling within self-compassion construct segments rather than the full 155,960 windows generated across entire session recordings, which include stretches of the interview outside any annotated construct.

## 5 Multimodal Preprocessing and Alignment

Recordings arrive at heterogeneous resolutions, frame rates, and bitrates, so all sessions are first standardized, video re-encoded to a common resolution and codec, audio resampled to a consistent rate, before any cross-modal alignment is attempted. All modalities then share a common temporal reference derived from these standardized recordings. Alignment proceeds through speaker diarization, automatic speech recognition, and interval-based matching to ELAN annotations.

![](images/31606c7c9289aea31f15ac71e607ea25697ef02621c8f98b9c589cc7f90dc7fd.jpg)  
Figure 3: Temporal alignment between raw video, audio, transcribed language, and ELAN annotations into aligned learning segments.

![](images/524915ce40273989c004f3f585bb71e78708e0d769a6e5998cca4253b5bb03bb.jpg)  
Figure 4: Illustration of MediaPipe-based facial blendshape and upper-body pose landmark extraction.

Figure 3 illustrates this alignment backbone across modalities.

## 5.1 Video

Facial blendshape coeficients and upper-body pose landmarks are extracted at 10 frames per second using MediaPipe [15], yielding 40 time steps per 4-second window.

Figure 4 illustrates this extraction process.

## 5.2 Audio

Speaker diarization is performed with pyannote.audio [3], and acoustic features are extracted with openSMILE using the eGeMAPSv02 configuration [10, 11], aligned to windows via temporal overlap and pooling.

## 5.3 Text

Automatic speech recognition is performed with Faster-Whisper (medium model, German) [20]. Text classification operates at the

transcript segment level; segment-level predictions are subsequently projected onto the fixed 4-second windows for fusion and evaluation.

## 6 Modeling and Fusion

Each modality is modeled independently under the same multi label formulation, producing three independent sigmoid outputs for {MF, SK, NEG}, and evaluated using the window-level protocol of Section 4 (with text projected from the segment level).

## 6.1 Unimodal Models

Video: A single-layer GRU [6] with hidden size 128 processes the 40-step frame sequence within each window; the final hidden state feeds a fully connected layer with three sigmoid heads and a dropout of 0.2. The model is trained with AdamW (learning rate $5 \times 1 0 ^ { - 4 }$ weight decay $2 \times 1 0 ^ { - 2 } )$ for up to 30 epochs with early stopping on validation loss, using per-class positive weighting in the binary cross-entropy loss to address class imbalance.

Audio: A logistic regression model (L2 penalty, liblinear solver, �=1.0), implemented as three independent one-vs-rest classifiers over standardized eGeMAPS functionals with per-class class-weighting, serves as the primary baseline; an exploratory MLP (hidden size 256) did not yield consistent gains and is not reported here in detail.

Text: A LLaMA-3.2 instruction model (3B parameters) [12] is adapted with QLoRA [8] (rank 16, alpha 32, 4-bit NF4 quantization) for parameter-eficient supervised fine-tuning at the transcript segment level, using context-enhanced input (target segment plus preceding transcript context) over 3 epochs with an efective batch size of 4.

## 6.2 Evaluation Protocol

Each model produces independent class probabilities $\hat { p } \in [ 0 , 1 ] ^ { 3 }$ without cross-class normalization. Threshold-dependent metrics (precision, recall, F1) use a fixed decision threshold of 0.5; PR-AUC and ROC-AUC are computed directly from continuous probabilities. Macro-averaged metrics are the unweighted mean across classes; micro-averaged metrics aggregate true/false positives across classes before computing the metric. Model selection uses validation perfor mance exclusively; no test-set information informs hyperparameter tuning, thresholding, or fusion weight selection.

## 6.3 Late Fusion

Given the limited number of annotated sessions and heterogeneous modality representations, we do not pursue joint end-to-end multi modal training. Instead, for modality set � ⊆ {Video, Audio, Text} and class $c \in$ {MF, SK, NEG}, each modality � ∈ � produces a probability $\begin{array} { r } { p _ { m , c } , } \end{array}$ and the fused probability is

$$
\hat { p } _ { c } = \sum _ { m \in M } w _ { m } \ : p _ { m , c } , \qquad w _ { m } \geq 0 , \ : \sum _ { m \in M } w _ { m } = 1 ,
$$

with weights shared across classes. Weight configurations are selected by validation macro-F1 on a fixed grid and applied without modification to the held-out test split; no unimodal models are retrained during fusion.

Figure 5 summarizes this architecture.

![](images/93a2f2d398d9bc0a9b1590eaed7b5f6857cf6bdc7ad7d77f1190492efc74821e.jpg)  
Figure 5: Probability-level late fusion: independent unimodal encoders produce �(� | modality), combined via fixed or validation-selected weighting.

## 7 Results

Evaluation is restricted to windows carrying at least one valid supervision label in the test split (� = 1,425), with class support of 0.687 (MF), 0.124 (SK), and 0.051 (NEG). Threshold-dependent metrics use a fixed decision threshold of 0.5.

Table 2: Window-level unimodal performance (test split). M-F1: macro-F1; �-F1: micro-F1.
<table><tr><td>Model</td><td>M-F1</td><td> $\mu { \mathrm { - } } \mathrm { F } 1$ </td><td>PR-AUC</td><td>ROC-AUC</td></tr><tr><td>Video (GRU)</td><td>0.331</td><td>0.610</td><td>0.30</td><td>0.77</td></tr><tr><td>Audio (LogReg)</td><td>0.252</td><td>0.379</td><td>0.15</td><td>0.47</td></tr></table>

Table 3: Majority-class baseline (test split), derived from class prevalence.
<table><tr><td>Class</td><td>Prevalence</td><td>Majority Label</td><td>F1</td></tr><tr><td>MF</td><td>0.687</td><td>Positive</td><td>0.814</td></tr><tr><td>SK</td><td>0.124</td><td>Negative</td><td>0.000</td></tr><tr><td>NEG</td><td>0.051</td><td>Negative</td><td>0.000</td></tr></table>

Table 3 reports the F1 achieved by always predicting the majority label per class, computed analytically from class prevalence. For SK and NEG, both unimodal models comfortably exceed this trivial baseline. For MF, however, the majority baseline (0.814) exceeds the strongest unimodal model’s F1 (Video, 0.723), indicating that F1 alone is a misleading measure of learned discrimination on this class given the severe imbalance; PR-AUC and ROC-AUC (Table 2) should be weighted more heavily than F1 when interpreting MF performance.

Table 4: Per-class F1 (window-level, test split).
<table><tr><td>Model</td><td>MFF1</td><td>SK F1</td><td>NEG F1</td></tr><tr><td>Video</td><td>0.723</td><td>0.224</td><td>0.046</td></tr><tr><td>Audio</td><td>0.522</td><td>0.234</td><td>0.000</td></tr></table>

![](images/a8fef871ced751440c4c562220d271ab3fd0a2861b51928339be41daf04982c3.jpg)  
Figure 6: Macro-F1 and micro-recall across unimodal and late fusion configurations (test split, window-level).

Audio fails to detect NEG entirely at the 0.5 threshold (F1 = 0.000), consistent with NEG’s low support (0.051) and audio’s comparatively weak overall discrimination (Table 2); we return to this in Section 8.

Text is evaluated separately at the segment level, since its supervision unit difers from the window-level video and audio models. The fine-tuned model reaches a macro-F1 of 0.49 at the segment level (MF F1 0.58, SK F1 0.38, NEG F1 0.50), the highest macro-F1 among individual modalities, reflecting the more direct semantic access that linguistic content provides for explicitly verbalized reflective states.

Table 5: Late fusion comparison (test split, window-level).
<table><tr><td>Configuration</td><td>Macro-F1</td><td>Micro-Recall</td></tr><tr><td>Audio (unimodal)</td><td>0.321</td><td>0.434</td></tr><tr><td>Video (unimodal)</td><td>0.331</td><td>0.606</td></tr><tr><td>Text + Audio</td><td>0.340</td><td>0.461</td></tr><tr><td>Text + Video</td><td>0.335</td><td>0.618</td></tr><tr><td>Audio + Video</td><td>0.345</td><td>0.633</td></tr><tr><td>Text + Audio + Video</td><td>0.350</td><td>0.645</td></tr></table>

Table 6: Validation-selected fusion weights.
<table><tr><td>Configuration</td><td>Wvideo</td><td>Waudio</td><td>Wtext</td></tr><tr><td>Video + Audio</td><td>0.85</td><td>0.15</td><td>一</td></tr><tr><td>Video + Text</td><td>0.85</td><td>一</td><td>0.15</td></tr><tr><td>Audio + Text</td><td></td><td>0.85</td><td>0.15</td></tr><tr><td>Tri-modal</td><td>0.33</td><td>0.33</td><td>0.33</td></tr></table>

Validation-selected weights favor video in every pairwise configuration that includes it $( w _ { \mathrm { v i d e o } } = 0 . 8 5 )$ , while the tri-modal configuration is best served by uniform weighting $( w _ { m } = 0 . 3 3$ for all �). The tri-modal fusion achieves the highest macro-F1 (0.350), an absolute improvement of 0.019 over the strongest window-level unimodal baseline (Video, 0.331), and also improves micro-recall over every unimodal or pairwise configuration.

## 8 Discussion

The results split cleanly along two lines: how well a class is supported in the annotation, and what kind of signal each modality actually has access to, behavioral, acoustic, or linguistic.

## 8.1 Class-Specific Behavior

Mindfulness (MF) has the highest support ratio (0.687 of evaluated windows) and the most stable learning behavior across modalities, consistent with the general pattern in modeling abstract, theorydriven constructs: better-supported classes with clearer behavioral or lexical correlates perform closer to well-studied afective categories, while sparser, more inferential classes lag behind [1, 2]. That stability comes with a catch: MF’s high prevalence puts the majorityclass baseline at F1 0.814 (Table 3), above every trained unimodal model’s F1 on this class. So MF’s strong F1 numbers owe more to class imbalance than to learned discrimination, and PR-AUC and ROC-AUC (Table 2) are the better guide to what the models actually capture here. Text-based modeling performs consistently for MF, as reflective awareness and descriptive language provide accessible lexical cues; video also captures non-verbal correlates of reflective engagement within short windows.

Self-Kindness (SK) has substantially lower support, which increases sensitivity to modeling choices. Text modeling captures explicit supportive self-address when present, but the same sentiment is often expressed in diferent words across speakers, contributing to variability; video and audio capture indirect afective correlates but show limited standalone reliability. SK’s low prevalence puts its majority-class baseline at zero, so unlike MF, every point of F1 the models earn here reflects real signal.

Negative components (NEG) aggregate self-judgment and over-identification and retain the lowest support ratio (0.051) despite this aggregation. Text modeling benefits from explicit evaluative language, but lexical ambiguity between reflective acknowledgment and evaluative judgment contributes to classification difficulty: a statement like "I didn’t handle that well" can read as balanced acknowledgment of a mistake or as harsh self-judgment, depending on tone and context. Video and audio may capture tension or heightened afect, but such cues are indirect and overlap with other states. As with SK, NEG’s majority baseline is zero, so the modest scores here are genuine, not an artifact of imbalance.

## 8.2 Modality-Specific Strengths and Constraints

Video captures frame-level facial activity and upper-body posture within 4-second windows via recurrent sequence modeling, which supports detection of short-term behavioral dynamics but limits modeling of gradual afective trajectories that unfold over longer spans. Audio, based on eGeMAPS functionals aggregated over diarization-defined segments, captures paralinguistic information that is less directly tied to construct semantics, and speaker variability introduces additional noise; the limited gap between logistic regression and a non-linear MLP suggests added model capacity does not substantially change discriminative capacity under current data conditions. Text is evaluated at the segment level rather than the window level, because a 4-second window often contains too little text, sometimes just a word or two, to carry reliable meaning on its own. This gives text the most direct semantic access to reflective and evaluative content, consistent with findings in adjacent domains that linguistic markers such as first-person pronoun use and self- versus other-focused language carry meaningful signal about self-relating and interpersonal stance in spoken interaction [21]. Its constraints are the flip side of this design: automatic transcription errors, and the need to project segment-level predictions onto windows for fusion. No single modality dominates across classes, and performance diferences are class-dependent, reflecting how constructs are diferentially expressed in reflective dialog.

## 8.3 Interpretation of Fusion Behavior

Incremental gains under pairwise and tri-modal fusion indicate partial complementarity: when one modality provides uncertain pre dictions, another provides supporting evidence. Validation-selected weights favor video in every pairwise configuration that includes it (Table 6), so most of the discriminative signal at the window level is behavioral, with audio and text adding correction on top rather than contributing an equal share. The uniform tri-modal weighting fits this picture: once all three are available, none of them fully replaces the others, but the gain each adds beyond video is modest. Weight exploration on the validation split showed smooth performance variation, and test performance aligned with validation trends, suggesting fusion did not rely on unstable configurations. Probability-level late fusion avoids joint optimization across heterogeneous feature spaces, which, given the limited number of sessions and pronounced class imbalance, prioritizes stability and interpretability over the added capacity of joint multimodal architectures.

## 9 Limitations and Future Work

Several of the limitations in this section follow directly from working with a newly collected, real-world corpus rather than an established benchmark, and are worth stating plainly rather than folding into the discussion above.

• Efective sample size: Sliding-window extraction increases the number of labeled instances (155,960 windows), but statistical independence is determined at the session level; the efective sample size is bounded by the 51 annotated sessions rather than the number of windows.

• Class imbalance and its source: Two of the original six theoretical components, Isolation and Common Humanity, were excluded from modeling due to sparse coverage (Section 3, Table 1). This imbalance is not purely a sampling artifact: the reflective sessions were designed to encourage constructive reflection and perspective-taking, which plausibly reduces the frequency of isolating or negatively valenced expression within this structured setting. The same imbalance also means F1 is a weaker measure of model quality on the best-supported class than on the others, as the majorityclass baseline in Section 7 shows.

• Single-corpus evaluation: All results are corpus-specific to one structured, interviewer-guided training environment; generalization to spontaneous reflection or other educational or alternative settings has not been assessed.

• Fixed temporal granularity: The 4-second, 1-second-hop windowing imposes a uniform discretization on reflective processes of inherently variable duration and may not align with construct boundaries.

• ASR and diarization noise: Text-based modeling relies on automatic speech recognition without manual correction, and speaker diarization is similarly imperfect; both introduce an upper bound on achievable text performance and add alignment noise that is applied consistently but not eliminated.

• Evaluation design: Evaluation is descriptive (F1, PR-AUC, ROC-AUC); no statistical significance testing or split-sensitivity analysis was performed. Probability calibration was also not performed, as the focus here was relative discriminative performance rather than deployment-ready probability estimates.

• Fusion capacity: Probability-level late fusion with shared, static weights is deliberately low-capacity; it does not model cross-modal feature interactions and treats classes symmetrically in weighting.

Future work includes cross-corpus and cross-domain validation, more expressive and class-aware fusion strategies, probability calibration and uncertainty estimation, and a closer analysis of what fine-tuned language models learn about theory-grounded linguistic markers of these constructs relative to their pretrained, non-finetuned counterparts, which we are pursuing as an extension of this work.

## 10 Data and Code Availability

Code for preprocessing, feature extraction, model training, and fusion will be released publicly upon acceptance. The dataset contains identifiable video and audio of human participants collected under a consent protocol that did not anticipate public redistribution (Section 11); data access beyond what is reported here is handled on a case-by-case basis and is not guaranteed. Sections 4 and 5 report the annotation protocol, window construction, and feature extraction pipeline in suficient detail to support replication on comparably collected data.

## 11 Ethical Considerations

The sessions underlying this dataset involve recorded facial video, audio, and speech from human participants engaging with emotionally sensitive material (perceived mistakes, self-evaluation, professional dificulty). Data collection followed institutional ethics review and informed consent procedures, including participants’ right to withdraw and to have recordings excluded from analysis. All modeling in this paper operates on de-identified, timestampaligned features rather than raw video or audio, and no identifying information is reported at the individual level. Because the underlying constructs concern psychological well-being, we treat this as a modeling and measurement study rather than a diagnostic or evaluative tool, and we do not claim that model predictions are suitable for individual-level clinical or pedagogical judgments about a specific participant. Data access and sharing constraints arising from this consent scope are discussed in Section 10. The right panel of the teaser figure (Figure 1) is an AI-generated illustration and does not depict a real individual.

## 12 Conclusion

This paper set out to see whether a theory-driven, higher-order construct like self-compassion could be consolidated into a workable supervision scheme, aligned across video, audio, and text, and detected with a small, imbalanced corpus. It can, though not evenly: the six-component theory reduces to a three-class scheme without losing its grounding, and a shared preprocessing pipeline puts the three modalities on the same timeline despite their diferent sampling rates and noise, but how well each class is actually detected still tracks how much annotated support it had to begin with, and for the best-supported class, a majority-class baseline is a real competitor to the trained models. Fusion adds a small but real improvement over the best single modality.

The gains here are modest, and that is mostly a function of scale, 51 sessions is not much data for three modalities and imbalanced classes. Self-compassion was one way into this problem; the same approach should carry over to other constructs that are theory-grounded and hard to observe directly. More data, richer fusion, and calibrated evaluation are the obvious next steps, both to strengthen self-compassion detection itself and to test how well this operationalization and modeling approach generalizes to other higher-order constructs and application settings that share the same core dificulty: real, theory-grounded, and only indirectly observable.

## Acknowledgments

This work was supported by the German Federal Ministry of Research, Technology and Space (BMFTR) under grant 16KIS2632K (Project MARSS).

## References

[1] Ghazaleh Bailey, Júlia Halamová, and Viktória Vráblová. 2023. Clients’ Facial Expressions of Self-Compassion, Self-Criticism, and Self-Protection in Emotion-Focused Therapy Videos. International Journal of Environmental Research and Public Health 20, 2. doi:10.3390/ijerph20021129

[2] Ghazaleh Bailey, Júlia Halamová, and Viktória Vráblová. 2024. Acoustic analysis of clients’ expression of self-compassion, self-criticism, and self-protection within emotion focused therapy video sessions. Frontiers in Psychology 15 (2024). doi:10 3389/fpsyg.2024.1363988

[3] Hervé Bredin. 2023. pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe. In Interspeech 2023. 1983–1987. doi:10.21437/Interspeech. 2023-105

[4] Lara Chehayeb, Katarzyna Olszynska, Chirag Bhuvaneshwara, and Dimitra Tsovaltzi. 2025. Efects of a Co-Regulation Model for MR Teacher Training: HRV and Self-Compassion as Indicators of Emotion Regulation. arXiv:2502.15383 [cs.HC] https://arxiv.org/abs/2502.15383

[5] Zebang Cheng, Zhi-Qi Cheng, Jun-Yan He, Jingdong Sun, Kai Wang, Yuxiang Lin, Zheng Lian, Xiaojiang Peng, and Alexander Hauptmann. 2024. Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning. arXiv preprint (2024).

[6] Kyunghyun Cho, Bart van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation. In Proceedings ofEMNLP 2014. 1724–1734. doi:10.3115/v1/D14-1179

[7] Huan Deng, Zhenguo Yang, Tianyong Hao, Qing Li, and Wenyin Liu. 2023. Multimodal Afective Computing With Dense Fusion Transformer for Interand Intra-Modality Interactions. IEEE Transactions on Multimedia 25 (2023), 6575–6587.

[8] Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Eficient Finetuning of Quantized Large Language Models. arXiv preprint (2023). arXiv:2305.14314

[9] Paul Ekman and Erika L. Rosenberg. 1997. What the Face Reveals. Oxford University Press.

[10] Florian Eyben, Klaus R. Scherer, Bjorn W. Schuller, Johan Sundberg, Elisabeth Andre, Carlos Busso, Laurence Y. Devillers, Julien Epps, Petri Laukka, Shrikanth S.

Narayanan, and Khiet P. Truong. 2016. The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Afective Computing. IEEE Transactions on Afective Computing 7, 2 (2016), 190–202. doi:10.1109/tafc.2015.2457417

[11] Florian Eyben, Martin Wöllmer, and Björn Schuller. 2010. Opensmile: The Munich Versatile and Fast Open-Source Audio Feature Extractor. In Proceedings ofthe 18th ACM International Conference on Multimedia. 1459–1462. doi:10.1145/1873951. 1874246

[12] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 Herd of Models. arXiv preprint (2024). arXiv:2407.21783

[13] Mark R. Leary, Eleanor B. Tate, Claire E. Adams, Ashley Batts Allen, and Jessica Hancock. 2007. Self-compassion and reactions to unpleasant self-relevant events: The implications of treating oneself kindly. Journal ofPersonality and Social Psychology 92, 5 (2007), 887–904. doi:10.1037/0022-3514.92.5.887

[14] Yuanyuan Liu, Haoyu Zhang, Yibing Zhan, Zijing Chen, Guanghao Yin, Lin Wei, and Zhe Chen. 2024. Noise-Resistant Multimodal Transformer for Emotion Recognition. International Journal ofComputer Vision 133, 5 (2024), 3020–3040. doi:10.1007/s11263-024-02304-3

[15] Camillo Lugaresi et al. 2019. MediaPipe: A Framework for Building Perception Pipelines. arXiv:1906.08172

[16] Angelica Moè and Idit Katz. 2020. Self-compassionate teachers are more autonomy supportive and structuring whereas self-derogating teachers are more controlling and chaotic: The mediating role of need satisfaction and burnout. Teaching and Teacher Education 96 (2020), 103173. doi:10.1016/j.tate.2020.103173

[17] Kristin Nef. 2003. Self-Compassion: An Alternative Conceptualization of a Healthy Attitude Toward Oneself. Selfand Identity 2, 2 (2003), 85–101. doi:10. 1080/15298860309032

[18] Kristin D. Nef. 2009. Self-Compassion, Self-Esteem, and Well-Being. Social and Personality Psychology Compass 3, 1 (2009), 1–12. doi:10.1111/j.1751-9004.2010. 00330.x

[19] Susan Nolen-Hoeksema, Blair E. Wisco, and Sonja Lyubomirsky. 2008. Rethinking Rumination. Perspectives on Psychological Science 3, 5 (2008), 400–424. doi:10. 1111/j.1745-6924.2008.00088.x

[20] Alec Radford et al. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. OpenAI technical report (2023).

[21] Jihan Ryu, Stephen Heisig, Caroline McLaughlin, Michael Katz, Helen S. Mayberg, and Xiaosi Gu. 2023. A natural language processing approach reveals firstperson pronoun usage and non-fluency as markers of therapeutic alliance in psychotherapy. iScience (2023). doi:10.1016/j.isci.2023.106860

[22] Klaus R. Scherer. 2003. Vocal communication of emotion: A review of research paradigms. Speech Communication 40, 1–2 (2003), 227–256. doi:10.1016/s0167- 6393(02)00084-5

[23] Donald A. Schön. 1983. The Reflective Practitioner. (1983).

[24] Minoo Shayaninasab and Bagher Babaali. 2024. Multi-Modal Emotion Recogni tion by Text, Speech and Video Using Pretrained Transformers.

[25] Clara Strauss, Billie Lever Taylor, Jenny Gu, Willem Kuyken, Ruth Baer, Fergal Jones, and Kate Cavanagh. 2016. What is compassion and how can we measure it? A review of definitions and measures. Clinical Psychology Review 47 (2016), 15–27. doi:10.1016/j.cpr.2016.05.004

[26] Dimitra Tsovaltzi, Raluca Judele, Thomas Puhl, and Armin Weinberger. 2015. Scripts, individual preparation and group awareness support in the service of learning in Facebook: How does CSCL compare to social networking sites? Computers in Human Behavior 53 (2015), 577–592. doi:10.1016/j.chb.2015.04.067

[27] Paul Waligora, Haseeb Aslam, Osama Zeeshan, Soufiane Belharbi, Alessandro Lameiras Koerich, Marco Pedersoli, Simon Bacon, and Eric Granger. 2024. Joint Multimodal Transformer for Emotion Recognition in the Wild. arXiv preprint (2024). arXiv:2403.10488

[28] Peter Wittenburg, Hennie Brugman, Albert Russel, Alex Klassmann, and Han Sloetjes. 2006. ELAN: A Professional Framework for Multimodal Research. Proceedings of the 5th International Conference on Language Resources and Evaluation.

[29] Kaicheng Yang, Hua Xu, and Kai Gao. 2020. CM-BERT: Cross-Modal BERT for Text-Audio Sentiment Analysis. In Proceedings ofthe 28th ACM International Conference on Multimedia. 521–528. doi:10.1145/3394171.3413690

[30] Min-Ling Zhang and Zhi-Hua Zhou. 2014. A Review on Multi-Label Learning Algorithms. IEEE Transactions on Knowledge and Data Engineering 26, 8 (2014), 1819–1837. doi:10.1109/tkde.2013.39