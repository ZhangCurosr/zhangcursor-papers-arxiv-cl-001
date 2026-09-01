# Augmenting Interviewer Judgments of Patient Experience with Automatic Language Analysis

Aowen Shi<sup>1</sup>, Michal Balazia<sup>1</sup>, Danilo Postin<sup>2</sup>, Rene Hurlemann´ <sup>2</sup>

Jan Alexandersson<sup>3</sup>, Franc¸ois Bremond ´ <sup>1</sup>, Philipp Muller ¨ <sup>4</sup>

<sup>1</sup>INRIA Universite C´ ote d’Azur ˆ , Valbonne, France, aowen.shi@inria.fr, michal.balazia@inria.fr, francois.bremond@inria.fr

<sup>2</sup>Carl von Ossietzky University of Oldenburg, Germany, danilo.postin1@uni-oldenburg.de, rene.hurlemann@uni-oldenburg.de

<sup>3</sup>German Research Center for Artificial Intelligence, Saarbrucken, Germany, jan.alexandersson@dfki.de¨

<sup>4</sup>Max Planck Institute for Intelligent Systems, Stuttgart, Germany, phmueller@is.mpg.de

Abstract—Understanding how psychiatric patients subjectively experienced a clinical conversation is important for feedback and alliance-related process monitoring. While interviewers form post-session judgments about patient experience, these judgments do not always match patients’ self-reports. Automatic approaches for predicting perceived interaction quality from conversation have been proposed, but it remains unclear whether such approaches can complement human judgment rather than simply replicate it. To address this gap, we evaluate a cliniciansupport framework in which post-session interviewer ratings are combined with automatic language-based predictions to estimate patient-reported interaction quality in free clinical interviews. We assess this integration across multiple standard model types, including Ridge, SVR, MLP, GRU, and BiLSTM, all trained on sentence embeddings extracted from dyadic transcripts of 107 free conversations between psychiatric patients and interviewers. Our results show that combining interviewer judgments with model predictions through simple averaging yields the strongest overall performance. The interviewer-only baseline reached a Pearson correlation of 0.365. Among fully automatic models, Ridge achieved the strongest Pearson correlation (r = 0.286), while BiLSTM achieved $r \ = \ 0 . 2 7 0$ . The strongest result was obtained by BiLSTM interviewer integration (r = 0.403). Our findings suggest that automatic language analysis and interviewer judgment capture complementary aspects of patient experience and that their combination provides a more accurate approximation of the patient’s own report than either source alone.

Index Terms—clinician support, interaction quality prediction, dyadic language modeling, therapeutic alliance, major depressive disorder

## I. INTRODUCTION

How patients feel during interactions with clinic personnel is crucial for patient satisfaction and therapy outcomes. This holds true across clinical disciplines [1]–[3]. In psychotherapy, the quality of interaction between patient and therapist is conceptualized as the therapeutic alliance [4], [5]. The quality of therapeutic alliance was shown to be a consistent predictor of therapy outcome across all major schools of psychotherapy [6], [7]. Due to its importance, it is crucial for clinic personnel to assess the quality of a patient’s experience. Despite this importance, several studies have shown that it can be difficult for clinical personnel to accurately judge the quality of a patient’s experience from interacting with them, including underestimating perceived alliance and overestimating positive patient emotions [8], [9].

Affective computing holds great promise to provide automatic approaches that can assist in understanding patient experience. Indeed, approaches to estimate conversation quality have been proposed in a variety of scenarios, including group discussions [10], speed dating interactions [11], and in mental health conversations such as counseling and psychotherapy [12]–[14]. With the success of transformer-based language models, text-based approaches have seen increasing success [15], [16]. At the same time, these methods offer advantages with respect to patient comfort and subjective privacy compared to video-based approaches [17]. However, these approaches for estimating conversation quality were only investigated separately from human judgments. As a result, it remains unclear to what extent such approaches might be able to help interviewers in improving their understanding of patient experience.

In our work, we close this gap by investigating whether automatic language-based approaches to interaction quality estimation can be successfully combined with judgments obtained from the patient’s interviewer. We systematically evaluate this integration across five standard model types, all operating on sentence embeddings extracted separately for patient and interviewer streams. The resulting predictions are subsequently averaged with the interviewer’s judgment of patient experience.

We evaluate this framework on 107 free clinical interviews from the German cohort of the MePheSTO corpus [18]. Our results show that interviewer integration (Pearson correlation $r ~ = ~ 0 . 4 0 3 \pm 0 . 0 3 0 )$ consistently outperforms both the interviewer-only baseline $( r = 0 . 3 6 5 )$ and the best fully automatic model $( r = 0 . 2 8 6 \pm 0 . 0 4 4 )$ , demonstrating that the two sources carry complementary. Additional speaker-stream analyses show that the performance of the fully automatic approach is primarily driven by the interviewer-side transcript.

## II. RELATED WORK

## A. Patient-experienced interaction quality and clinician

Many psychotherapy studies have demonstrated the significant clinical importance of how patients experience therapeutic interactions. In particular, the therapeutic alliance is one of the process variables most closely associated with treatment outcomes across psychotherapy models, making patients’ experience of the therapeutic relationship crucial not only at the descriptive level but also at the prognostic level [6], [7]. Research on the breakdown, repair, and feedback mechanisms of the therapeutic alliance further confirms that monitoring patients’ experience of the interaction is crucial for retaining patients, adjusting treatment plans, and improving treatment outcomes [19]–[21].

At the same time, research on the psychotherapy process has shown that patient and therapist perspectives on the patient’s subjective experience and the quality of the therapeutic relationship are related but not interchangeable. For example, Hartmann et al. [8] found that therapists tend to underestimate patients’ perceived sense of therapeutic alliance, while Atzil-Slonim et al. showed that therapists tracked patients’ emotions with some accuracy but still tended to overestimate positive emotions [9]. Research on patient-therapist agreement and disagreement shows that stronger alliance agreement is frequently linked to better symptom trajectories, reduced dropout rates, or better treatment outcomes [22], [23]. Overall, this implies that (1) clinician judgments of the interaction are informative but may misrepresent the patient’s subjective experience, and (2) achieving a higher agreement between clinician judgments and patient subjective experience has the potential to improve treatment. Our study builds on these observations and explores whether automatic language analysis has the potential to improve clinician judgments of subjective patient experience.

## B. Prediction of interaction quality from conversation

The development of automatic approaches for predicting interaction quality has received growing interest in affective computing. Previous research has modeled constructs such as rapport, engagement, and perceived conversation quality from behavioral evidence in natural interactions from a variety of behavior modalities [10], [11], [24].

In the field of psychotherapy, research indicates that spoken interaction contains measurable process signals. Earlier studies used natural language processing techniques to quantify the coding of psychotherapy processes and the assessment of therapist skills. For example, in motivational interviewing, these studies demonstrated that clinically meaningful session qualities can be inferred from lexical content extracted from audio recordings [25]–[27]. Recent work has also explored AI-generated patient simulations for assessing motivational interviewing sessions, further illustrating the growing role of language-based AI methods in psychotherapy process assessment [28]. More directly relevant to our research context, Goldberg et al. [12] showed that machine learning models trained on session linguistic content could predict clientrated therapeutic alliance from psychotherapy recordings, while later work further identified alliance-related language markers in transcripts and developed language-model-based frameworks for inferring working alliance from psychotherapy dialogue [13], [14]. These studies share our general framing of predicting patient-reported relational experience from conversational language, but treat automatic models as standalone systems evaluated against a held-out patient label. A key question they leave open is whether such automatic estimates carry information that is independent of, and therefore complementary to, the judgment of a human observer who was present during the interaction. This is the gap our study directly addresses.

Recent multimodal studies suggest that incorporating audio and video in certain contexts may further improve the predictive performance of therapeutic alliance [29]. However, such approaches introduce additional technical and privacy constraints that limit their applicability in routine clinical settings. Our work therefore focuses on language-based modeling, which offers a more scalable and privacy-preserving starting point for automatic interaction quality analysis.

## III. DATASET AND LABELS

## A. Clinical Interviews from the MePheSTO corpus

The present analysis is based on the German cohort of the MePheSTO corpus [18]. The German portion consists of audio and video recordings of 107 dyadic clinical sessions between psychiatric patients and interviewers. The patient sample consisted of individuals screened with the structured clinical interview for DSM-5 (SCID-5) and meeting criteria for a Major Depressive Episode. Patients were recruited from inpatient services of the Karl-Jaspers-Klinik (Bad Zwischenahn, Germany), in cooperation with the Department of Psychiatry, University of Oldenburg. The interviews were free-format and unstructured: participants could talk about any topic of their choice, and natural, unconstrained speech was recorded. The interviews lasted 47 minutes on average, with durations ranging from 16 to 81 minutes. All sessions were conducted by trained research assistants (Medicine and Psychology graduate students) in a video-mediated setup in which the patient and interviewer were located in separate rooms and interacted remotely. The setup yields separate patient-side and interviewer-side audio recordings rather than a single mixed-channel recording, which enables the speakerstream comparisons reported in this paper.

The patient sample comprises 38 patients (15 female, 23 male) with a mean age of 33.84 years (SD = 13.39, range = 18–62). Baseline depressive symptom severity, assessed with the BDI-II, had a mean of 30.63 (SD = 11.92). Across patients, the number of interview sessions ranged from 1 to $^ { 4 , }$ with a mean of 2.82 interviews per patient. After each interview session, patients and interviewers completed a postinteraction questionnaire to evaluate the interaction. Responses were recorded on visual analog scales (VAS) ranging from 0 to 100. The present study uses three of these post-interaction items as the label of clinical experience.

## B. Interaction quality measure

We operationalize free interview experience using three post-interaction items from the MePheSTO protocol [18]. The MePheSTO post-interaction questionnaire comprises 10 items and draws on the Working Alliance Inventory (WAI) [30] as a conceptual basis for capturing alliance-related aspects within a study design involving up to four free-format, unstructured interviews. The subset analyzed in this study was restricted to the only three items available in parallel patient-report and interviewer-estimate form, allowing us to compare patients self-reported experience with interviewers’ perspective-taking judgments of the same interaction. In addition, these items capture complementary aspects of a supportive interview experience: perceived helpfulness of the conversation, ease of sharing personal information, and perceived mood change after the interaction. Table I summarizes the original item wording and the interpretive notes used in this study.

![](images/7cf0433a22e5a79865a6200ef45f75827968ef75851562737750688a115333b0.jpg)  
Fig. 1. Distributions of the patient-reported overall composite and the observed interviewer overall estimate on the 106 sessions. Dashed lines indicate the mean.

For each session i, the primary target is the overall patientreported score, defined as the mean of three patient-side items

$$
y _ { i } ^ { \mathrm { P } } = \frac { y _ { i } ^ { \mathrm { m o o d } } + y _ { i } ^ { \mathrm { h e l p f u l } } + y _ { i } ^ { \mathrm { p e r s o n a l } } } { 3 }\tag{1}
$$

where y<sup>mood</sup>, $y _ { i } ^ { \mathrm { { h e l p f u l } } }$ , and $y _ { i } ^ { \mathrm { p e r s o n a l } }$ denote the patient-side ratings of ”After this conversation, I feel better,” ”I have the feeling that what we discussed today was helpful,” and ”Today, it was easy for me to share personal information with my conversation partner” respectively. Similarly, $y _ { i } ^ { \mathrm { I } }$ represents the interviewer’s overall evaluation, calculated as the average of three corresponding interviewer-side items. This composite is used as the main target because the three items capture complementary aspects of perceived interview quality: mood improvement, helpfulness, and relational openness. Conceptually, these aspects relate to Bordin’s three dimensions of the therapeutic alliance, bond, goals, and tasks [4]. However, the composite does not fully operationalize therapeutic alliance. This is intentional because the MePheSTO questionnaire was constrained to a small number of post-interaction items and because the recorded sessions were free clinical interviews rather than ongoing psychotherapy sessions. We therefore interpret the composite as a patient-reported interaction-quality target related to therapeutic alliance, rather than as a complete scale for measuring it.

All 107 sessions have complete patient-reported ratings, while one session is missing the interviewer post-interaction questionnaire, leaving 106 sessions used for all analyses. Among these 106 sessions, the mean patient-side composite is $6 9 . 5 ~ \mathrm { ( S D ~ = ~ } ~ I 8 . 9 ,$ range = 24.3–100.0) and the mean interviewer-side composite is $6 6 . I \ ( \mathrm { S D } = I 5 . 9 ,$ , range = 27– 100.0). Figure 1 shows the distributions of $y _ { i } ^ { \mathrm { P } }$ and $y _ { i } ^ { \mathrm { I } }$ across the 106 sessions. Both distributions are left-skewed, with patients reporting slightly higher scores on average than interviewers.

## IV. METHODS

Figure 2 gives an overview of our methodology. For each session, we extract separate language representations for the patient and interviewer streams, train regression models to predict the patient-reported experience, and then combine the resulting predictions with the interviewer’s post-session rating by averaging. Our goal is to systematically evaluate whether the integration of interviewer ratings with automatic prediction models is able to consistently improve the accuracy of estimating patients’ subjective experience.

## A. Preprocessing and dyadic language representation

Separate speaker channels make role-specific transcripts available for both speakers. Patient and interviewer speech streams are transcribed with WhisperX [31]. The transcript text is segmented into fixed overlapping 30-second windows with a 10-second stride. The 30-second window with a 10- second stride was used as a preprocessing choice rather than an optimized hyperparameter. Since full interviews are too long for single-input encoding, windowing converts each conversation into local temporal segments. The overlap reduces sensitivity to window boundaries. Each window is embedded using MPNet base v2, a multilingual sentence transformer [16] that supports over 50 languages, including German. We selected this model because the MePheSTO corpus was recorded in German. Using a multilingual model ensures that the embeddings capture the semantic content of the sessions without requiring language-specific fine-tuning. The sentence transformer yields an aligned 768-dimensional window-level embedding sequence for the patient and interviewer transcripts.

## B. Regression models

We compare two types of regression approaches that differ in how they summarize each speaker stream: pooled models that compress the full stream into a single session-level vector and sequence models that preserve temporal structure before pooling. This distinction is motivated by the question of whether the ordering and dynamics of language within a session carry additional predictive information beyond an aggregate summary. To assess whether any benefit of integration is specific to a particular modelling choice or holds more generally, we evaluate five models: pooled models Ridge, SVR and MLP, and sequence models GRU and BiLSTM.

Given window w in session i of $W _ { i }$ aligned windows, let $\mathbf { e } _ { i w } ^ { P } , \mathbf { e } _ { i w } ^ { I } \in \mathbb { R } ^ { 7 6 8 }$ denote the patient and interviewer window embeddings, respectively, for the pooled models. Similarly, for the sequence models, we define $\mathbf { h } _ { i w } ^ { P } , \mathbf { h } _ { i w } ^ { I } \in \mathbb { R } ^ { 7 6 8 }$ as the patient-side and interviewer-side encoded window-level hidden representations obtained after the shared window encoder and the selected sequence encoder.

TABLE I  
POST-INTERACTION QUESTIONNAIRE ITEMS ANALYZED IN THIS STUDY.
<table><tr><td>Target</td><td>Patient wording</td><td>Interviewer wording</td><td>Interpretive note</td></tr><tr><td>Mood</td><td>After this conversation, I feel better.</td><td>I have the feeling that my conversation partner feels better after our conversation.</td><td>Perceived mood-related impact of the conversation.</td></tr><tr><td>Helpfulness</td><td>I have the feeling that what we discussed today was helpful.</td><td>I have the feeling that what we discussed today helped my conversation partner.</td><td>Perceived helpfulness or benefit of the conversation.</td></tr><tr><td>Personal sharing</td><td>Today, it was easy for me to share personal information with my conversation partner.</td><td>I have the feeling that it was easy today for my conversation partner to share personal informa- tion with me.</td><td>Relational openness and enabled personal disclosure.</td></tr></table>

![](images/a37417daa78e0142625cf95fad358722b373507d05d79ebabf8be5c428b874ff.jpg)  
Fig. 2. Overview of our approach to predict patient experience by incorporating automatic language analysis with interviewer ratings. Stage 1 consists of three sequential preprocessing substeps: transcription, windowing, and embedding generation.

Pooled models. For Ridge and SVR, each speaker stream is mean-pooled directly from the window embeddings:

$$
\mathbf { z } _ { i } ^ { \mathrm { p o o l } } = \left[ \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \mathbf { e } _ { i w } ^ { P } ; \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \mathbf { e } _ { i w } ^ { I } \right]\tag{2}
$$

For the MLP, a shared window encoder $\phi ( \cdot )$ is first applied to each window embedding, and the encoded features are then mean-pooled within each stream:

$$
\mathbf { z } _ { i } ^ { \mathrm { { m l p } } } = \left[ \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \phi \big ( \mathbf { e } _ { i w } ^ { P } \big ) ; \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \phi \big ( \mathbf { e } _ { i w } ^ { I } \big ) \right]\tag{3}
$$

Sequence models. For sequence models, we preserve the ordered window embeddings for each speaker; patient and interviewer sequences are encoded separately, pooled with masked mean pooling, concatenated by late fusion, and mapped to a scalar prediction with a regression head:

$$
\mathbf { z } _ { i } ^ { \mathrm { s e q } } = \left[ \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \mathbf { h } _ { i w } ^ { P } ; \frac { 1 } { W _ { i } } \sum _ { w = 1 } ^ { W _ { i } } \mathbf { h } _ { i w } ^ { I } \right]\tag{4}
$$

## C. Prediction settings

We investigate three different ways to utilize interviewer intuition and automatic model predictions.

Raw interviewer. The interviewer rating y<sup>I</sup> is used directly as a proxy for the patient’s report, serving as the humanjudgment baseline.

Fully automatic. Depending on the model family, the fully automatic prediction is obtained from one of the session-level representations

$$
\hat { y } _ { i } ^ { \mathrm { t e x t } } = \left\{ \begin{array} { l l } { f ^ { \mathrm { p o o l } } ( \mathbf { z } _ { i } ^ { \mathrm { p o o l } } ) } & { \mathrm { f o r ~ R i d g e ~ a n d ~ S V R } } \\ { f ^ { \mathrm { m l p } } ( \mathbf { z } _ { i } ^ { \mathrm { m l p } } ) } & { \mathrm { f o r ~ M L P } } \\ { f ^ { \mathrm { s e q } } ( \mathbf { z } _ { i } ^ { \mathrm { s e q } } ) } & { \mathrm { f o r ~ G R U ~ a n d ~ B i L S T M } } \end{array} \right.\tag{5}
$$

where $f ^ { \mathrm { p o o l } } , \ : f ^ { \mathrm { m l p } }$ $f ^ { \mathrm { s e q } }$ denote the family-specific regressors applied to the corresponding representations $\mathbf { z } _ { i }$

Interviewer integration. The interviewer integration is combined with the raw interviewer rating using a fixed arith-

metic mean:

$$
\hat { y } _ { i } ^ { \mathrm { a v g } } = \frac { \hat { y } _ { i } ^ { \mathrm { t e x t } } + y _ { i } ^ { I } } { 2 }\tag{6}
$$

This combination requires no additional learned parameters and is fully transparent. The fused prediction is computed on held-out sessions after the fully automatic model has been trained on the corresponding training folds, so no information from the test fold is used in the combination.

## V. EVALUATION

## A. Training setup and evaluation metrics

All evaluations are conducted on the 106 sessions that include complete patient and interviewer ratings, ensuring that the interviewer baseline, fully automatic models, and interviewer integration results are evaluated on a directly comparable set.

a) Cross-validation protocol: We adopt participant-level nested GroupKFold (5 outer folds and 4 inner folds) so that outer test folds contain only unseen participants. This is important because sessions from the same participant can share lexical habits, recurring topics, and reporting tendencies. Hyperparameter selection is confined to the inner training folds, and all reported predictions are obtained from held-out outer-fold evaluations.

b) Hyperparameter search: Search spaces are summarized in Table II. The final search spaces were informed by preliminary development-stage runs and fixed before the reported nested-CV experiments. These pilot runs were used only to rule out clearly unsuitable configurations and to center the search around stable operating regions; all reported model selection and performance estimates were obtained within the nested cross-validation protocol. Neural models were trained using AdamW, early stopping, and up to 80 epochs.

c) Variance estimation: To obtain stable performance estimates, we repeated the nested-CV procedure over 40 randomized runs for all models. For each run, we used a different seed to create the participant-level cross-validation splits. For neural models, we additionally varied the network initialization seed.

d) Metrics: We report Pearson r, and mean absolute error (MAE). Pearson r is the primary metric as it captures how well predictions recover the ordering of sessions, while MAE provides a complementary summary of absolute predictive error. They are widely used in prior work on automatic therapeutic alliance prediction [12], [13]. We report the mean and standard deviation of held-out outerfold metrics across the 40 randomized runs. For the main Pearson-correlation comparisons, we additionally report runlevel bootstrap 95% confidence intervals for the mean Pearson r and paired Wilcoxon signed-rank tests across runs.

## B. Overall Results

Table III summarizes the main results on the overall patientreported clinical interview experience prediction. The raw interviewer rating provided a competitive human-judgment baseline $( r = 0 . 3 6 5 , \mathrm { M A E } = 1 5 . 7 1 7 )$ ). Fully automatic models also achieved meaningful predictive performance, with the best fully automatic result obtained by the Ridge model $( r = 0 . 2 8 6 \pm 0 . 0 4 4 )$ . Among fully automatic models, SVR achieved the lowest MAE $( 1 5 . 1 2 5 \pm 0 . 4 6 6 )$ , while the BiL-STM obtained $r = 0 . 2 7 0 \pm 0 . 0 8 4$ and $\mathrm { M A E } = 1 5 . 2 5 6 \pm 0 . 5 9 8$

The strongest overall performance was obtained by the interviewer-integration setting, represented by the prediction ${ \hat { y } } _ { i } ^ { \mathrm { a v g } } .$ . The BiLSTM interviewer-integration model achieved the highest Pearson correlation $( r ~ = ~ 0 . 4 0 3 \pm 0 . 0 3 0 ;$ runlevel bootstrap 95% CI [0.393, 0.412]) and the lowest MAE $( 1 4 . 2 1 4 { \pm } 0 . 3 1 2 )$ . Across model types, the integrated prediction $\hat { y } _ { i } ^ { \mathrm { a v g } }$ consistently outperformed the raw interviewer baseline and improved over the corresponding fully automatic variants. Paired Wilcoxon signed-rank tests on Pearson r confirmed the key comparisons: interviewer integration improved over interviewer-only by $\Delta r = + 0 . 0 3 7 8 , p = 3 . 9 2 \times 1 0 ^ { - 9 }$ , and over fully automatic BiLSTM by $\Delta r = + 0 . 1 3 3 0 , p = 9 . 0 9 { \times } 1 0 ^ { - 1 3 }$ Overall, these results indicate that interviewer judgment and language-based prediction can be combined productively. To contextualize MAE, we evaluated a trivial training-mean baseline, which obtained $\mathbf { M A E } = 1 6 . 1 1 3 \pm 0 . 2 0 3$ across the 40 randomized runs. Compared with the patient-side composite SD of 18.9, the MAE reduction of BiLSTM interviewer integration $( 1 4 . 2 1 4 \pm 0 . 3 1 2 )$ is modest on the 0–100 scale, but improves upon both the mean predictor and intervieweronly baseline.

Figure 3 shows scatter plots of predicted values versus patient-reported rating for interviewer-only predictions, BiL-STM fully automatic predictions, and BiLSTM interviewer integration. The interviewer panel (left) contains one point per matched session, whereas the BiLSTM and integration panels (middle and right) show held-out predictions from each of the 40 randomized runs. Therefore, each session appears 40 times in those two panels. The interviewer plot shows a moderate positive association with substantial scatter around the identity line, consistent with the competitive but imperfect correlation of $r = 0 . 3 6 5$ . The fitted line is shallower than the identity line, indicating that interviewer ratings tend to be compressed toward the middle of the scale. The BiLSTM fully automatic plot (middle) also shows a positive trend $( r = 0 . 2 7 0 \pm 0 . 0 8 4 )$ but its predictions occupy an even narrower vertical range than the interviewer ratings. The corresponding regression line is therefore shallower, indicating stronger range compression and an underestimation of session-to-session variability in patientreported ratings. The BiLSTM interviewer integration plot (right) shows improved agreement overall $( r = 0 . 4 0 3 { \pm } 0 . 0 3 0 )$ with predictions shifted closer to the identity line than in the fully automatic case. However, the fitted regression line remains shallower than the identity line, so the integrated prediction still exhibits range compression even though its overall correlation and error are improved.

## C. Speaker Ablation

To analyze where the predictive signal resides within the conversation, we evaluate three input configurations for the language-based models: patient-only, which uses only the patient embedding stream; interviewer-only, which uses only the interviewer embedding stream; and dual-stream, which combines both streams by late fusion. Table IV compares patient-only, interviewer-only, and dual-stream inputs in the fully automatic setting for Ridge and BiLSTM, which were the best-performing pooled and sequence models in Table III.

HYPERPARAMETER SEARCH SPACES USED WITHIN THE INNER FOLDS OF NESTED GROUPKFOLD.
<table><tr><td>Model</td><td>Searched hyperparameters</td></tr><tr><td>Ridge</td><td> $\alpha \in \{ 0 . 0 1 , 0 . 1 , 1 , 1 0 , 1 0 0 \}$ </td></tr><tr><td>SVR</td><td> $C \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 \} ; \gamma \in \{ \mathrm { s c a l e } , 0 . 0 0 1 , 0 . 0 1 , 0 . 1 \} ; \epsilon \in \{ 0 . 0 1 , 0 . 1 , 0 . 5 \}$ </td></tr><tr><td>MLP</td><td>hidden dim  $\in \{ 2 5 6 , 5 1 2 \}$  ; learning rate  $\in \{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \} ;$  dropout  $\in \{ 0 , 0 . 1 \}$  weight decay  $\in \{ 0 , 1 0 ^ { - 4 } \}$ </td></tr><tr><td>GRU and BiLSTM</td><td>recurrent hidden size ∈ {64, 128}; learning rate  $\in \{ 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } \} ;$  dropout ∈ {0, 0.1}; weight decay  $\in \{ 0 , 1 0 ^ { - 4 } \}$ </td></tr></table>

![](images/36e645e77c418ffdc36801c4923c69ecba80b5fac403f83817533cab7dae32dd.jpg)

![](images/324db909407d519acfdb736a82cbabf94ded32b30d118a1ad8653dc7311131e6.jpg)

![](images/04b2ccfd52491e057cf6baac3236deabfd57e71a39b80d49560073b0b69b262d.jpg)  
Fig. 3. Session-level scatter plots for the matched held-out overall comparisons. Left: interviewer ratings. Middle: BiLSTM fully automatic prediction. Right: BiLSTM interviewer integration. The dashed line denotes the identity line, and the solid line denotes the fitted regression line.

TABLE III  
PERFORMANCE ON THE OVERALL PATIENT-REPORTED CLINICAL INTERVIEW EXPERIENCE PREDICTION. VALUES AFTER ± INDICATE STANDARD DEVIATION ACROSS 40 RUNS.
<table><tr><td>Prediction method</td><td>Pearson r</td><td>MAE</td></tr><tr><td>Interviewer</td><td>0.365</td><td>15.717</td></tr><tr><td>Fully Automatic</td><td></td><td></td></tr><tr><td>Ridge</td><td> ${ \bf 0 . 2 8 6 \pm 0 . 0 4 4 }$ </td><td> $1 6 . 7 1 7 \pm 0 . 7 1 4$ </td></tr><tr><td>SVR</td><td> $0 . 2 4 0 \pm 0 . 0 7 8$ </td><td> $\mathbf { 1 5 . 1 2 5 \ : \pm { \ : 0 . 4 6 6 } }$ </td></tr><tr><td>MLP</td><td> $0 . 1 9 7 \pm 0 . 1 0 6$ </td><td> $1 5 . 7 4 6 \pm 0 . 7 7 3$ </td></tr><tr><td>GRU</td><td> $0 . 2 3 7 \pm 0 . 1 1 1$ </td><td> $1 5 . 4 0 5 \pm 0 . 6 1 1$ </td></tr><tr><td>BiLSTM</td><td> $0 . 2 7 0 \pm 0 . 0 8 4$ </td><td> $1 5 . 2 5 6 \pm 0 . 5 9 8$ </td></tr><tr><td>Interviewer Integration</td><td></td><td></td></tr><tr><td>Ridge</td><td> $0 . 3 9 0 \pm 0 . 0 2 5$ </td><td> $1 4 . 6 6 9 \pm 0 . 3 5 9$ </td></tr><tr><td>SVR</td><td> $0 . 3 8 9 \pm 0 . 0 2 3$ </td><td> $1 4 . 3 1 0 \pm 0 . 2 2 1$ </td></tr><tr><td>MLP</td><td> $0 . 3 7 6 \pm 0 . 0 3 3$ </td><td> $1 4 . 5 0 7 \pm 0 . 3 3 9$ </td></tr><tr><td>GRU</td><td> $0 . 3 9 5 \pm 0 . 0 3 6$ </td><td> $1 4 . 2 9 5 \pm 0 . 3 2 3$ </td></tr><tr><td>BiLSTM</td><td> $\mathbf { 0 . 4 0 3 \ : \pm 0 . 0 3 0 }$ </td><td> ${ \bf 1 4 . 2 1 4 \pm 0 . 3 1 2 }$ </td></tr></table>

Across both model types, patient-only models were the weakest, while interviewer-only inputs achieved the strongest Pearson correlations. For Ridge, interviewer-only reached $r ~ = ~ 0 . 2 9 8 \pm 0 . 0 4 0$ , compared with $r ~ = ~ 0 . 2 8 6 \pm 0 . 0 4 4$ for dual-stream and $r ~ = ~ 0 . 1 9 2 \pm 0 . 0 6 9$ for patient-only. For BiLSTM, interviewer-only reached $r = 0 . 2 8 5 \pm 0 . 1 0 6$ compared with $r ~ = ~ 0 . 2 7 0 \pm 0 . 0 8 4$ for dual-stream and $r = 0 . 0 7 9 \pm 0 . 1 1 9$ for patient-only. MAE showed a similar pattern for Ridge, while the BiLSTM dual-stream model achieved slightly lower MAE than interviewer-only. Overall, these results suggest that interviewer-side language carries the most readily extractable predictive signal in the fully automatic setting, whereas patient-side language alone is less predictive under the current modeling setup.

TABLE IV  
COMPARISON OF PATIENT-ONLY, INTERVIEWER-ONLY, AND DUAL-STREAM INPUTS FOR RIDGE AND BILSTM IN THE FULLY AUTOMATIC SETTING. VALUES AFTER ± INDICATE STANDARD DEVIATION ACROSS 40 RUNS.
<table><tr><td>Input streams</td><td>Pearson r</td><td>MAE</td></tr><tr><td>Ridge</td><td></td><td></td></tr><tr><td>Dual stream</td><td> $0 . 2 8 6 \pm 0 . 0 4 4$ </td><td> $1 6 . 7 1 7 \pm 0 . 7 1 4$ </td></tr><tr><td>Interviewer-only</td><td> $\mathbf { 0 . 2 9 8 \ : \pm 0 . 0 4 0 }$ </td><td> $\mathbf { 1 6 . 5 1 9 \ : \pm { \ : 0 . 8 3 0 } }$ </td></tr><tr><td>Patient-only</td><td> $0 . 1 9 2 \pm 0 . 0 6 9$ </td><td> $1 7 . 0 9 8 \pm 0 . 9 3 5$ </td></tr><tr><td>BiLSTM</td><td></td><td></td></tr><tr><td>Dual stream</td><td> $0 . 2 7 0 \pm 0 . 0 8 4$ </td><td> ${ \bf 1 5 . 2 5 6 \pm 0 . 5 9 8 }$ </td></tr><tr><td>Interviewer-only</td><td> ${ \bf 0 . 2 8 5 \pm 0 . 1 0 6 }$ </td><td> $1 5 . 3 2 3 \pm 0 . 8 6 0$ </td></tr><tr><td>Patient-only</td><td> $0 . 0 7 9 \pm 0 . 1 1 9$ </td><td> $1 6 . 4 9 0 \pm 1 . 0 0 3$ </td></tr></table>

## D. Pooling Strategy Ablation

For the BiLSTM model, we additionally compare three pooling strategies: masked mean pooling, attention pooling, and joint attention pooling. Table V shows that masked mean pooling consistently matched or outperformed both attention pooling and joint attention pooling in both the fully automatic and the interviewer integration setting. As the simpler and slightly better performing strategy, masked mean pooling is therefore used in all other reported BiLSTM results.

TABLE V  
ABLATION OF POOLING USING BILSTM. VALUES AFTER ± INDICATE STANDARD DEVIATION ACROSS 40 RUNS.
<table><tr><td>Pooling</td><td>Pearson r</td><td>MAE</td></tr><tr><td>Fully Automatic</td><td></td><td></td></tr><tr><td>Masked mean pooling</td><td> ${ \bf 0 . 2 7 0 \pm 0 . 0 8 4 }$ </td><td> ${ \bf 1 5 . 2 5 6 \pm 0 . 5 9 8 }$ </td></tr><tr><td>Attention pooling</td><td> $0 . 2 5 4 \pm 0 . 0 9 9$ </td><td> $1 5 . 4 9 0 \pm 0 . 7 2 8$ </td></tr><tr><td>Joint attention</td><td> $0 . 1 8 5 \pm 0 . 1 1 6$ </td><td> $1 5 . 9 1 6 \pm 0 . 8 3 2$ </td></tr><tr><td>Interviewer Integration</td><td></td><td></td></tr><tr><td>Masked mean pooling</td><td> $\mathbf { 0 . 4 0 3 \ : \pm { \ : 0 . 0 3 0 } }$ </td><td> $\mathbf { 1 4 . 2 1 4 \ : \pm { \ : 0 . 3 1 2 } }$ </td></tr><tr><td>Attention pooling</td><td> $0 . 3 9 9 \pm 0 . 0 3 4$ </td><td> $1 4 . 2 3 9 \pm 0 . 3 7 3$ </td></tr><tr><td>Joint attention</td><td> $0 . 3 7 3 \pm 0 . 0 4 0$ </td><td> $1 4 . 5 5 5 \pm 0 . 4 1 4$ </td></tr></table>

## VI. DISCUSSION AND LIMITATIONS

Main findings. Our results support three main conclusions. First, patient-reported clinical interview experience can be estimated from the language used during the interaction, as fully automatic models achieved meaningful predictive performance on the overall target. Second, interviewer ratings provided a competitive human-judgment baseline, but the strongest overall results were obtained when interviewer ratings were combined with language-based predictions. Third, within the fully automatic setting, interviewer-side language was more informative than patient-side language in terms of Pearson correlation, while dual-stream inputs did not consistently outperform interviewer-only inputs.

The consistent benefit of integration across all five model types is the central finding of this study. It demonstrates that the benefit is not an artifact of a particular architectural choice, but reflects a genuine complementarity between the two information sources. Interviewer ratings reflect a postsession, in-context human evaluation, whereas automatic predictions derive their estimates from distributional patterns in the transcript. The performance gains from combining these sources therefore indicate that they capture different, rather than redundant, aspects of the patient’s experience.

The speaker-stream ablation offers a more fine-grained account of where this signal originates. The finding that interviewer-side language is more informative than patientside language suggests that supportive session experience is partly constituted by interviewer behaviour, including how the interviewer structures the session, responds to disclosures, and creates space for the patient to speak. Patient language, by contrast, may be more strongly shaped by symptom severity, idiosyncratic disclosure style, and topic selection, making it a noisier signal with respect to patient-reported experience. Dual-stream inputs remained competitive, but did not consistently improve over interviewer-only inputs under the present modeling setup. The two streams are therefore asymmetric in informativeness: interviewer-side language appears to contain the strongest readily extractable signal for this task, while patient-side language alone is less predictive.

These findings extend previous psychotherapy research that establishes that patient and clinician perspectives are related but not interchangeable [8], [9] to the computational domain, showing that automatic language-based predictions carry signal partly independent of interviewer evaluation. Rather than replacing human judgment, automatic prediction appears most useful as an additional evidence source, particularly when the goal is to approximate the patient’s perspective. The simplicity of the integration strategy, a fixed arithmetic average requiring no additional training, makes this approach practically deployable.

Limitations. Several limitations should be noted. First, the dataset is small (107 sessions, 106 with interviewer ratings), limiting the stability of model comparisons and the generalisability of conclusions. Replication across larger, more diverse corpora covering different clinical settings, patient populations, and languages is necessary before broader claims can be made. Secondly, while these three questionnaire items captured complementary aspects of patient-reported interaction quality, they should not be interpreted as a complete operationalization of the therapeutic alliance, nor should our findings be directly equated with results obtained in an ongoing psychotherapy setting. Third, we deliberately focused on textbased modelling for reasons of privacy, scalability, and deployment feasibility; the present results therefore characterise what can be recovered from transcript-based representations rather than establishing an upper bound on multimodal approaches. Fourth, the arithmetic mean fusion may underestimate the gains achievable with learned combination strategies, which future work with larger samples should explore. Finally, the finding that interviewer-side language was more predictive than patient-side language should not be interpreted causally; it reflects a more readily extractable signal under the current data and modelling setup rather than evidence that interviewer behaviour alone determines patient experience.

## VII. CONCLUSION

We investigated whether language-based automated predictions could combine patient-reported interaction quality predictions with post-interview interviewer ratings in freeflowing clinical interviews. We evaluated this integration approach across five standard models using 107 interview transcripts from the MePheSTO corpus. The results showed that a simple average of automated predictions and interviewer ratings consistently outperformed either source alone. This consistent advantage suggests that automatic language analysis and interviewer judgment provide complementary predictive information.

The simplicity and consistency of our integration strategy suggest that it can be used in a deployable framework in which automated language analysis tools can enhance, rather than replace, post-clinical assessments.

## ETHICAL IMPACT STATEMENT

This study analyzes transcript-based representations of psychiatric interviews from the German MePheSTO corpus to estimate patients’ subjective experience of a clinical conversation. The study was approved by the medical ethics committee of the Carl-von-Ossietzky University of Oldenburg (approval number 2021-108) and conducted in accordance with the latest revision of the Declaration of Helsinki. Participants provided written informed consent after receiving a complete description of the study. The data are confidential due to the sensitivity of psychiatric information and the potential risks of privacy breaches. Accordingly, this work is limited to research use on de-identified conversational transcripts and does not involve automated clinical decision-making or deployment in patient care. We stress that the models evaluated here are intended to support research on interaction quality and should not replace clinician judgment or patients’ own reports. Any future clinical application would require further validation, clear governance and safeguards against harmful or inappropriate use. The generalizability of our findings may be limited, as we only studied patients from a single clinic in Germany who had a Major Depressive Episode.

## ACKNOWLEDGMENT

This research was in parts funded by the French National Research Agency ANR under the $\mathrm { U C A } ^ { \mathrm { J E D I } }$ Investments into the Future (project number ANR-15-IDEX-01) and by the German Ministry for Education and Research BMBF (grant number 01IS20075).

## REFERENCES

[1] M. A. Stewart, “Effective physician-patient communication and health outcomes: a review,” CMAJ: Canadian medical association journal, vol. 152, no. 9, p. 1423, 1995.

[2] J. N. Fuertes, A. Mislowack, J. Bennett, L. Paul, T. C. Gilbert, G. Fontan, and L. S. Boylan, “The physician–patient working alliance,” Patient education and counseling, vol. 66, no. 1, pp. 29–36, 2007.

[3] S. Priebe and R. Mccabe, “Therapeutic relationships in psychiatry: the basis of therapy or therapy in itself?” International Review ofPsychiatry, vol. 20, no. 6, pp. 521–526, 2008.

[4] E. S. Bordin, “The generalizability of the psychoanalytic concept of the working alliance.” Psychotherapy: Theory, research & practice, vol. 16, no. 3, p. 252, 1979.

[5] A. O. Horvath, A. Del Re, C. Fluckiger, and D. Symonds, “Alliance in¨ individual psychotherapy.” Psychotherapy, vol. 48, no. 1, p. 9, 2011.

[6] C. Fluckiger, A. C. Del Re, B. E. Wampold, and A. O. Horvath, “The¨ alliance in adult psychotherapy: A meta-analytic synthesis.” Psychotherapy, vol. 55, no. 4, p. 316, 2018.

[7] C. Fluckiger, A. Del Re, D. Wlodasch, A. O. Horvath, N. Solomonov,¨ and B. E. Wampold, “Assessing the alliance–outcome association adjusted for patient characteristics and treatment processes: A metaanalytic summary of direct comparisons.” Journal of Counseling Psychology, vol. 67, no. 6, p. 706, 2020.

[8] A. Hartmann, A. Joos, D. E. Orlinsky, and A. Zeeck, “Accuracy of therapist perceptions of patients’ alliance: Exploring the divergence,” Psychotherapy Research, vol. 25, no. 4, pp. 408–419, 2015.

[9] D. Atzil-Slonim, E. Bar-Kalifa, H. Fisher, G. Lazarus, I. Hasson-Ohayon, W. Lutz, J. Rubel, and E. Rafaeli, “Therapists’ empathic accuracy toward their clients’ emotions.” Journal of Consulting and Clinical Psychology, vol. 87, no. 1, p. 33, 2019.

[10] P. Muller, M. X. Huang, and A. Bulling, “Detecting low rapport during¨ natural interactions in small groups from non-verbal behaviour,” in Proceedings of the 23rd International Conference on Intelligent User Interfaces, 2018, pp. 153–164.

[11] J. Vargas-Quiros, O. Kapcak, H. Hung, and L. Cabrera-Quiros, “In-<sup>¨</sup> dividual and joint body movement assessed by wearable sensing as a predictor of attraction in speed dates,” IEEE Transactions on Affective Computing, vol. 14, no. 3, pp. 2168–2181, 2021.

[12] S. B. Goldberg, N. Flemotomos, V. R. Martinez, M. Tanana et al., “Machine learning and natural language processing in psychotherapy research: Alliance as example use case,” Journal of Counseling Psychology, vol. 67, no. 4, pp. 438–448, 2020.

[13] B. Lin, D. Bouneffouf, Y. Landa, R. Jespersen, C. Corcoran, and G. Cecchi, “Compass: Computational mapping of patient-therapist alliance strategies with language modeling,” Translational Psychiatry, vol. 15, no. 1, p. 166, 2025.

[14] J. Ryu, S. Heisig, C. McLaughlin, M. Katz, H. S. Mayberg, and X. Gu, “A natural language processing approach reveals first-person pronoun usage and non-fluency as markers of therapeutic alliance in psychotherapy,” IScience, vol. 26, no. 6, 2023.

[15] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the associationfor computational linguistics: human language technologies, volume 1 (long and short papers), 2019, pp. 4171–4186.

[16] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[17] J. Torous and L. W. Roberts, “The ethical use of mobile health technology in clinical psychiatry,” The Journal of nervous and mental disease, vol. 205, no. 1, pp. 4–8, 2017.

[18] A. Konig, P. M¨ uller, J. Tr¨ oger, H. Lindsay, J. Alexandersson, J. Hinze,¨ M. Riemenschneider, D. Postin, E. Ettore, A. Lecomte et al., “Multimodal phenotyping of psychiatric disorders from social interaction: Protocol of a clinical multicenter prospective study,” Personalized Medicine in Psychiatry, vol. 33, p. 100094, 2022.

[19] J. D. Safran and J. C. Muran, “Resolving therapeutic alliance ruptures: Diversity and integration,” Journal of clinical psychology, vol. 56, no. 2, pp. 233–243, 2000.

[20] C. F. Eubanks, J. C. Muran, and J. D. Safran, “Alliance rupture repair: A meta-analysis,” Psychotherapy, vol. 55, no. 4, pp. 508–519, 2018.

[21] M. J. Lambert, J. L. Whipple, and M. Kleinstauber, “Collecting and¨ delivering progress feedback: A meta-analysis of routine outcome monitoring,” Psychotherapy, vol. 55, no. 4, pp. 520–537, 2018.

[22] S. Jennissen, C. Nikendei, J. C. Ehrenthal, H. Schauenburg, and U. Dinger, “Influence of patient and therapist agreement and disagreement about their alliance on symptom severity over the course of treatment: A response surface analysis,” Journal of Counseling Psychology, vol. 67, no. 3, pp. 326–336, 2020.

[23] R. Moshe-Cohen, Y. Kivity, J. D. Huppert, D. H. Barlow et al., “Agreement in patient-therapist alliance ratings and its relation to dropout and outcome in a large sample of cognitive behavioral therapy for panic disorder,” Psychotherapy Research, vol. 34, no. 1, 2024.

[24] C. Raman, N. R. Prabhu, and H. Hung, “Perceived conversation quality in spontaneous interactions,” IEEE Transactions on Affective Computing, vol. 14, no. 4, pp. 2901–2912, 2023.

[25] Z. E. Imel, M. Steyvers, and D. C. Atkins, “Computational psychotherapy research: Scaling up the evaluation of patient-provider interactions,” Psychotherapy, vol. 52, no. 1, pp. 19–30, 2015.

[26] Z. E. Imel, B. T. Pace, C. S. Soma, M. Tanana, T. Hirsch, J. Gibson, P. Georgiou, S. Narayanan, and D. C. Atkins, “Design feasibility of an automated, machine-learning based feedback system for motivational interviewing.” Psychotherapy, vol. 56, no. 2, p. 318, 2019.

[27] M. Tanana, K. A. Hallgren, Z. E. Imel, D. C. Atkins, and V. Srikumar, “A comparison of natural language processing methods for automated coding of motivational interviewing,” Journal of substance abuse treatment, vol. 65, pp. 43–50, 2016.

[28] S. Yosef, M. Zisquit, B. Cohen, A. K. Brunstein, K. Bar, and D. Friedman, “Assessing motivational interviewing sessions with ai-generated patient simulations,” in Proceedings of the 9th Workshop on Computational Linguistics and Clinical Psychology, 2024, pp. 1–11.

[29] K. Aafjes-Van Doorn, M. Cicconet, J. F. Cohn, and M. Aafjes, “Predicting working alliance in psychotherapy: A multi-modal machine learning approach,” Psychotherapy Research, vol. 35, no. 2, pp. 256–270, 2025.

[30] A. O. Horvath and L. S. Greenberg, “Development and validation of the working alliance inventory.” Journal of counseling psychology, vol. 36, no. 2, p. 223, 1989.

[31] M. Bain, J. Huh, T. Han, and A. Zisserman, “Whisperx: Timeaccurate speech transcription of long-form audio,” arXiv preprint arXiv:2303.00747, 2023.