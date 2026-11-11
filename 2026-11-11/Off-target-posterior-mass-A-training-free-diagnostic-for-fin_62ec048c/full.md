# Off-target posterior mass: A training-free diagnostic for fine-tuning spoken language diarization probes

Zhuoyan Tao,<sup>a)</sup> Aditya Kommineni, and Shrikanth Narayanan

Signal and Image Processing Institute, Ming Hsieh Department ofElectrical and Computer Engineering, University of Southern California, Los Angeles, California 90089, USA

zhuoyantao@gmail.com, akommine@usc.edu, shri@usc.edu

Abstract:Spoken Language Diarization (SLD) models provide frame-level language information of a speech recording. SLD modeling pipeline includes feature extraction from pre-trained speech models followed by clustering. Fine-tuning pre-trained speech models has shown performance gains for SLD with its magnitude being dataset dependent. Off-target Posterior Mass (OTPM), computed based on posterior language identification probabilities is proposed as a diagnostic metric to forecast performance gains from SLD fine-tuning. OTPM obtains positive correlation with probe gains, 0.56 on unseen synthetic bilingua data and 0.23 on real-world corpora. These results indicate promising potential of OTPM as a diagnostic metric for SLD.

[Editor: —]

https://doi.org/

Date: 11 November 2026

Keywords: Spoken Language Diarization, Speech Foundation Models, Language Identification.

## 1. Introduction

Spoken Language Diarization (SLD) models produce time-stamped segments for distinct languages in a given speech recording.<sup>1,2</sup> SLD models are vital for multilingual speech processing and facilitate downstream processing of utterances in code-switched and code-mixed contexts, that occur often in real-world settings. While SLD is analogous to speaker diarization, it has idiosyncratic challenges: (1) languages within a family share acoustic and phonotactic structure, and code-switching produces overlapping segments with no well-defined acoustic boundary<sup>3</sup> (2) real-world corpora are often imbalanced, with one or two languages contributing a large portion of speech and the rest appearing as short, sparse turns.<sup>4,5</sup>

Most SLD systems comprise two stage pipelines.<sup>2,6,7</sup> Embedding/posterior Language Identification (LID) vector extraction from a pre-trained speech model (such as Whisper<sup>8</sup> or MMS<sup>9</sup>) for overlapping speech segments of a speech recording. Following this, the embeddings/posterior LID vectors are clustered (often using Agglomerative Hierarchical Clustering<sup>1,10,11</sup>) into the underlying languages in the recording. Other methods for SLD include change-point detection methods<sup>12</sup> with attention and temporal smoothing, supervised frame-level classification<sup>3</sup> and End-to-End systems such as SAGE-LD.<sup>5,6,13</sup> Evaluation of SLD models for the generated clusters are performed with language diarization error rate (LDER) metric, computed similar to speaker diarization error rate with language labels in place of speaker labels (Eq. (1)). Fine-tuning the pre-trained model embeddings/posterior LID vectors has been found to be an effective method for improving SLD performance in prior works.<sup>7</sup> Here, we will refer to this fine-tuning as training a probe and the difference in LDER between the pre-trained embedding and fine-tuned embedding clustering is referred to as probe gain.

While majority of prior work in the domain of SLD focuses on improving absolute diarization performance, in this work we probe whether it would be feasible to forecast the probe gain obtained from fine-tuning with a collected corpora, and we would want this forecasting to be in a training-free manner. Being able to predict whether a given corpus would provide noticeable improvements on SLD task for a pre-trained speech model is valuable indicator for data curation without requiring explicit training, and would mark as an indicator of whether annotations should be performed. Prior works in the domain of vision and language have explored similar questions, in the context of identifying pre-trained embedding quality through metrics such as NCE,<sup>14</sup> H-score,<sup>15</sup> LEEP<sup>16</sup> and LogME,<sup>17</sup> which score a pre-trained model’s outputs or features against the labelled target data, typically to rank candidate pre-trained models. Alternatively, label-free accuracy prediction methods do not require labelled data, such as in the case of AutoEval by regressing accuracy on distribution shift across synthetic test sets,<sup>18</sup> and Average Threshold Confidence<sup>19</sup> by learning a threshold for model confidence. Our proposed Off-Target Posterior Mass (OTPM) metric is in a training-free and label-free accuracy prediction paradigm, and uses the posterior LID probabilities from pre-trained speech models to compute proportion of probability assigned to non-target languages for the SLD task. We show that this metric is correlated to probe gain obtained for SLD task on different synthetically generated and real-world datasets. The contributions of this work are as follows:

• We propose Off-Target Posterior Mass (OTPM) as a metric for forecasting the probe gains obtained for Spoken Language Diarization task on speech foundation models.

• On an unseen set of 19 synthetically generated bilingual language pair data, OTPM obtains a Spearman correlation of 0.56 and 0.69 with probe gains for Whisper base and large-v3 models respectively.

• When translating to real-world corpora, while OTPM is able to better align with probe gain compared to baselines, the correlation is smaller with 0.23 and 0.27 for Whisper base and large-v3 models.

## 2. Methodology

![](images/c8a1ffdb5e3a8397e31b58f483952107e8791602328251a66d2d877aa3510190.jpg)

![](images/af6d395d204f1b217a9cfe769e6a93a19905d1dec687021a0c1903065686dd64.jpg)  
Fig. 1. Off-target LID posterior mass (OTPM) as a training-free diagnostic for SLD probe gain. (a) Each short window of the corpus audio is passed through a pre-trained speech model, which produces a multilingual LID posterior. (b) Posterior mass for Target languages (T marked in red) carry most of the mass in majority of cases; OTPM is the fraction assigned to non-target languages $( \mathcal { L } \setminus \mathcal { T }$ marked in gray), averaged over windows. (c) Illustration of the ordinary least squares (OLS) fit between measured OTPM and probe gain. Based on the OTPM of an unseen language pair, we can predict the probe gain as shown with the orange marker (input OTPM of 0.3 yields a predicted gain $\hat { \Delta }$ of ∼6.9 pt).

## 2.1 Spoken language diarization

Given a speech recording and a set $\tau$ of target languages for which we want to build the SLD system, the objective is to output time-stamped segments, each labelled with one of the languages in T. A cascaded SLD system is composed of three stages: (1) voice activity detection (VAD) to identify speech regions, (2) segmentation of speech utterances into short overlapping windows followed by feature extraction from a pre-trained speech model (often a language identification posterior $p _ { w }$ as shown in Fig. 1(a)), and (3) clusters the windows to form the corresponding language clusters. Given labels for each cluster, evaluation is performed using Language Diarization Error Rate (LDER) metric.

$$
\mathrm { L D E R } = \frac { T _ { \mathrm { m i s s } } + T _ { \mathrm { F A } } + T _ { \mathrm { c o n f } } } { T _ { \mathrm { r e f } } } ,\tag{1}
$$

LDER is analogous to Diarization Error Rate in speaker diarization context which is computed as the summation of missed $( T _ { \mathrm { m i s s } } )$ , falsely detected $( T _ { \mathrm { F A } } )$ and wrongly labelled $( T _ { \mathrm { c o n f } } )$ speech time divided by the total reference speech time.<sup>20</sup> In addition to LDER, we observe that for some multilingual speech recordings the output of the clustering model provides a single language label indicating that the model is not able to diarize the recording. We refer to this phenomenon as collapse and report these values alongside LDER.

## 2.2 Off-target LID posterior mass

Let the language identification posterior $p _ { w } \in \mathbf { R } ^ { | \mathcal { L } | }$ of the pre-trained model on a window w support a fixed set of L languages (for instance, Whisper supports 99 languages). Target languages $\tau \subseteq { \mathcal { L } }$ are the languages the SLD model must diarize, fixed by the downstream task; non-target languages form the complement ${ \mathcal { L } } \backslash \tau$ , i.e., every other language the model supports (red and gray bars, respectively, in Fig. 1(b)).

Definition. Let the corpus consist of speech utterances $f \in F ,$ and let $W _ { f }$ be the windows of utterance f that is used as input to the clustering stage (Sec. 3). Off-Target Posterior Mass (OTPM) is defined as the posterior mass the pre-trained model assigns to non-target languages, averaged over the windows of each utterance:

$$
\mathrm { O T P M } = \frac { 1 } { | F | } \sum _ { f \in F } \frac { 1 } { | W _ { f } | } \sum _ { w \in W _ { f } } \ : \sum _ { l \in \mathcal { L } \backslash \mathcal { T } } p _ { w } ( l ) ,\tag{2}
$$

where $p _ { w } ( l )$ is the posterior probability the model assigns to language l on window w. For example, when performing SLD with Hindi and English we have $\mathcal { T } = \{ \mathsf { h i } , \mathsf { e n } \}$ , if a window has $p ( \mathsf { h i } ) { = } 0 . 5 5$ and $p ( \mathsf { e n } ) { = } 0 . 0 5$ , then the OTPM for this window would be 0.40 i.e. the posterior mass assigned to non-target languages. OTPM lies in the range [0, 1], with values close to zero indicating that the pre-trained posterior probabilities are assigned entirely to target languages; whereas values close to 1 imply almost all of the posterior probability is assigned to non-target languages $( \mathcal { L } \backslash \mathcal { T } )$ . Based on the definition in Eq. (2), OTPM depends on the downstream corpus, the target set of languages and the pre-trained speech model, but does not need explicit language labels (only requires the set of languages in the corpus) and does not require clustering/fine-tuning. To get an accurate value of OTPM, it must be computed on the windows of the recordings that are representative of the corpus.

## 2.3 Probe gain

SLD performance in a cascaded system can be improved through fine-tuning the posterior probability distributions produced by the pre-trained speech models for individual speech windows. We refer to a classifier trained on target-language data on top of the frozen pre-trained model as a probe, wherein the pre-trained model is frozen. Probe Gain refers to the improvement in LDER obtained when clustering the probe’s outputs instead of the pre-trained posterior probabilities.

$$
\Delta \ = \ \mathrm { L D E R } _ { \mathsf { P r e t r a i n } } \ - \ \mathrm { L D E R } _ { \mathsf { P r o b e } } ,\tag{3}
$$

where $\mathrm { L D E R } _ { \mathsf { P r o b e } }$ refers to LDER from clustering with posterior probabilities with probe model and $\mathrm { L D E R } _ { \mathsf { P r e t r a i n } }$ refers to LDER on test set from using posterior probabilities from pre-trained model. A larger $\Delta$ indicates that the pre-trained mode posteriors do not adequately capture the distinction between the target languages.

Probe Variants. In this work, we test OTPM with two probe variants, both logistic regressions on frozen features: (1) the posterior probe is trained on top of pre-trained LID posterior $p _ { w }$ (99 dimensions for Whisper base, 100 for large-v3); (2) the encoder probe uses standardized final-layer encoder output as input, mean-pooled over the temporal window (resulting in 512 and 1280 dimension vector for Whisper base and large-v3 respectively). In both cases, the probe’s class probabilities replace the pre-trained posterior as the clustering input.

## 2.4 Hypothesis

We hypothesize that Off-target posterior mass (OTPM) calculated for a corpus of utterances can be used as a proxy for the fine-tuning probe gain (∆) that can be obtained for SLD of the target languages in the corpus. This hypothesis is intuitive as larger OTPM indicates that fine-tuning would re-distribute the posterior mass from non-target languages to target-languages, thereby leading to better diarization. Hence, we would expect that larger the OTPM value, higher the expected gain from fine-tuned SLD probe. To establish this relationship, we build an ordinary least squares model (as illustrated in Fig. 1(c)) to predict SLD probe gain from OTPM for a set of training SLD language pairs and tested on unseen corpora and languages.

## 3. Experimental setup

Data. Firstly, hypothesis stated in Sec. 2.4 was tested on synthetically generated bilingual corpora from language pairs sampled from FLEURS dataset.<sup>21</sup> FLEURS<sup>21</sup> is a multilingual read-speech corpus of monolingual utterances, which lets us choose the target languages for diarization enabling testing across widely and less commonly spoken languages. We fit on one training set of 32 language pairs and test the fits on a held-out set of 19 unseen test pairs. The 32 training pairs are Yoruba–Amharic, Swahili–Hausa, Bengali–Tamil, Vietnamese–Thai, French–German, Japanese–Korean, Spanish– Portuguese, Filipino–Indonesian, Assamese–Nepali, Persian–Pashto, Marathi–Gujarati, Malay–Filipino, Vietnamese–Khmer, Dutch–Swedish, Arabic–Hebrew, Italian–Romanian, Russian–Ukrainian, Polish–Czech, Bulgarian–Macedonian, Lithuanian– Latvian, Finnish–Estonian, Turkish–Azerbaijani, Kazakh–Uzbek, Kannada–Telugu, Punjabi–Sindhi, Danish–Norwegian, Afrikaans–Luxembourgish, Catalan–Galician, Shona–Lingala, Mandarin–Burmese, Georgian–Armenian, and Javanese– Maori. Synthetic data for each language pair was generated from concatenation of 10 test-split utterances per language with¯ 0.3 s gaps; each language pair comprises 45 utterances unless one of the languages data size was shorter than the required duration. Following this, OTPM was computed for each language pair resulting in a range of 0.015 to 0.978 across the 32 training pairs.

To test the generalizability of hypothesis, validation on real-world multilingual speech corpora was performed: (1) SEAME<sup>22</sup> Mandarin–English (30 development recordings, 21.0 h of audio, 11.1 h of reference speech) (2) VoxPopuli<sup>23</sup> German and English (40 full European Parliament sessions whose aligned speech is at least 90% in those two languages; 36.2 h of audio, 9.0 h of aligned reference speech) and (3) DISPLACE 2024,<sup>2</sup> a far-field multi-party conversational benchmark with three target languages (English, Hindi, Malayalam; 32 evaluation files, 17.2 h of reference speech under three language labels). Across all corpora, evaluation was performed on unseen recordings, with training performed on 150 SEAME, 60 VoxPopuli and 34 DISPLACE recordings.

Diarization pipeline. All experiments performed throughout the manuscript share a common cascaded pipeline. For the real corpora, Silero VAD<sup>24</sup> filters non-speech segments. For synthetically generated data, each recording is treated as a single speech region. Each speech region is then split into 10 s windows at a 0.4 s hop. If a region is shorter than 20 s, it is merged into a single window. Window level features are then extracted from pre-trained/fine-tuned models followed by cosine agglomerative hierarchical clustering (average linkage, threshold from a two-component Gaussian-mixture calibration of the similarity score and offset 0.015). Additionally, one VBx resegmentation pass<sup>25</sup> (Fa=1.5, Fb=2) was performed. LDER is computed using $\mathsf { d s c o r e } ^ { 2 0 }$ with no collar and includes overlapping speech in scoring.

Models. Whisper base (74M) is the pre-trained model used for all main results in this work. Synthetic data experiments were repeated with Whisper large-v3 (1.55B) in Sec. 4.6 to demonstrate generalizability of observed results.

Statistical analysis. We fit ordinary least squares model between SLD probe gain and OTPM $( \hat { \Delta } = \alpha + \beta { \bf O T P M } )$ and report the slope with a 95% residual-bootstrap CI (10k resamples; a pairs bootstrap for the N=50 reference runs), in-sample $R ^ { 2 }$ and leave-one-out $Q ^ { 2 }$ . Correlation between OTPM and probe gain was tested with one-sided Spearman $\rho ,$ with a stratified $\rho$ (the file-weighted mean of the per-corpus values) across the files of real-world corpora and permuted for within corpora. To

(a) Whisper base, posterior probe  
![](images/f6c3870910141fbd0af6670f93a2d81ddfdc74c4f28b11e74f3665b1a8606e2a.jpg)  
(c) Whisper large-v3, posterior probe

(b) Whisper base, encoder probe  
![](images/7f4d9790993dd8b2b31689e67f481b85f8be4cd21b23d55aff27ca500a48bc3f.jpg)  
(d) Whisper large-v3, encoder probe

![](images/c6e556e354c9952c6dd832284b668e3df247d99b83a223d11e8ddac253ad8744.jpg)  
Training synthetic language pairs  
Unseen synthetic evaluation pairs

![](images/1dd449e9d297c29b6ccaca370a6912039ce70d1c8005745594276e0de2291d26.jpg)  
Real corpus  
95% prediction interval of the fit

Fig. 2. Probe gain vs. OTPM at N=400 training utterances per language, (a) Whisper base posterior-probe, (b) Whisper base encoder-probe, (c) Whisper large-v3 posterior probe and (d) Whisper large encoder probe. Blue Circles indicate the 32 training synthetic language pairs. Plus markers indicate the 19 unseen synthetic evaluation pairs. Black line shows the OLS fit on the training pairs and each panel reports its slope $\beta ,$ in-sample $R ^ { 2 }$ and leave-one-out $Q ^ { 2 }$ (95% CIs of β ). Shaded regions in gray mark 95% prediction interval. The encoder-probe fits use all 32 training pairs. The posterior-probe fits leave out the training pairs which produce collapsed posteriors. Labelled examples: pa+sd (Punjabi–Sindhi) and hi+ur (Hindi–Urdu); kk+uz, mr+gu and sn+ln are Kazakh–Uzbek, Marathi–Gujarati and Shona–Lingala. Markers in diamonds in subplot (a, b) indicate real corpora, whose probes are trained on in-domain recordings.

test statistically significant differences between LDER of pre-trained and fine-tuned models, p-values from centered paired bootstrap across evaluation recordings with 10k resamples were compared.

## 4. Results

To test the hypothesis stated in Sec. 2.4, firstly we train an OLS model with 32 training synthetic bilingual corpora using the OTPM measured on the corpora and the probe gain (∆) obtained from fine-tuning with the corpora. This OLS model is then used to predict the probe gain on an unseen set of 19 synthetically generated bilingual language pairs. Then, we test whether the OLS models predictions translate to real-world corpora through comparing the predicted probe gain with OTPM to relevant baselines. Finally, we show that the observed fit between OTPM and probe gain is generalizable across models through replicating synthetic data experiments with Whisper large-v3 model.

## 4.1 OTPM is indicative ofSLD probe gain

OLS fit to predict probe gain (for encoder and posterior probe) from observed OTPM values for synthetic language pairs from FLEURS dataset has a positive slope as shown in Fig. 2. Fine-tuned probes (encoder probe, posterior probe) are trained on each language pair with N=400 utterances, with LDER and OTPM reported on test utterances. This observation aligns with the stated hypothesis that fine-tuning probes are able to provide improvements in diarization ability through directing the posterior mass on non-target languages onto target languages. For the encoder probe, we observe a large positive slope β=33.75 (95% CI [25.1, 42.4]) alongside $R ^ { 2 } { = } 0 . 6 5 , \bar { Q } ^ { 2 } { = } 0 . 6 0$ whereas posterior probe results indicate a more moderate positive slope with β=13.61 [6.6, 20.6] and $R ^ { 2 } { = } 0 . 3 1 , Q ^ { 2 } { = } 0 . 1 4$ . Poor fit for posterior probe is attributed to three language pairs, for which the fit predicts far more gain than observed: Punjabi–Sindhi $( \mathrm { O T P M { = } 0 . 8 2 , \Delta _ { T r u e } { = } { - } 2 . 2 0 , }$ $\Delta _ { \mathrm { P r e d } } { = } 1 3 . 1$ 1 pt), Shona–Lingala $( \mathrm { O T P M { = } 0 . 9 8 , } \Delta _ { \mathrm { T r u e } } { = } 2 . 0 9 , \Delta _ { \mathrm { P r e d } } { = } 1 5 . 3 )$ and Marathi–Gujarati $( \mathrm { O T P M { = } 0 . 4 0 , \Delta _ { T r u e } { = } 1 . 1 2 , }$ $\Delta _ { \mathrm { P r e d } } { = } 7 . 4 )$ , while the encoder probe obtains LDER gains of 32.45, 37.74 and 24.78 on the same pairs. For these three pairs, clustering with posterior probe leads to collapse, a single cluster rather than two distinct clusters for the two languages resulting in minimal probe gains. Additional discussion related to clustering collapse into single cluster is provided in Sec. 5. Re-computing without the four collapsed pairs (Sec. 5) provides a steeper slope for posterior probe, $\beta { = } 2 1 . 7 5 \ [ 1 6 . 4 , 2 7 . 4 ]$ and a much better fit $( R ^ { 2 } { = } 0 . 6 8 , Q ^ { 2 } { = } 0 . 6 2 , n { = } 2 8 )$ , as shown in Fig. 2.

Table 1. Slope β of probe gain on OTPM [95% CI] by training-set size N (utterances per language). $Q ^ { 2 }$ at $N { = } 4 0 0$ is reported for 32 FLEURS training pairs. Non-coll. indicates OLS fit with language pairs whose posterior probe do not collapse at that training size N (non-collapsed language pairs respectively are n=26, 28, 29, 29, 29). Tests of the N×OTPM interaction: p=0.035, 0.020, 0.69 (rows 1–3 respectively). <sup>†</sup>Probes trained on reserved test-split clips.
<table><tr><td>Probe</td><td> $N { = } 5 0 ^ { \dagger }$ </td><td>100</td><td>200</td><td>300</td><td>400</td><td> $Q ^ { 2 }$ </td></tr><tr><td>Posterior, all</td><td>5.4 [−2.9, 15.9] 8.2 [1.9, 14.9] 12.2 [5.7, 18.8] 13.2 [6.5, 20.1] 13.6 [6.6, 20.6]0.14</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Posterior, non-coll. 18.8 [11.1, 25.8] 16.7 [10.1, 23.2] 20.6 [15.3, 26.1] 22.1 [17.1, 27.5] 22.8 [17.6, 28.1] 0.67</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Encoder, all</td><td>36.8 [29.1, 43.5] 33.4 [24.8, 41.9]33.6 [24.7, 42.5] 33.7 [24.9, 42.3] 33.7 [25.1, 42.4]0.60</td><td></td><td></td><td></td><td></td><td></td></tr></table>

## 4.2 Generalization of OLS fit to unseen pairs

While the OLS fit on the 32 training pairs is able to show linear relationship between OTPM and SLD probe gain, we test whether we are able to predict probe gains on unseen language pairs. This is tested by generating 19 additional unseen language pair corpora from FLEURS dataset. The observed probe gains for each of the 19 unseen pairs are plotted in Fig. 2 both for encoder and posterior probe settings. Firstly, the Spearman correlation between OTPM and probe gain on the unseen pairs is positive for both probes: $\rho { = } 0 . 6 8$ (one-sided $\scriptstyle { p = 0 . 0 0 1 } )$ for the posterior probe on the 18 pairs not flagged for collapse, and $\rho { = } 0 . 5 6$ (one-sided $\scriptstyle { p = 0 . 0 0 6 } )$ for the encoder probe. While the correlation between OTPM and probe gains is high, the absolute magnitude of predictions indicate mixed results. While posterior-probe fit consistently outperformed a mean prediction baseline (mean squared error, MSE, 44.74 vs. 92.19; out-of-sample $R ^ { 2 } { = } 0 . 5 2 )$ , the encoder-probe fit perform worse than constant mean prediction (120.87 vs. 64.32; $R ^ { 2 } { = } - 0 . 8 8 )$ .

## 4.3 Impact oftraining-set size on SLD probe gain

To test whether the positive slope between OTPM and probe gain arises primarily from size of training data, we perform OLS fits for training set size $N \in \{ 5 0 , 1 0 0 , 2 0 0 , 3 0 0 , 4 0 0 \}$ while holding evaluation utterances and clustering algorithm fixed. Results are reported in Table 1, and for all training set sizes a positive slope is observed on OLS fit for both encoder probe and posterior probe. While the encoder probe slope across different training set sizes remains constrained to a small range 33.4–33.7, posterior probe OLS slope increases from 5.4–13.6. Within posterior probe slopes, there is stark increase in slope from $N { = } 5 0$ to N=200 followed by a plateau. We observe that this increase in slope is primarily arising from clustering collapse of Yoruba–Amharic languages into a single cluster, leading to noticeably lower SLD probe gain in smaller training sets.

## 4.4 Translation ofOLSfit to real-world speech

While experiments using FLEURS dataset on synthetically generated data establish the relationship between OTPM and SLD probe gain, we test the extent to which the OLS model is able to translate to real-world speech corpora. Results for OTPM values along with the true SLD probe gains and predicted gains on three real-world corpora, both for Posterior probe and Encoder probe are reported in Table 2. Firstly, we observe that OTPM predictions of probe gains are noticeably higher compared to actual gains. This could be owing to the mismatch between the manner in which synthetic data was constructed compared to real world multilingual speech corpora. While the synthetic data is composed of utterances that contains equal proportions of both languages in a given recording, often real world corpora contain recordings that are primarily single language with code-switched or code-mixed second language. This results in a sim-to-real distribution shift which leads to our OLS model over-predicting the expected gains.

Although OTPM over-predicts the gains on real corpora, OTPM is a better predictor of probe gains compared to using LDER of pre-trained model in the OLS model (posterior-probe MSE 42.6 vs. LDER MSE 428). Within each corpus, OTPM orders recordings by probe gain as indicated by the per-recording Spearman $\rho ,$ averaged over corpora, is 0.23 for the encoder probe and 0.27 for the posterior probe (Table 2), with highest correlation on VoxPopuli dataset.

Table 2. Results on the three real corpora for Whisper base. $\Delta _ { \mathrm { p o s t } }$ and $\Delta _ { e n c }$ refer to the true LDER probe gains measured on posterior and encoder probes for the respective corpora with p-values indicating whether the gain was significantly above zero. $\Delta _ { \mathrm { p o s t } } ^ { \mathrm { O T P M } }$ and $\Delta _ { \mathrm { p o s t } } ^ { \mathrm { L D E R } }$ refer to the posterior probe gains predicted from OLS models trained using OTPM and baseline LDER respectively. Within-corpus ρ indicates how well OTPM orders the recordings of each corpus by corresponding probe gains. SEAME, VoxPopuli and DISPLACE datasets have 30, 40 and 32 test recordings respectively with collapsed recordings for (pre-trained / posterior / encoder) for each dataset being SEAME 13/14/16, VoxPopuli 8/11/9 and DISPLACE 11/14/10. Collapsed recordings are ignored during computation of Spearman correlation. Average over corpora reports the average Spearman correlation for recordings across all three corpora combined.
<table><tr><td colspan="3"></td><td></td><td></td><td>Predicted</td><td> $\Delta _ { \mathrm { p o s t } }$ </td><td colspan="2">Within-corpus ρ</td></tr><tr><td>Corpus</td><td>OTPM</td><td> $\mathbf { L D E R } _ { \mathsf { p r e } }$ </td><td> $\Delta _ { \mathrm { p o s t } } \left( p \right)$ </td><td> $\Delta _ { \mathbf { e n c } } \left( p \right)$ </td><td> $\Delta _ { \mathrm { p o s t } } ^ { \mathrm { O T P M } }$ </td><td> $\Delta _ { \mathrm { p o s t } } ^ { \mathrm { L D E R } }$ </td><td>Encoder</td><td>Posterior</td></tr><tr><td>SEAME</td><td>0.392</td><td>68.61</td><td></td><td>+3.34 (0.003) +3.81 (0.002)</td><td>8.93</td><td>33.77</td><td>0.06</td><td>0.33</td></tr><tr><td>VoxPopuli</td><td>0.374</td><td>25.82</td><td></td><td>+6.20 (0.004) +6.62 (0.003)</td><td>8.53</td><td>9.65</td><td>0.43</td><td>0.34</td></tr><tr><td>DISPLACE</td><td>0.458</td><td>37.05</td><td>+2.28 (0.17) +4.44 (0.0025)</td><td></td><td>10.37</td><td>15.98</td><td>0.27</td><td>0.15</td></tr><tr><td colspan="9">Average over corpora (one-sided  $p )$ </td></tr></table>

Table 3. Whisper base vs. large-v3, $N { = } 4 0 0$ . Fit on the 32 training pairs for posterior probe (n indicates the language pairs that did not have a posterior collapse during clustering, β refers to the slope of fit along with 95% CI and $Q ^ { 2 } )$ . Test on the 19 unseen language pairs: Spearman $\rho _ { \mathrm { t e s t } }$ and out-of-sample $R _ { \mathrm { t e s t } } ^ { 2 }$ against a constant baseline (<sup>∗</sup> indicates 95% bootstrap interval of the improvement excludes zero).
<table><tr><td></td><td colspan="5">Posterior probe</td><td colspan="5">Encoder probe</td></tr><tr><td></td><td></td><td>n β [95% CI]</td><td> $Q ^ { 2 }$ </td><td> $\rho _ { \mathrm { t e s t } }$ </td><td> $R _ { \mathrm { t e s t } } ^ { 2 }$ </td><td></td><td>n β [95% CI]</td><td> $Q ^ { 2 }$ </td><td> $\rho _ { \mathrm { t e s t } }$ </td><td> $R _ { \mathrm { t e s t } } ^ { 2 }$ </td></tr><tr><td>base</td><td></td><td>28 21.8 [16.4, 27.4] 0.62 0.68</td><td></td><td></td><td>0.52</td><td></td><td>32 33.7 [25.1,42.4] 0.60 0.56 -0.88</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>1arge-v3 31 17.7 [12.2,24.5] 0.43 0.65 0.50*</td><td></td><td></td><td></td><td></td><td>32 31.0 [23.2, 39.9] 0.57 0.69</td><td></td><td></td><td>0.53</td></tr></table>

## 4.5 Comparison ofOTPM to baseline metric

OTPM is one metric that could be used to predict SLD probe gains. To demonstrate its effectiveness, both in terms of ease of computation and predictability of probe gains, we compare its performance to other plausible metrics. Baseline LDER as a metric refers to using LDER obtained from clustering with pre-trained model posteriors to predict the SLD probe gain, whose OLS equation would be: $\hat { \Delta } = \alpha + \beta \mathrm { { L D E R } _ { P r e t r a i n } }$ . Within FLEURS baseline LDER provides a better fit than OTPM $( Q ^ { 2 } { = } 0 . 7 2$ for posterior probe and $Q ^ { 2 } { = } 0 . 8 1$ for encoder probe). This could be owing to baseline LDER computation accounting for confusion between the two target languages explicitly, while OTPM does not. However OTPM has two distinct advantages, firstly OTPM is computed in a training-free and label-free manner without requiring language level annotations, whereas pre-trained LDER requires recording level labels alongside clustering. Secondly, when applied to real-world corpus we observe that the baseline LDER OLS fit over-predicts the probe gain on every dataset compared to OTPM as reported in Table 2.

## 4.6 Generalization ofprobe gainfit with Whisper large-v3

In addition to Whisper-base model, we repeat the synthetic-data experiments with Whisper large-v3 (1.55B parameters) to show generalization across models. From the results in Table 3, the OLS fit of OTPM on probe gain with 32 training pairs gives a leave-one-out $Q ^ { 2 } { = } 0 .$ .43 for the posterior probe (n=31 non-collapsed pairs) and $Q ^ { 2 } { = } 0 . 5 7$ for the encoder probe. On the test pairs, the Spearman correlation between OTPM and probe gain is significantly positive: one-sided $\scriptstyle p = 0 . 0 0 2$ for the posterior probe on the 18 test pairs that do not collapse, and $\scriptstyle { p = 0 . 0 0 0 5 }$ for the encoder probe. Unlike with base, the frozen posterior-probe fit also beats the constant prediction baseline with a confident margin (95% interval of the reduction in mean squared error [5.9, 151.9]). These observations validate our hypothesis that OTPM can act as a training-free metric to forecast SLD probe gains on unseen corpora. As Whisper large-v3 is much larger compared to Whisper-base and consequently is able to better model multilingual speech utterances, the OTPM range observed across different language pairs is smaller than

Whisper-base. The mean OTPM values across the test and evaluation pairs for Whisper-base is 0.46 while that of Whisper large-v3 is much smaller at 0.26. Consequently, Whisper large-v3 SLD probe provides lower mean probe gain of 5.8 LDER compared to 8.7 for Whisper-base.

## 5. Discussion

Impact of pre-training hours on OTPM. While OTPM is able to track probe gains, one of the possible explanations for this could be that OTPM is indicator of the number of pre-training hours corresponding to a specific language, larger the pre-training data better the models ability at recognizing the language. Spearman correlation of −0.77 between log of Whisper’s speech-recognition training hours $\left( H _ { \mathrm { P r e t r a i n } } \right) ^ { 8 }$ and observed OTPM for the language indicate that the number of pre-training hours have a noticeable impact on OTPM observed. When using $H _ { \mathrm { P r e t r a i n } }$ to forecast SLD probe gains instead of OTPM, we observe that for posterior probe provides a Spearman ρ of 0.9 for training pairs $( \rho _ { O T P M } ^ { \mathrm { { P o s t e r i o r } } } = 0 . 8 4 )$ , and a Spearman ρ of 0.83 for training pairs $( \rho _ { O T P M } ^ { \mathrm { E n c o d e r } } = 0 . 7 4 )$ . However, on the unseen 19 language pairs pre-training hours noticeably underperforms OTPM with $\rho _ { O T P M } ^ { \mathrm { P o s t e r i o r } } = 0 . 6 8$ vs $\rho _ { \mathrm { P r e t r a i n H o u r s } } ^ { \mathrm { P o s t e r i o r } } { = } 0 . 3 2$ , and $\rho _ { O T P M } ^ { \mathrm { E n c o d e r } } = 0 . 5 6 \ \mathrm { v s } \ \rho _ { \mathrm { P r e t r a i n H o u r s } } ^ { \mathrm { E n c o d e r } } = 0 . 3 1$ . This result indicates that OTPM is able to capture information specific to the dataset and recording conditions beyond number of pre-training hours of the languages, indicated by the better correlation values.

Utility of OTPM as a metric for SLD modeling. The primary utility of OTPM arises from the ability to guide whether a given dataset is useful as a fine-tuning dataset prior to annotating the dataset for language labels. Based on the value of the OTPM, a user can make an informed decision of whether the obtained probe gain justifies annotation cost and corresponding fine-tuning compute requirements. Within a real corpus, OTPM points to the recordings where a probe helps most, though at a stratified ρ of about 0.25 it is a weak guide for any single recording (Sec. 4.4).

Posterior Probe Collapse. In some instances during clustering of the posterior probabilities to obtain language diarization intervals, we observe that the results return a single language despite the recording containing two distinct languages. While no encoder probe model encounters this for the 32 training language pairs, six pairs for posterior probe at $N { = } 5 0$ , four at N=100, and on three pairs at N=200 on (Marathi–Gujarati, Punjabi–Sindhi, Shona–Lingala) result in posterior collapse. While the collapse for some language pairs (Kazakh–Uzbek, Kannada–Telugu, Yoruba–Amharic) was resolved at large training sample sizes, the rest persists across training sample sizes, consistent with both target language posteriors leaking onto the same off-target languages in similar proportions, which prevents the posterior probe from being able to separate the languages.

## 6. Limitations

Proposed OTPM model is able to provide robust results on synthetically generated data and generalize to unseen languages in this setting but the OLS model trained on synthetic data shows limited generalizability to real-world data. We hypothesize that this gap primarily arises from our synthetically generated data containing equal proportions of both languages in the recordings, whereas real-world recordings contain majority of a single language with parts of second language in between. Hence, a more realistic set of synthetically generated data could potentially generalize better to real-world corpora. Additionally, while OTPM accounts for the posterior probabilities assigned to non-target languages in SLD, it does not explicitly model language confusion between the target languages (which are non-negligible for languages in the same family<sup>26</sup>). This phenomenon is illustrated with two examples, firstly in Turkish–Azerbaijani has OTPM=0.08 but has probe gains of +7.54 (posterior) and +21.23 pt (encoder) against predictions of +2.2 and +3.0, and Hindi–Urdu shows similar observations with OTPM 0.07 but encoder-probe gain +15.71 against +2.49 predicted. This could be another factor that explains the limited generalizability of the proposed OTPM metric.

## 7. Conclusion

We proposed off-target LID posterior mass (OTPM), the proportion of a pre-trained model’s LID posterior that falls outside the target languages, as a way to forecast SLD probe gains from unlabelled audio. Results on synthetic data indicate that OTPM is able to generalize on unseen language pairs for predictions of probe gain, both on Whisper base and Whisper large-v3 models. Additionally, through ablations on the training-set size and comparison to baselines we show that OTPM is able to capture meaningful information pertaining to probe gains. While results on real-world data provide positive correlation for recording level probe gain, on corpus level data we see that OTPM suffers from lack of robustness owing to gap between synthetic and real-world data characteristics and lack of accounting for language confusion within target languages. Future works should attempt to account for these limitations so as to develop robust diagnostic metrics for SLD modeling.

## Acknowledgments

Large language model assistance was used for writing code, generating plots and formatting tables. The core research ideas and scientific content are solely the authors’, and LLMs were not used. All text and code included as a part of the manuscript was verified by the authors.

## Author Declarations

The authors have no conflicts to disclose.

## Data Availability

All datasets used in this work are available publicly. FLEURS and VoxPopuli are available from their original releases.<sup>21,23</sup> SEAME and DISPLACE 2024 corpora are available from their original distributors under their respective licenses.<sup>2,22</sup> The code and per-pair results that support the findings of this study will be open-sourced upon manuscript acceptance.

## References and links

<sup>1</sup> J. Mishra and S. R. M. Prasanna, “Implicit self-supervised language representation for spoken language diarization,” IEEE/ACM Transactions on Audio, Speech, and Language Processing 32, 3393–3407 (2024) doi: 10.1109/TASLP.2024.3426978.

<sup>2</sup> S. B. Kalluri, P. Singh, P. Roy Chowdhuri, A. Kulkarni, S. Baghel, P. Hegde, S. Sontakke, D. K T, S. M. Prasanna, D. Vijayasenan, and S. Ganapathy, “The Second DISPLACE Challenge: DIarization of SPeaker and LAnguage in Conversational Environments,” in Interspeech 2024 (2024), pp. 1630–1634, doi: 10.21437/Interspeech.2024-1833.

<sup>3</sup> Y. Wang, H. Liu, and L. P. Garcia, “Bridging Child-Centered Speech Language Identification and Language Diarization via Phonetics,” in Interspeech 2024 (2024), pp. 5148–5152, doi: 10.21437/Interspeech.2024-159.

<sup>4</sup> H. Liu, X. Zhang, H. Zhang, L. P. Garcia-Perera, A. W. H. Khong, E. S. Chng, and S. Watanabe, “Aligning speech to languages to enhance code-switching speech recognition,” IEEE Transactions on Audio, Speech and Language Processing 33, 4712–4725 (2025) doi: 10.1109/TASLPRO.2025.3629290.

<sup>5</sup> G. Frost, E. Morris, J. Jansen van Vüren, and T. Niesler, “Fine-tuned self-supervised speech representations for language diarization in multilingual code-switched speech,” in Artificial Intelligence Research, edited by A. Pillay, E. Jembere, and A. Gerber, Springer Nature Switzerland, Cham (2022), pp. 246–259.

<sup>6</sup> S. Lee, W. Choi, J. Kim, and H.-G. Kang, “Sage-ld: Towards scalable and generalizable end-to-end language diarization via simulated data augmentation,” arXiv preprint arXiv:2510.00582 (2025).

<sup>7</sup> J. Kalda, T. Alumae, M. Lebourdais, H. Bredin, S. Baroudi, and R. Marxer, “TalTech-IRIT-LIS Speaker and Language Diarization Systems for DISPLACE 2024,” in Interspeech 2024 (2024), pp. 1635–1639, doi: 10.21437/Interspeech.2024-2462.

<sup>8</sup> A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of the 40th International Conference on Machine Learning, edited by A. Krause, E. Brunskill, K. Cho, B. Engelhardt, S. Sabato, and J. Scarlett, PMLR (2023), Vol. 202 of Proceedings of Machine Learning Research, pp. 28492–28518, https://proceedings.mlr.press/v202/radford23a.html.

<sup>9</sup> V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi, A. Baevski, Y. Adi, X. Zhang, W.- N. Hsu, A. Conneau, and M. Auli, “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research 25(97), 1–52 (2024) http: //jmlr.org/papers/v25/23-1318.html.

<sup>10</sup> G. Sell, D. Snyder, A. McCree, D. Garcia-Romero, J. Villalba, M. Maciejewski, V. Manohar, N. Dehak, D. Povey, S. Watanabe, and S. Khudanpur, “Diarization is Hard: Some Experiences and Lessons Learned for the JHU Team in the Inaugural DIHARD Challenge,” in Interspeech 2018 (2018), pp. 2808–2812, doi: 10.21437/Interspeech.2018-1893.

<sup>11</sup> T. J. Park, K. J. Han, M. Kumar, and S. Narayanan, “Auto-tuning spectral clustering for speaker diarization using normalized maximum eigengap,” IEEE Signal Processing Letters 27, 381–385 (2020) doi: 10.1109/LSP.2019.2961071.

<sup>12</sup> J. Mishra, A. Agarwal, and S. R. M. Prasanna, “Spoken language diarization using an attention based neural network,” in 2021 National Conference on Communications (NCC) (2021), pp. 1–6, doi: 10.1109/NCC52529.2021.9530035.

501–505, doi: 10.21437/Interspeech.2023-1109

<sup>14</sup> A. T. Tran, C. V. Nguyen, and T. Hassner, “Transferability and hardness of supervised classification tasks,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2019)

<sup>15</sup> Y. Bao, Y. Li, S.-L. Huang, L. Zhang, L. Zheng, A. Zamir, and L. Guibas, “An information-theoretic approach to transferability in task transfer learning,” in 2019 IEEE International Conference on Image Processing (ICIP) (2019), pp. 2309–2313, doi: 10.1109/ICIP.2019.8803726.

<sup>16</sup> C. Nguyen, T. Hassner, M. Seeger, and C. Archambeau, “LEEP: A new measure to evaluate transferability of learned representations,” in Proceedings of the 37th International Conference on Machine Learning, edited by H. D. III and A. Singh, PMLR (2020), Vol. 119 of Proceedings ofMachine Learning Research, pp. 7294–7305, https://proceedings.mlr.press/v119/nguyen20b.html.

<sup>17</sup> K. You, Y. Liu, J. Wang, and M. Long, “Logme: Practical assessment of pre-trained models for transfer learning,” in Proceedings of the 38th International Conference on Machine Learning, edited by M. Meila and T. Zhang, PMLR (2021), Vol. 139 of Proceedings of Machine Learning Research, pp. 12133–12143, https://proceedings.mlr.press/v139/you21b.html.

<sup>18</sup> W. Deng and L. Zheng, “Are labels always necessary for classifier accuracy evaluation?,” in 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2021), pp. 15064–15073, doi: 10.1109/CVPR46437.2021.01482.

<sup>19</sup> S. Garg, S. Balakrishnan, Z. C. Lipton, B. Neyshabur, and H. Sedghi, “Leveraging unlabeled data to predict out-of-distribution performance,” in International Conference on Learning Representations (2022), https://openreview.net/forum?id=o\_HsiMPYh\_x.

<sup>20</sup> N. Ryant, K. Church, C. Cieri, A. Cristia, J. Du, S. Ganapathy, and M. Liberman, “The Second DIHARD Diarization Challenge: Dataset, Task, and Baselines,” in Interspeech 2019 (2019), pp. 978–982, doi: 10.21437/Interspeech.2019-1268.

<sup>21</sup> A. Conneau, M. Ma, S. Khanuja, Y. Zhang, V. Axelrod, S. Dalmia, J. Riesa, C. Rivera, and A. Bapna, “Fleurs: Few-shot learning evaluation of universal representations of speech,” in 2022 IEEE Spoken Language Technology Workshop (SLT) (2023), pp. 798–805, doi: 10.1109/SLT54892.2023.10023141.

<sup>22</sup> D.-C. Lyu, T.-P. Tan, E. S. Chng, and H. Li, “SEAME: a Mandarin-English code-switching speech corpus in south-east asia,” in Interspeech 2010 (2010), pp. 1986–1989, doi: 10.21437/Interspeech.2010-563

representation learning, semi-supervised learning and interpretation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), edited by C. Zong, F. Xia, W. Li, and R. Navigli, Association for Computational Linguistics, Online (2021), pp. 993–1003, https://aclanthology.org/2021.acl-long.80/, doi: 10.18653/v1/2021.acl-long.80.

<sup>24</sup> S. Team, “Silero vad: pre-trained enterprise-grade voice activity detector (vad), number detector and language classifier” , https://github.com/snakers4/silero-vad (2024).

<sup>25</sup> F. Landini, J. Profant, M. Diez, and L. Burget, “Bayesian hmm clustering of x-vector sequences (vbx) in speaker diarization: Theory, implementation and analysis on standard tasks,” Computer Speech & Language 71, 101254 (2022) https://www.sciencedirect.com/science/article/pii/S0885230821000619 doi: https://doi.org/10.1016/j.csl.2021.101254.

<sup>26</sup> A. Kommineni, R. Hebbar, S. Petrosyan, P. Khobragade, S. Kadiri, M. A. Rentería, J. Lee, and S. Narayanan, “Can speech foundation models effectively identify languages in low-resource multilingual aging populations?,” JASA Express Letters 5(9) (2025)