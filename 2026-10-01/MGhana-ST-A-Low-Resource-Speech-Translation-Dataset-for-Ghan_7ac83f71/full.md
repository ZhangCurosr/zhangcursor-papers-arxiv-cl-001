# MGhana-ST: A Low-Resource Speech Translation Dataset for Ghanaian Languages and an Analysis of Multilingual Training Trade-offs

Frank Lawrence Nii Adoquaye Acquaye<sup>1,2</sup>\* Eric George Parakal<sup>3</sup> Jesse Johnson<sup>2</sup> Kishankumar Bhimani<sup>4</sup> Jochebed Afua Basil<sup>1</sup>

<sup>1</sup>Ashesi University, Berekuso, Ghana <sup>2</sup>AdwumaTech AI, Accra, Ghana <sup>3</sup>HSE University, Moscow, Russia

<sup>4</sup>GIFT International Fintech Institute, Gandhinagar, India

{facquaye, jochebed.basil}@ashesi.edu.gh jesse.johnson@adwumatech.ai ericparakal@gmail.com info.bhimani@gmail.com

## Abstract

We present MGhana-ST, a speech translation dataset for four low-resource Ghanaian language varieties: Ga, Twi (Akuapem and Asante), Ewe, and Fante. MGhana-ST is an ongoing annotation effort; the experiments reported here use a fixed subset of approximately 16.1 hours of paired speech and English translation data. The audio is curated from two existing Ghanaian speech resources. The English translations in MGhana-ST are produced directly from the audio by 37 native-speaker annotators, rather than derived from source-language transcriptions as in the upstream resources, and are accompanied by verbal and non-verbal event annotations.

Using Whisper-small as a backbone, we examine monolingual and multilingual training under severe data scarcity, reporting all baseline results as means over three training seeds. We find that flat multilingual training benefits no variety in this regime. Ga and Twi are unchanged within seed variance (+0.51 and +0.06 BLEU against monolingual standard deviations of 1.63 and 2.20), while Ewe declines by 6.99 BLEU and Fante by 5.11. The two varieties that degrade are Ewe, which is both linguistically distinct and drawn from a different source corpus, and Fante, the least-resourced variety. We further compare empirical crosslingual transfer with typology-based similarity and find that, in this four-language setting, transfer BLEU identifies which language pairs interact more closely than URIEL similarity does, although neither measure predicts which varieties benefit from joint training.

We also report a methodological finding of independent interest. An earlier single-run version of this analysis found positive transfer for three of four varieties; that result did not survive replication across seeds. For Ga and Twi,

monolingual baselines trained on roughly 1.6 to 6.2 hours of audio have seed standard deviations of 1.63 and 2.20 BLEU, roughly five and thirty times those of the corresponding multilingual models (0.35 and 0.07). When the monolingual condition is the noisier one, a single-run comparison can show apparent transfer of this size from seed variation alone. We release MGhana-ST to support future research on African language speech technologies and low-resource multilingual speech translation.

## 1 Introduction

Multilingual speech models have significantly expanded the scope of automatic speech recognition and speech translation. Whisper (Radford et al., 2023), trained on 680,000 hours of audio across 99 languages, achieves strong performance on a range of speech benchmarks, while the Massively Multilingual Speech project (Pratap et al., 2024) extends multilingual speech modelling to more than 1,100 languages. However, scaling multilingual training also introduces a central tension: parameter sharing across languages can improve performance through positive transfer, but it can also degrade performance through negative transfer when jointly trained languages are insufficiently compatible (Wang et al., 2020; Arivazhagan et al., 2019). This trade-off is especially consequential in lowresource settings, where the amount of supervised data is too limited to absorb poorly matched multilingual signals.

Speech translation (ST) enables direct translation from speech in a source language to text in a target language without requiring an intermediate automatic speech recognition (ASR) step (Bérard et al., 2016; Duong et al., 2016). In our setting, reliable ASR support for Ga, Twi, Ewe, and Fante remains limited, making end-to-end speech translation a practical modelling choice. The subset used in our experiments comprises about 16.1 hours of labelled speech-translation data across the four varieties, a regime in which model design and multilingual training strategy can materially affect final system quality.

The Ghanaian languages considered here provide a useful testbed for studying multilingual transfer under partial linguistic relatedness. Twi (in its Akuapem and Asante varieties) and Fante belong to the Akan branch and are more closely related to one another than either is to Ga or Ewe (see the language family column in Table 1). Relatedness within this set is uneven: some language pairs are more likely to benefit from multilingual transfer, while others may introduce cross-language interference. These properties make the setting wellsuited to studying when multilingual speech translation helps, when it hurts, and whether task-specific transfer signals are more informative than typological similarity measures for designing multilingual training configurations.

Beyond introducing MGhana-ST as a new resource for low-resource Ghanaian speech translation, we use the dataset as a controlled benchmark for analysing multilingual training under severe data scarcity. We distinguish between the dataset itself, which is intended as a reusable community resource, and the empirical observations obtained from our experimental study. The latter should be interpreted as evidence from the modelling choices considered here rather than as intrinsic properties of the dataset.

Our experiments yield four main observations. First, flat multilingual training benefits no variety at this data scale: Ga and Twi are unchanged within seed variance, while Ewe and Fante both degrade substantially. Second, the varieties that degrade share identifiable characteristics: Ewe is the only Gbe variety and is also drawn from a different source corpus, while Fante is the least-resourced variety (1.62h). These observations are consistent with the possibility that joint training reallocates limited capacity toward better-represented varieties, although our experiments do not isolate this mechanism. Third, empirically measured cross-lingual transfer BLEU identifies which language pairs interact more closely than URIEL typological similarity does within our four-language setting, but it does not predict which varieties benefit under joint training. Fourth, and methodologically, monolingual baselines at this scale carry seed standard deviations of up to 2.20 BLEU, comparable to the single-run gains of +1.68 to +2.95 BLEU that our own earlier analysis reported; single-run comparisons are therefore not adequate in this regime.

Our contributions comprise two distinct parts: a new dataset resource and an empirical study conducted on it. We keep these separate throughout, and readers interested only in the resource may read Section 3 in isolation.

## Dataset contribution.

• Audio-grounded translation and event annotation: All English translations in MGhana-ST were produced by 37 native-speaker annotators listening to the audio, together with verbal and non-verbal event tags.

• Dataset Release: We publicly release MGhana-ST, a multi-language Ghanaian speech translation resource covering Ga, Twi, Ewe, and Fante, together with per-language corpus statistics and an explicit manifest defining the experimental subset.

## Empirical study.

• Multilingual Training Analysis: We compare monolingual and multilingual speech translation models for four low-resource Ghanaian language varieties across three training seeds.

• Empirical Transfer Study: We analyse cross-lingual transfer within our language set and compare transfer-based and typologybased relatedness measures.

• Seed sensitivity at 16 hours: We quantify training-seed variance for both monolingual and multilingual configurations and show that it is large enough to reverse the sign of reported transfer effects.

## 2 Related Work

Large-scale multilingual speech models such as Whisper (Radford et al., 2023), MMS (Pratap et al., 2024), and mSLAM (Bapna et al., 2022) establish broad multilingual capability across many languages and tasks. More recent work, including SpeechMatrix (Duquenne et al., 2023) and SeamlessM4T (Barrault et al., 2023), extends multilingual speech translation to a large number of languages within unified models. However, these systems primarily focus on scaling capabilities across many languages rather than investigating the specific dynamics of positive and negative transfer within small sets of partially related varieties at severe data scarcity.

Speech resources covering African languages have expanded in recent years, including massively multilingual benchmarks such as FLEURS (Conneau et al., 2023), accented-speech recognition corpora such as AfriSpeech-200 (Olatunji et al., 2023), and speech synthesis corpora such as BibleTTS (Meyer et al., 2022). For Ghanaian languages specifically, UGSpeechData (Wiafe et al., 2025) and the Financial Inclusion Speech Dataset (Asamoah Owusu et al., 2022) provide transcribed speech. Most of these resources were designed primarily for ASR or speech synthesis rather than end-to-end speech translation. MGhana-ST addresses this gap by providing an audio-grounded speech translation dataset for four Ghanaian language varieties and an analysis of multilingual training under severe data scarcity. Our positioning is complementary: while large-scale systems demonstrate broad capability, this work examines the fine-grained transfer dynamics that those aggregate evaluations do not reveal, using typologybased similarity (URIEL) and empirical transfer BLEU as alternative lenses on language relatedness.

## 3 The MGhana-ST Dataset

MGhana-ST is a curated dataset of short speech clips in Ghanaian languages paired with English translations and annotations for non-verbal events such as laughter, applause, and background noise. It is available at https://huggingface. co/datasets/adwumatech-ai/mghana-st (Acquaye et al., 2025).

Licensing. The annotation layer contributed by this work, comprising the English translations and the verbal and non-verbal event tags, is released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0). The underlying audio remains under the terms set by its original distributors: the Financial Inclusion Speech Dataset is distributed under CC BY 4.0 (Asamoah Owusu et al., 2022), and Ewe audio from UGSpeechData under CC BY-NC-ND 4.0 (Wiafe et al., 2025), which restricts it to non-commercial use.

## 3.1 Defining the Experimental Subset

MGhana-ST is an incremental project: annotation is ongoing, and further audio is being processed for inclusion. The experimental subset is the fixed portion of the data used for every result reported in this paper, summarised in Table 1; it comprises approximately 16.1 hours across the four varieties. The public release is the annotated data available at the repository above.

The subset is defined by manifest, not by revision alone. The annotation state underlying our experiments corresponds to commit 4996b709abd584464a6cfb7bd256504fd5eed7a0 of the dataset repository, dated 25 March 2026. We release a manifest at splits/manifest.json in the dataset repository, listing the audio\_id, start\_ms, and end\_ms of every segment assigned to train, validation, and test. This manifest is the authoritative description of what our experiments used.

Audio availability. Of the 90,583 annotated segments across the four varieties, 11,445 (12.6%) reference audio files that were not present locally and therefore never entered preprocessing. Their distribution is highly uneven across varieties: 39.9% of Ewe segments and 30.4% of Fante segments, against 0.03% for Twi and none for Ga (Table 7, Appendix B). We have not determined whether these files are absent from the upstream distributions or absent only from our copy, nor whether their absence is systematic. The Ewe and Fante portions of the experimental subset should therefore be understood as samples of their annotated material whose selection we do not fully characterise.

## 3.2 Sources

MGhana-ST aggregates audio from two existing Ghanaian speech resources. Ewe samples were drawn from the transcribed portion of UGSpeech-Data (Wiafe et al., 2025), a multilingual Ghanaian speech dataset covering Akan, Ewe, and other languages. The remainder was drawn from the Financial Inclusion Speech Dataset (Asamoah Owusu et al., 2022), which covers Akuapem Twi, Asante Twi, Fante, and Ga and leans heavily on the financial technology domain.

Corpus provenance is not evenly distributed across language varieties: all Ewe data is drawn from UGSpeechData, whereas Ga, Twi, and Fante are drawn entirely from the Financial Inclusion Speech Dataset. Because the two corpora differ systematically in domain, recording channel, speaking style, and utterance duration, Ewe differs from the other three varieties along these dimensions as well as linguistically. We return to the implications of this in Section 5 and the Limitations.

What is new in MGhana-ST. The source audio is reused from the two corpora above. The annotation layer is not. MGhana-ST provides translations produced by native speakers listening to the audio directly, rather than translations of written transcriptions as in the upstream resources. Annotators additionally tagged verbal and non-verbal events (laughter, applause, background noise), which have no counterpart upstream.

## 3.3 Languages and Coverage

The experimental subset covers four Ghanaian language varieties: Twi, Ewe, Ga, and Fante. Table 1 reports its train, validation, and test durations per variety. Twi and Fante belong to the Akan branch and are closely related, whereas Ga (Ga-Dangme) and Ewe (Gbe) are more distant from the Akan varieties and from each other.

Split methodology. Splits are constructed at the audio-file level: all segments originating from a single recording are assigned to exactly one of train, validation, or test, so no audio used for training appears in validation or test.

Corpus characteristics. Table 2 reports perlanguage utterance counts and durations. Utterances are short: mean duration is 1.33 s for Fante, 1.68 s for Ga, and 1.90 s for Twi. Ewe, drawn from image-description recordings, averages 3.31 s. English reference diversity differs sharply across varieties (Table 3): 38.4% of Fante test references repeat another reference, against only 1.5% for Ewe. MGhana-ST should therefore be understood as a word- and phrase-level speech translation resource for three of its four varieties, and a mixed phraseand sentence-level resource for Ewe.

<table><tr><td>Language</td><td>Family</td><td>Train (h)</td><td>Val (h)</td><td>Test (h)</td></tr><tr><td>Twi</td><td>Akan</td><td>6.21</td><td>0.87</td><td>0.79</td></tr><tr><td>Ga</td><td>Ga-Dangme</td><td>2.48</td><td>0.29</td><td>0.27</td></tr><tr><td>Ewe</td><td>Gbe</td><td>2.47</td><td>0.37</td><td>0.34</td></tr><tr><td>Fante</td><td>Akan</td><td>1.62</td><td>0.19</td><td>0.18</td></tr><tr><td>Total</td><td></td><td>12.78</td><td>1.72</td><td>1.58</td></tr></table>

Table 1: The MGhana-ST experimental subset: train, validation, and test audio hours per language. This is the fixed subset used for all experiments reported here.

## 3.4 Annotation and Quality Assurance

All English translations and non-verbal event tags were produced by 37 annotators, each a native speaker of the language they annotated. All annotators were employed and paid by AdwumaTech AI and gave informed consent for their translations and annotations to be released. Quality assurance operated at three levels: redundant annotation (a subset of clips received multiple independent translations; 0.3–4.2% of test utterances per variety, Table 3), guided native-speaker annotation (teams working from written guidelines covering event annotation and idiomatic expression handling), and two-round independent verification by more experienced annotators. As an independent check on annotation hygiene, we audited every bracketed event tag in the corpus. Of 56,117 translations containing an event tag, 36 (0.06%) are malformed, with the remaining 99.94% conforming exactly to the documented convention.

## 3.5 Intended Use and Limitations

MGhana-ST is intended to support research on speech recognition, speech translation, and speech representation learning for low-resource Ghanaian languages. It is not designed for forensic speaker identification or sensitive demographic inference. In addition, the short duration of many utterances limits the extent to which long-range speaker characteristics can be captured.

## 4 Experimental Setup

Our task is speech-to-English translation. We use the encoder of Whisper-small (Radford et al., 2023) (244M parameters in the full model) as the speech backbone. Input speech is converted to 80- channel log-Mel spectrograms and passed through the 12-layer Whisper-small encoder; the resulting 768-dimensional states are linearly projected to $d _ { \mathrm { m o d e l } } = 2 5 6$ and consumed by a custom 4-layer Transformer decoder with 8 attention heads, trained from scratch, which autoregressively produces English text.

To adapt the encoder in the low-resource regime, we freeze its lower layers and fine-tune only the top n layers, with n chosen by a preliminary sweep on Ga (Appendix A). The decoder is trained from random initialisation in all configurations.

The overall training objective is

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { t r a n s } } + \alpha \mathcal { L } _ { \mathrm { L I D } } , } \end{array}\tag{1}
$$

where $\alpha$ weights an auxiliary languageidentification objective. The monolingual (M1) and multilingual baseline (M2) configurations use $\alpha = 0$ , so they optimise the translation loss alone; M4 sets $\alpha \ = \ 0 . 1$ M3 instead applies Domain-Adversarial Training (DANN) with a gradient-reversal layer and λ = 0.1; its domain-classification term is not represented by Eq. 1. Auxiliary objective variants (M3 and M4) are reported in Appendix D.

<table><tr><td>Language</td><td>Family</td><td>Train utts.</td><td>Val utts.</td><td>Test utts.</td><td>Total utts.</td><td>Total (h)</td><td>Mean dur. (s)</td></tr><tr><td>Twi</td><td>Akan</td><td>11,849</td><td>1,593</td><td>1,484</td><td>14,926</td><td>7.87</td><td>1.90</td></tr><tr><td>Ga</td><td>Ga-Dangme</td><td>5,278</td><td>624</td><td>639</td><td>6,541</td><td>3.05</td><td>1.68</td></tr><tr><td>Ewe</td><td>Gbe</td><td>2,714</td><td>405</td><td>344</td><td>3,463</td><td>3.18</td><td>3.31</td></tr><tr><td>Fante</td><td>Akan</td><td>4,328</td><td>522</td><td>524</td><td>5,374</td><td>1.99</td><td>1.33</td></tr></table>

Table 2: Per-language corpus characteristics of the experimental subset. Utterance counts include all annotations; where a clip carries more than one independent translation, all references are retained. Total hours may differ from the sum of the per-split hours in Table 1 by 0.01 h due to rounding.

<table><tr><td>Language</td><td>Unique refs.</td><td>Target overlap</td><td>Multi-ref. rate</td></tr><tr><td>Twi</td><td>1,078</td><td>27.4%</td><td>2.7%</td></tr><tr><td>Ga</td><td>465</td><td>27.2%</td><td>0.6%</td></tr><tr><td>Ewe</td><td>339</td><td>1.5%</td><td>0.3%</td></tr><tr><td>Fante</td><td>323</td><td>38.4%</td><td>4.2%</td></tr></table>

Table 3: Reference diversity on the test split. Target overlap is the share of test utterances whose English reference is not unique (1 − unique refs./test utts., with test utterance counts from Table 2). Ewe references are almost entirely unique, whereas over a third of Fante references are repeated.

## 4.1 Reproducibility and Seed Protocol

Two distinct random seeds govern our experiments.

The split seed controls the audio-level assignment of recordings to train, validation, and test. It is fixed at 42 for every experiment in this paper. All runs therefore share a single, identical data partition, which is published as the manifest described in Section 3.1.

The training seed controls parameter initialisation, batch ordering, and dropout. Where we report means and standard deviations, they are computed over training seeds {1, 2, 3} with the split held fixed. The variance we report is thus training variance under a fixed partition.

Runs are not bitwise deterministic (deterministic: false), so exact per-run reproduction is not guaranteed even at a fixed seed; the reported means and standard deviations are the reproducible quantities.

Which results carry error bars. The M1 and M2 configurations are run at three seeds and reported as mean ± sample standard deviation. The auxiliary conditioning experiments and the crosslingual transfer matrix are single-run at training seed 42 and are reported as such (Appendices C and D). We do not compute deltas between multiseed and single-run configurations, as this would confound treatment effects with the particular seed draw.

## 4.2 Dataset

Main experiments use the four-language experimental subset described in Table 1, comprising 12.78 hours of training audio across Twi (6.21h), Ga (2.48h), Ewe (2.47h), and Fante (1.62h). All splits are partitioned at the audio-file level to prevent overlap between training and evaluation.

Baseline scope. We do not include a cascaded ASR→MT baseline, because no ASR system we are aware of supports Ga, Twi, Ewe, or Fante at acceptable accuracy. A cascade is therefore not realisable as a deployable system in this setting, which is characteristic of truly low-resource languages and motivates end-to-end speech translation.

## 4.3 Model Architecture

Figure 1 summarises the architecture. The top n = 4 Whisper encoder layers are fine-tuned while lower layers are frozen, selected to balance retention of pretrained acoustic structure with taskspecific adaptation (see Appendix A for the depth selection analysis).

## 4.4 Linguistic Relatedness Ordering

To compare different notions of relatedness within our language set, we consider URIEL syntactic similarity (Littell et al., 2017), geographic distance, and empirical cross-lingual transfer BLEU. Table 4 summarises these measures for all language pairs; the full pairwise transfer matrix is reported in Appendix C.

The measures do not fully agree. URIEL ranks Ga–Twi as the closest pair, whereas transfer BLEU identifies Twi–Fante as the strongest pair. Because transfer BLEU is derived directly from model behaviour on the target task, we use it as the primary ordering signal in our analysis. The gap between Twi–Fante (18.15 BLEU) and every other pair (0.13–1.23) is more than an order of magnitude, far larger than the seed variance measured elsewhere in this paper. The divergence among these measures suggests that general typological similarity does not necessarily correspond to taskspecific transfer behaviour in low-resource speech translation.

![](images/2f89b40f86d0e672fd76a98e6565a98c7cbb852c0cdd65de86af2222313747e3.jpg)  
Figure 1: MGhana-ST inference graph. Input speech is processed by the Whisper feature extractor and encoder, projected to the decoder dimension, and decoded autoregressively to produce an English translation.

Transfer BLEU should not be interpreted as reflecting linguistic relatedness alone. It may also be influenced by factors such as dataset size, speaker variability, recording conditions, and domain overlap. Notably, all Ewe data comes from a different source corpus than the other three varieties, so domain and channel similarity is partially confounded with linguistic relatedness. We therefore treat transfer BLEU as a task-specific empirical measure that aggregates linguistic, acoustic, and domain similarity.

## 5 Results and Analysis

## 5.1 Does Multilingual Training Help?

Table 5 reports monolingual (M1) and flat multilingual (M2) performance as means over three training seeds. The answer, at this data scale, is no: multilingual training does not reliably improve any of the four varieties, and substantially degrades two.

<table><tr><td>Pair</td><td>URIEL sim. ↑</td><td>Geo. dist. ↓</td><td>Transfer BLEU↑</td></tr><tr><td>Twi – Fante</td><td>0.772</td><td>0.109</td><td>18.15</td></tr><tr><td>Ga-Ewe</td><td>0.789</td><td>0.087</td><td>0.13</td></tr><tr><td>Ga − Twi</td><td>0.874</td><td>0.119</td><td>0.68</td></tr><tr><td>Twi – Ewe</td><td>0.794</td><td>0.165</td><td>0.14</td></tr><tr><td>Ga – Fante</td><td>0.767</td><td>0.211</td><td>1.23</td></tr><tr><td>Ewe – Fante</td><td>0.772</td><td>0.218</td><td>0.18</td></tr></table>

Table 4: Pairwise language relatedness across complementary measures. Transfer BLEU is the mean of the two directional zero-shot scores for each pair, single-run at seed 42. URIEL identifies Ga–Twi as closest, while transfer BLEU identifies Twi–Fante as the strongest transfer pair.

Ga and Twi are unchanged. Ga gains +0.51 BLEU and Twi +0.06. Both are small relative to their monolingual seed standard deviations (1.63 and 2.20 respectively). We therefore treat both as null results rather than as gains.

Ewe and Fante degrade. Ewe falls from 15.48 to 8.49 BLEU (−6.99), and Fante from 47.61 to 42.50 (−5.11). Both effects exceed twice the larger of the two conditions’ standard deviations and are corroborated by chrF (−8.64 and −3.89). These are the substantive findings of this section.

Which varieties pay. The two varieties harmed are the typologically isolated one and the leastresourced one. Ewe is the only Gbe variety in the set, and additionally the only variety drawn from a different source corpus, so it shares neither family structure nor domain with the majority of the training pool. Fante, at 1.62h, has the smallest training allocation of the four, roughly a quarter of Twi’s 6.21h. Twi, the largest, is unaffected. This is the pattern expected if joint training under fixed capacity reallocates representational resources toward the better-represented and better-connected varieties.

## 5.2 Seed Variance and the Cost of Single-Run Reporting

The seed standard deviations in Table 5 are themselves informative, and they differ between configurations in a way that matters for how such comparisons are reported.

For Ga and Twi, the monolingual models are markedly less stable than the multilingual ones: ±1.63 versus ±0.35 BLEU for Ga, and ±2.20 versus ±0.07 for Twi. For Ewe and Fante the ordering reverses (±0.56 versus ±1.15, and ±0.79 versus ±1.43). The multilingual model sees two to eight times as much training audio per epoch as any single monolingual model, which plausibly stabilises Ga and Twi. That the two degraded varieties are also the ones whose multilingual runs vary most is consistent with their performance depending on how shared capacity is allocated in each run, although we have not tested this directly.

<table><tr><td>Lang.</td><td>M1 (mono)</td><td>M2 (multi)</td></tr><tr><td>BLEU</td><td> $3 5 . 2 5 \pm 1 . 6 3$ </td><td> $3 5 . 7 6 \pm 0 . 3 5$   $+ 0 . 5 1$ </td></tr><tr><td>Ga Twi</td><td> $1 8 . 7 9 \pm 2 . 2 0$ </td><td> $1 8 . 8 5 \pm 0 . 0 7$   $+ 0 . 0 6$ </td></tr><tr><td>Ewe</td><td> $1 5 . 4 8 \pm 0 . 5 6$ </td><td> $8 . 4 9 \pm 1 . 1 5$   $\mathbf { - 6 . 9 9 }$ </td></tr><tr><td>Fante</td><td> $4 7 . 6 1 \pm 0 . 7 9$ </td><td> $4 2 . 5 0 \pm 1 . 4 3$  -5.11</td></tr><tr><td>chrF</td><td></td><td></td></tr><tr><td>Ga</td><td> $4 8 . 5 5 \pm 3 . 3 0$ </td><td> $4 9 . 5 0 \pm 0 . 7 9$   $+ 0 . 9 5$ </td></tr><tr><td>Twi</td><td> $3 0 . 8 6 \pm 0 . 9 5$ </td><td> $3 0 . 5 8 \pm 0 . 8 4$   $- 0 . 2 8$ </td></tr><tr><td>Ewe</td><td> $3 4 . 3 4 \pm 0 . 1 0$ </td><td> $2 5 . 7 0 \pm 1 . 8 6$   $\mathbf { - 8 . 6 4 }$ </td></tr><tr><td>Fante</td><td> $6 1 . 7 5 \pm 0 . 5 4$ </td><td> $5 7 . 8 6 \pm 1 . 4 5$  -3.89</td></tr></table>

Table 5: Monolingual (M1) versus flat multilingual (M2) performance per variety, mean ± sample standard deviation over three training seeds with the data split held fixed. Bold marks deltas whose magnitude exceeds twice the larger of the two standard deviations. Corpuslevel M2 BLEU is $2 2 . 5 0 \pm 0 . 2 7$ . Per-seed values are in Appendix F.

This matters methodologically. Where the monolingual condition is the noisier one, a single-run M1–M2 comparison pairs a noisy monolingual number with a stable multilingual one, so the sign of the reported delta is largely determined by where the monolingual draw happens to fall. An earlier single-run version of this analysis, at training seed 42, reported +2.95 (Ga), +1.68 (Twi), −4.46 (Ewe), and +0.46 (Fante), that is, three gains and one loss. Under replication, all three apparent gains disappear, and Fante’s reverses sign.

We do not believe this failure mode is specific to our setup. Low-resource multilingual comparisons are often reported from a single run, and wherever monolingual baselines trained on a few hours of audio are less stable than the pooled model, as they are for Ga and Twi here, single-run reporting can produce apparent positive transfer from seed variation alone. Our results suggest that comparisons at this data scale should report seed variance for both conditions as a matter of course.

## 5.3 Separating Family from Domain

Because all Ewe data is drawn from UGSpeech-Data while Ga, Twi, and Fante come entirely from the Financial Inclusion Speech Dataset, the Ewe degradation admits two explanations: family distance (Gbe vs. Akan) and domain mismatch. Ga provides a partial control, since it shares corpus provenance with Twi and Fante but does not share their language family. If domain similarity alone drove transfer, Ga should transfer strongly to Twi and Fante; instead Ga→Twi and Ga→Fante reach only 0.98 and 1.48 BLEU (Table 8, Appendix C), far below Twi→Fante and Fante→Twi (23.36 and 12.94). Domain overlap alone is therefore not sufficient to generate transfer in this dataset.

## 5.4 Effect of Linguistic Relatedness

To examine the role of linguistic relatedness, we compare multilingual outcomes with pairwise relatedness measures and cross-lingual transfer scores. The strongest zero-shot transfer is observed between Twi and Fante, which is consistent with their closer linguistic relationship and shared corpus provenance. By contrast, language pairs involving Ewe and Ga show substantially weaker transfer.

The multi-seed results qualify how this should be read. Strong pairwise transfer between Twi and Fante does not translate into Fante benefiting from joint training; Fante is one of the two varieties that degrades. High measured transfer therefore predicts that a language can draw on a related pool, not that it will be allocated capacity within it. Transfer BLEU remains the more direct task-specific indicator of which pairs interact in this setting, but it is not by itself a predictor of which languages gain under joint training.

These comparisons suggest that, in our setting, task-specific transfer measurements are more informative than general typological similarity for anticipating which languages interact.

## 5.5 Target-Side Overlap and Reference Diversity

Because MGhana-ST is drawn from narrowdomain source corpora, English reference translations repeat across examples, which inflates BLEU without reflecting translation capability. Crosslanguage score comparisons are not like-for-like. Fante’s high monolingual score (47.61 BLEU from 1.62h of training audio) is likely influenced by reference repetition rather than indicating that Fante is an easier language: only 323 distinct English strings cover its 524 test utterances. Ewe sits at the opposite extreme, with 1.5% target overlap and 3.31 s mean duration, so its lower absolute BLEU (15.48) reflects a harder generation task. Absolute

BLEU should therefore not be compared across the four varieties. This does not affect the withinlanguage comparisons that carry our main findings: the M1–M2 contrast holds the test set fixed for each language, so the deltas are measured against each variety’s own baseline.

Further details on prompt reuse and domain overlap appear in Appendix E.

## 6 Discussion

Our results support four broader points about lowresource multilingual speech translation.

First, multilingual training should not be assumed to provide benefits at all at this data scale. In MGhana-ST, joint training left two varieties unchanged and substantially degraded the other two. Practitioners building for these varieties at comparable data volumes should treat monolingual training as the default and multilingual training as a hypothesis requiring per-language validation.

Second, the varieties that degrade share identifiable characteristics. The typologically isolated variety and the least-resourced variety degraded, while Twi, the best-resourced variety, did not. Notably, Fante degraded despite belonging to the strongest transfer pair in the set, indicating that relatedness to the training pool does not protect a language whose share of that pool is small.

Third, typology-based similarity and taskspecific transfer behaviour are not interchangeable. Empirical cross-lingual transfer identifies which language pairs interact more accurately than URIEL similarity does, yet it does not predict which languages benefit under joint training.

Fourth, and methodologically, the reliability of results in this regime is itself at stake. Our own earlier single-run analysis reported the opposite headline conclusion. For the two varieties on which that analysis showed its largest gains, Ga and Twi, the monolingual models were roughly five to thirty times less stable across seeds than the multilingual model. Our results suggest that multi-seed reporting for both conditions should be considered standard practice for sub-twenty-hour multilingual comparisons.

These observations arise from experiments on MGhana-ST using Whisper-small under the training configurations considered here, and should be interpreted as empirical findings within this experimental setting rather than universal properties of multilingual speech translation.

## Limitations

Scale and language imbalance. The experimental subset contains only about 16.1 hours of paired speech and translation data. Twi accounts for the largest share (6.21h of 12.78h of training audio), while Fante is the least represented at 1.62 hours. Model comparisons are therefore sensitive to data composition.

Unavailable audio and subset composition. As reported in Section 3.1, 12.6% of annotated segments reference audio not present in our working copy (39.9% for Ewe and 30.4% for Fante). We have not established whether these files are absent upstream or only locally, nor whether their absence correlates with speaker, recording session, or collection batch. This does not compromise the within-language M1–M2 contrasts, since all configurations train and evaluate on identical data, but it does limit what the Ewe and Fante results can be said to represent about those varieties in general.

Incomplete multi-seed coverage. The auxiliary conditioning experiments and the transfer matrix are single-run. Given seed standard deviations of up to 2.20 BLEU, these should be treated as exploratory. Extending multi-seed coverage to all configurations is a priority for future work.

Split variance not measured. All runs share a single data partition. Our error bars therefore capture training variance only. Variance across data partitions is not measured.

Corpus provenance. MGhana-ST aggregates two source corpora that differ in domain, recording channel, speaking style, and utterance duration, unevenly distributed across varieties. Language identity is therefore partially confounded with acoustic and domain characteristics. We use Ga as a partial control to bound this effect, but a clean separation would require matched-domain data for all four varieties.

Representational coverage. MGhana-ST covers four varieties from one region of Ghana and should not be taken to represent Ghanaian languages broadly. Domain coverage is narrow, dominated by financial-technology scenarios for three varieties. Dialectal coverage within Twi is not characterised (Akuapem and Asante are merged). Speaker demographic distributions for the subset are not reported.

Encoder depth selection. The encoder finetuning depth was chosen from a single-run sweep scored on the Ga test split of a preliminary partition (Appendix A). The selected depth is held fixed across all configurations, so it does not favour M1 or M2, but it is not a held-out choice.

Utterance length and reference diversity. Three of the four varieties consist largely of single words and short phrases (mean duration 1.33– 1.90 s), and their English references repeat substantially. MGhana-ST does not support conclusions about sentence-level speech translation, and absolute BLEU is not comparable across its four varieties.

Bitwise reproducibility. Training runs are not bitwise deterministic, so individual runs cannot be reproduced exactly even at a fixed seed. The reported means and standard deviations are the reproducible quantities.

## Use of AI Assistants

AI assistants were used during the preparation of this paper for coding support, editorial refinement, and language editing. They were not used to generate experimental results, analyses, or conclusions, all of which are the authors’ own.

## Acknowledgments

We thank the annotators and collaborators who contributed to data collection, translation, and dataset preparation. We also acknowledge support from Ashesi University and AdwumaTech AI.

## References

Frank Lawrence Nii Adoquaye Acquaye, Insan-Aleksandr Latipov, and Attila Kertész-Farkas. 2023. Hypernym information and sentiment bias probing in distributed data representation. In Proceedings of the 2023 15th International Conference on Machine Learning and Computing.

Frank Lawrence Nii Adoquaye Acquaye, Eric George Parakal, Jesse Johnson, Kishankumar Bhimani, and Jochebed Afua Basil. 2025. MGhana-ST: A speech translation dataset for Ghanaian languages. https://huggingface.co/datasets/ adwumatech-ai/mghana-st. Hugging Face dataset.

Naveen Arivazhagan, Ankur Bapna, Orhan Firat, Dmitry Lepikhin, Melvin Johnson, Maxim Krikun, Mia Xu Chen, Yuan Cao, George F. Foster, Colin Cherry, Wolfgang Macherey, Zhifeng Chen, and

Yonghui Wu. 2019. Massively multilingual neural machine translation in the wild: Findings and challenges. Preprint, arXiv:1907.05019.

D. Asamoah Owusu, A. Korsah, B. Quartey, S. Nwolley Jnr., D. Sampah, D. Adjepon-Yamoah, and L. Omane Boateng. 2022. Financial inclusion speech dataset. https://github.com/Ashesi-Org/ Financial-Inclusion-Speech-Dataset.

Ankur Bapna, Colin Cherry, Yu Zhang, Ye Jia, Melvin Johnson, Yong Cheng, Simran Khanuja, Jason Riesa, and Alexis Conneau. 2022. mslam: Massively multilingual joint pre-training for speech and text. arXiv preprint arXiv:2202.01374.

Loïc Barrault, Yu-An Chung, Mariano Cora Meglioli, David Dale, Ning Dong, Paul-Ambroise Duquenne, Hady Elsahar, Hongyu Gong, Kevin Heffernan, John Hoffman, Christopher Klaiber, Pengwei Li, Daniel Licht, Jean Maillard, Alice Rakotoarison, Kaushik Ram Sadagopan, Guillaume Wenzek, Ethan Ye, Bapi Akula, and 48 others. 2023. Seamlessm4t: Massively multilingual & multimodal machine translation. arXiv preprint arXiv:2308.11596.

Alexandre Bérard, Olivier Pietquin, Laurent Besacier, and Christophe Servan. 2016. Listen and translate: A proof of concept for end-to-end speech-to-text translation. In NIPS Workshop on end-to-end learningfor speech and audio processing.

Alexis Conneau, Min Ma, Simran Khanuja, Yu Zhang, Vera Axelrod, Siddharth Dalmia, Jason Riesa, Clara Rivera, and Ankur Bapna. 2023. FLEURS: Few-shot learning evaluation of universal representations of speech. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 798–805.

Long Duong, Antonios Anastasopoulos, David Chiang, Steven Bird, and Trevor Cohn. 2016. An attentional model for speech translation without transcription. In Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies, pages 949–959. Association for Computational Linguistics.

Paul-Ambroise Duquenne, Hongyu Gong, Ning Dong, Jingfei Du, Ann Lee, Vedanuj Goswami, Changhan Wang, Juan Pino, Benoît Sagot, and Holger Schwenk. 2023. Speechmatrix: A large-scale mined corpus of multilingual speech-to-speech translations. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 16251–16269.

Patrick Littell, David R Mortensen, Ke Lin, Katherine Kairis, Carlisle Turner, and Lori Levin. 2017. Uriel and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors. In Proceedings of the 15th Conference of the European Chapter ofthe Associationfor Computational Linguistics: Volume 2, Short Papers, pages 8–14. Association for Computational Linguistics.

Josh Meyer, David Ifeoluwa Adelani, Edresson Casanova, Alp Öktem, and 1 others. 2022. BibleTTS: a large, high-fidelity, multilingual, and uniquely African speech corpus. In Proc. Interspeech 2022, pages 2383–2387.

Tobi Olatunji, Tejumade Afonja, Aditya Yadavalli, Chris Chinenye Emezue, and 1 others. 2023. AfriSpeech-200: Pan-african accented speech dataset for clinical and general domain ASR. Transactions of the Association for Computational Linguistics, 11:1669–1685.

Vineel Pratap, Andros Tjandra, Bowen Shi, Paden Tomasello, Arun Babu, Sayani Kundu, Ali Elkahky, Zhaoheng Ni, Apoorv Vyas, Maryam Fazel-Zarandi, Alexei Baevski, Yossi Adi, Xiaohui Zhang, Wei-Ning Hsu, Alexis Conneau, and Michael Auli. 2024. Scaling speech technology to 1,000+ languages. Journal ofMachine Learning Research, 25(97):1–52.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR.

Zirui Wang, Zachary C Lipton, and Yulia Tsvetkov. 2020. On negative interference in multilingual models: Findings and a meta-learning treatment. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4438–4450. Association for Computational Linguistics.

Isaac Wiafe, Jamal-Deen Abdulai, Akon Obu Ekpezu, Raynard Dodzi Helegah, Elikem Doe Atsakpo, Charles Nutrokpor, Fiifi Baffoe Payin Winful, and Kafui Kwashia Solaga. 2025. UGSpeechData: A Multilingual Speech Dataset of Ghanaian Languages.

## Appendix

## A Encoder Fine-tuning Depth Selection

Before the main experiments, we performed a sweep over the number of unfrozen top Whispersmall encoder layers, with $n \in \{ 0 , 2 , 4 , 6 , 1 2 \}$ , on the Ga subset only, using a preliminary partition (2.46h train, 0.29h val, 0.30h test) that predates the final experimental subset. Scores are on that partition’s test split.

Performance is relatively stable across $n \_ \in$ {0, 2, 4, 6}, with BLEU highest under full freezing $( n = 0 , 3 6 . 7 5 )$ and chrF highest at $n = 4 ( 5 3 . 3 1 )$ Unfreezing all 12 layers substantially reduces both metrics (BLEU 29.04, chrF 44.53), consistent with degradation of pretrained representations under extensive fine-tuning in a low-resource setting. We select n = 4 as a practical operating point because it achieves the highest chrF while maintaining BLEU close to the strongest observed values. Two caveats apply: the sweep predated preparation of the full multilingual data, and it is single-run, so the BLEU spread across $n \in \{ 0 , 2 , 4 , 6 \}$ (1.54) is smaller than the monolingual Ga seed standard deviation (1.63). The ranking among moderate depths is not resolvable from this sweep.

<table><tr><td>Unfrozen layers (n)</td><td>BLEU</td><td>chrF</td></tr><tr><td>0</td><td>36.75</td><td>49.59</td></tr><tr><td>2</td><td>35.21</td><td>50.81</td></tr><tr><td>4*</td><td>36.25</td><td>53.31</td></tr><tr><td>6</td><td>36.63</td><td>49.45</td></tr><tr><td>12</td><td>29.04</td><td>44.53</td></tr></table>

Table 6: Encoder fine-tuning depth sweep on the Ga test split of the preliminary partition (single-run). <sup>∗</sup>Depth selected for main experiments.

## B Segment Retention Audit

Not every annotated segment enters the experimental subset (Table 7). The largest exclusion is eventonly segments, whose translation consists solely of a bracketed event tag with no accompanying words: 48,811 segments (53.9% of the 90,583 annotated segments), at rates of 66.2% for Ga, 59.7% for Twi, 46.0% for Fante, and 29.5% for Ewe. The second is unavailable audio (11,445 segments, 12.6%), concentrated in Ewe (39.9%) and Fante (30.4%). Thirteen further segments are excluded for other reasons, leaving 30,314 retained segments. The full annotation set also contains segments labelled with languages other than the four varieties studied here; these are outside the experimental subset and are not counted in Table 7.

<table><tr><td>Category</td><td>Ga</td><td>Twi</td><td>Ewe</td><td>Fante</td></tr><tr><td>Annotated segs.</td><td>19,370</td><td>37,100</td><td>11,333</td><td>22,780</td></tr><tr><td>Audio missing</td><td>0</td><td>12</td><td>4,516</td><td>6,917</td></tr><tr><td>Event-only</td><td>12,822</td><td>22,156</td><td>3,346</td><td>10,487</td></tr><tr><td>Other</td><td>7</td><td>3</td><td>3</td><td>0</td></tr><tr><td>Retained</td><td>6,541</td><td>14,929</td><td>3,468</td><td>5,376</td></tr><tr><td>Retention rate</td><td>33.8%</td><td>40.2%</td><td>30.6%</td><td>23.6%</td></tr></table>

Table 7: Segment retention by exclusion category. Ten retained segments (Twi 3, Ewe 5, Fante 2) do not appear in the final splits, so the totals in Table 2 are slightly lower.

## C Cross-Lingual Transfer BLEU Matrix

Table 8 reports the full cross-lingual transfer evaluation, in which each M1 monolingual model is

evaluated on all four language test sets. These are single-run results at seed 42.
<table><tr><td rowspan="2">Model</td><td colspan="2">Ga</td><td colspan="2">Twi</td><td colspan="2">Ewe</td><td colspan="2">Fante</td></tr><tr><td>BLEU</td><td>chrF</td><td>BLEU</td><td>chrF</td><td>BLEU</td><td>chrF</td><td>BLEU</td><td>chrF</td></tr><tr><td>M1-ga</td><td>34.76</td><td>49.25</td><td>0.98</td><td>8.25</td><td>0.07</td><td>7.13</td><td>1.48</td><td>9.24</td></tr><tr><td>M1-twi</td><td>0.38</td><td>8.11</td><td>17.91</td><td>29.70</td><td>0.17</td><td>9.21</td><td>23.36</td><td>43.17</td></tr><tr><td>M1-ewe</td><td>0.18</td><td>8.06</td><td>0.11</td><td>9.87</td><td>14.06</td><td>31.01</td><td>0.23</td><td>6.94</td></tr><tr><td>M1-fante</td><td>0.97</td><td>9.06</td><td>12.94</td><td>23.72</td><td>0.13</td><td>9.74</td><td>45.73</td><td>59.83</td></tr></table>

Table 8: Cross-lingual transfer matrix: each M1 model is evaluated on all four language test sets (single-run, seed 42). Bold diagonal entries are in-language results. Off-diagonal entries are zero-shot transfers.

## D Auxiliary Conditioning

These results are exploratory and single-run at seed 42. Because Section 5.2 establishes that single-run differences at this data scale can exceed the effects of interest, we report them as directional evidence only. All comparisons are made against the singlerun M2 baseline at seed 42 (Ga 37.71, Twi 19.59, Ewe 9.60, Fante 46.19; corpus 23.72).

We evaluate two auxiliary objectives. M3 applies Domain-Adversarial Training (DANN) with gradient reversal to encourage language-invariant encoder representations (λ = 0.1). M4 applies Language Identification (LID) supervision to encourage language-discriminative representations (α = 0.1 in Equation 1).

Invariance redistributes. DANN improves Ewe substantially, from 9.60 to 13.23 BLEU (+3.63). However, Fante declines by 4.57 BLEU and Twi by 0.83, while Ga is unchanged. Corpus-level BLEU is slightly below M2, so DANN reallocates performance rather than improving it uniformly.

Discriminativeness does not help. LID conditioning underperforms M2 for all four languages (Ga −1.59, Fante −1.76, Ewe −0.64, Twi −0.51), with corpus BLEU 0.76 below M2. Encouraging the encoder to represent language identity more sharply does not mitigate interference at this seed. We have not verified that either objective changes what the encoder represents; probing the encoder states for language identity, in the manner of prior probing work on distributed representations (Acquaye et al., 2023), would test this directly.

## E Target-Side Overlap and Prompt Reuse

Reference repetition has two consequences. First, Fante reference repetition is a direct consequence of the collection protocol: speakers of a given dialect recorded a shared list of approximately 130 prompt sentences. Because our splits are constructed at the audio-file level, a prompt sentence recorded by one speaker may appear in training while the same sentence recorded by another appears in test. Scores on Ga, Twi, and Fante reflect a setting in which the space of target sentences is small and partially shared across splits, and should be read as performance on a closed prompt inventory. Ewe, drawn from image-description recordings, is not subject to this and shows correspondingly low reference repetition (1.5%).

<table><tr><td>Run</td><td>Ga</td><td>Twi</td><td>Ewe</td><td>Fante</td><td>Corpus</td></tr><tr><td>BLEU</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>M2 (multi)</td><td>37.71</td><td>19.59</td><td>9.60</td><td>46.19</td><td>23.72</td></tr><tr><td>M3 (DANN)</td><td>37.74</td><td>18.76</td><td>13.23</td><td>41.62</td><td>23.38</td></tr><tr><td>M4 (LID)</td><td>36.12</td><td>19.08</td><td>8.96</td><td>44.43</td><td>22.96</td></tr></table>

Table 9: Auxiliary conditioning results (single-run, seed 42). All comparisons are against M2 at the same seed, not against the three-seed mean in the main text.

Second, reference repetition raises a natural objection to using transfer BLEU as a relatedness signal (Section 4.4): if Twi and Fante share a domain and a stock of high-frequency English formulae, strong Twi–Fante transfer might reflect a shared phrase inventory. Ga shares the same source corpus and domain as Twi and Fante, with comparable reference repetition (27.2% for Ga against 27.4% for Twi), yet Ga–Twi transfer is negligible (0.98 and 0.38 BLEU). Domain overlap and reference repetition are thus not sufficient to generate transfer; the Akan relationship shared by Twi and Fante is what distinguishes the two cases.

## F Per-Seed Results

Table 10 reports the individual seed results underlying the means in Table 5.

<table><tr><td>Cfg</td><td>Seed</td><td>Ga Twi</td><td>Ewe</td><td>Fante</td></tr><tr><td>BLEU</td><td></td><td></td><td></td><td></td></tr><tr><td>M1</td><td>1</td><td>36.83 17.25</td><td>16.10</td><td>46.82</td></tr><tr><td>M1</td><td>2</td><td>33.57 21.31</td><td>15.31</td><td>47.60</td></tr><tr><td>M1</td><td>3</td><td>35.35 17.80</td><td>15.02</td><td>48.40</td></tr><tr><td>M2 1</td><td>35.98</td><td>18.78</td><td>7.21</td><td>41.48</td></tr><tr><td>M2</td><td>2</td><td>35.35 18.92</td><td>9.44</td><td>44.13</td></tr><tr><td>M2</td><td>3</td><td>35.94</td><td>18.85 8.81</td><td>41.89</td></tr></table>

Table 10: Per-seed BLEU for M1 and M2, split held fixed. Corpus-level M2 BLEU is 22.30, 22.81, and 22.39 for seeds 1–3.

## G Training Details

Training hyperparameters: batch size 16 for 60 epochs, learning rate 0.0005, 500 warmup steps, gradient clipping 1.0. Checkpoints saved every 5 epochs. Top 4 Whisper-small encoder layers finetuned; lower layers frozen. Decoder trained from scratch. Explicit language conditioning disabled for M1/M2 baselines. The full 12-layer Whispersmall encoder is used as the speech backbone. Decoding uses beam size 4. The annotation state corresponds to commit 4996b709abd584464a6c fb7bd256504fd5eed7a0 (25 March 2026). Environment: Python 3.12.12, PyTorch 2.10.0+cu126, Transformers 5.1.0, on a single NVIDIA A100- SXM4-40GB GPU.