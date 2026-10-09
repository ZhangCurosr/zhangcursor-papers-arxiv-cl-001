# PROSODY-TO-TEXT: PREDICTING TEXT FROM LOW-PASS FILTERED SPEECH

David Portesˇ and Ales Horˇ ak´

Natural Language Processing Centre   
Faculty of Informatics, Masaryk University   
Botanicka 68a, 602 00 Brno, Czech Republic xportes,hales@fi.muni.cz

## ABSTRACT

While predicting prosody from text is an established task in the field, the opposite direction, predicting text that fits a given prosodic pattern, remains largely overlooked. We find this unfortunate, because this opposite direction could lead to some very interesting use cases. Therefore, in this paper, we make the first steps in the prosody-to-text direction by investigating how much of the original sentence can be recovered from its prosodic pattern. To this end, we fine-tune the Whisper model using only the 12 lowest Mel bins (low-pass filter with approximately 450Hz cutoff), and obtain surprisingly accurate results (WER 36%), with 10% of utterances being recovered perfectly, and 40% of utterances having Word Error Rate at or below 25%. We also find that, given the correct prefix, the next token was predicted correctly in 79% of cases. Our results suggest that the relationship between low-frequency speech features and lexical content is much stronger than previously thought, and we believe that directing more attention to this topic might open the door to new applications, such as using prosody to guide text generation of modern LLMs.

Index Terms— prosody, low-pass filter, Automatic Speech Recognition, Whisper, Conformer

## 1. INTRODUCTION

The text-prosody relationship has been extensively studied in the literature, however, the overwhelming majority of research is concerned with predicting prosody from text, due to its use in speech synthesis [1, 2, 3]. The opposite direction, that is, predicting text from prosody, remains largely unexplored. While past research indicates that spoken-word recognition relies mainly on segmental features, with prosody playing a supplementary role [4, 5], we were curious whether contemporary Automatic Speech Recognition models (ASR) could recover more than is generally thought possible from prosody alone. If so, such prosody-to-text model could be used to recover unintelligible or corrupted speech, help transcribe dysarthric speech, or be used by researchers to study the prosody-text relationship from a new angle. Additionally, even a partial result could be useful. For example, one could use such model in combination with modern LLMs, in order to steer the model towards texts that fit a desired prosody on a topic of user’s choice [6, 7], thus unlocking a new way of human-LLM interaction.

Therefore, in this paper, we investigate how much lexical content can modern speech recognition models recover from the prosodic pattern of an utterance. As a first estimate, we fine-tune the Whisper-Medium model [8] using only the lowest 12 Mel bins out of 80, which corresponds roughly to a low-pass filter with a 450Hz cutoff, on 750k samples from the Yodas dataset [9]. Low-pass filtering at approximately 400–500 Hz is a common technique to delexicalize speech while retaining prosodic information [10, 11, 12].

We obtain a surprisingly low Word Error Rate on the held out Yodas test set (WER 36%), and on a separate Animevox [13] dataset (WER 60%). In both datasets, there was a non-trivial percentage of sentences which were transcribed perfectly (10%, 5%), and, in some cases, even sentences longer than 12 words were transcribed correctly or with minimum errors. On the Yodas test set, given the correct prefix, the model was able to predict the next token correctly 79% of the time, and 96% of the time within the top 10.

In order to determine how much of this result can be attributed directly to prosodic features, we trained a Conformerbased model [14] to generate artificial 12-bin Mel spectrograms based exclusively on the Fundamental frequency (f<sub>0</sub>) and Energy derived from the speech. We then continued finetuning our Whisper model on these generated spectrograms. We found that, on the Yodas test set, using artificial spectrograms caused WER to increase to 69%. However, given the correct prefix, the model was still able to predict the next token correctly 64% of the time, and 92% of the time in the top 10 predictions.

While we were not able to attribute our results directly to the prosodic features, we did find that the relationship between low-frequency features of speech and its lexical content is stronger than generally believed. Our fine-tuned Whisper-Medium model, which has much fewer parameters than State of the art models, was able to provide meaningful reconstruction on a significant portion of sentences without access to most segmental features. Our findings suggest that this might be an underexplored area with great potential, and we aim to investigate it further using additional prosodic features and larger datasets. In order to facilitate further research in this area, we publish our models and code<sup>1</sup>.

![](images/b9bc9d8a9f738fd228da955723a4a422a15184b711d4e29913f335ae399a3f44.jpg)  
Fig. 1. The global oracle WER@k curves.

## 2. RELATED WORK

Predicting text from prosody is an extremely underexplored topic. In fact, we were not able to find any works dealing with predicting text from prosody, nor low-frequency features of speech. The closest branch of research is probably the Temporal Patterns (TRAPS) [15] approach to Automatic Speech Recognition (ASR), which proposes to conduct ASR based on long temporal patterns within different critical frequency bands, and combine bands with highest signal-to-noise ratio into the final prediction. This implicitly includes our problem, which would be roughly equivalent to the lowest critical band. However, most of the research is dated, and we were not able to find any breakdown of accuracy by band, which we could compare to our results.

From the more recent works, the most similar is probably [16], where its authors estimate the mutual information between prosody and text. However, they approach the problem from the text to prosody direction, by training a LLM to predict prosody features, and they do not investigate the relationship from the prosody to text direction.

There is a number of works that treat prosody as supplementary information to an established lexical-related task, such as speech translation [17, 18], or spoken language modeling [19, 20]. However, prosody is not used as the sole input in these papers. The only exception we are aware of is [21], which investigates the role of prosody as supplementary information in Spoken Question Answering. Here, the authors also report the performance using only the prosodic condition, noting that the results were weaker but still very much above chance.

![](images/798d9d27ff28d5fa6870ce4ebc8eabaff0588b00feebc3a8b43d1f855e664104.jpg)  
Fig. 2. Cumulative distribution over WER.

## 3. METHODS

In this section, we describe the training of our models, and the evaluation methods used.

## 3.1. Choice of model and features

The choice of the Whisper model was driven mainly by the fact that it is one of the most prominent ASR models, trained on a large amount of data (680k hours). While newer models could be used, it would be difficult to find a test dataset created after their release, in order to rule out any contamination. We used the Whisper-Medium version, since we found that larger versions require much bigger batch sizes for the fine-tuning to converge.

As our representation of prosody, we used the Fundamental frequency (f<sub>0</sub>) and Energy, as these are the most prominent prosodic features, often used together in speech synthesis methods [1, 2]. Fundamental frequency represents the perceived pitch of the voice, while energy reflects the loudness of the speech signal. We decided to avoid using any text-derived features, such as duration, as these would not be available in potential real applications. Additional spectral features are left to future work.

## 3.2. Training: Whisper-Low-pass (LP)

In order to predict text based only on the low-frequency features of speech, we fine-tuned the Whisper-Medium model using only the lowest 12 Mel bins out of the 80 bins generated by the Whisper preprocessor. We did this by setting all bins above 12 to zero, in order to preserve the shape of the data. In the Whisper Mel spectrogram, this means that the highest bin we used was centered at 446Hz, tapering down towards 484Hz, where the 13th bin is centered.

We fine-tuned 750k English samples from the Yodas dataset [9], using only samples with duration below 30 s, with a 3k validation set. We stopped fine-tuning after 1 epoch due to signs of overfitting. Fine-tuning was conducted on a NVIDIA A100 GPU using batch size 64 with 4 gradient accumulation steps, making the effective batch size 256. The process took 12 hours.

![](images/8d2643c5f9b0526fb6096eeaef5f4888c3cdfddeaf43577edc23f0a9fb03a5c9.jpg)  
Fig. 3. Percentage of times the correct token was present in top k predictions given correct prefix.

## 3.3. Training: Whisper-Prosody

Since the Whisper model expects Mel spectrograms as input, we cannot feed it the $f _ { 0 }$ and Energy prosodic features directly. For this reason, we first trained a Conformer model to generate 750k artificial 12-bin Mel spectrograms based solely on these features, using data from the Yodas dataset. The spectrograms were zero-padded to match the 80 bin shape, and they were subsequently used to further fine-tune the Whisper-Low-Pass checkpoint for an additional epoch. Fine-tuning was conducted on a NVIDIA A100 GPU, with the same parameters and training time as in the Whisper-Low-pass case.

## 3.4. Training: Conformer

To train the Conformer model, we extracted $f _ { 0 }$ using YAAPT algorithm [22], with the hop size of 5ms and frame size of 20ms on 16kHz sampled audio, resulting in $f _ { 0 }$ signal sampled at 200Hz. The unvoiced frames were interpolated over using the interpolation algorithm in YAAPT. Energy was extracted using the PRAAT package [23], also at 200Hz. We trained for two epochs on 750k samples from the Yodas dataset using a batch size of 256. The training took 48 hours on an NVIDIA A40 GPU. The total number of parameters of the Conformer model was 7,645,198.

## 3.5. Evaluation

We evaluated each model on 3000 held out samples of the Yodas dataset with duration under 30s. The Yodas dataset was released two years after the Whisper models, however, since the data in the dataset might be older, test set contamination with the non-public Whisper training set cannot be ruled out. For this reason, we also evaluated 1,000 samples from the AnimeVox dataset [13], which was made using the show Frieren: Beyond Journey’s end (2023), released a year after the Whisper model checkpoints. For the Whisper-Prosody model, the input Mel spectrograms were generated using the Conformer model based exclusively on the prosodic features $f _ { 0 }$ and Energy. We used beam search with 10 beams in all our evaluations. As a baseline, we use the Whisper-Large model without any fine-tuning, which was evaluated using the Lowpass condition.

![](images/2daa7aa483f5b7c67f2063f24f242486eae4bd1d236b73397dde2d0934956156.jpg)

![](images/80c6d415d27305a7f15be772de7bee51908fecd0affa90121d40b972d5c7bdf8.jpg)  
Fig. 4. Sample reconstruction of the lowest 12 bins of the Whisper Mel spectrogram by our Conformer model.

Table 1. Global Oracle WER@k (%)
<table><tr><td colspan="4">Yodas</td><td colspan="3">AnimeVox</td></tr><tr><td>WER</td><td>LP</td><td>Prosody</td><td> $\mathrm { B a s e } _ { \mathrm { L P } }$ </td><td>LP</td><td>Prosody</td><td> ${ \mathrm { B a s e } } _ { \mathrm { L P } }$ </td></tr><tr><td>@1</td><td>36.7</td><td>68.9</td><td>100.1</td><td>60.0</td><td>82.4</td><td>113.8</td></tr><tr><td>@2</td><td>34.6</td><td>66.0</td><td>98.2</td><td>55.2</td><td>76.7</td><td>111.5</td></tr><tr><td>@5</td><td>31.9</td><td>62.9</td><td>95.4</td><td>49.5</td><td>70.2</td><td>107.2</td></tr><tr><td>@10</td><td>30.4</td><td>61.0</td><td>93.2</td><td>45.9</td><td>66.4</td><td>103.7</td></tr></table>

Table 2. Next token distribution metrics
<table><tr><td></td><td colspan="3">Yodas</td><td colspan="3">AnimeVox</td></tr><tr><td>Metric</td><td>LP</td><td>Prosody</td><td> $\mathrm { B a s e } _ { \mathrm { L P } }$ </td><td>LP</td><td>Prosody</td><td>BaseLP</td></tr><tr><td>Num Samples</td><td>3000</td><td>3000</td><td>3000</td><td>1000</td><td>1000</td><td>1000</td></tr><tr><td>Top1 Pct</td><td>78.80</td><td>64.90</td><td>44.50</td><td>66.80</td><td>58.50</td><td>34.20</td></tr><tr><td>Avg Rank</td><td>17.33</td><td>39.87</td><td>532.44</td><td>81.52</td><td>107.14</td><td>959.82</td></tr><tr><td>Median Rank</td><td>1</td><td>1</td><td>2</td><td>1</td><td>1</td><td>2</td></tr><tr><td>Perplexity</td><td>2.61</td><td>4.91</td><td>29.05</td><td>3.78</td><td>5.79</td><td>48.54</td></tr></table>

## 4. RESULTS AND DISCUSSION

Word Error Rates (WER) for each model-dataset combination are shown in Table 1. Figure 1 visualizes the distribution of global WER values across all evaluated WER@k settings, while Figure 2 shows the cumulative distribution function of WER values. In Table 2, we analyze several properties of the next-token distribution given the correct prefix. Figure 3 shows the percentage of times the correct token appeared within the top-k predictions, given the correct prefix.

![](images/fd835f7a332c2cb7647191b8ffb007974afbf96a50b695bcb2c57ef3cd20c9b7.jpg)  
Fig. 5. Sample sentences with WER near the mean for each model. P is prediction, R is reference. Red indicates errors.

To illustrate average reconstruction quality, Figure 5 presents representative examples whose WER is close to the average in each model-dataset combination.

Regarding the quality of Mel spectrogram generation, the Mean Squared Error (MSE) between the predicted and ground-truth Mel spectrograms on the test set was 0.0322. Figure 4 shows a representative reconstruction.

We find that, given the correct prefix, the Whisper-Lowpass model predicted the correct next token approximately 79% of the time on the Yodas dataset and 67% on the AnimeVox dataset (Table 2). While this was insufficient for reliable reconstruction, the WER was very unstable, and a nontrivial portion of sentences were even reconstructed with 0% WER (10%/5%). There was also a substantial gap between the two datasets, which we suspect to be due to the domain mismatch between the fine-tuning dataset consisting of Youtube videos (Yodas) and Animevox, containing mostly animated voice acting in a fantasy setting.

We were not able to attribute these results purely to the prosodic features of $f _ { 0 }$ and Energy, as WER increased substantially when using the Whisper-Prosody pipeline. This may indicate that prosodic information was only partially responsible for the performance of the Whisper-Low-pass model, or that additional prosodic features are required. It may also have been caused by insufficient training data, as Whisper-Prosody relies on a more complex two-stage pipeline involving the Conformer model. Even so, the correct next-token rate of the Whisper-Prosody-based model remained above 58% on both datasets, suggesting the presence of a non-trivial relationship between $f _ { 0 } ,$ , Energy and lexical content.

Our results support the base hypothesis behind TRAPS [15] that individual frequency bands carry large portions meaning redundantly, at least in the lowest critical band. In this context, it would be interesting to see whether current state of the art ASR models draw on the low-frequency features to aid their predictions in cases where the signal in other bands is corrupted.

When compared to the base Whisper models, we find that this is not a capability that the Whisper model possessed natively, as our fine-tuned Whisper-Medium model’s WER (36 / 60%) is much lower than the WER of even the Whisper-Large model (WER 100 / 113%), across both datasets.

## 5. LIMITATIONS

The main limitation of our study is that, in case of the Yodas dataset, it is not possible to rule out test set contamination with the original Whisper training set, since it is not publicly available. We have also extracted the test set in an interleaved manner with the training set, meaning that the domain is mostly the same across both sets, and using a disjoint set of the Yodas dataset might yield worse results. However, the results on AnimeVox are guaranteed to be free from contamination, since the data comes from a TV show recorded one year after the Whisper checkpoints last modification date on the Huggingface website. The AnimeVox dataset is not an established dataset in the field, however, we were hard-pressed to find any other datasets with samples that are guaranteed to have been created after the Whisper model cutoff. The data used from the AnimeVox dataset also encompass only a single female speaker.

## 6. CONCLUSION

Our results reveal a surprisingly strong relationship between low-frequency features of speech and its lexical content. On the AnimeVox dataset, using only the lowest 12 Mel bins (450Hz), our fine-tuned Whisper-Medium model was able to correctly transcribe 5% of utterances, with 17% of utterances having Word Error Rate at or below 25%. Our results were even better on the Yodas test set, with 10% fully correct utterances, and 40% utterances at or below 25% WER. However, in this case, it is not possible to rule out test set contamination with the original Whisper training set. While the prosody-only results were weaker, there was still indication of a nontrivial prosody-text relationship, as our model’s next token predictions were correct in more than 58 percent of cases, given the correct prefix. All our results were obtained using the Whisper-Medium model, fine-tuned on very small data compared to today’s industrial standard, suggesting that there is still potential to improve the performance further, which could lead to new original use cases and applications. As future work, we plan to add additional prosodic features to see whether the performance on low-pass filtered data can be matched.

## Acknowledgments

Computational resources were provided by the e-INFRA CZ project (ID:90254), supported by the Ministry of Education, Youth and Sports of the Czech Republic.

## 7. REFERENCES

[1] Julio Cesar Galdino et al., “The evaluation of prosody in speech synthesis: a systematic review,” Journal ofthe Brazilian Computer Society, vol. 31, no. 1, pp. 465–486, 2025.

[2] Yi Ren et al., “Fastspeech 2: Fast and high-quality endto-end text to speech,” 2022, arXiv 2006.04558.

[3] Tadashi Ogura et al., “GST-BERT-TTS: Prosody Prediction Without Accentual Labels For Multi-Speaker TTS Using BERT With Global Style Tokens,” in Interspeech 2025, 2025, pp. 444–448.

[4] Andres Bux ´ o-Lugo, “Integration of contrastive prosody´ and segmental information during spoken word recognition,” Auditory Perception & Cognition, vol. 9, no. 1, pp. 43–63, 2026.

[5] Jie Chi, Maureen de Seyssel, and Natalie Schluter, “The role of prosody in spoken question answering,” in Findings of the Association for Computational Linguistics: NAACL 2025, 2025, pp. 8483–8494.

[6] David Portes, “Towards using speech melody to guideˇ large language models,” in Proceedings of Recent Advances in Slavonic Natural Language Processing, RASLAN 2023, 2023, pp. 133–141.

[7] David Portes and Ale ˇ s Hor ˇ ak, “Learning Optimal´ Prosody Embedding Codebook based on F0 and Energy,” in Proc. Interspeech 2025, 2025, pp. 4728–4732.

[8] Alec Radford et al., “Robust speech recognition via large-scale weak supervision,” 2022.

[9] Xinjian Li et al., “Yodas: Youtube-oriented dataset for audio and speech,” in 2023 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2023, pp. 1–8.

[10] Nicolas Audibert, Francesca Carbone, Maud Champagne-Lavau, Aurelien Said Housseini, and´ Caterina Petrone, “Evaluation of delexicalization methods for research on emotional speech,” in Interspeech 2023, 2023, pp. 2618–2622.

[11] Monja A. Knoll et al., “Effects of low-pass filtering on the judgment of vocal affect in speech directed to infants, adults and foreigners,” Speech Communication, vol. 51, no. 3, pp. 210–216, 2009.

[12] Sydney Lolli et al., “Sound frequency affects speech emotion perception: results from congenital amusia,” Frontiers in Psychology, vol. Volume 6 - 2015, 2015.

[13] Taresh Rajput, “Animevox,” 2025, https:// github.com/taresh18/AnimeVox.

[14] Anmol Gulati et al., “Conformer: Convolutionaugmented Transformer for Speech Recognition,” in Interspeech 2020, 2020, pp. 5036–5040.

[15] Hynek Heˇrmansky, “Traps- classifiers of temporal pat-´ terns,” in Proc. International Conference on Spoken Language Processing (ICSLP), Sydney, 1998, p. 5.

[16] Lukas Wolf et al., “Quantifying the redundancy between prosody and text,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, Dec. 2023, pp. 9765–9784, ACL.

[17] Ioannis Tsiamas, Matthias Sperber, Andrew Finch, and Sarthak Garg, “Speech is more than words: Do speechto-text translation systems leverage prosody?,” in Proceedings of the Ninth Conference on Machine Translation, USA, Nov. 2024, pp. 1235–1257, ACL.

[18] Giulio Zhou, Tsz Kin Lam, Alexandra Birch, and Barry Haddow, “Prosody in cascade and direct speech-to-text translation: a case study on Korean wh-phrases,” in Findings ofthe Associationfor Computational Linguistics: EACL 2024, Malta, Mar. 2024, pp. 674–683, ACL.

[19] Eugene Kharitonov et al., “Text-free prosody-aware generative spoken language modeling,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Dublin, Ireland, May 2022, pp. 8666–8681, ACL.

[20] Heeseung Kim et al., “Paralinguistics-aware speechempowered large language models for natural conversation,” in Proceedings of the 38th International Conference on Neural Information Processing Systems, Red Hook, NY, USA, 2024, NIPS ’24, Curran Associates Inc.

[21] Jie Chi, Maureen de Seyssel, and Natalie Schluter, “The role of prosody in spoken question answering,” in Findings of the Association for Computational Linguistics: NAACL 2025, Albuquerque, New Mexico, Apr. 2025, pp. 8483–8494, ACL.

[22] Kavita Kasi and Stephen A. Zahorian, “Yet Another Algorithm for Pitch Tracking,” in IEEE International Conference on Acoustics Speech and Signal Processing, Orlando, FL, USA, May 2002, pp. I–361–I–364, IEEE.

[23] Paul Boersma and Vincent Van Heuven, “Speak and unspeak with praat,” Glot International, vol. 5, no. 9/10, pp. 341–347, 2001.