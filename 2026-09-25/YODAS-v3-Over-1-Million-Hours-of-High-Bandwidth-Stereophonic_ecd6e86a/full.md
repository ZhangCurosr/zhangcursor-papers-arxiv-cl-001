# YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech

William Chen<sup>1,∗</sup>, Shinnosuke Takamichi<sup>2,∗</sup>, Sayaka Shiota<sup>3</sup>, Satoru Fukayama<sup>4</sup>, Samuele Cornell<sup>1</sup>, Shinji Watanabe<sup>1</sup>

<sup>1</sup> Carnegie Mellon University, USA, <sup>2</sup> Keio University, Japan <sup>3</sup> Tokyo Metropolitan University, Japan

<sup>4</sup> National Institute of Advanced Industrial Science and Technology (AIST), Japan

williamchen@cmu.edu, shinnosuke takamichi@keio.jp, swatanab@andrew.cmu.edu

## Abstract

We present YODAS v3, a weakly-labeled speech corpus containing over 1.1 million hours<sup>1</sup> of 48kHz multi-channel audio in 147 languages, released under a CC BY 3.0 license. YODAS v3 is not only the largest open speech dataset to date, but also the first truly large-scale speech corpus with high-fidelity stereo audio. We first provide the collection methodology for the corpus, where we introduce new techniques for gathering languagebalanced speech data. The effectiveness of our approach is shown by the language distribution of the crawled data: 22 languages in YODAS v3 have over 10K hours and 73 languages have over 5K hours of data. We then conduct extensive analyses on the composition of the data, such as the distribution of languages, audio quality, and transcription quality. Finally, we train baseline speech recognition and neural codec models to show the effectiveness of the dataset. Download at https: //huggingface.co/datasets/espnet/yodas3. Index Terms: multilingual, data, stereo, large-scale

## 1. Introduction

The rapid advancement of speech foundation models [1–14] is primarily driven by the availability of massive and diverse audio corpora [15–19]. While proprietary systems benefit from internal datasets that can scale to over 10 million hours [20, 21] of audio, previous work [5] has shown that the combination of nearly all open speech corpora only sums to a little over 1 million hours of audio as of late 2024<sup>2</sup>.

Furthermore, there is also a quality gap between open and propriety corpora. The vast majority of open-source speech datasets, such as YODAS v2 [17], VoxPopuli [15], MLS [23], and LibriLight [18] distribute data as monaural audio at sampling rates of 16 kHz or 24 kHz. While sufficient for standard ASR tasks, these constraints severely limit research in emerging domains that require high-resolution spatial audio, such as noisy multi-speaker ASR [24], high-fidelity audio codec modeling [25], full-band expressive TTS [26], spatial audio processing [27], and stereo speech enhancement [28]. Training models for these tasks requires high-resolution, multi-channel data, which has historically been scarce in open-access repositories.

Similar to how the release of corpora like VoxPopuli [15] and YODAS [17] helped enable new areas of research in open large-scale self-supervised learning [5,6,29,30] and large-scale supervised learning [2,8,31,32], respectively, our goal is to empower future research that requires high-fidelty and/or spatial audio. We therefore introduce YODAS v3, a dataset containing more than one million hours of multilingual speech, making it the largest open speech dataset. Unlike prior large-scale collections, YODAS v3 preserves quality by saving it in the original downloaded format: 48kHz multi-channel OPUS files. Compared to other large-scale corpora, YODAS v3 is well-balanced across languages: 22 languages in YODAS v3 each have over 10K hours of data and 73 languages have over 5K hours. YO-DAS v3 also includes weak supervision in the form of automatically generated language identity tags, speech recognition transcripts, and English translations, making it also the first open supervised corpus at such scale. We first outline our data collection method for YODAS v3, in which we propose crawling techniques to better collect data for medium and low resource languages, and leads to a language-balanced corpora composition. We then perform analyses of the crawled data, including language distribution and audio characteristics. Finally, we train models using the crawled data to show the viability of our approach for Automatic Speech Recognition (ASR) and Neural Audio Codec modeling.

## 2. Data Collection

## 2.1. Keyword Generation

We first create lists of keywords for YouTube video search. In YODAS v2 [17], a shared keyword list across all languages is constructed from a multilingual Wikipedia dump data. This approach biases the keywords towards that of high-resource languages (such as English). This also results in reduced search quality, since words from the wrong language are used in the search term. Instead, we apply the following language-specific filters to create a keyword list for each language. For a given language, we download that language’s Wikipedia dump file and filter keywords that satisfy all of the following conditions:

• Not consisting of only punctuation and numbers.

• Neither punctuation nor a digit is the first character.

• 3–30 characters in length.

• Neither a filename, DOI (digital object identifier), a Wikipedia Template file, nor a Wikipedia Request file.

The final step is to filter out excessively rare words so that the keyword list is a manageable size for searching. We first train a unigram-based subword tokenizer [36] on the filtered keywords for each language. We predefine a unigram vocabulary size: if we cannot construct a vocabulary at that size, we halve the size and try again until we successfully build a vocabulary. After training, we tokenize each keyword and compute its unigram likelihood. We finalized the keyword list for each language using only keywords with highest likelihood.

Table 1: A comparison of YODAS v3 with a other open large-scale speech datasets. YODAS v3 is the first public dataset to reach a scale ofover 1M hours while supporting high-bandwith stereo data.
<table><tr><td>Dataset</td><td>Languages</td><td>Size</td><td>Sampling Rate</td><td>Stereo</td><td>Labeled</td><td>License</td></tr><tr><td>Common Voice [33]</td><td>137</td><td>0.033M hours</td><td>48kHz</td><td>x</td><td>√</td><td>CC-0</td></tr><tr><td>MLS [23]</td><td>8</td><td>0.051M hours</td><td>16kHz</td><td>x</td><td>√</td><td>CC BY 4.0</td></tr><tr><td>FLEURS [34]</td><td>102</td><td>0.001M hours</td><td>16kHz</td><td>x</td><td>√</td><td>CC BY 2.5</td></tr><tr><td>VoxLingua107 [35]</td><td>107</td><td>0.007M hours</td><td>16kHz</td><td>x</td><td>√</td><td>CC BY 4.0</td></tr><tr><td>Librilight [18]</td><td>1</td><td>0.060M hours</td><td>16kHz</td><td>x</td><td>x</td><td>CC BY 4.0</td></tr><tr><td>VoxPopuli [15]</td><td>23</td><td>0.400M hours</td><td>16kHz</td><td>x</td><td>x</td><td>CC-0</td></tr><tr><td>Emilia [19]</td><td>6</td><td>0.100M hours</td><td>24kHz</td><td>x</td><td>√</td><td>CC BY NC 4.0</td></tr><tr><td>Unsupervised People&#x27;s Speech [22]</td><td>89</td><td>0.740M hours</td><td>48kHz</td><td>√</td><td>x</td><td>CC BY SA 4.0</td></tr><tr><td>YODAS v2 [17]</td><td>140</td><td>0.550M hours</td><td>24kHz</td><td>x</td><td>√</td><td>CC BY 3.0</td></tr><tr><td>YODAS v3 (this work)</td><td>147</td><td>1.100M hours</td><td>48kHz</td><td>√</td><td>√</td><td>CC BY 3.0</td></tr></table>

![](images/4179da68c8287c68ce63ed5748db04f3ef81c1898d00ed61382d994a8afdb069.jpg)  
Figure 1: Histogram of video upload date. “YODAS2” is the submission deadline ofASRU 2023.

## 2.2. Video Search

We search for videos IDs using the generated keyword list. Similar to YODAS v2, we limit the search to only videos that are uploaded under a Creative Commons license. We also prioritize more recently uploaded videos, since YouTube’s default search engine settings favor videos that are highly relevant to the keywords and therefore risks selecting only high-view-count videos. Prioritizing newer upload dates mitigates this issue and enables discovery of new videos.

Figure 1 shows the upload date histogram for the searched videos. Performing two video searches revealed that the setting effectively discovered newly uploaded videos not found in the previous search. To prevent data duplication, our final list of videos only contains IDs not found in YODAS v2, allowing both datasets to be combined in future work.

## 2.3. Downloading

We download the videos retrieved via the search and save their corresponding subtitles and audio. Audio is always downloaded and saved at the highest possible quality, which is 48kHz multichannel OPUS. We always download any user-uploaded manual subtitles in the video’s original language when they are available. Otherwise, we download the automatically generated YouTube subtitles in the original language. For non-English videos, we download the English subtitles as well, enabling future research on large-scale speech translation. All transcripts and translations are timestamped at the utterance level, allowing

Table 2: Examples ofthe metadatafor each downloaded video
<table><tr><td>Metadata</td><td>Example</td></tr><tr><td>Channels</td><td>2</td></tr><tr><td>Effective channels</td><td>2</td></tr><tr><td>Sampling rate</td><td>48kHz</td></tr><tr><td>Max bandwith</td><td>20kHz</td></tr><tr><td>Views</td><td>67</td></tr><tr><td>Language</td><td>French</td></tr><tr><td>Transcript</td><td>&quot;Bonjour mes amis...&quot;</td></tr><tr><td>Translation</td><td>&quot;Hello my friends...&quot;</td></tr><tr><td>Description</td><td>&quot;Une chanson pour...&quot;</td></tr></table>

YODAS v3 to be used for both short-form and long-form tasks. Finally, we also provide the descriptions of each video, which can be used for summarization or retrieval tasks. In general, we observed that manual subtitles were quite rare - they amount to only 3.7% of the total data (compared to 20% in YODAS v2). We hypothesize that this is a consequence of the improvements in automatic subtitling quality since the crawling of YODAS v2, making the human production of manual subtitles less necessary. An overview of the meta-data included with each downloaded audio is shown in Table 2.

## 3. Analysis

## 3.1. Language Distribution

A distribution of the data by language<sup>3</sup> is shown in Figure 2. The top 2 languages are Arabic and Welsh, each with roughly 25K hours of data. Japanese and English are the languages with the third and fourth amount of data at roughly 21K and 17K hours. Overall, 22 languages have over 10K hours of data and 73 languages have over 5K hours of data. This highlights the effectivness of our language-balanced video search approach (Section 2.1): the mean and median amount of data per language is 7400 and 5180 hours, respectively. Compared to most datasets, where English is often more than half of the dataset [17, 23, 33], English is less than 2% of YODAS v3.

## 3.2. Length Distribution

Having ample long-form data is critical for frontier audio processing tasks like conversational ASR [37], speech summarization, and dialogue modeling. In this section, we analyze the lengths of the original crawled data. Figure 3 shows the distribution of lengths per video, bucketed into minute-level groups. Overall, 1.7 million videos are longer than 5 minutes (1 million hours total) and 1.2 million are longer than 10 minutes (950K hours total). Interestingly, we observe that the curve follows a long-tail distribution, with more than 24% of the 4 million videos being less than one minute long. We attribute this observation to the rise in popularity of YouTube Shorts (short-form vertical videos similar to that of TikTok). Conversely, a manual qualitative analysis showed that a large amount of the longest videos (60+ minutes) are live-stream recordings. We leave further analysis these phenomena to future work.

![](images/ff223cab30b1482dfe8c38f2de5be05493c8f56a1a211f3b954b9f05d4a292a8.jpg)  
Figure 2: Data distribution ofthe top 50 languages in YODAS v3.

![](images/8cb5aaa3078a4ffaf69ebd8376b9623754ce3b5488e70f2ee1c3559b03e91baf.jpg)  
Figure 3: Distribution of videos by total length (log scale), bucketed into groups by minute.

## 3.3. True Bandwith Estimation

We estimated the effective audio bandwidth of all recordings in the corpus, which are stored in WebM/Opus format. This step is necessary because the Opus codec always produces 48 kHz PCM at decode time, regardless of its internal bandwidth mode (narrowband at 4 kHz, wideband at 8 kHz, up to fullband at most 20 kHz). The nominal sample rate of decoded audio therefore provides no information about actual spectral content. Additionally, recordings may originate from limited-bandwidth capture devices (e.g., built-in laptop microphones, USB headsets) whose effective frequency response could fall well below their digitization rate. For each file, we computed the shorttime Fourier transform (STFT) with an 8192-sample Hann window and 50% overlap. For each STFT frame, we identified the highest frequency bin whose magnitude exceeded 0.5% of the frame’s peak magnitude. Silent frames (root mean square (RMS) ≤ 10<sup>−6</sup>) were skipped. The estimated bandwidth for a recording is the maximum over all these analyzed frames. For recordings shorter than 5 minutes, the entire audio was decoded and analyzed. For longer recordings, five 60-second segments were drawn at random positions and the maximum bandwidth estimate across segments was retained. This estimated maximum frequency was mapped to the smallest standard sample rate whose Nyquist frequency equals or exceeds it, from the set 8, 16, 22.05, 24, 32, 44.1 kHz. This mirrors the classification used in the URGENT 2025 challenge [38]. The results are reported in Figure 4. We can see that most of the data (67.3%) has an effective bandwidth of 44.1 kHz and 92.5% of the data is above 32 kHz.

![](images/adff23480d98e69142165921f135eaf5ac1ebf53aeb6dec5960f94c44a094499.jpg)  
Figure 4: YODAS v3 data distribution by estimated effective bandwidth; over 92% ofrecordings exceed 32 kHz.

It should be noted that such bandwidth analysis is rarely conducted in prior dataset releases, despite being critical for understanding true audio fidelity. For instance, [38] found that CommonVoice and DNS5 LibriVox have effectively all samples (∼100% each) with mismatched bandwidths, while even LibriTTS [39] is affected in ∼25% of samples. Our analysis confirms that YODAS v3 is genuinely high-bandwidth, making it uniquely suited for tasks that require wideband audio.

## 3.4. Audio Channels

We also examine if the crawled data is truly multi-channel, or if the channels are in-fact duplicates of each other. For each pair of channels in an audio file, we subtract them from each other and compute the Root Mean Square (RMS) of the residual normalized by the average RMS of the two channels. We apply a threshold of 1e-3 to the result - values lower than the threshold indicate that the two compared channels are effectively identical. The distribution of results, in terms of the effective channel count, are shown in Figure 5. We find that over 71% of all audio files (about 780K hours) are truly multi-channel, with the vast majority in being stereo data. We note that there is also a small amount of data that are effectively 3-4 channels - a result of the inclusion of 5.1 surround sound audio data.

![](images/7ab93c8aac4b9c11e6fe8163c8775519508e4dba867168a604def77dcf8ab709.jpg)  
Figure 5: Log-scale distribution of data by effective number of channels in each audiofile

## 4. Experiments

## 4.1. Speech Recognition

Setup: We adopt a similar experimental setup to [17] for probing the quality of large-scale data by training monolingual ASR models on a random 7 language subset. We use 4500 hours for English and roughly 100 hours each for the other 6 languages. We process the data following the procedures proposed by OWSM v4 [31]: CTC-segmentation, language-based filtering, and CTC score-based filtering. We train multiple models, each corresponding to different thresholds of the CTC score quantile filter: $\theta _ { C T C } = 0 . 0 0$ (no filtering), 0.10, 0.20, and 0.30. Models are intialized from OWSM v4 base 102M [31] and trained for 40K steps using ESPnet [40]. We evaluate the models on 7 languages from CommonVoice, using Word Error Rate (WER) for the 6 alphabet-based languages and Character Error Rate (CER) for Japanese.

Results: The evaluation results are shown in Table 3. Unlike prior work on filtering and cleaning YODAS v2 [31], which observed inconsistent results when balancing between data scale and filtering strictness, we find that using more data (lower $\theta _ { C T C } )$ with YODAS v3 leads to improved results. In fact, WER almost always improves for all 7 languages as $\theta _ { C T C }$ decreases - the best models for each language (except for English) were trained without any score-based filtering, showing that YODAS v3 is directly usable out-of-the-box. This is a large improvement from YODAS v2, which required signficant filtering to avoid failed training runs (100+ WER). We hypothesize that this may be due to higher-quality automatic transcripts, which reflects the improvements in ASR performance since the release of YODAS v2 almost 3 years ago.

## 4.2. Neural Audio Codecs

Setup: We train neural audio codecs on a random 1200 hour subset of the data at sample rates of 16kHz, 24kHz, and 48kHz. All models use the DAC architecture [41]. Models are trained using the official configurations from the ESPnet-Codec toolkit [42], allowing for a fully fair comparison with models trained using existing datasets [42, 43]. Specifically, we compare against models trained on LibriTTS [43] (585 hours of English read speech)and AMUSE (33K hours of multilingual speech, music, and sound) [42]. We use the VERSA toolkit [44] for evaluation, measuring Short-Time Objective Intelligibility (STOI) [45] and WavLM embedding-based speaker similarity [46] on a 1000 video subset of YODAS v3 for in-domain testing and LibriTTS [43] for out-of-domain evaluation.

Table 3: WERs on different languages in Common Voice of ASR models trained on different datafiltering thresholds.
<table><tr><td>Threshold</td><td> $\mathrm { E n g . }$ </td><td>Deu.</td><td> $\mathrm { F r a . }$ </td><td> $\operatorname { J a p . }$ </td><td> $\mathrm { P o r } .$ </td><td>Rus.</td><td> ${ \mathrm { V i e . } }$ </td></tr><tr><td>0.00</td><td>15.9</td><td>14.4</td><td>17.2</td><td>27.7</td><td>11.1</td><td>15.3</td><td>20.6</td></tr><tr><td>0.10</td><td>15.3</td><td>14.6</td><td>17.8</td><td>29.2</td><td>11.5</td><td>15.4</td><td>21.1</td></tr><tr><td>0.20</td><td>15.3</td><td>15.4</td><td>18.8</td><td>30.9</td><td>12.2</td><td>16.0</td><td>20.8</td></tr><tr><td>0.30</td><td>15.6</td><td>16.3</td><td>21.8</td><td>34.5</td><td>13.4</td><td>16.7</td><td>23.1</td></tr></table>

Table 4: Codec evaluation results. AMUSE is a multi-domain mixture for speech, sounds, and music, used by ESPnet-Codec.
<table><tr><td></td><td></td><td>LibriTTS STOI</td><td>SPK</td><td>YODAS v3 STOI</td><td>SPK</td></tr><tr><td>Training Data Baselines</td><td>SR</td><td></td><td></td><td></td><td></td></tr><tr><td>LibriTTS</td><td>16kHz</td><td>0.95</td><td>0.76</td><td>0.74</td><td>0.69</td></tr><tr><td>LibriTTS</td><td>24kHz</td><td>0.97</td><td>0.83</td><td>0.80</td><td>0.78</td></tr><tr><td>AMUSE</td><td>16kHz</td><td>0.93</td><td>0.74</td><td>0.81</td><td>0.83</td></tr><tr><td>AMUSE</td><td>44kHz</td><td>0.95</td><td>0.82</td><td>0.90</td><td>0.77</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>This work</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>YODAS v3</td><td>16kHz</td><td>0.92</td><td>0.60</td><td>0.82</td><td>0.79</td></tr><tr><td>YODAS v3</td><td>24kHz</td><td>0.92</td><td>0.66</td><td>0.84</td><td>0.82</td></tr><tr><td>YODAS v3</td><td>48kHz</td><td>0.97</td><td>0.84</td><td>0.94</td><td>0.90</td></tr></table>

Results: All models perform worse on the YODAS v3 test set compared to LibriTTS (Table 4). This is expected since the audio domain is much more challenging - clips often have background noise and multiple sound sources. However, LibriTTS contains only clean single speaker recordings. The best performing model is the 48kHz DAC trained on YODAS v3 on both the in-domain and out-of-domain test sets with a STOI of 0.97 / 0.94 and a speaker similarity of 0.84 / 0.90, respectively. It outperforms all models trained on lower sampling rates and even models trained on more data (AMUSE, 33K hours), showing the effectiveness of the YODAS v3’s high-bandwith data.

## 5. Conclusion

This paper presents YODAS v3, a weakly-supervised corpus containing 1.1 million hours of high-bandwith stereo data across 147 languages. YODAS v3 is created using a novel language-balanced keyword search approach, allowing us to collect large amounts of data for medium and low-resource languages: 22 languages in YODAS v3 have over 10K hours and 73 languages have over 5K hours of data. We conduct several analyses on the collected audio, showing how the data is both truly stereo and high-bandwith. Finally, we show the downstream use cases of YODAS v3. We train train ASR models are different subsets of the data, yielding results that show the cleainliness of the YODAS v3 transcripts. We also develop new high-fidelity neural codecs that are competitive with the stateof-the-art while showing how previous approaches fail to generalize to the complex audio scenes found in web-scale data.

## 6. Acknowledgments

Parts of this work used the PSC Bridges2 system and Delta/DeltaAI system at NCSA through allocations CIS210014 and IRI120008P from the ACCESS program, supported by NSF grants #2138259,#:2138286, #:2138307, #:2137603, and #:2138296. This paper is also based on results obtained from a project, Programs for Bridging the gap between R&D and the IDeal society (society 5.0) and Generating Economic and social value (BRIDGE)/Practical Global Research in the AI × Robotics Services, implemented by the Cabinet Office, Government of Japan.

## 7. Generative AI Disclosure

Generative AI was used to refine the manuscript text, such as wording and prose. It was also used to help write and scale the code necessary for large-scale crawling and data analyses. The final versions of all artificats produced with the assistance of generative AI were created and verified by humans.

## 8. References

[1] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in ICML 2023, 2023.

[2] J. Tian, W. Chen, Y. Peng, J. Shi, S. Arora, S. Bharadwaj, T. Maekaku, Y. Shinohara, K. Goto, X. Yue, H. Yang, and S. Watanabe, “OpusLM: A Family of Open Unified Speech Language Models,” in Interspeech 2025, 2025, pp. 3259–3263.

[3] Y. Peng, J. Tian, B. Yan, D. Berrebbi, X. Chang, X. Li, J. Shi, S. Arora, W. Chen, R. Sharma, W. Zhang, Y. Sudo, M. Shakeel, J. weon Jung, S. Maiti, and S. Watanabe, “Reproducing Whisperstyle training using an open-source toolkit and publicly available data,” in ASRU 2023, 2023.

[4] A. Defossez´ et al., “Moshi: a speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[5] W. Chen, W. Zhang, Y. Peng, X. Li, J. Tian, J. Shi, X. Chang, S. Maiti, K. Livescu, and S. Watanabe, “Towards robust speech representation learning for thousands of languages,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, Eds. Miami, Florida, USA: Association for Computational Linguistics, Nov. 2024, pp. 10 205–10 224. [Online]. Available: https://aclanthology.org/2024.emnlp-main. 570/

[6] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi et al., “Scaling speech technology to 1,000+ languages,” arxiv:2305.13516, 2023.

[7] Y. Chu, J. Xu, Q. Yang, H. Wei, X. Wei, Z. Guo, Y. Leng, Y. Lv, J. He, J. Lin et al., “Qwen2-audio technical report,” arXiv preprint arXiv:2407.10759, 2024.

[8] W. Chen, J. Tian, Y. Peng, B. Yan, C.-H. H. Yang, and S. Watanabe, “OWLS: Scaling laws for multilingual speech recognition and translation models,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https://openreview.net/forum?id=xnPW7yYomF

[9] J. Tian, H. Wang, B.-H. Su, C.-y. Huang, Q. Wang, J. Shi, W. Chen, X. Gong, S. Arora, C.-J. Li et al., “Bagpiper: Solving open-ended audio tasks via rich captions,” arXiv preprint arXiv:2602.05220, 2026.

[10] W. Chen, P. Seetharaman, R. Kumar, O. Nieto, S. Watanabe, J. Salamon, and Z. Jin, “Audiochat: Unified audio storytelling, editing, and understanding with transfusion forcing,” arXiv preprint arXiv:2602.17097, 2026.

[11] L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, M. Duppenthaler, P.-A. Duquenne, B. Ellis, H. Elsahar, J. Haaheim et al., “Seamless: Multilingual expressive and streaming speech translation,” arxiv:2312.05187, 2023.

[12] L. Barrault, Y.-A. Chung, M. C. Meglioli, D. Dale, N. Dong, P.- A. Duquenne, H. Elsahar, H. Gong, K. Heffernan, J. Hoffman et al., “SeamlessM4T-massively multilingual & multimodal machine translation,” arxiv:2308.11596, 2023.

[13] A. Omnilingual, G. Keren, A. Kozhevnikov, Y. Meng, C. Ropers, M. Setzler, S. Wang, I. Adebara, M. Auli, C. Balioglu et al., “Omnilingual asr: Open-source multilingual speech recognition for 1600+ languages,” arXiv preprint arXiv:2511.09690, 2025.

[14] B. Shi, A. Tjandra, J. Hoffman, H. Wang, Y.-C. Wu, L. Gao, J. Richter, M. Le, A. Vyas, S. Chen et al., “Sam audio: Segment anything in audio,” arXiv preprint arXiv:2512.18099, 2025.

[15] C. Wang et al., “VoxPopuli: A Large-Scale Multilingual Speech Corpus for Representation Learning, Semi-Supervised Learning and Interpretation,” in ACL 2021, 2021.

[16] G. Chen et al., “GigaSpeech: An evolving, multi-domain ASR corpus with 10,000 hours of transcribed audio,” in Interspeech 2021, 2021.

[17] X. Li, S. Takamichi, T. Saeki, W. Chen, S. Shiota, and S. Watanabe, “YODAS: Youtube-oriented dataset for audio and speech,” in ASRU 2023, 2023.

[18] J. Kahn, M. Riviere, W. Zheng, E. Kharitonov, Q. Xu, P. Mazar\` e,´ J. Karadayi, V. Liptchinsky, R. Collobert, C. Fuegen, T. Likhomanenko, G. Synnaeve, A. Joulin, A. Mohamed, and E. Dupoux, “Libri-Light: A benchmark for ASR with limited or no supervision,” in ICASSP, 2020.

[19] H. He, Z. Shang, C. Wang, X. Li, Y. Gu, H. Hua, L. Liu, C. Yang, J. Li, P. Shi et al., “Emilia: An extensive, multilingual, and diverse speech dataset for large-scale speech generation,” arXiv preprint arXiv:2407.05361, 2024.

[20] Y. Zhang, W. Han, J. Qin, Y. Wang, A. Bapna, Z. Chen, N. Chen, B. Li, V. Axelrod, G. Wang et al., “Google USM: Scaling automatic speech recognition beyond 100 languages,” arxiv:2303.01037, 2023.

[21] G. Comanici, E. Bieber, M. Schaekermann, I. Pasupat, N. Sachdeva, I. Dhillon, M. Blistein, O. Ram, D. Zhang, E. Rosen et al., “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” arXiv preprint arXiv:2507.06261, 2025.

[22] S. Luger, R. Mosquera-Gomez, A. Miłowski, T. Vaughan,´ S. Hincapie-Monsalve, P. Ortiz Suarez, and K. Bollacker, “Building data infrastructure for low-resource languages,” in Proceedings of the Eighth Workshop on Technologies for Machine Translation of Low-Resource Languages (LoResMT 2025), A. K. Ojha, C.-h. Liu, E. Vylomova, F. Pirinen, J. Washington, N. Oco, and X. Zhao, Eds. Albuquerque, New Mexico, U.S.A.: Association for Computational Linguistics, May 2025, pp. 154–160. [Online]. Available: https://aclanthology.org /2025.loresmt-1.14/

[23] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A large-scale multilingual dataset for speech research,” in Interspeech 2020, pp. 2757–2761.

[24] S. Cornell, C. Boeddeker, T. Park, H. Huang, D. Raj, M. Wiesner, Y. Masuyama, X. Chang, Z.-Q. Wang, S. Squartini, P. Garcia, and S. Watanabe, “Recent trends in distant conversational speech recognition: A review of chime-7 and 8 dasr challenges,” Computer Speech and Language, vol. 97, p. 101901, 2026. [Online]. Available: https://www.sciencedirect.com/science/articl e/pii/S0885230825001263

[25] P. Mousavi, G. Maimon, A. Moumen, D. Petermann, J. Shi, H. Wu, H. Yang, A. Kuznetsova, A. Ploujnikov, R. Marxer, B. Ramabhadran, B. Elizalde, L. Lugosch, J. Li, C. Subakan, P. Woodland, M. Kim, H. yi Lee, S. Watanabe, Y. Adi, and M. Ravanelli, “Discrete audio tokens: More than a survey!” Transactions on Machine Learning Research, 2025. [Online]. Available: https://openreview.net/forum?id=eqNchtvc6v

[26] S. Brade, S. Anderson, R. Kumar, Z. Jin, and A. Truong, “Speakeasy: Enhancing text-to-speech interactions for expressive content creation,” in Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, ser. CHI ’25. New York, NY, USA: Association for Computing Machinery, 2025. [Online]. Available: https://doi.org/10.1145/3706598.3714263

[27] Z. Zheng, P. Peng, Z. Ma, X. Chen, E. Choi, and D. Harwath, “BAT: Learning to reason about spatial sounds with large language models,” in Forty-first International Conference on Machine Learning, 2024. [Online]. Available: https://openreview .net/forum?id=kao5hRX9YA

[28] C. Li, W. Zhang, W. Wang, R. Scheibler, K. Saijo, S. Cornell, Y. Fu, M. Sach, Z. Ni, A. Kumar et al., “Less is more: Data curation matters in scaling speech enhancement,” arXiv preprint arXiv:2506.23859, 2025.

[29] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, and M. Auli, “XLS-R: Self-supervised Cross-lingual Speech Rep resentation Learning at Scale,” in Interspeech 2022, 2022, pp. 2278–2282.

[30] M. Z. Boito, V. Iyer, N. Lagos, L. Besacier, and I. Calapodescu, “mhubert-147: A compact multilingual hubert model,” arXiv preprint arXiv:2406.06371, 2024.

[31] Y. Peng, M. Shakeel, Y. Sudo, W. Chen, J. Tian, C.-J. Lin, and S. Watanabe, “OWSM v4: Improving Open Whisper-Style Speech Models via Data Scaling and Cleaning,” in Interspeech 2025, 2025, pp. 2225–2229.

[32] M. Sekoyan, N. R. Koluguri, N. Tadevosyan, P. Zelasko, T. Bartley, N. Karpov, J. Balam, and B. Ginsburg, “Canary-1b-v2 & parakeet-tdt-0.6 b-v3: Efficient and high-performance models for multilingual asr and ast,” arXiv preprint arXiv:2509.14128, 2025.

[33] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in LREC 2020, 2020, pp. 4218–4222.

[34] A. Conneau et al., “FLEURS: Few-shot learning evaluation of universal representations of speech,” in SLT 2022, 2022.

[35] J. Valk and T. Alumae, “VOXLINGUA107: A dataset for spoken¨ language recognition,” in SLT 2021, 2021.

[36] T. Kudo, “Subword regularization: Improving neural network translation models with multiple subword candidates,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Melbourne, Australia, Jul. 2018, pp. 66–75.

[37] S. Cornell, M. S. Wiesner, S. Watanabe, D. Raj, X. Chang, P. Garcia, Y. Masuyam, Z.-Q. Wang, S. Squartini, and S. Khudanpur, “The chime-7 dasr challenge: Distant meeting transcription with multiple devices in diverse scenarios,” in 7th International Workshop on Speech Processing in Everyday Environments (CHiME 2023), 2023, pp. 1–6.

[38] W. Zhang, K. Saijo, S. Cornell, R. Scheibler, C. Li, Z. Ni, A. Kumar, M. Sach, W. Wang, Y. Fu, S. Watanabe, T. Fingscheidt, and Y. Qian, “Lessons Learned from the URGENT 2024 Speech Enhancement Challenge,” in Interspeech 2025, 2025, pp. 853–857.

[39] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “Libritts: A corpus derived from librispeech for textto-speech,” in Proc. Interspeech, 2019, pp. 1526–1530.

[40] S. Watanabe, T. Hori, S. Karita, T. Hayashi, J. Nishitoba, Y. Unno, N. Enrique Yalta Soplin, J. Heymann, M. Wiesner, N. Chen, A. Renduchintala, and T. Ochiai, “ESPnet: End-to-end speech processing toolkit,” in Interspeech 2018, 2018.

[41] R. Kumar, P. Seetharaman, A. Luebs, I. Kumar, and K. Kumar, “High-fidelity audio compression with improved rvqgan,” Advances in Neural Information Processing Systems, vol. 36, pp. 27 980–27 993, 2023.

[42] J. Shi et al., “Espnet-codec: Comprehensive training and evaluation of neural codecs for audio, music, and speech,” arXiv preprint arXiv:2409.15897, 2024.

[43] H. Zen, V. Dang, R. Clark, Y. Zhang, R. J. Weiss, Y. Jia, Z. Chen, and Y. Wu, “LibriTTS: A Corpus Derived from LibriSpeech for Text-to-Speech,” in Interspeech 2019, 2019, pp. 1526–1530.

[44] J. Shi, H.-j. Shim, J. Tian, S. Arora, H. Wu, D. Petermann, J. Q. Yip, Y. Zhang, Y. Tang, W. Zhang et al., “Versa: A versatile evaluation toolkit for speech, audio, and music,” arXiv preprint arXiv:2412.17667, 2024.

[45] C. H. Taal, R. C. Hendriks, R. Heusdens, and J. Jensen, “A shorttime objective intelligibility measure for time-frequency weighted noisy speech,” in 2010 IEEE International Conference on Acoustics, Speech and Signal Processing, 2010, pp. 4214–4217.

[46] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao, J. Wu, L. Zhou, S. Ren, Y. Qian, Y. Qian, J. Wu, M. Zeng, X. Yu, and F. Wei, “WavLM: Largescale self-supervised pre-training for full stack speech processing,” IEEE JSTSP, 2022.