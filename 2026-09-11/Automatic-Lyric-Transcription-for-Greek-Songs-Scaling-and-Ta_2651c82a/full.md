# Automatic Lyric Transcription for Greek Songs: Scaling and Task Composition Effects in Whisper Adaptation

Maria Frangiadaki <sup>ID</sup> <sup>1,∗∗</sup>, Dimitrios Damianos <sup>ID</sup> <sup>1</sup>, Kosmas Kritsis <sup>ID</sup> <sup>1</sup>, Vassilis Katsouros <sup>ID</sup>

<sup>1</sup> Institute for Language and Speech Processing, Athena R.C., Greece

{maria.frangiadaki, d.damianos, kosmas.kritsis, vsk}@athenarc.gr

## Abstract

Automatic Lyric Transcription (ALT) remains substantially more challenging than speech recognition due to melodic variability, rhythmic irregularity, and accompaniment interference. This is heightened in low-resource languages like Greek, where no prior benchmark for ALT exists. We present the first controlled study of Whisper adaptation for Greek ALT, investigating model scaling effects, task composition via multitask training in transcribe-translate ratios, and two-stage speech-tosinging adaptation. We also curate a segment-level aligned singing dataset based on the Greek Audio Dataset (GAD) using source separation and CTC forced alignment. Results show that scaling consistently improves performance, while multitask learning acts as a beneficial regularizer primarily for smallercapacity models. The 2-stage adaptation in Whisper Large-v3 achieves a Word Error Rate (WER) of 27.2%, a significant improvement over zero-shot baselines, establishing the first Greek ALT benchmark.

Index Terms: Automatic Lyric Transcription (ALT), Automatic Speech Recognition (ASR), Lyric Alignment, Whisper fine-tuning, Multitask Learning, Low-resource Languages

## 1. Introduction

Automatic Speech Recognition (ASR) and recent advances in large-scale multilingual pre-trained models have enabled a broad spectrum of everyday applications. However, ASR systems often degrade when the deployment domain differs from the training distribution. A particularly challenging and underexplored form of domain shift is singing voice. The task of converting sung audio into text, known as Automatic Lyric Transcription (ALT), lies at the intersection of ASR and Music Information Retrieval (MIR) and remains substantially more difficult than conventional speech recognition. Singing differs fundamentally from speech in both acoustic and linguistic structure. Large pitch excursions, sustained vowels, melisma (vowel elongation across multiple notes), rhythmic irregularity, instrumental accompaniment, and expressive articulation violate assumptions learned from speech-dominant corpora [1]. As a result, models trained primarily on spoken language struggle to generalize to sung vocals.

This acoustic domain shift is even more severe in lowresource languages. While English singing datasets have enabled recent progress in ALT, many languages lack systematically curated singing corpora, aligned annotations, and reproducible evaluation protocols. Greek, despite its rich musical tradition and morphological complexity, has not been studied in a controlled ALT setting. No established benchmark currently exists for Greek singing voice ASR, and the behavior of multilingual models under singing-domain adaptation remains unclear.

This work addresses ALT for Greek singing voice, focusing on the Whisper model [2] as a strong multilingual pretrained baseline. We investigate to what extent can adaptation of multilingual pre-trained ASR models such as Whisper improve lyric transcription accuracy, when transferring from speech to singing in a low-resource singing scenario. To answer this question, we develop a complete end-to-end pipeline for Greek ALT and conduct a controlled experimental study, which includes fine-tuning Whisper in various checkpoints under transcriptiononly training and evaluating multitask and two-stage learning strategies. The contributions of this work are as follows:

1. We present the first systematic benchmark for Greek ALT.

2. We curated a fully processed and aligned version of the Greek Audio Dataset (GAD) [3], including source-separated and aligned segmentation, translation pairs, and reproducible train-validation-test splits.

3. We conduct a controlled study of model scale and task composition for singing-domain adaptation in a low-resource language.

4. We introduce a task-pure batching scheme with languageaware pre-processing that stabilizes training on singing voice.

5. We propose a quantitative and qualitative error taxonomy tailored to Greek lyrics, highlighting singing-specific errors.

6. We prove that large-scale transcription-only and 2-stage adaptation is the most effective strategy for Greek ALT, while multitask learning effectively serves as regularization and improves Word Error Rate (WER) results for smaller-capacity models.<sup>1</sup>

## 2. Related Work

## 2.1. ALT

Early approaches to ALT relied on conventional ASR pipelines tailored to music, typically employing Hidden Markov Models (HMMs) combined with Gaussian Mixture Models (GMMs) or Deep Neural Networks (DNNs) adapted on singing data [4]. The release of benchmark datasets such as DALI [5] and DAMP-Sing [6] enabled systematic evaluation, highlighting the persistent acoustic mismatch between speech and singing. The shift towards end-to-end deep learning architectures, such as

Connectionist Temporal Classification (CTC) and Attentionbased Encoder-Decoder (AED) models, unified acoustic modeling and alignment [7]. Recent work has also extended these to multimodal setups, demonstrating that auxiliary cues like lip movements or note-level transcriptions can further stabilize decoding in low-SNR conditions [8, 9]. Despite these advances, lyric transcription remains a challenging task that requires robust domain adaptation.

## 2.2. Foundation Models and adaptation

The advent of large-scale, self-supervised foundation models has redefined the state-of-the-art in speech processing. Architectures like wav2vec 2.0 and its cross-lingual extension, XLS-R, learn general acoustic representations that transfer effectively to singing via fine-tuning [10, 11]. More recently, OpenAI’s Whisper [2] has demonstrated remarkable zero-shot robustness due to its massive multilingual pre-training. In the context of MIR, while Whisper exhibits strong performance on clean vocals, its zero-shot accuracy degrades significantly on polyphonic audio [1]. To mitigate this, approaches propose cascading Whisper with Large Language Models (LLMs) for error correction [12]. Further ASR techniques, such as employing parameter-efficient tuning (e.g., LoRA) for scalable specialization [13], or utilizing speech-text joint pretraining (e.g., SpeechLM [14]) further integrate acoustic and linguistic information. However, most existing research focuses on highresource languages, leaving the efficacy of such foundation models on low-resource singing languages largely unexplored.

Adapting ASR models to low-resource domains often necessitates specialized training strategies. Multitask learning, where the model is jointly optimized on auxiliary tasks such as translation, acts as a regularizer preventing overfitting on small target datasets. In the context of Whisper, the interplay between its transcription and translation tokens offers a unique avenue for multitask adaptation [15]. Additionally, staged fine-tuning strategies have been shown to stabilize ASR training in lowresource scenarios[16]. Data-centric strategies further bridge the speech-to-singing gap through voice-to-singing augmentation [17] and consistency loss regularization [18].

## 2.3. Greek Speech and Singing Recognition

Automatic Speech Recognition for the Greek language has seen progress through recent spoken corpora that strengthen the ASR infrastructure [19, 20]. Benchmarks demonstrate that while generic models like Whisper [2] perform sufficiently in spoken Greek, they struggle with dialectal variations and fastpaced articulation. To overcome data scarcity, recent frameworks leverage unsupervised domain adaptation. For example, M2DS2 [21] and MSDA [22] combine self-supervised pretraining with pseudo-label-based teacher-student training to effectively reduce domain mismatch in Modern Greek ASR. The intersection of Greek ASR and singing voice analysis is virtually non-existent in the literature. To date, there is no standardized benchmark for Greek ALT, and no study has systematically evaluated the transferability of multilingual foundation models to Greek singing. While existing Greek music datasets, such as Lyra [23], GAD [3] and the Greek Music Dataset (GMD) [24], provide valuable acoustic resources for general Music Information Retrieval (MIR) tasks, they lack the segment-level aligned text annotations required for end-to-end audio-to-lyrics transcription. This work bridges this gap by providing the first controlled evaluation of Whisper on Greek singing voice, establishing a baseline for future research in low-resource ALT.

## 3. The GAD-ALT Dataset

A core contribution of this work is the curation of the GAD-ALT corpus. ALT requires temporal synchronization between audio and text, so we extend the GAD [3], which was originally designed for genre classification, into an ASR-ready dataset. The GAD is a collection of 1,000 popular Greek songs spanning multiple genres (Urban, Folk, Rock, Hip Hop Pop). All entries are accompanied by genre annotations, lyrics, manually annotated mood labels, metadata, extracted audio features and links to the corresponding songs on YouTube, thus enabling researchers to obtain the raw audio when required.

## 3.1. Source Separation

To convert this into a reproducible ALT benchmark, substantial curation was required to resolve missing metadata, correct mismatches, and standardize lyric formatting. Because music recordings are polyphonic, we employ Hybrid Transformer Demucs (htdemucs ft) [25] for two-stem source separation, extracting vocal and accompaniment tracks. Extracted vocals are downmixed to mono, resampled to 16 kHz to match Whisper’s [2] input requirements, and converted into Kaldi format.

## 3.2. Forced Alignment and Bilingual Augmentation

For temporal alignment, we employ a customized version of the open-source CTC forced aligner [26]. Recordings are processed in 30-second overlapping windows. To filter out lowquality alignments, we compute a custom confidence score that combines aligned token percentage (50%), average CTC log-probabilities (30%) and duration regularity penalties (20%) [27]. To enable multitask experiments, each aligned Greek segment is translated into English at the segment level using the gpt-4o-mini model via the OpenAI API [28], with zero-shot prompting,temperature set to 0 and a maximum limit of 128 tokens, preserving temporal alignment. The final curated singing corpus is in Hugging Face form and comprises 17,458 aligned lyric segments (19.65 hours). To prevent data leakage, splitting is performed at the song level, partitioning the dataset into 13,750 training (78.8%), 1,892 validation (10.8%), and 1,816 test segments (10.4%), with a mean duration of 4 seconds. Each final entry contains: (i) the aligned audio segment, (ii) the normalized Greek transcription, and (iii) the English translation.

## 4. Methodology and Experimental Setup

## 4.1. Overview

The proposed methodology introduces a complete end-to-end pipeline for Greek ALT. After curating the GAD-ALT dataset, multilingual pretrained Whisper models [2] are adapted to the singing domain under controlled settings that vary in model scale, task composition, and staged speech-to-singing adaptation.

## 4.2. Whisper Adaptation Strategies

Zero-shot inference using the pretrained models serves as our out-of-domain baseline. We evaluate three model scales (Small, Medium, Large-v3) under two supervised fine-tuning regimes. A Transcription-only task (Greek audio → Greek text) and a Multitask setting (Greek audio → Greek transcription + English translation). For the latter, task-specific batches are interleaved in fixed ratios (2:1, 4:1) using a deterministic sampler to ensure task-homogeneous batches, allowing us to investigate if translation acts as a regularizer. The proposed multitask training framework is illustrated in Figure 1. To reduce the speech-tosinging domain gap, we also evaluate a Staged Adaptation strategy. For this, we additionally utilize a Greek subset of Mozilla Common Voice (v23.0) [29], which comprises approximately 37.5 hours of validated read speech. In Stage 1, we fine-tune Whisper solely on this Greek speech while keeping the entire encoder frozen, so that the updates are implemented exclusively to the decoder for language adaptation. Finally, in Stage 2, the model is fully unfrozen and fine-tuned on the singing corpus.

![](images/648b75b623f3e1813f16ab63fa21bd0b7ff589fc9f1a4a7f1ce34217c07f3018.jpg)  
Figure 1: Overview of the proposed multitask training frameworkfor the 2:1 transcription:translation ratio.

## 4.3. Training Details

All models are trained using the Hugging Face Seq2SeqTrainer on multi-GPU NVIDIA A100 nodes. Optimization is performed via AdamW for 5 epochs. Learning rate of $5 \times 1 0 ^ { \div 5 }$ performed better for Whisper Small and Medium, whereas $3 \times 1 0 ^ { - 5 }$ proved more suitable for Whisper Large-v3. Per-GPU batch size ranges from 4 to 8 segments depending on model scale, utilizing mixed-precision (FP16) for efficiency. To facilitate future research, our complete training pipeline and model configurations will be released as open-source.

## 5. Results

Evaluation is computed in normalized Word Error Rate (WER), after lowercasing, punctuation removal, and standard text normalization. The WER is computed using the jiwer library over aggregated segment-level predictions.

Table 1: Normalized Word Error Rate (WER %) on the Greek singing test set. The table contrasts model scaling against task composition (multitask ratios) and staged adaptation. Best adaptation results per model scale are highlighted in bold.
<table><tr><td>Training Setup</td><td>Small</td><td>Medium</td><td>Large-v3</td></tr><tr><td>Zero-shot</td><td>92.3</td><td>65.1</td><td>53.6</td></tr><tr><td>2:1 transcribe-translate</td><td>33.6</td><td>32.3</td><td>30.7</td></tr><tr><td>4:1 transcribe-translate</td><td>34.9</td><td>31.6</td><td>30.2</td></tr><tr><td>Transcribe-only</td><td>36.7</td><td>30.3</td><td>28.4</td></tr><tr><td>2 stages transcribe</td><td>36.6</td><td>30.1</td><td>27.2</td></tr></table>

## 5.1. Domain Gap Analysis

Table 1 summarizes normalized WER (%) across all Whisper model sizes and training configurations on the held-out Greek singing test set. Zero-shot evaluation reveals a clear scaling trend. Whisper Small fails almost completely with 92.3% WER, Whisper Medium achieves 65.1%, and Whisper Large reduces error further to 53.6%. Although increased model capacity partially mitigates degradation, performance remains insufficient for practical ALT. In contrast, supervised fine-tuning reduces WER significantly across larger models, demonstrating that scale alone cannot bridge the speech-to-singing domain gap. Thus, zero shot approaches like Lyricwiz [12] would not be ideal for a low resource language like Greek. While Whisper’s large-scale pretraining provides a strong foundation, it does not fully account for melodic prolongation, vowel stretching, rhythmic compression, and altered phoneme realizations characteristic of singing voice.

## 5.2. Capacity and Regularization

We systematically evaluated Whisper [2] across scales and training strategies to establish the first benchmark for Greek ALT. Our findings indicate that multitask learning primarily acts as a regularization mechanism for smaller-capacity models, while larger models benefit more from focused transcriptiononly adaptation. For Whisper Small (244M parameters), the 2:1 transcribe:translate configuration achieves the best WER performance (33.6%), outperforming transcription-only training (36.7%). Increasing transcription dominance to 4:1 slightly degrades performance (34.9%), yet remains superior to pure transcription. This pattern suggests that moderate auxiliary translation exposure provides beneficial regularization at low capacities, encouraging more stable encoder representations without overwhelming the primary transcription objective.

In contrast, as observed in Figure 2, as model capacity increases, the encoder-decoder architecture internalizes sufficient linguistic structure from pretraining and benefits more from target-focused specialization. For Whisper Medium (769M parameters), transcription-only fine-tuning yields the lowest WER (30.3%), outperforming both 2:1 (32.3%) and 4:1 (31.6%) mixtures. A similar trend is observed for Whisper Large-v3 (1.55B parameters), where transcription-only training achieves 28.4% WER.

## 5.3. Two-Stage Adaptation

Interestingly, our two-stage approach proved beneficial for the larger models. The Whisper Small performed better when fine-tuned directly on the singing data. With the 2-stage approach, it achieved 36.6% WER, a slightly better score than the transcribe-only fine-tune (36.7%), but still worse than the multitask training (33.6%). This may also be attributed to the characteristics of the intermediate speech corpus. The Greek Common Voice dataset [29] primarily consists of read speech with controlled prosody. As a result, Stage-1 adaptation likely reinforces clean speech patterns that remain acoustically distant from singing. Future work will investigate whether spontaneous, prosodically rich speech corpora (e.g., conversational or podcast-style speech) provide a more acoustically compatible intermediate domain. On the other hand, Whisper medium shows marginal gains from this approach (30.1% WER), while the Large-v3 model seems to have enough parameters to learn strong Greek language patterns from the speech data without losing the flexibility needed to adapt to the complex acoustics of songs later on, ultimately achieving the best performance of 27.2% WER.

## 5.4. Ablations of Source Separation and Augmentation

Interestingly, training and evaluating on isolated vocal stems yield measurable gains over using raw polyphonic mixtures. We fine-tuned and tested the model with the best WER score on raw polyphonic data, and it scored 33.4% WER . This suggests that residual accompaniment is not the primary error driver in this dataset, but vocal isolation is still a better choice. Furthermore, augmentation strategies based on SNR-controlled stem remixing, light reverberation, and mixed raw+vocals training consistently degraded performance, notably increasing WER compared to the vocal-only baseline. We therefore conclude that vocals-only training without artificial remixing is the most reliable configuration.

## 5.5. Qualitative Error Analysis

Beyond WER scores, we perform a structured qualitative analysis to characterize singing-specific error patterns in Greek. We manually inspected and categorized 200 transcription errors produced by the best-performing model.

• Semantic substitution (25.5%): The model occasionally replaces a word with a phonetically similar but semantically distinct alternative $( e . g . , ) ^ { 6 6 } x ^ { 6 } P l ^ { 3 }$ [hand] transcribed as “γέροι” [old men], or “κρίμα” [pity] as “χρήμα” [money]).

• Boundary drift (24.0%): Melismatic stretching often leads to incorrect segmentation, merging or splitting lexical units (e.g., “στην αμμουδιά ποτέ του” boundaries shifting to create the non-words “στην αμμου διαποτετου”), indicating difficulty aligning syllabic timing with word boundaries.

• Hallucinated or severely corrupted content (19.5%): High rhythmic density and rapid articulation frequently blur consonant clusters, resulting in phonotactically implausible syllables (e.g., a nonsensical string “κοντεριακος ατρειατα πατασιαζουν της”).

• Orthographic ambiguity (17.0%): Greek contains many homophones as well as multiple graphemic representations for similar vowel sounds $( e . g . , \iota / \eta / \mathfrak { e } \mathfrak { e } / \cup / \mathfrak { o } \mathfrak { e }$ all stand for ”i”, and ο/ω are both pronounced $" \mathrm { o } ^ { \prime \prime } )$ . Whisper transcriptions contain substitutions that preserve phonetic similarity but alter lexical meaning or grammar (e.g., “όλοι” [all, masculine plural] transcribed as “όλη” [all, feminine singular], or $" \boldsymbol { \sigma } \in \mathsf { c } \rho \mathsf { \Pi } \mathsf { v } \varepsilon \varsigma ^ { , * }$ [sirens] as “συρίνες” [spelling mistake]).

• Function-word deletion and insertion (12.0%): Short grammatical particles (e.g., “μη”, “πως”, “δε”) are often omitted in fast singing or spuriously inserted.

![](images/859212e6b8f888302761e551ab8cc3899e9311471a0099f47d997df06dce9c27.jpg)

(a) Scaling effect  
![](images/1e07873b0fe737a0e0fa046e536e4fcc269d17e60d7c594867d0acc7835ff4c7.jpg)  
(b) Multitask ratio effect  
Figure 2: Model scaling improves robustness, but singingdomain adaptation dominates performance. Multitask mixing primarily benefits smaller-capacity models.

• Morphological drift (2.0%): In several instances, the lemma is preserved but the inflection changes (e.g., the neuter adjective “πανάκριβο” altering its suffix to plural $\pi \alpha \times \alpha \times \rho ( \beta \alpha ^ { 3 } )$

Overall, fine-tuning substantially reduces error frequency but does not eliminate singing-specific categories. Larger models primarily decrease severity rather than altering the distribution of error types, indicating that melodic variability and articulation distortions remain central challenges for Greek ALT.

## 6. Conclusions & Future Work

In this work, we presented the first systematic benchmark for Greek ALT. We expand the GAD [3] to GAD-ALT, with segments, alignment, translations and Hugging Face splits. Through a comprehensive evaluation of Whisper [2] adaptation strategies, we demonstrated that while zero-shot inference suffers from a severe speech-to-singing domain gap, targeted fine-tuning dramatically reduces the Word Error Rate to 27.2%. Furthermore, our findings reveal that multitask learning (incorporating translation) acts as an effective regularizer for smaller-capacity models, whereas larger models benefit most from focused, transcription-only and two-stage adaptation. Future work will focus on expanding the curated singing corpus, exploring expressive and spontaneous speech corpora for more effective staged adaptation, using parameter-efficient finetuning methods, and integrating Greek-specific language models to better handle the morphological and rhythmic complexities of the singing voice.

## 7. Acknowledgements

The authors acknowledge the EuroHPC Joint Undertaking for awarding this project access to the EuroHPC supercomputer LEONARDO, hosted by CINECA (Italy) and the LEONARDO consortium, through a EuroHPC Development Access call (Project No. EUHPC-D27-063). This work received partial funding from the European High-Performance Computing Joint Undertaking (JU) under Grant Agreement No. 101234269 for the Pharos AI Factory project, as well as from the Greek Ministry of Digital Governance and Artificial Intelligence.

## 8. Generative AI Use Disclosure

Portions of this manuscript were refined with the assistance of generative AI tools for language editing and clarity. All experimental design, analysis, and scientific conclusions were developed independently by us.

## 9. References

[1] A. Kruspe, “More than words: Advancements and challenges in speech recognition for singing,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

[2] A. Radford et al., “Robust speech recognition via large-scale weak supervision,” in International Conference on Machine Learning (ICML), 2023.

[3] D. Makris, K. L. Kermanidis, and I. Karydis, “The greek audio dataset,” in IFIP International Conference on Artificial Intelligence Applications and Innovations. Springer, 2014, pp. 1–10.

[4] A. Mesaros and T. Virtanen, “Automatic recognition of lyrics in singing,” EURASIP Journal on Audio, Speech, and Music Processing, vol. 2010, no. 1, pp. 1–11, 2010.

[5] G. Meseguer-Brocal, A. Cohen-Hadria, and G. Peeters, “DALI: A large dataset of synchronized audio, lyrics and notes, automatically created using teacher-student machine learning paradigm,” in Proceedings of the International Society for Music Information Retrieval Conference (ISMIR), 2018.

[6] G. R. Dabike and J. Barker, “Automatic lyric transcription from karaoke vocal tracks: Resources and a baseline system,” in Proceedings ofInterspeech, 2019, pp. 579–583.

[7] D. Stoller, S. Durand, and S. Ewert, “End-to-end lyrics alignment for polyphonic music using an audio-to-character recognition model,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2019, pp. 181–185.

[8] X. Gu, L. Ou, D. Ong, and Y. Wang, “MM-ALT: A multimodal automatic lyric transcription system,” in Proceedings of the 30th ACM International Conference on Multimedia, 2022, pp. 3328– 3337.

[9] X. Gu et al., “Automatic lyric transcription and automatic music transcription from multimodal singing,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 20, no. 7, pp. 1–29, 2024.

[10] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Advances in Neural Information Processing Systems (NeurIPS), 2020.

[11] A. Babu et al., “XLS-R: Self-supervised cross-lingual speech representation learning at scale,” in Proceedings of Interspeech, 2022.

[12] L. Zhuo et al., “LyricWhiz: Robust multilingual zero-shot lyrics transcription by whispering to ChatGPT,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.

[13] Z. Song et al., “LoRA-Whisper: Parameter-efficient and extensible multilingual ASR,” in Proceedings ofInterspeech, 2024.

[14] Z. Zhang et al., “SpeechLM: Enhanced speech pre-training with unpaired textual data,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2023.

[15] R. J. Weiss, J. Chorowski, N. Jaitly, Y. Wu, and Z. Chen, “Sequence-to-sequence models can directly translate foreign speech,” in Proceedings ofInterspeech, 2017.

[16] L. G. Pillai, K. Manohar, B. K. Raju, and E. Sherly, “Multistage fine-tuning strategies for automatic speech recognition in lowresource languages,” arXiv preprint arXiv:2411.04573, 2024.

[17] S. Basak, S. Agarwal, S. Ganapathy, and N. Takahashi, “Endto-end lyrics recognition with voice to singing style transfer,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021.

[18] J. Huang et al., “Enhancing lyrics transcription on music mixtures with consistency loss,” in Proceedings ofInterspeech, 2025.

[19] G. Paraskevopoulos, C. Tsoukala, A. Katsamanis, and V. Katsouros, “The greek podcast corpus: Competitive speech models for low-resourced languages with weakly supervised data,” in Proceedings ofInterspeech, 2024.

[20] S. Vakirtzian et al., “Speech recognition for greek dialects: A challenging benchmark,” in Proceedings ofInterspeech, 2024, pp. 3974–3978.

[21] G. Paraskevopoulos, T. Kouzelis, G. Rouvalis, A. Katsamanis, V. Katsouros, and A. Potamianos, “Sample-efficient unsupervised domain adaptation of speech recognition systems: A case study for modern greek,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.

[22] D. Damianos et al., “MSDA: Combining pseudo-labeling and self-supervision for unsupervised domain adaptation in ASR,” in Proceedings ofInterspeech, 2025.

[23] C. Papaioannou, I. Valiantzas, T. Giannakopoulos, M. Kaliakatsos-Papakostas, and A. Potamianos, “A dataset for greek traditional and folk music: Lyra,” in Proceedings of the 23rd International Society for Music Information Retrieval Conference (ISMIR), Bengaluru, India, 2022, pp. 344–351.

[24] D. Makris, I. Karydis, and S. Sioutas, “The greek music dataset,” in Proceedings of the 16th International Conference on Engineering Applications ofNeural Networks (EANN). Rhodes, Greece: ACM, 2015, pp. 22:1–22:7.

[25] S. Rouard, F. Massa, and A. Defossez, “Hybrid transformers for´ music source separation,” in IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2023.

[26] M. Ashraf, “Ctc forced aligner,” https://github.com/ MahmoudAshraf97/ctc-forced-aligner, 2023.

[27] L. Kurzinger, D. Winkelbauer, L. Li, T. Watzel, and G. Rigoll,¨ “CTC-segmentation of large corpora for german end-to-end speech recognition,” in International Conference on Speech and Computer (SPECOM). Springer, 2020, vol. 12335, pp. 267–278.

[28] OpenAI, “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[29] R. Ardila et al., “Common voice: A massively-multilingual speech corpus,” in Proceedings of the 12th Language Resources and Evaluation Conference (LREC), 2020, pp. 4218–4222.