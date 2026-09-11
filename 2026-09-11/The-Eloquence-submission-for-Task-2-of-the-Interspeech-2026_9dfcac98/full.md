# The Eloquence submission for Task 2 of the Interspeech 2026 MLC-SLM challenge

Jordi Luque<sup>1,∗</sup>, Lorenzo Concina<sup>2,∗</sup>, Marco Matassoni<sup>2</sup>, Alessio Brutti<sup>2</sup>, Filippo Vella<sup>3</sup>

<sup>1</sup> Telefonica Innovaci ´ on Digital, Scientific Group´

<sup>2</sup> Fondazione Bruno Kessler

<sup>3</sup> Consiglio Nazionale delle Ricerche

jordi.luque@telefonica.com, {lconcina, brutti, matasso}@fbk.eu, filippo.vella@icar.cnr.it

## Abstract

This paper details the Eloquence team’s approach to Task 2 of the 2nd MLC-SLM challenge at Interspeech 2026, which involves multilingual Multiple-Choice Question Answering (MCQA) across 21 languages. Three approaches are explored. First, we fine-tune Voxtral-Mini-3B via LoRA with cross-lingual data augmentation, ASR transcript augmentation and timestamp-aware audio cropping, achieving 0.72 macroaccuracy on evaluation Phase 2. Second, we apply multimodal in-context learning (ICL) to the frozen Voxtral-24B model to correct a strong label bias, reaching 0.81, our best result. Third, a training-free retrieval system based on a three-layer voiceanchored memory combining acoustic identity, semantic content, and a knowledge graph achieves 0.68. All three systems substantially outperform the official baseline.

Index Terms: speech-LLMs, multiple-choice question answering, label bias mitigation, in-context learning, retrieval augmented generation, speech recognition

## 1. Introduction

Recent progress in Large Language Models (LLMs) has driven a broad shift toward Speech LLMs that unify speech perception and language understanding within a single model, moving beyond transcription-centric pipelines [1, 2] toward systems that can reason over both the acoustic and semantic content of spoken interaction [3, 4, 5, 6, 7, 8]. Real-world conversational speech [9, 10], however, differs substantially from read or single-speaker recordings: it contains overlapping turns, disfluencies, and multiple speakers, and this complexity is compounded in multilingual settings where training resources are unevenly distributed across languages [11]. The Multilingual Conversational Speech Language Model (MLC-SLM) Challenge [12] was introduced to address this gap by releasing large-scale, real-world multilingual conversational speech data together with a shared evaluation protocol. The first edition showed that current Speech LLMs largely solve transcription accuracy, while speaker diarization and deeper conversational understanding remained open; building on these findings, the second edition broadens language coverage and shifts emphasis toward diarization, acoustic understanding, and semantic understanding of full conversations<sup>1</sup>.

This paper reports on the participation of the Eloquence team in Task 2 of the second MLC-SLM challenge, Multilingual Conversational Speech Understanding. Task 2 requires systems to answer multiple-choice questions probing both the acoustic properties (e.g. speaker traits, emotion) and the semantic content of an entire multi-speaker conversation [13, 14]. Crucially, no oracle information is available at evaluation time: no pre-segmented utterances, no speaker labels, and no groundtruth diarization are provided, so a complete system must resolve who is speaking, what is said, and how it is said, directly from the raw recording. The task leaves the choice of architecture open, encouraging both pipeline-based and end-to-end solutions, and evaluates systems on their ability to answer questions about the conversation as a whole. We submitted three systems for Task 2, addressing the problem from different angles. The first, described in Section 3, fine-tunes Voxtral-Mini-3B via LoRA with cross-lingual data augmentation, ASR transcript augmentation, and timestamp-aware audio cropping. The second, described in Section 4, applies in-context learning to the frozen Voxtral-24B speech-LLM to correct a strong label bias without any parameter update. The third, described in Section 5, reframes the task as a retrieval problem: rather than adapting a Speech-LLM through training, it equips a frozen LLM with a persistent, voice-anchored memory layer queried at inference time.

![](images/984d9f1cd556bb1595e445a4f61b090fe6ade63badb1c652e148dc3fa4421e5d.jpg)  
Figure 1: Excerpt of 1141 006 Australian English dialogue.

## 2. Challenge and baseline description

The MLC-SLM challenge is focused on LLMs and their adaptation capability to different languages and contexts. The remarkable results shown by LLMs in many tasks can be replicated with difficulty in a specific context or in multilingual settings, where some resources can be scarce and difficult to find. The challenge provides a real-world multilingual conversational speech dataset, see Fig. 1, with the aim of paving the ground for speech models that can respond to human counterpart naturally in multilingual, dynamic and context-rich environments. The dataset comprises diverse conversational styles and captures the complexities of human dialog, including pauses, interruptions, and speaker overlaps. The first challenge comprised 11 languages, including five regional varieties of English [12]. The second MLC-SLM challenge, which organizers describe as more challenging than the first, expanded the language coverage by adding Tagalog, Urdu, and Turkish, as well as a regional dialects of Canadian French, Mexican Spanish, and Brasilian Portoguese. Task 1 is focused on multilingual conversational speech diarization and recognition, whereas Task 2, the task addressed by our system, is focused on multilingual conversational speech understanding. The training and development datasets provided are constructed using Gemini2.5-Pro, a multilingual multiple-choice question task that involves acoustic and semantic comprehension for training and development sets. The organizers provided a baseline system through Github<sup>2</sup>, where Qwen2.5-Omni-7B is fine tuned using the ms-swift toolkit [15]. The evaluation set of the challenge has been built as multiplechoice questions of speech comprehension with a similar procedure, and an additional manual review is used to rank the tasks.

## 3. Fine-Tuning Speech-LLM

## 3.1. Model and Training Setup

We fine-tune Voxtral-Mini-3B-2507 [16] using LoRA [17] (r=16, α=32, all linear modules) on 4×H100-64 GB GPUs (effective batch 32, learning rate $2 \times 1 0 ^ { - 5 }$ with linear decay). Each instance is a single-turn conversation placing audio and MCQ text in the user turn; the model emits a single letter (A– D) corresponding to the answer. Since all provided audio files appear in the challenge evaluation, extended fine-tuning leads to session memorisation. We restricted training to 0.25× an epoch, selecting the checkpoint minimizing the generalisation gap before memorisation sets in.

## 3.2. Cross-Lingual Data Augmentation

Using langdetect [18] we estimate that the evaluation set contains ≈52% cross-lingual question pairs (non-English audio, English question), while training data have 0% coverage. To account for this mismatch we translated non-English questions and options to English via NLLB-200 [19] (masking audio quotations, keeping native language terms for multilingual semantic questions) and train on a mixed dataset of original + translated copies (6,630 questions for 124 files, keeping 25 files for validation).

## 3.3. ASR Transcript Augmentation and Audio Cropping

We generate transcripts by fine-tuning Voxtral-Mini-3B as a LoRA ASR model: a Seed on MLC25 data, then an Adapted continuation with oversampled pre-training for the six new MLC26 languages. The Adapted model drops MLC26-new WER from 26.2% to 15.2%, with dramatic improvements on new languages (Table 1). Audio sessions are chunked into 1- minute segments with loop-detection to mitigate hallucinations. Transcripts are prepended as $" \mathrm { T } \mathrm { r a n } s \mathrm { c r i p t } : \{ \mathrm { t e x t } \} " $ to augment the MCQ prompt. Approximately 51% of the questions in the development set include timestamp references pointing to the audio excerpts containing the answers. To improve training and inference efficiency and to encourage the

Table 1: WER (%) Seed vs. Adapted Voxtral ASR. ∆ = Adapted−Seed. MLC25: official test set; MLC26 (<sup>†</sup>): development set.
<table><tr><td>Language</td><td></td><td>Seed Adapted</td><td>∆</td></tr><tr><td>English</td><td>7.0</td><td>6.9</td><td>-0.1</td></tr><tr><td>Spanish</td><td>8.2</td><td>7.9</td><td>-0.3</td></tr><tr><td>Vietnamese</td><td>9.1</td><td>9.0</td><td>-0.1</td></tr><tr><td>Thai</td><td>9.9</td><td>9.5</td><td>-0.4</td></tr><tr><td>Korean</td><td>8.8</td><td>10.1</td><td>+1.3</td></tr><tr><td>Portuguese</td><td>20.2</td><td>18.6</td><td>-1.6</td></tr><tr><td>Portuguese(Brazilian)†</td><td>18.4</td><td>10.9</td><td>-7.5</td></tr><tr><td>Russian</td><td>12.8</td><td>13.1</td><td>+0.3</td></tr><tr><td>Urdu†</td><td>140.1</td><td>13.0</td><td>-127.1</td></tr><tr><td>French</td><td>15.2</td><td>15.3</td><td>+0.1</td></tr><tr><td>French(Canadian)†</td><td>34.7</td><td>18.3</td><td>-16.4</td></tr><tr><td>German</td><td>17.5</td><td>17.3</td><td>-0.2</td></tr><tr><td>Tagalog†</td><td>49.7</td><td>20.6</td><td>-29.1</td></tr><tr><td>Italian</td><td>18.6</td><td>20.1</td><td>+1.5</td></tr><tr><td> $\mathrm { T u r k i s h } ^ { \dagger }$ </td><td>88.8</td><td>21.2</td><td>-67.6</td></tr><tr><td>Japanese</td><td>19.3</td><td>21.5</td><td>+2.2</td></tr><tr><td>MLC25 Overall</td><td>11.8</td><td>12.0</td><td>+0.2</td></tr><tr><td>MLC26 Overall†</td><td>26.2</td><td>15.2</td><td>-11.0</td></tr></table>

Table 2: Ablation offine-tuning components on Phase 1.
<table><tr><td>Training Data</td><td>Transc.</td><td>Crop</td><td>Acc.</td></tr><tr><td>3B Baseline</td><td>No</td><td>No</td><td>0.7162</td></tr><tr><td>+ NLLB-only</td><td>No</td><td>No</td><td>0.7382</td></tr><tr><td>+ Mixed</td><td>Yes</td><td>No</td><td>0.7283</td></tr><tr><td>+ Mixed</td><td>Yes</td><td>30s</td><td>0.7446</td></tr></table>

Speech-LLM to focus on specific audio content, we cropped the audio to a ±30s window around these automatically detected timestamps. This approach yielded our best Phase 1 score of 0.85. An ablation of the fine-tuning components is shown in Table 2 reporting results submitted in Phase 1.

## 4. In-Context Learning with Speech-LLM

We detected a systematic class-A prediction bias that affects both Voxtral model sizes. The 24B baseline model predicts class A for 63% of questions and class D for only 2%.

## 4.1. In-Context Learning

Prepending solved MCQ examples before the target question, i.e. few-shot In-Context Learning (ICL) corrects the bias in the 24B model for the English-audio/English-questions. Two textonly shots already shift class-A predictions from 63% to 44% and improve accuracy by +5.1 pp; scaling to six multimodal shots adds a further +1.5 pp, yielding our best score of 0.8095 (Table 3). Note that the same set of ICL examples is used for all evaluation instances, consisting exclusively of English examples extracted from the training set. We hypothesize that employing multilingual language-specific ICL strategies could further improve performance. Each multimodal shot places a 30 s audio clip in the user turn with an audio-focus prefix and MCQ text; the assistant turn is the bare answer letter. For 4-option questions, four shots cover one instance of each answer label (A, B, C, D) drawn from held-out audio, ensuring the model observes all four labels as valid outputs. For 2-option (A/B)

Table 3: Predicted label distributions and Phase 2 accuracy $( 9 , 4 7 0$ questions). All system use 30 s crop in inference and no ASR transcripts as augmented context. mm stands for multimodal shots.
<table><tr><td>System</td><td>A</td><td>B</td><td>C</td><td>D</td><td> $\operatorname { A c c } .$ </td></tr><tr><td>24B Baseline</td><td>63%</td><td>26%</td><td>9%</td><td>2%</td><td>0.743</td></tr><tr><td> $2 4 \mathrm { B } + \mathrm { I C L } { - 2 }$  text</td><td>44%</td><td>37%</td><td>14%</td><td>6%</td><td>0.795</td></tr><tr><td> $2 4 \mathrm { B } + \mathrm { I C L } { - 6 }$  mm</td><td>43%</td><td>37%</td><td>14%</td><td>6%</td><td>0.810</td></tr><tr><td>24B + ICL-2 + calib.</td><td>43%</td><td>37%</td><td>14%</td><td>6%</td><td>0.795</td></tr><tr><td>3B Baseline</td><td>65%</td><td>24%</td><td>8%</td><td>2%</td><td>0.701</td></tr><tr><td>3B fine-tuned</td><td>57%</td><td>32%</td><td>10%</td><td>1%</td><td>0.724</td></tr></table>

questions, two shots are used.

## 4.2. Inference-Time Calibration

We also evaluated an inference-time calibration strategy known as prior calibration (CBU, [20]), which subtracts null-prompt log-probabilities to eliminate positional priors. When applied to ICL-2, this method reached the same accuracy as the baseline ICL-2 (≈0.795), suggesting that in-context learning alone is sufficient to correct positional bias for the 24B model

## 5. Training-Free Retrieval Approach

## 5.1. A voice-anchored memory layer

Our third approach treats Task 2 not as a model-training problem but as a retrieval problem. We use a persistent voiceanchored memory layer that gives a standard LLM crosssession memory, semantic content memory, and a knowledge graph of facts and relationships [21], without training any component. We enroll the speech session in our system and then at query time we retrieve from it the relevant context needed to answer the question. The enrollment of audio files in this memory layer is accomplished by an enrollment pipeline which segments a multispeaker conversation by utterance through diarization, then a single Speech-LLM Voxtral Mini 3B [16] served via vLLM[22], performs the entire audio frontend: one call per utterance returns the transcription, language, gender, age, and both the acoustic and textual emotion labels. All of these extracted information are structured and stored in memory. The layer is then queried at runtime to assemble a structured fact sheet that is handed to a frozen, swappable downstream LLM and helps it answering the question based on the retrived information; the entire system is composed of pre-trained models at inference time. This approach allows us to enroll the multispeaker conversation once and then answer all the related questions efficiently without the need to process the audio file multiple times. For these experiments, we used LLM Qwen3-14B-$\bar { \bf A W Q } ^ { 3 }$

Shared utterance identifiers. Memory is organized as three persistent storage layers, illustrated in Figure 2: an acoustic-identity layer $L _ { \mathrm { a c } } ,$ a semantic-content layer $L _ { \mathrm { s e m } } ,$ , and a knowledge-graph layer $L _ { \mathrm { k g } }$ . Every turn ingested by the system — whether produced by diarization of a multi-speaker recording or by a chat turn — is assigned a single UUID u at ingestion, which is propagated as the primary key across the layers that record it. For audio utterances u is generated when the speaker embedding is written to $L _ { \mathrm { a c } }$ and reused as the record ID in $L _ { \mathrm { s e m } }$ and as the utterance-node identifier in $L _ { \mathrm { k g } } ;$ chat turns skip $L _ { \mathrm { a c } }$ and originate u in $L _ { \mathrm { s e m } }$ . The invariant is that whenever two layers store information about the same utterance, they store it under the same key, so cross-layer references resolve by direct lookup rather than approximate matching.

Acoustic identity $( L _ { \mathrm { a c } } ) .$ For each utterance, a 192- dimensional speaker embedding is extracted with Titanet-Large [23] and stored in a persistent ChromaDB collection. Records are tagged as enroll data (durable identity profiles for known speakers) or probe data (transient embeddings produced during a single query session). At query time, the layer supports identification (comparing a probe to enrolled profiles by top-k cosine retrieval and returning the closest identity above a similarity threshold) and resolution (matching a probe against other probes from the same session so that segments of the same unknown speaker are linked).

Semantic content $\left( L _ { \mathrm { s e m } } \right)$ . Utterance text—transcribed by Voxtral for audio and taken verbatim for chat—is encoded with all-MiniLM-L6-v2 [24] and stored in a parallel ChromaDB collection under the same UUIDs as $L _ { \mathrm { a c } }$ . Per-record metadata supports speaker-conditioned semantic search: a textual query can be restricted to utterances produced by one or more identified speakers, optionally filtered further by emotion, time, or dialogue membership.

Knowledge graph $( L _ { \mathrm { k g } } ) .$ . The knowledge graph layer answers how facts relate and how they evolve. It is a directed graph in NetworkX [25] with four types of nodes (speaker, utterance, entity, and dialogue) and two classes of edges. We use a local Qwen3-14B-AWQ served by vLLM as a local extractor to extract nodes and relationships from conversation transcription.

Structural edges are added eagerly when an utterance is ingested and require no LLM call:

• SAID (speaker → utterance)

• CONTAINS (dialogue → utterance)

• HAS PARTICIPANT (dialogue → speaker)

• KNOWS (speaker ↔ speaker), with co-occurrence counts accumulated across dialogues

Semantic edges computed by an LLM extractor:

• MENTIONS: from utterance to entity

• FACT: typed edges between entities

Each fact carries provenance and two timestamps, validfrom and valid-to: a new fact that contradicts an existing one does not delete it, but closes its validity interval, preserving history. Newly extracted entities are reconciled against the graph by an entity resolver that embeds each candidate with the same sentence-transformer used in $L _ { \mathrm { s e m } }$ and searches a dedicated entity collection, auto-merging above a high similarity threshold, treating low-similarity candidates as new, and deferring ambiguous cases to an LLM judge.

Emotion as cross-layer metadata. Each utterance is annotated with two emotion labels attached as metadata to $L _ { \mathrm { s e m } }$ and $L _ { \mathrm { k g } } \colon$ an acoustic label from prosody and a textual label from lexical content, both produced by Voxtral in the same call as the transcription. The two channels are kept independent rather than fused, since tone and words can disagree and both carry signal; acoustic emotion is computed only for audio utterances.

Fact sheet assembly. At query time, the retrieval pipeline reads the three layers and assembles a fact sheet. Given an optional probe clip, $L _ { \mathrm { a c } }$ identifies the speaker against the enrolled profiles; $L _ { \mathrm { s e m } }$ is queried with the question, filtered by the identified speaker(s); and $L _ { \mathrm { k g } }$ is read in one of three modes— profile, dialogue, or entity—selected by a lightweight question classifier, returning emotion counts, participant lists, or currently-valid facts respectively. The assembled fact sheet— per-utterance entries from $L _ { \mathrm { s e m } } .$ , speaker identity from $L _ { \mathrm { a c } } ,$ graph context from $L _ { \mathrm { k g } } ,$ and emotion labels—is the only context passed to the downstream LLM, with a prompt that constrains the model to rely on fact-sheet content alone.

![](images/08df33f1dfa7d81d953e673492d28a1ecfa202c4132262a25f3ec4babee04ee8.jpg)  
Figure 2: System architecture. Every utterance receives a UUID u at ingestion. Audio inputs are diarized into segments and enrolled into the three layers, while chat sessions skip the acoustic layer. At query time, the Retrieval component receives a user question (optionally with a probe audio clip), reads from each layer (dashed lines), and assembles a fact sheet that is handed to a swappable answer LLM.

## 5.2. Application to the MLC-SLM Task 2

Sessions are processed one at a time. Each conversation is diarized with pyannote.audio [26, 27, 28]; every resulting segment is embedded with Titanet, transcribed and attributetagged by Voxtral, and written to all three layers $( L _ { \mathrm { a c } } , \ L _ { \mathrm { s e m } } ,$ $L _ { \mathrm { k g } } )$ under a shared UUID. Diarized segments are clustered and matched against previously enrolled speakers, giving the system a cross-segment notion of who is speaking that is anchored in voice rather than in an external speaker label. Task 2 is posed as multiple choice. For each question, the system performs speaker-conditioned semantic retrieval over the enrolled utterances of the current session and traverses the knowledge graph, then assembles the fact sheet described in Section 5.1. The fact sheet, together with the question and its candidate options, is the only context given to a frozen Qwen answer LLM, which is constrained to emit a single option letter. Because identity, content, and emotion are resolved by the retrieval layer before the LLM is invoked, the answer model performs neither speaker verification nor long-context recall itself.

## 6. Results

Table 4 summarizes the accuracy scores of all three developed systems.

Fine-tuned Voxtral-3B. The combination of translated data augmentation and 30s cropping yielded our highest development performance (0.85). While the mixed dataset (see Table 2) and timestamp-based cropping improved accuracy by directing the audio encoder toward relevant segments, the gains from transcription-based context augmentation and overall finetuning did not generalize to the evaluation sets. The resulting drop in score to 0.72 is likely attributable to overfitting or the LLM learning language-specific shortcuts during the finetuning process to a small dataset.

Voxtral-24B with ICL. The 24B model with six multimodal in-context shots reaches 0.81 on Phase 2, our best result overall. ICL corrects the strong label bias while requiring zero parameter updates or data for training. However, prior calibration provide no further gain over ICL, see Table 3.

Table 4: Accuracy scores on the MLC-SLM Task 2 development and Phase-2 test sets.
<table><tr><td>System</td><td>Dev</td><td>Test</td></tr><tr><td>Official baseline</td><td>0.35</td><td>一</td></tr><tr><td>Voxtral-3B fine-tuned</td><td>0.85</td><td>0.72</td></tr><tr><td>Voxtral  $- 2 4 \mathrm { B } + \mathrm { I C L } { - 6 }$ </td><td>一</td><td>0.81</td></tr><tr><td>Train-free (multi-model)</td><td>0.78</td><td></td></tr><tr><td>Train-free (Voxtral frontend)</td><td>0.83</td><td>0.68</td></tr></table>

Training-free retrieval. We evaluated two variants dif fering only in the audio frontend: a multi-model pipeline (Whisper[29] + Wav2Vec2 + RoBERTa) and a unified Voxtral Mini 3B frontend. The unified frontend outperformed it on the dev set (0.83 vs. 0.78) while simplifying the pipeline, and was submitted to the evaluation set where it reached 0.68. This result is obtained without any training on the challenge corpus, indicating that a retrieval layer over frozen perception and language models is a competitive alternative to model adaptation, while remaining cheap to deploy and agnostic to the choice of downstream LLM.

## 7. Conclusions

We presented three systems for Task 2 of the second MLC-SLM challenge under a fully blind evaluation setting, utilizing no oracle segmentation, speaker labels, or diarization during inference. Our fine-tuning experiments demonstrate that NLLB-200 translation effectively bridges the 52% cross-lingual gap, and with timestamp-aware cropping achieved 0.85 on the development set; however it failed to generalize to the Phase 1 and Phase 2 evaluation sets due to overfitting. In contrast, applying in-context learning to the frozen Voxtral-24B model successfully mitigated strong pretrained label bias without parameter updates, achieving our best result of 0.81 on Phase 2. Finally, our training-free retrieval system (0.83 dev, 0.68 Phase 2) shows that a voice-anchored memory layer over frozen models is a competitive, low-cost alternative to task-specific adaptation. Together, these three approaches offer complementary trade-offs between accuracy, training cost, and deployment simplicity, placing the Eloquence team 5th in the final ranking.

## 8. Acknowledgments

This work has received funding from the European Union’s Horizon Europe research and innovation programme under the project ELOQUENCE (Grant Agreement No. 101135916). This work was supported by computational resources from the EuroHPC Joint Undertaking under the EuroHPC AI Factory grant EHPC-AIF-2026LS01-004.

## 9. References

[1] N. Zheng, Y. Lin, S. Tian, M. Li, Z. Lin, L. Xiao, and D. Tu, “Balancing ASR and diarization in end-to-end LLMs for multitalker speech recognition,” 2026.

[2] M. Shi, X. Xiao, R. Fan, S. Ling, and J. Li, “Train Short, Infer Long: Speech-LLM Enables Zero-Shot Streamable Joint ASR and Diarization on Long Audio,” 2026 IEEE International Conference on Acoustics, Speech and Signal Processing, 2026.

[3] P. Jing, W. Yucheng, F. Yangui, X. Yu, L. Xu, Z. Xizhuo, and Y. Kai, “A Survey on Speech Large Language Models,” arXiv preprint arXiv:2410.18908, 2024.

[4] W. Mingqiu, H. Wei, S. Izhak, W. Zelin, C. Chung-Cheng, W. Yuan, Cao amd Yongqiang, C. Nanxin, Z. Yu, S. Hagen, R. Paul, Z. Lukas, Y. Dian, M. Zhong, P. Golan, S. Nikhil, S. Johan, and W. Yonghui, “SLM: bridge the thin gap between speech and text foundation models,” arXiv preprint arXiv:2310.00230, 2023.

[5] L. Concina, J. Luque, A. Brutti, M. Matassoni, and Y. Zhang, “The Eloquence team submission for task 1 of MLC-SLM challenge,” arXiv preprint arXiv:2410.18908, 2024.

[6] C. Wang, H. Lu, X. Zhang, S. Liu, Y. Lu, J. Li, and Z. Wu, “Closing the modality reasoning gap for speech large language models,” in Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), M. Liakata, V. P. Moreira, J. Zhang, and D. Jurgens, Eds. San Diego, California, United States: Association for Computational Linguistics, Jul. 2026, pp. 18 821–18 835.

[7] D. Wang, J. Wu, J. Li, D. Yang, X. Chen, T. Zhang, and H. Meng, “MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark,” ArXiv, vol. abs/2506.04779, 2025.

[8] J. Peng, J. Du, C. Wang, H. Li, Y. Yang, Y. Wang, X. Gu, G. Chen, Y. Wang, J. Li, Z. Zhao, H. Wang, W. Tu, H. Li, D. Ma, L. Qian, Y. Xi, W. Wen, J. Guo, H. Zhang, S. Fan, W. Jiang, S. Wang, and K. Yu, “A Unified and Reproducible Experimentation Framework for Speech Understanding,” 2026.

[9] S. S. Kankanala, R. Chandra, and S. Ganapathy, “Benchmarking Humans And Machines On Complex Multilingual Speech Understanding Tasks,” IEEE International Conference on Acoustics, Speech and Signal Processing, 2026.

[10] S. Wang, Z. Sun, Z. Lin, C. Wang, Z. Pan, and L. Xie, “Msubench: Towards understanding the conversational multi-talker scenarios,” ArXiv, vol. abs/2508.08155, 2025.

[11] S. Fong, M. Matassoni, and A. Brutti, “Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages,” arXiv preprint arXiv:2508.05149 [, 2025.

[12] B. Mu, P. Guo, Z. Sun, S. Wang, H. Liu, M. Shao, L. Xie, E. S. Chng, L. Xiao, Q. Feng, and D. Wang, “Summary on The Multilingual Conversational Speech Language Model Challenge: Datasets, Tasks, Baselines, and Methods,” in 2026 IEEE Interna tional Conference on Acoustics, Speech and Signal Processing, 2026, pp. 19 442–19 446.

[13] Q. Wang, H. B. Sailor, J. H. M. Wong, T. Liu, S. Sun, W. Zhang, M. Huzaifah, N. F. Chen, and A. Aw, “Incorporating Contextual Paralinguistic Understanding in Large Speech-Language Models,” 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), pp. 1–8, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:280566781

[14] L. M. Maben, G. G. Lakshmy, S. Radhakrishnan, S. Arora, and S. Watanabe, “AURA: Agent for Understanding, Reasoning, and Automated Tool Use in Voice-Driven Tasks,” 2025 IEEE Automatic Speech Recognition and Understanding Workshop, pp. 1–4, 2025. [Online]. Available: https://api.semanticscholar.org/ CorpusID:280011205

[15] Y. Zhao, J. Huang, J. Hu, X. Wang, Y. Mao, D. Zhang, Z. Jiang, Z. Wu, B. Ai, A. Wang, W. Zhou, and Y. Chen, “SWIFT:A Scalable lightWeight Infrastructure for Fine-Tuning,” 2024. [Online]. Available: https://arxiv.org/abs/2408.05517

[16] A. H. Liu, A. Ehrenberg, A. Lo, C. Denoix, C. Barreau, G. Lample, J.-M. Delignon, K. R. Chandu, P. von Platen, P. R. Muddireddy et al., “Voxtral,” arXiv preprint arXiv:2507.13264, 2025.

[17] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in ICLR, 2022.

[18] M. M. Danilak, “langdetect: Language detection library ported from Google’s language-detection,” https://github.com/ Mimino666/langdetect, 2014, version 1.0.9.

[19] NLLB Team and M. R. C. jussa et al., “No language left behind:\` Scaling human-centered machine translation,” arXiv:2207.04672, 2022.

[20] Z. Zhao, E. Wallace, S. Feng, D. Klein, and S. Singh, “Calibrate Before Use: Improving Few-shot Performance of Language Models,” in Proc. ICML, 2021.

[21] N. H. Wils, S. P. Garijo, K. M. Blum, R. Gerndt, and T. Doernbach, “Smalltalk-KG: A Knowledge Graph Construction Framework for Dialog Personalization in Human-Robot Interaction,” 2026 IEEE International Conference on Advanced Robotics and its Social Impacts (ARSO), pp. 33–38, 2026. [Online]. Available: https://api.semanticscholar.org/CorpusID:288874560

[22] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. Gonzalez, H. Zhang, and I. Stoica, “Efficient memory management for large language model serving with PagedAttention,” in Proceedings ofthe 29th symposium on operating systems principles, 2023, pp. 611–626.

[23] N. R. Koluguri, T. Park, and B. Ginsburg, “Titanet: Neural model for speaker representation with 1d depth-wise separable convolutions and global context,” in 2022 IEEE international conference on acoustics, speech and signal processing, 2022, pp. 8102–8106.

[24] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “Minilm: Deep self-attention distillation for task-agnostic compression of pre-trained transformers,” Advances in neural information processing systems, vol. 33, pp. 5776–5788, 2020.

[25] A. Hagberg, P. J. Swart, and D. A. Schult, “Exploring network structure, dynamics, and function using NetworkX,” Los Alamos National Laboratory (LANL), Tech. Rep., 2007.

[26] H. Bredin, “Pyannote.audio 2.1 speaker diarization pipeline,” CNRS, Tech. Rep., 2023.

[27] ——, “pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,” in Proc. INTERSPEECH 2023, 2023.

[28] A. Plaquet and H. Bredin, “Powerset multi-class cross entropy loss for neural speaker diarization,” in Proc. INTERSPEECH 2023, 2023.

[29] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” in ICML, 2023.