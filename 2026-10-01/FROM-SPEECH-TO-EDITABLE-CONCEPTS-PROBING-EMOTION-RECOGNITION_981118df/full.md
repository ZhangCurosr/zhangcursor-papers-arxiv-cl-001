# FROM SPEECH TO EDITABLE CONCEPTS: PROBING EMOTION RECOGNITION WITH CONCEPT BOTTLENECK MODELS

Hezhao Zhang

Thomas Hain

School of Computer Science, University of Sheffield Sheffield, United Kingdom

## ABSTRACT

Speech emotion recognition (SER) is the task of assigning emotion labels to utterances. Early systems relied on acoustic features, whereas recent approaches combine multiple modalities, most commonly speech and text. Still, performance remains poor on many datasets. Large language models (LLMs) have therefore attracted interest for SER, as they can process diverse inputs jointly with instructions. However, direct audio input raises questions of explainability. To address similar questions in image classification, concept bottleneck models were introduced. This work adapts concept bottlenecks to SER to examine how individual predictions depend on transcripts, acoustic descriptions and speaker attributes. Experiments test three LLMs on CREMA-D, IEMO-CAP and MELD, with concepts extracted by separate tools. On scripted corpora, LLMs are strongly biased towards the transcript in the zero-shot setting, which lowers Macro-F1 from 27.8 to 5.8 on CREMA-D. Fine-tuning removes this bias, and the transcript raises Macro-F1 from 41.8 to 45.1. Removing speech rate changes 48% of Neutral predictions to Disgust on CREMA-D; removing intensity level on MELD changes predictions despite little change in Macro-F1. These findings show that aggregate performance changes alone do not capture the effects of concept removal on individual predictions.

Index Terms— speech emotion recognition, concept bottleneck models, large language models, interpretability

## 1. INTRODUCTION

Speech emotion recognition (SER) predicts the emotion label of an utterance from the speech signal [1]. Early systems trained classifiers on a wide range of carefully selected acoustic features such as pitch, energy and spectral descriptors [2, 3]. Later, neural networks learned acoustic representations directly from the raw waveform [4], and self-supervised speech representations turned out to be better as classifier input [5]. However, besides acoustic information, speech also carries lexical content [1, 6], and subsequent systems therefore combined the speech representations with text features from the transcript [7]. More recently, SER research has adopted large audio-language models (LALMs), models that extend large language models (LLMs) with an audio encoder and predict the emotion label directly from the audio given a text instruction [8, 9]. Despite these improvements recognition performance remains low on many corpora [10, 11]. Moreover, with the direct audio input, the LALM gives no indication of which information it relies on, so its decisions cannot be examined.

Several works make use of acoustic information in textual form. EmotionThinker [12] makes its reasoning visible by keeping the audio as input and generating a reasoning text that describes prosody before giving the label. SpeechCueLLM [13] and VowelPrompt [14] make the input tangible by extracting acoustic descriptions from the audio and providing them, together with the transcript, to a text LLM. However, retraining without individual descriptions or swapping descriptions between utterances [14] does not identify which description a fixed predictor’s decision depends on; generated reasoning may not faithfully reflect how the model reached its prediction [15].

Identifying which information affects a prediction requires the ability to control the model input and knowing what such input represents. A model that uses humanreadable form as input is therefore desirable. Image classification faced the same problem, and concept bottleneck models (CBMs) were introduced in response [16]. Here an image is first mapped to human-readable concepts, and the label is predicted from these alone. Hence every prediction can be traced to specific concepts and can be changed by editing them. Shin et al. [17] used such edits to examine how predictions respond to individual concepts.

We apply the CBM idea to SER: separate extractors describe each utterance by concepts that are human readable (Section 2). Then an LLM predicts the emotion label from these concepts, enabling single-concept removal with the predictor fixed to examine which class predictions change. Here concepts are the transcript, pitch, intensity and speech rate, as well as speaker age and gender. Together these describe the words spoken, intonation and speaker characteristics.

This work studies how the concepts affect emotion decisions: (1) what does each concept group contribute to recognition across corpora, before and after fine-tuning? (2) which acoustic concepts do the decisions depend on, and how does this dependence vary across corpora? The first question is addressed by comparing predictors that receive different concept groups (Section 4.1). The second by removing single acoustic concepts from the input of a predictor fine-tuned on all concepts (Section 4.2), with both analyses run on CREMA-D [18], IEMOCAP [19] and MELD [20] with three LLMs (Section 3).

<table><tr><td>Predict the emotion expressed in the utterance from the provided information. Transcript: (transcript)</td></tr><tr><td>Pitch level: (5 levels from very low to very high). Pitch variation: (5 levels from very low to very high).</td></tr><tr><td>Volume level: (5 levels from very low to very high). Volume variation: (5 levels from very low to very high).</td></tr><tr><td>Speech rate: (5 levels from very slow to very fast.</td></tr><tr><td>Estimated age: (6 bands from under 20 to 60+). Predicted sex: (Female or Male).</td></tr><tr><td>Allowed labels: (label set) Return exactly one label from the allowed labels. Do not provide an</td></tr></table>

## 2. SPEECH CONCEPT BOTTLENECK

A concept bottleneck model operates in two cascaded stages [16]: A concept extractor g maps an input x to a set of concepts c, and a predictor f predicts the label from c only,

$$
\hat { y } = f ( c ) , \qquad c = g ( x ) .\tag{1}
$$

Concepts are human-specified, interpretable properties of the input, whose predicted values form the input to f. A concept intervention changes one concept while f is held fixed and observes the change in yˆ [16, 17]. This work adopts this framework for SER: the input x is an utterance, the concepts c are properties of speech expressed as text, and the predictor f is an LLM that processes this text to produce one emotion label from a fixed set.

The concepts used are organised into three groups, each related to how emotion is expressed in speech (see Table 1). The transcript represents content and gives the words spoken, which carry emotion information through their meaning. The acoustic concepts are pitch, intensity and speech rate, three established acoustic correlates of emotion [21, 3, 13]. In addition, the expression and perception of emotion in speech vary with the speaker’s age and gender [22, 23]. To account for these differences, the speaker concepts include the age and gender of each speaker. Each group is obtained from a separate off-the-shelf extractor that was not trained on emotion labels (see Sec. 3.1).

## 3. IMPLEMENTATION AND EXPERIMENTS

## 3.1. Concept Extraction and Prompting

This study aims to provide insight into how the predictor makes use of the concepts, not how they are extracted. In order to extract concepts of high quality, each is obtained independently state of the art tools. The transcript is produced by Qwen3-ASR [24]. Pitch and intensity are measured with Praat [25], with the mean over the utterance giving the level and the standard deviation giving the variation. Speech rate is defined as the number of words in the transcript divided by the utterance duration [26]. Each acoustic measure is discretised into five levels with quantile thresholds fitted on the training set of each dataset and fold. The prompt is constructed with a simple description of the level (Table 1). The speaker concepts are derived from Vox-Profile [27], with the predicted age grouped into six ten-year bands from under 20

Table 1. Prompt template with all three concept groups. Unselected groups are omitted. Volume denotes intensity and Predicted sex the speaker’s gender.

to over 60. The predicted gender agrees with the metadata for 96.3% of CREMA-D and 95.2% of IEMOCAP utterances. Age metadata exist only for CREMA-D, where the predicted band is correct for only 35.9% of utterances and within one band for 78.2%. The prompt contains the selected concept groups and allowed labels, without specifying how concept values relate to emotions.

## 3.2. Experimental Design

To measure what each concept group contributes, the four combinations in Table 2 are evaluated with the zero-shot LLM and with a predictor fine-tuned on each combination alone. The gain from adding a group to utterance descriptions is considered its contribution, in zero-shot setting and after finetuning. To identify the relation between acoustic concepts and decisions , the predictor fine-tuned on all concepts is held fixed. The same utterance is assessed with and without a specific acoustic concept , (as defined in Sec. 2 ), thus providing some indication of its relevance.

## 3.3. Experimental Setup

Datasets. Three corpora are used in which emotion is carried by the words and by the voice to different degrees. In CREMA-D [18], twelve fixed neutral sentences are acted in six emotions, so only the voice carries emotion. IEMOCAP [19] holds scripted and improvised dyadic sessions performed by actors, and MELD [20] multi-party television dialogue, and in both the words carry emotion as well. CREMA-D has six classes and speaker-disjoint splits, IEMOCAP four classes and session-wise five-fold cross-validation, and MELD seven classes and official splits.

Models. Three open instruction-tuned LLMs serve as the predictor: Qwen2.5-7B-Instruct [28], Qwen2.5-Omni-7B [29], which is built on Qwen2.5-7B, and Llama-3.1-8B-Instruct [30]. Qwen2.5-Omni is also given the audio directly as a reference condition, zero-shot and fine-tuned.

Table 2. Test Macro-F1 (%) of each combination of concept groups, zero-shot and fine-tuned.
<table><tr><td></td><td></td><td colspan="4">Zero-shot</td><td colspan="4">Fine-tuned</td></tr><tr><td>Dataset</td><td>Model</td><td>T</td><td>A</td><td>TA</td><td>TAP</td><td>T</td><td>A</td><td>TA</td><td>TAP</td></tr><tr><td>CREMA-D</td><td>Qwen2.5</td><td>4.26</td><td>27.88</td><td>5.87</td><td>7.13</td><td>11.74</td><td>43.03</td><td>44.21</td><td>45.50</td></tr><tr><td></td><td>Qwen2.5-Omni</td><td>4.26</td><td>24.24</td><td>4.79</td><td>4.79</td><td>11.28</td><td>42.18</td><td>44.51</td><td>45.57</td></tr><tr><td></td><td>Llama 3.1</td><td>4.26</td><td>25.01</td><td>16.80</td><td>13.80</td><td>11.15</td><td>41.87</td><td>45.10</td><td>45.37</td></tr><tr><td>IEMOCAP</td><td>Qwen2.5</td><td>46.11</td><td>36.49</td><td>51.71</td><td>52.15</td><td>69.90</td><td>45.49</td><td>74.37</td><td>74.16</td></tr><tr><td></td><td>Qwen2.5-Omni</td><td>48.43</td><td>31.41</td><td>51.99</td><td>51.83</td><td>70.26</td><td>45.17</td><td>74.97</td><td>74.58</td></tr><tr><td></td><td>Llama 3.1</td><td>52.32</td><td>32.35</td><td>54.49</td><td>53.14</td><td>70.49</td><td>44.38</td><td>74.24</td><td>74.05</td></tr><tr><td>MELD</td><td>Qwen2.5</td><td>36.16</td><td>13.77</td><td>33.83</td><td>33.13</td><td>37.20</td><td>13.51</td><td>37.03</td><td>39.02</td></tr><tr><td></td><td>Qwen2.5-Omni</td><td>33.40</td><td>11.80</td><td>32.70</td><td>32.76</td><td>37.19</td><td>13.71</td><td>38.48</td><td>37.98</td></tr><tr><td></td><td>Llama 3.1</td><td>32.83</td><td>11.80</td><td>30.37</td><td>30.68</td><td>37.27</td><td>13.97</td><td>38.50</td><td>37.36</td></tr></table>

T: transcript; A: acoustic concepts; P: speaker concepts, all given as text. Bold: highest mean per model and setting; ties share the mark. Fine-tuned: mean of three runs (seed SD 0.1–3.6 points); IEMOCAP: five-fold means. Direct-audio reference: Table 3.

Training. Fine-tuning uses LoRA [31] with rank 16, α = 32 and dropout 0.05 on the attention and feed-forward projections. Optimisation uses AdamW [32] with a learning rate of $2 \times 1 0 ^ { - \hat { 4 } }$ , 10% warm-up and linear decay, for six epochs with a batch size of 16 in BF16. Each model is fine-tuned on each combination with three seeds and on each IEMOCAP fold.

Evaluation. Performance is measured as Macro-F1 under greedy decoding, and an output that cannot be parsed as a label counts as an error. The effect of a concept removal is the change in Macro-F1 relative to the full-input score of the same run.

## 4. RESULTS AND DISCUSSION

## 4.1. Predictive Performance

Table 2 reports Macro-F1 for each combination of concept groups, zero-shot and after fine-tuning. On CREMA-D, zero-shot models are strongly biased towards the transcript, which is detrimental. As the same sentences are spoken in every emotion, the models given the transcript alone classify every utterance as Neutral, giving 4.26 Macro-F1 for all of them. Although the acoustic concepts alone reach 27.88 for Qwen2.5, adding the transcript brings the score back to 5.87, with most predictions again Neutral, and the other two models drop in the same way.

However, the fine-tuned models show the opposite pattern. After fine-tuning, the models decide from the acoustic concepts, and Llama 3.1 reaches 41.87 with these alone. The transcript alone still gives only 11.15. Yet adding it to the acoustic concepts now raises the score to 45.10 rather than lowering it. The gain comes from combining the two inputs, and the reversal holds for all three models.

Table 3. Qwen2.5-Omni test Macro-F1 (%): highest-scoring concept combination from Table 2 versus direct audio input.
<table><tr><td rowspan="2">Dataset</td><td colspan="2">Zero-shot</td><td colspan="2">Fine-tuned</td></tr><tr><td>Concepts</td><td>Audio</td><td>Concepts</td><td>Audio</td></tr><tr><td>CREMA-D</td><td>24.24</td><td>54.95</td><td>45.57</td><td>78.67</td></tr><tr><td>IEMOCAP</td><td>51.99</td><td>69.32</td><td>74.97</td><td>82.66</td></tr><tr><td>MELD</td><td>33.40</td><td>34.91</td><td>38.48</td><td>41.07</td></tr></table>

Audio fine-tuning also adapts the encoder and projector. Fine-tuned scores average three runs.

On IEMOCAP and MELD, the same bias towards the transcript does no harm. These are conversational corpora, so the words change with the emotion, and the transcript alone already recognises it, giving 46.11 for Qwen2.5 on IEMO-CAP. With the acoustic concepts alone, zero-shot Qwen2.5 scores 36.49 on IEMOCAP, and adding the transcript raises this to 51.71, where the same addition lowered the score on CREMA-D. The acoustic concepts still improve recognition on IEMOCAP, where fine-tuning with both inputs reaches 74.37 against 69.90 with the transcript, but gains on MELD are small and inconsistent across models: the same comparison gives 37.03 against 37.20 for Qwen2.5.

Adding the speaker concepts changes the fine-tuned score by less than two points on every corpus, without a consistent direction. These small gains may reflect both limited within-speaker variation and errors in the extracted speaker concepts. Even at its best, the concept input stays below direct audio, and the gap depends on the corpus. Fine-tuned Qwen2.5-Omni reaches 45.57 on CREMA-D from concepts but 78.67 from audio, while the gap shrinks to 7.69 on IEMO-CAP and 2.59 on MELD (Table 3). The gap is largest where the emotion lies in the voice and smallest where it lies in the words.

![](images/9eb260439ef3f395dfe83de699d334dd4e074610227a5451e400e1f957e3f544.jpg)  
Fig. 1. Changes in Macro-F1 relative to the fine-tuned TAP scores in Table 2, after removing one acoustic concept while keeping the predictor fixed. Points show three-run means and error bars show ±1 standard deviation. IEMOCAP folds are averaged equally within each run.

## 4.2. Concept Interventions

Fig. 1 shows the change in Macro-F1 when one acoustic concept is removed from the input of the predictor fine-tuned on all concepts. How much a model depends on single acoustic concepts, and on which, follows the corpus. On CREMA-D, where the emotion lies in the acoustic concepts, removing any of the five lowers Qwen2.5’s Macro-F1, by 2.49 to 6.30. The largest loss comes from speech rate. In this corpus, speech rate separates Neutral from Disgust: Neutral utterances tend to be fast, with 37.0% in the fastest level and 6.5% in the slowest, whereas Disgust utterances tend to be slow, with 11.9% in the fastest level and 29.4% in the slowest. IEMOCAP, where the transcript carries more of the emotion, loses less than 0.6 for three of the five concepts. The exceptions are intensity level and variation, at 2.12 and 1.93, and intensity is again the concept that separates a class pair: 35.0% of Sad utterances have the lowest intensity level against 13.8% of Neutral. On MELD, no removal changes the score by more than 0.3. There, the emotions are spoken at nearly the same average speech rate, intensity and pitch, so the acoustic concepts separate them little: even the two classes furthest apart on speech rate, Happy and Disgust, differ by 0.7 of a level. The pattern holds for all three models, although for Qwen2.5-Omni pitch level and speech rate cost about the same on CREMA-D.

Fig. 2 removes the concept with the largest loss on CREMA-D and IEMOCAP, speech rate and intensity level, and intensity level on MELD for comparison, and follows the predictions that change. For each corpus it shows the label that changes most often, as a share of its predictions, and the label it turns into, split by the utterance’s level of the removed concept. These changes come almost entirely from the two highest or the two lowest levels of the removed concept, and few from the middle levels. Without speech rate, 48% of the Neutral predictions on CREMA-D become Disgust, all from the two fastest levels. Removing speech rate changes these Neutral predictions to Disgust, with most transitions occurring at high speech-rate levels. Without intensity level, 6% of the Sad predictions on IEMOCAP become Neutral, all from the two lowest levels. Angry and Happy predictions move to Neutral in the same way, from the loudest levels. On MELD, 7% of the Happy predictions become Neutral, mostly from the two loudest levels. Predictions also move from Neutral to Happy, despite little change in Macro-F1. The three models change the same levels and differ only in the size of the CREMA-D top level, from 32 to 40. These results reveal corpus-dependent prediction sensitivity to concept removal.

![](images/c68606a63f4fda117ac13c72fc158099eb3c3e8903e0ff894553553416072b52.jpg)  
level of the removed concept (1 lowest → 5 highest)  
Fig. 2. Predictions that change when one concept is removed, as a share of all predictions of the first class (%), by the utterance’s level of the removed concept. Bars pool the three models and whiskers span the per-model values.

## 5. CONCLUSION

The concept bottleneck framework was adopted for SER: each utterance is represented by transcript, acoustic and speaker concepts, and an LLM predicts the emotion from these concepts alone. Experiments were conducted in zeroshot and fine-tuned settings. In the zero-shot setting, the transcript concept dominates outcomes, which is found to be detrimental on scripted corpora, where the transcript is neutral. After fine-tuning, the contributions of transcripts and acoustic concepts vary across corpora. Concept-based prediction still underperforms direct audio input, with the largest gap on CREMA-D, where the fixed transcripts carry no emotion information. Removing individual acoustic concepts from a fixed predictor reveals corpus-dependent changes in recognition performance and class predictions. The selected class transitions concentrate at particular concept levels, while on MELD predictions change despite little change in Macro-F1. These results show that aggregate scores can conceal changes in individual predictions. The concept bottleneck enables examination of dependence through controlled changes to the predictor’s inputs.

## 6. COMPLIANCE WITH ETHICAL STANDARDS

This study uses existing data from the CREMA-D, IEMO-CAP, and MELD datasets, with no new participant recruitment or data collection. Ethical approval was not required for this secondary analysis.

## 7. ACKNOWLEDGMENTS

The authors declare no conflicts of interest. The authors used Claude Code and OpenAI Codex to assist with language editing and clarity improvements throughout the manuscript, as well as the development of experimental code. The authors reviewed and verified the resulting revisions and code and take full responsibility for the final content.

## 8. REFERENCES

[1] C. M. Lee and S. S. Narayanan, “Toward detecting emotions in spoken dialogs,” IEEE Transactions on Speech and Audio Processing, vol. 13, no. 2, pp. 293–303, Mar. 2005.

[2] B. Schuller, S. Steidl, and A. Batliner, “The INTERSPEECH 2009 emotion challenge,” in Interspeech 2009. Sept. 2009, pp. 312–315, ISCA.

[3] F. Eyben et al., “The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing,” IEEE Trans. Affect. Comput., vol. 7, no. 2, pp. 190–202, Apr. 2016.

[4] G. Trigeorgis et al., “Adieu features? End-to-end speech emotion recognition using a deep convolutional recurrent network,” in Proc. ICASSP, 2016, pp. 5200–5204.

[5] L. Pepino, P. Riera, and L. Ferrer, “Emotion Recognition from Speech Using wav2vec 2.0 Embeddings,” in Interspeech 2021. Aug. 2021, pp. 3400–3404, ISCA.

[6] S. Yoon, S. Byun, and K. Jung, “Multimodal speech emotion recognition using audio and text,” in Proc. SLT, 2018, pp. 112– 118.

[7] D. Sun, Y. He, and J. Han, “Using auxiliary tasks in multimodal fusion of wav2vec 2.0 and BERT for multimodal emotion recognition,” in Proc. ICASSP, 2023, pp. 1–5.

[8] C. Wang, M. Liao, Z. Huang, J. Wu, C. Zong, and J. Zhang, “BLSP-Emo: Towards empathetic large speech-language models,” in Proc. EMNLP, 2024, pp. 19186–19199.

[9] J. Mai, X. Xing, W. Chen, Y. Fang, and X. Xu, “AA-SLLM: An Acoustically Augmented Speech Large Language Model for Speech Emotion Recognition,” in Proc. Interspeech. 2025, pp. 4328–4332, ISCA.

[10] Z. Ma et al., “EmoBox: Multilingual Multi-corpus Speech Emotion Recognition Toolkit and Benchmark,” in Interspeech 2024. Sept. 2024, pp. 1580–1584, ISCA.

[11] H. Zhang, H.-C. Chou, S. Narayanan, and T. Hain, “VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs,” arXiv:2603.08936, 2026.

[12] D. Wang, S. Liu, T. Zhang, Y. Chen, J. Li, and H. Meng, “EmotionThinker: Prosody-Aware Reinforcement Learning for Explainable Speech Emotion Reasoning,” in Proc. ICLR, 2026, pp. 153708–153733.

[13] Z. Wu, Z. Gong, L. Ai, P. Shi, K. Donbekci, and J. Hirschberg, “Beyond Silent Letters: Amplifying LLMs in Emotion Recognition with Vocal Nuances,” in Findings ofACL: NAACL. 2025, pp. 2202–2218, Association for Computational Linguistics.

[14] Y. Wang et al., “VowelPrompt: Hearing Speech Emotions from Text via Vowel-level Prosodic Augmentation,” in Proc. ICLR, 2026, pp. 20439–20460.

[15] M. Turpin, J. Michael, E. Perez, and S. Bowman, “Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting,” in Proc. NeurIPS, 2023, vol. 36, pp. 74952–74965.

[16] P. W. Koh et al., “Concept Bottleneck Models,” in Proc. ICML, 2020, vol. 119 of PMLR, pp. 5338–5348.

[17] S. Shin, Y. Jo, S. Ahn, and N. Lee, “A Closer Look at the Intervention Procedure of Concept Bottleneck Models,” in Proc. ICML, 2023, vol. 202 of PMLR, pp. 31504–31520.

[18] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset,” IEEE Trans. Affect. Comput., vol. 5, no. 4, pp. 377–390, Oct. 2014.

[19] C. Busso et al., “IEMOCAP: Interactive emotional dyadic motion capture database,” Lang. Resour. Eval., vol. 42, no. 4, pp. 335–359, Dec. 2008.

[20] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, “MELD: A Multimodal Multi-Party Dataset for Emotion Recognition in Conversations,” in Proc. ACL. 2019, pp. 527–536, Association for Computational Linguistics.

[21] K. R. Scherer, “Vocal communication of emotion: A review of research paradigms,” Speech Communication, vol. 40, no. 1-2, pp. 227–256, Apr. 2003.

[22] A. Sen, D. Isaacowitz, and A. Schirmer, “Age differences in vocal emotion perception: On the role of speaker age and listener sex,” Cogn. Emot., vol. 32, no. 6, pp. 1189–1204, 2018.

[23] A. Lausen and A. Schacht, “Gender Differences in the Recognition of Vocal Emotions,” Front. Psychol., vol. 9, June 2018, Art. no. 882.

[24] X. Shi et al., “Qwen3-ASR Technical Report,” arXiv:2601.21337, 2026.

[25] Y. Jadoul, B. Thompson, and B. de Boer, “Introducing Parselmouth: A Python interface to Praat,” J. Phon., vol. 71, pp. 1–15, Nov. 2018.

[26] S. Yildirim et al., “An acoustic study of emotions expressed in speech,” in Proc. Interspeech, Oct. 2004, pp. 2193–2196.

[27] T. Feng et al., “Vox-Profile: A Speech Foundation Model Benchmark for Characterizing Diverse Speaker and Speech Traits,” arXiv:2505.14648, 2025.

[28] Qwen et al., “Qwen2.5 Technical Report,” arXiv:2412.15115, 2024.

[29] J. Xu et al., “Qwen2.5-Omni Technical Report,” arXiv:2503.20215, 2025.

[30] A. Grattafiori et al., “The Llama 3 Herd of Models,” arXiv:2407.21783, 2024.

[31] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” in Proc. ICLR, 2022.

[32] I. Loshchilov and F. Hutter, “Decoupled Weight Decay Regularization,” in Proc. ICLR, 2019.