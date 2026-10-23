# PREDICTING MUSIC VALENCE AND AROUSAL VIA TEXTUAL AND MULTIMODALIN-CONTEXT LEARNING IN LARGE LANGUAGE MODELS

Chi-En Dai, Chih-Pin Tan, Yi-Hsuan Yang

Graduate Institute of Communication Engineering, National Taiwan University, Taiwan

## ABSTRACT

While arousal can be reliably predicted from musical audio, emotional valence remains difficult to model due to its subjective and context-dependent nature. In this work, we investigate the potential of in-context learning (ICL) with large language models to advance continuous valence and arousal prediction. Rather than predicting emotion scores directly, our framework leverages contextual demonstrations from an auxiliary dataset to refine the initial predictions produced by a small, task-specific base model. We systematically evaluate a text-based ICL variant using lyrics and generated audio captions, alongside a multimodal variant that incorporates raw audio signals. Using the entire MER60 dataset strictly as a held-out test set, our experiments demonstrate substantial performance gains, particularly for valence. Text-based ICL improves the valence R<sup>2</sup> from 0.110 to 0.427 with 80 demonstration shots, while the multimodal variant achieves 0.346 with only 10 shots. These findings suggest that training-free ICL offers an effective inference-time strategy for current language models to predict continuous music emotion.

Index Terms— Music emotion recognition

## 1. INTRODUCTION

Music emotion recognition (MER) aims to estimate the perceived emotional response of listeners [1, 2]. Emotions are typically modeled using categorical labels or continuous dimensions. In this work, we focus on the continuous valence– arousal framework [3, 4], where arousal indicates emotional energy and valence measures emotional positivity or negativity. While numerous classical [5, 6] and deep learning methods [7, 8] have been proposed to predict valence and arousal from musical audio, valence prediction remains challenging due to its highly abstract and context-dependent nature [5, 9].

Recently, large language models (LLMs) and their multimodal counterparts (MLLMs) have demonstrated remarkable capabilities across diverse tasks [10–13]. However, their direct zero-shot performance in continuous valence–arousal prediction from musical audio remains largely unexplored. In our pilot study, we employed MER60 [14], a benchmark dataset of 60 Western pop songs compiled before 2011, to test two representative models: GPT-5 (a text-based LLM)

and Qwen3-Omni 30B [13] (an MLLM). Evaluating with the coefficient of determination $( R ^ { 2 } )$ , where 1.0 indicates perfect prediction and negative values signal severe performance degradation [15], GPT-5 achieved $R _ { A } ^ { 2 } ~ = ~ 0 . 6 9 9$ for arousal but $R _ { V } ^ { 2 } = - 0 . 8 0 1$ for valence. Similarly, Qwen3-Omni 30B obtained $R _ { A } ^ { 2 } = 0 . 7 6 4$ and $R _ { V } ^ { 2 } = 0 . 0 3 7$ . While both models perform competitively on arousal—approaching established baselines in the literature $( \mathbf { e . g . } , R _ { A } ^ { 2 } = 0 . 8 0 0$ reported in [6])—their zero-shot valence predictions fail dramatically, falling far below prior audio-based benchmarks (e.g., $R _ { V } ^ { 2 } = 0 . 2 7 0 \mathrm { i n } [ 6 ] )$ . Crucially, because the songs in MER60 have been released for over 15 years, likely the text descriptions or audio clips related to them were already included in the massive pre-training corpora of the LLM/MLLM. The fact that zero-shot valence prediction remains poor indicates that potential song exposure or text-based memorization alone is insufficient for zero-shot valence reasoning, highlighting the need for explicit, context-guided alignment.

The goal of this work is therefore to study whether we can effectively improve the ability of current LLMs and MLLMs to predict valence from music. More specifically, since training such large-scale models from scratch is computationally prohibitive, we ask whether we can leverage the prominent in-context learning (ICL) paradigm [16,17]. By exposing pretrained LLMs and MLLMs to contextual demonstrations consisting of music audio and emotion labels from an auxiliary dataset (specifically, CH818 [18]) that is completely disjoint from our test set (i.e., MER60), we aim to boost their valence prediction performance. Because this ICL approach operates entirely at inference time and requires no weight updates or task-specific fine-tuning [16], it provides a straightforward, flexible, and practical solution for MER.

Specifically, text-only LLMs cannot process raw audio signals directly, necessitating text-based “surrogates” to represent the musical content of demonstration songs during ICL. Therefore, as our first research question, we evaluate the effectiveness of song lyrics versus automatic audio captions as surrogates. We consider lyrics under the assumption that they convey rich semantic and thematic context essential for valence perception. Similarly, for audio captions, large audiolanguage models (e.g., Qwen-Audio [19]) can be prompted to extract high-level descriptions such as emotional atmosphere and mood dynamics. Furthermore, while MLLMs can ingest raw audio signals directly, we similarly evaluate whether augmenting the audio input with lyrics enhances valence prediction in a multimodal ICL setting.

![](images/8d7e72b26088095bf5f643266c2a09692a396ffd05dd763f1412dd7128742978.jpg)  
Fig. 1: Overview of the proposed framework. A base audio model produces initial valence–arousal (VA) predictions, which are refined via in-context learning using either a text-only LLM (bottom) or a multimodal LLM (top).

As our second research question, we study whether incorporating a small, task-specific deep learning model trained explicitly for valence prediction can further improve result. The core idea is that a compact base model can collaborate with the large language models by simplifying the in-context learning task. Specifically, instead of asking the language model to predict absolute valence values from scratch, we use the base model’s initial prediction as an anchor and prompt the LLM/MLLM to estimate a residual offset. This predicted delta is then added to the base estimate to refine the final emotion scores. This base model is trained using the entire auxiliary dataset CH818, ensuring that the overall framework maintains strict cross-dataset generalization.

To provide further insights, we systematically investigate key factors such as the required number of demonstration shots and the impact of LLM/MLLM model capacity. Furthermore, we conduct all experiments for both valence and arousal to examine whether the observed gains are dimensionspecific. Our findings show that ICL can indeed greatly improve the valence prediction capabilities of current LLMs and MLLMs, though current text-based and multimodal architectures exhibit different optimal configurations.

## 2. RELATED WORK

MER research can be categorized in several ways, according to the emotion model (categorical or continuous emotions [1]), the nature of affect (perceived or felt emotion [20]), or the input modality (e.g., audio signals, lyrics [9], or symbolic representations like MIDI [21]). In recent years, deep learning architectures incorporating convolutional/recurrent networks and pre-trained backbones have been studied for various MER settings [8, 21, 22]. Here, we focus specifically on continuous, perceived MER from audio clips.

The annotation quality of training datasets strongly affects the generalizability of MER models [6]. Recent audiobased continuous MER studies mostly rely on the DEAM dataset [23], largely due to its public accessibility. However, DEAM was annotated via crowdsourcing, leading to variable label quality. In contrast, older datasets such as MER60 [14] and CH818 [18] were collected with expert-supervised annotation protocols, yielding substantially higher reliability, though restricted distribution has historically limited their usage. As the creators of CH818 and MER60, we leverage both in this study. For evaluation integrity, we utilize CH818 (covering 818 Chinese pop clips with medium-to-high annotation rigor) as our auxiliary dataset for base model training and ICL demonstrations, while using the entire MER60 (60 Western pop with high annotation rigor) exclusively as the test set.

Recent advances in LLMs/MLLMs have spurred growing interest in applying foundation models to music [13, 19]. However, their application to MER remains nascent. To date, existing work such as Yang et al. [24] primarily explores LLMs for categorical emotion annotation for symbolic MIDI. In contrast, our work targets the more challenging task of continuous valence–arousal regression directly from musical audio. Furthermore, while ICL has proven highly effective at enabling LLMs to perform complex reasoning via demonstrations [16, 17], its potential for continuous regression in audio and music remains largely unexplored.

## 3. METHODOLOGY

Our framework consists of two stages, as illustrated in Fig. 1. Given an input audio x, a base model f(·) first predicts an initial valence–arousal (VA) estimate $\mathbf { b } = f ( \mathbf { x } ) \in [ 1 , 9 ] ^ { 2 }$ , where valence and arousal scores are normalized to the standard 1–9 scale. A second-stage LLM or MLLM g(·) then predicts a calibration offset ∆, yielding the final prediction $\hat { \mathbf { y } } = \mathbf { b } + \boldsymbol { \Delta } ,$ which is clipped to the valid VA range. Predicting the residual ∆ instead of absolute VA values allows the second-stage model to focus only on correcting the base prediction.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td rowspan="2">Training Data</td><td colspan="2">DEAM [23]</td><td colspan="2">CH818 [18]</td><td colspan="2">MER60 [14]</td></tr><tr><td> $R _ { V } ^ { 2 }$ </td><td> $R _ { A } ^ { 2 }$ </td><td> $R _ { V } ^ { 2 }$ </td><td> $R _ { A } ^ { 2 }$ </td><td> $R _ { V } ^ { 2 }$ </td><td> $R _ { A } ^ { 2 }$ </td></tr><tr><td>SVR-based [6]</td><td>&lt;1M</td><td>CH818</td><td>N/A</td><td>N/A</td><td>0.250</td><td>0.770</td><td>0.270</td><td>0.800</td></tr><tr><td>VGGish [25]</td><td>62M</td><td>DEAM</td><td>0.520</td><td>0.611</td><td>0.122</td><td>0.256</td><td>-0.071</td><td>0.424</td></tr><tr><td>Music2Emo [8]</td><td>100M</td><td>DEAM+EmoMusic+PMemo</td><td>0.518</td><td>0.623</td><td>0.204</td><td>0.276</td><td>-0.438</td><td>0.511</td></tr><tr><td>Base model (ours)</td><td>91M</td><td>DEAM</td><td>0.542</td><td>0.529</td><td>-0.024</td><td>0.099</td><td>-0.127</td><td>0.660</td></tr><tr><td>Base model (ours)</td><td>91M</td><td>CH818</td><td>-0.062</td><td>0.168</td><td>0.430</td><td>0.782</td><td>0.110</td><td>0.783</td></tr></table>

Table 1: Comparison of baseline models under within- and cross-dataset settings.

![](images/3cbfb077eaf9b1bd5c23d250b3974f63b4e4f1c922c2d4f1b7cf3f2b79af80fb.jpg)  
Fig. 2: Prompt structure of text and multimodal ICL, where $( v _ { \mathrm { b a s e } } , a _ { \mathrm { b a s e } } ) = \mathbf { b }$ denotes the prediction of the base model, and $( v _ { \mathrm { g t } } , a _ { \mathrm { g t } } ) = \mathbf { y }$ the ground-truth VA values for a clip.

In our implementation, the base model follows the audio branch of Music2Emo [8], replacing MERT [26] with BEATs embeddings [27]. For each clip, 5-second BEATs features are averaged into a 768-dimensional representation and fed into an MLP with batch normalization, ReLU, and dropout. The model is trained using AdamW (learning rate $1 0 ^ { - 4 } )$ , batch size 64, and mean squared error (MSE) loss for 150 epochs.

As the zero-shot performance of current LLMs or MLLMs for VA prediction is unsatisfactory, we employ ICL by prepending N demonstrations at inference time for each testing audio clip (i.e., the “query”), where N denotes the number of shots. We investigate two variants in this work.

For text-based ICL, as shown in Fig. 2(a), each demonstration consists of the base prediction b, lyrics ℓ, audio description s, ground-truth VA values y, and the target offset $\Delta = \mathbf { y } - \mathbf { b }$ . The N demonstrations and the query are serialized into a prompt, enabling the LLM to predict the calibration offset $\bar { \Delta } = \bar { g } ^ { ( \mathrm { t e x t - I C L } ) } ( \mathbf { b } , \ell , s )$ for the testing clip.

To obtain rich textual surrogates for the musical content, lyrics ℓ are transcribed using Whisper-large-v3 [28] on vocal tracks isolated via HTDemucs source separation [29]. For audio descriptions s, we prompt Qwen2-Audio [19] to generate one coherent text paragraph in English for each input audio clip that reflect “the listener’s feelings” covering the following aspects: i) overall emotion or mood of the song, ii) energy level and tempo feeling, iii) emotional color of melody and harmony, and iv) vocal or instrumental expression.

For multimodal ICL (MICL), as shown in Fig. 2(b), the multimodal variant replaces the audio description with the original audio while maintaining the same underlying prompt structure, yielding $\Delta = g ^ { ( \mathrm { M C L } ) } ( \mathbf { b } , \boldsymbol { \ell } , \mathbf { x } )$ . Audio clips are processed directly by the MLLM’s native audio encoder, interleaving audio tokens with text in the context window. This way, the model utilizes both acoustic and textual information.

## 4. EXPERIMENT

We use CH818 [18] for base model training and ICL demonstrations, and MER60 [14] as the held-out test set. CH818 consists of 818 clips (30s each) annotated by 3 dedicated music experts, providing consistent medium-to-high annotation reliability. MER60 comprises 60 clips (also 30s each), each labeled by 40 human subjects in a controlled laboratory setup using a 2D emotional plane interface, providing ground truth labels with high annotation rigor ideal for evaluating ICL refinement. For each clip, we use the average VA values across annotators as the ground truth.

## 4.1. Evaluation of the Base Model

We compare our base model with three existing ones, SVR [6], VGGish [25], and Music2Emo [8], trained with different datasets. As shown in Table 1, while all models perform well on the validation split of their native training data, they generalize poorly to the out-of-domain MER60, particularly for valence prediction. Our CH818-trained base model achieves the best cross-dataset performance among the deep learning models while remaining competitive with the classical SVR baseline, motivating its usage as the foundational anchor.

<table><tr><td>Number of shots</td><td> $\overline { { R _ { V } ^ { 2 } } }$ </td><td> $\overline { { R _ { A } ^ { 2 } } }$ </td></tr><tr><td>0-shot</td><td>-0.801</td><td> $\overline { { 0 . 6 9 9 } }$ </td></tr><tr><td>5-shot</td><td> $0 . 1 8 8 \pm 0 . 0 3 6$ </td><td> $0 . 8 0 5 \pm 0 . 0 1 4$ </td></tr><tr><td>10-shot</td><td> $0 . 2 7 4 \pm 0 . 0 5 4$ </td><td> $0 . 8 5 8 \pm 0 . 0 1 5$ </td></tr><tr><td>20-shot</td><td> $0 . 2 7 1 \pm 0 . 0 2 5$ </td><td>0.867 ± 0.010</td></tr><tr><td>40-shot</td><td> $0 . 3 5 4 \pm 0 . 0 3 6$ </td><td> $0 . 8 7 0 \pm 0 . 0 1 3$ </td></tr><tr><td>80-shot</td><td> $\mathbf { 0 . 4 2 7 \pm 0 . 0 5 4 }$ </td><td> $\mathbf { 0 . 8 7 4 } \pm 0 . 0 1 0$ </td></tr></table>

(a) Scaling effect of text-based ICL
<table><tr><td>ID</td><td>Base VA</td><td>Description</td><td>Lyric</td><td> $\overline { { R _ { V } ^ { 2 } } }$ </td><td> $\overline { { R _ { A } ^ { 2 } } }$ </td></tr><tr><td>1 2</td><td></td><td>√</td><td></td><td>0.002 0.209</td><td>0.535 0.676</td></tr><tr><td>3</td><td></td><td></td><td>√</td><td>-0.271</td><td>0.085</td></tr><tr><td>4</td><td>√</td><td>√</td><td></td><td>0.306</td><td>0.655</td></tr><tr><td>5</td><td>√</td><td></td><td>√</td><td>0.137</td><td>0.259</td></tr><tr><td>6</td><td></td><td>√</td><td>√</td><td>0.104</td><td>0.671</td></tr><tr><td>7</td><td>√</td><td>√</td><td>√</td><td>0.427</td><td>0.874</td></tr></table>

(b) Ablation study of input components under 80-shot  
Table 2: Text-based ICL results (mean±STD over five runs).

## 4.2. Evaluation of Text-based ICL and MICL

We use GPT-5 as the LLM for text-based ICL, and Qwen3- Omni 30B [13] for MICL. All inference runs adopt a decoding temperature of 0, and reported results are averaged over 5 independent runs with randomly sampled demonstrations.

Table 2(a) shows the effect of the number of demonstrations for text-based ICL. Without ICL, GPT-5 obtains a much lower $R _ { V } ^ { 2 }$ than the base model. Introducing only five shots greatly improves both $R _ { V } ^ { 2 }$ and $R _ { A } ^ { 2 }$ , while increasing the number of shots further benefits $R _ { V } ^ { 2 }$ . This indicates that contextual demonstrations effectively guide refinement, and that larger in-context sets provide richer information for valence.

Table 2(b) reports the corresponding input ablation under $N = 8 0 ( { \mathrm { i . e . } }$ ., the best setting here). Using the base prediction alone does not work well, while removing the base prediction also degrades performance, validating the proposed two-stage framework. Among individual inputs, the audio description is the most informative, whereas lyrics alone contribute little. The best performance is achieved by combining all three inputs, suggesting that the base prediction provides a stable anchor, the audio description supplies the primary contextual information, and the lyrics offer complementary cues.

Table 3 reports the result for MICL. Similar to text-based ICL, the MLLM performs poorly without demonstrations but improves substantially with only five shots, mainly for valence. Unlike text-based ICL, however, performance peaks at 10 shots and slightly degrades thereafter. We conjecture that the increasing context length caused by audio inputs limits the effectiveness of incorporating more demonstrations.

<table><tr><td>Number of shots</td><td> $\overline { { R _ { V } ^ { 2 } } }$ </td><td> $\overline { { R _ { A } ^ { 2 } } }$ </td></tr><tr><td>0-shot</td><td>0.037</td><td> $\overline { { 0 . 7 6 4 } }$ </td></tr><tr><td>5-shot</td><td> $0 . 2 8 3 \pm 0 . 1 0 9$ </td><td> $0 . 7 3 0 \pm 0 . 0 0 8$ </td></tr><tr><td>10-shot</td><td> ${ \bf 0 . 3 4 6 \pm 0 . 0 4 6 }$ </td><td> $0 . 7 6 8 \pm 0 . 0 1 8$ </td></tr><tr><td>20-shot</td><td> $0 . 2 4 7 \pm 0 . 0 4 1$ </td><td> $0 . 7 8 2 \pm 0 . 0 1 3$ </td></tr><tr><td>40-shot</td><td> $0 . 2 4 5 \pm 0 . 0 3 1$ </td><td> $\mathbf { 0 . 7 8 6 \pm 0 . 0 4 1 }$ </td></tr><tr><td>80-shot</td><td> $0 . 2 3 8 \pm 0 . 0 5 1$ </td><td> $0 . 7 8 3 \pm 0 . 0 2 6$ </td></tr></table>

(a) Scaling effect of multimodal ICL
<table><tr><td>ID</td><td>Base VA</td><td>Description</td><td>Lyric</td><td> $\overline { { R _ { V } ^ { 2 } } }$ </td><td> $\overline { { R _ { A } ^ { 2 } } }$ </td></tr><tr><td>1 2 3</td><td>√</td><td>√</td><td>√</td><td>0.176 -0.487 -0.339</td><td>0.796 0.760 0.663</td></tr><tr><td>4 5</td><td>√ √</td><td>√</td><td>√</td><td>0.251 0.346</td><td>0.773 0.768</td></tr><tr><td>6 7</td><td>√</td><td>√ √</td><td>√ √</td><td>-0.321 0.263</td><td>0.766 0.760</td></tr></table>

(b) Ablation study of input components under 10-shot  
Table 3: MICL results (mean±STD over five runs).

Table 3(b) presents the input ablation under N = 10. Unlike text-based ICL, the base prediction plays a more dominant role in MICL, as using it alone already achieves competitive performance, further supporting the two-stage framework. The best result is obtained by combining the base prediction with lyrics (ID 5). Adding audio descriptions slightly hurts $R _ { V } ^ { 2 }$ , suggesting that raw audio provides sufficient acoustic information, rendering audio descriptions redundant.

Finally, for 80-shot text-based ICL, replacing GPT-5 by Qwen2-7B, Qwen3.5-122B, and GPT-5-mini reduces $R _ { V } ^ { 2 }$ to 0.164, 0.285, 0.317, respectively. For 10-shot MICL, replacing Qwen3-Omni-30B by Qwen2.5-Omni-3B and Qwen2.5- Omni-7B reduces $R _ { V } ^ { 2 }$ to 0.123 and 0.308. This suggests that larger models lead to better result in general.

## 5. CONCLUSION

We proposed an ICL approach for dimensional MER, leveraging textual and multimodal demonstrations to predict offsets that refine a base model prediction. Experiments on the highquality but relatively small MER60 dataset demonstrate the effectiveness of this approach. Future work will seek ways to improve the multimodal ICL and expand the test data further. Code, model checkpoint, qualitative interpretable reasoning examples, prediction results on non-MER60 audio clips, and extension to full-song prediction are available at https:// github.com/DCN2001/MusicEmotionModeling.

## 6. REFERENCES

[1] Yi-Hsuan Yang and Homer H. Chen, “Machine recognition of music emotion: A review,” ACM Trans. Intell. Syst. Technol., vol. 3, no. 3, 2012.

[2] Jaeyong Kang and Dorien Herremans, “Are we there yet? A brief survey of music emotion prediction datasets, models, and outstanding challenges,” IEEE Trans. Affect. Comput., 2025.

[3] James A. Russell, “A circumplex model of affect,” J. Pers. Soc. Psychol., vol. 39, no. 6, pp. 1161–1178, 1980.

[4] Jonathan Posner, James A. Russell, and Bradley S. Peterson, “The circumplex model of affect: An integrative approach to affective neuroscience, cognitive development, and psychopathology,” Dev. Psychopathol., vol. 17, no. 3, pp. 715–734, 2005.

[5] Yi-Hsuan Yang et al., “A regression approach to music emotion recognition,” IEEE Trans. Audio Speech Lang. Process., vol. 16, no. 2, pp. 448–457, Feb. 2008.

[6] Xiao Hu and Yi-Hsuan Yang, “Cross-dataset and crosscultural music mood prediction: A case on Western and Chinese pop songs,” IEEE Trans. Affect. Comput., vol. 8, no. 2, pp. 228–240, 2017.

[7] Shawn Hershey et al., “CNN architectures for largescale audio classification,” in Proc. ICASSP, 2017.

[8] Jaeyong Kang and Dorien Herremans, “Towards unified music emotion recognition across dimensional and categorical models,” in Proc. ICPR, 2026.

[9] Remi Delbouys et al., “Music mood detection based on´ audio and lyrics with deep neural net,” in Proc. ISMIR, 2018.

[10] Jason Wei et al., “Chain-of-thought prompting elicits reasoning in large language models,” in Proc. NeurIPS, 2022.

[11] Paul K. Rubenstein et al., “AudioPaLM: A large language model that can speak and listen,” arXiv preprint arXiv:2306.12925, 2023.

[12] Changli Tang et al., “SALMONN: Towards generic hearing abilities for large language models,” arXiv preprint arXiv:2310.13289, 2023.

[13] Jin Xu et al., “Qwen3-Omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[14] Yi-Hsuan Yang and Homer H. Chen, “Predicting the distribution of perceived emotions of a music signal for content retrieval,” IEEE Trans. Audio Speech Lang. Process., vol. 19, no. 7, pp. 2184–2196, Sept. 2011.

[15] Norman R. Draper and Harry Smith, Applied Regression Analysis, McGraw-Hill, 1998.

[16] Tom B. Brown et al., “Language models are few-shot learners,” in Proc. NeurIPS, 2020, vol. 33.

[17] Rishabh Agarwal et al., “Many-shot in-context learning,” in Proc. NeurIPS, 2024, vol. 37, pp. 76930–76966.

[18] Xiao Hu and Yi-Hsuan Yang, “The mood of Chinese pop music: Representation and recognition,” J. Assoc. Inf. Sci. Technol., vol. 68, no. 8, pp. 1899–1910, 2017.

[19] Yunfei Chu et al., “Qwen-Audio: Advancing universal audio understanding via unified large-scale audiolanguage models,” arXiv preprint arXiv:2311.07919, 2023.

[20] Sander Koelstra et al., “DEAP: A database for emotion analysis using physiological signals,” IEEE Trans. Affect. Comput., vol. 3, no. 1, pp. 18–31, 2012.

[21] Hsiao-Tzu Hung et al., “EMOPIA: A multi-modal pop piano dataset for emotion recognition and emotionbased music generation,” in Proc. ISMIR, 2021.

[22] Yi Qiu, Yu Lin, and Yun Lin, “Music audio emotion regression using the fusion of convolutional neural networks and bidirectional long short-term memory models,” PeerJ Comput. Sci., vol. 11, pp. e3086, 2025.

[23] Anna Aljanaki et al., “Developing a benchmark for emotional analysis of music,” PLOS ONE, vol. 12, no. 3, pp. e0173392, 2017.

[24] Meng Yang et al., “Exploring the feasibility of LLMs for automated music emotion annotation,” in Proc. ISMIR, 2025.

[25] Dmitry Bogdanov et al., “Essentia: An audio analysis library for music information retrieval,” in Proc. ISMIR, 2013.

[26] Yizhi Li et al., “MERT: Acoustic music understanding model with large-scale self-supervised training,” in Proc. ICLR, 2024.

[27] Sanyuan Chen et al., “BEATs: Audio pre-training with acoustic tokenizers,” arXiv preprint arXiv:2212.09058, 2022.

[28] Alec Radford et al., “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, 2023.

[29] Alexandre Defossez, “Hybrid spectrogram and ´ waveform source separation,” arXiv preprint arXiv:2111.03600, 2021.