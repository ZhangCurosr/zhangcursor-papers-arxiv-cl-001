# Developing a Speech Language Model for Basque

Andoni Sudupe <sup>ID</sup> <sup>1,∗∗</sup>, Inma Hernaez Rioja ´ <sup>ID</sup> <sup>1</sup>, Ander Arriandiaga <sup>ID</sup> <sup>1</sup>, Christoforos Souganidis <sup>ID</sup> <sup>1</sup>, Ibon Saratxaga <sup>ID</sup> <sup>1</sup>, Eva Navas <sup>ID</sup> <sup>1</sup>

<sup>1</sup> HiTZ Basque Center for Language Technology - Aholab Signal Processing Laboratory, University of the Basque Country UPV/EHU, Spain

andoni.sudupe@ehu.eus, inma.hernaez@ehu.eus, ander.arriandiaga@ehu.eus, christoforos.souganidis@ehu.eus, ibon.saratxaga@ehu.eus, eva.navas@ehu.eus

## Abstract

Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, endto-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system’s viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages.

Index Terms: speech language model, human-computer interaction, low resource languages, synthetic approach, chatbot

## 1. Introduction

Recent developments in Large Language Models (LLMs) have paved the way for multi-modal interaction, combining textual data with other information modalities such as audio via Speech Language Models (SLMs). These systems bypass the traditional cascaded pipelines (ASR+LLM+TTS) to enable end-toend spoken interactions. Processing speech natively mitigates the inherent limitations these cascaded pipelines have, like semantic information loss, error accumulation, and high response latency. Due to these capabilities, the most advanced proprietary systems have adopted natively multi-modal architectures capable of processing and generating speech.

However, current SLMs are proprietary and primarily trained with English and Chinese resources, and therefore, minority languages face significant challenges in staying at the forefront of these advancements. While a performance gap already existed in text processing, introducing the speech modality worsens this disparity, as audio must inherently belong to the target language. Despite recent efforts in open-weight models like LLaMA-Omni [1], LLaMA-Omni2 [2] and Spirit LM [3], these systems remain almost exclusively focused on English and Chinese.

Developing an SLM requires addressing critical design decisions and data scarcity, many of which have not been properly explored for low-resource languages. This paper investigates the development, training recipes, and evaluation methods required to build an open SLM for Basque. Following LLaMA-

Omni, we use a Whisper model [4] as an encoder and a Unit HiFi-GAN [5] vocoder trained from HuBERT [6] tokens to generate the output speech. Although centered on a single language (Basque), our study likely generalizes to other similarly resourced languages that struggle to keep pace with the multimodal technological wave.

To address the severe scarcity of Basque speech instruction data, we create the first multi-modal spoken-instruction datasets for Basque using a synthetic methodology. Relying on open resources, we translated the English VoiceAssistant-400K [7] corpus into Basque using the Latxa-3.1-70B-Instruct<sup>1</sup> [8] model to ensure cultural and syntactic adaptation. Subsequently, the corresponding audio was synthesized using a Basque TTS system using different voices.

With our newly generated dataset, we can adapt the Basque instructed Latxa-3.1-8B-Instruct<sup>2</sup> [8] model for multi-modal spoken tasks. In our experiments, we find that:

• Synthetic data is sufficient to build a robust SLM for a lowresource language: By translating existing English corpora and synthesizing the audio, we successfully trained a model that achieves a score of 3.86 out of 5 for content quality, based on LLM-as-a-Judge evaluation.

• Modular architectures effectively transfer linguistic capabilities to the spoken modality: Our approach successfully bypasses the need for massive multi-modal pre-training. The resulting system achieves a normalized Word Error Rate (WER) of 9.94% for Basque words, comparable to the intelligibility accuracy achieved by the original LLaMA-Omni model in English.

The remainder of the paper is organized as follows: Section 2 details the system architecture of our modular Basque SLM. Section 3 describes the data curation and synthetic generation pipeline used to construct the multi-modal corpora. Section 4 outlines the multi-stage training methodology. Section 5 presents the evaluation framework alongside the performance results, and Section 6 summarizes our conclusions and discusses avenues for future work.

## 2. System Architecture

In this work, we follow a modular design for the SLM inspired by LLaMA-Omni [1]. This design choice is motivated by two main factors: (1) LLaMA-Omni is built upon the LLaMA-3.1- 8B-Instruct [9] backbone, for which a Basque-adapted version is already available; and (2) the use of pre-trained modules significantly reduces computational requirements during training, making it highly suitable for a low-resource language like

![](images/f0f3774574aa0e61eaf13e9301306f14a313700163ead1603e07cbed401c90a1.jpg)  
Figure 1: Left: Model architecture of our SLM, based on LLaMA-Omni. Right: Illustration of the two-stage training strategy.

Basque.

This modular approach involves attaching pre-trained modules to an already trained Large Language Model (LLM) to enable speech functionalities without requiring extensive architectural changes. This architecture is illustrated in Figure 1. First, an encoder processes the input speech into continuous representations. To enable the LLM to understand these extracted features, a trainable speech adaptor maps the speech representations directly into the LLM’s embedding space. This mapping is achieved by first downsampling the sequence to reduce the temporal length. Then, it projects the representations through a two-layer Feed-Forward Network (FFN) with a ReLU activation function to match the LLM’s embedding dimension. Once encoded, these representations are integrated into the LLM’s prompt, allowing the model to produce the corresponding response.

LLaMA-Omni distinguishes itself from other models through its unique approach to processing outputs. Traditionally, decoding responses into speech requires intermediate text generation. In this architecture, however, the LLM’s hidden states are decoded into speech and text in parallel, bypassing the dependency on intermediate text and enabling a very low response latency. The Speech Decoder takes the LLM’s hidden states (captured right before text generation) and first upsamples them to match the temporal sequence length of the audio. Then a non-autoregressive (NAR) Transformer predicts discrete tokens. These tokens correspond to pre-trained clusters of speech representations extracted by an encoder, which allows a vocoder to synthesize speech directly from them.

During inference, to enable very low latency streaming synthesis, tokens are processed in chunks. When the number of generated tokens reaches a pre-defined chunk size Ω, this segment is sent to the vocoder to synthesize a speech segment, which is then immediately played to the user.

Following this architecture, we instantiate our system with Basque-specific components. The backbone of our system is Latxa-3.1-8B-Instruct, an LLM fine-tuned specifically for Basque based on the LLaMA-3.1 architecture. Leveraging this model ensures that the system maintains high syntactic, morphological, and semantic comprehension of the Basque language during spoken interactions. To encode the input speech, we employ the Whisper-large-v3-eu [10] model, which has been adapted for Basque Automatic Speech Recognition (ASR). Finally, the discrete tokens that the Speech Decoder outputs are derived from a pre-trained mHuBERT-147 [11] model. This encoder was fine-tuned for ASR (as described in sections 3.1 and 4.1), and then grouped into 1000 clusters using K-Means. A

Unit HiFi-GAN vocoder, trained specifically on Basque speech datasets in a multi-speaker configuration, generates continuous high-fidelity audio waveforms from these discrete units.

## 3. Data

As described in the design section, the SLM model needs different modules; a Whisper encoder, a HiFi-GAN vocoder, an encoder to extract discrete tokens (which is a HuBERT model) and an LLM. The Basque variant of the Whisper model and the LLM are available so we only need to train (1) the HuBERT to extract the discrete tokens, (2) the vocoder and the (3) endto-end SLM. For this work we define 3 types of datasets; (1) a dataset with (audio, transcription) pairs for the HuBERT encoder, (2) a dataset with (discrete tokens, audio) pairs for the vocoder and (3) a dataset with (audio input, text output, discrete unit output) triplet data for the SLM.

## 3.1. HuBERT training Dataset

There are two main methods to train the HuBERT encoder for Basque: conducting a continuous pretraining using HuBERT’s Self Supervised framework or fine-tuning for a specific task. We opted for the latter approach, this decision is discussed in Section 4.1. To fine-tune a model for Automatic Speech Recognition (ASR), audio data and synchronized transcriptions are required. For this purpose, we used the composite-eu<sup>3</sup> dataset, which comprises three public datasets for Basque: Common-Voice [12], BasqueParliament [13] and OpenSLR [14]. These collections were curated and combined to yield 600 hours of transcribed Basque audios.

## 3.2. Vocoder Dataset

Originally, HiFi-GAN vocoders use Mel Spectrograms to infer speech, but some work has been done to instead use discrete tokens extracted from an encoder [5]. To train the Unit HiFi-GAN vocoder, pairs of discrete units and audio are required, and for this purpose a Basque TTS dataset<sup>4</sup> was used. It contains 2 speakers: Maider (female) and Antton (male). Each speaker has 13,500 recorded utterances, with an average duration of 4 seconds per audio, totaling around 16 hours of speech per speaker.

## 3.3. SLM Dataset

Training a Speech Language Model requires data in a specific format. This training requires triplet data (audio input, text output, discrete unit output) structured in a speech-like or spoken-dialogue structure. This format is highly uncommon and no such public dataset is available for Basque.

To address this scarcity of speech instruction data in the Basque language, we built a synthetic workflow inspired by the construction of multi-modal datasets like VoiceAssistant-400K [7] or InstructS2S-200K [1].

First, we extracted high-quality text instruction sets by translating the VoiceAssistant-400K dataset, which already follows the speech-like structure. For translation, we used the Latxa-3.1-70B-Instruct [8] model prompted with a strict rubric. This method removes the literal phrasing and parsing errors that conventional Machine Translation (MT) models often introduce.

Once the translation is completed, we used a VITS (Variational Inference with adversarial learning for end-to-end Textto-Speech) [15] system to synthesise the utterances. The questions were synthesized using 14 synthetic voices distributed evenly by gender to ensure speaker diversity [16]. For the target responses, we used the same speaker used to train the vocoder.

Finally, for the second stage of the training, we extracted the discrete tokens from the target response audios. For the extraction we used the finetuned HuBERT for Basque (see section 4.1). To convert the response speech into discrete units, we first passed them through the fine-tuned HuBERT and selected the 9th hidden layer [5]. Then we applied a K-Means clustering algorithm optimized for k = 1000 centroids [5]. The same Hu-BERT encoder and K-Means model must be used for the SLM and the vocoder, to avoid a mismatch in discrete tokens.

With this pipeline we successfully built a Basque spoken instruction dataset with approximately 450,000 questions with the corresponding answers, yielding around 550 hours of questions and over 3300 hours of answers, setting a viable baseline for end-to-end multi-modal alignment.

## 4. Training

To develop our Basque Speech Language Model (SLM), the overall process first involves adapting the individual components; fine-tuning the speech encoder to extract discrete tokens and training the vocoder to synthesize acoustic waveforms. Once these components are prepared, the complete SLM undergoes a two-stage training process. The following subsections detail the specific training procedures for the encoder, the vocoder, and the end-to-end SLM architecture.

## 4.1. HuBERT Training

To adapt the encoder for extracting Basque discrete tokens, we fine-tuned the model for an Automatic Speech Recognition (ASR) task. While continuous pre-training with new language data is typically the recommended approach, it requires a massive amount of data [6] that is currently unavailable for Basque. However, there is sufficient transcribed data to fine-tune the model for ASR.

As a starting point we selected the mHuBERT-147 [11] checkpoint. This model was trained with 147 languages, including Basque, and has demonstrated better performance than much larger models across various tasks.

The final encoder was trained using this configuration for 100 epochs with a batch size of 64. The training was conducted on a single NVIDIA H100 80GB GPU using the Hugging Face Transformers [17] library, taking approximately 400 hours to finish.

## 4.2. Vocoder Training

The selected vocoder, the Unit HiFi-GAN model, synthesizes waveforms directly from discrete representations. The training was implemented using the SpeechBrain [18] framework. Based on early experimentation comparing single-speaker and multi-speaker configurations, we adopted a multi-speaker strategy, which demonstrated superior acoustic quality and stability. To condition the vocoder on different voices, speaker embeddings were extracted using a pre-trained ECAPA-TDNN [19] model<sup>5</sup>.

The final vocoder was trained on the two target voices (Maider and Antton), as previous tests showed a 2-speaker setup effectively captures necessary paralinguistic features without the complexity of broader configurations. It was trained for 1000 epochs with a batch size of 128. This process was executed on a single NVIDIA H100 80GB GPU and required roughly 10 hours to complete.

## 4.3. SLM Training

As shown in Figure 1, the training of the SLM is conducted in two distinct stages: (1) Speech Understanding and Text Generation and (2) Speech Response Generation.

Stage 1 In the first stage, the primary objective is to teach the LLM the ability to understand spoken instructions and generate coherent text responses. During this phase, the speech encoder is kept completely frozen; only the speech adaptor and the LLM are trained. We train the LLM to output speech-like responses, prioritizing short and concise responses. The training is performed using a standard cross-entropy loss for next-token prediction.

Stage 2 Once the model can understand speech and generate text, the second stage focuses on generating the spoken response in parallel. In this phase, the speech encoder, the speech adaptor, and the LLM are all frozen, and only the nonautoregressive streaming speech decoder is trained. Taking the hidden states of the LLM as input, the decoder learns to predict the sequence of discrete speech units. This alignment is learned asynchronously using a Connectionist Temporal Classification (CTC) loss, which manages the variable sequence lengths without requiring explicit text-to-speech alignment in the training data.

For training we adapted the original codebase for LLaMA-Omni by implementing the missing training code and the dataset generation scripts. All the material and code used for this project is publicly available at HuggingFace and Github, respectively.

For both training stages we followed the recommended hyperparameters by LLaMA-Omni. Each stage was executed for 3 epochs using a configuration of 4 NVIDIA A100 80GB GPUs, taking less than 100 hours in total.

## 5. Evaluation and Results

To evaluate the performance of our architecture, we divided the evaluation into two main blocks: the individual components (the HuBERT discrete unit encoder and the Unit HiFi-GAN vocoder), and the complete end-to-end dialogue system.

<table><tr><td>Model</td><td>LLM Score (1-5) ↑</td><td>WER (%)↓</td><td>UTMOS↑</td><td>Streaming Latency (ms) ↓</td></tr><tr><td>SpeechGPT (Eng) SALMONN + Orca (Eng) Qwen2-Audio + Orca (Eng)</td><td>2.98 3.44 3.47</td><td>45.00 3.78 6.77</td><td>3.89 3.82 3.62</td><td>5587</td></tr><tr><td>LLaMA-Omni (Eng) LLaMA-Omni2-7B (Eng)</td><td>3.99 4.28</td><td>10.82 3.26</td><td>3.92 4.22</td><td>346 582</td></tr><tr><td>Ours (Eus)</td><td>3.86</td><td>9.94</td><td>3.07</td><td>370</td></tr></table>

Table 1: Comparison ofour Basque (Eus) model with English (Eng) SLMs. English values are as reported in [1] and [2], using different judges/ASR systems; WER on synthesized speech also depends on TTS and ASR quality, so cross-language figures are indicative, not strictly comparable.

## 5.1. Modules

Discrete Unit Encoder (HuBERT): Even though the HuBERT model is used to extract discrete units, it was fine-tuned on an Automatic Speech Recognition (ASR) task. To evaluate its learned representations we can use the standard metrics for ASR, Word Error Rate (WER) and Character Error Rate (CER). After training the model for 100 epochs, it achieved a 4.90% WER on the test split of the composite-eu dataset. Overall the encoder representations capture the phonetic properties of Basque effectively, approaching the State-of-the-Art (SoTA) Basque ASR baselines.

Vocoder (Unit HiFi-GAN): The vocoder synthesizes acoustic waveforms from discrete tokens. Its performance was measured using Mel Cepstral Distortion (MCD), where lower values (typically under 5-6 dB) indicate high synthesis quality. After training the model for 1000 epochs with 2 speakers, the Antton voice achieved a MCD score of 4.8 dB, and the Maider voice a score of 5.1 dB.

## 5.2. End-to-End SLM

We evaluated the full SLM pipeline using three metrics; the content quality of the text, the audio quality and the intelligibility of the speech response.

For content quality, we used an LLM-as-a-judge approach, prompting the Latxa-3.1-70B-Instruct model with a strict rubric to rate responses on a scale from 1 to 5 (LLM-as-a-judge Score) based on helpfulness, relevance, factual accuracy, and Basque fluency. This judge shares its family with our 8B backbone and 70B translation model (a possible self-preference bias); lacking a second judge or human annotators for Basque, we report this as a limitation. For audio quality, we used the UTMOS [20] metric, which complements the vocoder’s MCD reconstruction quality. This predictor was trained mainly on English and Japanese speech, so Basque scores are indicative rather than absolute. To assess speech intelligibility, we transcribed the generated audio using the state-of-the-art Basque Conformer ASR model<sup>6</sup> and calculated the WER and CER exclusively on Basque. All end-to-end scores are computed over a set of 4,419 synthetic instructions, disjoint from SLM training data; as both share the same synthetic pipeline and no real Basque speech was used at test time, this evaluation is in-distribution, and results on real user speech remain to be tested.

Our model achieves an average LLM-as-a-judge score of 3.86 out of 5. Regarding the spoken response evaluations, the model achieves a 3.07 UTMOS score, and an average WER of 9.94% and a CER of 1.62%.

When placed in the context of leading English multimodal models (see Table 1), our model performs competitively. Despite Basque being a low-resource language, its LLMas-a-judge score of 3.86 places closely behind LLaMA-Omni (3.99) and outperforms Qwen2-Audio (3.47) and SALMONN (3.44). Our model’s ASR-WER of 9.94% is comparable to LLaMA-Omni’s English WER (10.82%), with a remaining gap to LLaMA-Omni2-7B (3.26%). The largest gap is in response speech quality, where our 3.07 UTMOS falls behind all English baselines. The streaming latency of 370 ms, measured on an NVIDIA A100 80GB GPU with chunk size Ω = 40 (matching LLaMA-Omni’s setting), is close to LLaMA-Omni (346 ms) and far lower than cascaded pipelines like SpeechGPT (5587 ms).

## 6. Conclusions

In this work, we developed the first open-source end-to-end Spoken Language Model (SLM) for Basque, establishing a reproducible framework to bridge the multi-modal gap for lowresource languages without massive foundational pre-training.

Our findings show that in the absence of spoken-instruction data, synthesizing data via an automated workflow is highly effective. Translating high-quality English corpora using LLMs combined with multi-speaker TTS synthesis provides a robust, scalable baseline for multi-modal alignment. The resulting model achieves a high content quality score of 3.86 out of 5, proving highly competitive with established English-centric systems.

Furthermore, our results show that attaching specialized speech adaptors and parallel streaming decoders to a languagespecific text backbone successfully transfers native linguistic capabilities to the spoken modality. Architecturally, the model achieves excellent speech intelligibility with a Word Error Rate (WER) of 9.94% for Basque words, comparable to that reported for the original English LLaMA-Omni across different languages and ASR systems, while maintaining a remarkably low streaming latency of 370 ms.

Future work will focus on integrating more expressive TTS systems to close the acoustic quality gap, complementing our automatic evaluation with a second judge model and a smallscale human evaluation, and testing the system on real, nonsynthetic Basque speech.

## 7. Acknowledgments

This work has been partially funded by the projects 2022/TL22/00215335 ILENIA, and ’Desarrollo de modelos ALIA’, funded by the Ministerio de Transformacion Digital and´ by the Plan de Recuperacion, Transformaci´ on y Resiliencia –´ <sup>6</sup>Link to the state-of-the-art Conformer model

Funded by the European Union – NextGenerationEU and by the project IkerGaitu II funded by the Basque Government. These models were trained at Hyperion, hosted by the DIPC Supercomputing Center, and also at the Basque Government (EJIE) Data Processing Center.

## 8. References

[1] Q. Fang, S. Guo, Y. Zhou, Z. Ma, S. Zhang, and Y. Feng, “LLaMA-Omni: Seamless Speech Interaction with Large Language Models,” 2025. [Online]. Available: https://arxiv.org/ abs/2409.06666

[2] Q. Fang, Y. Zhou, S. Guo, S. Zhang, and Y. Feng, “LLaMA-Omni2: LLM-based Real-time Spoken Chatbot with Autoregressive Streaming Speech Synthesis,” 2025. [Online]. Available: https://arxiv.org/abs/2505.02625

[3] T. A. Nguyen, B. Muller, B. Yu, M. R. Costa-jussa, M. Elbayad, S. Popuri, C. Ropers, P.-A. Duquenne, R. Algayres, R. Mavlyutov, I. Gat, M. Williamson, G. Synnaeve, J. Pino, B. Sagot, and E. Dupoux, “Spirit LM: Interleaved Spoken and Written Language Model,” 2024. [Online]. Available: https://arxiv.org/abs/2402.05755

[4] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust Speech Recognition via Large-Scale Weak Supervision,” 2022. [Online]. Available: https: //arxiv.org/abs/2212.04356

[5] A. Polyak, Y. Adi, J. Copet, E. Kharitonov, K. Lakhotia, W.-N. Hsu, A. Mohamed, and E. Dupoux, “Speech Resynthesis from Discrete Disentangled Self-Supervised Representations,” 2021. [Online]. Available: https://arxiv.org/abs/2104.00355

[6] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units,” 2021. [Online]. Available: https://arxiv.org/abs/2106.07447

[7] Z. Xie and C. Wu, “Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming,” 2024. [Online]. Available: https://arxiv.org/abs/2408.16725

[8] O. Sainz, N. Perez, J. Etxaniz, J. F. de Landa, I. Aldabe, I. Garc´ıa-Ferrero, A. Zabala, E. Azurmendi, G. Rigau, E. Agirre, M. Artetxe, and A. Soroa, “Instructing Large Language Models for Low-Resource Languages: A Systematic Study for Basque,” 2025. [Online]. Available: https://arxiv.org/abs/2506.07597

[9] A. Dubey, A. Jauhri, A. Pandey, A. Kadian, and A. Al-Dahle, “The Llama 3 Herd of Models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

[10] X. de Zuazo, E. Navas, I. Saratxaga, and I. H. Rioja, “Whisper-LM: Improving ASR Models with Language Models for Low-Resource Languages,” 2025. [Online]. Available: https://arxiv.org/abs/2503.23542

[11] M. Z. Boito, V. Iyer, N. Lagos, L. Besacier, and I. Calapodescu, “mHuBERT-147: A Compact Multilingual HuBERT Model,” 2024. [Online]. Available: https://arxiv.org/abs/2406.06371

[12] R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, “Common Voice: A Massively-Multilingual Speech Corpus,” 2020. [Online]. Available: https://arxiv.org/abs/1912.06670

[13] A. Varona, M. Penagarikano, G. Bordel, and L. J. Rodriguez-Fuentes, “A Bilingual Basque–Spanish Dataset of Parliamentary Sessions for the Development and Evaluation of Speech Technology,” Applied Sciences, vol. 14, no. 5, 2024. [Online]. Available: https://www.mdpi.com/2076-3417/14/5/1951

[14] O. Kjartansson, A. Gutkin, A. Butryna, I. Demirsahin, and C. Rivera, “Open-Source High Quality Speech Datasets for Basque, Catalan and Galician,” in Proceedings of the 1st Joint Workshop on Spoken Language Technologiesfor Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL). Marseille, France: European

Language Resources association (ELRA), May 2020, pp. 21– 27. [Online]. Available: https://www.aclweb.org/anthology/2020. sltu-1.3

[15] J. Kim, J. Kong, and J. Son, “Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech,” 2021. [Online]. Available: https://arxiv.org/abs/2106.06103

[16] I. Hierro Muga, I. Saratxaga Couceiro, I. Hernaez Rioja,´ A. Arriandiaga Laresgoiti, and V. Garcia Romillo, “Neuronasareetan onarritutako sintesi-ahots pertsonalizatuen hobekuntzen garapena eta ebaluazioa,” IkerGazte. Nazioarteko ikerketa euskaraz, vol. 3, p. 235–242, mai. 2025. [Online]. Available: https://aldizkariak.ueu.eus/index.php/ikergazte/article/view/5998

[17] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “HuggingFace’s Transformers: State-of-theart Natural Language Processing,” 2020. [Online]. Available: https://arxiv.org/abs/1910.03771

[18] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong, J.-C. Chou, S.-L. Yeh, S.-W. Fu, C.-F. Liao, E. Rastorgueva, F. Grondin, W. Aris, H. Na, Y. Gao, R. D. Mori, and Y. Bengio, “SpeechBrain: A General-Purpose Speech Toolkit,” 2021. [Online]. Available: https://arxiv.org/abs/2106.04624

[19] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification,” in Interspeech 2020, ser. interspeech2020. ISCA, Oct. 2020, p. 3830–3834. [Online]. Available: http://dx.doi.org/10.21437/Interspeech.2020-2650

[20] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “UTMOS: UTokyo-SaruLab System for Voice-MOS Challenge 2022,” in Interspeech 2022, 2022, pp. 4521– 4525.