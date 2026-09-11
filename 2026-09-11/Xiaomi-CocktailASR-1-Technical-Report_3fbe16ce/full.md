# Xiaomi-CocktailASR-1 Technical Report

Xiaoai Plus, ASR team

Xiaomi Inc., China

## ABSTRACT

Recently, large language model (LLM) based ASR models have achieved significant progress, yet they generally lack support for multi-speaker scenarios, where the cocktail party problem remains a critical bottleneck for further advancing ASR. Existing TS-ASR methods, including end-to-end architectures with speaker embeddings and latest LLM-based explorations suffer from degraded single-speaker performance and the inability to reject when the target speaker is absent. In this paper, we propose Xiaomi-CocktailASR-1, an LLM-based end-to-end TS-ASR architecture. By utilizing reference speech as voiceprint prompts, it directly transcribes the target speaker’s speech without requiring speech separation. Xiaomi-CocktailASR-1 maintains competitive performance in single-speaker scenarios, comparable to mainstream ASR models. It also features a negative sample rejection capability, outputting empty text when the target speaker is absent from the mixed speech. Additionally, Xiaomi-CocktailASR-1 supports a Chain-of-Thought (CoT) reasoning mode to provide explicit reasoning steps. Extensive experiments on various synthetic and real-world multispeaker benchmarks demonstrate that Xiaomi-CocktailASR-1 achieves state-of-the-art performance, effectively addressing the cocktail party problem through a unified architecture that balances multispeaker and single-speaker recognition accuracy, along with rejection capability.

## 1 Introduction

Advances in LLMs have driven mainstream ASR toward Large Audio-Language Models (LALMs) trained on massive datasets with billions of parameters, such as Qwen3-ASR [1], StepAudio [2], and Seed-ASR [3]. By fully leveraging the strong modeling capabilities of LLMs, this paradigm integrate various ASR functions into a unified end-to-end architecture, achieving robust performance across multiple languages, dialects, timestamp prediction, and contextual understanding. However, existing models struggle with complex multi-speaker scenarios, and no open-source LALMs for TS-ASR have been specifically designed under such cocktail-party settings.

The cocktail party problem is a classic challenge in speech processing, defined as the difficulty of selectively focusing a specific speaker in complex multi-speaker scenarios. Overcoming this problem is essential for the advancement of ASR technology in real-world acoustic conditions. Current mainstream solutions for this scenario include multi-talker ASR (MT-ASR) and TS-ASR. MT-ASR systems [4, 5, 6] transcribe all speakers sequentially, typically following a “first-in, first-out” strategy, with the outputs of different speakers separated by special tokens. However, they cannot associate transcriptions with specific speaker identities, which limits their usefulness when targeting a specific speaker. In contrast, TS-ASR [7] leverages reference information from the target speaker to selectively transcribe the target speech while suppressing interference from others, making it more suitable for practical applications.

Traditional TS-ASR research primarily employs cascaded systems combining front-end Target Speaker Extraction (TSE) and back-end ASR, inevitably introducing system complexity and error accumulation [8, 9]. Later, some studies explored end-to-end architectures by integrating speaker embeddings with ASR models for joint optimization [10, 11, 12], but the effectiveness of such approaches is still constrained by independent speaker encoders. Recent studies have incorporated LLMs into TS-ASR to leverage their strong semantic capabilities [13, 14], yet these approaches still face significant limitations in complex, real-world scenarios.

![](images/3b11fd09a4d4ffa73d3459d0e9f52206159f43ac731b19b85d818a8df489c083.jpg)  
Figure 1: The introduction of Xiaomi-CocktailASR-1 capabilities.

In real-world interaction scenarios such as smart homes, wearable devices, and intelligent meetings, systems need to not only suppress multi-speaker interference but also dynamically adapt to complex acoustic environments. However, existing TS-ASR systems still face the following limitations in practical applications:

• Performance degradation in single-speaker scenarios: In personal or smart home scenarios, the target user often speaks alone. Existing models designed to suppress multi-speaker interference tend to over-suppress in single-speaker scenarios, which mistakenly suppresses the target speech and increases deletion errors. In practical applications, since the number of speakers cannot be predicted in advance, a system designed for either single-speaker ASR or multi-speaker TS-ASR will inevitably underperform in the other scenario. Thus, an ideal model must seamlessly handle both single-speaker and multi-speaker recognition tasks.

• Lack of rejection capability during target absence: In outdoor scenarios or meetings, the target speaker may be temporarily absent or silent, which is defined as a negative sample in this paper and requires the model to output empty text. However, existing models cannot determine whether the target speaker is present, often transcribing irrelevant speech when the target is absent, which leads to severe false triggers and degrades the user experience.

• Missing performance and interpretability gains from reasoning: As a task designed for complex scenarios, TS-ASR is naturally suited for CoT reasoning. Recent studies have shown that CoT reasoning brings performance gains across various speech tasks, and TCP [15] specifically demonstrates its effectiveness in TS-ASR. Moreover, the reasoning generated by CoT enhances interpretability and provides valuable information for downstream tasks.

To address these limitations, we propose Xiaomi-CocktailASR-1, an end-to-end TS-ASR system based on LLMs. By employing reference speech as voiceprint prompt, the system directly transcribes target speech without relying on explicit speech separation modules. As shown in Figure 1, the core contributions of Xiaomi-CocktailASR-1 lie in significantly improving both recognition accuracy and system reliability in complex acoustic scenarios. Xiaomi-CocktailASR-1 operates in two modes. In the standard mode, a unified prompt enables the model to handle single-speaker recognition, multi-speaker target-speaker ASR, and negative sample rejection within a single architecture. In the CoT mode, the model uses a separate prompt to generate explicit intermediate reasoning steps. Specifically, the main contributions are as follows:

![](images/edfef2a187866e84ce6da4541e35785ff618c7bff19db2bdfb031b8abfe28af4.jpg)  
Figure 2: The overview of Xiaomi-CocktailASR-1 framework.

• First, it achieves accuracy comparable to mainstream single-speaker ASR models in single-speaker scenarios, eliminating the need to switch models for inputs with different numbers of speakers.

• Second, Xiaomi-CocktailASR-1 incorporates a negative sample rejection mechanism that reliably outputs empty text when the target speaker is absent from the input mixture, effectively reducing false triggers in real-world interactions.

• Third, Xiaomi-CocktailASR-1 supports CoT reasoning, which generates explicit intermediate logical steps to provide high interpretability for the recognition process.

• Finally, the model achieves state-of-the-art performance on multiple simulated and real-world multi-speaker benchmarks.

## 2 Architecture

As illustrated in Figure 2, the overall architecture of Xiaomi-CocktailASR-1 consists of Audio Encoder, Adapter, and an LLM backbone network. The concatenation of reference speech and mixed speech is first fed into the Audio Encoder, which converts it into frame-level embeddings capturing both speech and speaker information, then passed through the Adapter and LLM to generate the target speaker’s transcription. The model details are described below.

## 2.1 Input Organization

The model input is formed by sequentially concatenating the reference speech, a one-second silence segment, and the mixed speech. For the reference speech, it is randomly sampled with durations from one to four seconds, providing sufficient information for voiceprint extraction without significant computational cost. As for the silence segment, it is inserted to explicitly distinguish the reference speech from the mixed signal and anchor the voiceprint features.

## 2.2 Model Structure

As illustrated on Figure 2, the concatenated audio is first fed into an Audio Encoder with 0.6B parameters. Derived from the self-supervised learning (SSL) model Data2Vec2 [16], this encoder converts speech signals into 1280-dimensional frame-level embeddings. By leveraging the mask prediction mechanism inherent in SSL [17], it naturally fuses semantic and speaker information within a single network, thereby eliminating the need for a separate speaker encoder. Regarding structural improvements, we incorporate a FBank processing module at the front end of the encoder to replace raw waveform inputs. This design significantly accelerates both training and inference speeds while maintaining model performance. Consequently, when processing multi-speaker mixed audio, the encoder treats the reference audio as a conditional prompt, adaptively focusing on and enhancing the target speaker’s speech representations during feature extraction to effectively suppress interference from irrelevant speakers.

The frame-level embeddings output by the Audio Encoder are subsequently processed by an Adapter module. Employing a lightweight linear network structure, this module performs cross-modal alignment by mapping the embeddings into the hidden feature space of the LLM. Simultaneously, a text tokenizer transforms the textual prompt into text embeddings. These aligned audio features and text embeddings are then concatenated and jointly fed into an LLM based on the Qwen3-8B [18] base model. Leveraging the powerful long-sequence modeling and context understanding capabilities of the LLM, the model interprets the joint audio-text input, focuses on the target speaker within mixed speech, and generates high-quality transcriptions of the target speaker.

## 3 Training

To enable Xiaomi-CocktailASR-1 to handle the classic cocktail party scenario as well as various practical situations encountered in real-world applications, the model requires four core capabilities: accurate multi-speaker target speech recognition, single-speaker recognition comparable to standard ASR models, negative sample rejection during target absence, and interpretable CoT reasoning. To achieve these four objectives, we construct specific training datasets and design a multi-stage pipeline, guiding the model to progressively acquire and balance these capabilities.

## 3.1 Mechanism

## 3.1.1 Multi-speaker Recognition

Multi-speaker recognition is the core capability of a TS-ASR system. It requires the model to accurately transcribe the target speech from overlapping multi-speaker speech while effectively suppressing interference from irrelevant speakers. To achieve this, the model is fine-tuned on large-scale real and synthetic multi-speaker mixtures. Furthermore, to enhance robustness in complex acoustic environments, we propose a speech noise mixing strategy during data augmentation. By adding environmental noise and background interference with a certain probability, this approach significantly improves the diversity and randomness of the training data. Specifically, by probabilistically introducing environmental noise and human speech noise, this strategy generates audio with random overlaps, effectively ensuring the randomness and diversity of overlapping speech in the training data.

## 3.1.2 Single-speaker Recognition

Although TS-ASR systems are designed for multi-speaker scenarios, the number of speakers in practical applications is unpredictable, and single-speaker cases represent the majority. For example, the overlap rate in the real-world AliMeeting dataset is merely 30% to 40% [19]. Therefore, a robust TS-ASR system must not only accurately extract the target speaker from multi-speaker mixtures but also achieve single-speaker recognition performance comparable to standard single-speaker ASR models. However, the fundamental challenge lies in the inherent trade-off between these capabilities, as enhancing the model’s generalization to handle single-speaker scenarios often compromises its specialized performance in suppressing complex multi-speaker interference.

To address this, single-speaker speech data is included in the training set, where reference and target speech pairs are constructed using different utterances from the same speaker. By optimizing the sampling ratio of single-speaker and multi-speaker training data, the model effectively mitigates over-suppression, achieving an optimal trade-off between suppressing multi-speaker interference and preserving single-speaker speech integrity.

## 3.1.3 Negative Sample Rejection

Negative sample rejection is defined as the capability to actively reject recognition and output empty text when the target speaker is absent, leaving only interfering speech in the input, thereby effectively preventing the mis-transcription of irrelevant content. The main challenge of negative sample training is that allowing empty outputs may compromise the model’s recognition of normal speech. Specifically, the model may incorrectly generate empty text even when the target speaker is actively speaking.

To address this, a negative sample training strategy is proposed, where inputs are concatenated with mismatched reference speech. By progressively increasing the proportion of such negative samples across training stages, the model learns to balance accurate rejection with correct recognition. Moreover, this rejection capability is directly internalized into the model weights through end-to-end supervised learning, requiring no additional rejection thresholds during inference.

![](images/eec0243eb753520b4b5a0a5fa1ee1dd8df1dfe9150fd72b6771e3779bf09afb6.jpg)  
Figure 3: Sample of CoT data for single-speaker, 2-speaker mixed speech and negative data. Text in different colors represents different types of information.

## 3.1.4 Chain-of-Thought

The CoT mode equips the model with the ability to explicitly output intermediate reasoning within thinking tags, followed by the final transcription within answer tags. Specifically, the reasoning includes the number of speakers, as well as each speaker’s gender and voiceprint similarity to the reference speech. This paradigm is realized by training the model on data containing CoT reasoning. This explicit reasoning process guides the model to focus on key voiceprint features, effectively reducing recognition errors in complex environments and ultimately improving both accuracy and interpretability.

## 3.2 Training Data

Multi-speaker TS-ASR Data This dataset comprises three main components, categorized by data source and generation method: (1) open-source multi-speaker data, including real-world overlapping recordings (AMI [20], AliMeeting [19]) and synthetic mixtures (LibriMix [21]); (2) simulated overlapping multi-speaker data, generated by mixing our proprietary single-speaker ASR data; (3) real-world daily conversational data, collected from both in-house recordings. The total scale of the dataset is approximately 400,000 hours.

Single-speaker TS-ASR Data In this dataset, both the reference speech and the target speech are derived from different utterances of the same speaker. This setup aims to prevent performance degradation in single-speaker scenarios. The total scale of this dataset is approximately 600,000 hours.

Negative Sample Data This dataset consists of samples where the reference speaker is absent from the target speech, specifically aiming to train the model’s negative sample rejection capability. The total scale is approximately 10,000 hours, with its sampling ratio dynamically adjusted across different training stages.

CoT Data As illustrated in the Figure 3, CoT reasoning are constructed using templates covering multi-speaker positive samples, single-speaker positive samples, and negative samples, each with different formats. Template information includes speaker count, gender of each speaker, and similarity levels of each speaker to the reference audio. Based on this information, the model comprehensively compares gender and speaker similarity to determine whether the similarity falls below a threshold of 3, thereby inferring the identity of the target speaker. Specifically, the speaker with the highest similarity is identified as the target speaker; if all similarities are below the threshold, it indicates that the target speaker is absent, and empty text should be output. The speaker with the highest similarity is identified as the target, while if all similarities fall below the threshold, the target speaker is considered absent and empty text is output.

![](images/1e40cee4ea6d540aee9d96e83a7a1cd55f68d3a46b201c5cb5e4b03fb776d909.jpg)  
Figure 4: The training pipeline of Xiaomi-CocktailASR-1.

Additionally, similarity levels are computed using the CAM++ [22] model to extract speaker embeddings and calculate cosine similarity scores between each source speech and the reference speech embeddings. The continuous similarity scores ranging from 0 to 1 are then mapped to five discrete levels from 1 to 5 through uniform quantization. This discretization prevents the model from focusing on insignificant numerical differences, thereby improving training stability and convergence efficiency.

## 3.3 Training Pipeline

Stage 1 ASR Base Model Training This stage integrates the pretrained Speech Encoder and the LLM base model through an Adapter to achieve cross-modal alignment, enabling the text-focused LLM to adapt to speech recognition tasks. The model is trained exclusively on standard ASR prompts and regular single-speaker ASR data without reference speech. This establishes a standard ASR Base model, equipping it with basic speech-to-text capabilities.

Stage 2 TS-ASR Base Model Training Building upon the ASR base model, this stage performs supervised finetuning using TS-ASR data and specialized TS-ASR prompts. This data is composed of single-speaker and multi-speaker signals, along with the corresponding target speaker reference speech. This training process enables the model to extract voiceprint features and locate the target speaker, establishing initial target speaker recognition capabilities. During this stage, we apply the aforementioned noise mixing strategy with a 50% probability to directly convert half of the existing data into multi-speaker overlapping speech. Furthermore, a small proportion of negative samples is introduced to equip the model with preliminary rejection capabilities without compromising core recognition performance.

Stage 3 TS-ASR CoT Training This stage incorporates CoT data to introduce reasoning capabilities. Specifically, the CoT prompts and data are mixed equally with the original TS-ASR data for joint training. Through this joint training process, the model learns to generate explicit intermediate reasoning within thinking tags when using the CoT prompt, while directly outputting the transcription under the standard prompt. Such a design not only provides interpretable reasoning but also maintains excellent performance in the standard mode, allowing users to flexibly enable or disable CoT reasoning based on practical needs.

Stage 4 TS-ASR Model Fine-tuning Finally, the model is optimized through both supervised fine-tuning and reinforcement learning (RL) [23] on carefully selected high-quality data to further enhance its comprehensive performance in complex scenarios. During this stage, the mixing ratios of different training data categories is controlled to ensure a balanced optimization of core capabilities, including multi-speaker recognition, single-speaker generalization, and negative sample rejection.

## 4 Evaluation

## 4.1 Benchmarks

We adopt three categories of evaluation sets to comprehensively assess the model’s recognition performance in both single-speaker and multi-speaker scenarios, along with additional negative samples to evaluate its rejection capability.

## 4.1.1 Multi-Talker Datasets

Synthetic Datasets For synthetic datasets, every speaker in a mixed utterance is alternately treated as the target speaker during evaluation, ensuring objectivity and fairness of the test results.

• LibriMix: This is a standard multi-speaker mixture constructed from the LibriSpeech (test-clean) corpus [24], with random signal-to-noise ratios (SNRs) ranging from 0 to 15 dB. This random energy distribution ensures that the model cannot rely on shortcuts such as simply transcribing the loudest speaker, thereby truly evaluating its core ability to distinguish speakers based on voiceprint features.

• LibriSpeechMix: In this dataset, the starting times of each speaker are randomized to eliminate order-based cues. This ensures that the model cannot rely on sequential shortcuts, such as always transcribing the first speaker, further proving its ability to focus on acoustic voiceprint features.

Real-world Datasets : These datasets mainly consist of far-field multi-speaker recordings with complex spontaneous overlapping speech. Based on official timestamp annotations, the continuous speech is segmented into independent utterances. This ensures that each evaluation segment contains a complete target speaker utterance, while the number of interfering speakers and acoustic conditions stay unpredictable.

• AMI-SDM [20]: A classic real-world English multi-speaker meeting corpus recorded with a single microphone. It contains extensive overlapping speech, heavy far-field reverberation, and realistic background noise, making it a strong benchmark for evaluating the model’s robustness under challenging acoustic conditions.

• AliMeeting-Far [19]: A real-world Chinese multi-speaker meeting corpus recorded using a far-field 8-channel microphone array (Channel 0 is used for evaluation). This dataset is primarily used to evaluate the model’s recognition capability in complex Chinese scenarios.

## 4.1.2 Singer Speaker Datasets

Single-speaker datasets are primarily used to evaluate whether the model can maintain high recognition accuracy while avoiding deletion errors caused by over-suppression.

• LibriSpeech [24]: A standard test set for English single-speaker speech recognition, mainly used to evaluate the model’s baseline performance in clean environments.

• AMI-IHM [20]: Individual headset microphone recordings derived from the AMI meeting corpus.

• AliMeeting-Near [19]: Near-field microphone recordings from the AliMeeting corpus, focusing on realworld Chinese meeting scenarios to evaluate the model’s generalization ability under Chinese single-speaker conditions.

## 4.1.3 Negative Datasets

Negative test sets are specifically designed to evaluate the model’s rejection capability when the input contains only interfering speakers. These evaluations verify whether the model can correctly output empty text instead of forcefully transcribing irrelevant speech or generating hallucinations.

• LibriSpeech Neg: This test set uses reference speech and input utterances sourced from LibriSpeech test-clean, with the reference randomly selected from speakers not present in the input.

• Aishell Neg [25]: These test sets focus on negative sample evaluation in Chinese scenarios, primarily testing the model’s rejection robustness in cross-lingual environments.

• Chinese in-house Neg: We collected negative samples from real-world scenarios to evaluate model rejection performance in practical applications.

## 4.2 Evaluation Metrics

• Target Speaker Word / Character Error Rate (TS-WER / TS-CER) ): These metrics measure the error rates specifically for the target speaker, with the reference text strictly matching the ground truth transcription. The TS-WER and TS-CER are evaluated on the English and Chinese test sets, respectively.

• Rejection Rate (RR): This metric is computed as the proportion of correctly rejected samples among all negative test samples, reflecting the model’s ability to reject non-target speakers. A higher rate indicates that the model can reliably detect the absence of the target speaker and produce an empty output, thus reducing the risk of false triggers.

• False Rejection Rate (FRR) : This metric is computed as the proportion of incorrectly rejected samples among all positive test samples, measuring the model’s failure rate when the target speaker is actually speaking. A lower value is crucial for practical system deployment, as it indicates minimal degradation to standard recognition performance.

• Non-empty WER: Since false rejections can cause significant fluctuations in the overall WER, we recalculate this metric by excluding samples with empty outputs. This adjustment eliminates the interference of rejections on WER statistics, providing a pure evaluation of transcription accuracy for non-empty outputs.

## 4.3 Results

## 4.3.1 Target Speaker ASR in multi-talker senorial

This section presents a systematic evaluation of the model’s comprehensive performance on multi-speaker test sets, which encompass both synthetic datasets and real-world datasets. Overall, as demonstrated in Table 1 and 2, Xiaomi-CocktailASR-1 achieves the SOTA performance across the evaluated benchmarks.

Table 1: Performance comparison on different datasets. Evaluated by WER(%).
<table><tr><td>Model</td><td>LibriMix 2mix</td><td>LibriMix 3mix</td><td>LibriSpeechMix 2mix</td><td>LibriSpeechMix 3mix</td></tr><tr><td>Xiaomi-CocktailASR-1</td><td>4.11</td><td>12.29</td><td>2.90</td><td>4.91</td></tr><tr><td>Qwen3-ASR-1.7b [1]</td><td>68.75</td><td>106.04</td><td>92.17</td><td>160.82</td></tr><tr><td>StepAudio2 [2]</td><td>71.23</td><td>121.08</td><td>92.72</td><td>164.66</td></tr><tr><td>Gemini-2.5-pro¹ [26]</td><td>48.41</td><td>76.10</td><td>30.69</td><td>52.34</td></tr><tr><td>TCP [15]</td><td>4.84</td><td>12.23</td><td></td><td></td></tr><tr><td>CONF-TSASR [8]</td><td></td><td></td><td>5.40</td><td>7.60</td></tr><tr><td>MT-LLM [13]</td><td>6.70</td><td>16.2</td><td></td><td></td></tr><tr><td>TS-VAD(460h) [12]</td><td>6.61</td><td>14.81</td><td>7.92</td><td>15.97</td></tr><tr><td>Whisper-SS-TTI [27]</td><td>7.97</td><td>21.97</td><td></td><td></td></tr><tr><td>Transformer-SA-ASR [28]</td><td></td><td></td><td>6.40</td><td>8.50</td></tr></table>

Specifically, existing top-tier ASR models such as Qwen3-ASR-1.7b and StepAudio2 lack the ability to recognize multiple speakers. Consequently, they are unable to correctly transcribe highly overlapping synthetic datasets, resulting in WERs between 60% and 160%. On the other hand, general multimodal LLMs such as Gemini-2.5-pro can incorporate reference speech through specific prompts, achieving TS-WERs between 30% and 80% on synthetic datasets and showing some potential for target speaker ASR. However, due to the lack of specialized training, their recognition performance in complex acoustic environments remains limited. In contrast, previous SOTA target speaker ASR models, such as TCP and CONF-TSASR, are specifically optimized for individual datasets, achieving excellent result under a single data distribution but struggling to generalize across all datasets with a single model. Benefiting from training on large-scale multi-speaker mixed corpora, the proposed Xiaomi-CocktailASR-1 successfully overcomes the above limitations. Compared to previous SOTA models, Xiaomi-CocktailASR-1 achieves significant performance improvements on synthetic datasets, with the TS-WER on LibriMix 2mix dropping from 4.84% to 4.11% (a 15.1% relative reduction) and on LibriSpeechMix 2mix from 5.40% to 2.90% (a 46% relative reduction). The experimental results clearly demonstrate that Xiaomi-CocktailASR-1 achieves exceptional performance on standard multi-speaker benchmarks, comprehensively outperforming prior TS-ASR models.

As shown in Table 2, on real-world datasets, ASR models like Qwen3-ASR-1.7b show improved performance compared to their synthetic data results, partly due to the presence of single-speaker segments that mitigate their degradation in complex overlapping conditions. However, their WERs still remains above 30% on these benchmarks, falling far short of practical requirements. For a fair and comprehensive, the comparison focuses on previous SOTA TS-ASR models. On the AMI-SDM dataset, the previous best TS-ASR model SQ-Whisper achieves a TS-WER of 22.0%, while Xiaomi-CocktailASR-1 achieves 21.81%, slightly surpassing this prior result. On the AliMeeting-Far dataset, the previous best MC-TS-ASR model obtains 27.5%, whereas Xiaomi-CocktailASR-1 substantially reduces the TS-WER to 20.63%. Overall, these experimental results demonstrate that Xiaomi-CocktailASR-1 consistently outperforms existing SOTA models and generalizes effectively across diverse datasets and acoustic scenarios.

Table 2: Performance comparison on real-world datasets. Evaluated by WER(%).
<table><tr><td>Model</td><td>AMI SDM</td><td>AliMeeting Far</td></tr><tr><td>Xiaomi-CocktailASR-1</td><td>21.81</td><td>20.63</td></tr><tr><td>Qwen3-ASR-1.7b</td><td>38.18</td><td>39.64</td></tr><tr><td>StepAudio2</td><td>110.50</td><td>76.82</td></tr><tr><td>Whisper Large-v2 [29]</td><td>36.40</td><td></td></tr><tr><td>Gemini-2.5-pro</td><td>52.95</td><td>56.75</td></tr><tr><td>SQ-Whisper [10]</td><td>22.0</td><td></td></tr><tr><td>MC-TS-ASR [30]</td><td></td><td>27.50</td></tr></table>

## 4.3.2 Single Speaker ASR

Table 3: Performance comparison on single-speaker datasets. Evaluated by FRR and Non-empty WER (%).
<table><tr><td>Model</td><td colspan="2">LibriSpeech</td><td colspan="2">AliMeeting-near</td><td colspan="2">AMI-ihm</td><td colspan="2">WenetSpeech(meeting)</td><td colspan="2">CommonVoice(zh)</td></tr><tr><td></td><td>FRR</td><td>Non-empty WER</td><td>FRR</td><td>Non-empty WER</td><td>FRR</td><td>Non-empty WER</td><td>FRR</td><td>Non-empty WER</td><td>FRR</td><td>Non-empty WER</td></tr><tr><td>Xiaomi-CocktailASR-1</td><td>0.36</td><td>1.73</td><td>0.38</td><td>6.57</td><td>0.01</td><td>8.89</td><td>0</td><td>5.81</td><td>0.73</td><td>4.95</td></tr><tr><td>Qwen3-ASR-1.7b</td><td>0</td><td>1.87</td><td>0</td><td>6.39</td><td>0</td><td>10.56</td><td>0</td><td>5.84</td><td>0</td><td>5.39</td></tr><tr><td>StepAudio2</td><td>0</td><td>1.58</td><td>0</td><td>6.82</td><td>0</td><td>37.54</td><td>0</td><td>5.46</td><td>0</td><td>5.07</td></tr><tr><td>Whisper Large-v2</td><td>0</td><td>2.70</td><td></td><td></td><td>0</td><td>16.90</td><td>1</td><td></td><td>0</td><td>26.8</td></tr><tr><td>Gemini-2.5-pro</td><td>21.31</td><td>6.77</td><td>14.95</td><td>16.10</td><td>24.30</td><td>21.02</td><td>0</td><td>27.58</td><td>0.003</td><td>14.01</td></tr></table>

As shown in Table 3, Xiaomi-CocktailASR-1 is compared against current mainstream single-speaker ASR models. The results show that Xiaomi-CocktailASR-1 performs comparably to these dedicated models on most benchmarks, while achieving substantial improvements on certain challenging datasets. Specifically, on the AMI-IHM test set, Xiaomi-CocktailASR-1 achieves a WER of 8.89%, outperforming the best baseline, Qwen3-ASR-1.7b, which yields 10.56%. On the LibriSpeech test set, Xiaomi-CocktailASR-1 achieves a WER of 1.73%, comparable to Qwen2-ASR-1.7b and StepAudio2. Furthermore, on both AliMeeting-Near and WenetSpeech(meeting), Xiaomi-CocktailASR-1 demonstrates highly competitive performance with WERs of 6.57% and 5.81% respectively, staying on par with the baseline Qwen3-ASR-1.7b (6.39% and 5.84%).

Since Xiaomi-CocktailASR-1 have negative sample rejection capability, it may occasionally result in a small probability of false rejection on positive samples. As shown in Table 3, Xiaomi-CocktailASR-1 achieves a low FRR of 0.36% on the LibriSpeech test set, with only a minor impact on the overall WER. In contrast, Gemini-2.5-pro exhibits a strong bias toward rejection, yielding a high FRR of 21.31% in single-speaker scenarios, which substantially degrades its recognition performance. This comparison suggests that integrating rejection capability without a carefully balanced training strategy tends to result in false rejection. By maintaining a low FRR alongside high transcription accuracy, Xiaomi-CocktailASR-1 achieves an effective balance between preserving recognition performance on target-present samples and correctly rejecting target-absent inputs.

## 4.3.3 Negative Rejection

Table 4: Performance on negative test sets. Evaluated by RR(%) ↑.
<table><tr><td>Model</td><td>LibriSpeech Neg</td><td>Aishell Neg</td><td>Chinese in-house Neg</td></tr><tr><td>Xiaomi-CocktailASR-1</td><td>79.59</td><td>75.35</td><td>68.54</td></tr><tr><td>Qwen3-ASR-1.7b</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Gemini-2.5-pro</td><td>81.79</td><td>64.20</td><td>54.7</td></tr><tr><td>StepAudio2</td><td>0</td><td>0</td><td>0</td></tr></table>

This section evaluates the rejection capability of the model when the target speaker is absent. Experimental results in Table 4 show that existing mainstream ASR models, such as Qwen3-ASR-1.7b and StepAudio2, completely lack a rejection mechanism, yielding rejection rates of 0%. In contrast, Xiaomi-CocktailASR-1 demonstrates stable rejection performance on both the English LibriSpeech Neg and Chinese Aishell Neg test sets, achieving rejection rates of 79.59% and 75.35%, respectively. Furthermore, on a real-world recorded Chinese in-house negative dataset, Xiaomi-CocktailASR-1 achieves a rejection rate of 68.54%, indicating its effectiveness in practical scenarios. When compared to the Gemini-2.5-pro model, which achieves 81.79% on LibriSpeech Neg and 64.2% on Aishell Neg, Xiaomi-CocktailASR-1 shows a slightly lower rejection rate in the English scenario but surpasses Gemini-2.5-pro in the Chinese scenario.

Gemini-2.5-pro exhibits a strong bias toward rejection during inference, meaning its negative sample metrics establish a high rejection baseline. The overall rejection performance of Xiaomi-CocktailASR-1 is highly comparable to this strict baseline, demonstrating that our model exhibits an reliable capability to reject negative samples. Notably, the negative sample test sets are constructed from the same source data as the synthetic multi-speaker and single-speaker test sets, allowing for a direct and fair quantification of this trade-off under identical acoustic conditions.

## 4.3.4 Chain-of-Thought

Table 5: Performance on CoT mode. Evaluated by WER(%).
<table><tr><td>Test Set</td><td>Standard mode</td><td>CoT mode</td></tr><tr><td>LibriMix 2mix</td><td>4.11</td><td>3.87</td></tr><tr><td>LibriMix 3mix</td><td>12.287</td><td>12.285</td></tr><tr><td>LibriSpeechMix 2mix</td><td>2.90</td><td>2.88</td></tr><tr><td>LibriSpeechMix 3mix</td><td>4.91</td><td>4.81</td></tr></table>

This section evaluates the performance of the CoT mechanism on synthetic multi-speaker datasets, showed in Table 5. Compared to the standard mode without CoT reasoning, introducing CoT yields a modest but consistent improvement in overall WER. Specifically, the WER on the LibriMix 2mix dataset shows a clear reduction of 0.24%. These results clearly demonstrate that the CoT mechanism effectively assists the model in making more accurate decisions in complex scenarios by guiding it through explicit logical reasoning. Beyond recognition accuracy, the intermediate reasoning steps generated by CoT provide valuable auxiliary information, such as estimated speaker counts and temporal activity patterns, which can potentially benefit downstream tasks requiring deeper speech understanding. This highlights the extensibility of the CoT mechanism beyond mere transcription.

## 5 Conclusion

This paper proposes Xiaomi-CocktailASR-1, an end-to-end TS-ASR architecture based on LLMs. By using reference au dio as a voiceprint prompt, Xiaomi-CocktailASR-1 directly transcribes target speech without explicit speech separation, achieving state-of-the-art performance on various synthetic and real-world multi-speaker benchmarks. Furthermore, the model is equipped with both negative sample rejection and CoT reasoning capabilities, while maintaining competitive recognition performance in single-speaker scenarios. These results demonstrate the effectiveness of leveraging LLMs for target-speaker ASR, and suggest promising directions for extending such reasoning-enhanced architectures to broader speech understanding tasks, such as multi-party conversation analysis and smart meeting assistance.

## References

[1] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, and Junyang Lin. Qwen3-asr technical report, 2026.

[2] Bin Lin, Bo Zhao, Boyong Wu, Chao Yan, Chen Wu, Cheng Yi, Chengyuan Yao, et al. Stepaudio 2.5 technical report, 2026.

[3] Ye Bai, Jingping Chen, Jitong Chen, Wei Chen, Zhuo Chen, Chuang Ding, Linhao Dong, Qianqian Dong, Yujiao Du, Kepan Gao, et al. Seed-asr: Understanding diverse speech and contexts with llm-based speech recognition, 2024.

[4] Takafumi Moriya, Shota Horiguchi, Marc Delcroix, Ryo Masumura, Takanori Ashihara, Hiroshi Sato, Kohei Matsuura, and Masato Mimura. Alignment-free training for transducer-based multi-talker asr. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025.

[5] Mohan Shi, Zengrui Jin, Yaoxun Xu, Yong Xu, Shi-Xiong Zhang, Kun Wei, Yiwen Shao, Chunlei Zhang, and Dong Yu. Advancing multi-talker asr performance with large language models. In 2024 IEEE Spoken Language Technology Workshop (SLT), pages 14–21, 2024.

[6] Naoyuki Kanda, Yashesh Gaur, Xiaofei Wang, Zhong Meng, and Takuya Yoshioka. Serialized output training for end-to-end overlapped speech recognition. In Interspeech 2020, pages 2797–2801, 2020.

[7] Naoyuki Kanda, Shota Horiguchi, Ryoichi Takashima, Yusuke Fujita, Kenji Nagamatsu, and Shinji Watanabe. Auxiliary interference speaker loss for target-speaker speech recognition. Interspeech 2019, pages 236–240, 2019.

[8] Yang Zhang, Krishna C. Puvvada, Vitaly Lavrukhin, and Boris Ginsburg. Conformer-based target-speaker automatic speech recognition for single-channel audio. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023.

[9] Zili Huang, Desh Raj, Paola García, and Sanjeev Khudanpur. Adapting self-supervised models to multi-talker speech recognition using speaker embeddings. In ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2023.

[10] Pengcheng Guo, Xuankai Chang, Hang Lv, Shinji Watanabe, and Lei Xie. Sq-whisper: Speaker-querying based whisper model for target-speaker asr. IEEE Transactions on Audio, Speech and Language Processing, 33:175–185, 2025.

[11] Hao Ma, Zhiyuan Peng, Mingjie Shao, Jing Li, and Ju Liu. Extending whisper with prompt tuning to target-speaker asr. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12516–12520, 2024.

[12] Chikara Maeda and Muhammad Shakeel. Joint target-speaker asr and activity detection. In Proc. Interspeech 2025, pages 1683–1687, 2025.

[13] Lingwei Meng, Shujie Hu, Jiawen Kang, Zhaoqing Li, Yuejiao Wang, Wenxuan Wu, Xixin Wu, Xunying Liu, and Helen Meng. Large language model can transcribe speech in multi-talker scenarios with versatile instructions. In ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5, 2025.

[14] Minsoo Kim and SangHun Kim. Target-speaker llm-asr with speaker-aware speech encoder. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 16732–16736, 2026.

[15] Yiru Zhang, Hang Su, Lichun Fan, Zhenbo Luo, and Jian Luan. Thinking in cocktail party: Chain-of-thought and reinforcement learning for target speaker automatic speech recognition, 2026.

[16] Alexei Baevski, Arun Babu, Wei-Ning Hsu, and Michael Auli. Efficient self-supervised learning with contextualized target representations for vision, speech and language. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 1416–1429. PMLR, 23–29 Jul 2023.

[17] Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. Wavlm: Large-scale self-supervised pre-training for full stack speech processing. IEEE Journal ofSelected Topics in Signal Processing, 16(6):1505–1518, 2022.

[18] Qwen Team. Qwen3 technical report, 2025.

[19] Fan Yu, Shiliang Zhang, Pengcheng Guo, Yihui Fu, Zhihao Du, Siqi Zheng, Weilong Huang, Lei Xie, Zheng-Hua Tan, DeLiang Wang, Yanmin Qian, Kong Aik Lee, Zhijie Yan, Bin Ma, Xin Xu, and Hui Bu. Summary on the ICASSP 2022 multi-channel multi-party meeting transcription grand challenge. In Proc. ICASSP. IEEE, 2022.

[20] Wessel Kraaij, Thomas Hain, Mike Lincoln, and Wilfried Post. The ami meeting corpus. In Proc. International Conference on Methods and Techniques in Behavioral Research, pages 1–4, 2005.

[21] Joris Cosentino, Manuel Pariente, Samuele Cornell, Antoine Deleforge, and Emmanuel Vincent. Librimix: An open-source dataset for generalizable speech separation, 2020.

[22] Hui Wang, Siqi Zheng, Yafeng Chen, Luyao Cheng, and Qian Chen. Cam++: A fast and efficient network for speaker verification using context-aware masking. In Interspeech 2023, pages 5301–5305, 2023.

[23] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024.

[24] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur. Librispeech: An asr corpus based on public domain audio books. In 2015 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 5206–5210, 2015.

[25] Hui Bu, Jiayu Du, Xingyu Na, Bengu Wu, and Hao Zheng. Aishell-1: An open-source mandarin speech corpus and a speech recognition baseline. In 2017 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O-COCOSDA), pages 1–5, 2017.

[26] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

[27] Lingwei Meng, Jiawen Kang, Yuejiao Wang, Zengrui Jin, Xixin Wu, Xunying Liu, and Helen Meng. Empowering Whisper as a Joint Multi-Talker and Target-Talker Speech Recognition System. In Interspeech 2024, pages 4653–4657, 2024.

[28] Naoyuki Kanda, Guoli Ye, Yashesh Gaur, Xiaofei Wang, Zhong Meng, Zhuo Chen, and Takuya Yoshioka. End-to-End Speaker-Attributed ASR with Transformer. In Interspeech 2021, pages 4413–4417, 2021.

[29] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett, editors, Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 28492–28518. PMLR, 23–29 Jul 2023.

[30] Mohan Shi, Jie Zhang, Zhihao Du, Fan Yu, Qian Chen, Shiliang Zhang, and Li-Rong Dai. A comparative study on multichannel speaker-attributed automatic speech recognition in multi-party meetings. In 2023 Asia Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC), pages 1943–1948, 2023.

## Contributors

All contributors are listed in alphabetical order by their latest names.

Core Contributors   
Lichun Fan   
Hang Su   
Yiru Zhang   
Contributors Tao Li Lian Li   
Yuquan Liang Chang Liu   
Yifeng Wang   
Wenhao Yang Ying Zeng

Supervisors Jian Luan Heng Qu Cong Zou