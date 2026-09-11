# Nuha-Speech: Building General-Purpose Arabic Speech-LLMs

Yingzhi Wang Elm Company, KSA ywang@elm.sa

Reem Alhazzani Elm Company, KSA ralhazzani@elm.sa

Muhammad Alqurishi Elm Company, KSA mualqurishi@elm.sa

## Abstract

As Speech Large Language Models (speech-LLMs) become increasingly multilingual, Arabic remains significantly underrepresented, highlighting the need for dedicated infrastructure to train and evaluate Arabic speech-LLMs.

To address this gap, we introduce Nuha-Speech, a comprehensive initiative to develop generalpurpose Arabic speech-LLMs spanning dataset construction, model training, and systematic evaluation. Specifically, we constructed a large-scale Arabic Speech Question-Answering (SQA) corpus comprising over 1.5 million training samples to allow instruction tuning over a broad range of core speech tasks. Then, the corpus was used for supervised fine-tuning based on Qwen-Omni model variants at different scales. Finally, we designed an evaluation framework featuring diverse tasks and tailored metrics. Through this work, we aim to establish foundational infrastructures for Arabic Speech-LLMs under constraints imposed by limited Arabic speech resources.

## 1 Introduction

Speech Large Language Models (Speech-LLMs) have seen swift advancements in recent years. However, so far, few models provide support for Arabic. The Octopus family models (Althubaiti et al., 2025) introduce Arabic-centric Speech-LLMs that can handle three tasks: Automatic Speech Recognition(ASR), Arabic–to-English speech translation, and dialect identification. The models follow a Salmonn-style (Tang et al., 2023) architecture incorporating both semantic and acoustic encoders, and include a distilled variant that achieves competitive performance. While Octopus represents a valuable effort toward enhancing Speech-LLMs with Arabic capabilities, compared to recent Speech-LLM advances, its task coverage remains very limited and lacks zero-shot generalization, limiting its utility as a general-purpose Speech-LLM.

Furthermore, its reliance on predominantly private datasets restricts its reproducibility and broader community adoption.

The Qwen family has released a series of highperformance speech-LLMs, with the level of Arabic language support varying from model to model. Qwen2-Audio(Chu et al., 2024) is a popular speech-LLM designed for voice-chat and audio analysis. It outperforms prior models on instruction-following benchmarks. Although Qwen2-Audio relies on Whisper-large-v3(Radford et al., 2023), which supports Arabic among many other languages, all available demos and test cases are only in English and Chinese, there is no explicit proof that the model can follow instructions given Arabic speech inputs. Qwen2.5-Omni(Xu et al., 2025a) similarly demonstrates strong end-to-end speech instruction following in over 29 languages, benefiting from its diverse multilingual training data. However, its public evaluation and examples have focused on English and Chinese, its ability in Arabic remains unverified and must be inferred. By contrast, Qwen3-Omni(Xu et al., 2025b) stands out as explicitly multilingual, including Arabic among the 19 supported speech-input languages. This enables strong Arabic speech comprehension across a wider range of linguistic contexts than most comparable models.

The scarcity of Arabic-capable speech-LLMs reflects several deep challenges. First, when considering non-ASR speech tasks, the number of publicly accessible Arabic speech corpora remains extremely limited. For instance, to date, there remains a striking lack of public Arabic speech corpora of sufficient scale for tasks such as Speech Emotion Recognition (SER) or Speech Question Answering (SQA). In addition, Arabic speech instructiontuning data is extremely scarce: the majority of speech instruction-following corpora are in English, and although a few have been translated into Arabic, genuine Arabic speech–instruction pairs are almost non-existent. Second, there is no widely adopted evaluation benchmark for Arabic speech LLMs. The existing benchmarks remain heavily centered on mainstream languages like English (Huang et al., 2024; Yang et al., 2024; Wang et al., 2025a,b), with most speech tasks derived from English datasets, making it difficult to assess models understanding of less-represented languages like Arabic.

To fill this gap, we propose Nuha-Speech, a comprehensive effort aimed at building generalpurpose Arabic speech LLMs via dataset construction, model training, and benchmark development. By combining public and curated datasets, we assembled a training dataset of 1.5 million Arabic speech instruction-following samples. This dataset spans core speech tasks covering both speech understanding and speech paralinguistics. Moreover, we leveraged mostly publicly available datasets and transparent, reproducible curation strategies, facilitating dataset replication by the research community. We then used this dataset to fine-tune three Qwen-Omni model variants at different parameter scales, aiming to equip them with full Arabic comprehension capabilities. Finally, we constructed test sets for all involved tasks, adopted different evaluation metrics to form a multi-task benchmark, and evaluated models in both pre- and post-finetuning settings. The results demonstrate that the fine-tuned models achieved clear improvements across all listed tasks.

## 2 Nuha-Speech Dataset

In this section, we outline how the training corpus was constructed for each task. Table 1 provides an overview of the collected datasets, including tasks, sources, and sizes. To unify all tasks within a shared semantic space, each sample is formatted as a {speech, instruction, output} tuple. In this work, we exclusively focus on training models with textonly outputs. For each task, we also designed a broad instruction set using GPT-5 to better support zero-shot generalization and avoid overfitting to narrow prompts.

## 2.1 Automatic Speech Recognition (ASR)

ASR serves as one of the most fundamental speech tasks. In our training, ASR is also regarded as the primary objective. We selected four largescale, multi-dialect public datasets (SADA(Alharbi et al., 2024), Common Voice(Ardila et al., 2019),

Table 1: Overview of the training corpus for Nuha-Speech.
<table><tr><td>Task</td><td>Data Source</td><td>#Samples</td></tr><tr><td rowspan="5">ASR</td><td>MGB-2</td><td>278K</td></tr><tr><td>MASC</td><td>263K</td></tr><tr><td>SADA</td><td>171K</td></tr><tr><td>Common Voice</td><td>78K</td></tr><tr><td>CoVoST-v2 (Ar2En)</td><td>5K</td></tr><tr><td>AST</td><td>ASR-Curated MGB-2 Closed-ended</td><td>150K</td></tr><tr><td rowspan="2">SQA</td><td></td><td>150K</td></tr><tr><td>MGB-2 Open-ended</td><td>150K</td></tr><tr><td>DI</td><td>ADI-17</td><td>150K</td></tr><tr><td>SER</td><td>Elevenlabs-Syn</td><td>20K</td></tr><tr><td rowspan="3">AR</td><td>SADA</td><td>5K</td></tr><tr><td>Common Voice</td><td>5K</td></tr><tr><td>Elevenlabs-Syn</td><td>10K</td></tr><tr><td>GR</td><td>SADA</td><td>51K</td></tr><tr><td rowspan="3">Multi-SQA</td><td>SADA</td><td>5K</td></tr><tr><td>Emotion Reasoning</td><td>10K</td></tr><tr><td></td><td>1501K</td></tr></table>

MASC(Al-Fetyani et al., 2023), and MGB-2(Ali et al., 2016)) and removed segments outside the 1–10s range to optimize training efficiency. This yields roughly 790K ASR training samples in total.

## 2.2 Automatic Speech Translation (AST)

Automatic Speech Translation (AST) maps sourcelanguage spoken input to target-language textual translation in an end-to-end way. AST is highly valuable in real-world workflows for Arabic, especially when translating into English. In our work, we adopt the Arabic-to-English split of the widely used CoVoST-v2 (Wang et al., 2021) dataset. Due to its very limited number of samples, we then randomly selected 150K samples from the ASR dataset collected above and translated their transcriptions into English using the Qwen3-32B<sup>1</sup> model to obtain a sufficient number of training samples.

## 2.3 Speech Question Answering (SQA)

In speech understanding, the SQA task focuses on querying the semantic and factual information given spoken content. It closely aligns with how humans naturally interact with conversational systems. However, there is currently no suitable Arabic dataset that includes both spoken context and textual QA pairs. Given the rich content of the MGB-2 dataset, we utilized its transcriptions as input to generate 150K closed-ended and 150K openended QA pairs using the Qwen3-32B model. The closed-ended QA focuses on enhancing structured query handling and exact content retrieval, while the open-ended QA promotes higher-level skills such as cross-sentence reasoning and expressive language generation.

## 2.4 Dialect Identification (DI)

Dialect Identification is a crucial task in Arabic speech processing, since one of Arabic’s defining features is its diverse dialects. Accurate DI is therefore important for building robust Arabic speech technologies, particularly in multilingual and multi-dialectal settings where dialectal variation can significantly affect downstream performance. For the DI task, we randomly selected 150K training samples from the well-established ADI-17 dataset, which is designed to classify 17 different Arabic dialects.

## 2.5 Speech Emotion Recognition (SER)

Speech emotion recognition in Arabic has long been constrained by the lack of large-scale opensource datasets. In our work, using an emotionconversion TTS model from Elevenlabs<sup>2</sup>, we synthesized 20K emotional samples across four balanced categories: angry, happy, sad, and neutral. Specifically, each category’s samples were randomly synthesized by 8 separate speakers in order to reduce the risk of speaker-specific bias. To ensure the effectiveness of emotion conversion, we activated style exaggeration and enforced stronger speaker similarity control during the synthesis process.

## 2.6 Age Recognition (AR)

Similarly, few Arabic speech datasets contain age annotations, and existing ones exhibit a strong bias toward adult and middle-aged speakers, offering minimal coverage of younger and elderly populations. In our work, we defined age classification into three groups: young (under 18), adult (18–60), and elder (above 60). We extracted all available young and elder samples from SADA and Common Voice, then added a balanced number of adult samples, resulting in a dataset of 10K samples. Additionally, we synthesized a balanced set of speech samples for all three age groups using the same TTS model from ElevenLabs, with 10 distinct speakers per group, resulting in another 10K samples.

## 2.7 Gender Recognition (GR)

Given the observed cross-lingual variability in gender recognition performance(Attanasio et al., 2024), we also included the GR task into our model training. A total of 51K samples were extracted from the SADA dataset, with an equal distribution of male and female speakers.

## 2.8 Multi-SQA

To enhance the model’s robustness and adaptability in complex spoken language scenarios, we also incorporated the multi-turn SQA task. First, leveraging the gender, age, and ASR annotations available in the SADA dataset, we constructed speechanalytics-style multi-turn SQA samples by merging these three isolated tasks, resulting in 5K samples with a balanced distribution across gender and age groups.

Second, a further multi-turn SQA task was designed that focused on emotion reasoning, where the model is first asked to recognize the speech emotion, then transcribe the spoken content, and finally infer the emotion correlation between the two modalities. We employed Qwen3-32B to generate 10K such cross-modality reasoning QA pairs based on designed SER dataset and its transcriptions.

## 3 Instruction Tuning

This section presents the baseline models and the instruction tuning setup.

## 3.1 Baseline Models

We used the Qwen-Omni model series as our finetuning baseline, taking advantage of the built-in speech input support and speech understanding capabilities, along with the Qwen LLM backbone’s proven generalization strength for low-resource languages like Arabic(Qian et al., 2024; Althnian et al., 2025). The included models are: Qwen2.5-omni-3B<sup>3</sup>, Qwen2.5-omni-7B<sup>4</sup> and Qwen3-omni-30B<sup>5</sup>.

Qwen2.5-Omni adopts the Whisper-large-v3 encoder used in Qwen2-Audio and is powered by LLM from the Qwen2.5 series. Qwen3 improves upon this by integrating a more sophisticated AuT audio encoder and leveraging a more modern, efficient Mixture-of-Experts (MoE) architecture from the Qwen3 model series. We incorporated both the 3B and 7B variants of Qwen2.5 into our candidate model pool in order to broaden the range of evaluated models and to enable a more direct and systematic comparison of performance across different model scales.

## 3.2 Experimental Setups

Table 2: Nuha-Speech training curriculum.
<table><tr><td>Stage</td><td>Task</td><td>Data</td><td>Module</td><td>#Epochs</td></tr><tr><td>1</td><td>ASR</td><td>ASR-790K</td><td>LLM (Lora)</td><td>3</td></tr><tr><td>2</td><td>All</td><td>ASR-200K Others-711K</td><td>LLM (Lora)</td><td>2</td></tr></table>

Following previous studies(Chu et al., 2024; Das et al., 2024; Du et al., 2025), we adopted a twostage training curriculum for Nuha-Speech, as illustrated in Table 2. In the first stage, the model was trained only to perform the ASR task, aiming to establish a strong audio-text mapping as the foundation for subsequent speech understanding tasks. In the second stage, we trained the models on all tasks. Given that the models had already acquired ASR capabilities in stage 1, we reduced the ASR training samples from 790K to a randomly sampled 200K subset. This not only helped preserve ASR performance but also reduced training time and improved balance across tasks. Recognizing that the audio encoders in all baseline models can already effectively process Arabic speech, we froze the encoders and trained only the LLM with LoRA(Hu et al., 2022) in both stages to stabilize training. According to the Arabic ASR leaderboard<sup>6</sup>, the Qwen3- Omni-30B model has demonstrated competitive ASR performance, ranking among the top-rated models in the benchmark. Therefore, for Qwen3- Omni-30B model we skipped the ASR-only Stage 1 and proceeded directly to Stage 2. In contrast, both Stage 1 and Stage 2 were applied to Qwen2.5- Omni-3B and Qwen2.5-Omni-7B models to complete the full training pipeline.

More specifically, we trained Qwen2.5-Omni-3B and Qwen2.5-Omni-7B using a global batch size of 16 and a learning rate of 1e-4 throughout both stages. For Qwen3-Omni-30B, we adopted a relatively smaller batch size of 8 while maintaining the same learning rate, and set the MoE router’s load balancing loss coefficient to 1e-3. LoRA was configured with a rank of 8, alpha of 32, and applied to all linear layers. Finally, corresponding to the parameter sizes of their respective baseline models, we refer to our fine-tuned models as Nuha-Speech-3B, Nuha-Speech-7B, and Nuha-Speech-30B.

## 4 Nuha-Speech Benchmark

In this section, we describe the construction of the test datasets and Nuha-Speech benchmark, followed by a detailed analysis of the experimental results.

## 4.1 Test Sets and Metrics

To comprehensively evaluate model performance, we adopted the same 7 tasks involved during the training stage, covering both speech understanding and speech paralinguistics. Table 3 summarizes the detailed evaluation setup, including tasks, datasets, metrics, and the number of samples.

Table 3: Overview of the evaluation setups: tasks, datasets, metrics, and number of samples.
<table><tr><td>Task</td><td>Data Source</td><td>Metric</td><td>#Samples</td></tr><tr><td>ASR</td><td>OUAAL</td><td>Average WER</td><td>47K</td></tr><tr><td>SQA</td><td>SD-QA-Syn</td><td>LLM as Judge</td><td>1.2K</td></tr><tr><td>AST</td><td>CoVoST-v2</td><td>Blue Sentence-Similarity</td><td>2.3K</td></tr><tr><td>DI</td><td>ADI-17</td><td>Accuracy</td><td>12K</td></tr><tr><td>SER</td><td>Synthesized Speech</td><td>Accuracy</td><td>1.1K</td></tr><tr><td>AR</td><td>Common Voice Elevenlabs-Syn</td><td>Accuracy</td><td>1.2K</td></tr><tr><td>GR</td><td>Common Voice</td><td>Accuracy</td><td>2K</td></tr></table>

Similar to the training set construction, the evaluation data also combines existing corpora with carefully curated samples. For ASR evaluation, we followed the previously mentioned Open Universal Arabic ASR Leaderboard (Wang et al., 2024), a multi-dialect benchmark built on 6 datasets with 46,757 test samples. We report results in terms of average Word Error Rate (WER). For AST, we use the Ar-En subset of the CoVoST-2 test set containing 2.3K samples. Evaluation metrics include BLEU<sup>7</sup> for lexical overlap and Gemma-based sentence similarity<sup>8</sup> to capture semantic alignment. For the SQA task, we utilize the SD-QA (Faisal et al., 2021) Arabic textual QA dataset and synthesize its contextual passages into speech using the XTTS-v2 (Casanova et al., 2024)<sup>9</sup> model with 58 distinct speaker voices. We then used Whisper-Large-v3<sup>10</sup> to transcribe the synthesized speech and applied a CER-based filtering criterion to remove degraded speech synthesis outputs, yielding 1,237 high-quality synthesized utterances. As for evaluation, we employed LLaMA-3.3-70B-Instruct<sup>11</sup> as an LLM judge, rating models’ outputs in terms of relevance, correctness, and conciseness against the corresponding reference answers. This results in an overall score ranging from 0 to 3 for each model output.

Table 4: Comprehensive performance benchmark of Nuha-Speech fine-tuned models and their original baselines.
<table><tr><td colspan="3">ASR</td><td colspan="2">AST</td><td rowspan="2">SQA LLM-Score↑</td><td rowspan="2">DI Acc%↑</td><td rowspan="2">SER Acc%↑</td><td rowspan="2">AR Acc%↑</td><td rowspan="2">GR Acc%↑</td></tr><tr><td>Model</td><td>WER%↓</td><td>CER%↓</td><td>Bleu↑</td><td>Sim↑</td></tr><tr><td>Qwen2.5-omni-3B</td><td></td><td></td><td>39.27</td><td>0.813</td><td>2.30</td><td>9.70</td><td>51.93</td><td>41.75</td><td>60.55</td></tr><tr><td>Nuha-Speech-3B</td><td>36.83%</td><td>17.18</td><td>48.32</td><td>0.884</td><td>2.60</td><td>57.83</td><td>88.14</td><td>83.00</td><td>98.85</td></tr><tr><td>Qwen2.5-omni-7B</td><td>74.74</td><td>48.33</td><td>42.08</td><td>0.836</td><td>2.37</td><td>13.66</td><td>72.14</td><td>33.33</td><td>86.30</td></tr><tr><td>Nuha-Speech-7B</td><td>34.91</td><td>16.09</td><td>50.60</td><td>0.896</td><td>2.66</td><td>72.65</td><td>86.64</td><td>73.00</td><td>99.50</td></tr><tr><td>Qwen3-omni-30B</td><td>30.71</td><td>13.67</td><td>48.71</td><td>0.889</td><td>2.59</td><td>16.78</td><td>80.23</td><td>42.67</td><td>82.25</td></tr><tr><td>Nuha-Speech-30B</td><td>29.66</td><td>14.55</td><td>51.73</td><td>0.908</td><td>2.62</td><td>76.02</td><td>84.88</td><td>85.75</td><td>94.30</td></tr></table>

For the dialect identification (DI) task, we directly incorporated the ADI-17(Shon et al., 2020) test set, which includes 12,150 samples evenly distributed over 17 dialect classes. For SER, we followed the same data synthesis strategy as in the training set and generated another 1.1K emotional samples with 4 balanced categories. For AR, we also sampled from the Common Voice test set and generated additional utterances using ElevenLabs. The final dataset comprises 1.2K samples evenly distributed across 3 age groups. For GR, we selected 2,000 gender-balanced samples from the Common Voice test set.

For the four speech paralinguistic tasks described above, we uniformly adopted accuracy as the evaluation metric, any response falling outside the defined domain or predicting an excluded class is considered a misclassification. The full evaluation scripts, LLM-as-Judge prompts and test sets used in our experiments will be made publicly available through GitHub<sup>12</sup>.

## 4.2 Results

Table 4 summarizes the performance of all evaluated models, including both the baseline and finetuned versions, across all seven tasks. From the model perspective, all models demonstrated improvements across all tasks after fine-tuning compared to the baseline. This improvement was more significant for the 3B and 7B models, whereas the 30B model showed smaller gains, since it has already obtained strong pre-trained semantic capabilities during pre-training. From the task perspective, paralinguistic tasks exhibited more substantial performance gains than semantic tasks after finetuning, suggesting a potential under-representation of Arabic speech paralinguistics tasks in the pretraining phase.

More specifically, for the ASR task, the Nuha-Speech-30B model maintained the solid performance that had already been established by its baseline. The Nuha-Speech-3B and Nuha-Speech-7B models both achieved significantly improved ASR capability in comparison with their baselines. For the AST task, all baseline models demonstrate a fundamental capability with reasonable BLEU scores and sentence similarities, and the Nuha-Speech-30B model consistently achieved the best performance across both metrics. Similarly, in the SQA evaluation, all baseline models attained an average score greater than two, with the fine-tuned Nuha-Speech-7B model marginally outperforming the other two. From the three semantic tasks above, we observe that although not all baseline models were directly trained on the Arabic ASR task, every model nonetheless acquired a certain degree of Arabic semantic understanding capacity, owing to the diversity of the training tasks through which Arabic semantics were introduced.

For speech paralinguistic tasks, we observe that baseline models initially exhibit very poor capability in dialect identification and age recognition, with accuracies close to random guessing. After fine-tuning, each model exhibited a dramatic, multifold increase in performance on both tasks, with Nuha-Speech-30B remaining the top-performing model. For the SER and GR tasks, the baseline models showed adequate initial performance, and after fine-tuning, they all further improved and reached excellent levels.

The results obtained from both speech understanding and speech paralinguistics benchmarks demonstrate the effectiveness of our dataset design and confirm the validity of the adopted fine-tuning approach.

It should be noted that we also included Qwen2- audio-7B-instruct<sup>13</sup> as an additional baseline model in our experiments due to its broader set of speech tasks involved in pretraining. However, after being fine-tuned on the same training corpus, it consistently underperformed across all evaluated tasks compared to fine-tuning the Qwen2.5-omni-7B model which is of the same size. We therefore excluded it from our final benchmark to maintain a more competitive set of baseline models. By introducing the first systematic Arabic speech multi-task benchmark, we hope this benchmark will establish a standard for the evaluation of future Arabic Speech-LLMs.

## 5 Conclusions

In this paper, we present a collection of generalpurpose Arabic speech-LLMs that, for the first time, provides unified support for broad Arabic speech tasks while enabling Arabic instructionfollowing. To achieve this goal, we constructed an Arabic speech-LLM training dataset under limitedresource conditions, while promoting reproducibility through the use of largely open-access datasets and transparent curation strategies. Additionally, we fully documented our fine-tuning protocols and established a common Arabic speech-LLM benchmark that jointly evaluates speech understanding and speech paralinguistics capabilities.

Our work aims to lay the foundation for the Arabic speech-LLM infrastructure and to provide useful guidance for future research and development in building more capable general-purpose Arabic speech-LLMs.

## References

Mohammad Al-Fetyani, Muhammad Al-Barham, Gheith Abandah, Adham Alsharkawi, and Maha Dawas. 2023. Masc: Massive arabic speech corpus. In 2022 IEEE Spoken Language Technology Workshop (SLT), pages 1006–1013. IEEE.

Sadeen Alharbi, Areeb Alowisheq, Zoltán Tüske, Kareem Darwish, Abdullah Alrajeh, Abdulmajeed Alrowithi, Aljawharah Bin Tamran, Asma Ibrahim, Raghad Aloraini, Raneem Alnajim, and 1 others. 2024. Sada: Saudi audio dataset for arabic. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 10286–10290. IEEE.

Ahmed Ali, Peter Bell, James Glass, Yacine Messaoui, Hamdy Mubarak, Steve Renals, and Yifan Zhang. 2016. The mgb-2 challenge: Arabic multi-dialect broadcast media recognition. In 2016 IEEE Spoken Language Technology Workshop (SLT), pages 279– 284. IEEE.

Alhanoof Althnian, Norah A Alzahrani, Shaykhah Z Alsubaie, Eman Albilali, Ahmed Abdelali, Nouf M Alotaibi, M Saiful Bari, Yazeed Alnumay, Abdulhamed Alothaimen, Maryam Saif, and 1 others. 2025. Araeval: An arabic multi-task evaluation suite for large language models. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 33025–33049.

Sara Althubaiti, Vasista Sai Lodagala, Tjad Clark, Yousseif Ahmed Elshahawy, Daniel Izham, Abdullah Alrajeh, Aljawahrah Bin Tamran, and Ahmed Ali. 2025. Octopus: Towards building the arabic speech llm suite. In Proceedings of The Third Arabic Natural Language Processing Conference, pages 425–435.

Rosana Ardila, Megan Branson, Kelly Davis, Michael Henretty, Michael Kohler, Josh Meyer, Reuben Morais, Lindsay Saunders, Francis M Tyers, and Gregor Weber. 2019. Common voice: A massivelymultilingual speech corpus. arXiv preprint arXiv:1912.06670.

Giuseppe Attanasio, Beatrice Savoldi, Dennis Fucci, and Dirk Hovy. 2024. Twists, humps, and pebbles: Multilingual speech recognition models exhibit gender performance gaps. arXiv preprint arXiv:2402.17954.

Edresson Casanova, Kelly Davis, Eren Gölge, Görkem Göknar, Iulian Gulea, Logan Hart, Aya Aljafari, Joshua Meyer, Reuben Morais, Samuel Olayemi, and 1 others. 2024. Xtts: a massively multilingual zero-shot text-to-speech model. arXiv preprint arXiv:2406.04904.

Yunfei Chu, Jin Xu, Qian Yang, Haojie Wei, Xipin Wei, Zhifang Guo, Yichong Leng, Yuanjun Lv, Jinzheng He, Junyang Lin, and 1 others. 2024. Qwen2-audio technical report. arXiv preprint arXiv:2407.10759.

Nilaksh Das, Saket Dingliwal, Srikanth Ronanki, Rohit Paturi, Zhaocheng Huang, Prashant Mathur, Jie Yuan, Dhanush Bekal, Xing Niu, Sai Muralidhar Jayanthi, and 1 others. 2024. Speechverse: A large-scale generalizable audio language model. arXiv preprint arXiv:2405.08295.

Yexing Du, Youcheng Pan, Ziyang Ma, Bo Yang, Yifan Yang, Keqi Deng, Xie Chen, Yang Xiang, Ming Liu, and Bing Qin. 2025. Making llms better manyto-many speech-to-text translators with curriculum learning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12466–12478.

Fahim Faisal, Sharlina Keshava, Md Mahfuz Ibn Alam, and Antonios Anastasopoulos. 2021. Sd-qa: Spoken dialectal question answering for the real world. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3296–3315.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3.

Chien-yu Huang, Ke-Han Lu, Shih-Heng Wang, Chi-Yuan Hsiao, Chun-Yi Kuan, Haibin Wu, Siddhant Arora, Kai-Wei Chang, Jiatong Shi, Yifan Peng, and 1 others. 2024. Dynamic-superb: Towards a dynamic, collaborative, and comprehensive instruction-tuning benchmark for speech. In ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 12136–12140. IEEE.

Zhaozhi Qian, Faroq Altam, Muhammad Alqurishi, and Riad Souissi. 2024. Cameleval: Advancing culturally aligned arabic language models and benchmarks. arXiv preprint arXiv:2409.12623.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pages 28492–28518. PMLR.

Suwon Shon, Ahmed M. Ali, Younes Samih, Hamdy Mubarak, and James R. Glass. 2020. Adi17: A finegrained arabic dialect identification dataset. ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 8244–8248.

Changli Tang, Wenyi Yu, Guangzhi Sun, Xianzhao Chen, Tian Tan, Wei Li, Lu Lu, Zejun Ma, and Chao Zhang. 2023. Salmonn: Towards generic hearing abilities for large language models. arXiv preprint arXiv:2310.13289.

Bin Wang, Xunlong Zou, Geyu Lin, Shuo Sun, Zhuohan Liu, Wenyu Zhang, Zhengyuan Liu, AiTi Aw, and Nancy Chen. 2025a. Audiobench: A universal benchmark for audio large language models. In Proceedings of the 2025 Conference of the Nations of

the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 4297–4316.

Changhan Wang, Anne Wu, Jiatao Gu, and Juan Pino. 2021. Covost 2 and massively multilingual speech translation. In Interspeech, volume 2021, pages 2247–2251.

Dingdong Wang, Jincenzi Wu, Junan Li, Dongchao Yang, Xueyuan Chen, Tianhua Zhang, and Helen Meng. 2025b. Mmsu: A massive multi-task spoken language understanding and reasoning benchmark. arXiv preprint arXiv:2506.04779.

Yingzhi Wang, Anas Alhmoud, and Muhammad Alqurishi. 2024. Open universal arabic asr leaderboard. arXiv preprint arXiv:2412.13788.

Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, and 1 others. 2025a. Qwen2. 5-omni technical report. arXiv preprint arXiv:2503.20215.

Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, and 19 others. 2025b. Qwen3-omni technical report. Preprint, arXiv:2509.17765. Technical Report.

Qian Yang, Jin Xu, Wenrui Liu, Yunfei Chu, Ziyue Jiang, Xiaohuan Zhou, Yichong Leng, Yuanjun Lv, Zhou Zhao, Chang Zhou, and 1 others. 2024. Airbench: Benchmarking large audio-language models via generative comprehension. arXiv preprint arXiv:2402.07729.