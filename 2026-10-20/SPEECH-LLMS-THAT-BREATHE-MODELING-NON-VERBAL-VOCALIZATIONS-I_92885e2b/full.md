# SPEECH LLMS THAT BREATHE: MODELING NON-VERBAL VOCALIZATIONS IN DIALOGUE

Heyang Liu<sup>1,2</sup>, Lixin Zhang<sup>1</sup>, Jiayi Huang<sup>1</sup>, Ziyang Cheng<sup>1,2</sup>, Yuhao Wang<sup>1</sup>, Ronghua Wu<sup>2</sup>, Qunshan Gu<sup>2</sup>, Yanfeng Wang<sup>1</sup>, Yu Wang<sup>1†</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Ant Group

## ABSTRACT

The user experience of speech interaction models has improved, benefiting from the modeling of paralinguistic attributes and advances in speech reconstruction paradigms. However, human communication follows extensive norms regarding non-verbal vocalizations (NVVs) spanning respiratory, affective, and discourse. The omission of these events imposes inherent limitations on the naturalness of current approaches. In this work, we cover 13 major types of NVVs and construct a natural dialogue dataset that conforms to human conversational norms through an automated synthesis pipeline. We enhance the ability of speech interaction models to generate contextually appropriate NVV events through multi-stage fine-tuning and NVV-aware GRPO (Group Relative Policy Optimization). Based on distinct modeling paradigms, we substantially improve the expressiveness of VocalNet and VITA-Qinyu in empathy interactions and roleplay dialogues, and further discuss the architectural necessity for achieving high-quality natural speech generation.

Index Terms— Speech LLM, non-verbal vocalization

## 1. INTRODUCTION

Human speech interaction conveys rich information beyond textual content, encompassing speaker-specific timbres, paralinguistic emotional cues, and diverse non-verbal vocalizations (NVVs). These characteristics complement the linguistic content and play an important role in conveying information that is difficult to capture through text alone. In particular, NVV events such as sighs, laughter, and affirmations can reflect subjective attitudes and conversational intentions, and are essential to expressive human dialogues [1]. Speech interaction models, especially speech large language models (Speech LLMs), receive users’ vocal instructions and generate corresponding spoken responses, moving toward increasingly natural and human-like interactions [2, 3, 4]. Beyond accurately generating linguistic content, an ideal Speech LLM should be capable of modeling the rich vocal behaviors that naturally accompany human communication. Recent research has increasingly explored the emotional characteristics, enabling more affective speech generation. However, the integration of NVVs appropriate for conversational norms remains underexplored, and current Speech LLMs still fail to naturally produce conversational vocal behaviors.

The advances in speech synthesis models have made it increasingly feasible to develop interactive models capable of producing NVV events. Recent approaches support a predefined set of NVV event types [5], while the latest closedsource models can incorporate appropriate NVV vocalizations while maintaining high-quality speech generation [6]. Natural conversation requires the model to determine when an NVV should occur, where it should be placed, and which type of NVV is appropriate for the conversational context. In this work, we investigate this problem and develop Speech LLMs capable of generating NVVs in accordance with human conversational norms. Specifically, we begin by training a text-based predictor capable of generating NVV tokens using Mandarin real-world speech data. To facilitate spoken interactions, we synthesize natural dialogues related to empathy-oriented interactions and role-play dialogues, and leverage an external LLM to annotate NVV events, with the NVV predictor to assess their appropriateness, and Minimax-HD 2.8 for high-quality speech synthesis. Beyond the conventional two-stage paradigm of speech-to-text and speechto-speech training, we further introduce NVV-aware GRPO to improve the contextual appropriateness of NVV events in generated responses. From two distinct architectures, we develop VocalNet-NV and VITA-NV, and demonstrate that multi-codebook modeling that preserves fine-grained acoustic information is a preferable choice for high-quality speech interactions. Our contributions are summarized as follows:

• We construct a data generation pipeline for conversational speech with contextually appropriate NVVs.

• We propose NVV-aware GRPO, which improves the appropriateness of model responses under substantial distribution discrepancies among NVV events.

• We compare two major speech generation paradigms, revealing the key limitations of single-codebook modeling for natural speech generation.

• Extensive experiments demonstrate substantial improvements in the naturalness and human-likeness of responses from both VocalNet-NV and VITA-NV.

## 2. RELATED WORKS

Speech interaction models accept spoken instructions from users and generate coherent, aligned textual and speech responses. Cascaded pipelines are considered suboptimal due to information loss, error propagation, and cumulative latency. In contrast, Speech LLMs achieve direct speech interactions. Representative Speech LLMs, including LLaMA-Omni and VocalNet, adopt single-stream speech tokens [2, 3]. The LLM backbones autoregressively generate semantic speech tokens and then combine with external tone prompts via flow matching to produce speech responses. Conversely, multicodebook models, represented by Qwen3-Omni and VITA-Qinyu, simultaneously decode all tokens required for speech reconstruction, which encompasses both high-level semantics and fine-grained acoustic details, such as speech features derived from residual vector quantization (RVQ) [7, 4]. Although both approaches enable seamless speech interaction, they generally lack NVV characteristics of human speech, thereby limiting their naturalness and anthropomorphism. JoyAI-Talker incorporates several NVV tokens. However, it has neither explored their placement appropriateness nor investigated the feasibility across model architectures [8].

## 3. METHODS

As shown in Figure 1 (b), regarding the covered NVV types, we comprehensively consider a wide range of open-source datasets and merge or distinguish categories according to their original class definitions [9, 10, 11, 12, 13, 14].

## 3.1. NVV Predictor

Modeling the contextual regularities of NVVs is essential for natural conversational systems. We utilize extensively collected real-world speech and unify the NVV labels to train Qwen3-8B, enabling it to generate NVV tokens alongside standard text. For isolated single-turn instances, we employ DeepSeek-V4-Pro [15] for contextual expansion to provide 2–3 plausible context sentences. The generated NVV predictor is able to distinguish the presence of a natural NVV from random placement. In the synthetic dialogue pipeline, it is employed to assess the plausibility of NVV labels and filter for simulated interactions.

## 3.2. Corpus Construction

For empathy interactions, instances are sourced from LCCC [16]. DeepSeek-V4-Pro is utilized to construct single-turn interactions with distinct affective inclinations based on topics and

user intents. For multi-turn role-play dialogues, our data originates from open-source corpora, retaining solely spoken conversational utterances. DeepSeek-V4-Pro is likewise employed to insert appropriate NVV tokens, and the NVV predictor is subsequently applied to verify their plausibility, yielding natural and pragmatically grounded data.

## 3.3. NVV-Centric Multi-stage Training

## 3.3.1. Speech-to-Text SFT

Given that the vast majority of NVV events are brief and without overlap with regular speech, we extend the model vocabulary with corresponding special tokens and perform supervised fine-tuning (SFT) on synthesized dialogues for text responses. This stage enables the model to generate responses containing NVV events and provides an effective foundation for subsequent NVV-aware reinforcement learning.

## 3.3.2. NVV-aware GRPO

To enhance the model’s comprehension of natural dialogues, we conduct NVV-aware GRPO (Group Relative Policy Optimization). The reward function comprises five key aspects:

$$
\begin{array} { r l } & { \mathcal { R } _ { n v v } = \lambda _ { \mathrm { a p p r o p r i a t e } } \mathcal { R } _ { \mathrm { a p p r o p r i a t e } } + \lambda _ { \mathrm { c o u n t } } \mathcal { R } _ { \mathrm { c o u n t } } } \\ & { \qquad + \lambda _ { \mathrm { c o n s i s t } } \mathcal { R } _ { \mathrm { c o n s i s t } } + \lambda _ { \mathrm { d i s t } } \mathcal { R } _ { \mathrm { d i s t } } + \lambda _ { \mathrm { f o r m a t } } \mathcal { R } _ { \mathrm { f o r m a t } } } \end{array}\tag{1}
$$

where $\mathcal { R } _ { \mathrm { f o r m a t } }$ indicates format reward:

$$
\mathcal { R } _ { \mathrm { f o r m a t } } = \left\{ \begin{array} { l l } { - 1 } & { \mathrm { u n d e f i n e d N V V ~ t o k e n s , u n c l o s e d ~ t a g s , e t c . } } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{2}
$$

Appropriateness reward: For appropriateness, we strengthen responses in which the NV token’s predicted probability by the predictor surpasses alternative NVV tokens within the given context. Specifically, we downweight contextually agnostic tokens (e.g., breath) to ensure a balanced reward. For context c and token t, the score is computed as:

$$
S _ { \mathrm { r e a s o n ( c , t ) } } = \left\{ \begin{array} { l l } { 1 } & { r a n k ( c , t ) = 1 , t \notin \{ b r e a t h , \ldots \} } \\ { 0 . 5 } & { r a n k ( c , t ) = 1 , t \in \{ b r e a t h , \ldots \} } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

The final reasonableness reward is computed as the average score across all NVV events, with an additional bonus awarded to fully plausible responses containing strongly context-dependent tokens.

$$
\begin{array} { r l } { \mathcal { R } _ { \mathrm { r e a s o n } } = \displaystyle \frac { 1 } { | \mathcal { C } | | \mathcal { T } | } \sum _ { c \in \mathcal { C } } \sum _ { t \in \mathcal { T } } S _ { \mathrm { r e a s o n } } ( c , t ) } & { } \\ { \quad \quad } & { + b \mathbb { I } \left[ \displaystyle \operatorname* { m i n } _ { c \in \mathcal { C } , t \in \mathcal { T } } S _ { \mathrm { r e a s o n } } ( c , t ) = 1 \right] . } \end{array}\tag{4}
$$

Count reward: The count reward is designed to penalize the model for aggressively generating excessive relatively safe

![](images/a73c13e804d83b78ffe75f9fac69571b2905e567bde6ec1308f050447057235f.jpg)  
Fig. 1. Method overview. (a) Construction pipeline for speech datasets with NVVs; (b) Covered 13 NVV types; (c) NVVcentric multi-stage training; (d) Reward design for NVV-aware GRPO; (e) Interaction examples in both scenarios.

NVV events, or conversely, outputting too few event tokens to evade potential errors. It is formulated as follows:

$$
\mathcal { R } _ { \mathrm { c o u n t } } = \operatorname* { m a x } \left( 0 , 1 - \frac { \left| n _ { \mathrm { p r e d } } - n _ { \mathrm { r e f } } \right| } { \operatorname* { m a x } ( n _ { \mathrm { r e f } } , 2 ) } \right)\tag{5}
$$

Consistency reward: The consistency reward depends on the distance between predicted and reference events and is calculated using class-level IoU (Intersection over Union). It prioritizes accurate predictions compared to reasonable events.

$$
\mathcal { R } _ { \mathrm { c o n s i s t } } = \frac { \mathrm { p r e d } \cap \mathrm { r e f } } { \mathrm { p r e d } \cup \mathrm { r e f } }\tag{6}
$$

Distribution reward: The distribution reward is to prevent the model from overproducing a particular NVV category and exploiting relatively safe or context-agnostic tokens. Specifically, we measure the discrepancy between the empirical distribution of predicted categories and their natural distribution in the training corpus using the total variation (TV) distance:

$$
\mathrm { T V } ( P _ { \mathrm { p r e d } } , P _ { \mathrm { r e f } } ) = \frac { 1 } { 2 } \sum _ { c } \left| P _ { \mathrm { p r e d } } ( c ) - P _ { \mathrm { r e f } } ( c ) \right| ,\tag{7}
$$

The distribution reward is defined as:

$$
\mathcal { R } _ { \mathrm { d i s t } } = \left\{ \begin{array} { l l } { 1 - \mathrm { T V } ( P _ { \mathrm { p r e d } } , P _ { \mathrm { r e f } } ) } & { \mathrm { i f } n _ { \mathrm { p r e d } } > 0 , } \\ { 0 } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{8}
$$

Using the full training dataset to construct the reference distribution prevents the reward from inheriting category biases introduced by over-sampling in the GRPO training set. Consequently, the reward penalizes excessive concentration on any single NVV category while encouraging the model to preserve a natural category distribution.

## 3.3.3. Speech-to-Speech SFT

This stage trains the model to incorporate fluent NVV events into spoken responses. Given the limited speaker diversity and scale of the synthesized data, we leverage open-source data for text-to-speech training to capture general representations of NVVs [9, 10, 11, 12, 13, 14]. Subsequently, the synthesized dialogue data is used for further training.

## 4. EXPERIMENTS

## 4.1. Non-verbal Token Prediction

We utilize Emilia-NV [13] and nonverbalspeech38k [12] to train Qwen3-8B to generate NVV tokens alongside standard text, as explained in Section 3.1. 200K instances are used for training. The evaluation set comprises 6,034 recordings spanning over 317K token positions with around 7K real NVV occurrences. For each position, the fine-tuned model is employed to predict the presence of an NVV, achieving an AUC-ROC of 0.7635. Furthermore, at real NVV positions, the median predicted probability is 75.0 times higher than non-NVV positions, demonstrating the effective capture of the vocalization distributions in natural conversations. Across the considered NVV categories, the ground-truth NVV label falls within the top-10 predictions with a probability of 92.2%, with more than half ranking within the top-2. We carefully formulate validation rules that reasonably relax probability thresholds for rare label classes. Evaluations on real conversations alongside negatively sampled random insertions demonstrate that the model accepts 88.6% of genuine dialogues while rejecting 91.3% of unnatural insertions.

<table><tr><td rowspan="3">Base Model</td><td rowspan="3">Phase</td><td colspan="4">Empathy Interaction</td><td colspan="4">Role-play Dialogue</td></tr><tr><td colspan="2">Appropriateness</td><td colspan="2">Consistency</td><td colspan="2">Appropriateness</td><td colspan="2">Consistency</td></tr><tr><td>Token (%)</td><td>Response (%)</td><td>Token (%)</td><td>Response (%)</td><td>Token (%)</td><td>Response (%)</td><td>Token (%)</td><td>Response (%)</td></tr><tr><td rowspan="3">VocalNet</td><td>S2T-SFT</td><td>92.5</td><td>85.2</td><td>55.9</td><td>22.4</td><td>97.1</td><td>96.8</td><td>38.5</td><td>25.3</td></tr><tr><td>NVV-GRPO</td><td>96.8</td><td>93.5</td><td>55.2</td><td>21.3</td><td>97.5</td><td>96.9</td><td>34.1</td><td>25.8</td></tr><tr><td>S2S-SFT</td><td>98.0</td><td>96.0</td><td>55.5</td><td>20.2</td><td>97.2</td><td>97.0</td><td>43.8</td><td>32.2</td></tr><tr><td rowspan="3">VITA-Qinyu</td><td>S2T-SFT</td><td>94.6</td><td>89.6</td><td>51.3</td><td>17.2</td><td>95.2</td><td>94.5</td><td>40.9</td><td>30.0</td></tr><tr><td>NVV-GRPO</td><td>95.2</td><td>91.0</td><td>52.5</td><td>17.0</td><td>96.2</td><td>96.0</td><td>44.1</td><td>33.0</td></tr><tr><td>S2S-SFT</td><td>92.2</td><td>84.6</td><td>52.9</td><td>18.1</td><td>95.6</td><td>95.0</td><td>41.5</td><td>27.0</td></tr></table>

Table 1. NVV appropriateness and reference consistency across training stages.

## 4.2. Speech Interaction Performance

## 4.2.1. Training Configuration and Indicators

We conduct training on two speech interaction models, VocalNet and VITA-Qinyu, with the same Qwen3-8B backbone [3, 4]. We first perform S2T-SFT for 3 epochs, followed by GRPO training involving 3K samples. For speech response generation, we conduct 1 epoch of TTS training and 3 epochs of S2S training using the same data as in the S2T stage, with the corresponding speech responses synthesized by Minimax-HD 2.8. For evaluation, we primarily assess the appropriateness of generated NVV events, following the same criteria used for corpus filtering. We additionally consider consistency with the reference annotations as a complementary metric, while acknowledging that a given query may admit multiple natural responses with different NVV labels. For the vocalization quality, 5 human raters are instructed to evaluate in an NV-MOS (mean opinion score) manner on a scale of 1-5. UTMOS [17] is reported as overall speech quality, and we transcribe the responses using Qwen3-ASR-1.7B [18] and calculate the character error rate (CER) for linguistic speech, as the insertion of NVV events may affect the pronunciation of surrounding linguistic content.

## 4.2.2. NV Position Performance

As shown in Table 1, after S2T-SFT, both models acquire the ability to generate NVV events that conform to conversational norms, with more than 85% of complete responses containing only appropriate NVV events. Performance is even stronger in multi-turn role-play dialogue. NVV-aware GRPO yields consistent improvements, particularly for empathy-oriented interactions. For VocalNet, token-level NVV appropriateness increases from 92.5% to 96.8%, while both models achieve response-level appropriateness above 90% on this dataset. Across the four sets of metrics corresponding to the two datasets, we observe consistent improvements, except for fluctuations in consistency with the ground-truth annotations, which mainly result from changes in response styles. S2S-SFT affects the two models differently. VocalNet preserves the capabilities of its backbone by fine-tuning only the speech generator, whereas VITA-Qinyu exposes additional LLM parameters due to its native architecture, resulting in a certain degree of performance degradation.

<table><tr><td rowspan="2">Model</td><td colspan="3">Empathy Interaction</td><td colspan="3">Role-play Dialogue</td></tr><tr><td>NV-MOS</td><td>UTMOS</td><td>CER (%)</td><td>NV-MOS</td><td>UTMOS</td><td>CER (%)</td></tr><tr><td>VocalNet-NV</td><td>3.78</td><td>2.85</td><td>1.20</td><td>3.94</td><td>3.04</td><td>1.78</td></tr><tr><td>VITA-NV</td><td>3.84</td><td>3.54</td><td>1.18</td><td>4.08</td><td>3.64</td><td>1.70</td></tr></table>

Table 2. Speech quality evaluation of model responses.

## 4.2.3. NV Vocalization Quality

VITA-NV consistently outperforms VocalNet-NV across all three evaluation metrics. It achieves an NV-MOS above 4 in role-play, indicating that it generates highly natural and fluent NVV events. VocalNet performs slightly worse, while a more noticeable inferiority is in the overall speech quality. VITA-NV achieves a UTMOS score above 3.5, while VocalNet sustains a score around 3.0. For linguistic pronunciation, both models maintain strong performance, with CER remaining below 2%. In the calculation, we exclude text corresponding to discourse interjections to avoid the influence of NVV transcription on the evaluation metrics. These results suggest that, under the evaluated architectures, single-codebook modeling has limited capacity for representing newly introduced vocal expressions while preserving strong speech quality, whereas multi-codebook reconstruction provides a more effective approach for high-quality speech generation.

## 5. CONCLUSION

Non-verbal vocalizations (NVVs) are an important component of natural and human-like speech interaction. We construct a high-quality data generation pipeline for an NVVcentric speech corpus, incorporate 13 major types of NVVs into Speech LLMs, and equip them with the ability to generate contextually appropriate NVV-aware responses through multi-stage training. In particular, NVV-aware GRPO further improves the appropriateness of NVV events in model outputs while mitigating the tendency to collapse into a limited subset of categories. Both subjective evaluations and objective metrics demonstrate the superior speech quality of VITA-NV, further highlighting the advantages of multi-codebook speech generation for natural and expressive speech interaction. These results suggest that modeling NVVs not only requires appropriate event selection and contextual integration, but also benefits from a speech generation paradigm with sufficient capacity to represent diverse vocal expressions.

## 6. REFERENCES

[1] Judith A Hall, Susan A Andrzejewski, and Jennelle E Yopchick, “Psychosocial correlates of interpersonal sensitivity: A meta-analysis,” Journal of nonverbal behavior, vol. 33, no. 3, pp. 149–180, 2009.

[2] Qingkai Fang, Shoutao Guo, Yan Zhou, Zhengrui Ma, Shaolei Zhang, and Yang Feng, “Llama-omni: Seamless speech interaction with large language models,” in International Conference on Learning Representations, 2025, vol. 2025, pp. 57607–57624.

[3] Yuhao Wang, Heyang Liu, Ziyang Cheng, Ronghua Wu, Qunshan Gu, Yanfeng Wang, and Yu Wang, “Vocalnet: Speech llms with multi-token prediction for faster and high-quality generation,” in Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 19595–19612.

[4] Jiacheng Xu, Heting Gao, Liufei Xie, Zhenchuan Yang, Lijiang Li, Yiting Chen, Bin Zhang, Meng Chen, Chaoyu Fu, Weifeng Zhao, et al., “Vita-qinyu: Expressive spoken language model for role-playing and singing,” arXiv preprint arXiv:2605.06765, 2026.

[5] Zhihao Du, Changfeng Gao, Yuxuan Wang, Fan Yu, Tianyu Zhao, Hao Wang, Xiang Lv, Hui Wang, Chongjia Ni, Xian Shi, et al., “Cosyvoice 3: Towards in-the-wild speech generation via scaling-up and posttraining,” arXiv preprint arXiv:2505.17589, 2025.

[6] Bowen Zhang, Congchao Guo, Geng Yang, Hang Yu, Haozhe Zhang, Heidi Lei, Jialong Mai, Junjie Yan, Kaiyue Yang, Mingqi Yang, et al., “Minimax-speech: Intrinsic zero-shot text-to-speech with a learnable speaker encoder,” arXiv preprint arXiv:2505.07916, 2025.

[7] Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[8] Yinhao Bai, Jinming Chen, Yafeng Chen, Wei Deng, Boya Dong, Nan Duan, Yu Gu, Weisheng Han, Yankun Huang, Ming Ke, et al., “Joyai-talker: Full-duplex speech interactive large model built for empathetic voice agents,” arXiv preprint arXiv:2608.01119, 2026.

[9] Kyra Wang and Dorien Herremans, “Disfluencyspeech– single-speaker conversational speech dataset with paralanguage,” in TENCON 2024-2024 IEEE Region 10 Conference (TENCON). IEEE, 2024, pp. 469–472.

[10] Maksim Borisov, Egor Spirin, and Daria Diatlova, “Nonverbaltts: A public english corpus of text-aligned

nonverbal vocalizations with emotion annotations for text-to-speech,” arXiv preprint arXiv:2507.13155, 2025.

[11] Zhuojun Wu, Dong Liu, Juan Liu, Yechen Wang, Linxi Li, Liwei Jin, Hui Bu, Pengyuan Zhang, and Ming Li, “Smiip-nv: A multi-annotation non-verbal expressive speech corpus in mandarin for llm-based speech synthesis,” in Proceedings of the 33rd ACM International Conference on Multimedia, 2025, pp. 12564–12570.

[12] Runchuan Ye, Yixuan Zhou, Renjie Yu, Zijian Lin, Kehan Li, Xiang Li, Xin Liu, Guoyang Zeng, and Zhiyong Wu, “A scalable pipeline for enabling non-verbal speech generation and understanding,” arXiv preprint arXiv:2508.05385, 2025.

[13] Huan Liao, Qinke Ni, Yuancheng Wang, Yiheng Lu, Haoyue Zhan, Pengyuan Xie, Qiang Zhang, and Zhizheng Wu, “Emilia-nv: A non-verbal speech dataset with word-level annotation for human-like speech modeling,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 17587–17591.

[14] Jialong Mai, Jinxin Ji, Xiaofen Xing, Chen Yang, Weidong Chen, Jingyuan Xing, and Xiangmin Xu, “Mnv-17: A high-quality performative mandarin dataset for nonverbal vocalization recognition in speech,” in ICASSP 2026-2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2026, pp. 18312–18316.

[15] Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al., “Deepseekv4: Towards highly efficient million-token context intelligence,” arXiv preprint arXiv:2606.19348, 2026.

[16] Yida Wang, Pei Ke, Yinhe Zheng, Kaili Huang, Yong Jiang, Xiaoyan Zhu, and Minlie Huang, “A large-scale chinese short-text conversation dataset,” in CCF International Conference on Natural Language Processing and Chinese Computing. Springer, 2020, pp. 91–103.

[17] Takaaki Saeki, Detai Xin, Wataru Nakata, Tomoki Koriyama, Shinnosuke Takamichi, and Hiroshi Saruwatari, “Utmos: Utokyo-sarulab system for voicemos challenge 2022,” arXiv preprint arXiv:2204.02152, 2022.

[18] Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, et al., “Qwen3-asr technical report,” arXiv preprint arXiv:2601.21337, 2026.