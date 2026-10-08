# Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation

Shunsuke Mitsumori<sup>∗†</sup>, Tomoya Mizumoto<sup>∗</sup>, Yusuke Fujita<sup>∗</sup>

<sup>∗</sup>SB Intuitions, <sup>†</sup>Waseda University, Tokyo, Japan

shunsuke.mitsumori@sbintuitions.co.jp

Abstract—Speech Language Model (SLM) performance often degrades on dialects due to data scarcity. Conventional text-tospeech (TTS) augmentation struggles to cover diverse dialects as it requires a certain amount of real dialect speech. We propose synthesizing pseudo-dialect speech by converting LLMgenerated dialect text via a standard-language TTS model, requiring zero real dialect speech. Additionally, we introduce intermediate standard-text prediction during training, acting as semantic normalization for downstream tasks. We evaluate dialect understanding via dialect-to-English speech translation across Japanese, German, and Chinese dialects. Compared to synthetic standard speech baselines, pseudo-dialect augmentation improves scores for Japanese (from 25.38 to 26.24) and German (from 31.57 to 32.47). Furthermore, the intermediate standardtext prediction effectively bridges the semantic gap, boosting performance to 28.26 for Japanese and from 11.67 to 16.37 for Chinese. These results suggest that our approach scales to various languages without requiring speech resources specific to each dialect.

Index Terms—Speech Language Models, Dialect Adaptation, Data Augmentation, Synthetic Data

## I. INTRODUCTION

Dialect speech possesses unique linguistic features (e.g., distinct vocabulary and grammar) and acoustic features that differ significantly from the standard language [1], [2]. Because of these discrepancies, conventional speech architectures have historically struggled with dialect robustness [3]–[5]. Even recent advancements in Speech Language Model (SLM) [6], [7], which leverage LLMs to achieve remarkable success in tasks such as speech recognition, translation, and dialogue [8]– [11], face severe performance degradation when encountering dialects [12], [13]. The degradation is primarily attributed to the lack of real dialect speech data for model training.

In the field of low-resource languages, data augmentation approaches using speech synthesis have been adopted to address such data scarcity. The primary approach is to convert text obtained via LLM or web collection into speech with a Text-to-Speech (TTS) model trained on real speech of the target language [14], [15]. However, this augmentation strategy cannot cover various dialects because it requires a sufficient amount of real dialect speech for training TTS models.

To expand dialect coverage without relying on real dialect audio, we focus on the potential of reusing standard-language speech resources. Since dialects are varieties of a standard language rather than completely distinct languages [1], they inherently share overlapping acoustic features. Leveraging this acoustic similarity, we propose a novel dialect adaptation framework that bypasses the need for dialect-specific TTS models by utilizing “pseudo-dialect” speech. Specifically, we first employ an LLM to rewrite standard text into text with dialectal linguistic features. Then, we utilize a standard-language TTS model to convert this augmented text into pseudo-dialect speech. Since this framework requires zero real dialect speech resources, it enables data augmentation for arbitrary dialects.

Furthermore, leveraging the property that pseudo-dialect speech inherently pairs with standard language text during its generation, we introduce multi-task learning with intermediate standard-text prediction. We exploit this standard text as an explicit intermediate output before predicting the final target task. By projecting the input into the standard language, we force the model to map the dialectal audio into a rich semantic space where the backend LLM can perform strong reasoning.

We verified the general effectiveness of the proposed framework through two main perspectives. To measure dialect understanding capabilities of SLM, we followed previous work [12] and evaluated the models using a dialect-to-English translation task. First, to test its effectiveness across languages, we evaluated SLMs across Japanese, German, and Chinese dialects. Second, to test architectural generalizability of our pseudo-dialect speech augmentation, we applied our augmentation to conventional end-to-end models (Whisper) and cascade models (Whisper+LLM) using Japanese data.

Experimental results showed that, compared to a synthetic standard speech baseline, the pseudo-dialect augmentation improved English translation BLEU scores for Japanese (from 25.38 to 26.24) and German (from 31.57 to 32.47). Furthermore, intermediate standard-text prediction boosted performance for Japanese (to 28.26) and Chinese (from 11.67 to 16.37). Regarding architectural generalizability, the pseudodialect speech augmentation also benefited the direct Whisper model (from 18.44 to 18.98) and the cascade system (from 29.65 to 30.44) over their respective synthetic standard speech baselines.

## II. PROPOSED METHOD

In this section, we formulate the proposed dialect adaptation framework. Our target task is to predict a translation text y from an input speech signal x by modeling p(y|x). In practice, directly training p(y|x) is challenging because paired data {(x,y)} are scarce. To obtain pseudo translation labels, we start from an automatic speech recognition (ASR) corpus of standard speech-text pairs $\{ ( x _ { s } , t _ { s } ) \}$ and apply a text translation model $p _ { \theta } ( y | t _ { s } )$ to the transcripts:

![](images/345f0635f1cfa9b1ce8c7209ad611ef6f406067b97310daddf759eebea7c336d.jpg)  
Fig. 1. Proposed pseudo-dialect speech augmentation. From a standard pair $\{ x _ { s } , t _ { s } \}$ and translation $y _ { s } ,$ an LLM and standard TTS model generate pseudo-dialect text $\tilde { t } _ { d }$ and speech $\tilde { x } _ { d } .$ This yields $\{ \widetilde { x } _ { d } , y _ { s } \}$ and $\{ \widetilde { x } _ { d } , t _ { s } , y _ { s } \}$ for single-task and multi-task SLM training, respectively.

$$
y _ { s } \sim p _ { \theta } ( y | t _ { s } ) .\tag{1}
$$

This yields pseudo-labeled speech translation data $\left\{ \left( x _ { s } , y _ { s } \right) \right\}$ However, the available ASR corpus mostly consists of standard speech $x _ { s } ,$ and the lack of dialect speech $x _ { d }$ is a fundamental bottleneck. To address this, we propose a dialect adaptation framework using pseudo-dialect speech augmentation (Figure 1) and multi-task learning with intermediate standard-text prediction.

## A. Pseudo-Dialect Speech Augmentation

To mitigate the shortage of dialect speech, we synthesize pseudo-dialect training examples without relying on real dialect speech. Given a standard text $t _ { s } ,$ we first generate a pseudo-dialect text $\tilde { t } _ { d }$ using an LLM-based dialect generation model $p _ { \phi } \big ( \tilde { t } _ { d } \big | t _ { s } \big )$

$$
\tilde { t } _ { d } \sim p _ { \phi } ( \tilde { t } _ { d } | t _ { s } ) .\tag{2}
$$

Next, we synthesize pseudo-dialect speech $\tilde { x } _ { d }$ from $\tilde { t } _ { d }$ using a standard TTS model $p _ { \psi } ( x | t )$ :

$$
\begin{array} { r } { \tilde { x } _ { d } \sim p _ { \psi } ( x | \tilde { t } _ { d } ) . } \end{array}\tag{3}
$$

Finally, we pair the synthesized speech $\tilde { x } _ { d }$ with the pseudo translation label $y _ { s }$ to obtain $\left\{ \left( \tilde { x } _ { d } , y _ { s } \right) \right\}$ for SLM training. Note that the TTS model $p _ { \psi } ( x | t )$ is trained on standard speech-text pairs, and thus $\tilde { x } _ { d }$ is typically perceived as non-native-sounding dialect speech. Nevertheless, because $\tilde { t } _ { d }$ contains dialectal lexical and syntactic cues, $\tilde { x } _ { d }$ provides useful supervision for dialect robustness.

## B. Intermediate Standard-Text Prediction for SLM

With the proposed augmentation, each training sample forms a triplet $\left( \tilde { x } _ { d } , t _ { s } , y _ { s } \right)$ . To effectively translate dialect speech $x _ { d } ,$ we exploit the standard text $t _ { s }$ as an explicit intermediate representation. By projecting the input into $t _ { s } ,$ we map the audio into a rich semantic space where the underlying LLM possesses the strongest semantic reasoning capabilities. This process acts as a semantic denoising step, effectively bridging the dialectal input and the target translation. We factorize the speech translation model as:

$$
p ( y | x ) = \sum _ { t _ { s } } p ( y | t _ { s } , x ) p ( t _ { s } | x ) .\tag{4}
$$

In practice, we approximate the marginalization over $t _ { s }$ by decoding the most probable sequence. This yields a chain-ofthought process that predicts the standard text $t _ { s }$ as a semantic bridge followed by the final translation $y .$ During training, the model is jointly optimized using both the pseudo-dialect data $\left( \tilde { x } _ { d } , t _ { s } , y _ { s } \right)$ and the original standard data $\left( x _ { s } , t _ { s } , y _ { s } \right)$

## III. EXPERIMENTAL SETUP

We verify the effectiveness of the pseudo-dialect speech augmentation and intermediate standard-text prediction, as proposed in Section II, within the SLM framework. Because the linguistic relationship between regional dialects and the standard language varies across different languages, we conduct our experiments on Japanese, Chinese, and German to evaluate the general effectiveness of our approach.

## A. Evaluation Tasks

To evaluate dialect understanding, we focus on the English translation task taking real dialect speech as input. This task verifies whether the model truly understands the underlying semantics of the dialect speech to predict the final translation (y) [12]. To evaluate the English translation performance, BLEU [16] scores were computed using SacreBLEU [17].

## B. Model and Training Setup

In this study, we constructed SLMs connecting a speech encoder and an LLM with a projector. Across all languages, we adopted Whisper large-v3 [18] as the speech encoder. For the LLM backend, we utilized Llama-3.1-Swallow-8B-Instructv0.3 [19] for Japanese, and Llama-3.1-8B-Instruct [20] for German and Chinese. The projector consists of two 1- dimensional convolution layers (kernel size 4, stride 2) with GELU activations and a final linear layer. Following previous studies on SLMs [21]–[24], the parameters of the speech encoder and the LLM were frozen during training, while only the projector parameters were updated.

TABLE I  
OUTPUT FORMATS FOR EACH MULTI-TASK LEARNING CONFIGURATION. VARIABLES FOLLOW THE NOTATION INTRODUCED IN SECTION 2, WHERE THE TRANSCRIPTION TARGET $( \tilde { t } _ { d }$ OR t ) DEPENDS ON THE INPUT SPEECH.
<table><tr><td>Output Targets</td><td>Transcription Standard text Translation text  $( \tilde { t } _ { d } \ \mathrm { o r } \ t _ { s } )$ </td><td> $\left( t _ { S } \right)$ </td><td>(y)</td><td>Output Format</td></tr><tr><td>ASR+English</td><td>√</td><td></td><td>√</td><td>&lt;ASR&gt;td or ts&lt;/ASR&gt;&lt;en&gt;y&lt;/en&gt;</td></tr><tr><td>Standard+English</td><td></td><td>√</td><td>√</td><td>&lt;standard&gt;ts&lt;/standard&gt;&lt;en&gt;y&lt;/en&gt;</td></tr><tr><td>ASR+Standard+English</td><td>√</td><td>√</td><td>√</td><td> ${ < \tt A S R > } \tilde { t } _ { d }$  or ts&lt;/ASR&gt;&lt;standard&gt;ts&lt;/standard&gt;&lt;en&gt;y&lt;/en&gt;</td></tr></table>

All SLM models were trained on 8 NVIDIA H100 (80GB) GPUs for 10 epochs using the AdamW optimizer. The effective batch size was set to 512, and the learning rate peaked at 1e-4. The SLM training times varied depending on the dataset sizes of each language. For Japanese, training took approximately 2 days without pseudo-dialect speech augmentation, and 4 days with the augmentation. For Chinese, it required slightly over 1 day without augmentation and approximately 2.5 days with it. For German, training took approximately 6 hours and 12 hours, respectively.

## C. Datasets

1) Evaluation Datasets: To evaluate dialect understanding, we utilized corpora containing real speech from regional dialects. For Japanese, we used CPJD [25] comprising real speech from 20 dialects. For German, we used SwissDial [26] consisting of 8 dialects. For Chinese, we utilized KeSpeech [27] containing 8 dialect subsets. To ensure accurate evaluation across all three languages, we generated high-quality English reference labels from their respective standard language parallel texts using GPT-4o. These real dialect speeches were strictly excluded from the training data.

2) Training Datasets: Depending on the experimental configurations, models for each language were trained using real standard language resources, either alone, augmented with synthesized pseudo-dialect speech, or augmented with synthetic standard speech (which is included for comparison purposes as detailed in Section III-D1).

For Japanese, we utilized ReazonSpeech large v2 comprising approximately 2.6M utterances [28], employing its transcripts as standard Japanese labels and as the source text for the data augmentation pipeline. Additionally, we included real standard speech from Speech-BSD (20,000 utterances) [29] and the Japanese subset of CoVoST2 (1,119 utterances) [30], along with their human-annotated Japanese transcripts and English translations.

For German, we used the German subset of Multilingual LibriSpeech [31] comprising approximately 0.4M utterances. For Chinese, we utilized a subset of 3M utterances from WenetSpeech [32]. These datasets served as standard speech training data and the source texts for the data augmentation pipeline.

Because ReazonSpeech, the German subset of Multilingual LibriSpeech, and WenetSpeech lack English translations, we generated their English labels using Qwen2.5-32B-Instruct [33]. We chose a different model from the LLM used to generate the English references for evaluation (GPT-4o) to prevent overfitting to the specific stylistic biases of a single machine translation system, following previous work [12].

## D. Experimental Configurations

1) Data Augmentation Configurations: We compare three training data settings to evaluate the proposed pseudo-dialect speech. All models in this comparison are trained as standard single-task models.

No Augmentation: Uses only the real speech data from Section III-C, serving as a baseline without synthetic data. This original training set consists of approximately 2.6M utterances for Japanese, 0.4M for German, and 3.0M for Chinese.

Synthetic standard speech: Adds synthesized standard utterances generated from the respective standard source texts described in Section III-C. By including a 1:1 match of synthesized utterances, this configuration doubles the training volume to a total of 5.2M for Japanese, 0.8M for German, and 6.0M for Chinese. For Japanese, we used Tsukasa-Speech [34] with reference speech from the JVS corpus [35]. For German and Chinese, Qwen-TTS [33] was utilized to convert the standard transcripts using the original standard speech as reference. This setup isolates the effect of pure data volume, verifying that performance gains come specifically from dialectal linguistic features.

Pseudo-dialect speech: Adds the proposed pseudo-dialect speech to the baseline dataset. The standard source transcripts for each language were translated into one of the target regional dialects chosen at random (20 dialects for Japanese, 8 for German, and 8 for Chinese). This dialectal rewriting was performed via gpt-oss-120b [36] across all languages. The rewritten texts were then synthesized into speech using the identical TTS setup as the synthetic standard speech for each language. To ensure a fair comparison, this configuration maintains the exact same total data volume as the synthetic standard speech setting.

2) Multi-task Learning Configurations: To verify the effectiveness of multi-task learning with intermediate standard-text prediction for the SLM, we trained models on the full dataset augmented with pseudo-dialect speech in Section III-D1. The designated target output formats for these multi-task configurations are defined in Table I. By generating this concatenated sequence, the joint probability formulated in Eq. 4 is implicitly optimized through the standard autoregressive nexttoken prediction (cross-entropy) loss. During evaluation of the multi-task models, the final English translation was obtained by extracting the text enclosed within the <en> and </en> tags using regular expressions.

TABLE II  
COMPARISON OF DIALECT UNDERSTANDING CAPABILITIES EVALUATED VIA DIALECT-TO-ENGLISH SPEECH TRANSLATION PERFORMANCE (BLEU) ACROSS MULTIPLE LANGUAGES, DATA AUGMENTATION SETTINGS, AND TASK OUTPUT FORMATS. FOR THE OUTPUT TARGETS, “ENGLISH” DENOTES DIRECT TRANSLATION, WHILE “ASR" AND “STANDARD" REPRESENT INTERMEDIATE SPEECH TRANSCRIPTION AND STANDARD-TEXT PREDICTION RESPECTIVELY.
<table><tr><td>Synthetic data added</td><td>Output Target</td><td>Japanese</td><td>German</td><td>Chinese</td></tr><tr><td>No Augmentation (baseline)</td><td>English</td><td>24.21</td><td>22.37</td><td>11.61</td></tr><tr><td>Synthetic standard speech</td><td>English</td><td>25.38</td><td>31.57</td><td>11.72</td></tr><tr><td>Pseudo-dialect speech</td><td>English</td><td>26.24</td><td>32.47</td><td>11.67</td></tr><tr><td>Pseudo-dialect speech</td><td>ASR+English</td><td>27.67</td><td>29.81</td><td>16.30</td></tr><tr><td>Pseudo-dialect speech</td><td>Standard+English</td><td>28.26</td><td>30.73</td><td>16.37</td></tr><tr><td>Pseudo-dialect speech</td><td>ASR+Standard+English</td><td>24.95</td><td>29.92</td><td>13.24</td></tr></table>

TABLE III

COMPARISON OF ENGLISH TRANSLATION PERFORMANCE (BLEU) ACROSS DIFFERENT TYPES OF DATA AUGMENTATION, SHOWING THE AVERAGE SCORE AND INDIVIDUAL RESULTS FOR 20 JAPANESE DIALECTS.
<table><tr><td>Synthetic data added</td><td>Avg.</td><td>Akita</td><td>Hiroshima</td><td>Iyo</td><td>Fukuoka</td><td>Awa</td><td>Hokkaido</td><td>Fukui</td><td>Kyoto</td><td>Miyazaki</td><td>Enshu</td></tr><tr><td>No Augmentation (baseline)</td><td>24.21</td><td>19.10</td><td>25.97</td><td>27.49</td><td>26.72</td><td>25.34</td><td>30.19</td><td>27.69</td><td>24.35</td><td>21.73</td><td>23.83</td></tr><tr><td>Synthetic standard speech</td><td>25.38</td><td>19.90</td><td>26.67</td><td>27.58</td><td>28.46</td><td>26.29</td><td>30.17</td><td>29.57</td><td>25.97</td><td>23.23</td><td>24.70</td></tr><tr><td>Pseudo-dialect speech</td><td>26.24</td><td>20.12</td><td>27.39</td><td>29.16</td><td>30.39</td><td>27.23</td><td>30.78</td><td>30.13</td><td>27.14</td><td>24.98</td><td>25.22</td></tr></table>

<table><tr><td>Synthetic data added</td><td>Avg.</td><td>Okayama</td><td>Saitama</td><td>Tosa</td><td>Morokata</td><td>Tsugaru</td><td>Iwaki</td><td>Izumo</td><td>Kanazawa</td><td>Nara</td><td>Osaka</td></tr><tr><td>No Augmentation (baseline)</td><td>24.21</td><td>24.83</td><td>28.66</td><td>25.36</td><td>10.60</td><td>14.30</td><td>27.75</td><td>22.55</td><td>21.68</td><td>26.67</td><td>28.19</td></tr><tr><td>Synthetic standard speech</td><td>25.38</td><td>25.96</td><td>31.03</td><td>26.78</td><td>10.12</td><td>14.65</td><td>30.12</td><td>24.92</td><td>20.04</td><td>28.81</td><td>29.58</td></tr><tr><td>Pseudo-dialect speech</td><td>26.24</td><td>28.97</td><td>31.98</td><td>28.14</td><td>12.41</td><td>16.70</td><td>30.05</td><td>23.28</td><td>21.45</td><td>28.40</td><td>29.45</td></tr></table>

TABLE IV

IMPACT OF INTERMEDIATE PREDICTION ON ENGLISH TRANSLATION (BLEU), SHOWING THE AVERAGE SCORE AND INDIVIDUAL RESULTS FOR 20 JAPANESE DIALECTS. ALL MODELS WERE TRAINED ON THE DATASET AUGMENTED WITH PSEUDO-DIALECT SPEECH.
<table><tr><td>Output Targets</td><td>Avg.</td><td>Akita</td><td>Hiroshima</td><td>Iyo</td><td>Fukuoka</td><td>Awa</td><td>Hokkaido</td><td>Fukui</td><td>Kyoto</td><td>Miyazaki</td><td>Enshu</td></tr><tr><td>English (Single-task)</td><td>26.24</td><td>20.12</td><td>27.39</td><td>29.16</td><td>30.39</td><td>27.23</td><td>30.78</td><td>30.13</td><td>27.14</td><td>24.98</td><td>25.22</td></tr><tr><td>ASR+English</td><td>27.67</td><td>23.26</td><td>30.24</td><td>31.27</td><td>33.50</td><td>28.58</td><td>30.96</td><td>31.51</td><td>27.94</td><td>26.96</td><td>26.29</td></tr><tr><td>Standard+English</td><td>28.26</td><td>22.20</td><td>32.53</td><td>31.68</td><td>32.25</td><td>29.72</td><td>31.21</td><td>30.27</td><td>29.88</td><td>26.93</td><td>26.39</td></tr><tr><td>ASR+Standard+English</td><td>24.95</td><td>19.23</td><td>25.12</td><td>25.93</td><td>29.18</td><td>27.13</td><td>27.45</td><td>29.42</td><td>26.46</td><td>22.22</td><td>25.36</td></tr></table>

<table><tr><td>Output Targets</td><td>Avg.</td><td>Okayama</td><td>Saitama</td><td>Tosa</td><td>Morokata</td><td>Tsugaru</td><td>Iwaki</td><td>Izumo</td><td>Kanazawa</td><td>Nara</td><td>Osaka</td></tr><tr><td>English (Single-task)</td><td>26.24</td><td>28.97</td><td>31.98</td><td>28.14</td><td>12.41</td><td>16.70</td><td>30.05</td><td>23.28</td><td>21.45</td><td>28.40</td><td>29.45</td></tr><tr><td>ASR+English</td><td>27.67</td><td>29.59</td><td>34.09</td><td>28.32</td><td>11.48</td><td>17.65</td><td>30.12</td><td>24.83</td><td>23.63</td><td>30.38</td><td>30.95</td></tr><tr><td>Standard+English</td><td>28.26</td><td>29.79</td><td>34.00</td><td>29.22</td><td>12.34</td><td>20.43</td><td>31.20</td><td>27.25</td><td>24.63</td><td>30.62</td><td>30.85</td></tr><tr><td>ASR+Standard+English</td><td>24.95</td><td>25.05</td><td>29.04</td><td>25.45</td><td>12.54</td><td>16.68</td><td>30.12</td><td>22.53</td><td>21.96</td><td>27.45</td><td>28.45</td></tr></table>

## IV. EXPERIMENTAL RESULTS

Table II presents the overall dialect understanding performance evaluated via English translation BLEU on the real dialect speech from Section III-C1, across multiple languages under the various data augmentation settings and multi-task learning configurations described in Section III-D1. The results indicate that our framework successfully improved the dialect understanding performance across all target languages. However, whether pseudo-dialect speech augmentation or multitask learning primarily drove these improvements highly depended on the linguistic properties of each language. To rigorously evaluate these performance differences, we conducted paired bootstrap resampling tests [37] with 1,000 iterations $( p < 0 . 0 5 )$ at the overall language or individual dialect levels. Note that Japanese dialect subsets have small sample sizes (∼250 utterances). Thus, non-significant results may stem from low statistical power rather than an absence of effect.

While Japanese demonstrated substantial improvements from both approaches, elevating the performance from the baseline of 24.21 to a maximum score of 28.26, German and Chinese exhibited contrasting behaviors regarding which method was effective. For German, data augmentation with pseudo-dialect speech successfully improved the dialect understanding performance compared to the augmentation with synthetic standard speech (+0.90 BLEU, $p < 0 . 0 0 1 )$ . Conversely, multi-task learning was not effective for German because direct translation is already effective. Forcing an intermediate text output likely caused error propagation rather than serving as a semantic bridge. In contrast, for Chinese, pseudo-dialect speech augmentation showed no noticeable effect. However, multi-task learning with intermediate standard Mandarin prediction improved the dialect understanding performance from 11.67 to 16.37 (+4.70 BLEU). Chinese dialects share an ideogram-based written system with the standard language. Therefore, they share a strong grammatical and lexical foundation despite different pronunciations. Mapping the speech input to this standard text space thus likely functioned as an effective semantic bridge for the LLM backend. Conversely, this shared text does not alter its character forms to reflect phonetic variations. Unlike phonetic languages, reproducing dialect-specific readings using a standard TTS model is uniquely challenging. Consequently, the TTS model failed to synthesize distinct dialectal speech, potentially limiting the benefit of audio augmentation. Furthermore, compared to the Standard+English setting, the ASR+Standard+English setting degraded performance across all three languages. This decline is likely attributed to the amplification of error propagation caused by passing through two intermediate predictions.

TABLE V  
QUALITATIVE ANALYSIS OF TRANSLATION OUTPUTS FROM THE ENGLISH TRANSLATION TASK. ASTERISK (\*) AND DAGGER (†) INDICATE FAILURE CASES DUE TO UNSEEN VOCABULARY AND ACOUSTIC GAPS.
<table><tr><td>Category</td><td>Input Audio (Dialect)</td><td>English Ground Truth</td><td>Baseline Output</td><td>Proposed Output</td></tr><tr><td>Improved 1 (Kyoto)</td><td>ほんで食材も たりひんかったんどす.</td><td>And we also ran out of ingredients.</td><td>And I also bought some ingredients.</td><td>And we also ran out of ingredients.</td></tr><tr><td>Improved 2 (Iwaki)</td><td>きっと素敵な休日に なっぺよ.</td><td>I’m sure it will be a wonderful holiday.</td><td>I hope you have a wonderful holiday.</td><td>I&#x27;m sure it will be a wonderful holiday.</td></tr><tr><td>Failure 1 (Kanazawa)*</td><td>がめるものがめるもの, 古いものが多いがやて.</td><td>The things I pick up tend to be old.</td><td>There are a lot of old things here.</td><td>There are a lot of old things here.</td></tr><tr><td>Failure 2 (Tsugaru)†</td><td>赤のスマホでも いいがもな.</td><td>A red smartphone might be nice, too.</td><td>I might as well use my smartphone.</td><td>Maybe it&#x27;s okay if it&#x27;s an iPhone.</td></tr><tr><td colspan="5">TABLE VI COMPARISON OF ENGLISH TRANSLATION OUTPUTS ACROSS MULTI-TASK CONFIGURATIONS. ALL MODELS WERE TRAINED ON THE IDENTICAL DATASET</td></tr><tr><td colspan="5">AUGMENTED WITH PSEUDO-DIALECT SPEECH, DIFFERING ONLY IN THEIR TARGET OUTPUT FORMATS. Output Content</td></tr><tr><td colspan="5">Output Target Ground Truth ASR:必要ねえのにスマートフォンばもってる理由は、緊急連絡用に、親がわさけだはんで。</td></tr><tr><td></td><td colspan="4">Standard：必要がないのにスマートフォンをもっている理由は、緊急連絡用にと親が私に与えたから。 English: The reason I have a smartphone even though I don&#x27;t need one is because my parents gave it to me for emergency contact purposes.</td></tr><tr><td>English (Single-task)</td><td colspan="4">English: The reason I have a smartphone even though I don&#x27;t need one is because my parents insisted, saying it&#x27;s for emergencies. ASR：必要ねえのにスマートフォンば持ってる理由は、緊急連絡用に、親が話さけだはんで。</td></tr><tr><td>ASR+English</td><td colspan="4">English: The reason for carrying a smartphone even though it&#x27;s not necessary is because it&#x27;s for emergency contact, and parents can&#x27;t talk.</td></tr><tr><td>Standard+English</td><td colspan="4">Standard：必要ないのにスマートフォンを持っている理由は緊急連絡用に親が買ってくれたから。 English: The reason I have a smartphone even though I don&#x27;t need one is because my parents bought it for me as an emergency contact.</td></tr></table>

We next examine the detailed performance across the 20 Japanese dialects. As shown in Table III, pseudo-dialect augmentation improved the average BLEU score to 26.24, significantly outperforming synthetic standard speech (25.38, $p < 0 . 0 0 1 $ . It brought gains over both the no-augmentation and synthetic standard speech baselines in 15 dialects (with 4 out of 20 dialects showing statistical significance, $p < 0 . 0 5 )$ .

Table IV demonstrates that incorporating the standard Japanese intermediate target (Standard+English) achieved further performance improvements, significantly outperforming the ASR+English target $( p < 0 . 0 5 )$ . Compared to the direct English translation, this configuration improved performance in 19 out of 20 dialects (with 9 dialects showing statistically significant improvements, $p < 0 . 0 5 )$ , including dialects where audio augmentation alone showed no gains. In contrast, the ASR target (ASR+English) was less stable, showing larger performance fluctuations in specific dialects like Hiroshima, Izumo, and Tsugaru. This instability underscores that mapping to a standard text space provides a more robust semantic bridge than literal transcriptions.

## V. ANALYSIS

This section provides a qualitative analysis to examine how our method improves dialect understanding and to identify remaining challenges. This evaluation is conducted using Japanese examples from Table V and Table VI, which showed clear performance improvements across both data augmentation and output format configurations. Through these cases, we demonstrate the specific advantages and limitations of pseudodialect speech augmentation, as well as the benefits of multitask learning.

## A. Improvement by Pseudo-Dialect Speech

As shown in the improvement examples, the proposed method is able to learn grammar and vocabulary unique to dialects. Specifically, the baseline model misrecognized the dialectal phrase “たりひんかった(tarihinkatta)” as “katta” that means “bought” in Improved 1. Similarly, in Improved 2, it misrecognized the dialectal ending “nappeyo” (meaning “will probably become”) as the phonetically similar standard expression “natteyo” (which expresses a desire or request). In contrast, the proposed method correctly recognized the dialect expressions in both examples. These results support that through learning with pseudo-dialect speech that has dialectal linguistic features even with standard language acoustic features, SLM can acquire knowledge regarding the grammar and vocabulary structure of dialects.

## B. Challenges of Pseudo-Dialect Speech

On the other hand, the failure examples highlight two primary challenges. First is uncovered dialect vocabulary due to limitations in LLM translation quality. In failure example 1, both models failed to recognize the dialect term $\cdots \nmid 0 ^ { s } \nmid 0 )$ る(gameru)” that means “pick $\boldsymbol { \mathbf { u } \mathbf { p } } ^ { \prime \prime }$ . Investigation of the training data revealed that all instances that should have been translated to “がめる(gameru)” were missed by the LLM. In failure example 2, both methods failed to recognize the standard word “<sup>赤</sup>(aka)” that means “red” due to the Tsugaru dialect accent. This indicates that the method has limitations in adapting to acoustic variations unique to dialects, since the pseudo-dialect speech is synthesized with standard language acoustic features.

## C. Improvement by Multi-task Learning

Table VI compares the outputs of the different models given a Tsugaru dialect input. The single-task model mistranslates the dialectal phrase “わ さ け だ(wasakeda)”, which means “someone gave it to me.” Furthermore, in the model jointly outputting transcriptions, the phrase is misrecognized as “話 $\partial ^ { \prime } + \varepsilon , 3$ which is not an existing word sequence but could be pronounced similarly, and this error propagates directly to the English translation. In contrast, the model jointly outputting standard Japanese successfully captures the core semantic meaning of parental giving, translating the phrase as “bought it for me.” This configuration yields the most contextually accurate English output among the compared models. This indicates that standard Japanese serves as an effective intermediate representation that bridges the semantic gap between dialects and English.

## VI. EFFECTIVENESS ACROSS DIFFERENT MODEL ARCHITECTURES

To verify whether our proposed pseudo-dialect speech augmentation is effective beyond the SLM architecture, we also examine its impact on alternative frameworks, specifically direct English translation using Whisper, and a cascade speech translation system combining Whisper and Swallow. For these models, Whisper and Swallow are trained independently. Unlike the frozen configuration of the SLM, Whisper undergoes full-parameter fine-tuning with 1,543.5M total and trainable parameters, whereas Swallow is efficiently tuned using LoRA [38] with 41.9M trainable parameters out of 8,072.2M total parameters. Consequently, the cascade system comprises 9,615.7M total parameters, with 1,585.4M trainable parameters. Due to these significant differences in trainable parameters, our evaluation focuses on the effectiveness of pseudo-dialect augmentation within each respective architecture rather than comparing absolute performance across them.

TABLE VII  
EFFECTIVENESS OF PSEUDO-DIALECT AUGMENTATION ACROSS ARCHITECTURES IN ENGLISH TRANSLATION (BLEU). “NO TRAINING” MEANS NO FINE-TUNING. “+ STANDARD” AND “+ PSEUDO-DIALECT” ADD SYNTHETIC STANDARD AND PSEUDO-DIALECT SPEECH. SLM SCORES MATCH AVERAGES IN TABLE III.
<table><tr><td>Dataset</td><td>SLM</td><td>Whisper</td><td>Cascade</td></tr><tr><td># Total params</td><td>8,683.7M</td><td>1,543.5M</td><td>9,615.7M</td></tr><tr><td># Trainable params</td><td>18.4M</td><td>1,543.5M</td><td>1,585.4M</td></tr><tr><td>No training</td><td></td><td>14.36</td><td>19.87</td></tr><tr><td>No Augmentation</td><td>24.21</td><td>17.78</td><td>29.78</td></tr><tr><td>+ Standard</td><td>25.38</td><td>18.44</td><td>29.65</td></tr><tr><td>+ Pseudo-dialect</td><td>26.24</td><td>18.98</td><td>30.44</td></tr></table>

All models were trained on 8 NVIDIA A100 (80GB) GPUs for 10 epochs using the AdamW optimizer. For the individual training of Whisper and Swallow, the effective batch size was 256, and the peak learning rate was 1e-5. Training Whisper took about 1.5 days without augmentation and 3 days with it, while Swallow required about 1 day and 2 days, respectively.

The results of the English translation task using the SLM, Whisper, and the cascade system are presented in Table VII. First, although the cascade system outperforms the SLM, this is primarily because both Whisper and Swallow underwent fine-tuning. Therefore, this result does not necessarily indicate the architectural superiority of one model over the other. Notably, not only for the SLM but also for both direct English translation using Whisper and the cascade translation system, applying the proposed pseudo-dialect speech augmentation improved translation performance compared to training with synthetic standard speech (+0.54 and +0.79 for Whisper and the cascade system, respectively, $p < 0 . 0 0 1 )$ . This demonstrates that, even after isolating the effect of simply increasing the data volume, the proposed pseudo-dialect speech augmentation is generally effective across various architectures.

## VII. CONCLUSION

In this study, we proposed dialect adaptation for SLM using pseudo-dialect speech synthesized without real dialect speech, combined with multi-task learning leveraging the corresponding standard language text. Experimental results demonstrated that our method improves dialect understanding performance across multiple languages and alternative model architectures. This serves as a practical and scalable strategy to expand the dialect coverage of SLM, eliminating the need for extensive dialect speech collection.

The qualitative analysis in Section V illustrated how the proposed method improves dialect understanding, while also highlighting the importance of LLM dialect translation quality and cases where the TTS system needs to capture dialectal accents. Therefore, establishing evaluation protocols for lowresource dialect text translation, as well as developing advanced speech synthesis technologies that replicate realistic dialectal features under low-resource constraints, remain key directions for future work.

[1] J. K. Chambers and P. Trudgill, Dialect and language, ser. Cambridge Textbooks in Linguistics. Cambridge University Press, 1998, p. 3–12.

[2] M. Zampieri and P. Nakov, Eds., Similar Languages, Varieties, and Dialects: A Computational Perspective, ser. Studies in Natural Language Processing. Cambridge University Press, 2021.

[3] S. Miwa and A. Kai, “Dialect Speech Recognition Modeling using Corpus of Japanese Dialects and Self-Supervised Learning-based Model XLSR,” in Interspeech 2023, 2023, pp. 4928–4932.

[4] B. Li, T. N. Sainath, K. C. Sim, M. Bacchiani, E. Weinstein, P. Nguyen, Z. Chen, Y. Wu, and K. Rao, “Multi-dialect speech recognition with a single sequence-to-sequence model,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018, pp. 4749–4753.

[5] S. Yoo, I. Song, and Y. Bengio, “A highly adaptive acoustic model for accurate multi-dialect speech recognition,” in ICASSP 2019-2019 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2019, pp. 5716–5720.

[6] W. Cui, D. Yu, X. Jiao et al., “Recent advances in speech language models: A survey,” in Proc. 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 13 943– 13 970.

[7] J. Peng, Y. Wang, B. Li, Y. Guo et al., “A survey on speech large language models for understanding,” IEEE Journal of Selected Topics in Signal Processing, vol. 20, pp. 2–31, 2024.

[8] D. Zhang, S. Li, X. Zhang et al., “SpeechGPT: Empowering large language models with intrinsic cross-modal conversational abilities,” in Proc. Findings of EMNLP, 2023, pp. 15 757–15 773.

[9] A. Defossez, L. Mazar´ e, M. Orsini´ et al., “Moshi: a speech-text foundation model for real-time dialogue,” arXiv preprint arXiv:2410.00037, 2024.

[10] Q. Fang, S. Guo, Y. Zhou et al., “LLaMA-omni: Seamless speech interaction with large language models,” in Proc. ICLR, 2025.

[11] W. Chen, Z. Ma, R. Yan et al., “SLAM-omni: Timbre-controllable voice interaction system with single-stage training,” in Proc. Findings of ACL, 2025, pp. 2262–2282.

[12] T. Mizumoto, Y. Fujita, H. Shi, L. Liu, A. Kojima, and Y. Sudo, “Evaluating Japanese dialect robustness across speech and text-based large language models,” in Proc. ASRU, 2025.

[13] Y. Chen, X. Yue, C. Zhang, X. Gao, R. T. Tan, and H. Li, “VoiceBench: Benchmarking LLM-based voice assistants,” Transactions of the Association for Computational Linguistics (TACL), vol. 14, pp. 378–398, 2026.

[14] T. Wang, L. Xu, W. Lu, and S. Cheng, “From tens of hours to tens of thousands: Scaling back-translation for speech recognition,” in Proc. EMNLP, 2025, pp. 12 461–12 475.

[15] A. Dao, D. B. Vu, H. H. Ha et al., “Speechless: Speech Instruction Training Without Speech for Low Resource Languages,” in Proc. Interspeech, 2025, pp. 3239–3243.

[16] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, “BLEU: a method for automatic evaluation of machine translation,” in Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, 2002, p. 311–318.

[17] M. Post, “A call for clarity in reporting BLEU scores,” in Proc. Third Conference on Machine Translation: Research Papers. Belgium, Brussels: Association for Computational Linguistics, Oct. 2018, pp. 186–191.

[18] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proc. ICML, vol. 202. PMLR, 2023, pp. 28 492–28 518.

[19] K. Fujii, T. Nakamura, M. Loem et al., “Continual pre-training for crosslingual llm adaptation: Enhancing japanese language capabilities,” in Proc. First Conference on Language Modeling, ser. COLM, 2024.

[20] AI@Meta, “Llama 3 model card,” https://github.com/metallama/llama3/blob/main/MODEL CARD.md, 2024.

[21] X. Wang, Y. Li, C. Fu, Y. Zhang, Y. Shen, L. Xie, K. Li, X. Sun, and L. Ma, “Freeze-omni: A smart and low latency speech-to-speech dialogue model with frozen llm,” ICML, 2025.

[22] Y. Du, Z. Ma, Y. Yang, K. Deng, X. Chen, B. Yang, Y. Xiang, M. Liu, and B. Qin, “Cot-st: Enhancing llm-based speech translation with multimodal chain-of-thought,” Proc. ACL, 2025.

[23] Z. Ma, G. Yang, Y. Yang, Z. Gao, J. Wang, Z. Du, F. Yu, Q. Chen, S. Zheng, S. Zhang et al., “Speech recognition meets large language model: Benchmarking, models, and exploration,” Proc. AAAI, 2025.

[24] G. Yang, Z. Ma, Z. Gao, S. Zhang, and X. Chen, “Ctc-assisted llm-based contextual asr,” Proc. SLT, 2024.

[25] S. Takamichi and H. Saruwatari, “CPJD corpus: Crowdsourced parallel speech corpus of Japanese dialects,” in Proc. LREC, 2018.

[26] P. Dogan-Schonberger, J. M¨ ader, and T. Hofmann, “Swissdial: Par-¨ allel multidialectal corpus of spoken swiss german,” arXiv preprint arXiv:2103.11401, 2021.

[27] Z. Tang, D. Wang, Y. Xu, J. Sun, X. Lei, S. Zhao, C. Wen, X. Tan, C. Xie, S. Zhou, R. Yan, C. Lv, Y. Han, W. Zou, and X. Li, “Kespeech: An open source speech dataset of mandarin and its eight subdialects,” in Proc. NeurIPS Datasets and Benchmarks, 2021.

[28] Y. Yin, D. Mori, and S. Fujimoto, “ReazonSpeech: A Free and Massive Corpus for Japanese ASR,” in Proc. 29th Annual Meeting of the Association for Natural Language Processing, 2023.

[29] S. Shimizu, C. Chu, S. Li, and S. Kurohashi, “Towards speech dialogue translation mediating speakers of different languages,” in Proc. Findings of ACL, 2023, pp. 1122–1134.

[30] C. Wang, A. Wu, and J. Pino, “Covost 2: A massively multilingual speech-to-text translation corpus,” arXiv preprint arXiv:2007.10310, 2020.

[31] V. Pratap, Q. Xu, A. Sriram, G. Synnaeve, and R. Collobert, “MLS: A Large-Scale Multilingual Dataset for Speech Research,” in Proc. Interspeech , 2020, pp. 2757–2761.

[32] B. Zhang, H. Lv, P. Guo, Q. Shao, C. Yang, L. Xie, X. Xu, H. Bu, X. Chen, C. Zeng, D. Wu, and Z. Peng, “Wenetspeech: A 10000+ hours multi-domain mandarin corpus for speech recognition,” in Proc. ICASSP, 2022.

[33] Qwen, :, A. Yang, B. Yang, B. Zhang et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2025.

[34] R. Soshyant, A. Meta, Cryptowooser, and Buttercream, “Tsukasa speech: Engineering the naturalness and rich expressiveness,” 2024.

[35] S. Takamichi, K. Mitsui, Y. Saito et al., “Jvs corpus: free japanese multispeaker voice corpus,” arXiv preprint arXiv:1908.06248, 2019.

[36] OpenAI, “gpt-oss-120b & gpt-oss-20b model card,” arXiv preprint arXiv:2508.10925, 2025.

[37] P. Koehn, “Statistical significance tests for machine translation evaluation,” in Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, D. Lin and D. Wu, Eds., Jul. 2004, pp. 388–395.

[38] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “Lora: Low-rank adaptation of large language models.” in ICLR, 2022.