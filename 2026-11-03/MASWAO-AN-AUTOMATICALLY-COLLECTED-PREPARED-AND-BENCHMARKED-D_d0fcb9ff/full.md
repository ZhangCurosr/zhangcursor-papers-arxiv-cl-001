# MASWAO: AN AUTOMATICALLY COLLECTED, PREPARED, AND BENCHMARKED DATASET FOR THE SOUTH AFRICAN SIGN LANGUAGE

A PREPRINT

Thapelo A. Sindane Data Science For Social Impact Lab Department of Computer Science University of Pretoria sindane.thapelo@tuks.co.za

Vukosi Marivate   
Data Science For Social Impact Lab   
Department of Computer Science   
University of Pretoria   
Lelapa AI   
vukosi.marivate@cs.up.ac.za

Herkulaas M. Combrink interdisciplinary center for digital futures Faculty of Economic and Management Sciences University of the Free State CombrinkHM@ufs.ac.za

Unarine L. Netshifhefhe Data Science For Social Impact Lab Department of Computer Science University of Pretoria u20505656@tuks.co.za

Abiodun Modupe Data Science For Social Impact Lab Department of Computer Science University of Pretoria abiodun.modupe@cs.up.ac.za

Ketro Sithole Data Science For Social Impact Lab Department of Computer Science University of Pretoria u18312889@tuks.co.za

November 3, 2026

## ABSTRACT

Recent advancements in sign language processing can be attributed to developments in both largescale dataset collection and algorithmic complexity in areas such as pose estimation techniques, signal processing through Transformer models, and model optimization. However, only a handful of sign languages, such as the American, British, and German, from over 200 globally recognized sign languages, reap the benefits of such advancements, while the rest lag behind. In this study, we aim to address this technological disparity by proposing a framework for the automatic collection and preparation of sign language datasets, with very minimal human-labour intervention, using the South African Sign Language (SASL) as a testbed. Specifically, we automatically collect SASL video content using public resources such as news broadcasts. Additionally, we convert their accompanying audio streams into text using available Automatic Speech Recognition (ASR) tools. Furthermore, we crop the sign language area in the videos and use gradient descent to locate valleys of non-motion to determine end-of-sentence points within long videos to create our weakly sentence-aligned parallel corpora – the Maswao (Signs) dataset. Finally, we benchmark the weakly aligned dataset on the task of Sign-to-text Translation using Recurrent Neural Networks and Transformer models to compare their individual intricacies. Our results show that Transformer models proposed in the literature outperform recurrent baselines, with the best-performing Transformer achieving BLEU-1 to BLEU-4 scores of 13.78, 5.21, 2.52, and 1.54, respectively, as well as ChrF = 18.09 and ROUGE-L = 10.51. These results indicate that the task remains promising and warrants further investigations.

Keywords Datasets creation · Sign-to-text Translation · Text-to-sign Translation · Recurrent Neural Networks · Transformer Models

![](images/cc49504f3195fc887beb6b70665d3428fba6ab84b9ab2d18ed958c2e9c352f01.jpg)  
(a)

![](images/7e66365120e0da9239857ac1b05ff19ce76fde0968796f842e7181fa2fa0330f.jpg)  
(b)  
Figure 1: News video example with sign language interpretation example (left) with topic model for all collected videos (right).

## 1 Introduction

Sign-to-text translation (STT) and Text-to-sign translation (TTS) are two complementary tasks aimed at enabling collegial communication between the Deaf and hearing communities Hwang et al. [2025]. In many cases, these two important subcomponents are addressed independently, with STT spearheading the race.

STT evolved from the task of sign language recognition Kong and Ranganath [2014], and can be defined as the process of translating complete sign-language sentences into grammatically sound spoken language sentences Camgoz et al. [2018]. Many works have been proposed to tackle this challenge, with earlier methods constrained by feature extraction techniques Tharwat et al. [2015], Zhang et al. [2016] transitioning to more advanced methods using convolutional neural networks (CNN), due to their enhanced hierarchical feature extraction mechanisms Tran et al. [2015].

More recently, prominent STT techniques adopted hybrid architectures comprised of deep CNN feature extractors such as ResNet Deng et al. [2009], I3D Carreira and Zisserman [2017], and Inception Szegedy et al. [2015] to model spatial relationships within the input images, and Transformer models Devlin et al. [2019] to learn the temporal dependencies of input sequences, drawing from advancements in Natural Language Processing. Regardless, the quality of available video datasets (e.g., poor video resolutions, inconsistent signer positioning, variation in signs, poor illumination, etc), and the inaccuracy of pose estimation techniques remain as the main challenges in STT to date De Coster et al. [2020].

Text-to-sign (TTS) on the other hand is the reverse process of STT Zhao et al. [2000]. Historically, TTS has often been approached as an avatar-rendering problem Cox et al. [2002]. However, due to a myriad of issues such as avatar stiffness, unnatural motion, the complexity of converting input signals into parametric representations, increased annotations costs, and high reliance on expert knowledge, research alternatives such as deep learning models using recurrent neural networks Stoll et al. [2018] and recently Transformer models Saunders et al. [2020], Diffusion models Tang et al. [2025], and generative adversarial neural networks Saunders et al. that generates photo-realistic sequences of sign sequences, are gaining momentum. Despite the impressive performance reported by these techniques, several challenges remain. These include regression to the mean Wahbi and Atsumi [2023], the accumulation of marginal errors (often referred to as drift) Saunders et al. [2020], and poor perceptual responses from Deaf communities Ventura et al. [2020], which continue to limit practical adoption and impact.

Importantly, the fuel that drives these rapid developments in both STT and TTS is the large-scale collection of parallel, annotated, and quality sign language datasets. The availability, and further refinement of such datasets remains the primary drivers of success in sign language processing. For example, datasets such as the PHOENIX2014T Koller et al. [2015], Chinese sign language dataset (CSL) Pu et al. [2020], and the How2Sign Duarte et al. [2021] are among the highly evaluated non-proprietary datasets in sign language processing history, with recent algorithmic advancements such as Transformers benchmarked, their limitations scribed, and state-of-the-arts (SOTA) established on them Fang et al. [2024]. However, similar benchmarks that shape research directions have not been established for many low-resourced sign languages, including South African Sign Language (SASL).

To address this gap, we propose a framework for the automatic collection, preparation, and benchmarking of SASL data that aims to minimise the resources required for dataset development while maintaining data quality. The framework is designed to be extensible to other low-resource sign languages with similar resource characteristics, such as the availability of unstructured public sign-language video content. More specifically, our collection includes the crawling of YouTube videos from a popular and notarized South African news outlet - the South African Broadcasting Corporation (commonly known as the SABC), which occasionally releases news video broadcasts that include a sign language interpreter, translating the presenter’s spoken utterances into SASL as shown in Figure 1a SABC.

From this source, we created the first weakly aligned spoken language to sign language dataset with over 2944 textvideo sentence-level examples, covering a wide vocabulary (see Figure 1b). Lastly, to accelerate research in STT for SASL, we benchmark the dataset using a set of state-of-the-art techniques, including MMPose Contributors [2020] for pose estimation, AlexNet-based CNN feature extraction, Recurrent Neural Networks, and Transformer models. This benchmarking establishes an initial state-of-the-art baseline for Maswao SASL dataset using an AlexNet + Transformer architecture.

The contribution of this paper is succinctly organized as follows:

1. We present the first weakly aligned parallel video-to-text corpus for South African Sign language (SASL) using publicly available news content - the Maswao (meaning ’signs’ in Northern Sesotho) dataset. This dataset comprises 2,944 weakly aligned spoken-language sentences paired with their equivalent sign-language video sentence representations.

2. We present the first benchmark on STT, using various pose estimation tools, deep CNNs, Recurrent networks, and Transformer models.

3. We establish a state-of-the-art AlexNet+Transformer model for our weakly aligned Maswao SASL dataset on the STT task.

Our work aims to accelerate advancements in South African Sign Language (SASL) research by proposing a reproducible and low-cost framework for the automatic curation of sign language datasets. To foster open and reproducible research, we publicly release our code, the Maswao dataset, and supporting resources here <sup>1</sup>, enabling other researchers to build upon and extend this work.

## 2 Related Works

According to the World Health Organization (WHO), there are over 466 million people suffering from some form of hearing loss, ranging from mild to severe hearing deficiency [WHO], Team. The WHO further projected that by the year 2070, approximately 1 billion people of the global population will be affected by hearing loss Kushalnagar [2019]. The primary mode of communication for many of these hearing-impaired individuals is sign language, with a footprint of over 200 diverse languages, with their own developed and complex grammar and structure. However, while communication between Deaf people with knowledge of the locally used sign language is possible, communication between the majority of the Deaf and the hearing communities presents many barriers reppling down to access challenges (e.g., health facilities, security services, entertainment, and many more). Interpreters have mostly been used to mediate communication between the Deaf and the hearing communities; however, the limited number of qualified interpreters often fail to meet the high demand Mohamed et al. [2022], Rastgoo et al. [2022]. Often leading to vulnerabilities in society The South African National Deaf Association [2014]. Consequently, automated sign-to-text translation and text-to-sign are emerging as promising approaches for bridging the communication divide.

End-to-end Sign-to-text (STT) translation systems historically adopted statistical approaches such as Hidden Markov Models Dreuw et al. [2007], Zafrulla et al. [2011]. These methods relied on hand-engineered features, limiting the quality and scalability of their approach to domain-constrained conditions. Subsequent to statistical approaches, classical machine learning techniques such as Conditional Random Fields, Random Forest, Support Vector Machines, and later multi-layered perceptrons took form Kong and Ranganath [2014]. However, these methods suffered the same fate as their predecessors as they also relied on feature extraction techniques to extract discriminative features Yang et al. [2008], Chavan et al. [2016]. Recently, with the advent of sophisticated deep neural network architectures, many STT pipelines have adopted complex systems using deep convolutional neural networks such as ResNet, EfficientNet, I3D, and Inception as their backbone, due to their capabilities in automatically extracting abstract latent representations of spatial data Alharthi and Alzahrani [2023]. Others have adopted deep recurrent neural networks such as LSTMs and GRUs, due to their ability to efficiently model temporal dependencies between words Camgoz et al. [2018]. However, a new wave of Transformer models drawing from successes in NLP has marked a new paradigm shift Ko et al. [2019], Ananthanarayana et al. [2021], Singla et al. [2024], Hu et al. [2023], Sreemathy et al. [2024], eliciting magnitudes of performance gains from the former methods, as datasets continue to become complex and large. For example, De Coster et al. [2020] compares LSTM models with Transformer models, using features extracted using OpenPose and ResNet-43. Their study shows that, while LSTM models perform better on the development samples, Transformer models towered over them on unseen instances, highlighting their generalization capabilities. Similarly, Camgoz et al.

[2020] proposed the first implementation of a Transformer architecture on the challenging PHOENIX2014T dataset, establishing a new state-of-the-art succeeding recurrent predecessors. Yin [2020] improved on the state-of-the-art by proposing a spatio-temporal multi-cue model stacked with a Transformer encoder for STT. De Coster et al. [2021] leverages the popular transfer learning technique using pre-trained BERT Devlin et al. [2019], and mBART-50 Tang et al. [2020] models to translate input videos into spoken language sequences. Recently, López et al. [2025] presents a CNN-based feature extractor using motion primitives derived from optical flows as inputs, which are then passed into a Transformer model for STT translation. Their comprehensive evaluation on multiple datasets: PHOENIX2014T, and Colombian sign language (CoL-SLTD) datasets, showed the effectiveness of their proposed technique.

Text-to-sign translation systems on the other hand adopted avatar rendering applications to produce sign language utterances from spoken language inputs such as text or voice Zhao et al. [2000], Cox et al. [2002, 2003], Lincoln and Cox [2003], Glauert et al. [2006]. However, issues such as restricted domain of discource leading to low vocabularies, increased annotation cost, and high reliance on specialized skillset for extracting motion capture, limited the application of these methods. To address these shortcomings, subsequent methods adopted deep learning methods such as recurrent neural networks, coupled with the emergence of pose estimation tools such as OpenPose Cao et al. [2019], and MediaPipe Lugaresi et al., to predict a sequence of joint coordinates of skeletal sequences $P = \{ P _ { M } ^ { K } \}$ with M frames and K joints, equivalent to sign language movements, given a spoken language sequence $X = \{ x _ { 1 } , x _ { 2 } , . . . , x _ { W } \}$ , with W words Stoll et al. [2020]. Recurrent neural networks, and alternate modifications such as LSTMs, Gated Recurrent Units, etc, are notoriously susceptible to the vanishing gradient problem Hochreiter [1998], and insufficiency in modeling long-temporal dependencies. Although hybrid models based on recurrent networks and attention mechanisms were proposed, they were bounded by their costly computational requirements, stemming from recurrent processing Stoll et al. [2018]. Following these methods, Transformer-based approaches, due to enhanced contextual representations, gained interest Zelinka and Kanis [2020], Saunders et al. [2020]. Consequently, Saunders et al. proposed a Transformer-based generative adversarial network that receives spoken language input sentences and converts them into pose-based sequences congruent to sign language gestures. Saunders et al. [2021] on the other hand proposed a system of multiple Transformer models, each specializing in specific feature representations, trained jointly to optimize the collective goal of sign language translation. Cui et al. [2022] presents a non-auto-regressive decoding scheme as a solution to mitigating the regression to the mean challenges, resulting in an interesting line of work building on their foundation Azevedo et al. [2024], RI et al. [2024]. Fang et al. [2024] proposes a large language model (LLM) framework that trains multiple Transformer models across several spoken-language–to–sign-language pairs. Meanwhile, diffusion-based and Transformer-based approaches are gaining traction for translating input text sequences into pose-based skeletal representations Tang et al. [2025], Wang et al. [2025].

The continued advancement of both STT and TTS research relies heavily on the availability of large-scale, high-quality, annotated datasets. In this regard, Duarte et al. [2021] introduced How2Sign, a multimodal and multi-view American Sign Language (ASL) corpus comprising over 80 hours of recordings paired with corresponding gloss annotations for sign language translation. Despite its scale, the dataset is limited to a single sign language and primarily supports translation between ASL and English. More recently, Tanzer and Zhang [2025] leveraged YouTube to construct a large-scale, open-domain multilingual sign language dataset spanning 25 sign languages, with the primary objective of translating sign language into the corresponding regional spoken language. Similarly, Avramidis et al. [2025] proposed a multilingual sign language corpus covering 12 sign languages, collected from publicly available sources including news broadcasts, government communications, and educational content. Although these datasets have substantially advanced sign language research, they include little to no representation of African sign languages or African spoken languages, highlighting a significant gap in the available resources for the African context.

South African Sign language is amongst the many under-represented languages, with only four findable literature works proposing technological solutions for sign language processing van Zijl [2006], McInnes [2014], Ndlovu and Alatise [2022], Chakoma and Baron [2023], spanning over 3 decades. Specifically, van Zijl [2006] presented a Synchronous Tree-Adjoining Grammar (STAG) parser that transforms input sentences into parse trees used to parametrize an avatar generation model. McInnes [2014] subsequently developed a low-cost hand glove for sign gesture collection. Ndlovu and Alatise [2022] uses smartphone cameras to collect their sign language dataset, and evaluate the quality of their processed datasets using classical feedforward neural networks. Recently, Chakoma and Baron [2023] presents a look-up approach that attempts to match input gestures with pre-stored sign gestures for TTS translation. However, these existing efforts focus on hand gestures or isolated signs and do not extend to sentence-level sign language translations, a limitation that the Maswao dataset seeks to address. Notably, none of the successive works from the four aforementioned work builds on the progress of existing work, rendering literature in the South African context extremely fragmented. Additionally, the two recent works Ndlovu and Alatise [2022], Chakoma and Baron [2023], do not explore Transformer advancements. To address these issues and more, we propose a continuous framework that includes the collection, and preparation of digital video resources from publicly accessible news outlets, which are then used to benchmark various processing techniques such as pose estimation tools, recurrent neural networks, and Transformer architectures.

![](images/62fe8bd9b897514e8ecf5e6e570116be3dafda3eda4d8c2a913a7c5be9ea00ff.jpg)  
Figure 2: Overview of the framework used to create the Maswao dataset.

## 3 Methodology

In this section we present a largely automated end-to-end pipeline to build a weakly aligned sign-to-text (STT) data resource from publicly accessible news broadcasts. Figure 2 shows a summary of our framework including: (i) downloading Youtube video created by SABC for all channels (ii) identifying and filtering videos without sign language interpretation (iii) generating transcriptions from the accompanying audio streams using ASR (iv) cropping and isolating the interpreter section of the videos and (v) deriving sentence-level video-to-text pairs via our proposed motion and gradient-based segmentation technique. This process resulted in our Maswao weakly aligned corpora. We then use the resulting Maswao corpus to benchmark key processing and modeling choices, including pose-estimation inputs, recurrent baselines, and Transformer-based SLT architectures.

## 3.1 Data Creation Processes

Digital sign language content is difficult to find for many low-resource sign languages. In the South African context, the South African Broadcasting Corporation (SABC) is one of the popular public news outlets that occasionally delivers spoken language news content with parallel sign language interpretation. The interpretation is done using the official SASL. The SABC releases their news recording on YouTube, where we crawled news feeds for all 11 official South African languages. To crawl the datasets, we used a python open-source library yt-dlp <sup>2</sup> to download Youtube videos, with properties such as channel (highlighting the language) set as ( English, Sepedi, Sesotho, Setswana, IsiZulu, etc), and extension type .webm to download videos of high quality where options are available – alternatively, we resorted to .mp4. Our capacity to download videos at a larger scale was restricted by Youtube bot detectors, requiring constant authentication after a few downloads, which we managed using manually downloaded cookies files. As such, we were able to download a total of only 2000 videos from SABC for all channels combined, together with their audio streams. In this study, we only report the implementation of the framework and experiments for the English (en) language, as the language with the largest crawl, and categorized the other official languages as the “Others” as shown in Table 1 describing the statistics of our Maswao dataset. As a result, the initial crawl for English resulted in a raw dataset of 377 raw videos. However, not all videos downloaded included sign language interpretation. Which meant that we had to first identify videos with interpreters included from the downloaded set, locate the position of the interpreter, and then crop the section with sign language interpretations. Fortunately, the SABC has a standard for placing a sign language interpreter in a box at the bottom right corner of the video frame, solving the location challenge, as shown in Figure 1a. This reduces the search space for interpreter-box location.

<table><tr><td>Others m</td><td>Lang.</td><td colspan="2" rowspan="5">Video Statistics</td></tr><tr><td>1623 3</td><td>Num. Raw Videos</td></tr><tr><td>1 19</td><td>Videos w/ Sign</td></tr><tr><td>1511 1</td><td>Avg. Resolution</td></tr><tr><td>5 1</td><td>Std. Resolution</td></tr><tr><td>3.19 × 1 10⁵</td><td>Var. Resolution</td></tr><tr><td>28.66 1</td><td>Avg. FPS</td></tr><tr><td>2.22 I</td><td>Std. FPS</td></tr><tr><td>4.91 1</td><td>Var. FPS Total</td></tr><tr><td>5.72 1</td><td>Hours Avg. Video</td></tr><tr><td>5.12 1</td><td>Length Std.</td></tr><tr><td>4.55 1</td><td>Video Length Var.</td></tr><tr><td>20.68 1 1 1</td><td>Video Length No. Signers</td></tr><tr><td>20.73 1</td><td>Largest Video (len.)</td></tr><tr><td>0.84 I</td><td>Smallest Video (len.)</td></tr><tr><td>2944 1</td><td>Num. Sent</td></tr><tr><td>50986 I</td><td>Num. Words</td></tr><tr><td>7815 I</td><td>Unique Vocab</td></tr></table>

Table 1: Represents the statistics of the video and text datasets. Shortened words include Number (Num), with (w/), Average (Avg.), Standard deviation (Std), Variance (Var), Frames per second (FPS), Sentence (Sent), and Vocabulary (vocab). The unit of measure for resolution is pixel. The Avg., Std., and Var. video length, and the Largest/Smallest Video columns, are measured in minutes. Largest video: aerttydScic (20m44s). Smallest video: lmMkRta0Se0 (0m50s).

<table><tr><td>Method</td><td>Total</td><td>Pred.</td><td>GT</td><td>TP</td><td>FP</td><td>FN</td><td>TN</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>FPR</td></tr><tr><td>Maswao-Interpreter-Extractor</td><td>377</td><td>82</td><td>67</td><td>55</td><td>27</td><td>12</td><td>283</td><td>0.671</td><td>0.821</td><td>0.738</td><td>0.087</td></tr></table>

Table 2: Performance of the MediaPipe-based interpreter detection on the English subset, evaluated against the manually labelled ground truth. Total: number of English videos processed. Pred.: videos the method flagged as containing an interpreter. GT: videos that contain an interpreter according to the manually corrected ground truth. TP (true positives): flagged by the method and present in the ground truth. FP (false positives): flagged by the method but absent from the ground truth. FN (false negatives): not flagged by the method but present in the ground truth. TN (true negatives): not flagged and absent from the ground truth. Prec.: $T P / ( T P + F \dot { P } )$ . Rec.: $T P \bar { / ( } T P + F N )$ . F1: harmonic mean of precision and recall. FPR: false positive rate, $F P / ( \dot { F P } + T N )$

Following this, our automated interpreter box isolation pipeline first extracts video frames from each video at predefined timestamps of 35 seconds and 65 seconds chosen empirically. Each frame is then divided into four equal quadrants by placing one horizontal line and one vertical line through the centre of the frame. Then, MediaPipe Face Detection Lugaresi et al. is applied to the bottom-right quarter regions of these frames using a confidence threshold of 0.3, where the returned normalized face coordinates are converted into pixel values and adjusted back to the original frame coordinates to establish the interpreter’s face position. We focus on this region as this is where the sign language interpreter is expected to appear, reducing unnecessary detections from the main video content. If a face is detected, the pipeline proceeds to the next steps, otherwise a video is skipped for a manual checking reducing the number of videos manually checked. For videos detected to contain a face, the final interpreter box is generated by expanding the detected face bounding box using padding parameters ("1.8 × face\_width" horizontally, "0.5 × face\_height" above, and "3.5 × face\_height" below), ensuring that the crop includes the interpreter’s upper body and surrounding white-bordered area of the interpreter box. The quality of videos, and actual size of the interpreter box varied across videos, making automatic box detection with static bounding box extension sizes sometimes challenging. Therefore, we manually checked failure cases of the automatic extraction method and manually cropped the interpreter videos using the Mediapipe interface to isolate the interpreter section from the rest of the video.

Table 2 shows the performance of our designed interpreter detection and extraction technique, reporting its true positive (TP), true negative (TN), false positives (FP), false negatives (FN), precision, recall, and F1 score, post manual corrections – that is, after we manually labelled which videos contain Interpreters and which ones do not contain them. This is done so we can evaluate this approach analytically. Overall, our method shows promising results with 67%, 82%, and 74% precision, recall, and F1 score respectively. While our interpreter extraction method still requires improvements as shown by elevated false positive – predicting 27 interpreters where they do not exist, it significantly reduces the time to go through videos without interpreters as reported by high true negative. This is expected as many videos on SABC do not include interpretation. A further manual inspection shows that FN (i.e. videos missed but contain interpreters) may be due to a number of factors such as: poor video resolution, interpreter boxes existing temporarily in the video, and changes in the expected location and shape of the box, where the interpreter box is slightly up increasing its likelihood of being chopped into the bottom and top quadrants and the box shape taking a more rectangular shape than the expected square-like shape. FP, in contrast, occur mainly in videos featuring public interviews or protests, where several people appear on screen and additional faces may fall within the region of interest, leading to misclassification. We leave further investigations and improvements targeting trade-offs between TP, TN, FP, and FN for future works. With this, the sign language videos are extracted from the news video corpus, resulting in an example video as shown in Figure 3.

![](images/47ab83eb5ee19326c2f5c1cedc8aa3736f9f4169a5268e4dbe7cff5d900168f1.jpg)  
(a) Frame 1

![](images/775335ce1686d45a7866fa46f1c3298bf9676f2f97bd637dd2005ebccd1f3fd2.jpg)  
(b) Frame 2

![](images/3b6c192376b0daefaa353acf004aaf06b0b1ac0b62dd25f1619db2c04f2f7449.jpg)  
(c) Frame 3

![](images/136698e821ad6f1c062e257bf288fb7cb99414e6da06a49117e8a983eb2c2695.jpg)  
(d) Frame 4

![](images/06b6ffc9592dd304c02710fd30f6f5d092d26729f10e52cbbda55d4cc73f5b1f.jpg)  
(e) Frame 5

![](images/19dab172767dc1cf5405d72b55db240169798c41842982067b86c3966393602a.jpg)  
(f) Frame 6  
Figure 3: Representative video frames sampled uniformly from downloaded sign language videos.

The parallel audio file can be trivially extracted from the .webm or .mp4 audio stream, which we stored as a .wav file using OpenCV <sup>3</sup> and ffmpeg <sup>4</sup>. Our objective was to align spoken language sentences with their corresponding sign language video sequences at sentence-level. However, the sign videos cropped are long videos that contain multiple sentences together with long audio files. To segment the audio streams, we used Automatic Speech Recognition (ASR) transcription to convert audio streams into text sequences for ease of tokenization using the NLTK toolkit. We expand on this in the next subsection – Automatic Speech Recognition (ASR) Transcription.

To segment sign language videos, on the other hand, we used a trivial motion detection algorithm that measures the average pixel-value difference between adjacent frames to discern motion and non-motion frames together with our proposed technique that uses gradient information (e.g direction) to locate probable segmentation points. We further defined N-breakpoints (where N is the number of sentences defined in the ASR phase during tokenization) to segment the videos using N lowest difference scores. We expand on each of these approaches subsequently.

## 3.1.1 Automatic Speech Recognition (ASR) Transcription

To transcribe the audio streams into text files, we used four state-of-the-art ASR models, namely, Whisper-large-v3 Radford et al. [2022], WAV2VEC Baevski et al. [2020], Canary-Qwen-2.5B, and Granite-speech-3.3-8b. The first two models are primarily selected due to their widespread use in ASR research, while the last two are the top two models on an Open ASR leaderboard hosted on HuggingFace<sup>5</sup>, where adapted versions of Whisper and WAV2VEC are also found to be performing competitively. However, after using these four models for transcription, a manual inspection showed that the WAV2VEC transcriptions were not satisfactory, due to poor punctuation, all words being upper-cased, among other issues, and were left out for the rest of the experiments (examples provided in the dataset release repository). Due to not having annotated reference transcription, it was challenging to automatically verify the actual quality of all the transcriptions from the models, and select which model does better compared to others, other than manual inspection of a few samples. In this case, we resorted to only verifying using grammar checks and transcriptions completeness through manual intervention. From this, we found Canary-2.5B to be generating quality transcriptions; however, it generated incomplete transcriptions, especially for long audio streams. Granite-speech-3.3-8b, on the other hand generated inconsistent transcriptions with multiple repetitions of phrases. The only model that generated quality transcriptions that required very little intervention for fixing was the Whisper-large-v3 model, which we generally adopt as the primary ASR transcriber for the remainder of this study.

Despite its strong performance, Whisper-large-v3 exhibited several recurring issues, including – misspelled proper names - especially of prominent public figures such as political leaders in South Africa (E.g., the Deputy president of South Africa “Paul Mashatile”, where Whisper-large-v3 generated - “Paul Machetele”, Canary-Qwen-2.5B generated “Paul Mashaetele” (close enough), and Granite-speech-3.3-8b generated “paul mashatele” (no capital letters to show this is a name of a person)); hallucination - the silence at the end of the audio stream provided noise to which Whisper-large v3 transcribed as being audio signals; traces of multilingual training - where traces of the Chinese characters occur in transcriptions; repetition of sentences - Whisper-large-v3 generated duplicated transcriptions where interestingly, the last sentence would typically be a correction of the preceding errors resulting in the correct transcription (however still in the transcription); and incorrect sentence delimitation. However, these issues are more pronounced in other models, while they seldom occur in Whisper. Therefore, we used Whisper-large-v3 as the primary transcription model for this study.

![](images/a2e4152cf43829564fe0ddf10216af30755b66f49bb094aed51ac80d635f2b36.jpg)  
(a) Motion Detection Analysis

![](images/4cb54c8c1c1f3b82652a35cf19a9e65354a1215c05e3b25d0494cf90320d1973.jpg)  
(b) Dataset Stratified Sampling  
Figure 4: Left: A motion-halt detection graph with gradient-informed segmentation points with sentence-size ratios enforced (S(N)), and gradient detection without sentence ratios (E(N)) breaking points. N is the number of sentences determined during the audio transcription phase. Right: dataset splitting with stratified sampling to maintain similar sentence distributions in training, development, and testing.

The resulting transcripts were manually post-edited to (i) standardize punctuation and sentence boundaries, (ii) remove incorrect or hallucinated segments, and (iii) translate code-switched words or non-English phrases into English where applicable.

With clean transcription files available, we then used the number of sentences within each transcription file as a proxy to determine the number of times to segment the parallel video sequences. For example, if the parallel transcription file of a specific sign language video contains N sentences, we immediately know we have to break the sign language video into N shorter clips. We expand on the video segmentation algorithm subsequently.

## 3.1.2 Sign Segmentation

Many complex algorithms have been proposed for motion detection Noh et al. [2024] and sign language segmentation Moryossef et al. [2023]. However, in our study, we relied on a simple yet effective algorithm that subtracts the computed average frame value between two adjacent frames, as an indicator of motion occurrence or non-occurrence. Firstly, the frames are converted into greyscale images, and then projected into numpy array values as a preliminary pre-processing step. Then for each frame, a single average pixel value is calculated by calculating the average of all pixels to represent the frame. Then, by subtracting a predecessor-frame average value from a successor-frame average value (adjacent frames), we were able to get a value δ that highlights changes in pixel values (between frames $F _ { t }$ and $F _ { t + 1 } )$ , which was enough to infer the absence or presence of motion, given the static background of our sign language videos (see Figure 1a of the cropped section). That is, the only changes outside of random noise were the movements relative to signing in most videos. A low difference value indicates no movement, while large difference values show plausible changes in movement or movements. From the list of differences δ, we identified the lowest N values as our breaking points, where N corresponds to the number of sentences in the equivalent transcription text file created in the ASR phase. That is, for N sentences in the transcription file, we expected N short video clips as segments. However, this did not perform well, as low points can appear within the occurrence of motion, leading to poorly cut videos. To address this, we used a gradient measure to find N points with the highest descent rates (indicating a drop (change) in the motion difference leading to a plateau (low point), i.e., a stop or a no-motion indicator), which improved breaking quality. That is, we used gradients to find N valleys of steepest descent. Empirical evaluation showed that a window size of 3 frames to measure gradient descent trajectories across the entire video showed improved results. Additionally, adjusting the video segmentation points based on the sentence-length distributions of the equivalent transcription files, further improved the quality of the gradient-based segmentation as shown in Figure 4a. This is done so that the longest sentence would have the longest video segment, while the shortest sentence in the transcription file will have the shortest video clip. Following this process, we chunked the long video stream into short segments that we aligned with the sentences in the equivalent transcribed file temporally (in order of appearance) to create our Maswao ’weakly’ aligned SASL datasets. This process resulted in a corpus of 2944 video-to-text weakly aligned sentence pairs and a diverse vocabulary set (see Table 1). More sophisticated motion detection and segmentation approaches that are heavily dependent on quality annotations such as prosody cues, or optical flow data Moryossef et al. [2023], Rao et al. [2025] are left for future work.

![](images/7ccb3eed962d71d6c77f1af857d17894042826c1f9fb7c0b6f23cd09a4082123.jpg)  
(a) BLEU-1

![](images/8c3b253a88cd91fc17e38f127fd199652e8ae00f2c47bf2994f50d3b8110ab90.jpg)  
(b) BLEU-2

![](images/a891e03597c7cd8a171749a6c51962e8502e538396f5144a5400ced3522b33cf.jpg)  
(c) BLEU-3

![](images/20d643f27d25809490e52107d574d7c9fec14df619844689df13f67f91368a4b.jpg)  
(d) BLEU-4  
Figure 5: BLEU Score Heatmaps: Impact of Batch Size and Learning Rate. Higher scores indicate better translation quality.

Maswao Training Details To ensure uniform sentence distribution across our training, development, and testing datasets, we used stratified sampling, with the sentence lengths as classes for dataset splitting. From this, we split the data into 2000, 444, 500 aligned video-to-text data for training, development, and testing as shown in Figure 4b (following the 80:20 split approach), respectively. This is to ensure that training and evaluation are done with the same dataset distribution.

## 3.2 Sign-to-text Translation Baselines

We validated the quality of the weakly aligned Maswao dataset through a series of training, development and testing experiments using three state-of-the-art sign-to-text translation (STT) baselines: Neural SLT (NSLT) Camgoz et al. [2018], Joint SLT Camgoz et al. [2020], and Gloss-Free End-to-End SLT (GloFE) Lin et al. [2023]. NSLT follows an encoder–decoder sequence-to-sequence design based on recurrent neural networks Luong et al. [2015], combined with either Luong et al. [2015] or Bahdanau et al. [2014] attention mechanisms. In contrast, Joint SLT and GloFE adopt Transformer backbones. This selection enables a direct comparison between recurrent sequence-to-sequence models and modern Transformer-based architectures. The baselines also differ in input modality. NSLT and Joint SLT operate on RGB image input, using AlexNet Krizhevsky et al. [2012] as the feature extractor, while GloFE uses skeletal pose features estimated with MMPose Contributors [2020]. Evaluating both image-based and pose-based inputs provides a more comprehensive assessment of the dataset and the effects of representation choices. For evaluation, we report BLEU-1 to BLEU-4 (Bilingual Evaluation Understudy) and ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation), consistent with common practice in STT benchmarking Yin and Read [2020], Wang et al. [2024], López et al. [2025].

Baseline Models Hyperparameters We experimented using mostly default hyper-parameter setups (where applicable) for training, development, and testing as originally proposed in respective works of our baseline models. These include using the same network architectures, such as the number of layers, batch size (32), optimizing algorithms (Adam optimizer), evaluation metrics – BLEU 1-4, with an addition of ChrF and ROUGE-L. Although Transformers showed improved translation performance over recurrent models, we further explored the hyper-parameter search by comparing different batch sizes in the set {16, 32, 48, 64, 128} with various learning rates {0.01, ..., 0.00001}. This highlighted that a batch size of 32 and a learning rate of 0.0001 illustrated improved performance as shown in Figure 5, which we used to train our model for much longer epochs to establish our best performing model.

## 4 Results

In this Section, we present quantitative evaluation of the Maswao corpus and summarize the baseline performance of sign-to-text translations.

## 4.1 Sign-to-Text Translation

Table 3 shows a performance comparison of recurrent-based neural networks (LSTM vs GRU with Luong et al. [2015] and Bahdanau et al. [2014] attention) trained with RGB features extracted using AlexNet CNN model – (NSLT), a Transformer-based model with RGB features extracted using AlexNet (using output features vector dimensions of 1024 and 4096) – Joint SLT, and a Transformer-based model trained with skeletal pose features extracted using a pose estimation tool, MMPose – GloFe, covering diverse modeling techniques and features extraction methodologies.

<table><tr><td>Model</td><td colspan="6">DEV</td><td colspan="6">Test</td></tr><tr><td></td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>CHRF</td><td>ROUGE</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>CHRF</td><td>ROUGE</td></tr><tr><td>NSLT-gru-luo</td><td>3.73</td><td>0.66</td><td>0.20</td><td>0.0</td><td>10.16</td><td>3.58</td><td>3.81</td><td>0.71</td><td>0.0</td><td>0.0</td><td>10.11</td><td>3.3</td></tr><tr><td>Joint SLT-1024</td><td>13.78</td><td>5.26</td><td>2.71</td><td>1.74</td><td>17.86</td><td>10.81</td><td>13.78</td><td>5.21</td><td>2.52</td><td>1.54</td><td>18.09</td><td>10.51</td></tr><tr><td>Joint SLT-4096</td><td>12.28</td><td>5.05</td><td>3</td><td>2.17</td><td>18.36</td><td>9.93</td><td>11.63</td><td>4.18</td><td>2.19</td><td>1.45</td><td>17.95</td><td>8.80</td></tr><tr><td>GloFE</td><td>12.66</td><td>3.68</td><td>1.79</td><td>1.18</td><td></td><td></td><td>12.37</td><td>3.39</td><td>1.35</td><td>0.79</td><td></td><td></td></tr><tr><td>Our SLT</td><td>12.93</td><td>4.92</td><td>2.68</td><td>1.79</td><td>18.33</td><td>10.03</td><td>12.86</td><td>4.67</td><td>2.28</td><td>1.46</td><td>18.25</td><td>9.64</td></tr></table>

Table 3: Translation performance on the Maswao DEV and Test sets. We report BLEU-1 to BLEU-4 (B1–B4), CHRF, and ROUGE scores for NSLT, GloFE, and Joint SLT models, and our proposed STT model. We ran each model five times with a different random seed and reported the average results.

From these results, it is apparent that Joint SLT outperforms recurrent-based networks and skeletal-based models across all metrics, BLEU 1-4, ChrF, and ROUGE-L. This is in line with observations highlighted in existing literature, that RGB-based feature extractions outperforms skeletal based feature extraction techniques Ko et al. [2019], Orbay and Akarun [2020]. This behaviour is compounded when we use larger dimension size (4096) features of the AlexNet model. We hypothesize that the performance degradation of skeletal estimation may be affected by the quality of the sign language videos, which warrants further investigation as studies showed that the quality of videos negatively impacted the estimation of subtle but crucial features, such the fingers De Coster et al. [2020]. Compounding the poor performance of competing models NSLT and GloFE, is that, due to a lack of matching resources used in their respective studies, their techniques, such as the contrastive learning (which required linguistic insights such as parts of speech tags) in GloFE, and supervision signals from glosses in NSLT where omitted from experiments, which may have negatively impacted training performance. Consequently, we adopted Joint SLT, with optimal training parameters (batch size=32, learning rate=1e<sup>−4</sup>), with longer early stopping patience=50 to train for longer times (epochs = 616) to establish our model – Our STT. Our STT translation model established a benchmark for our automatically collected and prepared, weakly aligned Maswao dataset, curving new paths for future works in sign language translation for the South African Sign Language (SASL) community. Moreover, this study pioneers works in automatic collection and preparation (i.e., alignment of sentences to parallel video chunks in long videos) of low-resourced sign language datasets with similar resource-availability characteristics.

Comparative analysis of recurrent architectures demonstrates that an GRU-based recurrent models integrated with Luong attention Luong et al. [2015] achieves superior performance compared to GRU with Bahdanau attention Bahdanau et al. [2014] or LSTM configurations using either Luong attention Luong et al. [2015] or Bahdanau attention Bahdanau et al. [2014]. Nevertheless, despite these findings, all recurrent-based models underperform relative to Transformer architectures and additionally incur substantially higher training times.

## 4.1.1 Additional Results

To complement lexical-overlap metrics, which penalise valid paraphrases and say little about meaning, we evaluated the best-performing sign-to-text model with BERTScore and MetricX. BERTScore compares candidate and reference sentences using contextual token embeddings from a pretrained language model, computing precision, recall and F1-score from soft token matches, so it rewards semantic similarity rather than exact word overlap Zhang et al. [2019]. MetricX-24 is a learned, regression-based translation quality metric built on mT5 and trained on human quality ratings, which predicts an error-style score where lower values indicate better translations Juraska et al. [2024]. Together, these metrics matter for our weakly aligned Maswao dataset because a sign video can be translated in several valid ways while only a single reference is available, so generated outputs may be semantically related to the reference without being an exact match. As shown in Table 4, our best-performing model obtains a BERTScore of 12.20 (P), 14.60 (R) and 13.23 (F1), and a MetricX error score of 21.85. These values indicate limited semantic fidelity and substantial translation errors, and the higher recall than precision suggests that the model captures part of the reference meaning but also produces additional or irrelevant content. Such results are expected given the difficulty of open-domain sign-to-text translation, which may be compounded by the diverse vocabulary of Maswao and by signer variability. Importantly, the objective of this work was not to develop the best possible model but to propose a dataset; the model serves as a reproducible baseline showing that Maswao can be used to train and evaluate sign-to-text systems, and the remaining headroom provides a clear benchmark for future work. To better understand these results, we further analyse the outputs of our best model through a qualitative comparison between the generated and reference translations, reported in the following section.

<table><tr><td rowspan="2">Model</td><td colspan="3">BERTScore</td><td rowspan="2">MetricX Score</td></tr><tr><td>P</td><td>R</td><td>F1</td></tr><tr><td>Our Model</td><td>12.20</td><td>14.60</td><td>13.23</td><td>21.85</td></tr></table>

Table 4: Complementary evaluation of the best-performing sign-to-text model using BERTScore and MetricX.

<table><tr><td rowspan=1 colspan=1>GTOur STT</td><td rowspan=1 colspan=1>And it was incumbent upon us to set up policies.This is but we do that we had the policies which we had to set up regarding thelanguage.</td></tr><tr><td rowspan=1 colspan=1>GTOur STT</td><td rowspan=1 colspan=1>Do you see police visibility?They are still shooting on the police around here?</td></tr><tr><td rowspan=1 colspan=1>GTOur STT</td><td rowspan=1 colspan=1>And it is spoken by blacks in not only so-called colored townships.That&#x27;s the Afrikaans language has been spoken as you just heard.</td></tr><tr><td rowspan=1 colspan=1>GTOur STT</td><td rowspan=1 colspan=1>The 6 am Central African Time you&#x27;re watching SABC News live from Johannes-burg.You&#x27;re watching SABC News live from Johannesburg in your headlines this hour.</td></tr></table>

Table 5: Translations from our neural network model using Maswao datasets. GT = Ground Truth, STT = System Output. Highlighted words indicate correct matches.

## 5 Qualitative Results

In this section, we share qualitative results of our best-performing sign language translation model. Table 5 shows four example translations of our model compared to the ground truth (GT) results.

From this, we can observe that our model is able to generate sound, coherent, and structurally correct sentences. Although these sentences do not generally match the semantics of the ground truth labels, our model is able to generate keywords (highlighted in green) appearing in the ground truth. Furthermore, the models were able to capture context (row 3) as Afrikaans is a language spoken by the colored communities of South Africa. This probably means that somewhere in the embedding space, the representations of the word colored and Afrikaans were collocated. However, this warrants further investigation. We hypothesize that with increased training examples and higher quality videos, better modeling, improved segmentations, results on the Maswao datasets would lead to more accurate translations.

## 6 Conclusion

The Deaf and Hard-of-hearing communities have been sidelined from many societal benefits, lending perpetuated marginalization of the communities and reduced access to important socio-economic activities. While advancements in sign-to-text and text-to-sign research have been used to address this from a communication-facilitation perspective, issues of unequal research efforts arise, where only a few languages reap the benefits of important strides. To address this disparity, we propose a reproducible framework that details the automatic collection and preparation of digital news content of sign languages, using the South African Sign Language (SASL) as a case study. Following this, we create sequences of weakly aligned sign language short videos to spoken language text sentences, using simple yet effective techniques such as automated speech recognition, non-motion detection with gradient descent, and sentence-length distributions of the accompanying transcription file weighting. The quality of our datasets is then evaluated using MMPose, AlexNet features, recurrent neural networks and Transformer architectures, highlighting the efficacy of our automated approach to dataset collection and preparation. Concretely, our results show that Transformer-based modeling with RGB features extracted using AlexNet, outperforms recurrent neural networks with the same features, and Transformer models with skeletal pose estimation features, on our proposed Maswao dataset.

We hope to see future works exploring other pose estimation techniques, such as OpenPose and MediaPipe, for improved skeletal pose features. Additionally, we hope to see improvements in video segmentation for better alignments of sentences. Furthermore, with this complex task and dataset, we hope to see improvements in model complexity to capture the intricacies in sign language translation. Moreover, Text-to-sign (TTS) has been considered a much more challenging task compared sign-to-text. As such, our future works will build on this to include experiments for TTS for building end-to-end sign language translation systems for Deaf communities. Finally, our next milestone is to scale the framework we proposed here to include multiple South African languages, with rigorous validation steps across phases to improve dataset quality.

## 7 Limitations

Due to financial and resource constraints, this study did not annotate ground truth video segments to evaluate the automatic segmentation technique proposed to create our weakly aligned sentence-level Maswao dataset. However, through a manual inspection of video samples, it can be observed that our proposed technique segments videos exactly where the movement of the signer is reaching a stop. Notably, isolated lexical items (e.g., the word “Interesting”), which appeared as sentence number 93 from a total of 94 sentences, from the video with YouTube id “1-UpxWii8Yg” was segmented correctly as shown in Segment 93 of folder “1-UpxWii8Yg” of our released dataset. This provides evidence of the effectiveness of the proposed gradient-descent and sentence-length–based segmentation strategy.

Similarly, the study did not have ground truth or labeled transcriptions to compare ASR techniques. As future work, it would be valuable to produce ground-truth transcriptions for a representative subset of the corpus and to investigate automatic transcriber selection using recent SOTA large language models (e.g., ChatGPT, LLaMA, and Gemini). In addition, exploring ranking-based selection frameworks may further improve the reliability and consistency of transcription model choice.

Although the adopted Automatic Speech Recognition (ASR) system produced high-quality transcriptions overall, the generated transcripts were not systematically validated by independent human translators. This is particularly relevant in cases where news presenters or interviewed persons code-switched into local languages, which can reduce ASR accuracy. Instead in this work, the authors triaged through the transcriptions to manually fix identified errors, similar to Tanzer and Zhang [2025]. While this approach enabled the construction of the dataset, a comprehensive human validation of the transcriptions would further improve dataset quality, quantify the impact of transcription errors, and help identify potential sources of noise that remain undetected.

Owing to YouTube’s licensing terms and our ethical responsibility to protect the privacy of the sign language interpreters whose faces appear in the videos, we are unable to publicly redistribute the original sign language video clips included in the Maswao dataset. Instead, we release the corresponding YouTube video identifiers (IDs) and the accompanying sentence-level text annotations, allowing researchers to reconstruct the dataset from the original publicly available sources while respecting both platform licensing requirements and the privacy of the individuals depicted.

## References

Eui Jun Hwang, Sukmin Cho, Huije Lee, Youngwoo Yoon, and Jong C Park. A spatio-temporal representation learning as an alternative to traditional glosses in sign language translation and production. In 2025 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), pages 3352–3362. IEEE, 2025.

WW Kong and Surendra Ranganath. Towards subject independent continuous sign language recognition: A segment and merge approach. Pattern Recognition, 47(3):1294–1308, 2014.

Necati Cihan Camgoz, Simon Hadfield, Oscar Koller, Hermann Ney, and Richard Bowden. Neural sign language translation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7784–7793, 2018.

Alaa Tharwat, Tarek Gaber, Aboul Ella Hassanien, Mohamed K Shahin, and Basma Refaat. Sift-based arabic sign language recognition system. In Afro-European Conferencefor Industrial Advancement: Proceedings ofthe First International Afro-European Conferencefor Industrial Advancement AECIA 2014, pages 359–370. Springer, 2015.

Chenyang Zhang, Yingli Tian, and Matt Huenerfauth. Multi-modality american sign language recognition. In 2016 IEEE International Conference on Image Processing (ICIP), pages 2881–2885. IEEE, 2016.

Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In Proceedings of the IEEE international conference on computer vision, pages 4489–4497, 2015.

Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.

Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017.

Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. In Proceedings ofthe IEEE conference on computer vision and pattern recognition, pages 1–9, 2015.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectiona transformers for language understanding. In Proceedings of the 2019 conference of the North American chapter

of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pages 4171–4186, 2019.

Mathieu De Coster, Mieke Van Herreweghe, and Joni Dambre. Sign language recognition with transformer networks. In Proceedings of the twelfth language resources and evaluation conference, pages 6018–6024, 2020.

Liwei Zhao, Karin Kipper, William Schuler, Christian Vogler, Norman Badler, and Martha Palmer. A machine translation system from english to american sign language. In Conference ofthe Associationfor Machine Translation in the Americas, pages 54–67. Springer, 2000.

Stephen Cox, Michael Lincoln, Judy Tryggvason, Melanie Nakisa, Mark Wells, Marcus Tutt, and Sanja Abbott. Tessa, a system to aid communication with deaf people. In Proceedings of the fifth international ACM conference on Assistive technologies, pages 205–212, 2002.

Stephanie Stoll, Necati Cihan Camgöz, Simon Hadfield, and Richard Bowden. Sign language production using neural machine translation and generative adversarial networks. In Proceedings ofthe 29th British Machine Vision Conference (BMVC 2018). British Machine Vision Association, 2018.

Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Progressive transformers for end-to-end sign language production. In European Conference on Computer Vision, pages 687–705, 2020.

Shengeng Tang, Feng Xue, Jingjing Wu, Shuo Wang, and Richang Hong. Gloss-driven conditional diffusion models for sign language production. ACM Transactions on Multimedia Computing, Communications and Applications, 21(4): 1–17, 2025.

Ben Saunders, Necati Cihan Camgöz, and Richard Bowden. Adversarial training for multi-channel sign language production. In The 31st British Machine Vision Virtual Conference. British Machine Vision Association.

Ehssan Wahbi and Masayasu Atsumi. Semi-autoregressive transformer for sign language production. In 人工知能学会全国大会論文集 第 37 回 (2023), pages 1U3IS2a01–1U3IS2a01. 一般社団法人 人工知能学会 , 2023.

Lucas Ventura, Amanda Duarte, and Xavier Giró-i Nieto. Can everybody sign now? exploring sign language video generation from 2d poses. arXiv preprint arXiv:2012.10941, 2020.

Oscar Koller, Jens Forster, and Hermann Ney. Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. Computer Vision and Image Understanding, 141:108–125, 2015.

Junfu Pu, Wengang Zhou, Hezhen Hu, and Houqiang Li. Boosting continuous sign language recognition via cross modality augmentation. In Proceedings of the 28th ACM international conference on multimedia, pages 1497–1505, 2020.

Amanda Duarte, Shruti Palaskar, Lucas Ventura, Deepti Ghadiyaram, Kenneth DeHaan, Florian Metze, Jordi Torres, and Xavier Giro-i Nieto. How2sign: a large-scale multimodal dataset for continuous american sign language. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, pages 2735–2744, 2021.

Sen Fang, Chen Chen, Lei Wang, Ce Zheng, Chunyu Sui, and Yapeng Tian. Signllm: Sign language production large language models. arXiv e-prints, pages arXiv–2405, 2024.

SABC. Sabc news – sabc – official website. URL https://www.sabc.co.za/sabc/sabc-news-division/.

MMPose Contributors. Openmmlab pose estimation toolbox and benchmark. https://github.com/open-mmlab/ mmpose, 2020. Accessed: 2025-11-17.

World Health Organization (WHO). Deafness and Hearing Loss. https://www.who.int/news-room/ fact-sheets/detail/deafness-and-hearing-loss. [Accessed 24-09-2025].

World Health Organization Team. World report on hearing — who.int. https://www.who.int/publications/i/ item/9789240020481. [Accessed 26-09-2025].

Raja Kushalnagar. Deafness and hearing loss. In Web accessibility: A foundation for research, pages 35–47. Springer, 2019.

Ammar Mohamed, Hesham Hefny, et al. A deep learning approach for gloss sign language translation using transformer. Journal ofComputing and Communication, 1(2):1–8, 2022.

Razieh Rastgoo, Kourosh Kiani, Sergio Escalera, Vassilis Athitsos, and Mohammad Sabokrou. All you need in sign language production. arXiv preprint arXiv:2201.01609, 2022.

The South African National Deaf Association. Aftermath of a fake sign language interpreter. http://www.sanda. org.za/assets/files/Aftermath-of-a-Fake-Sign-Language-Interpreter2014.pdf, 2014. Accessed: 2025-09-26.

Philippe Dreuw, David Rybach, Thomas Deselaers, Morteza Zahedi, and Hermann Ney. Speech recognition techniques for a sign language recognition system. hand, 60(80), 2007.

Zahoor Zafrulla, Helene Brashear, Thad Starner, Harley Hamilton, and Peter Presti. American sign language recognition with the kinect. In Proceedings ofthe 13th international conference on multimodal interfaces, pages 279–286, 2011.

Hee-Deok Yang, Stan Sclaroff, and Seong-Whan Lee. Sign language spotting with a threshold model based on conditional random fields. IEEE transactions on pattern analysis and machine intelligence, 31(7):1264–1277, 2008.

Poonam Chavan, Tushar Ghorpade, and Puja Padiya. Indian sign language to forecast text using leap motion sensor and rf classifier. In 2016 Symposium on Colossal Data Analysis and Networking (CDAN), pages 1–5. Ieee, 2016.

Nojood M Alharthi and Salha M Alzahrani. Vision transformers and transfer learning approaches for arabic sign language recognition. Applied sciences, 13(21):11625, 2023.

Sang-Ki Ko, Chang Jo Kim, Hyedong Jung, and Choongsang Cho. Neural sign language translation based on human keypoint estimation. Applied sciences, 9(13):2683, 2019.

Tejaswini Ananthanarayana, Nikunj Kotecha, Priyanshu Srivastava, Lipisha Chaudhary, Nicholas Wilkins, and Ifeoma Nwogu. Dynamic cross-feature fusion for american sign language translation. In 2021 16th IEEE International Conference on Automatic Face and Gesture Recognition (FG 2021), pages 1–8. IEEE, 2021.

Venus Singla, Seema Bawa, and Jasmeet Singh. Enhancing indian sign language recognition through data augmentation and visual transformer. Neural Computing and Applications, 36(24):15103–15116, 2024.

Hezhen Hu, Weichao Zhao, Wengang Zhou, and Houqiang Li. Signbert+: Hand-model-aware self-supervised pretraining for sign language understanding. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9): 11221–11239, 2023.

R Sreemathy, Mousami Turuk, Jayashree Jagdale, Mohammad Anish, et al. Indian sign language recognition using video vision transformer. In 2024 3rd International Conferencefor Advancement in Technology (ICONAT), pages 1–7. IEEE, 2024.

Necati Cihan Camgoz, Oscar Koller, Simon Hadfield, and Richard Bowden. Sign language transformers: Joint end-toend sign language recognition and translation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10023–10033, 2020.

Kayo Yin. Sign language translation with transformers. arXiv preprint arXiv:2004.00588, 2, 2020.

Mathieu De Coster, Karel D’Oosterlinck, Marija Pizurica, Paloma Rabaey, Severine Verlinden, Mieke Van Herreweghe, and Joni Dambre. Frozen pretrained transformers for neural sign language translation. Proceedings of Machine Translation Summit XVIII, 2021.

Yuqing Tang, Chau Tran, Xian Li, Peng-Jen Chen, Naman Goyal, Vishrav Chaudhary, Jiatao Gu, and Angela Fan. Multilingual translation with extensible multilingual pretraining and finetuning. arXiv preprint arXiv:2008.00401, 2020.

Fredy Alejandro Mendoza López, Jefferson Rodriguez, and Fabio Martínez. A multitask transformer to sign language translation using motion gesture primitives. arXiv preprint arXiv:2503.19668, 2025.

Stephen Cox, Michael Lincoln, Judy Tryggvason, Melanie Nakisa, Mark Wells, Marcus Tutt, and Sanja Abbott. The development and evaluation of a speech-to-sign translation system to assist transactions. International Journal of Human-Computer Interaction, 16(2):141–161, 2003.

Mike Lincoln and Stephen Cox. A comparison of language processing techniques for a constrained speech translation system. In 2003 IEEE International Conference on Acoustics, Speech, and Signal Processing, 2003. Proceedings.(ICASSP’03)., volume 1, pages I–I. IEEE, 2003.

John RW Glauert, Ralph Elliott, Stephen J Cox, Judy Tryggvason, and Mary Sheard. Vanessa–a system for communication between deaf and hearing people. Technology and disability, 18(4):207–216, 2006.

Zhe Cao, Gines Hidalgo, Tomas Simon, Shih-En Wei, and Yaser Sheikh. Openpose: Realtime multi-person 2d pose estimation using part affinity fields. IEEE transactions on pattern analysis and machine intelligence, 43(1):172–186, 2019.

Camillo Lugaresi, Jiuqiang Tang, Hadon Nash, Chris McClanahan, Esha Uboweja, Michael Hays, Fan Zhang, Chuo-Ling Chang, Juhyun Lee, et al. Mediapipe: A framework for perceiving and processing reality.

Stephanie Stoll, Necati Cihan Camgoz, Simon Hadfield, and Richard Bowden. Text2sign: towards sign language production using neural machine translation and generative adversarial networks. International Journal ofCompute Vision, 128(4):891–908, 2020.

Sepp Hochreiter. The vanishing gradient problem during learning recurrent neural nets and problem solutions. International Journal ofUncertainty, Fuzziness and Knowledge-Based Systems, 6(02):107–116, 1998.

Jan Zelinka and Jakub Kanis. Neural sign language synthesis: Words are our glosses. In Proceedings of the IEEE/CVF winter conference on applications ofcomputer vision, pages 3395–3403, 2020.

Ben Saunders, Necati Cihan Camgoz, and Richard Bowden. Mixed signals: Sign language production via a mixture of motion primitives. In Proceedings ofthe IEEE/CVF International Conference on Computer Vision, pages 1919–1929, 2021.

Zhenchao Cui, Ziang Chen, Zhaoxin Li, and Zhaoqi Wang. A pyramid semi-autoregressive transformer with rich semantics for sign language production. Sensors, 22(24):9606, 2022.

Rafael V Azevedo, Thiago M Coutinho, João P Ferreira, Thiago L Gomes, and Erickson R Nascimento. Empowering sign language communication: Integrating sentiment and semantics for facial expression synthesis. Computers & Graphics, 124:104065, 2024.

Minu RI et al. Tamil lang tsp: Tamil lang transformer neural text to sign production. International Arab Journal of Information Technology (IAJIT), 21(6), 2024.

Xu Wang, Shengeng Tang, Peipei Song, Shuo Wang, Dan Guo, and Richang Hong. Linguistics-vision monotonic consistent network for sign language production. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE, 2025.

Garrett Tanzer and Biao Zhang. Youtube-sl-25: A large-scale, open-domain multilingual sign language parallel corpus. In International Conference on Learning Representations, volume 2025, pages 81921–81934, 2025.

Eleftherios Avramidis, Vera Czehmann, Fabian Deckert, Lorenz Hufe, Aljoscha Lipski, Yuni Amaloa Quintero Villalobos, Tae Kwon Rhee, Mengqian Shi, Lennart Stölting, Fabrizio Nunnari, et al. The tub sign language corpus collection. In Adjunct Proceedings ofthe 25th ACM International Conference on Intelligent Virtual Agents, pages 1–7, 2025.

Lynette van Zijl. South african sign language machine translation project. In Proceedings ofthe 8th International ACM SIGACCESS Conference on Computers and Accessibility, pages 233–234, 2006.

Ben McInnes. South african sign language dataset development and translation: a glove-based approach. 2014.

Wonder Ndlovu and Mary Alatise. Application of artificial neural network on south african sign language recognition system. In 2022 International Conference on Artificial Intelligence, Big Data, Computing and Data Communication Systems (icABCD), pages 1–8. IEEE, 2022.

Shingirirai Chakoma and Philip Baron. Converting south african sign language to verbal. SAIEE Africa Research Journal, 114(2):49–57, 2023.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. Robust speech recognition via large-scale weak supervision, 2022. URL https://arxiv.org/abs/2212.04356.

Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems, 33:12449–12460, 2020.

Donghyeon Noh, Hojin Yoon, and Donghun Lee. A decade of progress in human motion recognition: A comprehensive survey from 2010 to 2020. IEEE Access, 12:5684–5707, 2024.

Amit Moryossef, Zifan Jiang, Mathias Müller, Sarah Ebling, and Yoav Goldberg. Linguistically motivated sign language segmentation. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 12703–12724, 2023.

Kavu Maithri Rao, Yasser Hamidullah, and Eleftherios Avramidis. Sign language video segmentation using temporal boundary identification. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 4: Student Research Workshop), pages 1213–1224, 2025.

Kezhou Lin, Xiaohan Wang, Linchao Zhu, Ke Sun, Bang Zhang, and Yi Yang. Gloss-free end-to-end sign language translation. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 12904–12916, Toronto, Canada, July 2023. Association for Computational Linguistics. doi:10.18653/v1/2023.acl-long.722. URL https://aclanthology.org/2023.acl-long.722.

Thang Luong, Hieu Pham, and Christopher D Manning. Effective approaches to attention-based neural machine translation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, page 1412. Association for Computational Linguistics, 2015.

Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473, 2014.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.

Kayo Yin and Jesse Read. Better sign language translation with stmc-transformer. In Proceedings of the 28th International Conference on Computational Linguistics, pages 5975–5989, 2020.

Qingshan Wang, Zhiwen Zheng, Qi Wang, Dazhu Deng, and Jiangtao Zhang. Generalizations of wearable device placements and sentences in sign language recognition with transformer-based model. IEEE Transactions on Mobile Computing, 23(10):10046–10059, 2024.

Alptekin Orbay and Lale Akarun. Neural sign language translation by learning tokenization. In 2020 15th IEEE international conference on automaticface and gesture recognition (FG 2020), pages 222–228. IEEE, 2020.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675, 2019.

Juraj Juraska, Daniel Deutsch, Mara Finkelstein, and Markus Freitag. MetricX-24: The Google submission to the WMT 2024 metrics shared task. In Barry Haddow, Tom Kocmi, Philipp Koehn, and Christof Monz, editors, Proceedings of the Ninth Conference on Machine Translation, pages 492–504, Miami, Florida, USA, November 2024. Association for Computational Linguistics. URL https://aclanthology.org/2024.wmt-1.35.