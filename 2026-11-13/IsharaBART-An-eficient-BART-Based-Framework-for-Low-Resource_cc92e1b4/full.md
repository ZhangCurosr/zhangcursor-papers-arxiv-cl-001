# IsharaBART: An eficient BART-Based Framework for Low-Resource Bangla Sign Language Translation

Aonmoy Das, Ananna Dev Aishi, Md. Fazlul Kader∗

Department of Electrical and Electronic Engineering, University of Chittagong, Chittagong-4331, Bangladesh.

## Abstract

Millions of Bangla Sign Language (BdSL) users lack reliable automatic translation tools. Existing systems are limited by data scarcity, unidirectional design, and computationally expensive architectures that are not suitable for low-resource deployment. We introduce IsharaBART, a bidirectional generative framework for gloss-sentence translation in BdSL. IsharaBART uses Bidirectional and Auto-Regressive Transformers (BART)-based condi tional language modeling with denoising learning objectives. We developed a parallel dataset of 1,000 professionally annotated gloss-sentence pairs, balanced across tenses and structures. We expanded the dataset to 4,831 pairs through four augmentation techniques: rule-based tense transformations, masked token substitutions, back-translation, and one-shot prompting. A compact denoising sequence-to-sequence model was pre-trained from scratch on monolingual Bangla text from six diferent corpora, then fine-tuned on the annotated and augmented datasets. We compare IsharaBART against four larger baseline models under identical conditions, where it achieves comparable or superior bidirectional performance. On the gloss-to-sentence (G2S) task, it attains Bilingual Evaluation Understudy (BLEU-4) score up to 51.7, and on the sentence-to-gloss (S2G) task, it achieves 61.40, outperform ing multilingual models despite having fewer than 10% of the parameters. Ablation studies demonstrate that pre-training provides 62.5–90.4% faster convergence and 11.5-118% higher BLEU-4 than scratch. Low-rank adaptation (LoRA) reduces trainable parameters by 99.3%, but with 28.0-52.2% loss of BLEU-4 compared to full-parameter fine-tuning. IsharaBART establishes a scalable and deployable pipeline for Bangla sign language bidirectional trans lation, enhancing accessibility in education, healthcare, and social contexts for the deaf and speech-impaired community, while providing a blueprint for similar low-resource languages like Bangla.

Keywords: Sign language translation, Bangla sign language, Gloss-sentence translation, Data augmentation, Transformer

## 1. Introduction

## 1.1. Sign Language and Sign Language Translation

Sign language stands as one of the most ancient and instinctive method of communication, which serves as the primary mode of communication for people with hearing or speech impairments. According to the World Health Organization, approximately 466 million people worldwide (6.1% of the global population) experience hearing loss, with projections indicating this number will surpass 900 million by 2050 [1]. These sign language users face persistent dificulties in accessing digital services, educational resources, and social participation due to the lack of efective and automatic translation systems. Sign languages are independent linguistic systems with their own lexicons, grammar, and syntax, conveying meaning through manual features such as hand shape, movement, and hand position, together with nonmanual features such as facial expression, mouthing, and eyebrow movement [2, 3]. Because sign languages rarely have a standardized written form, most technology built for the deaf community must work directly with video or motion data rather than with text, and this is the central technical dificulty that the present work addresses.

SLR methods have progressed from small-vocabulary classifiers trained on hand-crafted features and depth sensors, to deep networks that process body keypoints or raw video using convolutional and recurrent layers, and more recently to transformer-based encoders that better capture long-range temporal structure [4]. SLT methods build on this progress in two broad ways. Cascading approaches first recognize glosses from a video and then translate the gloss sequence into text, while end-to-end approaches translate directly from video features to text, sometimes still using glosses as an auxiliary training signal rather than an explicit intermediate step [4, 3].

## 1.2. Bangla Sign Language

For Bangla Sign Language (BdSL), which is spoken by the deaf and speech-impaired community in Bangladesh and parts of India, representing one of the world’s biggest linguistic regions, the challenges are significantly serious, as technological solutions remain severely underdeveloped compared to other high-resource sign languages such as American Sign Language (ASL) [5], Chinese Sign Language (CSL) [6] or German Sign Language (DGS) [7]. Hearing loss afects a substantial share of the population in the region: recent estimates place significant hearing impairment in Bangladesh at close to 9.6% of the population, or approximately 13.7 million people [8].

BdSL is linguistically distinct from spoken Bangla, with its own sign order and grammatical structure. Therefore, gloss–sentence translation requires cross-representation generation rather than simple word reordering. BdSL is also considerably more low-resource than languages such as ASL, DGS, or CSL. Large parallel corpora with video, gloss, and text annotations, do not yet exist for BdSL, and most prior BdSL work has focused on isolated alphabets or a small set of static signs rather than sentence-level translation [8]. Most existing BdSL resources are small and recent. Regional dialects and individual diferences in signing style add a further layer of variability, with the same word sometimes signed diferently across signers and regions. On top of this, there is no widely agreed standard for how BdSL should be glossed, a problem BdSL shares with sign languages in general [3], which makes it dificult to combine data from diferent sources or compare results across studies.

## 1.3. Motivation and Research Gap

BdSL recognition and translation research has mostly been in very resource limited settings. Most of the past recognition works have focused on individual sign gestures of words, vowels and consonants, and numbers [9, 10, 11, 12]. These works have been highly efective on isolated sign recognition, but their approach do not address the recognition and translation of continuous sign language gestures. Recent works have started to address continuous translation [13, 8]. These studies are limited to small and shallow architectures with limited capacity to generalize. The datasets used in these studies are limited by very small vocabularies. Moreover, a fundamental requirement for building efective sign language communication tools is bidirectional translation. Past works do not address bidirectional translation and evaluation. To the best of our knowledge, these systems do not utilize modern pre-trained language models or use architectures that could enable both G2S and sentence-to-gloss (S2G) generation.

These research gaps come from three fundamental challenges in low resource languages like BdSL. First, BdSL does not yet have large scale gloss-sentence pair parallel dataset. Such high quality dataset are essential to training high performing models. Second, existing studies use large, resource heavy multilingual models. These models may not have optimal tokenization or scoped abilities specific to BdSL gloss-sentence translation. Third, real-world application of translation tools require small, compact models that are capable of running on edge devices and mobile platforms with limited computational resources.

## 1.4. Contributions of This Study

In this work, we introduce IsharaBART, a bidirectional generative framework specifically designed for BdSL gloss-sentence translation that addresses the aforementioned fundamental gaps through these key contributions:

1. We developed IsharaBART, a small (56M) denoising seq2seq model pre-trained on monolingual Bangla text from six diferent corpora. IsharaBART achieved competitive bidirectional gloss-sentence translation performance while using substantially fewer parameters (78-90% fewer) than the baseline models.

2. We developed a parallel gloss-sentence dataset for BdSL, which contains 1,000 professionally annotated gloss-sentence pairs. The data is balanced across tenses and syntactic structures. The dataset also contains 3,831 synthetically augmented pairs using four complementary augmentation techniques.

3. We proposed a comprehensive framework for sign language gloss-sentence data augmentation and data validation that expands our 1,000-pair dataset to 4,831 pairs while preserving structural, semantic, and lexical consistency.

4. We conducted a detailed bidirectional evaluation of IsharaBART by comparing against four baseline models and diferent architectures under controlled experimental conditions (identical datasets, training protocols, and evaluation metrics). It is demon strated that monolingual pre-training enables dramatic convergence speedups (62.5- 90.4% faster) in gloss-sentence translation tasks.

5. We performed ablation studies comparing full fine-tuning against Low-rank adaptation (LoRA) to provide relevant information for deployment scenarios with diferent computational resources.

## 1.5. Organization of This Paper

The remainder of this paper is structured in the following way. Section 2 provides a detailed literature review in general sign language and BdSL translation. Section 3 describes the methodology used in the study. Section 4 describes the experimental design and evalu ation metrics. Section 5 reports and provides analysis of the results from the experiments and ablation studies. Section 5.7 further discusses the findings of the study and provides a more abstract implications of those results. Finally, Section 6 outlines the limitations of the study and provides potential future research directions to address those limitations.

## 2. Literature review

Research on automatic sign language processing is broadly categorized into two diferent tasks [3]. Sign Language Recognition (SLR) identifies signs in a video. It usually outputs these gestures as glosses or words in the order they appear in the video. Sign Language Translation (SLT) takes these sequence of glosses and outputs meaningful sentences to complete communication [3]. The reverse direction of generating gesture sequences from spoken texts also falls under this category.

## 2.1. SLR Evolution

SLR has progressed from classification of gestures by simpler hand centric feature extraction to more complex and deeper multimodal deep learning [14]. Early works used depth-based CNNs for fingerspelling [15], CNNs for 3D pose estimation of hands [16], RGBdepth-skeleton fusion [17], and restricted Boltzmann machines for fusing multidimentional features [18]. Later studies used 3D residual CNNs and BiLSTMs for continuous sign recognition [19]. Hasanov et al. studied isolated Azerbaijani fingerspelling [20]. Aiswarya et al. used keyframe selection to process continuous sign videos [21]. Hao et al. combined sensing-based CNNs with BiLSTM-based attention [22]. Renjith et al. used Transformers to combine motion and visual features [23]. Barrister et al. used multiscale Transformer and convolutional features to detect Mizo signs [24]. Wu et al. added sign cues to trimodal speech processing [25]. Talaat et al. developed a sign-language translation system that generates Arabic avatars in real time [26]. However, most studies focus on isolated-sign recognition or specific applications. Few address translation in both directions between sign-language glosses and sentences.

## 2.2. Transformer-Based SLT

Researchers moved from CNN-LSTM models to transformer-based models for continuous SLR and translation. Camgoz et al. [27] used transformers for end-to-end sign language translation. They tested their model on the RWTH-PHOENIX-Weather-2014T dataset. The dataset included 1,066 unique signs from 9 signers in unconstrained settings. Their model performed better than earlier gloss-to-text methods. It improved the BLEU-4 score from 9.58 to 21.80. Chaudhary et al. [28] proposed SignNet II. It uses a transformer-based architecture for two-way sign language translation. The model used a dual learning mechanism to train sign-to-text (SLT) and text-to-sign (TSL) networks together. This approach performed better than singly-trained networks and achieved noteworthy improvement in

BLEU-1 to BLEU-4 scores in both translation directions. Hao et al. [29] proposed a Convolutional Vision Transformer (CvT) model combining hierarchical convolutional tokenization with transformer self-attention mechanisms. They evaluated the model on two publicly available sign language datasets; a digit dataset (1,712 images, 10 classes) and an alphabetsymbol dataset (87,000 images, 29 classes). The CvT model achieved 99% accuracy, outperforming traditional CNNs and competing vision transformer (ViT) models. In [30], the authors proposed a transformer-based SLT system that relies exclusively on 2D body key points extracted by MediaPipe Holistic instead of raw video frames or pixel embeddings. They introduced spatial–temporal augmentation on keypoints, convolutional embeddings and CTC loss with positional encoding for Sign2Gloss. They also fine-tuned BART for Gloss2Text. On Phoenix14T, they achieved competitive gloss-based BLEU-4 of 21.30 and gloss-free BLEU-4 of 15.70 (test set), while on How2Sign the gloss-free model reached only 3.87 BLEU-4. Although the keypoint-only pipeline significantly reduces computational cost and improves deployability, performance remains limited by the narrow weather-news domain of the datasets. Jin et al. [31] proposed a sign language production (SLP) method that employs Locality Sensitive Hashing-Clustering to automatically group similar embed ded word vectors. This helps the model capture long-distance dependencies in text-to-pose sequence generation. They also used multi-scale convolutional feature extraction module to capture local gloss features. Their transformer-based architecture was evaluated on the PHOENIX14T dataset and demonstrated improvements over baseline SLP methods. However, the model is restricted to a single high-resource dataset and focuses solely on the production direction, leaving the inverse G2S translation task and low-resource sign languages entirely unaddressed.

## 2.3. BdSL-Specific Research and Challenges

Research on BdSL recognition has progressed from CNN-based isolated alphabets to transformer-based real-time translation systems, establishing critical datasets and benchmarks for the Bangladeshi Deaf community. Authors in [10] trained a CNN on the Ishara Bachon digit dataset, achieving 95% accuracy for classifying BdSL digits. This model was among the first deep-learning models for BdSL and used grayscale digit images. The primary limitation was that it was tested only on static, isolated digit images from a small dataset. Hossen et al. [9] developed a transfer-learning deep convolutional neural network (DCNN) to recognize 37 static BdSL alphabet signs. Using 1,147 images of 31 signers, fine-tuning a pretrained network yielded 96.3% training accuracy but only 84.7% on validation. The mode showed signs of overfitting, as indicated by the substantial diference between its training and validation accuracies. In [32], the researchers applied transfer learning with YOLOv5 for BdSL recognition. They extracted frames from a large BdSL video collection and annotated 24.2 GB of data. Using pre-trained YOLOv5, the system achieved 91.6% average classification accuracy, but the performance depends on video frame quality and annotation density. Deep detectors were deployed on an edge device in [11]. Using the Okkhornama BdSL image set (3,760 images, 49 classes), their YOLOv7-Tiny model achieved real-time inference on Nano. The limitation was that the dataset was small, and modest mAP meant many classes were still missed. Raihan et al. [12] proposed a CNN with squeeze-and-excitation (CNN–SE) for BdSL recognition. They trained on the KU-BdSL alphabet dataset (1,500 images of 30 signs) and incorporated a smartphone app. This model achieved 99.86% test accuracy on KU-BdSL. The near-perfect accuracy likely reflects limited variability in KU-BdSL, real-world performance might drop under varied lighting or new signers. The authors in [13] constructed a custom BdSL dataset and proposed a deep LSTM-based BdSL translator that modeled temporal sign sequences using stacked LSTM layers without relying on gloss-level supervision. Their system achieved strong translation accuracy on constrained sentence vocabularies. However, the approach is limited by scalability issues inherent to LSTMs and does not leverage large pre-trained language models or Transformer-based attention mechanisms. In [8], researchers developed a real-time sign-to-text BdSL translator. They curated a 102-word dataset with 1,200 sentences and extracted hand/body keypoints. A custom Transformer model (with eficient landmark encoding) mapped sign sequences to Bangla text, reportedly achieving the highest accuracy to date on this benchmark. Key limitations are that the system uses a small, fixed vocabulary of everyday words and relies on robust landmark tracking.

Table 1: Evolution of SLR and SLT: from hand-centric to transformer-based approaches.
<table><tr><td>Study</td><td>Year</td><td>Approach</td><td>Dataset/Language</td><td>Key Achievement</td><td>Limitations</td></tr><tr><td colspan="6">Phase I: Hand-Centric &amp; Multi-Modal Foundations (2015-2019)</td></tr><tr><td>Kang et al. [15]</td><td>2015</td><td>CNN + Depth Maps</td><td>31 ASL alphabets/digits</td><td>99.99% (same signer), 83.58- 85.49% (cross-subject)</td><td>Isolated gestures only; no temporal continuity</td></tr><tr><td>Oberweger et al. [16]</td><td>2015</td><td>CNN for 3D Joints</td><td>Depth-based hand pose</td><td>10-20mm mean joint error</td><td>Requires depth calibration; fails with occlusions</td></tr><tr><td>Huang et al. [17]</td><td>2015</td><td>3D CNN Multi-Modal</td><td>RGB-Depth-Skeleton</td><td>94.2% accuracy; first multi- modal fusion</td><td>Static feature fusion; lim- ited to short sequences</td></tr><tr><td>Rastgoo et al. [18]</td><td>2018</td><td>RBM Fusion</td><td>Massey/NYU/ASL datasets</td><td>11-28% improvement over prior work</td><td>Superseded by LSTM-based methods</td></tr><tr><td>Wei et al. [19]</td><td>2019</td><td>3D ResNet + BiLSTM</td><td>CSL</td><td>95.2% precision; sentence-level translation</td><td>Rule-based grammar; lim- ited scalability</td></tr><tr><td colspan="6">Phase II: Transformer Revolution (2020-2023)</td></tr><tr><td>Camgöz et al. [27]</td><td>2020</td><td>Sign Transformers</td><td>PHOENIX-2014T (1,066 signs)</td><td>21.80 BLEU-4 vs. 9.58 prior SOTA</td><td>Gloss-dependent; computa- tionally intensive</td></tr><tr><td>Chaudhary et al. [28]</td><td>2023</td><td>SignNet II (Dual)</td><td>Bidirectional SLT-TSL</td><td>Outperforms unidirectional on all BLEU metrics</td><td>Requires parallel sign-text training data</td></tr><tr><td>Hao et al. [29]</td><td>2025</td><td>CVT</td><td>SL digits (10 classes), alphabets (29 classes)</td><td>99% accuracy; beats CNNs and ViTs</td><td>Tested on isolated static im- ages only</td></tr><tr><td colspan="6">Phase III: BdSL-Specific Research (2018-2025)</td></tr><tr><td>Hossen et al. [9]</td><td>2018</td><td>Transfer Learning DCNN</td><td>1,147 images, 31 signers (BdSL)</td><td>96.3% train, 84.7% validation</td><td>Severe overfitting; small dataset</td></tr><tr><td>Islam et al. [10]</td><td>2019</td><td>CNN</td><td>Ishara-Bachon digits</td><td>95% digit classification</td><td>Grayscale only; no dynamic gestures</td></tr><tr><td>Karim et al. [32]</td><td>2023</td><td>YOLOv5 Transfer</td><td>24.2GB BdSL video frames</td><td>91.6% average accuracy</td><td>Quality-dependent; annotation-intensive</td></tr><tr><td>Siddique et al. [11]</td><td>2023</td><td>YOLOv7-Tiny Edge</td><td>Okkhornama (3,760 images, 49 classes)</td><td>Real-time on Jetson Nano</td><td>Low mAP; many missed classes</td></tr><tr><td>Raihan et al. [12]</td><td>2024</td><td>CNN-SE</td><td>KU-BdSL (1,500 images, 30 signs)</td><td>99.86% test accuracy</td><td>Dataset overfitting; limited variability</td></tr><tr><td>Das et al. [13]</td><td>2025</td><td>Stacked LSTM</td><td>BdSL words (100 words, 500 videos)</td><td>89% test accuracy</td><td>LSTM scalability; no large language model (LLM) inte-</td></tr><tr><td>Dhrubo et al. [8]</td><td>2025</td><td></td><td>Transformer + Landmarks 102 words, 1,200 sentences (BdSL)</td><td>Highest reported BdSL accu- racy</td><td>gration Fixed vocabulary; landmark-dependent</td></tr></table>

The literature on SLT shows a clear trajectory as depicted in Table 1 from isolated sensorbased recognition and small static datasets toward transformer-based end-to-end translation, but critical gaps remain for low-resource languages. Especially, the work on BdSL has largely been confined to small, isolated image/video datasets. Most studies focus on sign-to-text translation in one direction. They do not use large pre-trained language models to support both gloss-to-sentence and sentence-to-gloss translation. Some studies use simple models to process sign sequences. Others use large multilingual models. These models may not work well on devices with limited resources.

Our work addresses these problems. We develop a bidirectional generative framework for Bangla Sign Language (BdSL). We use a Bidirectional and Autoregressive Transformer (BART) model and we pre-train the model on multiple large Bangla text corpus. The model is then fine-tuned for both G2S and S2G translation direction. For pre-training we use conditional language modeling with specific denoising learning objectives. Our approach goes further than simple sequence mapping and uses training method that is very similar to the structural properties of BdSL. Our approach also attemps to reduce vocabulary limits and overfitting in earlier LSTM and transformer models. We describe each step, from data preparation to model training and evaluation. Our primary goal is to build a lightweight system that can produce gloss-sentence translation with high quality in both directions.

## 3. Methodology

In this section, we discuss the proposed methodology for the successful development of a language model for a bidirectional gloss-sentence translation system targeted towards BdSL. The proposed methodology to research and develop the primary language model responsible for sign language gloss to sentence synthesis is depicted in Figure 1, and a step-by-step description of the methodology is provided in the following subsections.

## 3.1. Data Acquisition

## 3.1.1. Pre-training Dataset

The pre-training corpus was assembled from six complementary Bangla text collections: Sangraha [33], OSCAR (Bangla) [34], C4 (Bangla) [35], Bangla Wikipedia [36], Bangla books review collection [37], and a Bangla dialogues [38] dataset. These sources were selected to provide broad coverage of written and conversational registers, genres, and domain styles necessary for robust language modeling in downstream sign-language tasks. To maintain eficient I/O for large-scale training, each cleaned dataset was tokenized and serialized into contiguous binary files. Separate files were produced for training and validation splits to enable streaming consumption by standard pre-training toolchains while preserving document boundaries. All corpora were tokenized with the trained tokenizer prior to serialization. The tokenized corpora are stored as contiguous binary streams (one stream per dataset per split) to support deterministic, high-throughput sampling during pre-training. A SentencePiece byte pair encoding tokenizer was trained from scratch on a 60 GB Bangla corpus. The resulting tokenizer has a vocabulary of 52,001 subwords, averaging 1.33 tokens per word (127.5% of ideal tokenization eficiency) and a character-to-token compression ratio of 2.96.

## 3.1.2. Fine-tuning Dataset

We created the fine-tuning dataset for G2S and S2G tasks in BdSL. First, we used Sketch Engine [39] and the bntenten21 corpus to find common Bangla words. We analyzed word frequency and part-of-speech (POS) labels. We ranked the words by frequency and checked their POS labels. We then selected 100 words. The list includes verbs (30%), adjectives (20%), nouns (20%), adverbs (10%), pronouns (10%), and conjunctions (10%).

We wrote ten sentences for each word. These sentences covered present, past, and future tenses in a 4:3:3 ratio. We also included diferent sentence structures. A professional BdSL interpreter reviewed all sentence pairs. We mainly used the BanglaTense [40] and

![](images/00c6f1dd79277e48c2449cd2bda8224895765294334a9c8ce5447fb19525ff97.jpg)  
Figure 1: Proposed methodology for developing IsharaBART.

BanglaSarc [41] datasets to collect sentences. We modified some sentences to keep their grammar consistent and make them suitable for gloss annotation.

## 3.2. Data Augmentation

To address the lack of data in BdSL research, we used four complementary augmentation techniques to expand our annotated dataset of 1,000 gloss-sentence pairs. These methods were designed to introduce lexical, syntactic, and semantic variations while preserving the core G2S mappings essential for sequence-to-sequence learning. The augmented data was manually filtered for grammatical correctness and semantic coherence, which resulted in an augmented dataset size of 3,831 pairs and a total of 4831 pairs (annotated + augmented). The final dataset contains 1,000 annotated pairs (20.7%), 951 pairs from tense transformation (19.7%), 881 pairs from masked token substitution (18.2%), 999 pairs from back-translation (20.7%), and 1,000 pairs from one-shot prompting (20.7%). Each method contributes a similar number of pairs.

We use four methods to expand the annotated dataset. These methods are based on existing data augmentation techniques. We describe our changes below.

Rule-based tense transformation changes the tense of each Bangla sentence using predefined grammar rules. For example, it can change a present-tense sentence to future tense. We also update the matching gloss tokens to keep both sides aligned. Previous studies used rules to generate gloss data for sign language translation [5, 42]. However, they generated glosses from monolingual spoken text. Our method starts with an existing aligned sentence and gloss pair. It changes both sides of the pair. We designed the tense rules specif ically for this study. Masked token substitution masks one content word (noun, verb, or adjective) in each sentence and replaces it with the top grammatically correct prediction of BanglaBERT [43], updating the corresponding gloss token. This follows contextual augmentation, where words are replaced by those a language model predicts from context [44], and its use on gloss data with a masked language model [45]. Back-translation paraphrasing translates each sentence to English with NLLB-200 [46] (1.3B parameters) and back to Bangla with IndicTrans2 [47] (1B parameters), and keeps the original gloss. Back-translation is a standard augmentation method for low-resource translation [48], and round-trip paraphrases paired with the original gloss have been used for Korean Sign Language [45]. Back-translation that requires a trained gloss-to-text model is unreliable when gloss data is scarce [42]. Our round trip only uses pre-trained text translation models, so it needs no gloss data. Failed translations and paraphrases that changed the meaning were removed during manual filtering. One-shot prompt-based synthesis uses Llama 3.3-70B [49] to generate new gloss-sentence pairs. Each prompt contains a single annotated pair and asks for a new pair with altered content and a similar grammatical style (Table 2). Prompt ing an LLM with seed examples to create new training data follows [50] and has been used to augment data for low-resource languages [51]. Ours is a minimal one-example variant. We use a 70B model because this kind of generation is expected to work better with larger models [50]. Since such outputs can contain invalid examples [50, 51], all generated pairs were parsed and validated for BdSL compatibility. Table 3 provides illustrative examples from each technique.

Table 2: Model and prompt template used for one-shot prompting augmentation.
<table><tr><td>Model</td><td>Provider Details</td></tr><tr><td>llama-3.3-70b-versatile Groq API</td><td>70B parameters; temperature=0.85 for diversity; max_tokens=150; top_p=1</td></tr><tr><td colspan="2">Prompt Template</td></tr><tr><td colspan="2">I will provide you an example of a BdSL Gloss and its corresponding Bangla Sen- tence. Example Input: Sentence: {Example Sentence} Gloss: {Example Gloss}</td></tr></table>

Table 3: Examples of the augmentation strategies used to generate additional gloss–sentence pairs. Colored text marks the token substituted by each technique (red: original, blue: replacement).
<table><tr><td>Technique</td><td>Step-by-Step Process and Example</td></tr><tr><td rowspan="5">Rule-Based Trans- formation</td><td>Original:</td></tr><tr><td>Sentence:G</td></tr><tr><td>Gloss:可G</td></tr><tr><td>Step 1: Identify verb tense/mood ( - imperative present)</td></tr><tr><td>Step 2: Apply tense transformation rule (imperative → future) Step 3: Transform verb: → Step 4: Update gloss accordingly:→  </td></tr><tr><td rowspan="9"></td><td>Augmented:</td></tr><tr><td>Sentence:可</td></tr><tr><td></td></tr><tr><td>Gloss：GG</td></tr><tr><td>Original: Sentence:</td></tr><tr><td>GloSS：G可G</td></tr><tr><td></td></tr><tr><td>Step 1: Randomly select a content word ()</td></tr><tr><td>Step 2: Mask the token: [MASK] Step 3: Use BanglaBERT to predict: [MASK] → Step 4: Substitute in sentence and update gloss</td></tr><tr><td rowspan="9"></td><td>Augmented: Sentence:</td></tr><tr><td>Gloss：利</td></tr><tr><td></td></tr><tr><td>Original:</td></tr><tr><td>Sentence：</td></tr><tr><td>Step 1: Translate to English using NLLB200: &quot;I see him often.&quot;</td></tr><tr><td>Step 2: Translate back to Bangla using IndicTrans2:</td></tr><tr><td>可府R可 Step 3: Generate gloss for paraphrased sentence</td></tr><tr><td>Augmented: Sentence:(paraphrased)</td></tr><tr><td rowspan="7"></td><td>Glos：</td></tr><tr><td>Seed Example:</td></tr><tr><td>Sentence:</td></tr><tr><td></td></tr><tr><td>Gloss:GG</td></tr><tr><td>Step 1: Provide seed pair to LLM (Llama-3.3-70B)</td></tr><tr><td>Step 2: Prompt LLM</td></tr><tr><td rowspan="9"></td><td></td></tr><tr><td>Step 3: LLM generates by changing subject/object/verb</td></tr><tr><td></td></tr><tr><td>Generated:</td></tr><tr><td>Sentence：</td></tr><tr><td>Gloss：当</td></tr></table>

## 3.3. Dataset Validation

We first evaluate the quality of the fine-tuning dataset through statistical analysis. The parameters we mainly analyse are sentence length, gloss-to-text length ratios, semantic embeddings, and lexical diversity.

The annotated and augmented datasets have very similar lengths. The mean number of words in the annotated and augmented gloss-sentence samples are 5.45 $( S D = 1 . 3 7 )$ and 5.51 $( S D = 1 . 5 5 )$ respectively (Figure 2a). Kernel density estimation (KDE) [52] shows that both datasets have similar gloss-to-text length ratio (GTR). The GTR distribution peaks at about 1.0 (Figure 2b). We also project the BanglaBERT embeddings [43] of the samples in both dataset into two dimensions using UMAP [53]. The annotated and augmented samples show a centroid separation of 1.04 units. The UMAP also shows an overlap of 25.0% (Figure 2c). These indicates small overlapping but distinct semantic characteristics. Heaps’ Law [54], $V = K \cdot N ^ { \beta }$ , shows vocabulary growth. The estimated $\beta$ values are 0.622 for the annotated dataset (vocabulary: 1,027), 0.575 for the augmented dataset (1,859), and 0.553 for the combined dataset (Figure 2d).

## 3.3.1. Human Evaluation of Dataset

We performed a human quality check on 300 randomly sampled gloss–sentence pairs drawn from the annotated and augmented dataset. Two BdSL signers independently rated each pair for adequacy (0–100) and fluency (1=Bad, 2=Moderate, 3=Good). The resultant statistics are in Table 4. They both received professional BdSL training from Krishti, a non profit organization. Regarding their educational background, one evaluator was pursuing a secondary school qualification. The other evaluator had completed their higher secondary education. The evaluation was conducted in person. A trained sign language interpreter was present during the evaluation. The interpreter helped the evaluators communicate and understand the instructions. Before starting the evaluation, we explained two terms, i.e. adequacy and fluency, to the evaluators. The evaluators assigned separate scores for adequacy and fluency on their respective scales. We recorded the scores given by the evaluators. For each sample, we calculated the average separately for adequacy and fluency. No third-party adjudication or any tie-breaking method was used to resolve the disagreements between two evaluators. The mean adequacy is 99.21 (SD = 7.11) with a median of 100. 99.3% of samples reach the high-quality adequacy threshold $( \geq 7 5 \% )$ . Fluency checks are equally positive. The overall mean fluency is 2.99 (SD = 0.11) and 98.0% of items were rated “Good” by both evaluators. When considering both criteria, 97.7% of pairs satisfy adequacy $\ge 7 5 \%$ and fluency = Good. Inter-annotator agreement analysis supports the reliability of these ratings. Pearson correlation for adequacy is $r = 0 . 7 2 8 \ ( \mathrm { p } { < } 0 . 0 0 1 )$ and the mean absolute error is only 0.75 points on the 0–100 DA scale. For fluency, weighted Cohen’s $\kappa = 0 . 4 9 5$ with exact agreement 98.7% and adjacent agreement 100%, indicating moderate κ. The intraclass correlation (ICC) for adequacy is 0.465, consistent with significant consistency given the large sample.

## 3.4. Initial Model Selection

For the G2S and S2G synthesis tasks, we selected a compact encoder–decoder BART [55] model. The architecture was configured from scratch to be small (56M parameters) and monolingual for Bangla, rather than adopting large pre-trained checkpoints. BART (encoder–decoder, denoising) is explicitly designed for sequence-to-sequence tasks and the denoising training is particularly helpful for learning robust mappings between two modalities of text (glosses and sentences). Although pre-trained multilingual and English-centric checkpoints (e.g., large GPT or BART variants) ofer strong priors, we deliberately refrain from using them for this study because pre-trained multilingual models contain many languages and suboptimal tokenization/representations for Bangla. Training a compact, monolingual model from scratch allows us to tailor the vocabulary, subword segmentation, and inductive bias specifically to Bangla morphology and the glossing conventions used in our annotation pipeline. Large pre-trained models are memory and compute-intensive at inference, complicating deployment on resource-constrained hardware (edge devices, low-cost servers). Our target use-cases (interactive annotation tools and real-time synthesis) require small, lowlatency models. Table 5 provides a comprehensive summary of the architecture of the model selected.

![](images/577a682d41dbdc8c7461ceb3b1c1d1d6d80a8f3a242f16171ed0482eabc39cd8.jpg)

![](images/80a3d7ab100c5abea36c9c1f36bfba94d327108e6013fcb5e780b385c7684482.jpg)  
(a) Sentence length distributions.

![](images/98fc5cc0b0c6bd76f752f021d422202a60ec6e2f7f3939272f9c5f1dea3fb925.jpg)  
(b) Gloss-to-text length ratios.

![](images/5dc5a3e04d674343a900d4b41a6f0e9f53df2aedc0ef8bb5998ab9f35169bb18.jpg)  
(c) Semantic space projection.

![](images/9537c9c0fab56053663e650df987fab6f8db9420ce75c07cdf4306841f88f3f8.jpg)  
(d) Lexical diversity under Heaps’ Law.  
Figure 2: Dataset validation using (a) sentence length distributions, (b) gloss-to-text length ratios, (c) UMAP projections of BanglaBERT embeddings, and (d) Heaps’ Law vocabulary growth.

## 3.5. Pre-training

Pre-training was carried out from scratch on the six tokenized corpora. To prevent large sources from dominating the pre-training we employed temperature-controlled sampling [56] across the six corpora. Let $s _ { i }$ denote the (byte or token) size of corpus i and $S = \textstyle \sum _ { j } s _ { j }$ The base sampling probability is

$$
p _ { i } ~ = ~ { \frac { s _ { i } } { S } } ,\tag{1}
$$

Table 4: Human evaluation results for the annotated + augmented dataset (300 samples). <sup>a</sup>Adequacy $\ge 7 5 \%$ AND Fluency = Good (both evaluators). r = Pearson correlation, $\kappa _ { w } = \mathrm { W e i g h t e d }$ Cohen’s Kappa, $\rho =$ Spearman’s rho. MAE = Mean Absolute Error, Exact = Exact agreement percentage. $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1$ $^ { * * * } p < 0 . 0 0 1$
<table><tr><td>Metric</td><td></td><td>Evaluator 1 Evaluator 2</td><td>Overall</td><td>Agreement</td></tr><tr><td colspan="5">Adequacy</td></tr><tr><td>Mean ± SD</td><td> $9 8 . 8 3 \pm 8 . 0 9$ </td><td> $9 9 . 5 8 \pm 5 . 9 5$ </td><td> $9 9 . 2 1 \pm 7 . 1 1$ </td><td> $r = 0 . 7 2 8 ^ { * * * }$ </td></tr><tr><td>Median</td><td>100</td><td>100</td><td>100</td><td> $\mathrm { M A E } = 0 . 7 5$ </td></tr><tr><td>High Quality (≥ 75%)</td><td>298 (99.3%)</td><td>298 (99.3%)</td><td>298 (99.3%)</td><td></td></tr><tr><td colspan="5">Fluency</td></tr><tr><td> $\mathrm { M e a n } \pm \mathrm { S D }$ </td><td> $2 . 9 8 \pm 0 . 1 4$ </td><td> $2 . 9 9 \pm 0 . 0 8$ </td><td> $2 . 9 9 \pm 0 . 1 1$ </td><td> $\kappa _ { w } = 0 . 4 9 5$ </td></tr><tr><td>Median</td><td>3.0</td><td>3.0</td><td>3.0</td><td> $\rho = 0 . 5 7 3 ^ { \ast \ast \ast }$ </td></tr><tr><td colspan="5">Fluency Distribution</td></tr><tr><td>Bad (1)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td>0 (0.0%)</td><td></td></tr><tr><td>Moderate (2)</td><td>6 (2.0%)</td><td>2 (0.7%)</td><td>8 (1.3%)</td><td></td></tr><tr><td>Good (3)</td><td>294 (98.0%)</td><td>298 (99.3%)</td><td>592 (98.7% of 600 ratings) Exact: 98.7%</td><td></td></tr><tr><td colspan="5">Overall Quality Assessment</td></tr><tr><td>Both criteria metª</td><td>293 (97.7%)</td><td></td><td>Adeq-Flu corr.</td><td> $\rho = 0 . 3 9 1 ^ { \ast \ast \ast }$ </td></tr></table>

and the temperature-adjusted probability used to draw training blocks

$$
\tilde { p } _ { i } = \frac { p _ { i } ^ { 1 / T } } { \sum _ { j } p _ { j } ^ { 1 / T } } ,\tag{2}
$$

where $T > 0$ is the sampling temperature (higher T flattens the distribution and increases sampling from smaller corpora).

## 3.5.1. General Pre-training Framework

Let $X = ( x _ { 1 } , x _ { 2 } , \dots , x _ { T } )$ denote the original Bangla token sequence sampled from the tokenized pre-training data D, where $x _ { t } \in \mathcal V$ is a token from the vocabulary. We define a stochastic corruption function $C : \mathcal { V } ^ { T }  \mathcal { V } ^ { T ^ { \prime } }$ , which is a composite function of documentlevel and token-level noising that transforms X into a noisy version $\tilde { X }$ . The IsharaBART model parameterized by $\theta ,$ is trained as a denoising autoencoder to reconstruct the original X given $\tilde { X }$ . The training objective minimizes the negative log-likelihood

$$
\mathcal { L } ( \boldsymbol { \theta } ) = - \mathbb { E } _ { \boldsymbol { x } \sim \mathcal { D } , \tilde { \boldsymbol { x } } \sim C ( \boldsymbol { x } ) } \left[ \sum _ { t = 1 } ^ { | \boldsymbol { x } | } \log P ( x _ { t } \mid \tilde { \boldsymbol { x } } , \boldsymbol { x } _ { < t } ; \boldsymbol { \theta } ) \right]\tag{3}
$$

Figure 3 illustrates the general pre-training framework of the initial model with a stepby-step depiction of the full workflow.

## 3.5.2. Document-Level Noising

Sentence permutation. To train the model to learn logical discourse and sentence relationships, we first apply sentence permutation. This is applied with probability 1.0 [55], shuffling sentences to disrupt global structure while preserving local coherence. Let the input sequence X be segmented into M sentences based on Bangla-specific delimiters (period and question mark). We define the segmented set as $\mathcal { S } = \{ S _ { 1 } , S _ { 2 } , \ldots , S _ { M } \}$ . We sample a random permutation $\sigma \in \Sigma _ { M }$ (the set of all permutations of length M) with probability $p _ { s e n t }$ The document-level corrupted sequence $\tilde { X } _ { d o c }$ is obtained by concatenating the permuted sentences

Table 5: Model configuration used for the gloss sequence to sentence modeling.
<table><tr><td>Property</td><td>BART</td></tr><tr><td>Model family</td><td>Seq2Seq denoising LM</td></tr><tr><td>Architecture</td><td>Encoder + decoder</td></tr><tr><td>Encoder layers</td><td>4</td></tr><tr><td>Decoder layers</td><td>4</td></tr><tr><td>Hidden dimension</td><td>512</td></tr><tr><td>Attention heads</td><td>8 / 8 (enc / dec)</td></tr><tr><td>FFN dimension</td><td>2048</td></tr><tr><td>Context size</td><td>512</td></tr><tr><td>Vocabulary</td><td>52001</td></tr><tr><td>Approx. parameters</td><td>~56M</td></tr><tr><td>Training style</td><td>Denoising / conditional generation</td></tr></table>

$$
\tilde { X } _ { d o c } = S _ { \sigma ( 1 ) } \oplus S _ { \sigma ( 2 ) } \oplus \cdots \oplus S _ { \sigma ( M ) }\tag{4}
$$

where ⊕ denotes sequence concatenation.

## 3.5.3. Token-Level Noising

After sentence permutation, we apply one of four distinct token-level noise transformations to $\tilde { X } _ { d o c }$ . Let $Z = ( z _ { 1 } , \ldots , z _ { L } ) $ be the tokens of $\tilde { X } _ { d o c } .$ . We select a transformation $f \in \{ f _ { i n f i l l } , f _ { d e l } , f _ { m a s k } , f _ { p e r m } \}$ based on a categorical distribution $\lambda _ { c h o i c e } = [ 0 . 3 , 0 . 2 , 0 . 2 , 0 . 3 ]$

Text infilling $( f _ { i n f i l l } )$ . This objective teaches the model to predict spans of missing text. We sample a set of span lengths from a Poisson distribution, where the length $l \sim \mathrm { P o i s s o n } ( \lambda = 3 )$ For a chosen span of tokens starting at index i with length l, denoted as $z _ { i : i + l - 1 }$ , we replace the entire span with a single unique mask token [MASK]. The corrupted sequence is

$$
\tilde { X } = ( \dots , z _ { i - 1 } , [ M A S K ] , z _ { i + l } , \dots )\tag{5}
$$

Unlike standard single token masking, a span of length l is replaced by exactly one mask token, forcing the model to learn how many tokens are missing.

Token deletion $( f _ { d e l } )$ . To improve the model’s robustness against dropped inputs (common in SLT), we delete tokens without leaving a trace. Given a sequence of length L, we calculate the number of tokens to remove as $N = \lfloor p _ { d e l } \cdot L \rfloor$ . We uniformly sample a subset of indices $\mathcal { T } _ { d e l } \subset \{ 1 , \ldots , L \}$ of size N without replacement. The corrupted sequence is formed by retaining only the tokens whose indices are not in the deletion set

$$
\tilde { X } = ( x _ { t } \mid t \not \in \mathcal { T } _ { d e l } )\tag{6}
$$

![](images/1055210fb2e099ad1da9efda9359cb479c167af13e99e0630ddfdff821ee1a21.jpg)  
Figure 3: Proposed framework for the pre-training with denoising objectives.

The model learns to identify positions where inputs are missing and insert the correct tokens.

Token masking $( f _ { m a s k } )$ . Following the standard single token masking objective, we randomly sample a set of indices M with probability $p _ { m a s k }$ . Tokens at these indices are replaced with the [MASK] token

$$
{ \tilde { x } } _ { t } = { \left\{ \begin{array} { l l } { [ M A S K ] } & { { \mathrm { i f ~ } } t \in { \mathcal { M } } } \\ { z _ { t } } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. }\tag{7}
$$

Word permutation $( f _ { p e r m } )$ . To model the flexible word order often found in sign language gloss sequence, we introduce an additional noising technique to shufle words within sentence boundaries. For a given sentence $S _ { j } ~ \subset ~ Z$ consisting of tokens $( w _ { 1 } , \ldots , w _ { k } )$ , we sample a random permutation π of $\{ 1 , \ldots , k \}$ . The sentence is transformed to

$$
\tilde { S } _ { j } = ( w _ { \pi ( 1 ) } , w _ { \pi ( 2 ) } , \ldots , w _ { \pi ( k ) } )\tag{8}
$$

Crucially, punctuation at the end of sentence is excluded from the shufle to preserve sentence boundaries.

## 3.6. Fine-tuning

After pre-training, we fine-tune the pre-trained weights of IsharaBART on two separate downstream tasks: G2S translation and S2G translation, enabling bidirectional translation capability. For ablation, we employ two methods of fine-tuning: full parameter fine-tuning and LoRA [57]. For G2S task, we fine-tune by minimizing the negative log-likelihood of the target Bangla sentence conditioned on the corresponding input gloss sequence and previously generated tokens, following standard sequence-to-sequence training for encoder-decoder language models. For the S2G task, the input is the Bangla sentence while the target becomes the corresponding gloss sequence. Given a parallel dataset $\mathcal { D } _ { \mathrm { f i n e t u n e } }$ of gloss-sentence pairs $( x , y )$ , the model parameters $\theta$ are optimized by minimizing the expected token-level loss

$$
\mathcal { L } _ { \mathrm { f i n e t u n e } } ( \theta ) = - \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { \mathrm { f i n e t u n e } } } \left[ \sum _ { i = 1 } ^ { | y | } \log P ( y _ { i } | \ x , y _ { < i } ; \theta ) \right]\tag{9}
$$

where x denotes the input token sequence, $y = ( y _ { 1 } , \dots , y _ { | y | } )$ the corresponding target token sequence, and $P ( y _ { i } \mid x , y _ { < i } ; \theta )$ the model’s conditional probability of the i-th target token given the source and all preceding target tokens.

## 4. Experimental Setup

We evaluate IsharaBART on bidirectional G2S and S2G translation tasks for BdSL, assessing the impact of data augmentation and comparing against strong multilingual and monolingual baselines. Our experiments are designed to answer two key questions: (1) Does pre-training a significantly smaller BART model with denoising objectives on monolingual Bangla text improve performance on low-resource downstream task of gloss translation? (2) How does synthetic data augmentation afect model generalization in this specialized domain?

## 4.1. Downstream Tasks

We fine-tune all models on two complementary translation directions. G2S takes a gloss sequence representing sign language structure as input and generates the corresponding natural Bangla sentence. S2G is the inverse task, mapping natural Bangla text to gloss sequences.

## 4.2. Baseline Models

We compare IsharaBART against four strong baseline architectures representing diverse pre-training objectives, architectural choices. mBART-50 [58] is a multilingual denoising autoencoder with an encoder–decoder Transformer architecture and approximately 611M parameters in the mBART-large-50 configuration, pre-trained on 50 languages including Bangla using a BART-style denoising objective, making it a natural baseline for our monolingual BART-style pre-training while controlling for architecture and objective but varying pre-training language coverage. mT5-Small [59] is a multilingual text-to-text Transformer with an encoder–decoder architecture and roughly 300M parameters, pre-trained on a massively multilingual corpus using a span-corruption objective in the T5 framework. BanglaT5 [38] is a monolingual adaptation of the T5 architecture pre-trained exclusively on Bangla corpora, with approximately 247M parameters, serving as a direct monolingual baseline that contrasts our BART-style denoising pre-training with a T5-style span-corruption objective under a Bangla-only setting and at a substantially larger parameter scale than IsharaBART. BanglaGPT [60] is a decoder-only autoregressive Transformer model for Bangla with 124M trainable parameters, corresponding to a GPT-2-style medium configuration adapted to Bangla, enabling us to assess how a pure generative pre-training paradigm compares to encoder-decoder architectures on gloss translation at a smaller parameter scale.

## 4.3. Experimental Conditions

IsharaBART is pre-trained on the six monolingual Bangla corpora using the denoising autoencoder objective described in Section 3. We conduct four fine-tuning experiments for each of the five models: G2S and S2G, each on annotated data only and on annotated + augmented data, yielding a $5 \times 2 \times 2$ design (20 fine-tuned models) that enables comprehensive analysis of data augmentation efects and task asymmetry across all five architectures. In addition, for IsharaBART alone, we separately ablate monolingual pre-training by comparing the pre-trained model against an identical architecture trained from scratch, isolating the contribution of pre-training independent of the cross-model comparison above. Detailed hyperparameters are provided in Appendix A.

## 4.4. Evaluation Metrics

We report recall-oriented understudy for gisting evaluation (ROUGE-1/2/L) [61], BLEU-$1 / 2 / 3 / 4$ [62], and crosslingual optimized metric for evaluation of translation (COMET) [63] scores as our primary evaluation metrics. ROUGE checks how much of the critical information from human-written text is successfully captured in the generated text. BLEU evaluates how many words or phrases in the generated text match reference text. We complement them with COMET, which evaluates the semantic meaning and contextual quality of the generated text. Model selection is performed based on BLEU-4 on the validation set, and all reported results are on the held-out test set.

## 5. Results and Analysis

The experiments are designed to isolate three questions: whether monolingual Bangla pre-training benefits low-resource gloss–sentence translation, whether synthetic augmentation improves generalization, and whether the resulting model ofers a favorable accuracy– eficiency trade-of. Across the controlled comparisons, three consistent findings emerge. First, monolingual pre-training substantially improves performance when only 1,000 annotated pairs are available and reduces the number of epochs required to reach peak validation BLEU-4. Second, augmentation improves translation quality in both directions, with a particularly large efect on S2G. Third, the 56M-parameter IsharaBART remains competitive with substantially larger baselines, achieving its strongest relative advantage on the bidirectional S2G task.

## 5.1. Pre-training Ablation and Convergence

We first isolate the contribution of monolingual Bangla pre-training by comparing IsharaBART initialized from the proposed pre-trained checkpoint with the same architecture trained from scratch under identical downstream conditions.

The pre-training of IsharaBART, which was conducted on monolingual Bangla text, showed fast convergence. This is shown in Figure 4. Convergence was achieved around 5,000 steps, as evidenced by the plateau in both losses. Further iterations yielded diminishing returns (<0.1 reduction per 1,000 steps). The pre-training was conducted for a total of 22.3 GPU hours on an NVIDIA A100 (80GB) with an efective batch size of 256. Given the limitation of computational resource, going beyond 30,000 steps was deemed impractical. It would have exceeded available resources without significant gains in performance.

![](images/4e572d94b257d031bb49e9cc99aebc36cc4f5763d468d3d69e76f8c31bf32aaf.jpg)  
Figure 4: IsharaBART pre-training loss convergence over 30k steps. Training loss (blue) and validation loss (red) exhibit a rapid descent through 5k iterations, stabilizing with negligible improvement.

![](images/f99289bdd5bedd5d76e14dc73ce53ccd5ea4193e3884759a3c3613500fe3f0f5.jpg)

![](images/2d3201cea5b10c7b430848bd633961a87c93db0e6df857aa2d3ece1e30141889.jpg)  
Figure 5: BLEU-4 convergence curves comparing pre-trained IsharaBART against from-scratch baselines across annotated and augmented datasets for bidirectional translation tasks. Stars indicate peak performance. The dotted horizontal line marks the best pre-trained performance for reference.

On the annotated dataset, pre-training increases G2S BLEU-4 from 19.83 to 43.30 and S2G BLEU-4 from 23.80 to 39.39, corresponding to relative improvements of 118.4% and 65.5%, respectively. The gains remain after augmentation: BLEU-4 increases from 45.30 to 51.74 for G2S and from 55.10 to 61.40 for S2G. Thus, pre-training is most valuable when parallel gloss-sentence data is scarce, but its benefit continues even after the training set is expanded.

Pre-training also reduces the number of epochs required to reach peak validation BLEU-4 (Figure 5, Figure 6). On the annotated data, G2S reaches peak performance in 5 epochs compared with 52 for the scratch model, while S2G requires 15 rather than 66 epochs, corresponding to 90.4% and 77.3% fewer epochs. With augmented data, the corresponding reductions are 62.5% for G2S (6 vs. 16 epochs) and 70.3% for S2G (11 vs. 37 epochs). These results indicate that monolingual Bangla pre-training improves both the quality and the

![](images/fed1d97ef60eb2753f3c980e9a9b98e55999119af9b67cd8c6712c73370ed75f.jpg)  
Figure 6: Multi-dimensional performance comparison between pre-trained IsharaBART and from-scratch baselines. The charts visualize five key metrics normalized to a 0–100 scale: BLEU-4, COMET, ROUGE-L, Quality (min-max normalized inverse loss), and Eficiency (percentage reduction in training epochs).

sample eficiency of downstream adaptation.

## 5.2. G2S Translation

On the annotated dataset, BanglaT5 achieves the highest scores (Table 6). It gets 48.91 BLEU-4, 0.95 COMET, and 81.71 ROUGE-L. IsharaBART gets 43.30 BLEU-4, 0.93 COMET, and 78.32 ROUGE-L. However, IsharaBART has only 56M parameters, while BanglaT5 has 247M.

Data augmentation reduces this gap. IsharaBART reaches 51.74 BLEU-4 and 0.95 COMET after augmentation. BanglaT5 reaches 52.24 BLEU-4. The diference is only 0.50 BLEU-4 points. IsharaBART also uses 77.3% fewer parameters. These results show that IsharaBART can achieve scores close to those of a much larger model with less computational capacity.

Augmentation raises IsharaBART’s G2S BLEU-4 score from 43.30 to 51.74. This is an increase of 8.44 points. The baseline models also improve after augmentation. This suggests that the added data helps diferent models, not just IsharaBART. The results also show that synthetic data can help reduce the performance gap between smaller and larger models.

We also present qualitative examples from the test set to show the performance of IsharaBART on the G2S translation task. These examples are tabularized in Table 7.

## 5.3. S2G Translation

The S2G results show a diferent pattern (Table 8). On the annotated dataset, IsharaBART achieves the highest BLEU-4 (39.39) and COMET (0.94). It closely surpasses BanglaT5 at 39.29 BLEU-4 and 0.93 COMET. After augmentation, IsharaBART reaches 61.40 BLEU-4, the highest score among all five models, while mBART-50 reaches 60.82 and BanglaT5 51.57. IsharaBART therefore outperforms the 611M-parameter mBART-50 by 0.58 BLEU-4 while using approximately 91% fewer parameters.

The efect of augmentation is particularly strong for S2G. IsharaBART improves from 39.39 to 61.40 BLEU-4, a gain of 22.01 points. This is considerably larger than the corresponding G2S gain of 8.44 points. This suggests that additional lexical and structural variation is especially valuable when learning the more specialized mapping from natural Bangla sentences to gloss sequences. The diference between the two directions also indicates that S2G remains the more dificult task for parameter-eficient fine-tuning. We also present qualitative examples from the test set that are tabularized in Table 9.

Table 6: Translation results for G2S on the annotated and augmented datasets.
<table><tr><td>Model</td><td>Params</td><td>R-1</td><td>R-2</td><td>R-L</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>COMET</td></tr><tr><td colspan="8">G2S – Annotated Dataset (1,000 pairs)</td></tr><tr><td>mBART-50</td><td>611M</td><td>77.86</td><td>63.36</td><td>76.63</td><td>65.15</td><td>54.64</td><td>46.39</td><td>39.93</td><td>0.92</td></tr><tr><td>mT5-Small</td><td>300M</td><td>77.76</td><td>62.37</td><td>76.20</td><td>63.78</td><td>52.78</td><td>44.76</td><td>38.76</td><td>0.90</td></tr><tr><td>BanglaT5</td><td>247M</td><td>82.94</td><td>70.65</td><td>81.71</td><td>72.31</td><td>62.84</td><td>55.16</td><td>48.91</td><td>0.95</td></tr><tr><td>BanglaGPT</td><td>124M</td><td>61.27</td><td>45.16</td><td>60.76</td><td>41.88</td><td>31.69</td><td>23.83</td><td>17.82</td><td>0.78</td></tr><tr><td>IsharaBART (ours)</td><td>56M</td><td>79.86</td><td>65.22</td><td>78.32</td><td>72.73</td><td>59.29</td><td>50.05</td><td>43.30</td><td>0.93</td></tr><tr><td colspan="10">G2S – Augmented Dataset (4,831 pairs)</td></tr><tr><td>mBART-50</td><td>611M</td><td>81.33</td><td>69.33</td><td>80.40</td><td>71.37</td><td>62.36</td><td>55.25</td><td>49.12</td><td>0.94</td></tr><tr><td>mT5-Small</td><td>300M</td><td>83.28</td><td>72.06</td><td>82.25</td><td>73.62</td><td>64.77</td><td>57.56</td><td>51.78</td><td>0.94</td></tr><tr><td>BanglaT5</td><td>247M</td><td>83.74</td><td>72.78</td><td>82.60</td><td>74.38</td><td>65.40</td><td>58.20</td><td>52.24</td><td>0.95</td></tr><tr><td>BanglaGPT</td><td>124M</td><td>72.36</td><td>57.72</td><td>71.29</td><td>56.87</td><td>46.32</td><td>37.97</td><td>31.26</td><td>0.88</td></tr><tr><td>IsharaBART (ours)</td><td>56M</td><td>82.22</td><td>70.51</td><td>80.95</td><td>77.97</td><td>66.11</td><td>57.98</td><td>51.74</td><td>0.95</td></tr></table>

IsharaBART demonstrates that competitive or superior performance is achievable with far fewer parameters (56M) than standard multilingual models (300-611M) when pre-training is carefully targeted to the task language. However, the task-dependent behavior suggests that no single architecture or pre-training strategy uniformly excels across all aspects of bidirectional gloss translation, and that the choice of model should be guided by the specific deployment scenario.

## 5.4. Human Evaluation

We conducted a human evaluation of IsharaBART in both G2S and S2G translation settings by the same two BdSL signers described in Section. They independently assessed 50 randomly sampled outputs for adequacy and fluency using a previously defined five point rating scale.A professionally trained sign language interpreter assisted us to facilitate communication. We conducted the evaluation in person and blind to the identity of the model. We calculated the mean of the two ratings for each criterion and evaluated sample. The results are summarized in Table 10. The results therefore reflect the average of the two evaluators’ individual assessments.

Overall, IsharaBART demonstrates strong semantic quality across both directions. The system achieves a high mean adequacy score of 88.8, with a median of 100 and 84% of samples rated as high quality $( \geq 7 5 \% )$ (Table 10). Similarly, S2G translation achieves a mean adequacy of 87.8, with 76% of outputs meeting the high-quality threshold. These results indicate that the model consistently preserves the core meaning of BdSL inputs.

In both directions, the median fluency is 3 (Good), with 77% of G2S and 73% of S2G outputs rated as fluent. The slightly lower fluency in S2G translation is expected. Gloss generation is more constrained and sensitive to ordering and deletion. Importantly, adequacy– fluency correlations are strong and statistically significant in both settings $( \rho = 0 . 6 9 4$ for

Table 7: Qualitative G2S translation examples across three performance levels. English translations of input glosses are approximate and shown in uppercase.
<table><tr><td>Input (Gloss)</td><td>Ground Truth (Bangla)</td><td>Predicted (Bangla)</td></tr><tr><td colspan="3">Excellent Performance</td></tr><tr><td>(YOU YOU SLOWLY SLOWLY GROW WILL, GOOD (You will try to become a good person when you grow up.)</td><td></td><td></td></tr><tr><td>PERSON WORK WORK TRY WILL)</td><td></td><td>(You will try to become a good person when you grow up.)</td></tr><tr><td>响可可,丽G (WHEN YOU SLOWLY SLOWLY GROW WILL,</td><td>]不,可何可床丽?</td><td>何啊不,的何响质网?</td></tr><tr><td>THEN YOU WORK WHAT WILL)</td><td>(When you grow up, what will you do?)</td><td>(When you grow up, what will you do?)</td></tr><tr><td>可夜啊何可</td><td>可夜</td><td>可可</td></tr><tr><td>(TEACHER STUDENT NEW BOOK READ TELL NOT)</td><td>(The teacher did not tell the students to read the new book.)</td><td>(The teacher did not tell the students to read the new book.)</td></tr><tr><td>可可</td><td>可可成可</td><td>可可能可</td></tr><tr><td>(MY HAND ONE LETTER HAVE)</td><td>(I have a letter.)</td><td>(I have a letter.)</td></tr><tr><td>(STUDENT PLAY FOR FIELD GOING-PAST)</td><td></td><td></td></tr><tr><td></td><td>(The students were going to the field to play.)</td><td>(The students were going to the field to play.)</td></tr><tr><td colspan="3">Moderate Performance</td></tr><tr><td>可可國</td><td>四可</td><td>可可可可</td></tr><tr><td>(THEN SKY CLOUD CLOUD WAS NOT)</td><td>(The sky was not cloudy at that time.)</td><td>(The sky was not dark at that time.)</td></tr><tr><td>RRRG</td><td></td><td>UG</td></tr><tr><td>(YOU HIS/HER ABOUT UNDERSTAND EASY) 河同國司河司爾國國</td><td>(You understand him/her well.)</td><td>(You used to understand him/her well.)</td></tr><tr><td>(WHEN HE/SHE CRY WAS, THEN I HIS/HER</td><td>何，可可 (When he/she was crying, I was holding his/her hand.)</td><td>网，可可可 (When he/she was crying, then I was holding his/her hand.)</td></tr><tr><td>HAND HOLD WAS)</td><td></td><td></td></tr><tr><td>(I CLASS FINISH AFTER EAT WILL)</td><td></td><td>可河啊啊</td></tr><tr><td></td><td>(Will eat after class.) </td><td>(I will eat after finishing class.) </td></tr><tr><td>(NAFIS HIS/HER BEFORE HABIT CHANGE WANT (Nafis wanted to change his habit.) WAS)</td><td></td><td>(Nafis wanted to change his habit.)</td></tr><tr><td colspan="3">Poor Performance (Limitations)</td></tr><tr><td></td><td>, </td><td>,</td></tr><tr><td>(FIRST I GO AFTER STUDY SIT)</td><td>(First, I went, then sat down to study.)</td><td>(First, I studied, then sat down to study.)</td></tr><tr><td>可不响</td><td>可何可响</td><td>可响响</td></tr><tr><td>(I NEWS PAPER READ) </td><td>(I read the newspaper.)</td><td>(I read a letter.)</td></tr><tr><td></td><td></td><td></td></tr><tr><td>(I WHO FRIEND THINK WAS, HE ONE FRAUD) </td><td>(The friend I thought I knew was actually a fraud.) </td><td>(The friend whom I consider a fraud.) </td></tr><tr><td>(YOU CAMPUS THERE ALONE WALK WILL)</td><td>(You will go to the campus alone on foot.)</td><td>(You will go to campus alone.)</td></tr><tr><td></td><td></td><td>可可</td></tr><tr><td>(YOU RAIN STOP AFTER WORK START DO)</td><td>(You will start working after the rain stops.)</td><td>(You will start working after stopping the rain.)</td></tr></table>

G2S and $\rho = 0 . 6 3 6$ for S2G). This suggests that semantically accurate translations are also perceived as more fluent by human evaluators. Inter-annotator agreement ranges from moderate to substantial, with Pearson correlations of 0.574 (G2S) and 0.450 (S2G) for adequacy, and weighted kappa values of 0.364 and 0.443 for fluency. IsharaBART satisfies both criteria for 66% of G2S outputs and 56% of S2G outputs when jointly considering adequacy and fluency.

## 5.5. Fine-tuning Ablation Study

To evaluate parameter-eficient fine-tuning for SLT, we compared full fine-tuning against LoRA across all experimental configurations. Table 11 presents comprehensive results.

LoRA provides a significant parameter-eficiency advantage for IsharaBART. LoRA reduces the number of trainable parameters to approximately 0.7% of the full model. However, this reduction in trainable parameters comes with a consistent decrease in translation performance across all evaluated settings. As shown in Table 11, BLEU-4 decreases by 28.0– 52.2% relative to full fine-tuning. The degradation is particularly significant for S2G, where on annotated data performance decreases from 39.39 to 22.15 BLEU-4 (-43.8%). On the augmented setting BLEU-4 decreases from 61.40 to 29.37 (-52.2%). In contrast, the corresponding G2S reductions are smaller, at 28.0% and 29.8%, respectively. Thus, the results indicate a clear accuracy–parameter-eficiency trade-of, with the cost of parameter-eficient adaptation depending strongly on the translation direction.

This behavior is consistent with recent empirical evidence that LoRA does not necessarily recover the performance of full fine-tuning when the downstream task requires substantial parameter adaptation. Under standard low-rank configurations, LoRA substantially underperforms full fine-tuning in several adaptation settings, and the parameter updates learned by full fine-tuning have ranks 10–100 times larger than typical LoRA configurations [64]. LoRA and full fine-tuning produce structurally diferent modifications to pretrained weights [65]. Vanilla LoRA can retain substantially less task-specific translation performance than full fine-tuning. This indicates that the LoRA constraint may prevent the model from making suficiently expressive parameter updates [66, 67].

Table 8: Translation results for S2G on the annotated and augmented datasets.
<table><tr><td>Model</td><td>Params</td><td>R-1</td><td>R-2</td><td>R-L</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>COMET</td></tr><tr><td colspan="8">S2G – Annotated Dataset (1,000 pairs)</td></tr><tr><td>mBART-50</td><td>611M</td><td>78.84</td><td>63.35</td><td>76.74</td><td>70.55</td><td>56.89</td><td>45.96</td><td>37.03</td><td>0.92</td></tr><tr><td>mT5-Small</td><td>300M</td><td>74.88</td><td>60.78</td><td>73.60</td><td>65.35</td><td>52.81</td><td>43.23</td><td>35.96</td><td>0.87</td></tr><tr><td>BanglaT5</td><td>247M</td><td>80.06</td><td>65.41</td><td>78.54</td><td>73.60</td><td>60.20</td><td>49.09</td><td>39.29</td><td>0.93</td></tr><tr><td>BanglaGPT</td><td>124M</td><td>58.60</td><td>43.38</td><td>58.01</td><td>45.97</td><td>33.23</td><td>23.54</td><td>16.19</td><td>0.77</td></tr><tr><td>IsharaBART (ours)</td><td>56M</td><td>77.16</td><td>63.01</td><td>75.78</td><td>69.08</td><td>56.71</td><td>47.18</td><td>39.39</td><td>0.94</td></tr><tr><td colspan="10">S2G – Augmented Dataset (4,831 pairs)</td></tr><tr><td>mBART-50</td><td>611M</td><td>84.65</td><td>76.10</td><td>83.78</td><td>80.91</td><td>72.60</td><td>66.16</td><td>60.82</td><td>0.94</td></tr><tr><td>mT5-Small</td><td>300M</td><td>83.70</td><td>74.85</td><td>82.78</td><td>79.67</td><td>71.19</td><td>64.42</td><td>58.92</td><td>0.93</td></tr><tr><td>BanglaT5</td><td>247M</td><td>82.46</td><td>71.65</td><td>81.01</td><td>77.35</td><td>66.82</td><td>58.42</td><td>51.57</td><td>0.94</td></tr><tr><td>BanglaGPT</td><td>124M</td><td>71.87</td><td>58.75</td><td>70.54</td><td>63.20</td><td>51.25</td><td>42.75</td><td>35.96</td><td>0.88</td></tr><tr><td>IsharaBART (ours)</td><td>56M</td><td>84.24</td><td>75.67</td><td>83.07</td><td>81.06</td><td>72.93</td><td>66.55</td><td>61.40</td><td>0.94</td></tr></table>

The diference in performance for diferent translation direction provides an additional insight. LoRA performs considerably better for G2S than for S2G. This suggests that the required task-specific adaptation may difer between the two directions. A plausible explanation is that G2S maps gloss sequences to natural Bangla sentences, an output space that is closer to the natural-language used during pre-training. In contrast, S2G maps natural Bangla sentences into gloss sequences, which has a more specialized and structurally dificult representation.

## 5.6. Dataset Comparison

To compare our dataset within the broader landscape of SLT resources, we compare it with prominent benchmarks and recent BdSL datasets. Table 12 summarizes key attributes such as size, domain, and SOTA performance metrics for G2S and S2G tasks where available. Our dataset has a modest scale of approximately 4,800 pairs (including augmentation). It achieves superior BLEU-4 scores (51.7 for G2S and 61.4 for S2G in augmented data) compared to larger benchmarks. This highlights the eficiency of our approach in low-resource settings, particularly for under-resourced languages like BdSL. In terms of BdSL, datasets like IsharaKhobor and Bangla-SGP provide valuable but domain-limited parallels.

## 5.7. Discussion

The results show that monolingual Bangla pre-training and data augmentation can improve bidirectional gloss-sentence translation. However, IsharaBART does not outperform all baselines on every metric. On G2S, BanglaT5 achieves higher BLEU-2, BLEU-3, and

Table 9: Qualitative S2G translation examples across three performance levels. English translations of input sentences are shown below the Bangla, and gloss translations are approximate uppercase renderings.
<table><tr><td>Input (Bangla)</td><td>Ground Truth (Gloss)</td><td>Predicted (Gloss)</td></tr><tr><td colspan="3">Excellent Performance 网不可不可喉可厕</td></tr><tr><td>G可GR (He/she had finished the work before even starting it.)</td><td>(HE/SHE WORK HAND START BEFORE WORK FINISHED</td><td>可可不可顺可 (HE/SHE WORK HAND START BEFORE WORK FINISHED</td></tr><tr><td>不可夜可可 (The teacher will not ask the students to read the new</td><td>WAS) 四夜啊河可 (TEACHER STUDENT NEW BOOK READ TELL NOT)</td><td>WAS) 可啊夜啊可 (TEACHER STUDENT NEW BOOK READ WILL NOT)</td></tr><tr><td>book.) T (The doctor will not prescribe any new medicine.)</td><td>(DOCTOR ANY NEW MEDICINE PRESCRIBE GIVE NOT)</td><td>T (DOCTOR ANY NEW MEDICINE PRESCRIBE GIVE NOT)</td></tr><tr><td> (Nafis wanted to change his habit.) 何网，可</td><td>  G (NAFIS HIS/HER BEFORE HABIT CHANGE WANT WAS) 河桐國可阿國司特国厕</td><td>G G   (NAFIS HIS/HER BEFORE HABIT CHANGE WANT WAS) 河利國可阿可爾國國</td></tr><tr><td>(When he/she was crying, I was holding his/her hand.) WAS)</td><td>(WHEN HE/SHE CRY WAS THEN I HIS/HER HAND HOLD</td><td>(WHEN HE/SHE CRY WAS THEN I HIS/HER HAND HOLD WAS)</td></tr><tr><td colspan="3">Moderate Performance</td></tr><tr><td>何，可何?</td><td>, </td><td>何可可,可何体</td></tr><tr><td>(When you grow up, what will you do?)</td><td>(WHEN YOU SLOWLY SLOWLY GROW WILL THEN YOU</td><td>(WHEN YOU SLOWLY SLOWLY GROW WILL THEN YOU</td></tr><tr><td></td><td>WORK WHAT WILL) CA</td><td>WHAT WORK WILL) G</td></tr><tr><td>(I was told not to eat at that restaurant.</td><td>(TO ME THAT RESTAURANT EAT DO FORBIDDEN SAY HAS</td><td>(TO ME THAT RESTAURANT EAT FORBIDDEN SAY HAS</td></tr><tr><td></td><td>BEEN) </td><td>BEEN) </td></tr><tr><td>(You cannot leave me.)</td><td>(YOU ME LEAVE GO CAN NOT)</td><td>(YOU ME LEAVE GO GO CAN NOT)</td></tr><tr><td>可可 (I could not understand what you wanted to say.)</td><td>可可响体可可 (I UNDERSTAND NOT YOU WHAT SAY WANT WAS)</td><td>可可可可 (I UNDERSTAND CAN NOT YOU WHAT SAY WANT IS)</td></tr><tr><td></td><td></td><td></td></tr><tr><td>(I had prepared dinner by the time you arrived.)</td><td>(YOU ARRIVE BEFORE I FOOD PREPARE KEEP WAS) Poor Performance (Limitations)</td><td>(YOU ARRIVE BEFORE I KEEP FOOD PREPARE KEEP WAS)</td></tr><tr><td colspan="3"></td></tr><tr><td></td><td></td><td></td></tr><tr><td>(We used to dress up as police officers when we were</td><td>(WE CHILDHOOD POLICE DRESS UP USED TO)</td><td>(WE POLICE OFFICER WAS)</td></tr><tr><td>kids.) </td><td></td><td></td></tr><tr><td>(I will eat after class.)</td><td>(I CLASS FINISH AFTER EAT WILL)</td><td>(CLASS INSIDE WHAT WILL)</td></tr><tr><td>可可不不 (You will start working after the rain stops.)</td><td></td><td></td></tr><tr><td>间夜同</td><td>(YOU RAIN STOP AFTER WORK START DO) 丽夜响网</td><td>(YOU RAIN STOP AFTER THAT START DO) 间</td></tr><tr><td>(Read the students&#x27; books regularly.) ,</td><td>(STUDENT BOOK READ REGULARLY REGULARLY) 可，可</td><td>(STUDENT THEY BOOK READ EVERY DAY)</td></tr><tr><td>(My mother would buy me the dress if I wanted it.)</td><td>(IF I DRESS BUY WANT MOTHER ME DRESS BUY GIVE WILL)</td><td>(IF I DRESS WANT MOTHER ME DRESS BUY GIVE WILL)</td></tr></table>

BLEU-4 scores. After augmentation, IsharaBART reaches a similar BLEU-4 score (51.74 versus 52.24). However, IsharaBART uses 56M parameters, while BanglaT5 uses 247M. On S2G, IsharaBART achieves the highest BLEU-4 score among the evaluated models in both data settings. With augmented data, it reaches 61.40 BLEU-4. The score for mBART-50 is 60.82, despite its 611M parameters. These results show that IsharaBART ofers a good balance between model size and translation performance.

The pre-training ablation provides further insight. Pre-training gives the largest relative gains when the model has only 1,000 annotated pairs. These gains remain after data augmentation. Pre-training also reduces the number of epochs needed to reach the best validation BLEU-4 score. These results suggest that monolingual Bangla pre-training helps translation when labeled data are limited.

Data augmentation also improves performance, but the gains difer by translation direction. For IsharaBART, BLEU-4 increases by 8.44 points on G2S and 22.01 points on S2G. One possible reason is that S2G requires the model to generate gloss sequences from natural Bangla sentences. The model must learn specific output patterns for this task. The annotated dataset may not provide enough examples of these patterns. Augmentation may help the model learn them. However, this explanation remains a hypothesis. The experiments do not identify the exact linguistic features behind the larger S2G gain.

This study extends previous BdSL research by evaluating both G2S and S2G translation in the same experimental framework. It also examines monolingual pre-training and targeted data augmentation. The results show that the two directions can produce diferent outcomes.

Table 10: Human evaluation of IsharaBART for G2S and S2G translation.Two BdSL signers evaluated 50 randomly sampled translations on adequacy (0–100 scale) and fluency (1=Bad, 2=Moderate, 3=Good). <sup>a</sup>Adequacy ≥ 75% AND Fluency = Good (both evaluators). r = Pearson correlation, $\kappa _ { w }$ = Weighted Cohen’s Kappa, $\rho = \mathrm { S }$ pearman’s rho. MAE = Mean Absolute Error, Exact = Exact agreement percentage. $^ { * } p < 0 . 0 5 , ^ { * * } p < 0 . 0 1 , ^ { * * * } p < 0 . 0 0 1$
<table><tr><td rowspan="2">Metric</td><td colspan="4">G2S</td><td colspan="4">S2G</td></tr><tr><td>Eval. 1</td><td>Eval. 2</td><td>Overall</td><td>Agreement</td><td>Eval. 1</td><td>Eval. 2</td><td>Overall</td><td>Agreement</td></tr><tr><td>Adequacy</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Mean ± SD Median</td><td>85.0 ± 30.3 92.5 ± 22.7 88.8 ± 26.9 100</td><td></td><td>100</td><td> $r = 0 . 5 7 4 ^ { * * * }$ </td><td>83.5 ± 30.1</td><td>92.0 ± 21.1</td><td>87.8 ± 26.2 100</td><td> $r = 0 . 4 5 0 ^ { * * * }$ </td></tr><tr><td>High quality (≥ 75%)</td><td>42 (84%)</td><td>100 46 (92%)</td><td>42 (84%)</td><td>MAE = 11.5</td><td>100 38 (76%)</td><td>100 38 (76%)</td><td>38 (76%)</td><td>MAE = 12.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Fluency Mean ± SD</td><td>2.64 ± 0.66</td><td>2.78 ± 0.46 2.71 ± 0.57</td><td></td><td></td><td>2.66 ± 0.69</td><td>2.56 ± 0.70</td><td>2.61 ± 0.69</td><td> $\kappa _ { w } = 0 . 4 4 3$ </td></tr><tr><td>Median</td><td>3.0</td><td>3.0</td><td>3.0</td><td> $\begin{array} { c } { { \kappa _ { w } = 0 . 3 6 4 ^ { * } } } \\ { { \rho = 0 . 3 5 8 ^ { * } } } \end{array}$ </td><td>3.0</td><td>3.0</td><td>3.0</td><td>ρ = 0.460***</td></tr><tr><td>Fluency distribution</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bad (1)</td><td>5 (10%)</td><td>1 (2%)</td><td>6 (6%)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Moderate (2)</td><td>8 (16%)</td><td>9 (18%)</td><td>17 (17%)</td><td></td><td>6 (12%)</td><td>6 (12%)</td><td>12 (12%)</td><td></td></tr><tr><td>Good (3)</td><td>37 (74%)</td><td>40 (80%)</td><td>77 (77%)</td><td>Exact: 78%</td><td>5 (10%) 39 (78%)</td><td>10 (20%)</td><td>15 (15%) 73 (73%)</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>34 (68%)</td><td></td><td>Exact: 70%</td></tr><tr><td>Overall quality</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Both criteria meta</td><td></td><td></td><td>33 (66%)</td><td> $\rho = 0 . 6 9 4 ^ { * * * } \ |$ </td><td></td><td></td><td>28 (56%)</td><td> $\rho = 0 . 6 3 6 ^ { * * * }$ </td></tr></table>

After augmentation, IsharaBART comes close to BanglaT5 on G2S. It achieves the highest S2G BLEU-4 score among the evaluated models. These diferences show why studies should evaluate both directions. Results from one direction may not reflect performance on the other. Model selection should also consider the target task and available computing resources.

Human evaluation provides further insight into translation quality. The mean adequacy scores are 88.8 for G2S and 87.8 for S2G. However, only 66% of G2S outputs and 56% of S2G outputs meet both adequacy and fluency criteria. This suggests that the model often preserves the intended meaning. However, producing fluent and consistent translations remains dificult. Inter-rater agreement was modest to moderate. The evaluation also included only 50 outputs per direction. These results show promise, but they do not establish reliable translation in every case.

The LoRA experiments show another trade-of. LoRA reduces the number of trainable parameters by 99.3%. However, BLEU-4 decreases by 28.0–52.2% compared with full finetuning. The decline is greater for S2G than for G2S under the tested configuration. This suggests that LoRA does not fully match full fine-tuning in these experiments.

## 6. Conclusion and Future Work

In this study, we presented IsharaBART, a bidirectional translation model for BdSL. It translates between gloss sequences and Bangla sentences in both directions. Our results show that monolingual pre-training and data augmentation improve translation quality. The model also reduces training time by 62.5% to 90.4%. It maintains good structural, semantic, and lexical consistency. We compared IsharaBART with four baseline models under the same conditions. The results show that our approach achieves comparable or better translation quality. Its small size and faster training may support accessibility tools for education, healthcare, and daily communication.

However, this study has some limitations. The annotated and augmented dataset is small compared to high-resource benchmarks. This may limit the model’s ability to handle rare sentence structures and glosses. The model also depends on accurate gloss sequences from a vision-based sign recognition system. We did not study this component. Recognition errors may occur due to poor lighting or diferences between signers. Our monolingual approach may also miss benefits from multilingual training.

Table 11: Comparison of full fine-tuning and low-rank adaptation (LoRA) for IsharaBART across annotated and augmented data. BLEU-4 and ROUGE-L are reported on a 0–100 scale, while COMET is reported on its native 0–1 scale. Changes are reported relative to full fine-tuning as absolute and relative diferences; negative values indicate degradation under LoRA.
<table><tr><td>Dataset</td><td>Translation Direction</td><td>Fine-Tuning Strategy</td><td>BLEU-4</td><td>COMET</td><td>ROUGE-L</td></tr><tr><td rowspan="3">Annotated</td><td rowspan="3">G2S</td><td>Full fine-tuning</td><td>43.30</td><td>0.9279</td><td>78.32</td></tr><tr><td>LoRA</td><td>31.17</td><td>0.8973</td><td>72.27</td></tr><tr><td>Change</td><td>-12.13 (-28.01%)</td><td>-0.0306 (-3.30%)</td><td>-6.05 (−7.72%)</td></tr><tr><td rowspan="3">Annotated</td><td rowspan="3">S2G</td><td>Full fine-tuning</td><td>39.39</td><td>0.9283</td><td>75.78</td></tr><tr><td>LoRA</td><td>22.15</td><td>0.9221</td><td>68.97</td></tr><tr><td>Change</td><td>-17.24 (-43.77%)</td><td>-0.0062 (-0.67%)</td><td>-6.81 (-8.99%)</td></tr><tr><td rowspan="3">Augmented G2S</td><td rowspan="3"></td><td>Full fine-tuning</td><td>51.74</td><td>0.9471</td><td>80.95</td></tr><tr><td>LoRA</td><td>36.31</td><td>0.9199</td><td>73.06</td></tr><tr><td>Change</td><td>-15.43 (-29.82%)</td><td>-0.0272 (-2.87%)</td><td>-7.89 (-9.75%)</td></tr><tr><td rowspan="3">Augmented</td><td rowspan="3">S2G</td><td>Full fine-tuning</td><td>61.40</td><td>0.9388</td><td>83.07</td></tr><tr><td>LoRA</td><td>29.37</td><td>0.9185</td><td>69.55</td></tr><tr><td>Change</td><td>-32.03 (-52.17%)</td><td>-0.0203 (-2.16%)</td><td>-13.52 (-16.27%)</td></tr></table>

Table 12: Overview of major gloss-sentence datasets and associated SOTA BLEU-4 performance for G2S and S2G tasks. IsharaBART achieves substantially higher scores than prior Bangla BdSL resources despite comparable scale, highlighting the efectiveness of Bangla-specific pre-training and augmentation.
<table><tr><td>Dataset</td><td>Language</td><td>Size</td><td>Domain</td><td>G2S</td><td>S2G</td></tr><tr><td>PHOENIX-14T [7, 68]</td><td>German (DGS)</td><td>8,257</td><td>Weather</td><td>18.75</td><td>20.68</td></tr><tr><td>How2Sign [69]</td><td>English (ASL)</td><td>35,000</td><td>Instructional</td><td>8.0</td><td></td></tr><tr><td>iSign [70]</td><td>English (ISL)</td><td>118,000</td><td>Open</td><td>17.6</td><td></td></tr><tr><td>IsharaKhobor [71]</td><td>Bangla (BdSL)</td><td>5,642</td><td>News</td><td></td><td></td></tr><tr><td>Bangla-SGP [72]</td><td>Bangla (BdSL)</td><td>4,000</td><td>Open</td><td>27.3</td><td></td></tr><tr><td>Ours</td><td>Bangla (BdSL)</td><td>4,831</td><td>Open</td><td>51.7</td><td>61.4</td></tr></table>

IsharaBART has a compact architecture and needs fewer training epochs. These features may help on devices with limited resources. However, we did not measure inference time, memory use, or energy consumption. We also did not test the model on mobile or edge devices. Its eficiency in real-world settings remains unclear.

The model translates gloss sequences and Bangla text. It does not process sign-language videos directly or generate visual signs. A complete translation system would need a sign recognition model and a method to generate sign-language output. IsharaBART could serve as the text translation component of such a system.

The approach may also work for other low-resource sign languages, but we have not tested this. Future work should expand the dataset and include more signers and regional gloss variations. Larger and more diverse groups of human evaluators should assess the translations. Human evaluations of the baseline models would also show whether the automatic scores reflect real improvements. Tests of inference speed, memory use, and energy consumption are also needed. Integrating video-based sign recognition would help assess the model in real-world communication systems.

Future work should expand the dataset through annotations from the deaf community. We should also explore end-to-end video-to-text models that do not depend on gloss inputs. Hybrid multilingual and monolingual pre-training may help the model handle linguistic variation. Another direction is to test span-corruption and hybrid denoising objectives. These methods may help close the remaining BLEU-3 and BLEU-4 gaps in G2S translation, especially compared with BanglaT5. The goal is to improve translation quality without increasing model size or training time. Finally, real-world deployment studies could help measure the model’s practical value and guide further improvements for other low-resource sign languages.

## Disclosure statement

The author(s) declare(s) no potential conflict of interest regarding the publication of this paper.

## Acknowledgment

This work was supported by the ICT fellowship (FY 2024–2025) under grant code 56.00.0000.052.33.0003.24–115 provided by the Information and Communication Technology Division, Ministry of Posts, Telecommunications and Information Technology, Government of the People’s Republic of Bangladesh.

## References

[1] B. O. Olusanya, A. C. Davis, H. J. Hofman, Hearing loss grades and the international classification of functioning, disability and health, Bulletin of the World Health Organization 97 (10) (2019) 725––728. doi:10.2471/blt.19.230367.

[2] J. Guo, P. Li, T. Cohn, Bridging sign and spoken languages: Pseudo gloss generation for sign language translation, in: Advances in Neural Information Processing Systems 38, Neural Information Processing Systems Foundation, Inc. (NeurIPS), San Diego, California, USA and Mexico City, Mexico, 2025, pp. 86134––86162. doi:10.52202/ 085713-2598.

[3] M. De Coster, D. Shterionov, M. Van Herreweghe, J. Dambre, Machine translation from signed to spoken languages: state of the art and challenges, Universal Access in the Information Society 23 (3) (2024) 1305––1331. doi:10.1007/s10209-023-00992-1.

[4] S. Tan, N. Khan, Z. An, Y. Ando, R. Kawakami, K. Nakadai, A review of deep learningbased approaches to sign language processing, Advanced Robotics 38 (23) (2024) 1649– –1667. doi:10.1080/01691864.2024.2442721.

[5] A. Othman, M. Jemni, English-ASL gloss parallel corpus 2012: ASLG-PC12, in: Proceedings of the LREC2012 5th Workshop on the Representation and Processing of Sign Languages: Interactions between Corpus and Lexicon, European Language Resources Association (ELRA), Istanbul, Turkey, 2012, pp. 151–154.

[6] H. Zhou, W. Zhou, W. Qi, J. Pu, H. Li, Improving sign language translation with monolingual data by sign back-translation, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Nashville, TN, USA, 2021, pp. 1316––1325. doi:10.1109/CVPR46437.2021.00137.

[7] N. C. Camgoz, S. Hadfield, O. Koller, H. Ney, R. Bowden, Neural sign language translation, in: 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, IEEE, Salt Lake City, UT, 2018, pp. 7784––7793. doi:10.1109/CVPR.2018.00812.

[8] T. H. Dhrubo, M. T. Reza, S. Sahariar, W. A. Mukto, Transformer based sign-totext translation for bangladeshi sign language, Scientific Reports 16 (1) (2026) 1154. doi:10.1038/s41598-025-30856-y.

[9] M. Hossen, A. Govindaiah, S. Sultana, A. Bhuiyan, Bengali sign language recognition using deep convolutional neural network, in: 2018 Joint 7th International Conference on Informatics, Electronics & Vision (ICIEV) and 2018 2nd International Conference on Imaging, Vision & Pattern Recognition (icIVPR), IEEE, Kitakyushu, Japan, 2018, pp. 369––373. doi:10.1109/ICIEV.2018.8640962.

[10] M. S. Islam, S. S. S. Mousumi, N. A. Jessan, A. S. A. Rabby, S. Abujar, S. A. Hossain, Ishara-bochon: The first multipurpose open access dataset for bangla sign language isolated digits, in: Recent Trends in Image Processing and Pattern Recognition, Vol. 1035, Springer Singapore, Singapore, 2019, pp. 420––428. doi:10.1007/ 978-981-13-9181-1\_37.

[11] S. Siddique, S. Islam, E. E. Neon, T. Sabbir, I. T. Naheen, R. Khan, Deep learningbased bangla sign language detection with an edge device, Intelligent Systems with Applications 18 (2023) 200224. doi:10.1016/j.iswa.2023.200224.

[12] M. J. Raihan, M. I. Labib, A. A. J. Jim, J. J. Tiang, U. Biswas, A.-A. Nahid, Bengalisign: A machine learning-based bengali sign language interpretation for deaf and nonverbal people, Sensors 24 (16) (2024) 5351. doi:10.3390/s24165351.

[13] A. Das, A. D. Aishi, M. U. Toha, M. F. Kader, Bangla sign language translator for deaf and speech impaired people using deep lstm, International Journal of Speech Technology 28 (2) (2025) 351––368. doi:10.1007/s10772-025-10184-2.

[14] Y. Zhang, Y. Han, Z. Zhu, X. Jiang, Y. Zhang, Artificial intelligence in sign language recognition: A comprehensive bibliometric and visual analysis, Computers and Electrical Engineering 120 (2024) 109854. doi:10.1016/j.compeleceng.2024.109854.

[15] B. Kang, S. Tripathi, T. Q. Nguyen, Real-time sign language fingerspelling recognition using convolutional neural networks from depth map, in: 2015 3rd IAPR Asian Conference on Pattern Recognition (ACPR), IEEE, Kuala Lumpur, Malaysia, 2015, pp. 136––140. doi:10.1109/ACPR.2015.7486481.

[16] M. Oberweger, P. Wohlhart, V. Lepetit, Training a feedback loop for hand pose estimation, in: 2015 IEEE International Conference on Computer Vision (ICCV), IEEE, Santiago, Chile, 2015, pp. 3316––3324. doi:10.1109/ICCV.2015.379.

[17] J. Huang, W. Zhou, H. Li, W. Li, Sign language recognition using 3d convolutional neural networks, in: 2015 IEEE International Conference on Multimedia and Expo (ICME), IEEE, Turin, Italy, 2015, pp. 1––6. doi:10.1109/ICME.2015.7177428.

[18] R. Rastgoo, K. Kiani, S. Escalera, Multi-modal deep hand sign language recognition in still images using restricted boltzmann machine, Entropy 20 (11) (2018) 809. doi: 10.3390/e20110809.

[19] C. Wei, W. Zhou, J. Pu, H. Li, Deep grammatical multi-classifier for continuous sign language recognition, in: 2019 IEEE Fifth International Conference on Multimedia Big Data (BigMM), IEEE, Singapore, Singapore, 2019, pp. 435––442. doi:10.1109/BigMM. 2019.00027.

[20] J. Hasanov, N. Alishzade, A. Nazimzade, S. Dadashzade, T. Tahirov, Development of a hybrid word recognition system and dataset for the azerbaijani sign language dactyl alphabet, Speech Communication 153 (2023) 102960. doi:10.1016/j.specom.2023. 102960.

[21] M. S. Aiswarya, R. Arockia Xavier Annie, Keyframe extraction algorithm for continuous sign-language videos using angular displacement and sequence check metrics, International Journal of Intelligent Systems 2024 (2024) 4725216. doi:10.1155/2024/4725216.

[22] W. Hao, C. Hou, Z. Zhang, X. Zhai, L. Wang, G. Lv, A sensing data and deep learningbased sign language recognition approach, Computers and Electrical Engineering 118 (2024) 109339. doi:10.1016/j.compeleceng.2024.109339.

[23] S. Renjith, A. Varghese, M. Rashmi, P. S.S., Transformer-based motion-visual integrated fusion for isolated sign language recognition, Computers and Electrical Engineering 130 (2026) 110902. doi:10.1016/j.compeleceng.2025.110902.

[24] B. R, A. K. V. D, A. K. V. D, Mizo hand sign language detection using a multiscale transformer-based hybrid feature extractor and fusion network, Computers and Electrical Engineering 131 (2026) 110916. doi:10.1016/j.compeleceng.2025.110916.

[25] J. Wu, X. Li, T. Li, F. Meng, Y. Kong, G. Yang, L. Senhadji, H. Shu, Cslnspeech: Solving the extended speech separation problem with the help of chinese sign language, Speech Communication 165 (2024) 103131. doi:10.1016/j.specom.2024.103131.

[26] F. M. Talaat, W. El-Shafai, N. F. Soliman, A. D. Algarni, F. E. Abd El-Samie, A. I. Siam, Real-time arabic avatar for deaf-mute communication enabled by deep learning sign language translation, Computers and Electrical Engineering 119 (2024) 109475. doi:10.1016/j.compeleceng.2024.109475.

[27] N. C. Camgoz, O. Koller, S. Hadfield, R. Bowden, Sign language transformers: Joint end-to-end sign language recognition and translation, in: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Seattle, WA, USA, 2020, pp. 10020––10030. doi:10.1109/CVPR42600.2020.01004.

[28] L. Chaudhary, T. Ananthanarayana, E. Hoq, I. Nwogu, Signnet ii: A transformer-based two-way sign language translation model, IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (11) (2023) 12896–12907. doi:10.1109/TPAMI.2022.3232389.

[29] J. Hao, H. Pan, A novel deep transformer based cvt model for sign language recognition in visual communication, Scientific Reports 16 (1) (2026) 1823. doi:10.1038/ s41598-025-31558-1.

[30] W. F. Maia, A. M. Lopes, S. A. David, Automatic sign language to text translation using mediapipe and transformer architectures, Neurocomputing 642 (2025) 130421. doi:10.1016/j.neucom.2025.130421.

[31] H. Jin, S. Zhang, Z. Yang, Q. Han, J. Cao, Exploiting locality sensitive hashing - clustering and gloss feature for sign language production, Speech Communication 171 (2025) 103227. doi:10.1016/j.specom.2025.103227.

[32] M. Karim, M. Ali, M. F. Hasan, M. Fatema-Tuj-Jahra, S. S. Sultana, D. M. Farid, Bangla sign language recognition using yolov5, in: 2023 IEEE 8th International Conference for Convergence in Technology (I2CT), IEEE, Lonavla, India, 2023, pp. 1––7. doi:10.1109/I2CT57861.2023.10126419.

[33] M. S. U. R. Khan, P. Mehta, A. Sankar, U. Kumaravelan, S. Doddapaneni, S. B, V. G, S. Jain, A. Kunchukuttan, P. Kumar, R. Dabre, M. Khapra, Indicllmsuite: A blueprint for creating pre-training and fine-tuning datasets for indian languages, in: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Association for Computational Linguistics, Bangkok, Thailand, 2024, pp. 15831––15879. doi:10.18653/v1/2024.acl-long.843.

[34] J. Abadji, P. Ortiz Suarez, L. Romary, B. Sagot, Towards a cleaner document-oriented multilingual crawled corpus, in: Proceedings of the Thirteenth Language Resources and Evaluation Conference, European Language Resources Association, Marseille, France, 2022, pp. 4344–4355. doi:10.48550/ARXIV.2201.06642.

[35] J. Dodge, M. Sap, A. Marasović, W. Agnew, G. Ilharco, D. Groeneveld, M. Mitchell, M. Gardner, Documenting large webtext corpora: A case study on the colossal clean crawled corpus, in: Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, Association for Computational Linguistics, Online and

Punta Cana, Dominican Republic, 2021, pp. 1286––1305. doi:10.18653/v1/2021. emnlp-main.98.

[36] I. Johnson, L.-A. Kafee, M. Redi, Wikimedia data for ai: a review of wikimedia datasets for nlp tasks and ai-assisted editing, in: Proceedings of the First Workshop on Advancing Natural Language Processing for Wikipedia, Association for Computational Linguistics, Miami, Florida, USA, 2024, pp. 91––101. doi:10.18653/v1/2024.wikinlp-1.14.

[37] M. Kabir, O. Bin Mahfuz, S. R. Raiyan, H. Mahmud, M. K. Hasan, Banglabook: A large-scale bangla dataset for sentiment analysis from book reviews, in: Findings of the Association for Computational Linguistics: ACL 2023, Association for Computational Linguistics, Toronto, Canada, 2023, pp. 1237––1247. doi:10.18653/v1/2023. findings-acl.80.

[38] A. Bhattacharjee, T. Hasan, W. U. Ahmad, R. Shahriyar, Banglanlg and banglat5: Benchmarks and resources for evaluating low-resource natural language generation in bangla, in: Findings of the Association for Computational Linguistics: EACL 2023, Association for Computational Linguistics, Dubrovnik, Croatia, 2023, pp. 726––735. doi:10.18653/v1/2023.findings-eacl.54.

[39] A. Kilgarrif, V. Baisa, J. Bušta, M. Jakubíček, V. Kovář, J. Michelfeit, P. Rychlý, V. Suchomel, The sketch engine: ten years on, Lexicography 1 (1) (2014) 7––36. doi: 10.1007/s40607-014-0009-9.

[40] M. H. I. Bijoy, U. Ayman, M. M. Islam, Banglatense: A large-scale dataset of bangla sentences categorized by tense: Past, present, and future, Data in Brief 59 (2025) 111400. doi:10.1016/j.dib.2025.111400.

[41] T. S. Apon, R. Anan, E. A. Modhu, A. Suter, I. J. Sneha, M. G. R. Alam, Banglasarc: A dataset for sarcasm detection, in: 2022 IEEE Asia-Pacific Conference on Computer Science and Data Engineering (CSDE), IEEE, Gold Coast, Australia, 2022, pp. 1––5. doi:10.1109/CSDE56538.2022.10089322.

[42] A. Moryossef, K. Yin, G. Neubig, Y. Goldberg, Data augmentation for sign language gloss translation, in: Proceedings of the 1st International Workshop on Automatic Translation for Signed and Spoken Languages (AT4SSL), Association for Machine Translation in the Americas, 2021, pp. 1–11.

[43] A. Bhattacharjee, T. Hasan, W. Ahmad, K. S. Mubasshir, M. S. Islam, A. Iqbal, M. S. Rahman, R. Shahriyar, Banglabert: Language model pretraining and benchmarks for low-resource language understanding evaluation in bangla, in: Findings of the Association for Computational Linguistics: NAACL 2022, Association for Computational Linguistics, Seattle, United States, 2022, pp. 1318––1327. doi:10.18653/v1/2022. findings-naacl.98.

[44] S. Kobayashi, Contextual augmentation: Data augmentation by words with paradigmatic relations, in: Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technolo-

gies, Volume 2 (Short Papers), New Orleans, Louisiana, 2018, pp. 452–457. doi: 10.18653/v1/N18-2072.

[45] J. Y. Jang, H. Park, S. Shin, S. Shin, B. Yoon, G. Gweon, Automatic gloss-level data augmentation for sign language translation, in: Proceedings of the Thirteenth Language Resources and Evaluation Conference, European Language Resources Association, Marseille, France, 2022, pp. 6808–6813.

[46] N. Team, M. R. Costa-jussa, J. Cross, et al., No language left behind: Scaling humancentered machine translation, arXiv preprint arXiv:2207.04672 (2022). doi:10.48550/ arXiv.2207.04672.

[47] J. Gala, P. A. Chitale, A. K. Raghavan, V. Gumma, S. Doddapaneni, A. K. M, J. A. Nawale, A. Sujatha, R. Puduppully, V. Raghavan, P. Kumar, M. M. Khapra, R. Dabre, A. Kunchukuttan, Indictrans2: Towards high-quality and accessible machine transla tion models for all 22 scheduled indian languages, Transactions on Machine Learning Research (2023). doi:10.48550/arXiv.2305.16307.

[48] R. Sennrich, B. Haddow, A. Birch, Improving neural machine translation models with monolingual data, in: Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Berlin, Germany, 2016, pp. 86–96. doi:10.18653/v1/P16-1009.

[49] L. Team, The llama 3 herd of models, arXiv preprint arXiv:2407.21783 (2024). doi: 10.48550/arXiv.2407.21783.

[50] Y. Wang, Y. Kordi, S. Mishra, A. Liu, N. A. Smith, D. Khashabi, H. Hajishirzi, Selfinstruct: Aligning language models with self-generated instructions, in: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Toronto, Canada, 2023, pp. 13484–13508. doi:10.18653/v1/2023. acl-long.754.

[51] C. Whitehouse, M. Choudhury, A. F. Aji, LLM-powered data augmentation for enhanced cross-lingual performance, in: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Singapore, 2023, pp. 671–686. doi: 10.18653/v1/2023.emnlp-main.44.

[52] S. Węglarczyk, Kernel density estimation and its application, ITM Web of Conferences 23 (2018) 00037. doi:10.1051/itmconf/20182300037.

[53] L. McInnes, J. Healy, N. Saul, L. Großberger, Umap: Uniform manifold approximation and projection, Journal of Open Source Software 3 (29) (2018) 861. doi:10.21105/ joss.00861.

[54] D. C. van Leijenhorst, T. P. van der Weide, A formal derivation of heaps’ law, Information Sciences 170 (2–4) (2005) 263––272. doi:10.1016/j.ins.2004.03.006.

[55] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, L. Zettlemoyer, Bart: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension, in: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Association for Computational Linguistics, Online, 2020, pp. 7871––7880. doi:10.18653/v1/2020.acl-main.703.

[56] A. CONNEAU, G. Lample, Cross-lingual language model pretraining, in: H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, R. Garnett (Eds.), Advances in Neural Information Processing Systems, Vol. 32, Curran Associates, Inc., 2019. doi: 10.48550/arXiv.1901.07291.

[57] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of large language models, in: The Tenth International Conference on Learning Representations, Virtual, 2022. doi:10.48550/arXiv.2106.09685.

[58] Y. Liu, J. Gu, N. Goyal, X. Li, S. Edunov, M. Ghazvininejad, M. Lewis, L. Zettlemoyer, Multilingual denoising pre-training for neural machine translation, Transactions of the Association for Computational Linguistics 8 (2020) 726––742. doi:10.1162/tacl\_a\_ 00343.

[59] L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, C. Raffel, mt5: A massively multilingual pre-trained text-to-text transformer, in: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Association for Computational Linguistics, Online, 2021, pp. 483––498. doi:10.18653/v1/2021.naacl-main.41.

[60] M. S. Salim, H. Murad, D. Das, F. Ahmed, Banglagpt: A generative pretrained transformer-based model for bangla language, in: 2023 International Conference on Information and Communication Technology for Sustainable Development (ICICT4SD), IEEE, Dhaka, Bangladesh, 2023, pp. 56––59. doi:10.1109/ICICT4SD59951.2023. 10303383.

[61] C.-Y. Lin, ROUGE: A package for automatic evaluation of summaries, in: Text Summarization Branches Out, Association for Computational Linguistics, Barcelona, Spain, 2004, pp. 74–81.

[62] K. Papineni, S. Roukos, T. Ward, W.-J. Zhu, Bleu: a method for automatic evaluation of machine translation, in: Proceedings of the 40th Annual Meeting on Association for Computational Linguistics - ACL ’02, Association for Computational Linguistics, Philadelphia, Pennsylvania, 2002, pp. 311–318. doi:10.3115/1073083.1073135.

[63] R. Rei, C. Stewart, A. C. Farinha, A. Lavie, Comet: A neural framework for mt evaluation, in: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), Association for Computational Linguistics, Online, 2020, pp. 2685––2702. doi:10.18653/v1/2020.emnlp-main.213.

[64] D. Biderman, J. Portes, J. J. G. Ortiz, M. Paul, P. Greengard, C. Jennings, D. King, S. Havens, V. Chiley, J. Frankle, C. Blakeney, J. P. Cunningham, LoRA learns less and

forgets less, Transactions on Machine Learning Research (2024). doi:10.48550/arXiv. 2405.09673.

[65] R. Shuttleworth, J. Andreas, A. Torralba, P. Sharma, Lora vs full fine-tuning: An illusion of equivalence, in: Advances in Neural Information Processing Systems 38, Neural Information Processing Systems Foundation, Inc. (NeurIPS), San Diego, California, USA and Mexico City, Mexico, 2025, pp. 193421––193456. doi:10.52202/ 085713-5808.

[66] T. Su, X. Peng, S. Thillainathan, D. Guzmán, S. Ranathunga, E.-S. Lee, Unlocking parameter-eficient fine-tuning for low-resource language translation, in: Findings of the Association for Computational Linguistics: NAACL 2024, Association for Computational Linguistics, Mexico City, Mexico, 2024, pp. 4217––4225. doi: 10.18653/v1/2024.findings-naacl.263.

[67] Y. Zeng, K. Lee, The expressive power of low-rank adaptation, in: The Twelfth International Conference on Learning Representations, OpenReview.net, Vienna, Austria, 2024. doi:10.48550/arXiv.2310.17513.

[68] S. M. Abdullah, A. Paul, S. Rayana, A. Kabir, Z. Masud, State-of-the-art translation of text-to-gloss using mbart : A case study of bangla (2025). arXiv:2504.02293, doi:10.48550/arXiv.2504.02293.

[69] A. Duarte, S. Palaskar, L. Ventura, D. Ghadiyaram, K. DeHaan, F. Metze, J. Torres, X. Giro-i Nieto, How2sign: A large-scale multimodal dataset for continuous american sign language, in: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, Nashville, TN, USA, 2021, pp. 2735––2744. doi: 10.1109/CVPR46437.2021.00276.

[70] A. Joshi, R. Mohanty, M. Kanakanti, A. Mangla, S. Choudhary, M. Barbate, A. Modi, isign: A benchmark for indian sign language processing, in: Findings of the Association for Computational Linguistics ACL 2024, Association for Computational Linguistics, Bangkok, Thailand and virtual meeting, 2024, pp. 10827––10844. doi:10.18653/v1/ 2024.findings-acl.643.

[71] H. A. Rubaiyeat, H. Mahmud, M. K. Hasan, Bangla sign language translation: Dataset creation challenges, benchmarking and prospects (2025). arXiv:2511.21533.

[72] N. Saha, R. Shahriyar, N. A. Roudra, S. Sakib, A. A. Rasel, Introducing a Bangla sentence – gloss pair dataset for Bangla sign language translation and research, in: Proceedings of the Fifteenth Language Resources and Evaluation Conference, ELRA Language Resource Association, Palma de Mallorca, Spain, 2026, pp. 10457–10466. doi:10.63317/38qenrwzegr9.

## Appendix A. Training Configuration and Hyperparameters

We provide complete training configurations for IsharaBART pre-training and fine-tuning experiments to ensure reproducibility. All models were trained using the Hugging Face Transformers library with PyTorch backend.

## Appendix A.1. IsharaBART Pre-training

IsharaBART was pre-trained on a monolingual Bangla corpus using a denoising autoencoder objective with custom corruption strategies. Table A.13 summarizes the key hyperparameters. We use polynomial learning rate decay as recommended in the original BART paper, with a relatively high initial learning rate $( 5 \times 1 0 ^ { - 4 } )$ to accommodate the smaller model size and focused pre-training corpus. Training was conducted in bfloat16 precision for numerical stability and computational eficiency. The efective batch size of 256 (64 per device × 4 gradient accumulation steps) balances training stability with memory constraints of encoder–decoder architectures.

## Appendix A.2. Fine-tuning Configuration

All models (IsharaBART and baselines) were fine-tuned on the gloss translation tasks using identical or carefully matched configurations to ensure fair comparison. Table A.14 presents the hyperparameters used for each model and Tale A.15 present the LoRA configuration for parameter-eficient fine-tuning of IsharaBART. Configuration diferences across models reflect preliminary validation experiments and model-specific characteristics: mT5- Small benefits from a higher learning rate $( 1 \times 1 0 ^ { - 3 } )$ due to its architecture and pre-training objective, while BanglaGPT requires a lower learning rate $( 1 \times 1 0 ^ { - 5 } )$ and larger gradient accumulation to stabilize training in the decoder-only setting.

For all encoder–decoder models (IsharaBART, mBART-50, mT5-Small, BanglaT5), we use beam search with 4 beams during generation for evaluation. Early stopping with patience of 3–5 epochs prevents overfitting on the low-resource fine-tuning datasets. Model selection is based on validation BLEU-4 for encoder–decoder models and validation loss for BanglaGPT, reflecting the diferent training objectives and generation strategies.

All experiments were conducted using Hugging Face Transformers (v4.57) and PyTorch 2.9 on NVIDIA A100 GPU for pre-training and Tesla T4 GPU for fine-tuning. For pretraining, we implemented a custom data collator that applies BART-style corruption including text infilling, sentence permutation, and token masking, with special handling for Bangla punctuation tokens (danda and question marks). Fine-tuning used the standard DataCollatorForSeq2Seq for dynamic padding.

Table A.13: IsharaBART pre-training configuration. All training used AdamW optimizer with $\beta _ { 1 } = 0 . 9 ,$ $\beta _ { 2 } = 0 . 9 9 9 , \epsilon = 1 0 ^ { - 8 }$
<table><tr><td rowspan=1 colspan=1>Hyperparameter</td><td rowspan=1 colspan=1>Value</td></tr><tr><td rowspan=1 colspan=1>Training StepsBatch Size (per device)Gradient AccumulationEffective Batch Size</td><td rowspan=1 colspan=1>30,000644 steps256</td></tr><tr><td rowspan=1 colspan=1>Learning RateLR ScheduleWarmup StepsWeight DecayMax Gradient Norm</td><td rowspan=1 colspan=1> $5 \times 1 0 ^ { - 4 }$ Polynomial decay3,0000.11.0</td></tr><tr><td rowspan=1 colspan=1>PrecisionGradient CheckpointingRandom Seed</td><td rowspan=1 colspan=1>bfloat16False42</td></tr><tr><td rowspan=1 colspan=1>Evaluation StrategySave StrategyModel Selection</td><td rowspan=1 colspan=1>Every 1,000 stepsEvery 5,000 stepsBest validation loss</td></tr></table>

Table A.14: Fine-tuning configuration for all models. ES = Early Stopping patience (epochs). All models use AdamW optimizer except where noted.
<table><tr><td>Hyperparameter</td><td>IsharaBART</td><td>mBART-50</td><td>mT5-Small</td><td>BanglaT5</td><td>BanglaGPT</td></tr><tr><td>Epochs</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td></tr><tr><td>Batch Size (per device)</td><td>16</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td>Gradient Accumulation</td><td>2</td><td>2</td><td>2</td><td>2</td><td>4</td></tr><tr><td>Effective Batch Size</td><td>32</td><td>32</td><td>32</td><td>32</td><td>64</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 3 }$ </td><td> $1 \times 1 0 ^ { - 4 }$ </td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>LR Schedule</td><td>Linear decay</td><td>Linear decay</td><td>Linear decay</td><td>Linear decay</td><td>Linear decay</td></tr><tr><td>Weight Decay</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td><td>0.01</td></tr><tr><td>Max Gradient Norm</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Precision</td><td>fp32</td><td>fp32</td><td>fp32</td><td>fp32</td><td>fp32</td></tr><tr><td>Beam Size</td><td>4</td><td>4</td><td>4</td><td>4</td><td></td></tr><tr><td>Generation Max Length</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>ES Patience</td><td>5</td><td>5</td><td>5</td><td>5</td><td>5</td></tr><tr><td>Model Selection Metric</td><td>BLEU-4</td><td>BLEU-4</td><td>BLEU-4</td><td>BLEU-4</td><td>Validation loss</td></tr></table>

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Task type</td><td>Seq2Seq language modeling</td></tr><tr><td>LoRA rank (r)</td><td>8</td></tr><tr><td>LoRA scaling factor (α) 16</td><td></td></tr><tr><td>LoRA dropout</td><td>0.1</td></tr><tr><td>Target modules</td><td>q_proj, k_proj, v_proj, out_proj</td></tr></table>

Table A.15: LoRA configuration used for parameter-eficient fine-tuning.