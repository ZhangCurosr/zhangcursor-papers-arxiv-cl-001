# GigaSpeech 3: Scaling Multi-Dialect Arabic Speech Recognition with Weakly Labeled Data

Junzhe Liu<sup>1</sup>, Yujie Tu<sup>2,3</sup>, Yifan Yang<sup>1</sup>, Peng Wang<sup>4</sup>, Haoran Wang<sup>1</sup>, Guodong Lin<sup>5</sup>, Leyan Liao<sup>6</sup>, Chengquan Zhang<sup>1</sup>, Zhifu Gao<sup>7</sup>, Xiangang Li<sup>7</sup>, Qinyuan Cheng<sup>8</sup>, Xipeng Qiu<sup>2,8</sup>, Wei-Qiang Zhang<sup>5,9</sup>, Guoguo Chen<sup>9,10</sup>, Kai Yu<sup>1</sup>, Xie Chen<sup>1,2,9</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>University of Chinese Academy of Sciences <sup>4</sup>The Chinese University of Hong Kong, Shenzhen <sup>5</sup>Tsinghua University <sup>6</sup>Dalian University of Technology <sup>7</sup>Alibaba Token Foundry <sup>8</sup>Fudan University <sup>9</sup>SpeechColab <sup>10</sup>Seasalt.ai

Abstract—Dialectal Arabic speech recognition remains constrained by the scarcity of dialect-resolved transcribed speech despite the abundance of web audio. We introduce GigaSpeech 3, a corpus of 39,000 hours and 10 million weakly transcribed speech segments covering Algerian, Egyptian, Iraqi, Saudi, and Syrian Arabic. Starting from 60 hours of manual transcription per dialect, we train labeling models using speech-recognition-biased self-supervised learning and apply them to dialect-matched web audio. Data selection and four rounds of noisy-student refinement for Saudi and Syrian Arabic yield the final corpus. We validate its utility through dialect-specific and multi-dialect speech recognition training and cross-domain evaluation. On GigaSpeechBench, the final dialect-specific models outperform their labeling models across all five dialects and achieve a macro-average word error rate of 23.86%, a 16.6% relative reduction from the strongest third-party system covering all five dialects, while outperforming the best third-party result on three dialects. A multi-dialect model using self-predicted dialect identities achieves a macroaverage word error rate of 25.51% while also supporting Modern Standard Arabic. Across eight conditions from three external corpora, a GigaSpeech 3-trained Zipformer achieves the lowest word error rate among six evaluated models in every condition, while adapting the Qwen3 speech recognizer on GigaSpeech 3 improves performance across all eight conditions. We release the corpus and recipes as a foundation for dialect-aware Arabic speech processing.

Index Terms—Arabic dialect speech recognition, multi-dialect automatic speech recognition, low-resource automatic speech recognition, self-supervised learning, noisy-student training

## I. INTRODUCTION

Arabic ASR is a multi-variety problem rather than a singlelanguage recognition setting. Modern Standard Arabic (MSA) is standardized and dominates education, news, and formal communication, whereas everyday speech consists of regional varieties with substantial differences in pronunciation, vocabulary, morphology, and spelling. A common coarse taxonomy distinguishes Maghrebi, Egyptian, Levantine, Gulf, and Iraqi or Mesopotamian Arabic, although boundaries vary and Iraqi is sometimes grouped with Gulf Arabic [1], [2]. Our five targets provide one country-level representative of each group: Algerian (DZA) for Maghrebi, Egyptian (EGY), Syrian (SYR) for Levantine, Saudi (SAU) for Gulf, and Iraqi (IRQ).

Arabic speech resources have developed along several directions. Challenge and benchmark datasets, including MGB-2, MGB-3, MGB-5, the Open Universal Arabic ASR Leaderboard, and GigaSpeechBench, have expanded evaluation across countries, domains, and recording conditions [3]–[7]. Large broadcast corpora such as QASR and SADA provide substantially more transcribed audio, but their coverage is shaped by Al Jazeera programming or Saudi television and therefore does not yield balanced country-level supervision for the five dialects studied here [8], [9]. Carefully annotated corpora such as Casablanca, CARDAMOM, and Hamsa provide country-level or finer dialect labels, but their scale of tens of hours primarily supports evaluation, adaptation, and linguistic analysis rather than the data-intensive training targeted here [10]–[12]. At the other extreme, web-scale collections make much more Arabic audio available: YODAS v3 reports roughly 25,000 hours of Arabic within a 1.1-million-hour multilingual corpus, yet its locale metadata are incomplete, and its Arabic partition does not provide reliable country-level dialect labels [13]. The remaining gap is therefore not the availability of Arabic speech data, but the lack of large-scale, dialect-specific training data with reliable dialect labels.

Preserving dialect identity matters because a pooled “Arabic” label hides systematic acoustic and lexical differences. We retain country-level labels through metadata-based collection and curation, then use them to organize weak supervision for dialect-specific and multi-dialect recognizers. The multidialect recognizer uses automatically predicted dialect IDs at inference. A matched comparison with known IDs shows that routing errors increase WER, especially for EGY and SYR. We therefore evaluate this practical cost while keeping improvements to dialect-identification methodology outside the claimed contribution.

Large-scale self-supervised pretraining offers one route around the transcription bottleneck. Methods such as wav2vec 2.0, HuBERT, and XLS-R learn speech representations from untranscribed audio and adapt them with much smaller labeled sets [14]–[16]. Multilingual pretraining can further share statistical strength across languages [17], [18], but a general multilingual model does not create dialectspecific transcripts for training a compact recognizer. In practice, dialect-matched web audio may be plentiful while the supervision needed to train Arabic dialect ASR remains scarce.

Automatic weak labeling turns this asymmetry into a corpus-construction strategy. A labeling model transcribes unlabeled speech, and a downstream model learns from the resulting weakly labeled audio–text pairs. Iterative labeling and related noisy-student methods have produced large gains in low-resource and multilingual ASR [19]–[22]. GigaSpeech 2 extends this idea into an end-to-end data pipeline that crawls web audio, generates transcripts, filters unreliable pairs, and iteratively refines the labels [23]. For Arabic dialects, however, the decisive question is whether automatically generated transcripts are accurate and diverse enough to become a reusable training corpus. A strong test is whether an independently initialized downstream model can improve over the labeling model without inheriting its pretrained parameters.

We present GigaSpeech 3, a large-scale weakly labeled speech corpus for the five target dialects. We partition web speech by dialect and keep labeled, unlabeled, and test channels disjoint. Starting from a small manually transcribed set per dialect, the ASR-biased self-supervised recipe introduced by VietASR produces labeling models for the larger matched collections [24]. Selection and NST refinement yield 39,028 hours and 10 million segments, with 4,361–11,461 hours per dialect. The released SAU/SYR partitions are filtered through the fourth NST round; their corresponding final recognizers are denoted $r _ { 4 }$ (NST round 4 models). We validate the weak supervision through new dialect-specific ASR models; a multidialect model pools the dialect inventories with separately labeled MSA data and uses predicted dialect IDs at inference.

The central result is that GigaSpeech 3 provides useful supervision beyond the labeling models that generated it. Final dialect-specific models, including NST r4 for SAU/SYR, improve over their corresponding labeling models on all five dialects and reach a 23.86 macro-average WER on GigaSpeechBench. The multi-dialect model reaches 25.51 with predicted LID, compared with 28.62 for the strongest complete third-party system. The dialect-specific models lead on DZA, EGY, and SAU; the strongest third-party system retains an advantage on IRQ and SYR. On eight external conditions from MASC, Casablanca, and SADA, a GigaSpeech 3-trained Zipformer obtains the lowest WER among the six evaluated models in every condition. Corpus-based Qwen3-ASR finetuning also improves it on all eight, supporting transfer across evaluation domains and model families.

Our main contributions are summarized as follows:

• Large-scale dialect-resolved corpus. We introduce GigaSpeech 3, comprising 39,028 hours and 10 million weakly transcribed segments across five Arabic dialects (DZA, EGY, IRQ, SAU, and SYR), bridging the gap between small manually annotated dialect corpora and large Arabic speech collections lacking reliable dialect labels.

• Validation of weak supervision. Final dialect-specific models outperform their labeling models across all five dialects, achieving a macro-average WER of 23.86 and a 16.6% relative reduction over the strongest third-party system covering all five dialects on GigaSpeechBench.

• Evaluation across settings and domains. We evaluate multi-dialect recognition with predicted-LID decoding and MSA coverage, quantify the WER degradation caused by predicted-LID routing relative to ground-truth LID routing, and demonstrate cross-domain improvements on MASC, Casablanca, and SADA, including gains from fine-tuning Qwen3-ASR on GigaSpeech 3.

## II. RELATED WORK

A. Low-resource ASR through self-supervised and multilingual learning

Self-supervised speech learning reduces the amount of transcription required to adapt an acoustic model. wav2vec 2.0 learns quantized latent targets with a contrastive objective, while HuBERT predicts clustered hidden units in masked regions [14], [15]. XLS-R scales this paradigm across languages so that low-resource targets can benefit from crosslingual pretraining [16]. A complementary line trains a single multilingual recognizer or adapts modular components across languages [17], [18], [25]. These approaches improve representations and parameter sharing, but they still require task-specific transcripts during adaptation. Our work uses self-supervised learning to construct dialect-specific labeling models, then converts dialect-matched unlabeled speech into a much larger weakly labeled corpus.

## B. ASR-biased speech pretraining

HuBERT obtains masked-prediction targets by clustering acoustic features and then representations learned in earlier pretraining iterations [15]. A substantial line of work has made these targets more specific to ASR by injecting supervision into their construction [26]. Supervision-guided codebooks introduced both PBERT, which uses phoneme alignments from a hybrid recognizer, and CTC clustering, which quantizes representations from a supervised CTC encoder; both recognizers can be trained on the same limited labeled subset used for downstream fine-tuning [27]. Chen et al. used an ASR encoder to bypass HuBERT’s initial acoustic-clustering iteration, reducing the compute required to obtain useful maskedprediction targets [28]. Later methods further close the loop between ASR fine-tuning and speech pretraining. ASBERT extracts linguistically informed discrete targets from a fine-tuned ASR encoder and uses them in the next pretraining round [29]. Biased SSL likewise fine-tunes the target-generating model toward ASR before reclustering its representations, improving both recognition accuracy and convergence speed [30]. CTCBERT addresses a related problem at the objective level: it replaces framewise cross-entropy with CTC so that masked prediction is less sensitive to errors in clustered or aligned target boundaries [31]. Taken together, these approaches show that labeled speech can guide the target generator, the iterative training path, or the pretraining loss.

VietASR builds on this prior line by combining ASR-biased iterative targets with an Fbank front end and a multi-rate

![](images/b9c38ec7d9e55ea3a758205e1de896f8ea8ae825dd4a811b0b4a51c356efaa35.jpg)  
Fig. 1. Overview of GigaSpeech 3 construction and validation. Channel-disjoint manual and web data train dialect-specific VietASR labeling models, whose hypotheses are filtered to form the five-dialect corpus. Independent Zipformer training and Qwen3-ASR adaptation evaluate the resulting weak supervision; labeled MSA is added only to the Multi-Dialect Zipformer and Qwen training recipes.

Zipformer [32] encoder. It starts from an RNN-T model trained on a small labeled set, clusters that encoder’s representations, predicts the resulting units on masked Fbank sequences, and fine-tunes the pretrained encoder for ASR. Later iterations repeat target extraction and pretraining with the improved encoder. This design avoids a purely acoustic first-round codebook and adapts the procedure to a compact recognizer that supports online and offline decoding. In the original Vietnamese study, a 68M-parameter model trained with 50 hours of labeled speech was competitive with much larger open and commercial systems [24], [33].

We adopt this recipe to obtain five dialect-specific SSLadapted labeling models; the recipe itself is a foundation rather than the claimed novelty of this work. Our main question begins after those models are trained: whether their transcripts constitute reusable weak supervision for independent downstream models. GigaSpeech 3 evaluates the released corpus through from-scratch Zipformer training and cross-architecture Qwen adaptation. Improvements from independently initialized models provide evidence for the utility of the weakly labeled data beyond the recognizers that generated it.

## C. Automatic weak labeling at scale

Automatic weak labeling uses a recognizer’s hypotheses as supervision for otherwise unlabeled audio. Iterative variants refresh those hypotheses as the acoustic model improves, and their effectiveness depends on decoding, augmentation, and control of label errors [19]. The approach also scales to multilingual settings: a supervised multilingual model can be adapted to a target language, used to label target-language audio, and followed by a final model trained from scratch or fine-tuned on the generated labels [20]. Large-scale studies on low-resource speech further combine multiple transcription and evaluation models to select useful audio–text pairs [21].

GigaSpeech 2 integrates these ideas into a reproducible corpus construction pipeline, including web-audio collection, automatic transcription, alignment, multidimensional filtering, and modified noisy-student training [23]. Our corpus construction follows the same broad principle, but the experimental focus differs in two respects. First, compact dialect-specific labeling models trained with ASR-biased SSL generate the Arabic dialect transcripts. Second, we separate the gain obtained from the initial weakly labeled corpus from the additional gain obtained through filtering and iterative refinement. This separation tests whether the initial weak labels are already useful before the more selective refinement stage.

Table I summarizes the technical progression represented by the GigaSpeech name. The original GigaSpeech scaled highquality supervised English ASR by collecting and validating public audio with available transcripts [34]. GigaSpeech 2 shifted the emphasis from existing text pairs to unlabeled web audio for three low-resource Southeast Asian languages, using general-purpose models for transcription, forced alignment, filtering, and iterative refinement [23]. GigaSpeech 3 retains large-scale weak labeling but moves the unit of coverage from language to dialect: limited dialect-specific transcripts first adapt the labeling models, and the generated corpus is then validated with independently initialized and pretrained downstream recognizers. Thus, the three corpora share a concern with scalable public-web speech, while differing in language scope, supervision source, and the role of trained models in corpus construction.

YODAS v3 provides a complementary form of web-scale weak supervision: 1.1 million hours of 48-kHz multichannel audio across 147 languages, with platform-provided transcripts when available [13]. Its paper reports roughly 25,000 hours of Arabic using the uploader locale as a language proxy. The current dataset release retains this field as locale\_lang, but the dataset card notes that it is missing for many recordings and organizes language directories by caption-track language when available. Consequently, its partition is broad Arabic rather than a reliable country-level dialect annotation, and the paper does not report Arabic ASR results. GigaSpeech 3 instead curates five dialects, regenerates their labels with dialect-specific recognizers, and evaluates those labels through Arabic dialect ASR models trained from scratch.

## D. Arabic and multi-dialect speech recognition

Spoken Arabic is commonly described through five coarse regional groups: Maghrebi, Egyptian, Levantine, Gulf, and Iraqi or Mesopotamian Arabic [1]. Maghrebi varieties cover much of North Africa and reflect substantial internal variation and contact with Berber and French; Levantine covers Syria, Lebanon, Jordan, and Palestine; Gulf covers the Arabian Peninsula and Gulf states; Egyptian forms a distinct group; and Iraqi is often treated separately because of its distinctive phonology and morphology, although some taxonomies place it within a broader Gulf group. This hierarchy is only approximate, as variation also occurs by country, city, and social group, but it motivates our selection of DZA, EGY, SYR, SAU, and IRQ as representatives of the five broad groups.

Arabic speech resources have historically concentrated on MSA and a few prominent dialects. MGB-2 released 1,200 hours of lightly supervised Al Jazeera broadcast speech, most of it MSA, with the remaining speech grouped into Egyptian, Gulf, Levantine, and North African varieties [3]. MGB-3 shifted evaluation toward 16 hours of multi-genre Egyptian YouTube speech and paired ASR with five-way identification of Egyptian, Gulf, Levantine, North African, and MSA speech [4]. MGB-5 subsequently introduced Moroccan ASR together with country-level dialect identification data collected from 17 Arabic-speaking countries [5]. These challenges established increasingly dialect-focused evaluations, but did not provide large-scale ASR training data for country-level varieties.

Larger broadcast resources improve training scale while retaining their source-domain concentration. QASR contains 2,000 hours of lightly transcribed, multi-dialect Al Jazeera speech and associated linguistic metadata [8]. SADA contributes 668 hours from 57 Saudi TV programs, covering read and spontaneous speech dominated by Saudi varieties with smaller amounts of Yemeni, Egyptian, and Levantine Arabic [9]. Casablanca added human-transcribed conversational speech for eight countries, including Algerian, Egyptian, Emirati, Jordanian, Mauritanian, Moroccan, Palestinian, and Yemeni Arabic [10]. Together, these corpora broaden domain and regional coverage, but their dialect inventories, scales, and transcription conventions remain unevenly distributed.

Recent manually annotated corpora address complementary gaps in dialect coverage and label granularity. CARDAMOM contains approximately 40 hours of YouTube speech from Egypt, Jordan, Lebanon, Mauritania, Palestine, and Saudi Arabia, with native-speaker transcriptions and labels for 21 subcountry micro-dialects, code-switching, and perceived gender [11]. It is designed for release as standoff annotations—video IDs, timestamps, transcripts, and metadata with reconstruction code—rather than redistributed audio. Hamsa provides approximately 11.6 hours of similarly focused Emirati speech and shows that dialect-matched fine-tuning improves pretrained Whisper models [12]. These resources prioritize precise gold supervision and fine-grained evaluation; GigaSpeech 3 instead scales country-level weak supervision to thousands of hours per target dialect and tests it with an independently initialized recognizer. CARDAMOM overlaps with our EGY and SAU coverage and resolves within-country varieties, whereas Hamsa supplies a complementary dialect-specific gold benchmark. Their results are therefore complementary rather than directly comparable baselines.

Modeling work has followed three main directions. Dialectspecific adaptation of Whisper and related multilingual models improves covered varieties but generalizes poorly to unseen dialects [35]. Joint systems share acoustic and lexical knowl edge across Arabic varieties, sometimes adding dialect tokens or combining Arabic with English and French to handle codeswitching [36]. Country-level routing has also been studied separately: ADI17 provides roughly 3,000 hours of weakly labeled YouTube speech from 17 countries plus 57 hours of manually refined development and test speech, illustrating both the utility and noise of assigning dialects from source channels [37]. Arabic-specific pretraining and weak supervision scale coverage further: ArTST-based studies analyze how pretraining dialect coverage affects generalization, while recent large weakly supervised systems combine broad pretraining with continual supervised fine-tuning [2], [38], [39]. Across these approaches, gains remain uneven by country and domain, showing that broad Arabic coverage does not remove the need for dialect-matched supervision [6].

GigaSpeechBench provides the common evaluation used in this work, covering real-world speech across low-resource languages, dialects, accents, domains, and age groups [7]. We use its released Arabic test sets and keep test channels separate from the channels used to collect training audio. The multi-dialect recognizer uses predicted dialect IDs in the main comparison, and a matched diagnostic with and without ground-truth LID guidance measures the effect of routing errors. Improving dialect identification itself is outside the contribution of this paper.

Arabic orthographic variation also makes evaluation sensitive to text normalization. Surface forms that are linguistically close can produce different WERs under different mappings of letter variants, punctuation, and dialect spellings. We therefore treat the normalizer, test manifest, and decoding condition as part of the experimental protocol. Results produced under incompatible protocols are reported separately and are not used to support direct superiority claims.

TABLE I  
LANGUAGE COVERAGE, DURATION, AND SUPERVISION TYPE IN THE GIGASPEECH SERIES.
<table><tr><td>Corpus</td><td>Language coverage</td><td>Duration (h)</td><td>Supervision type</td></tr><tr><td>GigaSpeech [34]</td><td>English</td><td>10k</td><td>Human-provided transcripts</td></tr><tr><td>GigaSpeech 2 [23]</td><td>Thai, Indonesian, and Vietnamese</td><td>30k</td><td>Refined weak labels</td></tr><tr><td>GigaSpeech 3 (ours)</td><td>Arabic: DZA, IRQ, SAU, SYR, and EGY</td><td>39k</td><td>Refined weak labels</td></tr></table>

## III. DATA AND PROBLEM SETUP

## A. Dialect scope

The primary study covers five country-level Arabic varieties: Algerian (DZA), Egyptian (EGY), Iraqi (IRQ), Saudi (SAU), and Syrian (SYR). Together they provide representatives of the Maghrebi, Egyptian, Mesopotamian, Gulf, and Levantine groups introduced in Section II. This mapping is deliberately coarse: each country contains substantial regional and social variation, and a country tag should not be interpreted as a complete linguistic taxonomy. Our goal is to construct useful dialect-matched supervision at a scale that is currently unavailable for these five evaluation conditions.

MSA is additional model coverage. The Multi-Dialect Zipformer uses the MGB-2 dataset; no MSA-specific VietASR model is trained. The Qwen adaptation study also uses manually labeled MSA speech from MGB-2 [3]. This additional supervision is excluded from the five-dialect weak-label inventory reported below.

## B. Web collection and split integrity

We collect candidate audio from public web channels and screen sources before model training. The source-level filters use channel identifiers and URLs, title metadata, publication dates, and recording duration to remove irrelevant or unsuitable material. Accepted recordings are segmented into utterance-like regions and pass basic audio-quality checks. A small subset is selected for professional transcription and forms the manually labeled set; the remaining material forms the dialect-specific unlabeled pool.

Separation is enforced at the channel level rather than only at the segment level. Channels assigned to the manually labeled set, the unlabeled collection, and the GigaSpeechBench evaluation data are mutually disjoint. This stronger unit of separation reduces the chance that repeated speakers, recurring programs, or channel-specific acoustic conditions appear on both sides of evaluation. It also makes the corpus-construction claim more demanding: each dialect-specific labeling model must generalize to new channels when generating weak labels.

The collected speech is naturally heterogeneous. It includes differences in recording equipment, background noise, speaking style, topic, and the amount of MSA or code-switching embedded in nominally dialectal content. We preserve this diversity in the large unlabeled pool and rely on dialectmatched labeling models plus downstream selection to construct the final corpus. The resulting task is therefore realworld dialectal speech recognition rather than recognition of a single controlled domain.

TABLE II  
TRAINING-DATA DURATION FOR THE FIVE TARGET DIALECTS (HOURS).
<table><tr><td rowspan="2">Dialect</td><td colspan="2">Duration (h)</td></tr><tr><td>Human-labeled Unlabeled Refined weak-labeled</td><td></td></tr><tr><td>DZA</td><td>61.1 17,812</td><td>11,461</td></tr><tr><td>IRQ</td><td>63.2 15,531</td><td>10,162</td></tr><tr><td>SAU</td><td>58.3 20,135</td><td>4,512</td></tr><tr><td>SYR</td><td>59.0 14,905</td><td>4,361</td></tr><tr><td>EGY</td><td>51.1 12,183</td><td>8,533</td></tr><tr><td>Total</td><td>292.7 80,566</td><td>39,028</td></tr></table>

## C. Supervision inventory

Table II summarizes the supervision available to the five target dialects. The source collections contain 12,183–20,135 hours of unlabeled speech per dialect, whereas only 51.1–63.2 hours per dialect are manually transcribed. The final weaklabel inventories contain ∼10 million segments and 39,028 hours, ranging from ∼4,361 hours for SYR to 11,461 hours for DZA. The released SAU and SYR partitions comprise the filtered weak-label data after four NST rounds.

## D. Position relative to existing Arabic resources

Table III separates two directions in Arabic speech-resource development. Casablanca provides human-labeled multidialect training and evaluation data with country labels, CAR-DAMOM emphasizes fine-grained regional and sociolinguistic annotation, and Hamsa supports Emirati ASR adaptation and evaluation [10]–[12]. These resources prioritize transcription quality and linguistic detail at a tens-of-hours scale. YODAS v3 instead provides web-scale multilingual audio and reports roughly 25,000 hours of Arabic, but its locale\_lang metadata are incomplete, and its current ar organization does not supply reliable country-level dialect labels [13].

GigaSpeech 3 occupies a different point in this design space. It uses a few dozen manually transcribed hours to train dialectspecific labeling models, converts much larger web collections into country-level weak supervision, and validates the resulting corpus with independently trained recognizers. The corpus is therefore not a substitute for fine-grained gold corpora; it supplies the scale required for representation learning and robust ASR training while retaining a usable dialect identity.

TABLE III  
COMPARISON WITH RECENT ARABIC SPEECH RESOURCES.
<table><tr><td>Resource</td><td>Country codes</td><td>Scale (h)</td><td>Supervision</td></tr><tr><td>Casablanca [10]</td><td>DZA, EGY, ARE, JOR, MRT, MAR, PSE, YEM</td><td>48</td><td>Human labels</td></tr><tr><td>CARDAMOM [11]</td><td>EGY, JOR, LBN, MRT, PSE, SAU</td><td>40</td><td>Human labels</td></tr><tr><td>Hamsa [12]</td><td>ARE</td><td>12</td><td>Human labels</td></tr><tr><td>YODAS v3 [13]</td><td>Unspecified</td><td>25k</td><td>Weak labels</td></tr><tr><td>GigaSpeech 3 (ours)</td><td>DZA, IRQ, SAU, SYR, EGY</td><td>39k</td><td>Refined weak labels</td></tr></table>

## IV. METHOD

## A. Overview

Figure 1 separates corpus construction into channel-disjoint data preparation, dialect-specific weak-label generation and refinement, and downstream corpus validation. We first collect and partition dialect-matched web speech. A VietASR ASRbiased self-supervised learning procedure converts a small labeled set and a large unlabeled pool into a dialect labeling model. Its transcripts are selected to form an initial weak-label inventory. SAU and SYR then undergo four rounds of noisystudent training (NST), and the final round produces their released inventories and corresponding $r _ { 4 }$ recognizers. New recognizers test whether the resulting audio–text pairs provide useful supervision beyond the labeling-model parameters.

## B. VietASR dialect-labeling model construction

The labeling-model pipeline follows the four-stage VietASR design [24]. In Stage 1, a labeled-only Zipformer transducer, denoted Iter0, is trained on the dialect’s manual transcripts. Although Iter0 is not yet competitive for every dialect, its encoder has learned speech–text alignment cues that are absent from purely acoustic features. Stage 2 runs this supervised encoder on speech and clusters its hidden representations into a discrete codebook. These clusters become ASR-biased targets: they preserve acoustic regularities while being shaped by a recognizer trained on the target dialect.

Stage 3 pretrains a fresh encoder on the large unlabeled pool. Contiguous spans of filter-bank features are masked, and the encoder predicts the cluster identity assigned to each masked frame. For an utterance X, masked positions M, and codebook labels $c _ { t } ,$ the objective is the masked cross-entropy

$$
\mathcal { L } _ { \mathrm { S S L } } = - \sum _ { t \in \mathcal { M } } \log p ( c _ { t } \mid \widetilde { X } ) ,\tag{1}
$$

where $\widetilde { X }$ denotes the masked feature sequence. This retains the computationally efficient Fbank front end and multi-rate Zipformer backbone while steering representation learning toward ASR-relevant distinctions.

In Stage 4, a transducer predictor and joiner are attached to the pretrained encoder, and the full recognizer is finetuned on the same manually labeled set. This produces the SSL-adapted labeling model, denoted Iter1, that generates the corpus transcripts. The Iter0–Iter1 comparison holds the quantity of manual supervision fixed and therefore measures the gain obtained by ASR-biased pretraining. VietASR permits later iterations that refresh the codebook from the improved encoder; the labeling models reported here use one SSL-plusfine-tuning iteration.

## C. Weak-label generation and selection

Each Iter1 labeling model decodes only the dialect-matched web channels that were excluded from the labeled and evaluation sets. The hypotheses are normalized into the same writing space used for downstream training, paired with their audio segments, and screened using duration, recognition score, duplication, and text-quality criteria. The output is a set $\widetilde { \mathcal { D } } _ { d } = \{ ( x _ { i } , \hat { y } _ { i } ) \}$ for dialect $d ,$ where $\hat { y } _ { i }$ is a model-generated transcript rather than a manual reference.

The distinction between an accurate labeling model and a useful weak-label corpus is central to the experimental design. A labeling model can achieve low WER because of its pretrained parameters even if its generated transcripts are not reusable. We therefore initialize the primary downstream Zipformer independently and optimize it only on $\bar { \mathcal { D } } _ { d }$ . Improvement over Iter1 under this constraint is evidence that the audio–text pairs contain useful training information beyond the labeling-model weights.

The final corpus contains 10 million segments and 39k hours across the five dialects. DZA, IRQ, and EGY contribute their selected weak-label inventories; SAU and SYR contribute the r4-filtered inventories produced by the refinement procedure below. Selection is dialect-specific: a common target number of hours would hide differences in source composition and label reliability.

## D. From-scratch training and iterative refinement

The primary downstream system is a Zipformer2 transducer trained from random initialization on weak labels only. Its predictor and joiner are learned together with the encoder, so neither acoustic representations nor decoding parameters are inherited from Iter1. Dialect-Specific models use one corpus partition and one dialect-specific output vocabulary. This setting is the most direct test of the released corpus because the downstream model can be reproduced from the audio–text pairs without access to the labeling-model checkpoint.

SAU and SYR additionally use GigaSpeech 2-style noisystudent training (NST), which iteratively refines weak supervision and trains successive recognizers [23]. We denote the initial weak-label-trained recognizer in each refinement chain by $r _ { 0 } .$ . It is the starting point before NST refinement, whereas Iter0 and Iter1 denote the labeled-only and SSL-adapted labeling models described above. These two numbering schemes therefore refer to different stages of corpus construction.

An NST round comprises filtering the current audio–text training pairs, training the next recognizer on the selected weak supervision, and regenerating hypotheses to continue refinement. We use $r _ { k }$ to denote the recognizer after k completed refinement rounds: $r _ { 1 } , r _ { 2 }$ , and $r _ { 3 }$ are intermediate models, and $r _ { 4 }$ is the final model after four rounds. A round is a complete refinement-and-training cycle, rather than a single optimization epoch. The round index identifies the model’s position in this chain, not a separate model architecture.

The released SAU and SYR partitions are the final, fourthround-filtered weak-label inventories, containing 4,512 and 4,361 hours, respectively. We refer to these inventories as the r4-filtered data and to the corresponding final recognizers as the NST r4 models. Their WERs are used in the main dialectspecific comparison; Table X traces their improvement from $r _ { 0 }$ through $r _ { 4 }$ . We restrict refinement to these two dialects because the adopted recipe assumes a sufficiently accurate labeling model.

## E. Multi-dialect training

The Multi-Dialect Zipformer pools the five weak-label corpora and the MGB-2 dataset. During training, the groundtruth dialect token is prepended to each transcript. The model learns to predict this token followed by the transcript, jointly learning dialect identification and ASR. Dialect tokens extend the shared output vocabulary; training uses no separate LID classification head or LID loss.

At inference, the main evaluation uses predicted IDs to condition recognition. Ground-truth-ID decoding is reserved for a matched diagnostic that measures routing errors. This factorization lets one acoustic model share evidence across varieties while preserving dialect-conditioned output behavior; improving dialect identification itself is outside our scope.

The multi-dialect experiment asks whether the weak-label corpora support one recognizer with competitive aggregate performance and MSA coverage. Differences between the Dialect-Specific and Multi-Dialect columns are not treated as a pure measure of cross-dialect transfer because the models also differ in data mixture, vocabulary, conditioning, and automatic-routing errors.

## F. Cross-architecture transfer

Finally, we fine-tune Qwen3-ASR-1.7B using weak transcriptions from our five dialect partitions together with manually labeled MSA speech from MGB-2, and evaluate the adapted model without and with ground-truth LID guidance at decoding. These are decoding conditions, not separate claims about training with or without dialect conditioning. This model is substantially larger and already pretrained, so it is not the primary controlled validation. Instead, it tests portability: if the generated transcripts improve both an independently initialized Zipformer and a pretrained large recognizer, their value is less likely to depend on one architecture or optimization recipe.

## V. EXPERIMENTAL SETUP

## A. Evaluation questions

The experiments are organized around five questions. RQ1 asks whether ASR-biased SSL improves a dialect labeling model when manual supervision is fixed. RQ2 asks whether a new Zipformer trained only on weak labels can improve further, thereby validating the generated corpus independently of the labeling-model parameters. RQ3 asks whether five dialect corpora can train one dialect-conditioned recognizer that remains competitive on every variety and also covers MSA. RQ4 asks whether the weak labels remain useful under iterative refinement and when transferred to Qwen3-ASR-1.7B. RQ5 asks whether the resulting models generalize to independently released Arabic corpora rather than benefiting only from the GigaSpeechBench evaluation domain.

These questions determine the comparison chain. Iter0 uses only manual transcripts. Iter1 adds SSL on the corresponding unlabeled collection and then fine-tunes on the same transcripts. The weak-only system discards the Iter1 parameters and trains on generated labels from scratch. The multi-dialect and Qwen experiments broaden the model class after the dialect-specific resource has been validated.

## B. Recognition systems

All primary recognizers use the icefall Zipformer2 transducer recipe. Dialect-Specific Zipformers use dialect-specific 500-unit BPE tokenizers where recorded, whereas the Multi-Dialect Zipformer uses a shared 1,500-unit BPE vocabulary and automatically predicted LID at decoding. Final Dialect-Specific SAU/SYR results use the NST r4 models corresponding to their r4-filtered released inventories. The multi-rate Zipformer encoder is paired with a stateless predictor and transducer joiner. Checkpoint averaging and decoding follow the settings recorded in the experiment registry.

The Qwen3-ASR-1.7B study uses the released base recognizer and two decoding conditions for the adapted model: without and with ground-truth LID guidance. Fine-tuning combines weakly transcribed dialect speech with manually labeled MSA data from MGB-2; the MSA condition therefore does not test weak supervision alone. QwenCleo<sup>1</sup> and Whisper-CS<sup>2</sup> provide contextual Arabic-specialized large-model baselines, but they are not data-matched controls.

## C. Test sets and normalization

We use the released GigaSpeechBench Arabic test sets as the primary evaluation [7]. The test audio and reference transcripts are identical for the proposed and off-the-shelf comparisons, and all hypotheses pass through the benchmark’s common Arabic normalizer. Historical exploratory runs that used a different manifest or normalization path are excluded from the reported comparison.

TABLE IV  
DURATION OF THE ARABIC EVALUATION SETS BY DIALECT (HOURS). DEVELOPMENT AND TEST PARTITIONS ARE MERGED FOR MASC, CASABLANCA, AND SADA.
<table><tr><td>Dataset DZA</td><td>Duration (h) IRQ SAU SYR EGY</td></tr><tr><td>GigaSpeechBench</td><td>20.0 20.4 20.3 20.2 20.5</td></tr><tr><td>MASC</td><td>0.1 0.7 1.4 6.6 4.5</td></tr><tr><td>Casablanca</td><td>1.9 2.0</td></tr><tr><td>SADA</td><td>一 19.9</td></tr><tr><td>Total</td><td>22.0 21.0 41.6 26.8 27.0</td></tr></table>

To test whether the conclusions depend on this single benchmark domain, we additionally evaluate on subsets of three independently released Arabic corpora: SADA [9], Casablanca [10], and MASC [40]. We retain the target varieties represented in our training data: DZA and EGY from Casablanca; DZA, IRQ, SAU, SYR, and EGY from MASC; and the SAUtargeted SADA evaluation set. For each corpus, its official development and test partitions are concatenated into one evaluation pool. Neither partition is used for model selection, hyperparameter tuning, checkpoint selection, or textnormalization design. Table IV summarizes the duration of the primary GigaSpeechBench test sets and the three external evaluation corpora by dialect.

For the external evaluation suite, we use the same final checkpoints as in the primary comparison, including the SAU/SYR NST r4 models. The Multi-Dialect Zipformer and fine-tuned Qwen use automatically predicted LID without ground-truth LID guidance. Decoding settings are fixed before evaluation, and the same Arabic text normalizer is applied to references and hypotheses. Results are reported for each corpus–variety pair instead of only after pooling all utterances. This prevents the much larger SADA and MASC-SYR subsets from obscuring performance on smaller conditions; in particular, the three-minute MASC-DZA subset is retained for completeness but is not treated as strong standalone evidence.

Word error rate (WER) is the primary metric and is reported as a percentage. The main system comparison averages DZA, EGY, IRQ, SAU, and SYR; the LID diagnostic and Qwen adaptation tables average these five dialects plus MSA. The external evaluation averages its eight corpus–variety conditions. All averages are unweighted means of the listed WERs, not pooled word-error rates, and should be compared only within the same coverage.

## D. External systems and dialect-ID diagnostic

The third-party values in Table VI are reproduced from the common GigaSpeechBench v3 evaluation table. The comparison includes commercial APIs, general multilingual recognizers, and publicly released Arabic-capable models. We use the strongest complete system as the principal external reference and italicize the best third-party result for each dialect; bold and underlining mark the best and second-best values overall. This avoids constructing a single baseline by selecting different systems after observing each test set.

The reported Multi-Dialect Zipformer is obtained by averaging checkpoints from epochs 4–8 and uses automatically predicted variety tags in the main GigaSpeechBench comparison. Using this same averaged model, a paired diagnostic compares decoding without and with ground-truth LID guidance and reports the classifier’s confusion matrix. Its WERs characterize routing cost under a matched model and test set. The classifier is an inference component; dialect-identification methodology is not claimed as a contribution.

The duration-stratified LID analysis uses the same averaged model with greedy decoding and predicted IDs. During training, the ground-truth dialect token is prepended to the transcript so that LID and ASR are learned jointly under the transducer objective. We group utterances into durations below 3 seconds, from 3 to less than 10 seconds, and at least 10 seconds. Within each variety and bucket, LID accuracy is the fraction assigned to the reference variety, and sample counts accompany the accuracies to show the evaluation support.

## VI. RESULTS AND ANALYSIS

## A. From manual labels to reusable weak supervision

Table V addresses RQ1 and RQ2 by comparing the labeling models with the final corpus-trained systems. ASR-biased SSL followed by labeled fine-tuning improves every dialect over its labeled-only Iter0 model. The relative WER reductions are 28.1% for DZA, 26.2% for IRQ, 27.6% for EGY, and 74.3% for SYR. SAU starts from a much stronger 13.17 WER and improves by a smaller 1.4% relative. The spread shows that SSL is most valuable where the few-dozen-hour supervised baseline remains weak.

The final Dialect-Specific Zipformers improve over their corresponding Iter1 labeling models on all five dialects. DZA, IRQ, and EGY reach 33.64, 29.99, and 29.32 WER, respectively, corresponding to relative reductions of 12.6%, 4.3%, and 7.8%. SAU and SYR reach 12.12 and 14.25 after four NST rounds, improving over Iter1 by 6.6% and 5.7%. Their released 4,511.62-hour and 4,360.75-hour inventories are the r4-filtered weak-label data. These two final results therefore measure the full construction-and-refinement pipeline; the round-wise analysis below isolates their NST gains.

VietASR and downstream corpus training address complementary bottlenecks. VietASR uses limited manual transcripts to build an accurate labeling model; weak labeling then creates much larger audio–text training sets for new recognizers. The downstream models learn from these pairs without consuming the manual transcripts or inheriting Iter1’s SSL-adapted parameters. Their performance supports the corpus as reusable supervision, while the SAU/SYR refinement chains account for the additional processing used in those two releases.

TABLE V  
GIGASPEECHBENCH WER (%) BY MODEL AND DIALECT. THE FINAL MODELS USE THE RELEASED WEAK TRANSCRIPTIONS; SAU AND SYR INCLUDE FOUR-ROUND NST REFINEMENT.
<table><tr><td>Model</td><td>DZA</td><td>IRQ</td><td>SAU</td><td>SYR</td><td>EGY</td><td>MSA</td></tr><tr><td>Labeled only</td><td>53.47</td><td>42.50</td><td>13.17</td><td>58.88</td><td>43.92</td><td>15.98</td></tr><tr><td>SSL + FT</td><td>38.47</td><td>31.35</td><td>12.98</td><td>15.11</td><td>31.81</td><td>一</td></tr><tr><td>Dialect-Specific</td><td>33.64</td><td>29.99</td><td>12.12</td><td>14.25</td><td>29.32</td><td></td></tr><tr><td>Multi-Dialect</td><td>36.78</td><td>28.85</td><td>12.81</td><td>16.33</td><td>32.80</td><td>10.83</td></tr></table>

## B. Comparison with third-party systems

Table VI compares systems with complete results on the five target dialects in GigaSpeechBench v3. The strongest complete third-party system is Qwen3.5-Omni-Plus, with a 28.62 macroaverage WER. Our final Dialect-Specific Zipformers reach 23.86, a 16.6% relative reduction. The automatically routed Multi-Dialect Zipformer reaches 25.51, a 10.9% reduction, while ground-truth LID guidance lowers its diagnostic result to 23.62. The dialect-specific aggregate includes the NST r4 models for SAU and SYR.

The final Dialect-Specific Zipformers outperform the best third-party values on DZA, SAU, and EGY by 23.9%, 26.8%, and 21.0%, respectively. Under automatic routing, Qwen3.5- Omni-Plus remains stronger on IRQ and SYR at 28.54 and 13.76, compared with 28.85 and 16.33 for the Multi-Dialect model. Ground-truth LID guidance lowers the results to 27.84 and 13.18 on these two dialects, showing that routing errors account for part of the remaining gap. The practical automatically routed system nevertheless improves substantially over the strongest complete third-party macro-average.

The primary systems are compact Zipformer transducers trained from the constructed corpus, whereas several baselines are large pretrained or commercial recognizers. Their lower macro-average WER shows the value of dialect-matched weak supervision for compact ASR. The remaining IRQ and SYR gaps also show that corpus scale and specialization do not guarantee superiority on every dialect.

## C. Cross-corpus validation

Table VII answers RQ5 on eight corpus–variety conditions from MASC, Casablanca, and SADA. A GigaSpeech 3-trained Zipformer obtains the lowest WER among the six evaluated systems in every condition. The Multi-Dialect Zipformer is strongest on four of the five MASC subsets and reduces the MASC macro-average from 32.64 for the Dialect-Specific Zipformers to 27.21. The Dialect-Specific Zipformers are stronger on MASC-SAU, both Casablanca subsets, and SADA. Across all eight conditions, their macro-average WERs are 34.39 and 31.95, respectively.

The two Zipformer configurations are complementary rather than uniformly ordered across corpora. Multi-dialect training helps most on MASC-DZA, IRQ, SYR, and EGY, whereas dialect-specific training transfers better to Casablanca and SADA. This reversal indicates that the benefit of pooling depends on corpus and variety, and motivates reporting each condition instead of a single pooled WER. The very small MASC-DZA subset should also be interpreted cautiously.

Ground-truth LID guidance does not provide consistent additional gains on these out-of-domain sets: the eight-condition macro-average is 32.63, compared with 31.95 under predicted-LID decoding. Most degradations occur on MASC, whose dialect subsets are relatively small, particularly DZA. This pattern may be related to limited evaluation support and domainspecific calibration of the LID component, and requires further study on larger out-of-domain sets.

The Qwen comparison provides a model-family-controlled test of corpus utility. Fine-tuning with GigaSpeech 3 lowers WER relative to the Qwen3-ASR-1.7B base model in all eight conditions, with relative reductions ranging from 6.9% to 31.0%. Its macro-average falls from 47.10 to 37.78, a 19.8% relative reduction. The adapted model also outperforms Whisper-CS in all eight conditions and QwenCleo in seven, with substantially lower macro-averages than both contextual baselines (53.64 and 51.76, respectively). Together with the Zipformer results, this cross-corpus consistency makes it less likely that the primary gains arise only from matching the GigaSpeechBench evaluation domain. We do not claim state of the art on these external corpora because the table compares only six models decoded under our fixed protocol.

## D. Dialect-specific versus multi-dialect training

With automatically predicted LID, the Multi-Dialect Zipformer improves over the final Dialect-Specific Zipformer on IRQ by 1.14 WER points. The Dialect-Specific Zipformers are stronger on DZA, EGY, SYR, and SAU by 3.14, 3.48, 2.08, and 0.69 points, respectively. The multi-dialect system therefore trades some dialect-specific accuracy for unified coverage, with a five-dialect macro-average of 25.51 compared with 23.86 for the dialect-specific models. It also recognizes MSA at 10.83 WER using MGB-2 training data.

This comparison is not a controlled measurement of crossdialect transfer. The Dialect-Specific and Multi-Dialect Zipformers differ in vocabulary, sampling mixture, MSA coverage, and dialect conditioning; the dialect-specific SAU/SYR models also use NST refinement. Automatic routing introduces an additional source of errors for the multi-dialect model. The practical conclusion is that the corpus supports both dialectspecific recognizers and one automatically routed multi-dialect recognizer, while the paired LID diagnostic below measures routing cost at a fixed checkpoint.

TABLE VI  
WER (%) ON FIVE GIGASPEECHBENCH V3 DIALECTS [7]. THIRD-PARTY SYSTEMS ARE ORDERED BY DECREASING AVERAGE WER; ITALICS MARK THE BEST THIRD-PARTY VALUE, AND BOLD/UNDERLINING MARK THE BEST/SECOND-BEST OVERALL.
<table><tr><td>Model</td><td>DZA</td><td>IRQ</td><td>SAU</td><td>SYR</td><td>EGY</td><td>Avg.</td></tr><tr><td>Dolphin Base</td><td>78.26</td><td>65.20</td><td>52.35</td><td>38.12</td><td>85.31</td><td>63.85</td></tr><tr><td>Dolphin Small</td><td>72.44</td><td>62.05</td><td>50.91</td><td>30.03</td><td>74.70</td><td>58.03</td></tr><tr><td>GPT-4o Transcribe</td><td>63.14</td><td>54.53</td><td>42.38</td><td>31.67</td><td>64.23</td><td>51.19</td></tr><tr><td>FunASR-Realtime</td><td>66.30</td><td>53.44</td><td>37.67</td><td>24.24</td><td>63.33</td><td>49.00</td></tr><tr><td>Whisper Large v3</td><td>72.02</td><td>51.04</td><td>32.79</td><td>19.12</td><td>69.78</td><td>48.95</td></tr><tr><td>Deepgram Nova 3</td><td>57.90</td><td>47.54</td><td>25.02</td><td>30.61</td><td>52.77</td><td>42.77</td></tr><tr><td>NVIDIA NeMo</td><td>62.66</td><td>43.22</td><td>29.28</td><td>20.13</td><td>54.83</td><td>42.02</td></tr><tr><td>Qwen3-ASR-1.7B</td><td>63.43</td><td>41.27</td><td>25.85</td><td>18.50</td><td>59.23</td><td>41.66</td></tr><tr><td>Meta OmniASR 3B</td><td>57.68</td><td>38.80</td><td>25.31</td><td>17.86</td><td>52.37</td><td>38.40</td></tr><tr><td>ElevenLabs Scribe v2</td><td>50.43</td><td>38.67</td><td>33.33</td><td>14.73</td><td>44.44</td><td>36.32</td></tr><tr><td>Qwen3-ASR-Flash</td><td>57.18</td><td>33.21</td><td>19.21</td><td>14.41</td><td>48.78</td><td>34.56</td></tr><tr><td>Chirp 3</td><td>53.11</td><td>35.71</td><td>16.76</td><td>24.13</td><td>42.71</td><td>34.48</td></tr><tr><td>Azure</td><td>51.22</td><td>34.61</td><td>20.09</td><td>17.74</td><td>47.65</td><td>34.26</td></tr><tr><td>Gemini 3.0 Flash</td><td>44.22</td><td>36.55</td><td>20.10</td><td>14.40</td><td>41.22</td><td>31.30</td></tr><tr><td>Qwen3.5-Omni-Plus</td><td>47.11</td><td>28.54</td><td>16.56</td><td>13.76</td><td>37.12</td><td>28.62</td></tr><tr><td>Dialect-Specific (ours)</td><td>33.64</td><td>29.99</td><td>12.12</td><td>14.25</td><td>29.32</td><td>23.86</td></tr><tr><td>Multi-Dialect w/o LID (ours)</td><td>36.78</td><td>28.85</td><td>12.81</td><td>16.33</td><td>32.80</td><td>25.51</td></tr><tr><td>Multi-Dialect w/ LID (ours)</td><td>35.29</td><td>27.84</td><td>12.62</td><td>13.18</td><td>29.16</td><td>23.62</td></tr></table>

TABLE VII

CROSS-CORPUS WER (%) ON MASC, CASABLANCA, AND SADA. THE SAME FINAL CHECKPOINTS ARE USED THROUGHOUT; “AVG.” IS THE UNWEIGHTED MEAN OVER EIGHT CORPUS–VARIETY CONDITIONS. BOLD AND UNDERLINING MARK THE BEST AND SECOND-BEST CONFIGURATIONS.
<table><tr><td>Model</td><td colspan="5">MASC</td><td colspan="2">Casablanca</td><td>SADA</td><td></td></tr><tr><td></td><td>DZA</td><td>IRQ</td><td>SAU</td><td>SYR</td><td>EGY</td><td>DZA</td><td>EGY</td><td>SAU</td><td>Avg.</td></tr><tr><td>Whisper-CS</td><td>59.49</td><td>51.96</td><td>45.05</td><td>28.43</td><td>41.45</td><td>90.52</td><td>51.85</td><td>60.39</td><td>53.64</td></tr><tr><td>QwenCleo</td><td>62.26</td><td>51.64</td><td>47.07</td><td>29.48</td><td>35.74</td><td>88.12</td><td>49.08</td><td>50.67</td><td>51.76</td></tr><tr><td>Qwen3-ASR-1.7B (Base)</td><td>56.08</td><td>41.79</td><td>34.81</td><td>22.51</td><td>40.80</td><td>82.89</td><td>57.83</td><td>40.06</td><td>47.10</td></tr><tr><td>Qwen3-ASR-1.7B FT w/o LID (ours)</td><td>48.61</td><td>36.03</td><td>24.03</td><td>20.95</td><td>36.29</td><td>62.59</td><td>42.76</td><td>30.97</td><td>37.78</td></tr><tr><td>Dialect-Specific (ours)</td><td>49.57</td><td>37.01</td><td>21.23</td><td>21.33</td><td>34.04</td><td>50.06</td><td>37.34</td><td>24.54</td><td>34.39</td></tr><tr><td>Multi-Dialect w/o LID (ours)</td><td>40.17</td><td>30.46</td><td>23.08</td><td>15.44</td><td>26.88</td><td>51.37</td><td>41.29</td><td>26.87</td><td>31.95</td></tr><tr><td>Multi-Dialect w/ LID (ours)</td><td>43.59</td><td>32.72</td><td>22.81</td><td>17.53</td><td>29.47</td><td>50.71</td><td>37.76</td><td>26.48</td><td>32.63</td></tr></table>

## E. Effect of ground-truth LID guidance

The last two rows of Table VI isolate the cost of automatic routing using the Multi-Dialect Zipformer. Across the five target dialects, predicted LID raises macro-average WER from 23.62 to 25.51. EGY and SYR incur the largest increases, at 3.64 and 3.15 points, whereas SAU changes by only 0.19 points. The “w/o LID” row uses predicted LID and is the practical Multi-Dialect condition; the paired “w/ LID” row changes only the LID guidance.

The condition with ground-truth LID guidance estimates performance when routing errors are removed. Table VIII characterizes the errors under automatic routing. MSA is identified almost perfectly, but only 58–74% of the dialectal utterances are classified correctly. A large share of DZA, EGY, and SAU is routed to MSA, while SYR is additionally confused with IRQ. Together with the WER results, these errors identify routing as a practical bottleneck; LID accuracy alone does not determine the size of the recognition penalty.

Duration-dependent confusion. Table IX provides a complementary duration-stratified analysis of the Multi-Dialect Zipformer’s LID predictions. Short utterances are a difficult condition for several dialects: accuracy rises from 39.8% below 3 seconds to 85.0% at 3–10 seconds for DZA, from 51.5% to 84.6% for IRQ, and from 8.4% to 76.0% for SAU. The short-utterance confusion matrices show a strong tendency to predict MSA: 69.3% of short SAU utterances, 62.2% of short SYR utterances, and 52.4% of short DZA utterances receive that label. This pattern is consistent with limited dialectdiscriminative evidence in short speech, although the analysis does not establish whether pooling, training composition, or linguistic overlap causes these errors.

TABLE VIII  
LID CONFUSION MATRIX FOR THE MULTI-DIALECT ZIPFORMER AVERAGED OVER EPOCHS 4–8. EACH CELL GIVES THE NUMBER OF UTTERANCES FOLLOWED BY ITS ROW PERCENTAGE; BOLD DIAGONAL ENTRIES ARE CORRECT PREDICTIONS.
<table><tr><td>True \ Pred.</td><td>DZA</td><td>IRQ</td><td>SAU</td><td>SYR</td><td>EGY</td><td>MSA</td><td>Acc.</td></tr><tr><td>DZA</td><td>10707 (64%)</td><td>44 (0%)</td><td>348 (2%)</td><td>18 (0%)</td><td>557 (3%)</td><td>4985 (30%)</td><td>64.3%</td></tr><tr><td>IRQ</td><td>78 (1%)</td><td>10241 (74%)</td><td>601 (4%)</td><td>127 (1%)</td><td>581 (4%)</td><td>2249 (16%)</td><td>73.8%</td></tr><tr><td>SAU</td><td>142 (1%)</td><td>663 (6%)</td><td>6757 (58%)</td><td>119 (1%)</td><td>465 (4%)</td><td>3443 (30%)</td><td>58.3%</td></tr><tr><td>SYR</td><td>216 (3%)</td><td>909 (11%)</td><td>556 (7%)</td><td>5004 (60%)</td><td>91 (1%)</td><td>1583 (19%)</td><td>59.9%</td></tr><tr><td>EGY</td><td>27 (0%)</td><td>205 (1%)</td><td>636 (4%)</td><td>44 (0%)</td><td>10159 (63%)</td><td>4961 (31%)</td><td>63.4%</td></tr><tr><td>MSA</td><td>6 (0%)</td><td>14 (0%)</td><td>14 (0%)</td><td>7 (0%)</td><td>15 (0%)</td><td>4529 (99%)</td><td>98.8%</td></tr></table>

TABLE IX

DURATION-STRATIFIED LID ACCURACY FOR THE MULTI-DIALECT MODEL AVERAGED OVER EPOCHS 4–8. EACH CELL REPORTS ACCURACY (%) FOLLOWED BY THE NUMBER OF UTTERANCES IN PARENTHESES.
<table><tr><td>Variety</td><td> $t < 3 ~ \mathrm { s }$ </td><td> $3 \leq t < 1 0 \mathrm { ~ s ~ }$ </td><td> $t \geq 1 0 ~ \mathrm { s }$ </td></tr><tr><td>DZA</td><td>39.8 (7,808)</td><td>85.0 (8,300) 84.6 (7,620)</td><td rowspan="6">99.1 (551) 96.6 (1,276) 64.4 (1,352) 61.4 (2,928)</td></tr><tr><td>IRQ</td><td>51.5 (4,981)</td><td>76.0 (7,436)</td></tr><tr><td>SAU</td><td>8.4 (2,801)</td><td></td></tr><tr><td>SYR</td><td>24.4 (394)</td><td>61.7 (5,037)</td></tr><tr><td>EGY MSA</td><td>65.6 (7,030) 97.9 (192)</td><td>62.3 (8,252) 54.1 (750) 98.8 (4,385) 100.0 (8)</td></tr></table>

Longer speech does not uniformly resolve dialect confusion. DZA and IRQ improve further above 10 seconds, whereas SAU peaks in the middle bucket and EGY accuracy decreases from 65.6% to 54.1% across the three buckets. SYR remains near 61% in both the middle and long buckets, so utterance duration alone does not explain its aggregate weakness. These are associations across different utterances, rather than a controlled test of increasing the duration of the same speech. They also do not directly quantify ASR degradation: SAU has poor short-utterance LID but only a 0.19-point aggregate WER difference between the two Multi-Dialect rows in Table VI. Paired WER measurements with and without ground-truth LID guidance within each duration bucket would be needed to localize the recognition cost.

These results separate two claims that are easy to conflate. Ground-truth LID guidance improves average WER relative to predicted-LID decoding in the paired Zipformer diagnostic, but reliable automatic dialect identification remains unresolved. This experiment does not compare dialect conditioning against a recognizer with no dialect input. Improving LID is model- and taxonomy-dependent and is outside our scope.

## F. Iterative refinement

Table X follows each noisy-student chain from its initial weak-label-trained recognizer $( r _ { 0 } )$ to the final recognizer after four refinement rounds $( r _ { 4 } )$ , using the round definitions in Section IV-D. Under a fixed evaluation protocol, WER decreases at every round: SAU improves from 13.13 to 12.12 and SYR from 14.95 to 14.25. These are absolute reductions of 1.01 and 0.70 WER points, or 7.7% and 4.7% relative, respectively. The largest single-round gains occur at different stages: $r _ { 1 }$ to $r _ { 2 }$ for SAU and $r _ { 0 }$ to $r _ { 1 }$ for SYR.

The r4 endpoints are the final SAU and SYR models used in the main comparison, and the released weak-label inventories for these dialects are the corresponding r4-filtered data. The round-wise results distinguish gains from refinement from those of initial weak supervision: the 7.7% and 4.7% reductions are measured relative to each chain’s $r _ { 0 }$ model, rather than to Iter1. In particular, $\mathbf { S A U } \mathbf { s } \ r _ { 0 }$ WER of 13.13 is slightly higher than Iter1’s 12.98; the all-five-dialect improvement claim applies to the final models, not uniformly to the initial weak-label models. The release therefore combines initially selected data for DZA, IRQ, and EGY with four-round-refined data for SAU and SYR.

## G. Cross-architecture transfer

Table XI addresses RQ4 with a pretrained 1.7B-parameter recognizer. Decoding the adapted model without groundtruth LID guidance reduces WER on all six reported varieties by 13.9–38.3% relative to the base model, lowering macro-average WER from 38.42 to 28.28 (26.4% relative). With ground-truth LID guidance, the reductions range from 25.3% to 44.4%, and the macro-average falls further to 24.10 (37.3% relative to the base model). Ground-truth LID guidance improves every reported dialect over predicted-LID routing, with the largest absolute reductions on DZA and EGY. This consistent routing gain indicates that LID errors remain an important bottleneck for the adapted multi-dialect recognizer.

The adapted Qwen variants outperform QwenCleo and Whisper-CS on every reported dialect, although those public fine-tunes are contextual rather than data-matched controls. More significant for our argument is the direction of transfer: adaptation using weakly labeled dialect data improves Qwen on all five target dialects, complementing the final Zipformer results. The MSA improvement comes from an adaptation setting that includes manually labeled MGB-2 speech and is not evidence for weak-label-only MSA training. The dialect results reduce the likelihood that the corpus is useful only because it matches one Zipformer training recipe.

TABLE X  
NOISY-STUDENT REFINEMENT ON SAU AND SYR (WER %). r<sub>0</sub> IS THE INITIAL WEAK-LABEL MODEL AND r<sub>1</sub>–r<sub>4</sub> ARE SUCCESSIVE REFINEMENTS;RELEASED DATA CORRESPOND TO r<sub>4</sub>.
<table><tr><td>Dialect</td><td>(initial)  $r _ { 0 }$ </td><td> $r _ { 1 }$ </td><td> $r _ { 2 }$ </td><td> $r _ { 3 }$ </td><td> $r _ { 4 }$  (final)</td><td>Rel. reduction</td></tr><tr><td>SAU</td><td>13.13</td><td>12.96</td><td>12.42</td><td>12.32</td><td>12.12</td><td>7.7%</td></tr><tr><td>SYR</td><td>14.95</td><td>14.69</td><td>14.51</td><td>14.29</td><td>14.25</td><td>4.7%</td></tr></table>

TABLE XI

LARGE-MODEL COMPARISON ON GIGASPEECHBENCH (WER %). FINE-TUNED QWEN USES WEAKLY LABELED DIALECT SPEECH AND MANUALLY LABELED MGB-2 SPEECH FOR MSA; AVG. IS THE UNWEIGHTED MACRO-AVERAGE.
<table><tr><td>Model</td><td>DZA</td><td>IRQ</td><td>SAU</td><td>SYR</td><td>EGY</td><td>MSA</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>QwenCleo</td><td>71.04</td><td>55.46</td><td>32.67</td><td>28.51</td><td>44.65</td><td>24.79</td><td>42.85</td></tr><tr><td>Whisper-CS</td><td>74.86</td><td>57.86</td><td>31.02</td><td>28.24</td><td>58.91</td><td>26.56</td><td>46.24</td></tr><tr><td>Qwen3-ASR-1.7B (Base)</td><td>63.43</td><td>41.27</td><td>25.85</td><td>18.50</td><td>59.23</td><td>22.22</td><td>38.42</td></tr><tr><td>Qwen3-ASR-1.7B FT w/o LID</td><td>47.71</td><td>34.48</td><td>16.81</td><td>15.93</td><td>41.02</td><td>13.72</td><td>28.28</td></tr><tr><td>Qwen3-ASR-1.7B FT w/ LID</td><td>39.25</td><td>30.47</td><td>14.63</td><td>13.82</td><td>32.91</td><td>13.53</td><td>24.10</td></tr></table>

## VII. DISCUSSION

## A. What the experiments establish

Independently initialized recognizers trained on the generated audio–text pairs outperform Iter1 on all five dialects. DZA, IRQ, and EGY use selected weak labels, while SAU and SYR use r4-filtered data. The round-wise experiment isolates refinement gains; external evaluation and Qwen adaptation demonstrate utility beyond GigaSpeechBench and one model family. VietASR supplies labeling models, while independent training tests the corpus beyond those models’ parameters.

## B. Scale, diversity, and label noise

Weak labels exchange transcription accuracy for coverage. Selection controls noisy hypotheses, while thousands of hours of dialect-matched speech provide diversity unavailable in the 51–63-hour manually transcribed sets. Volume alone does not explain the results: SAU begins with the largest pool but retains the smallest fraction after refinement, and SYR has the largest SSL gain despite similar retention. Labeling accuracy, source match, diversity, and residual noise remain intertwined; controlled scaling and quality-stratification experiments are needed to separate their effects.

The public-system comparison also bounds the performance claim. The final Dialect-Specific Zipformers lead on DZA, EGY, and SAU, whereas Qwen3.5-Omni-Plus remains stronger than automatic routing on IRQ and SYR. Groundtruth LID guidance narrows these gaps, showing that routing explains part of the difference.

Cross-corpus transfer is not uniform: the Multi-Dialect Zipformer is stronger on four MASC subsets, whereas the Dialect-Specific models lead on both Casablanca subsets and SADA.

Pooled training and specialization therefore respond differently to domain shift. Given the six-model comparison and threeminute MASC-DZA subset, these results are evidence of generalization rather than a state-of-the-art claim.

## C. Dialect conditioning as an interface

The Multi-Dialect model predicts a dialect token followed by the transcript under the same transducer objective. Its practical results include automatic LID errors; with the same averaged checkpoint, ground-truth IDs reduce five-dialect macroaverage WER from 25.51 to 23.62. Source metadata may supply IDs in curated settings, whereas open-audio routing remains unresolved and should be evaluated separately from ASR.

Duration-stratified results reinforce the value of retaining source metadata: short dialectal speech is often routed to MSA, yet longer inputs do not remove all confusion. Country-level tags also collapse within-country varieties, speaker communities, code-switching, and orthographic choices. Aggregate LID accuracy should therefore be interpreted with durationspecific behavior, and gold resources such as CARDAMOM, Casablanca, and Hamsa remain important for finer evaluation and adaptation.

## VIII. CORPUS RELEASE PLAN AND LIMITATIONS

The GigaSpeech 3 corpus, weak transcripts, and metadata are released for research and evaluation under a research-only license. We also provide construction and training recipes and an online demo; code uses a separate software license. The corpus includes selected audio, weak transcripts, dialect labels, and segment metadata, subject to source-platform terms.

We also provide a community proofreading platform<sup>3</sup> where volunteers can listen to released segments and propose transcript corrections for future corpus revisions; the current release has not been exhaustively human-verified.

The study has four main limitations: automatic dialectrouting errors affect multi-dialect ASR; WER conflates recognition errors with legitimate spelling variation; weak-label quantity, confidence, and source diversity are not independently controlled; and web data may inherit demographic, topical, speaker, and transcription biases. Numerical expressions and code-switching also remain limited. Future extensions should add targeted data and evaluation that separates recognition from formatting errors and checks preservation of foreign-language content.

## IX. CONCLUSION

We presented GigaSpeech 3, a 39K-hour weakly labeled corpus for Arabic dialect ASR. ASR-biased SSL, selection, and NST refinement turn web speech into reusable supervision. Final Dialect-Specific Zipformers improve over their labeling models on all five dialects and reach 23.86 macro-average WER on GigaSpeechBench, 16.6% below the strongest complete third-party system. A Multi-Dialect model with predicted LID reaches 25.51 and supports MSA. Across eight external conditions, a GigaSpeech 3-trained Zipformer leads the six evaluated models, while corpus adaptation consistently improves Qwen3-ASR. These results demonstrate utility across domains and model families and identify automatic dialect routing as a remaining challenge. We thank the Hugging Face team for supporting the hosting and public release of the GigaSpeech 3 corpus.

## X. ETHICS STATEMENT

All collected audio is sourced from materials released under a Creative Commons license. Personally identifiable information has been anonymized using rule-based scripts to remove identifiable content from the data. All annotators are compensated fairly by a professional data annotation company. GigaSpeech 3 is released for non-commercial research and education purposes only, under the dataset license, with ongoing maintenance for identified risks.

## REFERENCES

[1] O. F. Zaidan and C. Callison-Burch, “Arabic dialect identification,” Computational Linguistics, vol. 40, no. 1, pp. 171–202, 2014.

[2] A. Djanibekov et al., “Dialectal coverage and generalization in arabic speech recognition,” in Proc. ACL. Association for Computational Linguistics, 2025, pp. 29 490–29 502.

[3] A. Ali et al., “The MGB-2 challenge: Arabic multi-dialect broadcast media recognition,” in Proc. IEEE SLT, 2016, pp. 279–284.

[4] A. Ali, S. Vogel, and S. Renals, “Speech recognition challenge in the wild: Arabic MGB-3,” in Proc. IEEE ASRU, 2017, pp. 316–322.

[5] A. Ali et al., “The MGB-5 challenge: Recognition and dialect identification of dialectal arabic speech,” in Proc. IEEE ASRU, 2019, pp. 1026–1033.

[6] Y. Wang, A. Alhmoud, and M. Alqurishi, “Open universal arabic ASR leaderboard,” in Proc. Interspeech, 2025, pp. 1168–1172.

[7] Y. Tu et al., “GigaSpeechBench: A Real-World Multilingual Speech-to-Text Benchmark,” arXiv preprint arXiv:2606.28884, 2026.

[8] H. Mubarak et al., “QASR: QCRI Aljazeera Speech Resource—A Large Scale Annotated Arabic Speech Corpus,” in Proc. ACL-IJCNLP. Association for Computational Linguistics, 2021, pp. 2274–2285.

[9] S. Alharbi et al., “SADA: Saudi Audio Dataset for Arabic,” in Proc. IEEE ICASSP, 2024, pp. 10 286–10 290.

[10] B. Talafha et al., “Casablanca: Data and models for multidialectal arabic speech recognition,” in Proc. EMNLP. Association for Computational Linguistics, 2024, pp. 21 745–21 758.

[11] B. Talafha et al., “CARDAMOM: A Micro-Dialectal Arabic Speech Dataset for ASR,” arXiv preprint arXiv:2609.34481, 2026.

[12] M. Alyafeai et al., “Hamsa: A manually annotated emirati arabic corpus for speech and language technologies,” in Proc. Interspeech, 2026, pp. 5563–5567.

[13] W. Chen et al., “YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech,” in Proc. Interspeech, 2026, pp. 5367–5372.

[14] A. Baevski et al., “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in Proc. NeurIPS, vol. 33, 2020, pp. 12 449–12 460.

[15] W.-N. Hsu et al., “HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451– 3460, 2021.

[16] A. Babu et al., “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in Proc. Interspeech, 2022, pp. 2278–2282.

[18] F. M. Ramirez et al., “Anatomy of industrial scale multilingual ASR,” arXiv preprint arXiv:2404.09841, 2024.

[19] Q. Xu et al., “Iterative pseudo-labeling for speech recognition,” in Proc. Interspeech, 2020, pp. 1006–1010.

[20] L. Lugosch et al., “Pseudo-labeling for massively multilingual speech recognition,” in Proc. IEEE ICASSP, 2022, pp. 7687–7691.

[21] K. S. Bhogale et al., “Empowering low-resource language ASR via large-scale pseudo labeling,” in Proc. Interspeech, 2024, pp. 2519–2523.

[22] W. Kang et al., “Libriheavy: A 50, 000 hours ASR corpus with punctuation casing and context,” in Proc. ICASSP, Seoul, 2024.

[23] Y. Yang et al., “GigaSpeech 2: An Evolving, Large-Scale and Multidomain ASR Corpus for Low-Resource Languages with Automated Crawling, Transcription and Refinement,” in Proc. ACL. Association for Computational Linguistics, 2025, pp. 2673–2686.

[24] J. Zhuo et al., “VietASR: Achieving industry-level vietnamese ASR with 50-hour labeled data and large-scale speech pretraining,” in Proc. Interspeech, Rotterdam, 2025.

[25] Z. Yu et al., “Master-ASR: Achieving multilingual scalability and lowresource adaptation in ASR with modular learning,” in Proc. ICML, ser. Proceedings of Machine Learning Research, vol. 202, 2023, pp. 40 475– 40 487.

[26] Y. Yang et al., “Towards universal speech discrete tokens: A case study for ASR and TTS,” in Proc. ICASSP, Seoul, 2024.

[27] C. Wang et al., “Supervision-guided codebooks for masked prediction in speech pre-training,” in Proc. Interspeech, 2022, pp. 2643–2647.

[28] W. Chen et al., “Reducing barriers to self-supervised learning: HuBERT pre-training with academic compute,” in Proc. Interspeech, 2023, pp. 4404–4408.

[29] H. Y. Kim et al., “ASBERT: ASR-Specific Self-Supervised Learning with Self-Training,” in Proc. IEEE SLT, 2023, pp. 9–14.

[30] F. L. Kreyssig et al., “Biased self-supervised learning for ASR,” in Proc. Interspeech, 2023, pp. 4948–4952.

[31] R. Fan et al., “CTCBERT: Advancing Hidden-Unit BERT with CTC Objectives,” in Proc. IEEE ICASSP, 2023, pp. 1–5.

[32] Z. Yao et al., “Zipformer: A faster and better encoder for automatic speech recognition,” in Proc. ICLR, Vienna, 2024.

[33] Y. Yang et al., “k2SSL: A faster and better framework for self-supervised speech representation learning,” in Proc. ICME, Nantes, 2025.

[34] G. Chen et al., “GigaSpeech: An Evolving, Multi-Domain ASR Corpus with 10,000 Hours of Transcribed Audio,” in Proc. Interspeech, 2021, pp. 3670–3674.

[35] B. Talafha, A. Waheed, and M. Abdul-Mageed, “N-shot benchmarking of whisper on diverse arabic speech recognition,” in Proc. Interspeech, 2023, pp. 5092–5096.

[36] S. A. Chowdhury et al., “Towards one model to rule all: Multilingual strategy for dialectal code-switching arabic ASR,” in Proc. Interspeech, 2021, pp. 2466–2470.

[37] S. Shon et al., “ADI17: A Fine-Grained Arabic Dialect Identification Dataset,” in Proc. IEEE ICASSP, 2020, pp. 8244–8248.

[38] M. Salhab et al., “Advancing arabic speech recognition through largescale weakly supervised learning,” arXiv preprint arXiv:2504.12254, 2025.

[39] M. Salhab et al., “Munsit at NADI 2025 shared task 2: Pushing the boundaries of multidialectal arabic ASR with weakly supervised pretraining and continual supervised fine-tuning,” in Proc. ArabicNLP Shared Tasks. Association for Computational Linguistics, 2025, pp. 734–739.

[40] M. Al-Fetyani et al., “MASC: Massive Arabic Speech Corpus,” in Proc. IEEE SLT, 2023, pp. 1006–1013.