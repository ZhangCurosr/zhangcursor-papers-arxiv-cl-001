# Learning Speech Attribute Representations via Multi-Teacher Distillation

Maryem Bouziane

LIA

Avignon Universite´

Avignon, France

0009-0002-6483-7881

Salima Mdhaffar

LIA

Avignon Universite´

0000-0002-8472-6890

Yannick Esteve\`   
LIA   
Avignon Universite´   
Avignon, France   
0000-0002-3656-8883

Abstract—Recent post-training approaches have shown that selfsupervised speech encoders can be adapted to learn a single utterancelevel speech attribute through knowledge distillation. In this work, we investigate whether a single speech encoder can simultaneously preserve multiple complementary utterance-level speech representations. We propose a multi-attribute multi-teacher distillation framework that learns semantic, speaker, language, and emotion representations from four frozen teacher models using a shared SSL backbone and attribute-specific branches. Experiments on multilingual semantic retrieval, speaker verification, language identification, and speech emotion recognition show that the proposed model achieves performance close to dedicated single-task students while requiring only a single shared encoder. Furthermore, the learned layer aggregation strategies reveal that different speech attributes naturally emerge at different depths of the SSL encoder, providing new insights into the organization of speech representations.

## I. INTRODUCTION

Speech conveys multiple complementary types of information beyond its linguistic content. In addition to the lexical message, an utterance carries information about the speaker identity, the spoken language, emotional state, speaking style, and other paralinguistic characteristics. These complementary attributes are essential for a broad range of applications including multilingual speech retrieval, speaker verification, language identification, conversational agents, speech analytics, and affective computing. Nevertheless, they are generally modeled independently, leading to multiple task-specific encoders that duplicate computation and fail to exploit the common structure underlying speech.

Recent advances in self-supervised learning (SSL) have profoundly changed speech representation learning. Foundation models such as wav2vec 2.0 [1], HuBERT [2], WavLM [3], and w2v-BERT [4] are trained on thousands of hours of unlabeled speech and produce generic contextual representations that achieve remarkable performance across many downstream tasks. While these models naturally produce frame-level representations, many applications require utterance-level embeddings that summarize global properties of an entire speech segment. Rather than redesigning SSL pre-training objectives, several recent works have demonstrated that these foundation models can be effectively adapted through post-training, transferring knowledge from pretrained expert models into a shared speech encoder.

A particularly successful direction consists in learning semantic speech embeddings through teacher-student knowledge distillation. Frameworks such as SAMU-XLSR [5], SONAR [6], [7], and SENSE [8] align speech representations with multilingual sentence embedding spaces, enabling multilingual and multimodal retrieval. More recently, multi-teacher post-training has been proposed to jointly learn semantic and speaker representations within a single SSL speech encoder [9]. These results demonstrate that multiple utterance-level attributes can coexist in a unified representation while preserving competitive performance on both semantic retrieval and speaker verification.

However, extending this paradigm beyond two attributes remains an open challenge. Semantic content, speaker identity, language identity, and emotion emerge at different abstraction levels within SSL speech encoders and rely on heterogeneous supervision signals originating from different datasets and teacher models. Simply adding new objectives may lead to conflicting optimization dynamics and representation interference. Whether a single post-trained SSL encoder can simultaneously preserve these complementary dimensions of speech therefore remains largely unexplored.

In this paper, we address this challenge by proposing a holistic multi-teacher post-training framework that learns four complementary utterance-level representations from a single shared SSL speech encoder. The proposed framework distills semantic, speaker, language, and emotion representations from four specialized frozen teachers while introducing two key mechanisms to facilitate joint optimization. First, each attribute learns its own layer aggregation strategy, allowing different branches to exploit the encoder layers that are most informative for their target representation. Second, we propose a gain-based loss weighting strategy derived from monotask learning dynamics to balance heterogeneous teachers exhibiting different convergence behaviors.

We evaluate the proposed framework on four downstream tasks covering all distilled attributes: multilingual semantic retrieval, speaker verification, language identification, and speech emotion recognition.

The contributions of this work are fourfold:

• We significantly extend recent multi-teacher post-training approaches by jointly learning four complementary utterance-level speech attributes instead of two.

• We propose a holistic post-training framework for SSL speech encoders combining attribute-specific layer aggregation with a gain-based loss weighting strategy.

• We demonstrate that a single shared encoder preserves semantic, speaker, language, and emotion representations across four independent downstream evaluation tasks.

• We provide an analysis of layer utilization showing how different speech attributes emerge at different depths of a shared SSL speech encoder.

## II. RELATED WORK

## A. Post-training of speech foundation models

Recent work has increasingly viewed post-training as a distinct stage in the lifecycle of speech foundation models, complementing large-scale self-supervised pre-training. Rather than modifying the expensive pre-training process, post-training adapts a generic speech encoder to downstream objectives through techniques including domain adaptation, instruction tuning, continual learning, and knowledge distillation [10], [11]. Existing post-training approaches can be broadly categorized into four directions: adaptation, alignment, reasoning, and efficiency [12]. Such post-training approaches have also explored efficiency through model compression and knowledge distillation [13]. In this work, rather than compressing a single expert, we consolidate the knowledge of multiple specialized teacher models into a single shared speech encoder, reducing the number of deployed encoders while preserving multiple speech attributes. The resulting model replaces multiple task-specific encoders with a single backbone capable of producing semantic, speaker, language, and emotion representations.

## B. Learning utterance-level speech representations

While self-supervised speech encoders naturally produce contextual frame-level representations, many downstream applications require compact utterance-level embeddings that summarize an entire speech segment. Early efforts mainly focused on learning a single attribute. Semantic speech representations have received particular attention through teacher-student distillation, where speech utterances are aligned with multilingual sentence embedding spaces. Representative examples include SAMU-XLSR [5], SONAR [6], [7], and SENSE [8], which produce language-agnostic utterance embeddings suitable for multilingual speech-speech and speech-text retrieval. These approaches demonstrate that semantic information can be effectively transferred from powerful text encoders to speech foundation models while preserving cross-lingual generalization.

Beyond semantics, several studies have investigated utterancelevel representations dedicated to specific paralinguistic attributes. Speaker verification systems typically rely on speaker embeddings extracted from models such as ECAPA-TDNN [14], while language identification has similarly benefited from utterance-level embeddings trained on multilingual corpora such as VoxLingua107 [15]. More recently, speech emotion representation learning has attracted increasing attention, with models such as emotion2vec [16] demonstrating that large-scale self-supervised pre-training can produce transferable emotion embeddings across a wide range of downstream emotion recognition tasks.

Although these representations target different speech properties, they share a common characteristic: each embedding space is optimized independently for a single attribute. As a consequence, practical systems typically require several specialized encoders operating in parallel, increasing memory usage, computational cost, and deployment complexity. Only recently have a few studies begun to investigate whether multiple utterance-level representations can be jointly learned from a shared speech encoder through multiteacher distillation [9]. These first results suggest that multiple speech attributes can coexist within a common SSL encoder, opening new perspectives for unified utterance-level representation learning.

## III. METHOD

Figure 1 illustrates the proposed framework. A shared SSL speech encoder processes the input utterance and produces contextual hidden representations at all encoder layers. Four attribute-specific branches, corresponding to semantic content, speaker identity, language identity and emotion, independently aggregate these hidden representations through learnable layer interpolation, followed by attention pooling and a projection layer. Each branch is supervised by a frozen teacher model through cosine embedding distillation. The entire architecture is trained end-to-end while sharing a single speech encoder across all attributes.

We propose a holistic multi-teacher distillation framework for learning several utterance-level speech representations with a single shared speech encoder. Let $\mathcal { A }$ denote the set of target attributes:

$$
\begin{array} { r } { \mathcal { A } = \{ \mathrm { s e m } , \mathrm { s p k } , \mathrm { l a n g } , \mathrm { e m o } \} , } \end{array}
$$

corresponding respectively to semantic content, speaker identity, language identity, and emotion. Each attribute $a \in { \mathcal { A } }$ is associated with a frozen teacher model $T _ { a } .$ , which defines a target embedding space for that attribute. The student model consists of a shared speech foundation encoder followed by attribute-specific branches. Each branch produces one utterance-level embedding and is trained to match the corresponding teacher representation. Given a speech utterance $x _ { i }$ , each teacher produces a target embedding ${ \bf y } _ { i } ^ { a } = T _ { a } ( x _ { i } )$

## A. Shared encoder with attribute-specific branches

The student encoder is initialized from a pretrained speech foundation model. Given an input utterance $x _ { i } ,$ the encoder produces a sequence of hidden representations at different layers:

$$
\mathbf { H } _ { i } ^ { ( \ell ) } \in \mathbb { R } ^ { T _ { i } \times D } ,
$$

where $T _ { i }$ is the number of frames, D is the dimension of hidden representations, and $\ell \in \{ 1 , \ldots , L \}$ denotes the encoder layer. Instead of using a fixed hidden layer for all attributes, we allow each attribute-specific branch to learn which encoder layers are most useful for its corresponding teacher.

For each attribute $^ { a , }$ we attach a dedicated branch to the shared encoder. First, each hidden layer is projected into an attribute-specific feature space:

$$
\widetilde { \mathbf { H } } _ { i , a } ^ { ( \ell ) } = \mathbf { H } _ { i } ^ { ( \ell ) } \mathbf { W } _ { a } ^ { ( \ell ) \top } + \mathbf { b } _ { a } ^ { ( \ell ) } .
$$

where the projection maps the shared hidden dimension into an attribute-specific feature space.

Different speech attributes are known to emerge at different depths of SSL encoders. Lower layers mainly capture acoustic information, whereas deeper layers encode increasingly semantic information. Rather than selecting a fixed encoder layer for all attributes, we allow each branch to learn an attribute-specific interpolation over all hidden layers. To model the fact that different attributes may rely on different levels of abstraction, each branch learns a scalar score $s _ { a , \ell }$ for every encoder layer. These scores are converted into normalized interpolation weights using a softmax:

$$
\alpha _ { a , \ell } = \frac { \exp ( s _ { a , \ell } ) } { \sum _ { j = 1 } ^ { L } \exp ( s _ { a , j } ) } , \qquad \sum _ { \ell = 1 } ^ { L } \alpha _ { a , \ell } = 1 .
$$

The projected layer representations are then combined as:

$$
\widehat { \mathbf { Z } } _ { i , a } = \sum _ { \ell = 1 } ^ { L } \alpha _ { a , \ell } \widetilde { \mathbf { H } } _ { i , a } ^ { ( \ell ) } .
$$

Then a layer normalization is applied to obtain $\mathbf { Z } _ { i , a }$

The resulting frame-level sequence is converted into an utterancelevel representation using an attribute-specific single-head selfattention module [17]:

$$
\mathbf { p } _ { i , a } = \mathrm { A t t e n t i o n P o o l i n g } _ { a } ( \mathbf { Z } _ { i , a } ) .
$$

Finally, an optional projection maps the pooled representation in order to fit to the teacher embedding dimension:

$$
\mathbf { z } _ { i , a } = \mathrm { P r o j } _ { a } ( \mathbf { p } _ { i , a } ) .
$$

![](images/98752889035ec5dab29ac59d4f732ee8c9d5aca82ab3f59220e16dc56696b6c8.jpg)  
Fig. 1. Overview of the multi-teacher distillation architecture.

Both the student output $\mathbf { z } _ { i , a }$ and the teacher target $\mathbf { y } _ { i } ^ { a }$ are $\ell _ { 2 ^ { - } }$ normalized before computing the distillation loss.

## B. Multi-teacher distillation objective

For each attribute a, the student branch is trained to maximize the cosine similarity with the corresponding teacher representation. The attribute-specific loss is defined as:

$$
\mathcal { L } _ { i } ^ { a } = 1 - \cos ( \mathbf { z } _ { i , a } , \mathbf { y } _ { i } ^ { a } ) .
$$

A central difficulty in holistic multi-teacher distillation is that the different teachers have heterogeneous embedding dimensions, training dynamics, and levels of difficulty. Uniform loss weighting may therefore over-emphasize attributes that are easier to optimize or whose losses have larger magnitudes.

Mono-task learning provides an estimate of how much useful information can be transferred from each teacher into the student architecture. Attributes exhibiting larger learning gains tend to correspond to teachers that are easier for the student to approximate.

Assigning them larger weights encourages the Multi-Attribute optimization to focus on supervision signals that are better aligned with the student capacity, while avoiding over-emphasizing attributes that are intrinsically difficult or noisy.

We introduce a gain-based weighting strategy derived from monotask learning dynamics. For each attribute $^ { a , }$ we first train a monotask student using the same architecture but retaining only branch a. Let $\mathcal { L } _ { a } ^ { \mathrm { m o n o } } ( e )$ denote the normalized validation loss of this monotask model at epoch e. The learning gain for attribute a is defined as:

$$
g _ { a } = \mathcal { L } _ { a } ^ { \mathrm { m o n o } } ( 1 ) - \operatorname* { m i n } _ { e } \mathcal { L } _ { a } ^ { \mathrm { m o n o } } ( e ) .
$$

The final Multi-Attribute weight is obtained by normalizing the gains across attributes:

$$
\lambda _ { a } = \frac { g _ { a } } { \sum _ { b \in \mathcal { A } } g _ { b } } .
$$

This strategy assigns larger weights to attributes for which the student shows stronger learning progress in the corresponding monotask setting, while reducing the contribution of attributes that rapidly saturate or provide weaker training gains. Unlike dynamic weighting strategies such as uncertainty weighting or GradNorm [18], the proposed weights are estimated once from mono-task learning dynamics and remain fixed during multi-teacher training, avoiding additional optimization variables and training instability.

For a training utterance x<sub>i</sub>, the overall distillation loss is defined as

$$
\mathcal { L } _ { i } = \sum _ { a \in \mathcal { A } } \lambda _ { a } \mathcal { L } _ { i } ^ { a } ,\tag{1}
$$

Although four attributes are considered in this work, the proposed framework is generic. Adding a new utterance-level attribute only requires a frozen teacher model and an additional attribute-specific branch, without modifying the shared speech encoder or the optimization procedure.

## IV. EXPERIMENTS

In order to evaluate the relevance of our proposition, we benchmark it on different tasks related to the targeted sentence-level attributes.

## A. Teacher supervision

We target four utterance-level attributes in our multi-teacher student learning framework: semantics, speaker, language, and emotion continuous representations. For each attribute, we used a specialized teacher: the BGE-M3 text encoder (when transcription is available) or the SENSE speech encoder for semantics (BGE-M3 and SENSE share the same semantic continuous space), an ECAPA-TDNN speaker embeddings trained on VoxCeleb for speaker attribute, an ECAPA-TDNN language embedding model trained on VoxLingua107, and emotion2vec-base [16] as a teacher for emotion attribute. All teacher models remain frozen throughout training.

The training data of our sentence-level multiple attribute speech encoder combines Common Voice 19 [19] and a collection of dataset annotated with emotion labels in different languages. This list of emotion datasets used in our collection is presented in Table I.

From Common Voice 19, we use the 83 languages supported by BGE-M3. Common Voice is mainly used for semantic, speaker, and language supervision, while the emotion collection provides affective speech data for emotion supervision and additional nontranscribed speech. For the emotion collection, each corpus is split independently into 80% training, 10% development, and 10% test data. After preprocessing, Common Voice provides 8,205.92 hours of speech and the emotion corpora provide 92.59 hours for training. This results a total of 8,298.51 hours of training data. The emotion collection contains 31 speech emotion recognition datasets covering multiple languages, recording conditions, and emotion taxonomies. Table I reports the language, number of emotion classes, and duration of each emotion dataset.

TABLE I  
EMOTION DATASETS USED FOR EMOTION SUPERVISION.
<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Language</td><td rowspan=1 colspan=1># Classes</td><td rowspan=1 colspan=1>Hours</td></tr><tr><td rowspan=1 colspan=1>ESD [20]</td><td rowspan=1 colspan=1>Mixed</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>23.23</td></tr><tr><td rowspan=1 colspan=1>ASVP-ESD [21]</td><td rowspan=1 colspan=1>Mixed</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>11.35</td></tr><tr><td rowspan=1 colspan=1>EmoV-DB [22]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>7.53</td></tr><tr><td rowspan=1 colspan=1>MELD [23]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.36</td></tr><tr><td rowspan=1 colspan=1>PAVOQUE [24]</td><td rowspan=1 colspan=1>German</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6.28</td></tr><tr><td rowspan=1 colspan=1>SUBESCO [25]</td><td rowspan=1 colspan=1>Bangla</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>6.26</td></tr><tr><td rowspan=1 colspan=1>Emozionalmente[26]</td><td rowspan=1 colspan=1>Italian</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>5.04</td></tr><tr><td rowspan=1 colspan=1>IEMOCAP [27]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>4.50</td></tr><tr><td rowspan=1 colspan=1>CREMA-D [28]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>4.21</td></tr><tr><td rowspan=1 colspan=1>nEMO [29]</td><td rowspan=1 colspan=1>Polish</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2.46</td></tr><tr><td rowspan=1 colspan=1>ShEMO [30]</td><td rowspan=1 colspan=1>Persian</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>2.18</td></tr><tr><td rowspan=1 colspan=1>ASED [31]</td><td rowspan=1 colspan=1>Amharic</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.69</td></tr><tr><td rowspan=1 colspan=1>EMNS [32]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1.45</td></tr><tr><td rowspan=1 colspan=1>TESS [33]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1.28</td></tr><tr><td rowspan=1 colspan=1>RAVDESS [34]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>8</td><td rowspan=1 colspan=1>1.19</td></tr><tr><td rowspan=1 colspan=1>RESD [33]</td><td rowspan=1 colspan=1>Russian</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>1.17</td></tr><tr><td rowspan=1 colspan=1>JL-Corpus [35]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>1.13</td></tr><tr><td rowspan=1 colspan=1>CaFE [36]</td><td rowspan=1 colspan=1>French</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.86</td></tr><tr><td rowspan=1 colspan=1>eNTERFACE [37]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>AESDD [38]</td><td rowspan=1 colspan=1>Greek</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>0.55</td></tr><tr><td rowspan=1 colspan=1>CASIA [33]</td><td rowspan=1 colspan=1>Mandarin</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.51</td></tr><tr><td rowspan=1 colspan=1>SAVEE [39]</td><td rowspan=1 colspan=1>English</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.41</td></tr><tr><td rowspan=1 colspan=1>EMOVO [40]</td><td rowspan=1 colspan=1>Italian</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.40</td></tr><tr><td rowspan=1 colspan=1>TurEV-DB [41]</td><td rowspan=1 colspan=1>Turkish</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.37</td></tr><tr><td rowspan=1 colspan=1>EmoDB [42]</td><td rowspan=1 colspan=1>German</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=1 colspan=1>ANAD [43]</td><td rowspan=1 colspan=1>Arabic</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.27</td></tr><tr><td rowspan=1 colspan=1>OREAU [33]</td><td rowspan=1 colspan=1>French</td><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>0.27</td></tr><tr><td rowspan=1 colspan=1>URDU [44]</td><td rowspan=1 colspan=1>Urdu</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.22</td></tr><tr><td rowspan=1 colspan=1>MESD [45]</td><td rowspan=1 colspan=1>Spanish</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>0.14</td></tr><tr><td rowspan=1 colspan=1>Polish [33]</td><td rowspan=1 colspan=1>Polish</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>0.08</td></tr><tr><td rowspan=1 colspan=1>Hi, KIA [46]</td><td rowspan=1 colspan=1>Korean</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>0.07</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>92.59</td></tr></table>

## C. Sampling strategy

In order to balance the data sources across tasks and languages, we applied a resampling strategy similar to those proposed in [5] and [8], which were used to train the SAMU-XLSR and SENSE models, respectively. This strategy computes a sampling weight for each sample. These weights are then used by the sampler to randomly select samples when constructing each training batch.

## D. Implementation and training details

The shared speech encoder is initialized from w2v-BERT 2.0. We implement the multi-teacher model in SpeechBrain, based on the open-source SENSE implementation. All teacher embeddings are precomputed before training in order to reduce the training cost. We finetune the shared w2v-BERT 2.0 encoder with Adam using a learning rate of 10<sup>−5</sup>. The task-specific modules are optimized separately with Adadelta using an initial learning rate of 1.5. Training is performed with a batch size of 14 for both training and validation. The model is trained for 17,613 iterations using 32 H100 GPUs.

## E. Tasks

To evaluate the quality of the semantic representation learned by our approach, we assess the model on multilingual speech retrieval. We evaluate the learned speaker representations on a speaker verification benchmark, and the language branch through a language identification experiment based on a classifier trained on the extracted embeddings. Last, we evaluate the emotion branch on speech emotion recognition datasets using dataset-specific classifiers.

For each attribute, we compare three types of representations: the corresponding frozen teacher model, a single-task student model trained only for the considered attribute, and the proposed multiteacher student model. This comparison allows us to quantify the impact of joint training and to evaluate whether a unified encoder can preserve attribute-specific information while sharing most of its parameters across tasks.

1) Multilingual speech retrieval: We assess the semantic branch on multilingual speech retrieval. For each experiment, we define a query set and a search set. The query set contains speech utterances, while the search set contains the corresponding translated items in another language. The objective is to retrieve the correct translation of each query from all candidates in the search set. Each query utterance is encoded into an utterance-level embedding using the semantic branch of the evaluated model. When the search set contains speech, candidate utterances are encoded with the same speech encoder. When the search set contains text, target sentences are encoded with BGE-M3. Embeddings are normalized before retrieval, and candidates are ranked according to cosine similarity. Performance is reported with Recall@1. In this work, we restrict the semantic evaluation to VoxPopuli and FLEURS. VoxPopuli is used to evaluate multilingual speech retrieval on European languages, while FLEURS is used to assess generalization to a broader multilingual setting, including unseen and low-resource languages. We compare the semantic branch of the proposed multi-teacher model with the semanticonly student and the SENSE teacher.

2) Speaker verification: We evaluate the quality of speakerrelated information using the VoxCeleb1-O speaker verification protocol [47]. Speaker embeddings are extracted from enrolment and test utterances using the speaker branch of the evaluated student models, or directly using the ECAPA-TDNN speaker teacher. For each trial pair, verification scores are computed using cosine similarity between the two utterance-level speaker embeddings. We compare the proposed multi-teacher model with the frozen ECAPA-TDNN speaker teacher and with a single-task speaker student trained using the same architecture but retaining only the speaker branch. Performance is reported in terms of Equal Error Rate (EER) and minimum normalized Detection Cost Function (minDCF), with $P _ { \mathrm { t a r g e t } } = 0 . 0 1$ and $C _ { \mathrm { F A } } = C _ { \mathrm { M i s s } } = 1$

3) Language identification: To evaluate whether the language branch captures language-discriminative information, we perform a language identification experiment on VoxLingua107. For each model, we first extract utterance-level language embeddings using the corresponding language representation: the frozen language teacher, the single-task language student, or the language branch of the multi-teacher student. A lightweight classifier is trained on top of the extracted embeddings using the VoxLingua107 training set and evaluated on its development set. The embedding extractor remains frozen during this classifier training stage. This protocol measures the quality of the learned language embedding space independently of the multi-teacher training objective. Performance is reported in terms of classification error rate. A lower error rate indicates that the extracted embeddings contain more language-discriminative information.

4) Speech emotion recognition: We evaluate the emotion branch on speech emotion recognition datasets. For each emotion dataset, we extract utterance-level emotion embeddings from the train, validation, and test partitions using three representations: the frozen emotion teacher, the single-task emotion student, and the emotion branch of the multi-teacher student. For each dataset and each representation type, we train a separate emotion classifier on the training partition of that dataset. The validation partition is used for model selection, and the final performance is reported on the test partition. This datasetspecific evaluation protocol avoids mixing incompatible emotion taxonomies across corpora and allows each classifier to match the number of emotion classes defined for the corresponding dataset. The embedding extractor remains frozen during classifier training. The classifier is therefore used only as a probing model to assess how much emotion-discriminative information is contained in the extracted representations. Performance is reported using unweighted accuracy (UA), which gives equal importance to each emotion class and is therefore suitable for evaluating emotion recognition performance under class imbalance.

## F. Experimental results

In the table of results, we use the following notation. SENSE denotes the semantic teacher, ECAPA-spk denotes the speaker teacher, ECAPA-lang denotes the language teacher, and emotion2vecbase denotes the emotion teacher. SEM-only, SPK-only, LANGonly, and EMO-only refer to single-task students trained with the same architecture as the proposed model but optimized with only one attribute-specific objective. Multi-Attribute denotes our proposed model trained jointly with semantic, speaker, language, and emotion supervision.

1) Semantic attribute: Tables II and III report Recall@1 scores for cross-lingual speech-to-speech retrieval on VoxPopuli and FLEURS corpora. Overall, the Multi-Attribute model preserves semantic retrieval performance very well and often improves over the SEMonly student or the SENSE teacher. These results indicate that Multi-Attribute supervision does not degrade the semantic space; on the contrary, the additional objectives can act as a regularizer and improve semantic retrieval robustness.

2) Speaker verification: Table IV presents the speaker verification results on VoxCeleb1-O. The ECAPA-spk teacher and the SPKonly student obtain the same EER of 0.899%. The Multi-Attribute model obtains an EER of 0.915%, corresponding to a very small absolute increase of 0.016 points. The minDCF values are also close across the three systems. These results show that joint Multi-Attribute training introduces only a marginal cost for speaker verification, while allowing the same encoder to support several other utterance-level representations.

3) Language identification: Table V reports language identification performance. The ECAPA-lang teacher obtains the lowest error rate, with 6.58%. LANG-only remains close to the teacher, reaching

6.83%. The Multi-Attribute model obtains 7.02% with its language attribute representation, corresponding to an absolute increase of only 0.19 points compared with LANG-only and 0.44 points compared with the teacher. Interestingly, by combining the language attribute with the semantic attribute generated by our model, we outperform the results reached by the ECAPA-lang teacher, showing that such a multi-attribute speech encoder provides complementary information useful to process some tasks.

4) Speech emotion recognition: We finally evaluate the emotion branch on speech emotion recognition (SER). We compare three representations: the emotion2vec-base teacher, the EMO-only student, and the emotion branch of the Multi-Attribute model. Figure 2 reports Unweighted Accuracy (UA) for each dataset. The radar plot shows that in many datasets, Multi-Attribute matches or improves over both the EMO-only student and the emotion2vec-base teacher, indicating that the emotion branch benefits from the shared encoder rather than being systematically degraded by the additional objectives.

![](images/05c09db3a1064090d15f12de1cb3afe78e4a92a4f05d6bea364958fc76a8a887.jpg)  
Fig. 2. Unweighted Accuracy on speech emotion recognition datasets. The figure compares the emotion2vec-base teacher, the EMO-only student, and the emotion branch of the proposed Multi-Attribute model.

Overall, the results across the four evaluation tasks show that the proposed multi-teacher student can learn a unified speech encoder that preserves multiple utterance-level attributes simultaneously. Compared with single-task students, the Multi-Attribute model achieves competitive performance on speaker verification, language identification, and speech emotion recognition, while improving semantic retrieval on VoxPopuli and remaining competitive on FLEURS. These findings support the central hypothesis of this work: semantic, speaker, language, and emotion representations can be distilled into a single shared speech encoder with only limited trade-offs across attributes.

## G. Layer-wise analysis

We are also interested by the changes due to the Multi-Attribute learning within the internal representations of the SSL backbone model. We analyze the learned layer-interpolation weights $\alpha _ { \tau , \ell }$ associated with each attribute-specific branch. These weights give an indication on how much each encoder layer contributes to the final utterance-level representation of a given attribute τ.

TABLE II  
RECALL@1 SCORES (%) FOR SPEECH-TO-SPEECH RETRIEVAL ON VOXPOPULI.
<table><tr><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=1>X → EN</td><td rowspan=1 colspan=1>EN → Y</td><td rowspan=1 colspan=1>X→Y</td></tr><tr><td rowspan=1 colspan=1>fr-en pl-en nl-en es-en hr-en de-en ro-en</td><td rowspan=1 colspan=1>en-fr en-pl en-nl en-es en-hr en-de en-ro</td><td rowspan=1 colspan=1>fr-de hr-cs ro-fr hu-da de-fr cs-hr fr-ro da-hu</td></tr><tr><td rowspan=1 colspan=1>SENSE</td><td rowspan=1 colspan=1>96.55 96.46 95.75 96.48 96.50 94.71 96.839</td><td rowspan=1 colspan=1>6.54 96.25 95.71 96.37 96.31 94.12 97.16|</td><td rowspan=1 colspan=1>95.20 94.75 96.55 92.69 95.36 94.69 96.90 92.63</td></tr><tr><td rowspan=1 colspan=1>SEM-only</td><td rowspan=1 colspan=1>94.05 93.83 92.62 94.62 95.17 91.21 95.90|</td><td rowspan=1 colspan=1>93.96 94.66 92.77 94.12 95.96 90.65 96.24</td><td rowspan=1 colspan=1>90.79 94.03 95.5889.7191.94 93.81 96.06 90.73</td></tr><tr><td rowspan=1 colspan=1>Multi-Attribu</td><td rowspan=1 colspan=1>te95.19 94.85 94.55 95.45 95.41 92.68 96.1</td><td rowspan=1 colspan=1>095.10 94.85 94.62 95.30 95.52 92.29 96.37</td><td rowspan=1 colspan=1>92.43 93.63 95.7588.4192.3193.63 96.0788.46</td></tr></table>

TABLE III

RECALL@1 SCORES FOR MULTILINGUAL RETRIEVAL ON FLEURS.
<table><tr><td>Model</td><td>ml-en lb-en uz-en</td><td>bs-en</td><td>my-en</td><td>ny-ces</td><td></td><td>sd-fr xh-ar</td></tr><tr><td>SENSE</td><td>62.59 60.88</td><td>55.40 60.39</td><td>14.11</td><td>27.00</td><td></td><td>59.74 36.67</td></tr><tr><td>SEM-only</td><td>63.11 60.99</td><td>56.10 59.52</td><td>19.11</td><td>29.04</td><td>58.67</td><td>37.62</td></tr><tr><td>Multi-Attribute</td><td>62.90 60.66 56.10</td><td>59.85</td><td>20.59</td><td>25.37</td><td></td><td>57.92 36.19</td></tr></table>

TABLE IV

SPEAKER VERIFICATION RESULTS ON VOXCELEB1-O.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>EER (%) ↓</td><td rowspan=1 colspan=1>minDCF↓</td></tr><tr><td rowspan=1 colspan=1>ECAPA-spk</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.001104</td></tr><tr><td rowspan=1 colspan=1>SPK-only</td><td rowspan=1 colspan=1>0.899</td><td rowspan=1 colspan=1>0.001207</td></tr><tr><td rowspan=1 colspan=1>Multi-Attribute</td><td rowspan=1 colspan=1>0.915</td><td rowspan=1 colspan=1>0.001229</td></tr></table>

Figure 3 shows the layer weights learned by the Multi-Attribute model. The semantic branch mainly relies on intermediate layers, with the largest weights around layers 12-14. This suggests that semantic information is primarily captured by contextual middle-level representations. In contrast, speaker and language information are not concentrated in a narrow set of layers. Their weights are distributed across a broader part of the encoder, indicating that speaker- and language-related cues are distributed across several levels of the representation. The emotion branch follows a different trend, with stronger contributions from early-to-middle layers. This suggests that emotion-related information relies more on acoustic and prosodic cues, which are better preserved before the final contextual layers.

Figure 4 compares the layer-weight distributions of the singletask students. The comparison shows that Multi-Attribute training changes the way information is encoded in the shared encoder. For example, the SEM-only student assigns stronger weights to the last layers, whereas the semantic branch of the Multi-Attribute model shifts toward intermediate layers. Similarly, speaker and language information remain broadly distributed, but their layer usage is reorganized when the encoder is jointly optimized with the other objectives. These observations indicate that the Multi-Attribute model does not simply replicate the single-task representations. Instead, joint training redistributes attribute-specific information across the encoder layers. This depth-wise reorganization helps the shared encoder accommodate semantic, speaker, language, and emotion information within a single model, while preserving competitive performance across the downstream tasks.

TABLE V  
LANGUAGE IDENTIFICATION RESULTS ON VOXLINGUA107.
<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Classification error (%) ↓</td></tr><tr><td rowspan=1 colspan=1>ECAPA-lang teacher</td><td rowspan=1 colspan=1>6.58</td></tr><tr><td rowspan=1 colspan=1>LANG-only</td><td rowspan=1 colspan=1>6.83</td></tr><tr><td rowspan=1 colspan=1>Multi-Attribute Lang</td><td rowspan=1 colspan=1>7.02</td></tr><tr><td rowspan=1 colspan=1>Multi-Attribute Sem + Lang</td><td rowspan=1 colspan=1>6.39</td></tr></table>

![](images/4da9aacb02230d28804f6e49f194bf2a6497cd6741cfdf2b7263373245390e80.jpg)  
Fig. 3. Layer-interpolation weights learned by the four branches of the Multi-Attribute model. Semantic information is mainly extracted from intermediate layers, while speaker and language information are distributed across several encoder layers.

![](images/a02bd847fbff5b1b6182c46766b679d551734030160e3156c2c98135548f1baa.jpg)  
(a) SEM-only

![](images/3c63abffab03795198719cc0bba49ed932d193e7bd78cfacff2235b1748733c8.jpg)

![](images/e77e46f03ad8ccaddc49987928913dbcb713a801b02691f59c4770f9137d07b2.jpg)  
(c) LANG-only

(b) SPK-only  
![](images/8d8112b333aef19253c19a095e774e15351c44fb67f07de4c02b0f51d89b2a56.jpg)  
(d) EMO-only  
Fig. 4. Layer-interpolation weights learned by the single-task students. Compared with the Multi-Attribute model, these distributions show that joint training modifies the layer usage of each attribute branch.

## V. CONCLUSION

We presented a holistic multi-teacher post-training framework that enables a single self-supervised speech encoder to learn multiple complementary utterance-level speech representations through knowledge distillation. By jointly distilling semantic, speaker, language, and emotion representations from four specialized teacher models, the proposed approach replaces several task-specific encoders with a single shared backbone while preserving competitive performance across four independent downstream evaluation tasks.

Despite the heterogeneous nature of the supervision signals and datasets, the proposed model consistently approaches the performance of dedicated single-task students, while even improving semantic retrieval and showing that semantic and language representations can provide complementary information for language identification. Beyond the downstream results, the learned layer interpolation weights reveal that different speech attributes naturally emerge at different depths of the shared SSL encoder, providing new insights into the internal organization of speech foundation models.

More broadly, this work suggests that post-training can transform generic SSL speech encoders into holistic representation models capable of supporting multiple speech understanding tasks without requiring dedicated backbones. We believe that this direction opens a promising path toward universal speech foundation models producing rich, reusable utterance-level representations for a wide range of downstream applications.

## VI. ACKNOWLEDGMENTS

## REFERENCES

[1] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “Wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in Neural Information Processing Systems (NeurIPS), 2020.

[2] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “HuBERT: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021.

[3] S.-w. Chen, Y. Cheng, Y. Wang, Z. Huang, W. Chen, Z. Zhang, B. Yao, Y. Zhao, J. Wu, and X. Chen, “WavLM: Large-scale self-supervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, 2022.

[4] Y.-A. Chung, Y. Zhang, W. Han, C.-C. Chiu, J. Qin, R. Pang, and Y. Wu, “W2v-bert: Combining contrastive learning and masked language modeling for self-supervised speech pre-training,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021, pp. 244–250.

[5] S. Khurana, A. Laurent, and J. Glass, “SAMU-XLSR: Semanticallyaligned multimodal utterance-level cross-lingual speech representation,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1493–1504, 2022.

[6] P.-A. Duquenne, H. Schwenk, and B. Sagot, “SONAR: sentence-level multimodal and language-agnostic representations,” 2023. [Online]. Available: https://arxiv.org/abs/2308.11466

[7] “Joint speech and text machine translation for up to 100 languages,” Nature, vol. 637, no. 8046, pp. 587–593, 2025.

[8] S. Mdhaffar, H. Elleuch, C. Chellaf, H. Nguyen, and Y. Esteve, “SENSE\` models: an open source solution for multilingual and multimodal semantic-based tasks,” arXiv preprint arXiv:2509.12093, 2025.

[9] M. Bouziane, S. Mdhaffar, and Y. Esteve, “Learning multiple utterance-\` level attribute representations with a unified speech encoder,” arXiv preprint arXiv:2603.08312, 2026.

[10] B. Li, D. Hwang, Z. Huo, J. Bai, G. Prakash, T. N. Sainath, K. C. Sim, Y. Zhang, W. Han, T. Strohman et al., “Efficient domain adaptation for speech foundation models,” in ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[11] W. Ge, X. Wang, X. Liu, and J. Yamagishi, “Post-training for deepfake speech detection,” in 2025 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2025, pp. 1–8.

[12] “Post-training of speech foundation models,” Interspeech 2026 Special Session, 2026, https://sites.google.com/view/ptsfm.

[13] G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv preprint arXiv:1503.02531, 2015.

[14] B. Desplanques, J. Thienpondt, and K. Demuynck, “ECAPA-TDNN: Emphasized channel attention, propagation and aggregation in TDNN based speaker verification,” Interspeech 2020, 2020.

[15] J. Valk and T. Alumae, “Voxlingua107: a dataset for spoken language¨ recognition,” in 2021 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2021, pp. 652–658.

[16] Z. Ma, Z. Zheng, J. Ye, J. Li, Z. Gao, S. Zhang, and X. Chen, “emotion2vec: Self-supervised pre-training for speech emotion representation,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 15 747–15 760.

[17] P. Safari, M. India, and J. Hernando, “Self-attention encoding and pooling for speaker recognition,” in Proc. Interspeech 2020, 2020, pp. 941–945.

[18] Z. Chen, V. Badrinarayanan, C.-Y. Lee, and A. Rabinovich, “Gradnorm: Gradient normalization for adaptive loss balancing in deep multitask networks,” in International conference on machine learning. PMLR, 2018, pp. 794–803.

[19] R. Ardila, M. Branson, K. Davis, M. Kohler, J. Meyer, M. Henretty, R. Morais, L. Saunders, F. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” in Proceedings of the twelfth language resources and evaluation conference, 2020, pp. 4218–4222.

[20] K. Zhou, B. Sisman, R. Liu, and H. Li, “Seen and unseen emotional style transfer for voice conversion with a new emotional speech dataset,” in ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021, pp. 920–924.

[21] T. T. L. Dejoli, Q. He, and W. Xie, “Audio, speech and vision processing lab emotional sound database (asvp-esd),” 2021.

[22] A. Adigwe, N. Tits, K. E. Haddad, S. Ostadabbas, and T. Dutoit, “The emotional voices database: Towards controlling the emotion dimension in voice generation systems,” arXiv preprint arXiv:1806.09514, 2018.

[23] S. Poria, D. Hazarika, N. Majumder, G. Naik, E. Cambria, and R. Mihalcea, “Meld: A multimodal multi-party dataset for emotion recognition in conversations,” in Proceedings of the 57th annual meeting of the association for computational linguistics, 2019, pp. 527–536.

[24] I. Steiner, M. Schroder, and A. Klepp, “The pavoque corpus as a resource¨ for analysis and synthesis of expressive speech,” Proc. Phonetik & Phonologie, vol. 9, pp. 83–84, 2013.

[25] S. Sultana, M. S. Rahman, M. R. Selim, and M. Z. Iqbal, “Sust bangla emotional speech corpus (subesco): An audio-only emotional speech corpus for bangla,” Plos one, vol. 16, no. 4, p. e0250173, 2021.

[26] F. Catania, J. W. Wilke, and F. Garzotto, “Emozionalmente: A crowdsourced corpus of simulated emotional speech in italian,” IEEE Transactions on Audio, Speech and Language Processing, 2025.

[27] C. Busso, M. Bulut, C.-C. Lee, A. Kazemzadeh, E. Mower, S. Kim, J. N. Chang, S. Lee, and S. S. Narayanan, “Iemocap: Interactive emotional dyadic motion capture database,” Language resources and evaluation, vol. 42, no. 4, pp. 335–359, 2008.

[28] H. Cao, D. G. Cooper, M. K. Keutmann, R. C. Gur, A. Nenkova, and R. Verma, “Crema-d: Crowd-sourced emotional multimodal actors dataset,” IEEE transactions on affective computing, vol. 5, no. 4, pp. 377–390, 2014.

[29] I. Christop, “nemo: Dataset of emotional speech in polish,” in Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), 2024, pp. 12 111–12 116.

[30] O. M. Nezami, P. J. Lou, and M. Karami, “Shemo: a large-scale validated database for persian speech emotion detection,” Language Resources and Evaluation, pp. 1–16, 2019.

[31] E. A. Retta, E. Almekhlafi, R. Sutcliffe, M. Mhamed, H. Ali, and J. Feng, “A new amharic speech emotion dataset and classification benchmark,” ACM Transactions on Asian and Low-Resource Language Information Processing, vol. 22, no. 1, pp. 1–22, 2023.

[32] K. A. Noriy, X. Yang, and J. J. Zhang, “Emns /imz/ corpus: An emotive single-speaker dataset for narrative storytelling in games, television and graphic novels,” 2023. [Online]. Available: https://arxiv.org/abs/2305.13137

[33] Z. Ma, M. Chen, H. Zhang, Z. Zheng, W. Chen, X. Li, J. Ye, X. Chen, and T. Hain, “Emobox: Multilingual multi-corpus speech emotion recognition toolkit and benchmark,” arXiv preprint arXiv:2406.07162, 2024.

[34] S. R. Livingstone and F. A. Russo, “The ryerson audio-visual database of emotional speech and song (ravdess): A dynamic, multimodal set of facial and vocal expressions in north american english,” PloS one, vol. 13, no. 5, p. e0196391, 2018.

[35] J. James, L. Tian, and C. I. Watson, “An open source emotional speech corpus for human robot interaction applications.” in Interspeech, 2018, pp. 2768–2772.

[36] P. Gournay, O. Lahaie, and R. Lefebvre, “A canadian french emotional speech dataset,” in Proceedings of the 9th ACM multimedia systems conference, 2018, pp. 399–402.

[37] O. Martin, I. Kotsia, B. Macq, and I. Pitas, “The enterface’05 audiovisual emotion database,” in 22nd international conference on data engineering workshops (ICDEW’06). IEEE, 2006, pp. 8–8.

[38] N. Vryzas, R. Kotsakis, A. Liatsou, C. A. Dimoulas, and G. Kalliris, “Speech emotion recognition for performance interaction,” Journal of the Audio Engineering Society, vol. 66, no. 6, pp. 457–467, 2018.

[39] P. Jackson and S. Haq, “Surrey audio-visual expressed emotion (savee) database,” University of Surrey: Guildford, UK, 2014.

[40] G. Costantini, I. Iaderola, A. Paoloni, M. Todisco et al., “Emovo corpus: an italian emotional speech database,” in Proceedings of the ninth international conference on language resources and evaluation

(LREC’14). European Language Resources Association (ELRA), 2014, pp. 3501–3504.

[41] S. F. Canpolat, Z. Ormanoglu, and D. Zeyrek, “Turkish emotion voice˘ database (turev-db),” in Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-Resourced Languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL), 2020, pp. 368–375.

[42] F. Burkhardt, “A database of german emotional speech,” 2000.

[43] A. A. Alnuaim, M. Zakariah, A. Alhadlaq, C. Shashidhar, W. A. Hatamleh, H. Tarazi, P. K. Shukla, and R. Ratna, “Human-computer interaction with detection of speaker emotions using convolution neural networks,” Computational Intelligence and Neuroscience, vol. 2022, no. 1, p. 7463091, 2022.

[44] S. Latif, A. Qayyum, M. Usman, and J. Qadir, “Cross lingual speech emotion recognition: Urdu vs. western languages,” in 2018 International conference on frontiers of information technology (FIT). IEEE, 2018, pp. 88–93.

[45] M. M. Duville, L. M. Alonso-Valerdi, and D. I. Ibarra-Zarate, “The mexican emotional speech database (mesd): elaboration and assessment based on machine learning,” in 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021, pp. 1644–1647.

[46] T. Kim, S. Doh, G. Lee, H. Jeon, J. Nam, and H.-J. Suk, “Hi, kia: A speech emotion recognition dataset for wake-up words,” in 2022 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC). IEEE, 2022, pp. 1590–1595.

[47] A. Nagrani, J. S. Chung, W. Xie, and A. Zisserman, “Voxceleb: Largescale speaker verification in the wild,” Computer Speech & Language, vol. 60, p. 101027, 2020.