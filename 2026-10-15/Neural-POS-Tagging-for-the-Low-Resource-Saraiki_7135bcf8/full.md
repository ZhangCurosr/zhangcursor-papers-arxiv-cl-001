# Neural POS Tagging for the Low-Resource Saraiki

Usman Nawaz<sup>1</sup>, Muhammad Junaid Iqbal<sup>2</sup>, Muhammad Asaf<sup>3</sup>, Mariam Munsif Mir<sup>4</sup>, Faisal Rehman<sup>5</sup>, Tahir Alyas<sup>6</sup>, and Muhammad Hasnain<sup>7</sup>

<sup>1</sup> Independent Researcher, Palermo, Italy

<sup>2</sup> University of Rome Tor Vergata, Rome, Italy Muhammad.junaid.iqbal@uniroma2.it

<sup>3</sup> DIMES, University of Calabria, Rende, Italy

<sup>4</sup> School of ICT, Grifith University, Nathan, QLD 4111, Australia 5 University of Mianwali, Mianwali, Pakistan

6 Department of Computer Science, Lahore Garrison University, Lahore, Pakistan 7 School of Computer Science, Minhaj University, Lahore, Pakistan

Abstract. Part-of-speech tagging is a fundamental sequence labeling task and an important component for many downstream natural language processing applications. However, Saraiki, written from right to left in a Perso-Arabic script, remains an under-resourced Indo-Aryan language, despite being spoken as a mother tongue by 28,849,579 people (approximately 28.9 million) according to Pakistan’s 7th Population and Housing Census 2023. This paper presents a manually annotated Saraiki Universal POS tagging dataset developed from plain-text. We annotated the text with 17 POS categories. We evaluate three pretrained transformer encoders under the same fine-tuning setup: XLM-RoBERTa base, Urdu RoBERTa small, and multilingual DistilBERT. Experimental results over five random seeds show that XLM-RoBERTa base achieves the best performance, with 87.29% ± 0.21% development accuracy and 88.60% ± 0.21% test accuracy. Urdu RoBERTa small and multilingual DistilBERT also provide competitive results, suggesting that multilingual and related-script pretrained encoders are efective for Saraiki UPOS tagging. The dataset, evaluation setup, and baseline results introduced in this work provide a benchmark for future research on Saraiki NLP and low-resource sequence labeling.

Keywords: Saraiki · POS Tagging · Universal Dependencies · Low-Resource NLP · Transformer Models

## 1 Introduction

Part-of-speech tagging is a fundamental sequence labeling task that assigns grammatical categories such as noun, adjective, verb, and pronoun to tokens in a sentence. POS tags are useful for many downstream NLP tasks, including machine translation, syntactic parsing, named entity recognition, information extraction and linguistic corpus development [34]. For low-resource languages,

POS-tagged corpora are especially important because they provide a basic layer of linguistic annotation for building and evaluating NLP tools [11].

Saraiki, also spelled Siraiki, is an Indo-Aryan language spoken mainly in Pakistan, particularly in Southern Punjab and neighboring regions [2]. According to Pakistan’s 7th Population and Housing Census 2023, Saraiki is reported as the mother tongue of 28,849,579 people, approximately 28.9 million, representing 12.00% of Pakistan’s population [22]. It is written from right to left in a Perso-Arabic script and is closely related to Punjabi and Sindhi. Despite this sizeable speaker community and its rich literary and oral tradition, Saraiki remains under-resourced in computational linguistics. Compared with high-resource languages, Saraiki has limited annotated benchmark datasets and fewer publicly available NLP tools. This lack of resources makes it challenging to develop robust systems for automatic language processing [11].

Recent work has begun to address this gap. Alam et al. introduced the first Universal Dependencies treebank for Saraiki, containing 587 sentences and 7,597 tokens annotated using the Universal Dependencies framework [2]. Their work describes important syntactic and morphological properties of Saraiki, including split ergativity, light verb constructions, serial verb constructions, compounds, and relative clauses. We could not locate a public release of their annotated treebank. Other studies have proposed Saraiki-specific POS tagsets and linguistic annotation schemes. However, publicly available Saraiki UPOS resources and reproducible benchmarks for automatic POS tagging remain limited and there is still a need for additional manually annotated data and reproducible benchmark evaluations for automatic POS tagging.

Saraiki also presents challenges for NLP because of its linguistic and dialectal diversity. Recent work has shown that Saraiki dialects spoken in regions such as Multan and other areas exhibit noticeable lexical variation [30]. Such variation can afect vocabulary coverage, tokenization, and model generalization [11]. In addition, Saraiki shares script-level similarities with Urdu and Punjabi Shahmukhi, but it has its own grammatical and lexical properties [27, 1, 32]. These factors make it important to evaluate NLP models directly on Saraiki data rather than assuming that resources or models developed for related languages will transfer without loss [26].

Universal POS tagging provides a useful framework for developing such resources. The Universal POS tagset ofers a cross-linguistically consistent set of coarse-grained POS categories [18]. It is widely used in the Universal Dependencies project, which provides a common framework for morphological and syntactic annotation across languages [8, 21]. Using UPOS annotation for Saraiki makes the resource compatible with multilingual NLP research and with existing UD-style resources for related Indo-Aryan languages.

In this paper, we address the lack of benchmark resources for Saraiki POS tagging by introducing a manually annotated Universal POS tagging dataset. The corpus is developed from plain-text Saraiki data and annotated with the Universal POS tag inventory [8, 21]. It contains 17 UPOS categories and is divided into fixed train, development, and test splits to support reproducible evaluation.

We further establish transformer-based baseline results for Saraiki UPOS tagging. Specifically, we compare multilingual and related-script pretrained encoders under a consistent experimental setting, including XLM-RoBERTa base, Urdu RoBERTa small, and multilingual DistilBERT. Our results show that XLM-RoBERTa base performs best, while Urdu RoBERTa small and multilingual DistilBERT provide competitive baselines. These findings suggest that multilingual pretraining and related-script representations are useful for Saraiki sequence labeling.

We summarize the contributions of this paper as follows:

– We introduce a manually annotated Saraiki UPOS tagging dataset developed from plain-text Saraiki data, covering 17 Universal POS categories and released splits.

– We establish baseline results for Saraiki UPOS tagging by evaluating multilingual and related-script pretrained encoders under a consistent experimental setting.

## 2 Related Work

POS tagging has been widely studied as a core sequence labeling task in NLP [17, 33, 25]. Early POS tagging systems relied on rule-based methods, statistical models, and feature-based machine learning approaches [6, 5, 33]. More recent work has shifted toward neural and pretrained transformer-based models, which learn contextualized representations and can be fine-tuned for token-level classification tasks [12, 14, 9, 19]. For low-resource languages, however, the lack of annotated data remains a major limitation [11]. In such settings, manually annotated corpora and reproducible baseline systems are important for both linguistic research and NLP tool development.

Universal POS tagging provides a common framework for cross-linguistic POS annotation. The Universal POS tagset was introduced to support consistent annotation across languages using a small set of coarse-grained grammatical categories. It is now widely used in the Universal Dependencies framework, which provides annotation guidelines and treebanks for many languages [8, 21]. UDstyle annotation has supported multilingual research in parsing, morphology, and sequence labeling. However, coverage remains uneven, and many low-resource languages still lack suficiently large annotated corpora [11, 20].

Several Universal Dependencies resources have been developed for Indo-Aryan languages. Existing UD treebanks include resources for Hindi, Urdu, Marathi, and Punjabi [31, 4, 10, 24, 3]. These resources show the usefulness of UD annotation for South Asian languages, but they also highlight the uneven availability of linguistic resources across the Indo-Aryan family. Major languages such as Hindi and Urdu have received more attention than smaller or regionally concentrated languages. As a result, many Indo-Aryan languages remain under-represented in computational resources and benchmark evaluations.

Earlier Saraiki NLP work has also addressed POS tagset design and linguistic annotation proposed a POS tagset for morphological, syntactic, and lexical annotation of Saraiki [28]. Malik et al. [16] presented a hierarchical POS tagset for Saraiki with a richer set of language-specific categories. These studies are valuable because they document Saraiki-specific grammatical categories and provide foundations for linguistic annotation. However, language-specific tagsets may not be directly comparable across languages. In contrast, UPOS annotation supports multilingual comparison and compatibility with Universal Dependencies-style resources [8].

Saraiki also presents challenges because of dialectal and lexical variation spoken in diferent regions of Pakistan [30]. Their findings show that lexical choices vary across regional boundaries. Such variation is important for NLP because it can afect vocabulary coverage, tokenization, and model generalization. For POS tagging, dialectal variation may introduce diferent word forms or region-specific lexical items that are not equally represented in training data.

Transformer-based models have become the dominant approach for many sequence labeling tasks [35, 9]. XLM-RoBERTa is a multilingual transformer trained on an extensive amount of data and has achieved strong results on many cross-lingual tasks [7]. DistilBERT provides a smaller and faster distilled transformer model while retaining much of the performance of the original transformer architecture [29]. For Saraiki, related-script models are also relevant because Saraiki is written in Perso-Arabic script and shares script-level similarities with Urdu and Punjabi Shahmukhi [27, 1, 32]. Therefore, Urdu-focused pretrained models may provide useful representations for Saraiki text.

Overall, previous work has provided important foundations for Saraiki linguistic annotation, including POS tagsets and a small Universal Dependencies treebank. However, benchmark-style evaluation of pretrained transformer encoders for Saraiki UPOS tagging remains limited. This paper addresses this gap by manually annotating Saraiki plain text with Universal POS tags and evaluating three pretrained transformer models under a fixed train–development–test setup.

## 3 Dataset Construction

## 3.1 Data Source

The source data contains Saraiki text written in the Perso-Arabic script [13]. Since the original dataset was plain text, it did not provide token boundaries, POS labels, morphological features, or dependency relations. We therefore manually processed and annotated the data for Universal POS tagging. The goal was to create a resource suitable for training and evaluating automatic Saraiki POS taggers.

Saraiki is written from right to left and shares script-level properties with Urdu and Punjabi Shahmukhi. However, it also has its own lexical, morphological, and syntactic characteristics. For this reason, we kept the text in its original

Saraiki script and did not transliterate it into Roman script or any other writing system. This allows the resulting dataset to reflect the form in which Saraiki is commonly written and processed.

## 3.2 Preprocessing

Before annotation, we cleaned the raw text by removing empty lines, unusable rows, and duplicate or malformed entries where necessary. We also normalized whitespace while preserving the original script. Since the aim of this work is to evaluate Saraiki POS tagging in its native writing system, no translation or transliteration was applied.

The cleaned sentences were then manually segmented into tokens. Each sentence was converted into CoNLL-U style format, where each token appears on a separate line with its corresponding annotation fields [21]. Since the present work focuses on Universal POS tagging, we annotate the UPOS column and leave the XPOS, morphological-feature, and dependency fields unspecified [8].

## 3.3 UPOS Label Inventory

We annotate the dataset using the Universal POS tag inventory [21]. The UPOS tagset provides a small set of coarse-grained grammatical categories that can be applied consistently across languages. This makes the dataset compatible with Universal Dependencies-style annotation and allows comparison with POS tagging resources for other languages.

The dataset contains 17 UPOS categories:

<table><tr><td>ADJ ADP</td><td>ADV AUX</td></tr><tr><td>CCONJ DET</td><td>INTJ NOUN</td></tr><tr><td>NUM PART</td><td>PRON PROPN</td></tr><tr><td>PUNCT SCONJ</td><td>SYM VERB</td></tr><tr><td>X</td><td></td></tr></table>

Each token is assigned exactly one UPOS label. Tokens that function as common nouns are annotated as NOUN, while proper names are annotated as PROPN. Main lexical verbs are annotated as VERB, and auxiliary elements are annotated as AUX. Adpositions and postpositions are annotated as ADP. Punctuation marks are annotated as PUNCT. The label X is used for tokens that cannot be assigned to one of the regular grammatical categories.

## 3.4 Manual Annotation

The annotation was carried out manually using the Universal POS tag inventory. One annotator, who holds an MPhil in Linguistics and has experience in linguistic annotation, assigned a UPOS label to each token according to its grammatical function in context. Two PhD students independently reviewed the entire annotated dataset for correctness and made manual corrections where needed.

Dificult cases were considered in sentence context, since the same surface form may belong to diferent grammatical categories depending on usage.

The annotation process focused on producing a consistent token-level POS resource. Since there is no widely adopted large-scale Saraiki POS benchmark, we followed the Universal POS categories rather than a language-specific tagset. This choice supports cross-linguistic comparability and makes the dataset easier to use with existing multilingual NLP models and Universal Dependenciesoriented tools.

Some annotation decisions are especially relevant for Saraiki. Verbal elements were annotated according to their function as main verbs or auxiliaries. Nouns and proper nouns were distinguished based on whether the token referred to a general entity or a named entity. Pronouns, determiners, and relative forms were annotated according to their syntactic function in the sentence. These decisions are important because Saraiki shares several grammatical patterns with related Indo-Aryan languages, but also has language-specific variation.

## 3.5 Data Splits

After annotation, the dataset was divided into fixed train, development, and test splits. The training split is used for model fine-tuning, the development split is used for best checkpoint selection, and the test split is reserved for final evaluation. Keeping fixed splits allows future work to compare results under the same evaluation setting. Table 1 shows the distribution of sentences and tokens across the dataset splits.

Table 1. Dataset statistics for the training, development, and test splits.
<table><tr><td colspan="2">Dataset Split Sentences Tokens</td></tr><tr><td>Training</td><td>1,666 28,377</td></tr><tr><td>Development</td><td>208 3,636</td></tr><tr><td>Test</td><td>209 3,439</td></tr><tr><td>Total</td><td>2,083 35,452</td></tr></table>

## 4 Methodology

We formulate Saraiki Universal POS tagging as a token-level sequence labeling task. Given an input sentence

$$
S = ( z _ { 1 } , z _ { 2 } , \ldots , z _ { n } ) ,\tag{1}
$$

where $z _ { i }$ indicates the i-th token in the sequence, the goal is to assign a UPOS label to each token $z _ { i }$

$$
y _ { i } \in \mathcal { V }\tag{2}
$$

Table 2. Number of tokens per UPOS tag in each data split.
<table><tr><td>Tag</td><td>Train</td><td>Dev</td><td>Test</td><td>Total</td></tr><tr><td>ADJ</td><td>1,430</td><td>205</td><td>153</td><td>1,788</td></tr><tr><td>ADP</td><td>4,235</td><td>544</td><td>526</td><td>5,305</td></tr><tr><td>ADV</td><td>267</td><td>44</td><td>32</td><td>343</td></tr><tr><td>AUX</td><td>1,746</td><td>208</td><td>204</td><td>2,158</td></tr><tr><td>CCONJ</td><td>856</td><td>127</td><td>129</td><td>1,112</td></tr><tr><td>DET</td><td>73</td><td>12</td><td>8</td><td>93</td></tr><tr><td>INTJ</td><td>1</td><td>0</td><td>0</td><td>1</td></tr><tr><td>NOUN</td><td>8,915</td><td>1,241</td><td>1,050</td><td>11,206</td></tr><tr><td>NUM</td><td>487</td><td>69</td><td>44</td><td>600</td></tr><tr><td>PART</td><td>511</td><td>78</td><td>65</td><td>654</td></tr><tr><td>PRON</td><td>1,348</td><td>189</td><td>174</td><td>1,711</td></tr><tr><td>PROPN</td><td>3,651</td><td>312</td><td>495</td><td>4,458</td></tr><tr><td>PUNCT</td><td>2,479</td><td>284</td><td>258</td><td>3,021</td></tr><tr><td>SCONJ</td><td>602</td><td>68</td><td>73</td><td>743</td></tr><tr><td>SYM</td><td>16</td><td>0</td><td>0</td><td>16</td></tr><tr><td>VERB</td><td>1,749</td><td>255</td><td>228</td><td>2,232</td></tr><tr><td>X</td><td>11</td><td>0</td><td>0</td><td>11</td></tr><tr><td>Total</td><td>28,377</td><td>3,636</td><td>3,439</td><td>35,452</td></tr></table>

Here, $\mathcal { V }$ is the set of 17 Universal POS categories used in the annotated corpus.   
Each token receives exactly one UPOS label.

## 4.1 Model Architecture

We use pretrained transformer encoders for token classification [35, 9]. Given a tokenized input sentence, the encoder produces contextualized representations for the input tokens. These representations are passed through a dropout layer and a linear classification head to produce logits over the UPOS label set.

For an input token representation $\mathbf { h } _ { i } .$ , the model computes:

$$
\mathbf { o } _ { i } = \mathbf { W } \mathbf { h } _ { i } + \mathbf { b } ,\tag{3}
$$

where $\mathbf { o } _ { i } \in \mathbb { R } ^ { | \mathcal { V } | }$ is the output logit vector for token i, W and b are the parameters of the classification head, and $| \mathcal { V } | = 1 7$ . The predicted UPOS label is obtained by selecting the label with the highest logit:

$$
\hat { y } _ { i } = \arg \operatorname* { m a x } _ { k \in \mathcal { V } } o _ { i , k } .\tag{4}
$$

The same architecture is used for all pretrained encoders: a transformer encoder followed by dropout and a linear UPOS classification head. This ensures that the comparison focuses on the pretrained encoder rather than diferences in the downstream classifier. All encoder parameters and classification head parameters are fine-tuned during training.

We train all models using standard cross-entropy loss computed over the labeled token positions. Let I be the set of token positions included in loss computation. The loss is computed as:

$$
\mathcal { L } _ { \mathrm { C E } } = - \frac { 1 } { \vert \mathcal { T } \vert } \sum _ { i \in \mathcal { I } } \log \frac { \exp ( o _ { i , y _ { i } } ) } { \sum _ { k = 1 } ^ { \vert \mathcal { V } \vert } \exp ( o _ { i , k } ) } .\tag{5}
$$

## 5 Experimental Setup

We evaluate Saraiki UPOS tagging using the fixed splits described in Section 3 and repeat each experiment using five random seeds (42–46) and report the mean and sample standard deviation of accuracy.

## 5.1 Models

We compare three pretrained transformer encoders under the same token classification setup.<sup>8</sup> The selected models include multilingual and related-script encoders. XLM-RoBERTa base is included as a strong multilingual model trained on large-scale multilingual data [7]. Urdu RoBERTa small is included because Saraiki and Urdu are both written in Perso-Arabic script and may share useful script-level representations [27, 1]. Multilingual DistilBERT is included as a faster and lighter multilingual baseline [29].

All models are fine-tuned using AdamW optimization and a linear learning rate schedule with warmup [15]. We use a batch size of 16, a maximum sequence length of 256, and train for up to 10 epochs. Early stopping is applied when development accuracy does not improve for four consecutive epochs. The dropout probability is set to 0.10.

Table 3. Hyperparameter settings for fine-tuning.
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Maximum sequence length</td><td>256</td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Epochs</td><td>10</td></tr><tr><td>Learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Dropout</td><td>0.10</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Training objective</td><td>Cross-entropy</td></tr></table>

We implemented all experiments in PyTorch and used the Hugging Face Transformers library [23, 36]. The models are fine-tuned, meaning that both the pretrained encoder and the UPOS classification head are updated during training.

## 5.2 Evaluation Metrics

We use UPOS accuracy for the main model comparison. For a detailed analysis of the XLM-RoBERTa checkpoint selected by development accuracy, we also report per-class precision, recall, F1-score, support, and a confusion matrix.

Let N be the number of evaluated tokens, $y _ { i }$ the gold UPOS tag of token $i ,$ and $\hat { y } _ { i }$ its predicted tag. Accuracy is the proportion of correctly tagged tokens:

$$
\mathrm { A c c u r a c y } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } mathbf { 1 } \{ \hat { y } _ { i } = y _ { i } \} ,\tag{6}
$$

where $\mathbf { 1 } \{ \cdot \}$ equals 1 when its condition is true and 0 otherwise.

For each UPOS tag $^ { c , }$ let TP<sub>c</sub> be the number of tokens correctly predicted as $c , \mathrm { F P } _ { c }$ the number incorrectly predicted as $^ { c , }$ and $\mathrm { F N } _ { c }$ the number of gold c tokens predicted as another tag. Per-class precision, recall, and F1-score are defined as:

$$
P _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F P } _ { c } } } , \qquad R _ { c } = { \frac { \mathrm { T P } _ { c } } { \mathrm { T P } _ { c } + \mathrm { F N } _ { c } } } , \qquad F 1 _ { c } = { \frac { 2 P _ { c } R _ { c } } { P _ { c } + R _ { c } } } .\tag{7}
$$

The support of tag $c$ is its number of gold test tokens, $n _ { c } = \mathrm { T P } _ { c } + \mathrm { F N } _ { c }$

We report macro averages over the full set Y of 17 UPOS tags. For $M _ { c } \in$ $\{ P _ { c } , R _ { c } , F 1 _ { c } \}$ , the macro and support-weighted averages are:

$$
M _ { \mathrm { m a c r o } } = \frac { 1 } { | \mathcal { V } | } \sum _ { c \in \mathcal { V } } M _ { c } , \qquad M _ { \mathrm { w e i g h t e d } } = \frac { 1 } { N } \sum _ { c \in \mathcal { V } } n _ { c } M _ { c } , \quad | \mathcal { V } | = 1 7 .\tag{8}
$$

Following the evaluation implementation, a metric with a zero denominator is assigned zero. INTJ, SYM, and X have no gold test examples; their zero scores contribute to the 17-tag macro average but do not constitute measured performance on those tags.

## 6 Results and Analysis

Table 4 presents the mean development and test accuracy of the three evaluated transformer models over five random seeds. For each run, the checkpoint with the highest development accuracy was selected and evaluated on the held-out test set.

Table 5 shows that INTJ, SYM, and X have no gold examples in the test set. Their undefined per-class scores were set to zero when computing the 17-tag macro averages; the zeros do not represent measured performance on those tags.

Table 4. UPOS accuracy (%) over five random seeds, reported as mean $\pm$ sample standard deviation.
<table><tr><td>Model</td><td>Dev</td><td>Test</td></tr><tr><td>XLM-RoBERTa base</td><td> ${ \bf 8 7 . 2 9 \pm 0 . 2 1 }$ </td><td> $\mathbf { 8 8 . 6 0 \ : \pm { \ : 0 . 2 1 } }$ </td></tr><tr><td>Urdu RoBERTa small</td><td> $8 5 . 1 1 \pm 0 . 1 4$ </td><td> $8 6 . 8 5 \pm 0 . 2 1$ </td></tr><tr><td>Multilingual DistilBERT</td><td> $8 4 . 4 4 \pm 0 . 3 2$ </td><td> $8 6 . 0 3 \pm 0 . 3 0$ </td></tr></table>

Table 5. Per-class test results for XLM-RoBERTa (seed 46). Support is the number of gold test tokens.
<table><tr><td>UPOS</td><td>Precision</td><td>Recall</td><td>F1</td><td>Support</td></tr><tr><td>ADJ</td><td>0.6826</td><td>0.7451</td><td>0.7125</td><td>153</td></tr><tr><td>ADP</td><td>0.9616</td><td>0.9525</td><td>0.9570</td><td>526</td></tr><tr><td>ADV</td><td>0.7083</td><td>0.5312</td><td>0.6071</td><td>32</td></tr><tr><td>AUX</td><td>0.8883</td><td>0.8971</td><td>0.8927</td><td>204</td></tr><tr><td>CCONJ</td><td>0.8603</td><td>0.9070</td><td>0.8830</td><td>129</td></tr><tr><td>DET</td><td>0.4615</td><td>0.7500</td><td>0.5714</td><td>8</td></tr><tr><td>INTJ</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0</td></tr><tr><td>NOUN</td><td>0.8914</td><td>0.8676</td><td>0.8793</td><td>1,050</td></tr><tr><td>NUM</td><td>0.8478</td><td>0.8864</td><td>0.8667</td><td>44</td></tr><tr><td>PART</td><td>0.9655</td><td>0.8615</td><td>0.9106</td><td>65</td></tr><tr><td>PRON</td><td>0.8922</td><td>0.8563</td><td>0.8739</td><td>174</td></tr><tr><td>PROPN</td><td>0.8854</td><td>0.9051</td><td>0.8951</td><td>495</td></tr><tr><td>PUNCT</td><td>1.0000</td><td>1.0000</td><td>1.0000</td><td>258</td></tr><tr><td>SCONJ</td><td>0.8118</td><td>0.9452</td><td>0.8734</td><td>73</td></tr><tr><td>SYM</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0</td></tr><tr><td>VERB</td><td>0.7739</td><td>0.7807</td><td>0.7773</td><td>228</td></tr><tr><td>X</td><td>0.0000</td><td>0.0000</td><td>0.0000</td><td>0</td></tr><tr><td>Macro avg. (17 tags)</td><td>0.6842</td><td>0.6992</td><td>0.6882</td><td>3,439</td></tr><tr><td>Weighted avg.</td><td>0.8875</td><td>0.8857</td><td>0.8861</td><td>3,439</td></tr></table>

Table 6 presents the test-set confusion matrix for the XLM-RoBERTa checkpoint selected by development accuracy. Rows represent gold UPOS tags, and columns represent predicted tags. The model correctly labels 3,046 of 3,439 tokens. The most frequent errors are NOUN predicted as PROPN (53 tokens), PROPN predicted as NOUN (31), and NOUN predicted as ADJ (30). The INTJ, SYM, and X rows contain no examples because these tags are absent from the test set.

The results show that the choice of pretrained encoder has a clear efect on Saraiki UPOS tagging performance. XLM-RoBERTa base achieves the strongest mean result, with 87.29% development accuracy and 88.60% test accuracy. Urdu RoBERTa small obtains the second-best mean test result at 86.85%, followed by multilingual DistilBERT at 86.03%.

Table 6. Test-set UPOS confusion matrix for XLM-RoBERTa (seed 46). Rows are gold tags; columns are predicted tags.
<table><tr><td>Gold</td><td>AUX NUM NOUN CCONJ VERB </td><td></td><td></td><td></td><td></td><td></td><td>ADP PUNCT ADJ PART PROPN PRON SCONJ ADV DET SYM X INTJ</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AUX</td><td>183</td><td>0</td><td>1</td><td>0</td><td>19</td><td>0</td><td></td><td>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>00 0</td></tr><tr><td>NUM</td><td>0</td><td>39</td><td>3</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>0</td></tr><tr><td>NOUN</td><td>4</td><td>4</td><td>911</td><td>1</td><td>23</td><td>7</td><td>0</td><td>30</td><td>0</td><td>53</td><td>11</td><td>4</td><td>1</td><td>1</td><td>0 0</td><td>0</td></tr><tr><td>CCONJ</td><td>0</td><td>0</td><td>0</td><td>117</td><td>0</td><td>10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>VERB</td><td>13</td><td>0</td><td>26</td><td>0</td><td>178</td><td>0</td><td>0</td><td>5</td><td>1</td><td>1</td><td>3</td><td>0</td><td>0</td><td>1</td><td>0 0</td><td>0</td></tr><tr><td>ADP</td><td>0</td><td>1</td><td>5</td><td>17</td><td>1</td><td>501</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>PUNCT</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>258</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>ADJ</td><td>0</td><td>2</td><td>26</td><td>0</td><td>3</td><td>0</td><td>0</td><td>114</td><td>0</td><td>4</td><td>1</td><td>1</td><td>0</td><td>2</td><td>0 0</td><td>0</td></tr><tr><td>PART</td><td>1</td><td>0</td><td>3</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td><td>56</td><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>PROPN</td><td>1</td><td>0</td><td>31</td><td>0</td><td>3</td><td>0</td><td>0</td><td>12</td><td>0</td><td>448</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>PRON</td><td>1</td><td>0</td><td>9</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>0</td><td>0</td><td>149</td><td>7</td><td>2</td><td>3</td><td>0 0</td><td>0</td></tr><tr><td>SCONJ</td><td>1</td><td>0</td><td>2</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>69</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>ADV</td><td>2</td><td>0</td><td>5</td><td>0</td><td>2</td><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>1</td><td>17</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>DET</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>6</td><td>0 0</td><td>0</td></tr><tr><td>SYM</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>X</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr><tr><td>INTJ</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0 0</td><td>0</td></tr></table>

The ranking of the models is consistent across the development and test sets. XLM-RoBERTa base performs best on both splits, followed by Urdu RoBERTa small and multilingual DistilBERT. This consistency suggests that the observed diferences are not specific to a single split.

XLM-RoBERTa base improves over multilingual DistilBERT by 2.57 percentage points on the test set and over Urdu RoBERTa small by 1.75 percentage points. These results indicate that large-scale multilingual pretraining provides a measurable advantage for Saraiki UPOS tagging. At the same time, Urdu RoBERTa small remains competitive despite being smaller than XLM-RoBERTa base, suggesting that related-script pretraining can also be useful for Saraiki text.

Multilingual DistilBERT obtains the lowest accuracy among the three models, but it still provides a competitive lightweight baseline. Its result shows that smaller multilingual encoders can be useful when computational eficiency is important, although they may not match the accuracy of larger multilingual models.

Overall, the results suggest that multilingual and related-script pretrained encoders are suitable for Saraiki UPOS tagging. XLM-RoBERTa base is the strongest model under the current setup, while Urdu RoBERTa small and multilingual DistilBERT provide useful alternative baselines.

## 7 Discussion

The results demonstrate that pretrained transformer encoders can be efective for Saraiki UPOS tagging, but the choice of encoder remains important. The strongest performance is achieved by XLM-RoBERTa base, which suggests that broad multilingual pretraining and wider subword coverage are beneficial for low-resource token classification.

The performance of Urdu RoBERTa small is also notable. Although Urdu and Saraiki are diferent languages, both are written in Perso-Arabic script and share some orthographic patterns. This may help an Urdu-focused model represent Saraiki tokens more efectively than expected. However, Urdu RoBERTa small still remains below XLM-RoBERTa base, indicating that related-script pretraining is helpful but does not fully replace the benefits of broader multilingual pretraining.

These findings also highlight the value of building annotated resources for Saraiki. Existing Saraiki NLP resources remain limited, and manually annotated UPOS data can support the development and evaluation of automatic taggers. The dataset can also serve as a foundation for future work on morphological analysis, dependency parsing, and other downstream NLP tasks.

At the same time, the current work focuses only on coarse-grained UPOS tagging. Saraiki has rich morphology and several language-specific grammatical phenomena, including light verb constructions, serial verb constructions, split ergativity, and dialectal variation. Future work should therefore extend the corpus with morphological features, language-specific POS tags, and dependency relations.

Overall, the dataset and baseline results provide a starting point for more systematic research on Saraiki NLP and low-resource sequence labeling.

## 8 Conclusion

This paper introduced a manually annotated Saraiki Universal POS tagging dataset developed from plain-text Saraiki data. The corpus was annotated with the Universal POS tag inventory and converted into CoNLL-U format, providing fixed train, development, and test splits for reproducible evaluation.

We established transformer-based baselines for Saraiki UPOS tagging by finetuning XLM-RoBERTa base, Urdu RoBERTa small, and multilingual Distil-BERT under a consistent token classification setup. The results show that multilingual and related-script pretrained encoders are efective for Saraiki sequence labeling, with XLM-RoBERTa base providing the strongest overall performance among the evaluated models.

Overall, this work provides a benchmark resource and baseline results for Saraiki POS tagging. The dataset can support future research on low-resource NLP, Universal Dependencies-oriented annotation, and downstream Saraiki language processing. Future work will expand the corpus with additional data, morphological features, and dependency relations, and will explore further modeling approaches such as CRF decoding, parameter-eficient fine-tuning, and data augmentation.

## Limitations

This study has several limitations. First, the dataset is limited in size compared with POS-tagged corpora available for high-resource languages. Although the manually annotated corpus provides a useful benchmark for Saraiki UPOS tagging, larger datasets would likely improve model performance and support more reliable evaluation across diferent genres and dialects.

Second, the present work focuses only on Universal POS tags. UPOS categories are useful for cross-linguistic comparison, but they do not capture the full range of Saraiki-specific morphological and syntactic information. Saraiki has rich verbal morphology, case marking, light verb constructions, serial verb constructions, and other language-specific phenomena. Future versions of the corpus should include morphological features, language-specific POS tags, and dependency relations.

Third, INTJ, SYM, and X have no examples in the test set, so their test performance cannot be assessed. The per-class analysis is reported for one checkpoint selected using development accuracy; the main accuracy comparison is reported over five random seeds.

Fourth, the model comparison is limited to three pretrained transformer encoders. These models provide useful multilingual and related-script baselines, but additional models should be evaluated in future work. This includes larger multilingual encoders, regional language models, Urdu and Punjabi-focused models, and parameter-eficient fine-tuning methods.

Finally, the annotation process may involve dificult linguistic decisions, especially for ambiguous tokens whose POS category depends on context. Since Saraiki also has dialectal and lexical variation, some forms may be used diferently across regions. Although two experts reviewed the full corpus, the same sentences were not independently annotated from scratch by multiple annotators, and inter-annotator agreement was not measured. Future work should document more detailed annotation guidelines and use independent annotation of a representative subset to measure agreement.

## Data Availability

The annotated Saraiki UPOS corpus and its fixed training, development, and test splits are available at https://github.com/ancientintelligencelab/saraikiupos.

## Ethics Statement

This work uses Saraiki plain-text data and manually annotates it for Universal POS tagging. The dataset is intended for research on low-resource NLP and language technology development. The corpus may reflect biases, dialectal preferences, or domain-specific patterns present in the original text source. Models trained on this dataset should therefore be used with care, especially outside the domain represented by the data. The resource and models are not intended for high-stakes decision-making.

## References

1. Ahmed, T., Urooj, S., Hussain, S., Mustafa, A., Parveen, R., Adeeba, F., Hautli-Janisz, A., Butt, M.: The cle urdu pos tagset (2015)

2. Alam, M., Tyers, F., Hanink, E., K¨ubler, S.: Universal dependencies for saraiki. In: Proceedings of the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD)@ LREC-COLING 2024. pp. 188–197 (2024)

3. Arora, A.: Universal dependencies for punjabi. In: Proceedings of the Thirteenth Language Resources and Evaluation Conference. pp. 5705–5711 (2022)

4. Bhat, R.A., Sharma, D.M.: Dependency treebank of urdu and its evaluation. In: Proceedings of the Sixth Linguistic Annotation Workshop. pp. 157–165 (2012)

5. Brants, T.: Tnt–a statistical part-of-speech tagger. In: Sixth Applied Natural Language Processing Conference. pp. 224–231 (2000)

6. Brill, E.: A simple rule-based part of speech tagger. In: Speech and Natural Language: Proceedings of a Workshop Held at Harriman, New York, February 23-26, 1992 (1992)

7. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzm´an, F., Grave, E., Ott, M., Zettlemoyer, L., Stoyanov, V.: Unsupervised cross-lingual representation learning at scale. In: Proceedings of the 58th annual meeting of the association for computational linguistics. pp. 8440–8451 (2020)

8. De Marnefe, M.C., Manning, C.D., Nivre, J., Zeman, D.: Universal dependencies. Computational linguistics 47(2), 255–308 (2021)

9. Devlin, J., Chang, M.W., Lee, K., Toutanova, K.: Bert: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers). pp. 4171–4186 (2019)

10. Ehsan, T., Butt, M.: Dependency parsing for urdu: Resources, conversions and learning. In: Proceedings of the Twelfth Language Resources and Evaluation Conference. pp. 5202–5207 (2020)

11. Hedderich, M.A., Lange, L., Adel, H., Str¨otgen, J., Klakow, D.: A survey on recent approaches for natural language processing in low-resource scenarios. In: Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. pp. 2545–2568 (2021)

12. Huang, Z., Xu, W., Yu, K.: Bidirectional lstm-crf models for sequence tagging. arXiv preprint arXiv:1508.01991 (2015)

13. Kaleem Art Press: Kaleem art press saraiki literature corpus. Mozilla Data Collective (Dec 2025), https://mozilladatacollective.com/datasets/cmiq9nlul0042o207d0gnxv5p, dataset ID: cmiq9nlul0042o207d0gnxv5p; License: CC-BY-NC-4.0; accessed 20 June 2026

14. Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K., Dyer, C.: Neural architectures for named entity recognition. In: Proceedings of the 2016 conference of the North American chapter of the association for computational linguistics: human language technologies. pp. 260–270 (2016)

15. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. arxiv. arXiv preprint arXiv:1711.05101 10 (2019)

16. Malik, M.H., Ghous, H., Perveen, S.: A hierarchical part of speech tag set for saraiki language. Southern Journal of Arts & Humanities 1(02), 1–11 (2023)

17. Manning, C., Schutze, H.: Foundations of statistical natural language processing. MIT press (1999)

18. McDonald, R., Nivre, J., Quirmbach-Brundage, Y., Goldberg, Y., Das, D., Ganchev, K., Hall, K., Petrov, S., Zhang, H., T¨ackstr¨om, O., et al.: Universal dependency annotation for multilingual parsing. In: Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers). pp. 92–97 (2013)

19. Nawaz, U., Iqbal, M.J., Alyas, T., Asaf, M., Yaqoob, S., Raza, U.A., Nadim, M.A., Rafique, A., Rehman, F.: Shahiemotion: A benchmark dataset for punjabi shahmukhi emotion detection. In: Proceedings of the 1st Workshop on Multilinguality in the Era of Large Language Models (MeLLM 2026). pp. 211–220 (2026)

20. Nawaz, U., Napolitano, M., Karafillidis, I., Presti, L.L., Cascia, M.: Towards benchmarking old church slavonic lemmatization. In: Proceedings of the Third Workshop on the Bridges and Gaps between Formal and Computational Linguistics (BriGap-3). pp. 82–93 (2026)

21. Nivre, J., De Marnefe, M.C., Ginter, F., Hajic, J., Manning, C.D., Pyysalo, S., Schuster, S., Tyers, F., Zeman, D.: Universal dependencies v2: An evergrowing multilingual treebank collection. In: Proceedings of the twelfth language resources and evaluation conference. pp. 4034–4043 (2020)

22. Pakistan Bureau of Statistics: 7th population and housing census 2023: National census report. Tech. rep., Pakistan Bureau of Statistics, Government of Pakistan, Islamabad, Pakistan (2024), https://www.pbs.gov.pk/wpcontent/uploads/2020/07/National-Census-Report-2023.pdf

23. Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., et al.: Pytorch: An imperative style, highperformance deep learning library. Advances in neural information processing systems 32 (2019)

24. Ravishankar, V.: A universal dependencies treebank for marathi. In: Proceedings of the 16th international workshop on treebanks and linguistic theories. pp. 190–200 (2017)

25. Reimers, N., Gurevych, I.: Reporting score distributions makes a diference: Performance study of lstm-networks for sequence tagging. In: Proceedings of the 2017 conference on empirical methods in natural language processing. pp. 338– 348 (2017)

26. Ruder, S., Peters, M.E., Swayamdipta, S., Wolf, T.: Transfer learning in natural language processing. In: Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: Tutorials. pp. 15–18 (2019)

27. Sajjad, H., Schmid, H.: Tagging urdu text with parts of speech: A tagger comparison. In: Proceedings of the 12th Conference of the European Chapter of the ACL (EACL 2009). pp. 692–700 (2009)

28. Saleemi, F.J., Asghar, M.N., Iqbal, S., Chaudhryc, M.U., Yasird, M., Bazaie, S.U., Khan, M.Q.: A basic parts of speech (pos) tagset for morphological, syntactic and lexical annotations of saraiki language (2021)

29. Sanh, V., Debut, L., Chaumond, J., Wolf, T.: Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108 (2019)

30. Sarwat, S., Imran, M., Shahzad, M.W., Numra, M.: Lexical variation between two saraiki dialects (multani and riasti) in southern punjab pakistan. SYLWAN 164(7) (2020)

31. Tandon, J., Chaudhry, H., Bhat, R.A., Sharma, D.M.: Conversion from paninian karakas to universal dependencies for hindi dependency treebank. In: Proceedings of the 10th Linguistic Annotation Workshop held in conjunction with ACL 2016 (LAW-X 2016). pp. 141–150 (2016)

32. Tehseen, A., Ehsan, T., Liaqat, H.B., Ali, A., Al-Fuqaha, A.: Neural pos tagging of shahmukhi by using contextualized word representations. Journal of King Saud University-Computer and Information Sciences 35(1), 335–356 (2023)

33. Toutanova, K., Klein, D., Manning, C.D., Singer, Y.: Feature-rich part-of-speech tagging with a cyclic dependency network. In: Proceedings of the 2003 human language technology conference of the north american chapter of the association for computational linguistics. pp. 252–259 (2003)

34. Tseng, H., Jurafsky, D., Manning, C.D.: Morphological features help pos tagging of unknown words across language varieties. In: Proceedings of the fourth SIGHAN workshop on Chinese language processing (2005)

35. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)

36. Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., et al.: Transformers: State-of-the-art natural language processing. In: Proceedings of the 2020 conference on empirical methods in natural language processing: system demonstrations. pp. 38–45 (2020)