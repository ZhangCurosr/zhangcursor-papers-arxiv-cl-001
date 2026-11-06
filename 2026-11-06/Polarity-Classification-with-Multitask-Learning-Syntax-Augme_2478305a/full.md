# Polarity Classification with Multitask Learning Syntax Augmentation

Muhammad Imran1,\*, Carlos Gómez-Rodríguez

1 Universidade da Coruña, CITIC Depto. de Ciencias de la Computación y Tecnologías de la Información, Campus de Elviña s/n, 15071 A Coruña, Spain

Corresponding author: m.imran@udc.es

## Abstract

Sentiment Analysis (SA) is a fundamental NLP task concerned with the identification and categorization of subjective information in text. While lexicon-based methods offer transparency, they often struggle with linguistic phenomena such as negation and intensifi. cation. Syntactic parsing has been integrated into SA pipelines to address this, but conventional parsers introduce significant computational overhead. Recent work shows that reformulating dependency parsing as a Sequence Labeling (SEL) task can improve efficiency. We propose and compare two strategies for integrating syntactic knowledge into a neural sentiment analysis model under unified Multi-Task Learning (MTL). The first, Standard-MTL, jointly learns dependency parsing (via sequence labeling) and sentiment polarity classification using a shared encoder with two task-specific heads. The second, Contrastive-MTL, provides implicit syntactic guidance through a contrastive auxiliary objective and a syntax-biased attention pooling mechanism, without requiring explicit label prediction. Both rely on syntactic annotations during training and eliminate need for hand-crafted compositional rules or external parsing at inference. We evaluate eleven datasets spanning seven languages—the MARC corpus (English, Spanish, French, German) and seven datasets from SemEval-2022 Task 10 (English, Spanish, Norwegian, Basque, Catalan)—using four established SEL encodings. With the relative encoding, Contrastive-MTL outperforms a syntax-naive baseline on 10 of 11 datasets and surpasses the external rule-based pipeline (SELSP) that parses at inference on all six datasets where the latter applies, while being 1.5–9× faster at evaluation. Standard-MTL is more dataset- and encodingdependent. These results show implicit syntactic supervision during training can match or exceed explicit parsing at inference without computational overhead.

## 1 Introduction

Sentiment Analysis (SA) is a fundamental task in Natural Language Processing (NLP) that deals with the identification and categorization of subjective information in a given text, typically involving subjectivity detection and polarity classification (e.g., positive, negative, or neutral) (Cui et al., 2023). In this article, we focus on sentiment polarity classification: given an input text x (a sentence or document), predict its overall sentiment polarity y ∈ , where  typically contains labels such as positive, negative, and neutral. This is a sentenceor document-level classification task. We do not address fine-grained tasks like structured or aspectbased sentiment analysis.

Lexicon-based approaches leverage sentiment dictionaries like SO-CAL (Brooke et al., 2009) to detect polarity-bearing words, providing an intuitive and transparent approach to SA. The explainability of lexicon-based methods stands in contrast to that of “black-box" supervised machine learning models. In lexicon-based approaches, the predicted sentiment can be traced directly to specific lexical items, whereas supervised models, despite often achieving high accuracy, obscure the reasoning behind their predictions (Cui et al., 2023).

However, the accuracy of lexicon-based approaches depends on the handling of linguistic phenomena such as negation (“not good") and intensification (“very good"), which have the tendency to shift or modify sentiment polarity. The scope of these linguistic phenomena is quite complex. To handle it properly, researchers have integrated syntactic parsing within rule-based SA systems (Vilares et al., 2017; Imran et al., 2026b). By analyzing the syntactic structure of a sentence-often represented via Universal Dependencies (UD) (Zeman et al., 2023), rule-based systems can apply recursive, compositional rules to accurately determine the scope of polarity-shifting elements. Despite these advantages in accuracy and explainability, conventional syntactic parsers, such as Stanza (Qi et al., 2020), are computationally expensive, creating a bottleneck for large-scale or real-time SA applications (Gómez-Rodríguez and Vilares, 2018; Strzyz et al., 2019).

A promising solution to this computational bottleneck is the reformulation of dependency parsing as a Sequence Labeling (SEL) task (Gómez-Rodríguez and Vilares, 2018; Strzyz et al., 2019). SEL parsers rely on lightweight scripts (Roca et al., 2022) to linearize the tree structure into a sequence of labels, enabling the use of fast, off-the-shelf sequence taggers for syntactic analysis. Recent work by Imran et al. (2026b) successfully applied a Sequence Labeling Syntactic Parser (SELSP) within a rule-based sentiment analysis pipeline, demonstrating significant improvements in processing speed over traditional parsers while maintaining competitive accuracy for polarity classification.

While effective, the pipeline approach in (Imran et al., 2026b) treats parsing and sentiment analysis as two discrete stages: first, a full parse tree is generated via sequence labeling, and then a separate rule-based engine computes sentiment from that structure. In this article, we propose two alternative strategies for integrating syntactic knowledge under a unified Multi-Task Learning (MTL) framework. Unlike prior work that uses a pipeline architecture—where a sequence labeling parser first generates a full parse tree, followed by a separate rule-based engine that computes sentiment from that structure—our proposed MTL models leverage syntactic annotations (generated by an accurate off-the-shelf parser) during training to provide an auxiliary learning signal, thereby eliminating the need for hand-crafted compositional rules or external parsing at inference time. Our first variant, Standard-MTL, employs explicit multi-task supervision by jointly training a shared encoder to predict sentiment polarity and dependency relation labels simultaneously. In contrast, our second variant, Contrastive-MTL, integrates syntax implicitly through a contrastive auxiliary objective and a syntax-biased attention pooling mechanism. Crucially, both models share an identical baseline sentiment architecture; the auxiliary syntactic signals are discarded at inference to ensure that any performance gains stem solely from the internalized linguistic structure learned during training.

Recent advances in contrastive learning have demonstrated the value of both sentence-level and token-level objectives. Sentence-level methods such as SimCSE (Gao et al., 2021), DeCLUTR (Giorgi et al., 2021), and supervised contrastive learning (Khosla et al., 2020) learn robust representations by pulling together semantically similar sentences and pushing apart dissimilar ones. More recently, token-level objectives have proven effective in different settings: MEXMA (Janeiro et al., 2025) combines sentence-level and token-level objectives for cross-lingual sentence encoders by predicting masked tokens across languages, while Critical Tokens Matter (Lin et al., 2024) uses token-level contrastive estimation to identify and penalize critical tokens in LLM reasoning trajectories. These works establish that token-level signals can shape representations more effectively than sentence-level objectives alone. However, these methods either operate at the sentence/instance level or target crosslingual encoding and LLM reasoning; none uses dependency relations as token-level contrastive signals for sentiment analysis. Our Contrastive-MTL fills this gap by combining token-level syntax and sentiment contrastive losses with syntax-biased attention pooling, injecting syntactic knowledge without requiring parsing at the inference stage.

Our contributions are threefold: (1) We present two complementary multi-task learning strategies for integrating syntactic information into sentiment analysis: Standard-MTL, which jointly learns dependency parsing as sequence labeling and sentiment classification, and MTL-Contrastive, which injects syntactic guidance implicitly through a contrastive auxiliary objective and syntax-biased attention pooling. (2) We evaluate both MTL variants on eleven datasets: MARC product reviews (spanning four language subsets) and seven SemEval-2022 Task 10 corpora covering hotel, news, and multidomain reviews—across seven languages in total and using four sequence-labeling encodings. We then compare their performance against a singletask baseline (sentiment-only, without syntactic supervision), an external rule-based pipeline that parses at inference (SELSP), and two ablations that isolate the contributions of the contrastive losses and the attention bias. (3) Through a mechanism analysis on a SemEval-2022 English dataset combining linear probing, inference-time attention analysis, and learned-bias inspection, we show that the contrastive objective produces encoder representations that are linearly separable by dependency relation, and that the attention bias alone — without the contrastive anchor — degrades encoder-side

syntactic encoding.

## 2 Related Work

Methods for SA are broadly categorized into Machine Learning/Deep Learning (ML/DL) approaches, lexicon-based methods, and hybrid techniques.

## 2.1 Supervised ML/DL methods

Supervised ML/DL methods, including models based on LSTMs, CNNs, and pre-trained Transformers like BERT and RoBERTa, represent the current state of the art in academic benchmarks (Cui et al., 2023). These models excel when training and test data share similar characteristics but often require large, domain-specific annotated corpora and lack interpretability. Lexicon-based methods, conversely, utilize sentiment dictionaries and are prized for their simplicity, cross-domain robustness, and explainability, making them prevalent in industrial applications (Taboada et al., 2011; Hutto and Gilbert, 2014; Denecke and Reichenpfader, 2023). Hybrid approaches seek to combine the strengths of both, for example, by using lexiconderived features to augment the input to a neural classifier (Fu et al., 2018).

## 2.2 Syntactic Parsing for Sentiment Analysis

To address the challenge of determining the scope of polarity-shifting elements like negation, a wellestablished line of research employs syntactic rules that operate on parse trees. Universal Dependencies (UD) (Zeman et al., 2023) has emerged as the dominant framework for this purpose due to its cross-lingual consistency and extensive treebank resources. Systems like those described by (Vilares et al., 2017) and (Imran et al., 2026b; Kellert et al., 2025a) use UD parse trees to apply bottom-up, compositional rules that modify and aggregate sentiment scores based on syntactic relations (e.g., adverbial modifiers for intensification, negation cues). This approach provides a linguistically motivated, transparent alternative to heuristic window-based scope detection (Thelwall et al., 2010) or opaque ML classifiers for scope resolution (Singh and Paul, 2021).

## 2.3 Dependency Parsing as Sequence Labeling

The computational overhead of traditional transition-based or graph-based dependency parsers has motivated the development of faster alternatives. By encoding a dependency tree into a sequence of labels that can be predicted by a simple tagger, parsing is reduced to a sequence labeling task (Gómez-Rodríguez and Vilares, 2018). The CoDeLin framework (Strzyz et al., 2019; Roca et al., 2022) supports various encoding schemes, including absolute, relative and POS-based offsets and a range of bracketing-based encodings, which can be used to represent a dependency tree as a sequence of one label per word. (Imran et al., 2026b) demonstrated that a SEL parser (SELSP) trained with these encodings could be integrated into a rule-based SA pipeline, substantially improving parsing speed with minimal loss in downstream sentiment accuracy compared to Stanza.

## 2.4 Multi-Task Learning in NLP

Multi-Task Learning (MTL) is a paradigm in which a single model is trained to perform multiple related tasks simultaneously, leveraging shared representations to improve generalization and efficiency for all tasks (van der Goot et al., 2021). In NLP, MTL has been successfully applied to learn joint representations for a variety of tasks. Collobert and Weston (2008) pioneered a unified architecture for tasks like part-of-speech tagging, chunking, and named entity recognition. More recently, MTL has been used to infuse syntactic knowledge into semantic tasks. For instance, (Swayamdipta et al., 2018) proposed a model that jointly learns syntactic chunking and semantic composition for sentence representations. Similarly, (Zhang and Weiss, 2016) showed that training a parser jointly with a sentence-level classification task (like sentiment) can lead to improved parsing accuracy. In the domain of syntactic parsing itself, MTL has been employed to jointly predict part-of-speech tags, morphological features, and dependency trees (Kondratyuk and Straka, 2019).

The incorporation of syntactic knowledge, ranging from dependency relations to constituent structures, has proven beneficial for handling complex linguistic phenomena in sentiment analysis, such as negation and intensification (Poria et al., 2014; Gómez-Rodríguez et al., 2024). Moreover, the effectiveness of parsing as sequence labeling has been further demonstrated in challenging multilingual settings, including code-switched and lowresource languages (Kellert et al., 2025b). In structured sentiment analysis, Imran et al. (2026a) linearize sentiment dependency graphs over holders, targets, and expressions and decode them through sequence labeling. However, that work targets finegrained structured sentiment analysis: the sequence labeler predicts the graph structure itself, and the output at inference is a sentiment graph. In contrast, this study focuses on sentence/document-level sentiment polarity classification. Here, we use dependency parsing, formulated as sequence labeling, as an auxiliary training signal; the parser is discarded at inference, and the final output is a single polarity label for the input text. Thus, while both works exploit sequence labeling for dependency structures, they differ in task formulation, output space, and inference-time objective.

Beyond sentiment analysis, syntactic supervision via MTL has also proven beneficial for other sequence labeling tasks, such as biomedical named entity recognition (Imran et al., 2026c). Our work builds upon these principles by applying MTL specifically to the combination of dependency parsing (as sequence labeling) and sentiment polarity classification, offering a unified alternative to the two-stage rule-based pipelines such as Imran et al. (2026b); Kellert et al. (2025a) which require parsing at inference time.

## 2.5 Token-Level Contrastive Learning

Contrastive learning in NLP has largely focused on sentence-level objectives. SimCSE (Gao et al., 2021) uses dropout as noise to create positive pairs, DeCLUTR (Giorgi et al., 2021) constructs positive pairs from different spans of the same document, and supervised contrastive learning (Khosla et al., 2020) uses class labels to define positive pairs. Although, these methods learn powerful sentence representations but do not explicitly model token-level syntactic structures.

Token-level objectives have recently shown promise: MEXMA (Janeiro et al., 2025) combined sentence-level and token-level objectives for crosslingual sentence encoders by using the sentence representation in one language to predict masked tokens in another, improving bitext mining and downstream tasks. Critical Tokens Matter (Lin et al., 2024) introduced token-level contrastive estimation to identify “critical tokens" in LLM reasoning trajectories and penalized them via contrastive direct preference optimization, improving reasoning accuracy. Neither uses dependency relation labels as a contrastive signal nor syntax-guided attention for sentiment analysis. Our work differs by defining token-level contrastive losses conditioned on dependency relations and sentiment labels, combined with syntax-biased attention pooling. To our knowledge, this is the first such formulation for sentiment analysis.

## 3 Data and Preprocessing

We performed experiments on the Multilingual Amazon Reviews Corpus (MARC)¹ (Keung et al., 2020) a large collection of Amazon reviews for multilingual text in four languages (English, Spanish, French and German). Each review in this dataset provides a 5-star rating (1–5, mapped to labels 0–4 corresponding to very negative, negative, neutral, positive, and very positive, respectively) at the review level, making it well-suited for multilingual text classification. Furthermore, we performed experiments on the five datasets released in the SemEval-2022 Task 10: the OpeNER dataset (OpN; Agerri et al., 2013) with English and Spanish annotations, the English Multi-Perspective Question Answer corpus (MPQA; Wiebe et al., 2005) and the Darmstadt Universities corpus (DS; Toprak et al., 2010), the Norwegian NoReC dataset (Velldal et al., 2018) and the Basque and Catalan MultiBooked dataset (MB; Barnes et al., 2018).

In SemEval-2022 Task 10 datasets, every review is annotated with the individual polarity expressions and their polarity (positive or negative) as demonstrated by few simple examples of reviews in Table 1 from (Barnes et al., 2022) that only contain one polarity expression. Since our problem of interest is polarity classification, we extract review-level sentiment labels from the structured opinion annotations as described in Algorithm 1. Each review in the dataset contains an opinions field consisting of tuples (P, I, E), where P ∈ {Positive, Negative, Neutral} is the polarity, I ∈ {Strong, Standard or Average, Weak}, is the intensity, and E is the polar expression span. We discard reviews that contain no polarity expressions at all (i.e., the opinions field is empty), so our evaluation is comprised only on reviews that do contain subjectivity. For reviews with multiple polarity expressions, we compute an aggregate review-level polarity label by scoring each opinion based on both its polarity and intensity: Positive opinions contribute +2 if Strong and +1 otherwise, Negative opinions contribute —2 if Strong and —1 otherwise, and Neutral opinions are tracked but do not affect the score. The final gold polarity label is determined by comparing the aggregated positive and negative scores: the label is Positive if the positive score exceeds the negative score, Negative if the negative score exceeds the positive score, and Neutral if the scores are tied and at least one non-neutral or neutral opinion exists. We thus have a task of predicting three polarity labels (Positive, Negative, and Neutral) on these datasets. The statistics of these datasets, including the number of discarded and remaining reviews per split, are presented in Table 2.

Table 1: Polarity expression and corresponding polarity from the English training dataset in (Agerri et al., 2013)
<table><tr><td>Review</td><td>Polarity</td></tr><tr><td>Clean , renewed rooms.</td><td>Positive</td></tr><tr><td>Food very basic.</td><td>Negative</td></tr><tr><td>What can we say.</td><td>No Annotation</td></tr></table>

Each sentence is annotated with a review-level sentiment label. We used Biaffine Dependency Parser (Dozat and Manning, 2017) (dep-biaffinexlmr) available in the SuPar package ² to generate dependency parse trees (dependency head & relations). To cast dependency parsing as sequence labeling we relied on CoDeLin framework (Strzyz et al., 2019; Roca et al., 2022) which supports various encoding schemes to represent dependency trees as sequences of labels. Following CoDeLin, each word $w _ { i }$ in a sentence of length n is assigned a label $( x _ { i } , l _ { i } )$ , where $l _ { i }$ is the dependency relation and $x _ { i }$ encodes the position of its head. A dummy root node 0 is used. We consider four encodings:

• Absolute: $x _ { i }$ is the absolute index of the head, as in the CoNLL format.

• Relative: $x _ { i }$ is the head index minus the dependent index.

• BrkBased: $x _ { i }$ is a string of bracketing symbols that encodes the tree structure. Arcs are represented by symbols such $\mathrm { a s } < , > , \backslash ,$ and / at fencepost positions between words, allowing the tree to be reconstructed with a stack.

• Brk7Bits: a compact variant of BrkBased in which each bracketing component is encoded as a fixed 7-bit integer, reducing the label space (Gómez-Rodríguez et al., 2023).

Table 2: Dataset statistics by split: total rows, empty labels, and remaining usable rows
<table><tr><td>Dataset</td><td>Data Split</td><td>Total Rows</td><td>Discarded Rows</td><td>Remaining Rows</td></tr><tr><td rowspan="3"> $\mathrm { O p e N E R } _ { \mathrm { e n } }$ </td><td>Train Dev</td><td>1744</td><td>344</td><td>1400</td></tr><tr><td></td><td>249</td><td>51</td><td>198</td></tr><tr><td>Test</td><td>499</td><td>92</td><td>407</td></tr><tr><td rowspan="3"> $\mathrm { O p e N E R } _ { \mathrm { e s } }$ </td><td>Train</td><td>1438</td><td>186</td><td>1252</td></tr><tr><td>Dev</td><td>206</td><td>32</td><td>174</td></tr><tr><td>Test</td><td>410</td><td>48</td><td>362</td></tr><tr><td rowspan="3">Darmstadt_unis</td><td>Train</td><td>2253</td><td>1572</td><td>681</td></tr><tr><td>Dev</td><td>232</td><td>150</td><td>82</td></tr><tr><td>Test</td><td>318</td><td>214</td><td>104</td></tr><tr><td rowspan="3">MPQA</td><td>Train</td><td>5619</td><td>4371</td><td>1248</td></tr><tr><td>Dev</td><td>1991</td><td>1575</td><td>416</td></tr><tr><td>Test</td><td>2042</td><td>1660</td><td>382</td></tr><tr><td rowspan="3">MultiBookedca</td><td>Train</td><td>1174</td><td>172</td><td>1002</td></tr><tr><td>Dev</td><td>167</td><td>27</td><td>140</td></tr><tr><td>Test</td><td>335</td><td>54</td><td>281</td></tr><tr><td rowspan="3">MultiBookedeu</td><td>Train</td><td>1063</td><td>164</td><td>899</td></tr><tr><td>Dev</td><td>152</td><td>32</td><td>120</td></tr><tr><td>Test</td><td>305</td><td>65</td><td>240</td></tr><tr><td rowspan="3">NoReC</td><td>Train</td><td>8634</td><td>4313</td><td>4321</td></tr><tr><td>Dev</td><td>1531</td><td>748</td><td>783</td></tr><tr><td>Test</td><td>1272</td><td>633</td><td>639</td></tr><tr><td rowspan="3"> $\mathbf { M A R C } _ { \mathrm { e n } }$ </td><td>Train</td><td>200000</td><td>0</td><td>200000</td></tr><tr><td>Dev</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td>Test</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td rowspan="3"> $\mathbf { M A R C } _ { \mathrm { e s } }$ </td><td>Train</td><td>200000</td><td>0</td><td>200000</td></tr><tr><td>Dev</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td>Test</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td rowspan="3"> $\mathbf { M A R C } _ { \mathrm { f r } }$ </td><td>Train</td><td>200000</td><td>0</td><td>200000</td></tr><tr><td>Dev</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td>Test</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td rowspan="3"> $\mathbf { M A R C } _ { \mathrm { d e } }$ </td><td>Train</td><td>200000</td><td>0</td><td>200000</td></tr><tr><td>Dev</td><td>5000</td><td>0</td><td>5000</td></tr><tr><td>Test</td><td>5000</td><td>0</td><td>5000</td></tr></table>

## 4 Methodology

We propose two multi-task learning (MTL) strategies that integrate syntactic knowledge into a sentiment analysis model without requiring handcrafted rules or external parsing at inference time. Both approaches share a common transformer encoder and a sentiment classification head, but differ in how they leverage automatically obtained dependency annotations during training as shown in Figure 1. The first, Standard-MTL, casts dependency parsing (sequence labeling) as an explicit auxiliary task, jointly trained with sentiment prediction. The second, Contrastive-MTL, integrates syntax implicitly through a contrastive auxiliary objective and a syntax-biased attention pooling mechanism. The auxiliary syntactic components—including the dependency parsing head in Standard-MTL, and the syntax-guided attention pooling together with the contrastive projection head in Contrastive-MTL—are discarded after training. At inference, both models use a standard sentiment head over a plain pooling mechanism; in particular, Contrastive-MTL replaces the syntaxguided attention with a learned attention pooling that has no access to dependency relation labels.

Algorithm 1: Conversion of structured sentiment annotations to review-level polarity   
Input: A dataset D where each record contains an opinions field: a list of opinion tuples   
(P, I, E) with P ∈ {Positive, Negative, Neutral}, I ∈ {Strong, Standard, Average, Weak},   
and E the polar expression span (string or list of tokens).   
Output: A filtered dataset D' containing only reviews with a valid review-level polarity label,   
augmented with four new attributes: Polar\_expression, Polarity, Intensity, and   
gold\_polarity\_label.   
1 foreach record r ∈ D do   
2 L ← parse(r.opinions) ; // list of opinion dictionaries   
3 if L = ∅ then   
4 set all new attributes to empty string ;   
5 continue ;   
6 end   
r.Polar\_expression $ [ \operatorname { s p a n } ( E ) \mid ( P , I , E ) \in L ] ; \prime \prime$ use full span; join tokens if   
E is a list   
8 $r . \mathsf { P o l a r i t y } \gets \mathsf { j o i n } \big ( [ P \mid ( P , I , E ) \in L ] , \ast , \ast \big )$   
9 r.Intensity ←join $\left( [ I \mid ( P , I , E ) \in L ] , \dot { \mathbf { \Omega } } , \dot { \mathbf { \Omega } } \right)$   
10 posScore ← 0, negScore ← 0, hasNeutral ← false ;   
11 foreach (P, I, E) ∈ L do   
12 if P = Positive then   
13 if I = Strong then   
14 posScore ← posScore + 2   
15 end   
16 else   
17 posScore ← posScore + 1   
18 end   
19 else if P = Negative then   
20 if I = Strong then   
21 negScore ← negScore + 2   
22 end   
23 else   
24 negScore ← negScore + 1   
25 end   
26 else if P = Neutral then   
27 hasNeutral ← true   
28 end   
29 end   
30 if posScore > negScore then   
31 r.gold\_polarity\_label ←“Positive"   
32 else if negScore > posScore then   
33 r.gold\_polarity\_label ←“Negative"   
34 else if posScore = negScore and (posScore > 0 or hasNeutral) then   
35 r.gold\_polarity\_label ←“Neutral"   
36 end   
37 else   
38 r.gold\_polarity\_label ←“; // no valid opinion   
39 end   
40 end   
41 D′ ← {r ∈ D : r.gold\_polarity\_label ≠ “"} ; // remove discarded reviews   
42 return D' with review-level polarity labels

## 4.1 Standard Multi-Task Learning with Syntactic Supervision

The Standard-MTL model jointly learns to predict sentiment and dependency parses through explicit multi-task supervision. The architecture consists of a shared transformer encoder (XLM-RoBERTalarge, (Conneau et al., 2020)) and two separate task heads. The transformer-based XLM-RoBERTalarge was selected as the backbone encoder due to its proven effectiveness in cross-lingual settings and its ability to produce rich contextualized token representations essential for both dependency parsing (sequence labeling) and sentiment classification.

Given an input batch, the encoder produces contextualized token representations H $\in \mathbb { R } ^ { B \times L \times d }$ where d is the hidden size. For sentiment classification, a sentence-level representation is obtained via mean pooling over all non-padding tokens:

$$
\mathbf { h } _ { \mathrm { s e n t } } = \frac { \sum _ { j = 1 } ^ { L } \mathbf { H } _ { : , j , : } \cdot \mathbb { I } [ M _ { : , j } = 1 ] } { \sum _ { j = 1 } ^ { L } \mathbb { I } [ M _ { : , j } = 1 ] }
$$

where M is the attention mask. The pooled vector is passed through a sentiment classification head $\mathrm { H e a d } _ { \mathrm { s e n t } }$ , which consists of LayerNorm, a twolayer MLP with GELU activation and dropout (0.2), and a final linear projection to [| classes. We used the same architecture in our baseline model as well for the fair comparison.

For the auxiliary task, a dependency parsing head $\mathrm { H e a d _ { \mathrm { d e p } } }$ operates on every token's representation, consisting of a two-layer MLP: Linear $\cdot ( d , d ) \ \to \ \mathrm { G E L U } \ \to \ \mathrm { D r o p o u t } ( 0 . 1 ) \to \ \to \ \mathrm { { \Omega } }$ $\mathrm { L i n e a r } ( d , | \mathcal { P } | )$

The overall loss is a weighted sum of a labelsmoothed cross-entropy loss for sentiment and a standard cross-entropy loss for dependency parsing, with λ fixed to 0.3, where invalid positions (subtokens, padding) are ignored:

$$
{ \mathcal { L } } _ { \mathrm { M T L } } = { \mathcal { L } } _ { \mathrm { s e n t } } ( { \hat { y } } , y ) + \lambda \cdot { \mathcal { L } } _ { \mathrm { d e p } } ( \hat { \mathbf { p } } , \mathbf { p } )
$$

The sentiment loss uses label smoothing with

€ = 0.1:

$$
\mathcal { L } _ { \mathrm { s e n t } } = - \sum _ { c = 1 } ^ { | \mathcal { V } | } q _ { c } \log \hat { p } _ { c }
$$

where $\hat { p } = \operatorname { s o f t m a x } ( \hat { y } )$ is the predicted distribution, and $q _ { c } = ( 1 - \epsilon ) \mathbb { I } [ c = y ] + \epsilon / | y |$ is the smoothed target distribution that mixes the one-hot true label with a uniform prior.

By jointly optimizing sentiment classification and dependency parsing, the shared encoder learns to capture syntactic regularities—such as negation scope and modifier attachments—that are crucial to determine the sentiment polarity. The auxiliary parsing task prevents the encoder from overfitting to superficial lexical cues, encouraging it to build representations that are more robust to word order and syntactic variation. Consequently, the sentiment head benefits from a syntax-enriched feature space, leading to improved performance without requiring any additional computation at inference time.

## 4.2 Multi-Task Contrastive Learning with Syntax-Guided Attention

The Contrastive-MTL variant integrates syntactic information implicitly through two mechanisms: a syntax-guided attention pooling module and a pair of contrastive auxiliary objectives operating on token-level representations. The sentiment classification backbone is identical to that of Standard-MTL, except for the pooling step.

## 4.2.1 Syntax-Guided Attention Pooling

Instead of mean pooling, we use a learnable attention mechanism that can be biased by the dependency relations of the tokens. Given token representations H, attention scores are computed as:

$$
\mathbf { a } = \mathrm { s o f t m a x } \big ( \mathbf { W } _ { a } \mathbf { H } + \mathbf { b } _ { r _ { j } } ^ { \mathrm { r e l } } \big )
$$

where $\mathbf { W } _ { a } ~ \in ~ \mathbb { R } ^ { 1 \times d }$ is a learnable linear layer, and ${ \bf b } _ { r _ { j } } ^ { \mathrm { r e l } }$ is a scalar bias taken from an embedding lookup $\mathbf { E } _ { \mathrm { r e l } } \in \mathbb { R } ^ { | \mathcal { R } | \times 1 }$ indexed by the dependency relation label $r _ { j }$ of token $j .$ The relationdependent bias ${ \bf b } _ { r _ { j } } ^ { \mathrm { r e l } }$ is only available during training because it requires gold dependency relations. At inference, this term is omitted, and the model uses the learned linear attention ${ \bf W } _ { a }$ alone, i.e., $\mathbf { a } = \mathrm { s o f t m a x } \big ( \mathbf { W } _ { a } \mathbf { H } \big )$ The sentence representation is then the weighted sum:

$$
\mathbf { h } _ { \mathrm { s e n t } } = \sum _ { j = 1 } ^ { L } a _ { j } \cdot \mathbf { H } _ { : , j , : }
$$

![](images/fe717f92a55bafda2339d72c28c95d6af0ec658c9eba17d2be6a07180626ccbb.jpg)  
Figure 1: Overall architecture of the proposed Multi-Task Learning Framework: Joint Dependency Parsing (Standard-MTL) and Contrastive Syntax-Guided Attention (Contrastive-MTL)

This allows the model to learn which syntactic roles are more informative for sentiment, guided by the relation labels. The pooled representation is then fed into the sentiment head $\mathrm { ( H e a d _ { s e n t } ) }$

## 4.2.2 Contrastive Auxiliary Objectives

To further encourage the encoder to capture syntactic and sentiment regularities in the token space, we add contrastive losses on a projection head (Proj) that maps each token representation to a lowerdimensional space $\mathbb { R } ^ { d _ { \mathrm { p r o j } } }$ (with $d _ { \mathrm { p r o j } } = 2 5 6 )$

$$
\mathbf { Z } = \operatorname { P r o j } ( \mathbf { H } ) = \operatorname { L i n e a r } ( \operatorname { G E L U } ( \operatorname { L i n e a r } ( \mathbf { H } , d ) ) , d _ { \mathrm { p r o j } } ) )
$$

All token projections that correspond to valid firstsubword positions are normalized to the unit hypersphere. Two contrastive losses are then computed:

1. Syntax contrastive loss – tokens sharing the same dependency relation label (extracted from the parse string) are treated as positive pairs; those with different labels are negatives.

2. Sentiment contrastive loss – tokens belonging to sentences with the same sentiment label are positives; others are negatives (token-level sentiment labels are simply the sentence label repeated for all valid tokens).

For valid tokens with normalized projections $\mathbf { z } _ { i }$ and class labels $c _ { i } ,$ the contrastive loss computed as follows:

$$
\mathcal { L } _ { \mathrm { c o n t r a s t } } = - \frac { 1 } { | \mathcal { P } | } \sum _ { i \in \mathcal { P } } \log \frac { \frac { 1 } { | \mathcal { P } _ { i } | } \sum _ { j \in \mathcal { P } _ { i } } \exp ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { j } / \tau ) } { \sum _ { k \ne i } \exp ( \mathbf { z } _ { i } ^ { \top } \mathbf { z } _ { k } / \tau ) }
$$

where $\mathcal { P } _ { i } = \{ j \neq i : c _ { j } = c _ { i } \} , \mathcal { P } = \{ i :$ $\mathcal { P } _ { i } \neq \emptyset \}$ , and $\tau = 0 . 0 7$ . We compute this loss twice: once for $\mathcal { L } _ { \mathrm { s y n } }$ (using dependency relation labels as the class) and once for $\mathcal { L } _ { \mathrm { s e n t - c o n t r a s t } }$ (using sentence-level sentiment labels as the class). The positive term is normalized by $| \mathcal { P } _ { i } |$ per anchor i, preventing many-positive anchors from dominating and stabilizing training under relation/sentiment class imbalance; this is applied identically to $\mathcal { L } _ { \mathrm { s y n } }$ and $\mathcal { L } _ { \mathrm { s e n t - c o n t r a s t } } .$

## 4.2.3 Learnable Loss Weighting

The two auxiliary contrastive losses are combined with the main sentiment loss using uncertaintybased weighting. Following (Kendall et al., 2018), we introduce two learnable scalar parameters $\sigma _ { \mathrm { s y n } }$ and $\sigma _ { \mathrm { s e n t } }$ , which control the relative contribution of the syntax and sentiment contrastive objectives. The total training loss is:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { s e n t } } + \sum _ { \substack { t \in \{ \mathrm { s y n } , \mathrm { s e n t - c o n t r a s t } \} } } \left( \frac { 0 . 5 } { \sigma _ { t } ^ { 2 } } \mathcal { L } _ { t } + \log \sigma _ { t } \right)
$$

At the start of training, $\sigma _ { \mathrm { s y n } }$ and $\sigma _ { \mathrm { s e n t } }$ are initialized to $1 \left( \mathrm { i . e . , l o g } \sigma = 0 \right)$ . This formulation dynamically balances the magnitude of the auxiliary gradients, preventing one objective from dominating the optimization.

The contrastive auxiliary objectives and syntaxguided attention pooling implicitly distill syntactic knowledge into token representations. By pulling together tokens that share the same dependency relation or belong to sentences with the same sentiment, the model learns to associate specific syntactic roles (e.g., nsubj, amod) with sentiment expression, while the learnable attention bias further emphasizes relationally important tokens. This implicit syntactic shaping yields a more discriminative sentence representation, enabling the sentiment classifier to generalize better across syntactically diverse inputs—all without explicit parsing at the inference time.

## 4.3 Training Procedure

Table 3 summarizes the hyperparameters used to train the models. We use the XLM-RoBERTa-$\mathrm { l a r g e } ^ { 3 }$ encoder, initialized from pre-trained weights, with task-specific heads randomly initialized. Training is performed with the AdamW optimizer using two learning rates: $2 \times 1 0 ^ { - 5 }$ for the encoder parameters and $1 \times 1 0 ^ { - 4 }$ for the head parameters. A linear learning rate warmup is applied over the first 10% of the total training steps, followed by linear decay. Gradient norms are clipped to a maximum of 1.0.

We use a batch size of 16 and train for up to 50 epochs. Early stopping is employed on the development set macro $F _ { 1 }$ score with a patience of 7 epochs; the model state with the best development $F _ { 1 }$ is restored for final evaluation. All experiments are run with a fixed random seed (42) for reproducibility.

Table 3: Hyperparameters used to train the models
<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Encoder</td><td>XLM-RoBERTa-large</td></tr><tr><td>Encoder init. Head init.</td><td>Pre-trained Random</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Enc. learning rate</td><td> $2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Head learning rate LR schedule</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>Linear warmup + decay</td></tr><tr><td>Warmup</td><td>10% of steps</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr><tr><td></td><td></td></tr><tr><td>Batch size</td><td>16</td></tr><tr><td>Max epochs</td><td>50</td></tr><tr><td>Early stopping</td><td>7 epochs (dev  $F _ { 1 } )$ </td></tr><tr><td></td><td></td></tr><tr><td>Random seed</td><td>42</td></tr></table>

## 4.4 Evaluation

We measured the performance on the held-out test set and reported accuracy, macro-averaged precision, recall and $F _ { 1 }$ scores. Since the auxiliary tasks are discarded at inference, the evaluation procedure is identical for all models: only the sentiment logits are computed, and no additional parsing or syntactic information is required.

## 5 Results

## 5.1 Polarity Classification Performance

Table 4 presents a comparison of Standard Multi-Task Learning (Standard-MTL) and Contrastive Multi-Task Learning (Contrastive-MTL), both incorporating syntactic knowledge, against a baseline model without syntactic information and an external rule-based pipeline (SELSP) that requires parsing at inference time. While we report results for all encoding schemes, our primary analysis focuses on the relative encoding, which achieves the best overall performance and provides the most consistent improvements over both the baseline and SELSP. Note that SELSP only supports English and Spanish; consequently, it could not be applied to the remaining five datasets, which are in Catalan, Basque, Norwegian, French, and German.

With the relative encoding, Contrastive-MTL outperforms the baseline on 10 out of 11 datasets. The only exception is MPQA, where the baseline remains ahead (0.723 vs. 0.684). This consistency contrasts with Standard-MTL, which beats the baseline on only 6 out of 11 datasets with the same encoding. The gap between the two MTL variants is most pronounced on datasets with rich syntactic structure: on MultiBooked $\mathbf { l } _ { \mathrm { c a } } ,$ Contrastive-MTL improves over the baseline by +0.084 F1 versus +0.065 for Standard-MTL; on $\mathrm { O p e N E R } _ { \mathrm { e s } }$ , the improvement is +0.120 versus +0.083. On the MARC datasets, differences are narrower but consistent, with Contrastive-MTL achieving the best F1-scores across all four language subsets. Figure 2 visualises these ∆F1 values as a heatmap for all four encodings. Contrastive-MTL is positive on at least 9 of 11 datasets under every encoding (10 of 11 under BrkBased and Relative), while Standard-MTL is positive on at most 6 of 11 under every encoding and shows clearly negative regions on MPQA and MultiBookedeu.

Turning to SELSP, which is available for the six English and Spanish datasets, Contrastive-MTL with the relative encoding outperforms it on all six, whereas Standard-MTL beats SELSP on five out of six, falling short only on $\mathrm { O p e N E R _ { \mathrm { e s } } }$ (0.770 vs. 0.788) as visualized in Figure 3. This is particularly notable given that SELSP explicitly parses the input at inference time, while our MTL models discard the parser after training. On the two datasets where SELSP is most competitive—OpeNERes (SELSP F1 = 0.788) and $\mathrm { O p e N E R } _ { \mathrm { e n } }$ (SELSP F1 = 0.682)—Contrastive-MTL with the relative encoding achieves F1-scores of 0.807 and 0.804, respectively, demonstrating that implicit syntactic supervision during training can match or exceed the performance of explicit parsing at inference.

![](images/c7812744e1b2ead972b46280ea54739cb5f441a0ea08c74404e7802b6432acc7.jpg)  
Figure 2: ∆F1 relative to the syntax-naive baseline: (a) Standard-MTL and (b) Contrastive-MTL.

Although the relative encoding yields the strongest overall results, the pattern is consistent across the four tested encodings. Contrastive-MTL beats the baseline on 9–10 out of 11 datasets depending on the encoding, while Standard-MTL beats it on only 5–6. Figure 4 shows this encodingby-encoding robustness directly: Contrastive-MTL is at or above baseline on 39 of the 44 dataset encoding combinations (with a single exact tie at $\mathbf { M A R C } _ { \mathrm { e n } }$ under the Absolute encoding), and the only regressions are small $\mathrm { ( | \Delta | ~ \leq ~ 0 . 0 4 }$ all on MPQA, $\mathrm { O p e N E R } _ { \mathrm { e n } }$ and $\mathbf { M A R C } _ { \mathrm { e s } } )$ . In contrast, Standard-MTL drops below baseline in 21 of the 44 combinations, most severely on MPQA (down $\mathrm { t o \ - } 0 . 1 3 3 )$ . This indicates that the advantage of Contrastive-MTL is not an artifact of a particular encoding scheme but reflects a more effective integration of syntactic information.

## 5.2 Computational Efficiency

Table 6 reports training and evaluation times. SELSP is an external rule-based pipeline and is therefore not trained; its evaluation time is only reported for the English and Spanish datasets it supports. MTL variants require longer training times on some datasets relative to baseline, as seen on MultiBookedeu (652–664 vs. 263 s) or OpeNERes (653–851 vs. 459 s). However, the pattern reverses on other datasets: on NoReC, Standard-MTL trains faster than baseline (3026 vs. 4700 s), and on $\mathrm { O p e N E R } _ { \mathrm { e n } } .$ Contrastive-MTL is also faster (693 vs. 817 s). Between MTL variants, Contrastive-MTL is often more expensive to train than Standard-MTL (e.g., MPQA and $\mathbf { M A R C } _ { \mathrm { f r } } )$ , though not always $( \mathrm { e . g . , O p e N E R _ { \mathrm { e n } } }$ and Darmstadt\_unis). Inference times, by contrast, remain nearly identical across baseline, Standard-MTL, and Contrastive-MTL, since the parser is discarded after training.

This is not the case for SELSP, which must parse the input at inference time. Figure 5 summarises the inference cost across all datasets: the parser free MTL variants are statistically indistinguishable from the baseline (mean deviation $< ~ 6 \% )$ whereas SELSP is approximately 1.6–9.2× slower depending on the dataset, with the penalty widest on MPQA (9.2×) and Darmstadt\_unis (7.4×) and still clearly visible on the larger MARC corpora $( \approx 3 . 7 \times )$ . Thus, the MTL variants achieve competitive performance without the inference-time parsing overhead that makes SELSP costly.

## 6 Ablation Study

To isolate the contributions of the syntax bias and the contrastive losses, we define two ablation variants:

Table 4: Performance comparison of explicit versus implicit syntactic augmentation. Standard Multi-Task Learning (Standard-MTL) and Contrastive Multi-Task Learning (Contrastive-MTL) are compared against a syntax-naive baseline and an external rule-based pipeline (SELSP) across all datasets and encoding schemes. SELSP results are only reported for the English and Spanish datasets, as the system does not support the remaining languages (Catalan, Basque, Norwegian, French, and German)
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Encoding</td><td colspan="4">Standard-MTL</td><td colspan="4">Contrastive-MTL</td></tr><tr><td>Acc</td><td>Prec 0.546</td><td>Rec F1 0.549</td><td>Acc</td><td></td><td>Prec</td><td>Rec</td><td>F1 0.549</td></tr><tr><td>Darmstadt_unis</td><td>Baseline SELSP Brk7Bits BrkBased Absolute</td><td>0.822 0.414 0.581 0.843 0.574 0.867 0.576 0.867 0.576</td><td>0.553 0.414 0.562 0.584 0.581</td><td>0.399 0.567 0.580</td><td>0.822 0.414 0.865 0.911</td><td></td><td>0.546 0.581 0.575 0.608</td><td>0.553 0.414 0.582 0.612</td><td>0.399 0.579 0.610 0.579</td></tr><tr><td>MPQA</td><td>Relative Baseline SELSP Brk7Bits BrkBased Absolute Relative</td><td>0.856 0.882 0.327 0.902 0.891 0.898 0.710</td><td>0.570 0.582 0.751 0.707 0.381 0.350 0.714 0.665 0.590 0.606 0.662</td><td>0.578 0.573 0.723 0.283 0.680 0.598 0.677</td><td>0.867 0.889 0.882 0.327 0.890 0.887 0.891</td><td>0.591 0.751 0.381 0.756</td><td>0.580 0.745 0.810</td><td>0.579 0.601 0.707 0.350 0.714 0.707 0.716</td><td>0.595 0.723 0.283 0.729 0.725 0.744</td></tr><tr><td>MultiBookedca</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.879 0.886 0.904 0.904 0.907 0.889</td><td>0.578 0.643 0.707 0.707 0.693 0.695</td><td>0.605 0.641 0.683 0.683 0.657 0.739 0.707</td><td>0.590 0.883 0.642 0.886 0.693 0.904 0.693 0.904 0.664 0.900 0.907</td><td></td><td>0.916 0.643 0.704 0.704 0.695 0.715</td><td>0.650 0.641 0.716 0.716 0.713 0.748</td><td>0.684 0.642 0.709 0.709 0.702 0.726</td></tr><tr><td>MultiBookedeu</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.929 0.916 0.916 0.929 0.941</td><td>0.793 0.762 0.762 0.801 0.837</td><td>0.788 0.793 0.793 0.709 0.784 0.809</td><td>0.787 0.929 0.777 0.946 0.777 0.946 0.741 0.946</td><td></td><td>0.793 0.830 0.830 0.865</td><td>0.788 0.795 0.795 0.786</td><td>0.787 0.811 0.811 0.821</td></tr><tr><td>NoReC</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.835 0.817 0.817 0.832</td><td>0.550 0.607 0.607 0.612</td><td>0.555 0.626 0.626 0.609 0.609</td><td>0.933 0.552 0.835 0.611 0.842 0.611 0.842</td><td></td><td>0.785 0.550 0.652 0.652</td><td>0.803 0.555 0.624 0.624</td><td>0.789 0.552 0.633 0.633 0.632</td></tr><tr><td>OpeNERen</td><td>Baseline SELSP Brk7Bits BrkBased Absolute</td><td>0.812 0.906 0.705 0.926 0.909</td><td>0.613 0.753 0.777 0.786 0.753 0.738</td><td>0.628 0.789 0.705 0.750 0.762 0.745</td><td>0.821 0.606 0.826 0.765 0.906 0.682 0.705 0.931</td><td></td><td>0.627 0.623 0.753 0.777 0.814</td><td>0.660 0.639 0.789 0.705 0.741</td><td>0.630 0.765 0.682 0.761</td></tr><tr><td>OpeNERes</td><td>Relative Baseline SELSP Brk7Bits BrkBased</td><td>0.912 0.914 0.925 0.790 0.923 0.923</td><td>0.749 0.717 0.785 0.764 0.717 0.666 0.814 0.790 0.716 0.698 0.716 0.698</td><td>0.731 0.772 0.687 0.788 0.706 0.706</td><td>0.926 0.926 0.931 0.925 0.790 0.928</td><td>0.791 0.791 0.826 0.717 0.814</td><td>0.770</td><td>0.784 0.844 0.789 0.666 0.790 0.722</td><td>0.788 0.812 0.804 0.687 0.788 0.743</td></tr><tr><td>MARCen</td><td>Absolute Relative Baseline SELSP Brk7Bits</td><td>0.928 0.744 0.939 0.794 0.663 0.670 0.377 0.447</td><td>0.700 0.750 0.663 0.377</td><td>0.716 0.770 0.664 0.333</td><td>0.928 0.920 0.936 0.663 0.377</td><td></td><td>0.770 0.715 0.808 0.670 0.447</td><td>0.722 0.775 0.821 0.663 0.377</td><td>0.743 0.739 0.807 0.664 0.333</td></tr><tr><td></td><td>BrkBased Absolute Relative Baseline SELSP</td><td>0.663 0.662 0.662 0.655 0.660 0.664 0.660 0.658 0.619 0.619 0.384</td><td>0.663 0.662 0.660 0.660 0.619</td><td>0.662 0.656 0.661 0.659 0.619</td><td>0.662 0.669 0.665 0.667 0.619</td><td>0.673 0.671 0.619</td><td>0.670 0.662</td><td>0.662 0.669 0.665 0.667 0.619</td><td>0.665 0.670 0.664 0.666 0.619</td></tr><tr><td>MARCes</td><td>Brk7Bits BrkBased Absolute Relative Baseline</td><td>0.620 0.599 0.616 0.617 0.621</td><td>0.450 0.618 0.620 0.600 0.619</td><td>0.384 0.332 0.618 0.599 0.599 0.616 0.614 0.617 0.617</td><td>0.384 0.611 0.606 0.616 0.616</td><td></td><td>0.450 0.614 0.601 0.611 0.623</td><td>0.384 0.611 0.606 0.616 0.616</td><td>0.332 0.613 0.604 0.614 0.620</td></tr><tr><td>MARCfr</td><td>Brk7Bits BrkBased Absolute Relative</td><td>0.608 0.611 0.614 0.612 0.612</td><td>0.611 0.617 0.619 0.609</td><td>0.608 0.611 0.614 0.613 0.612 0.610</td><td>0.609 0.608 0.611 0.611 0.608 0.614 0.617</td><td></td><td>0.611 0.621 0.621 0.615 0.615</td><td>0.608 0.611 0.608 0.614 0.617</td><td>0.609 0.616 0.615 0.611</td></tr><tr><td>MARCde</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.657 0.660 0.665 0.652 0.649</td><td>0.606 0.653 0.658 0.662 0.651 0.660</td><td>0.612 0.657 0.660 0.665 0.652 0.649</td><td>0.605 0.654 0.657 0.658 0.660 0.651 0.651 0.657</td><td>0.680 0.670 0.665</td><td>0.653 0.675 0.673 0.660 0.662</td><td>0.657 0.680 0.670 0.665 0.657</td><td>0.616 0.654 0.677 0.671 0.661 0.659</td></tr></table>

Table 5: Ablation study isolating the contributions of Contrastive Learning and Syntax-Guided Attention: Performance of Contrastive-only MTL (contrastive loss only) Versus Syntax-only MTL (syntax bias only)
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Encoding</td><td colspan="4">Contrastive-only MTL</td><td colspan="4">Syntax-only MTL</td></tr><tr><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td><td>Acc</td><td>Prec</td><td>Rec</td><td>F1</td></tr><tr><td>Darmstadt_unis</td><td>Baseline Brk7Bits BrkBased</td><td>0.822 0.899 0.878</td><td>0.546 0.597 0.595</td><td>0.553 0.607 0.586</td><td>0.549 0.602 0.589</td><td>0.822 0.843 0.844</td><td>0.546 0.577 0.565</td><td>0.553 0.554 0.564</td><td>0.549 0.558 0.563</td></tr><tr><td>MPQA</td><td>Absolute Relative Baseline Brk7Bits BrkBased Absolute</td><td>0.856 0.876 0.882 0.875 0.895 0.875</td><td>0.578 0.568 0.592 0.581 0.751 0.707 0.580 0.731 0.708 0.711</td><td>0.570 0.586 0.723 0.596 0.587 0.718</td><td></td><td>0.867 0.867 0.882 0.875 0.898</td><td>0.578 0.578 0.751 0.576 0.709</td><td>0.581 0.581 0.707 0.596 0.663</td><td>0.579 0.579 0.723 0.585 0.677</td></tr><tr><td>MultiBookedca</td><td>Relative Baseline Brk7Bits BrkBased Absolute</td><td>0.883 0.886 0.900 0.900 0.882</td><td>0.693 0.643 0.691 0.691 0.679</td><td>0.766 0.698 0.641 0.713 0.713 0.700</td><td>0.737 0.695 0.642 0.698 0.698 0.686</td><td>0.902 0.855 0.886 0.868 0.868 0.868</td><td>0.738 0.561 0.643 0.673 0.673 0.668</td><td>0.717 0.592 0.641 0.693 0.693 0.692</td><td>0.726 0.573 0.642 0.681 0.681 0.676</td></tr><tr><td>MultiBookedeu</td><td>Relative Baseline Brk7Bits BrkBased Absolute</td><td>0.914 0.929 0.921 0.921 0.941</td><td>0.733 0.793 0.780 0.780 0.821</td><td>0.703 0.788 0.734 0.734 0.797</td><td>0.717 0.787 0.755 0.755 0.802</td><td>0.911 0.929 0.925 0.892 0.933</td><td>0.710 0.793 0.799 0.811 0.794</td><td>0.720 0.788 0.818 0.796 0.790</td><td>0.714 0.787 0.808 0.803 0.786</td></tr><tr><td>NoReC</td><td>Relative Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.925 0.835 0.826 0.826 0.835 0.837</td><td>0.799 0.550 0.609 0.609 0.601</td><td>0.818 0.555 0.622 0.622 0.599</td><td>0.808 0.552 0.612 0.612 0.600</td><td>0.933 0.835 0.809 0.809 0.832</td><td>0.807 0.550 0.585 0.585 0.645</td><td>0.730 0.555 0.608 0.608 0.615</td><td>0.763 0.552 0.595 0.595 0.625</td></tr><tr><td>OpeNERen</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.906 0.919 0.914 0.924</td><td>0.615 0.753 0.786 0.769 0.795</td><td>0.603 0.789 0.777 0.764 0.796</td><td>0.630 0.765 0.781 0.765 0.795</td><td>0.835 0.906 0.931 0.914 0.916</td><td>0.626 0.753 0.854 0.786 0.813</td><td>0.590 0.789 0.783 0.806 0.774</td><td>0.604 0.765 0.805 0.794 0.788</td></tr><tr><td>OpeNERes</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.912 0.925 0.923 0.923 0.898</td><td>0.793 0.717 0.736 0.736 0.718</td><td>0.750 0.666 0.724 0.724 0.855</td><td>0.765 0.687 0.727 0.727 0.741</td><td>0.914 0.925 0.925 0.925 0.928</td><td>0.772 0.717 0.736 0.736 0.607</td><td>0.778 0.666 0.740 0.740 0.600</td><td>0.774 0.687 0.738 0.738 0.603</td></tr><tr><td>MARCen</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.939 0.663 0.660 0.662 0.664</td><td>0.792 0.670 0.658 0.663 0.663</td><td>0.723 0.663 0.660 0.662 0.664</td><td>0.753 0.664 0.658 0.658 0.661</td><td>0.920 0.663 0.674 0.666 0.665</td><td>0.808 0.670 0.671 0.663 0.666</td><td>0.682 0.663 0.667 0.666 0.665</td><td>0.720 0.664 0.669 0.663 0.665</td></tr><tr><td>MARCes</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.666 0.619 0.606 0.609 0.614</td><td>0.663 0.619 0.624 0.619 0.619</td><td>0.666 0.619 0.606 0.609 0.614</td><td>0.663 0.619 0.608 0.610 0.614</td><td>0.664 0.619 0.608 0.607 0.611</td><td>0.663 0.619 0.610 0.610 0.617</td><td>0.664 0.619 0.608 0.607 0.611</td><td>0.661 0.619 0.608 0.607 0.612</td></tr><tr><td>MARCfr</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.604 0.608 0.614 0.609 0.605 0.604</td><td>0.609 0.611 0.622 0.624 0.616 0.620</td><td>0.604 0.608 0.614 0.609 0.605 0.604</td><td>0.605 0.609 0.616 0.613 0.609 0.608</td><td>0.616 0.608 0.614 0.608 0.612 0.603</td><td>0.612 0.611 0.618 0.606 0.614 0.607</td><td>0.616 0.608 0.614 0.608 0.612 0.603</td><td>0.614 0.609 0.614 0.605 0.608 0.602</td></tr><tr><td>MARCde</td><td>Baseline Brk7Bits BrkBased Absolute Relative</td><td>0.657 0.671 0.668 0.675 0.676</td><td>0.653 0.667 0.668 0.674 0.674</td><td>0.657 0.671 0.668 0.675 0.676</td><td>0.654 0.665 0.664 0.671 0.672</td><td>0.657 0.672 0.672 0.660 0.661</td><td>0.653 0.670 0.672 0.658 0.667</td><td>0.657 0.672 0.672 0.660 0.661</td><td>0.654 0.669 0.668 0.657 0.663</td></tr></table>

Table 6: Computational cost comparison between explicit (Standard MTL) and implicit (Contrastive MTL) syntactic augmentation strategies, reporting training and evaluation times in seconds computed on NVIDIA RTX 6000 GPU. For each MTL variant, we report the mean over the four encodings (Brk7Bits, BrkBased, Absolute, Relative). SELSP is an external rule-based pipeline and is therefore not trained; its evaluation time is only reported for the English and Spanish datasets.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Standard-MTL</td><td colspan="2">Contrastive-MTL</td></tr><tr><td>Train (s)</td><td>Eval (s)</td><td>Train (s)</td><td>Eval (s)</td></tr><tr><td rowspan="3">Darmstadt_unis</td><td>Baseline</td><td>206.061</td><td>0.678</td><td>206.061</td><td>0.678</td></tr><tr><td>SELSP</td><td></td><td>5.104</td><td></td><td>5.104</td></tr><tr><td>Ours</td><td>325.681</td><td>0.686</td><td>282.294</td><td>0.674</td></tr><tr><td rowspan="3">MPQA</td><td>Baseline</td><td>486.676</td><td>1.930</td><td>486.676</td><td>1.930</td></tr><tr><td>SELSP</td><td></td><td>17.056</td><td></td><td>17.056</td></tr><tr><td>Ours</td><td>420.828</td><td>1.848</td><td>559.545</td><td>1.930</td></tr><tr><td rowspan="2">MultiBookedca</td><td>Baseline</td><td>516.558</td><td>2.100</td><td>516.558</td><td>2.100</td></tr><tr><td>Ours</td><td>610.246</td><td>2.024</td><td>636.596</td><td>2.102</td></tr><tr><td rowspan="2">MultiBookedeu</td><td>Baseline</td><td>263.059</td><td>1.795</td><td>263.059</td><td>1.795</td></tr><tr><td>Ours</td><td>651.633</td><td>1.739</td><td>663.974</td><td>1.771</td></tr><tr><td rowspan="2">NoReC</td><td>Baseline</td><td>4699.542</td><td>4.801</td><td>4699.542</td><td>4.801</td></tr><tr><td>Ours</td><td>3026.135</td><td>4.680</td><td>3688.326</td><td>4.786</td></tr><tr><td rowspan="3">OpeNERen</td><td>Baseline</td><td>816.542</td><td>3.045</td><td>816.542</td><td>3.045</td></tr><tr><td>SELSP</td><td>1005.175</td><td>6.519</td><td></td><td>6.519</td></tr><tr><td>Ours</td><td></td><td>3.254</td><td>693.360</td><td>3.038</td></tr><tr><td rowspan="3">OpeNERes</td><td>Baseline SELSP</td><td>459.175</td><td>2.699</td><td>459.175</td><td>2.699</td></tr><tr><td></td><td></td><td>4.456</td><td></td><td>4.456</td></tr><tr><td>Ours</td><td>653.489</td><td>2.797</td><td>850.568</td><td>2.744</td></tr><tr><td rowspan="3">MARCen</td><td>Baseline</td><td>27756.875</td><td>20.705</td><td>27756.875</td><td>20.705</td></tr><tr><td>SELSP</td><td></td><td>76.638</td><td></td><td>76.638</td></tr><tr><td>Ours</td><td>26601.118</td><td>20.364</td><td>28436.375</td><td>19.840</td></tr><tr><td rowspan="3">MARCes</td><td>Baseline SELSP</td><td>26643.287</td><td>20.829</td><td>26643.287</td><td>20.829</td></tr><tr><td>Ours</td><td>26106.984</td><td>70.167 20.334</td><td>28490.370</td><td>70.167 20.754</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">MARCfr</td><td>Baseline Ours</td><td>17922.380 18737.200</td><td>25.499 25.378</td><td>17922.380 24623.209</td><td>25.499 25.194</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">MARCde</td><td>Baseline</td><td>23765.453</td><td>26.120</td><td>23765.453</td><td>26.120</td></tr><tr><td>Ours</td><td>25088.874</td><td>26.456</td><td>24485.178</td><td>26.328</td></tr></table>

![](images/8b8ea76aa40d1995d65ddd46b53d60439ffdffe1ef05d6784ef6c5616471387c.jpg)  
Figure 3: Relative-encoding F1 for Baseline, SELSP, Standard-MTL and Contrastive-MTL across all eleven datasets. SELSP is available only for the six English and Spanish datasets; the white region with N/A marks the five datasets for which SELSP is not applicable.

![](images/1a58d80a1aedd6e2a8b93ce2c7d6678a905d437ee579e597da8248f710fdb67d.jpg)  
Figure 4: Encoding-induced ∆F1 relative to the syntax-naive baseline, for Standard-MTL and Contrastive-MTL.

• Contrastive-only: Removes the dependencyrelation bias ${ \bf b } _ { r _ { j } } ^ { \mathrm { r e l } }$ from the attention pooling, replacing the syntax-guided attention with plain learned attention $\mathbf { a } = \mathrm { s o f t m a x } ( \mathbf { W } _ { a } \mathbf { H } )$ Both contrastive losses $( \mathcal { L } _ { \mathrm { s y n } }$ and Lsent-contrast) and the uncertainty-based loss weighting are retained. This isolates the contribution of the syntax-guided pooling.

• Syntax-only: Retains the syntax-guided attention pooling, but removes the projection head, both contrastive terms, and the learnable loss weights $\sigma _ { \mathrm { s y n } } , \sigma _ { \mathrm { s e n t } }$ , so the model is trained with $\mathcal { L } _ { \mathrm { s e n t } }$ only. This isolates the contribution of the contrastive objectives.

Table 5 presents the results of an ablation study comparing two configurations: a model with the contrastive objective removed ("Syntax-only MTL") and a model with the syntactic encoding removed ("Contrastive-only MTL"), evaluated against a shared baseline that uses neither component.

The Contrastive-only MTL variant exhibits the most consistent pattern: across most datasets and encodings, it matches or exceeds the shared baseline, and it frequently attains the best or tiedbest F1. This trend is especially clear on Darmstadt\_unis, MultiBooked\_ca, NoReC, OpeNERes, and $\mathbf { M A R C } _ { \mathrm { d e } } ,$ where the contrastive-only model broadly improves over the baseline. On MPQA, by contrast, it underperforms the baseline on three of the four encodings (Brk7Bits, BrkBased, and Relative), recovering above it only with the Absolute encoding. On $\mathrm { M u l t i B o o k e d } _ { \mathrm { e u } }$ and $\mathbf { M A R C } _ { \mathrm { e s } }$ , it still reaches best or tied-best scores in some settings, although these gains are less uniform. These results indicate that contrastive learning alone provides a robust and transferable benefit on many, though not

![](images/a3dbd2d9adeb29df789b32601c50a7ee39e39249950a6f165e03bb1b31874682.jpg)  
Figure 5: Inference time comparison (log scale) for Baseline, SELSP, Standard-MTL and Contrastive-MTL.

all, datasets.

The Syntax-only MTL variant shows a more mixed pattern. It yields notable improvements on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ and $\mathbf { M A R C } _ { \mathrm { e n } }$ , and matches the best contrastive-only result on $\mathrm { M u l t i B o o k e d } _ { \mathrm { e u } }$ and $\mathbf { M A R C } _ { \mathrm { e s } }$ in selected encodings. However, it often degrades performance relative to the baseline, particularly on $\mathbf { M A R C } _ { \mathrm { e s } } , \mathbf { M P Q A }$ , and $\mathbf { M A R C } _ { \mathrm { f r } }$ , as well as the Absolute encoding on $\mathrm { O p e N E R } _ { \mathrm { e s } } .$ Thus, syntax-guided attention alone does not reliably improve performance.

A direct comparison between the two ablated modules shows that Contrastive-only MTL achieves the highest single-component scores in more settings than Syntax-only MTL (seven datasets versus two, with two tied), indicating that the contrastive objective is the more robust individual component. However, neither ablation reproduces the full model's performance: taking the best encoding per dataset, the full Contrastive-MTL model outperforms both ablations on ten of the eleven datasets. This indicates that the syntax bias, while not reliably beneficial in isolation, provides complementary gains when combined with the contrastive objectives.

## 7 Mechanism Analysis

The ablation in Table 5 establishes that the contrastive objective drives downstream performance, but it leaves open the mechanism by which it does so. We examine that mechanism on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ as a case study from three angles: (i) which syntactic roles the trained model attends to at inference time (§7.1); (ii) what syntactic signal the training-time attention bias learned and subsequently discarded (§7.2); and (iii) whether the encoder internalizes syntactic structure, as measured by a linear probe on frozen representations (§7.3).

## 7.1 Attention Analysis

Figure 6 plots the mean inference-time attention weight assigned to each dependency relation, aggregated over all annotated test tokens as reported in Table 7. The highest attention is assigned to advmod $( \bar { a } = 0 . 0 7 6 )$ and amod $( \bar { a } = 0 . 0 6 6 )$ —the two relations that carry intensifiers (“very good") and sentiment-bearing adjectives (“clean rooms"). The neg ranks above the median $( \bar { a } = 0 . 0 5 4 )$ despite comprising only 63 test tokens. At the bottom of the distribution, the model assigns minimal attention to det $( \bar { a } = 0 . 0 2 4 )$ and poss $( \bar { a } = 0 . 0 2 8 )$ —the two function-word relations that carry no sentiment content. This is the pattern that rule-based sentiment systems predict: at inference time the model concentrates its pooling mass on the same syntactic roles that rule-based systems have hand-engineered for polarity composition. It is an emergent property of the encoder's learned representation under the contrastive objective—not of the syntax-guided attention bias, which is discarded at inference.

## 7.2 Learned Relation Bias

The right panel of Figure 6 shows the learned relation bias ${ \bf b } _ { r } ^ { \mathrm { r e l } }$ for each relation; the full per-relation values are reported in Table 7. Recall (Section 4) that this bias is present only during training and is discarded at inference; its magnitude and sign therefore show which relations the model wanted to emphasize during training, providing a direct probe of the syntactic knowledge the bias pathway absorbed.

The pattern is coherent with the attention analysis: advmod receives the largest positive bias (+0.67), consistent with intensification being a reliable cue for sentiment polarity, and neg receives a positive bias (+0.29), consistent with the model learning to upweight negation markers. amod and det both receive near-zero bias $( - 0 . 0 6 \mathrm { a n d } - 0 . 0 7 )$ indicating that the model found these relations uninformative as a bias signal—consistent with the fact that the polarity content of these relations is already carried by the token itself and does not need to be amplified at the pooling stage.

![](images/c4c20b76f69c6bba103116e09ed4f7c179e63b54b51f41e8b4ff0bf2e59ea545.jpg)  
Figure 6: Inference-time attention over UD relations on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ (Contrastive-MTL, Relative encoding, $n \geq 2 0 )$ Left panel: mean softmax attention weight per relation, with ±1 std across tokens; relations sorted by mean. Right panel: the learned training-time relation bias (Section 7.2), which is discarded at inference. Sentiment-relevant relations advmod, amod, and neg receive high attention; function-word relations det and poss receive low attention.

Table 7: Inference-time attention and training-time learned bias per UD relation on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ (Contrastive-MTL, Relative encoding, $n \geq 2 0 )$ 1
<table><tr><td>Relation</td><td>n</td><td>Attention</td><td>Bias</td></tr><tr><td>advmod</td><td>320</td><td>0.0762</td><td>+0.673</td></tr><tr><td>amod</td><td>375</td><td>0.0662</td><td>-0.060</td></tr><tr><td>dep</td><td>40</td><td>0.0580</td><td>+0.660</td></tr><tr><td>prep</td><td>31</td><td>0.0565</td><td>-0.034</td></tr><tr><td>cop</td><td>265</td><td>0.0537</td><td>-0.107</td></tr><tr><td>neg</td><td>63</td><td>0.0536</td><td>+0.291</td></tr><tr><td>mark</td><td>61</td><td>0.0531</td><td>+1.187</td></tr><tr><td>nsubj</td><td>527</td><td>0.0480</td><td>+1.056</td></tr><tr><td>nsubjpass</td><td>46</td><td>0.0354</td><td>-0.546</td></tr><tr><td>num</td><td>43</td><td>0.0337</td><td>-0.219</td></tr><tr><td>punct</td><td>76</td><td>0.0334</td><td>-1.701</td></tr><tr><td>aux</td><td>188</td><td>0.0324</td><td>-1.664</td></tr><tr><td>auxpass</td><td>50</td><td>0.0318</td><td>-0.764</td></tr><tr><td>nn</td><td>220</td><td>0.0316</td><td>+1.343</td></tr><tr><td>expl</td><td>21</td><td>0.0305</td><td>-0.120</td></tr><tr><td>poss</td><td>54</td><td>0.0279</td><td>-1.157</td></tr><tr><td>det</td><td>586</td><td>0.0238</td><td>-0.071</td></tr></table>

## 7.3 Representation Probing

To test whether the encoder itself internalizes syntax, we freeze the encoder and train a fresh linear probe on the frozen token representations to predict the gold UD relation of each annotated token. The probe is a single linear layer (d=1024 input, one output per relation class) trained for 30 epochs with AdamW (lr $1 0 ^ { - 3 } )$ , batch size 1024, and no regularization; it is trained on the training split and evaluated on the test split.

Table 8 reports per-relation accuracy for four models: Baseline (no auxiliary loss), Contrastive-MTL (attention bias + both contrastive losses), Contrastive-only MTL (both contrastive losses, no bias), and Syntax-only MTL (attention bias, no contrastive losses). Three findings stand out.

First, the baseline encoder is not learning syntax. It scores near or above 0.79 on det and punct relations that are predictable from the token itself $( ^ { \circ \circ } \mathrm { t h e } ^ { \circ \prime } , \ " \mathrm { a } ^ { \circ \prime } , \stackrel { \dots \wedge } { \cdot } )$ —and near chance on contextdependent relations (nsubj 0.69, cop 0.28, neg 0.08, advmod 0.37). The baseline's overall accuracy of 0.462 is therefore a lexical prior, not a syntactic signal.

Table 8: Per-relation linear-probe accuracy on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ (Relative encoding. Lexical relations are largely predictable from the token itself; context-dependent relations require the surrounding sentence. Contrastive-MTL uses the attention bias and both contrastive losses; Contrastive-only MTL drops the bias; Syntax-only MTL drops the contrastive losses. Best per row in bold; relations with n < 20 omitted.
<table><tr><td>Relation</td><td>n</td><td>Baseline</td><td>Contrastive-MTL</td><td>Contrastive-only MTL</td><td>Syntax-only MTL</td></tr><tr><td colspan="6">Lexical relations (predictable from token identity)</td></tr><tr><td>det</td><td>586</td><td>0.804</td><td>0.988</td><td>0.988</td><td>0.930</td></tr><tr><td>punct</td><td>76</td><td>0.790</td><td>0.974</td><td>0.961</td><td>0.263</td></tr><tr><td>amod</td><td>375</td><td>0.627</td><td>0.880</td><td>0.899</td><td>0.307</td></tr><tr><td>nn</td><td>220</td><td>0.218</td><td>0.832</td><td>0.832</td><td>0.077</td></tr><tr><td colspan="6">Context-dependent relations (require the sentence)</td></tr><tr><td>nsubj</td><td>527</td><td>0.685</td><td>0.941</td><td>0.920</td><td>0.368</td></tr><tr><td>advmod</td><td>320</td><td>0.369</td><td>0.906</td><td>0.909</td><td>0.022</td></tr><tr><td>cop</td><td>265</td><td>0.283</td><td>0.955</td><td>0.959</td><td>0.008</td></tr><tr><td>aux</td><td>188</td><td>0.160</td><td>0.947</td><td>0.957</td><td>0.016</td></tr><tr><td>neg</td><td>63</td><td>0.079</td><td>0.984</td><td>0.984</td><td>0.000</td></tr><tr><td>mark</td><td>61</td><td>0.033</td><td>0.918</td><td>0.934</td><td>0.016</td></tr><tr><td>auxpass</td><td>50</td><td>0.000</td><td>0.880</td><td>0.920</td><td>0.000</td></tr><tr><td>nsubjpass</td><td>46</td><td>0.000</td><td>0.174</td><td>0.783</td><td>0.000</td></tr><tr><td>Overall</td><td>3,042</td><td>0.462</td><td>0.904</td><td>0.917</td><td>0.298</td></tr></table>

Second, the contrastive objective creates syntactic structure. Both models trained with the contrastive losses—Contrastive-MTL and Contrastiveonly MTL—raise accuracy on every contextdependent relation to above 0.90 (nsubj 0.92–0.94, advmod 0.91, cop 0.96, neg 0.98), a jump of between +0.24 and +0.91 absolute over the baseline. On relations that are trivially predictable from token identity (det, punct), the gain is small or zero, as expected. The contrastive objective specifically internalizes the syntactic distinctions that are not lexically derivable.

Third, the attention bias without the contrastive objective degrades the encoder's syntactic encoding, even though the ablation in Table 5 shows that its downstream F1 remains competitive on $\mathrm { O p e N E R } _ { \mathrm { e n } } .$ Syntax-only MTL achieves probe accuracy of 0.298, well below the baseline's 0.462. The mechanism is a shortcut: with relation labels available to the pooling layer at each training step, the sentiment head can access relation information directly, and the encoder has no incentive to encode relations itself; sentiment fine-tuning then partially erases the pretrained syntactic features. The result is a model whose pooling layer knows which relations matter for sentiment but whose encoder knows less syntax than a model that was never trained on syntax at all. This is the same shortcut-learning phenomenon documented in the NLP literature (Gururangan et al., 2018; Niven and Kao, 2019).

The three signals converge on a coherent account: Contrastive-MTL on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ produces an encoder that (i) attains high linear separability by dependency relation on the relations that matter for sentiment, (ii) prioritizes those same relations at inference time through attention, and (iii) treats function-word relations as low-attention, low-bias background. Taken together, these three findings support the mechanism proposed in Section 4: the contrastive objective shapes the encoder to encode the syntactic distinctions that matter for sentiment.

## 8 Conclusion

In this work, we explored two multi-task learning strategies for integrating syntactic information into neural polarity classification models without incurring the computational overhead of external parsers at inference time. Our proposed approaches—Standard-MTL, which jointly learns dependency parsing as sequence labeling alongside polarity classification, and Contrastive-MTL, which injects syntactic guidance implicitly through a contrastive auxiliary objective and syntax-biased attention pooling—were evaluated on eleven multilingual benchmark datasets across four encoding schemes.

With the relative encoding, Contrastive-MTL outperforms the syntactically-naive baseline on 10 out of 11 datasets and Standard-MTL on the same proportion, with the most pronounced gains on OpeNERes (+0.120 F1) and MultiBookedca (+0.084 F1). Notably, with the same encoding, Contrastive-MTL also outperforms SELSP—an external rule-based pipeline that requires full parsing at inference time—on all six datasets where SELSP is applicable.

A mechanism analysis on $\mathrm { O p e N E R } _ { \mathrm { e n } }$ clarifies the source of these gains. Linear probes on frozen encoder representations show that the contrastive objective raises probe accuracy on dependency relations from 0.46 (baseline) to 0.92, with the gain concentrated on context-dependent relations (nsubj, advmod, cop, neg) rather than on relations predictable from token identity alone (det, punct). Attention analysis shows that inference-time pooling mass on the Contrastive-MTL model concentrates on precisely these relations and minimizes functionword relations, while the learned training-time bias on advmod and neg is positive and on det is near zero. These complementary signals support the interpretation that the contrastive objective shapes the encoder to encode the syntactic distinctions that matter for sentiment, and the sentiment head at inference reweights tokens by those distinctions.

Finally, both MTL variants discard the auxiliary syntactic components after training, so inference times remain nearly identical to the baseline, whereas SELSP is 1.6–9.2× slower at evaluation due to parsing overhead. This design preserves the efficiency of single-task models while offering a principled alternative to pipeline architectures that rely on full parse tree generation and hand-crafted compositional rules.

## 9 Limitations and Future Directions

Despite the promising results, we acknowledge the limitations of our study. First, the MTL models rely on syntactic annotations produced by an offthe-shelf biaffine dependency parser (Dozat and Manning, 2017) during training, so any systematic errors or biases in the parser's output may propagate to the MTL framework; the contrastive objective could, in principle, learn to cluster tokens around incorrect relation labels, potentially harming generalization.

These limitations open several avenues for future research. Extending the framework to finegrained and aspect-based sentiment analysis is a natural next step, where token-level contrastive objectives aligned with aspect terms and opinion targets could enable the model to learn richer associations between UD relations (e.g., nsubj, obj, amod) and specific sentiment expressions, thereby handling multi-aspect reviews with contrasting polarities. A second direction is to explore alternative syntactic representations—constituency parses, phrase-structure trees, or lightweight syntactic features such as negation scope markers and intensifier chains—under the same contrastive framework, in order to identify the optimal trade-off between linguistic richness and learning efficiency.

## 10 Acknowledgments

We acknowledge the European Research Council (ERC), which has funded this research under the Horizon Europe research and innovation programme (SALSA, grant agreement No 101100615), GAP (PID2022-139308OA-I00) funded by MI-CIU/AEI/10.13039/501100011033/ and by ERDF, EU; LATCHING (PID2023-147129OB-C21) funded by MICIU/AEI/10.13039/501100011033 and ERDF (EU), Ministry for Digital Transformation and Civil Service and “NextGenerationEU" PRTR under grant TSI-100925-2023-1, Xunta de Galicia (ED431C 2024/02), and Centro de Investigación de Galicia “CITIC", funded by the Xunta de Galicia through the collaboration agreement between the Consellería de Cultura, Educación, Formación Profesional e Universidades and the Galician universities for the reinforcement of the research centres of the Galician University System (CIGUS). Furthermore, this research was supported by the International, Interdisciplinary and Intersectoral Information and Communications Technology PhD programme (3-i ICT) granted to CITIC and supported by the European Union through the Horizon 2020 research and innovation programme under a Marie Skłodowska-Curie agreement (H2020-MSCA-COFUND), GA 101034261.

## References

Rodrigo Agerri, Montse Cuadros, Sean Gaines, and German Rigau. 2013. OpeNER: Open Polarity Enhanced Named Entity Recognition. Procesamiento del Lenguaje Natural, 51(0):215–218.

Jeremy Barnes, Toni Badia, and Patrik Lambert. 2018. MultiBooked: A Corpus of Basque and Catalan Hotel Reviews Annotated for Aspect-level Sentiment Classification. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), pages 656–660, Miyazaki, Japan. European Language Resources Association (ELRA).

Jeremy Barnes, Laura Oberlaender, Enrica Troiano, Andrey Kutuzov, Jan Buchmann, Rodrigo Agerri, Lilja Øvrelid, and Erik Velldal. 2022. SemEval 2022 task 10: Structured sentiment analysis. In Proceedings of the 16th International Workshop on Semantic Evaluation (SemEval-2022), pages 1280–1295, Seattle, United States. Association for Computational Linguistics.

Julian Brooke, Milan Tofiloski, and Maite Taboada. 2009. Cross-linguistic sentiment analysis: From English to Spanish. In Proceedings of the International Conference RANLP-2009, pages 50–54, Borovets, Bulgaria. Association for Computational Linguistics.

Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th international conference on Machine learning, pages 160–167.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2020. Unsupervised cross-lingual representation learning at scale. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 8440–8451.

Jingfeng Cui, Zhaoxia Wang, Seng-Beng Ho, and Erik Cambria. 2023. Survey on sentiment analysis: evolution of research methods and topics. Artificial Intelligence Review, pages 1–42.

Kerstin Denecke and Daniel Reichenpfader. 2023. Sentiment analysis of clinical narratives: a scoping review. Journal of Biomedical Informatics, 140:104336.

Timothy Dozat and Christopher D. Manning. 2017. Deep biaffine attention for neural dependency parsing. In International Conference on Learning Representations.

Xianghua Fu, Jingying Yang, Jianqiang Li, Min Fang, and Huihui Wang. 2018. Lexicon-enhanced 1stm with attention for general sentiment analysis. IEEE Access, 6:71884–71891.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference

on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

John Giorgi, Osvald Nitski, Bo Wang, and Gary Bader. 2021. DeCLUTR: Deep contrastive learning for unsupervised textual representations. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 879–895, Online. Association for Computational Linguistics.

Carlos Gómez-Rodríguez, Muhammad Imran, David Vilares, Elena Solera, and Olga Kellert. 2024. Dancing in the syntax forest: fast, accurate and explainable sentiment analysis with SALSA. In SEPLN-CEDI-PD 2024. Seminar of the Spanish Society for Natural Language Processing: Projects and System Demonstrations, volume 3729 of CEUR Workshop Proceedings, pages 12–17, A Coruña, Spain.

Carlos Gómez-Rodríguez, Diego Roca, and David Vilares. 2023. 4 and 7-bit labeling for projective and non-projective dependency trees. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6375–6384, Singapore. Association for Computational Linguistics.

Carlos Gómez-Rodríguez and David Vilares. 2018. Constituent parsing as sequence labeling. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1314–1324.

Suchin Gururangan, Swabha Swayamdipta, Omer Levy, Roy Schwartz, Samuel R. Bowman, and Noah A. Smith. 2018. Annotation artifacts in natural language inference data. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 107–112, New Orleans, Louisiana. Association for Computational Linguistics.

Clayton Hutto and Eric Gilbert. 2014. Vader: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings of the international AAAI conference on web and social media, volume 8, pages 216–225.

Muhammad Imran, Ana Ezquerro, Carlos Gómez-Rodríguez, Anders Søgaard, and David Vilares. 2026a. Structured sentiment analysis using sequence labeling as dependency graph parsing. arXiv preprint arXiv:2610.11695.

Muhammad Imran, Olga Kellert, and Carlos Gómez-Rodríguez. 2026b. A syntax-injected approach for faster and more accurate sentiment analysis. PeerJ Computer Science, 12:e3519.

Muhammad Imran, Olga Zamaraeva, and Carlos Gómez-Rodríguez. 2026c. Synner: syntax-infused named entity recognition in the biomedical domain. JAMIA open, 9(1):ooaf149.

João Maria Janeiro, Benjamin Piwowarski, Patrick Gallinari, and Loic Barrault. 2025. MEXMA: Tokenlevel objectives improve sentence representations. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 23960–23995, Vienna, Austria. Association for Computational Linguistics.

Olga Kellert, Muhammad Imran, Nicholas Matlis, Mahmud Uz Zaman, and Carlos Gómez-Rodríguez. 2025a. Evaluating compositional approaches for focus and sentiment analysis. In Intelligent Computing-Proceedings of the Computing Conference, pages 409–425. Springer.

Olga Kellert, Nemika Tyagi, Muhammad Imran, Nelvin Licona-Guevara, and Carlos Gómez-Rodríguez. 2025b. Parsing the switch: LLM-based UD annotation for complex code-switched and low-resource languages. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 15934– 15949, Suzhou, China. Association for Computational Linguistics.

Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 7482–7491.

Phillip Keung, Yichao Lu, György Szarvas, and Noah A Smith. 2020. The multilingual amazon reviews corpus. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 4563–4568.

Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in neural information processing systems, 33:18661–18673.

Dan Kondratyuk and Milan Straka. 2019. 75 languages, 1 model: Parsing universal dependencies universally. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pages 2779–2795.

Zicheng Lin, Tian Liang, Jiahao Xu, Qiuzhi Lin, Xing Wang, Ruilin Luo, Chufan Shi, Siheng Li, Yujiu Yang, and Zhaopeng Tu. 2024. Critical tokens matter: Token-level contrastive estimation enhances llm's reasoning capability. arXiv preprint arXiv:2411.19943.

Timothy Niven and Hung-Yu Kao. 2019. Probing neural network comprehension of natural language arguments. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4658–4664, Florence, Italy. Association for Computational Linguistics.

Soujanya Poria, Erik Cambria, Grégoire Winterstein, and Guang-Bin Huang. 2014. Sentic patterns: Dependency-based rules for concept-level sentiment analysis. Knowledge-Based Systems, 69:45–63.

Peng Qi, Yuhao Zhang, Yuhui Zhang, Jason Bolton, and Christopher D Manning. 2020. Stanza: A python natural language processing toolkit for many human languages. In Proceedings of the 58th annual meeting of the association for computational linguistics: system demonstrations, pages 101–108.

Diego Roca, David Vilares, and Carlos Gómez-Rodríguez. 2022. A system for constituent and dependency tree linearization. In Proceedings of V XoveTIC Conference, XoveTIC 2022, pages 146–147.

Prakash Kumar Singh and Sanchita Paul. 2021. Deep learning approach for negation handling in sentiment analysis. IEEE Access, 9:102579–102592.

Michalina Strzyz, David Vilares, and Carlos Gómez-Rodríguez. 2019. Viable dependency parsing as sequence labeling. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 717–723.

Swabha Swayamdipta, Sam Thomson, Kenton Lee, Luke Zettlemoyer, Chris Dyer, and Noah A Smith. 2018. Syntactic scaffolds for semantic structures. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 3772–3782.

Maite Taboada, Julian Brooke, Milan Tofiloski, Kimberly Voll, and Manfred Stede. 2011. Lexicon-based methods for sentiment analysis. Computational linguistics, 37(2):267–307.

Mike Thelwall, Kevan Buckley, Georgios Paltoglou, Di Cai, and Arvid Kappas. 2010. Sentiment strength detection in short informal text. Journal of the American Society for Information Science and Technology, 61(12):2544–2558.

Cigdem Toprak, Niklas Jakob, and Iryna Gurevych. 2010. Sentence and Expression Level Annotation of Opinions in User-Generated Discourse. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, pages 575–584, Uppsala, Sweden. Association for Computational Linguistics.

Rob van der Goot, Ahmet Üstün, Alan Ramponi, Ibrahim Sharaf, and Barbara Plank. 2021. Massive choice, ample tasks (MaChAmp): A toolkit for multitask learning in NLP. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 176–197, Online. Association for Computational Linguistics.

Erik Velldal, Lilja Øvrelid, Eivind Alexander Bergem, Cathrine Stadsnes, Samia Touileb, and Fredrik Jørgensen. 2018. NoReC: The Norwegian Review Corpus. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), pages 4186–4191, Miyazaki, Japan. European Language Resources Association (ELRA).

David Vilares, Carlos Gómez-Rodríguez, and Miguel A. Alonso. 2017. Universal, unsupervised (rule-based), uncovered sentiment analysis. Knowledge-Based Systems, 118:45–55.

Janyce Wiebe, Theresa Wilson, and Claire Cardie. 2005. Annotating expressions of opinions and emotions in language. Language resources and evaluation, 39:165-210.

Daniel Zeman, Joakim Nivre, Mitchell Abrams, Noëmi Aepli, Željko Agić, Lars Ahrenberg, and 1 others. 2023. Universal dependencies 2.12. In LINDAT/CLARIAH-CZ digital library at the Institute of Formal and Applied Linguistics (ÚFAL), Faculty of Mathematics and Physics, Charles University.

Yuan Zhang and David Weiss. 2016. Stack-propagation: Improved representation learning for syntax. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1557–1566.