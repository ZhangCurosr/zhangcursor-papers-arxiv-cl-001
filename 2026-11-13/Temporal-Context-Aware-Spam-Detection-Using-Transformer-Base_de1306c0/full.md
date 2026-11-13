# Temporal Context-Aware Spam Detection Using Transformer-Based NLP Models

Fariz Zeynalov University of Chicago zeynalov@uchicago.edu

Razim Mammadli Duke University rm620@duke.edu

## Abstract

Spam detection is usually treated as singlemessage classification, ignoring behavioral patterns that emerge across sequences of messages from the same sender. We study outbound SMS spam detection at the carrier level, where the goal is to predict at send time whether a carrier will flag a message as spam. Using a proprietary production database of 43,741 outbound SMS messages with UTC transmission timestamps and carrier delivery statuses, we report three sets of results. On an 8,749- message held-out test set, a TF-IDF + logistic regression baseline reaches AUC-ROC 0.8721, and a text-only DeBERTa-v3-base classifier reaches AUC-ROC 0.9011 and F1 0.8412. A deployed DistilBERT model augmented with inter-message temporal features reaches accuracy 0.8238, spam-class F1 0.8382, and recall 0.9128 at threshold 0.5 on a separate, classbalanced 20,000-message production evaluation set; these results are not directly comparable with the held-out test results. We also describe TemporalBertSpam, an architecture that encodes each sender’s message history with a causal Transformer, and outline the same-split evaluation needed to measure its contribution.

## 1 Introduction

SMS carriers run automated filters that flag outbound messages as spam based on content, sender reputation, and behavioral signals. These carrierside filters differ from classic spam classifiers, which label inbound unsolicited messages: here the task is to predict, at send time, whether the carrier network will block a message. This framing matters in practice, because a blocked message is never delivered and false positives harm legitimate senders.

Most prior NLP work on spam detection classifies single messages with transformer-based text encoders (Jamal et al., 2024; Asliyuksek et al., 2025). These approaches treat each message independently. However, spam campaigns often vary their wording (spintax) to evade per-message text filters while still showing detectable behavior: highvolume bursts, regular intervals between messages, and many recipients reached from a single phone number in a short time.

This paper studies whether information about the temporal sequence of a sender’s messages can complement message content. We frame three research questions:

• RQ1: Do inter-message temporal features (time delta ∆t and log(1 + ∆t)) improve detection over a text-only transformer baseline?

• RQ2: Does sequence-level modeling of sender histories improve detection over temporal features alone?

• RQ3: How do message content type (URL presence, urgency language) and sender volume relate to detection performance?

This paper reports baseline results on a heldout test set and production results for a deployed temporal-feature model. Answering RQ1 and RQ2 requires evaluating all systems on a single timeordered split, and RQ3 requires a subgroup analysis on that same split; we leave both to future work (see Limitations). Our dataset contains permessage timestamps, which are absent from common benchmarks such as the SMS Spam Collection (Almeida et al., 2011).

## 2 Related Work

Transformer-based text classifiers for spam Devlin et al. (2019) introduced BERT, which became the dominant paradigm for NLP classification. Sanh et al. (2019) distilled BERT into Distil-BERT, which is 40% smaller while retaining 97% of BERT’s language understanding performance, making it well suited to high-throughput inference. Jamal et al. (2024) fine-tune DistilBERT and RoBERTa on a three-class phishing/spam/ham task, reporting 97.1% and 99.0% test accuracy respectively. Their labels come from human-annotated corpora; our labels reflect network-level carrier decisions on outbound messages.

Combining text with non-textual features Asliyuksek et al. (2025) combine DistilBERT representations with static structural features of each email and report 99.62% accuracy on a combined corpus of 81,586 emails from seven datasets. Their non-textual features are computed per message, with no cross-message temporal component. Our approach instead targets behavioral patterns across sequences of messages from the same sender.

Behavioral and temporal signals in spam detection Network and security research has long used sender behavior rather than content. Ramachandran et al. (2007) classify email senders by their sending patterns across target domains, and Hao et al. (2009) detect email spammers with lightweight spatio-temporal network-level features. For SMS, Xu et al. (2012) detect spammers on the service side using noncontent features, including temporal and network features, without inspecting message text. At carrier scale, Jiang et al. (2013b) find that spam numbers with similar content show strongly similar sending patterns, and Jiang et al. (2013a) detect SMS spam numbers from their sending behavior in a large cellular network. Surveys of SMS spam filtering methods and data (Delany et al., 2012) focus mainly on content-based methods. Our work combines a transformer text encoder with temporal features and, in the full TemporalBertSpam design, sequence modeling over each sender’s history.

## 3 Problem Formulation

Let a sender s transmit messages $m _ { 1 } , m _ { 2 } , \ldots , m _ { n }$ at times $t _ { 1 } < t _ { 2 } < \cdots < t _ { n }$ . Each message $m _ { i }$ has text x<sub>i</sub> and a binary carrier label $y _ { i } \in \{ 0 , 1 \}$ , where $y _ { i } = 1$ means the carrier flagged the message as spam and $y _ { i } = 0$ means it was delivered. The task is to predict $y _ { i }$ from the text and temporal context of $m _ { i }$ and its predecessors from the same sender.

## 4 Dataset

## 4.1 Production SMS Database

Our data come from the outbound SMS database of a commercial CRM platform that sends messages through the Telnyx telecommunications API. Records were pulled from the platform’s PostgreSQL database in streaming batches of 10,000 rows, joining message records (text, sender number, send timestamp) with carrier delivery metadata (status and error codes).

Data scope The raw pull contains 59,342 outbound SMS messages sent between September 22 and December 19, 2025 (about 89 days), from multiple outbound phone numbers used by CRM customers for appointment reminders, marketing, and transactional notifications. We keep only the two terminal statuses that reflect carrier filtering decisions, spam (carrierflagged) and delivered, leaving 43,741 messages. Other statuses (delivery\_failed, queued, blocked, error, delivery\_ unconfirmed, sending\_failed, sent, read) do not cleanly reflect carrier content filtering and are excluded.

Label semantics Labels come from carrier delivery statuses, not human annotation. A message labeled spam (y = 1) was flagged by the carrier network as unsolicited or policy-violating; a message labeled delivered (y = 0) reached the recipient device.

Class distribution The dataset contains 18,283 spam messages (41.8%) and 25,458 delivered messages (58.2%). We report AUC-ROC alongside F1 because of this imbalance.

Timestamps and sender identity Each message has a UTC transmission timestamp at microsecond precision, recorded from Telnyx webhook events. The sender field holds the outbound phone number (E.164 format), which we use only to group messages into per-sender sequences. Sender numbers are not reported anywhere in this paper.

## 4.2 Preprocessing

Near-duplicate templates Spam campaigns often send many near-duplicate variants of one template. No messages were removed for nearduplication, so all 43,741 messages are used for modeling and near-duplicate variants may remain in the data. Section 8.2 describes a screening procedure for future experiments.

Time normalization Unix timestamps (seconds) are min-max normalized over the corpus:

$$
t _ { \mathrm { n o r m } } = \frac { t - t _ { \mathrm { m i n } } } { t _ { \mathrm { m a x } } - t _ { \mathrm { m i n } } } .
$$

Train/test split For the text baselines we use an 80/20 split stratified on the label, giving 34,992 training and 8,749 test messages (41.8% spam in each). Because this split is random, messages from the same sender and template can appear in both partitions. For the sequence model we specify a time-ordered split, with the earliest messages used for training and the most recent for testing, to avoid temporal leakage.

## 5 Methods

## 5.1 TF-IDF + Logistic Regression

The weakest baseline uses TF-IDF features (unigrams and bigrams, top 50,000 features, sublinear TF) with L2-regularized logistic regression (C = 1.0, liblinear solver).

## 5.2 Text-Only DeBERTa-v3-base

Our main text-only baseline fine-tunes DeBERTav3-base (He et al., 2023) (86M backbone parameters) for binary classification with BCEWith-LogitsLoss and a single output logit. DeBERTa uses disentangled attention over content and position representations (He et al., 2021), and DeBER-TaV3 adds ELECTRA-style pre-training (He et al., 2023). We use it as a strong text-only reference to test whether temporal information adds value beyond a strong content encoder.

Training details: AdamW $( \mathrm { l r } ~ = ~ 2 ~ \times ~ 1 0 ^ { - 5 }$ weight decay 0), batch size 32, 3 epochs, dynamic padding, maximum sequence length 128 tokens.

Label convention: In this experiment delivered messages are the positive class (label 1) and spam messages the negative class (label 0), following the platform’s internal convention. AUC-ROC is unaffected by this choice. The reported F1 is for the delivered class.

## 5.3 DistilBERT + Temporal Features

This model augments DistilBERT’s CLS representation with two temporal features: the normalized time delta $\Delta t _ { i } = t _ { i } - t _ { i - 1 }$ (0 for a sender’s first message) and log $( 1 + \Delta t _ { i } )$ . The features are projected to the CLS dimension by a learned linear layer and added to the CLS vector before the classification head. It uses temporal features without sequence modeling, and it is the model deployed in production.

## 5.4 TemporalBertSpam

The full model treats each sender’s history as a causal sequence and applies a second-stage Transformer over message representations.

Stage 1: Text encoding Each message $x _ { i }$ is encoded independently with distilbert-baseuncased; the final-layer CLS vector gives a 768- dimensional representation $h _ { i }$

Stage 2: Temporal feature projection The two temporal features are projected to 768 dimensions by a learned matrix $W _ { \mathrm { t i m e } } \in \mathbb { R } ^ { 7 6 8 \times 2 }$ and added to $h _ { i } { \mathrm { : } }$

$$
\tilde { h } _ { i } = h _ { i } + W _ { \mathrm { t i m e } } \left[ \Delta t _ { i } , \log ( 1 + \Delta t _ { i } ) \right] ^ { \top } .
$$

Stage 3: Sequence-level positional encoding The representations $\tilde { h } _ { 1 } , \dots , \tilde { h } _ { T }$ in a window receive sinusoidal positional encodings (Vaswani et al., 2017):

$$
\begin{array} { r l r } & { } & { \mathrm { P E } ( p , 2 k ) = \sin \Big ( p / 1 0 0 0 0 ^ { 2 k / d } \Big ) , } \\ & { } & { \mathrm { P E } ( p , 2 k + 1 ) = \cos \Big ( p / 1 0 0 0 0 ^ { 2 k / d } \Big ) , } \end{array}
$$

where $p \in \{ 0 , \ldots , T - 1 \}$ is the position of the message within the window and $d = 7 6 8$ . These encodings are distinct from DistilBERT’s learned token-position embeddings: DistilBERT’s embeddings encode the position of each token within one message, while the sinusoidal encoding marks the position of each message within the sender’s history, inside a separate Transformer.

Stage 4: Causal Transformer encoder The sequence passes through a 4-layer Transformer encoder (Vaswani et al., 2017) with 8 heads, feedforward dimension 2048, pre-layer normalization, GELU, and dropout 0.1. An upper-triangular attention mask makes position i attend only to positions $0 , \ldots , i$ . A 2-class linear head produces per-message logits.

Sequence construction Messages are grouped by sender, sorted by time, and split into sliding windows of $T = 3 2$ (stride 16 for training, 32 for evaluation). Shorter histories are left-padded with a padding mask. For a sender’s first message, no earlier context exists, so the prediction depends on that message’s content and temporal features only.

<table><tr><td>Model</td><td>AUC-ROC</td><td>F1</td><td>Acc.</td></tr><tr><td>TF-IDF + LR</td><td>0.8721</td><td></td><td>0.7912</td></tr><tr><td>DeBERTa-v3-base</td><td>0.9011</td><td>0.8412</td><td>0.8340</td></tr></table>

Table 1: Text baselines on the 8,749-message held-out test set (80/20 stratified split). F1 is for the delivered class (platform label convention, Section 5.2) and is reported for DeBERTa only.

Training configuration. AdamW $( \mathrm { l r } \ = \ 2 \times$ 10<sup>−5</sup>, weight decay 0.01), batch size 8, 5 epochs, linear warmup over 10% of steps, mixed precision, gradient clipping at norm 1.0, and cross-entropy over per-message predictions in each window.

## 6 Evaluation Setup

Metrics We report AUC-ROC (threshold-free and invariant to label encoding) for the text baselines, and F1 and accuracy for all models. Because the class balance differs between our two evaluation sets, we state the positive class for each F1 value.

Evaluation sets Text baselines are evaluated on the 8,749-message held-out test set. The deployed DistilBERT + temporal model is evaluated on a 20,000-message set drawn from the production monitoring system, with 10,000 spam and 10,000 delivered messages. Results across the two sets are not directly comparable.

## 7 Results

## 7.1 Held-Out Test Set

Table 1 shows the text baselines. DeBERTa reaches AUC-ROC 0.9011 and F1 0.8412, so the task is largely learnable from content alone, and the stronger encoder improves AUC by about 3 points over TF-IDF. These numbers are far below the 97– 99% accuracy reported by Jamal et al. (2024), but the tasks differ in domain, label source, and class setup, so the results are not directly comparable. Content-only models cannot see a sender’s history: two messages with identical text can carry different risk depending on whether they are a sender’s first message or one of hundreds sent in a burst.

## 7.2 Production Deployment Results

Table 2 reports the deployed model at threshold 0.5. The model favors recall: it catches 91.3% of carrierflagged spam, at the cost of 2,652 false positives among 10,000 delivered messages.

<table><tr><td rowspan=1 colspan=1>Metric            Value</td></tr><tr><td rowspan=3 colspan=1>Accuracy          0.8238Precision (spam)  0.7749Recall (spam)F1 (spam)</td></tr><tr><td rowspan=1 colspan=1>0.9128</td></tr><tr><td rowspan=1 colspan=1>0.8382</td></tr><tr><td rowspan=1 colspan=1>F1 (delivered)     0.8066</td></tr><tr><td rowspan=1 colspan=1>Macro F1         0.8224</td></tr><tr><td rowspan=1 colspan=1>True positives     9,128False positives     2,652True negatives     7,348</td></tr><tr><td rowspan=1 colspan=1>False negatives     872</td></tr><tr><td rowspan=1 colspan=1>Avg. confidence   0.753</td></tr></table>

Table 2: Deployed DistilBERT + temporal model on the 20,000-message production evaluation set (10,000 spam, 10,000 delivered) at threshold 0.5. Spam is the positive class.

<table><tr><td>Thr.</td><td>Acc.</td><td>Prec.</td><td>Rec.</td><td>F1</td><td>FP</td><td>FN</td></tr><tr><td>0.30</td><td>0.814</td><td>0.757</td><td>0.926</td><td>0.833</td><td>2972</td><td>744</td></tr><tr><td>0.35</td><td>0.815</td><td>0.760</td><td>0.920</td><td>0.833</td><td>2900</td><td>796</td></tr><tr><td>0.40</td><td>0.817</td><td>0.764</td><td>0.918</td><td>0.834</td><td>2840</td><td>816</td></tr><tr><td>0.45</td><td>0.821</td><td>0.769</td><td>0.916</td><td>0.836</td><td>2752</td><td>836</td></tr><tr><td>0.50</td><td>0.824</td><td>0.775</td><td>0.913</td><td>0.838</td><td>2652</td><td>872</td></tr><tr><td>0.55</td><td>0.825</td><td>0.779</td><td>0.909</td><td>0.839</td><td>2584</td><td>912</td></tr><tr><td>0.60</td><td>0.825</td><td>0.782</td><td>0.902</td><td>0.838</td><td>2512</td><td>980</td></tr><tr><td>0.65</td><td>0.826</td><td>0.785</td><td>0.899</td><td>0.838</td><td>2468</td><td>1012</td></tr><tr><td>0.70</td><td>0.824</td><td>0.786</td><td>0.888</td><td>0.834</td><td>2412</td><td>1116</td></tr><tr><td>0.75</td><td>0.807</td><td>0.801</td><td>0.816</td><td>0.808</td><td>2032</td><td>1836</td></tr><tr><td>0.80</td><td>0.734</td><td>0.848</td><td>0.571</td><td>0.682</td><td>1024</td><td>4292</td></tr></table>

Table 3: Threshold sweep on the production evaluation set. Best F1 (0.839) is at threshold 0.55 (bold). FP = false positives; FN = false negatives.

Threshold calibration Table 3 and Figure 1 show performance for thresholds from 0.30 to 0.80. F1 is nearly flat between 0.45 and 0.65 and peaks at 0.55 (0.839), a marginal gain over 0.5 (0.838). Recall falls sharply above 0.70. Production uses a threshold of 0.55. Because the threshold was chosen on the same set used for evaluation, these figures are optimistic.

Score distribution and calibration Figure 2 shows that predictions are bimodal: 7,112 messages score below 0.20 and 11,296 score at or above 0.70, with only 1,592 in between. Table 4 reports calibration in 0.1-wide probability buckets below 0.6. The two lowest buckets are well calibrated (error at most 0.011). Bucket 2 underestimates spam (predicted 0.247, observed 0.339), and buckets 4 and 5 overestimate it, with errors of 0.227 and 0.116. Predictions at or above 0.6 (11,532 messages) have an observed spam rate of 0.782, equal to the precision at threshold 0.6 in Table 3.

![](images/98c2f727a14ab43bd0143d32a97b8b00c3fb8c6a3f06edb1f151baedf68620e3.jpg)  
Figure 1: Precision, recall, and F1 across decision thresholds on the production evaluation set (from Table 3). The dotted line marks the production threshold of 0.55.

![](images/a21556781e2de8fe9c5a975b471bcfc86f68ba4bee476c0790a2824942fccc92.jpg)  
Figure 2: Distribution of predicted $p _ { \mathrm { s p a m } }$ on the 20,000- message production evaluation set (0.05-wide bins). The dashed line marks threshold 0.5. Most predictions fall below 0.20 or above 0.70, with few in between.

Model version history Table 5 lists the three versions promoted to production. Accuracy rose by 10.3 points between the first and second versions, while recall fell from 0.936 to 0.916, indicating a shift toward fewer false positives.

## 8 Discussion

## 8.1 Expected Behavior of the Full Model

The following are design hypotheses for Temporal-BertSpam, not measured results.

High-volume campaigns Burst sending produces short intervals $( \Delta t _ { i } \approx 0 )$ and many messages from one sender. The sequence encoder is designed to associate such patterns with higher spam probability, independently of text, which a per-message classifier cannot observe.

Spintax evasion When individual variants look benign to the text encoder, the sending pattern (regular intervals, many messages from one number)

<table><tr><td>Bucket</td><td>n</td><td>Avg. pred.</td><td>Observed</td><td>Error</td></tr><tr><td>[0.0, 0.1)</td><td>3732</td><td>0.025</td><td>0.021</td><td>0.004</td></tr><tr><td>[0.1, 0.2)</td><td>3380</td><td>0.141</td><td>0.130</td><td>0.011</td></tr><tr><td>[0.2, 0.3)</td><td>660</td><td>0.247</td><td>0.339</td><td>0.093</td></tr><tr><td>[0.3, 0.4)</td><td>204</td><td>0.338</td><td>0.353</td><td>0.015</td></tr><tr><td>[0.4, 0.5)</td><td>244</td><td>0.456</td><td>0.230</td><td>0.227</td></tr><tr><td>[0.5, 0.6)</td><td>248</td><td>0.552</td><td>0.436</td><td>0.116</td></tr></table>

Table 4: Calibration for predictions below 0.6 (8,468 of 20,000 messages). Observed is the actual spam rate in each bucket; Error is the absolute difference from the average prediction.
<table><tr><td>Version</td><td>Val. rows</td><td>Acc.</td><td>Prec.</td><td>Rec.</td></tr><tr><td>v20260502_223227</td><td>15k</td><td>0.717</td><td>0.651</td><td>0.936</td></tr><tr><td>v20260506_162645</td><td>15k</td><td>0.820</td><td>0.768</td><td>0.916</td></tr><tr><td>v20260506_223403</td><td>20k</td><td>0.824</td><td>0.775</td><td>0.913</td></tr></table>

Table 5: Production model version history.

may still provide a useful signal through the causal attention over earlier messages.

Cold start For a sender’s first message, no history exists, and the model relies on content and the temporal features of that message alone.

## 8.2 Near-Duplicate Screening for Future Experiments

We propose the following screening procedure. Each message is lowercased, whitespace is collapsed, and the text is represented as a TF-IDF vector over unigrams and bigrams, with weight $\mathrm { t f } ^ { \prime } ( t , d ) \cdot \log ( N / \mathrm { d f } ( t ) )$ and sublinear term frequency $\begin{array} { l } { { \sf t f ^ { \prime } } } \end{array} = \mathrm { ~ 1 ~ + ~ } \log ( { \sf t f } )$ Vectors are L2- normalized, so cosine similarity reduces to a dot product. Messages are processed in corpus order; a message is discarded if its maximum cosine similarity to any retained message is at least a threshold such as 0.9.

A high threshold catches variants that differ in only a few tokens, but spintax variants that swap many words or phrases share less vocabulary and fall below it. Since near-duplicates across the train/test split can inflate metrics, combining lexical screening with embedding-based deduplication (Reimers and Gurevych, 2019) is a priority for future experiments.

## 8.3 Positional Encoding Design

An alternative design would feed temporal features as extra token embeddings into DistilBERT. That would mix inter-message information into a model pre-trained only on intra-message token sequences.

Keeping a separate sequence Transformer with its own sinusoidal encoding keeps the two levels apart and makes the ablation clean: removing the sequence Transformer (the DistilBERT + temporal model) changes nothing in the text encoder. Sinusoidal encoding also adds no parameters, which matters given the limited number of distinct sender histories.

## 8.4 Comparison to Asliyuksek et al. (2025)

Asliyuksek et al. (2025) combine DistilBERT with static structural features and reach 99.62% accuracy on a combined email corpus. Our setting differs in three ways: (1) labels come from carrier decisions rather than curated corpora, which adds noise but reflects real network behavior; (2) our non-textual features are cross-message temporal signals rather than per-message statistics; (3) TemporalBertSpam explicitly models dependencies across a sender’s messages. Given these differences in domain, labels, and data, accuracy figures are not directly comparable.

## 8.5 Newer Encoders

DistilBERT (2019) and DeBERTaV3 (2021) predate several stronger encoders. ModernBERT (Warner et al., 2025) is trained on 2 trillion tokens with a native 8,192-token context; for SMS, which is short, its benefit would come mainly from better pre-training rather than longer context. Sentence-BERT (Reimers and Gurevych, 2019) and later embedding models could replace TF-IDF deduplication with semantic deduplication that catches paraphrased spintax. LLM-based embeddings could better capture commercial intent in novel phrasings.

A stronger text encoder would mainly help with hard cases: messages that are spam in intent but phrased in ways the encoder’s pre-training data did not cover. Behavioral signals such as burst sending and short intervals are independent of content, so we expect temporal information to remain useful even with a stronger encoder; testing this with ModernBERT or an LLM encoder at Stage 1 is a natural next experiment.

## 9 Conclusion

We study outbound SMS spam detection with carrier delivery labels and real transmission timestamps, using 43,741 production messages. A textonly DeBERTa-v3-base baseline reaches AUC-ROC 0.9011 and F1 0.8412 on a held-out test set, and a deployed DistilBERT + temporal model reaches spam-class F1 0.8382 and recall 0.9128 at threshold 0.5 on a separate, balanced 20,000- message production set. Unlike common public SMS spam benchmarks, the dataset includes permessage timestamps and carrier-decision labels. We also describe TemporalBertSpam, which models each sender’s history with a causal Transformer.

The key next step is to evaluate the TF-IDF, De-BERTa, DistilBERT + temporal, and Temporal-BertSpam models on one time-ordered split, which would directly answer RQ1 and RQ2, together with a subgroup analysis by message type and sender volume on the same split (RQ3). Further directions include near-duplicate removal, newer encoders at Stage 1, and few-shot methods for senders with sparse histories.

## Limitations

Incomplete comparison The text baselines and the deployed model are evaluated on different sets with different class balance (41.8% vs. 50% spam), so we cannot yet attribute performance differences to temporal features. The full TemporalBertSpam model has not been evaluated in this paper.

Evaluation protocol The text baselines use a random split, so messages from the same sender or template can appear in both training and test data, which may inflate their scores. The production evaluation set was drawn from the monitoring system, and we do not document its sampling procedure or rule out overlap with the validation data of the deployed model version. The production decision threshold was selected on the same set used for evaluation. Production results should therefore be read as optimistic.

Label noise and selection bias Carrier statuses are an imperfect proxy for spam, and carrier filtering policies may change over time. The production system uses a stochastic send policy that holds some messages; held messages receive no carrier label, so the labeled data are not a random sample of all attempted messages.

Near-duplicates No near-duplicate removal was applied, so templated variants may appear in both training and test data.

Generality The data come from one CRM platform, one messaging provider, and an 89-day window, and the dataset cannot be released because it is proprietary. Results may not transfer to other platforms, regions, or periods.

## Ethics Statement

Message text contains real SMS content sent by CRM users. We report only aggregate statistics and model metrics; and no message text, phone number, or sender identifier is reproduced. The data were accessed under the data use agreement governing the authors’ access to the platform’s systems.

## References

Tiago A. Almeida, José María Gómez Hidalgo, and Akebo Yamakami. 2011. Contributions to the study of SMS spam filtering: New collection and results. In Proceedings of the 11th ACM Symposium on Document Engineering (DocEng), pages 259–262. ACM.

H. Asliyuksek, O. Tonkal, and R. Kocaoglu. 2025. A comparative evaluation of a multimodal approach for spam email classification using DistilBERT and structural features. Electronics, 14(19):3855.

Sarah Jane Delany, Mark Buckley, and Derek Greene. 2012. SMS spam filtering: Methods and data. Expert Systems with Applications, 39(10):9899–9908.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186. Association for Computational Linguistics.

Shuang Hao, Nadeem Ahmed Syed, Nick Feamster, Alexander G. Gray, and Sven Krasser. 2009. Detecting spammers with SNARE: Spatio-temporal network-level automatic reputation engine. In Proceedings of the 18th USENIX Security Symposium. USENIX Association.

Pengcheng He, Jianfeng Gao, and Weizhu Chen. 2023. DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradientdisentangled embedding sharing. In International Conference on Learning Representations.

Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen. 2021. DeBERTa: Decoding-enhanced BERT with disentangled attention. In International Conference on Learning Representations.

Suhaima Jamal, Hayden Wimmer, and Iqbal H. Sarker. 2024. An improved transformer-based model for detecting phishing, spam and ham emails: A large language model approach. Security and Privacy, 7(5):e402.

Nan Jiang, Yu Jin, Ann Skudlark, and Zhi-Li Zhang. 2013a. Greystar: Fast and accurate detection of SMS spam numbers in large cellular networks using gray phone space. In Proceedings of the 22nd USENIX Security Symposium. USENIX Association.

Nan Jiang, Yu Jin, Ann Skudlark, and Zhi-Li Zhang. 2013b. Understanding SMS spam in a large cellular network: Characteristics, strategies and defenses. In Research in Attacks, Intrusions, and Defenses (RAID 2013), volume 8145 of Lecture Notes in Computer Science. Springer.

Anirudh Ramachandran, Nick Feamster, and Santosh Vempala. 2007. Filtering spam with behavioral blacklisting. In Proceedings of the 14th ACM Conference on Computer and Communications Security (CCS). ACM.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992. Association for Computational Linguistics.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, pages 5998–6008.

Benjamin Warner, Antoine Chaffin, Benjamin Clavié, Orion Weller, Oskar Hallström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, Griffin Thomas Adams, Jeremy Howard, and Iacopo Poli. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory efficient, and long context finetuning and inference. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2526–2547. Association for Computational Linguistics.

Qian Xu, Evan Wei Xiang, Qiang Yang, Jiachun Du, and Jieping Zhong. 2012. SMS spam detection using noncontent features. IEEE Intelligent Systems, 27(6):44–51.