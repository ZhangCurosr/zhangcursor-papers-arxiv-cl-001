# SEQUENTIAL ADAPTER STACKING FOR CROSS-LINGUAL LOW-RESOURCE ASR

Thai Thi Thanh Thao Dang, Mengjie Qian, Kate Knill

Department of Engineering, University of Cambridge, UK

## ABSTRACT

Extending large-scale multilingual automatic speech recognition (ASR) models to low-resource languages remains challenging. Model performance is skewed toward highresource languages and degrades sharply for languages with limited labeled data and pre-training exposure. To address this, we investigate parameter-efficient approaches for transferring knowledge from resource-rich source languages to low-resource target languages on Whisper. Alongside warm initialization and attention-based fusion, we propose Sequential Adapter Stacking, which places a trainable targetlanguage adapter on top of a frozen source-language adapter. Under controlled experiments, these approaches are evaluated on three target languages unsupported by Whisper – Asturian, Assamese, and Xhosa – using source languages with varying degrees of relatedness. Sequential Adapter Stacking with the closest related source consistently and significantly outperforms full fine-tuning across the three targets, with 5–8% relative WER reductions. These gains largely persist with only one hour of target training data.

Index Terms— Whisper, parameter-efficient fine-tuning, low-resource ASR, cross-lingual adapter transfer

## 1. INTRODUCTION

Large-scale multilingual speech models such as Whisper [1] and MMS [2] have markedly advanced Automatic Speech Recognition (ASR) across languages. However, performance remains highly uneven across languages and deteriorates sharply for languages absent from pre-training. Adapting such models to unsupported languages is particularly challenging when limited transcribed speech is available. Moreover, adaptation performance depends not only on the target language being seen during pre-training, but also on the model’s exposure to related languages [3], suggesting that representations learned from related languages could provide useful knowledge for target-language adaptation.

Full fine-tuning is a straightforward approach to language adaptation, but it updates the entire model and risks overwriting useful pre-trained representations when target data is scarce [4]. Parameter-efficient fine-tuning such as adapter tuning [5, 6], Low-Rank Adaptation (LoRA) [7], prefix tuning [8], and prompt tuning [9] instead adapt a small amount of parameters while keeping the pre-trained backbone frozen. Bottleneck adapters are particularly attractive, since independently trained language adapters can be reused and composed for cross-lingual transfer from resource-rich source languages to low-resource targets [10, 11]. Existing cross-lingual ASR approaches exploit source adapters in different ways. Dualadapter methods combine language-specific and languagegeneral modules [12], warm initialization transfers source knowledge by initializing target adaptation from a sourcelanguage module [13], and fusion-based methods combine information from multiple language adapters dynamically through learned weighting [14, 15, 16]. Despite these developments, how best to exploit a pre-trained source-language adapter when adapting a massively multilingual ASR model to an unsupported language remains under-explored.

A related question is which source language should be used. Language selection has historically received more attention in Natural Language Processing (NLP) than speech recognition. In NLP, no single feature reliably identifies the best source across tasks, and features grounded in the target dataset (e.g. subword overlap) and the model’s own representations (e.g. embedding similarity) have proven highly predictive of cross-lingual transfer [17, 18, 19]. In speech recognition, previous cross-lingual ASR studies commonly select source languages based on availability or linguistic intuition [13, 14, 15, 16]. Even when language similarity is quantified beforehand, it is used to predict transfer outcomes under a fixed pipeline rather than to drive an adaptation decision [20], and phonetic metrics alone are unreliable predictors [21]. This motivates jointly examining both how source knowledge is transferred and how source–target relatedness affects that transfer.

In this work, Whisper serves as a case study of a massively multilingual ASR backbone. Inspired by the modular architecture of MAD-X [22], we propose Sequential Adapter Stacking (SeqStack) for cross-lingual source-adapter transfer and use three complementary language similarity measures to rank candidate sources before adaptation. The approach is compared with existing transfer strategies, including warm initialization and weighted fusion, on Asturian, Assamese, and Xhosa, which span different levels of relatedlanguage pre-training coverage in Whisper. The contributions of this work are as follows: (1) the first evaluation, to our knowledge, of SeqStack for cross-lingual ASR on a massively multilingual backbone, where it yields statistically significant improvements over full fine-tuning and monolingual adapter tuning across all three target languages; (2) a controlled comparison of three source-adapter transfer strategies, i.e. warm initialization, attention-based fusion, and SeqStack, under matched training conditions; (3) a practical framework for extending multilingual ASR to unsupported languages by combining similarity-based source selection with SeqStack.

![](images/e895146f206443e9cdad34349e15c8522f17eacca4e984e99aa20034f55344fb.jpg)

## 2. METHODS

This section reviews the source-adapter transfer approaches compared in our evaluation (Section 2.1), presents Sequential Adapter Stacking (Section 2.2), and describes the three complementary language similarity measures used to select source languages prior to adaptation (Section 2.3). Throughout, t denotes the target language, and $\boldsymbol { S } = \{ s _ { 1 } , \ldots , s _ { K } \}$ is the set of auxiliary source languages. We use $\Phi _ { \ell }$ to represent a bottleneck adapter for language $\ell ,$ inserted into the frozen Whisper backbone with pre-trained weights Θ, and $\mathcal { D } _ { \ell }$ is the corresponding training set.

## 2.1. Compared Source-Adapter Transfer Approaches

Bottleneck adapters [5] insert a down-projection $\mathbf { W } _ { \mathrm { d o w n } } \in$ $\mathbb { R } ^ { d _ { \mathrm { m o d e l } } \times d } .$ , a non-linear activation $f ,$ and an up-projection $\mathbf { W } _ { \mathrm { u p } } \in \mathbb { R } ^ { d \times c }$ <sup>dmodel</sup> into each Transformer layer, with bottleneck dimension $d \ll d _ { \mathrm { m o d e l } }$ . Following the Pfeiffer configuration [6], a single adapter is applied to the Feed-Forward Network output h in each Whisper encoder and decoder layer:<sup>1</sup>

$$
\mathrm { A d a p t e r } ( \mathbf { h } ) = \mathbf { h } + f ( \mathbf { h } \mathbf { W } _ { \mathrm { d o w n } } ) \mathbf { W } _ { \mathrm { u p } } = \mathbf { h } + \Phi ( \mathbf { h } ) .
$$

Target-only Adapter Tuning (Adapter FT) trains a nearzero initialized target adapter $\Phi _ { t }$ on $\mathcal { D } _ { t }$ while keeping the Whisper backbone frozen. Two existing cross-lingual transfer strategies extend this baseline using a pre-trained source adapter. Warm-initialized Transfer (Warm-init.), adapted from LoRA-Whisper [13], initializes $\Phi _ { t }$ from a source adapter $\Phi _ { s }$ before training on $\mathcal { D } _ { t }$ . Target-inclusive Fusion (Target-incl.), inspired by SimAdapter [14], fuses pre-trained source and target adapters through layer-wise learned attention using a lightweight AdapterFusion variant [23]. Our variant projects queries and keys to $d _ { k } \ll d _ { \mathrm { m o d e l } }$ and directly reweights adapter outputs without a value projection, keeping the additional parameter count comparable to a single adapter.

## 2.2. Sequential Adapter Stacking

Motivated by the stacked-adapter design of MAD-X [22], SeqStack composes source and target knowledge serially in the cross-lingual ASR setting. While MAD-X stacks a trainable task adapter on a frozen language adapter, with the source

(a) MAD-X Framework [22]

(b) Sequential Stacking

Fig. 1: Framework of (a) MAD-X and (b) our proposed Sequential Stacking. Layer normalization is omitted for clarity.

language adapter replaced by the target language adapter at inference for zero-shot transfer, our approach stacks two language adapters (Figure 1). A source language adapter $\Phi _ { s _ { k } }$ is first trained monolingually on $\mathcal { D } _ { s _ { k } }$ and frozen. $\mathbf { A }$ near-zero initialized target language adapter $\Phi _ { t }$ is then stacked on top as the sole component optimized on $\mathcal { D } _ { t } \mathbf { : }$

$$
\mathbf { h } _ { s _ { k } } ^ { ( l ) } = \mathbf { h } ^ { ( l ) } + \Phi _ { s _ { k } } \bigl ( \mathbf { h } ^ { ( l ) } \bigr ) , \qquad \mathbf { h } ^ { \prime ( l ) } = \mathbf { h } _ { s _ { k } } ^ { ( l ) } + \Phi _ { t } \bigl ( \mathbf { h } _ { s _ { k } } ^ { ( l ) } \bigr ) .
$$

Gradients propagate only through $\Phi _ { t } .$ . The frozen source adapter provides a fixed, source-conditioned transformation of the backbone representation. The residual connection wiring also reflects each design’s goal: MAD-X keeps its language adapters substitutable at inference, whereas SeqStack’s source conditioning is meant to persist, so the target language adapter’s residual wraps the source-conditioned state $\mathbf { h } _ { s _ { k } }$

## 2.3. Source Language Selection

For each target, candidate sources are ranked before adaptation using three complementary measures of source–target relatedness computed from FLEURS [24] text and speech.

Encoder representation similarity (EncSim). This metric measures the similarity of Whisper encoder representations across languages using 100 utterances per language. First, hidden states for each utterance i of language ℓ are mean-pooled over non-padding frames to extract layer-wise vectors, which are L2-normalized before averaging within four layer bands B (early 0–5, mid-shallow 6–11, mid-deep 12–17, and deep 18–23). To mitigate transformer anisotropy [25], the global mean across candidate languages is then subtracted before computing each language centroid $\mathbf { c } _ { \ell } ^ { B }$ . Finally, the pairwise similarity is computed as the cosine distance between centroids: sim $\mathbf { \mathbb { \Lambda } } ^ { \mathsf { B } } ( \ell _ { 1 } , \ell _ { 2 } { \ ' } ) = ( \mathbf { c } _ { \ell _ { 1 } } ^ { \mathsf { B } } \cdot \mathbf { c } _ { \ell _ { 2 } } ^ { \mathsf { B } } ) / ( \| \mathbf { c } _ { \ell _ { 1 } } ^ { \mathsf { B } } \| \| \mathbf { c } _ { \ell _ { 2 } } ^ { \mathsf { B } } \| )$

Token coverage (TokCov). Decoder-side similarity is measured by Byte-Pair Encoding (BPE) token coverage, defined as the fraction of the target token stream appearing in the source corpus. Since Whisper’s tokenization granularity is heavily influenced by pre-training exposure [1, 26], aggregate token coverage alone may conflate genuine lexical sharing with byte-level overlap. To mitigate this, tokens are classified by the number of complete Unicode code points they span (e.g. subword, character, or byte), and coverage is decomposed by script class (e.g. Latin, Brahmic, or CJK).

Table 1: Target languages with candidate sources ranked by the similarity framework and Whisper pre-trained hours.
<table><tr><td>Target</td><td>|Source</td><td colspan="3">Hours EncSim TokCov(%) GenSim</td></tr><tr><td>Asturian</td><td>|Spanish es French fr Mandarin zh</td><td>11.1k 0.81 9.8k 0.61 23.4k 0.40</td><td>80.2 62.6 一</td><td>0.96 0.78 0.00</td></tr><tr><td>Assamese</td><td>|Bengali bn Hindi hi Mandarin zh</td><td>1.3 0.78 12.0 0.26 23.4k -0.12</td><td>27.2 0.0 0.0</td><td>0.80 0.40 0.00</td></tr><tr><td>Xhosa</td><td>|Zulu zu Swahili sw Mandarin zh</td><td>0.0 0.86 5.4 0.71 23.4k -0.38</td><td>84.9 76.1 27.5</td><td>1.00 0.61 0.00</td></tr></table>

EncSim: Deep-band centroid cosine similarity, where a value close to 1 indicates high similarity and -1 indicates maximal dissimilarity.

Genealogical relatedness (GenSim). This model-independent metric is measured by the cosine similarity between binary Glottolog family-tree vectors from URIEL [27].

All three measures produce the same source ranking for each target (Table 1). Our experiments then compare the highest-ranked (close) source, the second-ranked (medium) source, and Mandarin as a distant control.

## 3. EXPERIMENTAL SETUP

Data and Languages. The experiments are conducted on FLEURS [24] across Asturian, Assamese, and Xhosa (Table 2). These languages were selected because they share weak vanilla baselines (WER ≥ 50%) while representing three distinct tiers of unsupported languages: those with extensive (>1,000 hours), partial (10–1,000 hours), and minimal (<10 hours) pre-training exposure of their closest related languages (Table 1). For languages lacking a native Whisper language token, we use empirically related-language proxies: Spanish ⟨|es|⟩ for Asturian and Swahili ⟨|sw|⟩ for Xhosa. Although excluded from Whisper’s ASR pre-training, Assamese utilizes its native token ⟨|as|⟩ derived from the model’s speech translation data. To control for training volume, source language adapters are each trained on approximately 120 hours of speech by combining each language’s full FLEURS train split with data sampled from the train split of a supplementary corpus based on public availability. We use Common Voice [28] for Spanish, French, Mandarin and Swahili, IndicVoices [29] for Bengali and Hindi, and Swivuriso [30] for Zulu.

Pre-processing. Audio is resampled to 16kHz and filtered to 0.5–30 seconds. Text normalization follows a twotier design applied to both references and hypotheses before scoring, covering standard normalization (Unicode NFC, lowercasing, punctuation removal) and language-specific orthographic rules to preserve diacritics and elision.

Table 2: FLEURS dataset statistics for target languages.
<table><tr><td>Language</td><td>Lang. Token</td><td>Train (h)</td><td>Val (h)</td><td>Test (h)</td></tr><tr><td>Asturian</td><td>《|es|&gt;</td><td>7.5</td><td>0.9</td><td>2.4</td></tr><tr><td>Assamese</td><td>(|as|)</td><td>10.7</td><td>1.4</td><td>3.5</td></tr><tr><td>Xhosa</td><td>(|sw|&gt;</td><td>13.3</td><td>1.5</td><td>3.8</td></tr></table>

Training and Decoding Configurations. Whisper medium is utilized to balance computational efficiency and performance. To control for the trainable parameter count across adapter-based methods, Pfeiffer adapters adopt dimension $d = 2 5 6$ , and fusion layers use projection dimension $d _ { k } = 2 5 6$ . Models are optimized using AdamW with BF16 precision, an effective batch size of 32, and linear learningrate decay with 5% warmup. Based on a small grid search, peak learning rates are set to $1 \times 1 0 ^ { - 5 }$ for full fine-tuning, $3 \times 1 0 ^ { - 4 }$ for adapter methods, and $5 \times 1 0 ^ { - 5 }$ for fusion. Training runs up to 20 epochs with an early-stopping patience of 3 epochs based on validation WER. Decoding uses greedy search without external language model re-scoring.

Evaluation Metric. We report Word Error Rate (WER) and determine statistical significance using the Matched-Pairs Sentence-Segment Word Error test (MAPSSWE) at $p < 0 . 0 5$

## 4. RESULTS AND DISCUSSION

Baseline Performance. Vanilla Whisper achieves reasonable performance on Asturian with a WER of 52.5% but collapses on Xhosa and Assamese (>130% WER) (Table 3). Although monolingual adapter tuning substantially improves performance, full fine-tuning establishes a stronger reference for evaluating cross-lingual transfer methods, reducing WER from the vanilla baseline by 70% on Asturian (15.7%), 74% on Assamese (36.3%), and 69% on Xhosa (41.9%).

Source-Adapter Transfer Evaluation. Across all source– target pairs, Warm-initialized Transfer performs inconsistently against adapter tuning, improving on some pairs while degrading on others. The inconsistency persists even with the closest source, Asturian–Spanish, which falls short of full fine-tuning. This likely occurs because warm initialization primarily provides a better optimization starting point. Since monolingual adapter tuning already achieves strong performance from a near-identity initialization, the head start from source-learned weights provides negligible benefit.

Target-inclusive Fusion outperforms adapter tuning on certain pairs with a close source (Assamese–Bengali and Xhosa–Zulu) but lags behind full fine-tuning for all targets. On Asturian–Spanish, it degrades performance below adapter tuning. To understand this regression, we analyze how the fusion mechanism allocates attention weights across the frozen adapters. In the deep band, decoder attention correctly exceeds 70% on the Asturian adapter, whereas encoder attention drifts down to 40%. Since adapter tuning is already close to full fine-tuning, this persistent reliance on the Spanish adapter dilutes a near-ceiling system rather than adding useful knowledge. Prior works address similar attention drift using fusion guide loss [14] or dot-product mixture weighting [16]. However, these methods were validated on backbones with limited or no multilingual pre-training, leaving their effectiveness in our setting open. Under a constrained compute budget, we prioritize a structural alternative (SeqStack) that sidesteps the allocation problem rather than regularizing it.

Table 3: Low-resource language adaptation on FLEURS (%WER).
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=2>Source</td><td rowspan=1 colspan=1>Asturian  Assamese Xhosa</td></tr><tr><td rowspan=2 colspan=1>VanillaFull FTAdapter FT</td><td rowspan=2 colspan=2>一一一</td><td rowspan=1 colspan=1>52.5       141.2     133.3</td></tr><tr><td rowspan=1 colspan=1>16.8       40.5      49.6</td></tr><tr><td rowspan=3 colspan=1>Warm-init.</td><td rowspan=1 colspan=2>Close</td><td rowspan=1 colspan=1>16.6       33.6†     38.5†</td></tr><tr><td rowspan=2 colspan=2>MediumDistant</td><td rowspan=1 colspan=1>dium</td><td rowspan=1 colspan=1>17.3       39.1      45.4</td></tr><tr><td rowspan=1 colspan=1>16.8       42.3      44.1</td></tr><tr><td rowspan=3 colspan=1>Target-incl.</td><td rowspan=3 colspan=2>CloseMediumDistant</td><td rowspan=1 colspan=1>20.4       39.7      45.1</td></tr><tr><td rowspan=1 colspan=1>16.9       42.6      55.0</td></tr><tr><td rowspan=1 colspan=1>16.9       40.2     51.0</td></tr><tr><td rowspan=3 colspan=1>SeqStack</td><td rowspan=3 colspan=2>CloseMediumDistant</td><td rowspan=1 colspan=1>14.9†      34.1†     38.5†</td></tr><tr><td rowspan=1 colspan=1>15.9       38.1      44.3</td></tr><tr><td rowspan=1 colspan=1>16.3       40.4      44.3</td></tr></table>

Close/medium/distant sources are es/fr/zh for Asturian, bn/hi/zh for Assamese, and zu/sw/zh for Xhosa.  
† denotes $p < 0 . 0 5$ vs. Full fine-tuning (Full FT).

Sequential Adapter Stacking Evaluation. Across all targets, SeqStack with the target’s closest source language – Spanish for Asturian, Bengali for Assamese, and Zulu for Xhosa – improves significantly over both adapter tuning and full fine-tuning. The relative margin over full fine-tuning grows as related-language pre-training coverage shrinks (5% on Asturian, 6% on Assamese, and 8% on Xhosa). Furthermore, SeqStack’s performance validates our three-axis similarity framework. The framework identifies the relative order of candidate sources, and empirical results demonstrate that the closest source consistently performs better than both medium and distant controls. On Xhosa, where the medium and distant sources tie, we conduct an ablation to understand whether the performance gains stem from cross-lingual transfer or the raw parameter increase of stacking adapters. Doubling the capacity of the baseline monolingual adapter (d = 512) reduces WER from 49.6% to 45.6%, whereas SeqStack with the Zulu adapter at our controlled dimension d = 256 yields 38.5% WER. This suggests that languagespecific transfer drives the improvement on Xhosa–Zulu.

While the closest source consistently outperforms less related ones under SeqStack, medium sources do not reliably improve upon the distant control. We hypothesize that this stems from two factors. French’s Romance-general knowledge is plausibly redundant with Whisper’s extensive Romance pre-training, leaving Spanish’s advantage for Asturian branch-specific. Conversely, Xhosa’s lack of transfer from Swahili highlights a genealogical boundary, favoring Zulu, a closely related Nguni language, over a distant Bantu branch. Therefore, the similarity framework serves best as an ordinal instrument for closest source selection rather than a predictor of transfer magnitude. Overall, warm-initialized and fusionbased methods perform inconsistently across source–target language pairs, whereas SeqStack empirically does not fall below adapter tuning performance on any pairs. When the best source language is uncertain, SeqStack is a safer strategy for extending multilingual ASR to unsupported languages.

Table 4: FLEURS test WER (%) on one-hour target-data.
<table><tr><td>Method</td><td>Asturian</td><td>Assamese</td><td>Xhosa</td></tr><tr><td>Full FT</td><td>21.3</td><td>53.6</td><td>63.9</td></tr><tr><td>Adapter FT</td><td>23.1</td><td>100.0</td><td>70.0</td></tr><tr><td>SeqStack (Close)</td><td>21.8</td><td>47.0†</td><td>47.5†</td></tr></table>

† denotes p < 0.05 vs. Full FT.

Low-regime Analysis. To evaluate data efficiency under extreme low-resource conditions, we restrict target training data to a one-hour subset and compare SeqStack, paired with each target’s closest source, against full fine-tuning and adapter tuning trained on the same subset. The source adapters remain trained on their full 120-hour corpora. Full fine-tuning is functional across all targets. Adapter tuning under-performs substantially and collapses completely on Assamese (Table 4). In contrast, SeqStack largely maintains its advantage, achieving relative WER reductions over full fine-tuning of 12% on Assamese and 26% on Xhosa. The frozen source adapter provides both cross-lingual transfer and the architectural stability that adapter tuning lacks.

## 5. CONCLUSION

This work investigated adapting massively multilingual ASR backbones to unseen languages by optimizing both source selection and transfer mechanisms. Under controlled experiments, the proposed SeqStack proves robust across source– target pairs, mitigating the inconsistencies observed on warminitialized and fusion-based transfer methods. Paired with each target’s closest source, SeqStack significantly outperforms full fine-tuning, reducing WER by 5–8% on full training data and 12–26% on a one-hour subset. The gain is most substantial on Xhosa, where the backbone’s related-language coverage is weakest, and language-specific transfer materializes only with the closest related source. While SeqStack is architecturally applicable to other adapter-compatible speech foundation models, we focus on Whisper due to computational constraints, leaving its generalizability to other backbones for future work.

## 6. REFERENCES

[1] A. Radford et al., “Robust speech recognition via largescale weak supervision,” in Proc. ICML, 2022.

[2] V. Pratap, A. Tjandra, B. Shi, et al., “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research, 2024.

[3] A. Rouditchenko et al., “Comparison of multilingual self-supervised and weakly-supervised speech pretraining for adaptation to unseen languages,” in Proc. Interspeech, 2023, pp. 2268–2272.

[4] M. Qian et al., “Learn and Don’t Forget: Adding a New Language to ASR Foundation Models,” in Proc. Interspeech, 2024, pp. 2544–2548.

[5] N. Houlsby et al., “Parameter-efficient transfer learning for NLP,” in Proc. ICML, 2019.

[6] J. Pfeiffer et al., “AdapterHub: A framework for adapting transformers,” in Proc. EMNLP: System Demonstrations, 2020, pp. 46–54.

[7] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” in Proc. ICLR, 2022.

[8] X. L. Li and P. Liang, “Prefix-Tuning: Optimizing Continuous Prompts for Generation,” in Proc. ACL-IJCNLP (Volume 1: Long Papers), 2021, pp. 4582–4597.

[9] B. Lester et al., “The Power of Scale for Parameter-Efficient Prompt Tuning,” in Proc. EMNLP, 2021, pp. 3045–3059.

[10] Y. Liu and D. Qu, “Parameter-efficient fine-tuning of Whispe for low-resource speech recognition,” in Proc. AINIT. IEEE, 2024, pp. 1522–1525.

[11] H. Wang et al., “Low-resource speech recognition by fine-tuning Whisper with Optuna-LoRA,” Applied Sciences, vol. 15, no. 24, pp. 13090, 2025.

[12] G. I. Winata et al., “Adapt-and-adjust: Overcoming the long-tail problem of multilingual speech recognition,” in Proc. Interspeech, 2021, pp. 2451–2455.

[13] Z. Song et al., “LoRA-Whisper: Parameter-efficient and extensible multilingual ASR,” in Proc. Interspeech, 2024, pp. 3934–3938.

[14] W. Hou et al., “Exploiting adapters for cross-lingual low-resource speech recognition,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2021.

[15] Q. Hu et al., “Language fusion via adapters for lowresource speech recognition,” Speech Communication, 2024.

[16] Q. Hu et al., “CAM: A cross-lingual adaptation framework for low-resource language speech recognition,” Information Fusion, vol. 111, pp. 102506, 2024.

[17] Y.-H. Lin et al., “Choosing transfer languages for crosslingual learning,” in Proc. ACL, 2019, pp. 3125–3135.

[18] J. Eronen et al., “Zero-shot cross-lingual transfer language selection using linguistic similarity,” Information Processing & Management, 2023.

[19] T. K. Idris et al., “Can Embedding Similarity Predict Cross-Lingual Transfer? A Systematic Study on African Languages,” arXiv preprint arXiv:2601.03168, 2026.

[20] P. Wu et al., “Cross-Lingual Transfer for Speech Processing Using Acoustic Language Similarity,” in Proc. ASRU. IEEE, 2021, pp. 1050–1057.

[21] M. U. Farooq and T. Hain, “Investigating the Impact of Crosslingual Acoustic-Phonetic Similarities on Multilingual Speech Recognition,” in Proc. Interspeech, 2022, pp. 3849–3853.

[22] J. Pfeiffer et al., “MAD-X: An adapter-based framework for multi-task cross-lingual transfer,” in Proc. EMNLP, 2020, pp. 7654–7673.

[23] J. Pfeiffer et al., “AdapterFusion: Non-destructive task composition for transfer learning,” in Proc. EACL: Main Volume, 2021, pp. 487–503.

[24] A. Conneau et al., “FLEURS: Few-shot learning evaluation of universal representations of speech,” in Proc. SLT. IEEE, 2023, pp. 798–805.

[25] N. Godey et al., “Anisotropy is inherent to self-attention in transformers,” in Proc. EACL (Volume 1: Long Papers), 2024, pp. 35–48.

[26] S. Liang et al., “Beyond WER: Probing Whisper’s subtoken decoder across diverse language resource levels,” in Proc. EMNLP, 2025, pp. 31225–31235.

[27] P. Littell et al., “URIEL and lang2vec: Representing languages as typological, geographical, and phylogenetic vectors,” in Proc. EACL: Volume 2, Short Papers, 2017, pp. 8–14.

[28] R. Ardila et al., “Common Voice: A massivelymultilingual speech corpus,” in Proc. LREC, 2020, pp. 4218–4222.

[29] T. Javed et al., “IndicVoices: Towards building an inclusive multilingual speech dataset for Indian languages,” in Findings ofACL, 2024, pp. 10740–10782.

[30] V. Marivate et al., “Swivuriso: The South African next voices multilingual speech dataset,” arXiv preprint arXiv:2512.02201, 2025.