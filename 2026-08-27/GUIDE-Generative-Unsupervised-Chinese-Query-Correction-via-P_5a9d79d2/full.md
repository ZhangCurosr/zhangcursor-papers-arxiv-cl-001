# GUIDE: Generative Unsupervised Chinese Query Correction via Phonetic and Visual Shared-ID Encoding

Lei Yang<sup>1</sup>, Binbin Huang<sup>1,2</sup>\*, Jiwei Tan<sup>1</sup>, Xuhui Sui<sup>1</sup>, Chang Tu<sup>1</sup>, Yi Wang<sup>1</sup>, Han Li<sup>1</sup>

<sup>1</sup>Kuaishou Technology, <sup>2</sup>School of Data Science, Fudan University Correspondence: chentiancheng03@kuaishou.com

## Abstract

Chinese query correction (CQC) is important for search and query recommendation on content platforms, but supervised methods rely on large annotated correction pairs that are costly to maintain as query vocabularies evolve. Unsupervised correction with language models is attractive, yet in the short-query setting, unconstrained generation often over-corrects ambiguous inputs toward high-frequency phrases, causing intent drift. We propose GUIDE, a generative unsupervised framework for CQC based on a confuse-then-clarify paradigm. GUIDE encodes phonetically or visually confusable characters with shared-IDs and reconstructs the original query with an encoder–decoder architecture, which constrains correction to plausible confusion neighborhoods while learning from unlabeled query streams. A timedecayed, query-frequency-weighted objective further supports adaptation to rapidly changing query vocabularies. Experiments on QSpell 250K and a large-scale real-world dataset (KwaiSearch) show that GUIDE consistently outperforms strong baselines, while online A/B testing further confirms gains in correction quality and downstream engagement.

## 1 Introduction

Search queries directly express user intent and determine what content is retrieved and surfaced. On large content platforms such as YouTube and Tik-Tok, query recommendation modules—including search suggestions, autocomplete, and relatedquery recommendation—are often built from historical search logs and user interactions (Bacciu et al., 2024). As a result, misspellings and noncanonical variants in query streams can be repeatedly exposed, clicked, and re-mined, causing noisy forms to propagate through downstream retrieval, ranking, and recommendation pipelines (Gao et al., 2010; Sun et al., 2012). This makes Chinese query correction (CQC) an important problem in industrial search systems.

While Chinese spelling correction is often studied at the sentence level, CQC is harder in practice because it faces several query-specific constraints: (Yang et al., 2023; Su et al., 2025): (1) Short Context: queries provide little context, making it hard to decide whether to edit and what is intended; (2) Fast Shift: vocabularies change rapidly with new memes, influencers, and emerging entities; Creative Use: homophones can be intentional puns— e.g., “钱途” (lit. “money road”, i.e., “money- making prospects”) puns on its homophone “前途” (qiántú, “future prospects”)—so different is not always wrong; and (4) Few Labels: large-scale query annotations are expensive, and even millions of labeled pairs cover only a small fraction of real errors. These constraints make CQC fundamentally a problem of controlled editing: the system must decide not only what to correct, but also when to correct and how far it may deviate from the original expression.

Existing approaches only partially address this challenge. Supervised correctors learn a correction policy from labeled pairs (Zhang et al., 2020; Xu et al., 2021), but collecting and refreshing such annotations is expensive under fast vocabulary shift. Weakly supervised methods typically construct pseudo pairs through synthetic corruption (Liu et al., 2021; Li, 2022), which introduces a hand-designed noise process that may not match real query errors. Training-free LM-based correction is attractive (Hong et al., 2019; Zhou et al., 2024), but unconstrained or weakly constrained decoding often over-corrects short ambiguous queries, introduces unnecessary edits, or fails to preserve the original query length (Li et al., 2023; Liu et al., 2024).

These limitations suggest that effective unsupervised CQC should not rely on free-form rewriting. Instead, correction should stay within plausible confusion neighborhoods induced by realistic input errors and shaped by the language’s writing and input system. Based on this intuition, we propose GUIDE, a generative unsupervised framework built on a confuse-then-clarify paradigm. The core idea is to map confusable characters into shared-IDs and then train an encoder–decoder model to reconstruct the original character sequence. This shared-ID reconstruction objective turns character confusion structure into a learning signal: the encoder is encouraged to tolerate realistic ambiguity, while the decoder is forced to clarify it back into a concrete query. As a result, GUIDE learns controlled correction from unlabeled query streams without requiring manually annotated correction pairs.

For Chinese, these neighborhoods are naturally instantiated by the two dominant query errors: sound-alike substitutions and, to a lesser extent, look-alike ones (Liu et al., 2010; Wu et al., 2013a). These error patterns are closely tied to Chinese Input Method Editors (IMEs), such as pinyin and handwriting. Accordingly, we implement GUIDE with two complementary clustering strategies: phonetic shared-ID clustering based on pinyin confusability, and visual shared-ID clustering based on image-based character similarity. On top of these clustered inputs, we train an encoder–decoder Transformer with a time-decayed, query-frequencyweighted objective so that the model can continually adapt to recent and frequent queries in evolving search traffic.

Our contributions are as follows:

• We propose GUIDE, a Generative Unsupervised framework for Chinese query correction based on a confuse-then-clarify paradigm, where phonetic and visual shared-ID Encoding provides an explicit inductive bias for controlled correction under weak query context.

• We show how shared-ID reconstruction enables learning from unlabeled query streams without error–correction pairs.

• We release KwaiSearch, an in-house search-log dataset, and show strong results on both QSpell 250K and KwaiSearch, supported by online A/B testing.

## 2 Methodology

## 2.1 Problem Formulation and Overview

Let $\boldsymbol { \mathcal { D } } ~ = ~ \{ ( \boldsymbol { x } _ { i } , t _ { i } ) \} _ { i = 1 } ^ { N }$ denote a training set of N Chinese character queries collected from logs, where $\pmb { x } _ { i } = \{ x _ { i , 1 } , x _ { i , 2 } , \pmb { \cdot } \pmb { \cdot } \cdot \pmb { \cdot } , x _ { i , m _ { i } } \}$ is the observed (possibly incorrect) query at time $t _ { i }$ . The CQC task maps an input query $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ to a corrected query ${ \pmb y } _ { i } = \left\{ y _ { i , 1 } , y _ { i , 2 } , . . . , y _ { i , m _ { i } } \right\}$ , where ${ \bf { \nabla } } \mathbf { \mathbf { { y } } } _ { i }$ has the same length as $\mathbf { \nabla } _ { \mathbf { x } _ { i } }$ and differs only at erroneous character positions. In CQC, errors often arise from phonetically or visually similar candidates, with representative examples shown in Table 1.

<table><tr><td>Phonetically Similar Case</td></tr><tr><td>Input 土坡上的苟  $( \mathrm { g o u } )$  尾草。 PSC 苟(gǒu) 狗(gǒu) 够(gòu) 钩(gōu)</td></tr><tr><td>Visually Similar Case</td></tr><tr><td>Input 土坡上的狍(páo)尾草。</td></tr><tr><td>VSC 狍(páo) 狗(gǒu) 句(jù) 拘(jū)</td></tr><tr><td>Correct 土坡上的狗(gǒu)尾草。</td></tr><tr><td>Translation Bristlegrass on a Slope.</td></tr></table>

Table 1: Examples of phonetically similar candidates (PSC) and visually similar candidates (VSC) in Chinese spelling errors. Misspelled characters are marked in red, while the correct forms are indicated in blue. Phonetic errors are same-syllable confusions (often differing only in tone) from pinyin typing; visual errors are look-alike characters that typically share a radical/component.

To avoid unintended rewrites under weak query context, GUIDE constrains correction through character clustering and sequence reconstruction. As shown in Figure 1, the framework consists of two core stages: (1) character clustering, which maps an input query into a phonetic or visual shared-ID sequence, and (2) Chinese query correction, which reconstructs the original character sequence with an encoder–decoder Transformer.

## 2.2 Character Clustering

Character clustering groups Chinese characters by phonetic or visual similarity and maps each character to a shared ID. Formally, let $g ( \cdot )$ be a clustering-based mapping that converts a query $\mathbf { \delta } _ { \mathbf { \mathcal { X } } _ { i } }$ into a shared-ID sequence ${ \tilde { \pmb { x } } } _ { i } = { \pmb g } ( { \pmb x } _ { i } ) = \pmb { \mathrm { \Sigma } }$ $\{ \tilde { x } _ { i , 1 } , \ldots , \tilde { x } _ { i , m _ { i } } \}$ , where multiple confusable characters may share the same ID. This mapping is intentionally lossy: it removes fine-grained character identity at the encoder side, making the input more tolerant to typical user errors and restricting edits to plausible confusions. To cover dominant error types in practice, especially phonetic confusions and visually similar confusions, we design two complementary ways to construct g(·).

![](images/c44419819d0e2d695cea7e0b2cf08351b93c44ce5371c549c72fa5b7aa42b142.jpg)  
Figure 1: Overview of GUIDE with two clustering-based correction models: the left part shows the homophonic model, and the right part shows the visual-similarity model, based on pronunciation and character shape, respectively.

Homophonic Clustering. Most query typos on content platforms are sound-alike confusions caused by pinyin input. We therefore build a pinyinbased clustering strategy that maps each character to a pronunciation key without tones, and assigns the same shared ID to characters with the same tone–stripped pinyin. For polyphonic characters, we choose the most frequent tone–stripped pronunciation in query logs to determine the ID, which aligns the clusters with common user inputs and captures frequent phonetic confusions.

Visual Similarity Clustering. To handle another major error type—look-alike confusions—we cluster characters by visual similarity using imagebased character representations. We render each character as an image and encode it with a Vision Transformer (ViT) (Dosovitskiy et al., 2020) to obtain feature vectors, then cluster characters using cosine similarity between these representations. We use a threshold τ to determine whether two characters should be grouped into the same cluster, thereby controlling the granularity of visual neighborhoods.

## 2.3 Chinese Query Correction

We perform Chinese query correction with an encoder–decoder Transformer, where the encoder consumes shared-ID inputs and the decoder predicts the original character IDs. In the encoder stage, input query characters are mapped to shared-IDs, enabling the model to learn robust representations of error variants from large-scale, mostly error-free text data. The decoder then maps the shared-ID sequence back to the corresponding original character IDs. During inference, we decode the output sequence using beam search.

In practice, we train two separate correction models based on the two clustering strategies, namely the homophonic model and the visual-similarity model. The homophonic and visual-similarity models can be applied independently or combined. For deployment, we use a simple dynamic fusion without an explicit error-type classifier: for the same query, we feed the phonetic shared-ID sequence to the homophonic model and the visual shared-ID sequence to the visual-similarity model, and obtain decoder scores (token logits) for beam candidates from both models. We then compare the candidates from the two models and keep the one with the smaller score gap to the original query (i.e., the smaller change in normalized sequence log-likelihood relative to the original query), which helps prevent over-correction while still benefiting from beam search for plausible alternatives.

## 2.4 Training Objective with Time-decayed Frequency Reweighting

Given the shared-ID sequence $\tilde { \mathbf { x } } _ { i }$ obtained by the clustering-based mapping in Section 2.2, we then train GUIDE to reconstruct the original query $\pmb { x } _ { i } = \{ x _ { i , 1 } , . . . , x _ { i , m _ { i } } \}$ with a time-decayed frequency-reweighted negative log-likelihood:

$$
\mathcal { L } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } w _ { i } \sum _ { j = 1 } ^ { m _ { i } } \log p _ { \theta } ( x _ { i , j } \mid \tilde { \pmb { x } } _ { i } , x _ { i , < j } ) ,\tag{1}
$$

where $w _ { i } = \log ( 1 + c _ { i } ) \cdot \exp ( - \lambda \cdot \Delta t _ { i } )$ is a weight that depends on the query search count $c _ { i }$ and freshness $\Delta t _ { i }$ . This reweighting shifts learning focus toward frequent and recent queries while preserving the shared-ID reconstruction objective.

## 3 Experiments

This section evaluates GUIDE for CQC with both offline and online studies. We report offline results on the public QSpell 250K benchmark and an anonymized in-house dataset (KwaiSearch). We further validate GUIDE via an online A/B test on production traffic.

<table><tr><td rowspan="2"></td><td rowspan="2">Method</td><td rowspan="2">Model Size/Setting</td><td colspan="3">QSpell 250K</td><td colspan="3">KwaiSearch</td></tr><tr><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td rowspan="2">·dnS</td><td>BERT</td><td></td><td>|0.1891</td><td>0.4385</td><td>0.2642</td><td>0.0624</td><td>0.2330</td><td>0.0984</td></tr><tr><td>MASKED-FT</td><td></td><td>|0.2931</td><td>0.6247</td><td>0.3990</td><td>0.1603</td><td>0.5734</td><td>0.2505</td></tr><tr><td rowspan="9">·nsu·</td><td rowspan="2">SIMPLE-CSC</td><td>Qwen3-0.6B</td><td>0.3224</td><td>0.6693</td><td>0.4352</td><td>0.1450</td><td>0.6174</td><td>0.2348</td></tr><tr><td>Qwen3-1.7B</td><td>0.3282</td><td>0.7379</td><td>0.4543</td><td>0.1554</td><td>0.7156</td><td>0.2553</td></tr><tr><td>Qwen3-4B</td><td></td><td>0.3617</td><td>0.6950</td><td>0.4757</td><td>0.1732</td><td>0.6454</td><td>0.2730</td></tr><tr><td rowspan="6">LLM-ICL</td><td>Qwen3-0.6B</td><td>0-shot</td><td>|0.0365 0.0954</td><td>0.0528</td><td>0.0322</td><td>0.1290</td><td>0.0516</td></tr><tr><td>Qwen3-0.6B</td><td>10-shot</td><td>0.0352 0.1495</td><td>0.0569</td><td>0.0305</td><td>0.4424</td><td>0.0570</td></tr><tr><td>Qwen3-1.7B</td><td>0-shot 0.0786</td><td>0.3046</td><td>0.1249</td><td>0.0251</td><td>0.1859</td><td>0.0442</td></tr><tr><td>Qwen3-1.7B</td><td>10-shot</td><td>0.0975 0.5375</td><td>0.1650</td><td>0.0322</td><td>0.5010</td><td>0.0601</td></tr><tr><td>Qwen3-4B Qwen3-4B</td><td>0-shot 10-shot</td><td>0.1645 0.2620 0.2105</td><td>0.2021</td><td>0.0664 0.0730</td><td>0.1937</td><td>0.0989</td></tr><tr><td></td><td></td><td>0.5078</td><td>0.2976</td><td></td><td>0.3424</td><td>0.1203</td></tr><tr><td rowspan="3">GUIDE (ours)</td><td>3-layer 6-layer</td><td>Enc-Dec Enc-Dec</td><td>0.4244 0.4367</td><td>0.5341 0.5547</td><td>0.4730 0.4887</td><td>0.6738 0.6821</td><td>0.8476 0.8419</td><td>0.7508</td></tr><tr><td>12-layer</td><td>Enc-Dec</td><td>0.4321</td><td>0.5471</td><td>0.4829</td><td>0.6749</td><td>0.8364</td><td>0.7536</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>0.7474</td></tr></table>

Table 2: Main results on QSpell 250K and KwaiSearch. Sup./Unsup. denote supervised/unsupervised methods. Prec., Rec., and F1 denote precision, recall, and F1 score, respectively. bold indicates the best model for each dataset/metric.

## 3.1 Experimental Setup

Datasets. We evaluate GUIDE on two datasets that better reflect the practical CQC setting on content platforms. This differs from prior Chinese Spelling Correction (CSC) benchmarks (e.g., SIGHAN13/14/15 (Wu et al., 2013b; Yu et al., 2014; Tseng et al., 2015), Wang271K (Wang et al., 2018), and LEMON (Wu et al., 2023)), which are mostly sentence-oriented and less aligned with query noise patterns. Specifically, we use (i) QSpell 250K (Ye et al., 2025), a public simplified-Chinese query correction benchmark with short inputs; we train GUIDE using only the provided corrected queries; and (ii) KwaiSearch, an in-house dataset built from user search logs. It contains queries and short texts (e.g., titles and OCR-derived snippets) extracted from key search-result content, along with search-count and timestamp signals. <sup>1</sup> Table 3 summarizes the dataset statistics.

Evaluation Metrics. We report query-level Precision, Recall, and F1, which are standard in CSC.

<table><tr><td>Name</td><td>Usage</td><td>#Sent</td><td>#Error</td><td>Avg. Len.</td></tr><tr><td>QSpell 250K</td><td>Train Test</td><td>200K 50K</td><td>102K 26K</td><td>8.56 8.58</td></tr><tr><td>KwaiSearch</td><td>Train Test</td><td>180M 30K</td><td>Unknown 15K</td><td>9.67 7.35</td></tr></table>

Table 3: Statistics of all datasets. #Sent, #Error, and Avg. Len. denote the number of queries, error queries, and average query length, respectively.

Baseline Methods. We compare GUIDE with supervised and unsupervised baselines:

• BERT (Devlin et al., 2019): a supervised correction baseline that fine-tunes BERT-base-Chinese to predict corrected characters in context.

• MASKED-FT (Wu et al., 2023): a supervised fine-tuning strategy that masks non-error characters during training to reduce trivial copying and mitigate over-correction.

• SIMPLE-CSC (Zhou et al., 2024): a trainingfree, prompt-free LLM baseline that treats the LLM as a left-to-right language model and corrects text by constrained decoding, using a minimal-distortion constraint (phonetic/visual similarity) to keep the output close to the input.

• LLM-ICL: a prompting baseline using the Qwen3 series (Yang et al., 2025). We evaluate 0-shot and 10-shot settings on three model sizes (0.6B/1.7B/4B) with randomly sampled demonstrations.

<table><tr><td rowspan="2">Variant</td><td colspan="3">QSpell 250K</td><td colspan="3">KwaiSearch</td></tr><tr><td>Prec.</td><td>Rec.</td><td>F1</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Pho-only</td><td>0.4107</td><td>0.5230</td><td>0.4628</td><td>0.6661</td><td>0.8433</td><td>0.7443</td></tr><tr><td>Vis-only</td><td>0.0807</td><td>0.8152</td><td>0.1470</td><td>0.1326</td><td>0.7971</td><td>0.2274</td></tr><tr><td>Pho+Vis</td><td>0.42440.5341</td><td></td><td>0.4730</td><td>0.6738</td><td>0.8476 0.7508</td><td></td></tr></table>

Table 4: Comparison of clustering strategies in GUIDE. “Pho” and “Vis” denote the homophonic model and the visual-similarity model, respectively. “Pho+Vis” is the combined setting used in our main experiments.

Implementation Details. We instantiate GUIDE with three Transformer encoder–decoder sizes, using 3-, 6-, and 12-layer encoders and decoders, respectively. Unless otherwise specified, we use a single unified system that combines the homophonic and visual-similarity models. Since KwaiSearch has no ground-truth labels, supervised baselines are trained on QSpell 250K and evaluated on KwaiSearch. Detailed model configurations and training hyperparameters are provided in Appendix A.1.

## 3.2 Main Results

Table 2 reports results on QSpell 250K and KwaiSearch. GUIDE achieves the best overall performance on both datasets, with especially large gains on KwaiSearch. This suggests that constraining correction to phonetic/visual confusion neighborhoods is highly effective. On QSpell 250K, SIMPLE-CSC trades precision for recall, reflecting more aggressive rewriting behavior. In query recommendation, however, false corrections are often more harmful than missed typos, so the stronger precision–recall balance of GUIDE is better aligned with deployment needs. In-context prompting remains consistently weaker overall. Additional qualitative examples are provided in Appendix B.3.

## 3.3 Online A/B Testing

We further validate GUIDE in an online A/B test on real search traffic. Here we deploy the corrector at several high-traffic query entry points, where it corrects the candidate query set for recommendation: SUG search (search suggestions during typing), guess you search (pre-search query guesses surfaced to users), boxed-term search (search-box guesses of likely queries), and comment/bottombar guess search (lightweight guesses shown in comment or bottom-bar modules). As the production baseline, we compare with the previous online correction method. Compared with this baseline, GUIDE substantially improves userperceived quality: the misspelling rate drops by 80.1% (from 2.01% to 0.40%). We also observe consistent efficiency gains across multiple entry points, including SUG search PV (+0.152%), guess you search PV (+0.576%), boxed-term search PV (+0.707%), and comment/bottom-bar guess search PV (+0.986%), together leading to a +0.122% increase in overall search volume.

![](images/42d506003223b8c40d078167c2a4b6dd77a96f07a02431b10af2a6d04be2fc57.jpg)  
Clustering granularity (coarser finer)  
Figure 2: Sensitivity of GUIDE to phonetic and visual clustering granularity on KwaiSearch.

## 4 Discussion

## 4.1 Are Homophonic and Visual-Similarity Models Complementary?

To verify whether GUIDE benefits from modeling both dominant error sources in Chinese queries, we compare three settings: (i) only the homophonic model, (ii) only the visual-similarity model, and (iii) their combination (our default setting). As shown in Table 4, Pho+Vis achieves the best F1 on both datasets, indicating that the two clustering strategies are complementary. Overall, the homophonic model plays a more dominant role, while the visual-similarity model provides additional gains by covering look-alike errors that are not well captured by pronunciation cues.

## 4.2 How Sensitive Is GUIDE to Clustering Granularity?

We examine the effect of clustering granularity on correction quality. For homophonic clustering, we compare the default tone-stripped setting with two variants: a coarser pinyin setting and a finer tone-aware setting. The coarser setting merges common pinyin confusions that arise from regional accents in Mandarin and are directly reflected as typing errors in pinyin IMEs, including retroflex/non-retroflex initials ({z, zh}, {c, ch}, {s, sh}), nasal/lateral initials ({n, l}), labiodental confusions ({h, f}), and front/back nasal finals ({en, eng}, {in, ing}, {an, ang}). For visual clustering, we vary the threshold τ that controls neighborhood size. Figure 2 summarizes the trends, while the full numerical results are provided in Table 9. The default tone-stripped clustering gives the best overall F1 among phonetic variants, while τ = 0.5 performs best for visual similarity clustering. Overall, moderate confusion neighborhoods provide the best balance between coverage and noise.

## 5 Related Work

## 5.1 Chinese Spelling and Query Correction

Chinese spelling correction (CSC) is typically studied on sentence-level text with richer contextual information, whereas Chinese query correction (CQC) focuses on short search queries, where ambiguity is higher and vocabulary shift is faster. Because query correction directly affects retrieval and recommendation, CQC places stronger emphasis on intent preservation and control over editing behavior than general CSC.

Early correction methods often followed a noisychannel formulation with candidate generation and ranking (Brill and Moore, 2000). More recent CSC systems increasingly adopt PLM-based “detectthen-correct” architectures, such as Soft-Masked BERT (Zhang et al., 2020), to reduce unnecessary rewriting. For Chinese, phonetic and glyph similarity are especially important, and many models incorporate them through structured relations, multimodal features, including SpellGCN (Cheng et al., 2020) and ReaLiSe (Xu et al., 2021). Other efficient correction systems, such as FASPell (Hong et al., 2019), also show strong performance under realistic Chinese spelling noise, while newer datasets such as CSCD-NS broaden evaluation coverage (Hu et al., 2024).

Within the query setting, recent work highlights the mismatch between generic CSC assumptions and real search queries. Benchmarks such as QSpell 250K (Ye et al., 2025) better reflect naturally occurring query errors and rapidly changing entities. At the same time, recent studies show that strong language models can still over-correct short ambiguous inputs toward frequent or generic expressions (Li et al., 2023; Liu et al., 2024), motivating stronger mechanisms for edit control. To improve robustness on rare entities and shifting queries, some systems further incorporate retrieval or orchestration components, such as RACQC (Su et al., 2025) and Trigger<sup>3</sup> (Zhang et al., 2024).

Our work is related to these lines but differs in where the correction constraint is imposed. Prior CSC models typically inject phonetic or glyph knowledge into a supervised correction model, while training-free LLM approaches mainly constrain decoding at inference time. By contrast, GUIDE uses phonetic and visual similarity to define a shared-ID input space and learns correction through reconstruction from that lossy abstraction. This makes the confusion structure itself part of the training objective, rather than only an auxiliary feature or a decoding-time constraint.

## 5.2 Unsupervised Query Correction

To reduce reliance on annotated correction pairs, unsupervised and weakly supervised correction methods learn from raw text or logs via synthetic corruption, self-training, denoising, or masked reconstruction. A common strategy is to generate pseudo pairs by applying confusion-set substitutions or masking and then train a model to recover the original text (Liu et al., 2021; Li, 2022). Several studies further argue that realistic Chinese spelling errors are strongly shaped by the input process, and therefore simulate pinyin IME decoding to create pseudo data that better matches real error patterns (Hu et al., 2024). Other work reduces dependence on fixed confusion sets by using language-model scoring, self-supervised decoding, or inferencetime denoising (Jiang et al., 2024; Zhou et al., 2024). There are also flexible correction pipelines based on candidate evaluation with lightweight detectors and configurable candidate tables (Shao and Li, 2023).

Compared with these approaches, GUIDE takes a different view of unsupervised learning. Rather than constructing synthetic error–correction pairs, we learn directly from unlabeled query streams by mapping confusable characters into shared encoder IDs and training the model to reconstruct the original query. This confuse-then-clarify formulation avoids committing to a hand-designed corruption process while still imposing a strong inductive bias toward plausible IME-shaped edits. In this sense, GUIDE is closest in spirit to input-process-aware correction, but differs in using confusion neighborhoods as a lossy input abstraction and reconstruction signal for controlled generation.

## 6 Limitations and Future Work

GUIDE is designed for high-precision correction under weak query context, and this design choice also defines its current scope.

First, the framework focuses on lengthpreserving character substitution and does not directly address insertion, deletion, segmentation, or phrase-level rewriting errors. This is a practical trade-off: in our production setting, substitution errors—especially homophonic and visually similar ones—are among the most practically important query mistakes, making controlled substitution correction highly valuable. In practice, we view this as a deliberate decomposition choice: different query error types can be handled by the models or modules best suited to them, which is often a favorable design for industrial systems.

Second, although phonetic and visual shared-ID neighborhoods cover the dominant error sources in Chinese queries, they do not capture all realistic error types. Errors falling outside the constructed neighborhoods, or cases where the neighborhood itself is noisy, may still lead to missed corrections or over-correction. This is particularly relevant for visual clustering, whose neighborhoods can be substantially noisier than phonetic ones. At the same time, our results validate the effectiveness of this neighborhood-based design. While the current construction is not perfect, it can be further improved through continued refinement of clustering methods, and its failure cases remain relatively interpretable.

Third, our current deployment combines phonetic and visual signals through a simple heuristic selection strategy rather than a unified fusion model. We chose this decoupled design for stability and operational simplicity in production, but it leaves room for more principled integration, such as confidence-aware fusion, joint multimodal encoding, or learned routing between error types.

Finally, because our platform primarily serves Chinese queries, we focus this study on Chinese query correction. That said, the overall confuse-then-clarify paradigm is not inherently Chinese-specific, and can be generalized to other languages with language-appropriate neighborhood construction, such as phonological neighbors, keyboard-proximity errors, orthographic variants, or morphology-aware confusion structures. Exploring such extensions is a natural direction for future work.

## 7 Conclusion

We studied unsupervised Chinese query correction on content platforms, where weak context and IME-shaped noise make free-form rewriting especially prone to intent drift. GUIDE addresses this challenge through a confuse-then-clarify paradigm: by encoding queries in phonetic/visual shared-ID neighborhoods and reconstructing original character sequences, it enables controlled correction from unlabeled query streams. Across both offline benchmarks and online A/B testing, the results suggest that, for short-query correction, effective control over the edit space can be more important than unconstrained generation.

## Acknowledgments

This work was conducted at Kuaishou Technology. We thank the company for its support, and our colleagues on the search and recommendation team for helpful discussions and their assistance with data processing, deployment, and online experiments. We are especially grateful to Xuanping Li for valuable guidance and support throughout this work. We also thank the anonymous reviewers and area chairs for their constructive comments.

## References

Andrea Bacciu, Enrico Palumbo, Andreas Damianou, Nicola Tonellotto, and Fabrizio Silvestri. 2024. Generating query recommendations via LLMs. ArXiv:2405.19749.

Eric Brill and Robert C. Moore. 2000. An improved error model for noisy channel spelling correction. In Proceedings of the 38th Annual Meeting of the Association for Computational Linguistics, pages 286–293, Hong Kong. Association for Computational Linguistics.

Xingyi Cheng, Weidi Xu, Kunlong Chen, Shaohua Jiang, Feng Wang, Taifeng Wang, Wei Chu, and Yuan Qi. 2020. SpellGCN: Incorporating phonological and visual similarities into language models for Chinese spelling check. In Proceedings of the 58th Annual Meeting ofthe Associationfor Computational Linguistics, pages 871–881, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, and 1

others. 2020. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929.

Jianfeng Gao, Xiaolong Li, Daniel Micol, Chris Quirk, and Xu Sun. 2010. A large scale ranker-based system for search query spelling correction. In Proceedings of the 23rd International Conference on Computational Linguistics (Coling 2010), pages 358–366, Beijing, China. Coling 2010 Organizing Committee.

Yuzhong Hong, Xianguo Yu, Neng He, Nan Liu, and Junhui Liu. 2019. FASPell: A fast, adaptable, simple, powerful Chinese spell checker based on DAEdecoder paradigm. In Proceedings ofthe 5th Workshop on Noisy User-generated Text (W-NUT 2019), pages 160–169, Hong Kong, China. Association for Computational Linguistics.

Yong Hu, Fandong Meng, and Jie Zhou. 2024. CSCD-NS: a Chinese spelling check dataset for native speakers. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 146–159, Bangkok, Thailand. Association for Computational Linguistics.

Lai Jiang, Hongqiu Wu, Hai Zhao, and Min Zhang. 2024. Chinese spelling corrector is just a language learner. In Findings of the Association for Computational Linguistics: ACL 2024, pages 6933–6943, Bangkok, Thailand. Association for Computational Linguistics.

Piji Li. 2022. uChecker: Masked pretrained language models as unsupervised Chinese spelling checkers. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 2812–2822, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Yinghui Li, Haojing Huang, Shirong Ma, Yong Jiang, Yangning Li, Feng Zhou, Hai-Tao Zheng, and Qingyu Zhou. 2023. On the (in) effectiveness of large language models for chinese text correction. arXiv preprint arXiv:2307.09007.

Chao-Lin Liu, Min-Hua Lai, Yi-Hsuan Chuang, and Chia-Ying Lee. 2010. Visually and phonologically similar characters in incorrect simplified chinese words. In Coling 2010: Posters, pages 739–747, Beijing, China. Coling 2010 Organizing Committee.

Linfeng Liu, Hongqiu Wu, and Hai Zhao. 2024. Chinese spelling correction as rephrasing language model. Proceedings ofthe AAAI Conference on Artificial Intelligence, 38(17).

Shulin Liu, Tao Yang, Tianchi Yue, Feng Zhang, and Di Wang. 2021. PLOME: Pre-training with misspelled knowledge for Chinese spelling correction. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 2991–3000, Online. Association for Computational Linguistics.

Feiran Shao and Jinlong Li. 2023. Dual-detector: An unsupervised learning framework for chinese spelling check. In Pacific-Asia conference on knowledge discovery and data mining, pages 162–173. Springer.

Jinbo Su, Lingzhe Gao, Wei Li, Shihao Liu, Haojie Lei, Xinyi Wang, Yuanzhao Guo, Ke Wang, Daiting Shi, and Dawei Yin. 2025. RACQC: Advanced retrievalaugmented generation for Chinese query correction. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 675–689, Suzhou, China. Association for Computational Linguistics.

Xu Sun, Anshumali Shrivastava, and Ping Li. 2012. Fast multi-task learning for query spelling correction. In Proceedings ofthe 21st ACM International Conference on Information and Knowledge Management (CIKM’12), pages 285–294. ACM.

Yuen-Hsien Tseng, Lung-Hao Lee, Li-Ping Chang, and Hsin-Hsi Chen. 2015. Introduction to SIGHAN 2015 bake-off for Chinese spelling check. In Proceedings of the Eighth SIGHAN Workshop on Chinese Language Processing, pages 32–37, Beijing, China. Association for Computational Linguistics.

Dingmin Wang, Yan Song, Jing Li, Jialong Han, and Haisong Zhang. 2018. A hybrid approach to automatic corpus generation for Chinese spelling check. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2517–2527, Brussels, Belgium. Association for Computational Linguistics.

Hongqiu Wu, Shaohua Zhang, Yuchen Zhang, and Hai Zhao. 2023. Rethinking masked language modeling for Chinese spelling correction. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10743–10756, Toronto, Canada. Association for Computational Linguistics.

Shih-Hung Wu, Chao-Lin Liu, and Lung-Hao Lee. 2013a. Chinese spelling check evaluation at sighan bake-off 2013. In Proceedings of the Seventh SIGHAN Workshop on Chinese Language Processing, pages 35–42, Nagoya, Japan. Asian Federation of Natural Language Processing.

Shih-Hung Wu, Chao-Lin Liu, and Lung-Hao Lee. 2013b. Chinese spelling check evaluation at SIGHAN bake-off 2013. In Proceedings ofthe Seventh SIGHAN Workshop on Chinese Language Processing, pages 35–42, Nagoya, Japan. Asian Federation of Natural Language Processing.

Heng-Da Xu, Zhongli Li, Qingyu Zhou, Chao Li, Zizhen Wang, Yunbo Cao, Heyan Huang, and Xian-Ling Mao. 2021. Read, listen, and see: Leveraging multimodal information helps Chinese spell checking. In Findings ofthe Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 716–728, Online. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao,

Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Liner Yang, Xin Liu, Tianxin Liao, Zhenghao Liu, Mengyan Wang, Xuezhi Fang, and Erhong Yang. 2023. Is chinese spelling check ready? understanding the correction behavior in real-world scenarios. AI Open, 4:183–192.

Dezhi Ye, Haomei Jia, Junwei Hu, Tian Bowen, Jie Liu, Haijin Liang, Jin Ma, and Wenmin Wang. 2025. QSpell 250K: A large-scale, practical dataset for Chinese search query spell correction. In Proceedings of the 2025 Conference ofthe Nations ofthe Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 3: Industry Track), pages 148–155, Albuquerque, New Mexico. Association for Computational Linguistics.

Liang-Chih Yu, Lung-Hao Lee, Yuen-Hsien Tseng, and Hsin-Hsi Chen. 2014. Overview of SIGHAN 2014 bake-off for Chinese spelling check. In Proceedings of the Third CIPS-SIGHAN Joint Conference on Chinese Language Processing, pages 126–132, Wuhan, China. Association for Computational Linguistics.

Kepu Zhang, Zhongxiang Sun, Xiao Zhang, Xiaoxue Zang, Kai Zheng, Yang Song, and Jun Xu. 2024. Trigger<sup>3</sup>: Refining query correction via adaptive model selector. ArXiv:2412.12701.

Shaohua Zhang, Haoran Huang, Jicong Liu, and Hang Li. 2020. Spelling error correction with soft-masked BERT. In Proceedings ofthe 58th Annual Meeting of the Associationfor Computational Linguistics, pages 882–890, Online. Association for Computational Linguistics.

Houquan Zhou, Zhenghua Li, Bo Zhang, Chen Li, Shaopeng Lai, Ji Zhang, Fei Huang, and Min Zhang. 2024. A simple yet effective training-free promptfree approach to Chinese spelling correction based on large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 17446–17467, Miami, Florida, USA. Association for Computational Linguistics.

你是一个query纠错专家。请针对输  
入query，判断是否包含错别字，并  
进行合理纠错。  
Query: {text}  
Output:

示例：   
Query: ... × 10   
Output: ...   
Query: {text}   
Output:

## A Supplementary Experimental Settings

## A.1 Training Settings

This appendix provides the detailed training configurations for GUIDE. Unless otherwise specified, all reported results use the same training recipe across model sizes.

Hardware. We train GUIDE on a single machine equipped with 4× NVIDIA L20 GPUs. Mixed-precision training is enabled.

Model Configuration. GUIDE uses a standard Transformer encoder–decoder architecture. We evaluate three sizes with {3, 6, 12} encoder layers and {3, 6, 12} decoder layers, respectively. For all variants, we use d\_model = 768, d\_ff = 3072, # heads = 12, dropout = 0.1, and label smoothing = 0.1. The vocabulary consists of character IDs, and the encoder input is the shared-ID sequence produced by the clustering module (Section 2.2).

Optimization. We optimize the time-decayed, frequency-reweighted negative log-likelihood objective in Eq. 1. We use AdamW with $\beta _ { 1 } = 0 . 9$ $\beta _ { 2 } = 0 . 9 9 9 , \epsilon = 1 \times 1 0 ^ { - 5 }$ , weight decay = 0.01, and gradient clipping norm=1.0. The base learning rate is $5 \times 1 0 ^ { - 5 }$ , with a warmup of 5% steps of the total steps followed by a cosine decay schedule (e.g., inverse-square-root / cosine decay). Training runs for 2 epochs, with an effective batch size of 512 queries (micro-batch size 128 per GPU and gradient accumulation steps 4). We set the maximum input length to 16 characters and truncate longer sequences.

Data and Reweighting. For each training example $( \pmb { x } _ { i } , t _ { i } )$ , the shared-ID input $\tilde { \mathbf { x } } _ { i }$ is constructed by replacing each character with its phonetic/visual shared-ID (Section 2.2). The weight is $\begin{array} { r l } { w _ { i } } & { { } = } \end{array}$ log $\left( 1 + c _ { i } \right) \cdot \exp ( - \lambda \Delta t _ { i } )$ , where $\lambda = 0 . 0 0 7 7$ and $\Delta t _ { i }$ is measured in 90 days. We sample training data proportionally to $w _ { i }$ , and update weights every week.

Decoding and Checkpoint Selection. At inference time, we use beam search with beam size 10, length penalty 1.0, and maximum decoding length $m _ { i }$ (equal to the input length). For the unified system, we run both phonetic-shared-ID and visual-shared-ID encodings and select the encoding whose best beam candidate yields the smaller score gap to the original query (Section 2.3). We select the final checkpoint using validation query-level F1 (or NLL) on a held-out set, evaluated every 2000 steps.

## A.2 Baseline Settings

For supervised and fine-tuning baselines, we follow the original implementations and report the results in Section 3. For LLM-ICL, we use the Qwen3 series (0.6B/1.7B/4B) under 0-shot and 10-shot settings with randomly sampled demonstrations, using the prompt templates shown below.

Prompt Templates (LLM-ICL).   
0-shot.

English translation: “You are a query-correction expert. For the input query, decide whether it contains misspelled characters and correct them appropriately.”

你是一个query纠错专家。请针对输入query，进行合理的文本纠错，输出纠错后文本，禁止输出任何思考过程。

English translation: “You are a query-correction expert. For the input query, perform reasonable text correction and output only the corrected text; do not output any reasoning process.” (“示例” = “Examples”.)

## B Additional Experimental Results

## B.1 Ablation on the Training Objective and Sensitivity to λ

We ablate the two components of our training objective in Eq. (1) on KwaiSearch to isolate their contributions. Starting from a uniform objective, adding query-frequency reweighting alone improves F1 by about 11 points, and further adding the time-decay term yields an additional gain of about 4.6 points (Table 5). This confirms that frequency reweighting is the dominant factor, while time decay provides a consistent additional benefit by shifting learning focus toward recent queries.

<table><tr><td>Objective</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Uniform</td><td>0.5503</td><td>0.6470</td><td>0.5947</td></tr><tr><td>Frequency-only</td><td>0.6314</td><td>0.7974</td><td>0.7048</td></tr><tr><td>Full  $( \mathrm { t i m e \mathrm { - } d e c a y + f r e q . } )$ </td><td>0.6738</td><td>0.8476</td><td>0.7508</td></tr></table>

Table 5: Ablation of the training objective on Kwaisearch (3-layer model). Frequency reweighting alone adds ∼11 F1 points over the uniform objective, and time decay adds a further ∼4.6 points.
<table><tr><td>λ</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>0.0098 (ours)</td><td>0.6738</td><td>0.8476</td><td>0.7508</td></tr><tr><td>0.0133 (aggressive)</td><td>0.6680</td><td>0.8422</td><td>0.7451</td></tr><tr><td>0.0043 (conservative)</td><td>0.6817</td><td>0.7974</td><td>0.7350</td></tr></table>

Table 6: Sensitivity to the time-decay coefficient λ on Kwaisearch (3-layer model). Targeting 50% overlap (0.0133) or 80% overlap (0.0043) keeps F1 within ∼1.6 points of our default $\lambda = 0 . 0 0 9 8$

Importantly, λ is not tuned on the test set. We instead set it from a simple operational assumption: with an approximately 60% one-year query overlap and weekly model updates, requiring e $\cdot \lambda \cdot 5 2 = 0 . 6$ over the 52 weekly steps of a year gives $\lambda \ =$ 0.0098, which is the value used throughout our experiments. To show that this choice is robust, we also report more aggressive and more conservative alternatives obtained by targeting a 50% overlap $( \lambda = 0 . 0 1 3 3 )$ or an 80% overlap $( \lambda = 0 . 0 0 4 3 )$ . As shown in Table 6, both variants keep F1 within about 1.6 points of our default, indicating that GUIDE is not sensitive to the exact value of λ.

## B.2 Selection of the Visual Clustering Threshold τ

The visual clustering threshold τ is selected on a separate held-out development set consisting purely of visual errors, rather than on the Kwaisearch test set. We sweep τ for the visual-similarity model on this development set and report the results in Table 7. The threshold $\tau = 0 . 5$ is optimal on the development set, and we keep this value throughout all experiments. This makes the selection protocol explicit and ensures that τ is not tuned on any test data.

## B.3 Case Studies

We provide qualitative examples to complement the quantitative results in Section 3. Table 8 includes both representative good cases and bad cases for GUIDE. In good cases, GUIDE fixes typical IME-shaped confusions (sound-alike or lookalike) while preserving the remaining characters, producing outputs that better match the intended query. In bad cases, two common failure modes are observed: (i) over-correction, where a rare but valid query is rewritten toward a more frequent alternative, and (ii) missed correction, where the error lies outside the clustering neighborhood or requires extra context beyond the query itself. For instance, GUIDE corrects “时间同步板远离” → “时间同步板原理” and “臧易通怎么查核酸报 告” → “藏易通怎么查核酸报告”, but may overcorrect “播音苏杨” to the frequent phrase “播音素 $\yen 1$ , or keep “name英标怎么写” unchanged when the cue is weak; this mainly happens because short queries provide little context, so the model tends to favor high-probability frequent phrases unless the phonetic/visual neighborhood offers a strong and unambiguous correction signal.

<table><tr><td>Threshold τ</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>0.3 0.4 0.5 (selected) 0.5999</td><td>0.5974 0.5969</td><td>0.7613 0.7774 0.7890</td><td>0.6695 0.6753 0.6816</td></tr></table>

Table 7: Selection of the visual clustering threshold τ on a held-out development set of visual errors only. $\tau = 0 . 5$ is optimal and is used throughout.

<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=2>| Good (Pho)       |Good (Vis)</td></tr><tr><td rowspan=1 colspan=1>OriginalTarget</td><td rowspan=1 colspan=1>时间同步板远离时间同步板原理</td><td rowspan=1 colspan=1>臧易通怎么查核酸报告藏易通怎么查核酸报告</td></tr><tr><td rowspan=1 colspan=1>GUIDEBERTMASKED-FTSIMPLE-CSCLLM-ICL</td><td rowspan=1 colspan=1>时间同步板原理时间同步板远离时间同步被远离时间同步板远离时间同步板远离</td><td rowspan=1 colspan=1>藏易通怎么查核酸报告交易通怎么查核酸报易易通怎么查核酸报告臧易通怎么查核酸报告臧易通如何查询核酸报告</td></tr><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>| Bad (Over)</td><td rowspan=1 colspan=1>| Bad (Miss)</td></tr><tr><td rowspan=1 colspan=1>OriginalTarget</td><td rowspan=1 colspan=1>播音苏杨播音苏杨</td><td rowspan=1 colspan=1>name英标怎么写name音标怎么写</td></tr><tr><td rowspan=1 colspan=1>GUIDEBERTMASKED-FTSIMPLE-CSC</td><td rowspan=2 colspan=1>播音素养播音苏杨播音苏扬播音苏扬播音苏扬</td><td rowspan=2 colspan=1>name英标怎么写name英标怎么写name英标怎么写name英标怎么写name的英标是/nm/</td></tr><tr><td rowspan=1 colspan=1>LLM-ICL</td></tr></table>

Table 8: Qualitative case studies on representative query correction examples. We show the original query, the target correction, and the outputs of GUIDE and baseline methods for two good cases (Pho/Vis) and two failure cases (Over/Miss).

<table><tr><td>Setting</td><td>Prec.</td><td>Rec.</td><td>F1</td></tr><tr><td>Phonetic clustering</td><td></td><td></td><td></td></tr><tr><td>Coarse pinyin clusters Default homophonic clustering</td><td>0.6020</td><td>0.7650</td><td>0.6738 0.7443</td></tr><tr><td>Tone-aware clustering</td><td>0.6662 0.4585</td><td>0.8433 0.8266</td><td>0.5898</td></tr><tr><td>Visual clustering threshold</td><td></td><td></td><td></td></tr><tr><td>τ = 0.3</td><td>0.1245</td><td>0.7859</td><td>0.2149</td></tr><tr><td>τ = 0.4</td><td>0.1267</td><td>0.7909</td><td>0.2184</td></tr><tr><td>τ = 0.5</td><td>0.1326</td><td>0.7971</td><td>0.2274</td></tr><tr><td>τ = 0.6</td><td>0.1311</td><td>0.7938</td><td>0.2250</td></tr></table>

Table 9: Full numerical results for the clusteringgranularity analysis on KwaiSearch.
<table><tr><td>Month</td><td>09-01</td><td>10-01</td><td>11-01</td><td>12-01</td></tr><tr><td>Misspelling rate</td><td>2.01%</td><td>0.40%</td><td>0.46%</td><td>0.33%</td></tr></table>

Table 10: Post-rollout monthly tracking of the online misspelling rate after the full rollout of GUIDE (end of September 2025). The reduction is stable and sustained.

## C Online A/B Testing Details

The production baseline in our online A/B test (Section 3.3) is an LLM-based corrector: a Qwen model fine-tuned on tens of thousands of manually annotated correction pairs, using the same prompt as provided in this paper (Appendix A.2). We compare GUIDE against this deployed LLM-based correction model.

We run a bucketed A/B test from 2025-09-13 to 2025-09-22 (10 days) on 4.2% of production main-search traffic. Our single-day search page views (PV) are on the order of tens of millions, and the downstream query-recommendation scenarios fed by the corrector carry far larger PV than search itself; at this scale, the A/B conclusions are highly confident. The misspelling rate is measured by random sampling of online queries followed by manual annotation. During the bucketed test, GUIDE reduces the misspelling rate from 2.58% (base) to 0.86% (exp), and yields a +0.122% overall search-volume lift.

GUIDE was fully rolled out at the end of September 2025. We continue to monitor the online misspelling rate by month, as reported in Table 10. The reduction remains stable across subsequent months rather than being a transient effect, confirming a sustained improvement in correction quality.