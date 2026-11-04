# Constructing a Regional Slang Lexicon for Electricity Demands via Soft-Label Bidirectional Anchoring and Active Learning

Rong Chang<sup>1</sup>, Lijing Shi<sup>1</sup>, Zhaoxiang Zang<sup>2,3\*</sup>

<sup>1</sup>Yuxi Power Supply Bureau, Yunnan Power Grid Co., Ltd., Yuxi, 653100, China.

<sup>2\*</sup>Faculty of Information Engineering and Automation, Kunming University of Science and Technology, Kunming, 650504, China. <sup>3</sup>Key Laboratory of Artificial Intelligence in Yunnan Province, Kunming University of Science and Technology, Kunming, 650504, China.

\*Corresponding author(s). E-mail(s): zhaoxiang.zang@kust.edu.cn; Contributing authors: rong chang007@126.com; joshua.zang@outlook.com;

## Abstract

Power-grid customer service increasingly relies on community social media channels, where customers describe outages, billing issues, and equipment faults in informal regional dialect. These short, noisy messages exhibit non-standard morphology, semantic drift, and out-of-vocabulary expressions that make manual lexicon construction costly and degrade conventional NLP tools. This paper proposes a weak-supervision framework for constructing a regional slang lexicon from curated dialect anchors and power-domain seed terms. The framework generates candidates via Pointwise Mutual Information with soft-label bidirectional anchoring, retaining dialect-anchored terms while recording power-domain relevance as separate metadata. Candidates are then scored by combining Sentence-BERT semantic similarity with a jieba-based wordhood measure of lexical fixedness. Uncertainty-aware active learning with rule-based and retrieval-augmented generation (RAG) LLM oracles is applied under three query strategies, and the resulting lexicon is independently evaluated by a higher-capability LLM calibrated against human expert annotations via Cohen’s κ. Experiments on 11,177 real WeChat messages from Yuxi (Yunnan, China) achieve an evaluatorestimated dialect precision of DP=1.0 with zero false positives. A controlled ablation shows that soft-label layering retains 751% more candidates than strict

power-gating at identical precision, expanding the candidate pool from 41 to 349 terms. Human calibration $( n = 4 6 )$ yields $\kappa _ { \mathrm { d i a l e c t } } = 1 . 0$ and $\kappa _ { \mathrm { p o w e r } } = \mathbf { 0 . 8 5 } ,$ indicating perfect agreement on dialect identification and strong agreement on power-domain relevance. The resulting lexicon, comprising 25–33 dialect terms per configuration of which 4–5 are additionally power-relevant, provides a preliminary resource for dialect-aware power-grid customer service NLP, though practical utility remains to be demonstrated through task-based evaluation.

Keywords: Regional dialect lexicon; bidirectional anchoring; soft-label layering; Pointwise Mutual Information; active learning; power-grid customer service; LLM evaluation; short text mining; weak supervision

## 1 Introduction

## 1.1 Background and Motivation

The digital transformation of power-grid customer service under the Smart Grid 2.0 paradigm has changed how utilities interact with customers. In China, the State Grid Corporation and China Southern Power Grid have increasingly extended customer service from traditional voice hotlines such as 95598 to localized, community-based social media groups, especially WeChat communities organized by neighborhood, village, or transformer service area (Zonta et al. 2020; Chao 1968). This shift improves accessibility and response speed, but it also creates new challenges for automated intention recognition systems that were originally designed for formal and structured language.

The main dificulty is linguistic. In ethnically and linguistically diverse regions such as Yuxi City in central Yunnan Province, customers often report power outages, equipment faults, and billing inquiries using local dialectal expressions that difer from standard Mandarin in morphology, syntax, and semantics (Chao 1968; Norman 1988). The Yuxi dialect, a variety of Southwestern Mandarin, includes distinctive morphological markers (e.g., sentence-final particles such as 呢(ne), 噶(ga), 嘛(ma), and 哈(ha)), function words (e.g., the question marker 给有(gei you, “is there?”) and the intensifier 扎实(zhashi, “very” or “severe”)), and cases of semantic drift in which standard Chinese lexemes acquire domain-specific meanings (Zhang 1985). Table 1 illustrates representative cases of this phenomenon in power-grid contexts.

These expressions carry task-relevant semantic cues about fault severity (扎 实(zhashi) and 老火(laohuo)), user emotional state (鬼火(guihuo)), and request type (给有(gei you)). For example, a customer service NLP system that interprets 线路老 火(xianlu laohuo, literally “line old fire”) rather than “severe line fault” may misclassify complaint urgency and route the request incorrectly. Reliable handling of dialectal expressions is therefore important not only for model robustness, but also for maintaining service quality across linguistically diverse user populations (Blodgett et al. 2020; Bender et al. 2021).

Table 1 Semantic drift in Yuxi dialect terms within power-grid service contexts. Pinyin is provided for Chinese terms, and each example includes an English translation.
<table><tr><td>Chinese (pinyin)</td><td>term</td><td>Literal meaning</td><td>Power-grid mean- ing</td><td>Translated example</td></tr><tr><td>扎实(zhashi)</td><td>firm</td><td>Sturdy, solid,</td><td>Severe, very, intense</td><td>“雷打得扎实，变压器都跳了”(The thunder was intense; the transformer tripped).</td></tr><tr><td>鬼火(guihuo)</td><td></td><td>will-o&#x27;-the-</td><td>Phosphorescence, Anger, frustration, annoyance</td><td>“三天两头停电，哪个不鬼火?”(Outages every few days; who would not be frus- trated?).</td></tr><tr><td>给有(gei you)</td><td></td><td>wisp To give, to have</td><td>Is there? Does it exist? (question</td><td>“供电局呢领导，给有电了嘛?”(Power bureau leaders, is there electricity yet?).</td></tr><tr><td>着炸(zhao zha)</td><td></td><td>Affected + to explode</td><td>marker) Damaged by electri- cal surge, shorted</td><td>“表箱着炸掉了，快点来看哈”(The meter box was damaged by a surge; please come</td></tr><tr><td>老火(laohuo)</td><td></td><td>Old fire</td><td>Severe, terrible, crit- ical (intensifier)</td><td>and check it). “线路老火，快派人来修”(The line fault is severe; please send someone to repair it).</td></tr></table>

## 1.2 Technical Challenges

Mining regional electricity slang from WeChat group messages presents three technical challenges that distinguish this setting from prior work on domain-specific NLP.

The first challenge is data noise and sparsity. Unlike formal corpora such as customer service hotline transcripts or technical maintenance logs, WeChat group messages contain heterogeneous noise that can be grouped into three categories. Structural noise originates from the WeChat platform itself, including system-generated messages (e.g., “拍了拍” (pai le pai, profile tap notification) and “邀请加入了群聊” (invited a user to join the group chat)), XML-formatted metadata (<msg>...</msg> tags), wxid user identifiers embedded in export artifacts, and application-specific formatting that survives CSV export (Eisenstein 2013). Irrelevant noise includes cross-domain content that appears in power service groups, such as commercial advertisements, community administrative notices, public health announcements, and casual social chatter unrelated to electricity service (Baldwin et al. 2013). Visual noise arises from rapid mobile input, including typographical errors, homophone substitutions, and bracket-delimited emoji placeholders such as [捂脸] ([facepalm]) (Eisenstein 2013). In aggregate, structural noise accounts for over 30% of raw message volume, and the overall signal-to-noise ratio—the fraction of messages containing genuine electricity-related content—falls below 5% (Derczynski et al. 2013).

The second challenge is term ambiguity in short texts, a well-documented phenomenon in short-text stream clustering (Liang et al. 2016; Li et al. 2016). A single WeChat message typically contains 10–30 Chinese characters, leaving limited context for disambiguation. The Yuxi dialect intensifier 扎实(zhashi), which means “severe,” “very,” or “intense” in power-grid complaints, retains the standard Mandarin meaning of “sturdy” or “solid” in other domains. A message such as “扎实呢停电” (zhashi ne tingdian, roughly “the outage is severe”) therefore requires the system to recognize both the dialect particle 呢(ne) and the context-dependent meaning of 扎实, a combination that is dificult to resolve without domain-specific anchoring (Pennington et al. 2014).

The third challenge is the out-of-vocabulary (OOV) and few-shot dilemma. General-purpose pre-trained language models, including Chinese RoBERTa variants and multilingual BERT, are trained mainly on standard Mandarin corpora from news, encyclopedias, and formal publications (Cui et al. 2021; Devlin et al. 2019). Dialectspecific morphology, character substitutions, and regional expressions are often poorly represented in these corpora, which can degrade zero-shot transfer performance (Pires et al. 2019). Although fine-tuning on domain-specific annotated data could mitigate this problem, constructing a suficiently large labeled corpus for a low-resource dialect is costly (Hedderich et al. 2021; Lauscher et al. 2020). As a result, standard NLP pipelines deployed in dialect-heavy regions may perform substantially worse than they do on standard-language benchmarks (Joshi et al. 2020).

## 1.3 Limitations of Existing Approaches

Existing approaches to power-grid NLP have primarily focused on structured, formallanguage settings. Prior work has established foundations for named entity recognition (NER) and dispatch classification using annotated transcripts of 95598 hotline calls, which are characterized by a professional register, standard terminology, and relatively clean transcription. Multi-task learning frameworks have been proposed for few-shot power dispatch NER (Tan et al. 2023), and deep active learning algorithms have been applied to Chinese power text classification to balance annotation cost and model accuracy (Deng et al. 2024). These methods generally assume that input text follows formal register conventions and that vocabulary is drawn from standard technical terminology with manageable OOV rates. These assumptions are problematic for WeChat group messages, where dialect-inflected expressions create vocabulary coverage gaps on the order of 15–25% (Eisenstein 2013).

Beyond the power domain, computational approaches to dialect processing have focused mainly on dialect identification, namely classifying text by regional origin, rather than on the constructive problem of building a dialect lexicon from unlabeled text (Nguyen et al. 2016; Eisenstein 2017). Studies in Arabic dialectology (Zampieri et al. 2014), Swiss German processing, and Chinese dialect identification (Reimers and Gurevych 2019) show that statistical features can detect regional variation, but these methods typically assume the availability of reference resources and do not address how to bootstrap a lexicon from noisy, unlabeled data. Weak supervision techniques (Tok et al. 2021; Poyiadzi et al. 2022) and data programming frameworks (Tok et al. 2021; Ratner et al. 2016) provide foundations for generating training signals without full manual annotation, but their application to dialect lexicon construction remains insuficiently explored.

## 1.4 Proposed Approach and Contributions

We formulate regional slang lexicon construction as a noisy short-text semantic expansion problem and propose a four-stage weak-supervision framework that uses a small set of curated dialect anchors and power-domain seed terms rather than a fully labeled training corpus. The key idea is that verified dialect anchor words, i.e., morphemes known to be characteristic of the target dialect, can serve as statistical anchors for discovering additional dialect expressions through Pointwise Mutual Information (PMI) (Church and Hanks 1990). This anchor-driven design is combined with a soft-label bidirectional anchoring mechanism that records power-domain relevance without using it as an early filtering condition. In this way, the framework separates noise elimination from relevance assessment, two tasks that are often conflated in hard-gating approaches.

The pipeline proceeds in four stages. Stage 1 extracts bigram candidates using PMI anchored on 16 Yuxi dialect markers, with bidirectional anchoring, soft-label powerdomain layering, and a four-condition fragment filter used to remove grammatical constructions while preserving lexical compounds. Stage 2 classifies candidates using a dual-channel score that combines Sentence-BERT semantic similarity to 65 powerdomain seed terms with a jieba-based wordhood score that measures lexical fixedness through segmentation resistance (Reimers and Gurevych 2019; Johnson et al. 2021). Stage 3 performs uncertainty-aware active learning with three query strategies (Margin, TD-CAL, Random) (Settles 2012; Lewis and Gale 1994; Settles and Craven 2008) and two oracle types: rule-based morphological matching and RAG-augmented LLM assessment (Lewis et al. 2020) using DeepSeek V4 Flash (Deng et al. 2025). Stage 4 evaluates the constructed lexicon with a separate, higher-capability LLM (DeepSeek V4 Pro) calibrated against human expert annotations via Cohen’s κ (Cohen 1960), while retaining support for cross-model evaluation using a diferent model family.

We evaluate the framework on 11,177 real WeChat messages from Yuxi power-grid customer service groups and use a controlled ablation study to isolate the contribution of soft-label layering. The study makes four contributions. First, it introduces a bidirectional anchoring framework with soft-label power-domain layering that reaches an LLM-evaluator-estimated dialect precision of $\mathrm { D P = } 1 . 0$ while increasing candidate coverage by 751% over strict power-gating. Second, it presents a dual-track LLM evaluation system with an independent evaluator model calibrated against human expert annotations, yielding degenerate perfect agreement for dialect identification $( \kappa _ { \mathrm { d i a l e c t } } = 1 . 0 )$ and strong agreement for power-domain relevance $( \kappa _ { \mathrm { p o w e r } } = 0 . 8 5$ $n = 4 6 )$ , while noting that cross-model validation remains future work. Third, it compares two oracle types across three active learning query strategies and shows that oracle quality afects strategy diferentiation: under the high-acceptance LLM oracle (∼89%), all configurations converge to 4 power-dialect terms, whereas modest differentiation appears under the more selective rule-based oracle. Fourth, the ablation shows that soft-label layering preserves many dialect bigrams discarded by hard gating while maintaining the same dialect precision, suggesting a practical design principle for weak-supervision pipelines in low-resource settings.

## 2 Related Work

## 2.1 Natural Language Processing in Power Systems

NLP has been increasingly applied to power systems engineering and customer service as utilities accumulate unstructured text from maintenance logs, fault reports, hotline transcripts, and online service channels. Early work on message understanding and information extraction provides a methodological background for structured extraction from operational text (Grishman and Sundheim 1996). More recent work has introduced annotated hotline transcript datasets with entity and dispatch category labels, enabling baseline comparisons for architectures such as BiLSTM-CRF and BERT-based classifiers.

Several extensions have addressed limitations of these baselines. Multi-task learning frameworks combining NER with relation extraction have improved nested entity recognition in power dispatch contexts (Tan et al. 2023), while domain-adaptive pretraining can reduce vocabulary mismatch between general-domain language models and domain-specific terminology (Gururangan et al. 2020). In fault diagnosis, deep active learning has reduced annotation requirements for power equipment fault classification while maintaining accuracy above 90% (Deng et al. 2024). Knowledge-enhanced methods that incorporate power-grid topology into text encoders further support complex fault reasoning tasks.

However, most power-grid NLP studies assume text that is relatively clean, formally structured, and drawn from standard technical vocabulary. Hotline transcripts, maintenance logs, and dispatch records are typically produced in professional settings with controlled terminology and standardized reporting formats. Community chat logs difer in both form and content: they contain non-standard syntax, regional dialect, code-switching, emoji usage, and substantial topic drift (Baldwin et al. 2013). This diference is qualitative rather than only quantitative, because informal community messages introduce diferent noise distributions and diferent model failure modes.

## 2.2 Short Text Processing and Statistical Association Measures

The processing of short and sparse documents has been widely studied in information retrieval and text mining. Short texts, often defined as documents with fewer than 50 words, provide limited within-document co-occurrence evidence and therefore degrade the performance of standard topic models and clustering methods (Liang et al. 2016). The Topic-Enhanced Dirichlet Model (TEDM) addresses this problem by incorporating auxiliary word embeddings into topic inference, showing that external semantic knowledge can compensate for sparse local evidence (Li et al. 2016). The Online Semantic-Enhanced Graphical Model (OSGM) extends this idea to streaming scenarios by dynamically maintaining topic distributions as new short documents arrive (Liang et al. 2016). These studies motivate the use of external semantic information in short-text analysis, which in our setting is provided by power-domain seed terms and dialect anchors.

Statistical association measures are central to unsupervised term extraction and collocation discovery. Since the work of Church and Hanks on word association norms Church and Hanks (1990), Pointwise Mutual Information (PMI) has been widely used to measure statistical dependence between two words by comparing observed co-occurrence with the expected frequency under independence. In computational lexicography, PMI helps identify multi-word expressions, technical terms, and domainspecific collocations that may be missed by frequency-based methods alone (Evert

2005). Its normalized variant, NPMI, bounds association scores to [−1, 1], which facilitates cross-corpus threshold selection and improves robustness to corpus size variation (Bouma 2009).

In Chinese NLP, PMI-based methods face an additional dificulty because word boundaries are not explicitly marked. Segmentation is therefore required before association scores can be computed, and segmentation errors, especially on OOV terms and dialect expressions, can propagate into PMI estimation (Zhao et al. 2007). The jieba tokenizer provides a practical balance between speed and accuracy through its prefix dictionary and Hidden Markov Model-based OOV handling (Sun 2012). Our pipeline builds on jieba but explicitly registers dialect anchor words and power-domain seeds before tokenization to reduce fragmentation of dialect expressions.

## 2.3 Active Learning for Low-Resource Natural Language Processing

Active learning (AL) refers to a family of methods that selectively query informative unlabeled instances for annotation, with the goal of improving models under limited labeling budgets (Settles 2012). In the standard pool-based setting, a model is trained on a small labeled set, a query strategy selects instances from a larger unlabeled pool for oracle annotation, and the newly labeled instances are added to the training set. This cycle continues until the annotation budget is exhausted or model performance saturates.

The choice of query strategy afects AL eficiency. Uncertainty sampling selects instances for which the current model is least confident, using measures such as entropy, least confidence, or margin (Lewis and Gale 1994). Margin sampling selects instances with the smallest diference between the top two predicted class probabilities and is often efective for binary classification with a well-defined decision boundary (Lewis and Gale 1994; Tong and Koller 2000). Density-weighted methods add a representativeness criterion, preferring instances that are both uncertain and located in dense regions of the feature space (Settles and Craven 2008). Token-Density Contrastive Active Learning (TD-CAL) follows this principle by combining a margin-based uncertainty term with a density term based on token frequency and document entropy. In our setting, however, the standard TF-IDF-based density signal is nearly constant for rare dialect bigrams, all of which appear 2–3 times in the corpus. We therefore use PMI as a more discriminative density proxy, with values ranging from 2.4 to 11.8 for attested bigrams.

AL has been studied across NLP tasks including text classification (Tong and Koller 2000), named entity recognition (Shen et al. 2017), relation extraction, and machine translation (Hafari et al. 2009). Deep active learning combines AL with pre-trained representations, often coupling the query strategy with fine-tuning of Transformer-based models (Ein-Dor et al. 2020). In the power domain, Deng et al. Deng et al. (2024) showed that uncertainty-based AL with BERT embeddings reduced annotation requirements by approximately 60% for Chinese power text classification while maintaining accuracy above 90%. Across these studies, oracle quality and consistency are important: noisy or inconsistent oracles reduce the informativeness of selected instances and weaken the benefit of active sampling (Settles 2012). Our experiments examine this issue in an LLM-oracle setting, where the RAG-LLM oracle’s high acceptance consistency (∼89%) reduces diferentiation among query strategies.

## 2.4 Computational Dialectology and Lexicon Construction

Computational studies of dialect variation have expanded over the past decade with the availability of geo-located social media data and advances in representation learning (Nguyen et al. 2016). The dominant task has been dialect identification: given a text sample, determine which regional variety of a language it represents. Shared tasks such as the DSL (Discriminating between Similar Languages) challenge (Zampieri et al. 2014) and the VarDial evaluation campaign (Zampieri et al. 2017) have advanced methods based on character n-gram features, word embeddings, and neural architectures for dialect classification across Arabic, Spanish, Portuguese, and Chinese varieties.

Dialect feature detection is a more fine-grained task that identifies the morphological, lexical, or syntactic markers associated with a dialect. Work on Swiss German dialects has shown that neural models can detect regional morphological patterns from raw text, while studies in Arabic dialectology (Darwish et al. 2012) and Chinese dialect variation indicate that character-, word-, and syntax-level features provide comple mentary signals for regional classification. However, these methods usually operate in supervised or semi-supervised settings that require annotated dialect data, which is limited in the setting considered here.

The constructive problem of building a dialect lexicon from unlabeled text has received less attention. Traditional dialect lexicon construction often relies on manual curation by domain experts, as illustrated by regional vocabulary compilations such as the Yuxi Dialect Vocabulary Compilation (Zhang 1985). This approach is precise but dificult to scale across dialects and domains. Distributional methods that cluster words using contextual embeddings have been applied to general-domain lexicon induction (Rothe and Sch¨utze 2015), but dialect terms are often too sparse for such methods to work reliably without additional constraints (Vuli´c and Mrkˇsi´c 2018). Weak supervision methods, including data programming (Tok et al. 2021; Ratner et al. 2016) and Snorkel-based frameworks (Ratner et al. 2017), provide a basis for combining multiple noisy labeling functions. Their use for dialect lexicon construction, however, requires task-specific labeling functions, which motivates the bidirectional anchoring approach proposed in this study.

Semantic drift, in which words change meaning across time periods, communities, or domains, has been studied using distributional semantics. Hamilton et al. Hamilton et al. (2016) showed that word embedding spaces can track systematic historical semantic change and reveal patterns in the rate and direction of meaning shift. Our setting involves a synchronous form of semantic drift: the same character sequence (e.g., 扎实(zhashi), “sturdy” in standard Mandarin but “severe” in Yuxi power-grid complaints) can carry diferent meanings in standard Chinese and in Yuxi power-grid service contexts. The SBERT-based semantic classification stage is designed to capture this domain-specific shift by measuring proximity to power-domain seed embeddings.

## 2.5 Large Language Models for Evaluation and Annotation

Large language models (LLMs) are increasingly used for automated evaluation and annotation in NLP research (Brown et al. 2020). They have been applied to text quality assessment (Mao et al. 2024), factuality verification (Min et al. 2023), and semantic similarity judgment (Kocmi and Federmann 2023), often reaching agreement levels close to human annotators. LLMs have also been explored as oracles in active learning pipelines, where they can match or exceed rule-based oracles on some semantic classification tasks. At the same time, their use requires caution because of consistency limitations, hallucination risk, and annotation cost.

A central methodological concern in LLM-based evaluation is circularity when the same model family is used for both selection and evaluation (Brown et al. 2020; Bubeck et al. 2023). Recent studies have also documented judge bias and calibration challenges in LLM-based evaluation frameworks (Zheng et al. 2023). Even when diferent capability tiers are used, as in our setup with DeepSeek V4 Flash for selection and DeepSeek V4 Pro for evaluation, shared training data and model-family biases may still afect the evaluation. We address this concern in two ways. First, we calibrate the LLM evaluator against independent human annotations using Cohen’s $\kappa ,$ which provides a reliability measure that does not rely on model-internal confidence scores (Cohen 1960). Second, the codebase supports replacing the evaluator with a diferent model family (e.g., GLM-4-Plus) for cross-model validation, although this robustness check has not yet been empirically executed.

Cohen’s $\kappa ,$ introduced by Cohen Cohen (1960) and widely adopted in the behavioral sciences and NLP, measures inter-rater agreement for categorical labels while correcting for chance agreement. Values above 0.6 are conventionally interpreted as substantial agreement, and values above 0.8 as near-perfect agreement (Landis and Koch 1977). Applying κ to LLM evaluation calibration is a recent practice (Gilardi et al. 2023) that supports more transparent reliability assessment of LLM-generated annotations in research settings.

## 3 Methodology

## 3.1 Problem Formulation

Let the streaming WeChat corpus be $D = \{ d _ { 1 } , d _ { 2 } , \dots , d _ { n } \}$ , where each $d _ { i }$ is a raw message string containing text, metadata, and noise artifacts. A preprocessing filter $F ( \cdot )$ maps each raw message to a cleaned string, producing the preprocessed corpus D<sup>′</sup>. The goal is to construct a domain-specific lexicon $L = \{ ( w _ { 1 } , c _ { 1 } ) , \ldots , ( w _ { k } , c _ { k } ) \}$ , where $w _ { j }$ is a dialect expression and $c _ { j } \in { \mathcal { C } }$ denotes its semantic category. The category set C includes OUTAGE, REPAIR, EQUIPMENT, FAULT, BILLING, and OTHER, corresponding to the main dimensions of power-grid customer service interactions. In the current pipeline, these categories serve two internal purposes: organizing the 65 power-domain seed words into thematic groups (e.g., 停电(tingdian, power outage) and 没电(meidian, no electricity) are both OUTAGE seeds) and assigning each candidate a provisional category label based on its nearest seed in the SBERT embedding space. The evaluation, however, focuses only on two binary dimensions, has dialect features and power domain relevance, and does not assess categorylabel correctness. Category-level classification accuracy is therefore not measured, and the final dual-lexicon outputs $( L _ { d }$ and $L _ { p d } )$ do not include category labels in their primary quality metrics. We retain the category structure as scafolding for future downstream integration while making this scope limitation explicit.

Four-Stage Regional Slang Lexicon Construction Framework  
![](images/0feb6f7d0212076bbc40a2a721e092f2427140edf8b90cca0803869ea9347da9.jpg)  
Fig. 1 Overview of the four-stage pipeline.

Figure 1 summarizes the overall pipeline. The four stages are described in the following subsections.

## 3.2 Stage 1: Candidate Generation with Bidirectional Anchoring

Figure 2 shows the four-step candidate generation process from the raw corpus to the filtered candidate pool.

## 3.2.1 Adaptive Noise Filtering

Raw WeChat messages contain metadata and platform artifacts that must be removed before linguistic analysis. We apply an ordered set of regular-expression transformations based on noise patterns observed in the corpus. XML-formatted system messages are removed by matching <msg>...</msg> patterns with recursive nesting support. Platform interaction notifications, including profile tap messages (“拍了拍” (pai le pai)), group invitation messages (“邀请加入了群聊” (invited a user to join the group chat)), and message recall notifications (“撤回了一条消息” (recalled a message)), are removed by matching known templates. User identifiers in the wxid format are extracted and discarded. Application-specific bracket-delimited emoji placeholders ([捂脸] ([facepalm]), [呲牙] ([grinning face]), etc.), URLs, and @-mentions are also stripped.

![](images/7e3f2ec02e27cec4dea28aa5717c630df784babaf709c2e0f187dcd9eb93ec15.jpg)  
Fig. 2 Candidate generation pipeline: from raw corpus to filtered candidate pool.

After noise removal, messages are filtered to retain only Chinese characters (Unicode range U+4E00–U+9FA5), ASCII letters and digits, and a restricted set of punctuation marks (commas, periods, question marks, and exclamation marks). Messages with fewer than two Chinese characters after filtering are discarded as uninformative. From 13,980 raw messages collected over a one-year period, the preprocessing pipeline retained 11,177 cleaned messages, corresponding to a retention rate of approximately 79.9%.

## 3.2.2 Anchor-Driven Bigram Extraction

Candidate generation uses a curated set of Yuxi dialect anchor markers as statistical pivots for discovering associated expressions. The anchor set A contains 16 high-frequency Yuxi dialect morphemes selected from the Yuxi Dialect Vocabulary Compilation and reviewed by domain experts: 呢(ne), 挨(ai), 咋个(zage, “how/what”), 给有(gei you, “is there?”), 是呢(shi ne, “is it?”), 扎实(zhashi, “very/severe”), 鬼 火(guihuo, “frustrated”), 噶(ga), 木(mu), 冇 (mao, “not have”), 咋 整(za zheng, “what to do”), 着(zhao, dialect passive/afected marker), 嘛(ma), 哈(ha), 得行(de xing, “workable/acceptable”), and 老火(laohuo, “severe/critical”). These anchors are divided into two functional groups. Function word anchors (呢(ne), 噶(ga), 嘛(ma), 哈(ha), 咋个(zage), 给有(gei you), 着(zhao), 是呢(shi ne), 咋整(za zheng), 挨(ai), 木(mu), and 冇 (mao)) act as syntactic markers of dialect-specific grammatical constructions. Content word anchors (扎实(zhashi), 鬼火(guihuo), 老火(laohuo), and 得 行(de xing)) carry independent semantic meaning and can combine with domain terms to form dialect compounds.

All anchors and 65 power-domain seed words spanning the five service categories are registered with the jieba Chinese tokenizer so that they are preserved as atomic tokens during segmentation. In addition, 839 reference dialect terms, defined as entries of length $\geq 2$ from the Yuxi Dialect Vocabulary Compilation, are registered with jieba to reduce fragmentation of known dialect expressions. For example, this step keeps 扎实(zhashi, “very/severe”) as one token rather than splitting it into $\ddagger \lfloor + \ \xrightarrow { } $ , two individual characters without the dialect-specific compound meaning. This registration is used only for tokenization calibration and does not constitute label leakage: jieba is told which character sequences can be treated as words, not which candidates are correct lexicon entries. The corpus is then tokenized with these registrations active. For each pair of adjacent tokens $( w _ { 1 } , w _ { 2 } )$ whose co-occurrence frequency $f ( w _ { 1 } , w _ { 2 } ) \ge 2$ the Pointwise Mutual Information is computed:

$$
\mathrm { P M I } ( w _ { 1 } , w _ { 2 } ) = \log _ { 2 } \frac { P ( w _ { 1 } , w _ { 2 } ) } { P ( w _ { 1 } ) P ( w _ { 2 } ) } = \log _ { 2 } \frac { f ( w _ { 1 } , w _ { 2 } ) \cdot T } { f ( w _ { 1 } ) \cdot f ( w _ { 2 } ) } .\tag{1}
$$

Here, $f ( \cdot )$ denotes token frequency in the jieba-segmented corpus and $T$ is the total number of tokens. The merged form $w _ { 1 } w _ { 2 }$ is retained as a candidate if at least one component belongs to ${ \mathcal { A } } ,$ the PMI exceeds 1.5, and the merged form contains at least two Chinese characters. The PMI threshold of 1.5 was selected to balance recall and precision, with reference to prior work on collocation extraction using PMI (Church and Hanks 1990). Normalized PMI is retained only as a diagnostic value and for comparison with earlier pipeline variants:

$$
\operatorname { N P M I } ( w _ { 1 } , w _ { 2 } ) = { \frac { \operatorname { P M I } ( w _ { 1 } , w _ { 2 } ) } { - \log _ { 2 } { P ( w _ { 1 } , w _ { 2 } ) } } } .\tag{2}
$$

In the final experiments, NPMI is not used for candidate filtering; all six configurations use the raw PMI threshold described above. Beyond the original 16 anchors, dialect content-word anchors can be added from the reference vocabulary when they appear in the corpus. In the current corpus, this procedure added one content-word anchor.

## 3.2.3 Soft-Label Bidirectional Anchoring

An important design decision is how to handle dialect bigrams that do not contain explicit power-domain lexical content. A hard power-gating approach would discard any candidate whose component words do not contain electricity-domain characters or match power seed terms. This would remove dialect conversational markers such as 咋个整(zage zheng, “what to $\mathrm { d o } ^ { \prime \prime } )$ and 那咋整(na zage zheng, “then what to do”), which occur in outage complaint threads and may carry indirect service-domain value. We therefore use a soft-label approach: every dialect-anchored bigram that passes the PMI threshold is retained in the candidate pool, and two independent metadata tags are used to separate dialect anchoring from power-domain relevance. The power related tag captures power-domain lexical content through character-level and seed-term matching:

Soft-Label Bidirectional Anchoring vs. Hard Power Gate  
![](images/8a7e4b196ab07862eba8f0a7a22164e6b94ddd410994b30851c86922f4cb1f74.jpg)  
Relevance is tagged, not used as an early hard filter  
Fig. 3 Bidirectional anchoring: hard-gate versus soft-label comparison.

$$
\mathrm { p r } ( w _ { 1 } , w _ { 2 } ) = \mathbf { 1 } \big [ p _ { c } ( w _ { 1 } ) \vee p _ { c } ( w _ { 2 } ) \vee p _ { c } ( w _ { 1 } w _ { 2 } ) \big ] .\tag{3}
$$

Here, $p _ { c } ( w )$ (power chars) is true when any character of w belongs to a set of 21 characters commonly occurring in electricity-domain vocabulary (电, 线, 路, 闸, 表, 费, 火, 跳, 停, 修, 炸, 漏, 烧, 压, 杆, 变, 容, 开, 关, 供, 配; representative meanings include electricity, line, meter, outage, repair, and transformer-related concepts), or when w contains a substring from a supplementary set of 65 short power-domain seed terms (2– 3 character subsets). Substring matching captures terms such as 负荷(fuhe, load) and 来电(laidian, power restoration or incoming electricity) that express power-domain concepts but do not contain the explicit power characters.

Separately, a has dialect anchor tag records whether at least one component of the bigram is a content-word anchor $( \in \mathcal { A } \setminus \mathcal { A } _ { \mathrm { f u n c } } )$ . Content-word anchors such as 扎 实(zhashi, “very/severe”), 鬼火(guihuo, “frustrated”), 老火(laohuo, “severe/critical”), and 得行(de xing, “workable/acceptable”) carry independent semantic meaning and can combine with domain terms to form dialect compounds (e.g., 线路老火(xianlu laohuo, “severe line fault”)). This tag is stored for ablation analysis but does not afect the power related computation; the two signals are kept independent. Figure 3 contrasts the soft-label and hard-gate approaches.

The power related tag is used only for downstream analysis and does not control candidate entry into the pipeline. Section 4.3 quantifies the cost of hard power-gating by comparing the soft-label approach with a hard-gate configuration.

## 3.2.4 Fragment Filter and Unigram Path

The anchor-driven bigram extraction step is intentionally permissive and therefore captures some grammatical fragments in addition to lexical items. For example, 咋个 上个月(zage shanggeyue, “what about last month”) is a productive interrogative pattern rather than a fixed expression. We apply a four-condition fragment filter to remove such constructions while preserving genuine compounds. First, candidates shorter than two Chinese characters are discarded unless they appear in the whitelist of verified dialect expressions. Second, bigrams consisting entirely of dialect particle characters (着, 呢, 嘛, 噶, 哈, 了, 得, 过) are removed because they are not lexical items. Third, short-stem sufix constructions are removed: if a candidate ends with a dialect sufix (呢(ne), 噶(ga), 嘛(ma), 哈(ha), or 着(zhao)) and the remaining stem is shorter than two characters, the candidate is discarded unless it appears in a whitelist of 19 verified dialect expressions, including 不来电呢(bu laidian ne, “no electricity coming,” with dialect particle), 着电(zhao dian, “electrified” or “electric shock”), 停电哈(tingdian ha, “power outage,” with dialect softening particle), and 线路老火(xianlu laohuo, “severe line fault”). Fourth, a jieba segmentation check is applied: if jieba segments the candidate into multiple tokens and at least one resulting token is a pure dialect particle, the candidate is classified as a fragment. This condition captures cases such as 拿着(na zhe, “holding”), where the segmentation 拿+ 着reveals the aspect-marker role of 着rather than its dialect-marker role. Overall, the fragment filter removes approximately 85% of raw bigram candidates.

A parallel unigram extraction path captures individual dialect words that co-occur with power-domain seed terms in the same message. A unigram is retained if it satisfies four criteria: a minimum frequency of five occurrences, passing the fragment filter, a frequency below 1% of total tokens to exclude excessively common function words, and the presence of at least one character from the dialect-specific character set D. The character set D was iteratively refined across experiment runs. The final set includes uniquely dialectal characters and expressions (着, 呢, 嘛, 噶, 哈, 了, 得, 咋, 冇, 木, 挨, 扎实(zhashi, “very/severe”), 鬼火(guihuo, “frustrated”), and 黑隆冬(heilongdong, “very dark”)), excludes characters common in standard Chinese power vocabulary (电(dian, electricity), 炸(zha, explode or burn out)), and removes ambiguous characters (老(lao, old), 过(guo, over/past)) after they allowed standard Chinese terms such as 老百姓(laobaixing, ordinary people), 养老金(yanglaojin, pension), and 莫过 于(moguoyu, nothing more than) to enter the candidate set.

## 3.3 Stage 2: Semantic Classification

Candidates that pass the fragment filter are scored using two complementary signals: semantic proximity to power-domain seeds and lexical fixedness inferred from tokenization behavior. The SBERT backbone, paraphrase-multilingual-MiniLM-L12-v2 (Reimers and Gurevych 2019), produces 384-dimensional L2-normalized embeddings for multilingual semantic similarity. A FAISS IndexFlatIP index (Johnson et al. 2021) over the 65 power-domain seed embeddings supports exact cosine similarity search through inner products over L2-normalized vectors. For each candidate c, the semantic similarity score is defined as the maximum cosine similarity to any power-domain seed:

$$
\sin ( c ) = \operatorname* { m a x } _ { \substack { s _ { j } \in S _ { \mathrm { s e e d } } } } \cos ( v _ { c } , v _ { s _ { j } } ) .\tag{4}
$$

Her ${ \stackrel { \mathbf { \Delta } } { , } } \ v _ { c }$ and $v _ { s _ { j } }$ are the SBERT embeddings of the candidate and seed, respectively. The wordhood score measures lexical fixedness by examining whether jieba treats the candidate as an atomic unit:

$$
{ \mathrm { w o r d h o o d } } ( c ) = \operatorname* { m a x } \left( 0 , 1 - { \frac { k - 1 } { | c | } } \right) - 0 . 3 0 \cdot { \frac { p } { k } } .\tag{5}
$$

Here, k is the number of tokens produced by jieba.lcut(c), |c| is the character length of the candidate, and p is the number of resulting tokens that are pure dialect particles. The first term penalizes multi-token segmentations and therefore rewards candidates that resist fragmentation. The second term adds a penalty when the segmentation includes dialect particles, reflecting the observation that particle-containing multi-token outputs are usually grammatical constructions. The combined score is a weighted sum:

$$
\operatorname { s c o r e } ( c ) = 0 . 6 \cdot \operatorname { s i m } ( c ) + 0 . 4 \cdot \operatorname { w o r d h o o d } ( c ) .\tag{6}
$$

The weights $\alpha = 0 . 6$ and $\beta = 0 . 4$ were determined empirically during development. SBERT similarity provides the stronger signal for power-domain relevance, but it can assign high scores to common vocabulary items. The wordhood score provides a weaker but complementary signal by penalizing such fragmented or particle-like outputs. Candidates with scores at or above the auto-accept threshold $\tau _ { \mathrm { h i g h } } = 0 . 7 8$ enter the initial lexicon directly. Candidates with scores below the rejection threshold $\tau _ { \mathrm { l o w } } = 0 . 4 5$ are discarded. The remaining candidates form the uncertain pool U for active learning.

## 3.4 Stage 3: Active Learning with Twin Oracles

The uncertain pool U from Stage 2 is passed to an iterative active learning loop, as shown in Figure 4. In each round, a query strategy selects K candidates, the chosen oracle verifies them, accepted terms are added to the lexicon, and the loop continues for up to $R _ { \mathrm { m a x } }$ rounds.

The uncertain pool U is the selection space for active learning. We compare three query strategies. Margin Sampling selects the top-K candidates whose scores are closest to the decision boundary $\tau _ { \mathrm { b o u n d } } ~ = ~ 0 . 6 0$ , prioritizing candidates for which the classifier is least certain:

$$
c _ { \mathrm { m a r g i n } } = \underset { c \in U } { \arg \operatorname* { m i n } } \left| \mathrm { s c o r e } ( c ) - \tau _ { \mathrm { b o u n d } } \right| .\tag{7}
$$

TD-CAL combines uncertainty with density. Uncertainty is computed from the combined score, and density is estimated using PMI. This choice is motivated by the limited usefulness of standard TF-IDF-based density for rare dialect pairs:

$$
\mathrm { u n c e r t a i n t y } ( c ) = 1 . 0 - \mathrm { m i n } \Bigg ( \frac { \vert \mathrm { s c o r e } ( c ) - \tau _ { \mathrm { b o u n d } } \vert } { m _ { \mathrm { m a x } } } , 1 . 0 \Bigg ) ,\tag{8}
$$

![](images/b084f7f28b1d89ffa7d0a6dbe2ab5979d78e7699fa6bab6b3d48089264139025.jpg)  
Fig. 4 Active learning loop: iterative query–verify–update cycle with three query strategies and twin oracles.

where $m _ { \mathrm { m a x } } = \mathrm { m a x } ( \tau _ { \mathrm { h i g h } } - \tau _ { \mathrm { b o u n d } } , \tau _ { \mathrm { b o u n d } } - \tau _ { \mathrm { l o w } } ) = \mathrm { m a x } ( 0 . 1 8 , 0 . 1 5 ) = 0 . 1 8$ denotes the maximum possible distance from the decision boundary within the uncertain score range $[ \tau _ { \mathrm { l o w } } , \tau _ { \mathrm { h i g h } } )$

$$
\mathrm { d e n s i t y } ( c ) = \frac { \operatorname { P M I } ( c ) - \operatorname* { m i n } _ { c ^ { \prime } \in U } \operatorname { P M I } ( c ^ { \prime } ) } { \operatorname * { m a x } _ { c ^ { \prime } \in U } \operatorname { P M I } ( c ^ { \prime } ) - \operatorname * { m i n } _ { c ^ { \prime } \in U } \operatorname { P M I } ( c ^ { \prime } ) } .\tag{9}
$$

$$
\operatorname { s c o r e } _ { \operatorname { t d c a l } } ( c ) = 0 . 5 \cdot \operatorname { d e n s i t y } ( c ) + 0 . 5 \cdot \operatorname { u n c e r t a i n t y } ( c ) .\tag{10}
$$

Random Sampling serves as a lower-bound baseline. It selects candidates uniformly from U using a per-round seed (42 + round index), so that diferent random subsets are drawn in each round while experiment runs remain reproducible. We set $K = 5$ candidates per round and $R _ { \mathrm { m a x } } = 3$ rounds. This early stopping creates a regime where $| U | \gg K \times R _ { \mathrm { m a x } }$ , allowing diferent query strategies to select diferent subsets. The oracle query count of 15 reported in Table 3 corresponds to $K \times R _ { \mathrm { m a x } } = 5 \times 3$ queries per experiment. Auto-accepted candidates from Stage 2 enter the lexicon without consuming oracle queries in the default configuration.

Two oracle types are used to verify selected candidates. The rule-based oracle applies dialect morphological pattern matching across three pattern categories: full-word matches, where the candidate contains any of 21 known Yuxi dialect expressions; sufix with power-character co-occurrence, where the candidate ends with a dialect sufix, the stem is at least 2 characters, and the word contains a powerdomain character; and prefix matches, where the candidate begins with a dialect prefix and has suficient length. The RAG-LLM oracle retrieves the top-5 most relevant passages from a 30-entry bilingual power-domain knowledge base using SBERT embedding similarity, then constructs a two-dimensional assessment prompt for DeepSeek

V4 Flash with temperature 0.0. The prompt assesses has dialect features and power domain relevance using 6 few-shot annotated examples. Acceptance requires (has dialect ∨ power relevant) ∧ confidence $\geq 0 . 6$ . We use a permissive OR criterion at the oracle stage because the oracle is intended to retain potential dialect expressions, including borderline cases where the LLM may miss dialect features but identify power-domain relevance, or the reverse. The stricter AND criterion is applied only at the evaluation stage, where PDP measures the proportion of terms satisfying both conditions. This two-stage design reduces premature filtering of candidates that may be dialectal but lack explicit power-domain lexical markers, which is the type of candidate removed by hard power-gating. The low PDP values in Table 3 (0.121–0.160) reflect this permissive design: many accepted terms are dialectal but not power-relevant, consistent with the soft-label strategy of deferring relevance judgments. Robust JSON extraction from LLM responses uses three fallback steps: standard json.loads with markdown code-block extraction, definition-field preprocessing to identify definition boundaries and escape unescaped quotes, and field-by-field regex extraction.

## 3.5 Stage 4: Dual-Track LLM Evaluation

The final lexicon is evaluated by a separate LLM instance (DeepSeek V4 Pro) using the same two-dimensional assessment prompt but without acceptance gating. Using a diferent capability tier from the selection oracle (V4 Pro rather than V4 Flash) provides partial independence. The codebase also supports replacing the evaluator with a diferent model family, such as GLM-4-Plus or GPT-4o, for cross-model evaluation. Two quality metrics are defined:

$$
\mathrm { D P } = { \frac { | \{ w \in L : \mathrm { d i a l e c t } ( w ) = \mathrm { t r u e } \} | } { | L | } } .\tag{11}
$$

$$
\mathrm { P D P } = \frac { | \{ w \in L : \operatorname { d i a l e c t } ( w ) = \operatorname { t r u e } \wedge \operatorname { p o w e r } ( w ) = \operatorname { t r u e } \} | } { | L | } .\tag{12}
$$

DP (Dialect Precision) measures the proportion of lexicon terms that the evaluator identifies as carrying Yuxi dialect features. PDP (Power-Dialect Precision) applies a stricter criterion and measures the proportion of terms that are both dialectal and power-domain relevant. These metrics are calibrated against human expert annotations $( n ~ = ~ 4 6 )$ via Cohen’s κ, providing a reliability-checked basis for comparing experimental configurations.

## 4 Experiments

## 4.1 Experimental Setup

The experimental corpus contains 11,177 cleaned messages extracted from two WeChat group chat CSV exports covering approximately one year of Yuxi power-grid customer service interactions. After jieba tokenization with anchor and seed registration, the corpus contains 190,934 total tokens, 12,165 unique unigrams, and 48,395 unique bigrams. We also constructed a 30-entry bilingual knowledge base covering outage definitions, repair procedures, equipment descriptions, fault taxonomies, billing concepts,

Table 2 Hyperparameter configuration for all experiments.
<table><tr><td>Parameter</td><td>Value</td><td>Description</td></tr><tr><td>Thigh</td><td>0.78</td><td>Auto-accept threshold</td></tr><tr><td> $\tau _ { \mathrm { l o w } }$ </td><td>0.45</td><td>Rejection threshold</td></tr><tr><td>Tbound</td><td>0.60</td><td>Decision boundary</td></tr><tr><td> $\alpha ~ / ~ \beta$ </td><td>0.6 / 0.4</td><td>SBERT / wordhood weights</td></tr><tr><td>K</td><td>5</td><td>Candidates per AL round</td></tr><tr><td> $R _ { \mathrm { m a x } }$ </td><td>3</td><td>Maximum AL iterations</td></tr><tr><td>PMI threshold</td><td>1.5</td><td>Minimum bigram association</td></tr><tr><td>SBERT model</td><td>paraphrase- multilingual- MiniLM-L12-v2</td><td>Embedding backbone</td></tr><tr><td>Oracle LLM</td><td>DeepSeek V4 Flash</td><td>Selection oracle (RAG-LLM)</td></tr><tr><td>Evaluator LLM</td><td>DeepSeek V4 Pro</td><td>Independent evaluator</td></tr><tr><td>LLM temperature</td><td>0.0</td><td>Deterministic decoding</td></tr><tr><td>LLM seed</td><td>42</td><td>Fixed seed for LLM</td></tr><tr><td>RAG top-K</td><td>5</td><td>KB passages per query</td></tr><tr><td>Oracle min confidence</td><td>0.6</td><td>Minimum confidence</td></tr></table>

Yuxi dialect expressions with standard Chinese glosses, and morphological guidelines for distinguishing lexical items from grammatical patterns.

All experiments use a fixed random seed (42) for reproducibility. Table 2 summarizes the hyperparameter configuration. We compare six configurations: five soft-label configurations that vary the active learning query strategy (Margin, TD-CAL, Random) and oracle type (Rule, RAG-LLM), and one strict power-gating ablation (Margin + Rule). Candidate generation and semantic classification are kept constant across configurations, except for the power-gating condition in the ablation.

## 4.2 Candidate Generation and Filtering Performance

The candidate generation pipeline extracted 349 candidates from the 11,177-message corpus, including 347 bigram candidates and 2 unigrams after tightening the dialect character set. The fragment filter removed 291 candidates (83.4% of the total), leaving 58 candidates for semantic classification. The SBERT+wordhood classifier accepted 19 candidates into the initial lexicon $\left( \mathrm { s c o r e } \geq 0 . 7 8 \right)$ and placed 39 candidates into the uncertain pool $( 0 . 4 5 \leq \mathrm { s c o r e } < 0 . 7 8 )$ ; no candidates were rejected outright at this stage. Table 3 reports the main results for all six configurations after the full pipeline, including the evaluator-derived decomposition into dialect $( | L _ { d } | )$ and power-dialect $( | L _ { p d } | )$ subsets.

Figure 5 visualizes the trade-of between PDP and lexicon size across the five soft-label configurations.

Several patterns are observed. First, all six configurations achieve $\mathrm { D P = } 1 . 0 $ every term in each constructed lexicon is identified by the independent evaluator as carrying Yuxi dialect features, with no standard Chinese false positives. This result is attributable to the combined efect of three precision-oriented mechanisms: (i) bidirectional anchoring restricts bigram candidates to expressions containing verified dialect markers; (ii) the fragment filter removes grammatical constructions while preserving lexical compounds; and (iii) the dialect character set D and jieba default-dictionary

Table 3 Main experimental results for all six configurations. DP and PDP are computed by the independent LLM evaluator calibrated against $n = 4 6$ human annotations. $| L _ { d } |$ and $| L _ { p d } |$ denote the sizes of the dialect and power-dialect lexicon subsets, respectively.
<table><tr><td>Model</td><td>DP PDP</td><td></td><td> $| L _ { d } | \ | L _ { p d } |$ </td><td>Lexicon Queries</td><td></td></tr><tr><td> $\mathrm { M a r g i n } + \mathrm { R u l e } \ ( \mathrm { S o f t } { - } \mathrm { L a b e l } )$ </td><td>1.00.160</td><td>25</td><td>4</td><td>25</td><td>15</td></tr><tr><td> $\mathrm { T D - \bar { C } A L + \ R u l e }$ </td><td>1.0 0.148</td><td>27</td><td>4</td><td>27</td><td>15</td></tr><tr><td> $\mathrm { M a r g i n + L L M }$ </td><td>1.0 0.125</td><td>32</td><td>4</td><td>32</td><td>15</td></tr><tr><td> $\mathrm { P r o p o s e d \ ( T D \mathrm { - } C A L + L L M ) }$ </td><td>1.0 0.125</td><td>32</td><td>4</td><td>32</td><td>15</td></tr><tr><td> $\mathrm { R a n \dot { d } o m + L L M }$ </td><td>1.0 0.121</td><td>33</td><td>4</td><td>33</td><td>15</td></tr><tr><td> $\mathrm { M a r g i n + R u l e }$  (Hard Gate)</td><td>1.0 0.625</td><td>8</td><td>5</td><td>8</td><td>5</td></tr></table>

## Experimental Results Across Five Configurations

![](images/213f080c7248903164a5a66b4a599e1a86a55cb10a624475d2bebaceee853012.jpg)

![](images/33dd5c4a1314a4b07474f87adf58ea689fe65655a754425902824f3cc73c1fcf.jpg)  
Fig. 5 (a) Power-Dialect Precision (PDP) and (b) lexicon size across the five experiment configurations. Rule-oracle configurations (left two bars) achieve higher PDP on smaller lexicons; LLM-oracle configurations (right three bars) produce larger lexicons with lower PDP. See Table 3 for exact values.

exclusion filter reduce non-dialect unigram entries. These components remove diferent types of noise and together support high dialect precision.

Oracle type also afects the final lexicons. The RAG-LLM oracle produces larger lexicons (32–33 terms) than the rule-based oracle (25–27 terms), a diference of approximately 24%. This diference is consistent with the per-candidate acceptance rates: the LLM oracle accepts approximately 89% of presented uncertain candidates, whereas the rule oracle accepts roughly 47%. The higher LLM acceptance rate suggests that the LLM oracle recognizes dialect-related constructions beyond the rule oracle’s morphological templates. In contrast, the rule oracle achieves higher PDP (0.148–0.160 vs. 0.121–0.125), indicating that its stricter morphological matching selects more strongly for power-domain relevance.

The query strategy comparison shows limited diferentiation. With the LLM oracle, Margin, TD-CAL, and Random produce nearly identical lexicon sizes (32–33 terms)

and the same $| L _ { p d } |$ count (4). With the rule-based oracle, TD-CAL and Margin produce 27 and 25 terms, respectively, while both identify 4 power-dialect terms. This convergence, especially the uniform $| L _ { p d } | = 4$ across all five soft-label configurations, suggests a ceiling efect: the corpus appears to contain only a small number of observable power-dialect bigrams under the current extraction criteria. The high LLM oracle acceptance rate $( \sim 8 9 \% )$ further narrows the range in which query strategies can difer.

The power-dialect terms captured across configurations cover recurring complaint and query patterns in the Yuxi power-grid domain: 线路老火(xianlu laohuo, literally “line old fire,” meaning “severe line fault”), 不来电呢(bu laidian ne, “no electricity coming,” with the dialect particle 呢(ne)), 停电哈(tingdian ha, “power outage,” with the dialect softening particle $\mathbb { H } _ { \mathtt { H } } ^ { \Delta } ( \mathrm { h a } ) )$ , 给有停电(gei you tingdian, “is there a power outage?,” with the dialect question marker 给有(gei you)), and 着电(zhao dian, literally “afected by electricity,” meaning “electric shock” or “electrified”). The evaluator classifies these terms as carrying both dialect features and power-domain relevance. The hard-gate configuration identifies one additional power-dialect term $( | L _ { p d } | = 5 )$ from its smaller, more targeted candidate pool.

## 4.3 Ablation Study: Soft-Label Versus Hard Power-Gating

To isolate the contribution of soft-label layering, we compare two configurations that difer only in the power-gating strategy, using margin sampling with the rule-based oracle for three AL rounds. The hard-gate configuration applies the power related check as a binary filter: any bigram with $\mathrm { p r } ( w _ { 1 } , w _ { 2 } ) = 0$ is discarded before entering the pipeline. The soft-label configuration retains all dialect-anchored bigrams and stores the tag as metadata.

Table 4 presents the ablation results across five pipeline stages. The largest reduction occurs at candidate extraction, where strict power gating reduces the raw candidate count from 349 to 41, removing 308 dialect bigrams (88.3% of the total). These discarded bigrams include dialect conversational markers such as $\mathbb { H } \mathbb { E } \ \hat { \mathbb { 1 } } \mathbb { \underline { { \mu } } \underline { { \mathtt { b } } } \underline { { \mathtt { b } } } } ( \mathrm { z a g e }$ zheng, “what to do”) and 那咋整(na zage zheng, “then what to $\mathrm { d o } ^ { \prime \prime } )$ , which lack explicit power-domain lexical content but appear in outage complaint contexts as conversational signals associated with service requests. The reduction propagates through subsequent stages: the fragment-filter output decreases from 58 to 9 terms, the uncertain pool decreases from 39 to 5, and the final lexicon decreases from 25 to 8 terms. DP remains 1.0 in both conditions, suggesting that strict power gating does not improve dialect precision in this setting. The fragment filter, dialect character set, and jieba default-dictionary exclusion already remove major sources of non-dialect noise, whereas the strict gate removes additional dialect terms.

Figure 6 visualizes the throughput comparison across pipeline stages.

The ablation suggests that relevance determination should be delayed until later stages, where more contextual and semantic information is available. Strict power gating makes an early filtering decision using character-level power-domain matching, thereby combining two distinct tasks: noise elimination, which is handled by the fragment filter, and relevance assessment, which is handled by the oracle and evaluator. By keeping these tasks separate, soft-label layering preserves candidate diversity while relying on downstream stages to maintain precision.

Table 4 Soft-label versus strict power-gate ablation: pipeline throughput at each stage. Hard Gate applies the power-related check as a binary filter across all extraction paths (bigram, unigram, and anchor injection); Soft Label retains all dialect-anchored candidates. Both configurations use margin sampling with rule-based oracle at three AL rounds. ∆ = percentage increase of Soft Label over Hard Gate.
<table><tr><td>Pipeline Stage</td><td>Hard Gate</td><td>Soft Label</td><td>Δ</td></tr><tr><td>Candidates extracted</td><td>41</td><td>349</td><td>+751%</td></tr><tr><td>After fragment filter</td><td>9</td><td>58</td><td>+544%</td></tr><tr><td>Uncertain pool |U|</td><td>5</td><td>39</td><td>+680%</td></tr><tr><td>Final lexicon |L|</td><td>8</td><td>25</td><td>+213%</td></tr><tr><td>DP</td><td>1.0</td><td>1.0</td><td></td></tr></table>

## Soft-Label vs. Strict Power-Gate Ablation: Pipeline Throughput

![](images/56b9afb058742ffdcad5e89db3a8fb9141b5be62ee37085389ba64727152a200.jpg)  
Fig. 6 Soft-label vs. hard gate ablation: pipeline throughput at each stage. The soft-label approach (blue) consistently retains more candidates, with the largest gain at the extraction stage (+751%).

This comparison also shows an important trade-of. Although both approaches achieve DP=1.0, the hard gate yields higher PDP (0.625 vs. 0.160 for the comparable Margin+Rule soft-label configuration) and identifies more power-dialect terms $( | L _ { p d } | = 5 \mathrm { ~ v s . ~ } 4 )$ , because it pre-filters for candidates with explicit power-domain lexical content. This comes at the cost of a 68% reduction in total dialect terms relative to the soft-label configuration (8 vs. 25); equivalently, the soft-label configuration expands the final lexicon by 213% over hard gating. The soft-label approach therefore trades power-domain precision for coverage, retaining dialect terms that may not be directly power-related. Whether this trade-of is useful depends on the downstream task, which remains to be evaluated.

Table 5 Cohen’s κ between LLM evaluator (DeepSeek V4 Pro) and human expert annotations $( n = 4 6 )$
<table><tr><td>Dimension</td><td>κ</td><td>95% CI</td><td>Verdict</td></tr><tr><td>has_dialect_features</td><td>1.0</td><td></td><td>Degenerate (all True)</td></tr><tr><td>power_domain_relevance</td><td>0.85</td><td>[0.64, 1.0]</td><td>Strong agreement</td></tr></table>

## 4.4 Cohen’s Kappa Calibration

The LLM evaluator was calibrated against human expert annotations on $n = 4 6$ terms. These terms constitute the complete unduplicated lexicon pool obtained by merging all six experimental configurations (157 total entries reduced to 46 unique terms after deduplication). The calibration procedure was designed to select up to 100 terms, stratified into boundary-close (|combined score − decision boundary| $< 0 . 1 0 )$ and random subsets. Because the merged pool was smaller than the planned sample size, all 46 available terms were included (13 boundary-close and 33 random terms). A domain expert in Yuxi dialect and power-grid customer service assessed each term on the two retained evaluation dimensions. The is fixed expression dimension, originally included in the evaluation framework, was removed after an initial calibration round yielded $\kappa = 0 . 2 9$ , indicating low agreement between the human and LLM annotators on this concept in dialect contexts.

Table 5 presents the calibration results for the two retained dimensions. The has dialect features dimension shows perfect observed agreement: all 46 terms were annotated as dialectal by both the human expert and the LLM evaluator $( \kappa = 1 . 0 )$ . This result should be interpreted cautiously because the label distribution is degenerate; when all samples share the same label, κ provides limited discriminative information. For power domain relevance, the κ value of 0.85 exceeds the conventional 0.6 threshold for substantial agreement and is above the 0.8 threshold often interpreted as near-perfect agreement (Cohen 1960; Landis and Koch 1977). The confusion matrix contains 1 false negative, where the human annotator identified power relevance that the LLM evaluator missed, and 1 false positive, where the LLM evaluator identified power relevance that the human annotator did not. The remaining 44 of 46 terms were rated consistently.

The calibration sample of $n = 4 6$ supports the conclusion that κ exceeds the 0.6 reliability threshold for the power-domain dimension. The improved $\kappa _ { \mathrm { p o w e r } } = 0 . 8 5$ (up from 0.75 in the initial $n = 3 8$ calibration) indicates stable evaluator behavior on the expanded term set. The degenerate dialect result is inherent to the evaluated pool: a curated dialect lexicon is expected to contain dialect terms, so perfect agreement on this dimension is less informative than the power-domain relevance result.

## 4.5 Pipeline Performance Evolution

The pipeline evolved through three design iterations based on observed failure modes at diferent stages. Table 6 summarizes the quantitative progression.

Table 6 Pipeline performance across three design iterations.
<table><tr><td>Configuration</td><td>DP</td><td>PDP</td><td>Lexicon</td><td>Candidates</td><td>Unigram Noise</td></tr><tr><td>NPMI, no bidirectional anchoring</td><td>0.17-0.28</td><td>0.01-0.05</td><td>101-148</td><td>347-496</td><td>72-75%</td></tr><tr><td>PMI, strict power gate</td><td>1.0</td><td>0.63</td><td>8</td><td>41</td><td>0%</td></tr><tr><td>PMI, soft-label layering</td><td>1.0</td><td>0.12-0.16</td><td>25-33</td><td>349</td><td>0%</td></tr></table>

The initial configuration used NPMI-based candidate extraction, a three-dimension evaluation framework, and no bidirectional anchoring mechanism. Although this configuration produced the largest lexicons (101–148 terms), unigrams accounted for 72–75% of candidates and were dominated by standard Chinese vocabulary such as 供 电所(gongdiansuo, power supply ofice), 便民服务(bianmin fuwu, convenience service), and 养老保险(yanglao baoxian, pension insurance). These terms co-occurred with power-domain seed words in the corpus but did not carry dialect features, resulting in DP values of only 0.17–0.28 and PDP values near zero.

The introduction of PMI-based bidirectional anchoring as a hard power gate, together with a tighter dialect character filter and a two-dimension evaluation framework, eliminated unigram noise and achieved DP=1.0. However, the hard gate sharply reduced the candidate pool from several hundred terms to 41 and yielded lexicons of only 8 terms. This iteration improved precision but substantially reduced coverage.

The final configuration replaced the hard power gate with soft-label layering, used PMI as the TD-CAL density signal, and introduced early AL stopping at three rounds. This configuration expanded the candidate pool to 349 terms, an 8.5× increase over the hard-gate configuration, while maintaining DP=1.0. The larger uncertain pool (39 candidates vs. 5 under the hard gate) enabled multi-round active learning with strategy-specific selection patterns. Overall, the progression suggests that aggressive early filtering can remove useful signal along with noise, whereas deferring relevance decisions to later stages can preserve coverage while maintaining dialect precision.

## 5 Discussion

## 5.1 Architectural Synergies

The high dialect precision observed across all configurations appears to result from a cascade of complementary design choices rather than from any single component. Bidirectional anchoring reduces non-dialect bigrams, the fragment filter removes grammatical constructions, and the tightened dialect character set blocks non-dialect unigrams. These components address diferent error sources, so their combined efect improves precision. At the same time, this design introduces a maintenance cost: each component must be recalibrated when the pipeline is transferred to a new dialect or domain.

The decision to use PMI rather than NPMI reflects an empirical trade-of specific to this corpus and pipeline. NPMI ofers a useful theoretical advantage because normalization to [−1, 1] supports cross-corpus threshold comparison. In our setting,

Oracle Quality Moderates Active Learning Strategy Differentiation

![](images/fb36249a4aa5289b7cb6e709da120f23e86f38650e1cb1176ce3419c86d1a224.jpg)

![](images/8db59e89c225ff270067b19c13669212ceece111b80b6609567a3cd9d2c35a91.jpg)  
Fig. 7 (a) Oracle acceptance rates. (b) Stacked bar view: colored segments show accepted candidates where both oracles accept regardless of query strategy; gray segments show rejected candidates where strategies can diferentiate (11% for LLM vs. 53% for Rule).

however, two practical issues reduced its usefulness: the edit-distance-based typo filter removed legitimate multi-character compounds, including 线路老火(xianlu laohuo, severe line fault), which was flagged as a typo of the seed term 线路(xianlu, electrical line) at edit distance 2; and NPMI values for rare dialect pairs clustered tightly around 0.15–0.45, providing less discrimination than raw PMI values spanning 2.4– 11.8. This result should not be interpreted as a general claim that PMI is superior to NPMI. Rather, it shows that downstream processing choices, such as the interaction between typo filtering and association thresholds, can afect the practical behavior of an otherwise well-motivated metric.

## 5.2 Oracle Quality and the Limits of Active Learning Strategy Diferentiation

The reduced diferentiation among query strategies under a high-acceptance oracle is consistent with prior observations that AL efectiveness depends on oracle characteristics (Settles 2012). When an oracle accepts most presented candidates, the selection function of AL has limited room to afect the final lexicon. In our setting, the LLM oracle accepts 89% of queried candidates, leaving only an 11% rejection window in which strategy diferences could appear. With only 39 uncertain candidates and a single random seed, this window is too narrow to support statistically distinguishable conclusions. The result should therefore be treated as suggestive and requires replication with larger candidate pools and multiple seeds.

Figure 7 illustrates this pattern. Panel (a) compares the raw acceptance rates, and panel (b) shows the resulting diferentiation windows. The LLM oracle’s 11% rejection window leaves little room for strategy divergence, whereas the rule oracle’s broader 53% rejection window allows modest diferentiation.

The rule-based oracle, with an acceptance rate of approximately $4 7 \% ,$ provides a wider discrimination window. Under this oracle, TD-CAL selects 27 terms and Margin selects 25, although both identify the same number of power-dialect terms $( | L _ { p d } | = 4 )$ . This pattern suggests a hypothesis for future work: the expected diferentiation between AL strategies under oracle O may depend on Var[accept(O, c)] over the candidate distribution.

## 5.3 Practical Significance and Limitations

The constructed lexicon contains 25–33 terms across configurations. Although modest in size, it includes high-frequency dialect expressions that are relevant to power-grid customer service interactions in the Yuxi region. These expressions encode distinctions related to fault severity, user emotional state, and request urgency, which may be useful for downstream customer service NLP systems. A domain expert could enumerate many of these terms manually; the value of the automated pipeline lies in its reproducibility across new data collections, its calibrated evaluation procedure, and its modular structure for adaptation to other dialects or service domains.

Several limitations constrain the generalizability of these findings. First, the evaluation is restricted to one dialect (Yuxi), one domain (power grid), and one platform (WeChat), and transfer to other settings has not been empirically tested. Second, the evaluation relies on precision-oriented metrics (DP and PDP) computed by an LLM evaluator. We do not report recall against an expert-constructed gold lexicon, nor do we provide full human evaluation of the final lexicons beyond the $n = 4 6$ calibration set. As a result, we cannot quantify how much of the true regional power slang in the corpus is recovered. Third, the 11,177-message corpus limits the observable diversity of dialect bigrams, and additional power-dialect compounds may remain uncaptured. Fourth, the experimental comparisons are limited to internal variants of the proposed framework; comparisons with established phrase mining baselines such as AutoPhrase (Shang et al. 2018), SegPhrase (El-Kishky et al. 2014), C-value/NCvalue (Frantzi et al. 2000), TextRank (Mihalcea and Tarau 2004), and zero-shot LLM extraction would provide stronger evidence for the contribution of individual components. Fifth, the active learning experiment is underpowered: with $| U | = 3 9$ uncertain candidates and a single random seed, strategy-level conclusions cannot be tested statistically. Sixth, the evaluator shares the same model family as the selection oracle, and cross-model evaluation has not yet been executed. Seventh, the lexicon has not been evaluated on downstream tasks such as complaint classification, urgency detection, or automated ticket routing. Finally, several pipeline choices, including the PMI threshold (1.5), SBERT/wordhood weights (0.6/0.4), anchor set composition (16 terms), seed set (65 terms), and whitelist (25 terms), were tuned through iterative experimentation on the same corpus without a held-out validation set; their sensitivity remains to be analyzed.

## 5.4 Ethics and Reproducibility

The WeChat group data used in this study were collected from power-grid customer service groups operated by Yuxi Power Supply Bureau. Group members had consented to service-related message monitoring as part of the utility’s customer engagement framework. Before analysis, personal identifiers, including wxid usernames, phone numbers, and profile names, were removed through regex-based anonymization. The resulting dataset cannot be publicly released because of privacy constraints and the terms of the original data-sharing agreement. Researchers seeking to replicate the study may request access through the corresponding author, subject to institutional data governance approval.

To support reproducibility despite data constraints, we provide the experiment code, hyperparameter configuration, anchor and seed sets, knowledge base prompts, oracle few-shot examples, and evaluation rubrics. The pipeline is version-controlled and produces deterministic outputs when the random seeds and corpus are fixed. Model artifacts, including the SBERT backbone, DeepSeek API versions, and configuration parameters, are documented in Table 2. The calibration annotation template and the final constructed lexicons, including dialect and power-dialect subsets, are included in the experiment output directory.

## 5.5 Future Directions

Future work should address these limitations in several directions. Cross-dialect experiments, in which the Yuxi anchor set is replaced with markers from other regional Chinese dialects such as Sichuanese, Cantonese, or Northeastern Mandarin, would test the portability of the framework. An expert gold-standard lexicon constructed from a stratified corpus sample would enable recall-oriented evaluation through metrics such as recall, F1, and coverage. Comparisons with established phrase mining baselines, including AutoPhrase, SegPhrase, C-value/NC-value, TextRank, and zero-shot LLM extraction, would clarify the contribution of each pipeline component under a shared annotation protocol. Larger corpora, for example 100K+ messages per dialect, would improve coverage of long-tail dialect expressions and allow more reliable active learning comparisons with multiple random seeds and significance testing. Sensitivity analysis of the PMI threshold, score weights, anchor set, and seed set would quantify robustness. Further analysis of the relationship between oracle acceptance characteristics and AL strategy diferentiation could test whether Var[accept(O, c)] predicts strategy separation. Finally, downstream task evaluation and cross-model evaluator validation are needed to assess practical utility and reduce concerns about model-family circularity.

## 6 Conclusion

This paper presented a statistical-anchor-driven weak-supervision framework for constructing a regional slang lexicon from noisy short texts in a low-resource dialect setting. The framework combines bidirectional anchoring with soft-label power-domain layering, allowing dialect-anchored candidates to be retained even when they do not contain explicit power-domain lexical markers. On 11,177 WeChat customer service messages from Yuxi, the framework reaches an LLM-evaluator-estimated dialect precision of DP=1.0. The ablation study shows that soft-label layering expands the candidate pool by 751% and the final lexicon by 213% relative to strict power-gating while maintaining the same dialect precision. The dual-track LLM evaluation framework, calibrated against human expert annotations $( \kappa _ { \mathrm { d i a l e c t } } = 1 . 0 , \kappa _ { \mathrm { p o w e r } } = 0 . 8 5$ $n = 4 6 )$ , supports comparison across configurations, although cross-model validation and larger-scale human evaluation remain necessary.

The experiments also provide preliminary evidence that oracle characteristics afect active learning outcomes. Under the high-acceptance LLM oracle (∼89% acceptance), all three query strategies converge to the same $| L _ { p d } |$ , whereas modest diferentiation appears only in total lexicon size under the more selective rule-based oracle (∼47% acceptance). This finding should be interpreted cautiously because the uncertain pool is small and only one random seed is used. Larger candidate pools, multiple seeds, and formal statistical testing are needed before drawing general conclusions about the interaction between oracle properties and AL eficiency.

The modular structure of the framework separates candidate generation, semantic classification, oracle verification, and independent evaluation into distinct stages, which may support adaptation to other dialects or service domains. More broadly, the results suggest that weak-supervision pipelines for low-resource dialect settings can benefit from delaying relevance decisions until later stages, where richer semantic and contextual information is available. The practical utility of the constructed lexicon still needs to be tested through downstream tasks such as complaint classification, urgency detection, and ticket routing, as well as through cross-dialect generalization experiments.

Acknowledgments. The authors would like to thank the Editors and Reviewers for their review eforts and helpful feedback.

## Declarations

• Funding: Not applicable.

• Conflict of interest/Competing interests: The authors declare no conflicts of interest. The funders had no role in the design of the study; in the collection, analyses, or interpretation of data; in the writing of the manuscript; or in the decision to publish the results.

• Ethics approval and consent to participate: The WeChat group data used in this study were collected from power-grid customer service groups operated by Yuxi Power Supply Bureau. Group members had consented to service-related message monitoring as part of the utility’s customer engagement framework.

• Data availability: The dataset cannot be publicly released because of privacy constraints and the terms of the original data-sharing agreement. Researchers seeking to replicate the study may request access through the corresponding author, subject to institutional data governance approval.

• Code availability: The complete source code, selected data and experimental scripts are publicly available at https://github.com/zxzang/Construction-Methodof-Regional-Slang-Lexicon-for-Electricity-Demands.

• Author contribution: All authors contributed to the study conception and design. R.C. and L.S. collected the data and provided domain expertise. Z.Z. designed the methodology and experiments. All authors reviewed and approved the final manuscript.

## References

Baldwin T, Cook P, Lui M, et al (2013) How noisy social media text, how difrnt social media sources? In: Proceedings of IJCNLP, pp 356–364

Bender EM, Gebru T, McMillan-Major A, et al (2021) On the dangers of stochastic parrots: Can language models be too big? In: Proceedings of FAccT, pp 610–623

Blodgett SL, Barocas S, Daum´e III H, et al (2020) Language (technology) is power: A critical survey of “bias” in NLP. In: Proceedings of ACL, pp 5454–5476

Bouma G (2009) Normalized (pointwise) mutual information in collocation extraction. In: Proceedings of the Biennial GSCL Conference, pp 31–40

Brown T, Mann B, Ryder N, et al (2020) Language models are few-shot learners. In: Advances in Neural Information Processing Systems, pp 1877–1901

Bubeck S, Chandrasekaran V, Eldan R, et al (2023) Sparks of artificial general intelligence: Early experiments with GPT-4. arXiv preprint, arXiv:2303.12712

Chao YR (1968) A Grammar of Spoken Chinese. University of California Press, Berkeley

Church KW, Hanks P (1990) Word association norms, mutual information, and lexicography. Computational Linguistics 16(1):22–29

Cohen J (1960) A coeficient of agreement for nominal scales. Educational and Psychological Measurement 20(1):37–46

Cui Y, Che W, Liu T, et al (2021) Pre-training with whole word masking for Chinese BERT. IEEE/ACM Trans Audio, Speech, Lang Process 29:3504–3514

Darwish K, Magdy W, Mourad A (2012) Language processing for Arabic microblog retrieval. In: Proceedings of the 21st ACM International Conference on Information and Knowledge Management (CIKM), pp 2427–2430

Deng S, Li Q, Dai R, et al (2024) A Chinese power text classification algorithm based on deep active learning. Applied Soft Computing 150:111067. https://doi.org/10. 1016/j.asoc.2023.111067

Deng Z, Ma W, Han QL, et al (2025) Exploring DeepSeek: A survey on advances, applications, challenges and future directions. IEEE/CAA Journal of Automatica Sinica 12(5):872–893. https://doi.org/10.1109/JAS.2025.125498

Derczynski L, Ritter A, Clark S, et al (2013) Twitter part-of-speech tagging for all: Overcoming sparse and noisy data. In: Proceedings of RANLP, pp 198–206

Devlin J, Chang MW, Lee K, et al (2019) BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proceedings of NAACL-HLT, pp 4171– 4186

Ein-Dor L, Halfon A, Gera A, et al (2020) Active learning for BERT: An empirical study. In: Proceedings of EMNLP, pp 7949–7962

Eisenstein J (2013) What to do about bad language on the internet. In: Proceedings of NAACL-HLT, pp 359–369

Eisenstein J (2017) Identifying regional dialects in online social media. In: Handbook of Dialectology. Wiley, p 368–383

El-Kishky A, Song Y, Wang C, et al (2014) Scalable topical phrase mining from text corpora. Proceedings of the VLDB Endowment 8(3):305–316

Evert S (2005) The statistics of word cooccurrences: Word pairs and collocations. PhD thesis, University of Stuttgart

Frantzi K, Ananiadou S, Mima H (2000) Automatic recognition of multi-word terms: The C-value/NC-value method. International Journal on Digital Libraries 3(2):115– 130

Gilardi F, Alizadeh M, Kubli M (2023) ChatGPT outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences 120(30):e2305016120. https://doi.org/10.1073/pnas.2305016120

Grishman R, Sundheim B (1996) Message understanding conference-6: A brief history. In: Proceedings of COLING, pp 466–471

Gururangan S, Marasovi´c A, Swayamdipta S, et al (2020) Don’t stop pretraining: Adapt language models to domains and tasks. In: Proceedings of ACL, pp 8342– 8360, https://doi.org/10.18653/v1/2020.acl-main.740

Hafari G, Roy M, Sarkar A (2009) Active learning for statistical phrase-based machine translation. In: Proceedings of NAACL-HLT, pp 415–423

Hamilton WL, Leskovec J, Jurafsky D (2016) Diachronic word embeddings reveal statistical laws of semantic change. In: Proceedings of ACL, pp 1489–1501

Hedderich MA, Lange L, Adel H, et al (2021) A survey on recent approaches for natural language processing in low-resource scenarios. In: Proceedings of NAACL-HLT, pp 2545–2568

Johnson J, Douze M, J´egou H (2021) Billion-scale similarity search with GPUs. IEEE Trans Big Data 7(3):535–547

Joshi P, Santy S, Budhiraja A, et al (2020) The state and fate of linguistic diversity and inclusion in the NLP world. In: Proceedings of ACL, pp 6282–6293

Kocmi T, Federmann C (2023) Large language models are state-of-the-art evaluators of translation quality. In: Proceedings of EAMT, pp 193–203

Landis JR, Koch GG (1977) The measurement of observer agreement for categorical data. Biometrics 33(1):159–174

Lauscher A, Ravishankar V, Vuli´c I, et al (2020) From zero to hero: On the limitations of zero-shot cross-lingual transfer with multilingual transformers. In: Proceedings of EMNLP, pp 4483–4499

Lewis DD, Gale WA (1994) A sequential algorithm for training text classifiers. In: Proceedings of SIGIR, pp 3–12

Lewis P, Perez E, Piktus A, et al (2020) Retrieval-augmented generation for knowledgeintensive NLP tasks. In: Advances in Neural Information Processing Systems, pp 9459–9474

Li C, Duan Y, Wang H, et al (2016) Enhancing topic modeling for short texts with auxiliary word embeddings. In: Proceedings of ECIR, pp 598–611

Liang S, Yilmaz E, Kanoulas E (2016) Dynamic clustering of streaming short documents. In: Proceedings of KDD, pp 995–1004, (Extended journal version: IEEE Trans. Knowl. Data Eng. 2018, 30(8), 1553–1566)

Mao R, Chen G, Zhang X, et al (2024) GPTEval: A survey on assessments of Chat-GPT and GPT-4. In: Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING), Torino, Italy, pp 7844–7866

Mihalcea R, Tarau P (2004) TextRank: Bringing order into text. In: Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing, pp 404–411

Min S, Krishna K, Lyu X, et al (2023) FactScore: Fine-grained atomic evaluation of factual precision in long form text generation. In: Proceedings of EMNLP, pp 12076–12100

Nguyen D, Do˘gru¨oz AS, Ros´e CP, et al (2016) Computational sociolinguistics: A survey. Computational Linguistics 42(3):537–593

Norman J (1988) Chinese. Cambridge University Press, Cambridge

Pennington J, Socher R, Manning CD (2014) GloVe: Global vectors for word representation. In: Proceedings of EMNLP, pp 1532–1543

Pires T, Schlinger E, Garrette D (2019) How multilingual is multilingual BERT? In: Proceedings of ACL, pp 4996–5001

Poyiadzi R, Bacaicoa-Barber D, Cid-Sueiro J, et al (2022) The weak supervision landscape. In: Proceedings of the 2022 IEEE International Conference on Pervasive Computing and Communications Workshops (PerCom Workshops), Pisa, Italy, pp 218–223

Ratner A, De Sa C, Wu S, et al (2016) Data programming: Creating large training sets, quickly. In: Advances in Neural Information Processing Systems, pp 3567–3575

Ratner A, Bach SH, Ehrenberg H, et al (2017) Snorkel: Rapid training data creation with weak supervision. Proceedings of the VLDB Endowment 11(3):269–282

Reimers N, Gurevych I (2019) Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In: Proceedings of EMNLP-IJCNLP, pp 3982–3992

Rothe S, Sch¨utze H (2015) AutoExtend: Extending word embeddings to embeddings for synsets and lexemes. In: Proceedings of ACL, pp 1793–1803

Settles B (2012) Active Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, Morgan & Claypool Publishers

Settles B, Craven M (2008) An analysis of active learning strategies for sequence labeling tasks. In: Proceedings of EMNLP, pp 1070–1079

Shang J, Liu J, Jiang M, et al (2018) Automated phrase mining from massive text corpora. IEEE Trans Knowl Data Eng 30(10):1825–1837

Shen Y, Yun H, Lipton ZC, et al (2017) Deep active learning for named entity recognition. In: Proceedings of the 2nd Workshop on Representation Learning for NLP, pp 252–256

Sun J (2012) Jieba Chinese text segmentation (open source project). GitHub, https: //github.com/fxsjy/jieba

Tan Z, Chen J, Li W, et al (2023) Named entity recognition for few-shot power dispatch based on multi-task learning. Electronics 12(16):3476. https://doi.org/10. 3390/electronics12163476

Tok WH, Bahree A, Filipi S (2021) Practical Weak Supervision. O’Reilly Media, Inc.

Tong S, Koller D (2000) Support vector machine active learning with applications to text classification. Journal of Machine Learning Research 2:45–66

Vuli´c I, Mrkˇsi´c N (2018) Specialising word vectors for lexical entailment. In: Proceedings of NAACL-HLT, pp 1134–1145

Zampieri M, Tan L, Ljubeˇsi´c N, et al (2014) A report on the DSL shared task 2014. In: Proceedings of VarDial Workshop at COLING, pp 1–9

Zampieri M, Malmasi S, Nakov P, et al (2017) Findings of the VarDial evaluation campaign 2017. In: Proceedings of VarDial Workshop at EACL, pp 1–15

Zhang F (1985) Yuxi Dialect Vocabulary Compilation (玉溪方言词汇汇编). Yuxi Local Chronicles Ofice, Yunnan, China

Zhao H, Huang C, Li X (2007) Chinese word segmentation: A decade review. Journal of Chinese Information Processing 21(3):8–19

Zheng L, Chiang WL, Sheng Y, et al (2023) Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. In: Advances in Neural Information Processing Systems, pp 46595–46623

Zonta T, da Costa CA, da Rosa Righi R, et al (2020) Predictive maintenance in the Industry 4.0: A systematic literature review. Computers & Industrial Engineering 150:106889