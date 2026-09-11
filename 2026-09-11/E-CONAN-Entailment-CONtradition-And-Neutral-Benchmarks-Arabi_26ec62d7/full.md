Date of publication xxxx 00, 0000, date of current version xxxx 00, 0000.

Digital Object Identifier 10.1109/ACCESS.2024.Doi Numbe

# E-CONAN (Entailment, CONtradition And Neutral) Benchmarks: Arabic Textual Entailment and Natural Inference Datasets.

Khloud AL Jallad<sup>1</sup>, Nada Ghneim<sup>2</sup>, and Ghaida Rebdawi<sup>1</sup>

<sup>1</sup> Higher Institute for Applied Sciences and Technology, Damascus, Syria.   
<sup>2</sup> Arab International University, Daraa, Syria.

Corresponding author: Khloud AL Jallad (e-mail: Khloud.aljallad@hiast.edu.sy).

ABSTRACT Natural Language Inference (NLI), processes pairs of sentences to extract their semantic relations. NLI has been a hot research topic, integrated as a main component in other NLP applications, from morphological spelling correction tasks to higher-level tasks such as machine translation and information extraction. Despite significant advancements in textua inference across various languages all around the world, Arabic language still suffers from limited resources in this domain, specifically, when it comes to the scarcity of robust well-constructed benchmarks necessary for effective model tuning and generalization. To address this gap, this paper introduces E-CONAN benchmarks that are composed of sentences pairs from various sources: (1) automatically-translated pairs, (2) human-validated machine-translated pairs, (3) hand-crafted pairs from teaching Arabic as foreign language books, and (4) headlines pairs from different news channels containing rumors. E-CONAN contains two benchmark datasets, E-CONAN-2, a 2-way dataset (RTE) and E-CONAN-3, a 3-way dataset (NLI). Additionally, we have used E-CONAN benchmarks to evaluate 9 state-of-the-art multilingual pretrained models using zero-shot classification. Models were evaluated across the ArNLI, XNLI, and E-CONAN datasets. Results show that E-CONAN is a potentially valuable resource for evaluating model generalization and even for fine-tuning pre-trained models. Its diverse composition, derived from a combination of sources, offers a broader and more robust assessment compared to XNLI and ArNLI. Furthermore, results show that mDeBERTa model, pre-trained on 100 languages and fine-tuned on a combination of four machine-translated datasets, demonstrated superior performance on all datasets. It achieved accuracies of 71% and 86% on the E-CONAN-3 and XNLI datasets, respectively, outperforming all other evaluated models. In addition, we have evaluated 5 LLMs on E-CONAN-3 dataset. Best results were achieved by Gemma with an accuracy of 68%. Analyzing these results showed that most errors were between neutral and contradiction classes, thus we calculated results after reformulating E-CONAN-3 using 2-way labeling. Best results are achieved by Gemma and Qwen with an accuracy of 95%, 94% respectively. Moreover, we incorporated MARBERT as a representative Arabic-specific baseline and conducted performance evaluation comparison to demonstrate how Arabic-specific models scale against cross-lingual and LLM-based approaches on the E-CONAN benchmarks. Furthermore, we conducted detailed qualitative and quantitative error analysis to analyze frequent error patterns. Results show that while pretrained models are often misled by lexical overlap, LLMs are often misled by topical familiarity. E-CONAN benchmarks will be publicly available, we hope that it will enrich research community in Arabic textual entailment and natural language inference.

INDEX TERMS Textual Entailment, Arabic Natural Language Inference (NLI), Contradiction Detection, Textual Inference, ArNLI, XNLI, LLMs, 2-way entailment, 3-way entailment.

## I. INTRODUCTION

Recognizing Textual Entailment (RTE) is the task of detecting logical relation type between pair of sentences (Text as T and Hypothesis as H). When first introduced, RTE task was called 2-way RTE as it classified pairs into two relations (entail/not entail). The 3-way RTE term appeared in 2008 [1], as the task of determining entailment relation between pairs of sentences introduced three relations (entail/ contradict/ neutral or unknown). Later, the term Natural Language Inference (NLI) has been frequently used for the latter task. The studied semantic relations between the two pair sentences (T and H) are:

1- Entailment: Entailment occurs when T and H intersect or when one is contained in the other (T∩H≠ ∅ AND T∪H = Universe) OR ((T⊆H) OR (H ⊆T)). Figure 1(a) represents the Entailment relation using Algebra Sets notation. For instance, the relation between “Sam travelled to Paris” as T and “Sam travelled to France” as H, is T entails H, as T is a part of H. However, the opposite is not right as France or any country have many cities in it.

2- Contradiction: Contradiction occurs when T and H cannot be true together, i.e., if T is True then H is False and vice versa (T∩H= ∅ AND T∪H=Universe). Figure 1(b) represents the Contradiction using Algebra Sets notation. For instance, the relation between “I like reading” as T and “I hate reading” as H, is contradiction as they cannot be true at the same time.

3- Neutral: Neutral occurs when T and H have no semantic relations. Each one of them is a set in the universe, no intersection between them and their union is not the whole universe (T∩H= ∅ AND T∪H ≠ Universe). Figure 1(c) represents the Neutral relation using Algebra Sets notation. The relation between “I like reading” as T and “I hate oranges” as H, is neutral, as there is no contradiction neither entailment between them. The same relation applies if the sentence H was “I like oranges”.

![](images/50cd6a379ac0b9b7af607e295e597ad8526ba8283e761196e46e74dddc3d6a64.jpg)  
(a) Entailment Relation

![](images/7cc84da614c2a99270a191b0dac17c65ab6b2db8661bbc439f36dd9c59440f8a.jpg)  
(b) Contradiction Relation

![](images/9ceaa7f8d6125a3988ec1679e938e7b4785f2337bb878d2ccb20d8bd6a2ce22c.jpg)  
(c) Neutral Relation  
<sup>D</sup> is the Universe, Yellow is T, Blue is H.  
FIGURE 1. Pairs Inference Relations Types.

While significant advancements have been made in textual inference across many languages, Arabic remains under-resourced in this domain, with a lack of high-quality benchmarks that are essential for the proper tuning and generalization of textual inference models. While existing Arabic NLI benchmarks have established a foundational groundwork, they are typically constrained to a single data-generation paradigm. For instance, resources like Arabic XNLI and SNLI rely almost on machine translation, whereas others are strictly limited to a single domain or handcrafted format. This single constructing paradigm leaves a gap in evaluating models’ generalization across real-world linguistic variation. To address this, the proposed E-CONAN benchmarks purposefully unified all structural and linguistic advantages of these varied approaches into a unified benchmark. By merging human-validated machine translations, automatically translated texts, hand-crafted texts, and news headlines, E-CONAN captures a good level of linguistic diversity, spanning formal, casual, translated, and highly nuanced native Arabic structures. Consequently, it provides a uniquely challenging and comprehensive environment to test model generalization across multiple text styles. The main contributions of this paper are:

1- RTE Benchmark (E-CONAN-2 dataset) that contains 24,875 pairs of sentences, collected from ArNLI dataset, the Arabic translated section of SNLI dataset, the Arabic translated section of XNLI dataset, AnsStance Dataset, ArEntail Dataset.

2- NLI Benchmark (E-CONAN-3 dataset) that contains 18,875 pairs of sentences, collected from ArNLI dataset, the Arabic translated section of SNLI dataset, the Arabic translated section of XNLI dataset, AnsStance Dataset.

3- Evaluation of 10 pretrained models to compare E-CONAN datasets with three previous SoTA benchmarks datasets (ArNLI, XNLI, ArEntail).

4- Evaluation of 6 LLMs to compare E-CONAN datasets with three previous SoTA benchmarks datasets (ArNLI, XNLI, ArEntail).

This paper is organized as follows: Introduction is in section 1. Related works are shown in section 2. Section 3 contains the datasets construction process and datasets statistics. Section 4 shows the Evaluation Models. Results and discussions are presented in section 5. Error analysis is shown in section 6, with a conclusion in section 7.

## II. Related Works

RTE was first started as a challenge in PASCAL [2]. PASCAL stands for Pattern Analysis, Statistical Modelling and Computational Learning. PASCAL is a Network of Excellence funded by the European Union. It has established a distributed institute that brings together researchers and students across Europe, and is now reaching out to new countries all over the world. PASCAL run the Recognizing Textual Entailment Challenge series where each of them focus on RTE in specific applications.

Several PASCAL challenges were done by enriching examples each new challenge to represent different

1 Al Jazeera http://www.aljazeera.net/ Al Arabiya http://www.alarabiya.net/ and BBC Arabic http://www.bbc.co.uk/arabic/

levels of entailment reasoning, such as lexical, syntactic, morphological and logical. And most of them focus on four famous NLP applications: Information Retrieval (IR), Information Extraction (IE), Question Answering (QA), and multi-document summarization (SUM) as such systems must recognize the different forms in which their inputs and requested outputs might be expressed, so there is a need to have a framework for logic-based meaning-level representations using RTE. RTE datasets were initially annotated with two classes (entail/not entail), called 2-way RTE. Later, a 3-way RTE emerged with three annotations (entail/contradict/neutral) by splitting the 'not entail' class into 'neutral' if no semantic relation between pairs and 'contradiction' if opposite semantic meaning between them. The 3-way RTE is also called Natural Language Inference (NLI).

Table 1 shows comparisons between datasets of PASCAL RTE challenges.

TABLE I  
PASCAL RTE CHALLENGE RTE DATASETS
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Year</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>MainTask</td><td rowspan=1 colspan=1>MainSubtasks</td><td rowspan=1 colspan=1>AnnotationS</td></tr><tr><td rowspan=1 colspan=1>RTE1[2]</td><td rowspan=1 colspan=1>2006</td><td rowspan=1 colspan=1>It contains manuallycollected pairs of Text-Hypothesis. Hypothesis (H)length is same as Text(T)length and it was 1-2sentences.</td><td rowspan=1 colspan=1>E,IR,QA,SUM</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2-way</td></tr><tr><td rowspan=1 colspan=1>RTE2[3]</td><td rowspan=1 colspan=1>2006</td><td rowspan=1 colspan=1>It was divided into trainingand testing set. Then it wasdivided into 200 text-hypothesis pairs for differentapplications.</td><td rowspan=1 colspan=1>IE,IR,QA,SUM</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2-way</td></tr><tr><td rowspan=1 colspan=1>RTE3[4]</td><td rowspan=1 colspan=1>2007</td><td rowspan=1 colspan=1>No major changes on thedataset</td><td rowspan=1 colspan=1>IE,IR,QA,SUM</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>2-way</td></tr><tr><td rowspan=1 colspan=1>RTE4[5]</td><td rowspan=1 colspan=1>2008</td><td rowspan=1 colspan=1>It was the first conferencepaper that proposed RTE asthree-judgment task wherethey added the classcontradiction to the labels toexpress if two sentences donot entail same meaning butalso entail contradictedmeanings. The “Text&quot; waslonger than the text in RTE-3where the “Hypothesis&quot;length was the same.</td><td rowspan=1 colspan=1>IR,QA,SUM</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3-way</td></tr><tr><td rowspan=1 colspan=1>RTE5[6]</td><td rowspan=1 colspan=1>2009</td><td rowspan=1 colspan=1>The added value was thatTextual Entailment judgmentis done over a real corpus ofSummarization scenario</td><td rowspan=1 colspan=1>SearchPilot</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>3-way</td></tr><tr><td rowspan=1 colspan=1>RTE6[7]</td><td rowspan=1 colspan=1>2009</td><td rowspan=1 colspan=1>Saves the same features ofRTE-5</td><td rowspan=1 colspan=1>UpdateSummarization</td><td rowspan=1 colspan=1>NoveltyDetectionAnd</td><td rowspan=1 colspan=1>3-way</td></tr></table>

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>scenario</td><td rowspan=1 colspan=1>KBPValidationPilot</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>RTE7[8]</td><td rowspan=1 colspan=1>2011</td><td rowspan=1 colspan=1>Saves the same features ofRTE-5 and RTE-6</td><td rowspan=1 colspan=1>UpdateSummarizationscenario</td><td rowspan=1 colspan=1>NoveltyDetectionandKBPSlotFilling</td><td rowspan=1 colspan=1>3-way</td></tr></table>

In English, numerous NLI datasets were published in the last few years, such as WNLI [9], SNLI [10], MultiNLI [11]. Additionally, GLUE proposed RTE dataset [12] which is a combination of RTE1 [2], RTE2 [3], RTE3 [4], and RTE5 [6]. Some of these datasets were automatically translated to 15 languages including Arabic, such as XNLI [13] and SNLI [14]. However, this automatic translation introduced errors in meaning of pairs that affects labels and thus models’ performance.

Using these datasets, many state-of-the-art models were proposed such as: PaLM 540B[15], Vega\_v2\_6B[16], RoBERTa[17], SemBERT[18], XLNET[19], AlexaTM\_20B[20], BloombergGPT and GPT-NeoX and BLOOM\_176B [21], UnitedSynT5[22], ByT5[23] , Rethinking Coupling[24], mGPT[25], DeBERTa[26], SpanBERT[27], SqueezeBERT[28], DistilBERT[29].

As for Arabic language, there are still few datasets, as follows:

2-Way ArbTEDS [30] that consists of 618 texthypothesis pairs collected from Arabic news websites (Al Jazeera, Al Arabiya, BBC Arabic)<sup>1</sup> and from annotated pairs collected by hand. These pairs cover a number of domains such as politics, business, sport and general news. ArbTEDS was annotated by eight expert/non-expert volunteer annotators.

2-Way ArEntil Dataset [31] that contains 6000 sentence pairs collected from news headlines and manually labeled.

3-Way ArNLI [32] that contains 6366 pairs divided as (1932 entailment, 1073 contradiction, and 3361 neutral).

Some state-of-the-art models that contain Arabic language are: Facebook Bart Large MNLI [33], [34], Facebook RoBERTa-Large-MNLI [35] , Microsoft Multilingual MiniLM [36], Microsoft XLM-RoBERTa [36], Microsoft DeBERTaV3 [37]. In addition, Laurer et al. tuned many versions of pretrained multilingual models such as, mDeBERTa-V3-Base-XNLI-Multilingual-NLI-2mil7 [38],Moritzlaurer Ernie-M-Base-MNLI-XNLI [38], Moritzlaurer Multilingual-MiniLMv2-L6-MNLI-XNL [38], Moritzlaurer Multilingual-MiniLMv2-L12-MNLI-XNLI[38]. We have used many of them as zero-shot classification to test and validate our proposed datasets.

Recently, Natural Language Inference and cross-lingual inference have become critical tasks for benchmarking the reasoning capabilities of Large Language Models (LLMs). Recent studies have shifted from pre-trained models to evaluating LLMs, such as Allam [46] , Qwen [47], [48], Command R7B Arabic [53], DeepSeek-R1 [54] and Gemma [52] on complex semantic reasoning. However, evaluating Arabic NLI still faces a significant challenge due to the scarcity of high-quality, native Arabic NLI datasets, with many existing benchmarks relying on machine translation. Our work directly addresses this gap by introducing two Arabic NLI datasets and establishing a comprehensive evaluation framework across 5 cutting-edge LLMs and 10 foundational pre-trained models.

## III. E-CONAN Datasets

## a. DATASETS CONSTRUCTION

We have created two datasets: a 2-way dataset (RTE) named E-CONAN-2 and a 3-way dataset (NLI) named E-CONAN-3. These datasets were collected from several sources as follows: 1- ArNLI dataset [32] which contains: ArNLI dataset [32] which contains:

• Manually collected pairs.

ArbTEDS dataset<sup>2</sup> [30] which contains news collected from Arabic news channels, labeled manually by eight annotators, where an annotator agrees with at least one co-annotator (average around 91% between annotators.

Automatically translated pairs from English NLI datasets (SICK3 [39], PHEME4 [40], and Stanford Real Life contradiction corpus<sup>5</sup>[41]), then manually verified. Where the SICK inter–rater agreement was 84%, as an average, 84% of participants agreed with the majority vote in each pair.

2- The Arabic-translated section of SNLI dataset<sup>6</sup> [14], which contains the first 1,332 test pairs of SNLI dataset [10], manually translated by human experts. Each pair was validated by five independent annotators. It achieved a consensus rate of 98% for three annotators and a 58% consensus rate from all five annotators.

3- The Arabic-translated section of XNLI dataset [42], which was constructed as an evaluation set for Cross-lingual Language Understanding (XLU) by extending the development and test sets of the Multi-Genre Natural Language Inference Corpus (MultiNLI) to 15 languages, including lowresource languages such as Arabic. To ensure transparency regarding data quality, the authors reported specific translation metrics: an Ar-En BLEU score of 35.2, an En-Ar BLEU of 15.8, and a Word Translation P@1 of 51.9. Each pair was validated by five independent annotators. It achieved a consensus rate of 93% for three annotators.

4- AnsStance Dataset [43]: A previous work, which involved transforming stance datasets into Natural Language Inference (NLI) datasets, indicated that AnsStance's language complexity and word counts align closely with our collection. For AnsStance dataset inclusion, we considered s1, s2 as Text and Hypothesis, respectively.

5- ArEntail Dataset [31], contains sentence pairs collected from Arabic news headlines and manually labelled to indicate entailment / not entailment. This source was only used in the E-CONAN-2 dataset.

## b. Datasets Annotation Mapping

As for E-CONAN-3, the original NLI annotations were preserved without modification. To ensure consistency across combined datasets, we unified the AnsStance pair annotations by mapping them to the standard NLI labels, where the pairs were reannotated as follows:

"agree" class was converted to "entailment" class.

"disagree" class was converted to "contradiction" class.

all remaining classes were converted to "neutral" class.

5 "Stanford Real Life contradiction corpus", 2008: https://nlp.stanford.edu/projects/contradiction/ 6 "Arabic SNLI Dataset", 2018: https://bitbucket.org/nlpitu/xnli/src/master/

As for E-CONAN-2, All 2-way sources were preserved without modification. For sources containing 3-way annotations, we performed a label transformation to align them with a binary 2- way classification. The mapping was conducted as follows:

"entail" class was preserved without modification.

"contradiction" class and "neutral" class were merged in “not-entail” class.

A workflow diagram is shown in Figure 2 that illustrates the full pipeline from raw data sources to the final unified datasets.

Although source datasets were independently validated in their original studies, we acknowledge that inter-annotator agreement was not re-evaluated post-merging. This absence remains a limitation of this study.

## c. Datasets Statistics

E-CONAN-2 dataset contains 37% of samples in the class entail, 63% in the class not-entail. While, E-CONAN-3 dataset contains 32% of samples in the class entailment, 34% in the class neutral, and 34% in the class contradiction. Number of samples for each class in each dataset are shown in Table 2.

Statistics on lexical characteristics in Text and Hypothesis are shown in Table 3. It shows a minimum of 2 words by sentence and a maximum of 39 and 59 words in Text, Hypothesis, respectively. The average length of sentence is approximately 8 and 11 in Text, Hypothesis, respectively. This broad range of sentence lengths indicates the presence of both short and concise expressions as well as longer, information-rich constructions, contributing to lexical and structural diversity. The similarity of these statistics across E-CONAN-2 and E-CONAN-3 also demonstrates consistency in dataset construction while preserving linguistic variability.

Figure 3 illustrates the number of samples from each of our data sources. Notably, the highest number of samples are from XNLI and ArNLI. The lowest number of samples is from SNLI. ArEntail is exclusively used in the E-CONAN-2 dataset due to its binary classification of entailment versus non-entailment, as it does not categorize non-entailment into contradiction or neutral.

Figure 4 shows the domain distribution & semantic coverage of E-CONAN datasets, where the datasets span four distinct text styles: News (35%), Simple Everyday (30%), Social Media (20%), and Multi-Genre (15%), compiled from eight source datasets, including ArbTEDS, ArEntail, SNLI, SICK, PEHEM, AnsStance, XNLI, and Stanford Contradiction. The News domain (35%) contributes formal vocabulary, relatively complex syntactic structures, and information-dense content. The Simple Everyday Text domain (30%) provides straightforward lexical patterns suitable for evaluating baseline reasoning capabilities. Social Media data (20%) introduces informal language, while the Multi-Genre domain (15%) expands cross-domain semantic coverage. Collectively, this heterogeneous composition promotes lexical diversity and broad semantic representation, reducing the likelihood of models adapting to a narrow writing style or vocabulary.

Figure 5 shows the word cloud of E-CONAN-2 and E-CONAN-3, highlighting that it contains words from diverse domains such as geographical words $( \cos \alpha \sin \alpha ] 1 \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha , \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \cos \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \sin \alpha \ c o r c o r c o r c o m a \ c o m a l o r c o r c o m m m a l \ c o r c o m m a l \ c o r a l o r a l o r a l o r a l a l a l \ c o r a l o r a l o r a l a l \ c o r a l a l a l o r a l a l a l a l a t o r a l a l a l a l a l a l a l a l a l a t o r a l a l a l a l a l a n \ c o r a l a l a l a l a n c o r a l a l a l a l a n c o r a l a l a l a l a l a l a l a n c o r a l a l a l a l a l a l a n c o r a n c o r a l a n c o r a l a l a l a l a l a n c o r a l a l a l a l a l a l a l a l a l a l a l a l a l a n c o r a l a n c o r a l a l a l a l a l a l a l a l a l a l a n c o r a l a l a l a l a l a l a l a l a l a n c o r a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l a l$ $^ { 6 6 } \approx 2 0 0 0 ^ { 3 9 } \cdots )$ words political , $( \cdots \sin \alpha , \cos \alpha 1 1 ^ { \circ } , \cot \alpha 1 2 ^ { \circ } , \cos \alpha ) 2 ^ { \circ }$ words economical ,…) $( ^ { 6 6 } \div 3 5 ^ { 3 3 } , \cdots ) 2 3 0 ^ { 3 3 } , ^ { 4 } ) \cup ( 2 0 ) ^ { 6 }$ words general ,…) $( 4 ) _ { 2 } 3 ^ { 3 } , 5 6 5 6 1 0 1 3 , 5 6 5 6 1 3 , 5 6 , 1 9 3$ $w _ { 1 } = 1 2 . 5 6 . 5 6 \div 6 \div 2 = 3 6 . 5 6 ( \lambda )$

TABLE 2  
E-CONAN DATASETS CLASSES DISTRIBUTION
<table><tr><td>Dataset</td><td>Class</td><td># Samples</td></tr><tr><td rowspan="3">E-CONAN-2</td><td>Entail</td><td>9157</td></tr><tr><td>Not-Entail</td><td>15718</td></tr><tr><td>Total Pairs</td><td>24875</td></tr><tr><td rowspan="4">E-CONAN-3</td><td>Entailment</td><td>6157</td></tr><tr><td>Contradiction</td><td>6357</td></tr><tr><td>Neutral</td><td>6361</td></tr><tr><td>Total Pairs</td><td>18875</td></tr></table>

![](images/5dadcf9e68eec05829e1489d1243effdb9380ac99f2774129faa54b853f44fe1.jpg)

Figure3: E-CONAN Datasets Statistics of Sources Distribution  
![](images/a44b6a01c39b874088787d72b4b5f9c7d15ce1422d956861f89c3d10f9dd3542.jpg)  
Figure4: E-CONAN Datasets Statistics of Domain Distribution

![](images/15f7a8a864359c15dde50f050a2cc9258b8ee1727874ba05aaa9db990fc4d6e1.jpg)

We have conducted quantitative comparisons of state-ofthe-art (SoTA) benchmarks with E-CONAN-2, E-CONAN-3 benchmarks, as presented in Table 4 and Table 5, respectively. As shown in Table 4 and Table 5, our datasets maintain a robust distribution across classes. Notably, E-CONAN-3 achieves a highly optimized, nearperfect balance across the three standard NLI classes (Entailment: \~32.6%, Contradiction: \~33.7%, Neutral: \~33.7%), resolving the distribution skews found in older benchmarks like ArNLI.

Figure5: E-CONAN Word Cloud  
LEXICAL CHARACTERISTICS IN E-CONAN-2 AND E-CONAN-3  
TABLE 3
<table><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2>E-CONAN-2</td><td rowspan=1 colspan=2>E-CONAN-3</td></tr><tr><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>T</td><td rowspan=1 colspan=1>H</td></tr><tr><td rowspan=1 colspan=1>Max</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>59</td><td rowspan=1 colspan=1>39</td><td rowspan=1 colspan=1>59</td></tr><tr><td rowspan=1 colspan=1>Average</td><td rowspan=1 colspan=1>8.57</td><td rowspan=1 colspan=1>11.62</td><td rowspan=1 colspan=1>8.04</td><td rowspan=1 colspan=1>11.37</td></tr><tr><td rowspan=1 colspan=1>Min</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>2</td></tr></table>

TABLE 4  
QUANTITATIVE COMPARISONS OF E-CONAN-2 WITH SOTA BENCHMARK
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>ArEntail</td><td rowspan=1 colspan=1>E-CONAN-2</td></tr><tr><td rowspan=1 colspan=1>Entail</td><td rowspan=1 colspan=1>3000</td><td rowspan=1 colspan=1>9157</td></tr><tr><td rowspan=1 colspan=1>Not-Entail</td><td rowspan=1 colspan=1>3000</td><td rowspan=1 colspan=1>15718</td></tr><tr><td rowspan=1 colspan=1>Total pairs</td><td rowspan=1 colspan=1>6000</td><td rowspan=1 colspan=1>24875</td></tr></table>

TABLE 5

QUANTITATIVE COMPARISONS OF E-CONAN-3 WITH SOTA BENCHMARK
<table><tr><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>ArNLI</td><td rowspan=1 colspan=1>ArSNLI</td><td rowspan=1 colspan=1>ArXNLITest +Validation</td><td rowspan=1 colspan=1>E-CONAN-3</td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>1928</td><td rowspan=1 colspan=1>450</td><td rowspan=1 colspan=1>2485</td><td rowspan=1 colspan=1>6157</td></tr><tr><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1>1068</td><td rowspan=1 colspan=1>429</td><td rowspan=1 colspan=1>2486</td><td rowspan=1 colspan=1>6357</td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>3351</td><td rowspan=1 colspan=1>434</td><td rowspan=1 colspan=1>2491</td><td rowspan=1 colspan=1>6361</td></tr><tr><td rowspan=1 colspan=1>Total pairs</td><td rowspan=1 colspan=1>6347</td><td rowspan=1 colspan=1>1313</td><td rowspan=1 colspan=1>7462</td><td rowspan=1 colspan=1>18875</td></tr></table>

Beyond dataset size, the statistics of E-CONAN datasets indicate substantial linguistic diversity, semantic coverage, and class balance. The source distribution analysis reveals that E-CONAN integrates data from eight datasets spanning four domains: News, Simple Everyday Text, Social Media, and Multi-Genre. This heterogeneous composition promotes lexical diversity and broad semantic representation. In addition, E-CONAN incorporates sentence pairs originating from multiple construction paradigms, including automatically translated pairs, human-validated machine-translated pairs, hand-crafted educational examples, and news headline pairs involving rumors and factual inconsistencies. These diverse datageneration processes further broaden the linguistic and semantic coverage of the benchmark.

The lexical statistics indicate substantial variation in sentence lengths, ranging from short expressions to longer information-rich constructions. This variability contributes to lexical and structural diversity while maintaining consistency across E-CONAN-2 and E-CONAN-3.

Finally, the class distributions demonstrate improvements over several existing Arabic NLI benchmarks. In particular, E-CONAN-3 maintains a highly balanced distribution across the Entailment, Contradiction, and Neutral classes, providing a more balanced benchmark for evaluating semantic alignment and inference performance.

## IV. Evaluation Models

We used our dataset E-CONAN to evaluate some famous state-of-the-art pretrained models that cover Arabic language without any training examples (zeroshot classification), as shown in Figure 6. Most of these pretrained models were trained on MNLI only or on both MNLI and XNLI.

We used the following models:

Facebook Bart Large MNLI<sup>7</sup> [33], [34].

Moritzlaurer MiniLM-L6-MNLI8: Microsoft Multilingual MiniLM [36] tuned on MNLI dataset [11].

## a. Pretrained Models

Moritzlaurer Deberta-V3-Base-MNLI<sup>9</sup>: Microsoft DeBERTaV3 [37] tuned on MNLI dataset [11].

Moritzlaurer-mDeBERTa-V3-Base-XNLI-Multilingual-NLI-2mil7<sup>11</sup> [38]: a tuned version of a Microsoft mDeBERTa-v3-base [37] model, which was trained on CC100 multilingual dataset [44][45], tuned on both MNLI [11] and multilingual-NLI-26lang-2mil7 [38]datasets.

Moritzlaurer-mDeBERTa-V3-Base-MNLI-XNLI<sup>10</sup>[38]: a tuned version of a Microsoft mDeBERTa-v3-base [37] model, which was trained on CC100 multilingual dataset [44][45] with 100 different languages, tuned on both MNLI [11] and XNLI [42] datasets.

FacebookAI RoBERTa-Large-MNLI<sup>12</sup>[35]: which was trained on MNLI dataset [11].

Moritzlaurer Ernie-M-Base-MNLI-XNLI<sup>13</sup> [38]: a tuned version of Meta’s RoBERTa model, tuned on both MNLI [11] and XNLI [42] datasets.

MoritzlaurerMultilingual-MiniLMv2-L12-MNLI-XNLI<sup>15</sup>[38]: a Microsoft XLM-RoBERTa [36] model tuned on both MNLI [11] and XNLI [42] datasets.

Moritzlaurer Multilingual-MiniLMv2-L6-MNLI-XNLI<sup>14</sup> [38]: a Microsoft XLM-RoBERTa [36] model tuned on both MNLI [11] and XNLI [42] datasets.

Furthermore, to evaluate the performance of crosslingual models against Arabic pre-trained models, we utilized a version<sup>16</sup> of MARBERT [46] that was finetuned on the Arabic XNLI dataset (as illustrated in Figure 7). Specifically, we conducted experiments to compare the results of Arabic-specific pre-trained models with those of multilingual models.

## b. LLMs

Furthermore, we used our dataset E-CONAN to evaluate some famous state-of-the-art large language models that cover Arabic language, using zero-shot classification (See Figure 8). The comparison was based on the following LLMs:

1- ALLAM [46] ALLaM stands for Arabic Large Language Model, a series of large language models to support the ecosystem of Arabic Language Technologies (ALT). ALLAM models are based on an autoregressive decoder-only architecture and are pretrained on a mixture of Arabic and English texts via vocabulary expansion.

2- Qwen2.5 [47], [48] is a comprehensive series of large language models (LLMs) developed by Alibaba Cloud, based on a Transformer-based decoder-only architecture. The series includes both open-weight dense models (0.5B to 72B parameters) and proprietary Mixture-of-Experts (MoE) variants (Qwen2.5-Turbo and Qwen2.5- Plus).

3- Gemma3 [52]: an open-weight, multimodal model developed by Google. Ranging in size from 1 to 27 billion parameters. It has 128K-token context window. It supports multilingual (140+ languages) and multimodal input (text and images).

4- Command R7B Arabic [53] 7-billion parameter, open-weights, multilingual LLM by Cohere Labs The model has been trained and evaluated for performance in Arabic and English, but its training data includes samples from other languages. Command R7B Arabic supports a context length of 128,000 tokens. It has been trained specifically for tasks such as the generation step of Retrieval Augmented Generation (RAG) in Arabic and English.

5- DeepSeek-R1 [54]: A large-scale, open-weight Mixture-of-Experts model from DeepSeek AI (with 671B total parameters and ≈37B active per query). It is optimized for complex reasoning and logical tasks (e.g., math, coding, scientific reasoning). The maximum generation length is set to 32,768 tokens. Its pipeline incorporates two RL stages aimed at discovering improved reasoning patterns and aligning with human preferences.

## c. Experiments Setup:

We conducted all experiments using a zero-shot inference. To ensure fairness, transparency, and reproducibility, all models were evaluated using the same configuration without fine-tuning, fewshot examples, or task-specific adaptations. The decoding parameters were maintained strictly at their standard pre-configured default settings across all models. Specifically, we used a temperature of 0, and disabled penalty restrictions to guarantee deterministic and consistent outputs.

As for LLMs prompt, the inference was driven by a structured zero-shot prompt command provided to the models together with each text pair. The exact prompt template employed for the evaluation is formulated as follows: "Having the following sentences Text: {t} and Hypothesis {h} Answer: Let's classify the relation as one of the following classes ['entailment', 'contradiction', 'neutral']."

## V. Results and Discussion

As agreed in General Language Understanding Evaluation benchmark (GLUE) [12], the evaluation metrics for NLI is accuracy. For this reason, accuracy is the main metric that is used in all experiments.

## a. Pretrained Results

Accuracy percentages of cross-lingual pretrained models zero-shot classification results are shown for all studied datasets in Tables 6, 7. Macro precision, recall and F1 results are illustrated in Figures 9, 10 on ArNLI, E-CONAN and XNLI datasets respectively.

TABLE 6  
ACCURACY COMPARISONS WITH SOTA NLI ZERO-SHOT CLASSIFICATION
<table><tr><td rowspan=1 colspan=1>Model Name</td><td rowspan=1 colspan=1>ArXNLITest</td><td rowspan=1 colspan=1>ArXNLITest +Validation</td><td rowspan=1 colspan=1>ArNLI</td><td rowspan=1 colspan=1>E-CONAN-3</td></tr><tr><td rowspan=1 colspan=1>FacebookAI/roberta-large-mnli</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.23</td><td rowspan=1 colspan=1>0.34</td></tr><tr><td rowspan=1 colspan=1>Facebook/bart-large-mnli</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.25</td><td rowspan=1 colspan=1>0.36</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/MiniLM-L6-mnli</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.26</td><td rowspan=1 colspan=1>0.33</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/mDeBERTa-v3-base-mnli-xnli</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.70</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.61</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/DeBERTa-v3-base-mnli</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.46</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.64</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/ernie-m-base-mnli-xnli</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.68</td></tr></table>

TABLE 7  
ACCURACY COMPARISONS WITH SOTA RTE ZERO-SHOT
<table><tr><td rowspan=1 colspan=1>Model Name</td><td rowspan=1 colspan=1>ArEntail</td><td rowspan=1 colspan=1>E-CONAN-2</td></tr><tr><td rowspan=1 colspan=1>FacebookAI/roberta-large-mnli</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.5</td></tr><tr><td rowspan=1 colspan=1>Facebook/bart-large-mnli</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.66</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.84</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/MiniLM-L6-mnli</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.47</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/mDeBERTa-v3-base-mnli-xnli</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/DeBERTa-v3-base-mnli</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.6</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.77</td></tr><tr><td rowspan=1 colspan=1>MoritzLaurer/ernie-m-base-mnli-xnli</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.81</td></tr></table>

As for datasets comparisons, we note that lowest results were on ArNLI dataset. ArNLI is probably the hardest dataset as it only contains complex sentences that were carefully checked by humans. These sentences were either manually translated or manually created from scratch. On the other hand, the highest results were on XNLI dataset. This is likely due to the fact that many models were trained or fine-tuned on XNLI, or on similar machine-translated pairs. The E-CONAN datasets results were somewhere in the middle, as they include a mix of sentences: a part that contains machine-translated pairs checked by humans, an automatically translated part, a part collected from news articles and a part created from scratch by humans. As E-CONAN datasets offer this variety, we believe that it could be a good base to test how well models can generalize, or even to improve pre-trained models.

As for models’ evaluation comparisons, results show that the mDeBERTa model, pre-trained on 100 languages and fine-tuned on a combination of four machine-translated datasets, outperformed all other models on all datasets. Additionally, results show that all multilingual pre-trained models tuned both on XNLI and MNLI perform better than models tuned on MNLI alone.

## b. LLM Results:

Best result on ArNLI dataset was achieved by Qwen2.5 with an accuracy of 64% and lowest result was by Allam with an accuracy of 50%. Best result on AnsStance dataset was achieved by Gemma3 with an accuracy of 86% and lowest result was by Allam with an accuracy of 68%.

Best result on XNLI dataset was achieved by Gemma3 with an accuracy of 68% and lowest result was by Allam with an accuracy of 56%. Best result on SNLI dataset was achieved by Gemma3 with an accuracy of 48% and lowest result was by Allam with an accuracy of 32%. Best result on E-CONAN-3 dataset was achieved by Gemma3 with an accuracy of 68% and lowest result was by Allam with an accuracy of 55%.

As for results on E-CONAN-3 as 2-way, best result was achieved by Gemma3, Qwen2.5 with an accuracy of 95%, 94% respectively, and lowest result was by Allam with an accuracy of 66%.

TABLE 8  
COMPARISON OF LLM ZERO-SHOT CLASSIFICATION ACCURACY ON E-CONAN-3 DATASET & ALL ITS SOURCES
<table><tr><td rowspan=1 colspan=1>LLM</td><td rowspan=1 colspan=1>AnsStance</td><td rowspan=1 colspan=1>ArNLI</td><td rowspan=1 colspan=1>ArXNLITest+Validation</td><td rowspan=1 colspan=1>SNLI</td><td rowspan=1 colspan=1>E-CONAN-3</td><td rowspan=1 colspan=1>E-CONAN-3as 2-way</td></tr><tr><td rowspan=1 colspan=1>Allam</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.5</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.66</td></tr><tr><td rowspan=1 colspan=1>CommandR7B Arabic</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.92</td></tr><tr><td rowspan=1 colspan=1>DeepSeek</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.87</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.95</td></tr><tr><td rowspan=1 colspan=1>Qwen2.5</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.94</td></tr></table>

Macro precision, recall and F1 results of all used LLM on our created benchmarks and SoTA benchmarks are illustrated in Figures 11, 12, and 13.

When analyzing results, we notice that most errors were cause by the confusion between neutral and contradiction classes. Consequently, we did an experiment converting 3- way dataset to 2-way dataset (by merging contradiction and neutral into ‘not-entail’) and the obtained accuracy increased by at least 15% to 30%. Where Allam accuracy increased from %55 to %66, Command R7B Arabic accuracy increased from %64 to %92, DeepSeek accuracy increased from %61 to %87, Gemma3 accuracy increased from %.68 to %95, and Qwen2.5 accuracy increased from %64 to %94.

Moreover, we noticed that highest results were achieved on AnsStance and the lowest results were on SNLI. We think that this might be related to the pretrained dataset domain, as most of our studied models were pretrained on much more news data compared to logical data. In summary, the results highlight the importance of diverse, high-quality benchmarks for accurately assessing and improving model generalization, particularly in resource-limited languages such as Arabic. We do hope that future researches make use of E-CONAN datasets to further refine multilingual models to enhance RTE and NLI performance in diverse linguistic contexts.

Additionally, we incorporated MARBERT as a representative Arabic-specific baseline to enrich our comparative analysis. This shows a performance evaluation comparison between an Arabic-centric model, the topperforming cross-lingual model (mDeBERTa-v3-base-xnli multilingual-nli-2mil7), and the top-performing LLM(Gemma). As detailed in Tables 9 and 10, this comparison demonstrates how Arabic-specific models scale against cross-lingual and LLM-based approaches on the E-CONAN benchmarks. Our results indicate that Gemma achieved the highest performance across the majority of the datasets. Additionally, we observed that performance on 2-way classification significantly outperformed 3-way results; most errors in the 3-way setting occurred between the "neutral" and "contradiction" classes, which are merged into "not-entailment" in the 2- way configuration.

Regarding specific datasets, Gemma provided the best results on the most challenging dataset (ArNLI), outperforming both Arabic-specific and cross-lingual pretrained models by a significant margin. Conversely, mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 yielded the best results on the ArXNLI dataset. This superior performance may be due to the model’s prior training on MNLI in addition to XNLI, which likely contributed to its high scores on the E-CONAN-3 dataset.

To confirm the validity of our zero-shot evaluation shown in Tables 9, 10 on all used datasets, we performed a pairwise McNemar’s test with a Bonferroni correction for multiple comparisons. The statistical test comparing Gemma3 against mDeBERTa-v3-base-xnli-multilingualnli-2mil7on all datasets yielded a p-value <0.000001. This extremely low p-value confirms that the performance differences are highly statistically significant and mathematically meaningful.

TABLE 9  
COMPARATIVE ANALYSIS OF REPRESENTATIVE MODELS ON E-CONAN-3 AND SOTA BENCHMARKS
<table><tr><td rowspan=1 colspan=1>Dataset Name</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=3 colspan=1>ArNLI</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.44</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.44</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.54</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.67</td></tr><tr><td rowspan=3 colspan=1>ArXNLITest +Validation</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.72</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.86</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.67</td></tr><tr><td rowspan=3 colspan=1>ArSNLI</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.41</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.40</td><td rowspan=1 colspan=1>0.34</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.48</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.49</td></tr><tr><td rowspan=3 colspan=1>AnsStance</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.63</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.61</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.77</td></tr><tr><td rowspan=3 colspan=1>E-CONAN-3</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.61</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.71</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.67</td></tr></table>

COMPARATIVE ANALYSIS OF REPRESENTATIVE MODELS ON E-CONAN-2 AND SOTA BENCHMARKS
<table><tr><td rowspan=1 colspan=1>DatasetName</td><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Accuracy</td><td rowspan=1 colspan=1>Precision</td><td rowspan=1 colspan=1>Recall</td><td rowspan=1 colspan=1>F1</td></tr><tr><td rowspan=3 colspan=1>ArEntail</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.88</td></tr><tr><td rowspan=3 colspan=1>E-CONAN-2</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.74</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.83</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.95</td></tr></table>

## d. Zero-Shot Cross-Dataset Transfer Analysis

To demonstrate the utility, robustness, and challenging nature of our proposed E-CONAN-3 and E-CONAN-2 benchmarks, we conduct a cross-dataset transfer experiment. We tested nine state-of-the-art pretrained models originally trained on standard datasets (MNLI and XNLI). These models are evaluated directly on our newly introduced benchmarks, XNLI, and ArNLI as reference baselines. The empirical results are detailed in Table 6 and Table 7, respectively.

The evaluation yields several key insights regarding how well existing SOTA models transfer to data distributions:

The Generalization Gap: While multilingual models like MoritzLaurer/mDeBERTa-v3-basemnli-xnli perform exceptionally well on standard XNLI (achieving an accuracy of 0.87), their performance drops significantly to 0.70 when evaluated on E-CONAN-3. This performance degradation clearly indicates that our dataset introduce unique contextual distributions and complex semantic challenges that standard XNLI training does not fully capture.

Architecture Limitations: Conversely, Englishcentric models (such as RoBERTa and BART variants) exhibit very poor cross-dataset efficacy, failing to cross the 0.36 accuracy threshold on E-CONAN-3. Meanwhile, robust multilingual pipelines (mDeBERTa-v3 and ernie-m-base) retain moderate generalization capabilities (scoring 0.71 and 0.68, respectively), yet still reveal substantial room for improvement.

These cross-dataset transfer results validate the distinct value of E-CONAN-2 and E-CONAN-3 as benchmarks. By testing models without any supervised fine-tuning or parameter tuning, our evaluation successfully exposes the true boundaries of current state-of-the-art models, proving that high performance on traditional NLI datasets does not guarantee seamless adaptation to new, specialized domains.

## VI. Error Analysis

We conducted a detailed error analysis to better understand models’ performance. This included a qualitative evaluation via manual inspection to identify error patterns, as well as a quantitative analysis to measure the frequency of those patterns.

## a. Qualitative Error Analysis

To investigate the root causes of models’ failure, we conducted a manual qualitative analysis, identifying four primary failure modes that persist across both pretrained models and LLMs. (See representative samples in Table 11)

Lexical Overlap and Heuristic Bias: most models show a strong lexical overlap heuristic bias, frequently predicting Entailment when the premise text (T) and hypothesis (H) share highfrequency tokens, regardless of their logical relationship. For instance, when T describes "two dogs running" ( يركضان كلبان ( and H adds a specific location like "in the garden" $( 4 0 , 1 2 1 , 9 )$ (, the model predicts Entailment. This suggests that the model relies on bag-of-words similarity rather than recognizing that H contains new, unverified information, which logically requires a Neutral label.

Entity Mismatch and Lack of World Knowledge: A significant failure mode where models fail to distinguish between conflicting entities despite sharing the same action verbs. This is frequently observed across numerous instances; for example, where the model predicts Entailment for T: "Assad's children spent their vacation in a Russian camp" and H: "Assad's children spent their vacation in America". The model focus on the event type ("vacation") while ignoring the geographic contradiction between "Russia" and "America". This confirms a lack of world knowledge, where named entities are treated as identical tokens rather than distinct locations.

Linguistic Challenges: The complex morphology of Arabic introduces specific challenges regarding agreement and semantic:

Gender Disagreement (Morphological Mismatch): a frequent pattern observed in many cases where models often fail to recognize that the gender shift implies a different set of entities. For instance, when T involves "two young men sitting" $( i \omega ! \xrightarrow { } i 4 . 4 4 )$ and H involves "two young women sitting" ( تجلسان شابتان (, most models wrongly predict Entailment instead of Neutral.

Quantifier and Numerical Constraints: most models struggle with the logical boundaries of quantifiers. For example, the transition from a general plural in T ("children in the park") to a specific count in H ("three children in the park") is frequently misclassified as Entailment, ignoring the lack of numerical evidence in the premise text.

o Antonymy and Negation Blindness: many models demonstrate antonym blindness in the presence of high lexical overlap. For example, T: "lifting the "curfew $( 1 ) \sin ( \frac { \pi } { 2 } )$ :H and "continuation of the curfew" $P ( \mathcal { S } ^ { \bot } \Delta ) , P ^ { \bot } \Delta \Delta 1$ التجول ( is often misidentified as Entailment due to the shared "curfew" token, despite the explicit semantic contradiction.

Topical Hallucination: We found that models often get distracted by shared topics and emotions, leading to semantic over-generalization or overspecification. If two sentences share a keyword and an emotion, the model tends to mark them as Entailment, even if the facts don't align. For instance, some models link a premise text about a reporter's work to a hypothesis about a reporter's death just because they feel related, failing to see that there is no logical connection between the two. This tendency leads the model to jump to conclusions by adding specific details that don't exist in the text. For example, many models predict Entailment for T: "two people riding bikes" and H: "two people are in competitive race" as the model associates biking with racing, it assumes the specific context is true, failing to realize that it has moved from a general fact to an unverified guess.

QUALITATIVE ERROR ANALYSIS

TABLE 11
<table><tr><td rowspan=1 colspan=1>Prediction</td><td rowspan=1 colspan=1>Class</td><td rowspan=1 colspan=1>Hypothesis(H)</td><td rowspan=1 colspan=1>PremiseText (T)</td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1> $\cot \frac { \cos \angle \sin \angle A } { \cos \angle \cos \angle A }$ se</td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=1> </td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1> $( 3 5 \lfloor \frac { 3 } { 2 } \rfloor ) ( \frac { 2 } { 2 } ) \geq 3 \lfloor \frac { 3 } { 2 } \rfloor$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1> $1 0 0 0 0$ </td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Entailment</td><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Neutral</td><td rowspan=1 colspan=1>Contradiction</td><td rowspan=1 colspan=1> </td><td rowspan=1 colspan=1></td></tr></table>

## b. Quantitative Error Analysis

To better understand the characteristics of the challenges in Arabic NLI, we conducted a statistical error analysis focusing on representative models that showed significant performance gaps. Specifically, we analyzed MoritzLaurer/MiniLM-L6-mnli (representing cross-lingual pretrained models) and Allam (representing Large Language Models). By examining a representative subsample of errors from these two models, we identified frequent failure modes that highlight the divergence between traditional fine-tuned models and zero-shot LLM performance (see Figure 14).

Results show that while pretrained models are often misled by lexical overlap, LLMs are often misled by topical familiarity. This suggests that for Arabic NLI, both architectures prioritize similarity over logical consistency. Furthermore, both approaches suffer from a bias toward the Entailment class.

![](images/588bf2f10b8759a9bfafad7a2a85b110517bfc43f395fe569344002251fcfb6f.jpg)  
Figure14: Quantitative Error Analysis

## c. Class-Wise Performance and Error Patterns

While overall accuracy provides a useful snapshot of models’ performance, it does not tell how well the models handle specific types of linguistic relationships. To gain a deeper understanding, we studied class-wise performance using the classification reports (Table 12) and confusion matrices (Figure 15). This analysis highlights models strengths, common error patterns, and the robustness of proposed benchmarks across different linguistic relations.

## • Challenge of Neutral and Contradiction Classes:

Despite the overall strength of encoder-based models, a closer inspection of class-wise results highlights that the neutral and contradiction classes remain particularly challenging across all architectures. As shown in the confusion matrices (Figure 15), there is consistent misclassification between these two categories, with a noticeable tendency for both contradiction and entailment instances to be predicted as neutral.

Although Gemma3 achieves competitive overall accuracy, a closer examination of class-wise performance reveals a tendency to favor the majority class in more challenging inference scenarios. This behavior is particularly evident on ArSNLI and ArNLI, where the model frequently predicts the Neutral label. Consequently, while Neutral recall remains relatively high, performance on Contradiction and Entailment decreases, indicating difficulty in distinguishing between semantic classes in harder cases.

This difficulty can be attributed to several factors:

Subjective annotation boundaries: the datasets integrated into E-CONAN-3 rely heavily on crowdsourced annotations, where distinguishing between contradiction and neutral cases is often subjective. In particular, determining whether missing information implies non-contradiction (neutral) or direct conflict (contradiction) introduces ambiguity that is difficult for models to resolve consistently.

Linguistic and translation artifacts: since benchmarks such as XNLI and SICK include translated or machine-translated Arabic data, some of the original pragmatic and cultural cues may be lost. This can reduce the clarity of semantic contradictions, leading models to favor the neutral class when evidence for a strong conflict is not explicit.

Lexical overlap bias: models also tend to rely on surface-level lexical overlap as a heuristic. When premise and hypothesis share similar vocabulary but differ due to negation or indirect contextual shifts, models often struggle to capture the semantic difference, resulting in frequent confusion between contradiction and neutral labels.

## Robustness, Semantic Diversity, and Fine-Tuning Potential of E-CONAN-3

The consistent zero-shot performance observed across all evaluated models on E-CONAN-3 suggests that the benchmark offers advantages over individual existing datasets. Whereas several prior benchmarks exhibit label-specific biases that can encourage shortcut-based predictions, the aggregation of multiple data sources within E-CONAN-3 appears to mitigate these effects. By combining diverse datasets without extensive filtering, E-CONAN-3 increases contextual and linguistic diversity while reducing the influence of dataset-specific stylistic patterns. This broader coverage is reflected in more balanced class-wise performance and clearer diagonal structures in the confusion matrices.

Despite these improvements, E-CONAN-3 still inherits certain limitations from its underlying source datasets. In particular, some degree of domain imbalance and topic distribution bias remains, as the benchmark is skewed toward specific structural genres. As shown in our domain breakdown (see Figure 4), the data is mainly composed of news (35%) and simple everyday (30%) contexts, while social media (20%) and multi-genre (15%) settings are comparatively underrepresented. In addition, since E-CONAN-3 is built from aggregated datasets such as SNLI, XNLI, and SICK, residual effects of translation bias and crowdsourcing-related annotation bias may still be present. Although combining multiple datasets can help reduce the impact of individual dataset biases compared to single-source benchmarks, these inherited limitations should still be taken into account when interpreting model generalization performance.

Acknowledging these factors, E-CONAN-3 still provides a strong foundation for future fine-tuning efforts, potentially promoting better generalization and more robust semantic understanding in Arabic Natural Language Inference models.

## VII. Conclusion

This paper addresses a critical gap in Arabic Textual Entailment and Natural Language Inference (NLI) benchmarks by introducing E-CONAN datasets composed of diverse sentence pairs from multiple sources. E-CONAN contains both 2-way (RTE) named E-CONAN-2 and 3-way (NLI) named E-CONAN-3. E-CONAN sources are combination of automatically translated pairs, humanvalidated machine-translated pairs, hand-crafted pairs, and news-based pairs, offering wide-ranging evaluation datasets from different types of texts. To evaluate E-CONAN, we conducted zero-shot classification on nine state-of-the-art multilingual pre-trained models. Results on E-CONAN datasets were compared with results on ArNLI and XNLI datasets. The challenging nature of the ArNLI dataset, characterized by its high-quality human-validated and manually crafted data, led to the lowest observed performance. On the other hand, the XNLI dataset, likely benefiting from its commonness in model training, yielded the highest results. E-CONAN demonstrated an intermediate difficulty level making it a valuable tool for assessing model generalization. Its diverse composition, unlike the singular nature of ArNLI and the potentially biased machine-translated origins of XNLI, provides a more balanced and representative evaluation. As for best pretrained models in this paper experiment, results show that mDeBERTa model consistently outperformed all other models on all datasets, achieving accuracies of 0.71 and 0.86 on E-CONAN-3 and XNLI datasets, respectively. This highlights the effectiveness of large-scale multilingual pretraining and targeted fine-tuning dataset.

Moreover, we evaluated 5 LLMs on E-CONAN-3 dataset. Best results were achieved by Gemma with an accuracy of 0.68. Additionally, after deep investigation we found that most errors were between neutral and contradiction classes. Thus, we calculated results on E-CONAN-3 as 2-way, where best results were achieved by Gemma and Qwen with an accuracy of 0.95, 0.94 respectively.

In addition, we incorporated MARBERT as a representative Arabic-specific baseline and conducted performance evaluation comparison to demonstrate how Arabic-specific models scale against cross-lingual and

https://tac.nist.gov/publications/2009/additional.papers/RTE5\_overvi ew.proceedings.pdf

LLM-based approaches on the E-CONAN benchmarks. Moreover, we conducted detailed qualitative and quantitative error analysis to analyze frequent error patterns. Results show that while pretrained models are often misled by lexical overlap, LLMs are often misled by topical familiarity.

In summary, E-CONAN datasets could be a good contribution to the Arabic NLI research domain, offering robust and diverse evaluation datasets. The results highlight the importance of diverse, high-quality benchmarks for accurately assessing and improving model generalization, particularly in resource-limited languages such as Arabic. We do hope that future researches make use of E-CONAN datasets to further refine multilingual models to enhance RTE and NLI performance in diverse linguistic contexts.

While current E-CONAN benchmarks provide a robust evaluation for Modern Standard Arabic (MSA), regional Arabic dialects are not currently represented. Future work will involve datasets expansion to include various dialects, ensuring broader coverage of the linguistic variations characteristic of Arabic language.

## Abbreviations

NLI: Natural Language Inference

RTE: Recognizing Textual Entailment

## Data Availability

E-CONAN datasets are available on Hugging Face<sup>17</sup>.

## REFERENCES

[1] D. Giampiccolo, H. T. Dang, B. Magnini, I. Dagan, E. Cabrio, and W. B. Dolan, “The Fourth PASCAL Recognizing Textual Entailment Challenge,” in Text Analysis Conference, 2008. [Online]. Available: https://api.semanticscholar.org/CorpusID:12381965

[2] O. and M. B. Dagan Ido and Glickman, “The PASCAL Recognising Textual Entailment Challenge,” in Machine Learning Challenges. Evaluating Predictive Uncertainty, Visual Object Classification, and Recognising Tectual Entailment, I. and M. B. and d’Alché-B. F. Quiñonero-Candela Joaquin and Dagan, Ed., Berlin, Heidelberg: Springer Berlin Heidelberg, 2006, pp. 177–190.

[3] R. Bar-Haim et al., “The Second PASCAL Recognising Textual Entailment Challenge,” 2006. [Online]. Available: https://api.semanticscholar.org/CorpusID:13385138

[4] D. Giampiccolo, B. Magnini, I. Dagan, and B. Dolan, “The Third PASCAL Recognizing Textual Entailment Challenge,” in Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, S. Sekine, K. Inui, I. Dagan, B. Dolan, D. Giampiccolo, and B. Magnini, Eds., Prague: Association for Computational Linguistics, Jun. 2007, pp. 1–9. [Online]. Available: https://aclanthology.org/W07-1401

[5] D. Giampiccolo, H. T. Dang, B. Magnini, I. Dagan, E. Cabrio, and W. B. Dolan, “The Fourth PASCAL Recognizing Textual Entailment Challenge,” in Text Analysis Conference, 2008. [Online]. Available: https://api.semanticscholar.org/CorpusID:12381965

[6] L. Bentivogli, B. Magnini, I. Dagan, H. T. Dang, and D. Giampiccolo, “The Fifth PASCAL Recognizing Textual Entailment Challenge,” in

Proceedings of the Second Text Analysis Conference, TAC 2009, Gaithersburg, Maryland, USA, November 16-17, 2009, NIST, 2009. [Online]. Available:

[7] L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo, “The Sixth PASCAL Recognizing Textual Entailment Challenge,” in Text Analysis Conference, 2009. [Online]. Available: https://api.semanticscholar.org/CorpusID:858065

[8] L. Bentivogli, P. Clark, I. Dagan, and D. Giampiccolo, “The Seventh PASCAL Recognizing Textual Entailment Challenge,” Theory and Applications of Categories, 2011, [Online]. Available: https://api.semanticscholar.org/CorpusID:5791809

[9] H. J. Levesque, E. Davis, and L. Morgenstern, “The Winograd Schema Challenge,” in AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning, 2011. [Online]. Available: https://api.semanticscholar.org/CorpusID:15710851

[10] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning, “A large annotated corpus for learning natural language inference,” in Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, L. Màrquez, C. Callison-Burch, and J. Su, Eds., Lisbon, Portugal: Association for Computational Linguistics, Sep. 2015, pp. 632–642. doi: 10.18653/v1/D15-1075.

[11] N. and B. S. Williams Adina and Nangia, “A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference,” in Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), Association for Computational Linguistics, 2018, pp. 1112–1122. [Online]. Available: http://aclweb.org/anthology/N18-1101

[12] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman, “GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding,” in Proceedings of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for NLP, Brussels, Belgium: Association for Computational Linguistics, Nov. 2018, pp. 353–355. doi: 10.18653/v1/W18-5446.

[13] A. Conneau et al., “XNLI: Evaluating Cross-lingual Sentence Representations,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium: Association for Computational Linguistics, Oct. 2018, pp. 2475–2485. doi: 10.18653/v1/D18-1269.

[14] Ž. Agić and N. Schluter, “Baselines and Test Data for Cross-Lingual Inference,” in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), N. Calzolari, K. Choukri, C. Cieri, T. Declerck, S. Goggi, K. Hasida, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, S. Piperidis, and T. Tokunaga, Eds., Miyazaki, Japan: European Language Resources Association (ELRA), May 2018. [Online]. Available: https://aclanthology.org/L18-1614

[15] A. Chowdhery et al., “PaLM: Scaling Language Modeling with Pathways,” ArXiv, vol. abs/2204.02311, 2022, [Online]. Available: https://api.semanticscholar.org/CorpusID:247951931

[16] Q. Zhong et al., “Toward Efficient Language Model Pretraining and Downstream Adaptation via Self-Evolution: A Case Study on SuperGLUE,” ArXiv, vol. abs/2212.01853, 2022, [Online]. Available: https://api.semanticscholar.org/CorpusID:254246784

[17] Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” 2019.

[18] Z. Zhang et al., “Semantics-aware BERT for Language Understanding,” in AAAI Conference on Artificial Intelligence, 2019. [Online]. Available:

[19] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V Le, “XLNet: Generalized Autoregressive Pretraining for Language Understanding,” in Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d Alché-Buc, E. Fox, and R. Garnett, Eds., Curran Associates, Inc., 2019. [Online]. Available:

https://proceedings.neurips.cc/paper\_files/paper/2019/file/dc6a7e655 d7e5840e66733e9ee67cc69-Paper.pdf

[20] S. Soltan et al., “AlexaTM 20B: Few-Shot Learning Using a Large-Scale Multilingual Seq2Seq Model,” ArXiv, vol. abs/2208.01448, 2022, [Online]. Available: https://api.semanticscholar.org/CorpusID:251253416

[21] S. Wu et al., “BloombergGPT: A Large Language Model for Finance,” ArXiv, vol. abs/2303.17564, 2023, [Online]. Available: https://api.semanticscholar.org/CorpusID:257833842

[22] S. Banerjee, A. Mahajan, A. Agarwal, and E. Singh, “First Train to Generate, then Generate to Train: UnitedSynT5 for Few-Shot NLI,” CoRR, vol. abs/2412.09263, 2024, doi: 10.48550/ARXIV.2412.09263.

[23] L. Xue et al., “ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte Models,” Trans Assoc Comput Linguist, vol. 10, pp. 291–306, 2022, doi: 10.1162/tacl\_a\_00461.

[24] H. W. Chung, T. Févry, H. Tsai, M. Johnson, and S. Ruder, “Rethinking embedding coupling in pre-trained language models,” ArXiv, vol. abs/2010.12821, 2020, [Online]. Available: https://api.semanticscholar.org/CorpusID:225067567

[25] O. Shliazhko, A. Fenogenova, M. Tikhonova, A. Kozlova, V. Mikhailov, and T. Shavrina, “mGPT: Few-Shot Learners Go Multilingual,” Trans Assoc Comput Linguist, vol. 12, pp. 58–79, 2024, doi: 10.1162/tacl\_a\_00633.

[26] P. He, X. Liu, J. Gao, and W. Chen, “DeBERTa: Decoding-enhanced BERT with Disentangled Attention,” 2021.

[27] M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, and O. Levy, “SpanBERT: Improving Pre-training by Representing and Predicting Spans,” Trans Assoc Comput Linguist, vol. 8, pp. 64–77, 2020, doi: 10.1162/tacl\_a\_00300.

[28] F. Iandola, A. Shaw, R. Krishna, and K. Keutzer, “SqueezeBERT: What can computer vision teach NLP about efficient neural networks?,” in Proceedings of SustaiNLP: Workshop on Simple and Efficient Natural Language Processing, N. S. Moosavi, A. Fan, V. Shwartz, G. Glavaš, S. Joty, A. Wang, and T. Wolf, Eds., Online: Association for Computational Linguistics, Nov. 2020, pp. 124–135. doi: 10.18653/v1/2020.sustainlp-1.17.

[29] V. Sanh, L. Debut, J. Chaumond, and T. Wolf, “DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter,” ArXiv, vol. abs/1910.01108, 2019, [Online]. Available: https://api.semanticscholar.org/CorpusID:203626972

[30] M. Alabbas, “A Dataset for Arabic Textual Entailment,” in Proceedings of the Student Research Workshop associated with RANLP 2013, I. Temnikova, I. Nikolova, and N. Konstantinova, Eds., Hissar, Bulgaria: INCOMA Ltd. Shoumen, BULGARIA, Sep. 2013, pp. 7–13. [Online]. Available: https://aclanthology.org/R13-2002

[31] R. Obeidat, Y. Al-Harahsheh, M. Al-Ayyoub, and M. Gharaibeh, “ArEntail: manually-curated Arabic natural language inference dataset from news headlines,” Lang Resour Eval, 2024, doi: 10.1007/s10579-024-09731-1.

[32] K. Al Jallad and N. Ghneim, “ARNLI: ARABIC NATURAL LANGUAGE INFERENCE ENTAILMENT AND CONTRADICTION DETECTION,” Computer Science, vol. 24, no. 2, Mar. 2023, doi: 10.7494/csci.2023.24.2.4378.

[33] W. Yin, J. Hay, and D. Roth, “Benchmarking Zero-shot Text Classification: Datasets, Evaluation and Entailment Approach,” in Proceedings of the 2019 Conference on Empirical Methods in Natura Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds., Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 3914–3923. doi: 10.18653/v1/D19-1404.

[34] M. Lewis et al., “BART: Denoising Sequence-to-Sequence Pretraining for Natural Language Generation, Translation, and Comprehension,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds., Online: Association for Computational Linguistics, Jul. 2020, pp. 7871–7880. doi: 10.18653/v1/2020.aclmain.703.

[35] Y. Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” ArXiv, vol. abs/1907.11692, 2019, [Online]. Available: https://api.semanticscholar.org/CorpusID:198953378

[36] W. Wang, F. Wei, L. Dong, H. Bao, N. Yang, and M. Zhou, “MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers,” ArXiv, vol. abs/2002.10957, 2020, [Online]. Available: https://api.semanticscholar.org/CorpusID:211296536

[37] P. He, J. Gao, and W. Chen, “DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing,” ArXiv, vol. abs/2111.09543, 2021, [Online]. Available: https://api.semanticscholar.org/CorpusID:244346093

[38] M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers, “Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and BERT-NLI,” Political Analysis, vol. 32, no. 1, pp. 84–100, Jan. 2024, doi: 10.1017/pan.2023.20.

[39] M. Marelli, L. Bentivogli, M. Baroni, R. Bernardi, S. Menini, and R. Zamparelli, “SemEval-2014 Task 1: Evaluation of Compositional Distributional Semantic Models on Full Sentences through Semantic Relatedness and Textual Entailment,” in Proceedings of the 8th International Workshop on Semantic Evaluation (SemEval 2014), P. Nakov and T. Zesch, Eds., Dublin, Ireland: Association for Computational Linguistics, Aug. 2014, pp. 1–8. doi: 10.3115/v1/S14- 2001.

[40] P. Lendvai, I. Augenstein, K. Bontcheva, and T. Declerck, “Monolingual Social Media Datasets for Detecting Contradiction and Entailment,” in Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC’16), N. Calzolari, K. Choukri, T. Declerck, S. Goggi, M. Grobelnik, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds., Portorož, Slovenia: European Language Resources Association (ELRA), May 2016, pp. 4602–4605. [Online]. Available: https://aclanthology.org/L16-1729

[41] M.-C. de Marneffe, A. N. Rafferty, and C. D. Manning, “Finding Contradictions in Text,” in Proceedings of ACL-08: HLT, J. D. Moore, S. Teufel, J. Allan, and S. Furui, Eds., Columbus, Ohio: Association for Computational Linguistics, Jun. 2008, pp. 1039–1047. [Online]. Available: https://aclanthology.org/P08-1118

[42] A. Conneau et al., “XNLI: Evaluating Cross-lingual Sentence Representations,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, Eds., Brussels, Belgium: Association for Computational Linguistics, Oct. 2018, pp. 2475–2485. doi: 10.18653/v1/D18-1269.

[43] J. Khouja, “Stance Prediction and Claim Verification: An Arabic Perspective,” in Proceedings of the Third Workshop on Fact Extraction and VERification (FEVER), C. Christodoulopoulos, J. Thorne, A. Vlachos, O. Cocarascu, and A. Mittal, Eds., Online: Association for Computational Linguistics, Jul. 2020, pp. 8–17. doi: 10.18653/v1/2020.fever-1.2.

[44] G. Wenzek et al., “CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data,” in Proceedings of the Twelfth Language Resources and Evaluation Conference, N. Calzolari, F. Béchet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, and S. Piperidis, Eds., Marseille, France: European Language Resources Association, May 2020, pp. 4003–4012. [Online]. Available: https://aclanthology.org/2020.lrec-1.494

[45] A. Conneau et al., “Unsupervised Cross-lingual Representation Learning at Scale,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, D. Jurafsky, J. Chai, N. Schluter, and J. Tetreault, Eds., Online: Association for Computational Linguistics, Jul. 2020, pp. 8440–8451. doi: 10.18653/v1/2020.aclmain.747.

[46] M. S. Bari et al., “ALLaM: Large Language Models for Arabic and English,” in The Thirteenth International Conference on Learning Representations, 2025. [Online]. Available: https://openreview.net/forum?id=MscdsFVZrN

[47] A. Yang et al., “Qwen2 Technical Report,” arXiv preprint arXiv:2407.10671, 2024.

[48] A. Yang et al., “Qwen2.5 Technical Report,” arXiv preprint arXiv:2412.15115, 2024.

[49] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual Instruction Tuning,” in Advances in Neural Information Processing Systems, A. Oh, T.

Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., Curran Associates, Inc., 2023, pp. 34892–34916. [Online]. Available: https://proceedings.neurips.cc/paper\_files/paper/2023/file/6dcf277ea 32ce3288914faf369fe6de0-Paper-Conference.pdf

[50] H. Liu et al., “LLaVA-NeXT: Improved reasoning, OCR, and world knowledge,” Jan. 2024. [Online]. Available: https://llavavl.github.io/blog/2024-01-30-llava-next/

[51] H. Liu, C. Li, Y. Li, and Y. J. Lee, “Improved Baselines with Visual Instruction Tuning,” 2023, arXiv:2310.03744.

[52] G. Team, “Gemma 3,” 2025, [Online]. Available: https://goo.gle/Gemma3Report

[53] Y. Alnumay et al., “Command R7B Arabic: A Small, Enterprise Focused, Multilingual, and Culturally Aware Arabic LLM,” 2025. [Online]. Available: https://arxiv.org/abs/2503.14603

[54] DeepSeek-AI et al., “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning,” 2025. [Online]. Available: https://arxiv.org/abs/2501.12948

![](images/a7eadf3eaf40d71c2bc7588bf01cf1431a8de6a54de2c608d323ca1344282742.jpg)  
Khloud Al Jallad received the B.S. degree in informatics engineering from Arab International University, Syria, in 2014, and the M.D. degree in Big Data from Higher Institute for Applied Sciences and Technology (HIAST), Syria, in 2019. Currently a PhD candidate at HIAST, Syria. From 2017 till now, she has been working as a researcher and lecturer. Her research interests include NLP, DL, ML.

![](images/bdbf903a95f5f0b76b70793faecb8a66784feecc5e04f4af49dea02ccfb2281a.jpg)

Nada Ghneim received the Information Technology Engineering degree from Higher Institute for Applied Sciences and Technology (HIAST), Damascus, Syria, 1991, and the Postgraduate Degree (DEA) in Artificial Intelligence (Image, Robotics, Vision), from the National High School of Computer Science and Applied Mathematics in Grenoble (ENSIMAG), France, 1993, and a PhD degree in Language Sciences (Speech Communication) from “Institut de la Communication Parlée”-

Stendhal University, Grenoble, France, 1997. Associate Professor at the Faculty of Informatics \& Communication Engineering, at the Arab International University (AIU), Daraa, Syria. She is also a Lecturer, at HIAST, the Syrian Virtual University and the Information Technology Engineering Faculty at Damascus University. Professor Ghneim’s research interests include Artificial Intelligence and Natural Language Processing. She has many publications in Conferences, workshops, journals and books, mainly focusing on Arabic language processing at different modalities (speech, text and image), such as Text-to-Speech, Image Captioning, Sentiment Analysis, Inference detection, …

![](images/fc4b90ec9d0134fe098ce29c2c43d55c46e6d4abdcd5d380c24c4393a9c9a928.jpg)

Ghaida Rebdawi. received the DEA and Ph.D. degrees in applied automatic and informatics from the Institut National des Sciences Appliquées de Lyon, France, in 1987 and 1990, respectively. In 1991, she joined the Informatics Department, at the Higher Institute for Applied Sciences and Technology, as a Research Assistant Professor. She is actually a Full Professor and a Research Director, at the forementioned institute. She coauthored

many academic books in computer science, and more than 20 articles. Her research interests include software engineering, requirements engineering, and natural language processing.

![](images/98d26621ceaa4cc85558db91ca7cbd02046627c4e3797604f1e12cb1d451677d.jpg)  
Figure2: E-CONAN Datasets Construction Pipeline

![](images/e4beea7a5333c4fa5b376bdf41f1bfc1bed3770f51f14b007899cb078ccb86d2.jpg)  
Figure6: Proposed Evaluation of Cross-lingual Baseline Models

![](images/bcc5bda75e21113e4627e4a6ce93df0cb1c56cce7be73166182afe7424f7d713.jpg)  
Figure7: Proposed Evaluation of Arabic Baseline Model

![](images/1dee7ca46040dd26ad559786e78fcddaa0a0e4befd4a32eae66acc34ac0423bf.jpg)

Figure8: Proposed Evaluation of LLMs  
![](images/eee9bec2dd066ffee3face8ede8dc61cbb3022ef52527b7022bcbef33537a792.jpg)  
Z E R O - S H O T C L A S S I F I C A T I O N R E S U L T S O N E - C O N A N 2 - W A Y D A T A S E T

![](images/50847b7e2ed57c610bd1ca24903ebacd012b81ca4f357aec891489a46ee3b408.jpg)

![](images/7afc2f63fdb0a154c8f7e59485d76072116ad6eb6761dd15e08e6a73063ad13f.jpg)  
Figure9: Zero-Shot Classification Results on ArEntail, E-CONAN-2 Datasets

FacebookAI/roberta-large-mnli   
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7   
MoritzLaurer/mDeBERTa-v3-base-mnli-xnli   
MoritzLaurer/DeBERTa-v3-base-mnli   
MoritzLaurer/ernie-m-base-mnli-xnli facebook/bart-large-mnli   
MoritzLaurer/MiniLM-L6-mnli   
MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli MoritzLaurer/multilingual-MiniLMv2-L12-mnli-xnli

Z E R O - S H O T C L A S S I F I C A T I O N R E S U L T S O N E-C O N A N -3 D A T A S ET S  
![](images/5deb62e857a36323cb06ecc390b99c7eb0a3262fc11cdb26f03be9ee8a71f71d.jpg)

![](images/d5e6d3095d56395df849bffef8c2eebb060428e1034c88b4dac1554a173d4e08.jpg)

![](images/18e7944ed3f5347333e7846156381493f44ac42ffa30508752af61a54fbe3f0f.jpg)

![](images/ba39ac9af9ee65b392831b162828c22cdbe42e849c815d7d6c0e8f86b08c37d0.jpg)  
ZERO-SHOT CLASSIFICATION R ESULTS

![](images/c79782f00b2ccc080ab32aa03b6162b337900921af318476e851a7c5b6cc021b.jpg)

![](images/c52352cb85971392971e2092ca0e557800a2e4ae7f7f4391d3aaf5f01f8b417e.jpg)

![](images/6ba1830df40c62402a2a886af3ee11f95ff63371b3dbdf4ead38b0f403324cce.jpg)

![](images/72fd8bf50c5973a82b2eff857d09f472a30ba2344b813716722202f0eac5cc14.jpg)  
Z E R O - S H O T C L A S S I F I C A T I O N R E S U L T S O N A R N L I D A T A S E T

![](images/0c248eb76b7e88cb33c4d5fba03f198f47ae95693a4d7194c8eecc20be73dd2e.jpg)

![](images/dc09d7d16054281765a757f6f56dc7b549ea59582b4b772d482ee116139a31a9.jpg)

![](images/19db72339217b588082c853b2c1d1f796f6bf7a64ba3b271e075968de7b1f8f2.jpg)

![](images/46ca5eed8d1440bdc74fe4f518a1e8de3daef02beca0cf04a686e44c279705f4.jpg)  
Z E R O - S H O T C L A S S I F I C A T I O N R E S U L T S ON TEST XNLI DATASET

![](images/c023c593f01aac5681fc893955d4fced75a6ab3d21fa7b04957b25c46bbe52b1.jpg)  
Figure10: Results on E-CONAN-3, ArNLI and XNLI Datasets

![](images/c3ec71f5cec9c4a541ea3568916c653a8f8a6ae779efb8524cdcddbda0b87115.jpg)

![](images/4d3d0f38e0619fa06567f208216a419ed653345c6f4ef376c4c36db0f01d4115.jpg)

![](images/7c304219d3260d4e8fdb01a4e1961a6ec2a401c71823ecb7511d2074571192ed.jpg)

![](images/9404a849f2f86c43ac4de9ce51fca1c6bc12111a6bae65799ef475c0221055d6.jpg)

LLM RESULTS ON E-CONAN -3 AS 2-WAY  
Figure11: LLM Zero-Shot Classification on E-CONAN-3 as 2-way  
LLM RESULTS ON ANS -STANCE  
![](images/311f03494ee7502c46ce0ff6f7721955df8aa517e5658ea609f398d7ecd10cf4.jpg)  
Figure12: LLM Zero-Shot Classification on Ans-Stance Dataset

![](images/2ec872101ab244c7cc3d6367eed9a29921ec218b8b9136d6aa694f3b1f2cbcca.jpg)

![](images/469751bb20f84e130891907afa6547af455eb78cf9c01f7e9736a05be46d0357.jpg)

![](images/e132c29404ca86b4fbf3210db53f0c5143496b555c1324b5a798ba1f9606e088.jpg)

LLM RESULTS ON E -CONAN -3  
![](images/1258466c518dbfa612bfa32e78fdedb9da23104c99d6af00319ce8606da9fe2e.jpg)

LLM RESULTS ON ARNLI  
![](images/f52d58f667ce128c779461fdc9cf153534cf2be25cefd6d1421b7ac122ca425d.jpg)

LLM R ESULTS ON AR SN LI  
![](images/932e0c66d043d6cdff234e00ed89bbca3a1fa1106d052412ebe46303543857b0.jpg)

LLM RESULTS ON ARXNLI  
![](images/60a43722934a865a167f54cf6bf7554b6cf0a872730b92be2fc43ed94a718eec.jpg)  
Figure13: LLM Zero-Shot Classification on E-CONAN-3 and SoTA Benchmarks

TABLE 12  
CLASSIFICATION REPORT OF E-CONAN AND SOTA BENCHMARKS.
<table><tr><td rowspan=2 colspan=1>Dataset Name</td><td rowspan=2 colspan=1>Model</td><td rowspan=1 colspan=3>Precision</td><td rowspan=1 colspan=3>Recall</td><td rowspan=1 colspan=3>F1</td></tr><tr><td rowspan=1 colspan=1>Cconttton</td><td rowspan=1 colspan=1>Euntent</td><td rowspan=1 colspan=1>Neutrral</td><td rowspan=1 colspan=1>Ccontiton</td><td rowspan=1 colspan=1>Enennt</td><td rowspan=1 colspan=1>Neutrral</td><td rowspan=1 colspan=1>Ccontcction</td><td rowspan=1 colspan=1>ntament</td><td rowspan=1 colspan=1>Neutrral</td></tr><tr><td rowspan=3 colspan=1>ArNLI</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>0.56</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.41</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.31</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.38</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.52</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.57</td><td rowspan=1 colspan=1>0.56</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.70</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.77</td></tr><tr><td rowspan=3 colspan=1>ArXNLI</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.77</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.69</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.83</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.47</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.58</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.65</td><td rowspan=1 colspan=1>0.58</td></tr><tr><td rowspan=3 colspan=1>ArSNLI</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.51</td><td rowspan=1 colspan=1>0.43</td><td rowspan=1 colspan=1>0.33</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.42</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.37</td><td rowspan=1 colspan=1>0.37</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.27</td><td rowspan=1 colspan=1>0.34</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.4</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.48</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.32</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.05</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.09</td><td rowspan=1 colspan=1>0.48</td></tr><tr><td rowspan=3 colspan=1>AnsStance</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.14</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.24</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.08</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.85</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.15</td></tr><tr><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.90</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.07</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.14</td></tr><tr><td rowspan=2 colspan=1>E-CONAN-3</td><td rowspan=1 colspan=1>MarBERT-XNLI-Tuned</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.55</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.66</td><td rowspan=1 colspan=1>0.45</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.50</td></tr><tr><td rowspan=1 colspan=1>mDeBERTa-v3-base-xnli-multilingual-nli-2mil7</td><td rowspan=1 colspan=1>0.75</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.6</td><td rowspan=1 colspan=1>0.78</td><td rowspan=1 colspan=1>0.67</td><td rowspan=1 colspan=1>0.68</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.73</td><td rowspan=1 colspan=1>0.63</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Gemma3</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.50</td><td rowspan=1 colspan=1>0.54</td><td rowspan=1 colspan=1>0.53</td><td rowspan=1 colspan=1>0.81</td><td rowspan=1 colspan=1>0.63</td><td rowspan=1 colspan=1>0.64</td><td rowspan=1 colspan=1>0.62</td></tr></table>

![](images/072275eee3f160887bb778d9987f5066b0d56e408b73169a6b82e5f5359167d2.jpg)

![](images/dcaeba0caa6f6080aeccdb00875f60d4f9c4056db39282e8897d3f6ca9ad3be7.jpg)  
Figure15: Confusion Matrix of E-CONAN and SoTA Benchmarks

![](images/d1c4744e34aaac528b8c651bdd78c1e09b9657e8bb1405e4fae43e5059d8e03a.jpg)

![](images/ee22055388ff1a43226ce652059ed88fe2d02354413b9602d85f67622002b61a.jpg)

![](images/3e713cb76204cd2b1f58f964639c067657be2fc26f10e1470f2321ce99f7eac4.jpg)