# TDR: Tagalog Diacritic Restoration Dataset

Ivan Yuri De Leon<sup>1,2,∗,∗∗</sup>, Lanz Railey Fermin<sup>1,3,∗</sup>, Zachary Aldridge Africa<sup>1</sup>, Richell Isaiah Flores<sup>1</sup>

<sup>1</sup> Department of Mathematics, Ateneo de Manila University, Philippines

<sup>2</sup> Globe Telecom, Philippines

<sup>3</sup> Center for AI Research, Philippines

ivan.deleon@student.ateneo.edu, ecair.lfermin@deped.gov.ph, zachary.africa@student.ateneo.edu, riflores@ateneo.edu

## Abstract

Homographs are words that share identical spelling but differ in meaning and pronunciation. In Tagalog, these distinctions are disambiguated in writing through tuldik (diacritical marks), which are rarely preserved in contemporary Tagalog text. This scarcity of diacritics limits diacritic restoration, which is useful for downstream tasks such as prosody selection for textto-speech synthesis. To address this gap, this study introduces TDR, an annotated corpus of 40 frequent Tagalog homographs constructed from news text. It also introduces a collocationdriven retrieval method to expand the dataset and mitigate severe class imbalance among minor variants. Finally, we show that TDR supplies a capability absent from pretrained models, and that this capability can be learned by small, inexpensive models as well as by fine-tuned LLMs. TDR is publicly available on Hugging Face.

Index Terms: word sense disambiguation, diacritics, Filipino, dataset

## 1. Introduction

Homographs are words that share the same written form but differ in meaning and, in some cases, pronunciation [1]. Their forms are the same when written without distinguishing marks, so the intended reading must be inferred from context [2]. Hence, homographs are a core challenge in word sense disambiguation (WSD) [3].

In Tagalog, tuldik (diacritics) resolve this ambiguity in writing by marking syllable stress and the presence of a final glottal stop [4]. These cues often correspond to differences in meaning: for example, the undiacritized word puno may refer to puno\` (tree) or punoˆ (full), depending on context. Diacritics, thus, link a word’s written form to its intended pronunciation and meaning [5]. However, contemporary Tagalog writing rarely preserves them, especially in digital communication [6]. Distinct words then collapse into identical spellings, increasing ambiguity for human readers and computational models [7, 8].

The omission of tuldik has implications for downstream language technologies, such as machine translation and text-tospeech (TTS) systems [9, 10]. In TTS, an undiacritized homograph may receive a plausible but unintended reading, potentially resulting in incorrect stress or glottalization. This, consequently, leads to unnatural or unintelligible synthesized speech [11]. Work on Filipino grapheme-to-phoneme (G2P) conversion likewise highlights the role of sentence-level context in determining stress [12], suggesting that automatic diacritic restoration could serve as a useful preprocessing step for pronunciation-oriented applications.

Despite this potential, automatic diacritic restoration in Tagalog remains largely unexplored. Tagalog is a low-resource language with comparatively few large-scale annotated corpora, benchmarks, and linguistic resources [13], and existing Tagalog corpora generally omit diacritics [14, 15], limiting their use for tasks that require fine-grained phonological distinctions. The only known dedicated effort is a small, manually annotated homograph corpus evaluated with classical classifiers [16]. This contrasts with languages like Arabic, which has several largescale diacritized corpora and a long history of diacritization research [17]. Without comparable resources, progress on the development of automatic diacritic restorers for Tagalog remain hindered.

To address these limitations, we introduce the Tagalog Diacritic Restoration (TDR) dataset, a manually-annotated corpus of contemporary Tagalog news text covering 40 frequent homographs, with label reliability verified through independent reannotation. We also propose a collocation-driven framework to expand frequency of underrepresented homograph variants. Finally, we show that training on TDR provides diacritic restoration ability that pretrained models lack, for large language models (LLMs) as well as smaller encoder and classical models.

## 2. Related Work

TTS and Homograph Disambiguation. A TTS front-end must map written forms to appropriate pronunciations, and for homographs, the correct choice depends on context. Prior work has addressed this problem with supervised classifiers over lexical context [2], with context-word, collocation, and part-ofspeech features for Thai [10], and contextual embeddings [11]. Hansakunbuntheung and Thatphithakkul [18] argue that wordpronunciation pairs alone cannot evaluate such spellings, which they call heteronyms, because the correct reading is contextdependent. Tagalog homographs distinguished by tuldik are heteronyms in this sense, as they differ in stress and final glottal stop. Philippine-language work has addressed pronunciation through G2P models for Tagalog and other local languages [19] and, more recently, stress-aware sentence-level Filipino G2P [12]. To our knowledge, these works do not provide gold diacritic labels for Tagalog homographs in context. Restoring diacritics from context offers a text-level path to this disambiguation, and TDR provides the labeled data for it.

Diacritic Restoration. Diacritic restoration has been studied across languages using lexical and rule-based methods [20, 21], machine learning classifiers like Naive Bayes and

SVMs [22, 23], and more recently, neural sequence models including BiLSTMs and transformers [24, 25, 26]. For Tagalog, to our knowledge, [16] is the only prior study specifically addressing diacritic restoration. This unpublished capstone project framed the problem as a classification task, manually annotated sentences from the Tatoeba corpus covering 21 homographs, and evaluated traditional classifiers such as random forest and logistic regression. It also measured the asymmetry ratio adopted in Section 3.3, but did not address the imbalance among homograph variants. TDR builds on this work by covering 40 homographs, including the 21 from [16], alongside frequent candidates identified from a news corpus. It also provides inter-annotator agreement, uses collocation-driven retrieval to increase coverage of minor variants, and evaluates both LLMs and lightweight classifiers.

Association Measures. Association measures (AMs) quantify the tendency of words to co-occur more often than expected by chance, where a collocate (co-occurring word) associates with a node (target word) [27]. Different AMs capture distinct aspects of association, and their usage depends on the analytical objective [28]. Common measures include LogDice, a logarithmic coefficient which captures association strength; Mutual Information-2 (MI2), which emphasizes distinctiveness while reducing low-frequency pair bias; Delta P2 (∆P ), directional predictiveness of the node given a collocate; and t-score, which shows reliability of an association [29]. No single AM, however, is universally optimal since their behavior can vary with factors like syntactic structure [30, 31]. Collocations can provide information for WSD-related tasks, as different senses tend to occur with certain associations, consistent with the “one sense per collocation” principle [32, 33]. This is useful for retrieval: given that a homograph’s variants share one spelling in undiacritized text, collocates of a minor variant can serve as search leads for articles likely to contain it (Section 4).

## 3. Preliminary Corpus Construction

Overview. We first construct a preliminary dataset, the Pre-TDR, consisting of diacritized homographs drawn from Tagalog news text. This is based on the existing BalitaNLP corpus [34], and we restrict our focus to homographs that are both frequently occurring and genuinely confusing to native speakers.

## 3.1. Homograph Selection

Homographs covered in this study were selected using frequency, native-speaker salience, and prior work. Frequency ensures that targets occur often enough in natural text for disambiguation to matter in practice. Salience, assessed through native-speaker judgment, prioritizes homographs that readers can readily recognize as ambiguous, since a technically homographic word that is rarely confused is less useful to include as a disambiguation target.

We first extracted the 600 most frequent word forms from BalitaNLP. Then, for each form, we checked the UP Diksiyonaryong Filipino (UPDF) and treated it as a homograph candidate if its undiacritized spelling had multiple dictionary entries differing in diacritics. This yielded 100 candidates after applying the following exclusion principles: proper nouns (e.g., Hindi, ‘an Indian language’), infrequently used base verb forms (e.g., ako, ‘to take responsibility’), English-derived cognates that coincidentally share the same spelling (e.g., dating, ‘date’+‘-ing’), and pairs distinguished by botanical senses (UPDF entries labeled Bot.), as they rarely occur in general text.

Native Tagalog speakers among the authors then judged the remaining candidates for salience, retaining only words that are both in common use and likely to be confused by readers. Affixed forms that share a spelling with a target word, but differ in diacritics (e.g., kabilangˆ , ‘opposite’ vs. kab´ılang, ‘including’) and inflected forms (e.g., hul´ı and hul´ıng, both ‘last’) were treated as separate homographs. This yielded 30 homographs.

Finally, the 30-homograph list was compared with the inventory compiled by [16], which was based on native-speaker usage and interest. Eleven out of 21 homographs are already in the list, suggesting that the frequency-based selection captures a good number of homographs salient to Tagalog readers. To improve coverage and build on prior work, we added the remaining 10, yielding the final set of 40 homographs (Table 1).

## 3.2. Pre-TDR Construction

The initial Pre-TDR corpus was built by manually diacritizing news articles sourced from BalitaNLP, as sufficiently largescale diacritized Tagalog text was not available. A preliminary survey of web sources, including published journal articles [35], essays [36], and social media posts [37], found inconsistent use of tuldik across authors. Some texts contained partial diacritization, while others applied it more extensively, making these sources unsuitable for aggregation as reliable gold-standard annotations.

BalitaNLP [34] contains 351,755 Tagalog news articles from eight outlets, covering domains such as crime, entertainment, and sports. News text was selected over social mediaderived sources because the latter tend to exhibit non-standard spelling, code-switching, and other forms of linguistic variation that compromises consistency. The breadth and standardized conventions of news text, therefore, provide an apt source for constructing a contemporary Tagalog diacritization corpus.

For each of the 40 homographs, 220 distinct articles were sampled, with each article containing at least one instance of the target homograph and no overlap between samples for different targets. Each sample was stored in a separate folder, resulting in 8,800 total articles. Since sample articles may naturally contain other target homographs, all occurrences of any of the 40 homographs were still annotated.

A purpose-built annotation script locates each homograph instance in a given article using case-insensitive matching, displays it with a context window to ±60 characters, and asks the annotator to select the appropriate homograph variant or skip the instance. The selected variant is then written into the text and logged for traceability.

The 40 target-homograph folders were divided approximately equally into three sets, each assigned to two of the three annotators. Within each set, one author served as the primary annotator and the other as reviewer, with the reviewer checking every annotated occurrence and correcting when necessary. Roles were rotated across sets so that each annotator served both roles. All annotators are native Tagalog speakers, and the UPDF was consulted alongside native-speaker judgment for ambiguous cases. For most homographs, sentence context determines the intended variant, so the reviewer’s decision was deemed final. The exception is kilala´ / kilala´ , discussed in Section 5.3.2, which was resolved with a fixed decision rule. The initial workflow had no independent double annotation; so, inter-annotator agreement was assessed separately (Section 5.2). This process created the Pre-TDR corpus, comprising 8,800 articles, 100,087 sentences, and 38,096 diacritized homograph instances.

## 3.3. Homograph Asymmetry

Following [16], we quantify the imbalance of a homograph’s diacritized variants using an asymmetry ratio. Let $C _ { 1 }$ be the count of its most frequent variant (hereafter, major variant) and $C _ { 2 }$ the count of its second most frequent (minor variant). The ratio r is given by:

$$
r = { \frac { C _ { 2 } } { C _ { 1 } + C _ { 2 } } } ,\tag{1}
$$

which ranges from 0 (no minor variants) to 0.5 (equally frequent variants). Third variants are excluded from the computation because of their very low counts do not materially affect the comparison between the predominant variants. Only ika has a third variant with instances (ikaˆ, 2).

Homographs are grouped into three categories, consistent with [16]: extreme $( r < 0 . 1 )$ , moderate $( 0 . 1 \leq r < 0 . 2 5 )$ , and workable $( 0 . 2 5 \leq r \leq 0 . 5 )$ . These cutoffs are heuristic and are adopted for comparability with prior work. Accordingly, the categories are used to describe variant imbalance and to prioritize corpus expansion, rather than as empirical thresholds for model learning.

Severe imbalances can make the minor variant more difficult to learn because relatively few training examples expose a model to the contexts in which that variant occurs. Consequently, a model trained on imbalanced data may be biased towards the major variant. Furthermore, [16] notes that, for text data, this imbalance cannot be address via oversampling, since duplicating existing sentences does not give new, distinct examples of the minor variant. This motivates the collection of additional authentic instances of underrepresented variants.

In the Pre-TDR corpus, 10 homographs fall under workable asymmetry (galit, taong, puno, pagod, bukas, takot, kilala, tapos, kita, galing), 6 under moderate asymmetry (huli, buhay, ika, baka, dating, kabilang), and the remaining 24 under extreme asymmetry. This skew motivated the targeted expansion described in Section 4, which retrieves additional articles for moderate and extreme homographs, particularly by using collocates associated with the minor variant as leads.

## 4. Collocation-driven Expansion

Overview. Building on the severe class imbalance seen in Pre-TDR, we expand the corpus by retrieving additional BalitaNLP articles likely to contain the minor variants of moderately and extremely asymmetric homographs. Retrieval is primarily guided by collocates of the minor variant, with fallback strategies for variants with none. The added articles increased the proportion of minor variants enough for six homographs to move up by at least one category.

## 4.1. Collocate Identification

Using the association measures (AMs) described in Section 2, we identified collocates of each minor variant in Pre-TDR. Each collocate was scored with the four AMs in Table 2, computed from contingency tables of co-occurrence frequencies [27]. No single AM is optimal, as argued by [30]. so a collocate is retained only if it met all four cutoffs. These are commonly-used values in literature, but are still arbitrary since there is no current consensus on which AM values would yield the optimal collocates [38]. Hence, requiring all four gives a conservative collocate list that mitigates the weakness of any single AM.

Before extraction, function words (ligatures, prepositions, articles, determiners, conjunctions, and enclitics, as defined in [43]) were removed, so that candidates would be content words.

Table 1: The 40 target homographs in the Pre-TDR and TDR corpus, grouped by asymmetry type. Asterisks indicate additions from [16]. Cell shading denotes improvements relative to Pre-TDR (Yellow: extreme → moderate; Green: moderate → workable; Blue: extreme → workable).
<table><tr><td rowspan=1 colspan=1>Workable</td><td rowspan=1 colspan=1>Moderate</td><td rowspan=1 colspan=1>Extreme</td></tr><tr><td rowspan=1 colspan=1>bukas*</td><td rowspan=1 colspan=1>abala*</td><td rowspan=1 colspan=1>bagong    paano</td></tr><tr><td rowspan=1 colspan=1>galing</td><td rowspan=1 colspan=1>baka</td><td rowspan=1 colspan=1>bilang    pala</td></tr><tr><td rowspan=1 colspan=1>galit*</td><td rowspan=1 colspan=1>buhay</td><td rowspan=1 colspan=1>gabi      pamahalaan</td></tr><tr><td rowspan=1 colspan=1>gamit*</td><td rowspan=1 colspan=1>butas*</td><td rowspan=1 colspan=1>kaibigan  panalo</td></tr><tr><td rowspan=1 colspan=1>huli*</td><td rowspan=1 colspan=1>dating</td><td rowspan=1 colspan=1>kasama   pangulong</td></tr><tr><td rowspan=1 colspan=1>ika</td><td rowspan=1 colspan=1>huling</td><td rowspan=1 colspan=1>kaya      patuloy</td></tr><tr><td rowspan=1 colspan=1>kilala*</td><td rowspan=1 colspan=1>kabilang</td><td rowspan=1 colspan=1>lalaki     sakit</td></tr><tr><td rowspan=3 colspan=1>kitapagod*puno*takot*taongtapos</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>lamang    sasakyan</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=1>madaling  silangnasa      tayo</td></tr><tr><td rowspan=1 colspan=1></td></tr></table>

Table 2: Association measures used for collocate identification, with retention cutoffs. A candidate must meet allfour.
<table><tr><td>AM</td><td>Description</td><td>Value</td></tr><tr><td>LogDice [39]</td><td>Log-scaled Dice coefficient, measures strength of associa- tion regardless of corpus size.</td><td>≥ 7.0</td></tr><tr><td>MI-2 [40]</td><td>Bias-corrected mutual informa- tion, distinguishes exclusivity of homographs.</td><td>≥ 6.0</td></tr><tr><td>∆P2 [41]</td><td>Metric of collocation direction- ality, shows how strongly a col- locate predicts the node.</td><td>≥ 0.005</td></tr><tr><td>t-score [42]</td><td>Co-occurrence measure with adjusted variance, gives relia- bility that a collocate linkage is not due to chance.</td><td>≥ 2.0</td></tr></table>

Target homographs included in the list, like kayaˆ, were retained. Collocates were then extracted using GraphColl in #LancsBox X v5.0.3 [44] to compute for the AMs directly, while AntConc v4.2.4 [45] served as a secondary tool for concordance.

Extraction was run under two window settings, following the Collocation Parameter Notation of [31]: a rigid window (L1-R1) for fixed phrasal constructions and a loose window (L5-R5) for broader thematic associations. Both are commonlyused windows and were not tuned for Tagalog. Because function words were removed beforehand, window positions count the remaining words rather than the original text.

Table 3 lists some phrases from the rigid search that served as leads, as most variants yielded only one to three candidate collocates.

## 4.2. Targeted Article Retrieval

Figure 1 describes the current work’s procedure for retrieval, applied to each moderately or extremely asymmetric homograph depending how its minor variant appears in Pre-TDR.

If the minor variant occurs more than once in Pre-TDR, rigid and loose collocation searches were done. Rigid collocates gave fixed phrases, while loose collocates were grouped together by the annotators into “themes” and used as keyword queries even when no fixed construction exists. For instance, the loose collocates of abala ´ pointed to transportation disruptions and traffic.

Table 3: Representative collocates extracted for minor-variant homographs, used as search leadsfor corpus expansion.
<table><tr><td>Homograph</td><td>Collocate</td><td>Completed Phrase</td></tr><tr><td>abála</td><td>malaking</td><td>malaking abála</td></tr><tr><td>abála</td><td>mabawasan</td><td>mabawasan [ang] abála</td></tr><tr><td>báka</td><td>kalabaw</td><td>kalabaw [at] báka</td></tr><tr><td>buháy</td><td>manatiling</td><td>manatiling buháy</td></tr><tr><td>butás</td><td>bulsa</td><td>butás [ang] bulsa</td></tr><tr><td>datíng</td><td>lakas</td><td>lakas [ng] datíng</td></tr><tr><td>kabilâng</td><td>buhay</td><td>kabilâng buhay</td></tr><tr><td>kilála</td><td>ninyo</td><td>kilála ninyo</td></tr><tr><td>lalakí</td><td>deficit</td><td>lalakí [ang] deficit</td></tr><tr><td>sílang</td><td>sanggol</td><td>sílang [na] sanggol</td></tr></table>

If the minor variant never occurs, or no collocate met the thresholds defined in Table 2, two fallbacks were done: checking for alternate spellings of the homograph (e.g., pangkulong for pangulong´ , and brainstorming candidate phrases and themes from the variant’s meaning in UPDF (e.g., sinigang-related articles for gabi´ , an ingredient). Regardless of which decision path produced them, the resulting phrases and themes were queried into AntConc’s Concordance tool over the remaining unannotated articles of BalitaNLP.

![](images/ff61d4ddf8b2f3bf3d4ebc0ab956574c35dbbb7d01e246e45ae93d78cb3b00e2.jpg)  
Figure 1: The decision-making process in the targeted article retrieval. Note that the brainstorming based on dictionary meaning relied on native speaker’s intuition.

## 5. The TDR Corpus

Overview. This section describes the corpus produced by the target expansion described previously, its composition by homograph, and the reliability of its labels.

## 5.1. Corpus Expansion

A total of 1,265 articles were retrieved for moderately and extremely asymmetric homographs, following the process in Figure 1. They were added to the corresponding homograph folders and diacritized by the same annotator pair and procedure as Pre-TDR (see Section 3).

This expansion produced the final TDR corpus, whose growth is summarized in Table 4. It contains 48,459 diacritized homographs, a 27.2% increase from Pre-TDR. Notably, instances of non-major variants increased from 2,658 to 4,792 (80.29%) and their share of all instances from 6.98% to 9.89%, suggesting that the expansion procedure worked as intended. Six homographs improved by at least one asymmetry category, with none regressing, as marked in Table 1. However, 101 homograph matches were not diacritized for reasons discussed in Section 5.3.

Table 4: Corpus before and after collocation-driven expansion.
<table><tr><td>Corpus</td><td>Articles</td><td>Sentences</td><td>Homograph Counts</td></tr><tr><td>Pre-TDR</td><td>8,800</td><td>100,087</td><td>38,096</td></tr><tr><td>TDR</td><td>10,065</td><td>146,310</td><td>48,459</td></tr></table>

## 5.2. Homograph Breakdown

Table 5 summarizes the major and minor variant counts, asymmetry ratios, and inter-annotator agreement (IAA) for all 40 homographs in the TDR corpus. After corpus expansion, 13 homographs fall under workable asymmetry, 7 under moderate, and the remaining 20 under extreme.

Most of the extremely asymmetric homographs are disproportionately high-frequency, with 11 exceeding 1,000 instances, while some are function word-like, such as kayaˆ (a connective) and lamang ´ (an adverb of degree). Despite increasing the minor variant instances, the collocation-driven procedure is possibly limited for high-frequency function words, which occur across diverse contexts rather than with distinctive collocates. Meanwhile, minor variants of eight homographs (bagong, paano, sakit, panalo, kaibigan, madaling, pamahalaan, and patuloy) remained absent after expansion.

As mentioned in Section 3.2, the original annotator/reviewer setup did not have independent annotation; so, an additional inter-annotator agreement (IAA) study was done to assess label reliability. A sample of 26 articles (roughly 12% of the original 220-article pool) were reannotated as a balance between annotation effort and reliability, rather than using a formal power analysis. We sampled up to 13 articles from each variant; if the second variant had fewer than 13 instances, the remaining quota was filled by exhausting the most frequent variant until 26 articles were obtained. Homographs with fewer total instances were sampled first to avoid depleting rarer variants.

The sampled articles were independently marked by the third annotator, who had not participated in the original main annotator/reviewer pair for that homograph, and had no access to the original labels. Disagreements were resolved by consensus with reference to UPDF.

We report free-marginal κ [46], an agreement coefficient for categorical labels that does not require raters to assign a fixed number of cases to each category. Chance agreement is assumed uniform across the k categories, rather than from the raters’ label frequencies:

$$
\kappa = \frac { P _ { o } - P _ { e } } { 1 - P _ { e } } ; \quad P _ { e } = \frac { 1 } { k } ,
$$

where $P _ { o }$ is the observed agreement and k is the larger of the number of variants listed for the homograph and the number observed in the sample. This ensures that $k \geq 2$ even when only a single variant appears.

Agreement in the reannotated sample was high overall, with a pooled κ of 0.977 over 8,072 annotated instances. Note that, however, this figure is dominated by highly imbalanced homographs, where agreement is easy to reach when one variant dominates the sample. Mean κ is slightly lower for workable (0.938) and moderate (0.95) homographs than for extreme (0.99). The lowest values are pagod (0.848) and kabilang (0.859). Both have substantial minor-variant representation, but the source of their lower agreement requires further examination. Nonetheless, both values exceed 0.80, which is conventionally interpreted as strong agreement [47]. We therefore consider the labels sufficiently reliable for use, while noting that these two homographs are likely to contain residual annotation errors.

## 5.3. Annotation Decisions

Beyond choosing a variant, two decisions needed explicit rules: when to leave a matched string undiacritized, and how to label kilala´ / kilala´ , where context alone did not settle the variant.

## 5.3.1. Undiacritized Instances

A total of 101 homograph-pattern matches (0.21% of all matches in the released TDR corpus) were left undiacritized, for the following reasons.

• Misspellings. A typo may match a target homograph. For example, “Masustansiya ang sinigang dahil marami itong sangkap na gulay kaya ng kangkong, sitaw, okra, labanos, at talong” (“Sinigang is nutritious because it has many vegetable ingredients like kangkong, string beans, okra, radish, and eggplant”) uses kaya where gaya (“like”) was intended.

• Subword fragments. A match may be part of a larger word, as in “Kailangang aregluhin pa ito ng magka-bilang gobyerno” (“This still needs to be settled by both governments”), where bilang is part of magka-bilang (“both”), or of a URL slug, e.g., .../pinoy-nagdala-ng-bagong-covid-variant-sa-.../.

• Contracted forms. A match may be part of a contraction. For example, “Kung noon kapag ika’y nababagot at sa palagay mo ay kailangan mo...” (“Back then, when you got bored and felt that you needed...”) contains ika only as part of ika’y (ikaw + ay).

• Proper nouns with unknown pronunciation. For example, “...isang torneong pinamayagpagan ni Japanese Nasa Hataoka...” (“...a tournament dominated by Japan’s Nasa Hataoka...”) uses Nasa as a personal name. Proper nouns with well-known pronunciations, such as Silang, Cavite´ , were diacritized, since explicit diacritics benefit TTS when the pronunciation is known.

• Code-switched foreign words. For example, “Ang hindi ko ide-deny, I’m dating now at hindi siya ang rason...” (“What I won’t deny is that I’m dating now, and they aren’t the reason...”) uses the English gerund dating, not the Tagalog homograph dating´ / dat´ıng.

## 5.3.2. Disambiguating kilala and kil ´ ala´

Unlike most homographs, context alone could not reliably separate these two, and consultation with two Filipino linguistics professors gave conflicting recommendations. To ensure consistent annotation, we adapted a distinction based on the UPDF: kilala´ is a synonym of bantog (famous, well-known), while kilala´ denotes personal acquaintanceship (cf. kakilala´ ). The operational rule, then, bases on whether an explicit knower, the entity doing the knowing or has knowledge of a known party, appears in the sentence. If so, the instance is kilala´ ; otherwise, it is kilala´.

Kung napansin mo, parang nahihiya siya sa akin kasi nga, kilala´ ko siya noon pa. (If you noticed, they seem embarrassed around me because I’ve known them for a while now.)

Here, the explicit mention of a knower (ko, “I”) identifies the appropriate annotation as kilala´ . In contrast, kilala´ describes the known party as being known or famous without introducing a specific knower.

Si De Lima ay kilala bilang isa sa mga kritiko ni´ Duterte. (De Lima is known as one of Duterte’s critics.)

Only the known party (De Lima) is expressed, with no explicit knower, yielding kilala´. A more challenging case arises when an explicit knower is present while the known party is also independently famous, seemingly satisfying both definitions:

Kilala´ natin si Ai-ai delas Alas bilang Comedy Queen, ang BFF ni Sharon Cuneta. (We know Ai-ai delas Alas as the Comedy Queen, Sharon Cuneta’s best friend.)

Ai-ai delas Alas is widely known which, in isolation, would suggest kilala´. However, the annotation rule is structured to give precedence to the presence of an explicit knower; in this sentence, the knower is natin (“we”). So, the instance is annotated as kilala´ .

This rule was applied consistently throughout the corpus to ensure annotation consistency. Nevertheless, the disagreement observed among both annotators and consulted experts suggests that the distinction between kilala´ and kilala´ is not categorically resolved in contemporary Tagalog usage. Hence, this rule should be interpreted only as an operational annotation guideline, not a definitive linguistic analysis.

## 6. Experiments

We evaluate the utility of TDR on two tasks that differ in how much of the diacritic restoration problem a model must solve. Multiple-choice disambiguation (MCD) gives the candidate set and asks only which variant a homograph takes in context, isolating disambiguation from text generation and places classifiers, encoders, and LLMs on equal footing. Sentence reconstruction, meanwhile, asks a model to restore diacritics in an entire sentence – which is closer to how restoration could serve a TTS front-end and is the natural output for LLMs. We additionally test whether reconstruction output improves a downstream TTS system through a small-scale test.

## 6.1. Data Split

TDR was partitioned, at the article level, into two disjoint sets: a test pool of 1,040 articles, corresponding to the reannotated dataset used for IAA, and a training pool of the remaining 9,025 articles. Articles were then segmented into sentences. For the test benchmark, only sentences containing at least one diacritized target homograph were retained. Sentences where the same homograph appears more than once with different variants were excluded, since these cannot be assigned a single intended variant. For instance, “Kayaˆ nga nagtitiyaga sila ... ito na lang ang kaya´ nilang bilhin.” (“That’s precisely why they persevere ... because it’s all they can afford to buy.”) contains two different variants of kaya, and is therefore excluded.

Table 5: Variant counts, asymmetry ratio, and inter-annotator agreement (free-marginal κ) for all homographs in the TDR corpus. Rows in red have the lowest agreement value.
<table><tr><td>Homograph</td><td>Variant 1</td><td>Variant 2</td><td>Variant 3</td><td>Ratio</td><td>κ</td></tr><tr><td>galit</td><td>galít: 197</td><td>gálit: 191</td><td></td><td>0.4923</td><td>0.939</td></tr><tr><td>kilala</td><td>kilalá: 254</td><td>kilála: 238</td><td></td><td>0.4837</td><td>0.956</td></tr><tr><td>puno</td><td>punô: 170</td><td>punò: 158</td><td></td><td>0.4817</td><td>0.983</td></tr><tr><td>taong</td><td>táong: 666</td><td>taóng: 608</td><td></td><td>0.4772</td><td>0.954</td></tr><tr><td>galing</td><td>gáling: 368</td><td>galíng: 280</td><td></td><td>0.4321</td><td>0.975</td></tr><tr><td>huli</td><td>hulí: 333</td><td>húli: 232</td><td></td><td>0.4106</td><td>0.959</td></tr><tr><td>pagod</td><td>pagód: 205</td><td>págod: 125</td><td></td><td>0.3788</td><td>0.848</td></tr><tr><td>bukas</td><td>bukás: 231</td><td>búkas: 133</td><td></td><td>0.3654</td><td>0.947</td></tr><tr><td>gamit</td><td>gámit: 441</td><td>gamít: 255</td><td></td><td>0.3664</td><td>0.933</td></tr><tr><td>takot</td><td>tákot: 235</td><td>takót: 119</td><td></td><td>0.3362</td><td>0.873</td></tr><tr><td>tapos</td><td>tápos: 516</td><td>tapós: 224</td><td></td><td>0.3027</td><td>0.959</td></tr><tr><td>ika</td><td>iká: 491</td><td>ikà: 178</td><td>ikâ: 2</td><td>0.2661</td><td>0.920</td></tr><tr><td>kita</td><td>kíta: 404</td><td>kitá: 143</td><td></td><td>0.2614</td><td>0.950</td></tr><tr><td>abala</td><td>abalá: 225</td><td>abála: 71</td><td></td><td>0.2399</td><td>0.970</td></tr><tr><td>baka</td><td>bakâ: 804</td><td>báka: 226</td><td></td><td>0.2194</td><td>0.956</td></tr><tr><td>buhay</td><td>búhay: 1859</td><td>buháy: 411</td><td></td><td>0.1811</td><td>0.950</td></tr><tr><td>butas</td><td>bútas: 237</td><td>butás: 40</td><td></td><td>0.1444</td><td>0.964</td></tr><tr><td>huling</td><td>hulíng: 660</td><td>húling: 107</td><td></td><td>0.1395</td><td>0.973</td></tr><tr><td>dating</td><td>dáting: 1700</td><td>datíng: 221</td><td></td><td>0.1150</td><td>0.978</td></tr><tr><td>kabilang</td><td>kabílang: 979</td><td>kabilâng: 119</td><td></td><td>0.1084</td><td>0.859</td></tr><tr><td>kaya</td><td>kayâ: 5308</td><td>káya: 363</td><td></td><td>0.0640</td><td>0.984</td></tr><tr><td>lamang</td><td>lámang: 2968</td><td>lamáng: 150</td><td></td><td>0.0481</td><td>0.975</td></tr><tr><td>silang</td><td>siláng: 1515</td><td>sílang: 67</td><td></td><td>0.0424</td><td>0.989</td></tr><tr><td>bilang</td><td>bílang: 3569</td><td>biláng: 102</td><td></td><td>0.0278</td><td>1.000</td></tr><tr><td>lalaki</td><td>laláki: 737</td><td>lalakí: 12</td><td></td><td>0.0160</td><td>1.000</td></tr><tr><td>sasakyan</td><td>sasakyán: 696</td><td>sasákyan: 6</td><td></td><td>0.0085</td><td>0.988</td></tr><tr><td>pala</td><td>palá: 833</td><td>pála: 1</td><td></td><td>0.0012</td><td>0.989</td></tr><tr><td>tayo</td><td>táyo: 1702</td><td>tayô: 4</td><td></td><td>0.0023</td><td>1.000</td></tr><tr><td>gabi</td><td>gabí: 1191</td><td>gábi: 2</td><td></td><td>0.0017</td><td>0.989</td></tr><tr><td>pangulong</td><td>pangúlong: 1386</td><td>pangulóng: 1</td><td></td><td>0.0007</td><td>0.992</td></tr><tr><td>pamahalaan</td><td>pámahalaán: 819</td><td>pámahaláan: 0</td><td></td><td>0.0000</td><td>1.000</td></tr><tr><td>patuloy</td><td>patúloy: 1062</td><td>patulóy: 0</td><td>kasamá: 0</td><td>0.0000</td><td>1.000</td></tr><tr><td>kasama</td><td>kasáma: 2261</td><td>kasamâ: 2</td><td></td><td>0.0009</td><td>0.995</td></tr><tr><td>nasa</td><td>nása: 3365</td><td>nasà: 1</td><td></td><td>0.0003</td><td>1.000</td></tr><tr><td>bagong</td><td>bágong: 2039</td><td>bagóng: 0</td><td></td><td>0.0000</td><td>0.992</td></tr><tr><td>paano</td><td>paáno: 769</td><td>paanó: 0</td><td></td><td>0.0000</td><td>1.000</td></tr><tr><td>sakit</td><td>sakít: 606</td><td>sákit: 0</td><td></td><td>0.0000</td><td>1.000</td></tr><tr><td>panalo</td><td>panálo: 614</td><td>panaló: 0</td><td></td><td>0.0000</td><td>1.000</td></tr><tr><td>kaibigan</td><td>kaibígan: 698</td><td>kaibigán: 0</td><td></td><td>0.0000</td><td>1.000</td></tr><tr><td>madaling</td><td>madalîng: 554</td><td>madáling: 0</td><td></td><td>0.0000</td><td>1.000</td></tr></table>

For fine-tuning the neural models, the training pool was split approximately 90/10 at the sentence level, guided by homograph–variant groups. Separate splits yielded 28,742 training and 3,199 validation records for reconstruction, and 34,869 training and 3,882 validation questions for MCD. The test benchmark covers all 40 homographs, with 7,718 MCD items targeting distinct homographs within sentences and 6,113 reconstruction items, one per distinct undiacritized sentence.

## 6.2. Models

We compare non-neural baselines, fine-tuned encoders, and LLMs. No model uses class reweighting, so that each model sees the natural imbalance among homograph variants.

• Baselines. Two baselines were evaluated. First, a majority baseline model predicts each homograph’s most frequent variant. Second, a logistic regression (LR) baseline fits one classifier per homograph on TF-IDF features of the undiacritized sentence, with a maximum of 5,000 features. For each homograph, we select $C \in \{ 0 . 1 , 0 . 3 , 1 , 3 , 1 0 \}$ , n-gram range (unigram or bigram), and TF-IDF weighting (standard, binary, or sublinear) by stratified 5-fold cross-validation on macro-F1. The LR model is trained only on 24 homographs whose minor variant has at least five training sentences. The other 16 fall back to the majority prediction.

• Encoders. We fully fine-tune four encoders: RoBERTabase and RoBERTa-large [48], pretrained mainly on English, and their Tagalog counterparts RoBERTa-Tagalog-base and RoBERTa-Tagalog-large [13]. Each candidate is paired with the undiacritized sentence and scored by a shared head, trained with cross-entropy to rank the gold variant highest (3 epochs, learning rate $2 \times 1 0 ^ { - 5 }$ , effective batch size 16). The checkpoint with the best validation loss is saved.

• LLMs. We use Llama-3.1-8B-Instruct [49] and Llama-SEA-LION-v2-8B [50], the latter adapted for Southeast Asian languages. Each is evaluated zero-shot and fine-tuned, with a separate $\mathrm { Q L o R A }$ adapter per task trained on that task’s prompt format (4-bit NF4; rank 16, α = 32, dropout 0.05 on all attention and MLP projections; 3 epochs, learning rate $2 \times 1 0 ^ { - 4 }$ , effective batch size 16; loss on the answer only; best checkpoint by validation loss), giving four fine-tuned models.

Note that RoBERTa is evaluated on multiple-choice disambiguation only; LR and majority are evaluated on multiplechoice and simulated for reconstruction (see Section 6.4); while LLMs are evaluated on both.

## 6.3. Multiple-Choice Disambiguation

Setup. The model chooses the form a target homograph takes in an undiacritized sentence. Each item corresponds to one homograph instance and offers four candidate forms: all attested homograph variants, its undiacritized spelling, and invalid diacritized forms (when options are fewer than four). Candidates are shuffled once per item. LLMs receive the prompt below and answer with the chosen form. Its outputs are then NFCnormalized and parsed: an output counts as an answer if exactly one candidate appears in it (case-insensitive). Meanwhile, RoBERTa encoders score each candidate paired with the sentence and return the highest-scoring one. Baseline models predict a variant directly, so they always return a valid candidate.

System: You are a Tagalog diacritic restoration   
assistant. From the list of variants given, respond   
with ONLY the correct variant, copied exactly as   
written (including its diacritic marks). Do not in  
clude any explanation, reasoning, punctuation, or   
extra text.   
User: How should the word {homograph} be   
written with the correct diacritic marks in the sen  
tence “{sentence}”?   
Options:   
- {choice<sub>1</sub>}   
- {choice<sub>2</sub>}   
- {choice<sub>3</sub>}   
- {choice<sub>4</sub>}   
Answer:

Since accuracy is dominated by majority variants, we use macro-F1 as the primary metric. For a homograph h with variant set $V _ { h }$ attested in its instances, we first average the pervariant F1 scores within each homograph, then average across homographs:

$$
\mathrm { F } 1 _ { h } = \frac { 1 } { | V _ { h } | } \sum _ { v \in V _ { h } } \mathrm { F } 1 _ { h , v } ,\tag{2}
$$

$$
\mathsf { M a c r o - F l } = \frac { 1 } { | H | } \sum _ { h \in H } \mathsf { F l } _ { h } ,\tag{3}
$$

where H is the set of all homographs. Because both averages are unweighted, a minor variant counts as much as a major one, and every homograph counts equally regardless of its frequency. The latter makes the metric noisy: if a homograph’s minor variant has only one or two test instances, a single error can cut that homograph’s F1 by up to half. We therefore also report macro-F1 over the 20 workably and moderately asymmetric homographs (W+M F1), and weighted F1, in which each homograph’s F1 is weighted by its number of test instances.

Results. Table 6 reports the results. Fine-tuned SEA-LION performs best, closely followed by fine-tuned Llama. Both improve sharply over their zero-shot counterparts, which fall below even the majority baseline. This suggests that Tagalog diacritic disambiguation is not readily available from generalpurpose pretraining, and that TDR can serve as an effective finetuning data for acquiring it.

Table 6: Multiple-choice disambiguation results. W+M F1 considers workably and moderately asymmetric homographs only. Weighted: per-homograph F1 weighted by test instances.
<table><tr><td>Model</td><td>Macro-F1</td><td>W+M F1</td><td>Weighted F1</td></tr><tr><td>Majority</td><td>0.558</td><td>0.396</td><td>0.568</td></tr><tr><td>Logistic regression</td><td>0.800</td><td>0.817</td><td>0.794</td></tr><tr><td>RoBERTa-base</td><td>0.654</td><td>0.601</td><td>0.646</td></tr><tr><td>RoBERTa-large</td><td>0.559</td><td>0.505</td><td>0.558</td></tr><tr><td>RoBERTa-Tagalog-base</td><td>0.786</td><td>0.817</td><td>0.754</td></tr><tr><td>RoBERTa-Tagalog-large</td><td>0.815</td><td>0.844</td><td>0.826</td></tr><tr><td>Llama, zero-shot</td><td>0.276</td><td>0.307</td><td>0.263</td></tr><tr><td>Llama, fine-tuned</td><td>0.831</td><td>0.878</td><td>0.833</td></tr><tr><td>SEA-LION, zero-shot</td><td>0.330</td><td>0.330</td><td>0.301</td></tr><tr><td>SEA-LION, fine-tuned</td><td>0.838</td><td>0.898</td><td>0.849</td></tr></table>

TDR also supports competitive performance from much smaller models. For instance, RoBERTa-Tagalog-large comes within 0.024 macro-F1 of the fine-tuned LLMs, while LR remains competitive. Among the encoders, both Tagalog models outperform their English counterparts, with English RoBERTalarge notably trailing RoBERTa-base, suggesting that languagespecific pretraining matters more than model size in this setting. The gains from fine-tuning are largest on W+M F1, where minor variants are better represented. The fine-tuned LLMs also substantially outperform the majority baseline on this subset (0.878–0.898 versus 0.396).

Error analysis. The fine-tuned LLMs rarely answered in the expected format. Over 96% of their outputs required parsing, and fewer than 0.4% consisted of the correct form alone. Instead of a single form, they typically repeated their chosen form, either across lines or concatenated (e.g., the word bilang´ written several times). Parsing recovers the intended answer in most of these cases, since only one candidate appears. About 3% of outputs, however, contained no candidate or more than one, and were scored as incorrect. The zero-shot models show the opposite pattern: their outputs almost never required parsing, but they often chose the wrong form. Fine-tuning thus improved disambiguation substantially while weakening adherence to the answer format. A finer-grained analysis of errors by diacritic type (stress vs. glottal stop) and by position within the word is left to future work.

## 6.4. Sentence Reconstruction

Setup. Each item is an undiacritized test sentence, which the model must return with diacritics restored. Unlike multiplechoice disambiguation, this task provides no candidate list. The model must identify the target homographs and choose their variants. It therefore more closely resembles restoration for a TTS front-end.

LLMs receive the prompt below. Encoders are not evaluated because they only score given candidates. For the majorityclass baseline and LR, we simulate reconstruction by substituting their multiple-choice predictions for the target homographs in each sentence.

System: Modify the homographs in this Taga  
log sentence by adding the correct diacritic marks   
(tuldik).   
User: {sentence}

We report two sentence-level metrics. Exact match requires the output to equal the gold sentence. Recovery requires only that, for every target homograph, the variants appearing in the output match the gold variants in the same order, so changes elsewhere in the sentence are ignored. Target occurrences in the output are located by diacritic-stripped string matching. For the simulated baselines, the two metrics coincide by construction, since only target homographs are changed.

Table 7: Sentence reconstruction results, shown in percentages. For simulated baselines (<sup>†</sup>), exact match equals recovery.
<table><tr><td>Model</td><td>Exact Match</td><td>Recovery</td></tr><tr><td>Majority† LR^{+</td><td>85.20 93.73</td><td>85.20 93.73</td></tr><tr><td>Llama, zero-shot</td><td>0.00</td><td>0.02</td></tr><tr><td>Llama, fine-tuned</td><td>95.50</td><td>97.04</td></tr><tr><td>SEA-LION, zero-shot</td><td>0.18</td><td>0.74</td></tr><tr><td>SEA-LION, fine-tuned</td><td>95.47</td><td>96.99</td></tr></table>

Results. Table 7 shows the results. Zero-shot LLMs almost never return a correctly restored sentence (at most 0.74% recovery), indicating that its diacritic restoring capability is absent without task-specific training. After fine-tuning on TDR, Llama and SEA-LION are practically tied as they recover the gold variants in about 97% of sentences, and match the gold sentence in about 95.5%. The 1.5-point gap between recovery and exact match rates reflect occasional edits done outside the target homographs which include, but are not limited to, whitespace, punctuation, or changes to other words.

The majority baseline already recovers 85% of sentences, since most contain only major variants, so any further gain must come from the remaining sentences that require a minor variant. On both metrics, the fine-tuned LLMs outperform the zero-shot LLMs as well as the simulated majority and LR baselines. Because these baselines are given the target positions, this comparison is conservative with respect to the LLMs.

## 6.5. Small-scale TTS check

Setup. We test whether restored diacritics improve pronunciation using MOSS-TTS-v1.5, a Tagalog-capable TTS model [51]. Each item is synthesized in three conditions: undiacritized, LLM-restored, and gold (diacritized). Five sentences were sampled for the major and minor variants of the 13 workably asymmetric homographs and synthesized using the same seed and voice. For LLM-restored inputs, the fine-tuned Llama restores the sentence, after which only its prediction for the target homograph is spliced back to preserve matched contexts. Three independent annotators identify the variant pronounced in each clip or mark it unintelligible. The majority vote determines the perceived variant, and homograph pronunciation error rate (HPER) is the proportion of clips whose perceived variant differs from the gold variant.

Results. Table 8 shows that feeding the TTS model text restored by the TDR-finetuned model reduced the HPER from 35.66% to 12.40%, gaining nearly the same improvement obtained with the actual diacritized sentences from TDR. This corresponds to a relative reduction of about 65%, leaving the LLM-restored input nearly identical to the HPER of the gold diacritization. Inter-annotator aggrement was moderate (freemarginal $\kappa = 0 . 5 2 )$ , which may partly reflect annotators occasionally choosing the actual sentence context rather than the audio. This motivates stricter listening protocols in future work.

Table 8: HPER by input text, based on the majority vote of annotators. Clips with invalid LLM output and marked as unintelligible are excluded.
<table><tr><td>Input Text</td><td>N</td><td>HPER</td></tr><tr><td>Undiacritized</td><td>129</td><td>35.66</td></tr><tr><td>LLM-restored</td><td>129</td><td>12.40</td></tr><tr><td>Gold (diacritized)</td><td>130</td><td>11.54</td></tr></table>

Although small in scale, these results suggest that finetuning diacritic restoration models with TDR can be a useful step for improving TTS pronunciation of Tagalog words, especially homographs, whose pronunciation cannot be determined from spelling alone.

## 7. Limitations

TDR covers 40 homographs drawn from a single domain, news text from BalitaNLP, so findings may not transfer to other registers. Independent reannotation covered only the 1,040-article test pool. Collocation-driven expansion did not fully resolve class imbalance: function word-like homographs remain extremely asymmetric, and eight still had no minor variant. Because some minor variants have only one or two test instances, per-homograph F1 on extremely asymmetric homographs is unstable; W+M F1 mitigates but does not remove this. The benchmark also excludes sentences in which one homograph appears with conflicting variants, which a practical restorer must handle.

On the modeling side, we evaluate two 8B LLMs and four encoders, each fine-tuned in a single run with fixed hyperparameters, and report no confidence intervals. Fine-tuned LLMs required output parsing in multiple choice, so their scores depend on the parsing rule. A TTS-relevant error analysis, examining which restoration errors most affect synthesized pronunciation, is recommended for future work. Finally, the TTS check only used a single system and sentences from workable homographs.

## 8. Conclusion

We introduced TDR, a manually annotated corpus of 40 Tagalog homographs with diacritic marks in context, comprising 48,459 diacritized instances from 10,065 news articles. Collocationdriven retrieval increased the number of minor-variant instances by about 80%, and independent reannotation of the test pool showed high agreement (κ = 0.977). The evaluated zeroshot LLMs fell below the majority baseline in multiple-choice disambiguation and almost never restored a sentence correctly. Fine-tuning on TDR yielded the best performance among the evaluated models on both tasks, recovering the gold variants in about 97% of test sentences. Smaller models were also competitive: RoBERTa-Tagalog-large approached the macro-F1 of the fine-tuned LLMs, while per-homograph logistic regression trailed them by only a few points. TDR is publicly available on Hugging Face under the license conditions of BalitaNLP, supporting further work on Tagalog diacritic restoration and pronunciation-aware speech technology.

## 9. Generative AI Use Disclosure

Generative AI tools were used to assist with editing prose for clarity and conciseness, and with drafting portions of code used in data processing and experimentation. All research design, data annotation, experimental results, and analytical conclusions are the authors’ own. AI-assisted content was reviewed and verified by the authors prior to inclusion.

## 10. References

[1] L. R. Gottlob et al., “Reading homographs: Orthographic, phonologic, and semantic dynamics,” Journal ofExperimental Psychology: Human Perception and Performance, vol. 25, no. 2, p. 561, 1999.

[2] K. Gorman, G. Mazovetskiy, and V. Nikolaev, “Improving homograph disambiguation with supervised machine learning,” in Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), 2018.

[3] D. Madina, “Polysemy in cognitive linguistics,” Web of Teachers: Inderscience Research, vol. 2, no. 1, pp. 17–19, 2024.

[4] L. K. Santos, Balarila ng Wikang Pambansa. Manila, Philippines: Komisyon sa Wikang Filipino, 2018.

[5] V. Almario, Ed., Manwal sa Masinop na Pagsulat. Komisyon sa Wikang Filipino, 2014.

[6] J. C. Wells, “Orthographic diacritics and multilingual computing,” Language Problems and Language Planning, vol. 24, no. 3, pp. 249–272, 2000.

[7] M. Maroun and R. Hanley, “Diacritics improve comprehension of the arabic script by providing access to the meanings of heterophonic homographs,” Reading and Writing, vol. 30, pp. 319–335, 2017.

[8] N. Ljubesiˇ c, T. Erjavec, and D. Fi´ ser, “Corpus-based diacriticˇ restoration for south slavic languages,” in Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016), 2016, pp. 3612–3616.

[9] N. Ide and J. Veronis, “Introduction to the special issue on word´ sense disambiguation: The state of the art,” Computational Linguistics, vol. 24, no. 1, pp. 1–40, 1998.

[10] V. Tesprasit, P. Charoenpornsawat, and V. Sornlertlamvanich, “A context-sensitive homograph disambiguation in thai text-tospeech synthesis,” in Companion Volume of the Proceedings of HLT-NAACL 2003—Short Papers, 2003, pp. 103–105.

[11] M. Nicolis and V. Klimkov, “Homograph disambiguation with contextual word embeddings for TTS systems,” 2021.

[12] L. B. Marqueses, P. G. G. Silva, C. Cabatay, E. A. Tan, and A. F. Laguna, “Towards stress-aware sentence-level filipino g2p with weakly-supervised byt5 fine-tuning,” 2026. [Online]. Available: https://arxiv.org/abs/2609.09974

[13] J. C. B. Cruz and C. Cheng, “Improving large-scale lan guage models and resources for filipino,” arXiv preprint arXiv:2111.06053, 2021.

[14] S. N. Dita, R. E. O. Roxas, and P. Inventado, “Building online corpora of philippine languages,” in Proceedings of the 23rd Pacific Asia Conference on Language, Information and Computation. Waseda University, 2009, pp. 646–653.

[15] L. J. V. Miranda, “calamancy: A tagalog natural language processing toolkit,” arXiv preprint arXiv:2311.07171, 2023.

[16] R. I. S. Flores, “Diacritics restoration for homograph disambiguation in Tagalog: Corpus generation and experiments,” Capstone Project, Ateneo de Manila University, Apr. 2024.

[17] M. M. Almanea, “Automatic Methods and Neural Networks in Arabic Texts Diacritization: A Comprehensive Survey,” IEEE Access, vol. 9, pp. 145 012–145 032, 2021.

[18] C. Hansakunbuntheung and S. Thatphithakkul, “ConPro: Heteronym pronunciation corpus with context information for textto-phoneme evaluation in Thai,” in 2017 20th Conference of the Oriental Chapter of the International Coordinating Committee on Speech Databases and Speech I/O Systems and Assessment (O COCOSDA), 2017, pp. 1–6.

[19] A. Aquino, J. L. Tsang, C. R. Lucas, and F. de Leon, “G2p and asr techniques for low-resource phonetic transcription of tagalog, cebuano, and hiligaynon,” in 2019 International Symposium on Multimedia and Communication Technology (ISMAC), 2019, pp. 1–5.

[20] A. A. Neme and S. Paumier, “Restoring Arabic vowels through omission-tolerant dictionary lookup,” Language Resources and Evaluation, vol. 54, no. 2, pp. 487–551, 2020.

[21] C. Krstev, R. Stankovic, and D. Vitas, “Knowledge and rule-based´ diacritic restoration in Serbian,” in Proceedings of the Third International Conference on Computational Linguistics in Bulgaria (CLIB 2018), 2018, pp. 41–51.

[22] K. P. Scannell, “Statistical unicodification of African languages,” Language Resources and Evaluation, vol. 45, no. 3, pp. 375–386, 2011.

[23] T. A. Luu and K. Yamamoto, “A pointwise approach for Vietnamese diacritics restoration,” in 2012 International Conference on Asian Language Processing. IEEE, 2012, pp. 189–192.

[24] L. Stankevicius, M. Luko ˇ sevi ˇ cius, J. Kapo ˇ ciˇ ut¯ e-Dzikien ˙ e,˙ M. Briediene, and T. Krilavi˙ cius, “Correcting Diacritics andˇ Typos with a ByT5 Transformer Model,” Applied Sciences, vol. 12, no. 5, 2022. [Online]. Available: https://www.mdpi.com/ 2076-3417/12/5/2636

[25] T. D. A. Dang and T. T. T. Nguyen, “TDP – A Hybrid Diacritic Restoration with Transformer Decoder,” in Proceedings of PACLIC 2020, Hanoi, Vietnam, 2020.

[26] H. Mubarak, A. Abdelali, H. Sajjad, Y. Samih, and K. Darwish, “Highly effective Arabic diacritization using sequence to sequence modeling,” in Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), J. Burstein, C. Doran, and T. Solorio, Eds. Minneapolis, Minnesota: Association for Computational Linguistics, Jun. 2019, pp. 2390–2395. [Online]. Available: https://aclanthology.org/N19-1248/

[27] V. Brezina, Statistics in corpus linguistics: A practical guide. Cambridge University Press, 2018.

[28] D. Gablasova, V. Brezina, and T. McEnery, “Collocations in corpus-based language learning research: Identifying, comparing, and interpreting the evidence,” Language learning, vol. 67, no. S1, pp. 155–179, 2017.

[29] V. Brezina and T. McEnery, “Introduction to corpus linguistics,” in The Routledge handbook of second language acquisition and corpora. Routledge, 2020, pp. 11–22.

[30] Q. Su, C. Gu, and P. Liu, “Association measures for collocation extraction: Automatic evaluation on a large-scale corpus,” International Journal ofCorpus Linguistics, vol. 29, no. 1, pp. 59–86, 2024.

[31] V. Brezina, T. McEnery, and S. Wattam, “Collocations in context: A new perspective on collocation networks,” International jour nal of corpus linguistics, vol. 20, no. 2, pp. 139–173, 2015.

[32] R. K. Bisht, H. Dhami, and N. Tiwari, “An evaluation of different statistical techniques of collocation extraction using a probability measure to word combinations,” Journal of Quantitative Linguistics, vol. 13, no. 2-3, pp. 161–175, 2006.

[33] D. Yarowsky, “One sense per collocation,” in Human Language Technology: Proceedings of a Workshop Held at Plainsboro, New Jersey, March 21-24, 1993, 1993.

[34] K. L. T. Bunag and R. A. Esquivel, “Transformer-based˜ conditional language models to generate Filipino news articles,” in Proceedings of the International Conference on Industrial

Engineering and Operations Management. Manila, Philippines: IEOM Society International, 2023, pp. 2231–2237. [Online]. Available: https://ieomsociety.org/proceedings/2023manila/595. pdf

[35] Unibersidad ng Santo Tomas. (2024) Hasaan. [Online]. Available: https://hasaan.ust.edu.ph/mga-artikulo/

[36] A. Dormiendo. (2024) Tunugan. [Online]. Available: https: //tunugan.substack.com/

[37] V. Almario. (2024) Virgilio almario - facebook. [Online]. Available: https://www.facebook.com/MakatangRioAlma/

[38] F. Fkih and M. N. Omri, “Estimation of a priori decision threshold for collocations extraction: an empirical study,” International Journal of Information Technology and Web Engineering (IJITWE), vol. 8, no. 3, pp. 34–49, 2013.

[39] P. Rychly, “A lexicographer-friendly association score.” in\` RASLAN, 2008, pp. 6–9.

[40] J. Materna, “Domain collocation identification.” in RASLAN, 2009, pp. 17–21.

[41] U. Schneider, “δ p as a measure of collocation strength: Considerations based on analyses of hesitation placement in spontaneous speech,” Corpus Linguistics and Linguistic Theory, vol. 16, no. 2, pp. 249–274, 2020.

[42] X. Xu, A. Hardie, and V. Brezina, “Keyness metrics,” in The x Corpus-Based Language Studies (xCBLS) Book. Lancaster University, 2023, accessed: 2025-03-19. [Online]. Available: https://www.lancaster.ac.uk/fass/projects/corpus/ZJU/ xCBLS/chapters/A06.pdf

[43] A. Santiago and N. Tiangco, Makabagong Balarilang Filipino. Quezon City: Rex Printing Company Inc., 1991.

[44] V. Brezina and W. Platt, “#lancsbox x,” 2024, [Software]. [Online]. Available: http://lancsbox.lancs.ac.uk

[45] L. Anthony, “Antconc (version 4.3.1),” Tokyo, Japan, 2024, [Computer Software]. [Online]. Available: https: //www.laurenceanthony.net/software/AntConc

[46] J. J. Randolph, “Free-Marginal Multirater Kappa (multirater K [free]): An Alternative to Fleiss’ Fixed-Marginal Multirater Kappa,” Online submission, 2005.

[47] J. R. Landis and G. G. Koch, “The measurement of observer agreement for categorical data,” Biometrics, vol. 33, no. 1, pp. 159–174, 1977.

[48] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov, “RoBERTa: A Robustly Optimized BERT Pretraining Approach,” 2019. [Online]. Available: https://arxiv.org/abs/1907.11692

[49] A. Grattafiori, A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Vaughan, and et al., “The Llama 3 Herd of Models,” 2024. [Online]. Available: https://arxiv.org/abs/2407.21783

[50] R. Ng, T. N. Nguyen, Y. Huang, N. C. Tai, W. Y. Leong, W. Q. Leong, X. Yong, J. G. Ngui, Y. Susanto, N. Cheng, and et al., “SEA-LION: Southeast Asian Languages in One Network,” 2025. [Online]. Available: https://arxiv.org/abs/2504.05747

[51] Y. Gong, B. Jiang, Y. Zhao, Y. Yuan, K. Chen, Y. Jiang, C. Chang, D. Hong, M. Chen, R. Li, Y. Zhang, and et al., “Moss-tts technical report,” 2026. [Online]. Available: https://arxiv.org/abs/2603.18090