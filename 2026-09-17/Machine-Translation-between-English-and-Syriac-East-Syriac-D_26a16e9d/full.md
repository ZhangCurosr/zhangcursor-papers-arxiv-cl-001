# Machine Translation between English and Syriac (East Syriac Dialect) using Statistical Machine Learning

Hadiana Sliwa and Hossein Hassani University of Kurdistan Hewlêr Kurdistan Region - Iraq {hadiana.harun, hosseinh}@ukh.edu.krd

## Abstract

UNESCO considers the Assyrian (Syriac) language an endangered language. Although Assyrians speak the language worldwide, the speaking population is uncertain (ranging from 500,000 to 1,500,000). Syriac is also one of the least studied languages in Natural Language Processing (NLP). Despite advances in Machine Translation (MT) over the past decade, the lack of publicly available corpora and the orthographic complexity of the Syriac script, specifically the Madnkhaya script, have left this language entirely ignored in the computational linguistics literature. This study develops the first phrase-based Statistical MT (SMT) model for Englishto-Assyrian MT using the Moses framework. We created a dataset of 38,847 sentence pairs from the complete English and Syriac Bible, merging a pre-existing New Testament dataset with an Old Testament built from scratch through PDF extraction, using custom segmentation scripts and manual alignment review by three bilingual annotators. The Syriac side of the corpus undergoes diacritic removal and Byte-Pair Encoding tokenization to reduce orthographic sparsity before training. We trained and evaluated six models using diferent configurations and splitting-scheme ratios, language model order, distortion limits, and the inclusion of an Operation Sequence Model. The best-performing configuration achieves a word-level BLEU score of 23.54. Human evaluation by 11 native Assyrian speakers resulted in mean adequacy and fluency scores of 3.42 and 3.34 out of 5, respectively. These results are consistent with comparable low-resource SMT models trained on Biblical corpora for morphologically rich Semitic languages. The corpora, scripts, and trained model are publicly available, providing the research community with the first systematically curated English–Syriac dataset and a reproducible baseline for future MT and broader NLP work on this endangered language.

## 1 Introduction

The digital era has seen a revolutionary shift in how we communicate across linguistic barriers, primarily driven by the advancement of Machine Translation (MT). However, this progress has not been equal for all languages. Although high-resource languages like English and French have close to human-quality translation (Hassan et al., 2018), lowresource languages such as Kurdish (Ahmadi et al., 2022), Assyrian (Syriac) (van Peursen et al., 2023), and many other languages are left behind. Some of these languages have little to no presence on the internet, with rich texts and scriptures still on physical documents stored in places such as churches, mosques, and other historical sites. Although research on Assyrian handwritten character recognition has reached encouraging results, achieving a maximum accuracy of 95.70% with the MobileNet-V2 model (Armya and Abdulrazzaq,

2024; Majeed and Hassani, 2024), such models still need further development before full handwritten documents can be reliably digitized.

A further limitation specific to the Assyrian language is “font-based transliteration”, where digitized documents are written with software such as MS Word using Arabic characters, with a Syriac font applied on top so that the Arabic Unicode characters visually display Assyrian (Syriac) letters (Kiraz, 2011). The underlying encoding remains Arabic; only the glyphs are swapped. Currently, no software or script reliably converts this Arabic-encoded text into actual Syriac Unicode. This practice originated before a Unicode range existed for Syriac, and adoption of the proper encoding has remained limited due to unfamiliar keyboard layouts and the lack of default system support for Syriac in operating systems and software.

The main challenge addressed in this work is the extreme scarcity of linguistic data for the Assyrian language, which is classified by UNESCO as an endangered language. Historically, Syriac speakers mainly resided in Middle Eastern countries, including Iran, Iraq, Syria, and Turkey. However, instability in the region has led to large-scale migration to countries such as the United States, Australia, and Canada, where the new generations are increasingly less likely to learn to speak, read, or write the language. Most of the available online material is in non-Unicode formats or scanned documents, and the lack of reliable OCR and text-processing tools for Syriac significantly limits large-scale digitization and preservation.

This research investigates whether a pipeline of devocalization and orthographic normalization can bridge the data gap in low-resource settings, using the traditionally dataeficient Moses Statistical Machine Translation (SMT) framework to establish the first reproducible baseline for the English–Syriac pair.

The rest of this paper is organized as follows: Section 2 gives a brief overview of the Syriac language and script, Section 3 reviews related work, Section 4 describes the method, Section 5 presents the results and discussion, and Section 6 concludes.

## 2 The Syriac Language

Syriac is a form of Aramaic belonging to the Northwest Semitic group of languages, a group that also includes Hebrew, Ugaritic, and Phoenician; within the family tree, Aramaic and Arabic are grouped together under “Central Semitic”, distinct from East Semitic (Akkadian) and South Semitic (Ethiopic) (Brock, 2017). Its historical and religious significance lies in its role as a foundational language of early Syriac Christianity and as a “bridge culture” through which Greek philosophy, medicine, and science were transmitted to the Islamic world and eventually to Western Europe (Brock, 2017; Muraoka, 2005).

Syriac has two main dialects, West Syriac and East Syriac (Kiraz, 2013). These are not separate languages, but dialects that are mostly identical in grammar and lexicon, with diferences that are largely phonological and orthographic. This work focuses on East Syriac, used by the Church of the East and Chaldean communities, written in the Madnkhaya script, which emerged around the 6th century.

The Syriac alphabet is essentially consonantal, consisting of twenty-two letters written and read from right to left in a cursive, joined-up style (Muraoka, 2005). Three letters (Alap ܐ, Waw ܘ, and Yodh ܝ( are bivalent: they can function as consonants, act as vowel letters, or, in the case of Alap, be entirely silent. Six plosive consonants possess a twofold (hard/soft) pronunciation governed by the Rukakha and Qushshaya dots, and eight letters join only to the right, never connecting to the letter that follows them. Letter forms also change depending on their position within a word, and several letter pairs are visually similar and easily confused (e.g., Dalath $\textnormal { \textsf { 2 } V S } .$ Resh ܪ(.

East Syriac orthography uses seven vowel markers (West Syriac uses five), realised as combinations of dots placed above or below consonants, as summarised in Table 1. In addition to vowels, the diacritic system includes Seyāmē (a double dot marking plural nouns, e.g. ܡܠܟܐ ̈ /malkē/ ‘kings’ vs. ܡܠܟܐ /malkā/ ‘king’) and Mṭalqānā (a diagonal line marking a silent letter) (Muraoka, 2005). In everyday writing, native readers and writers routinely omit vowel diacritics, relying on consonantal context and morphological knowledge, a foundational characteristic of Semitic writing systems shared with Arabic and Hebrew. As Section 4.2 explains, this convention directly motivates the devocalization step in our preprocessing pipeline, because the same underlying word may appear fully vocalized, partially vocalized, or unvocalized in the source data.

Table 1: East Syriac vowel diacritics and their functions
<table><tr><td>Pronunciation</td><td>Mark</td><td>Name</td><td>Function / Value</td></tr><tr><td>/Ba/ /Bā/ /Bi/ /Bē/ /Bo/</td><td>山 山 山 山 </td><td>Ptakha Zqapha Zlameh Kirye Zlameh Yareekhe Rwakha Rwasa</td><td>short  $/ \mathrm { a } /$  vowel long  $/ \bar { \mathrm { a } } /$  vowel short  $/ \mathrm { i } /$  sound  $/ \epsilon /$  sound O sound oo or u sound</td></tr></table>

Morphologically, Syriac follows the Semitic root-and-pattern system: a single root can generate a large number of surface word forms through prefixation, sufixation, and cliticization, and nouns, adjectives, and verbs must agree in gender and number with their nucleus noun (Muraoka, 2005). Syntactically, Classical Syriac frequently uses Verb–Subject constructions, diverging from the Subject–Verb–Object order of English, a structural divergence with direct consequences for word alignment, discussed in Section 4.3.

## 3 Related Work

This section addresses research from two intersecting areas relevant to the current study: MT for low-resource languages and computational work on Syriac and its linguistically adjacent languages. We do not focus on high-resource languages, which mostly now use Neural Machine Translation because the research community is well aware of their status and they rely on very large-scale corpora or pre-trained models, which are unavailable for Syriac.

## 3.1 Machine Translation in Low-Resource Languages

MT has undergone a revolutionary transformation over the previous decades, evolving from rule-based systems to SMT, and then to the highly fluent Neural Machine Translation (NMT) models used today. Many approaches have been taken to improve MT in lowresource settings, such as active learning, data augmentation, and transfer learning, but the scarcity of data remains the critical constraint (Tafa et al., 2025).

Kumar et al. (2021) focused on adapting MT systems to low-resource language varieties and typologically related languages, proposing a transfer-learning framework named LangVarMT that adapts a model trained on a “standard” variety of a language to a related low-resource variety through embedding mapping, vocabulary recycling, back-translation, and a final training pivot. The framework outperformed all competitive baselines, including Ukrainian, Belarusian, Nynorsk, and several Arabic dialects; with only 10,000 monolingual sentences, it surpassed the strongest baselines by more than four BLEU points for Ukrainian. However, its success depends heavily on the relatedness of the standard and target varieties, and it cannot function for a language –in our case, the Syriac script– that has no suitable pre-trained “bridge” model.

Velayuthan et al. (2024) showed the importance of data quality over quantity for lowresource pairs. Although the final model was neural, statistical methods drove their bilingual filtering pipeline, merging rule-based cleaning with Jensen-Shannon Divergence to select high-quality samples. The model reached maximum performance at 100K sentences (50.04 chrF, 18.17 BLEU) before performance declined as a result of adding more low-quality data.

When datasets are limited, SMT is often preferred over NMT (Koehn and Knowles, 2017): it does not require large amounts of data and takes less time to train. Kakum and Sambyo (2022) implemented a phrase-based SMT system for Nyishi, a low-resource and endangered Indian language, collecting 30,000 pairs of sentences of English–Nyishi, crosschecked by native elders and language experts, and using the Moses toolkit, MGIZA, and IRSTLM toolkit. Tuning on 967 sentence pairs increased BLEU from 0.1419 to 0.1849 for Nyishi–English and from 0.0802 to 0.1411 for English–Nyishi.

## 3.2 Machine Translation in Less-Studied Languages Close to Syriac

At the time of writing, computational research dedicated specifically to Syriac in the MT domain remains remarkably scarce. An exception is Naaijer et al. (2023), in which the authors developed a transformer-based sequence-to-sequence model to parse the morphology of Ancient Syriac, training on 5,596 Syriac verses from the ETCBC database augmented with 22,946 verses of the Hebrew Masoretic Text. The model trained on Syriac data alone reached 89.3% accuracy, and the best Hebrew-augmented model reached 90.8%, a modest gain considering that the Hebrew data is almost four times larger than the Syriac dataset and substantially increases training cost.

Closer to our work, Liebeskind et al. (2024) built a publicly available Biblical parallel Aramaic–Hebrew corpus covering the entire Hebrew Bible and compared phrase-based SMT against RNN and Transformer NMT architectures. The results were impressive: SMT outperformed NMT in this low-resource context, achieving a BLEU score of 44.3 compared to NMT’s 35.8, with the lack of training data cited as the main obstacle for the neural models. Similarly, Guellil et al. (2007) developed a two-step framework that first transliterates Arabizi into Arabic script and then translates Algerian dialect into Modern Standard Arabic. While the neural approach won on transliteration, SMT outperformed NMT on the actual translation task (6.01 BLEU vs. scores often at 0.0), on a corpus of only 6,412 parallel sentences.

To summarize, across the reviewed literature, a consistent picture emerges: in lowresource settings, SMT is the preferred approach due to the lower data requirements, faster training, and robustness to sparse and noisy datasets (Koehn and Knowles, 2017; Liebeskind et al., 2024; Guellil et al., 2007). Since large-scale datasets for the Assyrian language are not available, this study follows the SMT approach.

## 4 Method

This research uses the Moses phrase-based SMT framework. Figure 1 shows the overall flow of the method, with each block explained in the following subsections.

![](images/417c2e9f44ece3ad59e87f60c1508c96ee5ffd4606fd4c55b775d66c999d4689.jpg)  
Figure 1: Overview of the research method

## 4.1 Data Collection and Curation

To address the lack of credible sources for Syriac translation, we collect data from sources such as Biblical and educational references. This stage includes data curation, corpus compilation, and deduplication.

## 4.2 Data Preprocessing

The stage consists of cleaning and normalization, devocalization, Byte-Pair Encoding tokenization of subwords, and data splitting.

Cleaning and normalization – Noise removal is applied to both sides of the corpus: a custom cleaning script (e.g., using Python) strips stray punctuation, symbols, inconsistent whitespace, and formatting characters introduced during extraction, and removes empty, duplicate, or one-sided sentence pairs. The most linguistically significant step at this stage is Syriac diacritic removal. As discussed in Section 2, native readers routinely omit vowel diacritics in everyday text, although the same underlying word may appear in multiple surface forms depending on vocalization. This sparsity is particularly problematic in low-resource settings, where each unique surface form competes for a limited training signal. We therefore apply a devocalization script that removes all diacritical marks from the Syriac data using their Unicode code points, creating a consistent consonantal representation, consistent with established practice in Semitic MT preprocessing, where vowel reduction is a standard step for Arabic and Hebrew (Habash and Sadat, 2006).

Subword tokenization – We apply Byte-Pair Encoding (BPE) on both sides of the corpus. BPE iteratively merges the most frequent character-sequence pairs, producing a subword vocabulary that sits between character-level and word-level representations (Sennrich et al., 2016). This suits the morphological characteristics of Syriac, which generates a large number of surface forms from a relatively small set of roots; BPE decomposes rare or unseen words into known subword units, substantially reducing the out-of-vocabulary problem.

Data splitting – After tokenization, the corpus is partitioned into training and test sets. To assess how translation performance scales with training data size, we train and evaluate under four split configurations: 95/5, 90/10, 80/20, and 70/30. Evaluating across multiple training sizes is an established methodology in low-resource MT research (Kakum and Sambyo, 2022; Velayuthan et al., 2024). All splits are created by random sampling without replacement, so test sentences are never seen during training. Moses performs tuning internally via MERT on a held-out tuning portion, so no separate development set is required.

## 4.3 Model Training

The Moses pipeline integrates three interdependent components: a word alignment model, a phrase translation table, and a target-language model. The overall configuration is summarized in Table 2.

Word alignment with GIZA++ – We use GIZA++, which implements the IBM alignment models 1–5 using the Expectation-Maximization algorithm (Al-Onaizan et al., 1999), run independently in both translation directions and symmetrized with the growdiag-final-and heuristic, the Moses default shown to produce the best balance between alignment precision and recall (Koehn et al., 2003). Alignment is particularly challenging for the English–Syriac pair: English follows SVO order while Syriac frequently uses Verb– Subject constructions, and Syriac’s prefixed prepositions and pronominal sufixes mean a single Syriac token may correspond to multiple English words. The devocalization applied earlier reduces surface variation and directly supports alignment quality.

Phrase table construction – Moses extracts all contiguous bilingual phrase pairs consistent with the word alignments up to a maximum phrase length, computing five feature scores per pair (forward and reverse translation probabilities, forward and reverse lexical weights, and a phrase penalty), combined in Moses’s log-linear model whose weights are optimized during tuning.

Language model – We train a target-side KenLM language model on the Syriac side of the training corpus with modified Kneser-Ney smoothing, which handles the sparse count distributions characteristic of low-resource datasets efectively. A trigram order is selected as appropriate for the baseline corpus size; higher-order models require more data to estimate reliably.

<table><tr><td>Component</td><td>Setting</td></tr><tr><td>Decoder Alignment tool Symmetrization heuristic Language model toolkit Language model order Smoothing Tuning algorithm Evaluation metric</td><td>Moses phrase-based SMT GIZA++ (IBM Models 1–5) grow-diag-final-and KenLM 3-gram (baseline) / 5-gram (enhanced) Modified Kneser-Ney MERT</td></tr></table>

Table 2: Moses SMT system configuration

Once the phrase table and language model are created, the log-linear model is tuned with MERT, which decodes a held-out tuning set repeatedly, compares the hypotheses against reference translations, and iteratively updates the feature weights to maximize BLEU. The full pipeline, alignment, phrase extraction, language model integration, and tuning, is coordinated through Moses’s train-model.perl script and executed locally; no cloud or GPU resources are required, which is one of the practical advantages of phrase-based SMT for low-resource settings (Koehn and Knowles, 2017). Training runs are performed independently for each of the four data splits, producing separate trained models for comparative evaluation.

## 4.4 Evaluation

We evaluate the trained models using two complementary approaches. The main automatic metric is BLEU (Papineni et al., 2002), which measures n-gram overlap between hypothesis and reference translations with a brevity penalty. Despite its well-documented limitations, insensitivity to synonymy and weaker correlation with human judgment in morphologically rich languages, BLEU remains the standard benchmark in SMT research and enables direct comparison against related low-resource Semitic MT work. BLEU is computed with the Moses scoring scripts across all four split configurations.

Automatic metrics alone are insuficient for a morphologically complex, low-resource language such as Syriac, where reference translations may themselves exhibit variation (Fernández and Matamala, 2021). We therefore complement BLEU with a structured human evaluation by a panel of bilingual native Assyrian speakers. Evaluators receive a standardized assessment form containing English source sentences drawn from web articles not seen during training, alongside their model-generated Syriac translations, and rate each translation on a 5-point scale along two dimensions: Adequacy (the degree to which the translation preserves the meaning of the source) and Fluency (the degree to which the translation reads as natural, grammatically well-formed Syriac). This two-dimensional framework is standard in MT evaluation and separates meaning preservation from output naturalness.

## 5 Results and Discussion

## 5.1 Data Collection

We collected data from biblical sources. The selection of Biblical text as the primary data source is motivated by practical and linguistic factors. Practically, the Bible is one of the very few texts for which a complete verse-aligned parallel exists between English and Syriac, owing to the Peshitta’s central role in the Assyrian Christian tradition. Linguistically, Biblical Syriac uses the morphological and orthographic features of the Madnkhaya script that this research targets. The domain limitation is acknowledged: the resulting model is suited to the vocabulary of the Biblical text and is not intended as a general-purpose translation system, consistent with comparable low-resource Semitic MT research where domain-specific corpora are used when general corpora are unavailable (Liebeskind et al., 2024).

For the New Testament portion ( 7,900 verses), we curate pre-existing verse-aligned English–Syriac text identified in academic and religious digital archives, passed through the same cleaning and normalization pipeline as the rest of the corpus. The Old Testament ( 23,000 verses) presents a considerably greater challenge: no pre-existing digital parallel corpus is available in a clean, verse-aligned format, so we build this portion from scratch in four stages. First, raw text is extracted from the source documents using Python-based PDF extraction scripts. Second, a custom segmentation script uses the chapter-and-verse numbering system present in both Biblical sources (e.g., 1:1, 1:2) as anchor points to split the continuous text into discrete verse-level segments, an established practice in Biblical corpus construction, where verse numbers serve as natural and reliable alignment anchors across language versions. Third, the segmented verse pairs are combined into a single parallel file, one matched English–Syriac pair per line. Fourth, a manual review phase corrects the noise, misalignment, and formatting errors that automated PDF extraction inevitably introduces, particularly when processing right-to-left Syriac script.

We also collected data from the Ministry of Education (the Kurdistan Regional Government in the Kurdistan Region of Iraq). The collected documents are written in the same Madnkhaya script. However, we received them after the deadline for completing this research, so we could not incorporate them into our model. This data source includes non-liturgical, secular Syriac text, a domain almost absent from existing digital resources for this language. These materials are being processed through the same normalization pipeline and, unlike the Biblical corpus, they will not be released publicly: under the terms agreed with the Ministry, they are shared privately with individual researchers on written request, for academic research purposes only, and are randomised at sentence level so that they cannot be reconstructed as a teaching curriculum. Requests may be directed to the corresponding author.

## 5.2 Corpus creation

We create a parallel English–Assyrian corpus of 38,847 aligned sentence pairs extracted from Biblical text.

Three individuals, the primary researcher and two collaborators with working knowledge of both English and Syriac, reviewed the aligned corpus. The Old Testament corpus is divided into sections, each inspected independently by a single reviewer, who corrects misalignments, removes pairs where extraction has failed, and flags verses where the Syriac text is incomplete. This division of labor is practical in low-resource NLP contexts where annotation resources are limited, and the resulting corpus represents a human-verified foundation for model training.

The parallel corpus contains 38,847 sentence pairs from the complete English and Syriac Bible; a sample is shown in Figure 2. Following preprocessing with clean-corpus-n.perl, sentence pairs where either side exceeded 100 BPE tokens were removed. The corpus was shufled with a fixed seed of 42 for reproducibility. A fixed tuning set of 1,000 sentences was held out for MERT across all models, and the remaining data was partitioned according to the four split ratios. Table 3 summarizes the corpus and split statistics, and Table 4 summarizes the six model configurations: Models 1–4 form the baseline set trained on identical configurations across four data splits, while Models 5 and 6 use the best-performing split (95/5) with architectural enhancements.

![](images/bd2cbcbe1d6573bb3ef5acf9b4d57b63ceb848b8a5566db5144f27c1409f5adb.jpg)  
Figure 2: Sample from the curated dataset

## 5.3 Experimental Setup

The experiments were performed according to Table 3.

## 5.4 Automatic Evaluation Results

Table 5 reports the word-level BLEU scores for all six models, visualized in Figure 3. All reported BLEU scores are computed at the word level using multi-bleu.perl, a script in Moses after de-BPE post-processing; BPE-level BLEU is shown for diagnostic purposes only, as BPE tokenization artificially inflates n-gram precision by shortening tokens (Sennrich et al., 2016).

The baseline results confirm the expected monotonic relationship between training data volume and translation quality: BLEU decreases from 22.54 at the 95/5 split to 17.81 at the 70/30 split, a drop of 4.73 points as the training set shrinks from 35,905 to 26,193 sentence pairs. This pattern is consistent with findings across low-resource SMT research, where data volume is the primary ceiling on translation quality (Kakum and Sambyo, 2022; Velayuthan et al., 2024).

<table><tr><td>Model</td><td>Split</td><td>Train</td><td>Tune</td><td>Test</td></tr><tr><td>Model 1 Model 2 Model 3 Model 4 Model 5</td><td>95/5 90/10 80/20 70/30 95/5</td><td>35,905 33,963 30,078 26,193 35,905</td><td>1,000 1,000 1,000 1,000 1,000</td><td>1,942 3,884 7,769 11,654 1,942</td></tr><tr><td colspan="2">Model 6 95/5 Total sentence pairs</td><td colspan="3">35,905 1,000 1,942 38,847</td></tr></table>

Table 3: Corpus and data split statistics

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>LM</td><td rowspan=1 colspan=1>Distortion</td><td rowspan=1 colspan=1>BPE</td><td rowspan=1 colspan=1>OSM</td><td rowspan=1 colspan=1>Split</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>3-gram</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>95/5</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>3-gram</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>90/10</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>3-gram</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>80/20</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Baseline</td><td rowspan=1 colspan=1>3-gram</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>70/30</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Enhanced (OSM)</td><td rowspan=1 colspan=1>5-gram</td><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>10k</td><td rowspan=1 colspan=1>Yes</td><td rowspan=1 colspan=1>95/5</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Enhanced (dl/BPE)</td><td rowspan=1 colspan=1>5-gram</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>20k</td><td rowspan=1 colspan=1>No</td><td rowspan=1 colspan=1>95/5</td></tr></table>

Table 4: Summary of all model configurations

Table 6 provides the n-gram precision breakdown for the best-performing model (Model 5). The brevity penalty of 1.000 confirms that Model 5 does not produce systematically shorter translations than the reference. The unigram precision of 48.7 indicates that approximately half of the words produced match the reference, a reasonable outcome for a morphologically rich target language where surface-form variation is high even for semantically equivalent outputs. The drop from unigram to 4-gram precision (48.7 12.7) reflects the dificulty of producing exact phrase sequences in Syriac, where inflectional variation means that even correct translations may not match the reference at the 3- and 4-gram level.

Efect of model enhancements – Model 5, which adds an Operation Sequence Model (OSM) and upgrades the LM from 3-gram to 5-gram, achieves the highest wordlevel BLEU of 23.54, an improvement of +1.00 over the identical-split baseline. The OSM provides an explicit reordering feature that captures longer-range word order changes through operation sequences (Durrani et al., 2011), particularly suitable for the English– Syriac pair given its SVO-to-VS structural divergence (Carpuat et al., 2010), while the 5-gram KenLM contributes more fluent target-side generation. Model 6, which increases the distortion limit from 6 to 12 and doubles BPE merge operations to 20,000, achieves 22.63 BLEU, a gain of only +0.09 over the baseline, suggesting that long-range reordering beyond the default limit is not the primary bottleneck for this corpus, and that the singleregister nature of the Biblical training data imposes a ceiling that additional reordering capacity cannot overcome.

## 5.5 Human Evaluation Results

Twenty English sentences translated by Model 5, drawn from online articles covering Biblical topics not in the training data, were each rated by eleven native Assyrian speakers on adequacy and fluency (5-point scales), yielding 220 individual ratings per dimension.

<table><tr><td>Model</td><td>Model/Split</td><td>Configuration</td><td>Test Size</td><td>Word BLEU</td><td>BPE BLEU*</td></tr><tr><td>1</td><td>Baseline 95/5</td><td>3-gram, dl=6, 10k BPE, MERT</td><td>1,942</td><td>22.54</td><td>23.95</td></tr><tr><td>2</td><td>Baseline 90/10</td><td>3-gram, dl=6, 10k BPE, MERT</td><td>3,884</td><td>21.76</td><td>22.91</td></tr><tr><td>3</td><td>Baseline 80/20</td><td>3-gram, dl=6, 10k BPE, MERT</td><td>7,769</td><td>19.85</td><td>20.93</td></tr><tr><td>4</td><td>Baseline 70/30</td><td>3-gram, dl=6, 10k BPE, MERT</td><td>11,654</td><td>17.81</td><td>18.84</td></tr><tr><td>5</td><td>Enhanced</td><td>OSM, 5-gram, dl=6, 10k BPE, MERT</td><td>1,942</td><td>23.54</td><td>24.60</td></tr><tr><td>6</td><td>Enhanced</td><td>5-gram, dl=12, 20k BPE, MERT</td><td>1,942</td><td>22.63</td><td>22.64</td></tr><tr><td colspan="6">* BPE BLEU shown for diagnostic comparison only; not used for evaluation</td></tr></table>

Table 5: BLEU results for all six models

![](images/8167f83605b022358424c5030f1d09c1ea6b7ce6c4a200cfc5ba728c60193239.jpg)  
Figure 3: Word-level BLEU scores across all six model configurations

Table 7 presents the aggregate statistics.

The mean adequacy of 3.42 and fluency of 3.34 imply that, on average, evaluators rated the model’s translations above the midpoint of the scale, indicating at least partial but meaningful preservation of meaning and some degree of target-language naturalness. The high standard deviations (1.42 and 1.49) reflect a fair amount of variation across both sentences and evaluators; native speakers of Assyrian difer in their exposure to Classical Syriac, their familiarity with the Biblical register, and their personal standards of grammaticality (Artstein and Poesio, 2008). The very strong correlation between adequacy and fluency (r = 0.865) indicates that evaluators tended to rate the same translations as both adequate and fluent, or both inadequate and disfluent, suggesting the two dimensions are tightly coupled in this context.

At the sentence level, three sentences achieved an average overall score at or above 4.0 (“Moses was born in Egypt”, 4.27; “Jesus rose from death three days after”, 4.09; “Peace rested upon the valley”, 4.05), all notably short, syntactically simple sentences built from high-frequency Biblical vocabulary. The lowest-scoring sentence (“God promised Abraham that his descendants would be as many as the stars”, 2.68) involves a complex embedded clause and a metaphorical comparison that the phrase-based model struggles to handle, consistent with the known limitations of SMT on complex syntactic and semantic structures (Koehn et al., 2003).

The distribution of individual ratings, shown in Figure 4, reveals a bimodal pattern: score 5 is the single most frequent rating for both dimensions (33.2% for adequacy, 34.5% for fluency), alongside a notable tail of low scores. This bimodality is characteristic of phrase-based SMT output in low-resource settings, where translations of short, highfrequency phrases are often excellent while translations of longer or more complex sentences are poor (Kakum and Sambyo, 2022).

<table><tr><td rowspan=1 colspan=1>BLEU</td><td rowspan=1 colspan=1>1-gram</td><td rowspan=1 colspan=1>2-gram</td><td rowspan=1 colspan=1>3-gram</td><td rowspan=1 colspan=1>4-gram</td><td rowspan=1 colspan=1>BP</td><td rowspan=1 colspan=1>Ratio</td></tr><tr><td rowspan=1 colspan=1>23.54</td><td rowspan=1 colspan=1>48.7</td><td rowspan=1 colspan=1>27.5</td><td rowspan=1 colspan=1>18.1</td><td rowspan=1 colspan=1>12.7</td><td rowspan=1 colspan=1>1.000</td><td rowspan=1 colspan=1>1.025</td></tr></table>

Table 6: N-gram BLEU breakdown for Model 5 (word-level)
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Total individual ratings</td><td>220</td></tr><tr><td>Mean adequacy score</td><td>3.42 / 5</td></tr><tr><td>Mean fluency score</td><td>3.34 / 5</td></tr><tr><td>Mean overall score</td><td>3.38 / 5</td></tr><tr><td>Std. deviation (adequacy)</td><td>1.42</td></tr><tr><td>Std. deviation (fluency)</td><td>1.49</td></tr><tr><td>Adequacy ≥ 3 (%)</td><td>69.5%</td></tr><tr><td>Fluency ≥ 3 (%)</td><td>65.0%</td></tr><tr><td>Adequacy-fluency correlation</td><td>0.865</td></tr></table>

Table 7: Overall human evaluation results (Model 5, 20 sentences, 11 evaluators)

![](images/d0349d1df9d8f58540bd7b4cb2e0e71a7676d0d1ba5d542721a93e96cd7c66fe.jpg)  
Figure 4: Distribution of adequacy and fluency ratings across all 220 judgments

## 5.6 Comparison with Related Work

The best word-level BLEU achieved in this study is 23.54 (Model 5). Table 8 places this result in context relative to comparable low-resource phrase-based SMT studies using Moses, Biblical or similarly domain-restricted corpora, and low-resource or morphologically rich target languages.

The score of 23.54 compares favorably with comparable low-resource SMT baselines on morphologically rich languages trained on Biblical corpora. Tedla and Yamamoto (2016), working on English–Tigrinya, another Semitic language with a comparable Bible-derived corpus, report BLEU scores in the low-to-mid twenties using Moses with morphological segmentation, placing our result in a directly comparable range. The substantially higher 44.30 reported by Liebeskind et al. (2024) for Aramaic–Hebrew reflects the linguistic proximity of that pair: shared script, largely shared vocabulary, and closely related morphology make alignment substantially easier than for the typologically distant English–Syriac pair. The very low BLEU reported for English–Nyishi underscores that BLEU scores are not comparable across language pairs, only within them (Papineni et al., 2002).

<table><tr><td>Study</td><td>Language Pair</td><td>Corpus Size</td><td>Best BLEU</td></tr><tr><td>This study (Model 5)</td><td>English → Syriac (Assyrian)</td><td>~36k pairs</td><td>23.54</td></tr><tr><td>Liebeskind et al. (2024)</td><td>Aramaic → Hebrew</td><td>~23k pairs</td><td>44.30</td></tr><tr><td>Kakum and Sambyo (2022)</td><td>English → Nyishi</td><td>30k pairs</td><td>0.18</td></tr><tr><td>Tedla and Yamamoto (2016)</td><td>English → Tigrinya</td><td>~31k pairs</td><td>~20</td></tr></table>

Table 8: Comparison with related low-resource phrase-based SMT results

## 5.7 Analysis of Model Behavior

Three patterns emerge across the six models. First, data volume is the primary ceiling: the 4.73-point drop from the 95/5 to the 70/30 split demonstrates that training data size is the dominant factor in translation quality, consistent with Velayuthan et al. (2024) and with the broader observation that phrase tables become sparser and less reliable as training data shrinks (Koehn and Knowles, 2017). Second, the OSM contributes meaningfully: the +1.00 BLEU gain in Model 5 is the single most efective improvement in this study, providing the decoder with an explicit reordering signal for the English–Syriac word-order divergence (Durrani et al., 2011). Third, higher distortion and larger BPE vocabularies ofer diminishing returns: Model 6’s negligible gain suggests that the corpus’s single-domain coverage is the binding constraint rather than reordering capacity or vocabulary coverage. The practical ceiling for Moses SMT on this corpus is approximately 23.5 BLEU, and pushing beyond it will require either additional parallel data or a transition to neural MT with cross-lingual transfer from a related higher-resource language such as Arabic (Boujkian, 2025).

## 5.8 Preliminary NMT Experiments

As a preliminary investigation into neural alternatives, fine-tuning experiments were conducted using two large multilingual NMT models, mBART-50 and NLLB-200, both finetuned on the same 95/5 data partition used for Models 1 and 5, with training taking approximately 43 hours per model on cloud GPU infrastructure. Both models produced empty output at inference time and could not be evaluated. The root cause was identified as the absence of Syriac (syr\_Syrc) from both models’ pre-trained vocabularies: without a valid target-language token to condition generation on, the decoder had no starting point and produced no output regardless of the input. These experiments are reported as a negative result; resolving the issue requires adding syr\_Syrc as a new token, initialising its embedding from Arabic, and setting forced\_bos\_token\_id correctly, as outlined in Section 6. The result reinforces the finding, consistent across the low-resource literature (Koehn and Knowles, 2017; Liebeskind et al., 2024; Guellil et al., 2007), that SMT remains the practical choice for language pairs absent from pre-trained multilingual vocabularies.

## 5.9 Limitations

Several limitations of this study should be acknowledged. The corpus is restricted entirely to Biblical text, so the model has no exposure to modern vernacular Assyrian, secular vocabulary, or non-liturgical discourse structures; its scores reflect performance within a narrow domain and should not be interpreted as measures of general-purpose translation quality. The single-reference BLEU evaluation underestimates true translation quality, as correct translations that difer from the reference in surface form, a common occurrence in morphologically rich languages, receive no credit (Papineni et al., 2002). The phrase-based architecture cannot capture deep semantic nuances or complex metaphors as efectively as neural architectures. Finally, the human evaluation panel, while suficient for interannotator agreement estimation (Artstein and Poesio, 2008), evaluates only the output of the best model and does not directly compare quality across models.

## 6 Conclusion and Future Work

This research presents the first published SMT model for the English-to-Syriac language pair, built using the Moses phrase-based framework and trained on a parallel corpus of 38,847 sentence pairs extracted from the complete English and Syriac Bible. The study shows that a functioning and reasonably accurate MT system can be created for this endangered low-resource language using publicly available tools and a corpus assembled from scratch, and it establishes quantitative baselines against which future systems for this language pair can be measured.

Across six model configurations, the outcome showed a reasonable translation quality, with word-level BLEU declining from 22.54 at the 95/5 split to 17.81 at the 70/30 split. The best configuration combines a 5-gram KenLM language model with an Operation Sequence Model for explicit reordering, achieving a word-level BLEU of 23.54; the +1.00 improvement over the identical-split baseline reflects the genuine benefit of addressing the SVO-to-VS structural divergence between English and Syriac. Human evaluation by eleven native Assyrian speakers produced mean adequacy and fluency scores of 3.42 and 3.34 out of 5, with strong performance on short, high-frequency constructions and weak performance on complex ones, the defining characteristic of phrase-based SMT in lowresource settings. Beyond the translation model, this work contributes a verse-aligned, manually verified parallel corpus prepared for public release, and a collection of secular Syriac documents from the Ministry of Education being processed for release through the same pipeline and available privately to researchers who request them for academic use.

Future work could be summarized as follows. First, expanding the parallel corpus beyond Biblical texts: the Ministry of Education documents will introduce secular vocabulary and register diversity, and sources such as the Tatoeba English–Assyrian dataset and the NENA (North-Eastern Neo-Aramaic) Corpus represent near-term options. Second, replacing language-agnostic BPE with a morphology-aware segmenter tailored to Syriac’s root-and-pattern system, following the significant BLEU gains shown for the closely comparable English–Tigrinya pair (Tedla and Yamamoto, 2016). Third, completing the NMT comparison: adding syr\_Syrc to the vocabularies of NLLB-200 and mBART-50, initializing its embedding from Arabic, would enable a direct neural comparison against the SMT baselines established here, and in the longer run, cross-lingual transfer from Arabic represents the most promising path toward a general-purpose system (Boujkian, 2025). Fourth, evaluation should move to multi-reference scoring to better handle surfaceform variation in Syriac, and the human evaluation panel could be expanded to include speakers with diferent levels of familiarity with the language.

## Ethical Consideration

The collected Biblical data are publicly available. The data from the Ministry of Education (the Kurdistan Regional Government in the Kurdistan Region of Iraq) could be

available upon request.

## Author Contributions

Conceptualization, Hadiana Sliwa (H.S.) and Hossein Hassani (H.H.); methodology, H.S. and H.H.; software, H.S.; validation, H.S.; formal analysis, H.S.; investigation, H.S.; resources, H.S. and H.H.; data curation, H.S.; preparing first draft, H.S.; revising the draft and preparing final manuscript, H.H.; visualization, H.S.; supervision, H.H.; project administration, H.H.

## Dataset Availability

The dataset is available here.

## Acknowledgments

We thank the two collaborators who participated in the manual alignment review, the eleven native Assyrian speakers who participated in the human evaluation, and the Assyrian-related department of the Ministry of Education (the Kurdistan Regional Government in the Kurdistan Region of Iraq) for providing secular Syriac documents and supporting their use in this research.

## References

Ahmadi, S., Hassani, H., and Jaf, D. Q. (2022). Leveraging Multilingual News Websites for Building a Kurdish Parallel Corpus. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 21(5), April.

Al-Onaizan, Y., Curin, J., Jahr, M., Knight, K., Laferty, J., Melamed, D., Och, F.-J., Purdy, D., Smith, N. A., and Yarowsky, D. (1999). Statistical Machine Translation: Final Report, JHU Workshop 1999. Final report, Center for Speech and Language Processing, Johns Hopkins University.

Armya, R. E. and Abdulrazzaq, M. B. (2024). Handwritten Character Recognition in Assyrian Language Using Convolutional Neural Network. Science Journal of University of Zakho, 12(1):105–115, Mar.

Artstein, R. and Poesio, M. (2008). Survey Article: Inter-Coder Agreement for Computational Linguistics. Computational Linguistics, 34(4):555–596.

Boujkian, S. (2025). Improving Low-Resource Machine Translation via Cross-Linguistic Transfer from Typologically Similar High-Resource Languages.

Brock, S. (2017). An Introduction to Syriac Studies. Gorgias Handbooks. Gorgias Press LLC, Piscataway, NJ, 3rd edition.

Carpuat, M., Marton, Y., and Habash, N. (2010). Improving Arabic-to-English Statistical Machine Translation by Reordering Post-verbal Subjects for Alignment. In Proceedings of the Human Language Technology Conference ofthe North American Chapter ofthe Association for Computational Linguistics (HLT-NAACL 2010), pages 178–183.

Durrani, N., Schmid, H., and Fraser, A. (2011). A Joint Sequence Translation Model with Integrated Reordering. In Dekang Lin, et al., editors, Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 1045–1054, Portland, Oregon, USA, June. Association for Computational Linguistics.

Fernández, A. and Matamala, A. (2021). Human Evaluation of Three Machine Translation Systems: From Quality to Attitudes by Professional Translators. Vigo International Journal of Applied Linguistics, pages 123–148, 01.

Guellil, I., Azouaou, F., and Abbas, M. (2007). Neural Vs Statistical Translation of Algerian Arabic Dialect written with Arabizi and Arabic letter. In Neural Vs Statistical Translation of Algerian Arabic Dialect written with Arabizi and Arabic letter, 11.

Habash, N. and Sadat, F. (2006). Arabic Preprocessing Schemes for Statistical Machine Translation. In Proceedings of the Human Language Technology Conference of the North American Chapter of the ACL (HLT-NAACL), pages 49–52.

Hassan, H., Aue, A., Chen, C., Chowdhary, V., Clark, J., Federmann, C., Huang, X., Junczys-Dowmunt, M., Lewis, W., Li, M., Liu, S., Liu, T.-Y., Luo, R., Menezes, A., Qin, T., Seide, F., Tan, X., Tian, F., Wu, L., Wu, S., Xia, Y., Zhang, D., Zhang, Z., and Zhou, M. (2018). Achieving Human Parity on Automatic Chinese to English News Translation.

Kakum, N. and Sambyo, K. (2022). Phrase-Based English–Nyishi Machine Translation, 09.

Kiraz, G. A. (2011). Forty years of syriac computing. Hugoye: Journal of Syriac Studies, 10(1):33–54.

Kiraz, G. (2013). The New Syriac Primer: An Introduction to the Syriac Language. Gorgias Handbooks. Gorgias Press, Piscataway, NJ.

Koehn, P. and Knowles, R. (2017). Six Challenges for Neural Machine Translation. In Thang Luong, et al., editors, Proceedings of the First Workshop on Neural Machine Translation, pages 28–39, Vancouver, August. Association for Computational Linguistics.

Koehn, P., Och, F. J., and Marcu, D. (2003). Statistical Phrase-based Translation. In Proceedings of NAACL-HLT, pages 48–54.

Kumar, S., Anastasopoulos, A., Wintner, S., and Tsvetkov, Y. (2021). Machine Translation into Lowresource Language Varieties. In Chengqing Zong, et al., editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 110–121, Online, August. Association for Computational Linguistics.

Liebeskind, C., Liebeskind, S., and Bouhnik, D. (2024). Machine Translation for Historical Research: A Case Study of Aramaic-Ancient Hebrew Translations. J. Comput. Cult. Herit., 17(2), February.

Majeed, A. and Hassani, H. (2024). Ancient but Digitized: Developing Handwritten Optical Character Recognition for East Syriac Script Through Creating KHAMIS Dataset. Department of Computer Science.

Muraoka, T. (2005). Classical Syriac: A Basic Grammar with a Chrestomathy, volume 19 of Porta Linguarum Orientalium, Neue Serie. Harrassowitz Verlag, Wiesbaden, 2nd, revised edition.

Naaijer, M., Sikkel, C., Coeckelbergs, M., Attema, J., and Van Peursen, W. T. (2023). A Transformerbased parser for Syriac morphology. In Adam Anderson, et al., editors, Proceedings of the Ancient Language Processing Workshop, pages 23–29, Varna, Bulgaria, September. INCOMA Ltd., Shoumen, Bulgaria.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. (2002). Bleu: a Method for Automatic Evaluation of Machine Translation. In Pierre Isabelle, et al., editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July. Association for Computational Linguistics.

Sennrich, R., Haddow, B., and Birch, A. (2016). Neural Machine Translation of Rare Words with Subword Units. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pages 1715–1725.

Tafa, T., Hashim, S., Othman, M., Alhussian, H., Nasser, M., Jadid Abdulkadir, S., Huspi, S., Adeyemo, S., and Bena, Y. (2025). Machine Translation Performance for Low-Resource Languages: A Systematic Literature Review. IEEE Access, PP:1–1, 01.

Tedla, Y. and Yamamoto, K. (2016). The Efect of Shallow Segmentation on English-Tigrinya Statistical Machine Translation. In 2016 International Conference on Asian Language Processing (IALP), page 20. IEEE, November.

van Peursen, W., Naaijer, M., Sikkel, C., Coeckelbergs, M., and Attema, J. (2023). A Transformerbased Parser for Syriac Morphology. In Adam Anderson, et al., editors, Proceedings of the Ancient Language Processing Workshop, pages 23–29. Incoma Ltd. Publisher Copyright: © RANLP-ALP 2023 - Proceedings of the Ancient Language Processing Workshop, associated with 14th International Conference on Recent Advances in Natural Language Processing.; 1st Workshop on Ancient Language Processing, ALP 2023 ; Conference date: 08-09-2023.

Velayuthan, M., Jayakody, D., De Silva, N., Fernando, A., and Ranathunga, S. (2024). Back to the Stats: Rescuing Low Resource Neural Machine Translation with Statistical Methods. In Barry Haddow, et al., editors, Proceedings of the Ninth Conference on Machine Translation, pages 901–907, Miami, Florida, USA, November. Association for Computational Linguistics.