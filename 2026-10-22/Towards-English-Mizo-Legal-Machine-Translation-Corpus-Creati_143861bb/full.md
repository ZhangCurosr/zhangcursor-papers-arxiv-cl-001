![](images/0699ef38053dae5ecb4c237e39185423659777e0c88d3b8d1d1a3e6e9e99a88a.jpg)

# Towards English–Mizo Legal Machine Translation: Corpus Creation, Analysis, and Benchmarks

Telem Joyson Singh <sup>ID</sup> <sup>∗∗</sup>, Sanabam Boyshana, Sanasam Ranbir Singh <sup>ID</sup> , Priyankoo Sarmah <sup>ID</sup>

Indian Institute of Technology Guwahati, Assam, India

tjoyson@iitg.ac.in, boyshanasanabam@gmail.com, ranbir@iitg.ac.in, priyankoo@iitg.ac.in

## Abstract

Indian court proceedings are held in English, so speakers of languages with limited translation technology have limited access to them. We present the first English–Mizo parallel dataset of Indian court judgments. It contains 13,168 sentence pairs from the Gauhati High Court and the Supreme Court of India. Professional native Mizo translators translated complete documents, sentence pairs were manually aligned, and two independent reviewers checked the results. Our study finds that English words are borrowed in a very selective way: official and legal terms are almost always kept in English, but words for everyday things like actions, situations, and results are usually expressed in native Mizo. Mizo diacritics are also used inconsistently across the corpus. When we fine-tune the NLLB-200 1.3B model on our training data, it scores 27.47 BLEU on a balanced legal test set of 1,000 sentence pairs—an improvement of 7.16 BLEU over the base model. However, it still uses English words about three times more often than a human translator does.

Index Terms: parallel corpus, legal translation, machine translation, mizo, low-resource language

## 1. Introduction

Access to justice depends on understanding the language of the courtroom. In India, Article 348 of the Constitution requires that proceedings in the Supreme Court and the High Courts be conducted in English, so the official text of judgments is in English even when the parties cannot read it. The Supreme Court and the e-Courts project have launched translation initiatives, but these target the languages listed in the Eighth Schedule [1, 2]. Languages outside the Schedule receive little institutional support, and their communities are left to rely on informal, unverified oral summaries of the decisions that affect them.

Mizo (also called Lushai) is one such language. It is a Tibeto-Burman language spoken by about 830,000 people (2011 Census of India [3]) and is an official language of Mizoram alongside English. Because it is not in the Eighth Schedule, it is usually excluded from the major Indic datasets and models [4, 5]. This gap is especially serious in the legal domain. Mizoram is under the Gauhati High Court, where the rulings are written only in English. As a result, Mizo speakers have almost no tools to help them read and understand court decisions themselves.

The main challenge is the domain, not just the volume of data. Most existing Mizo parallel texts are from religious writings or informal web content, which are very different from legal writing in length, sentence structure, citation style, and vocabulary. Models developed on such data transfer poorly to legal text, and until now there has been no legal test set to measure how large that gap is. As a result, progress has been less constrained by modeling capacity than by the lack of a reliable evaluation resource.

We address this with the first parallel corpus of Indian legal judgments for Mizo: 13,168 sentence pairs, 286,590 English and 291,751 Mizo tokens, drawn from civil, criminal and miscellaneous cases of the Gauhati High Court and the Supreme Court of India. Professional native Mizo translators with degrees in law, English, or Mizo translated entire documents, so that references and terminology remain consistent within a judgment. We then aligned the texts at the sentence level using embedding-based candidate generation followed by manual correction, and validated the result with two independent annotators (Cohen’s κ = 0.90).

Beyond releasing the data, we analyze its linguistic properties. Because Mizo has no indigenous common-law vocabulary, translating Indian legal English into Mizo involves systematic borrowing: 6,612 word types occur on both sides of the corpus, and 55.30 % of Mizo tokens are strings that also appear in the English vocabulary. The borrowing is selective. Names of institutions and statutory instruments—court, FIR, IPC, and affidavit—are kept in English almost all the time (close to 100 %). In contrast, words referring to people, legal processes, and outcomes—accused (8.5 %) and conviction (5.0 %)—are usually replaced with native Mizo words. We link this pattern to referential rigidity: the English form is kept exactly in those cases where replacing it could create ambiguity. We also observe substantial orthographic variation on the Mizo side. Circumflex tone marking appears only in 2.70 % of Mizo segments, 159 types are attested both with and without it, and the t/t contrast is largely unrealized–variation that affects tokenization and comparability between systems.

Finally, we create baseline systems by fine-tuning the NLLB-200 1.3B model [6]—one of the rare open-source, largescale translation models with explicit Mizo—using the 11,522- pair training set. On a stratified 1,000-pair held-out legal test , the model scores 27.47 BLEU, 50.39 chrF++, and 65.68 TER. Compared to the original, non-adapted model, this is an improvement of +7.16 BLEU, +7.83 chrF++, and a reduction of −12.13 TER. The error analysis shows that most of the remaining mistakes happen when the system has to decide whether to borrow English words. For legal terms, it keeps the English word in 21.7 % of the cases where the human translation uses a native Mizo word, but only 8.3 % the other way around. In other words, it borrows English terms about three times more often than the human translator.

Our contributions are as follows.

• We release the first English–Mizo parallel corpus in the legal domain: 13,168 professionally translated, manually aligned, and independently validated sentence pairs from Indian court judgments.

• We analyze the corpus quantitatively, including a retentionrate study of legal terminology that shows a systematic borrowing pattern, and an audit of Mizo orthography that identifies diacritic normalization as an open preprocessing question for Mizo NLP.

• We present both fine-tuned models and baseline systems on a stratified held-out legal test set, including an error analysis by stratum, thereby providing reference points for future research on Mizo legal translation.

## 2. Background

## 2.1. The Mizo language

Mizo is spoken mainly in Mizoram, with communities in neighboring northeast Indian states, Myanmar’s Chin State, and Bangladesh’s Chittagong Hill Tracts [7, 8]. The standard written variety is based on the Duhlian dialect.

Three properties matter for our corpus. First, Mizo is agglutinative, with extensive particles and derivational affixation, so a root appears on many surfaces; this is shown in the vocabulary and hapax counts in Section 4. Second, Mizo is tonal, but this tone is only partly written. Third, its writing system is based on the Latin alphabet and was created by Christian missionaries in the late nineteenth century. It includes two features that English does not have: the special letter t<sub>.</sub> (LATIN SMALL LETTER T WITH A DOT BELOW), which represents a different sound from plain t, and vowels with a circumflex accent that show both tone and vowel length. Both are optional in everyday writing, so running text stays often under-specified relative to the prescriptive standard; we quantify this in Section 4.3.

That missionary history similarly shapes the corpus. For decades, Bible translation was the main literate use of written Mizo, so religious texts dominate what is digitally available, while technical and administrative writing is scarce. Commonlaw concepts entered Mizo through English-medium administration and were never lexicalized as a coherent native system, so translating judicial prose depends on the borrowing behavior analyzed in Section 4.2.

## 2.2. Language technology for Mizo

Indian-language NLP has advanced mainly through corpora and models built for twenty-two scheduled languages [4, 5, 9]. Mizo is not a scheduled language, so most of these resources exclude it.

Mizo has entered modern multilingual NLP largely through massively multilingual projects. It appears as lus\_Latn in the NLLB-200 models and the FLORES benchmark [6, 10], which makes NLLB-200 a natural starting point and motivates our choice of base model in Section 5. Coverage does not guaranty quality, however: the Mizo training data in such systems is largely religious and crawled web text, and the evaluation sets are general-domain. Mizo-focused work has focused on building general-purpose parallel data and supervised and unsupervised NMT for the pair [11, 12, 13]; to our knowledge, no resource in the legal-domain has been available.

## 2.3. Legal-domain translation and corpora

Legal text remains a hard case for machine translation. Judgments use long, deeply nested sentences, fixed citation and casetitle patterns, formal honorific language, and terms whose nearsynonyms are not interchangeable: translating accused as convict, or petitioner as appellant changes the legal meaning. Here, correct terminology matters more than fluent language, and mistakes can harm the litigant.

In Europe, large multilingual legal corpora produced by institutional translation– parliamentary proceedings, statutes, and case law [14, 15, 16]—have supported extensive work on legal MT and terminology. In India, legal NLP has grown to include judgment summarisation, rhetorical role labeling, as well as pretrained legal language models, supported by parallel datasets for several Indian languages [17, 18, 19]. These resources stop at the scheduled languages, so Mizo is absent. Our corpus extends this line of work to a non-scheduled language.

## 3. Corpus Construction

The corpus is made up of English-language judgments from the Gauhati High Court and the Supreme Court of India, including civil, criminal, and other types of cases. All documents come from official court records that are already publicly available, so we did not apply any extra anonymization beyond how the judgments are originally published.

Document translation. The English judgments were translated into Mizo by a team of professional translators, all native Mizo speakers holding degrees in law, English, or Mizo. Translation was performed at the document level rather than sentence by sentence, so that translators could use the larger context to resolve ambiguity and keep terminology consistent within a judgment. Then each translated document was peer-reviewed by a second translator.

Sentence alignment. After translation and review, we aligned the documents at sentence level. Both sides were first segmented into sentences, using spaCy [20] for English and NLTK [21] for Mizo. We then used an aligner that relies on multilingual sentence embeddings to generate possible sentence matches, and we ranked these candidates using cosine similarity. Since legal sentences are often long and complicated, we did not rely on automation alone: human annotators went through the suggested segments, fixed one-to-many and manyto-one mappings and made sure each matched pair had the same meaning.

Validation. Two native Mizo speakers independently reviewed every pair to confirm that the two sentences were translations of each other. Inter-annotator agreement was high (Cohen’s κ = 0.90 [22]). Disagreements were referred to a third translator and resolved by a majority vote.

## 4. Corpus Analysis

This section describes the corpus in three dimensions: basic word statistics, how legal terms are borrowed and kept, and how consistently Mizo spelling is used. For all of this, we count only alphabetic tokens using Unicode-aware tokenization, and we exclude punctuation and standalone numbers.

## 4.1. Descriptive and lexical statistics

Table 1 summarizes the corpus at the segment and token level. It contains 13,168 sentence pairs, 286,590 English, and 291,751 Mizo running tokens.

With almost the same number of tokens, the Mizo side has 16 % more types than the English side, and 35.45 % of its types occur only once, compared to 24.00 % in English. Two factors contribute. First, Mizo derivational and particle morphology produces a larger inventory of distinct surface forms. Second, the corpus has a long tail of Mizo personal and place names that appear in cause titles and are transferred unchanged to both sides. The lower character-per-word figure for Mizo indicates its predominantly CV and CVC syllable structure.

Table 1: Descriptive statistics for the English–Mizo corpus.
<table><tr><td>Statistic</td><td>English</td><td>Mizo</td></tr><tr><td>Sentence pairs</td><td>13,168</td><td>13,168</td></tr><tr><td>Running tokens</td><td>286,590</td><td>291,751</td></tr><tr><td>Vocabulary (types)</td><td>10,317</td><td>11,996</td></tr><tr><td>Type-token ratio</td><td>0.036</td><td>0.041</td></tr><tr><td>Hapax legomena</td><td>2,476</td><td>4,253</td></tr><tr><td>as % of types</td><td>24.00</td><td>35.45</td></tr><tr><td>Characters</td><td>1,659,854</td><td>1,573,253</td></tr><tr><td>Characters per word</td><td>4.66</td><td>4.27</td></tr><tr><td>Mean sentence length</td><td>21.76</td><td>22.16</td></tr><tr><td>Median sentence length</td><td>16.0</td><td>17.0</td></tr><tr><td>Standard deviation</td><td>17.44</td><td>17.09</td></tr><tr><td>5th–95th percentile</td><td>5-56</td><td>5-55</td></tr><tr><td>Maximum sentence length</td><td>223</td><td>186</td></tr></table>

Table 2: Cumulative token coverage by vocabulary rank (%).
<table><tr><td>Top-k types</td><td>English</td><td>Mizo</td></tr><tr><td>100</td><td>52.73</td><td>51.55</td></tr><tr><td>500</td><td>73.98</td><td>74.96</td></tr><tr><td>1,000</td><td>82.09</td><td>83.11</td></tr><tr><td>5,000</td><td>96.69</td><td>96.34</td></tr></table>

![](images/c3b954a82e1ae9be68764c95c7ba93a6d2bbb5d66993f142d41236d0f4722e70.jpg)  
Figure 1: Zipf distribution of token frequencies.

Table 2 reports cumulative token coverage by frequency rank. Beyond the first hundred types, the two languages are closely matched, with Mizo slightly lower in rank 5,000, consistent with its heavier type tail. Figures 1 and 2 show that both sides follow the expected Zipfian frequency distribution and Heaps’ law vocabulary growth, with a higher Heaps exponent for Mizo $( \beta = 0 . 5 7 6 ~ \mathrm { v s } . 0 . 5 0 6 )$ . Mizo vocabulary keeps rising sharply with no clear plateau at 13,168 sentence pairs, indicating the legal corpus still lacks full lexical coverage and needs more data.

![](images/50ac8b05e013b457faaf6af2bce3c9bdb0274e02da56197d8707d45cebd62685.jpg)  
Figure 2: Vocabulary growth (Heaps’ law).

Table 3: Corpus-level lexical overlap (English–Mizo).
<table><tr><td>Measure</td><td>Value</td></tr><tr><td>Shared word types</td><td>6,612</td></tr><tr><td>Mizo tokens also in English vocab. Mean token-level Jaccard (per-pair)</td><td>55.30 % 0.4315</td></tr></table>

## 4.2. Lexical borrowing and terminology retention

Mizo doesn’t have its own traditional legal terms, so when Indian legal English is translated into Mizo, new words almost always have to be borrowed. This subsection measures how often that happens, and we see this borrowing as the most distinctive language feature in the whole set of the corpus.

The aggregate borrowing is substantial (Table 3): 6,612 word types occur on both sides, 55.30 % of Mizo running tokens are strings that also appear in the English vocabulary, and the mean Jaccard token-level overlap per-pair is 0.4315.

To separate selective borrowing from wholesale copying, we compute a retention rate for a hand-selected inventory of legal terms: the number of Mizo segments containing the term divided by the number of English segments containing it. Values near 100 % indicate that the English form is carried over; low values indicate that the concept is lexicalized natively in Mizo. The results are in Table 4.

The distribution is far from uniform, and the pattern is interpretable. Terms naming institutions, statutory instruments and fixed procedural labels—court, section, FIR, IPC, CrPC, affidavit, magistrate, bail—are retained at rates approaching 100 %. Terms naming statuses, acts and inferential outcomes— accused (8.5 %), conviction (5.0 %), judgment (23.6 %), evidence (37.1 %)—are overwhelmingly rendered with native Mizo words.

We see this pattern as measuring how strongly a term is tied to a specific reference. When a term refers to something whose identity is fixed by law or institutional designation and mistranslating it would create referential ambiguity, the English form is preserved. Where a term describes a condition, a process, or a conclusion that can be expressed compositionally, Mizo lexicalizes it: thubuai for a case or dispute, dan for law or act, hriattirna for notice, thuneihna for jurisdiction or authority. The rate for learned (26.3 %) is a revealing minor case: the honorific register typical of Indian legal English is largely not carried in Mizo, so translation here involves register normalization and

Table 4: Retention of English legal terms in the Mizo text. Ret. is the proportion ofMizo segments to English segments that include the term.
<table><tr><td>Term</td><td>En</td><td>Mz</td><td>Ret. (%)</td></tr><tr><td colspan="4">High retention: institutions and instruments</td></tr><tr><td>CrPC</td><td>17</td><td>17</td><td>100.0</td></tr><tr><td>FIR</td><td>46</td><td>46</td><td>100.0</td></tr><tr><td>court</td><td>1,616</td><td>1,616</td><td>100.0</td></tr><tr><td>police</td><td>230</td><td>230</td><td>100.0</td></tr><tr><td>IPC</td><td>47</td><td>47</td><td>100.0</td></tr><tr><td>magistrate</td><td>223</td><td>222</td><td>99.6</td></tr><tr><td>section</td><td>664</td><td>653</td><td>98.3</td></tr><tr><td>respondent</td><td>606</td><td>574</td><td>94.7</td></tr><tr><td>tribunal</td><td>16</td><td>15</td><td>93.8</td></tr><tr><td>affidavit</td><td>185</td><td>173</td><td>93.5</td></tr><tr><td>bail</td><td>219</td><td>201</td><td>91.8</td></tr><tr><td>act</td><td>692</td><td>563</td><td>81.4</td></tr><tr><td>advocate</td><td>396</td><td>309</td><td>78.0</td></tr><tr><td>Intermediate</td><td></td><td></td><td></td></tr><tr><td>appeal</td><td>217</td><td>162</td><td>74.7</td></tr><tr><td>appellant</td><td>285</td><td>211</td><td>74.0</td></tr><tr><td>witness</td><td>80</td><td>54</td><td>67.5</td></tr><tr><td>constitution</td><td>306</td><td>205</td><td>67.0</td></tr><tr><td>petitioner</td><td>902</td><td>597</td><td>66.2</td></tr><tr><td>counsel</td><td>664</td><td>323</td><td>48.6</td></tr><tr><td>writ</td><td>232</td><td>112</td><td>48.3</td></tr><tr><td colspan="4">Low retention: statuses and reasoning</td></tr><tr><td>order</td><td>514</td><td>201</td><td>39.1</td></tr><tr><td>suit</td><td>46</td><td>18</td><td>39.1</td></tr><tr><td>evidence</td><td>197</td><td>73</td><td>37.1</td></tr><tr><td>jurisdiction</td><td>86</td><td>24</td><td>27.9</td></tr><tr><td>learned</td><td>1,154</td><td>303</td><td>26.3</td></tr><tr><td>judgment</td><td>280</td><td>66</td><td>23.6</td></tr><tr><td>sentence</td><td>55</td><td>10</td><td>18.2</td></tr><tr><td>decree</td><td>22</td><td>4</td><td>18.2</td></tr><tr><td>accused</td><td>801</td><td>68</td><td>8.5</td></tr><tr><td>conviction</td><td>40</td><td>2</td><td>5.0</td></tr></table>

lexical substitution.

## 4.3. Mizo orthographic consistency

Written Mizo adds two features to the Latin alphabet: the letter t<sub>.</sub>, which contrasts with t, and circumflex vowels that mark tone and length. We assess how consistently the corpus uses both. Table 5 shows that both markers are rare. Circumflexes appear only in 2.70 % of the Mizo segments and 0.22 % of tokens, and 159 types occur both with and without them (Table 6). The letter t<sub>.</sub> is equally rare (282 tokens in 52 types), and the counts in Table 7 show that the t/t<sub>.</sub> contrast is largely not maintained.

We regard this as an open problem for Mizo NLP rather than a defect of the corpus. It is not yet clear whether models should be trained on text that keeps diacritics or on text where they are stripped, and the choice affects vocabulary size, segmentation cost, and the comparability of published results. We therefore release the corpus unnormalised, so that either convention can be applied downstream.

Table 5: Inventory of Mizo-specific orthographic characters on the Mizo side.
<table><tr><td>Character</td><td>Name</td><td>Count</td></tr><tr><td>â</td><td>a with circumflex</td><td>294</td></tr><tr><td>t.é</td><td>t with dot below</td><td>282</td></tr><tr><td></td><td>e with circumflex</td><td>171</td></tr><tr><td>û</td><td>u with circumflex</td><td>94</td></tr><tr><td>1</td><td>i with circumflex</td><td>41</td></tr></table>

Table 6: Most frequent Mizo types attested with inconsistent circumflex marking. Frequency is the combined count of both variants.
<table><tr><td>Base form</td><td>Attested variants</td><td>Freq.</td></tr><tr><td>chu</td><td>chu / chû</td><td>5,744</td></tr><tr><td>te</td><td>te / tê</td><td>2,973</td></tr><tr><td>tur</td><td>tur / tûr</td><td>1,858</td></tr><tr><td>dan</td><td>dan / dân</td><td>1,416</td></tr><tr><td>he</td><td>he / hê</td><td>1,257</td></tr><tr><td>awm</td><td>awm / âwm</td><td>1,238</td></tr><tr><td>tak</td><td>tak / tâk</td><td>876</td></tr><tr><td>dilna</td><td>dilna / dîlna</td><td>666</td></tr><tr><td>pek</td><td>pek / pêk</td><td>657</td></tr><tr><td>khan</td><td>khan / khân</td><td>578</td></tr></table>

Table 7: Realisation of the t/t<sub>.</sub> contrast for selected lexemes.
<table><tr><td>Plain</td><td>Freq.</td><td>Dotted</td><td>Freq.</td></tr><tr><td>tan</td><td>277</td><td>an</td><td>12</td></tr><tr><td>tha</td><td>111</td><td>tha</td><td>27</td></tr><tr><td>tul</td><td>109</td><td>ul</td><td>16</td></tr><tr><td>tawng</td><td>13</td><td>awng</td><td>4</td></tr></table>

## 5. Experimental Setup

## 5.1. Training data

The training split contains 11,522 sentence pairs from the corpus described in Section 4, with 245,838 English and 248,039 Mizo tokens (Table 8). It is formed after the set of benchmark tests is removed, and any sentence with more than 90 % words overlaps with a test sentence. As in the corpus as a whole, the Mizo side has the larger vocabulary (13,205 vs. 11,929 types) and the higher TTR (0.053 vs. 0.049), consistent with the morphology and orthographic variation reported in Section 4.

## 5.2. Test data

The benchmark test set contains 1,000 sentence pairs, with 21,990 English tokens and 23,150 Mizo tokens. Instead of choosing these examples at random, we deliberately focus on the kinds of sentences that make legal translation difficult: those with legal terms (230 pairs), abbreviations and acronyms (220), named entities (200), names of statutes and acts (150), and general legal writing (200); see Figure 3. The relatively high type–token ratio (0.180 in English, 0.159 in Mizo) is mainly due to the smaller size of this split.

Figure 4 shows how much the test vocabulary overlaps with the training data: 82.5 % of English and 79.7 % of Mizo distinct word types in the test set also appear in training, and coverage at the token level is 96.6 % and 96.5 %, respectively. This means that the test set is challenging in terms of unique words—about one in five Mizo word types is new—yet still manageable when reading continuous text.

Table 8: Statistics of the training and test splits.
<table><tr><td>Split</td><td>Lang.</td><td>Sent.</td><td>Tokens</td><td>Types</td><td>TTR</td></tr><tr><td rowspan="2">Train</td><td>English</td><td>11,522</td><td>245,838</td><td>11,929</td><td>0.049</td></tr><tr><td>Mizo</td><td></td><td>248,039</td><td>13,205</td><td>0.053</td></tr><tr><td rowspan="2">Test</td><td>English</td><td>1,000</td><td>21,990</td><td>3,968</td><td>0.180</td></tr><tr><td>Mizo</td><td></td><td>23,150</td><td>3,669</td><td>0.159</td></tr></table>

![](images/b7b01367e934748f1a6325e591dc48b0ba30d545b48f3bef336d2208938afe53.jpg)  
Figure 3: Sentence types in the 1,000-pair benchmark test set.

![](images/439e1d3405fa300423a95b38c79583f926d6dc8e227e7b977eacd318f2b56c41.jpg)  
Figure 4: Test set coverage by training set vocabulary.

## 5.3. Model

We use NLLB-200 [6], a Meta AI model that covers 200 languages, and trained on substantial mined parallel datasets. It is the natural choice here because it is one of the few large-scale open-source translation models with explicit support from Mizo (lus\_Latn), and serves as a strong baseline for low-resource translation. We evaluate both off the shelf and after fine-tuning and refer to the two systems as NLLB-200 1.3B and NLLB-FT.

## 5.4. Fine-tuning and implementation

Due to computational constraints, we use the 1.3 billion parameter variant of NLLB-200. Fine-tuning was performed with the Hugging Face transformer library [23] on a single NVIDIA A100 GPU, using AdamW [24] with a learning rate of 5×10<sup>−5</sup> and a linear warm-up schedule. We train for 5 epochs with a batch size of 4 and leave all other hyperparameters at their library defaults. We translate in the English→Mizo direction throughout.

Table 9: Machine translation results on the 1,000-pair benchmark.
<table><tr><td>System</td><td>BLEU↑</td><td>chrF++↑</td><td>TER↓</td></tr><tr><td>NLLB-200 1.3B</td><td>20.31</td><td>42.56</td><td>77.81</td></tr><tr><td>NLLB-FT</td><td>27.47</td><td>50.39</td><td>65.68</td></tr></table>

Table 10: NLLB-FTperformance by test-set stratum.
<table><tr><td>Stratum</td><td>n</td><td>BLEU↑</td><td>chrF++ ↑</td><td>TER↓</td></tr><tr><td>Legal terms</td><td>230</td><td>19.71</td><td>46.43</td><td>69.70</td></tr><tr><td>Abbreviations</td><td>220</td><td>34.05</td><td>53.58</td><td>63.73</td></tr><tr><td>Named entities</td><td>200</td><td>26.33</td><td>53.08</td><td>62.42</td></tr><tr><td>Act names</td><td>150</td><td>32.18</td><td>52.60</td><td>63.34</td></tr><tr><td>Others</td><td>200</td><td>21.14</td><td>46.27</td><td>68.38</td></tr><tr><td>Overall</td><td>1,000</td><td>27.47</td><td>50.39</td><td>65.68</td></tr></table>

## 5.5. Metrics

We report three complementary automatic metrics.

• BLEU: case-sensitive sacreBLEU [25], which fixes tokenisation and makes scores reproducible. Measures n-gram precision against the reference.

• chrF++: a character-level n-gram F-score [26], well suited to a morphologically rich language such as Mizo and robust to tokenization differences.

• TER: translation edit rate [27], the minimum number of insertions, deletions, substitutions, and shifts needed to convert the output to reference, normalized by reference length. Lower is better; it approximates the post-editing effort.

## 6. Results

Table 9 shows the overall results. When we fine-tune the model on legal texts from the target domain, all scores improve compared to the off-the-shelf model: BLEU goes from 20.31 to 27.47 (+7.16), chrF++ from 42.56 to 50.39 (+7.83), and TER drops from 77.81 to 65.68 (−12.13). Since we see gains in both a surface-based metric and an edit-based metric, the improvements are due to better wording and sentence structure, not quirks of a single measure. The size of this improvement also makes it clear that general-domain multilingual pre-training by itself is not sufficient for Mizo legal translation.

## 6.1. Analysis by stratum

Overall scores hide patterns that are specific to legal texts, so we take a closer look at NLLB-FT using the stratified benchmark. Throughout this analysis, we compare the system’s output to the actual choices made by the human translator in the reference, instead of measuring it against some abstract idea of “correctness.” This approach helps us tell apart real mistakes from acceptable differences in how words or terms are borrowed.

Table 10 shows that BLEU scores vary by more than 14 points across different strata. Scores are highest when the source text includes content that can be copied directly into the translation—such as abbreviations (34.05) and statute names (32.18)—and lowest for legal terms (19.71) and general prose (21.14), where the model has to pick an appropriate Mizo word instead of reusing the English one. In other words, the system is better at keeping English words than at replacing them with Mizo, and the next section shows that this single pattern explains most of the errors we see.

Table 11: Legal-terminology errors in the NLLB-FT output on the 230-segment legal-term stratum. Percentages are of segments in the stratum; n counts affected segments.
<table><tr><td>Error type</td><td>n (%)</td><td>Typical case</td></tr><tr><td>Over-retention of</td><td>50 (21.7)</td><td>learned, evidence</td></tr><tr><td>English form</td><td></td><td>kept verbatim</td></tr><tr><td>Spurious nativisation</td><td>19 (8.3)</td><td>affidavit → native</td></tr><tr><td>Variant equivalent</td><td>7 (3.0)</td><td>order: thupek /</td></tr><tr><td>(not an error) Meaning-shifting</td><td></td><td>thuchhuah</td></tr><tr><td>substitution</td><td>5 (2.2)</td><td>accused →</td></tr><tr><td>Untranslated copy</td><td>4 (1.7)</td><td>pawikhawihtu segment left in English</td></tr><tr><td>Truncation / omission</td><td>2 (0.9)</td><td>clause dropped</td></tr></table>

## 7. Error Analysis

To see where NLLB-FT still fails, we inspect its output on the benchmark. The ordering in Table 10 is partly an artifact: strings that can be copied are copied, and copying is rewarded by surface metrics even when the resulting segment is not a translation. We therefore analyze the three critical strata separately, combining automatic extraction of term, entity, and acronym mentions with manual inspection of the flagged segments.

## 7.1. Legal terminology

Section 4.2 showed that professional translators borrow selectively. The dominant failure of NLLB-FT is that it does not reproduce this pattern: it keeps English words where the translator uses a Mizo one. Across the 230 legal-term segments the model matches the reference choice in 150 (65.2 %). Where the two differ, over-retention dominates: in 50 segments (21.7 %) the model copies the English term although the reference nativizes it, against only 19 segments (8.3 %) in the opposite direction (Table 11). The system therefore borrows roughly three times more often than the translator. A smaller but more serious class is meaning-shifting substitution (5 segments), where the model picks a Mizo word that changes the legal meaning, for example rendering accused as pawikhawihtu (‘troublemaker’).

## 7.2. Named entities

The model usually handles named entities well. Out of the 842 entity mentions that the reference version copies word for word, the model reproduces 814 of them exactly (96.7 %). This means that party names, place names, case numbers, and dates are mostly reliable (see Table 12).

The 15 corrupted forms are systematic rather than random. Plural institutional names are silently singularized (District Councils → District Council), which can change the body to which it being referred to; statute titles are rewritten in canonical form (Criminal Procedure Code, 1973 → Code of Criminal Procedure, 1973), which is bibliographically correct but no longer the form used by the court; and low-frequency names drift orthographically (Practices → Practises).

Table 12: Named-entity behaviour in the NLLB-FT output, over the 842 entity mentions that the reference transfers verbatim.
<table><tr><td>Behaviour</td><td>n (%)</td><td>Example</td></tr><tr><td>Transferred correctly</td><td>814 (96.7)</td><td>Jogen Kalita</td></tr><tr><td>Corrupted form</td><td>15 (1.8)</td><td>Practices → Practises</td></tr><tr><td>Omitted</td><td>13 (1.5)</td><td>entity dropped</td></tr></table>

Table 13: Abbreviation and acronym behaviour in the NLLB-FT output, over the 208 acronym mentions that the reference retains.

<table><tr><td>Behaviour</td><td>n (%)</td><td>Example</td></tr><tr><td>Acronym preserved</td><td>203 (97.6)</td><td>ND&amp;PS, IPC</td></tr><tr><td>Lost or paraphrased</td><td>4 (1.9)</td><td>P.W.1 →</td></tr><tr><td>Case / punct. drift</td><td>1 (0.5)</td><td>Thuhretu-1 na  $C r P C \to C r P c$ </td></tr></table>

## 7.3. Abbreviations and acronyms

The acronym strings are the most reliable part of the output (Table 13). Of the 208 acronym mentions that each reference includes, 203 (97.6 %) are copied exactly. Four are either dropped or loosely translated (for example, P.W.1 becomes Thuhretu-1 na), and one has a minor change in the letter case (Cr.PC $ C r P c )$ . Differences like ND&PS versus NDPS, or Cr.P.C. versus Cr.PC and CrPC, already appear in both the source and the reference, and the system preserves them as they are. This matches the broader corpus pattern, where legal abbreviations such as IPC (100.0 %) and FIR (100.0 %) are kept on the Mizo side near their original rates.

## 8. Conclusion

We introduce the first English–Mizo parallel corpus in the legal domain: 13,168 sentence pairs taken from Indian court judgments. Professional native Mizo translators worked at the document level, and the sentence pairs were then manually aligned and independently validated. Two findings stand out. First, when Indian legal English is translated into Mizo, there is a clear pattern of borrowing: the names of institutions and laws are kept almost unchanged, while descriptions of events, processes, and outcomes are translated using native Mizo words. We interpret this pattern as a way to keep legal references very stable. Second, real-world Mizo writing does not use diacritics consistently. As a result, deciding whether and how to normalize diacritics is still an open preprocessing option, not a settled one.

Our baseline results show that fine-tuning on in-domain data really helps: the NLLB-200 1.3B model improves by +7.16 BLEU on a carefully sampled legal test set. However, our error analysis shows that the remaining problems are exactly in the areas where English borrowing is most likely. The fine-tuned system copies English words in the translation about three times more often than a human translator does.

This suggests two directions. On the modeling side, terminology-aware objectives or constrained decoding informed by a retention lexicon could help a system learn when to borrow rather than borrowing by default. On the evaluation side, terminology-sensitive metrics are needed, since BLEU and chrF++ credit copying.

## 9. Acknowledgements

We thank the professional translators and annotators who built this resource. The funding for data collection in this work was provided through the Mizo Language Model Training Hackathon, organized by the Digital India BHASHINI Division (Government of India) and the Government of Mizoram.

## 10. Generative AI Use Disclosure

Generative AI tools were used only for the language editing and polishing of the manuscript. They were not used to produce any part of the research content, the corpus, or the experiments.

## 11. References

[1] e-Committee, Supreme Court of India, “Vision document for phase III of the eCourts project,” https://ecommitteesci.gov.in/ document/vision-document-for-phase-iii-of-ecourts-project/, 2023, prepared by DAKSH (lead), Agami and Vidhi Centre for Legal Policy in collaboration with the e-Committee. Accessed 2026.

[2] Supreme Court of India, “Annual report 2019–2020,” https:// www.sci.gov.in/, 2020, describes the Supreme Court Vidhik Anuvaad Software (SUVAS), launched November 2019.

[3] Office of the Registrar General and Census Commissioner, India, “Census of india 2011: C-16 population by mother tongue, india,” https://censusindia.gov.in/nada/index.php/catalog/10191, 2011.

[4] G. Ramesh, S. Doddapaneni, A. Bheemaraj, M. Jobanputra, R. AK, A. Sharma, S. Sahoo, H. Diddee, M. J, D. Kakwani, N. Kumar, A. Pradeep, S. Nagaraj, K. Deepak, V. Raghavan, A. Kunchukuttan, P. Kumar, and M. S. Khapra, “Samanantar: The largest publicly available parallel corpora collection for 11 Indic languages,” Transactions of the Association for Computational Linguistics, vol. 10, pp. 145–162, 2022.

[5] J. Gala, P. A. Chitale, R. AK, V. Gumma, S. Doddapaneni, A. Kumar, J. Nawale, A. Sujatha, R. Puduppully, V. Raghavan, P. Kumar, M. M. Khapra, R. Dabre, and A. Kunchukuttan, “Indic-Trans2: Towards high-quality and accessible machine translation models for all 22 scheduled Indian languages,” Transactions on Machine Learning Research, 2023, arXiv:2305.16307.

[6] NLLB Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. Mejia Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. Spruit, C. Tran, P. Andrews, N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzmán, P. Koehn, A. Mourachko, C. Ropers, S. Saleem, H. Schwenk, and J. Wang, “No language left behind: Scaling human-centered machine translation,” arXiv preprint arXiv:2207.04672, 2022.

[7] L. Chhangte, “The grammar of simple clauses in Mizo,” in Papers in Southeast Asian Linguistics No. 11: Southeast Asian Syntax, ser. Pacific Linguistics A-77, D. Bradley, Ed. Canberra: The Australian National University, 1989, pp. 93–174.

[8] ——, “Mizo syntax,” Ph.D. dissertation, University of Oregon, Eugene, OR, USA, 1993.

[9] T. J. Singh, S. R. Singh, and P. Sarmah, “Can big models help diverse languages? investigating large pretrained multilingual models for machine translation of Indian languages,” in Proceedings of the 20th International Conference on Natural Language Processing (ICON). Goa University, Goa, India: NLP Association of India (NLPAI), Dec. 2023, pp. 663–669. [Online]. Available: https://aclanthology.org/2023.icon-1.66/

[10] NLLB Team, M. R. Costa-jussà, J. Cross, O. Çelebi, M. Elbayad, K. Heafield, K. Heffernan, E. Kalbassi, J. Lam, D. Licht, J. Maillard, A. Sun, S. Wang, G. Wenzek, A. Youngblood, B. Akula, L. Barrault, G. Mejia Gonzalez, P. Hansanti, J. Hoffman, S. Jarrett, K. R. Sadagopan, D. Rowe, S. Spruit, C. Tran, P. Andrews,

N. F. Ayan, S. Bhosale, S. Edunov, A. Fan, C. Gao, V. Goswami, F. Guzmán, P. Koehn, A. Mourachko, C. Ropers, S. Saleem, H. Schwenk, and J. Wang, “Scaling neural machine translation to 200 languages,” Nature, vol. 630, no. 8018, pp. 841–846, 2024.

[11] C. Lalrempuii, B. Soni, and P. Pakray, “An improved English-to-Mizo neural machine translation,” ACM Transactions on Asian and Low-Resource Language Information Processing, vol. 20, no. 4, 2021.

[12] V. Khenglawt, S. R. Laskar, S. Pal, P. Pakray, and A. K. Khan, “Language resource building and English-to-Mizo neural machine translation encountering tonal words,” in Proceedings ofthe WILDRE-6 Workshop within the 13th Language Resources and Evaluation Conference. Marseille, France: European Language Resources Association, 2022, pp. 48–54.

[13] C. Lalrempuii and B. Soni, “Investigating unsupervised neural machine translation for low-resource language pair English-Mizo via lexically enhanced pre-trained language models,” ACM Transactions on Asian and Low-Resource Language Information Processing, vol. 22, no. 8, pp. 1–18, 2023.

[14] P. Koehn, “Europarl: A parallel corpus for statistical machine translation,” in Proceedings of Machine Translation Summit X, Phuket, Thailand, 2005, pp. 79–86.

[15] R. Steinberger, B. Pouliquen, A. Widiger, C. Ignat, T. Erjavec, D. Tufi¸s, and D. Varga, “The JRC-Acquis: A multilingual aligned parallel corpus with 20+ languages,” in Proceedings of the 5th International Conference on Language Resources and Evaluation (LREC), Genoa, Italy, 2006, pp. 2142–2147.

[16] R. Steinberger, A. Eisele, S. Klocek, S. Pilos, and P. Schlüter, “DGT-TM: A freely available translation memory in 22 languages,” in Proceedings of the 8th International Conference on Language Resources and Evaluation (LREC), Istanbul, Turkey, 2012, pp. 454–459.

[17] T. J. Singh, H. Baruah, S. Ranbir Singh, A. Talukdar, N. Shahnaz, O. J. Singh, P. Sarmah, P. K. Dutta, S. Nandi, and P. Duara, “AssamLegalTrans: A parallel corpus, benchmark and analysis for English-Assamese machine translation of legal judgments,” in Proceedings ofthe Fifteenth Language Resources and Evaluation Conference. Palma de Mallorca, Spain: ELRA Language Resource Association, May 2026, pp. 4921–4930. [Online]. Available: https://aclanthology.org/2026.lrec-1.386/

[18] V. Parikh, V. Mathur, P. Mehta, N. Mittal, and P. Majumder, “Law-Sum: A weakly supervised approach for Indian legal document summarization,” arXiv preprint arXiv:2110.01188, 2021.

[19] S. Mahapatra, D. Datta, S. Soni, A. Goswami, and S. Ghosh, “MILPaC: A novel benchmark for evaluating translation of legal text to Indian languages,” ACM Transactions on Asian and Low-Resource Language Information Processing, vol. 24, no. 8, pp. 1–30, 2025.

[20] M. Honnibal, I. Montani, S. Van Landeghem, and A. Boyd, “spaCy: Industrial-strength natural language processing in Python,” https://spacy.io, 2020.

[21] S. Bird, E. Klein, and E. Loper, Natural Language Processing with Python. Sebastopol, CA, USA: O’Reilly Media, 2009.

[22] J. Cohen, “A coefficient of agreement for nominal scales,” Educational and Psychological Measurement, vol. 20, no. 1, pp. 37–46, 1960.

[23] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. Le Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “Transformers: State-of-the-art natural language processing,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations. Online: Association for Computational Linguistics, 2020, pp. 38–45.

[24] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in Proceedings ofthe International Conference on Learning Representations (ICLR), New Orleans, LA, USA, 2019.

[25] M. Post, “A call for clarity in reporting BLEU scores,” in Proceedings of the Third Conference on Machine Translation: Research Papers (WMT). Brussels, Belgium: Association for Computational Linguistics, 2018, pp. 186–191.

[26] M. Popovic, “chrF++: Words helping character n-grams,” in´ Proceedings of the Second Conference on Machine Translation (WMT). Copenhagen, Denmark: Association for Computational Linguistics, 2017, pp. 612–618.

[27] M. Snover, B. Dorr, R. Schwartz, L. Micciulla, and J. Makhoul, “A study of translation edit rate with targeted human annotation,” in Proceedings ofthe 7th Conference ofthe Associationfor Machine Translation in the Americas (AMTA), Cambridge, MA, USA, 2006, pp. 223–231.