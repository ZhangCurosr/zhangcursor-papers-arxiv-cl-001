# AraDetox: A Multi-Dialect Arabic Detoxification Dataset

Mo El-Haj   
VinUniversity, Vietnam   
Lancaster University, UK   
https://elhaj.uk

## Abstract

Arabic harmful-language detection has received considerable attention, yet Arabic text detoxification remains underexplored. We introduce AraDetox, a multi-dialect Arabic detoxification dataset comprising 10,500 harmful socialmedia posts and 84,000 detoxified rewrites generated using GPT-5 and Gemini 2.5 Flash across Modern Standard Arabic, Gulf, Levantine, and Egyptian Arabic. The generated outputs were assessed through human evaluation and automatic analyses of lexical change, semantic preservation, sentiment, and dialectal style. Results show that detoxification is primarily a meaning-preserving rewriting task: substantial lexical and structural reformulation is accompanied by consistently high semantic similarity. Human evaluation confirms successful harmfullanguage removal while largely preserving the original meaning. Dialectal analyses further indicate that the generated variants exhibit measurable stylistic alignment with reference Arabic dialect corpora. Comparison with existing resources highlights two complementary approaches to detoxification: minimal-edit lexical substitution and meaning-preserving reformulation. Our findings demonstrate that large-scale Arabic detoxification resources can be constructed through LLM-assisted generation and human verification. The dataset is publicly available at https://github.com/ ArabicNLP-UK/AraDetox to support future research on Arabic detoxification, safe text generation, and multi-dialect Arabic NLP.

## 1 Introduction

Research on harmful language in Arabic has largely focused on detection, resulting in numerous datasets and models for hate speech, ofensive language, and toxicity classification (Al Mandhari et al., 2024). Comparatively little attention has been paid to text detoxification, where harmful content is rewritten into a safer form while preserving its meaning. Creating detoxification datasets is particularly challenging because it requires annotators to rewrite ofensive, politically sensitive, and often dialectal content while preserving meaning, stance, target, and communicative intent. As a result, manual dataset creation is both costly and time-consuming (Logacheva et al., 2022; Dementieva et al., 2024). Recent advances in large language models (LLMs) provide a promising alternative, enabling large-scale generation of fluent rewrites that remove harmful language while retaining the underlying message. Unlike harmful-language detection, detoxification requires balancing two competing objectives: removing harmful expressions while preserving the original claim, stance, target, and communicative intent. This challenge is particularly pronounced in Arabic social-media discourse, where criticism, sarcasm, political disagreement, and identity-related language are often intertwined with ofensive expressions. As a result, successful detoxification cannot be reduced to simple lexical substitution, but instead requires broader meaning-preserving reformulation.

We introduce AraDetox, a large-scale multidialect Arabic detoxification dataset containing 10,500 harmful social-media posts and 84,000 detoxified rewrites generated using GPT-5 and Gemini 2.5 Flash across four Arabic varieties: Modern Standard Arabic (MSA), Gulf Arabic, Levantine Arabic, and Egyptian Arabic. The dataset was created using an LLM-assisted synthesis pipeline followed by human quality control by native Arabic speakers. We evaluate AraDetox using lexical change, semantic similarity, sentiment analysis, dialectal style analysis, and human assessment. Our findings show that Arabic detoxification is best characterised as a meaning-preserving rewriting task. Across models and dialects, substantial lexical and structural reformulation is accompanied by consistently high semantic similarity. Human evaluation further confirms that harmful language can be removed while largely preserving the original meaning. Beyond introducing a new resource, our work demonstrates that existing Arabic harmful-language datasets can be transformed into large-scale multi-dialect detoxification resources through LLM-assisted generation and human verification. The dataset is publicly available through the AraDetox repository<sup>1</sup>. The main contributions of this work are threefold. First, we introduce a large-scale Arabic detoxification resource containing eight rewrites for each source post across two LLM families and four Arabic varieties. Second, we provide a multi-perspective evaluation covering lexical reformulation, semantic preservation, sentiment, dialectal style, and independent human assessment. Third, we release the dataset to support reproducible research on Arabic detoxification and dialect-aware safe generation.

## 2 Related Work

Research on harmful language in Arabic has focused primarily on detection, resulting in datasets and models for hate speech, ofensive language, and abusive-language classification (Mubarak et al., 2021, 2022; Alghamdi et al., 2024), as well as Arabic language models such as AraBERT (Antoun et al., 2020) and ARBERT/MARBERT (Abdul-Mageed et al., 2021). Comparatively little attention has been paid to text detoxification, which aims to rewrite harmful content into safer forms while preserving meaning. ParaDetox introduced a paralleldata setting based on toxic and non-toxic paraphrase pairs (Logacheva et al., 2022), while subsequent work extended detoxification to multilingual settings through MultiParaDetox (Dementieva et al., 2024, 2025). These resources established detoxification as a generation task distinct from harmfullanguage detection, but existing Arabic resources remain relatively limited in scale and provide only a single detoxified rewrite per source text. For Arabic, the original MultiParaDetox dataset was derived from approximately 2,100 candidate posts and resulted in 1,181 manually annotated source–detoxified pairs after filtering and quality control. For our empirical comparison, we use the publicly available Arabic evaluation split released through Hugging Face<sup>2</sup>, which contains 400 source–detoxified pairs. Recent work has also shown that LLMs can generate high-quality synthetic datasets and annotations, making them increasingly useful for data creation and augmentation (Gilardi et al., 2023; Long et al., 2024; Nadăș et al., 2025; Yong et al., 2024). AraDetox builds on these developments by providing detoxified rewrites for Arabic social-media content across Modern Standard Arabic, Gulf, Levantine, and Egyptian Arabic using GPT-5 and Gemini. Unlike existing Arabic detoxification resources, AraDetox emphasises semantic preservation while retaining the original intent, stance, and target of the source text. AraDetox substantially expands the scale of Arabic detoxification resources and, to the best of our knowledge, is the first Arabic detoxification dataset to provide rewrites generated across multiple dialects and multiple LLMs within a single framework.

## 3 Methodology

## 3.1 Dataset Construction

AraDetox is derived from the Arabic Hate Speech Superset (Tonneau et al., 2024), which integrates ten Arabic hate-speech and abusive-language datasets: T-HSAB (Haddad et al., 2019), JHSC (Ahmad et al., 2024), MLMA (Ousidhoum et al., 2019), L-HSAB (Mulki et al., 2019), OSACT (Mubarak et al., 2020), Let-Mi (Mulki and Ghanem, 2021), Saudi Tweets (Alshaalan and Al-Khalifa, 2020), AraCOVID19-MFH (Ameur and Aliane, 2021), Brothers (Albadi et al., 2018), and Alsafari et al. (Alsafari et al., 2020). From the 16,000 harmful posts available in the superset, we selected 10,500 posts while retaining coverage across all ten constituent datasets. No class balancing was applied, and the source datasets were not equally represented because they varied substantially in size. The selection was intended to provide broad coverage of the available harmful-language data rather than a balanced classification sample. Table 1 summarises the resulting resource.

Using GPT-5 and Gemini 2.5 Flash, we generated detoxified rewrites in MSA, Gulf, Levantine, and Egyptian Arabic. Models were instructed to preserve the original meaning, target, stance, sentiment, and communicative intent while removing profanity, insults, slurs, dehumanising language, discriminatory expressions, and threats. For the dialectal variants, outputs were additionally required to adapt the source text into Gulf, Levantine, or Egyptian Arabic. These outputs therefore combine two transformations: harmful-language detoxification and adaptation to the requested Arabic variety.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Source posts</td><td>10,500</td></tr><tr><td>Detoxified outputs</td><td>84,000</td></tr><tr><td>Arabic varieties</td><td>MSA, Gulf, Levantine,</td></tr><tr><td>Generation models</td><td>Egyptian GPT-5, Gemini 2.5 Flash</td></tr><tr><td>Outputs per source post</td><td>8</td></tr><tr><td>Outputs generated by GPT-5</td><td>42,000</td></tr><tr><td>Outputs generated by Gemini</td><td>42,000</td></tr><tr><td>Total words</td><td>1,318,257</td></tr><tr><td>Average words per output</td><td>15.69</td></tr><tr><td>Median words per output</td><td>13</td></tr><tr><td>Human-evaluation sample</td><td>300 source posts</td></tr><tr><td>Human-evaluated outputs</td><td>2,400</td></tr><tr><td>Annotators</td><td>3</td></tr><tr><td>Annotation records</td><td>7,200</td></tr></table>

Table 1: Summary statistics for AraDetox.

The MSA outputs primarily represent detoxification within the same language variety, whereas the Gulf, Levantine, and Egyptian outputs jointly involve detoxification and dialect adaptation. Consequently, lexical and structural diferences between the source posts and the dialectal outputs cannot be attributed exclusively to detoxification. We therefore evaluate harmful-language removal, meaning preservation, and dialectal style as separate dimensions and interpret the dialectal results as evidence ofjoint detoxification and dialect adaptation. This produced eight detoxified variants per source post, yielding 84,000 detoxified instances and more than 1.3 million words. The full prompts are provided in Appendix A. Dataset construction incorporated automatic validation, regeneration ofinvalid outputs, and human quality control. Automatic checks identified missing fields, malformed JSON, null values, duplicated outputs, incomplete generations, and source–output alignment errors. Safety refusals, generic moderation responses, summaries, externalobserver descriptions, and outputs that substantially altered or added to the source meaning were treated as invalid and regenerated. A native Arabicspeaking reviewer subsequently verified that the regenerated outputs performed the requested detoxification task, preserved the source meaning, target, stance, and communicative intent, and reflected the requested Arabic variety. Further details ofthe generation procedure and quality-control workflow are provided in Appendix B.

As an additional quality-assurance step, 300 source posts were randomly selected from the 10,500-post collection. The sample was stratified to preserve the distribution of the ten source datasets across the ten source datasets and available harm categories. For each selected post, all eight detoxified variants were evaluated by three native Arabicspeaking annotators, yielding 2,400 generated outputs and 7,200 annotation records. The annotators involved in this evaluation were distinct from the native Arabic speaker who performed quality control during dataset construction. Appendix E reports the distribution of the full dataset and the human-evaluation sample across the constituent source datasets.

## 3.2 Evaluation Design

We evaluate AraDetox from five complementary perspectives: lexical change, semantic preservation, sentiment shift, dialectal style, and human evaluation. We additionally compare AraDetox with MultiParaDetox-Ar (Dementieva et al., 2024, 2025) to contextualise its detoxification behaviour relative to an existing Arabic detoxification resource. A key challenge in detoxification is preserving meaning while removing harmful content. Ofensive expressions often convey stance, criticism, and emotion rather than functioning as removable noise. For example, the insult خنزیر یا الكلب انت لا ”) No, you are the dog, you pig”) can be rewritten as إنك بل ،لا

are you,rather,No (“أنت من یتصرف بطر یقة غیرمقبولة the one behaving in an unacceptable manner”), preserving the criticism while removing the ofensive language. Appendix F presents additional qualitative examples from AraDetox. F1 score is not used as a generation-quality metric because detoxification does not have a single correct reference sequence from which token-level true positives, false positives, and false negatives can be meaningfully derived. Multiple lexically diferent rewrites may all be valid. We therefore combine surfacechange measures, semantic similarity, sentiment analysis, dialectal-style analysis, and human judgements, with each evaluation component measuring a distinct requirement ofsuccessful detoxification.

## 3.2.1 Lexical Change

We measure lexical change between each source post and its detoxified counterpart using six surfacelevel metrics: lexical overlap, Jaccard similarity, character-level edit distance, percentage of changed characters, word-level edit distance, and output length. Lexical overlap and Jaccard similarity measure how much of the original vocabulary is retained, with higher values indicating more conservative rewriting. Character- and word-level edit distances quantify the amount of rewriting required to transform the source post into the detoxified output, while the percentage of changed characters normalises this change by source length. Output length is included to capture whether detoxification tends to shorten, preserve, or expand the original text. These measures allow us to distinguish minimal-edit detoxification from broader sentencelevel reformulation.

## 3.2.2 Semantic Preservation

To quantify semantic fidelity, we compute cosine similarity between each source post and its detoxified counterpart using multilingual-e5-large (Wang et al., 2024). This embedding-based measure assesses whether the detoxified text remains semantically close to the original despite lexical and structural changes. To complement the pairwise similarity analysis, we examine the global organisation of the embedding space using UMAP. By visualising source posts, detoxified outputs, and external neutral Arabic corpora in a shared semantic space, we assess whether detoxified texts remain close to their source content while exhibiting characteristics associated with naturally occurring neutral language.

## 3.2.3 Sentiment and Neutrality Analysis

Detoxification should remove harmful language without eliminating criticism, disagreement, or negative opinions. We therefore examine whether detoxification systematically shifts sentiment by increasing neutral or positive predictions at the expense of negative ones. This follows the broader use of sentiment analysis for examining evaluative language and afect across diferent domains and languages, including financial discourse, social media, and multilingual settings (El-Haj et al., 2016; Alwakid et al., 2022; Hunter et al., 2023; Huy et al., 2026). Sentiment distributions are computed using three CAMeLBERT sentiment classifiers: CAMeLBERT-DA, CAMeLBERT-MSA, and CAMeLBERT-Mix (Inoue et al., 2021). Changes in positive, neutral, and negative predictions are analysed before and after detoxification to assess whether rewriting reduces hostile afect while preserving the original stance.

## 3.2.4 Dialectal Adaptation

AraDetox includes detoxified variants in Gulf, Levantine, and Egyptian Arabic. To examine whether these rewrites exhibit stylistic alignment with their intended varieties, we compare them against reference dialect corpora (Zaidan and Callison-Burch,

2014) using TF–IDF cosine similarity over word and character n-grams. This analysis provides corpus-level evidence of lexical and orthographic alignment, but it is sensitive to topic, register, corpus composition, and vocabulary shared across Arabic varieties. It is therefore treated as an exploratory measure of dialectal style rather than definitive validation of dialect authenticity. For example, the MSA phrase $\varepsilon \downarrow  1 1 g \ddot { o l } i 2 \dot { 1 }$ ”) they blocked the street”) may be realised as $\varepsilon \lrcorner \lrcorner$ سادین in Gulf Arabic, $1 _ { s } \sin$ $[ \mathcal { S } ^ { \{ \succ \} \dagger }$ in Levantine Arabic, and $[ \mathcal { S } ^ { \bot \bot } \vert \vert \mathcal { S } ^ { \vert } \dot { \mathcal { a } } \ddot { \mathcal { } }$ in Egyp-tian Arabic, illustrating meaning-preserving adaptation across Arabic varieties.

## 3.2.5 Human Evaluation Protocol

To assess the quality ofthe generated outputs, three native Arabic-speaking annotators independently reviewed a random sample of300 source posts. For each post, all eight detoxified variants were evaluated, yielding 2,400 generated outputs and 7,200 annotation records. Each output was assessed for offence removal, meaning preservation, and introduction ofnew content. Ofence removal was annotated using a three-way scheme: ofence not removed, offence removed, or not applicable when the source text did not contain ofensive content. Meaning preservation and new content were annotated using binary judgements. The full annotation guidelines are provided in Appendix D. Dialect authenticity was not included as a separate human-evaluation criterion. The dialectal findings therefore rely on corpus-level stylistic analysis and should not be interpreted as direct human validation of native-like dialect use.

## 3.2.6 Comparison with Existing Resources

To contextualise AraDetox, we compare it descriptively with MultiParaDetox-Ar using the same lexical, semantic, and sentiment-based measures. The resources difer in source-domain distribution, size, annotation procedure, and the initial sentiment and severity of their source texts. The comparison is therefore intended to characterise their rewriting behaviour rather than establish that one resource or construction method is superior. It also contrasts the relatively minimal-edit strategy represented by MultiParaDetox-Ar with the broader meaningpreserving reformulation and dialect adaptation represented by AraDetox.

## 4 Results

## 4.1 Lexical Change

Table 2 summarises the lexical transformations introduced by all eight detoxification systems. Overlap and Jaccard measure vocabulary retention between the original and detoxified texts, while CharEdit, Char%, WordEdit, and Word% capture character- and word-level modifications. Across all variants, detoxification results in substantial modification of the original text, indicating that the generated outputs frequently involve sentence-level reformulation rather than simple replacement of ofensive expressions. Among all systems, GPT-MSA is the most conservative. It achieves the lowest character-level edit distance (42.19), the lowest proportion ofmodified characters (61.58%), the lowest word-level edit distance (11.77), the lowest wordlevel modification rate (94.15%), and produces the shortest outputs (11.28 words on average). GPT-MSA therefore preserves more of the original surface form than the other systems.

In contrast, Gemini-MSA performs more extensive rewriting than GPT-MSA, exhibiting larger character- and word-level modifications, higher proportions of changed characters, and longer outputs. The dialectal variants generally involve even greater reformulation. Across both models, output lengths increase from an average of 13.21 words in the original posts to 15.66–16.79 words, while character-level modification rates exceed 78%. Normalised word-level edit rates range from 115.79% to 122.94%. Values above 100% occur because word-level edit distance is normalised by the original text length and includes insertions, deletions, and substitutions, allowing the number ofedit operations to exceed the number of words in the source text.

GPT-Gulf exhibits the highest lexical overlap (0.2942) and Jaccard similarity (0.1719), indicating greater retention of source vocabulary. In contrast, Gemini-MSA and Gemini-Egyptian show the lowest overlap scores, suggesting a greater degree of lexical reformulation. The lexical analyses suggest that Arabic detoxification is best characterised as a meaning-preserving rewriting task rather than simple lexical substitution. Whether these extensive modifications preserve the underlying message is examined through the semantic and human evaluations that follow.

## 4.2 Semantic Preservation

Table 3 reports semantic similarity between the original posts and the AraDetox variants using multilingual E5 embeddings. Across all variants, semantic similarity remains consistently high, with mean scores ranging from 0.9234 to 0.9308 and all examples exceeding the 0.70 similarity threshold. These results suggest that the generated rewrites preserve the core meaning of the source posts despite the substantial lexical modifications reported in Section 4.1.

The dialect-aware variants achieve slightly higher similarity scores than the MSA variants, with Levantine obtaining the highest mean similarity (0.9308), followed by Egyptian and Gulf Arabic. GPT-5 and Gemini also exhibit comparable levels of semantic preservation, with mean similarities of 0.9234 and 0.9249, respectively. The relatively small diferences across variants suggest that semantic fidelity is largely unafected by the choice of dialect or generation model.

The lexical and semantic results show that substantial reformulation does not necessarily come at the expense of semantic fidelity. The generated outputs often difer considerably from the source posts at the surface level, while remaining close in meaning. This supports treating detoxification as a meaning-preserving rewriting task rather than simple lexical substitution.

Figures 1 and 2 visualise the embedding space using UMAP. Across all variants, substantial overlap is observed between the original posts and their detoxified counterparts, with no clear separation between source texts and generated rewrites. Similar patterns are observed for the dialect-aware variants, whose outputs occupy largely the same regions of the embedding space as the original posts. These visualisations are consistent with the high semanticsimilarity scores reported in Table 3 and provide additional evidence that detoxification and dialect adaptation preserve the semantic structure of the source content.

## 4.3 Sentiment Shift

We investigate whether detoxification influences negative afect using three Arabic sentiment models: CAMeLBERT-DA, CAMeLBERT-MSA, and CAMeLBERT-Mix. Sentiment is not treated as a direct measure oftoxicity, since a post may remain critical, negative, or emotionally charged without being abusive, hateful, or discriminatory. Rather, sentiment provides an additional perspective on how detoxification alters the tone ofthe text while preserving its meaning and intent.

<table><tr><td>System</td><td>Overlap</td><td>Jaccard</td><td>CharEdit</td><td>Char%</td><td>WordEdit</td><td>Word%</td><td>Len</td></tr><tr><td>GPT-MSA</td><td>0.2062</td><td>0.1339</td><td>42.19</td><td>61.58</td><td>11.77</td><td>94.15</td><td>11.28</td></tr><tr><td>GPT-Gulf</td><td>0.2942</td><td>0.1719</td><td>45.52</td><td>81.24</td><td>13.08</td><td>115.79</td><td>15.99</td></tr><tr><td>GPT-Levantine</td><td>0.2737</td><td>0.1610</td><td>45.90</td><td>80.66</td><td>13.24</td><td>116.14</td><td>15.66</td></tr><tr><td>GPT-Egyptian</td><td>0.2537</td><td>0.1454</td><td>48.07</td><td>84.72</td><td>13.67</td><td>120.51</td><td>16.04</td></tr><tr><td>Gemini-MSA</td><td>0.2007</td><td>0.1201</td><td>52.74</td><td>83.61</td><td>14.59</td><td>119.38</td><td>16.76</td></tr><tr><td>Gemini-Gulf</td><td>0.2342</td><td>0.1353</td><td>49.90</td><td>83.90</td><td>14.39</td><td>122.94</td><td>16.79</td></tr><tr><td>Gemini-Levantine</td><td>0.2440</td><td>0.1471</td><td>47.69</td><td>78.88</td><td>13.94</td><td>117.08</td><td>16.27</td></tr><tr><td>Gemini-Egyptian</td><td>0.2188</td><td>0.1263</td><td>49.93</td><td>81.26</td><td>14.50</td><td>120.85</td><td>16.75</td></tr></table>

Table 2: Lexical change statistics.

<table><tr><td>Variant</td><td>Mean</td><td>Median</td><td>Min.</td><td>&gt;0.90</td></tr><tr><td>Levantine</td><td>0.9308</td><td>0.9349</td><td>0.7883</td><td>80.31%</td></tr><tr><td>Egyptian</td><td>0.9274</td><td>0.9307</td><td>0.7789</td><td>77.93%</td></tr><tr><td>Gulf</td><td>0.9256</td><td>0.9293</td><td>0.7700</td><td>76.15%</td></tr><tr><td>Gemini</td><td>0.9249</td><td>0.9278</td><td>0.7691</td><td>73.30%</td></tr><tr><td>GPT-5</td><td>0.9234</td><td>0.9278</td><td>0.7504</td><td>71.05%</td></tr></table>

Table 3: Semantic similarity using multilingual E5 embeddings.  
![](images/238d4fa21fedc38edaf828600d29003ca872bc73b38bdcc7efa51b7152b8b58f.jpg)  
Figure 1: UMAP visualisation of original posts, GPT-MSA detoxifications, and Gemini-MSA detoxifications using multilingual E5 embeddings.

Table 4 and Figure 3 report the percentage of posts classified as negative before and after detoxification. The original posts are classified as negative at high rates across all three sentiment models, ranging from 72.41% to 75.47%. After detoxification, the strongest reduction is observed for the MSA variants. GPT-MSA reduces negative sentiment to 49.20%–53.77%, while Gemini-MSA reduces it to 53.19%–60.26%. MSA detoxification substantially softens the afective tone of harmful posts.

The dialectal variants also reduce negative sentiment, but to a more limited extent. Across both GPT and Gemini, Gulf, Levantine, and Egyptian outputs remain closer to the original sentiment distribution, with negative rates generally between 65% and 74%. Dialectal rewriting preserves more of the original emotional tone, even after harmful expressions have been removed. The pattern is also visible in Figure 4, where the MSA variants show the clearest drop in average negative sentiment, while the dialectal variants remain nearer to the original posts.

![](images/c472a7f45bc2f5ceae2f7b2723307fc6b169e0fa51a658b5f8e2830f2ef9ca23.jpg)

Figure 2: UMAP visualisation of original posts and GPT/Gemini dialectal detoxified variants using multilingual E5 embeddings.
<table><tr><td>Model</td><td>Orig.</td><td>GPT-MSA</td><td>GPT-Gulf</td><td>GPT-Lev.</td><td>GPT-Egy.</td></tr><tr><td>DA</td><td>74.29</td><td>53.77</td><td>70.32</td><td>71.14</td><td>69.38</td></tr><tr><td>MSA</td><td>75.47</td><td>53.48</td><td>71.23</td><td>73.67</td><td>70.44</td></tr><tr><td>Mix</td><td>72.41</td><td>49.20</td><td>68.62</td><td>69.56</td><td>66.36</td></tr></table>

<table><tr><td>Model</td><td>Orig.</td><td>Gem-MSA</td><td>Gem-Gulf</td><td>Gem-Lev.</td><td>Gem-Egy.</td></tr><tr><td>DA</td><td>74.29</td><td>55.79</td><td>68.07</td><td>70.13</td><td>70.07</td></tr><tr><td>MSA</td><td>75.47</td><td>60.26</td><td>69.69</td><td>72.24</td><td>70.21</td></tr><tr><td>Mix</td><td>72.41</td><td>53.19</td><td>65.45</td><td>67.42</td><td>65.18</td></tr></table>

Table 4: Negative sentiment before and after detoxification using CAMeLBERT-DA, MSA and Mix models.

These results indicate that part ofthe negative affect detected by sentiment models is associated with abusive wording, insults, profanity, and other harmful linguistic constructions. Once such language is removed, the resulting texts are generally perceived as less negative, particularly in MSA. At the same time, the dialectal variants show that detoxification does not necessarily eliminate criticism, disagreement, or emotional stance. Rather, the generated

![](images/82c536f4e2c1e1ffa348ec569a8addd56b3929280afe8ef738452aad784e1f3b.jpg)  
Figure 3: Negative sentiment across GPT and Gemini variants using three CAMeLBERT models.

outputs preserve much of the communicative intent while reducing harmful expression.  
![](images/a7e61501ca8dcb010ef9d67c473f9f799ab89dc4b2377987a2798b5aeb56fe30.jpg)  
Figure 4: Average negative-sentiment rate across the three CAMeLBERT models.

## 4.4 Dialectal Style Analysis

AraDetox includes detoxified rewrites in MSA, Gulf, Levantine, and Egyptian Arabic. To examine corpus-level dialectal style, we compare the generated variants with reference texts from the Arabic-Dialects corpus using TF–IDF cosine similarity over word and character n-grams.

Figures 5 and 6 show the resulting similarity matrices. The word n-gram results provide the clearest evidence of dialectal alignment. Both MSA systems align most strongly with the MSA reference corpus, with Gemini-MSA achieving the highest MSA similarity (0.373), followed by GPT-MSA (0.350). The Egyptian variants likewise show strong alignment with the Egyptian reference corpus, with GPT-Egyptian obtaining the highest similarity (0.406), followed by Gemini-Egyptian (0.383).

The Gulf and Levantine results are less distinct. Gemini-Gulf aligns most strongly with the Gulf reference corpus (0.285), and Gemini-Levantine aligns most strongly with the Levantine reference corpus (0.297), suggesting clearer dialect-specific stylistic patterns. In contrast, GPT-Gulf and GPT-Levantine show their highest similarity with the Egyptian reference corpus, likely reflecting lexical overlap among informal Arabic varieties and the strong signal of the Egyptian reference set. The character n-gram analysis broadly confirms the

![](images/e07d4b48a2c8bac34401c6465771105585bc531d6fbdd3a7c2615e426dbfa859.jpg)

Figure 5: Word n-gram similarity to reference dialect corpora.  
![](images/c3b7ab4c3026d1edcfdabbcf05e38fb9e75608e3fe383d6fc630be301e51902d.jpg)  
Figure 6: Character n-gram similarity to reference dialect corpora.

MSA and Egyptian findings but provides weaker separation between Gulf and Levantine. Characterlevel features capture orthographic and morphological properties shared across Arabic dialects, making them less discriminative than word n-grams.

The results provide measurable evidence of corpus-level dialectal style, particularly for MSA and Egyptian Arabic. Gemini also shows stronger intended-variety alignment for Gulfand Levantine than GPT-5. However, the reference corpora consist largely of general social-media discussions and forum-style interactions, whereas AraDetox is derived from harmful, highly negative, politically sensitive, and identity-related discourse. Topic, register, and corpus composition may therefore influence the reported similarities. The weaker distinction between Gulf and Levantine further reflects the substantial lexical and orthographic overlap between informal Arabic varieties. These results should consequently be interpreted as evidence of stylistic alignment rather than definitive evidence of nativelike dialect authenticity.

## 4.5 Human Evaluation

Automatic metrics provide useful insights into lexical change, semantic similarity, and sentiment shifts, but they cannot directly determine whether harmful content has been removed while preserving the intended meaning of the original post. To address this limitation, we conducted a human evaluation following the protocol described in Section 3.2.5. Three native Arabic-speaking annotators independently evaluated all outputs in the evaluation sample. The annotators represented Saudi Gulf Arabic, Jordanian Levantine Arabic, and Egyptian Arabic and were regular users ofArabic social media. They were independent of the LLM-generation process and distinct from the native Arabic speaker who performed quality control during dataset construction.

The evaluation focused on three criteria central to detoxification: offence removal, meaningpreservation, and introduction ofnew content. These dimensions assess whether generated rewrites successfully remove harmful language, retain the original communicative intent, and avoid introducing information not present in the source text. To minimise potential bias, annotators were not informed that the texts had been generated by LLMs.

Inter-annotator agreement was substantial across all evaluation dimensions. Fleiss’ κ values ranged from 0.715 to 0.795, indicating consistent judgements among annotators and supporting the reliability of the evaluation results (Table 5).

<table><tr><td>Criterion</td><td>Agreement (%)</td><td>Fleiss&#x27; K</td></tr><tr><td>Offence Removal</td><td>96.44</td><td>0.795</td></tr><tr><td>Meaning Preservation</td><td>95.30</td><td>0.715</td></tr><tr><td>New Content</td><td>92.00</td><td>0.716</td></tr></table>

Table 5: Inter-annotator agreement for human evaluation.
<table><tr><td>Variant</td><td>Offence</td><td>Meaning</td><td>New</td></tr><tr><td>Gemini MSA</td><td>92.90</td><td>92.57</td><td>12.42</td></tr><tr><td>Gemini Gulf</td><td>92.35</td><td>91.46</td><td>15.52</td></tr><tr><td>Gemini Levantine</td><td>92.24</td><td>91.69</td><td>13.75</td></tr><tr><td>Gemini Egyptian</td><td>91.91</td><td>89.36</td><td>17.52</td></tr><tr><td>GPT MSA</td><td>87.25</td><td>86.25</td><td>13.08</td></tr><tr><td>GPT Gulf</td><td>87.25</td><td>91.57</td><td>24.83</td></tr><tr><td>GPT Levantine</td><td>87.47</td><td>91.80</td><td>24.28</td></tr><tr><td>GPT Egyptian</td><td>87.58</td><td>91.69</td><td>24.17</td></tr></table>

Table 6: Human evaluation results (%). Higher is better except for New Content.

Table 6 shows strong performance across all three evaluation dimensions. Ofence-removal rates range from 87.25% to 92.90%, showing that harmful language can generally be removed without substantially altering the underlying message. Gemini achieves the highest ofence-removal scores across all language varieties.

Meaning-preservation rates are also high, ranging from 86.25% to 92.57%. The generated rewrites therefore largely retain the original claims, targets, stances, and communicative intent of the source posts despite modifications introduced during detoxification.

The clearest diferences between the model families emerge in the introduction of new content. Gemini introduces additional information in 12.42%–17.52% of cases, whereas GPT’s dialectal variants do so in 24.17%–24.83% of outputs. Although GPT tends to produce more expansive rewrites, its meaning-preservation scores remain comparable to those of Gemini, suggesting that the additional content generally supplements rather than alters the core meaning of the source text.

The human evaluation findings are consistent with the automatic analyses. Both model families are efective at removing harmful language while preserving the intended meaning of the original posts. The results also point to diferent rewriting strategies: Gemini tends to produce more conservative reformulations, whereas GPT more frequently elaborates on the source content while maintaining similar levels of meaning preservation.

## 4.6 Comparison with Existing Arabic Detoxification Resources

To better understand the characteristics of AraDetox, we compare it descriptively with MultiParaDetox-Ar (Dementieva et al., 2024, 2025), one of the few publicly available Arabic detoxification resources. The comparison covers lexical change, semantic similarity, and sentiment shift. Because the two resources difer substantially in source distribution, size, annotation procedure, and source-text negativity, the results should be interpreted as a comparison ofdataset characteristics rather than as a controlled benchmark.

Table 7 reveals clear diferences in rewriting behaviour. MultiParaDetox-Ar follows a relatively conservative editing strategy, exhibiting higher word overlap (0.559), lower character-level modification (32.96%), and lower word-level modification (46.72%). In contrast, the AraDetox variants exhibit substantially greater reformulation, with wordoverlap scores ranging from 0.201 to 0.294 and considerably higher character- and word-level modification rates than those observed in MultiParaDetox-Ar. Despite these larger lexical and structural changes, semantic similarity remains consistently high across all AraDetox variants. MultiParaDetox-Ar achieves the highest E5 similarity (0.957), consistent with its minimal-edit approach. The AraDetox variants obtain E5 similarities between 0.919 and 0.930, indicating that extensive rewriting can occur while remaining semantically close to the original content. Among the AraDetox variants, the dialect-aware rewrites achieve the strongest semantic preservation, with Gulf and Levantine Arabic producing the highest similarity scores. These findings suggest that the two resources operationalise detoxification through diferent rewriting strategies. MultiParaDetox-Ar primarily reflects minimal-edit detoxification, characterised by conservative lexical modifications and high surface-level similarity to the source text. In contrast, AraDetox emphasises meaning-preserving reformulation, retaining the original claim, target, and stance while permitting substantially broader lexical and structural rewriting. This distinction is particularly evident in the dialectal variants, where detoxification is combined with adaptation to dialect-specific linguistic conventions.

<table><tr><td>Dataset</td><td>Pairs</td><td>Overlap</td><td>Char. %</td><td>Word %</td><td>E5</td></tr><tr><td>MultiParaDetox-Ar</td><td>400</td><td>0.559</td><td>32.96</td><td>46.72</td><td>0.957</td></tr><tr><td>AraDetox GPT-MSA</td><td>10,500</td><td>0.206</td><td>61.58</td><td>94.15</td><td>0.924</td></tr><tr><td>AraDetox GPT-Gulf</td><td>10,500</td><td>0.294</td><td>81.24</td><td>115.79</td><td>0.930</td></tr><tr><td>AraDetox GPT-Lev.</td><td>10,500</td><td>0.274</td><td>80.66</td><td>116.14</td><td>0.930</td></tr><tr><td>AraDetox GPT-Egy.</td><td>10,500</td><td>0.254</td><td>84.72</td><td>120.51</td><td>0.927</td></tr><tr><td>AraDetox Gemini-MSA</td><td>10,500</td><td>0.201</td><td>83.61</td><td>119.38</td><td>0.919</td></tr><tr><td>AraDetox Gemini-Gulf</td><td>10,500</td><td>0.234</td><td>83.90</td><td>122.94</td><td>0.924</td></tr><tr><td>AraDetox Gemini-Lev.</td><td>10,500</td><td>0.244</td><td>78.88</td><td>117.08</td><td>0.928</td></tr><tr><td>AraDetox Gemini-Egy.</td><td>10,500</td><td>0.219</td><td>81.26</td><td>120.85</td><td>0.925</td></tr></table>

Table 7: Comparison ofMultiParaDetox-Ar and AraDetox.

We further compare the datasets using negativesentiment predictions from three CAMeLBERT sentiment models. Table 8 should be interpreted with caution because the source datasets difer substantially. Only around 10–11% of MultiParaDetox-Ar source texts are classified as negative, compared with 72–75% for AraDetox, reflecting the more strongly negative nature of the harmful-language corpora from which AraDetox was derived.

Even with these diferences, distinct patterns emerge. MultiParaDetox-Ar shows relatively small reductions in negative sentiment (1.22–2.39 percentage points), whereas AraDetox MSA variants produce much larger reductions. GPT-MSA reduces negative predictions by 20.51–23.21 percentage points across the three sentiment models, while

Gemini-MSA achieves reductions of 15.21–19.22 points. The dialectal variants show smaller reductions, typically between 1.80 and 7.23 points. These results reinforce the distinction between detoxification and sentiment modification. AraDetox is not designed to convert negative opinions into positive ones, but to express criticism, disagreement, or opposition without insults, slurs, threats, or abusive language. As a result, many detoxified outputs remain negative in sentiment while no longer containing harmful expressions.

<table><tr><td>Model</td><td>Dataset</td><td>Orig.</td><td>Detox</td><td>Red.</td></tr><tr><td rowspan="10">DA</td><td>MultiParaDetox-Ar</td><td>9.92</td><td>7.53</td><td>2.39</td></tr><tr><td>GPT-MSA</td><td>74.29</td><td>53.77</td><td>20.51</td></tr><tr><td>GPT-Gulf</td><td>74.29</td><td>70.32</td><td>3.96</td></tr><tr><td>GPT-Lev.</td><td>74.29</td><td>71.14</td><td>3.14</td></tr><tr><td>GPT-Egy.</td><td>74.29</td><td>69.38</td><td>4.90</td></tr><tr><td>Gemini-MSA</td><td>74.29</td><td>55.79</td><td>18.50</td></tr><tr><td>Gemini-Gulf</td><td>74.29</td><td>68.07</td><td>6.22</td></tr><tr><td>Gemini-Lev.</td><td>74.29</td><td>70.13</td><td>4.15</td></tr><tr><td>Gemini-Egy.</td><td>74.29</td><td>70.07</td><td>4.22</td></tr><tr><td>MultiParaDetox-Ar</td><td>11.00</td><td>9.78</td><td>1.22</td></tr><tr><td rowspan="9">MSA</td><td>GPT-MSA</td><td>75.47</td><td>53.48</td><td>21.99</td></tr><tr><td>GPT-Gulf</td><td>75.47</td><td>71.23</td><td>4.24</td></tr><tr><td>GPT-Lev.</td><td>75.47</td><td>73.67</td><td>1.80</td></tr><tr><td>GPT-Egy.</td><td>75.47</td><td>70.44</td><td>5.03</td></tr><tr><td>Gemini-MSA</td><td>75.47</td><td>60.26</td><td>15.21</td></tr><tr><td>Gemini-Gulf</td><td>75.47</td><td>69.69</td><td>5.78</td></tr><tr><td>Gemini-Lev.</td><td>75.47</td><td>72.24</td><td>3.23</td></tr><tr><td>Gemini-Egy.</td><td>75.47</td><td>70.21</td><td>5.26</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="9">Mix</td><td>MultiParaDetox-Ar</td><td>10.14</td><td>7.78</td><td>2.36</td></tr><tr><td>GPT-MSA</td><td>72.41</td><td>49.20</td><td>23.21</td></tr><tr><td>GPT-Gulf</td><td>72.41</td><td>68.62</td><td>3.79</td></tr><tr><td>GPT-Lev.</td><td>72.41</td><td>69.56</td><td>2.85</td></tr><tr><td>GPT-Egy.</td><td>72.41</td><td>66.36</td><td>6.05</td></tr><tr><td>Gemini-MSA</td><td>72.41</td><td>53.19</td><td>19.22</td></tr><tr><td>Gemini-Gulf</td><td>72.41</td><td>65.45</td><td>6.96</td></tr><tr><td>Gemini-Lev.</td><td>72.41</td><td>67.42</td><td>4.99</td></tr><tr><td>Gemini-Egy.</td><td>72.41</td><td>65.18</td><td>7.23</td></tr></table>

Table 8: Negative sentiment before and after detoxification using CAMeLBERT-DA, MSA and Mix.

## 5 Conclusion

We introduce AraDetox, a large-scale Arabic detoxification dataset containing 10,500 harmful socialmedia posts and 84,000 detoxified rewrites across MSA, Gulf, Levantine, and Egyptian Arabic. The resource combines LLM-assisted generation with automatic validation, native-speaker quality control, and independent human evaluation. The findings show that Arabic detoxification frequently requires substantial reformulation while preserving the source claim, target, stance, and communicative intent. Human evaluation indicates high rates of ofence removal and meaning preservation, although some outputs introduce additional content. The dialectal variants also show measurable corpuslevel alignment with their intended Arabic varieties. AraDetox provides a substantial resource for developing and evaluating Arabic detoxification systems, while highlighting the need for continued work on dialect validation, pragmatic meaning preservation, and downstream evaluation.

## 6 Limitations

This study has several limitations. First, AraDetox was constructed primarily through GPT-5 and Gemini 2.5 Flash rather than large-scale human rewriting. Although automatic validation and nativespeaker quality control were applied, the outputs may retain model-specific stylistic patterns and should not be treated as naturally occurring human rewrites.

Second, detailed human evaluation covered 300 source posts, corresponding to 2,400 generated outputs, and therefore represents only a subset of the full 84,000-output resource. Rare source types or harm categories may be underrepresented.

Third, semantic-preservation analyses rely partly on multilingual embedding models. High cosine similarity does not guarantee preservation of pragmatic meaning, sarcasm, humour, presupposition, target, or subtle stance. Human judgements partly address this limitation but cannot eliminate it.

Fourth, dialectal evaluation relies on corpus-level word and character n-gram similarity, which may be influenced by topic, register, corpus composition, and shared vocabulary across Arabic varieties. As dialect authenticity was not separately evaluated by human annotators, these results should be interpreted as evidence of stylistic alignment rather than native-like dialect use.

Fifth, comparison with MultiParaDetox-Ar is descriptive because the resources difer in size, source distributions, and annotation procedures. We also do not yet assess whether models trained or finetuned on AraDetox outperform those trained on existing Arabic detoxification datasets. A controlled downstream benchmark remains future work.

Finally, the dataset was generated using two proprietary frontier model families. Evaluating smaller open-weight models would help determine sensitivity to model scale, architecture, and training data. AraDetox is also derived from Arabic social-media content and may not generalise to long-form, news, or formal political discourse.

The AraDetox dataset is publicly available at https://github.com/ArabicNLP-UK/ AraDetox to support transparency and reproducibility.

## 7 Ethical Considerations

This work involves harmful Arabic social-media content, including ofensive, abusive, political, sectarian, and identity-related language. The purpose of the study is to develop safer Arabic NLP resources and evaluation methods for neutralising harmful language. We minimise the reproduction ofharmful examples in the paper and report results in aggregate form.

The study included a human evaluation component involving up to ten native Arabic-speaking adult annotators. Ethical approval was granted by an Institutional Ethical Review Board for Biomedical Research in Vietnam. Annotators provided informed consent and were given detailed annotation guidance prior to participation. No personal, sensitive, or health-related data were collected, and all procedures were conducted in accordance with institutional policy, Vietnamese regulations, and internationally recognised ethical standards.

The detoxified outputs should not be treated as authoritative or universally acceptable rewrites, since judgements ofharmfulness and acceptable neutralisation are shaped by social, cultural, dialectal, and political context. We also recognise that automatic detoxification can introduce risks, including oversanitising legitimate political expression, weakening the speaker’s intended stance, or removing important evidence of abuse. Any practical deployment ofArabic detoxification systems should therefore include human oversight and clear moderation guidelines.

Outputs that produced refusals, malformed responses, summaries, external-observer descriptions, or substantial meaning changes were treated as invalid and regenerated during dataset construction. The released resource distinguishes the complete LLM-generated collection from the independently human-evaluated subset. Human-evaluation labels for ofence removal, meaning preservation, and new-content introduction are provided for the evaluated outputs, allowing users to identify and filter unsuccessful or uncertain generations. The complete synthetic collection should not be treated as a gold-standard dataset of human-authored detoxifications.

## References

Muhammad Abdul-Mageed, AbdelRahim Elmadany, and El Moatez Billah Nagoudi. 2021. ARBERT & MARBERT: Deep bidirectional transformers for Arabic. In Proceedings ofthe 59th Annual Meeting ofthe Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages

7088–7105, Online. Association for Computational Linguistics.

Ashraf Ahmad, Mohammad Azzeh, Eman Alnagi, Qasem Abu Al-Haija, Dana Halabi, Abdullah Aref, and Yousef AbuHour. 2024. Hate speech detection in the arabic language: corpus design, construction, and evaluation. Frontiers in Artificial Intelligence, 7:1345445.

Salim Al Mandhari, Mo El-Haj, and Paul Rayson. 2024. Is it ofensive or abusive? an empirical study ofhateful language detection of arabic social media texts. In Proceedings ofthe First International Conference on Natural Language Processing and Artificial Intelligence for Cyber Security, pages 137–146.

Nuha Albadi, Mohamed Kurdi, and Swapna Mishra. 2018. Are they our brothers? analysis and detection of religious hate speech in the arabic twittersphere. In Proceedings of AICS.

Seham Alghamdi, Youcef Benkhedda, Basma Alharbi, and Riza Batista-Navarro. 2024. AraTar: A corpus to support the fine-grained detection of hate speech targets in the Arabic language. In Proceedings of the 6th Workshop on Open-Source Arabic Corpora and Processing Tools (OSACT) with Shared Tasks on Arabic LLMs Hallucination and Dialect to MSA Machine Translation @ LREC-COLING 2024, pages 1–12, Torino, Italia. ELRA and ICCL.

Safa Alsafari, Samira Sadaoui, and Malek Mouhoub. 2020. Hate and ofensive speech detection on arabic social media. Online Social Networks and Media, 19:100096.

Raghad Alshaalan and Hend Al-Khalifa. 2020. Hate speech detection in saudi twittersphere: A deep learning approach. In Proceedings ofthefifth Arabic natural language processing workshop, pages 12–23.

Ghadah Alwakid, Taha Osman, Mahmoud El Haj, Saad Alanazi, Mamoona Humayun, and Najm Us Sama. 2022. Muldasa: Multifactor lexical sentiment analysis of social-media content in nonstandard arabic social media. Applied Sciences, 12(8):3806.

Mohamed Seghir Hadj Ameur and Hassina Aliane. 2021. Aracovid19-mfh: Arabic covid-19 multi-label fake news & hate speech detection dataset. Procedia Computer Science, 189:232–241.

Wissam Antoun, Fady Baly, and Hazem Hajj. 2020. AraBERT: Transformer-based model for Arabic language understanding. In Proceedings ofthe 4th Workshop on Open-Source Arabic Corpora and Processing Tools, with a Shared Task on Offensive Language Detection, pages 9–15, Marseille, France. European Language Resource Association.

Daryna Dementieva, Nikolay Babakov, and Alexander Panchenko. 2024. MultiParaDetox: Extending text detoxification with parallel data to new languages. In Proceedings ofthe 2024 Conference ofthe North

American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers), pages 124–140, Mexico City, Mexico. Association for Computational Linguistics.

Daryna Dementieva, Nikolay Babakov, Amit Ronen, Abinew Ali Ayele, Naquee Rizwan, Florian Schneider, Xintong Wang, Seid Muhie Yimam, Daniil Moskovskiy, Elisei Stakovskii, Eran Kaufman, Ashraf Elnagar, Animesh Mukherjee, and Alexander Panchenko. 2025. Multilingual and explainable text detoxification with parallel corpora. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 7998–8025, Abu Dhabi, UAE. Association for Computational Linguistics.

Mahmoud El-Haj, Paul Rayson, Steve Young, Andrew Moore, Martin Walker, Thomas Schleicher, and Vasiliki Athanasakou. 2016. Learning tone and attribution for financial text mining. In Proceedings ofthe Tenth International Conference on Language Resources and Evaluation (LREC’16), pages 1820–1825.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy of Sciences, 120(30):e2305016120.

Hatem Haddad, Hala Mulki, and Asma Oueslati. 2019. T-hsab: A tunisian hate speech and abusive dataset. In International conference on Arabic languageprocessing, pages 251–263. Springer.

Myra S Hunter, Mahmoud El-Haj, Eleanor Thorne, Amanda Griffiths, and Claire Hardy. 2023. # menopause: Examining the frequency ofcommunications about menopause on twitter between 2014 and 2022. Maturitas, 177:107806.

Hung Nguyen Huy, Mo El-Haj, Dawn Knight, and Paul Rayson. 2026. Freetxt-vi: A benchmarked vietnamese-english toolkit for segmentation, sentiment, and summarisation. arXiv preprint arXiv:2603.05690.

Go Inoue, Bashar Alhafni, Nurpeiis Baimukan, Houda Bouamor, and Nizar Habash. 2021. The interplay of variant, size, and task type in Arabic pre-trained language models. In Proceedings of the Sixth Arabic Natural Language Processing Workshop, Kyiv, Ukraine (Online). Association for Computational Linguistics.

Varvara Logacheva, Daryna Dementieva, Sergey Ustyantsev, Daniil Moskovskiy, David Dale, Irina Krotova, Nikita Semenov, and Alexander Panchenko. 2022. ParaDetox: Detoxification with parallel data. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6804–6818, Dublin, Ireland. Association for Computational Linguistics.

Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. 2024. On llmsdriven synthetic data generation, curation, and evaluation: A survey. In Findings of the Association

for Computational Linguistics: ACL 2024, pages 11065–11082.

Hamdy Mubarak, Hend Al-Khalifa, and Abdulmohsen Al-Thubaity. 2022. Overview of OSACT5 shared task on Arabic ofensive language and hate speech detection. In Proceedinsg of the 5th Workshop on Open-Source Arabic Corpora and Processing Tools with Shared Tasks on Qur’an QA and Fine-Grained Hate Speech Detection, pages 162–166, Marseille, France. European Language Resources Association.

Hamdy Mubarak, Kareem Darwish, Walid Magdy, Tamer Elsayed, and Hend Al-Khalifa. 2020. Overview of osact4 arabic ofensive language detection shared task. In Proceedings of the 4th Workshop on open-source arabic corpora and processing tools, with a shared task on offensive language detection, pages 48–52.

Hamdy Mubarak, Ammar Rashed, Kareem Darwish, Younes Samih, and Ahmed Abdelali. 2021. Arabic ofensive language on Twitter: Analysis and experiments. In Proceedings of the Sixth Arabic Natural Language Processing Workshop, pages 126–135, Kyiv, Ukraine (Virtual). Association for Computational Linguistics.

Hala Mulki and Bilal Ghanem. 2021. Let-mi: An arabic levantine twitter dataset for misogynistic language. arXiv preprint arXiv:2103.10195.

Hala Mulki, Hatem Haddad, Chedi Bechikh Ali, and Halima Alshabani. 2019. L-hsab: A levantine twitter dataset for hate speech and abusive language. In Proceedings ofthe third workshop on abusive language online, pages 111–118.

Mihai Nadăș, Laura Dioșan, and Andreea Tomescu. 2025. Synthetic data generation using large language models: Advances in text and code. IEEE Access.

Nedjma Ousidhoum, Zizheng Lin, Hongming Zhang, Yangqiu Song, and Dit-Yan Yeung. 2019. Multilingual and multi-aspect hate speech analysis. arXiv preprint arXiv:1908.11049.

Manuel Tonneau, Diyi Liu, Samuel Fraiberger, Ralph Schroeder, Scott A. Hale, and Paul Röttger. 2024. From languages to geographies: Towards evaluating cultural bias in hate speech datasets. In Proceedings of the 8th Workshop on Online Abuse and Harms (WOAH 2024), pages 283–311, Mexico City, Mexico. Association for Computational Linguistics.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Multilingual e5 text embeddings: A technical report. arXiv preprint arXiv:2402.05672.

Zheng-Xin Yong, Cristina Menghini, and Stephen Bach. 2024. Lexc-gen: Generating data for extremely lowresource languages with large language models and bilingual lexicons. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 13990–14009.

Omar F. Zaidan and Chris Callison-Burch. 2014. Arabic dialect identification. Computational Linguistics, 40(1):171–202.

## Appendix

## A Detoxification Prompt

For each source post, GPT-5 and Gemini 2.5 Flash were instructed to generate detoxified rewrites in Modern Standard Arabic (MSA), GulfArabic, Levantine Arabic, and Egyptian Arabic. The prompt emphasised meaning preservation while removing harmful language.

Rewrite the following Arabic social-media post into four civil and non-ofensive Arabic versions:

1. Modern Standard Arabic (MSA)

2. Gulf Arabic

3. Levantine Arabic

4. Egyptian Arabic

Preserve the original meaning, target, stance, criticism, sentiment, and communicative intent.

Remove profanity, insults, slurs, dehumanising language, discriminatory expressions, threats, calls for violence, and unnecessarily inflammatory wording.

Do not defend, refute, explain, summarise, factcheck, or respond to the statement. Rewrite directly from the perspective of the original author.

Produce natural social-media language in the requested variety and return the output as JSON:

{   
"msa\_detox":   
"gulf\_detox":   
"levantine\_detox":   
"egyptian\_detox":   
}

## B Generation Procedure and Quality Control

Detoxified rewrites were generated using GPT-5 and Gemini 2.5 Flash through their respective APIs. Generation was performed in batches using structured JSON output to ensure consistent formatting across all generated variants. The final resource contains 42,000 GPT-generated rewrites and 42,000 Gemini-generated rewrites, corresponding to eight detoxified variants for each of the 10,500 source posts. Additional generations were performed during prompt development, quality control, and regeneration of invalid outputs. GPT-5 and Gemini 2.5 Flash were selected as representatives of two contemporary frontier LLM families with strong multilingual generation capabilities. Gemini 2.5 Flash provides a favourable trade-of between generation quality, latency, and cost for large-scale dataset construction, while GPT-5 was included to provide a complementary model family and enable crossmodel comparisons. Our objective was not to compare LLM performance, but to investigate whether meaning-preserving detoxification remained consistent across diferent model families and Arabic language varieties.

<table><tr><td>Parameter</td><td>GPT-5</td><td>Gemini 2.5 Flash</td></tr><tr><td>API access</td><td>OpenAI API</td><td>Gemini API</td></tr><tr><td>Temperature</td><td>API default</td><td>API default</td></tr><tr><td>Top-p</td><td>API default</td><td>API default</td></tr><tr><td>Maximum output tokens</td><td>API default</td><td>API default</td></tr><tr><td>Structured JSON output</td><td>Yes</td><td>Yes</td></tr><tr><td>Prompt development</td><td>Iterative pilot testing</td><td>Iterative pilot testing</td></tr><tr><td>Automatic regeneration</td><td>Yes</td><td>Yes</td></tr><tr><td>Human quality control</td><td>Yes</td><td>Yes</td></tr></table>

Table 9: Generation settings used during dataset construction. Parameters not explicitly specified in the API requests used the provider’s default values.

Prompt development. The final prompt was developed through iterative pilot testing on 100 source posts. Earlier prompt versions were revised to reduce safety refusals, summaries, external-observer responses, malformed JSON, unsupported additions, and changes to the original target or stance. The final prompt was selected because it produced the most consistent structured outputs while preserving the source perspective and communicative intent. Table 9 summarises the generation settings used for both models.

The generation scripts implemented automatic validation checks to detect missing fields, malformed JSON, duplicated outputs, null values, and incomplete generations. Outputs that failed these checks were automatically flagged for regeneration. Retry mechanisms were also implemented to handle API failures, empty responses, malformed outputs, and safety-related refusals. Following generation, each batch underwent a quality-control stage performed by a native Arabic speaker. The objective of this stage was not to rewrite or annotate the outputs, but to verify that the generation process had been completed correctly and that the resulting texts were suitable for inclusion in the dataset.

Quality control focused on several aspects. The reviewer checked that each generated output corresponded to the correct source post and preserved its intended meaning, target, stance, and communicative intent. They also verified that the MSA, Gulf, Levantine, and Egyptian variants reflected the intended language variety and exhibited natural linguistic usage. The reviewer inspected the outputs for technical issues, including alignment errors between source posts and generated rewrites, inconsistencies across dialectal variants, and generation failures that were not detected automatically.

A particular challenge arose from the safety mechanisms of contemporary LLMs. When presented with highly ofensive or sensitive content, models occasionally refused the task and produced safety-oriented responses rather than detoxified rewrites (e.g. I cannotfulfil this request. I am programmed to be a helpful and harmless AI assistant). In other cases, the models adopted an external observer perspective by describing the content of a post rather than rewriting it from the perspective of the original author (e.g. The speaker appears unhappy with the government’s stance). Such outputs were considered invalid because they failed to perform the intended detoxification task.

When these issues occurred, the afected instances were regenerated and, where necessary, additional clarification was provided that the task formed part of a research experiment focused on meaning-preserving detoxification. The reviewer also checked for hallucinated content, omissions of important information, and cases where the generated text substantially altered the meaning of the original post.

This quality-control stage served as a human verification process rather than an annotation exercise. The reviewer did not manually rewrite the posts. Their role was to ensure that the LLMs correctly executed the detoxification task and that the generated outputs were complete, aligned with the source posts, linguistically appropriate, and suitable for subsequent analysis.

## C Experimental Pipeline

The experimental pipeline used in this paper consists of the following stages:

1. Collect harmful Arabic social-media posts from the Arabic Hate Speech Superset.

2. Generate detoxified Modern Standard Arabic (MSA) rewrites using GPT-5 and Gemini 2.5 Flash.

3. Generate detoxified GulfArabic rewrites using GPT-5 and Gemini 2.5 Flash.

4. Generate detoxified Levantine Arabic rewrites using GPT-5 and Gemini 2.5 Flash.

5. Generate detoxified Egyptian Arabic rewrites using GPT-5 and Gemini 2.5 Flash.

6. Compute lexical-change metrics between orig-

inal and detoxified texts.

7. Compute semantic similarity using multilingual E5 embeddings.

8. Visualise embedding spaces using UMAP.

9. Measure sentiment shift using three CAMeL-BERT sentiment models.

10. Evaluate dialectal style similarity using authentic Arabic dialect corpora.

11. Conduct human evaluation using three native Arabic-speaking annotators.

12. Compute inter-rater agreement and aggregate evaluation statistics.

## D Human Annotation Guidelines

Human annotators assessed each generated output according to the following criteria:

1. Offence Removal: whether harmful language present in the source text was successfully removed.

2. Meaning Preservation: whether the main meaning and communicative intent of the source text were preserved.

3. New Content: whether the generated output introduced information not present in the source text.

Ofence removal was annotated using a three-way scheme:

• 0 = ofence not removed,

• 1 = ofence removed,

• 2 = not applicable (no ofence present in the source text).

All remaining criteria were annotated using binary judgements, where 1 indicates that the criterion was satisfied and 0 indicates that it was not.

## E Human-Evaluation Sample Distribution

Table 10 compares the distribution ofthe complete AraDetox source collection with that of the 300- post human-evaluation sample.

<table><tr><td>Source dataset</td><td>Evaluation sample</td><td>Evaluation sample (%)</td></tr><tr><td>JHSC</td><td>30</td><td>10.00</td></tr><tr><td>MLMA</td><td>30</td><td>10.00</td></tr><tr><td>T-HSAB</td><td>30</td><td>10.00</td></tr><tr><td>L-HSAB</td><td>30</td><td>10.00</td></tr><tr><td>AraCOVID19-MFH</td><td>30</td><td>10.00</td></tr><tr><td>Brothers</td><td>30</td><td>10.00</td></tr><tr><td>Saudi Tweets</td><td>30</td><td>10.00</td></tr><tr><td>Alsafari et al.</td><td>30</td><td>10.00</td></tr><tr><td>Let-Mi</td><td>30</td><td>10.00</td></tr><tr><td>OSACT</td><td>30</td><td>10.00</td></tr><tr><td>Total</td><td>300</td><td>100.00</td></tr></table>

Table 10: Distribution of the 300 source posts included in the human-evaluation sample, with 30 posts sampled from each dataset.

## F Qualitative Examples

Tables 11 and 12 present representative examples from AraDetox. The examples illustrate a range of detoxification behaviours, including insult mitigation, meaning-preserving reformulation, stance preservation, and dialect-specific adaptation. The examples also highlight diferences between GPT-5 and Gemini, with some outputs favouring direct lexical substitution and others employing broader paraphrastic rewriting.

<table><tr><td>Source</td><td>GPT MSA</td><td colspan="2">GPT Gulf</td><td colspan="2">GPT Levantine</td><td colspan="2">GPT Egyptian</td></tr><tr><td>  </td><td>  c</td><td>J65U</td><td></td><td>JL5U</td><td></td><td>JL50</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>a</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SC</td><td>sU </td><td> </td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Representative detoxification examples from AraDetox across GPT-5 variants. The examples illustrate meaning-preserving detoxification and dialect-specific adaptation across MSA, Gulf, Levantine, and Egyptian Arabic.

<table><tr><td>Source</td><td>Gemini MSA</td><td>Gemini Levantine</td><td>Gemini Egyptian</td></tr><tr><td>  </td><td></td><td> </td><td> </td></tr><tr><td></td><td></td><td>Gi</td><td></td></tr><tr><td></td><td></td><td>a </td><td></td></tr><tr><td>C</td><td></td><td></td><td></td></tr></table>

Table 12: Representative detoxification examples from AraDetox across Gemini variants. The examples illustrate meaning-preserving detoxification and dialect-specific adaptation across MSA, Gulf, Levantine, and Egyptian Arabic.