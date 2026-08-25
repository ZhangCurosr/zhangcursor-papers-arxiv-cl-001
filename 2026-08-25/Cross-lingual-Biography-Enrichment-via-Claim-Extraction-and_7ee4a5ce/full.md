# Cross-lingual Biography Enrichment via Claim Extraction and Alignment

Yifei Song<sup>1,2</sup>, Ziyang Chen<sup>1,3</sup>, Emil Sayilov<sup>4</sup>, Claire Gardent<sup>1</sup>

<sup>1</sup>CNRS/LORIA <sup>2</sup>Université de Lorraine <sup>3</sup>Université Paris Dauphine - PSL

<sup>4</sup>ICube Laboratory, Université de Strasbourg

{yifei.song,ziyang.chen,claire.gardent}@loria.fr sayilov@unistra.fr

## Abstract

English Wikipedia is often treated as the default encyclopedic source, yet non-English Wikipedia editions can contain richer locally grounded information for long-tail figures. We study cross-lingual biography enrichment: enriching an existing English biography with facts supported by a non-English biography about the same person. Focusing on women from non-English-speaking contexts, we introduce CLAW-4L, a benchmark consisting of 300 Wikipedia biography pairs linking an English biography with its French, Chinese or Azerbaijani counterpart, along with claim annotations and a fine-grained claim-pair relation corpus. We propose a claim-based enrichment framework that extracts English claims from both biographies, aligns them to identify enrichment evidence from the non-English biography, and rewrites the English biography using the selected claims. Our results show that non-English Wikipedia biographies provide valuable evidence for improving English biography coverage, while lower-resource settings remain challenging<sup>1</sup>.

## 1 Introduction

Wikipedia is a foundational knowledge resource for NLP, supporting language-model pretraining, multilingual modeling, retrieval-augmented generation, and fact verification (Pires et al., 2019; Guu et al., 2020; Lewis et al., 2021). Yet Wikipedia coverage is uneven across languages: the same entity may be described with substantially different levels of detail across language editions (Roy et al., 2020). English Wikipedia is often treated as the default encyclopedic source, but it is not always the richest source for long-tail entities with stronger local relevance in non-English communities.

This asymmetry motivates our focus on Englishlanguage biography enrichment for women from non-English-speaking contexts, where figures may be well documented in local Wikipedia editions while remaining underspecified in English (Wagner et al., 2016). Prior work either generates short biographies from structured data (Vougiouklis et al., 2018; Gardent et al., 2017) or full biographies with retrieval-augmented web evidence (Fan and Gardent, 2022); neither directly studies enriching existing English biographies with curated non-English Wikipedia evidence.

![](images/8ec368223d80d5e0bc71eddd6d284b729da0c557b44c89c22c233c12853ea694.jpg)  
Figure 1: Generation trade-off across three representations of non-English evidence. Raw: the generator receives the original English biography and the raw non-English biography. Translation: the non-English biography is first translated into English, and the generator receives the original English biography and the translation. Claims: the generator receives the original English biography and English enrichment claims selected through cross-lingual claim extraction and alignment. Faint points show individual language–generator settings; large points average over three non-English languages and three open-source generators. Translation and Claims increase supported additions over Raw, while the Claims setting substantially reduces hallucination.

We study cross-lingual biography enrichment: given an English biography and a non-English biography of the same woman, the goal is to enrich the

![](images/d294b0b6c82f6a6335e34e08c836ea5a05654ed3d1548d210e30444359c8c8aa.jpg)  
Figure 2: Overview of our claim-based framework for cross-lingual biography enrichment. The framework first extracts claims from paired English and non-English biographies into a shared English space, then performs claim-pair alignment to identify non-English-side enrichment candidates not already covered by the English side. The selected enrichment claims are used as structured evidence for controlled English biography enrichment.

English version by incorporating additional information that appears in the non-English biography but is absent from the English one. Unlike generating a biography from scratch, this setting starts from an existing English article and uses a curated non-English encyclopedic article as additional evidence to enrich it.

To make this comparison explicit, we evaluate three settings that use the same rewriting generators and differ only in how the non-English biography is represented. In raw cross-lingual generation, the generator receives the original English biography together with the raw non-English biography, and must directly interpret, align, and integrate the non-English content. In translation-based generation, the non-English biography is first translated into English, after which the generator rewrites the original English biography using the translated text as evidence. In our claim-based setting, both biographies are mapped into a shared English claim space; claim alignment filters out information already covered by the English biography, and the generator receives the original English biography together with the remaining enrichment claims. This decomposition makes evidence selection explicit before generation, reduces the long-context and implicit cross-lingual alignment burden, and supports intermediate evaluation.

Figure 1 previews the resulting trade-off. Both translation-based and claim-based generation yield more supported additions than raw crosslingual generation, whereas the claim-based setting achieves a substantially lower hallucination rate. This suggests that English normalization facilitates information transfer, but explicit selection of novel evidence is important for reliable enrichment. Our contributions are as follows:

• We study cross-lingual biography enrichment, a task that uses non-English Wikipedia biographies as evidence for enriching underspecified English biographies.

• We provide benchmarks for evaluating crosslingual biography enrichment from either French, Chinese or Azerbaijani into English (CLAW-4L), cross-lingual claim extraction (CLAW-4L-CX) and English claim-pair relation classification (CLAW-4L-RC).

• We develop and evaluate a claim-based enrichment framework that explicitly selects non-English-supported information missing from the English biography before rewriting, improving the supported-addition versus hallucination trade-off over raw cross-lingual and translation-based generation.

## 2 Related Work

Wikipedia Biography Generation and Enrichment. Early biography-generation work framed the task as data-to-text generation from infoboxes or Wikidata, usually producing a first sentence or short summary (Lebret et al., 2016; Chisholm et al., 2017; Vougiouklis et al., 2018; Kaffee et al., 2018).

Recent systems also generate editable Wikipedia drafts for underrepresented groups from structured data (Mille et al., 2024). These approaches improve coverage, but remain structured-data-driven and short-form compared with enriching existing full biographies.

Retrieval-grounded systems instead synthesize Wikipedia articles from similar pages and web evidence (Banerjee and Mitra, 2016), with recent benchmarks emphasizing full-length structure, grounding, and citations (Zhang et al., 2025). In the biography domain, Fan and Gardent (2022) generate full-length English biographies using retrieved web evidence and show that women biographies are challenging due to limited evidence availability. Closest to our setting, Adak et al. (2025) enrich tail biographies with personal narratives. Our work is complementary: we use human-written non-English Wikipedia biographies as cross-lingual encyclopedic evidence and select non-English-side facts before generation.

Claim Extraction, Decomposition, and Alignment. Claim extraction has been studied through open information extraction and fact extraction for verification (Stanovsky et al., 2018; Thorne et al., 2018), but such methods often rely on taskspecific schemas or language-specific NLP tools. Recent LLM-based factuality work decomposes long-form text into atomic or verifiable claims, including FactScore (Min et al., 2023), FactCheck-GPT (Wang et al., 2024), VeriScore (Song et al., 2024), DnDScore (Wanner et al., 2025) and Claimify (Metropolitansky and Larson, 2025). These methods differ in how they decontextualize, decompose, and verify claims, but they are primarily developed for English inputs rather than cross-lingual claim extraction into a shared English space. In this work, we use these methods as candidate extraction frameworks and evaluate their cross-lingual adaptations empirically.

Cross-lingual fact-checking studies multilingual claim detection, retrieval, and verification (Gupta and Srikumar, 2021; Chang et al., 2023). Our setting differs in requiring English claims to be extracted from non-English biographies, and aligned to identify enrichment evidence. For alignment, NLI and fact-verification models provide useful baselines (Bowman et al., 2015; Tang et al., 2024), but their binary or three-way label spaces cannot distinguish whether a non-English-side claim is exactly covered, more specific, less specific, complementary, contradictory, or unrelated. We use this finer-grained relation space to support enrichmentoriented claim selection in CLAW-4L-RC.

## 3 Task and Method

Given $( B ^ { e n } , B ^ { X } )$ , two Wikipedia biographies about the same person where $B ^ { e n }$ is in English and $B ^ { X }$ is in either French, Chinese or Azerbaijani, the cross-lingual biography enrichment task consists in generating an English biography $\tilde { B } ^ { e n }$ that preserves the content of $B ^ { e n }$ while incorporating additional information supported by $B ^ { X }$

We frame this task as an LLM rewriting task where the input is the original English biography $B ^ { e n }$ augmented with $\mathcal { C } ^ { a d d }$ , the set of English claims that can be extracted from $B ^ { X }$ but not from $B ^ { e n }$

$$
\tilde { B } ^ { e n } = G ( B ^ { e n } , \mathcal { C } ^ { a d d } ) ,
$$

We define a claim as a predicate together with its arguments and modifiers and, when present, the corresponding hedge. However, since biographies are highly factual in nature, claims in our settings largely correspond to facts as hedged contexts are rare.

## 4 Datasets

We create three benchmarks for evaluation.

CLAW-4L (Cross-Lingual Alignment on Women biographies in 4 Languages) comprises 300 cross-lingual pairs of women biographies $( B ^ { e n } , B ^ { X } )$ , automatically labelled with claims. Here, $B ^ { e n }$ is in English while $B ^ { X }$ is in one of our three non-English languages: French, Chinese or Azerbaijani. We use this benchmark to evaluate biography enrichment i.e., the difference in information content between a generated biography and the original English biography (Section 6).

CLAW-4L-CX (Claim Extraction) consists of 600 sentences manually annotated with English claims, of which 300 are English sentences and 100 for each non-English language. We use this benchmark to evaluate different claim extraction models (cf. Section 5.2).

Finally, CLAW-4L-RC (Relation Classification) contains 600 claim pairs annotated with a label indicating the semantic relation between the two claims: is the relation between the two claims one of exact alignment, partial alignment, contradiction or irrelevance? We use this benchmark to evaluate the relation classifiers (Cf. Section 5.1).

## 4.1 CLAW-4L

CLAW-4L consists of 300 paired English and non-English Wikipedia biographies automatically labelled with claims.

We construct the biography pairs in two stages. First, we build oversized country-specific candidate pools from Wikidata by requiring that each entity be a woman, have the corresponding country of citizenship, and include both English and non-English-language Wikipedia sitelinks. We then retrieve and clean the paired biographies, removing markup and non-biographical trailing sections such as references and external links.

To favor biography pairs where the non-English biography is richer (more informative) than the corresponding English biography, we rank candidates by a language-calibrated non-English-to-English token ratio computed on cleaned biographies. The ratio normalizes raw token counts using language-specific inflation factors estimated from parallel multilingual data, reducing tokenizerinduced cross-lingual bias. We use it only as a heuristic proxy for non-English-side content richness, rather than as a direct measure of factual superiority. We then select 100 biographies per non-English language, while applying coarse occupation-aware balancing over artists, scientists, athletes, and politicians. And we annotate each biography with a set of silver claims using the claim extraction model described in Section 5.

Table 1 summarizes the resulting benchmark. Here r<sup>norm</sup> denotes the language-calibrated non-English-to-English token ratio, $r _ { s }$ the non-Englishto-English sentence ratio, and $r _ { c }$ the non-Englishto-English claim ratio. Claim counts are computed from automatically derived reference claim sets using the best claim extraction method identified in Section 5.2. As the table shows, non-English biographies are substantially richer across all three views, with 3.4× more tokens, 3.4× more sentences, and 2.9× more claims on average. Construction details are provided in Appendix B.

## 4.2 CLAW-4L-CX

CLAW-4L-CX consists of sentences from our four languages which are manually annotated with English claims. We create this benchmark by first selecting sentences with various levels of complexity and then crowdsourcing the corresponding claims.

Sentence Selection from CLAW-4L. Our goal is to sample sentences spanning different levels of claim complexity, to avoid evaluating only simple one-claim cases. We use automatically extracted claims as a bootstrapping signal for sentence selection, not as gold labels. Specifically, we run the GPT-5.1-assisted claim extraction pipeline adapted from Claimify (Metropolitansky and Larson, 2025) described in Section 5.2; its selection– decontextualization–decomposition structure provides sentence-level silver claims suitable for estimating claim complexity. We then bucket candidate sentences by silver claim count to obtain a balanced range of sentence complexity.

For each biography, we independently select one sentence from the English version and one from the non-English version, while enforcing a balanced distribution across claim-count buckets within each country-language setting. This yields 600 sentences in total: 300 in English and 300 in non-English languages, with 25 sentences per claim-count bucket, and an average of 2.63 and 2.72 claims per sentence, respectively. Details of the sentence selection procedure and bucket statistics are provided in Appendix B.8.

Crowd-Sourced Annotation. Annotators verify, revise, and supplement the automatically generated claim drafts for each selected sentence, producing the final human reference claim sets. We recruited 13 English-fluent annotators who were pursuing or had completed a Ph.D. in computer science or a closely related field; for each cross-lingual annotation task, annotators were native speakers of the corresponding non-English language. Four annotators participated in the French–English setting, and three in each of the Chinese–English and Azerbaijani–English settings. Before annotation, annotators read the task instructions and completed the task through the provided interface, shown in Appendix B.10. A post-hoc claim-level agreement analysis shows high consistency, with Aligned-F1 above 91% for all languages; details are in Appendix B.11.

## 4.3 CLAW-4L-RC

CLAW-4L-RC comprises 600 pairs of English claims annotated with labels describing the relation between the two claims. Each pair originates from paired English and non-English biographies.

Candidate claim pairs are drawn from the CLAW-4L benchmark and labelled using a twolevel label schema for enrichment-oriented alignment. At the top level, each pair is labeled as Aligned, Contradicted, or Not Relevant. For aligned pairs, we further distinguish exact equivalence (A=B), one-sided enrichment (A>B or B>A), and mutual enrichment (A↔B).

<table><tr><td>Pair</td><td>N</td><td>Tok (EN/Tgt)</td><td> $\mathbf { r _ { t } ^ { n o r m } }$ </td><td>Sent (EN/Tgt)</td><td> $\mathbf { r _ { s } }$ </td><td>Claim (EN/Tgt)</td><td> $\mathbf { r _ { c } }$ </td><td>Max  $\mathbf { T } _ { t g t }$ </td><td>Max  $\mathbf { C } _ { t g t }$ </td></tr><tr><td>EN→FR</td><td>100</td><td>465 / 1867</td><td>4.19</td><td>16.1 / 48.4</td><td>4.64</td><td>29.8 / 101.8</td><td>3.42</td><td>12852</td><td>705</td></tr><tr><td>EN→ZH</td><td>100</td><td>489 / 1041</td><td>2.83</td><td>15.8 / 27.2</td><td>2.01</td><td>29.3 / 72.4</td><td>2.47</td><td>8772</td><td>662</td></tr><tr><td>EN→AZ</td><td>100</td><td>544 / 1967</td><td>3.20</td><td>19.4 / 55.2</td><td>3.48</td><td>33.6 / 93.8</td><td>2.79</td><td>28420</td><td>1363</td></tr><tr><td>EN→X</td><td>300</td><td>500 / 1625</td><td>3.41</td><td>17.1 / 43.6</td><td>3.38</td><td>30.9 / 89.3</td><td>2.89</td><td>28420</td><td>1363</td></tr></table>

Table 1: Summary statistics of CLAW-4L. We report average token (Tok), sentence (Sent), and claim (Claim) counts for English (EN) and non-English biographies, along with normalized token ratios $( r _ { t } ^ { n o r m } )$ , sentence ratios $( r _ { s } ) .$ , and claim ratios $( r _ { c } )$ . Non-English biographies are substantially richer across all levels (e.g., 3.4× tokens and 2.9× claims on average). The maximum non-English biography length reaches 28,420 tokens, and the number of extracted claims reaches 1,363 per instance, highlighting the context length challenges of claim-level alignment and biography generation.

We first identify 100 exact-alignment pairs by selecting pairs with high LaBSE similarity and manually validating them. Starting from these 100 validated exact-alignment pairs, we construct the remaining relation types through controlled structured perturbation and GPT-5.1 rewriting, followed by human validation. Partial-alignment pairs are created by adding or removing non-conflicting detail, contradicted pairs by changing incompatible field values, and not-relevant pairs by pairing claims about different factual aspects. All generated pairs are manually checked against their intended labels, and invalid cases are corrected or rewritten. The final benchmark contains 600 claim pairs: 100 exact-alignment pairs, 300 partialalignment/enrichment pairs, 100 contradicted pairs, and 100 not-relevant pairs. Table 14 presents the label distribution and semantic similarity statistics; additional examples and construction details are provided in Appendix B.12.

## 5 Processing Claims

We first describe how we classify the relation between two claims, how we extract claims, and how we use the relation between two claims to identify "enrichment claims" i.e., claims that are present in the non-English biography but not in the English one. In the next section (Section 6), we describe our cross-lingual biography generation methods.

## 5.1 Classifying Claim Pairs

To support claim alignment and enrichment selection, we use an LLM-classifier that assesses the relation between two English-written claims. The classifier follows the coarse-to-fine schema of

CLAW-4L-RC: it first predicts whether the pair is aligned, contradicted, or not relevant, and for aligned pairs further distinguishes exact alignment, one-sided enrichment, and mutual enrichment. We select this classifier by evaluating open-source and closed-source LLMs, together with strong NLI baselines, on CLAW-4L-RC. We also test whether adding GPT-5.1-parsed structured claim fields (e.g., subject, predicate, object, etc.) improves relation judgment over claim text alone. Performance is summarized using three aggregate metrics: ARC, a macro-average over the three top-level relations; Align-FG, the average accuracy over the four finegrained aligned labels; and Overall, the macroaverage over all six labels. Full settings, NLI comparisons, and structured-field ablations are provided in Appendix C and Table 15. In the textonly setting used downstream, Qwen3.5-9B is the strongest open-source classifier, achieving 95.5 ARC, 89.5 Align-FG, and 93.2 Overall, close to GPT-5.1 at 95.9, 90.8, and 93.3. Structured fields do not consistently improve performance and can introduce parsing noise, so we use the Qwen3.5-9B text-only classifier in subsequent experiments.

## 5.2 Extracting Claims

We adapt five LLM-based claim extraction frameworks to our cross-lingual setting: FactScore, FactCheck-GPT, DnDScore, VeriScore, and Claimify, denoting the adapted variants with the prefix X-. Each adapted framework accepts either an English or a non-English biography and outputs English factual claims. To isolate extraction-framework differences from backbone model quality, we instantiate all five frameworks with GPT-5.1. Detailed method descriptions are provided in Appendix D.

We evaluate the adapted frameworks on CLAW-4L-CX, where each sentence is independently annotated by three annotators, yielding three acceptable human reference claim sets. We use these multiple references since sentence-level factual claims can vary in wording and in how contextual information is made explicit; Appendix B.11 shows high post-hoc agreement among annotators.

Because exact string matching cannot account for paraphrase or decontextualization differences, we evaluate extraction quality based on the relation predicted by the classifier from Section 5.1 between the extracted claim and the reference claim. We also compute degree-normalized claim-level credits: a generated claim that matches multiple reference claims receives reduced precision credit, and a reference claim matched by multiple generated claims receives reduced recall credit. Since each sentence has three human reference sets, we keep the reference set with the highest Exact-Aligned-F1 and report Aligned-F1 and Exact-Aligned-F1; the former counts exact and partial alignment as matches, while the latter counts only exact claim matches. Full metric definitions are provided in Appendix D.3.

Table 2 reports the F1 scores. X-Claimify performs best across languages, especially under Exact-Aligned-F1. We therefore use X-Claimify to extract English reference claim sets for the full CLAW-4L biography pairs. These reference claims are used to select enrichment claims (Section 5.3) and to support downstream generation evaluation (Section 6.2). Open-source backbone comparisons are provided in Appendix D and Table 18.

## 5.3 Identifying Enrichment Claims

Given C<sup>en</sup> the set of claims extracted from an English biography by the GPT-5.1 + X-Claimify extractor and $\bar { \mathcal { C } } ^ { \bar { X } }$ , the set of claims extracted from the corresponding non-English biography, we compute the set of enrichment claims $\mathcal { C } ^ { a d d }$ in two steps as follows. For efficiency at biography scale, we first retrieve candidate English claims for each non-English-side claim using semantic similarity computed by All-MPNet-Base-v2, keeping only pairs above a cosine threshold of 0.7 and within the top-5 nearest neighbors; non-retrieved pairs are treated as not relevant. We then apply the classifier from Section 5.1 to the retained pairs. The classifier predicts whether each pair is aligned, contradicted, or not relevant, with aligned pairs further categorized into exact alignment and one-sided or mutual enrichment.

Retrieval serves only as a permissive coarse filter before LLM-based relation verification, rather than as the final alignment decision. In raw cosine units, mean All-MPNet similarities range from 0.843 to 0.960 across aligned relation types and reach 0.885 for contradicted pairs, compared with only 0.359 for not-relevant pairs (shown on a 0–100 scale in Table 14). We therefore use 0.7 to remove clearly unrelated pairs while retaining semantically related candidates; on the full benchmark, 96% of targetside claims retain at least one candidate. We cap each candidate list at five to bound the number of subsequent verifier calls, with detailed candidatelist statistics provided in Appendix F.

We discard non-English-side claims already fully covered by English, including exact matches $( A { = } B )$ and English-more-specific alignments (A>B), where A is an English-side claim and B is a non-English-side claim. We retain non-Englishadditive alignments $( B { > } A$ and $A {  } B )$ , classifierlabeled conflicts (A⊥B), and non-English-side claims with no relevant English counterpart (A⊣B) as enrichment candidates. Conflicts are retained because both sides are Wikipedia-derived and may require contextual reconciliation rather than automatic filtering. The resulting set of selected claims, denoted C<sup>add</sup>, is used as structured evidence for downstream English biography rewriting.

Table 3 summarizes the enrichment statistics of CLAW-4L. Only 13.9% of non-English-side claims are already covered by English, while 26.2% add detail to partially aligned English claims, and 48.6% have no relevant English-side counterpart. Overall, 86.2% of non-English-side claims are selected as enrichment evidence.

## 6 Enriching English Biographies

We compare three methods for generating the target English biography and evaluate them on CLAW-4L using claim extraction, classification and alignment, asking which method most enriches the original English biographies in terms of claims.

In all settings, the system receives the original English biography together with some form of non-English-side evidence and is prompted to produce an English biography that preserves the original English content while incorporating the additional information supported by the non-English input.

For generation, we compare three strong openweight multilingual instruction-tuned generators: Qwen3.6-27B (Yang et al., 2025), Gemma-4-31Bit<sup>2</sup>, and Mistral-3.2-24B-it<sup>3</sup>. We choose these models to cover high-performing model families developed in different linguistic ecosystems, allowing us to examine whether enrichment behavior varies across Chinese, French, and lower-resource Azerbaijani settings.

<table><tr><td rowspan="3">Method</td><td colspan="2">EN</td><td colspan="2">FR</td><td colspan="2">ZH</td><td colspan="2">AZ</td><td colspan="2">Avg.</td></tr><tr><td>A-F1</td><td>E-F1</td><td>A-F1</td><td>E-F1</td><td>A-F1</td><td>E-F1</td><td>A-F1</td><td>E-F1</td><td>A-F1</td><td>E-F1</td></tr><tr><td>X-FactScore</td><td>52.89</td><td>27.83</td><td>57.89</td><td>35.23</td><td>52.51</td><td>26.30</td><td>54.04</td><td>39.51</td><td>54.33</td><td>32.22</td></tr><tr><td>X-VeriScore</td><td>68.82</td><td>46.92</td><td>70.01</td><td>52.91</td><td>68.12</td><td>42.94</td><td>66.30</td><td>55.11</td><td>68.31</td><td>49.47</td></tr><tr><td>X-DnDScore</td><td>41.91</td><td>11.61</td><td>46.16</td><td>16.08</td><td>44.75</td><td>16.02</td><td>44.76</td><td>22.87</td><td>44.40</td><td>16.65</td></tr><tr><td>X-FactCheck-GPT</td><td>59.60</td><td>43.70</td><td>62.13</td><td>49.87</td><td>54.70</td><td>49.82</td><td>52.91</td><td>44.38</td><td>57.34</td><td>46.94</td></tr><tr><td>X-Claimify</td><td>75.17</td><td>72.44</td><td>78.05</td><td>73.56</td><td>71.12</td><td>71.91</td><td>70.52</td><td>66.80</td><td>73.72</td><td>71.18</td></tr></table>

Table 2: Claim extraction framework comparison on CLAW-4L-CX using GPT-5.1 as the common backbone. A-F1 denotes Aligned-F1, which counts exact and partial alignment as matches; E-F1 denotes Exact-Aligned-F1, which counts only exact alignment. Avg. is the macro-average over EN, FR, ZH, and AZ.

<table><tr><td>Non-English-side status</td><td>Avg.</td><td>Share</td></tr><tr><td>Covered by EN</td><td>12.4</td><td>13.9%</td></tr><tr><td>Non-English-additive</td><td>23.4</td><td>26.2%</td></tr><tr><td>Conflict</td><td>10.2</td><td>11.4%</td></tr><tr><td>Unmatched</td><td>43.4</td><td>48.6%</td></tr><tr><td>Selected</td><td>77.0</td><td>86.2%</td></tr></table>

Table 3: Average non-English-side claim status over the three non-English languages. Covered by EN includes exact matches and English-more-specific alignments. Selected includes non-English-additive, conflict, and unmatched claims. Full language-level statistics are in Appendix F.

The three methods differ only in how information from the non-English input is represented and processed: (i) using the raw non-English biography as input with cross-lingual generation; (ii) using a machine-translated version of the non-English biography with an LLM rewriting conditioned on both the original English text and the translated content; and (iii) generating conditioned on the English biography together with the enrichment claims derived from the non-English biography.

## 6.1 Methods

Cross-Lingual Generation. This is the most direct baseline. We provide the original English biography together with the raw non-English biography and ask the generator to produce an enriched English rewrite based on these two inputs. Because biographies can be long (up to 28,420 tokens; Table 1), we process the non-English biography incrementally at the section level. We cap the English biography input at 4,096 tokens and the non-Englishside context at 2,048 tokens per step; when needed, long sections are further split. Each non-Englishside chunk updates the current English biography, with later generations conditioned on the previously enriched version.

Machine Translation and Generation Pipeline. This setting decouples translation from enrichment. We first translate the non-English biography into English using a language-specific translation model, and then apply the same incremental enrichment procedure as in the Cross-lingual Generation approach. To choose the translation model for each language, we compare candidate translators using GPT-5.1-extracted non-English-side claims as semantic references and select the model that best preserves non-English-side claims after translation. The selected translation models and evaluation details are provided in Appendix E.

Claim-Based Enrichment. This setting exploits the enrichment claims C<sup>add</sup> derived from the non-English biographies (cf. Section 5.3). We provide the original English biography, along with C<sup>add</sup>, and ask the generator to produce an enriched English rewrite. The selected claims are not mechanically appended to the original article; instead, they serve as structured evidence for rewriting the full biography while preserving and coherently integrating the original English content. Compared with the other two settings, this formulation filters out already-covered information before generation, reducing context length and focusing the model on incremental factual additions.

## 6.2 Evaluation

We evaluate generated biographies on CLAW-4L by comparing the claims extracted from the generated biographies $( \hat { \mathcal { C } } ^ { g e n } )$ with the claims extracted from the original English biography $( \mathcal { C } ^ { e n } )$ . To reduce large-scale evaluation cost, claim extraction is performed using X-Claimify with Gemma-4-31B-it, the strongest open-source English extractor in our backbone comparison. We compare generated claims against the reference pool $\mathcal { C } ^ { r e f } = \mathcal { C } ^ { e n } \cup \mathcal { C } ^ { t g t }$ . As in biography-level claim alignment (Section 5.3), for each generated claim, we retrieve candidate reference claims using All-MPNet-Base-v2 cosine similarity (threshold 0.7, top-5), then apply the claim-pair classifier to the retained pairs.

We report metrics for claim growth, factual support, hallucination, and an overall trade-off score. For each generated claim $B \in \hat { \mathcal { C } } ^ { g e n }$ , we categorize it against the reference pool with contradiction priority: B is contradicted if any retained reference claim A is labeled $A \bot B ;$ otherwise, B is valid if some reference claim fully supports it $\scriptstyle ( A = B$ or $A > B ) ;$ ; all remaining claims are unsupported. Let V, K, and U denote the valid, contradicted, and unsupported generated claims. We define

$$
\begin{array} { r } { \mathrm { V a l i d } \% = 1 0 0 \frac { \left| \mathcal V \right| } { \left| \hat { \mathcal C } ^ { g e n } \right| } , } \\ { \mathrm { H a l l u c . } \% = 1 0 0 \frac { \left| \mathcal K \right| + \left| \mathcal U \right| } { \left| \hat { \mathcal C } ^ { g e n } \right| } . } \end{array}
$$

Claim growth is measured by # $\mathsf { L } \mathrm { G a i n } = | \hat { \mathcal { C } } ^ { g e n } | -$ $| { \mathcal { C } } ^ { e n } |$ , with Coverage normalizing this gain by $| \mathcal { C } ^ { a d d } |$ , the enrichment claims from Section 5.3. We then define $\# \mathrm { V a l i d } = \mathrm { V a l i d } \times \# \mathrm { G }$ ain as an estimated supported gain, where Valid is Valid% converted to a proportion.

Our primary overall metric is Bal., which combines supported addition quantity and factual reliability. For each system s, we define

$$
\begin{array} { c } { g _ { s } = \# \mathrm { V a l i d } _ { s } , \qquad v _ { s } = \mathrm { V a l i d } \% _ { s } , } \\ { \tilde { x } _ { s } = \frac { x _ { s } - \operatorname* { m i n } _ { s ^ { \prime } } x _ { s ^ { \prime } } } { \operatorname* { m a x } _ { s ^ { \prime } } x _ { s ^ { \prime } } - \operatorname* { m i n } _ { s ^ { \prime } } x _ { s ^ { \prime } } } , \quad x \in \{ g , v \} , } \\ { \mathrm { B a l } _ { \cdot s } = 5 0 ( \tilde { g } _ { s } + \tilde { v } _ { s } ) . } \end{array}
$$

where the minimum and maximum are computed over all systems in Table 28. Bal. ranges from 0 to 100 and equally weights normalized supported claim growth and factual reliability. Since Valid% = 100 − Halluc.% under our categorization, a higher Bal. indicates more supported additions with less hallucination. Additional metric details and full model-level results are provided in Appendix G.

Human evaluation. We additionally conduct a writing-quality sanity check on 45 Mistralgenerated biographies: three languages, five lengthstratified biography pairs per language, and three enrichment methods. For each selected pair, we evaluate the Raw, Translation, and Claims outputs for the same entity. Three computer-science Ph.D. researchers who use English as a working language independently rate every output on a 1–5 Likert scale for Readability/Fluency, Coherence/Integration (including abrupt insertion of new information), and Wikipedia-Style Writing. Appendix G.4 provides the sampling protocol, full length-stratified results, and agreement statistics.

## 6.3 Results and Analysis

Figure 3 visualizes the primary Bal. score for all language, generator, and evidence-format combinations, while Figure 1 summarizes the average #Valid–hallucination trade-off across evidence formats. Full metric values are provided in Table 28.

![](images/38154eaa51f93ab3fc41a2e7059f0bb6d144924ca62bba57e59f5ea43cfc38dc.jpg)  
Figure 3: Overall generation trade-off across languages, generators, and non-English-evidence formats. Each cell reports Bal.; higher is better. Bolded cells mark the best evidence format for each language–generator pair.

Evidence format. Table 22 summarizes results across evidence formats by averaging across all languages and generators. Translation improves supported additions over raw non-English biographies, but does not reliably reduce hallucination; it increases #Valid from 15.89 to 19.05, while hallucination remains almost unchanged (29.49 vs. 29.59). Claim-based evidence achieves the best trade-off, with the highest average #Valid (20.47) and substantially lower hallucination (23.54). This suggests that our claim-based pipeline reduces three sources of difficulty: long non-English contexts, cross-lingual generation, and implicit alignment between English and non-English-side evidence.

Lower-resource exception. The only setting where translation outperforms claim evidence in Bal. is Azerbaijani with Mistral. We hypothesize that this reflects error propagation in the upstream claim pipeline for the lower-resource language: if claim extraction misses useful evidence, the downstream claim-based generator receives a cleaner but less complete evidence set. This is consistent with the cross-lingual claim extraction results in Table 2, where lower-resource inputs are more challenging than high-resource ones. In contrast, for Gemma and Qwen on Azerbaijani, claim evidence still improves Bal. over raw and translated biographies, suggesting that filtering and English normalization remain beneficial when the generator is more prone to hallucination.

Language and generator effects. Performance is substantially stronger for French and Chinese than for Azerbaijani, suggesting that enriching women from lower-resource language contexts remains challenging, even with evidence from the non-English Wikipedia. Across generators, Mistral achieves the highest Bal. in every languageevidence setting. The full results in Table 28 show that this is largely driven by lower hallucination rates while maintaining competitive supported additions, consistent with more conservative or bettergrounded rewriting behavior.

Human writing quality. Claims obtains the highest Readability/Fluency (3.933) and Wikipedia-Style Writing (3.822) scores, while Raw is slightly higher in Coherence/Integration (3.911 vs. 3.867). Translation scores 3.756, 3.733, and 3.689 on the three respective dimensions. Thus, using selected claims as structured evidence does not incur an evident writing-quality penalty relative to the other enrichment pipelines. This result is a descriptive sanity check rather than a significance-tested comparison with the original English biographies; Appendix G.4 provides the complete analysis.

Qualitative error analysis. We manually inspected 18 Mistral-generated biographies from six controlled cases. For each language, we selected one biography pair with the highest and one with the lowest number of verifier-supported output claims under its best-performing evidence format— Claims for French and Chinese, and Translation for Azerbaijani—and compared the Raw, Translation, and Claims outputs for the same entities. In these cases, claim-based enrichment generally added more targeted, fine-grained events and relations, whereas raw cross-lingual enrichment retained a more narrative style but provided less focused coverage. Translation recovered many details when reliable, but was more sensitive to source and translation noise. Missing claims mainly involved longtail works, institutional roles, and event metadata. Remaining hallucinations included incorrect temporal, geographic, and educational details, unsupported over-specific enumerations, and, in one difficult translation-based case, repetitive generation. Appendix G.3 details the selection protocol and provides representative examples.

## 7 Conclusion

We introduced cross-lingual biography enrichment, in which an existing English biography is enriched with a non-English Wikipedia biography of the same person. We constructed CLAW-4L, a benchmark of English biographies paired with French, Chinese, and Azerbaijani biographies about the same women, together with resources for evaluating claim extraction and alignment. We also developed a claim-based framework that maps both biographies into a shared English claim space, selects enrichment evidence from non-English data, and uses it for controlled English rewriting.

Experiments show that claim-based evidence offers the strongest trade-off between supported additions and hallucination in most language-generator settings. Compared with raw non-English biographies and translations, selected claims reduce longcontext burden and enhance interpretability. A complementary human evaluation finds no evident writing-quality penalty from using selected claims as rewriting evidence relative to the other enrichment pipelines. The Azerbaijani results also show that lower-resource settings remain difficult, suggesting that reliable cross-lingual enrichment depends on robust claim extraction and alignment.

## Limitations

Benchmark scope. Our benchmark focuses on women biographies across three non-English languages: French, Chinese, and Azerbaijani. This design lets us study enrichment in both higherresource and lower-resource settings, but it does not cover the full diversity of Wikipedia language editions, scripts, regions, or biography types. Extending the benchmark to more languages and entity groups is an important direction for future work.

Source reliability. We use Wikipedia biographies as curated encyclopedic evidence, but we do not assume that every statement in every language edition is factually correct. Conflicts between English and non-English-language claims may reflect errors, temporal mismatch, or missing context. Our framework surfaces such cases for downstream reconciliation rather than resolving factual truth against external sources.

Factual verification and revision. While claimbased evidence achieves the lowest hallucination rate among the three settings, post-generation verification could further strengthen factual reliability. A natural extension is a generate–verify– revise loop. After producing an initial biography, the system could decompose it into claims and verify each claim against the input evidence using claim-level factuality and grounding methods such as FactScore, VeriScore, FactCheck-GPT, or MiniCheck (Min et al., 2023; Song et al., 2024; Wang et al., 2024; Tang et al., 2024). Unsupported or contradicted claims could then be removed or corrected before a final rewriting step. Verification could further retrieve independent evidence from structured knowledge bases, citations associated with the source articles, and corroborating evidence from additional Wikipedia language editions, building on retrieval-grounded and citation-aware generation (Lewis et al., 2021; Fan and Gardent, 2022; Zhang et al., 2025). This extension would complement our current contribution of explicitly selecting non-English enrichment evidence before generation with evidence-based factuality control after generation. For deployment-oriented settings, evidence provenance could additionally be exposed to human editors, with unresolved conflicts triggering abstention rather than automatic insertion.

Pipeline errors. Our pipeline also depends on automatic claim extraction and claim-pair relation judgment. Although we evaluate these components on CLAW-4L-CX and CLAW-4L-RC, extraction or alignment errors can still propagate to enrichment generation, especially for lower-resource languages. This limitation is reflected in the Azerbaijani results, where upstream claim processing can miss useful non-English-side evidence.

Evaluation and reproducibility. Our automatic generation evaluation emphasizes claim-level supported additions and hallucinations. We supplement it with a human evaluation of readability, coherence, and Wikipedia-style writing, but this remains a small-scale sanity check covering 15 biography pairs, one generator, and three expert annotators. It does not directly compare enriched outputs with the original English biographies, and method labels were visible to annotators. Accordingly, it is not a comprehensive assessment of editorial quality or deployment readiness. We also use GPT-5.1 for reference claim extraction and several prompt-assisted construction steps; while we release prompts, scripts, and open-weight alternatives, exact reproduction may be affected by closedsource model changes.

Beyond English. A broader direction is to use cross-lingual evidence not only to enrich English biographies, but also to improve biography coverage in lower-resource Wikipedia editions. Our current formulation treats English as the generation target, reflecting its role as a default encyclopedic source in NLP. Future work should study bidirectional and many-to-many enrichment settings, where claims from multiple language editions are aligned and used to support biography expansion in lower-resource languages.

## Acknowledgments

We thank the anonymous reviewers for their feedback. This work received government funding managed by the French National Research Agency under France 2030, reference number “ANR-23- IACL-0004” (AI Chair Gardent: “Semantically Consistent LLM Based Text Generation”). Experiments presented in this paper were carried out using the Grid’5000 testbed, supported by a scientific interest group hosted by Inria and including CNRS, RENATER, and several universities as well as other organizations (see https: //www.grid5000.fr). This work was also granted access to the HPC resources of IDRIS under the allocation AD011016561 made by GENCI.

## References

Sayantan Adak, Pauras Mangesh Meher, Paramita Das, and Animesh Mukherjee. 2025. REVerSum: A multistaged retrieval-augmented generation method to en-

hance Wikipedia tail biographies through personal narratives. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track, pages 732–750, Abu Dhabi, UAE. Association for Computational Linguistics.

Siddhartha Banerjee and Prasenjit Mitra. 2016. Wikiwrite: generating wikipedia articles automatically. In Proceedings ofthe Twenty-Fifth International Joint Conference on Artificial Intelligence, IJCAI’16, page 2740–2746. AAAI Press.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.

Yi-Chen Chang, Canasai Kruengkrai, and Junichi Yamagishi. 2023. XFEVER: Exploring fact verification across languages. In Proceedings of the 35th Conference on Computational Linguistics and Speech Processing (ROCLING 2023), pages 1–11, Taipei City, Taiwan. The Association for Computational Linguistics and Chinese Language Processing (ACLCLP).

Andrew Chisholm, Will Radford, and Ben Hachey. 2017. Learning to generate one-sentence biographies from Wikidata. In Proceedings of the 15th Conference of the European Chapter of the Association for Computational Linguistics: Volume 1, Long Papers, pages 633–642, Valencia, Spain. Association for Computational Linguistics.

Angela Fan and Claire Gardent. 2022. Generating biographies on Wikipedia: The impact of gender bias on the retrieval-based generation of women biographies. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 8561–8576, Dublin, Ireland. Association for Computational Linguistics.

Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini. 2017. The WebNLG challenge: Generating text from RDF data. In Proceedings of the 10th International Conference on Natural Language Generation, pages 124–133, Santiago de Compostela, Spain. Association for Computational Linguistics.

Ashim Gupta and Vivek Srikumar. 2021. X-fact: A new benchmark dataset for multilingual fact checking. In Proceedings of the 59th Annual Meeting of the Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 675–682, Online. Association for Computational Linguistics.

Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Ming-Wei Chang. 2020. Realm: Retrievalaugmented language model pre-training. Preprint, arXiv:2002.08909.

Lucie-Aimée Kaffee, Hady Elsahar, Pavlos Vougiouklis, Christophe Gravier, Frédérique Laforest, Jonathon Hare, and Elena Simperl. 2018. Learning to generate Wikipedia summaries for underserved languages from Wikidata. In Proceedings ofthe 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 640–645, New Orleans, Louisiana. Association for Computational Linguistics.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. Preprint, arXiv:2309.06180.

Rémi Lebret, David Grangier, and Michael Auli. 2016. Neural text generation from structured data with application to the biography domain. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 1203–1213, Austin, Texas. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2021. Retrieval-augmented generation for knowledgeintensive nlp tasks. Preprint, arXiv:2005.11401.

Dasha Metropolitansky and Jonathan Larson. 2025. Towards effective extraction and evaluation of factual claims. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6996–7045, Vienna, Austria. Association for Computational Linguistics.

Simon Mille, Massimiliano Pronesti, Craig Thomson, Michela Lorandi, Sophie Fitzpatrick, Rudali Huidrom, Mohammed Sabry, Amy O’Riordan, and Anya Belz. 2024. Filling gaps in Wikipedia: Leveraging data-to-text generation to improve encyclopedic coverage of underrepresented groups. In Proceedings ofthe 17th International Natural Language Generation Conference: System Demonstrations, pages 16–19, Tokyo, Japan. Association for Computational Linguistics.

Sewon Min, Kalpesh Krishna, Xinxi Lyu, Mike Lewis, Wen-tau Yih, Pang Koh, Mohit Iyyer, Luke Zettlemoyer, and Hannaneh Hajishirzi. 2023. FActScore: Fine-grained atomic evaluation of factual precision in long form text generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 12076–12100, Singapore. Association for Computational Linguistics.

Telmo Pires, Eva Schlinger, and Dan Garrette. 2019. How multilingual is multilingual BERT? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4996–5001, Florence, Italy. Association for Computational Linguistics.

Dwaipayan Roy, Sumit Bhatia, and Prateek Jain. 2020. A topic-aligned multilingual corpus of Wikipedia articles for studying information asymmetry in low resource languages. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 2373–2380, Marseille, France. European Language Resources Association.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 467 others. 2026. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

Yixiao Song, Yekyung Kim, and Mohit Iyyer. 2024. VeriScore: Evaluating the factuality of verifiable claims in long-form text generation. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 9447–9474, Miami, Florida, USA. Association for Computational Linguistics.

Gabriel Stanovsky, Julian Michael, Luke Zettlemoyer, and Ido Dagan. 2018. Supervised open information extraction. In Proceedings of the 2018 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pages 885–895, New Orleans, Louisiana. Association for Computational Linguistics.

Liyan Tang, Philippe Laban, and Greg Durrett. 2024. MiniCheck: Efficient fact-checking of LLMs on grounding documents. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 8818–8847, Miami, Florida, USA. Association for Computational Linguistics.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. Fever: a large-scale dataset for fact extraction and verification. Preprint, arXiv:1803.05355.

Pavlos Vougiouklis, Hady Elsahar, Lucie-Aimée Kaffee, Christophe Gravier, Frédérique Laforest, Jonathon Hare, and Elena Simperl. 2018. Neural wikipedian: Generating textual summaries from knowledge base triples. Journal ofWeb Semantics, 52-53:1–15.

Claudia Wagner, Eduardo Graells-Garrido, David Garcia, and Filippo Menczer. 2016. Women through the glass ceiling: gender asymmetries in wikipedia. EPJ Data Science, 5(1).

Yuxia Wang, Revanth Gangi Reddy, Zain Muhammad Mujahid, Arnav Arora, Aleksandr Rubashevskii, Jiahui Geng, Osama Mohammed Afzal, Liangming Pan, Nadav Borenstein, Aditya Pillai, Isabelle Augenstein, Iryna Gurevych, and Preslav Nakov. 2024. Factcheck-bench: Fine-grained evaluation benchmark for automatic fact-checkers. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 14199–14230, Miami, Florida, USA. Association for Computational Linguistics.

Miriam Wanner, Benjamin Van Durme, and Mark Dredze. 2025. DnDScore: Decontextualization and decomposition for factuality verification in long-form text generation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 23609–23626, Suzhou, China. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, and 41 others. 2025. Qwen3 technical report. Preprint, arXiv:2505.09388.

Jiebin Zhang, Eugene J. Yu, Qinyu Chen, Chenhao Xiong, Dawei Zhu, Han Qian, Mingbo Song, Weimin Xiong, Xiaoguang Li, Qun Liu, and Sujian Li. 2025. WIKIGENBENCH:exploring full-length Wikipedia generation under real-world scenario. In Proceedings ofthe 31st International Conference on Computational Linguistics, pages 5191–5210, Abu Dhabi, UAE. Association for Computational Linguistics.

Yuze Zhao, Jintao Huang, Jinghan Hu, Xingjun Wang, Yunlin Mao, Daoze Zhang, Hong Zhang, Zeyinzi Jiang, Zhikai Wu, Baole Ai, Ang Wang, Wenmeng Zhou, and Yingda Chen. 2025. Swift:a scalable lightweight infrastructure for fine-tuning. Preprint, arXiv:2408.05517.

## A Implementation Details

## A.1 Inference Setup

All open-weight model inference is run with the MS-SWIFT (Zhao et al., 2025) framework and the vLLM (Kwon et al., 2023) inference engine on a single NVIDIA A100 GPU with 80 GB of memory. We use this setup for open-weight generation, translation, claim extraction backbone comparisons, and classifier experiments unless otherwise specified. Closed-source GPT-5.1 (Singh et al., 2026) calls are made through the OpenRouter API <sup>4</sup> and are used for reference claim extraction, prompt-assisted dataset construction, and closedsource comparison settings described in the main paper.

## A.2 Experimental Cost

The open-weight model inference for enrichment generation requires only 2 hours with the help of vLLM, but the claim relation evaluation takes 4 hours per generated biography. The claim extraction pipeline with GPT-5.1 + X-Claimify on our CLAW-4L costs around \$ 300 via API calls, and the entire extraction procedure takes 10 hours.

## A.3 Decoding Settings and Token Budgets

Table 4 reports the token budgets and decoding settings used in our generation, translation, and LLM-based verification experiments. We use deterministic decoding for verification and modelspecific generation settings for biography rewriting and translation. Qwen and Gemma generation runs use thinking disabled.

<table><tr><td colspan="6">Token Budgets</td></tr><tr><td>Task</td><td>Prompt</td><td>EN</td><td>non-English</td><td>Out.</td><td>Seed</td></tr><tr><td>Translation</td><td>4,096</td><td>4,096</td><td></td><td>8,192</td><td>42</td></tr><tr><td>Enrichment</td><td>2,048</td><td>4,096</td><td>2,048</td><td>8,192</td><td>42</td></tr><tr><td>Verification</td><td>8,192</td><td></td><td></td><td>8,192</td><td>42</td></tr><tr><td colspan="6">Translation / Enrichment</td></tr><tr><td>Model</td><td>Temp.</td><td>Top-p</td><td>Top-k</td><td>Pres. Rep.</td><td></td></tr><tr><td>Qwen3.6-27B</td><td>0.7</td><td>0.80</td><td>20</td><td>1.5</td><td>1.0</td></tr><tr><td>Gemma-4-31B-IT Mistral-Small-3.2-24B-IT</td><td>1.0</td><td>0.95</td><td>64</td><td>0.0</td><td>1.0</td></tr><tr><td></td><td>0.15</td><td>1.00</td><td>50</td><td>0.0</td><td>1.0</td></tr><tr><td colspan="6">Verification</td></tr><tr><td>Model</td><td>Temp.</td><td>Top-p</td><td>Top-k</td><td>Pres.</td><td>Rep.</td></tr><tr><td>Qwen3.5-9B</td><td>0.0</td><td>1.00</td><td>0</td><td>0.0</td><td>1.0</td></tr></table>

Table 4: Hyperparameters for generation and LLMbased verification experiments. The upper panel reports token budgets, where EN denotes the English biography, non-English denotes the non-English biography or enrichment context, and Out. denotes the maximum number of generated tokens. The lower panels report decoding settings for generation and verification. Pres. denotes presence penalty and Rep. denotes repetition penalty. Qwen and Gemma generation runs use thinking disabled.

## B Datasets Construction Details

In this section, we present the details of the creation of our evaluation datasets: (i) CLAW-4L, a crosslingual biography pairs benchmark, (ii) CLAW-4L-CX, a cross-lingual sentence-claim benchmark, and (iii) CLAW-4L-RC, an English claim-pair relation judgment benchmark.

## B.1 Data Collection from Wikidata

We first collect country-specific female candidates from Wikidata using three eligibility constraints: each entity must be (1) a human, (2) marked as female, and (3) associated with the corresponding country of citizenship. In addition, we require that the entity have both an English Wikipedia sitelink and a sitelink in the non-English language. We instantiate this procedure for three country-language settings: France/French, China/Chinese, and Azerbaijan/Azerbaijani.

To support subsequent filtering and sampling, we intentionally build candidate pools that are substantially larger than the final benchmark size. For the French and Chinese settings, we cap the pool size at 1,500 entities each. For the Azerbaijani setting, only 434 Wikidata entities satisfy all constraints, and we therefore use the full eligible pool. The final candidate pool used for selection contains 3,434 entities in total.

For each retained entity, we additionally collect basic metadata from Wikidata, including the entity identifier, page titles, sitelinks, and occupation labels. These metadata are used only for filtering, identifying, and coarse occupation-aware sampling.

## B.2 Biography Retrieval and Cleaning

For each candidate, we retrieve the English and non-English-language Wikipedia pages through the MediaWiki API. Since our goal is to compare biography content across languages, we apply a lightweight normalization procedure to both pages before computing any statistics.

Specifically, we remove markup artifacts and non-biographical content that would otherwise inflate page length without contributing useful biography evidence. This includes infobox templates, formatting noise, and trailing sections such as references, bibliography, notes, and external links. After cleaning, we retain the main readable biography prose together with its section structure when available. This cleaning procedure is designed to improve comparability across languages rather than to produce perfectly parallel biographies. In particular, the cleaned texts may still differ substantially in discourse structure and content organization, but the most obvious length distortions caused by nonbiographical material are removed.

## B.3 Length-based Tiering and Language Calibration

After cleaning, we compute biography lengths using the same multilingual tokenizer (google/mt5-small). For a non-English language L, we first define the raw non-English-to-English token ratio for a biography pair as

$$
r = { \frac { \# \mathrm { t o k e n s } ( \mathrm { n o n - E N } ) } { \# \mathrm { t o k e n s } ( \mathrm { E N } ) } } .\tag{1}
$$

Because token counts produced by a shared multilingual tokenizer may still reflect language- and script-specific segmentation tendencies, raw ratios can conflate content richness with tokenization bias. To reduce this effect, we estimate a languagespecific inflation factor for each non-English language using parallel multilingual data from FLO-$\mathrm { R E S + . } ^ { 5 }$ Concretely, given aligned sentence pairs $( x _ { e n } , x _ { L } )$ , we compute

$$
b _ { L } = \mathbb { E } _ { ( x _ { e n } , x _ { L } ) } \left[ \frac { \# \mathrm { t o k e n s } ( x _ { L } ) } { \# \mathrm { t o k e n s } ( x _ { e n } ) } \right] ,\tag{2}
$$

where all token counts are obtained with the same tokenizer.

We then define the normalized biography-level ratio as

$$
r ^ { \mathrm { n o r m } } = \frac { r } { b _ { L } } = \frac { \# \mathrm { t o k e n s } ( \mathrm { n o n \mathrm { - } E N } ) } { \# \mathrm { t o k e n s } ( \mathrm { E N } ) } \cdot \frac { 1 } { b _ { L } } .\tag{3}
$$

Table 5 reports the resulting language-specific inflation factors. As expected, the estimated factors differ substantially across non-English languages, reflecting systematic tokenization differences even for semantically aligned text. Notably, the calibration factors are greater than 1 for French and Azerbaijani, but below 1 for Chinese, indicating that the shared tokenizer tends to allocate relatively more tokens to the former two languages and fewer to the latter when compared with English.

We use r<sup>norm</sup> as a coarse heuristic for non-English-side content richness. Importantly, it is not intended as a direct measure of factual quality or factual correctness; rather, it provides a more comparable cross-lingual signal for prioritizing cases in which the non-English biography is likely to contain additional textual evidence beyond the English page. Since r<sup>norm</sup> remains centered around 1.0 for content-equivalent text under the calibration corpus, we partition candidates into three tiers:

• Tier $\mathbf { A } \colon r ^ { \mathrm { n o r m } } \geq 1 . 2 0$

• Tier B: $1 . 0 0 \leq r ^ { \mathrm { n o r m } } < 1 . 2 0$

• Tier C: $r ^ { \mathrm { n o r m } } < 1 . 0 0$

We adopt a tiered design rather than a hard filter because biography length remains only a proxy for content richness. Tiering lets us prioritize non-English-richer candidates while preserving enough flexibility for downstream sampling decisions on occupation coverage and country-specific composition.

<table><tr><td>Pair</td><td>Inflation Factor  $b _ { L }$ </td></tr><tr><td>EN→ZH</td><td>0.91</td></tr><tr><td>EN→FR</td><td>1.41</td></tr><tr><td>EN→AZ</td><td>1.36</td></tr></table>

Table 5: Language-specific token inflation factors estimated from FLORES+ using the google/mt5-small tokenizer. Values above 1 indicate that the non-English language tends to produce more tokens than English for semantically aligned text; values below 1 indicate the opposite.

## B.4 Sentence Segmentation and Sentence-level Statistics

For sentence-level statistics, we segment biographies using a multilingual neural sentence segmentation model (SaT) implemented via wtpsplit<sup>6</sup>. This model provides consistent sentence boundary detection across languages without relying on language-specific punctuation rules.

We define the sentence-level ratio analogously as

$$
r _ { s } = { \frac { \# \mathrm { s e n t e n c e s } ( \mathrm { n o n - E N } ) } { \# \mathrm { s e n t e n c e s } ( \mathrm { E N } ) } } .\tag{4}
$$

We do not apply an analogous calibration to sentence-level ratios. Unlike token counts, sentence segmentation is inherently less stable across languages and is more sensitive to discourse structure, punctuation conventions, and editorial practices. In addition, large-scale paragraph-aligned multilingual corpora covering lower-resource languages are scarce, making reliable estimation of sentence-level calibration factors impractical. We therefore treat sentence counts and sentence ratios only as complementary structural signals rather than precise units of semantic comparison.

## B.5 Tier-aware Selection

In this step, we aim to select candidates for whom the non-English-language page is more likely to contain additional biography content beyond the English page, while also balancing the occupationand country-wise distribution.

As described in B.3, we assign all 3,434 candidates into three tiers by normalized ratio r<sup>norm</sup>. For each country, we select 300 instances as the source pool for downstream benchmark construction. The selection procedure is tier-aware: we first prioritize candidates from Tiers A and B, and use Tier C as a fallback only when necessary to complete the country quota.

Algorithm 1 Country-wise sampling procedure for   
CLAW-4L   
Require: country-specific candidate set C, non-English size   
N, minimum A+B ratio ρ   
1: Keep only candidates with both English and non-English   
biographies   
2: Infer one coarse occupation bucket for each candidate:   
artist, scientist, athlete, politician, or other   
3: Partition candidates into Tier A, Tier B, and Tier C   
4: Let $N _ { A B } = \operatorname* { m i n } ( | A \cup B | , \lceil \rho N \rceil )$   
5: Select $N _ { A B }$ candidates from Tier A/B   
6: if occupation-aware balancing is enabled then   
7: Distribute the A/B non-English as evenly as possible   
across artist, scientist, athlete, and politician   
8: Fill any unassigned quota from the remaining Tier   
A/B pool   
9: else   
10: Select from Tier A first, then Tier B   
11: end if   
12: If fewer than N candidates have been selected, continue   
adding remaining Tier A/B candidates   
13: If the quota is still not met, fill the remaining slots from   
Tier C   
14: return final selected set

<table><tr><td>Country</td><td>Before</td><td>After</td></tr><tr><td>France</td><td>1500</td><td>300</td></tr><tr><td>China</td><td>1500</td><td>300</td></tr><tr><td>Azerbaijan</td><td>434</td><td>300</td></tr><tr><td>Total</td><td>3434</td><td>900</td></tr></table>

Table 6: Candidate-pool sizes change before/after tieraware selection.

Within the prioritized Tier A/B subset, we apply coarse occupation-aware sampling. Specifically, we pre-define four biography-rich occupation groups—artist, athlete, scientist, and politician— and infer for each candidate a single occupation bucket from its Wikidata occupation labels using keyword matching. When possible, the required Tier A/B quota is distributed approximately evenly across these four groups. This balancing is approximate rather than strict: if one group does not contain enough eligible A/B candidates, the remaining quota is filled from the rest of the available Tier A/B pool. Candidates outside the four focus groups are assigned to an other bucket. These candidates are not explicitly balanced during the initial A/B allocation, but they remain eligible for final selection when filling the remaining quota.

Algorithm 1 summarizes the country-wise sampling procedure. Table 6 reports the candidate-pool size changes.

## B.6 Stratified Biography Sampling for CLAW-4L

To obtain the final biography-claim benchmark, CLAW-4L, we design a stratified sampling strategy to reduce the dataset to 300 instances (100 per country), while maintaining the same data distribution as in the remaining 900-instance candidate pool.

Matched attributes. Within each country, we aimed to preserve the empirical distribution of the following attributes: (i) occupation bucket, (ii) selection tier, (iii) normalized non-English-to-English token ratio, (iv) non-English-side token length, (v) nationality-count bucket, and (vi) non-English-side infobox presence. Occupation buckets consist of scientist, athlete, politician, artist, and other. Nationality multiplicity was bucketed as 1, 2, or 3+. Infobox availability was treated as a binary attribute indicating whether non-empty structured infobox information was present on the non-English side.

non-English distributions. For each countryspecific 300-instance pool, we derived a 100-instance non-English distribution for every matched attribute. For categorical variables, we scaled empirical counts from 300 to 100 and converted the resulting fractional allocations into integers using the largest-remainder method. This preserves the original proportions as closely as possible under the fixed sample size constraint.

For continuous variables, we did not attempt to match only the mean or variance. Instead, we discretized values into country-specific quantile bins and matched the empirical histogram over bins. Specifically, we used quantile-based bins for normalized token ratio and non-English-side token count, then applied the same largest-remainder procedure to obtain integer non-English counts for each bin. This design preserves the shape of the source distribution more faithfully than moment matching alone.

Constrained subset optimization. Given the non-English distributions, we selected a subset of size 100 for each country using constrained local search. Occupation counts were enforced exactly as a hard constraint. We first generated an initial feasible subset by randomly sampling the required number of instances from each occupation bucket. We then refined this subset by repeatedly swapping within-occupation, replacing one selected instance with an unselected instance from the same occupation bucket. This guarantees that occupation counts remain fixed throughout optimization.

All remaining attributes were optimized jointly using a weighted mismatch objective:

$$
\begin{array} { r l } & { \mathcal { L } ( S ) = \lambda _ { \mathrm { t i e r } } d _ { \mathrm { t i e r } } ( S ) + \lambda _ { \mathrm { r a t i o } } d _ { \mathrm { r a t i o } } ( S ) } \\ & { ~ + \lambda _ { \mathrm { t o k } } d _ { \mathrm { t o k } } ( S ) + \lambda _ { \mathrm { n a t } } d _ { \mathrm { n a t } } ( S ) } \\ & { ~ + \lambda _ { \mathrm { i n f o } } d _ { \mathrm { i n f o } } ( S ) , } \end{array}\tag{5}
$$

where S is a candidate subset of size 100. Here, $d _ { \mathrm { t i e r } } , d _ { \mathrm { r a t i o } } , d _ { \mathrm { t o k } }$ , and $d _ { \mathrm { n a t } }$ are normalized $L _ { 1 }$ distances between the sampled and non-English count distributions for selection tier, normalized tokenratio bins, non-English-token bins, and nationalitycount bins, respectively, and $d _ { \mathrm { i n f o } }$ is the absolute difference in non-English-infobox presence rate. Occupation is excluded from the objective because it is enforced exactly.

In our implementation, we used $\lambda _ { \mathrm { t i e r } } ~ = ~ 4 . 0$ $\lambda _ { \mathrm { r a t i o } } = 2 . 0 , \lambda _ { \mathrm { t o k } } = 1 . 5 , \lambda _ { \mathrm { n a t } } = 1 . 0 $ , and $\lambda _ { \mathrm { i n f o } }$ 0.8, reflecting the priority of preserving the original tier structure while also maintaining close agreement on length- and metadata-related attributes.

Search procedure. Because this subset-selection problem is combinatorial, we used a multi-start local search strategy. For each country, we initialized the search from multiple random occupationmatched subsets and ran a fixed number of local swap steps from each initialization. We retained the subset with the lowest objective value across all runs. In practice, this procedure consistently found samples that matched the source distributions exactly or near-exactly on all tracked dimensions.

Outcome. The resulting 300-instance CLAW-4L closely tracks the composition of the 900-instance pool. Across all three country settings, occupation and selection-tier distributions were exactly matched. The remaining distributions over length bins, nationality-count bins, and infobox availability were also matched exactly in most cases, with only very small residual deviations in a few settings. This sampled biography pool was then used as the source set for constructing CLAW-4L-CX, our sentence-level human annotation benchmark.

We summarize the occupation-bucket distribution of the CLAW-4L in Table 7 and compare token- and sentence-level statistics between the 900-instance pool and the sampled CLAW-4L in Table 8.

## B.7 Draft Claim Generation for CLAW-4L-CX

Starting from CLAW-4L, we extract claims from both the English and non-English biographies using GPT-5.1. The extracted claims are served only as sentence selection signals for CLAW-4L-CX and drafts for human revisions.

In addition to producing a decontextualized English claim, the prompt is also designed to output a lightweight structured representation for each extracted claim. Concretely, each claim is parsed into the following fields:

$$
\begin{array} { c } { ( s \mathrm { u b j e c t , p r e d i c a t e , o b j e c t , t i m e } , } \\ { 1 \mathrm { o c a t i o n , r e a s o n , m a n n e r , h e d g e } ) } \end{array}
$$

This structured output (infobox) serves as an auxiliary representation for downstream benchmark construction. In particular, it supports structured claim editing during human annotation, and controlled relation construction in CLAW-4L-RC such as contradiction and asymmetric enrichment. An output example is shown below:

" subject ": " Maya Angelou ",   
" predicate ": " traveled ",   
" object ": " California " ,   
" time ": "1940" ,   
" location ": " California ",   
" reason ": " to join her mother " ,   
" manner ": "by train ",   
" hedge ": " according to some   
accounts ",   
" claim ": " According to some   
accounts , Maya Angelou traveled   
by train to California in 1940   
to join her mother ."   
}

## B.8 Sentence Selection for CLAW-4L-CX

After claim extraction, we selected one sentence from each biography in CLAW-4L to construct the sentence-claim benchmark CLAW-4L-CX. Our selection procedure was designed to balance two goals: preserving the natural distribution of sentence-level claim complexity in CLAW-4L, while enriching the final selected set with sentences whose extracted claims are more faithful, informative, and easier to validate.

Candidate sentences and claim-count buckets. For each biography, we treated every source sentence that yielded at least one extracted claim as a candidate annotation sentence. Let $S _ { b }$ denote the set of candidate sentences for biography $b ,$ and let $C ( s )$ denote the set of extracted claims associated with sentence $s \in S _ { b }$ . We first grouped candidate sentences into four claim-count buckets according to $| C ( s )$ |:

<table><tr><td>Country</td><td>N</td><td>Artist</td><td>Athlete</td><td>Politician</td><td>Scientist</td><td>Other</td></tr><tr><td>France</td><td>100</td><td>32 (32.00%)</td><td>26 (26.00%)</td><td>26 (26.00%)</td><td>10 (10.00%)</td><td>6 (6.00%)</td></tr><tr><td>China</td><td>100</td><td>30 (30.00%)</td><td>35 (35.00%)</td><td>19 (19.00%)</td><td>3 (3.00%)</td><td>13 (13.00%)</td></tr><tr><td>Azerbaijan</td><td>100</td><td>32 (32.00%)</td><td>29 (29.00%)</td><td>7 (7.00%)</td><td>7 (7.00%)</td><td>25 (25.00%)</td></tr><tr><td>Total</td><td>300</td><td>94 (31.33%)</td><td>90 (30.00%)</td><td>52 (17.33%)</td><td>20 (6.67%)</td><td>44 (14.67%)</td></tr></table>

Table 7: Occupation-bucket distribution in CLAW-4L, reported by country and in total. We consider four major categories (artist, athlete, politician, scientist), with remaining biographies grouped as other.
<table><tr><td>Split</td><td>Pair</td><td>N</td><td>Tokens (EN / non-EN)</td><td>Mean  $r _ { t }$ </td><td> $\mathbf { M e a n } \ r _ { t } ^ { n o r m }$ </td><td>Sent (EN / non-EN)</td><td>Mean  $r _ { s }$ </td></tr><tr><td rowspan="4">Candidate Pool</td><td>EN→FR</td><td>300</td><td>479.22 / 1673.88</td><td>5.88</td><td>4.16</td><td>16.02 / 42.53</td><td>4.47</td></tr><tr><td>EN→ZH</td><td>300</td><td>498.84 / 1090.05</td><td>2.75</td><td>3.00</td><td>16.92 / 28.59</td><td>2.24</td></tr><tr><td>EN→AZ</td><td>300</td><td>543.36 / 1713.26</td><td>4.43</td><td>3.26</td><td>18.43 / 47.05</td><td>3.74</td></tr><tr><td>EN→X</td><td>900</td><td>507.14 /1492.39</td><td>4.35</td><td>3.47</td><td>17.12 / 39.39</td><td>3.48</td></tr><tr><td rowspan="4">CLAW-4L</td><td>EN→FR</td><td>100</td><td>465.44 / 1867.14</td><td>5.92</td><td>4.19</td><td>16.08 / 48.41</td><td>4.64</td></tr><tr><td>EN→ZH</td><td>100</td><td>489.23 / 1041.28</td><td>2.60</td><td>2.83</td><td>15.75 / 27.18</td><td>2.01</td></tr><tr><td>EN→AZ</td><td>100</td><td>544.17 / 1967.12</td><td>4.35</td><td>3.20</td><td>19.37 / 55.21</td><td>3.48</td></tr><tr><td>EN→X</td><td>300</td><td>499.61 / 1625.18</td><td>4.29</td><td>3.41</td><td>17.07 / 43.60</td><td>3.38</td></tr></table>

Table 8: Main token- and sentence-level statistics for the 900-instance candidate pool and CLAW-4L. Raw token ratios $r _ { t }$ denote non-English-to-English token ratios, $r _ { t } ^ { n o r m }$ denotes ratios normalized by language-specific inflation factors estimated from parallel data, and sentence ratios $r _ { s }$ denote non-English-to-English sentence ratios computed from cleaned biographies using a multilingual neural sentence segmentation model (SaT).

EN: Before Sentences (x) vs Total Claims (y)  
![](images/ac07f52b26e4a8692e859c85a18a057ab516243b9ee086efdb0b632ea11b0a6b.jpg)

Target: Before Sentences (x) vs Total Claims (y)  
![](images/f22b5312edeecf166e35c4a3a545f41418d3d946ab7b8fdd127bc673c8841516.jpg)  
Figure 4: Relationship between source sentence count (x) and extracted claim count (y) per biography in CLAW-4L (EN on top, non-English language on bottom). Each point denotes one biography; dashed lines show least-squares fits (EN: y = 1.70x + 1.88, $R ^ { 2 }$ = 0.961; non-English: $y = 1 . 8 2 x + 9 . 1 6 , R ^ { 2 } = 0 . 9 4 1 )$ Claim extraction generally yields more claims than source sentences, with mean claims-per-sentence ratios of 1.89 (EN) and 2.14 (non-English).

Claim Count Distribution (EN vs Target)  
![](images/a4c3ac18ee018183d8125c0dad08e74141c6e279498690b5485e0d89790f17a2.jpg)  
Figure 5: Distribution of extracted claim counts per biography in CLAW-4L, comparing English (EN) and non-English (Target) biographies in a single overlaid histogram (shared bins; y-axis shows instance share). EN is more concentrated at lower claim counts, while non-English biographies exhibit a heavier right tail, indicating more high-claim instances.

<table><tr><td>Lang</td><td>Country</td><td>N</td><td>Total Claims</td><td>Mean Claims</td><td>Median Claims</td><td>Mean Sentence Coverage</td></tr><tr><td rowspan="4">EN</td><td>Overall</td><td>300</td><td>9261</td><td>30.87</td><td>20.00</td><td>0.966</td></tr><tr><td>Azerbaijan</td><td>100</td><td>3359</td><td>33.59</td><td>24.00</td><td>0.964</td></tr><tr><td>China</td><td>100</td><td>2926</td><td>29.26</td><td>19.50</td><td>0.967</td></tr><tr><td>France</td><td>100</td><td>2976</td><td>29.76</td><td>17.00</td><td>0.968</td></tr><tr><td rowspan="4">non-EN</td><td>Overall</td><td>300</td><td>26798</td><td>89.33</td><td>43.00</td><td>0.960</td></tr><tr><td>Azerbaijan</td><td>100</td><td>9379</td><td>93.79</td><td>45.50</td><td>0.947</td></tr><tr><td>China</td><td>100</td><td>7237</td><td>72.37</td><td>37.00</td><td>0.979</td></tr><tr><td>France</td><td>100</td><td>10182</td><td>101.82</td><td>42.00</td><td>0.955</td></tr></table>

Table 9: Claim extraction statistics for CLAW-4L. Mean Sentence Coverage averages, over biography instances, the fraction of source sentences that yield at least one extracted claim.

![](images/c5c55386a37cd1c477d8475a071f038850c0d9c5c6495d75bb49b5fd5643c853.jpg)  
Figure 6: Log-scale comparison of extracted claim counts per biography between English and non-English texts (Target) in CLAW-4L. Each point is one biography; the diagonal dashed line indicates parity $( y = x )$ Most points lie above the diagonal, showing that the non-English biographies generally yield more extracted claims than their English counterparts, while the log scale improves visibility across the long-tailed count range.

$$
\begin{array} { r l } & { \mathcal { B } _ { 1 } = \{ s : | C ( s ) | = 1 \} , } \\ & { \mathcal { B } _ { 2 } = \{ s : | C ( s ) | = 2 \} , } \\ & { \mathcal { B } _ { 3 } = \{ s : | C ( s ) | = 3 \} , } \\ & { \mathcal { B } _ { 4 } = \{ s : | C ( s ) | \geq 4 \} . } \end{array}
$$

This four-way partition provided a simple but effective proxy for sentence-claim complexity, separating simple, medium, and claim-dense sentences while avoiding overly sparse high-count buckets.

Selection features. For each candidate sentence $s ,$ we computed three sentence-level quality signals.

(1) Claim coverage. To estimate how well the extracted claims collectively cover the factual content of the source sentence, we concatenated all claims in $C ( s )$ into a single text string concat $\left( C ( s ) \right)$ , and computed multilingual sentence similarity with LaBSE:

$$
\begin{array} { r } { \mathrm { c o v } ( s ) = \cos \Big ( \phi _ { \mathrm { L a B S E } } ( s ) , \phantom { x x x x x x x x x x x x x x x x x x x x x x x x x x x x x } } \\ { \phi _ { \mathrm { L a B S E } } \big ( \mathrm { c o n c a t } ( C ( s ) ) \big ) \Big ) . } \end{array}\tag{6}
$$

where $\phi _ { \mathrm { L a B S E } } ( \cdot )$ denotes the sentence embedding function. Higher values indicate that the extracted claims more faithfully capture the factual core of the source sentence.

(2) Fact density. We next estimated how much structured factual information a sentence contributes relative to its length. For a claim $c \in C ( s )$ 2 let

$$
\begin{array} { r } { F ( c ) = \{ \mathrm { s u b j e c t , p r e d i c a t e , o b j e c t , t i m e } , } \\ { \mathrm { l o c a t i o n , r e a s o n , m a n n e r } , \mathrm { h e d g e } \} . } \end{array}
$$

We define an indicator $\mathbf { 1 } [ f ( c ) ]$ that equals 1 if field $f$ is non-empty, and 0 otherwise. For the hedge field, we set $\mathbf { 1 } [ \mathsf { h e d g e } ( c ) ] = 1$ only when the hedge value is not $" \mathsf { N o } ^ { \prime \prime }$ . We exclude the free-form text fields claim and source\_sent from this computation. Sentence-level fact density is then defined as

$$
\mathrm { d e n s } ( s ) = \frac { \sum _ { c \in C ( s ) } \sum _ { f \in F ( c ) } \mathbf { 1 } [ f ( c ) ] } { | s | _ { \mathrm { t o k } } } ,
$$

where $| s | _ { \mathrm { t o k } }$ is the token length of the source sentence. Higher values indicate that more structured factual information is packed into the sentence.

(3) Rewrite cost. Finally, we quantified how much rewriting was required to transform the source sentence into its decontextualized claims. For each claim $c \in C ( s )$ , we computed a normalized token-level edit distance

$$
\operatorname { n e d } ( s , c ) = \frac { \operatorname { E D } ( s , c ) } { \operatorname* { m a x } ( | s | _ { \operatorname { t o k } } , | c | _ { \operatorname { t o k } } ) } ,
$$

where $\operatorname { E D } ( \cdot , \cdot )$ denotes token-level Levenshtein edit distance. We then define sentence-level rewrite cost as the mean normalized edit distance across claims:

$$
\operatorname { r e w } ( s ) = { \frac { 1 } { | C ( s ) | } } \sum _ { c \in C ( s ) } \operatorname { n e d } ( s , c ) .
$$

Lower values indicate that the sentence can be converted into decontextualized claims with less rewriting, while larger values suggest stronger contextual dependence or more substantial reformulation.

Instance-level rank aggregation. Because these three features are measured on different scales, we did not combine them directly. Instead, for each biography b, we converted the candidate sentences in $S _ { b }$ into within-biography normalized rank scores in [0, 1]. Let $n _ { b } ~ = ~ | S _ { b } |$ . For each metric $m \ \in \ \{ \mathrm { c o v } $ , dens, rew}, we first assign each candidate sentence $s \in S _ { b }$ a rank ${ \mathrm { r a n k } } _ { m } ( s ) \in$ $\{ 1 , \dots , n _ { b } \}$ , where larger ranks always indicate better candidates. Thus, for claim coverage and fact density, higher raw values correspond to larger ranks, whereas for rewrite cost, lower raw values correspond to larger ranks. We then normalize these ranks as

$$
R _ { m } ( s ) = { \frac { \operatorname { r a n k } _ { m } ( s ) - 1 } { n _ { b } - 1 } } .
$$

This yields $R _ { m } ( s ) \in [ 0 , 1 ]$ , with 1 assigned to the best-ranked sentence and 0 to the worst-ranked sentence for that metric within the same biography. If a biography yields only one candidate sentence, we assign that sentence a normalized rank score of 1 for all metrics by definition. Let these normalized rank scores be denoted

$$
R _ { \mathrm { c o v } } ( s ) , \quad R _ { \mathrm { d e n s } } ( s ) , \quad R _ { \mathrm { r e w } } ( s ) .
$$

We then defined the overall sentence quality score as an unweighted rank aggregation:

$$
\operatorname { s c o r e } ( s ) = { \frac { R _ { \mathrm { c o v } } ( s ) + R _ { \mathrm { d e n s } } ( s ) + R _ { \mathrm { r e w } } ( s ) } { 3 } } .
$$

We used equal weights because no sentence-level development annotations were available for tuning feature importance, and the three signals were intended to play complementary roles: coverage captures faithfulness, density captures informational richness, and rewrite cost captures annotation difficulty.

Global complexity balancing. Selecting the highest-scoring sentence independently for each biography would tend to over-favor simpler oneclaim sentences. To preserve realism while maintaining sufficient complexity diversity, we therefore imposed global constraints on the distribution of claim-count buckets in the final selected set.

We began by computing the empirical distribution of candidate sentences across the four buckets $\boldsymbol { B } _ { 1 } , \ldots , \boldsymbol { B } _ { 4 }$ within each language side (English and non-English). We then applied a hard-coded bucket balance strategy: all claim buckets $( 1 - , 2 \bar { } , 3 \bar { } , 4 + -$ claim) share the same proportion (0.25) in the final CLAW-4L-CX. This design ensures that the final sentence set contained enough medium- and highcomplexity examples to evaluate extraction quality beyond the easiest cases.

Constrained final selection. Let $x _ { b , s } \in \{ 0 , 1 \}$ indicate whether candidate sentence $s \in S _ { b }$ is selected for biography b. We selected the final sentence set by maximizing the total sentence quality subject to two types of constraints:

$$
\operatorname* { m a x } _ { x } \sum _ { b } \sum _ { s \in S _ { b } } \operatorname { s c o r e } ( s ) x _ { b , s }
$$

subject to

$$
\sum _ { s \in S _ { b } } x _ { b , s } = 1 \qquad \forall b ,
$$

so that exactly one sentence is selected from each biography, and

$$
\sum _ { b } \sum _ { s \in S _ { b } : g ( s ) = k } x _ { b , s } = T _ { k } \qquad \forall k \in \{ 1 , 2 , 3 , 4 \} ,
$$

where $g ( s )$ is the claim-count bucket of sentence s, and $T _ { k }$ is the target number of selected sentences assigned to bucket k after the mild rebalance step described above.

In practice, this constrained selection procedure yielded a sentence set that remained close to the natural complexity distribution of the sampled biography pool while avoiding over-representation of trivial one-claim sentences. Sentence selection was performed independently for English and non-English biographies; consequently, the final EN and non-English sentences in CLAW-4L-CX are not sentence-aligned within biography pairs. The resulting 300 English and 300 non-Englishlanguage sentences constitute CLAW-4L-CX, our sentence-level human annotation benchmark.

## B.9 CLAW-4L-CX Statistics

Table 11 summarizes the claim-count distribution of the final selected sentences in CLAW-4L-CX. By construction, the benchmark contains exactly one selected sentence from each sampled biography, yielding 300 English and 300 non-Englishlanguage sentences overall. Importantly, the final sentence set does not collapse onto only the simplest one-claim cases. On the English side, the selected sentences contain, on average, 2.63 claims, with 25 instances per country across the 1-, 2-, 3-, and 4+-claim buckets. On the non-English side, the corresponding mean is 2.72 claims, with 25 instances per claim bucket and per country. This confirms that the final benchmark preserves a meaningful range of sentence-level complexity, including both relatively simple and claim-dense cases.

Table 12 shows that this diversity is preserved without sacrificing sentence quality. Compared with CLAW-4L, the final selected sentences in CLAW-4L-CX exhibit consistent positive shifts across all three normalized ranking dimensions. On the English side, the selected set improves by +0.4107 in coverage rank, +0.2095 in fact-density rank, and +0.1965 in rewrite-rank, yielding an overall score shift of +0.2722. On the non-English side, the corresponding gains are +0.3285, +0.2221, and +0.4030, with an overall improvement of +0.3179. These shifts show that the selected sentences are not merely representative of the claim-count distribution; they are also systematically better candidates for human annotation. Specifically, higher coverage indicates that the extracted claims more faithfully capture the factual content of the source sentence, higher fact density indicates that the sentence contains richer, structured factual content, and higher rewrite-rank (equivalently, lower rewrite cost) indicates that the selected sentences require less aggressive reformulation into decontextualized claims, making them easier to verify and refine during annotation.

Taken together, these results suggest that CLAW-4L-CX achieves the intended trade-off in dataset construction. It preserves sentence-level diversity by covering a broad range of claim-count buckets while simultaneously enriching for higher-quality claim-bearing sentences. This makes it a suitable sentence-claim benchmark for evaluating claim extraction quality against human annotation.

## B.10 CLAW-4L-CX Annotators and Interface

Annotators were volunteer researchers recruited through the authors’ academic and professional networks and were not financially compensated. Participation was voluntary, and annotators were informed that the annotations would be used for research.

We provide our annotation interfaces in Figure 7 and Figure 8.

## B.11 Post-hoc Inter-Annotator Agreement for CLAW-4L-CX

After selecting the claim-pair classifier described in Section 5.1, we use it for a post-hoc agreement analysis of the CLAW-4L-CX annotations. For each language, we compare the reference claim sets produced by different annotators for the same sentence. Given two annotator claim sets ${ \mathcal { C } } ^ { ( a ) }$ and ${ \mathcal { C } } ^ { ( b ) }$ we score all cross-set claim pairs with the classifier. For each criterion $k \in$ {aligned, exact-aligned}, let $E _ { k } ^ { a b }$ be the set of claim pairs satisfying that criterion, where aligned counts exact and partial alignments and exact-aligned counts only exact matches. We compute existence-based set precision and recall:

$$
\begin{array} { l } { { P _ { k } ^ { a b } = \displaystyle \frac { \left| \{ i : \exists j , ~ ( i , j ) \in E _ { k } ^ { a b } \} \right| } { \left| \mathcal { C } ^ { ( a ) } \right| } , } } \\ { { R _ { k } ^ { a b } = \displaystyle \frac { \left| \{ j : \exists i , ~ ( i , j ) \in E _ { k } ^ { a b } \} \right| } { \left| \mathcal { C } ^ { ( b ) } \right| } . } } \end{array}
$$

We then compute $F _ { 1 , k } ^ { a b } = 2 P _ { k } ^ { a b } R _ { k } ^ { a b } / ( P _ { k } ^ { a b } + R _ { k } ^ { a b } )$ For each language, we report the average precision, recall, and F1 over all annotator pairs for all multiply annotated sentences in that language. The results are shown in Table 10.

![](images/88e8528bd26d20cc24e095a45ac445c4cf81cfd566bd6841d80e0ee04d7fccb1.jpg)  
Figure 7: Human annotation interface for CLAW-4L-CX (1).

![](images/1e061c533f35b222d4e18e3d7154f3735969d6d85ab36d982484b5a7ff26b893.jpg)  
Figure 8: Human annotation interface for CLAW-4L-CX (2).

<table><tr><td rowspan="2">Lang</td><td colspan="3">Aligned</td><td colspan="3">Exact-Aligned</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td>AZ</td><td>91.4</td><td>91.0</td><td>91.1</td><td>74.5</td><td>73.9</td><td>73.9</td></tr><tr><td>EN</td><td>98.9</td><td>100.0</td><td>99.3</td><td>81.2</td><td>80.7</td><td>80.3</td></tr><tr><td>FR</td><td>98.1</td><td>98.5</td><td>98.0</td><td>87.6</td><td>85.8</td><td>85.7</td></tr><tr><td>ZH</td><td>98.6</td><td>98.7</td><td>98.6</td><td>81.7</td><td>82.9</td><td>81.9</td></tr></table>

Table 10: Post-hoc inter-annotator agreement for CLAW-4L-CX reference claim annotations. We compare claim sets produced by different annotators for the same sentence using the selected claim-pair classifier. A claim is counted as matched if it has at least one aligned or exact-aligned counterpart in the other annotator’s set. Scores are reported as percentages.

## B.12 Construction of CLAW-4L-RC

To evaluate claim-pair relations across English and non-English biographies, we construct CLAW-4L-RC, a 600-instance benchmark built from the same GPT-5.1 claim extraction outputs used in CLAW-4L-CX. Each instance consists of one claim from the English biography, one claim from the paired non-English biography, their structured claim fields, and a manually validated relation label. Because both sides are normalized into English claim verbalizations at the extraction stage, the claim-pair matching stage operates over English claim pairs, even though the underlying evidence comes from cross-lingual biography pairs.

Aligned seed pairs. We begin by mining candidate aligned pairs from the full claim sets extracted from CLAW-4L. For each paired biography, we compute LaBSE cosine similarity between every English-side claim and every non-English-side claim using their claim texts, and use the highestscoring pairs as alignment seeds. This screening step is used only to reduce the manual search space over the large Cartesian product of candidate claim pairs within each biography pair.

To ensure that the resulting positives are not dominated by trivial or repetitive templates, we apply diversity-aware filtering before final validation. In particular, we balance the final aligned subset across the three country groups, yielding 33 Chinese, 33 Azerbaijani, and 34 French pairs, and additionally constrain the distribution of claim lengths as well as over-represented life-event patterns such as birth- and death-related claims. The resulting 100 candidate pairs are then manually validated. During validation, annotators verify both the claim texts and their structured fields, so the final aligned positives are supported at both the textual and structured levels.

Relation schema. Each claim pair is labeled in two stages. First, we assign an alignment label from:

• Aligned: the two claims describe the same factual aspect and can both be true;

• Contradicted: the two claims describe the same factual aspect but assert incompatible values or states;

• Not Relevant: the two claims describe different factual aspects, even if they concern the same person.

If and only if the pair is labeled Aligned, we further assign an enrichment label from:

• A=B: neither claim contains meaningful extra atomic factual content beyond the other;

• A>B: Claim A contains the content of Claim B and adds additional non-conflicting detail;

• B>A: Claim B contains the content of Claim A and adds additional non-conflicting detail;

• A↔B: the two claims are aligned and non-contradictory, but each contains nonconflicting factual detail missing from the other.

We provide Table 13 to illustrate the examples of different claim pair relations.

Construction of non-positive relations. Starting from the manually validated aligned pairs, we construct the remaining relation types through controlled structured editing and GPT-5.1 rewriting.

For Contradicted pairs, we preserve the same underlying claim anchor while changing one field value to an incompatible alternative. Candidate replacement values are drawn from field-specific value pools built from the extracted claims in the sampled biography pool. In practice, we focus on fields whose modification naturally induces contradiction, such as time, object, and, in some cases, location.

For Not Relevant pairs, we pair claims that do not describe the same factual aspect. These include both easier negatives, obtained from claims drawn from different biography instances, and harder negatives, obtained from claims within the same biography that share the subject but refer to different events, roles, works, or attributes. We additionally create a subset of such examples through controlled field perturbation followed by GPT-5.1 rewriting.

<table><tr><td>Lang</td><td>Country</td><td>N</td><td>Mean Nclaims</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td></tr><tr><td rowspan="4">EN</td><td>Azerbaijan</td><td>100</td><td>2.67</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>China</td><td>100</td><td>2.61</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>France</td><td>100</td><td>2.60</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>Global</td><td>300</td><td>2.63</td><td>75</td><td>75</td><td>75</td><td>75</td></tr><tr><td rowspan="4">non-EN</td><td>Azerbaijan</td><td>100</td><td>2.68</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>China</td><td>100</td><td>2.84</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>France</td><td>100</td><td>2.63</td><td>25</td><td>25</td><td>25</td><td>25</td></tr><tr><td>Global</td><td>300</td><td>2.72</td><td>75</td><td>75</td><td>75</td><td>75</td></tr></table>

Table 11: Claim-count distribution of the final selected sentences in CLAW-4L-CX. Within each language side, rows report country-level statistics followed by a global summary. Here, B1, B2, B3, and B4 denote sentences yielding 1, 2, 3, and 4 or more extracted claims, respectively.
<table><tr><td>Lang</td><td>Country</td><td> $N _ { \mathrm { b e f o r e } }  N _ { \mathrm { a f t e r } }$ </td><td> $\Delta R _ { \mathrm { c o v } }$  ↑</td><td> $\Delta R _ { \mathrm { d e n s } }$  ←</td><td> $\Delta R _ { \mathrm { r e w } }$  ↑</td><td>∆Score ↑</td></tr><tr><td rowspan="4">EN</td><td>Azerbaijan</td><td>1837 → 100</td><td>0.4004</td><td>0.1347</td><td>0.2813</td><td>0.2721</td></tr><tr><td>China</td><td>1502 → 100</td><td>0.4127</td><td>0.2447</td><td>0.1714</td><td>0.2763</td></tr><tr><td>France</td><td>1523 → 100</td><td>0.4190</td><td>0.2490</td><td>0.1367</td><td>0.2682</td></tr><tr><td>Global</td><td>4862 → 300</td><td>0.4107</td><td>0.2095</td><td>0.1965</td><td>0.2722</td></tr><tr><td rowspan="4">non-EN</td><td>Azerbaijan</td><td>5141 → 100</td><td>0.3177</td><td>0.2201</td><td>0.4190</td><td>0.3189</td></tr><tr><td>China</td><td>2635 → 100</td><td>0.2522</td><td>0.2316</td><td>0.3657</td><td>0.2832</td></tr><tr><td>France</td><td>4511 → 100</td><td>0.4154</td><td>0.2144</td><td>0.4241</td><td>0.3513</td></tr><tr><td>Global</td><td>12287 → 300</td><td>0.3285</td><td>0.2221</td><td>0.4030</td><td>0.3179</td></tr></table>

Table 12: Shifts in normalized rank-based sentence selection statistics from CLAW-4L to the final selected sentences in CLAW-4L-CX. Here, $N _ { \mathrm { b e f o r e } }$ and $N _ { \mathrm { a f t e r } }$ denote the numbers of candidate and selected sentences, respectively. We report ∆ = after − before for normalized coverage rank, density rank, rewrite-rank, and the final aggregated score, where larger positive values indicate a better trade-off between informativeness and editable difficulty of the selected set.
<table><tr><td>Category</td><td>Rel.</td><td>Claim A</td><td>Claim B</td></tr><tr><td rowspan="4">Alig.</td><td>A=B</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School during her high school years.</td><td>Libby Lee attended Ying Wa Girls&#x27; School during her senior secondary school years.</td></tr><tr><td>A&gt;B</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School in Mid-Levels during her high school years.</td><td>Libby Lee attended Ying Wa Girls&#x27; School during her senior secondary school years.</td></tr><tr><td>B&gt;A</td><td>Libby Lee attended Ying Wa Girls&#x27; School during her senior secondary school years.</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School in Mid-Levels during her high school years.</td></tr><tr><td>A↔B</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School in Mid-Levels during her high school years.</td><td>Libby Lee attended Ying Wa Girls’ School in 1991.</td></tr><tr><td>Contr.</td><td>A⊥B</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School during her high school years.</td><td>Libby Lee attended Diocesan Girls&#x27; School during her senior secondary school years.</td></tr><tr><td>Not Rel.</td><td>A-B</td><td>Libby Lee Ha-yun attended Ying Wa Girls&#x27; School during her high school years.</td><td>In 1957, Huang Yifan died in London, England, at the age of 61.</td></tr></table>

Table 13: Examples of claim relations. Orange text highlights the additional details that appear only on one side of the claim. Red text highlights the contradictory key words between two claims.

For aligned pairs with non-symmetric detail, we construct one-sided and mutual enrichment cases by varying the amount of non-conflicting information on the two sides. We create A>B examples by deleting minor fields such as reason, manner, or hedge, or by coarsening values in fields such as time and location and obtain B>A by swapping claim A and claim B in A>B. We create A↔B examples by ensuring that each side retains at least one non-conflicting atomic detail that is absent from the other. After structured modification, GPT-5.1 is used to rewrite the affected claim text so that it remains fluent and consistent with the edited fields.

Human validation and fallback correction. All non-positive pairs produced by rule-based perturbation and GPT-5.1 rewriting are manually validated against their expected labels. Annotators check both the claim texts and the structured fields. When a generated example does not satisfy the intended relation label, we do not rely on automatic re-classification; instead, we manually correct or rewrite the pair until it matches the intended relation. As a result, all final instances in CLAW-4L-RC are human-validated at the pair level.

## B.13 CLAW-4L-RC Statistics

Table 14 reports the label distribution and semantic similarity scores of claim pairs in CLAW-4L-RC. The dataset contains 600 claim pairs in total: 400 aligned pairs, 100 contradicted pairs, and 100 not-relevant pairs. The aligned category is further divided into four balanced sub-labels, each containing 100 instances.

To provide an auxiliary validation of the constructed pairs, we compute sentence-level semantic similarity using two English embedding models: BGE-Large-En-v1.5<sup>7</sup> and All-MPNet-Base-v2<sup>8</sup>. For the Aligned row, we report the macro-average over the four aligned sub-labels. For the Total row, we report a label-size-weighted average over all categories, so that the aggregate score reflects the full 600-pair distribution rather than over-emphasizing the aligned category.

The similarity patterns are consistent across the two embedding models. Exact aligned pairs obtain the highest scores, followed by one-directional partial alignment and then mutual partial alignment. Contradicted pairs remain relatively close in embedding space, whereas not-relevant pairs receive substantially lower similarity scores. This pattern is consistent with our annotation design: aligned pairs should be semantically close, not-relevant pairs should be clearly distant, and contradicted pairs may still share substantial topical or lexical content despite containing incompatible information. These results provide additional evidence for the quality and internal consistency of the manually constructed CLAW-4L-RC dataset.

<table><tr><td>Label</td><td># BGE</td><td>MPNet</td></tr><tr><td>Aligned − Exact (A = B)</td><td>100 96.87</td><td>96.04</td></tr><tr><td>Aligned – Partial (A &gt; B)</td><td>100 91.15</td><td>89.79</td></tr><tr><td>Aligned – Partial (B &gt; A)</td><td>100 91.15</td><td>89.79</td></tr><tr><td>Aligned – Partial (A ↔ B)</td><td>100 85.47</td><td>84.27</td></tr><tr><td>Aligned</td><td>400 91.16</td><td>89.97</td></tr><tr><td>Contradicted (A ⊥ B)</td><td>100 88.47</td><td>88.47</td></tr><tr><td>Not Relevant (A— B)</td><td>100 51.31</td><td>35.86</td></tr><tr><td>Weighted Total</td><td>600 84.07</td><td>80.70</td></tr></table>

Table 14: Label distribution and semantic similarity in CLAW-4L-RC. BGE and MPNet denote cosine similarity scores computed with BGE-Large-En-v1.5 and All-MPNet-Base-v2 sentence embeddings, respectively; scores are reported on a 0-100 scale.

## C LLM Classifier Selection on CLAW-4L-RC

## C.1 Models

To select a reliable claim-pair relation classifier, we evaluate a range of instruction-tuned LLMs on CLAW-4L-RC and additionally compare against two strong NLI baselines, DeBERTa-v3-largemnli<sup>9</sup> and MiniCheck-7B<sup>10</sup>. We focus primarily on medium-sized open-source models in the 3B–14B range, balancing inference efficiency with expected verification accuracy. We also include GPT-5.1 as a strong closed-source reference model, representing an upper-bound classifier in our setting.

Specifically, we evaluate the following model families:

• Qwen3: Qwen3-4B, Qwen3-8B, and Qwen3- 14B;

• Qwen3.5: Qwen3.5-4B and Qwen3.5-9B;

• Llama-3: Llama-3.2-3B-IT and Llama-3.1- 8B-IT;

• Ministral-3: Ministral-3-3B-IT, Ministral-3- 8B-IT, and Ministral-3-14B-IT;

• Gemma-3: Gemma-3-4B-IT and Gemma-3- 12B-IT;

• GPT-5: GPT-5.1.

For each model, we evaluate two input settings: w/o infobox, where the classifier only receives the two claim texts, and w/ infobox, where the corresponding structured infobox (parsed from the claims by GPT-5.1; a structured infobox example is provided in Appendix B.7) is additionally provided as contextual evidence.

## C.2 Evaluation Setting and Aggregate Metrics

CLAW-4L-RC is a claim-pair relation classification benchmark with six fine-grained labels. Four labels correspond to aligned relations: exact alignment $( A { = } B )$ , one-sided enrichment from A to B (A>B), one-sided enrichment from B to A $( B { > } A )$ , and mutual non-conflicting enrichment $( A {  } B )$ . The remaining two labels capture nonaligned relations: contradiction $( A \bot B )$ and nonrelevance (A⊣B).

We report accuracy for each fine-grained label and further compute three aggregate metrics. Let

$$
s _ { = } , s _ { A > B } , s _ { B > A } , s _ {  } , s _ { \perp } , s _ { + }
$$

denote the per-label accuracies for $A { = } B , A { > } B$ $B { > } A , A {  } B , A { \bot } B$ , and A⊣B, respectively.

First, Align-FG measures fine-grained discrimination within the aligned category:

$$
\mathrm { A l i g n – F G } = \frac { s _ { = } + s _ { A > B } + s _ { B > A } + s _ {  } } { 4 } .
$$

This metric evaluates whether a classifier can distinguish different types of aligned and partially aligned relations, rather than merely recognizing that two claims are broadly related.

Second, ARC measures top-level relation classification accuracy by first averaging the four aligned sub-labels into a single aligned score, and then macro-averaging over the three top-level relation types:

$$
\mathsf { A R C } = \frac { \mathrm { A l i g n – F G } + s _ { \perp } + s _ { \mathrm { - l } } } { 3 } .
$$

This metric reflects performance on the main relation judgment task: aligned, contradicted, or not relevant, while avoiding over-weighting the aligned category, which contains four fine-grained sub-labels.

Finally, Overall is the macro-average over all six fine-grained labels:

$$
\mathrm { O v e r a l l } = \frac { s _ { = } + s _ { A > B } + s _ { B > A } + s _ {  } + s _ { \perp } + s _ { - } } { 6 } .
$$

Unlike ARC, this metric treats all fine-grained labels equally and therefore reflects the classifier’s full six-way classification performance.

## C.3 NLI Baseline Adaptation

We additionally compare against two strong NLIstyle baselines, MiniCheck-7B and DeBERTa-v3- large-mnli. Since these models are not designed for our six-way claim relation setting, we adapt their outputs into the CLAW-4L-RC label space as far as possible.

MiniCheck-7B. MiniCheck-7B outputs a binary label in {supported, unsupported}. We therefore apply it in both directions for a claim pair (A, B). If A supports B and B supports A, we map the pair to exact alignment (A=B). If A supports B but B does not support A, we map it to one-sided enrichment $( A > B ) ;$ symmetrically, if B supports A but A does not support B, we map it to $( B { > } A )$ . However, MiniCheck-7B cannot distinguish mutual partial alignment (A↔B), contradiction (A⊥B), or non-relevance (A⊣B), since all of these collapse into the same unsupported outcome. Accordingly, we only report the label-wise accuracies that can be directly computed.

## DeBERTa-v3-large-mnli. DeBERTa-v3-

large-mnli outputs a three-way NLI label in {entailment, neutral, contradiction}. As with MiniCheck, we evaluate each pair bidirectionally. If A entails B and B entails A, we map the pair to exact alignment (A=B). If A entails B and B is neutral with respect to A, we map it to $A > B ;$ symmetrically, if B entails A and A is neutral with respect to B, we map it to $B { > } A$ . If either direction yields contradiction, we map the pair to contradiction (A⊥B). However, this model still cannot distinguish mutual partial alignment (A↔B) from non-relevance (A⊣B), since both may appear as neutral in one or both directions. For this reason, we report only the directly computable label-wise accuracies and omit aggregate scores.

## C.4 Results

We provide the full results in Table 15 and summarize them in Figure 9. We highlight several key observations.

NLI baselines as partial classifiers. The NLI baselines provide a useful reference point but are inherently limited by label mismatch. MiniCheck-7B can only recover exact and one-sided support relations, while DeBERTa-v3-large-mnli additionally captures contradiction, but still cannot distinguish mutual partial alignment from non-relevance. Notably, DeBERTa-v3-large-mnli performs strongly on contradiction (95.0) and exact alignment (90.0), showing that supervised NLI training transfers well to coarse semantic consistency judgments. However, neither NLI baseline can model the finer enrichment distinctions central to CLAW-4L-RC, especially the mutual partial alignment case, which limits their usefulness as full classifiers in our setting.

![](images/026a026e099e24c13079a512f05395fa24a1948fc8f4df8399043e6b65e4cd6d.jpg)  
Figure 9: Comparison of the best-performing classifier from each model family on CLAW-4L-RC across three aggregate metrics (ARC, Align-FG, Overall). GPT-5.1 achieves the strongest performance, while Qwen3.5-9B is the closest open-source model across all metrics.

Strong open-source classifier. Qwen3.5-9B emerges as the strongest open-source classifier, consistently achieving performance close to GPT-5.1 across all aggregate metrics. In particular, the gap between Qwen3.5-9B and GPT-5.1 is small (e.g., 95.5 vs. 95.9 in ARC), while both models substantially outperform other open-source families.

Fine-grained alignment is more challenging. We observe a clear positive correlation between ARC and Align-FG, indicating that models with strong top-level relation classification also perform well on fine-grained alignment distinctions. However, Align-FG scores are consistently lower than ARC across all models, confirming that fine-grained alignment (e.g., distinguishing A>B, B>A, and A↔B) is a more challenging task. Among these, the mutual enrichment case (A↔B) is typically the hardest, suggesting that modeling bidirectional, non-conflicting information requires more precise semantic reasoning.

Limited benefit of structured infoboxes. Providing GPT-5.1-parsed infoboxes does not consistently improve performance. For stronger models (e.g., Qwen3-14B, Qwen3.5-9B, GPT-5.1), infobox inputs often slightly degrade performance, suggesting that additional structured context may introduce noise or redundancy. In contrast, smaller models occasionally benefit from infobox inputs, indicating that external structure can partially compensate for weaker internal representations. Overall, these results suggest that the effectiveness of infobox augmentation depends on model capacity and input fidelity.

Scaling trends within model families. Within most model families, larger models tend to achieve better performance, consistent with expected scaling behavior. This trend holds for Qwen3 and Qwen3.5, where performance improves monotonically with model size. An exception is observed in the Ministral-3 family, where the 3B model outperforms larger variants.

These findings justify our choice of Qwen3.5-9B as the default open-source classifier in our experiments.

## D Cross-lingual Claim Extraction Methods

This section summarizes the English claim extraction frameworks adapted in our experiments and clarifies how they are unified for cross-lingual evaluation on CLAW-4L-CX. Rather than comparing the original backbone models or source tasks of prior work, we focus on the functional design choices that matter for our setting and then describe how each method is adapted to the cross-lingual sentence-to-claims setting.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Info.</td><td colspan="4">Aligned</td><td rowspan="2">Contr.</td><td rowspan="2">Not Rel.</td><td colspan="3">Aggregate</td></tr><tr><td>A=B</td><td>A&gt;B B&gt;A</td><td>A↔B</td><td>A⊥B</td><td>A-B</td><td>ARC Align-FG</td><td>Overall</td></tr><tr><td colspan="9">NLI</td><td></td></tr><tr><td>MiniCheck-7B</td><td>w/o</td><td>85.0</td><td>82.0</td><td>82.0</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>DeBERTa-v3-large</td><td>w/o</td><td>90.0</td><td>86.0</td><td>86.0</td><td>x</td><td>95.0</td><td>x</td><td>x</td><td>X</td><td>x</td></tr><tr><td colspan="9">Qwen3</td><td></td></tr><tr><td>Qwen3-4B</td><td>w/o w/</td><td>52.0 37.0</td><td>83.0 81.0</td><td>61.0 75.0</td><td>22.0 37.0</td><td>63.0 66.0</td><td>91.0 97.0</td><td>69.5 73.5</td><td>54.5 57.5</td><td>62.0 65.5</td></tr><tr><td>Qwen3-8B</td><td>w/o w/</td><td>71.0 72.0</td><td>79.0 79.0</td><td>95.0 92.0</td><td>27.0 45.0</td><td>83.0 77.0</td><td>98.0 99.0</td><td>83.0 82.7</td><td>68.0 72.0</td><td>75.5 77.3</td></tr><tr><td>Qwen3-14B</td><td>w/o w/</td><td>95.0 85.0</td><td>72.0 61.0</td><td>77.0 71.0</td><td>71.0 75.0</td><td>95.0 90.0</td><td>99.0 100.0</td><td>90.9 87.7</td><td>78.8 73.0</td><td>84.8 80.3</td></tr><tr><td colspan="9">Qwen3.5</td><td></td></tr><tr><td>Qwen3.5-4B</td><td>w/o w/</td><td>71.0</td><td>72.0</td><td>93.0</td><td>88.0</td><td>93.0</td><td>90.0</td><td>88.0</td><td>81.0</td><td>84.5 80.2</td></tr><tr><td>Qwen3.5-9B</td><td>w/o w/</td><td>60.0 94.0</td><td>73.0 88.0</td><td>73.0 91.0</td><td>85.0 85.0</td><td>97.0 99.0</td><td>93.0 98.0</td><td>87.6 95.5</td><td>72.8 89.5</td><td>92.5 92.2</td></tr><tr><td colspan="9">Llama-3</td><td></td></tr><tr><td>Llama-3.2-3B-IT</td><td>w/o</td><td>43.0</td><td>29.0</td><td>9.0</td><td>60.0</td><td>41.0</td><td>45.0</td><td>40.4</td><td>35.3</td><td>37.8</td></tr><tr><td>Llama-3.1-8B-IT</td><td>w/ w/o</td><td>29.0 51.0</td><td>11.0 74.0</td><td>5.0 20.0</td><td>81.0 1.0</td><td>7.0 61.0</td><td>33.0 92.0</td><td>23.8 63.2</td><td>31.5 36.5</td><td>27.7 49.8 50.0</td></tr><tr><td colspan="9">Ministral-3</td><td></td></tr><tr><td>Ministral-3-3B-IT</td><td>w/o</td><td>88.0</td><td>24.0</td><td>56.0</td><td>23.0</td><td>70.0</td><td>99.0</td><td>72.3</td><td>47.8</td><td>60.0 53.8</td></tr><tr><td>Ministral-3-8B-IT</td><td>w/ w/o w/</td><td>81.0 76.0</td><td>12.0 44.0</td><td>45.0 36.0</td><td>8.0 14.0</td><td>79.0 92.0</td><td>98.0 94.0</td><td>71.2 76.2</td><td>36.5 42.5</td><td>59.3 53.3</td></tr><tr><td>Ministral-3-14B-IT</td><td>w/o</td><td>57.0 75.0</td><td>39.0 2.0</td><td>31.0 1.0</td><td>12.0 1.0</td><td>87.0 88.0</td><td>94.0 89.0</td><td>71.9 65.6</td><td>34.8 19.8</td><td>42.7 41.7</td></tr><tr><td colspan="9">Gemma-3</td><td></td></tr><tr><td>Gemma-3-4B-IT</td><td>w/o</td><td>87.0</td><td>37.0</td><td>38.0</td><td>0.0</td><td>40.0</td><td>71.0</td><td>50.5</td><td>40.5</td><td>45.5</td></tr><tr><td></td><td>w/</td><td>67.0</td><td>54.0</td><td>33.0</td><td>3.0</td><td>42.0</td><td>77.0</td><td>52.8</td><td>39.3</td><td>46.0</td></tr><tr><td>Gemma-3-12B-IT w/</td><td>w/o</td><td>87.0 79.0</td><td>78.0 73.0</td><td>15.0 22.0</td><td>12.0</td><td>71.0</td><td>97.0</td><td>72.0 71.0</td><td>48.0 46.0</td><td>60.0 58.5</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>10.0</td><td>70.0</td><td>97.0</td><td></td><td></td><td></td></tr><tr><td colspan="9">GPT-5</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.1</td><td>w/o w/</td><td>97.0 95.0</td><td>88.0 86.0</td><td>93.0 87.0</td><td>85.0 78.0</td><td>97.0 95.0</td><td>100.0 100.0</td><td>95.9 93.8</td><td>90.8 86.5</td><td>93.3 90.2</td></tr></table>

Table 15: Claim-pair relation classification results on CLAW-4L-RC across various classifiers including NLI models and LLMs. w/o denotes using only claim pairs as input, while w/ additionally provides GPT-5.1-parsed infoboxes. All metrics are accuracy (%). ARC is the macro-average over the three top-level relation types, where the Aligned score is first averaged over its four sub-labels. Align-FG measures fine-grained accuracy within the Aligned category (average over its four sub-labels). Overall is the macro-average over all six fine-grained labels. Best and second-best results in each column are marked in bold and underlined, respectively. Entries marked with ✗ are not directly computable because the corresponding classifier does not provide enough label resolution to distinguish that relation type.

## D.1 Core Operations in Claim Extraction

Decomposition. Decomposition breaks a sentence or passage into smaller factual units so that each claim can be checked independently. For example, the sentence "Marie Curie won the Nobel Prize in Physics in 1903 and the Nobel Prize in Chemistry in 1911" can be decomposed into two claims, one for each award event.

## Decontextualization. Decontextualization

rewrites a claim so that it can be understood and verified without relying on the surrounding context. For example, "In 1903, she won the Nobel Prize in Physics" becomes "Marie Curie won the Nobel Prize in Physics in 1903."

Verifiable Selection. Verifiable selection determines whether a sentence contains objectively checkable factual content and filters out nonverifiable material such as opinions, speculation, or advice. For instance, "Solar power could transform the future of humanity" is not directly verifiable, whereas "Solar power provided 3.9% of global electricity in 2021" is.

Disambiguation. Disambiguation handles cases with multiple plausible interpretations, typically by resolving ambiguity from context or abstaining when reliable resolution is not possible. For example, in "John met Paul after he won the award", the pronoun "he" may refer to either John or Paul. In such cases, a system should avoid producing a confident factual claim unless the referent can be resolved reliably.

## D.2 Adapted English Claim Extraction Frameworks

Table 16 summarizes the main functional differences among the compared extraction frameworks. We adapt five representative English claim extraction methods to our cross-lingual setting: FactScore, FactCheck-GPT, DnDScore, VeriScore, and Claimify. We prefix the adapted variants with X-. In all cases, we modify the original prompts so that the system accepts either an English or a non-English-language sentence as input and always outputs atomic factual claims in English. The goal is to map both English and non-English biographies into a shared English claim space for downstream alignment and enrichment. We provide the X-Claimify prompts in Appendix H; prompts and runnable scripts for all adapted frameworks are included in the accompanying repository.

X-FactScore. X-FactScore is the simplest variant and focuses on direct decomposition. Given a sentence, it extracts one or more factual claims in English without explicitly requiring decontextualization, verifiable filtering, or ambiguity handling beyond what the LLM performs implicitly.

X-DnDScore. X-DnDScore extends direct decomposition by explicitly decontextualizing and handling ambiguity. In a single prompt, it first decomposes the input sentence into subclaims and then rewrites each subclaim into a stand-alone English proposition using only the provided paragraph context. Unlike simpler decompositionbased methods, it explicitly asks the model to identify ambiguities and avoid confident extraction when core references, entities, or relations cannot be resolved reliably.

X-VeriScore. X-VeriScore jointly performs decomposition, decontextualization, and verifiable selection in a single prompt. It extracts as many finegrained verifiable facts as possible from the non-English sentence, while using the surrounding context only to resolve pronouns and definite descriptions. Each output claim must be self-contained, independently checkable, and written in English; if the sentence contains no verifiable factual content, the model is instructed to abstain by returning No verifiable claim.

X-FactCheck-GPT. X-FactCheck-GPT uses a two-stage extraction design. It first decomposes the input sentence into standalone English atomic facts, using the provided context to resolve references and decontextualize the output. It then applies a separate checkworthy classification step to determine whether the sentence contains factual content worth keeping. This staged design is intended to reduce noisy or weakly factual outputs while preserving decontextualized atomic claims.

X-Claimify. X-Claimify is the most structured pipeline among the compared methods. It combines verifiable selection, decontextualization, disambiguation, and decomposition in a multi-turn prompting process. In our cross-lingual setting, it is adapted so that claims extracted from non-English biographies are directly normalized into English, making them comparable to claims extracted from the English side.

<table><tr><td>Method (Year)</td><td>Uses Context</td><td>Decomp.</td><td>Decontext.</td><td>Verifiable Sel.</td><td>Disamb.</td><td>Execution</td></tr><tr><td>FactScore (2023)</td><td>x</td><td>√</td><td>x</td><td>x</td><td>x</td><td>Joint</td></tr><tr><td>DnDScore (2025)</td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td><td>In-prompt Seq.</td></tr><tr><td>VeriScore (2024)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>Joint</td></tr><tr><td>FactCheck-GPT (2024)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>x</td><td>Multi-turn</td></tr><tr><td>Claimify (2025)</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>Multi-turn</td></tr></table>

Table 16: Functional comparison of the English claim extraction frameworks adapted in our experiments. A check mark $( \checkmark )$ indicates that the corresponding operation is explicitly specified in the method prompt or pipeline. Uses Context indicates whether the method explicitly incorporates the surrounding sentence or paragraph context beyond the focal sentence. Execution describes how the operations are organized: Joint uses a single prompt without staged intermediate outputs, In-prompt Seq. performs the steps sequentially within one prompt, and Multi-turn splits them across multiple prompting rounds.

## D.3 Evaluation on CLAW-4L-CX

We evaluate claim extraction quality on CLAW-4L-CX, where each sentence is independently annotated by three human references. Because generated claims may differ from the references in wording and in how contextual information is made explicit, exact string matching is inappropriate. We therefore perform classifier-based matching at the claim-set level using the claim-pair relation classifier (Qwen3.5-9B) described in Section 5.1.

Let $\hat { \mathcal { C } } = \{ \hat { c } _ { 1 } , \hdots , \hat { c } _ { m } \}$ denote the set of generated claims for a sentence, and let $\boldsymbol { \mathcal { C } } ^ { ( r ) ^ { \smile } } =$ $\{ c _ { 1 } ^ { ( r ) } , \ldots , c _ { n } ^ { ( r ) } \}$ denote the reference claims provided by annotator $r \in \{ 1 , 2 , 3 \}$ . For each reference set $\mathcal { C } ^ { ( r ) }$ , we apply the classifier to all generated–reference claim pairs and obtain relation labels.

We use degree-normalized claim-level matching rather than one-to-one maximum matching. For a matching criterion $k \in \{ \mathrm { a l i g n } , \mathrm { e x a c t } \}$ , let $E _ { k } ^ { ( r ) }$ denote the generated–reference claim pairs whose classifier-predicted relation satisfies criterion k. The align criterion accepts both exact and partial alignment, while the exact criterion accepts only exact alignment. For each generated claim i and reference claim j, define

$$
\begin{array} { l } { d _ { i } ^ { k } = | \{ j ^ { \prime } : ( i , j ^ { \prime } ) \in E _ { k } ^ { ( r ) } \} | , } \\ { \bar { d } _ { j } ^ { k } = | \{ i ^ { \prime } : ( i ^ { \prime } , j ) \in E _ { k } ^ { ( r ) } \} | . } \end{array}
$$

We then assign claim-level precision and recall credits:

$$
p _ { i } ^ { k } = \left\{ { \begin{array} { l l } { 1 / d _ { i } ^ { k } , } & { d _ { i } ^ { k } > 0 , } \\ { 0 , } & { d _ { i } ^ { k } = 0 , } \end{array} } \right.
$$

$$
r _ { j } ^ { k } = \left\{ \begin{array} { l l } { 1 / \bar { d } _ { j } ^ { k } , } & { \bar { d } _ { j } ^ { k } > 0 , } \\ { 0 , } & { \bar { d } _ { j } ^ { k } = 0 . } \end{array} \right.
$$

This fan-out normalization reduces the precision credit of overly broad generated claims that align with multiple reference claims, and reduces the recall credit when multiple generated claims redundantly align with the same reference claim.

For each criterion $k ,$ we define precision, recall, and F1 as

$$
P _ { k } ^ { ( r ) } = \frac { 1 } { | \hat { \mathcal { C } } | } \sum _ { i = 1 } ^ { m } p _ { i } ^ { k } , \qquad R _ { k } ^ { ( r ) } = \frac { 1 } { | \mathcal { C } ^ { ( r ) } | } \sum _ { j = 1 } ^ { n } r _ { j } ^ { k } ,
$$

$$
F _ { k } ^ { ( r ) } = \frac { 2 P _ { k } ^ { ( r ) } R _ { k } ^ { ( r ) } } { P _ { k } ^ { ( r ) } + R _ { k } ^ { ( r ) } } .
$$

Aligned-F1 measures whether the extractor recovers the relevant factual content even under partial alignment, whereas Exact-Aligned-F1 more directly reflects factual specificity and decomposition quality.

For each sentence, we compute these scores separately against each of the three human reference sets. We then select the reference set that yields the highest exact-alignment score,

$$
r ^ { * } = \arg \operatorname* { m a x } _ { r \in \{ 1 , 2 , 3 \} } F _ { \mathrm { e x a c t } } ^ { ( r ) } ,
$$

and report

$$
\begin{array} { r } { \mathrm { A l i g n e d - F 1 } = F _ { \mathrm { a l i g n } } ^ { ( r ^ { * } ) } , } \\ { \mathrm { E x a c t - A l i g n e d - F 1 } = F _ { \mathrm { e x a c t } } ^ { ( r ^ { * } ) } . } \end{array}
$$

Using F1 in both cases allows us to jointly capture precision and coverage. Intuitively, Aligned-F1 measures semantic content recovery, whereas Exact-Aligned-F1 more strongly rewards exact claim formulation.

## D.4 Two-stage Selection Protocol

To separate the extraction-framework choice from the backbone choice, we adopt a two-stage evaluation protocol.

Stage 1: extraction framework selection. We first instantiate all compared extraction frameworks with GPT-5.1 and evaluate them on CLAW-4L-CX. Using a common strong backbone allows us to isolate differences in the extraction framework design rather than differences in the model capability. Based on this comparison, we select the best-performing extraction formulation.

Stage 2: open-source backbone selection. After fixing the best-performing extraction framework, we compare several open-source LLMs as practical backbones for cross-lingual claim extraction, including Qwen3.6-27B, Gemma-4-31B-it, and Mistral-3.2-24B-it. This second stage is motivated by cost, reproducibility, and scalability considerations. We use it to assess whether open-source models can approximate GPT-5.1 extraction quality, while keeping GPT-5.1 + X-Claimify as the reference extraction configuration for the main pipeline.

## D.5 Final Choice

Table 17 shows that X-Claimify is the most reliable extraction framework on all language splits. It achieves the strongest Exact-Aligned-F1 in English, French, and Chinese, and also obtains the best or near-best Aligned-F1. This pattern is important for our setting because downstream alignment requires claims that both recover the source content and preserve a comparable event-level granularity.

The weaker performance of X-DnDScore is consistent with its operation order. X-DnDScore first decomposes a sentence into subclaims and then decontextualizes each subclaim separately. In biography text, this can copy the same contextual information into multiple claims, creating overlapping or redundant claim verbalizations. In contrast, methods that resolve context before final decomposition are better able to produce factually independent claims, which improves exact matching against human references. X-Claimify further combines verifiable selection, disambiguation, decontextualization, and decomposition in a staged pipeline, which makes it better suited to sentences with coreference, partial names, and event-level biographical details.

We therefore use GPT-5.1 + X-Claimify as the reference extraction configuration for constructing English and non-English-side claim sets and for identifying non-English enrichment claims in downstream enrichment experiments. Table 18 compares open-source backbones as lower-cost alternatives. Overall, GPT-5.1 remains the strongest and most consistent backbone across languages. An exception is the French split, where Gemma-4-31B-it slightly exceeds GPT-5.1 on Aligned-F1 (82.62 vs. 78.05), indicating stronger broad content recovery, but remains lower on Exact-Aligned-F1 (70.18 vs. 73.56), indicating weaker exact claim formulation. For pipeline consistency and stronger aggregate precision, we keep GPT-5.1 + X-Claimify as the reference extraction configuration for all languages.

## E Machine Translation Model Selection for Biography Enrichment

To support the Machine Translation and Generation Pipeline setting in Section 6.1, we compare five candidate non-English-to-English translation models: the three open-source generator LLMs used in our enrichment experiments (Qwen3.6-27B, Gemma-4-31B-it, and Mistral-3.2-24B-it), together with two multilingual machine translation models, M2M100-1.2B and NLLB-200-3.3B.

We evaluate these translation models on a small but challenging subset of CLAW-4L. Specifically, for each non-English language (French, Chinese, and Azerbaijani), we select the three biographies with the largest token counts, yielding 9 biographies in total. These long biographies are particularly informative for translation model selection because they better reflect the context length and information density challenges encountered in the full enrichment setting.

## E.1 Translation Protocol

We translate each non-English biography into English with each of the five candidate models. As in the generation setting, we cannot always feed an entire long biography together with task instructions into the model at once, since increased context length substantially raises GPU memory requirements during inference. We therefore use the same incremental chunking strategy as in Section 6.1.

Since translation only requires the non-English biography as input, without the English biography, we use a larger per-chunk budget and cap the non-English-side input at 4,096 tokens. When a biography exceeds this limit, we split it incrementally using the same section-based procedure as in the generation setup, with newline-based and token-level fallback splitting for exceptionally long sections. The translated chunks are then concatenated in order to form the full English translation of the biography.

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Method</td><td colspan="3">Aligned</td><td colspan="3">Exact-Aligned</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="5">AZ</td><td>X-FactScore</td><td>71.38</td><td>47.19</td><td>54.04</td><td>37.07</td><td>47.73</td><td>39.51</td></tr><tr><td>X-VeriScore</td><td>68.79</td><td>68.00</td><td>66.30</td><td>57.27</td><td>56.31</td><td>55.11</td></tr><tr><td>X-DnDScore</td><td>64.72</td><td>36.87</td><td>44.76</td><td>19.09</td><td>32.95</td><td>22.87</td></tr><tr><td>X-FactCheck-GPT</td><td>56.63</td><td>51.88</td><td>52.91</td><td>43.69</td><td>45.93</td><td>44.38</td></tr><tr><td>X-Claimify</td><td>69.92</td><td>72.93</td><td>70.52</td><td>67.77</td><td>66.83</td><td>66.80</td></tr><tr><td rowspan="5">EN</td><td>X-FactScore</td><td>76.25</td><td>44.81</td><td>52.89</td><td>23.63</td><td>38.21</td><td>27.83</td></tr><tr><td>X-VeriScore</td><td>77.51</td><td>65.50</td><td>68.82</td><td>46.37</td><td>49.48</td><td>46.92</td></tr><tr><td>X-DnDScore</td><td>69.28</td><td>32.98</td><td>41.91</td><td>9.98</td><td>17.77</td><td>11.61</td></tr><tr><td>X-FactCheck-GPT</td><td>70.04</td><td>54.90</td><td>59.60</td><td>42.18</td><td>46.66</td><td>43.70</td></tr><tr><td>X-Claimify</td><td>77.72</td><td>75.76</td><td>75.17</td><td>72.82</td><td>73.18</td><td>72.44</td></tr><tr><td rowspan="5">FR</td><td>X-FactScore</td><td>77.93</td><td>49.54</td><td>57.89</td><td>31.12</td><td>44.67</td><td>35.23</td></tr><tr><td>X-VeriScore</td><td>77.43</td><td>66.76</td><td>70.01</td><td>52.03</td><td>57.54</td><td>52.91</td></tr><tr><td>X-DnDScore</td><td>72.56</td><td>37.57</td><td>46.16</td><td>13.22</td><td>24.87</td><td>16.08</td></tr><tr><td>X-FactCheck-GPT</td><td>70.71</td><td>58.10</td><td>62.13</td><td>48.75</td><td>52.25</td><td>49.87</td></tr><tr><td>X-Claimify</td><td>82.48</td><td>76.98</td><td>78.05</td><td>73.84</td><td>74.30</td><td>73.56</td></tr><tr><td rowspan="5">ZH</td><td>X-FactScore</td><td>69.93</td><td>45.61</td><td>52.51</td><td>22.83</td><td>34.28</td><td>26.30</td></tr><tr><td>X-VeriScore</td><td>76.70</td><td>64.68</td><td>68.12</td><td>41.99</td><td>45.63</td><td>42.94</td></tr><tr><td>X-DnDScore</td><td>71.48</td><td>35.14</td><td>44.75</td><td>14.64</td><td>23.52</td><td>16.02</td></tr><tr><td>X-FactCheck-GPT</td><td>63.86</td><td>50.84</td><td>54.70</td><td>47.85</td><td>53.53</td><td>49.82</td></tr><tr><td>X-Claimify</td><td>76.08</td><td>69.32</td><td>71.12</td><td>70.52</td><td>74.93</td><td>71.91</td></tr></table>

Table 17: Comparison of different claim extraction methods using GPT-5.1 against human reference claims on CLAW-4L-CX. Scores are percentages.

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Method</td><td colspan="3">Aligned</td><td colspan="3">Exact-Aligned</td><td rowspan="2">Avg-F1</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="4">AZ</td><td>GPT-5.1</td><td>69.92</td><td>72.93</td><td>70.52</td><td>67.77</td><td>66.83</td><td>66.80</td><td>68.66</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>64.00</td><td>70.29</td><td>64.90</td><td>59.59</td><td>52.42</td><td>54.35</td><td>59.62</td></tr><tr><td>Gemma-4-31B-IT</td><td>66.25</td><td>78.45</td><td>69.42</td><td>58.16</td><td>53.27</td><td>54.28</td><td>61.85</td></tr><tr><td>Qwen3.6-27B</td><td>67.30</td><td>75.42</td><td>69.23</td><td>63.90</td><td>58.73</td><td>59.88</td><td>64.56</td></tr><tr><td rowspan="4">EN</td><td>GPT-5.1</td><td>77.72</td><td>75.76</td><td>75.17</td><td>72.82</td><td>73.18</td><td>72.44</td><td>73.81</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>77.11</td><td>72.83</td><td>72.79</td><td>58.03</td><td>54.02</td><td>54.83</td><td>63.81</td></tr><tr><td>Gemma-4-31B-IT</td><td>76.52</td><td>77.79</td><td>75.42</td><td>64.61</td><td>64.09</td><td>63.83</td><td>69.63</td></tr><tr><td>Qwen3.6-27B</td><td>68.08</td><td>65.82</td><td>65.73</td><td>55.66</td><td>55.37</td><td>55.25</td><td>60.49</td></tr><tr><td rowspan="4">FR</td><td>GPT-5.1</td><td>82.48</td><td>76.98</td><td>78.05</td><td>73.84</td><td>74.30</td><td>73.56</td><td>75.81</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>79.89</td><td>75.72</td><td>75.90</td><td>58.75</td><td>56.65</td><td>56.50</td><td>66.20</td></tr><tr><td>Gemma-4-31B-IT</td><td>82.88</td><td>84.84</td><td>82.62</td><td>71.62</td><td>69.60</td><td>70.18</td><td>76.40</td></tr><tr><td>Qwen3.6-27B</td><td>83.25</td><td>81.26</td><td>80.35</td><td>65.38</td><td>64.22</td><td>64.12</td><td>72.24</td></tr><tr><td rowspan="4">ZH</td><td>GPT-5.1</td><td>76.08</td><td>69.32</td><td>71.12</td><td>70.52</td><td>74.93</td><td>71.91</td><td>71.52</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>74.87</td><td>73.63</td><td>72.33</td><td>46.32</td><td>45.78</td><td>45.74</td><td>59.04</td></tr><tr><td>Gemma-4-31B-IT</td><td>73.15</td><td>69.64</td><td>70.46</td><td>60.35</td><td>61.47</td><td>60.52</td><td>65.49</td></tr><tr><td>Qwen3.6-27B</td><td>74.06</td><td>72.39</td><td>71.99</td><td>56.70</td><td>57.21</td><td>56.31</td><td>64.15</td></tr></table>

Table 18: Claim extraction quality of X-Claimify using different claim generation models against human reference claims on CLAW-4L-CX. Scores are percentages. Avg-F1 is the average of Aligned F1 and Exact-Aligned F1.

## E.2 Claim-based Evaluation of Translated Biographies

We evaluate translated biographies using a claimbased protocol rather than surface-form overlap. For each original non-English biography, we first extract non-English-to-English reference claims using X-Claimify instantiated with GPT-5.1. These claims serve as the semantic reference for the factual content expressed in the source biography. We then apply the same X-Claimify + GPT-5.1 extraction pipeline to each translated English biography produced by the candidate translation models.

Because biography-level claim sets can be large, direct all-pairs claim comparison is computationally expensive. We therefore adopt a coarse-tofine matching strategy. First, for each translated claim, we retrieve candidate reference claims using all-mpnet-base-v2, keeping only pairs that satisfy a cosine similarity threshold of 0.7 and fall within the top-5 nearest neighbors. We use MP-Net rather than BGE because Table 14 suggests that MPNet provides stronger separation between aligned and clearly unrelated claim pairs. The threshold of 0.7 is intentionally loose and is used only to prune highly irrelevant pairs before claimlevel verification.

We then apply the degree-normalized claimlevel scoring protocol described in Appendix D.3 to the filtered candidate pairs. As in our extraction evaluation, we report both Aligned-F1 and Exact-Aligned-F1. Aligned-F1 measures whether the translation preserves the relevant factual content even under slight differences in granularity, whereas Exact-Aligned-F1 more strictly evaluates whether the translated biography preserves claims in a form that matches the reference at the atomic level.

Results are shown in Table 19, and the final per-language choices are summarized in Table 20. We make three main observations. First, Exact-Aligned scores are consistently much lower than coarse Aligned scores across all language pairs and models. This confirms that exact alignment is a substantially more demanding criterion for biography translation: while many translated claims remain semantically relevant to the source, exact claim preservation is highly sensitive to paraphrasing, granularity shifts, and small factual perturbations introduced during translation.

Second, the three open-source LLMs consistently outperform the two dedicated multilingual translation models across all three language pairs, under both Aligned-F1 and Exact-Aligned-F1. This pattern likely reflects two factors. One is model scale: the open-source LLMs are substantially larger than M2M100-1.2B and NLLB-200-3.3B. The other is context handling: biography translation in our setting is long-context and discourse-dependent, whereas the multilingual MT models are more constrained in usable context length and therefore less able to exploit broader biography context during incremental translation.

Third, translation difficulty differs across languages. French-to-English translation is overall the easiest setting, with the highest scores across models, while Chinese-to-English is the most challenging. For final model selection, Azerbaijani shows a consistent winner under both coarse and strict matching, leading us to select Gemma-4-31Bit. French is slightly less clear-cut: Gemma-4- 31B-it is marginally better on coarse Aligned-F1 (93.80 vs. 93.77), but Mistral-3.2-24B-it shows a substantially stronger result on Exact-Aligned-F1 (48.98 vs. 45.48). We therefore select Mistral-3.2- 24B-it for French. Chinese is also less clear-cut: Qwen3.6-27B achieves the best coarse Aligned-F1, while Gemma-4-31B-it performs better under Exact-Aligned-F1. We ultimately select Qwen3.6- 27B for Chinese, prioritizing stronger semantic claim preservation over stricter exact-form matching.

These language-specific translation choices are then used in the Machine Translation and Generation Pipeline setting in the main experiments.

## F Target-Side Enrichment Claim Selection

After extracting English and non-English claims, we align each non-English claim against candidate English claims using the classifier in Section 5.1. Because biography-level claim sets can be large, we avoid exhaustive all-pairs verification. For each non-English-side claim $B \in \mathcal { C } ^ { t g t }$ , we retrieve candidate English claims $A \in { \mathcal { C } } ^ { e n }$ using All-MPNet-Base-v2 cosine similarity, retaining only pairs whose similarity is at least 0.7 and that fall within the top-5 nearest English claims for B. Pairs not retained by this retrieval step are treated as not relevant and are not sent to the LLM classifier.

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Model</td><td colspan="3">Aligned</td><td colspan="3">Exact-Aligned</td></tr><tr><td>P</td><td>R</td><td>F1</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="5">AZ → EN</td><td>Qwen3.6-27B</td><td>92.27</td><td>90.53</td><td>91.39</td><td>34.49</td><td>32.92</td><td>33.69</td></tr><tr><td>Gemma-4-31B-it</td><td>92.66</td><td>91.51</td><td>92.08</td><td>36.64</td><td>33.70</td><td>35.11</td></tr><tr><td>Mistral-3.2-24B-it</td><td>91.08</td><td>90.94</td><td>91.01</td><td>34.57</td><td>32.23</td><td>33.36</td></tr><tr><td>M2M100-1.2B</td><td>78.64</td><td>64.86</td><td>71.09</td><td>27.28</td><td>10.37</td><td>15.03</td></tr><tr><td>NLLB-200-3.3B</td><td>88.08</td><td>86.02</td><td>87.04</td><td>32.33</td><td>29.89</td><td>31.06</td></tr><tr><td rowspan="5">FR → EN</td><td>Qwen3.6-27B</td><td>93.80</td><td>90.32</td><td>92.03</td><td>45.55</td><td>43.27</td><td>44.38</td></tr><tr><td>Gemma-4-31B-it</td><td>94.87</td><td>92.75</td><td>93.80</td><td>46.09</td><td>44.89</td><td>45.48</td></tr><tr><td>Mistral-3.2-24B-it</td><td>94.02</td><td>93.51</td><td>93.77</td><td>49.08</td><td>48.89</td><td>48.98</td></tr><tr><td>M2M100-1.2B</td><td>85.69</td><td>86.59</td><td>86.14</td><td>43.79</td><td>41.27</td><td>42.49</td></tr><tr><td>NLLB-200-3.3B</td><td>84.64</td><td>84.69</td><td>84.67</td><td>43.30</td><td>40.83</td><td>42.03</td></tr><tr><td rowspan="5">ZH → EN</td><td>Qwen3.6-27B</td><td>85.84</td><td>87.16</td><td>86.50</td><td>33.36</td><td>33.04</td><td>33.20</td></tr><tr><td>Gemma-4-31B-it</td><td>85.90</td><td>85.58</td><td>85.74</td><td>36.57</td><td>35.02</td><td>35.78</td></tr><tr><td>Mistral-3.2-24B-it</td><td>81.64</td><td>80.11</td><td>80.87</td><td>30.34</td><td>28.13</td><td>29.20</td></tr><tr><td>M2M100-1.2B</td><td>66.44</td><td>61.57</td><td>63.91</td><td>19.29</td><td>16.32</td><td>17.68</td></tr><tr><td>NLLB-200-3.3B</td><td>57.62</td><td>52.77</td><td>55.09</td><td>16.24</td><td>13.15</td><td>14.53</td></tr></table>

Table 19: Claim-based evaluation of candidate non-English-to-English translation models for biography translation. We report precision (P), recall (R), and F1 under both coarse-grained Aligned matching and strict Exact-Aligned matching. Best values within each language pair are shown in bold.

<table><tr><td>Language Pair</td><td>Selected MT Model</td></tr><tr><td>AZ → EN</td><td>Gemma-4-31B-it</td></tr><tr><td>FR → EN</td><td>Mistral-3.2-24B-it</td></tr><tr><td>ZH → EN</td><td>Qwen3.6-27B</td></tr></table>

Table 20: Final translation model selected for each non-English language.

Retrieval hyperparameters. Before top-k truncation, 67.43%, 50.32%, and 39.71% of target-side claims have at least three, five, and seven candidates above the 0.7 threshold, respectively. We therefore use top-5 as a middle operating point: it allows a broader candidate set than top-3 for claims with several plausible matches, while keeping the worst-case number of LLM verifier calls below that of top-7.

Let A denote an English claim and B denote a non-English-side claim. We group non-Englishside claims into five status categories according to their best relation to the English side. Exact matches (A=B) and English-more-specific alignments (A>B) are treated as already covered by English and are not selected. Non-English-morespecific and mutually complementary alignments (B>A and A↔B) are treated as non-Englishadditive because the non-English side contains nonconflicting information not fully expressed in English. We also retain classifier-labeled conflicts (A⊥B) and unmatched non-English claims (A⊣B)

as enrichment candidates. Since both sides are human-written Wikipedia biographies, conflicts are not treated as arbitrary web noise and may require contextual reconciliation. For example, an Englishside claim that a person received a master’s degree from Stanford and a non-English-side claim that she received a Ph.D. from Stanford may be judged contradictory in isolation, although the full biography may need to resolve whether these refer to distinct degrees or an incomplete English account. Unmatched claims often correspond to genuinely non-English-only biographical information. The downstream generator receives the selected claims together with the original English biography and is responsible for integrating only compatible, supported additions.

Table 21 shows that English biographies cover only a small fraction of non-English claims. In the average row, exact matches and Englishmore-specific alignments account for only 12.4 of 89.3 non-English claims (13.9%). In contrast, Target-additive alignments account for 23.4 claims (26.2%), indicating that non-English biographies often elaborate on events already mentioned in English with additional factual detail. Unmatched claims form the largest category, with 43.4 claims (48.6%), showing that nearly half of non-English claims have no relevant English-side counterpart. Classifier-labeled conflicts are smaller but nonnegligible (10.2 claims, 11.4%). Overall, 77.0 of

<table><tr><td rowspan="2">Direction</td><td rowspan="2">EN</td><td rowspan="2">Target</td><td>Exact</td><td>EN-More</td><td>Target-Add.</td><td>Conflict</td><td>Unmatched</td><td rowspan="2">Selected</td></tr><tr><td> $A { = } B$ </td><td> $A > B$ </td><td> $B { > } A / A {  } B$ </td><td> $A \bot B$ </td><td> $A { \dash } | B$ </td></tr><tr><td> $\mathrm { F R }  \mathrm { E N }$ </td><td>29.8</td><td>101.8</td><td>5.2</td><td>7.2</td><td>23.5</td><td>9.7</td><td>56.2</td><td>89.4</td></tr><tr><td> $\mathrm { A Z } \to \mathrm { E N }$ </td><td>33.6</td><td>93.8</td><td>6.9</td><td>8.6</td><td>26.0</td><td>12.4</td><td>39.9</td><td>78.3</td></tr><tr><td> $\mathrm { Z H } \to \mathrm { E N }$ </td><td>29.3</td><td>72.4</td><td>3.7</td><td>5.5</td><td>20.6</td><td>8.4</td><td>34.1</td><td>63.2</td></tr><tr><td>Avg.</td><td>30.9</td><td>89.3</td><td>5.3</td><td>7.1</td><td>23.4</td><td>10.2</td><td>43.4</td><td>77.0</td></tr></table>

Table 21: Average counts of English claims, non-English claims, and non-English claim status categories per biography pair. Here A denotes an English claim and B denotes a non-English claim. Exact and EN-More are already covered by English and are not selected. Target-Add. contains non-English claims with additional non-conflicting information. Conflict contains classifier-labeled contradictions between non-English claims and retrieved English candidates, while Unmatched contains non-English claims with no retrieved or classifier-relevant English-side counterpart. Selected is the set of target-side enrichment claims used as structured evidence for generation.

89.3 non-English claims (86.2%) are selected as enrichment candidates, supporting our use of non-English biographies as complementary evidence rather than parallel paraphrases of English biographies.

## G Enrichment Generation Evaluation

## G.1 Evaluation Metrics

We evaluate generated biographies in the same English claim space as Section 6.2. Let $\mathcal { C } ^ { e n }$ denote the original English claims, $\mathcal { C } ^ { a d d }$ the selected non-English enrichment claims, and $\hat { \mathcal { C } } ^ { g e n }$ the claims extracted from the generated biography. To determine support, we compare generated claims against the reference pool $\mathcal { C } ^ { r e f } = \mathcal { C } ^ { e n } \cup \mathcal { C } ^ { t g t }$ . Let A denote a reference claim and B a generated claim. We categorize each generated claim with contradiction priority: if any retained reference claim is labeled $A \bot B ,$ , then B is counted as contradicted. Otherwise, B is valid if some reference claim fully supports it, i.e., $A { = } B$ or $A { > } B .$ . Generated claims with neither contradiction nor full support are counted as unsupported, including generated-more-specific or mutually enriched relations $( B { > } A$ or $A {  } B )$ not-relevant pairs, and claims with no retrieved reference counterpart. Halluc.% is the sum of Unsup.% and Contr.%.

Claim growth is measured as

$$
\begin{array} { r } { \# \mathrm { G a i n } = | \hat { \mathcal { C } } ^ { g e n } | - | { \mathcal { C } } ^ { e n } | , \ } \\ { \mathrm { C o v e r a g e } = \frac { \# \mathrm { G a i n } } { | { \mathcal { C } } ^ { a d d } | } . \ } \end{array}
$$

We compute the estimated supported gain as

$$
\# \mathrm { V a l i d } = \mathrm { V a l i d } \times \# \mathrm { G a i n } ,
$$

where Valid is Valid% converted to a proportion. This estimates how much of the net claim growth is

supported by the reference pool, rather than counting all supported generated claims as new content.

The primary overall metric Bal. is defined in Section 6.2.

## G.2 Full Results

Table 22 summarizes the evidence-format averages used in the main analysis. Scores are averaged over all three non-English languages and all three generators for each evidence format. Table 28 reports the full generation results for all language, evidence-format, and generator combinations.

<table><tr><td>Evidence</td><td>#Valid↑</td><td>Halluc.%↓</td></tr><tr><td>Raw</td><td>15.89</td><td>29.49</td></tr><tr><td>Translation</td><td>19.05</td><td>29.59</td></tr><tr><td>Claims</td><td>20.47</td><td>23.54</td></tr></table>

Table 22: Average supported additions and hallucination rate across all non-English languages and generators for each evidence format.

## G.3 Qualitative Error Analysis

Case selection. We conduct a controlled manual analysis of outputs from Mistral-Small-3.2-24B-IT. For each language, we select the biography pairs with the lowest and highest numbers of verifiersupported output claims under the best-performing evidence format for that language (Claims for French and Chinese; Translation for Azerbaijani). We then inspect the Raw, Translation, and Claims outputs for the same entities. This yields 18 generated biographies: three languages, two cases per language, and three enrichment methods. Table 23 lists the selected cases. The selection scores are used only to identify contrasting cases; the error types are based on manual inspection.

<table><tr><td>Lang.</td><td>Case</td><td>Entity</td><td>Supported/Output</td></tr><tr><td>FR</td><td>Low</td><td>Tanina Mammeri</td><td>3/24</td></tr><tr><td>FR</td><td>High</td><td>Juliette Gréco</td><td>109/261</td></tr><tr><td>ZH</td><td>Low</td><td>Li Jiaqi</td><td>1/6</td></tr><tr><td>ZH</td><td>High</td><td>Xie Bingying</td><td>98/207</td></tr><tr><td>AZ</td><td>Low</td><td>Aygün Kazimova</td><td>1/536</td></tr><tr><td>AZ</td><td></td><td>High Franghiz Ali-Zadeh</td><td>113/239</td></tr></table>

Table 23: Cases selected for qualitative analysis. Scores are measured under Claims for FR and ZH and Translation for AZ; all three enrichment methods are inspected for every entity.

Table 24 illustrates the principal error patterns. Comparisons are abridged only for space. Because the analysis deliberately samples high- and lowsupport cases, these examples characterize recurring failure modes rather than their prevalence in the full benchmark.

<table><tr><td></td><td>Pattern and case Evidence-output comparison</td></tr><tr><td>Missing event metadata Juliette Gréco; Raw</td><td>Evidence: “made her debut in the play Victor ou les Enfants au pouvoir in November 1946.&quot; Output: &quot;began her career in the the- ater.&quot;</td></tr><tr><td>Temporal error Juliette Gréco; Claims</td><td>Evidence: “was born on 7 February 1927.&quot; Output: “was born on 6 February 1927.&quot;</td></tr><tr><td>Education / location error Franghiz Ali-Zadeh; Raw</td><td>Evidence: “studied piano at the Azer- baijan State Conservatory under Ulfan Khalilov.&quot; Output: “studied at the Moscow Conser- vatory.&quot;</td></tr><tr><td>Over-specific enumeration Xie Bingying; Translation</td><td>Evidence: a 2000 Huaxia Publishing House edition of Collected Works of Xie Bingying. Output: three separate volume-specific claims (Vols. 1–3), each attributed to An- hui Literary Publishing House in August 1999.</td></tr><tr><td>Repetitive generation Translation</td><td>The 6,384-word output repeatedly cy- cles through country variants of “She Aygün Kazimova; is a winner of the Golden Hit Parade&#x27; award in Russia [Ukraine, Belarus, . .. ].&quot;</td></tr></table>

Table 24: Representative omissions and hallucination patterns. Evidence is drawn from the reference pool; Output is the generated biography or its claim decomposition.

## G.4 Human Evaluation of Writing Quality

Sampling and annotation. We evaluate outputs from Mistral-Small-3.2-24B-IT, the strongest generator in our automatic evaluation. For each language, we sort the outputs from its best-performing evidence format by word count—Claims for French and Chinese, and Translation for Azerbaijani—and select five approximately evenly spaced ranks (0, 25, 50, 74, and 99). We then collect the Raw, Translation, and Claims outputs for the same five entities. This produces 45 outputs: three languages, five biography pairs per language, and three methods. The five buckets are therefore length-based sampling strata defined using the seed method, from shortest (B1) to longest (B5), rather than post-hoc word-count bins for each individual output.

Three computer-science Ph.D. researchers who use English as a working language independently rate all 45 outputs on a 1–5 Likert scale. Readability/Fluency covers grammar, wording, sentence flow, and naturalness; Coherence/Integration covers abrupt insertions, disconnected facts, and local and global flow; and Wikipedia-Style Writing covers encyclopedic tone, section structure, and biographical style. Method labels were retained in the annotation interface for traceability, so the evaluation was not method-blind. We therefore treat it as a writing-quality sanity check rather than a definitive editorial assessment.

<table><tr><td>Method</td><td>Read.</td><td>Coh.</td><td>Wiki.</td></tr><tr><td>Claims</td><td>3.933</td><td>3.867</td><td>3.822</td></tr><tr><td>Raw</td><td>3.844</td><td>3.911</td><td>3.778</td></tr><tr><td>Translation</td><td>3.756</td><td>3.733</td><td>3.689</td></tr></table>

Table 25: Human ratings of writing quality (1–5; higher is better), averaged over three annotators and 15 outputs per method. Read., Coh., and Wiki. denote Readability/Fluency, Coherence/Integration, and Wikipedia-Style Writing.

Length-stratified results. Table 26 reports the full method-by-length means. Writing quality generally declines as the sampled biographies become longer for all methods. Claims has the highest scores on all three dimensions in B5. This longestbucket pattern is descriptive: each method–bucket cell contains only three outputs (one per language) and nine ratings, and the Translation B5 cell includes the 6,384-word repetitive Azerbaijani output described in Appendix G.3.

Agreement and aggregate-score reliability. Table 27 reports ordinal Krippendorff’s α, which ranges from 0.507 to 0.537 for the subjective discourse-level judgments. Because the reported output scores average three independent ratings, we also compute two-way random-effects, absoluteagreement intraclass correlations. ICC(2,1) measures the reliability of a single rating, whereas ICC(2,3) measures the reliability of the mean of three ratings; the latter ranges from 0.801 to 0.817. ICC treats the Likert scale as approximately interval and is therefore complementary to the ordinal agreement statistic.

<table><tr><td>Dimension</td><td>Method</td><td>B1</td><td>B2</td><td>B3</td><td>B4</td><td>B5</td></tr><tr><td rowspan="3">Readability</td><td>Claims</td><td>4.889</td><td>4.333</td><td>4.000</td><td>3.222</td><td>3.222</td></tr><tr><td>Raw</td><td>4.889</td><td>4.333</td><td>3.444</td><td>3.556</td><td>3.000</td></tr><tr><td>Translation</td><td>4.889</td><td>4.111</td><td>3.667</td><td>3.667</td><td>2.444</td></tr><tr><td rowspan="3">Coherence</td><td>Claims</td><td>4.778</td><td>4.222</td><td>4.000</td><td>3.444</td><td>2.889</td></tr><tr><td>Raw</td><td>4.889</td><td>4.444</td><td>3.889</td><td>3.667</td><td>2.667</td></tr><tr><td>Translation</td><td>4.667</td><td>4.222</td><td>3.889</td><td>3.667</td><td>2.222</td></tr><tr><td rowspan="3">Wikipedia style</td><td>Claims</td><td>4.889</td><td>4.222</td><td>3.889</td><td>3.444</td><td>2.667</td></tr><tr><td>Raw</td><td>4.889</td><td>4.444</td><td>3.556</td><td>3.667</td><td>2.333</td></tr><tr><td>Translation</td><td>4.556</td><td>4.000</td><td>3.778</td><td>3.778</td><td>2.333</td></tr></table>

Table 26: Writing-quality ratings by method and length-based sampling stratum. Each cell averages nine ratings: three outputs (one per language), each rated by three annotators. Bold marks the highest score within each dimension and bucket, including ties.

<table><tr><td>Dimension</td><td>α</td><td>ICC(2,1) ICC(2,3)</td></tr><tr><td>Readability</td><td>0.512</td><td>0.572 0.801</td></tr><tr><td>Coherence</td><td>0.507</td><td>0.598 0.817</td></tr><tr><td>Wikipedia style 0.537</td><td></td><td>0.592 0.813</td></tr></table>

Table 27: Inter-annotator agreement and reliability of single-rater and three-rater mean scores. Krippendorff’s α uses the ordinal distance.

## H Main Prompt

Due to space constraints, this appendix reports the main prompts used in our pipeline. The complete prompt set, including variants for all adapted claim extraction frameworks and all enrichment generation methods, is provided in the accompanying public repository.

## H.1 Prompt for X-Claimify

• Step 1 - Selection: Figure 10 - Figure 12.

• Step 2 - Disambiguation: Figure 13 - Figure 15.

• Step 3 - Decomposition: Figure 16 - Figure 19.

## H.2 Prompt for LLM classifier

We provide the system prompt in Figure 22 and Figure 23. The user prompt is in Figure 24.

## H.3 Prompt for "English + non-English" Generation

We provide the system prompt in Figure 25 and the user prompt in Figure 26.

## H.4 Prompt for LLM-based Biography Translation

We provide the system prompt and the user prompt in Figure 27. We keep the same prompts for all three open-source LLMs.

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Model</td><td colspan="2">Claim Growth</td><td colspan="2">Supported Additions</td><td colspan="3">Hallucination</td><td>Overall</td></tr><tr><td>#Gain↑</td><td>Coverage%↑</td><td>Valid%↑</td><td>#Valid↑</td><td>Unsup.%↓</td><td>Contr.%↓</td><td>Halluc.%↓</td><td>Bal.↑</td></tr><tr><td colspan="2">Cross-Lingual Generation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">AZ</td><td>Gemma-4-31B-IT</td><td>+17.66</td><td>22.55</td><td>69.99</td><td>12.36</td><td>29.36</td><td>0.65</td><td>30.01</td><td>10.79</td></tr><tr><td>Qwen3.6-27B</td><td>+23.41</td><td>29.90</td><td>66.12</td><td>15.48</td><td>32.81</td><td>1.07</td><td>33.88</td><td>11.26</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+19.41</td><td>24.79</td><td>71.52</td><td>13.88</td><td>27.58</td><td>0.90</td><td>28.48</td><td>20.54</td></tr><tr><td rowspan="3">FR</td><td>Gemma-4-31B-IT</td><td>+21.94</td><td>24.54</td><td>72.43</td><td>15.89</td><td>27.08</td><td>0.49</td><td>27.57</td><td>30.33</td></tr><tr><td>Qwen3.6-27B</td><td>+27.22</td><td>30.45</td><td>68.35</td><td>18.61</td><td>31.06</td><td>0.58</td><td>31.65</td><td>28.77</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+24.94</td><td>27.90</td><td>74.08</td><td>18.47</td><td>25.23</td><td>0.70</td><td>25.92</td><td>44.24</td></tr><tr><td rowspan="3">ZH</td><td>Gemma-4-31B-IT</td><td>+19.53</td><td>30.90</td><td>71.86</td><td>14.03</td><td>26.90</td><td>1.24</td><td>28.14</td><td>22.03</td></tr><tr><td>Qwen3.6-27B</td><td>+24.79</td><td>39.22</td><td>67.77</td><td>16.80</td><td>30.59</td><td>1.64</td><td>32.23</td><td>20.62</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+24.19</td><td>38.28</td><td>72.44</td><td>17.52</td><td>26.22</td><td>1.34</td><td>27.56</td><td>36.24</td></tr><tr><td colspan="2">Machine Translation and Generation Pipeline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Gemma-4-31B-IT</td><td>+19.43</td><td>24.81</td><td>69.27</td><td>13.46</td><td>29.80</td><td>0.93</td><td>30.73</td><td>12.75</td></tr><tr><td>AZ</td><td>Qwen3.6-27B</td><td>+23.67</td><td>30.23</td><td>66.73</td><td>15.80</td><td>32.29</td><td>0.98</td><td>33.27</td><td>14.11</td></tr><tr><td></td><td>Mistral-Small-3.2-24B-IT</td><td>+32.47</td><td>41.47</td><td>72.10</td><td>23.41</td><td>26.66</td><td>1.24</td><td>27.90</td><td>56.54</td></tr><tr><td rowspan="2">FR</td><td>Gemma-4-31B-IT</td><td>+24.48</td><td>27.38</td><td>71.63</td><td>17.53</td><td>27.84</td><td>0.53</td><td>28.37</td><td>34.02</td></tr><tr><td>Qwen3.6-27B</td><td>+28.62</td><td>32.01</td><td>69.40</td><td>19.86</td><td>29.99</td><td>0.60</td><td>30.60</td><td>36.20</td></tr><tr><td rowspan="2"></td><td>Mistral-Small-3.2-24B-IT</td><td>+33.66</td><td>37.65</td><td>75.48</td><td>25.41</td><td>23.82</td><td>0.70</td><td>24.52</td><td>73.18</td></tr><tr><td>Gemma-4-31B-IT</td><td>+22.13</td><td>35.02</td><td>69.12</td><td>15.30</td><td>29.33</td><td>1.55</td><td>30.88</td><td>18.97</td></tr><tr><td rowspan="2">ZH</td><td>Qwen3.6-27B</td><td>+27.33</td><td>43.24</td><td>68.32</td><td>18.67</td><td>29.57</td><td>2.11</td><td>31.68</td><td>28.90</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+30.73</td><td>48.62</td><td>71.65</td><td>22.02</td><td>26.63</td><td>1.72</td><td>28.35</td><td>50.27</td></tr><tr><td colspan="2">Claim-Based Enrichment (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">AZ</td><td>Gemma-4-31B-IT</td><td>+22.19</td><td>28.34</td><td>71.67</td><td>15.90</td><td>27.91</td><td>0.43</td><td>28.33</td><td>28.25</td></tr><tr><td>Qwen3.6-27B</td><td>+28.32</td><td>36.17</td><td>69.61</td><td>19.71</td><td>29.79</td><td>0.60</td><td>30.39</td><td>36.25</td></tr><tr><td rowspan="2"></td><td>Mistral-Small-3.2-24B-IT</td><td>+22.30</td><td>28.48</td><td>77.94</td><td>17.38</td><td>21.51</td><td>0.55</td><td>22.06</td><td>51.07</td></tr><tr><td>Gemma-4-31B-IT</td><td>+25.37</td><td>28.38</td><td>74.78</td><td>18.97</td><td>24.91</td><td>0.31</td><td>25.22</td><td>48.00</td></tr><tr><td rowspan="3">FR</td><td>Qwen3.6-27B</td><td>+32.89</td><td>36.79</td><td>73.86</td><td>24.29</td><td>25.88</td><td>0.26</td><td>26.14</td><td>64.62</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+32.65</td><td>36.52</td><td>80.31</td><td>26.22</td><td>19.36</td><td>0.33</td><td>19.69</td><td>89.57</td></tr><tr><td>Gemma-4-31B-IT</td><td>+23.79</td><td>37.64</td><td>78.48</td><td>18.67</td><td>21.15</td><td>0.37</td><td>21.52</td><td>57.23</td></tr><tr><td rowspan="3">ZH</td><td>Qwen3.6-27B</td><td>+28.42</td><td>44.97</td><td>77.47</td><td>22.02</td><td>22.16</td><td>0.37</td><td>22.53</td><td>66.50</td></tr><tr><td>Mistral-Small-3.2-24B-IT</td><td>+25.07</td><td>39.67</td><td>84.05</td><td>21.07</td><td>15.65</td><td>0.30</td><td>15.95</td><td>81.42</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 28: Full generation results on CLAW-4L. Counts are averages per biography and rates are percentages. #Gain is the increase in English claims after enrichment, Coverage% normalizes #Gain by the number of selected non-English-side enrichment claims, and #Valid estimates the number of gained claims supported by the reference pool. Unsup.% aggregates unsupported, not-relevant, and unknown claims; Halluc.% is the sum of Unsup.% and Contr.%. Bal. is an overall trade-off score between supported additions and supported-claim rate.

```haskell
System Prompt for X-Claimify - Selection
You are an assistant to a fact - checker . You will be given an excerpt from a
biography . If it contains "[...]" , this means that you are NOT seeing all
sentences in the biography . You will also be given a particular sentence of
interest from the biography . The excerpt and sentence may be written in any
language . Your task is to determine whether this particular sentence contains
at least one specific and verifiable proposition , and if so , to return a
complete English sentence that contains only verifiable information .
Note the following rules :
- It does NOT matter whether the proposition is true or false .
- It does NOT matter whether the proposition is important to the biography .
- It does NOT matter whether the proposition contains ambiguous terms , e.g., a
pronoun without a clear antecedent . Assume that the fact - checker has the
necessary information to resolve all ambiguities .
You will NOT consider whether a sentence contains a citation when determining
if it has a specific and verifiable proposition .
Preserve all verifiable information in the sentence . Only remove content that
is subjective , evaluative , speculative , interpretive , or otherwise not
specifically verifiable .
If the sentence contains a specific and verifiable proposition , the final
rewritten sentence must be in English .
Do NOT simplify the sentence more than necessary .
You must consider the preceding and following sentences when determining if the
sentence has a specific and verifiable proposition . For example :
- if preceding sentence = " Marie Curie moved to Paris in 1891." and sentence = "
There she began studying at the Sorbonne ." then sentence contains a specific
and verifiable proposition .
if preceding sentence = " Frida Kahlo became known for her self - portraits ." and
sentence = " These works often drew on her personal experiences ." then
sentence contains a specific and verifiable proposition .
if preceding sentence = " Angela Merkel served as Chancellor of Germany ." and
sentence = "She later led the government during the financial crisis ." then
sentence contains a specific and verifiable proposition .
if preceding sentence = "Ada Lovelace worked with Charles Babbage ." and
sentence = " Lovelace described the machine 's potential in her notes ." then
sentence contains a specific and verifiable proposition .
if sentence = "Her life and career included many achievements across different
fields " and the following sentences expand on this point ( e . g . , give specific
achievements , positions , or dates ), then sentence is an introduction and
does NOT contain a specific and verifiable proposition .
if sentence = "In summary , Marie Curie left a lasting impact on science and
society " and the preceding sentences provide details on these topics , then
sentence is a conclusion and does NOT contain a specific and verifiable
proposition .
Here are some examples of sentences that do NOT contain any specific and
verifiable propositions :
She is widely regarded as an inspiring figure
Her achievements were truly remarkable
- Wangari Maathai played an important role in history
- Her story shows the power of perseverance
- This implies that she was a courageous person
Here are some examples of sentences that likely contain a specific and verifiable
proposition and how they can be rewritten to only include verifiable
information :
Her groundbreaking work in chemistry changed the world -> "She worked in
chemistry "
- She became one of the most influential politicians of her time -> "She was a
politician "
- Frida Kahlo 's distinctive artistic style made her one of Mexico 's most
celebrated painters -> " Frida Kahlo was a painter "
```  
Figure 10: Prompt for X-Claimify - Selection (1)

![](images/b7416b050511f85621bdc6588e961e7039af8d4b49f9abdb27d29905bae31e84.jpg)

![](images/25c0a540a661ebcec60dd52d8eea0bc92f7ea5a29ec01cbed2359aea6aa26ed8.jpg)  
Figure 11: Prompt for X-Claimify - Selection (2)  
Figure 12: Prompt for X-Claimify - Selection (3)

![](images/62fbd6fd4b9746e00e4e08f642de84f40ed948055ae94a4d2526363aaac7b393.jpg)  
Figure 13: Prompt for X-Claimify - Disambiguation (1)

System Prompt for X-Claimify - Disambiguation (Continued)   
4. Biography subject = " Angela Merkel ", Context = " Angela Merkel served as   
Chancellor of Germany . She later led the government during the financial   
crisis .", Sentence = " She later led the government during the financial   
crisis ."   
- For referential ambiguity , " She " is unclear . A group of readers shown the   
biography subject and the context would likely reach consensus that "She"   
refers to Angela Merkel .   
"the government " and "the financial crisis " are not further specified in the   
biography subject or the context , so they remain unchanged .   
DecontextualizedSentence : Angela Merkel later led the government during the   
financial crisis .   
5. Biography subject = " Jane Doe ", Context = " Jane Doe collaborated with several   
researchers during her early academic career .", Sentence = " They later moved   
to Berlin ."   
- For referential ambiguity , " They " is unclear in isolation . A group of readers   
shown the biography subject and the context would likely fail to reach   
consensus about whether " They " refers to Jane Doe and one specific   
collaborator , Jane Doe and several collaborators , or only the collaborators .   
- DecontextualizedSentence : Cannot be decontextualized   
Your output must adhere to the following format exactly . Do NOT remove any   
section headers . Only replace the content inside the <insert > tags . Do NOT   
add any extra headers , commentary , or text outside these five sections .   
Sentence :   
<insert the original sentence exactly as given >   
Incomplete Names , Acronyms , Abbreviations :   
<insert step -by - step reasoning about whether the sentence contains any partial   
names , acronyms , or abbreviations that can be resolved using the biography   
subject or the context >   
Linguistic Ambiguity in the Sentence :   
<insert step -by - step reasoning about referential ambiguity and structural   
ambiguity . If the sentence could be interpreted in multiple ways , explicitly   
write : "The sentence could be interpreted as: ..." before judging whether   
readers would likely reach consensus >   
Changes Needed to Decontextualize the Sentence :   
<insert a list of all changes needed to make the sentence fully decontextualized ;   
if no changes are needed , write " None "; if the sentence cannot be   
decontextualized , write "N/A">   
DecontextualizedSentence :   
<insert the final decontextualized sentence ; if the sentence cannot be   
decontextualized , write exactly " Cannot be decontextualized ">

Figure 14: Prompt for X-Claimify - Disambiguation (2)  
![](images/aa6440a756ed4afa60786ca3d9555af9733c7e4ad90d57e848035aca8459eb81.jpg)  
Figure 15: Prompt for X-Claimify - Disambiguation (3)

System Prompt for X-Claimify - Decomposition   
You are an assistant for a group of fact - checkers . You will be given a biography   
subject , an excerpt from a biography , and a particular sentence from the   
biography . If the excerpt contains "[...]" , this means that you are NOT   
seeing all sentences in the biography . The text before and after this   
sentence will be referred to as " the context ".   
Your task is to identify all specific and verifiable propositions in the sentence   
, decompose them if necessary , and output them as a JSON list of dictionaries   
, where each dictionary corresponds to exactly one claim .   
A proposition is " decontextualized " if:   
(1) it is fully self - contained , meaning it can be understood in isolation without   
the biography subject , the context , or the other propositions , AND   
(2) its meaning in isolation matches its meaning when interpreted alongside the   
biography subject , the context , and the other propositions .   
The propositions should be the simplest possible discrete units of information ,   
but they should NOT be decomposed into trivial subparts that are not   
independently worth fact - checking .   
Note the following rules :   
- Assume the sentence has already passed the selection and disambiguation stages .   
Do NOT decide whether the sentence is worth fact - checking . Only extract and   
decompose the specific and verifiable propositions that it contains .   
- A sentence may contain zero , one , or multiple specific and verifiable   
propositions .   
- Do NOT include subjective , evaluative , speculative , rhetorical , or interpretive   
content as propositions unless the sentence factually attributes that   
content to a source or speaker .   
- Do NOT use any external knowledge beyond what is stated in the biography   
subject , the context , and the sentence .   
- Do NOT include any citations .   
Sentences like the following do NOT contain a specific and verifiable proposition   
- Marie Curie 's legacy remains unmatched   
- Wangari Maathai became a symbol of hope for many people   
- Frida Kahlo 's life story continues to inspire generations   
- The exhibition marked a major cultural turning point   
- Scholars often regard her as one of the most visionary figures of her time   
- This suggests that she possessed extraordinary emotional strength   
Additional extraction rules :   
- Sometimes a sentence is partly evaluative or general , but still contains a   
specific and verifiable proposition . In such cases , extract only the   
verifiable part .   
- If the sentence contains coordination , apposition , or multiple factual units ,   
decompose it into as many specific and verifiable propositions as needed , but   
no further .   
- Do NOT decompose a proposition into trivial subparts that are not independently   
useful for fact - checking . For example , do NOT split "Ada Lovelace described   
the machine 's potential in her notes " into "Ada Lovelace had notes " and "The   
machine had potential ".   
- If a referential term , partial name , acronym , abbreviation , temporal expression   
, or location can be clarified using the biography subject or the context ,   
clarify it in the final proposition .   
- If a relative expression such as " there ", " then ", " later ", or " that year "   
cannot be made standalone from the visible context , do NOT guess . Only omit   
it if removing it does NOT materially change the factual content being   
extracted .   
- If the excerpt contains "[...]" , do NOT assume hidden information unless it is   
strongly implied by the sentence and the visible context .  
Figure 16: Prompt for X-Claimify - Decomposition (1)

```snap
System Prompt for X-Claimify - Decomposition (Continued)
For each claim , output one dictionary with exactly the following keys :
" subject ": the entity that the claim is about ; use a fully clarified string if
possible , otherwise null
" predicate ": the main relation or event expressed by the claim ; use a short
verbal phrase if possible , otherwise null
" object ": the main object , complement , non - English , or content of the claim ;
fill this whenever possible as part of the core subject - predicate - object
structure , even if it overlaps with a more specific semantic field such as "
location " or " time "
" time ": the time expression if explicitly stated or clearly resolvable from the
visible context ; otherwise null
" location ": the location expression if explicitly stated or clearly resolvable
from the visible context ; otherwise null
" reason ": the reason or purpose expression only if explicitly stated in the
claim ; otherwise null
" manner ": the manner or means expression only if explicitly stated in the claim
; otherwise null
" hedge ": the exact hedge word or phrase if the claim is hedged (for example : "
reportedly " , " may have " , " according to some accounts ") ; otherwise " No "
" claim ": the final specific , verifiable , and fully decontextualized claim as a
single string
" subject ", " predicate ", and " object " form the core claim frame and should be
filled whenever possible . Other fields provide optional semantic refinements .
Important formatting rules :
- Output must be valid JSON .
- Output must be a JSON list .
- Each item in the list must be a dictionary with exactly the keys listed above .
- All values must be either a string or null , except :
- " hedge ", which must be either a string or "No"
- " claim " , which must always be a single string
- If the sentence contains no specific and verifiable propositions , output [].
- Do NOT output any explanation , markdown , headers , or extra text outside the
JSON .
Here are some correct examples :
Example 1:
Biography subject = " Marie Curie "
Context = " Marie Curie moved to Paris in 1891. There she began studying at the
Sorbonne ."
Sentence = " There she began studying at the Sorbonne ."
Output =
<sup>[</sup><sub>{</sub>
" subject ": " Marie Curie ",
" predicate ": " began studying ",
" object ": " the Sorbonne " ,
" time ": null ,
" location ": " Paris ",
" reason ": null ,
" manner ": null ,
" hedge ": "No",
" claim ": " Marie Curie began studying at the Sorbonne in Paris ."
<sup>}</sup><sub>]</sub>
```  
Figure 17: Prompt for X-Claimify - Decomposition (2)

System Prompt for X-Claimify - Decomposition (Continued)   
Example 2:   
Biography subject = " Wangari Maathai "   
Context = " Wangari Maathai founded the Green Belt Movement in 1977 and later   
received the Nobel Peace Prize ."   
Sentence = " Wangari Maathai founded the Green Belt Movement in 1977 and later   
received the Nobel Peace Prize ."   
Output =   
<sup>[</sup><sub>{</sub>   
" subject ": " Wangari Maathai ",   
" predicate ": " founded ",   
" object ": " the Green Belt Movement ",   
" time ": "1977" ,   
" location ": null ,   
" reason ": null ,   
" manner ": null ,   
" hedge ": "No",   
" claim ": " Wangari Maathai founded the Green Belt Movement in 1977."   
} ,   
{   
" subject ": " Wangari Maathai ",   
" predicate ": " received ",   
" object ": " the Nobel Peace Prize ",   
" time ": null ,   
" location ": null ,   
" reason ": null ,   
" manner ": null ,   
" hedge ": " No " ,   
" claim ": " Wangari Maathai received the Nobel Peace Prize ."   
<sup>}</sup><sub>]</sub>   
Example 3:   
Biography subject = " Maya Angelou "   
Context = " According to some accounts , Maya Angelou traveled by train to   
California in 1940 to join her mother ."   
Sentence = " According to some accounts , Maya Angelou traveled by train to   
California in 1940 to join her mother ."   
Output =   
<sup>[</sup><sub>{</sub>   
" subject ": " Maya Angelou " ,   
" predicate ": " traveled ",   
" object ": " California ",   
" time ": "1940" ,   
" location ": " California ",   
" reason ": "to join her mother ",   
" manner ": " by train " ,   
" hedge ": " according to some accounts ",   
" claim ": " According to some accounts , Maya Angelou traveled by train to   
California in 1940 to join her mother ."   
<sup>}</sup><sub>]</sub>  
Figure 18: Prompt for X-Claimify - Decomposition (3)

User Prompt for X-Claimify - Decomposition   
Biography subject :   
{ subject }   
Excerpt :   
{ excerpt }   
Sentence :   
{ sentence }  
Figure 19: Prompt for X-Claimify - Decomposition (4)

```jsonl
System Prompt for Claim Infobox Extraction
You are an assistant for a group of fact - checkers .
You will be given exactly one already - written claim . Your task is NOT to extract
new claims from a sentence , NOT to split the claim into multiple propositions
, and NOT to rewrite the dataset . Your task is only to fill the infobox
fields that describe the given claim .
For the given claim , output exactly one JSON object with exactly these keys :
- " subject ": the entity that the claim is about ; use a fully clarified string if
possible , otherwise null
" predicate ": the main relation or event expressed by the claim ; use a short
verbal phrase if possible , otherwise null
" object ": the main object , complement , non - English , or content of the claim ;
fill this whenever possible as part of the core subject - predicate - object
structure , even if it overlaps with a more specific semantic field such as "
location " or " time "
" time ": the time expression if explicitly stated in the claim ; otherwise null
- " location ": the location expression if explicitly stated in the claim ;
otherwise null
" reason ": the reason or purpose expression only if explicitly stated in the
claim ; otherwise null
- " manner ": the manner or means expression only if explicitly stated in the claim
; otherwise null
- " hedge ": the exact hedge word or phrase if the claim is hedged , for example "
reportedly " , " may have " , or " according to some accounts "; otherwise " No "
- " claim ": copy the input claim exactly , except for trimming surrounding
whitespace
Example :
Claim : According to some accounts , Maya Angelou traveled by train to California
in 1940 to join her mother .
Output :
{
" subject ": " Maya Angelou ",
" predicate ": " traveled " ,
" object ": " California ",
" time ": "1940" ,
" location ": " California " ,
" reason ": "to join her mother ",
" manner ": " by train " ,
" hedge ": " according to some accounts " ,
" claim ": " According to some accounts , Maya Angelou traveled by train to
California in 1940 to join her mother ."
}
```  
Figure 20: Prompt for Extraction Infobox (1)

![](images/12a529a0ddbe5d47810239d3b2d774b0b59c6a39b065506971a121e06be833f7.jpg)  
Figure 21: Prompt for Extraction Infobox (2)

System Prompt for Claim Alignment - w/o infobox   
You are an expert cross - lingual claim relation annotator .   
First decide Alignment , then decide Enrichment only when Alignment is " Aligned ".   
Alignment choices :   
- Aligned   
- Contradicted   
- Not Relevant   
Enrichment choices ( only for aligned pairs ):   
- None   
- A > B   
- B > A   
- A <> B   
Label definitions :   
- Not Relevant : Claim A and Claim B are about different factual aspects , even if   
they refer to the same person / entity . A factual aspect can be a date , place ,   
occupation , relationship , event , award , work , role , quantity , or other   
biographical fact .   
Contradicted : Claim A and Claim B are about the same factual aspect , but assert   
mutually exclusive values or states , so they cannot both be true .   
- Aligned : Claim A and Claim B are about the same factual aspect and can both be   
true . This includes exact paraphrases , translation variants , precision   
differences , and non - conflicting added detail .   
- Enrichment None : Use when both claims contain the same set of atomic facts with   
no additional details on either side .   
- Enrichment A > B: Use when Claim A includes the factual content of Claim B and   
adds one or more non - conflicting atomic facts , details , qualifiers , dates ,   
locations , quantities , roles , or context .   
- Enrichment B > A: Use when Claim B includes the factual content of Claim A and   
adds one or more non - conflicting atomic facts , details , qualifiers , dates ,   
locations , quantities , roles , or context .   
- Enrichment A <> B: Use when Claim A and Claim B are aligned and non -   
contradictory , but Claim A contains one or more non - conflicting atomic facts   
missing from Claim B, and Claim B also contains one or more non - conflicting   
atomic facts missing from Claim A.   
Decision process :   
1) Decide Alignment first .   
2) If Alignment is not " Aligned ", Enrichment must be an empty string "".   
3) If Alignment is " Aligned " , Enrichment must be one of : " None " , " A > B " , " B > A   
", "A <> B".   
4) Mandatory bidirectional check before final output :   
- Evaluate once with the original order (A , B ) , and once with swapped order (B   
, A).   
- Final decision must be order - consistent : Alignment must stay the same after   
swapping .   
- For aligned pairs , Enrichment must invert after swapping ("A > B" <-> "B > A   
") , remain " None " , or remain " A <> B ".   
- If the two directions disagree , re - check atomic facts and resolve the   
inconsistency before output .  
Figure 22: Prompt for Alignment (1)

![](images/63b4da13d5c8709ea27c805eaeb51537b92026f55bd719b87b972ffc97eccd81.jpg)  
Figure 23: Prompt for Alignment (2)

![](images/16a47fee1761df370703ef498e9462ccc37d8ef12d21a795b1351cec1e209d82.jpg)  
Figure 24: Prompt for Alignment (3)

![](images/8d0d91876f2499acb669431567029c32e8d5fff9ceb53d5e8436b6abc569b5c4.jpg)  
Figure 25: System Prompt for Cross-Lingual Generation

User Prompt for Cross-Lingual Generation   
Task :   
Revise the current English biography by incorporating factual information from   
the non - English - language excerpt that is missing from the current English   
biography .   
Metadata :   
- qid : {{ qid }}   
- original\_language : {{ original\_language }}   
- non - English\_language : {{ non - English\_language }}   
english\_wikipedia\_name : {{ english\_wikipedia\_name }}   
original\_language\_wikipedia\_name : {{ original\_language\_wikipedia\_name }}   
wikidata\_name : {{ wikidata\_name }}   
- nationality : {{ nationality }}   
- occupations : {{ occupations }}   
Requirements :   
- Treat the current English biography as the primary reference when conflicts   
occur .   
- Integrate new facts from the non - English excerpt into the English biography   
when missing .   
- Added or revised facts must be supported by the non - English excerpt .   
- Do not introduce facts from outside the provided non - English excerpt .   
- If a fact already exists in English and is consistent , do not rewrite it   
unnecessarily .   
- Keep chronology and factual consistency .   
- Place newly added facts in contextually appropriate locations for fluent   
biography writing .   
- Avoid repeated statements about the same fact .   
- Keep the original section layout whenever possible ; add new sections only when   
necessary .   
- Do not output notes , bullets , or explanations .   
- Output only the full revised English biography .   
Current English biography :   
<<< CURRENT\_ENGLISH\_BIO\_START >>>   
{{ current\_english\_bio }}   
<<< CURRENT\_ENGLISH\_BIO\_END >>>   
non - English biography excerpt :   
<<<non - English\_BIO\_EXCERPT\_START >>>   
{{ non - English\_bio\_excerpt }}   
<<<non - English\_BIO\_EXCERPT\_END >>>  
Figure 26: User Prompt for Cross-Lingual Generation

```twig
Prompt for LLM-based Biography Translation
system_prompt : |
You are a professional translator for Wikipedia biographies .
Translate the source text into natural , faithful English .
Keep all facts , dates , names , numerals , and structure .
Do not omit any information .
Do not add explanations , notes , or comments .
Output only the translated English text .
user_prompt_template : |
Task : Translate the following biography chunk into English .
Metadata :
- original_language : {{ original_language }}
- non - English_language : {{ non - English_language }}
- english_wikipedia_name : {{ english_wikipedia_name }}
- original_language_wikipedia_name : {{ original_language_wikipedia_name }}
- wikidata_name : {{ wikidata_name }}
- nationality : {{ nationality }}
occupations : {{ occupations }}
- qid : {{ qid }}
- chunk_id : {{ chunk_id }}
chunk_index : {{ chunk_index }}
total_chunks_for_qid : {{ total_chunks_for_qid }}
Requirements :
- Preserve all facts and details faithfully .
- Keep section markers like ** SECTION_NAME ** if present .
- Keep list structure and paragraph boundaries when possible .
- Do not summarize and do not omit any content .
- Output only translated English text for this chunk .
Source text to translate :
<<< SOURCE_CHUNK_START >>>
{{ source_chunk }}
<<< SOURCE_CHUNK_END >>>
```  
Figure 27: Prompt for LLM-based Biography Translation