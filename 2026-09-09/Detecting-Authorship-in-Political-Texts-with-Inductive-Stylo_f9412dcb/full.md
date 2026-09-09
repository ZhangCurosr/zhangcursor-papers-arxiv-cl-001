# Detecting Authorship in Political Texts with Inductive Stylometry

Gennadii Iakovlev<sup>∗</sup>

Levente Littvay<sup>†</sup>

September 2026

## Abstract

Political texts are rarely authored by the nominal speaker alone. Tweets, speeches, reports, and oficial statements are drafted, edited, or harmonized by staf, yet political science has paid limited attention to the stylistic traces these hidden authors leave behind. This paper develops and stress-tests an inductive stylometric approach for recovering latent authorship structure in political communication, combining character 3-gram features with UMAP dimensionality reduction, and Burrows’ Delta. We apply the approach to six corpora that vary in length (from tweets to long documents), in mode (written and oral), and in language (English and Hungarian). The approach recovers near-disjoint analyst fingerprints in formal legal prose in both languages, sorts a politician’s tweets into validated subsets while uncovering additional insights, and distinguishes scripted from improvised speech. It fails, however, to resolve individual speechwriters within scripted corpora. Frequency-based stylometry is thus a powerful tool that, depending on authorial signal strength and institutional editing, can uncover authorship traces relevant to legislative studies, political communication, and policy research.

## 1 Introduction

Stylometry is a quantitative text analytical technique historically reserved for verifying authorship in disputed literary cases (Stamatatos, 2009). Perhaps the most famous modern application was the forensic identification of J.K. Rowling as the true author of the crime novel The Cuckoo’s Calling, unmasking her pseudonym “Robert Galbraith” through distinct syntactic fingerprints (Juola, 2015). Yet, despite the origins of quantitative stylometry traced to the identification of Hamilton as the author of twelve disputed Federalist papers (Mosteller and Wallace, 1964), and the explosion of computational text analysis in political science since the turn of the millennium (Grimmer and Stewart, 2013; Wilkerson and Casas, 2017), stylometric approaches have received little to no attention in the field.<sup>1</sup> We argue that neglecting stylometry is a critical oversight, particularly given the increasing opacity surrounding the true authorship of political texts. Political analysis should therefore account for the staf and other contributors who draft speeches, court decisions, laws, and diplomatic documents.

Our goal is to apply and stress-test classical stylometric approaches in the noisy domain of political communication, stretching their applicability across three distinct directions: from written literary texts to delivered speeches, from long-form documents to short social-media posts, and from English to the morphologically complex non-Indo-European language of Hungarian.

Our first two case studies apply the same algorithm to Congressional Research Service (CRS) reports, focusing on the American Law topic subset, where a small set of named staf analysts author documents that share topic, audience, and a highly formal legal register—a high-homogeneity English-language benchmark. We exploit this corpus in two nested forms. We begin with the single most controlled slice, the short-report series, in which document format and length are held constant alongside topic and register, isolating analyst style as cleanly as the data allow; we then relax the format control and pool the full American Law subset to show that the same analyst signal survives across document formats.

Third, we examine a corpus of Hungarian Ombudsman reports to test whether stylometric methods can recover diferent authors among oficial administrative legal texts—a genre written in a highly formal register that typically circulates without named individual authors. These reports, unusually, record the staf rapporteur who drafted each document, allowing us to validate what the method recovers against known authors. We chose Hungarian as it comes from a diferent language family than most European languages, allowing us to assess whether the approaches tested generalize broadly. We also use a Hungarian language case study, below, testing speech authorship.

Fourth, we aimed to benchmark these methods—most often scrutinized on novella-sized or longer works—on the short-format texts that increasingly define modern political communication. Using Donald Trump’s tweets from the Trump Twitter Archive (Brown, 2019), from his campaign launch through Election Day, we leveraged a known hardware distinction: tweets sent from an Android device (hypothesized to be Trump himself) versus an iPhone (hypothesized to be stafers) as a ground-truth proxy for authorship. This served to validate whether the stylometric signal persists in texts limited by character counts.

Our fifth case study sought to determine if stylometric approaches designed for static and written literary texts are applicable to speeches—texts that are convoluted by the "noise" of oral delivery and the editing process inherent to the collaborative nature of speechwriting. We utilized Donald Trump’s 2016 campaign speeches, distinguishing between those delivered on and of the teleprompter. This allowed us to test if stylometry can discriminate between the "Institutional Voice" (scripted) and the "Individual Voice" (ad-libbed), and whether inductive clustering could reveal distinct authorial hands within the scripted corpus.

Finally, to examine cross-linguistic generalizability and test the method’s inductive capacity in the absence of a validated ground truth, we applied the approach to a linguistically divergent case: the political speeches of Hungarian Prime Minister Viktor Orbán.

Our findings suggest that while political texts are indeed "messy", stylometry ofers significant analytical leverage. We find that the most basic stylistic clusters can efectively be established in both short and long political texts. While clear, individual "speechwriter clusters" did not inductively emerge from the speeches (suggesting a high degree of homogenization in formal addresses), stylometry significantly sharpened the separation between ad-lib delivery and teleprompter speeches and other deductively hypothesized groups (e.g., Android vs. iPhone).

## 2 Methods

Our empirical strategy relies on classical frequency-based stylometry, the family of techniques developed in authorship-attribution research over the past half century (Mosteller and Wallace, 1964; Grieve, 2007; Koppel, Schler and Argamon, 2009; Stamatatos, 2009). We construct feature sets familiar to computational social scientists: simple text-level indicators (such as length and sentiment), character 3-grams, and Burrows’ Delta on frequent words. We represent each document as a high-dimensional vector, reduce these representations to two dimensions with Uniform Manifold Approximation and Projection (UMAP) for visualisation, and assess whether the resulting layout aligns with known or hypothesised authorship signals. If the technique proves useful in cases where authorship is verifiable, we can be confident that it will also be useful as an inductive technique to detect stylometric patterns where authorship is unavailable.

## 2.1 Dimensionality reduction techniques

Frequency-based stylometry begins by converting texts into numerical form. Each document is represented as a vector of character trigram frequencies—sequences of three consecutive characters—yielding a document-by-feature matrix with several hundred columns in our corpora. This representation captures stylistic habits that are dificult to observe directly: spelling preferences, punctuation patterns, word endings, afixes, spacing, and other sub-word regularities. Documents with similar trigram patterns lie close together in this high-dimensional space. Such spaces can be analysed computationally, but visual inspection requires mapping the hundreds of original dimensions into two.

This is the purpose of dimensionality reduction. Unlike Principal Component Analysis, which extracts the strongest linear axes of variation, the methods most used for exploratory text visualization today are nonlinear and neighborhood-preserving: texts that are stylistically similar in the original space should remain close to one another in the plot.

The two most widely used methods of this kind are t-distributed stochastic neighbor embed ding, or t-SNE (van der Maaten and Hinton, 2008), and Uniform Manifold Approximation and Projection, or UMAP (McInnes, Healy and Melville, 2018).<sup>2</sup> Both are unsupervised: they arrange documents purely from the numerical patterns in the feature matrix, without access to authorship labels—exactly the property inductive stylometry requires.

The mathematical foundations of the two methods difer: t-SNE models pairwise similarities between points as probabilities and preserves them in a lower-dimensional map, while UMAP assumes the data lie on or near a lower-dimensional manifold and preserves its internal geometry. In our corpora, UMAP produced visibly crisper authorship separation than t-SNE.

The key practical issue in dimensionality reduction is the balance between local structure (the immediate neighborhood of each document) and global structure (the broader arrangement of clusters across the corpus). Too much emphasis on local detail produces a fragmented, carpetlike map in which broad communities are dificult to distinguish; too much emphasis on global separation produces very clear clusters at the cost of compressing their internal structure.

In UMAP, this balance is controlled mainly by the number of neighbors: a smaller neighbor count weights local structure, a larger one weights global structure.<sup>3</sup> For inductive stylometry we suggest emphasizing global communities with a relatively larger neighbor count, which makes the plots better suited for detecting broad authorship patterns; more local settings can serve as supplementary diagnostics when the internal structure of a cluster is of special interest.

## 2.2 Classic stylometric analysis

Our classic stylometric analysis relies on two complementary frequency-based approaches. The first is a character 3-gram model projected with UMAP, which operationalises the highdimensional representation and dimensionality-reduction logic described above. The second is Burrows’ Delta, a classic authorship-attribution measure based on frequent-word profiles. These two approaches form the main body of the analysis. Additional interpretable diagnostics— including sentiment, document length, sentence length, and related dictionary-based indicators are reported in Appendix D (Figures A3–A10).

For the character 3-gram analysis, we construct document-by-feature matrices of trigram frequencies, trimming features that occur fewer than five times across the corpus, and project them into two dimensions using UMAP with a cosine distance metric (McInnes, Healy and Melville, 2018). Every corpus is projected directly from its trimmed trigram matrix, with no intermediate reduction. Following the local–global trade-of discussed above, we scale the number of nearest neighbours with corpus size, using larger values on the bigger corpora to emphasise global community structure over fine-grained local detail.<sup>4</sup>

Alongside the trigram–UMAP analysis, we compute Burrows’ Delta on the most frequent words. Burrows’ Delta is a standard stylometric distance measure that compares texts or groups of texts according to their relative use of frequent words, especially common function words such as articles, prepositions, pronouns, and conjunctions (Burrows, 2002). These words are chosen subconsciously, and contain stable authorial habits (Kestemont, 2014); the measure has proven robust across repeated evaluations and refinements (Hoover, 2004; Argamon, 2008; Evert et al., 2017). This emphasis on style marks an important departure from much politicalscience text-as-data research. When the objective is to recover topics, positions, sentiment, frames, or policy content, frequent words are commonly removed as stopwords or downweighted through schemes such as TF–IDF so that rarer, more substantively distinctive terms dominate the representation. Stylometry reverses that priority. It focuses on the less directly contentbearing layer of language because frequent grammatical choices are ubiquitous, comparatively weakly governed by topic, and dificult for writers to regulate consistently. Function words may reveal little about a document’s subject, but their distribution can capture how a writer habitually constructs sentences and links clauses. Burrows’ Delta does not require dimensionality reduction: it directly yields distances between each text and the relevant authorial or

group reference profiles in a single dimension. The main text therefore reports the two most central stylometric diagnostics: character 3-gram UMAP plots and Burrows’ Delta results. Where authorship or production labels exist—CRS analysts, ombudsman rapporteurs, tweet device, teleprompter use—we evaluate how well the projection corresponds to the labels, summarising cluster quality with mean silhouette widths over the labelled groups, computed on the two-dimensional UMAP coordinates (Rousseeuw, 1987). In the two U.S. validation cases with two labels, we additionally report two-sample t-tests on the UMAP coordinates and the Burrows’ Delta scores. Where no labels are available, as in the Orbán corpus, the exercise is fully inductive.

Every case study is accompanied by a discriminant-validity battery, reported in Appendix D: the identical UMAP layout is re-colored by sentiment, text length, mean sentence length, mean word length, passive-voice density, and the coordination/subordination ratio, and the correlation of each factor with each UMAP dimension is computed. Low correlations indicate that the projection is not simply re-discovering document length or register. Dictionary-based sentiment scores, raw document length, sentence length, and related descriptive measures serve only as supplementary diagnostics and are reported in Appendix D.

All clustering, dimensionality reduction, inferential statistics, and visualisation are carried out in R.

## 3 Use Case 1: CRS Reports (American Law, Short-Report Series)

## 3.1 Data collection

Congressional Research Service (CRS) reports are confidential, non-partisan policy documents produced by named staf analysts for Members of Congress. The corpus was scraped from the EveryCRSReport project,<sup>5</sup> which systematises and republishes these documents. To hold subject matter and institutional register approximately constant, we restrict the sample to reports classified by EveryCRSReport under a single topic label.<sup>6</sup> Reports credited to two or more authors, as well as those without named authors that constitute roughly half of the corpus, are excluded from the analysis.

We test the stylometric methods on the American Law topic subset—1,109 single-authored reports by 282 distinct analysts after cleaning. Topic, audience, and a highly formal legal-prose register are therefore similar across every document, yet each report is attributed to a specific analyst. CRS reports are issued in several standardized formats distinguished by their reportnumber prefix: long reports (R, RL), short reports (RS), and rapid-response Insight pieces (IN), among others. Because format dictates document length and template, it is itself a stylometric confound: two analysts may appear to difer simply because one writes long reports and the other short ones. We therefore begin with a single format family—the short-report series (RS). The short-report series comprises 167 documents; we color the seven most prolific analysts and leave the remainder as grey background points.

## 3.2 Results

The character 3-gram UMAP (Figure 1) yields the cleanest analyst separation in the entire study (mean silhouette width +0.47 across the colored analysts, computed on the plotted twodimensional UMAP coordinates). The three most prolific analysts resolve into three essentially disjoint islands: Keith Bea (29 reports) occupies a tight cloud in the upper-left, Robert Keith (17) a clearly separated cloud in the lower-left, and Charles Doyle (8) a distinct region on the right, with efectively no overlap among them. The four less prolific analysts (Shawn Reese, Mildred Amer, Harold C. Relyea, and Kevin R. Kosar, each with six or seven reports) occupy their own local neighborhoods—more difuse, but still internally coherent. Burrows’ Delta corroborates the projection: within-analyst frequent-word profiles are markedly tighter than cross-analyst profiles (Appendix B, Figure A1). This slice fixes topic area, audience, genre, and document format by construction, but one confound survives the design: in our sample, each analyst’s reports concentrate in a distinct substantive area—27 of Bea’s 29 short reports treat emergency-management and disaster statutes, 16 of Keith’s 17 the congressional budget process, and 7 of Doyle’s 8 federal criminal law—so analyst identity and subject matter are empirically entangled. Three checks nevertheless separate the two. The Burrows’ Delta corroboration above rests on the 100 most frequent, and hence topically nearly empty, words; re-colouring the pooled American Law projection of Use Case 2 by oficial CRS topic tags yields a silhouette of −0.31, against +0.41 for analyst labels in identical coordinates; and analysts who write in several topic areas remain stylometrically themselves—in the wider CRS corpus, a report is more similar to its own author’s reports on other topics (mean trigram cosine 0.83) than to other analysts’ reports on the same topic (0.72; Appendix D.3, Table A1). Given the three checks, the clusters track individual style, even though each analyst has a specialization.

![](images/0a6b4a458fa9cf16419bf1ce39e4e9e4a8d0fe3e9b9a66216bdafcb636f24c45.jpg)  
Figure 1: UMAP of character 3-gram features for the seven most prolific analysts in the CRS American Law short-report (RS) series $( n = 1 6 7 )$ . Light grey = all other analysts. Holding document format constant alongside topic and register yields near-disjoint analyst clusters (silhouette +0.47).

## 4 Use Case 2: CRS Reports (Full American Law Subset)

## 4.1 Data collection

We next relax the format control of Use Case 1 and pool every report format within the American Law subset—long reports, short reports, Insight pieces, and legacy documents alike— yielding the full set of 1,109 documents attributed across 282 analysts. The corpus now mixes document lengths and templates, reintroducing format as a potential secondary axis of stylistic variation. This tests whether the analyst signal isolated in Use Case 1 survives when the format is no longer held fixed. The eight most prolific analysts are colored.

## 4.2 Results

Even with formats pooled, analyst-level structure remains strong, only marginally below the format-controlled short-report subset. The most prolific analysts again resolve into tight, well-separated islands (Figure 2): Keith Bea (67 reports) in the upper-left, Charles Doyle (73) in the upper-right, Robert Keith (47) at the bottom, and Robert Jay Dilger (26) on the left. The interior is visibly more mixed than in the short-report case, a second source of within-analyst spread introduced by format heterogeneity. Burrows’ Delta again shows within-analyst frequent-word profiles to be tighter than cross-analyst profiles (Appendix B, Figure A2). That the silhouette barely declines $( + 0 . 4 7  + 0 . 4 1 )$ when document format is no longer held constant indicates that the authorship signal dominates the format signal: pooling four document templates only slightly blurs the analyst clusters. A direct control rules out subject specialisation as the driver: re-colouring the same projection by the oficial CRS topic tags yields a silhouette of −0.31, against +0.41 for analyst labels in identical coordinates (Appendix D.3, Figure A5).

The pooled projection also reveals a pattern with methodological consequences: several analysts now occupy more than one island. Keith Bea’s reports split into the large, dense cloud on the far upper-left and a smaller satellite group near the centre of the map; Robert Keith’s documents form one dominant cluster at the bottom with a separate clump above and to its left; Barry J. McMillion and Eric Petersen likewise fragment into a principal cluster plus outlying groups. Because Use Case 1 showed that these same analysts form single coherent clusters once document format is held fixed, the most plausible reading is that the satellites are production artefacts: an analyst’s RS short reports and R/RL long reports difer enough in template, length, and boilerplate that character trigrams register them as distinct stylistic products. Two further mechanisms plausibly contribute. CRS document templates were revised repeatedly over the decades the corpus spans, so an analyst’s early and late reports inherit diferent boilerplate; and analysts change specializations over their careers, so their legal vocabulary changes too. Inductive stylometry recovers authorship within a production context: when format or period varies within a corpus, one writer’s documents may form several clusters. The islands remain internally pure: format separates documents by the same author without merging documents from diferent authors. An individual cluster can therefore still support an authorship interpretation, although the mapping from clusters to authors may be many-to-one. But this phenomenon is familiar with text analysts, such as topic modelers, where not keeping the text context and format adequately constant often yields topic clusters correlated with such contexts and formats. Clearly similar care must be taken with stylometric assessments.

Without analyst labels, the projection would still contain distinct islands and regions requiring substantive interpretation. The known analyst labels are what allow us to validate those regions as primarily authorial rather than merely exploratory clusters. This matters because it empowers research questions, such as attributing the roughly half of all anonymous CRS reports to their authors or finding and analyzing authorship patterns in fully anonymous political or policy documents.

![](images/a3b333f9530b3f2d43a4be280ce676cf076fe445e63d02c2c4eede571cafda87.jpg)  
Figure 2: UMAP of character 3-gram features for the top eight analysts in the full CRS American Law subset $( n = 1 , 1 0 9$ , all report formats pooled). Light grey = all other analysts. Analyst clusters remain clearly separated (silhouette +0.41) despite mixing document formats, but several analysts now occupy multiple islands, reflecting format-driven within-author splits.

## 5 Use Case 3: Hungarian Ombudsman Reports

## 5.1 Data collection

Having established that the approach recovers individual hands in formal English legal prose, we next ask whether it travels to a typologically distant language. The Hungarian ombudsman publishes detailed reports on individual complaints and own-initiative investigations. We analyse the reports issued during Máté Szabó’s tenure as Hungary’s general ombudsman— Parliamentary Commissioner for Civil Rights from 2007 and, after the ofice was reorganised under the new constitution, Commissioner for Fundamental Rights from 2012 until 2013. Al though every report is issued under the commissioner’s name, each names the staf rapporteur (előadó) who investigated the case and drafted the text, making this a labeled multi-author corpus of administrative legal documents in Hungarian. The full corpus contains 886 reports spanning 76 rapporteur labels; after filtering the unattributed category, 75 named rapporteurs remain. All documents share the same institutional genre, formal register, and legal subject matter, providing a tough cross-linguistic benchmark in which topic- and genre-driven clustering cannot account for rapporteur-level separation.

## 5.2 Results

The character 3-gram UMAP (Figure 3) reveals clear but uneven rapporteur-level structure among the eight most prolific drafters; the remaining rapporteurs appear as grey background points. Several rapporteurs resolve into unmistakable islands: dr. Győrfy Zsuzsanna’s reports form an elongated, nearly pure cluster along the right edge of the projection, dr. Zemplényi Adrienne and dr. Halász Zsolt each occupy compact local neighborhoods, and dr. Hajas Barnabás and dr. Bene Beáta concentrate in identifiable regions of the map. Other rapporteurs remain difuse, indicating that the strength of the recoverable fingerprint varies across individuals. Because all documents are administrative Hungarian legal texts sharing genre, register, and broad subject matter, the separation that does emerge cannot be attributed to topic, genre, or language-register diferences. The full discriminant-validity battery in Appendix D.4 (Figures A6–A7) returns near-zero correlations for sentiment, length, sentence length, word length, and syntactic factors as well, the cleanest such result in the study. The case demonstrates that character trigrams carry authorial signal even in an agglutinative, morphologically rich language, provided the underlying texts are written, formal, and independently drafted. It also previews the paper’s central substantive point: every one of these reports went out under a single nominal author—the commissioner—yet the stylistic fingerprints that the method recovers belong to the staf who actually drafted them.

![](images/16cb587c95c01257c358b94ac112abb9187306741c68ee7e32d2988f96622f68.jpg)  
Figure 3: UMAP of character 3-gram features for the eight most prolific rapporteurs in the Hungarian ombudsman corpus (Szabó era, n = 886). Grey = all other rapporteurs. Several rapporteurs form clean islands (most visibly dr. Győrfy on the right); others remain difuse.

## 6 Use Case 4: Trump Tweets by Device

## 6.1 Data collection

Moving from long-form reports to the opposite extreme of document length, we examine the canonical example of device-specific tweets. During the 2016 campaign, Donald Trump’s Twitter account was accessed primarily from an Android and an iPhone handset. Prior analysis established that the Android phone was usually in Trump’s own hands, whereas the iPhone was stafed (Robinson, 2016); subsequent linguistic work documented systematic register variation on the account over time (Clarke and Grieve, 2019). We downloaded the Trump Twitter Archive (Brown, 2019) and define the analysis window from Trump’s campaign launch on 16 June 2015 through Election Day on 8 November 2016. We exclude tweets sent from nonphone clients, rows the archive marks as retweets or deleted, and tweets that reproduce another account’s words verbatim. The resulting corpus contains 4,000 composed phone tweets—1,866 from Android and 2,134 from iPhone.

## 6.2 Results

The character 3-gram UMAP difers significantly by device on both dimensions, although the groups overlap substantially (Figure 4(a), Table 1). iPhone tweets cluster in a linkand hashtag-heavy campaign register, whereas Android tweets are dominated by unformatted prose, including policy and media attacks, reactions, and updates. This pattern is more consistent with a diference in production register than with a clean division of authorship.

The dictionary and frequent-word measures show the same production diference. iPhone tweets are shorter and more positive than Android tweets, although both gaps narrow after promotional formatting is excluded (Online Appendix), suggesting that they are partly associated with post type. Burrows’ Delta also difers significantly by device.<sup>7</sup> Because individual tweets are much shorter than the texts for which Delta was developed (Burrows, 2002; Eder, 2015), we calculate it both per tweet and on chronological 50-tweet chunks. The aggregated profiles are almost perfectly separable (Figure 5). The two streams are not contemporaneous: iPhone tweets occur later in the campaign, and their share of the account rises from two percent at launch to more than seventy percent by Election Day. Some device diferences may therefore reflect the account’s professionalisation over time (Clarke and Grieve, 2019).

Studies that use device as an authorship instrument therefore capture a meaningful but noisy division of labour on the account.

Formatting accounts for most of the device separation in the character 3-gram projection. Across the three embeddings in Figure 4(a)–(c), the mean device silhouette is 0.21 with formatting intact, 0.09 without URL characters, and 0.08 when hashtag characters are also excluded. The compact clusters that remain correspond mainly to repeated content, especially the campaign’s Make America Great Again sign-of, which appears in near-identical wording from both handsets. The small residual device diference follows the promotional register: iPhone staf posts remain shorter and more positive after links are excluded, while much of their prose uses Trump’s own idiom, including first-person attacks, signature epithets, and self-promotion (Clarke and Grieve, 2019). Device is therefore a noisy proxy for authorship. Burrows’ Delta still distinguishes the device labels, but that contrast cannot be interpreted as a clean diference between authors.

Table 1: Android vs. iPhone tweets (n = 4,000): dictionary metrics, Burrows’ Delta, and character 3-gram UMAP dimensions
<table><tr><td rowspan="2">Metric</td><td colspan="2">Device mean</td><td rowspan="2"> $\mathbf { S i g . } ^ { \mathrm { a } }$ </td></tr><tr><td>Android</td><td>iPhone</td></tr><tr><td>Dictionary features</td><td></td><td></td><td></td></tr><tr><td>Sentiment score</td><td>0.15</td><td>0.72</td><td>*</td></tr><tr><td>Tweet length (characters)</td><td>121.1</td><td>106.4</td><td>*</td></tr><tr><td>Burrows’Delta preferenceb</td><td></td><td></td><td></td></tr><tr><td>Per tweet</td><td>0.050</td><td>0.075</td><td>*</td></tr><tr><td>Per 50-tweet chunk</td><td>-0.211</td><td>0.239</td><td>*</td></tr><tr><td>Character 3-gram UMAPc</td><td></td><td></td><td></td></tr><tr><td>Dimension 1</td><td>1.68</td><td>-1.47</td><td>*</td></tr><tr><td>Dimension 2</td><td>-1.62</td><td>1.42</td><td>*</td></tr></table>

<sup>a</sup> Two-sample t-tests; <sup>∗</sup>p < 0.0001.  
b $\Delta _ { \mathrm { A n d r o i d } } - \Delta _ { \mathrm { i P h o n e } } ;$ positive values indicate greater similarity to the iPhone/staf profile.  
<sup>c</sup> Tests on UMAP coordinates are descriptive corroboration only: the coordinates carry no meaningful units and the embedding induces dependence across observations (see Methods).  
Character 3-gram UMAP: device separation vanishes as formatting characters are stripped (no tweet removed) Android (green) ys iPhone (orange). The separation is carried by the link and hashtag characters, not by the words

![](images/b470cfe6b54bf8a6a90ffecfaa9dd0669ed7227fc29d651e7932eb75b0d1f9be.jpg)  
Figure 4: Character 3-gram UMAPs of the composed Trump campaign tweets $( n = 4 , 0 0 0 )$ coloured by device (Android green, iPhone orange), under three formatting specifications. (a) With formatting intact, iPhone link-, hashtag-, and promotion-heavy tweets lie mostly on one side of the projection and Android unformatted prose on the other (mean device silhouette 0.21). (b) Excluding URL characters reduces the silhouette to 0.09. (c) Excluding hashtag characters as well reduces it to 0.08.

![](images/d344f63a94d2f80da9e684093b0b2fe4b077b869621c13241e649282478f0d52.jpg)  
Figure 5: Burrows’ Delta preference scores $( \Delta _ { \mathrm { A n d r o i d } } - \Delta _ { \mathrm { i P h o n e } } )$ for Android (green) and iPhone (orange) tweets, computed on chronological 50-tweet chunks within device. Positive values indicate greater similarity to the iPhone/staf profile. Aggregation renders the two device profiles almost perfectly separable.

## 7 Use Case 5: Trump Speeches (Teleprompter vs. Of-the-Cuf)

## 7.1 Data collection

Speeches add a layer that neither reports nor tweets possess: oral delivery, which interposes the speaker between whoever drafted the text and the transcript we observe. We gathered transcripts of 23 Trump campaign speeches from 2016–2017 and manually labelled 14 of them as teleprompter-assisted, based on the visible presence of a teleprompter in a video recording of the event; the remaining nine were coded as of-the-cuf. The speech corpus and teleprompter coding draw on the same Team Populism data used in the Guardian’s “Teleprompter Test” interactive on scripted and of-the-cuf Trump speeches (Smith et al., 2019). All delivery coding, including the identification of of-script passages used below, was completed from the video record before, and independently of, the stylometric analysis.

## 7.2 Results

Table 2 shows dictionary metrics, Burrows’ Delta, and the 3-gram UMAP dimensions. Sentiment is the only dictionary feature with a significant gap—impromptu speeches are markedly more positive—while speech length does not difer between the groups. The character 3-gram projection (Figure 6) separates the impromptu and teleprompter speeches with no overlap, although the two groups are not spatially distant. The trigram–UMAP projection separates the delivery styles, whereas t-SNE and truncated-SVD projections of TF–IDF features (not shown) leave them overlapping.

Positions within the scripted cluster correspond to observed departures from the prepared text. Teleprompter speeches extend from a compact, fully scripted core in the upper right toward the improvised cluster in the lower left. Video evidence shows repeated departures from the prepared text among speeches in the intermediate zone. The extreme case is the November 8, 2016 election-eve rally in Grand Rapids, Michigan: teleprompters were present, but the delivery was almost entirely improvised. Speeches with frequent, shorter ad-libs occupy intermediate positions, while addresses with almost no of-prompt passages, such as the August 20, 2016 rally in Fredericksburg, Virginia, lie at the scripted end. A speech-by-speech accounting of the observed of-script passages appears in Appendix C. The projection therefore captures a gradient of scriptedness rather than a binary distinction: the farther a teleprompter speech lies from the scripted core, the more of its delivery was improvised. Burrows’ Delta preference scores show a similar pattern (Figure 7): scripted speeches have a long tail toward the improvised profile, although function words still separate the two delivery modes almost perfectly.

Despite the involvement of multiple speechwriters in a campaign, the scripted corpus contains no internal subclusters that could be attributed to individual writers. Several factors may account for this. Collaborative drafting and editing may homogenise individual styles, and the amount of text per writer may be too small for reliable attribution (Eder, 2015). Delivery may also obscure individual styles: the transcripts record Trump’s timing, repetitions, and interjections alongside the prepared text. The analysis distinguishes scripted from improvised delivery but does not identify individual speechwriters.

Table 2: Of-the-cuf vs. teleprompted speeches: dictionary features, Burrows’ Delta, and character 3-gram UMAP dimensions
<table><tr><td rowspan="2">Metric</td><td colspan="2">Delivery-mode mean</td><td rowspan="2">Sig.</td></tr><tr><td>Off-the-cuff</td><td>Teleprompted</td></tr><tr><td>Dictionary features</td><td></td><td></td><td></td></tr><tr><td>Sentiment score</td><td>108.0</td><td>17.4</td><td>*</td></tr><tr><td>Speech length (words)</td><td>17,868</td><td>20,281</td><td></td></tr><tr><td>Burrows&#x27;Delta preferenceb Per speech</td><td>0.28</td><td>-0.32</td><td>*</td></tr><tr><td>Character 3-gram UMAPc</td><td></td><td></td><td></td></tr><tr><td>Dimension 1</td><td>-1.03</td><td>0.66</td><td>*</td></tr><tr><td>Dimension 2</td><td>-2.06</td><td>1.32</td><td>*</td></tr></table>

<sup>a</sup> Two-sample t-tests; $^ { * } p < 0 . 0 0 0 1$ . With 23 speeches we report significance levels rather than point p-values.  
b $\Delta { \mathrm { { u s e d } } } - \Delta$ <sub>NotUsed</sub>; positive values indicate greater similarity to the ofthe-cuf profile.

![](images/313d9d03f8ccf36b28a7f9561a63ee605f9c508420e56f9b3a75ced47a102b3f.jpg)

Figure 6: UMAP of character 3-gram features for 23 Trump campaign speeches (green = of-the-cuf, orange = teleprompted). The groups separate without overlap; the teleprompter speeches nearest the improvised cluster are those with documented of-script passages.  
![](images/97e83bcb7d4fd521ac9731a74956faad3c821574381af357bf41c778b79250a3.jpg)  
Figure 7: Burrows’ Delta preference scores $( \Delta _ { \mathrm { U s e d } } - \Delta _ { \mathrm { N o t U s e d } } ;$ positive = closer to the improvised profile) for 23 Trump speeches (green = of-the-cuf, orange = teleprompted). The long tail of the scripted distribution toward zero reflects partially improvised teleprompter speeches.

## 8 Use Case 6: Orbán’s Speeches in Hungarian

## 8.1 Data collection

The final case examines whether speeches attributed to Hungary’s prime minister, Viktor Orbán, show evidence of multiple authorship under conditions that make validation especially dificult. First, unlike in the previous case studies, no authorship labels are available to validate any clusters. Second, the texts are orally delivered and collaboratively produced, two features shared with the Trump corpus, where no speechwriter clusters emerged. Third, Hungarian has roughly 13 million native speakers, far fewer than English. It belongs to the Finno-Ugric branch of the Uralic family and has highly agglutinative morphology that difers substantially from the grammatical structure of Indo-European languages.

The Hungarian government website (kormany.hu) posts the most recent speeches of the Prime Minister, while older ones can be retrieved by parsing the Wayback Machine, which caches the most important web pages. We scraped a corpus of approximately 1,200 texts attributed to Orbán (2016–2022, only about half of them unique). The corpus also included interviews, social-media posts, and press conferences, which could form genre-based clusters unrelated to authorship. We filtered the delivered speeches in three steps. First, we used a large language model to classify texts as speeches, interviews, or press conferences and removed roughly one-fifth of the corpus (the prompt and specifications appear in Appendix A). Second, we applied dictionary-based filtering to eliminate chunks of English speech, social-media post announcements, and post-speech Q&A sessions. This yielded 466 speeches, which we finally spot-checked manually.

## 8.2 Results

To explore stylistic structure, we project the high-dimensional 3-gram matrix with UMAP (cosine distance, 15 nearest neighbours, minimum distance 0.05). Because no authorship labels exist to validate against, we instead probe the layout with the full discriminant-validity battery, re-colouring the identical projection by six non-authorship factors that could plausibly organise it: dictionary-based afect (tokens joined to the poltextLAB Hungarian political sentiment lexicon (Ring et al., 2024), +1 for positive and −1 for negative matches, summed within document and length-normalised), text length, mean sentence length, mean word length, passive-voice density, and the coordination/subordination ratio. Across the hyperparameter settings examined, the projection contains no stable clusters (Appendix D.7, Figure A10). The six variables reveal no stable divisions; neither do comparisons between early and late speeches or domestic and international audiences. The limited structure in the projection is associated with sentence-level formality rather than authorship. Mean sentence length has the strongest correlation $( | r _ { \mathrm { U 2 } } | = 0 . 4 2 )$ , followed by dictionary sentiment $( | r _ { \mathrm { U 2 } } | = 0 . 3 0 )$ and mean word length $( \left. r _ { \mathrm { U 1 } } \right. = 0 . 2 7 )$ . Text length is nearly unrelated to either axis $( | r _ { \mathrm { U 1 } } | = | r _ { \mathrm { U 2 } } | = 0 . 0 5 )$ indicating that the projection does not simply separate brief ceremonial remarks from long programmatic keynotes. Passive voice and the coordination/subordination ratio are negligible. Absolute correlations are reported because the orientation of each UMAP axis is arbitrary. In short, the corpus contains too little recoverable stylistic structure for inductive stylometry in this setting. The result resembles the Trump case in one respect: no speechwriter clusters emerge within the scripted texts. Unlike the Trump corpus, however, the Orbán corpus ofers no scripted/improvised contrast against which to validate the projection. Three factors may contribute to this result. Agglutinative morphology spreads stylistic signal across an enormous trigram vocabulary, diluting the frequency profile of any individual habit. Oral delivery and transcription overwrite the drafters’ orthographic fingerprints with the speaker’s own cadence. And the institutional editing process of a prime ministerial speechwriting ofice harmonises whatever individual signal survives the first two filters. The case marks the boundary of classical frequency-based stylometry, and we return to what might lie beyond it in the discussion.

## 9 Discussion

This paper demonstrates that a classical stylometric toolbox—character 3-grams and Burrows’ Delta on frequent words, with UMAP for dimensionality reduction—recovers important authorship or production signals in five of six politically diverse corpora. The approach generalizes across text length (from tweets to long-form reports), language (English and Hungarian), and modality (written versus orally delivered), but not unconditionally. Comparing the successful cases with the unsuccessful case identifies the conditions under which the method works.

Four conditions appear to govern when inductive stylometry succeeds. First, morphology affects performance but does not by itself determine success: character trigrams perform best on morphologically simple English, while the ombudsman case shows that agglutinative Hungarian is not a barrier when texts are written, formal, and independently drafted. Second, institutional editing must not erase author-level diferences. Our CRS and ombudsman cases succeed precisely because analysts and rapporteurs draft independently and are not subject to heavy cross-author revision; the speechwriter-level failures in both speech corpora are the mirror image of the same condition. Third, oral delivery places the speaker between the drafters and the transcript: delivery is a form of re-authorship. We can still tell scripted speech from improvised speech, but we can no longer tell the drafters apart. The single complete failure— Orbán’s speeches—combines all three adverse conditions at once: agglutinative language, oral delivery, and heavy institutional editing. Fourth, individual texts must be long enough to provide stable feature coverage. Burrows’ Delta was developed for texts upward of roughly 1,500 words (Burrows, 2002), and attribution reliability degrades sharply in small samples (Eder, 2015); the tweet corpus sits near the lower limit, and chunking into 50-tweet aggregates was required to stabilise the Delta profiles.

A UMAP projection cannot identify authors on its own. Apparent clusters may arise from topical, formatting, or production diferences as well as authorship (Marx, 2024). In the pooled CRS corpus, documents by the same analyst form separate format-based clusters, making the mapping from clusters to authors many-to-one. In the tweet corpus, the Android/iPhone separation reflects production register—link-and-promotion formatting versus composed prose—rather than two distinct authors. Each projection is therefore evaluated alongside Burrows’ Delta on function words, which carry minimal topical signal (Kestemont, 2014; Evert et al., 2017), checks against available labels, and the discriminant-validity tests reported in Appendix D. Interpretation rests on this supporting evidence, not on the projection alone.

Recovering authorship signals in institutional texts has direct consequences for how we evaluate political communication. If tweets, speeches, or policy documents bear the stylometric mark of a specific stafer, speechwriter, or institutional hand, claims about the nominal author’s beliefs, intentions, or rhetoric risk misattribution. In the tweet corpus, the Android/iPhone split captures a real diference in account use but does not map cleanly onto authorship; stylometry helps identify where the proxy breaks down. The framework could also be used to study speechwriter turnover, changes in the authorship of institutional texts, and staf involvement in ostensibly personal political communication.

The Orbán failure also charts the path forward. Where frequency-based features run out of signal, learned authorship representations—neural embeddings trained specifically to encode who writes rather than what is written (Soto et al., 2021)—ofer a language-agnostic alternative that may survive translation, transcription, and editing better than surface trigrams. The six corpora provide benchmarks for testing these representations because the divisions recovered by classical methods are already documented. As large language models enter campaign and government drafting, the question “whose line is it anyway?” becomes harder to answer, increasing the need for validated authorship methods.

Political science usually treats the nominal speaker as the author of a political text. This paper shows that the assumption can be tested. We chose corpora where authorship or production labels was known in advance—analysts, rapporteurs, devices, teleprompters. Inductive stylometry has shown useful to identify the analysts behind institutional prose, separate a politician’s own tweets from his staf’s, and find the moments a speaker leaves the script. Because it is validated where authorship is known, it can be further applied where it is unknown: the unsigned half of the CRS corpus, anonymous oficial reports, ghost-written statements, etc. We also identified possible failure modes. When heavy institutional editing, oral delivery, and complex morphology come together, the individual style becomes more obscure.

## Data Availability Statement

Replication materials—including all corpora and the R analysis code will be deposited in the Political Analysis Dataverse upon acceptance.

## Supplementary Material

The Appendix contains the LLM classification prompt used to filter the Orbán corpus (Appendix A), Burrows’ Delta density plots for the two CRS corpora (Appendix B), the extended speech-by-speech analysis of of-script passages in the Trump teleprompter corpus (Appendix C), and the full discriminant-validity battery for all six use cases (Appendix D).

<table><tr><td>Prompt: You are given a text snippet of a Hungarian speech of Viktor Orbán. Classify it into exactly one of the following categories by returning ONLY the number:</td></tr><tr><td>1 = Interview</td></tr><tr><td>2 = Press conference</td></tr><tr><td>3 = Delivered speech</td></tr><tr><td>No extra words or punctuation, just the digit 1, 2, or 3.</td></tr></table>

## Appendices

## A AI Prompt for Speech Classification (Orbán Corpus)

To filter the Orbán corpus (Use Case 6), each text was classified via the following prompt submitted to the OpenAI ChatGPT o3 model:

Texts that received classification 3 were retained; classifications 1 and 2 were discarded. The prompt was applied to text snippets of approximately 500 characters rather than full transcripts, both to reduce API cost and because brief excerpts are often suficient to distinguish the conversational register of an interview from the monologic register of a delivered speech. The remaining corpus was further cleaned with dictionary-based filters (removal of Englishlanguage chunks, social media post headers, and post-speech Q&A passages appended to some transcripts). This two-stage filtering yielded 466 unique, cleanly delimited speeches used in the main analysis.

To assess classification accuracy, a human reviewer manually checked a random sample of 50 classified texts; the human and LLM labels agreed in all 50 cases for the dominant speechversus-interview distinction. This result describes the reviewed sample and does not establish the corpus-wide misclassification rate. The classification was used as a practical datapreparation procedure rather than as a rigorously validated annotation protocol, and systematic errors may remain in edge cases (e.g., transcripts that begin mid-sentence, or speeches with unusually long Q&A preambles).

## B Burrows’ Delta for the CRS Corpora (Use Cases 1–2)

The figures below report the Burrows’ Delta evidence referenced in the main text for the two CRS corpora. They show, by analyst, the distribution of each document’s Delta distance to its own author’s frequent-word style profile. In both the format-controlled short-report series and the pooled American Law subset, within-analyst profiles are markedly tighter than cross-analyst profiles. This result is consistent with the trigram–UMAP projections and uses frequent function words, which carry little topical signal.

![](images/b5c0955f519afce41a46c2e78a36b0c2301c0ffe033e45ebfc20c5b1a6ec6b2c.jpg)  
Figure A1: Burrows’ Delta distance to each analyst’s own style profile, CRS American Law short-report (RS) series (Use Case 1). Lower values indicate greater within-analyst stylistic consistency.

![](images/9d8300715a02da5d6afed83be2739483d8a91158273dc1b838b636fd7a4ed570.jpg)  
Figure A2: Burrows’ Delta distance to each analyst’s own style profile, full CRS American Law subset (Use Case 2). Lower values indicate greater within-analyst stylistic consistency.

## C Use Case 5: Trump Speeches — Extended Analysis of UMAP Structure

The character 3-gram UMAP shows a strong overall separation between speeches delivered with and without a teleprompter. Most non-teleprompter speeches occupy the upper portion of the figure, while most teleprompter speeches form a separate group; several clearly scripted appearances lie at the lower-left extreme. This distribution associates teleprompter use with a diferent textual and stylistic profile.

The coding records the physical presence of a teleprompter; it does not show that Trump read continuously from prepared text. In several coded speeches, Trump appears to depart briefly from the script, making them more similar to impromptu speeches.

Teleprompters were present at the November 8, 2016 Final Election Eve Rally in Grand Rapids, Michigan, but the speech appears to have been delivered in an almost entirely improvised manner. The presence of the equipment therefore does not by itself indicate a fully scripted delivery style.

A second case is the October 13, 2016 West Palm Beach, Florida address, “Crossroads of Our Nation,” a teleprompter-coded speech in which the speaker repeatedly deviates from the prompt for short intervals. Of-script passages appear around 7:00, 8:00, and 8:30, followed by more extended improvised stretches beginning around 15:00, including a segment from roughly 17:53 to 19:07. Additional shorter departures occur around 19:26–19:40, 19:58, 20:09, 20:20, 20:48, 21:46, 23:00–23:15, 26:40, 31:20–31:43, 32:15–32:55, 37:47–37:55, 38:00–39:15, and 45:00. Many of these moments appear to involve jokes, ad libs, or spontaneous elaborations. Some speeches coded as using a teleprompter therefore contain a nontrivial share of improvised speaking time.

Other teleprompter speeches adhere more closely to the prepared script. The June 7, 2016 Primary Victory Speech in Briarclif Manor, New York appears to contain only occasional short departures, typically brief phrases or minor insertions. The August 20, 2016 rally in Fredericksburg, Virginia does not appear to contain meaningful of-prompt stretches and occupies the most distant part of the teleprompter cluster in the embedding.

The teleprompter category includes speeches that are strongly scripted throughout, speeches with short impromptu segments, and a few that may be largely improvised despite the presence of teleprompter hardware.

## D Discriminant Validity: UMAP Colored by Non-Authorship Factors

## D.1 Rationale and Method

One alternative explanation for the observed UMAP clustering is that surface-level textual properties, including length, sentiment, and syntactic register, shape the projection. If so, the clusters need not reflect authorship-level stylometric signals. We examine this possibility by re-coloring the same character 3-gram UMAP projections used in the main text according to six factors:

1. Sentiment score — dictionary-based positive/negative balance (bing for English; poltextLAB for Hungarian).

2. Text length — total character count per document.

3. Mean sentence length — average number of words per sentence.

4. Mean word length — average number of characters per word.

5. Passive-voice density — passive constructions per word, identified by a rule-based tagger.

6. Coordination/subordination ratio — count of coordinating conjunctions divided by count of subordinating conjunctions.

Each figure below reproduces the UMAP layout from the main text in a 3 × 2 panel grid, replacing the authorship colour label with a continuous colour gradient for one factor per panel. Pearson r between each factor and each UMAP dimension is printed as a subtitle in every panel. Low correlations and the absence of a systematic gradient matching the authorship clusters support a stylometric authorship interpretation and make these potential confounds less plausible explanations for the observed separation. Where correlations are elevated, we discuss the substantive interpretation below. The subsections follow the order of the use cases in the main text.

## D.2 Use Case 1: CRS Reports (American Law, Short-Report Series)

Interpretation. The short-report series is the most tightly controlled corpus in the study: topic area, institutional audience, formal genre, and document format are all held constant by construction. Because corpus-level genre and format are fixed, an association between a surface factor and the projection axes would reflect systematic diferences among analysts within this corpus.

Observed pattern. Text length is the most plausible remaining confound, but its correlations are weak $( r _ { \mathrm { U 1 } } = 0 . 2 9 , r _ { \mathrm { U 2 } } = - 0 . 0 3 )$ . Mean sentence length $( r _ { \mathrm { U 1 } } = 0 . 2 4 , r _ { \mathrm { U 2 } } = - 0 . 1 9 )$ and passivevoice density $( r _ { \mathrm { U 1 } } = - 0 . 1 7 , r _ { \mathrm { U 2 } } = - 0 . 0 2 )$ are also weak. Three factors reach more substantial magnitudes: the coordination/subordination ratio $( r _ { \mathrm { U 1 } } ~ = ~ - 0 . 6 3 , ~ r _ { \mathrm { U 2 } } = 0 . 5 0 )$ , sentiment $( r _ { \mathrm { U 2 } } = - 0 . 5 9 )$ , and mean word length $( r _ { \mathrm { U 2 } } = 0 . 4 8 )$ . The gradients align with the analyst clusters. The clearest example is the Keith Bea cluster at the top of the projection, which has both a high coordination/subordination ratio and low dictionary sentiment. Because every document shares format, topic, audience, and register, these associations are consistent with analyst-level diferences in syntactic architecture (a preference for coordinate over subordinate constructions) and in vocabulary that the sentiment dictionary scores as afective. These may be author-consistent habits encoded by the character trigrams rather than corpus-level confounds. Dictionary sentiment scores over specialised legal vocabulary should in any case be read cautiously, given the sparse coverage of the bing lexicon in this register.

## D.3 Use Case 2: CRS Reports (Full American Law Subset)

Interpretation. Every document in the CRS American Law corpus shares the same topic area, institutional audience, and formal genre. This limits the scope for corpus-level genre diferences to explain systematic diferences among analyst clusters; remaining associations may instead reflect author-level variation.

![](images/9ab43d43570b8c7447fd9ca375993b1781cd6796bafbdca966ae6a97c1dc2697.jpg)  
Figure A3: Discriminant validity for the CRS American Law short-report (RS) series (Use Case 1). The same character 3-gram UMAP as in the main text, re-coloured by six alternative factors. Pearson r with each UMAP dimension is printed as a panel subtitle.

CRS mandates balanced, non-advocacy prose, and legal policy analysis generally uses a neutral, non-partisan register. Sentiment should therefore be near-zero; a strong correlation would instead suggest analyst-level diferences in subtle afective framing despite these institutional constraints.

Text length is the most plausible non-trivial confound. Analysts who primarily handle brief overview reports will have shorter documents than those who specialise in comprehensive legislative histories. A strong association between length and UMAP position would indicate that the authorship clusters partly capture specialisation and report scope. Burrows’ Delta would nevertheless remain interpretable as authorship evidence because it normalises frequencies rather than raw counts.

Mean sentence length and mean word length control for syntactic complexity. Although legal writing generally employs complex sentences, analysts may difer in preferred sentence architecture, for example by preferring semicolons to full stops or enumerated lists to long paragraphs. Author-consistent patterns in these measures would support a stylometric interpretation.

In legal writing, passive-voice density may reflect the use of passive constructions to hedge or distance the author from normative claims. Analyst-level variation could capture individual style, diferential use of legal conventions, or both. An association with the analyst clusters would therefore help specify what the stylometric features capture.

![](images/999fca6433d542c1de81409ed1f654cf0f80d079cc054629a13114b99d60c521.jpg)  
Figure A4: Discriminant validity for CRS Reports, full American Law subset (Use Case 2). Character 3-gram UMAP re-coloured by six alternative factors. One extreme point is omitted from this display to prevent compression of the main cluster; this display choice is separate from the outlier filter described in the main text. Pearson r with each UMAP dimension is printed as a panel subtitle.

The coordination/subordination ratio captures another syntactic preference: coordinate structures $\mathrm { ^ { ( 6 6 ) } X }$ and Y and $Z ^ { \ ' } )$ versus embedded subordinate clauses. Character trigrams can encode this distinction, so moderate correlations aligned with the analyst clusters would support the stylometric interpretation.

Observed pattern. Text length $( | r | \leq 0 . 1 1 )$ , mean sentence length $( | r | \leq 0 . 1 1 )$ , and mean word length $( r _ { \mathrm { U 1 } } = - 0 . 2 0 )$ have weak associations with the projection, so these measures account for little of the analyst-level cluster structure. The largest correlations involve the coordination/subordination ratio $( r _ { \mathrm { U 1 } } = - 0 . 5 0 )$ , dictionary sentiment $( r _ { \mathrm { U 2 } } = - 0 . 4 0 )$ , and passive-voice density $( r _ { \mathrm { U 2 } } = - 0 . 3 4 )$ . Because every document shares topic, audience, and institutional writing conventions, these patterns are consistent with analyst-level diferences in syntactic architecture (coordinate versus subordinate constructions) and afective framing. Passive-voice density may reflect diferential use of the legal convention of hedging as well as individual style. The sentiment values should be read with some caution, as the bing dictionary’s coverage of specialized legal vocabulary is limited. In this controlled corpus, the weak topic- and length-related associations provide discriminant validity for the authorship interpretation, although they do not establish individual analyst style as the only possible

explanation.

Topic versus authorship. Analyst specialisation within American Law remains a possible confound because analysts have distinct beats. We test this by re-colouring the identical character-3-gram UMAP by the oficial CRS topic tags (Figure A5). Each report’s multi-label tag set is collapsed to its first tag other than American Law; reports carrying only the division tag form their own category, and tags with fewer than ten reports are grouped as Other. The topic partition is heavily interspersed (silhouette −0.31), compared with +0.41 for author labels in the same coordinates. Subject matter and authorship are moderately associated, as expected when analysts specialise (normalised mutual information 0.40), but the topic labels do not reproduce the spatial geometry. If topic drove the visible clusters, the topic partition would also form spatially coherent groups. These results are more consistent with authorship than subject matter as the main source of the observed structure.

Author portability across topics. The partition test fixes the projection and varies the labels. A complementary analysis fixes the author and varies the topic. Across the full CRS corpus beyond American Law, 36 analysts each wrote at least six single-authored reports in each of two or more distinct primary topic areas, yielding 942 documents. We compute pairwise cosine similarities between the documents’ character-3-gram profiles, using the same tokenization and trimming as the main pipeline, and average them by pair type (Table A1). Reports are more similar to reports by the same author on diferent topics (0.83) than to reports by other analysts on the same topic (0.72). The same-author similarity is nearly unchanged across topics (0.83, compared with 0.82 within topic), while the topic efect over the unrelated-pair baseline is +0.02, compared with an author efect of +0.13. A nearest-neighbour analysis yields a similar pattern. When each report’s nearest neighbour is restricted to documents outside its own author–topic cell, the neighbour shares the author in 59 percent of cases and the topic in 16 percent. These results indicate that similarities associated with analyst identity persist across topic areas.

Table A1: Author portability across topics: mean pairwise cosine similarity of character-3-gram profiles, CRS analysts writing in multiple topic areas (n = 942 reports, 36 analysts, 18 topics)
<table><tr><td>Pair type</td><td>Mean cosine</td><td>Pairs</td></tr><tr><td>Same author, different topic</td><td>0.83</td><td>8,526</td></tr><tr><td>Same author, same topic</td><td>0.82</td><td>9,193</td></tr><tr><td>Same topic, different author</td><td>0.72</td><td>50,336</td></tr><tr><td>Different author, different topic</td><td>0.70</td><td>375,156</td></tr></table>

Analysts qualify with at least six single-authored reports in each of two or more primary EveryCRSReport topic areas.

## D.4 Use Case 3: Hungarian Ombudsman Reports

Interpretation. Unlike the Orbán speeches, each Ombudsman report carries a named rapporteur label that provides a ground-truth authorship proxy for validation. The discriminant validity analysis assesses whether the rapporteur-level UMAP structure reflects individual style or report-level confounds.

Sentiment should be minimally informative because Ombudsman reports are legal-administrative texts written in a highly standardised, neutral register, with little afective vocabulary and limited variation across rapporteurs. Re-colouring the identical projection by dictionary sentiment, computed with the Hungarian-language poltextLAB political sentiment lexicon (Ring et al., 2024), shows no visible afective gradient organising the layout (Figure A7). The low sentiment–UMAP association supports the authorship interpretation and makes dictionary sentiment an unlikely explanation for the structure.

![](images/87b16eb755c02ab67f8278adb2609429b5f44fd5138cedf63e83c3af79d79844.jpg)  
Figure A5: Topic-versus-authorship control for the full CRS American Law subset. The identical character-3-gram UMAP of Figure A4, re-coloured by oficial CRS topic tags. The topic partition is heavily interspersed (silhouette −0.31), against +0.41 for author labels in the same coordinates.

Text length may correlate moderately if diferent rapporteurs handle complaints of diferent complexity or scope. All reports share the same formal structure, which limits corpus-level genre variation. A strong association would nevertheless suggest that UMAP is primarily recovering report complexity rather than authorship.

Mean sentence length controls for diferences in sentence architecture. Legal prose often uses long, subordinate-clause-heavy sentences; concentration of such sentences within one rapporteur’s documents would support the authorship interpretation, whereas a pattern crossing rapporteur boundaries would suggest a non-authorship confound.

In Hungarian, mean word length reflects morphology as well as style. Case types that require longer compound terms or more sufixes could make word length vary by rapporteur, producing a topic-related rather than author-related signal.

![](images/81aa2fb48fa8ec95a882c786482939945fe197b64fbc20f58cfb2f92aff58b63.jpg)  
Figure A6: Discriminant validity for Hungarian Ombudsman Reports (Use Case 3). Character 3-gram UMAP re-coloured by six alternative factors. Caution: the syntactic factors use English-language heuristics applied to Hungarian text; quantitative values should be interpreted with care. Pearson r with each UMAP dimension is printed as a panel subtitle.

Because these syntactic measures rely on English-language heuristics, passive-voice density and the coordination/subordination ratio should be treated as rough proxies.

Observed pattern. All six factors correlate weakly with both UMAP dimensions. The largest correlations are for mean sentence length $( | r | \leq 0 . 2 4 )$ and dictionary sentiment $( r _ { \mathrm { U 1 } } = 0 . 1 9 $ computed here with the Hungarian poltextLAB lexicon). The correlations for text length $( | r | \leq 0 . 1 4 )$ , mean word length $( | r | \le 0 . 0 6 )$ , passive-voice density $( | r | \leq 0 . 0 1 )$ , and the coordination/subordination ratio $( | r | \leq 0 . 0 2 )$ are near zero. In this corpus of legal-administrative Hungarian reports, none of the six surface-level factors explains much of the rapporteur-level cluster structure visible in the main-text UMAP. This makes text length, syntactic complexity, and writing register unlikely to drive the observed separation and is consistent with rapporteurlevel stylometric diferences.

## D.5 Use Case 4: Trump Tweets (Android vs. iPhone)

Interpretation. In this case, surface-form variables are expected to align with the device separation. The Android/iPhone contrast in the main text is organised by production mode: the iPhone stream contains links, hashtags, and promotional posts, whereas the Android stream contains composed prose. Because production register co-varies with length and surface form, correlations with length-related variables would support a register interpretation rather than a clean authorship split.

![](images/98cf20ee462c50c8fc6c2d8618113f4f81265f85f0bed40927275807dd49009c.jpg)  
Figure A7: Hungarian Ombudsman reports: the character 3-gram UMAP of the main text, re-coloured by length-normalised poltextLAB sentiment. No visible afective gradient organises the layout, making sentiment an unlikely explanation for the rapporteur-level structure.

Observed pattern. Mean word length $( r _ { \mathrm { U 1 } } = 0 . 5 2 )$ , mean sentence length $( r _ { \mathrm { U 1 } } = - 0 . 4 8 )$ and text length $( r _ { \mathrm { U 1 } } = - 0 . 2 3$ 7 $r _ { \mathrm { U 2 } } = - 0 . 3 5 )$ correlate moderately with the device-separating axes. Dictionary sentiment $( r _ { \mathrm { U 1 } } = - 0 . 0 2 , r _ { \mathrm { U 2 } } = - 0 . 0 2 )$ , passive-voice density $( r _ { \mathrm { U 1 } } = - 0 . 0 6 $ $r _ { \mathrm { U 2 } } = - 0 . 0 9 )$ , and the coordination/subordination ratio $( r _ { \mathrm { U 1 } } ~ = ~ - 0 . 0 3 , ~ r _ { \mathrm { U 2 } } = 0 . 1 0 )$ are efectively zero. This difers from the formal-prose corpora, where length-related factors were null and the recovered structure was authorial. Here those factors track the projection because the device label represents production register. The formatting-stripping analysis provides a direct test: removing URL characters reduces the mean device silhouette from 0.21 to 0.09, and removing hashtag characters as well reduces it to 0.08. This indicates that formatting accounts for most of the device separation.

Burrows’ Delta, leave-one-out. Each device profile is the mean frequent-word vector for that device, so every tweet contributes marginally to its own reference profile (self-weight 1/1,866 for Android, 1/2,134 for iPhone). We recomputed all scores against leave-one-out profiles, comparing each tweet only with profiles built from the other tweets. This recomputation changes the mean device scores by under half a percent and leaves the device ranking and separation reported in the main-text table unchanged. The frequent-word contrast is therefore not an artefact of self-inclusion, but it still concerns a label that the stripping ladder identifies with production register rather than authorial hand.

![](images/0c7809d8fad92b0454891bf852475ce9cb50cc00895f0c73e076ba519d1b2dd8.jpg)  
Figure A8: Discriminant validity for Trump Tweets (Use Case 4). The same character 3-gram UMAP as in the main text, re-coloured by six alternative factors. Pearson r with each UMAP dimension is printed as a panel subtitle.

## D.6 Use Case 5: Trump Speeches (Teleprompter vs. Of-the-Cuf)

Interpretation. The speech corpus difers from the tweet corpus. Its primary contrast is between two delivery modes by the same speaker: scripted teleprompter delivery and improvised ad-lib delivery, rather than between two individuals.

Sentiment difers significantly in the main text, with impromptu speeches scoring higher, so moderate correlations are expected. We assess whether this diference fully explains the UMAP separation. Burrows’ Delta on function words, which carry little afective content, also separates the two groups and therefore weighs against a purely sentiment-driven account.

Text length is unlikely to drive the separation. The main text reports no statistically significant length diference between teleprompter and of-the-cuf speeches, and duration depends more on event format than delivery style.

Mean sentence length captures a plausible fluency and complexity diference. Scripted text tends to use longer, more syntactically complex sentences, while spontaneous speech contains shorter bursts and false starts. A strong correlation would mean that sentence structure accounts for much of the UMAP separation; a moderate one would indicate a contribution

![](images/e3f4ecf117e7f045abfe47596e00e36a015e659bcda60037d5f0563ad274c4c3.jpg)  
Figure A9: Discriminant validity for Trump Speeches (Use Case 5). Character 3-gram UMAP re-coloured by six alternative factors. Pearson r with each UMAP dimension is printed as a panel subtitle.

## without dominance.

Mean word length may correlate negatively with the impromptu cluster because unscripted remarks use more colloquial vocabulary, including shorter and more common words. Such a pattern would show that word choice varies with delivery mode.

Passive-voice density should be higher in scripted policy language than in ad-lib remarks. A strong alignment with the projection would indicate that formal and informal register account for much of the separation.

The coordination/subordination ratio captures a related syntactic diference. Scripted text tends to use subordinating structures for precision (conditional clauses, relative clauses, embedded complements), whereas spontaneous speech favours simpler coordination (“and,” “but,” “or”). Alignment with the teleprompter/ad-lib split would support a register interpretation; weak correlations would indicate that the delivery-mode diference extends beyond these surface measures.

Observed pattern. The speech corpus has the highest correlations in the study. Sentiment $( r _ { \mathrm { U 1 } } = 0 . 5 5 , r _ { \mathrm { U 2 } } = - 0 . 8 7 )$ , mean sentence length $( r _ { \mathrm { U 1 } } = - 0 . 4 0 , r _ { \mathrm { U 2 } } = 0 . 8 3 )$ , and mean word length $( r _ { \mathrm { U 1 } } = - 0 . 4 6 , r _ { \mathrm { U 2 } } = 0 . 8 2 )$ align strongly with the UMAP axes. Passive-voice density $( r _ { \mathrm { U 1 } } = - 0 . 5 6 , r _ { \mathrm { U 2 } } = 0 . 6 3 )$ has moderate correlations, and the coordination/subordination correlations are at most 0.52 in absolute value. Text length remains low $( r _ { \mathrm { U 1 } } = - 0 . 3 2 , r _ { \mathrm { U 2 } } = 0 . 1 0 )$ consistent with the non-significant length diference reported in the main text. These results are expected because the scripted/impromptu distinction partly concerns register: scripted text has longer sentences, more formal vocabulary, and denser passive constructions. However, Burrows’ Delta on function words, which carry minimal afective or syntactic-complexity signal, produces near-perfect separation between the two groups (see the corresponding figure in the main text). Function-word diferences therefore remain alongside the measured register markers. The discriminant validity figures show that surface register explains part, but not all, of the structure.

## D.7 Use Case 6: Orbán Speeches in Hungarian

Because the Orbán corpus has no authorship labels, the discriminant-validity battery provides the primary diagnostic (Figure A10). The figure shows the character 3-gram UMAP re-coloured by six non-authorship factors; panel subtitles report Pearson r for each UMAP dimension. This section explains the factors in detail. Caution: the syntactic factors are computed using heuristics designed for English; their values for Hungarian text should be interpreted with caution.

![](images/3dab92dbf6abbd10941e229e2a4e417a9e8b450b0880da1a89ea0b3d58ecc162.jpg)  
Figure A10: Discriminant-validity battery for 466 delivered speeches by Viktor Orbán: the character 3-gram UMAP re-coloured by six non-authorship factors, with the Pearson r against each UMAP dimension printed as a panel subtitle. The layout has no stable clusters across the hyperparameter settings examined. Mean sentence length has the strongest association $( | r _ { \mathrm { U 2 } } | = 0 . 4 2 )$ , while sentiment and mean word length correlate more weakly. Sentiment is the length-normalised poltextLAB Hungarian score; the syntactic factors use English-language heuristics and are unreliable for Hungarian.

Interpretation. In the Orbán corpus, the character 3-gram UMAP in the main text is amorphous and non-clustered. We interpret this as a failure of frequency-based stylometry to recover author-level structure in this setting.

Sentiment may distinguish emotionally charged domestic or programmatic speeches from brief international statements. An association with a UMAP axis would indicate that afect explains part of the variation.

Text length is a plausible organising factor because the corpus mixes short ceremonial remarks with long ideological keynotes (e.g., the annual évértékelő addresses and the Tusnádfürdő lectures). This range could afect a projection that is sensitive to vocabulary richness, including one based on character trigrams.

Mean sentence length captures diferences in formality: programmatic speeches have longer, more complex sentences, while ceremonial and bilateral statements use shorter, more formulaic ones.

Hungarian’s agglutinative morphology makes mean word length relevant. Compound words and case sufixes produce longer tokens than their English equivalents, and variation across speech types may reflect topic or genre rather than authorial preference.

Passive-voice density and coordination/subordination ratio are both computed using Englishlanguage heuristics and are unreliable for Hungarian. Any correlations should be treated as artefacts of tokenization and pattern-matching on a morphologically rich language, not as substantive stylometric signals.

Observed pattern. Mean sentence length $( r _ { \mathrm { U 2 } } = - 0 . 4 2 )$ has the strongest correlation, followed by dictionary sentiment $( r _ { \mathrm { U 2 } } = - 0 . 3 0$ , length-normalised poltextLAB) and mean word length $( r _ { \mathrm { U 1 } } = - 0 . 2 7 )$ . These moderate correlations suggest that speech formality and genre (keynote vs. ceremonial address), rather than authorship, weakly organise the projection. Text length $( r _ { \mathrm { U 1 } } = 0 . 0 5 , r _ { \mathrm { U 2 } } = - 0 . 0 5 )$ is near zero despite the wide length range of the corpus, suggesting that UMAP is not simply sorting speeches by duration. Passive voice and the coordination/subordination ratio are also near zero, as expected because the English-language heuristics used to compute them have limited validity for Hungarian morphology. No single factor dominates the projection, but register and sentence-level formality weakly shape it. This is consistent with the main text’s conclusion that frequency-based stylometry fails to recover stable authorship structure in this setting.

## References

Airoldi, Edoardo M., Stephen E. Fienberg and Kiron K. Skinner. 2007. “Whose Ideas? Whose Words? Authorship of Ronald Reagan’s Radio Addresses.” PS: Political Science & Politics 40(3):501–506.

Argamon, Shlomo. 2008. “Interpreting Burrows’s Delta: Geometric and Probabilistic Foundations.” Literary and Linguistic Computing 23(2):131–147.

Brown, Brendan. 2019. “Trump Twitter Archive.” Online database. URL: http://www.trumptwitterarchive.com/

Burrows, John. 2002. ““Delta”: A Measure of Stylistic Diference and a Guide to Likely Authorship.” Literary and Linguistic Computing 17(3):267–287.

Clarke, Isobelle and Jack Grieve. 2019. “Stylistic Variation on the Donald Trump Twitter Account: A Linguistic Analysis of Tweets Posted between 2009 and 2018.” PLOS ONE 14(9):e0222062.

Eder, Maciej. 2015. “Does Size Matter? Authorship Attribution, Small Samples, Big Problem.” Digital Scholarship in the Humanities 30(2):167–182.

Evert, Stefan, Thomas Proisl, Fotis Jannidis, Isabella Reger, Stefen Pielström, Christof Schöch and Thorsten Vitt. 2017. “Understanding and Explaining Delta Measures for Authorship Attribution.” Digital Scholarship in the Humanities 32(suppl\_2):ii4–ii16.

Grieve, Jack. 2007. “Quantitative Authorship Attribution: An Evaluation of Techniques.” Literary and Linguistic Computing 22(3):251–270.

Grimmer, Justin and Brandon M. Stewart. 2013. “Text as data: The promise and pitfalls of automatic content analysis methods for political texts.” Political Analysis 21(3):267–297.

Hoover, David L. 2004. “Testing Burrows’s Delta.” Literary and Linguistic Computing 19(4):453–475.

Juola, Patrick. 2015. “The Rowling Case: A Proposed Standard Analytic Protocol for Authorship Attribution.” Digital Scholarship in the Humanities 30(suppl\_1):i100–i113.

Kestemont, Mike. 2014. Function Words in Authorship Attribution: From Black Magic to Theory? In Proceedings of the 3rd Workshop on Computational Linguistics for Literature. Association for Computational Linguistics pp. 59–66.

Koppel, Moshe, Jonathan Schler and Shlomo Argamon. 2009. “Computational Methods in Authorship Attribution.” Journal of the American Society for Information Science and Technology 60(1):9–26.

Marx, Vivien. 2024. “Seeing data as t-SNE and UMAP do.” Nature Methods 21(6):930–933.

McInnes, Leland, John Healy and James Melville. 2018. “UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction.” arXiv:1802.03426. URL: https://arxiv.org/abs/1802.03426

Mosteller, Frederick and David L. Wallace. 1964. Inference and Disputed Authorship: The Federalist. Reading, MA, USA: Addison-Wesley.

Ring, Orsolya, Martina Katalin Szabó, Csenge Guba, Bendegúz Váradi and István Üveges. 2024. “Approaches to sentiment analysis of Hungarian political news at the sentence level.” Language Resources and Evaluation . URL: https://link.springer.com/article/10.1007/s10579-023-09717-5

Robinson, David. 2016. “Text analysis of Trump’s tweets confirms he writes only the (angrier) Android half.” Blog post, Variance Explained. URL: http://varianceexplained.org/r/trump-tweets/

Rousseeuw, Peter J. 1987. “Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis.” Journal of Computational and Applied Mathematics 20:53–65.

Smith, David, Paul Lewis, Josh Holder and Frank Hulley-Jones. 2019. “The Teleprompter Test: Why Trump’s Populism Is Not His Own.” The Guardian interactive. Published 6 March 2019; accessed 10 July 2026.

URL: https://www.theguardian.com/world/ng-interactive/2019/mar/07/the-telepromptertest-why-trumps-populism-is-often-scripted

Soto, Rafael A. Rivera, Olivia Miano, Juanita Ordonez, Barry Chen, Aleem Khan, Marcus Bishop and Nicholas Andrews. 2021. Learning Universal Authorship Representations. In EMNLP.

Stamatatos, Efstathios. 2009. “A survey of modern authorship attribution methods.” Journal of the American Society for Information Science and Technology 60(3):538–556.

van der Maaten, Laurens and Geofrey Hinton. 2008. “Visualizing Data using t-SNE.” Journal of Machine Learning Research 9:2579–2605. URL: https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

Wilkerson, John and Andreu Casas. 2017. “Large-scale computerized text analysis in political science: Opportunities and challenges.” Annual Review of Political Science 20(1):529–544.