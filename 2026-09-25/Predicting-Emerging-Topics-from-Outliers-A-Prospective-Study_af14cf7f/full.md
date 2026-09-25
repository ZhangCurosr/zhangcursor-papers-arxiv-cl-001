# Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space

Evangelia Zve<sup>1,2</sup>\*, Gauvain Bourgne<sup>1</sup>, Jean-Gabriel Ganascia<sup>1</sup> <sup>1</sup>LIP6, Sorbonne Université, CNRS, {name.surname}@lip6.fr <sup>2</sup>Infopro Digital

## Abstract

Some documents that embedding-based topic models initially classify as noise later become founding members of emerging topics. At publication time, however, they appear as scattered points in embedding space and are difficult to distinguish from ordinary noise without the benefit of hindsight. We study whether such anticipatory outliers can be predicted prospectively, using only information available when a document first appears. We derive labels from the subsequent trajectories of outlier documents, distinguishing those that anticipate new topics from those that reinforce existing topics or remain isolated, and estimate label confidence through agreement across multiple embedding models. On two French news corpora, anticipatory outliers prove predictable at publication time. Under cross-validation, F rises from about 0.77 over the full eligible population to above 0.90 on high-consensus subsets, and remains at 0.76–0.80 under a strictly chronological evaluation. Predictive performance is driven mainly by geometric features capturing each outlier’s position in embedding space.

## 1 Introduction

In document streams such as news or scientific literature, the challenge is not only to summarize topics that are already visible, but to identify topics as they begin to emerge (Allan, 2002; Boutaleb et al., 2024; Ebadi et al., 2026). Despite recent advances in language models and representation learning, early topic detection remains difficult for embedding-based topic models.

Methods such as BERTopic (Grootendorst, 2022) place documents in a semantic space and cluster them, often using density-based algorithms such as HDBSCAN (McInnes et al., 2017). Documents outside dense regions are usually treated as noise and excluded. Yet these outliers do not all share the same fate (Zve et al., 2025). Some remain isolated, some later join topics that already existed, and some precede topics that had not yet formed.

![](images/95188e5b4aa030fefd72286d7541131f82ebf9283ffbe7e20ee0f403776c5478.jpg)  
Figure 1: Prospective prediction under retrospective supervision. Labels are reconstructed from later trajectories. Predictors are computed at publication time T<sub>A</sub>. Article i illustrates an anticipatory outlier.

Building on the temporal taxonomy of document trajectories introduced by Zve et al. (2026), we focus on documents that appear before the topics they later join. Retrospectively, this anticipatory status is observable once the future topic has formed. Prospectively, however, the same document is still only an outlier in the current embedding space. This gap raises a concrete prediction problem. Can publication-time evidence distinguish ordinary outliers from documents that will later become part of a topic that has not yet formed?

To address this, we reconstruct article trajectories from cumulative daily snapshots of a news stream. On each day, articles observed so far are clustered in embedding space, and the resulting clusters are aligned across consecutive snapshots, to track continuing topics and detect newly formed ones. This reconstruction allows us to determine whether publication-time outliers remain isolated, later join existing topics, or anticipate topics that form only after the article appears.

We formulate anticipatory outlier detection as a supervised binary classification task in which retrospective topic trajectories provide labels, while predictors are restricted to information available when each document first appears. Retrospective topic formation provides the supervision needed to train a prospective model.

We make three contributions. First, we frame anticipatory outlier detection as a prospective supervised article-level prediction task, with labels reconstructed from retrospective topic trajectories across multiple embedding models and confidence estimated through inter-model agreement. Second, we propose and discuss three complementary families of publication-time predictors: geometric position, textual content, and early social circulation of outlier articles. Third, we show that anticipatory outliers are predictable at publication time, both under cross-validation and under a strictly chronological evaluation, and identify the features that drive these predictions.

## 2 Related Work

Weak-signal analysis studies sparse, ambiguous observations whose significance becomes clear only in retrospect (Hiltunen, 2008; Ansoff, 1975). Related NLP work on Topic Detection and Tracking (Allan, 2002), first-story detection (Petrovic´ et al., 2010), burst detection (Kleinberg, 2002), and social-stream event detection (Becker et al., 2011) seeks early evidence of emerging developments, typically at the event, story, burst, or topic level. In contrast, work on topic-model outliers treats initially unclustered articles as the unit of analysis and asks whether they become early members of emerging topics (Zve et al., 2025, 2026).

Our task is also related to novelty and outlier detection, which identify deviations from regularities (Pimentel et al., 2014; Aggarwal, 2017). Here, outlier status is only the starting point. Rather than detecting outliers per se, we ask which publicationtime outliers later become precursors of new topics.

Our work also relates to temporal and embedding-based topic modeling. Classical dynamic topic models track latent topics from word co-occurrence patterns over time (Blei and Lafferty, 2006; Wang and McCallum, 2006). Neural and embedding-based topic models instead represent documents in dense semantic spaces, improving topic discovery through contextual representations (Dieng et al., 2020; Grootendorst, 2022). Recent emerging-topic systems track embedding-space topical structure by aligning clusters across time windows and monitoring nascent topics (Christophe et al., 2021; Boutaleb et al., 2024). These methods are well suited to tracking topics once repeated evidence has accumulated (Ebadi et al., 2026). We instead focus on an earlier stage, when articles remain unclustered and their future topical role is unclear, asking whether these trajectories can be predicted at publication time.

## 3 Trajectory-Based Supervision

We derive supervised labels from the later trajectories of outlier articles, following the taxonomy of Zve et al. (2026). On each day, all articles published so far are clustered in embedding space, which assigns each article either to a topic or to the outlier set. Aligning topics across consecutive days then tells us whether each topic is continuing or newly created. An article that was an outlier at publication time can therefore end up in one of three situations: it joins a topic that already existed when it appeared, it remains an outlier, or it joins a topic that only formed after its publication. The label is positive in this last, anticipatory case. We run this procedure separately for each embedding model and combine the results in Section 3.3. Figure 1 illustrates the setup with one anticipatory outlier.

## 3.1 Cumulative Clustering Setting

For each day t, the snapshot contains all articles observed up to and including t. Article texts are embedded, projected with UMAP (McInnes et al., 2018), and clustered using a density-based method. Such methods, including HDBSCAN (McInnes et al., 2017) and OPTICS (Ankerst et al., 1999), can identify low-density observations that do not belong to any cluster. These observations form the candidate outlier cases considered below.

Topic continuity is recovered by aligning clusters in consecutive daily snapshots. Each cluster is represented by the centroid of its article embeddings in the reduced space. The one-to-one matching problem between clusters at t − 1 and t is solved using the Hungarian algorithm (Kuhn, 1955), with cosine distance between centroids as the matching cost. A cluster at time t inherits the identity of a previous topic if its best match lies below the alignment threshold $\theta _ { \mathrm { a l i g n } }$ ; otherwise, it is treated as a new topic. This aligned sequence provides the temporal reference used to assign outlier trajectories.

## 3.2 Trajectories and Prediction Target

Using the notation of Zve et al. (2026), each article trajectory is characterized by three temporal variables: its publication time $T _ { A }$ , the creation time $T _ { T }$ of the topic it eventually joins, if any, and its first integration time $T _ { I }$ into that topic. The relative ordering of these events determines the trajectory category and defines the classification target.

The positive class consists of anticipatory outliers: articles that are unassigned at publication time and later join a topic that did not yet exist when they appeared, i.e., $T _ { A } < T _ { T } \leq T _ { I }$ . This includes $\tau \mathcal { O } A _ { \mathrm { f i r s t } }$ , where integration occurs when the topic is created, $T _ { A } < T _ { T } = T _ { I }$ , and $\tau \mathcal { O A } _ { \mathrm { l a t e } }$ where the topic is created after publication but the article joins it only later, $T _ { A } < T _ { T } < T _ { I }$

The negative class is the complement of this target within the set of articles that are outliers at $T _ { A }$ It includes articles that later join a topic that already existed when they appeared, denoted by $\tau \mathcal { O } \mathcal { D } _ { \mathrm { l a t e } }$ and articles that remain unassigned throughout the observation window, denoted by $\mathcal { O } = \mathcal { O } _ { \mathrm { r e c e n t } } \cup$ $\mathcal { O } _ { \mathrm { o l d } }$ . The supervised task is therefore the binary distinction $\mathcal { T O A } = \mathcal { T O A } _ { \mathrm { f r s t } } \cup \mathcal { T O A } _ { \mathrm { l a t e } }$ versus $\neg \mathcal { T O A } = \mathcal { T O D } _ { \mathrm { l a t e } } \cup \mathcal { O }$ , restricted to articles that are outliers at publication time.

## 3.3 Consensus-Based Filtering and Labeling

A central difficulty is that trajectory assignment is model-dependent. The embedding space, density structure, and cross-time topic alignment all depend on the upstream embedding model. We therefore do not treat any single model-specific trajectory assignment as definitive supervision. Instead, we run the trajectory-labeling pipeline separately for each embedding model and use intermodel agreement in two ways: to filter reliable publication-time outliers and to assign anticipatory versus non-anticipatory labels. This is aligned with previous work showing that agreement across imperfect labeling sources can serve as a proxy for label reliability (Snow et al., 2008; Ratner et al., 2017; Strehl and Ghosh, 2002).

Let $\mathcal { M }$ denote the set of embedding models. For each article $i ,$ let $o _ { i }$ be the number of models that classify the article as an outlier at publication time, and let $a _ { i }$ be the number of models that assign it to an anticipatory trajectory. We define labels using three consensus thresholds $\left( k _ { o } , k _ { a } , k _ { n } \right)$ , an outliereligibility threshold $k _ { o } ,$ a positive-label threshold $k _ { a }$ , and a negative-label threshold $k _ { n }$

$$
y _ { i } = { \left\{ \begin{array} { l l } { 1 , } & { o _ { i } \geq k _ { o } { \mathrm { a n d } } a _ { i } \geq k _ { a } , } \\ { 0 , } & { o _ { i } \geq k _ { o } { \mathrm { a n d } } a _ { i } \leq k _ { n } , } \\ { { \mathrm { u n c e r t a i n } } , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }
$$

![](images/8e51fe8474663f22a74c940b45ec331ab4553d35e7b4b8a269dc502251fe213e.jpg)  
Figure 2: Consensus labeling example. Rows are articles $i = 1 , \ldots , 4 ,$ , columns $m _ { 1 } { - } m _ { 1 1 }$ are the embedding models in ${ \mathcal { M } } .$ and $a _ { i }$ is the number of models assigning the $\mathcal { T O A }$ trajectory to article i (shaded cells). The four articles have already passed the eligibility filter $o _ { i } \geq k _ { o }$ Under $( k , k , 0 )$ , article 2 is positive for $k \leq 9$ , article 4 only for $k = 1$ , and articles 1 and 3 are negative for every k.

The condition $o _ { i } \geq k _ { o }$ defines the eligible population: articles reliably identified as publication-time outliers. Within this population, positives require at least $k _ { a }$ anticipatory votes, while negatives require at most $k _ { n }$ such votes. Articles satisfying neither rule are treated as uncertain and excluded from training and evaluation. Figure 2 illustrates the rule on four articles.

In our experiments, we use the conservative setting $( k _ { o } , k _ { a } , k _ { n } ) = ( k , k , 0 )$ . Thus, both outlier status and positive labels must be supported by at least k models, whereas negatives must receive no anticipatory vote. Varying k quantifies the coverage– confidence tradeoff. Larger values retain fewer articles but provide higher-confidence supervision. More general asymmetric rules are possible, but varying the three thresholds independently introduces additional calibration choices; we leave their systematic investigation to future work.

## 4 What Predicts Anticipatory Outliers?

If anticipatory outliers are meaningful early signals rather than residual noise, they should already leave detectable traces at $T _ { A }$ . We therefore propose and discuss three complementary families of $T _ { A }$ -based predictors, each capturing a different article-level source of evidence about later topic formation.

Geometric features locate the article in the publication-time embedding space, measuring whether it sits at the margins of established topic regions or in sparse residual areas where new structure may later form. Textual features characterize the article’s content, tone, and form, testing whether anticipatory trajectories are associated with distinctive framing, entity anchoring, or stylistic cues. Social features capture early circulation on X, testing whether articles that later join emerging topics already leave observable traces in audience reach or audience-overlap structure. All predictors use only $T _ { A }$ features, while labels are derived from later trajectories to prevent future-topic leakage.<sup>1</sup>

## 4.1 Geometric Features

Geometric features measure where an outlier lies in the UMAP-reduced embedding space of the cumulative topic snapshot at $T _ { A }$ . Let $x _ { i }$ denote the reduced embedding of article i, and let $C _ { T _ { A } }$ denote the set of topic clusters present at $T _ { A }$

We first measure proximity to existing topics. For each cluster $c \in C _ { T _ { A } }$ , we compute its centroid $\mu _ { c }$ and the Euclidean distance from $x _ { i }$ to $\mu _ { c } .$ . The nearest and second-nearest distances, $d _ { 1 i }$ and d<sub>2i</sub>, measure how far the article lies from the topic structure at $T _ { A }$ . Their difference, $\Delta _ { i } = d _ { 2 i } - d _ { 1 i }$ , defines the centroid margin: large values indicate proximity to one dominant topic, while small values indicate comparable proximity to multiple topics. Figure 3(a) illustrates this geometry.

Centroid distances treat all clusters as equally compact, even though some topics are more dispersed than others. We therefore compute a diagonal Mahalanobis distance to each existing topic cluster containing at least five articles, and retain the minimum value

$$
d _ { i } ^ { \mathrm { M a h } } = \operatorname* { m i n } _ { c \in C _ { T _ { A } } } \sum _ { j } \frac { ( x _ { i j } - \mu _ { c j } ) ^ { 2 } } { \sigma _ { c j } ^ { 2 } + \epsilon } ,
$$

where $\sigma _ { c j } ^ { 2 }$ is the coordinate-wise variance of cluster c, and ϵ prevents division by zero. This feature measures whether $x _ { i }$ is atypical relative to the dispersion of the nearest plausible topic, rather than only distant in Euclidean space.

We then measure local density around the article. Using only articles observed in the cumulative snapshot at $T _ { A }$ , we compute the mean Euclidean distance $\bar { d } _ { i } ^ { \mathrm { k N N } }$ from $x _ { i }$ to up to 20 nearest neighbors, depending on the number of articles available in the snapshot; see Figure 3(b). We also compute the standard deviation of these distances to capture neighborhood heterogeneity and distinguish uniformly sparse regions from mixed neighborhoods containing both close and distant neighbors.

A further group of features captures proximity to other outliers. Let $O ( T _ { A } )$ denote the set of articles classified as outliers in the cumulative snapshot at $T _ { A }$ . We record whether this set is non-empty, its size, and the mean Euclidean distance from $x _ { i }$ to up to ten nearest articles in $O ( T _ { A } )$ . These features distinguish outliers that are isolated in the embedding space from outliers that lie close to other publication-time outliers.

![](images/680e2ef6a20282a7cae35bc198cd252c6cfe517b312381f38eef5a9f4d8a19ea.jpg)  
Figure 3: Geometric predictors at publication time $T _ { A } \mathbf { \hat { \cdot } }$ (a) distances from outlier i to the nearest topic centroids and (b) an illustrative local-neighborhood configuration.

Finally, we include the outlierness score at $T _ { A }$ computed from the GLOSH score returned by HDBSCAN (Campello et al., 2015; McInnes et al., 2017). It measures how strongly an article is separated from dense regions in the HDBSCAN hierarchy. Higher values indicate that the article is weakly connected to nearby clusters. Unlike centroid-distance features, it captures local density structure rather than distance to topic centers.

Because each embedding model induces its own UMAP space, raw distance values are not directly comparable across models. We therefore convert each distance-based geometric feature to a within-model percentile rank. For feature $g _ { \colon }$ article i, and embedding model $m .$ , let $g _ { i m } ( T _ { A } )$ be the publication-time value and let $n _ { m } ( T _ { A } )$ be the number of rows in the corresponding model-specific feature table. We compute

$$
p _ { i m } ^ { ( g ) } ( T _ { A } ) = \frac { \mathrm { r a n k } _ { m , T _ { A } } ( g _ { i m } ( T _ { A } ) ) } { n _ { m } ( T _ { A } ) + 1 } ,
$$

using average ranks for ties. This yields percentileranked features in (0, 1), where larger values indicate larger within-model distances.

Article-level geometric predictors are obtained by aggregating model-specific feature values for each article. For each geometric feature, we compute the mean, median, and standard deviation across embedding models to capture average position, central tendency, and cross-model variation.

## 4.2 Textual Features

Textual features measure whether anticipatory outliers are distinctive in content or form. They are computed once per article from its textual content.

We first compute tone features. Subjectivity is obtained with the French version of TextBlob (Loria, 2018). Neutrality is derived from the French VADER compound score (Hutto and Gilbert, 2014). Articles with strong positive or negative polarity receive lower neutrality scores, whereas articles with weak affective polarity receive higher scores.

We then compute length and surface-complexity features. Character and word count measure the amount of textual information available. Average sentence length in words, average word length in characters, total syllables, and average syllables per word describe the surface complexity and density of this information.

Finally, we extract named entities with the French spaCy pipeline (Honnibal et al., 2020). The total number of entities measures the degree of entity anchoring in the article. The number of distinct entities captures how varied this anchoring is. We also retain counts for persons, organizations, locations, and miscellaneous entities, which indicate whether the article is tied to specific actors, institutions, places, or other named references. Such anchoring may precede cluster formation when later articles repeatedly refer to the same entities.

## 4.3 Social Features

Social features capture early circulation on X using URL-sharing events observed no later than the publication-time cumulative snapshot at $T _ { A }$ . If an article has no observed social trace by $T _ { A }$ , all social variables are set to zero, so that the absence of early circulation is encoded explicitly.

We first measure direct sharing activity. For each article URL, we record the number of distinct users who shared it, the median follower count, tweet count, and listed count of these users to capture the breadth and typical visibility of early circulation.

We then encode audience overlap through a weighted article–article co-sharing graph, obtained as a one-mode projection of the underlying user– article bipartite graph. Nodes correspond to article URLs observed by $T _ { A }$ . Two URLs are connected when at least one user shared both URLs by $T _ { A }$ , and the edge weight is the number of shared users. The derived metrics capture the extent to which new outlier articles circulate through overlapping audiences before a topic is semantically consolidated (Granovetter, 1973; Ugander et al., 2012).

From this graph we retain three measures. The weighted clustering coefficient (Onnela et al., 2005) indicates whether the article belongs to a locally coherent co-sharing neighborhood. The Louvain community size (Blondel et al., 2008) measures the scale of the audience group in which the article circulates; the partition is computed with a fixed random seed, so that community assignments are reproducible. The bridge ratio measures the share of an article’s co-sharing weight that connects outside its own community.

## 5 Experimental Setup

## 5.1 Datasets

We evaluate the prediction task on two French news corpora. HYDRONEWSFR, described in Zve et al. (2026), focuses on hydrogen-energy narratives. We curated CLIMATENEWSFR, a climatechange corpus collected with the same pipeline. It combines two complementary streams: news articles retrieved from Google News results using the GNews Python library, and X posts linking to news articles, collected through the official X API. The corpus was collected using the French keyword “changement climatique” (“climate change”) and includes the available title and lead paragraph.

<table><tr><td>Corpus</td><td>Period</td><td>Articles X posts</td><td></td></tr><tr><td>HYDRONEWSFR</td><td>20 Mar–8 Jun 2025</td><td>1,616</td><td>1,533</td></tr><tr><td></td><td>CLIMATENEWSFR 27 Mar–27 May 2025</td><td>2,445</td><td>1,046</td></tr></table>

Table 1: Summary of the two main corpora.

Both corpora are temporally ordered, multisource news streams with limited coverage gaps. This property is important for the present task, since the transition of publication-time outliers into later topics can only be evaluated reliably when the underlying stream is dense enough to support temporal continuity. CLIMATENEWSFR is broader and more heterogeneous, covering political, social, and environmental aspects of climate change, whereas HYDRONEWSFR is centered on hydrogen-related industrial, technological, and policy developments.

## 5.2 Labeling-Pipeline Configuration

In all experiments, we use an ensemble of 11 French-specific and multilingual embedding models. The full list is given in Appendix B.1. For each model, we concatenate each article’s title and lead paragraph, embed the resulting text, project the embeddings to 20 dimensions with UMAP, cluster them daily with HDBSCAN, and align clusters across time using centroid matching with threshold $\theta _ { \mathrm { a l i g n } } = 0 . 3 0$ . This configuration follows the setting reported by $\mathrm { Z v e }$ et al. (2026), which was found to yield strong inter-model agreement on the $\mathcal { T O A }$ versus non-TOA distinction. We adapt their publicly available code to derive trajectory labels.<sup>2</sup> We keep the HDBSCAN library defaults, min\_cluster\_size=5 and min\_samples=5.

## 5.3 Supervised Learning and Evaluation

The supervised dataset pairs the consensus labels from Section 3 with the article-level publicationtime predictors from Section 4. Articles with uncertain consensus labels are excluded from supervised training and evaluation. We also discard the first five days of each corpus as a warm-up window before training. We compare five classifiers: logistic regression with $\ell _ { 2 }$ regularization (Hoerl and Kennard, 1970), a linear support vector classifier (Cortes and Vapnik, 1995), a decision tree (Breiman et al., 1984), a random forest (Breiman, 2001), and XGBoost (Chen and Guestrin, 2016).

Because anticipatory outliers are the rarer class, we use $F _ { 1 }$ as the primary metric and report recall and precision as complementary measures. We estimate performance with two protocols. The main protocol is 5-fold article-level cross-validation, which measures how much information publicationtime features carry about later trajectories. We complement it with a chronological forward-chaining protocol, in which each article is predicted by a model trained only on strictly earlier articles, and which therefore approximates deployment conditions. We also report a constant-positive baseline that predicts $\mathcal { T O A }$ for all eligible articles. The complete setup is provided in Appendix B.3.

## 6 Results

We examine whether anticipatory outliers can be predicted from information available at publication time. The analysis proceeds in three steps. We first vary the consensus threshold used to filter and label publication-time outliers, assessing how prediction changes as supervision is supported by stronger agreement across embedding models. We then compare multiple classifier families under the selected consensus settings, to determine whether the signal is specific to one learning algorithm or remains stable across modeling choices. Finally, we evaluate prediction chronologically.

## 6.1 Effect of Agreement Thresholds

Table 2 reports the coverage–performance tradeoff obtained by varying the embedding-model consensus threshold $k ,$ using XGBoost at $T _ { A }$ . As introduced in Section 3.3, k specifies how many embedding models must agree that an article is a publication-time outlier and, for positive labels, that it follows an anticipatory trajectory $( \mathcal { T O A } )$

Increasing k makes the supervision set more selective. Fewer articles are retained, but their labels are supported by stronger inter-model agreement. $\mathrm { A t } \ k = 1$ , the model already reaches $F _ { 1 } = 0 . 7 6 5$ on HYDRONEWSFR and $F _ { 1 } = 0 . 7 7 8$ on CLIMATE-NEWSFR. Performance tends to improve under stricter agreement, consistent with reduced label noise, but high thresholds retain few positives and show greater fold-level variability. For the detailed classifier and interpretability analyses, we use $k = 4$ for HYDRONEWSFR (83 positives, 254 negatives) and $k = 6$ for CLIMATENEWSFR (68 positives, 330 negatives). These are the largest agreement thresholds before the positive class becomes very small and fold-level estimates become visibly less stable, as shown in Table 2.

## 6.2 Prediction at Publication Time

Using the selected corpus-specific values of $k ,$ we train and compare five classifier families on the corresponding article-level labeled subsets. Table 3 reports cross-validated performance at $T _ { A }$

All trained models substantially outperform the constant-positive baseline, indicating that publication-time features contain information about later integration into newly formed topics when trajectory labels are supported by consistent evidence across embedding models. XG-Boost obtains the highest $F _ { 1 }$ -score in both corpora: $0 . 9 1 2 { \scriptstyle \pm 0 . 0 6 9 }$ in HYDRONEWSFR and $0 . 9 6 9 { \scriptstyle \pm 0 . 0 4 7 }$ in CLIMATENEWSFR. Linear models are more recall-oriented, with Linear SVC and logistic regression reaching recall $0 . 9 4 1 { \pm } 0 . 0 6 4$ in HYDRONEWSFR, and logistic regression reaching recall $0 . 9 8 5 { \pm } 0 . 0 3 1$ in CLIMATENEWSFR.

## 6.3 Chronological Evaluation

Cross-validation lets a model see articles published after those it is tested on. To assess performance under deployment conditions, we additionally evaluate XGBoost with chronological forward chaining: at each cutoff t, the model is trained on all labeled articles with $T _ { A } \leq t$ and tested on those published in the following $W$ days, so that each evaluated article is predicted exactly once by a model that has seen only strictly earlier articles. Table 4 reports pooled $F _ { 1 }$ for $W = 7$ days; the full protocol and other window lengths are given in Appendix E.

<table><tr><td rowspan="2">k</td><td colspan="4">HYDRONEWSFR</td><td colspan="4">CLIMATENEWSFR</td></tr><tr><td>Retained</td><td>Positive</td><td>Negative</td><td> $F _ { \mathrm { 1 } } \pm \mathrm { s t d }$ </td><td>Retained</td><td>Positive</td><td>Negative</td><td> $F _ { \mathrm { 1 } } \pm \mathrm { s t d }$ </td></tr><tr><td>1</td><td>1235</td><td>544</td><td>691</td><td> $0 . 7 6 5 \pm 0 . 0 1 3$ </td><td>1941</td><td>875</td><td>1066</td><td> $0 . 7 7 8 \pm 0 . 0 1 2$ </td></tr><tr><td>2</td><td>773</td><td>276</td><td>497</td><td> $0 . 8 1 1 \pm 0 . 0 5 6$ </td><td>1390</td><td>489</td><td>901</td><td> $0 . 8 5 1 \pm 0 . 0 1 6$ </td></tr><tr><td>3</td><td>508</td><td>150</td><td>358</td><td> $0 . 8 9 5 \pm 0 . 0 2 2$ </td><td>1005</td><td>273</td><td>732</td><td> $0 . 8 9 5 \pm 0 . 0 2 0$ </td></tr><tr><td>4</td><td>337</td><td>83</td><td>254</td><td> $\mathbf { 0 . 9 1 2 \pm 0 . 0 6 9 }$ </td><td>766</td><td>182</td><td>584</td><td> $0 . 9 1 5 \pm 0 . 0 3 8$ </td></tr><tr><td>5</td><td>219</td><td>39</td><td>180</td><td> $0 . 8 6 0 \pm 0 . 1 1 6$ </td><td>570</td><td>115</td><td>455</td><td> $0 . 9 5 2 \pm 0 . 0 3 3$ </td></tr><tr><td>6</td><td>145</td><td>21</td><td>124</td><td> $0 . 9 1 0 \pm 0 . 0 9 2$ </td><td>398</td><td>68</td><td>330</td><td> $\mathbf { 0 . 9 6 9 \pm 0 . 0 4 7 }$ </td></tr><tr><td>7</td><td>81</td><td>13</td><td>68</td><td> $0 . 9 1 7 \pm 0 . 1 4 4$ </td><td>270</td><td>43</td><td>227</td><td> $0 . 9 4 0 \pm 0 . 0 1 6$ </td></tr><tr><td>8</td><td>40</td><td>6</td><td>34</td><td> $0 . 8 0 0 \pm 0 . 4 0 0$ </td><td>161</td><td>21</td><td>140</td><td> $0 . 9 4 9 \pm 0 . 0 6 3$ </td></tr></table>

Table 2: Effect of the agreement threshold k at $T _ { A }$ with XGBoost. Scores are cross-validated means ± standard deviations across folds. Bold marks the $F _ { 1 }$ scores of the selected k values used in the detailed analyses.

<table><tr><td></td><td colspan="3">HYDRONEWSFR</td><td colspan="3">CLIMATENEWSFR</td></tr><tr><td>Model</td><td>Precision</td><td> $F _ { \mathrm { 1 } } { \mathrm { - s c o r e } }$ </td><td>Recall</td><td>Precision</td><td> $F _ { \mathrm { 1 } } { \mathrm { - s c o r e } }$ </td><td>Recall</td></tr><tr><td>Baseline</td><td>0.246</td><td>0.395</td><td>1.000</td><td>0.171</td><td>0.291</td><td>1.000</td></tr><tr><td>Linear SVC</td><td>0.853</td><td>0.889</td><td>0.941</td><td>0.889</td><td>0.906</td><td>0.930</td></tr><tr><td>LogReg  $\ell _ { 2 }$ </td><td>0.838</td><td>0.882</td><td>0.941</td><td>0.895</td><td>0.936</td><td>0.985</td></tr><tr><td>Decision Tree</td><td>0.803</td><td>0.835</td><td>0.891</td><td>0.867</td><td>0.898</td><td>0.937</td></tr><tr><td>Random Forest</td><td>0.867</td><td>0.885</td><td>0.917</td><td>0.905</td><td>0.935</td><td>0.969</td></tr><tr><td>XGBoost</td><td>0.919</td><td>0.912</td><td>0.917</td><td>0.970</td><td>0.969</td><td>0.969</td></tr></table>

Table 3: Article-level precision, $F _ { \mathrm { 1 } } { \mathrm { - s c o r e . } }$ and recall at $T _ { A }$ under selected consensus settings: $k \ = \ 4$ for HYDRONEWSFR and $k = 6$ for CLIMATENEWSFR. Results are averaged over five article-level CV folds.

<table><tr><td>Corpus</td><td>Pos. Neg.</td><td> $F _ { 1 }$  [95% CI]</td><td>Base</td></tr><tr><td>HYDRONEWSFR</td><td>42</td><td>211 0.762 [0.65, 0.86]</td><td>0.285</td></tr><tr><td>CLIMATENEWSFR</td><td>26</td><td>316 60.800 [0.66, 0.91]</td><td>0.141</td></tr></table>

Table 4: Forward-chaining results at $T _ { A }$ with XGBoost and $W = 7 – \mathrm { d a y }$ test windows, under the selected consensus settings. Cells report $F _ { 1 }$ with a 95% percentile bootstrap interval and the constant-positive baseline.

As expected, chronological scores are lower than cross-validated ones, since each model is trained on fewer articles and cannot exploit any later structure of the stream. Both corpora nevertheless remain well above the constant-positive baseline even at the lower confidence bound, and results are stable across window lengths on the longer extended corpus. The cross-validated estimates should thus be read as an upper bound on the information carried by publication-time features, and the chronological estimates as a more conservative indication of prospective performance.

## 7 Interpretability Analysis

We analyze the learned XGBoost models from the cross-validated setting of Section 6.2 to identify which publication-time features contribute to T OA predictions. We first compare feature families through ablation, then inspect global SHAP importance and local out-of-fold explanations.

## 7.1 Ablation Study

Table 5 reports feature-family ablations for XG-Boost at $T _ { A }$ . “Only” rows use a single feature family, whereas “All w/o” rows remove one family from the full feature set. To test whether the main contrasts are stable across folds, we compare paired fold-level $F _ { 1 }$ scores from the same five article-level cross-validation splits using paired t-tests, with Benjamini–Hochberg correction (Benjamini and Hochberg, 1995).

Geometry accounts for most of the predictive signal. Using geometry alone nearly matches the full model in both corpora, with no significant $F _ { 1 }$ decrease in HYDRONEWSFR (0.900 versus 0.912) and no decrease in CLIMATENEWSFR (0.969 versus 0.969). In contrast, removing geometry yields a large and significant drop, from 0.912 to 0.385 in HYDRONEWSFR and from 0.969 to 0.226 in CLIMATENEWSFR. The precision–recall columns show that this effect is not driven by only one side of the tradeoff, as removing geometry lowers both precision and recall in both corpora.

Text and social features are weak as standalone predictors in both corpora, and both perform significantly below the geometry-only model. Removing either family from the full model does not significantly reduce $F _ { 1 }$ . This suggests that these features provide at most auxiliary, corpus-specific information. The partial exception is social-only prediction in HYDRONEWSFR, with moderate precision but very low recall. This pattern indicates that social features help identify a small subset of anticipatory cases, but do not characterize the class.

<table><tr><td></td><td colspan="3">HYDRONEWSFR</td><td colspan="3">CLIMATENEWSFR</td></tr><tr><td>Feature set</td><td>Precision</td><td> $F _ { 1 }$ </td><td></td><td>Recall Precision</td><td> $F _ { 1 }$ </td><td>Recall</td></tr><tr><td>All features</td><td>0.919</td><td>0.912</td><td>0.917</td><td>0.970</td><td>0.969</td><td>0.969</td></tr><tr><td>Only geometry</td><td>0.907</td><td>0.900</td><td>0.905</td><td>0.970</td><td>0.969</td><td>0.969</td></tr><tr><td>Only text</td><td>0.322</td><td>0.272</td><td>0.242</td><td>0.279</td><td>0.202</td><td>0.165</td></tr><tr><td>Only social</td><td>0.707</td><td>0.349</td><td>0.239</td><td>0.235</td><td>0.083‡</td><td>0.212</td></tr><tr><td>All w/o geometry</td><td>0.406</td><td>0.385†</td><td>0.371</td><td>0.285</td><td>0.226†</td><td>0.194</td></tr><tr><td>All w/o social</td><td>0.930</td><td>0.917</td><td>0.917</td><td>0.970</td><td>0.969</td><td>0.969</td></tr><tr><td>All w/o text</td><td>0.919</td><td>0.905</td><td>0.904</td><td>0.969</td><td>0.962</td><td>0.955</td></tr></table>

Table 5: Ablations for XGBoost at $T _ { A }$ . Results are $5 \textdegree$ fold CV means. Symbols mark significant $F _ { 1 }$ drops at $\alpha = 0 . 0 5 \colon 1$ † vs. all features; ‡ vs. geometry only.

## 7.2 SHAP Analysis

## 7.2.1 Global Importance

Table 6 reports the top global SHAP features for the XGBoost models, computed with TreeExplainer (Lundberg et al., 2020). Mean absolute SHAP values measure each feature’s average contribution magnitude, while Spearman correlations between feature values and signed SHAP values indicate whether larger feature values push predictions toward or away from TOA.

The top-ranked features are geometric in both corpora. In HYDRONEWSFR, the largest contribution comes from the median second-nearestcentroid distance $\mathrm { \langle | S H A P | } \quad = \quad 0 . 1 0 1 8 )$ , followed by the median nearest-centroid distance $( | \mathrm { S H A P } | = 0 . 0 5 4 4 )$ . In CLIMATENEWSFR, the strongest feature is the median standard deviation of 20-nearest-neighbor distances (|SHAP| = 0.0773), followed by median nearest- and secondnearest-centroid distances. These features are positively correlated with signed SHAP values, indicating that articles farther from existing topic centroids, or located in sparser and more heterogeneous local neighborhoods, receive higher TOA scores.

The number of outliers present in the snapshot at $T _ { A }$ shows the opposite pattern. In both corpora, larger outlier counts push predictions away from $\mathcal { T O A }$ , with strong negative correlations in HY-DRONEWSFR $( r = - 0 . 8 1$ and $r = - 0 . 8 7 )$ and CLIMATENEWSFR $( r = - 0 . 9 1 $ and $r = - 0 . 9 4 )$ ). Thus, the model does not treat the number of outliers present at $T _ { A }$ as evidence of anticipation. High $\mathcal { T O A }$ scores are instead associated with articles that tend to be distant from existing topic centroids, lie in sparse and heterogeneous local neighborhoods, and receive higher HDBSCAN outlierness scores.

Textual and social features have smaller and more corpus-specific effects, as shown in Table 7. In HYDRONEWSFR, the strongest textual feature is miscellaneous entity count, while total and personentity counts are negatively associated with $\tau \mathcal { O A }$ In CLIMATENEWSFR, subjectivity is the strongest textual feature. Social features contribute weakly in HYDRONEWSFR, led by weighted clustering in the article–article co-sharing graph, and have zero mean absolute SHAP value in CLIMATENEWSFR. These patterns support the ablation result, with nongeometric features providing auxiliary information rather than the main predictive signal.

<table><tr><td>Feature</td><td>Mean |SHAP| Spearman r</td><td></td></tr><tr><td>HYDRONEWSFR</td><td></td><td></td></tr><tr><td>ner_misc</td><td>0.0113</td><td>0.82***</td></tr><tr><td>ner_total_ents</td><td>0.0097</td><td>-0.79***</td></tr><tr><td>ner_person</td><td>0.0082</td><td>-0.81***</td></tr><tr><td>text_subjectivity</td><td>0.0071</td><td>0.82***</td></tr><tr><td>media_weighted_clustering</td><td>0.0027</td><td> $0 . 5 3 ^ { * * * }$ </td></tr><tr><td>media_community_size</td><td>0.0022</td><td>0.49***</td></tr><tr><td>CLIMATENEWSFR</td><td></td><td></td></tr><tr><td>text_subjectivity</td><td>0.0027</td><td> $0 . 7 3 ^ { * * * }$ </td></tr><tr><td>ner_misc</td><td>0.0025</td><td> $- 0 . 8 1 ^ { * * * }$ </td></tr></table>

Table 7: Main non-geometric SHAP features at $T _ { A }$ . Significance coding: $^ { * * * } p < 0 . 0 0 1$

## 7.2.2 Local Explanations

We examine one true-positive out-of-fold example from each corpus, selected to illustrate substantively interpretable later topics. Each article is predicted by a fold-specific model that was not trained on it. Retrospective topic reconstruction is then used to interpret the topic the article later joined.

HYDRONEWSFR. We consider the H2 Mobile article on Germany’s use of salt caverns for largescale hydrogen storage.<sup>3</sup> Its out-of-fold predicted probability at $T _ { A }$ is 0.9983. It is labeled positive, with 5 votes across the embedding-model ensemble. In the retrospective multilingual-e5-large reconstruction, $T _ { A }$ is 25 April 2025, while $T _ { T } = T _ { I } =$ 9 May 2025. The article therefore appears before the topic becomes a cluster in this reconstruction. The later topic is centered on large-scale underground hydrogen storage in salt formations and includes articles on Storengy’s storage projects, German cavern capacity, and pilot demonstrations in saline formations.

<table><tr><td colspan="4">HYDRONEWSFR</td><td colspan="4">CLIMATENEWSFR</td></tr><tr><td>Feature</td><td>Cat.</td><td>Mean |SHAP|</td><td>Spearman r</td><td>Feature</td><td>Cat.</td><td>Mean |SHAP|</td><td>Spearman r</td></tr><tr><td>d2_second_centroid_pct_median</td><td>G</td><td>0.1018</td><td> $0 . 6 9 ^ { * * * } \uparrow$ </td><td>knn_std_k20_pct_median</td><td>G</td><td>0.0773</td><td>0.28***↑</td></tr><tr><td>d1_nearest_centroid_pct_median</td><td>G</td><td>0.0544</td><td> $0 . 6 4 ^ { * * * } \dot { \uparrow }$ </td><td>d1_nearest_centroid_pct_median</td><td>G</td><td>0.0619</td><td> $0 . 4 8 ^ { * * * } \dot { \uparrow }$ </td></tr><tr><td>d2_second_centroid_pct_mean</td><td>G</td><td>0.0509</td><td> $0 . 6 7 ^ { * * * } \dot { \uparrow }$ </td><td>d2_second_centroid_pct_median</td><td>G</td><td>0.0495</td><td> $0 . 3 \dot { 8 } ^ { \ast \ast \ast } \dot { \uparrow }$ </td></tr><tr><td>d1_nearest_centroid_pct_mean</td><td>G</td><td>0.0420</td><td> $0 . 8 4 ^ { * * * } \dot { \uparrow }$ </td><td>n_recent_outliers_mean</td><td>G</td><td>0.0255</td><td> $- 0 . 9 1 ^ { \ast \ast \ast } \downarrow$ </td></tr><tr><td>knn_mean_k20_pct_median</td><td>G</td><td>0.0417</td><td> $0 . 4 8 ^ { * * * } \dot { \uparrow }$ </td><td>d1_nearest_centroid_pct_mean</td><td>G</td><td>0.0145</td><td> $0 . 8 6 ^ { * * * } \uparrow$ </td></tr><tr><td>knn_mean_k20_pct_mean</td><td>G</td><td>0.0264</td><td> $0 . 6 5 ^ { * * * } \stackrel { . } { \uparrow }$ </td><td>knn_mean_k20_pct_median</td><td>G</td><td>0.0139</td><td> $0 . 3 8 ^ { * * * } \stackrel { . } { \uparrow }$ </td></tr><tr><td>n_recent_outliers_median</td><td>G</td><td>0.0136</td><td> $- 0 . 8 1 ^ { \ast \ast \ast } \downarrow$ </td><td>outlier_score_std</td><td>G</td><td>0.0103</td><td> $0 . 6 7 ^ { * * * } \uparrow$ </td></tr><tr><td>outlier_score_median</td><td>G</td><td>0.0135</td><td> $0 . 8 9 ^ { * * * } \uparrow$ </td><td>outlier_score_mean</td><td>G</td><td>0.0074</td><td> $0 . 6 7 ^ { * * * } \stackrel { . } { \uparrow }$ </td></tr><tr><td>knn_std_k20_pct_mean</td><td>G G</td><td>0.0129</td><td> $0 . 3 6 ^ { * * * } \dagger$ </td><td>d2_second_centroid_pct_mean</td><td>G</td><td>0.0067</td><td> $0 . 3 7 ^ { * * * } \stackrel { . } { \uparrow }$ </td></tr><tr><td>n_recent_outliers_mean</td><td></td><td>0.0119</td><td> $- 0 . 8 7 ^ { * * * } \downarrow$ </td><td> $\mathsf { n \_ r e c e n t \_ o u t l i e r s \_ s t d }$ </td><td>G</td><td>0.0066</td><td> $- 0 . 9 4 ^ { * * * } \downarrow$ </td></tr></table>

Table 6: Top XGBoost features, ranked separately by mean absolute SHAP value. Arrows indicate the sign of the Spearman correlation between feature value and SHAP contribution. G = geometry, T = text, S = social. Significance coding: $^ { * * * } p < 0 . 0 0 1 , ^ { * * } p < 0 . 0 1 , ^ { * } p < 0 . 0 5$

The strongest positive contributions come from distances to existing topic structure, especially the median second-nearest-centroid distance (+0.2034), the median nearest-centroid distance (+0.1254), and the median 20-nearest-neighbor distance (+0.1087). The median outlier-score also contributes positively (+0.0382). Together, these features place the article away from existing hydrogen-topic centers and in a sparse neighborhood. Non-geometric effects are small: subjectivity (−0.0165) and the number of named entities (−0.0094) push slightly away from TOA, while the number of unique users sharing the article contributes positively (+0.0033).

CLIMATENEWSFR. We consider the CNRS article on archaeological sites threatened by climate change.<sup>5</sup> Its out-of-fold predicted probability at $T _ { A }$ is 0.9978. It is labeled positive, with 10 votes. In the retrospective mistral-embed reconstruction, $T _ { A } = 2$ April 2025, while $T _ { T } = T _ { I } = 1 7$ April 2025, so this article is a $\mathcal { T O A } _ { \mathrm { f i r s t } }$ case. The article therefore appears before climate-related cultural heritage and archaeology becomes a cluster in this reconstruction. The later topic includes articles on Greek historical sites threatened by climate change and on the transformation of cultural heritage worldwide under climate change, all published later than the CNRS article.<sup>6</sup>

The largest positive contributions are the median nearest-centroid distance (+0.2019), the median standard deviation of 20-nearest-neighbor distances (+0.1983), and the mean second-nearestcentroid distance (+0.1038). The number of outliers at $T _ { A }$ also supports the prediction: consistent with the negative global correlation reported above, the article was published when the number of outliers at $T _ { A }$ was low, and this low count contributes positively (+0.0832). HDBSCAN outlier-score variability contributes more modestly (+0.0327). These features place the article away from existing climate topics, in a sparse and heterogeneous neighborhood. Textual effects are much smaller: average sentence length is the main negative correction (−0.0109), while subjectivity contributes weakly toward TOA (+0.0052). Social features do not materially affect this local prediction.

In both examples, T OA predictions are driven by distance from existing topic centers combined with a non-uniform local neighborhood at $T _ { A }$

## 8 Conclusion

We introduced anticipatory outlier detection, a prospective task that predicts if a publication-time outlier will later become an early member of a topic that has not yet formed. On two French news corpora, we show that this trajectory is predictable from publication-time information, especially under stricter inter-model consensus labels, and that the signal persists under a strictly chronological evaluation. Prediction is driven mainly by embedding-space geometry: anticipatory outliers lie far from existing topic centroids, in sparse and uneven neighborhoods. Textual and social variables provide weaker, more corpus-specific signals.

More broadly, this work raises the question of whether anticipatory outliers reflect a general property of dynamic embedding spaces beyond news, and even beyond text. Similar signals may arise in images, video, or audio, when initially isolated representations later become part of emerging semantic or stylistic structures.

## Limitations

Our methodology provides evidence that publication-time outliers can contain signals of later topic formation within an embedding-based framework. The evaluation relies on consensus labels obtained across embedding models, mitigating dependence on any single embedding space. Qualitative examples further indicate that several predicted outliers correspond to coherent later topics. Nevertheless, external validation remains necessary. A systematic human-annotation study would help assess how these signals are perceived in terms of novelty, relevance, and real-world significance.

The study relies on two French news corpora with complementary scopes. HYDRONEWSFR is an existing dataset focused on a specialized industrial and policy domain, while CLIMATENEWSFR is a curated dataset covering a broader and more heterogeneous public issue. Both corpora are temporally ordered news streams built from Google News results and X-sharing activity of news articles, with articles drawn from multiple media sources. This provides relatively dense time series, reduces timeline discontinuities, and limits dependence on a single source while keeping the language and collection pipeline controlled across settings. However, the comparison remains limited to two domains within a single language and does not cover other media ecosystems or longer temporal scales. Replication on larger multilingual corpora and longer observation windows would further assess the transferability of the approach.

The supervised labels depend on the retrospective trajectory-reconstruction pipeline. In the main setting, embeddings are projected with 20- dimensional UMAP, clustered with HDBSCAN, and aligned across cumulative snapshots using a fixed topic-alignment threshold $( \theta _ { \mathrm { a l i g n } } = 0 . 3 0 )$ . This configuration follows prior trajectory-labeling work showing strong inter-model agreement for the anticipatory versus non-anticipatory distinction, with limited variation across UMAP dimensionalities (Zve et al., 2026). However, the present study adds a downstream prospective prediction task, so robustness at the labeling stage may not fully imply robustness of prediction scores. Future work should therefore test whether the observed stability across UMAP dimensionalities also holds for prediction, and should examine alternative dimensionality-reduction methods, clustering algorithms, and alignment thresholds.

Finally, social features are limited to observed X-sharing activity, which provides only a partial view of online circulation. This limited coverage may partly explain the weak contribution of social features in our models. Early diffusion may also occur on other social-media platforms or through channels not captured in the present data. Future work should extend the analysis to additional platforms and circulation channels.

## Ethical Considerations

In line with open-science principles, the code for reproducing the experiments is publicly available in a dedicated GitHub repository (Appendix A). The embedding ensemble combines open-source and API-based models to assess robustness across model families used in research and applied settings. Where possible, we favor compact models to reduce unnecessary computational cost and environmental impact.

The study uses news articles retrieved from Google News results using the GNews Python library, and social-media sharing traces collected through the official X API. We do not redistribute raw text or raw social-media traces, since these may contain identifiable user activity and may be subject to publishers’ rights, platform terms, personal-data restrictions, or copyright constraints. Data can be shared for research purposes only upon reasonable request.

The method is intended for research on weaksignal detection in dynamic text streams, not as a deployable monitoring or decision system. Its outputs are corpus-level signals, not evidence that an article is objectively important, novel, or likely to shape future events. Human annotation, external validation, and calibration would be required before any applied use. The main practical risk is a biased allocation of attention: some predictions may reflect corpus-specific patterns, and other relevant documents may be missed. Any adaptation should document data provenance, media coverage, language scope, and source selection, and account for uncertainty.

## Acknowledgments

EZ thanks Infopro Digital for granting her the time to pursue her PhD thesis alongside her work.

## References

Charu C. Aggarwal. 2017. Outlier Analysis, 2 edition. Springer, Cham.

James Allan, editor. 2002. Topic Detection and Tracking: Event-Based Information Organization. Kluwer Academic Publishers, Boston, MA.

Mihael Ankerst, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. 1999. OPTICS: Ordering points to identify the clustering structure. In Proceedings of the 1999 ACM SIGMOD International Conference on Management of Data, pages 49–60. ACM.

H. Igor Ansoff. 1975. Managing strategic surprise by response to weak signals. California Management Review, 18(2):21–33.

Hila Becker, Mor Naaman, and Luis Gravano. 2011. Beyond trending topics: Real-world event identification on twitter. In Proceedings of the Fifth International AAAI Conference on Weblogs and Social Media, pages 438–441.

Yoav Benjamini and Yosef Hochberg. 1995. Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1):289–300.

David M. Blei and John D. Lafferty. 2006. Dynamic topic models. In Proceedings of the 23rd International Conference on Machine Learning, pages 113– 120. ACM.

Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, and Etienne Lefebvre. 2008. Fast unfolding of communities in large networks. Journal of Statistical Mechanics: Theory and Experiment, 2008(10):P10008.

Allaa Boutaleb, Jérôme Picault, and Guillaume Grosjean. 2024. BERTrend: Neural topic modeling for emerging trends detection. In Proceedings of the Workshop on the Future of Event Detection, pages 1–17, Miami, Florida, USA. Association for Computational Linguistics.

Leo Breiman. 2001. Random forests. Machine Learning, 45(1):5–32.

Leo Breiman, Jerome H. Friedman, Richard A. Olshen, and Charles J. Stone. 1984. Classification and Regression Trees. Wadsworth, Belmont, CA.

Ricardo J. G. B. Campello, Davoud Moulavi, Arthur Zimek, and Jörg Sander. 2015. Hierarchical density estimates for data clustering, visualization, and outlier detection. ACM Transactions on Knowledge Discoveryfrom Data, 10(1):1–51.

Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 785–794. ACM.

Clément Christophe, Julien Velcin, Jairo Cugliari, Manel Boumghar, and Philippe Suignard. 2021. Monitoring geometrical properties of word embeddings for detecting the emergence of new topics. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 994– 1003. Association for Computational Linguistics.

Corinna Cortes and Vladimir Vapnik. 1995. Supportvector networks. Machine Learning, 20(3):273–297.

Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. 2020. Topic modeling in embedding spaces. Transactions ofthe Associationfor Computational Linguistics, 8:439–453.

Ashkan Ebadi, Alain Auger, and Yvan Gauthier. 2026. WISDOM: An AI-powered framework for emerging research detection using weak signal analysis and advanced topic modelling. Journal of Informetrics, 20(1):101759.

Mark S. Granovetter. 1973. The strength of weak ties. American Journal ofSociology, 78(6):1360–1380.

Maarten Grootendorst. 2022. BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv preprint arXiv:2203.05794.

Elina Hiltunen. 2008. The future sign and its three dimensions. Futures, 40(3):247–260.

Arthur E. Hoerl and Robert W. Kennard. 1970. Ridge regression: Biased estimation for nonorthogonal problems. Technometrics, 12(1):55–67.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spaCy: Industrialstrength natural language processing in python.

C. J. Hutto and Eric Gilbert. 2014. VADER: A parsimonious rule-based model for sentiment analysis of social media text. In Proceedings ofthe International AAAI Conference on Web and Social Media, volume 8, pages 216–225.

Jon Kleinberg. 2002. Bursty and hierarchical structure in streams. In Proceedings of the Eighth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 91–101. ACM.

Harold W. Kuhn. 1955. The hungarian method for the assignment problem. Naval Research Logistics Quarterly, 2(1–2):83–97.

Steven Loria. 2018. TextBlob documentation. https: //textblob.readthedocs.io/. Release 0.15.

Scott M. Lundberg, Gabriel Erion, Hugh Chen, Alex DeGrave, Jordan M. Prutkin, Bala Nair, Ronit Katz, Jonathan Himmelfarb, Nisha Bansal, and Su-In Lee. 2020. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence, 2(1):56–67.

Leland McInnes, John Healy, and Steve Astels. 2017. hdbscan: Hierarchical density based clustering. Journal ofOpen Source Software, 2(11):205.

Leland McInnes, John Healy, Nathaniel Saul, and Lukas Großberger. 2018. UMAP: Uniform manifold approximation and projection. Journal ofOpen Source Software, 3(29):861.

Jukka-Pekka Onnela, Jari Saramäki, János Kertész, and Kimmo Kaski. 2005. Intensity and coherence of motifs in weighted complex networks. Physical Review E, 71(6):065103.

Saša Petrovic, Miles Osborne, and Victor Lavrenko.´ 2010. Streaming first story detection with application to twitter. In Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics, pages 181–189. Association for Computational Linguistics.

Marco A. F. Pimentel, David A. Clifton, Lei Clifton, and Lionel Tarassenko. 2014. A review of novelty detection. Signal Processing, 99:215–249.

Alexander Ratner, Stephen H. Bach, Henry Ehrenberg, Jason Fries, Sen Wu, and Christopher Ré. 2017. Snorkel: Rapid training data creation with weak supervision. Proceedings of the VLDB Endowment, 11(3):269–282.

Rion Snow, Brendan O’Connor, Daniel Jurafsky, and Andrew Y. Ng. 2008. Cheap and fast—but is it good? evaluating non-expert annotations for natural language tasks. In Proceedings ofthe 2008 Conference on Empirical Methods in Natural Language Processing, pages 254–263. Association for Computational Linguistics.

Alexander Strehl and Joydeep Ghosh. 2002. Cluster ensembles—a knowledge reuse framework for combining multiple partitions. Journal ofMachine Learning Research, 3:583–617.

Johan Ugander, Lars Backstrom, Cameron Marlow, and Jon Kleinberg. 2012. Structural diversity in social contagion. Proceedings ofthe National Academy of Sciences, 109(16):5962–5966.

Xuerui Wang and Andrew McCallum. 2006. Topics over time: A non-markov continuous-time model of topical trends. In Proceedings of the 12th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 424–433. ACM.

Evangelia Zve, Gauvain Bourgne, Benjamin Icard, and Jean-Gabriel Ganascia. 2026. From noise to signal: When outliers seed new topics. In Proceedings of the Fifteenth Language Resources and Evaluation Conference, pages 7523–7533, Palma de Mallorca, Spain. ELRA Language Resource Association.

Evangelia Zve, Benjamin Icard, Alice Breton, Lila Sainero, Gauvain Bourgne, and Jean-Gabriel Ganascia. 2025. From outliers to topics in language

models: Anticipating trends in news corpora. In Proceedings ofthe 8th International Conference on Natural Language and Speech Processing, pages 385– 398. Association for Computational Linguistics.

## A Supplementary Materials

We provide supplementary materials at: https: //github.com/evangeliazve/aacl-public. The repository includes the pipeline for applying the method to other corpora, together with the scripts used to reproduce the classification, ablation, SHAP, and forward-chaining results reported in the paper.

Data collection used GNews for Google News results<sup>7</sup> and the official X API<sup>8</sup>.

## B Supplementary Experimental Details

## B.1 Embedding Models

Table 8 lists the embedding models used for trajectory reconstruction.

<table><tr><td>Model</td><td>Dim</td><td>Language</td><td>Access</td></tr><tr><td>sentence-camembert-base</td><td>768</td><td>French</td><td>Open</td></tr><tr><td>Solon-large-0.1</td><td>1024</td><td>French</td><td>Open</td></tr><tr><td>paraphrase-MiniLM-L12-v2</td><td>384</td><td>Multilingual</td><td>Open</td></tr><tr><td>paraphrase-mpnet-base-v2</td><td>768</td><td>Multilingual</td><td>Open</td></tr><tr><td>LaBSE</td><td>768</td><td>Multilingual</td><td>Open</td></tr><tr><td>multilingual-e5-large</td><td>1024</td><td>Multilingual</td><td>Open</td></tr><tr><td>arctic-embed-1-v2.0</td><td>1024</td><td>Multilingual</td><td>Open</td></tr><tr><td>bge-m3</td><td>1024</td><td>Multilingual</td><td>Open</td></tr><tr><td>text-embedding-3-small</td><td>1536</td><td>Multilingual</td><td>API</td></tr><tr><td>gemini-embedding-001</td><td>3072</td><td>Multilingual</td><td>API</td></tr><tr><td>mistral-embed</td><td>1024</td><td>Multilingual</td><td>API</td></tr></table>

Table 8: Embedding models used in the experiments.

## B.2 Feature Glossary

Table 9 presents the features proposed and discussed in this study, complementing Section 4. Geometric features are per-model quantities; the classifiers use their mean, median, and standard deviation across the eleven embedding models, aggregated as described in Section 4.1. Social and textual features are computed once per article; social features use the sharing events observed up to T<sub>A</sub>, and textual features depend only on the article text.

## B.3 Evaluation Protocol

Warm-up period. Before training, we use the first five daily snapshots of each corpus as a warmup period, so that the cumulative clustering and trajectory-reconstruction pipeline has sufficient prior context.

<table><tr><td>Family</td><td>Feature</td><td>Definition / interpretation</td></tr><tr><td>Geometric: topic distance</td><td>d1_nearest_centroid_pct</td><td>Within-model percentile rank of the Euclidean distance from the article embedding to the nearest non-noise topic centroid in the snapshot at  $T _ { A } .$ </td></tr><tr><td>Geometric: topic distance</td><td>d2_second_centroid_pct</td><td>Within-model percentile rank of the Euclidean distance to the second-nearest non- noise topic centroid at  $T _ { A } .$ </td></tr><tr><td>Geometric: topic distance</td><td>margin_d2_minus_d1_pct</td><td>Within-model percentile rank of the margin between the second-nearest and nearest centroid distances at  $T _ { A }$ </td></tr><tr><td>Geometric: topic shape</td><td>mahal_nearest_pct</td><td>Within-model percentile rank of the minimum diagonal Mahalanobis distance to an existing topic cluster at  $T _ { A }$ </td></tr><tr><td>Geometric: local density</td><td>knn_mean_k20_pct</td><td>Within-model percentile rank of the mean Euclidean distance to up to 20 nearest neighbors in the snapshot at  $T _ { A } ,$  fewer when the snapshot is smaller.</td></tr><tr><td>Geometric: local density</td><td>knn_std_k20_pct</td><td>Within-model percentile rank of the standard deviation of distances to these same neighbors at  $T _ { A } .$ </td></tr><tr><td>hood</td><td>Geometric: outlier neighbor- outlier_proto_mean_dist_pct</td><td>Within-model percentile rank of the mean distance to up to ten nearest articles classified as outliers in the snapshot at  $T _ { A } .$ </td></tr><tr><td>Geometric: outlierness</td><td>outlier_score</td><td>HDBSCAN outlierness score at  $T _ { A }$  , from the GLOSH score (Campello et al., 2015; McInnes et al., 2017).</td></tr><tr><td>Geometric: outlier pool</td><td>has_recent_outliers</td><td>Indicator that at least one article is classified as an outlier in the snapshot at  $T _ { A }$ </td></tr><tr><td>Geometric: outlier pool</td><td>n_recent_outliers</td><td>Number of articles classified as outliers in the snapshot at  $T _ { A }$ </td></tr><tr><td>Social diffusion</td><td>soc_unique_users</td><td>Number of distinct X users who shared the article URL up to  $T _ { A } .$ </td></tr><tr><td>Social diffusion</td><td>ollowers_count</td><td>soc_median_user_public_metrics_f Median follower count of the X users who shared the article URL up to  $T _ { A } .$ </td></tr><tr><td>Social diffusion</td><td>weet_count</td><td>soc_median_user_public_metrics_t Median lifetime tweet count of the X users who shared the article URL up to</td></tr><tr><td>Social diffusion</td><td>isted_count</td><td>soc_median_user_public_metrics_1 Median listed count of the X users who shared the article URL up to  $T _ { A } .$ </td></tr><tr><td>Media co-sharing graph</td><td>media_weighted_clustering</td><td>Weighted clustering coefficient (Onnela et al., 2005) of the article URL node in the co-sharing graph built from shares observed up to  $T _ { A }$ </td></tr><tr><td>Media co-sharing graph</td><td>media_bridge_ratio</td><td>Share of the article node&#x27;s co-sharing weight that connects outside its Louvain com- munity (Blondel et al., 2008) in the co-sharing graph at  $T _ { A }$ </td></tr><tr><td>Media co-sharing graph</td><td>media_community_size</td><td>Size of the Louvain community (Blondel et al., 2008) containing the article URL node in the co-sharing graph at  $\overset { \cdot } { T _ { A } }$ </td></tr><tr><td>Text style</td><td>text_subjectivity</td><td>Subjectivity of the article text, from the French version of TextBlob (Loria, 2018).</td></tr><tr><td>Text style</td><td>text_neutrality</td><td>Neutrality of the article text, from the French VADER compound polarity magnitude (Hutto and Gilbert, 2014).</td></tr><tr><td>Readability / length</td><td>avg_sentence_len_words</td><td>Average sentence length in words.</td></tr><tr><td>Readability / length Readability / length</td><td>avg_word_len_chars</td><td>Average word length in characters.</td></tr><tr><td></td><td>total_syllables</td><td>Approximate total number of syllables in the article text, counted with a rule-based French heuristic.</td></tr><tr><td>Readability / length</td><td>avg_syllables_per_word</td><td>Approximate average number of syllables per word.</td></tr><tr><td>Readability / length</td><td>len_chars</td><td>Total number of characters in the article text.</td></tr><tr><td>Readability / length</td><td>len_words</td><td>Total number of word tokens in the article text.</td></tr><tr><td>Named entities</td><td>ner_total_ents</td><td>Total number of named entities detected with the French spaCy pipeline (Honnibal et al., 2020).</td></tr><tr><td>Named entities</td><td>ner_distinct_ents</td><td>Number of distinct named-entity strings detected with the same pipeline (Honnibal</td></tr><tr><td>Named entities</td><td>ner_person</td><td>et al., 2020). Count of person entities detected with the same pipeline (Honnibal et al., 2020).</td></tr><tr><td>Named entities</td><td>ner_org</td><td>Count of organization entities detected with the same pipeline (Honnibal et al., 2020).</td></tr><tr><td>Named entities</td><td>ner_loc</td><td>Count of location entities detected with the same pipeline (Honnibal et al., 2020).</td></tr><tr><td>Named entities</td><td>ner_misc</td><td>Count of named entities not assigned to person, organization, or location categories (Honnibal et al., 2020).</td></tr></table>

Table 9: Glossary of predictors used in the supervised models.

Cross-validation. Performance is estimated with 5-fold article-level cross-validation with approximately 80/20 train–test split in each fold, with four folds used for training and the remaining fold held out for evaluation. Folds are constructed at the article level, so that all records associated with the same article are assigned to the same fold and no article can appear in both the training and test sets.

Preprocessing. All preprocessing is fit within each training fold. Missing values are medianimputed using the training split only. Logistic regression and Linear SVC additionally use standardization fit on the training split. Tree-based models are trained on imputed but unscaled features.

Class imbalance and baseline. For logistic regression, Linear SVC, decision tree, and random forest, we use class\_weight=balanced. For XG-Boost, scale\_pos\_weight is set once per corpus and consensus setting to $N _ { - } / \operatorname* { m a x } ( N _ { + } , 1 )$ , where $N _ { + }$ and N are the positive and negative counts in the selected labeled subset for that setting.

SHAP Analysis. For the XGBoost interpretability analysis, we compute probability-scale SHAP values with TreeExplainer (Lundberg et al., 2020).

## B.4 Classifier Hyperparameters

Table 10 details the fixed parameters of all classifiers used in the experiments.

Model Hyperparameters   
Random Forest n\_estimators=600, max\_depth=8, min\_samples\_split=20, min\_samples\_leaf=10,   
max\_features=sqrt, class\_weight=balanced   
LogReg ℓ<sub>2</sub> penalty=l2, solver=liblinear, max\_iter=2000, class\_weight=balanced   
Linear SVC max\_iter=5000, class\_weight=balanced   
Decision Tree max\_depth=6, min\_samples\_leaf=10, class\_weight=balanced   
XGBoost n\_estimators=300, max\_depth=4, learning\_rate=0.05, subsample=0.8,   
colsample\_bytree=0.8, objective=binary:logistic, eval\_metric=logloss,   
scale\_pos\_ $\mathsf { . w e i g h t { = } } N _ { - } / \operatorname* { m a x } ( N _ { + } , 1 )$  
Table 10: Classifier hyperparameters used in the experiments. We use random\_state=42 where applicable. For XGBoost, $N _ { + }$ and $N _ { - }$ denote the positive and negative counts in the selected labeled subset for the corresponding corpus–consensus setting.

## C Supplementary Classification Results

## C.1 Full Results at Selected k

Table 11 reports precision, recall, and F<sub>1</sub>-score with standard deviations across folds for the selected consensus settings, complementing Table 3.

## C.2 Classifier Results Across Agreement Thresholds

Table 12 reports classifier $F _ { 1 }$ -scores across different values of k at $T _ { A }$ . These results complement the main XGBoost threshold analysis in Section 6.1, by showing that the coverage–confidence pattern is stable across classifier families.

## C.3 Fold-level Ablation Comparisons

Table 13 reports paired fold-level $F _ { 1 }$ comparisons for the main ablation contrasts in Table 5. Tests are computed on the same five article-level crossvalidation folds used to produce the ablation results. We report the mean paired difference $\Delta F _ { 1 }$ , the paired t-test p-value, and the Benjamini–Hochberg corrected q-value (Benjamini and Hochberg, 1995).

## D Robustness Check: Extended Hydrogen Corpus

We repeated the experiment on an extended hydrogen corpus spanning 1 January 2025 to 8 June 2025. This analysis is not part of the main evaluation because social features were not uniformly available over the full period. We use only geometric and textual features and interpret the results as supporting evidence under a longer observation window. Table 14 shows the same pattern observed in the main experiments: stricter k retain fewer articles but yield stronger predictive performance.

## E Forward-Chaining Evaluation

This appendix details the chronological evaluation summarized in Section 6.3. At each cutoff t, the model is trained on all articles with $T _ { A } \le t$ and tested on those with $T _ { A } \in ( t , t + W ]$ ; the cutoff then advances by W, so each evaluated article is predicted exactly once, by a model trained only on strictly earlier articles. Origins are restricted to $t \leq T _ { A } ^ { \mathrm { m a x } } - W$ , so the final partial window is not evaluated. The warm-up before the first origin follows an a-priori minimum-training-size rule, two weeks for the main corpora and four weeks for the extended corpus of Appendix D. The decision threshold is fixed at 0.5 and XGBoost follows Appendix B.4, with scale\_pos\_weight recomputed on each training set and median imputation fit on training articles only. Articles in the warm-up and in the final partial window are not evaluated, which is why the counts in Table 15 are smaller than in Table 2. Brackets give 95% percentile bootstrap intervals on pooled $F _ { 1 }$ (2,000 article resamples). Results come from a single run in one pinned environment.

All three corpora exceed the constant-positive baseline even at the lower confidence bound. The extended corpus reaches the highest scores; it also covers a longer period and contains a higher proportion of positives. Under the window-fit rule, the extended corpus pools 129 positives and 290 negatives at $W = 1 4 \mathrm { d } ;$ with an 8-week warm-up, $F _ { 1 } = 0 . 8 5 1 [ 0 . 7 9 , 0 . 9 1 ]$ . Scores are stable across window lengths on the extended corpus and more variable on the two smaller ones, where each window contains few positives. Per-window tables, the full warm-up × window grid, and per-subclass predicted-positive rates are available upon request.

<table><tr><td rowspan="2">Model</td><td colspan="3">HYDRONEWSFR</td><td colspan="3">CLIMATENEWSFR</td></tr><tr><td>Precision</td><td>Recall</td><td> $F _ { 1 }$ </td><td>Precision</td><td>Recall</td><td> $F _ { 1 }$ </td></tr><tr><td>Baseline</td><td> $0 . 2 4 6 { \pm } 0 . 0 1 1$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 3 9 5 { \pm } 0 . 0 1 4$ </td><td> $0 . 1 7 1 { \scriptstyle \pm 0 . 0 1 7 }$ </td><td> $1 . 0 0 0 { \scriptstyle \pm 0 . 0 0 0 }$ </td><td> $0 . 2 9 1 { \scriptstyle \pm 0 . 0 2 5 }$ </td></tr><tr><td>Linear SVC</td><td> $0 . 8 5 3 { \scriptstyle \pm 0 . 0 9 5 }$ </td><td> $0 . 9 4 1 { \scriptstyle \pm 0 . 0 6 4 }$ </td><td> $0 . 8 8 9 { \pm } 0 . 0 5 6$ </td><td> $0 . 8 8 9 { \pm } 0 . 0 8 3$ </td><td> $0 . 9 3 0 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 9 0 6 { \scriptstyle \pm 0 . 0 5 1 }$ </td></tr><tr><td>LogReg l2</td><td> $0 . 8 3 8 { \pm } 0 . 0 7 3$ </td><td> $0 . 9 4 1 { \scriptstyle \pm 0 . 0 6 4 }$ </td><td> $0 . 8 8 2 { \pm } 0 . 0 4 0$ </td><td> $0 . 8 9 5 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 9 8 5 { \scriptstyle \pm 0 . 0 3 1 }$ </td><td> $0 . 9 3 6 { \pm } 0 . 0 4 4$ </td></tr><tr><td>Decision Tree</td><td> $0 . 8 0 3 { \scriptstyle \pm 0 . 0 8 3 }$ </td><td> $0 . 8 9 1 { \scriptstyle \pm 0 . 1 1 5 }$ </td><td> $0 . 8 3 5 { \scriptstyle \pm 0 . 0 5 1 }$ </td><td> $0 . 8 6 7 { \scriptstyle \pm 0 . 0 8 5 }$ </td><td> $0 . 9 3 7 { \pm } 0 . 0 5 8$ </td><td> $0 . 8 9 8 { \pm } 0 . 0 5 8$ </td></tr><tr><td>Random Forest</td><td> $0 . 8 6 7 { \scriptstyle \pm 0 . 1 0 3 }$ </td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $0 . 8 8 5 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 9 0 5 { \scriptstyle \pm 0 . 0 7 9 }$ </td><td> $0 . 9 6 9 { \pm } 0 . 0 3 8$ </td><td> $0 . 9 3 5 { \scriptstyle \pm 0 . 0 5 7 }$ </td></tr><tr><td>XGBoost</td><td> $0 . 9 1 9 { \pm } 0 . 1 1 2$ </td><td> $0 . 9 1 7 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td> $0 . 9 1 2 { \scriptstyle \pm 0 . 0 6 9 }$ </td><td> $0 . 9 7 0 { \scriptstyle \pm 0 . 0 3 7 }$ </td><td>0.969±0.062</td><td> $0 . 9 6 9 { \scriptstyle \pm 0 . 0 4 7 }$ </td></tr></table>

Table 11: Full article-level cross-validated performance at $T _ { A } .$ , reported as mean ± standard deviation across folds. Results use $k = 4$ for HYDRONEWSFR and $k = 6$ for CLIMATENEWSFR.

<table><tr><td colspan="16">Corpus k Retained Positive Negative Baseline Linear SVC LogReg</td></tr><tr><td rowspan="8">HYDRONEWSFR</td><td colspan="2">1</td><td colspan="2">1235 544</td><td>691</td><td>0.611</td><td>0.765</td><td>0.772 0.712</td><td colspan="2">Tree Forest XGBoost 0.761</td></tr><tr><td>2</td><td>773</td><td>276</td><td>497</td><td>0.525</td><td>0.815</td><td>0.813 0.743</td><td>0.811</td><td>0.765 0.811</td></tr><tr><td>3</td><td>508</td><td>150</td><td>358</td><td>0.455</td><td>0.851</td><td>0.8410.820</td><td>0.854</td><td>0.895</td></tr><tr><td>4</td><td>337</td><td>83</td><td>254</td><td>0.395</td><td>0.889</td><td>0.882 0.835</td><td>0.885</td><td>0.912</td></tr><tr><td>5</td><td>219</td><td>39</td><td>180</td><td>0.298</td><td>0.863</td><td>0.864 0.764</td><td>0.843</td><td>0.860</td></tr><tr><td>6</td><td>145</td><td>21</td><td>124</td><td>0.248</td><td>0.878</td><td>0.838 0.894</td><td>0.937</td><td>0.910</td></tr><tr><td>7</td><td>81</td><td>13</td><td>68</td><td>0.330</td><td>0.950</td><td>0.950 0.867</td><td>0.833</td><td>0.917</td></tr><tr><td>8</td><td>40</td><td>6</td><td>34</td><td>0.258</td><td></td><td>1.000 1.000 0.700</td><td>0.900</td><td>0.800</td></tr><tr><td rowspan="8">CLIMATENEWSFR</td><td>1</td><td>1941</td><td>875</td><td>1066</td><td>0.621</td><td>0.779</td><td>0.778 0.755</td><td>0.780</td><td>0.778</td></tr><tr><td>2</td><td>1390</td><td>489</td><td>901</td><td>0.519</td><td>0.827</td><td>0.827 0.769</td><td>0.823</td><td>0.851</td></tr><tr><td>3</td><td>1005</td><td>273</td><td>732</td><td>0.427</td><td>0.873</td><td>0.887 0.843</td><td>0.875</td><td>0.895</td></tr><tr><td>4</td><td>766</td><td>182</td><td>584</td><td>0.384</td><td>0.905</td><td>0.903 0.860</td><td>0.910</td><td>0.915</td></tr><tr><td>5</td><td>570</td><td>115</td><td>455</td><td>0.336</td><td>0.913</td><td>0.913 0.927</td><td>0.931</td><td>0.952</td></tr><tr><td>6</td><td>398</td><td>68</td><td>330</td><td>0.291</td><td>0.906</td><td>0.936 0.898</td><td>0.935</td><td>0.969</td></tr><tr><td>7</td><td>270</td><td>43</td><td>227</td><td>0.273</td><td>0.920</td><td>0.915 0.903</td><td>0.929</td><td>0.940</td></tr><tr><td>8</td><td>161</td><td>21</td><td>140</td><td>0.228</td><td>0.929</td><td>0.927 0.962</td><td>0.985</td><td>0.949</td></tr></table>

Table 12: Classifier results at $T _ { A }$ across symmetric consensus thresholds $( k , k , 0 )$ . Values are cross-validated mean $F _ { 1 }$ scores.

<table><tr><td>Corpus</td><td>Comparison</td><td> $\Delta F _ { 1 }$ </td><td>p</td><td>q</td></tr><tr><td>HYDRONEWSFR</td><td>All vs. all w/o geometry</td><td>+0.526</td><td> $7 . 1 9 \times 1 0 ^ { - 4 }$ </td><td> $1 . 8 3 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>All vs. only geometry</td><td>+0.011</td><td>0.536</td><td>0.577</td></tr><tr><td></td><td>All vs. all w/o social</td><td>-0.006</td><td>0.374</td><td>0.436</td></tr><tr><td></td><td>All vs. all w/o text</td><td>+0.006</td><td>0.374</td><td>0.436</td></tr><tr><td></td><td>Only geometry vs. only text</td><td>+0.628</td><td> $3 . 6 6 \times 1 0 ^ { - 4 }$ </td><td> $1 . 1 7 \times 1 0 ^ { - 3 }$ </td></tr><tr><td></td><td>Only geometry vs. only social</td><td>+0.551</td><td> $1 . 3 9 \times 1 0 ^ { - 4 }$ </td><td> $5 . 7 4 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>CLIMATENEWSFR</td><td>All vs. all w/o geometry</td><td>+0.743</td><td> $1 . 5 7 \times 1 0 ^ { - 4 }$ </td><td> $4 . 0 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>All vs. only geometry</td><td>0.000</td><td></td><td></td></tr><tr><td></td><td>All vs. all w/o social</td><td>0.000</td><td></td><td></td></tr><tr><td></td><td>All vs. all w/o text</td><td>+0.007</td><td> $0 . 3 7 4$ </td><td> $0 . 3 7 4$ </td></tr><tr><td></td><td>Only geometry vs. only text</td><td>+0.767</td><td> $7 . 7 4 \times 1 0 ^ { - 5 }$ </td><td> $3 . 8 9 \times 1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>Only geometry vs. only social</td><td>+0.886</td><td> $3 . 6 3 \times 1 0 ^ { - 4 }$ </td><td> $6 . 9 5 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table 13: Paired fold-level $F _ { 1 }$ comparisons for the selected ablation settings. Positive $\Delta F _ { 1 }$ means that the first condition outperforms the second. Dashes indicate identical fold-level $F _ { 1 }$ values or degenerate zero differences, for which the paired t-test is not defined.

<table><tr><td>k</td><td>Retained</td><td>Positive</td><td>Negative</td><td> $F _ { 1 }$ </td><td>Recall</td></tr><tr><td>1</td><td>2134</td><td>1056</td><td>1078</td><td>0.793</td><td>0.773</td></tr><tr><td>2</td><td>1390</td><td>605</td><td>785</td><td>0.871</td><td>0.865</td></tr><tr><td>3</td><td>965</td><td>387</td><td>578</td><td>0.922</td><td>0.931</td></tr><tr><td>4</td><td>656</td><td>240</td><td>416</td><td>0.923</td><td>0.910</td></tr><tr><td>5</td><td>465</td><td>163</td><td>302</td><td>0.942</td><td>0.939</td></tr><tr><td>6</td><td>321</td><td>107</td><td>214</td><td>0.981</td><td>0.981</td></tr><tr><td>7</td><td>204</td><td>71</td><td>133</td><td>0.981</td><td>0.988</td></tr><tr><td>8</td><td>120</td><td>48</td><td>72</td><td>0.991</td><td>0.983</td></tr></table>

Table 14: XGBoost results on the extended corpus.

<table><tr><td>Corpus</td><td>Pos.</td><td> $\mathrm { N e g . }$ </td><td> $W = 5 { \mathrm { d } }$ </td><td> $W = \mathrm { 7 d }$ </td><td> $W = 1 4 \mathrm { d }$ </td><td> $W = 2 1 { \mathrm { d } }$ </td><td>Base</td></tr><tr><td>HYDRONEWSFR</td><td>42</td><td>211</td><td>0.776 [0.67, 0.86]</td><td>0.762 [0.65, 0.86]</td><td>0.780 [0.68, 0.87]</td><td>0.696 [0.60, 0.79]</td><td>0.285</td></tr><tr><td>CLIMATENEWSFR</td><td>26</td><td>316</td><td>0.821 [0.70, 0.92]</td><td>0.800 [0.66, 0.91]</td><td>0.737 [0.59, 0.85]</td><td>0.778 [0.64, 0.89]</td><td>0.141</td></tr><tr><td>HYDRONEWSFR Ext.</td><td>131</td><td>345</td><td>0.875 [0.83, 0.92]</td><td>0.885 [0.84, 0.92]</td><td>0.874 [0.83, 0.91]</td><td>0.873 [0.83, 0.91]</td><td>0.432</td></tr></table>

Table 15: Forward-chaining results at $T _ { A }$ , for test windows of W days. Ext. denotes the extended hydrogen corpus of Appendix D. Cells report pooled $F _ { 1 }$ with a 95% percentile bootstrap interval; Base is the constant-positive baseline. Counts are the pooled evaluated articles at $W = 7 { \mathrm { d } }$ and vary slightly with W under the window-fit rule.