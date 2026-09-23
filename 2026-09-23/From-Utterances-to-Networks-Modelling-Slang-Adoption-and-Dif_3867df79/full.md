# From Utterances to Networks: Modelling Slang Adoption and Diffusion Across Subreddits

Xiaoning Wang<sup>1</sup>, Ted Underwood<sup>1</sup>, Zhewei Sun<sup>2</sup>

<sup>1</sup>School of Information Sciences, University of Illinois Urbana-Champaign xw109@illinois.edu, tunder@illinois.edu

<sup>2</sup>Toyota Technological Institute at Chicago zsun@ttic.edu

## Abstract

Adoption and diffusion of neologisms in online communities have received renewed attention in recent years. As internet slang terms such as APT, referring to a K-pop song, and phrases such as Canon Event meaning an embarrassing but pivotal event, go viral online, it becomes increasingly important to understand the mechanisms that contribute to their success. Prior studies have often explained slang diffusion either from the perspective of social interaction or from the linguistic properties of the slang itself, but rarely from both perspectives together. One major obstacle has been the high cost of annotating slang usage in large-scale online communication. Recent advances in large language models (LLMs), however, make it possible to use them as scalable annotators for such tasks. In this study, we first curate a human-annotated benchmark to evaluate LLM performance in detecting slang usage in real-world Reddit communication. We then leverage LLM-based annotations to model slang adoption and diffusion. Our results show that, at the slang-diffusion level, bridging positions in the network facilitate the spread of slang, whereas greater local connectedness has the opposite effect. At the user level, both network structure and semantic context influence how quickly a user adopts slang.

## 1 Introduction

Lexical change in social networks has received growing attention in recent years, as social interaction plays a central role in both the emergence of new lexical forms and the maintenance of existing ones (Milroy and Llamas, 2013). A lexical innovation typically begins when a speaker, also known as an inventor, coins a new term or assigns a new meaning to an existing form. Through social interaction, this innovation becomes available to potential adopters, who may then take it up in their own language usages. At this stage, lexical innovations that are better suited to speakers’ communicative contexts are more likely to be taken up and reproduced. The diffusion of lexical innovations therefore depends not only on patterns of social interaction, but also on the characteristics of the innovations themselves. We therefore distinguish between network features, which capture how users encounter slang through interaction, and linguistic features, which capture properties of the slang usage contexts that may affect adoption.

![](images/65c74ad43f347fa543e57b43fe85437ed95187d80b1a686b5a799f2332bbd1cf.jpg)  
Figure 1: An example of an annotated Reddit utterance from r/asoiaf. <glossary-sense> wraps slang terms used in their sense according to the community glossary, while <non-glossary-sense> wraps term in the words other senses. For example, in this subreddit, "book!" is a slang used to refer to a character as they appear in the book series A Song of Ice and Fire (e.g. "Book!Sansa").

Lexical innovation has also been conceptualized as a diffusion process analogous to social contagion. Prior work has modeled the spread of neologisms using epidemic models such as SIR (Jiang et al., 2021), while more recent work has used network-based simulation models to characterize how lexical innovations propagate between speakers (Ananthasubramaniam et al., 2024).

In most prior studies, they were primarily focused on network features (Hamilton et al., 2017; Zhu and Jurgens, 2021; Del Tredici and Fernández, 2018) and very little on linguistic features beyond the word forms (Stewart and Eisenstein, 2018). According to theory of innovation diffusion (Rogers, 2003), semantics, in the case of lexical innovation, is one of the necessary conditions determining whether potential adopters use an innovation. Another key obstacle preventing existing studies from incorporating linguistic features is the difficulty of collecting utterances that contain slang, since slang terms commonly recycles existing word forms (Warren, 1992; Eble, 2012). Existing studies relied heavily on dictionary-based detection (Zhu and Jurgens, 2021; Del Tredici and Fernández, 2018), because identifying slang usages at scale requires either substantial human annotation or automated word sense disambiguation (WSD), which has historically fallen short of acceptable accuracy. Recent advances in large language models (LLMs) have made it feasible to use LLMs as judges to detect and disambiguate slang across large corpora of utterances (Sun et al., 2024; Wuraola et al., 2024). In this work, we provide a benchmark for evaluating LLMs’ ability to detect slang in real-world Reddit conversations. We find that Gemma-4-E4B-it achieves up to 93.44% precision in WSD, suggesting that it can serve as a reliable annotator.

By leveraging LLMs, we can incorporate both network and linguistic features into the analysis of informal lexical change on online social media. We treat slang terms on Reddit as lexical innovations and study how slang diffuses and how users adopt these terms within a community. We focus on 54 subreddits that maintained moderator-curated slang glossaries (Lucy and Bamman, 2021), a dataset that enables two complementary analyses. First, it allows us to study which network and linguistic features predict whether a slang term becomes widely adopted within a community, as evidenced by its inclusion in the glossary. Second, it allows us to identify slang terms coined in one subreddit and trace how they are subsequently adopted by members of that community. Our analysis shows that slang diffused by users with greater bridging capital is more likely to spread, while greater diversity in usage contexts may both facilitate diffusion and create hurdles for new users learning and adopting these terms.

In this paper, we make the following contributions: 1) A publicly available benchmark of 1,333 real Reddit utterances with human-annotated, subreddit-specific English slang usages across 23 subreddits, supporting the evaluation of LLMbased slang detection and WSD in naturalistic conversation. 2) An evaluation of how network and linguistic features contribute to slang dissemination and adoption with fixed effect model and survival analysis.

## 2 Data

## 2.1 Reddit Corpus

To analyze slang diffusion and adoption, we follow Zhu and Jurgens (2021) in using comments posted to Reddit and treating each subreddit as a community. We use the ConvoKit package (Chang et al., 2020) as our source of the Reddit corpus, whose dataset spans ten years from 2008 to 2018, with the majority of data concentrated after 2013.

We obtain the moderator-curated glossary from Lucy and Bamman (2021), who compiled slang lists from the corresponding moderator threads. Each entry contains the slang term, its subreddit, and a definition. Our analysis covers the 54 subreddits that are available on ConvoKit.

## 2.2 Slang Detection Dataset

To evaluate LLM performance on slang detection in real Reddit comments, we curated a benchmark of 1,333 human-annotated comments across 23 subreddits. Unlike prior datasets that are based on subtitles rather than naturally occurring subreddit conversations (Sun et al., 2024), focus primarily on binary slang/non-slang classification (Aloraini, 2025), or are designed for related tasks such as slang comprehension (Mei et al., 2024) and sentiment analysis (Wuraola et al., 2024), our benchmark evaluates model performance on real utterances from subreddit conversations. In this setting, a single input may contain multiple slang candidates, each of which may be used either in its sense in glossary or not. Figure 1 shows one annotated example. Each entry is accompanied by metadata, including the source subreddit, the corresponding subreddit glossary, and the surrounding conversational context of the comment.

In practical use, we aim to provide real Reddit conversations directly to an LLM and ask it to identify slang usages without additional preprocessing. However, slang detection is challenging not only because slang often reuses existing word forms, requiring the model to distinguish slang senses from literal senses, but also because processing long conversational contexts can be difficult for LLM-based annotation. In contrast, newly coined slang forms are comparatively straightforward to detect. Therefore, we decompose slang detection into WSD and term retrieval to evaluate model performance more diagnostically. In Single-WSD, the model is given a Reddit comment and one target term and decides whether the term is used in its slang sense. This is the simplest setting because the target term is already provided. In List-WSD, the model is given a comment and a list of candidate terms and performs WSD for each term.

<table><tr><td>Model</td><td>Task</td><td>Precision</td><td>Recall</td><td>F1</td><td>N</td></tr><tr><td rowspan="2">Llama-3.1-8B-Instruct</td><td>Single-WSD</td><td>85.11</td><td>88.03</td><td>86.55</td><td>1,853</td></tr><tr><td>List-WSD</td><td>62.07</td><td>67.27</td><td>63.19</td><td>1,333</td></tr><tr><td>Ministral-3-8B-Instruct-2512</td><td>Single-WSD List-WSD</td><td>93.05 83.84</td><td>87.83 81.79</td><td>90.36 82.10</td><td>1,853 1,333</td></tr><tr><td>OLMo-3-7B-Instruct</td><td>Single-WSD</td><td>80.71</td><td>87.10</td><td>83.78</td><td>1,853</td></tr><tr><td></td><td>List-WSD</td><td>73.67</td><td>70.26</td><td>70.59</td><td>1,333</td></tr><tr><td rowspan="2">Gemma-4-E4B-it</td><td>Single-WSD</td><td>93.44</td><td>93.44</td><td>93.44</td><td>1,853</td></tr><tr><td>List-WSD</td><td>84.59</td><td>84.71</td><td>83.98</td><td>1,333</td></tr></table>

Table 1: Benchmark results for slang detection on the Single-WSD and List-WSD tasks. For Single-WSD, the model is given a Reddit comment, a target term, and its corresponding glossary definition, and determines whether the term is used in its glossary sense. For List-WSD, the model is given a Reddit comment and the full slang glossary and identifies which glossary terms are used in the comment. The best result for each task and metric is shown in bold.

## 3 Methodology

To examine how social-network structure and pragmatic language use shape the diffusion of slang across online communities, this study models the rate at which users adopt new slang as a function of both network and linguistic features. This section describes the features used in the analysis, how they are calculated, and how they capture the social and linguistic mechanisms underlying observed adoption behaviour.

## 3.1 Social Network Features

Prior work suggests that network centrality captures different forms of social capital relevant to language diffusion. Degree centrality has been used as a proxy for bonding capital, since it reflects how embedded a user is in their local interaction network (Shin, 2021). This aligns with sociolinguistic evidence from the Belfast Study, which found that dense social ties play a central role in dialect maintenance (Milroy and Llamas, 2013). The degree centrality $C _ { D }$ for user i is

$$
C _ { D } ( i ) = \frac { k _ { i } } { n - 1 }
$$

where $k _ { i }$ is the degree of node i and n is the total number of nodes in the network.

In contrast, weak ties are typically associated with linguistic change rather than maintenance. Sociolinguistic work has argued that users who maintain ties to everyday acquaintances are more likely to facilitate the spread of linguistic innovations across social groups (Milroy and Llamas, 2013). In network terms, these users possess bridging capital: they connect otherwise separated parts of the social structure and allow linguistic forms to move beyond tightly bonded local clusters. To capture this mechanism, we include Betweenness Centrality as a feature representing the extent to which slang users occupy bridging positions (Shin, 2021). Betweenness centrality measures how often a node sits on the shortest path between other pairs of nodes:

$$
C _ { B } ( i ) = \sum _ { s \neq i \neq t } \frac { \sigma _ { s t } ( i ) } { \sigma _ { s t } }
$$

where $\sigma _ { s t }$ is the total number of shortest paths from s to $t ,$ and $\sigma _ { s t } ( i )$ is the number of those paths that pass through i. Because raw betweenness centrality depends on network size—that is, as the number of nodes increases, the number of node pairs whose shortest paths could pass through a focal node also increases—we normalize each node’s raw betweenness by the maximum possible betweenness in a network of the same size. Specifically, normalized betweenness centrality is calculated as

$$
C _ { B } ^ { \prime } ( i ) = \frac { C _ { B } ( i ) } { N _ { m a x } }
$$

where

$$
N _ { m a x } = \frac { ( N - 1 ) ( N - 2 ) } { 2 }
$$

## 3.2 Linguistic Features

Prior work has shown that the linguistic context in which a word is used can influence its likelihood of successful adoption (Stewart and Eisenstein, 2018; Altmann et al., 2011). Retrieving slang terms is challenging because slang WSD often involves distinguishing a slang sense from an existing sense that shares the same word form. Recent advances in LLMs allow us to perform this WSD task with high accuracy; see Section 4.1 for our model evaluation. We leveraged LLMs to identify which glossary terms appear in each utterance and determine whether they are used with the slang senses defined in the glossary. In total, we identified 3, 965 slang terms across 54 subreddits.

After slang detection, we calculate the linguistic features for each slang term $W _ { i }$ . We use Sentence-BERT (Reimers and Gurevych, 2019) to embed the N utterances $U _ { W _ { i , 1 } } , U _ { W _ { i , 2 } } , \dots , U _ { W _ { i , n } }$ containing $W _ { i }$ , obtaining the corresponding embeddings $E _ { W _ { i , 1 } } , E _ { W _ { i , 2 } } , . . . , E _ { W _ { i , n } }$ . Based on these embeddings, we calculate two semantic features: Semantic Dispersion and Effective Topics.

Semantic Dispersion. We define the Semantic Dispersion as how flexibly a slang term has been adopted across contexts. The SD is calculated for each slang $W _ { i }$ by the mean pairwise cosine distance, formally:

$$
S D ( W _ { i } ) = \frac { 2 } { N ( N - 1 ) } \sum _ { i < j } 1 - \cos ( E _ { i _ { j } } , E _ { i _ { k } } )
$$

where N is the number of embeddings and cos(·) is the cosine similarity.

Effective Topics. Drawing the inspiration from Giulianelli et al. (2020), we define our Effective Topics to measure how many clusters of topics that a word appeared in. We take embeddings $\{ E _ { W _ { i } , 1 } , E _ { W _ { i } , 2 } . . . , E _ { W _ { i } , m } \}$ of word $W _ { i }$ and run the K-means algorithm to cluster the sense embeddings. The optimal K is determined by computing the silhouette score to judge how good the embedding fits its assigned cluster. We then calculate the Shannon Entropy H for a word $W _ { i }$ to measure how uncertain on which cluster any of the embeddings will belong to. Mathematically,

$$
H ( W _ { i } ) = - \sum _ { k = 1 } ^ { K } P _ { k } \mathrm { l o g } _ { 2 } P _ { k }
$$

where $\begin{array} { r } { P _ { k } = \frac { n _ { k } } { N } } \end{array}$ is the empirical proportion of usages of word $W _ { i }$ assigned to cluster $k ,$ , with $n _ { k }$ denotes the number of embeddings falling into cluster k. To obtain a more interpretable quantity, we exponentiate the entropy to recover the effective number of topics:

$$
\operatorname { E T } ( W _ { i } ) = 2 ^ { H ( W _ { i } ) }
$$

## 4 Experiments

## 4.1 LLM Detection

The first step of our analysis is to evaluate LLMs performance on slang detection. Slang detection is challenging for LLMs because they must infer the intended meaning of a term from its surrounding context, particularly when slang reuses existing word forms with different meanings, while also processing potentially long conversational contexts. To better understand the bottlenecks in LLM-based slang detection, we therefore decompose the task into two diagnostic settings:

Single-WSD. Given a Reddit utterance, one candidate slang term, and its glossary definition, the model decides whether the term is used with the glossary-defined slang sense. This setting isolates word sense disambiguation because the target term is already specified.

List-WSD. Given a Reddit utterance and the full slang glossary for the subreddit, the model determines which glossary terms are used with their slang meanings. This setting evaluates performance when the model must search over multiple candidate terms while also disambiguating their senses. To ensure a fair WSD setting, for both tasks, we also provide the model with the source subreddit and the conversational context of the utterance, defined as its parent and grandparent comments where available.

Batch Inference. During LLM-based slang detection, we used vLLM (Kwon et al., 2023) for batched inference on a single GH200 GPU, with the temperature set to 0 and top\_p set to 1. To reduce computational cost, we first used string matching to identify utterances containing candidate glossary terms. For each candidate utterance, we provided the model with only the corresponding subset of glossary entries, together with the surrounding context, and asked it to determine whether each candidate term was used with the slang sense defined in the glossary. This pre-filtering step was designed solely to improve inference efficiency.

![](images/068ef1168006771988b5b25785e54f30bd6cfe623a0fce76759eb39fef729d71.jpg)  
Figure 2: Diagram for fixed effects identification. Dashed gray arrows mark backdoors closed by subreddit, slang, and time fixed effects.

With this setup, processing 10,000 samples took approximately 4 minutes.

After LLM inference, we performed a separate post-hoc cleaning step to improve detection precision. Because our analysis focuses only on terms included in the glossary, and LLMs may hallucinate terms that do not appear in the original Reddit comment (Yang et al., 2026), we used string matching to remove detected terms that were either absent from the glossary or not present in the original utterance. This post-processing step therefore reduces false positives in the final slang detections.

Results. We evaluate four open-source language models: Llama-3.1-8B-Instruct (Grattafiori et al., 2024), Olmo-3-7B-Instruct (Olmo et al., 2026), Ministral-3-8B-Instruct (Liu et al., 2026), and Gemma-4-E4B-it (Team et al., 2026), on our human-annotated Reddit utterances. We restrict our evaluation to small, open-source models because our downstream analysis targets roughly 100 million utterances, at which scale inference cost dominates model choice.

As shown in Table 1, Gemma-4-E4B-it performs best on both tasks, achieving 93.44% F1 on Single-WSD and 83.98% F1 on List-WSD. The main bottleneck therefore appears not to be slang WSD itself, but rather processing long contexts while reliably following output instructions. List-WSD is particularly challenging because the prompt includes the glossary and surrounding context. Under this longer context, models often generate extra commentary, which contaminates outputs and increases inference time.

Based on these results, we adopt Gemma-4-E4Bit for slang detection in the remainder of the analysis.

## 4.2 Slang Level Analysis

Having established that LLM-based slang detection is sufficiently reliable, we use the LLM-detected data to compute both network and linguistic features and investigate how they predict the adoption rate. Following prior work on temporal network analysis (Zhu and Jurgens, 2021; Hamilton et al., 2017), we partition each subreddit’s continuous interaction stream into monthly windows. For each subreddit $S _ { i }$ with data spanning $T _ { i }$ months, we obtain a sequence of monthly interaction networks $\{ I _ { S _ { i } , 1 } , I _ { S _ { i } , 2 } , . . . , I _ { S _ { i } , T _ { i } } \}$ . We operationalize an interaction between two users as occurring when the two users are in close proximity, separated by at most two comments (Hamilton et al., 2017; Zhu and Jurgens, 2021).

For each window $I _ { S _ { i } , t } ,$ we compute both network and linguistic features and use them to predict the number of new adopters in the following window $I _ { S _ { i } , t + 1 }$ . This lagged design lets us probe how well current-window features explain future adoption, rather than merely correlating with concurrent usage. Rather than aggregating across all slang terms within a subreddit, we model adoption at the slang-term level to preserve term-level variation in the data. For each window $I _ { S _ { i } , t } ,$ let $\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { n _ { i , t } } \}$ denote the set of slang terms observed in that window. For each slang term $s _ { k }$ , the response variable $y _ { i , k , t }$ is defined as the number of new adopters of $s _ { k }$ in $I _ { S _ { i } , t + 1 }$

Network features. For each slang term $s _ { k }$ in window $I _ { S _ { i } , t }$ , let $\mathcal { U } _ { i , k , t }$ denote the set of users who used $s _ { k }$ during that window, with $| \mathcal { U } _ { i , k , t } | = M _ { i , k , t }$ . For each user u $\in \mathcal { U } _ { i , k , t }$ , we compute two centrality measures on $I _ { S _ { i } , t } \colon$ degree centrality, and betweenness centrality in the complete subreddit interaction network $I _ { S _ { i } , t }$ . We then average each measure across the $M _ { i , k , t }$ adopters to obtain a slang-level network feature vector $\mathbf { x } _ { i , k , t } ^ { \mathrm { n e t } }$

$$
\mathbf { x } _ { i , k , t } ^ { \mathrm { n e t } } = \frac { 1 } { M _ { i , k , t } } \sum _ { u \in \mathcal { U } _ { i , k , t } } \mathbf { c } ( u ; I _ { S _ { i } , t } ) ,
$$

where $\mathbf { c } ( u ; I _ { S _ { i } , t } )$ stacks the two centrality scores for user u in network $I _ { S _ { i } , t }$

Linguistic features. To compute linguistic features for $s _ { k }$ at window t, we sample 100 utterances containing $s _ { k }$ from $I _ { S _ { i } , t }$ and encode each utterance with SentenceBERT. From the resulting 100 embeddings, we compute Semantic Dispersion and Effective Topics as described in Section 3.2, yielding a two-dimensional linguistic feature vector $\mathbf { x } _ { i , k , t } ^ { \mathrm { { l i n g } } } .$

Combined covariates. Concatenating network and linguistic features gives the full covariate vector for each (subreddit, slang, window) triple:

$$
\mathbf { X } _ { i , k , t } = \left[ \mathbf { x } _ { i , k , t } ^ { \mathrm { n e t } } \parallel \mathbf { x } _ { i , k , t } ^ { \mathrm { l i n g } } \right] ,
$$

which is used to predict $y _ { i , k , t } .$

Model. Prior work has examined how network features and linguistic features each predict slang diffusion in isolation (Zhu and Jurgens, 2021). They explicitly note that external surges in adoption are essentially undetectable from the interaction data alone, yet may substantially shape adoption dynamics.

For example, in r/asoiaf (the subreddit devoted to George R. R. Martin’s A Song of Ice and Fire), new slang tends to emerge whenever a new book is released: the term ASOS, for instance, was coined and spread rapidly following the publication of A Storm of Swords. A different pattern played out in r/wallstreetbets during the 2021 GameStop short squeeze, when a massive influx of new users coincided with a sharp acceleration in the subreddit’s slang adoption rate. In both cases, an external event simultaneously shifted the network structure, the linguistic environment, and the rate of slang adoption, making it impossible to attribute adoption to any single feature without accounting for the underlying event.

To minimize the external effects, Figure 2 summarizes the sources of confounding we account for. We posit that three classes of factors jointly drive both the features and adoption:

• Subreddit factors: community culture, topic focus, moderation style, and stable demographic composition. Figure 4 illustrates how an exogenous event can affect slang usage.

• Slang factors: properties intrinsic to the slang term itself, including length, phonology, semantic class, and age. Figure 6 shows that slang terms with more common, everyday meanings may be used more frequently than other terms, such as SUV compared with cam.

• Period factors: Reddit-wide trends, platform changes, and viral slang moments that affect all communities simultaneously. Figure 7 shows that when a new game mode was released, usage of the corresponding slang term increased sharply and then stabilized. Another source of variation is the overall growth in subreddit activity showed in Figure 8, which can indirectly increase the observed frequency of slang usage.

• Subreddit × Period interactions: Subredditspecific events at particular moments, such as community drama, moderation changes, or sudden influxes of users. Figure 5 shows that r/AFL exhibits seasonal patterns in both overall utterance volume and slang usage across the year.

We estimate a fixed-effects negative binomial count model with a log link, where the outcome is the number of new adopters of a slang term in the following month. We include the log number of users at risk of adoption as an offset and compute cluster-robust standard errors at the subreddit level.

Controlling for confounding with fixed effects. To account for these shared drivers, we include fixed effects for subreddit, time period, subredditperiod interaction, and slang term. These fixed effects control for systematic differences across communities, periods, and individual slang terms.

<table><tr><td>Feature</td><td>Estimate</td><td>Std. Error</td><td>z value</td><td> $p$  value</td></tr><tr><td>Mean Betweenness</td><td> $0 . 1 0 3 ^ { * }$ </td><td>0.052</td><td>1.989</td><td>0.047</td></tr><tr><td>Mean Degree</td><td> $- 0 . 5 0 6 ^ { \ast \ast }$ </td><td>0.137</td><td>-3.704</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Effective Topics</td><td> $- 0 . 0 1 8 ^ { \ast \ast \ast }$ </td><td>0.003</td><td>-5.804</td><td> $< 0 . 0 0 1$ </td></tr><tr><td>Semantic Dispersion</td><td> $- 0 . 2 1 5 ^ { \ast \ast \ast }$ </td><td>0.053</td><td>-4.089</td><td> $< 0 . 0 0 1$ </td></tr></table>

Table 2: Regression Coefficients for Predicting Slang Adoption. Statistical significance is denoted by ${ ^ { * } p } < 0 . 0 5 ,$ $^ { * * } p < 0 . 0 1$ , and $^ { * * * } p < 0 . 0 0 1$

For example, when examining slang used in r/wow in March 2026, the subreddit and time are held fixed, while slang-term fixed effects account for each term’s baseline tendency to be adopted. The model then asks whether differences in Effective Topics are associated with differences in subsequent adoption, rather than attributing those differences to the community, time period, or slang term itself. In causal-inference terms, these fixed effects help block back-door paths from shared contextual factors to slang adoption (Huntington-Klein, 2021).

Results. Table 2 shows the results of fitting our fixed-effects model. Looking at the covariate estimates, betweenness centrality shows a positive effect on new adoption $( \beta = 0 . 1 0 3 , S E =$ $0 . 0 5 2 , z \ = \ 1 . 9 8 , p \ < \ 0 . 0 5 )$ , while degree centrality shows the largest negative effect $( \beta { \mathbf { \alpha } } = { \mathbf { \beta } }$ $- 0 . 5 0 6 , S E = 0 . 1 3 7 , z \ = \ - 3 . 7 0 4 , p \ < \ 0 . 0 0 1 )$ Both effective topics $( \beta ~ = ~ - 0 . 0 1 8 , S E ~ =$ $0 . 0 0 3 , z ~ = ~ - 5 . 8 0 4 , p ~ < ~ 0 . 0 0 1 )$ and semantic dispersion $( \beta ~ = ~ - 0 . 2 1 5 , S E ~ = ~ 0 . 0 5 3 , z ~ =$ $- 4 . 0 8 9 , p = 0 . 0 0 1 )$ show negative effects.

We also assessed robustness to LLM classification error by using sensitivity and specificity, adjusted for observed prevalence, to Monte Carlosimulate corrected adoption outcomes and refit the model. Table 5 showed all coefficients retained their signs and significance across simulations.

The network features therefore appear to play an important role in slang adoption than the linguistic features. Following the interpretation introduced earlier, we treat betweenness centrality as a proxy for bridging positions across otherwise weakly connected parts of the network, and degree centrality as a proxy for local connectedness and embeddedness. The positive coefficient on betweenness is consistent with Milroy and Milroy’s account of the role of weak ties in linguistic diffusion (Milroy and Llamas, 2013): when adopters have higher average betweenness, they are more likely to occupy bridging positions that connect different parts of the network, creating pathways through which slang can spread beyond local clusters. Degree centrality shows the opposite pattern. When adopters have higher average degree and are more densely connected within their local network, subsequent slang adoption decreases. Together, these results suggest that slang diffusion may benefit more from connections that bridge otherwise separated parts of a community than from dense local connectivity alone.

The linguistic features tell a more complex story. Slang used across more diverse contexts or topics is associated with fewer new adopters, possibly because such variation makes the slang harder for potential adopters to understand and use confidently. To examine whether this effect persists over time, we fit the same model to predict new adopters at $t + n .$ . As shown in Figure 3, the negative effect of semantic dispersion gradually approaches zero across longer horizons, while the centrality effects remain relatively stable. A Wald test confirms that only semantic dispersion changes significantly from $t + 1 \mathrm { t o } t + 3 0$ . This suggests that contextual diversity may create a temporary barrier to slang adoption that gradually fades over time.

## 4.3 User Level Analysis

In the Slang Level Analysis, we examined whether the social structure of slang adopters and slang semantics can predict the number of new adopters in a future time window. We found that network structure and slang semantics are strongly associated with future adoption. From the other side of the diffusion process, one question is whether the same set of features predicts when users will adopt a slang term? In this part of analysis, we used Cox Proportional Hazards (CPH) model (Bangdiwala, 1989) to incorporate both network and linguistic features to predict the time from a user’s first encounter with a slang term to their first use of that term.

Data. In this experiment, we shift to user-level analysis of slang adoption. For each subreddit k, slang term $s _ { k } .$ , and time window $I _ { S _ { k } , t }$ , let $\{ u _ { 1 } , u _ { 2 } , \ldots , u _ { n } \}$ denote the users who posted at least one utterance in $I _ { S _ { k } , t } .$ . For each user, we record two timestamps: t<sub>encounter</sub>, the time at which the user first encountered $s _ { k }$ , and $t _ { \mathrm { a d o p t } }$ , the time at which the same user first used $s _ { k }$ themselves.

![](images/121c0290d97ec490bda3d12ed125959bc6318324b8ece09761bacd786e993dcf.jpg)

Figure 3: Shows how the estimated effects of the four covariates change across forecast horizons. Betweenness centrality remains consistently positive, while degree centrality remains negative across all horizons. In contrast, the negative effects of effective topics and semantic dispersion gradually attenuate toward zero as the forecast horizon increases.
<table><tr><td>Variable</td><td> $\beta$ </td><td>HR</td><td>95% CI</td><td>p-value</td></tr><tr><td>Degree Centrality</td><td>0.190</td><td>1.209</td><td>[1.131,1.292]</td><td>&lt; .001</td></tr><tr><td>Betweenness Centrality</td><td>-0.123</td><td>0.884</td><td>[0.793, 0.986]</td><td>.027</td></tr><tr><td>Betweenness Centrality × log(t)</td><td>-0.055</td><td>0.946</td><td>[0.880, 1.018]</td><td>.137</td></tr><tr><td>Degree Centrality × log(t)</td><td>-0.012</td><td>0.989</td><td>[0.951, 1.027]</td><td>.559</td></tr><tr><td>Semantic Dispersion</td><td>0.057</td><td>1.059</td><td>[1.020, 1.099]</td><td>.003</td></tr><tr><td>Effective Topics</td><td>-0.077</td><td>0.926</td><td>[0.890, 0.963]</td><td>&lt; .001</td></tr></table>

Table 3: Cox proportional hazards model results. Time-varying effects are modeled as interactions with log(t).

We use the same definition of interaction as in Section 4.2. $u _ { i }$ is said to encounter $s _ { k }$ at time t if $u _ { i }$ posts a comment at time t that is separated by at most two comments to a comment c that contains $s _ { k } . \mathrm { A }$ user is considered an adopter of $s _ { k }$ if they subsequently post a comment containing s<sub>k</sub> denoted as $t _ { \mathrm { a d o p t } }$ . Users who encounter $s _ { k }$ but never post it themselves are treated as right-censored in the survival analysis. The resulting dataset contains tuples of the form $\{ u _ { i } , k , s _ { k } , t _ { \mathrm { e n c o u n t e r } } , t _ { \mathrm { a d o p t } } \}$ , with $t _ { \mathrm { a d o p t } } = \infty$ for censored users.

For predictive features, we reuse the set studied in Section 3: degree centrality and betweenness centrality as network features, and semantic dispersion and effective topics as linguistic features. All features are computed for user $u _ { i }$ at each time point from t<sub>encounter</sub> until $t _ { \mathrm { a d o p t } }$ , or through the end of the observation period if the user is right-censored.

Model. We use a Cox model allowing nonproportional hazards (Harrell, 2001) to estimate how network and linguistic features affect the timing of slang adoption. Given a set of users at risk of adoption, the model estimates the hazard function:

$$
h ( t \mid X ) = h _ { 0 , \operatorname { g r o u p } } ( t ) \exp ( { \beta ( t ) ^ { \top } X } ) ,
$$

where h(t | X) is the instantaneous adoption hazard for a user with covariate vector X at time t and $\beta$ is the vector of potentially time-varying loghazard coefficients for the covariates. We report exponentiated coefficients as hazard ratios: a ratio above 1 indicates that the covariate accelerates adoption, while a ratio below 1 indicates that it suppresses adoption. The model assumes that the effect of each covariate is constant over time but Schoenfeld-residual diagnostics indicated potential non-proportionality for degree and betweenness centrality; we therefore included interactions with log(t) for these covariates.

Results. The Cox proportional hazards results show a nuanced picture of individual slang adoption. Degree centrality is associated with faster slang adoption, whereas betweenness centrality is associated with slower adoption. Users with higher degree Centrality are more strongly embedded in the community and may therefore have more opportunities to encounter and learn its conventional language, making them faster to adopt a slang term once it begins circulating.

Betweenness centrality provides an important distinction between the ability to transmit information and the tendency to adopt it. Users with high betweenness occupy brokerage positions connecting otherwise separated parts of the network, which may make them particularly important for spreading a slang term once they have adopted it. However, occupying such a bridge position does not inherently imply a greater propensity to accept new linguistic conventions. Indeed, our Cox model shows the opposite association: users with higher betweenness adopt more slowly. One possible explanation is that brokers participate across more heterogeneous conversational environments and are consequently less embedded in the linguistic norms of any single local community. Thus, the network positions that are advantageous for carrying a term across communities need not be the same positions that facilitate rapid adoption. Exposure, adoption, and subsequent transmission may therefore represent related but distinct stages of the diffusion process.

The linguistic features reveal a similarly nuanced pattern. Semantic Dispersion is associated with faster individual adoption, whereas Effective Topics is associated with slower adoption. Although both features capture aspects of contextual breadth, they may characterize different forms of contextual variation. Greater semantic dispersion may provide users with varied evidence about how a slang term can be used, supporting the development of a more flexible representation of its meaning. In contrast, a term distributed across many distinct topical contexts may provide more fragmented or less coherent evidence, making it harder for a new user to infer a stable meaning and confidently adopt the term.

This distinction is broadly consistent with mixed findings in the word-learning literature. Contextual diversity has been associated with more efficient lexical processing and earlier vocabulary acquisition (Adelman et al., 2006; Jiménez and Hills, 2022), while recent experimental work has shown that encountering a novel word across multiple disconnected narrative contexts can disrupt the earliest stages of meaning learning (Hulme et al., 2023). Mak et al. (2021) reconcile these findings through an anchoring account: encountering a word initially in a restricted and coherent context can establish a stable representation, after which experience across more diverse contexts may enrich that representation. Our findings suggest a related distinction in slang adoption: contextual versatility may facilitate learning when different usages can be integrated into a coherent representation, whereas fragmentation across many topical environments may impede adoption.

## 5 Conclusion

In this study, we introduced a new benchmark for evaluating slang detection in real-world online social media text. We then used LLM-based slang detection to enable more fine-grained analyses of slang diffusion than was previously feasible. Building on this capability, we presented two analyses of slang use on Reddit. By combining social network and linguistic features, we found that slang diffusion is supported by adopters who occupy bridging positions in the network, whereas greater bonding capital among existing adopters is negatively associated with subsequent diffusion. At the individual level, however, users with greater degree centrality adopt slang more quickly, while users occupying high-betweenness brokerage positions adopt more slowly, suggesting that network positions that facilitate transmission are not necessarily the same as those that facilitate adoption.

Our results also show that contextual diversity is not a unitary property of slang diffusion. Greater semantic dispersion is associated with faster individual adoption, whereas usage across a larger number of distinct topics is associated with slower adoption. Together, these findings suggest that varied semantic usage may provide useful evidence for learning how a slang term can be used, while fragmentation across distinct topical contexts may make its meaning more difficult to establish. More broadly, our findings highlight that both the social structure through which slang travels and the linguistic contexts in which it appears shape how slang spreads through online communities.

## Limitations

Although we use fixed-effects models to block several backdoor paths and further analyze adoption with Cox models, we are still unable to make causal interpretations from this observational data. Important confounders may remain uncontrolled, such as users’ interests, activity levels, and prior familiarity with the slang terms.

Our interpretation of network features as forms of bonding and bridging social capital is also necessarily approximate. Prior work has offered different interpretations of bonding and bridging capital (Shin, 2021), and these concepts do not map one-toone onto any single network statistic. For example, dense local connectivity, degree centrality, PageRank, and betweenness centrality may each capture different aspects of social embeddedness or brokerage.

We used LLM-detected slang usages in our analysis, so classification errors introduced by the LLM may propagate to our downstream results. We therefore conducted a Monte Carlo sensitivity analysis (Table 5) to evaluate the robustness of our findings to plausible detection errors. We used the observed prevalence together with the estimated sensitivity and precision to derive prevalence-adjusted misclassification rates for the simulation. However, due to limited manually annotated data, we assume that the estimated detection performance of the LLM is transferable across subreddits. This assumption may not hold if slang usage or detection difficulty systematically differs across communities. As LLM-based slang detection continues to improve, we expect this source of measurement error to become less consequential in future work.

Following prior work such as Zhu and Jurgens (2021), our definition of adoption captures observable production rather than learning or comprehension. We count a user as an adopter only when they use the term in the relevant sense, which allows us to avoid treating mere exposure as adoption. However, users may learn or understand a term without producing it themselves. This distinction is important because contextual diversity may affect comprehension and production differently. For example, prior work suggests that encountering a word in more diverse contexts may not always facilitate early learning and may even be detrimental during the initial anchoring phase (Li et al., 2024). Our results therefore speak specifically to the production of subreddit-specific lexical items, rather than to lexical learning more broadly.

Finally, our analysis is also limited to slang terms included in the glossary, which predominantly contains slang that became sufficiently established to be documented. As a result, our findings primarily characterize the diffusion of relatively successful slang terms rather than slang innovations in general. Slang terms that emerged but failed to spread widely are likely underrepresented or absent from our data, so our conclusions should not be generalized to all newly coined slang.

## Ethics Statement

Risks. This research is based on data-driven analysis of publicly available online data. The study does not involve human participants and is not intended to introduce direct harm.

Offensive Content. We analyzed Reddit data that may contain explicit or offensive language. Because the released dataset contains naturally occurring Reddit conversations, it may also include such content. Researchers using the dataset should therefore be aware that it may contain both explicit language and potentially identifiable information originating from public Reddit posts.

Licenses. We use ConvoKit (Chang et al., 2020) for Reddit data under the MIT License. We also use vLLM (Kwon et al., 2023), an open-source library for LLM inference released under the Apache License 2.0. We use both ConvoKit and vLLM only for research purposes and in accordance with their respective licenses.

AI Assistance. We used an AI assistant for grammar checking and language polishing.

## Acknowledgments

This work used Delta and DeltaAI at the National Center for Supercomputing Applications (NCSA) through allocation HUM240002, CIS260828 from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program (Boerner et al., 2023), which is supported by U.S. National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296. We also used computational resources provided by the Beehive cluster at Toyota Technological Institute at Chicago.

## References

James S Adelman, Gordon D A Brown, and José F Quesada. 2006. Contextual diversity, not word frequency, determines word-naming and lexical decision times. Psychol. Sci., 17(9):814–823.

Afnan Aloraini. 2025. SlangTrack (ST) dataset.

Eduardo G Altmann, Janet B Pierrehumbert, and Adilson E Motter. 2011. Niche as a determinant of word fate in online groups. PLoS One, 6(5):e19009.

A Ananthasubramaniam, D Jurgens, and D M Romero. 2024. Networks and identity drive the spatial diffusion of linguistic innovation in urban and rural areas. Npj Complexity, 1(1).

Shrikant I. Bangdiwala. 1989. The wald statistic in proportional hazards hypothesis testing. Biometrical Journal, 31(2):203–211.

Timothy J. Boerner, Stephen Deems, Thomas R. Furlani, Shelley L. Knuth, and John Towns. 2023. Access: Advancing innovation: Nsf’s advanced cyberinfrastructure coordination ecosystem: Services & support. In Practice and Experience in Advanced Research Computing 2023: Computingfor the Common Good, PEARC ’23, page 173–176, New York, NY, USA. Association for Computing Machinery.

J P Chang, C Chiam, L Fu, A Wang, J Zhang, and C Danescu-Niculescu-Mizil. 2020. ConvoKit: A toolkit for the analysis of conversations. In Proceedings ofthe 21th Annual Meeting ofthe Special Interest Group on Discourse and Dialogue, pages 57–60.

M Del Tredici and R Fernández. 2018. The road to success: Assessing the fate of linguistic innovations in online communities. In Proceedings ofthe 27th International Conference on Computational Linguistics, pages 1591–1603.

Connie C. Eble. 2012. Slang & Sociability: In-group Language among College Students. University of North Carolina Press, Chapel Hill, NC.

Mario Giulianelli, Marco Del Tredici, and Raquel Fernández. 2020. Analysing lexical semantic change with contextualised word representations. In Proceedings of the 58th Annual Meeting of the Associationfor Computational Linguistics, pages 3960– 3973, Online. Association for Computational Linguistics.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

W L Hamilton, J Zhang, C Danescu-Niculescu-Mizil, D Jurafsky, and J Leskovec. 2017. Loyalty in online communities. In International AAAI Conference on Weblogs and Social Media. International AAAI Conference on Weblogs and Social Media, pages 540– 543.

Frank E Harrell, Jr. 2001. Cox proportional hazards regression model. In Regression Modeling Strategies, Springer Series in Statistics, pages 465–507. Springer New York, New York, NY.

Rachael C Hulme, Anisha Begum, Kate Nation, and Jennifer M Rodd. 2023. Diversity of narrative context disrupts the early stage of learning the meanings of novel words. Psychon. Bull. Rev., 30(6):2338–2350.

Nick Huntington-Klein. 2021. The effect. Taylor & Francis, London, England.

Menghan Jiang, Xiang Ying Shen, Kathleen Ahrens, and Chu-Ren Huang. 2021. Neologisms are epidemic: Modeling the life cycle of neologisms in china 2008-2016. PLoS One, 16(2):e0245984.

Eva Jiménez and Thomas T Hills. 2022. Semantic maturation during the comprehension-expression gap in late and typical talkers. Child Dev., 93(6):1727– 1743.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles.

Jiayin Li, Louise Wong, Catarina Rodrigues, Rachael C Hulme, Holly Joseph, Fiona E Kyle, and J S H Taylor. 2024. Contextual diversity and anchoring: Null effects on learning word forms and opposing effects on learning word meanings. Q. J. Exp. Psychol. (Hove), 77(11):2180–2198.

Alexander H. Liu, Kartik Khandelwal, Sandeep Subramanian, Victor Jouault, Abhinav Rastogi, Adrien Sadé, Alan Jeffares, Albert Jiang, Alexandre Cahill, Alexandre Gavaudan, Alexandre Sablayrolles, Amélie Héliou, Amos You, Andy Ehrenberg, Andy Lo, Anton Eliseev, Antonia Calvi, Avinash Sooriyarachchi, Baptiste Bout, and 101 others. 2026. Ministral 3. Preprint, arXiv:2601.08584.

L Lucy and D Bamman. 2021. Characterizing english variation across social media communities with BERT. Transactions ofthe Associationfor Computational Linguistics, 9:538–556.

Matthew H C Mak, Yaling Hsiao, and Kate Nation. 2021. Anchoring and contextual variation in the early stages of incidental word learning during reading. J. Mem. Lang., 118(104203):104203.

Lingrui Mei, Shenghua Liu, Yiwei Wang, Baolong Bi, and Xueqi Cheng. 2024. SLANG: New concept comprehension of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 12558–12575, Miami, Florida, USA. Association for Computational Linguistics.

Lesley Milroy and Carmen Llamas. 2013. Social Networks, chapter 19. John Wiley & Sons, Ltd.

Team Olmo, :, Allyson Ettinger, Amanda Bertsch, Bailey Kuehl, David Graham, David Heineman, Dirk Groeneveld, Faeze Brahman, Finbarr Timbers, Hamish Ivison, Jacob Morrison, Jake Poznanski, Kyle Lo, Luca Soldaini, Matt Jordan, Mayee Chen, Michael Noukhovitch, Nathan Lambert, and 50 others. 2026. Olmo 3. Preprint, arXiv:2512.13961.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics.

E M Rogers. 2003. The diffusion of innovations. The Free Press, New York.

B Shin. 2021. Exploring network measures of social capital: Toward more relational measurement. Journal ofPlanning Literature, 36(3):328–344.

Ian Stewart and Jacob Eisenstein. 2018. Making “fetch” happen: The influence of social and linguistic context on nonstandard word growth and decline. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, pages 4360–4370, Brussels, Belgium. Association for Computational Linguistics.

Zhewei Sun, Qian Hu, Rahul Gupta, Richard Zemel, and Yang Xu. 2024. Toward informal language processing: Knowledge of slang in large language models. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1683–1701, Mexico City, Mexico. Association for Computational Linguistics.

Gemma Team, Sherif El Abd, Vaibhav Aggarwal, Robin Algayres, Alek Andreev, Olivier Bachem, Ian Ballantyne, Cormac Brick, Victor Carbune, Michelle˘ Casbon, Mayank Chaturvedi, Aditya Chawla, Victor Cotruta, Alice Coucke, Phil Culliton, Robert Dadashi, Lucas Dixon, Mohamed Elhawaty, Utku Evci, and 304 others. 2026. Gemma 4 technical report. Preprint, arXiv:2607.02770.

Beatrice. Warren. 1992. Sense Developments: A Contrastive Study of the Development of Slang Senses and Novel Standard Senses in English. Acta Universitatis Stockholmiensis: Stockholm studies in English. Almqvist & Wiksell International.

Ifeoluwa Wuraola, Nina Dethlefs, and Daniel Marciniak. 2024. Understanding slang with LLMs: Modelling cross-cultural nuances through paraphrasing. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 15525– 15531, Miami, Florida, USA. Association for Computational Linguistics.

Joonho Yang, Seunghyun Yoon, Hwan Chang, Byeongjeong Kim, and Hwanhee Lee. 2026. Hallucinate at the last in long response generation: A case study on long document summarization. Preprint, arXiv:2505.15291.

Jian Zhu and David Jurgens. 2021. The structure of online social networks modulates the rate of lexical change. In Proceedings of the 2021 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2201–2218, Online. Association for Computational Linguistics.

## A LLM Prompts

## A.1 List Slang Sense Disambiguation

We use the following prompt for list slang sense disambiguation:

## A.2 Single Slang Sense Disambiguation

We use the following prompt for single slang sense disambiguation:

{   
"role": "system",   
"content": "You are a precise slang sense   
disambiguation assistant. Your job is to   
determine whether a given candidate term   
appearing in a user's utterance is used with   
the {{SUBREDDIT}} slang meaning defined in   
a provided glossary."   
},   
{   
"role": "user",   
"content": "GLOSSARY (authoritative):\n{{   
GLOSSARY}}\n\nCONTEXT (for disambiguation   
only):\n{{CONTEXT}}\n\nUTTERANCE:\n{{   
UTTERANCE}}\n\nCANDIDATE TERM (appears in   
the utterance):\n{{TERM}}\n\nTASK:\   
nDetermine whether the CANDIDATE TERM in the   
UTTERANCE is used with the {{SUBREDDIT}}   
slang meaning defined in the GLOSSARY.\n\

nDECISION RULES:\n1) The candidate term must correspond to a glossary entry (base form match allowed).\n2) Use the UTTERANCE and CONTEXT to determine the meaning.\n3) If the candidate term is used with the slang/ jargon meaning defined in the glossary, return True.\n4) If the candidate term is used with a literal, standard, or different meaning, return False.\n5) Only evaluate the provided candidate term. Do NOT extract or evaluate other words.\n\nOUTPUT FORMAT ( STRICT):\nReturn exactly one token:\nTrue\ nor\nFalse\n\nDo NOT include explanations, punctuation, JSON, or extra text."

## A.3 Human-Annotated Benchmark Statistics

Table 4 summarizes the human-annotated benchmark used for evaluating slang sense detection.

<table><tr><td>Statistic</td><td>Count</td></tr><tr><td>Number of utterances</td><td>1,333</td></tr><tr><td>Positive utterances</td><td>613</td></tr><tr><td>Negative utterances</td><td>720</td></tr><tr><td>Glossary-term occurrences</td><td>1,133</td></tr><tr><td>Occurrences in glossary senses</td><td>961</td></tr><tr><td>Occurrences in non-glossary senses</td><td>172</td></tr><tr><td>Unique glossary terms evaluated</td><td>345</td></tr></table>

Table 4: Summary statistics of the human-annotated slang sense disambiguation benchmark. Note that a single utterance may contain multiple slang instances; therefore, the 613 slang-containing utterances include 1,333 slang terms in total. All three authors independently annotated the same 50 samples using the same materials: the glossary and utterance context, with access to the Internet when needed to determine the meaning. Agreement among the three annotators yielded a Fleiss’ κ of 0.783, indicating very high inter-rater agreement.

## B Additional Results

This section provides additional plots and figures from our analyses.

<table><tr><td>Feature</td><td>Orig. β</td><td>Median β</td><td>2.5% Quant.</td><td>97.5% Quant.</td><td>Same Sign</td><td>Prop.</td></tr><tr><td>Mean Betweenness</td><td>0.103</td><td>0.074</td><td>0.069</td><td>0.078</td><td>1.00</td><td>p &lt; .05 1.00</td></tr><tr><td>Mean Degree</td><td>-0.506</td><td>-0.347</td><td>-0.353</td><td>-0.339</td><td>1.00</td><td>1.00</td></tr><tr><td>Effective Topics</td><td>-0.018</td><td>-0.004</td><td>-0.005</td><td>-0.003</td><td>1.00</td><td>1.00</td></tr><tr><td>Semantic Dispersion</td><td>-0.215</td><td>-0.064</td><td>-0.065</td><td>-0.063</td><td>1.00</td><td>1.00</td></tr></table>

Table 5: Monte Carlo robustness analysis accounting for LLM classification error. Orig. $\beta$ denotes the original coefficient estimate, while Median $\beta$ denotes the median coefficient across simulations. Same Sign reports the proportion of simulations with the same coefficient direction as the original estimate, and Prop. $p < . 0 5$ reports the proportion that remain statistically significant.

![](images/699464a264a9e64b0d5442149dba64616fe34d194538c2b7b372af2d2047c933.jpg)  
Figure 4: This figure shows the total number of utterances in r/boxoffice over time. The sharp increase in activity in April 2018 coincides with the release of Avengers: Infinity War. This exogenous event substantially increased subreddit activity, potentially affecting both the structure of the interaction network and the frequency of observed slang usage. Pearson’s r shows a high correlation between the number of slang utterances and the total number of utterances.

![](images/ded6787bf01d9c826a64ea138570110971efc09368b9e901214ede7ddee33ad2.jpg)  
Figure 5: This figure shows the total number of utterances in r/AFL over time. There are systematic fluctuations throughout the year due to the seasonal structure of the AFL competition.

![](images/403dc03d8900698c3afb2d7e7bed7338a0fd3ef2611014d4500b9ccfa6ff3491.jpg)  
Figure 6: This figure shows the usage trajectories of the slang terms SUV, traction, and cam in r/cars. The SUV refers to a more popular type of car and is used more frequently in the community, resulting in a substantially different growth trajectory from those of the other slang terms.

![](images/6a36d5c36df4e899db4370e5d8bd11162eb0ccb7daaee83c4c1764dd8b25d39b.jpg)

Figure 7: This figure shows the usage trajectories of the slang terms 2v2, barb, and ice wiz in r/ClashRoyale. The 2v2 game mode was released in March 2017, after which the usage of the term remained relatively stable.  
![](images/5c12e58f0a57b7cdaf2cd2db97ca3846dedbe64f103e19f0f5bbdc991360807b.jpg)  
Figure 8: This figure shows the distribution of the number of utterances across all 54 subreddits over time. There is a general upward trend in Reddit activity, likely reflecting the growth of the Reddit community and increased access to the internet.