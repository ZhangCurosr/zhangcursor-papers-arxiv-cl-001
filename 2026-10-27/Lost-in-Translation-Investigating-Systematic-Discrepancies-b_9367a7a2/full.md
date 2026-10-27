# Lost in Translation: Investigating Systematic Discrepancies between Parallel English and Chinese Names of American Chinese Restaurants

Hang Jiang<sup>1,4</sup>, Nanxi Liu<sup>1,2</sup>, Tiantian Wang<sup>4</sup>, Cassandra Overney<sup>1</sup>, Rukun Zhang<sup>1</sup>, Artemisia Luk<sup>1</sup>, Diyi Yang<sup>3</sup>, Deb Roy<sup>1</sup>,

Jad Kabbara<sup>1</sup>

<sup>1</sup>Massachusetts Institute of Technology, <sup>2</sup>Wellesley College, <sup>3</sup>Stanford University, <sup>4</sup>Northeastern University Correspondence: hang.jiang@northeastern.edu

## Abstract

Given that the Chinese are one of the oldest ethnic groups to immigrate to the US, their cuisine have become central to the US dining scene and an interesting avenue to study how Chinese immigrants have culturally adapted into American life. In that regard, previous research has focused on analyzing the English names that Chinese restaurants adopt, e.g., finding the use of familiar and typical words. To investigate how Chinese immigrants have culturally integrated in the US depending on social, geographic and demographic contexts, we investigate in this work the under-explored relationship between the Chinese and English names of American Chinese restaurants. Specifically, we conduct a study on 3,172 Chinese restaurants (10% of the total on Google Maps) sampled across the US. We present 8-dimension name frames to capture the underlying meanings behind names and create the NAMEFRAMES dataset containing 884 annotated parallel Chinese/English names. We also built an NLP pipeline to analyze the semantic discrepancies between names with both frame-based and embedding-based distance metrics. Our frame analysis shows that English names more often emphasize Location and Specialty, while Chinese names more often emphasize Positivity and Ambiance. Furthermore, we observe that for Chinese restaurants located in urban areas, those in counties with a higher concentration of Chinese population, or those that are relatively new, the discrepancies between the English and Chinese names are noticeably larger. Finally, interviews with 10 Chinese restaurant owners reveal additional interesting insights regarding naming conventions.

## 1 Introduction

The process of immigrant groups adapting their language, values, and behaviors to a new society is commonly known as cultural assimilation in sociology and ethnic studies. The United States, colloquially hailed as “a melting pot”, has a long history of immigration and therefore provides a unique perspective to researchers to study this phenomenon. Among the diverse immigrant groups in the US, those of Chinese ethnicity (referred to as “Chinese”) hold a significant place, because their path to assimilation is not in a “straight-line” fashion (Warner and Srole, 1945; Alba and Nee, 2014) but somehow “bumpy” and “segmented” (Portes and Zhou, 1993; Zhou and Xiong, 2005). Over time, Chinese immigrants have established themselves through the restaurant business, which is central to the dining scene in the United States. Different from other ethnic restaurants in the US, Chinese restaurants usually have parallel names in English and Chinese, which sometimes do not match in meaning (Figure 1). These parallel names reflect both an embrace of cultural assimilation and a preservation of their home culture identity to Chinese immigrants. This paper provides a systematic analysis of this cultural phenomenon by investigating patterns of divergence between Chinese and English names for over three thousand American Chinese restaurants (ACRs).

![](images/fe0b046407402bcace8fc8390960ed4e3cd10558ff8f62f90707553cf7ce897b.jpg)  
Figure 1: An example of mismatched English and Chinese ACR names. We hired Prolific workers to transcribe the names by browsing photos on Google Maps. The English name encodes the Personal Name frame, while the Chinese name encodes the Positivity frame.

Prior work has used ACR menus and reviews (Chen and Park, 2021; Luo et al., 2023) to study cultural assimilation of the Chinese community. This work focuses on names because names represent how people define and mark their own cultural identity and we are interested in studying how ACR owners represent themselves as immigrants from their ACR naming practice. Ferdman and Ingraham (2016) and Chin et al. (2023) have observed that ACR English names frequently feature cultural symbols such as “panda”, “dragon”, and “wok”. Through a qualitative analysis on 423 ACRs in Los Angeles County, Chen (2018) further categorized ACR names into various frames such as Place Names. This study revealed how ACR owners demonstrate cultural assimilation through the adoption of English names, terms evoking modern European sensibility, and concepts related to American fast-food takeout. However, previous research has never compared the Chinese names with the accompanying English names in ACRs.

This paper presents a novel study on the cultural assimilation of Chinese in American society by focusing, as a case study, on the phenomenon of systematic discrepancies between parallel English and Chinese names in ACRs. We sampled 3,172 American Chinese restaurants (ACRs), 10% of the total on Google Maps, across the US and enrolled crowdworkers (67 hours in total) to transcribe their Chinese names by reviewing Google Map photos and websites. Building on Chen (2018)’s work, we adapt 8-dimension name frames to capture various aspects of the underlying meaning of names. To automate the analysis, we build an NLP pipeline to compare the discrepancies between the parallel names with both frame-based and embeddingbased distance metrics. Finally, we examine how the difference in names is correlated with sociocontextual factors, contributing significantly to our understanding of this unique cultural practice. Our main contributions are as follows:

• We present a novel study that examines the cultural assimilation of Chinese in the US and uncovers systematic semantic discrepancies between English and Chinese names of ACRs.

• We adapt 8-dimension name frames based on prior work to capture underlying meanings of names and create the NAMEFRAMES dataset on 884 pairs of ACR names.

• We build an NLP pipeline to quantify semantic discrepancies with two metrics, showing that for ACRs in urban areas, those in counties with a higher percent of Chinese, or those that are relatively new, the discrepancies between their parallel names are significantly larger.

• We conduct ten qualitative interviews with ACR owners to further complement the computational analysis, which helps us better understand their motivations and experiences behind their naming practices.

## 2 Related Work

## 2.1 Names of American Chinese Restaurants

Traditionally, running American Chinese restaurants (ACRs) is an important means of livelihood for Chinese immigrants in the United States, leading to the birth of what we now know as American Chinese cuisine (Coe, 2009). For instance, the American national condiment “ketchup” is linked to the history of Fujianese immigrants opening Chinese restaurants in New York City (Jurafsky, 2014). Researchers from Washington Post (Ferdman and Ingraham, 2016) and Smithsonians<sup>1</sup> noticed that the English names of ACRs tend to be similar and stereotypical, such as “wok”, “garden”, “dragon”, etc. Chen (2018) did a qualitative study on the English naming practices of 423 ACRs in Los Angeles County, demonstrating how these names, through the incorporation of various name frames such as region names, family names, and transliterated hanzi, reflect the diversity within Chinese culinary traditions and the immigrants’ adaptation to American culture. Despite these insightful contributions, none of the previous work delved into the patterns existing between parallel English and Chinese names of ACRs. Previous studies have predominantly focused on the English names of these restaurants (Liu and Lin, 2009; Ferdman and Ingraham, 2016; Chen, 2018; Li, 2019), often overlooking the rich cultural narratives and adaptive strategies embedded in the parallel names. The discrepancies between these parallel names can offer a deeper understanding of the cultural adaptation processes of Chinese immigrants and their attempts to negotiate identity and representation in a multicultural context.

## 2.2 Computational Frame Analysis

Prior work in NLP has presented various computational methods to automate framing analysis in large-scale datasets (Ziems and Yang, 2021; Ali and Hassan, 2022; Ziems et al., 2023). If there is no pre-annotated dataset or codebook, clustering algorithms (Burscher et al., 2016) and topic modeling (DiMaggio et al., 2013; Nguyen, 2015) are two popular unsupervised approaches to analyze frames on large corpora. However, supervised neural networks are commonly applied to detect frames if there is an existing coding schema with enough annotated frame data for training (Ali and Hassan, 2022), including probabilistic soft logic (Johnson et al., 2017), LSTM (Naderi and Hirst, 2017), recursive neural network (Ji and Smith, 2017), and transformer-based language models such as BERT (Liu et al., 2019a; Tourni et al., 2021), RoBERTa (Khanehzar et al., 2019; Huguet Cabot et al., 2020; Mendelsohn et al., 2021), XLM-RoBERTa (Bhatia et al., 2021), and MultiBERT (Akyürek et al., 2020). Recently, large language models have shown promising results to automate social science research (Grossmann et al., 2023; Bail, 2023; Rytting et al., 2023; Shen et al., 2023; Karjus, 2023), including automatic frame detection and analysis (Ziems et al., 2023; Törnberg, 2023) in a zero-shot or few-shot setting.

## 2.3 Measuring Semantic Textual Similarity

Estimating the semantic textual similarity (STS) between text data is an open problem in NLP (Chandrasekaran and Mago, 2021). In earlier approaches, two text snippets were considered similar if they contain the same words or n-grams. However, lexicon-matching methods such as BoW and TF-IDF fail to consider that different words can share similar meanings. To address the drawback of the lexical measures, prior work has proposed knowledge-based methods such as edge-counting (Rada et al., 1989) and information content (IC)- based (Sánchez et al., 2012; Zhu and Iglesias, 2016) approaches, corpus-based methods such as LSA (Landauer and Dumais, 1997) and LDA (Sinoara et al., 2019), and neural networks-based methods such as word2vec (Wang et al., 2016; Hamilton et al., 2016b,a; Kutuzov et al., 2018), LSTM (Tai et al., 2015), and CNN (He and Lin, 2016) to measure textual similarity or quantify semantic change. Recently, transformer-based models such as BERT (Devlin et al., 2019), RoBERTa (Liu et al., 2019b), DistilBERT (Sanh et al., 2019), and T5 (Raffel et al., 2020) are pre-trained as language models and have been commmonly used to represent texts for measuring textual similarity. Reimers and Gurevych (2019) introduced Sentence-BERT (SBERT), which can derive semantically meaningful sentence embeddings that can be compared using cosine similarity. Reimers and Gurevych (2020) extended SBERT to support multiple languages. The most recent work in NLP focus on optimizing language models for STS (Gao et al., 2021; Chuang et al., 2022; Wu et al., 2022; Jiang et al., 2022; Li and Li, 2023), improved smaller language models by generating synthetic data from LLMs (Wang et al., 2023), or scaling sentence embeddings to larger language models such as T5 (Ni et al., 2021) and 2.7B OPT (Jiang et al., 2023).

## 3 NAMEFRAMES Dataset

We collected all American Chinese Restaurants (ACRs) with Google Maps API, sampled a subset of ACRs from each state across the U.S., and hired crowdworkers to double-transcribe Chinese names from online images. To quantify the information lost in translation, we adapted 8-dimensional name frames and created the NAMEFRAMES dataset. This dataset includes annotations of both English and Chinese names across 884 ACRs.

## 3.1 Data Collection

From 2023-07-13 to 2023-07-15, we collected U.S. restaurants categorized as “Chinese restaurant” using Outscrapper<sup>2</sup>, which provides public Google Maps Places API information such as names, websites, locations, ratings, prices, and photos. We filtered out chain and fast food restaurants because their names repeat frequently (Appendix A.1.1), yielding 31,548 unique ACRs. We then used stratified sampling by state to select 3,172 ACRs: 10% from states with more than 400 ACRs, 40 restaurants from states with 40–400 ACRs, and all restaurants from states with fewer than 40 ACRs. Finally, we removed duplicate Chinese–English name pairs. See Appendix A.1 for the distribution. See the distribution of ACRs in Appendix A.1.

## 3.2 Chinese Name Transcription

Google Maps API provides English names of the ACRs but not the Chinese names. To obtain accurate transcriptions of their Chinese names, we first obtained IRB exemption and hired two Chinesespeaking crowd workers to identify and transcribe the Chinese name of each restaurant in simplified Chinese characters by browsing its photos from

<table><tr><td>Frame</td><td>Simplified Definition</td><td>Selected Examples</td></tr><tr><td>Personal Name</td><td>first or last names with or without possessives</td><td>Lin&#x27;s Kitchen, P.F. Chang&#x27;s</td></tr><tr><td>Specialty</td><td>special dishes, cooking methods, or ingredients</td><td>Peking Duck House, Noodle Galaxy</td></tr><tr><td>Positivity</td><td>superlatives or other positive modifiers</td><td>Happy Lamb, Best Chinese Buffet</td></tr><tr><td>Culture</td><td>references to cultural associations</td><td>Panda Express, The Phoenix Club</td></tr><tr><td>Location</td><td>place names or regional origins</td><td>China Garden, Oriental Kitchen</td></tr><tr><td>Ambiance</td><td>dining atmosphere or intimate words of family or relationship</td><td>Jade Garden, Friendship BBQ</td></tr><tr><td>Creativity</td><td>puns or names derived from creative linguistic processes</td><td>Wok N&#x27;Roll, Noodology</td></tr><tr><td>Romanization</td><td>Romanized Chinese terms excluding locations and personal names</td><td>Ding Tai Fung, Hai Di Lao Hot Pot</td></tr></table>

Table 1: This table shows 8 frames with their simplified definitions and selected examples. The bold text spans indicate where the corresponding frame is encoded in the names. “Ding Tai Fung” means “wok, peace, and abundance”. “Hai Di Lao” means “catch from the deep ocean”. See full details of the frames in Appendix A.3.

Google Maps (see Figure 1). If there is any disagreement between two annotators, one of the authors reviewed the annotations to make a final decision. We observed a decent inter-rater agreement score 0.673 in Cohen’s Kappa. After transcribing the Chinese names of the 3,172 ACRs, we identified 1,722 ACRs with both English and Chinese names and 1,450 with only English names.

## 3.3 Name Frame Design

Chen (2018) proposed a few name frames to conduct qualitative analysis on meanings encoded in ACR English names. We adapted their framework and generalized these frames to both English and Chinese names. Table 1 shows the simplified frame definitions and selected examples. The complete codebook can be found in Appendix A.3.

## 3.4 Name Frame Annotation

We sampled 884 ACRs for frame annotation. Each restaurant was annotated by three co-authors who are native Chinese speakers with professional English proficiency. We annotated the data in two rounds, discussing disagreements after each round to refine the frame definitions and annotations. As shown in Table 4, inter-annotator agreement was high across the eight frames, with scores ranging from 0.78 to 0.935 for English names and 0.79 to 0.9 for Chinese names. To tune LLMs for labeling the remaining data, we randomly split the annotated data into 50% validation and 50% test sets for each language, removing duplicate names within each language.

## 4 Methods

First, we describe methods to detect name frames and compare their performance on NameFrames.

These frame detection models were used to automate frame analysis and to determine the information gained or lost when comparing English and Chinese names. We utilize both frame-based Jaccard distance and embedding-based Cosine distance to quantify the semantic discrepancies between names. These distance scores were subsequently correlated with socio-contextual factors. Finally, we conducted and analyzed ten qualitative interviews to deepen our understanding of the Chinese immigrant integration into US society.

## 4.1 Name Frame Detection

Task Formulation The task of name frame detection is defined in the following manner: Given a specific name denoted as N, our target is to identify if each specific frame $f _ { i }$ within the set F of all possible frames is encoded in the name N.

Methods We built rule-based and prompt-based methods to detect name frames from ACR English and Chinese names. To establish a strong baseline, we first developed a rule-based model (“Expert Rules” in Table 2) based on lexicons and/or heuristics for each name frame respectively. Details about the lexicon-based approach are discussed in Appendix B.1. Afterwards, we experimented with GPT-5.5<sup>3</sup>, Qianwen (qwen3.7-max)<sup>4</sup>, Kimi (kimi-k2.6)<sup>5</sup>, and Claude Sonnet 4.6 (claude-sonnet-4.6) <sup>6</sup> given that they are among the state-of-the-art multilingual LLMs. We do not include fine-tuned language models such as RoBERTa (Liu et al., 2019b) as baselines because they do not perform well on our small dataset where some labels are too sparse for effective finetuning.

<table><tr><td>Model</td><td>Personal Name</td><td>Speciality</td><td>Positivity</td><td>Culture</td><td>Location</td><td>Ambiance</td><td>Creativity</td><td>Romanization</td><td>Average</td></tr><tr><td colspan="10">Chinese Names (Weighted F1)</td></tr><tr><td>Expert Rules</td><td>88.09</td><td>97.39</td><td>90.81</td><td>94.79</td><td>88.18</td><td>93.44</td><td>95.20</td><td></td><td>92.56</td></tr><tr><td>Qwen3.7-Max</td><td>95.14</td><td>98.86</td><td>92.38</td><td>94.15</td><td>92.80</td><td>96.08</td><td>96.89</td><td></td><td>95.18</td></tr><tr><td>Kimi K2.6</td><td>94.23</td><td>97.42</td><td>88.64</td><td>94.93</td><td>94.37</td><td>93.46</td><td>97.05</td><td></td><td>94.30</td></tr><tr><td>GPT-5.5</td><td>96.01</td><td>98.02</td><td>92.68</td><td>95.38</td><td>93.99</td><td>94.71</td><td>96.88</td><td></td><td>95.38</td></tr><tr><td>Claude Sonnet 4.6</td><td>94.83</td><td>97.75</td><td>90.13</td><td>96.66</td><td>95.20</td><td>94.95</td><td>96.88</td><td></td><td>95.20</td></tr><tr><td colspan="10">English Names (Weighted F1)</td></tr><tr><td>Expert Rules</td><td>85.83</td><td>85.83</td><td>82.46</td><td>93.95</td><td>92.02</td><td>91.43</td><td>74.22</td><td>72.42</td><td>84.77</td></tr><tr><td>Qwen3.7-Max</td><td>96.79</td><td>97.46</td><td>91.61</td><td>96.95</td><td>91.11</td><td>90.34</td><td>99.21</td><td>91.79</td><td>94.41</td></tr><tr><td>Kimi K2.6</td><td>96.27</td><td>96.72</td><td>89.41</td><td>97.64</td><td>91.61</td><td>88.44</td><td>97.60</td><td>90.57</td><td>93.53</td></tr><tr><td>GPT-5.5</td><td>98.45</td><td>97.27</td><td>93.14</td><td>97.49</td><td>95.18</td><td>87.04</td><td>97.01</td><td>93.12</td><td>94.84</td></tr><tr><td>Claude Sonnet 4.6</td><td>97.62</td><td>97.27</td><td>92.46</td><td>96.35</td><td>86.46</td><td>85.95</td><td>97.98</td><td>94.08</td><td>93.52</td></tr></table>

Table 2: Performance of different approaches on Name Frame Detection in both English and Chinese ACR names. “Expert Rules” represents the rule-based system based on lexicons and heuristics. Qwen3.7-Max, Kimi K2.6, GPT-5.5, and Claude Sonnet 4.6 represent the best performance of each model across two prompt settings: (1) zero-shot, (2) few-shot, as described in Section 4.1.

Prompts We experimented with three prompting strategies: (1) name + definition (“zero-shot”), (2) name + definition + rules + examples (“few-shot”). The first strategy is zero-shot, which prompts LLMs to predict if the restaurant name encodes a certain frame given its name and definition. The second strategy few-shot, which aims to improve the strategy (1) by providing a few examples or rules of the target frame along with its name and definition. Strategy (2) provides the frame name, definition, expert-curated rules, and common types of examples to LLMs in a highly similar format as the codebook (see Appendix A.3). The format of these two prompt variants are discussed in the Appendix B.2.

Model Settings For model-based name-frame labeling, we used comparable settings across GPT-5.5, Qwen3.7-Max, Kimi K2.6, and Claude Sonnet 4.6 where supported: seed 42, reasoning disabled, batches of 20 labels per request, and an 800-token output cap. We set temperature to 0.0 for Qwen3.7- Max, Kimi K2.6, and Claude Sonnet 4.6; GPT-5.5 did not support user-specified temperature for this endpoint, so we relied on seed 42 for determinism.

Results In Table 2, we show the best results from each model due to space limits, with full promptsetting results in Appendix B.3. Overall, no single model dominates across all frames, but the newer LLMs generally outperform Expert Rules. GPT-5.5 achieves the highest average weighted F1 in both Chinese and English labeling, so we use it to

label the remaining data.

## 4.2 Measuring Name Discrepancies

We developed two different metrics to quantify discrepancies between the Chinese and English names. The first metric is based on name frames, which aims to capture typical meanings such as culture and interesting syntactic forms such as creativity. The second embedding-based metric is a standard NLP method to quantify semantic differences. In our study, we first translates the Chinese name to English and measures the embedding distance between translated and original English names. These two metrics are meant to capture slightly different aspects of linguistic discrepancies in names.

## 4.2.1 Frame-based Jaccard Distance (JD)

Name frames encode nuanced linguistic differences in meanings (e.g., culture) and forms (e.g., creativity) between English and Chinese names. Therefore, we can use frame-based Jaccard distance (JD) to measure the discrepancies between English and Chinese names. We define JD to be:

$$
J D ( F _ { E N } , F _ { C H } ) = 1 - \frac { | F _ { E N } \cap F _ { C H } | } { | F _ { E N } \cup F _ { C H } | }
$$

In the equations above, $F _ { E N }$ and $F _ { C H }$ indicates the set of name frames encoded in the English and Chinese names of the same restaurant respectively.

## 4.2.2 Embedding-based Cosine Distance (CD)

Alternatively, we measure semantic difference using name embeddings. We translate each Chinese restaurant name into English as literally as possible, then compute the distance between the translated Chinese name and the observed English name. Because literal name translation differs from standard machine translation, we evaluated Google Cloud Translation API Basic $\nabla 2 ^ { 7 }$ , GPT-5.5, and Claude Sonnet 4.6 against two human literaltranslation references for 99 restaurants. Claude Sonnet 4.6 achieved the strongest average embedding similarity and BERTScore, so we use its translations in the main analysis; full translation results are reported in Appendix B.5. We then use all-mpnet-base-v2<sup>8</sup> to compute embeddingbased semantic distance.

$$
C D ( E _ { T S } , E _ { E N } ) = 1 - \frac { E _ { T S } \cdot E _ { E N } } { \| E _ { T S } \| _ { 2 } \cdot \| E _ { E N } \| _ { 2 } }
$$

In the equation above, $E _ { T S }$ and $E _ { E N }$ indicate the embeddings of the translated and original English names of the same restaurant respectively.

## 4.2.3 Interview with ACR Owners

To understand how ACR owners create Mandarin and English names, we interviewed ten ACR owners in Massachusetts about factors influencing naming decisions like customer groups and location, and how the restaurant reflects the owner’s heritage and identity. We analyzed the interviews using affinity diagramming (Lucero, 2015) to identify key naming themes. Additional interview procedures are in Appendix C.1, and a fuller summary of results is in Appendix C.3.

## 5 Analysis & Discussion

We are interested in answering the following research questions (RQs): RQ1: are there systematic discrepancies in naming practices in ACRs? RQ2: what socio-contextual factors impact the ACR naming process? RQ3: do ACR owners deliberately adjust the parallel names to their audiences?

## 5.1 Discrepancies in Parallel Names

By applying the best frame detection models to the 1,722 ACRs with both Chinese and English names, we show the distribution of naming frames in Figure 2, using GPT5.5 labels with human overrides when available. The clearest pattern is that English names more often emphasize Location and Specialty, while Chinese names more often emphasize Positivity. Chinese names also show slightly higher counts for Culture and Ambiance, while

Creativity remains rare in both languages. These patterns suggest that English-facing names tend to foreground legible market information, such as geographic cues and cuisine type, whereas Chinese names more often preserve auspicious, evaluative, or culturally meaningful content.

![](images/e45360e28529c08ec21fe2f0a026a403888386d824f451bfbe768721c6d240d3.jpg)  
Figure 2: Percentage of unique English and Chinese names tagged with each frame.

## 5.2 Location

Location is an important factor shaping naming practices. We quantify semantic differences between parallel Chinese and English names using both frame-based Jaccard distance and embeddingbased cosine distance. As shown in Figure 3a, ANOVA tests show significant location differences for both metrics. Post-hoc Tukey HSD tests indicate that urban ACRs have significantly more divergent parallel names than rural ACRs for both metrics, while the urban–suburban and rural–suburban differences are not statistically significant. We conjecture that this pattern reflects the different audience structures of urban restaurant markets: in urban areas, where Chinese-speaking customers are more concentrated, ACRs may have stronger incentives to preserve distinctive Chinese-facing names while using more familiar English-facing names for broader local audiences.

The interview study provides a concrete example of the patterns shown in Figure 3a. In urban areas such as Boston’s Chinatown, where ACRs are abundant, owners face intense competition and strive to distinguish their establishments. For instance, R7 shared that a restaurant in Chinatown has to be “unique to stand out so that they can have business. There is a lot of competition for the same type of food, and other restaurants have been openfor longer.” R7 identified a type of Chinese food that was not being sold in Chinatown and incorporated distinctive elements of his cuisine into both the Mandarin and English names to attract patrons. Conversely, in rural and suburban settings with fewer ACRs, restaurant owners may prioritize personal connections and autonomy over differentiation. R9 decided to open a restaurant in a rural town to “escape Chinatown.” He and his partner picked their current location because it is “adventurous”, and they wanted to “start a new life with morefreedom.” The newfound freedom is reflected in R9’s choice of naming the restaurant in memory of his daughter and deciding not to create a Mandarin name.

## 5.3 Chinese Ethnic Percentage

We also test whether naming practices vary with local Chinese population share. Using 2022 ACS county-level estimates from NHGIS<sup>9</sup>, we divide counties into high- and low-Chinese-percent groups based on the sample median. As shown in Figure 3b, ACRs in high-Chinese-percent counties show greater frame-based name divergence than those in low-Chinese-percent counties (Jaccard distance: 0.594 vs. 0.539; Welch’s $p = 0 . 0 0 5 )$ . However, the continuous Pearson correlation with Chinese percentage is not statistically significant, and embedding-based distance shows almost no group difference (0.517 vs. 0.514; p = 0.800). These results suggest that Chinese population share is associated more clearly with categorical naming-frame divergence than with overall embedding distance. One interpretation is that ACR owners in areas with larger Chinese communities may be more comfortable using distinctive Chinese-facing naming frames while keeping English-facing names legible to broader local audiences.

The interviews show that owners strategically consider customer groups, especially age and cultural background, as their customers change over time. For example, R10 adapts food to a rural local population that is less than 1% Chinese, while R5 targets international Chinese students visiting Chinatown.

This customer awareness often shapes naming choices. Mandarin names help attract Chinesespeaking customers: R2, whose customers are about 50% Chinese, chose a Chinese name that signals regional dishes and noted that the mismatch between the Chinese and English names sparks curiosity among Chinese customers, who “want to investigate and try it.” English names, by contrast, are often chosen to be familiar and legible to American customers. R2 chose the English name first because the “name tells [American customers] it’s an Asian food.”

## 5.4 Opening Date

We also investigate whether the opening date is associated with the naming practice. Since it is difficult to obtain the opening date of these restaurants, we use the date from the earliest review for each restaurant on Google Maps to approximate the opening date. To make the periods as comparable as possible, we divide the dates into three equal-month periods: the first period “2004-01 to 2010-08”, the second period “2010-09 to 2017- 04”, and the third period “2017-05 to 2023-12”. We find that ACRs in the third period have the highest mean divergence under both Jaccard and Cosine distance. However, the updated ANOVA results are significant only for Cosine distance, and the post-hoc Tukey HSD tests do not show significant pairwise differences between periods. To avoid potential bias introduced by manually selecting time intervals, we directly calculated the Pearson correlation using “days since opening” to examine whether a temporal trend exists. We found a negative correlation between opening time and Cosine distance using Pearson correlation $( r ( 1 7 0 3 ) = - . 0 6 2 , p = . 0 1 0 )$ , suggesting that newer restaurants tend to have slightly more divergent parallel names; the corresponding Jaccarddistance association is not statistically significant. Our hypothesis is that in recent years ACR owners may have become more comfortable marking their Chinese identity using distinctive Chinese names while keeping familiar English names for local customers, though the updated temporal evidence is weaker than before.

Though most interviewees opened restaurants before 2011, several examples suggest how opening period shaped naming decisions. R4 named a 2003 restaurant after a popular late-1990s movie featuring Chinese characters, explaining that “it is good to have a simple name to be more acceptable by customers.” R6, whose restaurant opened in 2010, chose an English name associated with a popular Chinese dish that was receiving media attention, which “made American customers curious.” In contrast, the two owners who opened restaurants after 2016 emphasized personal identity: R1 wanted the name to “be personal” by including her Chinese name, while R7 said he “does not care if other people like the name” when creating and

![](images/3f1ea53fcffae05e545888d07ad4f772e79709568bb3c29feade407cecc7bbba.jpg)  
(a) Location

![](images/042272b44d60960b264784841c788352a43ca7c5a3d15862fdb85272b0976095.jpg)  
(b) Chinese Percentage

![](images/5078f1d1d0472463a8c3b99f974bfd3253fc004c26e99e4c4a7df8388da71b1c.jpg)  
(c) Opening Date  
Figure 3: These three figures show the semantic distance scores between parallel English and Chinese names across different location, Chinese percentage, and opening date groups. Jaccard distance is the frame-based FD score and Cosine distance is translation-based embedding cosine distance with all-mpnet-base-v2.

copyrighting it.

Owners also reflected on broader changes in ACRs. R3 confirms the pattern shown in Figure 3c, sharing that “many new restaurants with creative names are opening. Many owners are now international students orjoint ventures.”, though many also “opened and closed quickly.” R10 observed that “more authentic Chinesefood is getting here,” while R8 argued that “Chinatowns are dying and spreading to different communities,” linking this diffusion to greater confidence, English fluency, and integration into U.S. society.

## 5.5 Effect size of socio-contextual factors

To evaluate how much socio-contextual factors influence naming discrepancy, we performed multivariate regression followed by non-linear robustness checks (Appendix E). The multivariate regression results show that only rural location $( \beta \ =$ $- . 1 6 8 , p = . 0 1 4 )$ for Jaccard distance and opening date $( \beta = - . 0 1 8 , p = . 0 0 6 )$ for Cosine distance yielded statistically significant results. For both naming discrepancy measurements, the multivariate regression models showed low explanatory power $( R ^ { 2 } ~ < ~ 0 . 0 0 8 )$ . Random Forest regressor robustness check yielded non-positive explanatory power, suggesting no meaningful non-linear relationships are left undetected. We offer the following reflections on these experiment results. First, we acknowledge a pronounced class imbalance in our dataset, where rural entries account for only 2% of the sample compared to 91% for urban entries. However, this class imbalance is not caused by our sampling procedure, but by the data incompleteness in the demographic data source. Furthermore, the low explanatory power of our regression models indicates that restaurant naming cannot be adequately explained by a few macro-level factors; rather, owners’ choices are driven by more complex, idiosyncratic decision-making processes.

## 6 Conclusion

In conclusion, our paper studies cultural assimilation through discrepancies between parallel English and Chinese names of ACRs. We adapt an eightdimensional name-frame schema and create the NAMEFRAMES dataset on 884 pairs of ACR names. Using GPT-5.5 labels with human overrides, we find that English names more often emphasize Location and Specialty, while Chinese names more often emphasize Positivity, and to a lesser extent Culture and Ambiance. Applying the labels to the 1,722 ACRs with both English and Chinese names, we show that these patterns extend beyond the annotated sample. These results suggest that Englishfacing names foreground legible market information, whereas Chinese-facing names preserve more auspicious and culturally meaningful content.

We further show that name divergence is associated with context, though the strength of the relationship depends on the distance metric. ACRs in urban areas show greater Chinese–English name divergence than rural restaurants, while suburban restaurants generally fall between the two. Counties with higher Chinese population share show larger frame-based differences, but not consistently larger embedding-based differences. Opening-date analyses also suggest that more recently opened restaurants tend to exhibit greater embedding-based divergence, while frame-based results are weaker. Together, these findings suggest that ACR naming practices reflect both market-facing adaptation and the preservation of culturally meaningful Chinesefacing identities, with divergence shaped by local market, ethnic, and temporal context. We complement the computational analysis with ten qualitative interviews with ACR owners to better understand the motivations, audience considerations, and lived experiences behind these naming practices.

## Limitations

Names as a single, noisy signal. Restaurant names are one public-facing artifact among many, and they are shaped by forces that have little to do with cultural orientation: family history, prior tenants and inherited signage, landlord or municipal sign constraints, trademark availability, deliveryplatform display conventions, and simple inertia after ownership changes. A divergent name pair is therefore evidence of a naming practice, not a measurement of an owner’s identity or degree of assimilation, and we caution against reading our aggregate patterns as individual-level inferences.

Recoverability of Chinese names is not missingat-random. Of the 3,172 sampled ACRs, we recovered Chinese names for 1,722. The remaining 1,450 include both restaurants that genuinely have no Chinese name and restaurants whose Chinese signage was simply not visible in available Google Maps photos, Street View imagery, or websites. These two cases are systematically confounded with the very covariates we study: urban restaurants and those in high-Chinese-share counties are more likely to be photographed, to have interior menu photos, and to display Chinese signage prominently. Our analyses condition on successful recovery, so location and ethnic-share effects may be partly attributable to differential visibility rather than differential naming.

Orthographic and dialectal flattening. We instructed annotators to transcribe into simplified characters. Many ACRs, particularly those founded by Cantonese- and Toisanese-speaking immigrants, use traditional characters, and some romanize according to Cantonese rather than Mandarin. Normalizing to simplified Mandarin erases orthographic and dialectal information that is itself a marker of migration cohort and regional origin, and our Romanization frame is defined in ways that implicitly privilege Mandarin pinyin.

Frame schema and annotation. Our eight frames are adapted from a schema developed qualitatively on 423 English names in a single county, then generalized to Chinese. The resulting categories may not carve Chinese naming practice at its joints: conventions such as the 记 suffix, register distinctions among <sup>轩</sup>/<sup>楼</sup>/<sup>阁</sup>/<sup>坊</sup>, and numerological auspiciousness are absorbed into broad frames rather than modeled directly. Labels are binary and unweighted, so a name that gestures faintly at Culture and one built entirely around a cultural symbol are treated identically. Annotation was performed by co-authors who were not blind to the study hypotheses, which raises the possibility of hypothesis-consistent labeling despite high interannotator agreement.

Properties of the distance metrics. Jaccard distance over eight binary frames is coarse and takes only a small number of discrete values, and the label spaces are asymmetric: Romanization is undefined for Chinese names, which mechanically inflates JD for any English name carrying that frame. Cosine distance is computed after a translation step, so translation error propagates directly into the measure; all-mpnet-base-v2 is an Englishonly encoder trained on general text and is not well calibrated for very short proper-noun strings. The two metrics disagree in several analyses (notably Chinese population share), and we report both rather than adjudicating between them, but this also means our conclusions are metric-dependent.

Correlational design and inferential caveats. All socio-contextual analyses are observational. County-level ACS estimates are aggregates, so associations with Chinese population share are subject to the ecological fallacy: we do not observe the composition of any restaurant’s actual customer base. Median-splitting a continuous covariate discards information and can manufacture group differences that the continuous test does not support, which is precisely the pattern we observe. We test three socio-contextual factors under two metrics without correcting for multiple comparisons, so borderline results should be treated as exploratory.

Opening date proxy and survivorship. We approximate opening date by the earliest Google Maps review, which systematically compresses the histories of older restaurants (Google Maps reviews are sparse before the early 2010s) and misdates restaurants that rebranded, changed ownership, or relocated. Our sample also contains only restaurants operating at collection time, so any temporal trend confounds cohort effects with differential survival—and one of our own interviewees observed that many newer, creatively named restaurants “opened and closed quickly.”

Interview sample and analysis. Our qualitative component comprises ten owners in Massachusetts, eight of them urban, six from Fuzhou. Recruitment by walk-in and cold call selects for owners with time and willingness to talk. Because most participants declined recording, our data are English-language notes taken in real time during Mandarin-language interviews, so we have no verbatim transcripts; quoted material reflects a notetaker’s contemporaneous rendering rather than the participant’s exact words. Themes were identified by authors already familiar with the computational findings.

Reliance on proprietary LLMs. Frame detection and translation depend on closed, versioned commercial models accessed through APIs. These endpoints are not stable over time, which limits exact reproducibility even with fixed seeds, and we cannot rule out that widely circulated ACR names or prior work on this topic appeared in pretraining data. LLMs also encode culturally situated priors about what counts as “cultural,” “creative,” or market-legible, and those priors are plausibly Western-centric in ways that align with—rather than independently test—our hypotheses. We therefore treat model outputs as structured approximations validated against human labels on a limited sample, not as ground truth.

Scope. We study one diaspora, in one country, at one moment. “Chinese” is used throughout as an administrative convenience that flattens Cantonese, Fujianese, Taiwanese, Sichuanese, Hui, and Southeast Asian Chinese communities with quite different migration histories and naming conventions. We make no claims about other immigrant cuisines or about ACRs outside the United States.

Missing data in data source Our demographic data were sourced from IPUMS NHGIS. However, due to missing data in certain rural areas, we had to drop incomplete records during multivariate regression for effect size analysis. While this ensured complete feature coverage, it introduced class imbalance that potentially impacted data analysis.

## Ethical Considerations

Human subjects and consent. Both the transcription task and the owner interviews were reviewed by our institution’s IRB and received exempt determination). Interviews were conducted in Mandarin; participants received a verbal description of the study’s purpose and data handling and gave verbal informed consent before beginning. Most participants declined audio recording, and we honored this by taking written notes only. Participation was voluntary and participants could decline any question or withdraw at any point.

Crowdworker compensation and conditions. Transcription was performed by Prolific workers who self-identified as first-language Chinese speakers. Across the task, workers logged approximately 67 hours. We paid an effective rate of 15 dollars/hour, above the state minimum wage and above Prolific’s recommended floor. Surveys were piloted to calibrate time estimates, and we did not reject submissions on the basis of disagreement between the two independent annotators.

Privacy and identifiability. Restaurants are public commercial entities and their names and locations are publicly listed, so we treat name and location data as public record. Owners, however, are individuals, and small-business owners are readily re-identifiable from a combination of city, opening year, and cuisine. We refer to interviewees only as R1–R10, do not report restaurant names, exact locations, or the names of family members, and paraphrase or omit details from the interviews that would narrow identification to a single establishment—including several personal and family circumstances that participants shared. Some quoted material remains distinctive despite these steps; we retained it only where participants indicated they were comfortable with the detail being reported.

Data release. We release the NAMEFRAMES annotations—parallel name pairs with frame labels—together with our codebook, prompts, and analysis code, to support replication. We do not redistribute raw scraped Google Maps records, review text, photographs, or owner-associated metadata, both to respect platform terms of service and because aggregated business records are more easily repurposed for commercial targeting than the derived annotations we need for replication. No interview notes are released.

Framing and risk of essentialization. “Cultural assimilation” is a contested construct with a history of being used normatively to rank groups by their proximity to a dominant culture. Our namedistance measures are descriptive: a larger distance indicates that two names encode different meanings, not that an owner is less integrated, less authentic, or more foreign. We have tried to write against a reading in which divergence scores become an assimilation index applied to ethnic communities. Relatedly, frames such as Culture and Romanization risk reproducing an exoticizing gaze if read as markers of foreignness rather than as descriptive categories of naming practice.

Potential for misuse. Aggregated, geolocated data about ethnic businesses can be repurposed for targeting—commercially, in the form of marketing or acquisition pressure, and in worse cases as input to harassment or discrimination. Anti-Asian harassment of restaurant businesses in the U.S. has been documented and recent. This informed our decision to release annotations rather than a geolocated business registry, and we encourage anyone extending this work to weigh the same tradeoff.

Positionality. The annotation team consists of native Chinese speakers with professional English proficiency, several of whom are themselves immigrants to the United States; the interviews were conducted by Mandarin-speaking members of the team. This shared background aided rapport and interpretation of culturally specific names, and it also means our intuitions about what is “auspicious,” “creative,” or “market-legible” are embedded in the codebook rather than external to it.

## Acknowledgements

We used AI coding assistants for coding in the statistical experiments in Section 5.5. All AIgenerated code underwent two rounds of review by the authors, with manual verification where necessary. The text part in corresponding section was written by the authors. For the empirical/statistical results description, we consulted AI agent exclusively for wording accuracy. The authors retain full responsibility for all code and content in this paper.

## References

Afra Feyza Akyürek, Lei Guo, Randa Elanwar, Prakash Ishwar, Margrit Betke, and Derry Tanti Wijaya. 2020. Multi-label and multilingual news framing analysis. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 8614– 8624, Online. Association for Computational Linguistics.

Richard Alba and Victor Nee. 2014. Rethinking assimilation theory for a new era of immigration 1. In The New Immigrant in American Society, pages 2–50. Routledge.

Mohammad Ali and Naeemul Hassan. 2022. A survey of computational framing analysis approaches. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9335–9348, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Christopher A Bail. 2023. Can generative AI improve social science?

Vibhu Bhatia, Vidya Prasad Akavoor, Sejin Paik, Lei Guo, Mona Jalal, Alyssa Smith, David Assefa Tofu, Edward Edberg Halim, Yimeng Sun, Margrit Betke, Prakash Ishwar, and Derry Tanti Wijaya. 2021. Open-Framing: Open-sourced tool for computational framing analysis of multilingual data. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 242–250, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Bjorn Burscher, Rens Vliegenthart, and Claes H de Vreese. 2016. Frames beyond words: Applying cluster and sentiment analysis to news coverage of the nuclear power issue. Social Science Computer Review, 34(5):530–545.

Dhivya Chandrasekaran and Vijay Mago. 2021. Evolution of semantic similarity—a survey. ACM Computing Surveys (CSUR), 54(2):1–37.

Lindsey NH Chen. 2018. Of authenticity and assimilation: Names of american chinese restaurants. Names, 66(1):3–13.

Zaiqian Chen and Joonsuk Park. 2021. Analyzing cultural assimilation through the lens of yelp restaurant reviews. In 2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA), pages 1–10. IEEE.

Gabriel Chin et al. 2023. Symbols of survival: Finding the oldest chinese restaurants in the united states. Available at SSRN 4493301.

Yung-Sung Chuang, Rumen Dangovski, Hongyin Luo, Yang Zhang, Shiyu Chang, Marin Soljacic, Shang-Wen Li, Scott Yih, Yoon Kim, and James Glass. 2022. DiffCSE: Difference-based contrastive learning for sentence embeddings. In Proceedings of the 2022 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 4207–4218, Seattle, United States. Association for Computational Linguistics.

Andrew Coe. 2009. Chop suey: A cultural history of chinese food in the United States. Oxford University Press.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor

Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Paul DiMaggio, Manish Nag, and David Blei. 2013. Exploiting affinities between topic modeling and the sociological perspective on culture: Application to newspaper coverage of us government arts funding. Poetics, 41(6):570–606.

RA Ferdman and Christopher Ingraham. 2016. We analyzed the names of almost every chinese restaurant in america. this is what we learned. Washington Post.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Igor Grossmann, Matthew Feinberg, Dawn C Parker, Nicholas A Christakis, Philip E Tetlock, and William A Cunningham. 2023. Ai and the transformation of social science research. Science, 380(6650):1108–1109.

William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016a. Cultural shift or linguistic drift? comparing two computational measures of semantic change. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2116–2121, Austin, Texas. Association for Computational Linguistics.

William L. Hamilton, Jure Leskovec, and Dan Jurafsky. 2016b. Diachronic word embeddings reveal statistical laws of semantic change. In Proceedings ofthe 54th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1489–1501, Berlin, Germany. Association for Computational Linguistics.

Hua He and Jimmy Lin. 2016. Pairwise word interaction modeling with deep neural networks for semantic similarity measurement. In Proceedings ofthe 2016 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 937–948, San Diego, California. Association for Computational Linguistics.

Pere-Lluís Huguet Cabot, Verna Dankers, David Abadi, Agneta Fischer, and Ekaterina Shutova. 2020. The Pragmatics behind Politics: Modelling Metaphor, Framing and Emotion in Political Discourse. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 4479–4488, Online. Association for Computational Linguistics.

Yangfeng Ji and Noah Smith. 2017. Neural discourse structure for text categorization. arXiv preprint arXiv:1702.01829.

Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. 2023. Scaling sentence embeddings with large language models. arXiv preprint arXiv:2307.16645.

Ting Jiang, Jian Jiao, Shaohan Huang, Zihan Zhang, Deqing Wang, Fuzhen Zhuang, Furu Wei, Haizhen Huang, Denvy Deng, and Qi Zhang. 2022. Prompt-BERT: Improving BERT sentence embeddings with prompts. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 8826–8837, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Kristen Johnson, Di Jin, and Dan Goldwasser. 2017. Leveraging behavioral and social information for weakly supervised collective classification of political discourse on Twitter. In Proceedings ofthe 55th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 741–752, Vancouver, Canada. Association for Computational Linguistics.

Dan Jurafsky. 2014. The language of food: A linguist reads the menu. WW Norton & Company.

Andres Karjus. 2023. Machine-assisted mixed methods: augmenting humanities and social sciences with artificial intelligence. arXiv preprint arXiv:2309.14379.

Shima Khanehzar, Andrew Turpin, and Gosia Mikolajczak. 2019. Modeling political framing across policy issues and contexts. In Proceedings of the 17th Annual Workshop ofthe Australasian Language Technology Association, pages 61–66, Sydney, Australia. Australasian Language Technology Association.

Andrey Kutuzov, Lilja Øvrelid, Terrence Szymanski, and Erik Velldal. 2018. Diachronic word embeddings and semantic shifts: a survey. In Proceedings ofthe 27th International Conference on Computational Linguistics, pages 1384–1397, Santa Fe, New Mexico, USA. Association for Computational Linguistics.

Thomas K Landauer and Susan T Dumais. 1997. A solution to plato’s problem: The latent semantic analysis theory of acquisition, induction, and representation of knowledge. Psychological review, 104(2):211.

Saihong Li. 2019. A corpus-based multimodal approach to the translation of restaurant menus. Perspectives, 27(1):1–19.

Xianming Li and Jing Li. 2023. Angle-optimized text embeddings. arXiv preprint arXiv:2309.12871.

Haiming Liu and Lianlian Lin. 2009. Food, culinary identity, and transnational culture: Chinese restaurant business in southern california. Journal of Asian American Studies, 12(2):135–162.

Siyi Liu, Lei Guo, Kate Mays, Margrit Betke, and Derry Tanti Wijaya. 2019a. Detecting frames in news headlines and its application to analyzing news framing trends surrounding U.S. gun violence. In

Proceedings of the 23rd Conference on Computational Natural Language Learning (CoNLL), pages 504–514, Hong Kong, China. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019b. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.

Andrés Lucero. 2015. Using affinity diagrams to evaluate interactive prototypes. In Human-Computer Interaction–INTERACT 2015: 15th IFIP TC 13 International Conference, Bamberg, Germany, September 14-18, 2015, Proceedings, Part II 15, pages 231–248. Springer.

Yiwei Luo, Kristina Gligoric, and Dan Jurafsky. 2023.´ Othering and low prestige framing of immigrant cuisines in us restaurant reviews and large language models. arXiv preprint arXiv:2307.07645.

Julia Mendelsohn, Ceren Budak, and David Jurgens. 2021. Modeling framing in immigration discourse on social media. In Proceedings ofthe 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2219–2263, Online. Association for Computational Linguistics.

Nona Naderi and Graeme Hirst. 2017. Classifying frames at the sentence level in news articles. Policy, 9:4–233.

Viet-An Nguyen. 2015. Guided probabilistic topic models for agenda-setting and framing. Ph.D. thesis, University of Maryland, College Park.

Jianmo Ni, Gustavo Hernández Ábrego, Noah Constant, Ji Ma, Keith B Hall, Daniel Cer, and Yinfei Yang. 2021. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text models. arXiv preprint arXiv:2108.08877.

Susan Paykin, Moksha Menghaney, Q Lin, and M Kolak. 2021. Rural, suburban, urban classification for small area analysis. Healthy regions+ policies lab.—2021.—URL, 10.

Alejandro Portes and Min Zhou. 1993. The new second generation: Segmented assimilation and its variants. The annals ofthe American academy ofpolitical and social science, 530(1):74–96.

Roy Rada, Hafedh Mili, Ellen Bicknell, and Maria Blettner. 1989. Development and application of a metric on semantic nets. IEEE transactions on systems, man, and cybernetics, 19(1):17–30.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal ofMachine Learning Research, 21(1):5485–5551.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Nils Reimers and Iryna Gurevych. 2020. Making monolingual sentence embeddings multilingual using knowledge distillation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4512–4525, Online. Association for Computational Linguistics.

Christopher Michael Rytting, Taylor Sorensen, Lisa Argyle, Ethan Busby, Nancy Fulda, Joshua Gubler, and David Wingate. 2023. Towards coding social science datasets with language models. arXiv preprint arXiv:2306.02177.

David Sánchez, Montserrat Batet, David Isern, and Aida Valls. 2012. Ontology-based semantic similarity: A new feature-based approach. Expert systems with applications, 39(9):7718–7728.

Victor Sanh, Lysandre Debut, Julien Chaumond, and Thomas Wolf. 2019. Distilbert, a distilled version of bert: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.

Hong Shen, Tianshi Li, Toby Jia-Jun Li, Joon Sung Park, and Diyi Yang. 2023. Shaping the emerging norms of using large language models in social computing research. In Companion Publication ofthe 2023 Conference on Computer Supported Cooperative Work and Social Computing, pages 569–571.

Roberta A Sinoara, Jose Camacho-Collados, Rafael G Rossi, Roberto Navigli, and Solange O Rezende. 2019. Knowledge-enhanced document embeddings for text classification. Knowledge-Based Systems, 163:955–971.

Kai Sheng Tai, Richard Socher, and Christopher D. Manning. 2015. Improved semantic representations from tree-structured long short-term memory networks. In Proceedings of the 53rd Annual Meeting of the Associationfor Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1556– 1566, Beijing, China. Association for Computational Linguistics.

Petter Törnberg. 2023. Chatgpt-4 outperforms experts and crowd workers in annotating political twitter messages with zero-shot learning. arXiv preprint arXiv:2304.06588.

Isidora Tourni, Lei Guo, Taufiq Husada Daryanto, Fabian Zhafransyah, Edward Edberg Halim, Mona Jalal, Boqi Chen, Sha Lai, Hengchang Hu, Margrit Betke, Prakash Ishwar, and Derry Tanti Wijaya. 2021. Detecting frames in news headlines and lead images in U.S. gun violence coverage. In Findings of the

Associationfor Computational Linguistics: EMNLP 2021, pages 4037–4050, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368.

Zhiguo Wang, Haitao Mi, and Abraham Ittycheriah. 2016. Sentence similarity learning by lexical decomposition and composition. In Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers, pages 1340–1349, Osaka, Japan. The COLING 2016 Organizing Committee.

William Lloyd Warner and Leo Srole. 1945. The social systems of american ethnic groups.

Xing Wu, Chaochen Gao, Liangjun Zang, Jizhong Han, Zhongyuan Wang, and Songlin Hu. 2022. ESim-CSE: Enhanced sample building method for contrastive learning of unsupervised sentence embedding. In Proceedings ofthe 29th International Conference on Computational Linguistics, pages 3898– 3907, Gyeongju, Republic of Korea. International Committee on Computational Linguistics.

Min Zhou and Yang Sao Xiong. 2005. The multifaceted american experiences of the children of asian immigrants: Lessons for segmented assimilation. Ethnic and racial studies, 28(6):1119–1152.

Ganggao Zhu and Carlos A Iglesias. 2016. Computing semantic similarity of concepts in knowledge graphs. IEEE Transactions on Knowledge and Data Engineering, 29(1):72–85.

Caleb Ziems, Omar Shaikh, Zhehao Zhang, William Held, Jiaao Chen, and Diyi Yang. 2023. Can Large Language Models Transform Computational Social Science? Computational Linguistics, pages 1–53.

Caleb Ziems and Diyi Yang. 2021. To protect and to serve? analyzing entity-centric framing of police violence. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 957–976, Punta Cana, Dominican Republic. Association for Computational Linguistics.

## A Data Collection

## A.1 ACR Statistics

Include the number of ACRs per state with a huge table, including the total number and the sampled number. Also include a randomly picked ACR example along with the meta information we have.

<table><tr><td colspan="1" rowspan="1">state</td><td colspan="1" rowspan="1">sample_frequency</td><td colspan="1" rowspan="1">national_frequency</td></tr><tr><td colspan="1" rowspan="1">California</td><td colspan="1" rowspan="1">357</td><td colspan="1" rowspan="1">4265</td></tr><tr><td colspan="1" rowspan="1">New York</td><td colspan="1" rowspan="1">260</td><td colspan="1" rowspan="1">2979</td></tr><tr><td colspan="1" rowspan="1">Texas</td><td colspan="1" rowspan="1">165</td><td colspan="1" rowspan="1">1990</td></tr><tr><td colspan="1" rowspan="1">Pennsylvania</td><td colspan="1" rowspan="1">153</td><td colspan="1" rowspan="1">1830</td></tr><tr><td colspan="1" rowspan="1">New Jersey</td><td colspan="1" rowspan="1">130</td><td colspan="1" rowspan="1">1546</td></tr><tr><td colspan="1" rowspan="1">Florida</td><td colspan="1" rowspan="1">123</td><td colspan="1" rowspan="1">1596</td></tr><tr><td colspan="1" rowspan="1">Illinois</td><td colspan="1" rowspan="1">98</td><td colspan="1" rowspan="1">1152</td></tr><tr><td colspan="1" rowspan="1">Massachusetts</td><td colspan="1" rowspan="1">78</td><td colspan="1" rowspan="1">897</td></tr><tr><td colspan="1" rowspan="1">Ohio</td><td colspan="1" rowspan="1">73</td><td colspan="1" rowspan="1">971</td></tr><tr><td colspan="1" rowspan="1">Virginia</td><td colspan="1" rowspan="1">72</td><td colspan="1" rowspan="1">907</td></tr><tr><td colspan="1" rowspan="1">Michigan</td><td colspan="1" rowspan="1">66</td><td colspan="1" rowspan="1">783</td></tr><tr><td colspan="1" rowspan="1">Georgia</td><td colspan="1" rowspan="1">65</td><td colspan="1" rowspan="1">899</td></tr><tr><td colspan="1" rowspan="1">North Carolina</td><td colspan="1" rowspan="1">65</td><td colspan="1" rowspan="1">931</td></tr><tr><td colspan="1" rowspan="1">Washington</td><td colspan="1" rowspan="1">63</td><td colspan="1" rowspan="1">707</td></tr><tr><td colspan="1" rowspan="1">Maryland</td><td colspan="1" rowspan="1">58</td><td colspan="1" rowspan="1">723</td></tr><tr><td colspan="1" rowspan="1">Alaska</td><td colspan="1" rowspan="1">46</td><td colspan="1" rowspan="1">68</td></tr><tr><td colspan="1" rowspan="1">Connecticut</td><td colspan="1" rowspan="1">45</td><td colspan="1" rowspan="1">526</td></tr><tr><td colspan="1" rowspan="1">Hawaii</td><td colspan="1" rowspan="1">45</td><td colspan="1" rowspan="1">169</td></tr><tr><td colspan="1" rowspan="1">District Of Columbia</td><td colspan="1" rowspan="1">43</td><td colspan="1" rowspan="1">124</td></tr><tr><td colspan="1" rowspan="1">New Mexico</td><td colspan="1" rowspan="1">43</td><td colspan="1" rowspan="1">133</td></tr><tr><td colspan="1" rowspan="1">Missouri</td><td colspan="1" rowspan="1">42</td><td colspan="1" rowspan="1">619</td></tr><tr><td colspan="1" rowspan="1">Colorado</td><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1">480</td></tr><tr><td colspan="1" rowspan="1">Rhode Island</td><td colspan="1" rowspan="1">41</td><td colspan="1" rowspan="1">150</td></tr><tr><td colspan="1" rowspan="1">Arizona</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">464</td></tr><tr><td colspan="1" rowspan="1">Idaho</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">117</td></tr><tr><td colspan="1" rowspan="1">Louisiana</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">353</td></tr><tr><td colspan="1" rowspan="1">Montana</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">75</td></tr><tr><td colspan="1" rowspan="1">New Hampshire</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">177</td></tr><tr><td colspan="1" rowspan="1">Oregon</td><td colspan="1" rowspan="1">40</td><td colspan="1" rowspan="1">407</td></tr><tr><td colspan="1" rowspan="1">Arkansas</td><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">199</td></tr><tr><td colspan="1" rowspan="1">Indiana</td><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">474</td></tr><tr><td colspan="1" rowspan="1">Maine</td><td colspan="1" rowspan="1">39</td><td colspan="1" rowspan="1">148</td></tr><tr><td colspan="1" rowspan="1">Delaware</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">138</td></tr><tr><td colspan="1" rowspan="1">Kentucky</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">294</td></tr><tr><td colspan="1" rowspan="1">Nevada</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">287</td></tr><tr><td colspan="1" rowspan="1">West Virginia</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">140</td></tr><tr><td colspan="1" rowspan="1">Wyoming</td><td colspan="1" rowspan="1">36</td><td colspan="1" rowspan="1">53</td></tr><tr><td colspan="1" rowspan="1">Kansas</td><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1">263</td></tr><tr><td colspan="1" rowspan="1">South Dakota</td><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1">55</td></tr><tr><td colspan="1" rowspan="1">Utah</td><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1">200</td></tr><tr><td colspan="1" rowspan="1">Vermont</td><td colspan="1" rowspan="1">35</td><td colspan="1" rowspan="1">82</td></tr><tr><td colspan="1" rowspan="1">Alabama</td><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">370</td></tr><tr><td colspan="1" rowspan="1">Minnesota</td><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">415</td></tr><tr><td colspan="1" rowspan="1">Nebraska</td><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">178</td></tr><tr><td colspan="1" rowspan="1">Tennessee</td><td colspan="1" rowspan="1">34</td><td colspan="1" rowspan="1">453</td></tr><tr><td colspan="1" rowspan="1">North Dakota</td><td colspan="1" rowspan="1">33</td><td colspan="1" rowspan="1">41</td></tr><tr><td colspan="1" rowspan="1">Oklahoma</td><td colspan="1" rowspan="1">33</td><td colspan="1" rowspan="1">306</td></tr><tr><td colspan="1" rowspan="1">Iowa</td><td colspan="1" rowspan="1">31</td><td colspan="1" rowspan="1">294</td></tr><tr><td colspan="1" rowspan="1">Mississippi</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">202</td></tr><tr><td colspan="1" rowspan="1">South Carolina</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">489</td></tr><tr><td colspan="1" rowspan="1">Wisconsin</td><td colspan="1" rowspan="1">30</td><td colspan="1" rowspan="1">429</td></tr></table>

## A.1.1 Data Filtering

We first filtered out any restaurant that has the phrase ’fast food’,’chain’ or ’express’ in its google maps subtype, description, name or review. Then we curated a list of well-known chain Chinese restaurants in the U.S. to filter out the remaining chain restaurants that didn’t mark themselves as chain.

## A.2 NameFrames Dataset

## A.2.1 ACR Distribution across States

We categorized each restaurant as urban (2), suburban (1), or rural (0) based on its ZIP code. We developed this categorization using Paykin et al. (2021)’s guidelines for area classification.

![](images/2bb98ce3f0a11e94b8d37505179f91283df3c901da5dfaba57e226a7a1b832ad.jpg)  
Figure 4: Caption

## A.3 Codebook

Here are the specific changes we have made to Chen (2018)’s 9 name frames including (1) Personal Names, (2) Speciality, (3) Superlatives and Other Positive Modifiers, (4) Cultural Association, (5) Place Names, (6) Ambiance, (7) Others, (8) Transliterated Hanzi, (9) Express Takeout. First, we renamed (3) to Positivity, (4) to Culture, (5) to Location, and (8) to Romanization for simplicity in language. Second, we added a rule to include intimacy words related to family or relationship to the Ambiance frame, since we found that many ACRs like to use these words during our annotation. Third, we replaced the Others frame with a more specific Creativity frame and extended its definition to names with creative puns such as “Wok N’ Roll”. Fourth, we removed (9) in our analysis because the names of express ACRs tend to be highly repetitive and we excluded them in our analysis.

## A.3.1 English

• Personal names: usually, but not necessarily, surnames or first names. Here are some common examples:

– Surnames with a possessive: Qing’s Kitchen, Hoy’s Wok, P.F. Chang’s China Bistro

– Names without a possessive: China Lee, Hunan Mao, House of Louie, Mei Mei Dumplings

– Surnames with a form of address or courtesy title: Mr Chow, Madame Wu’s Asian Bistro, MaMa Lu’s Dumpling House

– Nicknames for family members or acquaintance: Ah-X, Liu Po

– English names: Paul’s Kitchen, House ofBenny, Bebe Fusion, Sam’s Chinese BBQ

## • Specialty: refers to certain food or dishes for which some restaurants are known for. Here are some common examples:

– Names with special meat, seafood, and tofu: Duck Queen, Fishball House, Little Fat Sheep, Seafood Village, Tofu House

– Names with BBQ and buffet: Sam’s Chinese BBQ, New Panda Buffet

– Names of tea-drinking restaurants: Green Tea House, Boba Tea Station, Tea Time

– Names with Dim Sum, Yum Cha, and Dim Sum dishes: Dim Sum Express, Yum Cha, Dumpling House, Tasty Bao, Egg Roll King

– Names with hot pot and pot: Mongolian Hot Pot, China Pot King

– Names with rice, noodles, congee: Noodle House, Rice N Noodles, The Congee, Porridge House

– Names with religious meanings, vegetarian, and other cultures: China Halal, Happy Veggie Garden, Buddha’s Feast

– Names with wok, snacks: China Wok, Rice Wok, Beijing Snacks

## • Superlatives and Other Positive Modifiers: refers to using superlatives and positive words in the names. Here are some examples:

– Names with a superlative: China Best Express, China Great Buffet, Perfect Chinese Food, Elite, The Legendary, No. 1 Noodle House

– Names with modifiers denoting superb taste: Delicious Food Corner, Tasty Wok Cuisine, Yummy Cafe, Luscious Dumplings, Chinese Gourmet Kitchen, Fresh China Kitchen

– Names with gold, golden, silver, diamond, jade: Gold Medal Chinese Food, Golden China, Silver Palace, Shanghai Diamond Garden

– Names with modifiers denoting good luck such as 8: 888 Fast Food, Lucky Seafood

– Names with words that elicit positive expectations: Happy Family, House of Joy, Sunny Palace, Pleasure Ocean, Nice Time Cafe

– Names with words that exude opulence: Grand Buffet, Feast from the East, Jumbo Chinese Fast Food

– Names with royalty implications: China King, King Fu, Kingchops, Empress Pavilion, Empire Palace, Regent China Inn, Royal Spring, VIP, The Dynasty

– Names with OK: OK Chinese Food

– Names with red: Red House

## • Ambiance: refers to a kind of atmosphere which makes customers feel comfortable and enjoy the dining experience, or it often indicates family or friendship, or being homesick, which shortens the psychological distance between the restaurant and the patrons. Here are some common examples:

– Names with the word “garden” that conveys a sense of tranquility and leisurely refinement: Garden and Ding’s, Mandarin Garden, Tasty Garden, New Garden

– Name with the word “palace,” which suggests the image of imperial-style fine dining: China Palace, Rowland Palace, Silver Palace

– Names that describe home-cooked meals in a more casual setting with words like “house,” “home,” “kitchen,” or “dining room”: House ofBenny, Joy House, Moon House, New RJ Chinese Kitchen, Tao Chinese Kitchen, Chinese Kitchen, Spicy Home, Yang’s Dining Room

– Names that evoke images of a simpler life or country living, like “hut”, “village”, “inn”, “bistro”: Boba Tea Hut, Green Village, Hunan Village, Bamboo Inn, Plum Tree Inn, Wok Inn, Madame Wu’s Asian Bistro, New Canton Bistro

– Names that contain family members: Brother’s Chinese Restaurant, Uncle Tom’s Hot Pot, Grandma’s Pancake, Mama’s Kitchen

– Names suggesting friendship: Friend’s house, Friends

– Names that remind patrons of their hometown: Hometown Szechuan, Home Chinese Cafe

• Creativity: refers to names that contain puns or words used creatively. Here are some examples:

– "wok with me" (sounds like “walk with me”), "pho king good" (sounds like “fucking good”), "rice to meet you" (sounds like “nice to meet you”), "Wok N Talk" (’wok’ ryhmes with ’talk’), etc

• Culture: refers to the Chinese, American, or other cultural symbols and references that appear in the names of restaurants. Here are some common examples:

– Names that contain animals or mystical creatures native to China: Panda House, Dragon Garden, Phoenix tavern

– Names that contain famous historical monuments: Great Wall, Forbidden City Restaurant, Golden Gate

– Names that contain plants native to China: Bamboo Place, Plum Blossom

– Names that contain famous historical or well-known figures: Jackie Chen’s Place

– Names that contain the name of a specific Dynasty name: the Tang Dynasty, the Song Dynasty

– Names that contain cultural artifacts: Lantern Garden, Silk House

• Location: Names that refer to Asia, China, or specific regions in China or other places. Here are some common examples:

– Names that refer to China, Asia, or other Asian countries: Chinese restaurant, Mandarin Garden, Asia Buffet, Chinese and Thai cuisine, Cathay House

– Names that refer to a specific province or city: Xi’an Noodles, ShanDong Garden, Beijing House, Hong Kong restaurant, Shanghai and Tokyo, Canton Bistro

– Names that contain the name of a town or street where the restaurant is located in: Wellesley Square sushi buffet, Brookline Wok

## • Romanized: containing Romanized Chinese characters that are not names of a person or names that indicate relationships. Here are some common examples:

– Examples: Lu Fu Zi, Wei Shi Jia, Shu Da Xia restaurant, Hai Di Lao Garden, Din Tai Fung, Yi Mei Deli

## A.3.2 Chinese

## Original Chinese

<sub>•</sub> 人名<sub>:</sub> 包括姓氏，名字，全名或者昵称。下面是一些例子<sub>:</sub>

– 餐馆名带有姓氏：林记烧腊，李记面馆，王家园，张师傅厨房

– 餐馆名带有名字：亚轩十八，威威，嘉嘉海鲜菜馆

– 餐馆名带有全名：李明餐厅，刘一手火锅

– 餐馆名带有昵称：六婆串串香，阿庆东北饺子馆

<sub>•</sub> 特色<sub>:</sub> 一般指带有一个餐馆名有特点的菜系，食物类型或者饮食形式。下面是一些例子：

– 餐馆名带有肉类，海鲜，豆腐种类：北京烤鸭餐厅，小肥羊，海鲜渔村，韩式豆腐汤

– 餐馆名带有烧烤或者自助：老村长烧烤屋，巴西烤肉，老友撸串店，金钱豹自助餐厅

– 餐馆名带有饮茶奶茶：时光奶茶

– 餐馆名带有点心：翠华茶餐厅，乐茶轩，一点心

– 餐厅名带有包子饺子：南翔小笼包，东北饺子馆

– 餐厅名带有火锅或者特色锅煲：蒙古火锅，砂锅居，成都冷锅，韩国海鲜豆腐煲

– 餐厅名带有米饭和面粉：小明炒饭，湖南米粉，绿色吴家粥铺，嘉和粥点，碰记热干 面

– 餐厅名带有伊斯兰特色，素食标签，或者其他文化标签：清真拉面，素食天地，阿曼尼萨汗美食城

– 餐厅名带有小吃：天津特色小吃，江浙小吃

<sub>•</sub> 最高级和其他积极的修饰语<sub>:</sub> 指代使用比较级，最高级的修饰语还有使用积极正能量的修饰词。以下是一些例子：

– 餐馆名指代好以及其比较级最高级的单词：京城第一，完美中餐厅，最好拉面店，顶好餐厅

– 餐馆名有指代味道好的修饰词：美味家园，好吃小馆，鲜芋仙，香喷喷厨房

– 餐馆名有金，银，玉，翡，翠，鼎，御, 宝等这些: 金山餐厅，聚宝盆，玉园，东方明珠

– 餐馆名有好运的修饰词比如发，<sub>8</sub>，八，好运：好运楼，发祥地，<sub>88</sub>酒家

– 餐馆名可以激起美好的期待：快乐家庭，阳光宫殿，好时光咖啡，圆满小厨，向日葵餐厅

– 餐馆名指代丰富富裕比如富，八，发，祥，嘉，聚，旺，和：盛大自助餐，富国餐 厅，东方盛宴

– 餐馆名指代皇家和尊贵：御尚坊，帝国餐厅，皇家特供，王朝餐厅，王品牛排

– 餐馆名包含<sub>OK</sub>：<sub>OK</sub>拉面餐馆名包含专业程度：大厨厨房，匠人厨房

– 餐馆名包含积极的意向比如红色，升日，花好月圆：红房子，日升餐厅

• 氛围: 包括暗示环境氛围的词语，一般作为地点的高级表达方式，给顾客一种舒适的体验。以下是一些例子：

– 餐馆名包含餐厅和厨房的高级表达方式：：南北小馆、新和小馆、食苑小厨、私房小厨、司膳房、御膳房

– 餐馆名包含古代高级地点词汇：美食轩、金鼎轩、凌波轩、湘德楼、海鲜酒家、川坝子酒楼、峨眉酒家、听涛阁、逐帆亭、望海楼、拾贝坊、采菱榭、御风堂、浣沙居

– 餐馆名包含带有艺术表达气息的地点名称：楼上花园、御香园、万秀园餐厅、熙春园餐厅

<sub>•</sub> 双关等创意题目<sub>:</sub> 包括双关、谐音等等各种有创意的修辞方法。以下是一些例子：

– 餐馆名包含双关或者谐音：四面八方（含有<sub>“</sub>面<sub>”</sub>，适合当面馆的名字），海食蜃楼（取自成语“海市蜃楼”），与食俱进（取自成语“与时俱进”），美食美客（取自成语“每时每刻”），天外飞鲜（取自成语“天外飞仙”）

– 餐馆名包含文化：老友记（电视剧名，也泛指老友相聚）、三味真火（“三味”可以指世间一切美味，也是红孩儿的看家本领；<sub>“</sub>真<sub>”</sub>字有真诚、真实的意思；<sub>“</sub>火<sub>”</sub>字寓意生意红火）、楚留香（本是人名，同时<sub>“</sub>留香<sub>”</sub>也意味着留下香气）

<sub>•</sub> 文化<sub>:</sub> 指餐厅名字中含有中华文化或美国文化符号或代表。以下是一些例子：

– 餐馆名包含动物：熊猫园，龙酒家<sub>,</sub> 凤凰茶楼

– 餐馆名包含历史古迹：长城饭店，天坛，金门

– 餐馆名包含植物：松鹤楼，梅花楼，绿竹园

– 餐馆名包含著名历史人物：小平饭店，秦始皇大酒楼

– 餐馆名包含朝代名称：唐朝饭店

– 餐馆名包含工艺品<sub>:</sub>大红灯笼，青瓷园

• 地点: 名字中含有与中国，亚洲，或其他地区、城市、街道的名字。以下是一些例子：

– 餐馆名包含中国或亚洲：中国餐厅，亚洲楼，华园，华明，中

– 餐馆名包含城市名：北京园，奉天小馆，川湘味，金粤园

– 餐馆名包含其他地方或街道名：曼哈顿小吃<sub>,</sub> 芬威楼

## English Translation

• Personal name: includes surname, first name, full name, or nicknames. Here are some examples:

– restaurant names containing surnames: Lin’s shaola, Lee’s Noodle restaurant, Wang’s garden, ChefZhang’s Kitchen

– restaurant names come with first names: Asia Pavillion 18, Weiwei, Jiajia Seafood Restaurant

– restaurant names with full name: Li Ming Restaurant, Liu Yishou Hotpot

– restaurant names with a nickname: Liu Po Delicious Skewers, Ah Qing Northeast Dumpling House

• Specialty: Generally refers to the cuisine, food types or dietary forms that are unique to a region or restaurant. Here are some examples

– Restaurant names include types of meat, seafood, and tofu: Beijing Roast Duck Restaurant, Little Sheep, Seafood Fishing Village, Korean Tofu Soup

– Restaurants with barbecue or buffet in their names: Old Village Chief’s BBQ House, Brazilian BBQ, Old Friend’s Skewers, Golden Leopard Buffet Restaurant

– restaurant names containing Yumcha or milk tea: Time Milk Tea

– Restaurant names indicating dim sum: Tsui Wah Tea Restaurant, Happy Tea House, One Dim Sum

– Restaurant names with steamed buns and dumplings: Nanxiang Steamed Dumplings, Northeast Dumpling House

– restaurant names containing hot pot or special pot: Mongolian hot pot, clay pot, Chengdu cold pot, Korean seafood tofu pot

– Restaurant names with rice and flour: Xiaoming Fried Rice, Hunan Rice Noodles, Green Wu’s Porridge Shop, Jiahe Porridge and Snacks, Pengji Hot Dry Noodles

– restaurant names with religous indications, vegetarian labels, or other cultural labels: Halal Ramen, Vegetarian World, Oman Nisa Khan Food City.

– restaurant names containing snacks: Tianjin special snacks, Jiangsu and Zhejiang snacks • Superlatives and other positive modifiers: refers to the use of comparatives, superlative modifiers and modifiers that use positive energy. Here are some examples:

– Restaurant names refer to good and its comparative superlative words: No. 1 in Beijing, Perfect Chinese restaurant, Best Ramen Shop, Top Good Restaurant

– restaurant names with modifiers that refer to good taste: Delicious Home, Delicious Bistro, Fresh Taro Fairy, Fragrant Kitchen

– Restaurant names include gold, silver, jade, jade, emerald, tripod, imperial, treasure, etc.: Jinshan Restaurant, Cornucopia, Jade Garden, Dongfang Ming beads

– restaurant names with good luck modifiers such as 8, Good Luck: Good Luck Building, Lucky Origin, 88 Restaurant

– Restaurant names that can arouse good expectations: Happy Family, Sunshine Palace, Good Time Coffee, Perfect Kitchen, Sunflower Restaurant

– Restaurant names refer to richness and opulence such as "rich", "fortune", "lucky", "beautiful", "Union", "prosperous", and: Shanda Buffet, Rich Country Cuisine Hall, Oriental Feast

– Restaurant names refer to royalty and nobility: Royall Kitchen, Empire Restaurant, Royal Special, Dynasty Restaurant, King Steak

– restaurant names containing "OK": OK Ramen

– restaurant name contains professional level: ChefKitchen, Artisan Kitchen

– Restaurant names containing positive connotations such as red, rising sun, flowers and full moon: Red House, Sunrise Restaurant

• Atmosphere: Including words that imply the atmosphere of the environment, generally used as a high-level expression of a place to give customers a comfortable experience. Here are some examples:

– Restaurant names include advanced expressions of restaurants and kitchens: North and South Bistro, New Peace Bistro, Snack Garden Kitchen, Private Kitchen, ROyal Kitchen, Imperial Kitchen

– The names of restaurants include ancient high-end location words: Delicacy Pavillion, Golden Ding Pavillion,WavesHouse, Hunan Virtue Building, Seafood Restaurant, Chuanbazi Restaurant, Emei Restaurant, Hearing Waves Pavilion, Chasing Boat Pavilion, Seaview Tower, Shell Picking Bistro, Lotus Picking Pavilion, Royal Wind Hall, Gauze Washing Home

– Restaurant names include place names with artistic expression: Upstairs Garden, Royal Fragrance Garden, Million Beauty Garden Restaurant, Bright Spring Garden Restaurant

• Creative topics such as puns: including puns, homonyms and other creative rhetorical methods. Here are some examples:

– Restaurant names contain puns or homophones: Four Directions and Eight Places (In the Chinese idiom Si Mian Ba Fang, whose literal translation if "Four Directions and Eight Places", the character of Mian is the same as the character for noodle, hence making the name suitable for a noodle restaurant), Food Mirage (taken from the Chinese idiom "mirage"), Advance with Food (taken from the Chinese idiom "advance with the times"), Gourmetfood and delicious food (Romanization "Mei Shi Mei Ke") (from the idiom "every moment"(Romanization "Mei Shi Mei Ke"). The Chinese characters in the idiom are replaced with characters of similar pronunciation in the restaurant name), Freshness from the Sky (from the Chinese idiom "Tianwai Feixian")

– restaurant names containing cultural or media symbols: Friends (the name of the TV series, also generally refers to the gathering of old friends), Three Zhen Fires ("Sanwei" can refer to all the delicious food in the world, and is also Honghaier’s special skill; the word "zhen" means sincerity and authenticity ;The word "fire" means business is booming), Chu Liuxiang (originally a person’s name, and "Liuxiang" also means leaving a fragrance)

• Culture: Refers to restaurants whose names contain symbols or representatives of Chinese culture or American culture. Here are some examples:

– Restaurant names include animals: Panda Garden, Dragon Restaurant, Phoenix Tea House

– The restaurant name includes historical monuments: Great Wall Hotel, Temple of Heaven, Golden Gate

– The restaurant name contains plants: Songhe Tower, Plum Blossom Tower, Green Bamboo Garden

– The restaurant names include famous historical figures: Xiaoping Hotel, Qin Shihuang Restaurant

– The restaurant name contains the name of the dynasty: Tang Dynasty Hotel

– The restaurant name contains handicrafts: Red Lantern, Celadon Garden

• Location: The name contains names related to China, Asia, or other regions, cities, and streets. Here are some examples:

– restaurant names containing words like "Chinese" or "Asian": Chinese Restaurant, Asia House, China Garden, China Bright

– restaurant names include city names: Beijing Garden, Fengtian Restaurant, Sichuan and Hunan Flavor, Jinyue Garden

– Restaurant names contain other place or street names: Manhattan Eats, Fenway House

<table><tr><td></td><td colspan="2">Chinese</td><td colspan="2">English</td></tr><tr><td></td><td>Cohen&#x27;s κ</td><td>Fleiss κ</td><td>Cohen&#x27;s κ</td><td>Fleiss κ</td></tr><tr><td>Personal Name</td><td>0.8235</td><td>0.8240</td><td>0.812</td><td>0.8119</td></tr><tr><td>Specialty</td><td>0.9034</td><td>0.9032</td><td>0.8294</td><td>0.8297</td></tr><tr><td>Positivity</td><td>0.7989</td><td>0.7989</td><td>0.8011</td><td>0.8002</td></tr><tr><td>Culture</td><td>0.8387</td><td>0.8381</td><td>0.8545</td><td>0.8542</td></tr><tr><td>Location</td><td>0.8856</td><td>0.8858</td><td>0.935</td><td>0.935</td></tr><tr><td>Ambiance</td><td>0.8788</td><td>0.8787</td><td>0.8331</td><td>0.8337</td></tr><tr><td>Creativity</td><td>0.8935</td><td>0.8928</td><td>0.8543</td><td>0.8541</td></tr><tr><td>Romanization</td><td>一</td><td>-</td><td>0.7822</td><td>0.7841</td></tr></table>

Table 4: This table shows the inter-annotator agreement (IAA) scores in Cohen’s Kappa and Fleiss Kappa on the English and Chinese names of 1000 ACRs. Romanization is not applicable to Chinese names, so we used “-”.

## A.3.3 Frame Annotation Agreement

Table 4 reports the inter-annotator agreement (IAA) scores among three annotators for each frame in Cohen’s Kappa and Fleiss Kappa on the English and Chinese names of 884 ACRs. Overall, we observe high IAA scores across 8 different frames.

## B Models

## B.1 Rule-based Method

For most of the categories, we developed a list of vocabulary for each and labeled the restaurant names with regex. Here are the lexicons.

• Chinese

– Location:“中”,"亚","华","北 京","香 港","上 海","川","湘","陕","山 东","粤","广 东","杭 州<sub>","</sub>南京<sub>","</sub>西安<sub>","</sub>重庆<sub>","</sub>成都<sub>","</sub>桂林<sub>"</sub>

<sub>–</sub> C<sub>ulture:"</sub>熊 猫<sub>","</sub>龙<sub>","</sub>凤<sub>","</sub>长 城<sub>","</sub>天 坛<sub>","</sub>金 门<sub>","</sub>松<sub>","</sub>鹤<sub>","</sub>梅 花<sub>","</sub>竹<sub>","</sub>小 平<sub>","</sub>秦 始 皇","汉","唐","宋","灯笼","瓷","丁香","荷","菊","功夫","丝绸"

– Ambiance<sub>:"</sub>小馆<sub>","</sub>厨<sub>","</sub>膳房<sub>","</sub>轩<sub>","</sub>楼<sub>","</sub>阁<sub>","</sub>亭<sub>","</sub>坊<sub>","</sub>榭<sub>","</sub>堂<sub>","</sub>居<sub>","</sub>园<sub>","</sub>家<sub>","</sub>宫<sub>","</sub>庭<sub>","</sub>村<sub>","</sub>乡<sub>",</sub> "屋","苑","吧","城","府","堂","屯","","庄","湾"

– Specialty<sub>:"</sub>烤<sub>","</sub>羊<sub>","</sub>牛<sub>","</sub>猪<sub>","</sub>鸡<sub>","</sub>鸭<sub>","</sub>鹅<sub>","</sub>海 鲜<sub>","</sub>豆 腐<sub>","</sub>串<sub>","</sub>自 助<sub>","</sub>茶<sub>","</sub>点 心<sub>","</sub>包 子<sub>","</sub>饺<sub>","</sub>锅<sub>","</sub>煲<sub>","</sub>饭<sub>","</sub>粉<sub>","</sub>粥<sub>","</sub>面<sub>","</sub>清 真<sub>","</sub>素 食<sub>","</sub>小 吃<sub>","</sub>馍<sub>","</sub>烧 烤<sub>","</sub>寿 司<sub>","</sub>腊<sub>","</sub>卤<sub>","</sub>辣<sub>","</sub>饼<sub>","</sub>烫<sub>","</sub>烩<sub>","</sub>鱼<sub>","</sub>米<sub>","</sub>渔<sub>","</sub>汤<sub>"</sub>

– Positivity<sub>:"</sub>第一<sub>","</sub>明<sub>","</sub>美<sub>","</sub>喜<sub>","</sub>好<sub>","</sub>鲜<sub>","</sub>香<sub>","</sub>金<sub>","</sub>银<sub>","</sub>玉<sub>","</sub>翡<sub>","</sub>翠<sub>","</sub>鼎<sub>","</sub>御<sub>","</sub>宝<sub>","</sub>八<sub>","</sub>发<sub>",</sub> "8","福","乐","满","祥","嘉","聚","旺","和","盛","富","御","帝","皇","王","鸿","冠","太 子<sub>","</sub>怡<sub>","</sub>佳<sub>","</sub>兴<sub>","</sub>升<sub>","</sub>锦<sub>","</sub>雅<sub>","</sub>顺<sub>","</sub>来<sub>","</sub>欢<sub>","</sub>悦<sub>","</sub>丰<sub>","</sub>隆<sub>","</sub>万<sub>","</sub>腾<sub>","</sub>丽<sub>","</sub>仁<sub>","</sub>朝 代","豪","强","神","仙","珍","荣","靓","超","利","棒","财","师","寿"

## • English

– Location:"chinese", "mandarin", "asia", "cathay", "china","Xi’an", "ShanDong", "Hunan", "beijing", "Hong Kong", "shanghai", "tokyo", "canton"

– Culture:"Panda", "Dragon", "Phoenix","Great Wall", "Forbidden City", "Golden Gate","Bamboo", "Plum",’lotus’,’tang’,’song’,’han’,’Jackie Chen’,’xiaoping

– Ambiance:’house’, ’home’, ’kitchen’, ’dining room’,"Brother", "Uncle", "Grandma", "Mama","friend", "friendship","Hometown", "home",’hut’, ’village’, ’inn’, ’bistro’,’garden’,’pavillion’,’palace

– Specialty:"Duck", "Fishball", "Sheep", "Seafood", "Tofu", "BBQ", "Buffet", "Green Tea", "Boba Tea", "Tea", "Dim Sum", "Yum Cha", "Dumpling", "Bao", "Egg Roll", "Hot Pot", "Pot", "Noodle", "Rice", "Noodles", "Congee", "Porridge", "Halal", "Veggie", "Vegan", "Kosher", "Wok", "Rice", "Snacks"

– Positivity:’best’, ’great’, ’perfect’, ’elite’, ’legendary’, ’No.1’, ’OK’, ’super’,’Delicious’, ’Tasty’, ’Yummy’, ’Luscious’, ’Gourmet’, ’Fresh’,’gold’, ’golden’, ’silver’, ’diamond’, ’jade’,’specialist’, ’master’, ’artisan’,’8’, ’lucky’, ’fortune’,’Happy’, ’Joy’, ’Sunny’, ’Pleasure’, ’Nice’,’Grand’, ’Feast’, ’Jumbo’,’King’, ’Empress’, ’Empire’, ’Regent’, ’Royal’, ’VIP’, ’Dynasty’,’red’

– Creativity:’with’,’to’,’and’,’in’,’out’

For the categories that can not be captured through keywords, we devised different methods for each. The Chinese Personal\_Name category is detected using a Chinese NER model that identifies the names of people in Mandarin. The Chinese Creativity category is detected through comparing a name with well-known Chinese proverbs, and a name is considered creative if its similarity score given by the model is above 0.8, which means that it resembles a Chinese proverb. The English Personal\_Name category is detected through a NER model as well, and the English Romanization category is detected using landid, a language identification tool.

## B.2 LLM-based Method

Here we provide an example of the name frame Ambiance:

• Chinese (zero-shot)

你正在为美国中餐馆名称研究数据集进行二元名称框架编码。

请使用谨慎、稳定的研究编码判断。你可以在心里核对规则，但最终答案必须只包含一个数字。

任务：判断该餐厅名称是否属于<sub>“</sub>氛围<sub>”</sub>类别。

如果属于，返回<sub>1</sub>；如果不属于，返回<sub>0</sub>。

最终回答必须正好是一个字符：<sub>0</sub>或<sub>1</sub>。

餐厅名称：<sub>{</sub> <sub>CN\_NAME</sub> <sub>}</sub>

<sub>“</sub>氛围<sub>”</sub>的定义：氛围指名称唤起就餐环境、舒适感、待客感、家庭感、友情、家的感觉、乡愁、悠闲雅致、热闹感或适合用餐的邀请感。

只要名称中任一部分明确符合该类别，整个名称就编码为<sub>1</sub>。

只使用定义进行判断；不要依赖示例，因为此提示不提供示例。

只回答0或1。

• Chinese (few-shot)

你正在为美国中餐馆名称研究数据集进行二元名称框架编码。请尽量复现论文中的人工标注标准。

任务：判断该餐厅中文名是否属于<sub>“</sub>氛围<sub>”</sub>类别。

餐厅名称：<sub>{</sub> <sub>CN\_NAME</sub> <sub>}</sub>

最终回答必须只包含一个字符：<sub>0</sub>或<sub>1</sub>。

如果属于<sub>“</sub>氛围<sub>”</sub>，回答<sub>1</sub>；如果不属于，回答<sub>0</sub>。

研究编码规则：

标为<sub>1</sub>：名称中出现明确的餐厅空间感、就餐环境、亲切感、家庭感、友情、乡愁、雅致感、热闹感或传统店铺意象。

标为<sub>0</sub>：名称只是地名、菜系、菜品、文化符号、人名<sub>/</sub>姓氏、普通品牌音译，或只是吉祥积极词但没有空间<sub>/</sub>氛围含义。

稳定标为1的线索：

<sub>-</sub>店铺空间：楼、酒楼、酒家、阁、亭、轩、堂、居、府、坊、馆、小馆、屋、厨、厨房。

-环境场景：园、苑、宫、庭、村、乡、庄、湾、城、巷、吧、茶座。

-亲切氛围：家、老家、人家、故乡、朋友、相聚、团圆、暖、雅、闲。

稳定标为0的线索：

<sub>-</sub>只有普通业态词：餐馆、餐厅、饭店，如果没有其他氛围线索，标<sub>0</sub>。

<sub>-</sub>纯菜品或烹饪方式：面、饺子、火锅、烧烤、香锅、外卖、快餐。

<sub>-</sub>只有地名、文化符号或积极词：北京、四川、龙、熊猫、福、金、旺。

以下线索和例子并不穷尽；请把规则应用到完整餐厅名。

正例：

<sub>-</sub> 松月小馆：<sub>1</sub>

<sub>-</sub>竹里雅轩：<sub>1</sub>

-云灯食坊：1

<sub>-</sub> 暖家厨房：<sub>1</sub>

反例：

-川椒面点：0

<sub>-</sub> 金福龙：<sub>0</sub>

\- 台北牛肉面：0

\- 王师傅水饺：0

请按以上规则判断，只回答0或1。

• Engilish (zero-shot)

You are coding Chinese restaurant names in the United States for a binary name-frame dataset. Apply the coding rubric consistently. Think through edge cases silently, then return a single-token answer: 0 or 1.

Task: decide whether the restaurant name belongs to the Ambiance category.

Return 1 if yes. Return 0 if no.

The final response must be exactly one character: 0 or 1.

Restaurant name: {EN\_NAME}

Definition of Ambiance: Ambiance refers to words that evoke dining atmosphere, comfort, hospitality, family, friendship, home, nostalgia, leisurely refinement, or an inviting place to eat.

If any part of the name clearly satisfies the category, code the entire name as 1.

Use the definition only; do not rely on examples because none are provided.

Answer only 0 or 1.

## • English (few-shot)

You are coding U.S. Chinese restaurant names for a research dataset.

Task: decide whether the English restaurant name belongs to the Ambiance category.

Restaurant name: {EN\_NAME}

Return exactly one character: 1 or 0. Rule: mark 1 when the name evokes a dining place, atmosphere, hospitality, family warmth, friendship, home, nostalgia, leisure, refinement, comfort, or an inviting place to eat.

High-confidence 1 cues:

\- inviting place words: garden, house, pavilion, villa, lounge, hall, inn

\- warmth or hospitality words: home, family, friendship, cozy, hearth, welcome

\- atmosphere or leisure words: moonlit, breeze, delight, elegant, peaceful, gathering

\- family member words: Mama, Uncle, Grandma, Auntie

High-confidence 0 cues:

\- cuisine or geography alone: Chinese, Asian, Beijing, Shanghai, Sichuan, Hunan

\- food or service format alone: noodle, dumpling, hot pot, buffet, express, takeout

\- positive/status words without atmosphere: lucky, golden, fortune, best, royal

The cues and examples given here are not exhaustive; apply the rule to the full restaurant name.

Independent examples:

\- Beijing Bistro: 1

\- Grandma’s Kitchen: 1

\- Peony Pavilion: 1

\- Mama Noodle: 1

\- Fengtian Bistro & Lounge: 1

\- Tianjin Pepper Noodle: 0

\- Chef Fong Express: 0

\- Auspicious Gold Wok: 0

\- Pacific Soy Buffet: 0

Answer only 0 or 1.

## B.3 Full Model Performance

The Table 2 is an aggregated version of Table 5 below. Table 5 shows the detailed performance of different LLM models with three prompting strategies.

<table><tr><td>Model</td><td>Personal Name</td><td>Speciality</td><td>Positivity</td><td>Culture</td><td>Location</td><td>Ambiance</td><td>Creativity</td><td>Romanization</td></tr><tr><td colspan="9">Chinese Names (Weighted F1)</td></tr><tr><td>Population Avg.</td><td>94.93</td><td>95.45</td><td>88.13</td><td>81.07</td><td>91.50</td><td>86.65</td><td>96.30</td><td>一</td></tr><tr><td>Expert Rules</td><td>88.09</td><td>97.39</td><td>90.81</td><td>94.79</td><td>88.18</td><td>93.44</td><td>95.20</td><td>一</td></tr><tr><td>Qwen3.7-Max (zero-shot)</td><td>95.14</td><td>97.55</td><td>86.49</td><td>64.45</td><td>88.25</td><td>77.50</td><td>96.89</td><td>一</td></tr><tr><td>Qwen3.7-Max (few-shot)</td><td>94.99</td><td>98.86</td><td>92.38</td><td>94.15</td><td>92.80</td><td>96.08</td><td>96.46</td><td></td></tr><tr><td>Kimi K2.6 (zero-shot)</td><td>94.06</td><td>94.29</td><td>81.48</td><td>62.38</td><td>87.82</td><td>73.86</td><td>96.66</td><td></td></tr><tr><td>Kimi K2.6 (few-shot)</td><td>94.23</td><td>97.42</td><td>88.64</td><td>94.93</td><td>94.37</td><td>93.46</td><td>97.05</td><td>一</td></tr><tr><td>GPT-5.5 (zero-shot)</td><td>95.44</td><td>85.20</td><td>88.17</td><td>55.74</td><td>90.35</td><td>83.83</td><td>92.85</td><td></td></tr><tr><td>GPT-5.5 (few-shot)</td><td>96.01</td><td>98.02</td><td>92.68</td><td>95.38</td><td>93.99</td><td>94.71</td><td>96.88</td><td></td></tr><tr><td>Claude Sonnet 4.6 (zero- shot)</td><td>94.73</td><td>94.49</td><td>85.09</td><td>84.85</td><td>89.21</td><td>78.82</td><td>96.76</td><td></td></tr><tr><td>Claude Sonnet 4.6 (few- shot)</td><td>94.83</td><td>97.75</td><td>90.13</td><td>96.66</td><td>95.20</td><td>94.95</td><td>96.88</td><td></td></tr><tr><td colspan="9">English Names (Weighted F1)</td></tr><tr><td>Population Avg.</td><td>95.32</td><td>93.96</td><td>89.83</td><td>91.04</td><td>83.30</td><td>82.64</td><td>97.22</td><td>87.65</td></tr><tr><td>Expert Rules</td><td>85.83</td><td>85.83</td><td>82.46</td><td>93.95</td><td>92.02</td><td>91.43</td><td>74.22</td><td>72.42</td></tr><tr><td>Qwen3.7-Max (zero-shot)</td><td>91.19</td><td>96.52</td><td>88.72</td><td>84.86</td><td>77.28</td><td>72.83</td><td>98.27</td><td>78.55</td></tr><tr><td>Qwen3.7-Max (few-shot)</td><td>96.79</td><td>97.46</td><td>91.61</td><td>96.95</td><td>91.11</td><td>90.34</td><td>99.21</td><td>91.79</td></tr><tr><td>Kimi K2.6 (zero-shot)</td><td>92.95</td><td>88.59</td><td>89.04</td><td>86.66</td><td>68.97</td><td>66.75</td><td>97.27</td><td>85.11</td></tr><tr><td>Kimi K2.6 (few-shot)</td><td>96.27</td><td>96.72</td><td>89.41</td><td>97.64</td><td>91.61</td><td>88.44</td><td>97.60</td><td>90.57</td></tr><tr><td>GPT-5.5 (zero-shot)</td><td>93.49</td><td>89.36</td><td>86.63</td><td>83.54</td><td>78.26</td><td>87.04</td><td>94.67</td><td>83.19</td></tr><tr><td>GPT-5.5 (few-shot)</td><td>98.45</td><td>97.27</td><td>93.14</td><td>97.49</td><td>95.18</td><td>84.33</td><td>97.01</td><td>93.12</td></tr><tr><td>Claude Sonnet 4.6 (zero- shot)</td><td>95.80</td><td>88.46</td><td>87.62</td><td>84.84</td><td>77.56</td><td>85.42</td><td>95.74</td><td>84.78</td></tr><tr><td>Claude Sonnet 4.6 (few- shot)</td><td>97.62</td><td>97.27</td><td>92.46</td><td>96.35</td><td>86.46</td><td>85.95</td><td>97.98</td><td>94.08</td></tr></table>

Table 5: Performance of different approaches on Name Frame Detection in both English and Chinese ACR names. “Expert Rules” represents the rule-based system based on lexicons and heuristics. We report performance for Qwen3.7-Max, Kimi K2.6, GPT-5.5, and Claude Sonnet 4.6 under two prompt settings: (1) “zero-shot”, (2) “fewshot”, respectively, as described in Section 4.1.

## B.4 GPT Translation Guideline

Here is a list of rules to translate Chinese restaurant names to English:

1. Translate each name character by character. No need to care about English grammar. Some examples: ’西海楼’ ’West Sea House

<sub>’</sub>福满楼<sub>’ ’Luck Full House</sub>

<sub>’</sub>锦灿<sub>’</sub> <sub>’Golden</sub> <sub>Splendor</sub>

’二舅烤肉’ ’Second Uncle Barbecue’

2. If a character or a phrase has a Chinese meaning and doesn’t look like a person’s name, return the literal translation of this character or phrase instead of the Romanized version and please pay special attention to this.

3. Make sure there’s as little Romanization in the translations as possible. Here are some examples for your reference. For instance,

’<sup>明苑阁</sup>’ ’Bright Garden Pavillion

<sub>’</sub>川香园<sub>’</sub> <sub>’Sichuan</sub> <sub>Fragrant</sub> <sub>Garden</sub>

’那家老院子’ ’Na’s Old Yard’

’<sup>芳芳小食</sup>’ ’Fang Fang Snacks’

Make sure to refer to these examples when you do the translation.

Give me the literal english translation of the Chinese restaurant name {CN\_NAME}. Only return the english translations and don’t return the Chinese names. No explanation needed. No double or single quotes needed.

## B.5 Translation Model Selection

We compared text similarity scores between GPT-5.5, Claude Snonnet 4.6 translations, and Google Translate. Both AI models outperformed Google Translate. While both AI models achieved comparable

results, Claude 4.6 Sonnet achieved marginally higher scores in five out of nine metrics compared to GPT-5.5. Therefore, we decided to use Claude Snonnet 4.6 to translate the Chinese names of our sampled restaurants.
<table><tr><td></td><td>Google Translate</td><td>GPT-5.5</td><td>Claude Sonnet 4.6</td></tr><tr><td>Cosine</td><td>0.6550</td><td>0.7598</td><td>0.7706</td></tr><tr><td>BERT F1</td><td>0.9198</td><td>0.9404</td><td>0.9432</td></tr><tr><td>BERT Precision</td><td>0.9161</td><td>0.9417</td><td>0.9434</td></tr><tr><td>BERT Recall</td><td>0.9241</td><td>0.9392</td><td>0.9431</td></tr><tr><td>ROUGE-1</td><td>0.4722</td><td>0.5975</td><td>0.5920</td></tr><tr><td>ROUGE-2</td><td>0.3003</td><td>0.3692</td><td>0.3781</td></tr><tr><td>ROUGE-L</td><td>0.4628</td><td>0.5914</td><td>0.5829</td></tr><tr><td>ROUGE-Lsum</td><td>0.4628</td><td>0.5914</td><td>0.5829</td></tr><tr><td>BLEU</td><td>0.0220</td><td>0.0488</td><td>0.0424</td></tr></table>

Table 6: Translation quality comparison across Google Translate, GPT-5.5, and Claude Sonnet 4.6.

## C Interview Details

## C.1 Interview Procedure

We conducted semi-structured interviews with ten ACR owners. We refer to the participants from R1 to R10. Details for each respondent can be found in Table 7. We follow the guideline<sup>10</sup> from Massachusetts State Office of Rural Health to classify the areas. To attain a diverse sample, we employed a stratified sampling approach to ensure that we reach owners from both urban and rural areas. We also sampled restaurants based on whether their Mandarin and English names were a close match or a mismatch. R9 was the only restaurant that didn’t have a Mandarin name.

We reached out to the selected restaurants through walk-ins or calls. Each interview was held in Mandarin and lasted around 30 minutes to an hour. We asked owners how the Mandarin and English restaurant names were created. We also asked questions about personalizing the names to different customer groups, how the restaurant location impacted the naming process, and how the restaurant reflects aspects of the owner’s heritage and identity. Appendix C.2 contains our interview guide.

Since a majority of participants didn’t feel comfortable with recording the conversations, we had a designated note-taker record detailed notes in English, which we used to analyze the interviews. Three authors went through the notes and coded them for themes using affinity diagramming, a popular analysis method in user interface design (Lucero, 2015) where text data is segmented into notes and then clustered into groups. Through multiple iterations, along with periodic discussions with the rest of the research team, we identified several key themes that impact restaurant naming decisions.

C.2 Interview Guide
<table><tr><td colspan="1" rowspan="1">Goal</td><td colspan="1" rowspan="1">Interview Questions</td></tr><tr><td colspan="1" rowspan="1">Introductions</td><td colspan="1" rowspan="1">Where are you from? → get a sense of their personal story[OPTIONAL] Were you born in the U.S. or grew up in the U.S.?</td></tr><tr><td colspan="1" rowspan="1">History ofrestaurant</td><td colspan="1" rowspan="1">When did the restaurant open?How long have you been living in the U.S. before you opened the restaurant?How many owners has there been?Family owned?How did you pick the restaurant location?</td></tr><tr><td colspan="1" rowspan="1">Restaurantname originstory</td><td colspan="1" rowspan="1">What is the story behind the name of this restaurant (both Chinese andEnglish names)? → get a sense of whether it was one person's idea ormultiple people were involved in the naming process.[OPTIONAL] Did you consult with other people when naming therestaurant? If so, who and how did they contribute to the namingprocess?[OPTIONAL] Did you consult other sources (like books or the internet)when naming the restaurant? If so, what?[OPTIONAL] Ask this question regarding the English name: at thetime you came up with the restaurant name, what was your comfortlevel in English? [provide a scale]O  Zero [no English], beginner [basic words], intermediate [canhave a simple conversation in English], proficient [can useEnglish for work], nativeDid the name change over time?When did the names change?What was happening around the time that the names changed?What kind of feelings or images do you want potential customers to havewhen they see the restaurant name?If they only talked about one name (Chinese vs English), can ask about theother. If the restaurant doesn't have a Chinese name, ask: Why don't youhave a Chinese restaurant name?</td></tr><tr><td colspan="1" rowspan="1">Personalizingname todifferentcustomergroups</td><td colspan="1" rowspan="1">Selectively ask questions here based on information shared during thewarm-up; Focus on the presentWhat kinds of people visit your restaurant?</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Do you observe different behaviors or purchases made by the differentgroups?How do the differences between these groups impact decisions you made onthe restaurant?Ask about naming if it isn't naturally brought up: How do thesedifferences relate to the differences between the Chinese and Englishnames?Do you think the names are effective in appealing to different customergroups?</td></tr><tr><td colspan="1" rowspan="1">How does thelocationinfluences thename</td><td colspan="1" rowspan="1">Selectively ask questions here based on information shared during thewarm-upGot a sense of customers, curious to know about the greater communityaround your restaurant. This food tastes so good I bet people come fromreally far to eat while others are more local. Do you have a sense of wherepeople are coming from?How would you describe the community living around your restaurant?How has the location of the restaurant impacted the name, if it did at all?Location is important but today a lot of people order food online. Do you havea general understanding or perception of your online customers? If so,elaborateHow do you try to interact with your online customers?What other information do you want to know about your onlinecustomers?</td></tr><tr><td colspan="1" rowspan="1">Reflectionquestionsrelating topersonalexperiences</td><td colspan="1" rowspan="1">How does this restaurant reflect parts of your identity?How did your experiences &lt;some statement summarizing the answer to thewhere you are from question&gt; impact how you opened and currently run therestaurant?How do you think Chinese restaurants in this area are changing? What areyour hopes and concerns?</td></tr><tr><td colspan="1" rowspan="1">Closingquestions</td><td colspan="1" rowspan="1">If you were going to rename the restaurant (either the English or Chinesename), what would the new name(s) be? Would you actually change thename(s)? Why or why not?</td></tr></table>

<table><tr><td>Respondent</td><td>Gender</td><td>English Proficiency</td><td>Customer Base</td><td>Area</td><td>Opening Year</td><td>Name Match</td></tr><tr><td>R1</td><td>Female</td><td>Proficient</td><td>Weekdays: 20% Chinese; Weekends: 50%</td><td>Urban</td><td>2017</td><td>Close match</td></tr><tr><td>R2</td><td>Female</td><td>Advanced</td><td>50% Chinese</td><td>Urban</td><td>2011</td><td>Mismatch</td></tr><tr><td>R3</td><td>Female</td><td>Intermediate</td><td>45% Chinese</td><td>Urban</td><td>2000</td><td>Mismatch</td></tr><tr><td>R4</td><td>Female</td><td>Intermediate</td><td>Weekdays: 50% Chinese; Weekends: 75%</td><td>Urban</td><td>2003</td><td>Close match</td></tr><tr><td>R5</td><td>Male</td><td>Basic</td><td>80% Chinese</td><td>Urban</td><td>1993</td><td>Mismatch</td></tr><tr><td>R6</td><td>Male</td><td>Proficient</td><td>40% Chinese</td><td>Urban</td><td>2010</td><td>Mismatch</td></tr><tr><td>R7</td><td>Male</td><td>Fluent</td><td>50% Chinese</td><td>Urban</td><td>2016</td><td>Mismatch</td></tr><tr><td>R8</td><td>Male</td><td>Proficient</td><td>40% Chinese</td><td>Urban</td><td>2005</td><td>Close match</td></tr><tr><td>R9</td><td>Male</td><td>Intermediate</td><td>1% Chinese</td><td>Rural</td><td>2012</td><td>No Chinese name</td></tr><tr><td>R10</td><td>Male</td><td>Intermediate</td><td>1% Chinese</td><td>Rural</td><td>2000</td><td>Close match</td></tr></table>

Table 7: Information about 10 ACR respondents that we interviewed. Each respondent is an owner, co-owner, or founder of an ACR in Massachusetts. The gender, English proficiency, percentage of Chinese customers, whether the restaurant is located in an urban or rural area, and the closeness between the Mandarin and English names are reported.

## C.3 Expanded Interview Results

## C.3.1 Impact of Customer Group

All owners strategically think about their customer groups, especially in regards to age and cultural background, observing how their customers change over time. For example, R10 caters their food to the local rural population, which is less than 1% Chinese, while R5 caters their food to the international Chinese students who come to Chinatown and want to try new foods.

In many cases, the owners’ awareness of their customer base directly impacted their choice in Chinese and English restaurant names. ACR Mandarin names help attract Chinese-speaking customers. R2 chose their Chinese name to attract Chinese customers who enjoy certain regional dishes, which is made clear in the name. Further, R2 is well-aware that the mismatch in their Chinese and English names denotes two different regions of China, which sparks the curiosity of many of their Chinese customers; “Chinese customers. . . want to investigate and try it.” About “8-9% of young customers ask where the name is comingfrom.” All but two ACRs came up with their Chinese names first. R1 noted that “Chinese names are easy to understand, because the “language is very rich,” explaining that their particular Chinese name holds a double meaning.

Similarly, ACR English restaurant names attract largely American customers. In the case of R9, which only has an English name, there is “no need” for a Chinese name, as it is located in a rural area where “there are no Chinese customers.” Many ACR owners chose their English names because they are familiar and easy to understand for American customers. R2 decided upon their English name first, because the “name tells [American customers] it’s an Asian food.” R6, whose Chinese and English names are mismatches, said of their English name: “We want to accommodate for American taste and have to change to fit their palette.” R4 named their restaurant after a popular movie featuring Chinese characters, saying “it is good to have a simple name to be more acceptable by customers.” Some ACR owners caught onto Western trends of trying new cuisines, and named their restaurants in accordance with the trends. R6’s name brings up associations with a popular Chinese dish at a time where journalists were reporting on it, which “made American customers curious.” R8, whose two names are a close match, expects American customers to google the restaurant name and “be curious about the place.” And for some ACR owners like R1 and R8, the English name plays a more active role, acting as a means to correct misconceptions of Chinese restaurants in America as “cheap, not very good, not clean” (R1). R1’s goal was to “attract westerners, have them try the Chinese food and change their mind,” establishing R1 as a “social destination, notjust place to eat.” Accordingly, “Westerners can say the name” and “it sounds sweet” (R1).

## C.3.2 Impact of Location

Several ACRs highlighted the importance of opening a restaurant in a “good neighborhood” with “better school districts” (R2). R3 chose to open her restaurant near a prestigious university to attract students, researchers, and professors. Those universities have a large group of Chinese international students who regularly eat at Chinese restaurants, motivating the need for a Mandarin name. R1 opened her restaurant in an area with a lot of development, which happens to be close to her former workplace. She wanted to “create a place forformer co-workers to come and keep in touch”, which was one reason why she incorporated her personal name in both the Mandarin and English restaurant names.

We observed a different relationship between location and naming decisions in Boston’s Chinatown, which has the largest concentration of ACRs in Massachusetts. R7 shared that a restaurant in Chinatown has to be “unique to stand out so that they can have business. There is a lot ofcompetitionfor the same type offood, and other restaurants have been openfor longer.” R7 identified a type of Chinese food that was not being sold in Chinatown. He highlighted aspects of the food in both the Mandarin and English names of his restaurant to stand out from other options in the area. Chinatown is drastically different from rural areas, where there are very few ACRs. R9 decided to open a restaurant in a rural town to “escape Chinatown.” He and his partner picked their current location because it is “adventurous”, and they wanted to “start a new life with more freedom.” The newfound freedom is reflected in R9’s choice of naming the restaurant in memory of his daughter and deciding not to create a Mandarin name.

## C.3.3 Impact of Personal Experience

A few ACR owners shared how their personal experience, including their family, origin, and personality, influenced their choice of restaurant names. Both R3 and R9 included the names of their daughters to “show love”. R1 included her Chinese given name in the ACR name to memorize her dad who was “an artist who came up with the name” (R1). R1’s dad passed away in China while she was still studying in America and did not make it to see him as the only daughter. A few owners (R1, R2, R4) also mark their or their partners’ origin in the ACR names such as “Hunan”, “Taiwanese”. R1 shared that she did this to “introduce my hometown to this area” and expressed, “I miss myfamily in the hometown.” 6 out of 10 respondents are from Fuzhou City, which is a community has long been known for its significant presence in the Chinese restaurant industry in the United States. It is interesting to see that none of the six ACRs owned by Fuzhounese encode their origins in the names. R8, an active member from the Fuzhouese community, confirmed our observation that most ACRs in the state are owned by immigrants from Fuzhou City, Fujian Providence. At last, we found that some owners wanted the name to “be personal” (R1) and “become aform ofselfexpression” (R7). R7, who decorated and named his ACR in a unique and fashionable manner, shared that he “did not care if other people like the name” when he created and copyrighted the name. However, some owners such as R3 had a different opinion, sharing that she did not like “her identity to be encoded in the restaurant names.”

## D Prolific Setup

We recruited Prolific workers from around the world whose first language is Chinese with an approval rate between 99% and 100%. We divided our sample of 1956 data into 70 Qualtrics surveys, with each survey completed by two prolific workers. We began our survey with a series of instructions and asked that the workers answer in simplified Chinese characters. Then, the workers find the names of restaurants using the links we provided in the survey. Each survey contains 30 - 50 restaurants. Screenshots of the questions can be found below. All our questions were written in both simplified Chinese and English.

## D.1 Prolific Worker Demographics

There are 20 unique participants who took the surveys. They are based in the UK, Canada, New Zealand, and Germany. 12 of them were born in mainland China, 4 in Hong Kong, 2 in the UK, 1 in Malaysia, and 1 in Taiwan. The distribution of their age and sex can be found below.

Welcome! In this survey, we invite you to assist us in annotating the Chinese names of 50 restaurants located in the US. Your participation will require approximately 20 minutes and your personal information will remain confidential. Here is a brief guideline on how to proceed:

Figure 5: Qualtrics survey English instructions  
![](images/e8430801c1be70c4e62d4fca3934f67d5b790ae3eb814be0a55c6bc988fcf941.jpg)  
Figure 6: Qualtrics survey Chinese instructions

![](images/ec1eb19b7c8c6f1cc94a278515d84452cff6060ae1a2a69ad0f8524adbb7a1cc.jpg)  
Figure 7: Qualtrics survey response instructions

Our survey requires that you type your answers in simplified Chinese scripts. Please exit the survey now if you can't type in simplified scripts. 请您用简体中文作答，如果您不会用简体 拼写请退出问卷。如果您的回答里有繁体字我们会拒绝接受

1 of 50. Please find the corresponding Chinese name of the restaurant "Chinnie's Kitchen &Bar" and type the Chinese name into the bar below. 请通过以下链接查找 Chinnie's Kitchen&Bar餐厅中文名（可以使用谷歌地图餐厅图片，360全景功能（查看使用方法），或餐厅网站）

•Google Maps 谷歌地图

• Webiste 餐厅网站(有时可能没有餐厅网站)

No Chinese name found 无中文名

Enter Chinese name below有中文名，在下面方框內输入名字（如果餐厅名字是繁体也请用简体输

![](images/18722868b79eeb48a1eb01fc4e9728949383c38b6218b90c4a0132daf03329a7.jpg)  
Figure 8: Qualtrics survey sample question

![](images/c92a9497b5f5514da830ab4b6d3a152ee8cdb4f44c6b7e518922f18057c86100.jpg)  
(a) Prolific workers sex distribution

![](images/20d80aa92d01786cc5b3a8cb292d30e4ecf48ef36654ee83bc4febadc564c894.jpg)  
(b) Prolific workers age distribution  
Figure 9: Demographic characteristics of Prolific workers

## E Regression Analysis

## E.1 Multivariate regression

<table><tr><td rowspan="2">Predictor</td><td colspan="2">Jaccard Distance</td><td colspan="2">Cosine Distance</td></tr><tr><td> $\beta$  (SE)</td><td>p</td><td> $\beta$  (SE)</td><td>p</td></tr><tr><td>Intercept</td><td>0.5706 (0.010)</td><td>0.000</td><td>0.5165 (0.007)</td><td>0.000</td></tr><tr><td>Chinese population  $\%$  (scaled)</td><td>0.0128 (0.010)</td><td>0.199</td><td>-0.0085 (0.007)</td><td>0.195</td></tr><tr><td>Time since opening (scaled)</td><td>-0.0128 (0.010)</td><td>0.194</td><td>-0.0179 (0.007)</td><td>0.006</td></tr><tr><td>Location (rural)</td><td>-0.1682 (0.069)</td><td>0.014</td><td>0.0181 (0.045)</td><td>0.688</td></tr><tr><td>Location (Suburb)</td><td>-0.0388 (0.039)</td><td>0.323</td><td>-0.0360 (0.026)</td><td>0.165</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.008</td><td></td><td>0.007</td><td></td></tr><tr><td>Adj.  $R ^ { 2 }$ </td><td>0.005</td><td></td><td>0.004</td><td></td></tr><tr><td>F-statistic</td><td>2.989</td><td> $( p = 0 . 0 1 8 0 )$ </td><td>2.611  $( p = 0 . 0 3 4 0 )$ </td><td></td></tr><tr><td></td><td></td><td>1.04</td><td></td><td></td></tr><tr><td>Max VIF</td><td colspan="3"></td><td colspan="2"></td></tr></table>

Note: Continuous predictors were standardized $( \mathrm { m e a n } = 0 , \mathrm { S D } = 1 )$ before fitting. Coefficients therefore reflect the expected change in the outcome per one standard deviation increase in the predictor. For location factor, urban category is chosen as reference category as it is the dominate category $\left( \mathrm { n } = 1 , 3 4 6 \right)$ compared to suburb $( { \bf n } = \mathrm { i } 0 0 )$ and rural (n = 31).

Table 8: Multi-factor regression results

## E.2 Random Forest regressor

<table><tr><td>DV</td><td>CV  $R ^ { 2 }$ </td><td>Train / Test  $R ^ { 2 }$ </td></tr><tr><td>Jaccard distance</td><td>-0.0782</td><td> $- 0 . 0 1 6 4 / - 0 . 0 1 3 1$ </td></tr><tr><td>Cosine distance</td><td>-0.0040</td><td>-0.0294  $I - 0 . 0 7 6 1$ </td></tr></table>

Note: To address class imbalance (Rural $n = 3 1 .$ , Suburb $n = 1 0 0 ,$ , Urban $n = 1 , 3 4 6 ) .$ , we used a stratified traintest split and assigned higher sample weights to minority location categories. Hyperparameters were selected via crossvalidation to maximize $\scriptstyle { \dot { R ^ { 2 } } }$ , for both distance measures we used max\_depth = 3 and min\_samples\_leaf = 50.

Table 9: Random Forest regression results.