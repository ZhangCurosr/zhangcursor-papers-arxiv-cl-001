# PunGraph: Retrieval-Enhanced Phonetic-Semantic Graph Reasoning for Pun Understanding

Yuchen Su<sup>1</sup>, Zijian Huang<sup>1</sup>, Yaotian Shi<sup>1</sup>, Shaoxin Zhong<sup>1</sup>, Ruofan Wang<sup>1</sup>, Mengze Li<sup>1</sup>, Yonghua Zhu<sup>2</sup>\*, Diana Benavides-Prado<sup>3</sup>, Michael Witbrock<sup>1</sup>,

<sup>1</sup>School of Computer Science, University of Auckland, New Zealand

<sup>2</sup>School of Computer and Information Technology, Shanxi University

<sup>3</sup>School of Electronic Engineering and Computer Science, Queen Mary University of London {ysu132,zhua764}@aucklanduni.ac.nz, zhuyonghua@sxu.edu.cn d.benavidesprado@qmul.ac.uk, m.witbrock@auckland.ac.nz

## Abstract

Puns are a challenging form of figurative language that exploit phonetic similarity and semantic ambiguity to convey multiple meanings. Although large language models (LLMs) demonstrate strong language understanding capabilities, they still struggle with pun reasoning due to limited phonetic modeling and uncontrolled end-to-end generation. We propose Pun-Graph, a retrieval-enhanced knowledge graph framework for pun understanding. PunGraph constructs a phonetic-semantic lexical graph using the Unisyn phonetic dictionary, IPA and G2P representations, and WordNet definitions, and retrieves candidate words or senses to constrain LLM reasoning within a structured candidate space. We further introduce WebPun, a new large-scale dataset containing 5,730 annotated heterographic and homographic puns. Experiments on SemEval-2017 and WebPun show that PunGraph consistently improves the performance of small-scale LLMs and achieves competitive results against strong proprietary models. Further analysis shows that retrievalguided phonetic and semantic constraints effectively reduce common reasoning errors in pun interpretation, highlighting the benefits of integrating structured knowledge with LLMs. We release our code and dataset at https: //github.com/ysu132/PunGraph.

## 1 Introduction

Puns are a linguistic phenomenon that exploit lexical polysemy or phonetic similarity to evoke multiple meanings within a single utterance, thereby creating a humorous effect (Partington, 2009; Kao et al., 2016). As illustrated in Figure 1, puns are generally categorized into two main types (Xu et al., 2024): homographic puns and heterographic puns, corresponding to semantic ambiguity and phonetic similarity, respectively. This effect arises from the interaction between phonological resemblance and contextual semantic reasoning, resulting in a humorous interpretation (Attardo, 2018).

![](images/7a51b144b34b6580eb4c215c82ff9ff25f55655fa3aaab100eb9c6522426df0c.jpg)  
Figure 1: The types of puns.

Pun reasoning plays a central role in computational humor understanding and remains an important challenge in natural language processing (Kao et al., 2016). It underlies a broad range of pun-related tasks, including detection (Miller et al., 2017; Zou and Lu, 2019), generation (Yu et al., 2018; Sun et al., 2022; Tian et al., 2022), and interpretation (Prnjak et al., 2023; Zangari et al., 2025). Among these, pun reasoning serves as a critical intermediate step, requiring accurate semantic analysis of the pun word to enable coherent interpretation of the entire sentence. While LLMs (Hurst et al., 2024) have demonstrated strong reasoning capabilities across a wide range of NLP tasks (Liu et al., 2023), they remain limited in handling complex linguistic phenomena such as puns (Xu et al., 2024; Sravanthi et al., 2024; Mi et al., 2025). This limitation is largely attributed to the scarcity of high-quality training data and the difficulty of integrating multimodal cues (e.g., phonetic information) (Su et al., 2025), which are essential for capturing both phonological similarity and implicit semantic shifts.

While state-of-the-art proprietary LLMs exhibit some capability in processing humor, open-source small-scale LLMs face exacerbated challenges when tasked with pun reasoning. Constrained by their limited parameter capacity and the scale of training data, these smaller models struggle to internally map the complex interactions between orthography, phonology, and polysemy without explicit structural guidance. Based on our empirical analysis, the severe degradation of small-scale LLMs in pun comprehension primarily stems from two core limitations: (1) phonological reasoning deficiency, where models struggle to accurately capture the phonetic relationship between a pun word and its latent alternative word; and (2) unconstrained semantic generation, where generated interpretations tend to deviate from the linguistic structure and intended semantic space of the pun.

To address these limitations, we propose Pun-Graph, a retrieval-enhanced knowledge graph framework for pun understanding. PunGraph constructs a structured lexical knowledge graph that explicitly models phonological associations and semantic relationships between words. This provides critical external knowledge to support pun reasoning, thereby addressing the phonological reasoning deficiency (Limitation 1). Unlike conventional endto-end approaches, PunGraph introduces a retrievalguided selection mechanism that retrieves candidate words or sense interpretations from the graph. By presenting these as explicit reasoning options, it guides model reasoning within a constrained candidate space, effectively mitigating the issue of unconstrained semantic generation (Limitation 2). In addition, to address the scarcity of high-quality training data, we introduce WebPun, a new largescale pun dataset collected and annotated from publicly available pun websites, enriching existing benchmarks with more diverse and up-to-date examples. We evaluate PunGraph on both public pun benchmarks and WebPun, and experimental results show that it consistently outperforms strong baselines and achieves competitive performance against large-scale models.

In summary, our contributions are:

• We analyze the limitations of current smallscale models in pun reasoning tasks and explore reasons for these limitations in the context of both heterographic and homographic pun reasoning.

• We propose PunGraph, the first knowledge graph reasoning-enhanced LLM framework for pun understanding tasks, promoting better reasoning accuracy based on the actual meaning of pun words.

• We construct a new pun reasoning dataset to supplement previously existing resources on pun understanding and further promote community interest in pun tasks.

## 2 Problem Analysis

The input to our system consists of a pun sentence $p$ and its corresponding pun word $w _ { p }$ .

$$
p = \{ w _ { 1 } , w _ { 2 } , \ldots , w _ { p } , \ldots , w _ { n } \}\tag{1}
$$

where $w _ { n }$ represents the words of the pun sentence. For heterographic puns, the task objective is to generate an alternative word $w _ { a }$ that shares the same or similar pronunciation as the pun word while conveying a different meaning. In contrast, for homographic puns, the goal is to infer multiple senses of the same word within a given context, which can be represented as $\left[ s _ { 1 } , s _ { 2 } , . . . , s _ { n } \right]$ . For simplicity, we restrict our formulation to the case where the pun involves two senses $[ s _ { 1 } , s _ { 2 } ]$

Existing LLMs, particularly small-scale models, struggle to capture the complex interplay between phonology and multiple semantic senses required for pun comprehension. To quantify this limitation, we systematically evaluate the performance of current LLMs on pun reasoning tasks.

For heterographic puns, we examine the ability of small-scale models, using Qwen-2.5-7B (Qwen et al., 2024) as a representative case study, to directly generate alternative words. As formulated below, performance is evaluated based on the International Phonetic Alphabet (IPA) (International Phonetic Association, 1999) similarity score between the original pun word and the generated alternative word:

$$
S _ { i p a } ( p _ { w } , w _ { a } ) = 1 - \frac { d _ { e d i t } ( \phi ( p _ { w } ) , \phi ( w _ { a } ) ) } { \operatorname* { m a x } ( | \phi ( p _ { w } ) | , | \phi ( w _ { a } ) | ) }\tag{2}
$$

where $\phi ( \cdot )$ denotes the IPA phonetic representation of a word, and $d _ { \mathrm { e d i t } }$ denotes the edit distance between two phonetic sequences. We define the similarity threshold as τ. Furthermore, we define the number of erroneous samples whose similarity scores are below the threshold as $N _ { < } ( \tau )$ , and the number of erroneous samples whose similarity scores are greater than or equal to the threshold as $N _ { \geq } ( \tau )$ , as formulated below.

$$
N _ { < } ( \tau ) = \sum _ { w _ { a } \in W } \mathbb { I } \big ( S _ { i p a } ( p _ { w } , w _ { a } ) < \tau \big )\tag{3}
$$

$$
N _ { \ge } ( \tau ) = \sum _ { w _ { a } \in W } \mathbb { I } \big ( S _ { i p a } ( p _ { w } , w _ { a } ) \ge \tau \big )\tag{4}
$$

![](images/f11fefdcc5696902be9d252e330b7dbcfef348ef49f785cb055ae102d78b67a8.jpg)  
Figure 2: Performance of Qwen2.5-7B on the heterographic pun replacement task under varying IPA similarity thresholds. The x-axis represents the IPA similarity threshold, and the y-axis denotes the number of erroneous samples. The increase in erroneous samples below the threshold suggests that most generation errors stem from insufficient phonetic similarity.

where $\mathbb { I } ( \cdot )$ is the indicator function.

As shown in Figure 2, as the threshold $\tau$ increases, $N _ { < } ( \tau )$ steadily rises, while $N _ { \geq } ( \tau )$ correspondingly decreases. This trend indicates that a substantial proportion of erroneous predictions generated by LLMs have low phonological similarity to the target pun words, suggesting that the predicted alternative words often deviate considerably from the phonological form required by the original pun. We thereby have:

Limitation 1: LLMs lack explicit constraints on phonetic similarity during the reasoning process, causing the generated alternative words to fail to satisfy the fundamental phonetic requirements of heterographic puns.

For homographic puns, we analyze the ability of the open-source model to generate polysemous explanations of synonyms. We define the set of word sense explanations generated by the model as G, and the set of dictionary sense definitions as $D _ { \colon }$ as shown below.

$$
G ( p _ { w } ) = \{ s _ { 1 } , s _ { 2 } \}\tag{5}
$$

$$
D ( p _ { w } ) = \{ d _ { 1 } , d _ { 2 } , \cdot \cdot \cdot , d _ { m } \}\tag{6}
$$

As shown in the formula, we evaluate the model’s generated results by combining definitions from the open-source dictionary, specifically by calculating the cosine similarity score $S _ { s e m } ( \cdot )$ between the generated definition and the dictionary definition.

$$
S _ { s e m } ( s _ { i } , d _ { j } ) = \cos { \left( e ( s _ { i } ) , e ( d _ { j } ) \right) }\tag{7}
$$

![](images/359afd3b08ccd506f45e20225397dc08836192767bb55edf345e5d7cf42d460a.jpg)  
Figure 3: Performance of LLM-generated homographic pun words interpretation under varying semantic similarity thresholds with WordNet. The horizontal axis represents the semantic similarity threshold, and the vertical axis represents the number of samples with similarity greater than the threshold.

$$
M ( s _ { i } ) = \operatorname* { m a x } _ { d _ { j } \in D ( p _ { w } ) } S _ { s e m } ( s _ { i } , d _ { j } )\tag{8}
$$

$$
\hat { y } = \mathbb { I } \left( M ( s _ { 1 } ) \ge \tau _ { s e m } \wedge M ( s _ { 2 } ) \ge \tau _ { s e m } \right)\tag{9}
$$

where $M ( s _ { i } )$ denotes the maximum matching similarity between a model-generated sense interpretation and the corresponding dictionary definitions. The model is considered to have correctly generated the two meanings of the pun only when both $M ( s _ { 1 } )$ and $M ( s _ { 2 } )$ exceed the predefined threshold. In addition, we define the proportion of positive samples $P _ { i n } ( \tau )$ that successfully match the dictionary definitions, as formulated below:

$$
P _ { i n } ( \tau ) = \frac { \sum _ { i \in P } \hat { y } _ { i } } { | P | }\tag{10}
$$

$$
P _ { o u t } ( \tau ) = 1 - P _ { i n } ( \tau )\tag{11}
$$

And the proportions of negative samples $N _ { i n } ( \tau )$

$$
N _ { i n } ( \tau ) = \frac { \sum _ { i \in N } \hat { y } _ { i } } { | N | }\tag{12}
$$

$$
N _ { o u t } ( \tau ) = 1 - N _ { i n } ( \tau )\tag{13}
$$

where $P _ { o u t } ( \tau )$ and $N _ { o u t } ( \tau )$ denote the proportions of positive and negative samples, respectively, whose generated sense explanations fail to match the dictionary definitions.

As shown in Figure 3, within the dictionary definition space, correct predictions consistently achieve higher matching scores than incorrect predictions. This indicates that erroneous predictions are less likely to align with the dictionary-defined senses of the target pun word, suggesting that the generated sense explanations often deviate from the intended semantic space, so we thus have:

Limitation 2: LLMs are prone to uncontrolled generation during reasoning, resulting in generated sense explanations that fail to establish meaningful semantic associations with the original pun words.

In summary, the underperformance of smallscale LLMs in pun comprehension primarily stems from phonological reasoning deficiency and unconstrained semantic generation, which are two core limitations that we address in PunGraph.

## 3 PunGraph

This section introduces PunGraph, a framework specifically designed for pun reasoning that addresses the two core limitations identified above: insufficient phonological reasoning and unconstrained semantic generation. PunGraph constructs a lexical-level knowledge graph from a pronunciation dictionary to model phonological relationships between pun words, and further integrates dictionary-based sense definitions to retrieve candidate interpretations, providing LLMs with a constrained reasoning space, as illustrated in Figure 4. Through this retrieval-enhanced reasoning process, PunGraph alleviates the weaknesses of small-scale LLMs in phonological modeling and semantic control, improving both the accuracy and controllability of pun reasoning.

## 3.1 Pun Graph Construction

To construct the knowledge graph, Inspired by (Manurung et al., 2008), we generate word triples based on phonetic similarity relations derived from the Unisyn phonetic dictionary (Fitt and Isard, 1999). We further incorporate IPA-based similarity modeling and grapheme-to-phoneme (G2P) (Bisani and Ney, 2008) augmentation to enhance phonetic associations between words, building upon the original phonetic representations provided by the dictionary. Specifically, we compute the cosine similarity between the IPA representations of word pairs and establish connections between pairs whose similarity scores exceed a threshold, thereby constructing triples in the standard form of source, relation, target. Furthermore, we integrate the large-scale English lexical dictionary WordNet (Miller, 1995) to supplement semantic information for words that can be matched within the graph. The formulation of the knowledge graph is defined as follows:

$$
\begin{array} { c } { { \mathcal { G } = \{ ( w _ { i } , r _ { \mathrm { p h o n } } , w _ { j } , S ( w _ { i } ) , S ( w _ { j } ) ) \mid } } \\ { { e \big ( \phi ( w _ { i } ) , \phi ( w _ { j } ) \big ) > 0 . 7 5 \} } } \end{array}\tag{14}
$$

where $w _ { i }$ and $w _ { j }$ denote word entities, $r _ { \mathrm { p h o n } }$ represents the phonetic similarity relation, and $S ( w _ { i } )$ and $s ( w _ { j } )$ denote the corresponding semantic interpretations derived from the dictionary. The function $e ( \cdot )$ denotes the phonetic similarity computation, with implementation details provided in $\mathsf { A p - }$ pendix D. In addition, we import the constructed graph into the Neo4j graph database <sup>1</sup> for storage and visualization, thereby facilitating subsequent structural analysis and retrieval operations.

## 3.2 Pun reasoning

After constructing the knowledge graph, we design the corresponding reasoning tasks for heterographic and homographic puns, respectively. For each sample, given a pun $p$ and its pun words $p _ { w }$ we first use prompts to guide the LLMs for preliminary reasoning to get alternative words $a _ { w }$ and interpretation $[ s _ { 1 } , s _ { 2 } ]$ , and then classify the results into two categories: correct predictions and incorrect predictions. For incorrect cases, we further introduce a graph retrieval enhancement mechanism to utilize external structured knowledge to assist the model in completing subsequent reasoning.

Heterographic Puns In this stage, we introduce a reasoning enhancement mechanism for heterographic puns. Specifically, as shown in Figure 2, based on the performance trend of IPA similarity, we set 0.6 as the key threshold. When the similarity is below this threshold, the proportion of incorrect predictions increases significantly, indicating that the model has difficulty generating reasonable replacement words that are phonetically consistent with the pun within this range. Therefore, we focus on screening and processing samples with IPA similarity below 0.6.

We retrieve one-hop adjacent near-homophones and semantically related words from the phonetic-semantic knowledge graph given a pun word and construct a constrained candidate set $[ a _ { 1 } , a _ { 2 } , a _ { 3 } . . . , a _ { n } ]$ . Specifically, the retrieval process traverses pronunciation similarity links and semantic association edges in the graph, where the average number of retrieved candidates is reported using the Candidate Retrieval Number (CRN) described in Appendix B.1. The retrieved candidates are subsequently formatted as explicit answer options and incorporated into the prompt, enabling the LLM to perform reasoning within a constrained candidate space and select the candidate that best fits the contextual semantics. Detailed prompt templates and candidate formatting strategies are provided in Appendix C.

![](images/45cb871bc76752c29b8b701f14dde1f6e42c793ecf4ecc2899fee27591a5ff4e.jpg)  
Figure 4: The Overview of PunGraph Framework

Homographic Puns We also consider the reasoning task for homographic puns. In the initial classification stage, we set the similarity threshold to 0.5, as Figure 3 shows that this value yields the largest gap between positive and negative cases in terms of dictionary coverage. Specifically, when the similarity exceeds 0.5, a larger proportion of positive instances fall within the dictionary-defined sense space, whereas when the similarity is below 0.5, negative instances are more likely to fall outside this space. Based on this observation, we select error cases with similarity below 0.5 for subsequent knowledge graph enhancement.

Given a pun word $p _ { w }$ we retrieve its associated entity attributes from the knowledge graph to obtain a set of candidate senses [s<sub>1</sub>, s<sub>2</sub>], which are then formulated as multiple-choice options. Similar to the heterographic puns, we guide LLMs to select the most appropriate option within this constrained candidate space, thereby producing the final sense prediction for the pun word.

## 4 WebPun Dataset

We proposed a new benchmark to address the scarcity of high-quality data in pun reasoning.

## 4.1 Data Preparation

Existing work on pun understanding mainly relies on the SemEval-2017 benchmark, which is limited in scale and diversity, making it insufficient for evaluating modern LLMs on pun reasoning. To address this gap, we construct WebPun which is a new pun dataset collected from public pun websites, including Pun.me<sup>2</sup> and Punpedia<sup>3</sup>, and collect a total of 24,880 samples. We filter the dataset to retain pun sentences containing more than five words, while removing non-English samples and entries that do not form complete sentences. In addition, we remove rare pun words, as well as phrasal puns. Ultimately, a total of 5,730 samples are retained in the final dataset.

Since some pun sentences do not provide annotations for heterographic and homographic categories, we design a novel automated classification method. Specifically, we first employ the latest closed-source large model, GPT-5.5 (OpenAI, 2026), to perform initial classification. To further improve annotation accuracy, we additionally introduce Gemini-3.5-flash (Google DeepMind, 2025b) and Claude-Opus-4-1 (Anthropic, 2025) as auxiliary classification models and conduct crosscomparisons among the outputs of the three LLMs. When inconsistencies arise across model predictions, the corresponding samples are further submitted for manual review to determine the final classification labels. Further data annotation and classification prompt details can be shown at Appendix A.1.

<sup>2</sup>https://pun.me/

<sup>3</sup>https://punpedia.org/

![](images/3e58e2a68c3c17f9ad6bc22921314ec8118606f40a9a75ee0ed7aaffa8fa6205.jpg)  
Figure 5: Part-of-speech distribution of pun words in the WebPun and SemEval datasets.

## 4.2 Data Analysis

After completing the annotation process, we further analyze the statistical characteristics of the dataset, as presented in Appendix B.1. WebPun contains 5,730 annotated pun instances, including 5,061 heterographic puns and 679 homographic puns. In addition, we conduct a part-of-speech analysis of the pun words in WebPun and compared the results with those of the SemEval dataset, as illustrated in Figure 5.

The results show that nouns account for more than half of the puns in the WebPun dataset, followed by verbs and adjectives. Together, these three categories constitute 95.9% of all puns in WebPun. This distribution is consistent with that of the SemEval-2017 dataset, where the proportion of nouns is 92.1%, suggesting that puns rely primarily on nouns, verbs, and adjectives to convey semantic ambiguity and humorous effects. Meanwhile, compared with existing pun datasets, WebPun further strengthens the coverage of the most common noun-based pun category, resulting in richer lexical diversity and semantic ambiguity. This characteristic makes WebPun a more challenging and effective benchmark for evaluating the pun understanding and reasoning capabilities of LLMs.

## 5 Experiments

## 5.1 Dataset

To verify the effectiveness of our method, we conducted experiments on SemEval 2017 Dataset (Augenstein et al., 2017) and WebPun, respectively. Table 2 shows the data statistics of the two datasets. For the phonetic-semantic knowledge graph, we construct a total of 46,208 word entities and 760,312 phonetic similarity relations. We empirically set the threshold for establishing phonetic similarity links to 0.75.

## 5.2 Baselines

We select a diverse set of baseline models for comparison with our method, which are categorized into three groups: large-scale LLMs, small-scale LLMs, and specialized reasoning methods.

The large-scale LLMs include GPT-4o (Hurst et al., 2024), Gemini 2.0 Flash (Google DeepMind, 2025a), and DeepSeek-V3.2 (DeepSeek-AI, 2025). The small-scale LLMs consist of MiniCPM-8.7B (Hu et al., 2024), Qwen-2.5-7B, Qwen-2.5-70B (Qwen et al., 2024), Qwen-3.5-27B (Qwen Team, 2026) and Llama4-maverick (Meta, 2025). For specialized reasoning methods, we reproduce the PunIntended framework (Zeng et al., 2024), while also considering the graph-based reasoning methods, GCR (Luo et al., 2024) and ReKG-MCTS (Song et al., 2025), for further comparison.

## 5.3 Evaluation Metrics

We conduct separate evaluations for heterographic and homographic puns. For heterographic puns, we first normalize word forms to reduce the impact of surface-level variations such as capitalization, singular/plural inflections, and tense changes. Inspired by (Su et al., 2026), we perform exact matching on the alternative words predicted by the model and adopt precision, recall, and F1 score as evaluation metrics. For homographic puns, we compute pairwise semantic similarity between the gold-standard sense explanations and the sense explanations generated by the model on both the SemEval 2017 Dataset (Augenstein et al., 2017; Su et al., 2026) and WebPun, and use accuracy and F1 score to evaluate the overall performance. A prediction is regarded as correct only when the similarity scores of the corresponding sense explanations both exceed a predefined threshold. In addition, we further define a Partial Matching Accuracy (PMA) metric, which is counted as correct under PMA when at least one of the two generated sense explanations achieves a similarity score higher than the predefined threshold with the corresponding goldstandard explanation.

## 6 Results and Discussion

## 6.1 Main Results

Table 1 presents the comparative results of Pun-Graph against different categories of baseline methods and table 3 reports the relative F1-score improvement of backbone small-scale LLMs. First, compared with direct reasoning using the same backbone models, PunGraph consistently improves performance across both datasets and both pun reasoning tasks, demonstrating the effectiveness of retrieval-enhanced structured knowledge augmentation. For example, with Qwen-2.5-7B, PunGraph achieves relative F1 improvements of 57.95% on SemEval and 30.45% on WebPun for heterographic puns. For homographic puns, it further yields gains of 4.31% and 4.70%, respectively. Similar improvements are observed on Qwen-3.5-27B and Llama4-Maverick, indicating that PunGraph is not tied to a specific backbone model, but serves as a general and effective external knowledge augmentation framework for pun reasoning. In addition, PunGraph remains highly competitive when compared with large-scale proprietary LLMs. Notably, on the SemEval heterographic pun task, PunGraph-Llama4-Maverick achieves an F1 score of 83.41, outperforming GPT-4o, Gemini-2.0 Flash, and DeepSeek-V3.2. These results suggest explicitly modeling phonological similarity and constraining reasoning within a structured semantic candidate space can effectively compensate for the limitations of smaller models in data coverage, implicit phonological reasoning, and controllable generation.

<table><tr><td rowspan="3">Model</td><td colspan="6">Semeval 2017 Dataset</td><td colspan="6">WebPun Dataset</td></tr><tr><td colspan="3">Heterographic Puns</td><td colspan="3">Homographic Puns</td><td colspan="3">Heterographic Puns</td><td colspan="3">Homographic Puns</td></tr><tr><td>Pre (↑)</td><td>Rec (↑)</td><td>F1 (↑)</td><td>Acc (↑)</td><td>PMA (↑)</td><td>F1 (↑)</td><td>Pre (↑)</td><td>Rec (↑)</td><td>F1 (↑)</td><td>Acc (↑)</td><td>PMA (↑)</td><td>F1 (↑)</td></tr><tr><td>Large-scale LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4o (Hurst et al., 2024)</td><td>76.95</td><td>82.11</td><td>79.45</td><td>76.27</td><td>98.54</td><td>87.35</td><td>86.86</td><td>88.39</td><td>87.62</td><td>70.85</td><td>98.65</td><td>84.75</td></tr><tr><td>Gemini-2.0 Flash (Google DeepMind, 2025a)</td><td>72.67</td><td>82.69</td><td>77.36</td><td>71.08</td><td>98.69</td><td>84.56</td><td>61.26</td><td>69.78</td><td>65.24</td><td>68.74</td><td>98.51</td><td>83.06</td></tr><tr><td>DeepSeek-V3.2 (DeepSeek-AI, 2025)</td><td>76.68</td><td>84.28</td><td>80.31</td><td>66.26</td><td>98.15</td><td>82.12</td><td>67.86</td><td>75.01</td><td>71.26</td><td>66.77</td><td>97.31</td><td>82.02</td></tr><tr><td>Small-scale LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MiniCPM-8.7B (Hu et al., 2024)</td><td>31.32</td><td>47.84</td><td>37.86</td><td>40.71</td><td>93.07</td><td>66.64</td><td>22.20</td><td>30.63</td><td>25.74</td><td>40.51</td><td>94.92</td><td>67.61</td></tr><tr><td>Qwen-2.5-7B (Qwen et al., 2024)</td><td>35.06</td><td>40.22</td><td>37.47</td><td>34.65</td><td>90.80</td><td>62.42</td><td>30.33</td><td>35.27</td><td>32.61</td><td>36.83</td><td>92.07</td><td>64.42</td></tr><tr><td>Qwen-3.5-27B (Qwen Team, 2026)</td><td>69.31</td><td>79.51</td><td>74.06</td><td>68.95</td><td>97.61</td><td>83.20</td><td>63.86</td><td>71.97</td><td>67.68</td><td>69.66</td><td>96.71</td><td>83.18</td></tr><tr><td>Llama4-Maverick (Meta, 2025)</td><td>69.94</td><td>78.04</td><td>73.77</td><td>66.26</td><td>97.84</td><td>82.00</td><td>72.06</td><td>76.88</td><td>74.39</td><td>67.56</td><td>97.61</td><td>82.59</td></tr><tr><td>Specialized Reasoning Methods</td><td></td><td></td><td></td><td></td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PunIntended (Zeng et al., 2024)</td><td>15.84</td><td>17.54</td><td>16.65</td><td>26.35</td><td>85.25</td><td>50.98</td><td>13.73</td><td>14.87</td><td>14.27</td><td>29.66</td><td>83.63</td><td>51.60</td></tr><tr><td>GCR (Luo et al., 2024)</td><td>40.89</td><td>68.44</td><td>51.19</td><td>43.04</td><td>93.12</td><td>37.19</td><td>36.86</td><td>42.65</td><td>39.54</td><td>45.32</td><td>94.26</td><td>39.74</td></tr><tr><td>ReKG-MCTS (Song et al., 2025)</td><td>61.74</td><td>78.11</td><td>68.97</td><td>22.11</td><td>78.04</td><td>48.41</td><td>57.46</td><td>72.61</td><td>64.15</td><td>21.46</td><td>82.93</td><td>52.17</td></tr><tr><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PunGraph-Qwen-2.5-7B</td><td>53.68</td><td>65.94</td><td>59.18</td><td>45.71</td><td>93.38</td><td>65.11</td><td>38.53</td><td>47.49</td><td>42.54</td><td>45.44</td><td>95.81</td><td>67.45</td></tr><tr><td>PunGraph-Qwen-3.5-27B</td><td>74.40</td><td>86.18 88.99</td><td>79.86</td><td>76.18</td><td>98.84</td><td>85.71</td><td>65.66</td><td>76.06</td><td>70.48</td><td>78.33</td><td>98.06</td><td>86.07</td></tr><tr><td>PunGraph-Llama4-Maverick</td><td>78.48</td><td></td><td>83.41</td><td>71.80</td><td>97.46</td><td>83.43</td><td>75.86</td><td>83.43</td><td>79.47</td><td>69.51</td><td>98.36</td><td>82.62</td></tr></table>

Table 1: Results of pun reasoning on the SemEval and WebPun datasets. Boldface indicates the best performance, while underlined values indicate the second-best performance among different methods. The models are categorized into Large-scale LLMs, Small-scale LLMs, Specialized Reasoning Methods, and Ours. Pre., Rec., F1, Acc., and PMA denote the evaluation metrics of precision, recall, F1-score, accuracy and partial matching accuracy, respectively.

<table><tr><td>Model</td><td>Semeval</td><td>WebPun</td></tr><tr><td>Homographic</td><td>1,298</td><td>668</td></tr><tr><td>Heterographic</td><td>1,098</td><td>5,061</td></tr></table>

Table 2: The statistics of Semeval-2017 dataset and WebPun dataset.

<table><tr><td>Model</td><td>SemEval Het.</td><td>SemEval Hom.</td><td>WebPun Het.</td><td>WebPun Hom.</td></tr><tr><td>Qwen-2.5-7B</td><td>+57.95%</td><td>+4.31%</td><td>+30.45%</td><td>+4.70%</td></tr><tr><td>Qwen-3.5-27B</td><td>+7.83%</td><td>+3.02%</td><td>+4.14%</td><td>+3.47%</td></tr><tr><td>Llama4-Maverick</td><td>+13.06%</td><td>+1.74%</td><td>+6.83%</td><td>+0.04%</td></tr></table>

Table 3: Relative F1-score improvement of PunGraph over the corresponding small-scale LLMs across different datasets and pun types. Het. and Hom. denote heterographic and homographic puns, respectively.

Finally, PunGraph consistently outperforms specialized reasoning baselines across both datasets. On SemEval, PunGraph-Llama4-Maverick surpasses ReKG-MCTS by 14.44 F1 on heterographic puns, while on WebPun it further achieves gains of +15.32 and +33.90 F1 on heterographic and homographic puns, respectively. These results highlight the advantage of PunGraph in modeling phonological-semantic interactions through taskspecific retrieval-enhanced reasoning.

## 6.2 Ablation Study

To evaluate the effectiveness of different pronunciation link construction methods in the knowledge graph, we conduct an ablation study to compare the impact of different phonetic connection strategies on model performance. Specifically, (1) w/o IPA removes only the IPA-based similarity modeling relations; (2) w/o G2P removes only the pronunciation association relations introduced through G2P augmentation; and (3) w/o Unisyn removes the similarity links constructed from the Unisyn pronunciation dictionary. Through these experiments, we aim to investigate the contribution of different phonetic modeling methods to the construction of pronunciation-aware graph connections and their influence on pun reasoning performance.

<table><tr><td>Model</td><td colspan="3">SemEval 2017 Dataset</td><td colspan="3">WebPun Dataset</td></tr><tr><td></td><td>Pre (↑)</td><td>Rec (↑)</td><td>F1 (↑)</td><td>Pre (↑)</td><td>Rec (↑)</td><td>F1 (↑)</td></tr><tr><td>Qwen2.5-7B</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PunGraph</td><td>53.68</td><td>65.94</td><td>59.18</td><td>38.53</td><td>47.49</td><td>42.54</td></tr><tr><td>w/o IPA</td><td>46.28</td><td>55.73</td><td>50.57</td><td>29.60</td><td>37.69</td><td>33.16</td></tr><tr><td>w/o G2P</td><td>48.78</td><td>60.46</td><td>53.99</td><td>29.73</td><td>38.82</td><td>33.67</td></tr><tr><td>w/o Unisyn</td><td>42.58</td><td>51.85</td><td>46.76</td><td>26.20</td><td>32.67</td><td>29.07</td></tr><tr><td colspan="7">Llama4-Maverick</td></tr><tr><td>PunGraph</td><td>78.48</td><td>88.99</td><td>83.41</td><td>75.86</td><td>83.43</td><td>79.47</td></tr><tr><td>w/o IPA</td><td>71.49</td><td>82.98</td><td>76.81</td><td>73.26</td><td>81.34</td><td>77.09</td></tr><tr><td>w/o G2P</td><td>72.67</td><td>83.82</td><td>77.85</td><td>73.40</td><td>81.25</td><td>77.12</td></tr><tr><td>w/o Unisyn</td><td>70.67</td><td>80.58</td><td>75.30</td><td>70.80</td><td>77.80</td><td>74.13</td></tr></table>

Table 4: Ablation results of different phonetic knowledge components in PunGraph using Qwen2.5-7B and Llama4-Maverick, including IPA-based similarity modeling, G2P augmentation, and the Unisyn pronunciation dictionary.

Table 4 reports the ablation results for different pronunciation knowledge components in Pun-Graph. The results demonstrate that integrating multiple pronunciation modeling strategies consistently improves performance across datasets and backbone models. For instance, on the SemEval-2017 dataset with Qwen2.5-7B as the backbone, removing IPA-based similarity modeling, G2P augmentation, and the Unisyn pronunciation dictionary leads to F1-score drops of 8.61%, 5.19%, and 12.42%, respectively. Notably, across both Qwen2.5-7B and Llama4-Maverick backbones and on both the SemEval-2017 and WebPun datasets, removing G2P consistently results in the smallest performance degradation, whereas removing the Unisyn pronunciation dictionary causes the most substantial decline. These findings suggest that pronunciation similarity relations derived from Unisyn play a central role in PunGraph’s pronunciation association modeling, while G2P augmentation primarily serves as a complementary mechanism that enhances pronunciation coverage and improves the robustness of the constructed graph.

## 6.3 Error Analysis

We conduct an error analysis to investigate the performance of the proposed method on the pun reasoning task and categorize the error types for both homographic and heterographic puns. Specifically, the errors can be grouped into three main categories: (1) missing phonetic similarity links in the graph or the failure of the dictionary to provide the corresponding definitions, which prevents the model from retrieving valid word or definition candidates; (2) incorrect selection among the retrieved candidate words or definition options; and (3) generation errors produced by the model itself, such as failing to follow the prompt instructions. We present the corresponding error statistics for Qwen-2.5-7B as a representative example according to homographic and heterographic puns.

![](images/9f3345646aa0291e7b63f92547929c5a79ae2874ce0493306195d21161af8bc7.jpg)  
Figure 6: Error cases of heterographic and homographic puns in the SemEval-2017 dataset. Selection Error, Missing KG, and Invalid Output denote cases where LLMs select incorrect candidates, the knowledge graph retrieval misses the ground-truth answer, and the model fails to follow the required output format, respectively.

As shown in the Figure 6, the primary source of errors in our method is the model’s tendency to confuse the correct answer with other candidate options, thereby leading to reasoning failures. This issue is particularly pronounced for homographic puns, where such errors account for 91.5% of all failures. In contrast, errors caused by knowledge graph retrieval account for only 19% and 5.9% in heterographic and homographic puns, respectively. These results indicate that the constructed phonetic similarity links and dictionary definitions provide effective coverage of the pun datasets.

## 7 Related Works

## 7.1 Pun Interpretation

Existing work on pun interpretation mainly follows two directions: semantics-based methods and pronunciation-aware methods. SemEval-2017 Task 7 formalized English pun processing into detection, location, and interpretation subtasks, providing a standard benchmark for later studies (Miller et al., 2017). Subsequent work extended this setting to joint detection and location, multilingual pun interpretation, and task-specific pun modeling (Zou and Lu, 2019; Prnjak et al., 2023; Chen et al., 2024). Semantics-based methods use lexical resources such as WordNet or distributed representations to model word senses and semantic relatedness (Miller, 1995; Miller et al., 2017; Zhou et al., 2020). They are suitable for homographic puns, but are less reliable for heterographic puns that depend on latent pronunciation-based alternatives. Pronunciation-aware methods address this issue by using pronunciation dictionaries, phonological resources, phoneme-level representations, or pronunciation-aware attention mechanisms (Manurung et al., 2008; Zhou et al., 2020). Pun generation studies further highlight the interaction among the pun word, the alternative word, and context (Yu et al., 2018; He et al., 2019; Sun et al., 2022; Mittal et al., 2022; Tian et al., 2022). However, semantic and phonological methods operate in different spaces, and LLMs may still miss alternative words or phonetic links (Xu et al., 2024; Zangari et al., 2025).

## 7.2 Retrieval-Augmented Reasoning

Retrieval-augmented generation (RAG) enhances LLMs by combining parametric model memory with non-parametric knowledge retrieved from external corpora, improving knowledge access and generation quality in knowledge-intensive tasks (Lewis et al., 2020). Conventional RAG mainly retrieves unstructured text passages, while GraphRAG introduces structured graph information, such as nodes, triples, paths, or subgraphs, into retrieval and generation (Peng et al., 2024). In knowledge graph reasoning, recent work retrieves graph paths or subgraphs as evidence for LLM reasoning (Luo et al., 2024; Song et al., 2025; Liu et al., 2025); GNN-RAG further uses graph neural networks to score candidate answers and retrieve connecting paths (Mavromatis and Karypis, 2025). These methods mainly focus on factual or entity-relation reasoning, rather than the joint phonological-semantic structure required for pun interpretation. PunGraph adapts retrievalaugmented reasoning to this setting by retrieving candidate words or senses from a task-specific lexical graph.

## 8 Conclusion

In this paper, we proposed PunGraph, a retrievalenhanced knowledge graph framework for pun understanding. By integrating phonetic similarity and semantic knowledge into a structured lexical graph, PunGraph enables LLMs to perform more controllable and accurate reasoning for both heterographic and homographic puns. We further introduced WebPun, a new large-scale pun reasoning dataset containing 5,730 annotated samples collected from online pun resources. Experimental results on SemEval-2017 and WebPun demonstrate that PunGraph consistently improves the performance of small-scale LLMs and achieves competitive results compared with strong proprietary models. Further analysis shows that retrieval-guided phonetic and semantic constraints effectively reduce common errors in pun reasoning.

## Limitations

Despite the promising performance of PunGraph, our work has several limitations. First, our framework is designed primarily for English and relies on English-specific lexical resources such as Unisyn and WordNet, which may limit its direct applicability to other languages. Second, the effectiveness of PunGraph depends on the coverage of the constructed phonetic-semantic knowledge graph; rare words, creative expressions, or unseen pun patterns may not be well represented in the graph, leading to retrieval failures. In addition, although retrieval constrains the reasoning space of LLMs, the final prediction still depends on the model’s ability to select the correct candidate, and incorrect candidate selection remains a major source of errors. Finally, WebPun is collected from online written pun resources and may not fully capture broader forms of humor such as spoken, multimodal, or culturally specific puns. We leave multilingual and multimodal extensions of PunGraph for future work.

## Ethical Considerations

The human-participant study presented in this paper was approved by the appropriate ethics committee of Shanxi University. All procedures involving human participants were conducted in accordance with the relevant ethical guidelines and regulations.

## Acknowledgments

This research is supported by the Strong AI Lab and the Natural, Artificial, and Organisation Intelligence Institute at the University of Auckland. The first author of this research is funded by the China Scholarship Council (CSC).

## References

Anthropic. 2025. System card: Claude opus 4 & claude sonnet 4. https://www-cdn.anthropic.com/ 4263b940cabb546aa0e3283f35b686f4f3b2ff47. pdf.

International Phonetic Association. 1999. Handbook of the International Phonetic Association: A guide to the use of the International Phonetic Alphabet. Cambridge University Press.

Salvatore Attardo. 2018. Universals in puns and humorous wordplay. Cultures and traditions ofwordplay and wordplay research, pages 89–110.

Isabelle Augenstein, Mrinal Das, Sebastian Riedel, Lakshmi Vikraman, and Andrew McCallum. 2017. Semeval 2017 task 10: Scienceie - extracting keyphrases and relations from scientific publications. CoRR, abs/1704.02853.

Maximilian Bisani and Hermann Ney. 2008. Jointsequence models for grapheme-to-phoneme conversion. Speech Communication, 50(5):434–451.

Yang Chen, Chong Yang, Tu Hu, Xinhao Chen, Man Lan, Li Cai, Xinlin Zhuang, Xuan Lin, Xin Lu, and Aimin Zhou. 2024. Are u a joke master? pun generation via multi-stage curriculum learning towards a humor llm. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 878–890.

DeepSeek-AI. 2025. Deepseek-v3.2: Pushing the frontier of open large language models.

Sue Fitt and Steve Isard. 1999. Synthesis of regional english using a keyword lexicon. In Proceedings of Eurospeech 1999.

Joseph L Fleiss and Jacob Cohen. 1973. The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability. Educational and psychological measurement, 33(3):613–619.

Google DeepMind. 2025a. Gemini 2.0 flash model card. Technical report, Google DeepMind.

Google DeepMind. 2025b. Gemini 3 flash model card. Technical report, Google DeepMind.

He He, Nanyun Peng, and Percy Liang. 2019. Pun generation with surprise. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1734–1744.

Shengding Hu, Yuge Tu, Xu Han, Chaoqun He, Ganqu Cui, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Weilin Zhao, et al. 2024. Minicpm: Unveiling the potential of small language models with scalable training strategies. arXiv preprint arXiv:2404.06395.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

International Phonetic Association. 1999. Handbook ofthe International Phonetic Association: A Guide to the Use of the International Phonetic Alphabet. Cambridge University Press.

Justine T Kao, Roger Levy, and Noah D Goodman. 2016. A computational model of linguistic humor in puns. Cognitive science, 40(5):1270–1285.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474.

Chenghao Liu, Qian Liu, Ziqin Zhu, Hao Fei, and Aniket Mahanti. 2025. David vs. goliath: Costefficient financial QA via cascaded multi-agent reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2025, Suzhou, China. Association for Computational Linguistics.

Yixin Liu, Alexander Richard Fabbri, Yilun Zhao, Pengfei Liu, Shafiq Joty, Chien-Sheng Wu, Caiming Xiong, and Dragomir Radev. 2023. Towards interpretable and efficient automatic reference-based summarization evaluation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 16360–16368.

Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and Shirui Pan. 2024. Graphconstrained reasoning: Faithful reasoning on knowledge graphs with large language models. arXiv preprint arXiv:2410.13080.

Ruli Manurung, Graeme Ritchie, Helen Pain, Annalu Waller, Rolf Black, and Dave O’Mara. 2008. Adding phonetic similarity data to a lexical database. Language Resources and Evaluation, 42(3):319–324.

Costas Mavromatis and George Karypis. 2025. GNN-RAG: Graph neural retrieval for efficient large language model reasoning on knowledge graphs. In Findings of the Association for Computational Linguistics: ACL 2025, pages 16682–16699.

Meta. 2025. Llama 4 model card. https: //github.com/meta-llama/llama-models/ blob/main/models/llama4/MODEL\_CARD.md. Official model card for the Llama 4 model family, including Llama 4 Maverick.

Maggie Mi, Aline Villavicencio, and Nafise Sadat Moosavi. 2025. Rolling the dice on idiomaticity: How llms fail to grasp context. In Proceedings of the

63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7314–7332.

George A Miller. 1995. Wordnet: a lexical database for english. Communications ofthe ACM, 38(11):39–41.

Tristan Miller, Christian F Hempelmann, and Iryna Gurevych. 2017. Semeval-2017 task 7: Detection and interpretation of english puns. In Proceedings of the 11th international workshop on semantic evaluation (SemEval-2017), pages 58–68.

Anirudh Mittal, Yufei Tian, and Nanyun Peng. 2022. AMBIPUN: Generating puns with ambiguous context. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1053–1062.

OpenAI. 2026. Gpt-5.5 system card. Technical report, OpenAI.

Alan Scott Partington. 2009. A linguistic account of wordplay: The lexical grammar of punning. Journal ofPragmatics, 41(9):1794–1809.

Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, and Siliang Tang. 2024. Graph retrieval-augmented generation: A survey. arXiv preprint arXiv:2408.08921.

Antonela Prnjak, Dennis R Davari, and Kristina Schmitt. 2023. Clef 2023 joker task 1, 2, 3: Pun detection, pun interpretation, and pun translation. In CLEF (Working Notes), pages 1909–1917.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Qwen Team. 2026. Qwen3.5: Towards native multimodal agents.

Xiaozhuang Song, Shufei Zhang, and Tianshu Yu. 2025. Rekg-mcts: Reinforcing llm reasoning on knowledge graphs via training-free monte carlo tree search. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 9288–9306.

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. Pub: A pragmatics understanding benchmark for assessing llms’ pragmatics capabilities. In Findings of the Association for Computational Linguistics: ACL 2024, pages 12075–12097.

Yuchen Su, Shaoxin Zhong, Yonghua Zhu, Ruofan Wang, Zijian Huang, Qiqi Wang, Na Zhao, Diana Benavides-Prado, and Michael Witbrock. 2026. Words at play: Benchmarking audio pun understanding in large audio-language models. arXiv preprint arXiv:2603.18678.

Yuchen Su, Yonghua Zhu, Ruofan Wang, Zijian Huang, Diana Benavides-Prado, and Michael Witbrock. 2025. A survey of pun generation: Datasets, evaluations and methodologies. Findings of the Association for Computational Linguistics: EMNLP 2025, pages 7375– 7395.

Jiao Sun, Anjali Narayan-Chen, Shereen Oraby, Shuyang Gao, Tagyoung Chung, Jing Huang, Yang Liu, and Nanyun Peng. 2022. Context-situated pun generation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 4635–4648.

Yufei Tian, Divyanshu Sheth, and Nanyun Peng. 2022. A unified framework for pun generation with humor principles. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 3253– 3261.

Zhijun Xu, Siyu Yuan, Lingjie Chen, and Deqing Yang. 2024. “a good pun is its own reword”: Can large language models understand puns? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 11766–11782.

Zhiwei Yu, Jiwei Tan, and Xiaojun Wan. 2018. A neural approach to pun generation. In Proceedings of the 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1650–1660.

Alessandro Zangari, Matteo Marcuzzo, Andrea Albarelli, Mohammad Taher Pilehvar, and Jose Camacho-Collados. 2025. Pun unintended: Llms and the illusion of humor understanding. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 27924–27959.

JingJie Zeng, Liang Yang, Jiahao Kang, Yufeng Diao, Zhihao Yang, and Hongfei Lin. 2024. “barking up the right tree”, a gan-based pun generation model through semantic pruning. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2119–2131.

Yichao Zhou, Jyun-yu Jiang, Jieyu Zhao, Kai-Wei Chang, and Wei Wang. 2020. “The Boating Store Had Its Best Sail Ever”: Pronunciation-attentive Contextualized Pun Recognition. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 813–822.

Yanyan Zou and Wei Lu. 2019. Joint detection and location of english puns. In Proceedings ofthe 2019 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2117–2123.

## A Annotation Details

## A.1 Classification Prompts

We report the prompt of LLMs for classifying pun types, as follows:

## Prompt for Pun Type Classification

System Prompt:   
You are an expert in English pun analysis.   
Your task is to classify the following sentence as   
either a homographic pun or a heterographic pun.   
Definitions:   
1. Homographic pun: A pun in which the same   
written word carries two or more different meanings   
within the sentence.   
Example: “I used to be a banker, but I lost interest.”   
→ “interest” has multiple meanings.   
2. Heterographic pun: A pun in which two different   
words or phrases have different spellings but similar   
pronunciation, creating humorous ambiguity.   
Example: “Dentists don’t like a hard day at the ori  
fice.” → “orifice” sounds similar to “office”.   
Instructions:   
Read the sentence carefully. Identify the main pun   
word. Determine whether the humor is created pri  
marily through semantic ambiguity of the same word,   
corresponding to a homographic pun, or phonetic   
similarity between different words, corresponding to   
a heterographic pun.   
Return valid JSON only in the following format:   
{   
p u n \_ t y p e " : " homographic pun " o r   
" h e t e r o g r a p h i c p u n "   
}   
Rules:   
Return only one label. The pun word must be the   
exact word appearing in the sentence. Do not include   
explanations. Output JSON only.   
User Input:   
sentence: {sentence}

## A.2 Data Annotation

Inspired by (Chen et al., 2024), we adopt a few-shot prompting strategy to perform preliminary annotation with large language models. Specifically, we first select three homographic and three heterographic pun examples from the SemEval dataset, and combine these examples together with their corresponding annotations and prompt instructions to guide the models in subsequent annotation tasks. Notably, for heterographic puns, the annotation target is the corresponding ground-truth replacement word, while for homographic puns, the annotation consists of the two sense interpretations of the pun word, uniformly represented in the format of [s<sub>1</sub>, s<sub>2</sub>].

Similar to the pun type classification process, we also employ three closed-source large language models for collaborative annotation. Samples with consistent annotations across all three models are further reviewed by an expert annotator for quality assurance. In addition, we place particular emphasis on cases where the three models produce inconsistent annotations. For such samples, three expert annotators <sup>4</sup> jointly conduct manual corrections, and the final annotation is determined through majority voting. Furthermore, we evaluate inter-annotator agreement by randomly sampling 150 annotated instances and measuring consistency using Fleiss’ Kappa (Fleiss and Cohen, 1973). The final agreement score is 0.54, indicating a moderate level of agreement among annotators and providing reasonable support for the consistency of the annotations.

## B Dataset details

## B.1 Data Statistics

We further conduct a statistical analysis of the SemEval and WebPun datasets, as shown in Table 6. Specifically, we report the average sentence length, total number of words, and number of unique pun words for each dataset. The results show that although WebPun contains substantially more heterographic pun samples than SemEval, the increase in the number of unique pun words is relatively limited. This suggests a noticeable reuse of high-frequency pun words in real-world online puns, where multiple pun expressions are often constructed around the same or semantically related core pun words.

In addition, we analyze the number of candidate items retrieved for each pun word in PunGraph, denoted as the Candidate Retrieve Number (CRN). For heterographic puns, CRN refers to the number of candidate words retrieved through one-hop phonetic relation search in the graph; for homographic puns, it refers to the number of candidate sense definitions retrieved based on lexical semantic entries. The statistics show that PunGraph produces a relatively large retrieval space on the SemEval 2017 dataset, with an average of 30.55 candidate words for each heterographic pun and 9.76 candidate sense definitions for each homographic pun. It is worth noting that a larger candidate space does not necessarily lead to better reasoning performance, as excessive candidates may introduce additional noise and increase the difficulty of candidate selection. Therefore, CRN is mainly used to characterize the coverage of the graph retrieval space, rather than as a direct indicator of downstream reasoning performance.

<table><tr><td>Pun Sentence</td><td>Pun Word</td><td>Label</td><td>Reasoning</td></tr><tr><td>I used to be a banker, but I lost interest.</td><td>interest</td><td>Homographic</td><td>financial interest earned from money, personal interest or enthusiasm.</td></tr><tr><td>When I grow up I wanna be a cup so I can fight crime.</td><td>cup</td><td>Heterographic</td><td>cop</td></tr><tr><td>Time flies like an arrow; fruit flies like a banana.</td><td>flies</td><td>Homographic</td><td>to move through the air, producing dual meanings in context.</td></tr><tr><td>The interrogators soon got a confection out of him.</td><td>confection</td><td>Heterographic</td><td>confession</td></tr></table>

Table 5: Examples of annotated pun types in the WebPun dataset.

<table><tr><td>Type</td><td>Total</td><td>Sentence</td><td>Words</td><td>OnePun</td><td>CRN</td></tr><tr><td>SemEval 2017 Dataset</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Heterographic</td><td>1,098</td><td>11.79</td><td>12,949</td><td>896</td><td>30.55</td></tr><tr><td>Homographic</td><td>1,298</td><td>11.71</td><td>15,194</td><td>928</td><td>9.76</td></tr><tr><td colspan="6">WebPun Dataset</td></tr><tr><td>Heterographic</td><td>5,061</td><td>6.95</td><td>35,190</td><td>1,663</td><td>22.50</td></tr><tr><td>Homographic</td><td>669</td><td>11.98</td><td>8,015</td><td>483</td><td>11.37</td></tr></table>

Table 6: Additional statistics of the SemEval-2017 and WebPun datasets. Total, Sentence, and Words denote the number of samples, average sentence length, and total number of words, respectively. OnePun refers to the number of unique pun words, while CRN denotes the Candidate Retrieve Number obtained from PunGraph.
<table><tr><td>WebPun</td><td>SemEval Het. (895)</td><td>SemEval Hom. (928)</td></tr><tr><td>Heterographic</td><td>170 (18.99%)</td><td>205 (22.09%)</td></tr><tr><td>Homographic</td><td>22 (2.46%)</td><td>144 (15.52%)</td></tr></table>

Table 7: Overlap of unique pun words between WebPun and SemEval-2017. SemEval-2017 is used as the denominator. Values indicate the number and percentage of overlapping unique pun words.

## B.2 Dataset Sample

Table 5 presents representative examples from WebPun with their corresponding pun words, types, and explanations.

## B.3 Dataset Novelty and Knowledge Coverage

To further examine the novelty of WebPun with respect to existing pun benchmarks, we analyze the overlap of unique pun words between WebPun and SemEval-2017. Using SemEval-2017 as the denominator, the overlap is 18.99% for heterographic puns and 15.52% for homographic puns, as shown in Table 7. The relatively limited overlap indicates that WebPun provides substantial complementary lexical coverage beyond SemEval-2017, despite the reuse of some high-frequency pun words across the two datasets.

We further evaluate the coverage of the constructed phonetic-semantic knowledge graph on both datasets. As shown in Table 8, the graph achieves 99.84% and 85.95% coverage for heterographic and homographic puns in WebPun, respectively, compared with 99.45% and 86.67% on SemEval-2017. Overall, the knowledge graph maintains consistently high coverage across both datasets, suggesting that the additional lexical diversity introduced by WebPun remains well supported by the phonetic and semantic resources used in PunGraph.

<table><tr><td>Dataset</td><td>Heterographic</td><td>Homographic</td><td>Average</td></tr><tr><td>WebPun</td><td>99.84%</td><td>85.95%</td><td>98.22%</td></tr><tr><td>SemEval-2017</td><td>99.45%</td><td>86.67%</td><td>92.53%</td></tr></table>

Table 8: Knowledge-graph coverage on WebPun and SemEval-2017.

Together, the relatively low overlap in unique pun words and the high knowledge-graph coverage suggest that WebPun complements SemEval-2017 with additional lexical and contextual diversity while remaining well supported by the structured knowledge used in PunGraph.

## C Retrieval Strategy

Regarding Section 3.2, we also provide the prompt template used to guide LLMs to process the retrieved candidate words according to the heterographic and homographic puns, as shown below:

## Prompt for Heterographic Pun reasoning

## System Prompt:

You are a linguist specializing in puns. Given a pun “{sentence}” and the pun word “{word}”, please choose the most likely intended real word in the original pun sentence.

Output only the option word of dictionary, with no additional text.

If there is no suitable candidate, please generate the most likely real word based on your linguistic knowledge and the context of the sentence, without being limited to the candidate list.

## User Input:

Sentence: The key to changing your performance ability is by tuning out criticism and staying musically octave.

Pun word: octave

Choices: {’artiste’, ’octant’, ’argive’, ’octet’, ’optics’, ’arctic’, ’octal’, ’fictive’, ’optic’, ’octavo’, ’active’}

<table><tr><td>Category</td><td>Phoneme Group</td><td>Similarity Basis</td></tr><tr><td>Consonant</td><td>{p, b, m}</td><td>bilabial; voicing / nasal variation</td></tr><tr><td>Consonant</td><td>{t, d, n}</td><td>alveolar; voicing / nasal variation</td></tr><tr><td>Consonant</td><td>{s, z}</td><td>alveolar fricatives; voicing contrast</td></tr><tr><td>Consonant</td><td>{k, g}</td><td>velar; voicing contrast</td></tr><tr><td>Consonant</td><td>{f, v}</td><td>labiodental fricatives; voicing contrast</td></tr><tr><td>Consonant</td><td>{1, r}</td><td>liquid consonants; approximant similarity</td></tr><tr><td>Vowel</td><td>{i, 1}</td><td>front vowels with adjacent height</td></tr><tr><td>Vowel</td><td>{e, ε, æ}</td><td>front vowels with similar tongue height</td></tr><tr><td>Vowel</td><td>{u, v}</td><td>back rounded vowels with adjacent height</td></tr><tr><td>Vowel</td><td>{, a}</td><td>back vowels with similar openness</td></tr><tr><td>Vowel</td><td>{ə, v}</td><td>central vowels with similar tongue position</td></tr><tr><td>Vowel</td><td>{5, ov, u}</td><td>back rounded vowel cluster</td></tr></table>

Table 9: Phonetically similar phoneme groups used for phonetic neighbor retrieval in PunGraph. Phoneme similarity is modeled as an undirected relation based on shared articulatory features.

![](images/4d2070a3546d18157cfac54af3193a2bf99e6717fc40bb67413470c5452cdbf6.jpg)

## D Phonetic Similarity

We first compute edit distance based on the IPA representations of words to determine their phonetic similarity. However, phonetic similarity is not uniform across phonemes. For example, the vowel /i/ is phonetically closer to /e/ than to /u/. To better capture these graded phonetic relationships, we construct a phoneme similarity table based on articulatory phonetic features <sup>5</sup> following (Association, 1999), as shown in Table 9. This allows phonemelevel similarity to be modeled at a finer granularity. During edit distance computation, phonemes belonging to the same similarity group are treated as equivalent matches.