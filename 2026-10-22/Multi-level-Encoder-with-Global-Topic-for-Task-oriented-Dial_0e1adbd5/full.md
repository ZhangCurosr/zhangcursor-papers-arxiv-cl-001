# Multi-level Encoder with Global Topic for Task-oriented Dialogue Summarization

Zhuoqi He, Peijie Huang<sup>\*</sup>, Yuhong Xu, Youming Peng, Mingzhi Xu, Xinyang Lin College of Mathematics and Informatics, South China Agricultural University, Guangzhou, China zhuoqihe497@163.com, pjhuang@scau.edu.cn, xuyuhong@scau.edu.cn, youmingpeng540@163.com, xdxulol@163.com, yizhiyang0111@gmail.com

Abstract—Task-oriented dialogue summarization aims to automatically extract key information to generate domain summaries to improve service efficiency and quality. Task-oriented dialogue is inherently logical and surrounds specific topic. How to effectively capture the dialogue topic and the most salient information becomes one of the major challenges of this task. In this paper, we propose a task-oriented dialogue summarization model utilizing a multi-level encoder with global topic (MLEGT). This model discovers temporal and spatial dependencies through a multi-level encoder, while simultaneously obtaining a simple yet effective global topic in the process of graph construction. It not only captures the crucial details but also breaks the long-range limit and discover the intrinsic structure across the dialogue. Then, a global topic guided pointer mechanism is added into the summrizer to capture ignored but important information, which is helpful to improve the accuracy of the summary. A comprehensive study of two public datasets, including a realworld large-scale Chinese police interrogation dataset and a medical reporting service dataset proves the superiority of our method on several strong baselines.

Index Terms—graph encoder, global topic, pointer mechanis

## I. INTRODUCTION

In an active industry service system, massive domainspecific task-oriented spoken dialogues conveying important information between human and human/agent are generated in real-time. Automatically summarizing such task-oriented dialogues may aid the archiving, indexing and retrieval of various records of oral communication such as sales interactions, medical diagnosis, complain report to improve the service. Therefore, how to transform a large number of task-oriented dialogues into natural, concise, and core-informed texts such as task-oriented dialogue summaries is gaining significantly increased attention [1]–[3].

Most of the existing dialogue summary works are mainly focused on open-domain dialogues, like chitchat [4]–[7], meeting [8]–[10], which are usually summarized by stringing all dialogue points to maintain an integral conversation flow [11]. Nevertheless, in task-oriented dialogues, dialogue speakers commonly have strong and clear motivations and aim to address issues about a specific task. To this end, recent works focus on developing topic or structure-aware models for task-oriented dialogue summarization. Zou et al. [1] leverage topic information that flows in the dialogue to help generate topic-aware summaries. Liu et al. [12] introduce a key point sequence as auxiliary label, which helps the model learn the logic of the summary. Zhao et al. [2] propose a dialogue state-aware summarization model to combine original dialogue text and structured dialogue state. However, these approaches remain at a coarse level in exploring the intrinsic structural features of dialogue. Gao et al. [3] propose a graph-based dialogue summarization model that uses structure fusion graph to capture the dialogue structure. However, this method is lacking in topic modeling.

![](images/53584cc237045cd771ffc172705896a69d1f91f6529fd5ded4f264dbc7723b04.jpg)  
Fig. 1. The Construction of Graphs.

To address the above issues, we design a graph-based taskoriented summarization model using multi-level encoder with global topic (MLEGT). A multi-level encoder is proposed to discover the temporal and spatial dependencies across the dialogue. Simultaneously a simple yet effective global topic is obtained in the process of graph construction, which helps us grasp the flow of information and discover the intrinsic structure across the dialogue. Then, we incorporate a global topic guided pointer mechanism into the summrizer to capture the ignored but important information while generating summary, which reduce the errors for better summarization. Figure 1 shows the construction of graphs how to capture the specific details across the dialogue. The experiments show that MLEGT achieves new state of the art on two public datasets, including a real-world large-scale Chinese police interrogation dataset and a medical reporting service dataset

![](images/4ff11c400f542f75c6d1d543052a10143e6ec07d9549e950da05f7c4128bb39f.jpg)  
Fig. 2. Overall Framework of MLEGT.

## II. PROPOSED MODEL

In this section, we introduce MLEGT model using multilevel encoder (Subsection II.A and II.B) with global topic guided pointer mechanism (in Subsection II.C) for taskoriented summarization. An overview is shown in Figure 2.

## A. Utterance Encoder

We employ the pre-trained BART [13] to encode each utterance independently: $\begin{array} { r l } { \{ h _ { i , 0 } , h _ { i , 1 } , . . . , h _ { i , L _ { \ast } ^ { i } } \} } & { { } = } \end{array}$ $E n c ( [ C L S ] , w _ { i , 1 } , . . . , w _ { i , L _ { \ast } ^ { i } } )$ , where Enc(·) is the encoder module in BART which outputs the vector representation $h _ { i , j }$ of j-th input word $w _ { i , j }$ in i-th utterance. Then we obtain the token-level Utterance $\mathsf { \bar { \boldsymbol { h } } } ^ { U } = \{ u _ { 1 } , . . . , u _ { L _ { d } } \}$ in this module.

## B. Graph Encoder

1) Graph Construction: In this subsection, following Gao et al. [3], we construct two types of graphs based on keywords and speakers, which is the Keywords Graph and the Speaker Graph respectively. We propose a new method in graph construction for effectively ensuring the connectivity of the graph. Our method is significantly different from Gao et al. [3] in the following two points: (1) We set the diagonal of the adjacency matrix be 1 so that each utterance can not only discover the relations among their neighbors but also focus on themselves. (2) We add a global node into each graph, which can help us obtain a simple but effective global topic in the process of graph construction.

Keywords Graph. The construction of Keywords Graph aims to discover the spatial dependencies among different utterances in the dialogue. Firstly, we obtain an utteranceskeywords adjacent matrix A based on keywords extracted from utterances by SpaCy. Each value $A _ { i , j }$ in the adjacent matrix corresponds the frequency of the j-th keyword occurrences in i-th utterance. Let A multiply by its own transpose matrix, and then we obtain the initial Keywords Graph $G _ { k }$

$$
G _ { k } = A A ^ { T } ,\tag{1}
$$

In addition, we set the diagonal of the Keywords Graph to 1, enabling each utterance to also pay attention to itself.

Speakers Graph. The construction of Speaker Graph aims to effectively gather the information of the same speaker and reduce the dispersion of information. Firstly, we obtain an utterances-speakers adjacent matrix B based on different speakers in the dialogue. Each value $B _ { i , j }$ in the adjacent matrix corresponds the specific speaker in i-th utterance. Here, we use a hot-dot to represent each speaker. Let B multiply by its own transpose matrix, and then we obtain the initial Speakers Graph $G _ { s }$

$$
G _ { s } = B B ^ { T } ,\tag{2}
$$

In addition, we set the diagonal of the Speakers Graph to 1, enabling each utterance to also pay attention to itself.

Global Node. Furthermore, we incorporate a global node into each graph, which is a hyper node [14] connected to all nodes, to aggregate non-local information across the dialogue. Each node builds relationships with its neighboring nodes through edges, while a global node connects to all nodes, serving as a central hub for aggregating global information. It can interact with all nodes and capture important features in the process of information propagation, thereby enhancing the model’s overall understanding and modeling capability of the graph structure.

2) Graph update: Graph encoder [15] is used to digest the structural information and get updated node representations. The token-level embeddings $h ^ { U }$ from the Utterance Encoder is added as the node representations for each graph. Given the constructed $G _ { k } = ( V _ { k } , E _ { k } )$ and $G _ { s } = ( V _ { s } , K _ { s } )$ , nodes V correspond to the utterances for each dialogue while edges E correspond to the similarity relations.

Utterance-Level embeddings. To update the representations of all the nodes, we apply a Graph Convolutional Network [16] to the Keywords Graph and Speaker Graph

respectively.

$$
h _ { k } ^ { G } = G C N ( A d d \& N o r m ( h ^ { U } , \sum _ { j \in N _ { i } } \frac { 1 } { | N _ { i } | } W _ { 1 } V _ { j } ) ,\tag{3}
$$

$$
h _ { s } ^ { G } = G C N ( A d d \& N o r m ( h ^ { U } , \sum _ { j \in N _ { i } } \frac { 1 } { | N _ { i } | } W _ { 2 } V _ { j } ) ,\tag{4}
$$

where Add&Norm denotes residual connection and layer normalization while $N _ { i }$ denotes the neighborhood of the i-th node. $h _ { k } ^ { G }$ and $h _ { s } ^ { G }$ are the outputs from the Utterance Encoder.

Since GCN does not share parameters, these two graphs together are then concatenated as $h _ { c } ^ { G }$

$$
h _ { c } ^ { G } = c o n c a t [ h _ { k } ^ { G } [ : - 1 ] ; h _ { s } ^ { G } [ : - 1 ] ] ,\tag{5}
$$

where $h _ { k } ^ { G } [ : - 1 ]$ and $h _ { s } ^ { G } [ : - 1 ]$ are the representations of all nodes except the global node for each graph respectively. Eventually, it is passed into the cross attention layer, which utilize the dual cross attention mechanism [17] across the graphs to further explore the long-range dependencies and uncover the intrinsic structure. And we obtain the utterancelevel embeddings,

$$
h _ { u } ^ { G } = C r o s s A t t n ( A d d \& N o r m ( h _ { c } ^ { G } ) , h ^ { U } , h ^ { U } ) ,\tag{6}
$$

Global Topic Representation. To better capture the flow of information and uncover the intrinsic structure across the dialogue, we introduced an additional global node that connects to all other nodes. It not only captures the implicit relationships among other nodes, but also forms a representation of each node’s contribution to the overall theme. It dynamically updates alongside the other nodes and we regard it as a global topic representation:

$$
h _ { t } ^ { G } = h _ { k } ^ { G } [ - 1 ] + h _ { s } ^ { G } [ - 1 ] ,\tag{7}
$$

where $h _ { k } ^ { G } [ - 1 ]$ and $h _ { s } ^ { G } [ - 1 ]$ are the representations of the global node for each graph respectively.

## C. Global Topic Guided Pointer Decoder

1) Global Topic Guided Pointer Mechanism: A pointer mechanism [18] is added into our model, aiming to capture the ignored but important information during the feature extracting and training.

Firstly, we obtain the last hidden states $h ^ { * }$ from the decoder. And we project $h ^ { * }$ and $h _ { t } ^ { G }$ into two new vector spaces separately to get the $P _ { c o p y }$

$$
P _ { c o p y } = s i g m o i d ( W _ { 3 } h ^ { * } + W _ { 4 } h _ { t } ^ { G } ) ,\tag{8}
$$

Then, we utilize the token-based embeddings $h ^ { U }$ and the global topic $h _ { t } ^ { G }$ to obtain the context vector $c ^ { * } .$ , helping the decoder to decide the position where to copy directly.

$$
c ^ { * } = m e a n ( W _ { 5 } h ^ { U } + W _ { 6 } h _ { t } ^ { G } , d i m = - 1 ) ,\tag{9}
$$

We obtain the decoder attention $a _ { t }$ from the decoder and do some transformation on it. Let it multiply by $P _ { c o p y }$ and get the new attention $\tilde { a } _ { t }$

$$
\tilde { a } _ { t } = m e a n ( a _ { t } , d i m = 1 ) \otimes P _ { c o p y } ,\tag{10}
$$

Finally, multiply $\tilde { a } _ { t } , P _ { c o p y }$ and $c ^ { * } ,$ , and we get the final copy generation $\tilde { P } _ { c o p y } .$

$$
\tilde { P } _ { c o p y } = c ^ { * } \otimes \tilde { a } _ { t } \otimes P _ { c o p y }\tag{11}
$$

2) Summary Generating: To integrate the utterance-level embeddings into the summary generation process, we employ a dual cross-attention mechanism in the model [17].

We first apply the self-attention on the masked output summary embeddings, and then use the output embedding $h ^ { O }$ to cross-attend to the token-level dialogue hidden states $h ^ { U }$ produced by the utterance encoder, and get the $P ^ { q }$

$$
P ^ { q } = M H A t t ( h ^ { O } , h ^ { U } ) ,\tag{12}
$$

where MHAtt is the standard multi-head attention layer. In a similar way, we apply a multi-head graph attention layer to the utterance-based hidden states $h _ { u } ^ { G }$ and get the $P ^ { s }$ , which aggregate useful knowledge from the updated graph nodes according to the state of each decoding step:

$$
P ^ { s } = M H A t t ( P ^ { q } , h _ { u } ^ { G } ) ,\tag{13}
$$

Eventually, we apply a fully connected feed-forward network on $P ^ { s }$ and get the generation distribution $P _ { g e n }$ . Then, we combine $P _ { g e n }$ with $\tilde { P } _ { c o p y }$ , and obtain the final distribution over the vocabulary of the generated summary.

$$
P _ { p r o b } = \tilde { P } _ { c o p y } + ( 1 - \tilde { P } _ { c o p y } ) \otimes P _ { g e n } .\tag{14}
$$

## D. Training Objective

We optimize all parameters of MLEGT by employing the cross-entropy loss between the generated summary and the ground truth summary as the training objective. And we initialize the relevant parameters in our Transformer-based utterance encoder and summary generator with the parameters from the pre-trained language model BART.

## III. EXPERIMENTS

## A. Experimental Setup

1) Datasets: The datasets used in this experiment are LC-SPIRT<sup>1</sup> [19] and Medical ${ \mathrm { D } } { \mathsf { S } } ^ { 2 }$ [20]. LCSPIRT is a large-scale Chinese text summarization corpus containing 38,500 textsummary pairs derived from police inquiry dialogue records, and the texts are deidentified to ensure privacy. Medical DS, on the other hand, consists of 3,260 dialogue-summary pairs collected from the pediatric department of a Chinese online healthcare community. For both datasets, we use the originally proposed train, validation, and test splits, comprising 70%, 10%, and 20% of the data, respectively.

2) Implementation Details: We use BART as the tokenlevel encoder, and the outputs are utilized into the follow utterance-level encoder as well. We apply Adam with a learning rate 1e-5 for BART and the training epoch is set to 15. All Transformer blocks have 4 layers, 8 heads, 1024 hidden units. All hyper-parameters tuning is conducted on the validation set.

3) Baselines: We compare MLEGT with following baselines: PGN [18], BART [13], T5-pegasus [21], TDS+NTM [1], GPT-4o (few-shot) [22] and SDDS [3]. We use the SDDS without the discourse parsing toolkit for fair comparison.

TABLE I  
ROUGE-1, ROUGE-2 AND ROUGE-L SCORES FOR DIFFERENT MODELS ONLCSPIRT AND MEDICAL DS TEST SET. THE NUMBERS WITH \* INDICATETHAT THE IMPROVEMENT OF OUR MODEL OVER ALL BASELINES ISSTATISTICALLY SIGNIFICANT WITH p < 0.05 UNDER T-TEST.
<table><tr><td rowspan="2">Model</td><td colspan="3">LCSPIRT</td><td colspan="3">Medical DS</td></tr><tr><td>Rouge-1</td><td>Rouge-2</td><td>Rouge-L</td><td>Rouge-1</td><td>Rouge-2</td><td>Rouge-L</td></tr><tr><td>PGN</td><td>40.03</td><td>25.14</td><td>31.7</td><td>62.7</td><td>55.77</td><td>57.23</td></tr><tr><td>BART</td><td>73.36</td><td>56.34</td><td>71.43</td><td>63.53</td><td>51.15</td><td>56.55</td></tr><tr><td>T5-pegasus</td><td>77.27</td><td>69.10</td><td>72.35</td><td>58.08</td><td>51.34</td><td>52.01</td></tr><tr><td>TDS+NTM</td><td>74.78</td><td>60.37</td><td>72.84</td><td>63.85</td><td>51.82</td><td>56.53</td></tr><tr><td>GPT-4o (few-shot)</td><td>80.04</td><td>67.56</td><td>78.59</td><td>61.85</td><td>45.39</td><td>55.25</td></tr><tr><td>SDDS</td><td>82.27</td><td>71.99</td><td>79.13</td><td>64.9</td><td>52.19</td><td>58.42</td></tr><tr><td>MLEGT (Ours)</td><td>85.77*</td><td>75.47*</td><td>83.34*</td><td>65.09</td><td>52.57</td><td>58.81*</td></tr><tr><td>w/o global topic</td><td>83.96</td><td>73.25</td><td>80.69</td><td>64.82</td><td>52.08</td><td>58.43</td></tr><tr><td>w/o pointer</td><td>85.47</td><td>74.34</td><td>82.84</td><td>64.34</td><td>52.17</td><td>57.87</td></tr></table>

## B. Experiments Results

1) Main Results: We assessed the quality of summarization using Rouge F1 score, including Rouge-1, Rouge-2, and Rouge-L metrics [23]. Table I presents the results on the test set for various models. We highlight best and second best results. Notably, our model surpasses all compared baselines, achieving state-of-the-art performance across all metrics. Specifically, on the LCSPIRT dataset, our model exhibits significant improvements, up by +3.5% in ROUGE-1, +3.48% in ROUGE-2, and +4.21% in ROUGE-L scores. We also find that our MLEGT can achieve consistently better performance than the strong baselines on Medical DS. To our surprise, PGN that performs poorly on LCSPIRT gets the best ROUGE-2 score on Medical DS. We conjecture the main reason is that there are numerous medical terms scattered throughout the long conversation structure in the medical dataset, and the pointer mechanism is good at capturing such information while generating summary.

2) Ablation Study: We further investigate the effect of each mechanism of MLEGT by removing the global node and the pointer mechanism, respectively. The results are also shown in Table I. The results of “w/o global topic” and “w/o pointer” demonstrate a notable decline on both LCSPIRT and Medical DS datasets. This decline underscores the critical role of the proposed global topic as well as the global topic guided pointer mechanism.

3) Human Evaluation: We conducted human evaluation to qualitatively evaluate the generated summaries. As our research focuses on task-oriented dialogue summarization, the informativeness is more critical than the conciseness and fluency for the generated summaries. We focus on how well the generated summaries covered the most important content and compare the accuracy of entity generation across different methods. We randomly selected 200 samples from the LCSPIRT test set and asked five human volunteers to manually annotate entity information such as the Start and End Year/Month/Day/Time, Character, and Place. Our volunteers then compared the entities in the generated summaries of MLEGT and SDDS with those in the reference summaries to count the correct rate of entity generation. As illustrated in Fig. 3, MLEGT demonstrates better quality in entity recognition, particularly for End Month, End Date, End Time, Character, and Place. It demonstrates that MLEGT can identify the more

![](images/3f1b6520caffe135349b5ae40ad1d995f441bf717fe17613b69c7f843e17bc1b.jpg)  
Fig. 3. The correct rate of entity generation on LCSPIRT.  
TABLE II

A GENERATED SUMMARY CASE. THE UNDERLINED PART IN THE REFERENCE REFERS TO THE ENTITY INFORMATION, AND THE RED PART IN THE GENERATED ABSTRACT IS THE WRONG GENERATED CONTENT.

Ref Summary: 2017年02月06日08时到2017年02月06日17时下午1218时张某某在 蝶湖湾   
花园<sub>1723201</sub> 被嫌疑人以其他入室手段窃走黑色华硕牌<sub>A45</sub>笔记本电脑<sub>1</sub>台。   
From 08:00 on February 6th, 2017 to 17:00 on February 6th, 2017 at 1218 p.m., a black   
Asus A45 laptop from Zhang was stolen by the suspect by other means in the Butterfly Lake   
Bay Garden 1723201.   
SDDS (baseline)<sub>: 2017</sub>年<sub>02</sub>月<sub>06</sub>日<sub>08</sub>时到<sub>2017</sub>年<sub>2</sub>月<sub>06</sub>某某在 昆山市玉山镇蝶湖湾花园 <sub>3</sub>   
栋被嫌疑人以其他入室手段窃走 台黑色华硕牌 型笔记本电脑 现金人民币 元。   
某某。银行卡等物品。都是网络通讯软件的。   
From 08:00 on February 6th, 2017 to February 6th, 2017, a black Asus A45 laptop from XX   
was stolen by the suspect by other means in the 3 building of the Butterfly Lake Bay Garden   
in Yushan Town, Kunshan City. . . The cash is 500 yuan. Something. Bank cards and other   
items. It’s all from the network communications software.   
MLEGT: 2017年02月06日08时到2017年03月06号17时下午1218时张某某在蝶湖湾花园 17232015   
被嫌疑人以其他入室手段窃走黑色华硕牌<sub>A45</sub>型笔记本电脑<sub>1</sub>台。   
From 08:00 on February 6th, 2017 to 17:00 on March 6th, 2017, at 1218 PM, a black Asus   
A45 laptop from Zhang was stolen by the suspect by other means in Butterfly Lake Bay   
Garden17232015.

important parts across the dialogue, thereby increasing the informativeness compared to the baseline.

4) Case Analysis: In this subsection, we present an close look for our MLEGT and the SDDS model via case studies. Table II presents a summary example from the LCSPIRT dataset generated by the MLEGT model, in comparison with the baseline SDDS model and the reference. The baseline SDDS model often tends to generate redundant contents and produce some crucial information errors. Compared to the baseline, our MLEGT model makes fewer errors and captures more critical information throughout the dialogue. This further demonstrates the effectiveness of the proposed global topic and the global topic guided pointer mechanism.

## IV. CONCLUSION

In this paper, we propose a model that utilizes a multi-level encoder with global topic (MLEGT) for task-oriented dialogue summarization, which enables to discover the temporal and spatial dependencies while obtain a simple yet effective global topic to better capture the crucial information. Experiments on two public data sets demonstrate the effectiveness of our proposed model in quantitative and qualitative evaluation.

## V. ACKNOWLEDGEMENTS

This work was supported by the National Natural Science Foundation of China (62306119 and 71472068).

## REFERENCES

[1] Y. Zou, L. Zhao, Y. Kang, J. Lin, M. Peng, Z. Jiang, C. Sun, Q. Zhang, X. Huang, and X. Liu, “Topic-oriented spoken dialogue summarization for customer service with saliency-aware topic modeling,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. vol. 35. Virtual Event: AAAI, 2021, pp. 14 665–14 673.

[2] L. Zhao, F. Zheng, K. He, W. Zeng, Y. Lei, H. Jiang, W. Wu, W. Xu, J. Guo, and F. Meng, “Todsum: Task-oriented dialogue summarization with state tracking,” CoRR, vol. abs/2110.12680, 2021. [Online]. Available: https://arxiv.org/abs/2110.12680

[3] S. Gao, X. Cheng, M. Li, X. Chen, J. Li, D. Zhao, and R. Yan, “Dialogue summarization with static-dynamic structure fusion graph,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics, vol. vol. 1: Long Papers. Toronto, Canada: Association for Computational Linguistics, 2023, pp. 13 858–13 873.

[4] Y. Lei, Y. Yan, Z. Zeng, K. He, X. Zhang, and W. Xu, “Hierarchical speaker-aware sequence-to-sequence model for dialogue summarization,” in IEEE International Conference on Acoustics, Speech and Signal Processing, ICASSP 2021, June 6-11, 2021, pp. 7823–7827. [Online]. Available: https://doi.org/10.1109/ICASSP39728.2021.9414547

[5] Y. L. et al, “A finer-grain universal dialogue semantic structures based model for abstractive dialogue summarization,” in Findings of the Association for Computational Linguistics: EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 16-20 November, 2021, pp. 1354– 1364. [Online]. Available: https://doi.org/10.18653/v1/2021.findingsemnlp.117

[6] Y. Li, B. Peng, P. He, M. Galley, Z. Yu, and J. Gao, “DIONYSUS: A pre-trained model for low-resource dialogue summarization,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023, A. Rogers, J. L. Boyd-Graber, and N. Okazaki, Eds. Association for Computational Linguistics, 2023, pp. 1368–1386. [Online]. Available: https://doi.org/10.18653/v1/2023.acllong.76

[7] Y. K. Jaeah You, “Topic-informed dialogue summarization using topic distribution and prompt-based modeling,” in The 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023, pp. 5657–5663.

[8] Z. Zhao, H. Pan, C. Fan, Y. Liu, L. Li, and M. Yang, “Abstractive meeting summarization via hierarchical adaptive segmental network learning,” in The World Wide Web Conference, WWW 2019, May 13-17, 2019, pp. 3455–3461. [Online]. Available: https://doi.org/10.1145/3308558.3313619

[9] M. Li, L. Zhang, H. Ji, and R. J. Radke, “Keep meeting summaries on topic: Abstractive multi-modal meeting summarization,” in Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers, pp. 2190–2196. [Online]. Available: https://doi.org/10.18653/v1/p19-1210

[10] S. Sun, R. Yuan, W. Li, Z. Cao, and S. Li, “Dialogue acts enhanced extract-abstract framework for meeting summarization,” Inf. Process. Manag., vol. 61, no. 2, p. 103635, 2024. [Online]. Available: https://doi.org/10.1016/j.ipm.2023.103635

[11] G. Shang, W. Ding, Z. Zhang, A. J. Tixier, P. Meladianos, M. Vazirgiannis, and J. Lorre, “Unsupervised abstractive meeting´ summarization with multi-sentence compression and budgeted submodular maximization,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, ACL 2018, Melbourne, Australia, July 15-20, 2018, Volume 1: Long Papers, pp. 664–674. [Online]. Available: https://aclanthology.org/P18-1062/

[12] C. Liu, P. Wang, J. Xu, Z. Li, and J. Ye, “Automatic dialogue summary generation for customer service,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, KDD 2019, Anchorage, AK, USA, August 4-8, 2019, pp. 1957– 1965. [Online]. Available: https://doi.org/10.1145/3292500.3330683

[13] M. Lewis, Y. Liu, N. Goyal, M. Ghazvininejad, A. Mohamed, O. Levy, V. Stoyanov, and L. Zettlemoyer, “BART: denoising sequence-tosequence pre-training for natural language generation, translation, and comprehension,” in Proceedings of the 58th Annual Meeting of the Association for Computational. Online: Association for Computational Linguistics, 2020, pp. 7871–7880.

[14] X. Feng, X. Feng, B. Qin, and X. Geng, “Dialogue discourse-aware graph model and data augmentation for meeting summarization,” in

Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence. Montreal, Canada: ijcai.org, 2021, pp. 3808–3814.

[15] X. Feng, X. Feng, and B. Qin, “Incorporating commonsense knowledge into abstractive dialogue summarization via heterogeneous graph networks,” in China National Conference on Chinese Computational Linguistics, ser. Lecture Notes in Computer Science, vol. vol. 12869. Hohhot, China: Springer, 2021, pp. 127–142.

[16] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” CoRR, vol. abs/1609.02907, 2016.

[17] X. Cheng, S. Gao, L. Liu, D. Zhao, and R. Yan, “Neural machine translation with contrastive translation memories,” CoRR, vol. abs/2212.03140, 2022.

[18] A. See, P. J. Liu, and C. D. Manning, “Get to the point: Summarization with pointer-generator networks,” CoRR, vol. abs/1704.04368, 2017.

[19] X. Xi, Z. Pi, and G. Zhou, “Global encoding for long chinese text summarization,” ACM Transactions on Asian and Low-Resource Language Information Processing (TALLIP), vol. 19, no. 6, pp. 1–17, 2020.

[20] Z. Wei, Q. Liu, B. Peng, H. Tou, T. Chen, X. Huang, K. Wong, and X. Dai, “Task-oriented dialogue system for automatic diagnosis,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics, vol. vol. 2: Short Papers. Melbourne, Australia: Association for Computational Linguistics, 2018, pp. 201– 207.

[21] J. Zhang, Y. Zhao, M. Saleh, and P. J. Liu, “PEGASUS: pre-training with extracted gap-sentences for abstractive summarization,” in Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 11 328–11 339. [Online]. Available: http://proceedings.mlr.press/v119/zhang20ae.html

[22] S. Shahriar, B. D. Lund, N. R. Mannuru, M. A. Arshad, K. Hayawi, R. V. K. Bevara, A. Mannuru, and L. Batool, “Putting gpt-4o to the sword: A comprehensive evaluation of language, vision, speech, and multimodal proficiency,” CoRR, vol. abs/2407.09519, 2024. [Online]. Available: https://doi.org/10.48550/arXiv.2407.09519

[23] C.-Y. Lin and F. J. Och, “Automatic evaluation of machine translation quality using longest common subsequence and skip-bigram statistics,” in Proceedings of the 42nd annual meeting of the association for computational linguistics. Barcelona, Spain: ACL, 2004, pp. 605–612.