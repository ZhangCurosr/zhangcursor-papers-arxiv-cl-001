# ExpArt-KG: Artwork Image Description Generation through Iterative Exploration of Knowledge Graphs

Yuta Kato<sup>1</sup>, Shintaro Ozaki<sup>2</sup>, Kazuki Hayashi<sup>2</sup>, Yusuke Sakai<sup>2</sup>, Hidetaka Kamigaito<sup>2</sup>, Katsuhiko Hayashi<sup>1,</sup> <sup>2</sup>, Taro Watanabe<sup>2</sup>

<sup>1</sup>The University of Tokyo, <sup>2</sup>Nara Institute of Science and Technology (NAIST)

{ukato6209, katsuhiko-hayashi}@g.ecc.u-tokyo.ac.jp

ozaki.shintaro.ou6@naist.ac.jp

{hayashi.kazuki.hl4, sakai.yusuke.sr9, kamigaito.h, taro}@is.naist.jp

## Abstract

Large Vision-Language Models (LVLMs) achieve strong performance on imagegrounded text generation and visual question answering. However, it remains difficult for them to comprehensively and accurately describe the factual relations among the entities and concepts associated with the objects depicted in an image. In this work, we propose a framework that efficiently exploits factual information from a knowledge graph via retrieval-augmented generation (RAG), with the goal of enabling LVLMs to generate detailed and accurate image explanations. Specifically, our method alternates between answer generation and knowledge-graph retrieval, and controls the search using a correctness judgment, thereby acquiring the necessary and sufficient factual information efficiently. We also construct a knowledge graph for the artwork domain (ExpArt-KG), in which the correspondence between images and entities is unambiguous. Applying the proposed method to this knowledge graph, we show experimentally that it improves the level of detail of artwork explanations and reduces the retrieval cost of external knowledge while maintaining generation quality comparable to that of iterating a fixed number of times.

## 1 Introduction

Large Vision-Language Models (LVLMs) (Liu et al., 2023; Bai et al., 2025b; Abdin et al., 2024) achieve strong performance on image-grounded text generation and visual question answering (Li et al., 2025b; Bai et al., 2023; Dai et al., 2023), one instance of which is the ability to generate detailed explanations of an image. In such settings, an LVLM is required not merely to recognize the people and objects contained in an image, but also to connect them with their surrounding knowledge.

However, it has been pointed out that LVLMs struggle to comprehensively and accurately explain the factual relations among the entities associated with what they recognize (Hayashi et al., 2024).

A practical way to compensate for this limitation of LVLMs is retrieval-augmented generation (RAG) (Lewis et al., 2020), which utilizes external knowledge. This framework generates answers on the basis of retrieved and reliable external information, thereby suppressing hallucinations and supplying the model with detailed knowledge. In particular, for handling complex factual relations among entities comprehensively and accurately, it is effective to exploit a knowledge graph, whose systematic structure makes the factual relations among entities explicit. While such an approach can extend the knowledge of a model without additional training, existing knowledge-graph-based methods fix the search depth or the number of iterations in advance, which creates a trade-off: a shallow search yields insufficient information, whereas a deep search increases the retrieval cost (Li et al., 2025a). Consequently, a general framework for efficiently retrieving useful factual relations and properly using them for explanation generation has not been sufficiently established.

In this work, we propose an iterative RAG framework that introduces a correctness judgment of the generated answer by a Large Language Model (LLM) (Yang et al., 2025; Achiam et al., 2023; Comanici et al., 2025; Grattafiori et al., 2024) acting as a judge (Zheng et al., 2023), and that dynamically explores a knowledge graph and regenerates the answer on the basis of that judgment, aiming at detailed image explanation generation by LVLMs. In addition, we construct a knowledge graph (ExpArt-KG) for extending the knowledge of LVLMs, targeting the artwork domain, in which a primary image and its entity are in one-to-one

correspondence.

Evaluating the explanation generation ability of an LVLM using the proposed method together with ExpArt-KG under evaluation metrics that cover several viewpoints, we confirm that iterative answer generation explores the knowledge graph efficiently and improves the level of detail of the generated explanations. We further show that iteration control based on the correctness judgment reduces the retrieval cost of external knowledge while maintaining generation quality comparable to that of iterating a fixed number of times.

## 2 Proposed Method

The proposed retrieval method iteratively performs a local exploration of the nodes corresponding to the entities contained in the query given to the LVLM, together with the generation and verification of an answer that uses the retrieved factual information. In this way, the method obtains useful factual information while keeping the retrieval cost of external knowledge low. Following prior work in which combining LLM self-verification with RAG through iterative answering was shown to improve performance (Hayashi et al., 2025), answer verification by an LLM is introduced so as to achieve both efficient exploration and high-quality explanation generation. The prompts used in the method are detailed in Appendix A.1.

The concrete procedure of the proposed method is as follows.

(1) Triple retrieval. Among the entities contained in the query, those that exist as nodes of the knowledge graph are extracted, and the triples containing them are retrieved.

(2) Triple selection. The triples obtained in Step 1 are ranked by TF-IDF (Salton and Buckley, 1988), and the top 10 triples are selected for each entity. In this ranking, the constituents of a triple are regarded as terms, and the set of triples containing a particular entity as a document. The detailed computation of TF-IDF is given in Appendix A.4.

(3) Answer generation. An augmented query is constructed by appending the triples selected in Step 2 to the original query, and it is given to the LVLM together with the target artwork (image) to generate an answer.

(4) Answer verification. The original query and the answer generated in Step 3 are given to an LLM, which verifies whether the answer is correct. Using forced decoding, the LLM outputs either “True” or “False”. If “True” is obtained, the answer is taken as the final answer of the LVLM; if “False” is obtained, triple retrieval and answer generation are performed again.

(5) Triple re-retrieval. When “False” is obtained in Step 4, triples are retrieved again by the procedures of Steps 1 and 2, using either the original query or the answer generated immediately before, and the answer is then regenerated and verified through Steps 3 and 4. This process is repeated until “True” is obtained or the maximum number of iterations is reached.

## 3 Knowledge Graph Construction

This section introduces a general procedure for constructing a knowledge graph suited to generating explanations of images. Based on the textual information contained in an arbitrary questionanswering dataset, entities and their relations are extracted by reference to an external knowledge base, and a knowledge graph is thereby constructed. The concrete construction procedure is as follows.

(1) Extraction of node candidates. From the strings contained in the questions, reference explanations, and any available metadata (such as the titles of images) of a question-answering dataset, the expressions that are candidate nodes of the knowledge graph are extracted.

(2) Node selection. Among the node candidates extracted in Step 1, those that exist as titles of English Wikipedia articles are adopted as entities. Furthermore, highly ambiguous words, for which multiple corresponding Wikidata (Vrandeciˇ c and´ Krötzsch, 2014) identifiers exist, and expressions denoting general concepts are excluded from the nodes, so that the resulting knowledge graph is dense and specialized to the target domain. This leaves only entities that are specific to, and unambiguous within, the target domain.

(3) Edge construction. For each node selected in Step 2, the corresponding Wikidata entity is referred to, and the predicates defined among them are obtained. Defining these predicates as edges that express semantic relations between nodes yields a knowledge graph in which the factual relations among entities are structured.

In this work, we applied this procedure to ExpArt (Hayashi et al., 2024) to construct ExpArt-KG, a knowledge graph about artworks.

## 4 Experimental Setup

Models. We used Qwen3-VL (Bai et al., 2025a) as the LVLM that generates image explanations, and Qwen3 (Yang et al., 2025) as the LLM that verifies answers (we refer to it as the validator LLM).

Dataset. From the test data of ExpArt,<sup>1</sup> we targeted the instances whose image URLs<sup>2</sup> were valid as of December 2025 and whose answers contained at least one entity of the constructed ExpArt-KG. From these we sampled approximately 25%, which we used for evaluation. Details are given in Appendix A.3.

Input settings. In this experiment, we used the following two question settings for each image.

(1) With Title: The LVLM generates an explanation from the image and a question that contains the title. The validator LLM judges correctness on the basis of the question containing the title and the answer.

(2) Without Title: When generating the image explanation, the LVLM is given the image and a question that does not contain the title. The validator LLM is likewise given only the question without the title and the answer, so that the correctness judgment is made solely on the basis of the logical consistency of the answer.

Through these settings, we examine the difference in generation and verification performance depending on the presence of the title, as well as the effect of the image title on the correctness judgment of an answer.

Knowledge graph retrieval. We define three variants of TF-IDF-based triple ranking: (1) treating predicates as terms and using their scores (PID), (2) treating adjacent entities as terms and using their scores (QID), and (3) treating both as terms and using the sum of the two scores (PID-QID). Details are given in Appendix A.4.

RAG settings. To examine the effectiveness of ExpArt-KG and of the proposed method (RAG-Validate), we compared RAG-Validate with a method that generates an answer only once without

RAG (Baseline) and a method that uses RAG and repeats answer generation five times (RAG-Loop5).

Evaluation metrics. For evaluation, we used BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and BERTScore (Zhang et al., 2020), which are standard in natural language generation tasks, in order to measure the lexical and semantic similarity between the generated text and the reference explanation. In addition, to measure the level of detail of explanations about artworks, we used Entity Coverage, Entity F1, and Entity Cooccurrence (Hayashi et al., 2024). The corresponding formulas are given in Appendix A.5.

(1) Entity Coverage evaluates the extent to which the generated text covers the artwork-related entities in the reference explanation, under two settings: exact match and partial match.

(2) Entity F1 compares the artwork-related entities contained in the generated text and in the reference explanation, and evaluates the frequency of entity occurrences and their appropriate usage.

(3) Entity Cooccurrence focuses on the cooccurrence relations among entities across the whole text and evaluates the coverage of cooccurrences. It adopts a length penalty analogous to the brevity penalty of BLEU (Papineni et al., 2002) to prevent overrating redundant explanations.

## 5 Results and Analysis

Table 1 reports the scores of the evaluation metrics under each setting. For all metrics except BERTScore, RAG-Validate and RAG-Loop5 improved over the Baseline. While BERTScore stays at a high level regardless of whether RAG is used, marked improvements are observed for Entity F1, Entity Coverage, and Entity Cooccurrence. These results indicate that, although the Baseline already produces grammatically fluent text, using ExpArt-KG and the proposed method substantially improves the accuracy and comprehensiveness of the entities contained in the explanations, and can raise the level of detail of the explanations with respect to the depicted subject and its surrounding knowledge. Regarding the way TF-IDF scores are computed, the largest improvement in the With Title setting was observed when only adjacent entities were used as the constituent terms. This suggests that using the entities of a triple, rather than its predicates, for retrieval improves the ability to identify highly specific entities. Moreover, for questions containing the title of the artwork, RAG-Validate attains scores comparable to RAG-Loop5.

<table><tr><td rowspan="2">RAG setting</td><td rowspan="2">TF-IDF</td><td rowspan="2">BLEU</td><td colspan="3">ROUGE</td><td rowspan="2">BERT Score</td><td rowspan="2">Entity F1 (1)</td><td colspan="2">Entity Coverage (↑)</td><td colspan="3">Entity Cooccurrence (↑)</td><td rowspan="2"></td><td rowspan="2">Output length</td></tr><tr><td>1</td><td></td><td>L</td><td>Exact</td><td></td><td>Partial n=0</td><td>n=1</td><td>n=2 n=∞</td></tr><tr><td colspan="10">With Title</td><td colspan="7"></td></tr><tr><td>RAG-Loop5</td><td>PID</td><td>3.22</td><td>30.73</td><td>9.02</td><td>19.02</td><td>84.51</td><td>41.47</td><td>38.52</td><td>45.07</td><td>7.93</td><td>8.77</td><td>8.89</td><td>8.90</td><td></td><td>121.6</td></tr><tr><td>RAG-Loop5</td><td>PID-QID</td><td>3.12</td><td>30.89</td><td>9.01</td><td>18.97</td><td>84.53</td><td>43.24</td><td>39.95</td><td>46.36</td><td>8.16</td><td>9.11</td><td>9.53</td><td></td><td>9.45</td><td>121.2</td></tr><tr><td>RAG-Loop5</td><td>QID</td><td>3.22</td><td>31.17</td><td>9.11</td><td>19.20</td><td>84.54</td><td>43.85</td><td>40.73</td><td>47.10</td><td></td><td>8.60</td><td>9.61</td><td>9.92</td><td>9.79</td><td>123.8</td></tr><tr><td>RAG-Validate</td><td>PID</td><td>3.15</td><td>30.55</td><td>8.92</td><td>18.90</td><td>84.53</td><td>41.89</td><td>37.93</td><td>44.45</td><td></td><td>7.89</td><td>8.43</td><td>8.46</td><td>8.41</td><td>116.3</td></tr><tr><td>RAG-Validate</td><td>PID-QID</td><td>3.18</td><td>30.66</td><td>8.96</td><td>18.92</td><td>84.55</td><td>42.17</td><td>38.69</td><td>45.34</td><td>8.47</td><td></td><td>8.71</td><td>9.01</td><td>8.95</td><td>118.1</td></tr><tr><td>RAG-Validate</td><td>QID</td><td>3.22</td><td>30.85</td><td>9.07</td><td>18.96</td><td>84.56</td><td>42.96</td><td>39.46</td><td>45.99</td><td></td><td>8.53</td><td>9.19</td><td>9.52</td><td>9.45</td><td>120.1</td></tr><tr><td>Baseline</td><td>一</td><td>1.18</td><td>24.89</td><td>5.53</td><td>15.94</td><td>83.62</td><td>16.93</td><td>14.82</td><td>22.67</td><td></td><td>1.76</td><td>1.55</td><td>1.53</td><td>1.44</td><td>91.3</td></tr><tr><td colspan="10">Without Title</td><td colspan="7"></td></tr><tr><td>RAG-Loop5</td><td>PID</td><td>0.54</td><td>23.92</td><td>3.95</td><td>14.88</td><td>82.61</td><td>11.29</td><td>10.02</td><td>17.33</td><td></td><td>1.05</td><td>1.20</td><td>1.27</td><td>1.24</td><td>119.6</td></tr><tr><td>RAG-Loop5</td><td>PID-QID</td><td>0.60</td><td>24.02</td><td>4.06</td><td>14.91</td><td>82.64</td><td>11.75</td><td>10.66</td><td>17.80</td><td>1.49</td><td>1.68</td><td></td><td>1.78</td><td>1.73</td><td>120.7</td></tr><tr><td>RAG-Loop5</td><td>QID</td><td>0.61</td><td>24.18</td><td>3.98</td><td>14.94</td><td>82.60</td><td>11.12</td><td>10.33</td><td>17.62</td><td></td><td>1.41</td><td>1.60</td><td>1.73</td><td>1.66</td><td>123.8</td></tr><tr><td>RAG-Validate</td><td>PID</td><td>0.43</td><td>22.58</td><td>3.51</td><td>14.14</td><td>82.62</td><td>8.59</td><td>7.94</td><td>15.22</td><td></td><td>0.76</td><td>0.95</td><td>0.97</td><td>0.95</td><td>102.7</td></tr><tr><td>RAG-Validate</td><td>PID-QID</td><td>0.43</td><td>22.51</td><td>3.52</td><td>14.08</td><td>82.61</td><td>8.63</td><td>7.76</td><td>15.21</td><td></td><td>0.77</td><td>0.75</td><td>0.78</td><td>0.78</td><td>103.8</td></tr><tr><td>RAG-Validate</td><td>QID</td><td>0.44</td><td>22.52</td><td>3.52</td><td>14.10</td><td>82.60</td><td>8.65</td><td>7.77</td><td>15.08</td><td></td><td>0.69</td><td>0.86</td><td>0.90</td><td>0.89</td><td>104.9</td></tr><tr><td>Baseline</td><td>1</td><td>0.29</td><td>20.59</td><td>3.18</td><td>13.26</td><td>82.65</td><td>5.95</td><td>5.79</td><td>12.55</td><td></td><td>0.54</td><td>0.52</td><td>0.50</td><td>0.51</td><td>82.1</td></tr></table>

Table 1: Scores obtained under the various settings of our experiments. Bold indicates the best value for each evaluation metric. Among the shaded rows, cyan denotes the proposed method and gray denotes the Baseline.

Figure 1 shows the transition of scores against the number of iterative answer generations for the Baseline, RAG-Loop5, and RAG-Validate. In the Without Title setting, the score of the first answer is markedly lower than in the With Title setting, but the score was confirmed to increase continuously as RAG-Loop5 repeated answer generation. In the With Title setting, on the other hand, the score also increases, but the magnitude of the increase tends to diminish after a certain number of iterations. This suggests that traversing the nodes of the knowledge graph multiple times makes it possible to gradually acquire factual information useful for answer generation. In particular, the fact that score improvements were confirmed even when the initial answer contained few related entities and thus offered few cues for exploration supports the view that iterative retrieval functions as multi-hop reasoning over the knowledge graph and effectively explores entities that are useful for answer generation.

Furthermore, the average number of iterative answer generations of RAG-Validate is 3.6 in the With Title setting; despite requiring fewer iterations than the fixed-count RAG-Loop5, it maintains answer scores at a level comparable to RAG-Loop5. This result shows that the proposed method—which judges the correctness of the answer generated by the LVLM with a validator LLM and terminates the iteration once the answer is deemed valid—avoids the unnecessary exploration that arises when a fixed number of iterations is always performed, and thus reduces the retrieval cost of external knowledge without degrading generation quality.

Finally, whereas RAG-Validate maintains scores comparable to RAG-Loop5 in the With Title setting, it obtains lower scores than RAG-Loop5 in the Without Title setting. The likely cause of this degradation is that the validator lacks the knowledge that serves as the basis for the correctness judgment, so that it fails to properly reject incorrect answers and terminates the exploration prematurely. This indicates that verification in RAG-Validate is based not on the mere fluency of the generated text, but on the factual consistency between the input title and the content of the answer.

## 6 Conclusion

Aiming to improve the level of detail of image explanations, we proposed a retrieval-augmented generation method that makes effective use of a knowledge graph. The proposed method iteratively performs correctness judgment and regeneration of the generated answer and dynamically explores the knowledge graph, which makes it possible to efficiently acquire entities that are useful for explanation generation. In addition, we constructed ExpArt-KG, a knowledge graph consisting of entities related to artworks. Our experimental results confirm that, by dynamically controlling the number of iterations according to the validity of the answer, the proposed method reduces the total number of iterations while maintaining generation quality comparable to that of iterating a fixed number of times, thereby achieving both high-quality explanation generation and a reduced retrieval cost for external knowledge.

![](images/53d1cbde3a1fef2a31f7efc50d9e9da97f85b85f1aaf05d4508227441d5fcc47.jpg)  
Figure 1: Scores obtained at each iteration of RAG-Loop5, together with those of the Baseline and RAG-Validate, under the QID setting. The horizontal axis denotes the number of iterations and the vertical axis denotes the score. Solid lines show the transition of the RAG-Loop5 scores, and the shaded regions indicate their 95% confidence intervals. Crosses mark the Baseline, diamonds the final RAG-Loop5 scores, and stars RAG-Validate, plotted at its average number of iterations (3.6 with the title and 1.7 without it).

## Acknowledgments

This work was supported by JSPS KAKENHI Grant Numbers JP23K28148 and JP24K02993.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. 2024. Phi-4 technical report. arXiv preprint arXiv:2412.08905.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. Preprint, arXiv:2308.12966.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, et al. 2025a. Qwen3-VL technical report. Preprint, arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025b. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.

Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. Advances in Neural Information Processing Systems, 36:49250–49267.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Kazuki Hayashi, Hidetaka Kamigaito, Shinya Kouda, and Taro Watanabe. 2025. IterKey: Iterative keyword generation with LLMs for enhanced retrieval augmented generation. In Second Conference on Language Modeling.

Kazuki Hayashi, Yusuke Sakai, Hidetaka Kamigaito, Katsuhiko Hayashi, and Taro Watanabe. 2024. Towards artwork explanation in large-scale vision language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 705–729, Bangkok, Thailand. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems, 33:9459– 9474.

Mufei Li, Siqi Miao, and Pan Li. 2025a. Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation. In ICLR 2025 Workshop on Foundation Models in the Wild.

Zongxia Li, Xiyang Wu, Hongyang Du, Fuxiao Liu, Huy Nghiem, and Guangyao Shi. 2025b. A survey of state of the art large vision language models: Alignment, benchmark, evaluations and challenges. Preprint, arXiv:2501.02189.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in Neural Information Processing Systems, 36:34892– 34916.

Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.

Gerard Salton and Christopher Buckley. 1988. Termweighting approaches in automatic text retrieval. Information Processing & Management, 24(5):513– 523.

Denny Vrandeciˇ c and Markus Krötzsch. 2014.´ Wikidata: A free collaborative knowledge base. Communications ofthe ACM, 57:78–85.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. In International Conference on Learning Representations.

<table><tr><td>Usage</td><td>Prompt</td></tr><tr><td>LVLM w/o KG</td><td>System: You are an assistant helping with visual question answering. User: Question: {question} Answer the question based on the provided image. Do not include any additional text other than the answer itself.</td></tr><tr><td>LVLM w/KG</td><td>System: You are an assistant helping with visual question answering using external knowledge. User: Below are factual triples retrieved from a knowledge graph to help answer the question. Each triple is formatted as: [SUBJECT, PREDICATE, OBJECT] Retrieved Facts: {triplets} Question: {question} Answer the question by integrating the image, your internal knowledge, and any retrieved facts that are relevant to the question. Do not include any additional text other than the answer itself.</td></tr><tr><td>Validator LLM</td><td>System: You are an assistant that validates answers. User: Question: {question } Answer: {answer} Judge whether the provided answer is correct and addresses the question. Respond with only &quot;True&quot; or &quot;False&quot;. Do not provide any additional explanation or text.</td></tr></table>

Table 2: Prompts used in the proposed method.

<table><tr><td>Abbreviation</td><td>Hugging Face Model ID</td></tr><tr><td>Qwen3-VL</td><td>Qwen/Qwen3-VL-8B-Instruct</td></tr><tr><td>Qwen3</td><td>Qwen/Qwen3-4B-Instruct-2507</td></tr></table>

Table 3: Correspondence between the models used and their Hugging Face IDs.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

## A Appendix

## A.1 Prompts

The prompts used in the proposed method are shown in Table 2.

## A.2 Details of the Models Used

Table 3 shows the correspondence between the abbreviations used in this paper and the Hugging Face model IDs.

<table><tr><td>Breakdown</td><td># Questions</td></tr><tr><td>ExpArt test data</td><td>5,227</td></tr><tr><td>After filtering</td><td>4,823</td></tr><tr><td>After sampling (evaluation data)</td><td>1,199</td></tr></table>

Table 4: Dataset statistics.

## A.3 Dataset Statistics

Table 4 shows the number of questions at each processing stage of the dataset. Note that, by the specification of the data, each question comes in two settings depending on whether the title is included.

## A.4 Details of TF-IDF-Based Triple Selection

In our method, the set of triples connected to an entity e is regarded as a document $D _ { e }$ , and a constituent of a triple $T \in D _ { e }$ (its predicate $p$ or its adjacent node $q )$ is regarded as a term t. Let $\mathcal { E }$ be the set of all entities of the knowledge graph and $N = | { \mathcal { E } } |$ its size, let Terms(D) denote the set of terms occurring in the triples of a document $D ,$ let $f _ { t , e }$ be the number of times the term t occurs in $D _ { e }$ , and let $n _ { t } = | \{ e ^ { \prime } \in \mathcal { E } : t \in \operatorname { T e r m s } ( D _ { e ^ { \prime } } ) \} |$ be the number of entities whose document contains $t .$ The TF-IDF value $w ( t , D _ { e } )$ , which represents the importance of the term $t ,$ is then defined as follows, where logarithmic normalization and smoothing are applied for numerical stability.

$$
w ( t , D _ { e } ) = \log ( 1 + f _ { t , e } ) \cdot \log \left( \frac { N + 1 } { n _ { t } + 1 } \right)\tag{1}
$$

The score $S ( T )$ of a triple $T$ whose predicate is $p$ and whose adjacent node is $q$ is computed as follows under each setting. (1) PID: $S ( T ) \ =$ $w ( p , D _ { e } ) , ( 2 ) { \bf Q I D } \colon S ( T ) = w ( q , D _ { e } )$ , (3) PID-QID: $S ( T ) = w ( p , D _ { e } ) + w ( q , D _ { e } )$ . Based on this score, we select the top-ranked triples for each entity.

## A.5 Definitions of the Entity Evaluation Metrics

Let a generated explanation consisting of k sentences be $G \ = \ \{ g _ { 1 } , . . . , g _ { k } \}$ and a reference explanation consisting of m sentences be $R =$ $\{ r _ { 1 } , \ldots , r _ { m } \}$ . Let the sets of entities extracted from these texts be $E _ { G } = \mathrm { E n t i t y } ( G )$ and $E _ { R } =$ $\mathrm { E n t i t y } ( R )$ , and let |G| and $| R |$ denote their respective total numbers of tokens. We define the length penalty $\operatorname { L P } ( G , R )$ for redundant generated text as follows.

$$
\operatorname { L P } ( G , R ) = \exp \left( - \operatorname* { m a x } \left( 0 . 0 , { \frac { | G | } { | R | } } - 1 \right) \right)\tag{2}
$$

Each evaluation metric is computed as follows. In what follows, $\operatorname { C o v } ( X , Y )$ denotes the proportion of the elements of $Y$ that are covered by X.

(1) Entity Coverage (EC): the proportion of the entities contained in R that are covered by $G ,$ computed by the following equation. The longest common subsequence is used to compute the partialmatch coverage rate.

$$
\operatorname { E C } ( G , R ) = \operatorname { C o v } ( E _ { G } , E _ { R } )\tag{3}
$$

(2) Entity F1 (EF1): the harmonic mean of the precision Prec and the recall Rec based on the frequency of entity occurrences. Letting $\# ( e , \cdot )$ be the number of occurrences of entity e in the target text and $\mathrm { C o u n t _ { c l i p } } ( e , G , R ) \ =$ min $\mathfrak { i } ( \# ( e , G ) , \# ( e , R ) )$ ), they are defined by the following equations.

$$
\mathrm { E F } _ { 1 } = \frac { 2 \times P r e c \times R e c } { P r e c + R e c }\tag{4}
$$

$$
P r e c = { \frac { \sum _ { e \in E _ { G } } { \mathrm { C o u n t } } _ { \mathrm { c l i p } } ( e , G , R ) } { \sum _ { e \in E _ { G } } \# ( e , G ) } }\tag{5}
$$

$$
R e c = { \frac { \sum _ { e \in E _ { R } } \mathrm { C o u n t } _ { \mathrm { c l i p } } ( e , G , R ) } { \sum _ { e \in E _ { R } } \# ( e , R ) } }\tag{6}
$$

(3) Entity Cooccurrence (ECooc): letting $\operatorname { C o } _ { n } ( \cdot )$ be a function that returns the pairs of entities cooccurring within a context window consisting of a sentence and its surrounding n sentences, it is computed by the following equation. We used the sentence splitter of NLTK to split the text into sentences.

$$
\operatorname { E C o o c } ( G , R ) = \operatorname { L P } ( G , R ) \times \operatorname { C o v } ( \operatorname { C o } _ { n } ( G ) , \operatorname { C o } _ { n } ( R ) )\tag{7}
$$