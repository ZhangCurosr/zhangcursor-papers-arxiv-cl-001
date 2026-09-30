# Bridging Semantic Gaps in RAG through Generated Context Knowledge Fusion

Xinkai Du<sup>1,2</sup>, Chao Lv<sup>1</sup>, Yalin Sun<sup>3</sup>, Quanjie Han<sup>1</sup>, Lei Yao<sup>1</sup>, and Maosong Sun<sup>2</sup> <sup>⋆</sup>

<sup>1</sup> Beijing Wanlian Zhilian Technology Corporation Limited, Beijing, China {duxinkai, lvchao, hanquanjie, yaolei}@wanlianyida.com

<sup>2</sup> Department of Computer Science and Technology, Tsinghua University, Beijing, China

sms@tsinghua.edu.cn

3 Sunshine Digital Intelligence Tech Co., Ltd., Beijing, China sunyalin-ghq@sinosig.com

Abstract. Retrieval-Augmented Generation has established itself as a fundamental framework in natural language processing, seamlessly integrating information retrieval with the generative capabilities of large language models. However, this process is fundamentally constrained by a critical challenge: semantic space mismatch between queries and retrieved contexts. We propose Knowledge-Aware Semantic Bridging (KASB), a novel framework that improves passage selection quality through semantic space alignment between queries and retrieved documents through intelligent knowledge fusion. Our approach leverages the complementary strengths of generative and retrieval-based knowledge through a multistage process that enhances both relevance and accuracy. We evaluate KASB on three popular open-domain Question Answering datasets to demonstrate the efectiveness of our approach.

Keywords: Large Language Model · Direct Preference Optimization · Retrieval Augmented Generation · Question Answering .

## 1 Introduction

Retrieval-Augmented Generation (RAG) [11] has become a cornerstone framework in natural language processing, combining information retrieval with the generative power of large language models (LLMs). The standard RAG pipeline consists of three stages: retrieval, re-ranking, and generation. In the retrieval phase, relevant documents are identified based on semantic similarity between the query and a knowledge corpus. However, this process is hindered by a key limitation: semantic space mismatch between queries and retrieved contexts. This mismatch arises from diferences in linguistic expression, abstraction levels, and terminology, even for semantically aligned content [6].

LLMs having acquired vast world knowledge through self-supervised pretraining [2], can generate contextually coherent responses during question answering. Yet, this generative capability is prone to hallucination: producing plausible but factually inaccurate information [7]. In contrast, retrieved knowledge from curated corpora tends to be more factually reliable but often lacks fine-grained relevance to the query. This trade-of between the relevance of generated content and the accuracy of retrieved content remains a fundamental challenge in current RAG systems.

We propose Knowledge-Aware Semantic Bridging (KASB), a novel framework that dynamically aligns the semantic spaces between queries and retrieved documents via adaptive knowledge fusion. Our method exploits the complementary advantages of generative knowledge and retrieval-based knowledge in a multi-stage paradigm, to substantially boost semantic relevance and factual accuracy simultaneously.

Our contributions can be summarized as follows:

We propose KASB, a reinforcement learning-enhanced methodology for dynamically aligning query and document representations.

– We introduce an unsupervised LLM generated context based evidence selector within the RAG framework, which is an efective approach that leverages explicit hypothetical document as signals to align the preferences of diferent components.

– Extensive experiments are conducted on three open domain QA datasets to demonstrate the efectiveness of KASB.

![](images/3e3f1287d9727f7f0dd2a41d1717a7330f1fffbab295ced4c2e675676c6b885b.jpg)  
Fig. 1. Overview of the KASB Framework. A finetuned generator is derived via the DPO algorithm on the training set to produce a more efective generative context generator, which then serves in the test phase. During inference, the finetuned generator first generates query-aligned generative contexts G; meanwhile, the evidence (representing the complete retrieved knowledge base) is processed through the generated context-based selector to identify relevant retrieved contexts C, which are then fused with the generated contexts [C; G] for final answer generation.

## 2 Related Work

In knowledge-intensive tasks, the retrieve-then-read paradigm has been dominant [11, 9, 6]. Early approaches relied on sparse retrievers such as BM25, while recent advances leverage dense contextualized representations, exemplified DPR [9], which significantly outperform traditional methods. Various strategies have since been explored to integrate the retrieved passages with large language models to improve the generation of answers [13].

LLMs have also been employed to generate auxiliary knowledge for downstream reasoning [12]. Building on this capability, recent work has explored using LLM-generated knowledge to enhance retrieval, particularly when relevant information is sparse or insuficiently covered by a fixed corpus. This has led to the emergence of the generate-then-read framework, which generates query-specific contexts instead of retrieving them from a fixed corpus [19]. Methods like HyDE [4] further refine retrieval by generating hypothetical document embeddings for dense indexing.

Building on retrieval-augmented and generation-augmented paradigms, recent work has explored hybrid approaches to mitigate their respective limitations: retrieved passages may be irrelevant, while generated ones may be plausible but inaccurate. To address this, [20] proposes merging passages based on compatibility scoring, and [3] introduces an unsupervised bi-reranking framework for fusing generated and retrieved knowledge. Our method integrates the complementary strengths of both paradigms through a multistage process that improves relevance and accuracy through a lightweight embedding-space selection mechanism.

KASB difers from HyDE [4] in that HyDE primarily generates a hypothetical document to construct a retrieval representation, whereas KASB starts from an arbitrary retriever’s candidate set and uses multiple generated contexts to select real evidence after retrieval. Compared with COMBO [20] and BRMGR [3], KASB uses generated contexts not only for fusion but also as a query-conditioned signal for passage scoring and adaptive evidence selection.

## 3 KASB: Knowledge-Aware Semantic Bridging Guided by Generated Context

## 3.1 Generated Context Generation

RAG inherently faces a semantic space mismatch between queries and retrieved contexts. This gap arises from discrepancies in linguistic expression, abstraction levels, and terminology usage, even when the core semantics are aligned. LLMs having internalized extensive world knowledge through self-supervised pretraining, can generate hypothetical relevant contexts tailored to the query’s semantic style. These generated contexts act as a semantic bridge to connect the query with potentially misaligned retrieved documents, addressing the fundamental limitation of traditional RAG’s retrieval phase.

For each query $q ,$ an LLM is prompted to generate a context ${ \mathcal { G } } _ { : }$ , formulated as

$$
\mathcal { G } = L L M ( q , \mathrm { p r o m p t } )\tag{1}
$$

## 3.2 Preference-Tuned Generated Context Generator

While LLMs can generate contexts well-aligned with user queries, their outputs frequently sufer from hallucinations, plausible but factually incorrect content that significantly undermine the reliability of downstream retrieval and generation processes [7, 14]. Traditional approaches to mitigating this issue often employ Reinforcement Learning from Human Feedback (RLHF) [15], which requires training and maintaining a separate reward model, introducing additional complexity and computational overhead.

Direct Preference Optimization (DPO) [16] provides an eficient alternative by directly optimizing the policy from preference pairs without an explicit reward model. To enhance the contextual generation capability of our language model, we fine-tune the generator on the training split of each dataset using preference pairs constructed through an automated process.

Specifically, we leverage the Exact Match (EM) metric [17] to evaluate generated contexts against ground-truth answers, designating contexts containing the correct answer as positive samples and those without as negative samples. This creates a training signal that explicitly encourages factually accurate context generation while maintaining semantic alignment with the query. The generator is subsequently optimized through DPO, resulting in more reliable and factually grounded contexts for semantic bridging without requiring human-annotated preference data.

$$
L _ { \mathrm { D P O } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( q , y _ { w } , y _ { l } ) \sim \mathcal { D } } [ \log \sigma ( \beta f ( q , y _ { w } , y _ { l } ) ) ]\tag{2}
$$

$$
\begin{array} { r } { f ( q , y _ { w } , y _ { l } ) = \log \frac { \pi _ { \theta } ( y _ { w } | q ) } { \pi _ { \mathrm { r e f } } ( y _ { w } | q ) } - \log \frac { \pi _ { \theta } ( y _ { l } | q ) } { \pi _ { \mathrm { r e f } } ( y _ { l } | q ) } } \end{array}\tag{3}
$$

where $\pi _ { \theta }$ denotes the model to be optimized, $\pi _ { \mathrm { r e f } }$ represents the original model to generate question contexts, and θ stands for the model optimization parameters.

This fine-tuning process aligns the LLM’s generated contexts with factual accuracy, ensuring they serve as reliable signals for subsequent retrieval.

## 3.3 Unsupervised Generated Context Based Retrieved Context Selection

Retrieved contexts from curated corpora are factually reliable but often contain irrelevant noise. Given the fine-tuned knowledge-generating model’s eficacy in producing query-relevant context, we posit that strong alignment between retrieved documents and this generated context serves as a reliable indicator of answer adequacy in top-ranked results.

Formally, we implement the document encoder $e n c _ { d }$ as a dual-encoder architecture, where for each query $q _ { j } ,$ , the similarity between retrieved contexts and the fine-tuned model’s generated knowledge is computed by:

$$
S ( g _ { i } , e _ { j } ) = < e n c _ { d } ( g _ { i } ) , e n c _ { d } ( e _ { j } ) >\tag{4}
$$

All the related context for query $q _ { j }$ is retrieved as follows:

$$
E _ { v } = \left\{ \arg \operatorname* { m a x } _ { e _ { j } \in E } S ( g _ { i } , e _ { j } ) \mid g _ { i } \in \mathcal { G } \right\}\tag{5}
$$

where $\mathcal { G }$ denotes all the generated context information produced by finetuning the large model for the query $q _ { j }$ , and $E _ { v }$ represents the retrieved context information corresponding to the query $q _ { j }$ .

## 3.4 Generated Context and Retrieved Context Combination

To determine the optimal number of retrieved documents, we introduce an adaptive top-k selection mechanism based on similarity score distribution analysis. For each query, we first compute the centroid of generated context embeddings:

$$
{ \overline { { g } } } = { \frac { 1 } { | { \mathcal { G } } | } } \sum _ { g _ { i } \in { \mathcal { G } } } { \mathrm { S B E R T } } ( g _ { i } )\tag{6}
$$

where $\mathcal { G }$ denotes the set of contexts generated by the fine-tuned model, and SBERT refers to the pre-trained sentence embedding model from the sentencetransformers library.

Subsequently, we calculate cosine similarities between the centroid $\overline { { g } }$ and all retrieved contexts, yielding an ordered sequence $\left\{ s _ { 1 } , s _ { 2 } , \ldots , s _ { n } \right\}$ sorted in descending order. To identify the optimal cutof point, we analyze the first-order diferences of these similarity scores:

$$
\varDelta _ { k } = s _ { k } - s _ { k + 1 }\tag{7}
$$

We use z-score to standardize all the first-order diference values. We select the first index $k ^ { * }$ where $\varDelta _ { k }$ significantly deviates from the mean z-score. If no such decline is found, we will use the second-order diference method to find the point with the maximum curvature to obtain the corresponding index $k ^ { * }$

Finally, the selected $\mathrm { t o p } { - } k ^ { * }$ retrieved contexts C and the generated contexts produced by the fine-tuned large model are taken as a whole as the context information of the query. The resulting context is then fed into the fine-tuned large language model to generate the answer.

## 4 Experimental Setup

To demonstrate the efectiveness of our method, we evaluate it on three popular open-domain Question Answering datasets.

Datasets: Following prior work [3], we conduct experiments on three widelyused open-domain Question Answering (QA) datasets: TriviaQA [8], Natural Questions (NQ) [10], and WebQuestions (WebQ) [1].

Evaluation Metric: For question answering, we employ the exact match (EM) score [17]. For the retrieval task, we adopt the standard top-K retrieval EM metric, which measures the proportion of questions where at least one passage among the top-K retrieved ones contains a text span that exactly matches the human-annotated answer.

Baselines: Comparable to [20], we employ single and two knowledge sources in comparison experiments to demonstrate our method’s efectiveness.

Retri-Only: Utilizes only the original retrieved knowledge directly as the input for the downstream reader model.

– Gen-Only: Adopts only the original LLM-generated knowledge as the sole input for the reader model.

– HyDE: A classic retrieval-augmented baseline that generates a hypothetical document for each query, encodes it into a dense representation, and uses this representation to retrieve relevant real documents [4].

– COMBO: Matches LLM-generated passages with their retrieved counterparts to form compatible knowledge pairs and fuses the paired knowledge for open question answering [20].

– BRMGR: An unsupervised Bi-Reranking for Merging Generated and Retrieved knowledge method, which optimizes the fusion of two knowledge sources via bi-directional reranking of retrieved and generated passages [3].

DPO Training Details: We implement DPO optimization using the LLaMA Factory framework [21] with a learning rate of 5 × 10<sup>−6</sup>, batch size 4, β = 0.1, and 3 training epochs. The reference model is the original Meta-Llama-3.1-8B-Instruct. Positive and negative contexts are constructed automatically from answer containment: a generated context containing the ground-truth answer is preferred over one that does not. We therefore refer to these as automatically constructed answer-containment preferences, rather than human preferences.

We use Meta-Llama-3.1-8B-Instruct with the prompt “Question: {question} Provide efective generated contexts for retrieving relevant information.” Five relevant contexts are generated per query and encoded with ms-marco-MiniLM-L6-v2.

For each query, if the generated contexts include at least one positive and one negative instance, the query and the corresponding positive/negative contexts are formatted into ShareGPT style. This construction is scalable but can produce false positives for polysemous strings and false negatives for aliases or paraphrases. The data statistics for DPO training are summarized in Table 1:

Table 1. Datasets statistics.
<table><tr><td>Datasets</td><td>Train</td><td>Dev</td></tr><tr><td>TriviaQA</td><td>75675</td><td>8750</td></tr><tr><td>NQ</td><td>91334</td><td>10039</td></tr><tr><td>WebQ</td><td>3906</td><td>-</td></tr></table>

## 5 Experimental Results

## 5.1 Main Results

Finetuned LLM Generation Ability To enhance the LLM’s capability in generating accurate and relevant contextual knowledge, we fine-tuned the generator using the DPO algorithm with carefully constructed positive and negative sample pairs. As demonstrated in Figure 2, the finetuned LLM generator achieved substantial improvements in average retrieval exact match scores, with absolute gains of 14.85%, 18.08%, and 16.75% on the TriviaQA, NQ, and WebQ test sets, respectively.

Question Answering In order to assess the performance of our method in open question answering, we employ the finetuned LLM generator as the reader model. The results of this evaluation are presented in Table 2.

Table 2 clearly illustrates that our proposed method KASB achieves the strongest overall performance across all three datasets. Specifically, methods leveraging two knowledge sources (retrieved and generated) consistently outperform those relying on a single knowledge source alone. For example, even the weakest two-source method BRMGR surpasses the best single-source method on NQ and WebQ, highlighting the advantage of fusing complementary retrieved and generated knowledge for open question answering.

Among the two-source baselines, KASB delivers the most substantial gains. This underscores the efectiveness of our knowledge alignment and semantic boosting strategies in optimizing the fusion of dual knowledge sources. Additionally, when examining single-source methods, Gen-Only outperforms Retri-Only on TriviaQA and NQ, while Retri-Only exhibits stronger performance on WebQ, revealing the complementary strengths of generated and retrieved knowledge across datasets with diferent characteristics.

## 5.2 Ablation Analysis

When both retrieved and generated knowledge are available, we use them as contextual input to prompt the fine-tuned LLM, which then directly generates the answer. The model’s accuracy is evaluated using exact match (EM) score against the ground truth.

Table 2. Exact match scores on test dataset.
<table><tr><td>Methods</td><td>TriviaQA</td><td>NQ</td><td>WebQ</td></tr><tr><td>Single Knowledge</td><td></td><td></td><td></td></tr><tr><td>Retri-Only</td><td>62.2</td><td>48.6</td><td>48.3</td></tr><tr><td>Gen-Only</td><td>67.5</td><td>50.3</td><td>41.6</td></tr><tr><td>Two Knowledge</td><td></td><td></td><td></td></tr><tr><td>HyDE</td><td>72.3</td><td>53.4</td><td>54.6</td></tr><tr><td>COMBO</td><td>74.6</td><td>54.2</td><td>53.0</td></tr><tr><td>BRMGR</td><td>68.6</td><td>52.2</td><td>53.4</td></tr><tr><td>KASB</td><td>75.4</td><td>59.6</td><td>61.2</td></tr></table>

Retrieval Performance Boost via Semantic Bridging To assess the improvements in retrieval performance brought about by semantic bridging, we conduct dedicated experiments where passages are retrieved using both unsupervised and supervised retrievers. The unsupervised retrievers selected for our experiments include BM25, MSS [18], and Contriever [5], whereas the supervised retrievers consist of DPR [9] and MSS-DPR [18].

Retrieval accuracy at Top-20 and Top-100 on the test sets of the three datasets, evaluated on the basis of the top-1000 retrieved passages, is reported in Table 3. As illustrated in the table, our proposed KASB method yields consistent improvements in retrieval performance across all retrievers, with notably substantial gains achieved for unsupervised models including BM25, MSS [18], and Contriever [5].

Table 3. Top-20, 100 retrieval accuracy on the test set of datasets for the top-1000 retrieved passages.
<table><tr><td>Retriever</td><td>TriviaQA Top-20 Top-100|Top-20 Top-100|Top-20 Top-100</td><td>NQ</td><td></td><td>WebQ</td></tr><tr><td colspan="5">Unsupervised Retrievers</td></tr><tr><td>MSS MSS + KASB</td><td>67.2 79.1 81.3 85.0</td><td>60.0 77.3</td><td>75.6 81.5</td><td>49.2 68.4 62.8 75.8</td></tr><tr><td>BM25</td><td>76.4 83.2</td><td>62.9</td><td>78.3</td><td>62.4 75.5</td></tr><tr><td>BM25 + KASB Contriever</td><td>84.3 87.2 73.9</td><td>75.6 67.9</td><td>84.3</td><td>65.6 77.4 80.1</td></tr><tr><td>Contriever + KASB</td><td>82.9 85.7 87.4</td><td>81.2</td><td>80.6 86.5</td><td>65.7 70.1 81.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5">Supervised Retrievers</td></tr><tr><td>DPR</td><td></td><td></td><td></td><td></td></tr><tr><td>DPR + KASB</td><td>79.8 85.1</td><td>79.2</td><td>85.7</td><td>74.6 81.6</td></tr><tr><td></td><td>84.8 87.4</td><td>83.5</td><td>87.4</td><td>78.2 84.2</td></tr><tr><td>MSS-DPR</td><td>81.9 86.6</td><td>81.4</td><td>88.1</td><td>76.9 84.6</td></tr><tr><td>MSS-DPR + KASB</td><td>87.5 88.3</td><td>84.3</td><td>89.7</td><td>81.4 85.3</td></tr></table>

![](images/55f814f2c52ccfbf921e7b3572997083d468b264310411ec80609f4a5ef57087.jpg)  
Fig. 2. Average retrieval exact match score obtained by finetuned vs original LLM generator on generated contexts.

Efect of the finetuned Generated Context Generator Comparing KASB with the w/o DPO variant in Table 4, we observe that the fine-tuned LLM achieves significantly higher performance across all three dataset compared to the baseline model without direct preference optimization. This indicates that the fine-tuning process, particularly through direct preference optimization enhances the model’s ability to generate high-quality responses. The substantial improvement in exact match scores suggests that the fine-tuned LLM aligns better with the automatically constructed answer-containment preferences and produces more accurate answers when integrated with retrieved and generated knowledge.

Importance of the Retrieved and Generated Knowledge Fusion Comparing KASB with the ablated variants in Table 4, both knowledge sources contribute to performance. Removing generated knowledge (GK) causes a larger drop than removing retrieved knowledge (RK), while removing DPO has a smaller efect. This supports the interpretation that the main benefit comes from the interaction of generated-context guidance, evidence selection, and two-source fusion rather than from DPO alone.

Table 4. Question answering performance (Exact Match). GK and RK denote generated knowledge and retrieved knowledge, respectively. The combined knowledge results are computed by union of single knowledge sources.
<table><tr><td>Model</td><td>TriviaQA</td><td>NQ</td><td>WebQ</td></tr><tr><td>KASB</td><td>75.4</td><td>59.6</td><td>61.2</td></tr><tr><td>w/o GK</td><td>69.9</td><td>54.9</td><td>56.3</td></tr><tr><td>w/o RK</td><td>73.2</td><td>56.8</td><td>59.5</td></tr><tr><td>w/o DPO</td><td>74.6</td><td>57.7</td><td>59.8</td></tr></table>

## 5.3 Parameter Sensitivity Analysis

Figure 3 examines the sensitivity of KASB to two critical hyperparameters: the DPO coeficient β and the initial number of retrieved candidates. Performance peaks at $\beta = 0 . 1$ suggesting that moderate preference strength optimally balances alignment with retained knowledge diversity. For retrieved candidates, performance stabilizes after 800 documents, indicating that KASB can work effectively with standard retrieval pool sizes without requiring exhaustive retrieval.

The adaptive top-k mechanism yields average k<sup>⋆</sup> values of 24.3 (TriviaQA), 18.7 (NQ), and 15.2 (WebQ). To directly examine whether adaptive selection is preferable to a globally fixed cutof, we further compare fixed $k \in \{ 5 , 1 0 , 2 0 , 5 0 \}$ in Table 5. The fixed-k results show a consistent non-monotonic trend: increasing k initially improves performance by adding useful evidence, while overly large k introduces more less-relevant passages and degrades performance. In contrast, the adaptive strategy achieves the best result on all three datasets. This indicates that a single global k is suboptimal because the appropriate amount of evidence varies with the query-specific score distribution.

Table 5. Fixed-k versus adaptive evidence selection. EM is reported on the three QA datasets; Avg. is the mean across datasets.
<table><tr><td>Selection</td><td>TriviaQA</td><td>NQ</td><td>WebQ</td><td>Avg.</td></tr><tr><td>Fixed-k, k = 5</td><td>73.2</td><td>58.2</td><td>59.2</td><td>63.5</td></tr><tr><td>Fixed-k, k = 10</td><td>73.8</td><td>58.5</td><td>59.8</td><td>64.0</td></tr><tr><td>Fixed-k, k = 20</td><td>74.2</td><td>58.9</td><td>59.7</td><td>64.3</td></tr><tr><td>Fixed-k, k = 50</td><td>73.5</td><td>58.1</td><td>59.4</td><td>63.7</td></tr><tr><td>Adaptive k*</td><td>75.4</td><td>59.6</td><td>61.2</td><td>65.4</td></tr></table>

## 5.4 Discussion

Despite KASB’s efectiveness, several limitations warrant discussion. First, the quality of semantic bridging is constrained by the generative model’s knowledge coverage and accuracy; generated contexts may be outdated, redundant, hallucinated, or conflicting with retrieved evidence. Second, our DPO preferences are automatically constructed from answer containment rather than human judgments, so they may reward surface overlap and do not establish semantic correctness; answer-masked or answer-absent evaluation would be a useful diagnostic. Third, the fixed-k comparison shows that performance first improves as more evidence is included and then declines when overly large cutofs introduce lessrelevant passages, supporting the use of query-specific adaptive selection. Fourth, we do not explicitly perform deduplication or contradiction resolution between generated and retrieved knowledge. Finally, the current evaluation is limited to single-turn open-domain QA, and we do not claim cross-dataset generalization of the DPO-tuned generator without additional transfer experiments.

![](images/e985ff11d2aa935ad6296600bce2991197509c68812271504af109d790845b3b.jpg)

![](images/0ebc08447f62527e29f2ef4829e5cbd664c5c0eed8cbb5e2616a87fa150a63df.jpg)  
Fig. 3. Parameter sensitivity analysis showing the efect of DPO coeficient β and initial candidate pool size on EM performance across datasets.

## 6 Conclusion

In this paper, we propose Knowledge-Aware Semantic Bridging (KASB), a lightweight post-retrieval framework that uses generated contexts to guide evidence selection and fuse selected retrieved passages with generated knowledge. Experiments on three single-turn open-domain QA datasets show consistent improvements in retrieval accuracy and answer correctness across diverse retrievers. The results support generated-context-guided evidence selection as a practical system-level integration for improving RAG.

## Acknowledgements

This work is supported by Beijing Municipal Science and Technology Plan Project(Z241100001324025).

## References

1. Berant, J., Chou, A., Frostig, R., Liang, P.: Semantic parsing on Freebase from question-answer pairs. In: Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing. pp. 1533–1544 (2013)

2. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J.D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners. Advances in Neural Information Processing Systems 33, 1877–1901 (2020)

3. Du, X., Han, Q., Lv, C., Liu, Y., Sun, Y., Shu, H., Shan, H., Sun, M.: Improving generated and retrieved knowledge combination through zero-shot generation. In: ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 1–5. IEEE (2025)

4. Gao, L., Ma, X., Lin, J., Callan, J.: Precise zero-shot dense retrieval without relevance labels. In: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1762–1777 (2023)

5. Izacard, G., Caron, M., Hosseini, L., Riedel, S., Bojanowski, P., Joulin, A., Grave, E.: Unsupervised dense information retrieval with contrastive learning. arXiv preprint arXiv:2112.09118 (2021)

6. Izacard, G., Grave, E.: Leveraging passage retrieval with generative models for open domain question answering. In: Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume. pp. 874–880 (2021)

7. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y.J., Madotto, A., Fung, P.: Survey of hallucination in natural language generation. ACM Computing Surveys 55(12), 1–38 (2023)

8. Joshi, M., Choi, E., Weld, D.S., Zettlemoyer, L.: TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In: Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1601–1611 (2017)

9. Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.t.: Dense passage retrieval for open-domain question answering. In: Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). pp. 6769–6781 (2020)

10. Kwiatkowski, T., Palomaki, J., Redfield, O., Collins, M., Parikh, A., Alberti, C., Epstein, D., Polosukhin, I., Devlin, J., Lee, K., et al.: Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics 7, 453–466 (2019)

11. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive NLP tasks. Advances in Neural Information Processing Systems 33, 9459–9474 (2020)

12. Liu, J., Liu, A., Lu, X., Welleck, S., West, P., Le Bras, R., Choi, Y., Hajishirzi, H.: Generated knowledge prompting for commonsense reasoning. In: Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 3154–3169 (2022)

13. Liu, Y., Yavuz, S., Meng, R., Moorthy, M., Joty, S., Xiong, C., Zhou, Y.: Exploring the integration strategies of retriever and large language models. arXiv preprint arXiv:2308.12574 (2023)

14. Maynez, J., Narayan, S., Bohnet, B., McDonald, R.: On faithfulness and factuality in abstractive summarization. In: Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. pp. 1906–1919 (2020)

15. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. In: Advances in Neural Information Processing Systems. vol. 35, pp. 27730–27744 (2022)

16. Rafailov, R., Sharma, A., Mitchell, E., Manning, C.D., Ermon, S., Finn, C.: Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36, 53728–53741 (2023)

17. Rajpurkar, P., Zhang, J., Lopyrev, K., Liang, P.: SQuAD: 100,000+ questions for machine comprehension of text. In: Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing. pp. 2383–2392 (2016)

18. Sachan, D., Patwary, M., Shoeybi, M., Kant, N., Ping, W., Hamilton, W.L., Catanzaro, B.: End-to-end training of neural retrievers for open-domain question answering. In: Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 6648–6662 (2021)

19. Yu, W., Iter, D., Wang, S., Xu, Y., Ju, M., Sanyal, S., Zhu, C., Zeng, M., Jiang, M.: Generate rather than retrieve: Large language models are strong context generators. In: The Eleventh International Conference on Learning Representations (2023)

20. Zhang, Y., Khalifa, M., Logeswaran, L., Lee, M., Lee, H., Wang, L.: Merging generated and retrieved knowledge for open-domain QA. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 4710– 4728 (2023)

21. Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z.: LlamaFactory: Unified eficient fine-tuning of 100+ language models. In: Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations). pp. 400–410 (2024)