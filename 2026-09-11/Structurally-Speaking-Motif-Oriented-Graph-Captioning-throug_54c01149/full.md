# Structurally Speaking: Motif-Oriented Graph Captioning through Bidirectional Graph-Text Translation

Hsiao-Ying Lu and Dongyu Liu and Kwan-Liu Ma {hyllu,dyuliu,klma}@ucdavis.edu Department of Computer Science, University of California, Davis

## Abstract

Graph captions should help readers understand graph structure, rather than simply translate adjacency matrices into long textual edge lists. A useful graph caption abstracts connectivity into recognizable motifs, such as hubs, paths, cycles, cliques, and bridges, because these motifs provide compact structural units that are easier to read, compare, and recover. In this paper, we study motif-oriented graph captioning as a bidirectional graph-text translation task, where captions must both preserve enough topology for graph recovery and express the graph through concise motif-level descriptions. We show that direct prompting of GPT-5.1 often produces graph-recoverable captions by enumerating node-to-node connections, but these captions are verbose and can contain inconsistent motif interpretations. To address this gap, we introduce Structurally Speaking, a lightweight structured prompting protocol that guides translation between explicit connectivity and motif-level abstraction. Experiments on a synthetic motif-based dataset show that structured prompting produces shorter and more motif-consistent captions while maintaining comparable graph recovery. These results suggest that explicit topology-to-motif reasoning guidance can make LLM-generated graph captions more interpretable without model finetuning.

## 1 Introduction

Graphs encode relationships among entities, but raw representations such as adjacency matrices are difficult to read and compare directly. Graph captions can make graph structure more accessible by translating connectivity into natural language. However, a useful caption should do more than list edges: it should abstract topology into recognizable motifs, such as hubs, paths, cycles, cliques, bridges, and tails. These motifs provide compact structural units that help readers understand how local connections compose into larger graph patterns.

This distinction matters because graph recoverability alone is not sufficient for evaluating graphcaption quality. A caption may encode all edges through enumeration while offering little interpretable structural abstraction. Such captions can be verbose, difficult to read, or inconsistent in their motif-level structural inference. We therefore argue that interpretable graph captioning should be evaluated along two dimensions: structural recoverability and motif-level compactness.

Motivated by this gap, we formulate bidirectional graph-caption translation as a controlled diagnostic task. Given a graph adjacency matrix, the model generates a caption; given a caption, the model recovers the graph. This formulation allows us to probe whether LLMs can move between raw graph topology and motif-oriented natural language. Under direct prompting, GPT-5.1 often produces graph-recoverable captions, but these captions rely heavily on explicit node-to-node connection descriptions and sometimes contain inaccurate or self-contradictory motif interpretations. This suggests that direct prompting often falls back on edge enumeration rather than producing compact motif-level structural inference, even when graph recovery is successful.

To address this issue, we introduce Structurally Speaking, a lightweight structured prompting protocol for topology-to-motif abstraction. For graphto-caption translation, the protocol guides the model to convert adjacency information into local neighborhoods, analyze motifs, and generate a compact caption. For caption-to-graph translation, it guides the model to parse motif descriptions, assign nodes, construct edges, and recover the adjacency matrix. This is not merely caption style transfer, because the model must infer motif structure from raw topology while preserving enough information for graph recovery.

We evaluate this formulation using a synthetic motif-based dataset and two cycle-consistency evaluations. Graph-Caption-Graph measures whether generated captions preserve enough structural information for graph recovery, while Caption-Graph-Caption measures whether motif-oriented descriptions remain accurate and compact after translation through graph structure. Experiments show that Structurally Speaking reduces verbosity and inconsistent motif inferences while maintaining comparable graph recovery. In summary, we contribute: (1) a bidirectional graph-captioning formulation that separates structural recoverability from motiflevel abstraction; (2) a cycle-consistency evaluation that reveals recoverable captions can still be verbose and motif-inconsistent; and (3) a lightweight structured prompting protocol that improves caption compactness and motif consistency without fine-tuning.

## 2 Related Work

Recent work on LLMs and graphs spans three related directions. First, graph-to-text generation produces natural language from graph-structured inputs such as meaning representations, knowledge graphs, and scientific graphs (Ribeiro et al., 2021), with recent work evaluating LLMs for graph-totext generation through planning and groundingoriented tasks (He et al., 2025). While closely related, these works focus mainly on semantic graph structures and text fluency or factuality. We instead study motif-oriented captioning of raw topology, where captions should expose compact structural abstractions while remaining recoverable.

A second line of work instead optimizes how topology is serialized or represented for LLMs, including adjacency linearization (Fatemi et al., 2023), learned graph encodings for frozen LLMs (Perozzi et al., 2024), and broader graph transformation strategies (Yu et al., 2026). A third line of work shifts from graph representation and generation to graph reasoning, learning, and query execution, including LLM-graph learning frameworks (Jin et al., 2024; Shang and Huang, 2025; You et al., 2025), graph-specialized prompting and structured interfaces (Tang et al., 2024; Wang et al., 2024; Jiang et al., 2023; Li et al., 2025), empirical analyses of graph reasoning generalization (Guo et al., 2023; Zhang et al., 2024), and natural-language-to-graph-query systems (Liang et al., 2024; Hains et al., 2019).

## 3 Problem Formulation

Given a graph G = (V, E), graph-to-caption translation aims to generate a natural-language caption C that summarizes the topology of G. Given a caption C, caption-to-graph translation aims to reconstruct a graph G<sup>ˆ</sup>. A high-quality caption should satisfy two criteria: graph recoverability and motif abstraction.

Specifically, we define a motif-oriented caption as one that satisfies the following requirements:

• identifies dominant motifs, such as stars, paths, cycles, cliques, and wheels;

• describes node roles, such as hubs, rim nodes, bridge nodes, and leaves;

• describes deviations or perturbations, such as missing edges or added chords;

• avoids exhaustive edge enumeration unless specific connections are needed for graph recovery.

To support this task, we construct a synthetic motif-based graph dataset containing 220 undirected, unweighted graphs, each controlled to be within 30 nodes. Graphs are initialized from common motifs, including stars, cycles, paths, cliques, and wheels, and are then perturbed with random edge additions and removals to introduce structural variation. This process produces graphs with diverse topologies and varying levels of motif complexity. Illustrations are provided in Appendix A.

From these 220 graphs, we select 40 representative examples for caption annotation. These graphs are chosen to cover diverse motif families and perturbation levels. Each selected graph is paired with a human-verified motif-oriented caption, as shown in Figure 1. The captions are written in free-form natural language rather than templates to preserve linguistic diversity and allow natural rephrasings. This set of 40 graph-caption pairs serves as the unified test set for our experiments and analyses. It is intentionally curated for controlled diagnosis rather than benchmark-scale evaluation.

For LLM input, each graph is represented as an adjacency matrix serialized into text. This representation preserves explicit topology while remaining compatible with autoregressive language models.

## 4 Direct Prompting Analysis

We first examine GPT-5.1 under direct prompting to probe how a strong LLM approaches bidirectional graph-caption translation without taskspecific guidance or fine-tuning.

![](images/bb92c227279996a32d30941df0cf5fd0f702c9e9a43ac7efa590c9179b3d4a1d.jpg)  
Figure 1: A wheel-motif graph with one added edge as structural variation. Its adjacency matrix is paired with a human-verified motif-oriented caption that compactly describes the central hub, rim structure, and additional rim connection. In contrast, the generated caption is graph-recoverable but verbose and sometimes self-contradictory, relying heavily on explicit node-to-node connection descriptions rather than compact motif-level abstraction.

Table 1: Representative direct-prompting failure modes.
<table><tr><td>Mode</td><td>Selected evidence in Figure 1</td></tr><tr><td>(1) (2)</td><td>“3-2-11-12-10-9-8-7-6-5-4-3”, “2-3&quot;, “2-1&quot;, etc. Direct caption is visibly much longer than the</td></tr><tr><td>(3)</td><td>motif-oriented caption. Alternates among path, star, triangle, and fork de- scriptions, obscuring the wheel motif.</td></tr></table>

Our results (Table 2) show that direct prompting often preserves enough connectivity information for graph recovery. However, across captions, we observe three recurring failure modes: (1) edge enumeration instead of motif abstraction, (2) low caption compactness, and (3) inconsistent or inaccurate motif interpretation. Figure 1 provides a representative example, annotated in Table 1. These observations show that graph recoverability can be achieved through edge enumeration, and therefore does not by itself provide sufficient evidence of motif-level abstraction. This motivates a prompting strategy that separates connectivity extraction from motif abstraction before generating the final caption.

## 5 Structurally Speaking

To better balance graph recovery and motif-level abstraction, we introduce Structurally Speaking, a structured chain-of-thought reasoning protocol for bidirectional graph-caption translation. This protocol is designed to enforce an intermediate representation between adjacency matrices and captions: neighbor lists expose explicit local connectivity, while motif analysis groups local edges into higher-level structural units. Instead of asking the model to directly produce captions or graphs in a single step, our protocol decomposes each translation direction into graph-specific reasoning stages before generating the final output.

The structured reasoning protocol follows the prompt templates below. The exact prompts used are provided in Appendix B. For graph-to-caption translation, we use:

Step 1: Convert adjacency matrix to neighbor list (0- indexed).

Step 2: Analyze structure (i.e., what motifs are in this pattern).

Step 3: Generate final caption.

For caption-to-graph translation, we use:

Step 1: Parse structural descriptions (i.e., identify the number of nodes and motifs).

Step 2: Assign indices and layout (i.e., assign nodes to different motifs).

Step 3: Create edge list (i.e., analyze the edge required to construct the motifs).

Step 4: Build neighbor listfor each node.

Step 5: Convert neighbor list to adj matrix.

The graph-to-caption template moves from connectivity extraction to motif-level abstraction, while the caption-to-graph template maps motiforiented language back to explicit topology. We compare three prompting settings. Direct prompting uses only the task instruction. Notably, the direct prompt already asks the model to describe graph motifs, as shown in Appendix B. Zero-shot structured prompting uses the Structurally Speaking templates without labeled examples, testing the reasoning scaffold alone. Few-shot structured prompting uses the same templates with example graph-caption pairs with full human-verified intermediate reasoning paths and motif-oriented captions, serving as a demonstration-informed upper bound for this task; these examples are disjoint from the 40-example test set. All methods use the same underlying GPT-5.1 model and fixed decoding settings (Singh et al., 2025).

## 6 Evaluation

Cycle-Consistency Evaluation. We evaluate two cycle-consistency settings. In Graph-Caption-

Table 2: Cycle-consistency evaluation using GPT-5.1 for all prompting methods. Graph-Caption-Graph evaluates graph recovery using edge precision, recall, and F1. Caption-Graph-Caption evaluates caption reconstruction using ROUGE-1 precision and recall, and caption compactness using average generated caption length (in characters).
<table><tr><td rowspan="2">Prompting Method</td><td colspan="3">Graph-Caption-Graph</td><td colspan="3">Caption-Graph-Caption</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>ROUGE-1 Precision</td><td>ROUGE-1 Recall</td><td>Avg. Length</td></tr><tr><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.07636</td><td>0.50933</td><td>1222.1</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>0.95751</td><td>0.97172</td><td>0.22378</td><td>0.55846</td><td>313.225</td></tr><tr><td>Few-shot Structured</td><td>0.99688</td><td>0.99688</td><td>0.99667</td><td>0.45166</td><td>0.53706</td><td>136.175</td></tr></table>

Graph, the model generates a caption from a graph and then reconstructs a graph from that caption. This measures graph recovery, i.e., whether the caption preserves enough information to recover the original topology. Given the original edge set E and reconstructed edge set $\hat { E } ,$ we compute edge precision $\begin{array} { r } { \begin{array} { r } { P \ = \ \frac { | E \cap \hat { E } | } { | \hat { E } | } } \end{array} } \end{array}$ , recall $R ~ = ~ \textstyle \frac { | \bar { E } \cap \hat { E } | } { | E | }$ , and $\begin{array} { r } { F 1 = \frac { 2 P R } { P + R } } \end{array}$

In Caption-Graph-Caption, the model reconstructs a graph from a reference caption and then generates a new caption from the reconstructed graph. This measures whether motif-oriented descriptions remain accurate and compact after translation through graph structure. We treat the humanwritten motif-oriented caption as the reference and the model-generated caption as the hypothesis. We use ROUGE (Lin, 2004) as a lightweight proxy for lexical overlap with human-verified motif captions, interpreted together with caption length (in characters) and qualitative inspection. We use the rouge-score implementation with stemming enabled. ROUGE-1 recall measures how much motifrelevant content is covered, while ROUGE-1 precision reflects how much additional wording is introduced. Thus, lower precision with longer captions may indicate verbose adjacency-oriented details rather than missing motif content alone. Together, these cycles evaluate both structural faithfulness and motif-level abstraction quality.

Structured Reasoning Analysis. As introduced in section 5, we evaluate three prompting schemes using GPT-5.1. Table 2 shows three main trends.

First, direct prompting achieves perfect edge precision, recall, and F1 in Graph-Caption-Graph, but produces the longest captions and lowest ROUGE-1 precision in Caption-Graph-Caption. This indicates that recovery is driven mainly by edge enumeration rather than motif abstraction. Its moderate ROUGE-1 recall may partly reflect this verbosity: the generated captions mention several motif-relevant terms, increasing lexical coverage, but these terms are sometimes embedded in inaccurate or self-contradictory motif interpretations.

Second, zero-shot structured prompting greatly reduces caption length and improves ROUGE-1 precision and recall over direct prompting, suggesting that the Structurally Speaking scaffold helps elicit more focused topology-to-motif abstraction without any labeled examples. Its small drop in Graph-Caption-Graph recall indicates that more compact captions may omit some edge details.

Third, few-shot structured prompting maintains near-perfect graph recovery while producing the shortest captions and highest ROUGE-1 precision. This suggests that demonstrations help align the desired output convention with motif-level abstraction, enabling compact captions without sacrificing recoverability. Qualitative examples in Figure 1 and Appendix C further show that structured prompting results in shorter captions that actually correspond to better motif abstraction, not just shorter text.

We therefore interpret the few-shot setting as a demonstration-informed upper-bound condition, while the zero-shot setting more directly tests the benefit of our structured reasoning scaffold. Overall, these results suggest that structured prompting helps balance graph recoverability with compact and accurate motif-oriented captioning. Additional detailed results are provided in Appendix D.

## 7 Conclusion

We studied graph captioning as a diagnostic probe of whether LLMs can infer motif-level structure from raw graph topology. Through bidirectional graph-caption translation, we evaluated both graph recovery and motif-oriented caption compactness. Our cycle-consistency evaluation shows that direct prompting can preserve topology through verbose edge enumeration, but does not necessarily reflect interpretable motif-level abstraction. Structurally Speaking reduces this mismatch by guiding topology-to-motif reasoning, eliciting more accurate and consistent motif-level structural inference while maintaining comparable graph recovery.

## Limitations

This work is a controlled diagnostic study rather than a comprehensive benchmark. Our experiments use a synthetic motif-based dataset and a single LLM, so the results may not generalize to all graph families, larger graphs, or other models. The captioned evaluation set is also small because motiforiented captions require human verification. In addition, ROUGE-1 and caption length provide only approximate signals of motif-oriented caption quality; we therefore interpret them together with graph recovery metrics and qualitative examples. Future work should expand the dataset, evaluate additional models, and incorporate human judgments of caption usefulness and abstraction level.

## Ethical Considerations

This work uses synthetic graph structures and human-written motif-oriented captions, and does not involve personal, sensitive, or human-subject data. As such, we do not identify direct ethical risks from the dataset itself. The main consideration is a general risk of LLM-based graph-language systems: generated captions may appear fluent and plausible while omitting, distorting, or overspecifying structural information. We address this risk in our study by evaluating captions through graph recovery, caption reconstruction, caption length, and qualitative inspection rather than relying on fluency alone. In practical applications, generated graph captions should be verified against the underlying graph before being used for analysis or decision-making.

## References

Bahare Fatemi, Jonathan Halcrow, and Bryan Perozzi. 2023. Talk like a graph: Encoding graphs for large language models. arXiv preprint arXiv:2310.04560.

Jiayan Guo, Lun Du, Hengyu Liu, Mengyu Zhou, Xinyi He, and Shi Han. 2023. Gpt4graph: Can large language models understand graph structured data? an empirical evaluation and benchmarking. arXiv preprint arXiv:2305.15066.

Gaetan Hains, Youry Khmelevsky, and Thibaut Tachon. 2019. From natural language to graph queries. In 2019 IEEE Canadian Conference of Electrical and Computer Engineering (CCECE), pages 1–4, Edmonton, Canada. IEEE.

Jie He, Yijun Yang, Wanqiu Long, Deyi Xiong, Victor Gutierrez Basulto, and Jeff Z. Pan. 2025. Evaluating and improving graph to text generation with large

language models. In Proceedings ofthe 2025 Conference ofthe Nations ofthe Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 10219–10244, Albuquerque, New Mexico. Association for Computational Linguistics.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Structgpt: A general framework for large language model to reason over structured data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9237–9251.

Bowen Jin, Gang Liu, Chi Han, Meng Jiang, Heng Ji, and Jiawei Han. 2024. Large language models on graphs: A comprehensive survey. IEEE Transactions on Knowledge and Data Engineering, 36(12):8622– 8642.

Qianlong Li, Chen Huang, Shuai Li, Yuanxin Xiang, Deng Xiong, and Wenqiang Lei. 2025. Graphotter: Evolving llm-based graph reasoning for complex table question answering. In Proceedings of the 31st International Conference on Computational Linguistics, pages 5486–5506.

Yuanyuan Liang, Keren Tan, Tingyu Xie, Wenbiao Tao, Siyuan Wang, Yunshi Lan, and Weining Qian. 2024. Aligning large language models to a domainspecific graph database for nl2gql. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, CIKM ’24, page 1367–1377, New York, NY, USA. Association for Computing Machinery.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In Text Summarization Branches Out, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Bryan Perozzi, Bahare Fatemi, Dustin Zelle, Anton Tsitsulin, Mehran Kazemi, Rami Al-Rfou, and Jonathan Halcrow. 2024. Let your graph do the talking: Encoding structured data for llms. arXiv preprint arXiv:2402.05862.

Leonardo F. R. Ribeiro, Martin Schmitt, Hinrich Schütze, and Iryna Gurevych. 2021. Investigating pretrained language models for graph-to-text generation. In Proceedings ofthe 3rd Workshop on Natural Language Processing for Conversational AI, pages 211–227, Online. Association for Computational Linguistics.

Wenbo Shang and Xin Huang. 2025. A survey of large language models on generative graph analytics: Query, learning, and applications. IEEE Transactions on Knowledge and Data Engineering, 37(12):6799– 6819.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, and 1 others. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2024. Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 491–500.

Jianing Wang, Junda Wu, Yupeng Hou, Yao Liu, Ming Gao, and Julian McAuley. 2024. Instructgraph: Boosting large language models via graph-centric instruction tuning and preference alignment. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 13492–13510.

Yuxin You, Zhen Liu, Xiangchao Wen, Yongtao Zhang, and Wei Ai. 2025. Large language models meet graph neural networks: a perspective of graph mining. Mathematics, 13(7):1147.

Shuo Yu, Yingbo Wang, Ruolin Li, Guchun Liu, Yanming Shen, Shaoxiong Ji, Bowen Li, Fengling Han, Xiuzhen Zhang, and Feng Xia. 2026. Graph2text or graph2token: A perspective of large language models for graph learning. ACM Trans. Inf. Syst., 44(3).

Yizhuo Zhang, Heng Wang, Shangbin Feng, Zhaoxuan Tan, Xiaochuang Han, Tianxing He, and Yulia Tsvetkov. 2024. Can llm graph reasoning generalize beyond pattern memorization? In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 2289–2305.

## A Motif-Based Graphs

Figure 2 shows examples from our synthetic motifbased graph dataset.

## B Prompting Schemes

For the three prompting schemes evaluated in this paper, we use the following exact instructions to prompt GPT-5.1.

• Direct Prompting:

Caption You are to generate a graph pattern in its ad  
to jacency matrix form that matches the given   
Graph: caption. Please write the adjacency matrix in a clearformat, starting with ’Adjacency Matrix:’ followed by the matrix itself, expressed as list of lists in string form.

Graph You are to generate graph pattern caption de-$t o$ scribing the graph motifsfrom the given adja-$C a p \mathrm { . }$ cency matrix. Start the output with ’Caption:’ tion: and then write the caption.

• Zero-shot structured prompting:

![](images/82aa072f21095011f9437be274eef1ca9b776cf0bc7fcd801e90828624fd5178.jpg)  
Figure 2: Examples from our synthetic motif-based graph dataset. Each row corresponds to a base motif family: cycle, star, path, wheel, and clique. The left column shows clean motif examples, while the right column shows examples from the same motif family with a single edge addition or removal perturbation. The two columns illustrate clean and perturbed cases but do not necessarily depict paired versions of the same graph.

Caption You are to generate a graph pattern that   
to matches the above given caption. Please rea-  
Graph: son through these steps: step 1: Parse structural descriptions (i.e., identify the number of nodes and motifs), step 2: Assign indices and layout (i.e., assign nodes to different motifs), step 3: Create edge list (i.e., analyze the edge required to construct the motifs), step 4: Build neighbor list for each node, step 5: convert neighbor list to adj matrix. Please output the reasoning stepsfor a given graph pattern caption and start the outputfor the last step with ’Adjacency Matrix: ’ followed by the matrix itself, expressed as list of lists in string form.   
Graph You are to generate graph pattern captions.   
to Please reason through these steps: step 1:   
$C a p \mathrm { - }$ convert adjacency to produce neighbor lists,   
tion: step 2: analyze structure (i.e., what motifs are in this pattern), step 3: generate final caption. Please output the reasoning steps for a given adjacency matrix and start the outputfor the last step with ’Caption: ’.

• Few-shot structured prompting:

![](images/c7d73cf8195715023bba918f0cfb0df2da44634087aa6668620eab5fc360e5e5.jpg)  
Figure 3: Examples of generated captions across prompting settings. The top row shows perturbed motif graphs with human-verified motif-oriented captions, followed by captions generated by direct prompting, zero-shot structured prompting, and few-shot structured prompting. Structured prompting reduces verbosity and self-contradictory descriptions, while better focusing on motif-oriented graph patterns.

![](images/29c3459613ceeb9b4bbc242d8d26317f5716c7b9caff96e551fa4c06c44307ee.jpg)

## C Additional Qualitative Examples

Figure 3 presents additional examples comparing generated captions across prompting settings. These examples illustrate how the captions differ in motif abstraction, structural accuracy, specificity, and length. Overall, these supplementary qualitative results further support our claim that structured prompting reduces verbosity and self-contradictory descriptions while producing captions that better capture motif-oriented graph patterns.

## D Additional Quantitative Analyses

We further analyze model performance across graph perturbation settings and motif families. These analyses use the same evaluation metrics as in the main paper: Graph-Caption-Graph F1 for graph recovery, ROUGE-1 precision and recall for Caption-Graph-Caption caption reconstruction, and average generated caption length for caption compactness.

As shown in Table 3, all prompting schemes perform better on clean motif graphs than on perturbed graphs. This is expected because edge additions and removals introduce structural variation beyond the base motif patterns. To avoid inflating the aggregate results with many easy clean examples, our test set includes only a small number of clean graphs, with two clean examples per motif family. As a result, the overall results in Table 2 more closely reflect performance on perturbed graphs, which better approximate the structural irregularities found in natural graph patterns.

Table 4 further shows that Wheel patterns are among the most challenging motif families across prompting schemes. This is also expected because a wheel combines a central hub with a rim structure, requiring the model to recognize both starlike and cycle-like organization and their composition. Nevertheless, the Wheel-specific results are broadly consistent with the overall trends in Table 2: structured prompting improves caption compactness and motif-oriented reconstruction while maintaining comparable graph recovery. This suggests that Structurally Speaking helps guide the model through motif composition even for more structurally complex patterns such as wheels.

Table 3: Cycle-consistency evaluation on clean and perturbed motif graphs using GPT-5.1, where perturbed graphs contain random edge additions or removals from the base motif structure.
<table><tr><td rowspan="2">Graph Type</td><td rowspan="2">Prompting Method</td><td colspan="3">Graph-Caption-Graph</td><td colspan="3">Caption-Graph-Caption</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>ROUGE-1 Precision</td><td>ROUGE-1 Recall</td><td>Avg. Length</td></tr><tr><td rowspan="3">Clean</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.100823</td><td>0.500799</td><td>665.5</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.266728</td><td>0.571888</td><td>200.7</td></tr><tr><td>Few-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.575181</td><td>0.631384</td><td>94.5</td></tr><tr><td rowspan="3">Perturbed</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.068201</td><td>0.512170</td><td>1407.633</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>0.943351</td><td>0.962294</td><td>0.209464</td><td>0.553990</td><td>350.733</td></tr><tr><td>Few-shot Structured</td><td>0.995833</td><td>0.995833</td><td>0.995556</td><td>0.410488</td><td>0.505617</td><td>150.067</td></tr></table>

Table 4: Cycle-consistency evaluation by base motif family using GPT-5.1.
<table><tr><td rowspan="2">Motif</td><td rowspan="2">Prompting Method</td><td colspan="3">Graph-Caption-Graph</td><td colspan="3">Caption-Graph-Caption</td></tr><tr><td>Precision</td><td>Recall</td><td>F1</td><td>ROUGE-1 Precision</td><td>ROUGE-1 Recall</td><td>Avg. Length</td></tr><tr><td rowspan="3">Cycle</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.070145</td><td>0.484059</td><td>1078.444</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.243284</td><td>0.557181</td><td>357.111</td></tr><tr><td>Few-shot Structured</td><td>0.986111</td><td>1.0</td><td>0.992593</td><td>0.526199</td><td>0.538382</td><td>121.889</td></tr><tr><td rowspan="3">Star</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.067775</td><td>0.560572</td><td>1395.375</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>0.848930</td><td>0.895803</td><td>0.263899</td><td>0.566327</td><td>292.375</td></tr><tr><td>Few-shot Structured</td><td>1.0</td><td>0.984375</td><td>0.991667</td><td>0.409470</td><td>0.532376</td><td>154.375</td></tr><tr><td rowspan="3">Path</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.083385</td><td>0.424597</td><td>781.2</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.292959</td><td>0.595345</td><td>228.2</td></tr><tr><td>Few-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.514075</td><td>0.560210</td><td>115.6</td></tr><tr><td rowspan="3">Wheel</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.062481</td><td>0.507626</td><td>1780.545</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>0.955372</td><td>0.972944</td><td>0.165656</td><td>0.549543</td><td>369.182</td></tr><tr><td>Few-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.400842</td><td>0.542979</td><td>148.636</td></tr><tr><td rowspan="3">Clique</td><td>Direct Prompting</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.110122</td><td>0.535666</td><td>664.167</td></tr><tr><td>Zero-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.194631</td><td>0.545271</td><td>256.333</td></tr><tr><td>Few-shot Structured</td><td>1.0</td><td>1.0</td><td>1.0</td><td>0.444836</td><td>0.480310</td><td>126.5</td></tr></table>