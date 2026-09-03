# SALA: Semantic-Aware Logical Alignment for Complex Reasoning in In-Context Learning

Zhao Ji<sup>1,2</sup>, Wenqing Chen<sup>1,2</sup>\*, Zhixuan Chu<sup>3</sup>, Jianxing Yu<sup>4,5</sup>, Jingping Liu<sup>1,2</sup>, Shanhe Zhao<sup>6</sup>, Zibin Zheng<sup>1,2</sup>

<sup>1</sup>School of Software Engineering, Sun Yat-sen University, Zhuhai, China <sup>2</sup>Zhuhai Key Laboratory of Trusted Large Language Models, Zhuhai, China <sup>3</sup>The State Key Laboratory of Blockchain and Data Security, Zhejiang University, Hangzhou, China

<sup>4</sup>School of Artificial Intelligence, Sun Yat-sen University, Zhuhai, China <sup>5</sup>Key Laboratory of Sustainable Tourism Smart Assessment Technology, Ministry of Culture and Tourism, Zhuhai, China <sup>6</sup>Merchants Union Consumer Finance Company Limited, Shenzhen, China

## Abstract

Effective in-context learning (ICL) for complex reasoning relies on selecting the right demonstrations. Traditional retrieval methods based on surface similarity fail to capture the underlying problem-solving logic. Recent logic-based methods address this by matching predefined reasoning steps, but the rigid rules and exactmatch criteria is improper to handle flexible or diverse reasoning processes. To address the problem, we propose SALA, a Semantic-Aware Logical Alignment framework. Instead of relying on a fixed inventory, SALA automatically learns task-specific reasoning operations. It then embeds these operations into a continuous semantic space and uses dynamic time warping (DTW) to align the reasoning sequences. This approach allows for soft, flexible matching of reasoning logic while remaining highly interpretable. Experiments across four reasoning benchmarks and three LLMs demonstrate that SALA outperforms existing demonstration selection methods. Further analysis confirms the roles of the operation induction and the logical semantic alignment.

## 1 Introduction

Large Language Models (LLMs) (Ouyang et al., 2022; Achiam et al., 2023; Touvron et al., 2023; Bai et al., 2023; Ye et al., 2023b; Bahrini et al., 2023) have shown strong in-context learning (ICL) ability (Dong et al., 2024; Wies et al., 2023; Luo et al., 2024; Bhattamishra et al., 2024), enabling them to solve complex reasoning tasks from a few labeled demonstrations. ICL performance largely depends on demonstration selection, since useful examples can improve accuracy (Ye et al., 2023a), whereas mismatched or noisy examples may introduce reasoning biases and degrade performance (Zhang et al., 2025c). The importance of selecting useful examples is further enhanced in recent agentic LLM systems, where reasoning experiences are stored as memories and retrieved to guide future tasks (Ouyang et al., 2025).

![](images/9c518835041f24d7fa2a925ebd7d5bd24a03be48f629102ab8a4be041911d0ac.jpg)  
Figure 1: An example of the induction process of taskspecific operations.

Existing studies have improved demonstration selection in general domain (Rubin et al., 2022; Li et al., 2023; Ye et al., 2023a; Yang et al., 2023; Qin et al., 2024). However, most retrieval models rely on embedding similarity and the reasoning logic can be obscured by surface semantics.

The key question is how to represent demonstrations for reasoning tasks. Recent reasoningaware methods construct intermediate representations before retrieval, including generated reasoning paths (Qin et al., 2024), reasoning patterns (Zhang et al., 2025b), latent reasoning skills (Xu et al., 2024), and reasoning graphs (Lin et al., 2025). These methods suggest that raw question embeddings are often insufficient, since reasoning-relevant signals can be diluted by topics, entities, and surface semantics. Problem-solving logic (PSL) takes a more symbolic route by representing each problem as a sequence of predefined reasoning operations and matching demonstrations at the operation level (Ma et al., 2025). Such operation-based representations reduce the influence of surface wording and provide a more explicit view of the problemsolving process.

Despite these advances, existing representations still struggle to make reasoning logic both comparable and adaptable. Free-form rationales or reasoning paths can express diverse reasoning processes, but their natural-language form often makes the representation noisy and unstable. Symbolic operation sequences, as explored by PSL, make reasoning steps easier to compare, but fixed operation spaces and exact matching rules limit their ability to generalize across task-specific and semantically equivalent reasoning patterns. This motivates a logical alignment framework that represents reasoning explicitly while aligning it semantically.

To this end, we propose SALA, a Semantic-Aware Logical Alignment framework for reasoningoriented demonstration selection. SALA constructs a task-adaptive operation space by inducing reasoning operations from downstream data, enabling demonstrations to be represented with problemsolving units beyond the predefined operation set. Figure 1 illustrates how SALA induces additional task-specific operations when the predefined operation set cannot fully express the reasoning logic of a downstream question. Then we embed operation descriptions into a continuous semantic space and applies dynamic time warping (DTW) to align reasoning-operation sequences (Sakoe and Chiba, 1978). In this way, we keep the reasoning representation explicit while enabling soft semantic alignment beyond exact symbolic correspondence.

We conduct experiments on four reasoning benchmarks across three LLMs. SALA achieves better average performance against similarity-based, learning-based, and reasoning-aware demonstration selection baselines. Ablation studies further show that both task-adaptive operation construction and semantic sequence alignment contribute to the improvement. The key contributions of this work are as follows:

• We propose SALA, a reasoning-oriented demonstration selection framework that represents problem-solving logic with explicit operation sequences and aligns them in semantic space.

• We introduce a task-adaptive operation space that extends predefined operations with reusable reasoning units induced from downstream data, enabling more expressive problem-solving representations without manual operation design.

• We design a semantic DTW-based alignment strategy to measure logical similarity across reasoning-operation sequences with different lengths and decomposition granularities.

• We validate SALA on four reasoning benchmarks and three LLMs, showing consistent average gains over recent state-of-the-art baselines and confirming the roles of operation construction and semantic alignment through ablations.

## 2 Related Work

Existing work on ICL demonstration selection (Dong et al., 2024) has studied this problem from lexical, semantic, learned, diversity-aware, and reasoning-aware perspectives. We review the most relevant lines of work in general and reasoning domains.

## 2.1 General Demonstration Selection

Early demonstration selection methods retrieve examples according to lexical overlap or sentencelevel semantic similarity. Sparse retrieval methods such as BM25 (Robertson and Zaragoza, 2009) estimate query-example word overlap, while embedding-based methods use pretrained encoders such as BERT (Devlin et al., 2019) to retrieve semantically similar demonstrations. Later studies improve this basic retrieval paradigm by considering supportiveness, diversity, and compositionality. For example, support-example selection chooses demonstrations according to their usefulness for the target query (Li and Qiu, 2023). DPP-based selection balances relevance and diversity (Yang et al., 2023), and compositional exemplar selection models interactions among demonstrations beyond nearest-neighbor retrieval (Ye et al., 2023a). Iterative selection further shows that demonstration retrieval can be treated as a multi-step process rather than a one-shot nearest-neighbor search (Qin et al., 2024).

![](images/c7be151944eefb1073e44aa2634baf4a5dfb91ab3fa43c3ad686712b95f1804d.jpg)  
Figure 2: Overview of the SALA framework. SALA constructs a task-adaptive operation set, parses queries and demonstrations into reasoning-operation sequences, and applies DTW-based semantic alignment to retrieve logically similar demonstrations for ICL prompting.

Another line of work formulates demonstration selection as a learnable retrieval or optimization problem. EPR trains a retriever from languagemodel feedback (Rubin et al., 2022), while UDR learns a unified retriever for cross-task demonstration selection (Li et al., 2023). More recent methods use model preference, gradient matching, reinforcement learning, or many-shot optimization to improve selection quality (Zhang et al., 2025c,a; Wang et al., 2025; Purohit et al., 2025). These methods improve the retrieval of useful demonstrations, but their selection signals are still mostly derived from input similarity or model-level feedback rather than the reasoning process itself. For complex reasoning tasks, a helpful demonstration should not only be topically or semantically related to the query, but also follow a compatible problemsolving logic.

## 2.2 Reasoning Demonstration Selection

Recent work has started to construct reasoningoriented representations. One direction uses naturallanguage reasoning descriptions. Skill-KNN rewrites inputs into skill-based descriptions before applying embedding-based retrieval (An et al., 2023). Luo et al. (2023) extend retrieval-based ICL to chainof-thought (CoT) prompting, and iterative demonstration selection uses generated reasoning paths to guide example retrieval (Qin et al., 2024). These methods keep the representation flexible, but the reasoning signal is still expressed in free-form text, which can make the retrieval noisy or unstable.

A second direction introduces more structured reasoning representations. Reasoning-pattern methods select demonstrations according to task-specific reasoning patterns (Zhang et al., 2025b), LaRS learns latent reasoning skills from CoT rationales (Xu et al., 2024), and RGER represents intermediate reasoning steps as reasoning graphs for exemplar retrieval (Lin et al., 2025). For mathematical reasoning, LMS3 further shows that useful demonstrations should balance semantic similarity and inference stability (Liu et al., 2025). These methods provide stronger reasoning-oriented signals than raw question embeddings, but latent skills are less directly interpretable, and graph-based representations require more complex structure matching across examples.

PSL-guided ICL is the most closely related symbolic approach (Ma et al., 2025). It represents each problem as a sequence of predefined QDMR-style reasoning operations (Wolfson et al., 2020) and selects demonstrations through operation-level exact matching. While this shows the value of explicit operation sequences for representing problem-solving logic, it still relies on a fixed operation space and rigid symbolic matching. SALA moves beyond symbolic matching by combining operation-level reasoning representations with semantic alignment. It constructs a task-adaptive operation space from downstream data and aligns operation sequences in semantic space with DTW, allowing explicit reasoning representations to be compared flexibly across diverse reasoning processes.

## 3 Methodology

As shown in Figure 2, SALA follows a four-stage pipeline for reasoning-oriented demonstration selection, with adaptive operation construction and DTW-based semantic alignment serving as the two key mechanisms.

## 3.1 Problem Formulation

Let $\mathcal { D } = \{ ( q _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ denote the demonstration pool derived from the training set of a downstream task, where $q _ { i }$ and $y _ { i }$ are the question and its answer, respectively. Given a test query $q ^ { * }$ , the goal is to select k demonstrations from $\mathcal { D }$ that support the target LLM in solving $q ^ { * }$

For reasoning tasks, we model the problem-solving logic of a question as an ordered sequence of reasoning operations. Given an operation set $\mathcal { O }$ , a parser maps a question q into

$$
\pi ( q ; \mathcal { O } ) = [ o _ { 1 } , \ldots , o _ { m } ] , \quad o _ { j } \in \mathcal { O } .\tag{1}
$$

After constructing the task-adaptive operation set $\mathcal { O } _ { t a s k }$ , we denote the query sequence and the candidate sequence as $Q ^ { * } = \pi ( q ^ { * } ; { \mathcal { O } } _ { t a s k } )$ and $E _ { i } =$ $\pi ( q _ { i } ; \mathcal { O } _ { t a s k } )$ , respectively. Demonstration selection is then formulated as:

$$
\mathcal { D } _ { k } ( \boldsymbol { q } ^ { * } ) = \mathrm { T o p K } _ { ( q _ { i } , y _ { i } ) \in \mathcal { D } } S i m ( Q ^ { * } , E _ { i } ) ,\tag{2}
$$

where $S i m ( Q ^ { * } , E _ { i } )$ measures the alignment between the problem-solving logic of the query and that of the candidate demonstration. SALA focuses on constructing the task-adaptive operation set $\mathcal { O } _ { t a s k }$ and defining $S i m ( \cdot , \cdot )$ through semantic DTW alignment.

## 3.2 Adaptive Construction of Reasoning Operation Sets

To address the limited transferability of fixed predefined reasoning operation sets, we construct a task-adaptive reasoning operation set $\mathcal { O } _ { t a s k }$ by extending the 13 predefined QDMR reasoning operations $\mathcal { O } _ { p r e }$ (Wolfson et al., 2020) (see Appendix B for the full list) with operations induced from downstream training data. The construction process consists of candidate operation induction followed by two-stage deduplication.

## 3.2.1 Candidate Operation Induction

For each training question $q _ { i }$ in the candidate demonstration pool D, we condition an LLM on the predefined QDMR operation set $\mathcal { O } _ { p r e }$ and prompt it to identify additional operations only when the existing set is insufficient to cover the problem-solving logic of $q _ { i }$ . If no extension is needed, the model returns an empty result; otherwise, it outputs one or more candidate operation descriptions in the predefined format. Collecting the non-empty outputs over all questions yields a candidate operation pool $\mathcal { O } _ { c a n d } = \{ o _ { 1 } , o _ { 2 } , . . . , o _ { M } \}$ , where M is the number of candidate new operations. The full prompt used for this step is provided in Appendix C.1.

## 3.2.2 Two-Stage Deduplication of New Operations

To avoid redundancy and ensure that newly induced operations truly complement the predefined ones, we apply a two-stage deduplication procedure.

Stage 1: Heuristic Name-Based Deduplication. We first perform a lightweight heuristic deduplication over the names of candidate operations. For each candidate operation $o _ { k } \in \mathcal { O } _ { c a n d }$ , we extract its operation name, denoted as $n _ { k }$ . We then normalize the name by lowercasing it and removing spaces and underscore characters, yielding a normalized form $\tilde { n } _ { k }$ . Candidate entries whose extracted names contain obvious formatting artifacts or malformed markers are discarded at this stage. Let $\mathcal { O } _ { r e p }$ denote the retained representative operation set, initialized as an empty set. We process the candidates sequentially and compare the normalized name of the current candidate with the normalized operation names already stored in $\mathcal { O } _ { r e p }$

For a current candidate $o _ { k }$ with normalized name $\tilde { n } _ { k }$ , and an existing representative operation $\bar { o } _ { i } \in$ $\mathcal { O } _ { r e p }$ with normalized name $\tilde { n } _ { \bar { o } _ { i } }$ , we apply the following deterministic rules:

• If $\tilde { n } _ { k } \subseteq \tilde { n } _ { \bar { o } _ { i } }$ , we replace $\bar { o } _ { i }$ with $o _ { k }$

• If $\tilde { n } _ { \bar { o } _ { i } } \subseteq \tilde { n } _ { k } ,$ , we discard $o _ { k }$ .

• Otherwise, we keep both entries unchanged.

If no containment relation is found between $\tilde { n } _ { k }$ and any existing representative in $\mathcal { O } _ { r e p }$ , we append $o _ { k }$ to $\mathcal { O } _ { r e p } .$ . After all candidates are processed, $\mathcal { O } _ { r e p } = \{ \bar { o } _ { 1 } , \bar { o } _ { 2 } , . . . , \bar { o } _ { P } \}$ becomes the representative operation set produced by Stage 1. This heuristic removes near-duplicate operation names caused by minor formatting differences or partially overlapping name variants, while preserving the original full operation descriptions for subsequent processing. Because the procedure depends only on normalized names, containment checks, and candidate order, it is reproducible given the same candidate operation pool.

Stage 2: Function Duplication Detection Based on LLM Judgment. When $\mathcal { O } _ { r e p } = \{ \bar { o } _ { 1 } , \bar { o } _ { 2 } , \dots , \bar { o } _ { P } \}$ is obtained after Stage 1, we further remove functional redundancy through a sequential judgment process. We initialize the task-adaptive operation set as $\mathcal { O } _ { t a s k } ^ { ( 0 ) } = \mathcal { O } _ { p r e }$ . For each representative operation ${ \bar { o } } _ { i } \in { \mathcal { O } } _ { r e p }$ , we construct a function comparison prompt that includes the core function description of $\bar { o } _ { i }$ and the core function descriptions of all operations in the current operation set $\bar { \mathcal { O } } _ { t a s k } ^ { ( i - 1 ) }$ . The LLM outputs a binary judgment result $J ( \bar { o } _ { i } , \mathcal { O } _ { t a s k } ^ { ( i - 1 ) } ) \in$ $\{ 0 , 1 \}$ , where $J ( { \bar { o } } _ { i } , { \mathcal { O } } _ { t a s k } ^ { ( i - 1 ) } ) = 1$ means that $\bar { o } _ { i }$ functionally overlaps with at least one operation in $\mathcal { O } _ { t a s k } ^ { ( i - 1 ) }$ , and $J ( \bar { o } _ { i } , \mathcal { O } _ { t a s k } ^ { ( i - 1 ) } ) = 0$ means no overlap. The operation set is then updated recursively:

$$
\mathcal { O } _ { t a s k } ^ { ( i ) } = \left\{ \begin{array} { l l } { \mathcal { O } _ { t a s k } ^ { ( i - 1 ) } \cup \{ \bar { o } _ { i } \} , } & { \mathrm { i f ~ } J ( \bar { o } _ { i } , \mathcal { O } _ { t a s k } ^ { ( i - 1 ) } ) = 0 , } \\ { \mathcal { O } _ { t a s k } ^ { ( i - 1 ) } , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{3}
$$

After all representative operations are examined, the final task-adaptive reasoning operation set is $\mathcal { O } _ { t a s k } = \mathcal { O } _ { t a s k } ^ { ( P ) }$

This sequential process ensures that each accepted operation complements both the predefined operations and the previously accepted induced operations. The prompts for candidate induction (Section C.1) and deduplication (Section C.3) are fully documented in Appendix C.

## 3.3 Construction of the Reasoning Operation Embedding Library

To capture the functional semantic associations of operations and support semantic alignment, we convert discrete operations into continuous semantic embeddings and construct a reasoning operation embedding library, providing the foundation for subsequent similarity calculations.

First, we generate semantic description texts for each operation $o \in \mathcal { O } _ { t a s k }$ , denoted as desc(o). In our implementation, each desc(o) describes the core function of the corresponding operation, and the description texts are used as inputs to a pretrained embedding model.

Next, we use a pretrained text embedding model as the semantic encoder, denoted as $f _ { e m b } ( \cdot ) : s t r $ $\mathbb { R } ^ { d }$ , where d is the embedding dimension. Specifically, the implementation uses a Hugging Face embedding model to encode the operation description text and obtains the initial semantic embedding vector through mean pooling over the hidden states:

$$
u _ { o } ^ { i n i t } = f _ { e m b } ( d e s c ( o ) ) \in \mathbb { R } ^ { d }\tag{4}
$$

We apply $\ell _ { 2 }$ normalization to the initial vector before DTW matching:

$$
v _ { o } = \frac { u _ { o } ^ { i n i t } } { \parallel u _ { o } ^ { i n i t } \parallel _ { 2 } }\tag{5}
$$

where $v _ { o } \in \mathbb { R } ^ { d }$ and $\| v _ { o } \| _ { 2 } = 1$ is the final semantic embedding vector of operation o.

Finally, the mapping relationship, $\{ ( o , v _ { o } ) \mid o \in$ $\mathcal { O } _ { t a s k } \}$ , is used to form the reasoning operation embedding library $L _ { e m b }$ . In practice, the embeddings are precomputed once for all operations in $\mathcal { O } _ { t a s k }$ and then reused during retrieval.

## 3.4 DTW-Based Semantic Alignment

In this section, we propose a semantic alignment strategy that maps reasoning operation sequences to semantic embedding sequences and uses DTW to calculate sequence-level similarity (Sakoe and Chiba, 1978), enabling soft logical alignment across sequences of different lengths.

## 3.4.1 Reasoning Operation Sequence Parsing

Based on the constructed task-adaptive reasoning operation set $\mathcal { O } _ { t a s k }$ , we use an LLM to map each query and candidate demonstration into an ordered reasoning operation sequence. Each sequence represents the problem-solving process as a series of operations drawn from $\mathcal { O } _ { t a s k }$ . The parsing prompt is provided in Appendix C.2.

## 3.4.2 Semantic Sequence Generation

We retrieve the semantic embedding of each reasoning operation in the parsed reasoning operation sequence from the embedding library, converting discrete reasoning operation sequences into continuous semantic embedding sequences. For the query sequence $Q ^ { * } = [ o _ { 1 } ^ { * } , o _ { 2 } ^ { * } , \ldots , o _ { m } ^ { * } ]$ , its operation embedding sequence is $\mathbf { Q } ^ { * } = [ v _ { o _ { 1 } ^ { * } } , v _ { o _ { 2 } ^ { * } } , \ldots , v _ { o _ { m } ^ { * } } ]$ , where $v _ { o _ { i } ^ { * } }$ is the semantic embedding of the i-th query operation $o _ { i } ^ { * }$ . For a candidate demonstration sequence $E _ { i } ,$ we simplify it to $E = [ o _ { 1 } ^ { E } , o _ { 2 } ^ { E } , \dots , o _ { n } ^ { E } ]$ when computing DTW, and its semantic sequence is $\mathbf { E } = [ v _ { o _ { 1 } ^ { E } } , v _ { o _ { 2 } ^ { E } } , \dots , v _ { o _ { n } ^ { E } } ]$ , where $v _ { o _ { i } ^ { E } }$ is the semantic embedding of the j-th candidate operation $o _ { j } ^ { E }$

## 3.4.3 DTW-based Similarity Calculation

The DTW algorithm calculates the semantic similarity between the candidate semantic sequence E and the query semantic sequence $\mathbf { Q } ^ { \ast }$ through three steps. First, we construct a pairwise distance matrix $D \in \mathbb { R } ^ { m \times n }$ , where the element $D ( i , j )$ represents the semantic distance between the i-th vector $v _ { o _ { i } ^ { * } }$ in $\mathbf { Q } ^ { * }$ and the $j \mathrm { - t h }$ vector $v _ { o _ { j } ^ { E } }$ in E. Consistent with the implementation, we use Euclidean distance to measure this cost:

$$
D ( i , j ) = \| v _ { o _ { i } ^ { * } } - v _ { o _ { j } ^ { E } } \| _ { 2 } .\tag{6}
$$

where $\| \cdot \| _ { 2 }$ denotes the Euclidean norm. Since the operation embeddings are precomputed in a fixed d-dimensional space, this distance directly measures the semantic discrepancy between two reasoning operations.

Next, we search for the optimal warping path. The warping path $W = [ w _ { 1 } , w _ { 2 } , \dots , w _ { k } ]$ is a sequence of coordinate pairs $( i , j )$ that satisfies three constraints to ensure a valid alignment: the boundary constraint (starting at (1, 1) and ending at $( m , n )$ to cover the entire problem-solving logic), the monotonicity constraint (avoiding reverse matching to maintain consistency with the reasoning process), and the continuity constraint (ensuring continuous paths without jumps to avoid missing key reasoning steps). The optimal warping path is the one with the minimum cumulative cost among all feasible paths, where the cumulative cost $\gamma ( i , j )$ from the starting point (1, 1) to the point $( i , j )$ is calculated recursively as:

$$
\begin{array} { r l } & { \gamma ( i , j ) = D ( i , j ) + } \\ & { \operatorname* { m i n } \{ \gamma ( i - 1 , j ) , \gamma ( i , j - 1 ) , \gamma ( i - 1 , j - 1 ) \} . } \end{array}\tag{7}
$$

with the initial condition $\gamma ( 1 , 1 ) = D ( 1 , 1 )$

Finally, we calculate the sequence similarity. To reduce the impact of sequence length differences, we divide the cumulative DTW distance by the length of the optimal alignment path to obtain the average alignment distance, and then transform it into a bounded similarity score:

$$
S i m ( Q ^ { * } , E _ { i } ) = \frac { 1 } { 1 + \operatorname* { m i n } \left( \frac { \gamma ( m , n ) } { k } , 1 \right) } .\tag{8}
$$

where k denotes the length of the optimal warping path. Here, $\gamma ( m , n ) / k$ is the average DTW alignment distance per step. Since operation embeddings are $\ell _ { 2 } .$ -normalized unit vectors, their pairwise Euclidean distance lies in [0, 2], and the - min(·, 1) clipping bounds the effective distance to [0, 1]. Therefore, $S i m ( Q ^ { * } , E _ { i } ) \in [ 0 . 5 , 1 ]$ , and a value closer to 1 indicates higher semantic similarity and greater consistency in problem-solving logic between the demonstration and the query.

We rank all demonstrations by DTW semantic similarity in descending order and select the top-k most similar ones. Then, following an easy-to-hard curriculum (Ma et al., 2025), we sort the selected demonstrations in ascending order of reasoning operation sequence length.

## 4 Experiments

## 4.1 Experimental Setup

## 4.1.1 Datasets

We evaluate SALA on four reasoning benchmarks covering diverse task types and difficulty levels. SVAMP (Patel et al., 2021) contains 1,000 arithmetic word problems. GSM8K (Cobbe et al., 2021) consists of 1,319 high-quality grade-school math problems that typically require multi-step reasoning. CommonsenseQA (Talmor et al., 2019) is a multiple-choice commonsense reasoning benchmark with 10,881 questions, each associated with five answer options. StrategyQA (Geva et al., 2021) is an open-domain question-answering benchmark with 2,290 questions. Detailed dataset statistics and split information are provided in Appendix A.

## 4.1.2 Baselines

We compare SALA with seven representative ICL baselines, including Random sampling, EPR with a contrastive retriever (Rubin et al., 2022), BM25 retrieval (Robertson and Zaragoza, 2009), TopK-BERT retrieval (Devlin et al., 2019), and DPP-BERT, which combines BERT-based retrieval with a Determinantal Point Process to balance relevance and diversity (Chen et al., 2018; Yang et al., 2023). We also include two reasoning-aware baselines: PSL, which selects demonstrations using problemsolving logic guidance (Ma et al., 2025), and LMS3, which selects demonstrations based on semantic similarity and inference stability (Liu et al., 2025).

## 4.1.3 Implementation Details

We evaluate SALA on three LLMs: Llama3-8B-Instruct (Grattafiori et al., 2024), Qwen2.5-7B-Instruct (Yang et al., 2024), and DeepSeek-V4-Pro (DeepSeek-AI, 2026) (accessed via API). For each benchmark, demonstrations are retrieved from the training set. To induce task-specific reasoning operations, we use Seed-OSS-36B-Instruct (ByteDance Seed Team, 2025) on the training set of each benchmark. Using a separate LLM keeps operation construction independent of the evaluated target LLMs and ensures a fair comparison under the same operation space. For semantic matching, we use bertbase-uncased (Devlin et al., 2019) to encode reasoning operations. All experimental results are averaged over five runs. The number of selected examples for all baselines is based on the settings in PSL (Ma et al., 2025) for different benchmarks. Detailed configurations can be found in Appendix D.

<table><tr><td>Method</td><td>GSM8K</td><td>SVAMP</td><td>CMSQA</td><td>StrQA</td><td>Avg</td></tr><tr><td colspan="6">Llama3-8B-Instruct</td></tr><tr><td>Random</td><td>80.91</td><td>85.40</td><td>72.91</td><td>82.27</td><td>80.37</td></tr><tr><td>EPR</td><td>79.45</td><td>84.33</td><td>69.94</td><td>83.99</td><td>79.43</td></tr><tr><td>BM25</td><td>79.88</td><td>85.53</td><td>69.26</td><td>83.49</td><td>79.54</td></tr><tr><td>TopK-BERT</td><td>80.97</td><td>85.00</td><td>71.25</td><td>84.13</td><td>80.34</td></tr><tr><td>DPP-BERT</td><td>79.65</td><td>84.20</td><td>70.81</td><td>82.91</td><td>79.39</td></tr><tr><td>PSL</td><td>82.03</td><td>84.67</td><td>72.89</td><td>84.28</td><td>80.97</td></tr><tr><td>LMS3</td><td>82.65</td><td>85.40</td><td>73.89</td><td>84.89</td><td>81.71</td></tr><tr><td>SALA</td><td>82.59</td><td>87.13</td><td>74.07</td><td>87.25</td><td>82.76</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>Random</td><td>89.25</td><td>82.73</td><td>71.84</td><td>83.29</td><td>81.78</td></tr><tr><td>EPR</td><td>87.95</td><td>88.00</td><td>74.12</td><td>81.95</td><td>83.01</td></tr><tr><td>BM25</td><td>87.08</td><td>86.53</td><td>73.84</td><td>81.74</td><td>82.28</td></tr><tr><td>TopK-BERT</td><td>87.26</td><td>85.33</td><td>75.18</td><td>82.97</td><td>82.69</td></tr><tr><td>DPP-BERT</td><td>88.07</td><td>83.86</td><td>73.59</td><td>83.05</td><td>82.14</td></tr><tr><td>PSL</td><td>89.76</td><td>89.33</td><td>73.05</td><td>80.20</td><td>83.09</td></tr><tr><td>LMS3</td><td>91.52</td><td>83.40</td><td>73.23</td><td>81.98</td><td>82.53</td></tr><tr><td>SALA</td><td>91.16</td><td>91.13</td><td>74.73</td><td>83.90</td><td>85.23</td></tr><tr><td colspan="6">DeepSeek-V4-Pro</td></tr><tr><td>Random</td><td>94.54</td><td>90.33</td><td>74.45</td><td>90.25</td><td>87.39</td></tr><tr><td>BM25</td><td>93.87</td><td>92.73</td><td>76.59</td><td>91.59</td><td>88.70</td></tr><tr><td>TopK-BERT</td><td>96.94</td><td>92.20</td><td>77.61</td><td>92.37</td><td>89.78</td></tr><tr><td>DPP-BERT</td><td>95.45</td><td>93.00</td><td>75.67</td><td>92.58</td><td>89.18</td></tr><tr><td>PSL</td><td>96.48</td><td>91.86</td><td>76.71</td><td>93.68</td><td>89.68</td></tr><tr><td>SALA</td><td>97.12</td><td>94.67</td><td>79.28</td><td>94.61</td><td>91.42</td></tr></table>

Table 1: Main results across three LLMs. EPR and LMS3 are unavailable on DeepSeek-V4-Pro (API-only access). CMSQA denotes CommonsenseQA and StrQA denotes StrategyQA. Bold indicates the best result per model–benchmark.

## 4.2 Main Results

Table 1 summarizes the performance of SALA and the baseline methods. On Llama3-8B-Instruct, SALA achieves the best overall average and the top results on SVAMP, CommonsenseQA, and StrategyQA, while remaining competitive on GSM8K. On Qwen2.5-7B-Instruct, SALA again yields the best average, with the strongest results on SVAMP and StrategyQA. On DeepSeek-V4-Pro (DeepSeek-AI, 2026), EPR and LMS3 are omitted as they require model training or internal hidden states inaccessible through API calls. SALA achieves the best results across all four benchmarks with an average of 91.42%. These results show that SALA consistently improves ICL performance across models of different scales and architectures.

<table><tr><td>Variant</td><td>GSM8K</td><td>SVAMP</td><td>CMSQA</td><td>StrQA</td><td>Avg</td></tr><tr><td colspan="6">Llama3-8B-Instruct</td></tr><tr><td>w/o DTW+OPs</td><td>81.65</td><td>85.00</td><td>72.40</td><td>83.41</td><td>80.62</td></tr><tr><td>w/o OPs</td><td>82.20</td><td>87.06</td><td>73.73</td><td>86.20</td><td>82.30</td></tr><tr><td>w/o DTW</td><td>82.61</td><td>86.87</td><td>74.01</td><td>85.09</td><td>82.15</td></tr><tr><td>Full</td><td>82.59</td><td>87.13</td><td>74.07</td><td>87.25</td><td>82.76</td></tr><tr><td colspan="6">Qwen2.5-7B-Instruct</td></tr><tr><td>w/o DTW+OPs</td><td>88.63</td><td>86.67</td><td>73.55</td><td>80.49</td><td>82.34</td></tr><tr><td>w/o OPs</td><td>90.54</td><td>87.73</td><td>74.22</td><td>82.27</td><td>83.69</td></tr><tr><td>w/o DTW</td><td>89.20</td><td>90.20</td><td>73.02</td><td>81.16</td><td>83.40</td></tr><tr><td>Full</td><td>91.16</td><td>91.13</td><td>74.73</td><td>83.90</td><td>85.23</td></tr></table>

Table 2: Ablation study results. “w/o DTW” uses exact prefix matching similar to PSL; “w/o OPs” uses only the 13 predefined QDMR operations. CMSQA denotes CommonsenseQA and StrQA denotes StrategyQA.

![](images/7bf4b222a9b7c00430928b2ba867da0434729702cc6ed31881f3322a5677c185.jpg)  
Figure 3: The percentage of samples using task-specific reasoning operations in the training and test sets.

## 4.3 Ablation Study

To further demonstrate the effectiveness of the SALA framework, we conduct an ablation study on its key modules.

As shown in Table 2, removing either component leads to a consistent performance drop on both models, indicating that both task-adaptive operation construction and DTW-based semantic alignment are necessary for SALA. When the induced operations are removed, SALA falls back to the 13 predefined QDMR operations. This reduces the average accuracy by 0.5 points on Llama3-8B-Instruct and 1.5 points on Qwen2.5-7B-Instruct, showing that the additional operations help represent task-specific reasoning patterns that are not sufficiently covered by the predefined operation set. When DTW is removed and exact prefix matching is used instead, the average accuracy drops by 0.6 points on Llama3-8B-Instruct and 1.8 points on Qwen2.5-7B-Instruct. This suggests that semantic sequence alignment is important for handling semantically related operations and different decomposition granularities. Removing both components causes the largest degradation, with average drops of 2.1 and 2.9 points on the two models, respectively.

![](images/01de2abd3bf98ef2ab0e57e65043985c8c1dfc15ccf2c64286d121ea5e3731cf.jpg)

![](images/f92f4a8fadd92eb961acdde2dfe83cebd3480cfa8178802f039f4b0ed1057f60.jpg)  
Figure 4: Average reasoning operation sequence length with and without task-specific operations across four benchmarks. (a) Average reasoning sequence length of training set samples; (b) Average reasoning sequence length of test set samples.

## 4.4 SALA Analysis

## 4.4.1 Necessity of Inducing Task-Specific Operations

To examine the role of task-specific reasoning operations, we measure the proportion of samples whose reasoning sequences require such operations in both the training and test sets. As shown in Figure 3, this proportion exceeds 68% across all datasets. In GSM8K, it is above 98% for both the training and test sets, whereas for the remaining datasets it generally falls between 70% and 80%. These findings suggest that task-specific reasoning operations are widely involved in reasoning decomposition across diverse tasks. Some representative induced operations are listed in Appendix E.

## 4.4.2 Qualitative Case Analysis

To further understand how each module contributes to demonstration selection, we present four case studies. Cases 1–2 isolate the effect of task-specific operations (Module 1), while Cases 3–4 isolate DTW semantic alignment (Module 2).

Cases 1–2: Task-Specific Operations Condense Reasoning Sequences Figure 4 shows that adding task-specific operations consistently reduces average reasoning sequence length across all benchmarks. This compaction is critical for accurate demonstration matching. In Case 1 (Table 7 in Appendix F.1), with Modulo, the query sequence shrinks from 9 to 7 operations, and the matched demonstration is another remainder problem; without Modulo, the inflated division-multiplicationsubtraction sequence attracts a proportion problem with an incompatible reasoning pattern. In Case 2 (Table 8 in Appendix F.1), with Algebra, the sequence shrinks from 13 to 7 operations; the compact Define–Algebra pattern matches another equation-solving demonstration, while the 13-step QDMR-only decomposition attracts a multi-item subtraction problem.

Cases 3–4: DTW Overcomes Limitations of Prefix Matching Cases 3–4 use only the 13 QDMR operations to isolate Module 2. Case 3 (Table 9 in Appendix F.2) shows that DTW selects a percentagemarkup demonstration that matches the query’s reasoning pattern despite different third operations, while the longest valid prefix demonstration shares 6 of 7 operations but encodes a subtract-discount pattern incompatible with the query’s add-insurance requirement. Case 4 (Table 10 in Appendix F.2) shows that the longest valid prefix demonstration covers 5 of 7 operations but lacks any percentage step, causing the model to miss the tax computation entirely; DTW instead selects a semantically aligned percentage demonstration. Full case details with query examples and demonstration sequences are provided in Appendix F.

## 5 Conclusion and Future Work

We propose SALA, a reasoning-oriented demonstration selection framework that combines taskadaptive operation construction with semantic DTWbased alignment. SALA represents problem-solving logic as explicit reasoning-operation sequences, while allowing flexible matching in semantic space. Experiments on four reasoning benchmarks and three LLMs show that SALA consistently outperforms strong demonstration selection baselines. In future work, we plan to extend SALA to broader reasoning domains and explore richer reasoning structures beyond linear operation sequences.

## Limitations

SALA uses LLMs to induce task-adaptive operations and parse questions into reasoning-operation sequences. While this reduces the need for manual operation engineering, the induced operation set and parsed sequences may vary with the induction model and prompting strategy. In this work, we use a separate LLM and fixed prompts to keep the construction process consistent across experiments.

SALA currently represents problem-solving logic as a linear sequence of operations. This representation works well for the reasoning benchmarks studied in this paper, but more complex tasks may benefit from richer structures, such as hierarchical or graph-based reasoning representations. We leave the extension of SALA to these settings for future work.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China (62306344, 62276279), the Guangdong Basic and Applied Basic Research Foundation (2024A1515010253, 2026A1515011800, 2024B1515020032), the Open Research Fund of the State Key Laboratory of Blockchain and Data Security, Zhejiang University (Grant No. A2537), and the Guangdong S&T Programme Key-Area Research and Development Program of Guangdong Province (2026B0101100004).

Generative AI tools were used during the preparation and revision of this manuscript to polish the language, revise selected passages, check the consistency of terminology and reported numerical values across sections, and assist with LAT<sub>E</sub>X formatting. All AI-assisted changes incorporated into the manuscript were reviewed and verified by the authors, who take full responsibility for the paper’s content.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Shengnan An, Bo Zhou, Zeqi Lin, Qiang Fu, Bei Chen, Nanning Zheng, Weizhu Chen, and Jian-Guang Lou. 2023. Skill-based few-shot selection for in-context learning. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13472–13492, Singapore. Association for Computational Linguistics.

Aram Bahrini, Mohammadsadra Khamoshifar, Hossein Abbasimehr, Robert J. Riggs, Maryam Esmaeili, Rastin Mastali Majdabadkohne, and Morteza Pasehvar. 2023. Chatgpt: Applications, opportunities, and threats. In 2023 Systems and Information Engineering Design Symposium (SIEDS), pages 274–279.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, and 1 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Satwik Bhattamishra, Arkil Patel, Phil Blunsom, and Varun Kanade. 2024. Understanding in-context learning in transformers and LLMs by learning to learn discrete functions. In The Twelfth International Conference on Learning Representations.

ByteDance Seed Team. 2025. Seed-oss opensource models. https://github.com/ ByteDance-Seed/seed-oss.

Laming Chen, Guoxin Zhang, and Eric Zhou. 2018. Fast greedy map inference for determinantal point process to improve recommendation diversity. In Advances in Neural Information Processing Systems, volume 31. Curran Associates, Inc.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, and 1 others. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

DeepSeek-AI. 2026. Deepseek-v4: Towards highly efficient million-token context intelligence.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu,

Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A survey on in-context learning. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, pages 1107–1128, Miami, Florida, USA. Association for Computational Linguistics.

Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristotle use a laptop? a question answering benchmark with implicit reasoning strategies. Transactions of the Association for Computational Linguistics, 9:346– 361.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Xiaonan Li, Kai Lv, Hang Yan, Tianyang Lin, Wei Zhu, Yuan Ni, Guotong Xie, Xiaoling Wang, and Xipeng Qiu. 2023. Unified demonstration retriever for incontext learning. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 4644–4668, Toronto, Canada. Association for Computational Linguistics.

Xiaonan Li and Xipeng Qiu. 2023. Finding support examples for in-context learning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2023, pages 6219–6235, Singapore. Association for Computational Linguistics.

Yukang Lin, Bingchen Zhong, Shuoran Jiang, Joanna Siebert, and Qingcai Chen. 2025. Reasoning graph enhanced exemplars retrieval for in-context learning. In Proceedings of the 31st International Conference on Computational Linguistics, pages 9737– 9759, Abu Dhabi, UAE. Association for Computational Linguistics.

Jiayu Liu, Zhenya Huang, Chaokun Wang, Xunpeng Huang, ChengXiang Zhai, and Enhong Chen. 2025. What makes in-context learning effective for mathematical reasoning. In Forty-second International Conference on Machine Learning.

Man Luo, Xin Xu, Zhuyun Dai, Panupong Pasupat, Mehran Kazemi, Chitta Baral, Vaiva Imbrasaite, and Vincent Y Zhao. 2023. Dr.ICL: Demonstration-Retrieved In-Context Learning. In NeurIPS 2023 Workshops: R0-FoMo.

Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, and Mehran Kazemi. 2024. In-context learning with retrieved demonstrations for language models: A survey. Transactions on Machine Learning Research. Survey Certification.

Xuetao Ma, Wenbin Jiang, and Hua Huang. 2025. Problem-solving logic guided curriculum in-context learning for LLMs complex reasoning. In Findings of the Associationfor Computational Linguistics: ACL 2025, pages 8394–8412, Vienna, Austria. Association for Computational Linguistics.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. 2022. Training language models to follow instructions with human feedback. In Proceedings ofthe 36th International Conference on Neural Information Processing Systems, NIPS ’22, Red Hook, NY, USA. Curran Associates Inc.

Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. 2025. Reasoningbank: Scaling agent self-evolving with reasoning memory. CoRR, abs/2509.25140.

Arkil Patel, Satwik Bhattamishra, and Navin Goyal. 2021. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, pages 2080–2094, Online. Association for Computational Linguistics.

Kiran Purohit, Venktesh V, Sourangshu Bhattacharya, and Avishek Anand. 2025. Sample efficient demonstration selection for in-context learning. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 49959–49982. PMLR.

Chengwei Qin, Aston Zhang, Chen Chen, Anirudh Dagar, and Wenming Ye. 2024. In-context learning with iterative demonstration selection. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 7441–7455, Miami, Florida, USA. Association for Computational Linguistics.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Found. Trends Inf. Retr., 3(4):333–389.

Ohad Rubin, Jonathan Herzig, and Jonathan Berant. 2022. Learning to retrieve prompts for in-context learning. In Proceedings of the 2022 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 2655–2671, Seattle, United States. Association for Computational Linguistics.

H. Sakoe and S. Chiba. 1978. Dynamic programming algorithm optimization for spoken word recognition. IEEE Transactions on Acoustics, Speech, and Signal Processing, 26(1):43–49.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for

Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Xubin Wang, Jianfei Wu, Yichen Yuan, Deyu Cai, Mingzhe Li, and Weijia Jia. 2025. Demonstration selection for in-context learning via reinforcement learning. In Proceedings of the 42nd International Conference on Machine Learning, ICML’25. JMLR.org.

Noam Wies, Yoav Levine, and Amnon Shashua. 2023. The learnability of in-context learning. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, and Jonathan Berant. 2020. Break it down: A question understanding benchmark. Transactions ofthe Associationfor Computational Linguistics, 8:183–198.

Zifan Xu, Haozhu Wang, Dmitriy Bespalov, Xian Wu, Peter Stone, and Yanjun Qi. 2024. LaRS: Latent reasoning skills for chain-of-thought reasoning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 3624–3643, Miami, Florida, USA. Association for Computational Linguistics.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, and 22 others. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Zhao Yang, Yuanzhe Zhang, Dianbo Sui, Cao Liu, Jun Zhao, and Kang Liu. 2023. Representative demonstration selection for in-context learning with twostage determinantal point process. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 5443–5456, Singapore. Association for Computational Linguistics.

Jiacheng Ye, Zhiyong Wu, Jiangtao Feng, Tao Yu, and Lingpeng Kong. 2023a. Compositional exemplars for in-context learning. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 39818–39833. PMLR.

Junjie Ye, Xuanting Chen, Nuo Xu, Can Zu, Zekai Shao, Shichun Liu, Yuhan Cui, Zeyang Zhou, Chao Gong, Yang Shen, Jie Zhou, Siming Chen, Tao Gui,

Qi Zhang, and Xuanjing Huang. 2023b. A comprehensive capability analysis of gpt-3 and gpt-3.5 series models. Preprint, arXiv:2303.10420.

Jianfei Zhang, Bei Li, Jun Bai, Rumei Li, Yanmeng Wang, Chenghua Lin, and Wenge Rong. 2025a. Selecting demonstrations for many-shot in-context learning via gradient matching. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 11686–11704, Vienna, Austria. Association for Computational Linguistics.

Yufeng Zhang, Xuepeng Wang, Lingxiang Wu, and Jinqiao Wang. 2025b. Enhancing chain of thought prompting in large language models via reasoning patterns. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25985– 25993.

Zheng Zhang, Shaocheng Lan, Lei Song, Jiang Bian, Yexin Li, and Kan Ren. 2025c. Learning to select in-context demonstration preferred by large language model. In Findings of the Association for Computational Linguistics: ACL 2025, pages 11345–11360, Vienna, Austria. Association for Computational Linguistics.

## A Dataset Statistics

Table 3 summarizes the training and test splits of the four benchmark datasets used in this paper.

<table><tr><td>Dataset</td><td>Train</td><td>Test</td></tr><tr><td>GSM8K</td><td>7,473</td><td>1,319</td></tr><tr><td>SVAMP</td><td>700</td><td>300</td></tr><tr><td>CommonsenseQA</td><td>9,741</td><td>1,140</td></tr><tr><td>StrategyQA</td><td>1,603</td><td>687</td></tr></table>

Table 3: Training and test split statistics for the datasets used in our experiments.

## B Predefined QDMR Reasoning Operations

SALA extends the 13 predefined QDMR reasoning operations (Wolfson et al., 2020) with task-specific operations. Table 4 lists all 13 predefined operations with their core functions, example questions, and example sequences.

## C Prompt Templates

We provide the full prompt templates used in each step of the SALA pipeline. All prompts were used in English during our experiments.

## C.1 Operation Induction Prompt

This prompt instructs the LLM to induce reasoning operations not covered by the predefined 13.

<table><tr><td>Operation</td><td>Core Function</td><td>Example Question</td><td>Example Sequence</td></tr><tr><td>Select</td><td>Selects a specific entity or set</td><td>overall?</td><td>How many touchdowns were scored SELECT[&#x27;touchdowns&#x27;]; AGGREGATE[&#x27;number&#x27;, &#x27;#1&#x27;]</td></tr><tr><td>Filter</td><td>Selects a subset matching conditions</td><td>San Diego please.</td><td>I would like a flight from Toronto to SELECT[&#x27;flights&#x27;]; FILTER[&#x27;#1&#x27;, &#x27;from Toronto&#x27;]; FILTER[&#x27;#2&#x27;, &#x27;to San Diego&#x27;]</td></tr><tr><td>Arithmetic</td><td>Basic arithmetic (+, — , ×, ÷) on numeric attributes</td><td>How many more red objects are there than blue objects?</td><td>SELECT[&#x27;red objects&#x27;]; SELECT[&#x27;blue objects&#x27;]; PROJECT[&#x27;number&#x27;, &#x27;#1&#x27;]; PROJECT[&#x27;number&#x27;, &#x27;#2&#x27;]; ARITHMETIC[&#x27;difference&#x27;, &#x27;#3&#x27;, &#x27;#4&#x27;]</td></tr><tr><td></td><td>a threshold</td><td>Comparative Selects subset greater/less than Who are the authors with more than 500 papers?</td><td>SELECT[&#x27;authors&#x27;]; PROJECT[&#x27;papers&#x27;, &#x27;#1&#x27;]; GROUP[&#x27;number&#x27;, &#x27;#2&#x27;, &#x27;#1&#x27;]; COMPARATIVE[&#x27;#1&#x27;, &#x27;#3&#x27;, &#x27;more than 500&#x27;]</td></tr><tr><td>Superlative</td><td>Selects the entity with the extreme value</td><td>What is the keyword contained by the most papers?</td><td>SELECT[&#x27;papers&#x27;]; PROJECT[&#x27;keywords&#x27;, &#x27;#1&#x27;]; GROUP[&#x27;number&#x27;, &#x27;#1&#x27;, &#x27;#2&#x27;]; SUPERLATIVE[&#x27;#2&#x27;, &#x27;#3&#x27;, &#x27;highest&#x27;]</td></tr><tr><td>Aggregate</td><td>Computes mathematical properties (count, avg) of a set</td><td>How many states border Colorado?</td><td>SELECT[&#x27;Colorado&#x27;]; PROJECT[&#x27;border states&#x27;, &#x27;#1&#x27;]; AGGREGATE[&#x27;number&#x27;, &#x27;#2&#x27;]</td></tr><tr><td>Union</td><td>Merges two sets</td><td>Tell me the president and vice-president.</td><td>SELECT[&#x27;president&#x27;]; SELECT[&#x27;vice-president&#x27;]; UNION[&#x27;#1&#x27;, &#x27;#2&#x27;]</td></tr><tr><td>Intersection</td><td>Takes the intersection of two sets</td><td>Show parties with representatives in both New York and Pennsylvania.</td><td>SELECT[&#x27;representatives&#x27;]; FILTER[&#x27;#1&#x27;, &#x27;in New York&#x27;]; FILTER[&#x27;#1&#x27;, &#x27;in Pennsylvania&#x27;]; INTERSECTION[&#x27;parties&#x27;, &#x27;#2&#x27;, &#x27;#3&#x27;]</td></tr><tr><td>Project</td><td>Obtains a specific attribute of an entity</td><td>Who is the head coach of the Los Angeles Lakers?</td><td>SELECT[&#x27;Los Angeles Lakers&#x27;]; PROJECT[&#x27;head coach&#x27;, &#x27;#1&#x27;]</td></tr><tr><td>Sort</td><td>Arranges elements by a specified rule</td><td>Find student addresses sorted by monthly rental.</td><td>SELECT[&#x27;students&#x27;]; PROJECT[&#x27;addresses&#x27;, &#x27;#1&#x27;]; PROJECT[&#x27;monthly rental&#x27;, &#x27;#2&#x27;]; SORT[&#x27;#2&#x27;, &#x27;#3&#x27;]</td></tr><tr><td>Group</td><td>Computes properties per group element</td><td>How many female students per club?</td><td>SELECT[&#x27;clubs&#x27;]; FILTER[&#x27;#1&#x27;, &#x27;female students&#x27;]; GROUP[&#x27;number&#x27;, &#x27;#2&#x27;, &#x27;#1&#x27;]</td></tr><tr><td>Discard</td><td>Selects subset not satisfying a condition</td><td>Find professors not playing Canoeing.</td><td>SELECT[&#x27;professors&#x27;]; FILTER[&#x27;#1&#x27;, &#x27;playing Canoeing&#x27;]; DISCARD[&#x27;#1&#x27;, &#x27;#2&#x27;]</td></tr><tr><td>Boolean</td><td>condition</td><td>Determines if entity satisfies a Were Scott Derrickson and Ed Wood of the same nationality?</td><td>SELECT[&#x27;Scott Derrickson&#x27;]; SELECT[&#x27;Ed Wood&#x27;]; PROJECT[&#x27;nationality&#x27;, &#x27;#1&#x27;]; PROJECT[&#x27;nationality&#x27;, &#x27;#2&#x27;]; BOOLEAN[&#x27;#3&#x27;, &#x27;the same as&#x27;, &#x27;#4&#x27;]</td></tr></table>

Table 4: The 13 predefined QDMR reasoning operations used as the base inventory in SALA.

Task: Given an input question and the 13 existing reasoning operations (listed below), induce any new reasoning operations needed to solve the problem.

## Existing reasoning operations: {operations}

Requirements: Supplement with new operations. Identify any new reasoning operations from the input question that are not covered by the existing ones. Describe the core functionality, provide an example question, example decomposition, and corresponding reasoning operation sequence in the same format as above. Do not output any extraneous content. If you determine that the existing operations are sufficient to solve the input question, simply output “No new operation needed.”

Input question: {problem}

## C.2 Sequence Analysis Prompt

This prompt decomposes a question into an operation sequence using the extended operation set.

Decompose the input question into a reasoning operation sequence using the given reasoning op erations.

## Given reasoning operations: {operations}

Example Question: what flights are available tomorrow from denver to philadelphia? Reasoning operation sequence: [’SELECT[’flights’]’, ’FIL-TER[’#1’, ’from denver’]’, ’FILTER[’#2’, ’to philadelphia’]’, ’FILTER[’#3’, ’if available’]’] Operation name list: [’select’, ’filter’, ’filter’, ’filter’]

Follow the example format strictly and provide the reasoning operation sequence and corresponding operation name list for the input question below.

Input question: {question}

## C.3 LLM Deduplication Prompt

This prompt checks whether a newly induced operation overlaps with existing ones (Stage 2 of deduplication).

Task: Compare whether the functionality of the new reasoning operation overlaps with any existing reasoning operation, and determine whether the new operation is a composite operation.

Existing reasoning operations: {old\_batch}

Requirements: If the new operation’s functionality overlaps with any existing operation, or if the new operation can be decomposed into existing operations (i.e., it is composite), output ‘<1>’. Otherwise, output ‘<0>’.

New reasoning operation: {new}

## C.4 ICL Prompt Template

This template formats the retrieved demonstrations and query for the final ICL inference.

Before answering my question, there are some examples you can use for reference:

<example>{examples} </example>

Here is my question: {question}

Please answer it:

## C.5 Evaluation Prompts

We use three dataset-specific evaluation prompts to compare candidate answers against ground-truth answers.

## Numeric Evaluation (used for GSM8K, SV-AMP):

Your task is to compare the Candidate Answer with the Correct Answer and determine whether they are consistent.

Question: {question}

Candidate Answer: {candidate}

Correct Answer: {correct}

Criteria: - Numerical values are consistent if they represent the same quantity despite different formats (e.g., 0.88 is consistent with 88%). Return <1>. - Numerical values are also considered consistent if rounding leads to the same result (e.g., if the correct answer is 8 and the candidate is 7.96, return <1>).

Output: Compare whether the answers are consistent. If consistent, output <1>; otherwise, output <0>.

## Text Evaluation (used for CommonsenseQA):

Your task is to compare the Candidate Answer with the Correct Answer and determine whether they are consistent.

Question: {question}

Candidate Answer: {candidate}

Correct Answer: {correct}

Criteria: - Evaluate whether the Candidate Answer and Correct Answer share the same meaning in the context of the given question. If they use different words but convey the same meaning, consider them consistent.

Output: If consistent, output <1>; otherwise, output <0>.

## Boolean Evaluation (used for StrategyQA):

Your task is to compare the Candidate Answer with the Correct Answer and determine whether they are consistent.

Question: {question}

Candidate Answer: {candidate}

Correct Answer: {correct}

Criteria: - The Candidate Answer may be verbose while the Correct Answer is very concise. If the final conclusion of the Candidate Answer matches the boolean value (True or False) of the Correct Answer, consider them consistent.

Output: If consistent, output <1>; otherwise, output <0>.

## D Hyperparameter Settings

Table 5 lists the key hyperparameters used in our experiments.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Operation Induction LLM for induction</td><td>Seed-OSS-36B-Instruct</td></tr><tr><td>&amp; deduplication</td><td></td></tr><tr><td>Temperature</td><td>0.01</td></tr><tr><td>Max tokens</td><td>4,096</td></tr><tr><td>Semantic Embedding Embedding model</td><td>bert-base-uncased</td></tr><tr><td>Embedding</td><td>768</td></tr><tr><td>dimension Max input length</td><td>1,024 tokens</td></tr><tr><td>DTW Retrieval</td><td></td></tr><tr><td>Retrieval top-k</td><td>8</td></tr><tr><td></td><td></td></tr><tr><td>Inference &amp; Evaluation Target LLMs</td><td>Llama3-8B-Instruct,</td></tr><tr><td></td><td></td></tr><tr><td></td><td>Qwen2.5-7B-Instruct, DeepSeek-V4-Pro</td></tr><tr><td>Temperature</td><td>0.01</td></tr><tr><td>Max tokens</td><td></td></tr><tr><td></td><td>4,096</td></tr><tr><td>Evaluation LLM</td><td>GPT-4</td></tr></table>

Table 5: Key hyperparameter settings for SALA.

## E Induced Reasoning Operation Examples across Downstream Tasks

Table 6 presents representative reasoning operations induced by SALA on each downstream benchmark through the adaptive operation set construction process.

## F Case Study — Detailed Tables

This appendix provides the full detailed tables for the four case studies summarized in Section 4.4 (Table 11). Cases 1–2 isolate Module 1 (task-specific operations) and Cases 3–4 isolate Module 2 (DTW semantic alignment). All experiments use Llama3- 8B-Instruct.

## F.1 Effect of Task-Specific Operations on Reasoning Sequence Compactness

Module 1 induces task-specific operations that condense reasoning sequences which would otherwise require long chains of QDMR operations. This compactness directly affects which demonstrations are retrieved and whether the model produces the correct answer.

## F.1.1 Case 1: Modulo Condenses Remainder Computation

## F.1.2 Case 2: Algebra Condenses Variable Solving

## F.2 DTW Semantic Alignment vs. Prefix Subsequence Matching

To isolate Module 2, both methods in this subsection use only the 13 QDMR operations.

<table><tr><td>Operation</td><td>Core Function</td><td>Example Question</td><td>Example Sequence</td></tr><tr><td>Define</td><td>Declares an unknown variable representing a target quantity to be solved</td><td>Randy spent $10 on lunch. He then spent a quarter of his remaining money on an ice cream cone costing $5. What was his initial money?</td><td>DEFINE[&#x27;M&#x27;, &#x27;initial money&#x27;]; ARITHMETIC[&#x27;subtract&#x27;, &#x27;#1&#x27;, &#x27;10&#x27;]; ARITHMETIC[&#x27;quarter&#x27;, &#x27;#2&#x27;]; ALGEBRA[&#x27;#3&#x27;, &#x27;=&#x27;, &#x27;5&#x27;]</td></tr><tr><td>Algebra</td><td>Defines unknown variables, builds relations with known values, and solves via equations</td><td>Yvonne brings a box of chocolates to school. Half have nuts and half do not. The students eat 80% of the ones with nuts and half of the ones without nuts. If there are 28 left, how many were originally in the</td><td>DEFINE[&#x27;total chocolates&#x27;, &#x27;#x&#x27;]; ARITHMETIC[&#x27;division&#x27;, &#x27;#x&#x27;, 2]; ARITHMETIC[&#x27;division&#x27;, &#x27;#x&#x27;, 2]; ARITHMETIC[&#x27;multiply&#x27;, &#x27;#2&#x27;, &#x27;0.2&#x27;]; ARITHMETIC[&#x27;multiply&#x27;, &#x27;#3&#x27;, &#x27;0.5&#x27;]; ARITHMETIC[&#x27;addition&#x27;, &#x27;#4&#x27;, &#x27;#5&#x27;]; ALGEBRA[&#x27;#x&#x27;, &#x27;#6&#x27;, &#x27;=&#x27;, &#x27;28&#x27;]</td></tr><tr><td>Modulo</td><td>Computes the remainder after dividing one number by another</td><td>Emma&#x27;s bank account has $100. Each day of the week she spends $8. At the end of the week, she withdraws as many $5 bills as possible. How many dollars</td><td>SELECT[&#x27;initial amount&#x27;]; SELECT[&#x27;daily spending&#x27;]; SELECT[&#x27;days in a week&#x27;]; ARITHMETIC[&#x27;product&#x27;, &#x27;#2&#x27;, &#x27;#3&#x27;]; ARITHMETIC[&#x27;difference&#x27;, &#x27;#1&#x27;, &#x27;#4&#x27;]; MODULO[&#x27;#5&#x27;, &#x27;5&#x27;]</td></tr><tr><td colspan="4">remain? SVAMP</td></tr><tr><td>Constant</td><td>Obtains a fixed numeric value given directly or implicitly in the problem</td><td>Mary is baking a cake. The recipe calls for 5 cups of sugar and 14 cups of flour. She already put in 11 cups of flour. How many more cups of sugar than cups of flour does she &#x27;#7&#x27;, &#x27;#8&#x27;]</td><td>SELECT[&#x27;recipe&#x27;]; PROJECT[&#x27;sugar&#x27;, &#x27;#1&#x27;]; PROJECT[&#x27;flour&#x27;, &#x27;#1&#x27;]; SELECT[&#x27;flour&#x27;]; PROJECT[&#x27;already added&#x27;, &#x27;#1&#x27;]; CONSTANT[&#x27;0&#x27;]; ARITHMETIC[&#x27;subtract&#x27;, &#x27;#2&#x27;, &#x27;#6&#x27;]; ARITHMETIC[&#x27;subtract&#x27;, &#x27;#3&#x27;, &#x27;#5&#x27;]; ARITHMETIC[&#x27;subtract&#x27;,</td></tr><tr><td>Literal</td><td>Retrieves a specific numeric or If 479 students suggested adding attribute value directly stated in the problem</td><td>mashed potatoes while 489 suggested adding bacon, how many more students suggested bacon than</td><td>LITERAL[&#x27;mashed potato&#x27;, &#x27;479&#x27;]; LITERAL[&#x27;bacon&#x27;, &#x27;489&#x27;]; ARITHMETIC[&#x27;difference&#x27;, &#x27;#1&#x27;, &#x27;#2&#x27;]</td></tr><tr><td colspan="4">Retain Preserves the original attribute Ed had 12 more marbles than Doug. value of an entity when no</td></tr><tr><td></td><td>change to that attribute is described</td><td>17, how many marbles does Doug have now? StrategyQA</td><td>ARITHMETIC[&#x27;subtract&#x27;, &#x27;#3&#x27;, &#x27;#4&#x27;]; RETAIN[&#x27;Dougs marbles&#x27;, &#x27;#5&#x27;]</td></tr><tr><td colspan="4">EntityLinking Resolves a referring expression Was historical Dracula from a town ENTITYLINKING[&quot;historical Dracula&quot;]; PROJECT[&quot;birthplace&quot;, to its corresponding real-world in Bucharest? &quot;#1&quot;]; BOOLEAN[&quot;#2&quot;, &quot;in&quot;, &quot;Bucharest&quot;]</td></tr><tr><td>Relate</td><td>entity Obtains the value of a specific relationship between two or</td><td>What is the distance between</td><td>SELECT[&#x27;Dusseldorf&#x27;]; SELECT[&#x27;Stonehenge&#x27;];</td></tr><tr><td>Deductive</td><td>more entities Derives a conclusion through logical reasoning from</td><td>Dusseldorf and Stonehenge? Aristotle died in 322 BC. The Model Parliament was held in 1295. The House of Lords grew out of the</td><td>RELATE[&#x27;distance&#x27;, &#x27;#1&#x27;, &#x27;#2&#x27;] SELECT[&#x27;Aristotle&#x27;]; PROJECT[&#x27;death year&#x27;, &#x27;#1&#x27;]; SELECT[&#x27;Model Parliament&#x27;]; PROJECT[&#x27;held year&#x27;, &#x27;#3&#x27;]; SELECT[&#x27;House of Lords&#x27;]; PROJECT[&#x27;origin&#x27;, &#x27;#5&#x27;];</td></tr><tr><td colspan="4">Model Parliament. Was Aristotle a member of the House of Lords?</td></tr><tr><td>Contrast</td><td>Obtains the opposite or contrasting concept of a given concept or attribute</td><td>The troublemaker had been hoping handed down was quite what?</td><td>SELECT[&#x27;soft punishment&#x27;]; CONTRAST[&#x27;opposite&#x27;, &#x27;#1&#x27;]; for a soft punishment, but the ruling SELECT[&#x27;the ruling&#x27;]; PROJECT[&#x27;#2&#x27;, &#x27;#3&#x27;]</td></tr><tr><td>Similar</td><td>Finds entities or collections that are similar in type or features to a specified entity</td><td>A hurricane is similar to what other SELECT[&#x27;hurricane&#x27;]; SIMILAR[&#x27;wind events&#x27;, &#x27;#1&#x27;] wind event?</td><td></td></tr><tr><td>Source</td><td>Locates where or how a specific attribute of an entity can be found or accessed</td><td>Marmot was open, I might look where?</td><td>If I wanted to find out the hours that SELECT[&#x27;Marmot&#x27;]; SOURCE[&#x27;open hours&#x27;, &#x27;#1&#x27;]</td></tr></table>

Table 6: Representative reasoning operations induced by SALA on four downstream benchmarks.

## F.2.1 Case 3: Granularity Mismatch — Best Structural Match Excluded by Prefix Constraint

F.2.2 Case 4: Incomplete Reasoning — Longest Valid Prefix Omits Percentage Computation

Table 11 summarizes the four cases.

<table><tr><td colspan="2">Query Emma has $100 in her bank account. She spends $8 each day for a full week. At the end of the week, she withdraws as</td></tr><tr><td colspan="2">many $5 bills as possible. How many dollars remain in her account? (with Modulo)</td></tr><tr><td colspan="2">Ground Truth $4 w/ new OPs $4 √ w/o new OPs</td></tr><tr><td colspan="2">$2 × (QDMR only)</td></tr><tr><td colspan="2">With New Operations (Modulo) Query sequence [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC, MODULO]</td></tr><tr><td colspan="2" rowspan="2">(len=7) Matched demo</td></tr><tr><td rowspan="3">&quot;A factory packs 125 toys per day. Each box holds 8 toys. After filling full boxes, how many toys are</td></tr><tr><td>left?&quot; Demo sequence [SELECT, PROJECT, ARITHMETIC, MODULO]</td></tr><tr><td>This is another remainder problem; the model correctly performs the modulo computation.</td></tr><tr><td colspan="2">Query sequence [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC, ARITH- (len=9) METIC, ARITHMETIC]</td></tr><tr><td colspan="2">To simulate modulo, the sequence appends division → multiplication → subtraction. Matched demo “A recipe needs 3 cups of flour per 5 servings. How many cups for 20 servings?&quot; Demo sequence [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC] (len=7) Analysis This is a proportional reasoning problem, not a remainder problem. The model is misled by the similar</td></tr></table>

Table 7: Case 1: With Modulo, the compact sequence attracts another remainder problem. Without it, the inflated division-multiplication-subtraction chain attracts a proportion problem that shares the same surface pattern but has a different reasoning structure.

<table><tr><td colspan="2">Query Yvonne brings a box of chocolates to school. Half have nuts and half do not. The students eat 80% of the ones with nuts</td></tr><tr><td colspan="2">and half of the ones without nuts. If there are 28 chocolates left, how many chocolates were originally in the box? Ground Truth 80</td></tr><tr><td>w/ new OPs 80√ w/o new OPs 90 × With New Operations (Algebra)</td><td>(with Algebra) (QDMR only)</td></tr><tr><td colspan="2">Query sequence [DEFINE, ARITHMETIC, ARITHMETIC, ARITHMETIC, ARITHMETIC, ARITHMETIC, ALGE- (len=7) BRA] Matched demo “Randy spent $10 on lunch and a quarter of the remaining money on an ice cream cone costing $5. What was his initial money?&quot;</td></tr><tr><td colspan="2">Demo sequence [DEFINE, ARITHMETIC, ARITHMETIC, ARITHMETIC, ALGEBRA] (len=5) Analysis Both queries define an unknown variable and solve for it via an equation. The model recovers the algebraic</td></tr><tr><td colspan="2">structure and correctly outputs 80 Without New Operations (QDMR Only)</td></tr><tr><td colspan="2">(len=13) ARITHMETIC, ARITHMETIC, ARITHMETIC, ARITHMETIC, ARITHMETIC] Without DEFINE and ALGEBRA, each chocolate category must be separately selected and projected;</td></tr><tr><td colspan="2" rowspan="2">remaining amounts are computed stepwise. Matched demo</td></tr><tr><td rowspan="3"></td></tr><tr><td>“Danny has 94 guppies, 76 angelfish, 89 tiger sharks, and 58 Oscar fish. If he sells 30, 48, 17, and 24 respectively, how many fish remain?&quot; [SELECT, PROJECT, SELECT, PROJECT, SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC,</td></tr><tr><td>Demo sequence ...] Analysis This is a multi-item subtraction problem whose SELECT–PROJECT repetition is structurally similar to</td></tr></table>

Table 8: Case 2: With Algebra, the compact DEFINE–...–ALGEBRA pattern matches equation-solving demonstrations. QDMR-only decomposition produces a long chain of SELECT–PROJECT pairs that attracts structurally similar but logically unrelated multi-item problems.

<table><tr><td colspan="2">Query</td></tr><tr><td colspan="2">Janet buys a brooch for her daughter. She pays $500 for the material and another $800 for the jeweler to construct it. After that, she pays 10% of that to get it insured. How much did she pay? [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC]</td></tr><tr><td>Query sequence (len=7) Ground Truth</td><td>$1,430</td></tr><tr><td>DTW Prefix</td><td>$1,430 √ $1,170 ×</td></tr><tr><td>DTW-Selected Demonstration</td><td></td></tr><tr><td colspan="2">Demo “John commissions a drawing. A black and white costs $160. Color is 50% more. How much for both?&quot; Demo sequence [SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC]</td></tr><tr><td colspan="2">(len=5)</td></tr><tr><td colspan="2">Reasoning Extract base cost → apply percentage markup → sum. Prefix match Not a valid prefix.</td></tr><tr><td colspan="2">the query&#x27;s prefix. Excluded. Prefix-Selected Demonstration (longest valid prefix = 6)</td></tr><tr><td colspan="2">Demo &quot;A book costs $20 and a notebook costs $5. With a 10% total discount, final price?&quot; Demo sequence [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC] (len=6)</td></tr></table>

Table 9: Case 3: DTW selects a structurally aligned demo (single-item percentage markup) despite being excluded by prefix matching due to a granularity mismatch at position 3. Prefix matching instead selects a two-item discount demo whose entire sequence matches the query’s prefix, but whose final reasoning step (subtract discount) contradicts the query’s requirement (add insurance).
<table><tr><td colspan="2">Query</td></tr><tr><td colspan="2">Alice bought a $50 dress and a $30 bag. The store applies 8% sales tax. What is the total? Query sequence [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC]</td></tr><tr><td>(len=7) Ground Truth</td><td>$86.40</td></tr><tr><td colspan="2">DTW</td></tr><tr><td>Prefix DTW-Selected Demonstration</td><td>$86.40 √ $80 ×</td></tr><tr><td>Demo</td><td>“John commissions a drawing. A black and white costs $160. Color is 50% more. How much for both?&quot;</td></tr><tr><td colspan="2">Demo sequence</td></tr><tr><td>(len=5) Reasoning</td><td>[SELECT, PROJECT, ARITHMETIC, ARITHMETIC, ARITHMETIC] Extract base cost → apply percentage → sum.</td></tr><tr><td>Prefix match</td><td>Not a valid prefix. c3 = ARITHMETIC A  $\overset { \prime } { \cdot } q _ { 3 } = \mathrm { S E L E C T } .$  Excluded.</td></tr><tr><td colspan="2">Prefix-Selected Demonstration (longest valid prefix = 5) Demo</td></tr><tr><td>Demo sequence (len=5)</td><td>“A Toyota costs $20,000 and a Honda costs $25,000. What is the total cost?&quot; [SELECT, PROJECT, SELECT, PROJECT, ARITHMETIC]</td></tr><tr><td>Reasoning Prefix match</td><td>Extract two item costs → sum them.</td></tr><tr><td></td><td>Entire demo sequence  $c _ { 1 : 5 } = q _ { 1 : 5 } \checkmark .$  Valid prefix.</td></tr><tr><td colspan="2">Analysis</td></tr><tr><td></td><td>The DTW-selected demo teaches the model to apply a percentage after obtaining a base value, which matches the query&#x27;s “add tax&quot; logic. It is excluded by prefix matching because c3 ≠ q3. The prefix-</td></tr><tr><td></td><td>selected demo is a valid prefix covering 5 of 7 operations, but its reasoning stops at “sum two costs&quot; with no percentage step. The model follows this incomplete pattern, outputs $80 (the pre-tax subtotal), and</td></tr></table>

Table 10: Case 4: DTW selects a demonstration whose reasoning pattern (base value → percentage → total) matches the query, despite not being a valid prefix. Prefix matching selects a structurally simpler demo that is a valid prefix, but whose reasoning stops at “sum two costs” without any percentage computation, causing the model to omit the tax step.

<table><tr><td>Case</td><td>What is compared</td><td>Key insight</td></tr><tr><td>1</td><td>13 QDMR vs. + Modulo</td><td>Without Modulo, remainder is simulated via 3 extra ARITHMETIC steps, attracting proportion problems instead of remainder problems.</td></tr><tr><td>2</td><td>13 QDMR vs. + Algebra</td><td>Without Algebra, variable solving expands to 13-step SELECT-PROJECT chains, attracting multi-item subtraction problems.</td></tr><tr><td>3</td><td>Prefix vs. DTW</td><td>Prefix matching excludes the best structural match because  $c _ { 3 } \neq q _ { 3 } ;$  DTW spans the granularity gap. The longest valid prefix demo shares 6 of 7 operations but ends with “subtract&quot; instead  $\mathrm { o f } \ \mathrm { ^ { \bullet } a d d , ^ { \bullet } }$  misleading the model.</td></tr><tr><td>4</td><td>Prefix vs. DTW</td><td>The longest valid prefix demo matches 5 of the query&#x27;s 7 operations but has no percentage step, causing the model to miss the tax computation. DTW selects a semantically aligned demo  $( c _ { 3 } \neq q _ { 3 } )$  that includes percentage reasoning.</td></tr></table>

Table 11: Summary of the four case studies and their key insights.