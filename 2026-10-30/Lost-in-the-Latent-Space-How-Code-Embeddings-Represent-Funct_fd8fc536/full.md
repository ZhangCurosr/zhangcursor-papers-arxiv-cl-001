# Lost in the Latent Space: How Code Embeddings Represent Functional Intent and Implementation Fidelity

Lu Sun<sup>∗</sup>, Yuheng Huang<sup>∗B</sup>, Lei Ma<sup>∗†</sup>, Zhijie Wang<sup>‡</sup>

<sup>∗</sup> The University of Tokyo, Tokyo, Japan <sup>†</sup> University of Alberta, Edmonton, AB, Canada <sup>‡</sup> Concordia University, Montreal, QC, Canada

{7779117185, yuhenghuang42}@g.ecc.u-tokyo.ac.jp, ma.lei@acm.org, zhijie.wang@concordia.ca

Abstract—Code embedding models have become a key building block for many software engineering applications, yet their internal representations remain poorly understood. Existing evaluations tend to focus on downstream task performance, offering limited insight into whether a single embedding can support both functional abstraction and implementation-sensitive analysis. To address this gap, we present a large-scale empirical study of 14 state-of-the-art code embedding models. We evaluate their representational behavior through three complementary analyses: supervised probing for functional abstraction and implementation fidelity, zero-shot retrieval experiments to assess geometric similarity, and controlled perturbation experiments that measure embedding responses to semantics-preserving code transformations. The results reveal a consistent abstraction–fidelity tension: models that better preserve stylistic and structural signals tend to do so at the expense of semantic generalization, while models with similar retrieval performance can nevertheless respond very differently to the same implementation changes. Overall, our findings suggest that a single universal code representation remains out of reach, motivating more representation-aware evaluation, selection, and deployment of code embedding models in software engineering.

Index Terms—Code Embedding Models, Representation Learning, Empirical Study

## I. INTRODUCTION

Embedding models<sup>1</sup> have become a fundamental component of many Software Engineering (SE) approaches for comparing, retrieving, and prioritizing software artifacts at scale [1]–[6]. By mapping source code, requirements, commits, bug reports, or other artifacts into continuous vector representations, these models enable efficient search and analysis over large software repositories. They have become a core component of many SE tasks, including vulnerability detection [2], [7], [8], clone detection [9], defect prediction [10], code search [11], and impact analysis [12]. In these applications, the quality of the embedding space directly influences which relationships among software artifacts can be identified by downstream classifiers [2], [7], [9], [10], ranking algorithms [3], [13], [14], or human analysts [15]–[17].

Yuheng Huang is the corresponding author

The emergence of Large Language Models (LLMs) did not diminish the need for dedicated embedding models. Although LLMs provide powerful reasoning capabilities over code and natural language, their inference cost, latency, and context limitations make them unsuitable for many downstream or specialized SE tasks that require reasoning over large software repositories or external knowledge sources [13], [14], [18]. Lightweight embedding models thereby complement LLMs by efficiently searching and retrieval [1], [3], [13], [14], downstream prediction and ranking tasks [2], [4], [7], [10], and similarity-based analysis [19]–[22].

Similar to LLMs, embedding models are also intended to be general-purpose so that they can complement LLM-centric systems and be capable of supporting a diverse range of tasks. Rather than being optimized for a single downstream task, they aim to learn a representation space in which similar data points are mapped to nearby vectors while dissimilar ones remain distinguishable. Therefore, the same embedding space can support a wide range of downstream applications. From a deployment perspective, reusing a single embedding model across multiple workflows further reduces computational cost and simplifies system design. These practical benefits have been demonstrated by the success of general-purpose embedding models in natural language processing [23]–[26]. Consequently, recent code-embedding work increasingly embraces this design paradigm by building general-purpose code embedding models, e.g., multi-task retrieval models [27], [28].

However, we argue that existing evidence is insufficient to determine when this reuse assumption is reliable for SE applications. Previous studies assess code embeddings on individual downstream tasks [4], [19], [21], [29], [30], demonstrating task-specific effectiveness while providing only limited insight into whether a single embedding space can simultaneously preserve the diverse forms of software information required across tasks in the LLM era. Intuitively, an embedding is a compressed representation of a software artifact, preserving information useful for one notion of similarity but potentially reducing its ability to represent others. This raises a fundamental question:

To what extent can a single code embedding model serve as a reusable building block for software engineering applications, and how do its representational trade-offs constrain that role?

To answer this core research question, we study code embedding models along two complementary dimensions that capture recurring requirements in SE workflows. The first is high-level functional abstraction: the extent to which an embedding groups artifacts that implement the same requirement, solve the same problem, or exhibit similar algorithmic behavior despite differences in implementation. The second is implementation fidelity: the extent to which a representation faithfully preserves information about how a program is implemented. A high-fidelity representation retains naming conventions, formatting choices, syntactic idioms, structural organization, and other implementation-level regularities. This information is essential for tasks such as predicting compilability from invalid code, identifying implementation conventions, analyzing provenance, and reasoning about project-specific development patterns.

We evaluate these dimensions through three diverse analyses. First, we conduct lightweight probing experiments on frozen embeddings to determine whether functional and implementation-level information can be recovered from the learned representations. Second, we evaluate retrieval behavior directly, since many practical applications rely on the embedding geometry space’s neighborhood structure rather than a trained downstream classifier. Finally, we conduct a controlled perturbation analysis to explain the geometry behind these differences. We apply semantics-preserving code transformations as counterfactual interventions, each modifying a specific implementation characteristic while preserving the program’s intended behavior. By measuring embedding displacements under these controlled transformations, we characterize each model’s invariance profile, identifying which implementation variations are treated as irrelevant and which remain geometrically distinguishable. Collectively, these analyses provide a systematic characterization of the trade-offs embodied by modern code embedding models and offer practical guidance for selecting embedding models for different SE applications.

In summary, our contributions are as follows:

• A Representation-Aware Evaluation Framework: We introduce a systematic evaluation suite comprising four distinct sub-tasks across multiple programming languages. This benchmark is explicitly designed to isolate and evaluate the models through two operational axes: functional intent abstraction and form-level fidelity.

• A Large-Scale Empirical Study: We conduct a systematic evaluation of 14 SOTA embedding models, spanning both general-purpose and code-specific approaches. Our analysis provides insight into their representational preferences and abstraction behaviors, and reveals a consistent trade-off between stylistic fidelity and semantic abstraction.

• A Semantic-Preserving Perturbation Study: We design an unsupervised perturbation study using functionalitypreserving code transformations to examine the models underlying invariance behavior. By tracking the resulting geometric shifts, we identify which forms of local implementation variation are absorbed by the representation and which are preserved as distinct geometric signals.

To support future research, we make our code and datasets, including our independently collected AtCoder corpus, publicly available at Anonymous Git [31]. The repository also provides preliminary results on CodeSearchNet [3] and Big-CloneBench [32], together with additional uncertainty and robustness analyses of the observed correlations.

## II. BACKGROUND AND RELATED WORK

In this section, we first introduce the backgrounds about embedding models. Then we discuss the related work.

## A. Text and Code Embedding

1) Definition: An embedding model is a parameterized function that projects unstructured input data into a continuous high-dimensional vector space, commonly referred to as a latent space or embedding space. Formally, given an input domain $x ,$ such as a vocabulary of discrete tokens, an embedding model defines a mapping:

$$
f _ { \theta } : \mathcal { X } \xrightarrow { } \mathbb { R } ^ { d } ,\tag{1}
$$

where d is the dimensionality of the continuous vector space $( d \in \mathbb { Z } ^ { + } )$ and θ represents the learnable model parameters. An embedding model is trained by optimizing θ to minimize the distance between representations of related inputs while maximizing the distance between representations of unrelated inputs in the embedding space, guided by a task-specific supervision signal (e.g., semantic equivalence between programs). This is commonly formulated using contrastive, ranking, or reconstruction-based objectives [5], [11], [23], [33].

A downstream DNN classifier $H : \mathcal { X }  \mathcal { Y }$ built on top of the embedding model $f _ { \theta } ,$ , can be formally decomposed into the composition of two functions:

$$
H ( x ) = ( g _ { \phi } \circ f _ { \theta } ) ( x ) = g _ { \phi } ( f _ { \theta } ( x ) ) ,\tag{2}
$$

where $\mathcal { V }$ represents a discrete label space of K classes.

In this formulation, the embedding model $f _ { \theta } : \mathcal { X }  \mathbb { R } ^ { d }$ serves as a non-linear feature extractor, mapping the input to a continuous latent space. The function $g _ { \phi } : \mathbb { R } ^ { d }  \mathcal { V }$ operates as the tail classifier, mapping the d-dimensional embedding to the target label space. In practice, this formulation is the driving force behind many standard DNN-based code classification tasks. Across applications such as software vulnerability detection [2], [7], [8], impact analysis [12], just-in-time defect prediction [10], and code clone detection [9], the pipeline typically follows the same pattern: $f _ { \theta }$ encodes task-relevant code semantics, while a lightweight $g _ { \phi } ,$ usually a Multi-Layer Perceptron (MLP), produces the final prediction.

2) Advancement of general-purpose embedding models: Beyond task-specific code representation learning, a natural question is whether a general representation model can simultaneously capture how a program behaves and how it is implemented. This question has driven the rapid evolution of general code embedding models. Early research adapted NLP techniques [34] to source code, but researchers quickly recognized that Programming Language (PL) possess strict, formal grammatical rules unlike Natural Language (NL). As a result, subsequent work began incorporating structural priors explicitly into representation learning. Models such as Code2vec [35] and Code2seq [36] leveraged ASTs by extracting syntactic paths and embedding them into vector representations. Later work, including InferCode [37], advanced this direction through self-supervised learning on AST structures, training models to predict subtrees to enable more powerful embedding learning.

The introduction of the Transformer architecture marked a major shift toward large-scale pre-training for code representation [38]. Early encoder-based models such as CodeBERT [11] established a foundation for jointly embedding NL and PL into a shared vector space through masked language modeling. Subsequent models incorporated additional structural signals to better capture program semantics [39], [40]. Examples of such include GraphCodeBERT [41], UniXcoder [42], PLBART [33] and CodeT5 [43].

The paradigm has recently shifted toward unified representation spaces driven by contrastive learning and the immense scale of LLMs. OpenAI’s early work [23] demonstrated that training large models with contrastive objectives can produce highly expressive, general-purpose embeddings. This direction has continued to evolve with the emergence of large-scale code models [44], [45]. Building upon these foundations, recent embedding models such as CodeXEmbed [27] and CodeR [28] represent the current SOTA. Our work is orthogonal to the development of these novel architectures and pre-training paradigms. Rather than proposing a new model, we empirically investigate the latent spaces these systems construct, revealing fundamental trade-offs that can guide the design of nextgeneration training methodologies.

## B. Empirical Studies on Embedding Models

Evaluating embeddings as general-purpose, reusable representations requires moving beyond isolated downstream metrics. In Natural Language Processing (NLP) and adjacent domains, standardized benchmark suites such as MTEB [26] and MAEB [46] assess embedding spaces across diverse task families, reflecting the operational role of embeddings as task-agnostic, reusable representations. Rather than treating an embedding model as a specialized, single-purpose component, these benchmarks characterize the structural and semantic relations sustained within the latent space across use cases.

However, different from the benchmark-driven evaluation paradigm adopted in NLP, research on code embeddings in software engineering has largely focused on downstream task performance. Prior studies have investigated whether pre-trained code embeddings improve SE-specific prediction tasks [4], assessed their generalizability across distinct domains [19], examined how code-change representations predict patch correctness [29], and analyzed the behavioral characteristics of code embeddings in industrial production environments [21]. Recent empirical efforts further explore methodologies for optimizing embedding extraction from pre-trained models and guidelines for model selection in safety-critical contexts, such as vulnerability detection [30], [47]. While these studies conclusively demonstrate that model architecture and embedding configurations heavily influence downstream performance, their evaluations remain predominantly tied to extrinsic task utility. As a result, these studies primarily answer which embedding performs best for a particular application, providing comparatively limited insight into the information encoded by the representation space itself.

The growing adoption of LLM-centric systems further raises the importance of understanding embedding spaces beyond classifier-level downstream performance. In many practical workflows, frozen embeddings directly determine which code snippets are retrieved for search, repository-level context selection, or retrieval-augmented generation. This dependence has motivated a growing body of studies that investigate embedding quality in retrieval-oriented settings. CodeSearch-Net [3] established a large-scale benchmark for semantic code retrieval. CoSQA and CoSQA+ [48], [49] extended this perspective to question-driven retrieval, where models must align code with natural-language information needs rather than short search queries. Broader code information retrieval benchmarks [50] further expanded the evaluation scope, while CodeRAG-Bench [13] examined the role of retrieval quality in retrieval-augmented code generation pipelines.

Our work complements these task-centered paradigms. Rather than benchmark-hunting to determine which model achieves peak performance on a specific SE task, we systematically investigate the intrinsic properties and types of information that reusable code embedding spaces preserve. Specifically, we study code embeddings from two complementary representational perspectives:

• Functional Abstraction: The capacity to align functionally equivalent code fragments closely within the latent space, irrespective of variations in syntax, programming language, or implementation mechanics.

• Implementation Fidelity: The capacity to preserve structural anomalies, stylistic regularities, and surface-level implementation characteristics crucial for syntax-aware tasks.

To rigorously evaluate these dimensions, we employ a multifaceted methodology consisting of probing tasks to measure the recoverability of features from frozen embeddings, zeroshot retrieval tasks to observe localized geometric tendencies in deployment-like configurations, and controlled perturbation analysis to quantify vector shifts under precise semantic and syntactic code mutations. Rather than asking which embedding performs best on a particular benchmark, our goal is to understand what information different embedding spaces preserve, the trade-offs they embody, and whether they are suitable as general-purpose representations for software engineering.

## III. STUDY DESIGN

A central challenge in evaluating code embeddings is that there is no single notion of program similarity that serves all software engineering tasks. The same pair of programs may be regarded as similar because they implement the same functionality, or different because they employ distinct implementation strategies. Consequently, every embedding space implicitly prioritizes certain software properties while abstracting away others. Rather than treating these design choices as hidden assumptions, we make them explicit by asking three complementary questions: What software information remains recoverable from the embedding? Which software properties primarily determine geometric similarity? And which forms of implementation variation are intentionally absorbed by the representation?

In this study, we instantiate these questions through two complementary representational axes. The first is functional abstraction. A functionally abstract representation should place together implementations that realize the same computational intent despite differences in programming language, syntax, or program structure. The second is implementation fidelity. A high-fidelity representation should retain naming conventions, formatting choices, syntactic idioms, structural organization, and other implementation-level regularities that may be irrelevant to functional intent but remain valuable for form-sensitive software engineering tasks.

We first examine these axes through probing analysis. Probing has been widely used to study what information is encoded in neural representations by training lightweight predictors on frozen embeddings [51], [52]. In our setting, probing asks a simple scientific question: once source code has been compressed into embeddings, which software properties remain recoverable without updating the embedding model?

We instantiate the functional-abstraction axis through Algorithm Tag Prediction. Algorithm tags capture strategy-level concepts that relate implementations beyond their surface form, such as dynamic programming, graph traversal, or greedy algorithms. Although they do not fully characterize program semantics, they provide an observable signal of whether an embedding preserves high-level functional information across diverse implementations. We instantiate the implementation-fidelity axis through Author Attribution. Authorship is particularly suitable because it emerges from the aggregation of many subtle implementation-level cues, including naming habits, formatting preferences, code decomposition patterns, idiomatic constructs, and recurring local design choices, rather than any single syntactic feature.

Probing, however, only evaluates whether a learned prediction head can recover a signal from the embedding. In many practical applications, embeddings are used directly through their geometry, where nearest-neighbor relationships determine retrieval, clustering, and ranking behavior. We therefore complement probing with zero-shot retrieval experiments to test whether functional abstraction or implementation fidelity naturally governs geometric similarity.

TABLE I: Overview of our evaluations.
<table><tr><td>RQ</td><td>Evaluation</td><td>Intent</td></tr><tr><td>RQ1</td><td>1Algorithm-Tag Prediction</td><td>Measure recoverability of functional abstraction in frozen embeddings.</td></tr><tr><td></td><td>RQ1 Author Attribution</td><td>Measure recoverability of implementation fidelity in frozen embeddings.</td></tr><tr><td></td><td>RQ2 Cross-Language Functional Retrieval</td><td>Test whether code with the same problem tags is retrieved across languages.</td></tr><tr><td></td><td>RQ2 Cross-Task Author Retrieval</td><td>Test whether code by the same author is retrieved.</td></tr><tr><td>RQ3 Controlled</td><td>Transformation Analysis</td><td>Measure how controlled implementation changes move embeddings.</td></tr></table>

Finally, we use controlled perturbation analysis to move beyond task-level performance and examine the representation’s underlying invariance behavior. Following the broader interpretability principle of studying model behavior through controlled input changes and their corresponding changes in model responses [53], [54], we apply functionality-preserving code transformations that selectively modify implementation details while maintaining program behavior. We then measure which changes leave the embedding effectively unchanged and which produce substantial geometric shifts. This perspective characterizes the representation’s selective invariance [55], [56]: not merely whether it is robust to code variation, but which distinctions it suppresses, preserves, or amplifies.

We seek to investigate the following research questions:

• RQ1: How are functional abstraction and implementation fidelity related under lightweight probing?

• RQ2: How are functional abstraction and implementation fidelity related in retrieval?

• RQ3: Which implementation changes help explain the relationship observed in probing and retrieval?

The remainder of this section describes the datasets, benchmark construction, and evaluation protocols used to investigate these research questions. Table I provides a compact overview of the resulting five evaluations.

## A. Dataset Design

Our evaluation dataset is built from three public multilingual corpora: Open-R1 Codeforces [57], Google Code Jam (GCJ) [58], and HumanEval-X [59]. We use Codeforces for algorithm labels and cross-language functional retrieval, GCJ for authorship and author retrieval, and HumanEval-X for controlled transformations. The cleaning steps below define the datasets used by the RQs.

Codeforces (RQ 1,2). Open-R1 Codeforces is a dataset of competitive programming submissions. We use the C++, Java, and Python submissions, which represent the three most popular programming languages in the dataset, while retaining the problem-level algorithm tags. Submissions often mix the core algorithm with parsing, helpers, and problem-specific scaffolding, which can disperse the representational signal away from the target algorithmic evidence. We therefore clean the code within each language before constructing the aligned dataset. After structuring, each retained record keeps the source code, language, problem identity, contest identity, and algorithm tags. The final dataset contains 3,712 problems; for each problem, we keep one implementation per language, yielding 11,136 implementations.

TABLE II: Controlled implementation transformations.
<table><tr><td>Category</td><td>Controlled change</td></tr><tr><td></td><td>Data and Declaration Modify literals, initialization patterns, declarations, or value-preserving container forms.</td></tr><tr><td>Control Flow</td><td>Rewrite loops, branches, and condition structure while preserving the executed behavior.</td></tr><tr><td>Function</td><td>Restructure helper boundaries and abstraction struc- ture while preserving the externally tested API.</td></tr><tr><td>Dead/Bogus Code</td><td>Insert semantically inert code such as unreachable branches or unused helpers.</td></tr><tr><td>Format &amp; Rename</td><td>Modify lexical and layout cues such as local identi- fiers, whitespace, wrapping, and comments.</td></tr></table>

GCJ (RQ 1,2). GCJ is another dataset of competitive programming submissions with user profile information. We first exclude malformed rows and infer programming language from file extensions. Then the remaining records are deduplicated at the (language, username, task) level so repeated submissions for the same task do not dominate an author’s profile. Source code longer than 10,000 characters is also excluded. After structuring, each retained record keeps the source code, language, author identity, and task identity. To keep author-related tasks closed-set and balanced, the dataset retains authors with at least 30 valid implementations and uses the 50 most prolific authors in each language. This gives each retained class enough training examples while preventing a few prolific authors from dominating the evaluation. The dataset contains 6,684 C++, 5,078 Java, and 3,231 Python implementations.

HumanEval-X (RQ3). HumanEval-X was designed for evaluating the multilingual ability of code generation models. We retain 100 original programs per language from the C++, Java, and Python HumanEval-X solutions. Compared with longer competitive-programming submissions, these selfcontained functions reduce boilerplate and problem-context noise. Their clean, function-level structure makes HumanEval-X particularly well suited for perturbation analysis, where the goal is to isolate the representational effects of controlled implementation changes

Controlled Implementation Transformations (RQ3). To study which implementation characteristics are preserved or abstracted away by an embedding model, we construct controlled five categories of code transformations that modify one implementation dimension at a time while preserving program functionality. Table II lists the details.

Edit Magnitude. Transformation families specify what implementation characteristic is modified, while edit magnitude quantifies how much of the implementation changes. To distinguish these two factors, we measure edit magnitude at the logical-block level rather than individual tokens, allowing structural implementation changes to be captured while reducing sensitivity to superficial lexical edits. Let $B ( x ) =$ $\left( b _ { 1 } , \ldots , b _ { m } \right)$ denote the logical-block sequence extracted from code $x ,$ where each $b _ { i }$ is a maximal contiguous source block delimited by blank lines, statement-ending tokens, braces, or control-structure prefixes. Let $\mathcal { M } _ { x , x ^ { \prime } } = \mathcal { M } ( B ( x ) , B ( x ^ { \prime } ) )$ denote the non-equal opcode tuples returned after aligning $B ( x )$ and $B ( x ^ { \prime } )$ with SequenceMatcher [60].The edit magnitude ratio is defined as:

![](images/dbeaf877e3f4c53f11a88702a9d229244fff26bd50917aa40cc2d966defd4fb9.jpg)  
Fig. 1: An example of implementation changes from our study. The original program’s for-loop was replaced by a while-loop.

$$
r ( x , x ^ { \prime } ) = \operatorname* { m i n } \Biggl ( 1 , \frac { \sum _ { ( i _ { 1 } , i _ { 2 } , j _ { 1 } , j _ { 2 } ) \in \mathcal { M } _ { x , x ^ { \prime } } } \operatorname* { m a x } ( i _ { 2 } - i _ { 1 } , j _ { 2 } - j _ { 1 } ) } { \operatorname* { m a x } ( | B ( x ) | , | B ( x ^ { \prime } ) | , 1 ) } \Biggr ) .
$$

Based on the edit magnitude ratio, transformations are categorized into three edit-magnitude levels: low $( 0 \leq r \leq 0 . 3 3 )$ medium $( 0 . 3 3 < r \le 0 . 6 6 )$ , and high $( r ~ > ~ 0 . 6 6 )$ . For all transformation families except Format & Rename, comments are removed before constructing $B ( \cdot )$ so that edit magnitude reflects implementation changes rather than comment edits. For Format & Rename, comments are retained because they constitute part of the intended transformation.

Generation and Validation. We generate rewrites iteratively with GPT-5.4 xhigh under prompts specific to each transformation family. A transformed program is retained only if it satisfies four validation criteria:

1) the rewrite ratio falls within the intended strength range;   
2) all 100 black-box tests pass;

3) Claude Opus 4.5 independently verifies that the requested transformation has been correctly applied while preserving functionality;

4) sampled rewrites pass manual inspection.

Manual inspection covers 20% of the retained rewrites. Across the five transformation categories and three strength bands, the retained dataset contains 1,500 test-passing pairs per language and 4,500 pairs in total.

## B. RQ1: Probing Abstraction and Fidelity

RQ1 is instantiated through two probing tasks. For each embedding model, the encoder is frozen and separate lightweight prediction heads are trained for Algorithm-Tag Prediction and Author Attribution.

Algorithm-Tag Prediction. We evaluate whether algorithmic information remains recoverable from frozen embeddings using multi-label prediction of Codeforces algorithm tags. Separate probes are trained for C++, Java, and Python, and performance is measured by sample-averaged F1. We train and evaluate a separate probe for C++, Java, and Python and report sample-averaged F1.

• Labels: The prediction targets are 37-dimensional multi-hot vectors over the tags that remain after filtering rare tags.

• Split: The supervised split is 70/10/20 (train/validation/test) by problem. All retained implementations of the same problem stay in the same split, so the test set does not share a problem with the training set.

Author Attribution. We evaluate whether author-specific implementation patterns remain recoverable by predicting the author of each submission. Separate probes are trained for each language, and performance is measured by macro-F1.

• Labels: The prediction target is one author label among the 50 retained authors for each language.

• Split: After de-duplication, each author’s submissions are split 70/10/20 (train/validation/test), with no retained implementation appearing in more than one split.

## C. RQ2: Retrieval of Functionality and Authorship

RQ2 uses Codeforces and GCJ datasets to evaluate the same two representational axes through nearest-neighbor retrieval on frozen embeddings. Gallery items are ranked by cosine similarity, and retrieval performance is measured using Hit@10. Formally, for query $q ,$ let $G _ { q } ^ { ( 1 0 ) }$ be the first ten retrieved gallery items and $R _ { q }$ be the relevant set for that query:

$$
\mathrm { H i t @ 1 0 } ( q ) = \pmb { 1 } \{ G _ { q } ^ { ( 1 0 ) } \cap R _ { q } \neq \emptyset \} .\tag{3}
$$

We average this value over queries, and use language-macro averages for author retrieval.

Cross-Language Functional Retrieval. Each implementation is used as a query to retrieve functionally equivalent implementations in the other two programming languages.

• Query and gallery: Each query is an implementation in one language. The gallery contains retained implementations in the other languages.

• Relevance: The relevant items are retained implementations from the same Codeforces task as the query, sharing its problem-level algorithm tags.

• Ranking: Embeddings are normalized and ranked by cosine similarity. Performance is summarized with Hit@10.

Cross-Task Author Retrieval. Each implementation is used to retrieve other submissions written by the same author.

• Query and gallery: Each query is one implementation. The gallery contains retained implementations in the query language.

• Relevance: The relevant items are retained implementations by the query author.

TABLE III: Unified model pool. Models are grouped by official positioning and sorted by release year within each group. Parameter counts follow official documentation when available; a dash indicates no public parameter count.
<table><tr><td>Official model name (alias)</td><td>Params.</td><td>Year</td></tr><tr><td>Code embeddings</td><td></td><td></td></tr><tr><td>UniXCoder (UniXcoder) [42]</td><td>125M</td><td>2022</td></tr><tr><td>voyage-code-3 (Voyage-Code-3) [61]</td><td></td><td>2024</td></tr><tr><td>CodeR (CodeR) [28]</td><td>1.5B</td><td>2025</td></tr><tr><td>jina-code-embeddings-1.5b (Jina-Code- 1.5B) [62]</td><td>1.5B</td><td>2025</td></tr><tr><td>General-purpose embeddings</td><td></td><td></td></tr><tr><td>text-embedding-ada-002 (Ada-002) [63]</td><td></td><td>2022</td></tr><tr><td>E5-Base (E5-Base) [24]</td><td>0.1B</td><td>2022</td></tr><tr><td>BGE-M3 (BGE-M3) [25]</td><td>560M</td><td>2024</td></tr><tr><td>doubao-embedding-large (Doubao-</td><td></td><td>2024</td></tr><tr><td>Large) [64] text-embedding-3-large (TE3-Large) [65]</td><td></td><td></td></tr><tr><td>text-embedding-3-small (TE3-Small) [65]</td><td></td><td>2024</td></tr><tr><td>Llama-Embed-Nemotron-8B (Llama-Embed-</td><td>8B</td><td>2024</td></tr><tr><td>8B) [66]</td><td></td><td>2025</td></tr><tr><td>Qwen3-Embedding-0.6B (Qwen3-0.6B) [67]</td><td>0.6B</td><td>2025</td></tr><tr><td>Qwen3-Embedding-4B (Qwen3-4B) [67]</td><td>4B</td><td>2025</td></tr><tr><td>Qwen3-Embedding-8B (Qwen3-8B) [67]</td><td>8B</td><td>2025</td></tr></table>

• Ranking: Embeddings are normalized and ranked by cosine similarity. Performance is summarized with language-macro Hit@10.

## D. RQ3: Controlled Analysis of Implementation Changes

RQ3 evaluates representational invariance using the controlled implementation transformations described in Section III-A. For each original program x from HumanEval-X and its transformed counterpart $x ^ { \prime } .$ , we encode both using the same frozen model $f ( \cdot )$ and compute

$$
{ \mathrm { s h i f t } } ( x , x ^ { \prime } ) = { \bigl ( } 1 - \cos ( f ( x ) , f ( x ^ { \prime } ) ) { \bigr ) } \times 1 0 0 .
$$

Larger shifts indicate greater sensitivity to the applied implementation transformation. For each model, we summarize embedding shifts by language, transformation family, and edit magnitude.

## E. Experimental Settings

Model Selection. We evaluate 14 embedding models: four code-focused models and ten general-purpose models. Table III reports their official names, disclosed parameter counts, and release years. The pool spans release years from 2022 to 2025 and includes both compact encoder-scale models and billion-parameter embedders. Disclosed parameter counts range from roughly 0.1 B to 8 B (Table III), while some closedsource models did not disclose parameter counts.

Encoding and Probe Training. Each benchmark instance is encoded once using the documented model interface, with backbone parameters frozen. For each supervised task and language, all models share the same input formulation, split, single-hidden-layer head<sup>2</sup>, 100-epoch training budget, and validation-based checkpoint rule; the embedding is the only changing factor. Each supervised run is repeated five times with independent seeds, and the reported value is the average. Optimizer settings, batch sizes, and seed lists are fixed across models. RQ2 and RQ3 fit no classifier, using cosinesimilarity retrieval or scaled cosine-distance scores on frozen embeddings.

TABLE IV: Algorithm-tag prediction sample-F1 by language. Rows are sorted by the three-language average.
<table><tr><td>Model</td><td>C++</td><td>Java</td><td>Python</td></tr><tr><td>Doubao-Large</td><td>0.542</td><td>0.505</td><td>0.518</td></tr><tr><td>CodeR</td><td>0.518</td><td>0.532</td><td>0.512</td></tr><tr><td>Voyage-Code-3</td><td>0.514</td><td>0.530</td><td>0.515</td></tr><tr><td>Qwen3-4B</td><td>0.490</td><td>0.512</td><td>0.488</td></tr><tr><td>Qwen3-8B</td><td>0.487</td><td>0.504</td><td>0.498</td></tr><tr><td>Jina-Code-1.5B</td><td>0.484</td><td>0.497</td><td>0.495</td></tr><tr><td>Qwen3-0.6B</td><td>0.453</td><td>0.455</td><td>0.455</td></tr><tr><td>Llama-Embed-8B</td><td>0.445</td><td>0.446</td><td>0.462</td></tr><tr><td>Ada-002</td><td>0.471</td><td>0.414</td><td>0.441</td></tr><tr><td>TE3-Large</td><td>0.429</td><td>0.413</td><td>0.448</td></tr><tr><td>TE3-Small</td><td>0.404</td><td>0.385</td><td>0.405</td></tr><tr><td>UniXcoder</td><td>0.385</td><td>0.384</td><td>0.405</td></tr><tr><td></td><td></td><td></td><td>0.385</td></tr><tr><td>E5-Base</td><td>0.371</td><td>0.368</td><td></td></tr><tr><td>BGE-M3</td><td>0.356</td><td>0.348</td><td>0.395</td></tr></table>

Leakage Control. The benchmark uses problem-level splits, per-author de-duplication, and same-task exclusion where applicable. These controls reduce leakage within the constructed evaluations. Pre-training exposure to public corpora cannot be ruled out, so we disclose source identities and model release years (Table III) rather than claim contamination-free evaluation.

## IV. RESULTS

## A. RQ1: Probing Shows an Abstraction–Fidelity Split

RQ1 reveals a fundamental trade-off in frozen code embeddings: while both high-level functional intention and low-level implementation fidelity are recoverable, they are optimally supported by entirely different model profiles. At a high level, the results demonstrate that the embedding capturing algorithmic properties tend to suppress stylistic authorship signals, and vice versa. Across all 14 evaluated models, the Pearson correlation between the F1 scores (averaged across all target languages) for algorithm tag prediction and author attribution is negative $( r \ : = \ : - 0 . 4 7 3 )$ . This inverse relationship persists robustly when calculated within each individual language, yielding correlations of $r = - 0 . 3 9 2$ for $\mathbf { C } + + , r = - 0 . 5 8 3$ for Java, and $r = - 0 . 3 9 8$ for Python. Additional Pearson tests, paired-bootstrap BCa confidence intervals, and Spearman rank correlations are provided in our anonymous repository [31]. Rather than suggesting that one signal is completely absent when the other is present, these consistent negative correlations provide strong model-ranking evidence that optimization for one property inherently compromises the other.

TABLE V: Author-attribution macro-F1 by language. Rows are sorted by the three-language average.
<table><tr><td>Model</td><td>C++</td><td>Java</td><td>Python</td></tr><tr><td>TE3-Large</td><td>0.988</td><td>0.977</td><td>0.978</td></tr><tr><td>Llama-Embed-8B</td><td>0.990</td><td>0.969</td><td>0.945</td></tr><tr><td>Ada-002</td><td>0.988</td><td>0.935</td><td>0.940</td></tr><tr><td>TE3-Small</td><td>0.968</td><td>0.962</td><td>0.926</td></tr><tr><td>BGE-M3</td><td>0.917</td><td>0.954</td><td>0.924</td></tr><tr><td>Doubao-Large</td><td>0.923</td><td>0.924</td><td>0.921</td></tr><tr><td>E5-Base</td><td>0.920</td><td>0.909</td><td>0.869</td></tr><tr><td>UniXcoder</td><td>0.938</td><td>0.862</td><td>0.821</td></tr><tr><td>CodeR</td><td>0.871</td><td>0.799</td><td>0.849</td></tr><tr><td>Voyage-Code-3</td><td>0.800</td><td>0.817</td><td>0.826</td></tr><tr><td>Qwen3-0.6B</td><td>0.668</td><td>0.599</td><td>0.595</td></tr><tr><td>Qwen3-4B</td><td>0.587</td><td>0.459</td><td>0.417</td></tr><tr><td>Qwen3-8B</td><td>0.554</td><td>0.425</td><td>0.465</td></tr><tr><td>Jina-Code-1.5B</td><td>0.521</td><td>0.485</td><td>0.430</td></tr></table>

A closer examination of the model rankings further illustrates this trade-off. As reported in Table IV, certain encoders make functional concepts particularly accessible to a shallow probe, with Doubao-Large achieving the highest Algorithm-Tag Prediction scores for C++ and Python, while CodeR leads for Java. The rankings change markedly for implementation fidelity. As shown in Table V, Llama-Embed-8B achieves the best performance on C++, whereas TE3-Large leads on both Java and Python. Performance on this task also spans a much wider range, from near-perfect classification to a macro-F1 of only 0.417. Notably, these leaders are not uniformly code-specialized models: general-purpose embeddings such as Doubao-Large, TE3-Large, and Llama-Embed-8B match or exceed code-focused models on these probes.

The divergence becomes even clearer when considering the top-performing models. Across all three programming languages, the top-five models for the two probing tasks have no overlap. Because the two tasks use different label spaces and F1 averaging schemes, this observation is meaningful not in terms of absolute performance but in terms of relative model rankings. An engineer selecting an encoder for functional tag prediction would therefore overlook the strongest models for author attribution, and vice versa. Taken together, these results show that functional abstraction and implementation fidelity form competing representational axes in frozen code embeddings, implying that embedding model’s selection should be guided by the software property most relevant to the downstream application.

Finding 1: Probing reveals a clear abstraction–fidelity split. Both signals are recoverable, but their optimal model profiles are disjoint across all three languages. Moreover, code-specialized embeddings do not uniformly dominate general-purpose embeddings, especially on implementation-fidelity probes. Encoder selection should therefore be guided by the target software property rather than by model positioning alone.

TABLE VI: Hit@10 for cross-language retrieval. Functional retrieval ranks implementations in other languages; author retrieval ranks same-author implementations.
<table><tr><td>Model</td><td>Functional</td><td>Author</td></tr><tr><td>CodeR</td><td>0.996</td><td>0.481</td></tr><tr><td>Qwen3-8B</td><td>0.991</td><td>0.323</td></tr><tr><td>Qwen3-4B</td><td>0.990</td><td>0.332</td></tr><tr><td>Voyage-Code-3</td><td>0.985</td><td>0.471</td></tr><tr><td>Jina-Code-1.5B</td><td>0.985</td><td>0.338</td></tr><tr><td>Qwen3-0.6B</td><td>0.942</td><td>0.413</td></tr><tr><td>Doubao-Large</td><td>0.865</td><td>0.601</td></tr><tr><td>Ada-002</td><td>0.504</td><td>0.933</td></tr><tr><td>TE3-Large</td><td>0.500</td><td>0.948</td></tr><tr><td>BGE-M3</td><td>0.184</td><td>0.957</td></tr><tr><td>TE3-Small</td><td>0.176</td><td>0.952</td></tr><tr><td>E5-Base</td><td>0.147</td><td>0.935</td></tr><tr><td>Llama-Embed-8B</td><td>0.142</td><td>0.973</td></tr><tr><td>UniXcoder</td><td>0.126</td><td>0.888</td></tr></table>

We also examine whether embedding models exhibit systematic preferences across programming languages. Overall, the relative model rankings are highly consistent across C++, Java, and Python for both probing tasks, suggesting that the abstraction–fidelity trade-off is largely language independent. For example, Doubao-Large, CodeR, and Voyage-Code-3 consistently rank among the strongest models for Algorithm-Tag Prediction, whereas TE3-Large and Llama-Embed-8B remain the strongest performers for Author Attribution across all three languages. The absolute performance differences between languages are also modest for most models, indicating that representational preferences are driven primarily by the embedding model rather than the programming language.

## B. RQ2: Retrieval Shows a Stronger Abstraction–Fidelity Split

While RQ1 demonstrates that both computational-intent and implementation-realization evidence can be recovered through lightweight probes, RQ2 moves to a more practical setting by evaluating embeddings directly in retrieval, without relying on a trained prediction head. If a single embedding space supported both representational axes equally well, functionally related programs and implementation-aligned programs by the same author would both appear among a query’s nearest neighbors. Instead, direct retrieval reveals a pronounced separation between these two objectives.

This separation is reflected consistently across models. Across the 14 evaluated models, Cross-Language Functional Retrieval and Cross-Task Author Retrieval are strongly negatively correlated $( r ~ = ~ - 0 . 9 3 2 ;$ ; Table VI, Figure 2). The evaluated models fall into two specialized groups. Functional specialists, including CodeR, Qwen3-8B, Qwen3-4B, Voyage-Code-3, and Jina-Code-1.5B, excel at semantic retrieval (Hit@ $1 0 \geq 0 . 9 8 5 )$ but perform substantially worse on author retrieval (Hit@10: 0.323–0.481). In contrast, authororiented models such as BGE-M3, TE3-Small, E5-Base, Llama-Embed-8B, and UniXcoder achieve high author retrieval accuracy (Hit@10: 0.888–0.973) while exhibiting poor cross-language functional retrieval $( \mathrm { H i t } @ 1 0 ~ \le ~ 0 . 1 8 4 )$ . This geometric split can directly impact downstream utility. Selecting a model for cross-language functional search pushes authorial signals out of the local neighborhood, while optimizing for authorship tracking largely removes functionally related implementations across different programming languages.

Finding 2: Retrieval makes the abstraction–fidelity split clear. Evaluation results expose a clearer trade-off between the two axes. Across the 14 evaluated models, functional and author retrieval performance is strongly negatively correlated $( r ~ = ~ - 0 . 9 3 2 )$ Consequently, we argue that embedding selection should therefore be guided by the task property rather than a single overall ranking.

Interestingly, we also observe that scaling within the same embedding family can shift representational preferences. As the Qwen3 embedding model scales from 0.6B to 4B and 8B parameters, functional retrieval steadily improves (Hit@10: $0 . 9 4 2  0 . 9 9 1 )$ , while author retrieval declines correspondingly $( \mathrm { H i t } @ 1 0 ! 0 . 4 1 3 \to 0 . 3 2 3 )$ . This trend mirrors the probing results from RQ1, where larger Qwen3 variants achieved higher Algorithm-Tag Prediction performance at the expense of Author Attribution. Taken together, these findings suggest that increasing model scale strengthens representations of computational intent while gradually reducing the prominence of implementation-level information in the embedding space, particularly in the nearest-neighbor geometry.

## C. RQ3: Controlled Rewrites Reveal Selective Invariance

In RQ3, we treat each functionality-preserving mutation as a counterfactual intervention: the program’s tested behavior is held constant while specific aspects of its implementation are modified. Small embedding displacements indicate that the altered implementation detail has been abstracted away, whereas larger displacements indicate that it remains geometrically salient. As expected from the edit-strength levels defined in Section III-A, geometric displacement increases monotonically from low- to medium- to high-edit-magnitude transformations across every model and transformation family (Table VII), confirming that the metric responds consistently to progressively larger implementation changes. We therefore focus on how displacement differs across implementation transformation families and embedding models.

The results provide a geometric explanation for the abstraction–fidelity trade-off observed in RQ1 and RQ2. Models that excel at functional retrieval consistently exhibit smaller embedding shifts under all five transformation families than models that perform poorly on functional retrieval. The largest difference occurs for Function rewrites, where the average displacement is 5.508 for the high-functional group compared with 11.468 for the low-functional group. These results suggest that embeddings optimized for functional retrieval are more invariant to implementation-level variation.

![](images/21076b2b944dd62358127695cc38fb9cf0ff3391c84eee6a3391669680a024c7.jpg)  
Fig. 2: Same-problem versus same-author Hit@10 across the 14 models. Circles denote code-focused models, squares denote general-purpose models, the line is a linear fit.

TABLE VII: Scaled cosine distance from each original program to its test-passing transformed version, averaged over languages. low $( 0 \leq r \leq 0 . 3 3 )$ , med. $( 0 . 3 3 < r \leq 0 . 6 6 )$ , and high (r > 0.66) denote transformation strength.
<table><tr><td></td><td colspan="3">Data/Decl.</td><td colspan="3">Control</td><td colspan="3">Function</td><td colspan="3">Dead/Bogus</td><td colspan="3">Format/Rename</td></tr><tr><td>Model</td><td>low</td><td>med.</td><td>high</td><td>low</td><td>med.</td><td>high</td><td>low</td><td>med.</td><td>high</td><td>low</td><td>med.</td><td>high</td><td>low</td><td>med.</td><td>high</td></tr><tr><td>CodeR</td><td>1.717</td><td>4.601</td><td>10.209</td><td>1.460</td><td>3.344</td><td>7.282</td><td>4.227</td><td>7.838</td><td>15.403</td><td>4.123</td><td>10.032</td><td>22.136</td><td>3.262</td><td>5.926</td><td>12.204</td></tr><tr><td>Qwen3-8B</td><td>1.085</td><td>2.481</td><td>4.392</td><td>1.013</td><td>2.033</td><td>3.656</td><td>1.688</td><td>3.355</td><td>6.434</td><td>2.373</td><td>5.263</td><td>8.684</td><td>2.805</td><td>4.268</td><td>7.620</td></tr><tr><td>Qwen3-4B</td><td>1.090</td><td>2.392</td><td>4.168</td><td>0.995</td><td>2.039</td><td>3.479</td><td>1.725</td><td>3.465</td><td>6.416</td><td>2.086</td><td>4.919</td><td>10.526</td><td>3.134</td><td>4.589</td><td>7.407</td></tr><tr><td>Voyage-Code-3</td><td>0.605</td><td>1.704</td><td>3.794</td><td>0.558</td><td>1.390</td><td>2.853</td><td>0.959</td><td>2.321</td><td>5.370</td><td>1.733</td><td>3.881</td><td>8.935</td><td>1.572</td><td>2.413</td><td>5.080</td></tr><tr><td>Jina-Code-1.5B</td><td>1.706</td><td>5.181</td><td>10.053</td><td>1.269</td><td>3.121</td><td>6.888</td><td>2.617</td><td>6.426</td><td>14.370</td><td>3.942</td><td>11.361</td><td>27.248</td><td>2.670</td><td>4.063</td><td>9.306</td></tr><tr><td>Qwen3-0.6B</td><td>1.127</td><td>2.859</td><td>4.887</td><td>0.813</td><td>1.712</td><td>2.961</td><td>1.535</td><td>3.079</td><td>5.846</td><td>1.804</td><td>4.656</td><td>9.859</td><td>2.421</td><td>3.496</td><td>5.951</td></tr><tr><td>Doubao-Large</td><td>0.807</td><td>1.654</td><td>3.191</td><td>0.791</td><td>1.653</td><td>2.804</td><td>1.200</td><td>2.275</td><td>4.693</td><td>1.419</td><td>2.948</td><td>5.679</td><td>1.551</td><td>2.202</td><td>4.202</td></tr><tr><td>Ada-002</td><td>0.600</td><td>1.381</td><td>2.978</td><td>0.469</td><td>0.947</td><td>1.916</td><td>1.273</td><td>2.830</td><td>5.863</td><td>1.063</td><td>2.489</td><td>6.370</td><td>1.863</td><td>2.475</td><td>4.752</td></tr><tr><td>TE3-Large</td><td>2.522</td><td>5.857</td><td>11.413</td><td>2.189</td><td>4.225</td><td>7.622</td><td>4.652</td><td>9.403</td><td>17.683</td><td>4.716</td><td>8.629</td><td>17.870</td><td>5.238</td><td>7.616</td><td>13.785</td></tr><tr><td>BGE-M3</td><td>1.481</td><td>4.527</td><td>11.204</td><td>1.225</td><td>2.826</td><td>6.424</td><td>2.754</td><td>7.069</td><td>16.882</td><td>4.711</td><td>8.768</td><td>19.530</td><td>3.648</td><td>7.459</td><td>14.128</td></tr><tr><td>TE3-Small</td><td>1.292</td><td>3.189</td><td>7.482</td><td>1.008</td><td>2.062</td><td>4.508</td><td>4.327</td><td>10.304</td><td>19.787</td><td>2.544</td><td>5.971</td><td>16.709</td><td>4.391</td><td>5.865</td><td>11.091</td></tr><tr><td>E5-Base</td><td>0.630</td><td>1.883</td><td>5.571</td><td>0.580</td><td>1.298</td><td>3.088</td><td>1.456</td><td>3.516</td><td>8.377</td><td>1.911</td><td>4.049</td><td>9.607</td><td>1.357</td><td>2.620</td><td>6.070</td></tr><tr><td>Llama-Embed-8B</td><td>2.891</td><td>6.993</td><td>16.877</td><td>2.596</td><td>5.277</td><td>11.261</td><td>4.358</td><td>10.347</td><td>24.398</td><td>5.069</td><td>11.359</td><td>30.738</td><td>5.070</td><td>8.175</td><td>18.494</td></tr><tr><td>UniXcoder</td><td>3.854</td><td>9.130</td><td>24.349</td><td>3.528</td><td>6.838</td><td>16.678</td><td>6.880</td><td>14.428</td><td>37.139</td><td>9.484</td><td>20.674</td><td>45.312</td><td>6.627</td><td>11.692</td><td>28.849</td></tr></table>

However, this invariance is highly selective rather than uniform. Across all 14 models, Control-Flow transformations consistently produce the smallest geometric shifts. For example, under high-edit-magnitude transformations, Voyage-Code-3, Doubao-Large, and Qwen3-8B exhibit shifts of only 2.853, 2.804, and 3.656, respectively. In contrast, Dead/Bogus Code and Function transformations are consistently the most disruptive. UniXcoder, for example, reaches geometric shifts of 45.312 for Dead/Bogus Code and 37.139 for Function transformations, while Llama-Embed-8B reaches 30.738 and 24.398, respectively. These observations indicate that controlflow variations are widely abstracted away, whereas program decomposition and semantically inert code remain much more salient in the embedding space.

Surprisingly, models with comparable functional retrieval performance do not necessarily exhibit the same invariance profile. Voyage-Code-3 and Qwen3 series remain relatively insensitive across most transformation families, whereas CodeR and Jina-Code-1.5B are substantially more sensitive to Function and Dead/Bogus transformations despite achieving similarly strong retrieval performance. This suggests that comparable functional abstraction can arise through different representational strategies rather than a single notion of invariance.

Finding 3: Functional abstraction is achieved through selective implementation invariance rather than indiscriminate robustness. Embeddings optimized for functional retrieval consistently suppress implementation-level variation, but different models abstract away different types of implementation changes, indicating multiple representational strategies for achieving similar retrieval behavior.

## V. DISCUSSION AND FUTURE WORK

Our findings shed light on several future directions for code embedding models in software engineering.

Rethinking the Evaluation of Code Embedding Models. Current evaluations of code embedding models are typically organized around one or a small number of downstream tasks, implicitly treating strong benchmark performance as evidence of a generally useful representation. Our results suggest that this assumption should be revisited. Across probing (RQ1), retrieval (RQ2), and controlled transformation analysis (RQ3), we consistently observe that embedding models preserve different software properties to different degrees. Models that perform strongly on functional abstraction often rank poorly on implementation-sensitive tasks, while models with similar retrieval performance can nevertheless exhibit substantially different responses to implementation changes. These findings indicate that downstream performance alone provides only a partial view of an embedding’s representational behavior.

Building on these observations, future evaluation frameworks may characterize a broader range of software relationships than those considered in this study. While we examine functional abstraction, implementation fidelity, and implementation-level invariance, embedding models may also encode behavioral, structural, evolutionary, and projectspecific relationships. Future studies could therefore investigate representations of API usage, architectural roles, dependency structures, security properties, and software evolution through commits, refactorings, and code reviews. More importantly, our results suggest that embedding evaluation should become increasingly diagnostic rather than purely comparative. Instead of reporting only which model performs best, future studies may also investigate why embedding models succeed or fail by identifying which software properties are preserved, which are abstracted away, and how these representations evolve under realistic software development activities. Such evaluations would provide a richer understanding of embedding behavior while offering more actionable guidance for both model developers and practitioners.

Representation-Aware Deployment of Code Embeddings. Our study also has practical implications for the deployment of embedding models in software engineering systems. A common practice is to adopt a single embedding model as a general-purpose backbone for diverse downstream applications, including code search, repository retrieval, and software analytics. However, our results suggest that embedding models should not be regarded as interchangeable. Across RQ1 and RQ2, models that excel at functional abstraction consistently differ from those that preserve implementation-level information, while RQ3 further shows that even models with comparable retrieval performance may encode substantially different notions of implementation similarity.

Rather than selecting embedding models solely according to overall benchmark performance, practitioners should instead consider the software properties required by the target application. Models that strongly abstract implementation variation are naturally suited to semantic retrieval and search-oriented applications, whereas models that retain implementation-level information are more appropriate for provenance analysis, authorship attribution, and other style-sensitive tasks. More broadly, our perturbation analysis suggests that robustness to implementation variation should become an additional consideration during model selection, particularly for applications expected to operate over evolving codebases. From a sustainability perspective, representation-aware deployment also enables more effective reuse of existing embedding models. Rather than repeatedly training or deploying multiple taskspecific embeddings, practitioners can select representations whose behavioral characteristics already align with application requirements. This reduces unnecessary model development, reduces computational and maintenance overhead, and ultimately contributes to more sustainable AI software.

## VI. THREATS TO VALIDITY

Construct Validity. A primary threat to construct validity is whether our evaluation tasks adequately operationalize the broader concepts of functional abstraction and implementation fidelity. To mitigate this threat, we evaluate complementary properties through probing, retrieval, and controlled transformation analyses across multiple datasets and languages, providing converging evidence from different perspectives.

Internal Validity. One potential threat to internal validity is that an expressive downstream classifier could overcompensate for poor embeddings, obscuring differences in representation quality. To mitigate this, all models share identical probe architectures (a single-hidden-layer MLP), optimization settings, data splits, and training budgets so that the embedding representation remains the primary experimental variable. We also repeated each training run five times with different random seeds and report the average performance. In addition, controlled transformations are generated with an LLM and therefore may not fully represent edits produced during real software development. We mitigate this by requiring all transformed programs to pass the original test suite, validating transformation correctness using an independent LLM, and performing manual inspection on a sampled subset.

External Validity. One threat to external validity is that our experiment tasks are constructed from competitive programming and code generation datasets, which may not fully reflect the characteristics of industrial software projects. We partially mitigate this threat by using multiple datasets that capture complementary software engineering scenarios, although future studies should validate our findings on large-scale industrial repositories. Another threat is that our evaluation is limited to function-level code written in C++, Java, and Python. Future work may consider investigating repositorylevel representations and additional programming languages. Finally, although we evaluated a highly diverse pool of 14 SOTA models, ranging from compact encoders to billionparameter general-purpose embeddings, our findings may not generalize to future embedding models. Future work may extend our experiments to newer embedding architectures or models trained with substantially different objectives.

## VII. CONCLUSION

In this paper, we moved beyond monolithic downstream benchmarks to systematically deconstruct the latent spaces of 14 SOTA code embedding models. By evaluating these representations through the distinct lenses of supervised probing, geometric retrieval, and invariant perturbation, we exposed an inherent abstraction–fidelity tension. Our findings demonstrate that semantic generalization and implementationlevel sensitivity operate as competing design trade-offs. Furthermore, domain-specific pretraining is not a panacea, as code-specialized models do not uniformly outperform generalpurpose alternatives. These insights demonstrate that a single, universal code representation remains out of reach. For practitioners, model selection should be treated as an explicit alignment problem between a task’s unique requirements and an embedding’s intrinsic biases. For the research community, our results highlight the limitations of aggregate, single-score leaderboards. Rather than chasing a hypothetical all-purpose vector space, future work can focus on evaluation frameworks and model architectures that explicitly characterize, partition, or dynamically adapt to these distinct code properties.

## REFERENCES

[1] X. Gu, H. Zhang, and S. Kim, “Deep code search,” in Proceedings of the 40th international conference on software engineering, 2018, pp. 933–944.

[2] Y. Zhou, S. Liu, J. Siow, X. Du, and Y. Liu, “Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks,” Advances in neural information processing systems, vol. 32, 2019.

[3] H. Husain, H.-H. Wu, T. Gazit, M. Allamanis, and M. Brockschmidt, “Codesearchnet challenge: Evaluating the state of semantic code search,” arXiv preprint arXiv:1909.09436, 2019.

[4] Z. Ding, H. Li, W. Shang, and T.-H. P. Chen, “Can pre-trained code embeddings improve model performance? revisiting the use of code embeddings in software engineering tasks,” Empirical Software Engineering, vol. 27, no. 3, p. 63, 2022.

[5] Z. Xian, R. Huang, D. Towey, C. Fang, and Z. Chen, “Transformcode: A contrastive learning framework for code embedding via subtree transformation,” IEEE Transactions on Software Engineering, vol. 50, no. 6, pp. 1600–1619, 2024.

[6] Y. Wang, K. Duan, D. Zheng, E. Shi, F. Zhang, Y. Wang, J. Chen, X. Liu, Y. Ma, H. Zhang, Q. Wang, and Z. Zheng, “Towards an understanding of context utilization in code intelligence,” vol. 58, no. 11, Apr. 2026. [Online]. Available: https://doi.org/10.1145/3797261

[7] Z. Li, D. Zou, S. Xu, X. Ou, H. Jin, S. Wang, Z. Deng, and Y. Zhong, “Vuldeepecker: A deep learning-based system for vulnerability detection.” in NDSS. The Internet Society, 2018. [Online]. Available: http://dblp.uni-trier.de/db/conf/ndss/ndss2018.html#LiZXO0WDZ18

[8] X.-C. Wen, Z. Lin, C. Gao, H. Zhang, Y. Wang, and Q. Liao, “Repository-level graph representation learning for enhanced security patch detection,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 2025, pp. 1–13.

[9] J. Zhang, X. Wang, H. Zhang, H. Sun, K. Wang, and X. Liu, “A novel neural source code representation based on abstract syntax tree,” in Proceedings of the 41st International Conference on Software Engineering. IEEE Press, 2019, pp. 783–794.

[10] T. Hoang, H. J. Kang, D. Lo, and J. Lawall, “Cc2vec: Distributed representations of code changes,” in Proceedings of the ACM/IEEE 42nd international conference on software engineering, 2020, pp. 518–529.

[11] Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang, and M. Zhou, “CodeBERT: A pre-trained model for programming and natural languages,” in Findings of the Association for Computational Linguistics: EMNLP 2020, T. Cohn, Y. He, and Y. Liu, Eds. Online: Association for Computational Linguistics, Nov. 2020, pp. 1536–1547. [Online]. Available: https://aclanthology.org/2020.findings-emnlp.139/

[12] Y. Yan, N. Cooper, K. Moran, G. Bavota, D. Poshyvanyk, and S. Rich, “Enhancing code understanding for impact analysis by combining transformers and program dependence graphs,” Proceedings of the ACM on Software Engineering, vol. 1, no. FSE, pp. 972–995, 2024.

[13] Z. Z. Wang, A. Asai, X. V. Yu, F. F. Xu, Y. Xie, G. Neubig, and D. Fried, “CodeRAG-bench: Can retrieval augment code generation?” in Findings of the Association for Computational Linguistics: NAACL 2025, L. Chiruzzo, A. Ritter, and L. Wang, Eds. Albuquerque, New Mexico: Association for Computational Linguistics, Apr. 2025, pp. 3199–3214. [Online]. Available: https://aclanthology.org/2025.findings-naacl.176/

[14] Y. Tao, Y. Li, Y. Qin, and Y. Liu, “Retrieval-augmented code generation: A survey with focus on repository-level approaches,” arXiv preprint arXiv:2510.04905, 2025.

[15] C. Zeng, Y. Yu, S. Li, X. Xia, Z. Wang, M. Geng, L. Bai, W. Dong, and X. Liao, “degraphcs: Embedding variable-based flow graph for neural code search,” ACM Transactions on Software Engineering and Methodology, vol. 32, no. 2, pp. 1–27, 2023.

[16] D. Das, N. S. Mathews, A. Mathai, S. Tamilselvam, K. Sedamaki, S. Chimalakonda, and A. Kumar, “Comex: a tool for generating customized source code representations,” in 2023 38th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2023, pp. 2054–2057.

[17] L. Yang, J. Xu, Y. Zhang, H. Zhang, and A. Bacchelli, “Evacrc: Evaluating code review comments,” in Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2023, pp. 275–287.

[18] S. Zhao, Y. Shao, Y. Huang, J. Song, Z. Wang, C. Wan, and L. Ma, “Understanding the fundamental design decisions of retrieval-augmented generation systems,” ACM Transactions on Software Engineering and Methodology, 2026.

[19] H. J. Kang, T. F. Bissyande, and D. Lo, “Assessing the generalizability´ of code2vec token embeddings,” in 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019, pp. 1–12.

[20] Y. Huang, J. Song, Z. Wang, S. Zhao, H. Chen, F. Juefei-Xu, and L. Ma, “Look before you leap: An exploratory study of uncertainty analysis for large language models,” IEEE Transactions on Software Engineering, vol. 51, no. 2, pp. 413–429, 2025.

[21] B. Fein, M. Jungwirth, G. Fraser, and F. Kandlinger, “Challenges of deploying code embeddings: an industrial case study on method name generation,” Automated Software Engineering, vol. 33, no. 2, p. 53, 2026.

[22] F. Ribeiro, C. Spiess, P. Devanbu, and S. Nadi, “On LLMs’ internal representation of code correctness,” in Proceedings of the 48th International Conference on Software Engineering (ICSE ’26). Rio de Janeiro, Brazil: Association for Computing Machinery, 2026.

[23] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy et al., “Text and code embeddings by contrastive pre-training,” arXiv preprint arXiv:2201.10005, 2022.

[24] L. Wang, N. Yang, X. Huang, B. Jiao, L. Yang, D. Jiang, R. Majumder, and F. Wei, “Text embeddings by weakly-supervised contrastive pretraining,” arXiv preprint arXiv:2212.03533, 2022.

[25] J. Chen, S. Xiao, P. Zhang, K. Luo, D. Lian, and Z. Liu, “Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation,” 2024.

[26] N. Muennighoff, N. Tazi, L. Magne, and N. Reimers, “Mteb: Massive text embedding benchmark,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023, pp. 2014–2037.

[27] Y. Liu, R. Meng, S. Joty, silvio savarese, C. Xiong, Y. Zhou, and S. Yavuz, “CodeXEmbed: A generalist embedding model family for multilingual and multi-task code retrieval,” in Second Conference on Language Modeling, 2025. [Online]. Available: https://openreview.net/forum?id=z3lG70Azbg

[28] C. Li, J. Chen, Y. Shao, D. Lian, and Z. Liu, “Towards a generalist code embedding model based on massive data synthesis,” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025. [Online]. Available: https://openreview.net/forum?id=K3S0uNfUsY

[29] H. Tian, K. Liu, A. K. Kabore, A. Koyuncu, L. Li, J. Klein, and´ T. F. Bissyande, “Evaluating representation learning of code changes´ for predicting patch correctness in program repair,” in Proceedings of the 35th IEEE/ACM international conference on automated software engineering, 2020, pp. 981–992.

[30] Y. Zhao, L. Gong, Z. Huang, Y. Wang, M. Wei, and F. Wu, “Codingptms: How to find optimal code pre-trained models for code embedding in vulnerability detection?” in Proceedings of the 39th IEEE/ACM

International Conference on Automated Software Engineering, 2024, pp. 1732–1744.

[31] Anonymous Authors, “Our website,” https://anonymous.4open.science/ r/CodeEmbedBench-816922/, 2026, anonymous repository.

[32] J. Svajlenko, J. F. Islam, I. Keivanloo, C. K. Roy, and M. M. Mia, “Towards a big data curated benchmark of inter-project code clones,” in 2014 IEEE international conference on software maintenance and evolution. IEEE, 2014, pp. 476–480.

[33] W. Ahmad, S. Chakraborty, B. Ray, and K.-W. Chang, “Unified pre-training for program understanding and generation,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Online: Association for Computational Linguistics, Jun. 2021, pp. 2655–2668. [Online]. Available: https://www.aclweb.org/ anthology/2021.naacl-main.211

[34] T. Mikolov, K. Chen, G. Corrado, and J. Dean, “Efficient estimation of word representations in vector space,” in 1st International Conference on Learning Representations, ICLR 2013, Scottsdale, Arizona, USA, May 2-4, 2013, Workshop Track Proceedings, 2013. [Online]. Available: http://arxiv.org/abs/1301.3781

[35] U. Alon, M. Zilberstein, O. Levy, and E. Yahav, “Code2vec: Learning distributed representations of code,” Proc. ACM Program. Lang., vol. 3, no. POPL, pp. 40:1–40:29, Jan. 2019. [Online]. Available: http://doi.acm.org/10.1145/3290353

[36] U. Alon, S. Brody, O. Levy, and E. Yahav, “code2seq: Generating sequences from structured representations of code,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=H1gKYo09tX

[37] N. D. Bui, Y. Yu, and L. Jiang, “Infercode: Self-supervised learning of code representations by predicting subtrees,” in 2021 IEEE/ACM 43rd International Conference on Software Engineering (ICSE). IEEE, 2021, pp. 1186–1197.

[38] G. Chen, X. Xie, D. Tang, Q. Xin, and W. Liu, “Hedgecode: A multitask hedging contrastive learning framework for code search,” in 2025 IEEE/ACM 47th International Conference on Software Engineering (ICSE). IEEE, 2025, pp. 1857–1868.

[39] C. Liu, S. Lu, W. Chen, D. Jiang, A. Svyatkovskiy, S. Fu, N. Sundaresan, and N. Duan, “Code execution with pre-trained language models,” in Findings of the Association for Computational Linguistics: ACL 2023, A. Rogers, J. Boyd-Graber, and N. Okazaki, Eds. Toronto, Canada: Association for Computational Linguistics, Jul. 2023, pp. 4984–4999. [Online]. Available: https://aclanthology.org/2023.findings-acl.308/

[40] D. Guo, C. Xu, N. Duan, J. Yin, and J. McAuley, “Longcoder: a longrange pre-trained language model for code completion,” in Proceedings of the 40th International Conference on Machine Learning, ser. ICML’23, 2023.

[41] D. Guo, S. Ren, S. Lu, Z. Feng, D. Tang, S. LIU, L. Zhou, N. Duan, A. Svyatkovskiy, S. Fu, M. Tufano, S. K. Deng, C. Clement, D. Drain, N. Sundaresan, J. Yin, D. Jiang, and M. Zhou, “Graphcode{bert}: Pre-training code representations with data flow,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=jLoC4ez43PZ

[42] D. Guo, S. Lu, N. Duan, Y. Wang, M. Zhou, and J. Yin, “UniXcoder: Unified cross-modal pre-training for code representation,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 7212–7225. [Online]. Available: https://aclanthology.org/2022.acl-long.499/

[43] Y. Wang, W. Wang, S. Joty, and S. C. Hoi, “CodeT5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, M.-F. Moens, X. Huang, L. Specia, and S. W.-t. Yih, Eds. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, Nov. 2021, pp. 8696–8708. [Online]. Available: https: //aclanthology.org/2021.emnlp-main.685

[44] R. Li, L. B. allal, Y. Zi, N. Muennighoff, D. Kocetkov et al., “Starcoder: may the source be with you!” Transactions on Machine Learning Research, 2023, reproducibility Certification. [Online]. Available: https://openreview.net/forum?id=KoFOg41haE

[45] B. Roziere, J. Gehring, F. Gloeckle, S. Sootla, I. Gat, X. E. Tan, Y. Adi, J. Liu, R. Sauvestre, T. Remez et al., “Code llama: Open foundation models for code,” arXiv preprint arXiv:2308.12950, 2023.

[46] A. E. Assadi, I. Chung, C. Xiao, R. Solomatin, A. Jha, R. Chand, S. Singh, K. Wang, A. S. Khan, M. M. Nasser et al., “Maeb: Massive audio embedding benchmark,” arXiv preprint arXiv:2602.16008, 2026.

[47] Y. Zhao, L. Gong, H. Zhang, Y. Yu, and Z. Huang, “How to get better embeddings with code pre-trained models? an empirical study,” arXiv preprint arXiv:2311.08066, 2023.

[48] J. Huang, D. Tang, L. Shou, M. Gong, K. Xu, D. Jiang, M. Zhou, and N. Duan, “CoSQA: 20,000+ web queries for code search and question answering,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), C. Zong, F. Xia, W. Li, and R. Navigli, Eds. Online: Association for Computational Linguistics, Aug. 2021, pp. 5690–5700. [Online]. Available: https://aclanthology.org/2021.acl-long.442/

[49] J. Gong, Y. Wu, L. Liang, Z. Zheng, and Y. Wang, “Cosqa+: Enhancing code search dataset with matching code,” arXiv preprint arXiv:2406.11589, 2024.

[50] X. Li, K. Dong, Y. Q. Lee, W. Xia, H. Zhang, X. Dai, Y. Wang, and R. Tang, “Coir: A comprehensive benchmark for code information retrieval models,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2025, pp. 22 074–22 091.

[51] G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” in International Conference on Learning Representations (ICLR), 2017. [Online]. Available: https://openreview. net/forum?id=HJ4-rAVt

[52] J. Hewitt and P. Liang, “Designing and interpreting probes with control tasks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), K. Inui, J. Jiang, V. Ng, and X. Wan, Eds. Hong Kong, China: Association for Computational Linguistics, Nov. 2019, pp. 2733–2743. [Online]. Available: https://aclanthology.org/D19-1275/

[53] M. T. Ribeiro, S. Singh, and C. Guestrin, “” why should i trust you?” explaining the predictions of any classifier,” in Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, 2016, pp. 1135–1144.

[54] S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” Advances in neural information processing systems, vol. 30, 2017.

[55] F. Anselmi, L. Rosasco, and T. Poggio, “On invariance and selectivity in representation learning,” Information and Inference: A Journal of the IMA, vol. 5, no. 2, pp. 134–158, 2016.

[56] K. Lenc and A. Vedaldi, “Understanding image representations by measuring their equivariance and equivalence,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 991–999.

[57] G. Penedo, A. Lozhkov, H. Kydl´ıcek, L. B. Allal, E. Beeching, A. P.ˇ Lajar´ın, Q. Gallouedec, N. Habib, L. Tunstall, and L. von Werra,´ “Codeforces,” https://huggingface.co/datasets/open-r1/codeforces, 2025.

[58] J. Petrik and D. Chuda, “The effect of time drift in source code authorship attribution: Time drifting in source code - stylochronometry,” in Proceedings of the 22nd International Conference on Computer Systems and Technologies, ser. CompSysTech ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 87–92. [Online]. Available: https://doi.org/10.1145/3472410.3472445

[59] Q. Zheng, X. Xia, X. Zou, Y. Dong, S. Wang, Y. Xue, Z. Wang, L. Shen, A. Wang, Y. Li, T. Su, Z. Yang, and J. Tang, “Codegeex: A pre-trained model for code generation with multilingual benchmarking on humaneval-x,” in Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023, pp. 5673–5684.

[60] J. W. Ratcliff and D. E. Metzener, “Pattern matching: The gestalt approach,” Dr. Dobb’s Journal, vol. 13, no. 7, p. 46, 1988.

[61] Voyage AI. (2024, Dec.) voyage-code-3: more accurate code retrieval with lower dimensional, quantized embeddings. Voyage AI Blog, accessed March 26, 2026. [Online]. Available: https://blog.voyageai. com/2024/12/04/voyage-code-3/

[62] D. Kryvosheieva, S. Sturua, M. Gunther, S. Martens, and H. Xiao,¨ “Efficient code embeddings from code generation models,” in NeurIPS 2025 Fourth Workshop on Deep Learning for Code.

[63] OpenAI. (2022, Dec.) New and improved embedding model. [Online]. Available: https://openai.com/index/ new-and-improved-embedding-model/

[64] ByteDance Volcano Engine. (2024) doubao-embedding-large. Volcano Engine documentation, accessed March 26, 2026. [Online]. Available: https://www.volcengine.com/docs/84313/1254568

[65] OpenAI. (2024, Jan.) New embedding models and api updates. [Online]. Available: https://openai.com/index/ new-embedding-models-and-api-updates/

[66] Y. Babakhin, R. Osmulski, R. Ak, G. Moreira, M. Xu, B. Schifferer, B. Liu, and E. Oldridge, “Llama-embed-nemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks,” 2025. [Online]. Available: https://arxiv.org/abs/2511.07025

[67] Y. Zhang, M. Li, D. Long, X. Zhang, H. Lin, B. Yang, P. Xie, A. Yang, D. Liu, J. Lin, F. Huang, and J. Zhou, “Qwen3 embedding: Advancing text embedding and reranking through foundation models,” 2025. [Online]. Available: https://arxiv.org/abs/2506.05176