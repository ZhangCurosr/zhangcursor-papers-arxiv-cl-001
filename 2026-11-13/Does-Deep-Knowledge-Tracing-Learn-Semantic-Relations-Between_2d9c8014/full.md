# Does Deep Knowledge Tracing Learn Semantic Relations Between Knowledge Components?

Kia Karbasi<sup>1</sup>, Mahdi Saiedi<sup>2</sup>, Kevin Hong<sup>1</sup>, Gregory Pottie<sup>1</sup>

<sup>1</sup>University of California, Los Angeles <sup>2</sup>Sharif University of Technology

kiakarbasi@g.ucla.edu, msaiedi.042@gmail.com kevinhong1167@g.ucla.edu, pottie@ee.ucla.edu

Deep Knowledge Tracing (DKT) is a knowledge tracing (KT) model that updates a shared latent state from sequences of interactions, each pairing a knowledge component (KC) with a response. An interaction with one KC can therefore affect predictions for other KCs. We ask: Are the KC relations learned by DKT semantic? We construct text-derived semantic KC representations using nine pipelines to test whether the conclusions hold across alternative ways of constructing those representations.

The first test compares DKT-derived KC relation graphs with semantic relation structures induced by pairwise cosine similarities among the KC representations. Across the evaluated datasets, embedding models, and pipelines, the two structures align only weakly and inconsistently. The second test introduces semantic information into DKT; the pretrained text-derived representations do not consistently outperform the standard DKT baseline.

In the reduced-data experiments, semantic initialization yields a higher test area under the ROC curve (AUC) than the baseline at eight of nine nested sample sizes for ASSISTments 2017. KC-supervised fine-tuned BERT produces small positive AUC differences in some matched comparisons, but the study does not establish their stability across runs. These effects remain limited and inconsistent. The findings suggest that behavioral, temporal, structural, or psychometric signals may also shape DKT’s learned relations. DKT-derived relations are not primarily semantic: the tested text-derived semantic structures do not strongly recover the DKT-derived relation structure.

Keywords: knowledge tracing, semantic embeddings, question text, Deep Knowledge Tracing

## 1. INTRODUCTION

Knowledge tracing (KT) models estimate the probability that a student will answer the next question correctly from the student’s prior interaction history (Corbett and Anderson, 1995; Piech et al., 2015). In datasets such as ASSISTments, each interaction is linked to a question and one or more knowledge components (KCs), and the model must infer latent mastery over time. Bayesian Knowledge Tracing (BKT) does this with an explicit hidden-state probabilistic model, whereas Deep Knowledge Tracing (DKT) updates a shared latent state from sequences of KC-response interactions with recurrent networks. This shared-state design lets us ask which relations among questions and KCs support prediction.

In the original DKT example, responses to x-intercept exercises can update the model’s predicted probability of a correct response to y-intercept exercises (Piech et al., 2015). Because these KCs cover closely related mathematical content, this cross-KC behavior motivates our central question: Are the KC relations learned by DKT semantic?

In this paper, we evaluate the hypothesis that question text carries information about what a student is being asked to do and that DKT-derived KC relations might therefore be semantic. We test that hypothesis under controlled conditions rather than assume that text-derived semantic structure aligns with DKT-derived relations or that semantic initialization improves DKT’s predictive performance.

Educational questions are not arbitrary identifiers. They contain instructions, mathematical language, and cues about the skill being assessed. Modern text representation methods are designed to place semantically similar texts near one another in vector space (Devlin et al., 2019; Reimers and Gurevych, 2019; Gao et al., 2021). One hypothesis is that these semantic representations could provide a better inductive bias than sparse ID-based inputs.

In this study, we use nine pipelines to map question text into pretrained KC-response representations. The alternatives test whether the findings hold across different ways of constructing semantic KC structure.

We assess the semantic-relation question through two complementary tests. The first and most direct test compares the DKT-derived KC relation graph with the semantic KC relation structure induced by pairwise cosine similarities among text-derived KC representations. The second test replaces the default input embedding table in pyKT’s DKT implementation with pretrained embeddings derived from question text. The goal is to determine whether semantic information in question text, when mapped into KC-response representations, improves DKT’s predictive performance.

The principal result is weak structural alignment. DKT-derived KC relations are only weakly aligned with the text-derived semantic structures evaluated in this study. Across the pipelines studied here, pretrained text-derived representations also do not consistently outperform the standard DKT baseline. This result constrains what semantic similarity can explain under this setup and suggests that behavioral, temporal, structural, or psychometric signals may also shape DKT’s learned relations. That interpretation is consistent with later KT work that couples question information to additional structural or contextual signals rather than relying on text alone (Ghosh et al., 2020; Pardos and Dadu, 2017; Wang et al., 2022).

We investigate the following research questions:

• RQ1. Are the KC relations learned by DKT semantic?

• RQ2. Does introducing semantic information improve DKT’s modeling performance?

– RQ2.1. Does semantic initialization improve DKT’s predictive performance in the standard full-data setting?

– RQ2.2. Does semantic initialization improve DKT’s modeling performance when the amount of behavioral data is reduced?

The graph-alignment analysis directly answers RQ1. The full-data and reduced-data comparisons answer RQ2.1 and RQ2.2, respectively. The fine-tuned BERT experiments provide a mechanism analysis under RQ2.

In this paper, we make four contributions. First, we formulate a direct semantic hypothesis for DKT and test whether semantic KC representations recover the relation structure that DKT appears to use. Second, we develop and compare nine pipelines for constructing KC-response embeddings from question text. Third, we evaluate those embeddings both under the standard training setup and under reduced-data regimes. Fourth, under RQ2, we analyze whether aligning the text encoder with KC labels improves DKT’s predictive performance.

The rest of this paper is structured as follows. We first review the background most relevant to the study, then describe the methodology, report the results, and discuss what those results imply for KT.

## 2. BACKGROUND AND RELATED WORK

## 2.1. KNOWLEDGE TRACING AND DKT

DKT updates a shared latent state from sequences of KC-response interactions and therefore implicitly models relationships among KCs. Evidence that a student knows or does not know one KC can affect the model’s prediction concerning another KC.

The original DKT work also introduced an influence-score method for relating KCs based on how a correct response to one KC changes the predicted probability for another (Piech et al., 2015). Later work examined this graph-extraction procedure and proposed a modified repeatedresponse method for extracting exercise-relation graphs (Hong et al., 2025).

Later KT research suggests that the KC relations useful for prediction need not be semantic. DKT’s gains were followed by critiques showing that model behavior can be unstable or hard to interpret, and that preprocessing and benchmark details can materially affect conclusions (Khajah et al., 2016; Xiong et al., 2016; Yeung and Yeung, 2018). More recent architectures often improve predictive performance by modeling memory, recency, attention, question difficulty, or question-level structure more explicitly (Zhang et al., 2017; Pandey and Karypis, 2019; Sonkar et al., 2020; Ghosh et al., 2020). This literature motivates two tests: whether semantic structure corresponds to DKT-derived KC relations and whether semantic initialization improves DKT’s predictive performance without a more elaborate architecture.

Relation-aware KT models address a neighboring prediction problem by supplying, learning, or propagating over relations among concepts and exercises. Prerequisite-driven and structurebased models encode directed or multiple concept relations, graph neural KT treats knowledge structure as a graph, and heterogeneous or hierarchical graph models represent question–skill and exercise relations (Chen et al., 2018; Nakagawa et al., 2019; Tong et al., 2020; Luo et al., 2022; Tong et al., 2022; Duan et al., 2024). Those models use relation structure to improve predictive performance or interpret model behavior. This study instead asks a diagnostic question: whether the relation matrix extracted from baseline DKT by the original influence-score procedure aligns with text-derived KC cosine geometry.

## 2.2. PRETRAINED REPRESENTATIONS OF EDUCATIONAL QUESTIONS

Prior work has augmented KT models with exercise-side information, built exercise-aware architectures, proposed specialized question representations, and pretrained question embeddings for educational prediction (Su et al., 2018; Liu et al., 2019; Yin et al., 2019; Wang et al., 2022). At the same time, KC-imputation work suggests that question text may be insufficient by itself and that surrounding interaction context can be equally important (Pardos and Dadu, 2017).

Together, these findings motivate a controlled test of whether question text improves DKT’s predictive performance.

## 2.3. PRETRAINED EMBEDDINGS AS KT INPUTS

Distributional and contextual embedding methods map text into dense vector spaces in which lexical, topical, and semantic relations are partially preserved (Mikolov et al., 2013; Bojanowski et al., 2017; Pennington et al., 2014; Devlin et al., 2019; Reimers and Gurevych, 2019; Gao et al., 2021). Sentence-level embedding methods are especially relevant here because the study compares full questions and groups of questions rather than isolated tokens. The representation flow is:

question text → question embedding → KC representation → KT input.

If semantic structure were aligned with the dependencies useful for KT, then these dense representations could improve generalization, reduce sparsity, and increase predictive performance when less behavioral data are available. That expectation is consistent with related work. Exercise-enhanced and exercise-aware KT models incorporate richer representations of questions rather than treating them as anonymous IDs (Su et al., 2018; Liu et al., 2019). Questionaware KT architectures likewise treat question representation as a central design choice (Sonkar et al., 2020; Ghosh et al., 2020).

Related educational question-representation work, such as QuesNet, argues that test questions are heterogeneous objects containing both linguistic content and domain logic, which may not be well served by generic off-the-shelf embeddings alone (Yin et al., 2019). More directly, pretraining question embeddings for KT with structural constraints has been shown to improve predictive performance in settings where question–concept relations and difficulty are modeled explicitly (Wang et al., 2022).

At the same time, the literature also suggests reasons this hypothesis might fail. Pardos and Dadu show that for KC imputation, contextual information from neighboring problems can be as predictive as the problem text itself (Pardos and Dadu, 2017). The semantics relevant to educational prediction may therefore be distributed across wording, curriculum structure, and usage context rather than residing only in the text. In ASSISTments-style data, operational KC labels, student behavior, and dataset construction may not align perfectly with pure linguistic similarity.

Accordingly, we use semantic representations as a controlled probe of DKT. We test whether they align with DKT-derived relations and whether semantic initialization improves DKT’s predictive performance.

## 3. METHODOLOGY

## 3.1. DATA SOURCES AND TEXT COVERAGE

This study uses the ASSISTments 2009, 2012, and 2017 KT benchmarks (Abdelrahman et al., 2023; Feng et al., 2009; Prihar et al., 2022). Because semantic KC representations require question text, the analyses also rely on the Problem Bodies resource, which provides text for only a subset of ASSISTments items. Only questions and KCs with matched text receive textderived representations.

The text-linked subset is much smaller than the full interaction log, especially at the question level; this coverage constraint shapes every text-based analysis in the study. Table 1 reports the number of unique questions and KCs in each interaction log together with the corresponding counts in the matched Problem Bodies subset.

Table 1: Question and KC counts in the ASSISTments logs and in the matched Problem Bodies subset used for text-based analyses in this study.
<table><tr><td>Dataset</td><td># Questions</td><td># Questions in PB</td><td>#KCs</td><td># KCs in PB</td></tr><tr><td>ASSISTments 2009</td><td>26,688</td><td>7,836</td><td>123</td><td>76</td></tr><tr><td>ASSISTments 2012</td><td>179,999</td><td>32,792</td><td>265</td><td>162</td></tr><tr><td>ASSISTments 2017</td><td>3,162</td><td>638</td><td>102</td><td>68</td></tr></table>

The ASSISTments 2009 Problem Bodies count treats 311 and 311.0 as two matched KC identifier strings; numeric normalization merges them, so the DKT and RQ1 matrices use 75 modeled KCs. Using the modeled KC counts, the matched-question counts in Table 1 correspond to approximately 104.5, 202.4, and 9.4 questions per KC for ASSISTments 2009, 2012, and 2017, respectively.

All text-based analyses operate on trimmed text-linked datasets. Questions and KCs without matched Problem Bodies text are excluded from the semantic-representation and DKT comparison pipeline. The DKT baselines, semantic-initialization models, reduced-data experiments, BERT downstream experiments, and graph-alignment analyses use the same dataset representation. Throughout this paper, full-data refers to the complete trimmed dataset rather than the original unfiltered interaction log.

When dataset text availability is compared below, largest text-linked question pool and smallest text-linked question pool refer to the absolute matched-question counts in Table 1, not to the proportion of all questions or KCs that are matched.

## 3.2. CONSTRUCTING SEMANTIC KC REPRESENTATIONS

Each pipeline maps the question texts associated with a KC to two response-specific interaction vectors. The construction has three steps. The first is the question representation step, in which raw question text is mapped into a dense vector representation. The second is the KC construction step, in which one or more question-level representations are transformed into a single KC-level vector. The third is the response-state construction step, in which the KC representation is converted into the two response-specific interaction embeddings required by DKT. The initial question-to-embedding step is illustrated in Figure 1.

![](images/6a73b80656c3b227f371b4bfaefbcd8320bbc3e160da3ae589924bef7d9e55fe.jpg)  
Figure 1: Question text is mapped by an embedding model to a dense question vector.

We evaluate sentence-transformer models, commercial embedding APIs, and datasetspecific fine-tuned BERT encoders through nine semantic-representation pipelines. Each pipeline produces the two response-specific vectors required by DKT but combines the three construction steps differently.

## 3.2.1. Embedding Models Used in the Semantic Representation Pipelines

Table 2: Embedding models used in the semantic representation pipelines. Dimensionalities are those used in the evaluated configurations.
<table><tr><td>Family</td><td>Model</td><td>Dimensionality</td><td>Dataset Scope</td></tr><tr><td>OpenAI</td><td>text-embedding-3-smal1</td><td>1,536</td><td>All three</td></tr><tr><td>OpenAI</td><td>text-embedding-3-large</td><td>3,072</td><td>All three</td></tr><tr><td>OpenAI</td><td>text-embedding-ada-002</td><td>1,536</td><td>All three</td></tr><tr><td>Sentence- Transformer</td><td>all-MiniLM-L6-v2</td><td>384</td><td>All three</td></tr><tr><td>Sentence- Transformer</td><td>all-mpnet-base-v2</td><td>768</td><td>All three</td></tr><tr><td>Sentence-</td><td>multi-qa-mpnet-base-dot-v1</td><td>768</td><td>All three</td></tr><tr><td>Transformer Sentence-</td><td>al1-MiniLM-L12-v2</td><td>384</td><td>All three</td></tr><tr><td>Transformer Sentence-</td><td>paraphrase-MiniLM-L6-v2</td><td>384</td><td>All three</td></tr><tr><td>Transformer</td><td></td><td></td><td></td></tr><tr><td>Cohere</td><td>embed-v4.0</td><td>1,536</td><td>All three</td></tr><tr><td>Cohere</td><td>embed-english-v3.0</td><td>1,024</td><td>All three</td></tr><tr><td>Cohere Fine-tuned BERT</td><td>embed-multilingual-v3.0 ASSISTments 2009 KC classifier</td><td>384</td><td>All three</td></tr><tr><td></td><td></td><td>768</td><td>2009 only</td></tr><tr><td>Fine-tuned BERT</td><td>ASSISTments 2012 KC classifier</td><td>768</td><td>2012 only</td></tr></table>

The pipeline design supports the same nine constructions across embedding backbones while accommodating backbone-specific dimensionality and context-window limits.

## 3.3. NINE REPRESENTATION PIPELINES

The nine pipelines form a conceptual 3 × 3 design. The KC-construction dimension treats a KC as a semantic prototype or centroid (average), a concrete exemplar (sample), or a holistic document (stuff window). The response-state dimension treats response state as polarity along a representational axis (mirror), a separately appended attribute (stack antonym), or a condition inserted into the text before embedding (add words). Figure 2 maps the nine pipelines to these two dimensions.

The same pipelines can also be grouped by their execution order. In this notation, student action (SA) denotes the response state attached to a KC:

1. Question → KC → SA pipelines (Pipelines 1–4): raw question text is embedded first, KC embeddings are then created either by averaging all questions in a KC or by sampling a single representative question, and finally correctness is added using either a mirror transformation or a concatenated antonym embedding.

<table><tr><td rowspan=1 colspan=1>KC construction× response state</td><td rowspan=1 colspan=1>Mirrorpolarity</td><td rowspan=1 colspan=1>Stack antonymappended attribute</td><td rowspan=1 colspan=1>Add wordsconditioned text</td></tr><tr><td rowspan=1 colspan=1>Prototypeaverage</td><td rowspan=1 colspan=1>P1</td><td rowspan=1 colspan=1>P2</td><td rowspan=1 colspan=1>P7</td></tr><tr><td rowspan=1 colspan=1>Exemplarsample</td><td rowspan=1 colspan=1>P3</td><td rowspan=1 colspan=1>P4</td><td rowspan=1 colspan=1>P8</td></tr><tr><td rowspan=1 colspan=1>Holistic documentstuff window</td><td rowspan=1 colspan=1>P5</td><td rowspan=1 colspan=1>P6</td><td rowspan=1 colspan=1>P9</td></tr></table>

Figure 2: Conceptual 3 × 3 design of the nine semantic-representation pipelines.

2. KC → Question → SA pipelines (Pipelines 5–6): each KC is represented directly by concatenating as many of its question texts as will fit in a model-dependent context window and embedding that single stuffed text. Correctness is then added using either the mirror or stack-antonym transformation.

3. SA → Question → KC pipelines (Pipelines 7–8): correctness is injected before embedding by prefixing each question text with repeated CORRECT or INCORRECT tokens. The modified questions are then embedded and aggregated into separate correct and incorrect KC vectors using either averaging or single-question sampling.

4. SA → KC → Question pipeline (Pipeline 9): response state is inserted into each question text, the modified texts are concatenated into separate response-specific KC windows, and each window is embedded once.

Table 3: Nine evaluated pipelines for generating response-specific KC embeddings. SA denotes student action, the response state attached to a KC.
<table><tr><td></td><td>Pipeline Execution order</td><td>text operation</td><td>Question-Question representation step</td><td>KC step</td><td>Response-state construction construction step</td></tr><tr><td>P1</td><td>SA</td><td>question → KC → raw question embed raw question text</td><td></td><td>average</td><td>mirror</td></tr><tr><td>P2</td><td>SA</td><td>question → KC → raw question embed raw question text</td><td></td><td>average</td><td>stack antonym</td></tr><tr><td>P3</td><td>SA</td><td>question → KC → raw question embed raw question text</td><td></td><td>sample single question</td><td>mirror</td></tr><tr><td>P4</td><td>SA</td><td>question → KC → raw question embed raw question text</td><td></td><td>sample single question</td><td>stack antonym</td></tr><tr><td>P5</td><td>SA</td><td>KC → question → raw question embed stuffed KC text text</td><td></td><td>stuff window</td><td>mirror</td></tr><tr><td>P6</td><td>SA</td><td>KC → question → raw question embed stuffed KC text text</td><td></td><td>stuff window</td><td>stack antonym</td></tr><tr><td>P7</td><td>KC</td><td>SA → question → add words embed modified question</td><td></td><td>average</td><td>encoded in text</td></tr><tr><td>P8</td><td>KC</td><td>SA → question → add words embed modified question</td><td></td><td>sample single</td><td>encoded in text</td></tr><tr><td>P9</td><td>SA → KC → question</td><td></td><td>add words embed stuffed modified KC text</td><td>question stuff window</td><td>encoded in text</td></tr></table>

Two construction details matter for interpreting these pipelines. First, the mirror strategy keeps one response-specific KC vector as the base embedding e and maps the other to −e. Second, the stack\_antonym strategy concatenates the KC vector with a separately embedded response marker, doubling the dimensionality of the final response-specific input. By contrast, the add\_words pipelines do not apply a later response-state transformation at the KC level; response state is already embedded in the modified question text itself. In the experiments reported here, those prefixes are repeated twice before embedding.

Stuff-window construction first chooses the model-dependent context window, then concatenates as many question texts as fit, and embeds the resulting document once. OpenAI models use a larger window than the sentence-transformer models. Table 3 summarizes the nine combinations.

Averaging pools the question embeddings associated with a KC by arithmetic mean. Sampling selects one representative question and uses its embedding as the KC representation. Stuffing concatenates as many associated question texts as fit in the model-dependent window and embeds the document once.

Mirroring maps the two response states to opposite directions along the same representational axis. Stack-antonym concatenates a KC embedding with a separately embedded response marker. Add-words injects response state into each question text before embedding. Together, these families test whether the conclusions depend on where response state enters representation construction.

## 3.4. DKT-DERIVED KC RELATION GRAPHS

The relation graphs were produced for this study by applying the original DKT influence-score method to the same baseline DKT models whose predictive results are reported in Table 5. Following Piech et al. (Piech et al., 2015), we assign an influence score $J _ { i j }$ to each directed source–target pair:

$$
J _ { i j } = \frac { y ^ { ( j | i ) } } { \sum _ { k } y ^ { ( j | k ) } } ,
$$

where $y ^ { ( j | i ) }$ is the correctness probability assigned by the trained DKT model to target KC j on the second timestep, given a correct response to source KC i on the first timestep. The denominator sums over every source KC k in the same trimmed text-linked KC set. We use the resulting complete weighted relation matrix over that set as the DKT-derived comparison target for RQ1.

## 3.5. SEMANTIC RELATION STRUCTURE AND ALIGNMENT METRICS

We assess the semantic-relation question by comparing two structures over the same ordered set of text-linked KCs: the DKT-derived KC relation matrix and the geometry induced by the pretrained KC embeddings. For the RQ1 semantic graph, we use the first N rows of each 2Nrow pretrained interaction tensor to obtain one vector per KC. Pairwise cosine similarity over these KC vectors produces a symmetric semantic relation matrix.

Each dataset-specific KC index (keyid2idx.json) orders the trimmed text-linked KCs. The DKT-derived relation matrix and semantic cosine-similarity matrix are compared over exactly that same ordered set. KCs outside the trimmed text-linked set are excluded from the comparison.

The extracted DKT relation matrix is directed, whereas cosine similarity is symmetric. Reciprocal DKT weights are averaged to form a symmetric weighted matrix, and the diagonal is set to zero.

Pearson and Spearman correlations compare corresponding strict-upper-triangle entries of the symmetrized DKT matrix and the semantic cosine-similarity matrix; Spearman first replaces the values by their ranks. For top-k edge overlap, each matrix contributes its k highest-weight strict-upper-triangle edges at 1%, 5%, 10%, and 20% of the available edges, and overlap is normalized as $| A \cap B | / k$ . Exactly k entries are selected without expanding boundary ties.

As a non-text reference, we also construct a symmetric KC co-exposure matrix derived from the same training sequences used to fit DKT, counting each unordered KC pair at most once per student sequence and setting the diagonal to zero. The comparator collapses response order and correctness into student-level co-exposure and is compared with the DKT-derived matrix over the same KC set, ordering, strict-upper-triangle entries, and top-k conventions used for the semantic comparisons.

We use weak alignment descriptively: the reported correlations and edge overlaps are small in absolute value. We do not use inferential significance testing to make claims about graph recovery, and we do not estimate how stable the extracted DKT relation matrices are across retraining runs. The alignment magnitudes should therefore be read as descriptive comparisons rather than calibrated recoverability estimates.

## 3.6. INTRODUCING SEMANTIC INFORMATION INTO DKT

The baseline used in this study is the DKT model provided by the pyKT library (Liu et al., 2023). Each student interaction is represented as a discrete KC-response state and indexes an embedding lookup table. Let $i _ { t }$ denote the interaction index assigned to the KC-response pair $\left( \boldsymbol { c } _ { t } , \boldsymbol { r } _ { t } \right)$ . For N KCs, the embedding table contains one row for each of the 2N possible KCresponse states.

In the standard pyKT DKT setup, this interaction embedding matrix is randomly initialized and learned during training. If $E \in \mathbb { R } ^ { 2 N \times d _ { \mathrm { b a s e } } }$ denotes the embedding table and $e _ { i _ { t } }$ its $i _ { t } { \cdot } \mathrm { t h }$ row, then the recurrent model receives

$$
x _ { t } = e _ { i _ { t } } , \qquad E \sim \mathrm { r a n d o m \ i n i t i a l i z a t i o n } ,
$$

where $d _ { \mathrm { b a s e } }$ is the baseline embedding dimension. The pyKT DKT configuration used in this study defaults to $d _ { \mathrm { b a s e } } = 2 0 0$ and zero L2 weight decay. The baseline comparison is therefore between a randomly initialized learned interaction-embedding table and a pretrained interactionembedding table with the same role in the model.

Our modification replaces that randomly initialized lookup table with pretrained interaction embeddings constructed by the evaluated pipeline variants. Let $\tilde { E } \in \mathbb { R } ^ { 2 \tilde { N } \times d _ { \mathrm { s e m } } }$ denote the pretrained matrix whose rows are the response-specific KC embeddings generated from question text. The modified model uses

$$
x _ { t } = \tilde { e } _ { i _ { t } } , \qquad { \tilde { E } } = \left[ { \begin{array} { c } { { \tilde { e } _ { 1 } } } \\ { \vdots } \\ { { \tilde { e } _ { 2 N } } } \end{array} } \right] ,
$$

where each row is supplied by one of the nine semantic-representation pipelines. The pretrained matrix sets the lookup-table dimensionality and initializes its weights.

Each pipeline therefore supplies two response-specific vectors per KC to populate the 2Nrow table. Figure 3 depicts this shared replacement.

![](images/55109eaae3ed83a88f68557bec1c249dcbd4013527a2e2e86e473e56787a7ed8.jpg)

![](images/0ca58fff12dd26a97ee5b71a65badb45578f944c0b18592a1106b7df4cc95261.jpg)  
(b) Baseline DKT with randomly initialized interaction embeddings  
Figure 3: Pretrained embedding initialization. The panels compare the text-derived KC-response initialization with the baseline DKT interaction representation based on randomly initialized embeddings.

We evaluate predictive performance with area under the ROC curve (AUC) and accuracy.

## 3.7. ARCHITECTURE AND ROBUSTNESS VARIANTS

We evaluate several architecture variants as robustness analyses for RQ2.1. Each retains pretrained embedding initialization while changing a different part of DKT.

PROJECTION BOTTLENECK The projection-bottleneck variant inserts a learned projection layer before the recurrent model. As shown in Figure 4, this adds a learned compression stage between the pretrained interaction embedding and the recurrent computation. If $\boldsymbol { x } _ { t } ~ \in ~ \mathbb { R } ^ { d _ { \mathrm { s e m } } }$ denotes the pretrained interaction embedding, the projected representation is

$$
z _ { t } = W _ { \mathrm { p r o j } } x _ { t } + b _ { \mathrm { p r o j } } ,
$$

where $W _ { \mathrm { p r o j } } \in \mathbb { R } ^ { d _ { \mathrm { p r o j } } \times d _ { \mathrm { s e m } } }$ . In the bottleneck setting, $d _ { \mathrm { p r o j } } < d _ { \mathrm { s e m } } .$ , so the projection compresses the pretrained representation before sequence modeling. This serves two purposes. First, it adapts high-dimensional pretrained embeddings to the dimensionality expected by the recurrent model. Second, it forces the model to compress the semantic representation into a smaller space tailored to the prediction task, which may help filter irrelevant semantic variation.

Table 2 reports each backbone’s semantic dimension. Mirror and add-words pipelines retain that width for each response-specific input, whereas stack-antonym concatenates two backbonewidth vectors and therefore doubles it.

![](images/ccfaf7b98577ca83de191eae1cebb11547750a179a180434be6fc28d340592b2.jpg)  
Figure 4: Projection bottleneck applied to the pretrained-input DKT model.

REGULARIZATION The regularization variants tune dropout and, where configured, L2 weight decay. The 2009 direct-initialization condition is a dropout-regularized configuration; L2 weight decay is not part of the default pyKT DKT setting. These conditions test whether stronger regularization improves stability or generalization with high-dimensional pretrained inputs.

ADAPTIVE RECURRENT CAPACITY The adaptive-capacity variant, illustrated in Figure 5, changes the recurrent block rather than semantic input construction. We vary hidden size, recurrent depth, and dropout across configurations. In abstract form, the recurrent update becomes

$$
h _ { t } = \mathrm { L S T M } _ { \theta } ( u _ { t } , h _ { t - 1 } ) ,
$$

where $u _ { t } = x _ { t }$ in the direct-input case and $u _ { t } = z _ { t }$ in the projected-input case, while θ is selected from recurrent configurations with different hidden widths, depths, and dropout settings.

![](images/4d96356d9640077e4ff2d986af5bce90f95530bc836f9e494044d7ef7f121c71.jpg)  
Figure 5: Recurrent-capacity configurations in the pretrained-input DKT model.

SEMANTIC OUTPUT LAYER The final architecture variant is the Semantic Output Layer. As illustrated in Figure 6, this variant changes the prediction head so that the hidden state is scored against semantic KC representations instead of being sent directly to a conventional classifier. In the standard DKT setup, the hidden state is mapped directly to output logits over KCs by a learned linear layer. In the semantic-output variant, the hidden state is first projected into the same space as the projected pretrained KC or question embeddings, and prediction is then based on a dot product between the projected hidden state and those stored semantic representations. Writing $h _ { t }$ for the recurrent hidden state, the architecture is

$$
\hat { h } _ { t } = W _ { h } h _ { t } + b _ { h } , \qquad \bar { e } _ { k } = W _ { \mathrm { p r o j } } e _ { k } , \qquad s _ { t , k } = \hat { h } _ { t } ^ { \top } \bar { e } _ { k } ,
$$

where $e _ { k }$ is the stored pretrained representation associated with KC k, and $W _ { \mathrm { p r o j } }$ is the same projection family used on the input side.

Stored semantic targets are projected once into output space. One variant averages the two response-specific KC representations; another uses a single KC representation. This replaces the learned linear KC-output head with dot-product scores between the hidden state and stored semantic KC vectors. This reuse of input-embedding geometry at the output layer is analogous to weight tying in NLP language models, but the analogy is limited to the shared geometry: this DKT head scores KCs rather than language-model tokens.

![](images/5226b7d118793d248cf23f059c4efc6650861fcb3573292710d5d05eeb109445.jpg)  
Figure 6: Semantic output layer in the pretrained-input DKT model.

## 3.8. FULL-DATA EVALUATION AND SELECTION

For RQ2.1, the full-data setting uses all interactions in the trimmed text-linked dataset defined in Table 1. The pyKT preprocessing procedure assigns 20% of the processed sequences to the test set and divides the remaining 80% into five folds; fold 0 is used for validation and the other four folds for training. Each configuration combines a dataset, embedding model, representation pipeline, architecture variant, and any variant-specific settings. Across the full-data experiments, we evaluated approximately 1,065 distinct configurations; Supplementary Table 13 summarizes the 21 experiment exports. This count refers to the RQ2.1 full-data modeling sweep, not to the RQ1 graph-alignment sweep. Within each experimental condition, the candidate with the highest validation AUC is selected; Table 5 then reports its test AUC and test accuracy. Supplementary Tables 10–12 report the corresponding per-dataset pipeline rows, with each pipeline represented by the configuration selected by validation AUC.

## 3.9. REDUCED-DATA PROTOCOL

To test RQ2.2, we repeat the comparison with reduced datasets. This asks whether semantic representations provide a stronger inductive bias when behavioral data are limited.

The data-efficiency analysis is carried out on ASSISTments 2009, 2012, and 2017 by creating nested subsamples of each complete trimmed text-linked base dataset at 10%, 20%, . . . , 90% of that base dataset’s row count. In this protocol, one row denotes one student interaction consisting of a KC and response. The sampling procedure is designed to preserve skill coverage: for each dataset, a mandatory set of rows is first selected so that every KC appearing in the complete trimmed base dataset appears at least once in the sampled dataset, and the remaining rows are then randomly permuted with a fixed seed. The resulting samples are nested, so the 10% sample is a subset of the 20% sample, the 20% sample is a subset of the 30% sample, and so on. Each sampled dataset is then processed with the same pyKT train/validation/test split procedure used above; the partitions are regenerated at each percentage rather than held fixed across percentages.

Two conditions are compared on these sampled datasets. The first is the baseline pyKT DKT model without pretrained semantic embeddings. The second initializes the model with a pretrained interaction matrix. The semantic representations are constructed from the complete trimmed text-linked question/KC resource. Across reduced-data percentages, the entire behavioral dataset is subsampled before splitting while the same external semantic matrices are reused. For ASSISTments 2009, all nine pipelines are evaluated; for ASSISTments 2012 and 2017, the reduced-data experiments use Pipeline 3.

We tune each sampled condition with a dataset-specific hyperparameter sweep managed in Weights & Biases (W&B). Final reporting follows the same validation-selection rule used in the full-data benchmark runs: for each condition, the run with the highest validation AUC is selected, and the selected run’s test AUC and test accuracy are reported. For ASSISTments 2009, pipeline selection is nested inside that procedure, so the selected pretrained result at each sample size is the pipeline/run configuration with the highest validation AUC among the available semantic alternatives. Model selection is therefore based on validation performance rather than on the test set.

ASSISTments 2009 selects among nine pipelines, whereas ASSISTments 2012 and 2017 use fixed Pipeline 3. Because the search breadth differs by dataset, we treat the reduced-data patterns as descriptive rather than as controlled cross-dataset comparisons. The curves are based on one nested sampling path, so they do not show how results vary across alternative samples.

## 3.10. KC-SUPERVISED ENCODER ANALYSIS

We trained dataset-specific BERT KC classifiers for all three datasets. The ASSISTments 2009 and 2012 encoders were used for downstream DKT initialization. We retained the ASSISTments 2017 classifier as a diagnostic because it had both a small filtered training set and weak held-out classification performance. These models differ from the general-purpose encoders in Table 2 because they are explicitly trained to predict KC labels from problem text rather than relying on general semantic similarity alone.

Fine-tuning is formulated as a multiclass text-classification problem in which the input is the raw question text and the target is the numeric KC index associated with that question. For each dataset, the dataset question–KC mapping (questions\_with\_kc.csv) supplies the question–KC pairs. Rows with missing text or skill identifiers are dropped, and KC classes with fewer than two examples are removed before the remaining examples are split independently for each dataset into stratified train, validation, and test partitions with a 70/15/15 ratio and random seed 42.

All fine-tuning experiments start from bert-base-uncased (Devlin et al., 2019). The model is instantiated as BertForSequenceClassification with the number of output labels set to the number of valid KC classes in the dataset and with hidden-state outputs enabled so that the encoder can later be reused for embedding extraction. Training uses AdamW with learning rate $2 \times 1 0 ^ { - 5 }$ , weight decay 0.01, warmup over the first 10% of optimization steps, training batch size 16, evaluation batch size 32, maximum sequence length 128, and up to 10 epochs with early stopping based on validation loss using a patience of 3. Because the KC distributions are highly imbalanced, the loss is class-weighted cross-entropy with weights inversely proportional to class frequency in the training split. Gradient norms are clipped to 1.0, and the checkpoint with the lowest validation loss is kept.

After fine-tuning, the classification head is discarded and embeddings are extracted from the final-layer [CLS] representation, yielding a 768-dimensional vector for each question text. These per-question vectors are then passed through the same nine semantic-representation pipelines used for the general-purpose embedding models. For ASSISTments 2009 and 2012, the downstream tensors are constructed for the same trimmed text-linked KC sets used in the DKT comparisons and ordered by each dataset’s KC index (keyid2idx.json).

The ASSISTments 2009 and 2012 tensors are loaded into DKT as pretrained interaction embeddings. Downstream BERT initialization is limited to ASSISTments 2009 and 2012, and no cross-dataset transfer is performed.

## 4. RESULTS

## 4.1. RQ1: ARE THE KC RELATIONS LEARNED BY DKT SEMANTIC?

The three DKT-derived target graphs serve as dataset-specific comparison targets (Figure 7).

Their structures differ across datasets; the quantitative comparisons below test how well each dataset-specific semantic graph recovers them. ASSISTments 2012 is the dataset for which the embedding models achieve the highest, although still weak, agreement.

For RQ1, we compare DKT-derived and semantic graphs across the evaluated datasets, embedding models, pipelines, and architecture variants. Supplementary Table 9 defines the comparison sets and aggregation units. Table 4 reports the highest-Pearson configuration per dataset. ASSISTments 2012 is highest: embed\_english\_v3.0 with Pipeline 3 reaches Pearson 0.0894 and top-10% edge overlap 0.1571. ASSISTments 2009 is weaker, with bert\_ft\_ assist2009 and Pipeline 5 at Pearson 0.0573. ASSISTments 2017 is weakest, with all\_ MiniLM\_L6\_v2 and Pipeline 5 at Pearson 0.0247. All three semantic Pearson correlations in Table 4 are positive but small, and none indicates strong recovery of the DKT-derived graph.

![](images/7b254325c71fd0a1d6405c61bce832acc6d284a270dbf87f1bf1c33a4b7a0ade.jpg)  
Figure 7: DKT-derived target matrices for RQ1. Each panel shows the symmetrized, zerodiagonal complete weighted relation matrix used for the RQ1 comparison over the corresponding trimmed text-linked KC set. KCs follow the numeric index order in that dataset’s keyid2idx.json. Color normalization is panel-specific; the displayed colorbar corresponds to the ASSISTments 2017 panel.

Table 4: Highest-Pearson dataset-level alignment results between semantic KC representations and DKT-derived KC graphs. Each row is selected by Pearson correlation within that dataset’s evaluated configurations. Spearman and top-10% overlap are reported for the same Pearsonselected configuration. They are not selected separately by those metrics. Boldface marks the highest displayed value in each metric column.
<table><tr><td>Dataset</td><td>Embedding model</td><td>Pipeline</td><td>Pearson</td><td>Spearman</td><td>Top-10% overlap</td></tr><tr><td>ASSISTments 2009</td><td>bert_ft_assist2009</td><td>5</td><td>0.0573</td><td>0.0276</td><td>0.1151</td></tr><tr><td>ASSISTments 2012</td><td>embed_english_v3.0</td><td>3</td><td>0.0894</td><td>0.0782</td><td>0.1571</td></tr><tr><td>ASSISTments 2017</td><td>all_MiniLM_L6_v2</td><td>5</td><td>0.0247</td><td>0.0114</td><td>0.1404</td></tr></table>

As a non-text reference, training-sequence KC co-exposure aligns more strongly with the DKT-derived graph than the strongest semantic configuration on ASSISTments 2009 (Pearson 0.1590 versus 0.0573; Spearman 0.0852 versus 0.0276) and ASSISTments 2012 (Pearson 0.1457 versus 0.0894; Spearman 0.1433 versus 0.0782). The advantage is metric-dependent: top-10% edge overlap favors the comparator on ASSISTments 2009 (0.2086 versus 0.1151) but is essentially tied on ASSISTments 2012 (0.1556 versus 0.1571). On ASSISTments 2017, the comparator is negative on both correlation metrics (Pearson −0.0322, Spearman −0.0469). The comparator is a single unselected matrix per dataset, whereas the semantic values are maxima selected by Pearson across the alignment sweep.

Some embedding models align slightly better than others. Appendix Table 14 reports embedding-model means for the five models shown, with each model aggregated over its evaluated dataset–pipeline configurations. Within these displayed rows, bert\_ft\_assist2012 has the highest mean Pearson correlation (0.0321), followed by text-embedding-3- large; even the highest mean remains small.

The highest-Pearson configurations use bert\_ft\_assist2009 on ASSISTments 2009, embed\_english\_v3.0 on ASSISTments 2012, and all\_MiniLM\_L6\_v2 on ASSISTments 2017. These dataset-specific rankings do not identify one embedding model that consistently aligns with the relation structure extracted from DKT.

Appendix Table 15 reports pipeline means. Pipeline 5 has the highest mean Pearson correlation, followed by Pipelines 9 and 3, but none produces strong graph recovery.

Pipeline 5 has the highest alignment for ASSISTments 2009 and 2017 in the RQ1 alignment sweep, whereas Pipeline 3 is highest for ASSISTments 2012. Pipeline effects are therefore dataset-dependent and do not identify a universally successful semantic construction. Appendix Figure 12 shows that pipeline choice changes cosine-similarity structure without producing a common DKT-like organization.

ASSISTments 2012 contains the highest-Pearson configuration in the RQ1 alignment sweep. It also has the largest text-linked question pool in Table 1. A larger absolute pool of matched questions may help form more stable KC neighborhoods, but the observed alignment remains partial.

Figure 8 places the semantic and DKT-derived matrices for the highest-Pearson configuration in the RQ1 alignment sweep side by side: ASSISTments 2012 with embed\_english\_v3.0 and Pipeline 3. Because the panels use independent KC orders, the figure is intended to show overall matrix structure rather than cellwise agreement.

Figure 9 shows the corresponding direct-agreement diagnostics. The displayed coefficients are Pearson 0.089 and Spearman 0.078. The trend is positive but diffuse, and the highest-weight DKT edges are only partially recovered by the semantic graph.

![](images/f7117878255e4e735a1eb32d928e2dfecbef199f7db29ad7cce28329f1be1944.jpg)  
(a) Embedding cosine-similarity matrix

![](images/dda757c0a59ff397d3f117a28713d4a7d8bbfd67e346033a3c4005738973c361.jpg)  
(b) DKT-derived relation matrix (independently ordered)  
Figure 8: Qualitative side-by-side view for the highest-Pearson ASSISTments 2012 configuration in the RQ1 alignment sweep (embed\_english\_v3.0, Pipeline 3). The panels use independent KC orders and color scales, so the figure compares overall structure rather than matched cells.

![](images/445c9bacd8d74b1d90e092fbde1813048fca1c4fca1f50576f82031fc927004f.jpg)  
(a) Pairwise relation agreement

![](images/4cc818d80de28e6a3f5f2189bf4282ab671e5a73911b0dec6b939c67ba35a916.jpg)  
(b) Top-k edge overlap  
Figure 9: Direct-agreement diagnostics for the Pearson-selected ASSISTments 2012 configuration in Table 4. The pairwise panel reports Pearson (0.089) and Spearman (0.078); the coefficients are descriptive.

These comparisons answer RQ1 directly: semantic KC representations align with DKTderived KC graphs only weakly and inconsistently. Alignment is highest but partial in ASSISTments 2012, lowest in ASSISTments 2017, and intermediate in ASSISTments 2009. Embedding-model and pipeline choices shift the magnitude but not the conclusion: semantic similarity does not strongly recover the relation structure captured by the original DKT influence-score procedure.

## 4.2. RQ2.1: FULL-DATA MODELING PERFORMANCE

The reported validation-selected semantic configurations do not exceed the baseline DKT test AUC on any dataset, although some produce localized gains in test accuracy; the AUC shortfalls range from 0.0011 to 0.0240, and these descriptive comparisons do not establish a consistent advantage for semantic initialization.

Table 5 summarizes the validation-selected configuration for each full-data experimental condition in the study. Each row reports the associated test AUC and test accuracy, and the grouping labels follow the methodological variants introduced earlier. Supplementary Table 13 summarizes the 21 experiment exports represented in the RQ2.1 full-data sweep.

This result does not imply that language is irrelevant to KT. It shows that the semantic signal encoded by these pipelines, on its own, does not capture the dependencies that most strongly govern DKT predictions.

## 4.2.1. Pipeline Comparison

Appendix Tables 10–12 compare the semantic variants, and Table 5 provides the standard DKT baseline.

No pipeline is best across all three datasets. On ASSISTments 2009, Pipeline 1 achieved the highest semantic test AUC (0.8567) while Pipeline 4 achieved the highest semantic test accuracy (0.7870). On ASSISTments 2012, Pipelines 2, 3, and 6 tie for the highest reported semantic test AUC (0.7358), while Pipeline 7 achieved the highest semantic test accuracy (0.7382). On ASSISTments 2017, Pipeline 3 achieved the highest semantic test AUC (0.7207) and accuracy (0.6811). Pipeline results vary by dataset, but none of the reported semantic pipelines exceeds the baseline DKT test AUC on any dataset.

Table 5: Combined full-data results across datasets. For each experimental condition, the reported row is the validation-selected configuration, together with its test AUC and test accuracy. Boldface indicates the highest value in each dataset block at the reported precision. Across the reported validation-selected semantic configurations, none exceeded the baseline test AUC on any dataset, although some configurations show localized accuracy gains. Pipe. denotes pipeline, SemOut the semantic-output condition, and Acc. accuracy.
<table><tr><td>Dataset Variant</td><td></td><td>Embedding</td><td></td><td>Pipe. SemOut</td><td>Test AUC</td><td>∆AUC</td><td>Test Acc.</td></tr><tr><td>ASSISTments 2009</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2009</td><td>Baseline DKT</td><td></td><td>1</td><td>No</td><td>0.8582</td><td>+0.0000</td><td>0.7851</td></tr><tr><td>2009</td><td>Pretrained Embedding text-embedding- Initialization</td><td>3-small</td><td>P1</td><td>No</td><td>0.8550</td><td>-0.0031</td><td>0.7851</td></tr><tr><td>2009</td><td>Pretrained Embedding text-embedding- Initialization + Regu- 3-small larization</td><td></td><td>P1</td><td>No</td><td>0.8567</td><td>-0.0015</td><td>0.7865</td></tr><tr><td>2009</td><td>Projection Bottleneck</td><td>text-embedding- 3-large</td><td>P4</td><td>No</td><td>0.8552</td><td>-0.0030</td><td>0.7870</td></tr><tr><td>2009</td><td>Adaptive recurrent ca- embed-v4.0 pacity</td><td></td><td>P6</td><td>No</td><td>0.8342</td><td>-0.0240</td><td>0.7708</td></tr><tr><td>2009</td><td>Semantic Layer</td><td>Output text-embedding- 3-small</td><td>P1</td><td>Yes</td><td>0.8522</td><td>-0.0060</td><td>0.7811</td></tr><tr><td>2009</td><td>Semantic Layer + Bottleneck</td><td>Output bert_ft_ Projection assist2009</td><td>P6</td><td>Yes</td><td>0.8349</td><td>-0.0233</td><td>0.7727</td></tr><tr><td>ASSISTments 2012</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2012 Baseline DKT</td><td></td><td></td><td>1</td><td>No</td><td>0.7370</td><td>+0.0000</td><td>0.7437</td></tr><tr><td>2012</td><td>Pretrained Embedding text-embedding-</td><td></td><td>P2</td><td>No</td><td>0.7265</td><td>-0.0105</td><td>0.7344</td></tr><tr><td>2012</td><td>Initialization Projection Bottleneck</td><td>3-large text-embedding-</td><td>P6</td><td>No</td><td>0.7351</td><td>-0.0019</td><td>0.7373</td></tr><tr><td>2012</td><td>Projection Bottleneck</td><td>3-large embed-v4.0</td><td>P6</td><td></td><td></td><td></td><td>0.7367</td></tr><tr><td>2012</td><td>Adaptive recurrent ca- text-embedding-</td><td></td><td>P1</td><td>No No</td><td>0.7353 0.7317</td><td>-0.0016 -0.0053</td><td>0.7365</td></tr><tr><td>2012</td><td>pacity Semantic</td><td>3-large Output text-embedding-</td><td>P1</td><td>Yes</td><td>0.7356</td><td>-0.0014</td><td>0.7376</td></tr><tr><td>2012</td><td>Layer Semantic</td><td>3-large Output text-embedding-</td><td>P6</td><td>Yes</td><td>0.7358</td><td>-0.0011</td><td>0.7371</td></tr><tr><td></td><td>Layer + Bottleneck</td><td>Projection 3-large</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>ASSISTments 2017</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>2017</td><td>Baseline DKT Pretrained Embedding text-embedding-</td><td></td><td></td><td>No</td><td>0.7246</td><td>+0.0000</td><td>0.6889</td></tr><tr><td>2017</td><td>Initialization</td><td>3-small</td><td>P3</td><td>No</td><td>0.7207</td><td>-0.0039</td><td>0.6811</td></tr><tr><td>2017</td><td>Projection Bottleneck</td><td>text-embedding- 3-small</td><td>P4</td><td>No</td><td>0.7196</td><td>-0.0049</td><td>0.6808</td></tr><tr><td>2017</td><td>Projection Bottleneck with Regularization</td><td>text-embedding- 3-small</td><td>P6</td><td>No</td><td>0.7129</td><td>-0.0116</td><td>0.6758</td></tr><tr><td>2017</td><td>Adaptive recurrent ca- pacity</td><td>- text-embedding- 3-small</td><td>P6</td><td>No</td><td>0.7197</td><td>-0.0049</td><td>0.6803</td></tr><tr><td>2017</td><td>Semantic Layer</td><td>Output text-embedding- 3-large</td><td>P1</td><td>Yes</td><td>0.7195</td><td>-0.0051</td><td>0.6785</td></tr><tr><td>2017</td><td>Semantic Layer</td><td>Output text-embedding- 3-large</td><td>P1</td><td>Yes</td><td>0.7193</td><td>-0.0053</td><td>0.6798</td></tr></table>

## 4.3. RQ2.2: REDUCED-DATA MODELING PERFORMANCE

For RQ2.2, the favorable ASSISTments 2017 trend suggests that semantic initialization may improve predictive performance in some lower-data settings, but the weak or unstable effects on ASSISTments 2009 and 2012 do not establish a general sample-efficiency advantage.

The reduced-data evaluation summarized in Table 6 tests whether semantic initialization improves DKT’s predictive performance when less behavioral data are available. Model selection follows the protocol described earlier: for each sampled dataset, the representative run is selected by validation AUC, and the selected run’s test AUC and test accuracy are reported. For ASSISTments 2009, this includes a pipeline-selection step across all nine pipelines at each sample size; for ASSISTments 2012 and 2017, the final sampled-data setup uses Pipeline 3 only.

Table 6: Dataset-level summary of the data-efficiency experiments. Mean deltas are computed as pretrained minus baseline across the nine sampled percentages from 10% to 90% of each complete trimmed text-linked base dataset’s interaction-row count; one row represents one student KC-response interaction. The Positive AUC splits column counts how often the selected pretrained run exceeds the selected baseline run on test AUC. The Largest $\Delta \mathrm { A U C }$ and Smallest ∆AUC columns identify the sampled percentages with the largest and smallest pretrainedminus-baseline test-AUC differences, respectively. The Pipeline column identifies the pipeline used at the split with the largest difference.
<table><tr><td>Dataset</td><td>Mean ∆AUC Mean</td><td> $\Delta \mathsf { A c c } .$ </td><td>Positive AUC splits</td><td>Largest ΔAUC</td><td>Pipeline</td><td>Smallest ΔAUC</td></tr><tr><td>ASSISTments 2009</td><td>+0.0004</td><td>+0.0000</td><td>5/9</td><td>40%</td><td>P4</td><td>20%</td></tr><tr><td>ASSISTments 2012</td><td>+0.0002</td><td>-0.0028</td><td>6/9</td><td>80%</td><td>P3</td><td>40%</td></tr><tr><td>ASSISTments 2017</td><td>+0.0096</td><td>+0.0020</td><td>8/9</td><td>30%</td><td>P3</td><td>20%</td></tr></table>

Figure 10 reports the selected test AUC curves for the baseline and pretrained settings across sample sizes. The overall pattern is mixed. ASSISTments 2009 is unstable: the pretrained condition exceeds the baseline at five of the nine nested sample sizes, but the mean AUC gain is effectively zero and the selected pipeline changes with the sampling level. ASSISTments 2012 is similarly weak, with near-zero mean AUC change and a small decline in mean test accuracy. The most favorable dataset is ASSISTments 2017, for which the pretrained condition improves test AUC at eight of nine nested sample sizes and yields a mean gain of approximately 0.01 AUC.

Data-efficiency results: selected test AUC by sample size

![](images/44511dfe71849a400afaa8f8fac9b55a7d7f6328edca28432efc40b99a4e8c5f.jpg)

![](images/acb6d88ddd5ce6923a1ecd2d86d8748af5e8653900a074ea7bb6adacbd09ed2f.jpg)

![](images/5535079efc13a0766b27af1dded935f03bd8aa9e74fc65b64809f938d0a4420b.jpg)  
Figure 10: Selected test AUC by sample size for the baseline and pretrained data-efficiency experiments. Each sample size is a percentage of the complete trimmed text-linked base dataset’s interaction-row count, where one row represents one student KC-response interaction. Each curve uses the validation-selected run at each sampled percentage.

ASSISTments 2009 is the only reduced-data setting in which all nine pipelines were evaluated at each sample size. The detailed pipeline-by-pipeline view is provided in Appendix Figure 11. No pipeline is best across all splits: the selected pipeline switches among P3, P4, P6, and P8, so the reduced-data ranking varies by split. By contrast, the reduced-data experiments for ASSISTments 2012 and 2017 use Pipeline 3 throughout, so those trends isolate the effect of semantic initialization more narrowly but do not test pipeline diversity in the same way.

## 4.4. MECHANISM ANALYSIS: KC-SUPERVISED TEXT ENCODERS

The fine-tuned BERT experiments test a KC-label-supervised initialization strategy alongside the general-purpose embedding models used in the rest of the study. Instead of relying on general-purpose text similarity, these embeddings are derived from bert-base-uncased models fine-tuned to predict KC labels directly from question text. The representation should therefore be KC-discriminative for the dataset’s labels. High classification performance shows that the encoder separates those labels, not that it captures semantic structure in general. The mechanism analysis asks whether that label alignment translates into better KT performance once the embeddings are introduced into DKT.

The fine-tuning results confirm that the encoder learns dataset-specific KC distinctions with highly variable success across datasets. Table 7 reports test accuracies of 0.9967 and 0.9795 for ASSISTments 2009 and 2012, respectively, compared with 0.1354 for ASSISTments 2017. The filtered ASSISTments 2017 split contains 448 training examples and 96 test examples across 51 KC classes.

The ASSISTments 2017 classifier is included in Table 7 as a diagnostic of this data constraint rather than as a downstream initialization condition. Its small filtered training set and weak held-out performance made it unsuitable for the fine-tuned BERT DKT comparison, so the downstream results in Table 8 are limited to ASSISTments 2009 and 2012.

Table 7: KC-classification performance of the dataset-specific BERT classifiers after the stated filtering and split procedure. The #KCs column reports the classifier’s retained label count. The ASSISTments 2017 classifier is included as a diagnostic and was not used for downstream DKT initialization.
<table><tr><td>Dataset</td><td>#KCs</td><td>Train</td><td>Test</td><td>Test Acc.</td><td>Macro F1</td><td>Weighted F1</td></tr><tr><td>ASSISTments 2009</td><td>65</td><td>4297</td><td>921</td><td>0.9967</td><td>0.9665</td><td>0.9963</td></tr><tr><td>ASSISTments 2012</td><td>130</td><td>13894</td><td>2978</td><td>0.9795</td><td>0.8434</td><td>0.9772</td></tr><tr><td>ASSISTments 2017</td><td>51</td><td>448</td><td>96</td><td>0.1354</td><td>0.1095</td><td>0.0860</td></tr></table>

Table 8: Matched downstream DKT comparison for the KC-supervised BERT encoders. The matched baseline is the no-embedding-initialization DKT baseline used for the same downstream comparison. For each dataset, the baseline and fine-tuned rows are selected by validation AUC within their respective comparison conditions. Boldface marks the highest displayed test value within each dataset and metric, including ties.
<table><tr><td>Dataset</td><td>Condition</td><td>Selected pipeline Test AUC</td><td></td><td>ΔAUC</td><td>Test Acc.</td><td>∆Acc.</td></tr><tr><td>ASSISTments 2009</td><td>Matched baseline</td><td></td><td>0.8552</td><td></td><td>0.7875</td><td></td></tr><tr><td>ASSISTments 2009</td><td>Fine-tuned BERT P4</td><td></td><td>0.8581</td><td>+0.0029</td><td>0.7875</td><td>+0.0000</td></tr><tr><td>ASSISTments 2012</td><td>Matched baseline</td><td></td><td>0.7363</td><td></td><td>0.7384</td><td></td></tr><tr><td>ASSISTments 2012</td><td>Fine-tuned BERT P4</td><td></td><td>0.7364</td><td>+0.0001</td><td>0.7371</td><td>-0.0013</td></tr></table>

Table 8 shows that the KC-supervised BERT encoders produce small test-AUC gains over their matched baselines, but these gains do not alter the main conclusion of this study. On ASSISTments 2009, where Table 7 shows near-perfect KC classification, the selected fine-tuned BERT configuration improves test AUC by 0.0029 over its matched baseline while leaving test accuracy unchanged. On ASSISTments 2012, the selected fine-tuned BERT configuration is essentially tied with its matched baseline on AUC and slightly lower on accuracy. In both datasets, the selected downstream configuration uses Pipeline 4. One possible mechanism is that constructing KC embeddings from a single representative question and then injecting response state through the stack-antonym scheme makes the KC-supervised representation more useful within these sweeps.

Both AUC differences are small descriptive values from validation-selected configurations, and the accuracy results do not show a corresponding improvement. Greater KC discriminability at the text-encoder level may therefore improve DKT’s predictive performance in some configurations, but not consistently across the evaluated datasets and settings. The 2017 diagnostic shows what happens when the text-linked dataset is very small: the model struggles even to learn the KC classification problem itself, limiting the representations available for downstream KT.

## 5. DISCUSSION

## 5.1. INTERPRETING THE WEAK GRAPH ALIGNMENT

The graph-alignment analysis directly answers RQ1, the paper’s central question. If semantic KC representations encoded the same relational structure extracted from DKT’s behavior by the influence-score procedure, then the semantic KC graphs would be expected to align strongly with the DKT-derived KC graphs. Instead, even the highest-Pearson configurations show only weak and inconsistent alignment, which is insufficient to support the claim that semantic KC geometry strongly recovers the relation structure captured by the original DKT influence-score procedure. Under this text-based test, DKT-derived relations are not primarily semantic.

Alignment is highest on ASSISTments 2012, lower on ASSISTments 2009, and lowest on ASSISTments 2017, but it remains weak in all three datasets. Embedding-model and pipeline choices improve alignment modestly, but no setting produces graph agreement strong enough to explain reliable downstream gains.

The co-exposure reference helps interpret the weak semantic alignment. On two of three datasets, a simple structure derived from the training sequences aligns more strongly with the DKT-derived matrix than the best text-semantic graph does. This suggests that the extracted graph contains a behavioral co-exposure component that text-derived geometry recovers less well. It does not establish the source of the DKT relation structure, and it is not a stability estimate.

This dataset ordering is also plausible given Table 1. ASSISTments 2012 has the largest absolute text-linked question pool, ASSISTments 2017 the smallest, and ASSISTments 2009 falls between them. That pattern does not prove that data volume alone determines graph align ment, but it does suggest that the absolute number of available question texts constrains how well semantic KC relations can be estimated.

ASSISTments 2017 has much less text per KC: 638 matched questions across 68 KCs, compared with thousands of matched questions across the modeled KC sets in ASSISTments 2009 and 2012. This makes the 2017 semantic KC representations higher-variance and plausibly contributes to both the weak alignment and the weak KC classifier. Text sparsity alone does not explain the full dataset pattern: the co-exposure reference uses no question text and is also weakest on ASSISTments 2017.

Dataset size alone is still not a sufficient explanation. Even the higher-alignment configurations on ASSISTments 2012 remain only partially aligned with the DKT graph. These results are consistent with a structural mismatch between text-derived semantic geometry and the relation structure captured by the DKT influence-score procedure, although the present experiments do not isolate that mismatch from all measurement and modeling choices.

The KC-supervised BERT configurations in the RQ1 alignment sweep likewise show only weak alignment. Different pipelines induce visibly different cosine-similarity patterns, so the construction choice is not trivial. However, the pipeline grid does not reveal any construction whose heatmaps consistently reorganize the KC space into something that resembles the DKTderived graphs across datasets. These construction differences change the semantic geometry but do not produce strong recovery of the DKT-derived graphs.

The full-data experiments provide a complementary answer to RQ2.1. No reported semantic configuration selected by validation AUC exceeds the baseline DKT test AUC, although some configurations produce localized accuracy gains; pretrained semantic embeddings therefore do not reliably improve DKT’s predictive performance in the full-data setting. This result is consistent with the weak graph alignment: if DKT primarily depended on relations captured by the evaluated semantic matrices, replacing random interaction embeddings with text-derived representations should have produced a more reliable benefit. However, predictive performance also depends on representation construction, optimization, and model architecture, so the initialization experiments do not independently establish that DKT relations are not primarily semantic.

## 5.2. WHAT THE LIMITED FAVORABLE SIGNALS MEAN

The reduced-data experiments provide a mixed answer to RQ2.2. If pretrained semantic embeddings provided a strong inductive bias for KT, their advantage would be expected to emerge more clearly when less behavioral data are available. The favorable ASSISTments 2017 trend includes positive AUC changes at eight of nine nested sample sizes and an approximately 0.01 mean AUC improvement, whereas the effects on ASSISTments 2009 and 2012 are weak or unstable. This pattern does not establish a general sample-efficiency advantage. The study does not isolate the cause of this dataset contrast; the smaller absolute text-linked question pool on AS SISTments 2017 and the larger but potentially more heterogeneous KC space on ASSISTments 2012 may contribute, but neither explanation is established by these experiments.

The KC-supervised BERT analysis tests whether dataset-specific supervision changes the full-data result. Dataset-specific supervision makes semantic initialization more competitive, especially on ASSISTments 2009, yet the gains remain small and inconsistent and fail to produce a decisive full-data advantage. The mechanism analysis therefore qualifies but does not overturn the answer to RQ2.1.

These results limit direct transfer from text-only NLP representations to educational modeling. In KT, the relevant structure may depend on response behavior, temporal order, latent prerequisite dependencies, psychometric difficulty, and dataset-specific instructional organization. These factors may correlate with semantics, but they need not be reducible to semantics. The present results therefore suggest that KC relations in KT may occupy a latent learning space: a representation of how different KCs are organized in the learner’s knowledge state, only weakly aligned with the semantic geometry induced by text embeddings. A prerequisite relation can be directional, whereas cosine similarity is symmetric; semantic proximity therefore cannot represent every property of a directed learning relation.

## 5.3. BROADER IMPLICATIONS FOR KNOWLEDGE TRACING

Under the evaluated configurations, simply substituting semantic embeddings into the KT input representation does not reliably improve DKT’s predictive performance. Future approaches may need to combine text with behavioral, temporal, curricular, structural, or psychometric signals rather than only modify the input representation.

## 5.4. LIMITATIONS

This study uses three ASSISTments datasets, and question-text coverage is incomplete and uneven, as shown in Table 1. The study is restricted to the text-linked subset of each dataset, so the conclusions apply to questions and KCs with matched Problem Bodies coverage rather than to every interaction or KC in the original full logs. This scope is especially restrictive for ASSISTments 2017.

The reported comparisons are also conditional on the evaluated embedding models and nine representation pipelines. These choices include KC-supervised BERT encoders and several recurrent-model variants, but they do not exhaust all possible ways of representing semantic structure or integrating language into KT. RQ2 therefore tests these particular ways of introducing semantic representations into DKT. Those conditions also change input dimensionality and recurrent capacity, so they are not a pure test of semantic content alone. A dimension-matched random control would help separate these factors in future work.

Several reported AUC and accuracy differences are small descriptive values from validationselected configurations. The study does not quantify run-to-run variation for those selected fulldata runs or variability across alternative reduced-data samples, so the favorable values should not be read as statistically established effects.

The reported analyses follow the processed single-KC interaction representation used in the DKT comparisons, although ASSISTments interactions can involve one or more KCs. Richer treatments of multi-KC structure may change how semantic and behavioral relations are represented and remain outside the scope of the present experiments.

The reduced-data test-AUC pattern is favorable on ASSISTments 2017 but weak or unstable on ASSISTments 2009 and 2012. This dataset dependence does not support a general sampleefficiency claim.

Finally, the graph-alignment analysis depends on the complete weighted DKT relation matri ces over the trimmed text-linked KC sets produced by the original DKT influence-score extraction procedure and on the evaluated model and training setup. The comparison symmetrizes the directional DKT relations before comparing them with cosine similarity. The alignment findings therefore apply to this graph-extraction and comparison procedure, not to every possible notion of KT structure or conceptual relatedness.

We do not estimate how stable the extracted DKT relation matrices are across retraining runs, so the observed alignment magnitudes are not calibrated against the reproducibility of the DKT-derived graph itself. The co-exposure reference depends on the composition and length of the processed training sequences and collapses response order and correctness. It is a descriptive behavioral reference, not a stability ceiling for the DKT-derived graph.

## 6. CONCLUSION

We asked: Are the KC relations learned by DKT semantic? DKT models relations among KCs because an interaction with one KC can affect predictions for other KCs. In the operational sense tested here, the relations are not semantic: the tested text-derived semantic structures do not strongly recover the DKT-derived relation structure.

DKT configurations initialized with text-derived semantic KC representations do not reliably outperform the standard DKT baseline. The reduced-data test-AUC pattern is favorable on ASSISTments 2017, and KC-supervised BERT produces small AUC gains on ASSISTments 2009 and 2012, but these findings do not establish a consistent predictive advantage across the evaluated datasets and settings. We interpret these results as consistent with a learning-relevant latent space that is only weakly aligned with semantic proximity among questions and KCs.

## DECLARATION OF GENERATIVE AI SOFTWARE TOOLS IN THE WRITINGPROCESS

AI-based writing tools were used to assist with copyediting and language refinement; all research, analysis, interpretation, and conclusions are the author’s own.

## REFERENCES

ABDELRAHMAN, G., WANG, Q., AND NUNES, B. 2023. Knowledge Tracing: A Survey. ACM Comput. Surv. 55, 11 (Feb.), 224:1–224:37.

BOJANOWSKI, P., GRAVE, E., JOULIN, A., AND MIKOLOV, T. 2017. Enriching Word Vectors with Subword Information. Transactions ofthe Associationfor Computational Linguistics 5, 135–146.

CHEN, P., LU, Y., ZHENG, V. W., AND PIAN, Y. 2018. Prerequisite-Driven Deep Knowledge Tracing. In 2018 IEEE International Conference on Data Mining (ICDM). IEEE, Singapore, 39–48.

CORBETT, A. T. AND ANDERSON, J. R. 1995. Knowledge tracing: Modeling the acquisition of procedural knowledge. User Modelling and User-Adapted Interaction 4, 4, 253–278.

DEVLIN, J., CHANG, M.-W., LEE, K., AND TOUTANOVA, K. 2019. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In Proceedings of the 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers). 4171–4186.

DUAN, Z., DONG, X., GU, H., WU, X., LI, Z., AND ZHOU, D. 2024. Towards more accurate and interpretable model: Fusing multiple knowledge relations into deep knowledge tracing. Expert Systems with Applications 243, 122573.

FENG, M., HEFFERNAN, N., AND KOEDINGER, K. 2009. Addressing the assessment challenge with an online system that tutors as it assesses. User Modeling and User-Adapted Interaction 19, 3 (Aug.), 243–266.

GAO, T., YAO, X., AND CHEN, D. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 6894–6910.

GHOSH, A., HEFFERNAN, N., AND LAN, A. S. 2020. Context-Aware Attentive Knowledge Tracing. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2330–2339.

HONG, K., KARBASI, K., AND POTTIE, G. 2025. Extracting Causal Relations in Deep Knowledge Tracing. In Proceedings ofthe 18th International Conference on Educational Data Mining. International Educational Data Mining Society, 586–591.

KHAJAH, M., LINDSEY, R. V., AND MOZER, M. C. 2016. How Deep is Knowledge Tracing? In Proceedings ofthe 9th International Conference on Educational Data Mining. 94–101.

LIU, Q., HUANG, Z., YIN, Y., CHEN, E., XIONG, H., SU, Y., AND HU, G. 2019. EKT: Exercise-aware Knowledge Tracing for Student Performance Prediction. arXiv:1906.05658 [cs].

LIU, Z., LIU, Q., CHEN, J., HUANG, S., TANG, J., AND LUO, W. 2023. pyKT: A Python Library to Benchmark Deep Learning based Knowledge Tracing Models. arXiv:2206.11460 [cs].

LUO, Y., XIAO, B., JIANG, H., AND MA, J. 2022. Heterogeneous Graph Based Knowledge Tracing. In 2022 11th International Conference on Educational and Information Technology (ICEIT). IEEE, Chengdu, China, 226–231.

MIKOLOV, T., SUTSKEVER, I., CHEN, K., CORRADO, G. S., AND DEAN, J. 2013. Distributed Representations of Words and Phrases and their Compositionality. In Advances in Neural Information Processing Systems 26. 3111–3119.

NAKAGAWA, H., IWASAWA, Y., AND MATSUO, Y. 2019. Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network. In IEEE/WIC/ACM International Conference on Web Intelligence. ACM, Thessaloniki Greece, 156–163.

PANDEY, S. AND KARYPIS, G. 2019. A Self-Attentive model for Knowledge Tracing. arXiv:1907.06837 [cs, stat].

PARDOS, Z. A. AND DADU, A. 2017. Imputing KCs with Representations of Problem Content and Context. In Proceedings of the 25th Conference on User Modeling, Adaptation and Personalization. ACM, Bratislava Slovakia, 148–155.

PENNINGTON, J., SOCHER, R., AND MANNING, C. D. 2014. GloVe: Global Vectors for Word Representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP). 1532–1543.

PIECH, C., SPENCER, J., HUANG, J., GANGULI, S., SAHAMI, M., GUIBAS, L., AND SOHL-DICKSTEIN, J. 2015. Deep Knowledge Tracing. arXiv:1506.05908 [cs].

PRIHAR, E., SYED, M., OSTROW, K., SHAW, S., SALES, A., AND HEFFERNAN, N. 2022. Exploring Common Trends in Online Educational Experiments. Publisher: Zenodo.

REIMERS, N. AND GUREVYCH, I. 2019. Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing. 3982–3992.

SONKAR, S., WATERS, A. E., LAN, A. S., GRIMALDI, P. J., AND BARANIUK, R. G. 2020. qDKT: Question-centric Deep Knowledge Tracing. arXiv:2005.12442 [cs, stat].

SU, Y., LIU, Q., LIU, Q., HUANG, Z., YIN, Y., CHEN, E., DING, C., WEI, S., AND HU, G. 2018. Exercise-Enhanced Sequential Modeling for Student Performance Prediction. Proceedings of the AAAI Conference on Artificial Intelligence 32, 1 (Apr.).

TONG, H., WANG, Z., ZHOU, Y., TONG, S., HAN, W., AND LIU, Q. 2022. HGKT: Introducing Hierarchical Exercise Graph for Knowledge Tracing. arXiv:2006.16915 [cs].

TONG, S., LIU, Q., HUANG, W., HUANG, Z., CHEN, E., LIU, C., MA, H., AND WANG, S. 2020. Structure-Based Knowledge Tracing: An Influence Propagation View. In 2020 IEEE International Conference on Data Mining (ICDM). 541–550. ISSN: 2374-8486.

WANG, W., LIU, Q., HUANG, Z., CHEN, E., YIN, Y., AND SU, Y. 2022. Semantic-enhanced question embeddings pre-training for improving knowledge tracing. Information Sciences 608, 1022–1039.

XIONG, X., ZHAO, S., VAN INWEGEN, E. G., AND BECK, J. E. 2016. Going Deeper with Deep Knowledge Tracing. In Proceedings of the 9th International Conference on Educational Data Mining. 545– 550.

YEUNG, C.-K. AND YEUNG, D.-Y. 2018. Addressing Two Problems in Deep Knowledge Tracing via Prediction-Consistent Regularization. arXiv:1806.02180 [cs].

YIN, Y., LIU, Q., HUANG, Z., CHEN, E., TONG, W., WANG, S., AND SU, Y. 2019. QuesNet: A Unified Representation for Heterogeneous Test Questions. In Proceedings ofthe 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 1328–1336. arXiv:1905.10949 [cs, stat].

ZHANG, J., SHI, X., KING, I., AND YEUNG, D.-Y. 2017. Dynamic Key-Value Memory Networks for Knowledge Tracing. arXiv:1611.08108 [cs].

## A. SUPPLEMENTARY MATERIAL

## A.1. COMPARISON SETS AND AGGREGATION

Table 9: Comparison sets and aggregation units for the graph-alignment results. Because coverage varies across the evaluated configurations, the aggregate means are descriptive summaries rather than controlled effects of changing only one configuration dimension.
<table><tr><td>Term or display</td><td>Scope and aggregation unit</td></tr><tr><td>Configuration</td><td>One dataset-specific combination of embedding model, representation pipeline, and architecture variant.</td></tr><tr><td>RQ1 alignment sweep</td><td>The full set of evaluated dataset-embedding-model-representation-pipeline-architecture-variant combinations; coverage varies because not every model-pipeline-architecture combination is present in every dataset.</td></tr><tr><td>RQ2.1 full-data sweep</td><td>Approximately 1,065 distinct evaluated full-data configurations represented across the 21 experiment exports summarized in Table 13.</td></tr><tr><td>Displayed embedding-model summary</td><td>The five model means reported in Table 14; this display does not define which configurations contribute to the pipeline aggregate.</td></tr><tr><td>Embedding-model mean</td><td>Reported mean across each displayed model&#x27;s evaluated dataset-pipeline configurations.</td></tr><tr><td>Pipeline mean</td><td>Reported across-dataset aggregate for that pipeline.</td></tr><tr><td>Table 4</td><td>One configuration per dataset, selected by Pearson correlation from the RQ1 alignment sweep. Spearman and top-10% overlap are conditional diagnostics for those rows.</td></tr><tr><td>Tables 14–15</td><td>Reported aggregate means, not configuration-level maxima or a complete RQ1-sweep distribution.</td></tr></table>

## A.2. DETAILED PIPELINE COMPARISONS

Tables 10–12 report the per-dataset pipeline results. Across the three datasets, the highest-AUC semantic pipeline varies, and no semantic pipeline consistently outperforms the standard DKT baseline in test AUC.

Table 10: Detailed pipeline comparison for ASSISTments 2009, with each pipeline represented by its validation-selected configuration. Boldface marks the highest displayed test value in each metric column. Pipeline 1 achieved the highest semantic test AUC, while Pipeline 4 achieved the highest semantic test accuracy.
<table><tr><td>Pipeline</td><td>Selected architecture</td><td>Test AUC</td><td>Test Acc.</td></tr><tr><td>P1</td><td>Pretrained Embedding Initialization + Regularization</td><td>0.8567</td><td>0.7865</td></tr><tr><td>P2</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.8321</td><td>0.7702</td></tr><tr><td>P3</td><td>Projection Bottleneck</td><td>0.8542</td><td>0.7845</td></tr><tr><td>P4</td><td>Projection Bottleneck</td><td>0.8552</td><td>0.7870</td></tr><tr><td>P5</td><td>Projection Bottleneck</td><td>0.8535</td><td>0.7847</td></tr><tr><td>P6</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.8349</td><td>0.7727</td></tr><tr><td>P7</td><td>Projection Bottleneck</td><td>0.8536</td><td>0.7843</td></tr><tr><td>P8</td><td>Projection Bottleneck</td><td>0.8535</td><td>0.7852</td></tr><tr><td>P9</td><td>Projection Bottleneck</td><td>0.8542</td><td>0.7869</td></tr></table>

Table 11: Detailed pipeline comparison for ASSISTments 2012, with each pipeline represented by its validation-selected configuration. Boldface marks the highest displayed test value in each metric column, including ties. Pipelines 2, 3, and 6 tie for the highest semantic test AUC, while Pipeline 7 achieved the highest semantic test accuracy.
<table><tr><td>Pipeline</td><td>Selected architecture</td><td>Test AUC</td><td>Test Acc.</td></tr><tr><td>P1</td><td>Semantic Output Layer</td><td>0.7356</td><td>0.7376</td></tr><tr><td>P2</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7358</td><td>0.7377</td></tr><tr><td>P3</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7358</td><td>0.7374</td></tr><tr><td>P4</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7354</td><td>0.7376</td></tr><tr><td>P5</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7353</td><td>0.7374</td></tr><tr><td>P6</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7358</td><td>0.7371</td></tr><tr><td>P7</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7355</td><td>0.7382</td></tr><tr><td>P8</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7356</td><td>0.7372</td></tr><tr><td>P9</td><td>Semantic Output Layer + Projection Bottleneck</td><td>0.7350</td><td>0.7367</td></tr></table>

Table 12: Detailed pipeline comparison for ASSISTments 2017, with each pipeline represented by its validation-selected configuration. Boldface marks the highest displayed test value in each metric column. Pipeline 3 achieved the highest semantic test AUC and test accuracy.
<table><tr><td>Pipeline</td><td>Selected architecture</td><td>Test AUC</td><td>Test Acc.</td></tr><tr><td>P1</td><td>Pretrained Embedding Initialization</td><td>0.7205</td><td>0.6806</td></tr><tr><td>P2</td><td>Semantic Output Layer</td><td>0.7171</td><td>0.6757</td></tr><tr><td>P3</td><td>Pretrained Embedding Initialization</td><td>0.7207</td><td>0.6811</td></tr><tr><td>P4</td><td>Projection Bottleneck</td><td>0.7196</td><td>0.6808</td></tr><tr><td>P5</td><td>Projection Bottleneck</td><td>0.7137</td><td>0.6774</td></tr><tr><td>P6</td><td>Adaptive recurrent capacity</td><td>0.7197</td><td>0.6803</td></tr><tr><td>P7</td><td>Adaptive recurrent capacity</td><td>0.7141</td><td>0.6788</td></tr><tr><td>P8</td><td>Projection Bottleneck with Regularization</td><td>0.7009</td><td>0.6719</td></tr><tr><td>P9</td><td>Projection Bottleneck with Regularization</td><td>0.6997</td><td>0.6711</td></tr></table>

## A.3. FULL-DATA SWEEP COVERAGE

Table 13 summarizes the 21 experiment exports that represent approximately 1,065 distinct evaluated configurations in the RQ2.1 full-data sweep. Together, the exports contain 2,324 records, including 1,605 finished runs with usable validation and test AUC values. Repeated logged configurations may appear, so these record totals are not counts of distinct configurations.

Table 13: Experiment-export coverage for the RQ2.1 full-data sweep. The Exported column gives the number of records in each export; the Usable column gives the number of finished runs with usable validation and test AUC values. Selection is by validation AUC. Model labels: 3-small = text-embedding-3-small; 3-large = text-embedding-3-large; bert-ft-2009 = bert\_ft\_assist2009. Repeated logged configurations may appear in an export.
<table><tr><td>ID</td><td>Dataset</td><td>Condition / variant</td><td>Embedding model</td><td>Exported</td><td>Usable</td><td>Selection</td></tr><tr><td>S01</td><td>2009</td><td>Baseline DKT</td><td></td><td>159</td><td></td><td>139 Val. AUC</td></tr><tr><td>S02</td><td>2009</td><td>Pretrained initialization</td><td>3-small</td><td>40</td><td></td><td>40 Val. AUC</td></tr><tr><td>S03</td><td>2009</td><td>Initialization + regularization</td><td>3-small</td><td>10</td><td></td><td>10 Val. AUC</td></tr><tr><td>S04</td><td>2009</td><td>Projection bottleneck</td><td>3-large</td><td>27</td><td></td><td>27 Val. AUC</td></tr><tr><td>S05</td><td>2009</td><td>Adaptive recurrent capacity</td><td>embed-v4.0</td><td>27</td><td></td><td>27 Val. AUC</td></tr><tr><td>S06</td><td>2009</td><td>Semantic output</td><td>3-small</td><td>45</td><td></td><td>44 Val. AUC</td></tr><tr><td>S07</td><td>2009</td><td>Semantic output + projection</td><td>bert-ft-2009</td><td>27</td><td></td><td>27 Val. AUC</td></tr><tr><td>S08</td><td>2012</td><td>Baseline DKT</td><td></td><td>20</td><td></td><td>20 Val. AUC</td></tr><tr><td>S09</td><td>2012</td><td>Pretrained initialization</td><td>3-large</td><td>25</td><td></td><td>24 Val. AUC</td></tr><tr><td>S10</td><td>2012</td><td>Projection bottleneck</td><td>3-large</td><td>36</td><td></td><td>36 Val. AUC</td></tr><tr><td>S11</td><td>2012</td><td>Projection bottleneck</td><td>embed-v4.0</td><td>36</td><td></td><td>36 Val. AUC</td></tr><tr><td>S12</td><td>2012</td><td>Adaptive recurrent capacity</td><td>3-large</td><td>80</td><td></td><td>79 Val. AUC</td></tr><tr><td>S13</td><td>2012</td><td>Semantic output</td><td>3-large</td><td>34</td><td></td><td>33 Val. AUC</td></tr><tr><td>S14</td><td>2012</td><td>Semantic output + projection</td><td>3-large</td><td>124</td><td></td><td>121 Val. AUC</td></tr><tr><td>S15</td><td>2017</td><td>Baseline DKT</td><td></td><td>100</td><td></td><td>100 Val. AUC</td></tr><tr><td>S16</td><td>2017</td><td>Pretrained initialization</td><td>3-small</td><td>796</td><td></td><td>113 Val. AUC</td></tr><tr><td>S17</td><td>2017</td><td>Projection bottleneck</td><td>3-small</td><td>166</td><td></td><td>166 Val. AUC</td></tr><tr><td>S18</td><td>2017</td><td>Projection + regularization</td><td>3-small</td><td>60</td><td></td><td>60 Val. AUC</td></tr><tr><td>S19</td><td>2017</td><td>Adaptive recurrent capacity</td><td>3-small</td><td>143</td><td></td><td>137 Val. AUC</td></tr><tr><td>S20</td><td>2017</td><td>Semantic output, export 1</td><td>3-large</td><td>173</td><td></td><td>171 Val. AUC</td></tr><tr><td>S21</td><td>2017</td><td>Semantic output, export 2</td><td>3-large</td><td>196</td><td></td><td>195 Val. AUC</td></tr></table>

## A.4. ADDITIONAL DATA-EFFICIENCY DIAGNOSTIC

Figure 11 gives the full ASSISTments 2009 pipeline-by-pipeline reduced-data comparison. AS-SISTments 2009 is the only dataset in which all nine pipelines were compared at every sample size. The selected pipeline changes across splits, so the result does not favor one semantic construction.

ASSISTments 2009: all pipeline test AUC curves vs baseline  
![](images/317cb7a1a8dc9a24cc18eb6e925a327e85ee0044b220fe2fc0aae013c62a154e.jpg)  
Figure 11: ASSISTments 2009 data-efficiency comparison across all nine pipelines and the baseline. Each sample size is a percentage of the complete trimmed text-linked base dataset’s interaction-row count, where one row represents one student KC-response interaction. Each pipeline curve shows the validation-selected configuration for that pipeline at each sampled percentage.

## A.5. ADDITIONAL KC-GRAPH ALIGNMENT DIAGNOSTICS

Tables 14–15 report embedding-model and pipeline means for the KC-graph alignment analysis. The rankings vary, but no displayed aggregate strongly recovers the DKT-derived graph. Table 9 defines the comparison sets and aggregation units.

Table 14: Displayed embedding-model means. Models are ordered by mean Pearson correlation across each model’s evaluated dataset–pipeline configurations. Boldface marks the highest displayed value in each metric column, including ties.
<table><tr><td>Embedding model</td><td>Pearson</td><td>Spearman</td><td>Top-10% overlap</td></tr><tr><td>bert_ft_assist2012</td><td>0.0321</td><td>0.0260</td><td>0.1304</td></tr><tr><td>text-embedding-3-large</td><td>0.0297</td><td>0.0260</td><td>0.1268</td></tr><tr><td>bert_ft_assist2009</td><td>0.0217</td><td>0.0118</td><td>0.0999</td></tr><tr><td>all-MiniLM-L12-v2</td><td>0.0205</td><td>0.0178</td><td>0.1260</td></tr><tr><td>embed-v4.0</td><td>0.0187</td><td>0.0185</td><td>0.1229</td></tr></table>

Table 15: Reported pipeline summary across datasets. Pipelines are ordered by mean Pearson correlation. Boldface marks the highest displayed value in each metric column.
<table><tr><td>Pipeline</td><td>Pearson</td><td>Spearman</td><td>Top-10% overlap</td></tr><tr><td>5</td><td>0.0243</td><td>0.0233</td><td>0.1229</td></tr><tr><td>9</td><td>0.0238</td><td>0.0223</td><td>0.1214</td></tr><tr><td>3</td><td>0.0227</td><td>0.0191</td><td>0.1228</td></tr><tr><td>7</td><td>0.0189</td><td>0.0165</td><td>0.1208</td></tr><tr><td>1</td><td>0.0175</td><td>0.0142</td><td>0.1216</td></tr><tr><td>6</td><td>0.0167</td><td>0.0237</td><td>0.1231</td></tr><tr><td>4</td><td>0.0159</td><td>0.0243</td><td>0.1264</td></tr><tr><td>8</td><td>0.0138</td><td>0.0163</td><td>0.1144</td></tr><tr><td>2</td><td>0.0114</td><td>0.0104</td><td>0.1229</td></tr></table>

Pipeline choice changes the embedding geometry, but none of the shown variants closely resembles the DKT-derived graphs. Figure 12 provides the pipeline-by-dataset cosine-similarity comparison for text-embedding-3-large.

![](images/20aacc72320fafd664d1647564ce70a0fbd35feee11531f84dc61ea324495f1a.jpg)  
Figure 12: Pipeline-induced embedding-relation heatmaps for text-embedding-3-large.