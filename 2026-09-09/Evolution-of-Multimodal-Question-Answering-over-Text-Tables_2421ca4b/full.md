# Evolution of Multimodal Question Answering over Text, Tables, and Images: From Modality-Adaptive Extraction to Unified Language Representation

Abdullah Al Shafi

Department of CSE, Khulna University of Engineering & Technology, Bangladesh Email: abdullah@iict.kuet.ac.bd

Abstract—The rapid growth of multimodal data has intensified the need for question answering (QA) systems capable of reasoning across heterogeneous sources such as text, tables, and images. In this paper, we present a comprehensive methodological comparison of three influential frameworks, namely Multimodal Adaptive Extraction (MAE), Solar, and UniMMQA, tracing the evolution of multimodal question answering from modalityadaptive pipelines to fully unified architectures. We examine how each approach models cross-modal interactions, transforms heterogeneous inputs, and performs reasoning, highlighting key design differences in modality representation, reasoning, and answer generation. Our analysis demonstrates a clear shift from explicit modality-specific processing toward unified textcentric formulations enabled by pre-trained language models (PLMs). Empirical comparisons across benchmark datasets show that this transition leads to substantial improvements in both Exact Match (EM) and F1-Scores, with UniMMQA achieving the most consistent and scalable performance. Despite these advances, we identify persistent challenges, including information loss during modality transformation, error propagation in multi-stage pipelines, and limitations in capturing fine-grained cross-modal dependencies. Overall, this study provides a deeper understanding of current design trends and offers insights into the future direction of unified multimodal reasoning systems.

Index Terms—multimodal QA, conversational QA, modalityadaptive extraction, unified language representation, cross-modal reasoning

## I. INTRODUCTION

The rapid growth of digital information has led to an overwhelming information explosion, making it increasingly difficult for users to obtain concise and accurate answers [1]. Question answering (QA) systems have therefore become an important solution to information overload, with early efforts focusing on knowledge-based, document-based, and community-based settings [2], [3]. However, these systems are typically limited to unimodal or single-source inputs, which restricts their applicability in real-world scenarios where relevant information is distributed across heterogeneous modalities.

To address this limitation, recent research has moved toward multimodal question answering (MMQA) and its conversational extension, multimodal conversational question answering (MMCoQA), where systems are required to reason over multiple modalities such as text, tables, and images within multi-turn dialogues [4], [5]. In this setting, a key challenge is that different questions may require different modalities, and effective reasoning often depends on jointly leveraging complementary information across sources.

In this work, we analyze three representative recent approaches that tackle multimodal and conversational QA over text, tables, and images, namely modality-adaptive extraction (MAE) [6], unified language representation with retrievalaugmented reasoning (Solar) [7], unified language Representation with rationale-enhanced reasoning (UniMMQA) [8]. Although these works share the same goal of enabling QA over heterogeneous modalities, they differ in how multimodal information is represented, retrieved, and reasoned. [5] introduces the first ever conversational multimodal QA setting namely MMCoQA where questions evolve across dialogue turns and require reasoning over text, tables, and images. It highlights the difficulty of maintaining contextual understanding while selecting relevant evidence across modalities in a conversational setting. Then, Yu et al. [7] proposes a unified language representation framework called Solar that converts multimodal inputs into a text-centric representation, enabling a single QA model to handle heterogeneous sources. This approach simplifies multimodal reasoning by leveraging large pre-trained language models (PLMs), while relying on retrieval and ranking over unified textual evidence. After that, the authors in [8] further improves modality unification by transforming text, tables, and images into a shared textual space with rationale, enabling end-to-end multimodal QA using text-to-text generation models (T5). This line of work emphasizes simplicity and scalability by reducing multimodal QA to a unified generation problem.

Despite their differences, all three approaches reflect a common trend, shifting from modality-specific modeling toward unified text-centric multimodal reasoning, driven by the strong generalization capability of PLMs. However, challenges remain in preserving fine-grained multimodal information, maintaining cross-modal alignment, and handling complex conversational dependencies.

This paper provides a structured review of recent progress in multimodal and conversational question answering over heterogeneous sources. The main contributions are:

• We present a systematic comparison of three representative MMQA/MMCoQA frameworks, focusing on how they handle text, table, and image modalities within QA systems.

![](images/c340d76b26040c9d7ebea565150c5f288ffd25dff0af9165bf893573449a3d9b.jpg)  
Fig. 1: Visualization of Multimodal Adaptive Extraction (MAE) Pipeline [6].

• We analyze the evolution toward unified multimodal representations, highlighting how recent methods increasingly reformulate multimodal QA as a text-centric problem using pre-trained language models.

• We identify the key limitations of existing approaches, including information loss during modality transformation, limited cross-modal alignment, and challenges in conversational reasoning across heterogeneous sources.

The remainder of the paper is organized as follows. Section III shows the methodological evoluation in the field of MMQA. Sections IV and V describe the experimental setup and results respectively. Finally, Section VI concludes the paper.

## II. LITERATURE REVIEW

The research field of multimodal question answering develops through three main research areas which include conversational question answering, multimodal reasoning across different types of sources, and multimodal large language models (MLLMs) .

The research field of conversational QA established its early benchmarks through the development of CoQA [9] and QuAC [10] which demonstrated the necessity of dialectical context modeling. The latest research seeks to enhance contextual comprehension and system durability through the application of PLMs together with retrieval-augmented methods [11]. However, most current conversational QA systems still restricts their functions to handling text input, thus they fail to support multimodal reasoning capabilities.

The development of MMQA progressed through the creation of datasets which require simultaneous reasoning in textual content, tabular data and visual elements. MultimodalQA [5] and ManyModalQA [4] demonstrate the necessity of performing compositional reasoning to access multiple types of information. Recent research shows a growing trend toward using unified system designs together with extensive pretrained models to enhance cross-modal reasoning capabilities and system scalability. The process of connecting different modalities and managing various data distribution patterns presents ongoing difficulties.

Vision-language pretraining has made significant advancements through the development of large-scale models which learn from image-text pairs. The systems BLIP-2 [12] and PaLI [13] utilize pretrained vision encoders together with extensive language models to achieve high-performance results under zero-shot and few-shot testing conditions. The research shows that increasing the size of multimodal pretraining programs results in better outcomes for various tasks. However, the systems focus on perception and generation tasks but they lack optimization for intricate multi-hop reasoning tasks.

More recently, the emergence of MLLMs has reshaped the field. The various models which include Kosmos-1 [14], Kosmos-2 [15], LLaVA [16], and MiniGPT-4 [17] use instruction tuning and cross-modal alignment to enable large language models to understand visual information. The systems can process multiple types of input simultaneously while executing different reasoning tasks within one system. Nevertheless, they often exhibit limitations in factual grounding and may produce hallucinated outputs when reasoning over complex multimodal contexts.

Another line of research focuses on grounded and structured multimodal reasoning, aiming to improve the reliability and interpretability of model predictions. Researchers in recent studies investigate how to combine structured data representations with retrieval systems to improve reasoning performance in multimodal systems according to sources [18], [19], [20]. The methods boost task performance through their specific capabilities, but they need extra supervision and special system designs, which restrict their application to other situations.

## III. METHODOLOGICAL EVOLUTION

Recent multimodal QA approaches have evolved from modality-adaptive architectures such as MAE [6] to unified language-based frameworks like Solar [7], and further toward end-to-end reasoning-enhanced models such as UniMMQA [8].

## A. Multimodal Adaptive Extraction: MAE [6]

To address the challenges of MMCoQA, the Multimodal Conversational Question Answering system with Adaptive

![](images/0c5584fc686609288e4bc9e0e820df0657792bf0a7244e24237b224486422bfd.jpg)  
Fig. 2: Visualization of Solar Pipeline [7].

Extractors (MAE) [6] introduces a baseline framework that explicitly models modality selection, contextual understanding, and answer extraction across heterogeneous sources. The system is designed to operate over a collection of multimodal evidences, including textual passages, tables, and images, while incorporating conversational history to resolve contextdependent queries.

Let the current question at turn k be denoted as $q _ { k }$ and the corresponding conversational history be $\begin{array} { r l } { H _ { k } } & { { } = } \end{array}$ $\left\{ q _ { 1 } , a _ { 1 } , \dots , q _ { k - 1 } , a _ { k - 1 } \right\}$ . The system is also given a multimodal knowledge collection $C = \{ C _ { p } \cup C _ { t } \cup C _ { i } \}$ , where $C _ { p } .$ $C _ { t } .$ , and $C _ { i }$ represent passages, tables, and images, respectively. The objective is to retrieve relevant evidence from C and generate the answer $\hat { a } _ { k }$ for the current query.

At a high level, the MAE framework decomposes the task into three sequential components as shown in Fig. 1: (i) conversational question understanding, (ii) multimodal evidence retrieval, and (iii) adaptive answer extraction.

Conversational Question Understanding: First, the model incorporates conversational context by reformulating the current query using information from $H _ { k }$ , allowing it to resolve coreference and implicit references. The reformulated query $q _ { k } ^ { \prime }$ is then encoded into a dense vector representation using a transformer-based encoder, i.e., $v _ { q } = W _ { q } F _ { q } ( q _ { k } ^ { \prime } )$

Multimodal Evidence Retrieval: Modality-specific encoders are used to transform different types of knowledge into a unified embedding space. For textual passages, standard transformer-based encoders are applied directly to obtain semantic representations. For tables, each table is first linearized into a sequence format using row-wise concatenation of headers and cell values and then processed as text input by the encoder. For images, visual features are extracted using a pretrained ResNet, which converts each image into a fixed-dimensional feature vector. These representations are projected into the same embedding space and obtained semantic representation $v _ { p } , v _ { t } , v _ { q }$ for knowledge collection of textual passages, tables, and images respectively.

After that, the similarity scores $s _ { a }$ between the question representation $v _ { q }$ and all candidate knowledge items is calculated, and selects the $\mathrm { t o p } { - } N _ { r }$ most relevant items as evidence. These retrieved items may originate from different modalities, providing diverse supporting information for answering the question.

Adaptive Answer Extraction: A key aspect of this design is the adaptive selection process, which determines the most suitable extractor based on the input query and retrieved evidences. The model first predicts the most suitable modality for the question as a multi-class classification task, producing a probability distribution over text, table, and image modalities. Based on this prediction with highest probability $( s _ { b } )$ , a corresponding extractor is selected. The TextExtractor and TableExtractor operate as span prediction models that identify answer spans within the input sequences, while the ImageExtractor combines visual features with textual representations of candidate answers to infer the most probable answer with extraction probability $s _ { c } .$ Finally, Each candidate answer is assigned a score as a sum of retrieval relevance $( s _ { a } )$ modality confidence $( s _ { b } )$ , and extraction probability $( s _ { c } )$ , and the highest-scoring answer is returned as the final output.

Despite its modular design, MAE has several limitations. Although it retrieves evidence from multiple modalities, the model ultimately relies on a single predicted modality during the extraction stage. This design restricts its ability to handle questions that require joint reasoning across multiple modalities, where complementary information must be integrated. Furthermore, the reliance on modality prediction introduces potential errors, as incorrect classification may lead to the selection of an inappropriate extractor and degrade overall performance. These limitations highlight the need for more advanced approaches that enable unified cross-modal reasoning in question answering.

B. Unified Language Representation with Retrieval-Augmented Reasoning: Solar [7]

The Solar framework [7] presents multimodal question answering through transforming all modalities including text, tables, and images into one unified linguistic representation, followed by a unified QA pipeline operating solely over text.

![](images/7011c1fe30e9758fe4298c0d8cc83082a105439bca7bd7552faca140b727cad4.jpg)  
Fig. 3: Visualization of UniMMQA Pipeline [8].

Unified Language Representation (Modality Conversion Layer): The first core component of Solar transforms all input modalities into sequences of natural language, which enables modality-agnostic downstream processing.

Solar uses natural language templates to create deterministic linearization transformation for tabular data. The table data transforms into complete sentences through the process of converting each column and its respective value into spoken language. For example, from Fig. 2 the first row is converted into ‘Row one’s race is Santa Derby, the track is Santa Park’.

For images, the transformation is inherently lossy due to the absence of a native linguistic structure. The dual textualization system of Solar using both global and local representation provides mitigation of this problem. The global representation uses an image captioning model to create a complete scene description which includes necessary background information from the image title. The global captions provide an incomplete view of the information needed for QA because they miss essential detailed content. The local representation contains semantic elements that the object–attribute detection model has extracted through its detection of object types and attributes and actions. The local descriptors function as evidence at the micro-level which supports the main argument. The system creates an image representation by joining all global and local textual components through punctuation marks which results in a complete multi-level textual representation of the original image. The dual strategy preserves both complete contextual information and specific details which support reasoning processes.

For questions, especially in conversational settings, Solar does not treat each query independently. Instead, it constructs a contextualized query representation by concatenating the current question with prior dialogue history (previous questions and answers). This enables the model to implicitly capture coreference, ellipsis, and conversational dependencies within a purely textual formulation.

Unified Multimodal QA Model: The process of mapping different modalities into the same textual space Z enables Solar to redefine multimodal QA as a retrieval-augmented text generation (RAG) problem. The pipeline consists of three sequential stages:

1) Retrieval: The system first identifies a subset of relevant textual evidence from the unified space. The system uses dense retrieval mechanism to encode both the query and textual evidence into a shared embedding space through BERT. The system calculates similarity between the query and candidate clues through inner product computation which results in selection of top-K most relevant clues. The process establishes evidence from different sources to support the query which enables effective search space reduction.

2) Ranking: The retrieved candidates may still contain noise or weakly relevant information. Solar introduces its cross-encoder ranking model which simultaneously evaluates query input together with each candidate clue. The ranking model uses cross-attention to process the combined sequence of query and clue elements which enables detailed token interactions. Each candidate is assigned a relevance score via a sigmoid classifier. The top-N ranked clues are retained for the final stage.

3) Generation: The final answer is produced using a sequence-to-sequence generator (e.g., T5). The model receives its input through the combination of the query and the highest-ranked clues. The different types of clues now appear in the same format which allows for uninterrupted logical deductions between them.

However, the framework requires high-quality modality-totext conversion because image captioning and object detection errors lead to prediction errors which affect the entire pipeline. The process of image textualization creates inherent data loss because vital visual elements including spatial relationships and subtle visual elements cannot be represented in textual format. Furthermore, the process of transforming structured data into textual format results in a loss of clear structural connections that exist in the original data. The multi-stage retrieve–rank–generate pipeline creates error accumulation through its design because it lacks ability to retrieve missing evidence during the retrieval process.

C. Unified Language Representation with Explicit Rationaleenhanced Reasoning: UniMMQA [8]

The UniMMQA framework [8] is designed to solve MMQA by transforming heterogeneous inputs, including text, tables, and images into a unified textual representation and then leveraging a text-to-text PLMs for answer generation. Given a question $Q$ and a multimodal context $C = \{ I , T , P \}$ , where I, T, and $P$ denote image, table, and passage respectively, the framework systematically converts each modality into text while preserving as much structural and semantic information as possible. As illustrated in the 3, all transformed components are concatenated into a unified sequence and processed by a generative PLM (T5) to produce the final answer.

Diversified Image-to-Text Transformation: This is a core component of UniMMQA, which goes beyond conventional single-caption generation. UniMMQA first utilizes OCR technology to extract all visible textual components from images which include hidden texts that are found on signs and labels. Then, the system implements a vision-language captioning model which uses BLIP technology through a decoding method that combines Top-K [21] and Top-p [22] sampling techniques instead of taking greedy one. The system generates each subsequent token through a probability distribution $\begin{array} { r l } { P ( I _ { t } ^ { t e x t } \ \mid } & { { } I _ { 1 : t - 1 } ^ { t e x t } ) } \end{array}$ which first selects from the most probable tokens through Top-K and then enables multiple token selection through Top-p for balanced output. The system generates multiple captions which work together to show detailed visual information through their respective captions which display specific objects and background details that traditional systems would not display. The textual content of the captions together with the OCR results creates an enhanced written representation of visual material.

Table-to-Text Conversion: The UniMMQA system does not use template-based summarization because this method fails to maintain essential attributes needed for the task. The system uses a position-enhanced linearization method which creates a representation of content together with structural elements. The table serialization process organizes data in row format while using distinct indicators to define table elements. The table starts with headers marked as “header :” followed by rows which use the indexing system $( \mathrm { e . g . , \tilde { \ r o w \ 1 : } } \ . . . \ ^ { \mathrm { 3 } } )$ to identify their position. The table with M rows and N columns transforms into a sequence which maintains all cell contents together with their original positions. The system uses explicit encoding to protect relationships between column alignment with row identity which enables the model to retrieve precise data about attributes including height through template-based methods which discard unselected columns.

Multimodal Rationale Generation: Beyond simple modality unification, UniMMQA introduces a multimodal rationale generator to explicitly model cross-modal interactions. The system creates natural language explanations as intermediate outputs which describe how different modalities connect with each other through their relationships. A vision encoder (CLIP)

extracts visual features $I _ { f e a t u r e }$ from the system while the system uses a transformer-based language encoder to encode textual representations which include passages and linearized tables into $P _ { f e a t u r e }$ . The system combines these two representations which then proceed to the generative decoder for producing the rationale $R \ : = \ : F _ { r } ( I _ { f e a t u r e } , P _ { f e a t u r e } )$ , where $F _ { r }$ is the rationale generation model.

Answer Generation: The answer generation stage processes the complete problem as a sequence-to-sequence assignment. The question Q together with the unified textual context $C ^ { t e x t } \ = \ \{ I ^ { t e x t } , T ^ { t e x { \bar { t } } } , P , R \}$ creates a single input sequence that is passed to a pretrained model T5. The model uses negative log-likelihood as its fine-tuning objective to produce output tokens which match the ground truth answer sequence.

Despite its effectiveness, UniMMQA suffers from several limitations. The conversion of images into text, even with diversified captioning, may still miss fine-grained visual details (e.g., exact object counts), leading to incorrect reasoning. Similarly, although position-enhanced table linearization preserves all cell information, it significantly increases input length and complexity, which can negatively affect model efficiency and reasoning performance. Furthermore, both the image captioning and rationale generation modules are prone to hallucination, potentially introducing inaccurate or fabricated information that misguides the final answer generation.

## IV. EXPERIMENTAL SETUP

## A. Datasets

Experiments were conducted on several public benchmark datasets, including ManymodalQA [4], MultimodalQA [5], and MMConvQA [6]. ManymodalQA contains 10,190 questions, including 2,873 images, 3,789 text and 3,528 tables, which have been split into train/dev sets. MultimodalQA consists of 29,918 question-answer pairs cross image, table and text modalities, 35.7% of which require cross-modal reasoning. Extending the MultimodalQA [5] benchmark, MM-ConvQA contains 1,179 conversations and 5,753 questionanswering pairs. There are 218,285 passages, 10,042 tables and 57,058 images in MMConvQA, and about 24.4% of conversations require reasoning cross three modalities.

## B. Evaluation Metrics

Exact Match (EM) and F1 were adopted as the evaluation metrics for all datasets. EM measures whether the predicted answer exactly matches the ground-truth answer after normalization, assigning a score of 1 for an exact match and 0 otherwise. In contrast, F1 evaluates the token-level overlap between the predicted and ground-truth answers by considering both precision and recall, thereby providing partial credit for answers that are not exact but share common tokens.

## C. Baseline Models

Several baseline models were used for comparison, including open-retrieval conversational QA system (ORConvQA) [23], multimodalQA model (ManyModalQA) [4], Multi-Hop Implicit Decomposition (Implicit-Decomp) [5], MMQA-T5-Large [24], Pre-traind for Reasoning Model (PReasM)

[24], Structured Knowledge and Unified Retrieval-Generation (SKURG) [25], Multimodal Graph Transformer (MGT) [26].

ORConvQA [23] addresses conversational question answering in an open-retrieval setting by integrating a trainable retriever, a reranker, and a reader all based on Transformer, enabling end-to-end optimization of the retrieval and answer extraction process. In contrast, ManymodalQA [4] adopts a modality-aware pipeline, where a question-type classifier first identifies the relevant modality and then routes the query to the corresponding unimodal QA system.

Implicit-Decomp [5] uses programming techniques to break down complex questions into multiple reasoning steps that it solves through different modalities. The system generates programs which execute various single-modal and multi-modal QA components to produce the ultimate response. MMQA-T5 system uses sequence-to-sequence modeling together with table linearization to train T5 for multimodal QA applications. The system uses Implicit-Decomp for image-based questions which lets it integrate generation-based and decompositionbased methods.

PReasM [24] improves T5 through synthetic tabular data which enhances its reasoning abilities but maintains the existing routing system for processing image-based inquiries. SKURG [25] uses knowledge graphs to model connections between different modalities which enables more organized and linked reasoning processes. MGT [26] combines graphbased models with Transformer systems to enable multimodal graph learning which analyzes interactions among diverse data types.

TABLE I: Experimental results on MultimodalQA.
<table><tr><td rowspan="2">Method</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>Implicit-Decomp</td><td>48.8</td><td>55.5</td><td>49.3</td><td>55.9</td></tr><tr><td>MMQA-T5-Large</td><td>57.9</td><td>64.3</td><td>57.0</td><td>63.4</td></tr><tr><td>PReasM-Large</td><td>59.0</td><td>65.5</td><td>58.3</td><td>64.6</td></tr><tr><td>SKURG</td><td>59.4</td><td>63.8</td><td>一</td><td>一</td></tr><tr><td>MGT</td><td>52.1</td><td>57.7</td><td>一</td><td>一</td></tr><tr><td>Solar</td><td>59.8</td><td>66.1</td><td>一</td><td>一</td></tr><tr><td>UniMMQA (T5-Base)</td><td>67.9</td><td>74.0</td><td>67.0</td><td>72.9</td></tr><tr><td>UniMMQA (T5-Large)</td><td>71.3</td><td>77.1</td><td>70.1</td><td>75.8</td></tr><tr><td>UniMMQA (T5-3B)</td><td>75.5</td><td>81.7</td><td>73.7</td><td>80.1</td></tr></table>

## V. EXPERIMENTAL RESULT

## A. Overall Performance

MultimodalQA: Table I presents a comparison of UniM-MQA and Solar against several baseline methods on the MultimodalQA dataset. The baseline testing results show that Implicit-Decomp performs poorly because it can only process multimodal reasoning through its separate uni-modal and bimodal components. The unification of textual and tabular modalities in MMQA-T5 and PReasM shows that these models achieve major performance improvements through their partial modality integration method. The graph-based SKURG and MGT approaches cannot exceed the performance of these models because structural connections require effective joint representation learning to achieve success.

TABLE II: Experimental results on MMCoQA.
<table><tr><td rowspan="2">Method</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>ORConvQA</td><td>1.2</td><td>3.0</td><td>1.1</td><td>1.9</td></tr><tr><td>ManymodelQA</td><td>2.3</td><td>0.7</td><td>1.8</td><td>1.0</td></tr><tr><td>MAE</td><td>19.8</td><td>26.8</td><td>22.0</td><td>28.3</td></tr><tr><td>Solar</td><td>56.8</td><td>62.5</td><td>57.3</td><td>64.6</td></tr><tr><td>UniMMQA (T5-Base)</td><td>57.8</td><td>64.7</td><td>59.2</td><td>64.9</td></tr><tr><td>UniMMQA (T5-Large)</td><td>62.3</td><td>69.0</td><td>63.6</td><td>70.0</td></tr><tr><td>UniMMQA (T5-3B)</td><td>65.5</td><td>71.4</td><td>66.7</td><td>72.6</td></tr></table>

Solar provides further improvement and achieves competitive results, demonstrating incremental progress toward more effective multimodal understanding. However, its benefits remain restricted. UniMMQA shows superior performance to Solar and all baseline techniques because it uses a unified framework to model all three modalities with rationale. The performance of T5 shows continuous improvement until it reaches T5-3B, which demonstrates how UniMMQA can handle different pre-trained language model sizes. The results demonstrate that multimodal question answering achieves optimal performance through complete modality integration instead of using partial system integration or structural system enhancements.

TABLE III: Experimental results on ManymodalQA.
<table><tr><td>Method</td><td>EM</td><td>F1</td></tr><tr><td>ManymodalQA</td><td>39.7</td><td></td></tr><tr><td>ManymodelQA (w/ type)</td><td>46.3</td><td></td></tr><tr><td>UniMMQA (T5-Base)</td><td>45.4</td><td>45.7</td></tr><tr><td>UniMMQA (T5-Large)</td><td>50.0</td><td>50.4</td></tr><tr><td>UniMMQA (T5-3B)</td><td>52.1</td><td>52.4</td></tr></table>

MMCoQA: Table II shows the performance results of MAE, Solar, and UniMMQA against baseline methods on the MMCoQA test-dev set. The baseline models ORConvQA and ManymodelQA show extremely low EM and F1 scores, which demonstrate their inability to perform multimodal conversational question answering tasks that require multiple information sources to be integrated. Their poor performance highlights the inherent difficulty of the task when multimodal reasoning is not effectively modeled.

The introduction of MAE brings a substantial improvement over these baselines, with a large jump in both EM and F1 scores. The system demonstrates its operational success through its modular pipeline, which enables conversational comprehension and adaptive extraction processes. The existing system shows improved performance through its advancements yet still falls behind unified systems which perform at higher levels.

Solar further advances performance significantly, achieving strong results on both Development and Test sets. The study shows that better multimodal integration together with RAG increases answer quality. Nevertheless, UniMMQA maintains superior performance over Solar in all testing conditions. The smallest UniMMQA (T5-Base) version achieves better results than Solar while the larger T5-Large and T5-3B versions provide additional performance improvements.

TABLE IV: Performance comparison of UniMMQA across datasets and modalities (Image, Table, Text).
<table><tr><td>Dataset</td><td>Model</td><td colspan="2">Image</td><td colspan="2">Table</td><td colspan="2">Text</td><td colspan="2">Overall</td></tr><tr><td></td><td></td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td rowspan="3">MultimodalQA</td><td>UniMMQA (T5-Base)</td><td>66.7</td><td>69.2</td><td>65.5</td><td>71.4</td><td>68.3</td><td>76.4</td><td>67.9</td><td>74.0</td></tr><tr><td>UniMMQA (T5-Large)</td><td>69.8</td><td>72.2</td><td>69.9</td><td>75.6</td><td>71.9</td><td>80.0</td><td>71.3</td><td>77.1</td></tr><tr><td>UniMMQA (T5-3B)</td><td>73.8</td><td>76.7</td><td>73.4</td><td>79.7</td><td>76.2</td><td>84.1</td><td>75.5</td><td>81.7</td></tr><tr><td rowspan="3">ManymodalQA</td><td>UniMMQA (T5-Base)</td><td>46.6</td><td>46.9</td><td>60.7</td><td>61.1</td><td>30.2</td><td>30.4</td><td>45.4</td><td>45.7</td></tr><tr><td>UniMMQA (T5-Large)</td><td>48.5</td><td>48.6</td><td>67.5</td><td>68.2</td><td>34.9</td><td>35.1</td><td>50.0</td><td>50.4</td></tr><tr><td>UniMMQA (T5-3B)</td><td>49.8</td><td>50.2</td><td>58.0</td><td>58.3</td><td>40.9</td><td>41.3</td><td>52.1</td><td>52.4</td></tr><tr><td rowspan="3">MMConvQA</td><td>UniMMQA (T5-Base)</td><td>73.2</td><td>75.5</td><td>33.5</td><td>40.5</td><td>66.8</td><td>76.3</td><td>57.8</td><td>64.7</td></tr><tr><td>UniMMQA (T5-Large)</td><td>73.2</td><td>75.4</td><td>38.9</td><td>46.8</td><td>73.3</td><td>81.7</td><td>62.3</td><td>69.0</td></tr><tr><td>UniMMQA (T5-3B)</td><td>73.3</td><td>75.5</td><td>41.9</td><td>47.9</td><td>77.8</td><td>85.1</td><td>65.8</td><td>71.4</td></tr></table>

The continuous performance boost that results from using larger model sizes demonstrates that UniMMQA successfully utilizes larger PLMs to enhance its multimodal reasoning capabilities. The results show that both EM and F1 metrics experienced improvement because the system achieved better exact answer matching and improved semantic correctness. The results show a clear development path that starts from basic weak baselines and progresses to modular systems that use MAE and then reaches complete unified systems which include UniMMQA because it delivers the best performance with scalable results.

ManymodalQA: The performance comparison for the ManymodalQA dataset is demonstrated in Table III, which shows how UniMMQA performs against baseline methods. The baseline model, ManymodalQA, shows reasonable performance by leveraging heuristic strategies, while its variant with question type information (ManymodalQA w/ type) achieves a noticeable improvement. The process of selecting which modality to use for answering queries becomes crucial because explicit type information enables better access to appropriate uni-modal QA components.

On the other hand, UniMMQA utilizes unified framework instead of using explicit question type classification. Despite this, it achieves competitive and generally superior performance compared to the standard ManymodalQA baseline. The system demonstrates improved EM and F1 score performance as its model size progresses from T5-Base to T5-3B because it can better understand cross-modal interactions while using more extensive pre-trained language models.

However, UniMMQA (T5-Base) fails to exceed the performance of ManymodalQA w/ type because explicit modality supervision maintains its usefulness in specific situations. The research results demonstrate that heuristic and typeaware methods function as effective baseline methods, while UniMMQA unified modeling system delivers dependable performance which can expand its capabilities without needing extra help to identify different modes of operation.

## B. Modality-wise Performance

Model performance across questions that require information from different modalities is summarized in Table IV. The UniMMQA system performs better with text-based questions in the MultimodalQA dataset than with image and table-based questions despite its strong overall performance. The model proves better at processing textual information because its visual and tabular data processing capabilities remain restricted. The information loss occurs because images and tables must convert into textual formats which eliminate essential elements that enable accurate reasoning. The process of uniting different modal elements into one reasoning system presents difficulties that need to be solved.

On the ManymodalQA dataset, a different pattern emerges. The study shows that table-based questions produce better results than image-based and text-based questions because their performance exceeds the results shown in MultimodalQA for each specific technology. The ManymodalQA tables use simple table formats because they show only few rows of data, which makes table information easier to understand. The related passages present lengthy and complex content, which creates obstacles for people to understand context and perform reasoning tasks. The model obtains greater advantages from using structured tabular data than from using unstructured text data in this particular situation.

On the other hand, performance on table-based questions in MMConvQA lead to a major drop in performance, while image and text-based questions produce better results. The process of conversational reasoning for tabular data presents multiple challenges because it requires users to handle both table data and evolving dialogue information during their attempt to extract and combine structured content.

Overall: Overall, the results demonstrate that UniMMQA performance depends on the complexity and structure of its underlying modality. The model performs better with textual inputs yet its performance differs across datasets because of varying success in linearizing and combining different modalities. The results demonstrate that unified QA systems require better structural and visual semantics preservation methods and improved multimodal fusion techniques.

## VI. CONCLUSION

This paper presents a comparative analysis of MAE, Solar, and UniMMQA, highlighting the evolution of multimodal question answering from modular pipelines to unified textcentric frameworks. MAE establishes a strong foundation through explicit modality-aware retrieval and extraction, but remains limited by its reliance on single-modality decisions.

Solar advances this direction by transforming heterogeneous inputs into a unified textual space, enabling joint reasoning through a retrieve–rank–generate pipeline, although it suffers from information loss and error propagation. UniMMQA further improves upon this paradigm by incorporating richer modality-to-text transformations and explicit rationale generation, resulting in more robust and scalable performance across benchmarks. Overall, the findings demonstrate that unified approaches leveraging pre-trained language models are more effective for multimodal reasoning. Nevertheless, challenges such as preserving fine-grained multimodal information, mitigating hallucination, and improving computational efficiency remain open, providing important directions for future research.

## ACKNOWLEDGMENT

The author acknowledges the use of ChatGPT (OpenAI) as an assistive tool for language refinement, and conceptual structuring. All generated content was critically reviewed, validated, and appropriately adapted by the author, who takes full responsibility for the accuracy and integrity of the work.

## REFERENCES

[1] H. Abdel-Nabi, A. Awajan, and M. Z. Ali, “Deep learning-based question answering: a survey,” Knowledge and Information Systems, vol. 65, no. 4, pp. 1399–1485, 2023.

[2] Z. Shao, Z. Yu, M. Wang, and J. Yu, “Prompting large language models with answer heuristics for knowledge-based visual question answering,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 14 974–14 983.

[3] W. Li, W. Li, and Y. Wu, “A unified model for document-based question answering based on human-like reading strategy,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 32, no. 1, 2018.

[4] D. Hannan, A. Jain, and M. Bansal, “Manymodalqa: Modality disambiguation and qa over diverse inputs,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 05, 2020, pp. 7879– 7886.

[5] A. Talmor, O. Yoran, A. Catav, D. Lahav, Y. Wang, A. Asai, G. Ilharco, H. Hajishirzi, and J. Berant, “Multimodalqa: Complex question answering over text, tables and images,” arXiv:2104.06039, 2021.

[6] Y. Li, W. Li, and L. Nie, “Mmcoqa: Conversational question answering over text, tables, and images,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 4220–4231.

[7] B. Yu, C. Fu, H. Yu, F. Huang, and Y. Li, “Unified language representation for question answering over text, tables, and images,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 4756–4765.

[8] H. Luo, Y. Shen, and Y. Deng, “Unifying text, tables, and images for multimodal question answering,” in Findings of the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 9355–9367.

[9] S. Reddy, D. Chen, and C. D. Manning, “Coqa: A conversational question answering challenge,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 249–266, 2019.

[10] E. Choi, H. He, M. Iyyer, M. Yatskar, W.-t. Yih, Y. Choi, P. Liang, and L. Zettlemoyer, “Quac: Question answering in context,” in Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, 2018, pp. 2174–2184.

[11] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt¨ aschel¨ et al., “Retrievalaugmented generation for knowledge-intensive nlp tasks,” Advances in Neural Information Processing Systems, vol. 33, pp. 9459–9474, 2020.

[12] J. Li, D. Li, S. Savarese, and S. Hoi, “Blip-2: Bootstrapping languageimage pre-training with frozen image encoders and large language models,” in International Conference on Machine Learning. PMLR, 2023, pp. 19 730–19 742.

[13] X. Chen, X. Wang, S. Changpinyo, A. J. Piergiovanni, P. Padlewski, D. Salz, S. Goodman, A. Grycner, B. Mustafa, L. Beyer et al., “Pali: A jointly-scaled multilingual language-image model,” arXiv:2209.06794, 2022.

[14] S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K. Mohammed, B. Patra et al., “Language is not all you need: Aligning perception with language models,” Advances in Neural Information Processing Systems, vol. 36, pp. 72 096–72 109, 2023.

[15] Z. Peng, W. Wang, L. Dong, Y. Hao, S. Huang, S. Ma, and F. Wei, “Kosmos-2: Grounding multimodal large language models to the world,” arXiv:2306.14824, 2023.

[16] H. Liu, C. Li, Q. Wu, and Y. J. Lee, “Visual instruction tuning,” Advances in Neural Information Processing Systems, vol. 36, pp. 34 892– 34 916, 2023.

[17] D. Zhu, J. Chen, X. Shen, X. Li, and M. Elhoseiny, “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv:2304.10592, 2023.

[18] B. Liu, C. Lyu, Z. Min, Z. Wang, J. Su, and L. Wang, “Retrievalaugmented multi-modal chain-of-thoughts reasoning for large language models,” in 2025 International Joint Conference on Neural Networks (IJCNN). IEEE, 2025, pp. 1–8.

[19] W. Cui, K. Bi, J. Guo, and X. Cheng, “More: Multi-modal retrieval augmented generative commonsense reasoning,” in Findings of the Association for Computational Linguistics: ACL 2024, 2024, pp. 1178– 1192.

[20] J. Lee, Y. Wang, J. Li, and M. Zhang, “Multimodal reasoning with multimodal knowledge graph,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 10 767–10 782.

[21] A. Fan, M. Lewis, and Y. Dauphin, “Hierarchical neural story generation,” in Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2018, pp. 889–898.

[22] A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The curious case of neural text degeneration,” arXiv:1904.09751, 2019.

[23] C. Qu, L. Yang, C. Chen, M. Qiu, W. B. Croft, and M. Iyyer, “Openretrieval conversational question answering,” in Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 2020, pp. 539–548.

[24] O. Yoran, A. Talmor, and J. Berant, “Turning tables: Generating examples from semi-structured tables for endowing language models with reasoning skills,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 6016–6031.

[25] Q. Yang, Q. Chen, W. Wang, B. Hu, and M. Zhang, “Enhancing multimodal multi-hop question answering via structured knowledge and unified retrieval-generation,” in Proceedings of the 31st ACM International Conference on Multimedia, 2023, pp. 5223–5234.

[26] X. He and X. Wang, “Multimodal graph transformer for multimodal question answering,” in Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, 2023, pp. 189–200.