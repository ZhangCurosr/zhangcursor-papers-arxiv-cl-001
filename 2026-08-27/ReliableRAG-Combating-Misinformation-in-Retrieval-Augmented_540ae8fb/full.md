# ReliableRAG: Combating Misinformation in Retrieval-Augmented Generation via Reliability-Guided Reasoning Chains

Jinpu Jiang<sup>∗</sup> College of Software, Jilin University Changchun, Jilin, China jiangjp24@mails.jlu.edu.cn

Bo Yang   
Key Laboratory of Symbolic   
Computation and Knowledge   
Engineering of Ministry of Education,   
College of Computer Science and   
Technology, Jilin University   
Changchun, Jilin, China   
ybo@jlu.edu.cn   
Heow Pueh Lee   
Department of Mechanical   
Engineering, National University of   
Singapore   
Singapore   
mpeleehp@nus.edu.sg   
Xuan Wu<sup>∗</sup>   
College of Computer Science and   
Technology, Jilin University   
Changchun, Jilin, China   
wuuu22@mails.jlu.edu.cn   
You Zhou<sup>†</sup>   
Key Laboratory of Symbolic   
Computation and Knowledge   
Engineering of Ministry of Education,   
College of Computer Science and   
Technology, Jilin University   
Changchun, Jilin, China   
zyou@jlu.edu.cn   
Yanchun Liang   
School of Computer Science, Zhuhai   
College of Science and Technology   
Zhuhai, Guangdong, China   
ycliang@jlu.edu.cn

Wenhao Song College of Software, Jilin University Changchun, Jilin, China songwh23@mails.jlu.edu.cn

Hongwei Ge   
College of Computer Science and   
Technology, Dalian University of   
Technology   
Dalian, Liaoning, China   
hwge@dlut.edu.cn   
Chunguo Wu<sup>†</sup>   
Key Laboratory of Symbolic   
Computation and Knowledge   
Engineering of Ministry of Education,   
College of Computer Science and   
Technology, Jilin University   
Changchun, Jilin, China   
wucg@jlu.edu.cn

## Abstract

Retrieval-Augmented Generation (RAG) has emerged as a potent system architecture for addressing Question Answering (QA) tasks by integrating external information into Large Language Models (LLMs). However, the proliferation of false, inaccurate, and misleading information in news and social media poses a challenge to real-world RAG systems, particularly in multi-hop QA where complex, multi-step reasoning is easily misled by even a single deceptive misinformation segment within the misinformation-polluted source documents. While existing approaches primarily adopt paradigms of implicit alignment or explicit regulation to mitigate this issue, their inability to perceive fine-grained information reliability makes them highly susceptible to fine-grained deceptive misinformation that is semantically aligned with the question but factually incorrect, ultimately leading to the generation of erroneous answers. To address this limitation, we propose ReliableRAG, which, to the best of our knowledge, is the first reliability-driven framework designed to prevent the misleading efects of deceptive misinformation in multi-hop QA through the evaluation of fine-grained individual triples. Specifically, ReliableRAG first extracts information segments from source documents and represents them as fine-grained structured triples. It then performs fine-grained reliability quantification by synthesizing query-triple semantic relevance and triple credibility, ensuring the retention of only the top-� most reliable and non-redundant triples. By autoregressively constructing robust reasoning chains from these refined triples, the framework efectively consolidates trustworthy information and filters out deceptive misinformation, ensuring accurate answers that is faithful to reliable information. Experimental results on three multi-hop QA datasets demonstrate that ReliableRAG outperforms existing methods, significantly enhancing the factual reliability and robustness of RAG systems under deceptive misinformation injection.

CCS Concepts

• Computing methodologies → Natural language processing;   
• Information systems → Information retrieval.

## Keywords

Retrieval-Augmented Generation, Multi-hop QA, Misinformation Robustness, Reasoning Chains

## 1 Introduction

Retrieval-Augmented Generation (RAG), a potent system architecture for Question Answering (QA) tasks, empowers Large Language Models (LLMs) by incorporating external information [15, 30, 46]. To resolve multi-hop QA tasks that require complex, multi-step reasoning [9, 33, 42], this system requires retrieving the top-� documents most relevant to the given question � from a vast Web corpus to aggregate complementary information across disparate sources. These documents are then prepended to the question, with the resulting concatenation serving as the input to an LLM-based generator to produce the final answer [28]. However, the reliability of RAG can be substantially compromised when the retrieved documents contain maliciously generated deceptive misinformation, ultimately degrading their overall performance [8, 23, 25, 43], particularly in multi-hop Question Answering (QA) tasks. In such scenarios, even a few deceptive misinformation segments may propagate through the reasoning process, triggering a cascade of errors that corrupts the entire reasoning chain and yields incorrect answers [3, 10, 14, 41].

To improve the robustness ofRAG facing misinformation in multi hop QA, Recent studies primarily adopt two paradigms, namely implicit alignment and explicit regulation. Specifically, implicit alignment methods achieve this by training models to internalize criteria such as self-reflection signals, learned decision policies, or credibil ity awareness, enabling them to autonomously regulate reliance on retrieved information during generation and thereby enhance factual accuracy in retrieval-augmented systems [2, 17, 24]. However, these methods necessitate substantial computational resources and high-quality training data which are often dificult to acquire [5]. Moreover, the inherent domain gap between specialized training datasets and real-world test scenarios limits their generalization [44]. In contrast, explicit regulation approaches explicitly regulate the model’s reliance on retrieved information using external signals such as document credibility scores (i.e., the extent to which a document is free from misinformation) or knowledge graph structures, thereby enhancing factual accuracy and robustness in RAG systems [4, 21]. Despite these advancements, current explicit regulation methods exhibit critical shortcomings in perceiving fine-grained information reliability. As illustrated in Figure 1, these approaches often fail to discern between highly relevant but factually incorrect fine-grained deceptive misinformation, leading them to be misled by such misinformation. These misinformation subsequently propagate through and undermine the entire reasoning chain, ultimately yielding inaccurate answers [3, 48]. Therefore, in this work, we investigate the following research question:

![](images/dee32d9806143167e8fe2e7243f1e72d177b9ed2f8d4c48d965f81dcc71b8371.jpg)  
Figure 1: Illustration ofthe diferences between existing methods and ReliableRAG under the interference of misinformation. Green text denotes the correct answer, while red text represents a misleading incorrect answer. While implicit alignment and explicit regulation paradigms often falter when faced with semantically similar fine-grained deceptive misinformation, thereby leading to incorrect answers, ReliableRAG identifies the fine-grained reliability of information and constructs robust reasoning chains to prevent deceptive misinformation, ensuring accurate answers.

Can we equip RAG systems with fine-grained reliability awareness to discern deceptive misinformation within fine-grained information, and thereby prevent its propagation during multi-hop reasoning?

To address this research question, we propose ReliableRAG, which, to the best ofour knowledge, is the first reliability-driven framework designed to prevent the misleading efects of deceptive misinformation in multi-hop QA through the evaluation of fine-grained individual triples. The framework achieves this through three integrated modules: Triple Extraction, Triple Evaluation, and Chain Construction, which operate across two distinct phases. In the offline phase, the Triple Extraction module pre-processes each source document into fine-grained structured triple representations. Sub sequently, an LLM-based evaluator is employed to pre-assess these triples and assign each an interpretable credibility score (i.e., the extent to which a triple is free from misinformation). Collectively, these steps provide fine-grained information and pre-computed factors for subsequent retrieval and inference. During online phase, given the question �, the Triple Evaluation module first dynamically formu lates a unique search query for every individual existing reasoning chain constructed from the preceding hop by prepending its accumulated information to the question �. This one-to-one mapping ensures that each query is uniquely tailored to its corresponding chain to retrieve the supplementary information required to extend that chain further. Leveraging these queries, the module quantifies the reliability of each triple by synthesizing two factors: query-triple semantic relevance and triple credibility. Based on these fine-grained reliability scores, the module selects the top-� most reliable and non-redundant triples for each existing chain, thereby filtering out those containing deceptive misinformation to provide a highly trust worthy search space for constructing candidate reasoning chains for the current hop. Operating within this search space, the Chain Construction module employs a beam search strategy to autoregres sively construct multiple candidate reasoning chains. Specifically, the module employs a multiple-option reasoning prompt to guide an LLM-based selector in deciding, for each existing chain, whether to append the most probable triple from the search space to expand it, or to execute an early termination if the accumulated information within it is already suficient to answer �. Simultaneously, the mod ule updates the confidence score of each candidate chain based on the probability of the selector’s decision. According to these updated scores, only the top-� most confident candidate chains are retained as the input for the subsequent hop. In this manner, the iterative cycle of Triple Evaluation and Chain Construction continues until all chains terminate or a maximum hop limit is reached. Ultimately, the framework synthesizes the final robust chains into a refined and reliable supporting context, denoted as Z. This Z is prepended to the question � to form a unified input X, which is fed into the generator to produce the final answer.

To evaluate the efectiveness of ReliableRAG, we conduct extensive experiments on three multi-hop QA datasets adversarially augmented with deceptive misinformation. The experimental results demonstrate that ReliableRAG outperforms existing methods across all datasets, efectively mitigating the interference of misinformation. Further analyses indicate that by anchoring reasoning in reliable triples, our framework enhances the factual reliability and robustness of the RAG system, enabling more accurate answers even under deceptive misinformation injection.

The key contributions of this work are as follows.

I) We discover that prioritizing fine-grained reliability at the triple level facilitates the filtering of fine-grained deceptive misinformation, thereby preventing its propagation and promoting more robust and reliable multi-hop reasoning.

II) We propose ReliableRAG, which, to the best of our knowledge, is the first reliability-driven framework that features a novel Triple Evaluation module to dynamically quantify fine-grained information reliability at the triple level, thus preventing the misleading efects of deceptive misinformation in multi-hop QA and constructing a highly trustworthy search space.

III) ReliableRAG seamlessly integrates the Triple Evaluation module into the reasoning pipeline, enabling the Chain Construction module to leverage a highly trustworthy search space to autoregressively construct robust, multi-hop reasoning chains that are anchored in reliable triples, thereby preventing misinformation propagation before generating the final answer.

IV) We conduct extensive experiments on three multi-hop QA datasets adversarially augmented with deceptive misinformation.

The results demonstrate that ReliableRAG significantly outperforms existing methods, thereby enhancing the factual reliability and robustness of RAG systems and validating its efectiveness in systematically preventing misinformation propagation throughout the reasoning process.

## 2 Related Work

RAG advancements in multi-hop QA improve performance by facilitating the reasoning process through techniques such as iterative search, Reasoning with Attributions, and preference optimization [16, 22, 34]. However, these systems are vulnerable to misinformation, where plausible distractors or adversarial passages can significantly degrade performance and poison the generation process [3, 31, 36, 47]. To address this, prior work has evolved from early feature-based classification methods [11, 12, 35] to modern LLMbased credibility assessment approaches [26, 27] to detect misinformation. Based on this, recent studies have advanced robustness through two primary paradigms: implicit alignment and explicit regulation [2, 4, 17, 21, 24].

Implicit alignment methods achieve this by training models to internalize criteria such as self-reflection signals, learned decision policies, or credibility awareness, enabling them to autonomously regulate reliance on retrieved information during generation and thereby enhance factual accuracy in retrieval-augmented systems. For instance, Asai et al. [2] introduced a framework that generates reflection tokens to adaptively judge retrieval necessity and output relevance, thereby filtering out erroneous content. Pan et al. [24] employed instruction fine-tuning to teach LLMs to diferentiate and process information according to provided credibility indicators. Similarly, Lin et al. [17] utilized a reinforcement learning approach to balance internal parametric knowledge with external context, allowing the LLM to fall back on its own knowledge when encountering misleading information. However, these methods necessitate substantial computational resources and high-quality training data which are often dificult to acquire [5]. Moreover, the inherent domain gap between specialized training datasets and real-world test scenarios limits their generalization [44]. In contrast, explicit regulation approaches explicitly regulate the model’s reliance on retrieved information using external signals such as credibility scores or knowledge graph structures, thereby enhancing factual accuracy and robustness in RAG systems. Deng et al. [4] mitigated the influence of low-credibility documents by attenuating their corresponding attention weights during the decoding process. However, this approach relies on coarse-grained, document-level credibility scores, failing to perceive the fine-grained information reliability required for robust multi-hop QA. Furthermore, Liu et al. [21] proposed constructing knowledge graphs to identify factual-level conflicts via entropy-based filtering, efectively blocking the propagation of misinformation. Nevertheless, this method retrieves fine-grained structured triples based solely on semantic similarity, which may fail to discern fine-grained deceptive misinformation that is semantically aligned with the question but factually incorrect. Such misinformation subsequently propagates through and undermines the entire reasoning chain, ultimately yielding inaccurate answers [3, 48].

To address these limitations, we introduce ReliableRAG, a trainingfree framework that transforms documents into structured triples to quantify fine-grained reliability via query-triple semantic relevance and triple credibility, constructing robust reasoning chains to consolidate trustworthy information and filter out deceptive misinformation.

## 3 ReliableRAG

Given a multi-hop question �, the objective is to derive a correct answer � by identifying and integrating the top-� documents most relevant to $Q$ from the given set of documents $\mathcal { D } = \{ d _ { 1 } , d _ { 2 } , \dots , d _ { N } \}$ within at most � reasoning hops. These documents are pre-retrieved from external sources such as Wikipedia [34], where each document $d _ { j }$ comprises a title and the corresponding text. However, RAG is vulnerable to misinformative documents that are semantically relevant to $Q$ but factually incorrect, which can mislead the LLM-based generator $M _ { \mathrm { g e n } }$ into producing erroneous answers. To address this limitation, as illustrated in Figure 2, we introduce ReliableRAG, which comprises three core modules, namely Triple Extraction, Triple Evaluation, and Chain Construction. The source code of ReliableRAG is available online anonymously<sup>1</sup>.

## 3.1 Triple Extraction

Although LLMs are capable of processing long contexts, directly reasoning over coarse-grained documents often sufers from the lost-in-the-middle phenomenon [20], wherein LLMs tend to overlook critical information located in the middle of long inputs. This phenomenon is further exacerbated in multi-hop QA tasks, where relevant evidence is scattered across multiple documents, thereby making it easier for the generator to overlook key information and hindering the production of accurate answers.

To address this limitation, we incorporate this module that transforms each document $d _ { i } \in \mathcal { D } _ { i }$ , consisting of a title and text content, into multiple fine-grained structured triples, each comprising a sub ject, predicate, and object. Specifically, we leverage the In-Context Learning (ICL) ability of LLMs [6, 29] to prompt the LLM-based extractor $M _ { \mathrm { e x t } }$ to extract triples (see Appendix G.1 for experiments on various extractors). To facilitate fine-grained processing, the text content of target document $d _ { i }$ is first segmented into individual sentences based on periods. Following recent studies on knowledge graph construction [7, 40, 45], the extraction process is guided by an extraction prompt that incorporates both the document $d _ { i }$ (its title and segmented sentences) and three in-context exemplars. These three exemplars are the ones most semantically similar to the title and text of $\cdot d _ { i } ,$ selected from a manually annotated set while ensuring no overlap with the test datasets. Given this prompt, the extractor $M _ { \mathrm { e x t } }$ identifies the title, along with information segments within each sentence, as entities, subsequently inferring the relationships between them. This strategy leverages the inherent relevance between the title and each sentence within the text content, ensuring that both the extracted entities and their inferred relationships are substantively anchored to the core topic. Through this parallelized process, a single document yields multiple structured triples, which collectively form the set of triples T for the question �:

$$
\mathcal { T } = \{ t _ { j } = ( s _ { j } , p _ { j } , o _ { j } ) \} _ { j = 1 } ^ { M }\tag{1}
$$

where $s _ { j } , p _ { j } , o _ { j }$ denote the subject, predicate, and object of the �-th triple, respectively, with $M \geq K .$ . By adopting this design, this module produces a structured set of triples, thereby mitigating the lostin-the-middle issue. Ablation studies demonstrate that this module significantly contributes to the overall performance (see Section 4.3 for details).

## 3.2 Triple Evaluation

In this subsection, we delineate the process of Triple Evaluation, which comprises Chain Definition and Query Formulation followed by Fine-grained Reliability Quantification.

3.2.1 Chain Definition and Query Formulation. To search for the required information to autoregressively construct reasoning chains, we follow the query formulation in prior studies [7, 32, 34]. Let $C ^ { i } = \{ u _ { j } ^ { i } \} _ { j = 1 } ^ { | C ^ { i } | }$ denote the set of reasoning chains maintained at hop �, where |C<sup>�</sup> | represents the current number of reasoning chains.

![](images/12c18e24528ec78f65701ca1ec628f4bf64dd763e6c5c758c005d5bd6c1403eb.jpg)  
Figure 2: The overall pipeline of the ReliableRAG framework. ReliableRAG first employs the Triple Extraction module to extract information segments from documents and represent them as structured triples. Subsequently, the Triple Evaluation module performs fine-grained reliability quantification by synthesizing query-triple semantic relevance and triple credibility, ensuring the retention of only the top-� most reliable and non-redundant triples for robust multi-hop reasoning. Leveraging these refined triples, the Chain Construction module autoregressively constructs robust reasoning chains to consolidate trustworthy information and filter out deceptive misinformation, ultimately ensuring the generation of accurate answers.

Notably, the superscript � indicates the current reasoning hop rather than the actual depth of each individual chain. During online inference, the process begins at hop � = 1 with a primary query $q ^ { 1 }$ is first initialized solely from the question � to trigger the construction of the initial reasoning chains $C ^ { 1 }$ . For subsequent hops � ∈ {2, . . . , �}, a set of dynamic queries $\{ q _ { j } ^ { i } \} _ { j = 1 } ^ { | C ^ { i - 1 } | }$ is generated based on the existing chains in $C ^ { i - 1 }$ . Specifically, each query $q _ { j } ^ { i }$ is formed by prepending its corresponding reasoning chain $u _ { j } ^ { i - 1 }$ to the question �, thereby representing the triple information required at current hop � to further address $Q .$

3.2.2 Fine-grained Reliability Quantification. Since the documents contain deceptive misinformation, the extracted triples are also prone to inheriting misleading misinformation. These triples can propagate through the reasoning process, triggering a cascade of errors that corrupts the entire reasoning chain and ultimately leads to incorrect answers [3, 10, 14, 41]. To address this, we propose a dual-factor perception mechanism that formulates a fine-grained reliability score by synthesizing the triple’s semantic relevance to query $q _ { j } ^ { i }$ with triple credibility $( \mathrm { i . e . , }$ , the extent to which the triple is free from misinformation), thereby selecting the top-� most reliable triples from $\mathcal { T }$ for each $q _ { j } ^ { i }$ at each hop. To obtain the fine-grained reliability score, we first measure the semantic relevance of each triple to the query $q _ { j } ^ { i } .$ . Specifically, we employ a bi-encoder (see Appendix F.2 for experiments on diferent bi-encoder) to map $q _ { j } ^ { i }$ and each triple $t _ { k } \in \mathcal { T }$ into a unified feature space. Each triple $t _ { k }$ is represented as a formatted string $\langle s _ { k } ; p _ { k } ; o _ { k } \rangle$ . By encoding both the query and these formatted strings, we obtain the dense vector representations $h _ { i } ^ { i } , h _ { k } \in \mathbb { R } ^ { H }$ , where � denotes the dimensionality of the embedding space. The semantic relevance is then quantified by the cosine similarity between these vectors, reflecting how closely the information within the triple matches the query’s intent:

$$
\Phi ( h _ { j } ^ { i } , h _ { k } ) = \frac { h _ { j } ^ { i } \cdot h _ { k } } { | | h _ { j } ^ { i } | | | | h _ { k } | | }\tag{2}
$$

A higher value of $\Phi ( h _ { j } ^ { i } , h _ { k } )$ indicates a stronger semantic match. However, high semantic relevance alone does not guarantee the reliability of selected triples, as semantically relevant triples may still be factually incorrect. This motivates the need for an additional criterion to assess their credibility. Unlike recent studies that assign credibility scores to entire documents [4, 24], we propose a fine-grained assessment framework that operates at the triple level. Specifically, this framework evaluates each triple $t _ { k } = ( s _ { k } , p _ { k } , o _ { k } )$ based on its basic components: the subject, the predicate, and the object. It employs the LLM-based evaluator $\mathcal { M } _ { \mathrm { e v a } }$ to perform finegrained analyses on the credibility of these components (see $\mathrm { A p \cdot }$ pendix G.3 for experiments on diferent evaluators). Based on this qualitative analysis, the evaluator $\mathcal { M } _ { \mathrm { e v a } }$ subsequently derives an interpretable credibility score, denoted as $\eta _ { k }$ , for each triple, which is assigned as an additional, respective attribute. This process consists of the following steps:

Step 1: Entity Credibility Analysis. The subject $s _ { k }$ and object $o _ { k }$ are first independently analyzed to determine whether they rep resent authentic and established entities rather than spurious or hallucinated concepts. This stage provides the analytical basis regarding the factual existence of components for the subsequent evaluation.

Step 2: Relational Credibility Analysis. Building upon the entity analysis, the relationship described by the predicate $\mathcal { P } k$ between subject $s _ { k }$ and object $o _ { k }$ is analyzed to verify its authentic coherence. For instance, in the triple (Albert Einstein, was the first recipient in 1921 $o f ,$ the Nobel Prize in Physics), the analysis recognizes that while the entities are authentic, the specific relational claim is spurious. This provides the logical rationale for determining the triple’s credibility.

Step 3: Synthesis and Quantitative Scoring. Finally, the qualitative insights from the preceding steps are integrated to generate a finalized credibility score ranging from 0 to 10. This synthesis ensures that the numerical result is grounded in the established analytical steps, reflecting a comprehensive judgment of both entity and relational credibility.

The aforementioned analytical steps are performed by leveraging the ICL ability of the LLM. By employing prompts with few-shot demonstrations, the reasoning behavior of the evaluator $\mathcal { M } _ { \mathrm { e v a } }$ is constrained to align with this triple-level assessment framework’s logic. To enhance scalability, the framework utilizes parallel invocations for large-scale assessment by pre-assigning credibility scores to the triples $\mathcal { T }$ associated with each question $Q$ in an ofline manner. The specific prompt structure and design for this assessment are detailed in $\mathrm { A } { \mathrm { \cdot } }$ ppendix B. Upon deriving the semantic relevance $\Phi ( h _ { j } ^ { i } , h _ { k } )$ between each query $q _ { j } ^ { i }$ and each triple $t _ { k } \in \mathcal { T }$ , alongside the per-triple credibility score $\eta _ { k }$ , the dual-factor perception mechanism synthesizes these factors into the fine-grained reliability score. Formally, this fine-grained reliability score is computed as:

$$
R _ { j , k } ^ { i } = \alpha \cdot \Phi ( h _ { j } ^ { i } , h _ { k } ) + ( 1 - \alpha ) \cdot \eta _ { k }\tag{3}
$$

where � ∈ [0, 1] is a balancing coeficient. As each query $q _ { j } ^ { i }$ evolve at each hop �, the semantic relevance $\Phi ( h _ { j } ^ { i } , h _ { k } )$ is updated and the finegrained reliability score $R _ { j , k } ^ { i }$ is re-computed accordingly. To avoid the redundant selection of information, we mask the fine-grained reliability scores of triples already included in the existing reasoning chains $\dot { C } ^ { i - 1 }$ with an extremely large negative value before ranking. For each query $q _ { j } ^ { i }$ , this penalization ensures that the subsequent selection prioritizes unexplored information. Accordingly, we denote $S _ { j } ^ { i }$ as the set of top-� most reliable triples selected from the set T for each query $q _ { j } ^ { i }$ . Furthermore, these triples are both semantically relevant to $\boldsymbol { q } _ { j } ^ { i }$ and credible, serves as a highly trustworthy search space to facilitate the construction of the reasoning chains $C ^ { i }$ at hop �. Experimental results in Section 4.4.2 demonstrate that the dual-factor perception mechanism efectively filters out deceptive misinformation by identifying reliable triples. This mechanism plays a crucial role in providing efective guidance for constructing reasoning chains, thereby significantly enhancing the ReliableRAG’s overall robustness and final answer accuracy.

## 3.3 Chain Construction

The objective of this module is to autoregressively construct robust reasoning chains by adopting a beam search strategy, where at each hop �, the chain for each query $q _ { j } ^ { i }$ is progressively constructed by utilizing its respective set of top-� most reliable triples $S _ { j } ^ { i }$ . Unlike the prior study [7] that are susceptible to deceptive misinformation during chain construction, our approach uniquely prioritizes and leverages reliable triples to build robust reasoning chains. Specifically, we employ a multiple-option reasoning prompt to guide the LLM-based selector $M _ { \mathrm { s e l } }$ in identifying the top-� most probable triples from the set $S _ { j } ^ { i }$ to construct multiple candidate reasoning chains (experiments on diferent selectors are provided in Appendix G.2). Formally, at each hop �, the reasoning prompt is formulated as follows:

$$
I _ { j } ^ { i } = \Psi ( \mathrm { I n s t r } , \mathrm { E x e m p } , q _ { j } ^ { i } , S _ { j } ^ { i } )\tag{4}
$$

where Instr and Exemp denote the task instruction and the three incontext exemplars most semantically similar to the current question $Q ,$ selected from a manually annotated set, respectively. Following the prior study[7], these exemplars are curated to maintain no overlap with the test datasets. To construct the prompt $I _ { j } ^ { i } ,$ the function Ψ(·) first incorporates a specific task instruction and the selected exemplars into the context to guide the reasoning process. Subsequently, it organizes the termination option and the set of reliable triples $S _ { j } ^ { i }$ into an enumerated list. In this list, the termination option is assigned to option A, indicating that the current reasoning chain $u _ { j } ^ { i - 1 } \in C ^ { i - 1 }$ is suficient to answer the question $Q .$ The triples in $S _ { j } ^ { i }$ are then sequentially mapped to the subsequent option labels, such as B, C, D, and so forth, to form the complete candidate options. Guided by the reasoning prompt $I _ { j } ^ { i } ,$ the LLM-based selector $M _ { \mathrm { s e l } }$ identifies the label corresponding to the termination signal or the next possible triple that best facilitates answering the question $Q .$ By extracting the probability distribution over all option labels from $M _ { \mathrm { s e l } }$ , we select top-� highest-probability options to make multiple next-hop decisions. This selection is formulated as follows:

$$
\mathcal { E } _ { j } ^ { i } = \mathrm { T o p } { - } B _ { e _ { k } } \Big ( P _ { M _ { \mathrm { s e l } } } ( e _ { k } \mid I _ { j } ^ { i } ) \Big )\tag{5}
$$

where $\mathcal { E } _ { j } ^ { i }$ denotes the set of the � highest-probability options selected according to the probability distribution. Subsequently, for each selected option $\boldsymbol { e } _ { k } \in \mathcal { E } _ { j } ^ { i }$ , the previous reasoning chain $u _ { j } ^ { i - 1 }$ undergoes branching to form a candidate reasoning chain for current hop $i ,$ contingent on the type of the selected option. Specifically, if $e _ { k }$ represents the termination signal, the chain $u _ { j } ^ { i - 1 }$ is designated as a candidate reasoning chain for current hop � without further expansion. Otherwise, it is expanded by appending the triple $t _ { k } \in S _ { j } ^ { i }$ corresponding to the chosen option label. This branching process uses the beam search strategy to allow for the parallel exploration of multiple promising candidate chains, and is formally defined as:

$$
u _ { m } ^ { i } = \left\{ \begin{array} { l l } { u _ { j } ^ { i - 1 } , } & { e _ { k } \mathrm { ~ i s ~ o p t i o n ~ A ~ o r ~ } u _ { j } ^ { i - 1 } \mathrm { ~ i s ~ a l r e a d y ~ t e r m i n a t e d } , } \\ { u _ { j } ^ { i - 1 } \oplus t _ { k } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{6}
$$

where $u _ { m } ^ { i }$ denotes a candidate reasoning chain generated at hop � representing a branch derived from the previous chain $u _ { j } ^ { i - 1 }$ and the selected option $\boldsymbol { e } _ { k } \in \mathcal { E } _ { j } ^ { i } .$ . The operator ⊕ denotes the concatenation of the new triple $t _ { k }$ to the existing sequence $u _ { j } ^ { i - 1 }$ . Let $\rho _ { j , k } ^ { i }$ denote the selection probability of $e _ { k }$ for $u _ { j } ^ { i - 1 }$ , and the confidence of this branch is then measured by its cumulative probability, computed as follows:

$$
\omega _ { m } ^ { i } = \omega _ { j } ^ { i - 1 } \cdot \rho _ { j , k } ^ { i }\tag{7}
$$

where $\omega _ { j } ^ { i - 1 }$ is the confidence of the previous chain $u _ { j } ^ { i - 1 }$ . Starting from the set of reasoning chains $C ^ { i - 1 }$ , all candidate reasoning chains collected for the current hop � are ranked by their confidence, with only the top-� most confident candidate chains retained to form $C ^ { i }$ at hop �. This autoregressive construction process continues until the maximum hop $i = L$ is reached or all reasoning chains in the set have reached the termination signal. At the final hop $L ,$ which defines the maximum number of triples allowed in each chain, this module constructs the comprehensive and robust set of reasoning chains $C _ { \mathrm { f i n a l } }$ . Consequently, the sequence of triples within each reasoning chain in $C _ { \mathrm { f i n a l } }$ provides diverse and reliable evidence to derive the answer to the question $Q .$ . Ablation studies in Section 4.3 demonstrate that this module is indispensable to the overall performance of ReliableRAG. The complete procedure of the Reliability-Guided Chain Construction is formally detailed in Algorithm 1.

## 4 Experiments

To comprehensively evaluate ReliableRAG, we first conduct extensive experiments on three multi-hop QA datasets. Next, we conduct ablation studies to validate the efectiveness of each individual module. Finally, we perform sensitivity analyses to investigate the impact of key hyperparameters. Detailed experimental setup, the two synthesis strategies (TBS and SBS), and an in-depth case study are presented in Appendices A to C, respectively. Additional results including experiments under the evaluator-generated setting, additional analysis (including the ablation study for ReliableRAG-SBS), experiments on diferent extractor, evaluator, and selector, and analysis of evaluator capability are available in Appendices E to H.

Table 1: Main results between ReliableRAG and compared methods under both the ideal setting and the evaluator-generated setting. The best performance within each setting is highlighted in bold. ‘w/ Low-credibility Docs’ denotes the injection of one low-credibility document per question. The dash ‘-’ denotes that the method is either not applicable to the setting or identical in practice to another setting already reported.
<table><tr><td rowspan="3">Methods</td><td colspan="6">Ideal Setting</td><td colspan="6">Evaluator-Generated Setting</td></tr><tr><td colspan="6">HotPotQA 2WikiMultiHopQA</td><td colspan="2">HotPotQA</td><td colspan="3">2WikiMultiHopQA MuSiQue</td></tr><tr><td colspan="2">EM (%) F1 (%) EM (%)</td><td>F1 (%)</td><td></td><td colspan="2">MuSiQue EM (%) F1 (%)</td><td>EM (%) F1 (%)</td><td colspan="2">EM (%) F1 (%)</td><td>EM (%)</td><td colspan="2">F1 (%)</td></tr><tr><td>Fine-tuning Methods</td><td></td><td colspan="2">w/ Low-credibility Docs</td><td></td><td></td><td></td><td></td><td></td><td>w/ Low-credibility Docs</td><td></td><td></td><td></td></tr><tr><td>CAG-7B (EMNLP&#x27;24)</td><td>5.80</td><td>21.53</td><td>3.80</td><td>18.20</td><td>1.20</td><td>9.45</td><td>4.70</td><td>17.91</td><td>3.70</td><td>16.58</td><td>0.7</td><td>7.71</td></tr><tr><td>Self-RAG-7B (ICLR&#x27;24)</td><td>18.30</td><td>28.00</td><td>5.80</td><td>10.50</td><td>6.60</td><td>15.72</td><td></td><td></td><td></td><td></td><td>-</td><td></td></tr><tr><td>Knowledge-R1-7B (arxiv&#x27;25)</td><td>26.80</td><td>35.45</td><td>18.00</td><td>22.81</td><td>11.20</td><td>20.76</td><td>-</td><td>-</td><td></td><td></td><td></td><td>一</td></tr><tr><td colspan="2">Non-fine-tuning Methods</td><td colspan="6"></td><td colspan="6"></td></tr><tr><td colspan="2"></td><td colspan="6">w/o Low-credibility Docs</td><td colspan="6">w/o Low-credibility Docs</td></tr><tr><td>Naive LLM</td><td>16.50</td><td>24.42</td><td>20.10</td><td>23.96</td><td>2.10</td><td>6.50</td><td></td><td>一</td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla RAG Gem-B</td><td>43.90</td><td>55.47</td><td>30.30</td><td>36.93</td><td>17.40</td><td>25.61</td><td>-</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2"></td><td colspan="6">w/ Low-credibility Docs</td><td colspan="6">w/ Low-credibility Docs</td></tr><tr><td>Vanilla RAG</td><td>20.10</td><td>28.39</td><td>14.50</td><td>19.01</td><td></td><td>10.40</td><td>20.08</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Prompt Based</td><td>43.10</td><td>54.42</td><td>32.90</td><td>39.03</td><td>18.10</td><td>26.52</td><td>33.90</td><td>43.65</td><td>27.00</td><td>32.80</td><td>14.00</td><td>21.16</td></tr><tr><td>Exclusion</td><td></td><td></td><td></td><td></td><td></td><td></td><td>34.10</td><td>43.80</td><td>27.00</td><td>32.80</td><td>14.00</td><td>21.16</td></tr><tr><td>CrAM (AAAI&#x27;25)</td><td>42.70</td><td>54.30</td><td>27.30</td><td>32.64</td><td>20.40</td><td>28.71</td><td>32.00</td><td>40.31</td><td>17.80</td><td>22.31</td><td>14.00</td><td>23.76</td></tr><tr><td>TruthfulRAG (AAAI&#x27;26)</td><td>36.70</td><td>48.48</td><td>25.70</td><td>31.85</td><td>19.50</td><td>29.07</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReliableRAG-TBS (ours)</td><td>52.20</td><td>66.89</td><td>45.80</td><td>55.88</td><td>30.30</td><td>38.66</td><td>39.40</td><td>52.96</td><td>32.10</td><td>39.30</td><td>19.40</td><td>27.21</td></tr><tr><td>ReliableRAG-SBS (ours)</td><td>53.20</td><td>66.98</td><td>43.10</td><td>53.90</td><td>31.60</td><td>40.51</td><td>31.50</td><td>42.27</td><td>24.80</td><td>31.77</td><td>17.00</td><td>25.56</td></tr><tr><td>w/o Low-credibility Docs</td><td colspan="6"></td><td colspan="6">w/o Low-credibility Docs</td></tr><tr><td></td><td>Naive LLM Vanilla RAG</td><td>18.90</td><td>27.03 16.10</td><td></td><td>23.16</td><td>2.70 18.90</td><td>6.94 26.71</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>49.90</td><td>64.38</td><td>23.50</td><td>43.33</td><td></td><td></td><td>-</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="6">w/ Low-credibility Docs</td><td colspan="6">w/ Low-credibility Docs</td></tr><tr><td>Vanilla RAG Prompt Based</td><td></td><td>19.30</td><td>28.12 12.50</td><td>19.93</td><td>3.30</td><td>12.82</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>48.40</td><td>62.98</td><td>30.70</td><td>45.06</td><td>20.10</td><td>27.47</td><td>38.50</td><td>51.30</td><td>23.40</td><td>35.36</td><td>15.30</td><td>22.00</td></tr><tr><td>Exclusion</td><td></td><td></td><td></td><td>一</td><td></td><td></td><td>38.60</td><td>51.40</td><td>23.40</td><td>35.36</td><td>15.30</td><td>22.00</td></tr><tr><td>CrAM (AAAI&#x27;25)</td><td>50.80</td><td>64.43</td><td>20.00</td><td>39.64</td><td>19.80</td><td>27.48</td><td>33.40</td><td>44.21</td><td>19.20</td><td>33.81</td><td>8.30</td><td>15.93</td></tr><tr><td>TruthfulRAG (AAAI&#x27;26)</td><td>38.10</td><td>49.61</td><td>28.00</td><td>35.39</td><td>16.90</td><td>25.48</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ReliableRAG-TBS (ours)</td><td>53.40</td><td>67.02</td><td>45.40</td><td>56.19</td><td>29.10</td><td>36.73</td><td>40.30</td><td>52.45</td><td>29.90</td><td>38.45</td><td>19.50</td><td>25.43</td></tr><tr><td>ReliableRAG-SBS (ours)</td><td>55.10</td><td>68.80</td><td>43.70</td><td>59.67</td><td>30.90</td><td>38.55</td><td>32.30</td><td>42.93</td><td>23.60</td><td>34.00</td><td>17.60</td><td>24.84</td></tr><tr><td></td><td colspan="6">w/o Low-credibility Docs</td><td colspan="6">w/o Low-credibility Docs</td></tr><tr><td>Vanilla RAG</td><td>Naive LLM</td><td>19.60</td><td>27.22 22.50</td><td></td><td>26.41</td><td>3.40</td><td>8.02</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Vanilla RAG</td><td>36.30</td><td>49.41</td><td>29.10</td><td></td><td>36.19</td><td>15.00</td><td>21.89</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>18.50 26.05 11.20</td><td colspan="6">w/ Low-credibility Docs</td><td colspan="6">w/ Low-credibility Docs</td></tr></table>

## 4.1 Experimental Setups

4.1.1 Datasets. Experiments are conducted on three multi-hop QA datasets: HotPotQA [42], 2WikiMultiHopQA [9], and MuSiQue [33]. These datasets typically require 2–4 reasoning hops, with each question associated with 10, 10, and 20 Wikipedia-sourced documents, respectively. Following the preprocessing by Fang et al. [7], we filter out sensitive content. To balance computational eficiency with statistical significance, we randomly sample 1,000 questions from each test set for evaluation, while an additional 100 questions from each development set are utilized for hyperparameter tuning. Misinformation scenarios are simulated by injecting low-credibility documents (each containing LLM-generated deceptive misinformation) for each question, with performance evaluated under both Ideal and Evaluator-generated settings.

4.1.2 Compared Methods and Evaluation Metrics. ReliableRAG is compared against a diverse set of compared methods, including Naive LLM, Vanilla RAG, Prompt-based, Exclusion, Self-RAG [2], CAG [24], Knowledgeable-R1 [17], CrAM [4], TruthfulRAG [21]. To comprehensively evaluate the performance of all methods, we adopt two standard QA metrics: Exact Match (EM) [1, 13] and F1 score (F1) [19, 39].

## 4.2 Main Results

4.2.1 Performance Comparison. Table 1 presents the comparison between ReliableRAG and nine compared methods, covering both fine-tuning and non-fine-tuning methods. The results clearly demonstrate that ReliableRAG significantly outperforms all compared methods under the Ideal setting. Across the three tested LLMs, it delivers the highest EM performance on all test sets, achieving 55.10%, 49.50%, and 31.60% on HotPotQA, 2WikiMultiHopQA, and MuSiQue, respectively. Similarly, ReliableRAG attains the highest

Algorithm 1 Reliability-Guided Chain Construction   
Input: question $Q ,$ triples with injected misinformation   
$\mathcal { T } _ { \mathrm { m i s } } .$ , maximum number of reasoning hops $L ,$ number of reasoning   
chains to be retained $F ,$ beam size $B ,$ number of selected triples $K$   
output: Reasoning Chains $C _ { \mathrm { f i n a l } }$   
1: initialize reasoning chains $C ^ { 0 } \gets \emptyset$ and confidence score for   
each chain $\Omega ^ { 0 } \gets \check { \{ 1 . 0 \} }$   
2: for $i = 1$ to $L$ do   
3: Generate queries $\{ q _ { j } ^ { i } \} _ { j = 1 } ^ { | C ^ { i - 1 } | }$ from $Q$ and $C ^ { i - 1 } \# | C ^ { i - 1 } | \leq F$   
4: Obtain embeddings $\{ h _ { j } ^ { i } \} _ { j = 1 } ^ { | C ^ { i - 1 } | }$ and $\{ h _ { k } \} _ { k = 1 } ^ { M }$   
5: Compute similarity $\Phi ( \dot { h } _ { j } ^ { i } , \dot { h } _ { k } )$ for $j \in [ 1 , | C ^ { i - 1 } | \} ] , k \in [ 1 , M ]$   
6: Obtain credibility score $\eta _ { k }$ for each triple   
7: Compute reliability score $R _ { j , k } ^ { i }  \alpha \cdot \Phi ( h _ { j } ^ { i } , h _ { k } ) + ( 1 - \alpha ) \cdot \eta _ { k }$   
8: Mask reliability score $R _ { j , k } ^ { i }  - \infty , \forall t _ { k } \in u _ { j } ^ { i - 1 } , u _ { j } ^ { i - 1 } \in C ^ { i - 1 }$   
9: # Ensuring non-redundant triple selection   
10: Select top- $- K$ most reliable triples $S _ { j } ^ { i }$ by reliability   
11: Construct reasoning prompt $I _ { j } ^ { i } \gets \dot { \Psi }$ (Instr, Exemp, $q _ { j } ^ { i } , S _ { j } ^ { i } )$   
12: Identify top-� likely options $\bar { \mathcal { E } _ { j } ^ { i } } \gets \mathrm { T o p } { - } B _ { e _ { k } } ( P _ { M _ { \mathrm { s e l } } } ( e _ { k } \mid \bar { I _ { j } ^ { i } } ) )$   
13: for each option $e _ { k }$ in $\mathcal { E } _ { j } ^ { i }$ do   
14: if $e _ { k }$ is option A or $\stackrel { \cdot } { u } _ { j } ^ { i - 1 }$ is already terminated then   
15: Retain the current chain $u _ { m } ^ { i } \gets u _ { j } ^ { i - 1 }$   
16: else   
17: Extend the chain: $u _ { m } ^ { i } \gets u _ { j } ^ { i - 1 } \oplus t _ { k }$   
18: Update confidence score $\omega _ { m } ^ { i } = \omega _ { j } ^ { i - 1 } \cdot \rho _ { j , k } ^ { i }$   
19: # Accumulate confidence with selection probability   
20: Retain the top- ${ } _ { - F }$ candidate chains $C ^ { i }$ by confidence   
21: return $C _ { \mathrm { f i n a l } }$

EM performance under the evaluator-generated setting across all tested LLMs, yielding 40.30%, 33.80%, and 19.50% on HotPotQA, 2WikiMultiHopQA, and MuSiQue, respectively. These findings underscore the superior robustness of ReliableRAG in resisting the interference of deceptive misinformation, preventing its propagation during multi-hop reasoning, and maintaining stability during complex multi-hop QA tasks. Notably, the CAG method exhibits significant instruction-following dificulties and tends to generate excessive explanatory content, which leads to abysmal EM performance.

4.2.2 Robustness to Misinformation. To further validate robustness, we vary the number of low-credibility documents per question on all test sets, considering both ideal and evaluator-generated settings using Mistral-7B as the generator. As shown in Figure 3, ReliableRAG outperforms six compared methods with minimal performance drops. Results in evaluator-generated settings exhibit a similar pattern (see Appendix E.2 for detailed results). Interestingly, we observe that the EM performance of ReliableRAG-SBS on HotpotQA even improves slightly as the misinformation proportion rises. We hypothesize that our framework can uncover fine-grained correct information hidden even within low-credibility documents, subsequently transforming these fine-grained informational into structured triples that are efectively utilized to derive the final answer. We attribute this capability to the dual-factor perception mechanism within the Triple Evaluation module, which efectively captures these correct triples and integrates them into robust reasoning chains. This allows the framework to leverage valid fine-grained information while resisting deceptive misinformation, thereby enhancing the framework’s overall robustness. Furthermore, our framework ultimately yields a concise input context X for final generation, the input computational overhead of which is detailed in Appendix F.4 through experimental results.

```csv
0.55 0.70
0.50 0.65
0.45 0.60
0.40 0.55
0.35 0.50
<sub>E</sub>M 0.30 <sub>F</sub><sup>1</sup> 0.45
0.25 0.35
0.20 0.30
0.15 0.25
0.10 0.20
0.05 0.15
1 2 3 1 2 3
Misinformation Document Count Misinformation Document Count
(a) EM on HotPotQA (b) F1 on HotPotQA
0.50 0.60
0.45 0.55
0.40 0.50
0.35 0.45
0.40
0.30
M 0.25 <sub>F</sub><sup>1</sup> 0.35
0.20 0.25
0.15 0.20
0.10 0.15
0.05 0.10
0.00 0.05
1 2 3 1 2 3
Misinformation Document Count Misinformation Document Count
(c) EM on 2WikiMultiHopQA (d) F1 on 2WikiMultiHopQA
0.35 0.40
0.30 0.35
0.25 0.30
<sub>E</sub>M 0.20 <sub>F</sub><sup>1</sup> 0.25
0.15 0.20
0.10 0.15
0.05 0.10
0.00 0.05
1 2 3 1 2 3
Misinformation Document Count Misinformation Document Count
(e) EM on MuSiQue (f) F1 on MuSiQue
Prompt Based CrAM
CAG-7B TruthfulRAG
Self-RAG-7B ResilientRAG-TBS
Knowledge-R1-7B ResilientRAG-SBS
```  
Figure 3: Performance comparison between ReliableRAG and six compared methods on the test sets of three multi-hop QA datasets with varying numbers of low-credibility documents per question, evaluated under the ideal setting.

## 4.3 Ablation Studies

To evaluate the contribution of each component within ReliableRAG-TBS, we conduct a systematic ablation study. For the ‘w/o Chain Construction’ variant, instead of generating a reasoning chain, we employ E5-Mistral to retrieve the top-� most reliable triples directly from the set of triples $\mathcal { T }$ for each query. These triplets are then provided as the supporting context for the generator $M _ { \mathrm { { g e n } } }$ to produce the final answer. To ensure a fair comparison, we evaluate this variant across various values of � and report the corresponding results.

Results in Table 2 show that the complete ReliableRAG-TBS achieves best performance across all test sets (Detailed analysis for ReliableRAG-SBS is provided in Appendix F.3). Notably, removing Triple Extraction triggers the most substantial performance decline, represented by a 34.1% drop in EM on the HotpotQA test set, underscoring the triples’ pivotal role in organizing information for multi-hop QA. Similarly, excluding Triple Evaluation leads to a 15.6% EM decrease, highlighting its importance in preventing misinformation propagation during multi-hop reasoning. Furthermore, we observe a distinct performance threshold in the ablation variant without Chain Construction. Specifically, the EM performance of ReliableRAG-TBS on HotPotQA and MuSiQue begins to decline once the number of triples exceeds a certain threshold, typically between 15 and 25. This suggests that blindly increasing the number of triples does not yield further performance gains. This demonstrates the importance of Chain Construction in utilizing these triples to construct reasoning chains. Notably, while the removal of Chain Construction leads to a performance drop, the margin ofdecline remains relatively moderate compared to the preceding two modules. This indicates that the Triple Extraction and Triple Evaluation modules establish a search space comprising highly reliable triples, which is pivotal in preventing the propagation of misinformation during multi-hop reasoning. Building upon this, the Chain Construction module constructs robust multi-hop reasoning chains from the highly reliable triples within the search space. These chains aggregate as many logically coherent and reliable triples as possible, ultimately synthesizing them into a comprehensive and reliable supporting context, thereby further fortifying the system’s overall robustness and factual reliability, and ensuring final answer precision.

Table 2: Ablation results of ReliableRAG-TBS on the test sets of three multi-hop QA datasets. The Top-� Triple variants represent the performance where the supporting context Z is constructed by applying the TBS synthesis strategy to the top-� most reliable triples without the Chain Construction module.
<table><tr><td rowspan="2">Methods</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%) F1 (%)</td><td></td><td>EM (%)</td><td>F1 (%)</td><td>EM (%) F1 (%)</td><td></td></tr><tr><td>w/o Triple Extraction</td><td>19.30</td><td>28.12</td><td>12.50</td><td>19.93</td><td>3.30</td><td>12.82</td></tr><tr><td>w/o Triple Evaluation</td><td>37.80</td><td>49.42</td><td>30.10</td><td>37.16</td><td>17.20</td><td>25.81</td></tr><tr><td>w/o Chain Construction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Top-10 Triples</td><td>46.40</td><td>59.84</td><td>26.10</td><td>37.53</td><td>21.30</td><td>28.61</td></tr><tr><td>Top-15 Triples</td><td>46.40</td><td>60.01</td><td>29.70</td><td>42.39</td><td>23.90</td><td>31.96</td></tr><tr><td>Top-20 Triples</td><td>47.50</td><td>61.37</td><td>30.80</td><td>45.30</td><td>22.80</td><td>31.17</td></tr><tr><td>Top-25 Triples</td><td>47.00</td><td>60.98</td><td>32.20</td><td>48.38</td><td>23.10</td><td>31.67</td></tr><tr><td>ReliableRAG-TBS</td><td>53.40</td><td>67.02</td><td>45.40</td><td>56.19</td><td>29.10</td><td>36.73</td></tr></table>

## 4.4 Sensitivity analyses

4.4.1 Trade-ofin Triple Evaluation. We systematically evaluate the influence of the balancing coeficient � that regulates the dual-factor perception mechanism within the Triple Evaluation module. Using Llama3-8B as the generator, we perform a grid search for � ∈ [0, 1.0] across the development sets of three multi-hop QA datasets, each injected with one low-credibility document per question. As illustrated in Figure 4, ReliableRAG achieves its most balanced performance at $\alpha ^ { * } = 0 . 4$ , striking an optimal trade-of between query-triple semantic relevance and triple credibility. Specifically, while a larger � prioritizes semantic relevance, it renders the framework more susceptible to deceptive misinformation. Conversely, a smaller � more efectively suppresses misinformation but may overly restrict information utilization, thereby compromising the overall generation accuracy.

4.4.2 Efectiveness in Preventing Misinformation. Following the prior study [4] that quantifies the contribution of specific components by measuring the change in generation probability, we adopt the Indirect Efect (IE) to validate the eficacy of the dual-factor perception mechanism within the Triple Evaluation module and its role in guiding the construction of reasoning chains. The formal definition of IE is detailed in Appendix D. Specifically, across the development sets of three multi-hop QA datasets (each injected with one low-credibility document per question), we analyze the distribution of the IE for all ReliableRAG variants (including both TBS and SBS) at the optimal balancing coeficient $\alpha ^ { * } = 0 . 4$ . As illustrated in Figure 5, the distribution approximates a normal shape centered near zero, with a significant density shift toward the positive side. Notably, positive efects account for 70.5% of the cases, indicating that our balancing coeficient adjustment generally enhances reasoning correctness. These findings confirm that the dual-factor perception mechanism not only fortifies ReliableRAG against misinformation but also provides efective guidance for constructing reasoning chains, thereby bolstering the overall robustness and factual reliability of the RAG system.

![](images/8b6c4617912211ba0b7bd071f922037885002c84e6fce887f5866892233d8779.jpg)  
(a) ReliableRAG-TBS

![](images/b53dd17f6687bd393ccd67a59f997174baa0122a34fd16104bffe0d2e9ee207c.jpg)  
Figure 4: EM and F1 performance of ReliableRAG across various configurations of the balancing coeficient � that regulates the dual-factor perception mechanism within the Triple Evaluation module. Results compare (a) ReliableRAG-TBS and (b) ReliableRAG-SBS variants, evaluated under the ideal setting using the dev sets of three multi-hop QA datasets.

![](images/d668fd0af3dd4b0fe700c17582406927092d4bf4f0da55b4e47b05976c602211.jpg)  
Figure 5: IE density distribution under the optimal configuration $\alpha ^ { * } = 0 . 4 .$

4.4.3 Impact ofReasoning Depth. To investigate how the maximum allowable number of reasoning steps (i.e., the maximum reasoning chain length �) afects performance, we perform a sensitivity analysis ofReliableRAG by varying � from 1 to 6. As illustrated in Figure 6, across two dev sets, the average chain length exhibits sub-linear growth as � increases, eventually plateauing rather than increasing indefinitely. We observe that although the average chain length continues to grow albeit at a much slower rate as � increases from 4 to 6, this additional depth does not translate into detectable EM gains. Notably, when $L > 4 ,$ the EM performance remains virtually stagnant or even experiences a slight decline (additional results on the MuSiQue dataset are detailed in Appendix F.1). This suggests that excessively increasing the reasoning depth does not yield further gains. Instead, it may introduce deceptive misinformation that contaminates the reasoning process, ultimately undermining the system’s robustness.

![](images/d9955fea22fdbe2625e2ef178eae289012810431f672bf834e85c8a2a3b166dc.jpg)  
(a) HotPotQA

![](images/e426283de87abfd160279fed1e0180d3a2dff65ef996769ae54527340b533237.jpg)  
Figure 6: Impact of the maximum reasoning chain length � on the QA performance and average chain length of ReliableRAG-TBS on the dev sets of two multi-hop QA datasets, under an ideal setting where one low-credibility document is injected per question.

## 5 Conclusion

In this paper, we introduce ReliableRAG, a novel framework designed to combat deceptive misinformation in multi-hop QA. By dynamically evaluating fine-grained reliability at the triple level, our approach filters deceptive misinformation to construct reliable, autoregressive reasoning chains. Extensive experiments demonstrate that ReliableRAG significantly outperforms existing methods, providing RAG systems with a highly robust solution to ensure factual reliability and prevent misinformation propagation during multi-hop reasoning processes.

## Acknowledgments

To Robert, for the bagels and explaining CMYK and color spaces.

## References

[1] Ashkan Alinejad, Krtin Kumar, and Ali Vahdat. 2024. Evaluating the Retrieval Component in LLM-Based Question Answering Systems. arXiv:2406.06458 [cs.CL] https://arxiv.org/abs/2406.06458

[2] Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. 2024. Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection. In The Twelfth International Conference on Learning Representations. https:// openreview.net/forum?id=hSyW5go0v8

[3] Neeladri Bhuiya, Viktor Schlegel, and Stefan Winkler. 2024. Seemingly Plausible Distractors in Multi-Hop Reasoning: Are Large Language Models Attentive Readers?. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 2514–2528. doi:10.18653/v1/2024.emnlp-main.147

[4] Boyi Deng, Wenjie Wang, Fengbin Zhu, Qifan Wang, and Fuli Feng. 2025. CrAM: Credibility-Aware Attention Modification in LLMs for Combating Misinformation in RAG. Proceedings of the AAAI Conference on Artificial Intelligence 39, 22 (Apr. 2025), 23760–23768. doi:10.1609/aaai.v39i22.34547

[5] Peter Devine. 2025. ALoFTRAG: Automatic Local Fine Tuning for Retrieval Augmented Generation. arXiv:2501.11929 [cs.LG] https://arxiv.org/abs/2501. 11929

[6] Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Jingyuan Ma, Rui Li, Heming Xia, Jingjing Xu, Zhiyong Wu, Baobao Chang, Xu Sun, Lei Li, and Zhifang Sui. 2024. A Survey on In-context Learning. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 1107–1128. doi:10.18653/v1/2024.emnlp-main.64

[7] Jinyuan Fang, Zaiqiao Meng, and Craig MacDonald. 2024. TRACE the Evidence: Constructing Knowledge-Grounded Reasoning Chains for Retrieval-Augmented Generation. In Findings ofthe Association for Computational Linguistics: EMNLP 2024, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 8472–8494. doi:10.18653/v1 2024.findings-emnlp.496

[8] Cristina Garbacea, Samuel Carton, Shiyan Yan, and Qiaozhu Mei. 2019. Judge the Judges: A Large-Scale Evaluation Study of Neural Language Models for Online Review Generation. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 3968–3981. doi:10.18653/v1/D19-1409

[9] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings ofthe 28th International Conference on Computational Linguistics, Donia Scott, Nuria Bel, and Chengqing Zong (Eds.). International Committee on Computational Linguistics, Barcelona, Spain (Online), 6609–6625. doi:10.18653/v1/2020.coling-main.580

[10] Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A Survey on Hallucination in Large Language Models: Principles, Taxonomy, Challenges, and Open Questions. ACM Trans. Inf. Syst. 43, 2, Article 42 (Jan. 2025), 55 pages. doi:10.1145/3703155

[11] Rohit Kumar Kaliyar, Anurag Goswami, and Pratik Narang. 2021. FakeBERT: Fake news detection in social media with a BERT-based deep learning approach. 80, 8 (2021). doi:10.1007/s11042-020-10183-2

[12] Rohit Kumar Kaliyar and Navya Singh. 2019. Misinformation Detection on Online Social Media-A Survey. In 2019 10th International Conference on Computing, Communication and Networking Technologies (ICCCNT). 1–6. doi:10.1109/ ICCCNT45670.2019.8944587

[13] Ehsan Kamalloo, Nouha Dziri, Charles Clarke, and Davood Rafiei. 2023. Evaluating Open-Domain Question Answering in the Era of Large Language Models. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 5591–5606. doi:10.18653/v1/2023.acl-long.307

[14] Ioannis Kazlaris, Efstathios Antoniou, Konstantinos Diamantaras, and Charalampos Bratsas. 2025. From Illusion to Insight: A Taxonomic Survey of Hallucination Mitigation Techniques in LLMs. AI 6, 10 (2025). doi:10.3390/ai6100260

[15] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (Eds.), Vol. 33. Curran Associates, Inc., 9459–9474. https://proceedings.neurips.cc/paper\_ files/paper/2020/file/6b493230205f780e1bc26945df7481e5-Paper.pdf

[16] Yanyang Li, Shuo Liang, Michael Lyu, and Liwei Wang. 2024. Making Long-Context Language Models Better Multi-Hop Reasoners. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 2462–2475. doi:10.18653/v1/2024. acl-long.135

[17] Chenyu Lin, Yilin Wen, Du Su, Hexiang Tan, Fei Sun, Muhan Chen, Chenfu Bao, and Zhonghou Lyu. 2026. Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement. arXiv:2506.05154 [cs.CL] https://arxiv. org/abs/2506.05154

[18] Sheng-Chieh Lin, Akari Asai, Minghan Li, Barlas Oguz, Jimmy Lin, Yashar Mehdad, Wen-tau Yih, and Xilun Chen. 2023. How to Train Your Dragon: Diverse Augmentation Towards Generalizable Dense Retrieval. In Findings ofthe Association

for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 6385–6400. doi:10.18653/v1/2023.findings-emnlp.423

[19] Xueling Lin and Lei Chen. 2018. Domain-aware multi-truth discovery from conflicting sources. Proc. VLDB Endow. 11, 5 (Oct. 2018), 635–647. doi:10.1145 3177732.3177739

[20] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. Transactions ofthe Association for Computational Linguistics 12 (2024), 157–173. doi:10.1162/tacl\_a\_00638

[21] Shuyi Liu, Yu-Ming Shang, and Xi Zhang. 2026. TruthfulRAG: Resolving Factual level Conflicts in Retrieval-Augmented Generation with Knowledge Graphs. Proceedings ofthe AAAI Conference on Artificial Intelligence 40, 38 (Mar. 2026), 32168– 32176. doi:10.1609/aaai.v40i38.40489

[22] Tianci Liu, Haoxiang Jiang, Tianze Wang, Ran Xu, Yue Yu, Linjun Zhang, Tuo Zhao, and Haoyu Wang. 2025. RoseRAG: Robust Retrieval-augmented Generation with Small-scale LLMs via Margin-aware Preference Optimization. In Findings ofthe Association for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 13036–13054. doi:10.18653/v1/ 2025.findings-acl.676

[23] Liangming Pan, Wenhu Chen, Min-Yen Kan, and William Yang Wang. 2023. Attacking Open-domain Question Answering by Injecting Misinformation. In Proceedings ofthe 13th International Joint Conference on Natural Language Processing and the 3rd Conference ofthe Asia-Pacific Chapter ofthe Association for Computational Linguistics (Volume 1: Long Papers), Jong C. Park, Yuki Arase, Baotian Hu, Wei Lu, Derry Wijaya, Ayu Purwarianti, and Adila Alfa Kris nadhi (Eds.). Association for Computational Linguistics, Nusa Dua, Bali, 525–539. doi:10.18653/v1/2023.ijcnlp-main.35

[24] Ruotong Pan, Boxi Cao, Hongyu Lin, Xianpei Han,Jia Zheng, Sirui Wang, Xunliang Cai, and Le Sun. 2024. Not All Contexts Are Equal: Teaching LLMs Credibility aware Generation. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 19844–19863. doi:10.18653/v1/2024.emnlp-main.1109

[25] Yikang Pan, Liangming Pan, Wenhu Chen, Preslav Nakov, Min-Yen Kan, and William Wang. 2023. On the Risk of Misinformation Pollution with Large Language Models. In Findings ofthe Association for Computational Linguistics: EMNLP 2023, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 1389–1403. doi:10.18653/v1/2023.findings-emnlp.97

[26] Kellin Pelrine, Anne Imouza, Camille Thibault, Meilina Reksoprodjo, Caleb Gupta, Joel Christoph, Jean-François Godbout, and Reihaneh Rabbany. 2023. Towards Reliable Misinformation Mitigation: Generalization, Uncertainty, and GPT-4. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 6399–6429. doi:10.18653/v1/2023.emnlpmain.395

[27] Dorian Quelle and Alexandre Bovet. 2024. The Perils and Promises of Fact-Checking with Large Language Models. Frontiers in Artificial Intelligence Volume 7 - 2024 (2024). doi:10.3389/frai.2024.1341697

[28] Gowtham Ramesh, Makesh Narsimhan Sreedhar, and Junjie Hu. 2023. Single Sequence Prediction over Reasoning Graphs for Multi-hop QA. In Proceedings of the 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational Linguistics, Toronto, Canada, 11466–11481. doi:10. 18653/v1/2023.acl-long.642

[29] Stephanie Schoch and Yangfeng Ji. 2025. The Good, the Bad, and the Debatable: A Survey on the Impacts of Data for In-Context Learning. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (Eds.). Association for Computational Linguistics, Suzhou, China, 29798–29812. doi:10. 18653/v1/2025.emnlp-main.1514

[30] Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Enting Chen, Damien Graux, Andre Melo, Ruofei Lai, Zeren Jiang, Zhongyang Li, Ye Qi, Yang Ren, Dandan Tu, and Jef Z. Pan. 2025. GeAR: Graphenhanced Agent for Retrieval-augmented Generation. In Findings ofthe Association for Computational Linguistics: ACL 2025, Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 12049–12072. doi:10.18653/v1/2025.findings-acl.624

[31] Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, and Claire Cardie. 2025. Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks. arXiv:2412.16708 [cs.IR] https://arxiv.org/ abs/2412.16708

[32] Haitian Sun, Tania Bedrax-Weiss, and William Cohen. 2019. PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (Eds.). Association for Computational Linguistics, Hong Kong, China, 2380–2390. doi:10.18653/v1/D19-1242

[33] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539–554. doi:10.1162/tacl\_a\_00475

[34] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Proceedings ofthe 61st Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (Eds.). Association for Computational

Linguistics, Toronto, Canada, 10014–10037. doi:10.18653/v1/2023.acl-long.557

[35] Vaibhav, Raghuram Mandyam Annasamy, and Eduard Hovy. 2019. Do Sentence Interactions Matter? Leveraging Sentence Level Representations for Fake News Classification. In Proceedings ofthe Thirteenth Workshop on Graph-Based Methods for Natural Language Processing (TextGraphs-13), Dmitry Ustalov, Swapna Somasundaran, Peter Jansen, Goran Glavaš, Martin Riedl, Mihai Surdeanu, and Michalis Vazirgiannis (Eds.). Association for Computational Linguistics, Hong Kong, 134–139. doi:10.18653/v1/D19-5316

[36] Fei Wang, Xingchen Wan, Ruoxi Sun, Jiefeng Chen, and Sercan O Arik. 2025. Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models. In Proceedings ofthe 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 30553–30571. doi:10. 18653/v1/2025.acl-long.1476

[37] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2024. Text Embeddings by Weakly-Supervised Contrastive Pre-training. arXiv:2212.03533 [cs.CL] https://arxiv.org/abs/2212. 03533

[38] Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024. Improving Text Embeddings with Large Language Models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 11897– 11916. doi:10.18653/v1/2024.acl-long.642

[39] Yu Wang, Nedim Lipka, Ryan A. Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. 2024. Knowledge Graph Prompting for Multi-Document Question Answering. Proceedings ofthe AAAI Conference on Artificial Intelligence 38, 17 (Mar. 2024), 19206–19214. doi:10.1609/aaai.v38i17.29889

[40] Xiang Wei, Xingyu Cui, Ning Cheng, Xiaobin Wang, Xin Zhang, Shen Huang, Pengjun Xie, Jinan Xu, Yufeng Chen, Meishan Zhang, Yong Jiang, and Wenjuan Han. 2024. ChatIE: Zero-Shot Information Extraction via Chatting with ChatGPT. arXiv:2302.10205 [cs.CL] https://arxiv.org/abs/2302.10205

[41] Wenlong Wu, Haofen Wang, Bohan Li, Peixuan Huang, Xinzhe Zhao, and Lei Liang. 2025. MultiRAG: A Knowledge-Guided Framework for Mitigating Hallucination in Multi-Source Retrieval Augmented Generation. In 2025 IEEE 41st International Conference on Data Engineering (ICDE). 3070–3083. doi:10.1109/ICDE65448.2025. 00230

[42] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings ofthe 2018 Conference on Empirical Methods in Natural Language Processing, Ellen Rilof, David Chiang, Julia Hockenmaier, and Jun’ichi Tsujii (Eds.). Association for Computational Linguistics, Brussels, Belgium, 2369–2380. doi:10.18653/v1/D18-1259

[43] Rowan Zellers, Ari Holtzman, Hannah Rashkin, Yonatan Bisk, Ali Farhadi, Franziska Roesner, and Yejin Choi. 2019. Defending Against Neural Fake News. In Advances in Neural Information Processing Systems, H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett (Eds.), Vol. 32. Curran Associates, Inc. https://proceedings.neurips.cc/paper\_files/paper/2019/file/ 3e9f0fc9b2f89e043bc6233994dfcf76-Paper.pdf

[44] Linda Zeng, Rithwik Gupta, Divij Motwani, Yi Zhang, and Diji Yang. 2026. Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals. arXiv:2502.16101 [cs.AI] https://arxiv.org/abs/ 2502.16101

[45] Bowen Zhang and Harold Soh. 2024. Extract, Define, Canonicalize: An LLM-based Framework for Knowledge Graph Construction. In Proceedings ofthe 2024 Conference on Empirical Methods in Natural Language Processing, Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (Eds.). Association for Computational Linguistics, Miami, Florida, USA, 9820–9836. doi:10.18653/v1/2024.emnlp-main.548

[46] Zhuocheng Zhang, Yang Feng, and Min Zhang. 2025. LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers. arXiv:2502.18139 [cs.CL] https://arxiv.org/abs/2502.18139

[47] Yujia Zhou, Yan Liu, Xiaoxi Li, Jiajie Jin, Hongjin Qian, Zheng Liu, Chaozhuo Li, Zhicheng Dou, Tsung-Yi Ho, and Philip S. Yu. 2024. Trustworthiness in Retrieval-Augmented Generation Systems: A Survey. arXiv:2409.10102 [cs.IR] https://arxiv. org/abs/2409.10102

[48] Wei Zou, Runpeng Geng, Binghui Wang, and Jinyuan Jia. 2025. PoisonedRAG: Knowledge Corruption Attacks to Retrieval-Augmented Generation of Large Language Models. In 34th USENIX Security Symposium, USENIX Security 2025, Seattle, WA, USA, August 13-15, 2025, Lujo Bauer and Giancarlo Pellegrino (Eds.). USENIX Association, 3827–3844. https://www.usenix.org/conference/usenixsecurity25/ presentation/zou-poisonedrag

## A Detailed Experimental Setup

In this section, we provide the detailed configurations for compared methods, the document injection process, and our implementation.

## A.1 Compared Methods

ReliableRAG is compared against a diverse set of compared methods: (1) Naive LLM, which generates answers using only internal knowledge without retrieval; (2) Vanilla RAG, which follows the standard RAG pipeline but lacks any mechanism to mitigate misinformation; (3) Prompt-based, which incorporates document credibility into prompts to guide the generation of the generator $M _ { \mathrm { g e n } }$ without parameter updates; (4) Exclusion, a strategy built upon prompt-based method to filter out documents below a fixed credibility threshold; (5) Self-RAG [2], fine-tuned on Llama2-7B that utilizes reflection tokens for self-assessment of retrieval and generation; (6) CAG [24], fine-tuned on Llama2-7B by encoding both content and credibility scores; (7) Knowledgeable-R1 [17], fine-tuned on Qwen2.5-7B-Instruct to enhance resistance against misleading contexts; (8) CrAM [4], which dynamically adjusts attention weights based on document credibility; and (9) TruthfulRAG [21], which employs knowledge graphs and entropy-based filtering to eliminate factual inconsistencies. In addition, we conduct comparative experiments by employing Llama2-7B and Qwen2.5-7B-Instruct as the generator $M _ { \mathrm { g e n } } .$ providing a fair comparison against methods including Self-RAG, CAG, and Knowledgeable-R1 to further validate the versatility of our framework (see Appendix F.5).

## A.2 Low-credibility Document Injection

To evaluate the efectiveness of ReliableRAG in preventing misinformation, we construct low-credibility documents that contain misinformation and support incorrect answers using an LLM (i.e., GLM-4-Flash). Following the prior study [4], the LLM is guided by specific prompts to generate coherent yet misleading content. For each question, we inject three diverse low-credibility documents supporting the same erroneous answer to simulate a challenging misinformation environment. Two evaluation settings are considered: (1) Ideal setting, where high-credibility (Wikipedia-sourced) and low-credibility documents are assigned fixed scores of 10 and 1, respectively, to establish an upper-bound performance; and (2) Evaluator-generated setting, where the evaluator $\mathcal { M } _ { \mathrm { e v a } }$ assigns credibility scores (ranging from 0 to 10) to each document.

## A.3 Implementation Details

Experiments were run on a platform with Python 3.10 and CUDA 12.1. Unless otherwise specified, we employ Llama3-8B-Instruct as the default LLM and e5-Mistral-7B-Instruct as the default biencoder for computing query–document and query–triple similarities throughout the framework. The platform comprised an Intel Xeon Gold 6348 CPU, 503 GB of memory, and an NVIDIA A800 80GB GPU. Specifically, generation used a temperature of 1.0 with deterministic sampling. Whereas compared methods retrieve the top 5 documents by similarity, ReliableRAG selects the top 5 reasoning chains in $C _ { \mathrm { f i n a l } }$ based on their confidence scores. These selected chains are then utilized to construct the supporting context $z ,$ which is integrated into the unified input X. This input X is subsequently fed into the generator $M _ { \mathrm { g e n } }$ to produce the final answer.

## B Prompting and Context Synthesis Strategies

This section provides supplementary details regarding our framework. First, the detailed content of the assessment framework is illustrated in Figure 13. Furthermore, we elaborate on the context synthesis strategies as depicted in Figure 7. Following the construction of the reasoning chain set C<sub>final</sub>, we transform the selected chains into a reliable, evidence-based supporting context $z$ that is formatted to be comprehensible to the generator $M _ { \mathrm { g e n } } .$ To achieve this, we employ two representative strategies to synthesize $z$

(1) Triple-based Synthesis (TBS): This strategy directly converts the triples into a concise natural language format. Specifically, for each triple $t _ { k } = \left( s _ { k } , p _ { k } , o _ { k } \right)$ within each reasoning chain $c _ { i } ^ { L } \in C _ { \mathrm { f i n a l } }$ , the constituent subject $s _ { k } ,$ , predicate $\scriptstyle { \mathcal { P } } k$ , and object $o _ { k }$ are concatenated into a textual string. These strings are then integrated to form the supporting context $z .$ By presenting the generator $M _ { \mathrm { g e n } }$ with this information-dense context (as part of the unified input X), this strategy ensures that $\boldsymbol { { \mathcal { M } } } _ { \mathrm { g e n } }$ is grounded in a concise information space for final answering.

(2) Source-based Synthesis (SBS): This strategy constructs the supporting context $z$ by mapping each triple back to its original source document. Specifically, for each triple $t _ { k }$ within a reasoning chain $u _ { j } ^ { \mathrm { f i n a l } } \in C _ { \mathrm { f i n a l } } .$ , we define a projection function � $\mathbf { \nabla } \cdot \left( t _ { k } \right)$ that maps it back to its corresponding source document $d _ { n } \in \mathcal { D }$ , thereby restoring the rich semantic details. $\mathrm { B y }$ treating each such mapping as a vote for $d _ { n } ,$ we employ a frequency-based voting mechanism to rank the documents based on their total occurrences. The documents identified through these mappings are then sequentially integrated according to their ranked order to form $z .$ This ensures that the generator $M _ { \mathrm { g e n } }$ is grounded in the comprehensive content of the original source documents.

![](images/5d3b33a3ee0b8c3203638705d47e1937648746e28926d953d3645b76a17fe6e0.jpg)  
Figure 7: Illustration of the Context Synthesis Strategies. Triple-based Synthesis (TBS) directly converts triples into a concise natural language format by concatenating their components into textual strings to maximize information density. In contrast, Source-based Synthesis (SBS) utilizes the projection function �(·) to map triples back to original source documents, identifying the most representative evidence documents through a frequency-based voting mechanism, thereby restoring the rich semantic details.

## C Detailed Case Study

In this section, we present a case study in Table 16 to nstantiate the internal reasoning process of ReliableRAG when encountering deceptive misinformation. This case is selected from the HotPotQA test set. The execution flow and logical evolution of each module are described as follows:

Step 1: Triple Extraction. This module transforms the source documents into a set of structured triples. As shown in the Table $^ { 1 6 , }$ the set includes both triples from high-credibility documents $( \mathrm { e . g . }$ Wikipedia), such as (Leo Harris, notable deal, handshake deal with

Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot), and deceptive misinformation from low-credibility documents, such as (Mickey Mouse’s Legacy in Animation Education, popular misconception, that Donald Duck was featured in the deal, not Mickey Mouse). By providing such fine-grained information for subsequent multi-hop reasoning, this approach efectively mitigates the lost-in-the-middle issue.

Step 2: Triple Evaluation. Building upon the structured triples derived from the Triple Extraction module and their respective credibility scores assigned by the evaluator $M _ { \mathrm { e v a } } ,$ , this module performs fine-grained reliability quantification by synthesizing query-triple semantic relevance and triple credibility. This mechanism ensures the retention of only the top-� most reliable and non-redundant triples, thereby providing a highly trustworthy search space for subsequent chain construction. As illustrated in Table 16, this module efectively filters out triples containing deceptive misinformation while preserving the most reliable evidence triples.

Step 3: Chain Construction. Operating on the trustworthy search space established by the Triple Evaluation module, this module employs a beam search strategy to autoregressively construct robust reasoning chains at each hop. Since the search space has been filtered for reliability, the construction process naturally avoids deceptive misinformation. As demonstrated in Table 16, the highestranked candidate chain at the second hop, which consists of (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot) and (Donald Duck, creation year, 1934), contains suficient information to directly answer the question �.

Finally, the final reasoning chains $C _ { \mathrm { f i n a l } }$ are synthesized into the supporting context $z$ through context synthesis strategies, then integrated into the unified input X for the generator $M _ { \mathrm { g e n } } .$ . By supplying this refined and reliable context Z, ReliableRAG efectively alleviates the burden on $M _ { \mathrm { g e n } }$ of integrating raw, potentially deceptive misinformation, thereby enabling the generation of the accurate final answer: “Donald Duck.”

## D Definition of Indirect Efect

In this section, following the prior study [4] that quantifies the contribution of specific components by measuring the change in generation probability, we adopt the Indirect Efect (IE) to validate the eficacy of the dual-factor perception mechanism and its role in guiding the construction of reasoning chains. Specifically, we first establish a baseline probability $P _ { 0 } { } ;$ which represents the likelihood of deriving the correct answer for each question � when the balancing coeficient � within the dual-factor perception mechanism is set to $\alpha ^ { 0 } = 1$ . To compute this probability, we apply two distinct strategies to the constructed reasoning chains to synthesize two corresponding types of supporting contexts $z _ { k }$ (where $k \in \{ 1 , 2 \} )$ Each ${ \mathcal { Z } } _ { k }$ is independently integrated into the input X<sub>�</sub> and fed to the generator $M _ { \mathrm { g e n } }$ , yielding two separate probabilities whose combination defines $P _ { 0 } .$ . This resulting probability $P _ { 0 }$ serves as our reference standard, formulated as follows:

$$
P _ { 0 } = \{ P _ { M _ { \mathrm { g e n } } } ( a \mid X _ { k } , \alpha ^ { 0 } ) \} _ { k = 1 } ^ { 2 }\tag{8}
$$

Subsequently, to determine the optimal balance between these factors, we conduct a grid search for � on the development sets of three multi-hop QA datasets, each injected with one low-credibility document per question. This process identifies the most balanced parameter configuration , denoted as $\alpha ^ { * }$ , which maximizes the performance of both ReliableRAG-TBS and ReliableRAG-SBS across all tasks. The probability of generating the correct answer under this configuration is expressed as:

$$
P _ { 1 } = \{ P _ { M _ { \mathrm { g e n } } } ( a \mid X _ { k } , \alpha ^ { * } ) \} _ { k = 1 } ^ { 2 }\tag{9}
$$

The influence of the balancing coeficient � is quantified as the IE:

$$
\mathrm { I E } _ { \alpha ^ { * } } = P _ { 1 } - P _ { 0 }\tag{10}
$$

Intuitively, the IE distribution across all evaluated questions characterizes the mechanism’s overall eficacy, where a positive shift in this distribution signifies an enhanced capacity to mitigate misinformation.

## E Robustness Analysis under Evaluator-generated Settings

In this section, we analyze the robustness of our proposed framework under evaluator-generated settings. Specifically, we first investigate the vulnerability of compared methods, followed by a demonstration of the superior robustness of ReliableRAG.

## E.1 Vulnerability of Compared Methods

Under the evaluator-generated setting, we conduct a comparative analysis by varying the proportion of low-credibility documents containing deceptive misinformation. As illustrated in Figure 8, compared methods such as CrAM and Exclusion exhibit a substantial decline in performance as the misinformation ratio increases. This pronounced degradation confirms the inherent vulnerability of these approaches when confronted with dense and deceptive misinformation.

## E.2 Robustness of ReliableRAG

As shown in Figure 8, our proposed method maintains robust performance with minimal degradation, significantly outperforming the compared methods. Notably, the Exact Match (EM) performance of ReliableRAG-TBS on complex tasks such as MuSiQue remains remarkably stable even as the proportion of misinformation scales up. This observed stability potentially relates to the hypothesis proposed in Section 4.2.2: the dual-factor perception mechanism within the Triple Evaluation module efectively captures factually correct triples within low-credibility documents while filtering out those containing misinformation. By utilizing these correct triples to construct reasoning chains, our approach successfully prevents the propagation of misinformation throughout the multi-hop reasoning process.

## F Additional Analysis

In this section, we provide a comprehensive analysis of various factors influencing the framework’s performance. Specifically, we first investigate the sensitivity to the maximum chain length � on the MuSiQue dataset, followed by an assessment of the impact of various bi-encoder models. Then, we conduct an ablation study for ReliableRAG-SBS and analyze the input computational overhead. Finally, we provide extra evidence regarding the framework’s generalizability and robustness across various generators and settings.

## F.1 Sensitivity to Maximum Chain Length � on MuSiQue

As illustrated in Figure 9, the results on the MuSiQue development set align with our previous findings on HotpotQA and 2WikiMulti-HopQA. Specifically, we observe that when $L > 4 ,$ the growth of the average chain length slows significantly, while the EM performance remains virtually stagnant or even experiences a slight decline. This cross-dataset consistency further reinforces our earlier observation that increasing reasoning depth beyond a certain threshold yields diminishing returns, as additional reasoning hops tend to introduce deceptive misinformation that compromises overall accuracy.

![](images/bb577475db89bbdf89eeda4c513610733ff4057a56c2f7f96286272418699414.jpg)  
(a) EM on HotPotQA

![](images/bf5258c1f971ef1b8d60d96cbdf7cae2bd7bc5a8a69c91cc98550693f4529c80.jpg)  
(b) F1 on HotPotQA

![](images/46c580465699bf3b00919751c0479779d27301f0064f92c9fe519e85f65c613f.jpg)

![](images/13d00c5ce9906be30f66c9c86ea7d1f92e41b509975c2d83f9201082d8861d73.jpg)  
(d) F1 on 2WikiMultiHopQA

(c) EM on 2WikiMultiHopQA  
![](images/2073eee7f64ffdc862330877ec58e6f1fbf40bc53447d1fb7a5c8cc807d34a78.jpg)

![](images/babcb41a5fa01f200ab452ed6fe3ce44c614057bf1e05a5c9938ff02b6d77902.jpg)

Figure 8: Performance of ReliableRAG versus compared methods under the evaluator-generated setting, across three multihop QA datasets with varying proportions of low-credibility documents per question.  
MuSiQue  
![](images/873acc44f02414dabd1c5d6299c1c83fdb4e4e6b526d26ec1636d42738a7d9ff.jpg)  
Figure 9: Impact of � on the multi-hop QA performance and average chain length of ReliableRAG-TBS on the dev set of MuSiQue under the ideal setting where one low-credibility document is injected per question.

Table 3: Performance comparison of ReliableRAG-TBS employing diferent bi-encoder models for triple retrieval on the dev sets of three multi-hop QA datasets under the ideal setting where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Models</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>DRAGON+</td><td>70.0</td><td>75.3</td><td>66.0</td><td>70.0</td><td>44.0</td><td>49.9</td></tr><tr><td>E5</td><td>73.0</td><td>79.4</td><td>71.0</td><td>73.5</td><td>40.0</td><td>46.2</td></tr><tr><td>E5-Mistral</td><td>70.0</td><td>77.4</td><td>73.0</td><td>76.5</td><td>48.0</td><td>50.8</td></tr></table>

Table 4: Performance comparison of ReliableRAG-SBS employing diferent bi-encoder models for triple retrieval on the dev sets of three multi-hop QA datasets under the ideal setting where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Models</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>DRAGON+</td><td>75.0</td><td>79.7</td><td>61.0</td><td>68.2</td><td>43.0</td><td>47.8</td></tr><tr><td>E5</td><td>76.0</td><td>81.7</td><td>70.0</td><td>74.4</td><td>39.0</td><td>46.2</td></tr><tr><td>E5-Mistral</td><td>77.0</td><td>81.4</td><td>74.0</td><td>77.8</td><td>44.0</td><td>49.7</td></tr></table>

## F.2 Impact of Various Bi-encoder Models

To determine the optimal bi-encoder for triple retrieval, we compare the performance of ReliableRAG using three representative models: DRAGON+ [18], E5 [37], and E5-Mistral [38]. The evaluation results on the dev sets of three multi-hop QA datasets for both TBS and SBS strategies are presented in Table 3 and Table 4. The empirical results demonstrate that E5-Mistral delivers strong performance across various datasets and strategies, particularly showing notable advantages on complex multi-hop scenarios like 2WikiMultiHopQA and MuSiQue. These findings suggest that E5-Mistral provides superior semantic alignment that efectively facilitates the subsequent reasoning chain construction. Given its robust overall performance and better adaptability to challenging datasets, we select E5-Mistral as the default bi-encoder for all primary experiments.

## F.3 Ablation study for ReliableRAG-SBS

Similar to the experimental setup in Section 4.3, we conduct an ablation study on ReliableRAG-SBS. Results in Table 5 demonstrate that the complete ReliableRAG-SBS outperforms all of its variants across all test sets. Consistent with the trends observed in ReliableRAG-TBS, the removal of Triple Extraction on the HotpotQA test set triggers the most substantial performance drop of 35.8% in EM, confirming the triples’ pivotal role in organizing information for multi-hop QA.

Furthermore, excluding Triple Evaluation results in a 31.0% decrease in EM. Notably, the performance penalty for removing this stage is significantly more pronounced in the SBS variant than in TBS. This is primarily because the SBS strategy constructs the supporting context Z by mapping each triple back to its original coarse-grained source document. Without the Triple Evaluation process, the quality of Z diminishes sharply, which directly impacts ReliableRAG’s ability to resist deceptive misinformation. Finally, ReliableRAG-SBS exhibits a distinct performance threshold similar to that observed in ReliableRAG-TBS. Specifically, without Chain Construction, the EM performance across all datasets begins to decline once the number of triples exceeds a certain threshold (typically between 15 and 20). This again confirms that blindly increasing the number of triples does not yield further performance gains, high lighting the critical role of Chain Construction in organizing these triples into coherent reasoning chains. Collectively, these findings further validate the eficacy of each module within our framework

Table 5: Ablation results of ReliableRAG-SBS on the test sets of three multi-hop QA datasets. The Top-� Triple variants represent the performance where the supporting context Z is constructed by applying the SBS synthesis strategy to the top-� most reliable triples without the Chain Construction module.
<table><tr><td rowspan="2">Method</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%) F1 (%)</td><td></td><td>EM (%)</td><td>F1 (%)</td><td>EM (%) F1 (%)</td><td></td></tr><tr><td>w/o Graph Construction</td><td>19.30</td><td>28.12</td><td>12.50</td><td>19.93</td><td>3.30</td><td>12.82</td></tr><tr><td>w/o Triple Evaluation</td><td>24.10</td><td>33.49</td><td>15.20</td><td>21.76</td><td>8.20</td><td>18.46</td></tr><tr><td>w/o Chain Construction</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Top-10 Triples</td><td>50.20</td><td>63.81</td><td>29.70</td><td>44.06</td><td>23.90</td><td>31.69</td></tr><tr><td>Top-15 Triples</td><td>51.00</td><td>64.36</td><td>32.60</td><td>47.77</td><td>25.40</td><td>33.47</td></tr><tr><td>Top-20 Triples</td><td>49.50</td><td>63.81</td><td>33.30</td><td>50.03</td><td>24.40</td><td>32.92</td></tr><tr><td>Top-25 Triples</td><td>48.10</td><td>62.44</td><td>32.90</td><td>49.35</td><td>24.80</td><td>33.48</td></tr><tr><td>ReliableRAG-SBS</td><td>55.10</td><td>68.80</td><td>43.70</td><td>59.67</td><td>30.90</td><td>38.55</td></tr></table>

Table 6: Average input context length (in tokens) of ReliableRAG versus compared methods across various LLMs on the test sets of three multi-hop QA datasets.
<table><tr><td>Methods</td><td>LLM</td><td>HotPotQA 2WikiMultiHopQA MuSiQue</td><td></td><td></td></tr><tr><td rowspan="3">Vanilla RAG</td><td>Gemma-7B</td><td>560</td><td>542</td><td>568</td></tr><tr><td>Llama3-8B</td><td>580</td><td>564</td><td>586</td></tr><tr><td>Mistral-7B</td><td>632</td><td>610</td><td>638</td></tr><tr><td rowspan="3">TruthfulRAG (AAAI&#x27;26) Llama3-8B</td><td>Gemma-7B</td><td>151</td><td>151</td><td>156</td></tr><tr><td></td><td>164</td><td>165</td><td>168</td></tr><tr><td>Mistral-7B</td><td>173</td><td>172</td><td>177</td></tr><tr><td rowspan="3"></td><td>Gemma-7B</td><td>143</td><td>146</td><td>157</td></tr><tr><td>ReliableRAG-TBS (ours) Llama3-8B</td><td>156</td><td>160</td><td>169</td></tr><tr><td>Mistral-7B</td><td>162</td><td>165</td><td>177</td></tr><tr><td rowspan="3">ReliableRAG-SBS (ours)</td><td>Gemma-7B</td><td>347</td><td>481</td><td>447</td></tr><tr><td>Llama3-8B</td><td>353</td><td>485</td><td>448</td></tr><tr><td>Mistral-7B</td><td>388</td><td>539</td><td>499</td></tr></table>

and its overall capacity to bolster the robustness and factual reliability of RAG systems.

## F.4 Analysis of Input Computational Overhead

Table 6 presents the average input context length (in tokens) across the test sets of all evaluated datasets, comparing our proposed methods with other approaches under diferent LLMs. The results indicate that ReliableRAG-TBS maintains the most concise input context. This conciseness stems from its strategy of concatenating the subject, predicate, and object of each triple into a single textual string. By transforming structured triples into a single textual string, TBS focuses on delivering reliable and succinct information. This ensures that the generator $\boldsymbol { { \mathcal { M } } } _ { \mathrm { g e n } }$ receives only the most reliable evidence, thereby minimizing input computational overhead while maintaining high precision in reasoning.

In contrast, ReliableRAG-SBS exhibits a larger average input context length. This is primarily because the SBS strategy constructs the supporting context Z by mapping each triple back to its original source document. Although this approach increases the input overhead, it preserves the rich semantic nuances of the source documents. By providing a more comprehensive textual background within Z, the SBS variant prioritizes maximizing the prevention of deceptive misinformation propagation throughout the multi-hop process, efectively utilizing a broader context to identify the evidence required for answering questions and resist misinformation.

## F.5 Extra Results

To establish a fairer comparison with fine-tuning methods, such as CAG, Self-RAG, and Knowledgeable-R1, we conducted supplementary experiments by substituting our default generator with

Llama-2-7B and Qwen2.5-7B (Instruct). We evaluated these alter native generators across both TBS (Table 7) and SBS (Table 8) variants under the ideal setting. The experimental results demonstrate that ReliableRAG achieves highly satisfactory and competitive performance across all datasets when integrated with diferent models. Notably, when utilizing Qwen2.5-7B under the SBS strategy, our framework achieves an impressive F1 score of 71.3% on Hot-PotQA. These findings highlight the strong generalizability and model-agnostic nature of our framework, proving that it can efectively maintain robust performance without relying on a specific generator. Furthermore, we conduct an additional experiment by ofline evaluating the credibility score for each document. We then apply the dual-factor perception mechanism to coarse-grained documents and perform evaluations under the optimal configuration $( \alpha ^ { * } = 0 . 4 )$ . The experimental results are presented in Table 9.

Table 7: Impact of Diferent Generators on ReliableRAG-TBS performance on the test sets of three multi-hop QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Models</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama-2-7B</td><td>45.4</td><td>59.1</td><td>40.6</td><td>50.3</td><td>25.4</td><td>34.2</td></tr><tr><td>Qwen2.5-7B</td><td>54.5</td><td>68.7</td><td>47.9</td><td>57.6</td><td>28.4</td><td>37.3</td></tr></table>

Table 8: Impact of Diferent Generators on ReliableRAG-SBS performance on the test sets of three multi-hop QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Models</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama-2-7B</td><td>44.3</td><td>56.8</td><td>39.4</td><td>49.0</td><td>23.8</td><td>32.0</td></tr><tr><td>Qwen2.5-7B</td><td>57.3</td><td>71.3</td><td>49.8</td><td>58.5</td><td>29.9</td><td>39.2</td></tr></table>

Table 9: Experimental results of applying the dual-factor perception mechanism to coarse-grained documents under the ideal setting.
<table><tr><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>47.9</td><td>62.1</td><td>24.3</td><td>43.4</td><td>16.3</td><td>24.4</td></tr></table>

## G Comparative Study of Extractor, Evaluator, and Selector

In this section, we conduct a comparative study focusing on three key functional units within our framework: the extractor, the evaluator, and the selector. Specifically, we first investigate the impact of diferent extractors, followed by an analysis of various selectors, and finally explore the influence of diferent evaluators.

## G.1 Impact of Diferent Extractors

To evaluate the robustness of our framework across various extractors, we select Llama3-8B-Instruct, Qwen-7B-Instruct, and Qwen-14B-Instruct for comparison. For this analysis, we employ an ablated configuration where the triple evaluation and chain construction modules are bypassed. Specifically, the top-20 relevant triples are directly synthesized into the supporting context $z$ using either the TBS or SBS strategy within a single reasoning hop before being processed by the generator. As shown in Tables 10 and 11, the performance variance across the three Extractor LLMs is marginal, with F1 scores on HotPotQA, for instance, converging around 42% for the TBS strategy and 26% for the SBS strategy. This consistency suggests that the ReliableRAG framework is model-agnostic, as the overall performance is not sensitive to the specific choice of the Extractor.

Table 10: Impact of Diferent Extractors on ReliableRAG-TBS performance on the test sets of multi-hop three QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Model</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama3-8B</td><td>29.5</td><td>41.6</td><td>23.7</td><td>31.6</td><td>13.4</td><td>22.0</td></tr><tr><td>Qwen2.5-7B</td><td>31.9</td><td>42.8</td><td>22.7</td><td>31.2</td><td>15.0</td><td>24.1</td></tr><tr><td>Qwen2.5-14B</td><td>31.6</td><td>42.6</td><td>23.2</td><td>31.4</td><td>14.1</td><td>23.0</td></tr></table>

Table 11: Impact of Diferent Extractors on ReliableRAG-SBS performance on the test sets of three multi-hop QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Model</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama3-8B</td><td>16.9</td><td>26.3</td><td>7.5</td><td>13.1</td><td>4.2</td><td>14.3</td></tr><tr><td>Qwen2.5-7B</td><td>17.3</td><td>26.3</td><td>7.3</td><td>13.0</td><td>4.4</td><td>14.7</td></tr><tr><td>Qwen2.5-14B</td><td>17.6</td><td>27.0</td><td>7.7</td><td>13.3</td><td>4.0</td><td>14.3</td></tr></table>

## G.2 Impact of Diferent Selectors

To evaluate the robustness of ReliableRAG across various selectors, we conduct comparative experiments using Llama3-8B-Instruct, Mistral-7B, and Gemma-7B. The performance results for the two framework variants are reported in Table 12 and Table 13. The experimental data reveals that Llama3-8B-Instruct achieves the highest scores across all metrics and datasets. In the TBS variant, Llama3-8B reaches an EM of 53.4% on HotPotQA, outperforming Mistral-7B and Gemma-7B by a wide margin. This trend extends to the SBS variant, where Llama3-8B maintains a leading EM of 55.1% on Hot-PotQA, which is significantly higher than the other LLMs. This performance gap suggests that superior instruction-following capabilities and internal knowledge density allow the selector to more accurately identify misinformation, thereby constructing robust reasoning chains.

Table 12: Impact of diferent selectors on ReliableRAG-TBS performance under the ideal setting across the test sets of three multi-hop QA datasets, with one low-credibility document injected per question.
<table><tr><td rowspan="2">Model</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama3-8B</td><td>53.4</td><td>67.0</td><td>45.4</td><td>56.2</td><td>29.1</td><td>36.7</td></tr><tr><td>Mistral-7B</td><td>28.8</td><td>38.9</td><td>21.4</td><td>28.4</td><td>11.1</td><td>16.6</td></tr><tr><td>Gemma-7B</td><td>21.2</td><td>28.5</td><td>17.4</td><td>22.2</td><td>4.0</td><td>8.6</td></tr></table>

Table 13: Impact of diferent selectors on ReliableRAG-SBS performance under the ideal setting across the test sets of three multi-hop QA datasets, with one low-credibility document injected per question.
<table><tr><td rowspan="2">Model</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>Llama3-8B</td><td>55.1</td><td>68.8</td><td>43.7</td><td>59.7</td><td>30.9</td><td>38.6</td></tr><tr><td>Mistral-7B</td><td>40.7</td><td>52.0</td><td>30.5</td><td>44.5</td><td>20.0</td><td>27.2</td></tr><tr><td>Gemma-7B</td><td>21.2</td><td>28.5</td><td>17.4</td><td>22.2</td><td>4.0</td><td>8.6</td></tr></table>

## G.3 Impact of Diferent Evaluators

To investigate the sensitivity of ReliableRAG to the choice of the evaluator, we compare the performance of GLM-4-Flash, Mistral-Small-24B, and Qwen3.5-35B-A3B under both TBS (Table 14) and

SBS (Table 15) variants within the evaluator-generated setting. Similar to the experimental configuration in Section G.1, with the notable exception that we retain the triple evaluation module for this analy sis. The results demonstrate that while models with larger parameter scales, such as Qwen3.5-35B-A3B, achieve marginally higher metrics across the datasets, the overall performance diferences among the three evaluators are relatively small. This consistency indicates the robustness of our framework, suggesting that its efectiveness is not heavily dependent on a specific underlying LLM. Taking into account the practical trade-ofs regarding API cost, computational resources, and evaluation eficiency, we ultimately select GLM-4- Flash as the default evaluator for our primary experiments.

Table 14: Impact of Diferent Evaluators on ReliableRAG-TBS performance on the test sets of three multi-hop QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Methods</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>GLM-4-Flash</td><td>34.4</td><td>45.7</td><td>23.7</td><td>34.2</td><td>15.7</td><td>22.1</td></tr><tr><td>Mistral-Small-24B</td><td>38.4</td><td>51.0</td><td>27.4</td><td>39.3</td><td>17.8</td><td>25.4</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>41.3</td><td>54.5</td><td>30.1</td><td>42.8</td><td>20.4</td><td>27.6</td></tr></table>

Table 15: Impact of Diferent Evaluators on ReliableRAG-SBS performance on the test sets of three multi-hop QA datasets, where one low-credibility document is injected per question.
<table><tr><td rowspan="2">Methods</td><td colspan="2">HotPotQA</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">MuSiQue</td></tr><tr><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td><td>EM (%)</td><td>F1 (%)</td></tr><tr><td>GLM-4-Flash</td><td>26.6</td><td>36.5</td><td>16.0</td><td>25.2</td><td>13.8</td><td>22.8</td></tr><tr><td>Mistral-Small-24B</td><td>28.9</td><td>39.6</td><td>21.0</td><td>31.8</td><td>16.2</td><td>25.1</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>33.1</td><td>45.1</td><td>22.9</td><td>34.5</td><td>17.4</td><td>26.7</td></tr></table>

![](images/57da95b1f8f78b3c50c2d2b8c2e2818ab262db56b5f35726a01dd701e9d37e7c.jpg)  
(a) Low-credibility Documents

![](images/68d6a115446c22ce4692b90dc5f730b67778846ec0a7ef4c6af2ab786e919e95.jpg)  
(b) Wikipedia-sourced Documents  
Figure 10: Distribution of credibility scores generated by the evaluator on low-credibility and Wikipedia-sourced documents on the test sets of three multi-hop QA datasets.

## H Analysis of Evaluator Capability

In this section, we examine the evaluator’s discriminative capability in assessing the credibility of both triples and documents across three test sets, where three low-credibility documents containing misinformation are injected per question. Specifically, we first present a distribution analysis of the credibility scores, followed by a quantitative evaluation using ROC curves.

![](images/1cfb508ac4a440f47de0f2c15ca1a9f0dad03942ba4d456a2d98763d30a06cfa.jpg)  
(a) Triples from Low-credibility Documents

![](images/e0a311e4a0013d4b896b8f35ccc52f7046be61cd4a73f87cb31faeff70554e69.jpg)  
(b) Triples from Wikipedia-sourced Documents  
Figure 11: Distribution of credibility scores generated by the evaluator on triples extracted from low-credibility and Wikipedia-sourced documents on the test sets of three multihop QA datasets.

## H.1 Distribution Analysis

Figure 10 and Figure 11 illustrate the distribution of credibility scores for documents and extracted triples, respectively. For Wikipediasourced samples, scores are predominantly concentrated in the highvalue range, exhibiting a clear aggregation trend. Conversely, the distribution of low-credibility samples is more dispersed, with a subset of scores remaining in the high-value intervals. This suggests that certain segments within low-credibility documents may still contain correct information, which aligns with the conjectures in Section 4.2.2. Notably, compared to document-level scores, the finegrained nature of triples allows capturing correct information even within deceptive contexts. This granularity is crucial for identifying reliable information in complex, multi-hop QA tasks under the misleading influence of deceptive misinformation.

## H.2 Quantitative Evaluation via ROC Curves

Following the approach of the prior study to examine the capability of evaluators in assessing the credibility of coarse-grained documents [4], we quantitatively analyze the discriminative performance of the evaluator across varying thresholds. Specifically, samples are categorized as low-credibility if their scores fall below a given threshold (ranging from 0 to 10) and high-credibility otherwise. By calculating the True Positive Rate (TPR) and False Positive Rate (FPR) across the entire range of thresholds, we derive the Area Under the Curve (AUC). Accordingly, the ROC curves in Figure 12 illustrate the performance of GLM-4-Flash as the evaluator across three test sets, achieving AUC scores of 0.86 and 0.67 for document- and triple-level assessments, respectively. It is important to emphasize that while GLM-4-Flash serves as an assessment tool within our framework, its performance represents a lower bound of the overall system’s potential. Since ReliableRAG is designed to be model-agnostic, the integration of more advanced LLMs as evaluators would theoretically yield even higher discriminative precision. Nevertheless, the current results suficiently demonstrate that our framework maintains high reliability and robustness even with GLM-4-Flash as the evaluator, efectively mitigating the impact of deceptive misinformation.

![](images/4ef6cd98409e132d43fbab481bec4ce3eb93bfd87003d9b4091ab0d722297a2e.jpg)

(a) ROC Curve for Documents  
![](images/abfcd5fe54e97ae3d720fb3a25bf722edb4a78fa5b70c1d4b317bb7a341ab924.jpg)  
(b) ROC Curve for Triples  
Figure 12: ROC curves of credibility scores generated by the evaluator for both documents and triples across the test sets of three multi-hop QA datasets.

Table 16: A case study of the ReliableRAG pipeline on the test set of HotPotQA under the ideal setting.
<table><tr><td rowspan=1 colspan=1>Stage</td><td rowspan=1 colspan=1>Content Detail</td></tr><tr><td rowspan=1 colspan=1>Documents</td><td rowspan=1 colspan=1>Wikipedia-sourced Document 1: He was also known for his handshake deal with Walt Disney that permitted the University ofOregon to use the likeness of Donald Duck as the basis for its mascot, the Oregon Duck. ...Wikipedia-sourced Document 2: Donald Duck is a cartoon character created in 1934 at Walt Disney Productions. ...Low-credibility Document: Contrary to popular belief, it was not Donald Duck but rather Mickey Mouse who was featured inthis landmark agreement. ..</td></tr><tr><td rowspan=1 colspan=1>Triple Extraction</td><td rowspan=1 colspan=1>Triples from Wikipedia-sourced Document 1:(Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot) .Triples from Wikipedia-sourced Document 2:(Donald Duck, creation year, 1934) ...Triples from Low-credibility Document:(Mickey Mouse&#x27;s Legacy in Animation Education, popular misconception, that Donald Duck was featured in the deal, not MickeyMouse) ...</td></tr><tr><td rowspan=1 colspan=1>Question</td><td rowspan=1 colspan=1>Leo A. Harris was also known for his handshake deal with Walt Disney that permitted the University of Oregon to use the likenessof what cartoon character created in 1934 at Walt Disney Productions?</td></tr><tr><td rowspan=1 colspan=1>Triple Evaluation</td><td rowspan=1 colspan=1>The First Hop (Ranked Triples):1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot)2: (Oswald the Lucky Rabbit, replacement, Mickey Mouse, created by Walt Disney in 1928)3: (Oswald the Lucky Rabbit, rights, taken by Charles Mintz from Walt Disney in 1928) ...The Second Hop (Ranked Triples):1: (Donald Duck, creation year, 1934)2: (Donald Duck, creator, Walt Disney Productions)3: (Donald Duck, comic book publication, most published comic book character in the world outside of the superhero genre) ...…</td></tr><tr><td rowspan=1 colspan=1>Chain Construction</td><td rowspan=1 colspan=1>The First Hop (Ranked Candidate Chains):1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot)2: (Oswald the Lucky Rabbit, creators, Ub Iwerks and Walt Disney)3: (Oswald the Lucky Rabbit, replacement, Mickey Mouse was created by Walt Disney in 1928) ...The Second Hop (Ranked Candidate Chains):1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),(Donald Duck, creation year, 1934)2: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),(Donald Duck, physical characteristics, anthropomorphic white duck with a yellow-orange bill and legs and feet)3: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot).(Donald Duck, comic book publication, most published comic book character in the world outside of the superhero genre) ...…</td></tr><tr><td rowspan=1 colspan=1>Final Chains</td><td rowspan=1 colspan=1>1: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),(Donald Duck, creation year, 1934)2: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot).(Donald Duck, creation year, 1934), (Donald Duck, creator, Walt Disney Productions), (Donald Duck, comic book publication, mostpublished comic book character in the world outside of the superhero genre)3: (Leo Harris, notable deal, handshake deal with Walt Disney to use Donald Duck as the basis for the Oregon Duck mascot),(Donald Duck, creation year, 1934), (Daisy Duck, creation year, 1940), (Donald Duck, creator, Walt Disney Productions) ...</td></tr><tr><td rowspan=1 colspan=1>Generated Answer</td><td rowspan=1 colspan=1>Donald Duck</td></tr></table>

![](images/7b0d92b9b75aebe21b8e000b664ecdd2cd8a0fed9f4812ee6c670994725aa08a.jpg)  
Figure 13: Content of the assessment framework. For instance, in the triple (Albert Einstein, was thefirst recipient in 1921 of, the Nobel Prize in Physics), the analysis recognizes that while the entities are authentic, the specific relational claim is inaccurate. Integrating this analysis, the evaluator $M _ { \mathbf { e v a } }$ generates the final credibility score.