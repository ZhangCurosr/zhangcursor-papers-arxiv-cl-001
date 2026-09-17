# Relationally Guided Use Case Modeling with LLMs

Guangyu Wang <sup>a,b</sup>, Bangqi Li <sup>a</sup>, Ji Wu <sup>a,c,∗</sup> and Zhijun Shao <sup>a</sup>

<sup>a</sup>School ofComputer Science and Engineering, Beihang University, Beijing, 100191, Beijing, China

<sup>b</sup>Xi’an Aeronautics Computing Technique Research Institute, AVIC, Xi’an, 710065, Shaanxi, China

<sup>c</sup>Engineering Research Center of Integration and Application of Digital Learning Technology, Ministry of Education, China

## A R T I C L E I N F O

Keywords: Use Case Modeling Use Case Flow Construction Large Language Models

## A BS T RA C T

Use case flows are important elements of use case modeling because they support downstream software engineering activities, including requirements analysis, architectural and detailed design, and test case generation. However, constructing them manually is costly and expertise-intensive, while existing automated approaches still struggle to preserve semantic consistency, control-flow logic, data-flow logic, and the intended system boundary, especially when identifying branch points and generating alternative flows. To address this problem, we propose FlowGen for complete use case flow construction. FlowGen uses LLM-based Semantic Information Processing (SIP) to extract semantic elements, constructs a Semantic Relational Graph (SRG) encoded by an enhanced R-GAT for basic flow generation (BFGen), and further supports branch point prediction through BPP and branch-conditioned alternative flow generation through AFGen. Evaluations on 13 public and 7 industrial datasets show that FlowGen consistently outperforms competitive baselines in all three core components. In particular, BFGen improves over the best baseline by 14% in Precision, 7–25% in Recall, 11–30% in F1, and 10–19% in AUC; BPP improves Precision by 30–110%, Recall by 33–91%, and F1 by 32–117%; AFGen improves Precision by 8–23%, F1 by 5–18%, and AUC by 0.6–2.5%. Moreover, we validate the efectiveness of the LLM-based SIP module and the attention preservation factor in BFGen, analyze the impact of requirement completeness on BFGen, and examine how diferent scopes of branch-related context afect AFGen.

## 1. Introduction

Use case modeling is widely used to capture and organize high-level functional requirements as structured interaction scenarios between actors and a target system [1, 2]. Such scenarios support downstream analysis, software design, and testing activities, such as requirements analysis [3], architecture and detailed design [4, 5, 6], and test case generation [7]. In this paper, we focus on automating the construction of use case flows, which constitute the central behavioral part of a use case specification, including the basic flow that describes the main success scenario and the alternative flows that handle exceptional or deviating situations. In particular, we consider branch points, namely the steps at which a base flow, i.e., the basic flow or another alternative flow, should branch into a corresponding alternative flow.

Although engineers can author such flows manually, doing so is time-consuming and depends heavily on domain expertise [8, 9, 10]. Over the past decades, a series of automated methods have been proposed to support use case flow construction [11, 12, 13, 14, 15, 16, 17]. These methods have improved automation to a considerable extent, but their limitations have become more evident as software requirements grow more complex and more domainspecific. In particular, many earlier rule-based approaches rely heavily on handcrafted templates, syntactic patterns, and sentence-level parsing, and are therefore efective only in restricted domains or input styles. More importantly, they often fail to capture cross-sentence semantics, branchtriggering conditions, and inter-step data-flow dependencies. For alternative flow generation, many methods treat the base flow as plain text and derive divergent scenarios through predefined patterns [15, 18], which weakens the explicit connection between a branch point and the corresponding exception-handling flow.

Recent advances in deep learning [19, 16], especially Large Language Models (LLMs) [20, 21], have opened new possibilities for automating use case flow construction. Compared with rule-based methods, LLMs provide stronger natural language processing capabilities and can better exploit domain knowledge through pretraining [22, 23]. Nevertheless, LLM-based generation is still inadequate for this task. First, LLMs may fail to fully capture the contextual dependencies needed for constructing structured software artifacts from requirements, including use case flows, especially when the input is long or structurally complex [24, 25, 26]. Second, they may generate plausible but outof-scope behaviors, thereby violating the system boundary of the target use case. Third, they remain weak at branchpoint reasoning. In particular, deciding whether a given step should trigger an exceptional response, and maintaining semantic and logical continuity between that triggering step and the generated alternative flow, requires conditional and causal reasoning that current LLMs often fail to perform reliably [25, 27]. As a result, generated alternative flows may be misaligned with the base flow from which they originate.

These limitations indicate that automating use case flows is not merely a text generation problem. The core challenge is to preserve the logic encoded in the requirement. Preserving this logic is essential because use case flows guide subsequent requirements analysis, software design, and testing; errors in the generated flows may therefore propagate to these downstream activities. This logic has at least four aspects. First, the generated flow should preserve semantic consistency: actions should remain aligned with the relevant entities, data items, and domain terms described in the requirement. Second, it should preserve control-flow logic: the flow should follow the intended sequencing and diverge at the appropriate branch points. Third, it should preserve data-flow logic: the availability and state of data across steps should constrain what later steps can legitimately do. Fourth, it should preserve the system boundary: generated behavior should remain within the intended scope of the target use case rather than drifting to related but out-of-scope functionality.

To address these challenges, we propose FlowGen, which comprises three modules: Basic Flow Generation (BFGen), Branch Point Prediction (BPP), and Alternative Flow Generation (AFGen). BFGen combines semantic extraction through LLM-based Semantic Information Processing (SIP), Semantic Relational Graph (SRG) construction, and enhanced R-GAT encoding to make semantic elements, flow dependencies, and branch-triggered relations explicit. BPP models branch-triggering control-flow logic in base flows, and AFGen uses branch-local context together with globally encoded use case semantics to maintain continuity between branch points and generated alternative flows, thereby helping the generated flows remain within the intended system boundary. To support branch-point prediction, we also introduce a semi-automatic pipeline that leverages LLM suggestions and expert validation to annotate branch points in public datasets.

We evaluate FlowGen on 13 public datasets and 7 industrial datasets. Overall, FlowGen consistently outperforms competitive baselines across basic flow generation, branch point prediction, and alternative flow generation, with average gains over the best baselines of 38.60% in Precision, 40.07% in Recall for the tasks where Recall improves, 38.92% in F1 score, and 8.16% in AUC. The comparisons against LLM-based baselines and the Sequence Transformer baseline indicate that SRG-based graph conditioning is useful for preserving branch-to-flow alignment and requirement logic. The results further confirm the efectiveness of the SIP module and the attention preservation factor in BFGen, show that BFGen remains robust under incomplete requirements, and demonstrate that using an appropriate branch-context scope contributes to AFGen.

The main contributions of this paper are as follows:

• We extend our previous BFGen [28], which addresses only basic flow generation, to FlowGen for complete use case flow generation by further introducing branch point prediction and alternative flow generation;

• We introduce BPP for identifying branch points in base flows by explicitly modeling branch-triggering control-flow logic;

• We design AFGen, an alternative-flow decoder that integrates branch-local context with globally encoded use case semantics to produce flows coherent with their triggering branches while remaining within the intended system boundary;

• We design a semi-automatic annotation pipeline to extend publicly available datasets from 8 application domains by supplementing branch point annotations to enable BPP.

The paper is structured as follows: Section 2 summarizes and positions our work relative to prior studies. Section 3 presents the framework of our proposed approach FlowGen and its details. Section 4 describes our experimental setup, including datasets, evaluation metrics, research questions, and experimental setting. Section 5 presents our experimental results and analysis. Section 6 discusses potential internal, external, and construct threats to our study. Section 7 concludes the paper and outlines directions for future work.

## 2. Related Work

## 2.1. Use Case Flow Construction

Automatically deriving use case flows from high-level natural-language requirements has long been studied as an important problem in use case modeling. Among existing approaches, rule-based methods combined with NLP techniques are some of the most frequently cited solutions. These approaches rely on syntactic patterns and rules, such as extracting action-object pairs through part-of-speech tagging, to parse requirement sentences, identify primary actions and associated entities, and subsequently construct event flows using predefined templates [16, 12, 7, 13, 29]. To address the deficiency in action and entity extraction caused by natural language ambiguity, researchers have encoded domain knowledge into predefined rules [17, 30, 24]. However, such rules typically lack the necessary generalizability across domains. In contrast, FlowGen leverages LLM-based semantic extraction to identify domain-specific concepts, actions, and entities in a data-driven manner, achieving better crossdomain generalizability.

Several studies have incorporated formal methods, machine learning, or neural models to improve automation [11, 16, 9]. For example, Elrakaiby et al. [11] refine requirements through a calculus-based process guided by refinement operators. Al-Hroob et al. [16] employ NLP tools and neural networks to extract actors and actions from requirements. Ko et al. [9] use verb clustering and external knowledge to detect omitted steps in use case scenarios. Although these methods improve automation for specific subtasks, several of them still require non-trivial manual efort, such as defining refinement operators, validating extracted elements, or preparing external knowledge sources. Most of them also focus on local syntactic evidence or task-specific heuristics. In contrast, our work aims at a more automated pipeline that preserves the logic of use case construction beyond local syntactic evidence, including sequential control-flow logic and data-flow dependencies among actions and objects.

Recently, there has been increasing interest in applying LLMs to requirements engineering [23]. Existing studies mainly confirm the LLM’s potential of requirement specification generation [21], and explore tasks such as requirement elicitation [31], requirement specification mining [32], and high-level requirements modeling [20]. These studies show that LLMs are useful for understanding requirement text, but they also expose persistent limitations such as weak domain grounding and hallucination [30, 24, 33, 22]. More importantly, existing LLM-based studies have not directly addressed structured use case flow construction, where flow continuity and action-object consistency must be preserved explicitly [25, 24].

Beyond generating the main success scenario, complete use case flow construction also requires identifying where the base flow may diverge into exception-handling behavior. To the best of our knowledge, few studies have specifically addressed the task of branch point prediction to derive alternative flows. Some research eforts on event identification in use cases are implicitly relevant to the recognition of branching conditions [14, 34, 35, 36]. Jurkiewicz and Nawrocki [14] proposed an approach for automatically identifying potential exceptional events in the main scenario of a use case. Their method extracts actors, activities, and data objects from the action steps in the basic flow and predicts exceptional events using inference rules inductively derived from existing use cases. Their method heavily relies on manually predefined data object attributes, and its efectiveness is constrained by both the predefined inference rules and the limited coverage of the existing use cases. Williams et al. [34] developed an ontology-driven framework that recommends potential security concerns by inferring likely security threats through ontology reasoning and predefined concern mappings of the actors, actions, and assets, and their relations specified in use case specifications. However, the efectiveness of inferring security threats relies on the manual eforts: annotating ontology instances in use case specifications, defining a security concern taxonomy, and validating the inferred threats. In contrast, FlowGen predicts branch points by capturing semantic information and structural dependency from the use case description and the base flow context, without requiring manually constructed domain-specific knowledge or ontology instance annotations.

Most approaches to alternative flow generation rely on rule-based methods applied to use case descriptions [37, 18, 15]. Ko et al. [15] proposed a pattern-based method to systematically refine use case descriptions by suggesting possible alternative scenarios. To automatically recommend potential branches or exception paths, this method introduces a catalog of use case specification patterns to capture the typical relations between the basic flow and alternative flows. However, its efectiveness is restricted by the manually defined specification patterns. In addition, the provided patterns focus primarily on structural or syntactic information rather than semantics in use case specifications. Unlike pattern-based methods, FlowGen combines

LLM-based semantic extraction with explicit branch point prediction instead of relying on predefined templates. The predicted branch context is then used to guide alternative flow generation, thereby better preserving the semantic and logical continuity between a base flow and its corresponding alternative flow.

## 2.2. Graph Neural Networks in Software Engineering

Graph neural networks (GNNs) have been widely adopted in software engineering because they are efective at modeling structured relations that are dificult to capture with purely sequential text encoders. Prior studies have used GNNs for software modeling [38], model recommendation [39, 40], efort estimation [41], clone detection [42], and fault localization [43]. The common advantage of these methods is that they exploit graph structure to integrate local and global dependencies into learned representations.

Our work builds on this observation but uses GNNs in a diferent way. Rather than modeling code structure or issue relations, we model requirement-to-flow logic. The SRG used in FlowGen represents use case descriptions and flows as a task-specific typed and weighted graph over extracted semantic elements, with relations capturing intra-flow dependencies, description-to-flow grounding, and branch-triggered cross-flow connections. This representation is designed specifically for use case flow generation, where semantic consistency, flow continuity, and exceptionhandling logic are all central.

The efectiveness of GNNs in software engineering tasks depends on the scale and quality of labeled datasets [44]. However, such datasets are scarce in practice, particularly for domain-specific scenarios, due to the high cost of manual annotation. To address this challenge and enable FlowGen, we designed a semi-automatic pipeline to annotate and complete multiple public datasets.

## 3. Our Approach: FlowGen

In this section, we introduce the pipeline of FlowGen, as illustrated in Fig. 1. In Section 3.1, we formally define the elements extracted from requirement descriptions and use case flows, and present the method for constructing an SRG, laying the foundation for FlowGen. In Section 3.2, an LLMequipped Semantic Information Processing module extracts the defined elements and constructs the SRG. This graph is then fed into the BFGen module (Section 3.3) to generate the basic flow. Building upon the SRG and the base flow, potential branch points are predicted (Section 3.4) and used to guide AFGen in generating alternative flows (Section 3.5).

## 3.1. Semantic Relational Graph Construction

In this subsection, we define the semantic elements extracted from a use case and the relations used to construct the SRG. In this work, we adopt the SRG as a task-specific typed and weighted dependency graph to expose the requirement logic required by the downstream modules.

![](images/543262551f3e0ff96d6300c32c4d1878a1dbf25bc693b675c5fd3b17b7b18f11.jpg)

Figure 1: The Overall Process of FlowGen. SIP: Semantic Information Processing; SRG: Semantic Relational Graph; (a): Basic Flow Generation; (b): Branch Point Prediction; (c): Alternative Flow Generation.  
![](images/5a3d6514da05c2df4ac30377abd00c8967ab5b06462bd72b02eea6fff01af270.jpg)  
Figure 2: Illustrative Semantic Relational Graph Fragment. Blue nodes denote core words, orange nodes denote basic flow actions/objects, green nodes denote alternative flow actions/objects, and red borders indicate the branch point. Edges illustrate representative relations, including action/object/core word sequence or transition $\left( E _ { 1 } / E _ { 2 } / E _ { 6 } \right)$ , action-object relation (� ), description-to-flow mapping $\left( E _ { 4 } / E _ { 5 } \right)$ , and branch-triggered cross-flow relations $( E _ { 7 } / E _ { 8 } )$

As illustrated in Fig. 2, the functional requirement description of a use case is usually presented in natural language, where domain-specific terms, actions, and content words are used to describe what the use case performs. Formally, the textual content of a use case description (����) can be represented as �-tuples:

$$
\begin{array} { r } { D e s c = < c _ { 1 } , c _ { 2 } , \ldots , c _ { n } > , c _ { i } \in C } \end{array}\tag{1}
$$

where � denotes the set of core words appearing in the use case description, to specify what the use case does. A use case specification (���) consists of one basic flow �� and zero or more alternative flows ��. Each alternative flow stems from one branch point ��.

$$
U C S = B F \cup \bigcup _ { k = 1 } ^ { n } A F ^ { ( B P _ { k } ) }\tag{2}
$$

The basic flow �� of a use case is presented as a sequence of action steps. Each step typically consists of an action and its associated objects, specifying the stepwise behavior.

$$
B F = [ < a _ { 1 } , o _ { 1 } > , . . . , < a _ { p } , o _ { p } > ] , a _ { i } \in A , o _ { i } \in O , 1 \leq i \leq p\tag{3}
$$

where $p$ is the total number of steps in the basic flow, � is the set of actions, and � is the set of objects involved in actions. A branch point $B P$ refers to a specific action step within a $B F$ or an $A F ,$ where system status or input might have exception(s) that need to be handled in the corresponding $A F .$ . Taking the branch points in the basic flow as an example: $B P ~ = ~ \{ B P _ { 1 } , . . . , B P _ { k } \} , 0 ~ \leq ~ k ~ \leq ~ p .$ Each alternative flow $A F _ { i } ^ { ( B P _ { k } ) }$ consists of $m _ { k }$ action steps and specifies how the exception triggered at $B P _ { k }$ is handled:

$$
\begin{array} { r } { A F _ { i } ^ { ( B P _ { k } ) } = [ < a _ { 1 } , o _ { 1 } > , . . . , < a _ { m _ { k } } , o _ { m _ { k } } > ] , } \\ { a _ { j } \in A , o _ { j } \in O , 1 \leq j \leq m _ { k } } \end{array}\tag{4}
$$

In the graph representation, each step, whether it is a step in the basic flow, a step in the alternative flow, or a branch point, is operationalized through its constituent action and object nodes.

We group the relations in the SRG into three categories: Intra-flow dependency relations. $E _ { 1 } \subseteq A \times A \colon$ Sequential relation of two consecutive actions in a basic flow or an alternative flow; $E _ { 2 } \subseteq O \times O { \mathrm { : } }$ Transitional relation of two data objects accessible in a common data flow; $E _ { 3 } \subseteq A \times O { \mathrm { : } }$ The action-object relation between an action and an object, meaning the action accesses the object and applies to the action steps in both basic flow and alternative flows. These relations capture the local logic of control flow and data dependency, together with the action-object semantic consistency within a flow.

Description-to-flow grounding relations. $E _ { 4 } \subseteq C \times A$ & $E _ { 5 } \subseteq C \times O { \mathrm { : } }$ The mapping relation between a core word in the functional description of a use case and an action or object in the corresponding basic flow and alternative flows; $E _ { 6 } \subseteq C \times C \colon$ : Sequential relation of two core words in the functional description of a use case. $E _ { 4 } , E _ { 5 } ,$ and $E _ { 6 }$ connect the requirement description to the flow representation by linking core words to actions and objects, and by preserving the sequential context among core words in the description. These relations ground generated flows in the semantics of the original requirement and help preserve the system boundary of the use case.

Branch-triggered cross-flow relations.

$E _ { 7 } \subseteq A _ { B S F } \times A _ { A F } \colon$ The triggering relation between the branch point and the first action of the corresponding alternative flow. $E _ { 8 } \subseteq O _ { B S F } \times O _ { A F } \colon$ The triggering relation between the object in the base flow and the object in the corresponding alternative flow. $E _ { 7 }$ and $E _ { 8 }$ are introduced to explicitly model the logic of control flow and data dependency between the triggering step and the exceptionhandling flow derived from it.

In practical datasets, some action sequences, object transitions, and cross-flow correspondences occur repeatedly within a domain and reflect stable domain conventions. For example, in cloud service applications, the "activate service" action typically follows the "deactivate service" action, and both frequently appear in related use cases. Similarly, in applications with security concerns, such as banking or e-commerce, the "authentication" action almost invariably triggers the "grant access" action. Both actions consistently work on fixed objects—such as user credentials or session tokens—and these objects appear more frequently than others in these applications. We therefore associate each relation with a frequency-based weight. The weight does not claim that the relation is universally more correct; instead, it serves as a salience prior indicating how prominent that relation is in the observed requirements and flows of the domain [45, 46]. In the definition $w ( e ) ~ = ~ k \cdot f ( e )$ , the constant � controls scaling, while the relative importance among relations is determined by their observed frequencies �(�).

Based on the extracted nodes, typed relations, and relation weights, the use case can be specified as an SRG $G = ( V , E , W )$ , where $V = C \cup A \cup O$ denotes the set of core word, action, and object nodes, � is the set of typed edges constructed from the relations among these nodes, and � contains the corresponding weights. This graph serves as the structured representation on which the subsequent modules perform logic-aware generation and prediction.

## 3.2. Semantic Information Processing

In this subsection, we present the Semantic Information Processing (SIP) module, designed to extract the defined semantic information from requirement descriptions and use case flows. SIP has three sequential tasks: Sentence Simplification, Sentence Splitting, and Semantic Information Extraction.

Software requirements in natural language often include modifiers and compound sentences to describe the scenarios or behaviors in which the software system interacts with its actors. This significantly increases the dificulty of extracting semantic information from requirements using NLP tools or LLMs [47]. To improve the efectiveness of semantic information extraction, FlowGen first simplifies sentences and then splits compound sentences into simple ones. The Sentence Simplification task removes unnecessary modifiers from a sentence while preserving its core words. The Sentence Splitting task splits a compound sentence into simple ones without changing the sequence of verbs and objects. Subsequently, FlowGen extracts semantic information: (1) core words from the use case description, such as domain terms, actions, and other content words; (2) actions and objects from action steps in use case flows.

To improve the accuracy and eficiency of extraction, we use LLMs with strong natural language processing capabilities [48, 49]. Building upon existing structured prompt frameworks [24, 21], we modularize the prompt into three components, forming a triplet $\mathcal { P } = ( \mathcal { R } , \mathcal { T } , \tau )$

![](images/c1ace158c874a191a086d5c16af701d0e8ca13ff12f6c0737e147d0fcb04d706.jpg)  
Figure 3: An Example of The Prompt of Semantic Information Processing

Role Description (): This component assigns a role to the LLM, guiding it to apply task-specific knowledge during processing to align with the task domain.

Task Description ( ): This component provides a comprehensive task description and output schema to ensure the results can be processed consistently in downstream tasks.

Inputs (): This component specifies the concrete inputs of the task. The input of the Sentence Simplification task is a requirement description or use case flow, which is then fed to the Sentence Splitting task. The Semantic Information Extraction task takes the result of the splitting task as input.

Fig. 3 demonstrates the execution of tasks using an action step statement from a basic flow in the eANCI [50] dataset, where the original sentence is highlighted in yellow and the LLM outputs in red. In Task 1, the LLM removes adverbs such as "successfully" in the input as they are deemed non-essential to the core meaning of the step. In Task 2, the LLM splits the compound sentence into three simple ones based on the action execution order derived from semantic analysis. In Task 3, the simple sentences from Task 2 are processed to extract actions and objects, resulting in (action, object) pairs such as “accesses” (action) and “the Civil Defense feature” (object). Another scenario in Task 3 involves extracting core words from use case descriptions. This requires adding a domain-specific term extraction instruction to Task Description  while keeping other components unchanged. As shown in Fig. 2, the term "Administrator account" is identified alongside other core words (e.g., "Login") via the extraction task.

With the results of the Semantic Information Extraction task, FlowGen establishes relations among the extracted semantic elements and assigns weights based on the relation definition in Section 3.1. Specifically, $E _ { 1 }$ and $E _ { 2 }$ capture the sequences of actions and objects respectively within the action steps of the use case flow; $E _ { 3 }$ captures actions and the accessed objects within action steps; $E _ { 4 }$ and $E _ { 5 }$ capture the mapping between the use case description and the corresponding use case flow, linking each core word to each action and object; $E _ { 6 }$ captures the sequence of core words within the use case descriptions; $E _ { 7 }$ and $E _ { 8 }$ capture the mapping relations between the base flow and the alternative flows at branch points. Relation weights follow the frequency-based salience prior, i.e., $w ( e ) = k \cdot f ( e )$ with � = 1 by default.

## 3.3. Basic Flow Generation (BFGen)

After constructing the SRG and extracting semantic information through the SIP module, FlowGen generates the basic flow using the BFGen module, which serves as the structural backbone for the entire use case flows. BFGen is designed to capture the contextual dependencies among actions and objects, and to predict a coherent sequence of action steps that specifies the main scenario of a use case. BFGen leverages an enhanced Relational Graph Attention Network [51] to encode the SRG. By integrating neighborhood information, relation types and weights into the attention mechanism through a multi-layer structure, BFGen enables each node embedding to not only represent its local semantics but also capture long-range dependencies of diferent types and weights within a use case context. We introduce an attention preservation factor that explicitly regulates the balance between attention-based embedding aggregation and the uniform embedding aggregation. This factor is incorporated into the message-passing mechanism of R-GAT, yielding the following enhanced propagation rule:

$$
h _ { i } ^ { l + 1 } = \sigma \left( \sum _ { j \in \mathcal { N } ( i ) } \left( \lambda \alpha _ { i j } + \left( 1 - \lambda \right) \right) W ^ { l } h _ { j } ^ { l } \right)\tag{5}
$$

where $h _ { i } ^ { l + 1 }$ is the updated embedding of node � at layer � + 1, obtained by aggregating the transformed embeddings from its neighboring nodes in <sup></sup>(�) at the �-th layer. $W ^ { l }$ is a trainable parameter matrix, and �(⋅) is a nonlinear activation function. $\lambda \in [ 0 , 1 ]$ is the attention preservation factor. By dynamically adjusting the impact of �, the model can flexibly capture important relations while preventing overfitting to noise or weak associations. The attention coeficient $\alpha _ { i j }$ for the edge from node � to � is computed as shown in Eq. (6):

$$
\alpha _ { i j } = \frac { \exp { \left( \sigma \left( r _ { i j } \mathbf { a } ^ { \top } \left[ W ^ { l } h _ { i } ^ { l } \| W ^ { l } h _ { j } ^ { l } \right] \right) \right) } } { \sum _ { k \in \mathcal { N } ( i ) } \exp { \left( \sigma \left( r _ { i k } \mathbf { a } ^ { \top } \left[ W ^ { l } h _ { i } ^ { l } \| W ^ { l } h _ { k } ^ { l } \right] \right) \right) } } + w _ { i j }\tag{6}
$$

where $r _ { i j }$ and $r _ { i k }$ denote the edge representation associated with edges $( i , j )$ and (�, �), � is a learnable parameter vector, and $w _ { i j }$ denotes the edge weight.

BFGen employs scoring functions, as shown in Eq. (7) and Eq. (8), combined with binary cross entropy (BCE) loss [52] to predict nodes and train the model. For each input, i.e., use case description, BFGen computes the probability across all action/object nodes as the prediction relevance score and outputs the set of positive nodes whose scores exceed the predefined threshold �. Specifically, let $C _ { D e s c }$ denote the associated core word nodes that occur in a functional description ����, and let $V = A \cup O$ denote the union of action and object nodes from basic flows in the training set. For each core word node $c \in C _ { D e s c }$ and each action/object node $v \in V$ , given that R-GAT has a total of � layers, the Hadamard product of the final-layer representations, $\mathbf { h } _ { c } ^ { L }$ and $\mathbf { h } _ { v } ^ { L }$ , is used with sigmoid activation to generate a relevance score:

$$
s _ { c , v } = \mathrm { s i g m o i d } ( \mathbf { h } _ { c } ^ { L } \odot \mathbf { h } _ { v } ^ { L } )\tag{7}
$$

The final score $s _ { v } ^ { ( D e s c ) }$ for each action/object node � of ���� in the basic flow is obtained by averaging over all $s _ { c , v } .$

$$
s _ { v } ^ { ( D e s c ) } = \frac { 1 } { | C _ { D e s c } | } \sum _ { c \in C _ { D e s c } } s _ { c , v } .\tag{8}
$$

The embeddings $H _ { B F } = [ h _ { 1 } ^ { B F } , h _ { 2 } ^ { B F } , . . . , h _ { p } ^ { B F } ]$ represent the semantics of the most contextually and domain-aligned set of actions and objects, collectively forming a sequence of action steps that constitute the corresponding basic flow for a given use case description.

## 3.4. Branch Point Prediction (BPP)

Following the generation of the basic flow, FlowGen proceeds to generate alternative flows to handle potential exceptions. As a crucial prerequisite, it first identifies potential branch points, i.e., specific action steps within a use case flow where exceptions may trigger conditional or exceptional alternative flows. Although a branch point is defined conceptually at the action-step level, each step is represented in the SRG through its constituent action and object nodes. Accordingly, BPP performs node-level learning. We design an encoder–decoder architecture, where the encoder leverages the enhanced R-GAT introduced in Section 3.3 to generate contextual embeddings, and the decoder, implemented as the BPP module, assesses the likelihood of each step being a branch point. Specifically, BPP adopts an attention-based relational reasoning mechanism, where each flow node (action/object) attends to neighboring nodes in the graph to infer its importance. Given the encoded node embeddings $H \ = \ [ h _ { 1 } , h _ { 2 } , . . . , h _ { n } ]$ produced by the encoder, and the embeddings of the base flow nodes $\bar { H } _ { B S F } \ = \ [ h _ { 1 } ^ { B S F } , h _ { \gamma } ^ { B S F } , . . . , h _ { n } ^ { B S F } ]$ , the probability of each node embedding is computed as:

$$
\begin{array} { r } { P ( B P _ { i } ) = \sigma \big ( W _ { p } [ h _ { i } ^ { B S F } \| \mathrm { A t t n } ( h _ { i } ^ { B S F } , H ) ] + b _ { p } \big ) } \end{array}\tag{9}
$$

where $\mathrm { A t t n } ( h _ { i } ^ { B S F }$ , �) denotes the attention-weighted contextual embedding aggregated from all nodes in �, � is the sigmoid activation; $W _ { p }$ and $b _ { p }$ are trainable parameters. The model is trained with BCE loss:

$$
\mathcal { L } _ { B P } = - \frac { 1 } { p } \sum _ { i = 1 } ^ { p } [ y _ { i } \log P ( B P _ { i } ) + ( 1 - y _ { i } ) \log ( 1 - P ( B P _ { i } ) ) ] ,\tag{10}
$$

where $y _ { i } ~ \in ~ \{ 0 , 1 \}$ indicates whether the node $v _ { i } ^ { B S F }$ is a branch point. $v _ { i } ^ { B S F }$ is recognized as a branch point when $P ( B P _ { i } ) \ge \tau ,$ , where � is an empirical threshold. The predicted branch points $B P = \{ B P _ { t } \}$ and their contextual embeddings are then provided as input to the Alternative Flow Generation (AFGen) module (Section 3.5), which constructs the sequence of action steps to handle the potential exceptions (as responses) branching from the identified branch points.

## 3.5. Alternative Flow Generation (AFGen)

Based on the predicted branch points, the base flow in which they are located, and the use case description, AF-Gen generates alternative flows using an encoder–decoder architecture. Consistent with BPP, the encoder reuses the contextual embeddings produced by the enhanced R-GAT in Section 3.3, ensuring all input elements reside in a shared semantic space. The decoder, distinct from that of BPP, comprises two key components: (1) Context Conditioning, which fuses the context around the branch point with the semantics of the first node in the alternative flow into a unified initial representation, so as to anchor how the divergence starts from the identified branch. (2) Attention-based Decoding, which incrementally and iteratively constructs the step sequence of the alternative flow by attending to the initial representation, along with the local context and global use case semantics.

Context Conditioning The decoder performs autoregressive generation, where each action step in the alternative flow is generated sequentially based on the previously generated steps and the initial hidden state. The initial hidden state of the decoder $h _ { z _ { 0 } }$ encodes contextual information and initializes the autoregressive decoding process by combining two components: $\hat { h } _ { C _ { i } }$ that captures the local semantics around the branch point, and the first node (action/object) embedding in the alternative flow, denoted $h _ { A F _ { 0 } }$ . During training, the first node of the reference alternative flow is used as a start-node signal to reduce the semantic gap between the identified branch point and the beginning of the target alternative flow. $\hat { h } _ { C _ { i } }$ is computed by aggregating node representations in the local neighborhood of the branch point. Specifically, for a given branch point $B P _ { i } .$ , contextual information is derived from its �-hop neighboring flow nodes, where the parameter � controls the scope of the neighborhood.

$$
\hat { h } _ { C _ { i } } = \frac { 1 } { \left| \mathcal { N } _ { k } ( B P _ { i } ) \right| } \sum _ { v \in \mathcal { N } _ { k } ( B P _ { i } ) } h _ { v }\tag{11}
$$

where $\mathcal { N } _ { k } ( B P _ { i } )$ denotes the set of nodes reachable from $B P _ { i }$ within � hops (including $B P _ { i }$ itself), and $h _ { v }$ is the embedding of node $v$ produced by the enhanced R-GAT encoder.

$h _ { z _ { 0 } }$ is computed by concatenating $\hat { h } _ { C _ { i } }$ with the first node of the alternative flow $h _ { A F _ { 0 } }$ at step $t ~ \equiv ~ 0$ , allowing the decoder to maintain access to the context encoded in the enhanced R-GAT while inheriting the local semantics of the branch.

$$
h _ { z _ { 0 } } = \mathrm { R e L U } \left( W _ { h } \left[ \hat { h } _ { C _ { i } } \lVert h _ { A F _ { 0 } } \right] \right) ,\tag{12}
$$

where $W _ { h }$ is a learnable transformation matrix and [||] denotes vector concatenation.

Attention-based Decoding At each decoding step �, the decoder generates the next node, attending to the entire context encoded by the enhanced R-GAT. The attention weight $\alpha _ { t }$ over encoder representations $H = [ h _ { 1 } , . . . , h _ { n } ]$ is computed as:

$$
\begin{array} { r l } & { \alpha _ { t } = \mathrm { s o f t m a x } \left( w ^ { \top } \operatorname { t a n h } \left( W _ { a } \left[ h _ { z _ { t - 1 } } \| H \right] \right) \right) , } \\ & { c _ { t } = \displaystyle \sum _ { j = 1 } ^ { n } \alpha _ { t } ^ { ( j ) } h _ { j } , \quad h _ { z _ { t } } = \mathrm { G R U } \left( \left[ x _ { t - 1 } \| c _ { t } \right] , h _ { z _ { t - 1 } } \right) , } \end{array}\tag{13}
$$

where $w$ is a trainable parameter vector used to compute attention weights over the encoder states, $x _ { t - 1 }$ is the embedding of the previous output token and $c _ { t }$ is the attentionweighted context vector. Starting from the initial state $h _ { z _ { 0 } } ,$ the decoder recurrently updates its hidden state $h _ { z _ { t } }$ as it processes each generated step in the alternative flow sequence.

The output probability distribution for the next token $y _ { t }$ is computed by projecting the hidden state $h _ { z _ { t } }$ onto the candidate set of actions/objects:

$$
p ( y _ { t } \mid y _ { < t } , A F _ { i } ) = \mathrm { s o f t m a x } ( W _ { o } h _ { z _ { t } } )\tag{14}
$$

where $W _ { o }$ is a learnable weight matrix that maps the contextaware hidden state $h _ { z _ { t } }$ to logits over the output candidate set. The softmax function normalizes the resulting scores to yield a valid probability distribution over potential next actions or objects, conditioned on the generation history $y _ { < t }$ and the specific alternative flow $A F _ { i }$ context. AFGen is trained to minimize the negative log-likelihood of the target sequence, formulated as:

$$
\mathcal { L } _ { \mathrm { d e c o d e r } } = - \sum _ { t = 1 } ^ { T } \log p ( y _ { t } ^ { * } \mid y _ { < t } , A F _ { i } ) ,\tag{15}
$$

where $y _ { t } ^ { * }$ denotes the reference token at step �. The encoderdecoder architecture and the above computational mechanism enable AFGen to integrate local branching cues and long-range dependencies within the SRG, ensuring that the generated alternative flows are accurate and semantically consistent with both the requirement and the corresponding base flow.

## 4. Experiments

In this section, we detail the experimental design and setup used to evaluate the efectiveness of FlowGen. We first introduce the collected public and industrial datasets, and the branch point annotation in Section 4.1. Subsequently, we introduce the specific evaluation metrics for generation and prediction tasks in Section 4.2. Finally, we outline the seven research questions (RQs) that guide our investigation in Section 4.3, and specify the experimental environment and parameter settings in Section 4.4.

## 4.1. Datasets

We construct a comprehensive dataset composed of 13 public datasets and 7 proprietary industrial datasets provided by Huawei. The public datasets originate from diverse software domains—including healthcare (iTrust[53]), governance (eANCI[50]), supply chain management (viper[54]), education (SMOS[50]), and others—exhibiting high syntactic complexity, such as frequent use of subordinate clauses, nested conditions, and intricate sentence structures. The requirements in industrial datasets, written in Simplified Chinese with simple and concise sentence structures, are collected from the Network Cloud Engine-Transport (NCE-T) product. This product has been deployed over the past decade in network and service management, disaster recovery, automated operations and maintenance, and optical networking. Table 1 summarizes the key statistics of these datasets. In principle, the number of alternative flows should correspond to the number of branch points—one for each alternative flow. However, among 9 of the 13 public datasets that contain alternative flows, only one provides branch point annotations. Therefore, the implicit branch points in the other 8 datasets need to be annotated.

## 4.1.1. Branch Point Annotation

Accurate generation of alternative flows relies on precise branch point annotations. To annotate missing branch points in 8 public datasets, we design a semi-automated pipeline that integrates LLM-based inference with expert validation. For every use case that contains alternative flows without explicit branch point steps, we recover the branch points using the procedure illustrated in Fig. 4.

LLM-based Branch Point Inference. Inputs are the use case description, the base flow, and the corresponding alternative flow. The LLM is prompted to identify and output the most plausible step index in the base flow where the corresponding alternative flow diverges, i.e., its branch point.

Independent Expert Verification. The LLM-inferred branch point suggestions are independently reviewed by five software engineers, with industrial experience in requirements analysis and development ranging from six to ten years. They evaluate if the predicted branch points accurately correspond to the branching semantics indicated by the alternative flow. If an engineer deems the LLM’s prediction inaccurate, they are required to propose the appropriate branch point(s).

Final Annotation Determination. For each alternative flow, the LLM-predicted branch point is retained if approved by a majority of engineers. Otherwise, the final branch point is determined through consensus among the engineers candidate proposals.

## Table 2

Table 1  
Overview of Public and Industrial Datasets  
Note: BF = Basic Flow; AF = Alternative Flow; Sys. = System; O&M = Operations & Maintenance.
<table><tr><td></td><td>Dataset Name</td><td>Domain</td><td># of Use Cases</td><td># of BF Steps</td><td># of BF Unique Steps</td><td># of AF</td><td># of AF Steps</td><td># of AF Unique Steps</td></tr><tr><td rowspan="14">Public Datasets</td><td>EasyClinic</td><td>Laboratory Management</td><td>30</td><td>210</td><td>187</td><td>58</td><td>200</td><td>113</td></tr><tr><td>hats</td><td>GUI for Program Transformation</td><td>28</td><td>183</td><td>162</td><td>45</td><td>163</td><td>117</td></tr><tr><td>eANCI</td><td>Governance</td><td>139</td><td>602</td><td>480</td><td>0</td><td>0</td><td>0</td></tr><tr><td>keepass</td><td>Password Management</td><td>14</td><td>66</td><td>56</td><td>38</td><td>81</td><td>61</td></tr><tr><td>eTour</td><td>Tourism</td><td>58</td><td>272</td><td>230</td><td>0</td><td>0</td><td>0</td></tr><tr><td>viper</td><td>Supply Chain Management</td><td>39</td><td>105</td><td>92</td><td>18</td><td>23</td><td>18</td></tr><tr><td>Trust</td><td>Healthcare</td><td>34</td><td>356</td><td>346</td><td>43</td><td>57</td><td>50</td></tr><tr><td>gamma j</td><td>Web Store</td><td>26</td><td>132</td><td>104</td><td>14</td><td>44</td><td>25</td></tr><tr><td>inventory</td><td>Inventory Management Sys.</td><td>10</td><td>35</td><td>33</td><td>12</td><td>34</td><td>24</td></tr><tr><td>inventory 2.0</td><td>Inventory Management Sys.</td><td>21</td><td>191</td><td>184</td><td>21</td><td>48</td><td>35</td></tr><tr><td>SMOS</td><td>Education</td><td>67</td><td>214</td><td>197</td><td>0</td><td>0</td><td>0</td></tr><tr><td>model manager</td><td>Research Task Management</td><td>7</td><td>81</td><td>81</td><td>0</td><td>0</td><td>0</td></tr><tr><td>pnnl 01</td><td>Diagnostic</td><td>5</td><td>45</td><td>45</td><td>9</td><td>34</td><td>33</td></tr><tr><td></td><td>Sys. Platform Disaster Recovery</td><td>111</td><td>1142</td><td>408</td><td>319</td><td>638</td><td>155</td></tr><tr><td rowspan="7">Industrial Datasets</td><td>02</td><td>&amp; Backup</td><td>5</td><td>58</td><td>15</td><td>35</td><td>70</td><td>8</td></tr><tr><td>03</td><td>Network Device</td><td>319</td><td>2549</td><td>826</td><td>564</td><td>1128</td><td>266</td></tr><tr><td>04</td><td>Management Network Service</td><td>4</td><td>36</td><td>25</td><td>16</td><td>32</td><td>13</td></tr><tr><td>05</td><td>Management Intelligent Optical Network</td><td>5701</td><td>31173</td><td>6145</td><td>14543</td><td>29086</td><td>2649</td></tr><tr><td>06</td><td>Automated O&amp;M</td><td>235</td><td>235</td><td>223</td><td>1506</td><td>3012</td><td>373</td></tr><tr><td>07</td><td>Optical Virtual</td><td>131</td><td>131</td><td>124</td><td>584</td><td>1168</td><td>337</td></tr><tr><td></td><td>Private Network</td><td>6984</td><td>37816</td><td>9963</td><td>17825</td><td>35818</td><td>4277</td></tr></table>

Annotation Reliability Analysis. To assess the quality and reliability of the branch point annotations, we report the inter-annotator agreement (the degree of consensus among annotators) and LLM–human alignment on the annotations. First, we compute Fleiss’ � [55] over all annotated branch points to quantify the level of agreement among the five experts. This provides a standard measure of annotation consistency. Second, we measure the alignment between the engineers’ final annotations and the LLM’s predictions. Notably, a Fleiss’ � value of 1 indicates identical annotations from all five experts, while an LLM–Human alignment value of 1 indicates that the LLM’s prediction exactly matches the experts’ final agreed branch points.

Table 2 summarizes the number of alternative flows without branch points in each dataset (i.e., the number of branch points to be annotated), the corresponding Fleiss � among the five expert annotators, and the proportion of LLM-suggested branch points matching the final results. As shown in Table 2, the average Fleiss’ � of 0.778 indicates substantial inter-annotator agreement, validating the reliability of manual annotations. The average LLM–human alignment score of 0.888 demonstrates the efectiveness of LLM-based branch point prediction. Together, these results demonstrate that the proposed semi-automated annotation pipeline produces highly accurate branch point annotations. The final branch-point annotations and the corresponding extended dataset files are released as part of the replication package.

Annotation Reliability and LLM Alignment for Branch Points.
<table><tr><td>Dataset</td><td>#AF Missing BP</td><td> $\mathsf { F l e i s s ^ { \prime } } \kappa$ </td><td>LLM-Human Alignment</td></tr><tr><td>keepass</td><td>37</td><td>0.677</td><td>0.865</td></tr><tr><td>gamma j</td><td>14</td><td>1.000</td><td>0.929</td></tr><tr><td>inventory</td><td>12</td><td>0.567</td><td>0.917</td></tr><tr><td>hats</td><td>1</td><td>1.000</td><td>1.000</td></tr><tr><td>pnnl</td><td>9</td><td>1.000</td><td>1.000</td></tr><tr><td>viper</td><td>18</td><td>0.691</td><td>0.889</td></tr><tr><td>inventory 2.0</td><td>21</td><td>0.561</td><td>0.905</td></tr><tr><td>Trust</td><td>5</td><td>0.724</td><td>0.6</td></tr><tr><td>Overall(Mean)</td><td>117</td><td>0.778</td><td>0.888</td></tr></table>

## 4.1.2. Dataset Statistics

Since FlowGen relies on the extracted core words, actions, and objects to construct semantic relations and supervise the enhanced R-GAT model, its performance is highly sensitive to the richness of those elements present in the datasets. As shown in Table 3, we compare the number of nodes extracted by the mature NLP toolkit, Stanford CoreNLP, and two representative LLMs, ERNIE 4.0 Turbo and GPT-4o. The numbers of extracted nodes from industrial datasets (3,906 / 14,594 / 11,603) are larger than those from public datasets (3,482 / 3,894 / 4,019). This indicates that industrial datasets may contain richer domain-specific content, such as specialized terms, data objects, and actions. Moreover, the two LLMs can extract a greater variety and quantity of elements than the traditional NLP tool—14,594 and 11,603 compared to 3,906 in the industrial datasets— justifying the motivation for employing LLMs in FlowGen. This richer set of extracted elements further enhances the construction of the SRG, particularly the representation effectiveness of the R-GAT encoder. The number of edges (i.e., relations between nodes) is not reported separately, since the edge count is inherently dependent on the number of nodes. Specifically, the counts for $E _ { 1 } – E _ { 2 } , \ E _ { 4 } – E _ { 8 }$ are directly related to the number of nodes extracted, while the count for �3 is derived from the number of action steps in a use case flow.

Table 3  
Node Statistics for Public and Industrial Datasets
<table><tr><td rowspan="2"></td><td rowspan="2">Tool</td><td colspan="2">Basic Flow</td><td colspan="2">Alternative Flow</td><td>Use Case Description</td><td rowspan="2">Sum</td></tr><tr><td># of Actions</td><td># of Objects</td><td># of Actions</td><td># of Object</td><td># of Core words</td></tr><tr><td rowspan="3">Public Datasets</td><td>Stanford CoreNLP</td><td>684</td><td>1277</td><td>217</td><td>181</td><td>1123</td><td>3482</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>908</td><td>1387</td><td>182</td><td>156</td><td>1261</td><td>3894</td></tr><tr><td>GPT-4o</td><td>786</td><td>1464</td><td>190</td><td>268</td><td>1311</td><td>4019</td></tr><tr><td rowspan="2">Industrial</td><td>Stanford CoreNLP</td><td>851</td><td>1070</td><td>349</td><td>331</td><td>1305</td><td>3906</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>778</td><td>6429</td><td>179</td><td>1943</td><td>5265</td><td>14594</td></tr><tr><td>Datasets</td><td>GPT-4o</td><td>710</td><td>4535</td><td>293</td><td>921</td><td>5144</td><td>11603</td></tr></table>

![](images/78923565f08a5352b62037eba6efb80ff14680b67f12eb4d4081a4c37b4ded64.jpg)  
Figure 4: Semi-automated Branch Point Annotation

## 4.2. Evaluation Metrics

To quantitatively evaluate the efectiveness of FlowGen, we employ multiple metrics to evaluate its three modules— BFGen, BPP, and AFGen separately.

## 4.2.1. Metricsfor BFGen and AFGen

BFGen and AFGen generate use case flows from given requirement descriptions. We employ the four well-known metrics: ���������, ������, �1 �����, and ���� ����� ����� (���) to evaluate the efectiveness and performance of the two modules, based on the matches between the generated use case flows and the corresponding reference flows.

$$
\begin{array} { r } { P r e c i s i o n = \displaystyle \frac { | R \cap G | } { | G | } , R e c a l l = \displaystyle \frac { | R \cap G | } { | R | } , } \\ { F 1 = \displaystyle \frac { 2 \times P r e c i s i o n \times R e c a l l } { P r e c i s i o n + R e c a l l } } \end{array}\tag{16}
$$

where � denotes the set of nodes generated by BFGen or AFGen, and � denotes the set of nodes in the corresponding reference flow. ��������� measures the accuracy, ������ measures the completeness, and �1 measures the overall balance of the generation.

Literal requirement descriptions are the common input of BFGen and AFGen. Due to the inherent ambiguity of natural language, similar words might have very diferent meanings in diferent domain contexts. To quantitatively assess how well the modules diferentiate near-synonymous terms and detect domain term mismatches in diferent domains, we introduce ��� as an evaluation metric, calculated as follows:

$$
\begin{array} { l } { \displaystyle { A U C = \frac { 1 } { | P | \cdot | N | } \sum _ { i \in P } \sum _ { j \in N } \Big [ \mathbb { I } \big ( p ( i ) > p ( j ) \big ) } } \\ { \displaystyle { \qquad + 0 . 5 \times \mathbb { I } \big ( p ( i ) = p ( j ) \big ) \Big ] } } \end{array}\tag{17}
$$

where � and � denote positive (i.e., matched in the reference flow) and negative (i.e., not matched) nodes, respectively. �(�) is the predicted probability that node � belongs to the generated flow. �(⋅) is the indicator function. If node � is present in the outputs of BFGen or AFGen, �(�) will be 1.0, otherwise 0.

## 4.2.2. Metricsfor Branch Point Prediction (BPP)

Branch points might occur in any step of a base flow. Normally, branch points are sparsely and unevenly distributed in use case flows. A use case flow may have zero to the maximal number of action steps as branch points. A single global metric is insuficient for evaluating the efectiveness of branch point prediction. Therefore, we employ �����- averaged ���������, ������, and �1 ����� to assess overall step-level prediction performance, and �����- averaged ���������, ������, and �1 ����� to ensure fair evaluation across individual base flows, particularly those that are short or contain few branch points. For each use case �� ∈ ��:

$$
P r e c i s i o n _ { u c } = \frac { \left| B P _ { u c } ^ { \mathrm { p r e d } } \cap B P _ { u c } ^ { \mathrm { r e f } } \right| } { \left| B P _ { u c } ^ { \mathrm { p r e d } } \right| } ,
$$

$$
R e c a l l _ { u c } = \frac { \left| B P _ { u c } ^ { \mathrm { p r e d } } \cap B P _ { u c } ^ { \mathrm { r e f } } \right| } { \left| B P _ { u c } ^ { \mathrm { r e f } } \right| } ,\tag{18}
$$

$$
F 1 _ { u c } = \frac { 2 \times P r e c i s i o n _ { u c } \times R e c a l l _ { u c } } { P r e c i s i o n _ { u c } + R e c a l l _ { u c } }
$$

where $B P _ { u c } ^ { \mathrm { p r e d } }$ and $B P _ { u c } ^ { \mathrm { r e f } }$ denote the predicted and reference branch point set for use case �� respectively. The macroaveraged and micro-averaged metrics are computed as follows, where |��| is the number of use cases.

$$
\begin{array} { r } { P r e c i s i o n _ { m a c r o } = \frac { 1 } { | U C | } \displaystyle \sum _ { u c \in U C } P r e c i s i o n _ { u c } , } \\ { R e c a l l _ { m a c r o } = \frac { 1 } { | U C | } \displaystyle \sum _ { u c \in U C } R e c a l l _ { u c } , } \\ { F 1 _ { m a c r o } = \frac { 1 } { | U C | } \displaystyle \sum _ { u c \in U C } F 1 _ { u c } } \end{array}\tag{19}
$$

$$
P r e c i s i o n _ { m i c r o } = \frac { \sum _ { u c \in U C } \left| B P _ { u c } ^ { \mathrm { p r e d } } \cap B P _ { u c } ^ { \mathrm { r e f } } \right| } { \sum _ { u c \in U C } \left| B P _ { u c } ^ { \mathrm { p r e d } } \right| } ,
$$

$$
R e c a l l _ { m i c r o } = \frac { \sum _ { u c \in U C } \left| B P _ { u c } ^ { \mathrm { p r e d } } \cap B P _ { u c } ^ { \mathrm { r e f } } \right| } { \sum _ { u c \in U C } \left| B P _ { u c } ^ { \mathrm { r e f } } \right| } ,\tag{20}
$$

$$
F 1 _ { m i c r o } = \frac { 2 \times P r e c i s i o n _ { m i c r o } \times R e c a l l _ { m i c r o } } { P r e c i s i o n _ { m i c r o } + R e c a l l _ { m i c r o } }
$$

## 4.3. Research Questions (RQs) & Baseline Setup

RQ1: How efective is BFGen in generating basic flows compared to the baseline methods?

The goal of RQ1 is to investigate whether the basic flows generated by BFGen are of high quality and whether it has advantages over the baselines. To answer this question, we design a comparative experiment to compare the performance of BFGen against the four baseline methods that lie in three categories: rule-based method, LLM-based method, and GNN-based method. We select three representative rule-based methods [12, 15, 13]. Given that some rules in these methods are tailored for specific input formats, we consolidate the broadly applicable rules into a unified framework. Additionally, we replace the earlier NLP tools used in these methods for parsing requirement descriptions with the more mature Stanford CoreNLP[56] to feed those methods with consistent inputs. LLM-based methods are currently considered the most prominent ones. Since our datasets are specified in English and Simplified Chinese, we select two established models—GPT-4o from OpenAI [48] and ERNIE 4.0 Turbo from Baidu [49]—to minimize the impact of linguistic diferences on the LLMs’ performance. Since BFGen uses the R-GAT model, we select the GNNbased RGAT-with-BERT method [51], which is related to BFGen and has shown strong performance, as a baseline. This method utilizes pre-trained BERT to generate hybrid representations and employs the R-GAT model to encode syntactic dependency graphs and learn syntactic embeddings.

RQ2: How efective is the SIP module in contributing to the overall performance of BFGen?

To ensure the enhanced R-GAT can precisely model the various nodes and relations, we design the SIP module to extract the semantic information in terms of core words, actions, and objects, and their relations. RQ2 aims to evaluate the validity of our approach by investigating whether the SIP module contributes positively to the overall efectiveness of BFGen.

We design an ablation study to evaluate the impact of SIP on BFGen, and set BFGen without SIP (BFGen w/o SIP) as the first baseline for comparison with the full BFGen. We further include a second baseline, BFGen without preprocessing (BFGen w/o Preproc.), to evaluate the contribution of the two preprocessing tasks: Sentence Simplification and Splitting. It should be noted that we select Stanford CoreNLP as the text parsing tool for providing semantic information in BFGen w/o SIP. Since BFGen w/o Preproc. relies on LLMs to extract semantic information, we evaluate it with two distinct models—GPT-4o and ERNIE 4.0 Turbo—to mitigate the potential randomness by LLM.

RQ3: How efective are diferent values of the attention preservation factor for BFGen?

As specified in Section 3.3, we introduce the attention preservation factor, �, to enable the R-GAT to capture important relations. RQ3 systematically examines how � afects the eficacy of BFGen. We conduct a hyperparameter sensitivity experiment to examine the impact of � on the efectiveness of BFGen, evaluating BFGen’s performance as � varies over its considered range.

RQ4: How does the completeness of requirements afect the efectiveness of BFGen?

In modern software engineering practices, especially in agile software development, requirements are progressively elaborated through iterations [57]. At the beginning, requirements briefly specify what a system should do, without details on processing flow, data definition and validation. This type of requirement is usually called a high-level requirement [58]. Since BFGen aims to generate the basic flow from a given use case description, the completeness of the use case description may afect its efectiveness. Therefore, it is necessary to evaluate BFGen with various levels of incompleteness in the use case description.

We design a requirement completeness sensitivity experiment to evaluate BFGen’s performance with incomplete requirements by implementing Random Masking [59] of content words in requirements to simulate diferent levels of completeness. By increasing the number of masked content words, we simulate requirements with varying degrees of incompleteness, using fully complete requirements (100%) as the baseline.

RQ5: How efective is the BPP in identifying branch points in base flows?

To assess how efectively BPP identifies branch points in given base flows, we conduct a comparative experiment against a set of representative baselines. To the best of our knowledge, there is no published research directly focused on predicting branch points in use case flows. The most relevant studies, based on our literature review, are rule-based event identification techniques that recognize conditional steps in requirement descriptions [14, 34, 35, 36]. However, these approaches are tightly coupled with specific datasets that are not publicly accessible. Moreover, these studies do not provide the necessary implementation code or the specific rules they employ, making replication or adaptation to the datasets in this research impossible. Therefore, we consider two categories of baselines: LLM-based baselines and a Structure-Agnostic Sequence Transformer baseline. For the LLM-based baselines, we use GPT-4o and ERNIE 4.0 Turbo. GPT-4o is a highly competitive model with strong overall performance, while ERNIE 4.0 Turbo addresses potential shortcomings of GPT-4o in processing Simplified Chinese text. To further strengthen the comparison, we implement a Structure-Agnostic Sequence Transformer (Sequence Transformer) baseline. This baseline uses the same node features, supervision signals, data splits, and evaluation protocol as our approach, but replaces SRG-based graph propagation with sequence modeling over the ordered base flow nodes. It encodes the ordered base flow node sequence with a Transformer encoder and predicts branch-point labels for each base flow node via a feed-forward classification head.

RQ6: How efective is AFGen in generating alternative flows given a use case description, its base flow and the corresponding branch points?

To address RQ6, we design a comparative experiment to evaluate AFGen against baseline methods. To the best of our knowledge, the only method we found for alternative flow generation (Ko et al. [15]), which is rule-based, was published 10 years ago. The necessary dataset, rules, and implementation details are no longer accessible, which prevents replication. Given this lack of reproducibility, we use GPT-4o, ERNIE 4.0 Turbo, and the previously introduced Structure-Agnostic Sequence Transformer (Sequence Transformer) baseline for comparison. For alternative flow generation, the Sequence Transformer baseline employs a Transformer decoder conditioned on the branch point and its fixed 1-hop local context to autoregressively generate the alternative flow action sequence.

RQ7: How does the scope of contextual information used to initialize the decoder’s hidden state afect the quality of generated alternative flows?

The initialization of the decoder’s hidden state plays a key role in the quality of alternative flows generated by AFGen. Specifically, it encodes contextual information from the �-hop neighborhood of the branch point, where the parameter � governs the scope of contextual aggregation and thus directly afects AFGen ’s ability to capture both local and long-range dependencies in the SRG.

To investigate RQ7, we conduct a hyperparameter sensitivity experiment on the context scope parameter � with respect to the efectiveness of AFGen. Specifically, we evaluate AFGen under five values of �, ranging from nodelocal (� = 0) to graph-global of � = ���: (1) 0-hop (� = 0): aggregates only the branch point’s own embedding, without any contextual information; (2) 1-hop (� = 1): uses information from directly connected neighbors; (3) 2-hop (� = 2): uses information from second-order neighbors; (4) 3-hop (� = 3): uses information from third-order neighbors; (5) All nodes (� = ���): uses all information in the graph. We evaluate the impact of each � setting on the quality of generated alternative flows using standard metrics: �1 ����� and ���.

## 4.4. Experiment Settings

To conduct the designed experiments and to answer the research questions, we perform systematic grid searches to identify the optimal parameter configurations that maximize task-specific performance.

In the experiments for answering RQ1 and RQ3, the learning rate for BFGen was set to 0.1, and the dropout rate was set to 0.3. In the experiments for answering RQ2, the learning rate was set to 1e-4, and the dropout rate was set to 0.2. In the experiments for answering RQ4, the learning rate was set to 0.01, and the dropout rate was set to 0.35.

For the experiments answering RQ5 and RQ6, we use diferent settings based on the dataset. Specifically, for the industrial NCE-T datasets, the learning rate was set to 2e-4, and the attention preservation factor � was set to 0.8, while for public datasets, the learning rate was set to 1e-4, and � to 0.84. To prevent overfitting, the early stopping patience was fixed at 20 epochs across all experimental runs.

For all the experiments except those investigating the sensitivity of the parameter � and the specific configurations for RQ5 and RQ6 mentioned above, the value of � was fixed at 0.9. Threshold � is set to 0.85. The threshold � for determining whether a node is positive is set to 0.5. Regarding the usage of LLMs in baselines and the SIP module, the temperature parameter was set to the default value (1.0) to ensure result stability.

The experiments were conducted on a server equipped with 22 vCPUs (Intel(R) Xeon(R) Platinum 8470Q), 110GB of RAM, and a single NVIDIA RTX PRO 6000 GPU (96GB), running on a Linux operating system.

Comparative Experimental Results of BFGen and Baselines on Basic Flow Generation: ���������, ������, �1 ����� and ��� Across Public and Industrial Datasets - RQ1
<table><tr><td>Dataset</td><td>Approach</td><td>Precision</td><td>Recall</td><td>F1</td><td>AUC</td></tr><tr><td rowspan="5">Public Datasets</td><td>Rule-based</td><td>0.364</td><td>0.180</td><td>0.215</td><td>0.090</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.510</td><td>0.407</td><td>0.417</td><td>0.204</td></tr><tr><td>GPT-4o</td><td>0.371</td><td>0.288</td><td>0.279</td><td>0.144</td></tr><tr><td>R-GAT with BERT</td><td>0.229</td><td>0.118</td><td>0.156</td><td>0.708</td></tr><tr><td>BFGen</td><td>0.582 / +14.12%</td><td>0.508 / +24.82%</td><td>0.543 / +30.22%</td><td>0.846 / +19.49%</td></tr><tr><td rowspan="5">Industrial Datasets</td><td>Rule-based</td><td>0.200</td><td>0.242</td><td>0.202</td><td>0.121</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.347</td><td>0.209</td><td>0.240</td><td>0.104</td></tr><tr><td>GPT-4o</td><td>0.221</td><td>0.156</td><td>0.166</td><td>0.078</td></tr><tr><td>R-GAT with BERT</td><td>0.543</td><td>0.454</td><td>0.494</td><td>0.786</td></tr><tr><td>BFGen</td><td>0.621 /+14.36%</td><td>0.488 / +7.49%</td><td>0.547  / +10.73%</td><td>0.865 /+10.05%</td></tr></table>

BFGen w/o SIPBFGen w/o Preproc. (ERNIE) BFGen w/o Preproc. (GPT) BFGen (ERNIE) BFGen (GPT)

For reproducibility, all raw experimental data, branch point annotations, baseline implementations, and other materials in the experiments are included in the replication package, available at https://github.com/WGYbuaa/FlowGen.

## 5. Results and Analysis

## 5.1. RQ1: How efective is BFGen in generating basic flows compared to the baseline methods?

Table 4 presents the comparative experimental results between BFGen and the four baselines. These results represent the average values obtained from several independent experiments, indicating that, among all methods, BFGen achieves the best performance on both public and industrial datasets.

As shown in Table 4, the rule-based baseline method exhibits limited efectiveness. Despite using mature NLP toolkits like Stanford CoreNLP, the method fails to handle the linguistic ambiguity and diferentiate terms with domainspecific meanings. Moreover, some pre-defined rules are too rigid to adapt to diferent presentation styles and domain scenarios. In contrast, BFGen leverages the SIP module— enabled by LLMs with robust text processing capabilities and rich domain knowledge—to achieve significant improvements across all four metrics on both public and industrial datasets.

The performance of LLM baseline methods, ERNIE 4.0 Turbo and GPT-4o, both with robust generation capabilities, yields similar results, slightly surpassing the rulebased methods. However, after rigorously analyzing the basic flows generated by the two LLMs, we identified hallucinations. Some action steps in the generated basic flows operate out of system boundaries. This violates the fundamental restriction of the use case. For instance, in the eANCI dataset, a use case describes the system’s functionality of presenting knowledge about fire causes to the public. However, the LLM generates a basic flow describing firefighters fire suppression procedures and the operational mechanisms of fire protection systems, which are relevant to firefighters work but not to this case, as they are not dictated in the given functional requirement descriptions. This indicates that the two LLMs fail to adequately incorporate the necessary contextual information from the functional descriptions. This is the key reason why BFGen outperforms LLM methods.

![](images/91679a9e8f27cde5d5129b0f495009f787b7e776e77fa1da430497591107c2db.jpg)  
Figure 5: RESULTS OF ABLATION STUDY FOR SIP MOD-ULE - RQ2. Note: Preproc. = Preprocessing.

The SIP module equipped with an LLM in BFGen focuses on extracting terms, actions, and objects from requirement descriptions. They are then fed into the enhanced R-GAT to capture the necessary contextual information. The LLM does not directly participate in step sequence generation, efectively mitigating the hallucination inherent in LLMs.

The large scale and high domain knowledge density of the industrial datasets provide rich information for GNN training, enabling both the R-GAT with BERT baseline and BFGen to achieve better performance. However, the R-GAT with BERT baseline cannot adjust the importance of node connections according to domain and requirement context— a limitation that hinders further performance improvement. To overcome the pitfalls, BFGen obtains more precise information with the SIP module using LLM, and proposes the attention preservation factor to dynamically adjust the influence of edge weights to prioritize the most contextually relevant node connections, thereby further achieving enhanced accuracy of capturing semantic nuances and domainspecific matches.

## 5.2. RQ2: How efective is the SIP module in contributing to the overall performance of BFGen?

We conduct this ablation study on public datasets, which exhibit higher syntactic complexity than industrial datasets and thus provide a more stringent test for validating the efectiveness of the SIP module. As shown in Fig. 5, both variants of the complete BFGen—equipped with LLMs—achieve strong performance, with BFGen (GPT) outperforming all other methods.

![](images/a337c334bbdc734d3e09c687cfb9d04fbb914c681c17f5651ec54f4e9410af14.jpg)  
Figure 6: Performance vs. Attention Preservation Factor (�)- RQ3

BFGen outperforms BFGen w/o SIP. Compared to Stanford CoreNLP, the rich domain knowledge and extensive training of LLMs integrated in the SIP allow BFGen to handle domain-specific terms and matches, greatly enhancing the accuracy of information extraction, which is essential for the subsequent modeling and training of the enhanced R-GAT.

BFGen also outperforms BFGen w/o Preproc. Though BFGen w/o Preproc. performs well in extracting semantic elements from simple sentences or sentences with few modifiers, its performance sharply decreases when dealing with complex sentences and compound sentences. Some complex sentences containing multiple verbs and nouns are dificult to parse for definitive action-object relations. In contrast, the complete BFGen with sentence simplification and splitting can systematically parse and disambiguate input sentences (e.g., clarifying verbs and their accessed objects), greatly reducing the noise in the data and the complexity of information extraction.

The experimental results demonstrate that the LLMequipped SIP module and the two preprocessing tasks, Sentence Simplification and Splitting, contribute significantly to enhancing the performance of BFGen. Interestingly, we observe that certain LLM-enabled baseline methods (e.g., BFGen w/o Preproc. (ERNIE)) underperform BFGen without SIP on certain metrics such as F1 score. This is primarily due to their unstable output and intrinsic hallucinations that lead to excessive extraction of inaccurate information in complex sentences, which induces erroneous or missing relations, ultimately degrading the overall performance.

## 5.3. RQ3: How efective are diferent values of the attention preservation factor for BFGen?

We conduct hyperparameter sensitivity experiments on industrial datasets to investigate the impact of the attention preservation factor � on BFGen. Compared to public datasets, industrial datasets contain more domain-specific matches—such as sequential actions and fixed interactions— which render model performance more sensitive to variations in �. Notably, when � <0.5, the model exhibits a sharp decline in both convergence stability and generalization capability. Consequently, our analysis focuses on the performance behavior for $\lambda \ge 0 . 5$

![](images/ce8b55479176db7f51924e49bff5fe4a01afbaac00745c5719d20d8fa4664581.jpg)  
Figure 7: Results of Requirement Completeness Sensitivity Experiments - RQ4

Fig. 6 illustrates that BFGen ’s performance is highly sensitive to �, and the resulting curve can be segmented into four phases: (1) When � is in the range of [0.50, 0.65), lower values of � weaken the contextual information capture and modeling ability of the R-GAT module, leading to low performance. (2) As � increases to [0.65, 0.80), the model can better distinguish the importance of various connections in the data with the captured contextual information, resulting in a significant performance improvement. (3) When � is in the range of [0.80, 0.90), the model performance reaches a high level since the diferences between diferent relationships captured in the model are used in adequate learning and training. (4) When � is set in the range of [0.90, 1], model performance decreases. This stems from overly focusing on local details and small variations, which amplifies sensitivity to noise in the data. Therefore, for datasets with high domain knowledge density, the model achieves a balance between attention-based and uniform embedding aggregation when $\lambda \in [ 0 . 8 0 , 0 . 9 0 )$ , efectively emphasizing critical relations in the context while mitigating the impact of noise from distant information.

## 5.4. RQ4: How does the completeness of requirement afect the efectiveness of BFGen?

Fig. 7 presents the results of sensitivity analysis on requirement completeness. The five datasets on the right show the performance retention of BFGen (relative to the baseline, in percentage) as requirement completeness decreases. The findings indicate that: (1) With at least 80% of the information retained, BFGen’s performance shows only slight fluctuation (F1 and AUC drop less than 5%), indicating strong tolerance to insuficient information. This provides strong evidence of its applicability in practical industrial scenarios since it is hard to ensure the completeness of requirements; (2) As the completeness decreases further, BFGen gradually loses its ability to recognize certain key actions or objects, particularly in the scenarios with the absence of domainspecific or contextual core words; (3) The slower decline rate of AUC compared to F1 score as the completeness decreases indicates that BFGen consistently assigns higher probabilities to positive samples, suggesting its underlying feature ranking capability remains stable. Therefore, the experimental results demonstrate that BFGen exhibits strong robustness to the completeness of requirement descriptions, maintaining stable performance even when completeness drops to 80%.

Table 5  
Comparative Experimental Results of BPP and Baselines On Branch Point Prediction : ���������, ������ and �1 ����� Across Public and Industrial Datasets - RQ5.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Approach</td><td colspan="3">Macro-Average</td><td colspan="3">Micro-Average</td></tr><tr><td>Precision</td><td>Recall</td><td>F1 Score</td><td>Precision</td><td>Recall</td><td>F1 Score</td></tr><tr><td rowspan="5">Public Datasets</td><td>GPT-4o</td><td>0.333</td><td>0.278</td><td>0.300</td><td>0.097</td><td>0.214</td><td>0.133</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.400</td><td>0.600</td><td>0.467</td><td>0.184</td><td>0.500</td><td>0.269</td></tr><tr><td>Sequence Transformer</td><td>0.180</td><td>0.383</td><td>0.244</td><td>0.238</td><td>0.417</td><td>0.303</td></tr><tr><td>BPP (GPT)</td><td>0.389</td><td>0.794</td><td>0.501</td><td>0.402</td><td>0.726</td><td>0.518</td></tr><tr><td>BPP (ERNIE)</td><td>0.657/+64.25%</td><td>0.900/+50.00%</td><td>0.717/+53.53%</td><td>0.500/+110.08%</td><td>0.956/+91.20%</td><td>0.657/+116.83%</td></tr><tr><td rowspan="5">Industrial Datasets</td><td>GPT-4o</td><td>0.198</td><td>0.074</td><td>0.097</td><td>0.220</td><td>0.022</td><td>0.040</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.376</td><td>0.206</td><td>0.230</td><td>0.302</td><td>0.076</td><td>0.122</td></tr><tr><td>Sequence Transformer</td><td>0.550</td><td>0.732</td><td>0.599</td><td>0.600</td><td>0.738</td><td>0.662</td></tr><tr><td>BPP (GPT)</td><td>0.747</td><td>0.937</td><td>0.818</td><td>0.740</td><td>0.973</td><td>0.840</td></tr><tr><td>BPP (ERNIE)</td><td>0.792/+44.00%</td><td>0.976/+33.33%</td><td>0.869/+45.08%</td><td>0.782/+30.33%</td><td>0.986/+33.60%</td><td>0.873/+31.87%</td></tr></table>

## 5.5. RQ5: How efective is the BPP in identifying branch points in base flows?

As shown in Table 5, BPP significantly outperforms all baseline methods across all metrics on both datasets, demonstrating consistently robust performance. This clearly highlights its dual strengths: efectively handling a wide range of use cases (including those with dense or sparse branch points), while maintaining high overall prediction accuracy.

Further analysis reveals an interesting contrast: the two LLM-based baselines perform better on public datasets than on industrial datasets, whereas both the Sequence Transformer baseline and BPP exhibit the opposite trend, achieving superior performance on the industrial datasets. We attribute this divergence to the intrinsic characteristics of the two datasets. The industrial datasets contain many more branch points with similar syntactic structure, more regular basic flow patterns, and substantially more training samples, enabling supervised models to learn stable local and global regularities more efectively. In particular, the Sequence Transformer baseline benefits from these data characteristics because branch points in the industrial datasets are often associated with recurring local sequential cues that can be captured by sequence modeling alone. In contrast, the public datasets are more flexible and diverse in syntax, domain, and branching style, and the relatively limited training data constrains the modeling capacity of both BPP and the Sequence Transformer baseline. Meanwhile, the LLM baselines perform worse on the industrial datasets primarily because it exhibits much more domain-specific knowledge and contains more specialized terms, while general-purpose LLMs lack the necessary prior training for such industrial scenarios, making accurate prediction challenging.

Further analysis of the LLM-based baselines’ outputs shows that a small number of errors stem from step-index misalignment. For example, LLMs generated the content of step 2 but labeled it as step 3—even though step indices were explicitly annotated in the use case flow and the prompt explicitly instructed them to use zero-based indexing for all predictions, which is crucial for use case flows with repeated operations. These errors were introduced due to LLMs’ inherent limitations as sequence-to-sequence models: they lack an explicit mechanism to align input steps with output labels [25], revealing a mismatch between the linear, context-only modeling paradigm of LLMs and the structured, position-sensitive nature of our task. In contrast, BPP leverages an enhanced R-GAT encoder to model use cases as a graph rather than text sequences, capturing both semantics and structural dependencies, and thereby inherently avoiding such misalignment errors.

Compared with the Sequence Transformer baseline, BPP also maintains a distinct advantage, suggesting that branchpoint identification depends not only on local sequential cues but also on branch-triggering control-flow logic and its relation to the surrounding flow context, both of which are explicitly propagated over the SRG rather than left implicit in sequence-only modeling.

## 5.6. RQ6: How efective is AFGen in generating alternative flows given a use case description, its base flow and the corresponding branch points?

As shown in Table 6, AFGen significantly outperforms the baseline models in Precision, F1 score, and AUC on both datasets, demonstrating its efectiveness in generating high-quality alternative flows. However, it achieves slightly lower Recall compared to the LLM-based baselines. Further analysis of the LLM baselines’ outputs reveals that they tend to produce overly verbose alternative flows containing redundant or irrelevant steps. Taking the public datasets as an example, statistics show that in the test set, the alternative flows generated by the GPT-4o baseline have an average of 11 action steps, while those generated by ERNIE 4.0 Turbo have an average of 15.79 steps. In contrast, the ground truth alternative flows contain only an average of 4.79 steps. This so-called "over-generation" strategy increases the number of matches with actions and objects, thereby inflating recall.

Comparative Experimental Results of AFGen and Baselines on Alternative Flow Generation : ���������, ������, �1 ����� and ��� Across Public and Industrial Datasets - RQ6.
<table><tr><td>Dataset</td><td>Approach</td><td>Precision</td><td>Recall</td><td>F1 Score</td><td>AUC</td></tr><tr><td rowspan="5">Public Datasets</td><td>GPT-4o</td><td>0.195</td><td>0.568</td><td>0.244</td><td>0.284</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.165</td><td>0.680</td><td>0.245</td><td>0.340</td></tr><tr><td>Sequence Transformer</td><td>0.241</td><td>0.416</td><td>0.305</td><td>0.826</td></tr><tr><td>AFGen (GPT)</td><td>0.286</td><td>0.272</td><td>0.279</td><td>0.831 / +0.61%</td></tr><tr><td>AFGen (ERNIE)</td><td>0.297  / +23.24%</td><td>0.455 / -33.09%</td><td>0.359 / +17.70%</td><td>0.827</td></tr><tr><td rowspan="5">Industrial Datasets</td><td>GPT-4o</td><td>0.121</td><td>0.431</td><td>0.176</td><td>0.216</td></tr><tr><td>ERNIE 4.0 Turbo</td><td>0.063</td><td>0.410</td><td>0.104</td><td>0.205</td></tr><tr><td>Sequence Transformer</td><td>0.712</td><td>0.403</td><td>0.515</td><td>0.930</td></tr><tr><td>AFGen (GPT)</td><td>0.584</td><td>0.430 / -0.23%</td><td>0.495</td><td>0.941</td></tr><tr><td>AFGen (ERNIE)</td><td> $0 . 7 7 2 \mathrm { ~ / ~ } / + 8 . 4 3 \%$ </td><td>0.419</td><td>0.543  / +5.44%</td><td> $0 . 9 5 3 \mathrm { ~ / ~ } + 2 . 4 7 \%$ </td></tr></table>

However, it introduces a large number of incorrect predictions, severely compromising precision, as reflected in the substantially lower Precision and F1 scores.

In contrast, AFGen generates more accurate alternative flows by leveraging the enhanced R-GAT to extract neighboring node information around branch points, thus producing representations that capture both semantic and structural dependencies. The generated alternative flows are not only semantically sound and consistent with the original use case logic, but also achieve a balance between coverage and accuracy.

Additionally, the experimental results for RQ6 exhibit a trend similar to that observed in RQ5: the LLM-based baselines perform better on public datasets than on industrial datasets, whereas the Sequence Transformer baseline and AFGen achieve superior performance on industrial datasets. As discussed in RQ5, industrial datasets contain a large number of structurally regular alternative flows, with partial similarity in their content, and provide abundant training samples, enabling supervised models to learn more efectively. Both the Sequence Transformer baseline and AFGen benefit from these data characteristics. The Sequence Transformer baseline can already model a considerable portion of these flows from local sequential patterns alone, whereas AFGen further benefits from graph-based modeling of flow dependencies, which makes it more robust to interference from domain-specific terms in the textual descriptions. In contrast, the limited training data in public datasets restricts AFGen ’s ability to generalize from diverse alternative operations, while the LLM-based baselines, benefiting from pretraining on large-scale general corpora, demonstrate stronger generalization capabilities, leading to relatively better performance on these datasets.

The Sequence Transformer baseline is a stronger comparator than the LLM baselines and is competitive with AF-Gen. Nevertheless, AFGen still maintains an overall advantage. This can be attributed in part to SRG-based graph conditioning, which better preserves branch-to-flow alignment and requirement logic—including semantic consistency, control-flow logic, and data-flow logic—than sequence-only modeling.

![](images/9fbb2eeddcf6df39ffc59a1c6d351e4cc414c39c39a4621d6e84054999153326.jpg)

![](images/d5adbad588b0b6e243e7988001d3530f88a6c49a71ef156f7d1792909d497fe7.jpg)  
Figure 8: Performance vs. Contextual Scope (�) - RQ7.

## 5.7. RQ7: How does the scope of contextual information used to initialize the decoder’s hidden state afect the quality of generated alternative flows?

As shown in Fig. 8, the performance ofAFGen (ERNIE) exhibits a clear trend when using diferent scopes of context information: both F1 score and AUC peak at the 1-hop neighborhood, while dropping significantly when extending to the context of 2-hop, 3-hop, or All Nodes. This suggests that incorporating close neighbors’ information is most efective for guiding alternative flow generation. The poor performance of the 0-hop setting—which uses only the branch point’s embedding without any contextual input—confirms the importance of local structural and semantic cues in capturing functional dependencies within the SRG. In contrast, when the context range extends beyond 1-hop, redundant or noisy information from distant nodes may degrade model performance. The All Nodes setting, in particular, underperforms the 1-hop setting, suggesting that aggregating over all nodes may blur the fine-grained semantic distinctions critical for accurate step prediction. These results imply that a moderate scope of context— specifically, 1-hop neighbors—is optimal for initializing the decoder in AFGen. It strikes a balance between leveraging suficient local semantics and avoiding interference from irrelevant long-range dependencies.

## 6. Threats To Validity

Threats to the internal validity pertain to experimental biases and errors that may originate from four primary sources: language unification of public datasets, natural language parsing tools, the branch-point annotation process, and the hyperparameter configurations set during model training. (1) Given the presence of Italian and English in public datasets, we follow the methodology proposed by Hey et al. [50], utilizing the high-performance translation engine DeepL [60] to unify the language into English. This process efectively mitigates potential cross-lingual bias risks. We integrate the translated dataset into the replication package to ensure methodological transparency. (2) To mitigate potential biases stemming from limited NLP accuracy, we leverage excellent LLMs, GPT-4o and ERNIE 4.0 Turbo, to enhance the SIP module’s capabilities. However, we acknowledge the inherent probabilistic nature of generative models. Even with fixed hyperparameters (e.g., temperature), minor variations in output may occur. To minimize this threat, we conducted multiple runs for critical steps and reported averaged results. (3) In the branch point annotation pipeline, the five engineers review the LLM suggestions independently of one another, but the review is not blind to the suggested branch point. This may introduce anchoring bias. Nevertheless, the substantial Fleiss’ � and the high LLM-human alignment suggest that the final annotations remain reasonably stable. (4) During the modeling and training, critical hyperparameters—including �, early stopping epochs, and learning rates—as well as the initialization context scope for AFGen ’s hidden state have a substantial impact on model performance. The hyperparameter settings in our experiments were derived from results of the grid searches we conducted. However, we explicitly acknowledge that hyperparameter optimization requires more comprehensive empirical validation. In subsequent research phases, we plan to conduct dedicated empirical studies to establish evidence-based guidelines for hyperparameter selection.

Threats to external validity primarily concern the generalizability of FlowGen. For this study, we rigorously validated the proposed approach using 13 public datasets and 7 industrial datasets spanning diverse domains. This multisource validation strategy surpasses conventional academic benchmarks confined to singular domains, thereby better approximating real-world engineering scenarios. Notably, these datasets collectively contain 6,984 use cases, with the largest sub-dataset containing 5,701 use cases—already exhibiting characteristics of a large-scale dataset. While current empirical validation demonstrates FlowGen’s preliminary generalizability, adhering to the "more-is-better" principle, we plan to implement FlowGen on more diverse datasets across additional domains.

Threats to construct validity stem from the appropriateness of evaluation metrics. We employ Precision, Recall,

F1 score, and AUC as the assessment framework. Precision, Recall, and F1 score are representative multi-objective classification metrics [61], widely adopted as performance indicators for models [53]. AUC is included as a complementary indicator of discriminative performance [62]. Moreover, to address the class imbalance inherent in the branch point prediction task, we report results using both macro- and micro-average metrics. These metrics provide a fair and consistent basis for comparing FlowGen with all baseline methods. Nevertheless, they mainly assess overlap and discrimination at the node or branch-point level, and do not fully capture higher-level properties of complete use case flows, such as end-to-end executability.

## 7. Conclusion

This paper presents FlowGen for complete use case flow generation from high-level requirements. FlowGen integrates LLM-based semantic extraction, SRG-based relational modeling, basic flow generation, branch point prediction, and alternative flow generation, together with a semiautomatic pipeline for supplementing branch-point annotations in public datasets.

Evaluations on 13 public and 7 industrial datasets demonstrate that FlowGen outperforms baseline methods across major metrics. In particular, for branch point prediction and alternative flow generation, the comparisons against LLM-based baselines and the Sequence Transformer baseline indicate that SRG-based graph conditioning is useful for preserving branch-to-flow alignment and requirement logic beyond sequence-only modeling. Ablation studies and hyperparameter sensitivity experiments validate the efectiveness of the LLM-based semantic information processing module and the introduced attention preservation factor in BFGen, while also revealing the impact of contextual scope on AFGen ’s performance. Furthermore, a requirement completeness sensitivity experiment confirms that BFGen maintains robust performance even when input requirements are incomplete.

Future work will focus on exploring how preconditions and postconditions in use case specifications can be efectively modeled and used to guide the generation of both basic and alternative flows, and how more explicit system boundary representations can be introduced to further constrain out-of-scope generation.

## Acknowledgments

This work was supported in part by the Integration and Application of Digital Learning Technology Ministry of Education Innovation under Grant 1431005.

## References

[1] K. E. Wiegers, J. Beatty, Software requirements, Pearson Education, 2013.

[2] S. Tiwari, A. Gupta, A systematic literature review of use case specifications research, Information and Software Technology 67 (2015) 128–158.

[3] T. Yue, L. C. Briand, Y. Labiche, Facilitating the transition from use case models to analysis models: Approach and experiments, ACM Transactions on Software Engineering and Methodology (TOSEM) 22 (2013) 1–38.

[4] V. Vranić, J. Lang, M. L. Nores, J. J. P. Arias, J. Solano, G. Laseca, Use case modeling in a research setting of developing an innovative pilgrimage support system, Universal Access in the Information Society 23 (2024) 1543–1560.

[5] X. Liu, Y. Liu, Y. Zhuang, W. Hou, Ucd-llm: A use case diagram requirement modeling multi-agent framework with large language model, Information and Software Technology (2025) 107955.

[6] J. L. Santos, L. E. G. Martins, J. S. Molléri, Requirements extraction from model-based systems engineering: A systematic literature review, Journal of Systems and Software 226 (2025) 112407.

[7] C. Wang, F. Pastore, A. Goknil, L. C. Briand, Automatic generation of acceptance test cases from use case specifications: an nlp-based approach, IEEE Transactions on Software Engineering 48 (2020) 585–616.

[8] X. Lian, J. Ma, H. Lv, L. Zhang, Reqcompletion: domain-enhanced automatic completion for software requirements, Requirements Engineering (2025) 1–21.

[9] D. Ko, S. Kim, S. Park, Automatic recommendation to omitted steps in use case specification, Requirements Engineering 24 (2019) 431– 458.

[10] G. Wang, J. Wu, H. Yang, Q. Sun, T. Yue, Test architecture generation by leveraging bert and control and data flows, in: International Conference on Engineering of Complex Computer Systems, Springer, 2024, pp. 125–145.

[11] Y. Elrakaiby, A. Borgida, A. Ferrari, J. Mylopoulos, Care: a refinement calculus for requirements engineering based on argumentation theory, Software and Systems Modeling 21 (2022) 2113–2132.

[12] M. Jahan, Z. S. H. Abad, B. Far, Generating sequence diagram from natural language requirements, in: 2021 IEEE 29th International Requirements Engineering Conference Workshops (REW), IEEE, 2021, pp. 39–48.

[13] T. Yue, L. C. Briand, Y. Labiche, atoucan: an automated framework to derive uml analysis models from use case models, ACM Transactions on Software Engineering and Methodology (TOSEM) 24 (2015) 1– 52.

[14] J. Jurkiewicz, J. Nawrocki, Automated events identification in use cases, Information and Software Technology 58 (2015) 110–122.

[15] D. Ko, S. Park, Y. Kim, S. Park, S. Kim, Suggesting alternative scenarios using use case specification patterns for requirement completeness, International Journal of Software Engineering and Knowledge Engineering 26 (2016) 927–951.

[16] A. Al-Hroob, A. T. Imam, R. Al-Heisa, The use of artificial neural networks for extracting actions and actors from requirements document, Information and Software Technology 101 (2018) 1–15.

[17] H. Zhao, J. Wang, P. Liang, W. Huang, A requirements refinement approach for service-based systems, in: 2018 IEEE 9th International Conference on Software Engineering and Service Science (ICSESS), 2018, pp. 495–498. doi:10.1109/ICSESS.2018.8663869.

[18] M. Makino, A. Ohnishi, Scenario generation using diferential scenario information, IEICE TRANSACTIONS on Information and Systems 95 (2012) 1044–1051.

[19] R. Sanyal, B. Ghoshal, et al., A hybrid approach to extract conceptual diagram from software requirements, Science of Computer Programming 239 (2025) 103186.

[20] M. B. Chaaben, L. Burgueño, H. Sahraoui, Towards using fewshot prompt learning for automating model completion, in: 2023 IEEE/ACM 45th International Conference on Software Engineering: New Ideas and Emerging Results (ICSE-NIER), IEEE, 2023, pp. 7– 12.

[21] M. Krishna, B. Gaur, A. Verma, P. Jalote, Using llms in software requirements specifications: an empirical evaluation, in: 2024 IEEE 32nd International Requirements Engineering Conference (RE), IEEE, 2024, pp. 475–483.

[22] B. Jin, G. Liu, C. Han, M. Jiang, H. Ji, J. Han, Large language models on graphs: A comprehensive survey, IEEE Transactions on Knowledge and Data Engineering (2024).

[23] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy, H. Wang, Large language models for software engineering: A systematic literature review, ACM Trans. Softw. Eng. Methodol. 33 (2024).

[24] D. Jin, S. Zhao, Z. Jin, X. Chen, C. Wang, Z. Fang, H. Xiao, An evaluation of requirements modeling for cyber-physical systems via llms, arXiv preprint arXiv:2408.02450 (2024).

[25] Y. Liu, D. Li, K. Wang, Z. Xiong, F. Shi, J. Wang, B. Li, B. Hang, Are llms good at structured outputs? a benchmark for evaluating structured output capabilities in llms, Information Processing & Management 61 (2024) 103809.

[26] Y. K. Lal, V. Cohen, N. Chambers, N. Balasubramanian, R. Mooney, Cat-bench: Benchmarking language model understanding of causal and temporal dependencies in plans, in: Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 19336–19354. doi:10.18653/v1/2024.emnlp-main.1077.

[27] H. Chi, H. Li, W. Yang, F. Liu, L. Lan, X. Ren, T. Liu, B. Han, Unveiling causal reasoning in large language models: Reality or mirage?, in: Advances in Neural Information Processing Systems, 2024. doi:10.52202/079017-3064.

[28] G. Wang, B. Li, J. Wu, Z. Shao, Bfgen: Basic flow generation for refining requirements via llm and relational graph attention networks, in: 2025 25th International Conference on Software Quality, Reliability and Security (QRS), IEEE, 2025, pp. 120–131.

[29] A. M. Alashqar, Automatic generation of uml diagrams from scenario-based user requirements, Jordanian Journal of Computers and Information Technology 7 (2021).

[30] C. Wang, B. Wang, P. Liang, J. Liang, Assessing uml diagrams by gpt: Implications for education, Journal of Systems and Software (2025) 112709.

[31] S. Ren, H. Nakagawa, T. Tsuchiya, Combining prompts with examples to enhance llm-based requirement elicitation, in: 2024 IEEE 48th Annual Computers, Software, and Applications Conference (COMP-SAC), IEEE, 2024, pp. 1376–1381.

[32] S. Mandal, A. Chethan, V. Janfaza, S. Mahmud, T. A. Anderson, J. Turek, J. J. Tithi, A. Muzahid, Large language models based automatic synthesis of software specifications, arXiv preprint arXiv:2304.09181 (2023).

[33] Y. Li, Z. Li, P. Wang, J. Li, X. Sun, H. Cheng, J. X. Yu, A survey of graph meets large language model: Progress and future directions, arXiv preprint arXiv:2311.12399 (2023).

[34] I. Williams, X. Yuan, M. Anwar, J. T. McDonald, An automated security concerns recommender based on use case specification ontology, Automated Software Engineering 29 (2022) 42.

[35] A. Rago, C. Marcos, J. A. Diaz-Pace, Assisting requirements analysts to find latent concerns with reassistant, Automated Software Engineering 23 (2016) 219–252.

[36] M. Makino, A. Ohnishi, A method of scenario generation with diferential scenario, in: 2008 16th IEEE International Requirements Engineering Conference, IEEE, 2008, pp. 337–338.

[37] H. Shudo, A. Ohnishi, A method for exception scenarios generation using templates of exceptions, in: 2010 Asia Pacific Software Engineering Conference, IEEE, 2010, pp. 13–22.

[38] A. C. Marcén, A. Iglesias, R. Lapeña, F. Pérez, C. Cetina, A systematic literature review of model-driven engineering using machine learning, IEEE Transactions on Software Engineering 50 (2024) 2269–2293.

[39] J. Di Rocco, C. Di Sipio, D. Di Ruscio, P. T. Nguyen, A gnnbased recommender system to assist the specification of metamodels and models, in: 2021 ACM/IEEE 24th International Conference on Model Driven Engineering Languages and Systems (MODELS), IEEE, 2021, pp. 70–81.

[40] H. Liu, Y. Dong, Q. Ke, Z. Zhou, Reco: A modular neural framework for automatically recommending connections in software models, in:

2024 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), 2024, pp. 637–648. doi:10.1109/ SANER60148.2024.00070.

[41] H. Phan, A. Jannesari, Heterogeneous graph neural networks for software efort estimation, in: Proceedings of the 16th ACM/IEEE International Symposium on Empirical Software Engineering and Measurement, 2022, pp. 103–113.

[42] N. Mehrotra, A. Sharma, A. Jindal, R. Purandare, Improving crosslanguage code clone detection via code representation learning and graph neural networks, IEEE Transactions on Software Engineering 49 (2023) 4846–4868.

[43] M. N. Rafi, D. J. Kim, A. R. Chen, T.-H. Chen, S. Wang, Towards better graph neural network-based fault localization through enhanced code representation, Proceedings of the ACM on Software Engineering 1 (2024) 1937–1959.

[44] J. R. Stevens, D. Das, S. Avancha, B. Kaul, A. Raghunathan, Gnnerator: A hardware/software framework for accelerating graph neural networks, in: 2021 58th ACM/IEEE Design Automation Conference (DAC), IEEE, 2021, pp. 955–960.

[45] K. Sparck Jones, A statistical interpretation of term specificity and its application in retrieval, Journal of documentation 28 (1972) 11–21.

[46] C. Manning, H. Schutze, Foundations of statistical natural language processing, MIT press, 1999.

[47] S. Narayan, C. Gardent, Hybrid simplification using deep semantics and machine translation, in: The 52nd annual meeting of the association for computational linguistics, 2014, pp. 435–445.

[48] OpenAI, Hello gpt-4o, \https://openai.com/index/hello-gpt-4o/, 2024.

[49] Y. Sun, S. Wang, Y. Li, S. Feng, X. Chen, H. Zhang, X. Tian, D. Zhu, H. Tian, H. Wu, Ernie: Enhanced representation through knowledge integration, arXiv preprint arXiv:1904.09223 (2019).

[50] T. Hey, F. Chen, S. Weigelt, W. F. Tichy, Improving traceability link recovery using fine-grained requirements-to-code relations, in: 2021 IEEE International Conference on Software Maintenance and Evolution (ICSME), IEEE, 2021, pp. 12–22.

[51] Y. Meng, X. Pan, J. Chang, Y. Wang, Rgat: A deeper look into syntactic dependency information for coreference resolution, in: 2023 International Joint Conference on Neural Networks (IJCNN), IEEE, 2023, pp. 1–8.

[52] P. Hurtik, S. Tomasiello, J. Hula, D. Hynar, Binary cross-entropy with dynamical clipping, Neural Computing and Applications 34 (2022) 12029–12041.

[53] H. Gao, H. Kuang, W. K. Assunção, C. Mayr-Dorn, G. Rong, H. Zhang, X. Ma, A. Egyed, Triad: Automated traceability recovery based on biterm-enhanced deduction of transitive links among artifacts, in: Proceedings of the IEEE/ACM 46th International Conference on Software Engineering, 2024, pp. 1–13.

[54] A. Ferrari, G. O. Spagnolo, S. Gnesi, Pure: A dataset of public requirements documents, in: 2017 IEEE 25th international requirements engineering conference (RE), IEEE, 2017, pp. 502–505.

[55] J. L. Fleiss, Measuring nominal scale agreement among many raters., Psychological bulletin 76 (1971) 378.

[56] C. D. Manning, M. Surdeanu, J. Bauer, J. R. Finkel, S. Bethard, D. McClosky, The stanford corenlp natural language processing toolkit, in: Proceedings of 52nd annual meeting of the association for computational linguistics: system demonstrations, 2014, pp. 55–60.

[57] D. Russo, The agile success model: a mixed-methods study of a large-scale agile transformation, ACM Transactions on Software Engineering and Methodology (TOSEM) 30 (2021) 1–46.

[58] M. P. E. Heimdahl, N. G. Leveson, Completeness and consistency in hierarchical state-based requirements, IEEE transactions on Software Engineering 22 (2002) 363–377.

[59] C. Huang, Q. Xu, Y. Wang, Y. Wang, Y. Zhang, Self-supervised masking for unsupervised anomaly detection and localization, IEEE Transactions on Multimedia 25 (2022) 4426–4438.

[60] DeepL — deepl.com, https://www.deepl.com/, 2026.

[61] M.-L. Zhang, Z.-H. Zhou, A review on multi-label learning algorithms, IEEE transactions on knowledge and data engineering 26

(2013) 1819–1837.

[62] J. Huo, Y. Gao, Y. Shi, H. Yin, Cross-modal metric learning for auc optimization, IEEE Transactions on Neural Networks and Learning Systems 29 (2018) 4844–4856.