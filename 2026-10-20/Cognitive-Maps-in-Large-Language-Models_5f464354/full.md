# Cognitive Maps in Large Language Models

Albert Qin,<sup>1,</sup> <sup>∗</sup> Jonathan Cohen,<sup>2</sup> and Gautam Reddy<sup>1,</sup> <sup>†</sup>

<sup>1</sup>Department of Physics, Princeton University, Princeton, New Jersey

<sup>2</sup>Princeton Neuroscience Institute, Princeton University, Princeton, New Jersey

It has long been proposed that animals construct cognitive maps to support flexible behavior in novel environments. We tested whether large language models form similar representations using relational reasoning tasks and causal “mosaic experiments” that manipulate internal activations in open-weight language models. Models beyond the ten-billion-parameter scale successfully inferred unstated relationships between entities and generalized to synthetic entity names. Across model and task families, models generated abstract representations of relational structure and combined them additively with representations of entity identity, binding entities to roles within that structure. Models stored observed relationships as pairwise links that they composed retrospectively when queried, rather than prospectively positioning each entity in a global coordinate system. Binding was mediated by a sparse, distributed set of key–value pairs; by perturbing these pairs, we implanted “false memories” that altered model behavior in predictable ways. Overall, our findings reveal an internal organization analogous to a cognitive map that explains how language models reason with limited information.

## I. INTRODUCTION

A hallmark of cognition is the ability to identify structure and make inferences from limited experience. A classic example of this ability is observed during spatial navigation: with suficient experience in a maze, animals can plan a novel route to reach a goal [1]. In hu mans, this ability is observed across virtually all domains that involve relational reasoning. For example, if we are told that Alice’s daughter is Beth and Beth’s daughter is Carol, we can infer that Carol is Alice’s granddaughter, even when this relationship is not directly stated. These abilities are made possible by the fact that the same relational structures recur across many contexts: navigation on a flat terrain must obey the rules of two-dimensional geometry, and all human families share similar structure even though the identities of family members difer from family to family. It has long been proposed that animals construct internal representations—variously referred to as schemata [2], mental models [3], or cognitive maps [1, 4]—that capture such underlying structure to support flexible behavior beyond what has been directly experienced.

To construct such a representation, a learner must combine previously acquired knowledge about the relevant relational structure (e.g., the abstract structure of a family tree) with information specific to the current experience (e.g., the names “Alice”, “Beth”, and “Carol”). Several lines of theoretical work, spanning memory [5–7], structure learning [8–11], and abstract reasoning [12–14], can be understood as addressing complementary aspects of this problem. Although their details difer, many of these theoretical models share the feature that information specific to the current context is bound to a separable and reusable (i.e., abstract) representation of relational structure. This organization allows previously acquired structural knowledge to be rapidly applied to the particulars of a new context. Therefore, to the extent that these theories account for aspects of human cognition and patterns in neural and behavioral data, they suggest that animal brains construct cognitive maps whose organization reflects the recurrence of relational structures across contexts.

This hypothesis immediately raises the question of whether analogous representational strategies can arise in a diferent system that learns from data with sim ilar statistical features. Modern large language models provide a clear example of such a system. These models exhibit strikingly broad capabilities that arise through training on large natural language corpora [15– 17]. Crucially, these corpora contain recurring semantic structures that are instantiated by diferent entities [18, 19], which in turn may arise from the relational structures that recur across the situations that language describes. Therefore, learning from these corpora using objectives that favor generalization across contexts (e.g., next-token prediction) may promote representational strategies that capture recurring relational structure. Importantly, such strategies are not prescribed in advance but rather must emerge through training; their particular form must therefore be characterized empirically.

Here, we studied the behavior and internal representations of trained large language models when asked to perform a set of relational-reasoning tasks. We found that models could infer unstated relationships and retained this ability when familiar entity names were replaced with novel ones (Fig. 1). By perturbing the models’ internal states, we found evidence for an abstract representation of relational structure that is separable from entity identity (Fig. 2). Representations of structure and identity combined additively (Fig. 3). Multi-step inference relied primarily on separately represented relationships that were composed when required by a query (Fig. 4). Finally, two separate, sparse sets of key–value pairs supported retrieval of structural information from entity names and vice versa (Fig. 5). Together, our findings reveal how language models organize information in their internal states to support flexible relational reasoning.

## II. RESULTS

## A. Large language models infer unstated relationships and generalize to novel entity names

To study how language models use structural knowledge to organize newly presented information for inference, we designed a set of relational-reasoning tasks that require models to answer questions about relationships implied by a short prompt. On each trial, models received a context stating two relationships among three entities, then were asked a question about a relationship determined by that context (Fig. 1A). We evaluated models across 14 relational-reasoning settings spanning four task families: family trees, ordering relations (e.g., age comparisons), signed interactions (e.g., biological interaction networks), and equivalence relations (Fig. 1B; SI Sec. VIII).

We studied the behavior of a collection of 11 openweight transformer models spanning multiple model families (Gemma-3, Llama-3.1, Qwen-3) and scales (∼600M to ∼70B parameters) on these tasks. Models were evaluated without task-specific fine-tuning or in-context demonstrations. Across most relational reasoning settings, suficiently large models reliably answered questions about unstated relationships implied by the context (Figs. 1C; S8–S21). Furthermore, performance was preserved when the entity names were replaced with synthetic names that lacked domain-appropriate semantic associations (Figs. 1D; S8–S21). Together, these observations demonstrate a capacity for inference that exhibits a form of systematicity [20]: the models’ abilities extended immediately to novel instantiations of familiar relational structures.

## B. Probing internal representations in transformer models

We next asked about the structure of the internal representations that support these abilities. We begin by identifying the relevant internal states in a transformerbased language model [21] that carry information about the stated relationships, then describe our approach to probing their organization.

To process text, these models divide the text into tokens, associate an initial vector with each token, then update these vectors through a sequence of layers (Fig. 2A). We collect the vectors associated with the relationship statements into an object we call the activation tensor:

$$
\mathbf { A } = \left\{ \mathbf { h } _ { t l } \right\} _ { t = t _ { \mathrm { s t a r t } } , \ldots , t _ { \mathrm { e n d } } } ,\tag{1}
$$

where $\mathbf { h } _ { t l }$ denotes the vector generated at token position t and layer $l ,$ and $t _ { \mathrm { s t a r t } }$ and $t _ { \mathrm { e n d } }$ delimit the relevant segment of the prompt.

Within each layer, the self-attention mechanism permits the retrieval of information from earlier token positions. This is mediated by key–value pairs generated from each vector by learned linear projections: the keys guide retrieval, while the values supply the retrieved information. We collect the key–value pairs associated with the relationship statements into an object we call the key–value memory:

$$
\begin{array} { r } { \mathbf { M } = \Big \{ ( \mathbf { k } _ { t l } ^ { ( i ) } , \mathbf { v } _ { t l } ^ { ( i ) } ) \Big \} _ { l = t _ { \mathrm { s t a r t } } , \dots , t _ { \mathrm { e n d } } , } } \\ { i = 1 , \dots , N _ { \mathrm { l a y e r s } } } \end{array}\tag{2}
$$

where $( \mathbf { k } _ { t l } ^ { ( i ) } , \mathbf { v } _ { t l } ^ { ( i ) } )$ denotes the ith key–value pair at token position t and layer l, and $N _ { \mathrm { K V } }$ is the number of key– value pairs generated per vector.

Crucially, the dependence of the models’ responses on the stated relationships is mediated entirely through A and M. Specifically, information about the relationships is encoded in A, then made available for retrieval through the key–value memory M:

$$
\mathrm { r e l a t i o n s h i p s \longrightarrow \bf A } \longrightarrow \bf M \xrightarrow { \mathrm { q u e r y } } \mathrm { r e s p o n s e } .\tag{3}
$$

Questions about the structure of the cognitive map can therefore be operationalized as questions about the organization of information in A and M.

To probe the structure of these representations, we designed a family of in silico experiments—here termed mosaic experiments—based on the approach described in [22]. In each experiment, we constructed a “mosaic” activation tensor by recombining parts of “source” activation tensors generated as the model processed two or more diferent prompts. Mosaic key–value memories were constructed in the same way. For example, in Fig. 2B, we recorded activation tensors for two prompts describing the same relational structure using diferent entity names, then constructed $\mathbf { A } _ { \mathrm { m o s a i c } }$ by replacing the row of $\mathbf { A } _ { \mathrm { c o n t r o l } }$ corresponding to the first “Beth” token with the corresponding row for $^ { 6 6 } \mathrm { E v e } ^ { 9 5 }$ in $\mathbf { A } _ { \mathrm { d o n o r } }$ . We then presented the model with a query while holding the internal states corresponding to A or M fixed at the values specified by the mosaic representation; all other internal states were computed normally. The model’s responses under such a manipulation provide a behavioral readout of the relationships supported by the mosaic representation. As we show next, diferent hypotheses about the organization of information in A or M make diferent predictions about the outcomes of such experiments.

![](images/8c0cd52db83cdbbb995c4d6179245bdcb287f0be35237990ddf37c591b3010d1.jpg)  
FIG. 1. Behavioral task, model performance, and generalization. A, Task structure. B, Relational reasoning settings. Solid black arrows denote examples of stated relationships; dashed blue arrows denote an implied relationship. C, Model performance in the family-tree setting. Colors denote model families; points denote individual models. Gemma-3-27B, indicated here, is shown as an example in subsequent main-text figures; corresponding results across models and relational reasoning settings are reported in the SI. D, Generalization to synthetic names. Green denotes domain-appropriate names (here, common English names); orange denotes synthetic names (here, random three-letter strings). For each condition, 50 trials are shown; a new set of entity names is sampled for each trial. ‹A› and ‹C› denote the first and third entity names on each trial. Each trial contributes one point for each of the three plotted tokens. Probability ratios are token probabilities divided by the geometric mean across the plotted answer tokens.

## C. Relational structure is represented separately from entity identity

We first asked how a single relationship (e.g., “Alice has daughter Beth”) is encoded in the activation tensor A. We considered two hypotheses:

Under the attribute tagging hypothesis, models encode relationships as an attribute of one of the participating entities (Fig. 2C, left). This hypothesis is motivated by empirical evidence indicating that language models express facts learned during training by tagging representations at entity tokens with features that encode attributes of those entities [23–25]. Analogously, relationships introduced in the prompt could be encoded by constructing such features on the fly. For example, the models could encode the relationship “Alice has daughter Beth” by adding to the representations at “Beth” a feature corresponding to the attribute “has mother Alice” (e.g., using a conjunctive code [26, 27] for “has mother”

and “Alice”). Because the complete relationship is not available until the latter entity (e.g., “Beth”) is processed, such a feature is constrained to appear in the row of A corresponding to the latter entity token.

An alternative hypothesis, the identity–structure binding hypothesis, is that models carry an abstract, nameindependent representation of the relationship, then bind the names to that representation (Fig. 2C, right). This hypothesis is related to recent proposals that language models develop internal mechanisms that can sup port computations associated with symbolic processing [22, 28–32]. Under this hypothesis, changing the entity names should leave the abstract representation of the relationship intact while altering the name-specific information bound to that representation.

The mosaic experiment illustrated in Fig. 2B was designed to distinguish between the two hypotheses. In this experiment, $\mathbf { A } _ { \mathrm { m o s a i c } }$ was constructed from two source tensors, $\mathbf { A } _ { \mathrm { c o n t r o l } }$ and $\mathbf { A } _ { \mathrm { d o n o r } }$ , recorded as models processed contexts with the same relational structure but diferent entity names. For example, the control and donor contexts could read “Alice has daughter Beth; Beth has daughter Carol” and “Daisy has daughter Eve; Eve has daughter Fiona”, respectively. In this example, the mosaic would be constructed by replacing the row in $\mathbf { A } _ { \mathrm { c o n t r o l } }$ corresponding to the first occurrence of “Beth” with the row in $\mathbf { A } _ { \mathrm { d o n o r } }$ corresponding to the first occurrence of “Eve”.

![](images/30bf6bdab0171c3eb3b4574711e850db3c802c7efb9e835c44501e6fe5b32ce6.jpg)

![](images/55222cc2dde78ad88419ba3930a8dd6d331b210a36878e9f5ae75ae355ae2bc4.jpg)

![](images/84c9bdf6751474b2a35dee1c0e6a871f713c3d376f7bf2d0d77bac03413a5b8a.jpg)

C  
![](images/c1b8e488cbf1747163d233afe709eaec27ca56a71133c9e13b57889ad8c9268f.jpg)

E  
![](images/1812b9072b772e093d6c84b8ae00f8fbfb6922ebc3dc1e6e218f665895eb9a65.jpg)

![](images/c30a8a7739320c90f132810c9c51ade7bd58928c989cfd2a82b24604cb4ad490.jpg)

![](images/892b134cfba2577b2469ebc6d51d3cb814e09ee06958e57ddaed03983a5cb590.jpg)

F  
![](images/10c0f3e84cf5730dd4dcbb0a6e9a13479033e31ec10fbf79a1df511a9c7a3902.jpg)

![](images/93d3493724348f6d039082ac8568e7a7dd0e206c07075db42c464c25c3502c05.jpg)  
FIG. 2. Mosaic experiments probe the organization of information within the activation tensor. A, Information flow through a transformer. Each square represents a vector at one layer and token position; arrows indicate the direction of causal dependence. Tokenization is schematic in this and subsequent diagrams. Later vectors retrieve information through keys (green) and values (orange) generated from earlier vectors (inset). The vector at the most recent token position of the last layer determines the probability distribution over the next token. The activation tensor A comprises the set of intermediate vectors in the segment of the prompt containing the relationships (dashed box). The key–value memory M is diagrammed separately in Fig. 5A. B, Schematic of a mosaic experiment. (1) Source tensors are recorded. (2) Selected activations are recombined into a mosaic tensor. (3) The model processes a query while activations in the mosaic region remain fixed at their mosaic values. Activations at subsequent token positions (unfilled squares) are computed normally; next-token probabilities provide a behavioral readout of the relationships supported by the mosaic. C, Competing hypotheses for the organization of information in the activation tensor. Labels indicate the information carried by each highlighted region under each hypothesis. D, Predictions for the outcome of the experiment in B. E, Outcome of the experiment. Each plot compares model responses for the control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in Fig. 1D. Fifty trials are shown per condition. Names in the query headings are illustrative. Results are shown for Gemma-3-27B in the family-tree setting; corresponding analyses for other models and relational-reasoning settings are reported in the SI. F, Inferred organization of the mosaic. Shapes denote the three structural roles in the relational structure. Entity names are bound to these roles at the corresponding token rows. “Beth” and “Eve” are both bound to the middle role.

The two hypotheses make diferent predictions for the outcome of this experiment. The attribute tagging hypothesis predicts that the substituted row in A<sub>mosaic</sub> carries the feature “has mother Daisy”. Models should therefore behave as if Eve’s mother were Daisy (Fig. 2D, left). In contrast, the identity-structure binding hypothesis predicts that replacing Beth’s row in $\mathbf { A } _ { \mathrm { c o n t r o l } }$ with the corresponding row from $\mathbf { A } _ { \mathrm { d o n o r } }$ leaves a nameindependent representation of the relationship intact while changing the name bound to that representation from “Beth” to “Eve”. Models should therefore behave as if Eve’s mother were Alice (Fig. 2D, right). Note that we use the names “Alice”, “Beth”, . . . , and the family tree setting for illustrative purposes; for each mosaic experiment, names were resampled on each trial and analogous experiments were performed for all relational reasoning tasks.

Consistent with the identity-structure binding hypothesis, we found that models could recover the relationship “Eve has mother $\mathrm { A l i c e } ^ { \mathrm { \ ' } }$ , but not “Eve has mother Daisy”, from $\mathbf { A } _ { \mathrm { m o s a i c } }$ (Fig. 2E, top). Unexpectedly, we also found that models could recover the relationship “Eve’s daugh ter is Carol” (Fig. 2E, middle), and the two-step relationship “Alice’s granddaughter is Carol” and its inverse (Fig. 2E, bottom). We observed similar results when models answered queries requiring an entity name, such as “Who is Eve’s mother?” (Fig. S6). Thus, models recovered relationships from $\mathbf { A } _ { \mathrm { m o s a i c } }$ as if Eve’s name had become associated with Beth’s place in the full control family tree. This pattern suggests a more general version of the identity–structure binding hypothesis, where the activation tensor binds a representation of each entity name to some component—common to $\mathbf { A } _ { \mathrm { c o n t r o l } }$ and $\mathbf { A } _ { \mathrm { d o n o r } }$ representing some feature of the prompt that is independent of entity identity.

What feature of the prompt does this component represent? We consider three possibilities. One possibility is that it represents the sequential structure of the prompt. Specifically, A could bind tokens to a sequence of position codes:

$$
\begin{array} { r l r } & { } & { \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } , \mathcal { P } _ { 3 } , \mathcal { P } _ { 4 } , \mathcal { P } _ { 5 } , \mathcal { P } _ { 6 } } \\ & { } & { ( \mathcal { P } _ { 1 }  \mathrm { A l i c e } ) , ( \mathcal { P } _ { 2 }  \mathrm { h a s ~ d a u g h t e r } ) , \dots . } \end{array}\tag{4}
$$

Under this account, $\mathbf { A } _ { \mathrm { m o s a i c } }$ would encode the sequence “Alice has daughter Eve; Beth has daughter Carol.” This would explain why models treated Alice as Eve’s mother (Fig. 2E, top); however, it is inconsistent with the models’ belief that Carol is Eve’s daughter or Alice’s granddaughter (Fig. 2E, middle and bottom). As an additional test, we evaluated models on mosaic tensors designed to permute either the positional signals supplied by the transformer architecture or any internally constructed codes for global token position. Neither manipulation appreciably changed the models’ behavior (Figs. S1, S2).

A second possibility is that the name-independent component of A represents two abstract mother– daughter relationships. Entity names could then be bound to the mother and daughter roles in each relationship:

$$
\begin{array} { r l } & { \mathcal { M } _ { 1 } \xrightarrow { \mathrm { h a s ~ d a u g h t e r } } \mathcal { D } _ { 1 } , \ \mathcal { M } _ { 2 } \xrightarrow { \mathrm { h a s ~ d a u g h t e r } } \mathcal { D } _ { 2 } } \\ & { \qquad ( \mathcal { M } _ { 1 }  \mathrm { A l i c e } ) , \ ( \mathcal { D } _ { 1 }  \mathrm { B e t h } ) , } \\ & { \qquad ( \mathcal { M } _ { 2 }  \mathrm { B e t h } ) , \ ( \mathcal { D } _ { 2 }  \mathrm { C a r o l } ) . } \end{array}\tag{5}
$$

Because the two relationships are linked only through the shared name “Beth”, the account likewise cannot explain the middle or bottom observations in Fig. 2E.

A third possibility is that the name-independent component of A represents the full relational structure of the prompt, including the fact that the daughter in the first clause and the mother in the second clause correspond to the same entity. The abstract structure represented in A could then be depicted as a three-node structure in which the two relationships share a middle node:

$$
\begin{array} { r l } & { \triangle \xrightarrow { \mathrm { h a s ~ d a u g h t e r } } \bigcup , \bigstar \bigstar \bigstar \xrightarrow { \mathrm { h a s ~ d a u g h t e r } } \bigstar \bigcirc } \\ & { \qquad ( \triangle  \mathrm { A l i c e } ) , ( \bigstar \bigstar \mathrm { B e t h } ) , } \\ & { \qquad ( \bigstar \bigstar \mathrm { B e t h } ) , ( \bigcirc  \mathrm { C a r o l } ) . } \end{array}\tag{6}
$$

This account explains all three of our observations (Fig. 2F). Hereafter, we use the term structural role to refer to an entity’s position within the overall relational structure.

To test the generality of this representational organization, we performed analogous mosaic experiments across relational reasoning settings and models. Similar patterns held across every setting (Figs. S22–S35) and model (Figs. S78–S88) tested, provided the model could perform the underlying relational reasoning task.

## D. Representations of entity identity and relational structure combine additively.

We next asked how an entity name is bound to its corresponding structural role. One natural hypothesis is that the binding is mediated by the co-occurrence, within the same row of A, of a pattern of activity representing an entity’s name and a pattern of activity representing its structural role. Under the simplest version of this hypothesis, which we call the parallelogram hypothesis, A can be written as the sum of a name-independent component carrying the relational structure and a namedependent component carrying entity identity (Fig. 3A, left). A row of A containing name n bound to structural role $s ,$ denoted $\mathbf { a } ( n , s )$ , would decompose as:

$$
\mathbf { a } ( n , s ) \approx \phi _ { \mathrm { n a m e } } ( n ) + \phi _ { \mathrm { r o l e } } ( s ) ,\tag{7}
$$

![](images/14ec6c92a4d06a81a924db99fb70a87e8679b2146a5bc4b0a36dc651b4322447.jpg)  
FIG. 3. Entity identity and structural role combine additively. A, Additive encoding hypothesis. The activation tensor decomposes into a name-independent component (pink) carrying the relational structure and a name-dependent component (teal) carrying entity identity (left). Geometrically, the representations obtained by pairing any two entity names with any two structural roles form a parallelogram (right). B, Schematic of a mosaic experiment to test this hypothesis. Two donor tensors place the same entity name in diferent structural roles; subtracting the corresponding rows provides an estimate of the role-swap vector δ. Adding this vector to one entity row and subtracting it from another in a control tensor produces the mosaic. C, Predicted outcome under additive encoding. The mosaic should exchange the roles of two control entities. D, Outcome of the experiment. Each plot compares model responses for the control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in Fig. 1D. Fifty trials are shown per condition. Names in the query headings are illustrative. ‹A›, ‹B›, and ‹C› denote the three entity names in the control context in order of first appearance. Results are shown for Gemma-3-27B in the family-tree setting; corresponding analyses for other models and relational-reasoning settings are reported in the SI.

where $\phi _ { \mathrm { n a m e } }$ and $\phi _ { \mathrm { r o l e } }$ map entity names and structural roles, respectively, to their corresponding patterns of activity. This hypothesis is equivalent to the geometric statement that the four representations corresponding to any two structural roles and two entity names form a parallelogram (Fig. 3A, right). Analogous representational geometries underlie classical models of analogical reasoning [33] and have been observed in learned word embeddings [34]. The parallelogram hypothesis predicts that the change in activity associated with changing an entity’s structural role should not depend on the name of that entity:

$$
\mathbf { a } ( n , s _ { 2 } ) - \mathbf { a } ( n , s _ { 1 } ) \approx \phi _ { \mathrm { r o l e } } ( s _ { 2 } ) - \phi _ { \mathrm { r o l e } } ( s _ { 1 } ) \equiv \delta _ { s _ { 1 } \to s _ { 2 } } .\tag{8}
$$

We refer to $\delta _ { s _ { 1 }  s _ { 2 } }$ as a role-swap vector, because adding and subtracting $\delta _ { s _ { 1 }  s _ { 2 } }$ to and from the representations of two entities with roles $s _ { 1 }$ and $s _ { 2 }$ is predicted to make models behave as if the two entities had exchanged roles.

Fig. 3B illustrates the design of a mosaic experiment that tests this prediction. We first recorded two donor tensors, $\mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 1 ) }$ and $\mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 2 ) } ,$ from two prompts that placed the same entity in two diferent structural roles. If name and role combine additively, subtracting the corresponding entity rows in $\mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 1 ) }$ and $\mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 2 ) }$ cancels their shared name component, yielding an estimate of the role-swap vector. To build the mosaic tensor $\mathbf { A } _ { \mathrm { m o s a i c } } .$ we added and subtracted the estimated roleswap vector to and from the relevant rows of a control tensor, A<sub>control</sub>, recorded from a prompt containing a different set of entity names. The parallelogram hypothesis predicts that this operation should exchange the roledependent components of the two rows while leaving their name-dependent components unchanged, causing models to recover relationships from $\mathbf { A } _ { \mathrm { m o s a i c } }$ as if the two control entities had exchanged roles (Fig. 3C).

Models’ responses were consistent with this prediction (Fig. 3D). Similar patterns held across most relational-reasoning settings (Figs. S36–S49) and mod els (Figs. S89–S99) that we tested. These observations demonstrate that the activation tensor decomposes lin early into a name-independent component carrying the relational structure and a name-dependent component carrying the names of the entities that instantiate it.

## E. Multi-step inference relies primarily on a graph representation of observed relationships.

We next asked how relational information is organized within the name-independent component of A. We considered two hypotheses:

Under the retrospective–composition hypothesis, the name-independent component of A represents the relational structure as a labeled, directed graph (Fig. 4A, left). Structural roles correspond to nodes in this graph, and each observed relationship corresponds to an edge between two nodes. Because multi-step relationships are represented implicitly as paths through the graph, this representational strategy requires models to retrospectively compose the relevant relationships once a query is asked. This computation is related to classical accounts of model-based planning over an internal world model [35–37].

![](images/1e04399fefd7451d9a7ec11723d5b4b4a39df6bdf0e19f3f6ee3444b5880f77e.jpg)  
FIG. 4. A mosaic experiment distinguishes retrospective and prospective composition of relationships. A, Competing hypotheses for the representation of relational structure in the activation tensor. Relational structure could be represented as a graph of observed relations that are composed retrospectively when a query is asked (left), or as a set of relational “locations” that reflect prospective composition of relationships (right). B, Schematic of a mosaic experiment that tests these hypotheses. The control and donor prompts difer only in the identity of the first relationship. Rows from the start of the second relationship phrase through the final entity in the control tensor are replaced with the corresponding donor rows. C, Predicted outcome under each hypothesis. D, Outcome of the experiment. Each plot compares model responses for the control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in Fig. 1D. Fifty trials are shown per condition. Names in the query headings are illustrative. Results are shown for Gemma-3-27B in the family-tree setting; corresponding analyses for other models and relational-reasoning settings are reported in the SI.

An alternative hypothesis, the prospective–composition hypothesis, is that models prospectively compose relationships as they are encountered so that the nameindependent activity in A associated with each entity reflects the full sequence of relationships leading to it (Fig. 4A, right). One implementation would represent structural roles as locations in an internal coordinate system and relationships as transformations between those locations. Successively applying these transformations as relationships are encountered would determine the location of each new entity. If the transformations compose in the same way as the corresponding relationships, then multi-step relationships would be automatically reflected in the locations and would be recoverable at query time without retracing the intermediate relationships. This computation is directly analogous to path integration in models of animal spatial navigation [5, 38]; related schemes have also been proposed for representing small family trees [39].

To distinguish between these possibilities, we designed a mosaic experiment to test whether or not the taskrelevant name-independent activity associated with the final entity is dependent on the full history of preced ing relationships (Fig. 4B). In this experiment, $\mathbf { A } _ { \mathrm { m o s a i c } }$ was constructed from two source tensors, $\mathbf { A } _ { \mathrm { c o n t r o l } }$ and $\mathbf { A } _ { \mathrm { d o n o r } } ,$ recorded as models processed two prompts that difered only in the identity of the first relationship. For example, the control and donor contexts could read “Alice has daughter Beth; Beth has daughter Carol” and “Alice has mother Beth; Beth has daughter Carol”, respectively. We then constructed $\mathbf { A } _ { \mathrm { m o s a i c } }$ by replacing the rows associated with the final entity in $\mathbf { A } _ { \mathrm { c o n t r o l } }$ with the corresponding rows from $\mathbf { A } _ { \mathrm { d o n o r } } \ ( \mathrm { F i g . \ 4 B } )$ .

Under the retrospective–composition hypothesis, the substituted rows encode only the relationship that directly introduces the final entity. Thus, at query time, models should compose the first relationship from the control prompt with the second relationship from the donor prompt; the multi-step relationships supported by $\mathbf { A } _ { \mathrm { m o s a i c } }$ should therefore match those implied by the control prompt (Fig. 4C, left). In contrast, under the prospective–composition hypothesis, the substituted rows reflect the full history of preceding relationships and encode the final entity’s location as specified by the donor prompt. The multi-step relationships supported by $\mathbf { A } _ { \mathrm { m o s a i c } }$ should therefore match those implied by the donor prompt (Fig. 4C, right).

Consistent with the retrospective–composition hypothesis, we found that the multi-step relationships that models recovered from $\mathbf { A } _ { \mathrm { m o s a i c } }$ generally matched those implied by the control prompt (Fig. 4D). Similar patterns held across relational reasoning tasks (Figs. S50–S63) and models (Figs. S100–S110). However, in a subset of conditions, the models’ responses deviated systematically from this pattern, shifting toward—and in some cases closely matching—the responses implied by the donor prompt (Figs. S3, S104D). These deviations were most apparent in the family-tree setting.

Overall, these observations suggest that models rely primarily on a graph-like representation of the observed relationships in A and compose the relevant relationships retrospectively when a query is asked. However, in certain settings, models may also rely on other sources of information in A that reflect the prospective composition of observed relationships. Consistent with this possibility, directly visualizing representations at diferent token positions revealed a history-dependent component at relation tokens (Fig. S4). Why models rely on such information for certain queries but not others remains unclear. One possibility is that this variation across queries reflects some statistical feature of the training data, as suggested for other behavioral idiosyncrasies in language models [40].

## F. Separate sets of key–value pairs support name-to-role and role-to-name retrieval.

Our preceding findings suggested that A decomposes into two components, one representing entity identity and the other structural role. We next asked how models use each kind of information to access the other. To address this question, we studied the organization of information in the key–value memory M, from which the models directly retrieve information (Fig. 5A).

Recall that M consists of a large set of key–value pairs constructed from A via linear projection. Together with our finding that representations of name and role combine linearly in A, this suggests a natural hypothesis for how information is organized in M to support retrieval of an entity’s role from its name and vice versa. Specifically, certain linear projections could produce key–value pairs in M for which the key carries information about an entity’s name and the value carries information about its role, supporting retrieval of an entity’s role from its name. We refer to these hypothesized key–value pairs as name-to-role pairs. A separate set of role-to-name pairs with role information in the keys and name information in the values could support retrieval in the reverse direction (Fig. 5B).

The hypothesis that the model relies on these two kinds of key–value pairs to answer queries makes two key predictions. First, only certain entries in M should need to carry information about the names in the prompt. Specifically, name-dependent information is required in the keys but not in the values of name-to-role pairs; the reverse is true of role-to-name pairs. Second, name-torole pairs should be accessed before role-to-name pairs. For example, to answer “Who is Beth’s mother?”, models must first retrieve Beth’s role from her name, then identify her mother’s role and retrieve the name bound to it.

To test these predictions, we constructed mosaic key–value memories designed to restrict where namedependent information was available to the model. To do this, we replaced selected keys or values with their averages across prompts where the names had been resampled. In $\mathbf { M } _ { \mathrm { k e y s - f i r s t } }$ , name-dependent information was available only in the keys from the first half of M and in the values from the second half of M; $\mathbf { M } _ { \mathrm { v a l u e s - f i r s t } }$ used the reverse arrangement (Fig. 5C). Our hypothesis predicts that models should be able to answer queries using $\mathbf { M } _ { \mathrm { k e y s - f i r s t } } ,$ but not $\mathbf { M } _ { \mathrm { v a l u e s - f i r s t } } .$

Consistent with this prediction, substituting $\mathbf { M } _ { \mathrm { k e y s - f i r s t } }$ for the models’ key–value memory preserved performance, whereas substituting $\mathbf { M } _ { \mathrm { v a l u e s } }$ −first did not (Fig. 5D). Similar patterns held across relational reasoning tasks (Figs. S64A–S77A) and models (Figs. S111A–S121A). These observations establish the existence of a set of name-to-role and role-to-name pairs suficient to support the models’ behavior.

We next sought to localize these pairs within M. Starting from $\mathbf { M } _ { \mathrm { k e y s - f i r s t } }$ , we greedily averaged out namedependent information from the remaining keys and values, selecting at each step the replacement estimated to have the smallest efect on behavior (Fig. S5A,B). The resulting mosaic preserved the models’ behavior while confining name dependence to a very sparse subset of key– value pairs (Fig. 5E,F), distributed broadly across layers (Fig. S5C). By construction, the surviving pairs retained name-dependent information only in their keys or only in their values, making them candidates for name-to-role pairs and role-to-name pairs, respectively.

![](images/7d6d42abd37d850ffd60717ad98b8de7893f0450f90b37a04d79ddbbda7c1873.jpg)  
FIG. 5. Identifying and editing key–value pairs that link entity names and structural roles. A, Key–value memory and information flow in a transformer. Linear projections transform vectors in the activation tensor into keys (green) and values (orange), which together form the key–value memory M. For simplicity, only one key–value pair is shown at each layer and token position. Vectors at subsequent token positions generate queries that are compared with the keys (upward dotted arrows) to determine which values are retrieved (downward solid arrows). B, Hypothesized key–value pairs supporting name-to-role and role-to-name retrieval. C, Keys-first and values-first mosaic memories. Grey boxes indicate keys or values replaced with averages across prompts with the same relational structure but resampled entity names. D, Model behavior with the control memory (red), the keys-first (green), and the values-first (orange) mosaics. E, Mosaic memory retaining name-dependent information in a sparse set of candidate key–value pairs identified by iterative pruning. F, Model behavior with the control memory (red) and sparse mosaic memory (purple). G, Selective editing of a role-to-name association. Values of candidate role-to-name pairs associated with a particular entity in the control memory are replaced with their counterparts from a donor memory (left). This is predicted to change the model’s answer to a query about a relationship while preserving its answer to the inverse query (right). H, Model behavior with the control memory (red) and edited mosaic memory (purple). In D, F, and H, results are shown for Gemma-3-27B in the family-tree setting. Fifty trials are shown per condition. Names in the query headings are illustrative. $\langle \mathrm { A } \rangle , \langle \mathrm { B } \rangle ,$ and ‹C› denote the three entity names in the control context in order of first appearance; ‹D› denotes the first name in the donor context. Probability ratios are defined as in Fig. 1D. Corresponding analyses for other models and relational-reasoning settings are reported in the SI.

Finally we asked if, by modifying these candidate pairs, we could plant specific “false memories” in M. If the candidate pairs mediate name–role binding as we have proposed, then answering a particular query (e.g., “Who is Alice’s daughter?”) and its inverse $( \mathrm { e . g . }$ , “Who is Beth’s mother?”) should rely on disjoint subsets of the candidate pairs. It should therefore be possible to plant a “unidirectional false memory” in M that is expressed only when a relationship is queried in a particular direction.

We demonstrated this phenomenon in the control– donor paradigm from Fig. 2. We began by recording two source memories, $\mathbf { M } _ { \mathrm { c o n t r o l } }$ and $\mathbf { M } _ { \mathrm { d o n o r } }$ , as models processed two prompts with the same relational structure but diferent entity names. We then edited the candidate role-to-name pairs associated with “Alice” by replacing their values in $\mathbf { M } _ { \mathrm { c o n t r o l } }$ with the corresponding values from $\mathbf { M } _ { \mathrm { d o n o r } } ,$ leaving all other entries unchanged (Fig. 5G). We found that under this manipulation, models continued to identify Beth as Alice’s daughter, but identified Daisy, rather than Alice, as Beth’s mother (Fig. 5H). Thus, we were able to plant “unidirectional false memories” in M: the relationships recovered by the models under these manipulations could not be reconciled with a single, logically consistent family tree. Related experiments showed that editing candidate nameto-role pairs altered models’ responses across relational reasoning tasks (Figs. S64–S77, panels D,E) and models (Figs. S111–S121, panels D,E).

## III. DISCUSSION

Taken together, our results demonstrate that relational information in the model’s internal state is organized into linearly separable components representing relational structure and entity identity. The structural component encodes the observed relationships as a graph; edges of this graph are composed at query time to infer unstated relationships. A set of key–value pairs binds the structural component to representations of the entity names in the current context, allowing the same computations to be reused across diferent sets of entities.

This representational organization provides one realization of the longstanding idea that knowledge can be organized into reusable structures that are later instantiated with the particulars of current experience [1– 3, 41]. In our setting, such knowledge may be acquired during training through repeated exposure to examples that share the same relational structures, then instantiated in the name-independent component of the activation tensor when an appropriate prompt is presented. The separation of this structural component from namedependent information accounts for the systematicity of inference [20] that we observed here. More generally, our results add to a growing body of evidence [22, 28–32, 42] indicating that neural networks can learn to carry out abstract computations commensurate with the capacity for symbolic processing. Crucially, in contrast to certain neuro-symbolic approaches that achieve abstract reasoning by building in symbolic structure by design [43–45], the representational organization we describe arises without the need to prespecify any primitive operations or representations that are recognizably symbolic.

What conditions, then, favor the emergence of this organization? Given the minimal set of operations built into the transformer architecture, our findings suggest that the following ingredients may be suficient: first, a data distribution exhibiting certain symmetries (e.g., regularities in language that hold across changes in entity identity); second, a mechanism for flexibly binding together otherwise distinct representations to form associations relevant to the current context (e.g., the key– value structure of self-attention); and third, a learning mechanism driven by prediction error on these data (e.g., gradient descent on a next-token prediction objective).

In the mammalian hippocampus and nearby structures, striking patterns of neural activity observed during spatial navigation [4, 46, 47] and related tasks [48, 49] have long been interpreted as signatures of a cognitive map. It has been proposed that these patterns of activity are consistent with a map that separates the representation of structural and sensory information [5–7]. Similar ingredients may be at work here as in language models: spatial navigation involves exploiting geometric regularities that recur across environments, and the hippocam pus has been proposed to supply a mechanism for rapidly and flexibly binding together information from distinct processing systems [50–52]. Taken together, these parallels may provide another example of how artificial and biological systems, despite profound diferences in their physical substrates, nonetheless converge on similar computational principles.

Like many other capabilities of language models [16, 17], performance on our relational reasoning tasks improved markedly with model size. Beyond increasing model size, reasoning performance can also be improved by allowing models to generate intermediate reasoning tokens [53, 54]. What accounts for these improvements? If inference resembles retrospective search over a graph-like representation, both could provide greater capacity for the intermediate computations required for such a search. Beyond this, reasoning may also change the representation over which inference is performed. Here, we have treated the intermediate states produced as the model processes the prompt as the substrate of the cognitive map, because these states make information in the context available to influence later processing. Crucially, the same is true of the intermediate states produced as the model generates reasoning tokens. Therefore, once models are allowed to reason, the distinction between constructing a cognitive map and performing inference over it becomes blurred: models may dynamically refine and restructure the cognitive map as inference unfolds.

## CODE AVAILABILITY

Code and data for reproducing the results presented in this article are available at https://github.com/ albertqin8221/cognitive-maps-in-llms/releases.

## ACKNOWLEDGMENTS

We thank Tim Buschman, Tim Lin, Keith Murray, and the members of the Reddy Lab for helpful discussions. AQ was supported by a fellowship from the Fannie and John Hertz Foundation. GR was partially supported by a joint research agreement between NTT Research Inc. and Princeton University, a grant from Coeficient Giving, and a seed grant from the Princeton AI Lab. The analyses presented in this article were performed on Della, a computing cluster managed and supported by Princeton Research Computing, a consortium of groups including the Princeton Institute for Computational Science and Engineering (PICSciE) and Research Computing at Princeton University.

[1] E. C. Tolman, Psychological Review 55, 189 (1948).

[2] F. C. Bartlett, Remembering: A Study in Experimental and Social Psychology (Cambridge University Press, Cambridge, UK, 1932).

[3] K. J. W. Craik, The Nature of Explanation (Cambridge University Press, Cambridge, UK, 1943).

[4] J. O’Keefe and L. Nadel, The Hippocampus as a Cognitive Map (Clarendon Press, Oxford, 1978).

[5] J. C. R. Whittington, T. H. Muller, S. Mark, G. Chen, C. Barry, N. Burgess, and T. E. J. Behrens, Cell 183, 1249 (2020).

[6] S. Chandra, S. Sharma, R. Chaudhuri, and I. Fiete, Nature 638, 739 (2025).

[7] C. J. Kymn, S. Mazelet, A. Thomas, D. Kleyko, E. P. Frady, F. T. Sommer, and B. A. Olshausen, in Advances in Neural Information Processing Systems, Vol. 37 (2024) pp. 39128–39157.

[8] C. Kemp and J. B. Tenenbaum, Proceedings of the National Academy of Sciences 105, 10687 (2008).

[9] D. E. Rumelhart and P. M. Todd, in Attention and Performance XIV: Synergies in Experimental Psychology, Artificial Intelligence, and Cognitive Neuroscience, edited by D. E. Meyer and S. Kornblum (MIT Press, Cambridge, MA, 1993) pp. 3–30.

[10] D. E. Rumelhart, in Theoretical Issues in Reading Comprehension: Perspectives from Cognitive Psychology, Linguistics, Artificial Intelligence, and Education, edited by R. J. Spiro, B. C. Bruce, and W. F. Brewer (Lawrence Erlbaum Associates, Hillsdale, NJ, 1980) pp. 33–58.

[11] D. George, R. V. Rikhye, N. Gothoskar, J. S. Guntupalli, A. Dedieu, and M. Lázaro-Gredilla, Nature Communications 12, 2392 (2021).

[12] T. W. Webb, I. Sinha, and J. D. Cohen, in International Conference on Learning Representations (2021) arXiv:2012.14601 [cs.AI].

[13] T. W. Webb, S. M. Frankland, A. Altabaa, S. Segert, K. Krishnamurthy, D. Campbell, J. Russin, T. Giallanza, R. O’Reilly, J. Laferty, and J. D. Cohen, Trends in Cognitive Sciences 28, 829 (2024).

[14] T. L. Grifiths, B. M. Lake, R. T. McCoy, E. Pavlick, and T. W. Webb, Trends in Cognitive Sciences 10.1016/j.tics.2026.02.003 (2026), in press.

[15] T. B. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. M. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei, in Advances in Neural Information Processing Systems, Vol. 33 (Curran Associates, Inc., 2020) pp. 1877–1901, arXiv:2005.14165 [cs.CL].

[16] J. Wei, Y. Tay, R. Bommasani, C. Rafel, B. Zoph, S. Borgeaud, D. Yogatama, M. Bosma, D. Zhou, D. Metzler, E. H. Chi, T. Hashimoto, O. Vinyals, P. Liang, J. Dean, and W. Fedus, Transactions on Machine Learning Research (2022).

[17] A. Srivastava, A. Rastogi, A. Rao, et al., Transactions on Machine Learning Research (2023).

[18] C. F. Baker, C. J. Fillmore, and J. B. Lowe, in 36th Annual Meeting of the Association for Computational Lin-

guistics and 17th International Conference on Computational Linguistics, Volume 1 (Association for Computational Linguistics, Montreal, Quebec, Canada, 1998) pp. 86–90.

[19] N. Chambers and D. Jurafsky, in Proceedings of the Joint Conference of the 47th Annual Meeting of the ACL and the 4th International Joint Conference on Natural Language Processing of the AFNLP, edited by K.-Y. Su, J. Su, J. Wiebe, and H. Li (Association for Computational Linguistics, Suntec, Singapore, 2009) pp. 602–610.

[20] J. A. Fodor and Z. W. Pylyshyn, Cognition 28, 3 (1988).

[21] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, in Advances in Neural Information Processing Systems, Vol. 30, edited by I. Guyon, U. von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. V. N. Vishwanathan, and R. Garnett (Curran Associates, Inc., 2017) pp. 5998– 6008, arXiv:1706.03762 [cs.CL].

[22] J. Feng and J. Steinhardt, in International Conference on Learning Representations, Vol. 2024 (2024) pp. 36391– 36413.

[23] K. Meng, D. Bau, A. Andonian, and Y. Belinkov, in Advances in Neural Information Processing Systems, Vol. 35 (Curran Associates, Inc., 2022) pp. 17359–17372.

[24] E. Hernandez, A. S. Sharma, T. Haklay, K. Meng, M. Wattenberg, J. Andreas, Y. Belinkov, and D. Bau, Linearity of relation decoding in transformer language models (2023), arXiv:2308.09124 [cs.CL].

[25] M. Geva, J. Bastings, K. Filippova, and A. Globerson, in Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics, Singapore, 2023) pp. 12216–12235.

[26] O. Barak, M. Rigotti, and S. Fusi, Journal of Neuroscience 33, 3844 (2013).

[27] S. M. Frankland, R. Marjieh, M. Nurisso, J. Fluegemann, T. W. Webb, G. Petri, R. L. Lewis, and J. D. Cohen, PsyArXiv (2026), preprint, version 5.

[28] N. Prakash, T. Shaham, T. Haklay, Y. Belinkov, and D. Bau, in International Conference on Learning Representations, Vol. 2024 (2024) pp. 8057–8082.

[29] Y. Yang, D. I. Campbell, K. Huang, M. Wang, J. D. Cohen, and T. W. Webb, in Proceedings of the 42nd International Conference on Machine Learning, Proceedings of Machine Learning Research, Vol. 267 (PMLR, 2025) pp. 70515–70549.

[30] R. Assouel, D. Campbell, Y. Bengio, and T. Webb, Visual symbolic mechanisms: Emergent symbol processing in vision language models (2025), arXiv:2506.15871 [cs.CV].

[31] K. Cui, N. Prakash, S. Messica, A. Raina, D. Bau, A. Torralba, and T. Rott Shaham, The dual mechanisms of spatial variable binding in vision-language models (2026), arXiv:2603.22278 [cs.CV].

[32] N. Prakash, N. Shapira, A. S. Sharma, C. Riedl, Y. Belinkov, T. Rott Shaham, D. Bau, and A. Geiger, Language models use lookbacks to track beliefs (2025), arXiv:2505.14685 [cs.CL].

[33] D. E. Rumelhart and A. A. Abrahamson, Cognitive Psychology 5, 1 (1973).

[34] T. Mikolov, W.-t. Yih, and G. Zweig, in Proceedings of the 2013 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Association for Computational Linguistics, Atlanta, Georgia, 2013) pp. 746–751.

[35] C. E. Shannon, The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science 41, 256 (1950).

[36] A. D. de Groot, Thought and Choice in Chess (Mouton, The Hague, 1965) originally published in Dutch as Het denken van den schaker in 1946.

[37] M. G. Mattar and N. D. Daw, Annual Review of Neuroscience 49, 435 (2026).

[38] Y. Burak and I. R. Fiete, PLoS Computational Biology 5, e1000291 (2009).

[39] A. Paccanaro and G. E. Hinton, in Advances in Neural Information Processing Systems, Vol. 14, edited by T. G. Dietterich, S. Becker, and Z. Ghahramani (MIT Press, 2001) pp. 857–864.

[40] R. T. McCoy, S. Yao, D. Friedman, M. D. Hardy, and T. L. Grifiths, Proceedings of the National Academy of Sciences 121, e2322420121 (2024).

[41] T. E. J. Behrens, T. H. Muller, J. C. R. Whittington, S. Mark, A. B. Baram, K. L. Stachenfeld, and Z. Kurth-Nelson, Neuron 100, 490 (2018).

[42] D. Campbell, S. Kumar, T. Giallanza, T. L. Grifiths, and J. D. Cohen, Human-like geometric abstraction in large pre-trained neural networks (2024), arXiv:2402.04203 [cs.AI].

[43] J. Mao, C. Gan, P. Kohli, J. B. Tenenbaum, and J. Wu, in International Conference on Learning Representations (2019).

[44] K. Ellis, C. Wong, M. Nye, M. Sablé-Meyer, L. Morales, L. Hewitt, L. Cary, A. Solar-Lezama, and J. B. Tenenbaum, in Proceedings of the 42nd ACM SIGPLAN Inter-

national Conference on Programming Language Design and Implementation (Association for Computing Machinery, 2021) pp. 835–850.

[45] R. Manhaeve, S. Dumančić, A. Kimmig, T. Demeester, and L. De Raedt, in Advances in Neural Information Processing Systems, Vol. 31 (Curran Associates, Inc., 2018).

[46] J. O’Keefe and J. Dostrovsky, Brain Research 34, 171 (1971).

[47] T. Hafting, M. Fyhn, S. Molden, M.-B. Moser, and E. I. Moser, Nature 436, 801 (2005).

[48] S. Neupane, I. Fiete, and M. Jazayeri, Nature 630, 704 (2024).

[49] A. O. Constantinescu, J. X. O’Reilly, and T. E. J. Behrens, Science 352, 1464 (2016).

[50] N. J. Cohen and H. Eichenbaum, Memory, Amnesia, and the Hippocampal System (MIT Press, Cambridge, MA, 1993).

[51] T. J. Teyler and P. DiScenna, Behavioral Neuroscience 100, 147 (1986).

[52] S. J. Gershman, I. Fiete, and K. Irie, Neuron 113, 1694 (2025).

[53] M. Nye, A. J. Andreassen, G. Gur-Ari, H. Michalewski, J. Austin, D. Bieber, D. Dohan, A. Lewkowycz, M. Bosma, D. Luan, C. Sutton, and A. Odena, in ICLR 2022 Workshops: DL4C (2022).

[54] J. Wei, X. Wang, D. Schuurmans, M. Bosma, B. Ichter, F. Xia, E. H. Chi, Q. V. Le, and D. Zhou, in Advances in Neural Information Processing Systems, Vol. 35 (Curran Associates, Inc., 2022) pp. 24824–24837.

## Supplementary Information

## CONTENTS

I. Transformer Architecture and Notation 14   
II. Mosaic Experiments 16   
III. Detailed Methods 18   
IV. Tests of Position-Code Hypotheses 22   
V. Evidence for Prospective Composition of Relationships 24   
VI. Identifying Name-to-Role and Role-to-Name Pairs 26   
VII. Efects of Query Format 27   
VIII. Prompt Templates 30   
IX. Analogous Experiments Across Relational Reasoning Tasks 35   
X. Analogous Experiments Across Model Families and Scales 65   
References 85

## I. TRANSFORMER ARCHITECTURE AND NOTATION

Expanding on the discussion in the main text, we begin with a formal account of information flow through an autoregressive transformer [1] and introduce notation used throughout the Supplementary Information.

## A. General organization of a transformer

A transformer represents and processes text as a sequence of tokens. We denote the one-hot representation of a token at position t by $\mathbf { e } _ { t } \in \{ 0 , 1 \} ^ { N _ { \mathrm { v o c a b } } }$ , where $N _ { \mathrm { v o c a b } }$ is the vocabulary size. Tokens are mapped to vectors via an embedding matrix $\breve { W } _ { \mathrm { e m b e d } } \stackrel { \sim } { \in } \mathbb { R } ^ { \tilde { D } _ { \mathrm { m o d e l } } \times N _ { \mathrm { v o c a b } } } . ^ { 1 }$ Applying this matrix to a sequence of tokens produces an initial sequence of hidden vectors:

$$
\{ \mathbf { h } _ { t , 1 } \} _ { t = 1 } ^ { N _ { \mathrm { t o k e n s } } } = \{ W _ { \mathrm { e m b e d } } \mathbf { e } _ { t } \} _ { t = 1 } ^ { N _ { \mathrm { t o k e n s } } } .\tag{1}
$$

In general, we use $\mathbf { h } _ { t , l } \in \mathbb { R } ^ { D }$ <sup>model</sup> to denote the hidden vector at token position t and layer l.

Information processing in a transformer occurs by successively updating the sequence of hidden vectors through a stack of layers. Each layer applies two operations in succession.<sup>2</sup> The first operation is self-attention:

$$
\tilde { \mathbf { h } } _ { t , l } = \mathbf { h } _ { t , l } + \mathrm { S e l f A t t e n t i o n } _ { l } \left( \{ \mathbf { h } _ { s , l } \} _ { s = 1 } ^ { N _ { \mathrm { t o k e n s } } } \right) _ { t } .\tag{2}
$$

As we discuss below, self-attention permits the hidden vector at position t to retrieve information from hidden vectors at positions $s \leq t .$ The second operation is the feedforward operation:

$$
\mathbf { h } _ { t , l + 1 } = \tilde { \mathbf { h } } _ { t , l } + \mathrm { F e e d f o r w a r d } _ { l } ( \tilde { \mathbf { h } } _ { t , l } ) .\tag{3}
$$

The feedforward operation acts on the hidden vector at each token position independently. As these two operations are repeated across the stack of layers, the resulting hidden vectors $\mathbf { h } _ { t , l }$ become context-sensitive representations of the tokens at their corresponding positions t.

After the hidden vectors have passed through the final layer, the sequence of vectors is mapped back to the vocabulary via an unembedding matrix $W _ { \mathrm { u n e m b e d } } \in \mathbb { R } ^ { N _ { \mathrm { v o c a b } } \times D _ { \mathrm { m o d e l } } }$ . This produces a sequence of logit vectors:

$$
\left\{ \ell _ { t } \right\} _ { t = 1 } ^ { N _ { \mathrm { t o k e n s } } } = \left\{ W _ { \mathrm { u n e m b e d } } \mathbf { h } _ { t , N _ { \mathrm { l a y e r s } } + 1 } \right\} _ { t = 1 } ^ { N _ { \mathrm { t o k e n s } } } .\tag{4}
$$

Each logit vector $\boldsymbol { \ell } _ { t } \in \mathbb { R } ^ { N _ { \mathrm { v o c a b } } }$ determines the model’s predicted distribution over the next token:

$$
p _ { t } ( a ) = \frac { \exp ( \ell _ { t , a } ) } { \sum _ { b = 1 } ^ { N _ { \mathrm { v o c a b } } } \exp ( \ell _ { t , b } ) } ,\tag{5}
$$

where $p _ { t } ( a )$ is the predicted probability of token a following position t. Consequently, diferences between logits equal log probability ratios.

## B. Self-attention

We next define the self-attention operation. Self-attention operates on a sequence of hidden vectors $\{ \mathbf { h } _ { t } \} _ { t = 1 } ^ { N _ { \mathrm { t o k e n s } } }$ and allows each vector to retrieve information from preceding vectors. Its output is:<sup>3</sup>

$$
\mathrm { S e l f A t t e n t i o n } ( \{ { \bf h } _ { s } \} ) _ { t } = \sum _ { i = 1 } ^ { N _ { \mathrm { h e a d s } } } \sum _ { s \leq t } \underbrace { A _ { t  s } ^ { ( i ) } } _ { \mathrm { { r e t r i e v a l } } } \cdot \underbrace { W _ { O } ^ { ( i ) } W _ { V } ^ { ( i ) } { \bf h } _ { s } } _ { \mathrm { { r e t r i e v e d } } } .\tag{6}
$$

As observed in prior work [2], self-attention can be viewed as a set of parallel key–value memory systems, each associated with an “attention head” (here indexed by $i = 1 , \dots , N _ { \mathrm { h e a d s } } )$ . The characteristic feature of a key–value memory system is a separation between what information is retrieved and where that information is retrieved from. Here, the product of the two matrices $W _ { O } ^ { ( i ) } \in \mathbb { R } ^ { D _ { \mathrm { m o d e l } } \times D _ { \mathrm { h e a d } } }$ and $W _ { V } ^ { ( i ) } \in \mathbb { R } ^ { D _ { \mathrm { h e a d } } \times D _ { \mathrm { m o d e l } } }$ determines what information can be retrieved from each position.<sup>4</sup> The source of the retrieved information is determined by the lower-triangular entries of the attention weight matrix $A _ { t  s } ^ { ( i ) }$ , which specify, for each position t, how strongly each previous position s is accessed.

Each entry of $A _ { t  s } ^ { ( i ) }$ is computed from the hidden vectors $\mathbf { h } _ { s }$ and $\mathbf { h } _ { t }$ by applying separate linear transformations to generate a key from $\mathbf { h } _ { s }$ and a query from $\mathbf { h } _ { t } ,$ taking their dot product, and applying a nonlinearity:

$$
A _ { t  s } ^ { ( i ) } = \sigma (  { \underset { \mathrm { k e y } } { \underbrace { W _ { K } ^ { ( i ) } \mathbf { h } _ { s } } } , \underset { \mathrm { q u e r y } } { \underbrace { W _ { Q } ^ { ( i ) } \mathbf { h } _ { t } } } }  ) .\tag{7}
$$

Here, $W _ { K } ^ { \left( i \right) } , W _ { Q } ^ { \left( i \right) } \in \mathbb { R } ^ { D _ { \mathrm { h e a d } } \times D _ { \mathrm { m o d e l } } }$ . Transformers use a scaled softmax for the nonlinearity:

$$
\sigma ( z _ { t s } ) = \frac { \exp ( { z _ { t s } / \xi } ) } { \sum _ { s ^ { \prime } \leq t } \exp ( { z _ { t s ^ { \prime } } / \xi } ) } ,\tag{8}
$$

where $\xi$ is a scaling factor of order $\sqrt { D _ { \mathrm { h e a d } } }$

Notice that the information retrieved from previous token positions depends on the hidden vectors $h _ { t }$ only through the $D _ { \mathrm { h e a d } }$ -dimensional keys and values:

$$
\mathbf { k } _ { t } ^ { ( i ) } = W _ { K } ^ { ( i ) } \mathbf { h } _ { t } , \qquad \mathbf { v } _ { t } ^ { ( i ) } = W _ { V } ^ { ( i ) } \mathbf { h } _ { t } .\tag{9}
$$

In practice, when sampling autoregressively from a transformer, the state retained between successive token-generation steps is the complete collection of keys and values, referred to as the key–value cache. In the main text, we call this object the key–value memory to emphasize its functional role as the stored information that later tokens can access through attention.

## C. Feedforward processing

For our purposes, only two aspects of the feedforward operation are salient: (1) it acts independently at each token position, and (2) it implements a highly nonlinear transformation containing a large fraction of the model’s parameters. For the models we studied here [3–5], the feedforward operation happens to be a gated multilayer perceptron:

$$
\mathrm { F e e d f o r w a r d } ( \mathbf { h } _ { t } ) = { W } _ { \mathrm { d o w n } } \left[ \phi ( { W } _ { \mathrm { g a t e } } \mathbf { h } _ { t } ) \odot ( { W } _ { \mathrm { u p } } \mathbf { h } _ { t } ) \right] ,\tag{10}
$$

where $W _ { \mathrm { g a t e } } , W _ { \mathrm { u p } } \in \mathbb R ^ { D _ { \mathrm { f e e d f o r w a r d } } \times D _ { \mathrm { m o d e l } } }$ and $W _ { \mathrm { d o w n } } \in \mathbb { R } ^ { D _ { \mathrm { m o d e l } } \times D _ { \mathrm { f e e d f o r w a r d } } } . ^ { 5 }$ Here, ⊙ denotes element-wise multiplication and $\phi$ is a gating nonlinearity.

## D. Architectural variations

Our description above omits several architectural details of the models we studied [3–5]. For completeness, we briefly describe some of these details, though they are not central to the analyses presented in this study.

First, in the models we study, positional information is supplied by rotary positional embeddings [6], which apply position-dependent rotations to the queries and keys:

$$
\widetilde { \mathbf { q } } _ { t } ^ { ( i ) } = R _ { t } \mathbf { q } _ { t } ^ { ( i ) } , \qquad \widetilde { \mathbf { k } } _ { t } ^ { ( i ) } = R _ { t } \mathbf { k } _ { t } ^ { ( i ) } .\tag{11}
$$

Here, each $R _ { t } \in \mathbb { R } ^ { D _ { \mathrm { h e a d } } \times D _ { \mathrm { h e a d } } }$ is a block diagonal rotation matrix. Attention weights are then computed from the rotated queries and keys.

Second, the self-attention and feedforward operations are interleaved with normalization steps that rescale the inputs to these operations. In some of the models we study, additional normalization is applied to the outputs of these operations, and/or to the query and key vectors before attention weights are computed.

Third, the models we study use grouped-query attention [7]. In this architecture, attention heads are divided into groups whose members share the same keys and values but generate distinct queries.

Fourth, some models use local attention in some layers, restricting information flow to a fixed window of preceding positions. Because all prompts we use in this study are shorter than this window, this detail does not afect our analyses.

## II. MOSAIC EXPERIMENTS

In this section, we describe the basic logic and structure of a mosaic experiment. We begin by re-defining the two objects of study introduced in the main text: the activation tensor A and the key–value memory M. We then formalize the account of information flow through these two objects developed in the main text by expressing the model’s computation as a composition of encoding, projection, and readout operations. We next explain how mosaic experiments allow us to study the organization of information in A and M. Finally, we describe some advantages of experiments of this form.

## A. Gross structure of information flow

We now give a perspective on the gross structure of information flow through an autoregressive transformer as it processes the prompts used in this study. To develop this perspective, we consider how information flows through the representations generated by the three segments of each prompt: the instructions, the relationship statements, and the query. Each segment generates a distinct subset of the hidden vectors $\mathbf { h } _ { t l }$ and key–value pairs $( \mathbf { k } _ { t l } ^ { ( i ) } , \mathbf { v } _ { t l } ^ { ( i ) } )$ , whose dependence on the other segments is constrained by the model’s causal architecture. These constraints allow us to associate each subset with a distinct computational role:

1. Hidden vectors and key–value pairs generated by the instruction segment (e.g., Read the context describing a series of relationships...) can depend on the instructions themselves, but not on the specific instance of the task introduced later in the prompt. Therefore, their natural computational role is to store information about the task and condition how subsequently presented information is processed.

2. Hidden vectors and key–value pairs generated by the relationship statements (e.g., Alice has daughter Beth. Beth has daughter Carol.) incorporate information about the specific task instance presented in the prompt. Their natural computational role is to carry this instance-specific information forward for use in subsequent computation. As in the main text, we refer to these collections of hidden vectors and key–value pairs as the activation tensor and key–value memory, respectively, and define them formally below.

3. Hidden vectors generated by the query segment (e.g., Question: Who is Carol’s grandmother? Answer:) depend on the query itself and, through the representations described above, on the preceding relationship statements. Their natural computational role is therefore to integrate information from the query with information retrieved from the relational context to compute an answer.

To formalize these ideas, we now write the model’s input–output map as the composition of encoding, projection, and readout operations. To begin, as in the main text, we define the activation tensor, A, as the collection of hidden vectors generated by the relationship statements:

$$
\mathbf { A } = \left\{ \mathbf { h } _ { t , l } \right\} _ { { t = t _ { \mathrm { s t a r t } } } , \ldots , { t _ { \mathrm { e n d } } } } \in \mathbb { R } ^ { T _ { \mathrm { r e l } } \times N _ { \mathrm { l a y e r s } } \times D _ { \mathrm { m o d e l } } } ,\tag{12}
$$

and we define the key–value memory, M, as the associated collection of key–value pairs<sup>6</sup>:

$$
{ \bf M } = \left\{ ( { \bf k } _ { t , l } ^ { ( i ) } , { \bf v } _ { t , l } ^ { ( i ) } ) \right\} _ { { t = t _ { \mathrm { s t a r t } } } , \dots , { t _ { \mathrm { e n d } } } \atop { i = 1 , \dots , N _ { \mathrm { l a y e r s } } } } \in \left( { \mathbb { R } } ^ { T _ { \mathrm { r e l } } \times N _ { \mathrm { l a y e r s } } \times N _ { \mathrm { h e a d s } } \times D _ { \mathrm { m o d e l } } } \right) ^ { 2 } .\tag{13}
$$

Here, $t _ { \mathrm { s t a r t } }$ and $t _ { \mathrm { e n d } }$ denote token positions corresponding to the beginning and end of the relationship statements<sup>7</sup>, respectively, and $T _ { \mathrm { r e l } } = t _ { \mathrm { e n d } } - t _ { \mathrm { s t a r t } } + 1$

The activation tensor and key–value memory serve as intermediate representations in a factorization of the model’s computation into encoding, projection, and readout operations. The activation tensor is a function only of the names $\mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C }$ and the relationships $\mathcal { R } _ { A B } , \mathcal { R } _ { B C }$ described by the context:

$$
\mathbf { A } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) ,
$$

$$
\Phi : \mathrm { N a m e s } ^ { 3 } \times \mathrm { R e l a t i o n s } ^ { 2 } \longrightarrow \mathbb { R } ^ { T _ { \mathrm { c o n t e x t } } \times N _ { \mathrm { l a y e r s } } \times D _ { \mathrm { m o d e l } } } .\tag{14}
$$

Here, Φ can be interpreted as an encoder that maps the relationship statements to an activation tensor.<sup>8</sup>

The key–value memory, M, is in turn a function of the activation tensor, A:

$$
\begin{array} { r l } & { \mathbf { M } = \Pi ( \mathbf { A } ) , } \\ & { \Pi : \mathbb { R } ^ { T _ { \mathrm { c o n t e x t } } \times N _ { \mathrm { l a y e r s } } \times D _ { \mathrm { m o d e l } } } \longrightarrow ( \mathbb { R } ^ { T _ { \mathrm { c o n t e x t } } \times N _ { \mathrm { l a y e r s } } \times N _ { \mathrm { h e a d s } } \times D _ { \mathrm { h e a d } } } ) ^ { 2 } , } \\ & { \mathbf { A } \mapsto \mathbf { M } = \Big \{ ( \mathbf { k } _ { t , l } ^ { ( i ) } , \mathbf { v } _ { t , l } ^ { ( i ) } ) \Big \} , \quad \mathbf { k } _ { t , l } ^ { ( i ) } = R _ { t } W _ { K } ^ { ( i , l ) } \mathbf { h } _ { t , l } , \ \mathbf { v } _ { t , l } ^ { ( i ) } = W _ { V } ^ { ( i , l ) } \mathbf { h } _ { t , l } . } \end{array}\tag{15}
$$

Here, Π can be interpreted as a projector that generates a key–value memory from the activation tensor. Π acts by applying the projection matrices ${ \cal W } _ { K } ^ { ( i , l ) }$ and $W _ { V } ^ { ( i , l ) }$ to generate key–value pairs, then applying the rotation matrices $R _ { t }$ to the keys to supply positional information.

Finally, the models’ response is a function of both the query segment and the relationship statements. The dependence on the latter is mediated entirely through M:

$$
\begin{array} { r l } & { \ell _ { a n s } = \Psi ( \boldsymbol { \mathcal { Q } } , \mathbf { M } ) , } \\ & { \quad \Psi : \mathrm { Q u e r i e s } \times ( \mathbb { R } ^ { T _ { \mathrm { c o n t e x t } } \times N _ { \mathrm { l a y e r s } } \times N _ { \mathrm { h e a d s } } \times D _ { \mathrm { h e a d } } } ) ^ { 2 } \longrightarrow \mathbb { R } ^ { N _ { \mathrm { v o c a b } } } . } \end{array}\tag{16}
$$

Here, Ψ can be interpreted as a readout operation that uses the key-value memory M to answer the query Q.

In summary, the models’ computation can be expressed as a composition of three operations: (1) encoding the relationship statements into the activation tensor $\mathbf { A } ; \mathbf { \Gamma } ( 2 )$ constructing the key–value memory M from A; and (3) using M to answer the query. More formally, writing the context as $\mathcal { C } = ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } )$ and the query as $\mathcal { Q } ,$ the model’s input–output map can be written in the form:

$$
F ( \mathcal { Q } , \mathcal { C } ) = \Psi \left( \mathcal { Q } , \Pi \left( \Phi ( \mathcal { C } ) \right) \right) .\tag{17}
$$

## B. Structure of a mosaic experiment

We now describe the structure of a mosaic experiment, the experimental approach used in this study to probe the organization of information in A and M. The basic logic of these experiments is to construct modified representations from those generated by the model, then use the model to read out information from the modified representations. Our approach builds on similar causal interventions used to study “binding $\mathrm { I D s } ^ { \mathrm { v } }$ [8]. For each experiment, we performed the following three steps:

1. We first recorded the representations generated by a set of source contexts $\mathcal { C } _ { 1 } , \mathcal { C } _ { 2 } , \ldots$ . Depending on the experiment, we recorded either activation tensors or key–value memories:

$$
\begin{array} { r l } & { \mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , \ldots = \Phi ( \mathcal { C } _ { 1 } ) , \Phi ( \mathcal { C } _ { 2 } ) , \ldots } \\ & { \mathbf { M } _ { 1 } , \mathbf { M } _ { 2 } , \ldots = \Pi ( \Phi ( \mathcal { C } _ { 1 } ) ) , \Pi ( \Phi ( \mathcal { C } _ { 2 } ) ) , \ldots . } \end{array}\tag{18}
$$

In practice, we evaluated $\Phi ( \cdot )$ by generating a prompt that instantiates each source context, then running a forward pass while recording the inputs to each layer at the relevant token positions. Similarly, we evaluated $\Pi ( \Phi ( \cdot ) )$ by running a forward pass, then recording the relevant entries of the key–value cache.

2. Next, we constructed a “mosaic representation” by recombining the recorded source representations:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { m o s a i c } } = f ( \mathbf { A } _ { 1 } , \mathbf { A } _ { 2 } , \dots ) } \\ { \mathbf { M } _ { \mathrm { m o s a i c } } = f ( \mathbf { M } _ { 1 } , \mathbf { M } _ { 2 } , \dots ) . } \end{array}\tag{19}
$$

The specific transformation $f ( \dots )$ is chosen to test a hypothesis about how information is organized within A or M.

3. Finally, we used the model to answer a query using the modified representation:

$$
\begin{array} { r l } & { \ell _ { \mathrm { a n s } } ^ { \prime } = \Psi ( \mathcal { Q } , \Pi ( \mathbf { A } _ { \mathrm { m o s a i c } } ) ) } \\ & { \ell _ { \mathrm { a n s } } ^ { \prime } = \Psi ( \mathcal { Q } , \mathbf { M } _ { \mathrm { m o s a i c } } ) . } \end{array}\tag{20}
$$

In practice, we evaluated $\Psi ( \cdot , \Pi ( \mathbf { A } ) )$ by running a forward pass on a prompt containing the query, while replacing the inputs to each layer at positions corresponding to the relationship statements with the hidden vectors h from A. Similarly, we evaluated Ψ(·, M) by appending M to the key–value cache for the instruction segment, then running the model on the query using the resulting cache.

## C. Advantages of experiments of this form

We wish to emphasize three advantages of experiments of this general form.

First, our approach is causal rather than correlational. Specifically, we directly manipulate the internal representations of the model, then use the model to read out the efect of these manipulations. Because the model itself is used as the “decoder”, the representational features identified by our experiments are task-relevant by construction. In comparison, fitting external encoding or decoding models to predict neural representations from task variables or vice versa can reveal the existence of certain representational features [9, 10], but does not directly demonstrate the causal relevance of these features for behavior.

Second, our approach is well suited to studying representational features distributed across many model components. In particular, because our manipulations extend across attention heads, token positions, and layers, we can test hypotheses about representational format without first isolating a sparse sub-circuit in the model specific to our task.

Third, our approach facilitates the study of representational format without the reconstruction of the circuits that produce or use it. Although we use the model both to produce source representations and to read out from mosaic representations, we need not understand how it implements these operations to study the organization of A and M. In practice, we have found that our approach is well suited to formulating specific hypotheses about representationa format, then designing experiments whose outcomes distinguish between those hypotheses.

## III. DETAILED METHODS

## A. General experimental procedures

For each model, prompt template, and experiment, we evaluated 50 randomly sampled entity assignments. Entity names were drawn from domain-appropriate pools, except in the experiments shown in Fig. 1, where we used random three-letter strings selected to tokenize as a single token when preceded by a space. Where source contexts required distinct control and donor entities, we sampled their names jointly without replacement. On each trial, we recorded the source representations, combined them to construct the mosaic, then read out responses to all queries specified by the task template. The figures show an informative subset of these queries; complete query results are provided in the accompanying data. We measured responses by extracting the probabilities (normalized over the full vocabulary) of the task’s answer tokens from the model’s next-token distribution. Each answer occupied a single token.

## B. Models and prompts

The models we studied are listed in Table I. The Hugging Face Transformers library was used to load and run the models in PyTorch. Models were loaded in bfloat16 precision.

<table><tr><td>Model</td><td>Hugging Face repository</td><td>Recorded revision</td></tr><tr><td>Gemma 3 4B</td><td>google/gemma-3-4b-it</td><td>093f9f388b31de276ce2de164bdc2081324b9767</td></tr><tr><td>Gemma 3 12B</td><td>google/gemma-3-12b-it</td><td>96b6f1eccf38110c56df3a15bffe176da04bfd80</td></tr><tr><td>Gemma 3 27B</td><td>google/gemma-3-27b-it</td><td>005ad3404e59d6023443cb575daa05336842228a</td></tr><tr><td></td><td></td><td>Llama 3.1 8B Instruct meta-1lama/Llama-3.1-8B-Instruct 0e9e39f249a16976918f6564b8830bc894c89659</td></tr><tr><td></td><td></td><td>Llama 3.1 70B Instruct meta-1lama/Llama-3.1-70B-Instruct 1605565b47bb9346c5515c34102e054115b4f98b</td></tr><tr><td>Qwen3 0.6B</td><td>Qwen/Qwen3-0.6B</td><td>c1899de289a04d12100db370d81485cdf75e47ca</td></tr><tr><td>Qwen3 1.7B</td><td>Qwen/Qwen3-1.7B</td><td>70d244cc86ccca08cf5af4e1e306ecf908b1ad5e</td></tr><tr><td>Qwen3 4B</td><td>Qwen/Qwen3-4B</td><td>1cfa9a7208912126459214e8b04321603b3df60c</td></tr><tr><td>Qwen3 8B</td><td>Qwen/Qwen3-8B</td><td>b968826d9c46dd6066d109eabc6255188de91218</td></tr><tr><td>Qwen314B</td><td>Qwen/Qwen3-14B</td><td>40c069824f4251a91eefaf281ebe4c544efd3e18</td></tr><tr><td>Qwen3 32B</td><td>Qwen/Qwen3-32B</td><td>9216db5781bf21249d130ec9da846c4624c16137</td></tr></table>

TABLE I. Models used in the study.

For each relational reasoning setting, we defined a prompt template containing an instruction segment, relationship statements, and a query segment. The complete templates are provided in Section VIII. Concrete prompts were generated by substituting entity names and relationships into slots of fixed token length. To preserve token positions across instances of a given template, we used single-token entity names and padded relationship substitutions with trailing space tokens to fill their allotted slots. Prompts were then formatted using each model’s chat template. Fo Qwen3 models, thinking mode was disabled.

## C. Recording and readout

To record a source activation tensor $\mathbf { A } = \Phi ( { \mathcal { C } } )$ , we generated a prompt instantiating C and ran a forward pass, using PyTorch forward pre-hooks to record the inputs to each transformer decoder block at positions corresponding to the relationship statements. The recorded span extended from the first entity name through the punctuation terminating the final relationship statement. We stacked these inputs along the layer axis to form A. Similarly, to record a source key–value memory $\mathbf { M } = \Pi ( \Phi ( { \boldsymbol { \mathcal { C } } } ) )$ , we recorded and assembled the key–value cache entries over the same token span.

To read out from an activation tensor A, we evaluated $\Psi ( \mathcal { Q } , \Pi ( \mathbf { A } ) ) ,$ ) using a prompt containing the instruction segment, dummy relationship statements, and the query Q. During the forward pass, we used PyTorch forward prehooks to replace the inputs to each transformer decoder block within the relationship span with the hidden vectors from A, leaving inputs at other positions unmodified.<sup>9</sup> To read out from a key–value memory M, we initialized a key–value cache using the prompt prefix preceding the relationship statements and appended M. We then generated the query segment corresponding to Q and ran the model on this segment using the resulting cache.

## D. Mosaics

Here, we give the exact mosaic constructions used to generate each figure. Because the mosaic experiments defined by the constructions below can be performed for any model and/or prompt template, we give the constructions using the mathematical language developed in section II. We will write $\mathbf { A } _ { t }$ for the slice of the activation tensor corresponding to token position t (each $\mathbf { \bar { A } } _ { t } \in \mathbb { R } ^ { \tilde { N _ { \mathrm { l a y e r s } } } \times D _ { \mathrm { m o d e l } } } )$ , and ${ \bf K } _ { t l i }$ and $\mathbf { V } _ { t l i }$ for the key and value vector, respectively, at position $t ,$ layer l, and key–value head i (each $\mathbf { K } _ { t l i } , \mathbf { V } _ { t l i } \in \mathbb { R } ^ { D _ { \mathrm { h e a d } } } )$ . Token positions are identified by their slots in the prompt template: for example, $t _ { \langle \mathrm { B 1 } }$ and $t _ { \langle \mathrm { B 2 } }$ denote the first and second occurrences of entity $B .$

## Figure 1.

For the behavioral measurements in Fig. 1, we ran ordinary forward passes without recording or modifying interna representations. This is equivalent to reading out from an unmodified activation tensor:

$$
\begin{array} { r l } & { \mathbf { A } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) , } \\ & { \mathbf { A } _ { \mathrm { m o s a i c } } = \mathbf { A } . } \end{array}\tag{21}
$$

(22)

## Figure 2.

We recorded activation tensors from two source contexts:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { c o n t r o l } } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) } \\ { \mathbf { A } _ { \mathrm { d o n o r } } = \Phi ( \mathcal { N } _ { D } , \mathcal { N } _ { E } , \mathcal { N } _ { F } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) . } \end{array}\tag{23}
$$

The names $\mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { N } _ { D } , \mathcal { N } _ { E } , \mathcal { N } _ { F }$ were sampled without replacement. For the mosaic shown in Fig. 2, the re placement was done at the token position corresponding to the first occurrence of the second entity:

$$
( \mathbf { A } _ { \mathrm { m o s a i c } } ) _ { t } = \left\{ { \begin{array} { l l } { ( \mathbf { A } _ { \mathrm { d o n o r } } ) _ { t } , } & { t = t _ { < \mathrm { B } 1 } , } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } , } & { \mathrm { o t h e r w i s e } } \end{array} } \right. .\tag{24}
$$

Results for analogous substitutions at other entity positions are included in the accompanying data.

## Figure 3.

We recorded activation tensors from three source contexts:

$$
\begin{array} { r } { { \bf A } _ { \mathrm { c o n t r o l } } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) } \\ { \bf A _ { \mathrm { d o n o r } } ^ { ( 1 ) } = \Phi ( \mathcal { N } _ { D } , \mathcal { N } _ { E } , \mathcal { N } _ { F } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) } \\ { \bf A _ { \mathrm { d o n o r } } ^ { ( 2 ) } = \Phi ( \mathcal { N } _ { F } , \mathcal { N } _ { E } , \mathcal { N } _ { D } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) . } \end{array}\tag{25}
$$

The names $\mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { N } _ { D } , \mathcal { N } _ { E } , \mathcal { N } _ { F }$ were sampled without replacement. We constructed a diference tensor by subtracting the slice corresponding to $\mathcal { N } _ { D }$ in the first position from the slice corresponding to the same name in the final position:

$$
\delta = ( \mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 2 ) } ) _ { t _ { \mathrm { < T 1 } } } - ( \mathbf { A } _ { \mathrm { d o n o r } } ^ { ( 1 ) } ) _ { t _ { \mathrm { < A 1 } } , } .\tag{26}
$$

We then added and subtracted δ to and from diferent slices of the control tensor to generate the mosaic:

$$
( \mathbf { A } _ { \mathrm { m o s a i c } } ) _ { t } = \left\{ { \begin{array} { l l } { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } + \delta , } & { t = t _ { \langle \mathbf { A 1 } \rangle } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } - \delta , } & { t = t _ { \langle \mathbf { C 1 } \rangle } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right. .\tag{27}
$$

$$
F i g u r e \ 4 .
$$

We recorded activation tensors from two source contexts that difered in the first relationship:

$$
\begin{array} { r } { \mathbf { A } _ { \mathrm { c o n t r o l } } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) } \\ { \mathbf { A } _ { \mathrm { d o n o r } } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } ^ { - 1 } , \mathcal { R } _ { B C } ) , } \end{array}\tag{28}
$$

where $\mathcal { R } _ { A B } ^ { - 1 }$ denotes the inverse relationship of $\mathcal { R } _ { A B }$ . To generate the mosaic, we replaced control slices corresponding to the second relationship phrase and the final entity with the corresponding donor slices:<sup>10</sup>

$$
( \mathbf { A } _ { \mathrm { m o s a i c } } ) _ { t } = \left\{ { \begin{array} { l l } { ( \mathbf { A } _ { \mathrm { d o n o r } } ) _ { t } , } & { t = t _ { \mathrm { \cdot R \cdot B C \mathrm { \cdot } } } , \ldots , t _ { \mathrm { \cdot C 1 \mathrm { \cdot } } } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } , } & { \mathrm { o t h e r w i s e } } \end{array} } \right. .\tag{29}
$$

Here, $t _ { \mathrm { < R - B C } } ,$ denotes the first token of the second relationship phrase.

## Figure 5.

We constructed key–value mosaics by replacing vectors in a control memory with corresponding vectors from an “average memory”. The average memory was computed over $N _ { \mathrm { t r i a l s } } = 1 0 0$ separately sampled entity assignments, with the relationships held fixed. Formally, we evaluated:

$$
\left( \mathbf { K _ { \mathrm { a v e r a g e } } } , \mathbf { V _ { \mathrm { a v e r a g e } } } \right) = \frac { 1 } { N _ { \mathrm { t r i a l s } } } \sum _ { n = 1 } ^ { N _ { \mathrm { t r i a l s } } } \Pi \left( \Phi \left( \mathcal { N } _ { A } ^ { ( n ) } , \mathcal { N } _ { B } ^ { ( n ) } , \mathcal { N } _ { C } ^ { ( n ) } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } \right) \right) .\tag{30}
$$

Keys-first and values-first mosaics. This experiment was partially inspired by [11]. We recorded key-value memories from one source context:

$$
( \mathbf { K _ { \mathrm { c o n t r o l } } } , \mathbf { V _ { \mathrm { c o n t r o l } } } ) = \Pi \big ( \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) \big ) .\tag{31}
$$

We generated the mosaic by replacing selected key and value vectors in the control memory with the corresponding vectors from the average memory. We took $L _ { \mathrm { s p l i t } } ~ = ~ \lfloor N _ { \mathrm { l a y e r s } } / 2 \rfloor$ for all models except Qwen, for which we used $L _ { \mathrm { s p l i t } } = \lfloor 2 N _ { \mathrm { l a y e r s } } / 3 \rfloor . ^ { 1 1 }$ For the keys-first mosaic, we replaced values at layers $l \le L _ { \mathrm { s p l i t } }$ and keys at layers $l > L _ { \mathrm { s p l i t } } \mathrm { . }$

$$
( \mathbf { K } _ { \mathrm { k e y s f i r s t } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { l \leq L _ { \mathrm { s p l i t } } } \\ { ( \mathbf { K } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { l > L _ { \mathrm { s p l i t } } } \end{array} \right. , \ ( \mathbf { V } _ { \mathrm { k e y s f i r s t } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { l \leq L _ { \mathrm { s p l i t } } } \\ { ( \mathbf { V } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { l > L _ { \mathrm { s p l i t } } } \end{array} \right. .\tag{32}
$$

For the values-first mosaic, we reversed this assignment:

$$
( \mathbf { K } _ { \mathrm { v a l u e s - f r e s t } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { K } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { l \leq L _ { \mathrm { s p l i t } } } \\ { ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { l > L _ { \mathrm { s p l i t } } } \end{array} \right. , \quad ( \mathbf { V } _ { \mathrm { v a l u e s - f r e s t } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { V } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { l \leq L _ { \mathrm { s p l i t } } } \\ { ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { l > L _ { \mathrm { s p l i t } } } \end{array} \right. .\tag{33}
$$

Sparse mosaics. We retained control keys and values at selected sites and replaced all other vectors with their averages. We denote the sets of sites retained by the iterative pruning procedure by $K _ { \mathrm { r e m } }$ and $V _ { \mathrm { r e m } } ;$ the pruning procedure is described in Section VI.

$$
( \mathbf { K } _ { \mathrm { s p a r s e } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { ( t , l , i ) \in K _ { \mathrm { r e m } } } \\ { ( \mathbf { K } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { \mathrm { o t h e r w i s e } } \end{array} \right. , \quad ( \mathbf { V } _ { \mathrm { s p a r s e } } ) _ { t l i } = \left\{ \begin{array} { l l } { ( \mathbf { V } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } & { ( t , l , i ) \in V _ { \mathrm { r e m } } } \\ { ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .\tag{34}
$$

Edited mosaics. We recorded control and donor memories from two source contexts:

$$
( \mathbf { K _ { \mathrm { c o n t r o l } } } , \mathbf { V _ { \mathrm { c o n t r o l } } } ) = \Pi \big ( \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) \big )\tag{35}
$$

The names $\mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { N } _ { D } , \mathcal { N } _ { E } , \mathcal { N } _ { F }$ were sampled without replacement. To edit the values associated with the first entity, we replaced control values at sites in $V _ { \mathrm { r e m } }$ at position $t _ { \langle \mathrm { A 1 } }$ with the corresponding donor values. All keys and all other values were retained from the control memory:

$$
\begin{array} { r l } & { ( \mathbf { K } _ { \mathrm { e d i t e d } } ) _ { t l i } = ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } , } \\ & { ( \mathbf { V } _ { \mathrm { e d i t e d } } ) _ { t l i } = \left\{ ( \mathbf { V } _ { \mathrm { d o n o r } } ) _ { t l i } , \quad ( t , l , i ) \in V _ { \mathrm { r e m } } \mathrm { a n d } t = t _ { \mathrm { < A l } } \right\} } \\ & { ( \mathbf { V } _ { \mathrm { c o n t r o l } } ) _ { t l i } , \quad \mathrm { o t h e r w i s e } } \end{array} .\tag{36}
$$

Other edits were defined similarly.

$$
F i g u r e \ S I .
$$

We recorded an activation tensor from a single source context:

$$
\mathbf { A } _ { \mathrm { c o n t r o l } } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) .\tag{37}
$$

We constructed the mosaic by exchanging the slices at the first occurrences of the first and second entities, leaving all other slices unchanged:

$$
( \mathbf { A } _ { \mathrm { m o s a i c } } ) _ { t } = \left\{ { \begin{array} { l l } { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t _ { \langle \mathbf { B } 1 \rangle } } , } & { t = t _ { \langle \mathbf { A } 1 \rangle } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t _ { \langle \mathbf { A } 1 \rangle } } , } & { t = t _ { \langle \mathbf { B } 1 \rangle } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } , } & { { \mathrm { o t h e r w i s e } } } \end{array} } \right. .\tag{38}
$$

During readout, Π applies the rotation matrices associated with the destination positions, so each exchanged slice receives the positional information associated with the other’s original position.

## Figure S2.

We recorded activation tensors from three prompts containing the same entity names and relationships but difering in the number of space tokens inserted immediately before the relationship statements. We defined $\Delta = t _ { \langle \mathrm { B } 1 \rangle } - t _ { \langle \mathrm { A } 1 }$ and inserted $0 , \Delta ,$ or $2 \Delta$ space tokens. Writing $\Phi _ { + s }$ for recording with s inserted tokens, we obtained:

$$
\begin{array} { r } { { \bf A } _ { - \Delta } = \Phi ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) \qquad } \\ { { \bf A } _ { \mathrm { c o n t r o l } } = \Phi _ { + \Delta } ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) \qquad } \\ { { \bf A } _ { + \Delta } = \Phi _ { + 2 \Delta } ( \mathcal { N } _ { A } , \mathcal { N } _ { B } , \mathcal { N } _ { C } , \mathcal { R } _ { A B } , \mathcal { R } _ { B C } ) . \qquad } \end{array}\tag{39}
$$

The subscripts $- \Delta$ and $+ \Delta$ denote shifts relative to the control prompt. In the equations below, token indices identify positions within the relationship span, so a given index refers to the same template slot across the three tensors.

We replaced the control slice at $\langle \mathrm { A } 1 \rangle$ with the corresponding slice from $\mathbf { A } _ { + \Delta }$ , and the control slice at ‹B1› with the corresponding slice from $\mathbf { A } _ { - \Delta }$

$$
( \mathbf { A } _ { \mathrm { m o s a i c } } ) _ { t } = \left\{ \begin{array} { l l } { ( \mathbf { A } _ { + \Delta } ) _ { t _ { \mathrm { < A 1 } } \mathrm { , } } , } & { t = t _ { \mathrm { < A 1 } \mathrm { , } } } \\ { ( \mathbf { A } _ { - \Delta } ) _ { t _ { \mathrm { < B 1 } } \mathrm { , } } , } & { t = t _ { \mathrm { < B 1 } \mathrm { , } } } \\ { ( \mathbf { A } _ { \mathrm { c o n t r o l } } ) _ { t } , } & { \mathrm { o t h e r w i s e } } \end{array} \right. .\tag{40}
$$

With this choice of $\Delta$ , the donor slice inserted at ‹A1› was recorded at the absolute token position occupied by ‹B1 in the control prompt, and vice versa. We read out from the mosaic using the control prompt’s padding:

$$
\ell _ { \mathrm { a n s } } ^ { \prime } = \Psi _ { + \Delta } \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( \mathcal { Q } , \Pi _ { + \Delta } \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( \mathbf { A } _ { \mathrm { m o s a i c } } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) .\tag{41}
$$

## IV. TESTS OF POSITION-CODE HYPOTHESES

In this section, we present the results of two mosaic experiments designed to test whether the putative representation of abstract relational structure in the activation tensor instead corresponds to a sequence of positional codes. Specifically, we considered the hypothesis that the activation tensor associates information about token identity with positional codes,

$$
\begin{array} { r l } & { \phantom { \frac { 1 } { 2 } } \cdots \cdot \phantom { \frac { 1 } { 2 } } \mathcal { P } _ { 1 2 8 } , \mathcal { P } _ { 1 2 9 } , \mathcal { P } _ { 1 3 0 } , \mathcal { P } _ { 1 3 1 } , \phantom { \frac { 1 } { 2 } } \cdot \cdot \cdot } \\ & { \phantom { \frac { 1 } { 2 } } \cdot \phantom { \frac { 1 } { 2 } } \cdot \cdot \cdot , ( \mathcal { P } _ { 1 2 8 }  \mathrm { A l i c e } ) , ( \mathcal { P } _ { 1 2 9 }  \mathrm { h a s } ) , ( \mathcal { P } _ { 1 3 0 }  \mathrm { d a u g h t e r } ) , ( \mathcal { P } _ { 1 3 1 }  \mathrm { B e t h } ) , \phantom { \frac { 1 } { 2 } } \cdot \cdot \cdot , } \end{array}\tag{42}
$$

and that the model uses these associations to answer queries. Note that under this account, the task-relevant information retained after context presentation is nothing more than the ordered sequence of tokens in the prompt; the interpretation of that sequence as a set of relationships is deferred until a query is presented.

In principle, positional information could either be supplied by the rotary positional embeddings applied when constructing M from A or could already be present in A as learned codes for global token position. We designed one mosaic experiment to test each possibility.

In the first experiment, we exchanged the slices of the activation tensor corresponding to the first occurrences of the first and second entities (Eq. 38). Recall that when constructing the key–value memory, Π applies position-dependent rotations determined by each slice’s position (Eq. 15). Therefore, this manipulation exchanges the externally supplied positional information associated with the two entity representations. If the model reconstructs relationships at query time using this positional information, this manipulation should cause it to behave as if the relationship between the first two entities had been reversed. Instead, the models’ responses were largely unchanged (Fig. S1).

A  
![](images/9fe0bb2084bbe1dccc836f8125cdaea9c71cea79fdbf460b83830062f981e3c6.jpg)

B  
![](images/4ccb1c9cb6e41b2a9ed54ad01140fb858c07b3e27c51d6f0da41594f38e7fc5c.jpg)

C  
![](images/39047ba881610407abf2d69075dd77a2492719aaf5129384ab2838e934cd106c.jpg)

D  
![](images/d236e7db927ec962c876880a9df2b3a02a521c645a03ef4d2f1ca14e2d92746d.jpg)  
FIG. S1. Exchanging externally supplied positional information does not appreciably change model responses. Data is shown for Gemma-3-27B. A–D, Responses to the queries indicated above each plot for control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in the main text. Fifty trials are shown per condition.

This result may initially seem surprising: the manipulation appears to exchange the order of two tokens without appreciably changing the model’s responses. However, note that we exchange rows of A after the model has processed the context. Therefore, if during context processing the model assigns patterns of activity to the relevant entity tokens that track either their relational role or their original token position, those patterns of activity would travel with the corresponding rows and remain associated with the same entity names.

Those activity patterns could still reflect learned codes for global token positions. We therefore designed a second experiment to exchange these codes, if present, between the two entity representations by changing the token positions at which entity representations were computed. Specifically, we inserted space tokens before the relationship statements to record representations of entities at diferent absolute token positions. Table II gives an example of three prompts used to record source activation tensors on a particular trial. The middle $( + \Delta )$ prompt was used to record $\mathbf { A } _ { \mathrm { c o n t r o l } } ;$ ; this allowed us to obtain “Alice” and “Beth” representations from prompts where the names occurred later and earlier, respectively, than in the control prompt.

The ofsets were chosen so that “Alice” in the +2∆ prompt occurs at “Beth”’s position in the control prompt, and $\mathrm { \Delta ^ { 6 } b 6 t h } ^ { \mathrm { 7 } }$ in the +0 prompt occurs at “Alice”’s position in the control prompt. We constructed $\mathbf { A } _ { \mathrm { m o s a i c } }$ by replacing Alice’s row in $\mathbf { A } _ { \mathrm { c o n t r o l } }$ with the corresponding row from the +2∆ prompt, and Beth’s row with the corresponding row from the +0 prompt (Eq. (40)). In the example shown in Table II, the “Alice” representation in $\mathbf { A } _ { \mathrm { m o s a i c } }$ should carry the 134th positional embedding, and the “Beth” representation should carry the 131st positional embedding. In general, this manipulation should exchange any learned positional codes associated with the two entities while preserving information about their identities and relational roles. Under the hypothesis that the model reconstructs relationships from the token–position associations at query time, it should recover the reversed first relationship, “Beth has daughter Alice”. Instead, the model’s responses were largely unchanged (Fig. S2).

A  
![](images/dae75ba0fa0adac910fe64e68b26f9270aa6335a4d424cf89bd1e3d93aa65d95.jpg)

B  
![](images/36f48ddb63fabb4eaf1f80d68512591c7775859307cb38f461c5e3d5da977979.jpg)

C  
![](images/1072b6d1e10b1839cec7ee64b807e306575a9b1ecb0e3b01598e7ba64943b1af.jpg)

D  
![](images/e11da0df00705a399c67f788da6d3a4b384bd3eb82b783946141ba81f2fe549c.jpg)  
FIG. S2. Exchanging any global positional codes carried by the activation tensor does not appreciably change model responses. Data is shown for Gemma-3-27B. A–D, Responses to the queries indicated above each plot for control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in the main text. Fifty trials are shown per condition.

<table><tr><td>Token index</td><td>+0 prompt</td><td>+∆ prompt</td><td>+2∆ prompt</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>126</td><td>Context</td><td>Context</td><td>Context</td></tr><tr><td>127</td><td>:</td><td>:</td><td>:</td></tr><tr><td>128</td><td>Alice</td><td>U</td><td>U</td></tr><tr><td>129</td><td>has</td><td>U</td><td>U</td></tr><tr><td>130</td><td>daughter</td><td>U</td><td>U</td></tr><tr><td>131</td><td>Beth</td><td>Alice</td><td>U</td></tr><tr><td>132</td><td></td><td>has</td><td>U</td></tr><tr><td>133</td><td>Beth</td><td>daughter</td><td>U</td></tr><tr><td>134</td><td>has</td><td>Beth</td><td>Alice</td></tr><tr><td>135</td><td>daughter</td><td></td><td>has</td></tr><tr><td>136</td><td>Carol</td><td>Beth</td><td>daughter</td></tr><tr><td>137</td><td></td><td>has</td><td>Beth</td></tr><tr><td>138</td><td>\n\n</td><td>daughter</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>•</td><td></td><td></td><td>•</td></tr><tr><td></td><td>•</td><td>·</td><td></td></tr></table>

TABLE II. Example of prompts used to construct the global-position control. The three prompts contain identical relationship statements, preceded by 0, ∆, or 2∆ inserted space tokens, where $\Delta$ is the token distance between the first and second entities. The +∆ prompt serves as the control; the other two supply representations of ${ } ^ { 6 6 } \mathrm { A l i c e } ^ { 3 }$ and “Beth” recorded at the other’s absolute positions in the control prompt. The symbol $\sqcup$ denotes a space.

## V. EVIDENCE FOR PROSPECTIVE COMPOSITION OF RELATIONSHIPS

Recall that in the main text, we presented two hypotheses for the form of the representation of abstract relational structure in the activation tensor:

1. Graph-based representation. Observed relationships are represented as “edges” in A, then composed retrospectively when a query is asked.

2. Location-based representation. As relationships are encountered, models compose them prospectively to assign each entity a “location” in a shared coordinate system.

Recall further that the mosaic experiment presented in Fig. 4 and eqs. (28), (29) was designed to distinguish between these two hypotheses. As we argued in the main text, the former hypothesis predicts that $\mathbf { A } _ { \mathrm { m o s a i c } }$ supports the same responses to two-step queries as $\mathbf { A } _ { \mathrm { c o n t r o l } } ;$ the latter predicts that it supports the same responses as $\mathbf { A } _ { \mathrm { d o n o r } }$ . We found that responses typically matched those obtained from the control tensor (Fig. 4).

Here, we present an exception to this tendency. Figure S3 shows the models’ responses to four queries given a mosaic activation tensor constructed using the same procedure as in Fig. 4B, but with the donor and control contexts exchanged. An additional exception, obtained with Llama-3.1-70B using the original donor and control assignment, is shown in Fig. S104D.

A  
![](images/e9d290e8c29a6750cb02cc394501035a4dc5e5d9a3a5d511694ef1abdf26707e.jpg)

B  
![](images/4f8c314f4ccb9722814f6e62e99b93f9fa9b4959ec90b9333b009604e89f94e8.jpg)

C  
![](images/776dc48328fe11e558004b53bca033eea3307353f059bafd429896a6b19eeea6.jpg)

D  
![](images/16a208c51943504b7910e88cdb4699151537ed0fc2cb9882085a954694dd66ea.jpg)  
FIG. S3. Evidence for a location-based representation in the family-tree setting. In all four panels, the donor context was $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ⟨B⟩. ⟨B⟩ has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } }$ , the control context was $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ⟨B⟩. ⟨B⟩ has daughter $\langle \mathrm { C } \rangle ^ { \eta } ,$ and the mosaic was constructed as in main-text Fig. 4. Data is shown for Gemma-3-27B. A–D, Responses to the queries indicated above each plot for control (red) and mosaic (purple) activation tensors. Probability ratios are defined as in the main text. Fifty trials are shown per condition.

In panels S3A and S3C above, the model’s responses follow the two-hop relationship implied by the donor, rather than the control context. The entries of $\mathbf { A } _ { \mathrm { d o n o r } }$ that were transferred into $\mathbf { A } _ { \mathrm { c o n t r o l } }$ to construct $\mathbf { A } _ { \mathrm { m o s a i c } } \ ( \mathrm { i . e . }$ ., the rows of $\mathbf { A } _ { \mathrm { d o n o r } }$ corresponding to the “has daughter” and “Carol” tokens) must therefore supply some information beyond the relationship between “Beth” and “Carol” that is responsible for this change in behavior.

![](images/5b334aaf322b5732790fe9b23bbcc6579568e9b7016c3fff4c130a7b43b26d5e.jpg)

One possibility is that the transferred entries carry a copy of the “edge” between “Alice” and “Beth”. The results in panels S3A and S3C could then be explained if the model sometimes uses this copy—which in $\mathbf { A } _ { \mathrm { m o s a i c } }$ encodes the relationship between “Alice” and “Beth” in the donor context—when retrospectively composing relationships. However, this account does not readily explain the disrupted response to the one-hop relationship shared by the control and donor contexts (Fig. S3D).

A second possibility is that the transferred entries carry information reflecting the prospective composition of the two relationships in the donor context (i.e., the transferred entries carry the information “Carol is Alice’s granddaughter”). Such a “location-based” account naturally explains the observation in Fig. S3D: the location assigned to “Carol” in the donor context, combined with the location assigned to “Beth” in the control context, would no longer encode the relationship “Beth is Carol’s mother”.

Notably, when provided with the same mosaic tensor, the model’s responses to other queries follow the twohop relationship implied by the control context (Fig. S3B), as predicted by the graph-based hypothesis. Therefore, both location-based information (reflecting prospective composition of relationships) and graph-based information (requiring retrospective composition of relationships) may be available in the activation tensor. We do not yet understand what factors cause the model to rely on one type of information or the other at query time.

Consistent with this possibility, direct visualizations of A suggest that activity within some entries of A depends primarily on the most recent relationship, whereas activity at others depends on the full history of relationships. At each token position, the sequence of hidden vectors in A across successive layers traces a trajectory through activity space $\mathbb { R } ^ { D _ { \mathrm { m o d e l } } }$ . We visualized these trajectories by projecting them onto the first two principal components computed from hidden vectors collected across prompts at the model’s midpoint. We found that at the second relationship token (e.g., “daughter”), the trajectories depend on both relationships in the context (Fig. S4A). In contrast, at the final entity token (e.g., “Carol”), they appear to depend primarily on the most recently encountered relationship (Fig. S4B). This contrast is suggestive of the coexistence of graph-based and location-based information in the activation tensor.

A  
![](images/b413147d33993f9685cef45b9f2f2e6c33ce0f47d09a9ccedf30d227a5b70421.jpg)

B  
![](images/8d8d7a35c1942bee28b411247330bde42eaf3d47c7bda3b9bcda55e50949336e.jpg)  
FIG. S4. PCA visualizations of activation trajectories in the family-tree setting. Each trajectory follows the hidden vector at a fixed token position across successive layers. Trajectories were centered by subtracting the layer-wise mean across relational structures and name resamplings, then projected onto the first two principal components computed from population activity halfway through the model. Faint lines show individual trajectories; solid lines show averages over prompts with the same relational structure but diferent entity names. Visualizations are shown for Llama-3.1-70B; the corresponding Gemma-3-27B projections were harder to interpret, possibly because the activation norm grew more rapidly across layers. $\mathbf { A } ,$ Activations at the second relationship token. B, Activations at the final entity token.

The exceptions that we describe here are easiest to find in the family-tree setting. We also observed occasiona shifts from control-like toward donor-like responses in some transitive inference settings (not shown). However, these cases are harder to interpret because the models do not reliably answer “Unknown” when asked about a relationship that the control context leaves undetermined—for example, whether Alice is taller than Carol given “Alice is taller than Beth; Beth is shorter than Carol”.

## VI. IDENTIFYING NAME-TO-ROLE AND ROLE-TO-NAME PAIRS

In this section, we describe the iterative pruning procedure used to find candidate name–to–role and role–to–name key–value pairs. Briefly, we iteratively “erased” name-dependent information from keys and values by replacing them with their averages across prompts with resampled names. At each iteration, we used the gradient of a scalar measure of task performance to estimate the keys or values whose erasure would least afect the models’ behavior.

We represented the retained vectors using two binary masks, $m ^ { ( K ) } , m ^ { ( V ) } \in \{ 0 , 1 \} ^ { T _ { \mathrm { c o n t e x t } } \times N _ { \mathrm { l a y e r s } } \times N _ { \mathrm { h e a d s } } }$ . To facilitate diferentiation with respect to the mask entries, we expressed the replacement rule as a linear interpolation between control and average key–value memories, treating the mask entries as continuous variables:

$$
\begin{array} { r l } & { \quad ( \mathbf { K } _ { \mathrm { i n t e r p } } ) _ { t l i } = ( \mathbf { K } _ { \mathrm { a v e r a g e } } ) _ { t l i } + m _ { t l i } ^ { ( K ) } \big ( ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } - ( \mathbf { K } _ { \mathrm { a v e r a g e } } ) _ { t l i } \big ) , } \\ & { \quad \quad ( \mathbf { V } _ { \mathrm { i n t e r p } } ) _ { t l i } = ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } + m _ { t l i } ^ { ( V ) } \big ( ( \mathbf { V } _ { \mathrm { c o n t r o l } } ) _ { t l i } - ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } \big ) , } \\ & { \mathbf { M } _ { \mathrm { i n t e r p } } ( m ^ { ( K ) } , m ^ { ( V ) } ) = ( \mathbf { K } _ { \mathrm { i n t e r p } } , \mathbf { V } _ { \mathrm { i n t e r p } } ) . } \end{array}\tag{43}
$$

We measured task performance by computing the average diference between the correct answer logit and the mean logit of the other candidate answers over a predefined collection of queries:

$$
\mathrm { S c o r e } ( f ( \cdot ) ) = \frac { 1 } { N _ { \mathrm { q u e r i e s } } } \sum _ { Q \in \mathrm { Q u e r i e s } } \left( f _ { a ^ { * } ( Q ) } ( Q ) - \operatorname* { m e a n } _ { a \neq a ^ { * } ( Q ) } f _ { a } ( Q ) \right) .\tag{44}
$$

Here, $f _ { a } ( \mathcal { Q } )$ denotes the logit for answer $^ { a , }$ and $a ^ { * } ( \mathfrak { Q } )$ is the correct answer.

On each iteration, the marginal efect of pruning each remaining key or value was estimated by evaluating the derivative of the scalar score with respect to each entry of either mask:

$$
\begin{array} { r l } & { \frac { \partial } { \partial m _ { t l i } ^ { ( K ) } } \mathrm { S c o r e } \left( \boldsymbol { \Psi } \left( \cdot , \mathbf { M } _ { \mathrm { i n t e r p } } ( m ^ { ( K ) } , m ^ { ( V ) } ) \right) \right) , } \\ & { \frac { \partial } { \partial m _ { t l i } ^ { ( V ) } } \mathrm { S c o r e } \left( \boldsymbol { \Psi } \left( \cdot , \mathbf { M } _ { \mathrm { i n t e r p } } ( m ^ { ( K ) } , m ^ { ( V ) } ) \right) \right) . } \end{array}\tag{45}
$$

Note that derivatives with respect to each entry of $m ^ { ( K ) }$ and $m ^ { ( V ) }$ can be evaluated simultaneously in a single backward pass.

In practice, most key and value vectors could be erased with little efect on the score. We therefore began with a coarse schedule that erased a fixed fraction of the remaining vectors at each iteration. Once the score fell below a threshold, we rewound to an earlier point in the trajectory and continued erasing vectors one at a time, providing finer resolution around the decline in performance (Algorithm 1). Finally, in practice, we averaged the score over queries from several prompts with independently sampled entity names and used the gradient of this mean score to guide pruning.

Algorithm 1 Iterative erasure of key–value vectors   
Require: Initial keys-first mask m<sub>0</sub>; score S(m); coarse fraction η; threshold τ ; rewind factor $\rho$   
1: m ← m   
2: S<sub>initial</sub> ← S(m<sub>0</sub>); S<sub>erased</sub> ← S(0)   
3: while N(m) > 0 do ▷ Coarse erasure   
4: s ← S(m); g ← ∇<sub>m</sub>S(m)   
5: Set to zero the ⌈ηN(m)⌉ retained mask entries with the smallest |g|   
s − S<sub>erased</sub>   
6: if < τ then   
S<sub>initial</sub> − S<sub>erased</sub>   
7: N<sub>target</sub> ← min(N(m<sub>0</sub>), ⌈ρN(m)⌉)   
8: Rewind to the most recent mask with at least $N _ { \mathrm { t a r g e t } }$ retained entries   
9: break   
10: end if   
11: end while   
12: while N(m) > 0 do ▷ Fine erasure   
13: g ← ∇<sub>m</sub>S(m)   
14: Set to zero the retained mask entry with the smallest |g|   
15: end while   
16: return the pruning trajectory, excluding steps undone by rewinding

Figure S5 shows the pruning trajectory and the distribution of retained keys and values for the sparse mask used in Fig. 5 of the main text. Several features of these results are of note:

1. Name-dependent information could be “erased” from the vast majority of key and value vectors with little efect on the score.

2. The score began to decline significantly when name information in roughly 200 keys or values remained, corresponding to approximately 1% of the total number of keys and values. Therefore, a small subset of sites was suficient to support retrieval. The subsequent decline in score was gradual, consistent with the relevant information being distributed across the retained sites.

3. The candidate name-to-role and role-to-name pairs were distributed across depth.<sup>12</sup>

4. Candidate name-to-role pairs were concentrated at entity-name tokens and the tokens immediately following them. This may reflect the copying of information from entity-token positions to subsequent positions by previous-token heads [12].

5. At some points along the pruning trajectory, both the retained set of keys and values and its complement could support retrieval when tested separately (data not shown). This suggests that the candidate key–value pairs we identified are suficient, but not necessary, for retrieval.

A  
![](images/611cc3b7f093947fdf9727a173a8e5ecd56a3a1eb8ec1cf2cbe339e3683b31bb.jpg)

B  
![](images/62ba73646d9304f2e103157008371681382fb55cfe2f8f0efb18aea319aa2ab0.jpg)

C  
![](images/3d9f4f667c0dddae3ed96b260eca91dd9b2be50398c62a542b41e6398b5bab79.jpg)

![](images/98ffe7fbb2f6cafab51e7e569ac7db39886f62445c31f91af503a21537958e55.jpg)  
FIG. S5. Iterative erasure of key and value vectors. Results are shown for Gemma-3-27B in the family-tree setting with “who is” queries. A, Full erasure trajectory. Scores were evaluated on the same prompts used to guide pruning. B, Expanded view of the trajectory. The vertical dashed line marks the transition to single-site erasure; the marker identifies the mask used for the main-text figure. C, Distribution of keys and values retained by the mask used for the main-text figure.

## VII. EFFECTS OF QUERY FORMAT

In the main text, we presented results using both yes-or-no queries (e.g., “Is Carol’s grandmother Alice?”) and queries requiring retrieval of a name (e.g., “Who is Carol’s grandmother?”).<sup>13,14</sup> We later tested both query formats for each experiment and found that, in all cases, the results were consistent with the account developed in the main text. Here, we present selected experiments using the alternative query format, and describe how the results of the experiment change with the alternative format.

Figure S6 shows the results of the mosaic experiment in main-text Fig. 2 using queries that require the model to retrieve a name. Three aspects of these data deserve comment:

1. The query “Who is Eve’s mother?” cleanly diferentiates between the two hypotheses illustrated in the main text—under the attribute-tagging hypothesis, the model should respond “Daisy”; under the identity–structure binding hypothesis, it should respond “Alice”. The model responds “Alice” and assigns almost no additional probability to “Daisy” under the mosaic compared with the control (Fig. S6A).<sup>15</sup>

2. The model responds “Carol” to the question “Who is Eve’s daughter?” and remains capable of answering the two-hop questions “Who is Carol’s grandmother?” and “Who is Alice’s granddaughter?” (Fig. S6B, E, F). These observations parallel those in the main text that support the conclusion that the name-independent component of the activation tensor represents the complete relational structure instantiated by the prompt.

3. When asked about Alice’s daughter or Carol’s mother, the model splits probability mass between “Beth” and “Eve”. This is consistent with the schematic presented in the main text, where both “Beth” and “Eve” are bound to the same structural role.

A  
![](images/3ae8f405bf2e0550f82cfae3deb9b90fc3efae60f5c6b6bcbd7edf292737781f.jpg)

C  
![](images/8d6fd71ded49741fc4dfdd912f8f1766f2d4aa8dd6facedcf01f87e8dbaa00b9.jpg)

E  
![](images/5777eba86a779443b7d65bfbbcf7cd7a8e3f94cf53283400c2b309cc8121b21d.jpg)

B  
![](images/4e2127d23e5d02c06482e6408c37fd816aaf6500df9d6c8dff7d32937d1af966.jpg)

D  
![](images/c03872f5cd351ee193b54a01553f924bf698f9bb494bd682e98c6c2414b54a60.jpg)

F  
![](images/17129e01b59b303a6da25729cf7fa34e8824bf77b166793062c58d6071ae2c83.jpg)  
FIG. S6. Results under the who-is query format are consistent with identity-structure separation. Responses are plotted for control (red) and mosaic (purple) activation tensors. The mosaic was constructed as in main-text Fig. 2. Results are shown for Gemma-3-27B in the family-tree setting. Probability ratios are defined as in the main text. Fifty trials are shown per condition, with entity names independently resampled on each trial. The labels ‹A›, ‹B›, and ‹C› denote the control names in order of first appearance; ‹D›, ‹E›, and ‹F› denote the corresponding donor names.

Figure S7 shows the results of the experiments involved in main text Fig. 5 using yes-or-no queries. We note that to answer a yes-or-no question (e.g., “Is Beth’s mother Alice?”), models can use at least two strategies. First, they can retrieve the information “Beth’s mother is Alice” using the mechanism we described in the main text, compare the retrieved name with the “Alice” in the query, then respond “yes”. Alternatively, they can retrieve the structural roles corresponding to “Alice” and “Beth”, compare them, and respond “yes”. Note that the second strategy does not require role-to-name pairs.

The following observations support the suficiency of the second strategy for answering yes-or-no questions:

1. We constructed an additional keys-only mosaic, which is similar to the keys-first mosaic, except that values after the layer split are also replaced with their averages across prompts with resampled names:

$$
\begin{array} { r l } & { ( \mathbf { K } _ { \mathrm { k e y s - o n l y } } ) _ { t l i } = \left\{ ( \mathbf { K } _ { \mathrm { c o n t r o l } } ) _ { t l i } , \quad l \leq L _ { \mathrm { s p l i t } } , \right. } \\ & { \left. ( \mathbf { K } _ { \mathrm { k e y s - o n l y } } ) _ { t l i } , \quad l > L _ { \mathrm { s p l i t } } , \right. } \\ & { \left. ( \mathbf { V } _ { \mathrm { k e y s - o n l y } } ) _ { t l i } = ( \mathbf { V } _ { \mathrm { a v e r a g e } } ) _ { t l i } . \right. } \end{array}\tag{46}
$$

This mosaic removes name-dependent information from all values, including those of candidate role-to-name pairs.<sup>16</sup> We found that the model tolerated this manipulation with no major drop in performance (Fig. S7A).

2. When we applied the iterative pruning procedure starting from the keys-first mosaic, the resulting sparse mosaic retained name-dependent information in 143 keys and 16 values while largely preserving performance (Fig. S7B– E). The much smaller number of retained values is consistent with a greater reliance on name-to-role pairs for these queries.

3. We edited the key–value memory using a procedure analogous to that described in the main text. In Fig. S7F and G, we replaced the keys of candidate name-to-role pairs associated with the first entity with the corresponding donor keys. This intervention caused the model to answer yes to both “Is Beth’s mother Daisy?” and “Is Daisy’s daughter Beth?”. Thus, editing keys associated with an entity altered responses to yes-or-no queries in both directions.

A  
![](images/3924393e48c306a4d771548259207e43c33f06103d789edfe53f712d58fff85d.jpg)

B  
![](images/861de70a76d722843b86a2a20f727b46c27776f706e2b55f3f27b04e00b809d3.jpg)

C  
![](images/146a8f6ebfd29aaf22b5e5a62da82156d184d3da8327c7100d0ac26c3e49d6cf.jpg)

D  
![](images/cf1f96b9cce8714f52e5b67c6d4eae00385d9480af4ca6a828a5ce26ea38b197.jpg)

![](images/2f7164001370b6c6b02fd7e08aa5cd5efbdc262c5d92b39e8a014449b8f5ec09.jpg)

E  
![](images/fe2da4e9ed9b0dd7e2760129e4b25d041cac6157ac5934e835ffcdf34ee2c3d2.jpg)

F  
![](images/630069aa480eda060332b2ffb8ef9fe149eecc6c78462c80e8f5095e032e6081.jpg)

G  
![](images/0633a0730331b8f6bb49175f238fa6e60db88438c7ae857cde881d64797dc7ae.jpg)  
FIG. S7. Retrieval under the yes-or-no query format involves a sparse set of name-to-role pairs. Results are shown for Gemma-3-27B in the family-tree setting. In all mosaic experiments, probability ratios are defined as in the main text and fifty trials are shown per condition. The labels $\langle \mathrm { A } \rangle ,$ ‹B›, and ‹C› denote the control names in order of first appearance, and ‹D› denotes the first donor name; these names are resampled on each trial. A, Restricting name-dependent information to keys before the layer split preserves performance. B, Full pruning trajectory starting from the keys-first mosaic. Scores were evaluated on the same prompts used to guide pruning. C, Expanded view of the trajectory. The vertical dashed line marks the transition to single-site pruning; the marker identifies the mask used in the subsequent panels. D, Distribution of the 143 keys and 16 values retained by this mask across token positions and layers. E, Performance is largely preserved when name-dependent information is restricted to the 143 keys and 16 values identified by iterative pruning. F,G, Editing candidate name-to-role pairs changes the model’s responses in both query directions.

## VIII. PROMPT TEMPLATES

In this section, we provide prompt templates for the fourteen relational reasoning tasks used in this study. The set of tasks was chosen to span four task families<sup>17</sup>: family trees (“family trees”, “family trees (name-retrieval queries)”, “ancestor–descendant relationships”); ordinal relationships (“company hierarchies”, “age comparisons”, “height comparisons”, “speed comparisons”, “east–west relationships”, “north–south relationships”, “ahead–behind relationships”, “exact age diferences”); signed interactions (“gene regulatory networks”, “neural circuits”); and equivalence (“propositional equivalence”, “variable equality”). In several of these task families, queries are most naturally phrased as yes-or-no questions, so we use this query format for all tasks and additionally include a name-retrieval format for the family-tree task.

## A. Family Trees

Instructions:

1) Read the context describing a series of relationships between people in a family.

2) Keep track of who is related to whom as you read.

3) You may assume that everyone who appears in the context is female.

4) At the end, answer the question. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

5) If the statement is true, output 'Yes'.

6) If the statement is false, output 'No'.

7) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

8) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› has ‹daughter/mother› ‹Beth›. ‹Beth› has ‹daughter/mother› ‹Carol›.

Question: Is ‹Carol›'s ‹grandmother› ‹Alice›?

Answer:

## B. Family Trees (Name-Retrieval Queries)

Instructions:

1) Read the context describing a series of relationships between people in a family.

2) Keep track of who is related to whom as you read.

3) You may assume that everyone who appears in the context is female.

4) At the end, answer the question. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› has ‹daughter/mother› ‹Beth›. ‹Beth› has ‹daughter/mother› ‹Carol›.

Question: Who is ‹Carol›'s ‹grandmother›?

Answer:

Instructions:   
1) Read the context describing a series of relationships between people in a family.   
2) Keep track of who is related to whom as you read.   
3) You may assume that everyone who appears in the context is female.   
4) At the end, answer the question. You may use:   
(a) Information stated in the context.   
(b) Any facts that follow from the context by valid logical inference.   
5) If the statement is true, output 'Yes'.   
6) If the statement is false, output 'No'.   
7) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.   
8) Output exactly one word (no punctuation or extra text).   
Context: ‹Alice› has ‹daughter/mother› ‹Beth›. ‹Beth› has ‹daughter/mother› ‹Carol›.   
Question: Is ‹Carol› ‹a descendant› of ‹Alice›?   
Answer:

## C. Ancestor–Descendant Relationships

## D. Company Hierarchies

Instructions:

1) Read the context describing the organizational structure in a company.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice›'s ‹boss/employee› is ‹Beth›. ‹Beth›'s ‹boss/employee› is ‹Carol›.

Question: Is ‹Carol› ‹senior› to ‹Alice›?

Answer:

## E. Age Comparisons

Instructions:

1) Read the context describing a series of comparisons between people.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› is ‹older/younger› than ‹Beth›. ‹Beth› is ‹older/younger› than ‹Carol›.

Question: Is ‹Carol› ‹younger› than ‹Alice›?

Answer:

## F. Height Comparisons

Instructions:

1) Read the context describing a series of comparisons between people.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› is ‹taller/shorter› than ‹Beth›. ‹Beth› is ‹taller/shorter› than ‹Carol›.

Question: Is ‹Carol› ‹shorter› than ‹Alice›?

Answer:

## G. Speed Comparisons

Instructions:

1) Read the context describing a series of comparisons between people.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› is ‹faster/slower› than ‹Beth›. ‹Beth› is ‹faster/slower› than ‹Carol›.

Question: Is ‹Carol› ‹slower› than ‹Alice›?

Answer:

## H. East–West Relationships

Instructions:

1) Read the context describing a series of spatial relationships between locations.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: The ‹library› is ‹east/west› of the ‹museum›. The ‹museum› is ‹east/west› of the ‹station›.

Question: Is the ‹station› ‹west› of the ‹library›?

Answer:

## I. North–South Relationships

Instructions:

1) Read the context describing a series of spatial relationships between locations.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: The ‹library› is ‹north/south› of the ‹museum›. The ‹museum› is ‹north/south› of the ‹station›.

Question: Is the ‹station› ‹south› of the ‹library›?

Answer:

## J. Ahead–Behind Relationships<sup>18</sup>

Instructions:

1) Read the context describing a series of spatial relationships between locations.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: The ‹library› is ‹ahead/behind› of the ‹museum›. The ‹museum› is ‹ahead/behind› of the ‹station›.

Question: Is the ‹station› ‹behind› of the ‹library›?

Answer:

## K. Exact Age Diferences<sup>19</sup>

Instructions:

1) Read the context describing a series of comparisons between people.

2) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

3) If the statement is true, output 'Yes'.

4) If the statement is false, output 'No'.

5) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

6) Output exactly one word (no punctuation or extra text).

Context: ‹Alice› is ‹one year older/one year younger› than ‹Beth›. ‹Beth› is ‹one year older/one year ,→ younger› than ‹Carol›.

Question: Is ‹Carol› ‹two years younger› than ‹Alice›?

Answer:

## L. Gene Regulatory Networks

Instructions:

1) Read the context describing a simple gene regulatory network.

2) For this task, genes are identified by three-letter strings.

3) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

4) If the statement is true, output 'Yes'.

5) If the statement is false, output 'No'.

6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

7) Output exactly one word (no punctuation or extra text).

Context: ‹axs› ‹activates/represses› ‹bpy›. ‹bpy› ‹activates/represses› ‹chh›.

Question: Does ‹axs› ‹repress› ‹chh›?

Answer:

## M. Neural Circuits

Instructions:

1) Read the context describing a simple vertebrate neural circuit.

2) For this task, neurons are identified by three-letter strings.

3) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

4) If the statement is true, output 'Yes'.

5) If the statement is false, output 'No'.

6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

7) Output exactly one word (no punctuation or extra text).

Context: ‹axs› makes ‹glutamatergic/GABAergic› projections to ‹bpy›. ‹bpy› makes ‹glutamatergic/GABAergic› ,→ projections to ‹chh›.

Question: Does ‹axs› ‹inhibit› ‹chh›?

Answer:

## N. Propositional Equivalence

Instructions:

1) Read the context describing a series of relationships between propositions.

2) For this task, propositions are numbered by a roman numeral.

3) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

4) If the statement is true, output 'Yes'.

5) If the statement is false, output 'No'.

6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

7) Output exactly one word (no punctuation or extra text).

Context: ‹I› is ‹equivalent to/incompatible with› ‹II›. ‹II› is ‹equivalent to/incompatible with› ‹III›.

Question: Is ‹III› ‹equivalent to› ‹I›?

Answer:

## O. Variable Equality

Instructions:

1) Read the context describing a series of relationships between variables.

2) For this task, variables are identified by a single letter.

3) At the end, determine whether the statement in the question is true. You may use:

(a) Information stated in the context.

(b) Any facts that follow from the context by valid logical inference.

4) If the statement is true, output 'Yes'.

5) If the statement is false, output 'No'.

6) If the context does not uniquely determine an answer, output 'Unknown'. Do not guess.

7) Output exactly one word (no punctuation or extra text).

Context: ‹x› is ‹equal/not equal› to ‹y›. ‹y› is ‹equal/not equal› to ‹z›.

Question: Is ‹z› ‹equal› to ‹x›?

Answer:

## IX. ANALOGOUS EXPERIMENTS ACROSS RELATIONAL REASONING TASKS

To assess the generality of the representational strategies we describe in the main text, we performed analogous experiments across various combinations of relational reasoning tasks (Section VIII) and models (Table I). Here, we present the results of these experiments for Gemma-3-27B on all relational reasoning tasks; in the next section, we present results for all models on the “family trees” task. The accompanying code can be used to generate analogous plots for model–task combinations not shown here.

Across models and tasks, the qualitative results were broadly similar to those presented in the main text, with the following notable exceptions:

1. Exceptions in settings with poor baseline task performance (e.g., Figs. S32E, S46B,D, and S81). When the model cannot reliably perform the underlying task, mosaic experiments may yield unexpected results. Because mosaic experiments use the models themselves to assess what information is available in the manipulated activation tensor, poor task performance makes these results dificult to interpret.

2. Exceptions in the additivity experiments (e.g., Figs. S42, S43, and S44). These exceptions were limited to the ordinal relationships task family. We speculate that they arise from redundant information in rows of A that follow entity-token rows. Such redundancy could arise if information from entity-token positions propagates to later positions (e.g., through previous-token heads [12]). Because the manipulation in Eq. (27) leaves these later rows unchanged, they may contain information that conflicts with the modified entity-token rows. Unexpected behavior could arise if the model retrieves information from both sets of rows at query time. Consistent with this idea, pruning experiments retain candidate name-to-role pairs at both entity-token positions and later positions (e.g., Fig. S70C).

3. Exceptions in the prospective–retrospective experiments (Fig. S104D). As discussed in Section V, such exceptions suggest that the activation tensor may contain information reflecting the prospective composition of relation ships.

4. Exceptions in the editing experiments (e.g., Figs. S68E, S69D, and S70D). We speculate that these exceptions arise from redundancy in the key–value memory. Specifically, the pruning procedure described in Section VI is designed to identify a set of key–value pairs for which retaining name-dependent information is suficient to mediate retrieval of all relationships. However, key–value pairs outside this subset may carry redundant information that influences behavior when the identified pairs are perturbed (see [14] for evidence of redundant mechanisms in another setting). If this is the case, editing only the key–value pairs identified by the pruning procedure could leave conflicting information in unedited pairs, potentially accounting for some of the unexpected responses.

## A. Behavior and generalization across tasks

A  
![](images/26279e0c6a1ebd1f3e83b42761b31dbc4c7dd4d32c63c661e6f09af9cd5c0108.jpg)

B  
![](images/e50a0706d9826045e4098e96a889dfba36152de9c027909c9a7dc37481254e71.jpg)  
FIG. S8. Behavior and generalization: family trees. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/03bf245a35ec7265fc14a44e114e9a955fa5c143b9bd2db367c44a0c87fdaa1b.jpg)

B  
![](images/342ec89fa331c26b3b11d427fb30ce8f85c756f3d8df42512954b0985807483d.jpg)  
FIG. S9. Behavior and generalization: ancestor–descendant relationships. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/24963ffb3672bdc1324deb063c5af320111cd45949d32393e34df91707d5c28c.jpg)

B  
![](images/4a438bba42fe1ce0a46cc5aff3553f274ae2c12c860d993fb004b52411780e71.jpg)  
FIG. S10. Behavior and generalization: company hierarchies. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context $\mathrm { i s } \ ^ { 6 6 } \langle \mathrm { A } \rangle _ { \mathrm { { s } } } ^ { \prime }$ boss is ‹B›. ‹B›’s boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/5f26fc63e01b5b7883d7e7e056a1ec648fffcbcda4b3ae6016fb880ada40ab6d.jpg)

B  
![](images/d085bf892c50dfb2669df355520e85166157123e21fa9e95a19799a4a0bb7409.jpg)  
FIG. S11. Behavior and generalization: age comparisons. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is older than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/fdeafd032544ede06b3ec5be9c2703313c1ebb92f50d3e772b65c51433b28ab6.jpg)

B  
![](images/573ad23fe49fec65849d475ef5e9bfe63f63267450bd09d417aed8b4502fd3fd.jpg)  
FIG. S12. Behavior and generalization: height comparisons. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is taller than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { 5 9 } .$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/8153e4d886609738dc37916baff1342016bf89ae90bdca9e31f5092b3011c385.jpg)

B  
![](images/50a15a82cbabc3071dc11a29f4dd5ca9f92c62e3fd8716ce44f7eb907612c3e8.jpg)  
FIG. S13. Behavior and generalization: speed comparisons. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is faster than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } }$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/7f783845724b54b2731060505afbdb0edf5fac1b5257b0298754dbdcb12f91cc.jpg)

B  
![](images/c15dc545f5bfe4c8d7db63c30f41b1fdc0d02a86bad73fab56cca71ea79dfab1.jpg)  
FIG. S14. Behavior and generalization: east–west relationships. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is “The $\langle \mathrm { A } \rangle$ is east of the ‹B›. The ‹B› is east of the ‹C›”. Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/60b0460e6f4d0136d7fc4788a4492878b544ef80333981271f2be0c8c17e6d19.jpg)

B  
![](images/d974ed802e6f145faa7c08d65fa6879431b73e951e6dfe736373eb6a486ddd19.jpg)  
FIG. S15. Behavior and generalization: north–south relationships. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is “The ‹A› is north of the ‹B›. The ‹B› is north of the ‹C›”. Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/202aee19ce9c965d232717f9168d37cb57a8730b356b75472ae3cbd5df7d0c12.jpg)

B  
![](images/a16bb29c46eaa454f89ff99f82fcc6eccae70f24126e3151c0494c939b6c4355.jpg)  
FIG. S16. Behavior and generalization: ahead–behind relationships. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is “The ‹A› is ahead of the ‹B›. The ‹B› is ahead of the ‹C›”. The wording “behind $\mathrm { o f } ^ { \prime \prime }$ is a grammatical error in the prompt template; the intended wording is “behind”. It is retained here to match the prompts presented to the model. Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/9f186710e82b5cd75e14af682f37ed15ff27b542b893230d736aa51e794bea32.jpg)

B  
![](images/97392c64b00a4168bc979004157fe366be31c65ecb312b9090f821c14cc3dcde.jpg)

FIG. S17. Behavior and generalization: exact age diferences. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context $\mathrm { i s } \ ^ { 6 6 } \langle \mathrm { A } \rangle$ is one year older than ‹B›. ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.  
A  
![](images/82a532707917dfa56b81e8d7d5deaf1caa31cc6da95ee8d6e92381fef8deba36.jpg)

B  
![](images/6271049470d51a52f131f05f0be1e1823a3a778e81d8051bee24d1f312519045.jpg)  
FIG. S18. Behavior: gene regulatory networks. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ activates $\langle \mathrm { B } \rangle .$ ‹B› activates $\langle \mathrm { C } \rangle ^ { 5 9 } .$ . For this setting, we did not compare domain-appropriate and synthetic names because we used synthetic names for the names of genes.

A  
![](images/6b5a5c7de578b6b7831f5802240ad20c5a2a60ebca3ad2b78c74ef3147f43a2b.jpg)

B  
![](images/64e5b1f06744a4464b0cbf8525a3cb025124eb6d777eb63fee4aa2694bd50982.jpg)  
FIG. S19. Behavior: neural circuits. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ makes glutamatergic projections to ‹B›. ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . For this setting, we did not compare domain-appropriate and synthetic names because we used synthetic names for the names of neurons.

A  
![](images/6684a7d5621864143b78dfb3c162ce6968b8f33483428c59b6a0db7b36c81738.jpg)

B  
![](images/2693721a7c4da7da7e98da0fd4badb90150ac546d91b98bf508e9dd9aeaa8657.jpg)  
FIG. S20. Behavior and generalization: propositional equivalence. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equivalent to ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { 5 9 } .$ . Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

A  
![](images/db81b703769e99e6e9b20b01da27aee745985ca16f69d67fbf388c1f333c8f02.jpg)

B  
![](images/9971d072e445b9a60e7219a7d5ae47582f977e8ce5fe2f20f9f01a52e0249d60.jpg)  
FIG. S21. Behavior and generalization: variable equality. A, Accuracy on a selected set of two-hop queries plotted against model parameter count for all models studied. B, Gemma-3-27B responses to the indicated query. The context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equal to ‹B›. ‹B› is equal to ‹C›”. Responses are shown for both domain-appropriate (green) and synthetic (orange) names.

## B. Identity–structure separability across tasks

A  
![](images/4c01efe70048ca95189f1f9d82d105ac655ae4a984eed822728e1035b2aca0b2.jpg)

C  
![](images/30dfb8218d4887dd7854e14e5dc91224ea013c12a54f31a68ef8eaa472039e60.jpg)

E  
![](images/335f909eab88f3effeaf1f37f093e452f1a61f757f1ee63da57d4c8363a52d74.jpg)

B  
![](images/d09d9532fa06d612149909b791f31f222b05672ee3db9658201af537667ba143.jpg)

D  
![](images/86fb15b5655bcd86822d342538e71400ce0cc6a61d2e19279c349a3b9001353b.jpg)

F  
![](images/9f6c1d0c666de7c82004e35b14988ff7ae0eaddab84d1f821949cf8b7de63e7b.jpg)

FIG. S22. Identity–structure separability: family trees. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/b7e065b07d3b5826101fa42becbc965ccf378d499ce0a8776d059a28e9868141.jpg)

C  
![](images/d84bb0f17c6bc8e5e843fb2ef1aa2dd50e7a1538376fc2eb1902adf8c1bebd25.jpg)

E  
![](images/2d7393857fb66f231583e23865b195663f5e3649631b7f965c2d4e61e24f7f4f.jpg)

B  
![](images/259b99d4a1b968b24c48b73d2ef78ab18fa7d1376994f157dca36c534f3e4083.jpg)

D  
![](images/ec9e1da7f41183e95e41d1c506cf1ffdeea25b4054fffcfafc647bdf5a49e7ff.jpg)

F  
![](images/aef12e2809cb585374ab65e4755b2e189f349e6bdc8043012d00425238043a44.jpg)  
FIG. S23. Identity–structure separability: ancestor–descendant relationships. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 2.

A  
![](images/4eac82b62a60b6feba02e903a38594fd97121f857a349f69cec17ac1b50b50dd.jpg)

C  
![](images/362a5d2e16500863c2805da772d3e93caf2d89679073b9b226b593c2f554f3b2.jpg)

![](images/894b8c4205f0e2d21e78c6af5009516685e7c35de9f37d1c05a1872171519e31.jpg)  
B  
D

E  
![](images/80365156e80ffbdbaaa2d5ddccc2ff79449b80ee1f64e88ad4bbc29bb8682146.jpg)

![](images/fb317d6736f8824c7e57fc87eb987a3a7af9181129658017eea438a30d050d7f.jpg)

F  
![](images/e4e282eabedce0b328f072625ca1bc3cd7bc4938f005c8c12b5fcc1771942ef1.jpg)

FIG. S24. Identity–structure separability: company hierarchies. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle \mathrm { \bar { s } }$ boss is ‹B›. ‹B›’s boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 } \langle \mathrm { D } \rangle \mathrm { \dot { s } }$ boss is ‹E›. ‹E›’s boss is $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/268b679615b82dc997c48659d32319eec912731df8d747568ad234ad97faa736.jpg)

C  
![](images/bad07008747640174674e1fbb7cc24b9c31e62b001174ccf2e1bb6c687cb5f3b.jpg)

E  
![](images/49bf7dfbf410568b3eb0f237682133fd6de83e8617734441da23b46b692248b3.jpg)

B  
![](images/765a6a09a6cc708cb31c71f0e25180b2436cb3af3d4035a8b303c5a8c6eeab70.jpg)

D  
![](images/31cbd2ce85ae04b5c2dd592de6d82a1c8212a4d35f293df70c3218764242ee27.jpg)

F  
![](images/e1bd97c4dcba7239de1dbcd7812d683c0074cdd21a6b3779de83bd580efc0664.jpg)  
FIG. S25. Identity–structure separability: age comparisons. Results are shown for Gemma-3-27B. The control context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { A } >$ is older than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › is older than $\langle \mathrm { E } \rangle$ . ‹E› is older than $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 2.

A  
![](images/0c71dd5919bc939987693b9d0548cf451b3bc49a164c38190021625a42b93967.jpg)

C  
![](images/4d811a20ef7a2728a54930a2dea1f0469dc35072c994652bd86a3afdf71cb428.jpg)  
E

B  
![](images/1f3ee6540ed4e4913dc5e2274d0f651b66cf469e7bf4a2c4ff3b5d404afbb59f.jpg)  
D

![](images/d80915ad0342e485607e24c569aa3e33f4da088e4a3325692f973ee43f74798c.jpg)

![](images/0eaf7597c49bb72fb343c9eb0cdcc2bc8603e6e8f865824fd927f631f5fc7cc0.jpg)

F  
![](images/d7c5a241aca7bf54e5ba105a76b7fd025f28f18dffe0778651c79df6ded34ba9.jpg)

FIG. S26. Identity–structure separability: height comparisons. Results are shown for Gemma-3-27B. The contro context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is taller than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is taller than ‹E›. ‹E› is taller than $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/a52f6fbfa56ff1641b6c76445c34645d63b66a3bf83555585ba0945ff672d9b0.jpg)

C  
![](images/d7797d7e421b58182dcb88d1cada740a734cf3b181b074ffc263c6b5c8879a24.jpg)

E  
![](images/05814b8cb546a8c5afb5bbe942aef1fa6b53ad06452dd0c5851e1b7bd39aa81b.jpg)

B  
![](images/fe8f7d368a456957058e91ac14e4b7ba4dd91d423684a8eb68e883a11d44c89a.jpg)

D  
![](images/3810e9a4ba29b1f3e349fe465ce0ff6b1cfd6a348dd0d55d6be357dfd62c9406.jpg)

F  
![](images/099e509346aa1a6077f64bcbfd8c114e59c5732cd8c6220c1abfcd604ae812f1.jpg)  
FIG. S27. Identity–structure separability: speed comparisons. Results are shown for Gemma-3-27B. The control context is “‹A› is faster than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › is faster than ‹E›. ‹E› is faster than $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/e3a74094fee4c1fb763fa3d50c103e9f9247b9b69dd1291ae870468435476480.jpg)

C  
![](images/58352f3a51ea4dea80476125ac82f6370babc51328d78b994b73da6b79ffa129.jpg)  
E  
D

B  
![](images/86893be8d464a12c765a778e78093b06249486bb0ef8a8fd39f886cf7dd24fd5.jpg)

![](images/13a665e2fd4d40f7ae3bc86351f957fa2ff716703156b8d83c5e3140f3f567a7.jpg)

![](images/5999b9659d3dc60f00621020feef54a348c31384a727a2c3ee7c0b306d50c05e.jpg)

F  
![](images/864405d7988043165211a9e400c280cc3a5ba0144536290a557db8796bf1e0fe.jpg)

FIG. S28. Identity–structure separability: east–west relationships. Results are shown for Gemma-3-27B. The control context is “The ‹A› is east of the ‹B›. The ‹B› is east of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹D› is east of the ‹E›. The ‹E› is east of the ‹F›”. The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/3b2d93920ca298dc33755b3b70d1e805e034f6ccc0af23cd7f89696dc865fc39.jpg)

C  
![](images/0ed48a1f1553f5e9bb1a5a7c2ac5671b179c6859832b9bc35681705e3681c316.jpg)

E  
![](images/b76c76acd8e7e7ed02cb26b41d650fce662c666b9e72b383677558a1f638dc18.jpg)

B  
![](images/3cde1bb20ab37c1e3adf203fced6d688fe8e6f0adb5b41bf365d0828ee36d962.jpg)

D  
![](images/9fc3e727ffb29b639b0daa1fc21ec6b005eb286a9a2d41393182d1331e6ab306.jpg)

F  
![](images/66080d1ca143bac724b3b1ab9e30dc04113d0731b19bdaa084e5f7fc02c252db.jpg)  
FIG. S29. Identity–structure separability: north–south relationships. Results are shown for Gemma-3-27B. The control context is “The ‹A› is north of the ‹B›. The ‹B› is north of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹D› is north of the ‹E›. The ‹E› is north of the $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 2.

A  
![](images/6ab5ed2090bde635a9c7b99a81b9ae6d1e84ffe56f5d58a0d722f3edefeb409d.jpg)

C  
![](images/6251f96604745c65a5148e20a8eedaa492a535c515888f4dcf1372800e1d5a62.jpg)  
B  
D

E  
![](images/251ad76776a94042045646e1004f0208ff4ad405ff103e0f34677d9a7e55ddd6.jpg)

![](images/a38daa450d73eb0d128626abcba1480619f13b1ee1e0bbb3fae4301996d51f95.jpg)

![](images/0c4b841d8a52d97250dac846ff60ed62163b78a5dcb30e35f1ac48a39b04ad58.jpg)

F  
![](images/e2dd71a8d3e3ae8230fa97daeb61e224bee7a9be0af513ea7ad5096a9c270411.jpg)

FIG. S30. Identity–structure separability: ahead–behind relationships. Results are shown for Gemma-3-27B. The control context is “The ‹A› is ahead of the ‹B›. The ‹B› is ahead of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹D› is ahead of the ‹E› The ‹E› is ahead of the $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 2. The wording ‘behind $\mathrm { o f } ^ { \prime }$ is a grammatical error in the prompt template; the intended wording is ‘behind’. It is retained here to match the prompts presented to the model.  
A  
![](images/c4f96b48a7a8a07188bb38c83cee36724537ce32e695bb99d3e0d07a1d7ab023.jpg)

C  
![](images/908ba7e3600696380db239396b2fd132f3e7fb9ffee6debaac09dcd42e750ef5.jpg)

E  
![](images/908cb03050325b82d4171e52aca78850bf3a11615377602b3b3ec5dd40c55f89.jpg)

B  
![](images/bb58d8de6f4ede2a1890025daf24cc9378ee90b97db9c94c1957d690366e8e75.jpg)

D  
![](images/1793afd570c90077a8c91e5ea5925ed7bd26d416e687a41f1d31b08ba4ffaa6c.jpg)

F  
![](images/e06793bc0c3368dd8298ae52d62a737bccf1a54b7b7684f0764d30039f3ad288.jpg)  
FIG. S31. Identity–structure separability: exact age diferences. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is one year older than ‹B›. ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is one year older than ‹E›. ‹E› is one year older than $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 2.

A  
![](images/0aa690bd7fae355ad952e4c4e4b86c4aee313d936b58dac2884126e7928ef7c5.jpg)

C  
![](images/55149e8754ae1bafcd863ffce4c0e03c6c06e1889ec64b6d159f379d6e0d549a.jpg)

![](images/3b8d22fe641b82c1b105dac0dd998fcdc2fcb2cd19a453a508c4b6463a49c405.jpg)  
B  
D

E  
![](images/1180a048f09007181defdafb6ec2a95b2d35526135e45bdd2ffb0d3fcdb03058.jpg)

![](images/6acd61eb37c0540645c6b3a487ca0ba01254111a503eabe7f5779bf585308fc4.jpg)  
FIG. S32. Identity–structure separability: gene regulatory networks. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ activates ‹B›. ‹B› activates $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “‹D› activates ‹E›. ‹E› activates $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/5b195fd6f01a1907692f0f11a34dca9cbd7c4e8c21f4714e42c9d86fc7c2d731.jpg)

C  
![](images/028ff7ed5bf5b3a47a2007172039041dc5f22ab8ba0d73c8dc8a046fcc56ec69.jpg)

E  
![](images/6fdb9fd11b0080adf7039775f534738ec719e97ad3d416ea2d7af904b5e8111f.jpg)

B  
![](images/9b674a319940534c1a31bf767a6c095c235e87b00b03b711d51e73c708799571.jpg)

D  
![](images/c28a1db5503bfdcf9a3ff267c4b4f48f7abc7d9bd87086d82ed5108c806473e7.jpg)  
FIG. S33. Identity–structure separability: neural circuits. Results are shown for Gemma-3-27B. The control context $\mathrm { i s } \ ^ { 6 6 } \langle \mathrm { A } \rangle$ makes glutamatergic projections to ‹B›. ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ makes glutamatergic projections to ‹E›. ‹E› makes glutamatergic projections to $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/fb338a3a1b99963129b7e485f18a1ead48ec97487924207cd2c08ce80a7c28d5.jpg)  
C

![](images/c77d365d5a53206b27f95f8181fbf1c55cc283b041d154c14b3e9df2b3d91434.jpg)

![](images/4cdd9299842d72a48929d23bb445c3798c34106c62f2788f2077936f197ecce2.jpg)  
B  
D

E  
![](images/43cfbcf792c4803d6992b74bed2de9165ac02e5046b85d56966f1c8db52985d3.jpg)

![](images/f45033c8284535b0ca1e5c7e28ede7c61d43b59bb19a63fcb114bdc26dbb0a37.jpg)

F  
![](images/de4cc158ec4e3e7cab9e395183c586d0d037ee106e36e14db61094366be26dab.jpg)

FIG. S34. Identity–structure separability: propositional equivalence. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equivalent to ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is equivalent to ‹E›. ‹E› is equivalent to $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/c23aa2fea9e5e152b8ecb94d17d0473bf7176681eb1a95d0ad9ee4df68d0fb0d.jpg)

C  
![](images/c0cd0032eb3a267d42027421d35103591f2b35efe88889afef8057752c7da181.jpg)

E  
![](images/4de6672f353baa5e33a85b00636bb6418353be279b85bf37d189c5ec9d7e041a.jpg)

B  
![](images/26d40a5fe6e8191817d25ea47fc93fd196043590310640e28aa05f02c2be092f.jpg)

D  
![](images/20a667ec2e525faa10edad22dcf7446395ecc028107edb1c7acca77c06dc0c10.jpg)

F  
![](images/ddafa76b13b61df37bca55d375e5620f1ebf4086dc798d19930141ad155e1571.jpg)  
FIG. S35. Identity–structure separability: variable equality. Results are shown for Gemma-3-27B. The control context $\mathrm { i s } \ ^ { 6 6 } \langle \mathrm { A } \rangle$ is equal to ‹B›. ‹B› is equal to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ is equal to ‹E›. ‹E› is equal to $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.

C. Additivity of representations of identity and structure across tasks

A  
![](images/8b621beaf7739ac1df03190f45ed4b08f9ea2304a300c67b0ed949d9a250b4ff.jpg)

B  
![](images/de7f7d5f6adf750649d3737b1093c5b17a869ffb1838ef0c1747e25d82bc943b.jpg)

C  
![](images/dd07ce05be1b29364f13fb0c9b1cca5467daec64e76c0967175b81fb25088855.jpg)

D  
![](images/d265005d28e82d3e363c5f9c29642927b061c0b186b375b7ff08f68423c52906.jpg)

FIG. S36. Additivity: family trees. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter $\langle \mathrm { B } \rangle$ ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/871c98ba2eb3723be4d70b509bc753fd69ba655e669407981c951967e2dd6011.jpg)

B  
![](images/fb7a39ecce547c6c587fde9f7e8af6d62b04d3848fc366bae42f4e8c6ce73f87.jpg)

C  
![](images/9bad2604ea28df1ac17f5d545e87d8375c0b966f75afc08ddd631d497c81a7fc.jpg)

D  
![](images/5c98dc7699c2881ea20572db5a358b78e06499c349bc7c0a79dbab4c372513e1.jpg)

FIG. S37. Additivity: ancestor–descendant relationships. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor contexts $\mathrm { a r e } \ ^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/ed963d9d7173c820e70aadd86748fe8f1dbfa2b27ba1a89b7229d8daa0f55b1b.jpg)

B  
![](images/b8588b419e728c69906f159bf32805aa1d1aecf5a69649705a104304de64c864.jpg)

C  
![](images/0d43575f95a59fbeea34da63a3c7ed150647e3e33615c2c801bdf8c443b28f5c.jpg)

D  
![](images/a800d87bc71750843cbd2a5a374bc4976cc5b452a1a97f080f413db399999349.jpg)  
FIG. S38. Additivity: company hierarchies. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle \mathrm { \dot { s } }$ boss is ‹B›. ‹B›’s boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle D \rangle \mathrm { \bar { s } }$ boss is $\langle \mathrm { E } \rangle$ . ‹E›’s boss is $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle \mathrm { \bar { s } }$ boss is ‹E›. $\langle \mathrm { E } \rangle \mathrm { \bar { s } }$ boss is $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.

A  
![](images/bd4894eb54e81369de7fa382898d6829d1e24cb13d66b73418d91f17845c08f5.jpg)

B  
![](images/8a25ad24c12949060169d9af0684cadf6102eb82ff388d1f846c87e7c837db5a.jpg)

C  
![](images/24a40fe8c26bb3d5c9a8e0c766e5229c3e0304db5a9ce681ca5f5e6591341a15.jpg)

D  
![](images/3874295d28f37727603fc670d7365b8b1da5da975b9a621e580cf1fcf7b95225.jpg)

FIG. S39. Additivity: age comparisons. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is older than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ is older than ‹E›. ‹E› is older than $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is older than ‹E›. ‹E› is older than $\langle \mathrm { D } \rangle ^ { \mathfrak { n } }$ . The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/666e7a5219c35ac3da4a3d0d0c4d591e7c91a169ec536093341955219365e58e.jpg)

B  
![](images/65db044c12c6c62a9aa9dc2e00460dd2ee4937ee899e82e17ada98ebbbd3b811.jpg)

C  
![](images/644b66cc0834e527b79b3fd0ee583a6df25623090a3bffc79908ab08e12b81ec.jpg)

D  
![](images/b9cbe37712fdf5bf3dd4cf3da36182bff292f1a746a3ec16241d11fee891e008.jpg)

FIG. S40. Additivity: height comparisons. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is taller than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ is taller than ‹E›. ‹E› is taller than $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is taller than ‹E›. ‹E› is taller than $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/638ddb817032556e1b320361e1f72fe16f68d4c2231cec335cae908a0f987a23.jpg)

B  
![](images/3419bc426791d6e48fd66758fd6d0751f2141ededcc0131f23df5c3bd6ba81cc.jpg)

C  
![](images/22cba53f215e48a6d5013fd64fa41c418172d0c3aff2c2f9c9ba054001c4edb9.jpg)

D  
![](images/82754aaa1f51fb359c6be90d352fe87470cf3e42810c225ef4ce57519be0e615.jpg)  
FIG. S41. Additivity: speed comparisons. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is faster than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ is faster than ‹E›. ‹E› is faster than $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is faster than ‹E›. ‹E› is faster than $\left. \mathrm { D } \right. ^ { \mathfrak { n } } ,$ . The mosaic was constructed as in main-text Fig. 3.

A  
![](images/27460ac4347da1ef102ec0b4af0bd79aaed7b6bfacb23ce317d059ef3bd0c0e8.jpg)

B  
![](images/bc4d710e921355a588e7edc1cf89e2f6da17a3db42f30f459c9ffcae189e2079.jpg)

C  
![](images/f94f670782c5c734f10e8ddb28f1139ab36e09aed731121f3ba2dbb427be8f81.jpg)

D  
![](images/f34c6c7a9ee31a08076883fb2c46f03c0bd6975ac4c53ff08d59ba51e88e79ed.jpg)

FIG. S42. Additivity: east–west relationships. Results are shown for Gemma-3-27B. The control context is “The $\langle \mathrm { A } \rangle$ is east of the ‹B›. The ‹B› is east of the $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor contexts are “The ‹D› is east of the ‹E›. The ‹E› is east of the $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and “The ‹F› is east of the ‹E›. The ‹E› is east of the $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/de66a19ceb85f6f4d755e2a3e20a83234a1dd1174ae3bf139b1270ba8b8b8e19.jpg)

B  
![](images/f6e9b498f90ddc91f622c205b9856c03ccd0617fecb142c0d7834503fb4f7f77.jpg)

C  
![](images/f96d43530303297d7cc8de5e22530eb8aaad106565caffcc710d212ae9e732a9.jpg)

D  
![](images/21debcb33733291d636b64ea0f0c0082aeb5567dd28dd99fc90d368652eeac9e.jpg)

FIG. S43. Additivity: north–south relationships. Results are shown for Gemma-3-27B. The control context is “The ‹A› is north of the ‹B›. The ‹B› is north of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are “The ‹D› is north of the ‹E›. The ‹E› is north of the $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and “The ‹F› is north of the ‹E›. The ‹E› is north of the $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/f6dc5737d5fdcc95d2bcdd90d12db3845d338268fedfb83a42fa5f54417afd3f.jpg)

B  
![](images/21804d649dd4b418a2b7002f8027308949271b643e0df6c89cb4b26f44c94bd2.jpg)

C  
![](images/d609e5ad10c19097365141208dc46303b8a61f51af74d38bb7791496dafb99e7.jpg)

D  
![](images/4abac9f3dbcfc767a95c303363462429310972a3096b4fd0708fae02983e42e3.jpg)  
FIG. S44. Additivity: ahead–behind relationships. Results are shown for Gemma-3-27B. The control context is $^ { \mathrm { \mathfrak { s o r } } } \mathrm { T h e }$ $\langle \mathrm { A } \rangle$ is ahead of the ‹B›. The ‹B› is ahead of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are “The ‹D› is ahead of the ‹E›. The ‹E› is ahead of the $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and “The ‹F› is ahead of the ‹E›. The ‹E› is ahead of the $\left. \mathrm { D } \right. ^ { \mathfrak { n } } ,$ . The mosaic was constructed as in main-text Fig. 3. The wording “behind $\mathrm { o f } ^ { \prime \prime }$ is a grammatical error in the prompt template; the intended wording is “behind”. It is retained here to match the prompts presented to the model.

A  
![](images/f98a0801c9201bf06da5000e6a9365411c27dc17a2625e89065c97517f4ee45a.jpg)

B  
![](images/adec0b51c59bfe750abd652fe01cd3ef74c97aba0ffbe02c313c8f55976dc00f.jpg)

C  
![](images/5017589513a18a6f34096d54513675b544569f11c5501d209b847f847bdfbcef.jpg)

D  
![](images/c60a613858d53125a9fa8950631da247c22eac1b55542db10c067badd7090da5.jpg)

FIG. S45. Additivity: exact age diferences. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is one year older than $\langle \mathrm { B } \rangle$ . ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ is one year older than $\langle \mathrm { E } \rangle$ . ‹E› is one year older than $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is one year older than ‹E›. ‹E› is one year older than $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/966bc9774b9d4beab5b9b57d40aee2953eed05d8cf8c990154f9a4fff3d8cb50.jpg)

B  
![](images/7d93b9a0a55ab1e9ed375149f793f38cdda2f8d3ca2e80a025b030781aee2e65.jpg)

C  
![](images/8327dc7789baeffc1826c829c77983059a8a24fc396ccca89cff5f22b095b2d2.jpg)

D  
![](images/5f12fb8efbba85573663e2756a819bab5b3acbd3b732f1812c16d4ca1ebe2b0b.jpg)

FIG. S46. Additivity: gene regulatory networks. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ activates ‹B›. ‹B› activates $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ › activates ‹E›. ‹E› activates $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ activates ‹E›. ‹E› activates $\langle \mathrm { D } \rangle ^ { \mathfrak { n } }$ . The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/17e605134fa47c4f00ed54695d663273eb58df52cf481b214982571baaf13136.jpg)

B  
![](images/9c92aa4243dd5dba76f1de527d9c2b7e4594fb4d3231476b251a2df55c5fcbbe.jpg)

C  
![](images/3c9bc9802dbb211a0b0502c04cbf520b2bb9b72d47dd4f393ee7c1bea0fab203.jpg)

D  
![](images/0ebca119848ec046e39aa9974794e09ed3c7e8c1cc10c8fd4a7395b755d2b53a.jpg)  
FIG. S47. Additivity: neural circuits. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ makes glutamatergic projections to $\langle \mathrm { B } \rangle$ . ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ makes glutamatergic projections to ‹E›. ‹E› makes glutamatergic projections to $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ makes glutamatergic projections to $\langle \mathrm { E } \rangle$ . ‹E› makes glutamatergic projections to $\left. \mathrm { D } \right. ^ { \mathfrak { n } } ,$ . The mosaic was constructed as in main-text $\mathrm { F i g . 3 } .$

![](images/4bd4079ac66628ac79e3c2ca58f6ce84a676178e6a55de41c3d209c59cb0cd5f.jpg)

FIG. S48. Additivity: propositional equivalence. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equivalent to ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } ;$ the donor contexts $\mathrm { a r e } ^ { 6 6 } \langle \mathrm { D } \rangle$ is equivalent to ‹E›. ‹E› is equivalent to $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is equivalent to $\langle \mathrm { E } \rangle$ . ‹E› is equivalent to $\langle \mathrm { D } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 3. Because exchanging entities $^ { 6 6 } \langle \mathrm { A } \rangle ^ { , 3 }$ and $\mathrm { ^ { 6 6 } C O } ^ { 5 5 }$ leaves the equivalence relationships unchanged, these panels cannot distinguish whether the two entities have swapped roles. We nevertheless include them for completeness.  
A  
B  
![](images/92875a29c4e5da2d348ef703e6d28b519014a9c93b26c0d1b1a8df30ccd11067.jpg)  
FIG. S49. Additivity: variable equality. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equal to $\langle \mathrm { B } \rangle .$ ‹B› is equal to $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ is equal to ‹E›. ‹E› is equal to $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ is equal to ‹E›. ‹E› is equal to $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 3. Because exchanging entities $^ { 6 6 } < \mathrm { A } > ^ { 5 }$ and $^ { 6 6 } < C > ^ { 5 }$ leaves the equivalence relationships unchanged, these panels cannot distinguish whether the two entities have swapped roles. We nevertheless include them for completeness.

D. Retrospective and prospective information across tasks

A  
![](images/f0464cadbae8c033144eab434e6df935f492f4389719e92cc512876287a3c753.jpg)

B  
![](images/66cf21ac5264c14440143604268dff5c1c824204c26ee2e5e3b94f9968666506.jpg)

C  
![](images/113de808ec9e83b67a7a34bee042d848028418a1d8d00bd4128e35f2b152835d.jpg)

D  
![](images/ee667ec5decf8d5ada79a106619e9938941e2dfc2f77275085508d06417942af.jpg)

FIG. S50. Retrospective and prospective information: family trees. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter ‹C›”. The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/2b2b1d64b08dea37126381beb9e377fad5db9a925a04f5a2e8028f782b320df1.jpg)

B  
![](images/9065e0b814bfefe245cb3270df75a1f71f76181afb933fe58b6264a9b7cc31cd.jpg)

C  
![](images/e1a4f907c2c56e59e78c4d311effd6b506da3260fb74268523ff0e23e0ae0c7d.jpg)

D  
![](images/8e7443e74f09929b726b7c3e0ea9c0a0ad6b043b905f7404523f7e588aa88e1f.jpg)

FIG. S51. Retrospective and prospective information: ancestor–descendant relationships. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/9930bdcc356ba1551c8c2981209b522413c608c3ceccbd16ea33edb8d87f79c3.jpg)

B  
![](images/cc68728bb8001ae924117e03bfde39d04e2cc35f5b19db5d708aa7ba789cb018.jpg)

C  
![](images/10a2697515f1ec4bc53953664f72377762df41c179d1822443582913d8e853c1.jpg)

D  
![](images/a80e07fa06049aa13f89256466f6264dfefcb9f293c1df6e20abd3a6c31c9ada.jpg)  
FIG. S52. Retrospective and prospective information: company hierarchies. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle \mathrm { \bar { s } }$ boss is ‹B›. $\langle \mathrm { B } \rangle \mathrm { \bar { s } }$ boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle ^ { \prime }$ s employee is ‹B›. $\langle \mathrm { B } \rangle \mathrm { \bar { s } }$ boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ The mosaic was constructed as in main-text Fig. 4.

A  
![](images/81b035201fbb19c94d32c8da0181e48b9fda11786bcd394e6afc473e41b3e3cb.jpg)

B  
![](images/c95aaf144824c4cfc1a2c0eb189635c181b9d3772c39783da4d7b7728fdf0a53.jpg)

C  
![](images/ae861b9c8c7a28adc312508b3caeebd8ab4706f6aaa59222540279ac4f17e0f4.jpg)

D  
![](images/1e32a2a9955b2c72ce9f4e34b24f9d97a991df12ebf1856cf7e97ea1c5f2b1c8.jpg)

FIG. S53. Retrospective and prospective information: age comparisons. Results are shown for Gemma-3-27B. The control context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { A } \rangle$ is older than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is younger than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/8bfceb88d45bcbcdec009de64b5f9efc9e850a6479bc5a9fc4a4ff011c12fa54.jpg)

B  
![](images/d19a650e97c5cf4829d0750febc9adab59fd40770600e02742a2aeb5bddb20aa.jpg)

C  
![](images/04bacd80edcc655842bef96018b21337c91cd0f8cc9b4d0df44c4d400854207d.jpg)

D  
![](images/400dc6b14eee12a7fc5f61e956684d9abb35c9fb5095162c1602d76669a3a7bb.jpg)

FIG. S54. Retrospective and prospective information: height comparisons. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is taller than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is shorter than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/78b33bbd0b7c15dbfd7f9e5b94ae6481e058bd845b8d4d9711198ece72ebe521.jpg)

B  
![](images/b2ef61db4579d71afcdfc7f4f1167169d3367b9f97da7e74c8b51efc89a725ee.jpg)

C  
![](images/ba89e95d85e840709cd470dd222ddfaeee94045c6608687f88a3cdd253278878.jpg)

D  
![](images/7efcb29b2646a5eab60c0dd11d7f686d475458bf4172496ea0641e627bf9d13e.jpg)  
FIG. S55. Retrospective and prospective information: speed comparisons. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is faster than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is slower than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 4.

A  
![](images/a2f40f6b131424f03953b2de4c3d3c7bef9ae0a30097f95792e2b52218ff6a89.jpg)

B  
![](images/e3ec3a57e373d2ecc8211310667e0d69c09aa37609ee697dfb4b23c1ae6908be.jpg)

C  
![](images/d631ee0bfe08e7242d12c6a74c864d4e89b380d57df02272683b02d1dd6af276.jpg)

D  
![](images/40b52385296b60431a256aad8d8b6a170a964be3d3078c4dfaddc0bda8d48089.jpg)

FIG. S56. Retrospective and prospective information: east–west relationships. Results are shown for Gemma-3-27B. The control context is “The ‹A› is east of the ‹B›. The ‹B› is east of the ‹C›”; the donor context is “The $\langle \mathrm { A } \rangle$ is west of the ‹B›. The ‹B› is east of the $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/0cca344dbf99d9a1cc78f1904f9727f6ec56f1b5ce0c8193f80ccd18b4fc81a6.jpg)

B  
![](images/827a76413cb1cce75206523e5e7c227d7b8219767b93ca7f5b18114ffb431853.jpg)

C  
![](images/d3959ea3458d56c8b33adb99d72b110082a3af532f6224d4e858d29141df3fcf.jpg)

D  
![](images/2a67ef7948a9d7a261d02e378f49d67f1c59d62605414b576d14fc5ee5143a4a.jpg)

FIG. S57. Retrospective and prospective information: north–south relationships. Results are shown for Gemma-3- 27B. The control context is “The ‹A› is north of the ‹B›. The ‹B› is north of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹A› is south of the ‹B›. The ‹B› is north of the $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/62ddd26c4f11942fd507f57286525ec019914e79b264202bf2139b36cb0f87a6.jpg)

B  
![](images/3fc3558a42b216c2b29da9c1bee64a33546e56b1bd342cb9d9ecd858b82ecf44.jpg)

C  
![](images/aa0cabee6c9094b7bb67083394c9f5c47b53aae19956a976f2feaa2c2b180115.jpg)

D  
![](images/414caaeea75e327e1db9792814c751f3a7dd740a1e59cb87082cef5d67dd3811.jpg)  
FIG. S58. Retrospective and prospective information: ahead–behind relationships. Results are shown for Gemma 3-27B. The control context is “The ‹A› is ahead of the ‹B›. The ‹B› is ahead of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The $\langle \mathrm { A } \rangle$ is behind of the ‹B›. The ‹B› is ahead of the $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 4. The wording “behind $\mathrm { o f } ^ { \prime \prime }$ is a grammatical error in the prompt template; the intended wording is “behind”. It is retained here to match the prompts presented to the model.

A  
![](images/829001d2940b2ae7c6f6fe27682ac4941792326876bc9d4339db723ef6124c5f.jpg)

B  
![](images/c11aebf02d60d14c7a4cd40dbcacca7e611abd90a9d59cbd898b7a78bc8f124f.jpg)

C  
![](images/a17e3f4336cdc1339d2d4029ba6306646554b9ef17625795654e5ddeaa450dbb.jpg)

D  
![](images/08b0b01041d82335e0d79c2fa9f826cae64cc0102cf3b3b809065972834d383e.jpg)

FIG. S59. Retrospective and prospective information: exact age diferences. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is one year older than ‹B›. ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › is one year younger than ‹B›. ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/49d7e9639377b5b70e02241f1429ed9929319a8bcf3967214909676c474d1451.jpg)

B  
![](images/40446410c6b4ee644b6feb25f828afcd3264435b2836f8105b21de16c85f0dec.jpg)

FIG. S60. Retrospective and prospective information: gene regulatory networks. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ activates ‹B›. ‹B› activates $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ represses ‹B›. ‹B› activates $\langle \mathrm { C } \rangle ^ { \mathfrak { N } }$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/5933c4802c2b36b52237b21177fef3f608d0e822dc5cf4be93f93524bb31f4b5.jpg)

B  
![](images/e2b936f28af658237207caa7a76a25e7cfd0f34fee6ffaeefeea04c3aaa7efb5.jpg)  
FIG. S61. Retrospective and prospective information: neural circuits. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ makes glutamatergic projections to ‹B›. ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ makes GABAergic projections to ‹B›. ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.

A  
![](images/f8e2c566865523303151a23ce00ce151944d99d9c6a46cd564dc80be5f46f6f3.jpg)

B  
![](images/85922269a663a8abf19cb0aaa5806dbb9cf08c31558bf363f773692fc0f85067.jpg)

FIG. S62. Retrospective and prospective information: propositional equivalence. Results are shown for Gemma-3- 27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equivalent to ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is incompatible with ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/9bb9d2043724aa6557e803443fafc897e6761cf1a277cfd1c6bd4abd8df652dd.jpg)

B  
![](images/91782aeddb87639b608418ee173a21ff5d329d7d0fc8e62d8cba5381a1c96329.jpg)  
FIG. S63. Retrospective and prospective information: variable equality. Results are shown for Gemma-3-27B. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equal to $ { \langle \mathrm { B } \rangle }$ . ‹B› is equal to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is not equal to ‹B›. ‹B› is equal to $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.

## E. Key–value erasure and editing across tasks

A  
![](images/fece1728614d66e615220e69997b7bbb2fc07e6b8b8ff251f1059607eedfa4a0.jpg)  
D  
B

![](images/f3fd66b3acc694f2241a7173b5b802c4810d587dabf6d4ce92c019b65a9c3e22.jpg)

C  
![](images/5ebd9ea6cdd24d9c94c60dfc3c85c4532fb696c52a5ae8763dfa1e629410ade0.jpg)

![](images/13b4db867c315ebf949a44c18c2fa7116f5cc0e63256b302e7c87e5658b1c2c1.jpg)

E  
![](images/86f44e14f414749049041cc313eb851d3a21a66dfd2b257377f8cb8bf0b976ee.jpg)

![](images/c5e2791cb8ab0529b1b9180de3fe5ffac8232583a45458c4e740f10843aa2aa4.jpg)  
FIG. S64. Key–value erasure and editing: family trees. Results are shown for Gemma-3-27B. Procedures follow main text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. $\langle \mathrm { E } \rangle$ has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 176 keys and 20 values retained by this mask, together comprising 0.99% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/d3720f758a01c66c3ab6de2745a4f3b4dc6fc3cd6f10524a068d5b05db63ab82.jpg)  
B

![](images/2ee9cd39b46b24597b8bfb9d4841e692e697a4135bc85ac758ec05fff9e4278b.jpg)

C  
![](images/a1e0b1d307d7ead42c8557eefd030d23b2a845a5d4afede0f3e48c047811b43b.jpg)  
D

![](images/23f270757454db8a03d6b904b472e82fd0bcd9c7f64c1e2ff9c9f207ff25b383.jpg)

E  
![](images/51c4985a387dc3ab7aad2fa4b2a595ce38c8833ccad4a018c9072aa202330d3c.jpg)

![](images/32ac599dd7603de4c45834fffbcbd26ea9e3e4ede12f6bcdbe27bfd5beceabb1.jpg)  
FIG. S65. Key–value erasure and editing: company hierarchies. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle \mathrm { \dot { s } }$ boss is ‹B›. $\langle \mathrm { B } \rangle \mathrm { \bar { s } }$ boss is $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 } \langle \mathrm { D } \rangle \mathrm { \dot { s } }$ boss is ‹E›. ‹E›’s boss is $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 193 keys and 30 values retained by this mask, together comprising 0.80% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/3db2da6abdd5528ea870939325f0a8ba78b7968a05aa8860f3e019812a5a47bb.jpg)  
B  
D

![](images/f3385e17e33a001517b7b0589b62a11ce168223b316515a908989d7d3766665a.jpg)

C  
![](images/a0129574b70ac87da3467dab4a395e0c0a55e8e27cc966517a9832e38ae54fd5.jpg)

![](images/18ca315132edb3ac36226f3a47b7694a7f07f06ece825088ba929f2c87b00141.jpg)

![](images/0a10ebf0698ba883da2379cf9584a7541e85d9a59deadfb9ee7cb0f46b5002c7.jpg)

E  
![](images/9aa9b862b88fe2dc5f5a57545802e842145a01a5df43ae7fed08552c91dc52b3.jpg)  
FIG. S66. Key–value erasure and editing: age comparisons. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is older than ‹B›. ‹B› is older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is older than ‹E›. ‹E› is older than ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 833 keys and 112 values retained by this mask, together comprising 3.97% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/44fdf2a98a6673c871a4c79fafbfba0d4a1dda61e2815fe17e2c4ea6c5b16c49.jpg)  
D  
B

![](images/44121c067d47ac6058817d8ddaebe32857fc7a2dc901ff6f7a52ee11a90449ab.jpg)

C  
![](images/a57ad10132d5b19525a07447eba76dda875da4d1dbe31051ae6d8b865ce07ad1.jpg)

![](images/860e51ae3ef391f36836f9345d324644b141c4b99ae566e7effa50290dffb427.jpg)

E  
![](images/ae16a941d32a494bd8b027ec814a7477e3a32919d8ca1c40f55b1ba02ceac108.jpg)

![](images/b1afb0a9b64a0c0488aed240d7ee1a052f5c7e3c0e9f0d6e01b0062f6eb770ab.jpg)  
FIG. S67. Key–value erasure and editing: height comparisons. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is taller than ‹B›. ‹B› is taller than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is taller than ‹E›. ‹E› is taller than $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 1,034 keys and 263 values retained by this mask, together comprising 5.45% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/c21678c8d683f87e629356dbd8dbe001e885ae2f1a4da25b76f1171010b86580.jpg)  
B  
D

![](images/f36fce687bde01f7810bb637d8b43337dceaba3babd5ff0e2722a1f4bf02030f.jpg)

C  
![](images/ecbfc09f456ca6611e7e2d0de0fa16e9d4ec1dedb5728a6774007ad6b38dcc24.jpg)

![](images/2cce68a5cc4d985c124f33fe7908f21e9d5785898f231cef969cd00bb3797a30.jpg)

![](images/2d6d1c87106a8000dd3db7f780769df02c2f9edc5c61869ab320be5aabfa72d7.jpg)

E  
![](images/5ef47344f1346d440c497d6f4563639d2d75e35490ac02de5e7bc1d85db0424c.jpg)  
FIG. S68. Key–value erasure and editing: speed comparisons. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is faster than ‹B›. ‹B› is faster than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › is faster than ‹E›. ‹E› is faster than $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 307 keys and 29 values retained by this mask, together comprising 1.41% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/d0575af180c308a3a31f05cea92f6fdc9aedf54fa0e2db7195acba6dc236a442.jpg)  
B

![](images/8e5d09211d5c15bd56481b139d6f206d1594e8005b05554644b546fed852501e.jpg)  
D

C  
![](images/0dec81ffa8ccb14d14158dcf42bde4e23005a9fae7426803f899ce119ab9a998.jpg)

![](images/8d130b0bae2f75277ee637db3ac27694f0b60bddffac9e843ee948de2726a5aa.jpg)

E  
![](images/353962cef0dafb9c793fa6ac1b67465358dd79634861a2d064cb58d73c57ff21.jpg)

![](images/08c977b71c2934ded7bb052ab45b563f249c9bc5ef3914dbc6647400e6310273.jpg)  
FIG. S69. Key–value erasure and editing: east–west relationships. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is “The ‹A› is east of the ‹B›. The ‹B› is east of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹D› is east of the ‹E›. The ‹E› is east of the ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 585 keys and 41 values retained by this mask, together comprising 2.10% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/80121b5fe465ed3458f17ea0b9182b06d969f23674ea2410ad07757f5803c129.jpg)  
B  
D

![](images/a7ae2cda427464526e6bfbbb4c4edd0784bcf23e8b6fccf99bc9fba42ee7492b.jpg)

C  
![](images/b4d4285176883c1f6a35a0ce853c64ea3c1896e28a658da972a8b4018f2f4f5e.jpg)

![](images/8e77503ac6bb083e277d5879958ab14c26d5b81a959488324e96082b7628a012.jpg)

![](images/8137fdec0570590790956bb87cafb1ed4b1c82f6e29bf5a34102e1e11476746c.jpg)

E  
![](images/dd09632a4e2c2d9b65e97035e6984df235fc3e066c1ad9466ade775ad46aa09c.jpg)  
FIG. S70. Key–value erasure and editing: north–south relationships. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is “The ‹A› is north of the ‹B›. The ‹B› is north of the $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “The ‹D› is north of the ‹E›. The ‹E› is north of the $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 485 keys and 21 values retained by this mask together comprising 1.70% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/404e63d2f1cf6188e5d7f77d10c66ce5ee5884d0d04234c62d27e1d2460f9445.jpg)  
D  
B

![](images/16d57e98e9a425536a993d47a8547b132d456167e39fc0078b8caaee4e07138b.jpg)

C  
![](images/c739bd779e6a4f7d4831de899c770fd0e7cfb801d180e9c9ea36fe8729814aca.jpg)

![](images/5a59fd9bd5a2380e22a02506b87af300458b8f6ee3fe6328982b27ac108be5ce.jpg)

E  
![](images/7b70acf3bdd8d61bf9be15494732ec259cbf0a66b1a05cfc4a7e4f73a9275e39.jpg)

![](images/028f07ddc4938a5343f6057beb7f1fec07e022a995567dcbb21fa993d5b94e63.jpg)  
FIG. S71. Key–value erasure and editing: ahead–behind relationships. Results are shown for Gemma-3-27B. Proce dures follow main-text Fig. 5. The control context is “The ‹A› is ahead of the ‹B›. The ‹B› is ahead of the ‹C›”; the donor context is “The ‹D› is ahead of the ‹E›. The ‹E› is ahead of the $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 1104 keys and 79 values retained by this mask, together comprising 3.98% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys. The wording “behind of” is a grammatical error in the prompt template; the intended wording is “behind”. It is retained here to match the prompts presented to the model.

A  
![](images/e3abc3da176b49f706d940782199193b702c1a1ad2069c7c5fc0e5d48aa53259.jpg)  
D  
B

![](images/fcd67d7c93514ca56c8906f5df7b3eb9cce771045e1d986e7693f67a241af5cf.jpg)

C  
![](images/7ee9e4d565dab344673cbcf8e6a3f0d9067268ea031fdd8417190f98b46556e2.jpg)

![](images/8bbccabc824587278804f020f95e1c69544c30bee9c2bff9cbe19cd4cbabc88a.jpg)

![](images/06703db396fcc9726df5e536a7737ea2ee536cf1a42892506cffd184d68edca8.jpg)

E  
![](images/7ed605e33b898b9b75c65f2e2fd9eccd0de5cfe3157249725b3ea143c21b7fab.jpg)  
FIG. S72. Key–value erasure and editing: ancestor–descendant relationships. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 188 keys and 30 values retained by this mask, together comprising 1.10% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/02b4a60e935063fb3ad31d69525c57871ab78ab847949eeca5b41dc7d78ce5d8.jpg)  
B  
D

![](images/1897bf2407bb55ed4bcaf554f5af46d27f646ac097c7b5daa6fd230233a3b6fc.jpg)

C  
![](images/83f13d624e4ce1601a292439923483ba9c4767b6a65d919e44f7b7f169eff90d.jpg)

![](images/676951720f8aee8c0ed6717544d40c29d04de92cff2e3fd36f6b409106f935b7.jpg)

E  
![](images/8fc52c450ec6069243aa7ccc8a57e57c89aeeace475e78658d4b2ff2577ca3bc.jpg)

![](images/5e477cdbf01cf6236577269688ca6c9c4ed74a78b5a575529fde67d0212d9248.jpg)  
FIG. S73. Key–value erasure and editing: exact age diferences. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is one year older than ‹B›. ‹B› is one year older than $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is one year older than ‹E›. ‹E› is one year older than ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 449 keys and 91 values retained by this mask, together comprising 1.70% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/b3adddfcaa501f09248edc20085a0a0996fa3f1198aeb082befad77c925c117a.jpg)  
D  
B

![](images/4a6c26996816c4bd7d03357bac00957b1baa1b3a434b9a9c0b8e01af56bb3855.jpg)

C  
![](images/26ff5036c16e262bbba9ed00aa4f7ab1fde3e74093bdbcaba990645f568b5060.jpg)  
E

![](images/efe4f3a34fb9ddf98461c363b564ef45f226ed2a3d067f5ecd4221595008330b.jpg)

![](images/63fd5d129cd8d0e3426446b7536ac4464b36691963034b854a1122fe47010d03.jpg)

![](images/00f8f449b1d3b42b6a62d6ac70ef3a0a854e2a444dabf9df6c568912233da987.jpg)  
FIG. S74. Key–value erasure and editing: gene regulatory networks. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { A } >$ activates ‹B›. ‹B› activates $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ activates $\langle \mathrm { E } \rangle$ . ‹E› activates $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 2,333 keys and 386 values retained by this mask, together comprising 11.42% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/ace47eae67b2738e4cd94d89933242f4b196878170edfd047300387d3830ba97.jpg)  
D  
B

![](images/fa00d2df4f17205a7b851beb294e41c08542dd16e3472714fef765ccfa86ef0f.jpg)

C  
![](images/ec9d369365faa10e92b934881a2a9a73ac72ff80ab56ef96fd722812a36a1b2a.jpg)

![](images/4dea73f054f2900af3594c7c05154ddf7444c404f5468e8790c31eb2a2fba556.jpg)

E  
![](images/bd72ef43a039ff08b38bb2c4647b47b202f30c13685574879582db60292853bc.jpg)

![](images/9185397ee038f7fef5ab64286117bd50b4e9a287bdf1a93e19e07df8dfff3d48.jpg)  
FIG. S75. Key–value erasure and editing: neural circuits. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ makes glutamatergic projections to ‹B›. ‹B› makes glutamatergic projections to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ makes glutamatergic projections to ‹E›. ‹E› makes glutamatergic projections to $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. $\mathbf { C } ,$ Distribution of the 292 keys and 26 values retained by this mask, together comprising 0.73% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/285affd9e4df4c580de99befedefcbf34da81f4f963ea5cffecbeebaa7c7f52b.jpg)  
D  
B

![](images/4b067527c4809a03e9aa8d80fb6560caf947bb5c9cda094c39c4f1105253513e.jpg)

C  
![](images/c4d18999827293c960e86f6be49f11cd51a18ff12bfea68f4ba619c5cdd31996.jpg)

![](images/603ae95e708e1ba0437061a88a91a31c48b3f25c40b7117889df7d2da5a57f33.jpg)

E  
![](images/fa35973a4cc40e5b3589017738d36c419793d331b377d9d47d54bf7af796f0a6.jpg)

![](images/9de75bb0f72afc14ffa46f5e8835417f77491359289c6d477ea30d770d1a7b4d.jpg)  
FIG. S76. Key–value erasure and editing: propositional equivalence. Results are shown for Gemma- $\mathrm { ^ { - 3 - 2 7 B } }$ . Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equivalent to ‹B›. ‹B› is equivalent to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D }$ › is equivalent to ‹E›. ‹E› is equivalent to $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. $\mathbf { B } ,$ Final portion of the pruning trajectory. C, Distribution of the 381 keys and 24 values retained by this mask, together comprising 1.70% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/a975b4e56629110022b014724001740e9df1018eb72a4bb8f1917b64964b5d69.jpg)  
D  
B

![](images/23fe1cf34e64a5488ac6fc0e8a3fa35a19394d5aab320a0eb9a0d910d8c4ac93.jpg)

C  
![](images/80bab404ae858c394ab8dcb0ccafae99adff4e6759f77e32bde98152e9ef7e0f.jpg)

![](images/81ca9b8010f26275ff126f558643a5e164cfa44a349bddf7d5b3ed10d96bc13b.jpg)

E  
![](images/f8df5e308fbc23bb55d8a74f82b595e0102283c0c80998750ab79437357e40f1.jpg)

![](images/d85d91100a7f5516ec7e8069531637b4042dfb3e765c7b6ab45a16b2cdd4a548.jpg)  
FIG. S77. Key–value erasure and editing: variable equality. Results are shown for Gemma-3-27B. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ is equal to ‹B›. ‹B› is equal to $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ is equal to $\langle \mathrm { E } \rangle$ ‹E› is equal to ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 275 keys and $3 2$ values retained by this mask, together comprising 1.11% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

## X. ANALOGOUS EXPERIMENTS ACROSS MODEL FAMILIES AND SCALES

## A. Identity–structure separability across models

A  
![](images/ecf70f2198520f36d3a9670e2d3f9d9a34e6c20d1e7bfaa1ad61cad48f00772d.jpg)

C  
![](images/0a5dd32a39cb8a8f199401429db7898fdd8fb756bc907c75b331c0947b05bd03.jpg)

E  
![](images/30334b039a7107c5a144dfa60416912b0f9ae8b2b8da64eaf60a91514b255463.jpg)

B  
![](images/06b3f86a6e877403dde9cfb48776eb5fe37cfc2a515845f1818652df59d525cf.jpg)

D  
![](images/31c4b4cfe33f661bb665e8bc8c5aaabe8f30593d12d9ed27aa07fb722f7f2168.jpg)

F  
![](images/d2421f6387703d8f5d476218b44eba7f94baca2966d243f5d1536b28396b5f13.jpg)

FIG. S78. Identity–structure separability: Gemma-3-4B. Results are shown for the “family-trees” setting. The control context is “‹A› has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/9848f0e0dbaf9bd7a7f8726d5e4240ab288e7b7f9420e9afb5e22dae982b38e4.jpg)

C  
![](images/bd2dff25db6d243058940c217740be19fa455f67366ed86c7430f09aa3d11791.jpg)

E  
![](images/c10e9e9bdec5143075510f14437fc207600c8b48ba4a00dbf029f10d12ea47fb.jpg)

B  
![](images/80d1ca2b597f825fd2eae2e06db22c60c05edc5e9b5b28348001c0747da4c483.jpg)

D  
![](images/f3f5a4ecc62e91b259a35dd822771e3494cebd524f14cd42fc96bb60283b534d.jpg)

F  
![](images/6cd6a35014ddb22a5826dabacf4182ce1227ee301cd25fe4e1c51079a1df9d5c.jpg)  
FIG. S79. Identity–structure separability: Gemma-3-12B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/22ea30a626521ab5d2f462b66ff0b18539666b3fd6d10330eb6738df1e2032d7.jpg)

C  
![](images/ff6613170853baf05ce8a6c647e4690d8c43b147cb9014b574ccbddc0bbe993f.jpg)  
E

B  
![](images/58c3709a30ad4b87e06d718baf638c867b57e3ca7dd9122db9e5a76ab7c46f56.jpg)  
D

![](images/b4104b77c8d34f325a1ca41996f08dd2a9b3b98afc70ae0daf4633b1f3987cfd.jpg)

![](images/5c0e8f77ac0d6304cf9b8c3c6b87f4706a81ff1ecaf36faa40ef473d98745e10.jpg)

F  
![](images/f30f3799f58cf723c990c3cb54950d25918c69f2539ce36bffe855b3aa4db3e6.jpg)

FIG. S80. Identity–structure separability: Gemma-3-27B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/40c93a4ea71897de0ed6083283b834d65a9270c5b86d19caf3e1d80602d7ad63.jpg)

C  
![](images/cc7f8804813bd37aa790ec6a4135b73eb63c96029da9eac3ded6f133cf8b7c7a.jpg)

E  
![](images/26b670e6335e83260420612da3ac577e94e16e49ed611de500054d5ba3c96e67.jpg)

B  
![](images/bc69eebf6aebca4c7e6d9bceebee8d558411d1cc7bd9120718aa7ad57a9ebac1.jpg)

D  
![](images/6414ea86df133ea55e23040d086aa5a4b97c4e51f4fd97b77a473120b047b2d6.jpg)

F  
![](images/2bca9bfa669fb7f2397f5e2512d7f2691afa5c8f34c22c9e86e407152ab49c2e.jpg)  
FIG. S81. Identity–structure separability: Llama-3.1-8B. Results are shown for the “family-trees” setting. The contro context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/e23b2b23df547b3bf300ec4859f638f3e3c972a5066fd2d9d0187543331bd303.jpg)

C  
![](images/148f27210855a47dacb6c6555ec00ea21b8405c0ac27d8b8cf300c62d4474581.jpg)  
E  
D

B  
![](images/9a4d172832191ef828a52a94919aedd0d5890d6fea60dcef53ae22fb9eef4675.jpg)

![](images/2c71972762a6ff57cd24445a209bc4ea54b14ef4912fc901f6ada4226214b1b9.jpg)

![](images/83caae9362e6114e4b6eb55174dee214070fe1214ab1565a1340c3cf53704929.jpg)

F  
![](images/5b30cd4fc329db69c5b0e5773735da416a566b31d08a4dbb4e3635a01421d73a.jpg)

FIG. S82. Identity–structure separability: Llama-3.1-70B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/b1cb47829af2ea4c54d84e69e1a14845fa1db5b1a02897dd14455ee6addbe0aa.jpg)

C  
![](images/f28824e98cd69b810a22d1133140d9a356538c5a5200638ad41e62fdc4471818.jpg)

E  
![](images/b80570083f0c668b45f99ca03f8c63126c2a0ca5e24d8d8627e787ba60417a6a.jpg)

B  
![](images/22c21f66bab7c6c9d546643300fd179dfbc3993d98f09e04d854091fa9b813f3.jpg)

D  
![](images/911a83507e275a184bd29a49a42cd060e0b7e9c726024c364098b2b1d6ffe2b7.jpg)

F  
![](images/01d0caaa61beb912e5f18bfc829e0b5783d350b93044aba30906cbceebc8f033.jpg)  
FIG. S83. Identity–structure separability: Qwen-3-0.6B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ › has daughter ‹E›. ‹E› has daughter $\therefore \mathrm { F } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/03b76b93212518c7796226b3812bb5d3beac84067884553713940c8457492788.jpg)

C  
![](images/72530498b0f905e67121e8c46275ada03c11366733fb6096486d2b5029e69e6d.jpg)  
E  
D

B  
![](images/fcab1741e1b96680db880279c9d3c0154e63d1d2f41390c62f60c801c9e5ad80.jpg)

![](images/f4b5966302c232236947b32658d6b2f0885334579e967116a0b676c5ae104bf4.jpg)

![](images/b125b5f8ffd6dbf77839e10de45f6abb2032c2dc1bbe3c76b95afcb1cd9ec430.jpg)

F  
![](images/8414a82baed5b2234084eefed13924d2be054658afb67fdc294d114cf88c17b8.jpg)

FIG. S84. Identity–structure separability: Qwen-3-1.7B. Results are shown for the “family-trees” setting. The control context is “‹A› has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/f0644850e33c5e741b66f908cd6383e38de781f4c85e056cbc3a6e7b694d16aa.jpg)

C  
![](images/fcc1942f2c7a3e8ebdb9802a2d6fcf797a3532e4d63b6f8a6955af48cdc1f947.jpg)

E  
![](images/953024ce7e3cc0be7015bdea60d1805e6224782908e422000c1716b42a70146b.jpg)

B  
![](images/e0dc80062944e9976f8152d6b160b6de096cec04300e8eca8daa52b2c7732cbf.jpg)

D  
![](images/e93e1b8d69d5220b7ed4272d73ca66721152b4a58bce3a764d9afd63d8ed1abd.jpg)

F  
![](images/02c3700161be78bbd6405bf77cb4d41eb2ea7304989394ce13515f3c4531ba0a.jpg)  
FIG. S85. Identity–structure separability: Qwen-3-4B. Results are shown for the “family-trees” setting. The contro context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/1b19169accb5af1638d3ca06659795c96c99e4867859226fd9ef01570337f6e3.jpg)

C  
![](images/04cdefbc777a00cb1200fe3b4c2d121e247a7af7574f9b0e64941f65c7056562.jpg)  
E

B  
![](images/cd54bec67c8626195c3173f42a10d0f359452c8d285793350b016c15b3b21144.jpg)  
D

![](images/f17518d9d5e722058dc0fa43ce760ec41db347bcb4f4cc0993c544c39d387459.jpg)

![](images/dfdb5c32fde4a874166104b8a7e0c1bbf438c9ef49b0154b72efcc1584d0aae6.jpg)

F  
![](images/5540c4ac19dc2b64f849aa91bc58171aae3a133c82089378c4db2915579eb13c.jpg)

FIG. S86. Identity–structure separability: Qwen-3-8B. Results are shown for the “family-trees” setting. The contro context is “‹A› has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ The mosaic was constructed as in main-text Fig. 2.  
A  
![](images/9663e37d41cd173be8324e181d1396693e547b5fd1c80408a1034c5ea663e2fa.jpg)

C  
![](images/8e2f6a8f24f5988444fe7261d1504dadd975ffd1b45245ec0b3bd2a1158d9e15.jpg)

E  
![](images/1d4e3b6fadf002456fe26b6edf480d5afe2de21cf1a08f7c807c7659b7c3ea30.jpg)

B  
![](images/77c98326376b65376f7dab713127843284f7a53116f170841dd839aa5a6dd1e0.jpg)

D  
![](images/dc98c75851408ec47f576a3bbc177ead17555523e727eea4167e5555fb72e162.jpg)

F  
![](images/ff1b01121860c52dfd4530206eb7ae94b0ff948e535bc94b91f4fc95eed041a5.jpg)  
FIG. S87. Identity–structure separability: Qwen-3-14B. Results are shown for the “family-trees” setting. The contro context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.

A  
![](images/01ac163eb1d9ec8ac28668c6b2fac2fbcdb991e315cabb2303e19ce4beb288f4.jpg)

C  
![](images/0336c2744add7873dfb38b0568f9c822bd2051eec178e8eb2d4f7b639c94c8cc.jpg)  
B

E  
![](images/636ae17e7dab1b3e0fa8c4802f8869301eeb0b192c7eb5c462cb489cb524fbd8.jpg)

D  
![](images/d49404f7d7019c6f06de106023c9e6d96da37fe29560024254e79b244cd4a65e.jpg)

![](images/cb4482fc8a897a39d2d083317c605d4fe362f876c6b2e627391e0a1bd9b6107d.jpg)

F  
![](images/73c64d20bb9c82356d16e169cb798286b75cdcabda986a90a5120f6d6f6e54f3.jpg)  
FIG. S88. Identity–structure separability: Qwen-3-32B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 2.

B. Additivity of representations of identity and structure across models

A  
![](images/bcfb9a17de3f3025fb598ae85edf31d64774abea6df3f276cc374ca8d3564320.jpg)

B  
![](images/f66700faf2027a54ed448a79578c6149a70d73b42299a35ed56771284c030fb6.jpg)

C  
![](images/8b236940459fdcbfce26a2a84e35509654984dceba9d44c8b24e6e521b7565be.jpg)

D  
![](images/dcf9748184bd7257730436c361f3af38d2633553df186ad4738d151e905ba538.jpg)

FIG. S89. Additivity: Gemma-3-4B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are “‹D› has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/655f0a386d78543e1ef9c0f556fcaa3230d554f48ffef7357a355d6e317bc5be.jpg)

B  
![](images/08f27537969d9e1c9bc53f43182b08b66769ec39aa8ec6eeda0a5bed26ad68b0.jpg)

C  
D  
![](images/9768b70646d7e0a58b2999f763bbd4f49150c240876c38d8707d428188f07ae7.jpg)

![](images/4aea25d208c33470f59e33536ec716dea383ec10dd4a732c8de3033a38003cb7.jpg)

FIG. S90. Additivity: Gemma-3-12B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts $\mathrm { a r e } \ ^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/0fedfd2bb84c293ad54c422f4f3b58dfc5871004b98f28e16776607904e1552b.jpg)

B  
![](images/7f0b7c98f0f7faa6f73a83bc42786922b08189af7d72a3faa540c609855fb9fa.jpg)

C  
![](images/42e09bfbfb5ed6b12445133ca48fabb72e80ef1153b0434ace4b4e768a65e33b.jpg)

D  
![](images/981fc0a749899df5059ec6dcc6627a1d5fbe90baba5335fca1bdc9e324abd13f.jpg)  
FIG. S91. Additivity: Gemma-3-27B. Results are shown for the ${ } ^ { \mathrm { * } } \mathrm { f a m i l y - t r e e s } ^ { \mathrm { 3 } }$ setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter $ { \langle \mathrm { B } \rangle }$ . ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. $\langle \mathrm { E } \rangle$ has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } ,$ . The mosaic was constructed as in main-text Fig. 3.

A  
![](images/34c95ff2de0290ed06463a31737b29b746cfa5a020b44433ee3c20b7ce3c8546.jpg)

B  
![](images/a2d37440d00d9fbf1189988865d16b240bf2ef517affb9c4cff0cac95db677f3.jpg)

C  
![](images/17e84400ce9316836cba7318e8a309efd047c688b5c654bb8ff09ef41696c05b.jpg)

D  
![](images/a3a5754691b30ef9e383ce66646129c5f8d44609edb086254904fa681fa2fc03.jpg)

FIG. S92. Additivity: Llama-3.1-8B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are “‹D› has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/96e2516b132812c6cfd14c9c7cbc83d6b5be532c0c7f701b12cd88c2cc105b12.jpg)

B  
![](images/2e07afeab4b8da590c6b029b2f378cd9fd754ac918b36134a2c4f96d5e36735d.jpg)

C  
D  
![](images/88840c24f7789389ae2eef135fba3325e3e1bb238956ac179c7d75d2dfbab6ed.jpg)

![](images/20130bb88f0752a781047961e4e1d347c35c29697bba2ffda457ea60a17657f2.jpg)

FIG. S93. Additivity: Llama-3.1-70B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter ‹D›”. The mosaic was constructed as in main-text ${ \mathrm { F i g . ~ } } 3 .$  
A  
![](images/02829d34a5d3b4ed5a90f7ec17b28339d1fc1fc16463d1351c8fb8d6fa083b8d.jpg)

B  
![](images/a5fbec69b6f47441b0a2e0ccf3ddff2e6e2814035660dafeff8d47ce053df223.jpg)

C  
![](images/3cc27cb629dbe4e45eb3f2df4af5563c839e24149ed3353a5607ef6d18c308bf.jpg)

D  
![](images/4823a3cbbc370679131bc99e06a0a95e092b4d8a3f773f6833c1c493270d27cc.jpg)  
FIG. S94. Additivity: Qwen-3-0.6B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.

A  
![](images/e4450cf924d8cdb5a3200f6637420cc7eb9fc0af1f26b9dc1e50f1d9d9b86fa5.jpg)

B  
![](images/5ba2111fd64794d3666c2f35017357e0bf0f615558d4e885f89a625a81b8e391.jpg)

C  
![](images/c7d5be48206a439d2b8ebbe70eb12fe02400914252886e1a10c63cf306d03847.jpg)

D  
![](images/6c06a0c37d7b022c3b0427b20e6b594c7ff2c6b453211f89dc48c38834e8856a.jpg)

FIG. S95. Additivity: Qwen-3-1.7B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter ‹C›”; the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/d16da34f43a2eaf0db4cf2b696d9e29123e406c67cdd49c3ca36dd099be70aa9.jpg)

B  
![](images/ce5fe5130e9fdfd1399f03ade4be0f6dd065c0691c4a3f14103d31f09b9ce32b.jpg)

C  
D  
![](images/723c83cabb0c6d59b941911dfd8f49832091a78ac8b9675aa80b5c097597dafc.jpg)

![](images/b108384ab5ceea0187a6433467ad0f005dd0872d3444cc986088a64d0ef449d1.jpg)

FIG. S96. Additivity: Qwen-3-4B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ ; the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { D } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/7ffffeb4815e681ffd24f32ed30c2a1fcc4a745a38d7f85769e1aa61b9534b9a.jpg)

B  
![](images/97d92cec13a97fc59ef1cd5fce13f290b127d8136f1085c192860b4ae5c9589e.jpg)

C  
![](images/7ebbb0e5771ec0e19dda1e62f746540bacfa7fec9c7dbac395eef7955232cb53.jpg)

D  
![](images/21e0a4cb6aaf768a5edefbb7b722a8fcbe786e27fd053eea1965bb408d76293b.jpg)  
FIG. S97. Additivity: Qwen-3-8B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.

A  
![](images/c34c19e6069697fb17fce3c547f673e023d89be7695982baddd0a4cc84f7ec53.jpg)

B  
![](images/a7e4fed344260f7cf5ea13cbba162a03d059a7a9d0e4e85b45a4ef5b8e2dda58.jpg)

C  
![](images/7ad20b8d8617c4d1fe66ffdd6b61a0e373d90144f6deebf5c3d4c6938797560b.jpg)

D  
![](images/dab7a27839737a204f36ca4b2e13c72e82c968a9ece031c0b028cb3baa4656e6.jpg)

FIG. S98. Additivity: Qwen-3-14B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\left. \mathrm { D } \right. ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.  
A  
![](images/bb0fb3fe45c7390a99dc5a4df81a19e80242602e6c05ccb360f2f69d0cf5736a.jpg)

B  
![](images/709165b0463721098625e866d824e5241cfb5ab9e5f443c8ab74ac4fb261ae2f.jpg)

C  
![](images/fee2527ca50fa1ecf0017e027681388d7e4eb97f0e793a48db17645d64743cb6.jpg)

D  
![](images/d937b31f656dabcb04c28764f8b70d612406493103fd5a21678957f8cefb1f95.jpg)  
FIG. S99. Additivity: Qwen-3-32B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter $ { \langle \mathrm { B } \rangle }$ . ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ ; the donor contexts are $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { , \mathfrak { s } }$ and $^ { 6 6 } \langle \mathrm { F } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { D } \rangle ^ { \mathfrak { n } } .$ The mosaic was constructed as in main-text Fig. 3.

C. Retrospective and prospective information across models

A  
![](images/052aac0287a11327db6eee5c66a53d9933703e7ae49e37a06603622aef9362fd.jpg)

B  
![](images/cca5eb1d56e942dd11d4a9db7fee64993ba584942909f1c11b5f6fb6e13ab749.jpg)

C  
![](images/b79507d75e7e0100a19685bac4f1f4b58b9092f2719cbde6fe25bb0983a3b5a9.jpg)

D  
![](images/f1bad4c6c5aa004345ec3894d531b08e55e3a082781de462d3c83d689a310fa0.jpg)

FIG. S100. Retrospective and prospective information: Gemma-3-4B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/f154cb27677722573906043d80b3ea949b68ad659ce5bd2d62d3c0c14ed58358.jpg)

B  
![](images/f426b349bdf4feba1bc455297ce6e1e27504c9b01e31c0ab07fb44a2dcd44c88.jpg)

C  
![](images/022f5511d1751a284acc1f52afd4fa528f34e38a395eefad440513eed10cb3b4.jpg)

D  
![](images/5ab442bafe89dcc37f9323f0d0d9b86a8d93803e97adc8a5f1b89453d5c82ddc.jpg)

FIG. S101. Retrospective and prospective information: Gemma-3-12B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother $\langle \mathrm { B } \rangle$ . ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/8e294d1cef3ee13dcaf979601b626c917af30698466dcb8275d1c7438e7a0a66.jpg)

B  
![](images/3bcb3306e6d59b768c1afd31a0a5fc8427320d80e28185b489ece0a04caca779.jpg)

C  
![](images/885a3d46fb7e803ea74da1454ad6a62804e48f610646e051d7eeda2c8e936ab1.jpg)

D  
![](images/ae332694acba118b4acef86cf9afb683ee83c89ae014d9e20af013eaca4e9bb3.jpg)  
FIG. S102. Retrospective and prospective information: Gemma-3-27B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.

A  
![](images/a51bc1fe528ac8cc640db7f7f86a2d49a10c9e38270a487f42bfc93f2139dd5b.jpg)

B  
![](images/487e8c12302d8f5e3807e8a9bf3f42027059ed576156ab684df87ec9b0f21556.jpg)

C  
![](images/174b5547f7da55e726614b32a1c3c520da24976e6ee0bc32db4822eee55efad0.jpg)

D  
![](images/428e656f9d4da9fcb702cd2fbb137118c2e24eff04808f48098d55d66ca9bf7d.jpg)

FIG. S103. Retrospective and prospective information: Llama-3.1-8B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter ‹C›”. The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/9a690630794d061b46496824f2b7f910305bd69ea90e6fc222b58a7bbde6fb0b.jpg)

B  
![](images/a41a511c8be2284fb588ec90b2c961d676ecd3bfee393ec28a3c7d234b366bac.jpg)

C  
![](images/79e49c96f64468332197247310299b42a729c7e27c9924352a1cdc7ce5350ecb.jpg)

D  
![](images/fc5eb52a77bf0fcd9fdd27314c62a23b4cd1bdf8413098e6425e1b14dbaa30b1.jpg)

FIG. S104. Retrospective and prospective information: Llama-3.1-70B. Results are shown for the “family-trees” setting. The control context is “‹A› has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/63344c5d8107a955e5184dab270700425eeb8f4a28851c53c31e3f4ea1a640d5.jpg)

B  
![](images/58db5f83c99c451e509aba26bd39b8fd1860e8bb99c5556ae16e418d79dab4ed.jpg)

C  
![](images/63c91099639d38962f9e754391ae7b9772306d9f546abc418c5a6cbefd751e06.jpg)

D  
![](images/235e15138cb082213352988597856cc2b8f1f36fc2eb11669560fe17859de306.jpg)  
FIG. S105. Retrospective and prospective information: Qwen-3-0.6B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter ‹C›”; the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.

A  
![](images/a7ba82f6e4e53a0aef0141c5c8f4cd2cca816463a7d6c727db3977f95d2f7f01.jpg)

B  
![](images/33ea1c070b23228be73fe3b0e6b4eef1368ae05ce5d6a060cbcfa017c7570780.jpg)

C  
![](images/fb4faadaff4d91f791487ef85e9e46f01aafcdf311c25caad6b4af25990eb356.jpg)

D  
![](images/a717cab4346581933dc85a32cc8b56e43ff6db87d735b2510f156b83a02fd5ee.jpg)

FIG. S106. Retrospective and prospective information: Qwen-3-1.7B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter ‹C›”; the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/890dae2be59cad902141d0b14fac1956c672fd40fe8206a83f90d0cbbf75573c.jpg)

B  
![](images/073e684d35b6d244c478581c67b8ee82207e9e08189acaddcc9aedf0788a3913.jpg)

C  
![](images/2c7c22a1479e53c9624f8e9b562e91b49a6c0cfa5b017fb3dab298c99618aee3.jpg)

D  
![](images/3c46325be7195c1c93274f0b7820965de3bf4d56f78abd563b562b20ede810ed.jpg)

FIG. S107. Retrospective and prospective information: Qwen-3-4B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/72fe29760552e27b18e3a6f27a8bd33d4c7ab8ae5c9e45df36571abdba958b3a.jpg)

B  
![](images/19c5713dafd68d0bd44b6809a5d31c862b66d2fc7d1eff44ad4e52d18c6c2ea2.jpg)

C  
![](images/91cc4ccf4166b48c9cc858374d1207d8a293a5bf60db1b3aaed61735ce6ec7fa.jpg)

D  
![](images/ab79f46fa33ec52da0605ae20377f302da9bcaa74220a3bdfc582f3879508262.jpg)  
FIG. S108. Retrospective and prospective information: Qwen-3-8B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { n } } .$ . The mosaic was constructed as in main-text Fig. 4.

A  
![](images/b0165843071cd1aa9d48f9fb7720bbf95d6e4e01417a8c0b96cd44288f51f126.jpg)

B  
![](images/64de9f11f0c4c0a630fb0372b85093d49e51b33e22d1dbea8013857f322c48ee.jpg)

C  
![](images/c043b3f83ab4cfa0a46ad1b9eaa1eb365adedccd1a35cf3989d6bd3a80677788.jpg)

D  
![](images/a46f3c52b120ba37d45da2d8de23e509e76044c5f84d3a7cc215abf869524836.jpg)

FIG. S109. Retrospective and prospective information: Qwen-3-14B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } } .$ The mosaic was constructed as in main-text Fig. 4.  
A  
![](images/7b041a8687ebaa5717db7424baed289f4225563fb5bb46effc004371f3d64489.jpg)

B  
![](images/687525a2e7c894f2962966b14b4fe938f62dc0f2762b9e60bd782d65bc30edd1.jpg)

C  
![](images/a0a745d4a4c723822a1a4ad16d094982cd659b5642af4fd0d720e13240fd04ab.jpg)

D  
![](images/8e006b4302500cafbc60237008f0781b865481cf5e35e1eade8bcf02af85a753.jpg)  
FIG. S110. Retrospective and prospective information: Qwen-3-32B. Results are shown for the “family-trees” setting. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \flat } ;$ the donor context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has mother ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { s } }$ . The mosaic was constructed as in main-text Fig. 4.

D. Key–value erasure and editing across models

A  
![](images/73f6c9264e01bbf8d3862ea4f611c751db46f448e847f9f55e4f55d4b1e72e01.jpg)  
D  
B

![](images/924d79c26198c001a0c988984a691ded9c20f91e43d3813869d7fed7651ff38f.jpg)

C  
![](images/219985b5eb5ed4a35929e99addf6d3f4a574aaca9fe06803d83fbcae155a351a.jpg)  
E

![](images/e1fcd6bd5c9294962a74196fc2455ff22f5c3984fbd3f2e15bbab1d47e51f499.jpg)

![](images/02519fea8fcf9dca5977cf31a2c6e9cc895b5d4f5386baa2775bacb21b8de1a8.jpg)

![](images/c7d55b36745206357b980b0155a660abb8f151489290c6a64765714dd9a95f10.jpg)  
FIG. S111. Key–value erasure and editing: Gemma-3-4B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ has daughter $\langle \mathrm { E } \rangle . \langle \mathrm { E } \rangle$ has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 47 keys and 1 value retained by this mask, together comprising 1.76% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/9e5db2b9ccb94ffc23de9f63d5e4fe4c7a8d2d36c20c3fe711b4e27f38a14329.jpg)  
B

![](images/ca6a9ab9e9bc81f88fb82d622f3509dd1eb49164562e73aa29bc1a54a886cf89.jpg)

C  
![](images/1cab736f88b9f07b77f781559822d28fccabde8c3957b649b89c5e518d26ca96.jpg)  
D

![](images/c02143f2a8098839181b8e652eb8a03a1edc7f4df8345e1dd8a6c777772926a9.jpg)

![](images/a9e312aff9fbefc330817e006c26fe7161578f792ad66873bd1f54f931399f31.jpg)

E  
![](images/6d09d3eae735180f429377e71b184ef51c35b68f3c02b653a5c561a0a162a640.jpg)  
FIG. S112. Key–value erasure and editing: Gemma-3-12B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. $\mathbf { C } ,$ Distribution of the 92 keys and 10 values retained by this mask, together comprising 1.33% of al key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/e1c36ad266bf1146f70076773b578e572a65b978b17654ca4c3e4396f345be13.jpg)  
D  
B

![](images/7bbfaf3d813c74dca115ce59c7bc7d6cfa1f922984011f38ff320247805616d5.jpg)

C  
![](images/67eccc4eb654e7f88b04d9b810781efaa6b230cc9101c30333665bbd3a6a59d7.jpg)  
E

![](images/609db81edcee9c6f693bfe1ec6ed0858a076f2e27c28d8f3c45e17556be0cdb8.jpg)

![](images/2b585718e27e1e4808d89894ed0d29b8ee61a520935cb523e2a245a3cfade343.jpg)

![](images/e57388fa91a7a8a1226898f1b8e06a57f259c2178b0f855b1460c968ebc0ce96.jpg)  
FIG. S113. Key–value erasure and editing: Gemma-3-27B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 176 keys and 20 values retained by this mask, together comprising 0.99% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/8c66ce6c9c84772a5f79ef48162f64302703d779d75e5d02b55c54d68ee4a4e6.jpg)  
B

![](images/9043d952fa51796a55e271de1fa03063ac9867deea0a72b32d102bf4320f732a.jpg)  
D

C  
![](images/e1bcce55a9d7c97f34e8be84832b81cd7a527b460a617941e33ba06364f579bb.jpg)

![](images/e371218410fca3c383a37802cd8889a4d39d7fd6ffbb0c095558316701aa5c42.jpg)

E  
![](images/4f0e1ed7a3c92d53c45535e33b9ea59fd467158d779f396aab4e6421bbcdde2b.jpg)

![](images/59943468fe2d9dbcc53492357005420322d84a493240f57fef1112b59ad94753.jpg)  
FIG. S114. Key–value erasure and editing: Llama-3.1-8B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 616 keys and 104 values retained by this mask, together comprising 14.06% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the contro memory with the corresponding donor keys.

A  
![](images/2d329edc7deea5d1f2ee0c517da6d61d26e5b7e927853097fa08b95e5f252294.jpg)  
D  
B

![](images/e477fa1e99c6174310a02ce4ba03e4b831b44bc16d4e0df615fd7164c28ac7da.jpg)

C  
![](images/bed6bad303515daf5e081dd65bb93f507c3e36583d8cdcf89ec368296cf4e33e.jpg)

![](images/af4853ddb0bd0a4e766b89908a0b539f82f27b33c833b2cd9fb77b153824bba8.jpg)

E  
![](images/9e304c3563035ed3b689796fc36f568de5feb27a53d0ca62ef3d321cd5795e21.jpg)

![](images/b8473974a146e94760bcfe8c8b307347c8dffcbccd98e96367245df968b1618c.jpg)  
FIG. S115. Key–value erasure and editing: Llama-3.1-70B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 104 keys and 9 values retained by this mask, together comprising 0.88% of al key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/7e2ac11ac49b3b0d396503ed75ac43d17c9fb8e2cef5878960be749647baa5f1.jpg)  
B  
D

![](images/5aaed665649a39ac2ae0238a257204e708f208b5fa82660ccfa1cc66e3c84efe.jpg)

C  
![](images/6df147a1b58d6ea9bb08c644b04a5956a6f7b2c62d3219ab9ca44f587a972833.jpg)

![](images/9f5d2a9e0105069a3d89437adde4116a1e49efa3b5142ddd04358ac31f7f5c30.jpg)

E  
![](images/a71070c5674118bbcfea262ecf69e5ffed69a4d82b8dd71c5b5ee3092f536df7.jpg)

![](images/88e1e3b3990d53f744e4972c6efa88d5257b25d3e3477c787262a8c8325de880.jpg)  
FIG. S116. Key–value erasure and editing: Qwen-3-0.6B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is “‹D› has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } } .$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 194 keys and 1 value retained by this mask, together comprising 4.35% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/f715cf03d76a120e4b8504f469256b197f1233c5ca257665c1463e2dfb0f2ea6.jpg)  
B  
D

![](images/d85a11dcba0663ad74a333363fe37e4e5f858256c64c9db0e87e6c609eab669d.jpg)

C  
![](images/7a8c1c6a743203bf9b32da59657d23db7167bcb9f71bc4cf79c747d5f72d0157.jpg)

![](images/ba97efe3e1ac5fdeb31d1fc08bf751aaf0dde0db8137c19da49a3710a2966c6a.jpg)

![](images/2a5851ca0e095efa7915eccb580a5252253283fc5e99312a96b097aba9da8698.jpg)

E  
![](images/251949f4ce038eaa98540fd6166c8a29c74b3cd6285e78f826ff9ffcbca131c9.jpg)  
FIG. S117. Key–value erasure and editing: Qwen-3-1.7B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 90 keys and 0 values retained by this mask, together comprising 2.01% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/f1ef3e5201f13235ae82890af0abd5d3f94a0304f79c7da2c0ba46b7110473ab.jpg)  
D  
B

![](images/c146672f65206f22c5edb12905bdb510a28184e0729b3c3b8ff99cad5972ea04.jpg)

C  
![](images/b19324346a4848cdf112b354814e5820b8bf9e349e38b87c6af8315654506858.jpg)

![](images/4ccf15825554f9b8d6a364ced64f4edc1073b18b4e8a8deba42673266c280225.jpg)

E  
![](images/6d6aba26e32e5247b39d905de7aa171c1273c6c37375789bdd19868b308e70f4.jpg)

![](images/4940326d1d9dfc9aea86be9234d2779d8723caa4ea024ef70d119a7a7b30e29e.jpg)  
FIG. S118. Key–value erasure and editing: Qwen-3-4B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 456 keys and 21 values retained by this mask, together comprising 8.28% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/e3ba27540686e1548180266b937c3d950acccf45eb635fda150cc7afaddbf8c8.jpg)  
B  
D

![](images/3289955c10eba28f4cd85c9834f5413ebb83f24bd817f748a3cbd3df05c1dafd.jpg)

C  
![](images/0ca41419e99672bd6b781fc14a177f980a840c251120d3a4cb2c21a19d50e366.jpg)

![](images/8e3d0d77bd6a671453ea9ce166f331f8a3e7cf21053f14d35c7debd7fa66f7b8.jpg)

![](images/c9098e713872b1013ebc350a3ff3a6c1c3bbf702fcde3c1ded4b5e915e6e6256.jpg)

E  
![](images/e1f76353c38ce5b1d6be62ac290433944545e77bbdcf54aaf18ca5441d6bdd46.jpg)  
FIG. S119. Key–value erasure and editing: Qwen-3-8B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ › has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter $\langle \mathrm { E } \rangle$ . ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 162 keys and 1 value retained by this mask, together comprising 2.83% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/bae1c4929fab150ac15d2c3f2432df3ecb7431c7fa759b4feeb22ef825fa2169.jpg)  
D  
B

![](images/2e460db3be3dc486b2953cde7d79e021032e1c6ad19c103b947221fe0bc911c0.jpg)

C  
![](images/4f7a1172f9301eb970d342487f9acb59aeb1adde98dc13e0724a814ddda29276.jpg)

![](images/47cca2be84bd6f858e488b8f91c862233caeb03e5f182ca84237aebd1d501799.jpg)

E  
![](images/165170bf39f3cbcbfbb9c949c35176e950504b1819e8a180201fb20fd88f348e.jpg)

![](images/856df267c46554ca3296ccdc713159d987d21ae47d8fdd6a08bdf4110fcf6373.jpg)  
FIG. S120. Key–value erasure and editing: Qwen-3-14B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context is $^ { 6 6 } \langle \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter $\langle \mathrm { F } \rangle ^ { \mathfrak { s } }$ . A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 237 keys and 15 values retained by this mask, together comprising 3.94% of al key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.

A  
![](images/d30dc18b790f05d5a62f314fa72c320ad97f6ddf0640f1eed8afe43e1a09ad54.jpg)  
D  
B

![](images/9958d85b17f650ce4f3b8bb52388791fe52efb858ef4fab5d9544d44928300d5.jpg)

C  
![](images/c6377c8850bf6d79f0bb5d9ed5c370bcca487bbc99381960b5908dfbaa493429.jpg)

![](images/11a4a0d7bea19a91bf6a2fa28763897f3246734ba9f103266bc6cc085fc6c38a.jpg)

![](images/ed591a7046e0916fc0c0b52925203cdb042846280baa02533b92e9c5624bc8dd.jpg)

E  
FIG. S121. Key–value erasure and editing: Qwen-3-32B. Results are shown for the “family-trees” setting. Procedures follow main-text Fig. 5. The control context is $^ { 6 6 } \langle \mathrm { A } \rangle$ has daughter ‹B›. ‹B› has daughter $\langle \mathrm { C } \rangle ^ { \mathfrak { N } } ;$ the donor context $\mathrm { i s } \ ^ { 6 6 } < \mathrm { D } \rangle$ has daughter ‹E›. ‹E› has daughter ‹F›”. A, Responses for the keys-first, keys-only, and values-first memories. B, Final portion of the pruning trajectory. C, Distribution of the 183 keys and 30 values retained by this mask, together comprising 2.08% of all key/value sites. D, E, Responses before and after replacing selected keys associated with the first entity in the control memory with the corresponding donor keys.  
![](images/5d1ccb6394672d434f724428b313f1f80ff3f077a1018a1cf57ccf055dc69eab.jpg)

## REFERENCES

[1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, in Advances in Neural Information Processing Systems, Vol. 30 (Curran Associates, Inc., 2017).

[2] S. J. Gershman, I. Fiete, and K. Irie, Neuron 113, 1694 (2025).

[3] Gemma Team, Gemma 3 technical report (2025), arXiv:2503.19786.

[4] A. Grattafiori, A. Dubey, A. Jauhri, et al., The Llama 3 herd of models (2024), arXiv:2407.21783.

[5] A. Yang, A. Li, B. Yang, et al., Qwen3 technical report (2025), arXiv:2505.09388.

[6] J. Su, M. Ahmed, Y. Lu, S. Pan, B. Wen, and Y. Liu, Neurocomputing 568, 127063 (2024).

[7] J. Ainslie, J. Lee-Thorp, M. de Jong, Y. Zemlyanskiy, F. Lebron, and S. Sanghai, in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (Association for Computational Linguistics, 2023) pp. 4895–4901.

[8] J. Feng and J. Steinhardt, in International Conference on Learning Representations (2024).

[9] K. A. Norman, S. M. Polyn, G. J. Detre, and J. V. Haxby, Trends in Cognitive Sciences 10, 424 (2006).

[10] L. Paninski, J. Pillow, and J. Lewi, in Computational Neuroscience: Theoretical Insights into Brain Function, Progress in Brain Research, Vol. 165, edited by P. Cisek, T. Drew, and J. F. Kalaska (Elsevier, 2007) pp. 493–507.

[11] J. Jarvik and D. Botstein, Proceedings of the National Academy of Sciences 70, 2046 (1973).

[12] C. Olsson, N. Elhage, N. Nanda, N. Joseph, N. DasSarma, T. Henighan, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, S. Johnston, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, and C. Olah, In-context learning and induction heads (2022), arXiv:2209.11895 [cs.LG].

[13] T. W. Webb, I. Sinha, and J. D. Cohen, in International Conference on Learning Representations (2021).

[14] K. R. Wang, A. Variengien, A. Conmy, B. Shlegeris, and J. Steinhardt, in The Eleventh International Conference on Learning Representations (2023).